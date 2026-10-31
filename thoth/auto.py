"""One-call instrumentation: ``thoth.init()``.

Call it once at startup. It reads the connection from the environment and
governs every tool of each supported framework installed in the process, at
the one place that framework runs tools, so no tool is edited and nothing is
passed per call::

    import thoth

    thoth.init()  # THOTH_API_KEY, THOTH_API_URL, THOTH_TENANT_ID, THOTH_AGENT_ID

    with thoth.acting_for(request.user.email):
        graph.invoke(...)

Frameworks governed (more follow):

- LangChain tools, at BaseTool.run and BaseTool.arun, which every tool's
  invoke goes through, including the tools LangGraph's ToolNode runs.
- OpenAI Agents SDK function tools, at FunctionTool.on_invoke_tool, including
  tools created before init(). A refused call returns "Thoth refused this
  call: <reason>" to the model, as the SDK reports tool errors. Local shell,
  computer and apply_patch tools don't run through it and are not governed.
- CrewAI tools, at BaseTool.run/arun (and the Tool subclass @tool creates)
  for native tool calls and CrewStructuredTool.invoke/ainvoke for the text
  tool loop. A refusal returns "Thoth refused this call: <reason>" to the
  agent. A result CrewAI serves from its tool cache doesn't run the tool and
  isn't checked again.

Every call is sent to Thoth before the tool runs, and the server decides:
whether the agent observes or enforces is set there, per agent, never here.
A refusal raises ThothPolicyViolation instead of running the tool, failures
are fail-closed unless THOTH_FAIL_OPEN is set, and no route to the original
tool remains. ``approved_scope`` is optional: without it an agent should stay
in observe until an admin confirms its tools, because in enforce mode a call
with no risk score and no approved scope is held for approval.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass, field
import functools
import importlib
import inspect
import json
import os
import threading
from types import SimpleNamespace
from typing import Any, Callable
import uuid
import weakref

import httpx

from thoth._context import _CURRENT_SESSION, claim_checked_call
from thoth.emitter import HttpEmitter
from thoth.enforcer_client import EnforcerClient, enforcer_http_clients
from thoth.exceptions import ThothPolicyViolation
from thoth.logging_config import configure_thoth_logging_from_env
from thoth.models import EnforcementMode, ThothConfig
from thoth.session import SessionContext
from thoth.step_up import StepUpClient, step_up_http_clients
from thoth.tool_inventory import publish_tool_inventory
from thoth.tracer import Tracer, _tool_args_from_call

# Sessions named by the caller (acting_for(..., session_id=...)) whose block has
# ended, kept so a later request resuming them continues their tool-call
# history. Only these are ever dropped, oldest first; a session in use never is.
_MAX_RESUMABLE = 10_000


@dataclass(eq=False)
class _Actor:
    """Who governed tools act for, and the session they run in.

    While its acting_for block runs, the actor holds its session (and so the
    session's tool-call history), so the history the enforcer sees is never
    reset mid-session. Blocks for the same user and session ID share it.
    """

    user_id: str
    session_id: str
    named: bool = False
    tracer: Tracer | None = None
    runtime: _Runtime | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.user_id, self.session_id)


_ACTOR: contextvars.ContextVar[_Actor | None] = contextvars.ContextVar("thoth_actor", default=None)


@dataclass
class _Runtime:
    config: ThothConfig
    emitter: HttpEmitter
    enforcer_http: tuple[httpx.Client, httpx.AsyncClient]
    step_up_http: tuple[httpx.Client, httpx.AsyncClient]
    default_actor: _Actor
    reported: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Sessions some running block holds; an entry lives exactly as long as one does.
    live: weakref.WeakValueDictionary[tuple[str, str], Tracer] = field(default_factory=weakref.WeakValueDictionary)
    resumable: OrderedDict[tuple[str, str], Tracer] = field(default_factory=OrderedDict)

    def tracer(self) -> Tracer:
        actor = _ACTOR.get() or self.default_actor
        with self.lock:
            # Found on the actor's first governed call, and again only after
            # init() is called anew.
            if actor.tracer is None or actor.runtime is not self:
                tracer = self.live.get(actor.key) or self.resumable.pop(actor.key, None)
                if tracer is None:
                    config = self.config.model_copy(update={"user_id": actor.user_id})
                    tracer = Tracer(
                        config=config,
                        session=SessionContext(config, session_id=actor.session_id),
                        emitter=self.emitter,
                        enforcer=EnforcerClient(config, http=self.enforcer_http[0], async_http=self.enforcer_http[1]),
                        step_up=StepUpClient(config, http=self.step_up_http[0], async_http=self.step_up_http[1]),
                    )
                self.live[actor.key] = tracer
                actor.tracer, actor.runtime = tracer, self
            return actor.tracer

    def release(self, actor: _Actor) -> None:
        """At the end of a block: keep a named session so a later request can resume it."""
        if not actor.named or actor.tracer is None or actor.runtime is not self:
            return
        with self.lock:
            self.resumable[actor.key] = actor.tracer
            self.resumable.move_to_end(actor.key)
            while len(self.resumable) > _MAX_RESUMABLE:
                self.resumable.popitem(last=False)

    def report(self, tool: Any) -> None:
        """Send a tool's name and schema to the inventory the first time it is seen."""
        name = getattr(tool, "name", None)
        with self.lock:
            if not isinstance(name, str) or not name or name in self.reported:
                return
            self.reported.add(name)
        publish_tool_inventory(
            SimpleNamespace(tools=[tool]),
            tenant_id=self.config.tenant_id,
            agent_id=self.config.agent_id,
            api_url=self.config.resolved_api_url,
            api_key=self.config.api_key,
        )


_RUNTIME: _Runtime | None = None
_PATCHES: list[Callable[[], None]] = []
_INIT_LOCK = threading.Lock()


def _setting(value: str | None, variable: str) -> str:
    resolved = (value or os.getenv(variable) or "").strip()
    if not resolved:
        raise ValueError(f"thoth.init() needs {variable} (or pass it as an argument)")
    return resolved


def init(
    *,
    agent_id: str | None = None,
    tenant_id: str | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    user_id: str = "system",
    approved_scope: list[str] | None = None,
    environment: str | None = None,
    frameworks: list[str] | None = None,
) -> list[str]:
    """Govern every tool of the supported frameworks installed in this process.

    Every argument falls back to its environment variable: THOTH_AGENT_ID,
    THOTH_TENANT_ID, THOTH_API_URL, THOTH_API_KEY, THOTH_ENVIRONMENT.
    ``user_id`` is who the agent acts for when no ``acting_for`` block is
    active. ``frameworks`` limits which hooks are installed (default: all
    that are importable). Returns the frameworks now governed. Calling it
    again replaces the earlier configuration.
    """
    global _RUNTIME
    configure_thoth_logging_from_env()
    config = ThothConfig(
        agent_id=_setting(agent_id, "THOTH_AGENT_ID"),
        tenant_id=_setting(tenant_id, "THOTH_TENANT_ID"),
        api_url=_setting(api_url, "THOTH_API_URL"),
        api_key=_setting(api_key, "THOTH_API_KEY"),
        approved_scope=list(approved_scope or []),
        user_id=user_id,
        # The server applies the agent's observe or enforce mode; the SDK
        # always asks it and honours what it returns.
        enforcement=EnforcementMode.BLOCK,
        environment=(environment or os.getenv("THOTH_ENVIRONMENT") or "prod").strip().lower() or "prod",
        event_ingest_token=os.getenv("THOTH_EVENT_INGEST_TOKEN"),
    )
    with _INIT_LOCK:
        shutdown()
        _RUNTIME = _Runtime(
            config=config,
            emitter=HttpEmitter(
                api_url=config.resolved_api_url,
                api_key=config.api_key or "",
                event_ingest_token=config.resolved_event_ingest_token,
            ),
            enforcer_http=enforcer_http_clients(config),
            step_up_http=step_up_http_clients(config),
            default_actor=_Actor(user_id=user_id, session_id=str(uuid.uuid4())),
        )
        governed = []
        for name, install in _HOOKS.items():
            if frameworks is not None and name not in frameworks:
                continue
            undo = install()
            if undo is not None:
                _PATCHES.append(undo)
                governed.append(name)
        return governed


def shutdown() -> None:
    """Remove every hook init() installed. Tools run ungoverned afterwards."""
    global _RUNTIME
    while _PATCHES:
        _PATCHES.pop()()
    if _RUNTIME is not None:
        # Sends what is queued, then stops its sender thread.
        _RUNTIME.emitter.close()
    _RUNTIME = None


@contextmanager
def acting_for(user_id: str, *, session_id: str | None = None) -> Iterator[str]:
    """Run governed tools on behalf of user_id, in one session.

    Use one block per request or conversation; a session ID is generated when
    none is given, and yielded. Decisions, holds and the session's history are
    recorded against this user and session. Pass the conversation's
    session_id to continue it in a later request: its tool-call history
    carries over (for the 10,000 most recent sessions that have ended).
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("acting_for() needs the user the agent acts for")
    actor = _Actor(user_id=user_id.strip(), session_id=session_id or str(uuid.uuid4()), named=session_id is not None)
    token = _ACTOR.set(actor)
    try:
        yield actor.session_id
    finally:
        _ACTOR.reset(token)
        runtime = _runtime()
        if runtime is not None:
            runtime.release(actor)


def _runtime() -> _Runtime | None:
    return _RUNTIME


# --- LangChain (and LangGraph, which runs LangChain tools) -------------------


def _install_langchain() -> Callable[[], None] | None:
    try:
        BaseTool = importlib.import_module("langchain_core.tools").BaseTool
    except ImportError:
        return None
    original_run = BaseTool.run
    original_arun = BaseTool.arun

    @functools.wraps(original_run)
    def run(self: Any, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
        runtime = _runtime()
        name = getattr(self, "name", "")
        if runtime is None or not isinstance(name, str) or not name or claim_checked_call(name, _tool_args_from_call((tool_input,), {})):
            return original_run(self, tool_input, *args, **kwargs)
        runtime.report(self)
        tracer = runtime.tracer()
        token = _CURRENT_SESSION.set(tracer._session)
        try:
            return tracer.wrap_tool(name, lambda checked: original_run(self, checked, *args, **kwargs))(tool_input)
        finally:
            _CURRENT_SESSION.reset(token)

    @functools.wraps(original_arun)
    async def arun(self: Any, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
        runtime = _runtime()
        name = getattr(self, "name", "")
        if runtime is None or not isinstance(name, str) or not name or claim_checked_call(name, _tool_args_from_call((tool_input,), {})):
            return await original_arun(self, tool_input, *args, **kwargs)
        runtime.report(self)
        tracer = runtime.tracer()

        async def call(checked: Any) -> Any:
            return await original_arun(self, checked, *args, **kwargs)

        token = _CURRENT_SESSION.set(tracer._session)
        try:
            return await tracer.wrap_tool(name, call)(tool_input)
        finally:
            _CURRENT_SESSION.reset(token)

    BaseTool.run = run  # type: ignore[method-assign]
    BaseTool.arun = arun  # type: ignore[method-assign]

    def undo() -> None:
        BaseTool.run = original_run  # type: ignore[method-assign]
        BaseTool.arun = original_arun  # type: ignore[method-assign]

    return undo


# --- OpenAI Agents SDK (openai-agents) ----------------------------------------

_THOTH_GOVERNED = "__thoth_governed__"
_MISSING = object()


def _refused(reason: str) -> str:
    # As the SDK reports a tool's own errors: the model is told and the run
    # continues. The tool did not run, and Thoth recorded the refusal.
    return f"Thoth refused this call: {reason}"


def _govern_openai_agents_tool(tool: Any, raw: Callable[..., Any]) -> Callable[..., Any]:
    """The governed form of one FunctionTool's on_invoke_tool(ctx, input_json)."""

    async def on_invoke_tool(ctx: Any, input_json: str) -> Any:
        runtime = _runtime()
        name = getattr(tool, "name", "")
        try:
            parsed = json.loads(input_json) if input_json else {}
        except (TypeError, ValueError):
            parsed = None
        if runtime is None or not isinstance(name, str) or not name:
            return await raw(ctx, input_json)
        if not isinstance(parsed, dict):
            # Nothing Thoth can check, so the tool doesn't run.
            return _refused("the arguments aren't a JSON object, so they can't be checked")
        if claim_checked_call(name, parsed):
            return await raw(ctx, input_json)
        runtime.report(tool)
        tracer = runtime.tracer()

        async def call(checked: Any) -> Any:
            # Exactly what Thoth checked, never the raw input.
            return await raw(ctx, json.dumps(checked))

        token = _CURRENT_SESSION.set(tracer._session)
        try:
            return await tracer.wrap_tool(name, call)(parsed)
        except ThothPolicyViolation as exc:
            return _refused(exc.reason)
        finally:
            _CURRENT_SESSION.reset(token)

    setattr(on_invoke_tool, _THOTH_GOVERNED, True)
    return on_invoke_tool


def _install_openai_agents() -> Callable[[], None] | None:
    """Governs every FunctionTool, including tools created before init().

    The SDK runs a function tool through its ``on_invoke_tool`` field. A data
    descriptor on the class takes precedence over each instance's field, so
    every tool, whenever it was created, hands the SDK a governed callable.
    Hosted tools run at OpenAI. Local shell, computer and apply_patch tools
    don't run through on_invoke_tool and are not governed by this hook.
    """
    try:
        FunctionTool = importlib.import_module("agents.tool").FunctionTool
    except ImportError:
        return None
    if isinstance(FunctionTool.__dict__.get("on_invoke_tool"), property):
        return None
    previous = FunctionTool.__dict__.get("on_invoke_tool", _MISSING)
    cache_key = "_thoth_governed_on_invoke_tool"

    def get(self: Any) -> Any:
        # Each tool's own function stays in its instance field; the property,
        # a data descriptor, is found first while it is installed.
        raw = self.__dict__.get("on_invoke_tool")
        if raw is None or getattr(raw, _THOTH_GOVERNED, False):
            # A copy made from a governed tool already holds the governed form.
            return raw
        # The same governed callable on every read: the SDK checks that a
        # tool's on_invoke_tool doesn't change during approval evaluation.
        cached = self.__dict__.get(cache_key)
        if cached is not None and cached[0] is raw:
            return cached[1]
        governed = _govern_openai_agents_tool(self, raw)
        self.__dict__[cache_key] = (raw, governed)
        return governed

    def set_(self: Any, value: Any) -> None:
        self.__dict__["on_invoke_tool"] = value

    FunctionTool.on_invoke_tool = property(get, set_)

    def undo() -> None:
        # The instance fields show through again.
        if previous is _MISSING:
            del FunctionTool.on_invoke_tool
        else:
            FunctionTool.on_invoke_tool = previous

    return undo


# --- CrewAI --------------------------------------------------------------------


def _govern_crewai_method(original: Callable[..., Any], *, invoke: bool) -> Callable[..., Any]:
    """A governed form of a CrewAI tool method.

    ``invoke=True`` is CrewStructuredTool.invoke/ainvoke(input, config, **kwargs),
    checked by ``input``; otherwise BaseTool.run/arun(*args, **kwargs), checked
    by the call's arguments. A refusal returns "Thoth refused this call:
    <reason>" to the agent, as CrewAI reports tool errors, and the tool doesn't
    run.
    """

    def split(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
        """invoke's input (positional or ``input=``), and the rest of the call."""
        if args:
            return args[0], args[1:], kwargs
        return kwargs.get("input"), (), {key: value for key, value in kwargs.items() if key != "input"}

    def prepare(self: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, str] | None:
        runtime = _runtime()
        name = getattr(self, "name", "")
        if runtime is None or not isinstance(name, str) or not name:
            return None
        checked_args = _tool_args_from_call((split(args, kwargs)[0],), {}) if invoke else _tool_args_from_call(args, kwargs)
        if claim_checked_call(name, checked_args):
            return None
        runtime.report(self)
        return runtime.tracer(), name

    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def governed_async(self: Any, *args: Any, **kwargs: Any) -> Any:
            prepared = prepare(self, args, kwargs)
            if prepared is None:
                return await original(self, *args, **kwargs)
            tracer, name = prepared
            tool_input, rest, rest_kwargs = split(args, kwargs)

            async def call(*checked: Any, **checked_kwargs: Any) -> Any:
                # Exactly what Thoth checked.
                if invoke:
                    return await original(self, *checked, *rest, **rest_kwargs)
                return await original(self, *checked, **checked_kwargs)

            token = _CURRENT_SESSION.set(tracer._session)
            try:
                if invoke:
                    return await tracer.wrap_tool(name, call)(tool_input)
                return await tracer.wrap_tool(name, call)(*args, **kwargs)
            except ThothPolicyViolation as exc:
                return _refused(exc.reason)
            finally:
                _CURRENT_SESSION.reset(token)

        return governed_async

    @functools.wraps(original)
    def governed(self: Any, *args: Any, **kwargs: Any) -> Any:
        prepared = prepare(self, args, kwargs)
        if prepared is None:
            return original(self, *args, **kwargs)
        tracer, name = prepared
        tool_input, rest, rest_kwargs = split(args, kwargs)

        def call(*checked: Any, **checked_kwargs: Any) -> Any:
            # Exactly what Thoth checked.
            if invoke:
                return original(self, *checked, *rest, **rest_kwargs)
            return original(self, *checked, **checked_kwargs)

        token = _CURRENT_SESSION.set(tracer._session)
        try:
            if invoke:
                return tracer.wrap_tool(name, call)(tool_input)
            return tracer.wrap_tool(name, call)(*args, **kwargs)
        except ThothPolicyViolation as exc:
            return _refused(exc.reason)
        finally:
            _CURRENT_SESSION.reset(token)

    return governed


def _install_crewai() -> Callable[[], None] | None:
    """Governs CrewAI tools where agents run them: BaseTool.run/arun (and the
    Tool subclass @tool creates, which overrides them) for native tool calls,
    and CrewStructuredTool.invoke/ainvoke for the text-based tool loop."""
    try:
        base = importlib.import_module("crewai.tools.base_tool")
        structured = importlib.import_module("crewai.tools.structured_tool")
    except ImportError:
        return None
    targets = [
        (base.BaseTool, "run", False),
        (base.BaseTool, "arun", False),
        (base.Tool, "run", False),
        (base.Tool, "arun", False),
        (structured.CrewStructuredTool, "invoke", True),
        (structured.CrewStructuredTool, "ainvoke", True),
    ]
    originals = []
    for cls, attr, invoke in targets:
        original = cls.__dict__.get(attr)
        if original is None:
            continue
        originals.append((cls, attr, original))
        setattr(cls, attr, _govern_crewai_method(original, invoke=invoke))

    def undo() -> None:
        for cls, attr, original in originals:
            setattr(cls, attr, original)

    return undo


_HOOKS: dict[str, Callable[[], Callable[[], None] | None]] = {
    "langchain": _install_langchain,
    "openai_agents": _install_openai_agents,
    "crewai": _install_crewai,
}
