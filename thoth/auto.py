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
- Pydantic AI tool calls, at ToolManager._raw_execute, with the final
  validated arguments after Pydantic AI's own hooks, for function tools and
  MCP toolsets. A refusal returns "Thoth refused this call: <reason>" as the
  tool result. Output functions in every output mode (tool, TextOutput,
  native, prompted) are checked at _output.execute_output_function, which
  all of them call, and a refusal raises ThothPolicyViolation.
- Google ADK tools, at run_async, which every ADK tool path ends in (the
  normal caller, the tool thread pool and live streaming), after the
  before-tool callbacks. It is wrapped on BaseTool, on every subclass that
  defines its own, and on subclasses defined after init(). A refusal returns
  "Thoth refused this call: <reason>" as the tool result.
- Microsoft Agent Framework tools, at FunctionTool._invoke_function, which
  every tool path ends in (the function-invocation loop, agents used as tools
  and MCP tools), with the validated arguments after function middleware and
  approval. A refusal returns "Thoth refused this call: <reason>" as the tool
  result.
- Strands Agents tools, at the tool executor's _run_tool, which every tool
  call reaches (the event loop, direct agent.tool calls, the bidirectional
  loop and background tools) after hooks and tool middleware. A refusal ends
  the call with an error tool result, "Thoth refused this call: <reason>".
- AG2 (1.x) function tools, at FunctionTool._execute, which every function
  tool an Agent runs (@tool, Toolkit and subagent tools) reaches after the
  tool's and the agent's tool middleware. A refusal returns an error tool
  result, "Thoth refused this call: <reason>".
- Claude Agent SDK tool calls, for every query() and ClaudeSDKClient
  connection, with a PreToolUse hook that checks each call, including tools
  in allowed_tools, before the CLI runs it. A refusal denies the call with
  "Thoth refused this call: <reason>". The caller's options are not changed.
- Anthropic client tools, at call(input), which the tool runners
  (client.beta.messages.tool_runner and the session tool runner) use for
  every tool: @beta_tool and @beta_async_tool functions, the MCP helper's
  tools and built-in tools such as the memory tool. A refusal raises
  ToolError, which the runner sends to Claude as an error tool result,
  "Thoth refused this call: <reason>".
- MCP tool calls made with ClientSession.call_tool, directly or through a
  framework's MCP adapter. A call a framework hook already checked (the same
  arguments, under the MCP tool's name or a prefixed form of it) isn't checked
  twice. A refusal returns an error CallToolResult and nothing is sent.

Every call is sent to Thoth before the tool runs, and the server decides:
whether the agent observes or enforces is set there, per agent, never here.
A refusal raises ThothPolicyViolation instead of running the tool, failures
are fail-closed unless THOTH_FAIL_OPEN is set, and no route to the original
tool remains. ``approved_scope`` is optional: without it an agent should stay
in observe until an admin confirms its tools, because in enforce mode a call
with no risk score and no approved scope is held for approval.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
import contextvars
import copy
import dataclasses
from dataclasses import dataclass, field
import functools
import hashlib
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

from thoth._context import _CURRENT_SESSION, claim_checked_call, claim_checked_mcp_call
from thoth.emitter import HttpEmitter
from thoth.enforcer_client import EnforcerClient, enforcer_http_clients
from thoth.exceptions import ThothPolicyViolation
from thoth.logging_config import configure_thoth_logging_from_env
from thoth.models import DEFAULT_API_URL, EnforcementMode, ThothConfig
from thoth.session import SessionContext
from thoth.step_up import StepUpClient, step_up_http_clients
from thoth.tool_inventory import publish_tool_inventory
from thoth.tracer import Tracer, _tool_args_from_call

# Sessions named by the caller (acting_for(..., session_id=...)) whose block has
# ended, kept so a later request resuming them continues their tool-call
# history. Only these are ever dropped, oldest first; a session in use never is.
_MAX_RESUMABLE = 10_000


def named_session_id(tenant_id: str, agent_id: str, user_id: str, session_id: str) -> str:
    """The session ID Thoth sees for a caller-named session.

    The caller's ID is only unique within one user's sessions, so two users
    naming the same ID must never share a session in Thoth: the ID sent is
    derived from the tenant, agent, user and the caller's ID. It is stable
    across processes, so a later request resuming the session continues it,
    and the TypeScript SDK derives the same ID.
    """
    digest = hashlib.sha256("\x00".join(["thoth-named-session-v1", tenant_id, agent_id, user_id, session_id]).encode()).hexdigest()
    return f"{digest[0:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


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
                    wire_session_id = named_session_id(config.tenant_id, config.agent_id, actor.user_id, actor.session_id) if actor.named else actor.session_id
                    tracer = Tracer(
                        config=config,
                        session=SessionContext(config, session_id=wire_session_id),
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
        api_url=(api_url or os.getenv("THOTH_API_URL") or "").strip() or DEFAULT_API_URL,
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


# --- Pydantic AI -------------------------------------------------------------------


def _pydantic_ai_call(validated: Any) -> tuple[str, Any, Any]:
    """A validated Pydantic AI call's tool name, arguments and definition."""
    name = getattr(getattr(validated, "call", None), "tool_name", "")
    tool_def = getattr(getattr(validated, "tool", None), "tool_def", None)
    return name if isinstance(name, str) else "", getattr(validated, "validated_args", None), tool_def


async def _govern_pydantic_ai_call(validated: Any, run: Callable[[Any], Any]) -> Any:
    """Check a validated Pydantic AI call with Thoth, then run it with exactly
    the arguments checked. Raises ThothPolicyViolation when refused, or when
    the call can't be checked (no name, or arguments that aren't a dict)."""
    runtime = _runtime()
    assert runtime is not None
    name, tool_args, tool_def = _pydantic_ai_call(validated)
    if not name or not isinstance(tool_args, dict):
        raise ThothPolicyViolation(name or "unknown", "the call can't be checked, so it was not run")
    if claim_checked_call(name, tool_args):
        return await run(tool_args)
    runtime.report(SimpleNamespace(name=name, description=getattr(tool_def, "description", "") or ""))
    tracer = runtime.tracer()
    token = _CURRENT_SESSION.set(tracer._session)
    try:
        return await tracer.wrap_tool(name, run)(tool_args)
    finally:
        _CURRENT_SESSION.reset(token)


def _install_pydantic_ai() -> Callable[[], None] | None:
    """Governs Pydantic AI tool calls, and output functions.

    Tool calls are checked at ToolManager._raw_execute, where Pydantic AI hands
    a call's final, validated arguments to ``toolset.call_tool`` after its own
    capability hooks, for function tools and MCP toolsets alike. A refusal is
    returned to the model as the tool result.

    Output functions (``output_type=some_function``, ``TextOutput(fn)``,
    native and prompted outputs) don't run through _raw_execute. Every output
    mode calls them through one function, ``_output.execute_output_function``,
    so that is where they are checked, by the function's name and arguments.
    A refusal raises ThothPolicyViolation: text in its place would become the
    agent's typed final answer. Plain schema outputs call no function of yours.
    """
    try:
        ToolManager = importlib.import_module("pydantic_ai.tool_manager").ToolManager
        output_module = importlib.import_module("pydantic_ai._output")
    except ImportError:
        return None
    original_raw = ToolManager.__dict__.get("_raw_execute")
    original_output = getattr(output_module, "execute_output_function", None)
    if original_raw is None or original_output is None:
        return None

    @functools.wraps(original_raw)
    async def raw_execute(self: Any, validated: Any, *args: Any, **kwargs: Any) -> Any:
        if _runtime() is None:
            return await original_raw(self, validated, *args, **kwargs)

        async def run(checked: Any) -> Any:
            # Exactly the arguments Thoth checked.
            return await original_raw(self, dataclasses.replace(validated, validated_args=checked), *args, **kwargs)

        try:
            return await _govern_pydantic_ai_call(validated, run)
        except ThothPolicyViolation as exc:
            # The tool did not run; the model gets the refusal as the result.
            return _refused(exc.reason)

    @functools.wraps(original_output)
    async def execute_output_function(function_schema: Any, *, args: Any, **kwargs: Any) -> Any:
        if _runtime() is None:
            return await original_output(function_schema, args=args, **kwargs)
        function = getattr(function_schema, "function", None)
        name = getattr(function, "__name__", "") or "output_function"
        call = SimpleNamespace(
            call=SimpleNamespace(tool_name=name),
            tool=SimpleNamespace(tool_def=SimpleNamespace(description=getattr(function, "__doc__", "") or "")),
            validated_args=args,
        )

        async def run(checked: Any) -> Any:
            # Exactly the arguments Thoth checked.
            return await original_output(function_schema, args=checked, **kwargs)

        return await _govern_pydantic_ai_call(call, run)

    ToolManager._raw_execute = raw_execute
    output_module.execute_output_function = execute_output_function

    def undo() -> None:
        ToolManager._raw_execute = original_raw
        output_module.execute_output_function = original_output

    return undo


# --- Streaming tools -------------------------------------------------------------


async def _stream_checked(
    tracer: Tracer,
    name: str,
    args: dict[str, Any],
    stream: Callable[[Any], Any],
    result_of: Callable[[Any, Any], Any],
) -> Any:
    """Run a streaming tool inside its Thoth check and hand its events on as they come.

    ``stream(checked)`` starts the tool with exactly the arguments Thoth
    checked. The whole stream runs inside the checked call, in a task that
    copies the session, so Thoth records the tool's real outcome after it ends
    (``result_of(event, last)`` picks the result from the events). The tool and
    the consumer move in lockstep: the tool waits until each event has been
    handed on and the next one asked for, so events stream as they would
    without Thoth. A refusal raises ThothPolicyViolation before any event; a
    consumer that stops early stops the tool too.
    """
    events: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)

    async def run(checked: Any) -> Any:
        result = None
        async for event in stream(checked):
            await events.put(event)
            await events.join()
            result = result_of(event, result)
        return result

    # The task copies the context now, so the session covers the whole run.
    token = _CURRENT_SESSION.set(tracer._session)
    try:
        task = asyncio.ensure_future(tracer.wrap_tool(name, run)(args))
    finally:
        _CURRENT_SESSION.reset(token)
    next_event: asyncio.Future[Any] | None = None
    try:
        while True:
            next_event = asyncio.ensure_future(events.get())
            done, _ = await asyncio.wait({next_event, task}, return_when=asyncio.FIRST_COMPLETED)
            if next_event not in done:
                break
            yield next_event.result()
            events.task_done()
        while not events.empty():
            yield events.get_nowait()
            events.task_done()
        task.result()
    finally:
        if next_event is not None and not next_event.done():
            next_event.cancel()
        if not task.done():
            task.cancel()


# --- Google ADK ------------------------------------------------------------------


def _govern_adk_run_async(original: Callable[..., Any]) -> Callable[..., Any]:
    """The governed form of an ADK tool's run_async(*, args, tool_context)."""

    @functools.wraps(original)
    async def run_async(self: Any, *call_args: Any, args: Any = None, tool_context: Any = None, **kwargs: Any) -> Any:
        runtime = _runtime()
        if runtime is None:
            return await original(self, *call_args, args=args, tool_context=tool_context, **kwargs)
        name = getattr(self, "name", "")
        if not isinstance(name, str) or not name or not isinstance(args, dict):
            # Nothing Thoth can check, so the tool doesn't run.
            return _refused("the call can't be checked, so it was not run")
        if claim_checked_call(name, args):
            # Already checked: a subclass calling super().run_async with the same call.
            return await original(self, *call_args, args=args, tool_context=tool_context, **kwargs)
        runtime.report(self)
        tracer = runtime.tracer()
        if inspect.isasyncgenfunction(getattr(self, "func", None)):
            # A streaming tool (ADK's own test, in the live caller): run_async hands
            # back the stream, so the governed stream is handed back at once and the
            # tool runs inside the check behind it.
            async def stream(checked: Any) -> Any:
                res = await original(self, *call_args, args=checked, tool_context=tool_context, **kwargs)
                if inspect.isasyncgen(res):
                    async for item in res:
                        yield item
                else:
                    yield res

            async def governed() -> Any:
                try:
                    async for item in _stream_checked(tracer, name, args, stream, lambda item, _last: item):
                        yield item
                except ThothPolicyViolation as exc:
                    yield _refused(exc.reason)

            return governed()

        async def run(checked: Any) -> Any:
            # Exactly the arguments Thoth checked; the tool context passes through.
            return await original(self, *call_args, args=checked, tool_context=tool_context, **kwargs)

        token = _CURRENT_SESSION.set(tracer._session)
        try:
            return await tracer.wrap_tool(name, run)(args)
        except ThothPolicyViolation as exc:
            return _refused(exc.reason)
        finally:
            _CURRENT_SESSION.reset(token)

    setattr(run_async, _THOTH_GOVERNED, True)
    return run_async


def _install_google_adk() -> Callable[[], None] | None:
    """Governs every Google ADK tool at run_async.

    Every ADK tool path (the normal caller, the tool thread pool and live
    streaming) ends in ``tool.run_async(args=..., tool_context=...)``, after
    the before-tool callbacks, so the arguments checked are the ones run.
    Tool classes override run_async, so it is wrapped on BaseTool, on every
    subclass that defines its own, and on subclasses defined after init().
    """
    try:
        BaseTool = importlib.import_module("google.adk.tools.base_tool").BaseTool
    except ImportError:
        return None
    wrapped: list[tuple[type, Any]] = []

    def wrap(cls: type) -> None:
        original = cls.__dict__.get("run_async")
        if original is None or getattr(original, _THOTH_GOVERNED, False):
            return
        wrapped.append((cls, original))
        cls.run_async = _govern_adk_run_async(original)  # type: ignore[attr-defined]

    def walk(cls: type) -> None:
        wrap(cls)
        for subclass in cls.__subclasses__():
            walk(subclass)

    walk(BaseTool)
    previous_hook = BaseTool.__dict__.get("__init_subclass__")

    @classmethod  # type: ignore[misc]
    def init_subclass(cls: type, **kwargs: Any) -> None:
        if previous_hook is not None:
            previous_hook.__func__(cls, **kwargs)
        else:
            super(BaseTool, cls).__init_subclass__(**kwargs)
        wrap(cls)

    BaseTool.__init_subclass__ = init_subclass

    def undo() -> None:
        for cls, original in wrapped:
            cls.run_async = original  # type: ignore[attr-defined]
        if previous_hook is None:
            del BaseTool.__init_subclass__
        else:
            BaseTool.__init_subclass__ = previous_hook

    return undo


# --- Microsoft Agent Framework ---------------------------------------------------


def _govern_agent_framework_invoke(original: Callable[..., Any]) -> Callable[..., Any]:
    """The governed form of FunctionTool._invoke_function(call_kwargs)."""

    @functools.wraps(original)
    async def _invoke_function(self: Any, call_kwargs: Any) -> Any:
        runtime = _runtime()
        if runtime is None:
            return await original(self, call_kwargs)
        name = getattr(self, "name", "")
        if not isinstance(name, str) or not name or not isinstance(call_kwargs, dict):
            # Nothing Thoth can check, so the tool doesn't run.
            return _refused("the call can't be checked, so it was not run")
        # The framework's invocation context is runtime state, not a tool argument.
        context_name = getattr(self, "_context_parameter_name", None)
        passthrough = {key: value for key, value in call_kwargs.items() if key == context_name}
        args = {key: value for key, value in call_kwargs.items() if key != context_name}
        if claim_checked_call(name, args):
            return await original(self, call_kwargs)
        runtime.report(self)
        tracer = runtime.tracer()

        async def run(checked: Any) -> Any:
            # Exactly the arguments Thoth checked, plus the invocation context.
            return await original(self, {**checked, **passthrough})

        token = _CURRENT_SESSION.set(tracer._session)
        try:
            return await tracer.wrap_tool(name, run)(args)
        except ThothPolicyViolation as exc:
            return _refused(exc.reason)
        finally:
            _CURRENT_SESSION.reset(token)

    setattr(_invoke_function, _THOTH_GOVERNED, True)
    return _invoke_function


def _install_agent_framework() -> Callable[[], None] | None:
    """Governs Microsoft Agent Framework tools at FunctionTool._invoke_function.

    Every tool path (the chat client's function-invocation loop, agents used as
    tools, and MCP tools, which the framework builds as FunctionTools) runs
    through FunctionTool.invoke, which calls _invoke_function with the
    validated arguments after function middleware and the approval and
    argument-change checks, so the arguments checked are the ones run.
    """
    try:
        FunctionTool = importlib.import_module("agent_framework._tools").FunctionTool
    except ImportError:
        return None
    original = FunctionTool.__dict__.get("_invoke_function")
    if original is None or getattr(original, _THOTH_GOVERNED, False):
        return None
    FunctionTool._invoke_function = _govern_agent_framework_invoke(original)

    def undo() -> None:
        FunctionTool._invoke_function = original

    return undo


# --- Strands Agents --------------------------------------------------------------


def _govern_strands_run_tool(original: Callable[..., Any]) -> Callable[..., Any]:
    """The governed form of the Strands executor's _run_tool(ctx, extra_kwargs, tool_context)."""

    @functools.wraps(original)
    async def _run_tool(ctx: Any, *call_args: Any, **kwargs: Any) -> Any:
        runtime = _runtime()
        tool_use = getattr(ctx, "tool_use", None)
        if runtime is None or getattr(ctx, "tool", None) is None:
            # Ungoverned, or an unknown tool: nothing runs, Strands reports it.
            async for event in original(ctx, *call_args, **kwargs):
                yield event
            return
        name = tool_use.get("name") if isinstance(tool_use, dict) else None
        args = tool_use.get("input") if isinstance(tool_use, dict) else None
        if not isinstance(name, str) or not name or not isinstance(args, dict):
            # Nothing Thoth can check, so the tool doesn't run.
            yield _strands_refusal(tool_use, "the call can't be checked, so it was not run")
            return
        if claim_checked_call(name, args):
            async for event in original(ctx, *call_args, **kwargs):
                yield event
            return
        runtime.report(ctx.tool)

        def stream(checked: Any) -> Any:
            # Exactly the input Thoth checked; ExecuteToolContext is replaced, never mutated.
            checked_ctx = dataclasses.replace(ctx, tool_use={**tool_use, "input": checked})
            return original(checked_ctx, *call_args, **kwargs)

        try:
            async for event in _stream_checked(runtime.tracer(), name, args, stream, lambda event, last: getattr(event, "tool_result", last)):
                yield event
        except ThothPolicyViolation as exc:
            yield _strands_refusal(tool_use, exc.reason)

    setattr(_run_tool, _THOTH_GOVERNED, True)
    return _run_tool


def _strands_refusal(tool_use: Any, reason: str) -> Any:
    """The tool result a refused Strands call ends with, as Strands reports tool errors."""
    ToolResultEvent = importlib.import_module("strands.types.events").ToolResultEvent
    tool_use_id = tool_use.get("toolUseId") if isinstance(tool_use, dict) else None
    return ToolResultEvent({"toolUseId": str(tool_use_id), "status": "error", "content": [{"text": _refused(reason)}]})


def _install_strands() -> Callable[[], None] | None:
    """Governs Strands Agents tools at the tool executor's _run_tool.

    Every tool call (the agent's event loop, direct ``agent.tool.x()`` calls,
    the bidirectional loop and background tools) reaches the tool through
    ``_run_tool``, the terminal of the tool middleware chain, after
    BeforeToolCallEvent hooks and middleware have settled the input, so the
    input checked is the input run, for @tool functions, MCP tools and agents
    used as tools alike.
    """
    try:
        executor = importlib.import_module("strands.tools.executors._executor")
    except ImportError:
        return None
    original = executor.__dict__.get("_run_tool")
    if original is None or getattr(original, _THOTH_GOVERNED, False):
        return None
    executor._run_tool = _govern_strands_run_tool(original)

    def undo() -> None:
        executor._run_tool = original

    return undo


# --- AG2 ---------------------------------------------------------------------------


def _govern_ag2_execute(original: Callable[..., Any]) -> Callable[..., Any]:
    """The governed form of an AG2 FunctionTool's _execute(event, context)."""
    events = importlib.import_module("ag2.events")

    def refusal(event: Any, reason: str, error: Exception) -> Any:
        # An error result, as AG2 reports a tool's own failure, with the
        # refusal as its text rather than a traceback.
        return events.ToolErrorEvent(parent_id=event.id, name=event.name, error=error, result=events.ToolResult(_refused(reason)))

    @functools.wraps(original)
    async def _execute(self: Any, event: Any, context: Any, *args: Any, **kwargs: Any) -> Any:
        runtime = _runtime()
        if runtime is None:
            return await original(self, event, context, *args, **kwargs)
        name = getattr(self, "name", "")
        try:
            # The arguments after AG2's tool middleware, which may replace them.
            arguments = event.serialized_arguments
        except Exception as exc:
            return refusal(event, "the call can't be checked, so it was not run", exc)
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            return refusal(event, "the call can't be checked, so it was not run", ValueError("unchecked tool call"))
        if claim_checked_call(name, arguments):
            return await original(self, event, context, *args, **kwargs)
        function = getattr(getattr(self, "schema", None), "function", None)
        runtime.report(SimpleNamespace(name=name, description=getattr(function, "description", "") or "", input_schema=getattr(function, "parameters", None)))
        tracer = runtime.tracer()

        async def run(checked: Any) -> Any:
            # Exactly the arguments Thoth checked.
            call = copy.copy(event)
            call.serialized_arguments = checked
            return await original(self, call, context, *args, **kwargs)

        token = _CURRENT_SESSION.set(tracer._session)
        try:
            return await tracer.wrap_tool(name, run)(arguments)
        except ThothPolicyViolation as exc:
            return refusal(event, exc.reason, exc)
        finally:
            _CURRENT_SESSION.reset(token)

    setattr(_execute, _THOTH_GOVERNED, True)
    return _execute


def _install_ag2() -> Callable[[], None] | None:
    """Governs AG2 (1.x) function tools at FunctionTool._execute.

    Each function tool an AG2 Agent runs, from @tool, a Toolkit or a
    subagent tool, ends in FunctionTool._execute(event, context) after the
    tool's own and the agent's tool middleware, which may replace the
    arguments; the arguments checked are the ones run. MCP toolkits call
    their servers through mcp's ClientSession, which the MCP hook governs.
    Client tools run in the caller's own application, not in AG2. A refusal
    returns an error tool result, "Thoth refused this call: <reason>".
    """
    try:
        FunctionTool = importlib.import_module("ag2.tools.final.function_tool").FunctionTool
    except ImportError:
        return None
    wrapped: list[tuple[type, Any]] = []

    def walk(cls: type) -> None:
        original = cls.__dict__.get("_execute")
        if original is not None and not getattr(original, _THOTH_GOVERNED, False):
            wrapped.append((cls, original))
            cls._execute = _govern_ag2_execute(original)  # type: ignore[attr-defined]
        for subclass in cls.__subclasses__():
            walk(subclass)

    walk(FunctionTool)
    if not wrapped:
        return None

    def undo() -> None:
        for cls, original in wrapped:
            cls._execute = original  # type: ignore[attr-defined]

    return undo


# --- Claude Agent SDK ------------------------------------------------------------


def _governed_claude_options(options: Any) -> Any:
    """A governed copy of ClaudeAgentOptions for one query or client connection.

    The caller's options are never changed: hooks are copied before Thoth's are
    added. Options someone already instrumented by hand are left as they are.
    """
    runtime = _runtime()
    if runtime is None:
        return options
    if options is None:
        options = importlib.import_module("claude_agent_sdk.types").ClaudeAgentOptions()
    if getattr(options.can_use_tool, _THOTH_GOVERNED, False):
        return options
    governed = dataclasses.replace(options, hooks={event: list(matchers) for event, matchers in (options.hooks or {}).items()})
    integration = importlib.import_module("thoth.integrations.claude_agent_sdk")
    governed = integration.instrument_claude_agent_sdk_options(governed, runtime.tracer())
    if options.permission_prompt_tool_name:
        # The SDK refuses can_use_tool alongside a permission prompt tool; the
        # PreToolUse hook still checks every call.
        governed.can_use_tool = options.can_use_tool
    return governed


def _install_claude_agent_sdk() -> Callable[[], None] | None:
    """Governs every Claude Agent SDK query() and ClaudeSDKClient connection.

    Both entry points hand their options to the CLI in one place:
    InternalClient.process_query for query(), and ClaudeSDKClient._connect_inner
    for the client. Each gets a governed copy of its options there, with the
    PreToolUse hook that checks every tool call (including tools in
    allowed_tools) and the governed can_use_tool, so no options need changing.
    The tracer is the one for the user acting when the query starts or the
    client connects.
    """
    try:
        InternalClient = importlib.import_module("claude_agent_sdk._internal.client").InternalClient
        ClaudeSDKClient = importlib.import_module("claude_agent_sdk.client").ClaudeSDKClient
    except ImportError:
        return None
    process_query = InternalClient.__dict__.get("process_query")
    connect_inner = ClaudeSDKClient.__dict__.get("_connect_inner")
    if process_query is None or connect_inner is None or getattr(process_query, _THOTH_GOVERNED, False):
        # An SDK version without both entry points isn't governed at all, never halfway.
        return None

    @functools.wraps(process_query)
    async def governed_process_query(self: Any, prompt: Any, options: Any, *args: Any, **kwargs: Any) -> Any:
        async for message in process_query(self, prompt, _governed_claude_options(options), *args, **kwargs):
            yield message

    @functools.wraps(connect_inner)
    async def governed_connect_inner(self: Any, *args: Any, **kwargs: Any) -> Any:
        options = self.options
        self.options = _governed_claude_options(options)
        try:
            return await connect_inner(self, *args, **kwargs)
        finally:
            # The connection holds the governed hooks; the caller's options stay theirs.
            self.options = options

    setattr(governed_process_query, _THOTH_GOVERNED, True)
    setattr(governed_connect_inner, _THOTH_GOVERNED, True)
    InternalClient.process_query = governed_process_query
    ClaudeSDKClient._connect_inner = governed_connect_inner

    def undo() -> None:
        InternalClient.process_query = process_query
        ClaudeSDKClient._connect_inner = connect_inner

    return undo


# --- Anthropic client tools (anthropic) ---------------------------------------------


def _anthropic_tool_name(tool: Any) -> str:
    try:
        name = tool.name
    except Exception:
        # A built-in tool whose definition has no name.
        return ""
    return name if isinstance(name, str) else ""


def _govern_anthropic_call(original: Callable[..., Any]) -> Callable[..., Any]:
    """The governed form of an Anthropic tool's call(input)."""
    tool_error = importlib.import_module("anthropic.lib.tools").ToolError

    def prepare(self: Any, tool_input: Any) -> tuple[_Runtime | None, str]:
        runtime = _runtime()
        if runtime is None:
            return None, ""
        name = _anthropic_tool_name(self)
        if not name or not isinstance(tool_input, dict):
            # Nothing Thoth can check, so the tool doesn't run.
            raise tool_error(_refused("the call can't be checked, so it was not run"))
        if claim_checked_call(name, tool_input):
            # Already checked: a subclass calling super().call with the same input.
            return None, ""
        runtime.report(SimpleNamespace(name=name, description=getattr(self, "description", "") or "", input_schema=getattr(self, "input_schema", None)))
        return runtime, name

    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def acall(self: Any, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
            runtime, name = prepare(self, tool_input)
            if runtime is None:
                return await original(self, tool_input, *args, **kwargs)
            tracer = runtime.tracer()

            async def run(checked: Any) -> Any:
                # Exactly the input Thoth checked.
                return await original(self, checked, *args, **kwargs)

            token = _CURRENT_SESSION.set(tracer._session)
            try:
                return await tracer.wrap_tool(name, run)(tool_input)
            except ThothPolicyViolation as exc:
                # The runner sends a ToolError's content to Claude as an error result.
                raise tool_error(_refused(exc.reason)) from exc
            finally:
                _CURRENT_SESSION.reset(token)

        setattr(acall, _THOTH_GOVERNED, True)
        return acall

    @functools.wraps(original)
    def call(self: Any, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
        runtime, name = prepare(self, tool_input)
        if runtime is None:
            return original(self, tool_input, *args, **kwargs)
        tracer = runtime.tracer()
        token = _CURRENT_SESSION.set(tracer._session)
        try:
            return tracer.wrap_tool(name, lambda checked: original(self, checked, *args, **kwargs))(tool_input)
        except ThothPolicyViolation as exc:
            raise tool_error(_refused(exc.reason)) from exc
        finally:
            _CURRENT_SESSION.reset(token)

    setattr(call, _THOTH_GOVERNED, True)
    return call


def _install_anthropic() -> Callable[[], None] | None:
    """Governs the Anthropic client's runnable tools at call(input).

    The tool runners (client.beta.messages.tool_runner and the session tool
    runner) run every tool with tool.call(tool_use.input), so the input
    checked is the one run; a direct tool.call() is governed too. That covers
    @beta_tool and @beta_async_tool functions, the MCP helper's tools, and
    built-in tools such as the memory tool. Built-in tool classes override
    call, so it is wrapped on each class that defines its own, and on
    subclasses defined after init(). A refusal raises ToolError, "Thoth
    refused this call: <reason>", which the runner sends to Claude as an
    error tool result.
    """
    try:
        tools = importlib.import_module("anthropic.lib.tools")
    except ImportError:
        return None
    roots = [getattr(tools, name, None) for name in ("BetaFunctionTool", "BetaAsyncFunctionTool", "BetaBuiltinFunctionTool", "BetaAsyncBuiltinFunctionTool")]
    roots = [root for root in roots if isinstance(root, type)]
    if not roots:
        return None
    wrapped: list[tuple[type, Any]] = []
    hooks: list[tuple[type, Any]] = []

    def wrap(cls: type) -> None:
        original = cls.__dict__.get("call")
        if original is None or getattr(original, _THOTH_GOVERNED, False) or getattr(original, "__isabstractmethod__", False):
            return
        wrapped.append((cls, original))
        cls.call = _govern_anthropic_call(original)  # type: ignore[attr-defined]

    def walk(cls: type) -> None:
        wrap(cls)
        for subclass in cls.__subclasses__():
            walk(subclass)

    def subclass_hook(root: type, previous_hook: Any) -> Any:
        def init_subclass(cls: type, **kwargs: Any) -> None:
            if previous_hook is not None:
                previous_hook.__func__(cls, **kwargs)
            else:
                super(root, cls).__init_subclass__(**kwargs)
            wrap(cls)

        return classmethod(init_subclass)

    for root in roots:
        walk(root)
        previous_hook = root.__dict__.get("__init_subclass__")
        root.__init_subclass__ = subclass_hook(root, previous_hook)  # type: ignore[assignment]
        hooks.append((root, previous_hook))

    def undo() -> None:
        for cls, original in wrapped:
            cls.call = original  # type: ignore[attr-defined]
        for root, previous_hook in hooks:
            if previous_hook is None:
                del root.__init_subclass__
            else:
                root.__init_subclass__ = previous_hook

    return undo


# --- MCP client ------------------------------------------------------------------


def _mcp_refusal(reason: str) -> Any:
    """A refused MCP call's result, as MCP servers report tool errors."""
    types = importlib.import_module("mcp.types")
    return types.CallToolResult(content=[types.TextContent(type="text", text=_refused(reason))], is_error=True)


def _install_mcp_client() -> Callable[[], None] | None:
    """Governs every MCP tools/call made with ClientSession.call_tool.

    That covers apps calling MCP servers directly, ClientSessionGroup, and the
    MCP adapters of frameworks. A call a framework hook already checked (the
    same arguments, under the MCP tool's name or a prefixed form of it) isn't
    checked again; any other call is. A refusal returns an error
    CallToolResult, "Thoth refused this call: <reason>", and nothing is sent
    to the server.
    """
    try:
        ClientSession = importlib.import_module("mcp.client.session").ClientSession
    except ImportError:
        return None
    original = ClientSession.__dict__.get("call_tool")
    if original is None or getattr(original, _THOTH_GOVERNED, False):
        return None

    @functools.wraps(original)
    async def call_tool(self: Any, name: Any, arguments: Any = None, *args: Any, **kwargs: Any) -> Any:
        runtime = _runtime()
        if runtime is None:
            return await original(self, name, arguments, *args, **kwargs)
        checked_args = {} if arguments is None else arguments
        if not isinstance(name, str) or not name or not isinstance(checked_args, dict):
            # Nothing Thoth can check, so nothing is sent.
            return _mcp_refusal("the call can't be checked, so it was not run")
        if claim_checked_mcp_call(name, checked_args):
            return await original(self, name, arguments, *args, **kwargs)
        runtime.report(SimpleNamespace(name=name, description=""))
        tracer = runtime.tracer()

        async def run(checked: Any) -> Any:
            # Exactly the arguments Thoth checked are sent to the server.
            return await original(self, name, checked, *args, **kwargs)

        token = _CURRENT_SESSION.set(tracer._session)
        try:
            return await tracer.wrap_tool(name, run)(checked_args)
        except ThothPolicyViolation as exc:
            return _mcp_refusal(exc.reason)
        finally:
            _CURRENT_SESSION.reset(token)

    setattr(call_tool, _THOTH_GOVERNED, True)
    ClientSession.call_tool = call_tool

    def undo() -> None:
        ClientSession.call_tool = original

    return undo


_HOOKS: dict[str, Callable[[], Callable[[], None] | None]] = {
    "langchain": _install_langchain,
    "openai_agents": _install_openai_agents,
    "crewai": _install_crewai,
    "pydantic_ai": _install_pydantic_ai,
    "google_adk": _install_google_adk,
    "agent_framework": _install_agent_framework,
    "strands": _install_strands,
    "ag2": _install_ag2,
    "claude_agent_sdk": _install_claude_agent_sdk,
    "anthropic": _install_anthropic,
    "mcp_client": _install_mcp_client,
}
