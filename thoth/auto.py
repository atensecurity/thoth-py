"""One-call instrumentation: ``thoth.init()``.

Call it once at startup. It reads the connection from the environment and
governs every tool of each supported framework installed in the process, at
the one place that framework runs tools, so no tool is edited and nothing is
passed per call::

    import thoth

    thoth.init()  # THOTH_API_KEY, THOTH_API_URL, THOTH_TENANT_ID, THOTH_AGENT_ID

    with thoth.acting_for(request.user.email):
        graph.invoke(...)

Frameworks governed (more follow): LangChain tools, at BaseTool.run and
BaseTool.arun, which every tool's invoke goes through, including the tools
LangGraph's ToolNode runs.

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
import os
import threading
from types import SimpleNamespace
from typing import Any, Callable
import uuid

import httpx

from thoth._context import _CURRENT_SESSION, already_governed
from thoth.emitter import HttpEmitter
from thoth.enforcer_client import EnforcerClient, enforcer_http_clients
from thoth.logging_config import configure_thoth_logging_from_env
from thoth.models import EnforcementMode, ThothConfig
from thoth.session import SessionContext
from thoth.step_up import StepUpClient, step_up_http_clients
from thoth.tool_inventory import publish_tool_inventory
from thoth.tracer import Tracer

# Tracers kept for recent users and sessions; each is small, the clients are shared.
_MAX_TRACERS = 1024


@dataclass(frozen=True)
class _Actor:
    user_id: str
    session_id: str


_ACTOR: contextvars.ContextVar[_Actor | None] = contextvars.ContextVar("thoth_actor", default=None)


@dataclass
class _Runtime:
    config: ThothConfig
    emitter: HttpEmitter
    enforcer_http: tuple[httpx.Client, httpx.AsyncClient]
    step_up_http: tuple[httpx.Client, httpx.AsyncClient]
    default_actor: _Actor
    tracers: OrderedDict[_Actor, Tracer] = field(default_factory=OrderedDict)
    reported: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def tracer(self) -> Tracer:
        actor = _ACTOR.get() or self.default_actor
        with self.lock:
            tracer = self.tracers.get(actor)
            if tracer is not None:
                self.tracers.move_to_end(actor)
                return tracer
            config = self.config.model_copy(update={"user_id": actor.user_id})
            session = SessionContext(config, session_id=actor.session_id)
            tracer = Tracer(
                config=config,
                session=session,
                emitter=self.emitter,
                enforcer=EnforcerClient(config, http=self.enforcer_http[0], async_http=self.enforcer_http[1]),
                step_up=StepUpClient(config, http=self.step_up_http[0], async_http=self.step_up_http[1]),
            )
            self.tracers[actor] = tracer
            if len(self.tracers) > _MAX_TRACERS:
                self.tracers.popitem(last=False)
            return tracer

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
    recorded against this user and session.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("acting_for() needs the user the agent acts for")
    actor = _Actor(user_id=user_id.strip(), session_id=session_id or str(uuid.uuid4()))
    token = _ACTOR.set(actor)
    try:
        yield actor.session_id
    finally:
        _ACTOR.reset(token)


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
        if runtime is None or not isinstance(name, str) or not name or already_governed(name):
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
        if runtime is None or not isinstance(name, str) or not name or already_governed(name):
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


_HOOKS: dict[str, Callable[[], Callable[[], None] | None]] = {
    "langchain": _install_langchain,
}
