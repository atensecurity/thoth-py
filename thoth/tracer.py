# thoth/tracer.py
from __future__ import annotations

from datetime import UTC, datetime
import functools
import inspect
import logging
from typing import Any, Callable

from thoth.emitter import SqsEmitter
from thoth.enforcer_client import EnforcerClient
from thoth.exceptions import ThothPolicyViolation
from thoth.models import (
    BehavioralEvent,
    EnforcementMode,
    EventType,
    SourceType,
    ThothConfig,
)
from thoth.session import SessionContext
from thoth.step_up import StepUpClient

logger = logging.getLogger(__name__)


class Tracer:
    def __init__(
        self,
        config: ThothConfig,
        session: SessionContext,
        emitter: SqsEmitter,
        enforcer: EnforcerClient,
        step_up: StepUpClient,
    ) -> None:
        self._config = config
        self._session = session
        self._emitter = emitter
        self._enforcer = enforcer
        self._step_up = step_up

    def wrap_tool(self, tool_name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
                self._emit(tool_name, EventType.TOOL_CALL_PRE, str(args))
                try:
                    await self._aenforce(tool_name)  # async path — does not block event loop
                except ThothPolicyViolation as exc:
                    self._emit(tool_name, EventType.TOOL_CALL_BLOCK, exc.reason, violation_id=exc.violation_id)
                    raise
                result = await fn(*args, **kwargs)
                self._session.record_tool_call(tool_name)
                self._emit(tool_name, EventType.TOOL_CALL_POST, str(result))
                return result

            return async_wrapped

        @functools.wraps(fn)
        def sync_wrapped(*args: Any, **kwargs: Any) -> Any:
            self._emit(tool_name, EventType.TOOL_CALL_PRE, str(args))
            try:
                self._enforce(tool_name)
            except ThothPolicyViolation as exc:
                self._emit(tool_name, EventType.TOOL_CALL_BLOCK, exc.reason, violation_id=exc.violation_id)
                raise
            result = fn(*args, **kwargs)
            self._session.record_tool_call(tool_name)
            self._emit(tool_name, EventType.TOOL_CALL_POST, str(result))
            return result

        return sync_wrapped

    def _enforce(self, tool_name: str) -> None:
        """Synchronous enforcement check. Raises ThothPolicyViolation on BLOCK."""
        if self._config.enforcement == EnforcementMode.OBSERVE:
            return
        decision = self._enforcer.check(
            tool_name=tool_name,
            session_id=self._session.session_id,
            tool_calls=list(self._session.tool_calls),
        )
        if decision.is_step_up and decision.hold_token:
            decision = self._step_up.wait(decision.hold_token)
        if decision.is_block:
            raise ThothPolicyViolation(
                tool_name=tool_name,
                reason=decision.reason or "blocked by Thoth policy",
                violation_id=decision.violation_id,
            )

    async def _aenforce(self, tool_name: str) -> None:
        """Async enforcement check. Uses non-blocking I/O. Raises ThothPolicyViolation on BLOCK."""
        if self._config.enforcement == EnforcementMode.OBSERVE:
            return
        decision = await self._enforcer.acheck(
            tool_name=tool_name,
            session_id=self._session.session_id,
            tool_calls=list(self._session.tool_calls),
        )
        if decision.is_step_up and decision.hold_token:
            decision = await self._step_up.await_decision(decision.hold_token)
        if decision.is_block:
            raise ThothPolicyViolation(
                tool_name=tool_name,
                reason=decision.reason or "blocked by Thoth policy",
                violation_id=decision.violation_id,
            )

    def _emit(self, tool_name: str, event_type: EventType, content: str, *, violation_id: str | None = None) -> None:
        event = BehavioralEvent(
            tenant_id=self._config.tenant_id,
            agent_id=self._config.agent_id,
            session_id=self._session.session_id,
            user_id=self._config.user_id,
            source_type=SourceType.AGENT_TOOL_CALL,
            event_type=event_type,
            tool_name=tool_name,
            content=content,
            approved_scope=self._config.approved_scope,
            enforcement_mode=self._config.enforcement,
            session_tool_calls=list(self._session.tool_calls),
            occurred_at=datetime.now(UTC),
            violation_id=violation_id,
        )
        self._emitter.emit(event)
