# thoth/tracer.py
from __future__ import annotations

from datetime import UTC, datetime
import functools
import inspect
import logging
from typing import Any, Callable, cast

from thoth.emitter import SqsEmitter
from thoth.enforcer_client import EnforcerClient
from thoth.exceptions import ThothPolicyViolation
from thoth.logging_config import configure_thoth_logging_from_env
from thoth.models import (
    BehavioralEvent,
    EnforcementDecision,
    EnforcementMode,
    EventType,
    SourceType,
    ThothConfig,
)
from thoth.session import SessionContext
from thoth.step_up import StepUpClient

logger = logging.getLogger(__name__)


def _to_jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return "[truncated]"
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list | tuple):
        return [_to_jsonable(v, depth=depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v, depth=depth + 1) for k, v in value.items()}
    return str(value)


def _tool_args_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any] | None:
    if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
        return cast(dict[str, Any], _to_jsonable(args[0]))
    if not args and kwargs:
        return cast(dict[str, Any], _to_jsonable(kwargs))
    if not args and not kwargs:
        return None
    payload: dict[str, Any] = {"args": _to_jsonable(list(args))}
    if kwargs:
        payload["kwargs"] = _to_jsonable(kwargs)
    return payload


def _apply_modified_call_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    modified_tool_args: dict[str, Any] | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if not isinstance(modified_tool_args, dict) or not modified_tool_args:
        return args, kwargs

    if "args" in modified_tool_args and isinstance(modified_tool_args["args"], list):
        mapped_kwargs = kwargs
        if isinstance(modified_tool_args.get("kwargs"), dict):
            mapped_kwargs = cast(dict[str, Any], modified_tool_args["kwargs"])
        return tuple(modified_tool_args["args"]), mapped_kwargs

    if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
        return (modified_tool_args,), {}

    if "arg0" in modified_tool_args:
        return (modified_tool_args["arg0"],), kwargs
    if "input" in modified_tool_args:
        return (modified_tool_args["input"],), kwargs

    indexed: list[tuple[int, Any]] = []
    for key, value in modified_tool_args.items():
        if not key.startswith("arg"):
            continue
        index_text = key[3:]
        if not index_text.isdigit():
            continue
        indexed.append((int(index_text), value))
    indexed.sort(key=lambda item: item[0])
    if indexed and indexed[0][0] == 0 and indexed[-1][0] == len(indexed) - 1:
        return tuple(value for _, value in indexed), kwargs

    return args, kwargs


class Tracer:
    def __init__(
        self,
        config: ThothConfig,
        session: SessionContext,
        emitter: SqsEmitter,
        enforcer: EnforcerClient,
        step_up: StepUpClient,
    ) -> None:
        configure_thoth_logging_from_env()
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
                tool_args = _tool_args_from_call(args, kwargs)
                try:
                    effective_args, effective_kwargs = await self._aenforce(
                        tool_name,
                        tool_args=tool_args,
                        call_args=args,
                        call_kwargs=kwargs,
                    )  # async path — does not block event loop
                except ThothPolicyViolation as exc:
                    self._emit(tool_name, EventType.TOOL_CALL_BLOCK, exc.reason, violation_id=exc.violation_id)
                    raise
                result = await fn(*effective_args, **effective_kwargs)
                self._session.record_tool_call(tool_name)
                self._emit(tool_name, EventType.TOOL_CALL_POST, str(result))
                return result

            return async_wrapped

        @functools.wraps(fn)
        def sync_wrapped(*args: Any, **kwargs: Any) -> Any:
            self._emit(tool_name, EventType.TOOL_CALL_PRE, str(args))
            tool_args = _tool_args_from_call(args, kwargs)
            try:
                effective_args, effective_kwargs = self._enforce(
                    tool_name,
                    tool_args=tool_args,
                    call_args=args,
                    call_kwargs=kwargs,
                )
            except ThothPolicyViolation as exc:
                self._emit(tool_name, EventType.TOOL_CALL_BLOCK, exc.reason, violation_id=exc.violation_id)
                raise
            result = fn(*effective_args, **effective_kwargs)
            self._session.record_tool_call(tool_name)
            self._emit(tool_name, EventType.TOOL_CALL_POST, str(result))
            return result

        return sync_wrapped

    def _enforce(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        *,
        call_args: tuple[Any, ...] = (),
        call_kwargs: dict[str, Any] | None = None,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Synchronous enforcement check.

        Returns potentially modified call args/kwargs. Raises ThothPolicyViolation
        when policy blocks or defers execution.
        """
        kwargs = dict(call_kwargs or {})
        if self._config.enforcement == EnforcementMode.OBSERVE:
            return call_args, kwargs
        pending_tool_calls = list(self._session.tool_calls)
        if not pending_tool_calls or pending_tool_calls[-1] != tool_name:
            pending_tool_calls.append(tool_name)
        decision = self._enforcer.check(
            tool_name=tool_name,
            session_id=self._session.session_id,
            tool_calls=pending_tool_calls,
            tool_args=tool_args,
        )
        self._log_decision(tool_name, decision, async_path=False)
        if decision.is_step_up and decision.hold_token:
            decision = self._step_up.wait(decision.hold_token)
            self._log_decision(tool_name, decision, async_path=False, phase="step_up_resolved")
        if decision.is_defer:
            reason = decision.defer_reason or decision.reason or "deferred pending additional context"
            if decision.defer_timeout_seconds and decision.defer_timeout_seconds > 0:
                reason = f"{reason} (retry in {decision.defer_timeout_seconds}s)"
            raise ThothPolicyViolation(
                tool_name=tool_name,
                reason=reason,
                violation_id=decision.violation_id,
            )
        if decision.is_block:
            raise ThothPolicyViolation(
                tool_name=tool_name,
                reason=decision.reason or "blocked by Thoth policy",
                violation_id=decision.violation_id,
            )
        if decision.is_modify:
            return _apply_modified_call_args(call_args, kwargs, decision.modified_tool_args)
        return call_args, kwargs

    async def _aenforce(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        *,
        call_args: tuple[Any, ...] = (),
        call_kwargs: dict[str, Any] | None = None,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Async enforcement check using non-blocking I/O.

        Returns potentially modified call args/kwargs. Raises ThothPolicyViolation
        when policy blocks or defers execution.
        """
        kwargs = dict(call_kwargs or {})
        if self._config.enforcement == EnforcementMode.OBSERVE:
            return call_args, kwargs
        pending_tool_calls = list(self._session.tool_calls)
        if not pending_tool_calls or pending_tool_calls[-1] != tool_name:
            pending_tool_calls.append(tool_name)
        decision = await self._enforcer.acheck(
            tool_name=tool_name,
            session_id=self._session.session_id,
            tool_calls=pending_tool_calls,
            tool_args=tool_args,
        )
        self._log_decision(tool_name, decision, async_path=True)
        if decision.is_step_up and decision.hold_token:
            decision = await self._step_up.await_decision(decision.hold_token)
            self._log_decision(tool_name, decision, async_path=True, phase="step_up_resolved")
        if decision.is_defer:
            reason = decision.defer_reason or decision.reason or "deferred pending additional context"
            if decision.defer_timeout_seconds and decision.defer_timeout_seconds > 0:
                reason = f"{reason} (retry in {decision.defer_timeout_seconds}s)"
            raise ThothPolicyViolation(
                tool_name=tool_name,
                reason=reason,
                violation_id=decision.violation_id,
            )
        if decision.is_block:
            raise ThothPolicyViolation(
                tool_name=tool_name,
                reason=decision.reason or "blocked by Thoth policy",
                violation_id=decision.violation_id,
            )
        if decision.is_modify:
            return _apply_modified_call_args(call_args, kwargs, decision.modified_tool_args)
        return call_args, kwargs

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

    def _log_decision(
        self,
        tool_name: str,
        decision: EnforcementDecision,
        *,
        async_path: bool,
        phase: str = "enforce",
    ) -> None:
        trace_id = self._config.enforcement_trace_id or self._session.session_id
        logger.debug(
            (
                "thoth %s decision (%s path) tool=%s decision=%s "
                "authorization_decision=%s hold_token=%s reason_code=%s "
                "reason=%s trace_id=%s session_id=%s"
            ),
            phase,
            "async" if async_path else "sync",
            tool_name,
            decision.decision.value,
            decision.authorization_decision,
            decision.hold_token,
            decision.decision_reason_code,
            decision.reason,
            trace_id,
            self._session.session_id,
        )
