# thoth/tracer.py
from __future__ import annotations

from datetime import UTC, datetime
import functools
import inspect
import logging
from time import perf_counter
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
UTC = UTC


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


def _payload_size(value: Any) -> int:
    rendered = str(value)
    return len(rendered.encode("utf-8", errors="replace"))


def _result_summary(result: Any) -> dict[str, Any]:
    return {
        "result_type": type(result).__name__,
        "result_size_bytes": _payload_size(result),
    }


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


def _decision_context(decision: EnforcementDecision) -> dict[str, Any]:
    return {
        "decision_envelope_version": decision.decision_envelope_version,
        "enforcement_trace_id": decision.enforcement_trace_id,
        "decision_reason_code": decision.decision_reason_code,
        "action_classification": decision.action_classification,
        "authorization_decision": decision.authorization_decision or decision.decision.value,
        "defer_timeout_seconds": decision.defer_timeout_seconds,
        "step_up_timeout_seconds": decision.step_up_timeout_seconds,
        "risk_score": decision.risk_score,
        "latency_ms": decision.latency_ms,
        "pack_id": decision.pack_id,
        "pack_version": decision.pack_version,
        "rule_version": decision.rule_version,
        "regulatory_regimes": list(decision.regulatory_regimes),
        "matched_rule_ids": list(decision.matched_rule_ids),
        "matched_control_ids": list(decision.matched_control_ids),
        "policy_references": list(decision.policy_references),
        "model_signals": list(decision.model_signals),
        "fastml_features": dict(decision.fastml_features or {}),
        "score_components": decision.score_components,
        "top_contributors": list(decision.top_contributors),
        "decision_evidence": decision.decision_evidence,
        "receipt": decision.receipt,
    }


def _merge_decision_context(
    primary: EnforcementDecision,
    fallback: EnforcementDecision,
) -> dict[str, Any]:
    primary_ctx = _decision_context(primary)
    fallback_ctx = _decision_context(fallback)
    merged: dict[str, Any] = {}
    for key in primary_ctx:
        primary_value = primary_ctx[key]
        fallback_value = fallback_ctx.get(key)
        if isinstance(primary_value, list):
            merged[key] = primary_value or (fallback_value if isinstance(fallback_value, list) else [])
            continue
        if primary_value is None:
            merged[key] = fallback_value
            continue
        merged[key] = primary_value
    return merged


def _violation_from_decision(
    tool_name: str,
    reason: str,
    decision: EnforcementDecision,
    *,
    fallback_decision: EnforcementDecision | None = None,
) -> ThothPolicyViolation:
    context = _merge_decision_context(decision, fallback_decision) if fallback_decision is not None else _decision_context(decision)
    return ThothPolicyViolation(
        tool_name=tool_name,
        reason=reason,
        violation_id=decision.violation_id or (fallback_decision.violation_id if fallback_decision else None),
        decision_envelope_version=context.get("decision_envelope_version"),
        decision_reason_code=context.get("decision_reason_code"),
        action_classification=context.get("action_classification"),
        authorization_decision=context.get("authorization_decision"),
        enforcement_trace_id=context.get("enforcement_trace_id"),
        fastml_features=context.get("fastml_features"),
        score_components=context.get("score_components"),
        top_contributors=context.get("top_contributors"),
        decision_evidence=context.get("decision_evidence"),
        defer_timeout_seconds=context.get("defer_timeout_seconds"),
        step_up_timeout_seconds=context.get("step_up_timeout_seconds"),
        risk_score=context.get("risk_score"),
        latency_ms=context.get("latency_ms"),
        pack_id=context.get("pack_id"),
        pack_version=context.get("pack_version"),
        rule_version=context.get("rule_version"),
        regulatory_regimes=context.get("regulatory_regimes"),
        matched_rule_ids=context.get("matched_rule_ids"),
        matched_control_ids=context.get("matched_control_ids"),
        policy_references=context.get("policy_references"),
        model_signals=context.get("model_signals"),
        receipt=context.get("receipt"),
    )


def _policy_violation_metadata(exc: ThothPolicyViolation) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "decision_envelope_version": exc.decision_envelope_version,
        "decision_reason_code": exc.decision_reason_code,
        "action_classification": exc.action_classification,
        "authorization_decision": exc.authorization_decision,
        "defer_timeout_seconds": exc.defer_timeout_seconds,
        "step_up_timeout_seconds": exc.step_up_timeout_seconds,
        "risk_score": exc.risk_score,
        "latency_ms": exc.latency_ms,
        "pack_id": exc.pack_id,
        "pack_version": exc.pack_version,
        "rule_version": exc.rule_version,
        "regulatory_regimes": exc.regulatory_regimes,
        "matched_rule_ids": exc.matched_rule_ids,
        "matched_control_ids": exc.matched_control_ids,
        "policy_references": exc.policy_references,
        "model_signals": exc.model_signals,
        "enforcement_trace_id": exc.enforcement_trace_id,
        "fastml_features": exc.fastml_features,
        "score_components": exc.score_components,
        "top_contributors": exc.top_contributors,
        "decision_evidence": exc.decision_evidence,
        "receipt": exc.receipt,
    }
    return {k: v for k, v in metadata.items() if v is not None}


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
                tool_args = _tool_args_from_call(args, kwargs)
                started = perf_counter()
                self._emit(
                    tool_name,
                    EventType.TOOL_CALL_PRE,
                    "tool invocation requested",
                    metadata={
                        **self._base_tool_metadata(tool_name, tool_args),
                        "event_phase": "pre",
                    },
                )
                try:
                    effective_args, effective_kwargs = await self._aenforce(
                        tool_name,
                        tool_args=tool_args,
                        call_args=args,
                        call_kwargs=kwargs,
                    )  # async path — does not block event loop
                except ThothPolicyViolation as exc:
                    self._emit(
                        tool_name,
                        EventType.TOOL_CALL_BLOCK,
                        exc.reason,
                        violation_id=exc.violation_id,
                        metadata={
                            **self._base_tool_metadata(tool_name, tool_args),
                            "event_phase": "block",
                            "duration_ms": int((perf_counter() - started) * 1000),
                            **_policy_violation_metadata(exc),
                        },
                    )
                    raise
                result = await fn(*effective_args, **effective_kwargs)
                self._session.record_tool_call(tool_name)
                self._emit(
                    tool_name,
                    EventType.TOOL_CALL_POST,
                    "tool invocation completed",
                    metadata={
                        **self._base_tool_metadata(tool_name, tool_args),
                        "event_phase": "post",
                        "duration_ms": int((perf_counter() - started) * 1000),
                        "authorization_decision": "ALLOW",
                        **_result_summary(result),
                    },
                )
                return result

            return async_wrapped

        @functools.wraps(fn)
        def sync_wrapped(*args: Any, **kwargs: Any) -> Any:
            tool_args = _tool_args_from_call(args, kwargs)
            started = perf_counter()
            self._emit(
                tool_name,
                EventType.TOOL_CALL_PRE,
                "tool invocation requested",
                metadata={
                    **self._base_tool_metadata(tool_name, tool_args),
                    "event_phase": "pre",
                },
            )
            try:
                effective_args, effective_kwargs = self._enforce(
                    tool_name,
                    tool_args=tool_args,
                    call_args=args,
                    call_kwargs=kwargs,
                )
            except ThothPolicyViolation as exc:
                self._emit(
                    tool_name,
                    EventType.TOOL_CALL_BLOCK,
                    exc.reason,
                    violation_id=exc.violation_id,
                    metadata={
                        **self._base_tool_metadata(tool_name, tool_args),
                        "event_phase": "block",
                        "duration_ms": int((perf_counter() - started) * 1000),
                        **_policy_violation_metadata(exc),
                    },
                )
                raise
            result = fn(*effective_args, **effective_kwargs)
            self._session.record_tool_call(tool_name)
            self._emit(
                tool_name,
                EventType.TOOL_CALL_POST,
                "tool invocation completed",
                metadata={
                    **self._base_tool_metadata(tool_name, tool_args),
                    "event_phase": "post",
                    "duration_ms": int((perf_counter() - started) * 1000),
                    "authorization_decision": "ALLOW",
                    **_result_summary(result),
                },
            )
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
        step_up_initial: EnforcementDecision | None = None
        if decision.is_step_up and decision.hold_token:
            step_up_initial = decision
            decision = self._step_up.wait(decision.hold_token)
            self._log_decision(tool_name, decision, async_path=False, phase="step_up_resolved")
        if decision.is_defer:
            reason = decision.defer_reason or decision.reason or "deferred pending additional context"
            if decision.defer_timeout_seconds and decision.defer_timeout_seconds > 0:
                reason = f"{reason} (retry in {decision.defer_timeout_seconds}s)"
            raise _violation_from_decision(
                tool_name,
                reason,
                decision,
                fallback_decision=step_up_initial,
            )
        if decision.is_block:
            raise _violation_from_decision(
                tool_name,
                decision.reason or "blocked by Thoth policy",
                decision,
                fallback_decision=step_up_initial,
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
        step_up_initial: EnforcementDecision | None = None
        if decision.is_step_up and decision.hold_token:
            step_up_initial = decision
            decision = await self._step_up.await_decision(decision.hold_token)
            self._log_decision(tool_name, decision, async_path=True, phase="step_up_resolved")
        if decision.is_defer:
            reason = decision.defer_reason or decision.reason or "deferred pending additional context"
            if decision.defer_timeout_seconds and decision.defer_timeout_seconds > 0:
                reason = f"{reason} (retry in {decision.defer_timeout_seconds}s)"
            raise _violation_from_decision(
                tool_name,
                reason,
                decision,
                fallback_decision=step_up_initial,
            )
        if decision.is_block:
            raise _violation_from_decision(
                tool_name,
                decision.reason or "blocked by Thoth policy",
                decision,
                fallback_decision=step_up_initial,
            )
        if decision.is_modify:
            return _apply_modified_call_args(call_args, kwargs, decision.modified_tool_args)
        return call_args, kwargs

    def _base_tool_metadata(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None,
    ) -> dict[str, Any]:
        trace_id = self._config.enforcement_trace_id or self._session.session_id
        metadata: dict[str, Any] = {
            "sdk_language": "python",
            "environment": self._config.environment,
            "enforcement_trace_id": trace_id,
            "tool_call": {
                "name": tool_name,
                "arguments": _to_jsonable(tool_args or {}),
            },
        }
        if tool_args:
            metadata["tool_args"] = _to_jsonable(tool_args)
        if self._config.purpose:
            metadata["purpose"] = self._config.purpose
            metadata["purpose_context"] = self._config.purpose
        if self._config.data_classification:
            metadata["data_classification"] = self._config.data_classification
        if self._config.task_context:
            metadata["task_context"] = _to_jsonable(self._config.task_context)
            metadata["delegation_context"] = _to_jsonable(self._config.task_context)
        return metadata

    def _emit(
        self,
        tool_name: str,
        event_type: EventType,
        content: str,
        *,
        violation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event = BehavioralEvent(
            tenant_id=self._config.tenant_id,
            agent_id=self._config.agent_id,
            session_id=self._session.session_id,
            user_id=self._config.user_id,
            purpose=self._config.purpose,
            data_classification=self._config.data_classification,
            task_context=self._config.task_context,
            initiated_by=(str(self._config.task_context.get("initiated_by") or self._config.task_context.get("initiatedBy") or "").strip() or None),
            task_id=(str(self._config.task_context.get("task_id") or self._config.task_context.get("taskId") or "").strip() or None),
            delegation_chain=[
                str(item).strip() for item in (self._config.task_context.get("chain") if isinstance(self._config.task_context.get("chain"), list) else []) if str(item).strip()
            ],
            source_type=SourceType.AGENT_TOOL_CALL,
            event_type=event_type,
            tool_name=tool_name,
            content=content,
            metadata={k: v for k, v in (metadata or {}).items() if v is not None},
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
            ("thoth %s decision (%s path) tool=%s decision=%s authorization_decision=%s hold_token=%s reason_code=%s reason=%s trace_id=%s session_id=%s"),
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
