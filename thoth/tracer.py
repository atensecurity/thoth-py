# thoth/tracer.py
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
import enum
import functools
import inspect
import json
import logging
from time import perf_counter
from typing import Any, Callable, cast
import uuid

from thoth._context import governed_call
from thoth.emitter import HttpEmitter, SqsEmitter
from thoth.enforcer_client import EnforcerClient
from thoth.exceptions import ThothPolicyViolation
from thoth.logging_config import configure_thoth_logging_from_env
from thoth.models import (
    BehavioralEvent,
    EnforcementDecision,
    EnforcementMode,
    EventType,
    HumanExplanation,
    SourceType,
    ThothConfig,
)
from thoth.session import SessionContext
from thoth.step_up import StepUpClient

logger = logging.getLogger(__name__)
UTC = UTC


@dataclass(frozen=True, slots=True)
class _EnforcementOutcome:
    effective_args: tuple[Any, ...]
    effective_kwargs: dict[str, Any]
    decision: EnforcementDecision | None
    initial_decision: EnforcementDecision | None = None


_MAX_JSONABLE_DEPTH = 64
_MAX_JSONABLE_NODES = 100_000


class ToolArgumentsTooLargeError(ValueError):
    """The tool arguments are too large or too tangled to copy for the policy check."""


def _to_jsonable(value: Any) -> Any:
    """Copy ``value`` into plain JSON types without losing structure.

    The copy is what the enforcer evaluates, while the tool runs on the original
    objects, so it must keep every field the policy might read (for example the
    recipient of a message). Models, dataclasses, sets and enums keep their
    contents, and nothing is replaced by a placeholder: a value nested deeper
    than the limit is kept as its JSON text and a cycle is marked. Every field of
    a model or dataclass is kept, as the policy may need any of them. Arbitrary
    objects are not walked, since they can hold credentials; they become their
    string form. A value with more than 100,000 nodes (a shared reference
    repeated until it explodes, for one) raises ``ToolArgumentsTooLargeError`` so the
    call is refused rather than evaluated on a partial copy.
    """
    return _copy_jsonable(value, 0, frozenset(), [_MAX_JSONABLE_NODES])


def _copy_jsonable(value: Any, depth: int, seen: frozenset[int], budget: list[int]) -> Any:
    budget[0] -= 1
    if budget[0] < 0:
        raise ToolArgumentsTooLargeError("tool arguments are too large to evaluate")
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, enum.Enum):
        return _copy_jsonable(value.value, depth, seen, budget)
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if id(value) in seen:
        return "[circular]"
    if depth > _MAX_JSONABLE_DEPTH:
        return json.dumps(value, default=str)
    inner = seen | {id(value)}
    if isinstance(value, dict):
        return {str(k): _copy_jsonable(v, depth + 1, inner, budget) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_copy_jsonable(v, depth + 1, inner, budget) for v in value]
    if isinstance(value, set | frozenset):
        items = [_copy_jsonable(v, depth + 1, inner, budget) for v in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump()
        except Exception:
            dumped = None
        if dumped is not None:
            return _copy_jsonable(dumped, depth + 1, inner, budget)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _copy_jsonable(getattr(value, f.name, None), depth + 1, inner, budget)
            for f in dataclasses.fields(value)
        }
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


class ModifiedArgumentsNotApplicableError(ValueError):
    """The enforcer modified the arguments in a shape the SDK cannot map back onto the call."""


def _apply_modified_call_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    modified_tool_args: dict[str, Any] | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Map the enforcer's modified arguments back onto the call.

    The shape is decided by how the call was sent (see ``_tool_args_from_call``):
    a single dict argument, keyword arguments only, or ``{"args": [...], "kwargs": {...}}``.
    Raises ``ModifiedArgumentsNotApplicableError`` rather than run the original,
    unmodified arguments when the modification cannot be applied.
    """
    if not isinstance(modified_tool_args, dict) or not modified_tool_args:
        raise ModifiedArgumentsNotApplicableError("the enforcer modified the call but returned no arguments")

    if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
        return (modified_tool_args,), {}

    if not args and kwargs:
        return (), dict(modified_tool_args)

    if "args" in modified_tool_args and isinstance(modified_tool_args["args"], list):
        mapped_kwargs = kwargs
        if isinstance(modified_tool_args.get("kwargs"), dict):
            mapped_kwargs = dict(modified_tool_args["kwargs"])
        return tuple(modified_tool_args["args"]), mapped_kwargs

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

    raise ModifiedArgumentsNotApplicableError("the modified arguments do not match the shape of the call")


def _modification_to_apply(
    decision: EnforcementDecision,
    step_up_initial: EnforcementDecision | None,
) -> dict[str, Any] | None:
    """The modified arguments the call must run with, or None when it runs unchanged.

    A MODIFY decision carries them. A held action that is later approved does too:
    the hold only returns ALLOW, so the arguments the enforcer redacted when it
    held the call are on the initial decision. A MODIFY with no arguments yields
    an empty dict, which ``_apply_modified_call_args`` refuses.
    """
    if decision.is_modify:
        return decision.modified_tool_args or {}
    if decision.is_allow and step_up_initial is not None and step_up_initial.modified_tool_args:
        return step_up_initial.modified_tool_args
    return None


def _decision_context(decision: EnforcementDecision) -> dict[str, Any]:
    return {
        "decision_envelope_version": decision.decision_envelope_version,
        "enforcement_trace_id": decision.enforcement_trace_id,
        "action_attestation_id": decision.action_attestation_id,
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
        "fastml_features": dict(decision.fastml_features) if decision.fastml_features is not None else None,
        "score_components": decision.score_components,
        "top_contributors": list(decision.top_contributors),
        "decision_evidence": decision.decision_evidence,
        "receipt": decision.receipt,
        "terminal_receipt": decision.terminal_receipt,
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


def _enforcement_outcome_context(outcome: _EnforcementOutcome) -> dict[str, Any]:
    if outcome.decision is None:
        return {}
    if outcome.initial_decision is not None:
        return _merge_decision_context(outcome.decision, outcome.initial_decision)
    return _decision_context(outcome.decision)


def _violation_from_decision(
    tool_name: str,
    reason: str,
    decision: EnforcementDecision,
    *,
    fallback_decision: EnforcementDecision | None = None,
    explanation: HumanExplanation | None = None,
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
        action_attestation_id=context.get("action_attestation_id"),
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
        terminal_receipt=context.get("terminal_receipt"),
        explanation=explanation,
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
        "action_attestation_id": exc.action_attestation_id,
        "fastml_features": exc.fastml_features,
        "score_components": exc.score_components,
        "top_contributors": exc.top_contributors,
        "decision_evidence": exc.decision_evidence,
        "receipt": exc.receipt,
        "terminal_receipt": exc.terminal_receipt,
    }
    if exc.explanation is not None:
        explanation = exc.explanation
        if hasattr(explanation, "model_dump"):
            metadata["human_explanation"] = explanation.model_dump(mode="json")
        else:
            metadata["human_explanation"] = explanation
    return {k: v for k, v in metadata.items() if v is not None}


def _bind_decision_correlation(
    tool_name: str,
    decision: EnforcementDecision,
    *,
    action_attestation_id: str,
    enforcement_trace_id: str,
) -> EnforcementDecision:
    """Bind a response to the local request without accepting server rebinding."""
    if decision.action_attestation_id not in (None, "", action_attestation_id):
        raise ThothPolicyViolation(
            tool_name=tool_name,
            reason="enforcer response identifies a different action; tool was not executed",
            action_attestation_id=action_attestation_id,
            enforcement_trace_id=enforcement_trace_id,
            authorization_decision="BLOCK",
            decision_reason_code="action_attestation_id_mismatch",
        )
    if decision.enforcement_trace_id not in (None, "", enforcement_trace_id):
        raise ThothPolicyViolation(
            tool_name=tool_name,
            reason="enforcer response identifies a different trace; tool was not executed",
            action_attestation_id=action_attestation_id,
            enforcement_trace_id=enforcement_trace_id,
            authorization_decision="BLOCK",
            decision_reason_code="enforcement_trace_id_mismatch",
        )
    return decision.model_copy(
        update={
            "action_attestation_id": action_attestation_id,
            "enforcement_trace_id": enforcement_trace_id,
        }
    )


def _coerce_human_explanation(value: Any) -> HumanExplanation | None:
    if value is None:
        return None
    if isinstance(value, HumanExplanation):
        return value
    if isinstance(value, dict):
        try:
            return HumanExplanation.model_validate(value)
        except Exception:
            return None
    return None


def _resolve_action_attestation_id(config: ThothConfig) -> str:
    configured = (config.action_attestation_id or "").strip()
    if configured:
        return configured
    return str(uuid.uuid4())


class Tracer:
    def __init__(
        self,
        config: ThothConfig,
        session: SessionContext,
        emitter: SqsEmitter | HttpEmitter,
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
                action_attestation_id = _resolve_action_attestation_id(self._config)
                tool_args = _tool_args_from_call(args, kwargs)
                started = perf_counter()
                self._emit(
                    tool_name,
                    EventType.TOOL_CALL_PRE,
                    "tool invocation requested",
                    metadata={
                        **self._base_tool_metadata(
                            tool_name,
                            tool_args,
                            action_attestation_id,
                        ),
                        "event_phase": "pre",
                    },
                )
                try:
                    enforcement = await self._aenforce(
                        tool_name,
                        tool_args=tool_args,
                        call_args=args,
                        call_kwargs=kwargs,
                        action_attestation_id=action_attestation_id,
                    )  # async path — does not block event loop
                except ThothPolicyViolation as exc:
                    self._emit(
                        tool_name,
                        EventType.TOOL_CALL_BLOCK,
                        exc.reason,
                        violation_id=exc.violation_id,
                        metadata={
                            **self._base_tool_metadata(
                                tool_name,
                                tool_args,
                                action_attestation_id,
                            ),
                            "event_phase": "block",
                            "duration_ms": int((perf_counter() - started) * 1000),
                            **_policy_violation_metadata(exc),
                        },
                    )
                    raise
                with governed_call(tool_name, _tool_args_from_call(enforcement.effective_args, enforcement.effective_kwargs)):
                    result = await fn(*enforcement.effective_args, **enforcement.effective_kwargs)
                self._session.record_tool_call(tool_name)
                decision_context = _enforcement_outcome_context(enforcement)
                self._emit(
                    tool_name,
                    EventType.TOOL_CALL_POST,
                    "tool invocation completed",
                    metadata={
                        **self._base_tool_metadata(
                            tool_name,
                            tool_args,
                            action_attestation_id,
                        ),
                        "event_phase": "post",
                        "duration_ms": int((perf_counter() - started) * 1000),
                        **decision_context,
                        **_result_summary(result),
                    },
                )
                return result

            return async_wrapped

        @functools.wraps(fn)
        def sync_wrapped(*args: Any, **kwargs: Any) -> Any:
            action_attestation_id = _resolve_action_attestation_id(self._config)
            tool_args = _tool_args_from_call(args, kwargs)
            started = perf_counter()
            self._emit(
                tool_name,
                EventType.TOOL_CALL_PRE,
                "tool invocation requested",
                metadata={
                    **self._base_tool_metadata(
                        tool_name,
                        tool_args,
                        action_attestation_id,
                    ),
                    "event_phase": "pre",
                },
            )
            try:
                enforcement = self._enforce(
                    tool_name,
                    tool_args=tool_args,
                    call_args=args,
                    call_kwargs=kwargs,
                    action_attestation_id=action_attestation_id,
                )
            except ThothPolicyViolation as exc:
                self._emit(
                    tool_name,
                    EventType.TOOL_CALL_BLOCK,
                    exc.reason,
                    violation_id=exc.violation_id,
                    metadata={
                        **self._base_tool_metadata(
                            tool_name,
                            tool_args,
                            action_attestation_id,
                        ),
                        "event_phase": "block",
                        "duration_ms": int((perf_counter() - started) * 1000),
                        **_policy_violation_metadata(exc),
                    },
                )
                raise
            with governed_call(tool_name, _tool_args_from_call(enforcement.effective_args, enforcement.effective_kwargs)):
                result = fn(*enforcement.effective_args, **enforcement.effective_kwargs)
            self._session.record_tool_call(tool_name)
            decision_context = _enforcement_outcome_context(enforcement)
            self._emit(
                tool_name,
                EventType.TOOL_CALL_POST,
                "tool invocation completed",
                metadata={
                    **self._base_tool_metadata(
                        tool_name,
                        tool_args,
                        action_attestation_id,
                    ),
                    "event_phase": "post",
                    "duration_ms": int((perf_counter() - started) * 1000),
                    **decision_context,
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
        action_attestation_id: str,
    ) -> _EnforcementOutcome:
        """Synchronous enforcement check.

        Returns potentially modified call args/kwargs and the final decision.
        Raises ThothPolicyViolation when policy blocks or defers execution.
        """
        kwargs = dict(call_kwargs or {})
        if self._config.enforcement == EnforcementMode.OBSERVE:
            self._observe(tool_name, tool_args, action_attestation_id)
            return _EnforcementOutcome(call_args, kwargs, None)
        pending_tool_calls = list(self._session.tool_calls)
        if not pending_tool_calls or pending_tool_calls[-1] != tool_name:
            pending_tool_calls.append(tool_name)
        decision = self._enforcer.check(
            tool_name=tool_name,
            session_id=self._session.session_id,
            tool_calls=pending_tool_calls,
            tool_args=tool_args,
            action_attestation_id=action_attestation_id,
        )
        enforcement_trace_id = self._config.enforcement_trace_id or self._session.session_id
        decision = _bind_decision_correlation(
            tool_name,
            decision,
            action_attestation_id=action_attestation_id,
            enforcement_trace_id=enforcement_trace_id,
        )
        self._log_decision(
            tool_name,
            decision,
            async_path=False,
            action_attestation_id=action_attestation_id,
        )
        explanation: HumanExplanation | None = None
        if decision.is_block or decision.is_step_up:
            explanation = self._enforcer.explain(
                decision,
                tool_name=tool_name,
                session_id=self._session.session_id,
                tool_calls=pending_tool_calls,
                tool_args=tool_args,
                action_attestation_id=action_attestation_id,
            )
        step_up_initial: EnforcementDecision | None = None
        if decision.is_step_up and decision.hold_token:
            step_up_initial = decision
            decision = self._step_up.wait(decision.hold_token)
            decision = _bind_decision_correlation(
                tool_name,
                decision,
                action_attestation_id=action_attestation_id,
                enforcement_trace_id=enforcement_trace_id,
            )
            self._log_decision(
                tool_name,
                decision,
                async_path=False,
                phase="step_up_resolved",
                action_attestation_id=action_attestation_id,
            )
        if decision.is_step_up:
            raise _violation_from_decision(
                tool_name,
                decision.reason or "step-up approval unresolved; tool was not executed",
                decision,
                fallback_decision=step_up_initial,
                explanation=explanation,
            )
        if decision.is_defer:
            reason = decision.defer_reason or decision.reason or "deferred pending additional context"
            if decision.defer_timeout_seconds and decision.defer_timeout_seconds > 0:
                reason = f"{reason} (retry in {decision.defer_timeout_seconds}s)"
            raise _violation_from_decision(
                tool_name,
                reason,
                decision,
                fallback_decision=step_up_initial,
                explanation=explanation,
            )
        if decision.is_block:
            raise _violation_from_decision(
                tool_name,
                decision.reason or "blocked by Thoth policy",
                decision,
                fallback_decision=step_up_initial,
                explanation=explanation,
            )
        modification = _modification_to_apply(decision, step_up_initial)
        if modification is not None:
            try:
                effective_args, effective_kwargs = _apply_modified_call_args(call_args, kwargs, modification)
            except ModifiedArgumentsNotApplicableError as exc:
                raise _violation_from_decision(
                    tool_name,
                    f"{exc}; the call was not run",
                    decision,
                    fallback_decision=step_up_initial,
                    explanation=explanation,
                ) from exc
            return _EnforcementOutcome(effective_args, effective_kwargs, decision, step_up_initial)
        return _EnforcementOutcome(call_args, kwargs, decision, step_up_initial)

    async def _aenforce(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        *,
        call_args: tuple[Any, ...] = (),
        call_kwargs: dict[str, Any] | None = None,
        action_attestation_id: str,
    ) -> _EnforcementOutcome:
        """Async enforcement check using non-blocking I/O.

        Returns potentially modified call args/kwargs and the final decision.
        Raises ThothPolicyViolation when policy blocks or defers execution.
        """
        kwargs = dict(call_kwargs or {})
        if self._config.enforcement == EnforcementMode.OBSERVE:
            await self._aobserve(tool_name, tool_args, action_attestation_id)
            return _EnforcementOutcome(call_args, kwargs, None)
        pending_tool_calls = list(self._session.tool_calls)
        if not pending_tool_calls or pending_tool_calls[-1] != tool_name:
            pending_tool_calls.append(tool_name)
        decision = await self._enforcer.acheck(
            tool_name=tool_name,
            session_id=self._session.session_id,
            tool_calls=pending_tool_calls,
            tool_args=tool_args,
            action_attestation_id=action_attestation_id,
        )
        enforcement_trace_id = self._config.enforcement_trace_id or self._session.session_id
        decision = _bind_decision_correlation(
            tool_name,
            decision,
            action_attestation_id=action_attestation_id,
            enforcement_trace_id=enforcement_trace_id,
        )
        self._log_decision(
            tool_name,
            decision,
            async_path=True,
            action_attestation_id=action_attestation_id,
        )
        explanation: HumanExplanation | None = None
        if decision.is_block or decision.is_step_up:
            maybe_explanation: Any = None
            async_explain = getattr(self._enforcer, "aexplain", None)
            if callable(async_explain):
                maybe_explanation = async_explain(
                    decision,
                    tool_name=tool_name,
                    session_id=self._session.session_id,
                    tool_calls=pending_tool_calls,
                    tool_args=tool_args,
                    action_attestation_id=action_attestation_id,
                )
                if inspect.isawaitable(maybe_explanation):
                    maybe_explanation = await maybe_explanation
            else:
                sync_explain = getattr(self._enforcer, "explain", None)
                if callable(sync_explain):
                    maybe_explanation = sync_explain(
                        decision,
                        tool_name=tool_name,
                        session_id=self._session.session_id,
                        tool_calls=pending_tool_calls,
                        tool_args=tool_args,
                        action_attestation_id=action_attestation_id,
                    )
            explanation = _coerce_human_explanation(maybe_explanation)
        step_up_initial: EnforcementDecision | None = None
        if decision.is_step_up and decision.hold_token:
            step_up_initial = decision
            decision = await self._step_up.await_decision(decision.hold_token)
            decision = _bind_decision_correlation(
                tool_name,
                decision,
                action_attestation_id=action_attestation_id,
                enforcement_trace_id=enforcement_trace_id,
            )
            self._log_decision(
                tool_name,
                decision,
                async_path=True,
                phase="step_up_resolved",
                action_attestation_id=action_attestation_id,
            )
        if decision.is_step_up:
            raise _violation_from_decision(
                tool_name,
                decision.reason or "step-up approval unresolved; tool was not executed",
                decision,
                fallback_decision=step_up_initial,
                explanation=explanation,
            )
        if decision.is_defer:
            reason = decision.defer_reason or decision.reason or "deferred pending additional context"
            if decision.defer_timeout_seconds and decision.defer_timeout_seconds > 0:
                reason = f"{reason} (retry in {decision.defer_timeout_seconds}s)"
            raise _violation_from_decision(
                tool_name,
                reason,
                decision,
                fallback_decision=step_up_initial,
                explanation=explanation,
            )
        if decision.is_block:
            raise _violation_from_decision(
                tool_name,
                decision.reason or "blocked by Thoth policy",
                decision,
                fallback_decision=step_up_initial,
                explanation=explanation,
            )
        modification = _modification_to_apply(decision, step_up_initial)
        if modification is not None:
            try:
                effective_args, effective_kwargs = _apply_modified_call_args(call_args, kwargs, modification)
            except ModifiedArgumentsNotApplicableError as exc:
                raise _violation_from_decision(
                    tool_name,
                    f"{exc}; the call was not run",
                    decision,
                    fallback_decision=step_up_initial,
                    explanation=explanation,
                ) from exc
            return _EnforcementOutcome(effective_args, effective_kwargs, decision, step_up_initial)
        return _EnforcementOutcome(call_args, kwargs, decision, step_up_initial)

    def _base_tool_metadata(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None,
        action_attestation_id: str,
    ) -> dict[str, Any]:
        trace_id = self._config.enforcement_trace_id or self._session.session_id
        metadata: dict[str, Any] = {
            "sdk_language": "python",
            "environment": self._config.environment,
            "enforcement_trace_id": trace_id,
            "action_attestation_id": action_attestation_id,
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
        chain = self._config.task_context.get("chain")
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
            delegation_chain=[str(item).strip() for item in (chain if isinstance(chain, list) else []) if str(item).strip()],
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

    def _observe(self, tool_name: str, tool_args: dict[str, Any] | None, action_attestation_id: str) -> None:
        """Ask the enforcer about the call and let it run whatever the answer.

        Observe mode still sends the call to the enforcer: it is the enforcer that
        records what it would have held or blocked, which is what an operator reads
        to decide whether an agent is ready to enforce. The answer is only logged;
        it never blocks, holds or modifies the call, and an enforcer failure is
        swallowed.
        """
        pending_tool_calls = list(self._session.tool_calls)
        if not pending_tool_calls or pending_tool_calls[-1] != tool_name:
            pending_tool_calls.append(tool_name)
        try:
            decision = self._enforcer.check(
                tool_name=tool_name,
                session_id=self._session.session_id,
                tool_calls=pending_tool_calls,
                tool_args=tool_args,
                action_attestation_id=action_attestation_id,
            )
            self._log_decision(tool_name, decision, async_path=False, phase="observe", action_attestation_id=action_attestation_id)
        except Exception:  # noqa: BLE001 - observing must never fail the call
            logger.warning("thoth: observe-mode enforcer check failed for %s", tool_name, exc_info=True)

    async def _aobserve(self, tool_name: str, tool_args: dict[str, Any] | None, action_attestation_id: str) -> None:
        """Async twin of ``_observe``."""
        pending_tool_calls = list(self._session.tool_calls)
        if not pending_tool_calls or pending_tool_calls[-1] != tool_name:
            pending_tool_calls.append(tool_name)
        try:
            decision = await self._enforcer.acheck(
                tool_name=tool_name,
                session_id=self._session.session_id,
                tool_calls=pending_tool_calls,
                tool_args=tool_args,
                action_attestation_id=action_attestation_id,
            )
            self._log_decision(tool_name, decision, async_path=True, phase="observe", action_attestation_id=action_attestation_id)
        except Exception:  # noqa: BLE001 - observing must never fail the call
            logger.warning("thoth: observe-mode enforcer check failed for %s", tool_name, exc_info=True)

    def _log_decision(
        self,
        tool_name: str,
        decision: EnforcementDecision,
        *,
        async_path: bool,
        phase: str = "enforce",
        action_attestation_id: str | None = None,
    ) -> None:
        trace_id = self._config.enforcement_trace_id or self._session.session_id
        resolved_attestation_id = decision.action_attestation_id or action_attestation_id or self._config.action_attestation_id
        logger.debug(
            ("thoth %s decision (%s path) tool=%s decision=%s authorization_decision=%s reason_code=%s trace_id=%s action_attestation_id=%s session_id=%s"),
            phase,
            "async" if async_path else "sync",
            tool_name,
            decision.decision.value,
            decision.authorization_decision,
            decision.decision_reason_code,
            trace_id,
            resolved_attestation_id,
            self._session.session_id,
        )
