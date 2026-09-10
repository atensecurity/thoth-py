"""Thoth LangGraph integration.

Provides one-call instrumentation for LangGraph StateGraph/CompiledStateGraph
and LangChain/LangGraph tool lists.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import functools
import importlib
import inspect
import logging
import os
import time
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar
import uuid

from thoth._context import _CURRENT_SESSION
from thoth.emitter import HttpEmitter
from thoth.enforcer_client import EnforcerClient
from thoth.exceptions import ThothDeferredError, ThothPolicyViolation
from thoth.logging_config import configure_thoth_logging_from_env
from thoth.models import (
    BehavioralEvent,
    DecisionType,
    EnforcementDecision,
    EnforcementMode,
    EventType,
    SourceType,
    ThothConfig,
)
from thoth.session import SessionContext
from thoth.step_up import StepUpClient

if TYPE_CHECKING:
    from collections.abc import Iterable

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


def _require_langgraph() -> tuple[type[Any], type[Any], type[Any]]:
    """Load LangGraph classes lazily.

    Raises:
        ImportError: When langgraph is not installed.
    """

    try:
        graph_module = importlib.import_module("langgraph.graph")
        graph_state_module = importlib.import_module("langgraph.graph.state")
        prebuilt_module = importlib.import_module("langgraph.prebuilt")
    except ImportError as exc:
        raise ImportError("langgraph is required for this integration. Install it with: pip install langgraph") from exc
    StateGraph = graph_module.StateGraph
    CompiledStateGraph = graph_state_module.CompiledStateGraph
    ToolNode = prebuilt_module.ToolNode
    return StateGraph, CompiledStateGraph, ToolNode


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
        return dict(_to_jsonable(args[0]))
    if not args and kwargs:
        return dict(_to_jsonable(kwargs))
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
            mapped_kwargs = dict(modified_tool_args["kwargs"])
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
    }
    return {k: v for k, v in metadata.items() if v is not None}


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
    )


def _deferred_from_decision(
    tool_name: str,
    reason: str,
    decision: EnforcementDecision,
    *,
    fallback_decision: EnforcementDecision | None = None,
) -> ThothDeferredError:
    context = _merge_decision_context(decision, fallback_decision) if fallback_decision is not None else _decision_context(decision)
    return ThothDeferredError(
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
    )


def _bind_action_decision(tool_name: str, decision: EnforcementDecision, action_attestation_id: str) -> EnforcementDecision:
    """Carry the local action ID when a compatible server omits the echo."""
    if decision.action_attestation_id not in (None, "", action_attestation_id):
        # Evidence for another action cannot authorize this one. Do not attach
        # its receipt or policy evidence to the local action's denial event.
        raise ThothPolicyViolation(
            tool_name=tool_name,
            reason="enforcer response identifies a different action; tool was not executed",
            action_attestation_id=action_attestation_id,
            authorization_decision="BLOCK",
            decision_reason_code="action_attestation_id_mismatch",
        )
    return decision.model_copy(update={"action_attestation_id": action_attestation_id})


def _should_step_up_for_phi(tool_name: str, data_classification: str | None) -> bool:
    if str(data_classification or "").strip().upper() != "PHI":
        return False
    lowered = tool_name.lower()
    return any(token in lowered for token in ("write", "delete", "modify", "export"))


class _MockEnforcerClient:
    def __init__(self, config: ThothConfig, hold_decisions: dict[str, EnforcementDecision]) -> None:
        self._config = config
        self._hold_decisions = hold_decisions

    def _decision(self, tool_name: str, session_id: str) -> EnforcementDecision:
        in_scope = tool_name in self._config.approved_scope
        trace_id = self._config.enforcement_trace_id or session_id
        violation_id = f"vio_mock_{uuid.uuid4().hex[:12]}"

        if _should_step_up_for_phi(tool_name, self._config.data_classification):
            hold_token = f"tok_phi_{uuid.uuid4().hex[:12]}"
            allow = EnforcementDecision(
                decision=DecisionType.ALLOW,
                reason="mock step-up approved",
                authorization_decision="ALLOW",
                enforcement_trace_id=trace_id,
                decision_reason_code="mock_step_up_approved",
                latency_ms=2000.0,
            )
            self._hold_decisions[hold_token] = allow
            return EnforcementDecision(
                decision=DecisionType.STEP_UP,
                reason="mock PHI minimum-necessary step-up",
                authorization_decision="STEP_UP",
                hold_token=hold_token,
                enforcement_trace_id=trace_id,
                decision_reason_code="mock_phi_step_up",
                latency_ms=5.0,
            )

        if in_scope:
            return EnforcementDecision(
                decision=DecisionType.ALLOW,
                reason="mock allow: in approved scope",
                authorization_decision="ALLOW",
                enforcement_trace_id=trace_id,
                decision_reason_code="mock_scope_allow",
                latency_ms=5.0,
            )

        mode = self._config.enforcement
        if mode == EnforcementMode.OBSERVE:
            return EnforcementDecision(
                decision=DecisionType.ALLOW,
                reason="mock observe allow: out of scope",
                authorization_decision="ALLOW",
                enforcement_trace_id=trace_id,
                decision_reason_code="mock_observe_allow",
                latency_ms=5.0,
            )

        if mode in {EnforcementMode.STEP_UP, EnforcementMode.PROGRESSIVE}:
            hold_token = f"tok_step_up_{uuid.uuid4().hex[:12]}"
            allow = EnforcementDecision(
                decision=DecisionType.ALLOW,
                reason="mock step-up approved",
                authorization_decision="ALLOW",
                enforcement_trace_id=trace_id,
                decision_reason_code="mock_step_up_approved",
                latency_ms=2000.0,
            )
            self._hold_decisions[hold_token] = allow
            return EnforcementDecision(
                decision=DecisionType.STEP_UP,
                reason="mock out-of-scope step-up",
                authorization_decision="STEP_UP",
                hold_token=hold_token,
                enforcement_trace_id=trace_id,
                decision_reason_code="mock_scope_step_up",
                latency_ms=5.0,
            )

        return EnforcementDecision(
            decision=DecisionType.BLOCK,
            reason="mock block: tool not in approved scope",
            authorization_decision="BLOCK",
            violation_id=violation_id,
            enforcement_trace_id=trace_id,
            decision_reason_code="mock_scope_block",
            latency_ms=5.0,
        )

    def check(
        self,
        tool_name: str,
        session_id: str,
        tool_calls: list[str],
        tool_args: dict[str, Any] | None = None,
        action_attestation_id: str | None = None,
    ) -> EnforcementDecision:
        del tool_calls, tool_args, action_attestation_id
        return self._decision(tool_name, session_id)

    async def acheck(
        self,
        tool_name: str,
        session_id: str,
        tool_calls: list[str],
        tool_args: dict[str, Any] | None = None,
        action_attestation_id: str | None = None,
    ) -> EnforcementDecision:
        del tool_calls, tool_args, action_attestation_id
        return self._decision(tool_name, session_id)


class _MockStepUpClient:
    def __init__(self, hold_decisions: dict[str, EnforcementDecision]) -> None:
        self._hold_decisions = hold_decisions

    def wait(self, hold_token: str) -> EnforcementDecision:
        time.sleep(2.0)
        return self._hold_decisions.get(
            hold_token,
            EnforcementDecision(decision=DecisionType.BLOCK, reason="mock step-up token not found"),
        )

    async def await_decision(self, hold_token: str) -> EnforcementDecision:
        await asyncio.sleep(2.0)
        return self._hold_decisions.get(
            hold_token,
            EnforcementDecision(decision=DecisionType.BLOCK, reason="mock step-up token not found"),
        )


class _NoopEmitter:
    def emit(self, event: BehavioralEvent) -> None:
        del event


class _LangGraphRuntime:
    def __init__(
        self,
        config: ThothConfig,
        session: SessionContext,
        emitter: HttpEmitter | _NoopEmitter,
        enforcer: EnforcerClient | _MockEnforcerClient,
        step_up: StepUpClient | _MockStepUpClient,
    ) -> None:
        configure_thoth_logging_from_env()
        self._config = config
        self._session = session
        self._emitter = emitter
        self._enforcer = enforcer
        self._step_up = step_up

    def wrap_sync(self, tool_name: str, fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            action_attestation_id = str(uuid.uuid4())
            tool_args = _tool_args_from_call(tuple(args), dict(kwargs))
            started = perf_counter()
            self._emit(
                tool_name,
                EventType.TOOL_CALL_PRE,
                "tool invocation requested",
                metadata={
                    **self._base_tool_metadata(tool_name, tool_args, action_attestation_id),
                    "event_phase": "pre",
                },
            )
            original_call_args = tuple(args)
            original_call_kwargs = dict(kwargs)
            try:
                effective_args, effective_kwargs, decision = self._enforce_sync(
                    tool_name,
                    tool_args=tool_args,
                    call_args=tuple(args),
                    call_kwargs=dict(kwargs),
                    action_attestation_id=action_attestation_id,
                )
            except ThothDeferredError as exc:
                self._emit(
                    tool_name,
                    EventType.TOOL_CALL_POST,
                    exc.reason,
                    violation_id=exc.violation_id,
                    metadata={
                        **self._base_tool_metadata(tool_name, tool_args, action_attestation_id),
                        "event_phase": "post",
                        "duration_ms": int((perf_counter() - started) * 1000),
                        "authorization_decision": "DEFER",
                        **_policy_violation_metadata(exc),
                    },
                )
                raise
            except ThothPolicyViolation as exc:
                self._emit(
                    tool_name,
                    EventType.TOOL_CALL_BLOCK,
                    exc.reason,
                    violation_id=exc.violation_id,
                    metadata={
                        **self._base_tool_metadata(tool_name, tool_args, action_attestation_id),
                        "event_phase": "block",
                        "duration_ms": int((perf_counter() - started) * 1000),
                        **_policy_violation_metadata(exc),
                    },
                )
                raise

            result = fn(*effective_args, **effective_kwargs)
            self._session.record_tool_call(tool_name)
            metadata: dict[str, Any] = {
                **self._base_tool_metadata(tool_name, tool_args, action_attestation_id),
                "event_phase": "post",
                "duration_ms": int((perf_counter() - started) * 1000),
                "authorization_decision": decision.authorization_decision or decision.decision.value,
                **_result_summary(result),
                **_decision_context(decision),
            }
            if decision.is_modify:
                metadata["modification_reason"] = decision.modification_reason
                metadata["original_tool_args"] = _tool_args_from_call(
                    original_call_args,
                    original_call_kwargs,
                )
                metadata["modified_tool_args"] = _tool_args_from_call(
                    effective_args,
                    effective_kwargs,
                )
            self._emit(
                tool_name,
                EventType.TOOL_CALL_POST,
                "tool invocation completed",
                metadata={k: v for k, v in metadata.items() if v is not None},
            )
            return result

        return wrapped

    def wrap_async(self, tool_name: str, fn: Callable[P, Any]) -> Callable[P, Any]:
        @functools.wraps(fn)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> Any:
            action_attestation_id = str(uuid.uuid4())
            tool_args = _tool_args_from_call(tuple(args), dict(kwargs))
            started = perf_counter()
            self._emit(
                tool_name,
                EventType.TOOL_CALL_PRE,
                "tool invocation requested",
                metadata={
                    **self._base_tool_metadata(tool_name, tool_args, action_attestation_id),
                    "event_phase": "pre",
                },
            )
            original_call_args = tuple(args)
            original_call_kwargs = dict(kwargs)
            try:
                effective_args, effective_kwargs, decision = await self._enforce_async(
                    tool_name,
                    tool_args=tool_args,
                    call_args=tuple(args),
                    call_kwargs=dict(kwargs),
                    action_attestation_id=action_attestation_id,
                )
            except ThothDeferredError as exc:
                self._emit(
                    tool_name,
                    EventType.TOOL_CALL_POST,
                    exc.reason,
                    violation_id=exc.violation_id,
                    metadata={
                        **self._base_tool_metadata(tool_name, tool_args, action_attestation_id),
                        "event_phase": "post",
                        "duration_ms": int((perf_counter() - started) * 1000),
                        "authorization_decision": "DEFER",
                        **_policy_violation_metadata(exc),
                    },
                )
                raise
            except ThothPolicyViolation as exc:
                self._emit(
                    tool_name,
                    EventType.TOOL_CALL_BLOCK,
                    exc.reason,
                    violation_id=exc.violation_id,
                    metadata={
                        **self._base_tool_metadata(tool_name, tool_args, action_attestation_id),
                        "event_phase": "block",
                        "duration_ms": int((perf_counter() - started) * 1000),
                        **_policy_violation_metadata(exc),
                    },
                )
                raise

            result = await fn(*effective_args, **effective_kwargs)
            self._session.record_tool_call(tool_name)
            metadata: dict[str, Any] = {
                **self._base_tool_metadata(tool_name, tool_args, action_attestation_id),
                "event_phase": "post",
                "duration_ms": int((perf_counter() - started) * 1000),
                "authorization_decision": decision.authorization_decision or decision.decision.value,
                **_result_summary(result),
                **_decision_context(decision),
            }
            if decision.is_modify:
                metadata["modification_reason"] = decision.modification_reason
                metadata["original_tool_args"] = _tool_args_from_call(
                    original_call_args,
                    original_call_kwargs,
                )
                metadata["modified_tool_args"] = _tool_args_from_call(
                    effective_args,
                    effective_kwargs,
                )
            self._emit(
                tool_name,
                EventType.TOOL_CALL_POST,
                "tool invocation completed",
                metadata={k: v for k, v in metadata.items() if v is not None},
            )
            return result

        return wrapped

    def _observe_mode_allow(self, tool_name: str, decision: EnforcementDecision) -> EnforcementDecision:
        if decision.reason and "unavailable" in decision.reason.lower():
            logger.warning(
                "thoth: enforcer unavailable in observe mode, allowing tool=%s",
                tool_name,
            )
        return EnforcementDecision(
            decision=DecisionType.ALLOW,
            reason=decision.reason or "observe mode allows execution",
            authorization_decision="ALLOW",
            enforcement_trace_id=decision.enforcement_trace_id,
            action_attestation_id=decision.action_attestation_id,
            decision_reason_code=decision.decision_reason_code,
            action_classification=decision.action_classification,
            latency_ms=decision.latency_ms,
            pack_id=decision.pack_id,
            pack_version=decision.pack_version,
            rule_version=decision.rule_version,
            regulatory_regimes=decision.regulatory_regimes,
            matched_rule_ids=decision.matched_rule_ids,
            matched_control_ids=decision.matched_control_ids,
            policy_references=decision.policy_references,
            model_signals=decision.model_signals,
            fastml_features=decision.fastml_features,
            score_components=decision.score_components,
            top_contributors=decision.top_contributors,
            decision_evidence=decision.decision_evidence,
            receipt=decision.receipt,
        )

    def _enforce_sync(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None,
        *,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
        action_attestation_id: str,
    ) -> tuple[tuple[Any, ...], dict[str, Any], EnforcementDecision]:
        pending_tool_calls = self._session.pending_tool_calls(tool_name)
        decision = self._enforcer.check(
            tool_name=tool_name,
            session_id=self._session.session_id,
            tool_calls=pending_tool_calls,
            tool_args=tool_args,
            action_attestation_id=action_attestation_id,
        )
        decision = _bind_action_decision(tool_name, decision, action_attestation_id)
        self._log_decision(tool_name, decision, async_path=False)
        step_up_initial: EnforcementDecision | None = None

        if self._config.enforcement == EnforcementMode.OBSERVE:
            return call_args, call_kwargs, self._observe_mode_allow(tool_name, decision)

        if decision.is_step_up and decision.hold_token:
            step_up_initial = decision
            decision = self._step_up.wait(decision.hold_token)
            decision = _bind_action_decision(tool_name, decision, action_attestation_id)
            self._log_decision(tool_name, decision, async_path=False, phase="step_up_resolved")

        if decision.is_step_up:
            raise _violation_from_decision(
                tool_name,
                decision.reason or "step-up approval unresolved; tool was not executed",
                decision,
                fallback_decision=step_up_initial,
            )
        if decision.is_defer:
            reason = decision.defer_reason or decision.reason or "deferred pending additional context"
            if decision.defer_timeout_seconds and decision.defer_timeout_seconds > 0:
                reason = f"{reason} (retry in {decision.defer_timeout_seconds}s)"
            raise _deferred_from_decision(
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
            modified_args, modified_kwargs = _apply_modified_call_args(
                call_args,
                call_kwargs,
                decision.modified_tool_args,
            )
            return modified_args, modified_kwargs, decision
        return call_args, call_kwargs, decision

    async def _enforce_async(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None,
        *,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
        action_attestation_id: str,
    ) -> tuple[tuple[Any, ...], dict[str, Any], EnforcementDecision]:
        pending_tool_calls = self._session.pending_tool_calls(tool_name)
        decision = await self._enforcer.acheck(
            tool_name=tool_name,
            session_id=self._session.session_id,
            tool_calls=pending_tool_calls,
            tool_args=tool_args,
            action_attestation_id=action_attestation_id,
        )
        decision = _bind_action_decision(tool_name, decision, action_attestation_id)
        self._log_decision(tool_name, decision, async_path=True)
        step_up_initial: EnforcementDecision | None = None

        if self._config.enforcement == EnforcementMode.OBSERVE:
            return call_args, call_kwargs, self._observe_mode_allow(tool_name, decision)

        if decision.is_step_up and decision.hold_token:
            step_up_initial = decision
            decision = await self._step_up.await_decision(decision.hold_token)
            decision = _bind_action_decision(tool_name, decision, action_attestation_id)
            self._log_decision(tool_name, decision, async_path=True, phase="step_up_resolved")

        if decision.is_step_up:
            raise _violation_from_decision(
                tool_name,
                decision.reason or "step-up approval unresolved; tool was not executed",
                decision,
                fallback_decision=step_up_initial,
            )
        if decision.is_defer:
            reason = decision.defer_reason or decision.reason or "deferred pending additional context"
            if decision.defer_timeout_seconds and decision.defer_timeout_seconds > 0:
                reason = f"{reason} (retry in {decision.defer_timeout_seconds}s)"
            raise _deferred_from_decision(
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
            modified_args, modified_kwargs = _apply_modified_call_args(
                call_args,
                call_kwargs,
                decision.modified_tool_args,
            )
            return modified_args, modified_kwargs, decision
        return call_args, call_kwargs, decision

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
            "session_intent": self._config.session_intent,
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
        return {k: v for k, v in metadata.items() if v is not None}

    def _emit(
        self,
        tool_name: str,
        event_type: EventType,
        content: str,
        *,
        violation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        task_context = self._config.task_context
        chain = task_context.get("chain")
        event = BehavioralEvent(
            tenant_id=self._config.tenant_id,
            agent_id=self._config.agent_id,
            session_id=self._session.session_id,
            user_id=self._config.user_id,
            purpose=self._config.purpose,
            data_classification=self._config.data_classification,
            task_context=task_context,
            initiated_by=(str(task_context.get("initiated_by") or task_context.get("initiatedBy") or "").strip() or None),
            task_id=(str(task_context.get("task_id") or task_context.get("taskId") or "").strip() or None),
            delegation_chain=[str(item).strip() for item in (chain if isinstance(chain, list) else []) if str(item).strip()],
            source_type=SourceType.AGENT_TOOL_CALL,
            event_type=event_type,
            tool_name=tool_name,
            content=content,
            metadata={k: v for k, v in (metadata or {}).items() if v is not None},
            approved_scope=self._config.approved_scope,
            enforcement_mode=self._config.enforcement,
            session_tool_calls=self._session.tool_calls,
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
            ("thoth %s decision (%s path) tool=%s decision=%s authorization_decision=%s hold_token=%s reason_code=%s reason=%s trace_id=%s action_attestation_id=%s session_id=%s"),
            phase,
            "async" if async_path else "sync",
            tool_name,
            decision.decision.value,
            decision.authorization_decision,
            decision.hold_token,
            decision.decision_reason_code,
            decision.reason,
            trace_id,
            decision.action_attestation_id,
            self._session.session_id,
        )


def _coerce_task_context(task_context: str | dict[str, Any] | None) -> dict[str, Any]:
    if task_context is None:
        return {}
    if isinstance(task_context, dict):
        return dict(task_context)
    return {"context": str(task_context)}


def _build_runtime(
    *,
    agent_id: str,
    approved_scope: list[str],
    tenant_id: str,
    user_id: str,
    enforcement: str,
    api_key: str | None,
    api_url: str | None,
    session_id: str | None,
    session_intent: str | None,
    environment: str,
    enforcement_trace_id: str | None,
    purpose: str | None,
    data_classification: str | None,
    task_context: str | dict[str, Any] | None,
    fail_open: bool,
    event_ingest_token: str | None,
) -> _LangGraphRuntime:
    configure_thoth_logging_from_env()
    resolved_api_key = api_key or os.getenv("THOTH_API_KEY")
    resolved_event_ingest_token = event_ingest_token or os.getenv("THOTH_EVENT_INGEST_TOKEN")
    resolved_api_url = (api_url or os.getenv("THOTH_API_URL") or "").strip()
    if not resolved_api_url:
        raise ValueError("Thoth API URL is required (pass api_url or set THOTH_API_URL)")

    config = ThothConfig(
        agent_id=agent_id,
        approved_scope=approved_scope,
        tenant_id=tenant_id,
        user_id=user_id,
        enforcement=EnforcementMode(enforcement),
        api_key=resolved_api_key,
        event_ingest_token=resolved_event_ingest_token,
        api_url=resolved_api_url,
        session_intent=session_intent,
        purpose=purpose,
        data_classification=data_classification,
        task_context=_coerce_task_context(task_context),
        environment=(environment or os.getenv("THOTH_ENVIRONMENT") or "prod").strip().lower() or "prod",
        enforcement_trace_id=enforcement_trace_id,
        fail_open=fail_open,
    )
    session = SessionContext(config, session_id=session_id)
    _CURRENT_SESSION.set(session)

    mock_mode = os.getenv("THOTH_MOCK_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    if mock_mode:
        emitter: HttpEmitter | _NoopEmitter = _NoopEmitter()
    else:
        emitter = HttpEmitter(
            api_url=config.resolved_api_url,
            api_key=resolved_api_key or "",
            event_ingest_token=config.resolved_event_ingest_token,
        )

    if mock_mode:
        hold_decisions: dict[str, EnforcementDecision] = {}
        enforcer: EnforcerClient | _MockEnforcerClient = _MockEnforcerClient(config, hold_decisions)
        step_up: StepUpClient | _MockStepUpClient = _MockStepUpClient(hold_decisions)
    else:
        enforcer = EnforcerClient(config)
        step_up = StepUpClient(config)

    return _LangGraphRuntime(
        config=config,
        session=session,
        emitter=emitter,
        enforcer=enforcer,
        step_up=step_up,
    )


def _tool_name(tool: Any, fallback: str) -> str:
    name = getattr(tool, "name", None)
    if isinstance(name, str) and name.strip():
        return name
    return fallback


def _tool_async_callable(tool: Any, original_invoke: Callable[..., Any] | None, original_ainvoke: Callable[..., Any]) -> Callable[..., Any]:
    """Keep the standard LangChain executor fallback inside one enforcement call."""
    tools_module = importlib.import_module("langchain_core.tools")
    standard_fallbacks = (tools_module.StructuredTool.ainvoke, tools_module.Tool.ainvoke)
    if original_invoke is None or getattr(original_ainvoke, "__func__", None) not in standard_fallbacks:
        # Custom async implementations retain their own dispatch semantics.
        return original_ainvoke

    run_in_executor = importlib.import_module("langchain_core.runnables.config").run_in_executor

    @functools.wraps(original_ainvoke)
    async def invoke(input: Any, config: Any = None, **kwargs: Any) -> Any:  # noqa: A002 - LangChain's public keyword
        if tool.coroutine:
            return await original_ainvoke(input, config, **kwargs)
        # Upstream dispatches to self.invoke, which we also instrument. Use the
        # captured method so this action is checked once, without disabling the
        # wrappers on nested calls or mutating the shared tool during execution.
        return await run_in_executor(config, original_invoke, input, config, **kwargs)

    return invoke


def _wrap_tool_object(tool: Any, runtime: _LangGraphRuntime, *, default_name: str) -> Any:
    if getattr(tool, "__thoth_langgraph_wrapped__", False):
        return tool

    name = _tool_name(tool, default_name)
    original_invoke = None

    if hasattr(tool, "invoke") and callable(tool.invoke):
        original_invoke = tool.invoke
        wrapped_invoke = runtime.wrap_sync(name, original_invoke)
        try:
            object.__setattr__(tool, "invoke", wrapped_invoke)
        except Exception:
            tool.invoke = wrapped_invoke

    if hasattr(tool, "ainvoke") and callable(tool.ainvoke):
        original_ainvoke = tool.ainvoke
        wrapped_ainvoke = runtime.wrap_async(name, _tool_async_callable(tool, original_invoke, original_ainvoke))
        try:
            object.__setattr__(tool, "ainvoke", wrapped_ainvoke)
        except Exception:
            tool.ainvoke = wrapped_ainvoke

    try:
        object.__setattr__(tool, "__thoth_langgraph_wrapped__", True)
    except Exception:
        tool.__thoth_langgraph_wrapped__ = True
    return tool


def _iter_tool_nodes(graph_obj: Any, tool_node_type: type[Any]) -> Iterable[Any]:
    nodes = getattr(graph_obj, "nodes", None)
    if not isinstance(nodes, dict):
        return []

    seen: set[int] = set()
    found: list[Any] = []
    for node in nodes.values():
        candidates = [node, getattr(node, "runnable", None), getattr(node, "bound", None)]
        for candidate in candidates:
            if candidate is None:
                continue
            if not isinstance(candidate, tool_node_type):
                continue
            node_id = id(candidate)
            if node_id in seen:
                continue
            seen.add(node_id)
            found.append(candidate)
    return found


def _instrument_tool_node_instance(tool_node: Any, runtime: _LangGraphRuntime) -> None:
    tools_by_name = getattr(tool_node, "tools_by_name", None)
    if not isinstance(tools_by_name, dict):
        return

    for name, tool in list(tools_by_name.items()):
        tools_by_name[name] = _wrap_tool_object(tool, runtime, default_name=str(name))


def _instrument_graph(graph_obj: Any, runtime: _LangGraphRuntime) -> Any:
    _, _, tool_node_type = _require_langgraph()
    for tool_node in _iter_tool_nodes(graph_obj, tool_node_type):
        _instrument_tool_node_instance(tool_node, runtime)
    return graph_obj


def _instrument_tool_list(tools: list[Callable[..., Any]], runtime: _LangGraphRuntime) -> list[Callable[..., Any]]:
    governed: list[Callable[..., Any]] = []
    for idx, tool in enumerate(tools):
        if hasattr(tool, "invoke") or hasattr(tool, "ainvoke"):
            governed.append(_wrap_tool_object(tool, runtime, default_name=f"tool_{idx}"))
            continue

        tool_name = _tool_name(tool, getattr(tool, "__name__", f"tool_{idx}"))
        if inspect.iscoroutinefunction(tool):
            governed.append(runtime.wrap_async(tool_name, tool))
        else:
            governed.append(runtime.wrap_sync(tool_name, tool))
    return governed


def instrument_langgraph(
    graph_or_tools: Any,
    *,
    agent_id: str,
    approved_scope: list[str],
    tenant_id: str,
    user_id: str = "system",
    enforcement: str = "block",
    api_key: str | None = None,
    api_url: str | None = None,
    session_id: str | None = None,
    session_intent: str | None = None,
    environment: str = "prod",
    enforcement_trace_id: str | None = None,
    purpose: str | None = None,
    data_classification: str | None = None,
    task_context: str | dict[str, Any] | None = None,
    fail_open: bool = False,
    event_ingest_token: str | None = None,
) -> Any:
    """Instrument LangGraph tools, ToolNode graphs, or compiled graphs in one call.

    Args:
        graph_or_tools: A StateGraph, CompiledStateGraph, or list of LangGraph tools.
        agent_id: Unique identifier for this agent.
        approved_scope: Tool names allowed by policy.
        tenant_id: Your Thoth tenant identifier.
        user_id: User initiating the session (default: ``"system"``).
        enforcement: ``"observe"``, ``"block"``, ``"step_up"``, or ``"progressive"``.
        api_key: Thoth API key (or ``THOTH_API_KEY`` env var).
        api_url: Thoth API URL (or ``THOTH_API_URL`` env var).
        session_id: Optional fixed session ID. Generated if omitted.
        session_intent: Optional policy intent for this workflow.
        environment: Policy environment selector (default: ``"prod"``).
        enforcement_trace_id: Optional correlation ID for enforcement traces.
        purpose: Optional purpose context.
        data_classification: Optional sensitivity label (e.g., ``"PHI"``).
        task_context: Optional task context string/dict.
        fail_open: Allow execution on retryable/unreachable enforcer failures.
        event_ingest_token: Optional dedicated telemetry ingest token.

    Returns:
        Same type as input:
        - StateGraph in -> StateGraph out
        - CompiledStateGraph in -> CompiledStateGraph out
        - list in -> list out
    """

    state_graph_type, compiled_graph_type, _ = _require_langgraph()
    runtime = _build_runtime(
        agent_id=agent_id,
        approved_scope=approved_scope,
        tenant_id=tenant_id,
        user_id=user_id,
        enforcement=enforcement,
        api_key=api_key,
        api_url=api_url,
        session_id=session_id,
        session_intent=session_intent,
        environment=environment,
        enforcement_trace_id=enforcement_trace_id,
        purpose=purpose,
        data_classification=data_classification,
        task_context=task_context,
        fail_open=fail_open,
        event_ingest_token=event_ingest_token,
    )

    if isinstance(graph_or_tools, list):
        return _instrument_tool_list(graph_or_tools, runtime)
    if isinstance(graph_or_tools, state_graph_type):
        return _instrument_graph(graph_or_tools, runtime)
    if isinstance(graph_or_tools, compiled_graph_type):
        return _instrument_graph(graph_or_tools, runtime)

    raise TypeError("graph_or_tools must be a StateGraph, CompiledStateGraph, or list of callables/tools")


def instrument_tool_node(
    tools: list[Callable[..., Any]],
    *,
    agent_id: str,
    approved_scope: list[str],
    tenant_id: str,
    user_id: str = "system",
    enforcement: str = "block",
    api_key: str | None = None,
    api_url: str | None = None,
    session_id: str | None = None,
    session_intent: str | None = None,
    environment: str = "prod",
    enforcement_trace_id: str | None = None,
    purpose: str | None = None,
    data_classification: str | None = None,
    task_context: str | dict[str, Any] | None = None,
    fail_open: bool = False,
    event_ingest_token: str | None = None,
) -> Any:
    """Instrument a list of tools and return a governed LangGraph ToolNode."""

    _, _, tool_node_type = _require_langgraph()
    governed_tools = instrument_langgraph(
        tools,
        agent_id=agent_id,
        approved_scope=approved_scope,
        tenant_id=tenant_id,
        user_id=user_id,
        enforcement=enforcement,
        api_key=api_key,
        api_url=api_url,
        session_id=session_id,
        session_intent=session_intent,
        environment=environment,
        enforcement_trace_id=enforcement_trace_id,
        purpose=purpose,
        data_classification=data_classification,
        task_context=task_context,
        fail_open=fail_open,
        event_ingest_token=event_ingest_token,
    )
    return tool_node_type(governed_tools)


def thoth_graph(
    *,
    agent_id: str,
    approved_scope: list[str],
    tenant_id: str,
    user_id: str = "system",
    enforcement: str = "block",
    api_key: str | None = None,
    api_url: str | None = None,
    session_id: str | None = None,
    session_intent: str | None = None,
    environment: str = "prod",
    enforcement_trace_id: str | None = None,
    purpose: str | None = None,
    data_classification: str | None = None,
    task_context: str | dict[str, Any] | None = None,
    fail_open: bool = False,
    event_ingest_token: str | None = None,
) -> Callable[[Callable[P, Any]], Callable[P, Any]]:
    """Decorator that instruments ToolNode tools on the graph returned by a builder."""

    def decorator(builder: Callable[P, Any]) -> Callable[P, Any]:
        @functools.wraps(builder)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> Any:
            graph = builder(*args, **kwargs)
            return instrument_langgraph(
                graph,
                agent_id=agent_id,
                approved_scope=approved_scope,
                tenant_id=tenant_id,
                user_id=user_id,
                enforcement=enforcement,
                api_key=api_key,
                api_url=api_url,
                session_id=session_id,
                session_intent=session_intent,
                environment=environment,
                enforcement_trace_id=enforcement_trace_id,
                purpose=purpose,
                data_classification=data_classification,
                task_context=task_context,
                fail_open=fail_open,
                event_ingest_token=event_ingest_token,
            )

        return wrapped

    return decorator


__all__ = [
    "instrument_langgraph",
    "instrument_tool_node",
    "thoth_graph",
]
