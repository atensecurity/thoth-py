"""Minimal, allowlisted telemetry projection.

Authorization requests intentionally keep their complete inputs.  This module is
only for retained telemetry sent by emitters.
"""

from __future__ import annotations

from typing import Any

from thoth.models import BehavioralEvent, EventType

_CONTENT = {
    EventType.LLM_INVOCATION: "thoth_sdk_session_start",
    EventType.TOOL_CALL_PRE: "tool invocation requested",
    EventType.TOOL_CALL_POST: "tool invocation completed",
    EventType.TOOL_CALL_BLOCK: "tool invocation blocked",
}
_STRING_FIELDS = {
    "sdk_language",
    "environment",
    "enforcement_trace_id",
    "action_attestation_id",
    "decision_id",
    "event_phase",
    "authorization_decision",
    "decision_reason_code",
    "action_classification",
    "pack_id",
    "pack_version",
    "result_type",
    "decision_envelope_version",
}
_NUMBER_FIELDS = {
    "duration_ms",
    "result_size_bytes",
    "risk_score",
    "latency_ms",
    "rule_version",
    "defer_timeout_seconds",
    "step_up_timeout_seconds",
}
_ID_LIST_FIELDS = {
    "regulatory_regimes",
    "matched_rule_ids",
    "matched_control_ids",
    "policy_references",
}
_RECEIPT_STRING_FIELDS = {
    "receipt_id",
    "decision_id",
    "signature",
    "signing_algorithm",
    "key_id",
    "schema_version",
}
_RECEIPT_DECISION_FIELDS = {"authorization_decision", "decision_reason_code", "outcome"}
_EVIDENCE_STRING_FIELDS = {
    "decision_id",
    "decision_reason_code",
    "authorization_decision",
    "action_classification",
    "decision_envelope_version",
    "pack_id",
    "pack_version",
}
_EVIDENCE_NUMBER_FIELDS = {"risk_score", "latency_ms", "rule_version"}


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, str)]


def _safe_receipt(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    result = {key: value[key] for key in _RECEIPT_STRING_FIELDS if isinstance(value.get(key), str)}
    decision = value.get("decision")
    if isinstance(decision, dict):
        safe = {key: decision[key] for key in _RECEIPT_DECISION_FIELDS if isinstance(decision.get(key), str)}
        if safe:
            result["decision"] = safe
    return result or None


def _safe_evidence(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {}
    for key in _EVIDENCE_STRING_FIELDS:
        if isinstance(value.get(key), str):
            result[key] = value[key]
    for key in _EVIDENCE_NUMBER_FIELDS:
        if isinstance(value.get(key), (int, float)) and not isinstance(value.get(key), bool):
            result[key] = value[key]
    for key in _ID_LIST_FIELDS:
        if (items := _string_list(value.get(key))) is not None:
            result[key] = items
    policy = value.get("policy")
    if isinstance(policy, dict):
        safe_policy: dict[str, Any] = {}
        for key in ("policy_id", "policy_version", "rule_id"):
            if isinstance(policy.get(key), str):
                safe_policy[key] = policy[key]
        for key in ("matched_rule_ids", "matched_control_ids", "policy_references"):
            if (items := _string_list(policy.get(key))) is not None:
                safe_policy[key] = items
        if safe_policy:
            result["policy"] = safe_policy
    return result or None


def telemetry_event(event: BehavioralEvent) -> dict[str, Any]:
    """Return a new wire payload without arbitrary application-controlled data."""
    source = event.metadata or {}
    metadata: dict[str, Any] = {"telemetry_capture": "minimal"}
    for key in _STRING_FIELDS:
        if isinstance(source.get(key), str):
            metadata[key] = source[key]
    for key in _NUMBER_FIELDS:
        if isinstance(source.get(key), (int, float)) and not isinstance(source.get(key), bool):
            metadata[key] = source[key]
    for key in _ID_LIST_FIELDS:
        if (items := _string_list(source.get(key))) is not None:
            metadata[key] = items
    if receipt := _safe_receipt(source.get("receipt")):
        metadata["receipt"] = receipt
    if evidence := _safe_evidence(source.get("decision_evidence")):
        metadata["decision_evidence"] = evidence
    if event.tool_name:
        metadata["tool_call"] = {"name": event.tool_name}

    return {
        "event_id": event.event_id,
        "tenant_id": event.tenant_id,
        "agent_id": event.agent_id,
        "session_id": event.session_id,
        "user_id": event.user_id,
        "source_type": event.source_type.value,
        "event_type": event.event_type.value,
        "tool_name": event.tool_name,
        "content": _CONTENT[event.event_type],
        "metadata": metadata,
        "approved_scope": list(event.approved_scope),
        "enforcement_mode": event.enforcement_mode.value,
        "session_tool_calls": list(event.session_tool_calls),
        "occurred_at": event.occurred_at.isoformat(),
        "ttl": event.ttl,
        "violation_id": event.violation_id,
    }
