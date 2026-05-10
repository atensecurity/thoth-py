# thoth/models.py
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
import os
import time
from typing import Any
import uuid

from pydantic import BaseModel, Field, model_validator


class EnforcementMode(StrEnum):
    OBSERVE = "observe"
    STEP_UP = "step_up"
    BLOCK = "block"
    PROGRESSIVE = "progressive"


class SourceType(StrEnum):
    AGENT_TOOL_CALL = "agent_tool_call"
    AGENT_LLM_INVOCATION = "agent_llm_invocation"


class EventType(StrEnum):
    TOOL_CALL_PRE = "TOOL_CALL_PRE"
    TOOL_CALL_POST = "TOOL_CALL_POST"
    TOOL_CALL_BLOCK = "TOOL_CALL_BLOCK"
    LLM_INVOCATION = "LLM_INVOCATION"


class DecisionType(StrEnum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    STEP_UP = "STEP_UP"
    MODIFY = "MODIFY"
    DEFER = "DEFER"


_DECISION_ALIASES: dict[str, DecisionType] = {
    "ALLOW": DecisionType.ALLOW,
    "BLOCK": DecisionType.BLOCK,
    "DENY": DecisionType.BLOCK,
    "STEP_UP": DecisionType.STEP_UP,
    "CHALLENGE": DecisionType.STEP_UP,
    "ESCALATE": DecisionType.STEP_UP,
    "REVIEW": DecisionType.STEP_UP,
    "MODIFY": DecisionType.MODIFY,
    "MODIFIED": DecisionType.MODIFY,
    "TRANSFORM": DecisionType.MODIFY,
    "DEFER": DecisionType.DEFER,
    "DEFERRED": DecisionType.DEFER,
    "HOLD": DecisionType.DEFER,
}


_TTL_90_DAYS = 90 * 24 * 60 * 60
UTC = UTC


def _tenant_scoped_event_id(tenant_id: str, event_id: str | None) -> str:
    tenant = (tenant_id or "").strip() or "unknown"
    raw_event_id = (event_id or "").strip() or str(uuid.uuid4())
    if raw_event_id.startswith(f"{tenant}:"):
        return raw_event_id
    return f"{tenant}:{raw_event_id}"


class BehavioralEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    agent_id: str | None = None
    session_id: str
    user_id: str
    purpose: str | None = None
    data_classification: str | None = None
    task_context: dict[str, Any] = Field(default_factory=dict)
    initiated_by: str | None = None
    task_id: str | None = None
    delegation_chain: list[str] = Field(default_factory=list)
    source_type: SourceType
    event_type: EventType
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    approved_scope: list[str] = Field(default_factory=list)
    enforcement_mode: EnforcementMode = EnforcementMode.BLOCK
    session_tool_calls: list[str] = Field(default_factory=list)
    tool_name: str = ""
    endpoint_id: str | None = None
    hostname: str | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ttl: int = Field(default=0)
    violation_id: str | None = None
    # WORM tamper-evident chain fields
    chain_index: int = 0
    hash: str = ""
    previous_hash: str = ""
    signature: str = ""

    @model_validator(mode="after")
    def set_ttl(self) -> BehavioralEvent:
        self.event_id = _tenant_scoped_event_id(self.tenant_id, self.event_id)
        if self.ttl == 0:
            self.ttl = int(time.time()) + _TTL_90_DAYS
        return self


class ThothConfig(BaseModel):
    agent_id: str
    approved_scope: list[str]
    tenant_id: str
    user_id: str = "system"
    enforcement: EnforcementMode = EnforcementMode.BLOCK
    # Supply THOTH_API_KEY to use Aten's managed service.
    # Events are sent over HTTPS; no AWS credentials required.
    api_key: str | None = None
    # Optional dedicated token for event ingestion.
    # When set, SDK sends X-Thoth-Event-Ingest-Token on /v1/events/batch.
    event_ingest_token: str | None = None
    api_url: str | None = None
    step_up_timeout_minutes: int = 15
    step_up_poll_interval_seconds: int = 5
    # Declare the purpose of this session (HIPAA minimum-necessary).
    # When a compliance pack defines session_scopes, tools outside the declared
    # intent are step-up-challenged even if they appear in the approved scope.
    session_intent: str | None = None
    # Optional purpose context for policy/DLP enforcement.
    purpose: str | None = None
    # Optional data sensitivity label for policy/DLP enforcement.
    data_classification: str | None = None
    # Optional delegation/task context with initiated_by/task_id/chain keys.
    task_context: dict[str, Any] = Field(default_factory=dict)
    # Env-scoped policy lookup at enforcer side ("dev", "staging", "prod", ...).
    environment: str = "prod"
    # Optional correlation identifier propagated across enforcer/fastml/deepllm.
    # Defaults to session_id when omitted.
    enforcement_trace_id: str | None = None

    @property
    def resolved_api_url(self) -> str:
        """Ingest/events API base URL: THOTH_API_URL env var > api_url field."""
        if override := os.getenv("THOTH_API_URL"):
            return override.rstrip("/")
        if self.api_url:
            return self.api_url.rstrip("/")
        raise ValueError("Thoth API URL is required (set api_url or THOTH_API_URL)")

    @property
    def resolved_enforcer_url(self) -> str:
        """Enforcer base URL (single-URL contract): same as resolved_api_url."""
        return self.resolved_api_url

    @property
    def resolved_event_ingest_token(self) -> str:
        """Dedicated event-ingest token: THOTH_EVENT_INGEST_TOKEN > config field."""
        if override := os.getenv("THOTH_EVENT_INGEST_TOKEN"):
            return override.strip()
        return (self.event_ingest_token or "").strip()


class EnforcementDecision(BaseModel):
    decision: DecisionType
    decision_envelope_version: str | None = None
    enforcement_trace_id: str | None = None
    authorization_decision: str | None = None
    decision_reason_code: str | None = None
    action_classification: str | None = None
    reason: str | None = None
    violation_id: str | None = None
    hold_token: str | None = None
    risk_score: float | None = None
    fastml_features: dict[str, float] | None = None
    score_components: dict[str, Any] | None = None
    top_contributors: list[dict[str, Any]] = Field(default_factory=list)
    decision_evidence: dict[str, Any] | None = None
    latency_ms: float | None = None
    pack_id: str | None = None
    pack_version: str | None = None
    rule_version: int | None = None
    regulatory_regimes: list[str] = Field(default_factory=list)
    matched_rule_ids: list[str] = Field(default_factory=list)
    matched_control_ids: list[str] = Field(default_factory=list)
    policy_references: list[str] = Field(default_factory=list)
    model_signals: list[str] = Field(default_factory=list)
    receipt: dict[str, Any] | None = None
    modified_tool_args: dict[str, Any] | None = None
    modification_reason: str | None = None
    defer_reason: str | None = None
    defer_timeout_seconds: int | None = None
    step_up_timeout_seconds: int | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_decision(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if not payload.get("decision_reason_code"):
            payload["decision_reason_code"] = payload.get("decisionReasonCode")
        if not payload.get("action_classification"):
            payload["action_classification"] = payload.get("actionClassification")
        if not payload.get("authorization_decision"):
            payload["authorization_decision"] = payload.get("authorizationDecision")
        if payload.get("risk_score") is None and payload.get("riskScore") is not None:
            payload["risk_score"] = payload.get("riskScore")
        if not payload.get("enforcement_trace_id"):
            payload["enforcement_trace_id"] = payload.get("enforcementTraceId")
        if payload.get("fastml_features") is None and payload.get("fastmlFeatures") is not None:
            payload["fastml_features"] = payload.get("fastmlFeatures")
        if payload.get("score_components") is None and payload.get("scoreComponents") is not None:
            payload["score_components"] = payload.get("scoreComponents")
        if not payload.get("top_contributors") and payload.get("topContributors") is not None:
            payload["top_contributors"] = payload.get("topContributors")
        if payload.get("decision_evidence") is None and payload.get("decisionEvidence") is not None:
            payload["decision_evidence"] = payload.get("decisionEvidence")
        if payload.get("decision_evidence") is None and isinstance(payload.get("metadata"), dict):
            payload["decision_evidence"] = payload["metadata"].get("decision_evidence")
        if not payload.get("decision_envelope_version"):
            payload["decision_envelope_version"] = payload.get("decisionEnvelopeVersion")
        if not payload.get("decision_envelope_version") and isinstance(payload.get("decision_evidence"), dict):
            payload["decision_envelope_version"] = payload["decision_evidence"].get("decision_envelope_version")
        if payload.get("latency_ms") is None and payload.get("latencyMs") is not None:
            payload["latency_ms"] = payload.get("latencyMs")
        if not payload.get("pack_id"):
            payload["pack_id"] = payload.get("packId")
        if not payload.get("pack_version"):
            payload["pack_version"] = payload.get("packVersion")
        if payload.get("rule_version") is None and payload.get("ruleVersion") is not None:
            payload["rule_version"] = payload.get("ruleVersion")
        if not payload.get("regulatory_regimes") and payload.get("regulatoryRegimes") is not None:
            payload["regulatory_regimes"] = payload.get("regulatoryRegimes")
        if not payload.get("matched_rule_ids") and payload.get("matchedRuleIds") is not None:
            payload["matched_rule_ids"] = payload.get("matchedRuleIds")
        if not payload.get("matched_control_ids") and payload.get("matchedControlIds") is not None:
            payload["matched_control_ids"] = payload.get("matchedControlIds")
        if not payload.get("policy_references") and payload.get("policyReferences") is not None:
            payload["policy_references"] = payload.get("policyReferences")
        if not payload.get("model_signals") and payload.get("modelSignals") is not None:
            payload["model_signals"] = payload.get("modelSignals")
        if payload.get("modified_tool_args") is None and payload.get("modifiedToolArgs") is not None:
            payload["modified_tool_args"] = payload.get("modifiedToolArgs")
        if not payload.get("modification_reason"):
            payload["modification_reason"] = payload.get("modificationReason")
        if not payload.get("defer_reason"):
            payload["defer_reason"] = payload.get("deferReason")
        if payload.get("defer_timeout_seconds") is None and payload.get("deferTimeoutSeconds") is not None:
            payload["defer_timeout_seconds"] = payload.get("deferTimeoutSeconds")
        if payload.get("step_up_timeout_seconds") is None and payload.get("stepUpTimeoutSeconds") is not None:
            payload["step_up_timeout_seconds"] = payload.get("stepUpTimeoutSeconds")
        if payload.get("violation_id") is None and payload.get("violationId") is not None:
            payload["violation_id"] = payload.get("violationId")
        if payload.get("hold_token") is None and payload.get("holdToken") is not None:
            payload["hold_token"] = payload.get("holdToken")
        raw = payload.get("decision")
        if not raw:
            raw = payload.get("authorization_decision")
        key = str(raw or "").strip().upper()
        payload["decision"] = _DECISION_ALIASES.get(key, DecisionType.BLOCK)
        if not payload.get("reason"):
            if payload["decision"] == DecisionType.MODIFY:
                payload["reason"] = payload.get("modification_reason")
            elif payload["decision"] == DecisionType.DEFER:
                payload["reason"] = payload.get("defer_reason")
        return payload

    @property
    def is_allow(self) -> bool:
        return self.decision == DecisionType.ALLOW

    @property
    def is_block(self) -> bool:
        return self.decision == DecisionType.BLOCK

    @property
    def is_step_up(self) -> bool:
        return self.decision == DecisionType.STEP_UP

    @property
    def is_modify(self) -> bool:
        return self.decision == DecisionType.MODIFY

    @property
    def is_defer(self) -> bool:
        return self.decision == DecisionType.DEFER
