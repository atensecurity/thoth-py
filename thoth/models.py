# thoth/models.py
from __future__ import annotations

from datetime import datetime, timezone
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
UTC = timezone.utc


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
    source_type: SourceType
    event_type: EventType
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    approved_scope: list[str] = Field(default_factory=list)
    enforcement_mode: EnforcementMode = EnforcementMode.PROGRESSIVE
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
    enforcement: EnforcementMode = EnforcementMode.PROGRESSIVE
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
    authorization_decision: str | None = None
    decision_reason_code: str | None = None
    action_classification: str | None = None
    reason: str | None = None
    violation_id: str | None = None
    hold_token: str | None = None
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
