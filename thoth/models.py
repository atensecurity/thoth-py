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


_TTL_90_DAYS = 90 * 24 * 60 * 60


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
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ttl: int = Field(default=0)
    violation_id: str | None = None

    @model_validator(mode="after")
    def set_ttl(self) -> BehavioralEvent:
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
    api_url: str | None = None
    step_up_timeout_minutes: int = 15
    step_up_poll_interval_seconds: int = 5
    # Declare the purpose of this session (HIPAA minimum-necessary).
    # When a compliance pack defines session_scopes, tools outside the declared
    # intent are step-up-challenged even if they appear in the approved scope.
    session_intent: str | None = None

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


class EnforcementDecision(BaseModel):
    decision: DecisionType
    reason: str | None = None
    violation_id: str | None = None
    hold_token: str | None = None

    @property
    def is_allow(self) -> bool:
        return self.decision == DecisionType.ALLOW

    @property
    def is_block(self) -> bool:
        return self.decision == DecisionType.BLOCK

    @property
    def is_step_up(self) -> bool:
        return self.decision == DecisionType.STEP_UP
