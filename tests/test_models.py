# tests/test_models.py
from datetime import datetime, timezone

import pytest
from thoth.models import BehavioralEvent, DecisionType, EnforcementDecision, EnforcementMode, EventType, SourceType, ThothConfig


def test_behavioral_event_requires_mandatory_fields():
    with pytest.raises(Exception):
        BehavioralEvent()  # missing required fields


def test_behavioral_event_construction():
    event = BehavioralEvent(
        event_id="evt_123",
        tenant_id="trantor",
        session_id="sess_abc",
        user_id="user_xyz",
        agent_id="invoice-processor",
        source_type=SourceType.AGENT_TOOL_CALL,
        event_type=EventType.TOOL_CALL_PRE,
        content="read:invoices",
        approved_scope=["read:invoices", "write:slack"],
        enforcement_mode=EnforcementMode.PROGRESSIVE,
        session_tool_calls=["read:invoices"],
        occurred_at=datetime.now(timezone.utc),
    )
    assert event.event_id == "evt_123"
    assert event.source_type == SourceType.AGENT_TOOL_CALL
    assert event.ttl > 0  # auto-set to 90 days


def test_thoth_config_defaults():
    config = ThothConfig(
        agent_id="my-agent",
        approved_scope=["read:data"],
        tenant_id="trantor",
    )
    assert config.enforcement == EnforcementMode.PROGRESSIVE
    with pytest.raises(ValueError, match="Thoth API URL is required"):
        _ = config.resolved_enforcer_url


def test_thoth_config_resolved_enforcer_url_matches_api_url():
    config = ThothConfig(
        agent_id="my-agent",
        approved_scope=["read:data"],
        tenant_id="trantor",
        api_key="thoth_live_key",
        api_url="https://enforce.trantor.atensecurity.com",
    )
    assert config.resolved_api_url == "https://enforce.trantor.atensecurity.com"
    assert config.resolved_enforcer_url == config.resolved_api_url


def test_thoth_config_env_api_url_overrides_field(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("THOTH_API_URL", "https://enforce.env.atensecurity.com")
    config = ThothConfig(
        agent_id="my-agent",
        approved_scope=["read:data"],
        tenant_id="trantor",
        api_url="https://enforce.field.atensecurity.com",
    )
    assert config.resolved_enforcer_url == "https://enforce.env.atensecurity.com"


def test_enforcement_decision_allow():
    decision = EnforcementDecision(decision=DecisionType.ALLOW)
    assert decision.is_allow
    assert not decision.is_block
    assert not decision.is_step_up
    assert not decision.is_modify
    assert not decision.is_defer


def test_enforcement_decision_block():
    decision = EnforcementDecision(
        decision=DecisionType.BLOCK,
        reason="Tool not in approved scope",
        violation_id="vio_abc",
    )
    assert decision.is_block
    assert decision.reason == "Tool not in approved scope"


def test_enforcement_decision_normalizes_alias_and_payload():
    decision = EnforcementDecision(
        authorization_decision="modify",
        modification_reason="path normalized",
        modified_tool_args={"path": "/tmp/safe.txt"},
    )
    assert decision.decision == DecisionType.MODIFY
    assert decision.is_modify
    assert decision.reason == "path normalized"
    assert decision.modified_tool_args == {"path": "/tmp/safe.txt"}


def test_enforcement_decision_parses_reason_code_and_action_classification_aliases():
    decision = EnforcementDecision.model_validate(
        {
            "authorization_decision": "BLOCK",
            "decisionReasonCode": "policy_scope_violation",
            "actionClassification": "write",
            "reason": "tool not in approved scope",
            "violation_id": "vio_123",
        }
    )
    assert decision.is_block
    assert decision.decision_reason_code == "policy_scope_violation"
    assert decision.action_classification == "write"
