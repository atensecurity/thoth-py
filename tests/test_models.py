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
    assert config.resolved_enforcer_url == "http://enforcer:8080"


def test_enforcement_decision_allow():
    decision = EnforcementDecision(decision=DecisionType.ALLOW)
    assert decision.is_allow
    assert not decision.is_block
    assert not decision.is_step_up


def test_enforcement_decision_block():
    decision = EnforcementDecision(
        decision=DecisionType.BLOCK,
        reason="Tool not in approved scope",
        violation_id="vio_abc",
    )
    assert decision.is_block
    assert decision.reason == "Tool not in approved scope"
