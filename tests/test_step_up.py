# tests/test_step_up.py
import httpx
import pytest
import respx
from thoth.models import DecisionType, ThothConfig
from thoth.step_up import StepUpClient


@pytest.fixture
def config():
    return ThothConfig(
        agent_id="test-agent",
        approved_scope=["read:data"],
        tenant_id="trantor",
        step_up_timeout_minutes=1,
        step_up_poll_interval_seconds=1,
    )


@respx.mock
def test_returns_allow_when_approved(config):
    respx.get("http://enforcer:8080/v1/enforce/hold/tok_123").mock(return_value=httpx.Response(200, json={"decision": "ALLOW"}))
    client = StepUpClient(config)
    decision = client.wait("tok_123")
    assert decision.is_allow


@respx.mock
def test_returns_block_when_rejected(config):
    respx.get("http://enforcer:8080/v1/enforce/hold/tok_456").mock(return_value=httpx.Response(200, json={"decision": "BLOCK", "reason": "rejected by approver"}))
    client = StepUpClient(config)
    decision = client.wait("tok_456")
    assert decision.is_block


@respx.mock
def test_returns_block_on_timeout():
    # Use 0 minute timeout so the loop immediately exits as timed-out
    timeout_config = ThothConfig(
        agent_id="test-agent",
        approved_scope=["read:data"],
        tenant_id="trantor",
        step_up_timeout_minutes=0,
        step_up_poll_interval_seconds=1,
    )
    respx.get("http://enforcer:8080/v1/enforce/hold/tok_789").mock(return_value=httpx.Response(200, json={"decision": "STEP_UP"}))
    client = StepUpClient(timeout_config)
    decision = client.wait("tok_789")
    assert decision.is_block  # timeout -> block
    assert "timeout" in (decision.reason or "").lower()
