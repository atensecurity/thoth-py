# tests/test_step_up.py
import httpx
import pytest
import respx

from thoth.models import ThothConfig
from thoth.step_up import StepUpClient


@pytest.fixture
def config():
    return ThothConfig(
        agent_id="test-agent",
        approved_scope=["read:data"],
        tenant_id="trantor",
        api_url="http://enforcer:8080",
        step_up_timeout_minutes=1,
        step_up_poll_interval_seconds=1,
    )


@respx.mock
def test_returns_allow_when_approved(config):
    respx.get(f"{config.resolved_enforcer_url}/v1/enforce/hold/tok_123").mock(return_value=httpx.Response(200, json={"resolved": True, "resolution": "ALLOW"}))
    client = StepUpClient(config)
    decision = client.wait("tok_123")
    assert decision.is_allow


@respx.mock
def test_returns_block_when_rejected(config):
    respx.get(f"{config.resolved_enforcer_url}/v1/enforce/hold/tok_456").mock(
        return_value=httpx.Response(200, json={"resolved": True, "resolution": "BLOCK", "reason": "rejected by approver"})
    )
    client = StepUpClient(config)
    decision = client.wait("tok_456")
    assert decision.is_block


@respx.mock
def test_returns_block_when_rejected_with_deny_alias(config):
    respx.get(f"{config.resolved_enforcer_url}/v1/enforce/hold/tok_deny").mock(
        return_value=httpx.Response(
            200,
            json={
                "resolved": True,
                "resolution": "DENY",
                "reason": "reviewer denied",
                "decision_reason_code": "policy_scope_violation",
                "action_classification": "write",
            },
        )
    )
    client = StepUpClient(config)
    decision = client.wait("tok_deny")
    assert decision.is_block
    assert decision.decision_reason_code == "policy_scope_violation"
    assert decision.action_classification == "write"


@respx.mock
def test_returns_block_on_timeout():
    # Use 0 minute timeout so the loop immediately exits as timed-out
    timeout_config = ThothConfig(
        agent_id="test-agent",
        approved_scope=["read:data"],
        tenant_id="trantor",
        api_url="http://enforcer:8080",
        step_up_timeout_minutes=0,
        step_up_poll_interval_seconds=1,
    )
    respx.get(f"{timeout_config.resolved_enforcer_url}/v1/enforce/hold/tok_789").mock(return_value=httpx.Response(200, json={"resolved": False, "resolution": None}))
    client = StepUpClient(timeout_config)
    decision = client.wait("tok_789")
    assert decision.is_block  # timeout -> block
    assert "timeout" in (decision.reason or "").lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_async", [False, True])
@pytest.mark.parametrize("second_async", [False, True])
async def test_timeout_evidence_is_isolated_between_actions(config, first_async, second_async):
    """Tracer annotates timeout decisions; later actions must receive fresh state."""
    client = StepUpClient(config.model_copy(update={"step_up_timeout_minutes": 0}))
    try:
        first = await client.await_decision("first") if first_async else client.wait("first")
        second = await client.await_decision("second") if second_async else client.wait("second")
        assert first is not second
        first.action_attestation_id = "first-action"
        first.enforcement_trace_id = "first-trace"
        first.receipt = {"receipt_id": "first-receipt"}
        assert second.is_block
        assert second.action_attestation_id is None
        assert second.enforcement_trace_id is None
        assert second.receipt is None
    finally:
        client.close()
        await client.aclose()
