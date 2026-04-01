# tests/test_enforcer_client.py
import httpx
import pytest
import respx
from thoth.enforcer_client import EnforcerClient
from thoth.models import DecisionType, EnforcementDecision, EnforcementMode, ThothConfig


@pytest.fixture
def config():
    return ThothConfig(
        agent_id="test-agent",
        approved_scope=["read:data"],
        tenant_id="trantor",
    )


@respx.mock
def test_returns_allow_decision(config):
    respx.post("http://enforcer:8080/v1/enforce").mock(return_value=httpx.Response(200, json={"decision": "ALLOW"}))
    client = EnforcerClient(config)
    decision = client.check("read:data", session_id="sess_1", tool_calls=[])
    assert decision.is_allow


@respx.mock
def test_returns_block_decision(config):
    respx.post("http://enforcer:8080/v1/enforce").mock(
        return_value=httpx.Response(
            200,
            json={
                "decision": "BLOCK",
                "reason": "out of scope",
                "violation_id": "vio_abc",
            },
        )
    )
    client = EnforcerClient(config)
    decision = client.check("write:s3", session_id="sess_1", tool_calls=[])
    assert decision.is_block
    assert decision.reason == "out of scope"


@respx.mock
def test_falls_back_to_allow_on_http_error(config):
    """Non-fatal: enforcer unreachable -> allow (observe fallback)."""
    respx.post("http://enforcer:8080/v1/enforce").mock(side_effect=httpx.ConnectError("refused"))
    client = EnforcerClient(config)
    decision = client.check("read:data", session_id="sess_1", tool_calls=[])
    assert decision.is_allow  # fallback, not an exception


@respx.mock
def test_falls_back_to_allow_on_timeout(config):
    respx.post("http://enforcer:8080/v1/enforce").mock(side_effect=httpx.TimeoutException("timeout"))
    client = EnforcerClient(config)
    decision = client.check("read:data", session_id="sess_1", tool_calls=[])
    assert decision.is_allow
