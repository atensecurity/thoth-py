# tests/test_enforcer_client.py
import httpx
import json
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
        api_url="http://enforcer:8080",
    )


@respx.mock
def test_returns_allow_decision(config):
    respx.post(f"{config.resolved_enforcer_url}/v1/enforce").mock(return_value=httpx.Response(200, json={"decision": "ALLOW"}))
    client = EnforcerClient(config)
    decision = client.check("read:data", session_id="sess_1", tool_calls=[])
    assert decision.is_allow


@respx.mock
def test_returns_block_decision(config):
    respx.post(f"{config.resolved_enforcer_url}/v1/enforce").mock(
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
def test_falls_back_to_block_on_http_error(config):
    """Fail-closed: enforcer unreachable -> block."""
    respx.post(f"{config.resolved_enforcer_url}/v1/enforce").mock(side_effect=httpx.ConnectError("refused"))
    client = EnforcerClient(config)
    decision = client.check("read:data", session_id="sess_1", tool_calls=[])
    assert decision.is_block
    assert "unavailable" in (decision.reason or "").lower()


@respx.mock
def test_falls_back_to_block_on_timeout(config):
    respx.post(f"{config.resolved_enforcer_url}/v1/enforce").mock(side_effect=httpx.TimeoutException("timeout"))
    client = EnforcerClient(config)
    decision = client.check("read:data", session_id="sess_1", tool_calls=[])
    assert decision.is_block
    assert "unavailable" in (decision.reason or "").lower()


@respx.mock
def test_sends_tool_args_payload(config):
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        return httpx.Response(200, json={"decision": "ALLOW"})

    respx.post(f"{config.resolved_enforcer_url}/v1/enforce").mock(side_effect=_handler)
    client = EnforcerClient(config)
    decision = client.check(
        "read:data",
        session_id="sess_1",
        tool_calls=["read:data"],
        tool_args={"path": "/tmp/patient.txt", "recursive": False},
    )
    assert decision.is_allow
    payload = json.loads(captured["body"])
    assert payload["tool_args"] == {"path": "/tmp/patient.txt", "recursive": False}
    assert payload["environment"] == "prod"
    assert payload["enforcement_trace_id"] == "sess_1"


@respx.mock
def test_sends_custom_environment_and_trace_id(config):
    captured: dict = {}

    config.environment = "dev"
    config.enforcement_trace_id = "trace_abc_123"

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        return httpx.Response(200, json={"decision": "ALLOW"})

    respx.post(f"{config.resolved_enforcer_url}/v1/enforce").mock(side_effect=_handler)
    client = EnforcerClient(config)
    decision = client.check("read:data", session_id="sess_1", tool_calls=[])
    assert decision.is_allow
    payload = json.loads(captured["body"])
    assert payload["environment"] == "dev"
    assert payload["enforcement_trace_id"] == "trace_abc_123"
