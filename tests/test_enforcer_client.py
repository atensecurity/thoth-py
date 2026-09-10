# tests/test_enforcer_client.py
import json

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
def test_fail_open_allows_on_timeout():
    config = ThothConfig(
        agent_id="test-agent",
        approved_scope=["read:data"],
        tenant_id="trantor",
        api_url="http://enforcer:8080",
        fail_open=True,
    )
    respx.post(f"{config.resolved_enforcer_url}/v1/enforce").mock(side_effect=httpx.TimeoutException("timeout"))
    client = EnforcerClient(config)
    decision = client.check("read:data", session_id="sess_1", tool_calls=[])
    assert decision.is_allow
    assert "fail-open" in (decision.reason or "").lower()


@respx.mock
def test_fail_open_allows_on_retryable_status():
    config = ThothConfig(
        agent_id="test-agent",
        approved_scope=["read:data"],
        tenant_id="trantor",
        api_url="http://enforcer:8080",
        fail_open=True,
    )
    respx.post(f"{config.resolved_enforcer_url}/v1/enforce").mock(return_value=httpx.Response(500, json={"error": "boom"}))
    client = EnforcerClient(config)
    decision = client.check("read:data", session_id="sess_1", tool_calls=[])
    assert decision.is_allow
    assert "status=500" in (decision.reason or "")


@respx.mock
def test_fail_open_still_blocks_on_auth_failure():
    config = ThothConfig(
        agent_id="test-agent",
        approved_scope=["read:data"],
        tenant_id="trantor",
        api_url="http://enforcer:8080",
        fail_open=True,
    )
    respx.post(f"{config.resolved_enforcer_url}/v1/enforce").mock(return_value=httpx.Response(403, json={"error": "forbidden"}))
    client = EnforcerClient(config)
    decision = client.check("read:data", session_id="sess_1", tool_calls=[])
    assert decision.is_block
    assert "status=403" in (decision.reason or "")


@respx.mock
def test_blocks_with_http_status_context(config):
    respx.post(f"{config.resolved_enforcer_url}/v1/enforce").mock(
        return_value=httpx.Response(
            403,
            json={"error": "forbidden", "message": "api key expired"},
        )
    )
    client = EnforcerClient(config)
    decision = client.check("read:data", session_id="sess_1", tool_calls=[])
    assert decision.is_block
    assert "status=403" in (decision.reason or "")


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
    assert payload["action_attestation_id"]
    assert payload["identity_binding"]["agent_id"] == "test-agent"
    assert payload["identity_binding"]["tenant_id"] == "trantor"
    assert payload["identity_binding"]["user_id"] == "system"


@respx.mock
def test_sends_model_and_mcp_context_payload(config):
    captured: dict = {}

    config.model_name = "gpt-5.6"
    config.model_provider = "openai"
    config.model_artifact_id = "artifact-prod-a"
    config.model_artifact_version = "2026.08.19"
    config.auth_context = {"principal": "svc://thoth-runtime"}
    config.delegation_context = {"task_id": "task-42"}
    config.mcp_runtime_identity = "mcp-runtime-prod"
    config.request_metadata = {"source": "sdk-test"}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        return httpx.Response(200, json={"decision": "ALLOW"})

    respx.post(f"{config.resolved_enforcer_url}/v1/enforce").mock(side_effect=_handler)
    client = EnforcerClient(config)
    decision = client.check("read:data", session_id="sess_1", tool_calls=[])
    assert decision.is_allow

    payload = json.loads(captured["body"])
    assert payload["model_name"] == "gpt-5.6"
    assert payload["model_provider"] == "openai"
    assert payload["model_artifact_id"] == "artifact-prod-a"
    assert payload["model_artifact_version"] == "2026.08.19"
    assert payload["auth_context"]["principal"] == "svc://thoth-runtime"
    assert payload["auth_context"]["service_identity"] == "mcp-runtime-prod"
    assert payload["delegation_context"]["task_id"] == "task-42"
    assert payload["metadata"]["mcp_runtime_identity"] == "mcp-runtime-prod"
    assert payload["metadata"]["source"] == "sdk-test"


@respx.mock
def test_sends_custom_environment_and_trace_id(config):
    captured: dict = {}

    config.environment = "dev"
    config.enforcement_trace_id = "trace_abc_123"
    config.action_attestation_id = "attest_abc_123"

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
    assert payload["action_attestation_id"] == "attest_abc_123"


@respx.mock
def test_explain_returns_human_explanation(config):
    respx.post(f"{config.resolved_enforcer_url}/v1/explain").mock(
        return_value=httpx.Response(
            200,
            json={
                "violation_id": "vio_1",
                "agent_id": "test-agent",
                "user_id": "system",
                "tool_name": "write:s3",
                "what_happened": "Your agent attempted an action.",
                "why_it_was_blocked": "Policy denied the action.",
                "what_to_do_next": "Contact #secops.",
                "business_impact": "Potential data loss.",
                "request_url": "",
                "severity": "high",
            },
        )
    )
    client = EnforcerClient(config)
    explanation = client.explain(
        EnforcementDecision(decision=DecisionType.BLOCK, violation_id="vio_1"),
        tool_name="write:s3",
        session_id="sess_1",
        tool_calls=["write:s3"],
        tool_args={"bucket": "prod"},
    )
    assert explanation is not None
    assert explanation.violation_id == "vio_1"
    assert "Contact #secops" in explanation.what_to_do_next


@respx.mock
def test_explain_returns_none_on_failure(config):
    respx.post(f"{config.resolved_enforcer_url}/v1/explain").mock(side_effect=httpx.ConnectError("boom"))
    client = EnforcerClient(config)
    explanation = client.explain(
        EnforcementDecision(decision=DecisionType.BLOCK, violation_id="vio_1"),
        tool_name="write:s3",
        session_id="sess_1",
        tool_calls=["write:s3"],
    )
    assert explanation is None
