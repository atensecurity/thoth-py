# tests/test_instrumentor.py
from unittest.mock import MagicMock, patch

import pytest
import thoth
from thoth import ThothPolicyViolation
from thoth.models import DecisionType, EnforcementDecision, EnforcementMode, ThothConfig


class FakeTool:
    name = "read:data"

    def run(self, x):
        return f"result:{x}"


class FakeAgent:
    def __init__(self, tools):
        self.tools = tools


def test_instrument_returns_agent():
    agent = FakeAgent(tools=[FakeTool()])
    result = thoth.instrument(
        agent,
        agent_id="my-agent",
        approved_scope=["read:data"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )
    assert result is agent  # returns same object, mutated


def test_instrument_wraps_tools():
    agent = FakeAgent(tools=[FakeTool()])
    with patch("thoth.instrumentor.EnforcerClient") as MockEnforcer, patch("thoth.instrumentor.HttpEmitter"):
        MockEnforcer.return_value.check.return_value = EnforcementDecision(decision=DecisionType.ALLOW)
        thoth.instrument(
            agent,
            agent_id="my-agent",
            approved_scope=["read:data"],
            tenant_id="trantor",
            api_url="https://enforcer.example",
        )
        # Tool should still be callable
        result = agent.tools[0].run("test")
        assert result == "result:test"


def test_instrument_raises_on_block():
    agent = FakeAgent(tools=[FakeTool()])
    with patch("thoth.instrumentor.EnforcerClient") as MockEnforcer, patch("thoth.instrumentor.HttpEmitter"):
        MockEnforcer.return_value.check.return_value = EnforcementDecision(
            decision=DecisionType.BLOCK,
            reason="blocked",
            violation_id="vio_001",
        )
        thoth.instrument(
            agent,
            agent_id="my-agent",
            approved_scope=[],  # nothing allowed
            tenant_id="trantor",
            enforcement="block",
            api_url="https://enforcer.example",
        )
        with pytest.raises(ThothPolicyViolation):
            agent.tools[0].run("test")


def test_instrument_claude_agent_sdk_delegates():
    options = object()
    wrapped_options = object()
    with (
        patch("thoth.instrumentor.EnforcerClient") as MockEnforcer,
        patch("thoth.instrumentor.HttpEmitter"),
        patch(
            "thoth.integrations.claude_agent_sdk.instrument_claude_agent_sdk_options",
            return_value=wrapped_options,
        ) as mock_integration,
    ):
        MockEnforcer.return_value.check.return_value = EnforcementDecision(decision=DecisionType.ALLOW)
        result = thoth.instrument_claude_agent_sdk(
            options,
            agent_id="my-agent",
            approved_scope=["Read"],
            tenant_id="trantor",
            api_url="https://enforcer.example",
        )

    assert result is wrapped_options
    mock_integration.assert_called_once()


def test_instrument_uses_thoth_environment_env_var(monkeypatch: pytest.MonkeyPatch):
    agent = FakeAgent(tools=[FakeTool()])
    monkeypatch.setenv("THOTH_ENVIRONMENT", "dev")
    with patch("thoth.instrumentor.EnforcerClient"), patch("thoth.instrumentor.HttpEmitter"), patch("thoth.instrumentor.Tracer") as mock_tracer:
        thoth.instrument(
            agent,
            agent_id="my-agent",
            approved_scope=["read:data"],
            tenant_id="trantor",
            api_url="https://enforcer.example",
        )

    assert mock_tracer.call_count == 1
    config = mock_tracer.call_args.kwargs["config"]
    assert config.environment == "dev"


def test_instrument_uses_event_ingest_token_env_var(monkeypatch: pytest.MonkeyPatch):
    agent = FakeAgent(tools=[FakeTool()])
    monkeypatch.setenv("THOTH_EVENT_INGEST_TOKEN", "ingest-token-123")
    with patch("thoth.instrumentor.EnforcerClient"), patch("thoth.instrumentor.Tracer"), patch(
        "thoth.instrumentor.HttpEmitter"
    ) as mock_emitter:
        thoth.instrument(
            agent,
            agent_id="my-agent",
            approved_scope=["read:data"],
            tenant_id="trantor",
            api_url="https://enforcer.example",
        )

    assert mock_emitter.call_count == 1
    call = mock_emitter.call_args
    assert call.kwargs["event_ingest_token"] == "ingest-token-123"
