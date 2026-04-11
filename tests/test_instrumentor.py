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
