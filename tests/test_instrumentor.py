# tests/test_instrumentor.py
from unittest.mock import patch

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


class NestedToolchain:
    def fetch(self, payload):
        return self.parse(payload)

    def parse(self, payload):
        return payload.get("id")

    def _hidden(self, payload):
        return payload.get("id")


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
    with patch("thoth.instrumentor.EnforcerClient"), patch("thoth.instrumentor.Tracer"), patch("thoth.instrumentor.HttpEmitter") as mock_emitter:
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


def test_instrument_toolchain_wraps_nested_methods():
    toolchain = {"datadog": NestedToolchain()}
    with patch("thoth.instrumentor.EnforcerClient"), patch("thoth.instrumentor.HttpEmitter"):
        governed = thoth.instrument_toolchain(
            toolchain,
            agent_id="my-agent",
            approved_scope=["datadog.fetch", "datadog.parse"],
            tenant_id="trantor",
            enforcement="observe",
            api_url="https://enforcer.example",
        )

    assert governed is toolchain
    result = governed["datadog"].fetch({"id": "sig-1"})
    assert result == "sig-1"
    session = thoth.get_current_session()
    assert session is not None
    assert "datadog.fetch" in session.tool_calls
    assert "datadog.parse" in session.tool_calls


def test_instrument_anthropic_wraps_nested_tool_dict():
    tool_fns = {
        "data_sources": {
            "cloudtrail": lambda payload: payload.get("signal_id"),
        }
    }
    with patch("thoth.instrumentor.EnforcerClient"), patch("thoth.instrumentor.HttpEmitter"):
        governed = thoth.instrument_anthropic(
            tool_fns,
            agent_id="my-agent",
            approved_scope=["data_sources.cloudtrail"],
            tenant_id="trantor",
            enforcement="observe",
            api_url="https://enforcer.example",
        )

    assert isinstance(governed["data_sources"], dict)
    result = governed["data_sources"]["cloudtrail"]({"signal_id": "sig-1"})
    assert result == "sig-1"
    session = thoth.get_current_session()
    assert session is not None
    assert "data_sources.cloudtrail" in session.tool_calls


def test_instrument_anthropic_auto_depth_wraps_deep_nested_tool():
    tool_fns: dict[str, object] = {}
    cursor: dict[str, object] = tool_fns
    path_parts = [f"level{i}" for i in range(12)]
    for part in path_parts:
        child: dict[str, object] = {}
        cursor[part] = child
        cursor = child
    cursor["cloudtrail"] = lambda payload: payload.get("signal_id")

    tool_name = ".".join(path_parts + ["cloudtrail"])
    with patch("thoth.instrumentor.EnforcerClient"), patch("thoth.instrumentor.HttpEmitter"):
        governed = thoth.instrument_anthropic(
            tool_fns,
            agent_id="my-agent",
            approved_scope=[tool_name],
            tenant_id="trantor",
            enforcement="observe",
            api_url="https://enforcer.example",
        )

    node = governed
    for part in path_parts:
        node = node[part]
    result = node["cloudtrail"]({"signal_id": "sig-1"})
    assert result == "sig-1"
    session = thoth.get_current_session()
    assert session is not None
    assert tool_name in session.tool_calls


def test_toolchain_function_map_collects_public_callables():
    toolchain = {"datadog": NestedToolchain()}
    function_map = thoth.toolchain_function_map(toolchain)

    assert "datadog.fetch" in function_map
    assert "datadog.parse" in function_map
    assert "datadog._hidden" not in function_map
    assert callable(function_map["datadog.fetch"])


def test_toolchain_function_map_includes_private_when_enabled():
    toolchain = {"datadog": NestedToolchain()}
    function_map = thoth.toolchain_function_map(toolchain, include_private=True)

    assert "datadog._hidden" in function_map


def test_toolchain_function_map_respects_max_depth():
    tool_fns = {"level1": {"level2": {"cloudtrail": lambda payload: payload.get("signal_id")}}}
    function_map = thoth.toolchain_function_map(tool_fns, max_depth=1)
    assert "level1.level2.cloudtrail" not in function_map
