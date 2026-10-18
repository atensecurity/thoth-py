"""Setup metadata remains advisory and does not modify tool behavior."""

from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel
import pytest

from thoth.tool_inventory import _publish, publish_tool_inventory


def test_instrumentation_metadata_published_without_classes():
    tool = SimpleNamespace(name="orders.delete", description="Delete an order", args_schema={"type": "object"})
    agent = SimpleNamespace(tools=[tool])
    with patch("thoth.tool_inventory.httpx.Client") as client:
        request = client.return_value.__enter__.return_value.post
        _publish(agent, "tenant", "agent", "https://api.test", "runtime-key")
        args, kwargs = request.call_args
        assert args[0] == "https://api.test/tenant/thoth/tool-inventory"
        assert kwargs["json"]["tools"] == [{"name": "orders.delete", "description": "Delete an order", "input_schema": {"type": "object"}}]
        assert kwargs["json"]["source"] == "python_sdk"
        assert agent.tools[0] is tool


def test_pydantic_tool_schema_is_published():
    class Arguments(BaseModel):
        url: str

    agent = SimpleNamespace(tools=[SimpleNamespace(name="fetch", args_schema=Arguments)])
    with patch("thoth.tool_inventory.httpx.Client") as client:
        _publish(agent, "tenant", "agent", "https://api.test", None)
        tool = client.return_value.__enter__.return_value.post.call_args.kwargs["json"]["tools"][0]
        assert tool["input_schema"]["properties"]["url"]["type"] == "string"


@pytest.mark.parametrize("error", [TypeError, ValueError, RuntimeError])
def test_broken_schema_does_not_prevent_instrumentation(error):
    def broken_schema():
        raise error("schema unavailable")

    tool = SimpleNamespace(name="opaque", args_schema=SimpleNamespace(model_json_schema=broken_schema))
    agent = SimpleNamespace(tools=[tool])
    with patch("thoth.tool_inventory.httpx.Client") as client:
        _publish(agent, "tenant", "agent", "https://api.test", None)
        metadata = client.return_value.__enter__.return_value.post.call_args.kwargs["json"]["tools"][0]
        assert metadata["input_schema"] == {}
        assert agent.tools[0] is tool


def test_inventory_is_deferred_and_can_be_disabled(monkeypatch):
    with patch("thoth.tool_inventory.threading.Thread") as thread:
        publish_tool_inventory(SimpleNamespace(tools=[]), tenant_id="t", agent_id="a", api_url="https://api.test", api_key=None)
        assert thread.call_args.kwargs["daemon"] is True
        thread.return_value.start.assert_called_once()
        monkeypatch.setenv("THOTH_TOOL_INVENTORY", "off")
        thread.reset_mock()
        publish_tool_inventory(SimpleNamespace(tools=[]), tenant_id="t", agent_id="a", api_url="https://api.test", api_key=None)
        thread.assert_not_called()


def test_non_json_schema_and_nested_values_are_safe():
    agent = SimpleNamespace(
        tools=[
            SimpleNamespace(name="bad", args_schema={"value": {1, 2}}),
            SimpleNamespace(name="good", args_schema={"properties": {"url": {"type": "string", "default": "token", "enum": ["secret"]}}}),
        ]
    )
    with patch("thoth.tool_inventory.httpx.Client") as client:
        _publish(agent, "t/slash", "a", "https://api.test", None)
        args, kwargs = client.return_value.__enter__.return_value.post.call_args
        assert "/t%2Fslash/" in args[0]
        assert kwargs["json"]["tools"][0]["input_schema"] == {}
        assert kwargs["json"]["tools"][1]["input_schema"]["properties"]["url"] == {"type": "string"}


def test_schema_parameter_named_default_and_broken_name():
    class BrokenName:
        @property
        def name(self):
            raise RuntimeError("name unavailable")

    agent = SimpleNamespace(tools=[SimpleNamespace(name="good", args_schema={"properties": {"default": {"type": "string", "default": "secret"}}}), BrokenName()])
    with patch("thoth.tool_inventory.httpx.Client") as client:
        _publish(agent, "t", "a", "https://api.test", None)
        tools = client.return_value.__enter__.return_value.post.call_args.kwargs["json"]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "good"
        assert tools[0]["input_schema"]["properties"]["default"] == {"type": "string"}
