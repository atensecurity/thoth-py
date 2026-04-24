from unittest.mock import patch

import thoth
from thoth import ThothClient


def test_thoth_client_is_exported() -> None:
    assert thoth.ThothClient is ThothClient


def test_wrap_delegates_to_instrument_with_defaults() -> None:
    client = ThothClient(
        agent_id="agent-1",
        approved_scope=["read:data"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )
    agent = object()
    with patch("thoth.client.instrument", return_value=agent) as mock_instrument:
        result = client.wrap(agent, user_id="alice@example.com")

    assert result is agent
    mock_instrument.assert_called_once_with(
        agent,
        agent_id="agent-1",
        approved_scope=["read:data"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
        user_id="alice@example.com",
    )


def test_wrap_openai_tools_delegates_to_instrument_openai() -> None:
    client = ThothClient(
        agent_id="agent-1",
        approved_scope=["search_docs"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )
    tools = {"search_docs": lambda _: "ok"}
    wrapped_tools = {"search_docs": lambda _: "wrapped"}
    with patch("thoth.client.instrument_openai", return_value=wrapped_tools) as mock_openai:
        result = client.wrap_openai_tools(tools, enforcement="block")

    assert result is wrapped_tools
    mock_openai.assert_called_once_with(
        tools,
        agent_id="agent-1",
        approved_scope=["search_docs"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
        enforcement="block",
    )


def test_wrap_claude_agent_sdk_delegates() -> None:
    client = ThothClient(
        agent_id="agent-1",
        approved_scope=["Read"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )
    options = object()
    wrapped_options = object()
    with patch("thoth.client.instrument_claude_agent_sdk", return_value=wrapped_options) as mock_claude:
        result = client.wrap_claude_agent_sdk(options, enforcement="block")

    assert result is wrapped_options
    mock_claude.assert_called_once_with(
        options,
        agent_id="agent-1",
        approved_scope=["Read"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
        enforcement="block",
    )
