from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from thoth.integrations.claude_agent_sdk import instrument_claude_agent_sdk_options
from thoth.models import DecisionType, EnforcementDecision, EnforcementMode, EventType, ThothConfig
from thoth.session import SessionContext
from thoth.tracer import Tracer


@dataclass
class FakePermissionResultAllow:
    behavior: str = "allow"
    updated_input: dict[str, Any] | None = None
    updated_permissions: list[Any] | None = None


@dataclass
class FakePermissionResultDeny:
    behavior: str = "deny"
    message: str = ""
    interrupt: bool = False


@dataclass
class FakeHookMatcher:
    matcher: str | None = None
    hooks: list[Any] = field(default_factory=list)


class FakeClaudeAgentOptions:
    def __init__(self) -> None:
        self.can_use_tool: Any = None
        self.hooks: dict[str, list[FakeHookMatcher]] | None = None


def _make_tracer() -> Tracer:
    config = ThothConfig(
        agent_id="test-agent",
        approved_scope=["Read", "Write"],
        tenant_id="trantor",
        enforcement=EnforcementMode.PROGRESSIVE,
        api_url="https://enforcer.example",
    )
    session = SessionContext(config, session_id="sess_123")
    emitter = MagicMock()
    enforcer = MagicMock()
    step_up = MagicMock()
    return Tracer(config=config, session=session, emitter=emitter, enforcer=enforcer, step_up=step_up)


def _fake_sdk_types() -> dict[str, type[Any]]:
    return {
        "ClaudeAgentOptions": FakeClaudeAgentOptions,
        "PermissionResultAllow": FakePermissionResultAllow,
        "PermissionResultDeny": FakePermissionResultDeny,
        "HookMatcher": FakeHookMatcher,
    }


@pytest.mark.asyncio
async def test_instruments_options_and_allows_tool() -> None:
    tracer = _make_tracer()
    tracer._enforcer.acheck = AsyncMock(return_value=EnforcementDecision(decision=DecisionType.ALLOW))
    with patch("thoth.integrations.claude_agent_sdk._load_claude_agent_sdk_types", return_value=_fake_sdk_types()):
        options = instrument_claude_agent_sdk_options(FakeClaudeAgentOptions(), tracer)

    assert callable(options.can_use_tool)
    assert options.hooks is not None
    assert "PostToolUse" in options.hooks
    assert "PostToolUseFailure" in options.hooks

    result = await options.can_use_tool("Read", {"path": "/tmp/a.txt"}, object())
    assert isinstance(result, FakePermissionResultAllow)
    assert result.updated_input == {"path": "/tmp/a.txt"}
    assert tracer._session.tool_calls == ["Read"]

    events = [call.args[0] for call in tracer._emitter.emit.call_args_list]
    assert any(event.event_type == EventType.LLM_INVOCATION for event in events)
    pre_event = next(event for event in events if event.event_type == EventType.TOOL_CALL_PRE)
    assert pre_event.event_type == EventType.TOOL_CALL_PRE
    assert pre_event.tool_name == "Read"


@pytest.mark.asyncio
async def test_denies_blocked_tool_with_policy_reason() -> None:
    tracer = _make_tracer()
    tracer._enforcer.acheck = AsyncMock(
        return_value=EnforcementDecision(
            decision=DecisionType.BLOCK,
            reason="tool not allowed",
            violation_id="vio_123",
        )
    )
    with patch("thoth.integrations.claude_agent_sdk._load_claude_agent_sdk_types", return_value=_fake_sdk_types()):
        options = instrument_claude_agent_sdk_options(FakeClaudeAgentOptions(), tracer)

    result = await options.can_use_tool("Bash", {"command": "rm -rf /"}, object())
    assert isinstance(result, FakePermissionResultDeny)
    assert result.message == "tool not allowed"
    assert tracer._session.tool_calls == []
    block_event = tracer._emitter.emit.call_args_list[-1].args[0]
    assert block_event.event_type == EventType.TOOL_CALL_BLOCK
    assert block_event.violation_id == "vio_123"


@pytest.mark.asyncio
async def test_chains_existing_can_use_tool_with_modified_input() -> None:
    tracer = _make_tracer()
    tracer._enforcer.acheck = AsyncMock(
        return_value=EnforcementDecision(
            decision=DecisionType.MODIFY,
            modified_tool_args={"input": {"path": "/tmp/safe.txt"}},
            reason="sanitized",
        )
    )
    seen: dict[str, Any] = {}

    async def existing_callback(tool_name: str, tool_input: dict[str, Any], context: Any) -> FakePermissionResultAllow:
        seen["tool_name"] = tool_name
        seen["tool_input"] = dict(tool_input)
        seen["context"] = context
        return FakePermissionResultAllow()

    options = FakeClaudeAgentOptions()
    options.can_use_tool = existing_callback
    with patch("thoth.integrations.claude_agent_sdk._load_claude_agent_sdk_types", return_value=_fake_sdk_types()):
        instrument_claude_agent_sdk_options(options, tracer)

    result = await options.can_use_tool("Read", {"path": "/tmp/unsafe.txt"}, object())
    assert isinstance(result, FakePermissionResultAllow)
    assert result.updated_input == {"path": "/tmp/safe.txt"}
    assert seen["tool_name"] == "Read"
    assert seen["tool_input"] == {"path": "/tmp/safe.txt"}
    assert tracer._session.tool_calls == ["Read"]


@pytest.mark.asyncio
async def test_post_tool_hooks_emit_events() -> None:
    tracer = _make_tracer()
    tracer._enforcer.acheck = AsyncMock(return_value=EnforcementDecision(decision=DecisionType.ALLOW))
    with patch("thoth.integrations.claude_agent_sdk._load_claude_agent_sdk_types", return_value=_fake_sdk_types()):
        options = instrument_claude_agent_sdk_options(FakeClaudeAgentOptions(), tracer)

    post_hook = options.hooks["PostToolUse"][0].hooks[0]
    failure_hook = options.hooks["PostToolUseFailure"][0].hooks[0]
    await post_hook({"tool_name": "Read", "tool_response": {"ok": True}}, None, {})
    await failure_hook({"tool_name": "Bash", "error": "command failed"}, None, {})

    assert tracer._emitter.emit.call_args_list[-2].args[0].event_type == EventType.TOOL_CALL_POST
    assert tracer._emitter.emit.call_args_list[-1].args[0].event_type == EventType.TOOL_CALL_BLOCK
