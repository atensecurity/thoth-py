from __future__ import annotations

import asyncio
import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from thoth.integrations.claude_agent_sdk import instrument_claude_agent_sdk_options
from thoth.models import DecisionType, EnforcementDecision, EnforcementMode, EventType, ThothConfig
from thoth.session import SessionContext
from thoth.telemetry import telemetry_event
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


@pytest.mark.asyncio
async def test_post_hook_retains_matching_authorization_once_without_sensitive_fields() -> None:
    secret = "SYNTHETIC-CLAUDE-DECISION-SECRET"
    tracer = _make_tracer()
    tracer._enforcer.acheck = AsyncMock(
        return_value=EnforcementDecision(
            decision=DecisionType.ALLOW,
            decision_reason_code="policy_scope_allow",
            decision_evidence={
                "policy": {"policy_id": "policy-safe-001"},
                "secret": secret,
            },
            receipt={"receipt_id": "receipt-safe-001", "secret": secret},
            reason=secret,
        )
    )
    with patch("thoth.integrations.claude_agent_sdk._load_claude_agent_sdk_types", return_value=_fake_sdk_types()):
        options = instrument_claude_agent_sdk_options(FakeClaudeAgentOptions(), tracer)

    context = SimpleNamespace(tool_use_id="toolu-safe-001")
    await options.can_use_tool("Read", {"path": secret}, context)
    post_hook = options.hooks["PostToolUse"][0].hooks[0]
    hook_input = {
        "tool_name": "Read",
        "tool_input": {"path": secret},
        "tool_response": {"secret": secret},
        "tool_use_id": "toolu-safe-001",
    }
    await post_hook(hook_input, None, {})

    post = tracer._emitter.emit.call_args_list[-1].args[0]
    retained = telemetry_event(post)["metadata"]
    assert retained["authorization_decision"] == "ALLOW"
    assert retained["decision_reason_code"] == "policy_scope_allow"
    assert retained["decision_evidence"]["policy"]["policy_id"] == "policy-safe-001"
    assert retained["receipt"]["receipt_id"] == "receipt-safe-001"
    assert secret not in json.dumps(retained, sort_keys=True)

    await post_hook(hook_input, None, {})
    repeated = telemetry_event(tracer._emitter.emit.call_args_list[-1].args[0])["metadata"]
    assert "authorization_decision" not in repeated
    assert "decision_reason_code" not in repeated


@pytest.mark.asyncio
async def test_concurrent_post_hooks_keep_decisions_scoped_to_provider_action_ids() -> None:
    tracer = _make_tracer()

    async def decide(**kwargs: Any) -> EnforcementDecision:
        action_id = kwargs["action_attestation_id"]
        return EnforcementDecision(
            decision=DecisionType.ALLOW,
            decision_reason_code=f"allow_{action_id}",
        )

    tracer._enforcer.acheck = AsyncMock(side_effect=decide)
    with patch("thoth.integrations.claude_agent_sdk._load_claude_agent_sdk_types", return_value=_fake_sdk_types()):
        options = instrument_claude_agent_sdk_options(FakeClaudeAgentOptions(), tracer)

    await asyncio.gather(
        options.can_use_tool("Read", {"path": "first"}, SimpleNamespace(tool_use_id="toolu-first")),
        options.can_use_tool("Read", {"path": "second"}, SimpleNamespace(tool_use_id="toolu-second")),
    )
    post_hook = options.hooks["PostToolUse"][0].hooks[0]
    for action_id in ("toolu-second", "toolu-first"):
        await post_hook(
            {
                "tool_name": "Read",
                "tool_input": {"path": action_id},
                "tool_response": {"ok": True},
                "tool_use_id": action_id,
            },
            None,
            {},
        )

    posts = [telemetry_event(call.args[0])["metadata"] for call in tracer._emitter.emit.call_args_list if call.args[0].event_type == EventType.TOOL_CALL_POST]
    assert [post["action_attestation_id"] for post in posts] == ["toolu-second", "toolu-first"]
    assert [post["decision_reason_code"] for post in posts] == ["allow_toolu-second", "allow_toolu-first"]


@pytest.mark.asyncio
async def test_configured_action_id_correlates_post_decision_without_provider_id() -> None:
    tracer = _make_tracer()
    tracer._config.action_attestation_id = "configured-action-001"
    tracer._enforcer.acheck = AsyncMock(
        return_value=EnforcementDecision(
            decision=DecisionType.ALLOW,
            decision_reason_code="configured_scope_allow",
        )
    )
    with patch("thoth.integrations.claude_agent_sdk._load_claude_agent_sdk_types", return_value=_fake_sdk_types()):
        options = instrument_claude_agent_sdk_options(FakeClaudeAgentOptions(), tracer)

    await options.can_use_tool("Read", {"path": "safe"}, object())
    post_hook = options.hooks["PostToolUse"][0].hooks[0]
    await post_hook(
        {"tool_name": "Read", "tool_input": {"path": "safe"}, "tool_response": {"ok": True}},
        None,
        {},
    )

    retained = telemetry_event(tracer._emitter.emit.call_args_list[-1].args[0])["metadata"]
    assert retained["action_attestation_id"] == "configured-action-001"
    assert retained["authorization_decision"] == "ALLOW"
    assert retained["decision_reason_code"] == "configured_scope_allow"


@pytest.mark.asyncio
async def test_duplicate_configured_action_id_does_not_cross_wire_post_decisions() -> None:
    tracer = _make_tracer()
    tracer._config.action_attestation_id = "configured-action-duplicate"
    reasons = iter(("first_allow", "second_allow"))
    tracer._enforcer.acheck = AsyncMock(
        side_effect=lambda **_: EnforcementDecision(
            decision=DecisionType.ALLOW,
            decision_reason_code=next(reasons),
        )
    )
    with patch("thoth.integrations.claude_agent_sdk._load_claude_agent_sdk_types", return_value=_fake_sdk_types()):
        options = instrument_claude_agent_sdk_options(FakeClaudeAgentOptions(), tracer)

    await options.can_use_tool("Read", {"path": "first"}, object())
    await options.can_use_tool("Read", {"path": "second"}, object())
    post_hook = options.hooks["PostToolUse"][0].hooks[0]
    await post_hook({"tool_name": "Read"}, None, {})

    retained = telemetry_event(tracer._emitter.emit.call_args_list[-1].args[0])["metadata"]
    assert retained["action_attestation_id"] == "configured-action-duplicate"
    assert "authorization_decision" not in retained
    assert "decision_reason_code" not in retained


@pytest.fixture
def real_sdk() -> Any:
    """Require the actual runtime in readiness runs; skip visibly in minimal installs."""
    if os.environ.get("THOTH_REQUIRE_CLAUDE_AGENT_SDK") == "1":
        return importlib.import_module("claude_agent_sdk.types")
    return pytest.importorskip("claude_agent_sdk.types", reason="optional Claude SDK is not installed")


def _tool_events(tracer: Tracer) -> list[Any]:
    return [call.args[0] for call in tracer._emitter.emit.call_args_list if call.args[0].event_type != EventType.LLM_INVOCATION]


def _post_payload(sdk: Any, tool_use_id: str, tool_input: dict[str, Any], *, failure: bool = False) -> dict[str, Any]:
    common = {
        "session_id": "sess_123",
        "transcript_path": "/tmp/synthetic-transcript",
        "cwd": "/tmp",
        "tool_name": "Write",
        "tool_input": tool_input,
        "tool_use_id": tool_use_id,
    }
    if failure:
        return sdk.PostToolUseFailureHookInput(**common, hook_event_name="PostToolUseFailure", error="synthetic tool failed")
    return sdk.PostToolUseHookInput(**common, hook_event_name="PostToolUse", tool_response={"ok": True})


@pytest.mark.asyncio
@pytest.mark.parametrize("hold_token", [None, "pending-hold"])
async def test_real_sdk_unresolved_step_up_denies_without_side_effects(real_sdk: Any, tmp_path: Path, hold_token: str | None) -> None:
    tracer = _make_tracer()
    tracer._enforcer.acheck = AsyncMock(return_value=EnforcementDecision(decision=DecisionType.STEP_UP, hold_token=hold_token))
    tracer._enforcer.aexplain = AsyncMock(return_value=None)
    tracer._step_up.await_decision = AsyncMock(return_value=EnforcementDecision(decision=DecisionType.STEP_UP))
    callback = AsyncMock(return_value=real_sdk.PermissionResultAllow())
    options = instrument_claude_agent_sdk_options(real_sdk.ClaudeAgentOptions(can_use_tool=callback), tracer)
    target = tmp_path / "must-not-exist.txt"
    result = await options.can_use_tool("Write", {"path": str(target)}, real_sdk.ToolPermissionContext(tool_use_id="toolu_pending"))
    if isinstance(result, real_sdk.PermissionResultAllow):
        target.write_text("unauthorized side effect")

    assert not target.exists()
    assert isinstance(result, real_sdk.PermissionResultDeny)
    assert "step-up" in result.message
    callback.assert_not_awaited()
    assert tracer._session.tool_calls == []
    events = _tool_events(tracer)
    assert [event.event_type for event in events] == [EventType.TOOL_CALL_PRE, EventType.TOOL_CALL_BLOCK]
    assert [event.metadata["action_attestation_id"] for event in events] == ["toolu_pending"] * 2
    assert events[-1].metadata["authorization_decision"] == "STEP_UP"
    if hold_token is None:
        tracer._step_up.await_decision.assert_not_awaited()
    else:
        tracer._step_up.await_decision.assert_awaited_once_with(hold_token)


@pytest.mark.asyncio
async def test_real_sdk_step_up_waits_for_approval_before_allowing(real_sdk: Any, tmp_path: Path) -> None:
    tracer = _make_tracer()
    target = tmp_path / "approved.txt"
    callback = AsyncMock(return_value=real_sdk.PermissionResultAllow())
    tracer._enforcer.acheck = AsyncMock(return_value=EnforcementDecision(decision=DecisionType.STEP_UP, hold_token="approval-hold"))
    tracer._enforcer.aexplain = AsyncMock(return_value=None)

    async def approve(hold_token: str) -> EnforcementDecision:
        assert hold_token == "approval-hold"
        assert not target.exists()
        callback.assert_not_awaited()
        assert tracer._session.tool_calls == []
        return EnforcementDecision(decision=DecisionType.ALLOW)

    tracer._step_up.await_decision = AsyncMock(side_effect=approve)
    options = instrument_claude_agent_sdk_options(real_sdk.ClaudeAgentOptions(can_use_tool=callback), tracer)
    context = real_sdk.ToolPermissionContext(tool_use_id="toolu_approved")
    tool_input = {"path": str(target)}
    result = await options.can_use_tool("Write", tool_input, context)
    assert isinstance(result, real_sdk.PermissionResultAllow)
    Path(result.updated_input["path"]).write_text("approved effect")
    await options.hooks["PostToolUse"][-1].hooks[0](_post_payload(real_sdk, context.tool_use_id, result.updated_input), None, {})

    assert target.read_text() == "approved effect"
    tracer._step_up.await_decision.assert_awaited_once_with("approval-hold")
    callback.assert_awaited_once_with("Write", tool_input, context)
    assert tracer._session.tool_calls == ["Write"]
    events = _tool_events(tracer)
    assert [event.event_type for event in events] == [EventType.TOOL_CALL_PRE, EventType.TOOL_CALL_POST]
    assert [event.metadata["action_attestation_id"] for event in events] == ["toolu_approved"] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_id", [None, "  explicit-action  "])
@pytest.mark.parametrize("decision", [DecisionType.ALLOW, DecisionType.BLOCK, DecisionType.MODIFY])
async def test_real_sdk_permission_controls_synthetic_side_effects(real_sdk: Any, tmp_path: Path, configured_id: str | None, decision: DecisionType) -> None:
    """Exercise real SDK callback types locally; this does not run a Claude model loop."""
    tracer = _make_tracer()
    tracer._config.action_attestation_id = configured_id
    original = tmp_path / "original.txt"
    modified = tmp_path / "modified.txt"
    tracer._enforcer.acheck = AsyncMock(
        return_value=EnforcementDecision(
            decision=decision,
            reason="synthetic policy",
            violation_id="vio_synthetic" if decision == DecisionType.BLOCK else None,
            modified_tool_args={"input": {"path": str(modified)}},
        )
    )
    options = instrument_claude_agent_sdk_options(real_sdk.ClaudeAgentOptions(), tracer)
    context = real_sdk.ToolPermissionContext(tool_use_id="toolu_write")
    result = await options.can_use_tool("Write", {"path": str(original)}, context)
    expected_id = "explicit-action" if configured_id else "toolu_write"

    assert tracer._enforcer.acheck.await_args.kwargs["action_attestation_id"] == expected_id
    if isinstance(result, real_sdk.PermissionResultAllow):
        Path(result.updated_input["path"]).write_text("synthetic effect")
        await options.hooks["PostToolUse"][-1].hooks[0](_post_payload(real_sdk, context.tool_use_id, result.updated_input), None, {})

    if decision == DecisionType.BLOCK:
        assert isinstance(result, real_sdk.PermissionResultDeny)
        assert result.message == "synthetic policy"
        assert not original.exists()
        assert not modified.exists()
        assert tracer._session.tool_calls == []
        assert _tool_events(tracer)[-1].violation_id == "vio_synthetic"
        assert _tool_events(tracer)[-1].metadata["authorization_decision"] == "BLOCK"
    else:
        target = modified if decision == DecisionType.MODIFY else original
        assert target.read_text() == "synthetic effect"
        assert not (original if decision == DecisionType.MODIFY else modified).exists()
        assert tracer._session.tool_calls == ["Write"]

    events = _tool_events(tracer)
    assert len(events) == 2
    assert events[0].event_type == EventType.TOOL_CALL_PRE
    assert events[1].event_type == (EventType.TOOL_CALL_BLOCK if decision == DecisionType.BLOCK else EventType.TOOL_CALL_POST)
    assert [event.metadata["action_attestation_id"] for event in events] == [expected_id] * 2


@pytest.mark.asyncio
async def test_real_sdk_concurrent_same_name_calls_keep_lifecycle_ids(real_sdk: Any) -> None:
    tracer = _make_tracer()
    both_started = asyncio.Event()
    started = 0

    async def check(**kwargs: Any) -> EnforcementDecision:
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=2)
        return EnforcementDecision(decision=DecisionType.ALLOW)

    tracer._enforcer.acheck = AsyncMock(side_effect=check)
    options = instrument_claude_agent_sdk_options(real_sdk.ClaudeAgentOptions(), tracer)
    await asyncio.gather(*(options.can_use_tool("Write", {"path": name}, real_sdk.ToolPermissionContext(tool_use_id=name)) for name in ("toolu_first", "toolu_second")))
    for name, failure in [("toolu_second", True), ("toolu_first", False)]:
        hook = options.hooks["PostToolUseFailure" if failure else "PostToolUse"][-1].hooks[0]
        await hook(_post_payload(real_sdk, name, {"path": name}, failure=failure), None, {})

    events = _tool_events(tracer)
    assert [event.metadata["action_attestation_id"] for event in events] == ["toolu_first", "toolu_second", "toolu_second", "toolu_first"]
    assert events[-2].content == "synthetic tool failed"
    assert events[-2].event_type == EventType.TOOL_CALL_BLOCK
    assert events[-1].event_type == EventType.TOOL_CALL_POST
    assert {call.kwargs["action_attestation_id"] for call in tracer._enforcer.acheck.await_args_list} == {"toolu_first", "toolu_second"}


@pytest.mark.asyncio
@pytest.mark.parametrize("hook_name", ["PostToolUse", "PostToolUseFailure"])
async def test_missing_provider_id_does_not_invent_post_correlation(real_sdk: Any, hook_name: str) -> None:
    tracer = _make_tracer()
    tracer._enforcer.acheck = AsyncMock(side_effect=lambda **_: EnforcementDecision(decision=DecisionType.ALLOW))
    options = instrument_claude_agent_sdk_options(real_sdk.ClaudeAgentOptions(), tracer)
    for _ in range(2):
        await options.can_use_tool("Write", {"path": "same-name"}, real_sdk.ToolPermissionContext())
    ids = [event.metadata["action_attestation_id"] for event in _tool_events(tracer)]
    assert len(set(ids)) == 2
    assert all(str(UUID(value)) == value for value in ids)
    assert [call.kwargs["action_attestation_id"] for call in tracer._enforcer.acheck.await_args_list] == ids

    hook = options.hooks[hook_name][-1].hooks[0]
    await hook({"tool_name": "Write"}, None, {})
    assert "action_attestation_id" not in _tool_events(tracer)[-1].metadata
    assert "authorization_decision" not in _tool_events(tracer)[-1].metadata
    assert "decision_reason_code" not in _tool_events(tracer)[-1].metadata
    await hook({"tool_name": "Write", "tool_use_id": "unseen-provider-id"}, None, {})
    assert _tool_events(tracer)[-1].metadata["action_attestation_id"] == "unseen-provider-id"
    assert "authorization_decision" not in _tool_events(tracer)[-1].metadata
    assert "decision_reason_code" not in _tool_events(tracer)[-1].metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("hook_name", ["PostToolUse", "PostToolUseFailure"])
@pytest.mark.parametrize("configured_id", [None, "  explicit-action  "])
async def test_post_hook_id_precedence(real_sdk: Any, hook_name: str, configured_id: str | None) -> None:
    tracer = _make_tracer()
    tracer._config.action_attestation_id = configured_id
    options = instrument_claude_agent_sdk_options(real_sdk.ClaudeAgentOptions(), tracer)
    hook = options.hooks[hook_name][-1].hooks[0]
    for payload_id, argument_id, expected in [
        ("payload-id", "argument-id", "payload-id"),
        (None, "argument-id", "argument-id"),
        (None, None, None),
    ]:
        await hook({"tool_name": "Write", "tool_use_id": payload_id}, argument_id, {})
        assert _tool_events(tracer)[-1].metadata.get("action_attestation_id") == ("explicit-action" if configured_id else expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_result", ["deny", "allow", "allow_modified"])
async def test_real_sdk_existing_callback_result_is_preserved(real_sdk: Any, existing_result: str) -> None:
    tracer = _make_tracer()
    tracer._enforcer.acheck = AsyncMock(return_value=EnforcementDecision(decision=DecisionType.MODIFY, modified_tool_args={"input": {"path": "policy-safe"}}))
    result = (
        real_sdk.PermissionResultDeny(message="local denial", interrupt=True)
        if existing_result == "deny"
        else real_sdk.PermissionResultAllow(updated_input={"path": "callback-safe"} if existing_result == "allow_modified" else None)
    )
    callback = AsyncMock(return_value=result)
    existing_hook = AsyncMock(return_value={})
    existing_matcher = real_sdk.HookMatcher(hooks=[existing_hook])
    options = real_sdk.ClaudeAgentOptions(can_use_tool=callback, hooks={"PostToolUse": [existing_matcher]})
    instrument_claude_agent_sdk_options(options, tracer)
    context = real_sdk.ToolPermissionContext(tool_use_id="toolu_callback")
    assert await options.can_use_tool("Write", {"path": "unsafe"}, context) is result
    callback.assert_awaited_once_with("Write", {"path": "policy-safe"}, context)
    assert options.hooks["PostToolUse"][0] is existing_matcher
    if existing_result == "deny":
        assert result.interrupt
        assert tracer._session.tool_calls == []
        event = _tool_events(tracer)[-1]
        assert event.event_type == EventType.TOOL_CALL_BLOCK
        assert event.metadata["action_attestation_id"] == "toolu_callback"
    else:
        assert result.updated_input == {"path": "callback-safe" if existing_result == "allow_modified" else "policy-safe"}
        assert tracer._session.tool_calls == ["Write"]
