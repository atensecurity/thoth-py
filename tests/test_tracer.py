# tests/test_tracer.py
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from thoth import ThothPolicyViolation
from thoth.emitter import SqsEmitter
from thoth.enforcer_client import EnforcerClient
from thoth.models import DecisionType, EnforcementDecision, EnforcementMode, EventType, ThothConfig
from thoth.session import SessionContext
from thoth.step_up import StepUpClient
from thoth.tracer import Tracer


@pytest.fixture
def config():
    return ThothConfig(
        agent_id="test-agent",
        approved_scope=["read:data"],
        tenant_id="trantor",
        enforcement=EnforcementMode.BLOCK,
    )


@pytest.fixture
def tracer(config):
    session = SessionContext(config)
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    step_up = MagicMock(spec=StepUpClient)
    enforcer.check.return_value = EnforcementDecision(decision=DecisionType.ALLOW)
    return Tracer(config=config, session=session, emitter=emitter, enforcer=enforcer, step_up=step_up)


def test_allows_in_scope_tool(tracer):
    tool = MagicMock(return_value="invoice data")
    wrapped = tracer.wrap_tool("read:data", tool)
    result = wrapped("arg1")
    assert result == "invoice data"
    tool.assert_called_once_with("arg1")


def test_emits_pre_and_post_events(tracer):
    tool = MagicMock(return_value="ok")
    wrapped = tracer.wrap_tool("read:data", tool)
    wrapped()
    assert tracer._emitter.emit.call_count == 2  # PRE + POST
    pre_event = tracer._emitter.emit.call_args_list[0].args[0]
    post_event = tracer._emitter.emit.call_args_list[1].args[0]
    assert pre_event.metadata["event_phase"] == "pre"
    assert pre_event.metadata["tool_call"]["name"] == "read:data"
    assert pre_event.metadata["sdk_language"] == "python"
    assert pre_event.metadata["action_attestation_id"]
    assert post_event.metadata["event_phase"] == "post"
    assert post_event.metadata["authorization_decision"] == "ALLOW"
    assert post_event.metadata["result_type"] == "str"
    assert isinstance(post_event.metadata["duration_ms"], int)


def test_decision_debug_log_omits_sensitive_reason_and_hold_token(config, caplog):
    tracer = Tracer(
        config=config,
        session=SessionContext(config),
        emitter=MagicMock(spec=SqsEmitter),
        enforcer=MagicMock(spec=EnforcerClient),
        step_up=MagicMock(spec=StepUpClient),
    )
    decision = EnforcementDecision(
        decision=DecisionType.STEP_UP,
        authorization_decision="STEP_UP",
        decision_reason_code="approval_required",
        reason="SYNTHETIC-PHI-SECRET-REASON",
        hold_token="SYNTHETIC-PHI-SECRET-HOLD",
    )

    with caplog.at_level(logging.DEBUG, logger="thoth.tracer"):
        tracer._log_decision(
            "read:data",
            decision,
            async_path=False,
            action_attestation_id="action-safe-001",
        )

    rendered = caplog.text
    assert "SYNTHETIC-PHI-SECRET-REASON" not in rendered
    assert "SYNTHETIC-PHI-SECRET-HOLD" not in rendered
    assert "approval_required" in rendered
    assert "action-safe-001" in rendered


def test_records_tool_call_in_session(tracer):
    tool = MagicMock(return_value="ok")
    wrapped = tracer.wrap_tool("read:data", tool)
    wrapped()
    assert "read:data" in tracer._session.tool_calls


def test_enforce_includes_current_tool_in_session_history(tracer):
    tool = MagicMock(return_value="ok")
    wrapped = tracer.wrap_tool("read:data", tool)
    wrapped()
    tracer._enforcer.check.assert_called_once()
    _, kwargs = tracer._enforcer.check.call_args
    assert kwargs["tool_calls"] == ["read:data"]
    assert kwargs["action_attestation_id"]


def test_raises_policy_violation_on_block(config):
    session = SessionContext(config)
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    step_up = MagicMock(spec=StepUpClient)
    enforcer.check.return_value = EnforcementDecision(
        decision=DecisionType.BLOCK,
        reason="out of scope",
        violation_id="vio_123",
        decision_reason_code="tool_not_allowed_for_session_intent",
        action_classification="context_deny",
        authorization_decision="DENY",
        risk_score=87.5,
        pack_id="engineering",
        model_signals=["moses_action:block"],
        enforcement_trace_id="trace-abc",
        score_components={"model_score": 87.5},
        top_contributors=[{"feature": "drift_score", "contribution_points": 35.0}],
        decision_evidence={"decision": "BLOCK", "authorization_decision": "DENY"},
    )
    t = Tracer(config=config, session=session, emitter=emitter, enforcer=enforcer, step_up=step_up)
    tool = MagicMock()
    wrapped = t.wrap_tool("write:s3", tool)
    with pytest.raises(ThothPolicyViolation) as exc:
        wrapped()
    assert "write:s3" in str(exc.value)
    assert exc.value.decision_reason_code == "tool_not_allowed_for_session_intent"
    assert exc.value.authorization_decision == "DENY"
    assert exc.value.risk_score == 87.5
    assert exc.value.pack_id == "engineering"
    assert exc.value.model_signals == ["moses_action:block"]
    assert exc.value.enforcement_trace_id == "trace-abc"
    assert exc.value.score_components == {"model_score": 87.5}
    assert exc.value.decision_evidence == {"decision": "BLOCK", "authorization_decision": "DENY"}
    tool.assert_not_called()  # tool never ran
    block_event = emitter.emit.call_args_list[1].args[0]
    assert block_event.metadata["risk_score"] == 87.5
    assert block_event.metadata["pack_id"] == "engineering"
    assert block_event.metadata["model_signals"] == ["moses_action:block"]
    assert block_event.metadata["enforcement_trace_id"] == "trace-abc"
    assert block_event.metadata["score_components"] == {"model_score": 87.5}
    assert block_event.metadata["decision_evidence"] == {"decision": "BLOCK", "authorization_decision": "DENY"}


def test_block_violation_includes_human_explanation(config):
    session = SessionContext(config)
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    step_up = MagicMock(spec=StepUpClient)
    enforcer.check.return_value = EnforcementDecision(
        decision=DecisionType.BLOCK,
        reason="out of scope",
        violation_id="vio_123",
    )
    enforcer.explain.return_value = {
        "what_happened": "Your agent attempted to delete data.",
        "what_to_do_next": "Contact #secops.",
    }

    t = Tracer(config=config, session=session, emitter=emitter, enforcer=enforcer, step_up=step_up)
    wrapped = t.wrap_tool("write:s3", MagicMock())

    with pytest.raises(ThothPolicyViolation) as exc:
        wrapped()

    assert exc.value.explanation is not None
    assert "Contact #secops." in str(exc.value.explanation)
    enforcer.explain.assert_called_once()


def test_waits_for_step_up_then_allows(config):
    config.enforcement = EnforcementMode.STEP_UP
    session = SessionContext(config)
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    step_up = MagicMock(spec=StepUpClient)

    enforcer.check.return_value = EnforcementDecision(decision=DecisionType.STEP_UP, hold_token="tok_abc")
    step_up.wait.return_value = EnforcementDecision(decision=DecisionType.ALLOW)

    t = Tracer(config=config, session=session, emitter=emitter, enforcer=enforcer, step_up=step_up)
    tool = MagicMock(return_value="done")
    wrapped = t.wrap_tool("write:slack", tool)
    result = wrapped()
    assert result == "done"
    step_up.wait.assert_called_once_with("tok_abc")


@pytest.mark.asyncio
@pytest.mark.parametrize("async_tool", [False, True])
@pytest.mark.parametrize("hold_token", [None, "pending-hold"])
async def test_unresolved_step_up_never_executes_tool(config, tmp_path, async_tool, hold_token):
    """A missing hold or still-pending approval is not permission to execute."""
    session = SessionContext(config)
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    step_up = MagicMock(spec=StepUpClient)
    initial = EnforcementDecision(decision=DecisionType.STEP_UP, hold_token=hold_token)
    pending = EnforcementDecision(decision=DecisionType.STEP_UP)
    enforcer.check.return_value = initial
    enforcer.acheck = AsyncMock(return_value=initial)
    enforcer.explain.return_value = None
    enforcer.aexplain = AsyncMock(return_value=None)
    step_up.wait.return_value = pending
    step_up.await_decision = AsyncMock(return_value=pending)
    tracer = Tracer(config=config, session=session, emitter=emitter, enforcer=enforcer, step_up=step_up)
    target = tmp_path / "must-not-exist.txt"

    def write_tool():
        target.write_text("unauthorized side effect")

    async def async_write_tool():
        write_tool()

    wrapped = tracer.wrap_tool("write:file", async_write_tool if async_tool else write_tool)
    if async_tool:
        with pytest.raises(ThothPolicyViolation) as violation:
            await wrapped()
    else:
        with pytest.raises(ThothPolicyViolation) as violation:
            wrapped()

    assert not target.exists()
    assert session.tool_calls == []
    assert "step-up" in violation.value.reason
    assert violation.value.authorization_decision == "STEP_UP"
    events = [call.args[0] for call in emitter.emit.call_args_list]
    assert [event.event_type for event in events] == [EventType.TOOL_CALL_PRE, EventType.TOOL_CALL_BLOCK]
    assert events[0].metadata["action_attestation_id"] == events[1].metadata["action_attestation_id"]
    if hold_token is None:
        step_up.wait.assert_not_called()
        step_up.await_decision.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("async_tool", [False, True])
@pytest.mark.parametrize("pending_features", [None, {}, {"risk": 0.2}], ids=["missing", "empty", "replacement"])
async def test_pending_step_up_preserves_decision_evidence(config, async_tool, pending_features):
    session = SessionContext(config)
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    step_up = MagicMock(spec=StepUpClient)
    initial_evidence = {"policy": "approval-required"}
    initial = EnforcementDecision(
        decision=DecisionType.STEP_UP,
        hold_token="pending-hold",
        fastml_features={"risk": 0.7},
        decision_evidence=initial_evidence,
        violation_id="vio_pending",
        decision_reason_code="approval_required",
    )
    pending = EnforcementDecision(
        decision=DecisionType.STEP_UP,
        fastml_features=pending_features,
    )
    enforcer.check.return_value = initial
    enforcer.acheck = AsyncMock(return_value=initial)
    enforcer.explain.return_value = None
    enforcer.aexplain = AsyncMock(return_value=None)
    step_up.wait.return_value = pending
    step_up.await_decision = AsyncMock(return_value=pending)
    tracer = Tracer(config=config, session=session, emitter=emitter, enforcer=enforcer, step_up=step_up)
    tool = AsyncMock() if async_tool else MagicMock()
    wrapped = tracer.wrap_tool("Write", tool)

    if async_tool:
        with pytest.raises(ThothPolicyViolation) as violation:
            await wrapped()
        step_up.await_decision.assert_awaited_once_with("pending-hold")
    else:
        with pytest.raises(ThothPolicyViolation) as violation:
            wrapped()
        step_up.wait.assert_called_once_with("pending-hold")

    tool.assert_not_called()
    assert session.tool_calls == []
    expected_features = initial.fastml_features if pending_features is None else pending_features or None
    assert violation.value.fastml_features == expected_features
    assert violation.value.decision_evidence == initial_evidence
    assert violation.value.violation_id == "vio_pending"
    assert violation.value.decision_reason_code == "approval_required"
    events = [call.args[0] for call in emitter.emit.call_args_list]
    assert [event.event_type for event in events] == [EventType.TOOL_CALL_PRE, EventType.TOOL_CALL_BLOCK]
    assert events[-1].metadata.get("fastml_features") == expected_features
    assert events[-1].metadata["decision_evidence"] == initial_evidence
    assert events[-1].violation_id == "vio_pending"
    assert events[-1].metadata["decision_reason_code"] == "approval_required"


def test_modify_rewrites_tool_args(config):
    session = SessionContext(config)
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    step_up = MagicMock(spec=StepUpClient)
    enforcer.check.return_value = EnforcementDecision(
        decision=DecisionType.MODIFY,
        modified_tool_args={"input": "sanitized"},
    )

    t = Tracer(config=config, session=session, emitter=emitter, enforcer=enforcer, step_up=step_up)
    tool = MagicMock(return_value="ok")
    wrapped = t.wrap_tool("write:slack", tool)
    result = wrapped("original")
    assert result == "ok"
    tool.assert_called_once_with("sanitized")


def test_defer_raises_policy_violation(config):
    session = SessionContext(config)
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    step_up = MagicMock(spec=StepUpClient)
    enforcer.check.return_value = EnforcementDecision(
        decision=DecisionType.DEFER,
        defer_reason="awaiting human context",
        defer_timeout_seconds=30,
    )

    t = Tracer(config=config, session=session, emitter=emitter, enforcer=enforcer, step_up=step_up)
    tool = MagicMock(return_value="should not run")
    wrapped = t.wrap_tool("write:slack", tool)
    with pytest.raises(ThothPolicyViolation, match="awaiting human context"):
        wrapped("x")
    tool.assert_not_called()


def test_observe_mode_allows_out_of_scope(config):
    config.enforcement = EnforcementMode.OBSERVE
    session = SessionContext(config)
    emitter = MagicMock(spec=SqsEmitter)
    # Enforcer not called in observe mode
    enforcer = MagicMock(spec=EnforcerClient)
    step_up = MagicMock(spec=StepUpClient)

    t = Tracer(config=config, session=session, emitter=emitter, enforcer=enforcer, step_up=step_up)
    tool = MagicMock(return_value="ok")
    wrapped = t.wrap_tool("write:s3", tool)
    result = wrapped()
    assert result == "ok"
    enforcer.check.assert_not_called()


@pytest.mark.asyncio
async def test_wrap_async_tool_executes(base_config):
    """wrap_tool must await async functions using the non-blocking async enforce path."""
    session = SessionContext(base_config)
    emitter = SqsEmitter(queue_url=None)
    enforcer = MagicMock(spec=EnforcerClient)
    # Async wrapped tools call acheck, not check.
    enforcer.acheck = AsyncMock(return_value=EnforcementDecision(decision=DecisionType.ALLOW))
    step_up = MagicMock(spec=StepUpClient)
    tracer = Tracer(
        config=base_config,
        session=session,
        emitter=emitter,
        enforcer=enforcer,
        step_up=step_up,
    )

    async def async_tool(x: int) -> int:
        return x * 2

    wrapped = tracer.wrap_tool("async_tool", async_tool)
    result = await wrapped(5)
    assert result == 10  # would be coroutine object if not properly awaited
    enforcer.acheck.assert_awaited_once()
    _, kwargs = enforcer.acheck.call_args
    assert kwargs["tool_calls"] == ["async_tool"]
    enforcer.check.assert_not_called()  # sync path must not be invoked for async tools


@pytest.mark.asyncio
async def test_async_tool_blocked_raises(base_config):
    """Async wrapped tools raise ThothPolicyViolation on BLOCK via the async enforce path."""
    session = SessionContext(base_config)
    emitter = SqsEmitter(queue_url=None)
    enforcer = MagicMock(spec=EnforcerClient)
    enforcer.acheck = AsyncMock(return_value=EnforcementDecision(decision=DecisionType.BLOCK, reason="async blocked"))
    step_up = MagicMock(spec=StepUpClient)
    tracer = Tracer(
        config=base_config,
        session=session,
        emitter=emitter,
        enforcer=enforcer,
        step_up=step_up,
    )

    async def async_tool() -> str:
        return "should not reach"

    wrapped = tracer.wrap_tool("async_tool", async_tool)
    with pytest.raises(ThothPolicyViolation):
        await wrapped()


def test_wrap_tool_preserves_name(base_config):
    """functools.wraps must preserve the function name."""
    session = SessionContext(base_config)
    emitter = SqsEmitter(queue_url=None)
    enforcer = MagicMock(spec=EnforcerClient)
    enforcer.check.return_value = EnforcementDecision(decision=DecisionType.ALLOW)
    step_up = MagicMock(spec=StepUpClient)
    tracer = Tracer(
        config=base_config,
        session=session,
        emitter=emitter,
        enforcer=enforcer,
        step_up=step_up,
    )

    def my_named_tool() -> None:
        pass

    wrapped = tracer.wrap_tool("my_named_tool", my_named_tool)
    assert wrapped.__name__ == "my_named_tool"


def test_log_decision_excludes_hold_token(config, caplog):
    session = SessionContext(config)
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    step_up = MagicMock(spec=StepUpClient)
    tracer = Tracer(config=config, session=session, emitter=emitter, enforcer=enforcer, step_up=step_up)

    decision = EnforcementDecision(
        decision=DecisionType.STEP_UP,
        hold_token="tok_step_up_123",
    )

    with caplog.at_level(logging.DEBUG, logger="thoth.tracer"):
        tracer._log_decision("write:slack", decision, async_path=False)

    assert "hold_token=tok_step_up_123" not in caplog.text
    assert "decision=STEP_UP" in caplog.text
