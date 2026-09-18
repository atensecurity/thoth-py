# tests/test_tracer.py
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx
from thoth import ThothPolicyViolation
from thoth.emitter import SqsEmitter
from thoth.enforcer_client import EnforcerClient
from thoth.models import DecisionType, EnforcementDecision, EnforcementMode, EventType, ThothConfig
from thoth.session import SessionContext
from thoth.step_up import StepUpClient, _coerce_hold_payload
from thoth.telemetry import telemetry_event
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


def test_sync_post_event_preserves_final_allow_decision_evidence(config):
    config.enforcement_trace_id = "trace-server-001"
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    enforcer.check.return_value = EnforcementDecision(
        decision=DecisionType.ALLOW,
        authorization_decision="ALLOW",
        decision_reason_code="policy_scope_allow",
        action_classification="context_allow",
        enforcement_trace_id="trace-server-001",
        pack_id="engineering",
        matched_rule_ids=["rule-safe-001"],
        decision_evidence={
            "decision_reason_code": "policy_scope_allow",
            "authorization_decision": "ALLOW",
            "policy": {"policy_id": "policy-safe-001"},
        },
    )
    tracer = Tracer(
        config=config,
        session=SessionContext(config),
        emitter=emitter,
        enforcer=enforcer,
        step_up=MagicMock(spec=StepUpClient),
    )

    assert tracer.wrap_tool("read:data", MagicMock(return_value="ok"))() == "ok"

    post = emitter.emit.call_args_list[-1].args[0]
    retained = telemetry_event(post)["metadata"]
    assert retained["authorization_decision"] == "ALLOW"
    assert retained["decision_reason_code"] == "policy_scope_allow"
    assert retained["action_classification"] == "context_allow"
    assert retained["enforcement_trace_id"] == "trace-server-001"
    assert retained["pack_id"] == "engineering"
    assert retained["matched_rule_ids"] == ["rule-safe-001"]
    assert retained["decision_evidence"]["policy"]["policy_id"] == "policy-safe-001"


@pytest.mark.asyncio
async def test_async_post_event_preserves_final_allow_decision_evidence(config):
    config.enforcement_trace_id = "trace-server-async"
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    enforcer.acheck = AsyncMock(
        return_value=EnforcementDecision(
            decision=DecisionType.STEP_UP,
            authorization_decision="STEP_UP",
            decision_reason_code="approval_required",
            hold_token="hold-safe-001",
        )
    )
    enforcer.aexplain = AsyncMock(return_value=None)
    step_up = MagicMock(spec=StepUpClient)
    step_up.await_decision = AsyncMock(
        return_value=EnforcementDecision(
            decision=DecisionType.ALLOW,
            authorization_decision="ALLOW",
            decision_reason_code="approved_by_human",
            action_classification="context_allow",
            enforcement_trace_id="trace-server-async",
            matched_control_ids=["control-safe-001"],
        )
    )
    tracer = Tracer(
        config=config,
        session=SessionContext(config),
        emitter=emitter,
        enforcer=enforcer,
        step_up=step_up,
    )

    async def tool() -> str:
        return "ok"

    assert await tracer.wrap_tool("read:data", tool)() == "ok"

    post = emitter.emit.call_args_list[-1].args[0]
    retained = telemetry_event(post)["metadata"]
    assert retained["authorization_decision"] == "ALLOW"
    assert retained["decision_reason_code"] == "approved_by_human"
    assert retained["action_classification"] == "context_allow"
    assert retained["enforcement_trace_id"] == "trace-server-async"
    assert retained["matched_control_ids"] == ["control-safe-001"]
    step_up.await_decision.assert_awaited_once_with("hold-safe-001")


def test_successful_post_telemetry_excludes_sensitive_decision_fields(config):
    secret = "SYNTHETIC-POST-DECISION-SECRET"
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    enforcer.check.return_value = EnforcementDecision(
        decision=DecisionType.ALLOW,
        decision_reason_code="policy_scope_allow",
        reason=secret,
        modification_reason=secret,
        modified_tool_args={"password": secret},
        decision_evidence={
            "decision_reason_code": "policy_scope_allow",
            "secret": secret,
        },
        receipt={
            "receipt_id": "receipt-safe-001",
            "secret": secret,
        },
    )
    tracer = Tracer(
        config=config,
        session=SessionContext(config),
        emitter=emitter,
        enforcer=enforcer,
        step_up=MagicMock(spec=StepUpClient),
    )

    tracer.wrap_tool("read:data", MagicMock(return_value="ok"))(password=secret)

    post = emitter.emit.call_args_list[-1].args[0]
    retained = telemetry_event(post)
    rendered = json.dumps(retained, sort_keys=True)
    assert secret not in rendered
    assert retained["metadata"]["decision_reason_code"] == "policy_scope_allow"
    assert retained["metadata"]["receipt"]["receipt_id"] == "receipt-safe-001"


@pytest.mark.asyncio
@pytest.mark.parametrize("async_path", [False, True], ids=["sync", "async"])
async def test_sparse_allow_preserves_local_lifecycle_correlation(config, async_path):
    config.enforcement_trace_id = "trace-local-001"
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    sparse = EnforcementDecision(decision=DecisionType.ALLOW)
    enforcer.check.return_value = sparse
    enforcer.acheck = AsyncMock(return_value=sparse)
    tracer = Tracer(
        config=config,
        session=SessionContext(config),
        emitter=emitter,
        enforcer=enforcer,
        step_up=MagicMock(spec=StepUpClient),
    )

    if async_path:

        async def tool() -> str:
            return "ok"

        assert await tracer.wrap_tool("read:data", tool)() == "ok"
    else:
        assert tracer.wrap_tool("read:data", MagicMock(return_value="ok"))() == "ok"

    pre, post = [call.args[0] for call in emitter.emit.call_args_list]
    assert pre.metadata["enforcement_trace_id"] == "trace-local-001"
    assert post.metadata["enforcement_trace_id"] == "trace-local-001"
    assert pre.metadata["action_attestation_id"]
    assert post.metadata["action_attestation_id"] == pre.metadata["action_attestation_id"]


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("async_path", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("configured_trace", "expected_trace"),
    [(" trace-abc ", "trace-abc"), (" \t\n ", "session-trace-fallback")],
    ids=["trimmed", "blank-fallback"],
)
async def test_normalized_trace_matches_request_server_and_lifecycle(
    async_path,
    configured_trace,
    expected_trace,
):
    config = ThothConfig(
        agent_id="test-agent",
        approved_scope=["read:data"],
        tenant_id="trantor",
        enforcement=EnforcementMode.BLOCK,
        api_url="https://enforcer.example",
    )
    config.enforcement_trace_id = configured_trace
    captured: dict[str, str] = {}

    def enforce(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured["trace_id"] = payload["enforcement_trace_id"]
        return httpx.Response(
            200,
            json={
                "decision": "ALLOW",
                "enforcement_trace_id": payload["enforcement_trace_id"],
                "action_attestation_id": payload["action_attestation_id"],
            },
        )

    respx.post("https://enforcer.example/v1/enforce").mock(side_effect=enforce)
    enforcer = EnforcerClient(config)
    emitter = MagicMock(spec=SqsEmitter)
    tracer = Tracer(
        config=config,
        session=SessionContext(config, session_id="session-trace-fallback"),
        emitter=emitter,
        enforcer=enforcer,
        step_up=MagicMock(spec=StepUpClient),
    )
    try:
        if async_path:

            async def tool() -> str:
                return "ok"

            assert await tracer.wrap_tool("read:data", tool)() == "ok"
        else:
            assert tracer.wrap_tool("read:data", MagicMock(return_value="ok"))() == "ok"
    finally:
        enforcer.close()
        await enforcer.aclose()

    assert config.enforcement_trace_id == ("trace-abc" if expected_trace == "trace-abc" else None)
    assert captured["trace_id"] == expected_trace
    pre, post = [call.args[0] for call in emitter.emit.call_args_list]
    assert pre.event_type == EventType.TOOL_CALL_PRE
    assert post.event_type == EventType.TOOL_CALL_POST
    assert pre.metadata["enforcement_trace_id"] == post.metadata["enforcement_trace_id"] == expected_trace
    assert pre.metadata["action_attestation_id"] == post.metadata["action_attestation_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("async_path", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("server_action_id", "server_trace_id", "reason_code"),
    [
        ("action-other", None, "action_attestation_id_mismatch"),
        (None, "trace-other", "enforcement_trace_id_mismatch"),
    ],
)
async def test_contradictory_server_correlation_fails_closed(
    config,
    async_path,
    server_action_id,
    server_trace_id,
    reason_code,
):
    config = ThothConfig(
        **config.model_dump(exclude={"enforcement_trace_id"}),
        enforcement_trace_id=" trace-local-001 ",
    )
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    contradictory = EnforcementDecision(
        decision=DecisionType.ALLOW,
        action_attestation_id=server_action_id,
        enforcement_trace_id=server_trace_id,
    )
    enforcer.check.return_value = contradictory
    enforcer.acheck = AsyncMock(return_value=contradictory)
    tracer = Tracer(
        config=config,
        session=SessionContext(config),
        emitter=emitter,
        enforcer=enforcer,
        step_up=MagicMock(spec=StepUpClient),
    )
    tool = AsyncMock(return_value="must-not-run") if async_path else MagicMock(return_value="must-not-run")
    wrapped = tracer.wrap_tool("read:data", tool)

    with pytest.raises(ThothPolicyViolation) as caught:
        if async_path:
            await wrapped()
        else:
            wrapped()

    tool.assert_not_called()
    assert caught.value.decision_reason_code == reason_code
    pre, block = [call.args[0] for call in emitter.emit.call_args_list]
    assert pre.metadata["enforcement_trace_id"] == block.metadata["enforcement_trace_id"] == "trace-local-001"
    assert pre.metadata["action_attestation_id"] == block.metadata["action_attestation_id"]
    assert block.metadata["decision_reason_code"] == reason_code


@pytest.mark.asyncio
@pytest.mark.parametrize("async_path", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("resolution", ["ALLOW", "BLOCK"])
async def test_hold_token_resolution_preserves_initial_and_terminal_evidence(
    config,
    async_path,
    resolution,
):
    secret = "SYNTHETIC-HOLD-RECEIPT-SECRET"
    config.enforcement_trace_id = "trace-hold-001"
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    initial = EnforcementDecision(
        decision=DecisionType.STEP_UP,
        hold_token="hold-safe-001",
        decision_reason_code="approval_required",
        pack_id="regulated-actions",
        decision_evidence={
            "policy": {"policy_id": "policy-safe-001"},
            "secret": secret,
        },
        receipt={"receipt_id": "receipt-initial", "signature": "sig-initial", "secret": secret},
    )
    terminal = _coerce_hold_payload(
        {
            "resolved": True,
            "resolution": resolution,
            "terminal_receipt": {
                "receipt_id": "receipt-terminal",
                "signature": "sig-terminal",
                "hold": {"reason": secret, "resolution": resolution},
                "secret": secret,
            },
        }
    )
    enforcer.check.return_value = initial
    enforcer.acheck = AsyncMock(return_value=initial)
    enforcer.explain.return_value = None
    enforcer.aexplain = AsyncMock(return_value=None)
    step_up = MagicMock(spec=StepUpClient)
    step_up.wait.return_value = terminal
    step_up.await_decision = AsyncMock(return_value=terminal)
    tracer = Tracer(
        config=config,
        session=SessionContext(config),
        emitter=emitter,
        enforcer=enforcer,
        step_up=step_up,
    )
    tool = AsyncMock(return_value="ok") if async_path else MagicMock(return_value="ok")
    wrapped = tracer.wrap_tool("read:data", tool)

    if resolution == "ALLOW":
        result = await wrapped() if async_path else wrapped()
        assert result == "ok"
        tool.assert_called_once_with()
        terminal_event = emitter.emit.call_args_list[-1].args[0]
        assert terminal_event.event_type == EventType.TOOL_CALL_POST
    else:
        with pytest.raises(ThothPolicyViolation) as caught:
            if async_path:
                await wrapped()
            else:
                wrapped()
        tool.assert_not_called()
        assert caught.value.authorization_decision == "BLOCK"
        terminal_event = emitter.emit.call_args_list[-1].args[0]
        assert terminal_event.event_type == EventType.TOOL_CALL_BLOCK

    retained = telemetry_event(terminal_event)["metadata"]
    assert retained["authorization_decision"] == resolution
    assert retained["decision_reason_code"] == "approval_required"
    assert retained["pack_id"] == "regulated-actions"
    assert retained["enforcement_trace_id"] == "trace-hold-001"
    assert retained["decision_evidence"]["policy"]["policy_id"] == "policy-safe-001"
    assert retained["receipt"]["receipt_id"] == "receipt-initial"
    assert retained["terminal_receipt"]["receipt_id"] == "receipt-terminal"
    assert secret not in json.dumps(retained, sort_keys=True)


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
    config.enforcement_trace_id = "trace-abc"
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
        authorization_decision="MODIFY",
        decision_reason_code="minimum_necessary_transform",
        modified_tool_args={"input": "sanitized"},
    )

    t = Tracer(config=config, session=session, emitter=emitter, enforcer=enforcer, step_up=step_up)
    tool = MagicMock(return_value="ok")
    wrapped = t.wrap_tool("write:slack", tool)
    result = wrapped("original")
    assert result == "ok"
    tool.assert_called_once_with("sanitized")
    post_event = emitter.emit.call_args_list[-1].args[0]
    assert post_event.metadata["authorization_decision"] == "MODIFY"
    assert post_event.metadata["decision_reason_code"] == "minimum_necessary_transform"


@pytest.mark.asyncio
async def test_async_modify_rewrites_tool_args_once_and_preserves_decision(config):
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    enforcer.acheck = AsyncMock(
        return_value=EnforcementDecision(
            decision=DecisionType.MODIFY,
            authorization_decision="MODIFY",
            decision_reason_code="minimum_necessary_transform",
            modified_tool_args={"input": "sanitized"},
        )
    )
    tracer = Tracer(
        config=config,
        session=SessionContext(config),
        emitter=emitter,
        enforcer=enforcer,
        step_up=MagicMock(spec=StepUpClient),
    )
    calls: list[str] = []

    async def tool(value: str) -> str:
        calls.append(value)
        return "ok"

    result = await tracer.wrap_tool("write:slack", tool)("original")

    assert result == "ok"
    assert calls == ["sanitized"]
    post_event = emitter.emit.call_args_list[-1].args[0]
    assert post_event.metadata["authorization_decision"] == "MODIFY"
    assert post_event.metadata["decision_reason_code"] == "minimum_necessary_transform"


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
    post_event = emitter.emit.call_args_list[-1].args[0]
    assert "authorization_decision" not in post_event.metadata
    assert "decision_reason_code" not in post_event.metadata
    assert "decision_evidence" not in post_event.metadata
    retained = json.loads(json.dumps(telemetry_event(post_event)))
    assert retained["enforcement_mode"] == "observe"
    assert "authorization_decision" not in retained["metadata"]
    assert "decision_reason_code" not in retained["metadata"]
    assert "decision_evidence" not in retained["metadata"]


@pytest.mark.asyncio
async def test_observe_mode_async_post_has_no_fabricated_decision(config):
    config.enforcement = EnforcementMode.OBSERVE
    emitter = MagicMock(spec=SqsEmitter)
    enforcer = MagicMock(spec=EnforcerClient)
    tracer = Tracer(
        config=config,
        session=SessionContext(config),
        emitter=emitter,
        enforcer=enforcer,
        step_up=MagicMock(spec=StepUpClient),
    )

    async def tool() -> str:
        return "ok"

    assert await tracer.wrap_tool("write:s3", tool)() == "ok"
    enforcer.acheck.assert_not_called()
    post_event = emitter.emit.call_args_list[-1].args[0]
    assert "authorization_decision" not in post_event.metadata
    assert "decision_reason_code" not in post_event.metadata
    assert "decision_evidence" not in post_event.metadata
    retained = json.loads(json.dumps(telemetry_event(post_event)))
    assert retained["enforcement_mode"] == "observe"
    assert "authorization_decision" not in retained["metadata"]
    assert "decision_reason_code" not in retained["metadata"]
    assert "decision_evidence" not in retained["metadata"]


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
