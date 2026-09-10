"""Exercise standard LangChain async-to-sync dispatch through real tools."""

import asyncio
from contextvars import ContextVar
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool, Tool

from thoth.exceptions import ThothDeferredError, ThothPolicyViolation
from thoth.enforcer_client import EnforcerClient
from thoth.integrations.langgraph import instrument_langgraph
from thoth.models import DecisionType, EnforcementDecision, ThothConfig


@pytest.fixture
def runtime():
    with (
        patch("thoth.integrations.langgraph.EnforcerClient") as enforcer_class,
        patch("thoth.integrations.langgraph.StepUpClient") as step_up_class,
        patch("thoth.integrations.langgraph.HttpEmitter") as emitter_class,
    ):
        enforcer = enforcer_class.return_value
        enforcer.check.return_value = EnforcementDecision(decision=DecisionType.ALLOW)
        enforcer.acheck = AsyncMock(return_value=EnforcementDecision(decision=DecisionType.ALLOW))
        step_up = step_up_class.return_value
        step_up.await_decision = AsyncMock(return_value=EnforcementDecision(decision=DecisionType.ALLOW))
        yield enforcer, step_up, emitter_class.return_value


def govern(tool, **options):
    return instrument_langgraph(
        [tool],
        agent_id="fallback-agent",
        tenant_id="fallback-tenant",
        approved_scope=[tool.name],
        api_url="https://enforcer.example",
        **options,
    )[0]


def events(emitter):
    return [call.args[0] for call in emitter.emit.call_args_list]


@pytest.mark.parametrize("tool_type", [StructuredTool, Tool])
@pytest.mark.parametrize("outcome", ["ALLOW", "BLOCK", "DEFER", "MODIFY", "approval", "rejection"])
@pytest.mark.asyncio
async def test_fallback_enforces_once_before_side_effect(runtime, tool_type, outcome):
    enforcer, step_up, emitter = runtime
    effects = []

    def execute(text: str) -> str:
        """Record an observable effect."""
        effects.append(text)
        return text

    governed = govern(tool_type.from_function(execute, name="execute", description="Record an observable effect."))
    held = outcome in {"approval", "rejection"}
    enforcer.acheck.return_value = EnforcementDecision(
        decision=DecisionType.STEP_UP if held else DecisionType(outcome),
        hold_token="hold-1" if held else None,
        reason="policy result",
        enforcement_trace_id="trace-1",
        decision_reason_code="policy-test",
        modified_tool_args={"text": "sanitized"} if outcome == "MODIFY" else None,
    )

    async def resolve(token):
        assert token == "hold-1"
        assert effects == []
        await asyncio.sleep(0)
        return EnforcementDecision(
            decision=DecisionType.ALLOW if outcome == "approval" else DecisionType.BLOCK,
            reason="approval result",
            enforcement_trace_id="trace-1",
            decision_reason_code="approval-test",
        )

    step_up.await_decision.side_effect = resolve
    denied = outcome in {"BLOCK", "DEFER", "rejection"}
    if denied:
        exception = ThothDeferredError if outcome == "DEFER" else ThothPolicyViolation
        with pytest.raises(exception) as caught:
            await governed.ainvoke({"text": "original"})
        assert caught.value.enforcement_trace_id == "trace-1"
    else:
        assert await governed.ainvoke({"text": "original"}) == ("sanitized" if outcome == "MODIFY" else "original")

    assert effects == ([] if denied else ["sanitized" if outcome == "MODIFY" else "original"])
    enforcer.acheck.assert_awaited_once()
    enforcer.check.assert_not_called()
    assert step_up.await_decision.await_count == int(held)
    step_up.wait.assert_not_called()
    emitted = events(emitter)
    expected_end = "TOOL_CALL_BLOCK" if outcome in {"BLOCK", "rejection"} else "TOOL_CALL_POST"
    assert [event.event_type.value for event in emitted] == ["TOOL_CALL_PRE", expected_end]
    assert emitted[-1].metadata["enforcement_trace_id"] == "trace-1"
    assert emitted[-1].metadata["decision_reason_code"] == ("approval-test" if held else "policy-test")
    assert emitted[-1].session_tool_calls == ([] if denied else ["execute"])
    action_id = enforcer.acheck.call_args.kwargs["action_attestation_id"]
    assert action_id
    assert {event.metadata["action_attestation_id"] for event in emitted} == {action_id}
    if denied:
        assert caught.value.action_attestation_id == action_id


@pytest.mark.asyncio
async def test_fallback_preserves_executor_context_config_and_callbacks(runtime):
    context = ContextVar("fallback-test", default="missing")
    seen, callback_events = [], []

    class Callback(BaseCallbackHandler):
        def on_tool_start(self, *args, **kwargs):
            callback_events.append("start")

        def on_tool_end(self, *args, **kwargs):
            callback_events.append("end")

    def execute(text: str, config: RunnableConfig) -> str:
        """Observe framework configuration inside the executor."""
        seen.append((text, context.get(), config["configurable"]["target"]))
        return text

    governed = govern(StructuredTool.from_function(execute))
    token = context.set("request-context")
    try:
        assert (
            await governed.ainvoke(
                {"text": "allowed"},
                config={"configurable": {"target": "staging"}, "callbacks": [Callback()]},
            )
            == "allowed"
        )
    finally:
        context.reset(token)
    assert seen == [("allowed", "request-context", "staging")]
    assert callback_events == ["start", "end"]
    runtime[0].acheck.assert_awaited_once()
    runtime[0].check.assert_not_called()


@pytest.mark.asyncio
async def test_fallback_does_not_bypass_recursive_tool_authorization(runtime):
    enforcer, _, emitter = runtime
    effects = []

    def execute(text: str) -> str:
        """Attempt a separate nested action."""
        effects.append(text)
        return governed.invoke({"text": "nested"})

    governed = govern(StructuredTool.from_function(execute))
    enforcer.check.return_value = EnforcementDecision(decision=DecisionType.BLOCK, reason="nested denied")
    with pytest.raises(ThothPolicyViolation, match="nested denied"):
        await governed.ainvoke({"text": "outer"})
    assert effects == ["outer"]
    enforcer.acheck.assert_awaited_once()
    enforcer.check.assert_called_once()
    assert enforcer.check.call_args.kwargs["tool_args"] == {"text": "nested"}
    assert [event.event_type.value for event in events(emitter)] == ["TOOL_CALL_PRE", "TOOL_CALL_PRE", "TOOL_CALL_BLOCK"]


@pytest.mark.asyncio
async def test_concurrent_fallback_calls_keep_independent_decisions(runtime):
    enforcer, _, emitter = runtime
    effects = []
    arrived = asyncio.Event()
    count = 0

    def execute(text: str) -> str:
        """Record only allowed calls."""
        effects.append(text)
        return text

    async def decide(**kwargs):
        nonlocal count
        count += 1
        if count == 2:
            arrived.set()
        await asyncio.wait_for(arrived.wait(), timeout=2)
        return EnforcementDecision(decision=DecisionType.BLOCK if kwargs["tool_args"]["text"] == "denied" else DecisionType.ALLOW)

    enforcer.acheck.side_effect = decide
    governed = govern(StructuredTool.from_function(execute))
    outcomes = await asyncio.gather(governed.ainvoke({"text": "allowed"}), governed.ainvoke({"text": "denied"}), return_exceptions=True)
    assert outcomes[0] == "allowed"
    assert isinstance(outcomes[1], ThothPolicyViolation)
    assert effects == ["allowed"]
    assert enforcer.acheck.await_count == 2
    enforcer.check.assert_not_called()
    phases = [event.event_type.value for event in events(emitter)]
    assert phases.count("TOOL_CALL_PRE") == 2
    assert phases.count("TOOL_CALL_POST") == phases.count("TOOL_CALL_BLOCK") == 1
    action_ids = {call.kwargs["action_attestation_id"] for call in enforcer.acheck.call_args_list}
    assert len(action_ids) == 2
    for action_id in action_ids:
        matching = [event for event in events(emitter) if event.metadata["action_attestation_id"] == action_id]
        assert len(matching) == 2


@pytest.mark.asyncio
async def test_custom_async_override_is_preserved(runtime):
    calls = []

    class CustomTool(StructuredTool):
        async def ainvoke(self, input, config=None, **kwargs):
            calls.append(input)
            return "custom-result"

    def execute(text: str) -> str:
        """The override must not be replaced with this fallback."""
        raise AssertionError("synchronous body must not run")

    governed = govern(CustomTool.from_function(execute))
    assert await governed.ainvoke({"text": "value"}) == "custom-result"
    assert calls == [{"text": "value"}]
    runtime[0].acheck.assert_awaited_once()
    runtime[0].check.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_approval_does_not_execute_or_affect_next_call(runtime):
    enforcer, step_up, _ = runtime
    effects = []
    waiting = asyncio.Event()

    def execute(text: str) -> str:
        """Record an approved effect."""
        effects.append(text)
        return text

    async def wait_for_approval(token):
        waiting.set()
        await asyncio.Event().wait()

    step_up.await_decision.side_effect = wait_for_approval
    enforcer.acheck.return_value = EnforcementDecision(decision=DecisionType.STEP_UP, hold_token="pending")
    governed = govern(StructuredTool.from_function(execute))
    task = asyncio.create_task(governed.ainvoke(input={"text": "cancelled"}))
    try:
        await asyncio.wait_for(waiting.wait(), timeout=2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert effects == []
    enforcer.acheck.return_value = EnforcementDecision(decision=DecisionType.ALLOW)
    assert await governed.ainvoke(input={"text": "next"}) == "next"
    assert effects == ["next"]
    assert enforcer.acheck.await_count == 2
    enforcer.check.assert_not_called()


@pytest.mark.parametrize("path", ["sync", "async", "fallback"])
@pytest.mark.parametrize("hold_token", [None, "", "hold-unresolved"])
def test_unresolved_approval_never_executes(runtime, path, hold_token):
    enforcer, step_up, emitter = runtime
    effects = []

    def execute(text: str) -> str:
        """Record only authorized side effects."""
        effects.append(text)
        return text

    async def aexecute(text: str) -> str:
        return execute(text)

    pending = EnforcementDecision(
        decision=DecisionType.STEP_UP,
        hold_token=hold_token,
        enforcement_trace_id="pending-trace",
        decision_reason_code="approval-required",
        violation_id="pending-violation",
    )
    enforcer.check.return_value = pending
    enforcer.acheck.return_value = pending
    unresolved = EnforcementDecision(decision=DecisionType.STEP_UP)
    step_up.wait.return_value = unresolved
    step_up.await_decision.return_value = unresolved
    governed = govern(StructuredTool.from_function(execute, coroutine=aexecute if path == "async" else None))

    with pytest.raises(ThothPolicyViolation, match="step-up approval unresolved") as caught:
        if path == "sync":
            governed.invoke({"text": "must-not-run"})
        else:
            asyncio.run(governed.ainvoke({"text": "must-not-run"}))
    assert effects == []
    assert caught.value.enforcement_trace_id == "pending-trace"
    assert caught.value.violation_id == "pending-violation"
    assert caught.value.decision_reason_code == "approval-required"
    emitted = events(emitter)
    assert [event.event_type.value for event in emitted] == ["TOOL_CALL_PRE", "TOOL_CALL_BLOCK"]
    assert emitted[-1].metadata["authorization_decision"] == "STEP_UP"
    assert emitted[-1].session_tool_calls == []
    assert enforcer.check.call_count == int(path == "sync")
    assert enforcer.acheck.await_count == int(path != "sync")
    assert step_up.wait.call_count == int(bool(hold_token) and path == "sync")
    assert step_up.await_decision.await_count == int(bool(hold_token) and path != "sync")


@pytest.mark.parametrize("decision", ["ALLOW", "BLOCK", "STEP_UP"])
@pytest.mark.parametrize("echo", ["missing", "matching", "mismatched"])
@pytest.mark.asyncio
async def test_fallback_with_real_enforcer_http_client(decision, echo, monkeypatch):
    monkeypatch.delenv("THOTH_API_URL", raising=False)
    monkeypatch.delenv("THOTH_MOCK_MODE", raising=False)
    effects = []

    def execute(text: str) -> str:
        """Exercise the actual HTTP decision parsing before the effect."""
        effects.append(text)
        return text

    config = ThothConfig(
        agent_id="fallback-agent",
        tenant_id="fallback-tenant",
        approved_scope=["execute"],
        api_url="https://enforcer.example",
    )
    client = EnforcerClient(config)
    try:
        with (
            respx.mock as transport,
            patch("thoth.integrations.langgraph.EnforcerClient", return_value=client),
            patch("thoth.integrations.langgraph.StepUpClient") as step_up,
            patch("thoth.integrations.langgraph.HttpEmitter") as emitter,
        ):

            def respond(request):
                payload = {"decision": decision, "enforcement_trace_id": "http-trace"}
                if echo != "missing":
                    payload["actionAttestationId"] = json.loads(request.content)["action_attestation_id"] if echo == "matching" else "unrelated-action"
                return httpx.Response(200, json=payload)

            route = transport.post("https://enforcer.example/v1/enforce").mock(side_effect=respond)
            governed = govern(StructuredTool.from_function(execute))
            if decision == "ALLOW" and echo != "mismatched":
                assert await governed.ainvoke({"text": "effect"}) == "effect"
            else:
                with pytest.raises(ThothPolicyViolation):
                    await governed.ainvoke({"text": "effect"})
            assert effects == (["effect"] if decision == "ALLOW" and echo != "mismatched" else [])
            assert route.call_count == 1
            assert json.loads(route.calls[0].request.content)["tool_args"] == {"text": "effect"}
            if echo == "mismatched":
                assert events(emitter.return_value)[-1].metadata["decision_reason_code"] == "action_attestation_id_mismatch"
            else:
                assert events(emitter.return_value)[-1].metadata["enforcement_trace_id"] == "http-trace"
            action_id = json.loads(route.calls[0].request.content)["action_attestation_id"]
            assert {event.metadata["action_attestation_id"] for event in events(emitter.return_value)} == {action_id}
            step_up.return_value.wait.assert_not_called()
            step_up.return_value.await_decision.assert_not_called()
    finally:
        client.close()
        await client.aclose()


@pytest.mark.parametrize("path", ["sync", "async", "fallback"])
@pytest.mark.parametrize("echo_id", [None, "", True], ids=["missing-id", "empty-id", "echoed-id"])
def test_action_id_is_unique_per_call_without_mutating_decisions(runtime, path, echo_id):
    enforcer, _, emitter = runtime
    effects = []
    # The JSON normalizer converts an empty ID to None; also exercise an empty
    # value assigned by a custom response adapter after validation.
    legacy = EnforcementDecision(decision=DecisionType.ALLOW).model_copy(update={"action_attestation_id": "" if echo_id == "" else None})

    def execute(text: str) -> str:
        """Record a permitted action."""
        effects.append(text)
        return text

    async def aexecute(text: str) -> str:
        return execute(text)

    def decide(**kwargs):
        return EnforcementDecision(decision=DecisionType.ALLOW, action_attestation_id=kwargs["action_attestation_id"]) if echo_id else legacy

    enforcer.check.side_effect = decide
    enforcer.acheck.side_effect = decide
    governed = govern(StructuredTool.from_function(execute, coroutine=aexecute if path == "async" else None))
    for value in ["first", "second"]:
        if path == "sync":
            assert governed.invoke({"text": value}) == value
        else:
            assert asyncio.run(governed.ainvoke({"text": value})) == value
    calls = enforcer.check.call_args_list if path == "sync" else enforcer.acheck.call_args_list
    ids = [call.kwargs["action_attestation_id"] for call in calls]
    assert len(ids) == len(set(ids)) == 2
    assert [event.metadata["action_attestation_id"] for event in events(emitter)] == [ids[0], ids[0], ids[1], ids[1]]
    assert effects == ["first", "second"]
    assert legacy.action_attestation_id == ("" if echo_id == "" else None)


@pytest.mark.parametrize("path", ["sync", "async", "fallback"])
@pytest.mark.parametrize("phase", ["initial", "approval"])
@pytest.mark.parametrize("returned_decision", ["ALLOW", "MODIFY"])
def test_response_for_another_action_cannot_authorize_execution(runtime, path, phase, returned_decision):
    enforcer, step_up, emitter = runtime
    effects, requests = [], []

    def execute(text: str) -> str:
        """Only a decision for this action may reach this effect."""
        effects.append(text)
        return text

    async def aexecute(text: str) -> str:
        return execute(text)

    unrelated = EnforcementDecision(
        decision=DecisionType(returned_decision),
        action_attestation_id="another-action",
        modified_tool_args={"text": "changed"},
        receipt={"signature": "unrelated-receipt"},
        decision_evidence={"action": "another-action"},
    )

    def decide(**kwargs):
        requests.append(kwargs)
        if phase == "initial":
            return unrelated
        return EnforcementDecision(
            decision=DecisionType.STEP_UP,
            hold_token="hold-1",
            action_attestation_id=kwargs["action_attestation_id"],
        )

    enforcer.check.side_effect = decide
    enforcer.acheck.side_effect = decide
    step_up.wait.return_value = unrelated
    step_up.await_decision.return_value = unrelated
    governed = govern(StructuredTool.from_function(execute, coroutine=aexecute if path == "async" else None))
    with pytest.raises(ThothPolicyViolation) as caught:
        if path == "sync":
            governed.invoke({"text": "must-not-run"})
        else:
            asyncio.run(governed.ainvoke({"text": "must-not-run"}))
    assert effects == []
    assert caught.value.decision_reason_code == "action_attestation_id_mismatch"
    assert caught.value.action_attestation_id == requests[0]["action_attestation_id"]
    assert caught.value.authorization_decision == "BLOCK"
    assert caught.value.receipt is None
    assert caught.value.decision_evidence is None
    emitted = events(emitter)
    assert [event.event_type.value for event in emitted] == ["TOOL_CALL_PRE", "TOOL_CALL_BLOCK"]
    assert {event.metadata["action_attestation_id"] for event in emitted} == {requests[0]["action_attestation_id"]}
    assert emitted[-1].metadata["decision_reason_code"] == "action_attestation_id_mismatch"
    assert "receipt" not in emitted[-1].metadata
    assert "decision_evidence" not in emitted[-1].metadata
    assert emitted[-1].session_tool_calls == []


@pytest.mark.parametrize("echo", ["missing", "matching", "mismatched"])
@pytest.mark.asyncio
async def test_observe_mode_preserves_policy_semantics_but_rejects_wrong_action(runtime, echo):
    enforcer, _, emitter = runtime
    effects = []

    def execute(text: str) -> str:
        """Observe mode still executes valid policy denials."""
        effects.append(text)
        return text

    def decide(**kwargs):
        action_id = None if echo == "missing" else kwargs["action_attestation_id"] if echo == "matching" else "other-action"
        return EnforcementDecision(decision=DecisionType.BLOCK, action_attestation_id=action_id)

    enforcer.acheck.side_effect = decide
    governed = govern(StructuredTool.from_function(execute), enforcement="observe")
    if echo == "mismatched":
        with pytest.raises(ThothPolicyViolation):
            await governed.ainvoke({"text": "value"})
    else:
        assert await governed.ainvoke({"text": "value"}) == "value"
    assert effects == ([] if echo == "mismatched" else ["value"])
    action_id = enforcer.acheck.call_args.kwargs["action_attestation_id"]
    assert {event.metadata["action_attestation_id"] for event in events(emitter)} == {action_id}
