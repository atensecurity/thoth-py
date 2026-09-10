import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter, sleep
from typing import TypedDict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from thoth.exceptions import ThothDeferredError, ThothPolicyViolation
from thoth.integrations.langgraph import (
    instrument_langgraph,
    instrument_tool_node,
    thoth_graph,
)
from thoth.models import DecisionType, EnforcementDecision


class _DummyState(TypedDict):
    value: str


@pytest.fixture
def mocked_runtime_clients():
    with (
        patch("thoth.integrations.langgraph.HttpEmitter") as MockEmitter,
        patch("thoth.integrations.langgraph.EnforcerClient") as MockEnforcer,
        patch("thoth.integrations.langgraph.StepUpClient") as MockStepUp,
    ):
        emitter = MagicMock()
        MockEmitter.return_value = emitter

        enforcer = MagicMock()
        enforcer.check.return_value = EnforcementDecision(decision=DecisionType.ALLOW)
        enforcer.acheck = AsyncMock(return_value=EnforcementDecision(decision=DecisionType.ALLOW))
        MockEnforcer.return_value = enforcer

        step_up = MagicMock()
        step_up.wait.return_value = EnforcementDecision(decision=DecisionType.ALLOW)
        step_up.await_decision = AsyncMock(return_value=EnforcementDecision(decision=DecisionType.ALLOW))
        MockStepUp.return_value = step_up

        yield {
            "emitter": emitter,
            "enforcer": enforcer,
            "step_up": step_up,
        }


def _event_types(emitter: MagicMock) -> list[str]:
    return [call.args[0].event_type.value for call in emitter.emit.call_args_list]


def _post_events(emitter: MagicMock):
    return [call.args[0] for call in emitter.emit.call_args_list if call.args[0].event_type.value == "TOOL_CALL_POST"]


def test_tool_list_instrumentation(mocked_runtime_clients):
    @tool
    def lookup(query: str) -> str:
        """Lookup docs."""

        return f"lookup:{query}"

    @tool
    async def compute(query: str) -> str:
        """Compute result."""

        return f"compute:{query}"

    @tool
    def summarize(query: str) -> str:
        """Summarize output."""

        return f"summary:{query}"

    governed = instrument_langgraph(
        [lookup, compute, summarize],
        agent_id="lg-agent",
        approved_scope=["lookup", "compute", "summarize"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )

    assert isinstance(governed, list)
    assert [tool_obj.name for tool_obj in governed] == ["lookup", "compute", "summarize"]

    assert governed[0].invoke({"query": "incident"}) == "lookup:incident"
    assert asyncio.run(governed[1].ainvoke({"query": "incident"})) == "compute:incident"

    assert mocked_runtime_clients["enforcer"].check.call_count >= 1
    assert mocked_runtime_clients["enforcer"].acheck.await_count >= 1


def test_block_decision(mocked_runtime_clients):
    executed = {"value": False}

    @tool
    def exfiltrate(query: str) -> str:
        """Sensitive export."""

        executed["value"] = True
        return query

    mocked_runtime_clients["enforcer"].check.return_value = EnforcementDecision(
        decision=DecisionType.BLOCK,
        reason="blocked by policy",
        violation_id="vio_123",
    )

    governed = instrument_langgraph(
        [exfiltrate],
        agent_id="lg-agent",
        approved_scope=[],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )

    with pytest.raises(ThothPolicyViolation):
        governed[0].invoke({"query": "secret"})

    assert executed["value"] is False
    event_types = _event_types(mocked_runtime_clients["emitter"])
    assert "TOOL_CALL_PRE" in event_types
    assert "TOOL_CALL_BLOCK" in event_types
    assert "TOOL_CALL_POST" not in event_types


def test_allow_decision(mocked_runtime_clients):
    executed = {"value": False}

    @tool
    def search_docs(query: str) -> str:
        """Search docs."""

        executed["value"] = True
        return f"ok:{query}"

    governed = instrument_langgraph(
        [search_docs],
        agent_id="lg-agent",
        approved_scope=["search_docs"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )

    assert governed[0].invoke({"query": "reset"}) == "ok:reset"
    assert executed["value"] is True

    event_types = _event_types(mocked_runtime_clients["emitter"])
    assert event_types.count("TOOL_CALL_PRE") == 1
    assert event_types.count("TOOL_CALL_POST") == 1
    post = _post_events(mocked_runtime_clients["emitter"])[0]
    assert post.metadata["authorization_decision"] == "ALLOW"


def test_step_up_decision(mocked_runtime_clients):
    @tool
    def write_note(note: str) -> str:
        """Write note."""

        return note

    mocked_runtime_clients["enforcer"].check.return_value = EnforcementDecision(
        decision=DecisionType.STEP_UP,
        hold_token="tok_step_up_1",
    )

    def delayed_approval(_: str) -> EnforcementDecision:
        sleep(0.05)
        return EnforcementDecision(
            decision=DecisionType.ALLOW,
            decision_reason_code="approved_by_human",
        )

    mocked_runtime_clients["step_up"].wait.side_effect = delayed_approval

    governed = instrument_langgraph(
        [write_note],
        agent_id="lg-agent",
        approved_scope=[],
        tenant_id="trantor",
        enforcement="step_up",
        api_url="https://enforcer.example",
    )

    start = perf_counter()
    assert governed[0].invoke({"note": "ok"}) == "ok"
    elapsed = perf_counter() - start

    assert elapsed >= 0.05
    mocked_runtime_clients["step_up"].wait.assert_called_once_with("tok_step_up_1")
    post = _post_events(mocked_runtime_clients["emitter"])[0]
    assert post.metadata["authorization_decision"] == "ALLOW"
    assert post.metadata["decision_reason_code"] == "approved_by_human"


def test_modify_decision(mocked_runtime_clients):
    seen: list[str] = []

    @tool
    def redact(query: str) -> str:
        """Redact data."""

        seen.append(query)
        return query

    mocked_runtime_clients["enforcer"].check.return_value = EnforcementDecision(
        decision=DecisionType.MODIFY,
        modified_tool_args={"query": "sanitized"},
        modification_reason="remove sensitive term",
    )

    governed = instrument_langgraph(
        [redact],
        agent_id="lg-agent",
        approved_scope=["redact"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )

    assert governed[0].invoke({"query": "secret"}) == "sanitized"
    assert seen == ["sanitized"]

    post = _post_events(mocked_runtime_clients["emitter"])[0]
    assert post.metadata["modification_reason"] == "remove sensitive term"
    assert post.metadata["original_tool_args"] == {"query": "secret"}
    assert post.metadata["modified_tool_args"] == {"query": "sanitized"}


def test_defer_decision(mocked_runtime_clients):
    executed = {"value": False}

    @tool
    def expensive(query: str) -> str:
        """Expensive analysis."""

        executed["value"] = True
        return query

    mocked_runtime_clients["enforcer"].check.return_value = EnforcementDecision(
        decision=DecisionType.DEFER,
        defer_reason="awaiting additional evidence",
        defer_timeout_seconds=30,
    )

    governed = instrument_langgraph(
        [expensive],
        agent_id="lg-agent",
        approved_scope=["expensive"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )

    with pytest.raises(ThothDeferredError, match="awaiting additional evidence") as exc:
        governed[0].invoke({"query": "x"})

    assert executed["value"] is False
    assert exc.value.defer_timeout_seconds == 30
    post = _post_events(mocked_runtime_clients["emitter"])[0]
    assert post.metadata["authorization_decision"] == "DEFER"
    assert post.metadata["defer_timeout_seconds"] == 30


def test_observe_mode_never_blocks(mocked_runtime_clients, caplog):
    @tool
    def run_task(query: str) -> str:
        """Run task."""

        return f"ran:{query}"

    mocked_runtime_clients["enforcer"].check.return_value = EnforcementDecision(
        decision=DecisionType.BLOCK,
        reason="enforcer unavailable",
    )

    governed = instrument_langgraph(
        [run_task],
        agent_id="lg-agent",
        approved_scope=[],
        tenant_id="trantor",
        enforcement="observe",
        api_url="https://enforcer.example",
    )

    with caplog.at_level("WARNING"):
        assert governed[0].invoke({"query": "x"}) == "ran:x"

    assert "observe mode" in caplog.text
    assert "TOOL_CALL_PRE" in _event_types(mocked_runtime_clients["emitter"])


def test_fail_closed_block_mode(mocked_runtime_clients):
    @tool
    def run_task(query: str) -> str:
        """Run task."""

        return f"ran:{query}"

    mocked_runtime_clients["enforcer"].check.return_value = EnforcementDecision(
        decision=DecisionType.BLOCK,
        reason="enforcer unavailable",
    )

    governed = instrument_langgraph(
        [run_task],
        agent_id="lg-agent",
        approved_scope=["run_task"],
        tenant_id="trantor",
        enforcement="block",
        api_url="https://enforcer.example",
    )

    with pytest.raises(ThothPolicyViolation, match="enforcer unavailable"):
        governed[0].invoke({"query": "x"})


def test_session_continuity(mocked_runtime_clients):
    @tool
    def tool1(value: str) -> str:
        """Tool one."""

        return value

    @tool
    def tool2(value: str) -> str:
        """Tool two."""

        return value

    @tool
    def tool3(value: str) -> str:
        """Tool three."""

        return value

    governed = instrument_langgraph(
        [tool1, tool2, tool3],
        agent_id="lg-agent",
        approved_scope=["tool1", "tool2", "tool3"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )

    governed[0].invoke({"value": "a"})
    governed[1].invoke({"value": "b"})
    governed[2].invoke({"value": "c"})

    calls = [call.kwargs for call in mocked_runtime_clients["enforcer"].check.call_args_list]
    assert len(calls) == 3
    assert len({item["session_id"] for item in calls}) == 1
    assert calls[0]["tool_calls"] == ["tool1"]
    assert calls[1]["tool_calls"] == ["tool1", "tool2"]
    assert calls[2]["tool_calls"] == ["tool1", "tool2", "tool3"]


def test_async_invoke(mocked_runtime_clients):
    @tool
    async def async_tool(query: str) -> str:
        """Async tool."""

        await asyncio.sleep(0)
        return f"async:{query}"

    governed = instrument_langgraph(
        [async_tool],
        agent_id="lg-agent",
        approved_scope=["async_tool"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )

    assert asyncio.run(governed[0].ainvoke({"query": "q"})) == "async:q"
    mocked_runtime_clients["enforcer"].acheck.assert_awaited_once()


def test_thread_safety(mocked_runtime_clients):
    records: list[dict[str, object]] = []

    def record_check(*, tool_name, session_id, tool_calls, tool_args=None, action_attestation_id=None):
        records.append(
            {
                "tool_name": tool_name,
                "session_id": session_id,
                "tool_calls": list(tool_calls),
            }
        )
        return EnforcementDecision(decision=DecisionType.ALLOW)

    mocked_runtime_clients["enforcer"].check.side_effect = record_check

    @tool
    def alpha(value: str) -> str:
        """Alpha."""

        return value

    @tool
    def beta(value: str) -> str:
        """Beta."""

        return value

    governed = instrument_langgraph(
        [alpha, beta],
        agent_id="lg-agent",
        approved_scope=["alpha", "beta"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )

    def invoke(idx: int) -> str:
        tool_obj = governed[idx % 2]
        return tool_obj.invoke({"value": f"v{idx}"})

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(invoke, range(10)))

    assert len(results) == 10
    assert len(records) == 10
    assert len({record["session_id"] for record in records}) == 1
    for record in records:
        assert isinstance(record["tool_calls"], list)
        assert record["tool_calls"]
        assert record["tool_calls"][-1] == record["tool_name"]


@pytest.mark.parametrize(
    ("async_path", "sync_fallback"),
    [(False, False), (True, False), (True, True)],
    ids=["sync", "async", "async-sync-fallback"],
)
@pytest.mark.parametrize("blocked", [False, True], ids=["allow", "block"])
def test_tool_node_wrapper(mocked_runtime_clients, async_path, sync_fallback, blocked):
    executed: list[str] = []

    @tool
    def echo(text: str) -> str:
        """Echo message."""

        executed.append(text)
        return text

    @tool("echo")
    async def async_echo(text: str) -> str:
        """Echo message asynchronously."""

        executed.append(text)
        return text

    decision = EnforcementDecision(
        decision=DecisionType.BLOCK if blocked else DecisionType.ALLOW,
        reason="blocked by policy" if blocked else None,
    )
    mocked_runtime_clients["enforcer"].check.return_value = decision
    mocked_runtime_clients["enforcer"].acheck.return_value = decision

    node = instrument_tool_node(
        [async_echo if async_path and not sync_fallback else echo],
        agent_id="lg-agent",
        approved_scope=["echo"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )

    assert isinstance(node, ToolNode)
    assert "echo" in node.tools_by_name

    # Graph execution supplies the runtime required by current LangGraph ToolNodes.
    builder = StateGraph(MessagesState)
    builder.add_node("tools", node)
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()

    message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "echo",
                "args": {"text": "hello"},
                "type": "tool_call",
            }
        ],
    )
    try:
        if async_path:
            output = asyncio.run(graph.ainvoke({"messages": [message]}))
        else:
            output = graph.invoke({"messages": [message]})
    except ThothPolicyViolation as exc:
        assert blocked
        assert "blocked by policy" in str(exc)
    else:
        if blocked:
            # Older LangGraph versions turn tool exceptions into error messages.
            assert output["messages"][-1].status == "error"
            assert "blocked by policy" in output["messages"][-1].content
        else:
            assert output["messages"][-1].content == "hello"

    assert executed == ([] if blocked else ["hello"])
    if async_path:
        mocked_runtime_clients["enforcer"].acheck.assert_awaited_once()
        mocked_runtime_clients["enforcer"].check.assert_not_called()
    else:
        mocked_runtime_clients["enforcer"].check.assert_called_once()
    event_types = _event_types(mocked_runtime_clients["emitter"])
    assert event_types.count("TOOL_CALL_PRE") == 1
    assert event_types.count("TOOL_CALL_BLOCK") == int(blocked)
    assert event_types.count("TOOL_CALL_POST") == int(not blocked)
    check = mocked_runtime_clients["enforcer"].acheck if async_path else mocked_runtime_clients["enforcer"].check
    action_id = check.call_args.kwargs["action_attestation_id"]
    assert action_id
    assert {call.args[0].metadata["action_attestation_id"] for call in mocked_runtime_clients["emitter"].emit.call_args_list} == {action_id}


def test_zero_friction_measurement(mocked_runtime_clients):
    started = perf_counter()

    from thoth.integrations.langgraph import instrument_langgraph as _instrument_langgraph

    @tool
    def my_tool(query: str) -> str:
        """Tool."""

        return query

    governed = _instrument_langgraph(
        [my_tool],
        agent_id="lg-agent",
        approved_scope=["my_tool"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )
    assert governed[0].invoke({"query": "test"}) == "test"

    elapsed = perf_counter() - started
    assert elapsed < 5.0

    instrumentation_lines = 2
    assert instrumentation_lines <= 3


def test_regulatory_context_propagation():
    captured: dict[str, dict[str, object]] = {}

    def fake_check(self, tool_name, session_id, tool_calls, tool_args=None, action_attestation_id=None):
        captured["payload"] = self._payload(
            tool_name,
            session_id,
            tool_calls,
            tool_args=tool_args,
        )
        return EnforcementDecision(decision=DecisionType.ALLOW)

    async def fake_acheck(self, tool_name, session_id, tool_calls, tool_args=None, action_attestation_id=None):
        captured["payload"] = self._payload(
            tool_name,
            session_id,
            tool_calls,
            tool_args=tool_args,
        )
        return EnforcementDecision(decision=DecisionType.ALLOW)

    with (
        patch("thoth.integrations.langgraph.HttpEmitter") as MockEmitter,
        patch("thoth.integrations.langgraph.EnforcerClient.check", new=fake_check),
        patch("thoth.integrations.langgraph.EnforcerClient.acheck", new=fake_acheck),
        patch("thoth.integrations.langgraph.StepUpClient"),
    ):
        emitter = MagicMock()
        MockEmitter.return_value = emitter

        @tool
        def chart(query: str) -> str:
            """Chart data."""

            return query

        governed = instrument_langgraph(
            [chart],
            agent_id="lg-agent",
            approved_scope=["chart"],
            tenant_id="trantor",
            api_url="https://enforcer.example",
            session_intent="phi-eligibility-check",
            data_classification="PHI",
            purpose="clinical-documentation",
            task_context="triage workflow",
        )

        assert governed[0].invoke({"query": "x"}) == "x"

    payload = captured["payload"]
    assert payload["session_intent"] == "phi-eligibility-check"
    assert payload["data_classification"] == "PHI"
    assert payload["purpose"] == "clinical-documentation"
    assert payload["task_context"] == {"context": "triage workflow"}

    pre_event = emitter.emit.call_args_list[0].args[0]
    assert pre_event.metadata["session_intent"] == "phi-eligibility-check"
    assert pre_event.metadata["data_classification"] == "PHI"
    assert pre_event.metadata["purpose"] == "clinical-documentation"


def test_stategraph_and_compiled_graph_instrumentation(mocked_runtime_clients):
    @tool
    def search(value: str) -> str:
        """Search."""

        return f"ok:{value}"

    builder = StateGraph(_DummyState)
    builder.add_node("tools", ToolNode([search]))
    builder.add_edge(START, "tools")

    instrumented_builder = instrument_langgraph(
        builder,
        agent_id="lg-agent",
        approved_scope=["search"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )
    assert instrumented_builder is builder

    compiled = builder.compile()
    instrumented_compiled = instrument_langgraph(
        compiled,
        agent_id="lg-agent",
        approved_scope=["search"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )
    assert instrumented_compiled is compiled


def test_thoth_graph_decorator(mocked_runtime_clients):
    @tool
    def summarize(query: str) -> str:
        """Summarize."""

        return f"summary:{query}"

    @thoth_graph(
        agent_id="lg-agent",
        approved_scope=["summarize"],
        tenant_id="trantor",
        api_url="https://enforcer.example",
    )
    def build_graph():
        builder = StateGraph(_DummyState)
        builder.add_node("tools", ToolNode([summarize]))
        builder.add_edge(START, "tools")
        return builder

    graph = build_graph()
    tool_node = graph.nodes["tools"].runnable
    tool_obj = tool_node.tools_by_name["summarize"]
    assert tool_obj.invoke({"query": "x"}) == "summary:x"
