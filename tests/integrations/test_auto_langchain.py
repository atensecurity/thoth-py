"""thoth.init(): every LangChain and LangGraph tool governed with no per-tool code."""

import asyncio
import json
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

import thoth
from thoth import auto
from thoth.exceptions import ThothPolicyViolation

ENV = {
    "THOTH_AGENT_ID": "trader",
    "THOTH_TENANT_ID": "acme",
    "THOTH_API_URL": "https://enforce.example.test",
    "THOTH_API_KEY": "thoth_test_key",
}


class FakeThoth:
    """The enforcer over HTTP: records each check and answers per tool."""

    def __init__(self):
        self.checks: list[dict] = []
        self.decisions: dict[str, str] = {}

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/enforce":
            return httpx.Response(404)
        body = json.loads(request.content)
        self.checks.append(body)
        decision = self.decisions.get(body["tool_name"], "ALLOW")
        return httpx.Response(200, json={"decision": decision, "reason": f"{decision} by test policy"})

    def clients(self, _config):
        transport = httpx.MockTransport(self.handle)
        return (
            httpx.Client(base_url="https://enforce.example.test", transport=transport),
            httpx.AsyncClient(base_url="https://enforce.example.test", transport=transport),
        )


@pytest.fixture
def server(monkeypatch):
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    fake = FakeThoth()
    monkeypatch.setattr(auto, "enforcer_http_clients", fake.clients)
    monkeypatch.setattr(auto, "HttpEmitter", MagicMock())
    inventory = MagicMock()
    monkeypatch.setattr(auto, "publish_tool_inventory", inventory)
    fake.inventory = inventory
    yield fake
    thoth.shutdown()


ran: list[str] = []


@tool
def list_orders(account: str) -> str:
    """List open orders for an account."""
    ran.append(f"list_orders:{account}")
    return f"orders for {account}"


@tool
def wire_funds(amount: int) -> str:
    """Send a wire transfer."""
    ran.append(f"wire_funds:{amount}")
    return "sent"


@tool
async def fetch_quote(symbol: str) -> str:
    """Fetch a quote."""
    ran.append(f"fetch_quote:{symbol}")
    return f"{symbol} 101.5"


@pytest.fixture(autouse=True)
def reset_ran():
    ran.clear()


def test_init_names_the_missing_setting(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="THOTH_AGENT_ID"):
        thoth.init()


def test_every_tool_is_checked_with_its_arguments_and_actor(server):
    assert thoth.init() == ["langchain"]
    with thoth.acting_for("alice@acme.test", session_id="s-1") as session_id:
        assert list_orders.invoke({"account": "A-1"}) == "orders for A-1"
    assert session_id == "s-1"
    assert ran == ["list_orders:A-1"]
    (check,) = server.checks
    assert check["tool_name"] == "list_orders"
    assert check["tool_args"] == {"account": "A-1"}
    assert check["agent_id"] == "trader"
    assert check["user_id"] == "alice@acme.test"
    assert check["session_id"] == "s-1"


def test_a_refused_tool_never_runs(server):
    server.decisions["wire_funds"] = "BLOCK"
    thoth.init()
    with pytest.raises(ThothPolicyViolation):
        wire_funds.invoke({"amount": 1_000_000})
    assert ran == []


def test_async_tools_are_checked(server):
    server.decisions["fetch_quote"] = "BLOCK"
    thoth.init()
    with pytest.raises(ThothPolicyViolation):
        asyncio.run(fetch_quote.ainvoke({"symbol": "ACME"}))
    server.decisions["fetch_quote"] = "ALLOW"
    assert asyncio.run(fetch_quote.ainvoke({"symbol": "ACME"})) == "ACME 101.5"
    assert ran == ["fetch_quote:ACME"]
    assert [check["tool_name"] for check in server.checks] == ["fetch_quote", "fetch_quote"]


def test_langgraph_tool_node_is_governed_without_changes(server):
    graph_module = pytest.importorskip("langgraph.graph")
    prebuilt = pytest.importorskip("langgraph.prebuilt")

    server.decisions["wire_funds"] = "BLOCK"
    thoth.init()
    builder = graph_module.StateGraph(graph_module.MessagesState)
    builder.add_node("tools", prebuilt.ToolNode([list_orders, wire_funds]))
    builder.add_edge(graph_module.START, "tools")
    builder.add_edge("tools", graph_module.END)
    graph = builder.compile()
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "list_orders", "args": {"account": "A-2"}, "id": "c1"},
            {"name": "wire_funds", "args": {"amount": 5}, "id": "c2"},
        ],
    )
    # LangGraph 1.x's ToolNode doesn't turn the refusal into a message for the
    # model: it stops the run.
    with pytest.raises(ThothPolicyViolation):
        graph.invoke({"messages": [call]})
    assert "wire_funds:5" not in ran, "the refused tool never ran"
    assert sorted(check["tool_name"] for check in server.checks) == ["list_orders", "wire_funds"]

    server.decisions.clear()
    result = graph.invoke({"messages": [call]})
    contents = {message.tool_call_id: message.content for message in result["messages"][1:]}
    assert contents == {"c1": "orders for A-2", "c2": "sent"}


def test_sessions_follow_acting_for(server):
    thoth.init(user_id="service-account")
    list_orders.invoke({"account": "A"})
    with thoth.acting_for("bob@acme.test") as first:
        list_orders.invoke({"account": "B"})
    with thoth.acting_for("bob@acme.test") as second:
        list_orders.invoke({"account": "C"})
    default, bob_1, bob_2 = server.checks
    assert default["user_id"] == "service-account"
    assert (bob_1["user_id"], bob_1["session_id"]) == ("bob@acme.test", first)
    assert bob_2["session_id"] == second != first
    with pytest.raises(ValueError, match="needs the user"), thoth.acting_for(" "):
        pass


def test_a_session_keeps_its_history_however_many_others_run(server):
    thoth.init()
    with thoth.acting_for("carol@acme.test"):
        list_orders.invoke({"account": "I"})
        for index in range(1500):
            with thoth.acting_for(f"user-{index}@acme.test"):
                list_orders.invoke({"account": "J"})
        wire_funds.invoke({"amount": 2})
    last = server.checks[-1]
    assert last["user_id"] == "carol@acme.test"
    assert last["session_tool_calls"] == ["list_orders", "wire_funds"], "the session's history was not reset"


def test_a_named_session_continues_across_blocks(server):
    thoth.init()
    with thoth.acting_for("dana@acme.test", session_id="conversation-7"):
        list_orders.invoke({"account": "K"})
    # A later request resumes the conversation; an overlapping one shares it.
    with thoth.acting_for("dana@acme.test", session_id="conversation-7"):
        with thoth.acting_for("dana@acme.test", session_id="conversation-7"):
            wire_funds.invoke({"amount": 3})
        list_orders.invoke({"account": "K2"})
    assert server.checks[-1]["session_tool_calls"] == ["list_orders", "wire_funds", "list_orders"]
    # Another user's session with the same ID is a different session.
    with thoth.acting_for("erin@acme.test", session_id="conversation-7"):
        list_orders.invoke({"account": "L"})
    assert server.checks[-1]["session_tool_calls"] == ["list_orders"]


def test_a_tool_already_wrapped_is_checked_once(server):
    thoth.init()
    runtime = auto._runtime()
    wrapped = runtime.tracer().wrap_tool("list_orders", list_orders.invoke)
    assert wrapped({"account": "D"}) == "orders for D"
    assert len(server.checks) == 1


def test_each_tool_is_reported_to_the_inventory_once(server):
    thoth.init()
    list_orders.invoke({"account": "E"})
    list_orders.invoke({"account": "F"})
    wire_funds.invoke({"amount": 1})
    reported = [call.args[0].tools[0].name for call in server.inventory.call_args_list]
    assert reported == ["list_orders", "wire_funds"]
    assert server.inventory.call_args.kwargs["agent_id"] == "trader"


def test_shutdown_removes_the_hooks(server):
    thoth.init()
    thoth.shutdown()
    list_orders.invoke({"account": "G"})
    assert server.checks == []
    assert ran == ["list_orders:G"]


def test_frameworks_limits_the_hooks(server):
    assert thoth.init(frameworks=[]) == []
    list_orders.invoke({"account": "H"})
    assert server.checks == []
