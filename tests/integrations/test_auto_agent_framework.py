"""thoth.init(): every Microsoft Agent Framework tool governed, with no per-tool code."""

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("agent_framework")

from agent_framework import Agent, BaseChatClient, ChatResponse, ChatResponseUpdate, Content, FunctionInvocationContext, Message, tool
from agent_framework._tools import FunctionInvocationLayer

import thoth
from thoth import auto

ENV = {
    "THOTH_AGENT_ID": "trader",
    "THOTH_TENANT_ID": "acme",
    "THOTH_API_URL": "https://enforce.example.test",
    "THOTH_API_KEY": "thoth_test_key",
}

ran: list[str] = []


# Defined at import time, before init(), as agents usually are.
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
def lookup(symbol: str, ctx: FunctionInvocationContext) -> str:
    """Look up a symbol, with the invocation context."""
    ran.append(f"lookup:{symbol}:{type(ctx).__name__}")
    return f"{symbol} found"


class FakeThoth:
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
    monkeypatch.setattr(auto, "publish_tool_inventory", MagicMock())
    ran.clear()
    yield fake
    thoth.shutdown()


class CallingClient(FunctionInvocationLayer, BaseChatClient):
    """Calls one tool, then answers with what the tool returned."""

    def __init__(self, tool_name: str, args: dict[str, Any]) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.args = args

    def _reply(self, messages: Any) -> list[Any]:
        results = [content.result for message in messages for content in message.contents if content.type == "function_result"]
        if results:
            last = results[-1]
            text = " ".join(str(getattr(item, "text", item)) for item in last) if isinstance(last, list) else str(last)
            return [Content.from_text(f"done: {text}")]
        return [Content.from_function_call(call_id="call-1", name=self.tool_name, arguments=self.args)]

    def _inner_get_response(self, *, messages: Any, stream: bool, options: Any, **kwargs: Any) -> Any:
        contents = self._reply(messages)
        if stream:
            # Agents used as tools run their agent streaming.
            async def updates() -> Any:
                yield ChatResponseUpdate(role="assistant", contents=contents)

            return self._build_response_stream(updates())

        async def response() -> ChatResponse:
            return ChatResponse(messages=[Message(role="assistant", contents=contents)])

        return response()


def run(tools: list[Any], tool_name: str, args: dict[str, Any]) -> str:
    agent = Agent(client=CallingClient(tool_name, args), instructions="Trade carefully.", tools=tools)
    return asyncio.run(agent.run("go")).text


def test_function_tools_are_checked_with_their_arguments(server):
    assert "agent_framework" in thoth.init()
    with thoth.acting_for("alice@acme.test"):
        final = run([list_orders, wire_funds], "list_orders", {"account": "A-1"})
    assert "orders for A-1" in final
    assert ran == ["list_orders:A-1"]
    (check,) = server.checks
    assert check["tool_name"] == "list_orders"
    assert check["tool_args"] == {"account": "A-1"}
    assert check["user_id"] == "alice@acme.test"


def test_a_refused_tool_never_runs_and_the_model_is_told(server):
    server.decisions["wire_funds"] = "BLOCK"
    thoth.init()
    final = run([wire_funds], "wire_funds", {"amount": 1_000_000})
    assert ran == []
    assert "Thoth refused this call" in final


def test_the_invocation_context_is_not_sent_but_still_reaches_the_tool(server):
    thoth.init()
    final = run([lookup], "lookup", {"symbol": "ACME"})
    assert "ACME found" in final
    assert ran == ["lookup:ACME:FunctionInvocationContext"]
    (check,) = server.checks
    assert check["tool_args"] == {"symbol": "ACME"}


def test_an_agent_used_as_a_tool_is_checked_and_so_are_its_tools(server):
    thoth.init()
    analyst = Agent(name="analyst", client=CallingClient("list_orders", {"account": "B-2"}), tools=[list_orders])
    final = run([analyst.as_tool(name="ask_analyst")], "ask_analyst", {"task": "open orders?"})
    assert "orders for B-2" in final
    assert [check["tool_name"] for check in server.checks] == ["ask_analyst", "list_orders"]
    assert server.checks[0]["tool_args"] == {"task": "open orders?"}

    server.checks.clear()
    ran.clear()
    server.decisions["ask_analyst"] = "BLOCK"
    final = run([analyst.as_tool(name="ask_analyst")], "ask_analyst", {"task": "open orders?"})
    assert "Thoth refused this call" in final
    assert ran == []
    assert [check["tool_name"] for check in server.checks] == ["ask_analyst"]


def test_a_direct_invoke_is_governed(server):
    server.decisions["wire_funds"] = "BLOCK"
    thoth.init()
    result = asyncio.run(wire_funds.invoke(arguments={"amount": 5}))
    assert ran == []
    assert "Thoth refused this call" in str(result[0].text)


def test_shutdown_restores_the_tools(server):
    thoth.init()
    thoth.shutdown()
    run([list_orders], "list_orders", {"account": "C"})
    assert server.checks == []
    assert ran == ["list_orders:C"]
