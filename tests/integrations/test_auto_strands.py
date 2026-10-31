"""thoth.init(): every Strands Agents tool governed, with no per-tool code."""

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("strands")

from strands import Agent, tool
from strands.models.model import Model

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


class CallingModel(Model):
    """Calls one tool, then answers with what the tool returned."""

    def __init__(self, tool_name: str, args: dict[str, Any]) -> None:
        self.tool_name = tool_name
        self.args = args

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> Any:
        return {}

    async def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
        yield

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, **kwargs: Any) -> Any:
        results = [block["toolResult"] for message in messages for block in message["content"] if "toolResult" in block]
        yield {"messageStart": {"role": "assistant"}}
        if results:
            text = " ".join(item.get("text", "") for item in results[-1]["content"])
            yield {"contentBlockDelta": {"delta": {"text": f"done: {text}"}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            return
        yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "call-1", "name": self.tool_name}}}}
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(self.args)}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}


def run(tools: list[Any], tool_name: str, args: dict[str, Any]) -> str:
    agent = Agent(model=CallingModel(tool_name, args), tools=tools, callback_handler=None)
    return str(agent("go"))


def test_function_tools_are_checked_with_their_arguments(server):
    assert "strands" in thoth.init()
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


def test_a_direct_tool_call_is_governed(server):
    server.decisions["wire_funds"] = "BLOCK"
    thoth.init()
    agent = Agent(model=CallingModel("wire_funds", {}), tools=[wire_funds], callback_handler=None)
    result = agent.tool.wire_funds(amount=5)
    assert ran == []
    assert result["status"] == "error"
    assert "Thoth refused this call" in result["content"][0]["text"]


def test_an_agent_used_as_a_tool_is_checked_and_so_are_its_tools(server):
    thoth.init()
    analyst = Agent(name="analyst", description="Answers order questions.", model=CallingModel("list_orders", {"account": "B-2"}), tools=[list_orders], callback_handler=None)
    final = run([analyst.as_tool(name="ask_analyst")], "ask_analyst", {"input": "open orders?"})
    assert "orders for B-2" in final
    assert [check["tool_name"] for check in server.checks] == ["ask_analyst", "list_orders"]

    server.checks.clear()
    ran.clear()
    server.decisions["ask_analyst"] = "BLOCK"
    final = run([analyst.as_tool(name="ask_analyst")], "ask_analyst", {"input": "open orders?"})
    assert "Thoth refused this call" in final
    assert ran == []
    assert [check["tool_name"] for check in server.checks] == ["ask_analyst"]


def test_shutdown_restores_the_tools(server):
    thoth.init()
    thoth.shutdown()
    run([list_orders], "list_orders", {"account": "C"})
    assert server.checks == []
    assert ran == ["list_orders:C"]
