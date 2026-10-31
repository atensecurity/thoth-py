"""thoth.init(): every CrewAI tool governed where CrewAI's agents run it.

CrewAI's executor runs a tool one of two ways, and these tests call both:
native function calling calls ``tool.run(**args)``
(crewai/utilities/agent_utils.py, ``available_functions[name] = tool.run``),
and the text tool loop calls ``CrewStructuredTool.invoke(input=..., config=...)``
(crewai/tools/tool_usage.py).
"""

import asyncio
import json
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("crewai")

from crewai.tools import BaseTool, tool

import thoth
from thoth import auto

ENV = {
    "THOTH_AGENT_ID": "trader",
    "THOTH_TENANT_ID": "acme",
    "THOTH_API_URL": "https://enforce.example.test",
    "THOTH_API_KEY": "thoth_test_key",
}

ran: list[str] = []


# Defined at import time, before init(), as crews usually are.
@tool("list_orders")
def list_orders(account: str) -> str:
    """List open orders for an account."""
    ran.append(f"list_orders:{account}")
    return f"orders for {account}"


class WireFunds(BaseTool):
    name: str = "wire_funds"
    description: str = "Send a wire transfer."

    def _run(self, amount: int) -> str:
        ran.append(f"wire_funds:{amount}")
        return "sent"


wire_funds = WireFunds()


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


def test_native_tool_calls_are_checked_for_decorated_and_subclassed_tools(server):
    assert "crewai" in thoth.init()
    with thoth.acting_for("alice@acme.test"):
        assert list_orders.run(account="A-1") == "orders for A-1"
        assert wire_funds.run(amount=5) == "sent"
    assert ran == ["list_orders:A-1", "wire_funds:5"]
    assert [(check["tool_name"], check["tool_args"]) for check in server.checks] == [
        ("list_orders", {"account": "A-1"}),
        ("wire_funds", {"amount": 5}),
    ]
    assert server.checks[0]["user_id"] == "alice@acme.test"


def test_the_text_tool_loop_is_checked_through_invoke(server):
    thoth.init()
    structured = list_orders.to_structured_tool()
    assert structured.invoke(input={"account": "B-2"}, config={}) == "orders for B-2"
    assert ran == ["list_orders:B-2"]
    assert server.checks[-1]["tool_args"] == {"account": "B-2"}


def test_a_refused_tool_never_runs_and_the_agent_is_told(server):
    server.decisions["wire_funds"] = "BLOCK"
    thoth.init()
    assert wire_funds.run(amount=1_000_000).startswith("Thoth refused this call")
    structured = wire_funds.to_structured_tool()
    assert structured.invoke(input={"amount": 7}).startswith("Thoth refused this call")
    assert asyncio.run(wire_funds.arun(amount=8)).startswith("Thoth refused this call")
    assert ran == []


def test_shutdown_removes_the_hooks(server):
    thoth.init()
    thoth.shutdown()
    list_orders.run(account="C")
    assert server.checks == []
    assert ran == ["list_orders:C"]
