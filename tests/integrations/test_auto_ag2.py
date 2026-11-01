"""thoth.init(): AG2 (1.x) function tools governed at FunctionTool._execute."""

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("ag2.tools.final.function_tool")

from ag2 import Agent, tool
from ag2.events import ToolCallEvent, ToolErrorEvent, ToolResultsEvent
from ag2.testing import TestConfig, TrackingConfig

import thoth
from thoth import auto

ENV = {
    "THOTH_AGENT_ID": "trader",
    "THOTH_TENANT_ID": "acme",
    "THOTH_API_URL": "https://enforce.example.test",
    "THOTH_API_KEY": "thoth_test_key",
}

ran: list[str] = []


@tool
def create_order(account: str, amount: int) -> str:
    """Create an order."""
    ran.append(f"create_order:{account}:{amount}")
    return f"order for {account}"


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


def run(arguments: dict[str, Any], *, middleware: Any = None) -> Any:
    """One scripted turn calling create_order, then a reply. Returns what the
    model was handed as the tool's result."""
    config = TrackingConfig(TestConfig(ToolCallEvent("create_order", arguments=json.dumps(arguments)), "done", raise_tool_errors=False))
    governed_tool = create_order.with_middleware(middleware) if middleware else create_order
    agent = Agent("trader", config=config, tools=[governed_tool])
    asyncio.run(agent.ask("go"))
    (results,) = [call.args[0] for call in config.mock.call_args_list if isinstance(call.args[0], ToolResultsEvent)]
    (result,) = results.results
    return result


def text(result: Any) -> str:
    return " ".join(str(getattr(part, "content", part)) for part in result.result.parts)


def test_an_agent_tool_call_is_checked_with_its_arguments(server):
    assert "ag2" in thoth.init()
    with thoth.acting_for("alice@acme.test"):
        result = run({"account": "A-1", "amount": 5})
    assert ran == ["create_order:A-1:5"]
    assert "order for A-1" in text(result)
    (check,) = server.checks
    assert check["tool_name"] == "create_order"
    assert check["tool_args"] == {"account": "A-1", "amount": 5}
    assert check["user_id"] == "alice@acme.test"


def test_a_refused_call_never_runs_and_the_model_is_told(server):
    server.decisions["create_order"] = "BLOCK"
    thoth.init()
    result = run({"account": "A-1", "amount": 1_000_000})
    assert ran == []
    assert isinstance(result, ToolErrorEvent)
    assert text(result).startswith("Thoth refused this call")


def test_arguments_middleware_changed_are_the_ones_checked_and_run(server):
    thoth.init()

    async def cap_amount(call_next, event, context):
        event.serialized_arguments = {**event.serialized_arguments, "amount": 10}
        return await call_next(event, context)

    run({"account": "B-2", "amount": 999}, middleware=cap_amount)
    assert ran == ["create_order:B-2:10"]
    assert server.checks[0]["tool_args"] == {"account": "B-2", "amount": 10}


def test_shutdown_restores_the_tools(server):
    thoth.init()
    thoth.shutdown()
    run({"account": "C-3", "amount": 2})
    assert ran == ["create_order:C-3:2"]
    assert server.checks == []
