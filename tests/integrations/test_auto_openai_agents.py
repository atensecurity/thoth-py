"""thoth.init(): every OpenAI Agents SDK function tool governed with no per-tool code."""

import asyncio
import dataclasses
import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("agents")

import agents
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

import thoth
from thoth import auto

ENV = {
    "THOTH_AGENT_ID": "trader",
    "THOTH_TENANT_ID": "acme",
    "THOTH_API_URL": "https://enforce.example.test",
    "THOTH_API_KEY": "thoth_test_key",
}

ran: list[str] = []


# Created at import time, before init(), as agents usually define tools.
@agents.function_tool
def list_orders(account: str) -> str:
    """List open orders for an account."""
    ran.append(f"list_orders:{account}")
    return f"orders for {account}"


@agents.function_tool
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
    agents.set_tracing_disabled(True)
    ran.clear()
    yield fake
    thoth.shutdown()


class CallingModel(agents.Model):
    """Calls one function tool, then answers."""

    def __init__(self, name: str, arguments: dict[str, Any]):
        self.name, self.arguments, self.step = name, arguments, 0

    async def get_response(self, *args: Any, **kwargs: Any) -> agents.ModelResponse:
        self.step += 1
        if self.step == 1:
            output: list[Any] = [
                ResponseFunctionToolCall(type="function_call", call_id="call-1", name=self.name, arguments=json.dumps(self.arguments), id="fc-1", status="completed")
            ]
        else:
            output = [
                ResponseOutputMessage(
                    type="message",
                    id="msg-1",
                    role="assistant",
                    status="completed",
                    content=[ResponseOutputText(type="output_text", text="done", annotations=[])],
                )
            ]
        return agents.ModelResponse(output=output, usage=agents.Usage(), response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def run(name: str, arguments: dict[str, Any]) -> agents.RunResult:
    agent = agents.Agent(name="trader", instructions="Trade carefully.", model=CallingModel(name, arguments), tools=[list_orders, wire_funds])
    return asyncio.run(agents.Runner.run(agent, "go"))


def test_tools_created_before_init_are_checked_with_their_arguments(server):
    assert "openai_agents" in thoth.init()
    with thoth.acting_for("alice@acme.test"):
        result = run("list_orders", {"account": "A-1"})
    assert result.final_output == "done"
    assert ran == ["list_orders:A-1"]
    (check,) = server.checks
    assert check["tool_name"] == "list_orders"
    assert check["tool_args"] == {"account": "A-1"}
    assert check["user_id"] == "alice@acme.test"


def test_a_refused_tool_never_runs_and_the_model_is_told(server):
    server.decisions["wire_funds"] = "BLOCK"
    thoth.init()
    result = run("wire_funds", {"amount": 1_000_000})
    assert ran == []
    assert result.final_output == "done"
    outputs = [item for item in result.new_items if item.type == "tool_call_output_item"]
    assert "Thoth refused this call" in str(outputs[0].output)


def test_arguments_that_are_not_an_object_are_refused_without_a_check(server):
    thoth.init()
    output = asyncio.run(list_orders.on_invoke_tool(MagicMock(), '"A-1"'))
    assert output.startswith("Thoth refused this call")
    assert ran == []
    assert server.checks == []


def test_a_copy_is_governed_once_and_shutdown_restores_the_tools(server):
    thoth.init()
    copy = dataclasses.replace(list_orders)
    assert getattr(copy.on_invoke_tool, auto._THOTH_GOVERNED, False)
    asyncio.run(copy.on_invoke_tool(MagicMock(), json.dumps({"account": "C"})))
    assert len(server.checks) == 1, "a copy is not wrapped twice"
    thoth.shutdown()
    assert not getattr(list_orders.on_invoke_tool, auto._THOTH_GOVERNED, False)
    run("list_orders", {"account": "D"})
    assert len(server.checks) == 1, "no checks after shutdown"
    assert ran[-1] == "list_orders:D"
