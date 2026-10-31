"""thoth.init(): every Google ADK tool governed at run_async, with no per-tool code."""

import asyncio
import json
from typing import Any, AsyncGenerator
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("google.adk")

from google.adk.agents import LlmAgent
from google.adk.flows.llm_flows.tools._caller import _call_tool_async
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

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
def list_orders(account: str) -> str:
    """List open orders for an account."""
    ran.append(f"list_orders:{account}")
    return f"orders for {account}"


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


class CallingModel(BaseLlm):
    """Calls one tool, then answers with what the tool returned."""

    tool_name: str
    args: dict[str, Any]

    async def generate_content_async(self, llm_request: Any, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        responses = [part.function_response.response for content in llm_request.contents for part in (content.parts or []) if part.function_response is not None]
        if responses:
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=f"done: {responses[-1]}")]))
            return
        call = types.FunctionCall(id="call-1", name=self.tool_name, args=self.args)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=call)]))


def run(tools: list[Any], tool_name: str, args: dict[str, Any]) -> str:
    agent = LlmAgent(name="trader", model=CallingModel(model="fake", tool_name=tool_name, args=args), instruction="Trade carefully.", tools=tools)
    runner = InMemoryRunner(agent=agent, app_name="thoth-test")

    async def go() -> str:
        session = await runner.session_service.create_session(app_name="thoth-test", user_id="alice")
        final = ""
        async for event in runner.run_async(user_id="alice", session_id=session.id, new_message=types.Content(role="user", parts=[types.Part(text="go")])):
            for part in (event.content.parts if event.content else None) or []:
                if part.text:
                    final = part.text
        return final

    return asyncio.run(go())


def test_function_tools_are_checked_with_their_arguments(server):
    assert "google_adk" in thoth.init()
    with thoth.acting_for("alice@acme.test"):
        final = run([FunctionTool(list_orders), FunctionTool(wire_funds)], "list_orders", {"account": "A-1"})
    assert "orders for A-1" in final
    assert ran == ["list_orders:A-1"]
    (check,) = server.checks
    assert check["tool_name"] == "list_orders"
    assert check["tool_args"] == {"account": "A-1"}
    assert check["user_id"] == "alice@acme.test"


def test_a_refused_tool_never_runs_and_the_model_is_told(server):
    server.decisions["wire_funds"] = "BLOCK"
    thoth.init()
    final = run([FunctionTool(wire_funds)], "wire_funds", {"amount": 1_000_000})
    assert ran == []
    assert "Thoth refused this call" in final


def test_a_tool_class_defined_after_init_is_governed(server):
    thoth.init()

    class Shout(BaseTool):
        def __init__(self) -> None:
            super().__init__(name="shout", description="Shout a message.")

        async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
            ran.append(f"shout:{args['text']}")
            return "shouted"

    server.decisions["shout"] = "BLOCK"
    assert asyncio.run(Shout().run_async(args={"text": "hi"}, tool_context=None)).startswith("Thoth refused this call")
    assert ran == []


def test_shutdown_restores_the_tools(server):
    thoth.init()
    thoth.shutdown()
    run([FunctionTool(list_orders)], "list_orders", {"account": "C"})
    assert server.checks == []
    assert ran == ["list_orders:C"]


async def ticker(symbol: str):
    """Stream prices for a symbol."""
    ran.append(f"ticker:{symbol}")
    yield f"{symbol} 1"
    yield f"{symbol} 2"


def test_a_live_streaming_tool_is_checked_before_its_stream_starts(server):
    # Live runs dispatch streaming tools through _call_tool_async, which ends in run_async.
    thoth.init()

    async def stream() -> Any:
        res = await _call_tool_async(FunctionTool(ticker), args={"symbol": "ACME"}, tool_context=None)
        return [item async for item in res] if hasattr(res, "__aiter__") else res

    assert asyncio.run(stream()) == ["ACME 1", "ACME 2"]
    assert server.checks[-1]["tool_name"] == "ticker"

    ran.clear()
    server.decisions["ticker"] = "BLOCK"
    assert str(asyncio.run(stream())).startswith("Thoth refused this call")
    assert ran == []
