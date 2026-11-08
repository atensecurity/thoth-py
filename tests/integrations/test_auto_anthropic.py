"""thoth.init(): the Anthropic client's runnable tools governed at call(input)."""

import asyncio
import importlib
import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

# anthropic 1.x has ToolError and talks through httpx2; the shared libs environment pins 0.x, which has neither.
pytest.importorskip("anthropic", minversion="1.0.0")
pytest.importorskip("anthropic.lib.tools")
pytest.importorskip("httpx2")

import anthropic
from anthropic.lib.tools import BetaAbstractMemoryTool, ToolError, beta_async_tool, beta_tool
import httpx2

import thoth
from thoth import auto

ENV = {
    "THOTH_AGENT_ID": "trader",
    "THOTH_TENANT_ID": "acme",
    "THOTH_API_URL": "https://enforce.example.test",
    "THOTH_API_KEY": "thoth_test_key",
}

ran: list[str] = []


@beta_tool
def create_order(account: str, amount: int) -> str:
    """Create an order.

    Args:
        account: The account.
        amount: How much.
    """
    ran.append(f"create_order:{account}:{amount}")
    return f"order for {account}"


@beta_async_tool
async def cancel_order(order_id: str) -> str:
    """Cancel an order.

    Args:
        order_id: The order.
    """
    ran.append(f"cancel_order:{order_id}")
    return f"cancelled {order_id}"


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


class FakeClaude:
    """The Messages API: asks for one tool call, then ends the turn."""

    def __init__(self, tool: str, tool_input: dict[str, Any]):
        self.tool, self.tool_input = tool, tool_input
        self.requests: list[dict] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if len(self.requests) == 1:
            content = [{"type": "tool_use", "id": "toolu_1", "name": self.tool, "input": self.tool_input}]
            stop = "tool_use"
        else:
            content, stop = [{"type": "text", "text": "done"}], "end_turn"
        return httpx2.Response(
            200,
            json={
                "id": f"msg_{len(self.requests)}",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": content,
                "stop_reason": stop,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    def tool_result(self) -> dict:
        (block,) = self.requests[1]["messages"][-1]["content"]
        return block


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


def run_sync(claude: FakeClaude, tool: Any) -> None:
    client = anthropic.Anthropic(api_key="test", http_client=httpx2.Client(transport=httpx2.MockTransport(claude.handle)))
    runner = client.beta.messages.tool_runner(model="claude-test", max_tokens=64, tools=[tool], messages=[{"role": "user", "content": "go"}])
    runner.until_done()


def run_async(claude: FakeClaude, tool: Any) -> None:
    async def go() -> None:
        client = anthropic.AsyncAnthropic(api_key="test", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(claude.handle)))
        runner = client.beta.messages.tool_runner(model="claude-test", max_tokens=64, tools=[tool], messages=[{"role": "user", "content": "go"}])
        await runner.until_done()

    asyncio.run(go())


def test_a_runner_call_is_checked_with_the_input_it_runs(server):
    assert "anthropic" in thoth.init()
    claude = FakeClaude("create_order", {"account": "A-1", "amount": 5})
    with thoth.acting_for("alice@acme.test"):
        run_sync(claude, create_order)
    assert ran == ["create_order:A-1:5"]
    (check,) = server.checks
    assert check["tool_name"] == "create_order"
    assert check["tool_args"] == {"account": "A-1", "amount": 5}
    assert check["user_id"] == "alice@acme.test"
    assert claude.tool_result()["content"] == "order for A-1"


def test_a_refused_call_never_runs_and_claude_is_told(server):
    server.decisions["create_order"] = "BLOCK"
    thoth.init()
    claude = FakeClaude("create_order", {"account": "A-1", "amount": 1_000_000})
    run_sync(claude, create_order)
    assert ran == []
    result = claude.tool_result()
    assert result["is_error"] is True
    assert "Thoth refused this call" in json.dumps(result["content"])


def test_an_async_runner_call_is_checked(server):
    server.decisions["cancel_order"] = "BLOCK"
    thoth.init()
    claude = FakeClaude("cancel_order", {"order_id": "O-9"})
    run_async(claude, cancel_order)
    assert ran == []
    assert [check["tool_name"] for check in server.checks] == ["cancel_order"]
    assert claude.tool_result()["is_error"] is True


def test_a_direct_call_is_governed(server):
    server.decisions["create_order"] = "BLOCK"
    thoth.init()
    with pytest.raises(ToolError, match="Thoth refused this call"):
        create_order.call({"account": "A-1", "amount": 1})
    assert ran == []


def test_input_that_cannot_be_checked_never_runs(server):
    thoth.init()
    with pytest.raises(ToolError, match="can't be checked"):
        create_order.call("not a dict")
    assert server.checks == []


def test_a_built_in_tool_defined_after_init_is_governed(server):
    server.decisions["memory"] = "BLOCK"
    thoth.init()

    class Memory(BetaAbstractMemoryTool):
        def view(self, command):
            return "seen"

        def create(self, command):
            ran.append(f"memory:create:{command.path}")
            return "created"

        def str_replace(self, command):
            return "replaced"

        def insert(self, command):
            return "inserted"

        def delete(self, command):
            return "deleted"

        def rename(self, command):
            return "renamed"

    with pytest.raises(ToolError, match="Thoth refused this call"):
        Memory().call({"command": "create", "path": "/memories/notes.md", "file_text": "secret"})
    assert ran == []
    assert server.checks[0]["tool_args"]["path"] == "/memories/notes.md"


def test_an_mcp_tool_from_the_helper_is_checked_once(server):
    mcp_server = pytest.importorskip("mcp.server.mcpserver")
    client_type = pytest.importorskip("mcp").Client
    async_mcp_tool = importlib.import_module("anthropic.lib.tools.mcp").async_mcp_tool

    app = mcp_server.MCPServer("brokerage")

    @app.tool()
    def close_account(account: str) -> str:
        """Close an account."""
        ran.append(f"close_account:{account}")
        return f"closed {account}"

    assert {"anthropic", "mcp_client"} <= set(thoth.init())

    async def go() -> Any:
        async with client_type(app) as client:
            (listed,) = (await client.session.list_tools()).tools
            return await async_mcp_tool(listed, client.session).call({"account": "D-4"})

    asyncio.run(go())
    assert ran == ["close_account:D-4"]
    # Checked by the Anthropic hook; the MCP hook doesn't check it again.
    assert [check["tool_name"] for check in server.checks] == ["close_account"]


def test_shutdown_restores_the_tools(server):
    thoth.init()
    thoth.shutdown()
    assert create_order.call({"account": "C-3", "amount": 2}) == "order for C-3"
    assert server.checks == []
