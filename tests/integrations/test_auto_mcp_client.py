"""thoth.init(): every MCP tools/call made with ClientSession.call_tool governed."""

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("mcp.server.mcpserver")

from mcp import Client
from mcp.server.mcpserver import MCPServer

import thoth
from thoth import auto
from thoth._context import governed_call

ENV = {
    "THOTH_AGENT_ID": "trader",
    "THOTH_TENANT_ID": "acme",
    "THOTH_API_URL": "https://enforce.example.test",
    "THOTH_API_KEY": "thoth_test_key",
}

ran: list[str] = []

server_app = MCPServer("brokerage")


@server_app.tool()
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


def call(args: dict[str, Any], *, inside: tuple[str, dict[str, Any]] | None = None) -> Any:
    async def go() -> Any:
        async with Client(server_app) as client:
            if inside is None:
                return await client.session.call_tool("create_order", args)
            # As a framework hook runs a call it has checked.
            with governed_call(*inside):
                return await client.session.call_tool("create_order", args)

    return asyncio.run(go())


def text(result: Any) -> str:
    return " ".join(getattr(block, "text", "") for block in result.content)


def test_a_direct_call_is_checked_with_its_arguments(server):
    assert "mcp_client" in thoth.init()
    with thoth.acting_for("alice@acme.test"):
        result = call({"account": "A-1", "amount": 5})
    assert "order for A-1" in text(result)
    assert ran == ["create_order:A-1:5"]
    (check,) = server.checks
    assert check["tool_name"] == "create_order"
    assert check["tool_args"] == {"account": "A-1", "amount": 5}
    assert check["user_id"] == "alice@acme.test"


def test_a_refused_call_is_never_sent(server):
    server.decisions["create_order"] = "BLOCK"
    thoth.init()
    result = call({"account": "A-1", "amount": 1_000_000})
    assert ran == []
    assert result.is_error
    assert text(result).startswith("Thoth refused this call")


@pytest.mark.parametrize("checked_name", ["create_order", "brokerage_create_order", "brokerage-create_order", "brokerage.create_order"])
def test_a_call_a_framework_hook_checked_is_not_checked_again(server, checked_name):
    thoth.init()
    args = {"account": "B-2", "amount": 7}
    call(args, inside=(checked_name, args))
    assert ran == ["create_order:B-2:7"]
    assert server.checks == []


@pytest.mark.parametrize(
    ("checked_name", "checked_args"),
    [
        ("summarize", {"account": "B-2", "amount": 7}),  # another tool, same arguments
        ("brokerage_create_order", {"account": "B-2", "amount": 8}),  # same tool, other arguments
        ("create_orderx", {"account": "B-2", "amount": 7}),  # a name that only contains it
    ],
)
def test_any_other_call_inside_a_checked_one_is_checked(server, checked_name, checked_args):
    server.decisions["create_order"] = "BLOCK"
    thoth.init()
    result = call({"account": "B-2", "amount": 7}, inside=(checked_name, checked_args))
    assert ran == []
    assert result.is_error
    assert [check["tool_name"] for check in server.checks] == ["create_order"]


def test_shutdown_restores_the_session(server):
    thoth.init()
    thoth.shutdown()
    call({"account": "C", "amount": 1})
    assert server.checks == []
    assert ran == ["create_order:C:1"]
