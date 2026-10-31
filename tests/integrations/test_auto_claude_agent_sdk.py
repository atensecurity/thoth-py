"""thoth.init(): every Claude Agent SDK query and client governed, with no options changes.

A fake CLI speaks the SDK's control protocol over a custom transport: it
answers initialize, then asks the SDK's registered PreToolUse hooks about one
tool call, as the real CLI does before running a tool, and records the answer.
"""

import asyncio
import json
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher, query
from claude_agent_sdk._internal.transport import Transport

import thoth
from thoth import auto

ENV = {
    "THOTH_AGENT_ID": "trader",
    "THOTH_TENANT_ID": "acme",
    "THOTH_API_URL": "https://enforce.example.test",
    "THOTH_API_KEY": "thoth_test_key",
}


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
    yield fake
    thoth.shutdown()


class FakeCLI(Transport):
    """The CLI end of the control protocol, for one tool call."""

    def __init__(self, tool_name: str, tool_input: dict[str, Any]) -> None:
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.outbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.pre_tool_use_ids: list[str] = []
        self.answers: list[dict[str, Any]] = []
        self.pending: set[str] = set()

    async def connect(self) -> None:
        pass

    def is_ready(self) -> bool:
        return True

    async def end_input(self) -> None:
        pass

    async def close(self) -> None:
        await self.outbox.put(None)

    async def write(self, data: str) -> None:
        for line in data.splitlines():
            message = json.loads(line)
            if message.get("type") == "control_request" and message["request"].get("subtype") == "initialize":
                for matcher in (message["request"].get("hooks") or {}).get("PreToolUse", []):
                    self.pre_tool_use_ids.extend(matcher["hookCallbackIds"])
                await self.outbox.put({"type": "control_response", "response": {"subtype": "success", "request_id": message["request_id"], "response": {}}})
                await self.ask_hooks()
            elif message.get("type") == "control_response" and message["response"].get("request_id") in self.pending:
                self.pending.discard(message["response"]["request_id"])
                self.answers.append(message["response"].get("response") or {})
                if not self.pending:
                    await self.finish()

    async def ask_hooks(self) -> None:
        if not self.pre_tool_use_ids:
            await self.finish()
            return
        for index, callback_id in enumerate(self.pre_tool_use_ids):
            request_id = f"hook-{index}"
            self.pending.add(request_id)
            hook_input = {
                "hook_event_name": "PreToolUse",
                "session_id": "sess-1",
                "transcript_path": "/tmp/transcript",
                "cwd": "/tmp",
                "tool_name": self.tool_name,
                "tool_input": self.tool_input,
                "tool_use_id": "toolu_1",
            }
            await self.outbox.put(
                {
                    "type": "control_request",
                    "request_id": request_id,
                    "request": {"subtype": "hook_callback", "callback_id": callback_id, "input": hook_input, "tool_use_id": "toolu_1"},
                }
            )

    async def finish(self) -> None:
        await self.outbox.put(
            {"type": "result", "subtype": "success", "duration_ms": 1, "duration_api_ms": 1, "is_error": False, "num_turns": 1, "session_id": "sess-1", "total_cost_usd": 0}
        )
        await self.outbox.put(None)

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        while (message := await self.outbox.get()) is not None:
            yield message

    def decisions(self) -> list[str]:
        return [answer.get("hookSpecificOutput", {}).get("permissionDecision", "none") for answer in self.answers]


async def run_query(cli: FakeCLI, options: ClaudeAgentOptions | None = None) -> None:
    async for _ in query(prompt="go", options=options, transport=cli):
        pass


def test_query_checks_every_tool_call_including_allowed_tools(server):
    assert "claude_agent_sdk" in thoth.init()
    cli = FakeCLI("Bash", {"command": "ls"})
    options = ClaudeAgentOptions(allowed_tools=["Bash"])
    with thoth.acting_for("alice@acme.test"):
        asyncio.run(run_query(cli, options))
    assert cli.decisions() == ["none"]  # allowed: left to the SDK's own rules
    (check,) = server.checks
    assert check["tool_name"] == "Bash"
    assert check["tool_args"] == {"command": "ls"}
    assert check["user_id"] == "alice@acme.test"
    # The caller's options are unchanged.
    assert options.hooks is None
    assert options.can_use_tool is None


def test_a_refused_call_is_denied_with_the_reason(server):
    server.decisions["Bash"] = "BLOCK"
    thoth.init()
    cli = FakeCLI("Bash", {"command": "rm -rf /"})
    asyncio.run(run_query(cli))
    assert cli.decisions() == ["deny"]
    reason = cli.answers[0]["hookSpecificOutput"]["permissionDecisionReason"]
    assert reason.startswith("Thoth refused this call")


def test_the_callers_own_hooks_still_run(server):
    seen: list[str] = []

    async def own_hook(hook_input: dict[str, Any], _tool_use_id: Any, _context: Any) -> dict[str, Any]:
        seen.append(hook_input["tool_name"])
        return {}

    thoth.init()
    cli = FakeCLI("Read", {"path": "notes.txt"})
    options = ClaudeAgentOptions(hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[own_hook])]})
    asyncio.run(run_query(cli, options))
    assert seen == ["Read"]
    assert len(server.checks) == 1
    assert len(options.hooks["PreToolUse"]) == 1


def test_options_instrumented_by_hand_are_not_checked_twice(server):
    thoth.init()
    tracer = auto._runtime().tracer()
    options = thoth.integrations.claude_agent_sdk.instrument_claude_agent_sdk_options(ClaudeAgentOptions(), tracer)
    cli = FakeCLI("Read", {"path": "notes.txt"})
    asyncio.run(run_query(cli, options))
    assert len(server.checks) == 1


def test_a_client_connection_is_governed(server):
    server.decisions["Write"] = "BLOCK"
    thoth.init()
    cli = FakeCLI("Write", {"path": "/etc/passwd"})
    options = ClaudeAgentOptions()

    async def connect() -> None:
        async with ClaudeSDKClient(options=options, transport=cli) as client:
            async for _ in client.receive_response():
                pass

    asyncio.run(connect())
    assert cli.decisions() == ["deny"]
    assert options.hooks is None


def test_shutdown_restores_the_sdk(server):
    thoth.init()
    thoth.shutdown()
    cli = FakeCLI("Bash", {"command": "ls"})
    asyncio.run(run_query(cli))
    assert cli.answers == []
    assert server.checks == []
