"""thoth.init(): every Pydantic AI tool call governed with no per-tool code."""

import json
from unittest.mock import MagicMock

import httpx
from pydantic import BaseModel
import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent, TextOutput
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import thoth
from thoth import auto
from thoth.exceptions import ThothPolicyViolation

ENV = {
    "THOTH_AGENT_ID": "trader",
    "THOTH_TENANT_ID": "acme",
    "THOTH_API_URL": "https://enforce.example.test",
    "THOTH_API_KEY": "thoth_test_key",
}

ran: list[str] = []


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


def calling(tool_name: str, args: dict) -> FunctionModel:
    """A model that calls one tool, then answers with what the tool returned."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [part for message in messages for part in getattr(message, "parts", []) if isinstance(part, ToolReturnPart)]
        if returns:
            return ModelResponse(parts=[TextPart(f"done: {returns[-1].content}")])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args, tool_call_id="call-1")])

    return FunctionModel(respond)


def agent_with(model: FunctionModel) -> Agent:
    agent = Agent(model)

    @agent.tool_plain
    def list_orders(account: str) -> str:
        """List open orders for an account."""
        ran.append(f"list_orders:{account}")
        return f"orders for {account}"

    @agent.tool_plain
    def wire_funds(amount: int) -> str:
        """Send a wire transfer."""
        ran.append(f"wire_funds:{amount}")
        return "sent"

    return agent


def test_a_tool_call_is_checked_with_its_validated_arguments(server):
    agent = agent_with(calling("list_orders", {"account": "A-1"}))
    assert "pydantic_ai" in thoth.init()
    with thoth.acting_for("alice@acme.test"):
        result = agent.run_sync("What's open on A-1?")
    assert result.output == "done: orders for A-1"
    assert ran == ["list_orders:A-1"]
    (check,) = server.checks
    assert check["tool_name"] == "list_orders"
    assert check["tool_args"] == {"account": "A-1"}
    assert check["user_id"] == "alice@acme.test"


def test_a_refused_tool_never_runs_and_the_model_is_told(server):
    server.decisions["wire_funds"] = "BLOCK"
    agent = agent_with(calling("wire_funds", {"amount": 1_000_000}))
    thoth.init()
    result = agent.run_sync("Send it")
    assert ran == []
    assert result.output.startswith("done: Thoth refused this call")


class Answer(BaseModel):
    answer: str


def test_output_tools_are_not_checked(server):
    thoth.init()
    agent = Agent(calling("final_result", {"answer": "42"}), output_type=Answer)
    assert agent.run_sync("Answer").output == Answer(answer="42")
    assert server.checks == []


def test_an_output_function_is_checked_before_it_runs(server):
    sent: list[str] = []

    def send_summary(recipient: str) -> str:
        """Email the summary (the agent's final step)."""
        sent.append(recipient)
        return f"sent to {recipient}"

    thoth.init()
    agent = Agent(calling("final_result", {"recipient": "board@acme.test"}), output_type=send_summary)
    assert agent.run_sync("Wrap up").output == "sent to board@acme.test"
    assert sent == ["board@acme.test"]
    assert [(check["tool_name"], check["tool_args"]) for check in server.checks] == [
        ("send_summary", {"recipient": "board@acme.test"}),
    ]


def test_a_refused_output_function_never_runs(server):
    sent: list[str] = []

    def send_summary(recipient: str) -> str:
        """Email the summary."""
        sent.append(recipient)
        return "sent"

    server.decisions["send_summary"] = "BLOCK"
    thoth.init()
    agent = Agent(calling("final_result", {"recipient": "attacker@evil.test"}), output_type=send_summary)
    with pytest.raises(ThothPolicyViolation):
        agent.run_sync("Wrap up")
    assert sent == []


def test_a_text_output_function_is_checked_and_refused(server):
    posted: list[str] = []

    def post_update(text: str) -> str:
        """Post the agent's text to the team channel."""
        posted.append(text)
        return "posted"

    def answer_in_text(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("Q4 is closed")])

    thoth.init()
    agent = Agent(FunctionModel(answer_in_text), output_type=TextOutput(post_update))
    assert agent.run_sync("Update the team").output == "posted"
    assert posted == ["Q4 is closed"]
    assert [(check["tool_name"], check["tool_args"]) for check in server.checks] == [("post_update", {"text": "Q4 is closed"})]

    server.decisions["post_update"] = "BLOCK"
    with pytest.raises(ThothPolicyViolation):
        agent.run_sync("Update the team")
    assert posted == ["Q4 is closed"], "the refused text output function never ran"


def test_shutdown_removes_the_hook(server):
    agent = agent_with(calling("list_orders", {"account": "C"}))
    thoth.init()
    thoth.shutdown()
    agent.run_sync("go")
    assert server.checks == []
    assert ran == ["list_orders:C"]
