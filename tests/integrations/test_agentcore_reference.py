"""The AgentCore reference agent (platform/public/thoth-agentcore-reference): ALLOW runs, DENY never runs, a hold waits."""

from http.server import ThreadingHTTPServer
import importlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("strands")

import thoth
from thoth import auto, step_up

REFERENCE = Path(__file__).resolve().parents[6] / "platform" / "public" / "thoth-agentcore-reference"

ENV = {
    "THOTH_AGENT_ID": "clinic-scheduler",
    "THOTH_TENANT_ID": "acme",
    "THOTH_API_URL": "https://enforce.example.test",
    "THOTH_API_KEY": "thoth_test_key",
}


@pytest.fixture
def reference(monkeypatch):
    monkeypatch.syspath_prepend(str(REFERENCE))
    for name in ("clinic_tools", "stub_model", "agent"):
        sys.modules.pop(name, None)
    modules = SimpleNamespace(
        tools=importlib.import_module("clinic_tools"),
        agent=importlib.import_module("agent"),
    )
    modules.tools.executed.clear()
    yield modules
    for name in ("clinic_tools", "stub_model", "agent"):
        sys.modules.pop(name, None)


class FakeThoth:
    """Decides by tool name; a hold is pending on its first poll and resolved on the next."""

    def __init__(self, executed):
        self.executed = executed
        self.checks = []
        self.decisions = {}
        self.ran_when_first_polled = None
        self.polls = 0
        self.hold_resolution = "ALLOW"

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/enforce":
            body = json.loads(request.content)
            self.checks.append(body["tool_name"])
            decision = self.decisions.get(body["tool_name"], "ALLOW")
            payload = {"decision": decision, "reason": f"{decision} by test policy"}
            if decision == "STEP_UP":
                payload["hold_token"] = "hold-1"
            return httpx.Response(200, json=payload)
        if request.url.path == "/v1/enforce/hold/hold-1":
            self.polls += 1
            if self.polls == 1:
                self.ran_when_first_polled = list(self.executed)
                return httpx.Response(200, json={"resolved": False})
            return httpx.Response(200, json={"resolved": True, "resolution": self.hold_resolution})
        return httpx.Response(404)

    def clients(self, _config):
        transport = httpx.MockTransport(self.handle)
        return (
            httpx.Client(base_url="https://enforce.example.test", transport=transport),
            httpx.AsyncClient(base_url="https://enforce.example.test", transport=transport),
        )


@pytest.fixture
def server(monkeypatch, reference):
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    fake = FakeThoth(reference.tools.executed)
    monkeypatch.setattr(auto, "enforcer_http_clients", fake.clients)
    monkeypatch.setattr(auto, "step_up_http_clients", fake.clients)
    monkeypatch.setattr(auto, "HttpEmitter", MagicMock())
    monkeypatch.setattr(auto, "publish_tool_inventory", MagicMock())

    async def no_wait(_seconds):
        return None

    # Do not sit out the five-second poll interval; the hold is still polled twice.
    monkeypatch.setattr(step_up, "asyncio", SimpleNamespace(sleep=no_wait))
    monkeypatch.setattr(step_up, "time", SimpleNamespace(monotonic=step_up.time.monotonic, sleep=lambda _s: None))
    reference.agent.start_governance(local=False)
    yield fake
    thoth.shutdown()


def ask(reference, prompt):
    return reference.agent.invoke({"prompt": prompt, "user_id": "nurse@acme.test"}, "session-1", stub=True)["response"]


def test_an_allowed_call_runs(reference, server):
    assert "cardiology follow-up" in ask(reference, "list appointments")
    assert reference.tools.executed == ["lookup_appointments:P-1001"]
    assert server.checks == ["lookup_appointments"]


def test_a_denied_call_never_runs_and_the_agent_is_told(reference, server):
    server.decisions["export_patient_record"] = "DENY"
    answer = ask(reference, "export the record")
    assert "Thoth refused this call" in answer
    assert reference.tools.executed == []
    assert server.checks == ["export_patient_record"]


def test_a_held_call_waits_for_approval_then_runs(reference, server):
    server.decisions["send_referral_packet"] = "STEP_UP"
    answer = ask(reference, "send a referral")
    assert server.polls == 2
    # While the hold was pending, the tool had not run.
    assert server.ran_when_first_polled == []
    assert reference.tools.executed == ["send_referral_packet:P-1001"]
    assert "sent the referral packet" in answer


def test_a_held_call_that_is_declined_never_runs(reference, server):
    server.decisions["send_referral_packet"] = "STEP_UP"
    server.hold_resolution = "DENY"
    answer = ask(reference, "send a referral")
    assert "Thoth refused this call" in answer
    assert reference.tools.executed == []


def test_the_agentcore_contract_ping_and_invocations(reference, server):
    http = ThreadingHTTPServer(("127.0.0.1", 0), reference.agent.make_handler(stub=True))
    threading.Thread(target=http.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{http.server_address[1]}"
        assert httpx.get(f"{base}/ping").json() == {"status": "Healthy"}
        reply = httpx.post(
            f"{base}/invocations",
            json={"prompt": "list appointments"},
            headers={reference.agent.SESSION_HEADER: "session-2"},
            timeout=30,
        )
        assert reply.status_code == 200
        assert "cardiology follow-up" in reply.json()["response"]
        assert httpx.post(f"{base}/invocations", content=b"not json").status_code == 400
        assert httpx.post(f"{base}/invocations", json={}).json() == {"error": "send a JSON body with a prompt"}
    finally:
        http.shutdown()


def test_the_key_is_read_from_secrets_manager_only_when_not_already_set(reference, monkeypatch):
    class Secrets:
        def __init__(self):
            self.calls = []

        def get_secret_value(self, **kwargs: str):
            self.calls.append(kwargs["SecretId"])
            return {"SecretString": "thoth_from_secret\n"}

    secrets = Secrets()
    monkeypatch.delenv("THOTH_API_KEY", raising=False)
    monkeypatch.delenv("THOTH_API_KEY_SECRET_ID", raising=False)
    reference.agent.load_api_key_from_secrets_manager(secrets)
    assert secrets.calls == []

    monkeypatch.setenv("THOTH_API_KEY_SECRET_ID", "thoth/clinic-scheduler")
    reference.agent.load_api_key_from_secrets_manager(secrets)
    assert secrets.calls == ["thoth/clinic-scheduler"]
    assert __import__("os").environ["THOTH_API_KEY"] == "thoth_from_secret"
    monkeypatch.delenv("THOTH_API_KEY")

    monkeypatch.setenv("THOTH_API_KEY", "already-set")
    reference.agent.load_api_key_from_secrets_manager(secrets)
    assert secrets.calls == ["thoth/clinic-scheduler"]
