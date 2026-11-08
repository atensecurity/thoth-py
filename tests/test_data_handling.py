# tests/test_data_handling.py
"""Client-side drop/redact/hash: exact wire payloads, never-raw, config, MODIFY, salt."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading
from typing import Any

import httpx
from pydantic import ValidationError
import pytest
from thoth import ThothPolicyViolation
from thoth import data_handling as dh
from thoth.enforcer_client import EnforcerClient
from thoth.models import DecisionType, EnforcementDecision, EnforcementMode, ThothConfig
from thoth.session import SessionContext
from thoth.telemetry import telemetry_event
from thoth.tracer import ARGS_WITHHELD_NOTE, Tracer

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = json.loads((FIXTURES / "data_handling_golden.json").read_text())
UNSET_GOLDEN = json.loads((FIXTURES / "enforce_payload_unset_golden.json").read_text())
SALT = "test-salt-1"
CANARY = "CANARY-9f3a-secret-value"
ENV_NAMES = ("THOTH_DATA_HANDLING_TOOL_ARGUMENTS", "THOTH_DATA_HANDLING_PROMPTS", "THOTH_DATA_HANDLING_SALT")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


class Wire:
    """A fake transport that records every request body and answers with a fixed decision."""

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.response = response or {"decision": "ALLOW"}

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/v1/explain":
            return httpx.Response(
                200,
                json={
                    "violation_id": "v1",
                    "what_happened": "w",
                    "why_it_was_blocked": "y",
                    "what_to_do_next": "n",
                    "business_impact": "b",
                    "severity": "low",
                },
            )
        return httpx.Response(200, json=self.response)

    def bodies(self, path: str | None = None) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if path is None or r.url.path == path]

    def raw(self) -> str:
        return "\n".join(r.content.decode() for r in self.requests)


def make_config(**overrides: Any) -> ThothConfig:
    base: dict[str, Any] = {
        "agent_id": "agent-1",
        "approved_scope": ["read_file"],
        "tenant_id": "ws-1",
        "user_id": "user-1",
        "api_url": "http://enforcer.invalid",
        "enforcement_trace_id": "trace-1",
    }
    base.update(overrides)
    return ThothConfig(**base)


def make_client(wire: Wire, **overrides: Any) -> EnforcerClient:
    config = make_config(**overrides)
    http = httpx.Client(base_url="http://enforcer.invalid", transport=httpx.MockTransport(wire.handle))
    async_http = httpx.AsyncClient(base_url="http://enforcer.invalid", transport=httpx.MockTransport(wire.handle))
    return EnforcerClient(config, http=http, async_http=async_http)


ARGS = {"path": f"/home/{CANARY}.txt", "count": 3, "nested": {"token": CANARY, "items": [1, "two", None, True]}}


# --------------------------------------------------------------------------- #
# Golden fixture                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_golden_cases(case):
    mode, salt, value = case["mode"], case["salt"], case["input"]
    if case["kind"] == "tool_arguments":
        keep, out = dh.transform_arguments(value, mode, salt)
    elif case["kind"] == "prompt":
        keep, out = dh.transform_prompt(value, mode, salt)
    else:
        keep, out = True, dh.transform_context(value, mode, salt)
    if case.get("expected_omitted"):
        assert keep is False
    else:
        assert keep is True
        assert dh.canonical_json(out) == dh.canonical_json(case["expected"])


def test_golden_hash_matches_an_independent_hmac():
    import hashlib
    import hmac

    expected = "hmac-sha256:" + hmac.new(b"golden-salt-v1", b'"alice@example.com"', hashlib.sha256).hexdigest()
    assert dh.hash_value("golden-salt-v1", "alice@example.com") == expected


def test_golden_prompt_key_list_is_the_fixtures_list():
    assert sorted(dh.PROMPT_KEYS) == GOLDEN["format"]["context_prompt_keys"]


def test_input_is_not_mutated():
    value = {"a": {"b": [1, {"c": "x"}]}}
    snapshot = json.dumps(value)
    for mode in dh.TOOL_ARGUMENT_MODES:
        dh.transform_arguments(value, mode, SALT)
    assert json.dumps(value) == snapshot


def test_bounds_node_budget_and_depth():
    wide = {f"k{i}": f"secret-{i}" for i in range(dh.MAX_NODES + 50)}
    out = dh.redact_arguments(wide)
    assert len(out) <= dh.MAX_NODES + 1
    assert out["_truncated"] == dh.TRUNCATED
    assert "secret-" not in json.dumps(out)
    long_list = [f"secret-{i}" for i in range(dh.MAX_NODES + 50)]
    out_list = dh.hash_arguments(long_list, SALT)
    assert out_list[-1] == dh.TRUNCATED and "secret-" not in json.dumps(out_list)


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #


def test_unset_means_no_data_handling():
    config = make_config()
    assert config.data_handling is None
    assert config.arguments_withheld is False


def test_options_and_env_configure(monkeypatch):
    config = make_config(data_handling_tool_arguments="Redact", data_handling_prompts=" hash ", data_handling_salt="s")
    assert (config.data_handling_tool_arguments, config.data_handling_prompts) == ("redact", "hash")
    assert config.arguments_withheld is True
    monkeypatch.setenv("THOTH_DATA_HANDLING_TOOL_ARGUMENTS", "drop")
    monkeypatch.setenv("THOTH_DATA_HANDLING_PROMPTS", "drop")
    monkeypatch.setenv("THOTH_DATA_HANDLING_SALT", "env-salt")
    env_config = make_config()
    handling = env_config.data_handling
    assert handling is not None
    assert (handling.tool_arguments, handling.prompts, handling.salt) == ("drop", "drop", "env-salt")
    # An explicit option wins over the environment.
    assert make_config(data_handling_tool_arguments="hash").data_handling_tool_arguments == "hash"


def test_empty_values_count_as_unset(monkeypatch):
    monkeypatch.setenv("THOTH_DATA_HANDLING_TOOL_ARGUMENTS", "  ")
    assert make_config().data_handling is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"data_handling_tool_arguments": "store"},
        {"data_handling_tool_arguments": "everything"},
        {"data_handling_tool_arguments": 5},
        {"data_handling_prompts": "redact"},
        {"data_handling_prompts": "store"},
    ],
)
def test_invalid_options_raise_at_init(overrides):
    with pytest.raises(ValidationError):
        make_config(**overrides)


@pytest.mark.parametrize(
    ("variable", "value"),
    [("THOTH_DATA_HANDLING_TOOL_ARGUMENTS", "hasj"), ("THOTH_DATA_HANDLING_TOOL_ARGUMENTS", "store"), ("THOTH_DATA_HANDLING_PROMPTS", "redact")],
)
def test_invalid_env_raises_at_init(monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValidationError):
        make_config()


def test_invalid_env_raises_from_init_and_instrument(monkeypatch):
    import thoth
    from thoth.auto import shutdown

    monkeypatch.setenv("THOTH_DATA_HANDLING_TOOL_ARGUMENTS", "bogus")
    with pytest.raises(ValidationError):
        thoth.init(agent_id="a", tenant_id="t", api_key="k", frameworks=[])
    with pytest.raises(ValidationError):
        thoth.instrument_anthropic({}, agent_id="a", approved_scope=[], tenant_id="t")
    shutdown()


def test_init_passes_options_through():
    import thoth
    from thoth import auto

    try:
        thoth.init(agent_id="a", tenant_id="t", api_key="k", frameworks=[], data_handling_tool_arguments="redact", data_handling_salt="s")
        assert auto._RUNTIME is not None
        assert auto._RUNTIME.config.data_handling_tool_arguments == "redact"
    finally:
        auto.shutdown()


def test_the_salt_is_never_printed_or_dumped():
    config = make_config(data_handling_tool_arguments="hash", data_handling_salt="TOP-SECRET-SALT")
    assert "TOP-SECRET-SALT" not in repr(config)
    assert "TOP-SECRET-SALT" not in json.dumps(config.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# Exact wire payloads                                                         #
# --------------------------------------------------------------------------- #


def expected_args(mode: str, salt: str = SALT) -> Any:
    keep, out = dh.transform_arguments(ARGS, mode, salt)
    return out if keep else None


@pytest.mark.parametrize("mode", ["hash", "redact", "drop"])
def test_enforce_wire_payload_per_argument_mode(mode):
    wire = Wire()
    client = make_client(wire, data_handling_tool_arguments=mode, data_handling_salt=SALT)
    client.check("read_file", "sess-1", ["read_file"], tool_args=ARGS, action_attestation_id="att-1")
    (body,) = wire.bodies("/v1/enforce")
    assert body["data_handling"] == {"tool_arguments": mode, "applied_at": "client", "version": 1}
    if mode == "drop":
        assert "tool_args" not in body
    else:
        assert body["tool_args"] == expected_args(mode)
        assert set(body["tool_args"]) == set(ARGS)  # keys kept
    if mode == "redact":
        assert body["tool_args"] == {"path": "<redacted>", "count": "<redacted>", "nested": {"token": "<redacted>", "items": ["<redacted>"] * 4}}
    assert CANARY not in wire.raw()


def test_hash_is_hmac_with_the_given_salt_and_matches_within_a_salt_only():
    wire = Wire()
    one = make_client(wire, data_handling_tool_arguments="hash", data_handling_salt="salt-a")
    two = make_client(wire, data_handling_tool_arguments="hash", data_handling_salt="salt-b")
    one.check("t", "s", ["t"], tool_args={"v": "same"})
    one.check("t", "s", ["t"], tool_args={"v": "same"})
    two.check("t", "s", ["t"], tool_args={"v": "same"})
    a1, a2, b1 = (b["tool_args"]["v"] for b in wire.bodies("/v1/enforce"))
    assert a1 == a2 != b1
    assert a1 == dh.hash_value("salt-a", "same")


@pytest.mark.parametrize(("mode", "present"), [("hash", True), ("drop", False)])
def test_prompt_mode_on_context_fields(mode, present):
    wire = Wire()
    client = make_client(
        wire,
        data_handling_prompts=mode,
        data_handling_salt=SALT,
        task_context={"task_id": "t-1", "initiated_by": "alice", "prompt": f"secret {CANARY}"},
        auth_context={"scope": "read", "message": CANARY, "nested": {"history": [CANARY]}},
        delegation_context={"chain": ["a", "b"], "note": CANARY},
        identity_binding={"device": "d-1"},
        request_metadata={"trace": "x", "query": CANARY},
        session_intent="support",
        purpose="triage",
    )
    client.check("read_file", "sess-1", ["read_file"], tool_args=ARGS, action_attestation_id="att-1")
    (body,) = wire.bodies("/v1/enforce")
    assert body["data_handling"] == {"prompts": mode, "applied_at": "client", "version": 1}
    # Identifiers and developer labels are kept so policy still works.
    assert body["task_context"]["task_id"] == "t-1" and body["task_context"]["initiated_by"] == "alice"
    assert body["auth_context"]["scope"] == "read"
    assert body["delegation_context"]["chain"] == ["a", "b"]
    assert body["identity_binding"]["device"] == "d-1"
    assert (body["session_intent"], body["purpose"]) == ("support", "triage")
    assert ("prompt" in body["task_context"]) is present
    assert ("message" in body["auth_context"]) is present
    assert ("query" in body["metadata"]) is present
    if present:
        assert body["task_context"]["prompt"] == dh.hash_value(SALT, f"secret {CANARY}")
    # Tool arguments were not configured, so they pass through (the canary there is the app's own).
    assert body["tool_args"] == ARGS
    assert body["task_context"].get("prompt") != f"secret {CANARY}"
    assert CANARY not in json.dumps({k: v for k, v in body.items() if k != "tool_args"})


def test_both_modes_together_and_explain_is_covered():
    wire = Wire({"decision": "BLOCK", "reason": "no", "violation_id": "v1"})
    client = make_client(
        wire,
        data_handling_tool_arguments="redact",
        data_handling_prompts="hash",
        data_handling_salt=SALT,
        task_context={"prompt": CANARY},
    )
    decision = client.check("read_file", "s", ["read_file"], tool_args=ARGS, action_attestation_id="a")
    client.explain(decision, tool_name="read_file", session_id="s", tool_calls=["read_file"], tool_args=ARGS, action_attestation_id="a")
    enforce, explain = wire.bodies("/v1/enforce")[0], wire.bodies("/v1/explain")[0]
    for body in (enforce, explain):
        assert body["data_handling"] == {"tool_arguments": "redact", "prompts": "hash", "applied_at": "client", "version": 1}
        assert body["tool_args"]["path"] == "<redacted>"
        assert body["task_context"]["prompt"].startswith("hmac-sha256:")
    assert explain["decision"] == "BLOCK"
    assert explain["violation_id"] == "v1"
    assert CANARY not in wire.raw()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["hash", "redact", "drop"])
async def test_async_paths_apply_the_same_transform(mode):
    wire = Wire({"decision": "BLOCK", "reason": "no"})
    client = make_client(wire, data_handling_tool_arguments=mode, data_handling_prompts="drop", data_handling_salt=SALT, task_context={"prompt": CANARY})
    decision = await client.acheck("read_file", "s", ["read_file"], tool_args=ARGS, action_attestation_id="a")
    await client.aexplain(decision, tool_name="read_file", session_id="s", tool_calls=["read_file"], tool_args=ARGS, action_attestation_id="a")
    assert len(wire.requests) == 2
    for body in wire.bodies():
        assert body["data_handling"]["tool_arguments"] == mode
        assert body.get("tool_args") == (None if mode == "drop" else expected_args(mode))
        assert "prompt" not in body.get("task_context", {})
    assert CANARY not in wire.raw()


@pytest.mark.parametrize("mode", ["hash", "redact", "drop"])
def test_never_raw_when_the_request_fails_or_is_retried(mode):
    seen: list[bytes] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.content)
        return httpx.Response(503, json={"error": "down"})

    config = make_config(data_handling_tool_arguments=mode, data_handling_prompts="drop", data_handling_salt=SALT, task_context={"prompt": CANARY}, fail_open=True)
    client = EnforcerClient(
        config,
        http=httpx.Client(base_url="http://enforcer.invalid", transport=httpx.MockTransport(handle)),
        async_http=httpx.AsyncClient(base_url="http://enforcer.invalid", transport=httpx.MockTransport(handle)),
    )
    decision = client.check("read_file", "s", ["read_file"], tool_args=ARGS)
    assert decision.is_allow  # fail-open fallback; the request still left only transformed values
    assert seen and all(CANARY.encode() not in body for body in seen)


def test_unset_payload_is_unchanged_golden():
    wire = Wire()
    config = make_config(
        approved_scope=["read_file", "send_email"],
        session_intent="support",
        purpose="triage",
        data_classification="internal",
        task_context={"task_id": "t-1", "initiated_by": "alice", "prompt": "summarise my medical notes"},
        identity_binding={"device": "d-1"},
        auth_context={"scope": "read", "message": "hello"},
        delegation_context={"chain": ["a", "b"]},
        model_name="m",
        model_provider="p",
        request_metadata={"trace": "x"},
        environment="prod",
    )
    client = EnforcerClient(
        config,
        http=httpx.Client(base_url="http://enforcer.invalid", transport=httpx.MockTransport(wire.handle)),
        async_http=httpx.AsyncClient(base_url="http://enforcer.invalid", transport=httpx.MockTransport(wire.handle)),
    )
    args = UNSET_GOLDEN["tool_args"]
    client.check("read_file", "sess-1", ["read_file"], tool_args=args, action_attestation_id="att-1")
    client.check("read_file", "sess-1", ["read_file"], action_attestation_id="att-2")
    first, second = wire.bodies("/v1/enforce")
    assert first == UNSET_GOLDEN["payloads"]["enforce"]
    assert second == UNSET_GOLDEN["payloads"]["enforce_no_args"]
    assert "data_handling" not in first
    dec = EnforcementDecision(decision=DecisionType.STEP_UP, decision_reason_code="x", violation_id="v-1", risk_score=0.5)
    assert (
        client._explain_payload(dec, tool_name="read_file", session_id="sess-1", tool_calls=["read_file"], tool_args=args, action_attestation_id="att-1")
        == UNSET_GOLDEN["payloads"]["explain"]
    )


# --------------------------------------------------------------------------- #
# Tracer: MODIFY ignored, args_withheld surfaced, telemetry                   #
# --------------------------------------------------------------------------- #


class RecordingEmitter:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def emit(self, event: Any) -> None:
        self.events.append(event)


def make_tracer(wire: Wire, **overrides: Any) -> tuple[Tracer, RecordingEmitter]:
    config = make_config(enforcement=EnforcementMode.BLOCK, **overrides)
    enforcer = EnforcerClient(
        config,
        http=httpx.Client(base_url="http://enforcer.invalid", transport=httpx.MockTransport(wire.handle)),
        async_http=httpx.AsyncClient(base_url="http://enforcer.invalid", transport=httpx.MockTransport(wire.handle)),
    )
    emitter = RecordingEmitter()
    step_up = type("NoStepUp", (), {"wait": lambda self, token: EnforcementDecision(decision=DecisionType.BLOCK, reason="timeout")})()
    tracer = Tracer(config=config, session=SessionContext(config), emitter=emitter, enforcer=enforcer, step_up=step_up)  # type: ignore[arg-type]
    return tracer, emitter


MODIFY = {"decision": "ALLOW", "authorization_decision": "MODIFY", "modified_tool_args": {"input": "FROM-SERVER"}}


def test_modify_is_applied_when_nothing_is_withheld():
    tool_calls: list[Any] = []
    tracer, _ = make_tracer(Wire(MODIFY))
    wrapped = tracer.wrap_tool("read_file", lambda **kw: tool_calls.append(kw) or "ok")
    wrapped(input="mine")
    assert tool_calls == [{"input": "FROM-SERVER"}]


@pytest.mark.parametrize("mode", ["hash", "redact", "drop"])
def test_modify_is_ignored_when_arguments_are_withheld(mode):
    tool_calls: list[Any] = []
    tracer, _ = make_tracer(Wire(MODIFY), data_handling_tool_arguments=mode)
    wrapped = tracer.wrap_tool("read_file", lambda **kw: tool_calls.append(kw) or "ok")
    assert wrapped(input="mine") == "ok"
    assert tool_calls == [{"input": "mine"}]


@pytest.mark.asyncio
async def test_modify_is_ignored_when_arguments_are_withheld_async():
    tool_calls: list[Any] = []
    tracer, _ = make_tracer(Wire(MODIFY), data_handling_tool_arguments="hash")

    async def tool(**kw: Any) -> str:
        tool_calls.append(kw)
        return "ok"

    assert await tracer.wrap_tool("read_file", tool)(input="mine") == "ok"
    assert tool_calls == [{"input": "mine"}]


def test_args_withheld_step_up_is_explained_to_the_user():
    wire = Wire({"decision": "STEP_UP", "decision_reason_code": "args_withheld", "reason": "needs review", "violation_id": "v1"})
    tracer, _ = make_tracer(wire, data_handling_tool_arguments="drop")
    ran: list[Any] = []
    with pytest.raises(ThothPolicyViolation) as excinfo:
        tracer.wrap_tool("read_file", lambda **kw: ran.append(kw))(path=CANARY)
    assert ran == []
    assert excinfo.value.decision_reason_code == "args_withheld"
    assert "args_withheld" in str(excinfo.value)
    assert ARGS_WITHHELD_NOTE in str(excinfo.value)
    assert "THOTH_DATA_HANDLING_TOOL_ARGUMENTS" in str(excinfo.value)
    assert CANARY not in wire.raw()


def test_other_step_ups_get_no_args_withheld_note():
    wire = Wire({"decision": "STEP_UP", "decision_reason_code": "policy", "reason": "needs review"})
    tracer, _ = make_tracer(wire, data_handling_tool_arguments="drop")
    with pytest.raises(ThothPolicyViolation) as excinfo:
        tracer.wrap_tool("read_file", lambda **kw: None)(path="x")
    assert "args_withheld" not in str(excinfo.value)


@pytest.mark.parametrize("mode", ["hash", "redact", "drop"])
def test_nothing_raw_reaches_the_server_or_telemetry_through_the_tracer(mode):
    wire = Wire({"decision": "BLOCK", "reason": "no", "violation_id": "v1"})
    tracer, emitter = make_tracer(wire, data_handling_tool_arguments=mode, data_handling_prompts="drop", task_context={"prompt": CANARY})
    with pytest.raises(ThothPolicyViolation):
        tracer.wrap_tool("read_file", lambda **kw: "x")(**ARGS)
    assert {r.url.path for r in wire.requests} == {"/v1/enforce", "/v1/explain"}
    assert CANARY not in wire.raw()
    # Events carry the application's values in memory but the telemetry projection drops them.
    assert emitter.events
    for event in emitter.events:
        assert CANARY not in json.dumps(telemetry_event(event))


# --------------------------------------------------------------------------- #
# Salt                                                                        #
# --------------------------------------------------------------------------- #


def test_process_salt_is_random_and_shared_across_clients():
    first = make_config(data_handling_tool_arguments="hash").data_handling
    second = make_config(data_handling_tool_arguments="hash").data_handling
    assert first is not None and second is not None
    assert first.salt == second.salt and len(first.salt) >= 32
    assert first.salt != "golden-salt-v1"


def test_process_salt_is_safe_under_threads(monkeypatch):
    monkeypatch.setattr(dh, "_process_salt", None)
    start = threading.Barrier(16)
    seen: list[str] = []

    def worker() -> None:
        start.wait()
        seen.append(dh.process_salt())

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == 16 and len(set(seen)) == 1


@pytest.mark.asyncio
async def test_concurrent_calls_hash_consistently():
    wire = Wire()
    client = make_client(wire, data_handling_tool_arguments="hash")  # random per-process salt
    await asyncio.gather(*(client.acheck("t", "s", ["t"], tool_args={"v": "same"}) for _ in range(20)))
    hashes = {b["tool_args"]["v"] for b in wire.bodies("/v1/enforce")}
    assert len(hashes) == 1


def test_threaded_clients_share_one_handler_safely():
    wire = Wire()
    client = make_client(wire, data_handling_tool_arguments="redact")
    threads = [threading.Thread(target=lambda: client.check("t", "s", ["t"], tool_args={"v": CANARY})) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wire.requests) == 16 and CANARY not in wire.raw()
