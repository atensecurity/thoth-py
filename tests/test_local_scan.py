"""Local scan: the shared golden fixtures, the pinned ruleset hash, the wire report, and failing closed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from thoth import local_scan as ls
from thoth.models import ThothConfig

from .test_data_handling import ARGS, CANARY, SALT, Wire, make_client, make_config

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = json.loads((FIXTURES / "detectors_golden.v1.json").read_text(encoding="utf-8"))
SPEC = json.loads(ls.SPEC_PATH.read_text(encoding="utf-8"))
FEATURES = Path(__file__).resolve().parents[2] / "features"

# Changing any detector changes this value on purpose. It is the same pin the enforcer tests hold;
# update the spec, the golden, both vendored copies and both pins in one pull request.
PINNED_RULESET_HASH = "3d4388b164bcf97a8823d2e6d3e03a6f5b10724d10faa539d59bcb38590eda44"
ENV = "THOTH_DATA_HANDLING_LOCAL_SCAN"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("THOTH_DATA_HANDLING_TOOL_ARGUMENTS", "THOTH_DATA_HANDLING_PROMPTS", "THOTH_DATA_HANDLING_SALT", ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def scanner() -> ls.LocalScanner:
    return ls.LocalScanner(SPEC, version="9.9.9")


# --------------------------------------------------------------------------- #
# Identity and packaging                                                       #
# --------------------------------------------------------------------------- #


def test_the_scanner_carries_the_pinned_ruleset_hash(scanner):
    assert ls.compute_ruleset_hash(SPEC) == SPEC["ruleset_hash"] == PINNED_RULESET_HASH
    assert scanner.ruleset_hash == PINNED_RULESET_HASH
    assert GOLDEN["ruleset_hash"] == PINNED_RULESET_HASH
    report = scanner.client_findings("tool", {"text": "hello"})
    assert report is not None
    assert report["scanner"] == {"name": "atensec-thoth", "version": "9.9.9", "ruleset_id": "thoth-detectors.v1", "ruleset_hash": PINNED_RULESET_HASH}


def test_a_spec_that_does_not_match_its_hash_is_refused():
    tampered = json.loads(json.dumps(SPEC))
    tampered["detectors"]["dlp"]["steps"][0]["pattern"] = "x"
    with pytest.raises(ls.LocalScanError):
        ls.LocalScanner(tampered)


def test_the_loaded_scanner_is_none_when_the_spec_cannot_be_read(monkeypatch):
    ls.load_scanner.cache_clear()
    monkeypatch.setattr(ls, "SPEC_PATH", Path("/nonexistent/detectors.json"))
    try:
        assert ls.load_scanner() is None
    finally:
        ls.load_scanner.cache_clear()


@pytest.mark.skipif(not FEATURES.is_dir(), reason="the shared features library is not in this checkout")
def test_vendored_copies_are_byte_identical_to_the_features_copies():
    assert ls.SPEC_PATH.read_bytes() == (FEATURES / "thoth_features" / "detectors.v1.json").read_bytes()
    assert (FIXTURES / "detectors_golden.v1.json").read_bytes() == (FEATURES / "fixtures" / "detectors_golden.v1.json").read_bytes()


def test_the_spec_ships_in_the_package_directory():
    assert ls.SPEC_PATH.parent.name == "thoth"
    assert ls.SPEC_PATH.is_file()


def test_the_module_imports_nothing_from_the_server():
    source = Path(ls.__file__).read_text(encoding="utf-8")
    for forbidden in ("enforcer", "thoth_features"):
        assert f"import {forbidden}" not in source and f"from {forbidden}" not in source


# --------------------------------------------------------------------------- #
# Golden fixtures                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("entry", GOLDEN["cases"], ids=lambda c: c["id"])
def test_golden_case(scanner, entry):
    # known_divergence cases record the Python server's Unicode answer; this scanner is Python, so it equals them too.
    assert scanner.scan(entry["tool_name"], entry["arguments"], entry.get("metadata")) == entry["expected"]


def test_golden_covers_every_group():
    assert {c["group"] for c in GOLDEN["cases"]} >= {"dlp", "bounds", "sensitive_path", "egress_host", "recipient", "negative", "known_divergence", "ambiguous"}


# --------------------------------------------------------------------------- #
# What leaves the process                                                      #
# --------------------------------------------------------------------------- #


def test_the_report_has_hostnames_and_domains_never_values_or_local_parts(scanner):
    args = {
        "to": "Very.Secret.Name+tag@Corp.Example.com",
        "url": "https://user:hunter2@Api.Example.org/path?token=abc",
        "path": "/data/patient_records.csv",
        "text": "ssn 123-45-6789 password=hunter2",
    }
    report = scanner.client_findings("send_email", args)
    text = json.dumps(report)
    for leaked in ("very.secret", "secret.name", "hunter2", "123-45-6789", "patient_records", "/data", "token=abc", "user:"):
        assert leaked not in text.lower()
    assert "corp.example.com" in text and "api.example.org" in text


def test_a_report_that_is_too_large_drops_checks_instead_of_being_sent():
    scanner = ls.LocalScanner(SPEC, version="1")
    domains = [f"a@{'x' * 63}.{'y' * 63}.{'z' * 63}.{'w' * 58}{i}.com" for i in range(10)]
    args = {"to": [*domains, *["Bob"] * 12], "url": "https://evil.example.com/"}
    full = scanner.client_findings("send_email", args)
    assert full is not None and "recipient" in full["scanned"]
    # The real bound is 4096 bytes; a smaller one proves the report is trimmed, not sent half.
    scanner._max_bytes = len(json.dumps(full, separators=(",", ":"))) - 1
    report = scanner.client_findings("send_email", args)
    assert report is not None
    assert len(json.dumps(report, separators=(",", ":")).encode()) <= scanner._max_bytes
    assert "recipient" not in report["scanned"] and all(f["kind"] != "recipient" for f in report["findings"])
    scanner._max_bytes = 10
    assert scanner.client_findings("send_email", args) is None


def test_non_dict_arguments_are_not_scanned(scanner):
    assert scanner.client_findings("tool", ["a", "b"]) is None
    assert scanner.client_findings("tool", None) is None


# --------------------------------------------------------------------------- #
# Failing closed                                                               #
# --------------------------------------------------------------------------- #


def test_a_check_that_raises_is_left_out_not_reported_clean(scanner, monkeypatch):
    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"cannot read {CANARY}")

    monkeypatch.setattr(scanner, "_sensitive_path", boom)
    result = scanner.scan("tool", {"path": "/x/patient.txt", "text": "123-45-6789"})
    assert "sensitive_path" not in result["scanned"]
    assert all(f["kind"] != "sensitive_path" for f in result["findings"])
    assert {"dlp", "egress_host", "recipient"} <= set(result["scanned"])
    assert [f for f in result["findings"] if f["kind"] == "dlp"] == [{"kind": "dlp", "category": "ssn", "count": 1}]


def test_every_check_failing_sends_no_report(scanner, monkeypatch):
    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("x")

    for name in ("_dlp", "_sensitive_path", "_egress", "_recipient_findings"):
        monkeypatch.setattr(scanner, name, boom)
    assert scanner.client_findings("tool", {"text": "x"}) is None


def test_a_truncated_walk_omits_dlp_and_recipient(scanner):
    result = scanner.scan("send_email", {"items": ["x"] * 6000, "to": "a@x.com"})
    assert "dlp" not in result["scanned"] and "recipient" not in result["scanned"]
    assert {"sensitive_path", "egress_host"} <= set(result["scanned"])


def test_a_self_referencing_argument_does_not_hang_or_raise(scanner):
    loop: dict[str, Any] = {"a": "123-45-6789"}
    loop["self"] = loop
    result = scanner.scan("tool", loop)
    assert "dlp" not in result["scanned"]


def test_a_scan_error_never_reaches_the_request_as_a_clean_report(monkeypatch):
    scanner = ls.load_scanner()
    assert scanner is not None

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("x")

    monkeypatch.setattr(scanner, "_dlp", boom)
    wire = Wire()
    client = make_client(wire, data_handling_tool_arguments="redact", data_handling_salt=SALT)
    client.check("read_file", "s", ["read_file"], tool_args={"text": "123-45-6789"})
    (body,) = wire.bodies("/v1/enforce")
    assert "dlp" not in body["client_findings"]["scanned"]
    assert body["client_findings"]["findings"] == []


# --------------------------------------------------------------------------- #
# On the wire                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["hash", "redact", "drop"])
def test_enforce_sends_client_findings_with_data_handling(mode):
    wire = Wire()
    client = make_client(wire, data_handling_tool_arguments=mode, data_handling_salt=SALT)
    args = {**ARGS, "email": "x", "url": "https://Evil.Example.com/x", "note": "ssn 123-45-6789"}
    client.check("read_file", "s", ["read_file"], tool_args=args)
    (body,) = wire.bodies("/v1/enforce")
    assert body["data_handling"]["tool_arguments"] == mode
    findings = body["client_findings"]
    assert findings["version"] == 1
    assert findings["scanner"]["ruleset_hash"] == PINNED_RULESET_HASH
    assert findings["scanned"] == ["dlp", "sensitive_path", "egress_host", "recipient"]
    assert findings["findings"] == [
        {"kind": "dlp", "category": "ssn", "count": 1},
        {"kind": "egress_host", "verdict": "not_allowlisted", "host": "evil.example.com"},
    ]
    assert CANARY not in wire.raw() and "123-45-6789" not in wire.raw()


def test_explain_carries_the_same_findings():
    from thoth.models import DecisionType, EnforcementDecision

    wire = Wire()
    client = make_client(wire, data_handling_tool_arguments="redact", data_handling_salt=SALT)
    decision = EnforcementDecision(decision=DecisionType.BLOCK, reason="r", violation_id="v1")
    client.explain(decision, tool_name="read_file", session_id="s", tool_calls=["read_file"], tool_args={"text": "123-45-6789"})
    (body,) = wire.bodies("/v1/explain")
    assert body["client_findings"]["findings"] == [{"kind": "dlp", "category": "ssn", "count": 1}]


@pytest.mark.asyncio
async def test_async_enforce_sends_client_findings():
    wire = Wire()
    client = make_client(wire, data_handling_tool_arguments="hash", data_handling_salt=SALT)
    await client.acheck("read_file", "s", ["read_file"], tool_args={"text": "123-45-6789"})
    (body,) = wire.bodies("/v1/enforce")
    assert body["client_findings"]["findings"] == [{"kind": "dlp", "category": "ssn", "count": 1}]


def test_a_call_with_no_arguments_still_reports_the_scan():
    wire = Wire()
    client = make_client(wire, data_handling_tool_arguments="drop")
    client.check("list_things", "s", [], tool_args=None)
    (body,) = wire.bodies("/v1/enforce")
    assert body["client_findings"]["scanned"] == ["dlp", "sensitive_path", "egress_host", "recipient"]
    assert body["client_findings"]["findings"] == []


def test_request_metadata_is_scanned_for_the_destination_host():
    wire = Wire()
    client = make_client(wire, data_handling_tool_arguments="redact", request_metadata={"url": "https://Meta.Example.net/x"})
    client.check("fetch", "s", [], tool_args={"q": "1"})
    (body,) = wire.bodies("/v1/enforce")
    assert {"kind": "egress_host", "verdict": "not_allowlisted", "host": "meta.example.net"} in body["client_findings"]["findings"]


def test_nothing_changes_when_data_handling_is_unset():
    wire = Wire()
    client = make_client(wire)
    client.check("read_file", "s", [], tool_args={"text": "123-45-6789"})
    (body,) = wire.bodies("/v1/enforce")
    assert "client_findings" not in body and "data_handling" not in body
    assert body["tool_args"] == {"text": "123-45-6789"}


def test_prompts_only_sends_arguments_and_no_findings():
    wire = Wire()
    client = make_client(wire, data_handling_prompts="hash", data_handling_salt=SALT)
    client.check("read_file", "s", [], tool_args={"text": "123-45-6789"})
    (body,) = wire.bodies("/v1/enforce")
    assert "client_findings" not in body and body["tool_args"] == {"text": "123-45-6789"}


@pytest.mark.parametrize("value", ["off", "false", "0", False])
def test_local_scan_can_be_turned_off(monkeypatch, value):
    if isinstance(value, str):
        monkeypatch.setenv(ENV, value)
        overrides: dict[str, Any] = {}
    else:
        overrides = {"data_handling_local_scan": value}
    wire = Wire()
    client = make_client(wire, data_handling_tool_arguments="redact", **overrides)
    client.check("read_file", "s", [], tool_args={"text": "123-45-6789"})
    (body,) = wire.bodies("/v1/enforce")
    assert "client_findings" not in body
    assert body["data_handling"]["tool_arguments"] == "redact"


def test_local_scan_is_on_by_default_and_an_unclear_value_raises(monkeypatch):
    assert make_config(data_handling_tool_arguments="hash").data_handling_local_scan is True
    monkeypatch.setenv(ENV, "maybe")
    with pytest.raises(ValueError):
        ThothConfig(agent_id="a", approved_scope=[], tenant_id="t", api_url="http://x.invalid")


def test_a_failed_scanner_load_sends_no_findings_but_still_withholds(monkeypatch):
    monkeypatch.setattr("thoth.enforcer_client.load_scanner", lambda: None)
    wire = Wire()
    client = make_client(wire, data_handling_tool_arguments="redact")
    client.check("read_file", "s", [], tool_args={"text": "123-45-6789"})
    (body,) = wire.bodies("/v1/enforce")
    assert "client_findings" not in body
    assert body["tool_args"] == {"text": "<redacted>"}
