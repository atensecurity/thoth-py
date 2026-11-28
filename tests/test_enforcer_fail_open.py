from __future__ import annotations

import datetime

import httpx

import thoth
from thoth import usage
from thoth.enforcer_client import MAX_REQUEST_BYTES, EnforcerClient
from thoth.models import ThothConfig


class _StubHTTPClient:
    def __init__(self, response: httpx.Response | None = None, exc: Exception | None = None) -> None:
        self._response = response
        self._exc = exc

    def post(self, *args, **kwargs):  # noqa: ANN002, ANN003
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return self._response


def _config(*, fail_open: bool) -> ThothConfig:
    return ThothConfig(
        agent_id="test-agent",
        approved_scope=["read:data"],
        tenant_id="trantor",
        api_url="https://enforce.example.com",
        fail_open=fail_open,
    )


def test_fail_open_allows_on_transport_error() -> None:
    client = EnforcerClient(_config(fail_open=True))
    client._http = _StubHTTPClient(exc=httpx.TimeoutException("timeout"))  # type: ignore[assignment]

    decision = client.check("read:data", session_id="sess-1", tool_calls=[])

    assert decision.is_allow
    assert "fail-open" in (decision.reason or "").lower()


def test_fail_open_allows_on_retryable_status() -> None:
    request = httpx.Request("POST", "https://enforce.example.com/v1/enforce")
    response = httpx.Response(503, request=request, json={"error": "upstream unavailable"})

    client = EnforcerClient(_config(fail_open=True))
    client._http = _StubHTTPClient(response=response)  # type: ignore[assignment]

    decision = client.check("read:data", session_id="sess-1", tool_calls=[])

    assert decision.is_allow
    assert "status=503" in (decision.reason or "")


def test_fail_open_still_blocks_on_auth_failure() -> None:
    request = httpx.Request("POST", "https://enforce.example.com/v1/enforce")
    response = httpx.Response(403, request=request, json={"error": "forbidden"})

    client = EnforcerClient(_config(fail_open=True))
    client._http = _StubHTTPClient(response=response)  # type: ignore[assignment]

    decision = client.check("read:data", session_id="sess-1", tool_calls=[])

    assert decision.is_block
    assert "status=403" in (decision.reason or "")


def test_request_too_large_blocks_even_when_fail_open() -> None:
    """413 is the enforcer refusing an oversized request: not an outage, so it never fails open."""
    request = httpx.Request("POST", "https://enforce.example.com/v1/enforce")
    response = httpx.Response(413, request=request, json={"error": "request_too_large", "limit_bytes": 4194304})

    for fail_open in (True, False):
        client = EnforcerClient(_config(fail_open=fail_open))
        client._http = _StubHTTPClient(response=response)  # type: ignore[assignment]

        decision = client.check("read:data", session_id="sess-1", tool_calls=[])

        assert decision.is_block
        assert "status=413" in (decision.reason or "")


def _billing_response() -> httpx.Response:
    request = httpx.Request("POST", "https://enforce.example.com/v1/enforce")
    body = {
        "type": "error",
        "error": {
            "type": "billing_error",
            "code": "usage_limit_reached",
            "message": "Your Thoth workspace has used its governed actions for this period. Upgrade your plan or buy more actions at https://app.atensecurity.com/settings?section=billing.",
            "upgrade_url": "https://app.atensecurity.com/settings?section=billing",
        },
    }
    return httpx.Response(402, request=request, json=body)


def test_billing_error_blocks_with_its_message_even_when_fail_open() -> None:
    for fail_open in (True, False):
        client = EnforcerClient(_config(fail_open=fail_open))
        client._http = _StubHTTPClient(response=_billing_response())  # type: ignore[assignment]

        decision = client.check("read:data", session_id="sess-1", tool_calls=[])

        assert decision.is_block
        assert decision.reason.startswith("Your Thoth workspace has used its governed actions")
        assert decision.decision_reason_code == "billing_usage_limit_reached"


def test_a_402_without_a_billing_error_is_an_ordinary_rejection() -> None:
    request = httpx.Request("POST", "https://enforce.example.com/v1/enforce")
    client = EnforcerClient(_config(fail_open=True))
    client._http = _StubHTTPClient(response=httpx.Response(402, request=request, text="payment required"))  # type: ignore[assignment]

    decision = client.check("read:data", session_id="sess-1", tool_calls=[])

    assert decision.is_block
    assert "status=402" in (decision.reason or "")


def test_oversized_request_blocks_locally_and_never_sends_even_when_fail_open() -> None:
    class _Raises:
        def post(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("an oversized request must not be sent")

    for fail_open in (True, False):
        client = EnforcerClient(_config(fail_open=fail_open))
        client._http = _Raises()  # type: ignore[assignment]

        decision = client.check("read:data", session_id="sess-1", tool_calls=[], tool_args={"blob": "a" * (MAX_REQUEST_BYTES + 1)})

        assert decision.is_block
        assert "enforcer_request_too_large" in (decision.reason or "")


class _AsyncStubHTTPClient:
    def __init__(self, response: httpx.Response | None = None, exc: Exception | None = None) -> None:
        self._response = response
        self._exc = exc

    async def post(self, *args, **kwargs):  # noqa: ANN002, ANN003
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return self._response


def _ok_response(**kwargs) -> httpx.Response:  # noqa: ANN003
    request = httpx.Request("POST", "https://enforce.example.com/v1/enforce")
    return httpx.Response(200, request=request, **kwargs)


_UNPARSEABLE_BODIES = {
    "decision_not_a_string": {"json": {"decision": 5}},
    "decision_true": {"json": {"decision": True}},
    "decision_null": {"json": {"decision": None}},
    "numeric_string_for_number": {"json": {"decision": "ALLOW", "risk_score": "5"}},
    "string_for_int": {"json": {"decision": "ALLOW", "step_up_timeout_seconds": "60"}},
    "unknown_decision": {"json": {"decision": "NOPE"}},
    "invalid_json": {"content": b"<html>not json</html>"},
    "empty_object": {"json": {}},
    # These two raise in the model (the first three already normalise to BLOCK there).
    "json_array": {"json": [1, 2]},
    "wrong_field_type": {"json": {"decision": "ALLOW", "risk_score": "high"}},
}


# The model already turns these into BLOCK itself; they stay here as regression guards.
_NORMALISED_BY_MODEL = {"unknown_decision"}


def _drain_usage() -> None:
    while usage.take_pending_usage() is not None:
        pass


def test_unparseable_success_response_blocks_even_when_fail_open_sync() -> None:
    """A 2xx body the SDK cannot read is a contract failure, not an outage: it never fails open."""
    for name, body in _UNPARSEABLE_BODIES.items():
        for fail_open in (True, False):
            client = EnforcerClient(_config(fail_open=fail_open))
            client._http = _StubHTTPClient(response=_ok_response(**body))  # type: ignore[assignment]

            decision = client.check("read:data", session_id="sess-1", tool_calls=[])

            assert decision.is_block, (name, fail_open)
            if name not in _NORMALISED_BY_MODEL:
                assert "not understood" in (decision.reason or "")


async def test_unparseable_success_response_blocks_even_when_fail_open_async() -> None:
    for name, body in _UNPARSEABLE_BODIES.items():
        for fail_open in (True, False):
            client = EnforcerClient(_config(fail_open=fail_open))
            client._async_http = _AsyncStubHTTPClient(response=_ok_response(**body))  # type: ignore[assignment]

            decision = await client.acheck("read:data", session_id="sess-1", tool_calls=[])

            assert decision.is_block, (name, fail_open)
            if name not in _NORMALISED_BY_MODEL:
                assert "not understood" in (decision.reason or "")


async def test_fail_open_still_allows_on_connect_error_and_503_async() -> None:
    client = EnforcerClient(_config(fail_open=True))
    client._async_http = _AsyncStubHTTPClient(exc=httpx.ConnectError("refused"))  # type: ignore[assignment]
    assert (await client.acheck("read:data", session_id="sess-1", tool_calls=[])).is_allow

    request = httpx.Request("POST", "https://enforce.example.com/v1/enforce")
    client._async_http = _AsyncStubHTTPClient(response=httpx.Response(503, request=request, json={}))  # type: ignore[assignment]
    assert (await client.acheck("read:data", session_id="sess-1", tool_calls=[])).is_allow


def test_fail_open_still_allows_on_connect_error_sync() -> None:
    client = EnforcerClient(_config(fail_open=True))
    client._http = _StubHTTPClient(exc=httpx.ConnectError("refused"))  # type: ignore[assignment]

    assert client.check("read:data", session_id="sess-1", tool_calls=[]).is_allow


def test_unparseable_success_response_consumes_pending_usage_and_logs_no_body(caplog) -> None:

    _drain_usage()
    thoth.report_usage(7, 3)
    client = EnforcerClient(_config(fail_open=True))
    client._http = _StubHTTPClient(response=_ok_response(content=b"secret-body-content"))  # type: ignore[assignment]

    with caplog.at_level("ERROR", logger="thoth"):
        decision = client.check("read:data", session_id="sess-1", tool_calls=[])

    assert decision.is_block
    # The server answered 2xx and counted the usage: resending it would double-count.
    assert usage.take_pending_usage() is None
    assert "secret-body-content" not in caplog.text
    assert "status=200" in caplog.text


async def test_unparseable_success_response_consumes_pending_usage_async() -> None:

    _drain_usage()
    thoth.report_usage(5, 2)
    client = EnforcerClient(_config(fail_open=True))
    client._async_http = _AsyncStubHTTPClient(response=_ok_response(json={"decision": "ALLOW", "risk_score": "high"}))  # type: ignore[assignment]

    decision = await client.acheck("read:data", session_id="sess-1", tool_calls=[])

    assert decision.is_block
    assert usage.take_pending_usage() is None


class _NeverCalledHTTP:
    def post(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("a request that could not be built must not be sent")


class _NeverCalledAsyncHTTP:
    async def post(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("a request that could not be built must not be sent")


_UNBUILDABLE_ARGS = {
    "nan": {"x": float("nan")},
    "infinity": {"x": float("inf")},
    "datetime": {"x": datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)},
    "bytes": {"x": b"abc"},
    "set": {"x": {1, 2}},
}


def test_request_that_cannot_be_built_blocks_even_when_fail_open_sync(caplog) -> None:
    for name, args in _UNBUILDABLE_ARGS.items():
        for fail_open in (True, False):
            _drain_usage()
            thoth.report_usage(4, 1)
            client = EnforcerClient(_config(fail_open=fail_open))
            client._http = _NeverCalledHTTP()  # type: ignore[assignment]

            with caplog.at_level("ERROR", logger="thoth"):
                decision = client.check("read:data", session_id="sess-1", tool_calls=[], tool_args=args)

            assert decision.is_block, (name, fail_open)
            assert "request could not be built" in (decision.reason or "")
            # The server never saw it: the usage rides on the next call.
            assert usage.take_pending_usage() == {"prompt_tokens": 4, "completion_tokens": 1}, (name, fail_open)
    assert "abc" not in caplog.text


async def test_request_that_cannot_be_built_blocks_even_when_fail_open_async() -> None:
    for name, args in _UNBUILDABLE_ARGS.items():
        for fail_open in (True, False):
            _drain_usage()
            thoth.report_usage(4, 1)
            client = EnforcerClient(_config(fail_open=fail_open))
            client._async_http = _NeverCalledAsyncHTTP()  # type: ignore[assignment]

            decision = await client.acheck("read:data", session_id="sess-1", tool_calls=[], tool_args=args)

            assert decision.is_block, (name, fail_open)
            assert "request could not be built" in (decision.reason or "")
            assert usage.take_pending_usage() == {"prompt_tokens": 4, "completion_tokens": 1}, (name, fail_open)


def _status_response(status: int) -> httpx.Response:
    request = httpx.Request("POST", "https://enforce.example.com/v1/enforce")
    return httpx.Response(status, request=request, json={"error": "x"})


def test_fail_closed_outage_statuses_say_unavailable_not_rejected_sync() -> None:
    for status in (429, 500, 503):
        client = EnforcerClient(_config(fail_open=False))
        client._http = _StubHTTPClient(response=_status_response(status))  # type: ignore[assignment]

        decision = client.check("read:data", session_id="sess-1", tool_calls=[])

        assert decision.is_block
        assert decision.reason == f"enforcer unavailable (status={status})"


def test_fail_closed_client_error_statuses_still_say_rejected_sync() -> None:
    for status in (400, 401, 403):
        client = EnforcerClient(_config(fail_open=False))
        client._http = _StubHTTPClient(response=_status_response(status))  # type: ignore[assignment]

        decision = client.check("read:data", session_id="sess-1", tool_calls=[])

        assert decision.is_block
        assert decision.reason == f"enforcer rejected request (status={status})"


async def test_fail_closed_statuses_reasons_async() -> None:
    for status, reason in (
        (429, "enforcer unavailable (status=429)"),
        (503, "enforcer unavailable (status=503)"),
        (401, "enforcer rejected request (status=401)"),
    ):
        client = EnforcerClient(_config(fail_open=False))
        client._async_http = _AsyncStubHTTPClient(response=_status_response(status))  # type: ignore[assignment]

        decision = await client.acheck("read:data", session_id="sess-1", tool_calls=[])

        assert decision.is_block
        assert decision.reason == reason


def _deep(depth: int) -> dict:
    node: dict = {"leaf": 1}
    for _ in range(depth):
        node = {"n": [node]}
    return node


def test_very_deep_tool_args_block_and_keep_usage_sync() -> None:
    for fail_open in (True, False):
        _drain_usage()
        thoth.report_usage(4, 1)
        client = EnforcerClient(_config(fail_open=fail_open))
        client._http = _NeverCalledHTTP()  # type: ignore[assignment]

        decision = client.check("read:data", session_id="sess-1", tool_calls=[], tool_args=_deep(5000))

        assert decision.is_block, fail_open
        assert "request could not be built" in (decision.reason or "")
        assert usage.take_pending_usage() == {"prompt_tokens": 4, "completion_tokens": 1}


async def test_very_deep_tool_args_block_and_keep_usage_async() -> None:
    for fail_open in (True, False):
        _drain_usage()
        thoth.report_usage(4, 1)
        client = EnforcerClient(_config(fail_open=fail_open))
        client._async_http = _NeverCalledAsyncHTTP()  # type: ignore[assignment]

        decision = await client.acheck("read:data", session_id="sess-1", tool_calls=[], tool_args=_deep(5000))

        assert decision.is_block, fail_open
        assert "request could not be built" in (decision.reason or "")
        assert usage.take_pending_usage() == {"prompt_tokens": 4, "completion_tokens": 1}
