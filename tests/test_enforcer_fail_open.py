from __future__ import annotations

import httpx

from thoth.enforcer_client import EnforcerClient
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
