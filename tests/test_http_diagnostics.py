import httpx

from thoth.http_diagnostics import auth_failure_hint, extract_http_error_detail


def test_extract_http_error_detail_from_json_message():
    response = httpx.Response(403, json={"error": "forbidden", "message": "api key expired"})
    assert extract_http_error_detail(response) == "api key expired"


def test_extract_http_error_detail_from_plain_text():
    response = httpx.Response(500, text="internal failure")
    assert extract_http_error_detail(response) == "internal failure"


def test_auth_failure_hint_for_forbidden_with_expiry_signal():
    hint = auth_failure_hint(403, "api key expired")
    assert hint is not None
    assert "no refresh token flow" in hint


def test_auth_failure_hint_ignored_for_non_auth_codes():
    assert auth_failure_hint(500, "boom") is None
