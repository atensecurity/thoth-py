from __future__ import annotations

from typing import Any

import httpx


def extract_http_error_detail(response: httpx.Response) -> str:
    """Extract a concise error detail from JSON or text responses."""
    detail = _extract_json_detail(response)
    if detail:
        return detail

    raw_text = (response.text or "").strip()
    if not raw_text:
        return "<empty>"
    return raw_text[:512]


def auth_failure_hint(status_code: int, detail: str) -> str | None:
    if status_code not in (401, 403):
        return None

    normalized = detail.lower()
    if status_code == 403 and "<html" in normalized:
        return (
            "403 HTML response usually means an ingress/WAF block before enforcer auth executes. "
            "Check ALB/WAF metrics and exclude /v1/events* telemetry paths from managed body-inspection rules "
            "while keeping auth, IP reputation, and scoped rate limits enabled."
        )
    if any(token in normalized for token in ("expired", "invalid", "forbidden", "unauthorized", "scope")):
        return (
            "Thoth API keys authenticate each request directly (no refresh token flow). "
            "Re-issue or re-scope the key with `thothctl api-keys create ... --permission execute` "
            "and update THOTH_API_KEY/THOTH_API_KEY_FILE on the endpoint."
        )
    return "Auth was rejected. Verify endpoint/fleet/agent scope, key TTL, and tenant binding. Thoth API keys are used directly per request and must be rotated when expired."


def _extract_json_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        return ""

    if isinstance(payload, dict):
        for key in ("message", "error_description", "error", "reason", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:512]
        if isinstance(payload.get("errors"), list):
            errors = payload["errors"]
            if errors and isinstance(errors[0], dict):
                first_error = errors[0]
                for key in ("message", "error", "reason", "detail"):
                    value = first_error.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()[:512]

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        first_item: dict[str, Any] = payload[0]
        for key in ("message", "error", "reason", "detail"):
            value = first_item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:512]

    return ""
