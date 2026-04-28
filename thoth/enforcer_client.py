# thoth/enforcer_client.py
from __future__ import annotations

import logging
from typing import Any

import httpx

from thoth.http_diagnostics import auth_failure_hint, extract_http_error_detail
from thoth.logging_config import configure_thoth_logging_from_env
from thoth.models import DecisionType, EnforcementDecision, ThothConfig

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=2.0)
_FALLBACK = EnforcementDecision(
    decision=DecisionType.BLOCK,
    reason="enforcer unavailable",
)


def _blocked_with_reason(reason: str) -> EnforcementDecision:
    return EnforcementDecision(decision=DecisionType.BLOCK, reason=reason)


class EnforcerClient:
    def __init__(self, config: ThothConfig) -> None:
        configure_thoth_logging_from_env()
        self._config = config
        headers = {"x-api-key": config.api_key} if config.api_key else {}
        # resolved_enforcer_url follows the single-URL contract and mirrors resolved_api_url.
        enforcer_url = config.resolved_enforcer_url
        self._http = httpx.Client(base_url=enforcer_url, headers=headers, timeout=_TIMEOUT)
        self._async_http = httpx.AsyncClient(base_url=enforcer_url, headers=headers, timeout=_TIMEOUT)

    def _payload(
        self,
        tool_name: str,
        session_id: str,
        tool_calls: list[str],
        tool_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        trace_id = self._config.enforcement_trace_id or session_id
        payload: dict[str, Any] = {
            "agent_id": self._config.agent_id,
            "tenant_id": self._config.tenant_id,
            "user_id": self._config.user_id,
            "tool_name": tool_name,
            "session_id": session_id,
            "session_tool_calls": tool_calls,
            "approved_scope": self._config.approved_scope,
            "enforcement_mode": self._config.enforcement.value,
            "environment": self._config.environment,
            "enforcement_trace_id": trace_id,
        }
        if tool_args is not None:
            payload["tool_args"] = tool_args
        if self._config.session_intent is not None:
            payload["session_intent"] = self._config.session_intent
        return payload

    def check(
        self,
        tool_name: str,
        session_id: str,
        tool_calls: list[str],
        tool_args: dict[str, Any] | None = None,
    ) -> EnforcementDecision:
        """Synchronous enforce call. Fail-closed -- returns BLOCK on any error."""
        try:
            resp = self._http.post("/v1/enforce", json=self._payload(tool_name, session_id, tool_calls, tool_args=tool_args))
            resp.raise_for_status()
            return EnforcementDecision.model_validate(resp.json())
        except httpx.HTTPStatusError as exc:
            response = exc.response
            detail = extract_http_error_detail(response)
            hint = auth_failure_hint(response.status_code, detail)
            logger.error(
                "thoth: enforcer rejected request (status=%s url=%s tool=%s detail=%s)%s",
                response.status_code,
                str(response.request.url),
                tool_name,
                detail,
                f" hint={hint}" if hint else "",
                exc_info=True,
            )
            return _blocked_with_reason(
                f"enforcer rejected request (status={response.status_code})"
            )
        except Exception:
            logger.error(
                "thoth: enforcer unreachable, fail-closed fallback to BLOCK for tool=%s",
                tool_name,
                exc_info=True,
            )
            return _FALLBACK

    async def acheck(
        self,
        tool_name: str,
        session_id: str,
        tool_calls: list[str],
        tool_args: dict[str, Any] | None = None,
    ) -> EnforcementDecision:
        """Async enforce call. Fail-closed -- returns BLOCK on any error."""
        try:
            resp = await self._async_http.post("/v1/enforce", json=self._payload(tool_name, session_id, tool_calls, tool_args=tool_args))
            resp.raise_for_status()
            return EnforcementDecision.model_validate(resp.json())
        except httpx.HTTPStatusError as exc:
            response = exc.response
            detail = extract_http_error_detail(response)
            hint = auth_failure_hint(response.status_code, detail)
            logger.error(
                "thoth: enforcer rejected request (async, status=%s url=%s tool=%s detail=%s)%s",
                response.status_code,
                str(response.request.url),
                tool_name,
                detail,
                f" hint={hint}" if hint else "",
                exc_info=True,
            )
            return _blocked_with_reason(
                f"enforcer rejected request (status={response.status_code})"
            )
        except Exception:
            logger.error(
                "thoth: enforcer unreachable (async), fail-closed fallback to BLOCK for tool=%s",
                tool_name,
                exc_info=True,
            )
            return _FALLBACK

    def close(self) -> None:
        self._http.close()

    async def aclose(self) -> None:
        await self._async_http.aclose()
