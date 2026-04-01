# thoth/enforcer_client.py
from __future__ import annotations

import logging
from typing import Any

import httpx

from thoth.models import DecisionType, EnforcementDecision, ThothConfig

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=2.0)
_FALLBACK = EnforcementDecision(decision=DecisionType.ALLOW)


class EnforcerClient:
    def __init__(self, config: ThothConfig) -> None:
        self._config = config
        headers = {"x-api-key": config.api_key} if config.api_key else {}
        # resolved_enforcer_url is a @property that checks THOTH_ENFORCER_URL at call time,
        # so the client is built with the correct URL even when the env var is set after import.
        enforcer_url = config.resolved_enforcer_url
        self._http = httpx.Client(base_url=enforcer_url, headers=headers, timeout=_TIMEOUT)
        self._async_http = httpx.AsyncClient(base_url=enforcer_url, headers=headers, timeout=_TIMEOUT)

    def _payload(self, tool_name: str, session_id: str, tool_calls: list[str]) -> dict[str, Any]:
        return {
            "agent_id": self._config.agent_id,
            "tenant_id": self._config.tenant_id,
            "user_id": self._config.user_id,
            "tool_name": tool_name,
            "session_id": session_id,
            "session_tool_calls": tool_calls,
            "approved_scope": self._config.approved_scope,
            "enforcement_mode": self._config.enforcement.value,
        }

    def check(
        self,
        tool_name: str,
        session_id: str,
        tool_calls: list[str],
    ) -> EnforcementDecision:
        """Synchronous enforce call. Non-fatal -- returns ALLOW on any error."""
        try:
            resp = self._http.post("/v1/enforce", json=self._payload(tool_name, session_id, tool_calls))
            resp.raise_for_status()
            return EnforcementDecision.model_validate(resp.json())
        except Exception:
            logger.warning(
                "thoth: enforcer unreachable, falling back to ALLOW for %s",
                tool_name,
                exc_info=True,
            )
            return _FALLBACK

    async def acheck(
        self,
        tool_name: str,
        session_id: str,
        tool_calls: list[str],
    ) -> EnforcementDecision:
        """Async enforce call. Non-fatal -- returns ALLOW on any error."""
        try:
            resp = await self._async_http.post("/v1/enforce", json=self._payload(tool_name, session_id, tool_calls))
            resp.raise_for_status()
            return EnforcementDecision.model_validate(resp.json())
        except Exception:
            logger.warning(
                "thoth: enforcer unreachable (async), falling back to ALLOW for %s",
                tool_name,
                exc_info=True,
            )
            return _FALLBACK

    def close(self) -> None:
        self._http.close()

    async def aclose(self) -> None:
        await self._async_http.aclose()
