# thoth/step_up.py
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from thoth.models import DecisionType, EnforcementDecision, ThothConfig

logger = logging.getLogger(__name__)

_TIMEOUT_DECISION = EnforcementDecision(
    decision=DecisionType.BLOCK,
    reason="step-up auth timeout — no approver response",
)
_HTTP_TIMEOUT = httpx.Timeout(connect=2.0, read=6.0, write=2.0, pool=2.0)


def _coerce_hold_payload(payload: Any) -> EnforcementDecision:
    """
    Convert hold-status payloads into EnforcementDecision.

    Enforcer /v1/enforce/hold/{token} returns HoldToken shape:
      { resolved: bool, resolution: "ALLOW" | "BLOCK" | null, ... }
    Older clients may still receive direct decision-shaped payloads.
    """
    if isinstance(payload, dict):
        decision = payload.get("decision")
        if isinstance(decision, str):
            try:
                return EnforcementDecision(decision=DecisionType(decision), reason=payload.get("reason"))
            except ValueError:
                pass

        resolved = bool(payload.get("resolved"))
        resolution = payload.get("resolution")
        if resolved and isinstance(resolution, str):
            try:
                return EnforcementDecision(decision=DecisionType(resolution), reason=payload.get("reason"))
            except ValueError:
                pass
        if not resolved:
            return EnforcementDecision(decision=DecisionType.STEP_UP)

    return EnforcementDecision(decision=DecisionType.STEP_UP)


class StepUpClient:
    def __init__(self, config: ThothConfig) -> None:
        self._config = config
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        self._http = httpx.Client(base_url=config.resolved_enforcer_url, timeout=_HTTP_TIMEOUT, headers=headers)
        self._async_http = httpx.AsyncClient(base_url=config.resolved_enforcer_url, timeout=_HTTP_TIMEOUT, headers=headers)

    def wait(self, hold_token: str) -> EnforcementDecision:
        """Synchronous poll until approved/blocked or timeout. Returns BLOCK on timeout."""
        deadline = time.monotonic() + self._config.step_up_timeout_minutes * 60
        while time.monotonic() < deadline:
            try:
                resp = self._http.get(f"/v1/enforce/hold/{hold_token}")
                resp.raise_for_status()
                decision = _coerce_hold_payload(resp.json())
                if not decision.is_step_up:
                    return decision
            except Exception:
                logger.warning("thoth: error polling hold token %s", hold_token, exc_info=True)
            time.sleep(self._config.step_up_poll_interval_seconds)
        logger.warning("thoth: step-up auth timed out for hold_token=%s", hold_token)
        return _TIMEOUT_DECISION

    async def await_decision(self, hold_token: str) -> EnforcementDecision:
        """Async poll until approved/blocked or timeout. Does not block the event loop."""
        deadline = time.monotonic() + self._config.step_up_timeout_minutes * 60
        while time.monotonic() < deadline:
            try:
                resp = await self._async_http.get(f"/v1/enforce/hold/{hold_token}")
                resp.raise_for_status()
                decision = _coerce_hold_payload(resp.json())
                if not decision.is_step_up:
                    return decision
            except Exception:
                logger.warning("thoth: error polling hold token (async) %s", hold_token, exc_info=True)
            await asyncio.sleep(self._config.step_up_poll_interval_seconds)
        logger.warning("thoth: step-up auth timed out for hold_token=%s", hold_token)
        return _TIMEOUT_DECISION

    def close(self) -> None:
        self._http.close()

    async def aclose(self) -> None:
        await self._async_http.aclose()
