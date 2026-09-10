# thoth/enforcer_client.py
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any
import uuid

import httpx

from thoth.http_diagnostics import auth_failure_hint, extract_http_error_detail
from thoth.logging_config import configure_thoth_logging_from_env
from thoth.models import DecisionType, EnforcementDecision, HumanExplanation, ThothConfig

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=2.0)
_FAIL_CLOSED_FALLBACK = EnforcementDecision(
    decision=DecisionType.BLOCK,
    reason="enforcer unavailable",
)
_FAIL_OPEN_FALLBACK = EnforcementDecision(
    decision=DecisionType.ALLOW,
    reason="enforcer unavailable (fail-open)",
)


def _blocked_with_reason(reason: str) -> EnforcementDecision:
    return EnforcementDecision(decision=DecisionType.BLOCK, reason=reason)


def _allowed_with_reason(reason: str) -> EnforcementDecision:
    return EnforcementDecision(decision=DecisionType.ALLOW, reason=reason)


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


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
        action_attestation_id: str | None = None,
    ) -> dict[str, Any]:
        trace_id = self._config.enforcement_trace_id or session_id
        attestation_id = action_attestation_id or self._config.action_attestation_id or str(uuid.uuid4())
        identity_binding: dict[str, Any] = {
            "agent_id": self._config.agent_id,
            "tenant_id": self._config.tenant_id,
            "user_id": self._config.user_id,
        }
        identity_binding.update(dict(self._config.identity_binding or {}))
        auth_context = dict(self._config.auth_context or {})
        metadata = dict(self._config.request_metadata or {})
        runtime_identity = (self._config.mcp_runtime_identity or "").strip()
        if runtime_identity:
            metadata.setdefault("mcp_runtime_identity", runtime_identity)
            auth_context.setdefault("service_identity", runtime_identity)

        payload: dict[str, Any] = {
            "agent_id": self._config.agent_id,
            "tenant_id": self._config.tenant_id,
            "user_id": self._config.user_id,
            "identity_binding": identity_binding,
            "tool_name": tool_name,
            "session_id": session_id,
            "session_tool_calls": tool_calls,
            "approved_scope": self._config.approved_scope,
            "enforcement_mode": self._config.enforcement.value,
            "environment": self._config.environment,
            "enforcement_trace_id": trace_id,
            "action_attestation_id": attestation_id,
        }
        if metadata:
            payload["metadata"] = metadata
        if tool_args is not None:
            payload["tool_args"] = tool_args
        if self._config.session_intent is not None:
            payload["session_intent"] = self._config.session_intent
        if self._config.purpose is not None:
            payload["purpose"] = self._config.purpose
        if self._config.data_classification is not None:
            payload["data_classification"] = self._config.data_classification
        if self._config.task_context:
            payload["task_context"] = self._config.task_context
        if self._config.model_name is not None:
            payload["model_name"] = self._config.model_name
        if self._config.model_provider is not None:
            payload["model_provider"] = self._config.model_provider
        if self._config.model_artifact_id is not None:
            payload["model_artifact_id"] = self._config.model_artifact_id
        if self._config.model_artifact_version is not None:
            payload["model_artifact_version"] = self._config.model_artifact_version
        if auth_context:
            payload["auth_context"] = auth_context
        if self._config.delegation_context:
            payload["delegation_context"] = dict(self._config.delegation_context)
        return payload

    def _explain_payload(
        self,
        decision: EnforcementDecision,
        *,
        tool_name: str,
        session_id: str,
        tool_calls: list[str],
        tool_args: dict[str, Any] | None = None,
        action_attestation_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = self._payload(
            tool_name=tool_name,
            session_id=session_id,
            tool_calls=tool_calls,
            tool_args=tool_args,
            action_attestation_id=action_attestation_id,
        )
        payload.update(
            {
                "decision": decision.decision.value,
                "decision_reason_code": decision.decision_reason_code,
                "action_classification": decision.action_classification,
                "violation_id": decision.violation_id,
                "risk_score": decision.risk_score,
                "regulatory_regimes": decision.regulatory_regimes,
                "matched_rule_ids": decision.matched_rule_ids,
                "matched_control_ids": decision.matched_control_ids,
                "policy_references": decision.policy_references,
                "step_up_timeout_seconds": decision.step_up_timeout_seconds,
            }
        )
        return payload

    def check(
        self,
        tool_name: str,
        session_id: str,
        tool_calls: list[str],
        tool_args: dict[str, Any] | None = None,
        action_attestation_id: str | None = None,
    ) -> EnforcementDecision:
        """Synchronous enforce call. Returns fallback decision on errors."""
        try:
            resp = self._http.post(
                "/v1/enforce",
                json=self._payload(
                    tool_name,
                    session_id,
                    tool_calls,
                    tool_args=tool_args,
                    action_attestation_id=action_attestation_id,
                ),
            )
            resp.raise_for_status()
            return EnforcementDecision.model_validate(resp.json())
        except httpx.HTTPStatusError as exc:
            response = exc.response
            detail = extract_http_error_detail(response)
            hint = auth_failure_hint(response.status_code, detail)
            if self._config.resolved_fail_open and _is_retryable_status(response.status_code):
                logger.warning(
                    "thoth: enforcer returned retryable status=%s, fail-open fallback to ALLOW for tool=%s detail=%s",
                    response.status_code,
                    tool_name,
                    detail,
                    exc_info=True,
                )
                return _allowed_with_reason(f"enforcer unavailable (status={response.status_code}, fail-open)")
            logger.error(
                "thoth: enforcer rejected request (status=%s url=%s tool=%s detail=%s)%s",
                response.status_code,
                str(response.request.url),
                tool_name,
                detail,
                f" hint={hint}" if hint else "",
                exc_info=True,
            )
            return _blocked_with_reason(f"enforcer rejected request (status={response.status_code})")
        except Exception:
            if self._config.resolved_fail_open:
                logger.warning(
                    "thoth: enforcer unreachable, fail-open fallback to ALLOW for tool=%s",
                    tool_name,
                    exc_info=True,
                )
                return _FAIL_OPEN_FALLBACK
            logger.error(
                "thoth: enforcer unreachable, fail-closed fallback to BLOCK for tool=%s",
                tool_name,
                exc_info=True,
            )
            return _FAIL_CLOSED_FALLBACK

    async def acheck(
        self,
        tool_name: str,
        session_id: str,
        tool_calls: list[str],
        tool_args: dict[str, Any] | None = None,
        action_attestation_id: str | None = None,
    ) -> EnforcementDecision:
        """Async enforce call. Returns fallback decision on errors."""
        try:
            resp = await self._async_http.post(
                "/v1/enforce",
                json=self._payload(
                    tool_name,
                    session_id,
                    tool_calls,
                    tool_args=tool_args,
                    action_attestation_id=action_attestation_id,
                ),
            )
            resp.raise_for_status()
            return EnforcementDecision.model_validate(resp.json())
        except httpx.HTTPStatusError as exc:
            response = exc.response
            detail = extract_http_error_detail(response)
            hint = auth_failure_hint(response.status_code, detail)
            if self._config.resolved_fail_open and _is_retryable_status(response.status_code):
                logger.warning(
                    "thoth: enforcer returned retryable status=%s (async), fail-open fallback to ALLOW for tool=%s detail=%s",
                    response.status_code,
                    tool_name,
                    detail,
                    exc_info=True,
                )
                return _allowed_with_reason(f"enforcer unavailable (status={response.status_code}, fail-open)")
            logger.error(
                "thoth: enforcer rejected request (async, status=%s url=%s tool=%s detail=%s)%s",
                response.status_code,
                str(response.request.url),
                tool_name,
                detail,
                f" hint={hint}" if hint else "",
                exc_info=True,
            )
            return _blocked_with_reason(f"enforcer rejected request (status={response.status_code})")
        except Exception:
            if self._config.resolved_fail_open:
                logger.warning(
                    "thoth: enforcer unreachable (async), fail-open fallback to ALLOW for tool=%s",
                    tool_name,
                    exc_info=True,
                )
                return _FAIL_OPEN_FALLBACK
            logger.error(
                "thoth: enforcer unreachable (async), fail-closed fallback to BLOCK for tool=%s",
                tool_name,
                exc_info=True,
            )
            return _FAIL_CLOSED_FALLBACK

    def explain(
        self,
        decision: EnforcementDecision,
        *,
        tool_name: str,
        session_id: str,
        tool_calls: list[str],
        tool_args: dict[str, Any] | None = None,
        action_attestation_id: str | None = None,
    ) -> HumanExplanation | None:
        try:
            resp = self._http.post(
                "/v1/explain",
                json=self._explain_payload(
                    decision,
                    tool_name=tool_name,
                    session_id=session_id,
                    tool_calls=tool_calls,
                    tool_args=tool_args,
                    action_attestation_id=action_attestation_id,
                ),
            )
            resp.raise_for_status()
            explanation = HumanExplanation.model_validate(resp.json())
            self._notify_webhook_background(explanation)
            return explanation
        except Exception:
            logger.debug(
                "thoth: explain request failed (sync) tool=%s violation_id=%s",
                tool_name,
                decision.violation_id,
                exc_info=True,
            )
            return None

    async def aexplain(
        self,
        decision: EnforcementDecision,
        *,
        tool_name: str,
        session_id: str,
        tool_calls: list[str],
        tool_args: dict[str, Any] | None = None,
        action_attestation_id: str | None = None,
    ) -> HumanExplanation | None:
        try:
            resp = await self._async_http.post(
                "/v1/explain",
                json=self._explain_payload(
                    decision,
                    tool_name=tool_name,
                    session_id=session_id,
                    tool_calls=tool_calls,
                    tool_args=tool_args,
                    action_attestation_id=action_attestation_id,
                ),
            )
            resp.raise_for_status()
            explanation = HumanExplanation.model_validate(resp.json())
            await self._anotify_webhook_background(explanation)
            return explanation
        except Exception:
            logger.debug(
                "thoth: explain request failed (async) tool=%s violation_id=%s",
                tool_name,
                decision.violation_id,
                exc_info=True,
            )
            return None

    def _notify_webhook_background(self, explanation: HumanExplanation) -> None:
        url = (self._config.notification_webhook_url or "").strip()
        if not url:
            return

        payload = explanation.model_dump(mode="json")

        def _send() -> None:
            try:
                httpx.post(url, json=payload, timeout=2.0)
            except Exception:
                logger.debug("thoth: notification webhook delivery failed", exc_info=True)

        threading.Thread(target=_send, daemon=True).start()

    async def _anotify_webhook_background(self, explanation: HumanExplanation) -> None:
        url = (self._config.notification_webhook_url or "").strip()
        if not url:
            return
        payload = explanation.model_dump(mode="json")

        async def _send() -> None:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    await client.post(url, json=payload)
            except Exception:
                logger.debug("thoth: async notification webhook delivery failed", exc_info=True)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(_send())

    def close(self) -> None:
        self._http.close()

    async def aclose(self) -> None:
        await self._async_http.aclose()
