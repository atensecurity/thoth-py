# thoth/exceptions.py


from typing import Any


class ThothPolicyViolation(Exception):  # noqa: N818
    def __init__(
        self,
        tool_name: str,
        reason: str,
        violation_id: str | None = None,
        decision_envelope_version: str | None = None,
        decision_reason_code: str | None = None,
        action_classification: str | None = None,
        authorization_decision: str | None = None,
        defer_timeout_seconds: int | None = None,
        step_up_timeout_seconds: int | None = None,
        enforcement_trace_id: str | None = None,
        action_attestation_id: str | None = None,
        fastml_features: dict[str, float] | None = None,
        score_components: dict[str, Any] | None = None,
        top_contributors: list[dict[str, Any]] | None = None,
        decision_evidence: dict[str, Any] | None = None,
        risk_score: float | None = None,
        latency_ms: float | None = None,
        pack_id: str | None = None,
        pack_version: str | None = None,
        rule_version: int | None = None,
        regulatory_regimes: list[str] | None = None,
        matched_rule_ids: list[str] | None = None,
        matched_control_ids: list[str] | None = None,
        policy_references: list[str] | None = None,
        model_signals: list[str] | None = None,
        receipt: dict[str, Any] | None = None,
        explanation: Any | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.reason = reason
        self.violation_id = violation_id
        self.decision_envelope_version = decision_envelope_version
        self.decision_reason_code = decision_reason_code
        self.action_classification = action_classification
        self.authorization_decision = authorization_decision
        self.defer_timeout_seconds = defer_timeout_seconds
        self.step_up_timeout_seconds = step_up_timeout_seconds
        self.enforcement_trace_id = enforcement_trace_id
        self.action_attestation_id = action_attestation_id
        self.fastml_features = dict(fastml_features or {}) if fastml_features else None
        self.score_components = dict(score_components or {}) if score_components else None
        self.top_contributors = [dict(item) for item in (top_contributors or []) if isinstance(item, dict)]
        self.decision_evidence = dict(decision_evidence or {}) if decision_evidence else None
        self.risk_score = risk_score
        self.latency_ms = latency_ms
        self.pack_id = pack_id
        self.pack_version = pack_version
        self.rule_version = rule_version
        self.regulatory_regimes = list(regulatory_regimes or [])
        self.matched_rule_ids = list(matched_rule_ids or [])
        self.matched_control_ids = list(matched_control_ids or [])
        self.policy_references = list(policy_references or [])
        self.model_signals = list(model_signals or [])
        self.receipt = dict(receipt or {}) if receipt else None
        self.explanation = explanation
        super().__init__(f"Thoth blocked tool '{tool_name}': {reason}" + (f" (violation_id={violation_id})" if violation_id else ""))


class ThothDeferredError(ThothPolicyViolation):
    def __init__(
        self,
        tool_name: str,
        reason: str,
        violation_id: str | None = None,
        decision_envelope_version: str | None = None,
        decision_reason_code: str | None = None,
        action_classification: str | None = None,
        authorization_decision: str | None = None,
        defer_timeout_seconds: int | None = None,
        step_up_timeout_seconds: int | None = None,
        enforcement_trace_id: str | None = None,
        action_attestation_id: str | None = None,
        fastml_features: dict[str, float] | None = None,
        score_components: dict[str, Any] | None = None,
        top_contributors: list[dict[str, Any]] | None = None,
        decision_evidence: dict[str, Any] | None = None,
        risk_score: float | None = None,
        latency_ms: float | None = None,
        pack_id: str | None = None,
        pack_version: str | None = None,
        rule_version: int | None = None,
        regulatory_regimes: list[str] | None = None,
        matched_rule_ids: list[str] | None = None,
        matched_control_ids: list[str] | None = None,
        policy_references: list[str] | None = None,
        model_signals: list[str] | None = None,
        receipt: dict[str, Any] | None = None,
        explanation: Any | None = None,
    ) -> None:
        super().__init__(
            tool_name=tool_name,
            reason=reason,
            violation_id=violation_id,
            decision_envelope_version=decision_envelope_version,
            decision_reason_code=decision_reason_code,
            action_classification=action_classification,
            authorization_decision=authorization_decision,
            defer_timeout_seconds=defer_timeout_seconds,
            step_up_timeout_seconds=step_up_timeout_seconds,
            enforcement_trace_id=enforcement_trace_id,
            action_attestation_id=action_attestation_id,
            fastml_features=fastml_features,
            score_components=score_components,
            top_contributors=top_contributors,
            decision_evidence=decision_evidence,
            risk_score=risk_score,
            latency_ms=latency_ms,
            pack_id=pack_id,
            pack_version=pack_version,
            rule_version=rule_version,
            regulatory_regimes=regulatory_regimes,
            matched_rule_ids=matched_rule_ids,
            matched_control_ids=matched_control_ids,
            policy_references=policy_references,
            model_signals=model_signals,
            receipt=receipt,
            explanation=explanation,
        )
