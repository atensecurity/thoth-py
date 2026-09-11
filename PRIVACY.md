# Telemetry privacy boundary

Thoth sends full tool arguments and task context only to authorization endpoints.
Retained HTTP and SQS telemetry uses an allowlisted minimal projection: lifecycle
fields, opaque correlation identifiers, policy codes, numeric risk evidence, and a
restricted receipt/evidence subset. Free-form arguments, context, results, errors,
explanations, receipt envelopes, and unknown metadata are not serialized.

Applications must use non-sensitive opaque values for tenant, user, agent, session,
event, action, violation, receipt, policy, and rule identifiers. The projection is
not a secret scanner and intentionally preserves those identifiers for audit
correlation.
