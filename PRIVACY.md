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

## Advisory tool inventory

Instrumentation sends tool names, descriptions, and input-schema structure with tenant and agent IDs to your configured Thoth API. Schema `default`, `example`, `examples`, `const`, and `enum` values are removed recursively. Descriptions and other schema text may still contain sensitive information: avoid embedding secrets in tool metadata. Set `THOTH_TOOL_INVENTORY=off` to disable inventory collection without disabling runtime enforcement. Uploads run in the background and do not grant execution authority.
