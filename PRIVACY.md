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

## Withholding values at the source (opt-in)

By default the SDK sends full tool arguments and task context to the authorization endpoints
(`/v1/enforce`, `/v1/explain`) so that value-based rules and outbound data checks can run. You can opt
in, per client, to withhold them with `data_handling_tool_arguments` (`hash`, `redact`, `drop`) and
`data_handling_prompts` (`hash`, `drop`), or the matching `THOTH_DATA_HANDLING_*` variables. Nothing
changes when they are unset.

With a mode set, the SDK does not send tool-argument values to Thoth. It transforms them in your
process before the request is built and declares what it did in a `data_handling` field on the request.
Prompt-bearing text and long free text in `task_context`, `auth_context`, `delegation_context`,
`identity_binding` and `request_metadata` get the prompts mode; identifiers in those dicts are sent as given,
and so are `session_intent`, `purpose` and `data_classification`, which you set.

Still sent to Thoth: tool names, decisions and their evidence, key names of tool arguments (except in
`drop` mode), session ids, user ids, agent and tenant ids, timing, the tool-call sequence, and the
identifiers and labels above. `hash` values are HMAC-SHA256 keyed with a salt that stays with your
client (`THOTH_DATA_HANDLING_SALT`, else random per process); Thoth does not hold it, and equal values
match only under the same salt. A low-entropy value can be confirmed by anyone who has the salt.

Effect on decisions: rules and outbound data checks that need argument values cannot run on values Thoth
does not have. A call that such a rule would examine is held for review (`STEP_UP`, reason code
`args_withheld`) rather than decided blind, and with `drop` some checks fall back on tool-name heuristics.
This option limits what Thoth receives; it does not make any statement about what your own tools, logs
or other services do with the values.
