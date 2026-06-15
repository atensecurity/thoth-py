# Runtime Policy Enforcement for LangGraph Agents

## 1) The gap

LangGraph gives you a strong runtime for stateful agent workflows, tool routing, retry behavior, and
human-in-the-loop control points. LangSmith gives you visibility into traces, latencies, and failure
patterns. That stack is good at telling you what happened and where it happened.

What it does not give you by default is a first-class, policy-driven authorization decision at the exact
tool-call boundary before side effects. In production systems, that is where the operational risk sits.
The risky call is not the model token stream. It is the outbound HTTP request to an unapproved domain,
the file write into a sensitive directory, or the bulk export query that crosses a data boundary.

Prompt guardrails are useful, but they are advisory. They rely on model behavior and prompt fidelity.
When tool calls become action-bearing operations, teams need deterministic policy outcomes that can
interrupt execution when needed, not just annotate traces after the fact.

This is why observability and enforcement are complementary layers. LangSmith helps you debug and improve.
Runtime enforcement decides whether an action is allowed right now, with an auditable decision record.

## 2) Where enforcement needs to sit

If you inspect a typical LangGraph agent path, the execution chain is roughly: `StateGraph` orchestration,
model output that includes tool calls, then `ToolNode` dispatch to concrete tool implementations. The
critical point is between “tool call selected” and “tool implementation invoked.” That is where a policy
engine has full context and can still prevent side effects.

Putting controls only at the network edge is too late for a lot of agent behavior. The edge can block
packets, but it usually does not understand the agent intent, session context, delegated task lineage,
or tool-level semantics like “this call is a write that mutates customer data.” You need action-level
context, not only transport-level context.

Prompt-level controls are also insufficient for the same reason. They run upstream of execution and
cannot reliably guarantee that a particular tool invocation will not happen under adversarial or edge
conditions. They influence behavior; they do not enforce behavior.

The SDK layer is where these signals come together cleanly:

- Current tool name and arguments
- Session call history (`session_tool_calls`)
- Purpose/data classification/task context
- Enforcement mode (`observe`, `step_up`, `block`, `progressive`)
- Correlation IDs for downstream audit (`enforcement_trace_id`)

At this layer, the runtime can request a policy decision, then execute one of three concrete branches:

1. `ALLOW`: proceed with tool execution.
2. `STEP_UP`: hold for approval before proceeding.
3. `BLOCK`: stop execution and raise a structured policy violation.

That is the control point that closes the “we can see it, but we cannot stop it” gap.

## 3) The 3-line integration

The integration is small and can be done without changing your graph topology.

```python
toolchain = FinancialToolchain(workspace_root)
governed = thoth.instrument_toolchain(toolchain, agent_id="langgraph-finance-agent", ...)
tools = build_tools(governed)
```

`governed` keeps the same callable surface, but each tool call now runs through pre-execution
enforcement and emits decision-context metadata for audit.

## 4) Shadow mode in practice

The first week should usually be `observe` mode. You want baseline data without changing user-visible
behavior. In that period, teams usually learn three things quickly.

First, call patterns are noisier than expected. Agent tools that looked read-only in design frequently
show hidden write paths in execution (temporary files, state sync operations, webhook callbacks). Shadow
mode exposes that before enforcement starts blocking production.

Second, high-value policy signals are often in metadata, not binary outcomes. Fields like
`decision_reason_code`, `matched_rule_ids`, `policy_references`, and `model_signals` help separate
real control gaps from policy tuning issues.

Third, approval workflows have edge cases that are easy to miss in tabletop reviews. A common one is the
AskUser path: the agent tries to request human approval, but the approval request itself carries risky
action context (for example, a prompt that embeds sensitive data or escalates outside intended scope).
In a mature policy setup, that AskUser attempt can be blocked or stepped up, rather than treated as
automatically safe just because it is “an approval flow.”

That specific pattern surprises teams because they expected “ask human” to always reduce risk. In practice,
approval actions are still actions and need the same policy scrutiny as any other tool call.

## 5) Writing your first policy

Start with one concrete policy tied to one concrete agent. Keep it narrow and auditable.

```rego
package thoth.policies.langgraph.financial_agent

import future.keywords.if
import future.keywords.in

principal := object.get(input, "principal", {})
action := object.get(input, "action", {})
context := object.get(input, "context", {})

action_name := lower(trim(sprintf("%v", [object.get(action, "name", "")]), " \t\n\r"))
action_payload := object.get(action, "payload", {})
destination_domain := lower(trim(sprintf("%v", [object.get(action_payload, "domain", "")]), " \t\n\r"))
record_count := to_number(object.get(action_payload, "record_count", 0))

allowlisted_domains := {"api.github.com", "hn.algolia.com", "www.sec.gov"}

deny[reason] if {
  principal != {}
  context != {}
  action_name == "fetch_account_profile"
  destination_domain != ""
  not destination_domain in allowlisted_domains
  reason := sprintf("BLOCK: outbound domain %q is not allowlisted", [destination_domain])
}

step_up[reason] if {
  principal != {}
  context != {}
  action_name == "write_payout_file"
  reason := "STEP_UP: payout file writes require approval"
}

flag[reason] if {
  principal != {}
  context != {}
  action_name == "read_customer_records"
  record_count > 10
  reason := sprintf("FLAG: bulk read of %.0f records", [record_count])
}

allow if {
  count(deny) == 0
}
```

Line by line, this policy does four things:

1. Normalizes core input objects (`principal`, `action`, `context`) so policy evaluation is robust.
2. Hard-blocks outbound profile fetches to non-allowlisted domains.
3. Forces approval on payout file writes.
4. Flags large record reads for audit while still allowing execution.

That is enough to create meaningful control without overfitting.

## 6) Moving to enforce

Before switching from shadow to enforce, verify: baseline stability over at least a week, clear owners for
every expected `STEP_UP` path, explicit rollback for policy bundles, and CI checks for agent tool-pattern
drift. If those are in place, moving to enforce is usually operationally boring, which is what you want.

## 7) Close

LangGraph already gives a strong execution runtime. Adding runtime policy enforcement at the tool boundary
turns that runtime into a controllable production surface, not just an observable one. Working examples are
available at github.com/atensecurity.
