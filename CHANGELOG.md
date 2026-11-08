# Changelog

All notable changes to `atensec-thoth` are documented in this file.

## Unreleased

### Added

- Opt-in withholding of values at the source. Set `data_handling_tool_arguments`
  (`hash`, `redact` or `drop`) and `data_handling_prompts` (`hash` or `drop`) on
  `init()`, the `instrument*()` functions or `ThothConfig`, or the environment
  variables `THOTH_DATA_HANDLING_TOOL_ARGUMENTS`, `THOTH_DATA_HANDLING_PROMPTS` and
  `THOTH_DATA_HANDLING_SALT`. Hashes use a client-held salt. The request declares
  what was applied, a rule that needs argument values holds the call
  (`args_withheld`) and says why, and modified arguments are never applied to a
  call whose arguments were withheld: if the server returns a rewrite anyway, the
  call fails closed (the tool does not run, `ThothPolicyViolation`) because that
  is a server or contract problem. Nothing changes when unset; an invalid
  value raises when the client is created.
- Local scanning when tool arguments are withheld. With
  `data_handling_tool_arguments` set to `hash`, `redact` or `drop`, the SDK scans the
  raw arguments (and the destination host in `request_metadata`) in your process
  before withholding them, and sends `client_findings` next to `data_handling`:
  counts and categories of credentials, email addresses, phone numbers, SSN-shaped
  numbers and confidentiality phrases, whether a path looks sensitive, a destination
  hostname, and recipient domains. The report never contains values, paths, local
  parts or display names. The detectors are heuristic. The scanner identifies itself
  as `atensec-thoth` with ruleset `thoth-detectors.v1`, ruleset hash
  `db797fde60f300719b9dac8a360efd6ce8b1f7313587cc714aebf7a93428434b`; a workspace
  relies on the report only after it sets `accept_local_scan` and lists that hash in
  `accepted_scanner_ruleset_hashes`. A check that fails, hits the depth or value
  bound, or would not fit the 4 KB limit is left out of `scanned` instead of
  reported clean. `THOTH_DATA_HANDLING_LOCAL_SCAN=off` (or
  `data_handling_local_scan=False` on `ThothConfig`) sends no findings. A workspace
  with `require_client_findings` holds calls from a client that withholds arguments
  and sends no findings. Nothing changes when data handling is not configured.
- The local scan leaves `recipient` out of `scanned`, instead of reporting it clean or milder, when a
  recipient field holds one of the 23 spaces that Python splits words on and ASCII does not (U+001C
  to U+001F, U+0085, U+00A0, U+1680, U+2000 to U+200A, U+2028, U+2029, U+202F, U+205F, U+3000).
  Accented letters and other text in recipient fields still scan.

## 0.7.0 - 2026-11-06

### Fixed

- `ThothPolicyViolation` and `ThothDeferredError` can be copied and pickled. A
  framework that re-raises the exception from a task (AG2 does) rebuilt it from
  its message alone and failed with `TypeError: ... missing 1 required positional
  argument: 'reason'`, so a refused call surfaced as that error instead of
  telling the model the call was refused.
- Observe mode now sends each call to the enforcer, as the product defines
  observe (evaluate, record what would have been held or blocked, allow). The
  SDK used to skip the enforcer entirely when `enforcement="observe"`, so those
  agents produced no "would have blocked" data and the ready-to-enforce checks
  had nothing to read. The tool always runs with its original arguments, a hold
  or modification in the answer is ignored, and an enforcer failure never fails
  the call.
- Enforce, explain and hold-status requests send the API key in both
  `X-Api-Key` and `Authorization: Bearer`, as event ingestion already does. The
  SDK sent only `X-Api-Key` to enforce and only Bearer to the hold poll, so
  behind an ingress that strips one of them either every call was blocked or a
  held call could never be approved.
- A call the enforcer modifies now runs with the modified arguments in every call
  shape. Keyword-only calls (`fn(to=..., body=...)`) used to run with the original
  arguments, because only positional and single-dict calls were mapped. A held
  call that is then approved also runs with the arguments the enforcer redacted
  when it held the call (the approval used to start the original call). A
  modification the SDK cannot map back onto the call, or one that arrives with no
  arguments, now refuses the call with a `ThothPolicyViolation` instead of running
  the original arguments.
- Event telemetry now includes the receipt's `signature_key_id`, the field the
  enforcer uses for the signing key. The allow-list only had `key_id`, so the
  key never reached events.
- The copy of the tool arguments sent to the enforcer no longer loses
  structure. Pydantic models, dataclasses, sets and enums keep their contents
  (they used to become a plain string), and nothing nested up to 64 levels is
  replaced by `"[truncated]"`; deeper values are kept as JSON text. The tool
  still runs on your original objects, so a policy can now see the same fields
  the tool does, such as the recipient of a message. Arbitrary objects (a client,
  a session) are not walked, since they can hold credentials, and a model's or
  dataclass's every field is kept. Arguments with more than
  100,000 values, such as one shared object repeated until it explodes, raise
  `ToolArgumentsTooLargeError` and the call does not run.
- A call the enforcer allows with its arguments redacted or filtered (outbound
  DLP redaction, or purpose-limited retrieval) now runs with the modified
  arguments. The enforcer sends that as `decision` ALLOW next to
  `authorization_decision` MODIFY, and the SDK used to read only `decision`, so the
  tool ran with the original arguments while the decision log said MODIFY. A
  stricter `decision` (BLOCK, STEP_UP or DEFER) still wins. Update to get the
  redaction applied; the TypeScript SDK was not affected.
- `wrap_autogen_tools` takes the Thoth settings (`agent_id`, `tenant_id`,
  `approved_scope` and the rest) as its docstring showed; it used to accept
  only a tracer, so the documented example raised `TypeError`. Passing a
  tracer still works. It is for AutoGen 0.2-style function maps (AG2 0.x).

### Added

- `thoth.init()` now also governs the Anthropic client's tools: every tool
  the tool runner (`client.beta.messages.tool_runner`) or the session tool
  runner calls, and any direct `tool.call()`. That covers `@beta_tool` and
  `@beta_async_tool` functions, MCP tools from `mcp_tool` and
  `async_mcp_tool` (checked once, not again by the MCP hook), and built-in
  tools such as your memory tool, including classes defined after `init()`.
  Each call is checked with the input it runs with; a refused call never
  runs, and Claude gets "Thoth refused this call: <reason>" as an error tool
  result. Tested with `anthropic` 2.3.0.
- `thoth.init()` now also governs AG2 (1.x): every function tool an AG2
  `Agent` runs, from `@tool`, a `Toolkit` or a subagent tool, is checked
  with the arguments it runs with, after the tool's and the agent's tool
  middleware. A refused call never runs, and the model gets "Thoth refused
  this call: <reason>" as an error tool result. Tested with `ag2` 1.2.2.

### Changed

- An explicit `api_url` now wins over `THOTH_API_URL`, then the environment
  variable, then the default `https://api.atensecurity.com`, the same order the
  TypeScript and Go SDKs use. `ThothConfig` used to let the environment variable
  override a value passed in config, so a workload that sets both now talks to
  the `api_url` it passed. Blank values are ignored.
- The `autogen` extra now installs AG2 (`ag2>=1.0`), and an `ag2` extra is
  added. It used to install `pyautogen`, which is now a proxy for Microsoft's
  `autogen-agentchat`, an API Thoth didn't govern. For Microsoft's agents,
  use Microsoft Agent Framework, which `thoth.init()` governs.
- `api_url` defaults to `https://api.atensecurity.com`, the API for trial and
  self-serve workspaces, when neither `api_url` nor `THOTH_API_URL` is set,
  including for `thoth.init()`. Previously the SDK refused to start without
  one. Enterprise workspaces keep setting their own host.

## 0.6.0 - 2026-10-31

### Fixed

- `acting_for(user, session_id=...)`: Thoth now sees a session ID derived from
  the tenant, agent, user and your session ID, rather than your ID itself, so
  two users who name the same session never share one in Thoth. The ID is
  stable across processes, so resuming a conversation still continues its
  history, and the TypeScript SDK derives the same ID.
- Claude Agent SDK: `instrument_claude_agent_sdk_options` now checks every
  tool call in a PreToolUse hook, including tools in `allowed_tools` and calls
  in bypass-permissions mode, which the SDK never passes to `can_use_tool` and
  which Thoth therefore didn't check before. An allowed call is still left to
  the SDK's rules and your `can_use_tool`, which reuses the hook's decision
  instead of checking again. A call whose arguments Thoth would change is
  denied by the hook.

### Added

- `thoth.init()` now also governs MCP tool calls made with the `mcp`
  package's `ClientSession.call_tool`, whether your code calls an MCP server
  directly or through a framework's MCP adapter. A call a framework hook has
  already checked (the same arguments, under the MCP tool's name or a prefixed
  form of it, such as `github_create_issue` for `create_issue`) isn't checked
  or counted twice; any other call is checked. A refused call is never sent to
  the server and returns an error result, "Thoth refused this call: <reason>".
  Tested with `mcp` 2.3.0.
- `thoth.init()` now also governs the Claude Agent SDK: every `query()` and
  `ClaudeSDKClient` connection gets Thoth's PreToolUse check, so each tool
  call, including tools in `allowed_tools`, is checked before the CLI runs it,
  without passing options through `instrument_claude_agent_sdk_options`. Your
  own hooks still run, your options object is not changed, and options you
  instrumented by hand are not checked twice. A refused call is denied with
  "Thoth refused this call: <reason>". Tested with `claude-agent-sdk` 0.2.176.
- `thoth.init()` now also governs Strands Agents tools: `@tool` functions,
  MCP tools and agents used as tools, whether the model calls them or your
  code calls `agent.tool.<name>()` directly. Each call is checked after
  Strands' hooks and tool middleware, with exactly the input that runs. A
  refused call never runs and ends with an error tool result, "Thoth refused
  this call: <reason>". Tested with `strands-agents` 1.63.0.
- `thoth.init()` now also governs Microsoft Agent Framework tools: `@tool`
  functions, agents used as tools and MCP tools, at the point every tool call
  runs, after function middleware and approval, with exactly the arguments
  Thoth checked. The framework's invocation context still reaches the tool and
  is not sent to Thoth. A refused call never runs, and the model gets "Thoth
  refused this call: <reason>" as the tool result. Tested with
  `agent-framework-core` 1.24.0.
- `thoth.init()` now also governs Google ADK tools (`FunctionTool`, MCP
  tools, agent tools and your own `BaseTool` subclasses, including ones
  defined after `init()`), checked at `run_async`, where every ADK tool path
  ends, after the before-tool callbacks, and run with exactly the arguments
  Thoth checked. A refused call never runs, and the model gets "Thoth refused
  this call: <reason>" as the tool result. Tested with `google-adk` 2.13.0's
  `LlmAgent` and `InMemoryRunner`.
- `thoth.init()` now also governs Pydantic AI tool calls (function tools and
  MCP toolsets), checked with the final validated arguments, after Pydantic
  AI's own capability hooks, and run with exactly what Thoth checked. A
  refused call never runs, and the model gets "Thoth refused this call:
  <reason>" as the tool result; a call that can't be checked is refused.
  Output functions are checked before they run in every output mode (tool,
  `TextOutput`, native and prompted), by the function's name and arguments,
  and a refusal raises `ThothPolicyViolation`, since text in its place would
  become the agent's typed final answer; plain schema outputs call no
  function of yours. Tested with `pydantic-ai-slim` 2.64.0's `Agent`
  and `FunctionModel`.
- `thoth.init()` now also governs CrewAI tools, including tools defined
  before `init()` runs: `@tool` tools and `BaseTool` subclasses, whether an
  agent calls them natively (`tool.run`) or through CrewAI's text tool loop
  (`CrewStructuredTool.invoke`). Each call is checked with its arguments and
  runs with exactly what Thoth checked. A refused call never runs, and the
  agent gets "Thoth refused this call: <reason>". A result CrewAI serves from
  its tool cache doesn't run the tool and isn't checked again. Tested with
  `crewai` 1.15.33.
- `thoth.init()` now also governs OpenAI Agents SDK (`openai-agents`)
  function tools, including tools defined before `init()` runs. Each call is
  checked with the model's arguments and runs with exactly what Thoth checked;
  arguments that aren't a JSON object are refused. A refused call never runs,
  and the model gets "Thoth refused this call: <reason>" as the tool's output.
  Local shell, computer and apply_patch tools are not covered by this hook.
  Tested with `openai-agents` 0.24.0's `Runner`.
- `thoth.init()`: one call at startup governs every LangChain tool in the
  process, including the tools LangGraph's `ToolNode` runs, with no change to
  any tool. It reads
  `THOTH_AGENT_ID`, `THOTH_TENANT_ID`, `THOTH_API_URL` and `THOTH_API_KEY`;
  `approved_scope` is optional. The server's observe or enforce mode for the
  agent applies, a refusal raises `ThothPolicyViolation` before the tool runs,
  and each tool is reported to the tool inventory the first time it is used.
  `thoth.shutdown()` removes the hooks.
- `thoth.acting_for(user_id, session_id=None)`: a block in which governed
  tools act for that user, in one session (generated when not given).
- A tool already wrapped by `instrument()`, `instrument_toolchain()` or
  `instrument_langgraph()` is checked once, not again by `init()`'s hook.

## 0.5.25 - 2026-10-23

### Added

- Billing errors: when the enforcer answers HTTP 402 with a `billing_error`
  (governed actions used up, trial ended, or payment overdue), the call is
  blocked with Thoth's message as the reason and `decision_reason_code`
  `billing_<code>`. The message is logged once per process. `fail_open` does
  not apply to billing errors.
- Approval link: while a call waits on a step-up hold, the SDK logs one warning
  naming the tool, when the hold expires and the dashboard link where a
  reviewer approves or denies it, instead of waiting silently. It needs an
  enforcer that returns `approval_url` on the hold status; with an older
  enforcer the SDK stays quiet as before.

## 0.5.24 - 2026-10-18

### Added

- Tool inventory: on instrumentation, the SDK sends each tool's name,
  description and input-schema structure to your Thoth API in the background,
  so Thoth can propose an effect class for an admin to confirm. Proposals are
  advisory: they never grant execution, and only an admin's confirmation
  changes decisions. Collection runs off the startup path, cannot raise into
  your code, and times out after 2 seconds.

### Privacy

- Schema `default`, `example`, `examples`, `const` and `enum` values are
  removed recursively before sending; parameters with those names are kept.
  Descriptions are sent as written, so keep secrets out of tool metadata.
- Set `THOTH_TOOL_INVENTORY=off` to disable collection.

## 0.5.23 - 2026-09-18

### Fixed

- The tracer discarded the final enforcement decision after computing effective
  tool arguments, so POST telemetry hardcoded `ALLOW` and omitted server-issued
  reason codes and structured decision evidence. The real outcome is now
  retained and reported.
- A resolved `STEP_UP` no longer loses the initial policy evidence or the
  canonical terminal receipt. Final authorization semantics are merged with the
  initial reason, policy and receipt evidence, and the terminal receipt is
  preserved separately.
- A sparse server response no longer erases locally established correlation.
  Sparse decisions bind to the local trace and action IDs.

### Security

- A server response whose action or trace ID contradicts the locally established
  one now **fails closed before the tool executes**, rather than proceeding with
  ambiguous correlation.

### Added

- `EnforcementDecision` and the policy exceptions gain an optional
  `terminal_receipt` field. Existing callers are unaffected.
- Claude Agent SDK post-success and post-failure hooks now carry decision
  context, correlated through a bounded single-use map keyed on the provider
  tool-use ID. The map holds at most 1024 pending entries; a reused, missing,
  unseen or evicted ID yields empty context rather than another action's, so
  evidence correlation fails closed without changing the authorization result.

### Unchanged

- Tool execution, blocking, deferral, step-up and argument-modification
  semantics, final `MODIFY` outcomes, and current `OBSERVE` behaviour. The
  enforcement return shape remains private to the SDK implementation.

## 0.5.22 - 2026-09-11

### Added

- Added first-class LangGraph support through `instrument_langgraph()`,
  `instrument_tool_node()`, and `@thoth_graph`, including synchronous and
  asynchronous tool execution, concurrent call history, mock mode, and explicit
  `ThothDeferredError` handling.
- Added per-action attestation IDs across authorization, Claude Agent SDK hooks,
  lifecycle events, policy errors, and decision evidence. Provider tool-use IDs
  are used for correlation only and are never represented as authenticated
  workload identity.
- Added model name/provider/artifact context, authentication and delegation
  context, MCP runtime identity, and caller metadata to authorization requests.
- Added human-explanation retrieval for blocked and step-up decisions with
  optional bounded webhook delivery.
- Added wheel-content validation and clean installed-package tests for the base
  package and the Anthropic, AutoGen, Claude, LangChain, LangGraph, and OpenAI
  extras.

### Fixed

- Return a fresh approval-timeout decision for every sync and async call, so
  blocked-event evidence cannot inherit an earlier action's attestation ID.
- Reject unresolved or malformed step-up results without executing the tool,
  and keep synchronous and asynchronous fallback enforcement to one decision
  per action.
- Reject Claude callback results that identify a different action, and avoid
  correlating concurrent callbacks by tool name when no trusted action ID is
  available.
- Govern callable-only tool collections and LangChain-style duck-typed
  `_run`/`run` tools even when their framework classes are unavailable.
- Correct optional dependency declarations and restore SDK type-check coverage.

### Changed

- Minimized retained HTTP and SQS telemetry with an explicit allowlist. Tool
  arguments, free-text context, results, errors, explanations, full receipts,
  and unknown metadata remain available to authorization but are not retained
  by default.
- Added bounded HTTP and SQS delivery retries with stable event and FIFO
  deduplication identifiers. Retrying telemetry never re-executes a governed
  tool or repeats its authorization decision.
- Added SQS partial-batch failure handling, process-local delivery counters,
  and bounded shutdown through `close(timeout=...)`.
- Documented the telemetry privacy boundary and the limits of process-local
  delivery status.

## 0.5.21 - 2026-06-20

### Changed

- Added configurable resilience mode via `ThothConfig.fail_open` / `THOTH_FAIL_OPEN`.
  When enabled, enforcer transport failures and retryable statuses (`429`, `5xx`)
  return `ALLOW` so agent execution can continue.
- Auth failures (`401`/`403`) remain fail-closed and continue returning `BLOCK`.
- Added regression tests for fail-open timeout, retryable status, and auth-failure handling.
- Added LangSmith coexistence integration coverage
  (`tests/integrations/test_langsmith_compat.py`) to ensure `traceable` tools
  continue to execute under Thoth governance wrappers.
- Added observability-wrapper coexistence integration coverage for Datadog-like,
  OpenTelemetry-like, and Sentry-like instrumentation patterns
  (`tests/integrations/test_observability_compat.py`).
- Hardened `instrument()` wrapping for LangChain-style duck-typed tools so
  `_run` and `run` are both governed even when LangChain classes are not
  importable at runtime.
- Fixed generic `.tools` wrapping for callable-only tool entries (plain
  functions/callables), which are now replaced with governed callables instead
  of bypassing enforcement.
- Added regression tests for callable-only tool entries and LangChain
  duck-typed `_run`/`run` wrapping in `tests/test_instrumentor.py`.
- Added first-class LangGraph instrumentation APIs:
  `instrument_langgraph()`, `instrument_tool_node()`, and `@thoth_graph`.
- Added mock-mode LangGraph decision simulation (`THOTH_MOCK_MODE=true`) for
  local end-to-end development without a live enforcer.
- Added `ThothDeferredError` and LangGraph DEFER handling that raises without
  executing the tool and emits defer metadata in tool telemetry.
- Added a thread-safe session call-history path for concurrent LangGraph
  branches via `SessionContext.pending_tool_calls(...)`.

## 0.5.20 - 2026-06-15

### Added

- Added a LangGraph runtime enforcement example at
  `examples/langchain/langgraph_runtime_enforcement.py` that demonstrates
  observe and progressive enforcement flows using `instrument_toolchain()`.
- Added a companion policy example at
  `examples/langchain/policies/research_agent.rego` covering allow, step-up,
  block, and flag outcomes for a research-agent toolchain.
- Added a cookbook notebook at
  `examples/langchain/cookbook_agent_governance.ipynb` with end-to-end shadow
  and enforce walkthroughs.
- Added a reusable composite GitHub Action at
  `.github/actions/thoth-policy-check/` for static policy checks against
  Python agent tool-call patterns.
- Added a static analysis helper script at
  `scripts/langchain/thoth_policy_check.py` used by the action.
- Added a vendor-neutral RFC issue draft for LangGraph tool interception hooks
  at `docs/contributions/langchain_rfc_tool_hooks.md`.
- Added an engineering blog draft at
  `docs/blog/langchain-runtime-enforcement.md`.

## 0.1.17 - 2026-05-18

### Added

- Added `instrument_toolchain()` for one-call recursive instrumentation of nested toolchains
  (dict/list/object graphs).
- Added `toolchain_function_map()` to derive framework function maps (for example AutoGen)
  directly from governed toolchain objects.

### Changed

- Updated `instrument_anthropic()` and `instrument_openai()` to wrap nested callables by
  dotted path names.
- `instrument_toolchain()` traversal is now automatic by default (`max_depth=None`) with cycle
  protection; callers can still pass an explicit depth cap if needed.

## 0.1.16 - 2026-05-14

### Changed

- Prepared release automation for the upcoming `v0.1.16` public SDK tag.
- Added a public-repo release workflow at `.github/workflows/release.yml`.
- Release automation now triggers on `v*` tags pushed by the internal mirror workflow.
- PyPI publication now uses Trusted Publisher OIDC on environment `pypi`, with `PYPI_TOKEN`
  fallback support.
- GitHub release notes are now sourced from versioned sections in this changelog.
- Standardized package naming/docs from `aten-thoth` to `atensec-thoth`.
- Expanded enforcement decision normalization and aliases (`DENY→BLOCK`, `CHALLENGE/ESCALATE→STEP_UP`,
  `TRANSFORM→MODIFY`, `HOLD→DEFER`) with richer decision-envelope fields.
- Added SDK handling for `MODIFY` and `DEFER` decisions:
  - `MODIFY` can rewrite tool arguments before execution.
  - `DEFER` now raises `ThothPolicyViolation` with defer timeout context.
- Improved async governance path so wrapped async tools use async enforcer calls (`acheck`) end-to-end.
- Added expanded policy/telemetry context propagation:
  `tool_args`, `environment`, `enforcement_trace_id`, `session_intent`, `purpose`,
  `data_classification`, and `task_context`.
- Added expanded `ThothPolicyViolation` metadata for downstream logging/incident handling:
  decision reason codes, model features/signals, pack/rule metadata, and signed receipt payload.
- Improved HTTP diagnostics for auth/ingress failures with actionable hints for 401/403 responses.

## 0.5.10 - 2026-05-05

### Changed

- Declared the current stable Python SDK release line in a versioned changelog.
- Added customer-facing release-note structure for future tagged releases.
