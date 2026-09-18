# Changelog

All notable changes to `atensec-thoth` are documented in this file.

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
