# Changelog

All notable changes to `atensec-thoth` are documented in this file.

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
