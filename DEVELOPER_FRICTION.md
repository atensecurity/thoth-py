# Python SDK Developer Friction (Measured 2026-06-19)

## Scope and method
- Environment: Linux sandbox (no outbound package index access).
- Command attempted: `python3 -m pip install -e backend/python/libs/thoth`
- Time measured with `/usr/bin/time -p`.

## Time to first event
- `pip install aten-thoth` equivalent (local editable install): **failed after 12.04s**.
- Failure observed: dependency bootstrap required `poetry-core>=1.8.3` from PyPI, but DNS/network was blocked.
- Mock-mode first end-to-end governed run (`python3 DEMO_AGENT.py`): **0.86s wall time**.

## Instrumentation lines required
- Minimal practical `instrument()` call currently requires explicit governance context.
- Measured minimum clear setup in this repo: **8 lines** (agent_id, approved_scope, tenant_id, enforcement, api_key, api_url).
- Result: does **not** yet meet the "5 lines or fewer" pilot goal.

## Error clarity observed
- Install failure text clearly points to missing `poetry-core` download and DNS resolution failure.
- Runtime policy violations are clear (`ThothPolicyViolation` includes reason + violation_id).

## If a developer is confused
- They will likely retry install and then manually inspect packaging dependencies.
- They may not realize quickly that network egress restrictions, not Thoth code, caused installation failure.

## Top 3 friction points likely to stall a pilot
1. Packaging dependency chain requires networked build dependencies during install.
2. Baseline instrumentation still needs several explicit identifiers (not 2-env-var zero-config).
3. Cross-instrumentation compatibility (Datadog/LangSmith/Otel/Sentry) is not covered by automated Python integration tests yet.
