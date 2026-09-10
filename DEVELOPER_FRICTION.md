# Python SDK Developer Friction (Measured 2026-06-21)

## Scope and method
- Environment: Linux audit host.
- Timings measured with `/usr/bin/time -f 'elapsed=%e'`.
- Two install paths measured:
  - restricted DNS/sockets (default sandbox)
  - network-enabled temporary virtualenv

## Time to first event
- Restricted mode:
  - `pip install atensec-thoth`: **failed after 9.00s** (DNS resolution error).
  - `pip install aten-thoth` (legacy name): **failed after 8.78s** (same DNS restriction).
- Network-enabled mode:
  - Fresh venv `pip install --no-cache-dir atensec-thoth`: **13.52s**.
- Runtime demo:
  - `python3 DEMO_AGENT.py` (mock mode): **~1.5s** end-to-end, no live Thoth server required.

## Instrumentation lines required
- Practical `instrument()` setup requires explicit governance context (`agent_id`, `approved_scope`, `tenant_id`, `api_url`): typically **6-9 lines**.
- Result: does **not** yet meet the "<=5 lines" target for first governed event.

## Error clarity observed
- Install failures clearly indicate package index/DNS reachability issues.
- Runtime policy failures are explicit (`ThothPolicyViolation` includes reason and violation ID).

## If a developer is confused
- They may misread network/package-index failures as SDK breakage.
- They may follow stale docs still referencing `aten-thoth` instead of `atensec-thoth`.

## Top 3 friction points likely to stall a pilot
1. Network policy can block package installation in corporate environments.
2. Minimal instrumentation still needs multiple required identifiers.
3. Naming drift (`aten-thoth` vs `atensec-thoth`) in older examples can cause onboarding mistakes.
