# Thoth Policy Check Action

Use this action to catch policy violations before they reach production by statically analyzing agent tool-call patterns in a Python agent file against an OPA policy file, producing a shadow report and violation list without running a live agent.

## Inputs

| Name | Required | Default | Description |
|---|---|---|---|
| `agent_file` | Yes | - | Path to the agent Python file. |
| `policy_file` | Yes | - | Path to the OPA policy file. |
| `enforcement_mode` | No | `observe` | `observe` or `enforce`. |
| `fail_on_violations` | No | `false` | `true` or `false`. |

## Outputs

| Name | Description |
|---|---|
| `violations` | JSON list of policy violations found. |
| `shadow_report` | JSON summary of what would have been blocked/stepped-up/flagged. |
| `pass_fail` | Boolean (`true`/`false`) indicating action outcome. |

## Workflow Example

```yaml
- uses: atensecurity/thoth@v1
  with:
    agent_file: src/my_agent.py
    policy_file: policies/production.rego
    enforcement_mode: observe
```
