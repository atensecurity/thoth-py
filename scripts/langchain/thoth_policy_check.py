#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

DECISION_ALLOW = "ALLOW"
DECISION_BLOCK = "BLOCK"
DECISION_STEP_UP = "STEP_UP"
DECISION_FLAG = "FLAG"


@dataclass(frozen=True)
class ToolPattern:
    name: str
    category: str
    inferred_domain: str | None
    inferred_record_count: int | None


def _tool_category(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("write", "delete", "update", "create")):
        return "write"
    if any(token in lowered for token in ("fetch", "http", "api", "web", "search")):
        return "http"
    if any(token in lowered for token in ("record", "read", "list", "query", "get")):
        return "read"
    return "generic"


def _infer_domain(name: str) -> str | None:
    if _tool_category(name) == "http":
        return "example.com"
    return None


def _infer_record_count(name: str) -> int | None:
    lowered = name.lower()
    if "record" in lowered or "customer" in lowered:
        return 20
    return None


def parse_tools(agent_path: Path) -> list[ToolPattern]:
    source = agent_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(agent_path))

    tool_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "tool":
                    tool_names.add(node.name)
                if isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Name) and decorator.func.id == "tool":
                        if decorator.args and isinstance(decorator.args[0], ast.Constant) and isinstance(decorator.args[0].value, str):
                            tool_names.add(decorator.args[0].value)
                        else:
                            tool_names.add(node.name)

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "tool":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    tool_names.add(node.args[0].value)

    patterns = [
        ToolPattern(
            name=name,
            category=_tool_category(name),
            inferred_domain=_infer_domain(name),
            inferred_record_count=_infer_record_count(name),
        )
        for name in sorted(tool_names)
    ]
    return patterns


def _collect_action_set(policy_text: str, block_name: str) -> set[str]:
    pattern = re.compile(
        rf"{block_name}\s*\[\s*reason\s*\]\s*if\s*\{{(.*?)\}}",
        flags=re.DOTALL,
    )
    action_pattern = re.compile(r'action_name\s*==\s*"([^"]+)"')
    actions: set[str] = set()
    for body in pattern.findall(policy_text):
        for match in action_pattern.findall(body):
            actions.add(match.strip())
    return actions


def parse_policy(policy_path: Path) -> dict[str, Any]:
    text = policy_path.read_text(encoding="utf-8")
    allowlist_pattern = re.compile(r"allowlisted_domains\s*:=\s*\{([^}]*)\}", flags=re.DOTALL)
    domain_pattern = re.compile(r'"([^"]+)"')

    allowlisted_domains: set[str] = set()
    allowlist_match = allowlist_pattern.search(text)
    if allowlist_match:
        for domain in domain_pattern.findall(allowlist_match.group(1)):
            allowlisted_domains.add(domain.strip().lower())

    block_actions = _collect_action_set(text, "deny")
    step_up_actions = _collect_action_set(text, "step_up")
    flag_actions = _collect_action_set(text, "flag")

    bulk_threshold = 10
    threshold_match = re.search(r"record_count\s*>\s*(\d+)", text)
    if threshold_match:
        bulk_threshold = int(threshold_match.group(1))

    http_block_non_allowlisted = "destination_domain" in text and "allowlisted_domains" in text

    return {
        "allowlisted_domains": allowlisted_domains,
        "block_actions": block_actions,
        "step_up_actions": step_up_actions,
        "flag_actions": flag_actions,
        "bulk_threshold": bulk_threshold,
        "http_block_non_allowlisted": http_block_non_allowlisted,
    }


def evaluate_patterns(patterns: list[ToolPattern], policy: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    allowlisted_domains: set[str] = policy["allowlisted_domains"]

    for pattern in patterns:
        decision = DECISION_ALLOW
        reason = "No matching deny/step_up/flag policy branch."
        evidence: dict[str, Any] = {
            "tool_name": pattern.name,
            "category": pattern.category,
        }

        if pattern.name in policy["block_actions"]:
            decision = DECISION_BLOCK
            reason = "Explicit deny rule for action_name."
        elif pattern.name in policy["step_up_actions"]:
            decision = DECISION_STEP_UP
            reason = "Explicit step_up rule for action_name."
        elif pattern.name in policy["flag_actions"]:
            decision = DECISION_FLAG
            reason = "Explicit flag rule for action_name."
        elif pattern.category == "http" and policy["http_block_non_allowlisted"] and pattern.inferred_domain and pattern.inferred_domain.lower() not in allowlisted_domains:
            decision = DECISION_BLOCK
            reason = f"HTTP domain {pattern.inferred_domain!r} not in allowlisted_domains."
            evidence["inferred_domain"] = pattern.inferred_domain
        elif pattern.category == "read" and pattern.inferred_record_count is not None and pattern.inferred_record_count > int(policy["bulk_threshold"]):
            decision = DECISION_FLAG
            reason = f"Bulk read heuristic exceeds threshold ({policy['bulk_threshold']})."
            evidence["inferred_record_count"] = pattern.inferred_record_count

        results.append(
            {
                "tool_name": pattern.name,
                "decision": decision,
                "reason": reason,
                "evidence": evidence,
            }
        )
    return results


def build_reports(
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    violations = [row for row in results if row["decision"] != DECISION_ALLOW]
    shadow_report = {
        "total_tools": len(results),
        "would_allow": sum(1 for row in results if row["decision"] == DECISION_ALLOW),
        "would_block": sum(1 for row in results if row["decision"] == DECISION_BLOCK),
        "would_step_up": sum(1 for row in results if row["decision"] == DECISION_STEP_UP),
        "would_flag": sum(1 for row in results if row["decision"] == DECISION_FLAG),
        "results": results,
    }
    return violations, shadow_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Static Thoth policy check for LangGraph/LangChain Python agent files.",
    )
    parser.add_argument("--agent-file", required=True)
    parser.add_argument("--policy-file", required=True)
    parser.add_argument("--enforcement-mode", choices=("observe", "enforce"), default="observe")
    parser.add_argument("--fail-on-violations", choices=("true", "false"), default="false")
    parser.add_argument("--violations-out", required=True)
    parser.add_argument("--shadow-report-out", required=True)
    parser.add_argument("--pass-fail-out", required=True)
    args = parser.parse_args()

    patterns = parse_tools(Path(args.agent_file))
    policy = parse_policy(Path(args.policy_file))
    evaluations = evaluate_patterns(patterns, policy)
    violations, shadow_report = build_reports(evaluations)

    fail_on_violations = args.fail_on_violations == "true"
    pass_fail = not (fail_on_violations and len(violations) > 0)

    Path(args.violations_out).write_text(
        json.dumps(violations, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    Path(args.shadow_report_out).write_text(
        json.dumps(
            {
                "enforcement_mode": args.enforcement_mode,
                **shadow_report,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    Path(args.pass_fail_out).write_text(
        "true" if pass_fail else "false",
        encoding="utf-8",
    )

    print(f"tools_discovered={len(patterns)} violations={len(violations)} pass_fail={pass_fail}")
    return 0 if pass_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
