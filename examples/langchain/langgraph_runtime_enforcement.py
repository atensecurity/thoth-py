# LangGraph + Thoth runtime governance example.
# Gap: LangSmith shows what happened; Thoth governs what was allowed to happen.
# They are complementary: observability and pre-execution enforcement are different layers.
#
# 3-line instrumentation pattern:
#   toolchain = ResearchToolchain(workspace_root)
#   governed = thoth.instrument_toolchain(toolchain, ...)
#   tools = build_tools(governed)
#
# Create an API key at https://start.atensecurity.com and export:
#   export ATEN_API_KEY="..."

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
import uuid

import requests

import thoth
from thoth import ThothPolicyViolation
from thoth.enforcer_client import EnforcerClient
from thoth.models import EnforcementMode, ThothConfig
from thoth.session import SessionContext

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


@dataclass(frozen=True)
class RuntimeSettings:
    api_key: str
    tenant_id: str
    api_url: str
    user_id: str
    workspace_root: Path


class ResearchToolchain:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    def web_search(self, query: str, domain: str = "hn.algolia.com") -> str:
        if domain != "hn.algolia.com":
            # Keep control explicit so domain choices are visible to policy.
            return json.dumps(
                {
                    "error": "unsupported domain for demo web_search",
                    "domain": domain,
                },
                indent=2,
            )

        response = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": query, "tags": "story", "hitsPerPage": 3},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        hits = payload.get("hits", [])
        normalized = [
            {
                "title": hit.get("title"),
                "url": hit.get("url"),
            }
            for hit in hits
        ]
        return json.dumps({"domain": domain, "results": normalized}, indent=2)

    def fetch_external_api(self, url: str) -> str:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        body = response.json() if "application/json" in response.headers.get("content-type", "") else {"text": response.text[:4000]}
        return json.dumps(
            {
                "url": url,
                "status_code": response.status_code,
                "body": body,
            },
            indent=2,
        )

    def read_local_file(self, path: str) -> str:
        target = (self.workspace_root / path).resolve()
        if not str(target).startswith(str(self.workspace_root.resolve())):
            raise ValueError(f"path escapes workspace: {path}")
        return target.read_text(encoding="utf-8")

    def write_local_file(self, path: str, content: str) -> str:
        target = (self.workspace_root / path).resolve()
        if not str(target).startswith(str(self.workspace_root.resolve())):
            raise ValueError(f"path escapes workspace: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} bytes to {target}"

    def read_customer_records(self, record_count: int) -> str:
        records = [
            {
                "customer_id": f"cust-{index + 1:03d}",
                "status": "active",
                "region": "us-west-2",
            }
            for index in range(record_count)
        ]
        return json.dumps({"record_count": record_count, "records": records}, indent=2)


class ResearchState(MessagesState):
    objective: str
    step: int
    force_block: bool


def _tool_call(call_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "name": name,
        "args": args,
        "type": "tool_call",
    }


def planner_node(state: ResearchState) -> dict[str, Any]:
    step = int(state.get("step", 0))
    objective = str(state.get("objective", ""))
    force_block = bool(state.get("force_block", False))

    if step == 0:
        return {
            "step": 1,
            "messages": [
                AIMessage(
                    content="Gather public references from the web.",
                    tool_calls=[
                        _tool_call(
                            call_id=f"tool-{uuid.uuid4().hex[:8]}",
                            name="web_search",
                            args={"query": objective, "domain": "hn.algolia.com"},
                        )
                    ],
                )
            ],
        }

    if step == 1:
        return {
            "step": 2,
            "messages": [
                AIMessage(
                    content="Read internal records to cross-check volume.",
                    tool_calls=[
                        _tool_call(
                            call_id=f"tool-{uuid.uuid4().hex[:8]}",
                            name="read_customer_records",
                            args={"record_count": 12},
                        )
                    ],
                )
            ],
        }

    if step == 2 and force_block:
        return {
            "step": 3,
            "messages": [
                AIMessage(
                    content="Attempt external API fetch outside the allowlist.",
                    tool_calls=[
                        _tool_call(
                            call_id=f"tool-{uuid.uuid4().hex[:8]}",
                            name="fetch_external_api",
                            args={"url": "https://example.com"},
                        )
                    ],
                )
            ],
        }

    if step == 2:
        return {
            "step": 3,
            "messages": [
                AIMessage(
                    content="Write a local brief.",
                    tool_calls=[
                        _tool_call(
                            call_id=f"tool-{uuid.uuid4().hex[:8]}",
                            name="write_local_file",
                            args={
                                "path": "tmp/research_brief.md",
                                "content": "# Brief\n\nDraft generated by governed LangGraph run.\n",
                            },
                        )
                    ],
                )
            ],
        }

    if step == 3:
        return {
            "step": 4,
            "messages": [
                AIMessage(
                    content="Read the brief we just wrote and conclude.",
                    tool_calls=[
                        _tool_call(
                            call_id=f"tool-{uuid.uuid4().hex[:8]}",
                            name="read_local_file",
                            args={"path": "tmp/research_brief.md"},
                        )
                    ],
                )
            ],
        }

    return {"messages": [AIMessage(content=("Research workflow completed. Review tool traces and audit records " "for ALLOW / STEP_UP / BLOCK behavior."))]}


def route_after_planner(state: ResearchState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def build_graph(tools: list[Any]) -> Any:
    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner_node)
    builder.add_node("tools", ToolNode(tools, name="tools"))
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_after_planner)
    builder.add_edge("tools", "planner")
    return builder.compile()


def build_tools(governed: ResearchToolchain) -> list[Any]:
    @tool("web_search")
    def web_search(query: str, domain: str = "hn.algolia.com") -> str:
        """Search public web content for current context."""
        return governed.web_search(query=query, domain=domain)

    @tool("fetch_external_api")
    def fetch_external_api(url: str) -> str:
        """Call an external HTTP API."""
        return governed.fetch_external_api(url=url)

    @tool("read_local_file")
    def read_local_file(path: str) -> str:
        """Read a file from the local workspace."""
        return governed.read_local_file(path=path)

    @tool("write_local_file")
    def write_local_file(path: str, content: str) -> str:
        """Write a file into the local workspace."""
        return governed.write_local_file(path=path, content=content)

    @tool("read_customer_records")
    def read_customer_records(record_count: int) -> str:
        """Read synthetic customer records to simulate bulk data access."""
        return governed.read_customer_records(record_count=record_count)

    return [
        web_search,
        fetch_external_api,
        read_local_file,
        write_local_file,
        read_customer_records,
    ]


def decision_audit_record(decision: Any) -> dict[str, Any]:
    return {
        "decision_envelope_version": decision.decision_envelope_version,
        "enforcement_trace_id": decision.enforcement_trace_id,
        "authorization_decision": decision.authorization_decision or decision.decision.value,
        "decision_reason_code": decision.decision_reason_code,
        "action_classification": decision.action_classification,
        "reason": decision.reason,
        "violation_id": decision.violation_id,
        "hold_token": decision.hold_token,
        "risk_score": decision.risk_score,
        "latency_ms": decision.latency_ms,
        "pack_id": decision.pack_id,
        "pack_version": decision.pack_version,
        "rule_version": decision.rule_version,
        "regulatory_regimes": decision.regulatory_regimes,
        "matched_rule_ids": decision.matched_rule_ids,
        "matched_control_ids": decision.matched_control_ids,
        "policy_references": decision.policy_references,
        "model_signals": decision.model_signals,
        "fastml_features": decision.fastml_features,
        "score_components": decision.score_components,
        "top_contributors": decision.top_contributors,
        "decision_evidence": decision.decision_evidence,
        "receipt": decision.receipt,
    }


def violation_audit_record(exc: ThothPolicyViolation) -> dict[str, Any]:
    session = thoth.get_current_session()
    return {
        "event_type": "TOOL_CALL_BLOCK",
        "tool_name": exc.tool_name,
        "reason": exc.reason,
        "violation_id": exc.violation_id,
        "session_id": session.session_id if session else None,
        "metadata": {
            "decision_envelope_version": exc.decision_envelope_version,
            "enforcement_trace_id": exc.enforcement_trace_id,
            "authorization_decision": exc.authorization_decision,
            "decision_reason_code": exc.decision_reason_code,
            "action_classification": exc.action_classification,
            "risk_score": exc.risk_score,
            "latency_ms": exc.latency_ms,
            "pack_id": exc.pack_id,
            "pack_version": exc.pack_version,
            "rule_version": exc.rule_version,
            "regulatory_regimes": exc.regulatory_regimes,
            "matched_rule_ids": exc.matched_rule_ids,
            "matched_control_ids": exc.matched_control_ids,
            "policy_references": exc.policy_references,
            "model_signals": exc.model_signals,
            "fastml_features": exc.fastml_features,
            "score_components": exc.score_components,
            "top_contributors": exc.top_contributors,
            "decision_evidence": exc.decision_evidence,
            "receipt": exc.receipt,
        },
    }


def parse_settings(workspace_root: Path) -> RuntimeSettings:
    api_key = os.getenv("ATEN_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing ATEN_API_KEY. Create one at https://start.atensecurity.com.")

    tenant_id = os.getenv("THOTH_TENANT_ID", "basistheory").strip() or "basistheory"
    api_url = os.getenv("THOTH_API_URL", f"https://enforce.{tenant_id}.atensecurity.com").strip()
    user_id = os.getenv("THOTH_USER_ID", f"{os.getenv('USER', 'operator')}@local").strip()
    return RuntimeSettings(
        api_key=api_key,
        tenant_id=tenant_id,
        api_url=api_url,
        user_id=user_id,
        workspace_root=workspace_root,
    )


def instrument_research_toolchain(
    settings: RuntimeSettings,
    *,
    enforcement: str,
) -> ResearchToolchain:
    toolchain = ResearchToolchain(settings.workspace_root)
    governed = thoth.instrument_toolchain(
        toolchain,
        agent_id="langgraph-research-agent",
        approved_scope=[
            "web_search",
            "fetch_external_api",
            "read_local_file",
            "write_local_file",
            "read_customer_records",
        ],
        tenant_id=settings.tenant_id,
        user_id=settings.user_id,
        enforcement=enforcement,
        api_key=settings.api_key,
        api_url=settings.api_url,
        purpose="internal-research",
        data_classification="internal",
        task_context={
            "initiated_by": settings.user_id,
            "task_id": "langgraph-runtime-demo",
        },
    )
    return governed


def run_graph_once(
    settings: RuntimeSettings,
    *,
    enforcement: str,
    objective: str,
    force_block: bool,
) -> None:
    print(f"\n=== mode={enforcement} force_block={force_block} ===")
    governed = instrument_research_toolchain(settings, enforcement=enforcement)
    graph = build_graph(build_tools(governed))

    try:
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=objective)],
                "objective": objective,
                "step": 0,
                "force_block": force_block,
            }
        )
        final_message = result["messages"][-1]
        print("Final message:")
        print(final_message.content)
    except ThothPolicyViolation as exc:
        print("Policy violation raised by governed tool call:")
        print(json.dumps(violation_audit_record(exc), indent=2, sort_keys=True))


def preview_decisions(settings: RuntimeSettings) -> None:
    print("\n=== enforce decision preview ===")
    config = ThothConfig(
        agent_id="langgraph-research-agent",
        approved_scope=[
            "web_search",
            "fetch_external_api",
            "read_local_file",
            "write_local_file",
            "read_customer_records",
        ],
        tenant_id=settings.tenant_id,
        user_id=settings.user_id,
        enforcement=EnforcementMode.PROGRESSIVE,
        api_key=settings.api_key,
        api_url=settings.api_url,
    )
    session = SessionContext(config)
    enforcer = EnforcerClient(config)
    checks = [
        (
            "web_search",
            {
                "query": "langgraph ToolNode execution",
                "domain": "hn.algolia.com",
            },
        ),
        (
            "read_customer_records",
            {
                "record_count": 12,
            },
        ),
        (
            "write_local_file",
            {
                "path": "tmp/research_brief.md",
            },
        ),
        (
            "fetch_external_api",
            {
                "url": "https://example.com",
                "domain": urlparse("https://example.com").hostname,
            },
        ),
    ]
    prior_calls: list[str] = []
    try:
        for tool_name, tool_args in checks:
            call_args = dict(tool_args)
            if "domain" not in call_args and "url" in call_args:
                call_args["domain"] = urlparse(str(call_args["url"])).hostname

            decision = enforcer.check(
                tool_name=tool_name,
                session_id=session.session_id,
                tool_calls=prior_calls + [tool_name],
                tool_args=call_args,
            )
            prior_calls.append(tool_name)
            print(f"\n{tool_name}: {decision.decision.value}")
            print(json.dumps(decision_audit_record(decision), indent=2, sort_keys=True))
    finally:
        enforcer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangGraph runtime governance demo using Thoth observe + enforce modes.",
    )
    parser.add_argument(
        "--objective",
        default="Research pre-execution controls for LangGraph agents.",
        help="Research objective used by the planner node.",
    )
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root used by read/write file tools.",
    )
    args = parser.parse_args()

    settings = parse_settings(Path(args.workspace_root).resolve())

    # Shadow mode: observe only.
    run_graph_once(
        settings,
        enforcement="observe",
        objective=args.objective,
        force_block=False,
    )

    # Enforce mode: same graph with runtime policy checks enabled.
    preview_decisions(settings)
    run_graph_once(
        settings,
        enforcement="progressive",
        objective=args.objective,
        force_block=False,
    )
    run_graph_once(
        settings,
        enforcement="progressive",
        objective=args.objective,
        force_block=True,
    )


if __name__ == "__main__":
    main()
