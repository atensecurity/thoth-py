"""Reference LangGraph healthcare workflow with first-class Thoth instrumentation.

Run:
    THOTH_MOCK_MODE=true python langgraph_healthcare_agent.py

Optional env vars:
    THOTH_API_URL        (default: https://enforcer.mock.local)
    THOTH_API_KEY        (optional in mock mode)
    THOTH_TENANT_ID      (default: abridge)
    THOTH_USER_ID        (default: clinician@example.com)
    THOTH_ENFORCEMENT    (default: step_up in mock mode, block otherwise)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Literal, TypedDict
import uuid

try:
    import langchain  # type: ignore[import-not-found]

    if not hasattr(langchain, "debug"):
        langchain.debug = False  # type: ignore[attr-defined]
except Exception:
    langchain = None  # type: ignore[assignment]

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

import thoth
from thoth import ThothPolicyViolation


@dataclass(frozen=True)
class RuntimeSettings:
    tenant_id: str
    user_id: str
    api_url: str
    api_key: str | None
    enforcement: str


class AgentState(MessagesState):
    step: int
    patient_id: str
    visit_id: str


def _tool_call(call_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "name": name,
        "args": args,
        "type": "tool_call",
    }


@tool("retrieve_patient_record")
def retrieve_patient_record(patient_id: str) -> dict[str, Any]:
    """Retrieve patient demographics and active issues."""

    return {
        "patient_id": patient_id,
        "name": "Jordan Smith",
        "dob": "1984-01-14",
        "allergies": ["penicillin"],
        "active_problems": ["hypertension"],
    }


@tool("check_clinical_formulary")
def check_clinical_formulary(medication: str, patient_id: str) -> dict[str, Any]:
    """Check medication against mock formulary and patient profile."""

    return {
        "patient_id": patient_id,
        "medication": medication,
        "covered": medication.lower() != "experimental-drug-x",
        "tier": 1,
    }


@tool("generate_clinical_note")
def generate_clinical_note(visit_id: str, transcript: str) -> str:
    """Generate a structured mock clinical note."""

    return f"Visit {visit_id}:\n" "Assessment: Mild hypertension, stable.\n" f"Source transcript: {transcript[:120]}"


@tool("write_to_ehr")
def write_to_ehr(patient_id: str, note: str, visit_id: str) -> dict[str, Any]:
    """Persist note to EHR (sensitive PHI write operation)."""

    return {
        "status": "written",
        "patient_id": patient_id,
        "visit_id": visit_id,
        "record_id": f"ehr_{uuid.uuid4().hex[:10]}",
        "note_preview": note[:80],
    }


def build_settings() -> RuntimeSettings:
    mock_mode = os.getenv("THOTH_MOCK_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    enforcement_default = "step_up" if mock_mode else "block"
    return RuntimeSettings(
        tenant_id=os.getenv("THOTH_TENANT_ID", "abridge").strip() or "abridge",
        user_id=os.getenv("THOTH_USER_ID", "clinician@example.com").strip() or "clinician@example.com",
        api_url=os.getenv("THOTH_API_URL", "https://enforcer.mock.local").strip() or "https://enforcer.mock.local",
        api_key=(os.getenv("THOTH_API_KEY") or "").strip() or None,
        enforcement=(os.getenv("THOTH_ENFORCEMENT") or enforcement_default).strip().lower(),
    )


def planner_node(state: AgentState) -> dict[str, Any]:
    step = int(state.get("step", 0))
    patient_id = str(state.get("patient_id", "pt_1001"))
    visit_id = str(state.get("visit_id", "visit_9001"))

    if step == 0:
        return {
            "step": 1,
            "messages": [
                AIMessage(
                    content="Retrieving patient record.",
                    tool_calls=[
                        _tool_call(
                            call_id=f"tool-{uuid.uuid4().hex[:8]}",
                            name="retrieve_patient_record",
                            args={"patient_id": patient_id},
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
                    content="Checking formulary for lisinopril.",
                    tool_calls=[
                        _tool_call(
                            call_id=f"tool-{uuid.uuid4().hex[:8]}",
                            name="check_clinical_formulary",
                            args={
                                "medication": "lisinopril",
                                "patient_id": patient_id,
                            },
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
                    content="Generating clinical note draft.",
                    tool_calls=[
                        _tool_call(
                            call_id=f"tool-{uuid.uuid4().hex[:8]}",
                            name="generate_clinical_note",
                            args={
                                "visit_id": visit_id,
                                "transcript": "Patient reports occasional headache, blood pressure stable at home.",
                            },
                        )
                    ],
                )
            ],
        }

    if step == 3:
        note = "Assessment: stable hypertension. Plan: continue lisinopril 10mg daily."
        return {
            "step": 4,
            "messages": [
                AIMessage(
                    content="Attempting EHR write (sensitive).",
                    tool_calls=[
                        _tool_call(
                            call_id=f"tool-{uuid.uuid4().hex[:8]}",
                            name="write_to_ehr",
                            args={
                                "patient_id": patient_id,
                                "note": note,
                                "visit_id": visit_id,
                            },
                        )
                    ],
                )
            ],
        }

    return {"messages": [AIMessage(content=("Clinical workflow completed. Review audit records for per-tool " "ALLOW/BLOCK/STEP_UP outcomes."))]}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def build_graph(settings: RuntimeSettings):
    tools = [
        retrieve_patient_record,
        check_clinical_formulary,
        generate_clinical_note,
        write_to_ehr,
    ]

    governed_tools = thoth.instrument_langgraph(
        tools,
        agent_id="healthcare-workflow",
        approved_scope=[
            "retrieve_patient_record",
            "check_clinical_formulary",
            "generate_clinical_note",
        ],
        tenant_id=settings.tenant_id,
        user_id=settings.user_id,
        enforcement=settings.enforcement,
        api_key=settings.api_key,
        api_url=settings.api_url,
        session_intent="clinical-workflow-automation",
        purpose="clinical-documentation",
        data_classification="PHI",
        task_context="Abridge SWAT LangGraph workflow",
    )

    graph = StateGraph(AgentState)
    graph.add_node("agent", planner_node)
    graph.add_node("tools", ToolNode(governed_tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()


def main() -> None:
    settings = build_settings()
    print("== Thoth LangGraph Healthcare Agent ==")
    print(
        json.dumps(
            {
                "tenant_id": settings.tenant_id,
                "user_id": settings.user_id,
                "api_url": settings.api_url,
                "enforcement": settings.enforcement,
                "mock_mode": os.getenv("THOTH_MOCK_MODE", "false"),
            },
            indent=2,
        )
    )

    app = build_graph(settings)
    violations = 0

    try:
        result = app.invoke(
            {
                "messages": [HumanMessage(content="Run the clinical workflow for patient pt_1001.")],
                "step": 0,
                "patient_id": "pt_1001",
                "visit_id": "visit_9001",
            }
        )
        final_message = result["messages"][-1]
        print("\nFinal agent response:")
        print(final_message.content)
    except ThothPolicyViolation as exc:
        violations += 1
        print("\nPolicy violation:")
        print(f"- tool: {exc.tool_name}")
        print(f"- reason: {exc.reason}")
        if exc.violation_id:
            print(f"- violation_id: {exc.violation_id}")

    session = thoth.get_current_session()
    tool_calls = session.tool_calls if session else []
    write_calls = sum(1 for name in tool_calls if name == "write_to_ehr")

    print("\nSummary:")
    print(f"- tool calls: {len(tool_calls)}")
    print(f"- unique tools: {sorted(set(tool_calls))}")
    print(f"- write_to_ehr calls: {write_calls}")
    print(f"- violations blocked/raised: {violations}")
    if settings.enforcement == "step_up":
        print("- step_up expected for write_to_ehr in mock mode")


if __name__ == "__main__":
    main()
