"""
Thoth + LangGraph ReAct agent — getting started example.

LangGraph is the modern way to build stateful agents with LangChain.
Thoth wraps tools at the function level, so it works with any LangGraph
agent regardless of topology.

Prerequisites:
    pip install "aten-thoth" langgraph langchain-anthropic

Environment variables:
    THOTH_API_KEY      — your Thoth API key (get one at https://app.aten.security)
    ANTHROPIC_API_KEY  — your Anthropic API key
    THOTH_TENANT_ID    — your Thoth tenant ID
"""

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from thoth import ThothPolicyViolation
from thoth.emitter import HttpEmitter
from thoth.enforcer_client import EnforcerClient
from thoth.models import EnforcementMode, ThothConfig
from thoth.session import SessionContext
from thoth.step_up import StepUpClient
from thoth.tracer import Tracer

# ---------------------------------------------------------------------------
# 1. Define your tools
# ---------------------------------------------------------------------------


@tool
def search_database(query: str) -> str:
    """Search the internal knowledge base."""
    return f"[knowledge base results for '{query}']"


@tool
def send_slack_message(channel: str, message: str) -> str:
    """Send a message to a Slack channel."""
    return f"Message sent to #{channel}"


@tool
def export_data(table: str, file_format: str = "csv") -> str:
    """Export data from a database table."""
    return f"[{file_format} export of {table}]"


# ---------------------------------------------------------------------------
# 2. Build a governed Tracer and wrap tools individually
#    LangGraph tools are plain callables — use Tracer.wrap_tool directly.
# ---------------------------------------------------------------------------

api_key = os.environ["THOTH_API_KEY"]
tenant_id = os.environ["THOTH_TENANT_ID"]

config = ThothConfig(
    agent_id="langgraph-analyst",
    approved_scope=["search_database", "send_slack_message"],  # export_data not in scope → blocked
    tenant_id=tenant_id,
    user_id="alice@acme.com",
    enforcement=EnforcementMode.BLOCK,
    api_key=api_key,
)

session = SessionContext(config)
tracer = Tracer(
    config=config,
    session=session,
    emitter=HttpEmitter(api_url=config.api_url, api_key=api_key),
    enforcer=EnforcerClient(config),
    step_up=StepUpClient(config),
)

# Wrap each tool function — the @tool decorator wraps around the governed fn
governed_search = tracer.wrap_tool("search_database", search_database)
governed_slack = tracer.wrap_tool("send_slack_message", send_slack_message)
governed_export = tracer.wrap_tool("export_data", export_data)

# ---------------------------------------------------------------------------
# 3. Build and run the LangGraph agent
# ---------------------------------------------------------------------------

llm = ChatAnthropic(model="claude-sonnet-4-6")

# Pass governed callables as tools
agent = create_react_agent(llm, [governed_search, governed_slack, governed_export])

try:
    result = agent.invoke({"messages": [("user", "Search for Q4 sales data and post a summary to #analytics")]})
    print("Agent:", result["messages"][-1].content)
except ThothPolicyViolation as e:
    print(f"⚠ Policy violation on '{e.tool_name}': {e.reason}")
    if e.violation_id:
        print(f"  Violation ID: {e.violation_id} — view at https://app.aten.security")
