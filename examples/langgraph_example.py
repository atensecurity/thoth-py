"""
Thoth + LangGraph ReAct agent — getting started example.

LangGraph is the modern way to build stateful agents with LangChain.
Thoth wraps tools at the function level, so it works with any LangGraph
agent regardless of topology.

Prerequisites:
    pip install "atensec-thoth" langgraph langchain-anthropic

Environment variables:
    THOTH_API_KEY      — your Thoth API key (get one at https://app.atensecurity.com)
    ANTHROPIC_API_KEY  — your Anthropic API key
    THOTH_TENANT_ID    — your Thoth tenant ID
"""

import os

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

import thoth
from thoth import ThothPolicyViolation

# ---------------------------------------------------------------------------
# 1. Define your tools
# ---------------------------------------------------------------------------


class AnalyticsToolchain:
    """Single root object for all tool implementations."""

    def search_database(self, query: str) -> str:
        """Search the internal knowledge base."""
        return f"[knowledge base results for '{query}']"

    def send_slack_message(self, channel: str, message: str) -> str:
        """Send a message to a Slack channel."""
        return f"Message sent to #{channel}"

    def export_data(self, table: str, file_format: str = "csv") -> str:
        """Export data from a database table."""
        return f"[{file_format} export of {table}]"


# ---------------------------------------------------------------------------
# 2. Instrument the full toolchain with one SDK call
# ---------------------------------------------------------------------------

governed = thoth.instrument_toolchain(
    AnalyticsToolchain(),
    agent_id="langgraph-analyst",
    approved_scope=["search_database", "send_slack_message"],  # export_data not in scope → blocked
    tenant_id=os.environ["THOTH_TENANT_ID"],
    user_id="alice@acme.com",
    enforcement="block",
)

# ---------------------------------------------------------------------------
# 3. Build and run the LangGraph agent
# ---------------------------------------------------------------------------

llm = ChatAnthropic(model="claude-sonnet-4-6")

# Pass governed callables as tools
agent = create_react_agent(
    llm,
    [governed.search_database, governed.send_slack_message, governed.export_data],
)

try:
    result = agent.invoke({"messages": [("user", "Search for Q4 sales data and post a summary to #analytics")]})
    print("Agent:", result["messages"][-1].content)
except ThothPolicyViolation as e:
    print(f"⚠ Policy violation on '{e.tool_name}': {e.reason}")
    if e.violation_id:
        print(f"  Violation ID: {e.violation_id} — view at https://app.atensecurity.com")
