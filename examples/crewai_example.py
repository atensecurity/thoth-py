"""
Thoth + CrewAI — getting started example.

Prerequisites:
    pip install "atensec-thoth" crewai

Environment variables:
    THOTH_API_KEY      — your Thoth API key (get one at https://app.atensecurity.com)
    OPENAI_API_KEY     — your OpenAI API key (CrewAI default LLM)
    THOTH_TENANT_ID    — your Thoth tenant ID
"""

import os

import thoth
from crewai import Agent, Crew, Task
from crewai.tools import tool
from thoth import ThothPolicyViolation

# ---------------------------------------------------------------------------
# 1. Define your toolchain
# ---------------------------------------------------------------------------


class CrewAIToolchain:
    """Single root object for all tool implementations."""

    def web_search(self, query: str) -> str:
        """Search the web for current information."""
        return f"[web results for '{query}']"

    def send_report(self, recipient: str, content: str) -> str:
        """Email a report to a recipient."""
        return f"Report sent to {recipient}"

    def write_to_database(self, table: str, data: str) -> str:
        """Insert data into a database table."""
        return f"Written to {table}"


governed = thoth.instrument_toolchain(
    CrewAIToolchain(),
    agent_id="crewai-toolchain",
    approved_scope=["web_search", "send_report"],  # write_to_database NOT in scope → blocked
    tenant_id=os.environ["THOTH_TENANT_ID"],
    user_id="alice@acme.com",
    enforcement="block",
)

web_search_tool = tool("web_search")(governed.web_search)
send_report_tool = tool("send_report")(governed.send_report)
write_to_database_tool = tool("write_to_database")(governed.write_to_database)


# ---------------------------------------------------------------------------
# 2. Build CrewAI agents
# ---------------------------------------------------------------------------

researcher = Agent(
    role="Researcher",
    goal="Find accurate, up-to-date information on assigned topics",
    backstory="You are a detail-oriented research analyst.",
    tools=[web_search_tool],
    verbose=True,
)

reporter = Agent(
    role="Report Writer",
    goal="Summarize research findings and distribute them",
    backstory="You are a concise technical writer.",
    tools=[send_report_tool, write_to_database_tool],
    verbose=True,
)

# ---------------------------------------------------------------------------
# 3. Define tasks and run the crew
# ---------------------------------------------------------------------------

research_task = Task(
    description="Research the latest AI governance regulations in the EU",
    expected_output="A 3-paragraph summary of key findings",
    agent=researcher,
)

report_task = Task(
    description="Send a report of the research findings to compliance@acme.com",
    expected_output="Confirmation that the report was sent",
    agent=reporter,
)

crew = Crew(agents=[researcher, reporter], tasks=[research_task, report_task], verbose=True)

try:
    result = crew.kickoff()
    print("Crew result:", result)
except ThothPolicyViolation as e:
    print(f"⚠ Policy violation on '{e.tool_name}': {e.reason}")
