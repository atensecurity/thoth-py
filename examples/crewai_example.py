"""
Thoth + CrewAI — getting started example.

Prerequisites:
    pip install "aten-thoth" crewai

Environment variables:
    THOTH_API_KEY      — your Thoth API key (get one at https://app.atensecurity.com)
    OPENAI_API_KEY     — your OpenAI API key (CrewAI default LLM)
    THOTH_TENANT_ID    — your Thoth tenant ID
"""

import os

from crewai import Agent, Crew, Task
from crewai.tools import tool

import thoth
from thoth import ThothPolicyViolation

# ---------------------------------------------------------------------------
# 1. Define your tools
# ---------------------------------------------------------------------------


@tool("web_search")
def web_search(query: str) -> str:
    """Search the web for current information."""
    return f"[web results for '{query}']"


@tool("send_report")
def send_report(recipient: str, content: str) -> str:
    """Email a report to a recipient."""
    return f"Report sent to {recipient}"


@tool("write_to_database")
def write_to_database(table: str, data: str) -> str:
    """Insert data into a database table."""
    return f"Written to {table}"


# ---------------------------------------------------------------------------
# 2. Build CrewAI agents
# ---------------------------------------------------------------------------

researcher = Agent(
    role="Researcher",
    goal="Find accurate, up-to-date information on assigned topics",
    backstory="You are a detail-oriented research analyst.",
    tools=[web_search],
    verbose=True,
)

reporter = Agent(
    role="Report Writer",
    goal="Summarize research findings and distribute them",
    backstory="You are a concise technical writer.",
    tools=[send_report, write_to_database],
    verbose=True,
)

# ---------------------------------------------------------------------------
# 3. Instrument each agent with Thoth
#    thoth.instrument() detects CrewAI agents automatically.
#    THOTH_API_KEY is read from the environment automatically.
# ---------------------------------------------------------------------------

thoth.instrument(
    researcher,
    agent_id="crewai-researcher",
    approved_scope=["web_search"],
    tenant_id=os.environ["THOTH_TENANT_ID"],
    user_id="alice@acme.com",
    enforcement="block",
    # api_key=os.environ["THOTH_API_KEY"],  # explicit override; env var is used by default
)

thoth.instrument(
    reporter,
    agent_id="crewai-reporter",
    approved_scope=["send_report"],  # write_to_database NOT in scope → blocked
    tenant_id=os.environ["THOTH_TENANT_ID"],
    user_id="alice@acme.com",
    enforcement="block",
)

# ---------------------------------------------------------------------------
# 4. Define tasks and run the crew
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
