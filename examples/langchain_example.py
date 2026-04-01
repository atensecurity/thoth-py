"""
Thoth + LangChain AgentExecutor — getting started example.

Prerequisites:
    pip install "aten-thoth[langchain]" langchain langchain-openai

Environment variables:
    THOTH_API_KEY      — your Thoth API key (get one at https://app.aten.security)
    OPENAI_API_KEY     — your OpenAI API key
    THOTH_TENANT_ID    — your Thoth tenant ID
"""

import os

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

import thoth
from thoth import ThothPolicyViolation

# ---------------------------------------------------------------------------
# 1. Define your tools
# ---------------------------------------------------------------------------


@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    return f"[web results for '{query}']"


@tool
def read_file(path: str) -> str:
    """Read a file from the local filesystem."""
    with open(path) as f:
        return f.read()


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    with open(path, "w") as f:
        f.write(content)
    return f"Written to {path}"


# ---------------------------------------------------------------------------
# 2. Build the LangChain agent
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [web_search, read_file, write_file]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful research assistant."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---------------------------------------------------------------------------
# 3. Instrument with Thoth — one call wraps all tools automatically
#    THOTH_API_KEY is read from the environment automatically.
# ---------------------------------------------------------------------------

executor = thoth.instrument(
    executor,
    agent_id="langchain-research-agent",
    approved_scope=["web_search", "read_file"],  # write_file NOT in scope → blocked
    tenant_id=os.environ["THOTH_TENANT_ID"],
    user_id="alice@acme.com",
    enforcement="block",
    # api_key=os.environ["THOTH_API_KEY"],  # explicit override; env var is used by default
)

# ---------------------------------------------------------------------------
# 4. Run the agent — Thoth governs every tool call transparently
# ---------------------------------------------------------------------------

try:
    result = executor.invoke({"input": "Search for GDPR requirements and summarize them."})
    print("Result:", result["output"])
except ThothPolicyViolation as e:
    print(f"⚠ Policy violation on '{e.tool_name}': {e.reason}")
