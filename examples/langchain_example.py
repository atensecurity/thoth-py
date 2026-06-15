"""
Thoth + LangChain AgentExecutor — getting started example.

Prerequisites:
    pip install "atensec-thoth[langchain]" langchain langchain-openai

Environment variables:
    THOTH_API_KEY      — your Thoth API key (get one at https://app.atensecurity.com)
    OPENAI_API_KEY     — your OpenAI API key
    THOTH_TENANT_ID    — your Thoth tenant ID
"""

import os

import thoth
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from thoth import ThothPolicyViolation

# ---------------------------------------------------------------------------
# 1. Define your toolchain
# ---------------------------------------------------------------------------


class LangChainToolchain:
    """Single root object for all tool implementations."""

    def web_search(self, query: str) -> str:
        """Search the web for current information."""
        return f"[web results for '{query}']"

    def read_file(self, path: str) -> str:
        """Read a file from the local filesystem."""
        with open(path) as f:
            return f.read()

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file."""
        with open(path, "w") as f:
            f.write(content)
        return f"Written to {path}"


governed = thoth.instrument_toolchain(
    LangChainToolchain(),
    agent_id="langchain-research-agent",
    approved_scope=["web_search", "read_file"],  # write_file NOT in scope → blocked
    tenant_id=os.environ["THOTH_TENANT_ID"],
    user_id="alice@acme.com",
    enforcement="block",
)

tools = [
    tool("web_search")(governed.web_search),
    tool("read_file")(governed.read_file),
    tool("write_file")(governed.write_file),
]


# ---------------------------------------------------------------------------
# 2. Build the LangChain agent
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o", temperature=0)

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
# 3. Run the agent — Thoth governs every tool call transparently
# ---------------------------------------------------------------------------

try:
    result = executor.invoke({"input": "Search for GDPR requirements and summarize them."})
    print("Result:", result["output"])
except ThothPolicyViolation as e:
    print(f"⚠ Policy violation on '{e.tool_name}': {e.reason}")
