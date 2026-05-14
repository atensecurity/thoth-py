"""
Thoth + Anthropic Claude — getting started example.

Prerequisites:
    pip install "atensec-thoth" anthropic

Environment variables:
    THOTH_API_KEY      — your Thoth API key (get one at https://app.atensecurity.com)
    ANTHROPIC_API_KEY  — your Anthropic API key
    THOTH_TENANT_ID    — your Thoth tenant ID
"""

import os

import anthropic

import thoth
from thoth import ThothPolicyViolation

# ---------------------------------------------------------------------------
# 1. Define tool schemas (for Claude) and implementations (for you)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_docs",
        "description": "Search internal documentation",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "delete_record",
        "description": "Permanently delete a record from the database",
        "input_schema": {
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
        },
    },
]


def search_docs(query: str) -> str:
    """Your real implementation."""
    return f"[search results for '{query}']"


def delete_record(record_id: str) -> str:
    """Your real implementation."""
    return f"Record {record_id} deleted"


# ---------------------------------------------------------------------------
# 2. Instrument with Thoth
#    THOTH_API_KEY is read from the environment automatically.
# ---------------------------------------------------------------------------

governed = thoth.instrument_anthropic(
    {"search_docs": search_docs, "delete_record": delete_record},
    agent_id="claude-research-agent",
    approved_scope=["search_docs"],  # delete_record NOT in scope → step-up or block
    tenant_id=os.environ["THOTH_TENANT_ID"],
    user_id="alice@acme.com",
    enforcement="step_up",  # escalate sensitive actions instead of hard-blocking
    # api_key="your-key-here",  # or set THOTH_API_KEY env var
)

# ---------------------------------------------------------------------------
# 3. Standard Anthropic agentic loop — no changes needed here
# ---------------------------------------------------------------------------

client = anthropic.Anthropic()
messages = [{"role": "user", "content": "Search for our data retention policy and summarize it."}]

while True:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        tools=TOOLS,
        messages=messages,
    )

    if response.stop_reason == "end_turn":
        for block in response.content:
            if hasattr(block, "text"):
                print("Agent:", block.text)
        break

    tool_results = []
    for block in response.content:
        if block.type != "tool_use":
            continue
        fn = governed.get(block.name)
        if not fn:
            continue
        try:
            result = fn(block.input)  # Thoth enforcement runs here
        except ThothPolicyViolation as e:
            result = f"[blocked by policy: {e.reason}]"
            print(f"⚠ Policy violation on '{e.tool_name}': {e.reason}")

        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result),
            }
        )

    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results})
