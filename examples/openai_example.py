"""
Thoth + OpenAI — getting started example.

Prerequisites:
    pip install "aten-thoth" openai

Environment variables:
    THOTH_API_KEY      — your Thoth API key (get one at https://app.atensecurity.com)
    OPENAI_API_KEY     — your OpenAI API key
    THOTH_TENANT_ID    — your Thoth tenant ID
"""

import json
import os

import openai

import thoth
from thoth import ThothPolicyViolation

# ---------------------------------------------------------------------------
# 1. Define your tool schemas (for OpenAI) and implementations (for you)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search internal documentation",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
]


def search_docs(query: str) -> str:
    """Your real implementation."""
    return f"[search results for '{query}']"


def send_email(to: str, subject: str, body: str) -> str:
    """Your real implementation."""
    return f"Email sent to {to}"


# ---------------------------------------------------------------------------
# 2. Instrument tool functions with Thoth
#    THOTH_API_KEY is read from the environment automatically.
# ---------------------------------------------------------------------------

governed = thoth.instrument_openai(
    {"search_docs": search_docs, "send_email": send_email},
    agent_id="openai-support-bot",
    approved_scope=["search_docs"],  # send_email is NOT in scope → will be blocked
    tenant_id=os.environ["THOTH_TENANT_ID"],
    user_id="alice@acme.com",
    enforcement="block",
    # api_key="your-key-here",  # or set THOTH_API_KEY env var
)

# ---------------------------------------------------------------------------
# 3. Standard OpenAI agentic loop — no changes needed here
# ---------------------------------------------------------------------------

client = openai.OpenAI()
messages = [{"role": "user", "content": "Find docs about access control and email the summary to bob@acme.com"}]

while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        tools=TOOLS,
        messages=messages,
    )
    msg = response.choices[0].message
    messages.append(msg)

    if not msg.tool_calls:
        print("Agent:", msg.content)
        break

    for call in msg.tool_calls:
        fn = governed.get(call.function.name)
        if not fn:
            continue
        args = json.loads(call.function.arguments)
        try:
            result = fn(**args)  # Thoth enforcement runs here
        except ThothPolicyViolation as e:
            result = f"[blocked by policy: {e.reason}]"
            print(f"⚠ Policy violation on '{e.tool_name}': {e.reason}")

        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            }
        )
