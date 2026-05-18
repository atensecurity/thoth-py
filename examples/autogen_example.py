"""
Thoth + AutoGen (pyautogen / ag2) — getting started example.

Prerequisites:
    pip install "atensec-thoth" pyautogen

Environment variables:
    THOTH_API_KEY      — your Thoth API key (get one at https://app.atensecurity.com)
    OPENAI_API_KEY     — your OpenAI API key
    THOTH_TENANT_ID    — your Thoth tenant ID
"""

import os

import autogen

import thoth

# ---------------------------------------------------------------------------
# 1. Define your tool functions
# ---------------------------------------------------------------------------


class AutoGenToolchain:
    """Single root object for all tool implementations."""

    def search_docs(self, query: str) -> str:
        return f"[docs results for '{query}']"

    def send_email(self, to: str, subject: str, body: str) -> str:
        return f"Email sent to {to}"

    def delete_record(self, record_id: str) -> str:
        return f"Record {record_id} deleted"


# ---------------------------------------------------------------------------
# 2. Instrument the full toolchain with one SDK call
# ---------------------------------------------------------------------------

governed = thoth.instrument_toolchain(
    AutoGenToolchain(),
    agent_id="autogen-assistant",
    approved_scope=["search_docs", "send_email"],  # delete_record NOT in scope → blocked
    tenant_id=os.environ["THOTH_TENANT_ID"],
    user_id="alice@acme.com",
    enforcement="block",
)
function_map = thoth.toolchain_function_map(governed)

# ---------------------------------------------------------------------------
# 3. Wire governed tools into AutoGen
# ---------------------------------------------------------------------------

config_list = [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list,
        "functions": [
            {
                "name": "search_docs",
                "description": "Search internal documentation",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "send_email",
                "description": "Send an email",
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
            {
                "name": "delete_record",
                "description": "Delete a record from the database",
                "parameters": {
                    "type": "object",
                    "properties": {"record_id": {"type": "string"}},
                    "required": ["record_id"],
                },
            },
        ],
    },
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    function_map=function_map,  # governed functions — Thoth enforcement runs here
)

# ---------------------------------------------------------------------------
# 4. Run
# ---------------------------------------------------------------------------

user_proxy.initiate_chat(
    assistant,
    message="Search for our GDPR policy docs and email a summary to compliance@acme.com",
)
