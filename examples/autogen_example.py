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

from thoth.emitter import HttpEmitter
from thoth.enforcer_client import EnforcerClient
from thoth.integrations.autogen import wrap_autogen_tools
from thoth.models import EnforcementMode, ThothConfig
from thoth.session import SessionContext
from thoth.step_up import StepUpClient
from thoth.tracer import Tracer

# ---------------------------------------------------------------------------
# 1. Define your tool functions
# ---------------------------------------------------------------------------


def search_docs(query: str) -> str:
    """Search internal documentation."""
    return f"[docs results for '{query}']"


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"


def delete_record(record_id: str) -> str:
    """Delete a record from the database."""
    return f"Record {record_id} deleted"


# ---------------------------------------------------------------------------
# 2. Build Thoth tracer and wrap the function map
# ---------------------------------------------------------------------------

api_key = os.environ["THOTH_API_KEY"]
tenant_id = os.environ["THOTH_TENANT_ID"]

config = ThothConfig(
    agent_id="autogen-assistant",
    approved_scope=["search_docs", "send_email"],  # delete_record NOT in scope → blocked
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

governed = wrap_autogen_tools(
    {"search_docs": search_docs, "send_email": send_email, "delete_record": delete_record},
    tracer=tracer,
)

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
    function_map=governed,  # governed functions — Thoth enforcement runs here
)

# ---------------------------------------------------------------------------
# 4. Run
# ---------------------------------------------------------------------------

user_proxy.initiate_chat(
    assistant,
    message="Search for our GDPR policy docs and email a summary to compliance@acme.com",
)
