"""
Thoth + claude-agent-sdk query() example.

Prerequisites:
    pip install "aten-thoth[claude]"

Environment variables:
    THOTH_API_KEY     — your Thoth API key
    THOTH_TENANT_ID   — your Thoth tenant ID
    THOTH_API_URL     — your Thoth enforcer URL
"""

from __future__ import annotations

import os
from pathlib import Path

import anyio
from claude_agent_sdk import ClaudeAgentOptions, query

import thoth


async def prompt_stream():
    # claude-agent-sdk requires streaming mode for can_use_tool callbacks.
    yield {
        "type": "user",
        "message": {"role": "user", "content": "Read README.md and summarize security risks."},
        "parent_tool_use_id": None,
        "session_id": "thoth-sdk-demo-session",
    }


async def main() -> None:
    options = ClaudeAgentOptions(
        max_turns=1,
        allowed_tools=["Read"],  # Claude SDK auto-approval list
        disallowed_tools=["Bash"],  # explicit hard deny in Claude SDK
        cwd=Path.cwd(),
    )

    options = thoth.instrument_claude_agent_sdk(
        options,
        agent_id="claude-agent-sdk-demo",
        approved_scope=["Read"],  # Thoth policy scope
        tenant_id=os.environ["THOTH_TENANT_ID"],
        user_id="alice@example.com",
        enforcement="block",
        # api_key read from THOTH_API_KEY, api_url from THOTH_API_URL
    )

    async for message in query(prompt=prompt_stream(), options=options):
        print(message)


if __name__ == "__main__":
    anyio.run(main)
