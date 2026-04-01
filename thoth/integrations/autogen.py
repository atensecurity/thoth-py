"""
AutoGen (pyautogen / ag2) integration for Thoth.

Wraps the function map used by AutoGen's ``FunctionCallingAgent`` /
``AssistantAgent`` with Thoth governance so every function execution is
policy-checked before it runs.

Usage::

    import os
    import autogen
    import thoth
    from thoth.integrations.autogen import wrap_autogen_tools

    def search_docs(query: str) -> str:
        ...

    def send_email(to: str, subject: str, body: str) -> str:
        ...

    # Wrap the function map
    governed = wrap_autogen_tools(
        {"search_docs": search_docs, "send_email": send_email},
        agent_id="autogen-assistant",
        approved_scope=["search_docs"],          # send_email will be blocked
        tenant_id="acme-corp",
        user_id="alice@acme.com",
        api_key=os.environ["THOTH_API_KEY"],
        enforcement="block",
    )

    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config={"functions": [...], "config_list": [...]},
    )
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        function_map=governed,   # governed functions injected here
    )
"""

from __future__ import annotations

from typing import Any, Callable

from thoth.tracer import Tracer


def wrap_autogen_tools(
    tool_fns: dict[str, Callable[..., Any]],
    tracer: Tracer,
) -> dict[str, Callable[..., Any]]:
    """Wrap an AutoGen function map with Thoth governance.

    Args:
        tool_fns: AutoGen ``function_map`` dict — tool name → callable.
        tracer: A configured :class:`thoth.tracer.Tracer` instance.

    Returns:
        A new dict with the same keys but governance-wrapped callables.
    """
    return {name: tracer.wrap_tool(name, fn) for name, fn in tool_fns.items()}
