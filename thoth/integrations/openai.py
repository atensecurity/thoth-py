"""
OpenAI integration for Thoth.

This module wraps tool execution functions with Thoth governance for use in
an OpenAI function-calling / tool-calling loop. OpenAI's chat completions
return ``tool_calls`` on the assistant message; the developer executes them
and sends back ``tool`` role messages. Thoth intercepts at execution time.

Usage::

    from openai import OpenAI
    import thoth
    from thoth.integrations.openai import wrap_openai_tools
    import json

    client = OpenAI()

    # Define tools for OpenAI (schema only)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search internal documentation",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    # Wrap tool execution functions with Thoth governance
    raw_fns = {"search_docs": my_search_fn}
    wrapped_fns = thoth.instrument_openai(
        raw_fns,
        agent_id="support-bot-v2",
        approved_scope=["search_docs"],
        tenant_id="acme-corp",
    )

    # Standard OpenAI agentic loop
    messages = [{"role": "user", "content": "Find docs about access control"}]
    while True:
        response = client.chat.completions.create(
            model="gpt-5",
            tools=tools,
            messages=messages,
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            break
        messages.append(msg)
        for call in msg.tool_calls:
            fn = wrapped_fns.get(call.function.name)
            if fn:
                args = json.loads(call.function.arguments)
                result = fn(args)  # Thoth governance runs here
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                })
"""

from __future__ import annotations

from typing import Any, Callable

from thoth.tracer import Tracer


def wrap_openai_tools(
    tool_fns: dict[str, Callable[..., Any]],
    tracer: Tracer,
) -> dict[str, Callable[..., Any]]:
    """Wrap a dict of tool functions for use in an OpenAI tool-calling loop.

    Args:
        tool_fns: Mapping of tool name → callable. Each callable receives
            the parsed arguments dict from a ``tool_calls`` entry.
        tracer: A configured :class:`thoth.tracer.Tracer` instance.

    Returns:
        A new dict with the same keys but governance-wrapped callables.
        Calling any wrapped function triggers policy enforcement and emits
        a behavioral event.
    """
    return {name: tracer.wrap_tool(name, fn) for name, fn in tool_fns.items()}
