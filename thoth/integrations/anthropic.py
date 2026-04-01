"""
Anthropic Claude integration for Thoth.

This module wraps tool execution functions with Thoth governance for use in
an Anthropic agentic loop. The Anthropic SDK does not have an executor
object — the developer implements the loop and calls tools from ``tool_use``
content blocks. Thoth intercepts at the tool execution level.

Usage::

    import anthropic
    import thoth
    from thoth.integrations.anthropic import wrap_anthropic_tools

    client = anthropic.Anthropic()

    # Define tools for Claude (schema only — no code here)
    tools = [
        {
            "name": "search_docs",
            "description": "Search internal documentation",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]

    # Wrap tool execution functions with Thoth governance
    raw_fns = {"search_docs": my_search_fn}
    wrapped_fns = thoth.instrument_anthropic(
        raw_fns,
        agent_id="support-bot-v2",
        approved_scope=["search_docs"],
        tenant_id="acme-corp",
    )

    # Standard Anthropic agentic loop
    messages = [{"role": "user", "content": "Find docs about access control"}]
    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )
        if response.stop_reason == "end_turn":
            break
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                fn = wrapped_fns.get(block.name)
                if fn:
                    result = fn(block.input)  # Thoth governance runs here
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
"""

from __future__ import annotations

from typing import Any, Callable

from thoth.tracer import Tracer


def wrap_anthropic_tools(
    tool_fns: dict[str, Callable[..., Any]],
    tracer: Tracer,
) -> dict[str, Callable[..., Any]]:
    """Wrap a dict of tool functions for use in an Anthropic Claude agentic loop.

    Args:
        tool_fns: Mapping of tool name → callable. Each callable receives
            the ``input`` dict from a ``tool_use`` content block.
        tracer: A configured :class:`thoth.tracer.Tracer` instance.

    Returns:
        A new dict with the same keys but governance-wrapped callables.
        Calling any wrapped function triggers policy enforcement and emits
        a behavioral event.
    """
    return {name: tracer.wrap_tool(name, fn) for name, fn in tool_fns.items()}
