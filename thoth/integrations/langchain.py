"""
LangChain integration for Thoth.

Usage:
    import thoth
    from langchain.agents import AgentExecutor
    agent = AgentExecutor(tools=[...], ...)
    agent = thoth.instrument(agent, agent_id="...", ...)
"""

from __future__ import annotations

from typing import Any

from thoth.tracer import Tracer


def wrap_langchain_agent(agent: Any, tracer: Tracer) -> Any:
    """Wrap LangChain AgentExecutor tools."""
    tools = getattr(agent, "tools", [])
    for tool in tools:
        tool_name = getattr(tool, "name", str(tool))
        if hasattr(tool, "run"):
            tool.run = tracer.wrap_tool(tool_name, tool.run)
        if hasattr(tool, "_run"):
            tool._run = tracer.wrap_tool(tool_name, tool._run)
    return agent
