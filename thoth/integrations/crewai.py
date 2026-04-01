"""
CrewAI integration for Thoth.

Wraps CrewAI agent tools so every tool execution is policy-checked and
emitted as a behavioral event before the tool runs.

Usage::

    import os
    from crewai import Agent, Task, Crew
    from crewai.tools import tool
    import thoth
    from thoth.integrations.crewai import wrap_crewai_agent

    @tool("web_search")
    def web_search(query: str) -> str:
        \"\"\"Search the web for information.\"\"\"
        ...

    researcher = Agent(
        role="Researcher",
        goal="Find accurate information",
        tools=[web_search],
    )

    # Wrap agent tools with Thoth governance
    wrap_crewai_agent(
        researcher,
        agent_id="crewai-researcher",
        approved_scope=["web_search"],
        tenant_id="acme-corp",
        user_id="alice@acme.com",
    )

    # Or use thoth.instrument() directly — it detects CrewAI agents automatically
    thoth.instrument(
        researcher,
        agent_id="crewai-researcher",
        approved_scope=["web_search"],
        tenant_id="acme-corp",
    )
"""

from __future__ import annotations

from typing import Any

from thoth.tracer import Tracer


def wrap_crewai_agent(agent: Any, tracer: Tracer) -> Any:
    """Wrap a CrewAI Agent's tools with Thoth governance.

    Iterates over ``agent.tools`` and replaces the ``_run`` method (and
    ``run`` if present) on each tool with a governed wrapper.

    Args:
        agent: A ``crewai.Agent`` instance.
        tracer: A configured :class:`thoth.tracer.Tracer` instance.

    Returns:
        The same agent object with tools wrapped in-place.
    """
    tools = getattr(agent, "tools", []) or []
    for tool in tools:
        tool_name = str(getattr(tool, "name", None) or getattr(tool, "__name__", None) or tool)
        # CrewAI tools expose _run; some also expose run
        if hasattr(tool, "_run"):
            tool._run = tracer.wrap_tool(tool_name, tool._run)
        if hasattr(tool, "run"):
            tool.run = tracer.wrap_tool(tool_name, tool.run)
    return agent
