# thoth/instrumentor.py
from __future__ import annotations

import os
from typing import Any, cast

from thoth._context import _CURRENT_SESSION
from thoth.emitter import HttpEmitter
from thoth.enforcer_client import EnforcerClient
from thoth.logging_config import configure_thoth_logging_from_env
from thoth.models import EnforcementMode, ThothConfig
from thoth.session import SessionContext
from thoth.step_up import StepUpClient
from thoth.tracer import Tracer


def _build_components(
    agent_id: str,
    approved_scope: list[str],
    tenant_id: str,
    user_id: str,
    enforcement: str,
    api_key: str | None,
    api_url: str | None,
    session_id: str | None,
    session_intent: str | None = None,
    environment: str | None = None,
    enforcement_trace_id: str | None = None,
    event_ingest_token: str | None = None,
) -> tuple[ThothConfig, SessionContext, HttpEmitter, EnforcerClient, StepUpClient, Tracer]:
    """Construct the full Thoth component stack from caller parameters."""
    configure_thoth_logging_from_env()
    resolved_api_key = api_key or os.getenv("THOTH_API_KEY")
    resolved_event_ingest_token = event_ingest_token or os.getenv("THOTH_EVENT_INGEST_TOKEN")
    resolved_api_url = (api_url or os.getenv("THOTH_API_URL") or "").strip()
    if not resolved_api_url:
        raise ValueError("Thoth API URL is required (pass api_url or set THOTH_API_URL)")
    resolved_environment = (environment or os.getenv("THOTH_ENVIRONMENT") or "prod").strip().lower() or "prod"
    config = ThothConfig(
        agent_id=agent_id,
        approved_scope=approved_scope,
        tenant_id=tenant_id,
        user_id=user_id,
        enforcement=EnforcementMode(enforcement),
        api_key=resolved_api_key,
        event_ingest_token=resolved_event_ingest_token,
        api_url=resolved_api_url,
        session_intent=session_intent,
        environment=resolved_environment,
        enforcement_trace_id=enforcement_trace_id,
    )
    session = SessionContext(config, session_id=session_id)
    _CURRENT_SESSION.set(session)
    emitter = HttpEmitter(
        api_url=config.resolved_api_url,
        api_key=resolved_api_key or "",
        event_ingest_token=config.resolved_event_ingest_token,
    )
    enforcer = EnforcerClient(config)
    step_up = StepUpClient(config)
    tracer = Tracer(config=config, session=session, emitter=emitter, enforcer=enforcer, step_up=step_up)
    return config, session, emitter, enforcer, step_up, tracer


def instrument(
    agent: Any,
    *,
    agent_id: str,
    approved_scope: list[str],
    tenant_id: str,
    user_id: str = "system",
    enforcement: str = "progressive",
    api_key: str | None = None,
    api_url: str | None = None,
    session_id: str | None = None,
    session_intent: str | None = None,
    environment: str | None = None,
    enforcement_trace_id: str | None = None,
    event_ingest_token: str | None = None,
) -> Any:
    """
    Instrument an AI agent with Thoth governance.
    Wraps all tools with enforce/emit hooks. Returns the same agent object.

    Args:
        session_intent: Declares the purpose of this session for HIPAA minimum-necessary
            enforcement. When the active compliance pack defines ``session_scopes``,
            tools outside the declared intent scope are step-up-challenged even if
            they appear in ``approved_scope``. Example: ``"phi_eligibility_check"``.
    """
    _, _, _, _, _, tracer = _build_components(
        agent_id,
        approved_scope,
        tenant_id,
        user_id,
        enforcement,
        api_key,
        api_url,
        session_id,
        session_intent,
        environment,
        enforcement_trace_id,
        event_ingest_token,
    )
    _wrap_agent_tools(agent, tracer)
    return agent


def instrument_anthropic(
    tool_fns: dict[str, Any],
    *,
    agent_id: str,
    approved_scope: list[str],
    tenant_id: str,
    user_id: str = "system",
    enforcement: str = "progressive",
    api_key: str | None = None,
    api_url: str | None = None,
    session_id: str | None = None,
    session_intent: str | None = None,
    environment: str | None = None,
    enforcement_trace_id: str | None = None,
    event_ingest_token: str | None = None,
) -> dict[str, Any]:
    """Instrument tool functions for use in an Anthropic Claude agentic loop.

    Wraps each callable in *tool_fns* with policy enforcement and behavioral
    event emission. Returns a new dict with the same keys but governed callables.

    Args:
        tool_fns: Mapping of tool name → callable (receives the ``input`` dict
            from a Claude ``tool_use`` content block).
        agent_id: Unique identifier for this agent.
        approved_scope: List of tool names that are in-policy.
        tenant_id: Your Thoth tenant identifier.
        user_id: User initiating the session. Defaults to ``"system"``.
        enforcement: ``"observe"`` | ``"block"`` | ``"step_up"`` |
            ``"progressive"`` (default).
        api_key: Thoth API key (or set ``THOTH_API_KEY`` env var).
        api_url: Optional tenant API base URL for both event ingestion and
            policy checks.
        session_id: Optional session ID; generated automatically if omitted.
        session_intent: Declares the purpose of this session for HIPAA
            minimum-necessary enforcement (e.g. ``"phi_eligibility_check"``).

    Returns:
        Dict of governance-wrapped callables keyed by tool name.
    """
    from thoth.integrations.anthropic import wrap_anthropic_tools

    _, _, _, _, _, tracer = _build_components(
        agent_id,
        approved_scope,
        tenant_id,
        user_id,
        enforcement,
        api_key,
        api_url,
        session_id,
        session_intent,
        environment,
        enforcement_trace_id,
        event_ingest_token,
    )
    return cast(dict[str, Any], wrap_anthropic_tools(tool_fns, tracer))


def instrument_openai(
    tool_fns: dict[str, Any],
    *,
    agent_id: str,
    approved_scope: list[str],
    tenant_id: str,
    user_id: str = "system",
    enforcement: str = "progressive",
    api_key: str | None = None,
    api_url: str | None = None,
    session_id: str | None = None,
    session_intent: str | None = None,
    environment: str | None = None,
    enforcement_trace_id: str | None = None,
    event_ingest_token: str | None = None,
) -> dict[str, Any]:
    """Instrument tool functions for use in an OpenAI tool-calling loop.

    Wraps each callable in *tool_fns* with policy enforcement and behavioral
    event emission. Returns a new dict with the same keys but governed callables.

    Args:
        tool_fns: Mapping of tool name → callable (receives the parsed
            arguments dict from an OpenAI ``tool_calls`` entry).
        agent_id: Unique identifier for this agent.
        approved_scope: List of tool names that are in-policy.
        tenant_id: Your Thoth tenant identifier.
        user_id: User initiating the session. Defaults to ``"system"``.
        enforcement: ``"observe"`` | ``"block"`` | ``"step_up"`` |
            ``"progressive"`` (default).
        api_key: Thoth API key (or set ``THOTH_API_KEY`` env var).
        api_url: Optional tenant API base URL for both event ingestion and
            policy checks.
        session_id: Optional session ID; generated automatically if omitted.
        session_intent: Declares the purpose of this session for HIPAA
            minimum-necessary enforcement (e.g. ``"phi_eligibility_check"``).

    Returns:
        Dict of governance-wrapped callables keyed by tool name.
    """
    from thoth.integrations.openai import wrap_openai_tools

    _, _, _, _, _, tracer = _build_components(
        agent_id,
        approved_scope,
        tenant_id,
        user_id,
        enforcement,
        api_key,
        api_url,
        session_id,
        session_intent,
        environment,
        enforcement_trace_id,
        event_ingest_token,
    )
    return cast(dict[str, Any], wrap_openai_tools(tool_fns, tracer))


def instrument_claude_agent_sdk(
    options: Any | None = None,
    *,
    agent_id: str,
    approved_scope: list[str],
    tenant_id: str,
    user_id: str = "system",
    enforcement: str = "progressive",
    api_key: str | None = None,
    api_url: str | None = None,
    session_id: str | None = None,
    session_intent: str | None = None,
    environment: str | None = None,
    enforcement_trace_id: str | None = None,
    event_ingest_token: str | None = None,
    emit_tool_lifecycle_hooks: bool = True,
) -> Any:
    """Instrument ``claude-agent-sdk`` options with Thoth governance.

    This wires Thoth into ``ClaudeAgentOptions.can_use_tool`` and can also
    attach lifecycle hooks for post-success/post-failure telemetry.

    Notes:
        ``claude-agent-sdk`` requires streaming mode for ``can_use_tool``
        callbacks. When calling ``claude_agent_sdk.query(...)``, pass prompt as
        an async iterable (not a plain string) when using this integration.
    """
    from thoth.integrations.claude_agent_sdk import instrument_claude_agent_sdk_options

    _, _, _, _, _, tracer = _build_components(
        agent_id,
        approved_scope,
        tenant_id,
        user_id,
        enforcement,
        api_key,
        api_url,
        session_id,
        session_intent,
        environment,
        enforcement_trace_id,
        event_ingest_token,
    )
    return instrument_claude_agent_sdk_options(
        options,
        tracer,
        emit_tool_lifecycle_hooks=emit_tool_lifecycle_hooks,
    )


def _wrap_agent_tools(agent: Any, tracer: Tracer) -> None:
    """Wrap tools on common agent shapes. Extend for each framework."""
    # LangChain AgentExecutor
    try:
        from langchain.agents import AgentExecutor  # type: ignore[import-not-found]

        if isinstance(agent, AgentExecutor):
            from thoth.integrations.langchain import wrap_langchain_agent

            wrap_langchain_agent(agent, tracer)
            return
    except ImportError:
        pass

    # CrewAI Agent
    try:
        from crewai import Agent as CrewAgent  # type: ignore[import-not-found]

        if isinstance(agent, CrewAgent):
            from thoth.integrations.crewai import wrap_crewai_agent

            wrap_crewai_agent(agent, tracer)
            return
    except ImportError:
        pass

    # Generic: any object with a .tools list
    tools = getattr(agent, "tools", None)
    if not tools:
        return
    for tool in tools:
        tool_name = getattr(tool, "name", str(tool))
        original_run = getattr(tool, "run", None) or (tool if callable(tool) else None)
        if original_run:
            wrapped = tracer.wrap_tool(tool_name, original_run)
            attr = "run" if hasattr(tool, "run") else "__call__"
            setattr(tool, attr, wrapped)
