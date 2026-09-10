"""Integration helpers for the official ``claude-agent-sdk`` Python package.

This module instruments ``ClaudeAgentOptions`` by wiring Thoth policy
enforcement into ``can_use_tool`` and optional hook callbacks.
"""

from __future__ import annotations

import importlib
from typing import Any

from thoth.exceptions import ThothPolicyViolation
from thoth.models import EventType
from thoth.tracer import Tracer, _policy_violation_metadata, _resolve_action_attestation_id


def _load_claude_agent_sdk_types() -> dict[str, type[Any]]:
    """Load runtime types from claude-agent-sdk lazily.

    The Thoth SDK does not require claude-agent-sdk unless this integration
    function is used.
    """
    try:
        types_mod = importlib.import_module("claude_agent_sdk.types")
    except ImportError as exc:
        raise ImportError('claude-agent-sdk is required for this integration. Install it with: pip install "claude-agent-sdk"') from exc

    return {
        "ClaudeAgentOptions": types_mod.ClaudeAgentOptions,
        "PermissionResultAllow": types_mod.PermissionResultAllow,
        "PermissionResultDeny": types_mod.PermissionResultDeny,
        "HookMatcher": types_mod.HookMatcher,
    }


def instrument_claude_agent_sdk_options(
    options: Any,
    tracer: Tracer,
    *,
    emit_tool_lifecycle_hooks: bool = True,
) -> Any:
    """Attach Thoth governance callbacks to ``ClaudeAgentOptions``.

    Args:
        options: An instance of ``claude_agent_sdk.types.ClaudeAgentOptions``.
        tracer: Configured Thoth tracer for enforce + emit behavior.
        emit_tool_lifecycle_hooks: When true, appends SDK hook callbacks that
            emit Thoth post-success and post-failure events.

    Returns:
        The same options object, mutated in-place with governance callbacks.

    Provider tool-use IDs supply correlation, not authenticated attestation proof.
    Hooks without a configured or provider ID cannot correlate a callback's
    generated fallback ID and therefore omit the action ID.
    """
    sdk_types = _load_claude_agent_sdk_types()
    ClaudeAgentOptions = sdk_types["ClaudeAgentOptions"]
    PermissionResultAllow = sdk_types["PermissionResultAllow"]
    PermissionResultDeny = sdk_types["PermissionResultDeny"]
    HookMatcher = sdk_types["HookMatcher"]

    if options is None:
        options = ClaudeAgentOptions()
    if not isinstance(options, ClaudeAgentOptions):
        raise TypeError("options must be an instance of claude_agent_sdk.types.ClaudeAgentOptions")

    model_name = str(getattr(options, "model", "") or "").strip() or "unspecified"
    tracer._emit(
        "claude_agent_sdk",
        EventType.LLM_INVOCATION,
        f"claude_agent_sdk_session_start model={model_name}",
    )

    existing_can_use_tool = options.can_use_tool

    def known_action_id(provider_id: Any) -> str | None:
        configured = (tracer._config.action_attestation_id or "").strip()
        if configured:
            return configured
        return (provider_id.strip() or None) if isinstance(provider_id, str) else None

    async def governed_can_use_tool(
        tool_name: str,
        tool_input: dict[str, Any],
        context: Any,
    ) -> Any:
        action_attestation_id = known_action_id(getattr(context, "tool_use_id", None)) or _resolve_action_attestation_id(tracer._config)
        metadata = tracer._base_tool_metadata(tool_name, tool_input, action_attestation_id)
        tracer._emit(
            tool_name,
            EventType.TOOL_CALL_PRE,
            str(tool_input),
            metadata={**metadata, "event_phase": "pre"},
        )
        try:
            call_args, _ = await tracer._aenforce(
                tool_name,
                tool_args=tool_input,
                call_args=(tool_input,),
                call_kwargs={},
                action_attestation_id=action_attestation_id,
            )
        except ThothPolicyViolation as exc:
            tracer._emit(
                tool_name,
                EventType.TOOL_CALL_BLOCK,
                exc.reason,
                violation_id=exc.violation_id,
                metadata={
                    **metadata,
                    **_policy_violation_metadata(exc),
                    "action_attestation_id": action_attestation_id,
                    "event_phase": "block",
                },
            )
            return PermissionResultDeny(message=exc.reason, interrupt=False)

        updated_input = call_args[0] if call_args and isinstance(call_args[0], dict) else tool_input
        if isinstance(updated_input, dict) and set(updated_input.keys()) == {"input"} and isinstance(updated_input.get("input"), dict):
            updated_input = updated_input["input"]

        if existing_can_use_tool is not None:
            result = await existing_can_use_tool(tool_name, updated_input, context)
            if isinstance(result, PermissionResultDeny):
                tracer._emit(
                    tool_name,
                    EventType.TOOL_CALL_BLOCK,
                    result.message,
                    metadata={**metadata, "event_phase": "block"},
                )
                return result
            if isinstance(result, PermissionResultAllow):
                if result.updated_input is None:
                    result.updated_input = updated_input
                tracer._session.record_tool_call(tool_name)
            return result

        tracer._session.record_tool_call(tool_name)
        return PermissionResultAllow(updated_input=updated_input)

    options.can_use_tool = governed_can_use_tool

    if emit_tool_lifecycle_hooks:
        hooks = dict(options.hooks or {})

        def hook_metadata(hook_input: dict[str, Any], tool_use_id: str | None) -> dict[str, Any]:
            action_id = known_action_id(hook_input.get("tool_use_id")) or known_action_id(tool_use_id)
            metadata = tracer._base_tool_metadata(
                str(hook_input.get("tool_name", "")),
                hook_input.get("tool_input"),
                action_id or "",
            )
            if action_id is None:
                # Never infer a callback match by tool name or generate a new ID here.
                metadata.pop("action_attestation_id")
            return metadata

        async def _post_tool_use(
            hook_input: dict[str, Any],
            _tool_use_id: str | None,
            _context: Any,
        ) -> dict[str, Any]:
            tracer._emit(
                str(hook_input.get("tool_name", "")),
                EventType.TOOL_CALL_POST,
                str(hook_input.get("tool_response", "")),
                metadata={**hook_metadata(hook_input, _tool_use_id), "event_phase": "post"},
            )
            return {}

        async def _post_tool_use_failure(
            hook_input: dict[str, Any],
            _tool_use_id: str | None,
            _context: Any,
        ) -> dict[str, Any]:
            tracer._emit(
                str(hook_input.get("tool_name", "")),
                EventType.TOOL_CALL_BLOCK,
                str(hook_input.get("error", "tool execution failed")),
                metadata={**hook_metadata(hook_input, _tool_use_id), "event_phase": "block"},
            )
            return {}

        hooks.setdefault("PostToolUse", []).append(
            HookMatcher(matcher=None, hooks=[_post_tool_use]),
        )
        hooks.setdefault("PostToolUseFailure", []).append(
            HookMatcher(matcher=None, hooks=[_post_tool_use_failure]),
        )
        options.hooks = hooks

    return options
