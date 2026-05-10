"""Integration helpers for the official ``claude-agent-sdk`` Python package.

This module instruments ``ClaudeAgentOptions`` by wiring Thoth policy
enforcement into ``can_use_tool`` and optional hook callbacks.
"""

from __future__ import annotations

import importlib
from typing import Any

from thoth.exceptions import ThothPolicyViolation
from thoth.models import EventType
from thoth.tracer import Tracer


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

    async def governed_can_use_tool(
        tool_name: str,
        tool_input: dict[str, Any],
        context: Any,
    ) -> Any:
        tracer._emit(tool_name, EventType.TOOL_CALL_PRE, str(tool_input))
        try:
            call_args, _ = await tracer._aenforce(
                tool_name,
                tool_args=tool_input,
                call_args=(tool_input,),
                call_kwargs={},
            )
        except ThothPolicyViolation as exc:
            tracer._emit(
                tool_name,
                EventType.TOOL_CALL_BLOCK,
                exc.reason,
                violation_id=exc.violation_id,
            )
            return PermissionResultDeny(message=exc.reason, interrupt=False)

        updated_input = call_args[0] if call_args and isinstance(call_args[0], dict) else tool_input
        if isinstance(updated_input, dict) and set(updated_input.keys()) == {"input"} and isinstance(updated_input.get("input"), dict):
            updated_input = updated_input["input"]

        if existing_can_use_tool is not None:
            result = await existing_can_use_tool(tool_name, updated_input, context)
            if isinstance(result, PermissionResultDeny):
                tracer._emit(tool_name, EventType.TOOL_CALL_BLOCK, result.message)
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

        async def _post_tool_use(
            hook_input: dict[str, Any],
            _tool_use_id: str | None,
            _context: Any,
        ) -> dict[str, Any]:
            tracer._emit(
                str(hook_input.get("tool_name", "")),
                EventType.TOOL_CALL_POST,
                str(hook_input.get("tool_response", "")),
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
