"""Integration helpers for the official ``claude-agent-sdk`` Python package.

This module instruments ``ClaudeAgentOptions``: a PreToolUse hook checks
every tool call with Thoth before it runs, including tools in
``allowed_tools``, which the SDK never passes to ``can_use_tool``. The
governed ``can_use_tool`` reuses the hook's decision for the same call, and
checks the call itself only when no hook ran for it.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

from thoth.exceptions import ThothPolicyViolation
from thoth.models import EventType
from thoth.tracer import Tracer, _enforcement_outcome_context, _policy_violation_metadata, _resolve_action_attestation_id

_MAX_PENDING_DECISIONS = 1024
_PRE_TOOL_USE_TIMEOUT_SECONDS = 16 * 60


def _hook_matcher(matcher_type: Any, hook: Any, *, timeout: float) -> Any:
    """A HookMatcher for every tool, with a timeout where the SDK supports one."""
    try:
        return matcher_type(matcher=None, hooks=[hook], timeout=timeout)
    except TypeError:
        return matcher_type(matcher=None, hooks=[hook])


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
    generated fallback ID and therefore omit the action ID and decision context.
    Pending decision context is bounded and consumed by the first matching
    post-success or post-failure hook.
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
    pending_decisions: dict[str, dict[str, Any]] = {}

    def provider_action_id(value: Any) -> str | None:
        return (value.strip() or None) if isinstance(value, str) else None

    def decision_map_key(provider_id: Any) -> str | None:
        return (tracer._config.action_attestation_id or "").strip() or provider_action_id(provider_id)

    def remember_decision(provider_id: Any, decision_context: dict[str, Any]) -> None:
        key = decision_map_key(provider_id)
        if key is None:
            return
        # A reused action correlation ID is ambiguous. Empty context fails evidence
        # correlation closed without affecting the authorization result.
        pending_decisions[key] = {} if key in pending_decisions else decision_context
        while len(pending_decisions) > _MAX_PENDING_DECISIONS:
            pending_decisions.pop(next(iter(pending_decisions)))

    def take_decision(provider_id: Any) -> dict[str, Any]:
        key = decision_map_key(provider_id)
        if key is None:
            return {}
        return pending_decisions.pop(key, {})

    def known_action_id(provider_id: Any) -> str | None:
        configured = (tracer._config.action_attestation_id or "").strip()
        if configured:
            return configured
        return (provider_id.strip() or None) if isinstance(provider_id, str) else None

    # Calls the PreToolUse hook has already allowed, by provider tool-use ID,
    # with the exact tool and input checked, so can_use_tool (which the SDK
    # calls only for tools not already allowed) doesn't check the same call
    # twice. If anything changed the call since (another hook rewriting its
    # input), it no longer matches and is checked again.
    hook_decided: dict[str, tuple[str, str]] = {}

    def fingerprint(tool_name: str, tool_input: Any) -> tuple[str, str]:
        try:
            return tool_name, json.dumps(tool_input, sort_keys=True, default=repr)
        except (TypeError, ValueError):
            return tool_name, f"unserializable:{id(tool_input)}"

    async def check(tool_name: str, tool_input: dict[str, Any], provider_id: Any) -> tuple[Any, Any]:
        """Thoth's check of one call: (enforcement, None) or (None, violation)."""
        action_attestation_id = known_action_id(provider_id) or _resolve_action_attestation_id(tracer._config)
        metadata = tracer._base_tool_metadata(tool_name, tool_input, action_attestation_id)
        tracer._emit(
            tool_name,
            EventType.TOOL_CALL_PRE,
            str(tool_input),
            metadata={**metadata, "event_phase": "pre"},
        )
        try:
            enforcement = await tracer._aenforce(
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
            return None, exc
        return enforcement, None

    def checked_input(enforcement: Any, tool_input: dict[str, Any]) -> dict[str, Any]:
        call_args = enforcement.effective_args
        updated_input = call_args[0] if call_args and isinstance(call_args[0], dict) else tool_input
        if isinstance(updated_input, dict) and set(updated_input.keys()) == {"input"} and isinstance(updated_input.get("input"), dict):
            updated_input = updated_input["input"]
        return updated_input

    def deny_output(reason: str) -> dict[str, Any]:
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": reason}}

    async def _pre_tool_use(hook_input: dict[str, Any], tool_use_id: str | None, _context: Any) -> dict[str, Any]:
        """Thoth's check before every tool call, including tools in allowed_tools,
        which the SDK never passes to can_use_tool. An allowed call returns no
        decision, so the SDK's own permission rules still apply."""
        tool_name = hook_input.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            return deny_output("Thoth couldn't identify this tool call.")
        raw_input = hook_input.get("tool_input")
        tool_input = raw_input if isinstance(raw_input, dict) else {}
        provider_id = hook_input.get("tool_use_id") or tool_use_id
        enforcement, violation = await check(tool_name, tool_input, provider_id)
        if violation is not None:
            return deny_output(f"Thoth refused this call: {violation.reason}")
        if checked_input(enforcement, tool_input) != tool_input:
            # A hook can't change input without overriding the SDK's own
            # permission rules, so a call Thoth would change is not run.
            tracer._emit(tool_name, EventType.TOOL_CALL_BLOCK, "arguments changed by policy", metadata={"event_phase": "block"})
            return deny_output("Thoth would change this call's arguments, which this integration doesn't apply, so it was not run.")
        tracer._session.record_tool_call(tool_name)
        if emit_tool_lifecycle_hooks:
            remember_decision(provider_id, _enforcement_outcome_context(enforcement))
        key = provider_action_id(provider_id)
        if key is not None:
            hook_decided[key] = fingerprint(tool_name, tool_input)
            while len(hook_decided) > _MAX_PENDING_DECISIONS:
                hook_decided.pop(next(iter(hook_decided)))
        return {}

    async def governed_can_use_tool(
        tool_name: str,
        tool_input: dict[str, Any],
        context: Any,
    ) -> Any:
        provider_id = getattr(context, "tool_use_id", None)
        key = provider_action_id(provider_id)
        decided = hook_decided.pop(key, None) if key is not None else None
        if decided is not None and decided == fingerprint(tool_name, tool_input):
            # Already checked and allowed by the PreToolUse hook: this call, unchanged.
            if existing_can_use_tool is not None:
                return await existing_can_use_tool(tool_name, tool_input, context)
            return PermissionResultAllow(updated_input=tool_input)

        # No hook allowed this exact call (a CLI without hooks, or input changed
        # since the hook's check): check it here.
        metadata = tracer._base_tool_metadata(tool_name, tool_input, known_action_id(provider_id) or "")
        enforcement, violation = await check(tool_name, tool_input, provider_id)
        if violation is not None:
            return PermissionResultDeny(message=violation.reason, interrupt=False)
        updated_input = checked_input(enforcement, tool_input)

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
                if emit_tool_lifecycle_hooks:
                    remember_decision(provider_id, _enforcement_outcome_context(enforcement))
            return result

        tracer._session.record_tool_call(tool_name)
        if emit_tool_lifecycle_hooks:
            remember_decision(provider_id, _enforcement_outcome_context(enforcement))
        return PermissionResultAllow(updated_input=updated_input)

    options.can_use_tool = governed_can_use_tool

    hooks = dict(options.hooks or {})
    # Longer than the default step-up wait, so a held call can be decided.
    hooks.setdefault("PreToolUse", []).append(_hook_matcher(HookMatcher, _pre_tool_use, timeout=_PRE_TOOL_USE_TIMEOUT_SECONDS))
    options.hooks = hooks

    if emit_tool_lifecycle_hooks:

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
            provider_id = hook_input.get("tool_use_id") or _tool_use_id
            tracer._emit(
                str(hook_input.get("tool_name", "")),
                EventType.TOOL_CALL_POST,
                str(hook_input.get("tool_response", "")),
                metadata={
                    **hook_metadata(hook_input, _tool_use_id),
                    **take_decision(provider_id),
                    "event_phase": "post",
                },
            )
            return {}

        async def _post_tool_use_failure(
            hook_input: dict[str, Any],
            _tool_use_id: str | None,
            _context: Any,
        ) -> dict[str, Any]:
            provider_id = hook_input.get("tool_use_id") or _tool_use_id
            tracer._emit(
                str(hook_input.get("tool_name", "")),
                EventType.TOOL_CALL_BLOCK,
                str(hook_input.get("error", "tool execution failed")),
                metadata={
                    **hook_metadata(hook_input, _tool_use_id),
                    **take_decision(provider_id),
                    "event_phase": "block",
                },
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
