from __future__ import annotations

from typing import Any, cast

from thoth.instrumentor import (
    instrument,
    instrument_anthropic,
    instrument_claude_agent_sdk,
    instrument_openai,
    instrument_toolchain,
    toolchain_function_map,
)


class ThothClient:
    """Backward-compatible facade for legacy Thoth SDK usage.

    Prefer the module-level APIs (`thoth.instrument*`) for new integrations.
    """

    def __init__(self, **defaults: Any) -> None:
        self._defaults = defaults

    def instrument(self, agent: Any, **kwargs: Any) -> Any:
        return instrument(agent, **self._merged(kwargs))

    def instrument_anthropic(self, tool_fns: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return cast(dict[str, Any], instrument_anthropic(tool_fns, **self._merged(kwargs)))

    def instrument_openai(self, tool_fns: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return cast(dict[str, Any], instrument_openai(tool_fns, **self._merged(kwargs)))

    def instrument_claude_agent_sdk(self, options: Any | None = None, **kwargs: Any) -> Any:
        return instrument_claude_agent_sdk(options, **self._merged(kwargs))

    def instrument_toolchain(self, toolchain: Any, **kwargs: Any) -> Any:
        return instrument_toolchain(toolchain, **self._merged(kwargs))

    def toolchain_function_map(self, toolchain: Any, **kwargs: Any) -> dict[str, Any]:
        merged = self._merged(kwargs)
        supported = {key: merged[key] for key in ("include_private", "max_depth") if key in merged}
        return cast(dict[str, Any], toolchain_function_map(toolchain, **supported))

    # Legacy aliases kept for backwards compatibility.
    def wrap(self, agent: Any, **kwargs: Any) -> Any:
        return self.instrument(agent, **kwargs)

    def wrap_anthropic_tools(self, tool_fns: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self.instrument_anthropic(tool_fns, **kwargs)

    def wrap_openai_tools(self, tool_fns: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self.instrument_openai(tool_fns, **kwargs)

    def wrap_claude_agent_sdk(self, options: Any | None = None, **kwargs: Any) -> Any:
        return self.instrument_claude_agent_sdk(options, **kwargs)

    def wrap_toolchain(self, toolchain: Any, **kwargs: Any) -> Any:
        return self.instrument_toolchain(toolchain, **kwargs)

    def build_function_map(self, toolchain: Any, **kwargs: Any) -> dict[str, Any]:
        return self.toolchain_function_map(toolchain, **kwargs)

    def _merged(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        merged = dict(self._defaults)
        merged.update(kwargs)
        return merged
