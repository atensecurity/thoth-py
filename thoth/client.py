from __future__ import annotations

from typing import Any

from thoth.instrumentor import instrument, instrument_anthropic, instrument_openai


class ThothClient:
    """Backward-compatible facade for legacy Thoth SDK usage.

    Prefer the module-level APIs (`thoth.instrument*`) for new integrations.
    """

    def __init__(self, **defaults: Any) -> None:
        self._defaults = defaults

    def instrument(self, agent: Any, **kwargs: Any) -> Any:
        return instrument(agent, **self._merged(kwargs))

    def instrument_anthropic(self, tool_fns: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return instrument_anthropic(tool_fns, **self._merged(kwargs))

    def instrument_openai(self, tool_fns: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return instrument_openai(tool_fns, **self._merged(kwargs))

    # Legacy aliases kept for backwards compatibility.
    def wrap(self, agent: Any, **kwargs: Any) -> Any:
        return self.instrument(agent, **kwargs)

    def wrap_anthropic_tools(self, tool_fns: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self.instrument_anthropic(tool_fns, **kwargs)

    def wrap_openai_tools(self, tool_fns: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self.instrument_openai(tool_fns, **kwargs)

    def _merged(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        merged = dict(self._defaults)
        merged.update(kwargs)
        return merged
