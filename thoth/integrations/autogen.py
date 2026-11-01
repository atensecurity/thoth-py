"""
AutoGen function-map integration for Thoth.

For AG2 1.x (``pip install atensec-thoth[ag2]``), call ``thoth.init()``
instead: it governs every function tool an AG2 ``Agent`` runs, with no
function map to wrap.

This module is for the AutoGen 0.2 API, which AG2 0.x (``ag2<1``, imported as
``autogen``) kept: it wraps the ``function_map`` a ``UserProxyAgent`` runs, so
every function execution is policy-checked before it runs. Microsoft's
AutoGen 0.4 and later (``autogen-agentchat``) has no function map; for
Microsoft's agents, use Microsoft Agent Framework, which ``thoth.init()``
governs.

Usage::

    import os
    import autogen
    from thoth.integrations.autogen import wrap_autogen_tools

    def search_docs(query: str) -> str:
        ...

    def send_email(to: str, subject: str, body: str) -> str:
        ...

    governed = wrap_autogen_tools(
        {"search_docs": search_docs, "send_email": send_email},
        agent_id="autogen-assistant",
        approved_scope=["search_docs"],          # send_email will be blocked
        tenant_id="acme-corp",
        user_id="alice@acme.com",
        api_key=os.environ["THOTH_API_KEY"],
        enforcement="block",
    )

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        function_map=governed,   # governed functions injected here
    )
"""

from __future__ import annotations

from typing import Any, Callable

from thoth.tracer import Tracer


def wrap_autogen_tools(
    tool_fns: dict[str, Callable[..., Any]],
    tracer: Tracer | None = None,
    **config: Any,
) -> dict[str, Callable[..., Any]]:
    """Wrap an AutoGen function map with Thoth governance.

    Args:
        tool_fns: AutoGen ``function_map`` dict — tool name → callable.
        tracer: A configured :class:`thoth.tracer.Tracer`, or None to build
            one from ``config``.
        **config: The Thoth settings, as for :func:`thoth.instrument_anthropic`
            (``agent_id``, ``approved_scope``, ``tenant_id``, ``api_key`` and
            the rest), when no tracer is given.

    Returns:
        A new dict with the same keys but governance-wrapped callables.
    """
    if tracer is not None:
        if config:
            raise TypeError("wrap_autogen_tools() takes a tracer or the Thoth settings, not both")
        return {name: tracer.wrap_tool(name, fn) for name, fn in tool_fns.items()}
    # A function map is a dict of plain callables, as the Anthropic loop's is.
    from thoth.instrumentor import instrument_anthropic  # noqa: PLC0415

    return instrument_anthropic(tool_fns, **config)
