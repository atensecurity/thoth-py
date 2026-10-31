# thoth/_context.py
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import contextvars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thoth.session import SessionContext

_CURRENT_SESSION: contextvars.ContextVar[SessionContext | None] = contextvars.ContextVar("thoth_session", default=None)

# The tool a Thoth wrapper has checked and is running now. thoth.init()'s
# framework hooks skip a call to the same tool inside it, so a tool wrapped
# explicitly and also reached through a hook is checked once, not twice.
_GOVERNED_TOOL: contextvars.ContextVar[str | None] = contextvars.ContextVar("thoth_governed_tool", default=None)


@contextmanager
def governed_call(tool_name: str) -> Iterator[None]:
    """Mark tool_name as already checked while its callback runs."""
    token = _GOVERNED_TOOL.set(tool_name)
    try:
        yield
    finally:
        _GOVERNED_TOOL.reset(token)


def already_governed(tool_name: str) -> bool:
    return _GOVERNED_TOOL.get() == tool_name


def get_current_session() -> SessionContext | None:
    """Return the active Thoth session for the current async context."""
    return _CURRENT_SESSION.get()
