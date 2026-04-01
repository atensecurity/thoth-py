# thoth/_context.py
from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thoth.session import SessionContext

_CURRENT_SESSION: contextvars.ContextVar[SessionContext | None] = contextvars.ContextVar("thoth_session", default=None)


def get_current_session() -> SessionContext | None:
    """Return the active Thoth session for the current async context."""
    return _CURRENT_SESSION.get()
