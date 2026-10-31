# thoth/_context.py
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from thoth.session import SessionContext

_CURRENT_SESSION: contextvars.ContextVar[SessionContext | None] = contextvars.ContextVar("thoth_session", default=None)


@dataclass
class _CheckedCall:
    name: str
    args: str
    claimed: bool = False


# The call a Thoth wrapper has checked and is running now. thoth.init()'s
# framework hooks may skip checking it again, but only once and only that
# exact call: the same tool with the same arguments. Any other call inside the
# wrapper, a repeat of it, or work started from it, is checked as usual.
_CHECKED_CALL: contextvars.ContextVar[_CheckedCall | None] = contextvars.ContextVar("thoth_checked_call", default=None)


def _canonical(args: Any) -> str:
    try:
        return json.dumps(args, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return f"unserializable:{id(args)}"


@contextmanager
def governed_call(tool_name: str, args: Any = None) -> Iterator[None]:
    """Record the call Thoth checked (tool and arguments) while its callback runs."""
    token = _CHECKED_CALL.set(_CheckedCall(tool_name, _canonical(args if args is not None else {})))
    try:
        yield
    finally:
        _CHECKED_CALL.reset(token)


def claim_checked_call(tool_name: str, args: Any) -> bool:
    """True, once, when this is the exact call a wrapper already checked and is running."""
    call = _CHECKED_CALL.get()
    if call is None or call.claimed or call.name != tool_name or call.args != _canonical(args if args is not None else {}):
        return False
    call.claimed = True
    return True


def get_current_session() -> SessionContext | None:
    """Return the active Thoth session for the current async context."""
    return _CURRENT_SESSION.get()
