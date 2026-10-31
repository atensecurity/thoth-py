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


# Separators frameworks put between a server or toolset prefix and an MCP
# tool's own name (github_create_issue, github-create_issue, github.create_issue).
_MCP_PREFIX_SEPARATORS = ("_", "-", ".", ":", "/")


def claim_checked_mcp_call(tool_name: str, args: Any) -> bool:
    """True, once, when this MCP call is the one a framework hook already checked.

    Frameworks often expose an MCP tool under a prefixed name, so the checked
    call may be named ``<prefix><separator><tool_name>``. The arguments must be
    identical, and it is claimed once, as with claim_checked_call (owner
    decision, 31 October 2026: a prefixed-name match, not a name-blind one).
    """
    call = _CHECKED_CALL.get()
    if call is None or call.claimed or call.args != _canonical(args if args is not None else {}):
        return False
    if call.name != tool_name and not any(call.name.endswith(separator + tool_name) for separator in _MCP_PREFIX_SEPARATORS):
        return False
    call.claimed = True
    return True


def get_current_session() -> SessionContext | None:
    """Return the active Thoth session for the current async context."""
    return _CURRENT_SESSION.get()
