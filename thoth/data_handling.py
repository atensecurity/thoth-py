# thoth/data_handling.py
"""Drop, redact or hash values at the source, before a request leaves the process.

Opt-in per client. With nothing configured this module changes nothing.

Modes
-----
Tool arguments (``THOTH_DATA_HANDLING_TOOL_ARGUMENTS`` or the
``data_handling_tool_arguments`` option):

``hash``
    Keys and structure are kept. Every leaf value (string, number, boolean) is
    replaced by ``"hmac-sha256:" + hex(HMAC-SHA256(salt, canonical_json(leaf)))``.
    ``null`` stays ``null``.
``redact``
    Keys and structure are kept. Every leaf, ``null`` included, is replaced by
    the fixed string ``"<redacted>"``.
``drop``
    No arguments are sent at all: the ``tool_args`` field is omitted.

Prompts (``THOTH_DATA_HANDLING_PROMPTS`` or ``data_handling_prompts``):

``hash``
    A prompt-bearing field is replaced by one hash of its whole value, in the
    same ``hmac-sha256:`` format.
``drop``
    The field is omitted.

Canonical JSON, for hashing, is ``json.dumps(value, sort_keys=True,
separators=(",", ":"), ensure_ascii=False)`` encoded as UTF-8. The salt is
the UTF-8 bytes of the salt string. The golden fixture
``tests/fixtures/data_handling_golden.json`` pins these formats so every Thoth
SDK produces the same bytes.

Salt
----
Hashing uses a salt held by the client. Thoth never sees it. It comes from the
``data_handling_salt`` option, else ``THOTH_DATA_HANDLING_SALT``, else a random
per-process salt. Two values hash equally only inside one salt, so with the
random default they match only within one process. Set the same salt on every
process that should be able to match values (for example a fleet of workers).
Anyone holding the salt can confirm a guess for a low-entropy value, so keep it
as secret as the values themselves.

Bounds
------
Nested arguments are walked to depth 16 and at most 5,000 values. Anything
deeper is replaced by ``"<truncated>"``; once the budget is spent a dict gains
the key ``"_truncated"`` with that value and a list ends with it. The output
never contains an original value that was not transformed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
import secrets
import threading
from typing import Any

TOOL_ARGUMENT_MODES = ("hash", "redact", "drop")
PROMPT_MODES = ("hash", "drop")
VERSION = 1
REDACTED = "<redacted>"
TRUNCATED = "<truncated>"
HASH_PREFIX = "hmac-sha256:"
MAX_DEPTH = 16
MAX_NODES = 5000
# Strings longer than this inside a context dict count as free text.
LONG_TEXT_CHARS = 512

ENV_TOOL_ARGUMENTS = "THOTH_DATA_HANDLING_TOOL_ARGUMENTS"
ENV_PROMPTS = "THOTH_DATA_HANDLING_PROMPTS"
ENV_SALT = "THOTH_DATA_HANDLING_SALT"
ENV_LOCAL_SCAN = "THOTH_DATA_HANDLING_LOCAL_SCAN"

# Context dicts (task_context, auth_context, delegation_context, identity_binding,
# request metadata) are mostly identifiers the policy engine reads. Values under
# these key names, at any depth, are model or user text: the prompts mode applies.
# Spelling-insensitive: ``User_Message`` and ``user-message`` match ``usermessage``.
PROMPT_KEYS = frozenset(
    {
        "prompt",
        "prompts",
        "systemprompt",
        "userprompt",
        "usermessage",
        "message",
        "messages",
        "input",
        "inputtext",
        "query",
        "question",
        "text",
        "content",
        "instruction",
        "instructions",
        "task",
        "goal",
        "description",
        "summary",
        "note",
        "notes",
        "comment",
        "conversation",
        "conversationcontext",
        "conversationinputtext",
        "conversationoutputtext",
        "history",
        "transcript",
        "completion",
        "response",
        "output",
        "toolresult",
        "tooloutput",
    }
)

_process_salt: str | None = None
_process_salt_lock = threading.Lock()


def process_salt() -> str:
    """The random salt shared by every client in this process, made once."""
    global _process_salt
    with _process_salt_lock:
        if _process_salt is None:
            _process_salt = secrets.token_hex(32)
        return _process_salt


def _normalize_mode(value: Any, allowed: tuple[str, ...], name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be one of {', '.join(allowed)}; got {value!r}")
    mode = value.strip().lower()
    if not mode:
        return None
    if mode not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}; got {value!r}. Thoth will not send the values unprotected, so fix the setting.")
    return mode


def resolve_tool_arguments_mode(value: Any) -> str | None:
    """The configured mode: option, else env, validated. None when unset. Raises on an invalid value."""
    if value is None or (isinstance(value, str) and not value.strip()):
        value = os.getenv(ENV_TOOL_ARGUMENTS)
        name = ENV_TOOL_ARGUMENTS
    else:
        name = "data_handling_tool_arguments"
    return _normalize_mode(value, TOOL_ARGUMENT_MODES, name)


def resolve_prompts_mode(value: Any) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        value = os.getenv(ENV_PROMPTS)
        name = ENV_PROMPTS
    else:
        name = "data_handling_prompts"
    return _normalize_mode(value, PROMPT_MODES, name)


def resolve_salt(value: Any) -> str | None:
    """The configured salt: option, else env. None means use the per-process salt."""
    if value is None or (isinstance(value, str) and not value.strip()):
        value = os.getenv(ENV_SALT)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("data_handling_salt must be a string")
    return value if value.strip() else None


def resolve_local_scan(value: Any) -> bool:
    """Whether to scan withheld arguments locally: option, else env, default True. Raises on an unclear value."""
    if value is None or (isinstance(value, str) and not value.strip()):
        value = os.getenv(ENV_LOCAL_SCAN)
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        word = value.strip().lower()
        if word in ("on", "true", "1", "yes"):
            return True
        if word in ("off", "false", "0", "no"):
            return False
    raise ValueError("data_handling_local_scan must be on or off")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def hash_value(salt: str, value: Any) -> str:
    digest = hmac.new(salt.encode("utf-8"), canonical_json(value).encode("utf-8"), hashlib.sha256).hexdigest()
    return HASH_PREFIX + digest


def _key_name(key: Any) -> str:
    return key if isinstance(key, str) else str(key)


def _normalized_key(key: str) -> str:
    return key.lower().replace("_", "").replace("-", "")


class _Budget:
    __slots__ = ("remaining",)

    def __init__(self, nodes: int) -> None:
        self.remaining = nodes


def _map_leaves(value: Any, leaf: Any, budget: _Budget, depth: int) -> Any:
    if budget.remaining <= 0 or depth > MAX_DEPTH:
        return TRUNCATED
    budget.remaining -= 1
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if budget.remaining <= 0:
                out["_truncated"] = TRUNCATED
                break
            out[_key_name(key)] = _map_leaves(item, leaf, budget, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        items: list[Any] = []
        for item in value:
            if budget.remaining <= 0:
                items.append(TRUNCATED)
                break
            items.append(_map_leaves(item, leaf, budget, depth + 1))
        return items
    return leaf(value)


def redact_arguments(value: Any) -> Any:
    """Structure and keys kept, every leaf replaced by ``<redacted>``."""
    return _map_leaves(value, lambda _leaf: REDACTED, _Budget(MAX_NODES), 0)


def hash_arguments(value: Any, salt: str) -> Any:
    """Structure and keys kept, every leaf but null replaced by its keyed hash."""
    return _map_leaves(value, lambda leaf: None if leaf is None else hash_value(salt, leaf), _Budget(MAX_NODES), 0)


def transform_arguments(value: Any, mode: str, salt: str) -> tuple[bool, Any]:
    """(keep, new value) for tool arguments under *mode*."""
    if mode == "drop":
        return False, None
    if mode == "redact":
        return True, redact_arguments(value)
    if mode == "hash":
        return True, hash_arguments(value, salt)
    raise ValueError(f"unknown tool arguments mode {mode!r}")


def transform_prompt(value: Any, mode: str, salt: str) -> tuple[bool, Any]:
    """(keep, new value) for a prompt-bearing value under *mode*."""
    if mode == "drop":
        return False, None
    if mode == "hash":
        return True, None if value is None else hash_value(salt, value)
    raise ValueError(f"unknown prompts mode {mode!r}")


def _walk_context(node: Any, mode: str, salt: str, depth: int = 0) -> Any:
    """Copy of a context value with prompt-bearing and long free-text values handled."""
    if depth > MAX_DEPTH:
        return TRUNCATED
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, item in node.items():
            name = _key_name(key)
            if _normalized_key(name) in PROMPT_KEYS:
                keep, new = transform_prompt(item, mode, salt)
                if keep:
                    out[name] = new
                continue
            out[name] = _walk_context(item, mode, salt, depth + 1)
        return out
    if isinstance(node, (list, tuple)):
        return [_walk_context(item, mode, salt, depth + 1) for item in node]
    if isinstance(node, str) and len(node) > LONG_TEXT_CHARS:
        # Free text under an unremarkable key. Cannot be omitted from a list or
        # a value position without changing structure, so it is hashed or
        # replaced by the redaction marker.
        return hash_value(salt, node) if mode == "hash" else REDACTED
    return node


def transform_context(value: Any, mode: str, salt: str) -> Any:
    return _walk_context(value, mode, salt)


@dataclass(frozen=True)
class DataHandling:
    """What this client applies before sending. Immutable; safe to share across threads."""

    tool_arguments: str | None
    prompts: str | None
    salt: str

    @classmethod
    def resolve(cls, tool_arguments: Any = None, prompts: Any = None, salt: Any = None) -> DataHandling | None:
        """Options, else environment. None when nothing is configured. Raises ValueError on an invalid value."""
        args_mode = resolve_tool_arguments_mode(tool_arguments)
        prompt_mode = resolve_prompts_mode(prompts)
        if args_mode is None and prompt_mode is None:
            return None
        return cls(args_mode, prompt_mode, resolve_salt(salt) or process_salt())

    def declaration(self) -> dict[str, Any]:
        """The ``data_handling`` field of the request, naming only what was applied."""
        out: dict[str, Any] = {}
        if self.tool_arguments is not None:
            out["tool_arguments"] = self.tool_arguments
        if self.prompts is not None:
            out["prompts"] = self.prompts
        out["applied_at"] = "client"
        out["version"] = VERSION
        return out

    def apply(
        self,
        payload: dict[str, Any],
        *,
        context_fields: tuple[str, ...] = ("task_context", "auth_context", "delegation_context", "identity_binding", "metadata"),
    ) -> dict[str, Any]:
        """Transform the value-carrying fields of an enforce or explain payload, in place, and declare it."""
        if self.tool_arguments is not None and "tool_args" in payload:
            keep, new = transform_arguments(payload["tool_args"], self.tool_arguments, self.salt)
            if keep:
                payload["tool_args"] = new
            else:
                del payload["tool_args"]
        if self.prompts is not None:
            for name in context_fields:
                if payload.get(name):
                    payload[name] = transform_context(payload[name], self.prompts, self.salt)
        payload["data_handling"] = self.declaration()
        return payload
