"""Report the model usage behind a governed call, so a daily budget has something to count.

Call :func:`report_usage` after each model call. The SDK adds what you reported to the
next governed call (the next ``/v1/enforce`` request) as ``prompt_tokens``,
``completion_tokens`` and, only if you passed it, ``estimated_cost_usd``, then forgets it:
usage is sent once, never twice. Several model calls before one governed call add up.

Thoth counts tokens first. Pass ``cost_usd`` only if you know the dollar figure yourself;
Thoth never works dollars out from tokens.

A value that is not a finite, non-negative number is ignored (with a warning), never
raised: reporting must not break the agent that is doing real work. A single report is
capped at 10,000,000 tokens and $1,000 (the server clamps the same way). The running total is
exact: if more than one request can carry (10,000,000 tokens or $1,000), the rest goes with the
following governed requests, nothing is dropped.

Usage reported after the agent's last governed call has no request to ride on and is not
counted; Thoth has no separate usage channel, and ``shutdown()`` does not send a decision
request just to carry it. Report usage before the last tool call you want counted, or make the
last model call before a governed one.

The pending usage is process-wide, not per task or thread: a daily budget is per agent, so
it does not matter which governed call carries the usage, and a model call made in one thread
or task still counts when the governed call runs in another. If one process runs several
agent identities, each identity's budget sees usage reported by any of them; run one client
per process for exact attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import threading
from typing import Any

logger = logging.getLogger(__name__)

MAX_TOKENS = 10_000_000
MAX_COST_USD = 1000.0


@dataclass
class _Pending:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None


class _Accumulator:
    """Exact running totals. Each report is clamped on the way in (see report_usage); the running
    total is never clamped, so many reports between governed calls are never silently dropped.
    take() hands out at most one request's maximum and leaves the rest for the next governed call."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending = _Pending()

    def add(self, prompt: int | None, completion: int | None, cost: float | None) -> None:
        with self._lock:
            if prompt is not None:
                self._pending.prompt_tokens = (self._pending.prompt_tokens or 0) + prompt
            if completion is not None:
                self._pending.completion_tokens = (self._pending.completion_tokens or 0) + completion
            if cost is not None:
                self._pending.cost_usd = (self._pending.cost_usd or 0.0) + cost

    def take(self) -> _Pending | None:
        """Up to MAX_TOKENS tokens (prompt and completion together) and MAX_COST_USD dollars, the server's
        per-request maximum; anything above stays pending for the next governed request."""
        with self._lock:
            pending = self._pending
            if pending.prompt_tokens is None and pending.completion_tokens is None and pending.cost_usd is None:
                return None
            out = _Pending()
            room = MAX_TOKENS
            if pending.prompt_tokens is not None:
                out.prompt_tokens = min(pending.prompt_tokens, room)
                room -= out.prompt_tokens
                pending.prompt_tokens -= out.prompt_tokens
            if pending.completion_tokens is not None:
                out.completion_tokens = min(pending.completion_tokens, room)
                pending.completion_tokens -= out.completion_tokens
            if pending.cost_usd is not None:
                out.cost_usd = min(pending.cost_usd, MAX_COST_USD)
                pending.cost_usd -= out.cost_usd
            # What is left stays; an emptied field goes back to "nothing pending".
            if not pending.prompt_tokens:
                pending.prompt_tokens = None
            if not pending.completion_tokens:
                pending.completion_tokens = None
            if not pending.cost_usd or pending.cost_usd < 1e-9:
                pending.cost_usd = None
            return out


_ACCUMULATOR = _Accumulator()


def _tokens(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        logger.warning("thoth: ignored %s: expected a non-negative number", name)
        return None
    return min(int(value), MAX_TOKENS)


def _cost(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        logger.warning("thoth: ignored cost_usd: expected a non-negative number")
        return None
    return min(float(value), MAX_COST_USD)


def report_usage(
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    *,
    cost_usd: float | None = None,
) -> None:
    """Record the usage of a model call; it rides on the next governed call."""
    prompt = _tokens(prompt_tokens, "prompt_tokens")
    completion = _tokens(completion_tokens, "completion_tokens")
    cost = _cost(cost_usd)
    if prompt is None and completion is None and cost is None:
        return
    _ACCUMULATOR.add(prompt, completion, cost)


def _field(source: Any, *names: str) -> Any:
    for name in names:
        value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
        if value is not None:
            return value
    return None


def report_usage_from(response: Any, *, cost_usd: float | None = None) -> None:
    """Report the usage block of an OpenAI-style or Anthropic-style response, if it has one.

    Reads ``usage.prompt_tokens``/``usage.completion_tokens`` (OpenAI, Azure OpenAI) or
    ``usage.input_tokens``/``usage.output_tokens`` (Anthropic). A response with no usage
    block reports nothing, and never raises.
    """
    try:
        usage = _field(response, "usage")
        if usage is None:
            return
        report_usage(
            _field(usage, "prompt_tokens", "input_tokens"),
            _field(usage, "completion_tokens", "output_tokens"),
            cost_usd=cost_usd,
        )
    except Exception:
        logger.debug("thoth: could not read usage from the response", exc_info=True)


def take_pending_usage() -> dict[str, Any] | None:
    """The usage fields to add to the next governed request, or None. Clears what it returns."""
    pending = _ACCUMULATOR.take()
    if pending is None:
        return None
    fields: dict[str, Any] = {}
    if pending.prompt_tokens is not None:
        fields["prompt_tokens"] = pending.prompt_tokens
    if pending.completion_tokens is not None:
        fields["completion_tokens"] = pending.completion_tokens
    if pending.cost_usd is not None:
        fields["estimated_cost_usd"] = pending.cost_usd
    return fields


def restore_pending_usage(fields: dict[str, Any] | None) -> None:
    """Put usage back after a governed call that did not reach the server, so it is sent once."""
    if not fields:
        return
    _ACCUMULATOR.add(fields.get("prompt_tokens"), fields.get("completion_tokens"), fields.get("estimated_cost_usd"))
