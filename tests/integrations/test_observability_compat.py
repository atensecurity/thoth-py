# tests/integrations/test_observability_compat.py
from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import pytest

import thoth
from thoth import ThothPolicyViolation
from thoth.models import DecisionType, EnforcementDecision


class FakeTool:
    name = "search"

    def __init__(self, run: Callable[[str], str]) -> None:
        self.run = run


class FakeExecutor:
    def __init__(self, tool: FakeTool) -> None:
        self.tools = [tool]


def _datadog_like_wrapper(events: list[str], fn: Callable[[str], str]) -> Callable[[str], str]:
    def wrapped(query: str) -> str:
        events.append("datadog:start")
        try:
            return fn(query)
        finally:
            events.append("datadog:end")

    return wrapped


def _otel_like_wrapper(events: list[str], fn: Callable[[str], str]) -> Callable[[str], str]:
    class _Span:
        def __enter__(self) -> None:
            events.append("otel:start")
            return None

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            _ = (exc_type, exc, tb)
            events.append("otel:end")
            return None

    def wrapped(query: str) -> str:
        with _Span():
            return fn(query)

    return wrapped


def _sentry_like_wrapper(events: list[str], fn: Callable[[str], str]) -> Callable[[str], str]:
    class _Span:
        def __enter__(self) -> None:
            events.append("sentry:start")
            return None

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            _ = (exc_type, exc, tb)
            events.append("sentry:end")
            return None

    def wrapped(query: str) -> str:
        with _Span():
            return fn(query)

    return wrapped


@pytest.mark.parametrize(
    ("name", "wrap"),
    [
        ("datadog", _datadog_like_wrapper),
        ("opentelemetry", _otel_like_wrapper),
        ("sentry", _sentry_like_wrapper),
    ],
)
def test_observability_wrapper_and_thoth_can_coexist_allow(name, wrap):
    events: list[str] = []

    def raw_run(query: str) -> str:
        events.append(f"{name}:run")
        return f"results:{query}"

    tool = FakeTool(wrap(events, raw_run))
    executor = FakeExecutor(tool)

    with patch("thoth.instrumentor.EnforcerClient") as MockEnforcer, patch("thoth.instrumentor.HttpEmitter"):
        MockEnforcer.return_value.check.return_value = EnforcementDecision(decision=DecisionType.ALLOW)
        governed = thoth.instrument(
            executor,
            agent_id=f"{name}-agent",
            approved_scope=["search"],
            tenant_id="trantor",
            api_url="https://enforcer.example",
        )
        result = governed.tools[0].run("incident 42")

        assert result == "results:incident 42"
        assert any(item.endswith(":run") for item in events)
        assert any(item.endswith(":start") for item in events)
        assert any(item.endswith(":end") for item in events)
        MockEnforcer.return_value.check.assert_called_once()

        session = thoth.get_current_session()
        assert session is not None
        assert "search" in session.tool_calls


@pytest.mark.parametrize(
    ("name", "wrap"),
    [
        ("datadog", _datadog_like_wrapper),
        ("opentelemetry", _otel_like_wrapper),
        ("sentry", _sentry_like_wrapper),
    ],
)
def test_observability_wrapper_and_thoth_can_coexist_block(name, wrap):
    events: list[str] = []

    def raw_run(query: str) -> str:
        events.append(f"{name}:run")
        return f"results:{query}"

    tool = FakeTool(wrap(events, raw_run))
    executor = FakeExecutor(tool)

    with patch("thoth.instrumentor.EnforcerClient") as MockEnforcer, patch("thoth.instrumentor.HttpEmitter"):
        MockEnforcer.return_value.check.return_value = EnforcementDecision(
            decision=DecisionType.BLOCK,
            reason="blocked by policy",
            violation_id=f"vio-{name}-001",
        )
        governed = thoth.instrument(
            executor,
            agent_id=f"{name}-agent",
            approved_scope=["search"],
            tenant_id="trantor",
            api_url="https://enforcer.example",
        )

        with pytest.raises(ThothPolicyViolation):
            governed.tools[0].run("incident 42")

        # Thoth blocks pre-execution, so wrapped observability spans should not start.
        assert events == []
