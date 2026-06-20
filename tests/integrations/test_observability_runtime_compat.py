# tests/integrations/test_observability_runtime_compat.py
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


def _ddtrace_wrapped(events: list[str]) -> Callable[[str], str]:
    ddtrace = pytest.importorskip("ddtrace")

    @ddtrace.tracer.wrap(name="search_tool")
    def run(query: str) -> str:
        events.append("ddtrace:run")
        return f"results:{query}"

    return run


def _otel_wrapped(events: list[str]) -> Callable[[str], str]:
    trace = pytest.importorskip("opentelemetry.trace")

    tracer = trace.get_tracer("thoth-test")

    def run(query: str) -> str:
        with tracer.start_as_current_span("search_tool"):
            events.append("otel:run")
            return f"results:{query}"

    return run


def _sentry_wrapped(events: list[str]) -> Callable[[str], str]:
    sentry_sdk = pytest.importorskip("sentry_sdk")

    sentry_sdk.init(
        dsn="https://public@example.com/1",
        traces_sample_rate=1.0,
        transport=lambda event: None,
    )

    def run(query: str) -> str:
        with sentry_sdk.start_span(op="tool", description="search_tool"):
            events.append("sentry:run")
            return f"results:{query}"

    return run


RUNTIME_WRAPPERS: dict[str, Callable[[list[str]], Callable[[str], str]]] = {
    "ddtrace": _ddtrace_wrapped,
    "opentelemetry": _otel_wrapped,
    "sentry": _sentry_wrapped,
}


@pytest.mark.parametrize("stack", ["ddtrace", "opentelemetry", "sentry"])
def test_runtime_observability_and_thoth_coexist_allow(stack: str) -> None:
    events: list[str] = []
    run = RUNTIME_WRAPPERS[stack](events)
    tool = FakeTool(run)
    executor = FakeExecutor(tool)

    with patch("thoth.instrumentor.EnforcerClient") as MockEnforcer, patch("thoth.instrumentor.HttpEmitter"):
        MockEnforcer.return_value.check.return_value = EnforcementDecision(decision=DecisionType.ALLOW)
        governed = thoth.instrument(
            executor,
            agent_id=f"{stack}-agent",
            approved_scope=["search"],
            tenant_id="trantor",
            api_url="https://enforcer.example",
        )

        result = governed.tools[0].run("incident 42")
        assert result == "results:incident 42"
        expected_event = "otel:run" if stack == "opentelemetry" else f"{stack}:run"
        assert events == [expected_event]
        MockEnforcer.return_value.check.assert_called_once()

        session = thoth.get_current_session()
        assert session is not None
        assert "search" in session.tool_calls


@pytest.mark.parametrize("stack", ["ddtrace", "opentelemetry", "sentry"])
def test_runtime_observability_and_thoth_coexist_block(stack: str) -> None:
    events: list[str] = []
    run = RUNTIME_WRAPPERS[stack](events)
    tool = FakeTool(run)
    executor = FakeExecutor(tool)

    with patch("thoth.instrumentor.EnforcerClient") as MockEnforcer, patch("thoth.instrumentor.HttpEmitter"):
        MockEnforcer.return_value.check.return_value = EnforcementDecision(
            decision=DecisionType.BLOCK,
            reason="blocked by policy",
            violation_id=f"vio-{stack}-001",
        )
        governed = thoth.instrument(
            executor,
            agent_id=f"{stack}-agent",
            approved_scope=["search"],
            tenant_id="trantor",
            api_url="https://enforcer.example",
        )

        with pytest.raises(ThothPolicyViolation):
            governed.tools[0].run("incident 42")

        assert events == []
