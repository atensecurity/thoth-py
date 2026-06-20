# tests/integrations/test_langsmith_compat.py
from unittest.mock import patch

import pytest

import thoth
from thoth.models import DecisionType, EnforcementDecision


def test_langsmith_and_thoth_instrumentation_can_coexist():
    """A LangSmith-traceable tool should still be governed by Thoth wrappers."""
    run_helpers = pytest.importorskip("langsmith.run_helpers")
    traceable = run_helpers.traceable

    class FakeTool:
        name = "search"

        @traceable(name="search_tool", run_type="tool")
        def run(self, query):
            return f"results:{query}"

    class FakeExecutor:
        tools = [FakeTool()]

    with (
        patch.object(run_helpers, "_setup_run", wraps=run_helpers._setup_run) as setup_run_spy,
        patch("thoth.instrumentor.EnforcerClient") as MockEnforcer,
        patch("thoth.instrumentor.HttpEmitter"),
    ):
        MockEnforcer.return_value.check.return_value = EnforcementDecision(decision=DecisionType.ALLOW)
        executor = thoth.instrument(
            FakeExecutor(),
            agent_id="lc-agent",
            approved_scope=["search"],
            tenant_id="trantor",
            api_url="https://enforcer.example",
        )
        result = executor.tools[0].run("incident 42")

        assert result == "results:incident 42"
        assert setup_run_spy.call_count >= 1
        MockEnforcer.return_value.check.assert_called_once()

        session = thoth.get_current_session()
        assert session is not None
        assert "search" in session.tool_calls
