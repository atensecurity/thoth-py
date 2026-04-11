# tests/integrations/test_langchain.py
from unittest.mock import MagicMock, patch

import pytest
from thoth.models import DecisionType, EnforcementDecision


def test_langchain_agent_executor_instrumented():
    """instrument() should wrap LangChain AgentExecutor tools."""

    # Minimal AgentExecutor duck-type
    class FakeTool:
        name = "search"

        def run(self, query):
            return f"results:{query}"

    class FakeExecutor:
        tools = [FakeTool()]

    import thoth

    with patch("thoth.instrumentor.EnforcerClient") as MockEnforcer, patch("thoth.instrumentor.HttpEmitter"):
        MockEnforcer.return_value.check.return_value = EnforcementDecision(decision=DecisionType.ALLOW)
        executor = thoth.instrument(
            FakeExecutor(),
            agent_id="lc-agent",
            approved_scope=["search"],
            tenant_id="trantor",
            api_url="https://enforcer.example",
        )
        result = executor.tools[0].run("test query")
        assert result == "results:test query"
