"""wrap_autogen_tools: an AutoGen 0.2-style function map, governed."""

from unittest.mock import MagicMock, patch

import pytest

import thoth
from thoth.integrations.autogen import wrap_autogen_tools


def search_docs(query: str) -> str:
    return f"results for {query}"


def test_the_documented_arguments_build_a_governed_map():
    with patch("thoth.instrumentor.EnforcerClient"), patch("thoth.instrumentor.HttpEmitter"):
        governed = wrap_autogen_tools(
            {"search_docs": search_docs},
            agent_id="autogen-assistant",
            approved_scope=["search_docs"],
            tenant_id="acme-corp",
            enforcement="observe",
            api_url="https://enforcer.example",
        )
    assert governed["search_docs"](query="access control") == "results for access control"
    session = thoth.get_current_session()
    assert session is not None
    assert "search_docs" in session.tool_calls


def test_a_tracer_still_works():
    tracer = MagicMock()
    tracer.wrap_tool.side_effect = lambda name, fn: fn
    governed = wrap_autogen_tools({"search_docs": search_docs}, tracer)
    tracer.wrap_tool.assert_called_once_with("search_docs", search_docs)
    assert governed["search_docs"]("q") == "results for q"


def test_a_tracer_and_settings_together_are_refused():
    with pytest.raises(TypeError, match="not both"):
        wrap_autogen_tools({"search_docs": search_docs}, MagicMock(), agent_id="a")
