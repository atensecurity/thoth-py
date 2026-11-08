"""thoth.init() with an Anthropic SDK that predates ToolError (0.x): its tools stay ungoverned, init still works."""

import sys
import types

from thoth import auto


def test_an_anthropic_sdk_without_toolerror_is_left_alone_and_says_so(monkeypatch, caplog):
    legacy = types.ModuleType("anthropic.lib.tools")
    legacy.BetaFunctionTool = type("BetaFunctionTool", (), {"call": lambda self, tool_input: tool_input})
    monkeypatch.setitem(sys.modules, "anthropic.lib.tools", legacy)

    with caplog.at_level("WARNING", logger="thoth.auto"):
        assert auto._install_anthropic() is None
    assert "Anthropic tools are NOT governed" in caplog.text
