# thoth/__init__.py
from importlib.metadata import PackageNotFoundError, version

from thoth._context import get_current_session
from thoth.auto import acting_for, init, shutdown
from thoth.client import ThothClient
from thoth.exceptions import ThothDeferredError, ThothPolicyViolation
from thoth.instrumentor import (
    instrument,
    instrument_anthropic,
    instrument_claude_agent_sdk,
    instrument_openai,
    instrument_toolchain,
    toolchain_function_map,
)
from thoth.integrations.langgraph import (
    instrument_langgraph,
    instrument_tool_node,
    thoth_graph,
)
from thoth.usage import report_usage, report_usage_from

try:
    __version__ = version("atensec-thoth")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "ThothClient",
    "ThothDeferredError",
    "ThothPolicyViolation",
    "__version__",
    "acting_for",
    "get_current_session",
    "init",
    "instrument",
    "instrument_anthropic",
    "instrument_claude_agent_sdk",
    "instrument_langgraph",
    "instrument_openai",
    "instrument_tool_node",
    "instrument_toolchain",
    "report_usage",
    "report_usage_from",
    "shutdown",
    "thoth_graph",
    "toolchain_function_map",
]
