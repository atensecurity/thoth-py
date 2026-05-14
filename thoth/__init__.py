# thoth/__init__.py
from importlib.metadata import PackageNotFoundError, version

from thoth._context import get_current_session
from thoth.client import ThothClient
from thoth.exceptions import ThothPolicyViolation
from thoth.instrumentor import (
    instrument,
    instrument_anthropic,
    instrument_claude_agent_sdk,
    instrument_openai,
)

try:
    __version__ = version("atensec-thoth")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "ThothClient",
    "ThothPolicyViolation",
    "__version__",
    "get_current_session",
    "instrument",
    "instrument_anthropic",
    "instrument_claude_agent_sdk",
    "instrument_openai",
]
