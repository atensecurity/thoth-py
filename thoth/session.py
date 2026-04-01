# thoth/session.py
from __future__ import annotations

import uuid

from thoth.models import ThothConfig


class SessionContext:
    def __init__(self, config: ThothConfig, session_id: str | None = None) -> None:
        self._config = config
        self.session_id: str = session_id or str(uuid.uuid4())
        self.tool_calls: list[str] = []
        self.token_spend: int = 0

    def record_tool_call(self, tool_name: str) -> None:
        self.tool_calls.append(tool_name)

    def record_token_spend(self, tokens: int) -> None:
        self.token_spend += tokens

    def is_in_scope(self, tool_name: str) -> bool:
        return tool_name in self._config.approved_scope
