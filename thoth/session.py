# thoth/session.py
from __future__ import annotations

import threading
import uuid

from thoth.models import ThothConfig


class SessionContext:
    def __init__(self, config: ThothConfig, session_id: str | None = None) -> None:
        self._config = config
        self.session_id: str = session_id or str(uuid.uuid4())
        self._tool_calls: list[str] = []
        self._token_spend: int = 0
        self._lock = threading.RLock()

    @property
    def tool_calls(self) -> list[str]:
        with self._lock:
            return list(self._tool_calls)

    @property
    def token_spend(self) -> int:
        with self._lock:
            return self._token_spend

    def record_tool_call(self, tool_name: str) -> None:
        with self._lock:
            self._tool_calls.append(tool_name)

    def pending_tool_calls(self, tool_name: str) -> list[str]:
        with self._lock:
            pending = list(self._tool_calls)
            if not pending or pending[-1] != tool_name:
                pending.append(tool_name)
            return pending

    def record_token_spend(self, tokens: int) -> None:
        with self._lock:
            self._token_spend += tokens

    def is_in_scope(self, tool_name: str) -> bool:
        return tool_name in self._config.approved_scope
