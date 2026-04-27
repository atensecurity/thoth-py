from __future__ import annotations

import logging
import os

_THOTH_LOGGER_NAME = "thoth"
_THOTH_ENV_HANDLER_NAME = "thoth-env-stream"
_THOTH_ENV_HANDLER_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_LEVEL_ALIASES = {"WARN": "WARNING"}
_VALID_LEVEL_NAMES = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


def _resolve_level_from_env() -> int | None:
    raw = (os.getenv("THOTH_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "").strip()
    if not raw:
        return None

    normalized = _LEVEL_ALIASES.get(raw.upper(), raw.upper())
    if normalized.isdigit():
        return int(normalized)
    if normalized not in _VALID_LEVEL_NAMES:
        return None
    return int(getattr(logging, normalized))


def configure_thoth_logging_from_env() -> None:
    """Configure the ``thoth`` logger from env vars when explicitly requested.

    Priority: ``THOTH_LOG_LEVEL`` → ``LOG_LEVEL``.
    If the process has no root handlers, attach a minimal stream handler so SDK
    logs remain visible in simple scripts.
    """
    level = _resolve_level_from_env()
    if level is None:
        return

    thoth_logger = logging.getLogger(_THOTH_LOGGER_NAME)
    thoth_logger.setLevel(level)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    for handler in thoth_logger.handlers:
        if handler.get_name() == _THOTH_ENV_HANDLER_NAME:
            handler.setLevel(level)
            return

    handler = logging.StreamHandler()
    handler.set_name(_THOTH_ENV_HANDLER_NAME)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_THOTH_ENV_HANDLER_FORMAT))
    thoth_logger.addHandler(handler)
    thoth_logger.propagate = False
