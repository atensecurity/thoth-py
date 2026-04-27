import logging

from thoth.logging_config import configure_thoth_logging_from_env


def test_configure_thoth_logging_uses_thoth_log_level(monkeypatch):
    monkeypatch.setenv("THOTH_LOG_LEVEL", "debug")
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    thoth_logger = logging.getLogger("thoth")
    original_level = thoth_logger.level

    try:
        thoth_logger.setLevel(logging.INFO)
        configure_thoth_logging_from_env()
        assert thoth_logger.level == logging.DEBUG
    finally:
        thoth_logger.setLevel(original_level)


def test_configure_thoth_logging_falls_back_to_generic_log_level(monkeypatch):
    monkeypatch.delenv("THOTH_LOG_LEVEL", raising=False)
    monkeypatch.setenv("LOG_LEVEL", "warning")

    thoth_logger = logging.getLogger("thoth")
    original_level = thoth_logger.level

    try:
        thoth_logger.setLevel(logging.INFO)
        configure_thoth_logging_from_env()
        assert thoth_logger.level == logging.WARNING
    finally:
        thoth_logger.setLevel(original_level)
