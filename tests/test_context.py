# tests/test_context.py
from thoth._context import _CURRENT_SESSION, get_current_session
from thoth.models import EnforcementMode, ThothConfig
from thoth.session import SessionContext


def test_get_current_session_returns_none_by_default():
    # Reset to default to isolate from other tests that may have set the contextvar
    token = _CURRENT_SESSION.set(None)
    try:
        assert get_current_session() is None
    finally:
        _CURRENT_SESSION.reset(token)


def test_get_current_session_returns_set_session():
    config = ThothConfig(
        agent_id="a",
        approved_scope=[],
        tenant_id="t",
        enforcement=EnforcementMode.OBSERVE,
        api_url="http://x",
        sqs_queue_url=None,
    )
    session = SessionContext(config)
    token = _CURRENT_SESSION.set(session)
    try:
        assert get_current_session() is session
    finally:
        _CURRENT_SESSION.reset(token)
