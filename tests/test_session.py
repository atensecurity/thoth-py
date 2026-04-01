# tests/test_session.py
from thoth.models import ThothConfig
from thoth.session import SessionContext


def make_config(**kwargs):
    return ThothConfig(
        agent_id="test-agent",
        approved_scope=["read:data", "write:slack"],
        tenant_id="trantor",
        **kwargs,
    )


def test_session_initializes_with_config():
    ctx = SessionContext(make_config())
    assert ctx.session_id  # auto-generated UUID
    assert ctx.tool_calls == []
    assert ctx.token_spend == 0


def test_record_tool_call():
    ctx = SessionContext(make_config())
    ctx.record_tool_call("read:data")
    ctx.record_tool_call("write:slack")
    assert ctx.tool_calls == ["read:data", "write:slack"]


def test_record_token_spend():
    ctx = SessionContext(make_config())
    ctx.record_token_spend(500)
    ctx.record_token_spend(300)
    assert ctx.token_spend == 800


def test_is_in_scope_true():
    ctx = SessionContext(make_config())
    assert ctx.is_in_scope("read:data") is True


def test_is_in_scope_false():
    ctx = SessionContext(make_config())
    assert ctx.is_in_scope("write:s3") is False


def test_session_id_is_stable():
    ctx = SessionContext(make_config())
    sid = ctx.session_id
    ctx.record_tool_call("read:data")
    assert ctx.session_id == sid


def test_custom_session_id():
    ctx = SessionContext(make_config(), session_id="my-session-123")
    assert ctx.session_id == "my-session-123"
