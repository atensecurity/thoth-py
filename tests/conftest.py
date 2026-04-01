# tests/conftest.py
import pytest
from thoth.models import EnforcementMode, ThothConfig


@pytest.fixture
def base_config():
    return ThothConfig(
        agent_id="test-agent",
        approved_scope=["read:data", "write:slack"],
        tenant_id="trantor",
        enforcement=EnforcementMode.BLOCK,
        enforcer_url="http://enforcer:8080",
    )
