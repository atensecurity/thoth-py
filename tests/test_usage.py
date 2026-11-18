# tests/test_usage.py
"""thoth.report_usage: model usage rides on the next governed call, once."""

import asyncio
import threading
import json
import math
from types import SimpleNamespace

import httpx
import pytest
import respx
import thoth
from thoth import usage
from thoth.enforcer_client import EnforcerClient
from thoth.models import ThothConfig


@pytest.fixture(autouse=True)
def _fresh_context():
    """Each test starts with no pending usage, whatever ran before it."""
    usage._ACCUMULATOR.take()
    yield
    usage._ACCUMULATOR.take()


@pytest.fixture
def config():
    return ThothConfig(agent_id="test-agent", approved_scope=["read:data"], tenant_id="trantor", api_url="http://enforcer:8080")


def _sent(route) -> list[dict]:
    return [json.loads(call.request.content) for call in route.calls]


@respx.mock
def test_usage_is_sent_with_the_next_governed_call_and_only_once(config):
    route = respx.post(f"{config.resolved_enforcer_url}/v1/enforce").mock(return_value=httpx.Response(200, json={"decision": "ALLOW"}))
    thoth.report_usage(120, 30)
    client = EnforcerClient(config)
    client.check("read:data", session_id="s", tool_calls=[])
    client.check("read:data", session_id="s", tool_calls=[])
    first, second = _sent(route)
    assert first["prompt_tokens"] == 120 and first["completion_tokens"] == 30
    assert "estimated_cost_usd" not in first, "dollars are sent only when the caller passed them"
    assert not {"prompt_tokens", "completion_tokens", "estimated_cost_usd"} & set(second)


@respx.mock
def test_several_model_calls_before_one_governed_call_add_up_and_cost_is_optional(config):
    route = respx.post(f"{config.resolved_enforcer_url}/v1/enforce").mock(return_value=httpx.Response(200, json={"decision": "ALLOW"}))
    thoth.report_usage(10, 5)
    thoth.report_usage(20, 15, cost_usd=0.25)
    EnforcerClient(config).check("read:data", session_id="s", tool_calls=[])
    (sent,) = _sent(route)
    assert (sent["prompt_tokens"], sent["completion_tokens"], sent["estimated_cost_usd"]) == (30, 20, 0.25)


@respx.mock
async def test_the_async_path_sends_usage_too(config):
    route = respx.post(f"{config.resolved_enforcer_url}/v1/enforce").mock(return_value=httpx.Response(200, json={"decision": "ALLOW"}))
    thoth.report_usage(7, 3)
    await EnforcerClient(config).acheck("read:data", session_id="s", tool_calls=[])
    (sent,) = _sent(route)
    assert (sent["prompt_tokens"], sent["completion_tokens"]) == (7, 3)


@respx.mock
def test_usage_is_not_lost_when_the_call_did_not_reach_the_server(config):
    route = respx.post(f"{config.resolved_enforcer_url}/v1/enforce")
    route.side_effect = [httpx.ConnectError("refused"), httpx.Response(200, json={"decision": "ALLOW"})]
    thoth.report_usage(50, 50)
    client = EnforcerClient(config)
    assert client.check("read:data", session_id="s", tool_calls=[]).is_block
    assert client.check("read:data", session_id="s", tool_calls=[]).is_allow
    failed, retried = _sent(route)
    assert failed["prompt_tokens"] == 50 and retried["prompt_tokens"] == 50 and retried["completion_tokens"] == 50


@respx.mock
def test_a_client_error_does_not_resend_usage(config):
    route = respx.post(f"{config.resolved_enforcer_url}/v1/enforce").mock(return_value=httpx.Response(403, json={"detail": "no"}))
    thoth.report_usage(50, 50)
    client = EnforcerClient(config)
    client.check("read:data", session_id="s", tool_calls=[])
    client.check("read:data", session_id="s", tool_calls=[])
    assert "prompt_tokens" in _sent(route)[0] and "prompt_tokens" not in _sent(route)[1]


@respx.mock
def test_explain_requests_do_not_take_the_pending_usage(config):
    respx.post(f"{config.resolved_enforcer_url}/v1/enforce").mock(return_value=httpx.Response(200, json={"decision": "ALLOW"}))
    thoth.report_usage(9, 9)
    client = EnforcerClient(config)
    decision = client.check.__self__.check("read:data", session_id="s", tool_calls=[])
    assert "prompt_tokens" not in client._explain_payload(decision, tool_name="read:data", session_id="s", tool_calls=[])


@pytest.mark.parametrize("bad", [-1, -0.5, math.nan, math.inf, -math.inf, True, "10", [5], {"n": 1}])
def test_invalid_values_are_ignored_and_never_raise(bad):
    thoth.report_usage(bad, bad, cost_usd=bad)
    assert usage.take_pending_usage() is None


def test_zero_is_a_valid_report():
    thoth.report_usage(0, 0)
    assert usage.take_pending_usage() == {"prompt_tokens": 0, "completion_tokens": 0}


def test_one_huge_report_is_clamped_to_the_per_request_maximum():
    thoth.report_usage(10**15, 10**15, cost_usd=1e12)
    assert usage.take_pending_usage() == {"prompt_tokens": usage.MAX_TOKENS, "completion_tokens": 0, "estimated_cost_usd": usage.MAX_COST_USD}
    assert usage.take_pending_usage() == {"completion_tokens": usage.MAX_TOKENS}
    assert usage.take_pending_usage() is None


def _drain() -> tuple[int, int, float]:
    prompt = completion = 0
    cost = 0.0
    while (fields := usage.take_pending_usage()) is not None:
        assert fields.get("prompt_tokens", 0) + fields.get("completion_tokens", 0) <= usage.MAX_TOKENS, "one request never exceeds the server's maximum"
        assert fields.get("estimated_cost_usd", 0.0) <= usage.MAX_COST_USD
        prompt += fields.get("prompt_tokens", 0)
        completion += fields.get("completion_tokens", 0)
        cost += fields.get("estimated_cost_usd", 0.0)
    return prompt, completion, cost


def test_many_reports_beyond_the_cap_are_split_across_requests_and_nothing_is_lost():
    for _ in range(25):  # 25 x (4M, 3M, $90) is far over one request's 10M tokens and $1,000
        thoth.report_usage(4_000_000, 3_000_000, cost_usd=90.0)
    prompt, completion, cost = _drain()
    assert (prompt, completion) == (100_000_000, 75_000_000)
    assert cost == pytest.approx(2250.0)


def test_a_report_after_the_cap_was_reached_still_counts():
    thoth.report_usage(usage.MAX_TOKENS, 0)
    thoth.report_usage(5, 0)
    assert _drain()[0] == usage.MAX_TOKENS + 5


def test_concurrent_reports_from_threads_and_tasks_are_all_counted():
    def worker():
        for _ in range(200):
            thoth.report_usage(1, 2)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()

    async def tasks():
        async def one():
            for _ in range(200):
                thoth.report_usage(1, 2)
                await asyncio.sleep(0)

        await asyncio.gather(*(one() for _ in range(8)))

    asyncio.run(tasks())
    for t in threads:
        t.join()
    assert _drain()[:2] == (24 * 200, 24 * 200 * 2)


def test_concurrent_takers_never_send_the_same_usage_twice():
    for _ in range(1000):
        thoth.report_usage(3, 1)
    taken: list[dict] = []
    lock = threading.Lock()

    def taker():
        while (fields := usage.take_pending_usage()) is not None:
            with lock:
                taken.append(fields)

    threads = [threading.Thread(target=taker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(f.get("prompt_tokens", 0) for f in taken) == 3000
    assert sum(f.get("completion_tokens", 0) for f in taken) == 1000


def test_usage_reported_with_no_governed_call_is_not_sent_by_shutdown():
    """Documented limit: there is no request to carry it, and shutdown() does not invent one."""
    thoth.report_usage(7, 7)
    thoth.shutdown()
    assert usage.take_pending_usage() == {"prompt_tokens": 7, "completion_tokens": 7}


def test_nothing_pending_means_no_fields():
    assert usage.take_pending_usage() is None


def test_report_usage_from_reads_openai_and_anthropic_shapes_and_never_raises():
    thoth.report_usage_from(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4)))
    assert usage.take_pending_usage() == {"prompt_tokens": 11, "completion_tokens": 4}
    thoth.report_usage_from({"usage": {"input_tokens": 8, "output_tokens": 2}})
    assert usage.take_pending_usage() == {"prompt_tokens": 8, "completion_tokens": 2}
    thoth.report_usage_from(SimpleNamespace(usage=None))
    thoth.report_usage_from(object())
    thoth.report_usage_from(None)
    assert usage.take_pending_usage() is None


def test_usage_reported_in_another_thread_or_task_still_counts():
    done = threading.Thread(target=lambda: thoth.report_usage(4, 1))
    done.start()
    done.join()

    async def other_task():
        thoth.report_usage(6, 2)

    asyncio.run(other_task())
    assert usage.take_pending_usage() == {"prompt_tokens": 10, "completion_tokens": 3}
