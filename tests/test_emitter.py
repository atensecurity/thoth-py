# tests/test_emitter.py
import json
import time
from datetime import datetime, timezone

import boto3
import pytest
from moto import mock_aws
from thoth.emitter import _BATCH_MAX, SqsEmitter
from thoth.models import BehavioralEvent, EventType, SourceType


@pytest.fixture
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")


@pytest.fixture
def sqs_queue(aws_credentials):
    with mock_aws():
        sqs = boto3.client("sqs", region_name="us-west-2")
        result = sqs.create_queue(
            QueueName="thoth-test.fifo",
            Attributes={"FifoQueue": "true", "ContentBasedDeduplication": "true"},
        )
        yield result["QueueUrl"]


def make_event(session_id: str = "sess_abc") -> BehavioralEvent:
    return BehavioralEvent(
        tenant_id="trantor",
        session_id=session_id,
        user_id="user_xyz",
        agent_id="test-agent",
        source_type=SourceType.AGENT_TOOL_CALL,
        event_type=EventType.TOOL_CALL_PRE,
        content="read:invoices",
        approved_scope=["read:invoices"],
        occurred_at=datetime.now(timezone.utc),
    )


def test_emit_is_nonblocking_queue_put(aws_credentials):
    """emit() enqueues without blocking -- does not start a thread per event."""
    with mock_aws():
        sqs = boto3.client("sqs", region_name="us-west-2")
        queue_url = sqs.create_queue(
            QueueName="thoth-nb.fifo",
            Attributes={"FifoQueue": "true", "ContentBasedDeduplication": "true"},
        )["QueueUrl"]
        emitter = SqsEmitter(queue_url=queue_url, region="us-west-2")
        event = make_event()
        # Should not raise or block
        emitter.emit(event)
        # Queue size may be 0 or 1 depending on drain timing -- either is valid
        assert emitter._queue.qsize() <= 1


def test_batch_of_events_calls_send_message_batch(sqs_queue):
    """Emitting multiple events results in send_message_batch calls."""
    with mock_aws():
        emitter = SqsEmitter(queue_url=sqs_queue, region="us-west-2")
        events = [make_event(f"sess_{i}") for i in range(3)]
        for e in events:
            emitter.emit(e)

        # Give background worker time to flush
        time.sleep(0.6)

        sqs = boto3.client("sqs", region_name="us-west-2")
        received = []
        for _ in range(5):
            resp = sqs.receive_message(QueueUrl=sqs_queue, MaxNumberOfMessages=10)
            received.extend(resp.get("Messages", []))
            if len(received) >= 3:
                break
            time.sleep(0.1)

        assert len(received) == 3
        event_ids = {json.loads(m["Body"])["event_id"] for m in received}
        assert event_ids == {e.event_id for e in events}


def test_queue_full_drops_silently(aws_credentials, monkeypatch):
    """When queue is at capacity, emit() drops events without raising."""
    with mock_aws():
        sqs = boto3.client("sqs", region_name="us-west-2")
        queue_url = sqs.create_queue(
            QueueName="thoth-full.fifo",
            Attributes={"FifoQueue": "true", "ContentBasedDeduplication": "true"},
        )["QueueUrl"]
        emitter = SqsEmitter(queue_url=queue_url, region="us-west-2")

        # Simulate a full internal queue by patching put_nowait to raise Full
        from queue import Full

        def raising_put(item):  # type: ignore[no-untyped-def]
            raise Full

        monkeypatch.setattr(emitter._queue, "put_nowait", raising_put)

        # Should not raise
        emitter.emit(make_event())


def test_atexit_flush_sends_remaining_events(aws_credentials):
    """_flush() drains remaining queue items and sends them."""
    with mock_aws():
        sqs = boto3.client("sqs", region_name="us-west-2")
        queue_url = sqs.create_queue(
            QueueName="thoth-flush.fifo",
            Attributes={"FifoQueue": "true", "ContentBasedDeduplication": "true"},
        )["QueueUrl"]
        emitter = SqsEmitter(queue_url=queue_url, region="us-west-2")

        # Directly place events in the internal queue, bypassing the worker
        events = [make_event(f"flush_{i}") for i in range(3)]
        for e in events:
            emitter._queue.put_nowait(e)

        # Call flush directly (simulating atexit)
        emitter._flush()

        sqs_client = boto3.client("sqs", region_name="us-west-2")
        received = []
        for _ in range(5):
            resp = sqs_client.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=10)
            received.extend(resp.get("Messages", []))
            if len(received) >= 3:
                break
            time.sleep(0.05)

        assert len(received) == 3


def test_emitter_noop_when_no_queue_url():
    """Should not raise -- non-fatal if SQS not configured."""
    emitter = SqsEmitter(queue_url=None, region="us-west-2")
    emitter.emit(make_event())  # must not raise
