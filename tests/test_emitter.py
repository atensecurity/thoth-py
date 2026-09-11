# tests/test_emitter.py
from datetime import UTC, datetime
import json
import time

import boto3
import httpx
from moto import mock_aws
import pytest

from thoth.emitter import HttpEmitter, SqsEmitter
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
        occurred_at=datetime.now(UTC),
    )


def make_sensitive_event() -> BehavioralEvent:
    canary = "SYNTHETIC-PHI-SECRET-4471"
    return BehavioralEvent(
        event_id="event-safe-001",
        tenant_id="trantor",
        session_id="session-safe-001",
        user_id="user-safe-001",
        agent_id="agent-safe-001",
        purpose=canary,
        data_classification=canary,
        task_context={"patient": canary},
        initiated_by=canary,
        task_id=canary,
        delegation_chain=[canary],
        source_type=SourceType.AGENT_TOOL_CALL,
        event_type=EventType.TOOL_CALL_BLOCK,
        tool_name="read_document",
        content=canary,
        approved_scope=["read_document"],
        session_tool_calls=["read_document"],
        violation_id="violation-safe-001",
        metadata={
            "sdk_language": "python",
            "environment": "dev",
            "enforcement_trace_id": "trace-safe-001",
            "action_attestation_id": "action-safe-001",
            "event_phase": "block",
            "authorization_decision": "BLOCK",
            "decision_reason_code": "policy_block",
            "action_classification": "read",
            "risk_score": 91.5,
            "tool_args": {"path": canary},
            "tool_call": {"name": "read_document", "arguments": {"path": canary}},
            "task_context": {"patient": canary},
            "human_explanation": {"summary": canary},
            "error": canary,
            "unknown": {"nested": canary},
            "receipt": {
                "receipt_id": "receipt-safe-001",
                "signature": "signature-safe-001",
                "signing_algorithm": "ED25519",
                "audit_envelope": {"auth_context": {"patient": canary}},
                "decision": {
                    "authorization_decision": "BLOCK",
                    "reason": canary,
                    "outcome": "blocked",
                },
            },
            "decision_evidence": {
                "decision_reason_code": "policy_block",
                "authorization_decision": "BLOCK",
                "risk_score": 91.5,
                "policy": {
                    "matched_rule_ids": ["rule-safe-001"],
                    "explanation": canary,
                },
                "unknown": canary,
            },
        },
    )


class RecordingSQSClient:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    def send_message_batch(self, *, QueueUrl, Entries):  # type: ignore[no-untyped-def]
        self.entries.extend(Entries)
        return {"Successful": [{"Id": entry["Id"]} for entry in Entries]}


class PartialFailureSQSClient:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def send_message_batch(self, *, QueueUrl, Entries):  # type: ignore[no-untyped-def]
        self.calls.append(Entries)
        if len(self.calls) == 1:
            return {
                "Successful": [{"Id": Entries[0]["Id"]}],
                "Failed": [{"Id": Entries[1]["Id"], "Code": "ServiceUnavailable", "SenderFault": False}],
            }
        return {"Successful": [{"Id": entry["Id"]} for entry in Entries], "Failed": []}


def test_http_retries_transient_failures_with_stable_ids():
    event = make_event()
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        status = 202 if len(requests) == 3 else 503
        return httpx.Response(status, request=request)

    emitter = HttpEmitter("https://ingest.example", "test-key")
    emitter._http.close()
    emitter._http = httpx.Client(transport=httpx.MockTransport(handle))
    result = emitter._send_batch([event])

    assert result.delivered_event_ids == [event.event_id]
    assert result.dropped_event_ids == []
    assert result.attempts == 3
    assert [json.loads(request.content)[0]["event_id"] for request in requests] == [event.event_id] * 3


def test_http_close_is_bounded_and_reports_delivery_status():
    emitter = HttpEmitter("https://ingest.example", "test-key")
    emitter._http.close()
    emitter._http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(202, request=request)))
    emitter.emit(make_event())

    status = emitter.close(timeout=2.0)

    assert status.pending == 0
    assert status.delivered == 1
    assert status.dropped == 0


def test_sqs_retries_only_failed_entries_with_stable_ids():
    events = [make_event("session-1"), make_event("session-2")]
    client = PartialFailureSQSClient()
    emitter = SqsEmitter(queue_url=None)
    emitter._queue_url = "https://sqs.example/queue.fifo"
    emitter._client = client

    result = emitter._send_batch(events)

    assert [len(call) for call in client.calls] == [2, 1]
    assert client.calls[0][1]["MessageDeduplicationId"] == client.calls[1][0]["MessageDeduplicationId"]
    assert set(result.delivered_event_ids) == {event.event_id for event in events}
    assert result.dropped_event_ids == []
    assert result.attempts == 2


def test_http_telemetry_uses_minimal_projection_without_mutating_authorization_event():
    event = make_sensitive_event()
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(202, request=request)

    emitter = HttpEmitter("https://ingest.example", "test-key")
    emitter._http.close()
    emitter._http = httpx.Client(transport=httpx.MockTransport(handle))
    emitter._send_batch([event])

    assert len(requests) == 1
    raw = requests[0].content.decode()
    assert "SYNTHETIC-PHI-SECRET-4471" not in raw
    sent = json.loads(raw)[0]
    assert sent["event_id"] == "trantor:event-safe-001"
    assert sent["session_id"] == "session-safe-001"
    assert sent["metadata"]["action_attestation_id"] == "action-safe-001"
    assert sent["metadata"]["authorization_decision"] == "BLOCK"
    assert sent["metadata"]["receipt"] == {
        "receipt_id": "receipt-safe-001",
        "signature": "signature-safe-001",
        "signing_algorithm": "ED25519",
        "decision": {"authorization_decision": "BLOCK", "outcome": "blocked"},
    }
    assert sent["metadata"]["decision_evidence"] == {
        "decision_reason_code": "policy_block",
        "authorization_decision": "BLOCK",
        "risk_score": 91.5,
        "policy": {"matched_rule_ids": ["rule-safe-001"]},
    }
    assert event.metadata["tool_args"] == {"path": "SYNTHETIC-PHI-SECRET-4471"}


def test_sqs_telemetry_uses_the_same_minimal_projection():
    event = make_sensitive_event()
    client = RecordingSQSClient()
    emitter = SqsEmitter(queue_url=None)
    emitter._queue_url = "https://sqs.example/queue.fifo"
    emitter._client = client
    emitter._send_batch([event])

    assert len(client.entries) == 1
    raw = client.entries[0]["MessageBody"]
    assert "SYNTHETIC-PHI-SECRET-4471" not in raw
    sent = json.loads(raw)
    assert sent["event_id"] == "trantor:event-safe-001"
    assert sent["metadata"]["enforcement_trace_id"] == "trace-safe-001"


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
