# thoth/emitter.py
from __future__ import annotations

import atexit
import logging
from queue import Empty, Full, Queue
import threading
from typing import Any

import boto3
import httpx

from thoth.models import BehavioralEvent

logger = logging.getLogger(__name__)

_QUEUE_MAX = 1000
_BATCH_MAX = 10
_DRAIN_TIMEOUT_S = 0.25


class SqsEmitter:
    def __init__(self, queue_url: str | None, region: str = "us-west-2") -> None:
        self._queue_url = queue_url
        self._client: Any | None = boto3.client("sqs", region_name=region) if queue_url else None
        self._queue: Queue[BehavioralEvent] = Queue(maxsize=_QUEUE_MAX)
        self._worker = threading.Thread(target=self._drain_loop, daemon=True, name="thoth-emitter")
        self._worker.start()
        atexit.register(self._flush)

    def emit(self, event: BehavioralEvent) -> None:
        """Non-blocking enqueue. Emits an error when queue pressure drops events."""
        if not self._queue_url:
            return
        try:
            self._queue.put_nowait(event)
        except Full:
            logger.error(
                "thoth: telemetry queue full, dropping event_id=%s (event dropped)",
                event.event_id,
            )

    def _drain_loop(self) -> None:
        while True:
            batch = self._collect_batch()
            if batch:
                self._send_batch(batch)

    def _collect_batch(self) -> list[BehavioralEvent]:
        batch: list[BehavioralEvent] = []
        try:
            batch.append(self._queue.get(timeout=_DRAIN_TIMEOUT_S))
            while len(batch) < _BATCH_MAX:
                batch.append(self._queue.get_nowait())
        except Empty:
            pass
        return batch

    def _send_batch(self, events: list[BehavioralEvent]) -> None:
        assert self._client is not None
        try:
            self._client.send_message_batch(
                QueueUrl=self._queue_url,
                Entries=[
                    {
                        "Id": str(i),
                        "MessageBody": e.model_dump_json(),
                        "MessageGroupId": e.session_id,
                        "MessageDeduplicationId": e.event_id,
                    }
                    for i, e in enumerate(events)
                ],
            )
        except Exception:
            logger.error(
                "thoth: failed to send telemetry batch of %d events (events dropped)",
                len(events),
                exc_info=True,
            )

    def _flush(self) -> None:
        """Drain remaining events on process exit (best-effort)."""
        remaining: list[BehavioralEvent] = []
        while True:
            try:
                remaining.append(self._queue.get_nowait())
                if len(remaining) == _BATCH_MAX:
                    self._send_batch(remaining)
                    remaining = []
            except Empty:
                break
        if remaining:
            self._send_batch(remaining)


_HTTP_TIMEOUT = httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=2.0)


class HttpEmitter:
    """Emitter for the Aten-hosted path. Sends events to the Thoth ingest API
    using an API key — no AWS credentials required."""

    def __init__(self, api_url: str, api_key: str) -> None:
        self._endpoint = f"{api_url.rstrip('/')}/v1/events/batch"
        api_key_value = (api_key or "").strip()
        # Send both auth header styles. Some customer ingress stacks strip or
        # transform Authorization while preserving X-Api-Key.
        headers = {"Content-Type": "application/json"}
        if api_key_value:
            headers["Authorization"] = f"Bearer {api_key_value}"
            headers["X-Api-Key"] = api_key_value
        self._http = httpx.Client(
            headers=headers,
            timeout=_HTTP_TIMEOUT,
        )
        self._queue: Queue[BehavioralEvent] = Queue(maxsize=_QUEUE_MAX)
        self._worker = threading.Thread(target=self._drain_loop, daemon=True, name="thoth-http-emitter")
        self._worker.start()
        atexit.register(self._flush)

    def emit(self, event: BehavioralEvent) -> None:
        """Non-blocking enqueue. Emits an error when queue pressure drops events."""
        try:
            self._queue.put_nowait(event)
        except Full:
            logger.error(
                "thoth: telemetry queue full, dropping event_id=%s (event dropped)",
                event.event_id,
            )

    def _drain_loop(self) -> None:
        while True:
            batch = self._collect_batch()
            if batch:
                self._send_batch(batch)

    def _collect_batch(self) -> list[BehavioralEvent]:
        batch: list[BehavioralEvent] = []
        try:
            batch.append(self._queue.get(timeout=_DRAIN_TIMEOUT_S))
            while len(batch) < _BATCH_MAX:
                batch.append(self._queue.get_nowait())
        except Empty:
            pass
        return batch

    def _send_batch(self, events: list[BehavioralEvent]) -> None:
        try:
            payload = [e.model_dump(mode="json") for e in events]
            self._http.post(self._endpoint, json=payload).raise_for_status()
        except httpx.HTTPStatusError as exc:
            response = exc.response
            first_event = events[0] if events else None
            body = (response.text or "").strip()
            detail = body[:512] if body else "<empty>"
            logger.error(
                (
                    "thoth: ingest API rejected telemetry "
                    "(status=%s url=%s tenant_id=%s agent_id=%s event_type=%s detail=%s)"
                ),
                response.status_code,
                str(response.request.url),
                getattr(first_event, "tenant_id", None),
                getattr(first_event, "agent_id", None),
                getattr(first_event, "event_type", None),
                detail,
                exc_info=True,
            )
        except Exception as exc:
            logger.error(
                "thoth: ingest API unreachable (%s) — dropping %d telemetry events",
                exc,
                len(events),
                exc_info=True,
            )

    def _flush(self) -> None:
        """Drain remaining events on process exit (best-effort)."""
        remaining: list[BehavioralEvent] = []
        while True:
            try:
                remaining.append(self._queue.get_nowait())
                if len(remaining) == _BATCH_MAX:
                    self._send_batch(remaining)
                    remaining = []
            except Empty:
                break
        if remaining:
            self._send_batch(remaining)
