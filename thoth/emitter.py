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
        """Non-blocking enqueue. Drops silently when queue is full."""
        if not self._queue_url:
            return
        try:
            self._queue.put_nowait(event)
        except Full:
            logger.warning("thoth: event queue full, dropping %s", event.event_id)

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
            logger.warning("thoth: failed to send batch of %d events", len(events), exc_info=True)

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
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=_HTTP_TIMEOUT,
        )
        self._queue: Queue[BehavioralEvent] = Queue(maxsize=_QUEUE_MAX)
        self._worker = threading.Thread(target=self._drain_loop, daemon=True, name="thoth-http-emitter")
        self._warned = False  # warn once on first failure, then stay quiet
        self._worker.start()
        atexit.register(self._flush)

    def emit(self, event: BehavioralEvent) -> None:
        """Non-blocking enqueue. Drops silently when queue is full."""
        try:
            self._queue.put_nowait(event)
        except Full:
            logger.warning("thoth: event queue full, dropping %s", event.event_id)

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
            self._warned = False  # reset on success so future failures are reported
        except Exception as exc:
            if not self._warned:
                logger.warning("thoth: ingest API unreachable (%s) — events will be dropped", exc)
                self._warned = True

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
