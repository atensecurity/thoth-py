# thoth/emitter.py
from __future__ import annotations

import atexit
from dataclasses import dataclass
import json
import logging
from queue import Empty, Full, Queue
import threading
import time
from typing import Any

import boto3
import httpx

from thoth.http_diagnostics import auth_failure_hint, extract_http_error_detail
from thoth.logging_config import configure_thoth_logging_from_env
from thoth.models import BehavioralEvent
from thoth.telemetry import telemetry_event

logger = logging.getLogger(__name__)

_QUEUE_MAX = 1000
_BATCH_MAX = 10
_DRAIN_TIMEOUT_S = 0.25
_MAX_ATTEMPTS = 3
_RETRY_DELAY_S = 0.1


@dataclass(frozen=True)
class DeliveryBatchStatus:
    delivered_event_ids: list[str]
    dropped_event_ids: list[str]
    attempts: int


@dataclass(frozen=True)
class DeliveryStatus:
    pending: int
    delivered: int
    dropped: int
    retried: int


class _DeliveryTracker:
    def _init_delivery(self) -> None:
        self._delivery_lock = threading.Lock()
        self._pending = 0
        self._delivered = 0
        self._dropped = 0
        self._retried = 0
        self._closed = threading.Event()

    def delivery_status(self) -> DeliveryStatus:
        with self._delivery_lock:
            return DeliveryStatus(self._pending, self._delivered, self._dropped, self._retried)

    def _record_result(self, result: DeliveryBatchStatus) -> None:
        with self._delivery_lock:
            completed = len(result.delivered_event_ids) + len(result.dropped_event_ids)
            self._pending = max(0, self._pending - completed)
            self._delivered += len(result.delivered_event_ids)
            self._dropped += len(result.dropped_event_ids)
            self._retried += max(0, result.attempts - 1)


class SqsEmitter(_DeliveryTracker):
    def __init__(self, queue_url: str | None, region: str = "us-west-2") -> None:
        configure_thoth_logging_from_env()
        self._init_delivery()
        self._queue_url = queue_url
        self._client: Any | None = boto3.client("sqs", region_name=region) if queue_url else None
        self._queue: Queue[BehavioralEvent] = Queue(maxsize=_QUEUE_MAX)
        self._worker = threading.Thread(target=self._drain_loop, daemon=True, name="thoth-emitter")
        self._worker.start()
        atexit.register(self.close)

    def emit(self, event: BehavioralEvent) -> None:
        """Non-blocking enqueue. Emits an error when queue pressure drops events."""
        if not self._queue_url:
            return
        with self._delivery_lock:
            try:
                self._queue.put_nowait(event)
                self._pending += 1
            except Full:
                self._dropped += 1
                logger.error(
                    "thoth: telemetry queue full, dropping event_id=%s (event dropped)",
                    event.event_id,
                )

    def _drain_loop(self) -> None:
        while not self._closed.is_set() or not self._queue.empty():
            batch = self._collect_batch()
            if batch:
                self._record_result(self._send_batch(batch))

    def _collect_batch(self) -> list[BehavioralEvent]:
        batch: list[BehavioralEvent] = []
        try:
            batch.append(self._queue.get(timeout=_DRAIN_TIMEOUT_S))
            while len(batch) < _BATCH_MAX:
                batch.append(self._queue.get_nowait())
        except Empty:
            pass
        return batch

    def _send_batch(self, events: list[BehavioralEvent]) -> DeliveryBatchStatus:
        assert self._client is not None
        pending = list(enumerate(events))
        delivered: list[str] = []
        dropped: list[str] = []
        attempts = 0
        while pending and attempts < _MAX_ATTEMPTS:
            attempts += 1
            entries = [
                {
                    "Id": str(index),
                    "MessageBody": json.dumps(telemetry_event(event), separators=(",", ":")),
                    "MessageGroupId": event.session_id,
                    "MessageDeduplicationId": event.event_id,
                }
                for index, event in pending
            ]
            try:
                response = self._client.send_message_batch(
                    QueueUrl=self._queue_url,
                    Entries=entries,
                )
            except Exception:
                if attempts < _MAX_ATTEMPTS:
                    time.sleep(_RETRY_DELAY_S * attempts)
                    continue
                break
            failed = {item.get("Id") for item in response.get("Failed", []) if not item.get("SenderFault")}
            terminal = {item.get("Id") for item in response.get("Failed", []) if item.get("SenderFault")}
            delivered.extend(event.event_id for index, event in pending if str(index) not in failed | terminal)
            dropped.extend(event.event_id for index, event in pending if str(index) in terminal)
            pending = [(index, event) for index, event in pending if str(index) in failed]
            if pending and attempts < _MAX_ATTEMPTS:
                time.sleep(_RETRY_DELAY_S * attempts)
        dropped.extend(event.event_id for _, event in pending)
        if dropped:
            logger.error(
                "thoth: failed to deliver %d telemetry events after %d attempts",
                len(dropped),
                attempts,
            )
        return DeliveryBatchStatus(delivered, dropped, attempts)

    def close(self, timeout: float = 5.0) -> DeliveryStatus:
        self._closed.set()
        self._worker.join(max(0.0, timeout))
        return self.delivery_status()

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


class HttpEmitter(_DeliveryTracker):
    """Emitter for the Aten-hosted path. Sends events to the Thoth ingest API
    using an API key — no AWS credentials required."""

    def __init__(self, api_url: str, api_key: str, event_ingest_token: str = "") -> None:
        configure_thoth_logging_from_env()
        self._init_delivery()
        self._endpoint = f"{api_url.rstrip('/')}/v1/events/batch"
        api_key_value = (api_key or "").strip()
        ingest_token_value = (event_ingest_token or "").strip()
        # Send both auth header styles. Some customer ingress stacks strip or
        # transform Authorization while preserving X-Api-Key.
        headers = {"Content-Type": "application/json"}
        if api_key_value:
            headers["Authorization"] = f"Bearer {api_key_value}"
            headers["X-Api-Key"] = api_key_value
        if ingest_token_value:
            headers["X-Thoth-Event-Ingest-Token"] = ingest_token_value
        self._http = httpx.Client(
            headers=headers,
            timeout=_HTTP_TIMEOUT,
        )
        self._queue: Queue[BehavioralEvent] = Queue(maxsize=_QUEUE_MAX)
        self._worker = threading.Thread(target=self._drain_loop, daemon=True, name="thoth-http-emitter")
        self._worker.start()
        atexit.register(self.close)

    def emit(self, event: BehavioralEvent) -> None:
        """Non-blocking enqueue. Emits an error when queue pressure drops events."""
        with self._delivery_lock:
            try:
                self._queue.put_nowait(event)
                self._pending += 1
            except Full:
                self._dropped += 1
                logger.error(
                    "thoth: telemetry queue full, dropping event_id=%s (event dropped)",
                    event.event_id,
                )

    def _drain_loop(self) -> None:
        while not self._closed.is_set() or not self._queue.empty():
            batch = self._collect_batch()
            if batch:
                self._record_result(self._send_batch(batch))

    def _collect_batch(self) -> list[BehavioralEvent]:
        batch: list[BehavioralEvent] = []
        try:
            batch.append(self._queue.get(timeout=_DRAIN_TIMEOUT_S))
            while len(batch) < _BATCH_MAX:
                batch.append(self._queue.get_nowait())
        except Empty:
            pass
        return batch

    def _send_batch(self, events: list[BehavioralEvent]) -> DeliveryBatchStatus:
        payload = [telemetry_event(e) for e in events]
        event_ids = [event.event_id for event in events]
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._http.post(self._endpoint, json=payload)
                response.raise_for_status()
                return DeliveryBatchStatus(event_ids, [], attempt)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {408, 429} or exc.response.status_code >= 500:
                    if attempt < _MAX_ATTEMPTS:
                        time.sleep(_RETRY_DELAY_S * attempt)
                        continue
                response = exc.response
                first_event = events[0] if events else None
                detail = extract_http_error_detail(response)
                hint = auth_failure_hint(response.status_code, detail)
                logger.error(
                    ("thoth: ingest API rejected telemetry (status=%s url=%s tenant_id=%s agent_id=%s event_type=%s attempts=%s)%s"),
                    response.status_code,
                    str(response.request.url),
                    getattr(first_event, "tenant_id", None),
                    getattr(first_event, "agent_id", None),
                    getattr(first_event, "event_type", None),
                    attempt,
                    f" hint={hint}" if hint else "",
                )
                return DeliveryBatchStatus([], event_ids, attempt)
            except Exception:
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(_RETRY_DELAY_S * attempt)
                    continue
                logger.error(
                    "thoth: ingest API unreachable; dropping %d telemetry events after %d attempts",
                    len(events),
                    attempt,
                    exc_info=True,
                )
                return DeliveryBatchStatus([], event_ids, attempt)
        return DeliveryBatchStatus([], event_ids, _MAX_ATTEMPTS)

    def close(self, timeout: float = 5.0) -> DeliveryStatus:
        self._closed.set()
        self._worker.join(max(0.0, timeout))
        if not self._worker.is_alive():
            self._http.close()
        return self.delivery_status()

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
