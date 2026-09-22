"""F1 — one typed, frozen, addressed message bus. Control rides the same bus.

Scope (deliberate): this is the STATE/CONTROL plane, not the token data
plane. Token streams stay on their SSE pipes — publishing every token through
a fan-out bus would tax the hot path for nothing. What travels here:

    job.created / job.status / job.done      lifecycle (published by the
                                             JobStore adapter, see wire below)
    control.cancel                           stop a generation, any transport
    catalog.changed                          on-disk model inventory changed
                                             (storage publishes, engine consumes)
    control.restart / control.* (future)     worker ops, module updates
    worker.* / keeper.* (future)             registries, health, keeper comms

Envelope: BusMessage — frozen, addressed (principal / channel / target /
job_id), serializable (to_dict/from_dict), so an HTTP or SSE relay can carry
it across processes verbatim. In-process today (one bus per process, same
one-gunicorn-process model as the JobStore); remote transports reach it
through thin route adapters that publish/subscribe on their behalf.

Delivery: each Subscription owns a bounded queue. Publishing never blocks —
on overflow the OLDEST message is dropped and the drop is counted loudly on
the subscription (a slow consumer must not stall cancels for everyone else).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# Topic conventions. Match exact ("control.cancel") or prefix-wildcard
# ("job.*" matches job.created, job.status, ...; "*" matches everything).
TOPIC_JOB_CREATED = "job.created"
TOPIC_JOB_STATUS = "job.status"
TOPIC_JOB_DONE = "job.done"          # any terminal state; payload carries which
TOPIC_CONTROL_CANCEL = "control.cancel"
# Catalog invalidation (partition protocol, see EXTRACTION_GUIDE §4): published by
# hugpy_storage after a download/delete changes what is on disk; consumed by
# hugpy_engine to drop its catalog/manifest caches. Payload convention:
# {"model_key": str|None, "reason": "download"|"delete"|"prune"|"reconcile"|...}.
TOPIC_CATALOG_CHANGED = "catalog.changed"


@dataclass(frozen=True)
class BusMessage:
    topic: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.time)
    source: Optional[str] = None      # transport/component that published
    principal: Optional[str] = None   # who caused it (F2 threads through here)
    channel: Optional[str] = None     # conversational context
    target: Optional[str] = None      # addressed recipient; None = broadcast
    job_id: Optional[str] = None
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic, "id": self.id, "ts": self.ts,
            "source": self.source, "principal": self.principal,
            "channel": self.channel, "target": self.target,
            "job_id": self.job_id, "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BusMessage":
        return cls(
            topic=str(d.get("topic") or ""),
            id=str(d.get("id") or uuid.uuid4().hex),
            ts=float(d.get("ts") or time.time()),
            source=d.get("source"), principal=d.get("principal"),
            channel=d.get("channel"), target=d.get("target"),
            job_id=d.get("job_id"), payload=dict(d.get("payload") or {}),
        )


def _topic_matches(pattern: str, topic: str) -> bool:
    if pattern == "*" or pattern == topic:
        return True
    if pattern.endswith(".*"):
        return topic.startswith(pattern[:-1])
    return False


class Subscription:
    """One consumer's mailbox. Iterate it (blocking) or get(timeout=...).
    close() detaches it from the bus and unblocks any pending get."""

    _SENTINEL = object()

    def __init__(self, bus: "Bus", topics: tuple[str, ...],
                 target: Optional[str], maxsize: int) -> None:
        self._bus = bus
        self.topics = topics
        self.target = target
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.closed = False

    def _matches(self, msg: BusMessage) -> bool:
        if self.target is not None and msg.target not in (None, self.target):
            return False
        return any(_topic_matches(p, msg.topic) for p in self.topics)

    def _offer(self, msg: BusMessage) -> None:
        while True:
            try:
                self._q.put_nowait(msg)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                    if self.dropped in (1, 100) or self.dropped % 1000 == 0:
                        logger.warning(
                            "bus subscription %s dropped %d message(s) "
                            "(slow consumer)", self.topics, self.dropped)
                except queue.Empty:
                    pass

    def get(self, timeout: Optional[float] = None) -> Optional[BusMessage]:
        try:
            item = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        return None if item is self._SENTINEL else item

    def __iter__(self) -> Iterator[BusMessage]:
        while not self.closed:
            item = self._q.get()
            if item is self._SENTINEL:
                return
            yield item

    def close(self) -> None:
        self.closed = True
        self._bus._detach(self)
        try:
            self._q.put_nowait(self._SENTINEL)
        except queue.Full:
            pass


class Bus:
    def __init__(self) -> None:
        self._subs: list[Subscription] = []
        self._lock = threading.Lock()

    def subscribe(self, *topics: str, target: Optional[str] = None,
                  maxsize: int = 256) -> Subscription:
        sub = Subscription(self, topics or ("*",), target, maxsize)
        with self._lock:
            self._subs.append(sub)
        return sub

    def _detach(self, sub: Subscription) -> None:
        with self._lock:
            try:
                self._subs.remove(sub)
            except ValueError:
                pass

    def publish(self, topic: Optional[str] = None, *,
                msg: Optional[BusMessage] = None, **fields: Any) -> BusMessage:
        """publish("control.cancel", job_id=...) or publish(msg=BusMessage(...))."""
        if msg is None:
            if not topic:
                raise ValueError("publish() needs a topic or a msg")
            msg = BusMessage(topic=topic, **fields)
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            if not sub.closed and sub._matches(msg):
                sub._offer(msg)
        return msg


bus = Bus()

_WIRED: dict[tuple[int, int], threading.Thread] = {}
_WIRED_LOCK = threading.Lock()


def wire_cancel(the_bus: Optional[Bus] = None, store=None) -> threading.Thread:
    """F1.3 — control messages act through the same substrate everywhere:
    a daemon thread subscribes to control.cancel and calls JobStore.cancel,
    which fires the cancel handle the owning stream attached. Idempotent per
    (bus, store) pair; call it from every process entrypoint that serves jobs
    (flask app factory, worker agent main)."""
    from hugpy_control.jobs import job_store as _default_store
    the_bus = the_bus or bus
    store = store if store is not None else _default_store
    key = (id(the_bus), id(store))
    with _WIRED_LOCK:
        th = _WIRED.get(key)
        if th is not None and th.is_alive():
            return th
        sub = the_bus.subscribe(TOPIC_CONTROL_CANCEL)

        def _run() -> None:
            for m in sub:
                if m.job_id:
                    store.cancel(m.job_id,
                                 reason=str(m.payload.get("reason") or ""))

        th = threading.Thread(target=_run, name="comms-cancel", daemon=True)
        th.start()
        _WIRED[key] = th
        return th


def wire_job_events(the_bus: Optional[Bus] = None, store=None,
                    source: Optional[str] = None) -> None:
    """Publish job lifecycle transitions onto the bus (job.created /
    job.status / job.done) via the store's on_change hook — the store never
    imports the bus, this adapter is the one seam between them."""
    from hugpy_control.jobs import job_store as _default_store, TERMINAL_STATUSES
    the_bus = the_bus or bus
    store = store if store is not None else _default_store

    def _on_change(job, prior: str) -> None:
        status = job.to_dict()["status"]
        if not prior:
            topic = TOPIC_JOB_CREATED
        elif status in TERMINAL_STATUSES:
            topic = TOPIC_JOB_DONE
        else:
            topic = TOPIC_JOB_STATUS
        the_bus.publish(topic, source=source, principal=job.principal,
                        channel=job.channel, target=job.worker,
                        job_id=job.id,
                        payload={"status": status, "prior": prior,
                                 "kind": job.kind, "model_key": job.model_key})

    store.on_change = _on_change
