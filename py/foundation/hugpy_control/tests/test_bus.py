"""The control bus: delivery, wildcards, addressing, overflow, and the topics
other packages agree on (TOPIC_CATALOG_CHANGED is published by storage and
consumed by engine)."""
from __future__ import annotations

from hugpy_control import bus as busmod
from hugpy_control.bus import Bus, BusMessage, TOPIC_CATALOG_CHANGED, TOPIC_JOB_CREATED
from hugpy_control.jobs import JobStore


def test_catalog_changed_topic_is_the_agreed_contract():
    assert TOPIC_CATALOG_CHANGED == "catalog.changed"
    b = Bus()
    sub = b.subscribe("catalog.*")
    b.publish(TOPIC_CATALOG_CHANGED, source="storage", payload={"model_key": "k", "reason": "download"})
    m = sub.get(timeout=1)
    assert m is not None and m.topic == TOPIC_CATALOG_CHANGED and m.payload["model_key"] == "k"


def test_exact_wildcard_and_target_matching():
    b = Bus()
    everything = b.subscribe()
    jobs = b.subscribe("job.*")
    mine = b.subscribe("control.cancel", target="w1")
    b.publish("job.status", job_id="j")
    b.publish("control.cancel", target="w2", job_id="j")
    b.publish("control.cancel", target="w1", job_id="j")
    b.publish("control.cancel", job_id="j")                     # broadcast reaches w1 too
    assert [m.topic for m in (everything.get(0.1) for _ in range(4))] == \
        ["job.status", "control.cancel", "control.cancel", "control.cancel"]
    assert jobs.get(0.1).topic == "job.status" and jobs.get(0.1) is None
    assert [m.target for m in (mine.get(0.1), mine.get(0.1))] == ["w1", None]
    assert mine.get(0.1) is None


def test_envelope_roundtrip_and_overflow_drops_oldest():
    m = BusMessage(topic="t", principal="p", payload={"a": 1})
    assert BusMessage.from_dict(m.to_dict()) == m
    b = Bus()
    sub = b.subscribe("t", maxsize=2)
    for i in range(5):
        b.publish("t", payload={"i": i})
    assert sub.dropped == 3
    assert [sub.get(0.1).payload["i"] for _ in range(2)] == [3, 4]
    sub.close()
    assert sub.get(0.1) is None and sub.closed


def test_wire_job_events_publishes_lifecycle_on_a_bare_store():
    b = Bus()
    store = JobStore()                     # in-process only, no mirror
    sub = b.subscribe("job.*")
    busmod.wire_job_events(b, store, source="test")
    store.create("m", id="j-1", kind="chat")
    m = sub.get(timeout=1)
    assert m is not None and m.topic == TOPIC_JOB_CREATED and m.job_id == "j-1"
    assert m.source == "test" and m.payload["kind"] == "chat"
