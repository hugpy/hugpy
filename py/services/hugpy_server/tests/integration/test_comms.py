"""Sanity: hugpy_control.jobs + hugpy_control.bus behavior, incl. the DISC-03 races."""
import threading

from hugpy_control.bus import (
    Bus, BusMessage, wire_cancel, wire_job_events, TOPIC_CONTROL_CANCEL,
)
from hugpy_control.jobs import JobStore, normalize_status


# --- status normalization / legacy aliases ---
def test_status_normalization_and_legacy_aliases():
    assert normalize_status("queued") == "pending"
    assert normalize_status("running") == "processing"
    assert normalize_status("completed") == "done"
    assert normalize_status("wat") == "pending", "garbage -> pending"
    assert normalize_status("streaming") == "streaming", "canonical passthrough"


# --- old download-caller API shape ---
def test_download_caller_api_shape():
    s = JobStore()
    j = s.create("qwen-7b")   # old positional signature
    assert j.model_key == "qwen-7b" and j.status == "pending"
    s.update(j.id, status="running", progress=0.5)   # old status name
    assert s.get(j.id).to_dict()["status"] == "processing", "legacy status write normalized"
    d = s.get(j.id).to_dict()
    for k in ("progress", "total_bytes", "attempt", "stalled", "created_at"):
        assert k in d, k


# --- chat lifecycle with request_id as job id ---
def test_chat_lifecycle_with_request_id_as_job_id():
    s = JobStore()
    j2 = s.create("m", id="rid-1", kind="chat", transport="web", channel="sess-9")
    assert j2.id == "rid-1", "job id is request id"
    s.on_output("rid-1")
    s.on_output("rid-1")
    snap = s.snapshot()
    mine = [x for x in snap if x["id"] == "rid-1"][0]
    assert mine["status"] == "streaming", "streaming after first output"
    assert mine["tokens"] == 2
    assert mine["wait"] <= mine["elapsed"]
    s.finish("rid-1")
    assert s.get("rid-1").to_dict()["status"] == "done"
    assert not [x for x in s.snapshot() if x["id"] == "rid-1"], \
        "terminal excluded from live snapshot"


# --- cancel: handle fires, teardown converts, first-terminal-wins ---
def test_cancel_handle_fires_and_teardown_converts():
    s = JobStore()
    fired = []
    s.create("m", id="rid-2")
    s.attach_cancel("rid-2", lambda: fired.append(1))
    assert s.cancel("rid-2", reason="user stop"), "cancel returns True on live job"
    assert fired == [1], "handle fired once"
    assert s.get("rid-2").to_dict()["status"] == "pending", "not force-marked cancelled"
    s.finish("rid-2")   # stream teardown, no explicit status
    assert s.get("rid-2").to_dict()["status"] == "cancelled", "teardown converts to cancelled"


def test_first_terminal_wins():
    s = JobStore()
    s.create("m", id="rid-3")
    s.finish("rid-3")                       # finished just before cancel arrives
    assert not s.cancel("rid-3"), "cancel on finished job is False"
    assert s.get("rid-3").to_dict()["status"] == "done", "stays done (first terminal wins)"
    s.update("rid-3", status="failed")      # late overwrite attempt
    assert s.get("rid-3").to_dict()["status"] == "done", "terminal not overwritten"


def test_cancel_before_attach_fires_on_attach():
    s = JobStore()
    s.create("m", id="rid-4")
    s.cancel("rid-4")
    late = []
    s.attach_cancel("rid-4", lambda: late.append(1))
    assert late == [1], "late-attached handle fires immediately"


# --- error-as-data ---
def test_error_as_data():
    s = JobStore()
    s.create("m", id="rid-5")
    s.finish("rid-5", error=RuntimeError("boom"))
    d6 = s.get("rid-5").to_dict()
    assert d6["status"] == "failed"
    assert d6["error"]["code"] == "RuntimeError"
    assert d6["error"]["message"] == "boom"


# --- bus: topics, wildcard, addressing, serialization ---
def test_bus_topics_wildcard_addressing_serialization():
    b = Bus()
    all_sub = b.subscribe("job.*")
    addr_sub = b.subscribe("control.cancel", target="worker-a")
    b.publish("job.created", job_id="x", payload={"kind": "chat"})
    m = all_sub.get(timeout=1)
    assert m is not None and m.topic == "job.created", "wildcard receives"
    b.publish("control.cancel", job_id="y", target="worker-b")
    assert addr_sub.get(timeout=0.1) is None, "addressed msg not delivered to other target"
    b.publish("control.cancel", job_id="z", target="worker-a")
    assert addr_sub.get(timeout=1).job_id == "z", "addressed msg delivered to its target"
    b.publish("control.cancel", job_id="w")   # broadcast reaches targeted sub too
    assert addr_sub.get(timeout=1).job_id == "w", "broadcast reaches targeted sub"
    rt = BusMessage.from_dict(m.to_dict())
    assert rt.topic == m.topic and rt.job_id == m.job_id, "envelope round-trips"


# --- wire_cancel: control message -> store.cancel -> handle fires ---
def test_wire_cancel_control_message_fires_handle():
    s2 = JobStore()
    b2 = Bus()
    th = wire_cancel(b2, s2)
    assert wire_cancel(b2, s2) is th, "wire_cancel idempotent"
    fired2 = threading.Event()
    s2.create("m", id="rid-6")
    s2.attach_cancel("rid-6", fired2.set)
    b2.publish(TOPIC_CONTROL_CANCEL, job_id="rid-6", payload={"reason": "stop"})
    assert fired2.wait(timeout=2), "bus cancel fires handle"
    assert s2.get("rid-6").cancel_requested


# --- wire_job_events: store transitions publish on the bus ---
def test_wire_job_events_publishes_transitions():
    s3 = JobStore()
    b3 = Bus()
    wire_job_events(b3, s3, source="test")
    ev_sub = b3.subscribe("job.*")
    s3.create("m", id="rid-7", kind="chat")
    s3.on_output("rid-7")
    s3.finish("rid-7")
    topics = [ev_sub.get(timeout=1).topic for _ in range(3)]
    assert topics == ["job.created", "job.status", "job.done"]


# --- retention: terminal jobs pruned ---
def test_terminal_retention_pruned():
    s4 = JobStore(retain_terminal=2, retain_secs=9999)
    for i in range(6):
        s4.create("m", id=f"t-{i}")
        s4.finish(f"t-{i}")
    s4.create("m", id="live")
    terminals = [j for j in s4.all() if j.terminal]
    assert len(terminals) <= 3, "terminal overflow pruned"  # 2 retained + latest batch edge


# --- resurrection: download retry reuses the job id (cancelled -> running) ---
def test_download_retry_resurrection():
    s5 = JobStore()
    s5.create("m", id="r-1", kind="download")
    s5.update("r-1", status="cancelled")
    s5.update("r-1", status="running")   # retry_download path
    assert s5.get("r-1").to_dict()["status"] == "processing", \
        "terminal->live resurrection allowed"
    assert not s5.get("r-1").cancel_requested, "resurrection resets cancel_requested"
    assert s5.get("r-1").ended_ts is None, "resurrection clears ended_ts"
    s5.update("r-1", status="completed")
    assert s5.get("r-1").to_dict()["status"] == "done", "resurrected job can re-terminate"
    assert s5.get("r-1").to_legacy_dict()["status"] == "completed", "legacy wire mapping"
