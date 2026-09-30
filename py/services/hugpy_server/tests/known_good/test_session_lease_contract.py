"""KNOWN-GOOD CONTRACT — client-declared session state (SESSION-LEASE-20260929).

Source under test: hugpy_control/sessions.py (table, lease, sweep), the
calllog start/end hooks that bind jobs to sessions, JobStore's wedge/orphan
expiry guard, and the hugpy_server routes GET /llm/sessions[/<sid>],
POST /llm/sessions/<sid>/lease, /llm/jobs/<id>/cancel + /llm/chat/cancel/<id>
(client request id resolution) and GET /llm/queue (quiet sessions hidden).

Deterministic: tmp SQLite for both the job mirror and the session table, an
explicit clock, Flask test clients; media_bus is a no-op read.
"""
from __future__ import annotations

import importlib
import os
import tempfile
import time

import pytest
from flask import Flask

os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-known-good-"))

J = importlib.import_module("hugpy_control.jobs")
S = importlib.import_module("hugpy_control.sessions")
calllog = importlib.import_module("hugpy_control.calllog")
from hugpy_control.shared import SqliteMirror  # noqa: E402
activity = importlib.import_module("hugpy_engine.dispatch.activity")
cr = importlib.import_module("hugpy_server.app.routes.comms_routes")
wr = importlib.import_module("hugpy_server.app.routes.worker_routes")

SESSION_FIELDS = {"session_id", "client_process", "pid", "user", "host", "platform",
                  "client", "state", "last_lease_at", "ttl", "current_turn",
                  "lease_requests", "last_error", "last_event", "created_at",
                  "updated_at", "lease_age_s", "lease_fresh", "in_flight"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    js = J.JobStore(mirror=SqliteMirror(path=str(tmp_path / "comms.db")))
    monkeypatch.setattr(J, "job_store", js)
    monkeypatch.setattr(activity, "job_store", js)
    monkeypatch.setattr(cr, "audit", lambda *a, **k: None)
    monkeypatch.setenv("HUGPY_SESSIONS_DB", str(tmp_path / "sessions.db"))
    monkeypatch.setenv("HUGPY_CALL_LOG", str(tmp_path / "calls.jsonl"))
    from hugpy_video.intel import media_bus
    monkeypatch.setattr(media_bus, "cancel", lambda jid: {"cancelled": False, "status": None})
    monkeypatch.setattr(wr, "list_workers", lambda: [])
    return js


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(cr.comms_bp)
    app.register_blueprint(wr.worker_bp)
    return app.test_client()


HDRS = {"X-Hugpy-Client-Session": "ha-s1", "X-Hugpy-Client-Turn": "turn-1",
        "X-Hugpy-Client-Request": "req-1", "X-Hugpy-Client-Process": "hugpy-agent:test",
        "X-Hugpy-Client-Pid": "4242", "X-Hugpy-Client-User": "op"}


def _job(store, jid, headers=HDRS):
    """Create a chat job inside a request context carrying the client headers —
    exactly how intake creates one (calllog start row binds the session)."""
    app = Flask(__name__)
    with app.test_request_context("/v1/chat/completions", method="POST", headers=headers):
        store.create("m", id=jid, kind="chat")
    return store.get(jid)


def _lease(client, sid="ha-s1", **body):
    body.setdefault("state", "waiting")
    body.setdefault("event", "lease")
    body.setdefault("ttl", 30)
    return client.post(f"/llm/sessions/{sid}/lease", json=body)


def test_job_start_binds_session_from_the_existing_header_names(store, client):
    """INVARIANT: a job created under X-Hugpy-Client-Session/Turn/Request is
    bound to that session; GET /llm/sessions/<sid> lists it in flight with the
    client's request id, and the row carries the declared process/pid/user.
    Established: SESSION-LEASE-20260929 (operator: sessions go pending/stale)."""
    _job(store, "v1-a")
    body = client.get("/llm/sessions/ha-s1").get_json()
    assert SESSION_FIELDS <= set(body)
    assert body["state"] == "active"
    assert body["client_process"] == "hugpy-agent:test" and body["pid"] == "4242"
    assert [(j["job_id"], j["client_request"], j["turn_id"]) for j in body["in_flight"]] \
        == [("v1-a", "req-1", "turn-1")]
    assert body["lease_fresh"] is False          # bound, but never leased


def test_get_sessions_list_shape_and_unknown_404(store, client):
    """INVARIANT: GET /llm/sessions -> {sessions:[row], count, states};
    GET /llm/sessions/<unknown> -> 404. Established: SESSION-LEASE-20260929."""
    _lease(client)
    body = client.get("/llm/sessions").get_json()
    assert set(body) == {"sessions", "count", "states"}
    assert body["count"] == 1 and SESSION_FIELDS <= set(body["sessions"][0])
    assert body["states"] == ["active", "waiting", "idle", "abandoned", "closed"]
    assert client.get("/llm/sessions/nope").status_code == 404


def test_lease_expiry_abandons_and_cancels_through_the_authoritative_path(store, client, monkeypatch):
    """INVARIANT: when a session's lease lapses (last_lease_at + ttl < now while
    active/waiting) the sweep marks it abandoned with last_error and cancels its
    live jobs via JobStore.cancel_authoritative. Established: SESSION-LEASE-20260929."""
    _job(store, "v1-b")
    t0 = time.time()
    assert S.lease("ha-s1", {"state": "waiting", "ttl": 30, "request_ids": ["req-1"]},
                   now=t0)["ok"]
    seen = []
    real = store.cancel_authoritative
    monkeypatch.setattr(store, "cancel_authoritative",
                        lambda jid, reason="": seen.append((jid, reason)) or real(jid, reason))
    assert S.sweep(now=t0 + 10) == []                      # fresh: untouched
    assert S.sweep(now=t0 + 31) == ["v1-b"]
    assert seen and seen[0][0] == "v1-b" and "lease lapsed" in seen[0][1]
    assert J.normalize_status(store.get("v1-b").status) == "cancelled"
    row = S.get("ha-s1")
    assert row["state"] == "abandoned" and "lease lapsed" in row["last_error"]


def test_session_that_never_leased_is_never_abandoned(store):
    """INVARIANT: absence of a lease is not evidence of death — an old client
    that sends identity headers but no lease is never abandoned/cancelled.
    Established: SESSION-LEASE-20260929 (old clients must be unaffected)."""
    _job(store, "v1-c")
    assert S.sweep(now=time.time() + 10_000) == []
    assert J.normalize_status(store.get("v1-c").status) == "pending"
    assert S.get("ha-s1")["state"] == "active"


def test_fresh_lease_protects_a_long_prefill_from_the_wedge_expiry(store, monkeypatch):
    """INVARIANT: a processing job with no forward progress past the wedge
    cutoff is NOT retired while its session's lease is fresh (long prefill);
    the same job without a fresh lease is retired as before.
    Established: SESSION-LEASE-20260929."""
    monkeypatch.setenv("HUGPY_JOB_STALLED_EXPIRY_SECONDS", "5")
    _job(store, "v1-d")
    _job(store, "v1-e", headers={**HDRS, "X-Hugpy-Client-Session": "ha-old",
                                 "X-Hugpy-Client-Request": "req-e"})
    for jid in ("v1-d", "v1-e"):
        store.update(jid, status="processing", worker="w1")
        store.get(jid).progressed_at = time.time() - 60      # silent prefill
    S.lease("ha-s1", {"state": "waiting", "ttl": 30, "request_ids": ["req-1"]})
    expired = store.expire_pending_orphans()
    assert "v1-d" not in expired and J.normalize_status(store.get("v1-d").status) == "processing"
    assert "v1-e" in expired


def test_turn_done_idle_cancels_orphans_and_hides_them_from_the_queue(store, client):
    """INVARIANT: turn_done/idle (and session_closed) cancel the session's live
    jobs the client no longer lists; GET /llm/queue never shows a job of an
    idle/closed/abandoned session as waiting, and counts agree with the rows.
    Established: SESSION-LEASE-20260929."""
    _job(store, "v1-f")
    _job(store, "v1-g", headers={**HDRS, "X-Hugpy-Client-Session": "ha-other",
                                 "X-Hugpy-Client-Request": "req-g"})
    q = client.get("/llm/queue").get_json()
    assert {r["request_id"] for r in q["active"]} == {"v1-f", "v1-g"}
    r = _lease(client, state="idle", event="turn_done", turn_id="turn-1", request_ids=[])
    assert r.status_code == 200 and r.get_json()["cancelled"] == ["v1-f"]
    q = client.get("/llm/queue").get_json()
    assert set(q) == {"active", "counts"}
    assert {r["request_id"] for r in q["active"]} == {"v1-g"}
    assert q["counts"] == {"waiting": 1, "active": 0, "total": 1}
    r = _lease(client, sid="ha-other", state="closed", event="session_closed")
    assert r.get_json()["session"]["state"] == "closed"
    assert client.get("/llm/queue").get_json()["active"] == []


def test_quiet_session_job_is_hidden_even_if_its_cancel_did_not_land(store, client, monkeypatch):
    """INVARIANT: the queue hides a live job whose session is quiet even when
    the cancel could not retire it (e.g. a sibling owns it and has not noticed
    yet). Established: SESSION-LEASE-20260929."""
    _job(store, "v1-h")
    monkeypatch.setattr(store, "cancel_authoritative",
                        lambda jid, reason="": {"cancelled": False, "mode": "noop"})
    _lease(client, state="closed", event="session_closed")
    q = client.get("/llm/queue").get_json()
    assert q["active"] == [] and q["counts"]["total"] == 0


def test_fresh_listed_request_survives_turn_done_of_another_turn(store, client):
    """INVARIANT: a turn_done that still lists an in-flight request id (state
    waiting) cancels nothing the client listed. Established: SESSION-LEASE-20260929."""
    _job(store, "v1-i")
    r = _lease(client, state="waiting", event="turn_done", turn_id="turn-1",
               request_ids=["req-1"]).get_json()
    assert r["cancelled"] == []
    assert J.normalize_status(store.get("v1-i").status) == "pending"


def test_cancel_by_client_request_id_resolves_to_the_job(store, client):
    """INVARIANT: POST /llm/jobs/<client request id>/cancel and
    /llm/chat/cancel/<client request id> resolve the client's id to the job
    central minted (v1-...), so a client abort and the Calls panel's cancel
    (which sends client_request first) hit the real job.
    Established: SESSION-LEASE-20260929."""
    _job(store, "v1-j")
    body = client.post("/llm/jobs/req-1/cancel", json={"reason": "user abort"}).get_json()
    assert body["job_id"] == "v1-j" and body["cancelled"] is True
    _job(store, "v1-k", headers={**HDRS, "X-Hugpy-Client-Request": "req-k"})
    body = client.post("/llm/chat/cancel/req-k").get_json()
    assert body["cancelled"] is True and body["request_id"] == "v1-k"
    # unknown ids stay an honest noop
    assert client.post("/llm/jobs/zzz/cancel").get_json()["cancelled"] is False


def test_lease_body_validation_and_retention(store, client):
    """INVARIANT: an unknown state degrades to active, ttl is clamped to
    [5, 600]; closed/abandoned/idle rows older than 24 h are pruned.
    Established: SESSION-LEASE-20260929."""
    row = _lease(client, state="bogus", ttl=99999).get_json()["session"]
    assert row["state"] == "active" and row["ttl"] == 600
    t0 = time.time()
    S.lease("ha-old", {"state": "closed", "event": "session_closed"}, now=t0 - 90_000)
    S.sweep(now=t0)
    assert S.get("ha-old") is None and S.get("ha-s1") is not None


def test_calls_rows_carry_session_state(store, client):
    """INVARIANT: /llm/calls rows with a client_session carry
    client_session_state (+ lease age) from the session table.
    Established: SESSION-LEASE-20260929."""
    _job(store, "v1-l")
    _lease(client, state="waiting", request_ids=["req-1"])
    rows = client.get("/llm/calls").get_json()["calls"]
    row = next(r for r in rows if r["id"] == "v1-l")
    assert row["client_session_state"] == "waiting"
    assert row["client_session_lease_fresh"] is True


def test_sessions_off_switch_is_harmless(store, client, monkeypatch):
    """INVARIANT: HUGPY_SESSIONS_DB=off disables the table without breaking
    intake, queue or cancel. Established: SESSION-LEASE-20260929."""
    monkeypatch.setenv("HUGPY_SESSIONS_DB", "off")
    _job(store, "v1-m")
    assert client.get("/llm/sessions").get_json()["sessions"] == []
    assert _lease(client).status_code == 503
    assert [r["request_id"] for r in client.get("/llm/queue").get_json()["active"]] == ["v1-m"]
