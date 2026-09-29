"""KNOWN-GOOD CONTRACT — call queue surfaces (JobStore, /llm/queue, /llm/jobs,
/llm/jobs/<id>/cancel) and job bookkeeping for the chat + /v1 intake paths.

Catalogue: docs/KNOWN-GOOD-CORE.md (area "call queue / relay").
Source under test: hugpy_control/jobs.py (JobStore), hugpy_engine/dispatch/
activity.py (the /api/llm/queue snapshot), hugpy_server routes comms_routes.py
(/llm/jobs, cancel) + worker_routes.py (/llm/queue), functions/chat/
streaming.py (_feed_job_from_status, stream_events) and routes/v1_routes.py
(_v1_events).

Deterministic: one JobStore over a tmp SQLite mirror is installed as THE store
for every module; the engine stream is a fake async generator; Flask test
clients drive the routes; media_bus is a no-op read.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import tempfile

import pytest
from flask import Flask

os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-known-good-"))

J = importlib.import_module("hugpy_control.jobs")
from hugpy_control.shared import SqliteMirror  # noqa: E402
activity = importlib.import_module("hugpy_engine.dispatch.activity")
streaming = importlib.import_module("hugpy_server.app.functions.chat.streaming")
v1 = importlib.import_module("hugpy_server.app.routes.v1_routes")
cr = importlib.import_module("hugpy_server.app.routes.comms_routes")
from hugpy_engine.schemas.event_schemas import (  # noqa: E402
    DoneEvent, ErrorEvent, StatusEvent, TokenEvent,
)

RESOLVED = "Qwen~Qwen3-Coder-Next-GGUF"
JOB_ROW_FIELDS = {"id", "status", "stage", "elapsed", "progressed_at", "stalled",
                  "worker", "error"}
QUEUE_ROW_FIELDS = {"request_id", "model_key", "model", "kind", "prompt", "state",
                    "stage", "progress", "input_tokens",   # prefill honesty, 2026-09-29
                    "elapsed", "wait", "tokens"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    """One JobStore over a tmp SQLite mirror, installed everywhere a module
    bound ``job_store`` by name."""
    js = J.JobStore(mirror=SqliteMirror(path=str(tmp_path / "comms.db")))
    monkeypatch.setattr(J, "job_store", js)
    monkeypatch.setattr(activity, "job_store", js)
    monkeypatch.setattr(streaming, "job_store", js)
    monkeypatch.setattr(cr, "audit", lambda *a, **k: None)
    from hugpy_video.intel import media_bus
    monkeypatch.setattr(media_bus, "cancel", lambda jid: {"cancelled": False, "status": None})
    return js


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(cr.comms_bp)
    return app.test_client()


def _collect(agen):
    async def _run():
        out = []
        async for e in agen:
            out.append(e)
        return out
    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# GET /api/llm/queue — concise waiting/active
# ---------------------------------------------------------------------------
def test_queue_snapshot_shape_and_state_transitions(store):
    """INVARIANT: the /api/llm/queue snapshot rows carry exactly
    {request_id, model_key, model, kind, prompt, state, stage, progress,
    input_tokens, elapsed, wait, tokens} (stage/progress/input_tokens added
    2026-09-29 so a processing row can say "prefill 12436/29451");
    state is 'waiting' (pending) → 'processing' (dispatched to a worker,
    pre-token) → 'active' (streaming); counts = {waiting, active, total} with
    processing counted in ``active``; terminal rows leave the live view;
    download jobs are excluded from this (inference) queue.
    Established: F5 unified job store; processing split 2026-09 (activity.py
    'a request assigned to a worker … is processing, not waiting'); frontier
    handoff 2026-09-28 (/api/llm/queue read path)."""
    store.create(RESOLVED, id="r1", kind="chat", prompt="hi")
    store.create("x", id="d1", kind="download")
    snap = activity.snapshot()
    assert [r["request_id"] for r in snap] == ["r1"]
    assert set(snap[0]) == QUEUE_ROW_FIELDS
    assert snap[0]["state"] == "waiting" and snap[0]["model_key"] == RESOLVED
    assert activity.counts() == {"waiting": 1, "active": 0, "total": 1}

    store.begin_dispatch("r1", worker="ae-worker")
    assert activity.snapshot()[0]["state"] == "processing"
    assert activity.counts() == {"waiting": 0, "active": 1, "total": 1}

    activity.on_token("r1")
    assert activity.snapshot()[0]["state"] == "active"
    assert activity.snapshot()[0]["tokens"] == 1

    activity.end("r1")
    assert activity.snapshot() == []
    assert activity.counts() == {"waiting": 0, "active": 0, "total": 0}


def test_llm_queue_route_wraps_active_and_counts(store):
    """INVARIANT: GET /llm/queue returns {"active": [rows], "counts": {...}}.
    Established: console activity view contract (worker_routes.llm_queue)."""
    wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    store.create(RESOLVED, id="q1", kind="chat")
    body = app.test_client().get("/llm/queue").get_json()
    assert set(body) == {"active", "counts"}
    assert body["counts"]["waiting"] == 1
    assert body["active"][0]["request_id"] == "q1"


# ---------------------------------------------------------------------------
# GET /llm/jobs?live=1 — detailed rows
# ---------------------------------------------------------------------------
def test_llm_jobs_live_rows_carry_the_status_fields(store, client, monkeypatch):
    """INVARIANT: /llm/jobs?live=1 rows carry id, status, stage, elapsed,
    progressed_at, stalled, worker, error (plus model_key etc.); ``stalled`` is
    computed fresh from progressed_at for an active row (never a stale stored
    bool); live=0 includes terminal rows; counts is {waiting, active, total}.
    Established: F5 / CON-01 jobs view; honest stall clock (HUGPY_JOB_STALL_
    SECONDS, default 90); frontier handoff 2026-09-28 (/llm/jobs?live=1)."""
    store.create(RESOLVED, id="j1", kind="v1", transport="v1")
    store.begin_dispatch("j1", worker="ae-worker")
    store.update("j1", stage="awaiting-capacity")
    store.create(RESOLVED, id="j2", kind="chat")
    store.finish("j2", error="boom")

    body = client.get("/llm/jobs?live=1").get_json()
    rows = {r["id"]: r for r in body["jobs"]}
    assert set(rows) == {"j1"}
    assert JOB_ROW_FIELDS <= set(rows["j1"])
    assert rows["j1"]["status"] == "processing" and rows["j1"]["worker"] == "ae-worker"
    assert rows["j1"]["stage"] == "awaiting-capacity" and rows["j1"]["stalled"] is False
    assert rows["j1"]["error"] is None and rows["j1"]["model_key"] == RESOLVED
    assert set(body["counts"]) == {"waiting", "active", "total"}

    all_rows = {r["id"]: r for r in client.get("/llm/jobs?live=0").get_json()["jobs"]}
    assert all_rows["j2"]["status"] == "failed"
    assert all_rows["j2"]["error"]["message"] == "boom"

    # Honest stall: an ACTIVE row with old progressed_at reads stalled=True.
    monkeypatch.setenv("HUGPY_JOB_STALL_SECONDS", "10")
    store.get("j1").progressed_at -= 60
    assert client.get("/llm/jobs").get_json()["jobs"][0]["stalled"] is True
    # A pending (queued, not dispatched) row is starved, not stalled.
    store.create(RESOLVED, id="j3", kind="chat")
    store.get("j3").progressed_at -= 60
    rows = {r["id"]: r for r in client.get("/llm/jobs").get_json()["jobs"]}
    assert rows["j3"]["stalled"] is False


# ---------------------------------------------------------------------------
# POST /llm/jobs/<id>/cancel — authoritative, terminal
# ---------------------------------------------------------------------------
def test_cancel_route_relays_to_live_owner_and_ends_terminal(store, client):
    """INVARIANT: POST /llm/jobs/<id>/cancel is the authoritative cancel path.
    With a LIVE owner (a stream attached its cancel handle) the route relays:
    the handle fires exactly once (this is what sets the relay's cancel_event
    → GeneratorExit → worker httpx stream aclose → llama slot busy flag
    clears), cancel_requested is raised, and the owner's teardown finish()
    lands the row terminal 'cancelled' with the dispatch lease cleared. The
    row then leaves the live queue. A second cancel on the terminal row
    reports cancelled:false / mode:noop — it never lies.
    Established: slice 9 defect 1 (authoritative cancel); h24 (2026-09-28:
    /v1 streams attach a live cancellation handle so cancel clears the slot)."""
    fired = {"n": 0}
    store.create(RESOLVED, id="live", kind="v1", transport="v1")
    store.begin_dispatch("live", worker="ae-worker")
    store.attach_cancel("live", lambda: fired.__setitem__("n", fired["n"] + 1))

    body = client.post("/llm/jobs/live/cancel", json={"reason": "operator"}).get_json()
    assert body["cancelled"] is True and body["mode"] == "relayed"
    assert fired["n"] == 1
    job = store.get("live")
    assert job.cancel_requested is True and not job.terminal   # owner tears down
    # The owner's finally (stream_events / _v1_events → finish) marks terminal.
    store.finish("live")
    job = store.get("live")
    assert job.status == "cancelled" and job.terminal
    assert job.dispatch_lease_until is None and job.dispatch_last_seen is None
    assert activity.snapshot() == []

    again = client.post("/llm/jobs/live/cancel", json={}).get_json()
    assert again["cancelled"] is False and again["mode"] == "noop"


def test_cancel_route_force_terminates_owner_less_job_and_noops_unknown(store, client):
    """INVARIANT: an owner-less job (pending, no handle) is force-marked
    terminal 'cancelled' in the store (mode 'store', persisted); an unknown id
    reports cancelled:false (mode 'noop'). Established: slice 9 defect 1."""
    store.create(RESOLVED, id="stuck", kind="chat")
    body = client.post("/llm/jobs/stuck/cancel", json={"reason": "op"}).get_json()
    assert (body["cancelled"], body["mode"], body["status"]) == (True, "store", "cancelled")
    assert store.get("stuck").terminal
    assert client.post("/llm/jobs/nope/cancel", json={}).get_json()["cancelled"] is False


def test_cancel_racing_ahead_of_attach_still_fires(store):
    """INVARIANT: a cancel that arrives BEFORE the stream attaches its handle
    fires the handle at attach time — a cancel is never lost to the race.
    Established: JobStore.attach_cancel contract (F1.3)."""
    fired = {"n": 0}
    store.create(RESOLVED, id="early", kind="chat")
    assert store.cancel("early") is True
    store.attach_cancel("early", lambda: fired.__setitem__("n", fired["n"] + 1))
    assert fired["n"] == 1


# ---------------------------------------------------------------------------
# Status events feed the job row (dispatch lease, stage)
# ---------------------------------------------------------------------------
def test_dispatch_banner_and_hold_status_feed_the_job_row(store):
    """INVARIANT: the relay's dispatch banner (served_by=worker) marks the job
    'processing' with the worker and a dispatch lease at SELECTION time; a
    hold status (awaiting-load / awaiting-capacity) records the stage +
    message and feeds progressed_at (a placed, loading call is never
    orphan-expired). Established: streaming._feed_job_from_status (t36 / 2026-09
    dispatch-lease fix)."""
    store.create(RESOLVED, id="s1", kind="chat")
    streaming._feed_job_from_status("s1", StatusEvent(
        request_id="s1", stage="dispatch", served_by="worker",
        worker_id="w1", worker_name="ae-worker"))
    job = store.get("s1")
    assert job.status == "processing" and job.worker == "ae-worker"
    assert job.dispatch_started_at is not None and job.dispatch_lease_until > job.dispatch_started_at
    before = job.progressed_at
    job.progressed_at -= 30
    streaming._feed_job_from_status("s1", StatusEvent(
        request_id="s1", stage="awaiting-capacity",
        message="waiting for its current request to finish…", worker_state="loaded"))
    job = store.get("s1")
    assert job.stage == "awaiting-capacity" and job.status == "processing"
    assert "waiting for its current request" in job.message
    assert job.progressed_at >= before - 1, "stage change fed the honest stall clock"
    # Never-dispatched pending rows are the only orphan-expiry candidates.
    assert "s1" not in store.expire_pending_orphans()


# ---------------------------------------------------------------------------
# Resolved key propagates into the job record (chat + /v1)
# ---------------------------------------------------------------------------
def _fake_stream_query(events):
    async def _sq(cancel_event=None, **prompt_kwargs):
        _sq.seen = prompt_kwargs
        for e in events:
            yield e
    _sq.seen = None
    return _sq


def test_v1_events_records_resolved_key_and_lifecycle(store, monkeypatch):
    """INVARIANT: /v1 (_v1_events) creates the shared job row under the
    RESOLVED model key it was handed (prompt_kwargs['model_key'] — the intake
    gate rewrote the caller's spelling to resolve_model_key's result), stamps
    the worker from the dispatch banner, flips streaming on the first token,
    persists usage from done, and lands 'done' — so queue, resolver and worker
    identity agree. An error event lands 'failed' with its message.
    Established: frontier handoff 2026-09-28 ('resolved key carried into
    prompt kwargs and job records'); incident 2026-09-25 (v1 rows stuck
    pending)."""
    engine = importlib.import_module("hugpy_engine")
    evs = [StatusEvent(request_id="v1a", stage="dispatch", served_by="worker",
                       worker_id="w1", worker_name="ae-worker"),
           TokenEvent(request_id="v1a", text="W"),
           DoneEvent(request_id="v1a", input_tokens=3, output_chunks=1,
                     finish_reason="stop",
                     usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})]
    sq = _fake_stream_query(evs)
    monkeypatch.setattr(engine, "stream_query", sq)
    kwargs = {"request_id": "v1a", "model_key": RESOLVED, "messages": [{"role": "user", "content": "hi"}]}

    out = _collect(v1._v1_events(dict(kwargs), call_data={"model": "coder-next"}))
    assert [e.type for e in out] == ["status", "token", "done"]
    assert sq.seen["model_key"] == RESOLVED, "the resolved key reaches the engine"
    job = store.get("v1a")
    assert job.model_key == RESOLVED and job.kind == "v1"
    assert job.worker == "ae-worker" and job.status == "done"
    assert job.tokens == 1 and job.total_tokens == 4
    assert job.request == {"model": "coder-next"}

    monkeypatch.setattr(engine, "stream_query", _fake_stream_query(
        [ErrorEvent(request_id="v1b", message="worker_busy: ae-worker has served …")]))
    _collect(v1._v1_events({"request_id": "v1b", "model_key": RESOLVED, "prompt": "x"}))
    job = store.get("v1b")
    assert job.status == "failed" and "worker_busy" in job.error.message


def test_chat_stream_events_registers_job_under_prompt_kwargs_model_key(store, monkeypatch):
    """INVARIANT: the console chat path (stream_events) registers the job under
    prompt_kwargs['model_key'] with kind 'chat', the request echoed, and lands
    it terminal on stream end (finish() runs in the finally, before aclose).
    Established: F5 job store; incident 2026-09-25 ordering (finalize first)."""
    engine = importlib.import_module("hugpy_engine")
    from hugpy_engine.wire.chat_schemas import ChatBody
    sq = _fake_stream_query([TokenEvent(request_id="c1", text="a"),
                             DoneEvent(request_id="c1", input_tokens=1, output_chunks=1,
                                       finish_reason="stop")])
    monkeypatch.setattr(engine, "stream_query", sq)
    body = ChatBody(model_key=RESOLVED, prompt="hi", request_id="c1", max_new_tokens=8,
                    transport="web")
    frames = _collect(streaming.stream_events(body))
    assert frames and b"data:" in frames[0]
    assert sq.seen["model_key"] == RESOLVED and sq.seen["request_id"] == "c1"
    job = store.get("c1")
    assert job.model_key == RESOLVED and job.kind == "chat" and job.transport == "web"
    assert job.status == "done" and job.tokens == 1
    assert job.request["model_key"] == RESOLVED


def test_v1_intake_rewrites_prompt_kwargs_to_the_resolved_key():
    """GUARD (source pin): the /v1 route resolves the caller's model spelling
    through resolve_model_key at intake and writes the RESULT back into
    prompt_kwargs['model_key'] before the job is created, rejecting an
    unresolvable key with a 400 (ValueError/KeyError caught).
    Established: frontier handoff 2026-09-28; slice 9 defect 3."""
    src = inspect.getsource(v1.v1_chat_completions)
    assert 'prompt_kwargs["model_key"] = _resolved_model_key' in src
    # Intake resolves via resolve_model_key(model_key=_mk, …); the explicit
    # model_format pin (operator 2026-09-29) rides alongside the key so intake
    # lands on the requested representation.
    assert "_resolved_model_key = resolve_model_key(" in src
    assert "model_key=_mk" in src
    assert "except (KeyError, ValueError)" in src
