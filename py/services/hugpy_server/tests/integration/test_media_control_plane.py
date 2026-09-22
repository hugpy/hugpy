"""Slice B — media jobs as first-class citizens of the /llm control plane.

Locks the THREE additive changes and, above all, PROVES chat/download behavior
is unchanged:

  B1  cancel fan-out    POST /llm/jobs/<id>/cancel reaches media_bus.cancel for a
                        media job, and stays purely comms-side for chat.
  B2  terminal_rows     SqliteMirror.terminal_rows(kinds) is media-gated; a
                        sibling process sees a finished MEDIA job but NEVER a
                        finished chat job (chat terminal stays local-retention).
  B3  JobError.retryable  every /llm/jobs error object carries "retryable"
                        (null for chat/download, real bool for media), coerce()
                        round-trips it, and the bridge preserves it.
"""
from __future__ import annotations

import os

import pytest

from hugpy_control.jobs import Job, JobError, JobStore, MEDIA_KINDS, job_store
from hugpy_control.shared import SqliteMirror
from hugpy_video.intel import job_bridge
from hugpy_video.intel import media_bus as _media_bus
from hugpy_video.intel import result_schema as _rs


# ---------------------------------------------------------------------------
# B2 — SqliteMirror.terminal_rows is media-gated (READ-side).
# ---------------------------------------------------------------------------
def test_terminal_rows_is_media_gated(tmp_path):
    m = SqliteMirror(os.path.join(tmp_path, "comms.db"))
    m.upsert(Job(id="t-med", kind="frame_extract", status="done",
                 transport="media").to_dict())
    m.upsert(Job(id="t-chat", kind="chat", status="done").to_dict())
    m.upsert(Job(id="t-med-live", kind="crop", status="processing",
                 transport="media").to_dict())          # live -> must be excluded
    trows = {d["id"]: d for d in m.terminal_rows(MEDIA_KINDS)}
    assert "t-med" in trows, "terminal_rows returns a finished media-kind row"
    assert "t-chat" not in trows, "terminal_rows EXCLUDES a finished chat-kind row"
    assert "t-med-live" not in trows, "terminal_rows EXCLUDES a live media row"
    assert m.terminal_rows(()) == [], "terminal_rows(()) is empty (never a full terminal scan)"
    assert "t-med" not in {d["id"] for d in m.live_rows()}, \
        "live_rows() still excludes terminal (unchanged)"


# ---------------------------------------------------------------------------
# B2 — snapshot(live_only=False) merges a SIBLING's terminal MEDIA row, and
# leaves chat terminal cross-process behavior UNCHANGED.
# ---------------------------------------------------------------------------
def test_snapshot_merges_sibling_terminal_media_only(tmp_path):
    db2 = os.path.join(tmp_path, "comms.db")
    A = JobStore(mirror=SqliteMirror(db2))   # "worker A" — actually runs the jobs
    B = JobStore(mirror=SqliteMirror(db2))   # "worker B" — a sibling, ran nothing

    A.create("scene", id="m-1", kind="generate_scene", transport="media")
    A.finish("m-1")                          # terminal + mirrored
    A.create("m", id="c-1", kind="chat", transport="web")
    A.finish("c-1")                          # terminal + mirrored

    bsnap_full = {d["id"] for d in B.snapshot(live_only=False)}
    assert "m-1" in bsnap_full, "B (sibling) SEES the finished media job via the new merge"
    assert "c-1" not in bsnap_full, "B (sibling) does NOT see the finished chat job (chat UNCHANGED)"

    bsnap_live = {d["id"] for d in B.snapshot(live_only=True)}
    assert "m-1" not in bsnap_live, "B live view excludes the terminal media row"

    asnap_full = {d["id"] for d in A.snapshot(live_only=False)}
    assert "c-1" in asnap_full, "A (owner) still sees its chat terminal via LOCAL retention"
    assert "m-1" in asnap_full, "A (owner) sees its media terminal locally too"

    # kind filter is honored by the terminal merge too: filtering to chat surfaces
    # no media terminal row on the sibling.
    bsnap_chat = {d["id"] for d in B.snapshot(kinds={"chat"}, live_only=False)}
    assert "m-1" not in bsnap_chat and "c-1" not in bsnap_chat


# ---------------------------------------------------------------------------
# B3 — JobError.retryable: emitted ALWAYS (nullable), coerce round-trips it,
# detail still works.
# ---------------------------------------------------------------------------
def test_joberror_retryable_is_nullable_and_round_trips():
    d_true = JobError(code="oom", message="no vram", retryable=True).to_dict()
    assert d_true.get("retryable") is True
    d_def = JobError(code="e", message="boom").to_dict()
    assert "retryable" in d_def and d_def["retryable"] is None, \
        "default JobError serializes retryable: null"
    c_false = JobError.coerce({"code": "x", "message": "y", "retryable": False})
    assert c_false.retryable is False and c_false.to_dict()["retryable"] is False
    d_detail = JobError(code="e", message="m", detail={"k": 1}).to_dict()
    assert d_detail.get("detail") == {"k": 1} and d_detail["retryable"] is None
    c_detail = JobError.coerce({"code": "e", "detail": {"z": 2}})
    assert c_detail.detail == {"z": 2}, "coerce() still preserves detail"


# ---------------------------------------------------------------------------
# B3 — bridge preserves retryable end to end (result_schema.JobError -> dict ->
# comms.JobError via coerce -> to_dict emits the real bool). _error_dict reads
# result.error, so the realistic shape is a JobResult wrapping the JobError —
# exactly what media_bus.run_claimed hands to on_terminal.
# ---------------------------------------------------------------------------
def test_bridge_preserves_retryable_end_to_end():
    res = _rs.JobResult(job_id="j", ok=False,
                        error=_rs.JobError(code="oom", message="no vram",
                                           retryable=True))
    ed = job_bridge._error_dict(res)
    assert ed is not None and ed.get("retryable") is True and ed.get("code") == "oom"
    res_f = _rs.JobResult(job_id="j", ok=False,
                          error=_rs.JobError(code="bad", message="x",
                                             retryable=False))
    assert job_bridge._error_dict(res_f).get("retryable") is False
    assert JobError.coerce(ed).to_dict()["retryable"] is True, \
        "bridge dict -> comms.JobError.coerce -> to_dict keeps the real bool"
    assert job_bridge._error_dict({"error": {"code": "x", "retryable": False}}) \
        .get("retryable") is False, "_error_dict tolerates a plain-dict result too"


# ---------------------------------------------------------------------------
# B1 — the real flask app builds and the new cancel rule is registered.
# ---------------------------------------------------------------------------
def test_cancel_rule_registered(server_app):
    _rules = {r.rule: r.methods for r in server_app.url_map.iter_rules()}
    assert "/llm/jobs/<job_id>/cancel" in _rules
    assert "POST" in _rules["/llm/jobs/<job_id>/cancel"]


# ---------------------------------------------------------------------------
# B1 — cancel fan-out semantics over the live app (test client).
# ---------------------------------------------------------------------------
@pytest.fixture
def media_cancel_spy(monkeypatch):
    """media_bus.cancel is patched with a smart spy for ALL cases: it returns a
    media result only for known media ids and a safe no-op otherwise, mirroring
    the real media_bus.cancel (which no-ops on a non-media id). The route calls
    it UNCONDITIONALLY so a queued media job with no local record still fans
    out; every case records the call, and chat/unknown just get the no-op."""
    calls = []
    media_ids = {"mc-1", "qm-1"}

    def _spy(jid):
        calls.append(jid)
        if jid in media_ids:
            return {"job_id": jid, "status": "cancelling", "cancelled": True}
        return {"job_id": jid, "status": None, "cancelled": False}

    monkeypatch.setattr(_media_bus, "cancel", _spy)
    return calls


def test_chat_cancel_stays_comms_side(client, media_cancel_spy):
    # chat job: comms-side cancel fires the local handle; media_bus is called
    # but no-ops, so the response stays transport=web / cancelled=true.
    fired = []
    job_store.create("m", id="nc-1", kind="chat", transport="web")
    job_store.attach_cancel("nc-1", lambda: fired.append(1))
    try:
        rb = client.post("/llm/jobs/nc-1/cancel", json={"reason": "user stop"}).get_json()
        assert rb.get("cancelled") is True and rb.get("transport") == "web"
        assert fired == [1], "chat cancel actually fired the local handle"
    finally:
        job_store.finish("nc-1")


def test_unknown_id_cancel_is_honest(client, media_cancel_spy):
    # unknown id: nothing anywhere -> cancelled false, null status/transport.
    ru = client.post("/llm/jobs/nope/cancel").get_json()
    assert ru.get("cancelled") is False
    assert ru.get("status") is None and ru.get("transport") is None


def test_running_media_cancel_fans_out_and_merges_planes(client, media_cancel_spy):
    # running media job (has a local transport=media record): fan-out reaches
    # media_bus.cancel and merges both planes.
    job_store.create("wan", id="mc-1", kind="studio_i2v", transport="media",
                     status="processing")
    try:
        rm = client.post("/llm/jobs/mc-1/cancel", json={}).get_json()
        assert "mc-1" in media_cancel_spy, "media cancel fanned out to media_bus.cancel(job_id)"
        assert rm.get("cancelled") is True and rm.get("status") == "cancelling"
        assert rm.get("transport") == "media"
    finally:
        job_store.finish("mc-1")


def test_queued_media_without_local_record_still_fans_out(client, media_cancel_spy):
    # QUEUED media job with NO local JobStore record (on_enqueue is mirror-only):
    # job_store.get -> None, so transport can't be read locally. The unconditional
    # media_bus.cancel still fans out, and the route infers transport=media from it.
    rq = client.post("/llm/jobs/qm-1/cancel", json={}).get_json()
    assert "qm-1" in media_cancel_spy
    assert rq.get("cancelled") is True and rq.get("status") == "cancelling"
    assert rq.get("transport") == "media"
