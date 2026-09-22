"""Slice 9 — authoritative honest cancel, pending-orphan expiry, reject-at-intake.

Field incident: job v1-b065322dab9942df87ab93f0853018ab (model
ponpoke/flux2-klein-9b-uncensored-text-encoder — a key in NO catalog) sat pending
with worker:null and zero progress for 14+ hours, surviving every restart, while
POST cancel returned {"cancelled": true} TWICE and the job lived on. Three fixes:

  1. Cancel is authoritative + honest. Dead-owner (pending/worker-null, or a
     mirror-only immortal row) -> force terminal `cancelled` in the STORE
     (persisted, survives reload), mode="store". Live owner -> relay, mode=
     "relayed". Nothing anywhere -> cancelled:false, mode="noop" (no lying).
  2. Pending-orphan expiry: a never-dispatched pending job past the threshold ->
     terminal `expired`, aged on progressed_at (NEVER `updated` — no view-driven
     resurrection). Event-driven (jobs-view / heartbeat), no timer.
  3. Reject at intake: a v1 submit naming an unresolvable model -> 4xx + known
     hint (dispatcher's own resolve_model_key); a known-but-absent model queues.

Run: venv/bin/python -m pytest tests/test_authoritative_cancel_and_orphan_expiry.py -q
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-slice9-"))

from hugpy_control import jobs as J
from hugpy_control.shared import SqliteMirror


def _store(tmp_path):
    db = str(tmp_path / "comms.db")
    return J.JobStore(mirror=SqliteMirror(path=db)), db


# ═══════════════ Defect 1 — authoritative, honest cancel ════════════════════
def test_dead_owner_cancel_marks_store_terminal_and_survives_reload(tmp_path):
    """THE operator's exact shape: pending, worker null, no attached handle."""
    js, db = _store(tmp_path)
    job = js.create("ponpoke/flux2-klein-9b-uncensored-text-encoder",
                    id="v1-b065322dab9942df87ab93f0853018ab", kind="v1")
    assert job.worker is None and not job.terminal        # pending, owner-less

    res = js.cancel_authoritative("v1-b065322dab9942df87ab93f0853018ab",
                                  "operator cancel")
    assert res == {"cancelled": True, "mode": "store", "status": "cancelled"}

    # Survives a restart: a FRESH store off the same mirror sees it terminal
    # (gone from the live view), not resurrected pending.
    js2 = J.JobStore(mirror=SqliteMirror(path=db))
    live_ids = [r["id"] for r in js2.snapshot(live_only=True)]
    assert "v1-b065322dab9942df87ab93f0853018ab" not in live_ids
    full = js2.snapshot(live_only=False)
    row = next((r for r in full if r["id"].startswith("v1-b065322")), None)
    # (v1 is not a MEDIA_KIND, so it won't surface in the cross-process terminal
    # merge — the point that matters is it is NOT live anymore.)
    assert row is None or row["status"] == "cancelled"


def test_dead_owner_cancel_is_idempotent(tmp_path):
    js, _ = _store(tmp_path)
    js.create("m", id="p", kind="v1")
    first = js.cancel_authoritative("p")
    second = js.cancel_authoritative("p")           # already terminal
    assert first["cancelled"] is True and first["mode"] == "store"
    # Second: already terminal -> nothing to change; honest noop, never a lie.
    assert second["cancelled"] is False and second["mode"] == "noop"


def test_live_owner_cancel_relays_and_fires_handle(tmp_path):
    js, _ = _store(tmp_path)
    fired = []
    js.create("m", id="live", kind="chat")
    js.attach_cancel("live", lambda: fired.append(1))   # a real producer owner
    res = js.cancel_authoritative("live")
    assert res["mode"] == "relayed" and res["cancelled"] is True
    assert fired == [1]                                 # handle fired
    # Not force-marked terminal here — the owner's teardown does that.
    assert js.get("live").status == "pending"


def test_unknown_id_cancel_does_not_lie(tmp_path):
    js, _ = _store(tmp_path)
    res = js.cancel_authoritative("nope")
    assert res == {"cancelled": False, "mode": "noop", "status": None}


def test_mirror_only_dead_owner_cancel(tmp_path):
    """A row present ONLY in the mirror (no local memory record — the post-restart
    shape) must still cancel authoritatively via the store layer."""
    js, db = _store(tmp_path)
    seed = js.create("m", id="mirror-only", kind="v1")   # writes to mirror
    # A DIFFERENT process (fresh store, empty _jobs) cancels it.
    other = J.JobStore(mirror=SqliteMirror(path=db))
    assert other.get("mirror-only") is None              # not held locally
    res = other.cancel_authoritative("mirror-only")
    assert res["cancelled"] is True and res["mode"] == "store"


# ═══════════════ Defect 2 — pending-orphan expiry ═══════════════════════════
def test_pending_orphan_past_threshold_expires(tmp_path, monkeypatch):
    monkeypatch.setenv("HUGPY_JOB_PENDING_EXPIRY_SECONDS", "1800")
    js, _ = _store(tmp_path)
    old = js.create("unresolvable", id="old", kind="v1")
    old.progressed_at = time.time() - 3600            # 1h ago > 30m
    js._mirror_upsert(old)
    expired = js.expire_pending_orphans()
    assert "old" in expired
    j = js.get("old")
    assert j.status == "expired"
    assert "never dispatched" in j.message


def test_pending_under_threshold_does_not_expire(tmp_path, monkeypatch):
    monkeypatch.setenv("HUGPY_JOB_PENDING_EXPIRY_SECONDS", "1800")
    js, _ = _store(tmp_path)
    fresh = js.create("m", id="fresh", kind="v1")     # progressed_at = now
    assert js.expire_pending_orphans() == []
    assert js.get("fresh").status == "pending"


def test_a_progressing_job_never_expires(tmp_path, monkeypatch):
    """Forward progress resets progressed_at, so a job that is moving is never
    expired even if it started long ago."""
    monkeypatch.setenv("HUGPY_JOB_PENDING_EXPIRY_SECONDS", "1800")
    js, _ = _store(tmp_path)
    prog = js.create("m", id="prog", kind="v1")
    prog.progressed_at = time.time() - 3600
    js.update("prog", progress=0.5)                   # real advance -> resets clock
    assert js.expire_pending_orphans() == []


def test_view_driven_writes_do_not_reset_the_expiry_clock(tmp_path, monkeypatch):
    """The resurrection bug guard: a mere VIEW (snapshot) bumps `updated` but NOT
    progressed_at, so it must not push an orphan's expiry deadline out."""
    monkeypatch.setenv("HUGPY_JOB_PENDING_EXPIRY_SECONDS", "1800")
    js, db = _store(tmp_path)
    old = js.create("unresolvable", id="old2", kind="v1")
    old.progressed_at = time.time() - 3600
    js._mirror_upsert(old)
    # A cross-process view (fresh store) reads it — a log/view write path.
    viewer = J.JobStore(mirror=SqliteMirror(path=db))
    _ = viewer.snapshot(live_only=True)               # VIEW (bumps `updated`)
    _ = viewer.snapshot(live_only=True)               # again
    # Still expirable — the view did not reset the movement clock.
    assert "old2" in viewer.expire_pending_orphans()


def test_pending_with_a_worker_is_not_expired(tmp_path, monkeypatch):
    """A pending row that HAS a worker is being dispatched (an owner exists) — not
    an orphan; leave it."""
    monkeypatch.setenv("HUGPY_JOB_PENDING_EXPIRY_SECONDS", "1800")
    js, _ = _store(tmp_path)
    j = js.create("m", id="assigned", kind="v1", worker="ae")
    j.progressed_at = time.time() - 3600
    js._mirror_upsert(j)
    js.expire_pending_orphans()
    assert js.get("assigned").status == "pending"


# ═══════════════ Defect 3 — reject at intake ════════════════════════════════
def test_unresolvable_model_rejects_at_intake(monkeypatch):
    """A v1 submit naming a model that resolves to nothing -> 400 + known hint.
    Drives the route's resolution gate directly via resolve_model_key (the
    dispatcher's own resolver)."""
    from hugpy_engine.resolvers import model_resolver as MR
    # A key that resolves NOWHERE (registry/catalog/worker). The operator's
    # literal flux2-klein string happens to exist in THIS dev registry, so use a
    # guaranteed-absent key — the mechanism (Unknown model_key -> KeyError with a
    # known-keys hint) is what the 4xx gate keys off, catalog contents aside.
    assert MR.assure_model_key("zzz-nonexistent-model-xyz-999") is None
    # hugpy_engine raises ValueError here (the monolith raised KeyError); the
    # route's gate catches both so the 4xx still carries the resolver's hint.
    with pytest.raises((KeyError, ValueError)) as ei:
        MR.resolve_model_key(model_key="zzz-nonexistent-model-xyz-999")
    msg = str(ei.value)
    assert "Unknown model_key" in msg
    assert "known:" in msg or "/models" in msg        # the hint the 4xx carries


def test_known_model_still_resolves_even_if_not_local(monkeypatch):
    """The boundary: a model that IS in the registry resolves (queues) even when
    its files aren't on disk — registry membership, not disk presence. Lazy
    download is the design; only truly-unresolvable keys reject."""
    from hugpy_engine.resolvers import model_resolver as MR
    known = sorted(MR.MODEL_REGISTRY.keys())
    assert known, "registry must have at least one model to test the boundary"
    # A known key resolves to itself (or its canonical form) — no raise.
    resolved = MR.resolve_model_key(model_key=known[0])
    assert resolved in MR.MODEL_REGISTRY


def test_intake_gate_wired_into_v1_route():
    """Guard against the gate silently reverting: the v1 route must call
    resolve_model_key on an explicit model_key before accepting."""
    import inspect
    from hugpy_server.app.routes import v1_routes
    src = inspect.getsource(v1_routes.v1_chat_completions)
    assert "resolve_model_key" in src
    assert "reject-at-intake" in src.lower() or "REJECT-AT-INTAKE" in src
    # The engine's resolver raises ValueError for an unknown key; the gate must
    # catch it (a KeyError-only clause would fall through to the engine).
    assert "ValueError" in src


# ═══════════════ The HTTP cancel route carries `mode` ═══════════════════════
def test_cancel_route_reports_mode(monkeypatch, tmp_path):
    """POST /llm/jobs/<id>/cancel gains `mode` additively and reports honestly."""
    import importlib
    from flask import Flask
    cr = importlib.import_module(
        "hugpy_server.app.routes.comms_routes")

    js, _ = _store(tmp_path)
    js.create("m", id="stuck", kind="v1")             # pending, owner-less

    import hugpy_control.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "job_store", js)
    # media_bus.cancel is a no-op read for a non-media id.
    from hugpy_video.intel import media_bus
    monkeypatch.setattr(media_bus, "cancel",
                        lambda jid: {"cancelled": False, "status": None})

    app = Flask(__name__)
    app.register_blueprint(cr.comms_bp)
    client = app.test_client()

    r = client.post("/llm/jobs/stuck/cancel", json={"reason": "op"})
    body = r.get_json()
    assert r.status_code == 200
    assert body["cancelled"] is True
    assert body["mode"] == "store"                    # authoritative store-terminal
    assert body["status"] == "cancelled"

    # A second cancel of the now-terminal job must NOT lie cancelled:true.
    r2 = client.post("/llm/jobs/stuck/cancel", json={})
    b2 = r2.get_json()
    assert b2["cancelled"] is False and b2["mode"] == "noop"
