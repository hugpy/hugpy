"""k2: central half of block propagation — the heartbeat REPLY carries the
operator model BLOCK set.

Journal-proven defect (ae, 2026-07-18): once ``unsloth~Qwen3-Coder-Next-GGUF``
was operator-blocked (models.blocked, aa4aea3), ae's worker-side slot-fill
reconciler kept retrying warm-loads of the blocked model for ~4.75 hours. A
block does NOT auto-unassign, so a still-assigned blocked model sat in
``state.assigned_models`` forever, invisible to the worker's own loops.

The fix has two halves; THIS file covers the central one:

  * Propagation — the block set rides the heartbeat REPLY (a plain, additive,
    omit-when-empty list), the exact wire idiom t28 calibration established
    (fed4ae8): published on a COPY of the reply, never persisted onto the
    stored worker record. ``workers_heartbeat`` in worker_routes.py.

The worker-side half (``_adopt_blocked_models``, the ``_fill_empty_slots``
skip, ``_kick_provision`` as the single choke, ``_reconcile_loop``) lives in
``py/fleet/hugpy_fleet/tests/test_block_propagation.py``.

Isolation: the worker registry is swapped for a tmpdir store
(``swap_worker_store``) and the settings store the blocklist rides is pointed
at a tmp file, so a block written here can never touch a live deployment.
"""

from __future__ import annotations

import importlib

import pytest
from flask import Flask

from worker_store_isolation import swap_worker_store

bl = importlib.import_module("hugpy_fleet.central.blocklist")
wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
W = importlib.import_module("hugpy_fleet.central.workers")


@pytest.fixture
def isolated_blocklist(tmp_path):
    """Point the settings store (what blocklist.py rides) at a tmp file."""
    from hugpy_control.settings import settings_store
    orig_path, orig_cache = settings_store._path, settings_store._cache
    settings_store._path = str(tmp_path / "settings.json")
    settings_store._cache = None
    try:
        yield
    finally:
        for k in list(bl.blocked_keys()):
            bl.unblock(k)
        settings_store._path, settings_store._cache = orig_path, orig_cache


@pytest.fixture
def client(isolated_blocklist):
    """A bare Flask app carrying only the worker blueprint, over an isolated
    registry with one registered worker (``wk-prop``)."""
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    app.config["TESTING"] = True
    with swap_worker_store(prefix="hugpy-block-prop-workers-"):
        W.worker_store.register(name="box", url="http://192.0.2.9:9100", worker_id="wk-prop")
        yield app.test_client()


def _beat(client):
    r = client.post("/llm/workers/wk-prop/heartbeat", json={})
    assert r.status_code == 200
    return r.get_json() or {}


def test_heartbeat_reply_carries_blocked_models(client):
    """Additive / omit-when-empty / sorted full set / reverts on unblock."""
    assert "blocked_models" not in _beat(client)

    bl.block("M~blocked-1")
    assert _beat(client).get("blocked_models") == ["M~blocked-1"]

    bl.block("A~blocked-2")
    assert _beat(client).get("blocked_models") == ["A~blocked-2", "M~blocked-1"]

    bl.unblock("M~blocked-1")
    bl.unblock("A~blocked-2")
    assert "blocked_models" not in _beat(client)


def test_blocked_models_never_persist_onto_stored_record(client):
    """Landmine proof: published on a COPY of the reply, never written back
    onto the stored worker record (the same idiom as calibration/reservations)."""
    bl.block("M~blocked-1")
    assert _beat(client).get("blocked_models") == ["M~blocked-1"]

    stored = client.get("/llm/workers/wk-prop").get_json() or {}
    assert "blocked_models" not in stored
    raw = W.worker_store._load().get("wk-prop") or {}
    assert "blocked_models" not in raw


def test_heartbeat_survives_blocklist_read_failure(client, monkeypatch):
    """A heartbeat never 5xxes even if the blocklist read is unhealthy."""
    def _boom():
        raise RuntimeError("settings store down")
    monkeypatch.setattr(wr, "_blocked_keys", _boom)
    r = client.post("/llm/workers/wk-prop/heartbeat", json={})
    assert r.status_code == 200
    assert "blocked_models" not in (r.get_json() or {})
