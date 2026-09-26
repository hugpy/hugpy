"""Studio placement END-TO-END through the REAL central worker registry.

Proves the full chain the follow-up wired: a worker heartbeat carrying a ``studio``
signal PERSISTS onto the central record (``WorkerStore.heartbeat`` stores it verbatim,
like ``comfy``), the fleet's ``FleetWorkerRegistry`` exposes that record on the engine's
placement seam, and ``studio_placement.resolve_worker`` then PICKS that worker for a real
studio render with NO ``HUGPY_STUDIO_WORKER`` override — i.e. video placed like an LLM.

Also asserts the request schemas actually carry ``studio`` (so the routes' ``studio=
body.studio`` is a valid mapping) and that a heartbeat WITHOUT the field leaves the record
untouched (legacy-safe).

The worker store is isolated + swapped (``swap_worker_store``) so nothing touches a live
registry, and the placement seam is restored after the test.
"""
from __future__ import annotations

import importlib

import pytest

from worker_store_isolation import swap_worker_store

from hugpy_engine import placement as seam
from hugpy_fleet.central.placement import FleetWorkerRegistry
from hugpy_video.intel.runners import studio_placement as SP
from hugpy_video.intel.studio.job import make_studio_i2v
from hugpy_server.app.routes.worker_routes import HeartbeatRequest, RegisterRequest

W = importlib.import_module("hugpy_fleet.central.workers")

_GIB = 1024 ** 3
_REAL_MODEL = "wan2.1-t2v-1.3b"


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    """No env override, unpinned studio ok, and the placement seam RESET after the test
    (so the installed FleetWorkerRegistry never leaks into another test)."""
    monkeypatch.delenv("HUGPY_STUDIO_WORKER", raising=False)
    monkeypatch.setenv("STUDIO_ALLOW_UNPINNED", "1")
    yield
    seam.set_worker_registry(None)


def _gpu(total_gib):
    return {"index": 0, "memory_total": int(total_gib * _GIB),
            "memory_free": int(4.0 * _GIB)}


def _studio_blob(models):
    return {"render": True, "models": list(models), "weights_root": "/mnt/llm_storage"}


def _real_spec():
    return make_studio_i2v(capability="t2v", width=832, height=480, fps=16,
                           vram_budget_gb=8.0, seed=0, prompt="a shot")


def test_schemas_carry_studio():
    """The route reads ``body.studio``; prove both request schemas map it."""
    blob = _studio_blob([_REAL_MODEL])
    assert RegisterRequest(name="w", studio=blob).studio == blob
    assert HeartbeatRequest(studio=blob).studio == blob
    # absent -> None (legacy agent), so the route passes studio=None (a no-op store-side)
    assert HeartbeatRequest().studio is None


def test_heartbeat_studio_persists_and_places():
    with swap_worker_store(prefix="hugpy-studio-place-") as store:
        rec = store.register(name="ae-worker", url="http://10.0.0.9:7003",
                             gpus=[_gpu(24.0)], worker_id="ae-w1")
        wid = rec["id"]
        # New workers land pending; approve so it may serve (parity with LLM routing).
        W.set_worker_admission(wid, "approved")

        # A heartbeat carrying the studio signal (the SAME kwarg the route passes as
        # studio=body.studio) — must persist verbatim onto the record.
        blob = _studio_blob([_REAL_MODEL])
        store.heartbeat(wid, gpus=[_gpu(24.0)], loaded_models=[], studio=blob)
        assert store.get(wid)["studio"] == blob, "studio must persist on the record"

        # Now place a real render THROUGH the engine placement seam (no env override).
        seam.set_worker_registry(FleetWorkerRegistry())
        url, refusal = SP.resolve_worker(_real_spec())
        assert refusal is None, refusal
        assert url == "http://10.0.0.9:7003", url

        # The worker knows the model on disk; a model it does NOT hold refuses by name.
        other = make_studio_i2v(capability="id_lock", width=832, height=480, fps=16,
                                 vram_budget_gb=8.0, seed=0, prompt="x",
                                 reference_images=("__probe__",))
        ourl, orefusal = SP.resolve_worker(other)
        # id_lock binds a VACE model the worker doesn't advertise -> named refusal, no url.
        assert ourl == "" and orefusal is not None
        assert orefusal.code == "studio_model_not_on_worker", orefusal


def test_pending_worker_not_placed():
    """A render-capable worker that is NOT approved must not be a studio target."""
    with swap_worker_store(prefix="hugpy-studio-pending-") as store:
        rec = store.register(name="pending-box", url="http://10.0.0.10:7003",
                             gpus=[_gpu(24.0)], worker_id="pend-1")
        wid = rec["id"]                     # left "pending" (never approved)
        store.heartbeat(wid, gpus=[_gpu(24.0)], loaded_models=[],
                        studio=_studio_blob([_REAL_MODEL]))
        seam.set_worker_registry(FleetWorkerRegistry())
        url, refusal = SP.resolve_worker(_real_spec())
        assert url == "" and refusal is not None
        assert refusal.code == "no_studio_worker", refusal


def test_register_reply_advertises_studio_weights_root(monkeypatch):
    """Central tells a worker its SHARED studio weights root in the register reply, so the
    worker's presence probe can check the root the render actually loads from. The value
    is the manifest single-source ``job.studio_weights_root``."""
    import importlib
    from flask import Flask
    from hugpy_video.intel.studio.job import studio_weights_root

    wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    wr._STUDIO_WEIGHTS_ROOT_CACHE.clear()          # fresh (module-global cache)
    try:
        app = Flask(__name__)
        app.register_blueprint(wr.worker_bp)
        app.config["TESTING"] = True
        with swap_worker_store(prefix="hugpy-swr-"):
            resp = app.test_client().post("/llm/workers/register", json={
                "name": "ae-worker", "url": "http://10.0.0.9:7003",
                "gpus": [{"index": 0, "memory_total": _GIB}]})
            assert resp.status_code == 200, resp.data
            body = resp.get_json()
            assert body.get("studio_weights_root") == studio_weights_root()
            assert body["studio_weights_root"]      # non-empty
    finally:
        wr._STUDIO_WEIGHTS_ROOT_CACHE.clear()


def test_heartbeat_without_studio_leaves_record_untouched():
    """A legacy heartbeat (no studio) must not wipe a previously-stored signal."""
    with swap_worker_store(prefix="hugpy-studio-legacy-") as store:
        rec = store.register(name="ae2", url="http://10.0.0.11:7003",
                             gpus=[_gpu(24.0)], worker_id="ae-2")
        wid = rec["id"]
        blob = _studio_blob([_REAL_MODEL])
        store.heartbeat(wid, studio=blob)
        store.heartbeat(wid, loaded_models=[])        # no studio kwarg
        assert store.get(wid)["studio"] == blob, "a studio-less beat must not wipe it"
