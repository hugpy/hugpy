"""Per-worker BOOT-LOAD STAR (boot_prewarm) — central's half.

The star = boot-load + ambiguity-priority, NOTHING else (operator RULING
2026-07-23, post-incident): (1) load on boot, once per process; (2) a priority
tie-break in central's worker ranking. It is NOT keep-warm, does NOT re-warm
after eviction and has NO eviction interaction. The identifier stays
``boot_prewarm`` (rename churn isn't worth it).

The three levers, for contrast:
  * ⭐ star (boot_prewarm) = boot-load ONCE + ambiguity ranking priority.
  * 🔒 static = warm AND eviction-protected. THE keep-warm tier.
  * 📌 pin = routing persistence only, never warms.

THIS file covers the state store + the server routes:
  * state: set / get / clear / replace / one-star-per-worker + persistence
    (engine-owned ``models_config`` flag store, exercised here because the
    routes below are its only caller);
  * routes: ``/llm/workers`` surfaces ``boot_prewarm``; the write route is
    operator-gated; the heartbeat & register replies carry it (omit-when-unset,
    landmine-proof: never persisted onto the stored record).

The worker-side boot-once adoption and the disk-vs-fetch discipline live in
``py/fleet/hugpy_fleet/tests/test_worker_boot_prewarm.py``; the ranking
tie-break in the fleet ``test_worker_wildcard.py``.
"""

from __future__ import annotations

import importlib
import os

import pytest
from flask import Flask

from worker_store_isolation import swap_worker_store

mc = importlib.import_module("hugpy_engine.config.models.models_config")
wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
oa = importlib.import_module("hugpy_server.app.operator_auth")
W = importlib.import_module("hugpy_fleet.central.workers")

OPERATOR = {"X-Operator-Token": "s3cret"}


@pytest.fixture
def isolated_flag_store(tmp_path, monkeypatch):
    """The star file lives beside MODELS_DISCOVERY_PATH; redirect that module
    global so the test never touches a live store."""
    monkeypatch.setattr(mc, "MODELS_DISCOVERY_PATH",
                        os.path.join(str(tmp_path), "model_discovery.json"))
    yield


@pytest.fixture
def gated_client(isolated_flag_store, monkeypatch):
    """Worker blueprint + operator gate (open mode + a token => the token is
    REQUIRED on sensitive routes; fleet reads stay open) over an isolated
    registry with a starred worker (``wk-star``) and a plain one (``wk-nostar``)."""
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", "s3cret")
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    oa.install_operator_gate(app)
    app.config["TESTING"] = True
    with swap_worker_store(prefix="hugpy-bpw-workers-"):
        W.worker_store.register(name="box", url="http://192.0.2.9:9100", worker_id="wk-star")
        W.worker_store.register(name="box2", url="http://192.0.2.10:9100", worker_id="wk-nostar")
        yield app.test_client()


# ── state store ──────────────────────────────────────────────────────────────

def test_flag_store_set_get_clear_replace(isolated_flag_store):
    assert mc.worker_boot_prewarm_state() == {}

    r = mc.set_worker_boot_prewarm("wk-a", "M~qwen7b", True)
    assert r["boot_prewarm"] == "M~qwen7b" and r["starred"] is True
    assert mc.worker_boot_prewarm_state() == {"wk-a": "M~qwen7b"}

    # one-star-per-worker: a new star REPLACES; a second worker is independent
    mc.set_worker_boot_prewarm("wk-a", "M~coder-next", True)
    assert mc.worker_boot_prewarm_state() == {"wk-a": "M~coder-next"}
    mc.set_worker_boot_prewarm("wk-b", "M~llama", True)
    assert mc.worker_boot_prewarm_state() == {"wk-a": "M~coder-next", "wk-b": "M~llama"}

    # clear IFF the key matches the current star
    mc.set_worker_boot_prewarm("wk-a", "M~not-current", False)
    assert mc.worker_boot_prewarm_state().get("wk-a") == "M~coder-next"
    mc.set_worker_boot_prewarm("wk-a", "M~coder-next", False)
    assert mc.worker_boot_prewarm_state() == {"wk-b": "M~llama"}

    # model_key=None + enabled=False clears unconditionally
    mc.set_worker_boot_prewarm("wk-b", None, False)
    assert mc.worker_boot_prewarm_state() == {}

    # legacy/bare on-disk shape tolerated
    mc.safe_dump_to_file(data={"wk-legacy": "M~old"}, file_path=mc._worker_boot_prewarm_path())
    assert mc.worker_boot_prewarm_state() == {"wk-legacy": "M~old"}


# ── routes ───────────────────────────────────────────────────────────────────

def test_set_route_is_operator_gated_and_persists(gated_client):
    r = gated_client.post("/llm/workers/wk-star/boot-prewarm", json={"model_key": "M~qwen7b"})
    assert r.status_code == 401
    assert mc.worker_boot_prewarm_state() == {}

    r = gated_client.post("/llm/workers/wk-star/boot-prewarm",
                          json={"model_key": "M~qwen7b"}, headers=OPERATOR)
    assert r.status_code == 200
    assert mc.worker_boot_prewarm_state().get("wk-star") == "M~qwen7b"


def test_star_is_surfaced_on_reads(gated_client):
    mc.set_worker_boot_prewarm("wk-star", "M~qwen7b", True)

    # GET map stays open (read tier, like the roster it mirrors)
    r = gated_client.get("/llm/workers/boot-prewarm")
    assert r.status_code == 200 and r.get_json() == {"wk-star": "M~qwen7b"}

    rows = gated_client.get("/llm/workers").get_json()
    by_id = {w.get("id"): w for w in rows}
    assert by_id["wk-star"].get("boot_prewarm") == "M~qwen7b"
    assert by_id["wk-nostar"].get("boot_prewarm") is None

    one = gated_client.get("/llm/workers/wk-star").get_json()
    assert one.get("boot_prewarm") == "M~qwen7b"


def test_heartbeat_and_register_replies_carry_star(gated_client):
    """Additive / omit-when-unset on the reply; never baked onto the stored record."""
    mc.set_worker_boot_prewarm("wk-star", "M~qwen7b", True)

    r = gated_client.post("/llm/workers/wk-star/heartbeat", json={})
    assert r.status_code == 200
    assert (r.get_json() or {}).get("boot_prewarm") == "M~qwen7b"

    r = gated_client.post("/llm/workers/wk-nostar/heartbeat", json={})
    assert r.status_code == 200
    assert "boot_prewarm" not in (r.get_json() or {})

    # landmine proof: the reply carry is a response-copy only
    raw = W.worker_store._load().get("wk-star") or {}
    assert "boot_prewarm" not in raw

    reg = gated_client.post("/llm/workers/register",
                            json={"name": "box", "worker_id": "wk-star",
                                  "gpus": [], "url": "http://192.0.2.9:9100"})
    assert reg.status_code == 200
    assert (reg.get_json() or {}).get("boot_prewarm") == "M~qwen7b"
    assert "boot_prewarm" not in (W.worker_store._load().get("wk-star") or {})


def test_heartbeat_survives_star_store_failure(gated_client, monkeypatch):
    """A heartbeat never 5xxes even if the star store read is unhealthy."""
    def _boom():
        raise RuntimeError("star store down")
    monkeypatch.setattr(wr, "worker_boot_prewarm_state", _boom)
    r = gated_client.post("/llm/workers/wk-star/heartbeat", json={})
    assert r.status_code == 200
    assert "boot_prewarm" not in (r.get_json() or {})
