"""Evictable external GPU lease — the restored wildcard-process feature.

Coverage for the four worker routes (/ops/external/register|unregister|set|
claim), the _evict_model external branch reaching a lease's control URL (with
non-evictable honored), the studio_reserve claim path, and the gpu_lease
supervisor loop (fake child + fake worker). See worker/gpu_lease.py and
worker/external_residents.py.

Run: venv/bin/python -m pytest tests/test_external_gpu_lease.py -v
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent as A
from hugpy_fleet.worker import external_residents as extres
from hugpy_fleet.worker import gpu_lease as GL
from hugpy_fleet.worker import studio_reserve as SR

GIB = 1 << 30


@pytest.fixture(autouse=True)
def _clean_registry():
    extres.clear()
    yield
    extres.clear()


@pytest.fixture
def client():
    state = A.WorkerState(name="t", url=None, worker_id="w-extlease")
    return A.build_app(state).test_client()


# ── the four routes ─────────────────────────────────────────────────────────
def test_register_shows_in_residents_then_unregister_clears(client):
    r = client.post("/ops/external/register",
                    json={"model_key": "ocr:bluebook", "pid": os.getpid(),
                          "vram_gib": 17.0, "control_url": "http://127.0.0.1:9310",
                          "note": "gpu_lease: run"})
    assert r.status_code == 200 and r.get_json()["ok"] is True

    got = client.get("/ops/residents").get_json()
    keys = [e["model_key"] for e in got["external"]]
    assert "ocr:bluebook" in keys
    row = next(e for e in got["external"] if e["model_key"] == "ocr:bluebook")
    assert row["evictable"] is True and row["vram_gib"] == 17.0

    u = client.post("/ops/external/unregister",
                    json={"model_key": "ocr:bluebook"}).get_json()
    assert u["ok"] is True and u["was_registered"] is True
    assert extres.get("ocr:bluebook") is None


def test_register_missing_key_400(client):
    r = client.post("/ops/external/register", json={"pid": 1})
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_set_updates_policy_and_forwards_to_supervisor(client, monkeypatch):
    client.post("/ops/external/register",
                json={"model_key": "ocr:bluebook", "pid": os.getpid(),
                      "vram_gib": 17.0, "control_url": "http://127.0.0.1:9310"})

    seen = {}
    import httpx

    class _Resp:
        status_code = 200

    def _fake_post(url, json=None, timeout=None):
        seen["url"] = url
        seen["json"] = json
        return _Resp()

    monkeypatch.setattr(httpx, "post", _fake_post)

    r = client.post("/ops/external/set",
                    json={"model_key": "ocr:bluebook", "evictable": False,
                          "resume": "disabled"}).get_json()
    assert r["ok"] is True
    assert r["resident"]["evictable"] is False
    assert r["resident"]["resume"] == "disabled"
    assert r["supervisor_acked"] is True
    assert seen["url"].endswith("/set")
    assert seen["json"] == {"evictable": False, "resume": "disabled"}
    # the worker-side registry is authoritative and updated.
    assert extres.get("ocr:bluebook")["evictable"] is False


def test_set_unknown_lease_404(client):
    r = client.post("/ops/external/set",
                    json={"model_key": "nope", "evictable": True})
    assert r.status_code == 404 and r.get_json()["ok"] is False


def test_claim_reached_when_card_already_free(client, monkeypatch):
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: 20 * GIB)
    r = client.post("/ops/external/claim",
                    json={"model_key": "ocr:bluebook",
                          "target_free_gib": 17.0}).get_json()
    assert r["ok"] is True and r["reached"] is True and r["evicted"] == []


def test_claim_infeasible_evicts_nothing(client, monkeypatch):
    # Free is short and nothing is reclaimable -> the feasibility pre-check
    # refuses to strip collateral; reached=False, evicted empty.
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: 1 * GIB)
    monkeypatch.setattr(A, "_vram_residents", lambda state: [])
    monkeypatch.setattr(A, "_comfy_headroom_candidates", lambda exclude=None: [])
    monkeypatch.setattr(A, "_comfy_process_vram", lambda *a, **k: 0)
    r = client.post("/ops/external/claim",
                    json={"model_key": "ocr:bluebook",
                          "target_free_gib": 17.0}).get_json()
    assert r["ok"] is True and r["reached"] is False and r["evicted"] == []
    assert "infeasible" in r["note"]


def test_claim_no_target_400(client):
    r = client.post("/ops/external/claim", json={"model_key": "ocr:bluebook"})
    assert r.status_code == 400 and r.get_json()["ok"] is False


# ── _evict_model external branch ────────────────────────────────────────────
def _evict_rig(monkeypatch):
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: 10 * GIB)
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: 10 * GIB)
    monkeypatch.setattr(A, "_evict_gate", lambda mk: (True, ""))
    monkeypatch.setattr(A, "loaded_model_keys", lambda: [])
    monkeypatch.setattr(A, "_gpu_process_vram", lambda: {})


def test_evict_evictable_lease_reaches_control_url(monkeypatch):
    _evict_rig(monkeypatch)
    state = A.WorkerState(name="t", url=None, worker_id="w")
    extres.register("ocr:bluebook", os.getpid(),
                    control_url="http://127.0.0.1:9310", vram_gib=17.0,
                    evictable=True)

    paused = {}
    import httpx

    class _Resp:
        status_code = 200

    def _fake_post(url, json=None, timeout=None):
        paused["url"] = url
        return _Resp()

    monkeypatch.setattr(httpx, "post", _fake_post)

    res = A._evict_model(state, "ocr:bluebook", force=False)
    assert res["host_mode"] == "external" and res["evicted"] is True
    assert paused["url"] == "http://127.0.0.1:9310/pause"
    # the lease is marked yielded (pid cleared) but the record is KEPT.
    assert extres.get("ocr:bluebook")["state"] == "yielded"


def test_evict_non_evictable_refused_without_force(monkeypatch):
    _evict_rig(monkeypatch)
    state = A.WorkerState(name="t", url=None, worker_id="w")
    extres.register("ocr:bluebook", os.getpid(),
                    control_url="http://127.0.0.1:9310", vram_gib=17.0,
                    evictable=False)

    import httpx
    calls = {"n": 0}

    def _fake_post(url, json=None, timeout=None):
        calls["n"] += 1
        raise AssertionError("non-evictable lease must not be paused")

    monkeypatch.setattr(httpx, "post", _fake_post)

    res = A._evict_model(state, "ocr:bluebook", force=False)
    assert res["host_mode"] == "external" and res["evicted"] is False
    assert "non-evictable" in res["reason"]
    assert calls["n"] == 0
    # still registered as a running resident (never touched).
    assert extres.get("ocr:bluebook")["state"] == "running"


def test_evict_non_evictable_paused_with_force(monkeypatch):
    _evict_rig(monkeypatch)
    state = A.WorkerState(name="t", url=None, worker_id="w")
    extres.register("ocr:bluebook", os.getpid(),
                    control_url="http://127.0.0.1:9310", vram_gib=17.0,
                    evictable=False)

    import httpx

    class _Resp:
        status_code = 200

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())
    res = A._evict_model(state, "ocr:bluebook", force=True)
    assert res["host_mode"] == "external" and res["evicted"] is True


def test_evictable_lease_is_a_comfy_headroom_candidate(monkeypatch):
    monkeypatch.setattr(A, "loaded_model_keys", lambda: [])
    monkeypatch.setattr(A, "_residency", lambda mk: "on_demand")
    monkeypatch.setattr(A, "_dispatch_last_used", lambda: {})
    extres.register("ocr:bluebook", os.getpid(), vram_gib=17.0, evictable=True)
    extres.register("ocr:locked", os.getpid(), vram_gib=17.0, evictable=False)
    cands = A._comfy_headroom_candidates(exclude=None)
    assert "ocr:bluebook" in cands            # evictable -> candidate
    assert "ocr:locked" not in cands          # non-evictable -> never a candidate


# ── studio_reserve claim path ───────────────────────────────────────────────
def test_studio_reserve_acquire_claims_registers_releases(monkeypatch):
    calls = []

    def _fake_post(path, payload, timeout=SR._HTTP_TIMEOUT_S):
        calls.append((path, payload))
        if path == "/ops/external/claim":
            return {"reached": True, "free_after": 21 * GIB, "evicted": []}
        return {"ok": True}

    monkeypatch.setattr(SR, "_post", _fake_post)

    release = SR.acquire("job42", "wan2.1-i2v-14b")
    paths = [p for p, _ in calls]
    assert "/ops/external/claim" in paths
    assert "/ops/external/register" in paths
    claim_payload = next(pl for p, pl in calls if p == "/ops/external/claim")
    assert claim_payload["include_external"] is True
    reg_payload = next(pl for p, pl in calls if p == "/ops/external/register")
    assert reg_payload["evictable"] is False and reg_payload["model_key"] == "studio:job42"

    release()
    assert ("/ops/external/unregister", {"model_key": "studio:job42"}) in calls


def test_studio_reserve_no_template_is_a_noop(monkeypatch):
    calls = []
    monkeypatch.setattr(SR, "_post",
                        lambda path, payload, timeout=None: calls.append(path))
    release = SR.acquire("job99", "some-llm-with-no-template")
    assert calls == []          # no reserve taken
    release()                   # safe no-op


# ── gpu_lease supervisor loop (fake child + fake worker) ─────────────────────
@pytest.fixture
def _restore_signals():
    orig = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    yield
    for s, h in orig.items():
        signal.signal(s, h)


def _lease_args(**over):
    base = dict(key="ocr:test", vram_gib=1.0, priority="evictable",
                resume="enabled", worker="http://127.0.0.1:9",
                control_host="127.0.0.1", control_port=0, min_idle=None,
                poll=0.05, grace=0.5, heartbeat=1000.0, max_failures=3)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_gpu_lease_completes_on_child_exit_zero(monkeypatch, _restore_signals):
    posts = []

    def _fake_post_json(url, body, timeout=30.0):
        posts.append((url, body))
        if url.endswith("/ops/external/claim"):
            return {"reached": True, "free_after": 2 * GIB, "evicted": []}
        return {}

    monkeypatch.setattr(GL, "_post_json", _fake_post_json)
    monkeypatch.setattr(GL, "_gpu_pids", lambda: {})

    args = _lease_args()
    cmd = [sys.executable, "-c", "import sys; sys.exit(0)"]
    rc = GL.Lease(args, cmd).run()
    assert rc == 0
    urls = [u for u, _ in posts]
    assert any(u.endswith("/ops/external/claim") for u in urls)
    assert any(u.endswith("/ops/external/register") for u in urls)
    assert any(u.endswith("/ops/external/unregister") for u in urls)


def test_gpu_lease_stop_child_kills_the_group(monkeypatch, _restore_signals):
    monkeypatch.setattr(GL, "_post_json", lambda *a, **k: {})
    monkeypatch.setattr(GL, "_gpu_pids", lambda: {})
    args = _lease_args()
    lease = GL.Lease(args, [sys.executable, "-c", "import time; time.sleep(30)"])
    lease.launch()
    assert lease.child.poll() is None         # running
    stopped = lease.stop_child("test")
    assert stopped is True
    assert lease.child.poll() is not None     # dead
    assert lease.evictions == 1


def test_gpu_lease_yield_with_resume_disabled_exits_75(monkeypatch,
                                                       _restore_signals):
    monkeypatch.setattr(GL, "_post_json", lambda *a, **k: (
        {"reached": True} if str(a[0]).endswith("/claim") else {}))
    monkeypatch.setattr(GL, "_gpu_pids", lambda: {})
    args = _lease_args(resume="disabled")
    lease = GL.Lease(args, [sys.executable, "-c", "import time; time.sleep(30)"])

    def _evict_when_running():
        for _ in range(200):
            if lease.control_url and lease.child and lease.child.poll() is None:
                break
            time.sleep(0.02)
        # reach the lease's OWN control URL, exactly as _external_pause does.
        import urllib.request
        import json as _json
        req = urllib.request.Request(
            lease.control_url + "/pause", data=_json.dumps({}).encode(),
            method="POST", headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()

    t = threading.Thread(target=_evict_when_running, daemon=True)
    t.start()
    rc = lease.run()
    t.join(timeout=5)
    assert rc == 75                     # EX_TEMPFAIL: yielded, resume disabled
    assert lease.evictions >= 1
