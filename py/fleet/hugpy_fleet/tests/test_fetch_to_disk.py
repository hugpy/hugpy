"""Fetch-to-disk vs load-to-VRAM split (operator ruling 2026-09-24).

/probe was an undocumented download+load combo. The fetch-only half is
POST /models/fetch on the worker agent: it downloads a model to the drive via
the normal provision path (ensure_model_present) and NEVER seats/warms it into
VRAM. These tests prove the route wiring (fetch kicks a load=False provision,
never a probe/load) and the core (_kick_provision(load=False) provisions but
does not seat, while the default load=True DOES seat).
"""
import importlib
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

agent = importlib.import_module("hugpy_fleet.worker.agent")
P = importlib.import_module("hugpy_storage.provision")
SL = importlib.import_module("hugpy_engine.serve.slots")


def _wait_until(cond, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return cond()


# ── the /models/fetch route ─────────────────────────────────────────────────
@pytest.fixture
def rig(monkeypatch):
    state = agent.WorkerState(name="t", url=None, worker_id="w-fetch", central_url=None)
    kicks, probes = [], []
    monkeypatch.setattr(agent, "_kick_provision",
                        lambda st, mk, purpose="reconcile", load=True:
                        kicks.append({"mk": mk, "purpose": purpose, "load": load}))
    # If the route ever fell back to the load path, this would record it.
    monkeypatch.setattr(agent, "_probe_model",
                        lambda mk, st: probes.append(mk) or {"ok": True, "model_key": mk})
    client = agent.build_app(state).test_client()
    return type("Rig", (), {"client": client, "kicks": kicks, "probes": probes})()


def test_fetch_route_kicks_a_load_false_provision_and_never_probes(rig, monkeypatch):
    monkeypatch.setattr(P, "model_is_local", lambda mk: False)
    r = rig.client.post("/models/fetch", json={"model_key": "org/m"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] and body["started"] and body["already_local"] is False
    assert rig.kicks == [{"mk": "org/m", "purpose": "fetch", "load": False}]
    assert rig.probes == []                       # a fetch is NEVER a load


def test_fetch_route_is_a_noop_when_already_on_disk(rig, monkeypatch):
    monkeypatch.setattr(P, "model_is_local", lambda mk: True)
    body = rig.client.post("/models/fetch", json={"model_key": "org/m"}).get_json()
    assert body["ok"] and body["already_local"] is True and body["started"] is False
    assert rig.kicks == [] and rig.probes == []


def test_fetch_route_requires_model_key(rig):
    r = rig.client.post("/models/fetch", json={})
    assert r.status_code == 400 and r.get_json()["ok"] is False


# ── the core: _kick_provision(load=False) provisions but does not seat ───────
class _RecProvision:
    def __init__(self):
        self.calls = []

    def __call__(self, model_key, central_url, progress=None, state=None, purpose=None):
        self.calls.append(model_key)
        return True


@pytest.fixture
def core(monkeypatch):
    prov = _RecProvision()
    seats, materializes = [], []
    monkeypatch.setattr(P, "ensure_model_present", prov)
    monkeypatch.setattr(P, "model_is_local", lambda mk: False)
    monkeypatch.setattr(agent, "_fill_empty_slots", lambda st: seats.append(True))
    monkeypatch.setattr(agent, "_materialize", lambda runner: materializes.append(True))
    monkeypatch.setattr(SL, "slots_enabled", lambda: True)   # seating would run if load
    monkeypatch.delenv("WORKER_PRELOAD", raising=False)
    return type("Core", (), {"prov": prov, "seats": seats, "materializes": materializes})()


def test_kick_provision_load_false_downloads_without_seating(core):
    state = agent.WorkerState(name="t", url=None, worker_id="w-core1", central_url=None)
    agent._kick_provision(state, "m-fetch", purpose="fetch", load=False)
    assert _wait_until(lambda: "m-fetch" in core.prov.calls)
    assert _wait_until(lambda: state._provisioning == set())
    assert core.prov.calls == ["m-fetch"]         # it DID pull to disk
    assert core.seats == [] and core.materializes == []   # but never seated/loaded


def test_kick_provision_default_load_seats_after_provisioning(core):
    state = agent.WorkerState(name="t", url=None, worker_id="w-core2", central_url=None)
    agent._kick_provision(state, "m-load", purpose="reconcile")   # load=True default
    assert _wait_until(lambda: "m-load" in core.prov.calls)
    assert _wait_until(lambda: len(core.seats) >= 1)              # seating DID run
