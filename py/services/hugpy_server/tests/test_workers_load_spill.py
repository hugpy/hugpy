"""TASK C (2026-07-25) — POST /llm/workers/<id>/load accepts an optional
``spill`` (n_gpu_layers / n_cpu_moe / …) and threads it through to the seat.

Context: a crashed MoE slot (ngl=-1, no --n-cpu-moe) is stuck EMPTY. The only
existing lever to set an explicit offload depth / MoE split is
/slots/<id>/relaunch, which REQUIRES an already-loaded model_key and 409s on
an empty slot — exactly the state a crashed slot is in. /load was the seat
lever but only ever took {model_key, spill?, force?, redownload?} with the
spill wired into the PERSISTED registry (assign_model), never forwarded to
the actual warm call that seats it right now.

This suite regresses, WITHOUT a live worker:
  * a plain /load (no spill) behaves byte-identically to before — the probe
    warm POST carries an empty body, and an omitted spill does NOT clear any
    previously-persisted override (matches /assign's omit-means-leave-alone
    convention);
  * an explicit {"n_gpu_layers": -1, "n_cpu_moe": 999} spill is (a) persisted
    via assign_model (same registry write /assign uses) AND (b) forwarded on
    the /probe warm POST body as {"spill": {...}}, so the seat that happens
    right now honors it instead of waiting for a later /infer call;
    validated (shape/keys) the same way /assign validates its spill — an
    unknown key or a bad n_gpu_layers value is refused with 400, not
        silently dropped or relayed to blow up on the worker;
  * unchanged 404 (unknown model key) / 409 (central missing files, disk,
    fit-preflight) refusal paths still fire ahead of the warm;
  * POST /llm/workers/<id>/load is operator-gated in _SENSITIVE (it was NOT
    before this change — a pre-existing gap fixed alongside the spill wire
    since we were already touching this route's contract).

Run: venv/bin/python -m pytest tests/test_workers_load_spill.py -q
"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-load-spill-test-"))

wr = importlib.import_module(
    "hugpy_server.app.routes.worker_routes")
oa = importlib.import_module("hugpy_server.app.operator_auth")

from flask import Flask

MODEL_KEY = "coder-next"
WORKER = {"id": "wid", "name": "ae", "status": "online", "url": "http://worker:9100"}


@pytest.fixture
def rig(monkeypatch):
    """Route every workers_load collaborator to controllable fakes — no real
    registry, no real HTTP, no real disk. Mirrors test_slot_relaunch_route.py's
    style (fake the collaborators, exercise the route's own logic)."""
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    client = app.test_client()

    state = {
        "models": {MODEL_KEY: {"key": MODEL_KEY}},
        "missing_reason": None,
        "disk_reason": None,
        "fit": {"fit": True, "reason": None},
        "assign_calls": [],
        "probe_calls": [],
        "probe_response": ({"ok": True, "fit": True}, 200),
    }

    monkeypatch.setattr(wr, "get_models_dict",
                        lambda dict_return=True: dict(state["models"]))
    monkeypatch.setattr(wr, "get_worker",
                        lambda wid: dict(WORKER) if wid == "wid" else None)
    monkeypatch.setattr(wr, "_central_missing_reason",
                        lambda mk: state["missing_reason"])
    monkeypatch.setattr(wr, "_disk_preflight_reason",
                        lambda worker, mk: state["disk_reason"])
    monkeypatch.setattr(wr, "_worker_fit", lambda mk, worker: dict(state["fit"]))

    def _fake_assign(worker_id, model_key, spill=None, source=None, retag=True):
        # PINS round (2026-09-23): assign_model gained provenance (source) and a
        # retag flag; /load stamps source="operator" (default) and retag when it
        # is not editing an existing spill. Recorded so the historical
        # {worker_id, model_key, spill} shape assertions stay intact.
        state["assign_calls"].append(
            {"worker_id": worker_id, "model_key": model_key, "spill": spill})
        state["assign_provenance"] = {"source": source, "retag": retag}
        return dict(WORKER)
    monkeypatch.setattr(wr, "assign_model", _fake_assign)

    class _FakeResp:
        def __init__(self, payload, status):
            self._payload, self.status_code = payload, status
            self.is_success = 200 <= status < 300

        def json(self):
            return self._payload

    class _FakeHttpx:
        @staticmethod
        def request(method, url, json=None, **kwargs):
            state["probe_calls"].append({"url": url, "json": json})
            payload, status = state["probe_response"]
            return _FakeResp(payload, status)

    # k59: the warm goes through worker_http, which calls httpx.request with a
    # split timeout — patching the real module (sys.modules) is what actually
    # takes effect, since the import happens inside the function body.
    import httpx as _real_httpx
    monkeypatch.setattr(_real_httpx, "request", _FakeHttpx.request)
    wh = importlib.import_module(
        "hugpy_fleet.central.worker_http")
    wh.reset_breakers()             # no failure streak carried in from elsewhere

    # set_load_report is imported lazily inside the background thread; stub
    # it so the thread doesn't touch a real registry file.
    workers_mod = importlib.import_module(
        "hugpy_fleet.central.workers")
    monkeypatch.setattr(workers_mod, "set_load_report", lambda *a, **k: None)

    return type("Rig", (), {"client": client, "state": state})()


def _wait_warm(rig, tries=50):
    """The warm runs on a daemon thread; poll briefly for the probe call to
    land instead of a fixed sleep (background thread finishes near-instantly
    against the fakes above — no real I/O)."""
    import time
    for _ in range(tries):
        if rig.state["probe_calls"]:
            return
        time.sleep(0.02)


# ═══════════ plain load (no spill) — unchanged behavior ═════════════════════
def test_plain_load_no_spill_is_unchanged(rig):
    r = rig.client.post("/llm/workers/wid/load", json={"model_key": MODEL_KEY})
    assert r.status_code == 200
    body = r.get_json()
    assert body["loaded"] == "loading" and body["assigned"] is True
    assert rig.state["assign_calls"] == [
        {"worker_id": "wid", "model_key": MODEL_KEY, "spill": None}]
    _wait_warm(rig)
    assert rig.state["probe_calls"] == [
        {"url": "http://worker:9100/probe/" + MODEL_KEY, "json": {}}]


# ═══════════ explicit spill — persisted AND forwarded to the warm ═══════════
def test_explicit_spill_persisted_and_forwarded_to_probe(rig):
    spill = {"n_gpu_layers": -1, "n_cpu_moe": 999}
    r = rig.client.post("/llm/workers/wid/load",
                        json={"model_key": MODEL_KEY, "spill": spill})
    assert r.status_code == 200
    assert rig.state["assign_calls"] == [
        {"worker_id": "wid", "model_key": MODEL_KEY, "spill": spill}]
    # PINS round: /load stamps operator provenance; a spill edit does NOT retag
    # the designation (retag only when no spill is supplied).
    assert rig.state["assign_provenance"] == {"source": "operator", "retag": False}
    _wait_warm(rig)
    assert rig.state["probe_calls"] == [
        {"url": "http://worker:9100/probe/" + MODEL_KEY,
         "json": {"spill": spill}}]


def test_partial_spill_n_cpu_moe_only(rig):
    spill = {"n_cpu_moe": 999}
    r = rig.client.post("/llm/workers/wid/load",
                        json={"model_key": MODEL_KEY, "spill": spill})
    assert r.status_code == 200
    assert rig.state["assign_calls"][0]["spill"] == spill
    _wait_warm(rig)
    assert rig.state["probe_calls"][0]["json"] == {"spill": spill}


# ═══════════ validation — same class of guard /assign applies ══════════════
def test_unknown_spill_key_refused_400(rig):
    r = rig.client.post(
        "/llm/workers/wid/load",
        json={"model_key": MODEL_KEY, "spill": {"not_a_real_key": 1}})
    assert r.status_code == 400
    assert "error" in r.get_json()
    assert rig.state["assign_calls"] == []          # never reached assign


def test_bad_n_gpu_layers_type_refused_400(rig):
    r = rig.client.post(
        "/llm/workers/wid/load",
        json={"model_key": MODEL_KEY, "spill": {"n_gpu_layers": "not-a-number"}})
    assert r.status_code == 400
    assert rig.state["assign_calls"] == []


def test_empty_spill_dict_is_valid_autofit(rig):
    r = rig.client.post("/llm/workers/wid/load",
                        json={"model_key": MODEL_KEY, "spill": {}})
    assert r.status_code == 200
    # An explicitly-empty spill IS a caller-supplied value (clears any prior
    # override) — distinct from omitting the field entirely.
    assert rig.state["assign_calls"] == [
        {"worker_id": "wid", "model_key": MODEL_KEY, "spill": {}}]


# ═══════════ existing refusal paths still fire ahead of the warm ═══════════
def test_unknown_model_key_404(rig):
    r = rig.client.post("/llm/workers/wid/load",
                        json={"model_key": "nope/nope"})
    assert r.status_code == 404
    assert rig.state["assign_calls"] == []


def test_central_missing_files_409(rig):
    rig.state["missing_reason"] = "no model directory"
    r = rig.client.post("/llm/workers/wid/load", json={"model_key": MODEL_KEY})
    assert r.status_code == 409
    assert rig.state["assign_calls"] == []


def test_fit_preflight_refusal_409(rig):
    rig.state["fit"] = {"fit": False, "reason": "won't fit VRAM"}
    r = rig.client.post("/llm/workers/wid/load", json={"model_key": MODEL_KEY})
    assert r.status_code == 409
    assert rig.state["assign_calls"] == []


def test_force_bypasses_fit_preflight_but_not_validation(rig):
    rig.state["fit"] = {"fit": False, "reason": "won't fit VRAM"}
    r = rig.client.post(
        "/llm/workers/wid/load",
        json={"model_key": MODEL_KEY, "force": True,
              "spill": {"n_gpu_layers": -1, "n_cpu_moe": 999}})
    assert r.status_code == 200
    assert rig.state["assign_calls"][0]["spill"] == {
        "n_gpu_layers": -1, "n_cpu_moe": 999}


# ═══════════ operator-gated ══════════════════════════════════════════════
def test_load_route_is_operator_gated():
    def _gated(path, method):
        return any(method in methods and rx.match(path)
                   for methods, rx in oa._SENSITIVE)
    assert _gated("/llm/workers/wid/load", "POST") is True
    assert _gated("/llm/workers/wid/load", "GET") is False
