"""Placement feasibility preview — OBSERVE-ONLY (Phase 1 item 3).

Covers `GET /llm/models/<model_key>/placement`: pure read that answers "if a
FLOATING call for model_key arrived, which workers could feasibly hold it, and
which would win" — WITHOUT taking any action (no assign/grant/probe/mutation).

Reuses `_worker_fit` and `_disk_preflight_reason` verbatim (not re-tested here
— those have their own coverage elsewhere); this file exercises the new route's
wiring: enumeration, winner selection, degrade-on-error, and no-mutation.

Run with the tree venv from `.../dev/abstract_hugpy_dev`:
    ./venv/bin/python -m pytest tests/test_placement_observe.py -v
"""
import os
import sys
import copy
import importlib
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault(
    "PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-placement-test-"))

wr = importlib.import_module(
    "hugpy_server.app.routes.worker_routes")

from flask import Flask

GIB = 2 ** 30

# ── fabricated worker registry (never touches real data) ───────────────────
SMALL_VRAM_WORKER = {
    "id": "w-small", "name": "small-box", "status": "online",
    "vram_free": 8 * GIB, "free_ram": 16 * GIB,
    "models": [], "loaded_models": [], "grants": {},
}
BIG_VRAM_WORKER = {
    "id": "w-big", "name": "ae", "status": "online",
    "vram_free": 64 * GIB, "free_ram": 128 * GIB,
    "models": [], "loaded_models": [], "grants": {},
}
OFFLINE_WORKER = {
    "id": "w-off", "name": "offline-box", "status": "offline",
    "vram_free": 64 * GIB, "free_ram": 128 * GIB,
    "models": [], "loaded_models": [], "grants": {},
}

MODEL_KEY = "some/model"


def _registry(*workers):
    # Deep copy so a test that (incorrectly) mutated a dict wouldn't bleed
    # into another test's assertions either.
    return [copy.deepcopy(w) for w in workers]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("HUGPY_OPERATOR_TOKEN", raising=False)
    monkeypatch.delenv("HUGPY_WORKER_ENROLL_REQUIRED", raising=False)
    monkeypatch.setenv("HUGPY_AUTH_MODE", "external")
    # Tokenless allowed under gradual rollout (matches transfer-auth tests).
    monkeypatch.setattr(wr, "verify_enrollment_token", lambda tok: False,
                        raising=False)
    # No real disk telemetry -> _disk_preflight_reason returns None (unknown,
    # don't block) unless a test overrides worker["disk"].
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    return app.test_client()


def _set_registry(monkeypatch, workers):
    monkeypatch.setattr(wr, "list_workers", lambda: workers, raising=False)


def _set_size(monkeypatch, size_bytes):
    monkeypatch.setattr(wr, "_model_gguf_bytes", lambda model_key: size_bytes,
                        raising=False)


def _get(client, model_key=MODEL_KEY):
    return client.get(f"/llm/models/{model_key}/placement")


# ── 55 GiB -> only the big worker is feasible; it wins ──────────────────────
def test_huge_model_feasible_only_on_big_worker(client, monkeypatch):
    registry = _registry(SMALL_VRAM_WORKER, BIG_VRAM_WORKER)
    _set_registry(monkeypatch, registry)
    _set_size(monkeypatch, 55 * GIB)

    r = _get(client)
    assert r.status_code == 200
    data = r.get_json()
    assert data["model_key"] == MODEL_KEY
    assert data["size_bytes"] == 55 * GIB

    by_name = {w["name"]: w for w in data["workers"]}
    small = by_name["small-box"]
    big = by_name["ae"]

    assert small["feasible"] is False
    assert small["reason"]                       # a real reason string
    assert "GiB" in small["reason"] or "GB" in small["reason"]

    assert big["feasible"] is True
    assert data["winner"] == "ae"
    assert "ae" in data["feasible_workers"]
    assert "small-box" not in data["feasible_workers"]


# ── 9 GiB (small) -> feasible everywhere; winner prefers gpu_resident/most vram_free ──
def test_small_model_feasible_everywhere_prefers_most_vram(client, monkeypatch):
    registry = _registry(SMALL_VRAM_WORKER, BIG_VRAM_WORKER)
    _set_registry(monkeypatch, registry)
    _set_size(monkeypatch, 9 * GIB)

    r = _get(client)
    assert r.status_code == 200
    data = r.get_json()

    assert all(w["feasible"] for w in data["workers"])
    assert set(data["feasible_workers"]) == {"small-box", "ae"}
    # Both fit VRAM outright (gpu_resident); winner = most vram_free = the big one.
    assert data["winner"] == "ae"
    assert "vram" in data["winner_reason"].lower()


# ── a worker that already has the model wins over an empty feasible one ────
def test_already_has_wins_over_empty_feasible(client, monkeypatch):
    small_with_model = copy.deepcopy(SMALL_VRAM_WORKER)
    small_with_model["models"] = [MODEL_KEY]   # operator-assigned = already_has
    registry = _registry(small_with_model, BIG_VRAM_WORKER)
    _set_registry(monkeypatch, registry)
    _set_size(monkeypatch, 9 * GIB)   # small enough to fit everywhere

    r = _get(client)
    assert r.status_code == 200
    data = r.get_json()

    by_name = {w["name"]: w for w in data["workers"]}
    assert by_name["small-box"]["already_has"] is True
    assert by_name["ae"]["already_has"] is False
    # Despite ae having far more free VRAM, the box that already holds it wins.
    assert data["winner"] == "small-box"
    assert "already holds" in data["winner_reason"]


# ── no feasible online worker -> winner=null with a reason ─────────────────
def test_no_feasible_online_worker_winner_null(client, monkeypatch):
    # Only a small worker, online, and the model is huge -> infeasible.
    registry = _registry(SMALL_VRAM_WORKER)
    _set_registry(monkeypatch, registry)
    _set_size(monkeypatch, 55 * GIB)

    r = _get(client)
    assert r.status_code == 200
    data = r.get_json()
    assert data["winner"] is None
    assert data["winner_reason"]
    assert data["feasible_workers"] == []


def test_no_online_worker_at_all_winner_null(client, monkeypatch):
    registry = _registry(OFFLINE_WORKER)
    _set_registry(monkeypatch, registry)
    _set_size(monkeypatch, 9 * GIB)

    r = _get(client)
    assert r.status_code == 200
    data = r.get_json()
    assert data["winner"] is None
    assert data["winner_reason"] == "no online worker"
    by_name = {w["name"]: w for w in data["workers"]}
    assert by_name["offline-box"]["online"] is False


# ── strictly observe-only: calling it (even twice) mutates NOTHING ─────────
def test_endpoint_mutates_nothing(client, monkeypatch):
    registry = _registry(SMALL_VRAM_WORKER, BIG_VRAM_WORKER)
    _set_registry(monkeypatch, registry)
    _set_size(monkeypatch, 55 * GIB)

    before = copy.deepcopy(registry)
    r1 = _get(client)
    r2 = _get(client)
    assert r1.status_code == 200 and r2.status_code == 200
    assert registry == before   # no worker dict field was written


# ── degrade gracefully: a worker whose fit/disk check raises never 500s ────
def test_degrades_gracefully_on_worker_error(client, monkeypatch):
    ok_worker = copy.deepcopy(BIG_VRAM_WORKER)
    registry = _registry(SMALL_VRAM_WORKER, ok_worker)
    _set_registry(monkeypatch, registry)

    def _boom(model_key, worker):
        if worker.get("id") == "w-small":
            raise RuntimeError("simulated sizing failure")
        return {"fit": True, "gpu_resident": True, "vram_free": worker.get("vram_free"),
                "ram_free": worker.get("free_ram"), "reason": None}

    monkeypatch.setattr(wr, "_worker_fit", _boom, raising=False)
    monkeypatch.setattr(wr, "_model_gguf_bytes",
                        lambda model_key: (_ for _ in ()).throw(RuntimeError("boom")),
                        raising=False)

    r = _get(client)
    assert r.status_code == 200   # never a 500 over one bad worker
    data = r.get_json()
    assert data["size_bytes"] is None   # size lookup failed -> degrade to None
    by_name = {w["name"]: w for w in data["workers"]}
    assert by_name["small-box"]["feasible"] is None
    assert "sizing failure" in (by_name["small-box"]["reason"] or "")
    assert by_name["ae"]["feasible"] is True
    assert data["winner"] == "ae"


# ── auth: same gate as the other transfer-adjacent reads ───────────────────
def test_requires_auth_when_enrollment_required(client, monkeypatch):
    monkeypatch.setenv("HUGPY_WORKER_ENROLL_REQUIRED", "1")
    _set_registry(monkeypatch, _registry(SMALL_VRAM_WORKER))
    _set_size(monkeypatch, 9 * GIB)
    assert _get(client).status_code == 401


def test_authenticated_ok_when_required(client, monkeypatch):
    monkeypatch.setenv("HUGPY_WORKER_ENROLL_REQUIRED", "1")
    monkeypatch.setattr(wr, "verify_enrollment_token",
                        lambda tok: tok == "hpw_valid", raising=False)
    _set_registry(monkeypatch, _registry(SMALL_VRAM_WORKER))
    _set_size(monkeypatch, 9 * GIB)
    r = client.get(f"/llm/models/{MODEL_KEY}/placement",
                   headers={"Authorization": "Bearer hpw_valid"})
    assert r.status_code == 200
