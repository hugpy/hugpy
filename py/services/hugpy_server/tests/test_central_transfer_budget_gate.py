"""Central storage-budget gate on BACKGROUND transfers (ae 1.2TB, 2026-07-17).

Operator ruling: "its central that distributed these downloads... it simply
needs to abide by the limits set within its own backend." The manifest route is
the FIRST endpoint both the per-file and the archive transports hit, so the
budget gate lives there and refuses a background-over-budget pull before a byte
moves — on either transport.

Asserted via a Flask test client on worker_bp, with the registry + budget view
stubbed (same pattern as test_worker_transfer_auth_range.py):

  * X-Transfer-Purpose: reconcile|assign (BACKGROUND) + resident+incoming over
    budget                                            -> 409 + machine reason
  * X-Transfer-Purpose: demand (a called model)       -> served (200)
  * NO purpose header (old agent mid-convergence)      -> served (200)
  * background but UNDER budget                         -> served (200)
  * background over budget but UNKNOWN worker / no budget -> served (200)

Run with the tree venv:
    venv/bin/python -m pytest tests/test_central_transfer_budget_gate.py -v
    (or: venv/bin/python tests/test_central_transfer_budget_gate.py)
"""
import os
import sys
import importlib
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault(
    "PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-budget-gate-test-"))

wr = importlib.import_module(
    "hugpy_server.app.routes.worker_routes")

from flask import Flask

# A temp "model" dir with one file; its size is the INCOMING size the gate sees
# (single-format total == this file, framework unknown -> whole listing).
MODEL_DIR = tempfile.mkdtemp(prefix="hugpy-budget-gate-model-")
FILE_REL = "weights.bin"
INCOMING = 100 * 1024 * 1024                 # 100 MiB model
with open(os.path.join(MODEL_DIR, FILE_REL), "wb") as _fh:
    _fh.write(b"\0" * INCOMING)

VALID_TOKEN = "hpw_valid_test_token"
AUTH = {"Authorization": f"Bearer {VALID_TOKEN}"}
WORKER = "wkr-1"

MANIFEST_URL = "/llm/models/testmodel/manifest"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("HUGPY_OPERATOR_TOKEN", raising=False)
    monkeypatch.delenv("HUGPY_WORKER_ENROLL_REQUIRED", raising=False)
    monkeypatch.setenv("HUGPY_AUTH_MODE", "external")

    monkeypatch.setattr(
        wr, "get_models_dict",
        lambda dict_return=True: {
            "testmodel": {"key": "testmodel", "hub_id": "org/testmodel",
                          "name": "testmodel"}},
        raising=False)
    monkeypatch.setattr(wr, "route_destination", lambda model: MODEL_DIR,
                        raising=False)
    monkeypatch.setattr(wr, "verify_enrollment_token",
                        lambda tok: tok == VALID_TOKEN, raising=False)

    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    return app.test_client()


_workers_mod = importlib.import_module(
    "hugpy_fleet.central.workers")


def _stub_budget(monkeypatch, *, budget, resident, worker_id=WORKER):
    """Stub worker_storage_view (the SAME accounting storage_proposal computes).

    The gate does a fresh ``from ...workers import worker_storage_view`` per call,
    so patching the attribute on that module object is what it will resolve."""
    def _view(wid):
        if wid != worker_id:
            return None
        return {"budget": budget, "hot_bytes": resident}
    monkeypatch.setattr(_workers_mod, "worker_storage_view", _view, raising=False)


# ── background over budget -> 409 ──────────────────────────────────────────
def test_background_over_budget_409(client, monkeypatch):
    # budget 150M, resident 100M, incoming 100M => 200M > 150M -> refuse.
    _stub_budget(monkeypatch, budget=150 * 1024 * 1024, resident=100 * 1024 * 1024)
    r = client.get(MANIFEST_URL, headers={**AUTH, "X-Worker-Id": WORKER,
                                          "X-Transfer-Purpose": "reconcile"})
    assert r.status_code == 409, r.get_data(as_text=True)
    body = r.get_json()
    assert body["code"] == "storage_budget_exceeded"
    assert body["worker_id"] == WORKER
    assert body["incoming_bytes"] == INCOMING
    assert body["would_use_bytes"] == 200 * 1024 * 1024
    assert "budget" in body["reason"]


def test_background_assign_purpose_also_gated(client, monkeypatch):
    _stub_budget(monkeypatch, budget=150 * 1024 * 1024, resident=100 * 1024 * 1024)
    r = client.get(MANIFEST_URL, headers={**AUTH, "X-Worker-Id": WORKER,
                                          "X-Transfer-Purpose": "assign"})
    assert r.status_code == 409


# ── demand is NEVER budget-refused centrally ───────────────────────────────
def test_demand_served_even_over_budget(client, monkeypatch):
    _stub_budget(monkeypatch, budget=150 * 1024 * 1024, resident=100 * 1024 * 1024)
    r = client.get(MANIFEST_URL, headers={**AUTH, "X-Worker-Id": WORKER,
                                          "X-Transfer-Purpose": "demand"})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()["total_bytes"] == INCOMING


# ── missing purpose header (old agent) -> treated as demand -> served ──────
def test_missing_purpose_served(client, monkeypatch):
    _stub_budget(monkeypatch, budget=150 * 1024 * 1024, resident=100 * 1024 * 1024)
    r = client.get(MANIFEST_URL, headers={**AUTH, "X-Worker-Id": WORKER})
    assert r.status_code == 200


# ── background UNDER budget -> served ──────────────────────────────────────
def test_background_under_budget_served(client, monkeypatch):
    # budget 500M, resident 100M, incoming 100M => 200M <= 500M -> serve.
    _stub_budget(monkeypatch, budget=500 * 1024 * 1024, resident=100 * 1024 * 1024)
    r = client.get(MANIFEST_URL, headers={**AUTH, "X-Worker-Id": WORKER,
                                          "X-Transfer-Purpose": "reconcile"})
    assert r.status_code == 200


# ── unknown worker / no budget -> serve (never refuse what we can't size) ──
def test_unknown_worker_served(client, monkeypatch):
    _stub_budget(monkeypatch, budget=150 * 1024 * 1024, resident=100 * 1024 * 1024)
    r = client.get(MANIFEST_URL, headers={**AUTH, "X-Worker-Id": "someone-else",
                                          "X-Transfer-Purpose": "reconcile"})
    assert r.status_code == 200


def test_no_managed_budget_served(client, monkeypatch):
    # budget None (unmanaged) -> the gate must not refuse (pre-feature behavior).
    _stub_budget(monkeypatch, budget=None, resident=100 * 1024 * 1024)
    r = client.get(MANIFEST_URL, headers={**AUTH, "X-Worker-Id": WORKER,
                                          "X-Transfer-Purpose": "reconcile"})
    assert r.status_code == 200


def test_background_no_worker_id_served(client, monkeypatch):
    # A background purpose but no worker id (shouldn't happen) -> serve.
    _stub_budget(monkeypatch, budget=150 * 1024 * 1024, resident=100 * 1024 * 1024)
    r = client.get(MANIFEST_URL, headers={**AUTH, "X-Transfer-Purpose": "reconcile"})
    assert r.status_code == 200


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
