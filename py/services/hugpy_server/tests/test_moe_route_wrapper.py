"""Regression (2026-10-01): POST /llm/workers/<id>/moe raised TypeError on every
toggle. The route calls the module-level ``set_moe(..., public_view=False)``
while the wrapper only took (worker_id, model_key, value). The wrapper now passes
keyword options through to WorkerStore.set_moe. Calls the real route against a
real (tmp) WorkerStore."""
from __future__ import annotations

import importlib

import pytest
from flask import Flask

W = importlib.import_module("hugpy_fleet.central.workers")
wr = importlib.import_module("hugpy_server.app.routes.worker_routes")

MK = "Qwen3-Coder-Next-GGUF"


@pytest.fixture
def client(monkeypatch, tmp_path):
    s = W.WorkerStore(path=str(tmp_path / "wk.json"))
    monkeypatch.setattr(W, "worker_store", s)
    monkeypatch.setattr(W, "required_pkg_version", lambda: None)
    monkeypatch.setattr(W, "moe_capable", lambda mk: True)
    w = s.register(name="ae", url="http://ae:9100")
    s.set_admission(w["id"], "approved")
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    return app.test_client(), s, w["id"]


def test_single_model_toggle_ok(client):
    c, s, wid = client
    r = c.post(f"/llm/workers/{wid}/moe", json={"model_key": MK, "value": True})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()["ok"] is True
    # stored under the model's CANONICAL registry key (bare or Owner~Repo,
    # whichever the catalog holds at the time)
    assert r.get_json()["worker"]["moe_by_model"][W._canonical_registry_key(MK)] is True
    r = c.post(f"/llm/workers/{wid}/moe", json={"model_key": MK, "value": None})
    assert r.status_code == 200, r.get_data(as_text=True)


def test_wrapper_accepts_route_keywords(client):
    _, _, wid = client
    out = W.set_moe(wid, MK, False, public_view=False)
    assert out and out["moe_by_model"][W._canonical_registry_key(MK)] is False
