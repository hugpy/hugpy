"""Regression (2026-10-01): pricing a planned split must never build public
worker views.

planned_split -> planned_need -> weights_margin_for -> _margin_records_for used
worker_store.all(), which builds EVERY worker's public view, whose model view
prices every model through planned_split again. The cycle recursed until the
stack ran out on each listing and starved central (/llm/workers hung; the
0.2.5.post33 rollout went unhealthy). The margin lookup now reads raw records.
"""
from __future__ import annotations

import importlib

import pytest

W = importlib.import_module("hugpy_fleet.central.workers")

GIB = 1 << 30
MK = "Qwen3-Coder-Next-GGUF"


@pytest.fixture
def store(monkeypatch, tmp_path):
    s = W.WorkerStore(path=str(tmp_path / "wk.json"))
    monkeypatch.setattr(W, "worker_store", s)
    monkeypatch.setattr(W, "required_pkg_version", lambda: None)
    ae = s.register(name="ae", url="http://ae:9100")
    s.set_admission(ae["id"], "approved")
    s.heartbeat(ae["id"], gpus=[{"index": 0, "name": "NVIDIA GeForce RTX 3090",
                                 "memory_total": 24 * GIB, "memory_free": 21 * GIB}],
                weights_margins={MK: {"model_key": MK, "margin": 1.04, "samples": 2,
                                      "device_class": "NVIDIA GeForce RTX 3090",
                                      "measured_at": 1000.0}})
    return s, ae["id"]


def test_margin_lookup_builds_no_public_view(store, monkeypatch):
    s, _ = store
    calls = {"n": 0}
    real = W._public_view

    def counting(worker):
        calls["n"] += 1
        return real(worker)

    monkeypatch.setattr(W, "_public_view", counting)
    rec = W.weights_margin_for(MK)
    assert rec and rec["margin"] == 1.04 and rec["worker"] == "ae"
    assert calls["n"] == 0


def test_listing_with_planned_need_does_not_recurse(store, monkeypatch):
    """A listing whose planned_split consults the margin store builds each
    worker's view exactly once."""
    s, wid = store
    calls = {"n": 0}
    real = W._public_view

    def counting(worker):
        calls["n"] += 1
        return real(worker)

    def fake_planned_need(worker, model_key, **kw):
        W.weights_margin_for(model_key)          # the call that used to cycle
        return None

    monkeypatch.setattr(W, "_public_view", counting)
    monkeypatch.setattr(W, "planned_need", fake_planned_need)
    raw = s.raw_all()[0]
    raw.setdefault("models", [MK])
    raw.pop("_model_view_cache", None)
    views = s.all()
    assert len(views) == 1 and calls["n"] == 1
