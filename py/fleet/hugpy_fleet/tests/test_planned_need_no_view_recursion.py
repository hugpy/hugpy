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


class _FakePG:
    """The PostgreSQL backend's read/transaction shape over the file store, so
    _load() takes the materializing path the live registry takes."""

    def __init__(self, store):
        self.s = store

    def read(self, fn):
        return fn()

    from contextlib import contextmanager

    @contextmanager
    def transaction(self, fn):
        workers = fn()
        yield workers


def test_cold_model_view_with_pg_backend_does_not_recurse(store, monkeypatch):
    """2026-10-01 second hang: raw_all() -> _load() -> _materialize_storage ->
    planned_split -> ... -> raw_all() recursed (RLock) whenever a model view was
    stale, i.e. on the first listing after a landing or after the hourly expiry."""
    s, wid = store
    with s._transaction() as workers:
        workers[wid]["models"] = [MK]
    depth = {"now": 0, "max": 0}
    real = s._materialize_storage_pass

    def counting(workers):
        depth["now"] += 1
        depth["max"] = max(depth["max"], depth["now"])
        try:
            return real(workers)
        finally:
            depth["now"] -= 1

    monkeypatch.setattr(s, "_materialize_storage_pass", counting)
    monkeypatch.setattr(W, "planned_split",
                        lambda worker, mk: {"margin": W.weights_margin_for(mk)})
    s._pg = _FakePG(s)
    s._cache = None
    for w in s._read_unlocked().values():
        w.pop("_model_view_cache", None)
    # Materialization rides the WRITE transaction since 85b9ab8 (reads never
    # materialize); a stale view there must still not recurse.
    with s._transaction():
        pass
    rows = s.all()
    assert rows and depth["max"] == 1
