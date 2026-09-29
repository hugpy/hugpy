"""KNOWN-GOOD CONTRACT — the Ctx slider's preview endpoint prices the need
per context setting (weights + KV at that ctx vs the worker's budget).

Catalogue: notes/KNOWN-GOOD-CORE.md (area "evict + fit / allocation").
Source under test: hugpy_server/app/routes/worker_routes.py
(workers_context_preview, _context_preview, _model_ctx_geometry).

Deterministic: the placement preflight and the model geometry are stubbed on
the module; a flask test_request_context carries the query string.
"""
from __future__ import annotations

import importlib

import pytest

flask = pytest.importorskip("flask")
WR = importlib.import_module("hugpy_server.app.routes.worker_routes")
spill = importlib.import_module("hugpy_engine.spill")

GIB = 1 << 30
MIB = 1 << 20
CTX_MAX = 262144


def _kv(ctx):
    return 2 * 36 * ctx * 8 * 128 * 2          # the 36-layer / 8-kv-head / 128 geometry


@pytest.fixture
def stubbed(monkeypatch):
    worker = {"id": "w1", "name": "ae-worker", "gpu": "RTX 3090"}
    monkeypatch.setattr(WR, "get_worker", lambda wid: worker if wid == "w1" else None)
    monkeypatch.setattr(WR, "_worker_fit", lambda mk, w: {
        "need": int(12 * GIB * 1.15), "need_raw": int(12 * GIB), "headroom": 1.15,
        "gpu_vram_free": int(22.3 * GIB), "calibration_correction": None, "reason": None})
    monkeypatch.setattr(WR, "_model_ctx_geometry", lambda mk: {
        "geometry": {"n_layers": 36, "n_kv_heads": 8, "head_dim": 128, "ctx_train": CTX_MAX},
        "ctx_max": CTX_MAX, "dtype_bytes": 2.0, "source": "gguf-header"})
    return worker


def test_context_preview_returns_kv_need_budget_and_the_largest_fitting_pct(stubbed):
    """INVARIANT: GET /llm/workers/<id>/context-preview?model_key&pct returns,
    for the requested pct, ctx = pct x max, kv_bytes = KV at that ctx (the
    same spill.kv_bytes arithmetic the worker's need uses), need_bytes =
    weights + kv, budget_bytes = per-device free - the 512 MiB compute
    cushion, fits, and max_fitting_pct over 5% steps — so the slider shows
    "at 25%: X of Y free" instead of a token count. Established: 2026-09-29."""
    app = flask.Flask(__name__)
    with app.test_request_context("/llm/workers/w1/context-preview?model_key=m&pct=25"):
        resp = WR.workers_context_preview("w1")
    body = resp.get_json()
    weights = int(12 * GIB * 1.15)                 # 13.8 GiB; KV at 100% is 9.0 GiB
    budget = int(22.3 * GIB) - int(spill._CTX_COMPUTE_RESERVE_BYTES)
    assert body["ctx_max"] == CTX_MAX and body["weights_bytes"] == weights
    assert body["budget_bytes"] == budget and body["reserve_bytes"] == 512 * MIB
    r = body["requested"]
    assert r["pct"] == 25 and r["ctx"] == 65536
    assert r["kv_bytes"] == _kv(65536)
    assert r["need_bytes"] == weights + _kv(65536)
    assert r["fits"] is (weights + _kv(65536) <= budget)
    # every 5% step is priced the same way, and the largest fitting pct is marked
    steps = {s["pct"]: s for s in body["steps"]}
    assert list(steps) == list(range(5, 101, 5))
    for p, s in steps.items():
        assert s["ctx"] == round(CTX_MAX * p / 100) and s["kv_bytes"] == _kv(s["ctx"])
        assert s["need_bytes"] == weights + s["kv_bytes"]
    expected_max = max(p for p, s in steps.items() if s["need_bytes"] <= budget)
    assert body["max_fitting_pct"] == expected_max
    assert steps[100]["fits"] is False               # 13.8 + 9.0 GiB KV > 21.8 GiB budget
    assert steps[expected_max]["fits"] is True


def test_context_preview_refuses_missing_inputs_and_prices_unknown_geometry_never_zero(stubbed, monkeypatch):
    app = flask.Flask(__name__)
    with app.test_request_context("/llm/workers/w1/context-preview"):
        resp = WR.workers_context_preview("w1")
    assert resp[1] == 400
    with app.test_request_context("/llm/workers/nope/context-preview?model_key=m"):
        resp = WR.workers_context_preview("nope")
    assert resp[1] == 404
    # geometry unreadable but the max is known: the KV heuristic prices it, never 0
    monkeypatch.setattr(WR, "_model_ctx_geometry", lambda mk: {
        "geometry": {}, "ctx_max": 8192, "dtype_bytes": 2.0, "source": None})
    with app.test_request_context("/llm/workers/w1/context-preview?model_key=m&pct=50"):
        body = WR.workers_context_preview("w1").get_json()
    assert body["requested"]["ctx"] == 4096 and body["requested"]["kv_bytes"] > 0
