"""Every completed call writes ITS OWN numbers (2026-09-23).

Before: the compute_actions ``call`` row carried only the token count, so every
call read tok/s 0/None and the console fell back to the EMA card (91.7 tok/s
for a model no prompt ever reached). Now the relay builds one per-call row
(tokens, wall duration, tokens/duration, the prompt/gen split, worker, caller)
and the store writes it verbatim; the headline is the mean over those rows.
"""
import importlib
import json

import pytest

MM = importlib.import_module("hugpy_fleet.central.model_metrics")


@pytest.fixture()
def store(tmp_path):
    return MM.ModelMetricsStore(path=str(tmp_path / "model_metrics.db"))


def test_per_call_row_is_this_calls_tokens_over_its_duration():
    from hugpy_engine.resolvers.remote import per_call_row
    worker = {"id": "w1", "name": "aeb", "model_alloc_modes": {"M": "gpu-only"}}
    meta = {"completion_tokens": 128, "prompt_tokens": 179, "elapsed_s": 1.6, "gen_s": 1.2,
            "prompt_s": 0.3, "engine_gen_s": 0.94, "source": "timings", "request_id": "req-1",
            "streaming": True, "ok": True}
    row = per_call_row(worker, "M", 136.9, meta, task="text-generation")
    assert row["tokens"] == 128 and row["duration_s"] == 1.6
    assert row["tok_per_s"] == pytest.approx(80.0)            # 128 / 1.6, not the engine's 136.9
    assert row["worker_card"] == "aeb:0" and row["outcome"] == "ok"
    d = row["detail"]
    assert d["engine_tok_s"] == 136.9 and d["prompt_s"] == 0.3 and d["prompt_tokens"] == 179
    assert d["caller"] == "api" and d["alloc_mode"] == "gpu-only" and d["request_id"] == "req-1"
    assert per_call_row(worker, "M", None, {"completion_tokens": 5}, caller="benchmark")["tok_per_s"] is None


def test_store_writes_the_per_call_row_verbatim(store):
    from hugpy_engine.resolvers.remote import per_call_row
    row = per_call_row({"name": "aeb"}, "M", 90.0, {"completion_tokens": 100, "elapsed_s": 2.0, "ok": True})
    assert store.record_call("M", 100.0, task="text-generation", compute_s=2.0, call=row) is True
    a = store.recent_actions(limit=5, action="call")[0]
    assert (a["tokens"], a["duration_s"], a["tok_per_s"], a["worker_card"]) == (100, 2.0, 50.0, "aeb:0")
    detail = a.get("detail") or json.loads(a.get("detail_json") or "{}")
    assert detail["task"] == "text-generation" and detail["engine_tok_s"] == 90.0


def test_store_without_call_keeps_the_old_row(store):
    assert store.record_call("M", 7.0, task="t") is True
    a = store.recent_actions(limit=1, action="call")[0]
    assert a["tokens"] == 7 and a["tok_per_s"] is None and a["duration_s"] is None


def test_placement_adapter_forwards_the_call_row(monkeypatch, store):
    from hugpy_fleet.central import placement as P
    adapter = P.FleetModelMetrics()
    monkeypatch.setattr(adapter, "_s", lambda: store)
    assert adapter.record_call("M", 10.0, task="t", compute_s=1.0,
                               call={"tokens": 10, "duration_s": 0.5, "tok_per_s": 20.0, "worker_card": "aeb:0"})
    assert store.recent_actions(limit=1, action="call")[0]["tok_per_s"] == 20.0


def test_call_stats_rows_map_the_aggregate():
    from hugpy_engine.model_index.repositories import CallsRepository

    # One CALL_STATS output row = model_name, worker, quant, alloc_mode, u,
    # g_worker, g_quant, g_u, then the 20 _STAT_COLS numbers (see the query in
    # model_index/query_registry.py: CALL_STATS). mean_tok_s is DERIVED here as
    # Σtokens / Σseconds — never a precomputed mean-of-rates column.
    #   stat tail order: n_calls, n_rated, tokens, seconds, p50, p90, min, max,
    #                    first_at, last_at, completion_tokens, n_no_tokens,
    #                    n_no_window, n_engine, n_stream, n_wall, n_bench,
    #                    n_bench_twins, n_no_quant, n_no_alloc
    def stat(n_calls, tokens, seconds, p90):
        return (n_calls, n_calls, tokens, seconds, 45.0, p90, 20.0, 70.0,
                1.0, 9.0, tokens, 0, 0, n_calls, 0, 0, 0, 0, 0, 0)

    class Cur:
        def execute(self, q, params=None):
            self.q, self.params = q, params

        def fetchall(self):
            return [
                # (m, w, quant, alloc, u=0) fully-grouped cell row: g_worker=0,
                # g_quant=0, g_u=0. 400 tokens / 8.0 s = 50.0 tok/s.
                ("M", "aeb", "Q4", "gpu_only", 0, 0, 0, 0, *stat(4, 400, 8.0, 60.0)),
                # model-level row (worker rolled up): g_worker=1, g_quant=1,
                # g_u=1. 500 tokens / 12.5 s = 40.0 tok/s.
                ("M", None, None, None, 0, 1, 1, 1, *stat(5, 500, 12.5, 60.0)),
            ]

    cur = Cur()
    rows = CallsRepository(None).call_stats(cur, ["M"])
    assert "AND c.model_name = ANY(%s)" in cur.q and "GROUPING SETS" in cur.q
    # name_forms is threaded twice: once as {model_expr} (folds aliases under
    # name_forms[0]) and once as the ANY() filter.
    assert cur.params == ("M", ["M"])
    cell = next(r for r in rows if r["level"] == "cell")
    model = next(r for r in rows if r["level"] == "model")
    assert cell["mean_tok_s"] == 50.0 and cell["n_calls"] == 4 and cell["p90"] == 60.0
    assert model["worker"] is None and model["mean_tok_s"] == 40.0       # 500 tokens / 12.5 s
    cur2 = Cur()
    CallsRepository(None).call_stats(cur2, None)
    assert "{model_filter}" not in cur2.q and cur2.params is None
