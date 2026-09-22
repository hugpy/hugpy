"""Per-query tok/s LEDGER: rollup math, JSONL roundtrip, report grouping, and
the state-snapshot the record path now stamps.

Companion to ``test_serve_metrics_ledger.py`` (which pins the rollup COLUMNS on
model_call_stats and the eviction-invariance). This file pins the JSONL side:
the ledger line, its ``state`` snapshot, and the ``(worker, model, config_key)``
report that ranks configurations by mean tok/s.

Run: venv/bin/python -m pytest tests/test_toks_report.py tests/test_serve_metrics_ledger.py -v
"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PROJECTS_HOME",
                      tempfile.mkdtemp(prefix="hugpy-toks-report-test-"))

W = importlib.import_module(
    "hugpy_fleet.central.workers")
TR = importlib.import_module("hugpy_fleet.toks_report")

from worker_store_isolation import swap_worker_store  # noqa: E402

LIVE_TOK_S = 115.25501867809274


@pytest.fixture()
def toks_log(monkeypatch):
    """Redirect HUGPY_TOKS_LOG at a fresh temp file for the duration."""
    fd, path = tempfile.mkstemp(prefix="hugpy-toks-", suffix=".jsonl")
    os.close(fd)
    os.unlink(path)
    monkeypatch.setenv("HUGPY_TOKS_LOG", path)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# ① rollup math — the {last_tok_s, avg_tok_s, n} step _rollup_tok_stats does.
# ---------------------------------------------------------------------------

class TestRollupTokStats:
    def test_first_sample_seeds_all_columns_and_n_is_one(self):
        row = {}
        W._rollup_tok_stats(row, 100.0, 1000.0)
        assert row["last_tok_s"] == pytest.approx(100.0)
        assert row["avg_tok_s"] == pytest.approx(100.0)   # EWMA seeds at sample
        assert row["n"] == 1
        assert row["last_ts"] == pytest.approx(1000.0)

    def test_second_sample_blends_avg_but_last_is_raw_and_n_increments(self):
        row = {"last_tok_s": 100.0, "avg_tok_s": 100.0, "n": 1,
               "last_ts": 1000.0}
        W._rollup_tok_stats(row, 200.0, 1001.0)
        assert row["last_tok_s"] == pytest.approx(200.0)   # raw
        assert row["avg_tok_s"] == pytest.approx(130.0)    # 0.3*200+0.7*100
        assert row["n"] == 2

    def test_model_key_is_stamped_only_when_given(self):
        row = {}
        W._rollup_tok_stats(row, 50.0, 1.0, model_key="m1")
        assert row["last_model"] == "m1"
        row2 = {}
        W._rollup_tok_stats(row2, 50.0, 1.0)
        assert "last_model" not in row2

    def test_bad_prior_n_recovers_to_one(self):
        row = {"n": "garbage"}
        W._rollup_tok_stats(row, 10.0, 1.0)
        assert row["n"] == 1


# ---------------------------------------------------------------------------
# ② JSONL append + tail read roundtrip.
# ---------------------------------------------------------------------------

class TestLedgerRoundtrip:
    def test_append_then_read_newest_first(self, toks_log):
        for i in range(3):
            W._append_toks_log({"ts": float(i), "worker_id": "w1",
                                "model_key": "m1", "tok_s": 10.0 + i})
        got = W.read_toks_log(limit=10)
        assert [e["tok_s"] for e in got] == [12.0, 11.0, 10.0]   # newest first

    def test_filters_by_worker_and_model(self, toks_log):
        W._append_toks_log({"ts": 1, "worker_id": "w1", "model_key": "a",
                            "tok_s": 1.0})
        W._append_toks_log({"ts": 2, "worker_id": "w2", "model_key": "b",
                            "tok_s": 2.0})
        assert [e["worker_id"] for e in W.read_toks_log(worker_id="w2")] == ["w2"]
        assert [e["model_key"] for e in W.read_toks_log(model_key="a")] == ["a"]

    def test_limit_is_honored_and_max_limit_raises_the_cap(self, toks_log):
        for i in range(20):
            W._append_toks_log({"ts": float(i), "worker_id": "w", "tok_s": 1.0})
        assert len(W.read_toks_log(limit=5)) == 5
        # default cap is 500; max_limit lets the fleet routes ask for more
        assert len(W.read_toks_log(limit=1000, max_limit=1000)) == 20

    def test_missing_log_reads_empty_never_raises(self, toks_log):
        assert W.read_toks_log(limit=10) == []


# ---------------------------------------------------------------------------
# ③ report grouping + percentiles — the shared function both surfaces use.
# ---------------------------------------------------------------------------

class TestPercentile:
    def test_empty_is_none(self):
        assert TR.percentile([], 50) is None

    def test_single_value(self):
        assert TR.percentile([42.0], 95) == 42.0

    def test_median_and_p95_interpolate(self):
        xs = [float(i) for i in range(1, 11)]      # 1..10
        assert TR.percentile(xs, 50) == pytest.approx(5.5)
        assert TR.percentile(xs, 95) == pytest.approx(9.55)


class TestGroupEntries:
    def _entry(self, w, m, ck, tok_s):
        return {"worker_id": w, "model_key": m, "tok_s": tok_s,
                "state": {"config_key": ck}}

    def test_groups_by_triple_and_computes_stats(self):
        entries = [
            self._entry("w1", "m1", "gguf|q4|co1|vram48g", 100.0),
            self._entry("w1", "m1", "gguf|q4|co1|vram48g", 200.0),
            self._entry("w1", "m1", "gguf|q4|co2|vram48g", 50.0),
        ]
        rows = TR.group_entries(entries)
        assert len(rows) == 2
        best = rows[0]
        assert best["config_key"] == "gguf|q4|co1|vram48g"
        assert best["n"] == 2
        assert best["mean"] == pytest.approx(150.0)
        assert best["p50"] == pytest.approx(150.0)

    def test_sorted_best_mean_first(self):
        entries = [
            self._entry("w1", "m1", "slow", 10.0),
            self._entry("w2", "m1", "fast", 300.0),
        ]
        rows = TR.group_entries(entries)
        assert [r["config_key"] for r in rows] == ["fast", "slow"]

    def test_filters_and_missing_config_key_groups_as_dash(self):
        entries = [
            {"worker_id": "w1", "model_key": "m1", "tok_s": 80.0},  # no state
            self._entry("w2", "m2", "x", 90.0),
        ]
        only = TR.group_entries(entries, worker="w1")
        assert len(only) == 1
        assert only[0]["config_key"] == "-"

    def test_non_positive_and_garbage_tok_s_skipped(self):
        entries = [
            self._entry("w", "m", "c", 0.0),
            self._entry("w", "m", "c", -5.0),
            {"worker_id": "w", "model_key": "m", "tok_s": "nope",
             "state": {"config_key": "c"}},
            self._entry("w", "m", "c", 42.0),
        ]
        rows = TR.group_entries(entries)
        assert len(rows) == 1
        assert rows[0]["n"] == 1
        assert rows[0]["mean"] == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# ④ config_key shape + record_serve_metrics stamps the state snapshot.
# ---------------------------------------------------------------------------

class TestConfigKey:
    def test_shape_is_backend_dtype_co_vram(self):
        assert W._config_key("gguf", "q4_k_m", 2, 48 << 30) == \
            "gguf|q4_k_m|co2|vram48g"

    def test_unknown_parts_degrade_to_question_marks_not_dropped(self):
        assert W._config_key(None, None, 0, None) == "?|?|co0|vram?"

    def test_vram_bucket_rounds_to_nearest_8gib(self):
        assert W._vram_bucket(24 << 30) == "24g"
        assert W._vram_bucket(int(23.9 * (1 << 30))) == "24g"
        assert W._vram_bucket(None) == "?"


class TestStateSnapshotOnRecord:
    def _worker(self, store):
        w = store.register(name="wk", url="http://127.0.0.1:9998",
                           worker_id="wid-snap", models=["tst/model-xyz"])
        store.set_admission(w["id"], "approved")
        store.heartbeat(
            w["id"], loaded_models=["tst/model-xyz"],
            gpus=[{"name": "L40S", "memory_total": 48 << 30,
                   "memory_free": 40 << 30}],
            pkg_version="9.9.9",
            storage={"disk_free": 500 << 30, "cache_used_bytes": 100 << 30})
        return w["id"]

    def test_record_stamps_a_state_object_with_config_key(self, toks_log):
        with swap_worker_store("hugpy-toks-snap-") as store:
            wid = self._worker(store)
            store.pick_for_model("tst/model-xyz")
            ok = store.record_serve_metrics(
                wid, "tst/model-xyz", tok_s=LIVE_TOK_S,
                prompt_tokens=33, completion_tokens=17, ttft_s=0.05,
                elapsed_s=0.2, inflight=1, streaming=False, ok=True,
                loaded_at_pick=True, max_tokens=128)
            assert ok

            entries = W.read_toks_log(worker_id=wid, limit=10)
            assert entries
            state = entries[0]["state"]
            # worker section off the STORED record (name/gpu/vram_total)
            assert state["worker"]["vram_total"] == 48 << 30
            assert state["worker"]["agent_version"] == "9.9.9"
            assert state["worker"]["gpu"] == "L40S"
            # vram + co-resident set
            assert state["vram"]["loaded_models"] == ["tst/model-xyz"]
            # request + outcome echoed from meta
            assert state["request"]["prompt_tokens"] == 33
            assert state["request"]["max_tokens"] == 128
            assert state["outcome"]["ok"] is True
            assert state["model"]["loaded_at_pick"] is True
            # config_key: unknown backend/dtype degrade, co1, 48 GiB bucket
            assert state["config_key"] == "?|?|co1|vram48g"

    def test_line_stays_small(self, toks_log):
        """<= ~2KB per line: a heavy state snapshot must not blow the JSONL."""
        import json
        with swap_worker_store("hugpy-toks-snap-") as store:
            wid = self._worker(store)
            store.pick_for_model("tst/model-xyz")
            store.record_serve_metrics(wid, "tst/model-xyz", tok_s=LIVE_TOK_S,
                                       prompt_tokens=33, completion_tokens=17)
            entries = W.read_toks_log(worker_id=wid, limit=10)
            line = json.dumps(entries[0], separators=(",", ":"))
            assert len(line) <= 2048
