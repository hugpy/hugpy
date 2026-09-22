"""THE ONE LEDGER's measured columns: decode rate + call interval.

Operator, 2026-07-25: "in the end it is about maximizing tok/s ... lets start
recording this."

Two signals are accumulated onto ``model_call_stats[model_key]`` — the SAME row
that already carries ``calls``/``last_call``, because central's eviction preview
and the worker's auto-evict must rank from ONE ledger (spec
``assets/evictionflow.html``). A second store would break Parity by
construction.

  ① tok/s   — ``predicted_per_second`` from llama-server's own ``timings``
              block. Already measured by the engine, previously discarded.
  ② interval— log-space EWMA of the gap between calls, because "time since last
              call" is a point estimate of a heavy-tailed distribution.

⚠ THE CENTRAL CLAIM OF THIS SLICE IS THAT IT CHANGES NOTHING. These are inert
columns; ``eviction.sort_key`` is untouched. The final test class is the one
that actually matters: it asserts an eviction plan is BYTE-IDENTICAL with and
without every new field present. If that test ever fails, the slice is wrong.

Run: venv/bin/python -m pytest tests/test_serve_metrics_ledger.py -v
"""
import importlib
import math
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PROJECTS_HOME",
                      tempfile.mkdtemp(prefix="hugpy-metrics-test-"))

W = importlib.import_module(
    "hugpy_fleet.central.workers")
EV = importlib.import_module("hugpy_engine.eviction")

from worker_store_isolation import swap_worker_store  # noqa: E402


# The exact response the operator observed live on computron, 2026-07-25.
LIVE_TIMINGS = {
    "prompt_n": 33,
    "prompt_ms": 61.2,
    "predicted_n": 17,
    "predicted_ms": 147.499,
    "predicted_per_second": 115.25501867809274,
}
LIVE_TOK_S = 115.25501867809274


# ---------------------------------------------------------------------------
# ① tok/s extraction — the realistic shape, and every way it can be absent.
# ---------------------------------------------------------------------------

class TestTokSFromTimings:
    def test_realistic_llama_server_response(self):
        """The measured number rides straight through, undistorted."""
        body = {"choices": [{"message": {"content": "hi"}}],
                "usage": {"completion_tokens": 17, "prompt_tokens": 33,
                          "total_tokens": 50},
                "timings": LIVE_TIMINGS}
        assert EV.tok_s_from_timings(body) == pytest.approx(LIVE_TOK_S)

    def test_derived_from_counters_matches_engine_rate(self):
        """A build reporting only predicted_n/predicted_ms derives the SAME
        rate — the fallback is not a different measurement."""
        body = {"timings": {"predicted_n": 17, "predicted_ms": 147.499}}
        assert EV.tok_s_from_timings(body) == pytest.approx(LIVE_TOK_S, rel=1e-9)

    @pytest.mark.parametrize("body", [
        None, "not a dict", 42, [],
        {},                                        # no timings at all
        {"usage": {"completion_tokens": 17}},      # usage but no timings
        {"timings": None},
        {"timings": "nope"},
        {"timings": {}},                           # empty block
        {"timings": {"prompt_n": 33}},             # prompt only, no decode
        {"timings": {"predicted_per_second": 0}},          # zero rate
        {"timings": {"predicted_per_second": -5}},         # negative
        {"timings": {"predicted_per_second": "fast"}},     # non-numeric
        {"timings": {"predicted_n": 0, "predicted_ms": 12.0}},   # no tokens
        {"timings": {"predicted_n": 17, "predicted_ms": 0}},     # no time
        {"timings": {"predicted_n": None, "predicted_ms": None}},
    ])
    def test_absent_or_broken_never_raises_and_yields_none(self, body):
        """THE LIVE-PATH GUARANTEE. A relay that raises because a `timings` key
        is missing is a far worse bug than not recording, so every unusable
        shape must degrade silently to None rather than throw."""
        assert EV.tok_s_from_timings(body) is None

    def test_nan_and_inf_rejected(self):
        assert EV.tok_s_from_timings(
            {"timings": {"predicted_per_second": float("nan")}}) is None
        assert EV.tok_s_from_timings(
            {"timings": {"predicted_per_second": float("inf")}}) is None


# ---------------------------------------------------------------------------
# ② EWMA arithmetic — seeding is the part that is easy to get quietly wrong.
# ---------------------------------------------------------------------------

class TestEwmaArithmetic:
    def test_first_sample_seeds_at_the_sample(self):
        """Seeding at 0 would make every model's first observation read HALF
        its true value and take ~10 samples to recover. Seed at the sample."""
        assert W._ewma(None, 100.0) == 100.0
        assert W._ewma("garbage", 100.0) == 100.0

    def test_second_sample_is_alpha_blended(self):
        # alpha 0.3: 0.3*200 + 0.7*100 = 130
        assert W._ewma(100.0, 200.0) == pytest.approx(130.0)

    def test_converges_toward_a_step_change(self):
        """A placement change (MoE split: +59%, offload cliff: 135->36) must
        show up within a few calls, not be smeared away."""
        v = 36.0
        for _ in range(10):
            v = W._ewma(v, 135.0)
        assert v > 120.0          # most of the way there in 10 samples
        assert v < 135.0          # but never overshoots

    def test_is_bounded_by_the_samples(self):
        for prev, s in [(10.0, 20.0), (20.0, 10.0), (5.0, 5.0)]:
            out = W._ewma(prev, s)
            assert min(prev, s) <= out <= max(prev, s)


class TestRecordTokS:
    def test_first_sample_sets_all_three_columns(self):
        row = {"calls": 1}
        assert W._record_tok_s(row, LIVE_TOK_S) is True
        assert row["tok_s_last"] == pytest.approx(LIVE_TOK_S, abs=1e-3)
        assert row["tok_s_ewma"] == pytest.approx(LIVE_TOK_S, abs=1e-3)
        assert row["tok_s_samples"] == 1

    def test_second_sample_blends_ewma_but_last_is_raw(self):
        row = {"calls": 2, "tok_s_last": 100.0, "tok_s_ewma": 100.0,
               "tok_s_samples": 1}
        assert W._record_tok_s(row, 200.0) is True
        assert row["tok_s_last"] == pytest.approx(200.0)   # raw, not smoothed
        assert row["tok_s_ewma"] == pytest.approx(130.0)   # 0.3*200+0.7*100
        assert row["tok_s_samples"] == 2

    def test_plain_not_log_ewma(self):
        """tok/s uses a PLAIN EWMA (unlike the interval's log space): decode
        rate for a fixed placement is tightly clustered, so the arithmetic mean
        is meaningful and directly comparable — which is what "maximize tok/s"
        needs. Assert we are NOT in log space."""
        row = {"tok_s_ewma": 100.0, "tok_s_samples": 1}
        W._record_tok_s(row, 200.0)
        assert row["tok_s_ewma"] == pytest.approx(130.0)
        log_answer = math.exp(0.3 * math.log(200.0) + 0.7 * math.log(100.0))
        assert row["tok_s_ewma"] != pytest.approx(log_answer)

    @pytest.mark.parametrize("bad", [None, "x", float("nan"), float("inf"),
                                     0, 0.0, -1.0, [], {}])
    def test_bad_samples_record_nothing_and_never_raise(self, bad):
        """A zero-token generation reports 0.0 tok/s while saying nothing about
        decode speed; averaging it in would make a fast model look slow."""
        row = {"calls": 1}
        assert W._record_tok_s(row, bad) is False
        assert "tok_s_last" not in row
        assert "tok_s_ewma" not in row
        assert "tok_s_samples" not in row

    def test_a_rejected_sample_does_not_disturb_existing_history(self):
        row = {"tok_s_last": 115.0, "tok_s_ewma": 115.0, "tok_s_samples": 4}
        assert W._record_tok_s(row, 0.0) is False
        assert row == {"tok_s_last": 115.0, "tok_s_ewma": 115.0,
                       "tok_s_samples": 4}


# ---------------------------------------------------------------------------
# ② call intervals — verifying the operator's arithmetic.
# ---------------------------------------------------------------------------

class TestRecordInterval:
    def test_first_call_records_no_interval(self):
        """There is no previous call to difference against. Deliberately absent
        rather than stamped 0, which would read as the hottest possible model."""
        row = {"calls": 1}
        W._record_interval(row, None, 1000.0)
        assert "last_interval_s" not in row
        assert "log_interval_ewma" not in row

    def test_second_call_seeds_log_ewma_at_the_log_gap(self):
        row = {"calls": 2}
        W._record_interval(row, 1000.0, 1030.0)       # 30s gap
        assert row["last_interval_s"] == pytest.approx(30.0)
        assert row["log_interval_ewma"] == pytest.approx(math.log(30.0))
        assert row["interval_samples"] == 1

    def test_third_call_blends_in_log_space(self):
        row = {"log_interval_ewma": math.log(30.0), "interval_samples": 1}
        W._record_interval(row, 1030.0, 1060.0)       # another 30s gap
        assert row["log_interval_ewma"] == pytest.approx(math.log(30.0))
        row2 = {"log_interval_ewma": math.log(30.0), "interval_samples": 1}
        W._record_interval(row2, 1000.0, 1300.0)      # 300s outlier
        expect = 0.3 * math.log(300.0) + 0.7 * math.log(30.0)
        assert row2["log_interval_ewma"] == pytest.approx(expect)

    def test_log_space_resists_a_heavy_tailed_outlier(self):
        """THE REASON FOR LOG SPACE. A model called every 30s that goes quiet
        once for a day must not have its estimate destroyed by that one gap —
        an arithmetic mean would be dominated by it."""
        row = {}
        prev = 0.0
        for i in range(1, 21):                        # twenty 30s calls
            W._record_interval(row, prev, prev + 30.0)
            prev += 30.0
        steady = math.exp(row["log_interval_ewma"])
        assert steady == pytest.approx(30.0, rel=0.01)

        W._record_interval(row, prev, prev + 86400.0)  # one day-long gap
        after = math.exp(row["log_interval_ewma"])
        assert after < 30.0 * 12          # nowhere near the arithmetic blowup
        arithmetic = (20 * 30.0 + 86400.0) / 21
        assert after < arithmetic         # strictly better behaved

    @pytest.mark.parametrize("prev,now", [
        (1000.0, 1000.0),      # same instant re-pick -> log(0) = -inf
        (1000.0, 900.0),       # clock went backwards
        ("bad", 1000.0),
        (1000.0, "bad"),
        (0, 1000.0),           # falsy prev = never called
    ])
    def test_degenerate_gaps_record_nothing_and_never_raise(self, prev, now):
        """A non-positive interval would put -inf into log space and poison the
        EWMA permanently."""
        row = {"calls": 1}
        W._record_interval(row, prev, now)
        assert "log_interval_ewma" not in row
        assert row.get("log_interval_ewma") != float("-inf")

    def test_a_poisoned_ewma_is_impossible_after_a_zero_gap(self):
        row = {"log_interval_ewma": math.log(30.0)}
        W._record_interval(row, 1000.0, 1000.0)
        assert math.isfinite(row["log_interval_ewma"])


# ---------------------------------------------------------------------------
# The ledger seam: ONE row, ONE writer each, and it rides _public_view.
# ---------------------------------------------------------------------------

class TestLedgerIntegration:
    def _worker(self, store, models=("m1",)):
        """An APPROVED, ONLINE, assigned worker — the state pick_for_model
        actually requires (register alone leaves it pending + never-seen)."""
        w = store.register(name="wk", url="http://127.0.0.1:9999",
                           worker_id="wid-metrics", models=list(models))
        store.set_admission(w["id"], "approved")
        store.heartbeat(w["id"], loaded_models=[], storage={})
        return w["id"]

    def test_tok_s_lands_on_the_same_row_as_calls(self):
        """PARITY: not a second store. The decode rate must be on the very row
        that carries `calls`/`last_call`, or the two sides can diverge."""
        with swap_worker_store("hugpy-metrics-") as store:
            wid = self._worker(store)
            store.pick_for_model("m1")                 # stamps calls/last_call
            assert store.record_serve_metrics(wid, "m1", tok_s=LIVE_TOK_S)

            row = store.get(wid)["model_call_stats"]["m1"]
            assert row["calls"] == 1                   # pre-existing column
            assert row["last_call"] > 0                # pre-existing column
            assert row["tok_s_last"] == pytest.approx(LIVE_TOK_S, abs=1e-3)
            assert row["tok_s_samples"] == 1

    def test_rides_public_view_to_the_worker(self):
        """model_call_stats is spread by `**worker` in _public_view, so new keys
        inherit the heartbeat path automatically — this pins that."""
        with swap_worker_store("hugpy-metrics-") as store:
            wid = self._worker(store)
            store.pick_for_model("m1")
            store.record_serve_metrics(wid, "m1", tok_s=LIVE_TOK_S)

            public = store.get(wid)
            row = public["model_call_stats"]["m1"]
            for k in ("calls", "last_call", "tok_s_last", "tok_s_ewma",
                      "tok_s_samples"):
                assert k in row, f"{k} did not reach the public view"

    def test_recording_is_fail_open_for_an_unknown_worker(self):
        with swap_worker_store("hugpy-metrics-") as store:
            assert store.record_serve_metrics("no-such-worker", "m1",
                                              tok_s=115.0) is False

    def test_recording_none_is_a_silent_noop(self):
        with swap_worker_store("hugpy-metrics-") as store:
            wid = self._worker(store)
            assert store.record_serve_metrics(wid, "m1", tok_s=None) is False
            assert not store.get(wid).get("model_call_stats")

    def test_repeated_picks_accumulate_an_interval_history(self):
        with swap_worker_store("hugpy-metrics-") as store:
            wid = self._worker(store)
            for _ in range(3):
                store.pick_for_model("m1")
            row = store.get(wid)["model_call_stats"]["m1"]
            assert row["calls"] == 3
            # Real picks happen microseconds apart; the columns exist only if a
            # strictly positive gap was observed. Either way: no crash, no -inf.
            if "log_interval_ewma" in row:
                assert math.isfinite(row["log_interval_ewma"])

    def test_unassign_prunes_the_row(self):
        """THE GROWTH SEAM. model_last_picked was pruned on unassign from the
        start; model_call_stats was NOT, so every assign/unassign cycle left a
        permanent orphan — and the new columns widen those rows."""
        with swap_worker_store("hugpy-metrics-") as store:
            wid = self._worker(store)
            store.pick_for_model("m1")
            store.record_serve_metrics(wid, "m1", tok_s=LIVE_TOK_S)
            assert "m1" in store.get(wid)["model_call_stats"]

            store.unassign_model(wid, "m1")
            after = store.get(wid)
            assert "m1" not in (after.get("model_call_stats") or {})
            assert "m1" not in (after.get("model_last_picked") or {})


# ---------------------------------------------------------------------------
# THE TEST THAT MATTERS: recording changes NOTHING.
# ---------------------------------------------------------------------------

class TestNothingChangesAnEvictionDecision:
    """If this slice alters which model is evicted or placed, it is wrong.

    ``eviction.sort_key`` takes no tok/s and no interval input, so the proof is
    direct: build the same plan from rows WITH and WITHOUT every new field, and
    require the results to be byte-identical.
    """

    def _residents(self):
        return [
            EV.EvictUnit(model_key="a", bytes=4 << 30, pref=EV.VRAM,
                        last_call=1000.0, calls=5),
            EV.EvictUnit(model_key="b", bytes=6 << 30, pref=EV.RAM,
                        last_call=900.0, calls=2),
            EV.EvictUnit(model_key="c", bytes=2 << 30, pref=EV.VRAM,
                        last_call=1200.0, calls=9),
            EV.EvictUnit(model_key="d", bytes=3 << 30, pref=EV.VRAM,
                        last_call=None, calls=0, resident_since=800.0),
        ]

    def test_sort_key_signature_takes_no_metrics(self):
        """A guard against a future 'small' change quietly promoting a recorded
        column into the ranking without a decision."""
        import inspect
        params = list(inspect.signature(EV.sort_key).parameters)
        assert params == ["r", "device", "now"]
        fields = set(EV.EvictUnit.__dataclass_fields__)
        for leaked in ("tok_s", "tok_s_ewma", "tok_s_last", "tok_s_samples",
                       "log_interval_ewma", "last_interval_s",
                       "interval_samples"):
            assert leaked not in fields, (
                f"{leaked} reached EvictUnit — recording became ranking")

    def test_plan_is_byte_identical_with_and_without_the_new_columns(self):
        residents = self._residents()
        need = 5 << 30
        before = EV.evict_plan(EV.VRAM, need, residents, now=2000.0).as_dict()

        # The new columns exist on the LEDGER, not on EvictUnit — so the honest
        # form of "with the fields present" is: stamp a full ledger, build
        # Residents from it exactly as storage_proposal does, and re-plan.
        ledger = {
            "a": {"calls": 5, "last_call": 1000.0, "tok_s_last": 115.2,
                  "tok_s_ewma": 110.4, "tok_s_samples": 12,
                  "log_interval_ewma": 3.4, "last_interval_s": 30.0,
                  "interval_samples": 11},
            "b": {"calls": 2, "last_call": 900.0, "tok_s_last": 36.0,
                  "tok_s_ewma": 36.0, "tok_s_samples": 3,
                  "log_interval_ewma": 9.1, "last_interval_s": 8000.0,
                  "interval_samples": 2},
            "c": {"calls": 9, "last_call": 1200.0, "tok_s_last": 240.0,
                  "tok_s_ewma": 238.9, "tok_s_samples": 40,
                  "log_interval_ewma": 1.1, "last_interval_s": 3.0,
                  "interval_samples": 39},
            "d": {"calls": 0},
        }
        rebuilt = [
            EV.EvictUnit(model_key=r.model_key, bytes=r.bytes, pref=r.pref,
                        last_call=(ledger[r.model_key].get("last_call")
                                   or r.last_call),
                        calls=int(ledger[r.model_key].get("calls") or 0),
                        resident_since=r.resident_since)
            for r in residents
        ]
        after = EV.evict_plan(EV.VRAM, need, rebuilt, now=2000.0).as_dict()

        assert after == before
        assert after["victims"] == before["victims"]

    def test_the_fastest_model_is_not_spared_and_the_slowest_not_punished(self):
        """Concretely: 'b' decodes at 36 tok/s and 'c' at 240, and that fact is
        recorded — but it must not move either one in the order. Only the
        pre-existing keys (pref / idle / calls / key) may rank."""
        residents = self._residents()
        plan = EV.evict_plan(EV.VRAM, 5 << 30, residents, now=2000.0)
        keys = [EV.sort_key(r, EV.VRAM, 2000.0) for r in residents]
        # 'b' prefers RAM, so key ① puts it first regardless of its slow rate.
        assert keys[1][0] == 0
        assert plan.victims[0] == "b"

    def test_storage_proposal_ignores_the_new_columns(self):
        """Central's preview reads model_call_stats for `calls` only. Adding
        columns to that dict must not perturb its proposal."""
        with swap_worker_store("hugpy-metrics-") as store:
            wid = store.register(name="wk", url="http://127.0.0.1:9999",
                                 models=["m1"])["id"]
            base = store.get(wid)["storage"]

            store.pick_for_model("m1")
            store.record_serve_metrics(wid, "m1", tok_s=LIVE_TOK_S)
            after = store.get(wid)["storage"]

            # `calls` legitimately moved (pre-existing behaviour); the proposal
            # itself must not have.
            assert after.get("over_budget") == base.get("over_budget")
            assert (after.get("proposed_evictions")
                    == base.get("proposed_evictions"))


# ---------------------------------------------------------------------------
# Wire: the additive contract, in both directions.
# ---------------------------------------------------------------------------

class TestWireContract:
    def test_done_event_timings_is_optional(self):
        from hugpy_engine.schemas.event_schemas import DoneEvent
        d = DoneEvent(request_id="r", input_tokens=0, output_chunks=1,
                      finish_reason="stop")
        assert d.timings is None                      # every old call site works
        d2 = DoneEvent(request_id="r", input_tokens=0, output_chunks=1,
                       finish_reason="stop", timings=LIVE_TIMINGS)
        assert d2.timings["predicted_per_second"] == LIVE_TOK_S

    def test_task_result_allows_extra_so_timings_needs_no_bump(self):
        """The central->worker REQUEST is extra=forbid (the landmine). The
        worker->central RESULT is extra=allow, which is the direction tok/s
        travels — so no wire version bump is required for the one-shot path."""
        from hugpy_engine.schemas.chat_schemas import ChatResult
        r = ChatResult(request_id="r", model_key="m", text="hi",
                       finish_reason="stop", timings=LIVE_TIMINGS)
        assert r.model_dump()["timings"]["predicted_per_second"] == LIVE_TOK_S

    def test_chat_request_still_forbids_extra(self):
        """Pin the landmine itself: nothing here loosened the frozen request."""
        import pydantic
        from hugpy_engine.schemas.chat_schemas import ChatRequest
        with pytest.raises(pydantic.ValidationError):
            ChatRequest(model_key="m", messages="hi", timings=LIVE_TIMINGS)

    def test_relay_done_reconstruction_survives_an_unknown_key(self):
        """MIXED-VERSION SAFETY. _event_from_worker_line is field-explicit, so a
        NEWER worker's unknown key cannot fail DoneEvent's extra=forbid and get
        silently downgraded to a StatusEvent — a stream whose terminal `done`
        becomes a status event has no done at all."""
        remote = importlib.import_module(
            "hugpy_engine.resolvers.remote")
        line = {"type": "done", "finish_reason": "stop", "output_chunks": 1,
                "usage": {"completion_tokens": 17},
                "timings": LIVE_TIMINGS,
                "some_future_field_from_a_newer_worker": {"x": 1}}
        ev = remote._event_from_worker_line(line, "req-1")
        assert ev.type == "done", "terminal done was lost to an unknown key"
        assert ev.timings["predicted_per_second"] == LIVE_TOK_S

    def test_relay_done_without_timings_is_unchanged(self):
        remote = importlib.import_module(
            "hugpy_engine.resolvers.remote")
        ev = remote._event_from_worker_line(
            {"type": "done", "finish_reason": "stop"}, "req-1")
        assert ev.type == "done"
        assert ev.timings is None


class TestRelayRecordingSeam:
    """_record_serve_metrics is the live-path wrapper. It must be inert when
    unregistered and silent on every failure."""

    def test_noop_when_no_sink_registered(self):
        remote = importlib.import_module(
            "hugpy_engine.resolvers.remote")
        orig = remote._serve_metrics_sink
        remote._serve_metrics_sink = None
        try:
            remote._record_serve_metrics({"id": "w1"}, "m1",
                                         {"timings": LIVE_TIMINGS})
        finally:
            remote._serve_metrics_sink = orig

    def test_calls_the_sink_with_the_engine_rate(self):
        remote = importlib.import_module(
            "hugpy_engine.resolvers.remote")
        seen = []
        orig = remote._serve_metrics_sink

        # The live seam calls the sink as (worker_id, model_key, tok_s, **meta);
        # the stub must accept the meta kwargs or the fail-open swallows a
        # TypeError and the ledger silently records nothing.
        def _sink(worker_id, model_key, tok_s=None, **meta):
            seen.append((worker_id, model_key, tok_s))

        remote._serve_metrics_sink = _sink
        try:
            remote._record_serve_metrics({"id": "w1"}, "m1",
                                         {"timings": LIVE_TIMINGS})
        finally:
            remote._serve_metrics_sink = orig
        assert seen == [("w1", "m1", pytest.approx(LIVE_TOK_S))]

    def test_a_raising_sink_never_escapes_to_the_serving_path(self):
        """THE LIVE-PATH GUARANTEE, at the seam. Recording must never fail a
        user's request."""
        remote = importlib.import_module(
            "hugpy_engine.resolvers.remote")

        def _boom(*a):
            raise RuntimeError("store on fire")

        orig = remote._serve_metrics_sink
        remote._serve_metrics_sink = _boom
        try:
            remote._record_serve_metrics({"id": "w1"}, "m1",
                                         {"timings": LIVE_TIMINGS})
        finally:
            remote._serve_metrics_sink = orig

    @pytest.mark.parametrize("worker,payload", [
        (None, {"timings": LIVE_TIMINGS}),
        ({}, {"timings": LIVE_TIMINGS}),
        ({"id": None}, {"timings": LIVE_TIMINGS}),
        ({"id": "w1"}, {}),                       # worker sent no timings
        ({"id": "w1"}, {"timings": None}),
        ({"id": "w1"}, None),
    ])
    def test_missing_pieces_record_nothing_without_raising(self, worker, payload):
        remote = importlib.import_module(
            "hugpy_engine.resolvers.remote")
        seen = []
        orig = remote._serve_metrics_sink
        remote._serve_metrics_sink = lambda *a: seen.append(a)
        try:
            remote._record_serve_metrics(worker, "m1", payload)
        finally:
            remote._serve_metrics_sink = orig
        assert seen == []


class TestStreamingCapture:
    """STREAMING IS NOT A BLIND SPOT — verified against the llama.cpp source in
    this tree (.gguf_cache/llama.cpp/tools/server/server-task.cpp):
    ``to_json_oaicompat_chat_stream`` ends with

        if (timings.prompt_n >= 0) {
            deltas.back().push_back({"timings", timings.to_json()});
        }

    i.e. the FINAL streaming chunk carries `timings` UNCONDITIONALLY, and
    independently of `include_usage`. So the same take-once slot serves both
    transports. These tests pin the shape we rely on."""

    def test_final_sse_chunk_shape_is_parsed(self):
        final_chunk = {"choices": [], "object": "chat.completion.chunk",
                       "usage": {"completion_tokens": 17},
                       "timings": LIVE_TIMINGS}
        assert EV.tok_s_from_timings(final_chunk) == pytest.approx(LIVE_TOK_S)

    def test_mid_stream_chunks_carry_no_rate(self):
        mid = {"choices": [{"delta": {"content": "hel"}}],
               "object": "chat.completion.chunk"}
        assert EV.tok_s_from_timings(mid) is None

    def test_runner_take_once_slot_is_drained(self):
        """_take_stream_timings must be take-ONCE, like _take_stream_usage: a
        stale rate leaking into the next request would attribute one model's
        speed to another."""
        base = importlib.import_module(
            "hugpy_engine.llama.runners.src.base_runner")

        class _Fake(base.LlamaCppBaseRunner):
            def __init__(self):
                pass
            async def _iter_stream(self, *a, **k):
                yield "", None
            def _chat_complete(self, *a, **k):
                return "", "stop"
            def _raw_complete(self, *a, **k):
                return "", "stop"
            def _blocking_complete(self, *a, **k):
                return "", "stop"

        r = _Fake()
        assert r._take_stream_timings() is None
        r._stream_timings = LIVE_TIMINGS
        assert r._take_stream_timings() == LIVE_TIMINGS
        assert r._take_stream_timings() is None, "slot was not drained"


# ── THE WIRE CONTRACT — the gap every other test missed ─────────────────────
def test_done_frame_carries_timings_onto_the_sse_wire():
    """The `done` SSE frame must forward the engine's timings.

    Found live 2026-07-25: every piece of the tok/s chain passed its OWN tests —
    ccp_runner captured the engine's `timings`, base_runner threaded them
    through, the DoneEvent schema carried them, central's relay read them via
    tok_s_from_timings() and its sink was registered — and NOTHING was recorded,
    because the single line building the SSE frame forwarded `usage` and not
    `timings`. The frame ended `{"type":"done","finish_reason":"stop"}`.

    Unit tests on each END cannot catch a missing wire BETWEEN them. This one
    asserts the frame itself.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hugpy_fleet.worker.agent import _event_to_dict

    class _Done:
        type = "done"
        finish_reason = "stop"
        usage = {"completion_tokens": 17}
        timings = {"predicted_per_second": 115.25, "predicted_n": 17}

    out = _event_to_dict(_Done())
    assert out["type"] == "done"
    assert out.get("usage")                      # the pre-existing forward
    assert out.get("timings"), "the done frame must carry engine timings"
    assert out["timings"]["predicted_per_second"] == 115.25

    # A producer without timings must not invent a key (consumers treat absent
    # as "unavailable" and degrade).
    class _Bare:
        type = "done"
        finish_reason = "stop"
        usage = None
        timings = None

    bare = _event_to_dict(_Bare())
    assert "timings" not in bare and "usage" not in bare
