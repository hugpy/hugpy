"""chaos sweep (k7): the sweep-point math (grid / ceiling / budget clamp / cliff
detection), the top-N selection + ranking evidence, and the snapshot->restore
zero-drift discipline of a whole-model sweep on the in-memory fake fleet."""
from __future__ import annotations

import json
from pathlib import Path

from chaos_fakes import GIB, FakeClient

from hugpy_ops.chaos import sweep
from hugpy_ops.chaos.schema import validate_observation


def test_median():
    assert sweep.median([12.0]) == 12.0
    assert sweep.median([10.0, 20.0]) == 15.0
    assert sweep.median([5.0, 100.0, 9.0]) == 9.0
    assert sweep.median([None, 8.0, None]) == 8.0
    assert sweep.median([]) is None


def test_ceiling_pct():
    assert sweep.ceiling_pct(2.3, 5.0) == 100
    assert sweep.ceiling_pct(5.0, 5.0) == 100
    assert sweep.ceiling_pct(45.0, 16.0) == int(100 * 16 / 45)


def test_sweep_points_grids():
    assert sweep.sweep_points(3.0, 6.0, big_model_gib=20.0) == [100, 85, 70, 55, 40, 25]
    assert sweep.sweep_points(30.0, 40.0, big_model_gib=20.0) == [100, 70, 40]
    overs = sweep.sweep_points(45.0, 16.0, big_model_gib=20.0)
    assert len(overs) >= 2 and overs[0] == sweep.ceiling_pct(45.0, 16.0)
    assert overs == sorted(overs, reverse=True)
    assert all(a > b for a, b in zip(overs, overs[1:]))


def test_point_budget_clamps_to_safe_cap():
    assert sweep.point_budget_gib(70, 4.0, 10.0) == (2.8, False)
    assert sweep.point_budget_gib(100, 20.0, 12.0) == (12.0, True)


def test_detect_cliff():
    curve = [{"vram_share_pct": 100, "tokens_per_s": 40.0},
             {"vram_share_pct": 85, "tokens_per_s": 38.0},
             {"vram_share_pct": 70, "tokens_per_s": 36.0},
             {"vram_share_pct": 55, "tokens_per_s": 6.0},
             {"vram_share_pct": 40, "tokens_per_s": 5.0}]
    cl = sweep.detect_cliff(curve)
    assert cl["cliff"] and cl["from_pct"] == 70 and cl["to_pct"] == 55
    assert abs(cl["rel_drop"] - (36 - 6) / 36) < 1e-3
    cl2 = sweep.detect_cliff([{"vram_share_pct": 100, "tokens_per_s": 30.0},
                              {"vram_share_pct": 50, "tokens_per_s": None}])
    assert cl2["cliff"] and cl2["to_pct"] == 50 and cl2["rel_drop"] == 1.0
    cl3 = sweep.detect_cliff([{"vram_share_pct": 100, "tokens_per_s": 20.0},
                              {"vram_share_pct": 40, "tokens_per_s": 19.5}])
    assert cl3["rel_drop"] < 0.1


def test_rank_targets_selection_and_exclusions():
    c = FakeClient()
    sel = sweep.rank_targets(c.models(), c.workers(), top_n=10)
    assert [t["model_key"] for t in sel["chosen"]] == ["small-gguf"]
    sg = sel["chosen"][0]
    assert sg["worker"] == "computron"
    assert sg["rank"] == 1 and sg["last_picked"] == 1000.0
    excl = {e["model_key"]: e["reason"] for e in sel["excluded"]}
    assert excl.get("huge-gguf") == "weights-exceed-hybrid"
    assert excl.get("tf-model") == "not-gguf"
    assert all(t["worker"] != "op" for t in sel["chosen"])


def test_rank_targets_orders_by_last_picked():
    models = [
        {"model_key": "hot", "framework": "gguf", "effective_bytes": 3 * GIB,
         "size_bytes": 3 * GIB, "model_max_length": 8192,
         "primary_task": "text-generation", "tasks": ["text-generation"], "blocked": False},
        {"model_key": "cold", "framework": "gguf", "effective_bytes": 3 * GIB,
         "size_bytes": 3 * GIB, "model_max_length": 8192,
         "primary_task": "text-generation", "tasks": ["text-generation"], "blocked": False}]
    workers = [{"id": "wid-ae", "name": "ae", "status": "online",
                "vram_total": 24 * GIB, "ram_total": 128 * GIB, "vram_free": 20 * GIB,
                "models": ["hot", "cold"], "loaded_models": [],
                "model_last_picked": {"hot": 5000.0, "cold": 1000.0},
                "spill_by_model": {}}]
    sel = sweep.rank_targets(models, workers, top_n=10)
    assert [t["model_key"] for t in sel["chosen"]] == ["hot", "cold"]


def test_whole_model_sweep_restores_with_zero_drift(tmp_path):
    c = FakeClient(materialize_alloc={"computron": {
        "kind": "slot", "vram_bytes": 2 * GIB, "rss_bytes": 3 * GIB,
        "n_gpu_layers": -1, "total_layers": 28, "ctx": 4096, "serving": False}})
    runner = sweep.SweepRunner(
        c, top_n=10, ctx_pct=50, budget_minutes=90, out_dir=str(tmp_path),
        max_new_tokens=8, warmup_tokens=4, timed_runs=2, chat_ceiling_s=30,
        headroom_gib=1.0, vram_safety_frac=1.0, assign_settle_s=0.0, settle_s=0.0,
        big_model_gib=20.0, worker_filter=None)
    runner._started = sweep.time.time()
    runner._budget_hit = False
    target = {"model_key": "small-gguf", "framework": "gguf",
              "effective_bytes": 2 * GIB, "worker": "computron",
              "worker_id": "wid-comp", "vram_total": 8 * GIB, "ram_total": 16 * GIB,
              "last_picked": 1000.0, "rank": 1}
    res = runner.sweep_model(target)
    assert res["points_fired"] >= 1
    assert all(validate_observation(o) == [] for o in runner.point_records)
    fired = [o for o in runner.point_records if o.get("kind") != "skip"]
    assert fired
    assert all(o["sweep"]["vram_share_pct"] is not None
               and o["sweep"]["requested_gib"] is not None for o in fired)
    assert fired[0]["sweep"]["actual_n_gpu_layers"] == -1
    assert fired[0]["sweep"]["actual_gpu_pct"] == 100
    assert len(c.unload_calls) >= len(fired)
    w_after = {x["name"]: x for x in c.workers()}
    assert w_after["computron"]["spill_by_model"]["small-gguf"] == {"n_gpu_layers": -1}
    assert runner.restore_ledger and runner.restore_ledger[0]["ok"] is True
    obs_lines = Path(runner.obs_path).read_text().strip().splitlines()
    assert len(obs_lines) == len(runner.point_records)
    assert all(json.loads(ln).get("sweep") is not None for ln in obs_lines)
