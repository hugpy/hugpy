"""ctx ceiling per KV cache type (compute.ctx_range / memory_plan.ctx_max).

The ruling under test (operator 2026-10-02): a context range per cache
quantization, decided by the SAME budgets as the quant-to-worker verdict.
Pure arithmetic on a synthetic dense geometry — no DB, no files."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import compute  # noqa: E402
from hugpy_engine import spill as _spill  # noqa: E402

GIB = 1024 ** 3
# 40 layers, 8 kv heads x 128 head dim, f16 -> 2*40*8*128*2 = 163840 B/token
GEO = {"n_layers": 40, "n_kv_heads": 8, "head_dim": 128, "ctx_train": 262144}
BPT_F16 = 2 * 40 * 8 * 128 * 2


def test_ctx_range_scales_by_cache_type():
    left = 3 * GIB
    r = compute.ctx_range(left, GEO, floor=1024, trained=262144)
    assert set(r) == {"f16", "q8_0", "q4_0"}
    assert r["f16"] == _spill._round_down_multiple(left / BPT_F16, 1024)
    # q8_0 is 34/32 bytes/elem, q4_0 is 18/32: ceilings scale inversely
    assert r["f16"] < r["q8_0"] < r["q4_0"]
    assert abs(r["q8_0"] - _spill._round_down_multiple(left / (BPT_F16 * (34 / 32) / 2), 1024)) <= 1024
    assert abs(r["q4_0"] - _spill._round_down_multiple(left / (BPT_F16 * (18 / 32) / 2), 1024)) <= 1024
    for v in r.values():
        assert v % 1024 == 0 and v <= 262144


def test_ctx_range_caps_at_trained_and_floors_at_zero():
    huge = compute.ctx_range(10 ** 15, GEO, floor=1024, trained=262144)
    assert all(v == 262144 for v in huge.values())
    none = compute.ctx_range(100, GEO, floor=1024, trained=262144)
    assert all(v == 0 for v in none.values())
    assert compute.ctx_range(3 * GIB, None, floor=1024, trained=262144) == {"f16": 0, "q8_0": 0, "q4_0": 0}


def test_memory_plan_carries_ctx_max_per_mode():
    size = 20 * GIB
    b = {"gpu_budget": 24 * GIB, "ram_budget": 120 * GIB, "gpu_total": 24 * GIB, "ram_total": 124 * GIB,
         "vram_reserve": 0, "ram_reserve": 4 * GIB}
    plan = compute.memory_plan("gguf", {"size_bytes": size}, GEO, None, b, None)
    for mode in ("gpu-only", "ram-only", "max-gpu", "max-ram"):
        assert "ctx_max" in plan[mode], mode
        assert plan[mode]["kv_device"] == ("ram" if mode == "ram-only" else "gpu")
    g = plan["gpu-only"]
    # the derived ctx IS the f16 ceiling for the mode
    assert g["ctx"] == g["ctx_max"]["f16"]
    # a quantized cache raises the ceiling within the same GPU budget
    assert g["ctx_max"]["q4_0"] > g["ctx_max"]["q8_0"] > g["ctx_max"]["f16"] > 0
    # ram-only is priced against the RAM budget, so its ceiling is the trained ctx here
    assert plan["ram-only"]["ctx_max"]["f16"] == 262144
    # the ceiling never exceeds what the budget can hold
    compute_b = _spill._CTX_COMPUTE_RESERVE_BYTES
    for t, per in (("f16", 2.0), ("q8_0", 34 / 32), ("q4_0", 18 / 32)):
        kv = _spill.kv_bytes_for_geo(GEO, g["ctx_max"][t], per)
        assert size + kv + compute_b <= b["gpu_budget"]
