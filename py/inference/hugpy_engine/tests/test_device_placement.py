"""Per-GPU-DEVICE placement — hugpy_engine.resolvers.device_placement.

The box-sum hides a hard physical fact: a diffusers/comfy/transformers PIPELINE
lives on ONE device, while a llama.cpp GGUF can tensor-split across the box's
cards. This module makes the per-device decision. The headline case (operator
2026-09-25): on a-brain (4x 24 GiB = 96 GiB), a 30 GiB DIFFUSERS model must NOT
be admitted to the GPU (no single card holds it), but a 30 GiB GGUF CAN — split
across the four cards.

Run both ways:
    PYTHONPATH=$(ls -d /srv/hugpy/src/hugpy/py/*/*/src | tr '\n' :) \
        python3 -m pytest py/inference/hugpy_engine/tests/test_device_placement.py -q
    python3 py/inference/hugpy_engine/tests/test_device_placement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib  # noqa: E402

dp = importlib.import_module("hugpy_engine.resolvers.device_placement")

GIB = 1 << 30


def _four_by_24(free_each=24 * GIB):
    """a-brain: four 24 GiB cards, all free."""
    return [dp.GpuDevice(index=i, free=free_each, total=24 * GIB) for i in range(4)]


def _single_24(free=24 * GIB):
    return [dp.GpuDevice(index=0, free=free, total=24 * GIB)]


# ── the headline 4x24 case ──────────────────────────────────────────────────
def test_30g_diffusers_not_admitted_to_gpu_on_4x24():
    """Non-splittable: 30 GiB pipeline fits NO single 24 GiB card -> none (the
    caller derives RAM/CPU). The box-sum of 96 GiB must NOT admit it."""
    plan = dp.plan_device_placement(30 * GIB, _four_by_24(),
                                    splittable=False, headroom=1.0)
    assert plan.kind == "none", plan
    assert not plan.ok
    assert "non-splittable" in plan.reason


def test_30g_gguf_splits_across_the_four_cards_on_4x24():
    """Splittable: the same 30 GiB GGUF tensor-splits across the cards."""
    plan = dp.plan_device_placement(30 * GIB, _four_by_24(),
                                    splittable=True, headroom=1.0)
    assert plan.kind == "split", plan
    # fewest cards, largest-first: 30 GiB needs 2 of the 24 GiB cards.
    assert plan.devices == (0, 1), plan.devices
    assert plan.main_gpu == 0
    assert len(plan.tensor_split) == 2
    assert abs(sum(plan.tensor_split) - 1.0) < 1e-6


def test_capacity_non_splittable_is_largest_single_card():
    devs = _four_by_24()
    assert dp.whole_pipeline_gpu_capacity(devs, splittable=False) == 24 * GIB
    assert dp.whole_pipeline_gpu_capacity(devs, splittable=True) == 96 * GIB


# ── single-device best-fit ──────────────────────────────────────────────────
def test_single_best_fit_picks_smallest_that_fits():
    """Best-fit packing: leave the big card free for a job that needs it."""
    devs = [dp.GpuDevice(0, free=8 * GIB, total=8 * GIB),
            dp.GpuDevice(1, free=24 * GIB, total=24 * GIB),
            dp.GpuDevice(2, free=12 * GIB, total=12 * GIB)]
    plan = dp.plan_device_placement(10 * GIB, devs, splittable=False, headroom=1.0)
    assert plan.kind == "single"
    assert plan.devices == (2,) and plan.main_gpu == 2, plan   # 12 fits, 8 doesn't, 24 is bigger


def test_a_model_that_fits_a_single_card_never_splits_even_if_splittable():
    plan = dp.plan_device_placement(20 * GIB, _four_by_24(),
                                    splittable=True, headroom=1.0)
    assert plan.kind == "single" and plan.devices == (0,), plan


# ── single-GPU boxes are byte-identical to the old box test ─────────────────
def test_single_gpu_box_fits():
    plan = dp.plan_device_placement(20 * GIB, _single_24(),
                                    splittable=False, headroom=1.0)
    assert plan.kind == "single" and plan.main_gpu == 0


def test_single_gpu_box_does_not_fit():
    plan = dp.plan_device_placement(30 * GIB, _single_24(),
                                    splittable=False, headroom=1.0)
    assert plan.kind == "none"


def test_single_gpu_capacity_equals_the_card():
    devs = _single_24()
    assert dp.whole_pipeline_gpu_capacity(devs, splittable=False) == 24 * GIB
    assert dp.whole_pipeline_gpu_capacity(devs, splittable=True) == 24 * GIB


# ── headroom is applied ─────────────────────────────────────────────────────
def test_headroom_can_tip_a_card_over():
    # 20 GiB * 1.15 = 23 GiB fits a 24 GiB card; * 1.3 = 26 GiB does not.
    devs = _single_24()
    assert dp.plan_device_placement(20 * GIB, devs, splittable=False,
                                    headroom=1.15).kind == "single"
    assert dp.plan_device_placement(20 * GIB, devs, splittable=False,
                                    headroom=1.3).kind == "none"


# ── degrade-not-guess ───────────────────────────────────────────────────────
def test_no_measured_device_is_none_not_zero():
    plan = dp.plan_device_placement(1 * GIB, [dp.GpuDevice(0, free=None, total=None)],
                                    splittable=False)
    assert plan.kind == "none"
    assert dp.whole_pipeline_gpu_capacity(
        [dp.GpuDevice(0, free=None, total=None)], splittable=False) is None


def test_devices_from_gpus_reads_heartbeat_shape():
    gpus = [{"index": 0, "memory_free": 5 * GIB, "memory_total": 24 * GIB},
            {"index": 1, "memory_free": 24 * GIB, "memory_total": 24 * GIB},
            "junk", {"memory_free": 1 * GIB, "memory_total": 8 * GIB}]  # no index -> pos
    devs = dp.devices_from_gpus(gpus)
    assert [d.index for d in devs] == [0, 1, 3]        # junk skipped; last uses pos 3
    plan = dp.plan_device_placement(20 * GIB, devs, splittable=False, headroom=1.0)
    assert plan.kind == "single" and plan.devices == (1,)   # only card 1 has 20 free


# ── plain-script runner ─────────────────────────────────────────────────────
def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    ok = fail = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[FAIL] {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            ok += 1
            print(f"[ok]   {t.__name__}")
    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(_main())
