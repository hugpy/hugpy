"""Planner card totals exclude the driver's reserved share (2026-10-02)."""
from hugpy_engine.model_index.planner.compute import worker_totals

GIB = 1 << 30
MIB = 1 << 20


def test_driver_reserve_is_not_budgetable():
    t = worker_totals({"gpus": [{"memory_total": 24 * GIB, "memory_reserved": 450 * MIB}]})
    assert t["gpu_total"] == 24 * GIB - 450 * MIB


def test_worker_without_the_field_keeps_raw_total():
    assert worker_totals({"gpus": [{"memory_total": 24 * GIB}]})["gpu_total"] == 24 * GIB


def test_multi_card_sums_usable():
    t = worker_totals({"gpus": [{"memory_total": 24 * GIB, "memory_reserved": 450 * MIB},
                                {"memory_total": 12 * GIB, "memory_reserved": 300 * MIB}]})
    assert t["gpu_total"] == 36 * GIB - 750 * MIB
