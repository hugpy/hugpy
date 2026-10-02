"""Encroachment in the planner's earmark (operator 2026-10-02): the worker's
gpu_budget is net of the measured external occupants (ComfyUI + foreign pids),
rounded up to 128 MiB; the driver share stays out via gpu_total."""
from hugpy_engine.model_index.planner import compute as C

MIB = 1 << 20
GIB = 1 << 30


def _payload(comfy_mib=94, foreign_mib=19, limit_gib=None):
    p = {"gpus": [{"memory_total": 8188 * MIB, "memory_reserved": 383 * MIB}],
         "ram_total": 16 * GIB,
         "pid_registry": {"models": [{"host_mode": "comfy", "vram_bytes": comfy_mib * MIB},
                                     {"host_mode": "subprocess", "model_key": "M", "vram_bytes": 3 * GIB},
                                     {"host_mode": "cuda_context", "vram_bytes": 94 * MIB}],
                          "unattributed": [{"pid": 1, "mib": foreign_mib}]}}
    if limit_gib is not None:
        p["limits"] = {"gpu_mem_gib": limit_gib}
    return p


def test_external_counts_comfy_and_foreign_only_rounded_up():
    assert C.external_vram_bytes(_payload()) == 128 * MIB          # 94 + 19 = 113 -> 128
    assert C.external_vram_bytes(_payload(comfy_mib=0, foreign_mib=0)) == 0
    assert C.external_vram_bytes({}) == 0


def test_budget_is_net_of_driver_and_encroachment(monkeypatch):
    monkeypatch.setattr(C._spill, "vram_reserve_bytes", lambda: 0)
    e = C.worker_earmark(C.worker_totals(_payload()))
    assert e["gpu_total"] == (8188 - 383) * MIB
    assert e["vram_encroach"] == 128 * MIB
    assert e["gpu_budget"] == (8188 - 383 - 128) * MIB


def test_limit_still_caps(monkeypatch):
    monkeypatch.setattr(C._spill, "vram_reserve_bytes", lambda: 0)
    e = C.worker_earmark(C.worker_totals(_payload(limit_gib=6)))
    assert e["gpu_budget"] == 6 * GIB
