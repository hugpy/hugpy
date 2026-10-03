"""Phantom in-process residents (2026-10-02 test fire): a dropped in-process
model's pid-registry row must not survive, or the worker PID's VRAM is split
across models long gone and the admission 'evicts' nothing."""
from hugpy_fleet.worker import agent as A
from hugpy_fleet.worker import pid_registry as P


def test_forget_absent_drops_only_that_host_mode():
    r = P.PidRegistry() if hasattr(P, "PidRegistry") else None
    if r is None:
        import pytest
        pytest.skip("no registry class")
    r.record_launch("gone", 1, "in_process")
    r.record_launch("held", 1, "in_process")
    r.record_launch("slot-model", 2, "subprocess")
    assert r.forget_absent("in_process", {"held"}) == ["gone"]
    keys = set(r._records)
    assert keys == {"held", "slot-model"}


def test_vram_residents_skips_inprocess_rows_not_held(monkeypatch):
    monkeypatch.setattr(A, "_inprocess_gpu_bytes", lambda: {"held": {"vram_bytes": 1, "gpu_index": 0}})
    monkeypatch.setattr(P, "snapshot_for_heartbeat", lambda: {"models": [
        {"model_key": "held", "vram_bytes": 3 << 30, "host_mode": "in_process"},
        {"model_key": "gone", "vram_bytes": 3 << 30, "host_mode": "in_process"}]})
    monkeypatch.setattr(A, "_slot_statuses", lambda: [])
    keys = [r["model_key"] for r in A._vram_residents(A.WorkerState(name="t", url=None, worker_id="w"))]
    assert "held" in keys and "gone" not in keys


def test_vram_residents_carry_measured_ram(monkeypatch):
    G = 1 << 30
    monkeypatch.setattr(A, "_inprocess_gpu_bytes",
                        lambda: {"cpu-model": {"vram_bytes": 0, "cpu_bytes": 60 * G, "device": "cpu", "gpu_index": None}})
    monkeypatch.setattr(P, "snapshot_for_heartbeat", lambda: {"models": [
        {"model_key": "cpu-model", "vram_bytes": 0, "host_mode": "in_process"}]})
    monkeypatch.setattr(A, "_slot_statuses", lambda: [])
    rows = {r["model_key"]: r for r in A._vram_residents(A.WorkerState(name="t", url=None, worker_id="w"))}
    assert rows["cpu-model"].get("ram_bytes") == 60 * G
