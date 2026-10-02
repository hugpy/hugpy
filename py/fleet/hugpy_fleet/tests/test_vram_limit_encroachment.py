"""VRAM admission free = the budget-bar remaining (2026-10-02):
min(device free, central GPU limit - worker usage)."""
from hugpy_fleet.worker import agent as A

GIB = 1 << 30


def test_no_limit_is_device_free(monkeypatch):
    monkeypatch.delitem(A._RUNTIME_SETTINGS, "gpu_limit_bytes", raising=False)
    assert A._limit_capped_free(5 * GIB, 2 * GIB) == 5 * GIB


def test_limit_below_card_binds(monkeypatch):
    monkeypatch.setitem(A._RUNTIME_SETTINGS, "gpu_limit_bytes", 6 * GIB)
    assert A._limit_capped_free(10 * GIB, 4 * GIB) == 2 * GIB
    assert A._limit_capped_free(1 * GIB, 4 * GIB) == 1 * GIB       # device free tighter
    assert A._limit_capped_free(10 * GIB, 7 * GIB) == 0            # over the limit
    assert A._limit_capped_free(10 * GIB, None) == 10 * GIB        # unmeasured usage


def test_central_limit_recorded_apart_from_the_spill_env(monkeypatch):
    monkeypatch.setitem(A._LOCAL_CAP_ENV, "gpu_mem_gib", None)
    A._apply_central_limits({"limits": {"gpu_mem_gib": 6.0}})
    assert A._RUNTIME_SETTINGS["gpu_limit_bytes"] == 6 * GIB
    A._apply_central_limits({"limits": {}})
    assert "gpu_limit_bytes" not in A._RUNTIME_SETTINGS
