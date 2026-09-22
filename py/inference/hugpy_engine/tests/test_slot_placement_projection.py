"""Allocation must survive entry points that do not supply slot load opts."""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
spec = importlib.util.spec_from_file_location(
    "hugpy_engine.serve.placement_test_slots",
    Path(__file__).resolve().parents[1] / "src/hugpy_engine/serve/slots.py")
slots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slots)


@pytest.mark.parametrize("wire,expected", [("-1", -1), ("off", 0), ("7", 7), ("auto", None)])
def test_placement_reaches_slot_without_caller_opts(monkeypatch, wire, expected):
    monkeypatch.setenv("HUGPY_N_GPU_LAYERS", wire)
    monkeypatch.setenv("HUGPY_N_CPU_MOE", "999")
    monkeypatch.setenv("HUGPY_ALLOC_MODE", "explicit")
    monkeypatch.setattr(slots, "_FIT_CHECK", None)
    monkeypatch.setattr(slots, "_MAKE_ROOM", None)
    pool = slots.SlotPool([])
    monkeypatch.setattr(pool, "statuses", lambda: [{"_control": "http://slot", "model_key": None}])
    sent = {}
    def post(url, body, timeout):
        sent.update(body)
        return {"endpoint": "http://child"}
    monkeypatch.setattr(slots, "_post", post)
    assert pool.endpoint_for("qwen") == "http://child"
    assert sent.get("n_gpu_layers") == expected
    assert sent["n_cpu_moe"] == "999"
    assert sent["alloc_mode"] == "explicit"
    sent.clear()
    pool.endpoint_for("qwen", opts={"n_gpu_layers": 3, "n_cpu_moe": 0})
    assert sent["n_gpu_layers"] == 3
    assert sent["n_cpu_moe"] == 0
