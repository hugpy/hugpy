"""t21 — the LIVE central-side mirror: _worker_fit accepts a model's VRAM band
FLOOR as admissible under contention (flask_app/.../worker_routes._worker_fit).

A model with an explicit VRAM tolerance band may, under contention, be seated at
its band floor (a smaller gpu_mem_gib = fewer GPU layers) rather than its target,
so central's feasibility math reports it as GPU-resident-admissible even when the
TARGET wouldn't fit free VRAM. Additive: never changes the base fit / gpu_resident
verdict, and a model with NO band is byte-identical to before t21.

Run: venv/bin/python -m pytest tests/test_flex_central_mirror.py -q
"""
import os
import sys
import importlib
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PROJECTS_HOME",
                      tempfile.mkdtemp(prefix="hugpy-flexmirror-test-"))

wr = importlib.import_module(
    "hugpy_server.app.routes.worker_routes")

GIB = 2 ** 30
MODEL = "some~gguf-model"


def _worker(spill=None):
    # 24 GiB card, only 6 GiB free -> an 8 GiB model can't be GPU-resident at
    # target, but CAN spill and (with a band) sit at its floor.
    return {"id": "w", "name": "ae", "gpu": "RTX 3090",
            "vram_total": 24 * GIB, "vram_free": 6 * GIB, "free_ram": 32 * GIB,
            "spill_by_model": ({MODEL: spill} if spill else {})}


def _fit(monkeypatch, spill):
    monkeypatch.setattr(wr, "_model_gguf_bytes", lambda mk: 8 * GIB)
    return wr._worker_fit(MODEL, _worker(spill))


def test_no_band_is_backcompat(monkeypatch):
    v = _fit(monkeypatch, None)
    assert v["fit"] is True and v["gpu_resident"] is False
    assert v["band_floor_admissible"] is None
    assert "partially offload" in v["reason"]


def test_band_floor_admissible_under_contention(monkeypatch):
    # target 5 GiB, ±10% of 24 GiB (=2.4) -> floor 2.6 GiB <= 6 GiB free -> yes.
    v = _fit(monkeypatch, {"gpu_mem_gib": 5, "gpu_mem_gib_deviation_pct": 10})
    assert v["gpu_resident"] is False                # target still doesn't fit
    assert v["band_floor_admissible"] is True
    assert v["band_floor_bytes"] is not None
    assert "band floor" in v["reason"]


def test_band_too_narrow_floor_still_doesnt_fit(monkeypatch):
    # target 8 GiB, ±5% of 24 GiB (=1.2) -> floor 6.8 GiB > 6 GiB free -> no.
    v = _fit(monkeypatch, {"gpu_mem_gib": 8, "gpu_mem_gib_deviation_pct": 5})
    assert v["band_floor_admissible"] is False
    assert "partially offload" in v["reason"]        # falls back to base reason
