"""``_worker_fit`` prices GPU residency PER DEVICE (operator 2026-09-25).

``gpu_resident`` ("fits VRAM outright", the term the capacity-outranks-residency
routing penalty reads) must price a non-splittable pipeline against the LARGEST
SINGLE card, not the box-wide pooled sum — else a 25 GiB diffusers model reads as
gpu_resident on a-brain's 4x24 (96 GiB pooled) where no single card holds it. A
GGUF keeps the box sum (it tensor-splits). Single-GPU boxes are unchanged.

Runs both ways:
    venv/bin/python tests/integration/test_worker_fit_per_device.py
    venv/bin/python -m pytest tests/integration/test_worker_fit_per_device.py -q
"""
from __future__ import annotations

import os
import sys
import tempfile
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
os.environ.setdefault("PROJECTS_HOME",
                      tempfile.mkdtemp(prefix="hugpy-worker-fit-perdev-"))

wr = importlib.import_module("hugpy_server.app.routes.worker_routes")

GIB = 2 ** 30


def _gpus(frees):
    return [{"index": i, "memory_free": f, "memory_total": 24 * GIB}
            for i, f in enumerate(frees)]


# a-brain: one nearly-full card + three with 20 GiB free. Pooled = 65 GiB, but
# the largest single card is 20 GiB.
ABRAIN = {"id": "abrain", "name": "abrain", "gpu": "4x3090",
          "vram_free": int(65 * GIB), "vram_total": int(96 * GIB),
          "free_ram": int(128 * GIB),
          "gpus": _gpus([5 * GIB, 20 * GIB, 20 * GIB, 20 * GIB])}


def test_engine_gpu_free_largest_card_for_non_splittable():
    assert wr._engine_gpu_free(ABRAIN, splittable=False, pooled=65 * GIB) == 20 * GIB
    assert wr._engine_gpu_free(ABRAIN, splittable=True, pooled=65 * GIB) == 65 * GIB


def test_engine_gpu_free_degrades_to_pooled_single_gpu():
    w = {"gpus": _gpus([20 * GIB])}
    assert wr._engine_gpu_free(w, splittable=False, pooled=20 * GIB) == 20 * GIB
    # no gpus[] -> the pooled figure verbatim
    assert wr._engine_gpu_free({}, splittable=False, pooled=7 * GIB) == 7 * GIB


class _patched:
    def __init__(self, gguf_bytes, manifest_bytes):
        self.gb, self.mb = gguf_bytes, manifest_bytes

    def __enter__(self):
        self.o = (wr._model_gguf_bytes, wr._model_manifest_bytes, wr._model_moe_fit)
        wr._model_gguf_bytes = lambda mk: self.gb
        wr._model_manifest_bytes = lambda mk: self.mb
        wr._model_moe_fit = lambda mk: None
        return self

    def __exit__(self, *exc):
        (wr._model_gguf_bytes, wr._model_manifest_bytes, wr._model_moe_fit) = self.o


def test_diffusers_25g_not_gpu_resident_on_4x24_despite_pooled_65g():
    # non-GGUF -> _model_gguf_bytes None -> manifest size, priced per single card.
    with _patched(gguf_bytes=None, manifest_bytes=int(25 * GIB)):
        fit = wr._worker_fit("sd-xl-big", ABRAIN)
    assert fit["gpu_splittable"] is False, fit
    assert fit["gpu_vram_free"] == int(20 * GIB), fit
    assert fit["gpu_resident"] is False, fit          # 25 GiB fits no single card
    # box-wide sum is still reported for display
    assert fit["vram_free"] == int(65 * GIB), fit


def test_gguf_25g_is_gpu_resident_on_4x24_via_box_sum():
    with _patched(gguf_bytes=int(25 * GIB), manifest_bytes=None):
        fit = wr._worker_fit("big-gguf", ABRAIN)
    assert fit["gpu_splittable"] is True, fit
    assert fit["gpu_vram_free"] == int(65 * GIB), fit
    assert fit["gpu_resident"] is True, fit           # splits across the cards


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
