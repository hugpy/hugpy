"""Central admission prices a MoE by its GPU-RESIDENT share, not the whole file.

``_worker_fit`` used to hard-block on the full artifact bytes vs VRAM+RAM, which
refused exactly the models the expert split makes serveable (a 45 GiB MoE whose
non-expert backbone + mmproj is ~1.5 GiB). Now the GPU term is the split's
gpu_bytes (central's cached ``_model_moe_gpu_bytes`` figure) and the expert
share is mmap-eligible page cache — NEVER hard-required against free RAM. The
refusal text names the basis. ``worker_fit_verdict`` grows the same optional
``moe_split_gpu_bytes`` (threaded from ``worker_can_hold``).

Runs both ways:
    venv/bin/python tests/test_worker_fit_moe.py
    venv/bin/python -m pytest tests/test_worker_fit_moe.py -q
"""
from __future__ import annotations

import os
import sys
import tempfile
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PROJECTS_HOME",
                      tempfile.mkdtemp(prefix="hugpy-worker-fit-moe-test-"))

wr = importlib.import_module(
    "hugpy_server.app.routes.worker_routes")
from hugpy_engine.alloc_modes import worker_fit_verdict

GIB = 2 ** 30


class _patched:
    """Patch wr._model_gguf_bytes + wr._model_moe_fit; restore on exit."""

    def __init__(self, sizes, moe):
        self.sizes, self.moe = sizes, moe

    def __enter__(self):
        self.o_sz, self.o_moe = wr._model_gguf_bytes, wr._model_moe_fit
        wr._model_gguf_bytes = lambda mk: self.sizes.get(mk)
        wr._model_moe_fit = lambda mk: self.moe.get(mk)
        return self

    def __exit__(self, *exc):
        wr._model_gguf_bytes, wr._model_moe_fit = self.o_sz, self.o_moe


# a 24 GiB card with 1 GiB VRAM free and 8 GiB RAM free — combined 9 GiB.
WORKER = {"id": "w1", "name": "moebox", "gpu": "RTX TEST",
          "vram_free": int(1 * GIB), "free_ram": int(8 * GIB),
          "vram_total": int(24 * GIB)}

SIZES = {
    "moe-45": int(45 * GIB),      # full artifact: never fits 9 GiB combined
    "dense-45": int(45 * GIB),
    "moe-too-big": int(45 * GIB),
}
MOE = {
    # (gpu_resident_bytes, expert_bytes)
    "moe-45": (int(1.5 * GIB), int(43.5 * GIB)),
    "moe-too-big": (int(12 * GIB), int(33 * GIB)),   # even the backbone overflows
}


def test_moe_passes_where_full_size_would_refuse():
    with _patched(SIZES, MOE):
        fit = wr._worker_fit("moe-45", WORKER)
    # 1.5 GiB GPU-resident share fits 9 GiB combined -> admissible
    assert fit["fit"] is True, fit
    # experts (43.5 GiB) were NOT hard-required against 8 GiB free RAM
    assert fit["moe_split_gpu_bytes"] == int(1.5 * GIB), fit
    assert fit["moe_expert_bytes"] == int(43.5 * GIB), fit
    # need is the split share x headroom (1.725 GiB), which exceeds 1 GiB free
    # VRAM -> honest partial-offload hint, not a refusal
    assert fit["gpu_resident"] is False, fit
    assert fit["need"] == int(int(1.5 * GIB) * wr.VRAM_HEADROOM), fit
    # need_raw keeps the on-disk artifact meaning
    assert fit["need_raw"] == int(45 * GIB), fit


def test_dense_unchanged_still_refused():
    with _patched(SIZES, MOE):
        fit = wr._worker_fit("dense-45", WORKER)
    assert fit["fit"] is False, fit
    assert fit["reason"] and fit["reason"].startswith("won't fit"), fit
    assert fit["moe_split_gpu_bytes"] is None, fit


def test_moe_refusal_names_the_basis():
    with _patched(SIZES, MOE):
        fit = wr._worker_fit("moe-too-big", WORKER)
    # 12 GiB GPU-resident > 9 GiB combined -> refuse, and say WHAT was priced
    assert fit["fit"] is False, fit
    assert fit["reason"].startswith("MoE:"), fit
    assert "GPU-resident" in fit["reason"], fit
    assert "mmap" in fit["reason"], fit


def test_moe_gpu_resident_when_vram_holds_the_backbone():
    roomy = dict(WORKER, vram_free=int(20 * GIB))
    with _patched(SIZES, MOE):
        fit = wr._worker_fit("moe-45", roomy)
    assert fit["fit"] is True and fit["gpu_resident"] is True, fit
    assert fit["reason"] is None, fit


# --------------------------------------------------------------------------- #
# worker_fit_verdict: optional moe_split_gpu_bytes (worker_can_hold threading)
# --------------------------------------------------------------------------- #
def test_verdict_dense_unchanged():
    assert worker_fit_verdict("gguf", int(45 * GIB),
                              int(24 * GIB), int(16 * GIB)) is False
    assert worker_fit_verdict("gguf", int(30 * GIB),
                              int(24 * GIB), int(16 * GIB)) is True
    assert worker_fit_verdict("gguf", None, int(24 * GIB), int(16 * GIB)) is None


def test_verdict_moe_split_is_the_static_ceiling():
    # REVISED 2026-08-28 (coder-next/computron, operator evidence): the split
    # loosens the GPU-side term ONLY — the box's combined VRAM+RAM remains the
    # physical ceiling for the WHOLE file, because the worker's own expert-RAM
    # preflight refuses a split whose expert tensors overflow host RAM.
    # Offering such a box per-request only harvests that refusal.
    # the 45 GiB MoE with a 1.5 GiB backbone CAN land on a 24+64 GiB box
    assert worker_fit_verdict("gguf", int(45 * GIB), int(24 * GIB),
                              int(64 * GIB),
                              moe_split_gpu_bytes=int(1.5 * GIB)) is True
    # ... but NOT on a box whose combined capacity can't hold the file at all
    # (computron: 8 GiB card + 16 GiB RAM vs a 45 GiB MoE — a PERMANENT no,
    # even though the 1.5 GiB backbone alone would fit)
    assert worker_fit_verdict("gguf", int(45 * GIB), int(8 * GIB),
                              int(16 * GIB),
                              moe_split_gpu_bytes=int(1.5 * GIB)) is False
    # a backbone bigger than combined capacity still refuses
    assert worker_fit_verdict("gguf", int(45 * GIB), int(4 * GIB),
                              int(4 * GIB),
                              moe_split_gpu_bytes=int(12 * GIB)) is False
    # unmeasured box: still no opinion
    assert worker_fit_verdict("gguf", int(45 * GIB), None, None,
                              moe_split_gpu_bytes=int(1.5 * GIB)) is None
    # unsizable file with a known split share: the split share votes alone
    assert worker_fit_verdict("gguf", None, int(24 * GIB), int(16 * GIB),
                              moe_split_gpu_bytes=int(1.5 * GIB)) is True


# --------------------------------------------------------------------------- #
# plain-script runner (pytest not required)
# --------------------------------------------------------------------------- #
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
