"""In-process diffusers honors central's per-GPU pin (HUGPY_MAIN_GPU -> cuda:N).

The worker's own process enumerates every card, so it can't restrict
CUDA_VISIBLE_DEVICES per request like a slot child — it addresses the chosen card
by global index. ``_cuda_device()`` is that resolution: bare "cuda" for the
primary card (byte-identical to before) or "cuda:N" for a central-chosen card.

Run:  python3 -m pytest tests/test_imagegen_device.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib  # noqa: E402

ig = importlib.import_module("hugpy_media.imagegen.imagegen_runner")


def _clear():
    os.environ.pop("HUGPY_MAIN_GPU", None)


def test_unpinned_is_bare_cuda_byte_identical():
    _clear()
    assert ig._main_gpu_index() == 0
    assert ig._cuda_device() == "cuda"


def test_device_zero_is_still_bare_cuda():
    os.environ["HUGPY_MAIN_GPU"] = "0"
    try:
        assert ig._cuda_device() == "cuda"        # primary card == today's path
    finally:
        _clear()


def test_pinned_card_is_cuda_n():
    os.environ["HUGPY_MAIN_GPU"] = "3"
    try:
        assert ig._main_gpu_index() == 3
        assert ig._cuda_device() == "cuda:3"
    finally:
        _clear()


def test_place_diffusers_honors_a_pinned_device():
    """The GPU branch places the whole pipeline on the central-chosen card."""
    os.environ["HUGPY_MAIN_GPU"] = "2"
    try:
        class _Pipe:
            def __init__(self): self.calls = []
            def to(self, dev): self.calls.append(("to", dev)); return self
        p = _Pipe()
        label = ig._place_diffusers_pipeline(p, cuda=True, model_key="m")
        assert p.calls == [("to", "cuda:2")], p.calls
        assert label == "cuda"
    finally:
        _clear()


def test_place_diffusers_default_stays_bare_cuda():
    _clear()
    class _Pipe:
        def __init__(self): self.calls = []
        def to(self, dev): self.calls.append(("to", dev)); return self
    p = _Pipe()
    ig._place_diffusers_pipeline(p, cuda=True, model_key="m")
    assert p.calls == [("to", "cuda")], p.calls


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
