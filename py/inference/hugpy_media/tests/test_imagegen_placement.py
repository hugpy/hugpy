"""imagegen_runner._place_diffusers_pipeline — diffusers spill mechanism (Slice C).

diffusers pipelines do NOT accept device_map/max_memory the transformers way, so
the honest spill mechanism is diffusers' own CPU-offload API, driven by the SAME
placement seam the transformers loaders read (spill.n_gpu_layers_intent /
alloc_mode_env — no parallel intent reader):

  * CPU-leaning intent (ram-only / n_gpu_layers "off"/0) -> enable_sequential_cpu_offload()
  * max-ram alloc_mode                                   -> enable_model_cpu_offload()
  * default (gpu-only / auto / no intent, or no cuda)    -> today's .to(cuda/cpu)
  * a pipeline class lacking the offload method (genuine capability gap) is
    logged ONCE and falls back to .to(cuda) — never a silent mode ignore.

No real diffusers load: a fake pipe records which of {.to, enable_model_cpu_offload,
enable_sequential_cpu_offload} was invoked. The intent is driven through the REAL
seam by setting HUGPY_N_GPU_LAYERS / HUGPY_ALLOC_MODE env.

Runs both ways:
    cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
    venv/bin/python -m pytest tests/test_imagegen_placement.py -q
    venv/bin/python tests/test_imagegen_placement.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib  # noqa: E402

ig = importlib.import_module("hugpy_media.imagegen.imagegen_runner")


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class _FakePipe:
    """Records the placement calls. ``offload_methods`` controls which
    enable_*_cpu_offload methods exist (to simulate a pipeline class that lacks
    one — the capability-gap path)."""

    def __init__(self, offload_methods=("enable_model_cpu_offload",
                                        "enable_sequential_cpu_offload"),
                 offload_raises=False):
        self.calls = []
        self._offload_raises = offload_raises
        self._present = set(offload_methods)

    def to(self, device):
        self.calls.append(("to", device))
        return self

    def _make(name):
        def _fn(self):
            if self._offload_raises:
                self.calls.append((name, "RAISED"))
                raise RuntimeError("offload boom")
            self.calls.append((name, None))
        return _fn

    enable_model_cpu_offload = _make("enable_model_cpu_offload")
    enable_sequential_cpu_offload = _make("enable_sequential_cpu_offload")

    def __getattribute__(self, item):
        # Hide an offload method if it wasn't declared present (simulate a
        # pipeline class that doesn't implement it).
        if item in ("enable_model_cpu_offload", "enable_sequential_cpu_offload"):
            present = object.__getattribute__(self, "_present")
            if item not in present:
                raise AttributeError(item)
        return object.__getattribute__(self, item)


def _set_env(**env):
    for k in ("HUGPY_N_GPU_LAYERS", "HUGPY_ALLOC_MODE"):
        os.environ.pop(k, None)
    for k, v in env.items():
        if v is not None:
            os.environ[k] = str(v)


def _clear_env():
    for k in ("HUGPY_N_GPU_LAYERS", "HUGPY_ALLOC_MODE"):
        os.environ.pop(k, None)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_no_cuda_goes_to_cpu():
    """Off the GPU: .to('cpu'), no offload, byte-identical to before."""
    _clear_env()
    pipe = _FakePipe()
    label = ig._place_diffusers_pipeline(pipe, cuda=False, model_key="m")
    assert pipe.calls == [("to", "cpu")], pipe.calls
    assert label == "cpu"


def test_default_intent_goes_to_cuda():
    """No env intent + cuda -> today's .to('cuda'), no offload (defaults promise)."""
    _set_env()
    try:
        pipe = _FakePipe()
        label = ig._place_diffusers_pipeline(pipe, cuda=True, model_key="m")
        assert pipe.calls == [("to", "cuda")], pipe.calls
        assert label == "cuda"
    finally:
        _clear_env()


def test_gpu_only_intent_goes_to_cuda():
    """gpu-only (n_gpu_layers=-1) is all-on-card -> .to('cuda'), no offload."""
    _set_env(HUGPY_N_GPU_LAYERS="-1")
    try:
        pipe = _FakePipe()
        label = ig._place_diffusers_pipeline(pipe, cuda=True, model_key="m")
        assert pipe.calls == [("to", "cuda")], pipe.calls
        assert label == "cuda"
    finally:
        _clear_env()


def test_cpu_intent_uses_sequential_offload():
    """ram-only / CPU-leaning intent -> enable_sequential_cpu_offload()."""
    for val in ("off", "0", "cpu"):
        _set_env(HUGPY_N_GPU_LAYERS=val)
        try:
            pipe = _FakePipe()
            label = ig._place_diffusers_pipeline(pipe, cuda=True, model_key="m")
            assert pipe.calls == [("enable_sequential_cpu_offload", None)], (val, pipe.calls)
            assert "sequential-offload" in label, label
        finally:
            _clear_env()


def test_max_ram_uses_model_offload():
    """max-ram alloc_mode -> enable_model_cpu_offload()."""
    _set_env(HUGPY_ALLOC_MODE="max-ram")
    try:
        pipe = _FakePipe()
        label = ig._place_diffusers_pipeline(pipe, cuda=True, model_key="m")
        assert pipe.calls == [("enable_model_cpu_offload", None)], pipe.calls
        assert "model-offload" in label, label
    finally:
        _clear_env()


def test_cpu_intent_wins_over_max_ram():
    """Explicit CPU-only intent + max-ram both set -> the stronger CPU-only
    (sequential) placement is chosen (matches the seam's own precedence: the
    n_gpu_layers 'cpu' intent is the operator saying 'off the card')."""
    _set_env(HUGPY_N_GPU_LAYERS="off", HUGPY_ALLOC_MODE="max-ram")
    try:
        pipe = _FakePipe()
        ig._place_diffusers_pipeline(pipe, cuda=True, model_key="m")
        assert pipe.calls == [("enable_sequential_cpu_offload", None)], pipe.calls
    finally:
        _clear_env()


def test_capability_gap_logged_and_falls_back_to_cuda():
    """A pipeline class WITHOUT enable_sequential_cpu_offload asked for ram-only
    -> honest fallback to .to('cuda') (the mode is not silently ignored — the
    label says the pipeline can't honor it)."""
    _set_env(HUGPY_N_GPU_LAYERS="off")
    try:
        pipe = _FakePipe(offload_methods=("enable_model_cpu_offload",))  # no sequential
        label = ig._place_diffusers_pipeline(pipe, cuda=True, model_key="m")
        assert pipe.calls == [("to", "cuda")], pipe.calls
        assert "unsupported" in label, label
    finally:
        _clear_env()


def test_offload_failure_falls_back_to_cuda():
    """If enable_model_cpu_offload() raises, fall back to .to('cuda') honestly."""
    _set_env(HUGPY_ALLOC_MODE="max-ram")
    try:
        pipe = _FakePipe(offload_raises=True)
        label = ig._place_diffusers_pipeline(pipe, cuda=True, model_key="m")
        assert pipe.calls == [("enable_model_cpu_offload", "RAISED"),
                              ("to", "cuda")], pipe.calls
        assert "failed" in label, label
    finally:
        _clear_env()


# --------------------------------------------------------------------------- #
# plain-script runner
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
