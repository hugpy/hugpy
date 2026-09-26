"""imagegen_runner ram-only (CPU-only) honoring + honest CUDA-OOM surfacing.

Incident 2026-09-25 01:12:58: central derived serve mode ram-only for sd-turbo on
computron (RTX 4060, 17.88 MiB free — another process held 6.83 GiB), but the
worker's diffusers text-to-image path DROPPED that intent: `cuda` was read purely
from torch.cuda.is_available(), so the loader priced fp16 against the tiny free
VRAM, elected a 4-bit bnb load with enable_model_cpu_offload, and OOM'd inside
bitsandbytes dequant on CUDA at generation. ram-only must instead run ENTIRELY on
the CPU — the GPU must NOT be touched at all.

These tests drive the REAL placement seam (spill.n_gpu_layers_intent via
HUGPY_N_GPU_LAYERS) with NO real torch/diffusers load: torch is a fake in
sys.modules, the pipeline is a fake that records device moves, and _trim_host_ram/
_release_cuda are neutered — so the assertions hold on any box, GPU or not.

    cd /srv/hugpy/src/hugpy/py/inference/hugpy_media
    python -m pytest tests/test_imagegen_ram_only.py -q
    python tests/test_imagegen_ram_only.py
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib  # noqa: E402

ig = importlib.import_module("hugpy_media.imagegen.imagegen_runner")

GiB = 2 ** 30


# --------------------------------------------------------------------------- #
# fakes / helpers
# --------------------------------------------------------------------------- #
class _FakePipe:
    """Records every device move + offload call so a test can assert the GPU was
    never touched."""

    def __init__(self):
        self.calls = []
        self.hooks_removed = False

    def to(self, device):
        self.calls.append(("to", device))
        return self

    def enable_model_cpu_offload(self):
        self.calls.append(("enable_model_cpu_offload", None))

    def enable_sequential_cpu_offload(self):
        self.calls.append(("enable_sequential_cpu_offload", None))

    def remove_all_hooks(self):
        self.hooks_removed = True


class _FakeAuto:
    __name__ = "_FakeAuto"
    _next = None
    last_kwargs = None

    @classmethod
    def from_pretrained(cls, model_dir, **kw):
        cls.last_kwargs = kw
        return cls._next


def _install_fakes(state, *, cuda_available=True):
    fake_torch = types.SimpleNamespace(
        float16="f16", float32="f32",
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
    )
    state.append(("__torch__", sys.modules.get("torch")))
    sys.modules["torch"] = fake_torch
    # neuter the cache/host trims (they'd import the fake torch and poke ctypes)
    state.append(("_release_cuda", ig._release_cuda))
    ig._release_cuda = lambda: None
    state.append(("_trim_host_ram", ig._trim_host_ram))
    ig._trim_host_ram = lambda: None
    state.append(("_weight_bytes", ig._weight_bytes))
    ig._weight_bytes = lambda _d: 12 * GiB          # sd-turbo-ish fp16 footprint
    state.append(("_free_vram_bytes", ig._free_vram_bytes))
    ig._free_vram_bytes = lambda: 18 * 2 ** 20      # 17.88 MiB — the incident card


def _restore(state):
    for name, val in state:
        if name == "__torch__":
            if val is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = val
        else:
            setattr(ig, name, val)


def _set_ngl(val):
    for k in ("HUGPY_N_GPU_LAYERS", "HUGPY_ALLOC_MODE"):
        os.environ.pop(k, None)
    if val is not None:
        os.environ["HUGPY_N_GPU_LAYERS"] = str(val)


def _clear_ngl():
    for k in ("HUGPY_N_GPU_LAYERS", "HUGPY_ALLOC_MODE"):
        os.environ.pop(k, None)


# --------------------------------------------------------------------------- #
# seam decoders
# --------------------------------------------------------------------------- #
def test_cpu_forced_reads_the_derived_serve_mode():
    for val in ("off", "0", "cpu", "none"):
        _set_ngl(val)
        try:
            assert ig._cpu_forced() is True, val
        finally:
            _clear_ngl()
    for val in (None, "auto", "-1", "20"):
        _set_ngl(val)
        try:
            assert ig._cpu_forced() is False, val
        finally:
            _clear_ngl()


def test_generator_device_is_cpu_under_ram_only():
    state = []
    _install_fakes(state, cuda_available=True)
    try:
        _set_ngl("off")
        assert ig._generator_device() == "cpu"      # ram-only: never the card
        _set_ngl("auto")
        assert ig._generator_device() == "cuda"      # GPU genuinely in use
    finally:
        _clear_ngl()
        _restore(state)


def test_generator_device_is_cpu_without_cuda():
    state = []
    _install_fakes(state, cuda_available=False)
    try:
        _clear_ngl()
        assert ig._generator_device() == "cpu"
    finally:
        _restore(state)


# --------------------------------------------------------------------------- #
# THE incident: ram-only must load fully on the CPU, never quantize, never CUDA
# --------------------------------------------------------------------------- #
def test_ram_only_load_runs_fully_on_cpu_and_skips_quant():
    """A ram-only load on a card with almost no free VRAM must NOT elect a 4-bit
    bnb load, must build in fp32, and must place the pipeline on the CPU — the
    GPU is never touched (this is the exact sd-turbo/computron case)."""
    state = []
    _install_fakes(state, cuda_available=True)
    pipe = _FakePipe()
    _FakeAuto._next = pipe
    _FakeAuto.last_kwargs = None
    _set_ngl("off")
    try:
        out_pipe, placement = ig._load_diffusers_pipeline(
            _FakeAuto, "/fake/dir", "sd-turbo",
            place_fn=ig._place_diffusers_pipeline,
        )
        assert out_pipe is pipe
        # no quantization was even attempted (cuda folded to False upstream)
        assert "quantization_config" not in _FakeAuto.last_kwargs, _FakeAuto.last_kwargs
        # fp32 on the CPU, never fp16
        assert _FakeAuto.last_kwargs["torch_dtype"] == "f32", _FakeAuto.last_kwargs
        # the GPU was never touched — CPU only
        assert pipe.calls == [("to", "cpu")], pipe.calls
        assert placement == "cpu", placement
    finally:
        _clear_ngl()
        _restore(state)


def test_default_intent_still_uses_the_gpu():
    """Regression guard: with NO ram-only intent, the effective-cuda decision is
    unchanged — the pipeline still goes to the GPU (defaults-are-promises)."""
    state = []
    _install_fakes(state, cuda_available=True)
    ig._free_vram_bytes = lambda: 40 * GiB          # roomy card -> no quant
    pipe = _FakePipe()
    _FakeAuto._next = pipe
    _FakeAuto.last_kwargs = None
    _clear_ngl()
    try:
        out_pipe, placement = ig._load_diffusers_pipeline(
            _FakeAuto, "/fake/dir", "sd-turbo",
            place_fn=ig._place_diffusers_pipeline,
        )
        assert out_pipe is pipe
        assert _FakeAuto.last_kwargs["torch_dtype"] == "f16", _FakeAuto.last_kwargs
        assert pipe.calls == [("to", "cuda")], pipe.calls
        assert placement == "cuda", placement
    finally:
        _clear_ngl()
        _restore(state)


# --------------------------------------------------------------------------- #
# honest CUDA-OOM surfacing (items 2 & 3)
# --------------------------------------------------------------------------- #
def test_honest_vram_error_names_intent_and_free_vram():
    state = []
    _install_fakes(state, cuda_available=True)
    # give the snapshot real numbers
    # accept the optional device index the runner now passes (real torch's
    # mem_get_info(device=None) signature; per-GPU probe, 2026-09-25).
    sys.modules["torch"].cuda.mem_get_info = lambda *a: (18 * 2 ** 20, 8 * GiB)
    _set_ngl("off")
    try:
        oom = RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB. "
                           "Process 1146566 has 6.83 GiB memory in use.")
        honest = ig._honest_vram_error(oom, "sd-turbo", "generation")
        assert honest is not None
        assert isinstance(honest, ig._HonestVRAMError)
        msg = str(honest)
        assert "sd-turbo" in msg
        assert "generation" in msg
        assert "intent=cpu" in msg                       # names the derived mode
        assert "GiB free" in msg                         # names free VRAM
        assert "6.83 GiB memory in use" in msg           # keeps torch's detail
        # still classifies as the retryable VRAM class (carries "out of memory")
        assert ig.is_retryable_vram_failure(honest) is True
    finally:
        _clear_ngl()
        _restore(state)


def test_honest_vram_error_passes_non_oom_through():
    """A non-OOM error is not rewritten (caller re-raises the original)."""
    assert ig._honest_vram_error(ValueError("bad prompt"), "m", "generation") is None


def test_honest_vram_error_not_double_wrapped():
    already = ig._HonestVRAMError("CUDA out of memory ... already honest")
    assert ig._honest_vram_error(already, "m", "load") is None


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
            import traceback
            print(f"[FAIL] {t.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            ok += 1
            print(f"[ok]   {t.__name__}")
    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(_main())
