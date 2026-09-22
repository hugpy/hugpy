"""imagegen_runner honest-footprint pricing + quant election + VRAM unwind (k66).

The t2i path used to load fp16 with no fit ladder, so FLUX.2-klein (transformer
~18 GiB + Qwen3 text_encoder ~16 GiB of bf16 weights) ballooned to 22.37 GiB and
OOM'd on a ~22.4-GiB-free card, and the OOM'd process then held the whole card
until an /ops/restart. These tests pin the two guarantees, WITHOUT a real load:

  1. Stage-2 pricing (_should_quantize / _elect_quantization) elects 4-bit from
     the artifact's REAL disk bytes vs budgetable free VRAM, BEFORE loading.
  2. Any failed load/generation deterministically releases the CUDA cache
     (_release_cuda), so failure returns to baseline VRAM instead of a zombie.

No diffusers/torch load happens: _load_diffusers_pipeline is driven through fake
pipeline classes, and _release_cuda is monkeypatched to a counter. free VRAM and
weight bytes are injected so the pricing decision is deterministic on any box.

    cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
    venv/bin/python -m pytest tests/test_imagegen_footprint.py -q
    venv/bin/python tests/test_imagegen_footprint.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib  # noqa: E402

import pytest  # noqa: E402

ig = importlib.import_module("hugpy_media.imagegen.imagegen_runner")

GiB = 2 ** 30


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class _FakePipe:
    """Records placement calls; can be told to OOM at .to() or at offload to
    exercise the mid-placement failure path."""

    def __init__(self, oom_on_to=False, oom_on_offload=False, has_offload=True):
        self.calls = []
        self.hooks_removed = False
        self._oom_on_to = oom_on_to
        self._oom_on_offload = oom_on_offload
        self._has_offload = has_offload

    def to(self, device):
        self.calls.append(("to", device))
        if self._oom_on_to:
            raise RuntimeError("CUDA out of memory (fake .to)")
        return self

    def enable_model_cpu_offload(self):
        self.calls.append(("offload", None))
        if self._oom_on_offload:
            raise RuntimeError("offload boom")

    def remove_all_hooks(self):
        self.hooks_removed = True

    def __getattribute__(self, item):
        if item == "enable_model_cpu_offload":
            if not object.__getattribute__(self, "_has_offload"):
                raise AttributeError(item)
        return object.__getattribute__(self, item)


class _FakeAuto:
    """Stands in for AutoPipelineForText2Image/Image2Image. ``value_error`` makes
    from_pretrained raise ValueError (drives the DiffusionPipeline fallback);
    ``load_oom`` makes it raise a CUDA OOM at load."""
    __name__ = "_FakeAuto"
    _next = None            # the _FakePipe from_pretrained returns
    value_error = False
    load_oom = False

    @classmethod
    def from_pretrained(cls, model_dir, **kw):
        cls.last_kwargs = kw
        if cls.load_oom:
            raise RuntimeError("CUDA out of memory (fake load)")
        if cls.value_error:
            raise ValueError("can't find a pipeline linked to FakeKleinPipeline")
        return cls._next


def _install_release_counter(monkeypatch_state):
    """Replace _release_cuda with a counter; return a mutable [count]."""
    box = [0]
    orig = ig._release_cuda

    def _counting():
        box[0] += 1
    ig._release_cuda = _counting
    monkeypatch_state.append(("_release_cuda", orig))
    # _trim_host_ram calls the real _release_cuda + malloc_trim; neuter it too so
    # a successful load doesn't touch a real (possibly absent) CUDA/glibc.
    orig_trim = ig._trim_host_ram
    ig._trim_host_ram = lambda: None
    monkeypatch_state.append(("_trim_host_ram", orig_trim))
    return box


def _restore(monkeypatch_state):
    for name, val in monkeypatch_state:
        setattr(ig, name, val)


def _force_free_vram(monkeypatch_state, value):
    orig = ig._free_vram_bytes
    ig._free_vram_bytes = lambda: value
    monkeypatch_state.append(("_free_vram_bytes", orig))


def _force_weight_bytes(monkeypatch_state, value):
    orig = ig._weight_bytes
    ig._weight_bytes = lambda _d: value
    monkeypatch_state.append(("_weight_bytes", orig))


def _force_cuda_true(monkeypatch_state):
    """_load_diffusers_pipeline reads torch.cuda.is_available(); force True so the
    cuda branch runs without depending on the box, and float16 dtype resolves."""
    import types
    fake_torch = types.SimpleNamespace(
        float16="f16", float32="f32",
        cuda=types.SimpleNamespace(is_available=lambda: True),
    )
    orig = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    monkeypatch_state.append(("__torch__", orig))


def _restore_torch(monkeypatch_state):
    for name, val in monkeypatch_state:
        if name == "__torch__":
            if val is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = val


# --------------------------------------------------------------------------- #
# stage-2 pricing
# --------------------------------------------------------------------------- #
def test_should_quantize_prices_disk_bytes_vs_free_vram():
    """The flux2-klein footprint: ~34 GiB of weights on a ~22 GiB-free card must
    elect 4-bit; a small model that fits must not."""
    klein = 34 * GiB
    free = 22 * GiB
    assert ig._should_quantize(klein, free, "auto") is True
    assert ig._should_quantize(6 * GiB, free, "auto") is False


def test_should_quantize_respects_always_never_and_unknown():
    assert ig._should_quantize(1 * GiB, 22 * GiB, "always") is True   # tiny, forced
    assert ig._should_quantize(99 * GiB, 22 * GiB, "never") is False  # huge, forced off
    assert ig._should_quantize(99 * GiB, None, "auto") is False       # unmeasurable -> no blind guess


def test_elect_quantization_logs_plan_and_elects_4bit():
    """auto + weights >> free -> a quant_config (bnb available in the tree venv)
    and a PLANNED footprint that is the SHRUNK size, not the fp16 size."""
    pytest.importorskip("bitsandbytes")   # [imagegen] extra: the real quant config
    pytest.importorskip("diffusers")
    state = []
    _force_weight_bytes(state, 34 * GiB)
    _force_free_vram(state, 22 * GiB)
    try:
        qc, plan = ig._elect_quantization("/fake/dir", "FLUX.2-klein", cuda=True)
    finally:
        _restore(state)
    assert plan["decision"].startswith("quantize-4bit"), plan
    assert plan["planned_bytes"] < plan["weight_bytes"], plan
    # bnb+diffusers are present in the tree venv, so the config is real.
    assert qc is not None


def test_elect_quantization_fp16_when_it_fits():
    state = []
    _force_weight_bytes(state, 6 * GiB)
    _force_free_vram(state, 22 * GiB)
    try:
        qc, plan = ig._elect_quantization("/fake/dir", "small-sd", cuda=True)
    finally:
        _restore(state)
    assert qc is None
    assert plan["decision"] == "fp16-whole", plan
    assert plan["planned_bytes"] == plan["weight_bytes"]


def test_never_env_disables_quant():
    os.environ["HUGPY_IMAGEGEN_QUANTIZE"] = "never"
    state = []
    _force_weight_bytes(state, 99 * GiB)
    _force_free_vram(state, 22 * GiB)
    try:
        qc, plan = ig._elect_quantization("/fake/dir", "huge", cuda=True)
    finally:
        _restore(state)
        os.environ.pop("HUGPY_IMAGEGEN_QUANTIZE", None)
    assert qc is None
    assert plan["decision"] == "fp16-whole"


def test_weight_bytes_sums_only_weight_files(tmp_path=None):
    import tempfile
    d = tempfile.mkdtemp()
    (Path(d) / "a.safetensors").write_bytes(b"x" * 1000)
    (Path(d) / "sub").mkdir()
    (Path(d) / "sub" / "b.bin").write_bytes(b"y" * 500)
    (Path(d) / "README.md").write_bytes(b"z" * 999)   # not a weight file
    assert ig._weight_bytes(d) == 1500


# --------------------------------------------------------------------------- #
# leak fix — deterministic VRAM unwind
# --------------------------------------------------------------------------- #
def test_load_oom_releases_cuda_and_reraises():
    """A load-time OOM must call _release_cuda (return reserved blocks) and
    propagate — NOT leave the card zombified (item I)."""
    state = []
    box = _install_release_counter(state)
    _force_cuda_true(state)
    _force_weight_bytes(state, 6 * GiB)   # small -> no quant path, plain load
    _force_free_vram(state, 22 * GiB)
    _FakeAuto.load_oom = True
    _FakeAuto.value_error = False
    try:
        raised = False
        try:
            ig._load_diffusers_pipeline(_FakeAuto, "/fake/dir", "m")
        except RuntimeError as exc:
            raised = True
            assert "out of memory" in str(exc)
        assert raised, "the OOM must propagate"
        assert box[0] >= 1, "the failed load must release the CUDA cache"
    finally:
        _FakeAuto.load_oom = False
        _restore(state)
        _restore_torch(state)


def test_place_oom_removes_hooks_and_releases():
    """OOM while moving to cuda (fits-priced small model, plain .to path) must
    strip hooks off the partial pipe and release the cache before re-raising."""
    state = []
    box = _install_release_counter(state)
    _force_cuda_true(state)
    _force_weight_bytes(state, 6 * GiB)
    _force_free_vram(state, 22 * GiB)
    pipe = _FakePipe(oom_on_to=True)
    _FakeAuto._next = pipe
    _FakeAuto.load_oom = False
    _FakeAuto.value_error = False
    try:
        raised = False
        try:
            ig._load_diffusers_pipeline(_FakeAuto, "/fake/dir", "m")
        except RuntimeError:
            raised = True
        assert raised
        assert pipe.hooks_removed, "partial pipe hooks must be removed on unwind"
        assert box[0] >= 1, "must release CUDA cache on unwind"
    finally:
        _restore(state)
        _restore_torch(state)


def test_valueerror_falls_back_to_diffusion_pipeline():
    """AutoPipeline with no mapping (Flux2Klein-style) falls back to the concrete
    DiffusionPipeline class rather than failing the load."""
    state = []
    _install_release_counter(state)
    _force_cuda_true(state)
    _force_weight_bytes(state, 6 * GiB)
    _force_free_vram(state, 22 * GiB)
    concrete = _FakePipe()
    _FakeAuto.value_error = True
    _FakeAuto.load_oom = False

    import types
    fake_diffusers = types.SimpleNamespace(
        DiffusionPipeline=types.SimpleNamespace(
            from_pretrained=lambda md, **kw: concrete))
    orig_diff = sys.modules.get("diffusers")
    sys.modules["diffusers"] = fake_diffusers
    try:
        pipe, placement = ig._load_diffusers_pipeline(_FakeAuto, "/fake/dir", "m")
        # fallback=True -> offload path taken (cuda + fallback), so enable_model_cpu_offload ran.
        assert pipe is concrete
        assert ("offload", None) in concrete.calls, concrete.calls
        assert "offload" in placement, placement
    finally:
        _FakeAuto.value_error = False
        if orig_diff is None:
            sys.modules.pop("diffusers", None)
        else:
            sys.modules["diffusers"] = orig_diff
        _restore(state)
        _restore_torch(state)


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
