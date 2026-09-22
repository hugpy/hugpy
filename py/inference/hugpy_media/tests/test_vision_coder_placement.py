"""vision_coder.VisionCoder — placement-seam max_memory resolution (t31).

vision_coder.py used to build its OWN hardcoded max_memory map ({0: "5GiB",
"cpu": "24GiB"}) for the transformers Qwen2.5-VL load, entirely outside the
spill.transformers_max_memory seam every other transformers loader (e.g.
managers/generate/coder.py) already honors — so explicit HUGPY_GPU_MEM_GIB /
HUGPY_CPU_MEM_GIB budgets and HUGPY_N_GPU_LAYERS placement intent (Max GPU /
CPU only / auto, t26/t27) silently did nothing for vision loads. Flagged as a
follow-up in 363d0ed ("Known separate path... vision_coder.py builds
hardcoded max_memory outside the seam").

Fix: VisionCoder.__init__ now calls spill.transformers_max_memory() and only
falls back to the loader's own gpu_max_memory/cpu_max_memory (5GiB/24GiB by
default) when the seam has no better answer (falsy) — so:
  * an explicit/placement-driven seam answer is HONORED (regression-proofed
    against the old hardcoded map reappearing), and
  * the legacy default is BYTE-IDENTICAL when the seam can't say anything
    (e.g. VRAM unreadable) — no-operator-config behavior does not regress.

This test never loads real model weights: it monkeypatches the module-level
get_transformers()-returned classes with lightweight fakes that just record
the from_pretrained() kwargs they receive, and monkeypatches
spill.transformers_max_memory directly (vision_coder imports it locally, at
call time, inside __init__ — so patching the spill module attribute is
sufficient; no need to patch a cached binding in vision_coder's namespace).

Runs both ways:
    cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
    venv/bin/python -m pytest tests/test_vision_coder_placement.py -q
    venv/bin/python tests/test_vision_coder_placement.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("torch")          # [transformers] extra: VisionCoder require()s both
pytest.importorskip("transformers")

vc = importlib.import_module("hugpy_media.vision.vision_coder")
spill = importlib.import_module("hugpy_engine.spill")


# --------------------------------------------------------------------------- #
# fakes — no real transformers load, just kwargs capture
# --------------------------------------------------------------------------- #
class _FakeGenerationConfig:
    def __init__(self):
        self.do_sample = None
        self.temperature = None
        self.top_p = None
        self.top_k = None
        self.use_cache = None


class _FakeVLModel:
    """Stand-in for Qwen2_5_VLForConditionalGeneration. Records the kwargs
    from_pretrained() was called with (module-level, so the test can inspect
    them after VisionCoder() returns) and returns an instance cheap enough to
    exercise the rest of __init__ (eval(), generation_config.*)."""
    last_kwargs: dict = {}

    def __init__(self):
        self.generation_config = _FakeGenerationConfig()

    def eval(self):
        return self

    def to(self, device):
        return self

    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        cls.last_kwargs = dict(kwargs)
        return cls()


class _FakeProcessor:
    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        return SimpleNamespace(kwargs=kwargs)


def _fake_get_transformers(name=None):
    if name == "Qwen2_5_VLForConditionalGeneration":
        return _FakeVLModel
    if name == "AutoProcessor":
        return _FakeProcessor
    raise AssertionError(f"unexpected get_transformers({name!r}) call")


def _make_cfg(device="cuda", device_map="auto",
              gpu_max_memory="5GiB", cpu_max_memory="24GiB"):
    torch = vc.get_torch()
    return vc.VisionCoderConfig(
        model_key="test-vision-model",
        model_dir="/nonexistent/does/not/matter/for/this/test",
        device=device,
        torch_dtype=torch.float32,
        device_map=device_map,
        gpu_max_memory=gpu_max_memory,
        cpu_max_memory=cpu_max_memory,
    )


class _patched:
    """Save/restore attrs so tests are independent under pytest or plain-script."""

    def __init__(self, obj, **kw):
        self.obj = obj
        self.kw = kw
        self.saved = {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.saved[k] = getattr(self.obj, k)
            setattr(self.obj, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.obj, k, v)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_clean_import():
    """The module imports cleanly (no syntax/import breakage from the edit)."""
    importlib.reload(vc)


def test_seam_honored_when_it_has_an_answer():
    """Explicit/placement-driven seam answer wins over the loader's own
    gpu_max_memory/cpu_max_memory defaults — the whole point of t31."""
    seam_answer = {0: "9.00GiB", "cpu": "12.00GiB"}
    with _patched(vc, get_transformers=_fake_get_transformers), \
         _patched(spill, transformers_max_memory=lambda: seam_answer):
        cfg = _make_cfg(gpu_max_memory="5GiB", cpu_max_memory="24GiB")
        vc.VisionCoder(cfg)
    assert _FakeVLModel.last_kwargs.get("max_memory") == seam_answer, \
        _FakeVLModel.last_kwargs
    assert _FakeVLModel.last_kwargs.get("device_map") == "auto"


def test_legacy_default_preserved_when_seam_has_no_answer():
    """seam returns None (no better answer, e.g. VRAM unreadable) -> the
    loader's OWN gpu_max_memory/cpu_max_memory apply, BYTE-IDENTICAL to the
    pre-t31 hardcoded map. No-operator-config behavior must not regress."""
    with _patched(vc, get_transformers=_fake_get_transformers), \
         _patched(spill, transformers_max_memory=lambda: None):
        cfg = _make_cfg(gpu_max_memory="5GiB", cpu_max_memory="24GiB")
        vc.VisionCoder(cfg)
    assert _FakeVLModel.last_kwargs.get("max_memory") == {
        0: "5GiB", "cpu": "24GiB",
    }, _FakeVLModel.last_kwargs


def test_legacy_default_preserved_when_seam_returns_empty_map():
    """An empty dict from the seam is also "no better answer" (falsy) -> same
    fallback as None, not a bare {} handed to from_pretrained."""
    with _patched(vc, get_transformers=_fake_get_transformers), \
         _patched(spill, transformers_max_memory=lambda: {}):
        cfg = _make_cfg(gpu_max_memory="5GiB", cpu_max_memory="24GiB")
        vc.VisionCoder(cfg)
    assert _FakeVLModel.last_kwargs.get("max_memory") == {
        0: "5GiB", "cpu": "24GiB",
    }, _FakeVLModel.last_kwargs


def test_custom_loader_defaults_still_apply_as_the_fallback():
    """A caller-supplied gpu_max_memory/cpu_max_memory (not just the literal
    5GiB/24GiB dataclass defaults) is what's used as the fallback -- confirms
    the fallback reads cfg, not a re-hardcoded literal."""
    with _patched(vc, get_transformers=_fake_get_transformers), \
         _patched(spill, transformers_max_memory=lambda: None):
        cfg = _make_cfg(gpu_max_memory="3GiB", cpu_max_memory="10GiB")
        vc.VisionCoder(cfg)
    assert _FakeVLModel.last_kwargs.get("max_memory") == {
        0: "3GiB", "cpu": "10GiB",
    }, _FakeVLModel.last_kwargs


def test_non_cuda_device_untouched():
    """device != 'cuda' (or device_map != 'auto') skips the seam entirely and
    keeps the pre-existing .to(device) path — the branch this task did not
    touch stays byte-identical."""
    seam_calls = []
    with _patched(vc, get_transformers=_fake_get_transformers), \
         _patched(spill, transformers_max_memory=lambda: seam_calls.append(1) or {0: "99GiB"}):
        cfg = _make_cfg(device="cpu", device_map="auto")
        vc.VisionCoder(cfg)
    assert seam_calls == [], "seam must not be consulted off the cuda+auto path"
    assert "max_memory" not in _FakeVLModel.last_kwargs, _FakeVLModel.last_kwargs
    assert "device_map" not in _FakeVLModel.last_kwargs, _FakeVLModel.last_kwargs


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
