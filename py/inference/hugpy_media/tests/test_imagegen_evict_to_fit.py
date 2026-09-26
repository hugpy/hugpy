"""imagegen_runner evict-to-fit before an in-process diffusers load/generate.

Incident 2026-09-25 01:44:55: sd-turbo studio frames landed on computron behind an
IDLE flux2-klein slot child (6.99 GiB); the diffusers path never evicted it and
CUDA-OOM'd. The in-process image pipeline loads LAZILY (after the runner is cached,
so dispatch.ensure_headroom_for_load never covers the real VRAM), so the runner now
calls a worker-registered evict-to-fit hook right before loading/generating, and —
because central derived ram-only from that contended snapshot — UPGRADES ram-only
to a GPU load once eviction reclaims room (else stays CPU / fails honestly naming
what couldn't be evicted).

No real torch/diffusers load: torch is a fake in sys.modules, the pipe/auto record
calls, the headroom hook is a fake that mutates a free-VRAM box.

    cd /srv/hugpy/src/hugpy/py/inference/hugpy_media
    python tests/test_imagegen_evict_to_fit.py
"""
from __future__ import annotations

import os
import sys
import threading
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib  # noqa: E402

ig = importlib.import_module("hugpy_media.imagegen.imagegen_runner")

GiB = 2 ** 30


class _FakePipe:
    def __init__(self):
        self.calls = []

    def to(self, device):
        self.calls.append(("to", device))
        return self

    def enable_model_cpu_offload(self):
        self.calls.append(("enable_model_cpu_offload", None))

    def remove_all_hooks(self):
        pass


class _FakeAuto:
    __name__ = "_FakeAuto"
    _next = None
    last_kwargs = None

    @classmethod
    def from_pretrained(cls, model_dir, **kw):
        cls.last_kwargs = kw
        return cls._next


def _install(state, *, free_box, cuda=True):
    fake_torch = types.SimpleNamespace(
        float16="f16", float32="f32",
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
    )
    state.append(("__torch__", sys.modules.get("torch")))
    sys.modules["torch"] = fake_torch
    for name, val in (("_release_cuda", lambda: None),
                      ("_trim_host_ram", lambda: None),
                      ("_weight_bytes", lambda _d: 3 * GiB),
                      ("_free_vram_bytes", lambda: free_box[0])):
        state.append((name, getattr(ig, name)))
        setattr(ig, name, val)
    state.append(("__hook__", ig._IMAGEGEN_HEADROOM_HOOK))


def _restore(state):
    for name, val in state:
        if name == "__torch__":
            if val is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = val
        elif name == "__hook__":
            ig.set_imagegen_headroom_hook(val)
        else:
            setattr(ig, name, val)


def _set_ngl(val):
    os.environ.pop("HUGPY_N_GPU_LAYERS", None)
    if val is not None:
        os.environ["HUGPY_N_GPU_LAYERS"] = str(val)


ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# --------------------------------------------------------------------------- #
# hook indirection (central-import-safe, best-effort)
# --------------------------------------------------------------------------- #
_saved = ig._IMAGEGEN_HEADROOM_HOOK
try:
    ig.set_imagegen_headroom_hook(None)
    check("no hook -> _ensure_imagegen_headroom is a silent no-op returning None",
          ig._ensure_imagegen_headroom("m", 1 * GiB) is None)

    seen = []
    ig.set_imagegen_headroom_hook(
        lambda mk, need, job_id: seen.append((mk, need, job_id)) or {"evicted": []})
    out = ig._ensure_imagegen_headroom("ckpt", 2 * GiB)
    check("hook called with (model_key, need_bytes, job_id) and its dict returned",
          seen == [("ckpt", 2 * GiB, None)] and out == {"evicted": []})

    def boom(mk, need, job_id):
        raise RuntimeError("nope")
    ig.set_imagegen_headroom_hook(boom)
    check("a raising hook is swallowed (never breaks the gen) -> None",
          ig._ensure_imagegen_headroom("m", 1 * GiB) is None)
finally:
    ig.set_imagegen_headroom_hook(_saved)


# --------------------------------------------------------------------------- #
# THE incident: ram-only UPGRADED to GPU once eviction reclaims the card
# --------------------------------------------------------------------------- #
state = []
free_box = [int(0.6 * GiB)]                 # card full behind an idle squatter
_install(state, free_box=free_box)
pipe = _FakePipe(); _FakeAuto._next = pipe; _FakeAuto.last_kwargs = None
_set_ngl("off")                             # central derived ram-only
hook_calls = []
def _evicting_hook(mk, need, job_id=None):
    hook_calls.append((mk, need))
    free_box[0] = 7 * GiB                    # eviction frees the squatter's VRAM
    return {"evicted": ["flux2-klein"], "skipped": [], "reached": True}
ig.set_imagegen_headroom_hook(_evicting_hook)
try:
    out_pipe, placement = ig._load_diffusers_pipeline(
        _FakeAuto, "/fake/dir", "sd-turbo", place_fn=ig._place_diffusers_pipeline)
    check("evict-to-fit hook was called with the priced fp16 need",
          hook_calls == [("sd-turbo", 3 * GiB)])
    check("ram-only UPGRADED to GPU after reclaim -> effective device cuda",
          ig._PIPELINE_DEVICE["sd-turbo"] == "cuda")
    check("loaded fp16 on the GPU (not fp32, not quantized)",
          _FakeAuto.last_kwargs["torch_dtype"] == "f16"
          and "quantization_config" not in _FakeAuto.last_kwargs)
    check("the whole pipeline went to the GPU", pipe.calls == [("to", "cuda")])
finally:
    _set_ngl(None)
    _restore(state)


# --------------------------------------------------------------------------- #
# ram-only STAYS CPU when nothing eligible frees enough (honest fallback)
# --------------------------------------------------------------------------- #
state = []
free_box = [int(0.6 * GiB)]
_install(state, free_box=free_box)
pipe = _FakePipe(); _FakeAuto._next = pipe; _FakeAuto.last_kwargs = None
_set_ngl("off")
def _gated_hook(mk, need, job_id=None):
    # only a busy resident; frees nothing
    return {"evicted": [], "reached": False,
            "skipped": [{"model_key": "flux2-klein",
                         "reason": "eviction gated: in-flight", "host_mode": "slot"}]}
ig.set_imagegen_headroom_hook(_gated_hook)
try:
    out_pipe, placement = ig._load_diffusers_pipeline(
        _FakeAuto, "/fake/dir", "sd-turbo", place_fn=ig._place_diffusers_pipeline)
    check("ram-only that still can't fit -> effective device cpu",
          ig._PIPELINE_DEVICE["sd-turbo"] == "cpu")
    check("CPU fallback loads fp32, no quant, fully on CPU",
          _FakeAuto.last_kwargs["torch_dtype"] == "f32"
          and "quantization_config" not in _FakeAuto.last_kwargs
          and pipe.calls == [("to", "cpu")] and placement == "cpu")
    check("the un-evictable resident is remembered for an honest error",
          any(s["model_key"] == "flux2-klein"
              for s in (ig._LAST_HEADROOM["sd-turbo"].get("skipped") or [])))
finally:
    _set_ngl(None)
    _restore(state)


# --------------------------------------------------------------------------- #
# honest OOM error NAMES what's resident and why it couldn't be evicted
# --------------------------------------------------------------------------- #
_set_ngl("off")
ig._LAST_HEADROOM["mX"] = {
    "evicted": ["coldish"],
    "skipped": [{"model_key": "BUSY", "reason": "eviction gated: in-flight"}]}
try:
    honest = ig._honest_vram_error(
        RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB."),
        "mX", "generation")
    msg = str(honest)
    check("honest error names the resident that could NOT be evicted + why",
          "could NOT evict: BUSY (eviction gated: in-flight)" in msg)
    check("honest error notes what WAS evicted but still fell short",
          "evicted ['coldish'] but still short" in msg)
finally:
    _set_ngl(None)
    ig._LAST_HEADROOM.pop("mX", None)


# --------------------------------------------------------------------------- #
# the OOM retry seam also runs evict-to-fit (so the retry has room)
# --------------------------------------------------------------------------- #
state = []
free_box = [1 * GiB]
_install(state, free_box=free_box)
os.environ["HUGPY_VRAM_RETRY_SETTLE_S"] = "0"
ig._MODEL_NEED["retry-m"] = 5 * GiB
retry_calls = []
ig.set_imagegen_headroom_hook(
    lambda mk, need, job_id=None: retry_calls.append((mk, need)) or {"evicted": []})
try:
    ig._settle_for_vram_retry({}, threading.Lock(), "retry-m")
    check("the retry seam re-drives evict-to-fit with the model's priced need",
          retry_calls == [("retry-m", 5 * GiB)])
finally:
    os.environ.pop("HUGPY_VRAM_RETRY_SETTLE_S", None)
    ig._MODEL_NEED.pop("retry-m", None)
    _restore(state)


print(f"\nall {ok} checks passed")
