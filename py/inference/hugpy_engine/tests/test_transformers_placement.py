"""spill.transformers_max_memory — engine-agnostic PLACEMENT INTENT (t26).

The console's Autofit / Max GPU / CPU only controls ride the ONE wire field
HUGPY_N_GPU_LAYERS (-1 / "off" / "auto"). For a GGUF model it's a layer count;
for a TRANSFORMERS model there are no gpu layers to count, so the field is read
as placement and mapped onto device_map/max_memory:
  * -1   ("Max GPU")  -> all-on-GPU: NO cpu budget in max_memory
  * 0/"off" ("CPU only") -> CPU-only: gpu budget 0.00GiB (binds even with a GPU)
  * "auto"/unset      -> fit-and-spill: gpu budget + cpu budget (unchanged)
This closes the "un-hidden dead knob" gap — Max GPU / CPU only on a transformers
model now do something honest, not nothing.

Runs like the other tests here: venv/bin/python tests/test_transformers_placement.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib
sp = importlib.import_module("hugpy_engine.spill")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


def test_transformers_placement():
    """Converted from the script-style suite: every `check` is an assert."""


    def _set(**env):
        """Set/clear the spill env for one case (None clears)."""
        for k in ("HUGPY_N_GPU_LAYERS", "HUGPY_GPU_MEM_GIB", "HUGPY_CPU_MEM_GIB", "HUGPY_N_GPU"):
            os.environ.pop(k, None)
        for k, v in env.items():
            if v is not None:
                os.environ[k] = str(v)


    # Stub the hardware probes so the mapping is deterministic without a GPU: a 16 GiB
    # free-VRAM card and 40 GiB free RAM. (transformers_max_memory reads these.)
    _orig = (sp.free_vram_bytes, sp.free_ram_bytes)
    GIB = 2 ** 30
    sp.free_vram_bytes = lambda: 16 * GIB
    sp.free_ram_bytes = lambda: 40 * GIB

    try:
        # ── n_gpu_layers_intent decoding ─────────────────────────────────────────
        _set(HUGPY_N_GPU_LAYERS=None)
        check("intent: unset -> auto", sp.n_gpu_layers_intent() == "auto")
        _set(HUGPY_N_GPU_LAYERS="auto")
        check("intent: 'auto' -> auto", sp.n_gpu_layers_intent() == "auto")
        _set(HUGPY_N_GPU_LAYERS="-1")
        check("intent: '-1' -> gpu", sp.n_gpu_layers_intent() == "gpu")
        _set(HUGPY_N_GPU_LAYERS="off")
        check("intent: 'off' -> cpu", sp.n_gpu_layers_intent() == "cpu")
        _set(HUGPY_N_GPU_LAYERS="0")
        check("intent: '0' -> cpu", sp.n_gpu_layers_intent() == "cpu")
        _set(HUGPY_N_GPU_LAYERS="cpu")
        check("intent: 'cpu' -> cpu", sp.n_gpu_layers_intent() == "cpu")
        _set(HUGPY_N_GPU_LAYERS="20")
        check("intent: positive int -> auto (no transformers analogue)",
              sp.n_gpu_layers_intent() == "auto")
        _set(HUGPY_N_GPU_LAYERS="garbage")
        check("intent: garbage -> auto (fail open)", sp.n_gpu_layers_intent() == "auto")

        # ── AUTOFIT (auto): gpu budget + cpu budget (today's behavior) ───────────
        _set(HUGPY_N_GPU_LAYERS="auto")
        mm = sp.transformers_max_memory()
        check("auto: has a GPU budget", 0 in mm and mm[0] != "0.00GiB")
        check("auto: has a cpu spill budget", "cpu" in mm and mm["cpu"] != "0.00GiB")

        # ── MAX GPU (-1): all-on-GPU — NO cpu budget in the map ──────────────────
        _set(HUGPY_N_GPU_LAYERS="-1")
        mm = sp.transformers_max_memory()
        check("maxgpu: GPU budget present", 0 in mm and mm[0] != "0.00GiB")
        check("maxgpu: NO cpu budget (forces the model onto the card)", "cpu" not in mm)

        # maxgpu WITH an explicit cpu_mem_gib alongside -> the explicit budget wins.
        _set(HUGPY_N_GPU_LAYERS="-1", HUGPY_CPU_MEM_GIB="8")
        mm = sp.transformers_max_memory()
        check("maxgpu+explicit cpu: honors the explicit cpu budget",
              mm.get("cpu") == "8.00GiB")

        # ── CPU ONLY ("off"): gpu budget 0.00GiB, binds even with a GPU present ──
        _set(HUGPY_N_GPU_LAYERS="off")
        mm = sp.transformers_max_memory()
        check("cpu-only: returns a map (binds even with a GPU)", mm is not None)
        check("cpu-only: gpu budget is 0.00GiB (nothing on the card)",
              mm.get(0) == "0.00GiB")
        check("cpu-only: has a cpu budget", mm.get("cpu") not in (None, "0.00GiB"))

        # "0" is the same CPU-only intent.
        _set(HUGPY_N_GPU_LAYERS="0")
        mm = sp.transformers_max_memory()
        check("cpu-only via '0': gpu budget 0.00GiB", mm.get(0) == "0.00GiB")

        # ── explicit gpu_mem_gib still wins for the GPU axis under auto ──────────
        _set(HUGPY_N_GPU_LAYERS="auto", HUGPY_GPU_MEM_GIB="5")
        mm = sp.transformers_max_memory()
        check("auto+explicit gpu: honors the explicit gpu budget", mm.get(0) == "5.00GiB")

        # ── multi-GPU: the placement class applies to every card index ───────────
        _set(HUGPY_N_GPU_LAYERS="off", HUGPY_N_GPU="2")
        mm = sp.transformers_max_memory()
        check("cpu-only multi-gpu: both card indices are 0.00GiB",
              mm.get(0) == "0.00GiB" and mm.get(1) == "0.00GiB")

        # ── no GPU visible + auto -> None (unchanged: stay on CPU by default) ────
        sp.free_vram_bytes = lambda: None
        _set(HUGPY_N_GPU_LAYERS="auto")
        check("auto + no GPU -> None (CPU by default, unchanged)",
              sp.transformers_max_memory() is None)
        # ...but CPU-only intent still returns a binding map even with no GPU read.
        _set(HUGPY_N_GPU_LAYERS="off")
        mm = sp.transformers_max_memory()
        check("cpu-only + no GPU read -> still a binding CPU map",
              mm is not None and mm.get(0) == "0.00GiB")
        sp.free_vram_bytes = lambda: 16 * GIB
    finally:
        sp.free_vram_bytes, sp.free_ram_bytes = _orig
        for k in ("HUGPY_N_GPU_LAYERS", "HUGPY_GPU_MEM_GIB", "HUGPY_CPU_MEM_GIB", "HUGPY_N_GPU"):
            os.environ.pop(k, None)

    print(f"\nall {ok} checks passed")
