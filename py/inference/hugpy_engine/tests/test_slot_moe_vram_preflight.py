"""TASK D (2026-07-25) — refuse an impossible ngl=-1 load fast instead of
stalling.

Field gap this closes: an MoE model launched with --n-gpu-layers -1 and no
--n-cpu-moe on a card far too small (coder-next, 48.4GB / 48 layers, on ae's
24GB 3090) doesn't OOM immediately — it stalls health-checking for ~10 minutes
across 8 retries before the caller gives up and falls back. _build_cmd already
had a fail-fast RAM refusal for the OPPOSITE case (ngl<=0, no GPU offload, not
enough host RAM); this suite covers its new inverse-VRAM sibling.

Contract under test (managers/serve/slot_agent.py, the `ngl == -1 and not
eff_n_cpu_moe` block right after the existing RAM preflight):
  * refuses (RuntimeError, clear/actionable message) when ngl is effectively
    -1, no MoE split is configured, and total GGUF bytes clearly exceed the
    measurable VRAM (free preferred, else total) by the margin;
  * NEVER refuses when VRAM is unmeasurable (both free and total None) —
    degrade honestly, don't block on missing data;
  * NEVER fires when an MoE split IS configured (eff_n_cpu_moe truthy) — the
    MoE auto/explicit/pin-gpu branches already own that math;
  * NEVER fires for a normal partial/positive ngl (only the -1 "all layers"
    case is in scope);
  * a model that DOES fit the card (need comfortably under VRAM) loads clean.

Run: venv/bin/python -m pytest tests/test_slot_moe_vram_preflight.py -q
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sa = importlib.import_module("hugpy_engine.serve.slot_agent")
spill = importlib.import_module("hugpy_engine.spill")

GIB = 1 << 30


@pytest.fixture
def cmd_rig(monkeypatch, tmp_path):
    """Route _build_cmd's collaborators exactly like test_moe_placement's rig:
    a fake native llama-server binary, --n-cpu-moe reported supported, autofit
    fully controllable, and a real-but-tiny GGUF file on disk (existence is
    checked; SIZE is separately monkeypatched via sa._total_gguf_bytes so the
    test doesn't depend on writing multi-GB fixtures)."""
    serve = importlib.import_module("hugpy_engine.serve.serve")
    monkeypatch.setattr(serve, "LLAMA_SERVER_BIN", "/bin/echo")
    monkeypatch.setattr(sa, "_server_supports_flag", lambda b, f: True)
    auto = {"value": -1}
    # **_kw so the stub tracks the real signature as it grows (n_ctx joined it
    # 2026-07-27 when the context reserve became per-model) — this suite is
    # about the inverse-VRAM guard, never about autofit's own arithmetic.
    monkeypatch.setattr(spill, "autofit_gpu_layers",
                        lambda p, free_vram=None, extra_reserve_bytes=0,
                        **_kw: auto["value"])
    # Dense (gguf_moe_detail sees no expert tensors) so the AUTO MoE-split
    # branches inside _build_cmd stay inert — this suite is about the NEW
    # inverse-VRAM guard, not the MoE auto-placement policy.
    monkeypatch.setattr(spill, "gguf_moe_detail", lambda path: {"is_moe": False})
    for env in ("HUGPY_ALLOC_MODE", "HUGPY_N_CPU_MOE", "HUGPY_N_GPU_LAYERS",
                "HUGPY_HOT_CACHE_ROOT", "HUGPY_MODEL_CACHE"):
        monkeypatch.delenv(env, raising=False)
    p = tmp_path / "model.gguf"
    p.write_bytes(b"\0" * 4096)              # real, non-empty, tiny placeholder
    monkeypatch.setattr(sa, "_total_gguf_bytes", lambda path: rig.need["value"])
    rig = type("Rig", (), {"auto": auto, "path": str(p),
                           "need": {"value": 48 * int(1e9)}})()
    return rig


def _set_vram(monkeypatch, *, free=None, total=None):
    monkeypatch.setattr(spill, "free_vram_bytes", lambda: free)
    monkeypatch.setattr(spill, "total_vram_bytes", lambda: total)


def test_ram_budget_uses_binary_gib(cmd_rig, monkeypatch):
    _set_vram(monkeypatch, free=24 * GIB, total=24 * GIB)
    monkeypatch.setattr(spill, "cpu_resident_bytes", lambda path, ngl: GIB)
    # One GiB must fit exactly; interpreting the budget as 1e9 bytes refuses.
    sa._build_cmd("qwen", n_gpu_layers=1, path=cmd_rig.path, cpu_mem_gib=1)
    with pytest.raises(RuntimeError, match="RAM budget"):
        sa._build_cmd("qwen", n_gpu_layers=1, path=cmd_rig.path, cpu_mem_gib=0.9)


# ═══════════ the refusal fires ══════════════════════════════════════════════
def test_refuses_impossible_ngl_minus_1_no_moe_split(cmd_rig, monkeypatch):
    """The coder-next/ae shape: 48GB model, -1, no split, 24GB card."""
    _set_vram(monkeypatch, free=24 * GIB, total=24 * GIB)
    with pytest.raises(RuntimeError) as exc:
        sa._build_cmd("coder-next", n_gpu_layers=-1, path=cmd_rig.path)
    msg = str(exc.value).lower()
    assert "vram" in msg
    assert "coder-next" in str(exc.value)
    assert "n_cpu_moe" in msg or "moe" in msg   # actionable: points at the fix


def test_refusal_uses_free_vram_when_available(cmd_rig, monkeypatch):
    # total is huge (misleading) but FREE is the real constraint (something
    # else resident) — the guard must prefer free over total when both exist.
    _set_vram(monkeypatch, free=2 * GIB, total=24 * GIB)
    with pytest.raises(RuntimeError):
        sa._build_cmd("coder-next", n_gpu_layers=-1, path=cmd_rig.path)


def test_refusal_falls_back_to_total_when_free_unmeasurable(cmd_rig, monkeypatch):
    _set_vram(monkeypatch, free=None, total=24 * GIB)
    with pytest.raises(RuntimeError):
        sa._build_cmd("coder-next", n_gpu_layers=-1, path=cmd_rig.path)


# ═══════════ the refusal must NOT fire ══════════════════════════════════════
def test_never_refuses_when_vram_unmeasurable(cmd_rig, monkeypatch):
    """Degrade-honestly doctrine: no GPU visibility must never block a load."""
    _set_vram(monkeypatch, free=None, total=None)
    argv, ngl, *_rest = sa._build_cmd("coder-next", n_gpu_layers=-1,
                                      path=cmd_rig.path)
    assert ngl == -1
    assert "-m" in argv                          # actually built a command


def test_never_fires_when_moe_split_is_configured(cmd_rig, monkeypatch):
    """Same impossible-looking size, but an explicit n_cpu_moe is set — the
    model is NOT meant to be fully GPU-resident, so total-bytes-vs-VRAM is the
    wrong comparison. Must not refuse."""
    _set_vram(monkeypatch, free=24 * GIB, total=24 * GIB)
    argv, ngl, *_rest, ncm = sa._build_cmd(
        "coder-next", n_gpu_layers=-1, n_cpu_moe=999, path=cmd_rig.path)
    assert ngl == -1 and ncm == 999


def test_never_fires_for_partial_positive_ngl(cmd_rig, monkeypatch):
    """Only the ngl==-1 (all-layers) case is in scope; a normal partial
    layer count must never trip this guard even if it's VRAM-tight."""
    _set_vram(monkeypatch, free=1 * GIB, total=1 * GIB)
    argv, ngl, *_rest = sa._build_cmd("coder-next", n_gpu_layers=17,
                                      path=cmd_rig.path)
    assert ngl == 17


def test_model_that_fits_loads_clean(cmd_rig, monkeypatch):
    """A model comfortably under the card's VRAM at ngl=-1 must load, not
    refuse — the guard has a clear margin, not a hair-trigger."""
    cmd_rig.need["value"] = int(4 * 1e9)          # ~4GB model
    _set_vram(monkeypatch, free=24 * GIB, total=24 * GIB)
    argv, ngl, *_rest = sa._build_cmd("small-model", n_gpu_layers=-1,
                                      path=cmd_rig.path)
    assert ngl == -1


def test_never_fires_when_total_bytes_unmeasurable(cmd_rig, monkeypatch):
    """A path _total_gguf_bytes can't size (e.g. stat failure after the initial
    existence check) must not fabricate a refusal from no data."""
    cmd_rig.need["value"] = None
    _set_vram(monkeypatch, free=1 * GIB, total=1 * GIB)
    argv, ngl, *_rest = sa._build_cmd("coder-next", n_gpu_layers=-1,
                                      path=cmd_rig.path)
    assert ngl == -1


def test_margin_avoids_false_refusal_near_the_boundary(cmd_rig, monkeypatch):
    """A model just slightly over the free VRAM (well inside the 1.15x
    margin) must not be refused — only a CLEAR, multi-GB-scale mismatch is."""
    cmd_rig.need["value"] = int(1.05 * 24 * 1e9)
    _set_vram(monkeypatch, free=24 * GIB, total=24 * GIB)
    argv, ngl, *_rest = sa._build_cmd("coder-next", n_gpu_layers=-1,
                                      path=cmd_rig.path)
    assert ngl == -1
