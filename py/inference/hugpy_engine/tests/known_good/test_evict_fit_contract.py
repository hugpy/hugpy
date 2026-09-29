"""KNOWN-GOOD CONTRACT — fit planning (engine side).

Catalogue: docs/KNOWN-GOOD-CORE.md (area "evict + fit / allocation").
Source under test: hugpy_engine/spill.py (moe_dense_first_plan, moe_split_need,
kv_bytes), hugpy_engine/serve/slot_agent.py (_gpu_only_moe_plan),
hugpy_engine/llama/runners/src/python_runner.py + llama/runners/get.py
(vision GGUF refuses the in-process fallback).

Deterministic: the MoE detail is a synthetic dict (no GGUF read), the vision
model directory is a tmp dir with a fake projector, llama_cpp is a stub module.
"""
from __future__ import annotations

import importlib
import sys
import types

import pytest

spill = importlib.import_module("hugpy_engine.spill")
sa = importlib.import_module("hugpy_engine.serve.slot_agent")

MIB = 1 << 20
GIB = 1 << 30


# ---------------------------------------------------------------------------
# MoE dense-backbone-first split — the Coder-Next shape
# ---------------------------------------------------------------------------
def _coder_next_like():
    """48 MoE blocks, 1000 MiB of expert tensors per block, a 2048 MiB dense
    backbone: ~48.9 GiB of weights, the shape of Qwen3-Coder-Next-GGUF on
    ae-worker (RSS mapped ~48.4 GiB; explicit split n_cpu_moe=39).  Sizes are
    in MiB so the arithmetic below is exact."""
    per_layer = 1000 * MIB
    layers = {i: per_layer for i in range(48)}
    return {"is_moe": True, "expert_count": 512, "expert_used": 10,
            "non_expert_bytes": 2048 * MIB,
            "expert_bytes": 48 * per_layer,
            "expert_bytes_by_layer": layers}


def test_moe_plan_numbers_for_a_coder_next_style_split():
    """INVARIANT (fit planner arithmetic): with a 17.7 GiB GPU budget and a
    7 GiB KV/context reserve, the dense backbone (2 GiB) is placed first, the
    reserve is charged before any expert lands, and the remainder buys expert
    layers from the TOP block index down. Numbers: 8 expert layers on GPU,
    ``--n-cpu-moe 40``, ~9.8 GiB GPU weights, ~39 GiB RAM experts.
    Established: k53 dense-first ruling 2026-07-31; live Coder-Next split
    recorded in the frontier handoff 2026-09-28 (planned ~17.7 GiB GPU +
    ~37.7 GiB RAM)."""
    d = _coder_next_like()
    budget = int(17.7 * GIB)
    reserve = 7 * GIB
    plan = spill.moe_dense_first_plan(d, budget, extra_reserve_bytes=reserve)
    assert plan["dense_fits"] is True
    assert plan["budget_bytes"] == budget - reserve
    assert plan["expert_layers_on_gpu"] == 8
    assert plan["n_cpu_moe"] == 40                      # blocks 0..39 experts → CPU
    assert plan["cpu_bytes"] == 40 * 1000 * MIB
    assert plan["gpu_bytes"] == 2048 * MIB + 8 * 1000 * MIB
    # The split need for that n_cpu_moe prices exactly the same placement.
    need = spill.moe_split_need(d, plan["n_cpu_moe"])
    assert need == {"cpu_bytes": plan["cpu_bytes"], "gpu_bytes": plan["gpu_bytes"],
                    "layers_on_cpu": 40}
    # Weights total is what the RSS mapping reports (mmap of the whole file).
    assert d["non_expert_bytes"] + d["expert_bytes"] == plan["cpu_bytes"] + plan["gpu_bytes"]


def test_moe_plan_backbone_first_and_reserve_charged_first():
    """INVARIANT: the dense backbone is always first in line; a budget that
    cannot hold it reports dense_fits=False and spills every expert; a reserve
    that eats the whole budget yields the all-experts-to-CPU sentinel.
    Established: k53 (2026-07-31)."""
    d = _coder_next_like()
    tiny = spill.moe_dense_first_plan(d, 1 * GIB)
    assert tiny["dense_fits"] is False and tiny["n_cpu_moe"] == spill.MOE_ALL_LAYERS
    assert tiny["cpu_bytes"] == d["expert_bytes"]
    eaten = spill.moe_dense_first_plan(d, 10 * GIB, extra_reserve_bytes=10 * GIB)
    assert eaten["n_cpu_moe"] == spill.MOE_ALL_LAYERS
    whole = spill.moe_dense_first_plan(d, 60 * GIB)
    assert whole["n_cpu_moe"] == 0 and whole["cpu_bytes"] == 0
    assert spill.moe_dense_first_plan({"is_moe": False}, 60 * GIB) is None


def test_gpu_only_moe_is_all_or_bust_never_a_silent_split():
    """INVARIANT: alloc mode gpu-only on a MoE that does not fit WHOLE raises
    GpuOnlyInfeasible naming max-gpu as the remedy; it never serves a partial
    expert split under the name "only". A fitting model yields n_cpu_moe 0;
    an unmeasurable card obeys the mode literally.
    Established: operator ruling 2026-07-31 (slot_agent._gpu_only_moe_plan)."""
    d = _coder_next_like()
    with pytest.raises(sa.GpuOnlyInfeasible) as ei:
        sa._gpu_only_moe_plan("coder-next", d, 20 * GIB, "free card")
    assert "max-gpu" in str(ei.value)
    assert sa._gpu_only_moe_plan("coder-next", d, 60 * GIB, "free card")["n_cpu_moe"] == 0
    assert sa._gpu_only_moe_plan("coder-next", d, None, "?")["n_cpu_moe"] == 0


# ---------------------------------------------------------------------------
# ctx_pct is a KV-cache (memory) contract, linear in context
# ---------------------------------------------------------------------------
def test_kv_bytes_is_linear_in_context_so_ctx_pct_scales_fit():
    """INVARIANT: KV bytes = 2 · layers · ctx · kv_heads · head_dim · dtype and
    are LINEAR in ctx tokens, so a ctx_pct target scales the KV term of the fit
    (GGUF and Transformers alike). ctx<=0 → None; missing geometry never
    silently prices zero.
    Established: slice 11 / t27 (2026-07-17); frontier handoff 2026-09-28
    ("Context affects KV-cache memory and therefore total GPU/RAM fit")."""
    full = spill.kv_bytes(ctx_tokens=262144, n_layers=48, n_kv_heads=8, head_dim=128)
    assert full == 2 * 48 * 262144 * 8 * 128 * 2
    half = spill.kv_bytes(ctx_tokens=131072, n_layers=48, n_kv_heads=8, head_dim=128)
    assert half * 2 == full
    assert spill.kv_bytes(ctx_tokens=0) is None
    assert spill.kv_bytes(ctx_tokens=4096) and spill.kv_bytes(ctx_tokens=4096) > 0


# ---------------------------------------------------------------------------
# GGUF vision + mmproj → native slot only
# ---------------------------------------------------------------------------
@pytest.fixture
def vl_dir(tmp_path):
    d = tmp_path / "Qwen2.5-VL-7B-Instruct-GGUF"
    d.mkdir()
    model = d / "Qwen2.5-VL-7B-Instruct-Q6_K.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 4096)
    proj = d / "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"
    with open(proj, "wb") as fh:
        fh.truncate(3 * MIB)
    return types.SimpleNamespace(dir=str(d), model=str(model), proj=str(proj))


@pytest.fixture
def vision_env(monkeypatch, vl_dir):
    """The in-process runner's collaborators, stubbed: a fake llama_cpp whose
    constructor counts calls, a registry row for 'vl', tmp weights."""
    getmod = importlib.import_module("hugpy_engine.llama.runners.get")
    prmod = importlib.import_module("hugpy_engine.llama.runners.src.python_runner")
    slots = importlib.import_module("hugpy_engine.serve.slots")
    shard = importlib.import_module("hugpy_engine.llama.runners.src.shard_server")
    calls = {"llama": 0}

    class _Llama:
        def __init__(self, model_path, **kw):
            calls["llama"] += 1

    fake = types.ModuleType("llama_cpp")
    fake.Llama = _Llama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    cfg = types.SimpleNamespace(name="vl", tasks=["text-generation"])
    for mod in (getmod, prmod):
        monkeypatch.setattr(mod, "get_model_config", lambda k: cfg)
        monkeypatch.setattr(mod, "ensure_serving_weights", lambda k: vl_dir.dir)
        monkeypatch.setattr(mod, "get_gguf_file", lambda d, c, prefer=None: vl_dir.model)
    monkeypatch.setattr(spill, "llama_kwargs", lambda p: {})
    monkeypatch.setattr(spill, "cpu_resident_bytes", lambda p, n: 0)
    monkeypatch.setattr(spill, "gguf_moe_detail", lambda p: {"is_moe": False})
    monkeypatch.setattr(spill, "rpc_servers", lambda: None)
    monkeypatch.setattr(spill, "free_vram_bytes", lambda: 6 * GIB)
    monkeypatch.setattr(spill, "total_vram_bytes", lambda: 8 * GIB)
    monkeypatch.setattr(shard, "ensure_vision_server", lambda k: None)
    monkeypatch.setattr(getmod, "_require_profile_ready", lambda k: None)
    monkeypatch.setattr(getmod, "_slot_serves_vision", lambda base: True)
    monkeypatch.setattr(slots, "slots_enabled", lambda: True)
    monkeypatch.setattr(getmod, "_REFUSED", {})
    monkeypatch.setattr(getmod, "_LLAMA_INSTANCES", {})
    monkeypatch.delenv("HUGPY_INPROCESS_VISION", raising=False)
    return types.SimpleNamespace(calls=calls, prmod=prmod, getmod=getmod)


def test_gguf_vision_with_mmproj_refuses_inprocess_fallback(vision_env, vl_dir):
    """INVARIANT: a GGUF that ships an mmproj/projector sidecar is served ONLY
    by a native llama-server slot launched with --mmproj. The in-process
    llama-cpp-python runner refuses it (ModelLoadFailure, load_class
    'vision_needs_slot') and never constructs a Llama — because that path
    would silently ignore images and hallucinate from text.
    Established: vision_needs_slot (2026-07); frontier handoff 2026-09-28
    (computron Qwen2.5-VL-7B Q6 + mmproj finding)."""
    lf = importlib.import_module("hugpy_engine.serve.load_failure")
    with pytest.raises(lf.ModelLoadFailure) as ei:
        vision_env.prmod.LlamaCppPythonRunner("vl")
    assert ei.value.load_class == "vision_needs_slot"
    assert "--mmproj" in str(ei.value)
    assert vision_env.calls["llama"] == 0


def test_vision_refusal_message_states_the_fit_numbers(vision_env, vl_dir):
    """INVARIANT: the refusal names the projector and states the fit
    (weights + projector vs free VRAM) so an operator can act — the
    computron report (7.74 GiB needed, ~6.14 GiB free) is reconstructible
    from the message. Established: frontier handoff 2026-09-28."""
    lf = importlib.import_module("hugpy_engine.serve.load_failure")
    err = vision_env.getmod.vision_needs_slot_failure(
        "vl", vl_dir.model, vl_dir.proj, reason="slots busy")
    assert isinstance(err, lf.ModelLoadFailure) and err.load_class == "vision_needs_slot"
    msg = str(err)
    assert "projector" in msg and "free VRAM" in msg and "--mmproj" in msg
    assert "slots busy" in msg and "images silently ignored" in msg
