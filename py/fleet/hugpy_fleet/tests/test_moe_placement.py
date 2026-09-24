"""MoE-PLACEMENT productization (2026-07-24 measured win, operator-grounded).

Measured on ae/3090 (Qwen3-Coder-Next, 80B-A3B MoE): the naive 17/48 layer
split gave ~15.2 tok/s @ 16.6 GiB VRAM; the MoE-aware split (n_gpu_layers=-1 +
llama-server --n-cpu-moe 999: experts to CPU, everything else + KV on GPU) gave
~24.1 tok/s @ 3.2 GiB — +59% AND 5x less VRAM. This suite proves the feature:

  * GGUF MoE detection from synthetic headers the REAL reader parses
    (expert_count KV = the definition; `_exps` tensor suffix = the per-tensor
    is_expert bit; router/shexp never match), shard-aware, cached;
  * ground-truth reconciliation against the real coder-next shards when local
    (expert 43.59 GiB / non-expert 1.49 GiB / ec 512 / euc 10 — keeper-parsed);
  * per-layer-aware split pricing (spill.moe_split_need) incl. partial N;
  * DENSE BACKBONE FIRST (operator ruling 2026-07-31): the GPU budget buys the
    non-expert tensors before any expert layer, under EVERY alloc mode —
    spill.moe_dense_first_plan's arithmetic and the _build_cmd matrix it feeds
    (a stored ngl=-1 no longer disables the split; a stated layer count still
    wins; zero-GPU modes plan nothing; dense -> unchanged; explicit n_cpu_moe
    always wins; engine-degrade paths);
  * relaunch accepts n_cpu_moe; slot admission verdict threads it;
  * expert-aware need: the fit checks pass the empty-card 41.6GB-MoE case;
    _vram_evict_to_fit re-targets to the split instead of an impossible full
    fit; calibration verdict honesty; feasibility with MoE sizing;
  * the n_cpu_moe knob: overrides coercion + spill env wire.

Run: venv/bin/python -m pytest tests/test_moe_placement.py -q
"""
import importlib
import os
import struct
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import the agent module FIRST (codebase name-collision landmine — see the
# sibling tests) and bind the real dispatch module via import_module.
from hugpy_fleet.worker import agent as A
from hugpy_engine import spill
from hugpy_engine import alloc_modes as AM

sa = importlib.import_module("hugpy_engine.serve.slot_agent")
SL = importlib.import_module("hugpy_engine.serve.slots")
OV = importlib.import_module("hugpy_engine.serve.overrides")
D = importlib.import_module("hugpy_engine.dispatch.dispatch")

GIB = 1 << 30


@pytest.fixture(autouse=True)
def _no_env_leak():
    """Restore the HUGPY_* spill wire after every test.

    This suite exercises agent._apply_spill, whose JOB is to write the spill
    knobs into os.environ — so a test that drives it LEAKS them into the process
    for every suite that runs afterwards. Measured 2026-07-27: this file left
    HUGPY_GPU_MEM_GIB=0.0 behind, which makes autofit_gpu_layers cap free VRAM at
    zero and return 0 layers, and that is what fails
    test_spill_vision_fit::test_vision_ngl_arg_env_overrides in a whole-directory
    run (the recorded 'cross-file pollution' landmine). Snapshot/restore here, so
    the leak stops at the file boundary. Asserts nothing; changes no behaviour."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("HUGPY_")}
    try:
        yield
    finally:
        for k in [k for k in os.environ if k.startswith("HUGPY_")]:
            if k not in saved:
                del os.environ[k]
        os.environ.update(saved)


# ═══════════ synthetic GGUF fixtures (the real reader parses these) ═════════
def _mk_gguf(path, *, tensors=(), block_count=None, expert_count=None,
             expert_used=None, dims_by_name=None):
    """A minimal-but-real GGUF v3: magic/version/counts, a KV table (uint32
    values), a tensor-info table (name/dims/type/offset), aligned data section.
    Tensor sizes are realized by consecutive offsets + real file padding, which
    is exactly how the reader prices them (offset deltas, no type table).

    ``dims_by_name`` optionally supplies a real dims tuple per tensor name (e.g.
    ``(2048, 512, 512)`` for a stacked expert weight), so the SHAPE backstop can
    be exercised; tensors absent from the map keep the historical 1-D ``(1,)``
    (which the shape method — needing nd>=3 — correctly ignores)."""
    import io
    buf = io.BytesIO()
    dims_by_name = dims_by_name or {}

    def ws(s):
        b = s.encode()
        buf.write(struct.pack("<Q", len(b)))
        buf.write(b)

    kvs = []
    if block_count is not None:
        kvs.append(("fake.block_count", block_count))
    if expert_count is not None:
        kvs.append(("fake.expert_count", expert_count))
    if expert_used is not None:
        kvs.append(("fake.expert_used_count", expert_used))
    buf.write(b"GGUF")
    buf.write(struct.pack("<I", 3))
    buf.write(struct.pack("<Q", len(tensors)))
    buf.write(struct.pack("<Q", len(kvs)))
    for key, val in kvs:
        ws(key)
        buf.write(struct.pack("<I", 5))          # uint32
        buf.write(struct.pack("<I", val))
    off = 0
    for name, size in tensors:
        ws(name)
        dims = dims_by_name.get(name, (1,))      # default 1-D (unused by name path)
        buf.write(struct.pack("<I", len(dims)))  # n_dims
        for d in dims:
            buf.write(struct.pack("<Q", d))
        buf.write(struct.pack("<I", 0))          # ggml type (unused)
        buf.write(struct.pack("<Q", off))
        off += size
    header = buf.getvalue()
    data_start = (len(header) + 31) // 32 * 32   # general.alignment default 32
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(b"\0" * (data_start - len(header)))
        fh.write(b"\0" * off)
    return str(path)


_MOE_TENSORS = (
    ("token_embd.weight", 512),
    ("blk.0.attn_q.weight", 1000),
    ("blk.0.ffn_gate_inp.weight", 96),           # router — NOT an expert
    ("blk.0.ffn_up_shexp.weight", 224),          # shared expert — NOT an expert
    ("blk.0.ffn_up_exps.weight", 4000),
    ("blk.0.ffn_down_exps.weight", 4000),
    ("blk.1.attn_q.weight", 1000),
    ("blk.1.ffn_gate_exps.weight", 8000),
    ("output.weight", 500),
)
# expert = 16000; non-expert = 512+1000+96+224+1000+500 = 3332


@pytest.fixture
def moe_gguf(tmp_path):
    return _mk_gguf(tmp_path / "moe.gguf", tensors=_MOE_TENSORS,
                    block_count=48, expert_count=512, expert_used=10)


@pytest.fixture
def dense_gguf(tmp_path):
    tensors = tuple((n, s) for n, s in _MOE_TENSORS if "_exps" not in n)
    return _mk_gguf(tmp_path / "dense.gguf", tensors=tensors, block_count=48)


# ═══════════ detection + byte split ═════════════════════════════════════════
def test_detects_moe_and_splits_bytes_by_kind(moe_gguf):
    d = spill.gguf_moe_detail(moe_gguf)
    assert d["is_moe"] is True
    assert d["expert_count"] == 512
    assert d["expert_used_count"] == 10
    assert d["sparsity"] == pytest.approx(10 / 512)
    assert d["expert_bytes"] == 16000
    assert d["non_expert_bytes"] == 3332
    # per-layer attribution (the future-partial pricing input)
    assert d["expert_bytes_by_layer"] == {0: 8000, 1: 8000}


def test_router_and_shared_experts_are_not_experts(moe_gguf):
    d = spill.gguf_moe_detail(moe_gguf)
    # ffn_gate_inp (router) + ffn_up_shexp (shared) priced GPU-side: their
    # bytes are inside non_expert_bytes, never expert_bytes.
    assert d["expert_bytes"] == 16000            # only the _exps tensors


def test_absent_expert_count_is_dense(dense_gguf):
    d = spill.gguf_moe_detail(dense_gguf)
    assert d["is_moe"] is False
    assert d["expert_bytes"] == 0


def test_zero_expert_count_is_dense(tmp_path):
    p = _mk_gguf(tmp_path / "zero.gguf", tensors=_MOE_TENSORS,
                 block_count=48, expert_count=0)
    assert spill.gguf_moe_detail(p)["is_moe"] is False


def test_missing_or_garbage_file_degrades_to_dense(tmp_path):
    assert spill.gguf_moe_detail("/nope/never.gguf") == {"is_moe": False}
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"not a gguf at all")
    assert spill.gguf_moe_detail(str(junk))["is_moe"] is False


def test_shard_aware_sums_across_shards(tmp_path):
    s1 = _mk_gguf(tmp_path / "m-00001-of-00002.gguf",
                  tensors=_MOE_TENSORS[:5], block_count=48,
                  expert_count=512, expert_used=10)
    _mk_gguf(tmp_path / "m-00002-of-00002.gguf", tensors=_MOE_TENSORS[5:])
    d = spill.gguf_moe_detail(s1)
    assert d["files"] == 2
    assert d["is_moe"] is True
    assert d["expert_bytes"] == 16000            # summed across both shards
    assert d["non_expert_bytes"] == 3332
    assert d["expert_bytes_by_layer"] == {0: 8000, 1: 8000}


def test_detail_is_cached_by_path_signature(moe_gguf, monkeypatch):
    first = spill.gguf_moe_detail(moe_gguf)

    def _boom(_p):
        raise AssertionError("re-parsed a cached header")
    from hugpy_storage import gguf_inspect as _gi
    monkeypatch.setattr(_gi, "_gguf_scan_moe", _boom)
    assert spill.gguf_moe_detail(moe_gguf) == first


# ═══════════ shape-derived detection (name-independent backstop) ════════════
# Empirically grounded 2026-07-24 against the real coder-next shards: an expert
# tensor is the STACKED one — n_dims>=3 with expert_count in the LAST dims slot
# (dims[-1]). These tensors are named unconventionally (NO _exps suffix) so ONLY
# the shape method can find them; their dims carry expert_count=512 in dims[-1].
_SHAPE_ONLY_TENSORS = (
    ("token_embd.weight", 512),
    ("blk.0.attn_q.weight", 1000),
    ("blk.0.router.weight", 96),                 # router — 2-D, not an expert
    ("blk.0.moe_up.weight", 4000),               # expert by SHAPE (nd=3), no _exps
    ("blk.0.moe_down.weight", 4000),             # expert by SHAPE
    ("blk.1.attn_q.weight", 1000),
    ("blk.1.moe_gate.weight", 8000),             # expert by SHAPE
    ("output.weight", 500),
)
_SHAPE_DIMS = {                                  # dims[-1]==512 == expert_count
    "blk.0.moe_up.weight": (2048, 512, 512),
    "blk.0.moe_down.weight": (512, 2048, 512),
    "blk.1.moe_gate.weight": (2048, 512, 512),
    "blk.0.router.weight": (2048, 512),          # 2-D: dims[-1]==512 but NOT nd>=3
}


def test_shape_detects_experts_when_naming_is_nonstandard(tmp_path, caplog):
    """expert_count set, tensors shaped [.,.,512] but NOT named *_exps -> the
    shape backstop finds them; a single WARNING names the nonconforming file."""
    p = _mk_gguf(tmp_path / "shape.gguf", tensors=_SHAPE_ONLY_TENSORS,
                 block_count=48, expert_count=512, expert_used=10,
                 dims_by_name=_SHAPE_DIMS)
    with caplog.at_level("WARNING"):
        d = spill.gguf_moe_detail(p)
    assert d["is_moe"] is True
    assert d["expert_count"] == 512 and d["expert_used_count"] == 10
    # experts = moe_up + moe_down + moe_gate = 4000+4000+8000; 2-D router excluded
    assert d["expert_bytes"] == 16000
    assert d["non_expert_bytes"] == 512 + 1000 + 96 + 1000 + 500
    assert d["expert_bytes_by_layer"] == {0: 8000, 1: 8000}
    assert any("nonstandard converter" in r.message for r in caplog.records)


def test_shape_2d_dim_collision_is_not_an_expert(tmp_path):
    """The nd>=3 guard: a 2-D tensor whose dims[-1] happens to equal
    expert_count is NOT an expert (only stacked/3-D weights are). The router
    here is 2-D (2048,512) -> excluded, so no false expert bytes."""
    p = _mk_gguf(tmp_path / "shape.gguf", tensors=_SHAPE_ONLY_TENSORS,
                 block_count=48, expert_count=512, expert_used=10,
                 dims_by_name=_SHAPE_DIMS)
    d = spill.gguf_moe_detail(p)
    assert d["expert_bytes"] == 16000            # router's 96 bytes stay non-expert


def test_count_but_no_experts_anywhere_falls_back_to_dense(tmp_path, caplog):
    """Header claims MoE (expert_count) but NEITHER name nor shape finds an
    expert tensor -> dense fallback (safe plain split) + one WARNING."""
    plain = tuple((n, s) for n, s in _MOE_TENSORS if "_exps" not in n)
    p = _mk_gguf(tmp_path / "liar.gguf", tensors=plain,
                 block_count=48, expert_count=512, expert_used=10)
    with caplog.at_level("WARNING"):
        d = spill.gguf_moe_detail(p)
    assert d["is_moe"] is False                  # safe fallback: no mispriced split
    assert d["expert_bytes"] == 0
    assert any("no expert tensors identifiable" in r.message
               for r in caplog.records)


def test_names_without_expert_count_are_dense(tmp_path, caplog):
    """REVERSE inconsistency: _exps-named tensors but NO expert_count -> names
    alone never activate the split; dense + one WARNING (metadata is the gate)."""
    p = _mk_gguf(tmp_path / "names.gguf", tensors=_MOE_TENSORS, block_count=48)
    with caplog.at_level("WARNING"):
        d = spill.gguf_moe_detail(p)
    assert d["is_moe"] is False
    assert d["expert_bytes"] == 0                # metadata gate: no split
    assert any("no positive expert_count" in r.message for r in caplog.records)


def test_conforming_file_logs_no_warning(moe_gguf, caplog):
    """A file where name and shape AGREE (or the standard _exps path) is silent —
    warnings are for DRIFT only, never the happy path."""
    with caplog.at_level("WARNING"):
        spill.gguf_moe_detail(moe_gguf)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


# ═══════════ per-layer-aware split pricing ══════════════════════════════════
def test_moe_split_need_all_and_sentinel(moe_gguf):
    d = spill.gguf_moe_detail(moe_gguf)
    for n in (None, 999, spill.MOE_ALL_LAYERS, 2):
        got = spill.moe_split_need(d, n)
        assert got["cpu_bytes"] == 16000
        assert got["gpu_bytes"] == 3332


def test_moe_split_need_partial_is_per_layer_exact(moe_gguf):
    d = spill.gguf_moe_detail(moe_gguf)
    one = spill.moe_split_need(d, 1)             # first layer's experts only
    assert one == {"cpu_bytes": 8000, "gpu_bytes": 3332 + 8000,
                   "layers_on_cpu": 1}
    zero = spill.moe_split_need(d, 0)
    assert zero["cpu_bytes"] == 0
    assert zero["gpu_bytes"] == 16000 + 3332


def test_moe_split_need_dense_is_none():
    assert spill.moe_split_need({"is_moe": False}) is None
    assert spill.moe_split_need(None) is None


# ═══ DENSE BACKBONE FIRST — the plan itself (operator ruling 2026-07-31) ═════
# The backbone (attention/router/shared experts/embeddings/output + the KV that
# rides with it) is touched by EVERY token; an expert tensor by ~10/512 of one.
# So a GPU budget buys the backbone FIRST and the experts get the remainder —
# under every allocation mode. These pin the arithmetic; the _build_cmd tests
# below pin that every mode actually reaches it.
def test_plan_dense_only_budget_spills_every_expert(moe_gguf):
    d = spill.gguf_moe_detail(moe_gguf)
    plan = spill.moe_dense_first_plan(d, 3332 + 100)
    assert plan["n_cpu_moe"] == spill.MOE_ALL_LAYERS
    assert plan["dense_fits"] is True
    assert (plan["cpu_bytes"], plan["gpu_bytes"]) == (16000, 3332)


def test_plan_spends_the_remainder_on_the_TOP_expert_layers(moe_gguf):
    """--n-cpu-moe keeps the FIRST N block indices' experts on the CPU, so the
    layers that ride the GPU are a SUFFIX — the plan fills it from the top."""
    d = spill.gguf_moe_detail(moe_gguf)
    plan = spill.moe_dense_first_plan(d, 3332 + 8000)
    assert plan["n_cpu_moe"] == 1 and plan["expert_layers_on_gpu"] == 1
    assert (plan["cpu_bytes"], plan["gpu_bytes"]) == (8000, 3332 + 8000)


def test_plan_that_covers_everything_says_zero(moe_gguf):
    d = spill.gguf_moe_detail(moe_gguf)
    plan = spill.moe_dense_first_plan(d, 1 << 30)
    assert plan["n_cpu_moe"] == 0 and plan["cpu_bytes"] == 0


def test_plan_charges_the_extra_reserve_before_anything_lands(moe_gguf):
    d = spill.gguf_moe_detail(moe_gguf)
    plan = spill.moe_dense_first_plan(d, 3332 + 8000,
                                      extra_reserve_bytes=8000)
    assert plan["n_cpu_moe"] == spill.MOE_ALL_LAYERS


def test_plan_backbone_larger_than_the_budget_still_leads(moe_gguf):
    """A budget too small even for the backbone never promotes an expert layer
    ahead of it — the backbone still goes first and the shortfall is stated."""
    d = spill.gguf_moe_detail(moe_gguf)
    plan = spill.moe_dense_first_plan(d, 100)
    assert plan["n_cpu_moe"] == spill.MOE_ALL_LAYERS
    assert plan["dense_fits"] is False


def test_plan_threshold_is_a_block_index_not_a_position():
    """A model whose first blocks are dense would be mispriced by position:
    --n-cpu-moe counts BLOCK INDICES, so keeping block 9's experts on the card
    while spilling block 5's means the threshold is 9, not 1."""
    detail = {"is_moe": True, "expert_bytes": 300, "non_expert_bytes": 100,
              "expert_bytes_by_layer": {5: 100, 7: 100, 9: 100}}
    plan = spill.moe_dense_first_plan(detail, 100 + 100)
    assert plan["n_cpu_moe"] == 9 and plan["cpu_bytes"] == 200


def test_plan_is_none_for_a_dense_model(dense_gguf):
    assert spill.moe_dense_first_plan(spill.gguf_moe_detail(dense_gguf),
                                      1 << 30) is None
    assert spill.moe_dense_first_plan(None, 1 << 30) is None


# ═══════════ ground truth: the real coder-next shards ═══════════════════════
_REAL_SHARD1 = ("/mnt/llm_storage/legacy/Qwen3-Coder-Next-GGUF/"
                "Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf")


@pytest.mark.skipif(not Path(_REAL_SHARD1).is_file(),
                    reason="real coder-next shards not present on this box")
def test_real_coder_next_header_reconciliation():
    """Keeper-parsed ground truth (2026-07-24, reconciled against the live ae
    card within ~0.2 GiB): expert 43.59 GiB (96.7%), non-expert 1.49 GiB,
    expert_count 512, expert_used_count 10, 48 expert-bearing blocks. The
    non-expert share + KV is what the measured 3.2 GiB VRAM footprint is."""
    d = spill.gguf_moe_detail(_REAL_SHARD1)
    assert d["is_moe"] is True and d["files"] == 4
    assert d["expert_count"] == 512 and d["expert_used_count"] == 10
    assert d["expert_bytes"] / GIB == pytest.approx(43.59, abs=0.05)
    assert d["non_expert_bytes"] / GIB == pytest.approx(1.49, abs=0.05)
    assert len(d["expert_bytes_by_layer"]) == 48
    need = spill.moe_split_need(d)
    assert need["gpu_bytes"] == d["non_expert_bytes"]
    assert need["cpu_bytes"] == d["expert_bytes"]


@pytest.mark.skipif(not Path(_REAL_SHARD1).is_file(),
                    reason="real coder-next shards not present on this box")
def test_real_shard_name_and_shape_agree_exactly():
    """The strongest consistency proof: on the REAL file the NAME method and the
    SHAPE backstop must select the IDENTICAL expert set (same bytes, same
    per-layer map, same hit count). Verified 2026-07-24: 144 _exps tensors, all
    3-D with dims[-1]==expert_count=512; a conforming file logs NO warning."""
    shards = spill._gguf_shard_paths(_REAL_SHARD1)
    n_name = n_shape = 0
    b_name = b_shape = 0
    ec = None
    for sh in shards:
        # Thread the count forward exactly as gguf_moe_detail does (only shard 1
        # carries expert_count; the hint lets shape fire on later shards too).
        scan = spill._gguf_scan_moe(sh, expert_count_hint=ec)
        if ec is None and scan.get("expert_count") is not None:
            ec = int(scan["expert_count"])
        n_name += scan["name_expert_hits"]
        n_shape += scan["shape_expert_hits"]
        b_name += scan["expert_bytes"]
        b_shape += scan["expert_bytes_shape"]
    assert n_name == n_shape == 144              # identical tensor set, both ways
    assert b_name == b_shape                     # byte-identical split
    assert (b_name / GIB) == pytest.approx(43.59, abs=0.05)


# ═══════════ _build_cmd — THE argv choke point (loads AND relaunches) ═══════
@pytest.fixture
def cmd_rig(monkeypatch, moe_gguf, dense_gguf):
    """Route _build_cmd's collaborators: a fake native llama-server that exists
    (/bin/echo), --n-cpu-moe supported, autofit controllable."""
    serve = importlib.import_module("hugpy_engine.serve.serve")
    monkeypatch.setattr(serve, "LLAMA_SERVER_BIN", "/bin/echo")
    monkeypatch.setattr(sa, "_server_supports_flag", lambda b, f: True)
    auto = {"value": 17}
    # The card, EXPLICITLY (k53). The dense-first plan prices the expert split
    # against a real VRAM budget, so leaving that to whatever GPU the test box
    # happens to have would make this suite's verdicts depend on the machine.
    # Default None = "unmeasurable", which is the historical degrade (backbone
    # on the GPU, all experts to CPU) — so every pre-k53 expectation below is
    # unchanged, and the budget-aware tests opt in by setting rig.vram["free"].
    vram = {"free": None}
    monkeypatch.setattr(spill, "free_vram_bytes", lambda: vram["free"])
    monkeypatch.setattr(spill, "vram_ctx_reserve_bytes",
                        lambda p, n_ctx=None: (0, "test", {}))
    # **_kw so the stub tracks the real signature as it grows (n_ctx joined it
    # 2026-07-27 when the context reserve became per-model) — this rig is about
    # the MoE placement policy, never about autofit's own arithmetic.
    monkeypatch.setattr(spill, "autofit_gpu_layers",
                        lambda p, free_vram=None, extra_reserve_bytes=0,
                        **_kw: auto["value"])
    for env in ("HUGPY_ALLOC_MODE", "HUGPY_N_CPU_MOE", "HUGPY_N_GPU_LAYERS",
                "HUGPY_HOT_CACHE_ROOT", "HUGPY_MODEL_CACHE"):
        monkeypatch.delenv(env, raising=False)
    return type("Rig", (), {"auto": auto, "vram": vram, "moe": moe_gguf,
                            "dense": dense_gguf})()


def _argv_pairs(argv):
    return {argv[i]: argv[i + 1] for i in range(0, len(argv) - 1)}


def test_auto_policy_moe_hybrid_becomes_expert_split(cmd_rig):
    cmd_rig.auto["value"] = 17                   # hybrid: partial layer split
    (argv, ngl, _c, _t, _cp, kind, total, ncm) = sa._build_cmd(
        "moe-model", path=cmd_rig.moe)
    pairs = _argv_pairs(argv)
    assert kind == "binary"
    # k64: the rig's fit-capable binary parses -1 as AUTO (llama.cpp params-fit
    # vocabulary), so "all layers" is spelled "all" in argv; the returned ngl
    # stays -1 (the internal encoding). --fit off keeps placement OURS.
    assert ngl == -1 and pairs["--n-gpu-layers"] == "all"
    assert pairs["--fit"] == "off"
    assert ncm == spill.MOE_ALL_LAYERS and pairs["--n-cpu-moe"] == "999"
    assert total == 48


def test_auto_policy_moe_fits_whole_STILL_SPLITS(cmd_rig):
    """POLICY 2026-07-25: the split is the DEFAULT for every applicable GGUF.

    A MoE that fits the card whole used to be pinned fully-on-GPU (moe_mode
    "pin-gpu", --n-cpu-moe 0). The ae measurement retires that exception: the
    split is BOTH faster (+59% tok/s) AND ~5x cheaper in VRAM, so pinning was
    strictly worse on both axes and monopolised a card other models could use.
    """
    cmd_rig.auto["value"] = -1                   # whole model fits
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
    pairs = _argv_pairs(argv)
    assert ngl == -1 and pairs["--n-gpu-layers"] == "all"
    assert ncm == spill.MOE_ALL_LAYERS and pairs["--n-cpu-moe"] == "999"


def test_auto_policy_moe_splits_at_every_autofit_verdict(cmd_rig):
    """The autofit verdict no longer gates the split at all — hybrid (17/48),
    fits-whole (-1) and nothing-fits (0) all land on the same default."""
    for verdict in (-1, 0, 1, 17, 47):
        cmd_rig.auto["value"] = verdict
        (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
        assert (ngl, ncm) == (-1, spill.MOE_ALL_LAYERS), f"autofit={verdict}"


def test_auto_policy_moe_fits_whole_degrades_when_experts_exceed_ram(cmd_rig,
                                                                     monkeypatch):
    """Viability is the ONE remaining condition: experts must fit budgetable
    host RAM. When they don't, fall back to the autofit placement — never
    refuse, never move bytes into RAM that isn't there."""
    monkeypatch.setattr(sa, "_mem_available_bytes", lambda: 1024)  # ~1 KB free
    monkeypatch.setattr(spill, "ram_reserve_bytes", lambda: 0)
    cmd_rig.auto["value"] = -1                   # whole model fits the card
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
    assert ngl == -1 and ncm is None and "--n-cpu-moe" not in argv
    cmd_rig.auto["value"] = 17                   # hybrid degrades to layer split
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
    assert ngl == 17 and ncm is None and "--n-cpu-moe" not in argv


def test_auto_policy_unmeasurable_ram_still_splits(cmd_rig, monkeypatch):
    """Degrade honestly: an unreadable /proc/meminfo must never block the
    default (never refuse/downgrade on unmeasurable data)."""
    monkeypatch.setattr(sa, "_mem_available_bytes", lambda: None)
    cmd_rig.auto["value"] = -1
    (_argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
    assert ngl == -1 and ncm == spill.MOE_ALL_LAYERS


def test_auto_policy_dense_is_byte_identical(cmd_rig):
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, ncm) = sa._build_cmd("dense-model", path=cmd_rig.dense)
    assert ngl == 17 and ncm is None
    assert "--n-cpu-moe" not in argv


def test_explicit_n_gpu_layers_wins_over_auto_split(cmd_rig):
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", n_gpu_layers=20,
                                             path=cmd_rig.moe)
    assert ngl == 20 and ncm is None
    assert "--n-cpu-moe" not in argv


def test_explicit_n_cpu_moe_always_wins(cmd_rig):
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", n_gpu_layers=-1,
                                             n_cpu_moe=12, path=cmd_rig.moe)
    pairs = _argv_pairs(argv)
    assert ngl == -1 and ncm == 12 and pairs["--n-cpu-moe"] == "12"


def test_explicit_n_cpu_moe_zero_pins_experts_on_gpu_at_every_autofit(cmd_rig):
    """The retired "pin-gpu" behaviour is still REACHABLE — it just stopped
    being the default. An operator who genuinely wants the experts on the card
    says so with n_cpu_moe=0, and that wins at every autofit verdict."""
    for verdict in (-1, 17):
        cmd_rig.auto["value"] = verdict
        (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", n_cpu_moe=0,
                                                 path=cmd_rig.moe)
        assert ncm == 0 and _argv_pairs(argv)["--n-cpu-moe"] == "0", verdict
        assert ngl == verdict          # explicit 0 does NOT re-target ngl to -1


def test_dense_is_byte_identical_at_every_autofit_verdict(cmd_rig):
    """The default only ever touches DETECTED MoE GGUFs. A dense model must
    emit exactly today's argv — no flag, autofit ngl untouched — whether it
    fits whole, partially, or not at all."""
    for verdict in (-1, 0, 17, 47):
        cmd_rig.auto["value"] = verdict
        (argv, ngl, *_rest, ncm) = sa._build_cmd("dense-model",
                                                 path=cmd_rig.dense)
        assert ngl == verdict and ncm is None, verdict
        assert "--n-cpu-moe" not in argv


def test_k37_mode_engine_disables_the_auto_split(cmd_rig, monkeypatch):
    monkeypatch.setenv("HUGPY_ALLOC_MODE", "max-ram")
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
    assert ngl == 17 and ncm is None and "--n-cpu-moe" not in argv


# ═══ THE PER-MODE DECISION TABLE (k37 alloc modes x the default split) ══════
# The five modes are operator levers. mode_to_spill encodes them onto the wire
# the worker reads, so a mode's split behaviour is a CONSEQUENCE of the wire it
# produces — these tests pin the whole table end-to-end so an encoding change
# can't silently flip a mode's placement.
#
# DENSE BACKBONE FIRST (operator ruling 2026-07-31): a mode no longer decides
# WHETHER the dense backbone gets the GPU — only how much card the model may
# claim. Every mode that grants any GPU budget therefore splits; the only
# no-split modes are the ones that grant no GPU at all.
#
#   max-gpu   -> {}                     -> split   (blank default; the card)
#   gpu-only  -> {"n_gpu_layers": -1}   -> WHOLE   (k55, 2026-07-31: "only"
#                                                   means every tensor on the
#                                                   card — n_cpu_moe 0, or a
#                                                   refusal naming max-gpu;
#                                                   never a partial split)
#   ram-only  -> {"n_gpu_layers": "off"}-> NO      (zero GPU: nothing to rank)
#   max-ram   -> {"alloc_mode":"max-ram"}->NO      (RAM holds it all here: the
#                                                   overflow to GPU is zero)
#   explicit  -> {"alloc_mode":"explicit"}->split  (its VRAM budget is still a
#                                                   budget the backbone leads)
@pytest.mark.parametrize("mode,wants_split", [
    ("max-gpu", True), ("ram-only", False),
    ("max-ram", False), ("explicit", True)])
def test_alloc_mode_decision_table(cmd_rig, monkeypatch, mode, wants_split):
    wire = AM.mode_to_spill(mode)
    # _apply_spill's mapping, applied by hand (the agent sets these envs before
    # the load; slot_agent reads them through spill.*_env()).
    ngl_wire = wire.get("n_gpu_layers")
    monkeypatch.delenv("HUGPY_ALLOC_MODE", raising=False)
    if wire.get("alloc_mode"):
        monkeypatch.setenv("HUGPY_ALLOC_MODE", str(wire["alloc_mode"]))
    # The n_gpu_layers wire reaches _build_cmd as a real argument, not an env.
    kwargs = {}
    if ngl_wire is not None:
        kwargs["n_gpu_layers"] = 0 if str(ngl_wire) == "off" else int(ngl_wire)
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe,
                                             **kwargs)
    if wants_split:
        assert ngl == -1 and ncm == spill.MOE_ALL_LAYERS, mode
    else:
        assert ncm is None and "--n-cpu-moe" not in argv, mode


# ═══ gpu-only IS STRICT (operator ruling 2026-07-31, k55) ═══════════════════
# "gpu-only must be honest to its name" — ALL tensors on the card, experts
# included. The mode's only wire encoding is an EXPLICIT n_gpu_layers=-1
# (mode_to_spill), so that is what these drive; a DEFAULTED -1 is the blank
# max-gpu default and keeps the dense-first split (proved further down).
def test_gpu_only_that_fits_whole_states_n_cpu_moe_zero(cmd_rig):
    """The fits-whole leaf: everything on the card, and SAID so in argv. The 0
    is not cosmetic — an explicit --n-cpu-moe 0 beats any inherited
    LLAMA_ARG_N_CPU_MOE, so the launched placement is env-hack-proof."""
    assert AM.mode_to_spill("gpu-only") == {"n_gpu_layers": -1}
    cmd_rig.auto["value"] = 17
    cmd_rig.vram["free"] = 1 << 30               # dwarfs the 19332-byte model
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", n_gpu_layers=-1,
                                             path=cmd_rig.moe)
    assert ngl == -1 and ncm == 0
    assert _argv_pairs(argv)["--n-cpu-moe"] == "0"


def test_gpu_only_that_does_not_fit_is_REFUSED_naming_max_gpu(cmd_rig):
    """The doesn't-fit leaf: a refusal, never a quiet partial split. The same
    budget under the blank default DOES split — that contrast is the ruling."""
    cmd_rig.auto["value"] = 17
    cmd_rig.vram["free"] = 3332 + 8000           # backbone + ONE expert layer
    with pytest.raises(sa.GpuOnlyInfeasible) as exc:
        sa._build_cmd("moe-model", n_gpu_layers=-1, path=cmd_rig.moe)
    msg = str(exc.value)
    assert "max-gpu" in msg and "gpu-only" in msg
    # ... while max-gpu (the blank default) serves the dense-first split from
    # the very same budget (k64: momentary free VRAM never promotes experts).
    (_argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
    assert (ngl, ncm) == (-1, spill.MOE_ALL_LAYERS)


def test_gpu_only_named_as_a_mode_is_the_same_statement(cmd_rig, monkeypatch):
    """HUGPY_ALLOC_MODE=gpu-only and the -1 wire are one operator statement, so
    the mode name alone is enough — no n_gpu_layers argument needed."""
    monkeypatch.setenv("HUGPY_ALLOC_MODE", "gpu-only")
    cmd_rig.auto["value"] = 17
    cmd_rig.vram["free"] = 1 << 30
    (_argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
    assert (ngl, ncm) == (-1, 0)
    cmd_rig.vram["free"] = 3332 + 100            # backbone only
    with pytest.raises(sa.GpuOnlyInfeasible):
        sa._build_cmd("moe-model", path=cmd_rig.moe)


def test_gpu_only_on_an_unmeasurable_card_obeys_the_mode_literally(cmd_rig,
                                                                   monkeypatch):
    """Degrade-not-guess cuts BOTH ways: an unreadable card cannot prove the
    model doesn't fit, so no refusal is invented — but neither is the blank
    default's all-experts-to-CPU split, which would serve a split nobody asked
    for under the name "only". gpu-only is obeyed literally and llama.cpp busts
    honestly if the card turns out too small."""
    monkeypatch.setattr(spill, "total_vram_bytes", lambda: None)
    cmd_rig.vram["free"] = None                  # unmeasurable
    cmd_rig.auto["value"] = 17
    (_argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", n_gpu_layers=-1,
                                              path=cmd_rig.moe)
    assert (ngl, ncm) == (-1, 0)


def test_gpu_only_unmeasurable_still_faces_the_inverse_vram_preflight(cmd_rig,
                                                                      monkeypatch):
    """A gpu-only plan made against an unmeasurable BUDGET priced nothing, so it
    must not inherit the priced plan's exemption from the "-1 onto a card that
    can't hold it" preflight — free VRAM being unreadable does not mean the
    card's TOTAL is."""
    monkeypatch.setattr(spill, "total_vram_bytes", lambda: 1000)  # < the model
    cmd_rig.vram["free"] = None
    cmd_rig.auto["value"] = 17
    with pytest.raises(RuntimeError) as exc:
        sa._build_cmd("moe-model", n_gpu_layers=-1, path=cmd_rig.moe)
    assert "VRAM" in str(exc.value)


def test_gpu_only_dense_model_is_untouched(cmd_rig):
    """The ruling is about MoE pricing. A dense model under gpu-only emits
    exactly today's argv — no flag, no new refusal path (a dense model that
    can't fit is still the inverse-VRAM preflight's business, untouched)."""
    cmd_rig.vram["free"] = 1 << 30
    cmd_rig.auto["value"] = -1
    (argv, ngl, *_rest, ncm) = sa._build_cmd("dense-model", n_gpu_layers=-1,
                                             path=cmd_rig.dense)
    assert ngl == -1 and ncm is None and "--n-cpu-moe" not in argv


def test_explicit_n_cpu_moe_still_wins_under_gpu_only(cmd_rig):
    """An operator who states the split himself is driving the numbers — the
    strict path never sees the load, at any budget."""
    cmd_rig.vram["free"] = 3332 + 100            # would refuse on its own
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", n_gpu_layers=-1,
                                             n_cpu_moe=999, path=cmd_rig.moe)
    assert ngl == -1 and ncm == 999
    assert _argv_pairs(argv)["--n-cpu-moe"] == "999"


def test_max_gpu_is_the_blank_default_and_therefore_splits(cmd_rig):
    """max-gpu is what an unconfigured model resolves to (derive_alloc_mode's
    fallthrough) AND what mode_to_spill encodes as the empty wire — so "the
    default is the MoE split" and "max-gpu splits" are the same statement."""
    assert AM.derive_alloc_mode({}) == "max-gpu"
    assert AM.mode_to_spill("max-gpu") == {}
    cmd_rig.auto["value"] = 17
    (_argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
    assert (ngl, ncm) == (-1, spill.MOE_ALL_LAYERS)


def test_max_ram_that_holds_the_whole_model_grants_no_gpu_and_never_splits(
        cmd_rig, monkeypatch):
    """max-ram fills RAM first and only the OVERFLOW reaches the GPU. With no
    overflow there is no GPU budget at all, so there is nothing to rank and the
    placement is byte-identical to before — at every autofit verdict."""
    monkeypatch.setenv("HUGPY_ALLOC_MODE", "max-ram")
    for verdict in (-1, 0, 17):
        cmd_rig.auto["value"] = verdict
        (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
        assert ncm is None and "--n-cpu-moe" not in argv, verdict
        assert ngl == verdict


def test_explicit_mode_still_leads_with_the_dense_backbone(cmd_rig, monkeypatch):
    """explicit puts the operator in charge of the NUMBERS (its VRAM/RAM
    budgets), not of the ORDERING. Its budget is still a GPU budget, so the
    backbone leads and the experts take the remainder (ruling 2026-07-31)."""
    monkeypatch.setenv("HUGPY_ALLOC_MODE", "explicit")
    for verdict in (-1, 0, 17):
        cmd_rig.auto["value"] = verdict
        (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
        assert (ngl, ncm) == (-1, spill.MOE_ALL_LAYERS), verdict
        assert _argv_pairs(argv)["--n-cpu-moe"] == "999"


def test_old_llama_server_degrades_to_layer_split(cmd_rig, monkeypatch):
    monkeypatch.setattr(sa, "_server_supports_flag", lambda b, f: False)
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
    assert ngl == 17 and ncm is None             # reverted, no unknown flag
    assert "--n-cpu-moe" not in argv


def test_python_child_degrades_to_layer_split(cmd_rig, monkeypatch):
    serve = importlib.import_module("hugpy_engine.serve.serve")
    monkeypatch.setattr(serve, "LLAMA_SERVER_BIN", None)   # no native engine
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, kind, _total, ncm) = sa._build_cmd(
        "moe-model", path=cmd_rig.moe)
    assert kind == "python" and ngl == 17 and ncm is None
    assert "--n-cpu-moe" not in argv


# ═══ k64: llama-server's params-fit changed -ngl's vocabulary ═══════════════
# The fit-capable binary parses ``-1`` as AUTO and runs ``--fit`` (default ON),
# which re-plans placement against device memory — the ae repro: the k53 plan
# said "all layers + --n-cpu-moe", the child logged "common_init_result:
# fitting params to device memory", loaded the whole MoE at 19.6 GiB, and the
# headroom sweep then evicted it (fit fills to a ~1 GiB margin, under the
# worker's ~10% pressure ceiling). The choke point now translates -1 -> "all"
# and states --fit off, so the placement decided here is the placement served.
def test_fit_capable_binary_gets_all_spelling_and_fit_off(cmd_rig):
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
    pairs = _argv_pairs(argv)
    assert ngl == -1                              # internal encoding unchanged
    assert pairs["--n-gpu-layers"] == "all" and pairs["--fit"] == "off"
    assert ncm == spill.MOE_ALL_LAYERS


def test_fit_capable_binary_keeps_explicit_layer_counts_verbatim(cmd_rig):
    """Only the -1 spelling changed; a stated count is already fit-proof (fit
    adjusts UNSET args) but still gets --fit off for determinism."""
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", n_gpu_layers=20,
                                             path=cmd_rig.moe)
    pairs = _argv_pairs(argv)
    assert ngl == 20 and pairs["--n-gpu-layers"] == "20"
    assert pairs["--fit"] == "off"


def test_legacy_binary_keeps_the_minus_one_spelling(cmd_rig, monkeypatch):
    """A pre-fit llama-server still reads -1 as "all layers" and rejects
    --fit as an unknown flag — its argv stays byte-identical to before."""
    monkeypatch.setattr(sa, "_server_supports_flag",
                        lambda b, f: f == "--n-cpu-moe")
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
    pairs = _argv_pairs(argv)
    assert ngl == -1 and pairs["--n-gpu-layers"] == "-1"
    assert "--fit" not in argv
    assert ncm == spill.MOE_ALL_LAYERS and pairs["--n-cpu-moe"] == "999"


# ═══ k64: the ACTIVE alloc mode reaches the slot process ════════════════════
def test_slot_load_route_forwards_alloc_mode(monkeypatch):
    seen = {}

    class _FakeSlot:
        model_key = None

        def load(self, model_key, *a, **kw):
            seen.update(kw)
            return {"ok": True}

    monkeypatch.setattr(sa, "Slot", _FakeSlot)
    monkeypatch.delenv("HUGPY_NO_LOCAL_SERVING", raising=False)
    app, _slot = sa.build_app()
    client = app.test_client()
    client.post("/load", json={"model_key": "m", "alloc_mode": "max-ram"})
    assert seen["alloc_mode"] == "max-ram"


def test_slot_load_projects_alloc_mode_into_env_and_clears_it(monkeypatch):
    """Slot.load writes the per-load alloc_mode into this process's env (what
    _moe_gpu_budget/_strict_gpu_only read) BEFORE _build_cmd, and clears it
    when absent — a mode must never leak onto the next model (the same rule as
    the agent's _SPILL_ENV_CLEAR_WHEN_ABSENT)."""
    s = sa.Slot.__new__(sa.Slot)
    s.model_key, s.profile_bin = None, None
    s._load_failures, s._load_backoff_until = {}, {}
    s.lock = threading.Lock()
    s._kill = lambda: None
    observed = {}

    def _capture_and_abort(*a, **kw):
        observed["mode"] = os.environ.get("HUGPY_ALLOC_MODE")
        raise RuntimeError("stop before spawn")

    monkeypatch.setattr(sa, "_build_cmd", _capture_and_abort)
    with pytest.raises(RuntimeError):
        s.load("m", alloc_mode="max-ram")
    assert observed["mode"] == "max-ram"
    with pytest.raises(RuntimeError):
        s.load("m")                              # absent -> cleared, no leak
    assert observed["mode"] is None


# ═══ DEFAULTED vs EXPLICIT -1 (the ae 2026-07-25 dead-GPU-slot regression) ═══
# -1 is overloaded: it is BOTH the fill-in default (serve.DEFAULT_LLAMA_NGL,
# materialized into ServeSpec.n_gpu_layers because the field is typed int) AND a
# load-bearing explicit force ("Max GPU" — runners.get, alloc_modes gpu-only,
# chaos/sweep, chaos/assortment, the worker MoE commit, overrides). The MoE auto
# policy gated on `n_gpu_layers is None`, so a DEFAULTED -1 read as "the operator
# demanded all 48 layers on the card" and coder-next (48.4 GB MoE) launched onto
# a 24 GB 3090 with no --n-cpu-moe, stalled, hard-capped, and backed off 600s.
#
# The fix is a provenance marker (slot_agent._NglDefaulted, an int subclass), so
# these two cases MUST diverge in policy while staying identical in value.
def test_defaulted_minus_one_still_gets_the_moe_auto_split(cmd_rig):
    """(a) UNSET (a fill-in -1 nobody asked for) -> the auto MoE split fires."""
    cmd_rig.auto["value"] = 17                   # hybrid: partial layer split
    defaulted = sa._NglDefaulted(-1)
    (argv, ngl, _c, _t, _cp, _kind, total, ncm) = sa._build_cmd(
        "moe-model", n_gpu_layers=defaulted, path=cmd_rig.moe)
    pairs = _argv_pairs(argv)
    # Exactly the same outcome as passing nothing at all.
    assert ngl == -1 and pairs["--n-gpu-layers"] == "all"
    assert ncm == spill.MOE_ALL_LAYERS and pairs["--n-cpu-moe"] == "999"
    assert total == 48


def test_explicit_minus_one_is_gpu_only_not_a_layer_instruction(cmd_rig):
    """(b) EXPLICIT -1 (runners.get / the gpu-only wire / chaos / the stored
    stamp) — re-ruled TWICE, and the second ruling stands.

    2026-07-31 (k53): it is not a per-layer instruction, so it must never
    disable the split into the 17/48 hybrid that stranded 31 layers of DENSE
    attention on the CPU. That still holds — no hybrid is reachable here.

    2026-07-31 (k55): but it is not a vague "as much as you can" either. It is
    mode_to_spill("gpu-only")'s wire, so it means ALL tensors on the card, and
    the answer is binary. On an UNMEASURABLE card nothing can be disproved, so
    the mode is obeyed literally (--n-cpu-moe 0) rather than degraded into the
    blank default's all-experts-to-CPU split, which nobody asked for."""
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", n_gpu_layers=-1,
                                             path=cmd_rig.moe)
    assert ngl == -1 and ncm == 0
    assert _argv_pairs(argv)["--n-cpu-moe"] == "0"


def test_explicit_positive_layer_count_is_still_obeyed_verbatim(cmd_rig):
    """The k14/k7 lever survives the ruling: a STATED layer count (seat at full
    offload, then relaunch DOWN through counts to measure the speed cliff) is a
    real per-layer instruction, so it is obeyed and no split is planned. Only
    the ambiguous -1 stamp was re-read."""
    cmd_rig.vram["free"] = 24 * GIB               # even with a measurable card
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", n_gpu_layers=20,
                                             path=cmd_rig.moe)
    assert ngl == 20 and ncm is None and "--n-cpu-moe" not in argv


def test_the_blank_default_never_promotes_experts_from_momentary_free_vram(
        cmd_rig, monkeypatch):
    """k64 (operator ruling 2026-07-31, supersedes k53's remainder-fill for
    free-VRAM-priced budgets): under the blank default / max-gpu the budget is
    the card's MOMENTARY free VRAM — not a stated contract — and it buys the
    dense backbone + KV only. The experts stay on CPU (MOE_ALL_LAYERS) no
    matter how empty the card happens to be at load instant.

    The retired behavior was the ae incident: an empty 3090 remainder-filled
    coder-next to n_cpu_moe=0, the whole 19.6 GiB landed on the card, rode the
    90% ceiling, and the idle-pressure headroom sweep evicted it minutes later
    — every call repaid a full cold load. Expert promotion needs a DEMAND: an
    explicit n_cpu_moe, gpu-only, or an explicit gpu_mem_gib contract (which
    keeps the k53 remainder fill — the sibling test below)."""
    for free in (3332 + 8000 + 10,               # backbone + one expert layer
                 8 * GIB):                       # card dwarfs the whole model
        cmd_rig.vram["free"] = free
        cmd_rig.auto["value"] = 17
        (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
        assert (ngl, ncm) == (-1, spill.MOE_ALL_LAYERS), free
        assert _argv_pairs(argv)["--n-cpu-moe"] == "999"
    # The same budget under gpu-only is a refusal, not a quieter split.
    cmd_rig.vram["free"] = 3332 + 8000 + 10
    with pytest.raises(sa.GpuOnlyInfeasible):
        sa._build_cmd("moe-model", n_gpu_layers=-1, path=cmd_rig.moe)


def test_a_card_that_holds_everything_states_n_cpu_moe_zero(cmd_rig):
    """A budget that covers the whole model puts the experts on the card too —
    and SAYS so with an explicit --n-cpu-moe 0 rather than omitting the flag, so
    an inherited LLAMA_ARG_N_CPU_MOE=999 (the transition-era ae unit hack)
    cannot silently spill them."""
    cmd_rig.vram["free"] = 8 * GIB
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", n_gpu_layers=-1,
                                             path=cmd_rig.moe)
    assert ngl == -1 and ncm == 0
    assert _argv_pairs(argv)["--n-cpu-moe"] == "0"


def test_a_card_that_only_holds_the_backbone_spills_every_expert(cmd_rig):
    """Budget covers the dense backbone and nothing more -> MOE_ALL_LAYERS: the
    measured ae default (+59% tok/s at 5x less VRAM), reached by pricing rather
    than by assumption."""
    cmd_rig.vram["free"] = 3332 + 100
    (_argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
    assert (ngl, ncm) == (-1, spill.MOE_ALL_LAYERS)


def test_the_context_reserve_is_charged_before_the_experts(cmd_rig, monkeypatch):
    """The KV cache lands on the card beside the weights, so the plan spends the
    budget on it BEFORE it hands anything to the experts — otherwise a split
    that "fits" OOMs the moment the context allocates."""
    monkeypatch.setattr(spill, "vram_ctx_reserve_bytes",
                        lambda p, n_ctx=None: (8000, "test", {}))
    cmd_rig.vram["free"] = 3332 + 8000 + 10       # would have bought one layer
    (_argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
    assert (ngl, ncm) == (-1, spill.MOE_ALL_LAYERS)


def test_gpu_mem_gib_budget_caps_what_the_experts_may_claim(cmd_rig):
    """A per-model VRAM contract (explicit mode's gpu_mem_gib) is the budget the
    plan spends — not the whole card. 16 GiB free but a 3332+8000-byte contract
    still means one expert layer."""
    cmd_rig.vram["free"] = 16 * GIB
    (_argv, ngl, *_rest, ncm) = sa._build_cmd(
        "moe-model", path=cmd_rig.moe,
        gpu_mem_gib=(3332 + 8000 + 10) / GIB)
    assert (ngl, ncm) == (-1, 1)


def test_explicit_n_cpu_moe_zero_on_an_impossible_fit_REFUSES_loudly(cmd_rig,
                                                                     monkeypatch):
    """n_cpu_moe=0 ("experts on the card") on a model that cannot fit the card
    is refused with an actionable error — it is NOT silently attempted, and it
    is NOT quietly overridden into a split.

    The distinction from the sibling test matters. A bare -1 is an AMBIGUOUS
    stamp applied in bulk, so when it is impossible we reinterpret it. An
    explicit n_cpu_moe=0 is an UNAMBIGUOUS demand, so when it is impossible we
    tell the operator instead of second-guessing them. Both avoid the stall;
    only one of them overrides a human.
    """
    monkeypatch.setattr(sa, "_total_gguf_bytes", lambda _p: 48 * 2 ** 30)
    monkeypatch.setattr("hugpy_engine.spill.free_vram_bytes",
                        lambda: 24 * 2 ** 30)
    with pytest.raises(RuntimeError, match="n_cpu_moe|VRAM"):
        sa._build_cmd("moe-model", n_gpu_layers=-1, n_cpu_moe=0,
                      path=cmd_rig.moe)


def test_defaulted_marker_is_value_identical_to_the_plain_int():
    """The marker may only carry PROVENANCE — never change the value. Every
    existing consumer (argv formatting, comparisons, json) must be unaffected."""
    import json
    d = sa._NglDefaulted(-1)
    assert d == -1 and int(d) == -1 and str(d) == "-1"
    assert json.dumps({"n": d}) == '{"n": -1}'
    assert isinstance(d, int)
    # ...and the unset predicate separates it from a plain int of equal value.
    assert sa._ngl_is_unset(d) is True
    assert sa._ngl_is_unset(None) is True
    assert sa._ngl_is_unset(-1) is False
    assert sa._ngl_is_unset(0) is False
    assert sa._ngl_is_unset(20) is False


def test_defaulted_zero_autofits_rather_than_pinning_cpu(cmd_rig):
    """A defaulted value is unset WHATEVER the int is: a box whose
    DEFAULT_LLAMA_NGL is 0 must still autofit, not silently serve on CPU."""
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest) = sa._build_cmd("dense-model",
                                        n_gpu_layers=sa._NglDefaulted(0),
                                        path=cmd_rig.dense)
    assert ngl == 17                              # autofit won, not the 0


def test_explicit_zero_still_pins_cpu(cmd_rig):
    """The counterpart: an explicit 0 ('CPU only') is still honored."""
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest) = sa._build_cmd("dense-model", n_gpu_layers=0,
                                        path=cmd_rig.dense)
    assert ngl == 0


def test_serve_spec_marks_the_fill_in_default_as_not_explicit(monkeypatch):
    """serve_spec_for must record WHETHER anyone chose a layer placement.
    ``slot_n_gpu_layers`` then carries that provenance to the slot while
    ``n_gpu_layers`` (unit/swap/supervisor argv) stays an ordinary int."""
    serve = importlib.import_module("hugpy_engine.serve.serve")

    # Nothing persisted -> the spec's ngl is only DEFAULT_LLAMA_NGL.
    blank = serve.ServeSpec(model_key="m", mode=serve.ServeMode.OFF)
    assert blank.ngl_explicit is False
    assert blank.n_gpu_layers == serve.DEFAULT_LLAMA_NGL
    assert sa._ngl_is_unset(blank.slot_n_gpu_layers) is True
    assert int(blank.slot_n_gpu_layers) == serve.DEFAULT_LLAMA_NGL
    # The plain field is untouched — argv builders keep seeing an int.
    assert not isinstance(blank.n_gpu_layers, sa._NglDefaulted)

    # Someone chose -1 ("Max GPU") -> explicit, policy must not override it.
    chosen = serve.ServeSpec(model_key="m", mode=serve.ServeMode.OFF,
                             n_gpu_layers=-1, ngl_explicit=True)
    assert sa._ngl_is_unset(chosen.slot_n_gpu_layers) is False
    assert chosen.slot_n_gpu_layers == -1


def test_slot_load_route_honors_the_ngl_defaulted_flag(monkeypatch):
    """The /load wire: ``ngl_defaulted: true`` marks the accompanying
    n_gpu_layers as a fill-in. Omitting it (every pre-existing caller) keeps the
    value explicit — byte-identical to before."""
    seen = {}

    class _FakeSlot:
        model_key = None

        def load(self, model_key, n_gpu_layers=None, *a, **kw):
            seen["ngl"] = n_gpu_layers
            return {"ok": True}

    # build_app() constructs its slot in a closure — swap the class it builds.
    monkeypatch.setattr(sa, "Slot", _FakeSlot)
    monkeypatch.delenv("HUGPY_NO_LOCAL_SERVING", raising=False)
    app, _slot = sa.build_app()
    client = app.test_client()

    client.post("/load", json={"model_key": "m", "n_gpu_layers": -1,
                               "ngl_defaulted": True})
    assert sa._ngl_is_unset(seen["ngl"]) is True and seen["ngl"] == -1

    client.post("/load", json={"model_key": "m", "n_gpu_layers": -1})
    assert sa._ngl_is_unset(seen["ngl"]) is False and seen["ngl"] == -1


# ═══════════ relaunch threads n_cpu_moe (k14 lever) ═════════════════════════
def test_slot_relaunch_accepts_and_forwards_n_cpu_moe():
    s = sa.Slot.__new__(sa.Slot)
    s.model_key = "coder"
    s.ngl, s.ctx, s.threads, s.cpus, s.gpu = -1, 4096, 6, None, None
    s.profile_bin = None
    s._load_failures, s._load_backoff_until = {}, {}
    s.lock = threading.Lock()
    seen = {}

    def _fake_load(model_key, **kw):
        seen["model_key"] = model_key
        seen.update(kw)
        return {"model_key": model_key, "n_gpu_layers": -1, "n_cpu_moe": 999}
    s.load = _fake_load
    out = s.relaunch(n_gpu_layers=-1, n_cpu_moe=999)
    assert seen["n_cpu_moe"] == 999 and seen["force"] is True
    assert out["relaunched"] is True and out["requested_n_cpu_moe"] == 999


# ═══════════ slot admission threads the MoE verdict ═════════════════════════
def test_endpoint_for_threads_moe_verdict_into_load_opts(monkeypatch):
    posts = []

    def _fake_post(url, body, timeout):
        posts.append((url, dict(body)))
        return {"endpoint": "http://fake:1"}
    monkeypatch.setattr(SL, "_post", _fake_post)
    monkeypatch.setattr(
        SL.SlotPool, "statuses",
        lambda self: [{"_control": "http://fake:1", "healthy": True,
                       "model_key": None}])
    monkeypatch.setattr(SL, "_FIT_CHECK", lambda mk: False)     # over ceiling
    monkeypatch.setattr(SL, "_EVICTION_POLICY", None)
    monkeypatch.setattr(SL, "_MAKE_ROOM", lambda mk: {
        "action": "partial", "n_gpu_layers": -1, "n_cpu_moe": 999,
        "evicted": []})
    pool = SL.SlotPool(urls=["http://fake:1"])
    ep = pool.endpoint_for("moe-model")
    assert ep == "http://fake:1"
    _url, body = posts[-1]
    assert body["n_gpu_layers"] == -1 and body["n_cpu_moe"] == 999


# ═══════════ expert-aware need — the fit checks ═════════════════════════════
_EMPTY_CARD_DET = {
    "total": int(41.6 * GIB * 1.15),             # the opaque full-file need
    "moe_split": {"n_cpu_moe": 999, "gpu_total": int(2.9 * GIB),
                  "cpu_bytes": int(43.5 * GIB), "path": "/x.gguf"},
}


def test_slot_fit_check_passes_empty_card_moe_case(monkeypatch):
    """The live refusal this fixes: /probe (and the boot star) on an EMPTY
    23.6 GiB card said fit:false for the 41.6GB MoE. Under the split the GPU
    need is the non-expert share -> passes; experts checked against RAM."""
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: int(23.6 * GIB))
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: int(23.6 * GIB))
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: int(60 * GIB))
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: dict(_EMPTY_CARD_DET))
    assert A._worker_slot_fit_check("coder-next") is True


def test_slot_fit_check_without_split_still_refuses(monkeypatch):
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: int(23.6 * GIB))
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: int(23.6 * GIB))
    det = {"total": _EMPTY_CARD_DET["total"]}    # no moe_split (dense/explicit)
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: det)
    assert A._worker_slot_fit_check("coder-next") is False


def test_slot_fit_check_moe_respects_ram_guard(monkeypatch):
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: int(23.6 * GIB))
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: int(23.6 * GIB))
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: dict(_EMPTY_CARD_DET))
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: int(10 * GIB))
    assert A._worker_slot_fit_check("coder-next") is False       # experts > RAM
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: None)
    assert A._worker_slot_fit_check("coder-next") is True        # unmeasurable: open


def test_slot_fit_check_prices_the_split_even_when_the_whole_thing_fits(monkeypatch):
    """"need-calc prices experts as GPU" (keeper note), fixed for the fit checks.

    A MoE that fits whole is now SERVED as a split, so the fit checks must ask
    the split's question first. Here the full 41.6G need breaches a 46G card's
    ~90% ceiling with only 44G free (44 - 41.6 = 2.4G < the 4.6G reserve); the
    typed 2.9G share clears it comfortably."""
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: int(46 * GIB))
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: int(44 * GIB))
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: int(60 * GIB))
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: dict(_EMPTY_CARD_DET))
    assert A._worker_slot_fit_check("coder-next") is True


def test_slot_fit_check_falls_back_to_full_need_when_experts_dont_fit_ram(monkeypatch):
    """Experts that can't fit RAM mean the split degrades to the autofit layer
    placement — so the check must fall THROUGH to the full need rather than
    return False outright. A roomy card still passes."""
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: int(64 * GIB))
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: int(64 * GIB))
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: int(10 * GIB))   # < 43.5G
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: dict(_EMPTY_CARD_DET))
    assert A._worker_slot_fit_check("coder-next") is True   # full need fits 64G


def test_contention_fit_check_prices_the_split_first(monkeypatch):
    """Same for the contention check: the typed share passes the ceiling gate
    on a card the opaque full need would fail."""
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: int(8 * GIB))
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: int(60 * GIB))
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: dict(_EMPTY_CARD_DET))
    assert A._worker_fit_check("coder-next") is True


def test_contention_fit_check_uses_split_need(monkeypatch):
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: int(23.6 * GIB))
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: int(60 * GIB))
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: dict(_EMPTY_CARD_DET))
    assert A._worker_fit_check("coder-next") is True
    det = {"total": _EMPTY_CARD_DET["total"]}
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: det)
    assert A._worker_fit_check("coder-next") is False


# ═══════════ _vram_evict_to_fit — the admission choke point ═════════════════
class _State:
    pass


@pytest.fixture
def evict_rig(monkeypatch):
    card = {"total": 24 * GIB, "free": 23 * GIB, "need": int(47.8 * GIB)}
    residents = {}
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: card["total"])
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: card["free"])
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: card.get("ram", 60 * GIB))
    # The mmap-eligibility ceiling (2026-08-28): the expert-RAM guard prices
    # against the BOX (total RAM), falling back to free when unmeasurable.
    # None by default so rig tests that only set "ram" exercise the fallback.
    monkeypatch.setattr(A, "_ram_total_bytes", lambda: card.get("ram_total"))
    monkeypatch.setattr(A, "_incoming_need_bytes", lambda mk: card["need"])
    monkeypatch.setattr(A, "_kv_need_bytes", lambda mk, cfg=None: (0, {}))
    monkeypatch.setattr(A, "_calib_correction", lambda mk: None)
    monkeypatch.setattr(A, "_vram_residents",
                        lambda s: [{"model_key": k, "vram_bytes": v,
                                    "host_mode": "subprocess", "alive": True}
                                   for k, v in residents.items()])
    monkeypatch.setattr(A, "_residency", lambda mk: "on-demand")
    monkeypatch.setattr(A, "_busy_slot_models", lambda: set())
    monkeypatch.setattr(A, "_queued_ahead_of", lambda mk: set())
    monkeypatch.setattr(A, "_trim_host_ram", lambda: None)
    monkeypatch.setattr(D, "last_used_snapshot", lambda: {})
    plan = {"value": {"path": "/x.gguf", "n_cpu_moe": 999,
                      "gpu_weight_bytes": int(2.5 * GIB),
                      "cpu_bytes": int(43.5 * GIB),
                      "detail": {"expert_count": 512, "expert_used_count": 10,
                                 "sparsity": 10 / 512}}}
    monkeypatch.setattr(A, "_moe_plan_for", lambda mk: plan["value"])
    A._MOE_SPLIT.clear()
    A._PARTIAL_NGL.clear()
    yield type("Rig", (), {"card": card, "residents": residents, "plan": plan})()
    A._MOE_SPLIT.clear()
    A._PARTIAL_NGL.clear()


def test_admission_retargets_impossible_full_fit_to_moe_split(evict_rig):
    """47.8G full need can NEVER fit a 24G card — the admission prices the
    split (2.875G GPU) instead: admits with the MoE verdict, evicts NOBODY."""
    evict_rig.residents["innocent"] = 10 * GIB
    verdict = A._vram_evict_to_fit(_State(), "coder-next")
    assert verdict["action"] == "partial"
    assert verdict["n_gpu_layers"] == -1 and verdict["n_cpu_moe"] == 999
    assert verdict["evicted"] == []              # no innocents evicted
    assert A._MOE_SPLIT["coder-next"]["n_cpu_moe"] == 999
    assert verdict["moe"]["cpu_bytes"] == int(43.5 * GIB)


def test_admission_moe_split_still_evicts_when_gpu_share_needs_room(evict_rig):
    # Split GPU need 2.875G; only 1G free; a 10G idle resident yields.
    evict_rig.card["free"] = 1 * GIB
    calls = []

    def _fake_evict(state, mk, force=False):
        calls.append(mk)
        freed = evict_rig.residents.pop(mk)
        evict_rig.card["free"] += freed
        return {"model_key": mk, "evicted": True, "vram_freed": freed,
                "host_mode": "subprocess"}
    A._evict_model, orig = _fake_evict, A._evict_model
    try:
        evict_rig.residents["idle"] = 10 * GIB
        verdict = A._vram_evict_to_fit(_State(), "coder-next")
    finally:
        A._evict_model = orig
    assert verdict["action"] == "partial" and verdict["n_cpu_moe"] == 999
    assert verdict["evicted"] == ["idle"]


def test_admission_moe_ram_guard_keeps_full_need(evict_rig):
    # Experts (43.5G) exceed the BOX (total RAM, the mmap-eligibility ceiling
    # since 2026-08-28) -> no re-target; the full-need path stands and refuses
    # honestly (no admit-then-thrash).
    evict_rig.card["ram"] = 20 * GIB
    evict_rig.card["ram_total"] = 20 * GIB
    verdict = A._vram_evict_to_fit(_State(), "coder-next")
    assert verdict["action"] == "refuse"
    assert "coder-next" not in A._MOE_SPLIT
    assert verdict["reason"]["moe_split"]["was_plan"] is False


def test_admission_moe_experts_fit_total_ram_despite_low_free(evict_rig):
    """The ae 2026-08-28 shape: experts (43.5G) fit the BOX (125G installed)
    while momentary free RAM (34G — mostly reclaimable page cache) is below
    them. Expert bytes are file-backed mmap page cache, so the split re-target
    must proceed instead of falling back to a doomed 47.8G full-need refusal."""
    evict_rig.card["ram"] = 34 * GIB
    evict_rig.card["ram_total"] = 125 * GIB
    verdict = A._vram_evict_to_fit(_State(), "coder-next")
    assert verdict["action"] == "partial"
    assert verdict["n_gpu_layers"] == -1 and verdict["n_cpu_moe"] == 999


def test_admission_fits_whole_ALSO_SPLITS_and_evicts_nobody(evict_rig):
    """POLICY 2026-07-25 (the admission half of the same change).

    A MoE whose FULL weights (18G) would fit only AFTER an eviction used to be
    admitted fully-on-GPU, evicting a 21G innocent to make room. The split is
    now the default placement, so admission prices the 2.875G GPU share the
    child will ACTUALLY take — which fits the free 6G (over the 2.4G ceiling
    reserve) outright, so nobody is evicted. Admission MUST agree with
    slot_agent._build_cmd: pricing the full 18G would reserve ~6x the VRAM the
    child takes and evict a neighbour for bytes that never land on the card."""
    evict_rig.card["need"] = 18 * GIB
    evict_rig.card["free"] = 6 * GIB              # 6 - 2.875 > 2.4G reserve
    A._evict_model, orig = (lambda *a, **k: pytest.fail("evicted an innocent"),
                            A._evict_model)
    try:
        evict_rig.residents["idle"] = 21 * GIB
        verdict = A._vram_evict_to_fit(_State(), "coder-next")
    finally:
        A._evict_model = orig
    assert verdict["action"] == "partial"
    assert verdict["n_gpu_layers"] == -1 and verdict["n_cpu_moe"] == 999
    assert verdict["evicted"] == []               # the innocent 21G resident stays
    assert A._MOE_SPLIT["coder-next"]["n_cpu_moe"] == 999


def test_admission_fits_whole_but_experts_exceed_ram_keeps_full_need(evict_rig):
    """The viability guard holds on the admission side too: experts that can't
    fit the BOX (total RAM — the mmap-eligibility ceiling, 2026-08-28) mean no
    re-target, and the full-need path stands unchanged."""
    evict_rig.card["need"] = 18 * GIB
    evict_rig.card["free"] = 22 * GIB             # 22 - 18 > 2.4G reserve
    evict_rig.card["ram"] = 20 * GIB              # experts are 43.5G
    evict_rig.card["ram_total"] = 20 * GIB
    verdict = A._vram_evict_to_fit(_State(), "coder-next")
    assert verdict["action"] == "proceed"
    assert verdict.get("n_cpu_moe") is None
    assert "coder-next" not in A._MOE_SPLIT


def test_admission_moe_first_preempts_dense_degenerate(evict_rig, monkeypatch):
    """ae 2026-08-28 transcript shape: a cold load whose spill shape skipped
    _moe_plan_for reached the DENSE partial-offload planner, which priced a
    3-layer dense floor (~3.2 GB layer-bytes) against the MoE-derived 1.5 GiB
    budget and refused "degenerate" — mixed bases in one verdict. A detected
    MoE must defer to the dense-backbone-first split BEFORE dense layer math:
    48 layers, 45.1 GiB dense, 1.49 GiB non-expert, 21.5 GB free VRAM ->
    admit via the MoE plan (-1 + --n-cpu-moe), never 'degenerate'."""
    from hugpy_engine.spill import MOE_ALL_LAYERS
    evict_rig.plan["value"] = None            # the spill shape that skipped the plan
    evict_rig.card["need"] = int(45.1 * GIB)
    evict_rig.card["free"] = int(21.5e9)      # 21.5 GB free of a 24G card
    evict_rig.card["ram"] = 34 * GIB
    evict_rig.card["ram_total"] = 125 * GIB   # experts fit the BOX (mmap)
    per_layer = int(43.59 * GIB) // 48
    detail = {"is_moe": True, "expert_bytes": per_layer * 48,
              "non_expert_bytes": int(1.49 * GIB),
              "expert_bytes_by_layer": {i: per_layer for i in range(48)}}
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: ("/x.gguf", 48))
    monkeypatch.setattr(A, "_moe_detail_for", lambda mk: detail)
    monkeypatch.setattr(A, "_moe_auto_gpu_budget",
                        lambda mk, p: int(1.5 * GIB))  # the non-expert cap
    verdict = A._vram_evict_to_fit(_State(), "coder-next")
    assert verdict["action"] == "partial", verdict
    assert verdict["n_gpu_layers"] == -1
    assert verdict["n_cpu_moe"] == MOE_ALL_LAYERS
    assert not verdict.get("reason")
    assert "degenerate" not in (verdict.get("note") or "")
    assert A._MOE_SPLIT["coder-next"]["n_cpu_moe"] == MOE_ALL_LAYERS


def test_clear_partial_ngl_also_clears_moe_commit():
    A._MOE_SPLIT["m"] = {"path": "/x.gguf", "n_cpu_moe": 999}
    A._clear_partial_ngl("m")
    assert "m" not in A._MOE_SPLIT


# ═══════════ calibration honesty ════════════════════════════════════════════
def test_moe_split_residency_never_reads_as_a_full_load(monkeypatch):
    monkeypatch.setattr(A, "_incoming_need_detail",
                        lambda mk: {"base_total": 48 * GIB, "weights": 47 * GIB,
                                    "kv": 0, "ctx_pct": None})
    monkeypatch.setattr(A, "_model_framework", lambda mk: "gguf")
    A._MOE_SPLIT["m"] = {"path": "/x.gguf", "n_cpu_moe": 999}
    try:
        s = A._build_calibration_success(
            "m", {"device": "cuda", "n_gpu_layers": -1,
                  "vram_bytes": 3 * GIB, "rss_bytes": GIB})
    finally:
        A._MOE_SPLIT.clear()
    assert s["verdict"] == "partial"             # excluded from the full ratio


# ═══════════ central feasibility with MoE sizing ════════════════════════════
def test_feasibility_moe_split_makes_MAX_GPU_selectable_not_gpu_only():
    """STRICT gpu-only (operator ruling 2026-07-31): "only" means ALL tensors on
    the card, experts included. The split used to relax THIS gate — a 45 GiB MoE
    on a 24 GiB card was offered gpu-only and then served with 43 GiB of experts
    in RAM, which silently redefined the mode's name. The split's feasibility
    home is max-gpu, whose contract already says spill."""
    dense = AM.feasible_modes("gguf", int(45 * GIB), 24 * GIB, 128 * GIB)
    assert "gpu-only" not in dense               # 45G on a 24G card: eliminated
    moe = AM.feasible_modes("gguf", int(45 * GIB), 24 * GIB, 128 * GIB,
                            moe_split_gpu_bytes=int(3 * GIB))
    assert "gpu-only" not in moe                 # the split does NOT relax it
    assert "max-gpu" in moe                      # spill-and-serve lives here
    assert moe == dense                          # the split changes no gate now


def test_feasibility_gpu_only_survives_when_the_WHOLE_moe_fits():
    """The mode is not banned for MoEs — it is priced honestly. A MoE that fits
    the card whole (experts included) keeps gpu-only, at the same headroom
    factor a dense model of the same size gets."""
    small = int(10 * GIB)
    moe = AM.feasible_modes("gguf", small, 24 * GIB, 128 * GIB,
                            moe_split_gpu_bytes=int(1 * GIB))
    assert "gpu-only" in moe
    assert moe == AM.feasible_modes("gguf", small, 24 * GIB, 128 * GIB)


def test_feasibility_dense_paths_are_unchanged():
    base = AM.feasible_modes("gguf", 5 * GIB, 24 * GIB, 128 * GIB)
    assert base == AM.feasible_modes("gguf", 5 * GIB, 24 * GIB, 128 * GIB,
                                     moe_split_gpu_bytes=None)
    tf = AM.feasible_modes("transformers", int(68 * GIB), 24 * GIB, 124 * GIB)
    # gpu-only stays eliminated (68G can never ride the card whole), but
    # max-gpu SURVIVES: it is a spill preference under the numbers rule
    # (fits GPU+RAM combined — 118ac64, operator ruling 2026-07-29), not a
    # fits-the-card-whole test.
    assert "gpu-only" not in tf and "max-gpu" in tf and "ram-only" in tf


# ═══════════ the knob: overrides + spill env wire ═══════════════════════════
def test_override_field_is_first_class_int():
    assert "n_cpu_moe" in OV.ALLOWED_FIELDS
    assert OV._coerce("n_cpu_moe", "999") == 999
    assert OV._coerce("n_cpu_moe", 12) == 12
    assert OV._coerce("n_cpu_moe", "") is None   # clears


def test_apply_spill_maps_and_clears_n_cpu_moe(monkeypatch):
    monkeypatch.delenv("HUGPY_N_CPU_MOE", raising=False)
    A._apply_spill({"n_cpu_moe": 999})
    assert os.environ.get("HUGPY_N_CPU_MOE") == "999"
    assert spill.n_cpu_moe_env() == 999
    A._apply_spill({})                           # absent -> cleared (no leak)
    assert "HUGPY_N_CPU_MOE" not in os.environ
    assert spill.n_cpu_moe_env() is None


def test_n_cpu_moe_rides_the_version_gate():
    assert "n_cpu_moe" in AM.NEW_SPILL_KEYS
    gated, note = AM.gate_spill_for_worker({"n_cpu_moe": 999}, "0.1.150", "old")
    assert gated == {} and note                  # never a silent dead knob
    gated, note = AM.gate_spill_for_worker({"n_cpu_moe": 999}, "0.1.203", "new")
    assert gated == {"n_cpu_moe": 999} and note is None


# ═══════════ the governing-plan resolver ════════════════════════════════════
@pytest.fixture
def plan_rig(monkeypatch, moe_gguf):
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (moe_gguf, 48))
    for env in ("HUGPY_N_GPU_LAYERS", "HUGPY_N_CPU_MOE", "HUGPY_ALLOC_MODE",
                "HUGPY_GPU_MEM_GIB"):
        monkeypatch.delenv(env, raising=False)
    # The card, EXPLICITLY (k55). Admission now prices the SAME dense-first plan
    # the slot launches, so it reads real VRAM — leaving that to whatever GPU
    # the test box happens to have would make these verdicts machine-dependent.
    # Default None = unmeasurable, which is the slot's own degrade (all experts
    # to CPU), so every pre-k55 expectation below is unchanged.
    vram = {"free": None}
    monkeypatch.setattr(spill, "free_vram_bytes", lambda: vram["free"])
    monkeypatch.setattr(spill, "vram_ctx_reserve_bytes",
                        lambda p, n_ctx=None: (0, "test", {}))
    return type("Rig", (), {"path": moe_gguf, "vram": vram})()


def test_plan_unmeasurable_card_defaults_to_all_experts(plan_rig):
    """The historical degrade, and still the honest one: with no VRAM reading
    the slot splits every expert to CPU, so admission prices exactly that."""
    plan = A._moe_plan_for("m")
    assert plan["n_cpu_moe"] == spill.MOE_ALL_LAYERS
    assert plan["cpu_bytes"] == 16000 and plan["gpu_weight_bytes"] == 3332


# ═══ ADMISSION PRICES THE DENSE-FIRST PLAN (k55 — k53's flagged follow-up) ═══
# The child spends the budget left after the backbone on EXPERT LAYERS, so
# pricing the split as backbone-only under-reserved every load with room to
# spare: the child took more VRAM than admission planned and the NEXT admission
# saw a fuller card than it expected. Admission and launch now run the same
# arithmetic on the same budget.
def test_plan_prices_all_experts_cpu_for_a_momentary_free_budget(plan_rig):
    """k64 (2026-07-31): a budget priced from momentary free VRAM never
    promotes experts — admission prices the backbone-only footprint the slot
    now actually launches, however roomy the card happens to be."""
    for free in (3332 + 8000, 1 << 20):          # roomy or tight: same placement
        plan_rig.vram["free"] = free
        plan = A._moe_plan_for("m")
        assert plan["n_cpu_moe"] == spill.MOE_ALL_LAYERS, free
        assert plan["gpu_weight_bytes"] == 3332, free
        assert plan["cpu_bytes"] == 16000, free


def test_plan_matches_the_slot_launch_byte_for_byte(plan_rig, monkeypatch):
    """The whole point: what admission reserves and what _build_cmd launches
    are one number. Same budget in, same n_cpu_moe and same GPU bytes out."""
    serve = importlib.import_module("hugpy_engine.serve.serve")
    monkeypatch.setattr(serve, "LLAMA_SERVER_BIN", "/bin/echo")
    monkeypatch.setattr(sa, "_server_supports_flag", lambda b, f: True)
    monkeypatch.setattr(spill, "autofit_gpu_layers",
                        lambda p, free_vram=None, extra_reserve_bytes=0,
                        **_kw: 17)
    # k64: any measurable momentary-free budget lands on the SAME placement —
    # backbone on the card, every expert on CPU — on BOTH sides of the seam.
    for budget in (3332 + 8000, 3332 + 100):
        plan_rig.vram["free"] = budget
        priced = A._moe_plan_for("m")
        (_argv, ngl, *_rest, ncm) = sa._build_cmd("m", path=plan_rig.path)
        assert (ngl, ncm) == (-1, spill.MOE_ALL_LAYERS), budget
        assert priced["n_cpu_moe"] == ncm, budget
        launched = spill.moe_split_need(
            spill.gguf_moe_detail(plan_rig.path), ncm)
        assert priced["gpu_weight_bytes"] == launched["gpu_bytes"], budget
        assert priced["cpu_bytes"] == launched["cpu_bytes"], budget


def test_plan_that_keeps_everything_on_the_card_is_no_split(plan_rig,
                                                            monkeypatch):
    """A budget that covers the whole model moves NOTHING to RAM — that is not
    a split, so the caller falls back to the dense full-need pricing (which is
    what the card will actually hold). k64: only a STATED contract
    (gpu_mem_gib) can reach this leaf now — a momentary-free budget always
    splits (the sibling test above)."""
    plan_rig.vram["free"] = 1 << 30
    monkeypatch.setenv("HUGPY_GPU_MEM_GIB", "1.0")   # contract covers it whole
    assert A._moe_plan_for("m") is None


def test_plan_is_none_when_the_reserve_eats_the_whole_budget(plan_rig,
                                                             monkeypatch):
    """Mirror of the slot: a budget consumed by the context/projector reserve
    plans no split at all, so admission must not price one either."""
    monkeypatch.setattr(spill, "vram_ctx_reserve_bytes",
                        lambda p, n_ctx=None: (1 << 40, "test", {}))
    plan_rig.vram["free"] = 3332 + 8000
    assert A._moe_plan_for("m") is None


def test_plan_honors_the_models_own_gpu_mem_gib_contract(plan_rig, monkeypatch):
    """The per-model VRAM budget caps the plan on BOTH sides of the seam (the
    slot caps autofit's free with it; here it is the HUGPY_GPU_MEM_GIB wire)."""
    plan_rig.vram["free"] = 1 << 30              # a card that fits everything
    monkeypatch.setenv("HUGPY_GPU_MEM_GIB", str((3332 + 8000) / float(GIB)))
    plan = A._moe_plan_for("m")
    assert plan["n_cpu_moe"] == 1 and plan["cpu_bytes"] == 8000


def test_need_detail_carries_the_planned_gpu_footprint(plan_rig, monkeypatch):
    """The fit checks consume ``moe_split.gpu_total`` — it must be the PLANNED
    GPU weights (backbone + the expert layers the budget bought) x the 1.15
    weights headroom, + the whole KV, not the backbone alone."""
    plan_rig.vram["free"] = 3332 + 8000
    monkeypatch.setattr(A, "_incoming_need_bytes", lambda mk: 19332)
    monkeypatch.setattr(A, "_kv_need_bytes", lambda mk, cfg=None: (500, {}))
    monkeypatch.setattr(A, "_calib_correction", lambda mk: None)
    det = A._incoming_need_detail("m")
    ms = det["moe_split"]
    # k64: the momentary-free budget plans backbone-only, so the planned GPU
    # footprint is the backbone x 1.15 + the whole KV.
    assert ms["n_cpu_moe"] == spill.MOE_ALL_LAYERS
    assert ms["gpu_total"] == int(3332 * 1.15) + 500
    assert ms["cpu_bytes"] == 16000


def test_plan_explicit_layer_designation_wins(plan_rig, monkeypatch):
    monkeypatch.setenv("HUGPY_N_GPU_LAYERS", "-1")        # gpu-only
    assert A._moe_plan_for("m") is None
    monkeypatch.setenv("HUGPY_N_GPU_LAYERS", "17")        # explicit count
    assert A._moe_plan_for("m") is None
    monkeypatch.setenv("HUGPY_N_GPU_LAYERS", "off")       # ram-only
    assert A._moe_plan_for("m") is None


def test_plan_mode_engine_wins(plan_rig, monkeypatch):
    monkeypatch.setenv("HUGPY_ALLOC_MODE", "max-ram")
    assert A._moe_plan_for("m") is None


def test_plan_explicit_n_cpu_moe_wins_and_prices_per_layer(plan_rig, monkeypatch):
    monkeypatch.setenv("HUGPY_N_CPU_MOE", "1")
    plan = A._moe_plan_for("m")
    assert plan["n_cpu_moe"] == 1
    assert plan["cpu_bytes"] == 8000             # layer 0's experts only
    monkeypatch.setenv("HUGPY_N_CPU_MOE", "0")   # experts on GPU: no split
    assert A._moe_plan_for("m") is None


def test_plan_dense_is_none(monkeypatch, dense_gguf):
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (dense_gguf, 48))
    for env in ("HUGPY_N_GPU_LAYERS", "HUGPY_N_CPU_MOE", "HUGPY_ALLOC_MODE"):
        monkeypatch.delenv(env, raising=False)
    assert A._moe_plan_for("m") is None


# ═══════════ THE DERIVED INITIAL ALLOCATION (operator's decision tree) ═══════
# Every model ALLOCATED to a worker gets an initial default derived from its own
# STRUCTURE, replacing the blanket stamp. The live bug this closes: on ae, 40 of
# 43 persisted allocations were a bare {"n_gpu_layers": -1}. None are MoE today,
# so it is currently harmless — but that stamp on a MoE reproduces the incident
# that stranded ae for 5.5h (a 48.4GB MoE launching -1 with no --n-cpu-moe onto
# a 24GB card, stalling 9 times). Derivation makes that stamp underivable.
#
# MEASURED ground truth used below (real files, this store):
#   coder-next Q4_K_M : non_expert 1.49 GiB / expert 43.59 GiB / 512 exp / 48 L
#   A3B-Genesis-APEX  : non_expert 1.90 GiB / expert 21.95 GiB / 256 exp / 40 L
#   A3B Q8_K_P        : non_expert 2.63 GiB / expert 37.97 GiB / 256 exp
#   Fable-5 q8_0      : non_expert 2.56 GiB / expert 33.38 GiB / 256 exp
_MEASURED = {
    "coder-next-Q4_K_M":  {"non_expert": 1.49, "expert": 43.59, "count": 512},
    "A3B-Genesis-APEX":   {"non_expert": 1.90, "expert": 21.95, "count": 256},
    "A3B-Q8_K_P":         {"non_expert": 2.63, "expert": 37.97, "count": 256},
    "Fable-5-q8_0":       {"non_expert": 2.56, "expert": 33.38, "count": 256},
}


def _moe_detail(name):
    """A gguf_moe_detail-shaped dict from the MEASURED numbers above."""
    m = _MEASURED[name]
    return {"is_moe": True, "expert_count": m["count"], "expert_used_count": 10,
            "expert_bytes": int(m["expert"] * GIB),
            "non_expert_bytes": int(m["non_expert"] * GIB),
            "expert_bytes_by_layer": {}, "files": 4}


def _total(name):
    m = _MEASURED[name]
    return int((m["non_expert"] + m["expert"]) * GIB)


# ── the MoE leaf: the split IS the allocation ────────────────────────────────
@pytest.mark.parametrize("name", sorted(_MEASURED))
def test_moe_default_is_the_derived_split_on_a_3090(name):
    """THE headline. Every measured MoE, on ae's real box (24 GiB 3090 /
    128 GiB RAM): the non-expert share fits the card with room to spare and the
    experts fit RAM, so the derived default is the SPLIT — never max-gpu, and
    never the bare -1 stamp."""
    got = AM.default_allocation("gguf", _total(name), 24 * GIB, 128 * GIB,
                                moe=_moe_detail(name))
    assert got["mode"] == "explicit"
    s = got["spill"]
    assert s["alloc_mode"] == "explicit"
    assert s["n_cpu_moe"] == AM.MOE_ALL_LAYERS   # experts to CPU
    assert s["n_gpu_layers"] == -1               # everything else on the card
    # the budgets DECLARE the two sides of the split, in GiB
    assert s["gpu_mem_gib"] == pytest.approx(_MEASURED[name]["non_expert"], abs=0.01)
    assert s["cpu_mem_gib"] == pytest.approx(_MEASURED[name]["expert"], abs=0.01)
    assert "MoE split" in got["why"]


def test_moe_45gib_model_needs_only_1_49gib_of_vram():
    """The entire point, stated as one assertion: a 45 GiB model whose derived
    GPU budget is 1.49 GiB. Priced WHOLE it 'cannot fit' a 24 GiB card and gets
    stamped; priced by STRUCTURE it fits with 22 GiB to spare."""
    d = _moe_detail("coder-next-Q4_K_M")
    total = _total("coder-next-Q4_K_M")
    assert total / GIB == pytest.approx(45.08, abs=0.05)   # 'cannot fit' 24 GiB
    got = AM.default_allocation("gguf", total, 24 * GIB, 128 * GIB, moe=d)
    assert got["spill"]["gpu_mem_gib"] == pytest.approx(1.49, abs=0.01)
    assert got["mode"] == "explicit"


def test_the_bare_minus_one_stamp_is_never_derived_for_a_moe():
    """The regression guard for the live bug: whatever the tree derives for a
    MoE, it is NEVER the bare {'n_gpu_layers': -1} that disables the load-time
    auto policy. If a -1 is present, the expert split MUST ride with it."""
    for name in _MEASURED:
        for gpu, ram in ((24 * GIB, 128 * GIB), (8 * GIB, 128 * GIB),
                         (24 * GIB, 16 * GIB), (4 * GIB, 8 * GIB)):
            s = AM.default_allocation("gguf", _total(name), gpu, ram,
                                      moe=_moe_detail(name))["spill"]
            assert s != {"n_gpu_layers": -1}
            if s.get("n_gpu_layers") == -1:
                assert s.get("n_cpu_moe"), (
                    "a derived -1 without an expert split is the ae incident")


def test_moe_experts_do_not_fit_ram_falls_through():
    """Tree leaf: the non-expert share fits the GPU but the experts do NOT fit
    RAM -> the operator's tree FALLS THROUGH rather than breaking, and the
    dense tail decides on the whole file.

    Note the arithmetic that constrains this leaf: experts < total, so any box
    whose RAM cannot hold the experts cannot hold the whole file either. The
    fall-through therefore always lands on the break leaf (max-gpu, honest
    refusal at the worker) — never on ram-only, and never on a split encoded
    from RAM that isn't there."""
    d = _moe_detail("coder-next-Q4_K_M")
    total = _total("coder-next-Q4_K_M")
    # 43.59 GiB of experts still fit a 46 GiB box, so this one DOES split
    assert AM.default_allocation("gguf", total, 24 * GIB, 46 * GIB,
                                 moe=d)["mode"] == "explicit"
    # 20 GiB of RAM holds neither the experts nor the file -> fall through
    got = AM.default_allocation("gguf", total, 24 * GIB, 20 * GIB, moe=d)
    assert got["mode"] == "max-gpu" and got["spill"] == {}
    assert "refuses honestly" in got["why"]


def test_moe_non_expert_share_too_big_for_the_card_but_fits_ram():
    """Tree leaf: even the non-expert share won't fit the GPU, whole fits RAM.

    Derives **max-ram**, not ram-only (operator, 2026-07-25: _"the 'max'
    settings were intended to be indicative of a preference for spill... the
    'only' designations are the truly stringent ones"_). A GGUF still keeps the
    layers that fit, so forbidding the card outright would cost ~5x throughput
    by today's cliff measurement (~36 tok/s partial vs ~7.5 at ngl=0). A
    default must never promise that.

    A tiny 1 GiB card against coder-next's 1.49 GiB of non-expert tensors."""
    got = AM.default_allocation("gguf", _total("coder-next-Q4_K_M"),
                                1 * GIB, 128 * GIB,
                                moe=_moe_detail("coder-next-Q4_K_M"))
    assert got["mode"] == "max-ram"
    assert got["spill"] == {"alloc_mode": "max-ram"}


def test_moe_fits_nothing_is_the_break_leaf():
    """Tree leaf '-- break': fits neither the GPU nor RAM. No fourth state is
    invented — max-gpu/{} so the WORKER refuses honestly with the real numbers
    at load, where the refusal can actually be explained."""
    got = AM.default_allocation("gguf", _total("coder-next-Q4_K_M"),
                                1 * GIB, 4 * GIB,
                                moe=_moe_detail("coder-next-Q4_K_M"))
    assert got["mode"] == "max-gpu" and got["spill"] == {}
    assert "refuses honestly" in got["why"]


# ── the dense / transformers leaves stay exactly as they were ────────────────
def test_dense_gguf_leaves_are_unchanged():
    # fits-whole -> gpu-only since the operator default order 2026-07-31
    fits = AM.default_allocation("gguf", 5 * GIB, 24 * GIB, 128 * GIB)
    assert fits["mode"] == "gpu-only" and fits["spill"] == {"n_gpu_layers": -1}
    big = AM.default_allocation("gguf", 68 * GIB, 24 * GIB, 128 * GIB)
    assert big["mode"] == "max-ram"
    huge = AM.default_allocation("gguf", 400 * GIB, 24 * GIB, 128 * GIB)
    assert huge["mode"] == "max-gpu" and huge["spill"] == {}


def test_transformers_leaves_match_the_established_behaviour():
    """The transformers branch was already correct; the tree must not move it."""
    for size, gpu, ram, want in ((5 * GIB, 24 * GIB, 124 * GIB, "gpu-only"),
                                 (68 * GIB, 24 * GIB, 124 * GIB, "ram-only"),
                                 (200 * GIB, 24 * GIB, 124 * GIB, "max-gpu")):
        got = AM.default_allocation("transformers", size, gpu, ram)
        assert got["mode"] == want
        assert got["mode"] == AM.feasible_default_mode("transformers", size,
                                                       gpu, ram)


def test_fitting_model_derives_gpu_only():
    """Operator default order 2026-07-31: a model that fits the card WHOLE
    derives gpu-only (full-card residency) — the 07-25 max-gpu departure is
    reversed now that gpu-only is strict-and-honest (k55): a fit miss under
    contention is an explicit refusal naming max-gpu, not a llama.cpp OOM.
    Oversized/unknown sizes never reach this leaf, so the default stays a
    success path per (model x worker)."""
    for engine in ("transformers", "gguf"):
        got = AM.default_allocation(engine, 5 * GIB, 24 * GIB, 128 * GIB)
        assert got["mode"] == "gpu-only"
        assert got["spill"] == {"n_gpu_layers": -1}


# ── degrade-not-guess: a default is never derived from a guessed number ──────
@pytest.mark.parametrize("size,gpu,ram", [
    (None, 24 * GIB, 128 * GIB),                 # unknown size
    (_total("coder-next-Q4_K_M"), None, 128 * GIB),   # unknown GPU total
])
def test_missing_measurements_degrade_to_todays_behaviour(size, gpu, ram):
    got = AM.default_allocation("gguf", size, gpu, ram,
                                moe=_moe_detail("coder-next-Q4_K_M"))
    assert got["mode"] == "max-gpu" and got["spill"] == {}
    assert "degrade-not-guess" in got["why"]


def test_unknown_ram_never_derives_a_split_or_a_ram_only():
    """RAM unknown: the split's CPU side is unpriceable and ram-only is
    unjustifiable, so both are off the table — max-gpu, and the worker's own
    auto policy still reaches the right placement."""
    got = AM.default_allocation("gguf", _total("coder-next-Q4_K_M"),
                                24 * GIB, None,
                                moe=_moe_detail("coder-next-Q4_K_M"))
    assert got["mode"] == "max-gpu" and got["spill"] == {}


def test_unreadable_moe_detail_degrades_to_the_dense_path():
    """A detected MoE whose split is unpriceable (missing byte counts) must not
    encode a split from numbers we don't have — it falls to the dense tail."""
    broken = {"is_moe": True, "expert_bytes": 0, "non_expert_bytes": 0}
    got = AM.default_allocation("gguf", 45 * GIB, 24 * GIB, 128 * GIB, moe=broken)
    assert got["mode"] == "max-ram"              # 45 GiB: dense tail, fits RAM
    assert "unpriceable" in got["why"]
    # no moe detail supplied at all -> pure dense path
    assert AM.default_allocation("gguf", 45 * GIB, 24 * GIB, 128 * GIB
                                 )["mode"] == "max-ram"
    # is_moe False is dense too (fits whole -> gpu-only, 2026-07-31 order)
    assert AM.default_allocation("gguf", 5 * GIB, 24 * GIB, 128 * GIB,
                                 moe={"is_moe": False})["mode"] == "gpu-only"


# ── the two views agree; the mirrored constant cannot drift ──────────────────
def test_feasible_default_mode_agrees_with_the_allocation_view():
    """feasible_default_mode is the NAME view of the same tree — it delegates
    for MoE, so the two can never disagree about one model."""
    d = _moe_detail("coder-next-Q4_K_M")
    t = _total("coder-next-Q4_K_M")
    for gpu, ram in ((24 * GIB, 128 * GIB), (1 * GIB, 128 * GIB),
                     (1 * GIB, 4 * GIB), (24 * GIB, None)):
        assert (AM.feasible_default_mode("gguf", t, gpu, ram, moe=d)
                == AM.default_allocation("gguf", t, gpu, ram, moe=d)["mode"])
    # ...and WITHOUT a moe detail they must still agree — which is the whole
    # point, so assert AGREEMENT, never a hardcoded mode.
    #
    # ⚠ This line previously asserted `== "max-gpu"`, encoding the very bug the
    # delegation above was written to prevent: feasible_default_mode carried a
    # blanket "a GGUF is ALWAYS max-gpu" short-circuit while default_allocation
    # walked the tree to max-ram. A 45 GiB GGUF on a 24 GiB card with 128 GiB
    # RAM really is max-ram (it cannot fit the card; RAM-first with the overflow
    # spilling to GPU is the honest answer). The console dropdown read the name
    # view and the emitting seam read the other, so the label could promise
    # max-gpu while the seam did max-ram — and once preference drives EVICTION
    # too (assets/evictionflow.html), a wrong label mispredicts which model
    # DIES, not merely where one lands.
    #
    # This file's own header warns that it "has TWICE shipped tests that
    # asserted the bug". This was the third. Pinning agreement instead of a
    # value is what stops a fourth.
    assert (AM.feasible_default_mode("gguf", t, 24 * GIB, 128 * GIB)
            == AM.default_allocation("gguf", t, 24 * GIB, 128 * GIB,
                                     moe={"is_moe": False})["mode"])


def test_moe_all_layers_mirror_matches_spill():
    """alloc_modes stays stdlib-pure, so it mirrors the sentinel rather than
    importing the GGUF reader. Assert the mirror so it cannot drift silently."""
    assert AM.MOE_ALL_LAYERS == spill.MOE_ALL_LAYERS


def test_derived_moe_spill_rides_the_existing_version_gate():
    """No new wire contract: every key the MoE leaf emits is already honored by
    a released worker, and the pair that needs gating is already in
    NEW_SPILL_KEYS. An old worker gets {} (max-gpu) — whose own auto MoE policy
    reaches the right placement anyway, so the downgrade is not a dud."""
    s = AM.default_allocation("gguf", _total("coder-next-Q4_K_M"), 24 * GIB,
                              128 * GIB, moe=_moe_detail("coder-next-Q4_K_M")
                              )["spill"]
    assert set(s) & AM.NEW_SPILL_KEYS            # gated by construction
    old, note = AM.gate_spill_for_worker(s, "0.1.150", "old-box")
    assert old == {} and note
    new, note2 = AM.gate_spill_for_worker(s, "0.1.208", "ae")   # the live fleet
    assert new == s and note2 is None
    # every emitted key is one the released worker actually reads
    assert set(s) <= set(A._SPILL_ENV)


def test_derived_moe_spill_reproduces_the_auto_policy_argv(cmd_rig, monkeypatch):
    """END-TO-END, the promise check: feed the DERIVED spill through the real
    worker seam (_apply_spill -> env -> the load opts the slot child gets) and
    assert _build_cmd emits the SAME argv the measured auto policy produces
    (--n-gpu-layers -1 --n-cpu-moe 999). This is what proves the derived
    default is a success path and not a plausible-looking dud — in particular
    that setting alloc_mode did not silence the split."""
    d = spill.gguf_moe_detail(cmd_rig.moe)
    got = AM.default_allocation("gguf", 45 * GIB, 24 * GIB, 128 * GIB, moe=d)
    A._apply_spill(got["spill"])
    try:
        # what llama.runners.get forwards to the slot child from that env
        assert os.environ["HUGPY_N_CPU_MOE"] == str(AM.MOE_ALL_LAYERS)
        assert os.environ["HUGPY_N_GPU_LAYERS"] == "-1"
        cmd_rig.auto["value"] = 17               # autofit would have said 17/48
        (argv, ngl, *_rest, ncm) = sa._build_cmd(
            "moe-model", path=cmd_rig.moe,
            n_gpu_layers=int(os.environ["HUGPY_N_GPU_LAYERS"]),
            n_cpu_moe=os.environ["HUGPY_N_CPU_MOE"])
        pairs = _argv_pairs(argv)
        assert ngl == -1 and pairs["--n-gpu-layers"] == "all"
        assert ncm == AM.MOE_ALL_LAYERS and pairs["--n-cpu-moe"] == "999"
    finally:
        A._apply_spill({})


def test_derived_split_matches_moe_split_need():
    """The derived budgets must equal what spill.moe_split_need prices for the
    all-experts-on-CPU split — one source of truth for the two sides."""
    d = _moe_detail("A3B-Genesis-APEX")
    need = spill.moe_split_need(d)
    s = AM.default_allocation("gguf", _total("A3B-Genesis-APEX"), 24 * GIB,
                              128 * GIB, moe=d)["spill"]
    assert s["gpu_mem_gib"] == pytest.approx(need["gpu_bytes"] / GIB, abs=0.01)
    assert s["cpu_mem_gib"] == pytest.approx(need["cpu_bytes"] / GIB, abs=0.01)


# ═══════════ the LAST HOP: in-process cannot express the split ══════════════
# Operator ruling 2026-07-26: "the auto should be moe if it is an moe model."
# Central derived the split correctly and put it on the wire; it died at the
# runner seam, where the in-process llama-cpp-python path has no --n-cpu-moe
# equivalent (spill.llama_kwargs carries n_gpu_layers/tensor_split/main_gpu
# only). An MoE reaching that path loads WHOLE: ngl=-1 drags the experts onto
# the card (coder-next: 1.49 GiB non-expert -> ~21 GiB observed on ae's 3090).
def test_llama_kwargs_cannot_express_the_expert_split(moe_gguf, monkeypatch):
    """REGRESSION ANCHOR for the root cause. If a future llama-cpp-python gains
    an expert-split parameter, this test fails and the refusal below can be
    replaced by actually passing it — that is the intended signal, not a break."""
    monkeypatch.delenv("HUGPY_RPC_SERVERS", raising=False)
    kw = spill.llama_kwargs(str(moe_gguf))
    assert "n_cpu_moe" not in kw, (
        "llama-cpp-python now expresses the expert split — teach the "
        "in-process path to pass it and relax the refusal in runners/get.py")
    assert spill.gguf_moe_detail(str(moe_gguf))["is_moe"] is True


def test_moe_refuses_the_in_process_fallback_instead_of_loading_whole(monkeypatch):
    """An MoE GGUF that reaches the in-process fallback must RAISE, not serve.

    Serving anyway would honor the request while silently discarding the
    allocation derived for it — the split IS the allocation, so dropping it is
    not a degraded success but a different (much worse) placement wearing the
    same name. Mirrors the profile/vision refusals at the same seam."""
    getmod = importlib.import_module(
        "hugpy_engine.llama.runners.get")

    # Force the seam: no slot can seat it, so control reaches the fallback.
    monkeypatch.setattr(getmod, "_require_profile_ready", lambda k: None)
    _slots = importlib.import_module("hugpy_engine.serve.slots")
    monkeypatch.setattr(_slots, "slots_enabled", lambda: False)
    # Measured MoE, priced like coder-next.
    monkeypatch.setattr(getmod, "get_model_config", lambda k: object(),
                        raising=False)
    monkeypatch.setattr(getmod, "ensure_serving_weights", lambda k: "/models/cn",
                        raising=False)
    monkeypatch.setattr(getmod, "get_gguf_file",
                        lambda d, c: "/models/cn/coder-next.gguf", raising=False)
    monkeypatch.setattr(
        "hugpy_engine.spill.gguf_moe_detail",
        lambda p: {"is_moe": True,
                   "non_expert_bytes": int(1.49 * GIB),
                   "expert_bytes": int(43.59 * GIB)})

    built = []
    monkeypatch.setattr(getmod, "LlamaCppPythonRunner",
                        lambda k: built.append(k), raising=False)

    with pytest.raises(getmod.LocalEngineUnavailable) as ei:
        getmod._build_runner("Qwen~Qwen3-Coder-Next-GGUF")

    msg = str(ei.value)
    assert "MoE GGUF" in msg and "--n-cpu-moe" in msg
    assert "1.49 GiB non-expert" in msg and "43.59 GiB" in msg
    assert not built, "in-process runner was built for an MoE — the split was dropped"


def test_dense_models_still_take_the_in_process_fallback(monkeypatch):
    """The guard is MoE-only: a dense GGUF must be untouched (and gguf_moe_detail
    degrades to {'is_moe': False} on any read failure, so the gate cannot fire
    on an unreadable header either)."""
    getmod = importlib.import_module(
        "hugpy_engine.llama.runners.get")
    monkeypatch.setattr(getmod, "_require_profile_ready", lambda k: None)
    _slots = importlib.import_module("hugpy_engine.serve.slots")
    monkeypatch.setattr(_slots, "slots_enabled", lambda: False)
    monkeypatch.setattr(getmod, "get_model_config", lambda k: object(),
                        raising=False)
    monkeypatch.setattr(getmod, "ensure_serving_weights", lambda k: "/models/d",
                        raising=False)
    monkeypatch.setattr(getmod, "get_gguf_file",
                        lambda d, c: "/models/d/dense.gguf", raising=False)
    monkeypatch.setattr("hugpy_engine.spill.gguf_moe_detail",
                        lambda p: {"is_moe": False})
    _ss = importlib.import_module(
        "hugpy_engine.llama.runners.src.shard_server")
    monkeypatch.setattr(_ss, "ensure_vision_server", lambda k: None)

    built = []

    class _R:
        def __init__(self, k):
            built.append(k)

    monkeypatch.setattr(getmod, "LlamaCppPythonRunner", _R, raising=False)
    getmod._build_runner("some-dense-model")
    assert built == ["some-dense-model"], "dense model was wrongly refused"


# ═══════════ probe fit honesty: price the split, not the VRAM delta ═════════
class _State:
    central_url = "http://central.invalid"


def _probe_gates(monkeypatch, A):
    """Satisfy _probe_model's pre-load gates (central metadata + locality) so the
    test reaches the FIT verdict, which is what these cases are about."""
    prov = importlib.import_module("hugpy_storage.provision")
    monkeypatch.setattr(prov, "ensure_model_registered",
                        lambda k, url=None: k, raising=False)
    monkeypatch.setattr(prov, "model_is_local", lambda k: True, raising=False)


def test_probe_fit_is_moe_aware_not_a_vram_delta(monkeypatch):
    """`fit` asked "did GPU free memory drop" — the RIGHT question for a dense
    model, the WRONG one for a split MoE. Under the derived split only the
    non-expert share lands on the card (coder-next: ~1.49 GiB of 45 GiB), so a
    PERFECTLY placed MoE barely trips the 64 MiB threshold and a large-ctx
    variant misses it outright, reporting fit:false for the configuration we
    actually want. Observed on ae: fit:false, vram_used:0 both with and without
    an explicit split. Now the verdict prices the non-expert share vs the card."""
    A = importlib.import_module("hugpy_fleet.worker.agent")

    class _Runner:
        base_url = "http://127.0.0.1:8101"

        def ensure_loaded(self):
            return None

    import hugpy_fleet.worker.imports as _WI
    monkeypatch.setattr(_WI, "get_model_path", lambda k, *a, **kw: "/tmp/fake-model.gguf")
    monkeypatch.setattr(A, "runner_for", lambda model_key: _Runner(),
                        raising=False)
    _probe_gates(monkeypatch, A)
    # A split MoE consumes almost no VRAM — the old rule's blind spot.
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: 20 * GIB, raising=False)
    monkeypatch.setattr("hugpy_engine.spill.total_vram_bytes",
                        lambda: 24 * GIB)
    monkeypatch.setattr(
        "hugpy_engine.spill.gguf_moe_detail",
        lambda p: {"is_moe": True,
                   "non_expert_bytes": int(1.49 * GIB),
                   "expert_bytes": int(43.59 * GIB)})

    res = A._probe_model("Qwen~Qwen3-Coder-Next-GGUF", _State())

    assert res["vram_used"] == 0, "rig: no VRAM delta (the split's whole point)"
    assert res["fit"] is True, (
        f"a correctly split MoE must FIT: 1.49 GiB non-expert on a 24 GiB card: {res}")
    basis = res.get("moe_fit_basis") or {}
    assert basis.get("is_moe") is True and "non-expert" in basis.get("why", "")


def test_probe_fit_refuses_an_moe_whose_non_expert_share_overflows(monkeypatch):
    """The inverse must still be honest: when even the non-expert share exceeds
    the card there is no split worth seating, so fit stays False."""
    A = importlib.import_module("hugpy_fleet.worker.agent")

    class _Runner:
        base_url = "http://127.0.0.1:8101"

        def ensure_loaded(self):
            return None

    import hugpy_fleet.worker.imports as _WI
    monkeypatch.setattr(_WI, "get_model_path", lambda k, *a, **kw: "/tmp/fake-model.gguf")
    monkeypatch.setattr(A, "runner_for", lambda model_key: _Runner(),
                        raising=False)
    _probe_gates(monkeypatch, A)
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: 20 * GIB, raising=False)
    monkeypatch.setattr("hugpy_engine.spill.total_vram_bytes",
                        lambda: 8 * GIB)
    monkeypatch.setattr(
        "hugpy_engine.spill.gguf_moe_detail",
        lambda p: {"is_moe": True,
                   "non_expert_bytes": int(30 * GIB),
                   "expert_bytes": int(43.59 * GIB)})

    res = A._probe_model("huge-moe", _State())
    assert res["fit"] is False


def test_probe_fit_unchanged_for_dense_models(monkeypatch):
    """Dense GGUFs keep the raw VRAM-delta rule verbatim (and an unreadable
    header degrades to dense, so the new branch cannot fire on a guess)."""
    A = importlib.import_module("hugpy_fleet.worker.agent")

    class _Runner:
        base_url = None

        def ensure_loaded(self):
            return None

    import hugpy_fleet.worker.imports as _WI
    monkeypatch.setattr(_WI, "get_model_path", lambda k, *a, **kw: "/tmp/fake-model.gguf")
    monkeypatch.setattr(A, "runner_for", lambda model_key: _Runner(),
                        raising=False)
    _probe_gates(monkeypatch, A)
    monkeypatch.setattr("hugpy_engine.spill.gguf_moe_detail",
                        lambda p: {"is_moe": False})
    seq = iter([20 * GIB, 16 * GIB])          # 4 GiB actually landed on the card
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: next(seq), raising=False)

    res = A._probe_model("dense-model", _State())
    assert res["fit"] is True and res["vram_used"] == 4 * GIB
    assert "moe_fit_basis" not in res
