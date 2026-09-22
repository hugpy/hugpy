"""Slice 11 / t27 — context (KV cache) quantified into fit need + as an
allocation variable.

Operator (2026-07-17): "the context can necessarily be quantified into ram
needed correct? ... this should be a variable as well based on percentage max."

KV cache = 2 (K+V) × n_layers × ctx × n_kv_heads × head_dim × dtype_bytes — the
attention key/value tensors the model holds for the whole context window, the
RAM/VRAM tax fit/admission ignored (weights-only). Per-model `ctx_pct` (1-100)
scales the model's max context; resolved ctx = pct × max, clamped to the engine
cap. need = weights + kv(resolved ctx) everywhere; kv=0 when ctx_pct unset
(byte-identical to today). Served -c honours the resolved value (contract).

Geometry verified 2026-07-17 against the CATALOG:
  * GGUF: Qwen2.5-Coder-3B q4 header — block_count=36, head_count=16,
    head_count_kv=2, embedding_length=2048 (head_dim=128), context_length=32768.
  * transformers: DavidAU/MN-GRAND-23.5B config.json — 81 layers, 32 heads,
    8 kv heads, head_dim=128, bfloat16.

Run: venv/bin/python -m pytest tests/test_kv_context_allocation.py -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib                                                 # noqa: E402
from hugpy_fleet.worker import agent as A
from hugpy_engine import spill
# managers/__init__ star-imports shadow the subpackage attrs — bind the REAL
# module via import_module (same landmine the sibling fit tests note).
_D = importlib.import_module("hugpy_engine.dispatch.dispatch")

GIB = 1 << 30


# ═══════════════ KV math against known geometry ════════════════════════════
def test_kv_bytes_exact_gguf_geometry():
    """Qwen2.5-Coder-3B: 36 layers, 2 kv heads, 128 head_dim, fp16, full 32768
    ctx = 2*36*32768*2*128*2 = 1.125 GiB exactly."""
    kv = spill.kv_bytes(ctx_tokens=32768, n_layers=36, n_kv_heads=2,
                        head_dim=128, dtype_bytes=2.0)
    assert kv == 2 * 36 * 32768 * 2 * 128 * 2
    assert round(kv / GIB, 3) == 1.125


def test_kv_bytes_exact_transformers_geometry():
    """MN-GRAND-23.5B: 81 layers, 8 kv heads, 128 head_dim, bf16, 8192 ctx =
    2*81*8192*8*128*2 = 2.53 GiB."""
    kv = spill.kv_bytes(ctx_tokens=8192, n_layers=81, n_kv_heads=8,
                        head_dim=128, dtype_bytes=spill._kv_dtype_bytes("bfloat16"))
    assert kv == 2 * 81 * 8192 * 8 * 128 * 2
    assert round(kv / GIB, 2) == 2.53


def test_kv_scales_linearly_with_ctx():
    full = spill.kv_bytes(ctx_tokens=32768, n_layers=36, n_kv_heads=2, head_dim=128)
    half = spill.kv_bytes(ctx_tokens=16384, n_layers=36, n_kv_heads=2, head_dim=128)
    assert half == full // 2


def test_quantized_kv_dtype_halves():
    fp16 = spill.kv_bytes(ctx_tokens=8192, n_layers=32, n_kv_heads=8, head_dim=128,
                          dtype_bytes=spill._kv_dtype_bytes("f16"))
    q8 = spill.kv_bytes(ctx_tokens=8192, n_layers=32, n_kv_heads=8, head_dim=128,
                        dtype_bytes=spill._kv_dtype_bytes("q8_0"))
    assert q8 == fp16 // 2


def test_kv_never_silently_zero_without_geometry():
    """Missing geometry -> conservative heuristic (× assumed layers × ctx), never
    zero — the ctx tax must never vanish."""
    kv = spill.kv_bytes(ctx_tokens=4096)          # no geometry at all
    assert kv is not None and kv > 0
    # with a known layer count it scales by layers.
    kv2 = spill.kv_bytes(ctx_tokens=4096, n_layers=80)
    assert kv2 > kv


def test_kv_zero_ctx_is_none():
    assert spill.kv_bytes(ctx_tokens=0, n_layers=36, n_kv_heads=2, head_dim=128) is None


# ═══════════════ geometry readers against the real catalog ═════════════════
GGUF_PATH = ("/mnt/llm_storage/models/gguf/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/"
             "qwen2.5-coder-3b-instruct-q4_k_m.gguf")
TRANSFORMERS_CFG = ("/mnt/llm_storage/models/transformers/DavidAU/"
                    "MN-GRAND-23.5B-Gutenberg-UNCENSORED-V2-GLM4.7-Thinking/config.json")


@pytest.mark.skipif(not Path(GGUF_PATH).is_file(),
                    reason="catalog GGUF not present on this box")
def test_reads_real_gguf_geometry():
    geo = spill._gguf_kv_geometry(GGUF_PATH)
    assert geo["n_layers"] == 36
    assert geo["n_kv_heads"] == 2
    assert geo["head_dim"] == 128
    assert geo["ctx_train"] == 32768


@pytest.mark.skipif(not Path(TRANSFORMERS_CFG).is_file(),
                    reason="catalog transformers config not present")
def test_reads_real_transformers_geometry():
    import json
    geo = spill._transformers_kv_geometry(json.load(open(TRANSFORMERS_CFG)))
    assert geo["n_layers"] == 81
    assert geo["n_kv_heads"] == 8
    assert geo["head_dim"] == 128
    assert geo["dtype"] == "bfloat16"


# ═══════════════ ctx_pct resolution + clamps ═══════════════════════════════
@pytest.fixture
def ctx_env(monkeypatch):
    A._RUNTIME_SETTINGS.clear()
    monkeypatch.setattr(A, "_model_max_ctx", lambda mk, cfg=None: 32768)
    yield
    A._RUNTIME_SETTINGS.clear()


def test_ctx_pct_reads_and_clamps(ctx_env):
    A._RUNTIME_SETTINGS.update({"ctx_pct": {"a": 50, "lo": 0, "hi": 250, "bad": "x"}})
    assert A._ctx_pct("a") == 50
    assert A._ctx_pct("lo") == 1          # clamped up
    assert A._ctx_pct("hi") == 100        # clamped down
    assert A._ctx_pct("bad") is None      # non-numeric
    assert A._ctx_pct("unset") is None    # absent -> default


def test_resolved_ctx_is_pct_of_max(ctx_env):
    A._RUNTIME_SETTINGS.update({"ctx_pct": {"m": 50}})
    ctx, pct, mx = A._resolved_ctx("m", {"framework": "transformers"})
    assert (ctx, pct, mx) == (16384, 50, 32768)


def test_resolved_ctx_clamps_to_engine_cap(ctx_env, monkeypatch):
    # 100% of 32768 for a GGUF is capped to DEFAULT_LLAMA_CTX (16384 by default).
    from hugpy_engine.serve import serve
    monkeypatch.setattr(serve, "DEFAULT_LLAMA_CTX", 16384)
    A._RUNTIME_SETTINGS.update({"ctx_pct": {"m": 100}})
    ctx, pct, mx = A._resolved_ctx("m", {"framework": "gguf"})
    assert ctx == 16384                   # capped, not 32768
    assert mx == 32768


def test_unset_ctx_pct_yields_no_resolution(ctx_env):
    ctx, pct, mx = A._resolved_ctx("m", {"framework": "gguf"})
    assert ctx is None and pct is None    # -> today's default ctx path


# ═══════════════ need = weights + kv, unified across fit paths ═════════════
@pytest.fixture
def need_rig(monkeypatch):
    """Stub the weight sizer + KV so need composition is deterministic."""
    monkeypatch.setattr(A, "_incoming_need_bytes", lambda mk: 21 * GIB)   # weights
    monkeypatch.setattr(A, "_kv_need_bytes",
                        lambda mk, cfg=None: (3 * GIB, {
                            "ctx_pct": 50, "ctx_resolved": 16384, "ctx_max": 32768,
                            "geometry_source": "geometry", "kv_bytes": 3 * GIB}))
    return monkeypatch


def test_incoming_need_detail_sums_weights_and_kv(need_rig):
    det = A._incoming_need_detail("m")
    assert det["weights"] == 21 * GIB
    assert det["kv"] == 3 * GIB
    assert det["total"] == 24 * GIB
    assert det["ctx_pct"] == 50


def test_need_is_weights_only_when_no_ctx(monkeypatch):
    """Back-compat: kv=0 when ctx_pct unset -> total == weights (byte-identical)."""
    monkeypatch.setattr(A, "_incoming_need_bytes", lambda mk: 21 * GIB)
    monkeypatch.setattr(A, "_kv_need_bytes",
                        lambda mk, cfg=None: (0, {"ctx_pct": None,
                                                  "ctx_resolved": None,
                                                  "ctx_max": 32768,
                                                  "geometry_source": None}))
    det = A._incoming_need_detail("m")
    assert det["kv"] == 0
    assert det["total"] == 21 * GIB       # weights only, exactly as today


def test_unmeasurable_weights_fails_open(monkeypatch):
    monkeypatch.setattr(A, "_incoming_need_bytes", lambda mk: None)
    det = A._incoming_need_detail("m")
    assert det["total"] is None           # fail-open unchanged


def test_all_three_fit_paths_use_the_same_need(monkeypatch):
    """_worker_fit_check, _worker_slot_fit_check, and _vram_evict_to_fit must all
    size against _incoming_need_detail's total — no path sees weights-only."""
    seen = []
    monkeypatch.setattr(A, "_incoming_need_detail",
                        lambda mk: (seen.append(mk),
                                    {"total": 24 * GIB, "weights": 21 * GIB,
                                     "kv": 3 * GIB, "ctx_pct": 50,
                                     "ctx_resolved": 16384, "ctx_max": 32768,
                                     "geometry_source": "geometry"})[1])
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: 30 * GIB)
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: 40 * GIB)
    # contention fit
    A._worker_fit_check("m")
    # slot ceiling fit
    A._worker_slot_fit_check("m")
    # slice-10 admission (fits at 30 free, 24 need, 4 reserve -> proceed)
    plan = A._vram_evict_to_fit(type("S", (), {})(), "m")
    assert seen.count("m") >= 3           # every path consulted the unified need
    assert plan["action"] == "proceed"


# ═══════════════ refusal shows the split ═══════════════════════════════════
def test_refusal_reports_weights_kv_split(monkeypatch):
    monkeypatch.setattr(A, "_incoming_need_detail",
                        lambda mk: {"total": 24 * GIB, "weights": 21 * GIB,
                                    "kv": 3 * GIB, "ctx_pct": 50,
                                    "ctx_resolved": 16384, "ctx_max": 32768,
                                    "geometry_source": "geometry"})
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: 24 * GIB)
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: 1 * GIB)    # can't fit
    monkeypatch.setattr(A, "_vram_residents", lambda s: [])        # nothing evictable
    plan = A._vram_evict_to_fit(type("S", (), {})(), "m")
    assert plan["action"] == "refuse"
    r = plan["reason"]
    assert r["needs_weights_bytes"] == 21 * GIB
    assert r["needs_kv_bytes"] == 3 * GIB
    assert r["ctx_pct"] == 50
    assert r["ctx_resolved"] == 16384
    # the human string carries "= 21.0 GB weights + 3.0 GB kv@50%ctx"
    assert "weights +" in r["reason"] and "kv@50%ctx" in r["reason"]


def test_admission_evicts_against_kv_inclusive_need(monkeypatch):
    """slice-10 admission must plan against weights+kv: a load that FITS on
    weights alone but NOT with kv triggers an eviction."""
    monkeypatch.setattr(A, "_incoming_need_detail",
                        lambda mk: {"total": 8 * GIB, "weights": 5 * GIB,
                                    "kv": 3 * GIB, "ctx_pct": 50,
                                    "ctx_resolved": 16384, "ctx_max": 32768,
                                    "geometry_source": "geometry"})
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: 24 * GIB)
    # 7G free: fits 5G weights (+2.4G reserve = 7.4 > 7 free? no) — with kv the
    # 8G need forces an eviction. Free after evicting the 6G idle -> 13G, fits.
    free = {"v": 7 * GIB}
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: free["v"])
    monkeypatch.setattr(A, "_vram_residents",
                        lambda s: [{"model_key": "idle", "vram_bytes": 6 * GIB,
                                    "host_mode": "subprocess", "alive": True}])
    monkeypatch.setattr(A, "_residency", lambda mk: "on-demand")
    monkeypatch.setattr(A, "_busy_slot_models", lambda: set())
    monkeypatch.setattr(A, "_actively_replying", lambda mk, busy=None: False)
    monkeypatch.setattr(A, "_queued_ahead_of", lambda subj: set())
    monkeypatch.setattr(A, "_trim_host_ram", lambda: None)
    monkeypatch.setattr(_D, "last_used_snapshot", lambda: {"idle": 1.0})

    def _fake_evict(state, mk, force=False):
        free["v"] += 6 * GIB              # reclaim
        return {"model_key": mk, "evicted": True, "vram_freed": 6 * GIB,
                "host_mode": "subprocess"}
    monkeypatch.setattr(A, "_evict_model", _fake_evict)
    plan = A._vram_evict_to_fit(type("S", (), {})(), "m")
    assert plan["action"] == "evicted"
    assert plan["evicted"] == ["idle"]    # kv-inclusive need forced the eviction


# ═══════════════ serving truth: resolved ctx is applied at load ════════════
def test_ctx_resolver_wires_resolved_ctx_into_serving(monkeypatch):
    """serve._ctx_for must serve the RESOLVED ctx when a resolver is registered
    — the allocation is a contract, not a hint."""
    from hugpy_engine.serve import serve
    monkeypatch.setattr(serve, "_effective_extra", lambda mk, cfg: {})   # no override
    serve.set_ctx_resolver(lambda mk, cfg=None: 16384)
    try:
        cfg = type("C", (), {"model_max_length": 32768})()
        assert serve._ctx_for(cfg, "m") == 16384      # resolved value served
    finally:
        serve.set_ctx_resolver(None)


def test_ctx_resolver_unset_is_todays_default(monkeypatch):
    """No resolver -> the historical capping path, byte-identical."""
    from hugpy_engine.serve import serve
    monkeypatch.setattr(serve, "_effective_extra", lambda mk, cfg: {})
    monkeypatch.setattr(serve, "DEFAULT_LLAMA_CTX", 16384)
    serve.set_ctx_resolver(None)
    cfg = type("C", (), {"model_max_length": 32768})()
    assert serve._ctx_for(cfg, "m") == 16384          # min(32768, 16384) as before


def test_explicit_llama_ctx_override_still_wins(monkeypatch):
    """A manual extra['llama_ctx'] override beats the ctx_pct resolver."""
    from hugpy_engine.serve import serve
    monkeypatch.setattr(serve, "_effective_extra", lambda mk, cfg: {"llama_ctx": 8000})
    serve.set_ctx_resolver(lambda mk, cfg=None: 16384)
    try:
        assert serve._ctx_for(type("C", (), {})(), "m") == 8000
    finally:
        serve.set_ctx_resolver(None)


# ═══════════════ /ops/config accepts ctx_pct (wire) ════════════════════════
def test_settings_keys_include_ctx_pct():
    assert "ctx_pct" in A._SETTINGS_KEYS


def test_effective_config_surfaces_ctx_pct():
    A._RUNTIME_SETTINGS.clear()
    A._RUNTIME_SETTINGS.update({"ctx_pct": {"m": 50}})
    try:
        out = A._effective_config()
        assert out.get("ctx_pct") == {"m": 50}
    finally:
        A._RUNTIME_SETTINGS.clear()


def test_ops_config_ctx_pct_merge_semantics():
    """The /ops/config merge branch: valid int merges, null/'' clears, and an
    out-of-range 0 is REJECTED (not silently cleared by the 0==False trap)."""
    # Reproduce the exact merge branch (isolated from the restart-scheduling app).
    def _merge(body, settings):
        if not isinstance(body.get("ctx_pct"), dict):
            return "bad-dict"
        cmerged = dict(settings.get("ctx_pct") or {})
        for mk, val in body["ctx_pct"].items():
            if val is None or val == "":
                cmerged.pop(mk, None)
                continue
            try:
                pv = int(val)
            except (TypeError, ValueError):
                return f"bad-{mk}"
            if not (1 <= pv <= 100):
                return f"range-{mk}={pv}"
            cmerged[mk] = pv
        if cmerged:
            settings["ctx_pct"] = cmerged
        else:
            settings.pop("ctx_pct", None)
        return settings
    assert _merge({"ctx_pct": {"a": 50}}, {}) == {"ctx_pct": {"a": 50}}
    assert _merge({"ctx_pct": {"a": None}}, {"ctx_pct": {"a": 50}}) == {}   # cleared
    assert _merge({"ctx_pct": {"a": 0}}, {}) == "range-a=0"                 # NOT cleared
    assert _merge({"ctx_pct": {"a": 101}}, {}) == "range-a=101"
    assert _merge({"ctx_pct": {"a": "x"}}, {}) == "bad-a"
