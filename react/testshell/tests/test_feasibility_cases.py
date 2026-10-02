"""Feasibility of the operator's test set on ae-worker's budgets (2026-10-02):

  * Anko            GGUF Q4_K_M, MoE 35B-A3B      — gpu-only ctx ceiling, MoE explicit reaches trained ctx
  * Qwen3.6-35B-A3B safetensors, hybrid MoE, 4-bit — bf16 never fits a 24 GiB card; 4-bit fits with a ctx
                                                    ceiling below trained at f16 (exceeds at 100%), fits at q8_0
  * MN-GRAND 23.5B  dense: safetensors (4-bit) + GGUF Q4_K_M — ceilings far below the 1,024,000 trained ctx

Numbers are the stored facts (model_quant_facts / hugpy.json) pinned here; no DB, no files."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import compute  # noqa: E402
from hugpy_engine import spill as _spill  # noqa: E402

GIB = 1024 ** 3
COMPUTE = _spill._CTX_COMPUTE_RESERVE_BYTES          # llama.cpp scratch (GGUF)
COMPUTE_T = compute.compute_reserve("transformers")   # HF activations + dequant workspace (2 GiB default)
# ae-worker: 24 GiB GPU (0 reserve), 124.90 GiB RAM - 4 GiB reserve (worker_budgets rev 6)
AE = {"gpu_total": 25769803776, "ram_total": 134112837632, "vram_reserve": 0, "ram_reserve": 4 * GIB,
      "gpu_budget": 25769803776, "ram_budget": 129817870336}


def ceiling(left, bpt):
    return _spill._round_down_multiple(left / bpt, 1024)


# ── Anko (models.id 92) ────────────────────────────────────────────────────
ANKO_SIZE = 21391448544
ANKO_GEO = {"n_layers": 40, "n_kv_heads": 4, "head_dim": 128, "ctx_train": 262144}     # 81920 B/token
ANKO_DET = {"is_moe": True, "expert_bytes": 19503513600, "non_expert_bytes": 1876945408,
            "expert_bytes_by_layer": {str(i): 19503513600 // 40 for i in range(40)}}


def test_anko_gpu_only_ceiling_and_moe_explicit():
    moe = compute.moe_split(ANKO_DET, ANKO_GEO, ANKO_SIZE, AE, engine="gguf")
    mem = compute.memory_plan("gguf", {"size_bytes": ANKO_SIZE}, ANKO_GEO, ANKO_DET, AE, moe)
    g = mem["gpu-only"]
    assert g["fits"] and g["ctx_max"]["f16"] == 46080 == ceiling(AE["gpu_budget"] - ANKO_SIZE - COMPUTE, 81920)
    assert g["ctx_max"]["q8_0"] == 88064 and g["ctx_max"]["q4_0"] == 165888
    assert moe["offered"] and moe["kv"]["ctx"] == 262144 and moe["spill"]["n_cpu_moe"] == compute.MOE_ALL_LAYERS
    assert mem["explicit"]["ctx_max"] == {"f16": 262144, "q8_0": 262144, "q4_0": 262144}
    assert mem["explicit"]["weights_gpu"] == 1876945408 and mem["explicit"]["weights_ram"] == 19503513600


# ── Qwen3.6-35B-A3B (models.id 78, transformers) ───────────────────────────
QWEN_SIZE = 71926865825
QWEN_BNB = 21578059747
QWEN_GEO = {"n_layers": 40, "n_kv_layers": 10, "n_kv_heads": 2, "head_dim": 256, "ctx_train": 262144,
            "state_bytes": 32931840, "hybrid": {"full_attention": 10, "linear": 30, "state_bytes": 32931840}}
QWEN_BPT = 2 * 10 * 2 * 256 * 2                                              # 20480 B/token at f16
QWEN_MOE = {"is_moe": True, "expert_bytes": 66035122176, "non_expert_bytes": 5868523232, "expert_count": 256,
            "expert_used_count": 8, "expert_bytes_by_layer": {**{str(i): 1610612736 for i in range(40)}, "mtp0": 1610612736}}
QWEN_V = {"file": None, "dir": "/x/Qwen3.6-35B-A3B", "size_bytes": QWEN_SIZE, "kv_geo": QWEN_GEO, "moe": QWEN_MOE,
          "bnb_4bit": {"eligible": True, "bytes": QWEN_BNB}}


def test_qwen_bf16_never_fits_the_card_and_transformers_offers_no_expert_split():
    out = compute._plan_variant("transformers", QWEN_V, AE, "dir")
    mem = out["memory"]
    assert mem["gpu-only"]["fits"] is False                       # 66.99 GiB on a 24 GiB card
    assert mem["ram-only"]["fits"] is True and mem["ram-only"]["ctx_max"]["f16"] == 262144
    # the MoE structure is a FACT, the expert split is not an offer: the
    # transformers loader has no --n-cpu-moe equivalent (serve the GGUF for that)
    assert out["moe"]["offered"] is False and "transformers loader" in out["moe"]["why"]
    assert out["moe"]["expert_count"] == 256 and out["moe"]["expert_used_count"] == 8
    assert "explicit" not in mem
    # hybrid state is priced once on the KV device, on top of the ctx-linear KV
    assert mem["ram-only"]["kv_bytes"] == QWEN_BPT * 262144 + 32931840


def test_qwen_4bit_fits_with_a_ctx_ceiling_that_100pct_exceeds_at_f16():
    out = compute._plan_variant("transformers", QWEN_V, AE, "dir")
    b4 = out["memory"]["bnb_4bit"]
    g = b4["gpu-only"]
    assert g["fits"] is True and g["weights_gpu"] == QWEN_BNB
    left = AE["gpu_budget"] - QWEN_BNB - COMPUTE_T - 32931840
    assert g["compute_bytes"] == COMPUTE_T > COMPUTE                               # transformers reserve, not llama.cpp's
    assert g["ctx_max"]["f16"] == ceiling(left, QWEN_BPT) == 97280 < 262144       # 100% at f16 EXCEEDS
    assert g["ctx_max"]["q8_0"] == ceiling(left, QWEN_BPT * (34 / 32) / 2) == 184320 < 262144
    assert g["ctx_max"]["q4_0"] == 262144                                           # only q4_0 restores the trained window
    # the priced plan at the ceiling leaves the reserve free: weights + KV + reserve <= budget
    assert QWEN_BNB + 32931840 + QWEN_BPT * g["ctx_max"]["f16"] + COMPUTE_T <= AE["gpu_budget"]
    # 4-bit does not change the loader: still no expert split for transformers
    assert out["moe"]["bnb_4bit"]["offered"] is False
    assert "bnb_moe_explicit" not in (out["auto"] or {})
    assert out["auto"]["bnb_4bit"]["mode"] == "gpu-only"


# ── MN-GRAND 23.5B dense (models.id 94 safetensors, 251 GGUF Q4_K_M) ───────
MN_GEO = {"n_layers": 81, "n_kv_layers": 81, "n_kv_heads": 8, "head_dim": 128, "ctx_train": 1024000}
MN_BPT = 2 * 81 * 8 * 128 * 2                                                # 331776 B/token
MN_ST_SIZE, MN_ST_BNB = 46862725176, 14058817552
MN_GGUF_SIZE = 14354741376


def test_mn_grand_safetensors_dense_4bit_ceiling():
    v = {"file": None, "dir": "/x/MN-GRAND", "size_bytes": MN_ST_SIZE, "kv_geo": MN_GEO, "moe": {"is_moe": False},
         "bnb_4bit": {"eligible": True, "bytes": MN_ST_BNB}}
    out = compute._plan_variant("transformers", v, AE, "dir")
    assert out["moe"]["offered"] is False and "bnb_4bit" in out["moe"] and out["moe"]["bnb_4bit"]["offered"] is False
    assert out["memory"]["gpu-only"]["fits"] is False and out["memory"]["ram-only"]["fits"] is True
    g4 = out["memory"]["bnb_4bit"]["gpu-only"]
    assert g4["fits"] is True
    assert g4["ctx_max"]["f16"] == ceiling(AE["gpu_budget"] - MN_ST_BNB - COMPUTE_T, MN_BPT) == 28672
    assert g4["ctx_max"]["q4_0"] == ceiling(AE["gpu_budget"] - MN_ST_BNB - COMPUTE_T, MN_BPT * (18 / 32) / 2) == 102400
    assert all(x < 1024000 for x in g4["ctx_max"].values())           # trained ctx is never reachable on this card


def test_mn_grand_gguf_q4_k_m_ceiling():
    mem = compute.memory_plan("gguf", {"size_bytes": MN_GGUF_SIZE}, MN_GEO, None, AE, None)
    g = mem["gpu-only"]
    assert g["fits"] and g["ctx"] == g["ctx_max"]["f16"] == ceiling(AE["gpu_budget"] - MN_GGUF_SIZE - COMPUTE, MN_BPT) == 32768
    assert "explicit" not in mem                                       # dense: no MoE split to state
    assert mem["ram-only"]["ctx_max"]["f16"] == min(1024000, ceiling(AE["ram_budget"] - MN_GGUF_SIZE, MN_BPT))


# ── the marker's own derivation, against the real dirs when present ───────
_QWEN_DIR = "/mnt/16T_toshiba/llm_storage/models/transformers/Qwen/Qwen3.6-35B-A3B"
_MN_DIR = "/mnt/16T_toshiba/llm_storage/models/transformers/DavidAU/MN-GRAND-23.5B-Gutenberg-UNCENSORED-V2-GLM4.7-Thinking"


@pytest.mark.skipif(not os.path.isdir(_QWEN_DIR), reason="model dir not mounted")
def test_marker_facts_qwen_from_config_and_headers():
    from hugpy_storage import hugpy_marker as m
    f = m.transformers_dir_facts(_QWEN_DIR)
    assert f["kv_geo"]["n_kv_layers"] == 10 and f["kv_geo"]["state_bytes"] == 32931840
    assert f["kv_cost"]["bytes_per_token"] == QWEN_BPT and f["kv_cost"]["basis"] == "config-formula"
    assert f["is_moe"] and f["expert_count"] == 256 and f["expert_used_count"] == 8
    assert f["expert_bytes"] == 66035122176 and f["non_expert_bytes"] == 5868523232
    assert len(f["expert_bytes_by_layer"]) == 41 and f["expert_bytes_by_layer"]["0"] == 1610612736


@pytest.mark.skipif(not os.path.isdir(_MN_DIR), reason="model dir not mounted")
def test_marker_facts_mn_grand_match_the_gguf_geometry():
    from hugpy_storage import hugpy_marker as m
    f = m.transformers_dir_facts(_MN_DIR)
    assert f["kv_cost"]["bytes_per_token"] == MN_BPT == 331776        # == the GGUF header-derived figure in the DB
    assert not f.get("is_moe")


# ── per-class explicit band (attention / experts on GPU) ───────────────────
def test_anko_explicit_band_caps_experts_by_attention_and_budget():
    moe = compute.moe_split(ANKO_DET, ANKO_GEO, ANKO_SIZE, AE, engine="gguf")
    band = compute.explicit_band(ANKO_DET, ANKO_GEO, ANKO_SIZE, AE, engine="gguf", ctx=moe["kv"]["ctx"])
    L = band["layers"]
    assert L == 40 and band["kv_ctx"] == 262144 and len(band["e_max_by_a"]) == L + 1
    # the band is UNIMODAL in a: on the left e <= a binds (experts can only ride
    # layers whose attention is on the GPU), on the right each extra attention
    # layer brings 512 MiB of KV that displaces an expert layer (465 MiB)
    em = band["e_max_by_a"]
    assert all(0 <= em[i] <= i for i in range(L + 1))
    peak = max(range(L + 1), key=lambda i: em[i])
    assert em[peak] == 23 and 22 <= peak <= 24
    assert all(em[i] == i for i in range(peak)) and all(em[i] >= em[i + 1] for i in range(peak, L))
    # all attention on the GPU with the full 20 GiB KV: 1.75 + 20 + 0.5 GiB leaves ~1.75 GiB -> 3 expert layers
    assert em[L] == 3
    # half the attention on the GPU (10 GiB KV): room for every expert layer of those 20
    assert em[20] == 20
    # RAM (120.9 GiB) never forces attention onto the GPU for a 20 GiB model
    assert band["a_min"] == 0
    # the stored MoE explicit verdict (attention 100%, experts 0%) is inside the band
    assert em[L] >= 0
