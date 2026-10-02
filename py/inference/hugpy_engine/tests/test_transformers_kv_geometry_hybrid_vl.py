"""GUARD (2026-10-02): the engine's transformers KV geometry reads the nested
text_config of a VL wrapper and counts only the KV-bearing layers of a hybrid,
the same rule as the DB facts. Surogate-3.5-2B was refused at "KV 64.0 GB at
ctx 262,144" (geometry-less heuristic) while its facts priced 3.0 GiB."""
from hugpy_engine import spill

VL_HYBRID = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "text_config": {
        "model_type": "qwen3_5_text", "num_hidden_layers": 24, "num_attention_heads": 8,
        "num_key_value_heads": 2, "head_dim": 256, "hidden_size": 2048,
        "max_position_embeddings": 262144, "full_attention_interval": 4,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 6,
    },
}


def test_vl_wrapper_is_read_and_only_full_attention_layers_hold_kv():
    geo = spill._transformers_kv_geometry(VL_HYBRID)
    assert geo["n_layers"] == 24 and geo["n_kv_layers"] == 6
    assert geo["n_kv_heads"] == 2 and geo["head_dim"] == 256 and geo["ctx_train"] == 262144
    kv = spill.kv_bytes_for_geo(geo, 262144)
    assert kv == 2 * 6 * 2 * 256 * 2 * 262144          # 3.0 GiB, not the 64 GB heuristic


def test_dense_flat_config_unchanged():
    geo = spill._transformers_kv_geometry({"num_hidden_layers": 32, "num_attention_heads": 32,
                                           "num_key_value_heads": 8, "hidden_size": 4096,
                                           "max_position_embeddings": 8192})
    assert geo["n_layers"] == 32 and geo["n_kv_layers"] == 32 and geo["head_dim"] == 128
