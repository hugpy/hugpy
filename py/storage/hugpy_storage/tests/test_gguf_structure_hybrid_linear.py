"""GUARD (2026-10-02): a Qwen3.5/3.6 hybrid's Gated-DeltaNet blocks carry a
fused attn_qkv beside their ssm_* tensors; they must classify as LINEAR (fixed
state, no KV), not full attention. Anko priced 20 GiB of KV at 262,144 instead
of 5 GiB because every block read as full."""
from hugpy_storage import gguf_inspect as gi


def _tensor(name):
    return {"name": name, "nbytes": 1024}


def test_linear_blocks_with_fused_qkv_are_linear(monkeypatch, tmp_path):
    f = tmp_path / "hybrid.gguf"
    f.write_bytes(b"x" * 64)
    linear = ["attn_qkv.weight", "attn_gate.weight", "ssm_a", "ssm_conv1d.weight", "ssm_out.weight"]
    full = ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"]
    tensors = []
    for i in range(8):
        for n in (full if i % 4 == 3 else linear):
            tensors.append(_tensor(f"blk.{i}.{n}"))
    header = {"file_size": 64, "tensors": tensors, "kv": {
        "general.architecture": "qwen35moe", "qwen35moe.block_count": 8,
        "qwen35moe.attention.head_count": 16, "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.key_length": 256, "qwen35moe.ssm.inner_size": 4096,
        "qwen35moe.ssm.state_size": 128, "qwen35moe.ssm.conv_kernel": 4,
        "qwen35moe.ssm.group_count": 16}}
    monkeypatch.setattr(gi, "gguf_read_header", lambda *a, **k: header)
    monkeypatch.setattr(gi, "_gguf_shard_paths", lambda p: [p])
    gi._STRUCT_CACHE.clear()
    st = gi.gguf_structure(str(f))
    kinds = [st["layers"][i]["attn"] for i in range(8)]
    assert kinds == ["linear", "linear", "linear", "full"] * 2
    assert all(st["layers"][i]["kv_elems_per_token"] == 0 for i in range(8) if i % 4 != 3)
    assert st["layers"][3]["kv_elems_per_token"] == 2 * (256 + 256)
    assert st["layers"][0]["state_bytes"] > 0
