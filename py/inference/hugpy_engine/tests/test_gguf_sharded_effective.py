"""Sharded-GGUF effective-size resolution (t33 regression).

A split/sharded GGUF ships as N files ``<stem>-00001-of-0000N.gguf`` … that are
ONE logical model, usually nested in a per-quant subdir. The old
``gguf_variants_detail`` listed servable ggufs with a SHALLOW ``os.listdir`` and
never grouped shards, so a sharded model resolved to NO servable variant (nested
shards were invisible) — ``effective_bytes`` came back ``None`` and the model
showed a wrong / missing "on disk" size (the Qwen3-Coder-Next case: a ~48GB
sharded coder read as no effective size at all, while the sibling single-file
unsloth repo reported correctly).

This suite locks the shard-aware behavior: shards are enumerated recursively,
folded into ONE variant whose bytes SUM the shards, with the ``-00001`` shard as
the entrypoint — while single-file quants, the vision quant+mmproj PAIR, and
non-GGUF dirs keep their existing contracts.

Runs under pytest:
    venv/bin/python -m pytest tests/test_gguf_sharded_effective.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Force the real ``abstract_hugpy_dev.imports`` package to load (a codebase
# name-collision can otherwise shadow it, breaking get_gguf_file resolution used
# for the effective-variant pick on multi-variant fixtures).
from hugpy_engine.config.main import get_gguf_file
from hugpy_engine.serve.overrides import (
    gguf_variants_detail,
    _gguf_variant_groups,
    _servable_gguf_files,
)


def _mk(files: dict) -> str:
    """Build a tmp model dir. Keys are relpaths (may include a subdir); values
    are sizes in bytes."""
    d = tempfile.mkdtemp()
    for rel, sz in files.items():
        full = os.path.join(d, rel)
        os.makedirs(os.path.dirname(full) or d, exist_ok=True)
        with open(full, "wb") as f:
            f.write(b"\0" * sz)
    return d


# ── one sharded family nested in a subdir (the real Qwen3-Coder-Next layout) ──
SHARDED_NESTED = {
    "Q4_K_M/model-Q4_K_M-00001-of-00003.gguf": 100,
    "Q4_K_M/model-Q4_K_M-00002-of-00003.gguf": 200,
    "Q4_K_M/model-Q4_K_M-00003-of-00003.gguf": 50,
}
SHARD_SUM = 350


def test_nested_shards_fold_into_one_variant_summed():
    d = _mk(SHARDED_NESTED)
    g = gguf_variants_detail("x", d, {"framework": "gguf"}) or {}
    # a shallow listdir would see ZERO ggufs here -> the old {} bug
    assert g, "sharded model must resolve a variant (was {} pre-fix)"
    assert len(g["variants"]) == 1, "N shards are ONE variant, not N"
    v = g["variants"][0]
    assert v["is_effective"] is True
    assert v["filename"] == "model-Q4_K_M-00001-of-00003.gguf"  # -00001 entrypoint
    assert v["bytes"] == SHARD_SUM                              # SUMMED, not shard-1
    assert g["effective_quant_bytes"] == SHARD_SUM
    assert g["effective_bytes"] == SHARD_SUM                    # mmproj 0 here
    assert g["effective_gguf"] == "model-Q4_K_M-00001-of-00003.gguf"


def test_effective_is_the_sum_not_the_first_shard():
    # The subtle half of the bug: even with recursion, without grouping the
    # effective size would be shard-1 alone (100), not the whole model (350).
    d = _mk(SHARDED_NESTED)
    g = gguf_variants_detail("x", d, {"framework": "gguf"}) or {}
    assert g["effective_bytes"] == SHARD_SUM
    assert g["effective_bytes"] != 100          # not the -00001 shard alone


def test_sharded_plus_mmproj_folds_projector_into_effective():
    files = dict(SHARDED_NESTED)
    files["mmproj-model.gguf"] = 30             # projector beside the model
    d = _mk(files)
    g = gguf_variants_detail("x", d, {"framework": "gguf"}) or {}
    assert g["mmproj_bytes"] == 30
    assert len(g["variants"]) == 1             # projector is NOT a servable variant
    assert g["effective_quant_bytes"] == SHARD_SUM
    assert g["effective_bytes"] == SHARD_SUM + 30


def test_two_sharded_quant_families_stay_separate_one_effective():
    files = {
        "Q4_K_M/m-Q4_K_M-00001-of-00002.gguf": 100,
        "Q4_K_M/m-Q4_K_M-00002-of-00002.gguf": 100,   # family total 200
        "Q8_0/m-Q8_0-00001-of-00002.gguf": 400,
        "Q8_0/m-Q8_0-00002-of-00002.gguf": 400,        # family total 800
    }
    d = _mk(files)
    g = gguf_variants_detail("x", d, {"framework": "gguf"}) or {}
    assert len(g["variants"]) == 2
    assert sum(1 for v in g["variants"] if v["is_effective"]) == 1
    by_name = {v["filename"]: v for v in g["variants"]}
    assert by_name["m-Q4_K_M-00001-of-00002.gguf"]["bytes"] == 200
    assert by_name["m-Q8_0-00001-of-00002.gguf"]["bytes"] == 800
    # deterministic auto-rank prefers q4_k_m
    eff = next(v for v in g["variants"] if v["is_effective"])
    assert eff["filename"] == "m-Q4_K_M-00001-of-00002.gguf"
    assert g["effective_bytes"] == 200


def test_sharded_family_mixed_with_single_file_quant():
    files = {
        "m-Q4_K_M-00001-of-00002.gguf": 100,       # flat shard set (family 200)
        "m-Q4_K_M-00002-of-00002.gguf": 100,
        "m-Q8_0.gguf": 500,                        # a single-file quant
    }
    d = _mk(files)
    g = gguf_variants_detail("x", d, {"framework": "gguf"}) or {}
    assert len(g["variants"]) == 2                 # one shard group + one single
    by_name = {v["filename"]: v for v in g["variants"]}
    assert by_name["m-Q4_K_M-00001-of-00002.gguf"]["bytes"] == 200
    assert by_name["m-Q8_0.gguf"]["bytes"] == 500
    eff = next(v for v in g["variants"] if v["is_effective"])
    assert eff["filename"] == "m-Q4_K_M-00001-of-00002.gguf"
    assert g["effective_bytes"] == 200


# ── existing contracts preserved ────────────────────────────────────────────
def test_single_file_multiquant_unchanged():
    # Several single-file quants: ONE effective (auto-rank q4_k_m), summed==dir
    files = {"m-Q2_K.gguf": 200, "m-Q4_K_M.gguf": 400, "m-Q8_0.gguf": 800}
    d = _mk(files)
    g = gguf_variants_detail("x", d, {"framework": "gguf"}) or {}
    assert len(g["variants"]) == 3
    eff = next(v for v in g["variants"] if v["is_effective"])
    assert eff["filename"] == "m-Q4_K_M.gguf"
    assert g["effective_bytes"] == 400             # the single quant, not the sum
    assert sum(v["bytes"] for v in g["variants"]) == 1400


def test_non_gguf_dir_returns_empty():
    d = _mk({"model.safetensors": 100, "config.json": 10})
    assert gguf_variants_detail("x", d, {"framework": "transformers"}) == {}
    assert _servable_gguf_files(d) == []


def test_group_helper_directly():
    # unit-level: the grouping helper folds shards, keeps singles, sums bytes
    files = [
        ("Q4_K_M/a-Q4_K_M-00001-of-00002.gguf", 10),
        ("Q4_K_M/a-Q4_K_M-00002-of-00002.gguf", 20),
        ("b-Q8_0.gguf", 99),
    ]
    variants = _gguf_variant_groups(files)
    by_name = {v["filename"]: v for v in variants}
    assert by_name["a-Q4_K_M-00001-of-00002.gguf"]["bytes"] == 30
    assert len(by_name["a-Q4_K_M-00001-of-00002.gguf"]["members"]) == 2
    assert by_name["b-Q8_0.gguf"]["bytes"] == 99
    assert by_name["b-Q8_0.gguf"]["members"] == ["b-Q8_0.gguf"]
