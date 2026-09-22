"""Worker fit-guard sizes a multi-quant GGUF by its SINGLE effective quant, not
the whole-dir sum (regression for the 2026-07-14 VRAM-fit over-count).

``agent._incoming_need_bytes`` is GGUF-effective-quant-aware (mirrors central
``model_meta`` via ``gguf_variants_detail``): gguf/llama_cpp size by the single
effective quant (+ its mmproj); other frameworks keep the weight-file sum.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from hugpy_engine.serve.overrides import gguf_variants_detail
from hugpy_fleet.worker import agent as A

SIZES = {
    "m.i1-IQ1_S.gguf": 1000,
    "m.i1-Q2_K.gguf": 2000,
    "m.i1-Q4_K_M.gguf": 4000,   # deterministic auto-rank winner
    "m.i1-Q8_0.gguf": 8000,
    "mmproj-m.gguf": 500,       # projector: part of effective size, not a variant
}


def _mkgguf(sizes):
    d = tempfile.mkdtemp()
    for name, sz in sizes.items():
        with open(os.path.join(d, name), "wb") as f:
            f.write(b"\0" * sz)
    return d


@pytest.fixture
def patch_cfg(monkeypatch):
    """Point the agent's config/route resolution at a fixture dir."""
    import hugpy_engine.config.main as CM
    import hugpy_storage.model_paths as MP

    def _apply(framework, dest):
        monkeypatch.setattr(CM, "get_model_config",
                            lambda mk, dict_return=False, **kw: {"framework": framework, "model_key": mk})
        monkeypatch.setattr(MP, "route_destination", lambda cfg, *a, **kw: dest)
        A._MODEL_DIR_CACHE.clear() if hasattr(A, "_MODEL_DIR_CACHE") else None
    return _apply


def test_effective_quant_is_single_file_plus_mmproj():
    d = _mkgguf(SIZES)
    g = gguf_variants_detail("x", d, {"framework": "gguf"}) or {}
    eff = g.get("effective_gguf")
    assert eff in SIZES and eff != "mmproj-m.gguf"
    assert g.get("effective_quant_bytes") == SIZES[eff]
    assert g.get("effective_bytes") == SIZES[eff] + SIZES["mmproj-m.gguf"]
    assert g.get("effective_bytes") < sum(SIZES.values())
    assert len(g.get("variants") or []) == 4
    assert sum(v["bytes"] for v in g["variants"]) == sum(SIZES.values()) - SIZES["mmproj-m.gguf"]


def test_gguf_need_is_effective_quant_times_headroom(patch_cfg):
    d = _mkgguf(SIZES)
    g = gguf_variants_detail("x", d, {"framework": "gguf"}) or {}
    patch_cfg("gguf", d)
    need = A._incoming_need_bytes("dan-multi")
    assert need == int(g["effective_bytes"] * 1.15)
    assert need < int(sum(SIZES.values()) * 1.15)


def test_non_gguf_need_is_weight_sum(patch_cfg):
    d = _mkgguf({"a.safetensors": 3000, "b.safetensors": 3000, "tokenizer.json": 10})
    patch_cfg("transformers", d)
    assert A._incoming_need_bytes("some-transformers") == int(6000 * 1.15)


def test_empty_gguf_dir_fails_open(patch_cfg):
    patch_cfg("gguf", tempfile.mkdtemp())
    assert A._incoming_need_bytes("empty") is None
