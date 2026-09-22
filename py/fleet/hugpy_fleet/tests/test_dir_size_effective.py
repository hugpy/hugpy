"""Dispatch disk detail prices weight_bytes as the artifact that LOADS, not the
multi-quant dir litter (the "53GB weight_bytes" bug).

``_dir_size_detail``/``loaded_disk_detail`` summed EVERY weight-extension file
recursively, so a GGUF repo holding many quantizations reported all of them as
the expected-VRAM proxy. Now, for gguf/llama_cpp, ``weight_bytes`` comes from
``effective_load_requirement`` (the designated/elected quant summed across
shards + mmproj) while ``model_bytes`` keeps its whole-dir on-disk meaning
verbatim (the worker_agent contract). The cache is keyed on (path, chosen
quant), so a ``gguf_file`` override change re-derives instead of serving the
stale figure forever.

Runs both ways:
    venv/bin/python tests/test_dir_size_effective.py
    venv/bin/python -m pytest tests/test_dir_size_effective.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PROJECTS_HOME",
                      tempfile.mkdtemp(prefix="hugpy-dir-size-test-"))
os.environ["SERVE_OVERRIDES_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="hugpy-dir-size-ov-"), "serve_overrides.json")

# Import the agent module FIRST so sys.modules['abstract_hugpy_dev.imports'] is
# the real package (a codebase name-collision otherwise clobbers it).
from hugpy_fleet.worker import agent as A
from hugpy_engine.dispatch import dispatch as D
from hugpy_engine.serve import overrides as OV


def _mkgguf(sizes):
    d = tempfile.mkdtemp()
    for name, sz in sizes.items():
        with open(os.path.join(d, name), "wb") as f:
            f.write(b"\0" * sz)
    return d


SIZES = {
    "m.i1-IQ1_S.gguf": 1000,
    "m.i1-Q2_K.gguf": 2000,
    "m.i1-Q4_K_M.gguf": 4000,   # deterministic auto-rank winner
    "m.i1-Q8_0.gguf": 8000,
    "mmproj-m.gguf": 500,       # projector: loads with the quant
    "README.md": 77,            # non-weight noise: in model_bytes only
}
DIR_TOTAL = sum(SIZES.values())                       # 15577
GGUF_SUM = DIR_TOTAL - SIZES["README.md"]             # every .gguf = old weight sum


def test_gguf_weight_bytes_is_elected_quant_only():
    d = _mkgguf(SIZES)
    cfg = {"framework": "gguf", "model_key": "mk-a"}
    out = D._dir_size_detail(d, model_key="mk-a", cfg=cfg)
    assert out.get("model_bytes") == DIR_TOTAL, out          # whole-dir, unchanged
    # elected q4_k_m + mmproj — NOT the 15.5k multi-quant litter
    assert out.get("weight_bytes") == 4000 + 500, out
    assert out["weight_bytes"] < GGUF_SUM


def test_non_gguf_walk_unchanged():
    d = _mkgguf({"a.safetensors": 3000, "b.safetensors": 3000,
                 "tokenizer.json": 10})
    out = D._dir_size_detail(d)                              # no model identity
    assert out.get("model_bytes") == 6010, out
    assert out.get("weight_bytes") == 6000, out


def test_cache_invalidates_on_gguf_file_override_change():
    d = _mkgguf(SIZES)
    cfg = {"framework": "gguf", "model_key": "mk-b"}
    out1 = D._dir_size_detail(d, model_key="mk-b", cfg=cfg)
    assert out1.get("weight_bytes") == 4000 + 500, out1
    try:
        OV.set_override("mk-b", {"gguf_file": "m.i1-Q8_0.gguf"})
        out2 = D._dir_size_detail(d, model_key="mk-b", cfg=cfg)
        # keyed on (path, chosen): the re-pin misses the cache and re-derives
        assert out2.get("weight_bytes") == 8000 + 500, out2
        assert out2.get("model_bytes") == DIR_TOTAL, out2
    finally:
        OV.set_override("mk-b", {"gguf_file": ""})
    out3 = D._dir_size_detail(d, model_key="mk-b", cfg=cfg)
    assert out3.get("weight_bytes") == 4000 + 500, out3


def test_loaded_disk_detail_threads_model_identity():
    d = _mkgguf(SIZES)
    import hugpy_engine.config.main as CM
    orig_keys = D.loaded_model_keys
    orig_cfg = CM.get_model_config
    import hugpy_storage.model_paths as MP
    orig_route = getattr(MP,
                         "route_destination", None)
    try:
        D.loaded_model_keys = lambda: [("mk-c", "chat")]
        CM.get_model_config = lambda mk, dict_return=False: {
            "framework": "gguf", "model_key": mk}
        MP.route_destination = (
            lambda cfg: d)
        out = D.loaded_disk_detail()
    finally:
        D.loaded_model_keys = orig_keys
        CM.get_model_config = orig_cfg
        if orig_route is not None:
            MP.route_destination = orig_route
    row = out.get("mk-c") or {}
    assert row.get("model_bytes") == DIR_TOTAL, out
    assert row.get("weight_bytes") == 4000 + 500, out


def test_effective_load_requirement_shape():
    d = _mkgguf(SIZES)
    req = OV.effective_load_requirement("mk-d", d, {"framework": "gguf"})
    assert req["weights_bytes"] == 4000 + 500, req
    assert req["gpu_bytes"] == 4000 + 500, req            # dense: gpu == weights
    assert req["cpu_bytes"] == 0, req
    assert req["chosen_gguf"] == "m.i1-Q4_K_M.gguf", req
    assert req["mmap_eligible"] is True, req
    assert "elected" in (req["basis"] or ""), req


def test_effective_load_requirement_non_gguf_duplicate_torch_excluded():
    d = _mkgguf({"model.safetensors": 5000, "pytorch_model.bin": 4800,
                 "config.json": 5})
    req = OV.effective_load_requirement("mk-e", d, {"framework": "transformers"})
    # the .bin twin never loads when a safetensors set shadows it
    assert req["weights_bytes"] == 5000, req
    assert req["gpu_bytes"] == 5000, req


# --------------------------------------------------------------------------- #
# plain-script runner (pytest not required)
# --------------------------------------------------------------------------- #
def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    ok = fail = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[FAIL] {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            ok += 1
            print(f"[ok]   {t.__name__}")
    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(_main())
