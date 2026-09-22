"""Projector-aware GGUF spill fit — the honest partial-offload split for a 7B
vision GGUF on a small (8 GB) card.

Background (2026-07-14, computron / RTX 4060 Laptop 8 GB): a Qwen2.5-VL-7B GGUF
is a PAIR — the language-model quant (~5-9 GB) PLUS a separate mmproj / CLIP
projector (~1.35 GB) that llama.cpp loads onto the GPU too. The autofit split
budgeted only the model file, so on an 8 GB card it either planned "all layers"
(then OOMed when the projector landed on top) or the native vision server used a
hardcoded ``-ngl 999`` (all layers) and never became healthy — so the model fell
through to a path that could not serve it and showed 0 GPU offload ("not spilling
at all"). The fix reserves the projector's VRAM BEFORE fitting language-model
layers, so a 7B VL GGUF gets an honest N-on-GPU / rest-in-RAM split.

Under test (pure math, no GPU, no engine, no network):
  * spill.vision_projector_bytes  — locates the mmproj sidecar, skips the main
    file, 0 for a text-only model / missing dir.
  * spill.autofit_gpu_layers(extra_reserve_bytes=…) — the reserve is subtracted
    from the VRAM budget; it can flip an "all fits" (-1) into a partial split and
    a partial split down to 0; extra_reserve_bytes=0 is byte-identical to before.
  * the realistic computron scenario — unsloth Qwen2.5-VL-7B UD-Q6_K_XL
    (6.96 GB, 28 layers) + 1.35 GB projector on ~6.75 GB budgetable VRAM yields a
    small POSITIVE partial split (not -1, not 0), and the projector reserve makes
    it fit fewer layers than the naive (projector-blind) fit.
  * shard_server._vision_ngl_arg — env override mapping (max/999/-1 -> all;
    explicit int passthrough; auto/unset -> projector-aware autofit).

Runs both ways:
    cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
    venv/bin/python -m pytest tests/test_spill_vision_fit.py -q
    venv/bin/python tests/test_spill_vision_fit.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_engine import spill

GIB = 2 ** 30


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sparse(path: str, nbytes: int) -> str:
    """Create a sparse file of exactly ``nbytes`` (0 real disk) — os.path.getsize
    reports the full length, which is all the fit math reads."""
    with open(path, "wb") as fh:
        if nbytes:
            fh.truncate(nbytes)
    return path


class _patched:
    """Save/restore module globals so each test is independent whether run under
    pytest (no fixtures needed) or as a plain script."""

    def __init__(self, **kw):
        self.kw = kw
        self.saved = {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.saved[k] = getattr(spill, k)
            setattr(spill, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(spill, k, v)


def _fixed_ctx_reserve(gib: str = "1.0"):
    """Pin HUGPY_VRAM_CTX_RESERVE_GIB to a KNOWN value for deterministic budgets.
    NOTE: a value of exactly 0 can't be expressed via env (0.0 or 2.5 -> 2.5), so
    tests that want ~no ctx reserve use a tiny positive fraction."""
    os.environ["HUGPY_VRAM_CTX_RESERVE_GIB"] = gib


# --------------------------------------------------------------------------- #
# vision_projector_bytes
# --------------------------------------------------------------------------- #
def test_projector_bytes_finds_sidecar():
    d = tempfile.mkdtemp()
    main = _sparse(os.path.join(d, "model-Q6.gguf"), 4096)
    _sparse(os.path.join(d, "mmproj-F16.gguf"), 1234)
    assert spill.vision_projector_bytes(main) == 1234
    # passing the directory works too
    assert spill.vision_projector_bytes(d) == 1234


def test_projector_bytes_skips_main_and_text_only():
    d = tempfile.mkdtemp()
    main = _sparse(os.path.join(d, "model-Q6.gguf"), 4096)
    # text-only model dir: no mmproj -> 0 (byte-identical text behaviour)
    assert spill.vision_projector_bytes(main) == 0
    # a main file that itself matches nothing, decoy non-gguf mmproj ignored
    _sparse(os.path.join(d, "mmproj-notes.txt"), 999)
    assert spill.vision_projector_bytes(main) == 0


def test_projector_bytes_missing_dir_is_zero():
    assert spill.vision_projector_bytes("/no/such/dir/model.gguf") == 0


# --------------------------------------------------------------------------- #
# autofit_gpu_layers: reserve arithmetic
# --------------------------------------------------------------------------- #
def test_reserve_zero_is_backward_compatible():
    _fixed_ctx_reserve("1.0")
    d = tempfile.mkdtemp()
    main = _sparse(os.path.join(d, "m.gguf"), 7 * GIB)   # 7 GiB, 28 layers -> 0.25/layer
    # NOTE the arithmetic below has no ``* 0.85``: _VRAM_SAFETY was lifted to
    # 1.0 (2026-07-25) because it double-counted the EXPLICIT ctx reserve —
    # measured 2.59 GiB of real KV+compute against a 2.5 GiB reserve, so the
    # multiplier was a second margin for a risk already covered. free_vram here
    # dropped 8 -> 6.75 GiB so the case still exercises PARTIAL fit; at 8 GiB
    # the model now (correctly) fits whole.
    with _patched(_gguf_layer_count=lambda p: 28):
        # budget = 10 - 1.0 = 9.0 GiB >= 7 GiB -> all fit
        assert spill.autofit_gpu_layers(main, free_vram=10 * GIB) == -1
        assert spill.autofit_gpu_layers(main, free_vram=10 * GIB, extra_reserve_bytes=0) == -1
        # partial: budget = 6.75 - 1.0 = 5.75 GiB ; 5.75/0.25 = 23
        assert spill.autofit_gpu_layers(main, free_vram=int(6.75 * GIB)) == 23


def test_reserve_flips_all_fit_to_partial():
    _fixed_ctx_reserve("1.0")
    d = tempfile.mkdtemp()
    main = _sparse(os.path.join(d, "m.gguf"), 7 * GIB)
    with _patched(_gguf_layer_count=lambda p: 28):
        # no projector -> all fit (-1).  (_VRAM_SAFETY is 1.0 since 2026-07-25,
        # so the budget is free_vram minus the reserves, with no multiplier.)
        assert spill.autofit_gpu_layers(main, free_vram=8 * GIB) == -1
        # a 1.35 GiB projector reserve: budget = 8 - 1.0 - 1.35 = 5.65 GiB < 7
        # -> partial; 5.65/0.25 = 22
        n = spill.autofit_gpu_layers(main, free_vram=8 * GIB,
                                     extra_reserve_bytes=int(1.35 * GIB))
        assert n == 22, n
        assert n != -1                       # the whole point: no longer "all"


def test_reserve_can_drive_to_zero():
    _fixed_ctx_reserve("1.0")
    d = tempfile.mkdtemp()
    main = _sparse(os.path.join(d, "m.gguf"), 7 * GIB)
    with _patched(_gguf_layer_count=lambda p: 28):
        # reserve larger than the whole budget -> 0 GPU layers (pure CPU/RAM)
        assert spill.autofit_gpu_layers(main, free_vram=10 * GIB,
                                        extra_reserve_bytes=9 * GIB) == 0


# --------------------------------------------------------------------------- #
# realistic computron scenario
# --------------------------------------------------------------------------- #
def test_computron_7b_vl_gets_honest_partial_split():
    """RTX 4060 Laptop 8 GB: unsloth Qwen2.5-VL-7B UD-Q6_K_XL (6.96 GB, 28
    layers) + 1.35 GB projector, ~6.75 GB budgetable VRAM, default 2.5 GB ctx
    reserve -> a SMALL positive partial split, and the projector reserve fits
    STRICTLY FEWER layers than a projector-blind fit (the OOM the fix prevents)."""
    os.environ.pop("HUGPY_VRAM_CTX_RESERVE_GIB", None)   # default 2.5 GiB
    d = tempfile.mkdtemp()
    main = _sparse(os.path.join(d, "Qwen2.5-VL-7B-Instruct-UD-Q6_K_XL.gguf"),
                   6_959_041_408)
    _sparse(os.path.join(d, "mmproj-F16.gguf"), 1_354_163_040)
    reserve = spill.vision_projector_bytes(main)
    assert reserve == 1_354_163_040, reserve
    fv = int(6.75 * GIB)
    with _patched(_gguf_layer_count=lambda p: 28):
        naive = spill.autofit_gpu_layers(main, free_vram=fv, extra_reserve_bytes=0)
        honest = spill.autofit_gpu_layers(main, free_vram=fv, extra_reserve_bytes=reserve)
    # honest split must be a genuine PARTIAL offload: some layers, not all, not none
    assert 0 < honest < 28, honest
    # reserving the projector's VRAM fits fewer layers than ignoring it
    assert honest < naive, (honest, naive)


# --------------------------------------------------------------------------- #
# shard_server._vision_ngl_arg env mapping
# --------------------------------------------------------------------------- #
def test_vision_ngl_arg_env_overrides():
    from hugpy_engine.llama.runners.src import shard_server as ss
    d = tempfile.mkdtemp()
    main = _sparse(os.path.join(d, "m.gguf"), 6 * GIB)
    mm = _sparse(os.path.join(d, "mmproj-F16.gguf"), int(1.3 * GIB))
    try:
        for val, want in (("max", "999"), ("all", "999"), ("999", "999"),
                          ("-1", "999"), ("10", "10"), ("0", "0")):
            os.environ["HUGPY_VISION_NGL"] = val
            assert ss._vision_ngl_arg(main, mm) == want, (val, want)
        # auto -> projector-aware autofit; pin free VRAM + layers for determinism
        os.environ["HUGPY_VISION_NGL"] = "auto"
        os.environ["HUGPY_VRAM_CTX_RESERVE_GIB"] = "1.0"
        with _patched(free_vram_bytes=lambda: 10 * GIB, _gguf_layer_count=lambda p: 28):
            got = ss._vision_ngl_arg(main, mm)
        # budget = 8.5 - 1.0 - 1.3 = 6.2 GiB ; per-layer 6/28=0.214 -> 28 (all)?
        # 6.2/0.214 = 28.9 capped to 28 -> but 6.2 < 6 is false (6.2>=6) => all fit "999"
        assert got == "999", got
        # tighten VRAM so it must partial-split
        with _patched(free_vram_bytes=lambda: int(4.5 * GIB), _gguf_layer_count=lambda p: 28):
            got2 = ss._vision_ngl_arg(main, mm)
        assert got2.isdigit() and 0 < int(got2) < 28, got2
    finally:
        os.environ.pop("HUGPY_VISION_NGL", None)
        os.environ.pop("HUGPY_VRAM_CTX_RESERVE_GIB", None)


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
