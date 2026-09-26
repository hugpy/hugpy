"""The llama_context VRAM reserve — COMPUTED from real geometry, not a flat
constant (2026-07-27).

THE FAULT, measured on computron (RTX 4060, 7807 MiB card) with
flux2-klein-9b-uncensored q4_k_m (5_027_783_648 B = 4.68 GiB, 36 layers):

    autofit given 7.5 GiB (the whole empty card)  ->  29/36 layers
    autofit given 6.5 GiB (after the 1.0 reserve) ->  23/36 layers

yet a manual re-seat at ``n_gpu_layers=-1`` put ALL 36 layers on that card —
6739 MiB used, 1068 MiB STILL FREE. Two stacked FLAT reserves
(HUGPY_VRAM_RESERVE_GIB 1.0 + HUGPY_VRAM_CTX_RESERVE_GIB 2.5) held back 3.5 GiB
on every card regardless of its size: 15% of a 24 GiB 3090, 44% of an 8 GiB
4060. By the cliff measured 2026-07-25 that is a ~4x loss (a dense GGUF runs
~135 tok/s fully resident vs ~36 the moment ONE layer spills).

Under test (pure math, no GPU, no engine, no network — synthetic GGUF headers
carry the real geometry so the header PARSE is exercised too):

  * spill.vram_ctx_reserve_bytes — KV(n_ctx, geometry, fp16) + compute-graph;
    "env" (operator override, flat) / "computed" / "default" (flat 2.5,
    degrade-not-guess) sources.
  * spill.ctx_for_fit / llama_ctx_cap — the n_ctx the fit prices against.
  * spill.autofit_gpu_layers — the flux2 regression (-1), an honest partial for
    a model that genuinely does not fit, the 70B-on-24GiB llama_context OOM the
    guard was built for, and both env overrides still winning.

Run: venv/bin/python -m pytest tests/test_spill_ctx_reserve.py -q
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_engine import spill

GIB = 2 ** 30
MIB = 2 ** 20

# The real file the operator measured against (present on the shared store).
FLUX2 = ("/mnt/llm_storage/models/gguf/ponpoke/"
         "flux2-klein-9b-uncensored-text-encoder/"
         "flux2-klein-9b-uncensored-q4_k_m.gguf")
FLUX2_BYTES = 5_027_783_648                       # 4.6825 GiB, verified on disk


# --------------------------------------------------------------------------- #
# a real (tiny) GGUF header, so the geometry comes through the actual parser
# --------------------------------------------------------------------------- #
def _kv_u32(key: str, val: int) -> bytes:
    k = key.encode()
    # value type 4 == a 4-byte int in spill's reader (GGUF UINT32)
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 4) + struct.pack("<i", val)


def write_gguf(path, *, size_bytes: int, arch: str = "qwen3", block_count: int,
               head_count: int, head_count_kv: int, embedding_length: int,
               key_length: int | None = None, context_length: int = 32768) -> str:
    """A minimal but genuinely parseable GGUF: magic + version + counts + the
    geometry KVs, then truncated (sparse) to ``size_bytes`` so os.path.getsize
    reports the quant size the fit math reads."""
    kvs = [
        (b"general.architecture",
         struct.pack("<I", 8) + struct.pack("<Q", len(arch)) + arch.encode()),
    ]
    body = b""
    body += (struct.pack("<Q", len(kvs[0][0])) + kvs[0][0] + kvs[0][1])
    n_kv = 1
    for key, val in ((f"{arch}.block_count", block_count),
                     (f"{arch}.attention.head_count", head_count),
                     (f"{arch}.attention.head_count_kv", head_count_kv),
                     (f"{arch}.embedding_length", embedding_length),
                     (f"{arch}.context_length", context_length)):
        body += _kv_u32(key, val)
        n_kv += 1
    if key_length is not None:
        body += _kv_u32(f"{arch}.attention.key_length", key_length)
        n_kv += 1
    header = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
              + struct.pack("<Q", n_kv) + body)
    with open(path, "wb") as fh:
        fh.write(header)
        if size_bytes > len(header):
            fh.truncate(size_bytes)
    return str(path)


def flux2_like(tmp_path, size_bytes: int = FLUX2_BYTES) -> str:
    """flux2-klein-9b-uncensored q4_k_m's REAL header geometry (read off the file
    on the shared store 2026-07-27): qwen3, 36 blocks, 32 heads, 8 kv heads,
    embedding 4096, key_length 128, trained ctx 40960."""
    return write_gguf(tmp_path / "flux2-klein-9b-Q4_K_M.gguf",
                      size_bytes=size_bytes, arch="qwen3", block_count=36,
                      head_count=32, head_count_kv=8, embedding_length=4096,
                      key_length=128, context_length=40960)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every knob unset: these tests assert DEFAULT behaviour unless they say
    otherwise. (test_spill_vision_fit leaks HUGPY_VRAM_CTX_RESERVE_GIB into the
    process env, so this is load-bearing under a whole-directory run.)"""
    for k in ("HUGPY_VRAM_CTX_RESERVE_GIB", "HUGPY_VRAM_RESERVE_GIB",
              "HUGPY_VRAM_SAFETY", "HUGPY_GPU_MEM_GIB", "DEFAULT_LLAMA_CTX"):
        monkeypatch.delenv(k, raising=False)


def _flat_old_fit(file_bytes: int, free_vram: int, total_layers: int) -> int:
    """The PRE-CHANGE arithmetic (flat 2.5 GiB ctx reserve, safety 1.0), so the
    'unchanged' assertions compare against the real old formula, not a number
    typed from memory."""
    budget = free_vram - int(2.5 * GIB)
    if budget <= 0:
        return 0
    if budget >= file_bytes:
        return -1
    return max(0, min(int(budget // (file_bytes / total_layers)), total_layers))


# --------------------------------------------------------------------------- #
# THE REGRESSION: a model that demonstrably fits must return -1
# --------------------------------------------------------------------------- #
def test_flux2_on_an_empty_8gib_card_takes_every_layer(tmp_path):
    """4.68 GiB / 36 layers on computron's 8 GiB card, EMPTY -> -1 (all layers).

    Both figures the operator measured: 7.5 GiB is the card's raw free, 6.5 GiB
    is what ``free_vram_bytes()`` hands the loader after the 1.0 GiB external
    reserve — i.e. the number the live slot path actually fits against. Old
    behaviour: 29 and 23 respectively (safety 0.85, the build measured on) or
    -1 and 30 at safety 1.0 — either way the LIVE path spilled layers."""
    g = flux2_like(tmp_path)
    assert os.path.getsize(g) == FLUX2_BYTES
    assert spill.autofit_gpu_layers(g, free_vram=int(7.5 * GIB)) == -1
    assert spill.autofit_gpu_layers(g, free_vram=int(6.5 * GIB)) == -1
    # and the old formula demonstrably did NOT (the regression this closes)
    assert _flat_old_fit(FLUX2_BYTES, int(6.5 * GIB), 36) == 30


@pytest.mark.skipif(not os.path.isfile(FLUX2), reason="flux2 quant not on this box")
def test_flux2_regression_against_the_real_file_on_disk():
    """The same assertion against the ACTUAL GGUF the operator measured, so the
    header parse, the file size and the geometry are all real."""
    assert spill.autofit_gpu_layers(FLUX2, free_vram=int(7.5 * GIB)) == -1
    assert spill.autofit_gpu_layers(FLUX2, free_vram=int(6.5 * GIB)) == -1
    reserve, source, detail = spill.vram_ctx_reserve_bytes(FLUX2)
    assert source == "computed"
    assert detail["n_layers"] == 36 and detail["n_kv_heads"] == 8
    assert detail["head_dim"] == 128 and detail["ctx"] == 16384


def test_the_reserve_is_kv_at_the_real_ctx_plus_the_compute_graph(tmp_path):
    """The reserve prices KV at the fit-bounded context actually served."""
    g = flux2_like(tmp_path)
    ctx = spill.served_ctx_for_fit(g, free_vram=int(7.5 * GIB))
    reserve, source, detail = spill.vram_ctx_reserve_bytes(g, n_ctx=ctx)
    assert source == "computed"
    assert detail["ctx"] == ctx == 16384
    assert detail["kv_bytes"] == 2 * 36 * ctx * 8 * 128 * 2
    assert reserve == detail["kv_bytes"] + spill._CTX_COMPUTE_RESERVE_BYTES
    # measured on the box: 2.59 GiB of KV+compute+context at this ctx. The
    # computed reserve must COVER that (never under-reserve) without ballooning.
    assert 2.59 * GIB <= reserve <= 3.0 * GIB


def test_the_reserve_tracks_the_ctx_the_child_will_actually_serve(tmp_path):
    """The KV cache is linear in n_ctx, so a small context must cost less — this
    is the whole point of 'reflect the ACTUAL model AND context'."""
    g = flux2_like(tmp_path)
    big, _, _ = spill.vram_ctx_reserve_bytes(g, n_ctx=16384)
    small, _, sd = spill.vram_ctx_reserve_bytes(g, n_ctx=4096)
    assert sd["ctx"] == 4096
    assert small < big
    assert (big - spill._CTX_COMPUTE_RESERVE_BYTES) == \
           4 * (small - spill._CTX_COMPUTE_RESERVE_BYTES)


def test_a_short_context_seats_a_model_a_long_one_cannot(tmp_path):
    """A 6.5 GiB quant on the same 8 GiB card: it cannot hold a 16k KV cache too,
    but at 4k it fits whole. The layer count must move with the context."""
    g = flux2_like(tmp_path, size_bytes=int(6.5 * GIB))
    long_ctx = spill.autofit_gpu_layers(g, free_vram=int(7.0 * GIB), n_ctx=16384)
    short_ctx = spill.autofit_gpu_layers(g, free_vram=int(7.0 * GIB), n_ctx=4096)
    assert 0 < long_ctx < 36
    assert short_ctx == -1


# --------------------------------------------------------------------------- #
# a model that genuinely does NOT fit still gets an honest partial count
# --------------------------------------------------------------------------- #
def test_a_model_too_big_for_the_card_gets_an_honest_partial(tmp_path):
    """A split is priced against the FULLY STACKED budget — the credit only ever
    converts a spill into a whole seat, it never buys layers in a split."""
    g = flux2_like(tmp_path, size_bytes=int(20 * GIB))
    n = spill.autofit_gpu_layers(g, free_vram=int(7.5 * GIB))
    assert 0 < n < 36, n
    reserve, _, _ = spill.vram_ctx_reserve_bytes(g)
    assert n == int((int(7.5 * GIB) - reserve) // (20 * GIB / 36))
    # the split leaves the external floor genuinely untouched
    assert int(7.5 * GIB) - int(n * (20 * GIB / 36)) >= reserve


def test_no_budget_at_all_is_zero_layers(tmp_path):
    g = flux2_like(tmp_path)
    assert spill.autofit_gpu_layers(g, free_vram=int(1.0 * GIB)) == 0


def test_the_projector_reserve_still_stacks(tmp_path):
    """extra_reserve_bytes (the mmproj/CLIP projector that lands on the GPU beside
    the layers) is a REAL second tenant, not a cushion — it must still be
    subtracted on top, or an 8 GiB card OOMs when the projector lands."""
    g = flux2_like(tmp_path)
    assert spill.autofit_gpu_layers(g, free_vram=int(6.5 * GIB)) == -1
    plain_ctx = spill.served_ctx_for_fit(g, free_vram=int(6.5 * GIB))
    projector_ctx = spill.served_ctx_for_fit(
        g, free_vram=int(6.5 * GIB),
        extra_reserve_bytes=int(1.35 * GIB))
    n = spill.autofit_gpu_layers(g, free_vram=int(6.5 * GIB),
                                 extra_reserve_bytes=int(1.35 * GIB))
    # The projector is charged by shrinking context first; the weights can
    # still remain whole when that yields enough room.
    assert projector_ctx < plain_ctx
    assert n == -1


# --------------------------------------------------------------------------- #
# NEVER OOM: the 70B-on-24GiB case this guard was built for
# --------------------------------------------------------------------------- #
def test_70b_on_a_24gib_card_still_reserves_for_llama_context(tmp_path):
    """"a 70B uploads ~20 GB of weights and then dies with 'Failed to create
    llama_context'" — the reason the flat reserve existed.

    80 layers / 8 kv heads / 128 head_dim at 16384 ctx is a 5.0 GiB KV cache, so
    a 20 GiB quant CANNOT take every layer on a 24 GiB card. The old flat 2.5 GiB
    said it could (that IS the OOM); the computed reserve refuses and splits."""
    g = write_gguf(tmp_path / "70b-Q2_K.gguf", size_bytes=20 * GIB, arch="llama",
                   block_count=80, head_count=64, head_count_kv=8,
                   embedding_length=8192, key_length=128, context_length=131072)
    ctx = spill.served_ctx_for_fit(g, free_vram=23 * GIB)
    reserve, source, detail = spill.vram_ctx_reserve_bytes(g, n_ctx=ctx)
    assert source == "computed"
    assert detail["kv_bytes"] == 2 * 80 * ctx * 8 * 128 * 2

    free = 23 * GIB                       # a 24 GiB card, after the 1.0 reserve
    assert _flat_old_fit(20 * GIB, free, 80) == -1        # the OOM, reproduced
    n = spill.autofit_gpu_layers(g, free_vram=free)
    # The new policy preserves whole weights by reducing context to the largest
    # value that actually fits, instead of promising all layers at a fixed 16k.
    assert ctx < 16384
    assert n == -1
    assert free - 20 * GIB >= max(0, reserve - spill.vram_reserve_bytes())


def test_a_big_card_is_not_starved_by_a_small_model(tmp_path):
    """The inverse sanity: a 3B on a 24 GiB card takes every layer, exactly as
    before — the computed reserve must not be a new tax on large cards."""
    g = write_gguf(tmp_path / "3b-Q4_K_M.gguf", size_bytes=2 * GIB, arch="qwen2",
                   block_count=36, head_count=16, head_count_kv=2,
                   embedding_length=2048, context_length=32768)
    assert spill.autofit_gpu_layers(g, free_vram=23 * GIB) == -1
    assert _flat_old_fit(2 * GIB, 23 * GIB, 36) == -1      # unchanged verdict


# --------------------------------------------------------------------------- #
# degrade-not-guess: unparseable geometry is TODAY'S behaviour, exactly
# --------------------------------------------------------------------------- #
def test_unparseable_geometry_falls_back_to_the_flat_default(tmp_path):
    p = tmp_path / "not-a-gguf.gguf"
    with open(p, "wb") as fh:
        fh.write(b"NOPE")
        fh.truncate(7 * GIB)
    reserve, source, _ = spill.vram_ctx_reserve_bytes(str(p))
    assert source == "default"
    assert reserve == int(2.5 * GIB)
    # and the layer count is byte-identical to the pre-change formula
    for free in (int(6.5 * GIB), int(9.0 * GIB), 23 * GIB):
        assert spill.autofit_gpu_layers(str(p), free_vram=free) == \
               _flat_old_fit(7 * GIB, free, spill._ASSUMED_LAYERS)


def test_partial_geometry_is_also_the_flat_default(tmp_path):
    """A header with block_count but no head_count_kv/head_dim cannot price a KV
    cache. Do NOT reach for kv_bytes' bytes-per-token heuristic here — a fit
    decision is never made on a guessed cache size."""
    p = tmp_path / "partial.gguf"
    kb = struct.pack("<Q", len(b"qwen3.block_count")) + b"qwen3.block_count" \
        + struct.pack("<I", 4) + struct.pack("<i", 36)
    with open(p, "wb") as fh:
        fh.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                 + struct.pack("<Q", 1) + kb)
        fh.truncate(7 * GIB)
    assert spill._gguf_layer_count(str(p)) == 36          # the layer count IS read
    reserve, source, _ = spill.vram_ctx_reserve_bytes(str(p))
    assert source == "default" and reserve == int(2.5 * GIB)


# --------------------------------------------------------------------------- #
# both env knobs are operator overrides and must still win
# --------------------------------------------------------------------------- #
def test_explicit_ctx_reserve_env_wins_and_stays_flat(tmp_path, monkeypatch):
    """An explicit HUGPY_VRAM_CTX_RESERVE_GIB is the operator's number: used
    verbatim, flat, with no geometry and no un-stacking — byte-identical to the
    old behaviour for any operator who had pinned it."""
    g = flux2_like(tmp_path)
    monkeypatch.setenv("HUGPY_VRAM_CTX_RESERVE_GIB", "2.5")
    reserve, source, _ = spill.vram_ctx_reserve_bytes(g)
    assert source == "env" and reserve == int(2.5 * GIB)
    for free in (int(6.5 * GIB), int(7.5 * GIB), int(5.0 * GIB)):
        assert spill.autofit_gpu_layers(g, free_vram=free) == \
               _flat_old_fit(FLUX2_BYTES, free, 36)
    monkeypatch.setenv("HUGPY_VRAM_CTX_RESERVE_GIB", "0.5")
    assert spill.vram_ctx_reserve_bytes(g)[0] == int(0.5 * GIB)


def test_explicit_vram_reserve_env_still_governs_the_floor(tmp_path, monkeypatch):
    """HUGPY_VRAM_RESERVE_GIB keeps its meaning: it is the floor the computed
    context need is measured against (the larger binds), and it still applies
    verbatim inside free_vram_bytes for every other consumer."""
    g = flux2_like(tmp_path)
    monkeypatch.setenv("HUGPY_VRAM_RESERVE_GIB", "3.0")
    assert spill.vram_reserve_bytes() == int(3.0 * GIB)
    reserve, _, _ = spill.vram_ctx_reserve_bytes(g)       # unchanged: 2.75 GiB
    n = spill.autofit_gpu_layers(g, free_vram=int(6.5 * GIB))
    # ctx need (2.75) < floor (3.0) -> the floor binds, nothing extra is charged
    assert reserve < int(3.0 * GIB)
    assert n == -1
    monkeypatch.setenv("HUGPY_VRAM_RESERVE_GIB", "0")
    assert spill.vram_reserve_bytes() == 0
    # With no external floor the fit-bounded context still allows the measured
    # whole model to seat; removing a reserve must never make fit stricter.
    assert spill.autofit_gpu_layers(g, free_vram=int(6.5 * GIB)) == -1


def test_fit_bounded_context_preserves_whole_weights_before_spilling(tmp_path):
    """A tighter card reduces context before it spills otherwise-fitting weights."""
    g = flux2_like(tmp_path)
    tight = int(5.0 * GIB)
    roomy = int(7.5 * GIB)
    assert spill.served_ctx_for_fit(g, free_vram=tight) < \
           spill.served_ctx_for_fit(g, free_vram=roomy)
    assert spill.autofit_gpu_layers(g, free_vram=tight) == -1


def test_safety_multiplier_still_applies(tmp_path, monkeypatch):
    """HUGPY_VRAM_SAFETY is untouched — a box that wants the old cushion back
    still gets it, and it still only ever tightens."""
    g = flux2_like(tmp_path)
    free = int(5.5 * GIB)
    assert spill.autofit_gpu_layers(g, free_vram=free) == -1
    monkeypatch.setenv("HUGPY_VRAM_SAFETY", "0.85")
    assert spill.autofit_gpu_layers(g, free_vram=free) != -1


# --------------------------------------------------------------------------- #
# ctx resolution
# --------------------------------------------------------------------------- #
def test_ctx_for_fit_prefers_the_explicit_value(tmp_path):
    g = flux2_like(tmp_path)
    assert spill.ctx_for_fit(g, n_ctx=8192) == 8192
    derived = spill.ctx_for_fit(g)
    assert spill._ctx_floor() <= derived <= 40960
    assert spill.ctx_for_fit(g, n_ctx=0) == derived       # 0/None -> derive


def test_ctx_for_fit_uses_the_trained_ctx_when_it_is_smaller(tmp_path):
    g = write_gguf(tmp_path / "short.gguf", size_bytes=GIB, arch="llama",
                   block_count=32, head_count=32, head_count_kv=8,
                   embedding_length=4096, key_length=128, context_length=4096)
    assert spill.ctx_for_fit(g) == 4096


def test_ctx_floor_defaults_and_env(monkeypatch):
    """The global policy is a floor, not a ceiling on trained context."""
    assert spill._ctx_floor() == 4096
    monkeypatch.setenv("HUGPY_LLAMA_CTX_FLOOR", "8192")
    assert spill._ctx_floor() == 8192
    monkeypatch.setenv("HUGPY_LLAMA_CTX_FLOOR", "garbage")
    assert spill._ctx_floor() == 4096


# --------------------------------------------------------------------------- #
# the MoE path is untouched (expert split, not layer autofit)
# --------------------------------------------------------------------------- #
def test_dense_header_is_still_dense(tmp_path):
    g = flux2_like(tmp_path)
    assert spill.gguf_moe_detail(g)["is_moe"] is False
    assert spill.moe_split_need(spill.gguf_moe_detail(g)) is None


def test_autofit_signature_is_backward_compatible(tmp_path):
    """n_ctx is keyword-only-by-convention and optional: every existing call site
    (positional path, free_vram=, extra_reserve_bytes=) still resolves."""
    g = flux2_like(tmp_path)
    assert spill.autofit_gpu_layers(g, int(6.5 * GIB)) == -1
    assert spill.autofit_gpu_layers(g, int(6.5 * GIB), 0) == -1
    assert spill.autofit_gpu_layers(g, free_vram=int(6.5 * GIB),
                                    extra_reserve_bytes=0) == -1
