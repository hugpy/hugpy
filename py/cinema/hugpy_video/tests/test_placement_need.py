"""DERIVED VRAM PLACEMENT NEED (2026-07-27) — the arithmetic, locked.

Covers ``_placement_need_gib`` / ``_latent_tokens`` / ``_placement_budget_gib`` /
``_quantized_move_supported`` / ``_should_place_whole_on_gpu`` in
``studio/runners/wan_i2v.py`` (shared: ``wan_vace.py`` calls the same predicate and
the same budget resolver, so every correction here reaches VACE with no edit there).

WHAT CHANGED AND WHY IT NEEDS A TEST. Until today the placement decision compared the
registry's VramEnvelope — a **DiT-only** planning number — against a flat 16.0 GB
margin. Those two quantities measure different things, and adding them produced
``8.2 + 16.0 = 24.2 > 23.56`` for wan2.1-t2v-1.3b: a refusal by **0.64 GB** on the one
card this fleet renders on. Every 480p render on ae took the slow offload branch
because of that sum. The decision is now built from MEASURED component bytes —
``DiT(precision) + UMT5-XXL + VAE [+ CLIP] [+ MoE second expert] + activations(tokens)``
— read from the safetensors headers under
``/mnt/llm_storage/video_intel/studio/weights/Wan-AI``.

⚠ SECOND PASS, SAME DAY: this file previously carried check [4] as a TRIPWIRE that
recorded — rather than fixed — an activation term which missed its own calibration by
2.64 GiB. That tripwire has now fired and been discharged. The intercept is DERIVED
from the two calibration points instead of typed in, and check [4] below asserts the
corrected arithmetic plus the residual that genuinely remains. What the tripwire
protected against is now impossible by construction; check [3b] asserts that too.

⚠ TWO PLACEMENT VERDICTS FLIPPED as a result, both away from a would-be OOM:
``wan2.1-i2v-14b-720p @fp8 832x480x29f`` read 23.476 ("fits 23.56 by 0.084") and is
really 26.116 (misses by 2.556); ``wan2.1-vace-14b @fp8`` at the same geometry read
22.298 and is really 25.432 (misses by 1.872). Both are pinned in check [6].

Same script style as the other studio suites (plain python, ``__main__`` guard,
numbered ``[n] PASS`` / ``[n] FAIL`` lines, every check independent, nonzero exit iff
any FAILED). Pure arithmetic: no GPU, no weights, no torch, no bitsandbytes.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_placement_need.py
"""
from __future__ import annotations

import logging
import os
import sys

logging.disable(logging.INFO)
os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_video.intel.studio.enums import Precision
from hugpy_video.intel.studio.registry import MODEL_REGISTRY
from hugpy_video.intel.studio.runners.wan_i2v import (
    _BNB_MIN_FOR_MOVE,
    _BYTES_PER_PARAM,
    _CLIP_PARAMS,
    _DECODE_WS_GIB,
    _PLACEMENT_MARGIN_GB,
    _TRANSFORMERS_MIN_FOR_MOVE,
    _UMT5_PARAMS,
    _VAE_PARAMS,
    _WAN_FOOTPRINTS,
    _WS_CAL_HI,
    _WS_CAL_LO,
    _WS_INTERCEPT_GIB,
    _WS_SLOPE_GIB_PER_TOKEN,
    _installed_quantized_move_ok,
    _latent_tokens,
    _placement_budget_gib,
    _placement_need_gib,
    _quantized_move_supported,
    _should_place_whole_on_gpu,
    _version_tuple,
)

# ae's RTX 3090 as torch actually reports it — the ONLY studio render target on this
# fleet (CAPABILITY-VIABILITY-MAP.md:5). Every "does it fit" question below is asked
# against this number, not against the nominal 24.
_AE_CARD_GIB = 23.56
_GIB = 1024.0 ** 3

# The 2026-07-07 ae incident's measured whole-GPU resident for wan2.1-t2v-1.3b @fp16
# 832x480x29f. Kept as a named constant because check [4] measures the model AGAINST
# it and reports the gap; it is evidence, not a target the code is tuned to hit.
_INCIDENT_MEASURED_GIB = 19.6

# A quantized render is only PLACED when the installed bitsandbytes/transformers can
# actually move it (see check [8]). This venv has NO bitsandbytes, so every "it fits,
# therefore place it" assertion about int8/nf4 below states the capability EXPLICITLY
# rather than depending on what happens to be installed where the suite runs.
_MOVE_OK = dict(quantized_move_ok=True)


def _between(value: float, lo: float, hi: float, label: str) -> None:
    assert lo <= value <= hi, f"{label}: {value:.4f} outside [{lo}, {hi}]"


# --------------------------------------------------------------------------- #
# [1] _latent_tokens matches Wan's own cadence: patch_size [1,2,2] over an 8x
#     spatially / 4:1 temporally compressed latent => 16 px per token, 4 frames per
#     latent frame, with the 4k+1 snap the runner already applies to frame counts.
#     Everything downstream (the activation term) is linear in this number, so if
#     the cadence is wrong every VRAM figure is wrong by the same factor.
# --------------------------------------------------------------------------- #
def test_latent_tokens_cadence():
    # 16 px per token in BOTH axes: 832/16 = 52, 480/16 = 30.
    assert _latent_tokens(832, 480, 1) == 52 * 30 * 1, _latent_tokens(832, 480, 1)
    assert _latent_tokens(1280, 720, 1) == 80 * 45 * 1, _latent_tokens(1280, 720, 1)
    # it FLOORS — a non-multiple-of-16 edge contributes no token, it never rounds up.
    assert _latent_tokens(848, 480, 1) == 53 * 30, _latent_tokens(848, 480, 1)
    assert _latent_tokens(847, 480, 1) == 52 * 30, _latent_tokens(847, 480, 1)

    # 4 frames per latent frame with the 4k+1 snap: a latent frame is added at
    # 1, 5, 9, ... and NOT in between, so 29..32 all cost 8 latent frames and 33 costs 9.
    for n, latent in ((1, 1), (2, 1), (4, 1), (5, 2), (8, 2), (9, 3),
                      (29, 8), (30, 8), (32, 8), (33, 9), (81, 21)):
        assert _latent_tokens(16, 16, n) == latent, (n, _latent_tokens(16, 16, n))

    # the geometries every other check in this file leans on, INCLUDING both
    # calibration points (45f and 81f) — check [3b] fits the line through them.
    assert _latent_tokens(832, 480, 29) == 12_480, _latent_tokens(832, 480, 29)
    assert _latent_tokens(832, 480, 45) == 18_720, _latent_tokens(832, 480, 45)
    assert _latent_tokens(832, 480, 81) == 32_760, _latent_tokens(832, 480, 81)
    assert _latent_tokens(1280, 720, 81) == 75_600, _latent_tokens(1280, 720, 81)

    # degenerate inputs clamp to 1 rather than returning 0 — a zero token count would
    # silently price a render as pure weights, which is the failure mode this replaces.
    assert _latent_tokens(832, 480, 0) == 52 * 30, _latent_tokens(832, 480, 0)
    assert _latent_tokens(8, 8, 1) == 1, _latent_tokens(8, 8, 1)


# --------------------------------------------------------------------------- #
# [2] THE HEADLINE: wan2.1-t2v-1.3b @fp16 at 832x480x29f — the exact tuple that lands
#     a clip on ae today — is priced from real bytes at 17.897 GiB and FITS the 23.56
#     GiB card with 5.663 to spare, so _should_place_whole_on_gpu says True. This is
#     the render that has been taking enable_model_cpu_offload() on every single call.
# --------------------------------------------------------------------------- #
def test_t2v_1_3b_fits_the_3090():
    need = _placement_need_gib("wan2.1-t2v-1.3b", Precision.FP16, 832, 480, 29)
    assert need is not None
    _between(need, 17.89, 17.91, "wan2.1-t2v-1.3b fp16 832x480x29f")
    _between(_AE_CARD_GIB - need, 5.65, 5.68, "headroom on ae")
    assert _should_place_whole_on_gpu(
        Precision.FP16, 8.2, _AE_CARD_GIB, model_id="wan2.1-t2v-1.3b",
        width=832, height=480, n_frames=29) is True

    # fp16 and bf16 cost the SAME (2 bytes/param) — the runner computes in bfloat16
    # either way, so the two registry rows must not disagree on placement.
    assert _placement_need_gib("wan2.1-t2v-1.3b", Precision.BF16, 832, 480, 29) == need

    # ...and so does FP32, because compute_dtype is HARDCODED to torch.bfloat16 in
    # run_wan_i2v. The table used to say 4.0 for fp32, which was unreachable AND wrong.
    assert _BYTES_PER_PARAM["fp32"] == 2.0, _BYTES_PER_PARAM["fp32"]
    assert _placement_need_gib("wan2.1-t2v-1.3b", Precision.FP32, 832, 480, 29) == need

    # and the longest clip the fleet can produce (81 frames) still fits: the activation
    # term is LINEAR in tokens, so +20,280 tokens buys ~1.3 GiB, not a cliff.
    long_need = _placement_need_gib("wan2.1-t2v-1.3b", Precision.FP16, 832, 480, 81)
    _between(long_need - need, 1.29, 1.31, "29f -> 81f activation delta")
    assert long_need <= _AE_CARD_GIB, long_need


# --------------------------------------------------------------------------- #
# [3] REGRESSION: the old arithmetic no longer governs. The registry STILL declares
#     8.2 for this row (nothing was rewritten to dodge the problem) and 8.2 + 16.0 =
#     24.2 STILL exceeds 23.56 — the legacy predicate reached without a model_id
#     refuses exactly as it always did. What changed is that the derived path is now
#     the one that answers, and it says yes.
# --------------------------------------------------------------------------- #
def test_legacy_dit_only_plus_flat_margin_no_longer_governs():
    cfg = MODEL_REGISTRY.get("wan2.1-t2v-1.3b")
    assert cfg is not None, "wan2.1-t2v-1.3b missing from the registry"
    envelope = cfg.vram.as_map().get(Precision.FP16)
    assert envelope == 8.2, envelope          # the fictional DiT-only number, unchanged
    assert _PLACEMENT_MARGIN_GB == 16.0, _PLACEMENT_MARGIN_GB   # the flat fudge, kept

    # the refusal, reproduced: 8.2 + 16.0 = 24.2 > 23.56, short by 0.64 GB.
    assert envelope + _PLACEMENT_MARGIN_GB > _AE_CARD_GIB
    _between(envelope + _PLACEMENT_MARGIN_GB - _AE_CARD_GIB, 0.63, 0.65, "the 0.64 GB miss")
    assert _should_place_whole_on_gpu(Precision.FP16, envelope, _AE_CARD_GIB) is False

    # SAME precision, SAME envelope, SAME card — with the identity and geometry passed,
    # the answer flips. That flip is the whole slice.
    assert _should_place_whole_on_gpu(
        Precision.FP16, envelope, _AE_CARD_GIB, model_id="wan2.1-t2v-1.3b",
        width=832, height=480, n_frames=29) is True

    # the real DiT (1.419e9 params @ 2 bytes = 2.64 GiB) is a THIRD of the declared
    # 8.2 — which is why lowering the margin could never have been the fix.
    _between(_WAN_FOOTPRINTS["wan2.1-t2v-1.3b"][0] * 2.0 / _GIB, 2.6, 2.7, "real DiT bf16")


# --------------------------------------------------------------------------- #
# [3b] THE CALIBRATION LINE PASSES THROUGH ITS OWN CALIBRATION POINTS.
#
#      This is the check that would have caught the whole class of defect this round
#      exists to fix. The shipped constants were slope 6.41e-5 (correct — it is the
#      two-point fit) with intercept 0.76, which is NOT the intercept that fit
#      produces: 4.6 - 6.410256e-5*18,720 = 3.400. The line therefore missed BOTH of
#      the points its own comment cited, by 2.64 GiB, in the OPTIMISTIC direction.
#
#      wan_i2v.py now DERIVES both constants from _WS_CAL_LO/_WS_CAL_HI rather than
#      restating them, so the code and the comment cannot drift apart. These
#      assertions exist so that reverting to typed-in literals fails LOUDLY here
#      rather than quietly under-pricing every render on the fleet.
# --------------------------------------------------------------------------- #
def test_workspace_line_passes_through_both_calibration_points():
    lo_tokens, lo_gib = _WS_CAL_LO       # 832x480x45f -> 18,720 tokens -> 4.6 GiB
    hi_tokens, hi_gib = _WS_CAL_HI       # 832x480x81f -> 32,760 tokens -> 5.5 GiB
    # the points are the geometries they claim to be — not free-floating numbers.
    assert _latent_tokens(832, 480, 45) == lo_tokens, lo_tokens
    assert _latent_tokens(832, 480, 81) == hi_tokens, hi_tokens

    def ws(tokens: int) -> float:
        return _WS_INTERCEPT_GIB + _WS_SLOPE_GIB_PER_TOKEN * tokens

    assert abs(ws(lo_tokens) - lo_gib) < 1e-9, (ws(lo_tokens), lo_gib)
    assert abs(ws(hi_tokens) - hi_gib) < 1e-9, (ws(hi_tokens), hi_gib)

    # and the constants themselves, to the digits the comments quote.
    _between(_WS_SLOPE_GIB_PER_TOKEN, 6.4102e-5, 6.4103e-5, "slope GiB/token")
    _between(_WS_INTERCEPT_GIB, 3.3999, 3.4001, "intercept GiB")

    # THE RETIRED VALUE, named so nobody reintroduces it by "restoring" a constant:
    # 0.76 is 2.64 GiB below the fit, and that constant offset propagated into EVERY
    # estimate this module makes, at every model and every geometry.
    _between(_WS_INTERCEPT_GIB - 0.76, 2.63, 2.65, "the correction the 0.76 needed")

    # the intercept is a CONSTANT, so the correction is exactly the same everywhere.
    # (The int8 rows differ because _BYTES_PER_PARAM["int8"] moved 1.06 -> 1.003 too.)
    for n in (29, 81):
        old = 13.6973 + 0.76 + 6.41e-5 * _latent_tokens(832, 480, n)
        new = _placement_need_gib("wan2.1-t2v-1.3b", Precision.FP16, 832, 480, n)
        _between(new - old, 2.63, 2.65, f"per-render correction @{n}f")


# --------------------------------------------------------------------------- #
# [4] DERIVED vs MEASURED — the decomposition, and the residual that is left.
#
#     The 2026-07-07 ae incident measured 19.6 GiB resident for this render, and
#     CAPABILITY-VIABILITY-MAP.md:483 decomposes it as DiT bf16 2.65 + UMT5 bf16 10.58
#     + VAE fp32 0.47 + ~5.9 GiB activations. The STATIC three reproduce exactly. The
#     ACTIVATION term does not, and — unlike the previous revision of this file — that
#     is no longer an intercept bug: with the intercept corrected to 3.400 the model
#     charges 4.200 GiB at 12,480 tokens, so the total is 17.897 and the residual
#     against the incident is 1.703 GiB.
#
#     WHY THE RESIDUAL IS RECORDED AND NOT ABSORBED. It cannot be closed by arithmetic
#     because the source data contradicts itself: the map wants ~5.90 GiB of
#     activations at 12,480 tokens (29f) while the calibration measures 4.60 at 18,720
#     (45f) — MORE activation at FEWER tokens, impossible for anything monotone in
#     tokens. At least one of those three points is mislabelled and nothing in the tree
#     says which. Adding a 1.703 constant to make the totals agree would be tuning the
#     model to the number it is then claimed to predict; this suite records the gap
#     instead, and asserts the safety property that actually matters (below).
# --------------------------------------------------------------------------- #
def test_static_decomposition_matches_the_measured_incident():
    dit = _WAN_FOOTPRINTS["wan2.1-t2v-1.3b"][0] * _BYTES_PER_PARAM["fp16"] / _GIB
    umt5 = _UMT5_PARAMS * 2.0 / _GIB          # bf16
    vae = _VAE_PARAMS * 4.0 / _GIB            # forced fp32 by the runner
    _between(dit, 2.64, 2.65, "DiT bf16")
    _between(umt5, 10.58, 10.59, "UMT5-XXL bf16")
    _between(vae, 0.47, 0.48, "AutoencoderKLWan fp32")
    static = dit + umt5 + vae
    _between(static, 13.69, 13.70, "static resident (map says 2.65+10.58+0.47=13.70)")

    # UMT5-XXL is the single largest resident — FOUR TIMES the 1.3B DiT it serves — and
    # it has no registry row anywhere. Any future "why is the card full" reads here.
    assert umt5 > 4 * dit, (umt5, dit)

    # the activation term the implementation actually charges at 832x480x29f.
    tokens = _latent_tokens(832, 480, 29)
    denoise_ws = _WS_INTERCEPT_GIB + _WS_SLOPE_GIB_PER_TOKEN * tokens
    _between(denoise_ws, 4.19, 4.21, "denoise workspace @29f")
    need = _placement_need_gib("wan2.1-t2v-1.3b", Precision.FP16, 832, 480, 29)
    assert abs(need - (static + max(denoise_ws, _DECODE_WS_GIB))) < 1e-6, need

    # THE RESIDUAL, stated as a number rather than as a hope.
    _between(_INCIDENT_MEASURED_GIB - need, 1.69, 1.72,
             "UNCLOSED derived-vs-measured gap @832x480x29f")

    # THE HYPOTHESIS THAT THE INCIDENT'S GEOMETRY LABEL WAS WRONG — CHECKED, REJECTED.
    # A reviewer suggested 19.6 was really this model at 1280x720x81f. Under the
    # CORRECTED intercept that prices at 21.944, i.e. 2.344 off — further from 19.6
    # than 832x480x29f's 1.703. (It looked like a match only against the broken 0.76
    # intercept, where 720p81f gave 19.31.) The recorded geometry stands.
    p720 = _placement_need_gib("wan2.1-t2v-1.3b", Precision.FP16, 1280, 720, 81)
    _between(p720, 21.94, 21.95, "wan2.1-t2v-1.3b fp16 1280x720x81f")
    assert abs(p720 - _INCIDENT_MEASURED_GIB) > abs(need - _INCIDENT_MEASURED_GIB), (
        p720, need)

    # THE GEOMETRIES THAT *DO* PRICE AT 19.6 — pinned so the docstring's "these are
    # coincidences" claim stays a checkable statement rather than a recollection. The
    # line is monotone in tokens, so it necessarily CROSSES 19.6 somewhere; the point
    # is that everywhere it does is a geometry this row cannot be asked for.
    cfg13 = MODEL_REGISTRY.get("wan2.1-t2v-1.3b")
    assert cfg13.max_frames == 81, cfg13.max_frames
    assert [(r.width, r.height) for r in cfg13.resolutions] == [(832, 480)], cfg13.resolutions
    for w, h, n, expect in ((832, 480, 93, 19.497), (832, 480, 97, 19.597),
                            (1280, 720, 41, 19.636)):
        got = _placement_need_gib("wan2.1-t2v-1.3b", Precision.FP16, w, h, n)
        assert abs(got - expect) < 0.001, (w, h, n, got, expect)
        # ...and every one of them is UNREACHABLE for this row: either past max_frames
        # or at a resolution the registry does not declare.
        assert n > cfg13.max_frames or (w, h) not in [
            (r.width, r.height) for r in cfg13.resolutions], (w, h, n)

    # ⚠ THE SAFETY PROPERTY. Under-pricing by 1.703 GiB is only tolerable while every
    # surviving whole-GPU verdict clears the card by MORE than the residual. The
    # thinnest one on the fleet is wan2.1-vace-1.3b @fp16 832x480x81f. If any future
    # edit narrows a fit below the residual, this fails and points at exactly this
    # note instead of surfacing later as an OOM on ae.
    thinnest = min(
        _AE_CARD_GIB - _placement_need_gib(mid, prec, 832, 480, n)
        for mid, precisions in (("wan2.1-t2v-1.3b", (Precision.FP16, Precision.INT8)),
                                ("wan2.1-vace-1.3b", (Precision.FP16, Precision.INT8)))
        for prec in precisions for n in (29, 81))
    _between(thinnest, 2.99, 3.00, "thinnest surviving whole-GPU headroom")
    assert thinnest > (_INCIDENT_MEASURED_GIB - need), (thinnest, need)


# --------------------------------------------------------------------------- #
# [5] THE MoE ROWS CANNOT FIT, AT ANY PRECISION THE REGISTRY DECLARES. wan2.2-*-a14b
#     is two ~14.29B experts; quantizing `transformer` does not touch `transformer_2`,
#     which loads UNQUANTIZED at bf16 = 26.6 GiB — more than the whole card, before a
#     single other component. The registry's FP8=20.0 / INT8=14.0-16.0 rows read as
#     though they fit; they do not, and this is what stops that fiction reaching a
#     render.
#
#     Also pins the CORRECTED wan2.2-i2v-a14b sidecar set: its model_index.json says
#     "image_encoder": [null, null] and no such directory exists on disk, so the row
#     must NOT be charged CLIP ViT-H. It was, to the tune of +1.177 GiB.
# --------------------------------------------------------------------------- #
def test_moe_a14b_never_fits_the_3090():
    for model_id, expected in (("wan2.2-t2v-a14b", 14.2885e9),
                               ("wan2.2-i2v-a14b", 14.2889e9)):
        params, has_image_encoder, extra = _WAN_FOOTPRINTS[model_id]
        assert params == expected, (model_id, params)
        assert extra == expected, (model_id, extra)     # transformer_2, same size
        # NEITHER wan 2.2 row has an image encoder — i2v included. Wan 2.2 conditions
        # i2v through the DiT's own 36 in_channels, not through CLIP ViT-H.
        assert has_image_encoder is False, model_id
        second_expert = extra * 2.0 / _GIB              # bf16, always
        _between(second_expert, 26.6, 26.7, f"{model_id} transformer_2 bf16")
        assert second_expert > _AE_CARD_GIB, "the SECOND EXPERT ALONE overflows the card"

    # the phantom that was being charged: CLIP ViT-H at bf16.
    _between(_CLIP_PARAMS * 2.0 / _GIB, 1.17, 1.18, "the spurious i2v-a14b CLIP charge")

    # The wan2.2 a14b rows were REMOVED from the seed (operator, 2026-08-13):
    # the footprint arithmetic above is the record of WHY; the registry must
    # not carry them (a re-added row would need the second expert charged).
    for model_id in ("wan2.2-t2v-a14b", "wan2.2-i2v-a14b"):
        assert MODEL_REGISTRY.get(model_id) is None, f"{model_id} removed 2026-08-13"


# --------------------------------------------------------------------------- #
# [6] THE 14B ROWS DO NOT FIT 832x480 AT ANY PRECISION — INCLUDING nf4.
#
#     ⚠ THIS IS THE VERDICT THAT FLIPPED, and it is the reason the intercept mattered.
#     With the 0.76 intercept, wan2.1-i2v-14b-720p @fp8 832x480x29f priced at 23.476
#     and "fit" 23.56 by 0.084 GiB — a margin smaller than the error in the constant
#     producing it. Under the corrected calibration it is 26.116, a MISS by 2.556. With
#     the INT8/FP8 blanket rule retired, that wrong verdict was a bare pipe.to("cuda")
#     with no offload fallback: a CUDA OOM after loading ~21 GiB of weights.
#
#     wan2.1-vace-14b flipped too, and for TWO independent reasons: the same intercept,
#     plus its parameter count being corrected 16.3951e9 -> 17.3376e9 (the row was
#     carrying the i2v 14B's count under the false gloss "same geometry"). 22.298 ->
#     25.432, a miss by 1.872.
# --------------------------------------------------------------------------- #
def test_i2v_14b_and_vace_14b_do_not_fit_even_quantized():
    # weights-only floor for the i2v 14B: DiT nf4 + UMT5 bf16 + VAE fp32 + CLIP bf16.
    dit_nf4 = _WAN_FOOTPRINTS["wan2.1-i2v-14b-720p"][0] * _BYTES_PER_PARAM["fp8"] / _GIB
    static = (dit_nf4 + _UMT5_PARAMS * 2.0 / _GIB + _VAE_PARAMS * 4.0 / _GIB
              + _CLIP_PARAMS * 2.0 / _GIB)
    _between(static, 20.8, 20.9, "i2v-14b nf4 static resident")

    need = _placement_need_gib("wan2.1-i2v-14b-720p", Precision.FP8, 832, 480, 29)
    _between(need, 26.10, 26.13, "i2v-14b nf4 832x480x29f")
    _between(need - _AE_CARD_GIB, 2.54, 2.57, "i2v-14b nf4 MISS")
    assert _should_place_whole_on_gpu(
        Precision.FP8, 18.0, _AE_CARD_GIB, model_id="wan2.1-i2v-14b-720p",
        width=832, height=480, n_frames=29, **_MOVE_OK) is False

    # the retired arithmetic, reproduced, so the size of the escape is on the record:
    # the SAME row read as fitting by 0.084 GiB under the 0.76 intercept.
    stale = static + 0.76 + 6.41e-5 * _latent_tokens(832, 480, 29) * (78_848 / 33_280)
    _between(stale, 23.47, 23.49, "what the 0.76 intercept said")
    assert stale <= _AE_CARD_GIB, stale        # ...and it "fit". That was the bug.

    # wan2.1-vace-14b: the parameter count is a DISK FACT, not the i2v 14B's.
    vace_params = _WAN_FOOTPRINTS["wan2.1-vace-14b"][0]
    assert vace_params == 17.3376e9, vace_params
    assert vace_params > _WAN_FOOTPRINTS["wan2.1-i2v-14b-720p"][0], "vace-14b is BIGGER"
    _between((vace_params - _WAN_FOOTPRINTS["wan2.1-i2v-14b-720p"][0]) * 2.0 / _GIB,
             1.75, 1.77, "bf16 cost of the correction")
    assert _WAN_FOOTPRINTS["wan2.1-vace-14b"][1] is False, "vace-14b has no image encoder"

    vace_need = _placement_need_gib("wan2.1-vace-14b", Precision.FP8, 832, 480, 29)
    _between(vace_need, 25.42, 25.45, "vace-14b nf4 832x480x29f")
    assert _should_place_whole_on_gpu(
        Precision.FP8, 20.0, _AE_CARD_GIB, model_id="wan2.1-vace-14b",
        width=832, height=480, n_frames=29, **_MOVE_OK) is False

    # nothing about the 14B rows fits at 480p, at any declared precision or length.
    for model_id in ("wan2.1-i2v-14b-720p", "wan2.1-vace-14b"):
        for precision in (Precision.BF16, Precision.FP8, Precision.INT8):
            for n in (29, 81):
                assert _placement_need_gib(model_id, precision, 832, 480, n) > _AE_CARD_GIB, (
                    model_id, precision, n)

    # bf16 is not close — the registry's "40.0"/"42.0" are the envelopes in the right
    # ballpark, and even they under-state it (the real bf16 DiT alone is 30.55 GiB).
    bf16 = _placement_need_gib("wan2.1-i2v-14b-720p", Precision.BF16, 832, 480, 29)
    _between(bf16, 48.0, 48.1, "i2v-14b bf16 832x480x29f")


# --------------------------------------------------------------------------- #
# [7] AN UNMEASURED MODEL KEEPS EXACTLY TODAY'S BEHAVIOUR. _placement_need_gib returns
#     None rather than guessing, and _should_place_whole_on_gpu falls all the way back
#     to the legacy model_gb + margin test — INCLUDING the INT8/FP8 early return, which
#     is only retired on the derived path (check [8]). A model nobody has weighed must
#     not become more permissive because a calculation exists for other models.
# --------------------------------------------------------------------------- #
def test_unknown_model_falls_back_to_legacy_unchanged():
    assert "ltx-video-0.9.7-dev" not in _WAN_FOOTPRINTS      # a real unweighed row
    assert _placement_need_gib("ltx-video-0.9.7-dev", Precision.BF16, 832, 480, 29) is None
    assert _placement_need_gib("", Precision.BF16, 832, 480, 29) is None
    # a precision outside the byte table is equally "unknown" — no guess, no default.
    assert _placement_need_gib("wan2.1-t2v-1.3b", "int4", 832, 480, 29) is None

    geo = dict(model_id="ltx-video-0.9.7-dev", width=832, height=480, n_frames=29)
    # the INT8/FP8 early return SURVIVES on the legacy path (it is the stale-but-safe
    # "never .to() a bnb pipeline" rule), even for a size that would trivially fit, and
    # even when the installed stack COULD do the move.
    assert _should_place_whole_on_gpu(Precision.INT8, 4.0, 24.0, **geo, **_MOVE_OK) is False
    assert _should_place_whole_on_gpu(Precision.FP8, 4.0, 24.0, **geo, **_MOVE_OK) is False
    # unquantized: still the flat model_gb + 16.0 <= budget comparison, to the decimal.
    assert _should_place_whole_on_gpu(Precision.FP16, 7.9, 24.0, **geo) is True   # 23.9
    assert _should_place_whole_on_gpu(Precision.BF16, 8.0, 24.0, **geo) is True   # 24.0
    assert _should_place_whole_on_gpu(Precision.BF16, 8.1, 24.0, **geo) is False  # 24.1
    assert _should_place_whole_on_gpu(Precision.BF16, 40.0, 24.0, **geo) is False
    # unknown budget or unknown size -> conservative offload, unchanged.
    assert _should_place_whole_on_gpu(Precision.BF16, 8.2, None, **geo) is False
    assert _should_place_whole_on_gpu(Precision.BF16, None, 24.0, **geo) is False
    # the margin stays tunable on the legacy path.
    assert _should_place_whole_on_gpu(Precision.BF16, 20.0, 24.0, margin=2.0, **geo) is True

    # PARTIAL geometry is also "unknown" — a measured model with no frame count must
    # take the legacy path rather than invent one. 8.2 + 16.0 = 24.2 > 23.56 -> False.
    assert _should_place_whole_on_gpu(
        Precision.FP16, 8.2, _AE_CARD_GIB, model_id="wan2.1-t2v-1.3b",
        width=832, height=480) is False
    assert _should_place_whole_on_gpu(
        Precision.FP16, 8.2, _AE_CARD_GIB, width=832, height=480, n_frames=29) is False
    # a missing budget beats everything, derived path or not.
    assert _should_place_whole_on_gpu(
        Precision.FP16, 8.2, None, model_id="wan2.1-t2v-1.3b",
        width=832, height=480, n_frames=29) is False


# --------------------------------------------------------------------------- #
# [8] THE QUANTIZED-MOVE GUARD. On the derived path the INT8/FP8 blanket rule is gone,
#     but it was replaced by a CAPABILITY CHECK, not by nothing.
#
#     In the installed diffusers 0.39.0 (pipeline_utils.py:562) an 8-bit pipeline is
#     moved to CUDA only when bitsandbytes >= 0.48.0 AND transformers > 4.58.0. When
#     the gate fails diffusers LOGS A WARNING AND FALLS THROUGH: the unquantized
#     components (VAE, UMT5) still go to CUDA via the `elif`, the quantized DiT does
#     not, and the render dies with a device mismatch at the first denoise step —
#     after paying a multi-GB load. The 4-bit gate (:558) is transformers > 4.44.0
#     with bitsandbytes >= 0.43.2 and strands identically.
#
#     So "it fits" is necessary and not sufficient. This venv has NO bitsandbytes at
#     all, which is exactly the environment the guard has to be correct in.
# --------------------------------------------------------------------------- #
def test_quantized_move_guard():
    # ── the pure version predicate, against the real gate boundaries ──────────
    assert _quantized_move_supported(Precision.INT8, "0.49.2", "5.12.1") is True   # ae
    assert _quantized_move_supported(Precision.INT8, "0.48.0", "5.12.1") is True   # floor
    assert _quantized_move_supported(Precision.INT8, "0.47.9", "5.12.1") is False
    assert _quantized_move_supported(Precision.INT8, "0.46.1", "5.12.1") is False  # OLD PIN
    assert _quantized_move_supported(Precision.INT8, "0.49.2", "4.58.0") is False  # STRICT >
    assert _quantized_move_supported(Precision.INT8, "0.49.2", "4.58.1") is True
    assert _quantized_move_supported(Precision.FP8, "0.43.2", "4.44.1") is True
    assert _quantized_move_supported(Precision.FP8, "0.43.1", "5.12.1") is False
    assert _quantized_move_supported(Precision.FP8, "0.49.2", "4.44.0") is False   # STRICT >
    # an unquantized precision has no gate to fail — .to() is unconditional there.
    for precision in (Precision.BF16, Precision.FP16, Precision.FP32):
        assert _quantized_move_supported(precision, None, None) is True, precision
    # UNKNOWN IS NOT "PROBABLY FINE": absent or unparseable => refuse to place.
    assert _quantized_move_supported(Precision.INT8, None, "5.12.1") is False
    assert _quantized_move_supported(Precision.INT8, "0.49.2", None) is False
    assert _quantized_move_supported(Precision.INT8, "not-a-version", "5.12.1") is False
    assert _version_tuple("5.12.1+cu128") == (5, 12, 1), _version_tuple("5.12.1+cu128")
    assert _version_tuple("0.49.2.dev0") == (0, 49, 2), _version_tuple("0.49.2.dev0")
    assert _version_tuple(None) is None and _version_tuple("") is None

    # ── the probe, in THIS venv (bitsandbytes is not installed) ───────────────
    assert _installed_quantized_move_ok(Precision.INT8) is False
    assert _installed_quantized_move_ok(Precision.FP8) is False
    assert _installed_quantized_move_ok(Precision.BF16) is True

    # ── the guard, wired into the decision ────────────────────────────────────
    geo = dict(model_id="wan2.1-t2v-1.3b", width=832, height=480, n_frames=29)
    for precision in (Precision.INT8, Precision.FP8):
        need = _placement_need_gib("wan2.1-t2v-1.3b", precision, 832, 480, 29)
        assert need is not None and need < _AE_CARD_GIB, (precision, need)
        # it FITS, and is still offloaded, because this box cannot perform the move.
        assert _should_place_whole_on_gpu(
            precision, 5.0, _AE_CARD_GIB, **geo, quantized_move_ok=False) is False
        # same call on a box that CAN (ae: bnb 0.49.2) places it — the point of
        # quantizing is to be placed, and the need calculation is what decides.
        assert _should_place_whole_on_gpu(
            precision, 5.0, _AE_CARD_GIB, **geo, quantized_move_ok=True) is True
        # the probe is the default when no override is passed.
        assert _should_place_whole_on_gpu(precision, 5.0, _AE_CARD_GIB, **geo) is False

    # the override is scoped to QUANTIZED precisions — it can never offload a bf16
    # pipeline that fits, because bf16 has no move gate to fail.
    assert _should_place_whole_on_gpu(
        Precision.FP16, 8.2, _AE_CARD_GIB, **geo, quantized_move_ok=False) is True

    # int8 is 1.003 bytes/param (int8 CB + one fp32 SCB per output row, i.e.
    # 1 + 4/in_features per Linear, plus ~0.1% non-Linear params left at bf16), so it
    # is still BIGGER than nf4's 0.5625 — but it is NOT the 1.06 this table used to
    # claim. LLM.int8()'s outlier columns come out of the ACTIVATION at matmul time;
    # they were never stored weight and never belonged in this constant.
    assert _BYTES_PER_PARAM["int8"] == 1.003, _BYTES_PER_PARAM["int8"]
    assert _BYTES_PER_PARAM["int8"] > _BYTES_PER_PARAM["fp8"]
    _between(_WAN_FOOTPRINTS["wan2.1-i2v-14b-720p"][0] * (1.06 - 1.003) / _GIB,
             0.86, 0.88, "GiB the 1.06 over-charged the 14B DiT")
    int8 = _placement_need_gib("wan2.1-t2v-1.3b", Precision.INT8, 832, 480, 29)
    nf4 = _placement_need_gib("wan2.1-t2v-1.3b", Precision.FP8, 832, 480, 29)
    assert int8 > nf4, (int8, nf4)


# --------------------------------------------------------------------------- #
# [9] THE BUDGET IS A CEILING, AND EVERY CEILING BINDS. A reviewer measured
#     POST /video/studio/i2v {"vram_budget_gb": 6.0} routing to INT8 (envelope 5.0
#     fits 6.0) and then placement answering whole-GPU=True against the 24 GB CARD —
#     a 6 GB request served with a 15 GiB placement. _placement_budget_gib settles the
#     semantics: the budget is the MIN of every ceiling that applies, so a declared
#     budget can only ever LOWER the answer and the decision fails toward offload.
#
#     ⚠ HONEST SCOPE. This closes the ceilings the runner can see (live device +
#     the manifest's STUDIO_MAX_VRAM_GB). The reviewer's 6.0 is
#     CapabilityRequest.vram_budget_gb, which is never plumbed to a runner at all —
#     see the seam named in _placement_budget_gib's docstring.
# --------------------------------------------------------------------------- #
def test_placement_budget_is_the_min_of_every_ceiling():
    assert _placement_budget_gib(23.56, 24.0) == 23.56    # ae today: the card binds
    assert _placement_budget_gib(24.0, 6.0) == 6.0        # a declared budget binds
    assert _placement_budget_gib(8.0, 24.0) == 8.0        # computron: the 4060 binds
    # unknown is not zero: a missing ceiling drops out instead of forcing offload.
    assert _placement_budget_gib(None, 24.0) == 24.0
    assert _placement_budget_gib(23.56, None) == 23.56
    assert _placement_budget_gib(None, None) is None      # -> caller answers offload

    # the render that fits ae outright is refused under a 6 GB declared budget, at the
    # SAME precision the router admitted on that budget. That is the whole finding.
    geo = dict(model_id="wan2.1-t2v-1.3b", width=832, height=480, n_frames=29)
    assert _should_place_whole_on_gpu(Precision.FP16, 8.2, _AE_CARD_GIB, **geo) is True
    assert _should_place_whole_on_gpu(
        Precision.FP16, 8.2, _placement_budget_gib(_AE_CARD_GIB, 6.0), **geo) is False
    assert _should_place_whole_on_gpu(
        Precision.INT8, 5.0, _placement_budget_gib(_AE_CARD_GIB, 6.0),
        **geo, **_MOVE_OK) is False

    # and the same quantized 14B refused on ae is refused on computron's 8 GiB 4060 —
    # the decision is against the resolved budget, not against a hardcoded 24.0.
    assert _should_place_whole_on_gpu(
        Precision.FP8, 18.0, 8.0, model_id="wan2.1-i2v-14b-720p",
        width=832, height=480, n_frames=29, **_MOVE_OK) is False


# --------------------------------------------------------------------------- #
# [11] THE PACKAGING PIN AND THE RUNTIME GUARD AGREE. Finding 3 had TWO halves and
#      they can drift apart independently: pyproject declares what a FRESH install
#      gets, _BNB_MIN_FOR_MOVE declares what the 8-bit move actually needs. The old
#      pin was bitsandbytes>=0.46.1 — BELOW the 0.48.0 the diffusers gate requires —
#      so `pip install .[imagegen]` produced a box that satisfied the dependency
#      solver and then silently stranded every INT8 whole-GPU render on the CPU.
#
#      Asserting the pin FLOOR is >= the gate FLOOR is the invariant. It fails loudly
#      if anyone relaxes the pin without also lowering what the code requires (which
#      they cannot, because the requirement is diffusers').
# --------------------------------------------------------------------------- #
def _media_pyproject() -> str | None:
    """The ``imagegen`` extra (diffusers + bitsandbytes pins) is declared by
    hugpy_media; locate its pyproject from the installed package (editable /
    checkout layout: src/hugpy_media -> ../../pyproject.toml)."""
    import importlib.util
    spec = importlib.util.find_spec("hugpy_media")
    if spec is None or not spec.submodule_search_locations:
        return None
    pkg_dir = list(spec.submodule_search_locations)[0]
    cand = os.path.join(os.path.dirname(os.path.dirname(pkg_dir)), "pyproject.toml")
    return cand if os.path.isfile(cand) else None


def test_bitsandbytes_pin_meets_the_move_gate():
    path = _media_pyproject()
    if path is None:
        import pytest
        pytest.skip("hugpy_media pyproject.toml not reachable (wheel install)")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    # the declared floor, parsed out of the imagegen extra rather than restated here.
    pins = [ln for ln in text.splitlines()
            if "bitsandbytes>=" in ln and not ln.strip().startswith("#")]
    assert len(pins) == 1, pins
    declared = _version_tuple(pins[0].split("bitsandbytes>=", 1)[1].strip(' "\',]'))
    assert declared is not None, pins[0]

    required = _BNB_MIN_FOR_MOVE["int8"]                      # (0, 48, 0)
    assert declared >= required, (declared, required)
    # the retired pin, named so "restoring" it fails HERE and not on ae at first denoise.
    assert _version_tuple("0.46.1") < required
    # and the pin, fed through the actual predicate, must ADMIT the move it promises.
    pinned = ".".join(str(p) for p in declared)
    assert _quantized_move_supported(Precision.INT8, pinned, "5.12.1") is True, pinned


# --------------------------------------------------------------------------- #
# [12] THE GATE TABLE STILL MATCHES THE LIBRARY IT MODELS. _BNB_MIN_FOR_MOVE /
#      _TRANSFORMERS_MIN_FOR_MOVE are a HAND COPY of version literals living in
#      diffusers' pipeline_utils.py. A diffusers upgrade that moves those literals
#      turns our guard into a confident lie in whichever direction the upgrade went.
#
#      So: read the installed source and assert the literals we encode are the
#      literals actually there. This is the only check in the file that touches an
#      installed package; it SKIPS (never fails) where diffusers is absent, because
#      the arithmetic suite must stay runnable on a box with no GPU stack.
# --------------------------------------------------------------------------- #
def test_encoded_gate_matches_installed_diffusers_source():
    # ⚠ LOCATE THE FILE WITHOUT IMPORTING IT (2026-07-27, round-2 review). This check
    # originally did ``import diffusers.pipelines.pipeline_utils`` purely to read
    # ``pu.__file__`` — which dragged torch + transformers + diffusers into sys.modules
    # for the WHOLE pytest process. Alphabetical collection puts this file before
    # test_studio_enhance.py, whose ``test_enhance_imports_are_gpu_stack_free`` then
    # failed in-sweep ("pulled ['torch','diffusers','transformers']") while passing
    # standalone. That guard is a REAL invariant — the enhance runners are the
    # GPU-less-central path — so the source-read is what yields, not the guard.
    # We only ever needed the file's TEXT, so resolve it off sysconfig instead: no
    # import, no sys.modules mutation, same assertion.
    import os as _os
    import sysconfig
    candidates = [sysconfig.get_paths().get(k) for k in ("purelib", "platlib")]
    src = None
    for base in [c for c in candidates if c]:
        path = _os.path.join(base, "diffusers", "pipelines", "pipeline_utils.py")
        if _os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                src = fh.read()
            break
    if src is None:                         # absent stack: nothing to check
        print("      (skipped: diffusers not installed here)")
        return

    # the 8-bit move gate: BOTH clauses, as diffusers writes them.
    assert 'is_transformers_version(">", "4.58.0")' in src, "8-bit transformers gate moved"
    assert 'is_bitsandbytes_version(">=", "0.48.0")' in src, "8-bit bnb gate moved"
    # the 4-bit move gate.
    assert 'is_transformers_version(">", "4.44.0")' in src, "4-bit transformers gate moved"

    # ...and those literals are exactly what our table encodes.
    assert _TRANSFORMERS_MIN_FOR_MOVE["int8"] == (4, 58, 0), _TRANSFORMERS_MIN_FOR_MOVE
    assert _TRANSFORMERS_MIN_FOR_MOVE["fp8"] == (4, 44, 0), _TRANSFORMERS_MIN_FOR_MOVE
    assert _BNB_MIN_FOR_MOVE["int8"] == (0, 48, 0), _BNB_MIN_FOR_MOVE

    # THE SILENT HALF, asserted from the source (see the gate note in wan_i2v.py): the
    # "cannot move it to cuda" warning is conditioned on bitsandbytes ALONE, so a box
    # with new bnb + old transformers is stranded with NO log line whatsoever. If a
    # future diffusers starts warning on the transformers clause too, this fails and
    # the comment claiming silence needs revisiting.
    warn_line = [ln for ln in src.splitlines()
                 if "is_loaded_in_8bit_bnb and device is not None" in ln
                 and "is_bitsandbytes_version" in ln]
    assert len(warn_line) == 1, warn_line
    assert "is_transformers_version" not in warn_line[0], warn_line[0]


CHECKS = [
    ("_latent_tokens: 16 px/token, 4 frames/latent frame, 4k+1 snap, clamps",
     test_latent_tokens_cadence),
    ("headline: wan2.1-t2v-1.3b fp16 832x480x29f = 17.897 GiB, FITS 23.56 -> whole-GPU",
     test_t2v_1_3b_fits_the_3090),
    ("regression: registry 8.2 + flat 16.0 = 24.2 > 23.56 no longer governs the decision",
     test_legacy_dit_only_plus_flat_margin_no_longer_governs),
    ("calibration: the workspace line passes through BOTH cited points (intercept 3.400)",
     test_workspace_line_passes_through_both_calibration_points),
    ("decomposition: static 13.70 matches the incident; 1.703 GiB residual RECORDED",
     test_static_decomposition_matches_the_measured_incident),
    ("MoE wan2.2-*-a14b: transformer_2 stays bf16, no image encoder -> nothing fits",
     test_moe_a14b_never_fits_the_3090),
    ("14B rows: nf4 MISSES 832x480x29f (26.116 / 25.432) — the verdict that flipped",
     test_i2v_14b_and_vace_14b_do_not_fit_even_quantized),
    ("unknown model/precision/geometry -> None + legacy margin path unchanged (INT8/FP8 False)",
     test_unknown_model_falls_back_to_legacy_unchanged),
    ("quantized move guard: fits is NOT sufficient — bnb/transformers must support .to()",
     test_quantized_move_guard),
    ("budget semantics: placement budget = MIN(card, declared); a 6 GB budget binds",
     test_placement_budget_is_the_min_of_every_ceiling),
    ("pyproject bitsandbytes pin floor >= the 8-bit move gate floor (0.46.1 -> 0.48.0)",
     test_bitsandbytes_pin_meets_the_move_gate),
    ("the encoded gate table still matches the INSTALLED diffusers pipeline_utils source",
     test_encoded_gate_matches_installed_diffusers_source),
]


def main() -> int:
    passed = failed = 0
    for i, (name, fn) in enumerate(CHECKS, 1):
        try:
            fn()
        except Exception as exc:  # surface EVERY divergence, not just the first
            failed += 1
            print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"[{i}] PASS  {name}")
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
