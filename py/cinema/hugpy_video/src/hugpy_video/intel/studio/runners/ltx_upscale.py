"""PREMIUM spatial-upscale runner (LTX latent upscaler) — the weight-backed UPRES
path that outranks the ffmpeg last-resort once its weights are staged. IMPORT-SAFE
and GRACEFULLY-DEGRADING today: preflight returns ``Err(StageError(...))`` as DATA,
never raises.

    run_ltx_upscale(manifest, out_root, start_image=None, should_cancel=None)
        -> Result[Artifact, StageError]

WHY IT DEGRADES TODAY: the weights ARE staged (2026-07-07, official
``Lightricks/ltxv-spatial-upscaler-0.9.7`` — public repo, NOT license-gated; the
earlier "0.9.8 / 401 license gate" story was a wrong repo id — that name only
exists as a third-party copy under ``linoyts/``) at the shared weights root, but
the executable latent-upscale path is not wired yet (see below) and needs a GPU
box. Preflight reports WEIGHTS_MISSING only on a box that can't see the shared
store. The diffusers import that the real path needs is LAZY (inside the runner,
after preflight passes), so this module never drags diffusers into app boot —
matching the wan runners' discipline.

    TO MAKE THIS RUNNER REAL (on a GPU render box, e.g. via the studio worker
    render seam):
      1. weights: already at STUDIO_WEIGHTS_ROOT/Lightricks/
         ltxv-spatial-upscaler-0.9.7 on the shared store (done 2026-07-07),
      2. deps: the diffusers LTX stack (ae already ships diffusers 0.39.0),
      3. wire the real latent-upscale path here (LTXLatentUpsamplePipeline,
         ffmpeg-assemble into the same content-addressed clip layout).

PREFLIGHT ORDER (mirrors the enhancement runners): source (SPEC) -> weights (box
capability). A source-less upres is a SPEC error on ANY box (SOURCE_MISSING),
reported before the box-level WEIGHTS_MISSING so it is not masked. NOTE the
weights-first (not deps-first) order: the point of THIS runner is the premium
weights, so WEIGHTS_MISSING is the honest headline on a box that hasn't staged them
(diffusers may or may not be installed; either way the weights are the blocker).

No pathlib anywhere. os.path only.
"""

from __future__ import annotations

import os
from typing import Callable

from hugpy_video.intel.studio.artifacts import Artifact
from hugpy_video.intel.studio.errors import Err, ErrorCode, Ok, Result, StageError
from hugpy_video.intel.studio.registry import MODEL_REGISTRY
from hugpy_video.intel.studio.schemas import RenderManifest
# The weights-root / local-dir resolution is IDENTICAL to the wan runners (mirror
# an HF ``org/name`` under STUDIO_WEIGHTS_ROOT). Reuse those PURE helpers — they pull
# only stdlib + the synthetic sidecar helpers (numpy/PIL), NOT the GPU stack, so this
# import stays app-boot-safe.
from hugpy_video.intel.studio.runners.wan_i2v import (
    _hot_weights_root,
    _local_model_dir,
    _resolve_model_dir,
    _weights_root,
)


def _resolve_source(manifest: RenderManifest) -> str | None:
    src = getattr(manifest, "source_video", "") or ""
    return src or None


def run_ltx_upscale(
    manifest: RenderManifest,
    out_root: str,
    start_image: str | None = None,
    should_cancel: "Callable[[], bool] | None" = None,
) -> Result[Artifact, StageError]:
    """Produce (or resume) an LTX-upscaled clip for ``manifest`` under ``out_root``.
    Returns ``Ok(Artifact)`` on the real path (once the license-gated weights are
    staged on the box), or ``Err(StageError)`` on any expected failure — including
    the graceful preflight failures that make this a no-op today (SOURCE_MISSING when
    the enhance carries no source clip, WEIGHTS_MISSING when the HF-gated weights are
    not on disk). Only a non-RenderManifest raises. ``start_image`` is part of the
    uniform runner signature but UNUSED (upscale conditions on the whole source clip)."""
    if not isinstance(manifest, RenderManifest):
        raise TypeError(
            f"manifest must be a RenderManifest; got {type(manifest).__name__}")

    # SOURCE-first (SPEC error, malformed on any box) — mirrors ffmpeg_enhance/wan_vace.
    source = _resolve_source(manifest)
    if source is None:
        return Err(StageError(
            ErrorCode.SOURCE_MISSING,
            "LTX upscale carries no source_video — a spatial-upscale enhance is "
            "defined by the clip it upscales; supply source_video",
            (("model_id", manifest.model_id), ("capability", manifest.capability.value))))
    if not os.path.isfile(source):
        return Err(StageError(
            ErrorCode.SOURCE_MISSING,
            f"LTX upscale source_video not found on disk: {source}",
            (("source_video", source), ("model_id", manifest.model_id))))

    # BOX capability: the HF-gated LTX upscaler weights are not on disk. This is the
    # headline blocker for THIS premium path (the weights are license-gated 401).
    cfg = MODEL_REGISTRY.get(manifest.model_id)
    weight_uri = cfg.weight_uri if cfg is not None else "Lightricks/ltxv-spatial-upscaler-0.9.7"
    # WEIGHTS root resolution honors the box-local HOT NVMe copy first (item 5), then
    # the shared/snapshot root — identical to the wan runners (_resolve_model_dir).
    hot = _hot_weights_root()
    weights_root = _weights_root(manifest)
    gate = (" These weights are staged on the SHARED store since 2026-07-07 "
            "(public repo, no license gate) — if missing on this box, mount the "
            f"shared weights root, stage a hot NVMe copy, or `hf download {weight_uri}` "
            "into one. The ffmpeg lanczos last-resort serves UPRES in the meantime.")
    if not hot and not weights_root:
        return Err(StageError(
            ErrorCode.WEIGHTS_MISSING,
            "no weights root set — neither STUDIO_WEIGHTS_HOT_ROOT (box-local NVMe) "
            "nor STUDIO_WEIGHTS_ROOT is configured to resolve the LTX spatial-upscaler "
            "weights against." + gate,
            (("model_id", manifest.model_id), ("weight_uri", weight_uri))))

    model_dir, _root_used = _resolve_model_dir(manifest, weight_uri)
    if not model_dir or not (os.path.isdir(model_dir)
            and os.path.isfile(os.path.join(model_dir, "model_index.json"))):
        return Err(StageError(
            ErrorCode.WEIGHTS_MISSING,
            "LTX spatial-upscaler weights not found on disk (hot NVMe "
            + (_local_model_dir(hot, weight_uri) if hot else "unset")
            + ", shared "
            + (_local_model_dir(weights_root, weight_uri) if weights_root else "unset")
            + ")." + gate,
            (("weight_uri", weight_uri),
             ("hot_root", hot or ""), ("shared_root", weights_root or ""))))

    # REAL PATH (only once the weights are staged on the box; never reached on this
    # dev VM). diffusers is imported LAZILY here so this module stays app-boot-safe.
    # Kept a graceful Err until the executable latent-upscale wiring lands with the
    # box errand, so a partial stage never crashes the studio.
    return Err(StageError(
        ErrorCode.WEIGHTS_MISSING,
        "LTX spatial-upscaler weights are present but the executable latent-upscale "
        "path is not wired yet (box errand step 3); the ffmpeg lanczos last-resort "
        "serves UPRES in the meantime.",
        (("model_id", manifest.model_id), ("model_dir", model_dir))))
