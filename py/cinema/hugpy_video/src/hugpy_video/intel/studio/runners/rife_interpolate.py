"""PREMIUM frame-interpolation runner (RIFE) — the weight-backed INTERP path that
outranks the ffmpeg last-resort once its assets are staged. IMPORT-SAFE and
GRACEFULLY-DEGRADING today: on this dev box (and any box where Practical-RIFE is
not yet vendored) preflight returns ``Err(StageError(...))`` as DATA, never raises.

    run_rife_interpolate(manifest, out_root, start_image=None, should_cancel=None)
        -> Result[Artifact, StageError]

WHY IT DEGRADES TODAY: RIFE (hzwer/Practical-RIFE) is NOT a pip/diffusers package —
it is a GitHub inference ARCH (an IFNet flow network + a small runner script) whose
weights (``flownet.pkl``) live on Google Drive / a GitHub release, not on PyPI or
the HF hub. There is nothing to ``pip install`` and nothing to lazily import here
today, so preflight reports DEPS_MISSING (the module/weights are not vendored) and
the runner is a graceful no-op until the box errand below is done.

    BOX ERRAND (to make this runner REAL, on the 4x3090 box):
      1. git clone https://github.com/hzwer/Practical-RIFE
      2. download the matching flownet weights (the repo's README links the current
         train_log/ set — e.g. the v4.x flownet.pkl) into Practical-RIFE/train_log/
      3. vendor the checkout so it is importable as the package below (or add it to
         PYTHONPATH), and place train_log/ under STUDIO_WEIGHTS_ROOT/hzwer/
         Practical-RIFE/ so the weights resolve next to the other studio weights.
      4. then wire the REAL path here (load IFNet, run pairwise 2x/4x interpolation
         to the manifest target fps, ffmpeg-assemble into the same content-addressed
         clip — same layout as ffmpeg_enhance / wan_vace).

PREFLIGHT ORDER (mirrors the enhancement runners): source (SPEC) -> RIFE module
(box capability). A source-less interp is a SPEC error on ANY box (SOURCE_MISSING),
reported before the box-capability DEPS_MISSING so it is not masked.

IMPORT SAFETY: stdlib only at module top. The Practical-RIFE import (torch + the
vendored arch) is LAZY, inside the runner, only after preflight passes — so today
it is never reached and this module never drags torch into app boot.

No pathlib anywhere. os.path only.
"""

from __future__ import annotations

import importlib.util
from typing import Callable

from hugpy_video.intel.studio.artifacts import Artifact
from hugpy_video.intel.studio.errors import Err, ErrorCode, Ok, Result, StageError
from hugpy_video.intel.studio.schemas import RenderManifest

# The dotted module the REAL path would import once Practical-RIFE is vendored. It
# is checked via find_spec (never actually imported) at preflight, so its absence is
# reported as DEPS_MISSING data rather than an ImportError at module top.
_RIFE_MODULE = "hugpy_video.intel.studio.runners._vendor.practical_rife"


from hugpy_video.intel.studio.runners.source import resolve_source as _resolve_source


def _rife_available() -> bool:
    """True iff the vendored Practical-RIFE arch is importable (find_spec, never
    imports). False on this box today — nothing is vendored yet."""
    try:
        return importlib.util.find_spec(_RIFE_MODULE) is not None
    except (ImportError, ValueError):
        return False


def run_rife_interpolate(
    manifest: RenderManifest,
    out_root: str,
    start_image: str | None = None,
    should_cancel: "Callable[[], bool] | None" = None,
) -> Result[Artifact, StageError]:
    """Produce (or resume) a RIFE-interpolated clip for ``manifest`` under
    ``out_root``. Returns ``Ok(Artifact)`` on the real path (once Practical-RIFE is
    vendored on the box), or ``Err(StageError)`` on any expected failure — including
    the graceful preflight failures that make this a no-op today (SOURCE_MISSING when
    the enhance carries no source clip, DEPS_MISSING when the Practical-RIFE arch is
    not vendored). Only a non-RenderManifest raises. ``start_image`` is part of the
    uniform runner signature but UNUSED (interpolation conditions on the whole clip)."""
    if not isinstance(manifest, RenderManifest):
        raise TypeError(
            f"manifest must be a RenderManifest; got {type(manifest).__name__}")

    # SOURCE-first (SPEC error, malformed on any box) — mirrors ffmpeg_enhance/wan_vace.
    source = _resolve_source(manifest)
    if source is None:
        return Err(StageError(
            ErrorCode.SOURCE_MISSING,
            "RIFE interpolation carries no source_video — a frame-interpolation "
            "enhance is defined by the clip it interpolates; supply source_video",
            (("model_id", manifest.model_id), ("capability", manifest.capability.value))))
    import os  # os.path only; local to keep the module top pathlib/stdlib-clean
    if not os.path.isfile(source):
        return Err(StageError(
            ErrorCode.SOURCE_MISSING,
            f"RIFE interpolation source_video not found on disk: {source}",
            (("source_video", source), ("model_id", manifest.model_id))))

    # BOX capability: the Practical-RIFE arch is not vendored/installed on this box.
    if not _rife_available():
        return Err(StageError(
            ErrorCode.DEPS_MISSING,
            "Practical-RIFE is not vendored on this box (hzwer/Practical-RIFE is a "
            "GitHub inference arch + flownet weights, not a pip/diffusers package). "
            "Clone github.com/hzwer/Practical-RIFE, fetch its train_log/ flownet "
            "weights, and vendor it as "
            f"{_RIFE_MODULE!r} (see the module docstring's BOX ERRAND). Until then "
            "the ffmpeg minterpolate last-resort serves INTERP.",
            (("model_id", manifest.model_id), ("weight_uri", "hzwer/Practical-RIFE"))))

    # REAL PATH (only once vendored on the box; never reached on this dev VM). Kept a
    # graceful Err so a partial vendor never crashes the studio — the executable RIFE
    # wiring lands with the box errand.
    return Err(StageError(
        ErrorCode.DEPS_MISSING,
        "Practical-RIFE arch is present but its executable interpolation path is not "
        "wired yet (box errand step 4); the ffmpeg minterpolate last-resort serves "
        "INTERP in the meantime.",
        (("model_id", manifest.model_id),)))
