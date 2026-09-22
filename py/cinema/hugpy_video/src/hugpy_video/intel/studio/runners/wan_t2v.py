"""REAL Wan t2v runner (Task 3b) — the text-to-video twin of ``run_wan_i2v``.

Text-to-video is the SAME weight-backed Wan path as i2v with NO conditioning
still: ``run_wan_i2v`` already takes its diffusers ``WanPipeline`` (t2v) branch
whenever ``start_image`` is None (i2v uses ``WanImageToVideoPipeline``). So
``run_wan_t2v`` is a thin, DRY delegation that forces ``start_image=None`` — it
inherits, byte-for-byte, every guarantee of ``run_wan_i2v``:

  * IMPORT SAFETY — torch/diffusers/transformers/bitsandbytes are imported LAZILY
    inside ``run_wan_i2v``, never at module top, so importing this module (or the
    studio package, or the Flask app) never drags in the heavy GPU stack.
  * GRACEFUL DEGRADATION — the ``DEPS_MISSING`` / ``NO_GPU`` / ``WEIGHTS_MISSING``
    preflight returns as DATA (an ``Err(StageError)``), never a raise, on a box
    that can't run Wan yet (this dev VM lacks bitsandbytes -> DEPS_MISSING).
  * REAL PATH — bitsandbytes int8/nf4 precision mapping, ``manifest.prompt`` /
    ``negative_prompt`` conditioning, ``should_cancel`` checkpoints wired into
    diffusers' ``callback_on_step_end`` interrupt, and the atomic
    content-addressed clip output (``<out_root>/<content_hash>/clip.mp4`` +
    ``manifest.json`` + ``provenance.json``). All of it is ``run_wan_i2v``'s
    already-complete WanPipeline (t2v) branch.

T2V is TEXT-ONLY: a start_image is meaningless, so any supplied still is
DELIBERATELY IGNORED (dropped to None) rather than used as conditioning — a t2v
clip is a pure function of prompt + seed + geometry.

No pathlib anywhere. os.path only (there is none here — pure delegation).
"""

from __future__ import annotations

from typing import Callable

from hugpy_video.intel.studio.artifacts import Artifact
from hugpy_video.intel.studio.errors import Result, StageError
from hugpy_video.intel.studio.schemas import RenderManifest
from hugpy_video.intel.studio.runners.wan_i2v import run_wan_i2v


def run_wan_t2v(
    manifest: RenderManifest,
    out_root: str,
    start_image: str | None = None,
    should_cancel: "Callable[[], bool] | None" = None,
    on_step: "Callable[[int, int], None] | None" = None,
) -> Result[Artifact, StageError]:
    """Produce (or resume) a Wan TEXT-to-video clip for ``manifest`` under
    ``out_root``.

    Delegates to ``run_wan_i2v`` with ``start_image`` forced to None so it takes
    the ``WanPipeline`` (t2v) branch. Any supplied ``start_image`` is IGNORED
    (t2v is text-only). Returns ``Ok(Artifact)`` on a real render (on the box), or
    a graceful ``Err(StageError)`` (DEPS_MISSING / NO_GPU / WEIGHTS_MISSING) on a
    box that can't run Wan yet. Only a genuine programmer error (a non-
    RenderManifest) raises — inherited from ``run_wan_i2v``.

    ``on_step`` (the denoise-progress sink, k57) is forwarded like ``should_cancel``
    — a t2v movie segment is exactly the long single render whose bar would
    otherwise never move."""
    return run_wan_i2v(
        manifest, out_root, start_image=None, should_cancel=should_cancel,
        on_step=on_step)
