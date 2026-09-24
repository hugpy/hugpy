"""REAL Wan VACE runner (B-3 + identity-lock + extend) — the studio's weight-backed
VACE executor over diffusers' unified control path, IMPORT-SAFE and GRACEFULLY-
DEGRADING now. It serves FOUR conditioning modes off the same pipeline
(``WanVACEPipeline``, diffusers 0.39):

  * v2v restyle (``source_video``): decode the source clip to per-frame control
    frames and drive ``video=`` (the historical B-3 path).
  * id_lock / reference-to-video (``reference_images``, capability ID_LOCK): load
    the subject reference image(s) as PIL and drive ``reference_images=`` — the
    identity is preserved across a freshly generated video (with no source, the
    pipeline zero-fills the control video + an all-ones mask = generate everything).
    Multiple references are ALL consumed (each prepended as a reference latent).
  * optional control still (``control_image`` + ``control_kind`` pose|depth|sketch):
    a single still repeated across the frame count as the ``video=`` control channel
    for composition blocking when there is no source_video.
  * VACE-EXTEND (``vace_context_frames``, studio-movie splice motion-carry): carry
    MOTION across a movie splice by conditioning on the parent clip's TRAILING frames
    instead of a single still. Drives the diffusers video+mask EXTEND idiom — ``video``
    is the kept context prefix + black placeholders, ``mask`` is black(keep) over the
    context and white(generate) over the rest, so the model continues the parent's
    motion rather than restarting from one frame.

Historically it only restyled/enhanced an existing clip (e.g. a movie-tier output).

    run_wan_vace(manifest, out_root, start_image=None, should_cancel=None)
        -> Result[Artifact, StageError]

It mirrors ``run_wan_i2v`` exactly — same signature, same content-addressed atomic
layout (``<out_root>/<content_hash>/clip.mp4`` + ``manifest.json`` +
``provenance.json``), same resume-on-hash (INV-6), same errors-as-data discipline
(INV-3), same bitsandbytes-quantized DiT (operator directive: "utilize
bitsandbytes"), same T1 cooperative-cancel wiring. The pure geometry / weights /
quant / deps helpers are REUSED from ``wan_i2v`` and the ffmpeg-assembly + sidecar
helpers from ``synthetic``, so the on-disk shape is byte-for-byte the same contract.

IMPORT SAFETY (hard requirement): torch / diffusers / transformers /
bitsandbytes are NEVER imported at module top — only lazily INSIDE the runner,
after preflight passes. Importing this module (or the studio package, or the
Flask app) pulls only stdlib + the studio's own light modules (numpy/PIL via the
reused ``synthetic``/``wan_i2v`` helpers), never the heavy GPU stack.

GRACEFUL DEGRADATION + errors-as-data preflight, in THIS order (returns
``Err(StageError(...))`` as DATA, never raises):
  * v2v render carries no / a nonexistent source_video       -> SOURCE_MISSING
  * missing torch/diffusers/transformers/bitsandbytes/accel  -> DEPS_MISSING
  * no CUDA device                                           -> NO_GPU
  * model weights not on disk under the weights root         -> WEIGHTS_MISSING
Only genuine programmer error (a non-RenderManifest) raises.

The SOURCE_MISSING check runs FIRST — a v2v render is DEFINED by the clip it
enhances, so a source-less request is a SPEC error that is malformed on ANY box
(GPU or not). Checking it before deps/GPU/weights means a source-less v2v is
reported as SOURCE_MISSING even on this GPU-less / bitsandbytes-less dev VM,
rather than masked by the box's DEPS_MISSING. (On this dev box a v2v render WITH a
real source degrades to DEPS_MISSING — bitsandbytes is absent — which is the
intended dev behavior: there is NO synthetic v2v stand-in, because enhancing a
real clip has no meaningful no-model equivalent; the graceful Err IS the answer.)

REAL PATH (runs only when preflight passes, i.e. on the 4x3090 box): loads the
diffusers ``WanVACEPipeline`` with a bitsandbytes-quantized
``WanVACETransformer3DModel`` (int8/nf4 per precision) + fp32 ``AutoencoderKLWan``
VAE, decodes the FULL source clip to control frames (ffmpeg, house style; resampled
to the manifest's frame count), runs VACE control (``video=`` conditioning) at the
manifest's resolution / frame-count / seed / sampler with the prompt/negative, and
ffmpeg-assembles the result into the same atomic content-addressed clip.

No pathlib anywhere. os.path only.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Callable

from hugpy_video.intel.studio.artifacts import Artifact
from hugpy_video.intel.studio.enums import Capability, Precision
from hugpy_video.intel.studio.errors import Err, ErrorCode, Ok, Result, StageError
from hugpy_video.intel.studio.manifest import render_manifest_to_dict
from hugpy_video.intel.studio.registry import MODEL_REGISTRY
from hugpy_video.intel.studio.schemas import RenderManifest
from hugpy_video.intel.studio.storage import atomic_write_text
# Reuse the synthetic runner's atomic/content-addressed plumbing so the VACE clip
# lands in the IDENTICAL on-disk layout (numpy/PIL house deps, NOT the heavy
# torch/diffusers stack — that stays lazy inside run_wan_vace).
from hugpy_video.intel.studio.runners.synthetic import (
    _CLIP_NAME,
    _MANIFEST_NAME,
    _PROVENANCE_NAME,
    _assemble_mp4,
    _provenance_dict,
)
# Reuse the Wan i2v runner's PURE helpers (no heavy deps): weights-root resolution,
# the 4k+1 temporal-cadence geometry snap, the precision->bitsandbytes quant map,
# and the find_spec-based dep probe. DRY — both Wan runners resolve weights /
# geometry / quant identically.
from hugpy_video.intel.studio.runners.wan_i2v import (
    _REQUIRED_DEPS,
    _bnb_config,
    _frame_to_pil,
    _hot_weights_root,
    _max_vram_gb,
    _missing_deps,
    _place_pipe,
    _prime_cuda_allocator,
    _resolve_model_dir,
    _should_place_whole_on_gpu,
    _wan_geometry,
    _weights_missing_msg,
    _weights_root,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Source-clip resolution (pure, no heavy deps) — the v2v INPUT
# --------------------------------------------------------------------------- #
from hugpy_video.intel.studio.runners.source import resolve_source as _resolve_source


# --------------------------------------------------------------------------- #
# Preflight — errors as data (returns a StageError to raise-as-Err, or None)
# --------------------------------------------------------------------------- #
def _preflight(manifest: RenderManifest) -> StageError | None:
    """Gate the real path. Returns a ``StageError`` (the caller wraps it in ``Err``)
    when this box can't run the render, or None when everything is present.

    ORDER: SPEC (conditioning input) -> deps -> GPU -> weights. The SPEC check is
    FIRST because a VACE render with no conditioning input is malformed on ANY box,
    and must surface as a spec error (SOURCE_MISSING / REFERENCE_MISSING) rather than
    be masked by a GPU-less box's DEPS_MISSING. The deps/GPU/weights checks then
    mirror ``wan_i2v._preflight``.

    A VACE render is DEFINED by at least one conditioning input:
      * ``source_video``        -> v2v restyle (the historical path);
      * ``reference_images``    -> id_lock (reference-to-video identity); REQUIRED for
        the ID_LOCK capability;
      * ``control_image``       -> a still control (pose/depth/sketch) for composition;
      * ``vace_context_frames`` -> VACE-EXTEND (studio-movie splice motion-carry): the
        parent clip's trailing frames as the KEPT prefix of a video+mask extension.
    A render carrying none of these is a spec error — REFERENCE_MISSING for an
    id_lock request (the flagship path), else SOURCE_MISSING (v2v / other VACE)."""
    # --- SPEC: at least one conditioning input; id_lock REQUIRES reference(s) ---
    source = _resolve_source(manifest)
    refs = tuple(getattr(manifest, "reference_images", ()) or ())
    control = getattr(manifest, "control_image", "") or ""
    ctx_frames = tuple(getattr(manifest, "vace_context_frames", ()) or ())
    is_id_lock = (manifest.capability == Capability.ID_LOCK)

    if is_id_lock and not refs:
        return StageError(
            ErrorCode.REFERENCE_MISSING,
            "id_lock (VACE reference-to-video) render carries no reference_images — "
            "an identity-locked render is defined by the subject reference image(s); "
            "supply at least one reference_image",
            (("model_id", manifest.model_id), ("capability", manifest.capability.value)),
        )
    if source is None and not refs and not control and not ctx_frames:
        return StageError(
            ErrorCode.REFERENCE_MISSING if is_id_lock else ErrorCode.SOURCE_MISSING,
            "VACE render carries no conditioning input — supply a source_video "
            "(v2v restyle), reference_images (id_lock), a control_image, or "
            "vace_context_frames (extend)",
            (("model_id", manifest.model_id), ("capability", manifest.capability.value)),
        )
    for f in ctx_frames:
        if not os.path.isfile(f):
            return StageError(
                ErrorCode.SOURCE_MISSING,
                f"VACE-extend context frame not found on disk: {f}",
                (("vace_context_frame", f), ("model_id", manifest.model_id)),
            )
    if source is not None and not os.path.isfile(source):
        return StageError(
            ErrorCode.SOURCE_MISSING,
            f"VACE source_video not found on disk: {source}",
            (("source_video", source), ("model_id", manifest.model_id)),
        )
    for r in refs:
        if not os.path.isfile(r):
            return StageError(
                ErrorCode.REFERENCE_MISSING,
                f"id_lock reference image not found on disk: {r}",
                (("reference_image", r), ("model_id", manifest.model_id)),
            )
    if control and not os.path.isfile(control):
        return StageError(
            ErrorCode.SOURCE_MISSING,
            f"VACE control_image not found on disk: {control}",
            (("control_image", control), ("model_id", manifest.model_id)),
        )

    # --- ENV: deps -> GPU -> weights (identical discipline to wan_i2v) ---
    missing = _missing_deps()
    if missing:
        return StageError(
            ErrorCode.DEPS_MISSING,
            "Wan VACE v2v needs GPU inference deps that are not installed: "
            + ", ".join(missing)
            + ". Install: pip install torch (CUDA build) diffusers transformers "
              "bitsandbytes accelerate",
            (("missing", ",".join(missing)),),
        )

    import torch  # lazy — only reached once torch is importable
    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    if not cuda_ok:
        return StageError(
            ErrorCode.NO_GPU,
            "no CUDA device available; Wan VACE v2v requires a CUDA GPU (the 4x3090 "
            "box) for bitsandbytes int8/nf4 inference",
            (("cuda", "unavailable"), ("model_id", manifest.model_id)),
        )

    cfg = MODEL_REGISTRY.get(manifest.model_id)
    if cfg is None:
        return StageError(
            ErrorCode.WEIGHTS_MISSING,
            f"model_id {manifest.model_id!r} is not in the studio registry",
            (("model_id", manifest.model_id),),
        )

    # WEIGHTS root resolution honors the box-local HOT NVMe copy first (item 5), then
    # the shared/snapshot root — identical discipline to wan_i2v (_resolve_model_dir).
    hot = _hot_weights_root()
    shared_root = _weights_root(manifest)
    if not hot and not shared_root:
        return StageError(
            ErrorCode.WEIGHTS_MISSING,
            "no weights root set — neither STUDIO_WEIGHTS_HOT_ROOT (box-local NVMe) "
            "nor STUDIO_WEIGHTS_ROOT is configured to resolve the Wan VACE weights "
            "against",
            (("model_id", manifest.model_id),),
        )

    model_dir, _root_used = _resolve_model_dir(manifest, cfg.weight_uri)
    if not model_dir or not (os.path.isdir(model_dir)
            and os.path.isfile(os.path.join(model_dir, "model_index.json"))):
        return StageError(
            ErrorCode.WEIGHTS_MISSING,
            "Wan VACE " + _weights_missing_msg(cfg.weight_uri, hot, shared_root),
            (("weight_uri", cfg.weight_uri),
             ("hot_root", hot or ""), ("shared_root", shared_root or "")),
        )
    return None


# --------------------------------------------------------------------------- #
# Full-clip control-frame decode (box-only; ffmpeg house style) — the VACE input
# --------------------------------------------------------------------------- #
def _read_control_frames(
    source_video: str, width: int, height: int, n_frames: int, frame_dir: str
) -> "tuple[list | None, str]":
    """Decode ``source_video``'s frames to PNGs (ffmpeg, mirroring
    ``synthetic._assemble_mp4`` / ``_extract_last_frame``: shutil.which, PIPE,
    returncode check — never raises on a plain ffmpeg failure), load them as RGB
    PIL images resized to ``(width, height)``, and RESAMPLE by nearest index to
    EXACTLY ``n_frames``.

    VACE's control video length must equal ``num_frames``, and ``num_frames`` is a
    pure function of the manifest (``_wan_geometry``, the 4k+1 snap) computed BEFORE
    any decode — so resume-on-hash stays valid regardless of the source clip's own
    frame count (unlike i2v, VACE consumes the WHOLE clip, not just its last frame).
    Returns ``(frames | None, stderr_tail)``; None signals an errors-as-data failure.
    """
    from PIL import Image  # house dep; lazy to honor the module's import discipline

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-y", "-i", source_video,
        "-vsync", "0",
        os.path.join(frame_dir, "src_%05d.png"),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return None, (result.stderr or "")[-500:]

    names = sorted(
        n for n in os.listdir(frame_dir)
        if n.startswith("src_") and n.endswith(".png"))
    if not names:
        return None, "ffmpeg decoded no frames from source_video"

    imgs: list = []
    for nm in names:
        with Image.open(os.path.join(frame_dir, nm)) as im:
            imgs.append(im.convert("RGB").resize((width, height), Image.LANCZOS))

    total = len(imgs)
    denom = max(1, n_frames - 1)
    picked = [imgs[min(total - 1, round(i * (total - 1) / denom))]
              for i in range(n_frames)]
    return picked, ""


# --------------------------------------------------------------------------- #
# VACE-EXTEND channels (pure; PIL-only, no heavy deps) — the video+mask idiom
# --------------------------------------------------------------------------- #
def _build_vace_extend_channels(context_pils, n_frames: int, width: int, height: int):
    """Build the diffusers ``WanVACEPipeline`` ``video`` + ``mask`` EXTEND channels from
    the parent's trailing context frames — carry MOTION across a movie splice by keeping
    the context frames and generating the rest. Returns ``(video, mask)``, each a list of
    PIL RGB frames of length ``n_frames``:

      * ``video`` = the K kept context frames (positions 0..K-1) + (n_frames-K) BLACK
        placeholders for the positions to GENERATE (their pixels are irrelevant — the
        mask marks them reactive, so the model synthesizes them).
      * ``mask``  = K BLACK frames (KEEP, mask value 0 -> the pipeline's "inactive"
        conditioning latents) + (n_frames-K) WHITE frames (GENERATE, mask value 1 ->
        "reactive").

    GROUNDED IN THE INSTALLED API (diffusers 0.39):
      * ``WanVACEPipeline.prepare_video_latents`` binarizes ``mask`` (``mask > 0.5 -> 1``)
        and splits ``inactive = video*(1-mask)`` (KEPT) vs ``reactive = video*mask``
        (GENERATED);
      * ``preprocess_conditions`` maps a PIL image ``[0,255] -> [-1,1] -> [0,1]``, so a
        BLACK mask frame lands at 0 (keep) and a WHITE mask frame at 1 (generate).
    Hence black=keep(context), white=generate. The caller guarantees ``0 < K < n_frames``.
    """
    from PIL import Image  # house dep; lazy to honor the module's import discipline

    k = len(context_pils)
    n_gen = n_frames - k
    black = Image.new("RGB", (width, height), (0, 0, 0))
    white = Image.new("RGB", (width, height), (255, 255, 255))
    video = list(context_pils) + [black] * n_gen   # kept context prefix + to-generate
    mask = [black] * k + [white] * n_gen           # keep the context, generate the rest
    return video, mask


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
def run_wan_vace(
    manifest: RenderManifest,
    out_root: str,
    start_image: str | None = None,
    should_cancel: "Callable[[], bool] | None" = None,
    on_step: "Callable[[int, int], None] | None" = None,
) -> Result[Artifact, StageError]:
    """Produce (or resume) a Wan VACE v2v clip for ``manifest`` under ``out_root``.

    Returns ``Ok(Artifact)`` on a real render (on the box), or ``Err(StageError)``
    on any expected failure — including the preflight failures that make this a
    graceful no-op on a source-less / GPU-less / weight-less box (SOURCE_MISSING /
    DEPS_MISSING / NO_GPU / WEIGHTS_MISSING). Only a genuine programmer error (a
    non-RenderManifest) raises.

    ``start_image`` is part of the shared runner contract (i2v conditioning still)
    but is UNUSED by VACE v2v — the conditioning here is the FULL source clip
    (``manifest.source_video``), decoded to a control video. It is accepted (and
    ignored) so the (framework, task) dispatch signature stays uniform.

    ``should_cancel`` is the OPTIONAL cooperative-cancel probe (T1): a zero-arg
    callable polled at the natural checkpoints (before load, between load and
    render, after render) and — during denoise — wired into the pipeline via
    diffusers' ``callback_on_step_end`` (the callback sets ``pipe._interrupt=True``
    so the loop breaks at the next step boundary). A cancel at any checked point
    returns ``Err(StageError(CANCELLED, ...))`` BEFORE any clip is written. TRUE
    mid-denoise interruption is BOX-ONLY — this GPU-less VM short-circuits at
    preflight, so the callback path only ever executes on the real box. None
    (default) = never cancel.

    ``on_step`` is the OPTIONAL denoise-progress sink (k57) — ``on_step(step, steps)``
    on each callback boundary, riding the same diffusers hook as the cancel probe.
    It is what makes the console's bar MOVE inside one long segment; best-effort, so
    a throwing sink can never break a render. None (default) = report nothing."""
    if not isinstance(manifest, RenderManifest):
        raise TypeError(
            f"manifest must be a RenderManifest; got {type(manifest).__name__}")

    content_hash = manifest.content_hash()
    width, height, fps, n_frames = _wan_geometry(manifest)
    out_dir = os.path.join(os.path.abspath(out_root), content_hash)
    clip_path = os.path.join(out_dir, _CLIP_NAME)

    # INV-6 resume FIRST: an existing non-empty clip is served as-is, with NO GPU
    # and NO reload — a box that rendered it can return it later even offline.
    if os.path.isfile(clip_path) and os.path.getsize(clip_path) > 0:
        return Ok(Artifact(
            path=clip_path, content_hash=content_hash, frames=n_frames,
            width=width, height=height, duration_s=n_frames / float(fps),
            resumed=True))

    # CUDA allocator defragmentation (item 7): set PYTORCH_CUDA_ALLOC_CONF BEFORE any
    # torch import (preflight below imports torch to probe CUDA). Shared helper with
    # wan_i2v. No-op + harmless on this GPU-less box.
    _prime_cuda_allocator()

    # PREFLIGHT: everything below the real path returns as DATA, never raises.
    # Order: source (spec) -> deps -> GPU -> weights (see _preflight).
    pf = _preflight(manifest)
    if pf is not None:
        return Err(pf)

    # ----------------------------------------------------------------------- #
    # REAL PATH — only reached on a box with a source clip + deps + CUDA + weights.
    # Never executes on the dev VM (preflight short-circuits above). Written
    # complete enough to run once the 4x3090 box is live.
    # ----------------------------------------------------------------------- #
    import torch
    from diffusers import (
        AutoencoderKLWan,
        BitsAndBytesConfig,
        UniPCMultistepScheduler,
        WanVACEPipeline,
        WanVACETransformer3DModel,
    )

    cfg = MODEL_REGISTRY.get(manifest.model_id)
    # WEIGHTS SOURCE (item 5): box-local hot NVMe copy first, else shared — a faster
    # LOAD only; does not affect content_hash.
    model_dir, weights_root_used = _resolve_model_dir(manifest, cfg.weight_uri)
    logger.info("wan vace: loading %s from %s (%s weights root)",
                cfg.weight_uri, model_dir, weights_root_used)
    source_video = _resolve_source(manifest)      # v2v restyle input (or None)
    # id_lock / control conditioning (preflight proved any supplied paths exist).
    reference_images = tuple(getattr(manifest, "reference_images", ()) or ())
    control_image = getattr(manifest, "control_image", "") or ""
    control_kind = getattr(manifest, "control_kind", "") or ""
    # VACE-EXTEND: the parent clip's trailing context frames (studio-movie splice
    # motion-carry). When present, the render is an EXTENSION — the context frames are
    # the KEPT prefix of a video+mask idiom, the rest is generated (see below).
    vace_context_frames = tuple(getattr(manifest, "vace_context_frames", ()) or ())
    compute_dtype = torch.bfloat16
    quant_config = _bnb_config(manifest.precision, BitsAndBytesConfig, torch)
    seed = manifest.seeds.global_seed
    steps = manifest.sampler.steps
    cfg_scale = manifest.sampler.cfg
    # C-prompt: text conditioning from the manifest (part of its content_hash). An
    # empty prompt is valid; an empty negative maps to None so the pipeline uses its
    # own default rather than an explicit "" negative.
    prompt = manifest.prompt
    negative_prompt = manifest.negative_prompt or None

    # PLACEMENT + SHIFT (shared decision with wan_i2v). Put a sub-envelope UNQUANTIZED
    # model wholly on the GPU; a bnb-quantized (INT8/FP8) VACE precision offloads (never
    # .to() a quantized pipeline). flow_shift is the manifest's recorded scheduler shift.
    model_gb = cfg.vram.as_map().get(manifest.precision)
    place_whole = _should_place_whole_on_gpu(
        manifest.precision, model_gb, _max_vram_gb(manifest),
        # Pass the identity + geometry so the decision uses MEASURED component bytes
        # (DiT + UMT5 + VAE [+ CLIP] + token-scaled activations) instead of the
        # registry's DiT-only number plus a flat fudge. Without these it silently
        # falls back to the legacy test.
        model_id=manifest.model_id, width=width, height=height, n_frames=n_frames)
    flow_shift = manifest.sampler.shift

    # Cooperative mid-render cancel wiring (T1). diffusers 0.39's
    # WanVACEPipeline.__call__ supports `callback_on_step_end`; the callback sets
    # `pipe._interrupt=True` so the denoise loop breaks at the next step boundary.
    # We ALSO re-check should_cancel() around the call. BOX-ONLY (preflight
    # short-circuits the GPU-less VM above).
    from hugpy_video.intel.studio.runners.cancel import cancel_step_callback
    _cancel_step_cb = cancel_step_callback(should_cancel, on_step, steps)

    call_extra: dict = {}
    if should_cancel is not None or on_step is not None:
        call_extra["callback_on_step_end"] = _cancel_step_cb

    src_frame_dir = None
    out_frame_dir = None
    tmp_mp4 = None
    try:
        os.makedirs(out_dir, exist_ok=True)

        # Cooperative cancel — BEFORE load (no weights touched yet if we bail).
        if should_cancel is not None and should_cancel():
            return Err(StageError(
                ErrorCode.CANCELLED, "cancelled before wan vace load",
                (("content_hash", content_hash), ("model_id", manifest.model_id))))

        # Build the VACE conditioning channels BEFORE loading multi-GB weights so a bad
        # input fails fast (errors-as-data). All inputs are in the manifest (content_hash),
        # so this is deterministic + resume-safe.
        #   * `video=` control channel: the source clip's per-frame frames (v2v restyle),
        #     OR a single control still (pose/depth/sketch) REPEATED across the frame
        #     count for composition blocking, OR omitted (pure reference-to-video, where
        #     the pipeline zero-fills the control video and generates everything).
        #   * `reference_images=` identity channel: the subject reference PIL image(s),
        #     each prepended by the pipeline as a VACE reference latent (diffusers 0.39
        #     WanVACEPipeline.__call__; ALL supplied references are consumed).
        from PIL import Image
        vace_call: dict = {}
        control_frames_repeated = 0
        extend_context_frames = 0
        extend_generated_frames = 0
        if vace_context_frames:
            # VACE-EXTEND: carry motion across a movie splice by CONDITIONING on the
            # parent's trailing frames instead of a single still. Load the K context frames
            # and build the diffusers video+mask extend channels (see
            # _build_vace_extend_channels for the installed-API grounding).
            K = len(vace_context_frames)
            if K >= n_frames:
                return Err(StageError(
                    ErrorCode.IO_ERROR,
                    f"vace-extend context ({K} frames) leaves no room to generate within "
                    f"num_frames={n_frames}",
                    (("content_hash", content_hash), ("model_id", manifest.model_id))))
            ctx_pils: list = []
            for f in vace_context_frames:
                try:
                    ctx_pils.append(
                        Image.open(f).convert("RGB").resize((width, height), Image.LANCZOS))
                except Exception as exc:  # noqa: BLE001 — bad context frame is input data
                    return Err(StageError(
                        ErrorCode.IO_ERROR,
                        f"could not load vace-extend context frame {f}: {exc}",
                        (("vace_context_frame", f),)))
            vace_call["video"], vace_call["mask"] = _build_vace_extend_channels(
                ctx_pils, n_frames, width, height)
            extend_context_frames = K
            extend_generated_frames = n_frames - K
        elif source_video:
            src_frame_dir = tempfile.mkdtemp(prefix=".srcframes-", dir=out_dir)
            control_frames, stderr_tail = _read_control_frames(
                source_video, width, height, n_frames, src_frame_dir)
            if control_frames is None:
                return Err(StageError(
                    ErrorCode.IO_ERROR,
                    f"could not decode control frames from source_video: {stderr_tail}",
                    (("source_video", source_video),)))
            vace_call["video"] = control_frames
        elif control_image:
            try:
                ctrl_pil = Image.open(control_image).convert("RGB").resize(
                    (width, height), Image.LANCZOS)
            except Exception as exc:  # noqa: BLE001 — bad control still is input data
                return Err(StageError(
                    ErrorCode.IO_ERROR,
                    f"could not load control_image ({control_kind}): {exc}",
                    (("control_image", control_image),)))
            # A single still repeated across the frame count = a STATIC composition
            # anchor (the VACE `video=` control expects a per-frame list). Recorded in
            # provenance so the static-vs-dynamic control is never silent.
            vace_call["video"] = [ctrl_pil] * n_frames
            control_frames_repeated = n_frames

        reference_pils: list = []
        if reference_images:
            for r in reference_images:
                try:
                    reference_pils.append(Image.open(r).convert("RGB"))
                except Exception as exc:  # noqa: BLE001 — bad reference is input data
                    return Err(StageError(
                        ErrorCode.REFERENCE_MISSING,
                        f"could not load reference image: {r} ({exc})",
                        (("reference_image", r),)))
            vace_call["reference_images"] = reference_pils

        # bitsandbytes-quantized VACE DiT transformer (int8 / nf4 per precision).
        tf_kwargs = {"subfolder": "transformer", "torch_dtype": compute_dtype}
        if quant_config is not None:
            tf_kwargs["quantization_config"] = quant_config
        transformer = WanVACETransformer3DModel.from_pretrained(model_dir, **tf_kwargs)
        # Wan's VAE is numerically sensitive; the diffusers Wan reference loads it in
        # fp32 (small relative to the DiT, so this is affordable).
        vae = AutoencoderKLWan.from_pretrained(
            model_dir, subfolder="vae", torch_dtype=torch.float32)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Cooperative cancel — BETWEEN load and render (weights loaded, nothing
        # rendered/written yet). Per-step interruption is handled by the callback.
        if should_cancel is not None and should_cancel():
            return Err(StageError(
                ErrorCode.CANCELLED, "cancelled after wan vace load, before render",
                (("content_hash", content_hash), ("model_id", manifest.model_id))))

        pipe = WanVACEPipeline.from_pretrained(
            model_dir, transformer=transformer, vae=vae, torch_dtype=compute_dtype)
        # SHIFT: wire the manifest's flow-match/UniPC shift into the scheduler so the
        # denoise uses exactly the recorded value (INV-1). None leaves the default.
        if flow_shift is not None:
            try:
                pipe.scheduler = UniPCMultistepScheduler.from_config(
                    pipe.scheduler.config, flow_shift=flow_shift)
            except Exception:
                pass  # diffusers build without flow_shift on UniPC — keep the default
        # PLACEMENT: whole pipeline to GPU when it fits unquantized, else offload +
        # engage the VRAM levers (item 4, shared _place_pipe). A bnb-quantized
        # (INT8/FP8) precision ALWAYS offloads (never .to() a quantized pipeline —
        # _should_place_whole_on_gpu returns False for INT8/FP8).
        _place_pipe(pipe, place_whole)

        # VACE unified control (diffusers 0.39 WanVACEPipeline.__call__): the built
        # `video=` control channel and/or `reference_images=` identity channel are
        # threaded via vace_call. With neither, the pipeline zero-fills the control
        # video + an all-ones mask (generate everything). output_type="pil" so
        # result.frames[0] is a list of PIL.Image for the save loop below.
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=n_frames,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
            output_type="pil",
            **vace_call,
            **call_extra,
        )

        # Cooperative cancel — AFTER render: if the callback interrupted the denoise
        # loop (pipe._interrupt), the pipeline still returns partial frames. Abort
        # here, BEFORE assembling/writing, so no clip lands at the addressed path.
        if should_cancel is not None and should_cancel():
            return Err(StageError(
                ErrorCode.CANCELLED, "cancelled mid-denoise (interrupted)",
                (("content_hash", content_hash), ("model_id", manifest.model_id))))

        # diffusers video pipelines return frames as result.frames[0]. We request
        # output_type="pil" but the per-frame type varies by pipeline/version
        # (wan_i2v got ndarray on ae 2026-07-07 and PIL-only .save() failed after
        # a full denoise) — normalize per-frame, same belt as wan_i2v.
        frames = result.frames[0]
        actual_frames = len(frames)

        out_frame_dir = tempfile.mkdtemp(prefix=".frames-", dir=out_dir)
        for i, fr in enumerate(frames):
            _frame_to_pil(fr).save(
                os.path.join(out_frame_dir, f"frame_{i:05d}.png"), "PNG")

        # Same atomic ffmpeg assembly + promotion as the synthetic / i2v runners.
        tmp_mp4 = os.path.join(out_dir, f".clip-tmp-{os.getpid()}.mp4")
        ok, stderr_tail = _assemble_mp4(out_frame_dir, tmp_mp4, fps)
        if not ok:
            return Err(StageError(
                ErrorCode.ASSEMBLY_FAILED,
                f"ffmpeg mux failed: {stderr_tail}",
                (("content_hash", content_hash), ("frames", str(actual_frames))),
            ))

        os.replace(tmp_mp4, clip_path)        # atomic promotion of the clip
        tmp_mp4 = None

        atomic_write_text(
            os.path.join(out_dir, _MANIFEST_NAME),
            json.dumps(render_manifest_to_dict(manifest), indent=2, sort_keys=True))
        # Provenance records WHICH weights root served (hot NVMe vs shared, item 5) and
        # the id_lock conditioning HONESTY (item: identity-lock) — sidecar-only fields;
        # NOT canonical inputs, never in content_hash.
        prov = _provenance_dict(manifest)
        prov["weights_root_used"] = weights_root_used
        # Multi-reference honesty: diffusers 0.39 consumes ALL supplied references (each
        # prepended as a reference latent), so supplied == consumed here. Recorded so a
        # future version that consumes fewer is caught by the drift, never silent.
        prov["reference_images_supplied"] = len(reference_images)
        prov["reference_images_consumed"] = len(reference_pils)
        if control_image:
            # Static control still repeated across the frame count (composition anchor):
            # recorded so the static-vs-per-frame control semantics are never silent.
            prov["control_kind"] = control_kind
            prov["control_frames_repeated"] = control_frames_repeated
        if vace_context_frames:
            # VACE-EXTEND honesty: record HOW the splice motion was carried — how many
            # parent frames were KEPT as the conditioning prefix (mask=0) and how many
            # frames were GENERATED (mask=1). Sidecar-only; never a content-hash input.
            prov["vace_extend"] = True
            prov["extend_context_frames"] = extend_context_frames
            prov["extend_generated_frames"] = extend_generated_frames
        atomic_write_text(
            os.path.join(out_dir, _PROVENANCE_NAME),
            json.dumps(prov, indent=2, sort_keys=True))
    except Exception as exc:  # inference/IO failure rides back as data (INV-3)
        name = type(exc).__name__
        is_oom = "OutOfMemory" in name or "out of memory" in str(exc).lower()
        return Err(StageError(
            ErrorCode.OOM if is_oom else ErrorCode.IO_ERROR,
            f"wan vace v2v {'ran out of VRAM' if is_oom else 'inference failed'}: {exc}",
            (("content_hash", content_hash), ("model_id", manifest.model_id)),
        ))
    finally:
        if tmp_mp4 is not None and os.path.isfile(tmp_mp4):
            try:
                os.remove(tmp_mp4)
            except OSError:
                pass
        for d in (src_frame_dir, out_frame_dir):
            if d is not None and os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)

    return Ok(Artifact(
        path=clip_path, content_hash=content_hash, frames=actual_frames,
        width=width, height=height, duration_s=actual_frames / float(fps),
        resumed=False))
