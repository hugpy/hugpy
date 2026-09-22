"""REAL Wan i2v runner (P0-6) — the first weight-backed executor behind the
studio's runner contract, structurally complete and plug-in ready for the
4x3090 box (bitsandbytes int8/nf4), IMPORT-SAFE and GRACEFULLY-DEGRADING now.

It mirrors ``synthetic.run_synthetic_i2v`` exactly:

    run_wan_i2v(manifest, out_root, start_image=None) -> Result[Artifact, StageError]

Same content-addressed atomic layout (``<out_root>/<content_hash>/clip.mp4`` +
``manifest.json`` + ``provenance.json``), same resume-on-hash, same errors-as-data
discipline (INV-3/INV-6). The ffmpeg assembly + sidecar helpers are REUSED from
``synthetic`` so the on-disk shape is byte-for-byte the same contract.

IMPORT SAFETY (hard requirement): torch / diffusers / transformers /
bitsandbytes are NEVER imported at module top — only lazily INSIDE the runner,
after preflight passes. Importing this module (or the studio package, or the
Flask app) pulls only stdlib + the studio's own light modules, so app boot never
drags in the heavy GPU stack and never fails on a box without it.

GRACEFUL DEGRADATION (this dev VM has NO GPU / NO weights): preflight returns
``Err(StageError(...))`` as DATA, never raises:
  * missing torch/diffusers/transformers/bitsandbytes/accelerate -> DEPS_MISSING
  * no CUDA device                                               -> NO_GPU
  * model weights not on disk under the weights root             -> WEIGHTS_MISSING
Only genuine programmer error (a non-RenderManifest) raises.

REAL PATH (runs only when preflight passes, i.e. on the box): loads the Wan i2v
pipeline via diffusers with a bitsandbytes quantized transformer (operator
directive: "utilize bitsandbytes"), runs i2v from ``start_image`` (or t2v when
None) at the manifest's resolution / frame-count / seed / sampler, writes the
frames, and ffmpeg-assembles them into the same atomic content-addressed clip.
Diffusers pipeline classes used: ``WanImageToVideoPipeline`` (i2v) /
``WanPipeline`` (t2v), ``WanTransformer3DModel`` (bnb-quantized),
``AutoencoderKLWan`` (fp32 VAE), ``diffusers.BitsAndBytesConfig``.

No pathlib anywhere. os.path only.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from typing import Callable

from hugpy_video.intel.studio.artifacts import Artifact
from hugpy_video.intel.studio.enums import Precision, Task
from hugpy_video.intel.studio.errors import Err, ErrorCode, Ok, Result, StageError
from hugpy_video.intel.studio.manifest import render_manifest_to_dict
from hugpy_video.intel.studio.registry import MODEL_REGISTRY
from hugpy_video.intel.studio.schemas import RenderManifest
from hugpy_video.intel.studio.storage import atomic_write_text
# Reuse the synthetic runner's atomic/content-addressed plumbing so the Wan clip
# lands in the IDENTICAL on-disk layout. These pull numpy/PIL (already house
# deps, present everywhere) — NOT the heavy torch/diffusers stack, which stays
# lazy inside run_wan_i2v. ``_extract_last_frame`` + ``_SOURCE_LASTFRAME_NAME`` are
# the B2 "extend the movie" helpers, shared so Wan extracts the source clip's last
# frame byte-identically to the synthetic prover.
from hugpy_video.intel.studio.runners.synthetic import (
    _CLIP_NAME,
    _MANIFEST_NAME,
    _PROVENANCE_NAME,
    _SOURCE_LASTFRAME_NAME,
    _assemble_mp4,
    _extract_last_frame,
    _geometry,
    _provenance_dict,
)

logger = logging.getLogger(__name__)

# Python deps the REAL inference path needs. Preflight reports any that are
# absent as DEPS_MISSING data (never an ImportError at module import).
# ``ftfy`` is required because the Wan pipelines' prompt-clean path imports it (the
# i2v/t2v denoise calls ``ftfy.fix_text`` on the prompt); a box missing it OOM-free
# used to surface it as a mid-load IO_ERROR AFTER loading ~14GB of weights (live
# 2026-07-07) — listing it here makes it an honest DEPS_MISSING at PREFLIGHT instead.
_REQUIRED_DEPS = (
    "torch", "diffusers", "transformers", "bitsandbytes", "accelerate", "ftfy")


# --------------------------------------------------------------------------- #
# Weights / geometry resolution (pure, no heavy deps)
# --------------------------------------------------------------------------- #
def _weights_root(manifest: RenderManifest) -> str | None:
    """The weights root, sourced FIRST from the manifest's captured env_snapshot
    (``STUDIO_WEIGHTS_ROOT`` was threaded there by ``env.to_snapshot()`` at build
    time, INV-5), falling back to the live process env. None if neither is set."""
    snap = dict(manifest.env_snapshot)
    return snap.get("STUDIO_WEIGHTS_ROOT") or os.environ.get("STUDIO_WEIGHTS_ROOT")


def _local_model_dir(weights_root: str, weight_uri: str) -> str:
    """Local on-disk dir for an HF-style ``org/name`` weight_uri, mirrored under
    the weights root (``<weights_root>/<org>/<name>``)."""
    parts = [p for p in weight_uri.split("/") if p]
    return os.path.join(weights_root, *parts)


# --------------------------------------------------------------------------- #
# HOT weights root (item 5) — a per-box NVMe copy that loads faster than the shared
# /mnt/llm_storage mount. Box-local ONLY; NEVER a canonical input.
# --------------------------------------------------------------------------- #
_HOT_WEIGHTS_ROOT_ENV = "STUDIO_WEIGHTS_HOT_ROOT"


def _hot_weights_root() -> str | None:
    """A per-box NVMe hot-copy weights root, read from the LOCAL PROCESS ENV ONLY
    (``STUDIO_WEIGHTS_HOT_ROOT``) — deliberately NOT from the manifest env_snapshot,
    which is captured on central and must not dictate a box-local path. None if unset
    or empty.

    The hot copy is a faster LOAD SOURCE only: it is never written into the manifest /
    env_snapshot, so it CANNOT change a clip's content_hash (weights come from the
    same bytes wherever they load from). Central builds the manifest without ever
    seeing this var; the worker resolves it here at render time."""
    root = os.environ.get(_HOT_WEIGHTS_ROOT_ENV)
    return root or None


def _resolve_model_dir(manifest: RenderManifest, weight_uri: str) -> tuple[str | None, str]:
    """Resolve the on-disk model dir for ``weight_uri`` PLUS a tag for WHICH root
    served it. Order (box-local NVMe hot copy first, then the shared/snapshot root):

      1. ``STUDIO_WEIGHTS_HOT_ROOT`` set AND
         ``<hot>/<org>/<name>/model_index.json`` present -> (``<hot_dir>``, "hot");
      2. hot root set but no hot copy -> COPY-ON-FIRST-USE (operator tiering
         doctrine 2026-08-13: llm_storage is the fleet's cold tier by contract,
         the worker drive is the hot tier; the one-time copy wait is the accepted
         price of every later load coming off local NVMe). The copy activates
         ATOMICALLY and evicts least-recently-used hot copies when the disk is
         short — see _ensure_hot_copy. Any shortfall falls through to…
      3. (``<shared_dir>`` | None, "shared") — ``_local_model_dir`` over the
         shared weights root from the manifest snapshot (or process env),
         UNCHANGED; None when no shared root is configured.

    The hot presence gate is ``model_index.json`` (the same completeness gate the
    shared preflight uses), so a partial / in-flight hot copy transparently falls back
    to the shared store rather than loading half a model."""
    hot = _hot_weights_root()
    shared_root = _weights_root(manifest)
    if hot:
        hot_dir = _local_model_dir(hot, weight_uri)
        if os.path.isfile(os.path.join(hot_dir, "model_index.json")):
            _touch_hot_used(hot_dir)
            return hot_dir, "hot"
        if shared_root:
            shared_dir = _local_model_dir(shared_root, weight_uri)
            if os.path.isfile(os.path.join(shared_dir, "model_index.json")):
                warmed = _ensure_hot_copy(hot, shared_dir, weight_uri)
                if warmed:
                    _touch_hot_used(warmed)
                    return warmed, "hot"
    if not shared_root:
        return None, "shared"
    return _local_model_dir(shared_root, weight_uri), "shared"


# ── hot-tier copy-on-first-use + LRU eviction (operator doctrine 2026-08-13) ──
# "It should evict models from that drive and rsync from central storage when
# the hot drive needs a model it doesn't have — the one-time wait is the
# tradeoff." Rules encoded here:
#   * ATOMIC activation: copy into <name>.partial-<pid>, os.rename to <name>
#     only when complete — the model_index.json gate above must NEVER see a
#     half-copied dir as servable.
#   * LRU eviction: only COMPLETE model dirs under the hot root are candidates
#     (a .partial being written by a concurrent warm is skipped), ordered by
#     the .hot-last-used marker _touch_hot_used maintains (dir mtime fallback),
#     never the model being warmed.
#   * SAFETY: if the hot root and shared root resolve to the same filesystem
#     path (misconfiguration), copy AND eviction are disabled outright —
#     eviction must never be able to delete canonical weights.
#   * Best-effort throughout: any failure cleans up its partial dir and falls
#     back to the shared root — slower, never broken.
_HOT_EVICT_MARGIN_BYTES = 10 * 2**30
_HOT_USED_MARKER = ".hot-last-used"


def _touch_hot_used(hot_dir: str) -> None:
    try:
        with open(os.path.join(hot_dir, _HOT_USED_MARKER), "w") as fh:
            fh.write(str(time.time()))
    except OSError:
        pass


def _hot_last_used(hot_dir: str) -> float:
    try:
        return os.path.getmtime(os.path.join(hot_dir, _HOT_USED_MARKER))
    except OSError:
        try:
            return os.path.getmtime(hot_dir)
        except OSError:
            return 0.0


def _dir_bytes(path: str) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def _hot_copy_candidates(hot_root: str, keep_dir: str) -> list[str]:
    """COMPLETE hot copies (never partials, never ``keep_dir``), LRU first."""
    out: list[str] = []
    try:
        for org in os.listdir(hot_root):
            org_dir = os.path.join(hot_root, org)
            if not os.path.isdir(org_dir):
                continue
            for name in os.listdir(org_dir):
                d = os.path.join(org_dir, name)
                if (os.path.isdir(d) and ".partial-" not in name
                        and os.path.realpath(d) != os.path.realpath(keep_dir)
                        and os.path.isfile(os.path.join(d, "model_index.json"))):
                    out.append(d)
    except OSError:
        return []
    return sorted(out, key=_hot_last_used)


def _ensure_hot_copy(hot_root: str, shared_dir: str, weight_uri: str) -> "str | None":
    """Warm ``weight_uri`` from the shared (cold) store into the hot root.
    Returns the hot dir on success, None on any shortfall (caller falls back to
    the shared store — functional, just slower)."""
    import shutil

    hot_dir = _local_model_dir(hot_root, weight_uri)
    # Misconfiguration guard: hot inside/equal to shared (or vice versa) would
    # let eviction delete canonical weights. Refuse the whole feature.
    hr, sr = os.path.realpath(hot_root), os.path.realpath(os.path.dirname(
        os.path.dirname(shared_dir)))
    if hr == sr or hr.startswith(sr + os.sep) or sr.startswith(hr + os.sep):
        logger.warning("hot-copy disabled: hot root %s overlaps shared root %s",
                       hot_root, sr)
        return None
    try:
        need = _dir_bytes(shared_dir)
        if need <= 0:
            return None
        st = os.statvfs(hot_root)
        free = st.f_bavail * st.f_frsize
        # Evict LRU complete copies until the copy fits (plus margin).
        if free < need + _HOT_EVICT_MARGIN_BYTES:
            for victim in _hot_copy_candidates(hot_root, hot_dir):
                logger.info("hot-copy: evicting LRU %s to fit %s", victim, weight_uri)
                shutil.rmtree(victim, ignore_errors=True)
                st = os.statvfs(hot_root)
                free = st.f_bavail * st.f_frsize
                if free >= need + _HOT_EVICT_MARGIN_BYTES:
                    break
        if free < need + _HOT_EVICT_MARGIN_BYTES:
            logger.warning("hot-copy: %s needs %.1fGiB but only %.1fGiB free after "
                           "eviction — loading from the shared store this time",
                           weight_uri, need / 2**30, free / 2**30)
            return None
        # Sweep stale partials (a crashed earlier warm), then copy atomically.
        parent = os.path.dirname(hot_dir)
        os.makedirs(parent, exist_ok=True)
        for entry in os.listdir(parent):
            if entry.startswith(os.path.basename(hot_dir) + ".partial-"):
                shutil.rmtree(os.path.join(parent, entry), ignore_errors=True)
        tmp = f"{hot_dir}.partial-{os.getpid()}"
        t0 = time.time()
        logger.info("hot-copy: warming %s (%.1fGiB) shared->hot — one-time cost, "
                    "every later load reads local NVMe", weight_uri, need / 2**30)
        shutil.copytree(shared_dir, tmp)
        if os.path.isdir(hot_dir):        # concurrent warm won the race
            shutil.rmtree(tmp, ignore_errors=True)
            return hot_dir if os.path.isfile(
                os.path.join(hot_dir, "model_index.json")) else None
        os.rename(tmp, hot_dir)
        logger.info("hot-copy: %s warmed in %.0fs", weight_uri, time.time() - t0)
        return hot_dir
    except Exception as exc:  # noqa: BLE001 — never fail a render over the cache
        logger.warning("hot-copy: warm of %s failed (%s) — loading from the "
                       "shared store this time", weight_uri, exc)
        try:
            shutil.rmtree(f"{hot_dir}.partial-{os.getpid()}", ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
        return None


def _weights_missing_msg(weight_uri: str, hot: str | None, shared_root: str | None) -> str:
    """A WEIGHTS_MISSING message that names BOTH roots that were tried (item 5), so a
    box operator can see whether the hot NVMe copy, the shared mount, or both are
    absent."""
    tried: list[str] = []
    if hot:
        tried.append("hot NVMe " + _local_model_dir(hot, weight_uri))
    if shared_root:
        tried.append("shared " + _local_model_dir(shared_root, weight_uri))
    where = "; ".join(tried) if tried else "no weights root configured"
    dl_root = shared_root or hot or "<weights_root>"
    return (f"weights not found on disk for {weight_uri} (looked in: {where}); "
            f"download with `huggingface-cli download {weight_uri} "
            f"--local-dir {_local_model_dir(dl_root, weight_uri)}`")


# Wan spatial grid: the VAE compresses space 8:1 and the DiT patchifies 2x2 latent
# pixels, so height/width must be multiples of 8*2 = 16 (diffusers' own check is
# ``vae_scale_factor_spatial * patch_size``). Temporal: 4:1 + 1 -> num_frames = 4k+1.
_WAN_SPATIAL_MULTIPLE = 16
_WAN_TEMPORAL_STRIDE = 4


def _snap_geometry(width: int, height: int, n_frames: int) -> tuple[int, int, int]:
    """Pure snap of a requested (w, h, frames) onto Wan's grid: w/h DOWN to the
    nearest multiple of 16 (floor 16), frames DOWN to the nearest 4k+1 (floor 1).
    Snapping down never grows the VRAM need of a geometry the router already priced."""
    m = _WAN_SPATIAL_MULTIPLE
    w = max(m, (int(width) // m) * m)
    h = max(m, (int(height) // m) * m)
    n = max(1, int(n_frames))
    n = ((n - 1) // _WAN_TEMPORAL_STRIDE) * _WAN_TEMPORAL_STRIDE + 1
    return w, h, n


def _wan_geometry(manifest: RenderManifest) -> tuple[int, int, int, int]:
    """(width, height, fps, n_frames) mirroring synthetic's ``_geometry`` but
    snapped to Wan's grid (``_snap_geometry``): the latent VAE compresses time 4:1,
    so the pipeline requires ``num_frames == 4*k + 1`` (e.g. 81), and space 8:1
    under a 2x2 patch, so height/width must be multiples of 16. Snapping here (not
    in the real path) keeps the resume check and the generation call agreeing on
    the exact geometry. Anything snapped is logged ONCE, explicitly."""
    width, height, fps, n = _geometry(manifest)
    w, h, nn = _snap_geometry(width, height, n)
    if (w, h, nn) != (width, height, n):
        logger.warning(
            "wan geometry snapped to the model grid: requested %dx%d x%df -> "
            "%dx%d x%df (width/height -> multiple of %d, frames -> 4k+1)",
            width, height, n, w, h, nn, _WAN_SPATIAL_MULTIPLE)
    return w, h, fps, nn


def _fit_image(img, width: int, height: int):
    """Resize a PIL conditioning image to EXACTLY (width, height): center-crop to
    the target aspect first (no distortion), then LANCZOS resample. The pipeline
    does its own preprocess too, but feeding it the exact grid size means the image
    latent and the noise latent can never disagree on spatial shape."""
    from PIL import Image
    img = img.convert("RGB")
    sw, sh = img.size
    if (sw, sh) == (width, height):
        return img
    target = width / float(height)
    src = sw / float(sh)
    if abs(src - target) > 1e-6:
        if src > target:                      # too wide -> crop width
            new_w = int(round(sh * target))
            left = (sw - new_w) // 2
            img = img.crop((left, 0, left + new_w, sh))
        else:                                 # too tall -> crop height
            new_h = int(round(sw / target))
            top = (sh - new_h) // 2
            img = img.crop((0, top, sw, top + new_h))
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    return img.resize((width, height), resample)


# --------------------------------------------------------------------------- #
# Checkpoint <-> task pairing (2026-08-21 incident) and resolution-variant pick
# --------------------------------------------------------------------------- #
# The Wan 2.1 i2v DiT takes 36 latent channels (16 noise + 4 mask + 16 image
# condition) and emits 16; the t2v DiT takes and emits 16. diffusers' t2v
# ``WanPipeline`` sizes its noise from ``transformer.config.in_channels``, so an i2v
# checkpoint driven through it makes 36-channel latents against a 16-channel
# prediction and the scheduler step dies with "The size of tensor a (36) must match
# the size of tensor b (16) at non-singleton dimension 1" -- AFTER loading ~14GB.
_WAN_I2V_IN_CHANNELS = 36
_WAN_T2V_IN_CHANNELS = 16


def _transformer_in_channels(model_dir: str) -> int | None:
    """``in_channels`` from ``<model_dir>/transformer/config.json`` (None if the
    file is absent/unreadable -- the check is then skipped, never fatal)."""
    try:
        with open(os.path.join(model_dir, "transformer", "config.json"),
                  encoding="utf-8") as fh:
            return int(json.load(fh).get("in_channels"))
    except Exception:
        return None


def _checkpoint_pairing_error(in_channels: int | None, has_start_image: bool,
                              weight_uri: str) -> str | None:
    """Pure: the human-readable CONFIG_ERROR reason when the checkpoint's DiT input
    width and the conditioning mode disagree, else None."""
    if in_channels is None:
        return None
    if in_channels == _WAN_I2V_IN_CHANNELS and not has_start_image:
        return (f"{weight_uri} is an IMAGE-to-video checkpoint (DiT in_channels="
                f"{in_channels}) but this render has no start_image, so it would "
                f"run through the text-to-video WanPipeline and fail with a 36-vs-16 "
                f"latent-channel mismatch. Provide start_image (or source_video to "
                f"extend), or pin a t2v model (e.g. wan2.1-t2v-1.3b).")
    if in_channels == _WAN_T2V_IN_CHANNELS and has_start_image:
        return (f"{weight_uri} is a TEXT-to-video checkpoint (DiT in_channels="
                f"{in_channels}) but this render supplies a start_image; the i2v "
                f"pipeline needs a 36-channel i2v DiT. Drop the image or pin an "
                f"i2v model (e.g. wan2.1-i2v-14b-720p).")
    return None


_WAN_480P_MAX_LONG_SIDE = 832


def _variant_for_geometry(weight_uri: str, width: int, height: int) -> str:
    """Pure: the Wan-AI checkpoint name whose training resolution matches the
    requested geometry. Wan 2.1 ships ``...-480P`` and ``...-720P`` i2v variants;
    the 720P one renders 480p but was not tuned for it. Returns ``weight_uri``
    unchanged when no sibling applies."""
    long_side = max(int(width), int(height))
    if weight_uri.endswith("-720P") and long_side <= _WAN_480P_MAX_LONG_SIDE:
        return weight_uri[:-len("-720P")] + "-480P"
    if weight_uri.endswith("-480P") and long_side > _WAN_480P_MAX_LONG_SIDE:
        return weight_uri[:-len("-480P")] + "-720P"
    return weight_uri


def _pick_checkpoint_variant(manifest: RenderManifest, weight_uri: str,
                             model_dir: str, width: int, height: int
                             ) -> tuple[str, str, str]:
    """(model_dir, weight_uri, note): swap to the resolution-matched sibling
    checkpoint ONLY when it is actually on disk (hot or shared root). Opt out with
    HUGPY_WAN_VARIANT_BY_RESOLUTION=0. The choice is logged and recorded in
    provenance; it is not part of the content_hash (the registry row is)."""
    if os.environ.get("HUGPY_WAN_VARIANT_BY_RESOLUTION", "1").strip() == "0":
        return model_dir, weight_uri, "variant pick disabled by env"
    want = _variant_for_geometry(weight_uri, width, height)
    if want == weight_uri:
        return model_dir, weight_uri, f"{weight_uri} matches {width}x{height}"
    alt_dir, _root = _resolve_model_dir(manifest, want)
    if alt_dir and os.path.isfile(os.path.join(alt_dir, "model_index.json")):
        logger.info("wan checkpoint variant: %dx%d -> using %s (%s) instead of %s",
                    width, height, want, alt_dir, weight_uri)
        return alt_dir, want, f"{want} picked for {width}x{height}"
    logger.info("wan checkpoint variant: %s would suit %dx%d but is not on disk; "
                "serving with %s", want, width, height, weight_uri)
    return model_dir, weight_uri, f"{want} not on disk; serving {weight_uri}"


# --------------------------------------------------------------------------- #
# Exception classification + per-spec retry budget (2026-08-21 incident)
# --------------------------------------------------------------------------- #
_SHAPE_ERROR_MARKERS = (
    "must match the size of tensor",
    "sizes of tensors must match",
    "shape mismatch",
    "size mismatch",
    "shape '[",                      # view()/reshape() failures
    "invalid for input of size",
    "expected input",                # conv channel-count mismatch
    "cannot be multiplied",          # mat1 and mat2 shapes
    "does not match",                # "... does not match the shape ..."
    "must be divisible by",
    "is not divisible by",
)
_CONFIG_ERROR_MARKERS = (
    "unexpected keyword argument",
    "has no attribute",
    "got an unexpected",
    "not supported for",
    "is not a valid",
)


def _classify_exception(exc: BaseException) -> tuple[ErrorCode, str]:
    """Pure: (ErrorCode, short label) for an exception raised by the real path.

      * CUDA OOM (torch.OutOfMemoryError / "out of memory")   -> OOM          (retryable)
      * tensor shape/size/channel mismatch (RuntimeError etc.) -> SHAPE_ERROR  (NOT retryable)
      * API/config mismatch (TypeError/AttributeError/KeyError,
        unexpected kwarg, missing attribute)                  -> CONFIG_ERROR (NOT retryable)
      * everything else (disk, ffmpeg, transport)             -> IO_ERROR     (retryable)
    """
    name = type(exc).__name__
    text = str(exc).lower()
    if "outofmemory" in name.lower() or "out of memory" in text:
        return ErrorCode.OOM, "ran out of VRAM"
    if any(m in text for m in _SHAPE_ERROR_MARKERS):
        return ErrorCode.SHAPE_ERROR, "hit a tensor shape mismatch"
    if (isinstance(exc, (TypeError, AttributeError, KeyError))
            or any(m in text for m in _CONFIG_ERROR_MARKERS)):
        return ErrorCode.CONFIG_ERROR, "hit a pipeline/config mismatch"
    return ErrorCode.IO_ERROR, "inference failed"


_FAILURES_NAME = "failures.json"
_RETRY_BUDGET_ENV = "HUGPY_WAN_RETRY_BUDGET"
_DEFAULT_RETRY_BUDGET = 3
_RETRYABLE_RUNNER_CODES = frozenset({ErrorCode.OOM, ErrorCode.IO_ERROR,
                                     ErrorCode.ASSEMBLY_FAILED, ErrorCode.NAN_IN_VAE})


def _retry_budget() -> int:
    raw = os.environ.get(_RETRY_BUDGET_ENV, "").strip()
    try:
        return max(1, int(raw)) if raw else _DEFAULT_RETRY_BUDGET
    except ValueError:
        return _DEFAULT_RETRY_BUDGET


def _record_failure(out_dir: str, code: ErrorCode, message: str) -> int:
    """Append this failure to ``<out_dir>/failures.json`` (the content-addressed
    dir, so the ledger is PER SPEC and survives the process). Returns the number of
    failures now recorded for ``code``. Best-effort: any IO problem returns 1."""
    path = os.path.join(out_dir, _FAILURES_NAME)
    try:
        try:
            with open(path, encoding="utf-8") as fh:
                ledger = json.load(fh)
        except (OSError, ValueError):
            ledger = {}
        if not isinstance(ledger, dict):
            ledger = {}
        row = ledger.setdefault(code.value, {"count": 0, "last": "", "ts": 0.0})
        row["count"] = int(row.get("count", 0)) + 1
        row["last"] = message[:500]
        row["ts"] = time.time()
        os.makedirs(out_dir, exist_ok=True)
        atomic_write_text(path, json.dumps(ledger, indent=2, sort_keys=True))
        return int(row["count"])
    except Exception:  # noqa: BLE001 -- the ledger never breaks error reporting
        return 1


def _budgeted_context(out_dir: str, code: ErrorCode, message: str,
                      base: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    """Context for an Err: records the failure and, once a RETRYABLE code has been
    seen ``_retry_budget()`` times for this spec, vetoes further retries with
    ("retryable", "false") -- which the bus adapter honours."""
    n = _record_failure(out_dir, code, message)
    ctx = base + (("failures", str(n)),)
    if code in _RETRYABLE_RUNNER_CODES and n >= _retry_budget():
        logger.warning("wan retry budget exhausted for %s: %d x %s -> not retryable",
                       dict(base).get("content_hash", "?"), n, code.value)
        ctx += (("retryable", "false"), ("retry_budget", f"exhausted:{n}"))
    return ctx


def _frame_to_pil(frame):
    """Normalize ONE pipeline output frame to a PIL.Image.

    Diffusers video pipelines vary in what ``result.frames[0]`` yields per frame
    even under ``output_type="pil"`` (proven on ae 2026-07-07: WanPipeline handed
    back numpy and the PIL-only save failed after a full denoise). Handles: PIL
    passthrough, torch-like tensors (``.cpu()`` duck-typed — torch never imported
    here), numpy HWC float [0,1] / uint8, CHW transposed, single-channel. Raises
    TypeError on anything else (rides back as errors-as-data)."""
    import numpy as np
    from PIL import Image
    if isinstance(frame, Image.Image):
        return frame
    if hasattr(frame, "cpu"):                      # torch tensor, duck-typed
        frame = frame.cpu().numpy()
    if isinstance(frame, np.ndarray):
        arr = frame
        if arr.dtype != np.uint8:
            arr = (np.clip(arr.astype("float32"), 0.0, 1.0)
                   * 255.0).round().astype(np.uint8)
        if (arr.ndim == 3 and arr.shape[0] in (1, 3, 4)
                and arr.shape[-1] not in (1, 3, 4)):
            arr = np.transpose(arr, (1, 2, 0))     # CHW -> HWC
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return Image.fromarray(arr)
    raise TypeError(f"unsupported pipeline frame type: {type(frame).__name__}")


def _missing_deps() -> list[str]:
    """Which of the heavy inference deps are absent — checked via find_spec so we
    never actually import (and thus never fail-loud) at preflight."""
    import importlib.util
    missing: list[str] = []
    for mod in _REQUIRED_DEPS:
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        except (ImportError, ValueError):
            missing.append(mod)
    return missing


# --------------------------------------------------------------------------- #
# Preflight — errors as data (returns a StageError to raise-as-Err, or None)
# --------------------------------------------------------------------------- #
def _preflight(manifest: RenderManifest) -> StageError | None:
    """Gate the real path. Returns a ``StageError`` (the caller wraps it in
    ``Err``) when the box can't run Wan i2v yet, or None when everything the real
    path needs is present. Order: deps -> GPU -> weights (each needs the prior)."""
    missing = _missing_deps()
    if missing:
        return StageError(
            ErrorCode.DEPS_MISSING,
            "Wan i2v needs GPU inference deps that are not installed: "
            + ", ".join(missing)
            + ". Install: pip install torch (CUDA build) diffusers transformers "
              "bitsandbytes accelerate ftfy",
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
            "no CUDA device available; Wan i2v requires a CUDA GPU (the 4x3090 "
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
    # the shared/snapshot root — see _resolve_model_dir. Neither configured is the
    # "no weights root" error; configured-but-model-absent names BOTH roots tried.
    hot = _hot_weights_root()
    shared_root = _weights_root(manifest)
    if not hot and not shared_root:
        return StageError(
            ErrorCode.WEIGHTS_MISSING,
            "no weights root set — neither STUDIO_WEIGHTS_HOT_ROOT (box-local NVMe) "
            "nor STUDIO_WEIGHTS_ROOT is configured to resolve the Wan weights against",
            (("model_id", manifest.model_id),),
        )

    model_dir, _root_used = _resolve_model_dir(manifest, cfg.weight_uri)
    if not model_dir or not (os.path.isdir(model_dir)
            and os.path.isfile(os.path.join(model_dir, "model_index.json"))):
        return StageError(
            ErrorCode.WEIGHTS_MISSING,
            "Wan " + _weights_missing_msg(cfg.weight_uri, hot, shared_root),
            (("weight_uri", cfg.weight_uri),
             ("hot_root", hot or ""), ("shared_root", shared_root or "")),
        )
    return None


# --------------------------------------------------------------------------- #
# Precision -> bitsandbytes quantization (operator directive: int8 / nf4)
# --------------------------------------------------------------------------- #
def _bnb_config(precision: Precision, BitsAndBytesConfig, torch):
    """Map the router-selected precision to a bitsandbytes quant config:
      * INT8      -> load_in_8bit  (bnb int8)
      * FP8       -> load_in_4bit + nf4  (the tightest bnb path, ~4bit)
      * BF16/FP16 -> None (caller has the VRAM; load unquantized in bf16)
    Returns None to mean "no bnb quantization"."""
    if precision == Precision.INT8:
        return BitsAndBytesConfig(load_in_8bit=True)
    if precision == Precision.FP8:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return None


# --------------------------------------------------------------------------- #
# GPU PLACEMENT decision (pure, no heavy deps — unit-tested without a GPU)
# --------------------------------------------------------------------------- #
# LEGACY: the flat headroom this decision used until 2026-07-27. Kept ONLY as the
# fallback for a model with no measured footprint (see _placement_need_gib), and as
# the record of why it existed. Its own incident note:
#
#   ae 3090, 2026-07-07: wan2.1-t2v-1.3b "8.2GB" placed whole-on-GPU actually
#   allocated 19.6GB and OOM'd at 832x480x29f next to comfy's 512MB.
#
# That 19.6 GiB was a MEASUREMENT and was never an argument against whole-GPU
# placement: it fits a 23.56 GiB card with 3.96 GiB spare. The bug was arithmetic:
# the registry envelope (8.2) counts the DiT ONLY, while the real resident is
# DiT + UMT5-XXL + VAE + activations. Comparing a DiT-only number against a flat
# 16 GB fudge made "8.2 + 16 = 24.2 > 23.56" — a refusal by 0.64 GB, produced by
# adding two numbers that measure different things.
#
# ⚠ THE DERIVED MODEL BELOW DOES NOT REPRODUCE 19.6 — it returns 17.897 GiB for that
# exact tuple. See the "UNCLOSED RESIDUAL" note on _placement_need_gib. Do not read
# the 19.6 here as something this file computes; it is a number a card once reported.
_PLACEMENT_MARGIN_GB = 16.0

# ── MEASURED COMPONENT FOOTPRINTS ───────────────────────────────────────────
# Parameter counts read from the safetensors HEADERS under
# /mnt/llm_storage/video_intel/studio/weights/Wan-AI on 2026-07-27 (sum of every
# tensor's shape product, de-duplicated across shards) and cross-checked against each
# transformer/config.json. These are facts about the weights, not estimates.
#
#   wan2.1-t2v-1.3b       DiT  1,418,996,800  = 1.4190e9  (dim 1536, ffn 8960, L30)
#   wan2.1-vace-1.3b      DiT  2,153,972,032  = 2.1540e9  (1.419e9 F32 base
#                                                + 0.735e9 BF16 / 15 VACE blocks)
#   wan2.1-i2v-14b-720p   DiT 16,395,083,584  = 16.3951e9 (dim 5120, ffn 13824, L40)
#   wan2.1-vace-14b       DiT 17,337,592,896  = 17.3376e9
#   wan2.2-t2v-a14b       DiT 14,288,491,584  = 14.2885e9 x2 (transformer + _2)
#   wan2.2-i2v-a14b       DiT 14,288,901,184  = 14.2889e9 x2 (transformer + _2)
#
# ⚠ CORRECTED 2026-07-27 (adversarial review): wan2.1-vace-14b was carried here as
# 16.3951e9 with the gloss "same geometry as the i2v 14B". The DIM geometry is indeed
# identical (both 5120/13824/40 layers, so _WAN_U below is shared), but the PARAMETER
# COUNTS are not, because the two models bolt different things onto the same trunk.
# Both start from the plain 14B T2V trunk of 14,288,491,584 params, then:
#   i2v-14b   + 2,106,592,000  image cross-attention (config image_dim 1280,
#                              in_channels 36)            -> 16,395,083,584
#   vace-14b  + 3,049,101,312  BF16 VACE control blocks
#                              (config vace_layers = 8)   -> 17,337,592,896
# The old number under-counted vace-14b by 0.942e9 params = 1.76 GiB at bf16.
#
# Shared sidecars, every Wan pipeline:
#   UMT5-XXL text encoder 5,680,910,336 = 5.6809e9 — loaded bf16 => 10.582 GiB. THE
#                                     LARGEST SINGLE RESIDENT, larger than an nf4 14B
#                                     DiT, and it has no registry row anywhere.
#   AutoencoderKLWan        126,892,531 = 0.1269e9 — forced fp32 => 0.473 GiB
#   CLIP ViT-H (i2v only)   632,076,800 = 0.6321e9 — bf16 => 1.177 GiB
_UMT5_PARAMS = 5.6809e9
_VAE_PARAMS = 0.1269e9
_CLIP_PARAMS = 0.6321e9

# params, has_image_encoder, extra_dit_params (MoE second expert, NEVER quantized)
#
# ⚠ has_image_encoder IS A DISK FACT, NOT A CAPABILITY FACT (corrected 2026-07-27).
# "It is an i2v model" does NOT imply a CLIP image encoder: wan2.2-i2v-a14b's
# model_index.json declares  "image_encoder": [null, null]  and there is no
# image_encoder/ directory under its weights at all — Wan 2.2 conditions i2v through
# the DiT's own 36 in_channels instead. Carrying True here added a phantom +1.177 GiB
# to every wan2.2-i2v-a14b estimate. Verified per row against model_index.json +
# the on-disk directory listing; only wan2.1-i2v-14b-720p actually has one
# (CLIPVisionModelWithProjection, 632,076,800 params).
_WAN_FOOTPRINTS: dict[str, tuple[float, bool, float]] = {
    "wan2.1-t2v-1.3b":     (1.4190e9, False, 0.0),
    "wan2.1-vace-1.3b":    (2.1540e9, False, 0.0),
    "wan2.1-i2v-14b-720p": (16.3951e9, True, 0.0),
    "wan2.1-vace-14b":     (17.3376e9, False, 0.0),
    "wan2.2-t2v-a14b":     (14.2885e9, False, 14.2885e9),
    "wan2.2-i2v-a14b":     (14.2889e9, False, 14.2889e9),
}

# Bytes per parameter by precision.
#
#   bf16/fp16  2.0     — the runner computes in torch.bfloat16 either way.
#   fp32       2.0     — NOT 4.0. ``compute_dtype`` is HARDCODED to torch.bfloat16 in
#                        run_wan_i2v and passed as ``torch_dtype`` to both
#                        WanTransformer3DModel.from_pretrained and the pipeline, so an
#                        fp32 manifest would still land in VRAM at 2 bytes/param. (No
#                        registry row declares FP32 today, so this is unreachable — but
#                        an unreachable row that lies is still a lie: it read as
#                        "fp32 costs double", which is what a reader would carry away.)
#   int8       1.003   — CORRECTED from 1.06 on 2026-07-27. A bitsandbytes
#                        Linear8bitLt stores CB (int8, 1 byte/param) plus SCB (one
#                        float32 PER OUTPUT ROW), i.e. 1 + 4/in_features bytes/param
#                        for that layer — 1.0026 at in_features 1536, 1.00029 at
#                        13824. Weighting every 2-D weight in each DiT by its own
#                        in_features and charging the residual non-Linear params
#                        (0.10% of the 1.3B, 0.04% of the 14B: norms, the 3-D patch
#                        conv, embeddings — bnb leaves these at compute dtype) at bf16
#                        gives a whole-DiT effective 1.00293 (1.3B) / 1.00110 (14B).
#                        1.003 is the conservative end of that range.
#                        The old gloss — "8 bits + bnb's fp16 outlier bookkeeping" —
#                        was also WRONG about the mechanism, not just the number:
#                        LLM.int8()'s outlier columns are extracted from the
#                        ACTIVATION at matmul time, not stored alongside the weight,
#                        so they belong in the workspace term below, never here.
#                        Consequence of the old 1.06: the 14B DiT was over-priced by
#                        0.87 GiB at int8 (16.185 vs 15.315).
#   fp8        0.5625  — nf4: 4 bits + an fp32 absmax per 64-element block
#                        (double-quant is not enabled here) = 0.5 + 0.0625.
_BYTES_PER_PARAM = {"bf16": 2.0, "fp16": 2.0, "fp32": 2.0, "int8": 1.003, "fp8": 0.5625}

# ── ACTIVATION WORKSPACE ────────────────────────────────────────────────────
# Calibrated on ae from two measured points at 832x480 for the 1.3B:
#   45f -> 18,720 latent tokens -> 4.6 GiB
#   81f -> 32,760 latent tokens -> 5.5 GiB
# i.e. +0.90 GiB for +14,040 tokens.
#
# ⚠ THE INTERCEPT IS DERIVED FROM THOSE POINTS, NOT TYPED IN (fixed 2026-07-27).
# The shipped constants were slope 6.41e-5 with intercept 0.76 — the slope is the
# two-point fit, but 0.76 is NOT the intercept that fit produces:
#     slope     = 0.90 / 14,040        = 6.410256e-5 GiB/token
#     intercept = 4.6 - slope*18,720   = 3.400 GiB   (and 5.5 - slope*32,760 = 3.400)
# The line the code drew therefore missed BOTH of its own calibration points by
# 2.64 GiB, and since the intercept is a constant, EVERY estimate in this module was
# 2.64 GiB light. That is the unsafe direction: it is what let
# wan2.1-i2v-14b-720p @fp8 832x480x29f read as 23.476 (a "fit" by 0.08 GiB) when the
# same calibration says 26.116 — a MISS by 2.56 — and with the INT8/FP8 early return
# retired that wrong verdict is a bare pipe.to("cuda") with no offload fallback.
# Computing the line from the two points below makes that class of drift impossible:
# there is no second place for the numbers to disagree.
#
# A 3.4 GiB constant term is large but not implausible for this box: the 2026-07-08
# allocator note in _prime_cuda_allocator records ~2.71 GiB RESERVED-BUT-UNALLOCATED
# on a failing render, and a CUDA context plus cuBLAS/cuDNN workspaces account for
# several hundred MB more. The intercept is where that lives.
#
# Cross-check on the SLOPE against structure: 10 live inner-dim buffers + 2 ffn
# buffers at bf16 predicts (10*1536 + 2*8960)*2 = 65.0 KiB/token vs the measured 68.8
# — within 6%, which is why extrapolating the slope to the 14B by its own
# (inner_dim, ffn_dim) is sound. (The structural cross-check speaks ONLY to the
# slope; it says nothing about the intercept, which is why the intercept has to come
# from the measured points.)
_WS_CAL_LO = (18_720, 4.6)   # 832x480x45f on ae
_WS_CAL_HI = (32_760, 5.5)   # 832x480x81f on ae
_WS_SLOPE_GIB_PER_TOKEN = ((_WS_CAL_HI[1] - _WS_CAL_LO[1])
                           / (_WS_CAL_HI[0] - _WS_CAL_LO[0]))     # 6.410256e-5
_WS_INTERCEPT_GIB = _WS_CAL_LO[1] - _WS_SLOPE_GIB_PER_TOKEN * _WS_CAL_LO[0]   # 3.400

# VAE decode workspace. ⚠ UNMEASURED ASSUMPTION — labelled as one (2026-07-27).
# It shipped as ``_WS_INTERCEPT_GIB + 0.60``; the 0.60 has no measurement anywhere in
# this tree, and coupling it to the DENOISE intercept was a category error besides —
# it made a decode estimate move whenever the denoise line was recalibrated. Pinned
# here to the numeric value it has always had (0.76 + 0.60 = 1.36) so this correction
# changes nothing about decode, and named so the assumption is visible.
#
# It is currently INERT and provably so: _placement_need_gib takes
# max(denoise_ws, decode_ws), and the denoise term is never below the intercept
# (3.400 GiB at one single token), so 1.36 can never govern at any geometry. Treat it
# as a placeholder for a measurement nobody has taken rather than as a live bound —
# and note that it is only defensible AT ALL because _place_pipe engages
# vae.enable_tiling() on both branches (see that docstring): untiled,
# AutoencoderKLWan._decode is a resolution-squared fp32 spike, not a constant.
_DECODE_WS_GIB = 1.36

_WS_REF_U = 33_280          # 10*1536 + 2*8960 for the 1.3B
_WAN_U = {                  # 10*inner_dim + 2*ffn_dim
    "wan2.1-t2v-1.3b": 33_280, "wan2.1-vace-1.3b": 33_280,
    "wan2.1-i2v-14b-720p": 78_848, "wan2.1-vace-14b": 78_848,
    "wan2.2-t2v-a14b": 78_848, "wan2.2-i2v-a14b": 78_848,
}


def _latent_tokens(width: int, height: int, n_frames: int) -> int:
    """Wan latent token count: patch_size [1,2,2] over an 8x spatially / 4:1
    temporally compressed latent = 16 px per token, 4 frames per latent frame."""
    return max(1, (width // 16) * (height // 16) * (((max(1, n_frames) - 1) // 4) + 1))


def _placement_need_gib(model_id: str, precision: "Precision", width: int,
                        height: int, n_frames: int) -> float | None:
    """TOTAL VRAM this render needs whole-on-GPU: DiT + sidecars + workspace.

    None when the model has no measured footprint — the caller then falls back to
    the legacy flat-margin test rather than guessing.

    This replaces comparing a DiT-only registry number against a flat 16 GB fudge.
    For wan2.1-t2v-1.3b @fp16 832x480x29f — the exact tuple that lands a clip on ae —
    it returns **17.897 GiB**, decomposed as:

        DiT      1.4190e9 @ bf16   =  2.6431
        UMT5-XXL 5.6809e9 @ bf16   = 10.5815
        VAE      0.1269e9 @ fp32   =  0.4727      static subtotal 13.6973
        denoise workspace, 12,480 latent tokens
                 3.400 + 6.410256e-5*12,480 =  4.2000
        ------------------------------------------------------
        total                                    17.8973   (fits 23.56 by 5.663)

    ⚠ UNCLOSED RESIDUAL — 1.703 GiB, RECORDED RATHER THAN PAPERED OVER (2026-07-27).
    The 2026-07-07 ae incident measured **19.6 GiB** resident for that same tuple, and
    CAPABILITY-VIABILITY-MAP.md repeats it in three places. This function returns
    17.897. Earlier revisions of this docstring asserted the two were the same number;
    they never have been, at any intercept this file has shipped. The honest state:

      * The two figures cannot BOTH be right. The map decomposes 19.6 as
        13.70 static + ~5.90 activations at 12,480 tokens (29f). The calibration this
        module is built on says 4.60 at 18,720 tokens (45f). More activation at FEWER
        tokens is impossible for any monotone-in-tokens model, so at least one of the
        three data points is mislabelled — and none of them has a provenance anywhere
        in this tree beyond the comment that cites it. We cannot tell which from here.
      * The "maybe the incident's GEOMETRY label is wrong" hypothesis was checked and
        DOES NOT HOLD. 1280x720x81f (75,600 tokens) prices at 21.944 GiB, which is
        2.344 off 19.6 — FURTHER away than 832x480x29f's 1.703. (That hypothesis was
        computed against the broken 0.76 intercept, where 720p81f gave 19.31 and did
        look like a match. Correcting the intercept dissolves it.) The 1.3B does pass
        THROUGH 19.6 between 832x480x93f (37,440 tokens -> 19.497) and 832x480x97f
        (39,000 -> 19.597), and again at 1280x720x41f (39,600 -> 19.636); but 93f/97f
        are both above the row's max_frames=81, the registry declares exactly ONE
        resolution for this row (832x480), and 1280x720 is therefore not a geometry it
        can be asked for at all. Those are coincidences of a monotone line crossing a
        value, not an explanation.
      * The residual points the UNSAFE way (we under-price by 1.7 GiB against the one
        directly measured whole-GPU resident this fleet has). It is not currently an
        OOM hazard because the thinnest surviving fit has more headroom than that:
        wan2.1-vace-1.3b @fp16 832x480x81f, the tightest row that still fits, clears
        23.56 by 2.994 GiB. test_placement_need.py pins that inequality, so if a
        future edit ever narrows a fit to less than the unexplained residual, the
        suite fails and says exactly this.

    CLOSING IT needs a measurement, not more arithmetic: torch.cuda.max_memory_-
    allocated() around a whole-GPU 832x480x29f run on ae, split denoise vs decode."""
    fp = _WAN_FOOTPRINTS.get(model_id)
    if fp is None:
        return None
    dit_params, has_image_encoder, extra_dit = fp
    prec = str(getattr(precision, "value", precision)).lower()
    bpp = _BYTES_PER_PARAM.get(prec)
    if bpp is None:
        return None
    gib = 1024.0 ** 3
    need = dit_params * bpp / gib
    # A MoE second expert is loaded UNQUANTIZED at bf16 — quantizing `transformer`
    # does not touch `transformer_2`. This is why the A14B rows do not fit.
    if extra_dit:
        need += extra_dit * 2.0 / gib
    need += _UMT5_PARAMS * 2.0 / gib          # bf16
    need += _VAE_PARAMS * 4.0 / gib           # forced fp32 by the runner
    if has_image_encoder:
        need += _CLIP_PARAMS * 2.0 / gib      # bf16
    u = _WAN_U.get(model_id, _WS_REF_U)
    tokens = _latent_tokens(width, height, n_frames)
    denoise_ws = _WS_INTERCEPT_GIB + _WS_SLOPE_GIB_PER_TOKEN * tokens * (u / _WS_REF_U)
    # VAE decode with tiling enabled is bounded; without tiling it is the spike that
    # most plausibly caused the 07-07 OOM, which is why _place_pipe now engages the
    # memory savers on BOTH branches. See _DECODE_WS_GIB: unmeasured, and currently
    # inert (it can never exceed the denoise term at any geometry).
    return need + max(denoise_ws, _DECODE_WS_GIB)


def _placement_budget_gib(*candidates: float | None) -> float | None:
    """The placement budget = the MINIMUM of every ceiling that applies (2026-07-27).

    ⚠ SEMANTICS, DECIDED AND WRITTEN DOWN. A VRAM budget is an ADMISSION CONSTRAINT,
    not a hint: whoever set it is entitled to have the render stay inside it. A
    physical card is a HARD ceiling. Neither ever licenses exceeding the other, so
    the only sound combination is the min — and because min() can only ever LOWER the
    budget, adopting it cannot turn a previously-safe offload into a new whole-GPU
    placement. It fails toward offload, which is the recoverable direction.

    WHY THIS EXISTS. A reviewer measured, 2026-07-27:
    ``POST /video/studio/i2v {"vram_budget_gb": 6.0, ...}`` routes to INT8 (the
    registry envelope 5.0 fits 6.0), and then placement compared the derived need
    (15.3 GiB at the time) against the CARD (24.0) and answered whole-GPU=True. The
    caller asked for 6 GB and got a 15 GiB placement — the budget governed WHICH
    MODEL was admitted and then stopped governing anything.

    ⚠ HALF OF THAT IS STILL OPEN, AND HERE IS EXACTLY WHERE. This helper closes the
    ceilings the runner can SEE: the live device and the manifest's declared
    ``STUDIO_MAX_VRAM_GB``. It cannot close the reviewer's 6.0, because that number is
    ``CapabilityRequest.vram_budget_gb`` and it is NEVER PLUMBED TO A RUNNER —
    ``video_intel/runners/studio_i2v.py::run_produce_clip`` calls
    ``resolve_studio_env(spec.out_root, master_fps=spec.fps)`` and lets ``max_vram_gb``
    fall to its 24.0 default, and ``RenderManifest`` has no budget field at all. Two
    files this module does not own, so the seam is named rather than cut.

    ⚠ AND WHEN IT IS CUT, DO NOT ROUTE IT THROUGH ``env_snapshot``. ``schemas.py`` puts
    ``env_snapshot`` inside the clip ``content_hash``, and an AUTOFIT budget is sized
    from the worker's MEASURED FREE VRAM (``studio_i2v._resolve_autofit``) — i.e. it
    is not stable for a fixed spec. Hashing it would re-address a clip on every render
    and destroy resume. Thread it as a non-hashed runner argument into this function
    instead; that is what the varargs shape here is for.

    ``None`` candidates are ignored (unknown ≠ zero). All-None returns None, which the
    caller reads as "no budget resolved" and answers offload."""
    known = [float(c) for c in candidates if c is not None]
    return min(known) if known else None


def _max_vram_gb(manifest: RenderManifest) -> float | None:
    """The GPU VRAM budget in GB for the PLACEMENT decision.

    THE LIVE DEVICE AND THE DECLARED CEILING BOTH BIND (2026-07-27) — this returns the
    MIN of them via ``_placement_budget_gib``, not the first one that resolves. The
    device comes from ``_platform.hardware.total_vram_bytes()`` (the same probe pair,
    torch ``mem_get_info`` → ``nvidia-smi``, the worker's own VRAM ceiling uses); the
    declared ceiling from the manifest's captured ``env_snapshot``
    ``STUDIO_MAX_VRAM_GB``, else the live process env. None if nothing resolves, which
    keeps the conservative offload behaviour.

    On today's fleet the min is a no-op in the safe direction (ae: device 23.56 vs
    declared 24.0 -> 23.56; computron: 8.0 vs 24.0 -> 8.0). It starts mattering the
    moment an operator declares a ceiling BELOW the card, which is precisely when
    "the card fits it" must stop being the answer.

    WHY READ THE CARD HERE AND NOT IN ``resolve_studio_env``: the env value rides
    ``env_snapshot``, and ``schemas.py`` puts ``env_snapshot`` INSIDE the clip
    ``content_hash``. Making the SNAPSHOT device-derived would re-address every
    clip the moment two boxes disagree (ae reads 23.56 GB, computron 8.0), so an
    identical spec would re-render instead of resuming. The placement decision is
    not part of the manifest, so reading the real card HERE is free of that.
    Deliberately the same split ``_hot_weights_root()`` uses: process-local truth
    for behaviour, snapshot for addressing.

    ⚠ WHAT THIS FIXES: ``resolve_studio_env`` defaults ``max_vram_gb`` to 24.0 and
    NEITHER CALLER EVER OVERRODE IT, so every placement decision the fleet ever
    made was measured against 24.0 — including on computron, an 8 GiB 4060."""
    device_gb: float | None = None
    try:
        from hugpy_platform.hardware import total_vram_bytes
        total = total_vram_bytes()
        if total and total > 0:
            device_gb = float(total) / (1024 ** 3)
    except Exception:  # noqa: BLE001 — a probe must never fail a render
        pass
    snap = dict(manifest.env_snapshot)
    raw = snap.get("STUDIO_MAX_VRAM_GB") or os.environ.get("STUDIO_MAX_VRAM_GB")
    declared_gb: float | None = None
    if raw not in (None, ""):
        try:
            declared_gb = float(raw)
        except (TypeError, ValueError):
            declared_gb = None
    # ⚠ A DEAF PROBE MUST FAIL CLOSED (2026-07-27, round-2 review). If the device probe
    # returns nothing we do NOT fall back to the declared ceiling, because the declared
    # value is almost never a declaration: ``resolve_studio_env`` DEFAULTS max_vram_gb to
    # 24.0 and no caller overrides it, so "24.0" overwhelmingly means "nobody said". Under
    # the retired flat margin that fabricated 24.0 was inert (8.2 + 16.0 = 24.2 > 24.0 ->
    # offload). Against derived need it is PERMISSIVE: wan2.1-t2v-1.3b fp16 needs 17.897,
    # which "fits" 24.0 -> a bare ``pipe.to("cuda")`` — on computron's 8 GiB 4060 that is
    # an OOM caused by a probe failure, not by a real budget. Unknown card => offload.
    if device_gb is None:
        return None
    # BOTH bind — see _placement_budget_gib. A malformed / absent declaration is
    # "unknown", not "zero", so it drops out of the min rather than forcing offload.
    return _placement_budget_gib(device_gb, declared_gb)


# --------------------------------------------------------------------------- #
# QUANTIZED-MOVE CAPABILITY GUARD (2026-07-27) — the runtime half of retiring the
# INT8/FP8 blanket rule. PURE + version-string driven, so it is unit-testable on a
# box with no bitsandbytes at all (this dev VM has none).
# --------------------------------------------------------------------------- #
# The exact gates in the INSTALLED diffusers 0.39.0, read out of
# site-packages/diffusers/pipelines/pipeline_utils.py (line numbers re-read from the
# venv 2026-07-27, inside DiffusionPipeline.to's per-module loop):
#
#   :541  if is_loaded_in_8bit_bnb and device is not None
#             and is_bitsandbytes_version("<", "0.48.0"):   logger.warning(...)
#   :559  if is_loaded_in_4bit_bnb and device is not None
#             and is_transformers_version(">", "4.44.0"):   module.to(device=device)
#   :562  if (is_loaded_in_8bit_bnb and device is not None
#             and is_transformers_version(">", "4.58.0")
#             and is_bitsandbytes_version(">=", "0.48.0")): module.to(device=device)
#   :569  elif not is_loaded_in_4bit_bnb and not is_loaded_in_8bit_bnb ...:
#                                                          module.to(device, dtype)
#
# ⚠ READ THE CONTROL FLOW, NOT THE WARNING. When the 8-bit gate at :562 fails, nothing
# moves the module and the ``elif`` at :569 is skipped too (its first clause is
# ``not is_loaded_in_8bit_bnb``, and the module IS 8-bit). The UNQUANTIZED components
# take that ``elif`` and DO go to CUDA. The pipeline therefore ends up SPLIT: VAE +
# UMT5 on the GPU, the quantized DiT stranded on the CPU, and the render dies with a
# device-mismatch RuntimeError at the FIRST denoise step — after paying a multi-GB
# load.
#
# ⚠ AND IN ONE HALF OF THE FAILURE SPACE IT IS COMPLETELY SILENT — verified by reading
# :541 rather than assuming. That warning ("…moving it to cuda via `.to()` is not
# supported") is conditioned ONLY on ``bitsandbytes < 0.48.0``. So:
#   * bnb < 0.48.0                      -> warned, then stranded.
#   * bnb >= 0.48.0, transformers <= 4.58.0 -> :541 does not fire, :562 still fails on
#     the transformers clause, and the DiT is stranded with NO log line at all.
# The second case is exactly a box that took the raised pin below but is behind on
# transformers. A warning nobody reads is not a guard; NO warning at all is worse,
# which is why this is a decision-time gate and not a log-grep.
# The 4-bit path fails the same way, one gate earlier: transformers <= 4.44.0 skips
# :559, the ``elif`` is skipped as well, and nf4 strands identically — also silently,
# since :541 is 8-bit-only.
# (bitsandbytes < 0.43.2 on 4-bit is the one LOUD variant — :559 runs and
# ModelMixin.to raises ValueError. Still a failure after the load; still guarded.)
#
# So whole-GPU placement of a bnb-quantized pipeline is only ever CHOSEN when the
# installed stack can actually perform the move. Everything else offloads, which is
# what the fleet did for its whole life and what always works.
#
# Versions verified in this venv 2026-07-27: diffusers 0.39.0, transformers 5.12.1,
# accelerate 1.14.0, bitsandbytes ABSENT. ae runs bitsandbytes 0.49.2.
_BNB_MIN_FOR_MOVE = {"int8": (0, 48, 0), "fp8": (0, 43, 2)}
_TRANSFORMERS_MIN_FOR_MOVE = {"int8": (4, 58, 0), "fp8": (4, 44, 0)}  # STRICT >


def _version_tuple(raw: str | None) -> tuple[int, ...] | None:
    """A PEP440-ish version string -> comparable int tuple, or None when unusable.

    Deliberately dumb: split the leading numeric-dot run and drop any suffix
    (``0.49.2.dev0``, ``5.12.1+cu128`` -> (0,49,2) / (5,12,1)). We only ever ask
    coarse "is it at least X" questions, and an UNPARSEABLE version must read as
    unknown -> guard closed, never as "probably fine"."""
    if not raw:
        return None
    head = str(raw).strip().split("+", 1)[0]
    parts: list[int] = []
    for chunk in head.split("."):
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else None


def _quantized_move_supported(precision: "Precision", bnb_version: str | None,
                              transformers_version: str | None) -> bool:
    """PURE: can THIS stack move a bnb-quantized pipeline to CUDA via ``pipe.to()``?

    True only for INT8/FP8 with both versions present AND satisfying the diffusers
    0.39 gates above (bnb ``>=``, transformers strictly ``>``). Any non-quantized
    precision returns True — there is nothing to guard, ``.to()`` is unconditional.
    A missing/unparseable version returns False: we cannot verify the move, so we do
    not risk the strand."""
    prec = str(getattr(precision, "value", precision)).lower()
    bnb_min = _BNB_MIN_FOR_MOVE.get(prec)
    if bnb_min is None:
        return True                        # bf16/fp16/fp32 — not quantized, no gate
    bnb = _version_tuple(bnb_version)
    tfm = _version_tuple(transformers_version)
    if bnb is None or tfm is None:
        return False
    return bnb >= bnb_min and tfm > _TRANSFORMERS_MIN_FOR_MOVE[prec]


def _installed_quantized_move_ok(precision: "Precision") -> bool:
    """``_quantized_move_supported`` against the versions actually installed here.

    LAZY + TOTAL: reads ``importlib.metadata`` (never imports torch/bitsandbytes), so
    this module stays app-boot-safe and this function is callable on a box with no GPU
    stack. Any probe failure is a False — unverifiable means offload."""
    import importlib.metadata as md

    def _v(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except Exception:  # noqa: BLE001 — absent/broken dist metadata == unknown
            return None

    return _quantized_move_supported(precision, _v("bitsandbytes"), _v("transformers"))


def _should_place_whole_on_gpu(
    precision: Precision,
    model_gb: float | None,
    max_vram_gb: float | None,
    margin: float = _PLACEMENT_MARGIN_GB,
    *,
    model_id: str | None = None,
    width: int | None = None,
    height: int | None = None,
    n_frames: int | None = None,
    quantized_move_ok: bool | None = None,
) -> bool:
    """PURE placement decision (no GPU — unit-testable). True => move the WHOLE pipeline
    to CUDA (``pipe.to("cuda")``); False => ``enable_model_cpu_offload()``.

    DERIVED, NOT FUDGED (2026-07-27). When the model has a measured footprint and the
    geometry is known, this compares the REAL total —
    ``DiT(precision) + UMT5 + VAE [+ CLIP] [+ MoE expert] + activations(tokens)`` —
    against the placement budget (``_placement_budget_gib``: the min of the live card
    and any declared ceiling). Before, it compared the registry's **DiT-only** number
    against a flat 16 GB margin, which is adding two quantities that measure different
    things: ``8.2 + 16.0 = 24.2 > 23.56`` refused wan2.1-t2v-1.3b by **0.64 GB** on a
    card where the derived need is 17.897 GiB with 5.663 GiB to spare (the 2026-07-07
    incident MEASURED 19.6 there; see the unclosed-residual note on
    ``_placement_need_gib`` — this function does not reproduce that number and no
    longer claims to). Every 480p render on this fleet took the slow offload branch
    because of that arithmetic.

    ⚠ THE INT8/FP8 BLANKET EARLY RETURN IS GONE — REPLACED BY A CAPABILITY CHECK, NOT
    BY NOTHING (corrected 2026-07-27). Its rationale ("calling ``.to()`` on an
    8-/4-bit model is unsupported") was stale AS A BLANKET RULE but is exactly right
    on an under-versioned stack, and dropping it outright opened a new failure: on
    diffusers 0.39 an 8-bit pipeline whose stack fails the version gates is not moved
    and not refused, it is SPLIT — unquantized parts to CUDA, the DiT left on the CPU
    — and dies at the first denoise after a multi-GB load. So a quantized precision
    is placed whole-on-GPU only when it BOTH fits and the installed
    bitsandbytes/transformers can actually perform the move
    (``_installed_quantized_move_ok``; see the gate table above). ``quantized_move_ok``
    overrides that probe for tests and for a caller that has already resolved it.

    Falls back to the legacy ``model_gb + margin`` test whenever the footprint or the
    geometry is unknown, so an unmeasured model keeps exactly today's conservative
    behaviour. A missing budget => False (offload)."""
    if max_vram_gb is None:
        return False
    if model_id and width and height and n_frames:
        need = _placement_need_gib(model_id, precision, width, height, n_frames)
        if need is not None:
            fits = need <= max_vram_gb
            # The move must be POSSIBLE, not just affordable. Consulted lazily and ONLY
            # for a bnb-quantized precision that would otherwise be placed — a bf16
            # render never pays for the probe, and an override passed by a caller who
            # is thinking about int8 can never accidentally offload an unquantized one.
            movable = True
            prec_key = str(getattr(precision, "value", precision)).lower()
            if fits and prec_key in _BNB_MIN_FOR_MOVE:
                movable = (_installed_quantized_move_ok(precision)
                           if quantized_move_ok is None else bool(quantized_move_ok))
            logger.info("wan placement: %s @%s %dx%dx%df needs %.2f GiB, budget "
                        "%.2f GiB -> %s", model_id,
                        getattr(precision, "value", precision), width, height,
                        n_frames, need, max_vram_gb,
                        "WHOLE-GPU" if fits and movable
                        else ("offload (fits, but the installed bitsandbytes/"
                              "transformers cannot move a quantized pipeline)"
                              if fits else "offload"))
            return fits and movable
    # Legacy path: unmeasured model or unknown geometry.
    if precision in (Precision.INT8, Precision.FP8):
        return False
    if model_gb is None:
        return False
    return (model_gb + margin) <= max_vram_gb


# --------------------------------------------------------------------------- #
# CUDA allocator defragmentation (item 7) — must run BEFORE torch imports
# --------------------------------------------------------------------------- #
def _prime_cuda_allocator() -> bool:
    """OPT-IN ONLY (HUGPY_CUDA_EXPANDABLE=1): set
    ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` before torch import to
    defragment the CUDA allocator (evidence: the 14B-int8 OOM failed an 80MB
    allocation with ~2.71 GiB reserved-but-unallocated).

    WHY OPT-IN (2026-07-08 ae incident): the original setdefault-on-by-default
    variant CRASH-LOOPED ae — os.environ survives the agent's re-exec, so one
    render primed the flag into the process lineage forever, and this
    driver/torch combo dies natively under expandable_segments (renders died in
    load ~30-40s; then even boot warm-ups crashed). The flag is only applied
    when the operator explicitly sets HUGPY_CUDA_EXPANDABLE=1 on a box.

    HONESTY: the setting only takes effect if the CUDA allocator has NOT already
    initialized in this process. On a worker whose agent avoids torch at boot that is
    normally true for the first studio render, but a prior in-process
    torch/transformers load may have initialized it — so we log whether ``torch`` was
    already imported at this point (sys.modules probe) so it is clear whether this line
    or the process env is doing the work. Returns True iff torch was already imported
    (the line is then likely a no-op for this process)."""
    already = "torch" in sys.modules
    if os.environ.get("HUGPY_CUDA_EXPANDABLE", "").strip() == "1":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    else:
        # DETOX: if a prior (0.1.158) prime leaked this exact value into the
        # re-exec-surviving environ, remove it so the box heals on converge.
        if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True":
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    logger.info("wan cuda allocator: PYTORCH_CUDA_ALLOC_CONF=%s (torch already imported: %s)",
                os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), already)
    return already


# --------------------------------------------------------------------------- #
# OFFLOAD-branch VRAM levers (item 4) — pure/duck-testable, no heavy deps here
# --------------------------------------------------------------------------- #
def _engage_memory_savers(pipe) -> list[str]:
    """On the OFFLOAD placement branch, engage diffusers' peak-VRAM levers so a
    14B-int8 i2v @480p fits next to comfy on a shared 24GB card (live 2026-07-07 it
    OOM'd by ~0.5GB). Each lever is best-effort: a diffusers build / pipeline / VAE
    that lacks it raises ``AttributeError``, which is caught and the lever skipped —
    never fail a render over a memory hint. Returns the names that engaged (also
    logged), so a duck-typed pipe can unit-test the wiring with no GPU.

    diffusers 0.39 surface (verified in-tree): ``DiffusionPipeline`` (base of all
    three Wan pipelines) has ``enable_attention_slicing``; ``AutoencoderKLWan`` has
    ``enable_tiling`` but NOT ``enable_slicing`` — so ``vae.enable_slicing()`` is
    the AttributeError-guarded lever that legitimately no-ops on the Wan VAE."""
    engaged: list[str] = []
    try:
        pipe.enable_attention_slicing()
        engaged.append("attention_slicing")
    except Exception:  # a memory HINT must never fail a render (keeper hardening)
        pass
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        try:
            vae.enable_tiling()
            engaged.append("vae_tiling")
        except Exception:  # a memory HINT must never fail a render (keeper hardening)
            pass
        try:
            vae.enable_slicing()
            engaged.append("vae_slicing")
        except Exception:  # a memory HINT must never fail a render (keeper hardening)
            pass
    if engaged:
        logger.info("wan offload VRAM levers engaged: %s", ", ".join(engaged))
    return engaged


def _place_pipe(pipe, place_whole: bool) -> list[str]:
    """Apply the placement decision to a loaded pipe and engage the VRAM levers on
    BOTH branches. Module-level (shared by wan_i2v + wan_vace) and duck-testable
    with no GPU. Returns the list of engaged levers.

    ⚠ THE SAVERS NOW RUN ON THE WHOLE-GPU BRANCH TOO (2026-07-27), and this is
    load-bearing rather than tidiness. ``vae.enable_tiling()`` is what makes VAE
    decode a BOUNDED constant instead of a resolution-squared spike: untiled,
    ``AutoencoderKLWan._decode`` runs full-frame fp32 convolutions and accumulates
    the whole clip via ``torch.cat``. That untiled decode — not the denoise — is
    the most plausible cause of the 2026-07-07 OOM the old flat margin was built
    to avoid.
    
    So it would have been exactly wrong to flip placement onto measured need while
    leaving this branch unguarded: ``_placement_need_gib`` prices decode as a bounded
    constant (``_DECODE_WS_GIB`` = 1.36 GiB), and that price is only honest if tiling
    is actually on. A calculation that assumes a lever nobody pulled is how you
    reproduce the incident you were trying to prevent.

    ⚠ AND THE PRICE IS AN ASSUMPTION, NOT A MEASUREMENT (2026-07-27). 1.36 GiB has no
    measured provenance in this tree, and it is currently inert anyway: the need
    calculation takes ``max(denoise_ws, _DECODE_WS_GIB)`` and the denoise term never
    drops below the 3.400 GiB intercept, so decode never governs at any geometry.
    Tiling is therefore load-bearing for a bound this file cannot yet defend with a
    number — keep it engaged on both branches until someone measures decode."""
    if place_whole:
        pipe.to("cuda")                        # whole pipeline on GPU (it fits)
        return _engage_memory_savers(pipe)     # tiling keeps decode bounded
    pipe.enable_model_cpu_offload()            # too big -> stream modules
    return _engage_memory_savers(pipe)


# --------------------------------------------------------------------------- #
# MULTI-GPU PLACEMENT (2026-08-21) — a-brain-Super-Server: 2x RTX 3090 24G on one
# board, NO NVLink, PCIe only. Pure planner (unit-tested with a fake device
# inventory, no GPU) + a BOX-ONLY applier with a single-GPU fallback.
#
#   component (default with >=2 GPUs): DiT on cuda:0; UMT5 text encoder + CLIP
#             image encoder + VAE on cuda:1. Encoders/VAE get pre/post hooks that
#             move their inputs to cuda:1 and their outputs back to cuda:0, so the
#             only PCIe traffic is one embedding tensor per encoder call and one
#             latent tensor per VAE encode/decode. The DiT is BF16 when it fits
#             cuda:0's FREE memory alone, else bnb nf4 ("FP8" in this registry).
#   layers:   BF16 DiT sharded across both cards with accelerate's device_map
#             (sequential blocks; activations cross PCIe once per block boundary
#             at the split). Aux modules on cuda:1 as in component mode. Falls
#             back to component when the two cards' per-device budgets cannot
#             hold the BF16 DiT.
#   single:   the historical one-card path (_place_pipe).
#
# NEVER POOLED: every capacity test is against ONE device's free memory. Mode from
# HUGPY_WAN_DEVICE_MODE (component|layers|single|auto); auto = component with 2+
# GPUs else single. Any pipeline/class that cannot take the placement logs why and
# falls back to single.
# --------------------------------------------------------------------------- #
_DEVICE_MODE_ENV = "HUGPY_WAN_DEVICE_MODE"
_DEVICE_MODES = ("auto", "component", "layers", "single")
# Per-device headroom: CUDA context (~0.4-0.6 GiB on a 3090) + allocator slack.
_DEVICE_RESERVE_GIB = 1.5


class DeviceLayout:
    """The chosen multi-GPU layout (plain class: hashable-by-identity, printable)."""
    __slots__ = ("mode", "requested", "transformer_device", "aux_device",
                 "quantize", "max_memory", "reason")

    def __init__(self, mode: str, requested: str, transformer_device: str,
                 aux_device: str, quantize: bool, max_memory: "dict | None",
                 reason: str) -> None:
        self.mode = mode
        self.requested = requested
        self.transformer_device = transformer_device
        self.aux_device = aux_device
        self.quantize = quantize
        self.max_memory = max_memory
        self.reason = reason

    def describe(self) -> str:
        extra = ""
        if self.mode == "layers" and self.max_memory:
            extra = " max_memory=" + ",".join(
                f"cuda:{k}={v}" for k, v in sorted(self.max_memory.items()))
        if self.mode == "component":
            extra = f" transformer={'nf4' if self.quantize else 'as-requested'}"
        return (f"mode={self.mode} (requested={self.requested}) "
                f"transformer->{self.transformer_device} "
                f"text_encoder/image_encoder/vae->{self.aux_device}{extra}; {self.reason}")


def _device_mode_from_env(env: "dict | None" = None) -> str:
    raw = (env if env is not None else os.environ).get(_DEVICE_MODE_ENV, "auto")
    raw = (raw or "auto").strip().lower()
    if raw not in _DEVICE_MODES:
        logger.warning("%s=%r is not one of %s; using auto", _DEVICE_MODE_ENV, raw,
                       "|".join(_DEVICE_MODES))
        return "auto"
    return raw


def _cuda_inventory(torch) -> list[tuple[float, float]]:
    """[(total_gib, free_gib)] per visible CUDA device, via ``mem_get_info`` (the
    same probe the platform VRAM helpers use). Empty on any failure."""
    out: list[tuple[float, float]] = []
    try:
        n = int(torch.cuda.device_count())
    except Exception:
        return out
    gib = 1024.0 ** 3
    for i in range(n):
        try:
            free, total = torch.cuda.mem_get_info(i)
            out.append((total / gib, free / gib))
        except Exception:
            out.append((0.0, 0.0))
    return out


def _component_sizes_gib(model_id: str, width: int, height: int, n_frames: int
                         ) -> "tuple[float, float, float, float] | None":
    """(dit_bf16, dit_nf4, aux, activations) in GiB from the measured footprints;
    None for a model without one (the planner then falls back to single)."""
    fp = _WAN_FOOTPRINTS.get(model_id)
    if fp is None:
        return None
    dit_params, has_image_encoder, extra_dit = fp
    gib = 1024.0 ** 3
    dit_bf16 = (dit_params + extra_dit) * 2.0 / gib
    dit_nf4 = dit_params * _BYTES_PER_PARAM["fp8"] / gib + extra_dit * 2.0 / gib
    aux = _UMT5_PARAMS * 2.0 / gib + _VAE_PARAMS * 4.0 / gib
    if has_image_encoder:
        aux += _CLIP_PARAMS * 2.0 / gib
    u = _WAN_U.get(model_id, _WS_REF_U)
    tokens = _latent_tokens(width, height, n_frames)
    act = max(_WS_INTERCEPT_GIB + _WS_SLOPE_GIB_PER_TOKEN * tokens * (u / _WS_REF_U),
              _DECODE_WS_GIB)
    return dit_bf16, dit_nf4, aux, act


def plan_device_layout(
    requested: str,
    devices: "list[tuple[float, float]]",
    dit_bf16_gib: float,
    dit_quant_gib: float,
    aux_gib: float,
    activations_gib: float,
    precision_is_bf16: bool = True,
    reserve_gib: float = _DEVICE_RESERVE_GIB,
) -> DeviceLayout:
    """PURE planner. ``devices`` = [(total_gib, free_gib)] per CUDA device; every
    fit test is per device (nothing is ever summed across cards except the layers
    shard budget, which is a per-card max_memory map, not a pool)."""
    requested = requested if requested in _DEVICE_MODES else "auto"
    n = len(devices)
    single = lambda why: DeviceLayout(  # noqa: E731
        "single", requested, "cuda:0", "cuda:0", False, None, why)
    mode = requested
    if mode == "auto":
        mode = "component" if n >= 2 else "single"
    if mode == "single":
        return single("single-GPU path" + ("" if n >= 2 else f" ({n} CUDA device(s))"))
    if n < 2:
        return single(f"{requested} needs 2+ CUDA devices, found {n}")
    free0 = devices[0][1]
    free1 = devices[1][1]
    aux_budget = free1 - reserve_gib
    if aux_gib > aux_budget:
        return single(f"aux modules need {aux_gib:.2f} GiB but cuda:1 has "
                      f"{aux_budget:.2f} GiB free after reserve")

    def _component(prefix: str) -> DeviceLayout:
        budget0 = free0 - reserve_gib - activations_gib
        if precision_is_bf16 and dit_bf16_gib <= budget0:
            return DeviceLayout("component", requested, "cuda:0", "cuda:1", False, None,
                                f"{prefix}bf16 DiT {dit_bf16_gib:.2f} GiB fits cuda:0 "
                                f"budget {budget0:.2f} GiB")
        if dit_quant_gib <= budget0:
            why = (f"bf16 DiT {dit_bf16_gib:.2f} GiB exceeds cuda:0 budget "
                   f"{budget0:.2f} GiB -> nf4 DiT {dit_quant_gib:.2f} GiB"
                   if precision_is_bf16 else
                   f"quantized DiT {dit_quant_gib:.2f} GiB fits cuda:0 budget "
                   f"{budget0:.2f} GiB")
            return DeviceLayout("component", requested, "cuda:0", "cuda:1", True, None,
                                prefix + why)
        return single(f"{prefix}even the quantized DiT ({dit_quant_gib:.2f} GiB) "
                      f"exceeds cuda:0 budget {budget0:.2f} GiB -> offload path")

    if mode == "component":
        return _component("")
    # layers: shard the BF16 DiT by per-card budgets
    b0 = free0 - reserve_gib - activations_gib
    b1 = free1 - reserve_gib - aux_gib - activations_gib
    if b0 > 0 and b1 > 0 and dit_bf16_gib <= b0 + b1:
        max_memory = {0: f"{b0:.2f}GiB", 1: f"{b1:.2f}GiB"}
        return DeviceLayout("layers", requested, "cuda:0", "cuda:1", False, max_memory,
                            f"bf16 DiT {dit_bf16_gib:.2f} GiB sharded over per-card "
                            f"budgets {b0:.2f}+{b1:.2f} GiB")
    return _component(f"layers: bf16 DiT {dit_bf16_gib:.2f} GiB exceeds per-card "
                      f"budgets {max(b0, 0):.2f}+{max(b1, 0):.2f} GiB -> component; ")


def _plan_layout_for(manifest: RenderManifest, torch, width: int, height: int,
                     n_frames: int) -> DeviceLayout:
    """Bind the env mode + live inventory + measured sizes into a layout."""
    requested = _device_mode_from_env()
    devices = _cuda_inventory(torch)
    sizes = _component_sizes_gib(manifest.model_id, width, height, n_frames)
    if sizes is None:
        layout = DeviceLayout("single", requested, "cuda:0", "cuda:0", False, None,
                              f"no measured footprint for {manifest.model_id}")
    else:
        is_bf16 = manifest.precision in (Precision.BF16, Precision.FP16)
        layout = plan_device_layout(requested, devices, sizes[0], sizes[1], sizes[2],
                                    sizes[3], precision_is_bf16=is_bf16)
    logger.info("wan device layout: %s | devices=%s", layout.describe(),
                ", ".join(f"cuda:{i} {t:.1f}G total/{f:.1f}G free"
                          for i, (t, f) in enumerate(devices)) or "none")
    return layout


def _move_to_device(obj, device, torch):
    """Recursively move tensors inside tensors / lists / tuples / dicts / HF
    ModelOutput objects (dict subclasses) to ``device``."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):                  # includes transformers ModelOutput
        for k in list(obj.keys()):
            obj[k] = _move_to_device(obj[k], device, torch)
        return obj
    if isinstance(obj, tuple):
        return tuple(_move_to_device(o, device, torch) for o in obj)
    if isinstance(obj, list):
        return [_move_to_device(o, device, torch) for o in obj]
    latent_dist = getattr(obj, "latent_dist", None)
    if latent_dist is not None and hasattr(latent_dist, "parameters"):
        try:
            obj.latent_dist = type(latent_dist)(latent_dist.parameters.to(device))
        except Exception:  # noqa: BLE001
            pass
        return obj
    sample = getattr(obj, "sample", None)
    if isinstance(sample, torch.Tensor):
        try:
            obj.sample = sample.to(device)
        except Exception:  # noqa: BLE001
            pass
    return obj


def _bridge_module(module, own_device, out_device, torch, methods=("forward",)
                   ) -> list:
    """Make ``module`` live on ``own_device`` while looking, to its caller, like it
    lives on ``out_device``: inputs are moved in, outputs moved back. ``forward``
    is bridged with torch hooks; other entry points (the VAE's ``encode``/
    ``decode`` bypass forward) are wrapped directly. Returns zero-arg undo
    callables (remove hook / restore method) so a failed placement can be unwound."""
    undo: list = []
    module.to(own_device)
    if "forward" in methods:
        def _pre(_m, args, kwargs):
            return (_move_to_device(args, own_device, torch),
                    _move_to_device(kwargs, own_device, torch))
        def _post(_m, _args, output):
            return _move_to_device(output, out_device, torch)
        h1 = module.register_forward_pre_hook(_pre, with_kwargs=True)
        h2 = module.register_forward_hook(_post)
        undo.extend((h1.remove, h2.remove))
    for name in methods:
        if name == "forward":
            continue
        fn = getattr(module, name, None)
        if fn is None:
            continue
        def _wrapped(*args, __fn=fn, **kwargs):
            args = _move_to_device(args, own_device, torch)
            kwargs = _move_to_device(kwargs, own_device, torch)
            return _move_to_device(__fn(*args, **kwargs), out_device, torch)
        setattr(module, name, _wrapped)
        undo.append(lambda _m=module, _n=name: _m.__dict__.pop(_n, None))
    return undo


def _pin_execution_device(pipe, device, torch) -> None:
    """diffusers derives ``_execution_device`` from the FIRST nn.Module component
    (the text encoder for Wan) — which in component/layers mode sits on cuda:1.
    Pin both ``device`` and ``_execution_device`` to the DiT's card so prompts,
    noise and timesteps are created where the DiT runs."""
    dev = torch.device(device)
    pinned = type(pipe.__class__.__name__ + "Pinned", (pipe.__class__,), {
        "_execution_device": property(lambda self: dev),
        "device": property(lambda self: dev),
    })
    pipe.__class__ = pinned


def _apply_device_layout(pipe, layout: DeviceLayout, torch) -> list[str]:
    """BOX-ONLY: place a loaded pipe per ``layout``. Returns the engaged levers.
    Raises on any placement problem — the caller falls back to single."""
    tdev = layout.transformer_device
    adev = layout.aux_device
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        raise RuntimeError("pipeline has no .transformer to place")
    undo: list = []
    try:
        if layout.mode == "component":
            transformer.to(tdev)
        # layers: the transformer was loaded with device_map and must not be moved.
        for name in ("text_encoder", "image_encoder"):
            mod = getattr(pipe, name, None)
            if mod is not None:
                undo += _bridge_module(mod, adev, tdev, torch)
        vae = getattr(pipe, "vae", None)
        if vae is not None:
            undo += _bridge_module(vae, adev, tdev, torch,
                                   methods=("forward", "encode", "decode"))
        _pin_execution_device(pipe, tdev, torch)
    except Exception:
        for fn in reversed(undo):   # unwind so the single-GPU fallback starts clean
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
        raise
    return _engage_memory_savers(pipe)


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
def run_wan_i2v(
    manifest: RenderManifest,
    out_root: str,
    start_image: str | None = None,
    should_cancel: "Callable[[], bool] | None" = None,
    on_step: "Callable[[int, int], None] | None" = None,
) -> Result[Artifact, StageError]:
    """Produce (or resume) a Wan i2v clip for ``manifest`` under ``out_root``.

    Returns ``Ok(Artifact)`` on a real render (on the box), or ``Err(StageError)``
    on any expected failure — including the preflight failures that make this a
    graceful no-op on a GPU-less / weight-less box (DEPS_MISSING / NO_GPU /
    WEIGHTS_MISSING). Only a genuine programmer error (a non-RenderManifest)
    raises.

    ``should_cancel`` is an OPTIONAL cooperative-cancel probe (Task 1): a zero-arg
    callable polled at the natural checkpoints (before load, between load and
    render, after render) and — during denoise — wired into the pipeline via
    diffusers' ``callback_on_step_end`` (the callback sets ``pipe._interrupt=True``
    so the loop breaks at the next step boundary). A cancel at any checked point
    returns ``Err(StageError(CANCELLED, ...))`` BEFORE any clip is written. NOTE:
    TRUE mid-denoise interruption is BOX-ONLY — this GPU-less VM short-circuits at
    preflight below, so the callback path only ever executes on the real box. None
    (default) = never cancel.

    ``on_step`` is an OPTIONAL denoise-progress sink (k57): ``on_step(step, steps)``
    is called at each ``callback_on_step_end`` boundary with the 1-based step and
    the total. It rides the SAME diffusers callback the cancel probe uses, so it
    costs nothing extra, and it is the ONLY honest source of within-clip progress —
    without it a single-segment render has no measurable movement between "started"
    and "done" and the console's bar sits at 0 for the whole render. Best-effort:
    the sink is wrapped, so a slow/throwing consumer can never break a render.
    None (default) = report nothing (unchanged behaviour)."""
    if not isinstance(manifest, RenderManifest):
        raise TypeError(
            f"manifest must be a RenderManifest; got {type(manifest).__name__}")

    # ── IDENTITY GUARD: refuse what this runner cannot honour ────────────────
    # THIS RUNNER NEVER READS reference_images (grep it: zero occurrences). VACE
    # does. So an id_lock request that lands HERE silently produces a plausible
    # clip of the WRONG PERSON, with no error — the worst failure shape there is.
    #
    # It is reachable: CAPABILITY_TASKS[ID_LOCK] = (VACE_CONTROL, I2V), and the
    # I2V fallback binds whenever no VACE model fits the geometry — INCLUDING the
    # studio routes' own 512x512 default, which is outside vace-1.3b's 480p
    # envelope. models_seed's comment claimed "in practice VACE always wins ... so
    # id_lock never silently routes to a runner that would ignore the references";
    # that was true only inside the 480p envelope and is corrected there now.
    #
    # Fail LOUD instead of dropping the identity. This is the point of harm, so
    # the guard lives here — it holds no matter how routing later changes.
    if getattr(manifest, "reference_images", None):
        return Err(StageError(
            ErrorCode.NO_CAPABLE_MODEL,
            f"identity lock requested ({len(manifest.reference_images)} reference "
            f"image(s)) but this render bound the i2v runner, which cannot consume "
            f"them — the identity would be silently dropped. Wan-VACE is the "
            f"reference-to-video path; it maxes at 480p, so request a geometry "
            f"within 832x480 (the studio default) for id_lock."))

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
    # torch import (preflight below imports torch to probe CUDA), so it precedes the
    # very first import in this render flow and the sys.modules log stays honest. No-op
    # + harmless on this GPU-less box (preflight then returns DEPS_MISSING before torch).
    _prime_cuda_allocator()

    # PREFLIGHT: everything below the real path returns as DATA, never raises.
    pf = _preflight(manifest)
    if pf is not None:
        return Err(pf)

    # ----------------------------------------------------------------------- #
    # REAL PATH — only reached on a box with deps + CUDA + weights on disk.
    # Never executes on the dev VM (preflight short-circuits above). Written
    # complete enough to run once the 4x3090 box is live.
    # ----------------------------------------------------------------------- #
    import torch
    from diffusers import (
        AutoencoderKLWan,
        BitsAndBytesConfig,
        UniPCMultistepScheduler,
        WanImageToVideoPipeline,
        WanPipeline,
        WanTransformer3DModel,
    )
    from diffusers.utils import load_image

    cfg = MODEL_REGISTRY.get(manifest.model_id)
    # WEIGHTS SOURCE (item 5): prefer the box-local hot NVMe copy if it holds the
    # model, else the shared root — a faster LOAD only; does not affect content_hash.
    model_dir, weights_root_used = _resolve_model_dir(manifest, cfg.weight_uri)
    # 480P/720P checkpoint variant by requested geometry (only if the sibling is on
    # disk) — recorded in provenance below.
    model_dir, weight_uri_used, variant_note = _pick_checkpoint_variant(
        manifest, cfg.weight_uri, model_dir, width, height)
    logger.info("wan i2v: loading %s from %s (%s weights root; %s)",
                weight_uri_used, model_dir, weights_root_used, variant_note)
    compute_dtype = torch.bfloat16
    quant_config = _bnb_config(manifest.precision, BitsAndBytesConfig, torch)
    effective_precision = manifest.precision
    # MULTI-GPU layout (component / layers / single) — decided BEFORE the transformer
    # loads because layers mode loads it sharded (device_map) and component mode may
    # quantize it to fit cuda:0 alone. Logged once, with the per-device inventory.
    layout = _plan_layout_for(manifest, torch, width, height, n_frames)
    if layout.mode == "component" and layout.quantize and quant_config is None:
        quant_config = _bnb_config(Precision.FP8, BitsAndBytesConfig, torch)
        effective_precision = Precision.FP8
        logger.warning("wan device layout: bf16 DiT will not fit %s alone -> loading "
                       "the transformer bnb-nf4 (effective precision fp8, recorded in "
                       "provenance)", layout.transformer_device)
    if layout.mode == "layers" and quant_config is not None:
        quant_config = None
        effective_precision = Precision.BF16
        logger.info("wan device layout: layers mode shards a BF16 DiT; dropping the "
                    "requested %s quantization", manifest.precision.value)
    seed = manifest.seeds.global_seed
    steps = manifest.sampler.steps
    cfg_scale = manifest.sampler.cfg
    # C-prompt: text conditioning from the manifest (part of its content_hash). An
    # empty prompt is valid (image-conditioned i2v); an empty negative maps to None
    # so the pipeline uses its own default rather than an explicit "" negative.
    prompt = manifest.prompt
    negative_prompt = manifest.negative_prompt or None

    # PLACEMENT decision (operator directive: put sub-envelope models WHOLLY on the GPU
    # instead of parking ~15GB in worker RAM via offload). Pure + precomputed here;
    # applied per pipe below. A bnb-quantized (INT8/FP8) precision is placed only when
    # it BOTH fits and the installed bitsandbytes/transformers can actually perform the
    # move (_installed_quantized_move_ok) — otherwise the historical offload path.
    model_gb = cfg.vram.as_map().get(manifest.precision)
    place_whole = _should_place_whole_on_gpu(
        manifest.precision, model_gb, _max_vram_gb(manifest),
        # Pass the identity + geometry so the decision uses MEASURED component bytes
        # (DiT + UMT5 + VAE [+ CLIP] + token-scaled activations) instead of the
        # registry's DiT-only number plus a flat fudge. Without these it silently
        # falls back to the legacy test.
        model_id=manifest.model_id, width=width, height=height, n_frames=n_frames)
    # SHIFT: the flow-match/UniPC scheduler shift RECORDED in the manifest (set by
    # resolve_sampler from the resolution: 3.0 @480p, 5.0 @720p+). None (unset) leaves
    # the pipeline's own default scheduler untouched.
    flow_shift = manifest.sampler.shift

    def _prepare_pipe(pipe):
        """Apply the manifest's scheduler shift + the placement decision to a loaded
        pipe. Wiring shift here (not just recording it) closes the gap where
        manifest.sampler.shift existed but was never consumed — the denoise now uses
        EXACTLY the value in the manifest (INV-1)."""
        if flow_shift is not None:
            try:
                # Wan denoises with a flow-prediction UniPC scheduler; from_config keeps
                # the model's own scheduler config and only overrides flow_shift.
                pipe.scheduler = UniPCMultistepScheduler.from_config(
                    pipe.scheduler.config, flow_shift=flow_shift)
            except Exception:
                # A diffusers build whose UniPC lacks flow_shift: keep the default
                # scheduler rather than fail the render (shift is still in the manifest).
                pass
        # Placement + the VRAM levers (item 4), engaged on BOTH branches. _place_pipe
        # offloads an over-budget (or unmovably-quantized) pipe and engages attention
        # slicing + VAE tiling/slicing; a model that fits goes wholly to CUDA.
        if layout.mode in ("component", "layers"):
            try:
                _apply_device_layout(pipe, layout, torch)
                logger.info("wan device layout applied: %s", layout.describe())
                return
            except Exception as exc:  # noqa: BLE001 — fall back, never fail a render here
                if layout.mode == "layers":
                    # the DiT is already sharded; only the aux placement failed
                    logger.warning("wan device layout: layers aux placement failed (%s: %s); "
                                   "keeping the sharded DiT, aux modules stay where "
                                   "loaded", type(exc).__name__, exc)
                    _engage_memory_savers(pipe)
                    return
                logger.warning("wan device layout: %s placement not supported by %s (%s: "
                               "%s) -> falling back to single", layout.mode,
                               type(pipe).__name__, type(exc).__name__, exc)
        _place_pipe(pipe, place_whole)

    # Cooperative mid-render cancel wiring (Task 1). diffusers 0.39's
    # WanImageToVideoPipeline.__call__ supports `callback_on_step_end`; the callback
    # sets `pipe._interrupt=True` so the denoise loop breaks at the next step
    # boundary. We ALSO re-check should_cancel() around the call so a cancel is
    # still honored if a box's diffusers lacks the callback param. This whole path
    # is BOX-ONLY (preflight short-circuits the GPU-less VM above).
    def _cancel_step_cb(pipe_ref, step_index, timestep, cb_kwargs):
        if should_cancel is not None and should_cancel():
            pipe_ref._interrupt = True   # diffusers checks self.interrupt each step
        if on_step is not None:
            try:
                on_step(int(step_index) + 1, int(steps))
            except Exception:  # noqa: BLE001 — telemetry never breaks a render
                pass
        return cb_kwargs

    call_extra: dict = {}
    if should_cancel is not None or on_step is not None:
        call_extra["callback_on_step_end"] = _cancel_step_cb

    frame_dir = None
    tmp_mp4 = None
    try:
        os.makedirs(out_dir, exist_ok=True)

        # Cooperative cancel — BEFORE load (no weights touched yet if we bail).
        if should_cancel is not None and should_cancel():
            return Err(StageError(
                ErrorCode.CANCELLED, "cancelled before wan load",
                (("content_hash", content_hash), ("model_id", manifest.model_id))))

        # B2 chain — "extend the movie": condition an i2v render on the source clip's
        # LAST FRAME when no start_image was given, BEFORE loading multi-GB weights so
        # a bad source fails fast (errors-as-data). source_video is in the manifest
        # (content_hash), so the extend is deterministic + resume-safe. t2v is
        # text-only (task != I2V) -> the source is carried, never used. BOX-ONLY like
        # the rest of this real path (the GPU-less VM short-circuits at preflight).
        if (start_image is None and (manifest.source_video or "")
                and manifest.task == Task.I2V):
            last_frame = os.path.join(out_dir, _SOURCE_LASTFRAME_NAME)
            ok, stderr_tail = _extract_last_frame(manifest.source_video, last_frame)
            if not ok:
                return Err(StageError(
                    ErrorCode.IO_ERROR,
                    f"could not extract last frame from source_video: {stderr_tail}",
                    (("source_video", manifest.source_video),)))
            start_image = last_frame

        # CHECKPOINT <-> CONDITIONING pairing (2026-08-21 incident): an i2v checkpoint
        # with no start image would take the t2v WanPipeline branch and die mid-denoise
        # on a 36-vs-16 latent-channel mismatch AFTER loading ~14GB. Refuse HERE, as
        # deterministic CONFIG_ERROR data, before any weight is touched.
        pairing = _checkpoint_pairing_error(
            _transformer_in_channels(model_dir), start_image is not None, weight_uri_used)
        if pairing is not None:
            logger.error("wan i2v refused (config_error): %s", pairing)
            return Err(StageError(
                ErrorCode.CONFIG_ERROR, pairing,
                (("content_hash", content_hash), ("model_id", manifest.model_id),
                 ("weight_uri", weight_uri_used),
                 ("start_image", "present" if start_image else "absent"),
                 ("retryable", "false"))))

        # bitsandbytes-quantized DiT transformer (int8 / nf4 per precision).
        tf_kwargs = {"subfolder": "transformer", "torch_dtype": compute_dtype}
        if quant_config is not None:
            tf_kwargs["quantization_config"] = quant_config
        if layout.mode == "layers":
            # accelerate device_map shard over both cards, bounded PER CARD.
            tf_kwargs["device_map"] = "balanced"
            tf_kwargs["max_memory"] = dict(layout.max_memory or {})
        try:
            transformer = WanTransformer3DModel.from_pretrained(model_dir, **tf_kwargs)
        except Exception as exc:  # noqa: BLE001
            if layout.mode != "layers":
                raise
            logger.warning("wan device layout: sharded (layers) load failed (%s: %s) -> "
                           "falling back to single", type(exc).__name__, exc)
            layout = DeviceLayout("single", layout.requested, "cuda:0", "cuda:0", False,
                                  None, f"layers load failed: {type(exc).__name__}")
            tf_kwargs.pop("device_map", None)
            tf_kwargs.pop("max_memory", None)
            quant_config = _bnb_config(manifest.precision, BitsAndBytesConfig, torch)
            effective_precision = manifest.precision
            if quant_config is not None:
                tf_kwargs["quantization_config"] = quant_config
            transformer = WanTransformer3DModel.from_pretrained(model_dir, **tf_kwargs)
        # Wan's VAE is numerically sensitive; the diffusers Wan reference loads it
        # in fp32 (it is small relative to the DiT, so this is affordable).
        vae = AutoencoderKLWan.from_pretrained(
            model_dir, subfolder="vae", torch_dtype=torch.float32)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Cooperative cancel — BETWEEN load and render (weights loaded, nothing
        # rendered/written yet). Per-step interruption during render is handled by
        # the callback below.
        if should_cancel is not None and should_cancel():
            return Err(StageError(
                ErrorCode.CANCELLED, "cancelled after wan load, before render",
                (("content_hash", content_hash), ("model_id", manifest.model_id))))

        if start_image is not None:
            # --- i2v ---
            pipe = WanImageToVideoPipeline.from_pretrained(
                model_dir, transformer=transformer, vae=vae,
                torch_dtype=compute_dtype)
            # Placement + scheduler shift (see _prepare_pipe). A model that fits the
            # budget goes wholly to CUDA — including a bnb-quantized one, but ONLY when
            # the installed stack can move it (see _should_place_whole_on_gpu).
            _prepare_pipe(pipe)
            # C-prompt: the manifest's text prompt (+ negative) drives conditioning.
            # i2v is image-conditioned, so an empty prompt is still valid.
            # The conditioning still is resized to EXACTLY the snapped grid (center
            # crop, no distortion) so the image latent and the noise latent agree.
            cond_image = _fit_image(load_image(start_image), width, height)
            logger.info("wan i2v: conditioning image %s -> %dx%d, num_frames=%d, "
                        "flow_shift=%s, pipeline=%s", start_image, width, height,
                        n_frames, flow_shift, type(pipe).__name__)
            result = pipe(
                image=cond_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=n_frames,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                output_type="pil",
                **call_extra,
            )
        else:
            # --- t2v (start_image is None) ---
            pipe = WanPipeline.from_pretrained(
                model_dir, transformer=transformer, vae=vae,
                torch_dtype=compute_dtype)
            # Placement + scheduler shift (see _prepare_pipe).
            _prepare_pipe(pipe)
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
        # output_type="pil", but the actual per-frame type varies by pipeline/
        # version (list of PIL, ndarray (T,H,W,C) float [0,1], torch tensor) —
        # the FIRST real render on ae (2026-07-07) got ndarray and PIL-only
        # .save() failed AFTER a full denoise. Normalize per-frame.
        frames = result.frames[0]
        actual_frames = len(frames)

        frame_dir = tempfile.mkdtemp(prefix=".frames-", dir=out_dir)
        for i, fr in enumerate(frames):
            _frame_to_pil(fr).save(
                os.path.join(frame_dir, f"frame_{i:05d}.png"), "PNG")

        # Same atomic ffmpeg assembly + promotion as the synthetic runner.
        tmp_mp4 = os.path.join(out_dir, f".clip-tmp-{os.getpid()}.mp4")
        ok, stderr_tail = _assemble_mp4(frame_dir, tmp_mp4, fps)
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
        # Provenance records WHICH weights root served (hot NVMe vs shared) — a
        # sidecar-only field (item 5); it is NOT a canonical input, so it never
        # participates in the content_hash.
        prov = _provenance_dict(manifest)
        prov["weights_root_used"] = weights_root_used
        prov["weight_uri_used"] = weight_uri_used
        prov["checkpoint_variant_note"] = variant_note
        prov["effective_precision"] = getattr(effective_precision, "value",
                                              str(effective_precision))
        prov["device_layout"] = layout.describe()
        prov["geometry_used"] = {"width": width, "height": height,
                                 "num_frames": n_frames, "fps": fps}
        atomic_write_text(
            os.path.join(out_dir, _PROVENANCE_NAME),
            json.dumps(prov, indent=2, sort_keys=True))
    except Exception as exc:  # inference/IO failure rides back as data (INV-3)
        # OOM -> retryable; tensor shape mismatch -> SHAPE_ERROR (deterministic, NOT
        # retryable); API/config mismatch -> CONFIG_ERROR (NOT retryable); else
        # IO_ERROR (retryable, but under the per-spec retry budget).
        code, label = _classify_exception(exc)
        message = f"wan i2v {label}: {exc}"
        base = (("content_hash", content_hash), ("model_id", manifest.model_id),
                ("exception", type(exc).__name__),
                ("geometry", f"{width}x{height}x{n_frames}f"))
        if code in (ErrorCode.SHAPE_ERROR, ErrorCode.CONFIG_ERROR):
            base += (("retryable", "false"),)
        logger.error("wan i2v failed [%s]: %s", code.value, message)
        return Err(StageError(code, message, _budgeted_context(out_dir, code, message, base)))
    finally:
        if tmp_mp4 is not None and os.path.isfile(tmp_mp4):
            try:
                os.remove(tmp_mp4)
            except OSError:
                pass
        if frame_dir is not None and os.path.isdir(frame_dir):
            shutil.rmtree(frame_dir, ignore_errors=True)

    return Ok(Artifact(
        path=clip_path, content_hash=content_hash, frames=actual_frames,
        width=width, height=height, duration_s=actual_frames / float(fps),
        resumed=False))
