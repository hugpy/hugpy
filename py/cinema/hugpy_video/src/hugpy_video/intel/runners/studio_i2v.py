"""Studio i2v BUS RUNNER (B2) — the boundary between the media bus and the studio
spine. Pure ``(StudioI2VSpec, job_id) -> JobResult`` (map §6): expected failures
are DATA (a ``JobResult(ok=False, JobError(...))``), never a raise; only a genuine
programmer error would raise, and ``media_bus.run_claimed`` is the one place that
catches that.

It does four things and nothing else:
  1. Resolve a concrete ``StudioEnv`` from worker defaults (INV-5; no operator env).
  2. Lift the JSON-safe spec into a studio ``CapabilityRequest`` and call
     ``produce_clip`` — the studio's own spine (router -> manifest -> runner ->
     content-addressed clip). Resume (INV-6) is handled inside ``produce_clip``: an
     identical spec re-run reuses the existing clip, no regeneration.
  3. Translate the studio Result at THIS boundary — and only here — via
     ``_stage_error_to_job_error``: studio's ``StageError`` becomes the bus's own
     ``JobError``. Since the Task 2 collapse there is ONE JobError class
     (result_schema.JobError IS comms.jobs.JobError); studio's ``StageError`` is a
     SEPARATE studio-layer vocabulary that this seam ADAPTS into that unified
     JobError — a translation at the single boundary, not a merge, leaving studio's
     errors.py self-contained (StageError reconciliation is TODO(P0-1)).
  4. Ingest the produced ``clip.mp4`` into the media store so the clip is cataloged
     as a ``MediaRef`` (kind="video"), carried out on ``JobResult.outputs``.

STUDIO RENDER OFFLOAD (option a). A REAL-model render (e.g. wan2.1-t2v-1.3b) does
not fit central's GPU-less control-plane box, so this adapter can DELEGATE the
``produce_clip`` execution to a studio GPU worker (``worker_agent/studio_render.py``)
while central keeps the whole control plane and media_jobs.db stays single-writer
here. The decision is data-driven and reversible:

  * ``HUGPY_STUDIO_WORKER`` unset            -> in-process (today's behavior).
  * resolves to a SYNTHETIC / ffmpeg / unroutable binding -> in-process (central
    renders those fine, no model load).
  * resolves to a REAL model AND a worker is set -> delegate: POST /studio/render,
    poll GET status, forward progress + cancel, then INGEST the clip from the
    SHARED content-addressed path (central + worker share /mnt/llm_storage — no b64
    round-trip). The post-render handling (ingest on Ok, JobError on Err) is
    IDENTICAL to the in-process path — the same ``run_produce_clip`` builds the
    request on both sides and the same ``_stage_error_to_job_error`` classifies an
    Err (executed on the worker, rebuilt verbatim here).
  * ``HUGPY_STUDIO_FORCE_REMOTE=1`` is a TEST-ONLY override that delegates even a
    synthetic render (so the offload path is provable with no GPU / no release).

Failure modes are errors-as-data, never a hang or a stuck 'running' row: a worker
unreachable AT KICK-OFF falls back to the in-process path (which on central is the
graceful NO_GPU/DEPS preflight); a worker that dies or times out AFTER accepting
the job returns a retryable ``JobError`` (worker_lost / delegation_timeout).

THE DELEGATION DEADLINE IS PROGRESS-AWARE (k91, 2026-08-06). It used to be a pure
WALL CLOCK on the render: ``HUGPY_STUDIO_DELEGATE_TIMEOUT_S`` (30 min) started at the
RUNNING transition and cancelled whatever was on the card when it expired. That killed
a LIVE render — a 14-segment Cinema movie whose segment 0 was at step 19/32 with the
worker reporting ~86 s/step under GPU contention, i.e. ~46 min of honest work for that
one segment. A clock that cannot tell "wedged" from "slow" answers the wrong question:
the thing worth ending is a render that has STOPPED MOVING, not one that is merely
long. So the loop now tracks the last time the worker's own progress ADVANCED (phase /
step / steps / queue position — the blob it already publishes) and produces
``delegation_timeout`` when there has been NO advance for ``HUGPY_STUDIO_STALL_TIMEOUT_S``
(default 900s). ``HUGPY_STUDIO_DELEGATE_TIMEOUT_S`` is RETAINED, unrenamed and at its
existing 1800s default, but demoted to an ABSOLUTE CEILING on a running render — the
backstop for a worker that keeps emitting plausible movement forever. Its default is
therefore SMALLER than a single real long render, and an operator running long renders
raises it (the live drop-in
``/etc/systemd/system/hugpy-api-dev.service.d/studio-worker.conf`` sets 14400); the
stall window is what should normally fire. The overall cap (queue wait + render) is
unchanged.

Heavy studio imports (which transitively pull numpy/PIL via the synthetic runner)
are LAZY — done inside the functions — so importing this module (which the bus does
at boot, via ``runners/__init__``) stays cheap and can never break app boot.

No pathlib anywhere. os.path only.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace

from hugpy_video.intel.media_store import ingest
from hugpy_video.intel.result_schema import JobError, JobResult

logger = logging.getLogger(__name__)

# StageError codes worth a retry (transient/resource), vs. policy/routing failures
# where the SAME spec would deterministically fail again. Kept as the string values
# so this module needs no studio-enum import at top.
_RETRYABLE_CODES = frozenset({"oom", "nan_in_vae", "assembly_failed", "io_error"})

# --- offload env knobs (all optional; unset => today's in-process behavior) ---
_WORKER_ENV = "HUGPY_STUDIO_WORKER"            # base URL of the studio GPU worker
_FORCE_REMOTE_ENV = "HUGPY_STUDIO_FORCE_REMOTE"  # TEST-ONLY: delegate even synthetic
_POLL_ENV = "HUGPY_STUDIO_POLL_INTERVAL_S"     # status poll cadence (s)
_TIMEOUT_ENV = "HUGPY_STUDIO_DELEGATE_TIMEOUT_S"  # ABSOLUTE render ceiling once RUNNING (s)
_STALL_ENV = "HUGPY_STUDIO_STALL_TIMEOUT_S"       # no-forward-progress window (s)
_OVERALL_CAP_ENV = "HUGPY_STUDIO_OVERALL_CAP_S"   # overall wall-clock cap incl. queue wait (s)
# Kick-off retry (item 2): a ConnectionReset/refused at kick-off (the post-restart /
# converge socket window) retries within this window before falling back in-process.
_KICKOFF_RETRY_WINDOW_ENV = "HUGPY_STUDIO_KICKOFF_RETRY_WINDOW_S"
_KICKOFF_RETRY_INTERVAL_ENV = "HUGPY_STUDIO_KICKOFF_RETRY_INTERVAL_S"
_DEFAULT_POLL_S = 2.0
# THREE clocks, in the order they should fire (see the header's k91 note):
#   * the STALL window is the PRIMARY deadline — silence from the worker's own progress
#     blob (no phase/step/steps/queue-position advance) for this long means the render
#     has stopped moving, which is the only thing worth cancelling. A render that is
#     merely SLOW keeps resetting this clock and is never touched. 900s mirrors comms'
#     own wedged-active retirement window (comms/jobs.py::_stalled_expiry_seconds), so
#     "this job is wedged" means the same number on both sides of the bridge.
#   * the RENDER ceiling is the ABSOLUTE backstop on a render that is actually RUNNING
#     (it does NOT penalize time a job spent WAITING in the worker's queue) — for a
#     worker that emits plausible-but-meaningless movement forever. ⚠ Its 1800s default
#     is DELIBERATELY retained (never renamed, never widened under an operator) but is
#     SMALLER than a single real long render: a 32-step Wan segment at ~86 s/step under
#     GPU contention is ~46 min. An operator who runs long renders RAISES it — the live
#     drop-in /etc/systemd/system/hugpy-api-dev.service.d/studio-worker.conf sets 14400.
#   * the OVERALL cap is a separate wall-clock ceiling over the WHOLE delegation (queue
#     wait + kick-off-409 retries + render) so a wedged worker can never hang a job forever.
_DEFAULT_STALL_TIMEOUT_S = 900.0               # 15 min of NO forward progress = wedged
_DEFAULT_TIMEOUT_S = 1800.0                     # absolute ceiling on a RUNNING render
_DEFAULT_OVERALL_CAP_S = 7200.0                # 2 h: render budget + a few queued jobs ahead
_DEFAULT_KICKOFF_RETRY_WINDOW_S = 30.0         # retry a reset/refused kick-off for ~30s
_DEFAULT_KICKOFF_RETRY_INTERVAL_S = 5.0        # ...every ~5s (also the queue-full retry cadence)
_MAX_POLL_ERRORS = 5                            # consecutive poll failures => worker_lost
_KICKOFF_TIMEOUT_S = 30.0                       # POST /studio/render connect budget
_CANCEL_TIMEOUT_S = 15.0
_STATUS_TIMEOUT_S = 20.0

# --- AUTOFIT (blank budget -> fit to the serving worker's free VRAM) ------------------
# Operator doctrine (2026-07-12): a BLANK vram budget must NOT be a guaranteed-fail low
# guess ("if a model needs 14GB and it's blank, just do 14, otherwise a fail is 100%
# likely"). When a spec's ``vram_budget_gb`` is None, ``render_clip`` resolves the target
# worker via the pluggable seam and sizes the routing budget to that worker's MEASURED
# free VRAM (freshest heartbeat), minus a safety margin.
#
# SAFETY MARGIN: a render's real footprint runs a bit OVER the router's planning estimate
# (activations, allocator fragmentation, other procs / the OS on the card), so hold back
# the LARGER of 10% or 2GB — 2GB dominates on small cards, 10% dominates on big ones.
_AUTOFIT_MARGIN_FRACTION = 0.10
_AUTOFIT_MARGIN_FLOOR_GB = 2.0
# ``gpus[].memory_total`` / ``memory_free`` are stored in BYTES (nvidia-smi MiB *
# 1024*1024, see _platform.hardware.detect_gpus), so convert with GiB (1024**3) — the
# honest inverse of the MiB source and the more conservative GB number.
_BYTES_PER_GIB = 1024 ** 3

# SYNTHETIC IS OPT-IN (operator, 2026-07-27). ``synthetic-i2v``/``synthetic-t2v`` are
# provers: their output is noise derived from seed + geometry. They are registered as a
# last resort for i2v/t2v ONLY (v2v/id_lock register none and refuse honestly — that is
# the behaviour being copied here). Unless this is explicitly set, a render that binds a
# synthetic model REFUSES instead of writing a blob that looks finished. The two
# ``*-synthetic`` presets set it per-request; nothing else should.
_ALLOW_SYNTHETIC_ENV = "STUDIO_ALLOW_SYNTHETIC"


def _allow_synthetic() -> bool:
    return os.environ.get(_ALLOW_SYNTHETIC_ENV) == "1"


from hugpy_platform.versioning import module_version

def _pkg_version() -> str:
    return module_version("hugpy_video")


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _stage_error_to_job_error(stage_error) -> JobError:
    """BOUNDARY adapter: studio ``StageError`` -> bus ``JobError``. The ONE place the
    two error vocabularies meet; a translation, not a merge."""
    code = getattr(stage_error.code, "value", str(stage_error.code))
    # A runner/router can DOWNGRADE retryability in the error's own context (the
    # router's ``_NOT_RETRYABLE`` marker; the Wan runner's per-content_hash retry
    # budget writes ("retryable", "false") once the same spec has failed with a
    # "transient" code HUGPY_WAN_RETRY_BUDGET times). The context can never UPGRADE
    # a deterministic code to retryable.
    ctx = dict(getattr(stage_error, "context", ()) or ())
    vetoed = str(ctx.get("retryable", "")).strip().lower() == "false"
    return JobError(
        code=code,
        message=str(stage_error),
        retryable=(code in _RETRYABLE_CODES) and not vetoed,
    )


# --------------------------------------------------------------------------- #
# ClipOutcome — the SHARED render currency of ``render_clip`` (below), consumed by
# BOTH the single-clip bus adapter (``run_studio_i2v``) and each studio-movie segment
# (``runners.studio_movie``). A studio render settles as EITHER a produced clip (its
# content-addressed SHARED path + geometry) OR an already-translated bus ``JobError``.
# The in-process ``produce_clip`` Result AND a delegated worker payload both normalize
# INTO this one type, so the two callers read an identical shape and each does its own
# catalog/record: the single-clip adapter ``ingest``s the path into a JobResult; the
# movie reads ``frames`` for its splice/trim math, records the node, and ingests too.
# On failure ``error`` is ALWAYS a bus JobError (the StageError->JobError translation
# happened at the single boundary already: in-process via ``_stage_error_to_job_error``,
# delegated on the WORKER via ``artifact_result_to_payload``), never a studio StageError.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClipOutcome:
    ok: bool
    # produced-clip fields (ok=True; None on failure). ``path`` is the SHARED
    # content-addressed clip the caller ingests; the geometry is what a caller needs
    # without re-probing (the movie's branch/trim math reads ``frames``).
    path: "str | None" = None
    content_hash: "str | None" = None
    frames: "int | None" = None
    width: "int | None" = None
    height: "int | None" = None
    duration_s: "float | None" = None
    resumed: bool = False
    # failure field (ok=False): the already-translated bus JobError (retryable set).
    error: "JobError | None" = None
    # AUTOFIT provenance (honesty in the artifact). ``effective_budget_gb`` is the vram
    # budget the router actually resolved for this render (== the explicit budget, OR the
    # worker's measured-free-minus-margin when the spec budget was blank/None), and
    # ``budget_source`` labels HOW it was chosen: ``"explicit"`` (a pinned number),
    # ``"autofit:<worker>"`` (sized to a named worker's GPU CAPACITY, which the reservation
    # engine then evicts to free), or ``"unresolved"`` (no worker / no VRAM data — the
    # render REFUSES; there is no fallback budget). Stamped by ``render_clip`` on
    # BOTH ok and err outcomes; the movie runner records them per segment in movie.json.
    # Defaulted so every other construction site (worker-payload normalize, in-process) is
    # unchanged — render_clip fills them via dataclasses.replace.
    effective_budget_gb: "float | None" = None
    budget_source: "str | None" = None


# --------------------------------------------------------------------------- #
# Shared spine helpers — used by BOTH the central in-process path (below) and the
# worker studio-render thread (worker_agent/studio_render.py imports these), so a
# delegated render is constructed and executed IDENTICALLY to a local one.
# --------------------------------------------------------------------------- #
def build_capability_request(spec):
    """Lift a ``StudioI2VSpec`` into a studio ``CapabilityRequest``. The SINGLE
    source of the request construction, so the in-process render, the remote
    worker render, and the delegation DECISION rule all agree exactly (this is the
    request the router resolves and the manifest is built from)."""
    from hugpy_video.intel.studio.enums import Capability
    from hugpy_video.intel.studio.schemas import CapabilityRequest, Resolution

    return CapabilityRequest(
        capability=Capability(spec.capability),
        target_resolution=Resolution(spec.width, spec.height, spec.fps),
        vram_budget_gb=spec.vram_budget_gb,
        # B2 chain: carry the source clip on the request (routing does not key on it;
        # produce_clip reads spec.source_video for the manifest + extend). None -> None.
        source_video=getattr(spec, "source_video", None),
        # DIRECT MODEL CHOICE: thread the pin so the router binds the requested model or
        # returns a clear Err. Also steers the delegation DECISION (a pinned real model
        # delegates to the GPU worker), since should_delegate/resolves_to_real_model both
        # build the request through here. getattr keeps older spec dicts (no model_id)
        # working. None -> auto-pick.
        pinned_model_id=getattr(spec, "model_id", None),
    )


def run_produce_clip(spec, should_cancel, on_step=None):
    """Build env + seeds from ``spec`` and run ``produce_clip``; return the studio
    ``Result[Artifact, StageError]`` verbatim.

    The ONE place a ``StudioI2VSpec`` is turned into a ``produce_clip`` call —
    shared so the central in-process path and the worker render thread render the
    same pixels for the same spec. ``should_cancel`` is the cooperative-cancel
    probe threaded down to the runner (Task 1); ``on_step(step, steps)`` is the
    optional denoise-progress sink (k57) threaded down beside it. Keyword-default,
    so every existing 2-arg caller is unchanged."""
    from hugpy_video.intel.studio.job import resolve_studio_env
    from hugpy_video.intel.studio.produce import produce_clip
    from hugpy_video.intel.studio.schemas import SeedBundle

    request = build_capability_request(spec)
    env = resolve_studio_env(spec.out_root, master_fps=spec.fps)
    seeds = SeedBundle(global_seed=spec.seed, stage_seeds=(("base", spec.seed),))
    return produce_clip(
        request,
        env=env,
        out_root=spec.out_root,
        seeds=seeds,
        # SAMPLER OVERRIDES: pass the spec's optional steps/cfg (None = unset -> the bound
        # model's family default fills them in produce_clip). getattr keeps older spec
        # dicts (no steps/cfg fields) working.
        steps=getattr(spec, "steps", None),
        cfg=getattr(spec, "cfg", None),
        # CLIP LENGTH: same getattr idiom as steps/cfg, for the same reason — an older
        # spec dict without the field keeps working. This is the last seam between the
        # HTTP spec and the manifest; without it StudioI2VSpec.requested_frames is a
        # knob nothing reads (the exact defect a 2026-07-27 review found one layer down).
        requested_frames=getattr(spec, "requested_frames", None),
        start_image=spec.start_image,
        prompt=getattr(spec, "prompt", "") or "",
        negative_prompt=getattr(spec, "negative", "") or "",
        # B2 chain: the prior-tier clip (movie/scene mp4) this job extends. Carried
        # into the manifest; the i2v runner extends from its last frame when there is
        # no start_image. None -> "" (no source) inside produce_clip.
        source_video=getattr(spec, "source_video", None),
        # IDENTITY LOCK (id_lock): reference image paths + optional VACE control still,
        # carried into the manifest (canonical inputs). The VACE runner consumes them
        # (reference-to-video / control channel). getattr keeps older spec dicts working.
        reference_images=getattr(spec, "reference_images", None),
        control_image=getattr(spec, "control_image", None),
        control_kind=getattr(spec, "control_kind", None),
        # VACE-EXTEND: the parent clip's trailing context frames (studio-movie splice
        # motion-carry). Carried into the manifest; the VACE runner builds the video+mask
        # extend idiom from them. getattr keeps older spec dicts (no field) working.
        vace_context_frames=getattr(spec, "vace_context_frames", None),
        should_cancel=should_cancel,
        on_step=on_step,
    )


def artifact_result_to_payload(result) -> dict:
    """Serialize a ``produce_clip`` ``Result`` into the JSON-safe payload the worker
    returns over HTTP and central consumes.

    ``Ok(Artifact)`` -> ``{"ok": True, "path": <shared clip path>, ...geometry}``;
    central re-``ingest``s that path (which re-probes geometry). ``Err(StageError)``
    -> ``{"ok": False, "error": {...JobError fields}}`` — the SAME
    ``_stage_error_to_job_error`` mapping the in-process path uses runs HERE (on
    the worker), so the code + ``retryable`` classification lives in exactly one
    place and central rebuilds the JobError verbatim."""
    if result.is_err():
        je = _stage_error_to_job_error(result.error)
        return {"ok": False, "error": {
            "code": je.code, "message": je.message, "retryable": je.retryable}}
    art = result.unwrap()
    return {
        "ok": True,
        "path": art.path,
        "content_hash": art.content_hash,
        "frames": art.frames,
        "width": art.width,
        "height": art.height,
        "duration_s": art.duration_s,
        "resumed": art.resumed,
    }


def _payload_to_clip_outcome(payload: dict) -> ClipOutcome:
    """Normalize a worker render payload (from ``artifact_result_to_payload``) into a
    ``ClipOutcome`` WITHOUT ingesting — the caller (single-clip or movie) does its own
    catalog. Ok -> the SHARED clip path + geometry; Err -> the rebuilt JobError (the
    worker already ran ``_stage_error_to_job_error``, so this is a verbatim rebuild, no
    re-translation). A malformed/empty payload is itself errors-as-data (retryable
    internal), and an ok-without-path never becomes a false success."""
    if not isinstance(payload, dict) or "ok" not in payload:
        return ClipOutcome(ok=False, error=JobError(
            code="internal",
            message="studio worker returned no render result payload",
            retryable=True))
    if not payload.get("ok"):
        err = payload.get("error") or {}
        return ClipOutcome(ok=False, error=JobError(
            code=err.get("code", "internal"),
            message=err.get("message", "studio worker render failed"),
            retryable=bool(err.get("retryable", False))))
    path = payload.get("path")
    if not path:
        return ClipOutcome(ok=False, error=JobError(
            code="internal",
            message="studio worker reported ok but no clip path",
            retryable=True))
    return ClipOutcome(
        ok=True,
        path=path,
        content_hash=payload.get("content_hash"),
        frames=payload.get("frames"),
        width=payload.get("width"),
        height=payload.get("height"),
        duration_s=payload.get("duration_s"),
        resumed=bool(payload.get("resumed", False)))


def _payload_to_job_result(payload: dict, job_id: str) -> JobResult:
    """Turn a worker render payload (from ``artifact_result_to_payload``) into a
    ``JobResult``, applying the EXACT post-``produce_clip`` handling the in-process
    path uses: Ok -> ``ingest`` the SHARED clip path into the media store (mints the
    video MediaRef); Err -> rebuild the JobError. Central-only (it does the
    ingest). A malformed/empty payload is itself errors-as-data. Thin wrapper over
    ``_payload_to_clip_outcome`` so the payload->JobError classification lives in ONE
    place (this shape is still exercised directly by the offload conformance test)."""
    outcome = _payload_to_clip_outcome(payload)
    if not outcome.ok:
        return JobResult(job_id=job_id, ok=False, error=outcome.error)
    # SHARED filesystem: central + worker both see /mnt/llm_storage, so the clip the
    # worker wrote is ingestable here directly — identical to the in-process path.
    ref = ingest(outcome.path, kind_hint="video")
    return JobResult(job_id=job_id, ok=True, outputs=(ref,))


# --------------------------------------------------------------------------- #
# Delegation decision + HTTP loop (central side)
# --------------------------------------------------------------------------- #
def _studio_worker_base() -> str:
    return (os.environ.get(_WORKER_ENV) or "").strip().rstrip("/")


def resolve_studio_worker(spec) -> str:
    """PLUGGABLE worker-RESOLUTION seam: the base URL of the studio GPU worker THIS
    render should delegate to (``""`` = none -> in-process). The single place the
    "which worker" question is answered, so both ``should_delegate`` and ``render_clip``
    agree, and a future resolver DROPS IN here without reshaping the delegation helper.

    TODAY: returns the ``HUGPY_STUDIO_WORKER`` env target (one global studio worker).

    OPERATOR-DIRECTED TARGET ARCHITECTURE (2026-07-12): delegation targeting is NOT a
    studio-specific concern — it belongs to the STANDARD model-routing layer. Central
    has no studio models ASSIGNED to it, so a render that binds a real studio model
    (wan / vace / ltx) should route to the worker that OWNS that model, via the registry
    (``workers_for_model(bound_model_id, capability)``, honoring the capability + the
    workers' in-flight gates). Those studio models are not yet first-class registry
    entries (a FOLLOW-UP slice registers them); once they are, the registry-based
    resolver replaces the body HERE and ``HUGPY_STUDIO_WORKER`` DEMOTES to an override /
    fallback (e.g. ``return _studio_worker_base() or _registry_worker(spec)``). Kept a
    pure function OF THE SPEC so that resolver can read the bound model + capability off
    it with no signature change to this seam or its callers."""
    return _studio_worker_base()


from hugpy_video.intel.net import url_host as _url_host


def _autofit_from_worker(base: str) -> "tuple[float, str] | None":
    """``(effective_budget_gb, worker_name)`` for the registry worker whose ``url`` host
    matches ``base`` — its GPU **CAPACITY** minus the safety margin — or ``None`` when no
    worker matches / it reports no usable VRAM.

    ⚠ CAPACITY, NOT FREE (operator ruling 2026-07-27: *"it needs to evict like everything
    else, not assess what it thinks it should set a budget for"*). This used to size to
    ``memory_free``, which made the ROUTING BUDGET a function of whatever the LLM
    residents happened to be holding that second. Measured on ae the same day, minutes
    apart: 3.30 / 5.84 / 7.39 / 6.67 GB — crossing the 6.00 GB cheapest-real-model floor
    in BOTH directions, so an identical request bound a real Wan model or the synthetic
    prover depending on the second it was submitted.

    Sizing to capacity states what the CARD can hold; the reservation engine
    (``video_intel/reservation/engine.py``, which already has a ``studio_i2v`` template)
    then makes room for it the same way every other allocation does — comfy flush, then
    ``/ops/evict`` largest-first with ``force=false`` so the worker's own gate keeps
    🔒static / actively-replying / queued-ahead residents safe — and raises
    ``ReservationRefused`` if it genuinely cannot clear the card. Need, then eviction,
    then an honest refusal: never a quietly smaller model.

    The capacity source is the worker record's per-GPU ``gpus[].memory_total`` (bytes,
    freshest heartbeat). A single render binds ONE device, so we size to the LARGEST single
    GPU's free VRAM (== ``gpus[0]`` on the single-GPU boxes today), never the multi-GPU sum
    (which would over-state what one render can use). Lazy import of central's worker store:
    the studio spine stays boot-cheap and ``runners.scene`` already crosses this exact
    boundary; any import failure (e.g. a pure worker-side context) degrades to None."""
    host = _url_host(base)
    if not host:
        return None
    # Placement seam (EXTRACTION_GUIDE §4): the fleet installs its worker
    # registry into ``hugpy_engine.placement`` at composition time; with nothing
    # installed the null registry lists no workers and the caller refuses.
    try:
        from hugpy_engine.placement import get_worker_registry
        workers = list(get_worker_registry().list_workers(online_only=False) or ())
    except Exception:  # noqa: BLE001
        return None
    for w in workers:
        if _url_host(w.get("url") or "") != host:
            continue
        gpus = [g for g in (w.get("gpus") or []) if isinstance(g, dict)]
        totals = [g.get("memory_total") for g in gpus
                  if isinstance(g.get("memory_total"), (int, float)) and g.get("memory_total") > 0]
        if not totals:
            return None      # matched the box but it reports no VRAM -> caller refuses
        total_gib = max(totals) / _BYTES_PER_GIB
        margin = max(total_gib * _AUTOFIT_MARGIN_FRACTION, _AUTOFIT_MARGIN_FLOOR_GB)
        effective = total_gib - margin
        if effective <= 0:
            return None      # a card smaller than the margin -> caller refuses
        return effective, str(w.get("name") or w.get("id") or "worker")
    return None              # no registry row for this worker URL -> caller refuses


def _resolve_autofit(spec) -> "tuple[object, float, str]":
    """Resolve a BLANK (None) ``vram_budget_gb`` to a concrete routing budget, returning
    ``(spec, effective_budget_gb, budget_source)``.

    * An EXPLICIT budget is a passthrough — the spec is returned UNCHANGED (byte-identical
      delegation payload + routing), ``budget_source == "explicit"``.
    * A None budget is AUTOFIT: resolve the target worker via the pluggable seam and size
      the budget to that card's CAPACITY (minus margin) — what the card can hold once the
      reservation engine has made room, NOT what happens to be free right now.
      ``budget_source == "autofit:<worker>"``.
    * No worker resolvable / no VRAM data -> ``(spec, None, "unresolved")``. There is
      deliberately NO fallback number.

    ⚠ THE FALLBACK IS GONE (operator, 2026-07-27). This used to substitute
    ``_AUTOFIT_FALLBACK_BUDGET_GB = 0.5`` — a value whose only property is that it sits
    under every real model's floor and therefore binds the synthetic prover. "I could not
    measure the worker" is an ERROR, not a budget; manufacturing one turned an
    infrastructure hiccup into a finished-looking noise blob. The caller refuses instead.

    The None is resolved to a NUMBER here, BEFORE any router probe (``resolves_to_real_model``
    / ``produce_clip`` both key on ``vram_budget_gb``), via ``dataclasses.replace`` so the
    frozen spec carried onward (and delegated to the worker) holds the concrete budget."""
    if spec.vram_budget_gb is not None:
        return spec, float(spec.vram_budget_gb), "explicit"
    base = resolve_studio_worker(spec)
    resolved = _autofit_from_worker(base) if base else None
    if resolved is None:
        logger.warning("studio autofit: no GPU capacity readable for %r — refusing "
                       "rather than inventing a budget", base or "(no worker)")
        return spec, None, "unresolved"
    eff, name = resolved
    logger.info("studio autofit: sized budget to %.2fGB from worker %s GPU CAPACITY "
                "(- margin); the reservation engine makes room for it", eff, name)
    return replace(spec, vram_budget_gb=eff), eff, f"autofit:{name}"


def _wants_remote(spec) -> bool:
    """The render-side half of the delegation decision (worker-INDEPENDENT): the
    TEST-ONLY force-remote override, else whether the spec binds a REAL (non-synthetic)
    model. Factored so ``should_delegate`` and ``render_clip`` share ONE rule."""
    if os.environ.get(_FORCE_REMOTE_ENV) == "1":
        return True
    return resolves_to_real_model(spec)


def _binds_synthetic(spec) -> "bool | None":
    """Did the router bind a SYNTHETIC prover for ``spec``? ``None`` = it did not bind
    anything (router Err, or a probe failure).

    Deliberately distinct from ``not resolves_to_real_model(spec)``, which folds those
    two cases together. They need opposite handling: a synthetic binding is the silent
    dross path this gate exists to stop, whereas an unroutable request already has a
    precise router error (NO_CAPABLE_MODEL / VRAM_EXCEEDED) that must reach the caller
    UNCHANGED — v2v and id_lock register no synthetic model at all and have always
    surfaced exactly that. Replacing it with a generic 'no real model fits' would throw
    away the reason."""
    try:
        from hugpy_video.intel.studio.registry import MODEL_REGISTRY
        from hugpy_video.intel.studio.router import CapabilityRouter

        res = CapabilityRouter().resolve(build_capability_request(spec))
        if res.is_err():
            return None
        cfg = MODEL_REGISTRY.get(res.unwrap().model_id)
        if cfg is None:
            return None
        # ⚠ GATE ON THE FRAMEWORK, NOT ON cfg.synthetic. The `synthetic` flag carries
        # TWO different meanings in models_seed and this gate only means one of them:
        #   * Framework.SYNTHETIC (synthetic-i2v / synthetic-t2v) — output is noise
        #     derived from seed + geometry. Nothing real. THIS is what to refuse.
        #   * Framework.FFMPEG (ffmpeg-minterpolate, ffmpeg-lanczos-upscale) and
        #     CODEFORMER — REAL transforms of REAL pixels, flagged synthetic only so
        #     the premium RIFE/LTX rows outrank them ("LAST-RESORT" in their own
        #     comments). Refusing these is refusing a working render.
        # Gating on cfg.synthetic (as this did on 2026-07-27) killed UPRES and
        # INTERPOLATE outright: their premium runners return Err unconditionally
        # (weights/deps missing), so the ffmpeg last-resort IS the working path, and
        # the opt-in gate blocked it. Two live product surfaces, dead, from a flag
        # that meant "rank me last" being read as "I produce dross".
        return bool(getattr(cfg.family, "value", cfg.family) == "synthetic")
    except Exception:  # noqa: BLE001
        return None


def resolves_to_real_model(spec) -> bool:
    """Read-only router probe: does ``spec`` bind a REAL (non-synthetic) model at
    its vram budget? A synthetic / ffmpeg last-resort binding, or an unroutable
    request, -> False (central handles those in-process; an unroutable request
    surfaces its router Err-as-data on the in-process path unchanged). Never
    raises — a probe failure conservatively keeps the render local."""
    try:
        from hugpy_video.intel.studio.registry import MODEL_REGISTRY
        from hugpy_video.intel.studio.router import CapabilityRouter

        res = CapabilityRouter().resolve(build_capability_request(spec))
        if res.is_err():
            return False
        cfg = MODEL_REGISTRY.get(res.unwrap().model_id)
        return bool(cfg is not None and not cfg.synthetic)
    except Exception:  # noqa: BLE001
        return False


def should_delegate(spec) -> bool:
    """True iff this render should be sent to a studio GPU worker: the resolver yields
    a worker target AND (the force-remote test override is on OR the spec binds a real
    model). No worker resolved => never delegate (in-process, unchanged). ``render_clip``
    resolves the worker ONCE itself (this predicate is the public/tested decision rule).

    AUTOFIT: a blank (None) budget is resolved to a concrete number FIRST so the real-model
    probe sees a routable budget (an explicit budget is unchanged — the passthrough)."""
    resolved, _eff, _src = _resolve_autofit(spec)
    return bool(resolve_studio_worker(resolved)) and _wants_remote(resolved)


def _http_post_json(url: str, payload: dict, timeout: float):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.getcode(), (json.loads(body) if body else {})


def _http_get_json(url: str, timeout: float):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.getcode(), (json.loads(body) if body else {})


def _timeout_outcome(render_id: str, base: str, cancel_url: str,
                     reason: str = "exceeded the delegation timeout") -> ClipOutcome:
    """A delegation deadline fired: best-effort tell the worker to stop, then return a
    retryable delegation_timeout ``ClipOutcome`` (never a stuck 'running' row). The
    delegation loop returns this so BOTH callers settle it their own way.

    ``reason`` names WHICH of the three clocks fired (stall window / absolute render
    ceiling / overall cap) and carries into the JobError message, because "timed out"
    alone does not tell an operator whether to raise a knob or go look at the worker —
    the exact ambiguity that made the k91 incident read as a healthy 30-min budget doing
    its job. Defaulted so the retained ``_timeout_result`` shape is unchanged."""
    try:
        _http_post_json(cancel_url, {}, timeout=_CANCEL_TIMEOUT_S)
    except Exception:  # noqa: BLE001
        pass
    logger.warning("studio render %s cancelled on %s — %s", render_id, base, reason)
    return ClipOutcome(ok=False, error=JobError(
        code="delegation_timeout",
        message=f"studio render on {base} was cancelled: {reason}",
        retryable=True))


def _timeout_result(job_id: str, base: str, cancel_url: str) -> JobResult:
    """JobResult wrapper over ``_timeout_outcome`` (retained public shape). A delegation
    deadline fired -> best-effort stop the worker + a retryable JobError."""
    outcome = _timeout_outcome(job_id, base, cancel_url)
    return JobResult(job_id=job_id, ok=False, error=outcome.error)


# The worker-published fields that constitute FORWARD PROGRESS. Deliberately an
# ALLOW-LIST, not "the blob changed": a worker that stamps a wall-clock timestamp (or
# any other free-running field) into its progress dict would otherwise keep the stall
# clock alive forever and re-create the very hang the stall window exists to bound.
# These are the fields the worker actually moves — the denoise step, the coarse phase,
# a frame counter, and the queue position it counts down while WAITING.
_PROGRESS_ADVANCE_KEYS = ("phase", "stage", "step", "steps", "frame", "frames",
                          "position", "progress")


def _progress_key(status, st: dict):
    """A comparable fingerprint of everything the worker says it has ACHIEVED for this
    render: the coarse status plus the allow-listed fields of its progress blob. The
    delegation loop resets its stall clock whenever this value CHANGES, so "moving" is
    defined by the worker's own published movement and nothing else. Total by
    construction — a missing/ill-typed progress blob simply contributes nothing (the
    status alone still fingerprints), never a raise inside the poll loop."""
    prog = st.get("progress") if isinstance(st, dict) else None
    fields = ()
    if isinstance(prog, dict):
        fields = tuple((k, prog.get(k)) for k in _PROGRESS_ADVANCE_KEYS if k in prog)
    return (status, (st.get("position") if isinstance(st, dict) else None), fields)


def _delegate_to_worker(base: str, spec, render_id: str, *,
                        should_cancel=None, progress_sink=None):
    """Delegate the render to the studio worker at ``base`` and settle it.

    Returns a ``ClipOutcome`` once the remote render settles (done/error/cancelled/
    timeout/worker-lost/queue-full), OR ``None`` to signal "kick-off failed, fall
    back to the in-process path" — used ONLY before the worker has accepted the job
    (an unreachable worker at kick-off is exactly the scout's "worker unreachable ->
    current behavior"). Once the worker returns 202 we never fall back (that would
    risk a double render); a later failure becomes a retryable JobError.

    ``render_id`` is the WORKER-SIDE render key (in the POST body + status/cancel URLs)
    — for a single-clip job it IS the bus job_id; for a MOVIE SEGMENT it is a per-
    segment id (a movie posts many renders under its one bus job, and the worker keys
    by this id, so each segment must be distinct or the worker would dedup them as one
    "exists" job). ``should_cancel`` (the bus is_cancelling probe on the OWNING job)
    and ``progress_sink`` (where a queued-position / live-progress blob is emitted) are
    INJECTED so the single-clip path forwards to ``media_bus`` on its own job while the
    movie relays cancel from the MOVIE job and nests each segment's progress. Both
    default to ``media_bus`` on ``render_id`` (the historical single-clip behavior),
    so an existing 3-arg caller is unchanged.

    QUEUE (item 1): a busy worker now QUEUES a second render (202 accepted="queued"
    + position) rather than 409ing, so the delegation just polls it through. A 409
    means the worker's bounded queue is FULL — worker_busy now means WAIT, so we
    retry kick-off with backoff inside the OVERALL cap, only failing worker_busy when
    that window expires. KICK-OFF RESET RETRY (item 2): a ConnectionReset/refused at
    kick-off (post-restart socket window) retries for ~30s before falling back."""
    from hugpy_video.intel import media_bus

    # Injected cancel/progress hooks; default to media_bus on render_id (single-clip).
    if should_cancel is None:
        should_cancel = lambda: media_bus.is_cancelling(render_id)  # noqa: E731
    if progress_sink is None:
        progress_sink = lambda blob: media_bus.set_progress(render_id, blob)  # noqa: E731

    spec_dict = asdict(spec)
    central_version = _pkg_version()
    start_payload = {"job_id": render_id, "spec": spec_dict,
                     "central_version": central_version}

    poll_s = _float_env(_POLL_ENV, _DEFAULT_POLL_S)
    render_budget = _float_env(_TIMEOUT_ENV, _DEFAULT_TIMEOUT_S)
    stall_window = _float_env(_STALL_ENV, _DEFAULT_STALL_TIMEOUT_S)
    overall_cap = _float_env(_OVERALL_CAP_ENV, _DEFAULT_OVERALL_CAP_S)
    reset_window = _float_env(_KICKOFF_RETRY_WINDOW_ENV, _DEFAULT_KICKOFF_RETRY_WINDOW_S)
    retry_interval = _float_env(_KICKOFF_RETRY_INTERVAL_ENV, _DEFAULT_KICKOFF_RETRY_INTERVAL_S)

    started_at = time.time()
    overall_deadline = started_at + overall_cap    # whole-delegation ceiling
    reset_deadline = started_at + reset_window      # connection-reset retry window
    cancel_url = base + "/studio/cancel/" + render_id

    # ---- kick off the remote render (retry: 409-full within overall cap; a reset/
    #      refused within the reset window; other errors -> fall back in-process) --
    body = None
    while True:
        try:
            _code, body = _http_post_json(
                base + "/studio/render", start_payload, timeout=_KICKOFF_TIMEOUT_S)
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 409:
                # Worker's bounded queue is FULL. worker_busy means WAIT: retry within
                # the overall cap, only failing worker_busy when the window expires.
                if time.time() >= overall_deadline:
                    logger.warning("studio worker %s queue full past overall cap — "
                                   "render %s", base, render_id)
                    return ClipOutcome(ok=False, error=JobError(
                        code="worker_busy",
                        message=f"studio worker queue full past the delegation "
                                f"window: {base}",
                        retryable=True))
                logger.info("studio worker %s queue full (409) — retrying kick-off "
                            "in %.1fs (render %s)", base, retry_interval, render_id)
                time.sleep(retry_interval)
                continue
            # Any other HTTP error at kick-off: the worker never started this render,
            # so fall back to the in-process path (graceful NO_GPU/DEPS on central).
            logger.warning("studio worker %s rejected /studio/render (HTTP %s) — "
                           "falling back in-process for render %s", base, exc.code, render_id)
            return None
        except (urllib.error.URLError, OSError) as exc:
            # ConnectionReset/refused/unreachable (e.g. the post-restart converge
            # socket window, seen twice live): retry within the reset window, then
            # fall back in-process (the scout's "worker unreachable" behavior).
            if time.time() >= reset_deadline:
                logger.warning("studio worker %s unreachable at kick-off past retry "
                               "window (%s: %s) — falling back in-process for render %s",
                               base, type(exc).__name__, exc, render_id)
                return None
            logger.info("studio worker %s kick-off connection error (%s: %s) — "
                        "retrying in %.1fs (render %s)",
                        base, type(exc).__name__, exc, retry_interval, render_id)
            time.sleep(retry_interval)
            continue
        except ValueError as exc:
            # Malformed response body — not transient; fall back in-process.
            logger.warning("studio worker %s returned an unparseable kick-off body "
                           "(%s) — falling back in-process for render %s",
                           base, exc, render_id)
            return None

    worker_version = str(body.get("pkg_version") or "")
    if worker_version and worker_version != central_version:
        logger.warning("studio offload VERSION SKEW: central=%s worker=%s (render %s) "
                       "— delegating anyway; behavior may differ across versions",
                       central_version, worker_version, render_id)
    logger.info("studio render %s delegated to %s (accepted=%s, position=%s, worker_pkg=%s)",
                render_id, base, body.get("accepted"), body.get("position"),
                worker_version or "?")

    # ---- poll to settlement ----------------------------------------------------
    # THREE clocks (see the module header's k91 note + the _DEFAULT_* block):
    #   * STALL — the primary deadline. ``last_advance_at`` is the last time the worker's
    #     own progress fingerprint (_progress_key) CHANGED; silence past ``stall_window``
    #     means the render stopped moving. A slow-but-advancing render never trips it.
    #   * RENDER ceiling — absolute, starts only at the RUNNING transition (a queued job
    #     is not charged for its wait).
    #   * OVERALL cap — the ceiling over queue wait + render.
    status_url = base + "/studio/render/" + render_id
    cancel_sent = False
    consecutive_errors = 0
    render_deadline = None            # set when we first observe status == "running"
    last_advance_at = time.time()     # the worker has just ACCEPTED — that is movement
    last_key = None                   # last observed _progress_key (None = nothing yet)

    def _deadline_reason() -> "str | None":
        """WHICH deadline (if any) has fired right now, phrased for the operator, or
        None while the render is still inside all three. Named clocks rather than a
        bare bool so ``delegation_timeout`` says whether to raise a knob or go look at
        the worker."""
        now = time.time()
        if now > overall_deadline:
            return (f"the overall delegation cap of {overall_cap:.0f}s "
                    f"({_OVERALL_CAP_ENV}, queue wait + render) elapsed")
        if render_deadline is not None and now > render_deadline:
            return (f"the absolute render ceiling of {render_budget:.0f}s "
                    f"({_TIMEOUT_ENV}) elapsed while it was running — raise that env "
                    f"if this render is legitimately longer")
        idle = now - last_advance_at
        if idle > stall_window:
            return (f"the worker reported no forward progress for {idle:.0f}s "
                    f"(stall window {stall_window:.0f}s, {_STALL_ENV})")
        return None

    while True:
        time.sleep(poll_s)

        # Forward a cancel INTENT until it is actually DELIVERED (then keep polling
        # until the worker settles). ``should_cancel`` probes the OWNING bus job (the
        # movie relays the MOVIE job's cancel to the per-segment worker render;
        # single-clip probes its own job).
        #
        # ``cancel_sent`` flips ONLY on a successful POST. It used to be set even when
        # the POST raised, so a single transient blip on the worker link dropped the
        # user's cancel FOREVER: nothing retried it, and the render then ran to the
        # 30-min render budget / 2-h overall cap while the console showed
        # "cancelling". That link is known to blip (five consecutive refused status
        # polls were observed 2026-08-04), so the one-shot was a real loss of the
        # cancel. Retries ride the existing poll tick, and a worker that stays
        # unreachable is still bounded by _MAX_POLL_ERRORS -> worker_lost.
        if not cancel_sent:
            try:
                cancelling = should_cancel()
            except Exception:  # noqa: BLE001
                cancelling = False
            if cancelling:
                try:
                    _http_post_json(cancel_url, {}, timeout=_CANCEL_TIMEOUT_S)
                    logger.info("studio render %s: forwarded cancel to %s", render_id, base)
                    cancel_sent = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("studio render %s: cancel POST failed (retrying "
                                   "next poll): %s", render_id, exc)

        # Poll status.
        try:
            _code, st = _http_get_json(status_url, timeout=_STATUS_TIMEOUT_S)
            consecutive_errors = 0
        except Exception as exc:  # noqa: BLE001
            consecutive_errors += 1
            logger.warning("studio render %s: status poll failed (%d/%d): %s",
                           render_id, consecutive_errors, _MAX_POLL_ERRORS, exc)
            if consecutive_errors >= _MAX_POLL_ERRORS:
                return ClipOutcome(ok=False, error=JobError(
                    code="worker_lost",
                    message=f"studio worker {base} unreachable after "
                            f"{consecutive_errors} status polls",
                    retryable=True))
            # A FAILED poll is not movement — the stall clock keeps ticking through it
            # (a worker we cannot reach is, by definition, not reporting progress).
            reason = _deadline_reason()
            if reason is not None:
                return _timeout_outcome(render_id, base, cancel_url, reason)
            continue

        status = st.get("status")

        # FORWARD PROGRESS: reset the stall clock whenever the worker's own published
        # fingerprint changes (a new denoise step, a phase transition, a queue position
        # counting down). This — not the wall clock — is what decides whether a render
        # is alive, so a 46-minute 32-step segment ticking one step every ~86s is never
        # cancelled while it advances (k91).
        key = _progress_key(status, st)
        if key != last_key:
            last_key = key
            last_advance_at = time.time()

        # QUEUED (item 1): the render is WAITING behind another on the worker. Keep
        # polling, forward the queue position as progress so the console shows it, and
        # bound the wait by the OVERALL cap only (the render budget hasn't started).
        # WAITING IS NOT STALLING: a job sitting at a stable position behind a long
        # render is healthy, so the stall clock is held open across the whole queue
        # wait — preserving this branch's pre-k91 "overall cap only" bound exactly.
        if status == "queued":
            position = st.get("position")
            last_advance_at = time.time()
            try:
                progress_sink({"phase": "queued", "position": position})
            except Exception:  # noqa: BLE001
                pass
            if time.time() > overall_deadline:
                logger.warning("studio render %s queued past overall cap on %s",
                               render_id, base)
                return _timeout_outcome(
                    render_id, base, cancel_url,
                    f"it was still QUEUED when the overall delegation cap of "
                    f"{overall_cap:.0f}s ({_OVERALL_CAP_ENV}) elapsed")
            continue

        # RUNNING transition: start the render budget clock (not charged for the wait).
        if status == "running" and render_deadline is None:
            render_deadline = time.time() + render_budget

        # Forward live progress best-effort (whatever the worker exposes).
        prog = st.get("progress")
        if prog is not None:
            try:
                progress_sink(prog)
            except Exception:  # noqa: BLE001
                pass

        if status in ("done", "error"):
            # Both carry a result payload; error-as-data (incl. a cancelled render,
            # which is an Err(CANCELLED) produce_clip result -> code "cancelled").
            return _payload_to_clip_outcome(st.get("result") or {})
        if status in (None, "unknown"):
            # The worker forgot this render (restarted between accept and now).
            return ClipOutcome(ok=False, error=JobError(
                code="worker_lost",
                message=f"studio worker {base} no longer knows render {render_id} "
                        f"(worker restarted?)",
                retryable=True))
        # status == "running" -> keep polling until it settles, it STOPS MOVING (the
        # stall window), or it outlives the absolute ceiling / overall cap.
        reason = _deadline_reason()
        if reason is not None:
            return _timeout_outcome(render_id, base, cancel_url, reason)


def render_clip(spec, *, render_id: str, should_cancel=None, progress_sink=None,
                produce=None) -> ClipOutcome:
    """Render ONE studio clip for ``spec`` — DELEGATED to a studio GPU worker when the
    resolver yields a target AND the spec binds a real model, else IN-PROCESS — and
    return a normalized ``ClipOutcome``. THE shared render primitive: both the single-
    clip bus adapter (``run_studio_i2v``) and each studio-movie segment call this so the
    delegate-or-inline decision LADDER, the delegation loop, and the error translation
    live in exactly one place.

    The decision ladder (identical to the single-clip path's historical rule):
      * resolver yields no worker (``HUGPY_STUDIO_WORKER`` unset today) -> in-process.
      * spec binds a SYNTHETIC / ffmpeg / unroutable model -> in-process (central
        renders those fine; force-remote test override still delegates a synthetic).
      * spec binds a REAL model AND a worker resolves -> DELEGATE (poll, forward
        progress, relay cancel, ingest-from-shared-path via the SHARED filesystem).
      * worker unreachable AT KICK-OFF -> in-process fallback (the graceful NO_GPU/DEPS
        preflight on central). A worker that dies AFTER accepting -> a retryable
        ClipOutcome error (worker_lost / delegation_timeout), never a double render.

    ``render_id`` keys the render on the WORKER (single-clip: the bus job_id; a movie
    segment: a distinct per-segment id). ``should_cancel`` is the cooperative-cancel
    probe (threaded into ``produce_clip`` in-process AND relayed to the worker when
    delegating). ``progress_sink`` receives queued-position / live-progress blobs (the
    movie nests these per segment). ``produce`` is the in-process render function
    (defaults to ``run_produce_clip``); it is injectable so the movie's module-level
    render seam — which its tests patch — stays the inline execution point. All three
    default to the single-clip behavior (media_bus on ``render_id`` / ``run_produce_clip``)."""
    from hugpy_video.intel import media_bus

    if should_cancel is None:
        should_cancel = lambda: media_bus.is_cancelling(render_id)  # noqa: E731
    if progress_sink is None:
        progress_sink = lambda blob: media_bus.set_progress(render_id, blob)  # noqa: E731
    if produce is None:
        produce = run_produce_clip

    # --- AUTOFIT: resolve a BLANK (None) budget to a concrete one BEFORE any router
    #     probe / produce_clip (both key on vram_budget_gb). An explicit budget is a
    #     passthrough (spec unchanged). The resolved (budget, source) is stamped onto the
    #     returned ClipOutcome so the movie runner can record it honestly in movie.json.
    spec, eff_gb, budget_source = _resolve_autofit(spec)

    def _stamp(outcome: ClipOutcome) -> ClipOutcome:
        return replace(outcome, effective_budget_gb=eff_gb, budget_source=budget_source)

    # --- REFUSE rather than invent a budget (operator 2026-07-27) -------------------
    # A blank budget we could not resolve used to become 0.5 -> synthetic -> a noise blob
    # written as a finished clip. An unmeasurable worker is an honest failure.
    if budget_source == "unresolved":
        return _stamp(ClipOutcome(ok=False, error=JobError(
            code="gpu_unavailable",
            message=("cannot size this render: no studio GPU worker is resolvable or it "
                     "reports no VRAM. Set HUGPY_STUDIO_WORKER to a live GPU worker, or "
                     "pass an explicit vram_budget_gb."),
            retryable=True)))

    # --- SYNTHETIC IS OPT-IN (operator 2026-07-27) ----------------------------------
    # Checked BEFORE delegation so it fires on both paths. Without this the router's
    # last-resort synthetic binding ALSO flipped `_wants_remote` to False, which sent the
    # render to the GPU-LESS central in-process path — where noise is the only possible
    # output. So a busy card silently relocated real work off the GPU and returned a blob.
    # i2v/t2v are the only capabilities that can do this; v2v/id_lock register no
    # synthetic model and have always refused honestly — this makes the two agree.
    #
    # Gated on _binds_synthetic (not `not resolves_to_real_model`) so an UNROUTABLE
    # request keeps its precise router error instead of this generic one.
    if _binds_synthetic(spec) is True and not _allow_synthetic():
        return _stamp(ClipOutcome(ok=False, error=JobError(
            code="no_capable_model",
            message=(f"no real model fits {eff_gb:.2f} GB for this render "
                     f"(budget source: {budget_source}) — the router bound the synthetic "
                     f"prover, which produces noise, not a render. Raise vram_budget_gb, "
                     f"or set {_ALLOW_SYNTHETIC_ENV}=1 if you actually want the prover."),
            retryable=False)))

    # --- studio render offload (option a): resolve the worker ONCE, then decide. -----
    base = resolve_studio_worker(spec)
    if base and _wants_remote(spec):
        outcome = _delegate_to_worker(
            base, spec, render_id,
            should_cancel=should_cancel, progress_sink=progress_sink)
        if outcome is not None:
            return _stamp(outcome)    # settled remotely (never fall back after 202)
        logger.info("studio render %s: in-process fallback (worker kick-off failed)",
                    render_id)

    # --- in-process render (historical path; unchanged semantics) --------------
    # Cooperative mid-render cancel (Task 1): the studio never imports media_bus — only
    # this adapter does — so the zero-arg ``should_cancel`` probe is threaded DOWN into
    # the spine. A cancel makes produce_clip's runner abort BEFORE writing a clip and
    # return Err(StageError(CANCELLED)) -> JobError(code="cancelled", retryable=False).
    # NOTE: on the GPU-less control-plane central this is where autofit's fallback budget
    # (0.5) lands — the synthetic path, unchanged (a real model there returns NO_GPU as data).
    # Denoise-step progress for the IN-PROCESS render (k57), forwarded to the same
    # sink the delegated path publishes the worker's blob to — so a bar moves during
    # a long render wherever it physically executes. Threaded ONLY when `produce` is
    # the real render function: the seam is injectable and its test fakes take two
    # positional args, so an unconditional kwarg would break them.
    if produce is run_produce_clip:
        result = produce(spec, should_cancel, on_step=lambda step, steps: (
            progress_sink({"phase": "rendering", "step": step, "steps": steps})))
    else:
        result = produce(spec, should_cancel)
    if result.is_err():
        # ONE boundary: studio StageError -> bus JobError (the delegated path did the
        # identical translation on the worker, so ClipOutcome.error is always a JobError).
        return _stamp(ClipOutcome(ok=False, error=_stage_error_to_job_error(result.error)))
    art = result.unwrap()
    return _stamp(ClipOutcome(
        ok=True, path=art.path, content_hash=art.content_hash, frames=art.frames,
        width=art.width, height=art.height, duration_s=art.duration_s,
        resumed=art.resumed))


def run_studio_i2v(spec, job_id: str) -> JobResult:
    """Run a studio i2v job through ``produce_clip`` and return a ``JobResult``.

    ``Ok(Artifact)`` -> ``JobResult(ok=True, outputs=(clip MediaRef,))``; the ref
    carries the clip path (uri) + resolved geometry/duration + a minted asset id.
    ``Err(StageError)`` -> ``JobResult(ok=False, error=JobError(...))``. Nothing here
    raises on an expected failure.

    Delegation, the in-process fallback, cancel + progress forwarding, and the
    StageError->JobError translation all live in the shared ``render_clip`` (the bus
    job_id is the worker render key); this adapter just settles the ``ClipOutcome``
    into a ``JobResult`` — ``ingest``ing the SHARED clip path on Ok exactly as before."""
    outcome = render_clip(spec, render_id=job_id)
    if not outcome.ok:
        return JobResult(job_id=job_id, ok=False, error=outcome.error)
    # Ok: the clip exists on the SHARED store (in-process or worker-written). Catalog it
    # exactly as the movie/scene runners do — ingest probes it once and mints a video
    # MediaRef carried on outputs.
    ref = ingest(outcome.path, kind_hint="video")
    return JobResult(job_id=job_id, ok=True, outputs=(ref,))
