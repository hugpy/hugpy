"""Per-task reservation templates as DATA + the measured-overlay loader.

The schema mirrors ``dev/VIDEO-TASK-SEQUENCES.md`` §5 (the operator-approved
shape). One template per heavy bus-job ``name``; the engine reads a template,
computes the run's PEAK GPU need, and pre-claims the card.

Two-layer sourcing (task item 1):
  1. BUILT-IN seed — the estimates transcribed from the doc's §3 stage tables
     (``footprint_source="estimate"``). These are PLANNING envelopes, not
     reservation guarantees (the doc's load-bearing caveat).
  2. MEASURED OVERLAY — the sibling p7 agent fills
     ``<DEFAULT_ROOT>/comms/reservations/measured.json`` from real calibration
     renders on ``ae``. When a measured number is present it WINS over the
     estimate (per-stage ``vram_bytes_measured`` and/or a task-level
     ``peak_vram_bytes``). We are a READ-ONLY consumer of that file — p7 owns it.

The overlay loader is deliberately TOLERANT of the file's shape (p7 is still
filling it): it accepts a per-task map whose value may carry ``peak_vram_bytes``,
a ``stages`` map (stage_name -> {vram_bytes_measured|vram_bytes|bytes}), or a
flat stage_name -> bytes map. Unknown keys are ignored; a malformed file
degrades to the estimates (never raises).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_GIB = 1024 ** 3

# measured.json memo (see _read_measured): {"key": (path, mtime_ns, size), "value": {}}
_measured_lock = threading.Lock()
_measured_cache: Dict[str, Any] = {"key": None, "value": {}}


def _measured_path() -> str:
    """p7's read-only measured overlay. Env-overridable for tests."""
    env = (os.environ.get("HUGPY_RESERVATIONS_MEASURED") or "").strip()
    if env:
        return env
    base = (os.environ.get("HUGPY_RESERVATIONS_DIR") or "").strip()
    if not base:
        try:
            from hugpy_platform.constants import DEFAULT_ROOT
            base = os.path.join(str(DEFAULT_ROOT), "comms", "reservations")
        except Exception:  # noqa: BLE001 — degrade to a sane default, never raise
            base = os.path.expanduser("~/.hugpy/reservations")
    return os.path.join(base, "measured.json")


# --------------------------------------------------------------------------- #
# schema (§5)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stage:
    """One stage of a task's GPU timeline (§5 ``stages[]`` element)."""
    name: str
    process: str                         # P-central | P-worker | P-comfy | P-studio | P-identity | ffmpeg
    host: str                            # resolved fleet box (today: "ae" for heavy video)
    gpu_affinity: Optional[str]          # device the stage claims; None for CPU stages
    models: Tuple[Dict[str, Any], ...] = ()   # [{key, weight_uri?, disk_bytes?, precision?}]
    vram_bytes_est: Optional[int] = None      # planning estimate at nominal geometry/precision
    vram_bytes_measured: Optional[int] = None  # filled from p7 measured.json (wins)
    exclusive: bool = False              # whole-pipeline denoise/mesh — needs the card alone
    out_of_band_vram: bool = False       # P-comfy / P-identity — invisible to the LLM engine
    duration_class: str = "seconds"      # ms | seconds | minutes | long
    count_expr: str = "1"                # per-stage multiplicity (SEQUENTIAL on 1 GPU today)
    footprint_source: str = "estimate"   # "estimate" | "measured"

    def vram_bytes(self) -> Optional[int]:
        """Effective per-stage VRAM: measured wins over the estimate."""
        if self.vram_bytes_measured is not None:
            return int(self.vram_bytes_measured)
        return None if self.vram_bytes_est is None else int(self.vram_bytes_est)


@dataclass(frozen=True)
class ReservationTemplate:
    """One heavy bus-job's reservation template (§5 top-level)."""
    task: str
    orchestrator: str                    # where the fat runner loop runs (never the GPU)
    delegation: str                      # none | studio_worker | identity_render | dispatch_plane
    gpu_affinity: str                    # task-level card the reservation holds (today "ae")
    stages: Tuple[Stage, ...]
    co_resident: Tuple[str, ...] = ()    # stage names that hold VRAM SIMULTANEOUSLY
    reservation_window: str = "whole_run"  # whole_run | per_stage
    preempts: Tuple[str, ...] = ()       # residents the engine must yield first (advisory)
    exclusive_heavy: bool = True         # operator's whole-run pre-claim set (task (a))
    peak_vram_bytes_measured: Optional[int] = None  # task-level measured peak (wins)

    # -- peak sizing -------------------------------------------------------- #
    def peak_bytes(self) -> Optional[int]:
        """The GPU bytes the reservation must guarantee on ``gpu_affinity``.

        A task-level MEASURED peak wins outright. Otherwise: the max over
        (a) the largest single stage and (b) the SUM of the co-resident group —
        because co-resident stages hold VRAM simultaneously (the movie's t2i
        checkpoint + vision judge), while everything else is sequential on the
        single card today (count_expr fan-out does NOT multiply VRAM — §5 note).
        None when no stage carries a number (fully-unmeasured task) — the engine
        then treats make-room as best-effort (proceed, don't refuse blindly)."""
        if self.peak_vram_bytes_measured is not None:
            return int(self.peak_vram_bytes_measured)
        singles = [s.vram_bytes() for s in self.stages if s.vram_bytes() is not None]
        max_single = max(singles) if singles else None
        co = [s.vram_bytes() for s in self.stages
              if s.name in self.co_resident and s.vram_bytes() is not None]
        co_sum = sum(co) if co else None
        vals = [v for v in (max_single, co_sum) if v is not None]
        return max(vals) if vals else None

    def as_dict(self) -> Dict[str, Any]:
        """JSON-safe view for GET /llm/reservations introspection / tests."""
        return {
            "task": self.task,
            "orchestrator": self.orchestrator,
            "delegation": self.delegation,
            "gpu_affinity": self.gpu_affinity,
            "reservation_window": self.reservation_window,
            "exclusive_heavy": self.exclusive_heavy,
            "peak_vram_bytes": self.peak_bytes(),
            "co_resident": list(self.co_resident),
            "preempts": list(self.preempts),
            "stages": [
                {
                    "name": s.name, "process": s.process, "host": s.host,
                    "gpu_affinity": s.gpu_affinity,
                    "vram_bytes_est": s.vram_bytes_est,
                    "vram_bytes_measured": s.vram_bytes_measured,
                    "vram_bytes": s.vram_bytes(),
                    "exclusive": s.exclusive,
                    "out_of_band_vram": s.out_of_band_vram,
                    "duration_class": s.duration_class,
                    "count_expr": s.count_expr,
                    "footprint_source": (
                        "measured" if s.vram_bytes_measured is not None
                        else s.footprint_source),
                    "models": list(s.models),
                }
                for s in self.stages
            ],
        }


# --------------------------------------------------------------------------- #
# BUILT-IN seed — transcribed from dev/VIDEO-TASK-SEQUENCES.md §3 (estimates).
# Every vram_bytes_measured is None here on purpose: p7's measured.json overlays
# it. Numbers are PLANNING envelopes (the doc's caveat), sized conservatively so
# make-room clears the known collision (the 17.7 GB agent-brain squat).
# --------------------------------------------------------------------------- #
def _gib(n: float) -> int:
    return int(n * _GIB)


_WAN_VACE = {"key": "wan2.1-vace-1.3b", "weight_uri": "Wan-AI/Wan2.1-VACE-1.3B",
             "disk_bytes": 25748804175, "precision": "int8"}
_WAN_T2V = {"key": "wan2.1-t2v-1.3b", "weight_uri": "Wan-AI/Wan2.1-T2V-1.3B",
            "disk_bytes": 18897856000, "precision": "int8"}

# The Wan denoise envelope: registry INT8 est ~5-6 GB, but the MEASURED ae
# incident (2026-07-07) showed wan2.1-t2v-1.3b allocating ~19.6 GB @ 832x480x29f
# (DiT + UMT5 encoder + VAE + activations; _PLACEMENT_MARGIN_GB=16). We seed the
# HONEST measured-incident envelope (20 GB) so a reservation actually clears the
# card, not the optimistic registry INT8 number.
_WAN_PEAK = _gib(20)

TEMPLATES: Dict[str, ReservationTemplate] = {
    # ---- studio i2v (single Wan clip) — §3.4 -------------------------------
    "studio_i2v": ReservationTemplate(
        task="studio_i2v", orchestrator="P-central", delegation="studio_worker",
        gpu_affinity="ae", reservation_window="whole_run",
        preempts=("llm_agent_brain", "comfy_idle_checkpoint"),
        stages=(
            Stage(name="wan_denoise", process="P-studio", host="ae",
                  gpu_affinity="ae:cuda:0", models=(_WAN_T2V,),
                  vram_bytes_est=_WAN_PEAK, exclusive=True,
                  duration_class="minutes", count_expr="1"),
        ),
    ),
    # ---- studio movie (NLE row of Wan clips) — §3.5 ------------------------
    # Segments render SEQUENTIALLY on the one card, so peak == ONE Wan pipeline;
    # the reservation WINDOW is long (N x minutes) and holds ae for the movie.
    "generate_studio_movie": ReservationTemplate(
        task="generate_studio_movie", orchestrator="P-central",
        delegation="studio_worker", gpu_affinity="ae",
        reservation_window="whole_run",
        preempts=("llm_agent_brain", "comfy_idle_checkpoint"),
        stages=(
            Stage(name="wan_segment_denoise", process="P-studio", host="ae",
                  gpu_affinity="ae:cuda:0", models=(_WAN_VACE,),
                  vram_bytes_est=_WAN_PEAK, exclusive=True,
                  duration_class="long", count_expr="len(segments)"),
        ),
    ),
    # ---- identity reconstruction (id_lock stills / turntable) — §3.6 -------
    "identity_reconstruction": ReservationTemplate(
        task="identity_reconstruction", orchestrator="P-central",
        delegation="studio_worker", gpu_affinity="ae",
        reservation_window="whole_run",
        preempts=("llm_agent_brain", "comfy_idle_checkpoint"),
        stages=(
            Stage(name="wan_vace_view", process="P-studio", host="ae",
                  gpu_affinity="ae:cuda:0", models=(_WAN_VACE,),
                  vram_bytes_est=_WAN_PEAK, exclusive=True,
                  duration_class="long", count_expr="len(views)"),
        ),
    ),
    # ---- identity mesh build (incl. one-click generate) — §3.7 ------------
    # Sequential chain: (optional) Wan T-pose -> VLM front-select -> Hunyuan3D
    # mesh -> texture -> Blender turntable. Peak == the LARGER of the Wan render
    # and the Hunyuan mesh (both want the card; the window spans the whole chain).
    # The Hunyuan / texture / turntable stages are UNMEASURED in the doc (null) —
    # seeded with conservative estimates, awaiting p7's measured overlay.
    "identity_mesh_build": ReservationTemplate(
        task="identity_mesh_build", orchestrator="P-central",
        delegation="identity_render", gpu_affinity="ae",
        reservation_window="whole_run",
        preempts=("llm_agent_brain", "comfy_idle_checkpoint"),
        stages=(
            Stage(name="tpose_render", process="P-studio", host="ae",
                  gpu_affinity="ae:cuda:0", models=(_WAN_VACE,),
                  vram_bytes_est=_WAN_PEAK, exclusive=True,
                  duration_class="minutes", count_expr="1 if pose=='t-pose' else 0"),
            Stage(name="front_select_vlm", process="P-worker", host="ae",
                  gpu_affinity="ae:cuda:0",
                  models=({"key": "Qwen2.5-VL-3B-Instruct-GGUF",
                           "disk_bytes": 3268329184},),
                  vram_bytes_est=_gib(3.3), exclusive=False,
                  duration_class="seconds"),
            Stage(name="mesh_build", process="P-identity", host="ae",
                  gpu_affinity="ae:cuda:0", out_of_band_vram=True,
                  models=({"key": "Hunyuan3D-2mini", "disk_bytes": 25258967765},),
                  vram_bytes_est=_gib(16), exclusive=True,
                  duration_class="minutes"),
            Stage(name="texture_bake", process="P-identity", host="ae",
                  gpu_affinity="ae:cuda:0", out_of_band_vram=True,
                  models=({"key": "Hunyuan3D-2mini"},),
                  vram_bytes_est=_gib(12), exclusive=True,
                  duration_class="minutes", count_expr="1 if texture else 0"),
            Stage(name="turntable_blender", process="P-identity", host="ae",
                  gpu_affinity="ae:cuda:0", out_of_band_vram=True,
                  vram_bytes_est=_gib(4), duration_class="minutes"),
        ),
    ),
    # ---- char360 (video -> per-character 360 view-sets) — §3.8 ------------
    # Small CV models (YOLO + insightface), out-of-band. Operator lists it among
    # the exclusive-heavy set (§task a) — it still competes for the card and mesh
    # builds FAIL when the 3090 is squatted — so it gets a (modest) whole-run
    # claim. Its small peak means make-room rarely has to evict anything.
    "identity_video_extract": ReservationTemplate(
        task="identity_video_extract", orchestrator="P-central",
        delegation="identity_render", gpu_affinity="ae",
        reservation_window="whole_run",
        preempts=("llm_agent_brain",),
        stages=(
            Stage(name="cv_track_embed", process="P-identity", host="ae",
                  gpu_affinity="ae:cuda:0", out_of_band_vram=True,
                  vram_bytes_est=_gib(3), exclusive=False,
                  duration_class="minutes", count_expr="1"),
        ),
    ),
    # ---- identity from-video (k94: char360 -> one Hunyuan3D GLB per character) ----
    # ONE chained job on the render service: the cv_track_embed stage of
    # identity_video_extract followed by identity_mesh_build's mesh + texture stages
    # (per detected character). Peak == the Hunyuan mesh stage; whole-run window.
    "identity_from_video": ReservationTemplate(
        task="identity_from_video", orchestrator="P-central",
        delegation="identity_render", gpu_affinity="ae",
        reservation_window="whole_run",
        preempts=("llm_agent_brain", "comfy_idle_checkpoint"),
        stages=(
            Stage(name="cv_track_embed", process="P-identity", host="ae",
                  gpu_affinity="ae:cuda:0", out_of_band_vram=True,
                  vram_bytes_est=_gib(3), exclusive=False,
                  duration_class="minutes", count_expr="1"),
            Stage(name="mesh_build", process="P-identity", host="ae",
                  gpu_affinity="ae:cuda:0", out_of_band_vram=True,
                  models=({"key": "Hunyuan3D-2mini", "disk_bytes": 25258967765},),
                  vram_bytes_est=_gib(16), exclusive=True,
                  duration_class="minutes"),
            Stage(name="texture_bake", process="P-identity", host="ae",
                  gpu_affinity="ae:cuda:0", out_of_band_vram=True,
                  models=({"key": "Hunyuan3D-2mini"},),
                  vram_bytes_est=_gib(12), exclusive=True,
                  duration_class="minutes", count_expr="1 if texture else 0"),
        ),
    ),
    # ---- generate_movie (vision-judge co-resident peak) — §3.3 ------------
    # NOT in the operator's exclusive-heavy list, but §5's worked co-resident
    # example: peak == t2i checkpoint + vision judge held SIMULTANEOUSLY. Reserved
    # so the two-model peak doesn't collide with the brain. exclusive_heavy=False:
    # it stage-windows (only makes room if actually short), never force-evicts.
    "generate_movie": ReservationTemplate(
        task="generate_movie", orchestrator="P-central", delegation="dispatch_plane",
        gpu_affinity="ae", reservation_window="whole_run", exclusive_heavy=False,
        preempts=("llm_agent_brain",),
        co_resident=("frame_render", "vision_judge"),
        stages=(
            Stage(name="frame_render", process="P-comfy", host="ae",
                  gpu_affinity="ae:cuda:0", out_of_band_vram=True,
                  models=({"key": "comfy-dreamshaper-8", "disk_bytes": 2132647699},),
                  vram_bytes_est=_gib(4), exclusive=False,
                  duration_class="minutes", count_expr="sum(seg_n)"),
            Stage(name="vision_judge", process="P-worker", host="ae",
                  gpu_affinity="ae:cuda:0",
                  models=({"key": "Qwen2.5-VL-3B-Instruct-GGUF",
                           "disk_bytes": 3268329184},),
                  vram_bytes_est=_gib(3.3), exclusive=False,
                  duration_class="seconds",
                  count_expr="len(segments)*max_attempts"),
        ),
    ),
}


# --------------------------------------------------------------------------- #
# measured overlay (read-only consumer of p7's measured.json)
# --------------------------------------------------------------------------- #
def _read_measured() -> Dict[str, Any]:
    """Best-effort read of p7's measured.json. {} on any problem (missing file,
    parse error, wrong shape) — the estimates then stand. Never raises.

    MEMOIZED on (path, mtime, size) — k57. ``load_template`` is called once PER ROW
    by the placement projection behind GET /video/jobs, so an un-cached read meant
    one blocking open()+parse of a file on the shared /mnt storage for every job in
    the listing (a leading term in the >60s listing hang). The stat is the cache key,
    so an operator rewriting measured.json is still picked up on the next call."""
    path = _measured_path()
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        key = (path, None, None)
    with _measured_lock:
        if _measured_cache.get("key") == key:
            return _measured_cache["value"]
    value = _read_measured_uncached(path)
    with _measured_lock:
        _measured_cache["key"] = key
        _measured_cache["value"] = value
    return value


def _read_measured_uncached(path: str) -> Dict[str, Any]:
    """The actual file read + shape tolerance behind the memo."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("reservation measured.json unreadable at %s: %s", path, exc)
        return {}
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reservation measured.json at %s unparseable — using "
                       "estimates: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    # Tolerate a top-level {"tasks": {...}} or {"templates": {...}} wrapper.
    for wrap in ("tasks", "templates"):
        inner = data.get(wrap)
        if isinstance(inner, dict):
            return inner
    return data


def _as_int(v: Any) -> Optional[int]:
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _stage_measured_map(task_entry: Any) -> Tuple[Optional[int], Dict[str, int]]:
    """(task_peak_measured, {stage_name: measured_bytes}) from ONE task's entry.

    Accepts several shapes (p7 is still settling the file):
      {"peak_vram_bytes": N, "stages": {name: {"vram_bytes_measured": N}|N}}
      {"stages": {name: N}}
      {name: N, ...}  (flat stage map)
    """
    if not isinstance(task_entry, dict):
        return None, {}
    task_peak = _as_int(task_entry.get("peak_vram_bytes")
                        or task_entry.get("peak_vram_bytes_measured")
                        or task_entry.get("peak"))
    out: Dict[str, int] = {}
    stages = task_entry.get("stages")
    src = stages if isinstance(stages, dict) else task_entry
    for name, val in src.items():
        if name in ("peak_vram_bytes", "peak_vram_bytes_measured", "peak",
                    "stages", "task", "notes"):
            continue
        if isinstance(val, dict):
            b = _as_int(val.get("vram_bytes_measured")
                        or val.get("vram_bytes") or val.get("bytes")
                        or val.get("measured"))
        else:
            b = _as_int(val)
        if b is not None and b > 0:
            out[str(name)] = b
    return task_peak, out


def _apply_overlay(tmpl: ReservationTemplate,
                   measured: Dict[str, Any]) -> ReservationTemplate:
    entry = measured.get(tmpl.task)
    if entry is None:
        return tmpl
    task_peak, stage_map = _stage_measured_map(entry)
    if task_peak is None and not stage_map:
        return tmpl
    new_stages = tuple(
        replace(s, vram_bytes_measured=stage_map[s.name],
                footprint_source="measured") if s.name in stage_map else s
        for s in tmpl.stages
    )
    return replace(tmpl, stages=new_stages,
                   peak_vram_bytes_measured=task_peak
                   if task_peak is not None else tmpl.peak_vram_bytes_measured)


# --------------------------------------------------------------------------- #
# public loader
# --------------------------------------------------------------------------- #
def is_reservable(task: str) -> bool:
    """True when ``task`` (a bus job name) has a reservation template — i.e. it is
    a heavy GPU video task the engine pre-claims for. Light tasks (crop, frame/
    audio extract, generate_image, generate_scene) return False -> no claim."""
    return task in TEMPLATES


def reservable_tasks() -> Tuple[str, ...]:
    return tuple(sorted(TEMPLATES))


def load_template(task: str) -> Optional[ReservationTemplate]:
    """The reservation template for ``task`` with p7's measured overlay applied
    (measured wins), or None when ``task`` is not a reservable heavy video task.
    Best-effort: an overlay problem degrades to the built-in estimates."""
    base = TEMPLATES.get(task)
    if base is None:
        return None
    try:
        measured = _read_measured()
    except Exception:  # noqa: BLE001 — overlay is best-effort; estimates stand
        measured = {}
    if not measured:
        return base
    try:
        return _apply_overlay(base, measured)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reservation overlay for %s failed — using estimates: %s",
                       task, exc)
        return base
