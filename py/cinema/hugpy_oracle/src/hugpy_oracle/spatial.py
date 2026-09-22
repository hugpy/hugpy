"""Spatial contract (k116): Fold 1 (spatial data) → Fold 2 (geometric inference)
→ Fold 3 (stylistic rendering), as typed artifacts the oracle can validate
BEFORE a model is admitted.

    FOLD 1 SPATIAL DATA          FOLD 2 GEOMETRIC INFERENCE     FOLD 3 STYLISTIC OUTPUT
    tier 1 static rig/keyframes  tier 1 spatial token routing   tier 1 static style map
    tier 2 real-time mocap       tier 2 dense conditioning      tier 2 dynamic CFG schedule
    tier 3 simulation/physics    tier 3 neural scene (NeRF/3DGS) tier 3 neural/temporal render

Geometry is authoritative for spatial facts (invariant 10). Diffusion paints
appearance; it may not silently redefine trajectories, camera, collisions or
placement. This module owns:

* the **canonical coordinate contract** — right-handed, Y-up, −Z forward,
  metres, column-major, quaternion xyzw, metric depth in camera space — and
  lossless conversions INTO it from the other common conventions, with the
  conversion history recorded on the manifest;
* ``SpatialSceneManifest`` — the JSON control envelope (directive §8). Large
  binaries are referenced by URI + checksum, never embedded;
* ``validate_manifest`` — rejects the nine faults: unknown coordinate space,
  missing units, invalid frame range, frame-rate mismatch, missing asset,
  checksum failure, unresolved entity id, camera/conditioning resolution
  mismatch, cross-run/revision contamination;
* ``CameraIntrinsics.project`` — camera-space point → pixel / NDC / depth, so
  conditioning passes and evaluators share ONE projection;
* ``ConditioningRequest`` — the exact Fold 1 → Fold 2 payload: per-frame
  alignment (frame indices + timestamps), passes, resolution, strengths;
  ``frame_alignment_report`` proves passes line up frame-for-frame;
* ``tone_profile`` — the versioned 0–10 tone tensor → rendering controls.
  Geometry strength never drops below its floor unless the operator
  explicitly disables geometry (invariant 10 again);
* ``TierFallback`` — a fallback is an explicit, versioned record with a
  reason. Never silent.

Honesty notes baked into the types: Tier 1 token routing is *approximate
layout*, not a guarantee (``ConditioningRequest.hard_containment`` is only
true for tier ≥ 2 with masks); Tier 3 does not "eliminate" distortion — it is
measured against thresholds (``DriftThresholds``).

Stdlib only; frozen slotted dataclasses; ``to_dict``/``from_dict``; os.path.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterable

__all__ = [
    "CANONICAL", "CameraIntrinsics", "CameraSpec", "CaptureTier", "ConditioningPass",
    "ConditioningRequest", "ConditioningSpec", "CoordinateSystem", "DriftThresholds", "EntitySpec",
    "FaultCode", "InferenceTier", "ProvenanceSpec", "RenderSpec", "RenderTier", "SimulationSpec",
    "SpatialFault", "SpatialSceneManifest", "SpatialValidation", "StyleSpec", "TierFallback",
    "TierProfile", "Timebase", "ToneProfile", "convert_points", "frame_alignment_report",
    "manifest_json_schema", "tone_profile", "validate_manifest",
]

SCHEMA_VERSION = "1.0"
TONE_PROFILE_VERSION = "tone_v1"
DEFAULT_MIN_GEOMETRY_STRENGTH = 0.35


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, default=str).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# vocabularies
# --------------------------------------------------------------------------- #


class CaptureTier(int, Enum):
    STATIC_RIG = 1        # pre-authored rigs / keyframes / vertex caches (glTF/USD canonical)
    REALTIME_MOCAP = 2    # optical / skeletal / facial / hand / camera / object tracks
    SIMULATION = 3        # physics: gravity, rigid body, collision, cloth, hair, soft body


class InferenceTier(int, Enum):
    TOKEN_ROUTING = 1     # boxes / keypoints / region tokens / attention masks — approximate
    DENSE_CONDITIONING = 2  # depth, normals, silhouette, segmentation, pose, flow, occlusion
    NEURAL_SCENE = 3      # neural mesh / NeRF / 3DGS / 4DGS / neural avatar — separate backends


class RenderTier(int, Enum):
    STATIC_STYLE = 1      # fixed prompts / adapters / materials / guidance
    DYNAMIC_SCHEDULE = 2  # step- and time-dependent CFG / control / identity / style schedules
    NEURAL_RENDER = 3     # temporal diffusion + neural rendering + PBR + camera response


class ConditioningPass(str, Enum):
    BBOX = "bbox"
    KEYPOINTS = "keypoints"
    REGION_TOKENS = "region_tokens"
    DEPTH = "depth"
    NORMALS = "normals"
    SILHOUETTE = "silhouette"
    SEGMENTATION = "segmentation"
    INSTANCE = "instance"
    POSE = "pose"
    OPTICAL_FLOW = "optical_flow"
    MOTION_VECTORS = "motion_vectors"
    OCCLUSION = "occlusion"
    MATERIAL_ID = "material_id"
    NEURAL_SCENE_STATE = "neural_scene_state"


TIER1_PASSES = frozenset({ConditioningPass.BBOX, ConditioningPass.KEYPOINTS, ConditioningPass.REGION_TOKENS})
TIER2_PASSES = frozenset({ConditioningPass.DEPTH, ConditioningPass.NORMALS, ConditioningPass.SILHOUETTE,
                          ConditioningPass.SEGMENTATION, ConditioningPass.INSTANCE, ConditioningPass.POSE,
                          ConditioningPass.OPTICAL_FLOW, ConditioningPass.MOTION_VECTORS,
                          ConditioningPass.OCCLUSION, ConditioningPass.MATERIAL_ID})
HARD_CONTAINMENT_PASSES = frozenset({ConditioningPass.SILHOUETTE, ConditioningPass.SEGMENTATION,
                                     ConditioningPass.INSTANCE})


class FaultCode(str, Enum):
    UNKNOWN_COORDINATE_SPACE = "unknown_coordinate_space"
    MISSING_UNITS = "missing_units"
    INVALID_FRAME_RANGE = "invalid_frame_range"
    FRAME_RATE_MISMATCH = "frame_rate_mismatch"
    MISSING_ASSET = "missing_asset"
    CHECKSUM_FAILURE = "checksum_failure"
    UNRESOLVED_ENTITY = "unresolved_entity"
    RESOLUTION_MISMATCH = "resolution_mismatch"
    CROSS_RUN_CONTAMINATION = "cross_run_contamination"
    TIER_PASS_MISMATCH = "tier_pass_mismatch"
    GEOMETRY_FLOOR_VIOLATION = "geometry_floor_violation"


# --------------------------------------------------------------------------- #
# coordinate contract
# --------------------------------------------------------------------------- #

_HANDEDNESS = ("right", "left")
_AXES = ("X", "Y", "Z", "-X", "-Y", "-Z")
_UNITS_TO_M = {"meters": 1.0, "metres": 1.0, "m": 1.0, "centimeters": 0.01, "cm": 0.01,
               "millimeters": 0.001, "mm": 0.001, "inches": 0.0254, "feet": 0.3048}
_MATRIX_ORDER = ("column_major", "row_major")
_QUAT_ORDER = ("xyzw", "wxyz")
_DEPTH = ("metric", "normalized")


@dataclass(frozen=True, slots=True)
class CoordinateSystem:
    handedness: str = "right"
    up_axis: str = "Y"
    forward_axis: str = "-Z"
    world_units: str = "meters"
    matrix_order: str = "column_major"
    quaternion_order: str = "xyzw"
    depth_convention: str = "metric"         # metric (camera-space metres) | normalized [0,1] w/ near,far

    def __post_init__(self) -> None:
        if self.handedness not in _HANDEDNESS:
            raise ValueError(f"unknown handedness {self.handedness!r}")
        if self.up_axis not in _AXES or self.forward_axis not in _AXES:
            raise ValueError(f"unknown axis up={self.up_axis!r} forward={self.forward_axis!r}")
        if self.up_axis.lstrip("-") == self.forward_axis.lstrip("-"):
            raise ValueError("up and forward axes must differ")
        if not self.world_units:
            raise ValueError("world_units is required (missing units)")
        if self.world_units not in _UNITS_TO_M:
            raise ValueError(f"unknown world_units {self.world_units!r}")
        if self.matrix_order not in _MATRIX_ORDER or self.quaternion_order not in _QUAT_ORDER:
            raise ValueError("unknown matrix/quaternion order")
        if self.depth_convention not in _DEPTH:
            raise ValueError(f"unknown depth convention {self.depth_convention!r}")

    @property
    def scale_to_m(self) -> float:
        return _UNITS_TO_M[self.world_units]

    def to_dict(self) -> dict[str, Any]:
        return {"handedness": self.handedness, "up_axis": self.up_axis, "forward_axis": self.forward_axis,
                "world_units": self.world_units, "matrix_order": self.matrix_order,
                "quaternion_order": self.quaternion_order, "depth_convention": self.depth_convention}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CoordinateSystem":
        return cls(**{k: d[k] for k in cls.__slots__ if k in d})  # type: ignore[arg-type]


CANONICAL = CoordinateSystem()


def _axis_vec(axis: str) -> tuple[float, float, float]:
    sign = -1.0 if axis.startswith("-") else 1.0
    i = "XYZ".index(axis.lstrip("-"))
    v = [0.0, 0.0, 0.0]
    v[i] = sign
    return (v[0], v[1], v[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _basis(system: CoordinateSystem) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Rows: canonical X, Y, Z expressed in the source system's axes. The
    canonical frame is (right, up, back) = (X, Y, Z) with forward = −Z."""
    up = _axis_vec(system.up_axis)
    fwd = _axis_vec(system.forward_axis)
    right = _cross(fwd, up) if system.handedness == "right" else _cross(up, fwd)
    back = tuple(-c for c in fwd)
    return right, up, back


def convert_points(points: Sequence[Sequence[float]], src: CoordinateSystem,
                   dst: CoordinateSystem = CANONICAL) -> list[tuple[float, float, float]]:
    """Re-express ``points`` from ``src`` in ``dst`` (default canonical):
    axis permutation/sign for up/forward/handedness, plus unit scale. Lossless
    up to float rounding; ``convert_points(convert_points(p, a, b), b, a)``
    round-trips within 1e-9 relative."""
    rs, us, bs = _basis(src)
    rd, ud, bd = _basis(dst)
    scale = src.scale_to_m / dst.scale_to_m
    out: list[tuple[float, float, float]] = []
    for p in points:
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        # coordinates in the canonical (right, up, back) frame
        cr = (x * rs[0] + y * rs[1] + z * rs[2]) * scale
        cu = (x * us[0] + y * us[1] + z * us[2]) * scale
        cb = (x * bs[0] + y * bs[1] + z * bs[2]) * scale
        # back out into dst axes: dst = cr*rd + cu*ud + cb*bd
        out.append((cr * rd[0] + cu * ud[0] + cb * bd[0],
                    cr * rd[1] + cu * ud[1] + cb * bd[1],
                    cr * rd[2] + cu * ud[2] + cb * bd[2]))
    return out


# --------------------------------------------------------------------------- #
# time, camera, entities
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Timebase:
    fps: float
    start_frame: int
    end_frame: int
    duration_seconds: float

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    def frames(self) -> tuple[int, ...]:
        return tuple(range(self.start_frame, self.end_frame + 1))

    def timestamps(self) -> tuple[float, ...]:
        return tuple((f - self.start_frame) / self.fps for f in self.frames())

    def to_dict(self) -> dict[str, Any]:
        return {"fps": self.fps, "start_frame": self.start_frame, "end_frame": self.end_frame,
                "duration_seconds": self.duration_seconds}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Timebase":
        return cls(float(d["fps"]), int(d["start_frame"]), int(d["end_frame"]), float(d["duration_seconds"]))


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels. ``project`` maps a CAMERA-SPACE point
    (canonical: camera looks down −Z) to pixel, NDC [-1,1] and metric depth."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def project(self, point_cam: Sequence[float]) -> dict[str, Any] | None:
        x, y, z = float(point_cam[0]), float(point_cam[1]), float(point_cam[2])
        depth = -z                      # canonical forward is −Z: in front ⇔ z < 0
        if depth <= 1e-9:
            return None                 # behind the camera: not projectable (say so)
        u = self.fx * (x / depth) + self.cx
        v = self.cy - self.fy * (y / depth)   # image v grows downward; canonical Y is up
        ndc_x = (u / self.width) * 2.0 - 1.0
        ndc_y = 1.0 - (v / self.height) * 2.0
        return {"u": u, "v": v, "ndc": (ndc_x, ndc_y), "depth_m": depth,
                "in_frame": 0.0 <= u < self.width and 0.0 <= v < self.height}

    def normalized_depth(self, depth_m: float, near: float, far: float) -> float:
        return max(0.0, min(1.0, (depth_m - near) / (far - near)))

    def to_dict(self) -> dict[str, Any]:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CameraIntrinsics":
        return cls(float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]), int(d["width"]), int(d["height"]))


@dataclass(frozen=True, slots=True)
class CameraSpec:
    track_uri: str
    intrinsics_uri: str | None = None
    intrinsics: CameraIntrinsics | None = None
    near_meters: float = 0.1
    far_meters: float = 100.0
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"track_uri": self.track_uri, "intrinsics_uri": self.intrinsics_uri,
                "intrinsics": self.intrinsics.to_dict() if self.intrinsics else None,
                "near_meters": self.near_meters, "far_meters": self.far_meters, "checksum": self.checksum}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CameraSpec":
        intr = d.get("intrinsics")
        return cls(d["track_uri"], d.get("intrinsics_uri"),
                   CameraIntrinsics.from_dict(intr) if intr else None,
                   float(d.get("near_meters", 0.1)), float(d.get("far_meters", 100.0)), d.get("checksum"))


@dataclass(frozen=True, slots=True)
class EntitySpec:
    entity_id: str
    entity_type: str                      # character | prop | environment | light | camera_rig
    mesh_uri: str | None = None
    rig_uri: str | None = None
    animation_uri: str | None = None
    identity_reference_ids: tuple[str, ...] = ()
    checksum: str | None = None
    source_format: str | None = None      # fbx | obj | bvh | gltf | usd — import provenance

    @property
    def uris(self) -> tuple[str, ...]:
        return tuple(u for u in (self.mesh_uri, self.rig_uri, self.animation_uri) if u)

    def to_dict(self) -> dict[str, Any]:
        return {"entity_id": self.entity_id, "entity_type": self.entity_type, "mesh_uri": self.mesh_uri,
                "rig_uri": self.rig_uri, "animation_uri": self.animation_uri,
                "identity_reference_ids": list(self.identity_reference_ids), "checksum": self.checksum,
                "source_format": self.source_format}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EntitySpec":
        return cls(d["entity_id"], d.get("entity_type", "prop"), d.get("mesh_uri"), d.get("rig_uri"),
                   d.get("animation_uri"), tuple(d.get("identity_reference_ids") or ()), d.get("checksum"),
                   d.get("source_format"))


@dataclass(frozen=True, slots=True)
class SimulationSpec:
    enabled: bool = False
    cache_uri: str | None = None
    engine: str | None = None
    engine_version: str | None = None
    settings_revision: int = 0
    seed: int | None = None
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "cache_uri": self.cache_uri, "engine": self.engine,
                "engine_version": self.engine_version, "settings_revision": self.settings_revision,
                "seed": self.seed, "checksum": self.checksum}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SimulationSpec":
        return cls(bool(d.get("enabled", False)), d.get("cache_uri"), d.get("engine"), d.get("engine_version"),
                   int(d.get("settings_revision", 0)), d.get("seed"), d.get("checksum"))


@dataclass(frozen=True, slots=True)
class ConditioningSpec:
    requested_passes: tuple[ConditioningPass, ...]
    output_uri: str
    geometry_strength: float = 0.9
    identity_strength: float = 0.85
    width: int | None = None              # must equal render resolution when set
    height: int | None = None
    geometry_disabled_by_operator: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"requested_passes": [p.value for p in self.requested_passes], "output_uri": self.output_uri,
                "geometry_strength": self.geometry_strength, "identity_strength": self.identity_strength,
                "width": self.width, "height": self.height,
                "geometry_disabled_by_operator": self.geometry_disabled_by_operator}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ConditioningSpec":
        return cls(tuple(ConditioningPass(p) for p in d.get("requested_passes") or ()), d["output_uri"],
                   float(d.get("geometry_strength", 0.9)), float(d.get("identity_strength", 0.85)),
                   d.get("width"), d.get("height"), bool(d.get("geometry_disabled_by_operator", False)))


@dataclass(frozen=True, slots=True)
class StyleSpec:
    tone: float = 5.0
    profile_id: str = "balanced_v1"
    cfg_schedule_id: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.tone) <= 10.0:
            raise ValueError("tone must be within [0, 10] (operator scale; convert unit-scale "
                             "tone with oracle.tone_scale.to_operator)")

    def to_dict(self) -> dict[str, Any]:
        return {"tone": self.tone, "profile_id": self.profile_id, "cfg_schedule_id": self.cfg_schedule_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StyleSpec":
        return cls(float(d.get("tone", 5.0)), d.get("profile_id", "balanced_v1"), d.get("cfg_schedule_id"))


@dataclass(frozen=True, slots=True)
class RenderSpec:
    width: int
    height: int
    seed: int
    model_route: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "seed": self.seed, "model_route": self.model_route}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RenderSpec":
        return cls(int(d["width"]), int(d["height"]), int(d["seed"]), d.get("model_route", "auto"))


@dataclass(frozen=True, slots=True)
class ProvenanceSpec:
    generation_snapshot_id: str
    screenplay_revision: int
    shot_plan_revision: int
    continuity_revision: int
    registry_version: str | None = None
    conversion_history: tuple[str, ...] = ()   # e.g. "fbx(Z-up,cm,left)->canonical"

    def to_dict(self) -> dict[str, Any]:
        return {"generation_snapshot_id": self.generation_snapshot_id,
                "screenplay_revision": self.screenplay_revision, "shot_plan_revision": self.shot_plan_revision,
                "continuity_revision": self.continuity_revision, "registry_version": self.registry_version,
                "conversion_history": list(self.conversion_history)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ProvenanceSpec":
        return cls(d["generation_snapshot_id"], int(d["screenplay_revision"]), int(d["shot_plan_revision"]),
                   int(d["continuity_revision"]), d.get("registry_version"), tuple(d.get("conversion_history") or ()))


@dataclass(frozen=True, slots=True)
class TierProfile:
    capture: CaptureTier = CaptureTier.STATIC_RIG
    inference: InferenceTier = InferenceTier.DENSE_CONDITIONING
    render: RenderTier = RenderTier.STATIC_STYLE

    def to_dict(self) -> dict[str, Any]:
        return {"capture": int(self.capture), "inference": int(self.inference), "render": int(self.render)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TierProfile":
        return cls(CaptureTier(int(d.get("capture", 1))), InferenceTier(int(d.get("inference", 2))),
                   RenderTier(int(d.get("render", 1))))


@dataclass(frozen=True, slots=True)
class TierFallback:
    """Explicit, versioned, reproducible: which fold, from which tier to which,
    why, and the policy/version that decided it. Never silent."""
    fold: str                      # capture | inference | render
    requested: int
    actual: int
    reason: str
    decided_by: str                # e.g. "router:no NeRF backend seated" / "operator"
    policy_version: str = "tier_fallback_v1"

    def to_dict(self) -> dict[str, Any]:
        return {"fold": self.fold, "requested": self.requested, "actual": self.actual, "reason": self.reason,
                "decided_by": self.decided_by, "policy_version": self.policy_version}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TierFallback":
        return cls(d["fold"], int(d["requested"]), int(d["actual"]), d["reason"], d["decided_by"],
                   d.get("policy_version", "tier_fallback_v1"))


@dataclass(frozen=True, slots=True)
class DriftThresholds:
    """Tier 3 does not eliminate distortion; it is measured against these."""
    landmark_reprojection_px: float = 6.0
    silhouette_iou_min: float = 0.85
    depth_rel_error_max: float = 0.08
    normal_angle_deg_max: float = 12.0
    flow_warp_error_max: float = 0.06
    camera_drift_m_max: float = 0.05
    identity_similarity_min: float = 0.8
    flicker_max: float = 0.04

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SpatialSceneManifest:
    run_id: str
    segment_id: str
    artifact_revision: int
    tier_profile: TierProfile
    timebase: Timebase
    coordinate_system: CoordinateSystem
    camera: CameraSpec
    entities: tuple[EntitySpec, ...]
    conditioning: ConditioningSpec
    style: StyleSpec
    render: RenderSpec
    provenance: ProvenanceSpec
    simulation: SimulationSpec = field(default_factory=SimulationSpec)
    fallbacks: tuple[TierFallback, ...] = ()
    thresholds: DriftThresholds = field(default_factory=DriftThresholds)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "run_id": self.run_id, "segment_id": self.segment_id,
            "artifact_revision": self.artifact_revision, "tier_profile": self.tier_profile.to_dict(),
            "timebase": self.timebase.to_dict(), "coordinate_system": self.coordinate_system.to_dict(),
            "camera": self.camera.to_dict(), "entities": [e.to_dict() for e in self.entities],
            "simulation": self.simulation.to_dict(), "conditioning": self.conditioning.to_dict(),
            "style": self.style.to_dict(), "render": self.render.to_dict(),
            "provenance": self.provenance.to_dict(), "fallbacks": [f.to_dict() for f in self.fallbacks],
            "thresholds": self.thresholds.to_dict(),
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SpatialSceneManifest":
        return cls(
            run_id=d["run_id"], segment_id=d["segment_id"], artifact_revision=int(d["artifact_revision"]),
            tier_profile=TierProfile.from_dict(d.get("tier_profile") or {}),
            timebase=Timebase.from_dict(d["timebase"]),
            coordinate_system=CoordinateSystem.from_dict(d["coordinate_system"]),
            camera=CameraSpec.from_dict(d["camera"]),
            entities=tuple(EntitySpec.from_dict(e) for e in d.get("entities") or ()),
            conditioning=ConditioningSpec.from_dict(d["conditioning"]),
            style=StyleSpec.from_dict(d.get("style") or {}), render=RenderSpec.from_dict(d["render"]),
            provenance=ProvenanceSpec.from_dict(d["provenance"]),
            simulation=SimulationSpec.from_dict(d.get("simulation") or {}),
            fallbacks=tuple(TierFallback.from_dict(f) for f in d.get("fallbacks") or ()),
            thresholds=DriftThresholds(**(d.get("thresholds") or {})),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    def with_fallback(self, fb: TierFallback) -> "SpatialSceneManifest":
        tp = self.tier_profile
        if fb.fold == "capture":
            tp = replace(tp, capture=CaptureTier(fb.actual))
        elif fb.fold == "inference":
            tp = replace(tp, inference=InferenceTier(fb.actual))
        elif fb.fold == "render":
            tp = replace(tp, render=RenderTier(fb.actual))
        return replace(self, tier_profile=tp, fallbacks=self.fallbacks + (fb,),
                       artifact_revision=self.artifact_revision + 1)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SpatialFault:
    code: FaultCode
    message: str
    where: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "message": self.message, "where": self.where}


@dataclass(frozen=True, slots=True)
class SpatialValidation:
    faults: tuple[SpatialFault, ...]
    checked: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.faults

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "faults": [f.to_dict() for f in self.faults], "checked": list(self.checked)}


def validate_manifest(m: SpatialSceneManifest | Mapping[str, Any], *,
                      expected_run_id: str | None = None,
                      expected_revision: int | None = None,
                      expected_snapshot_id: str | None = None,
                      expected_fps: float | None = None,
                      known_entity_ids: Iterable[str] | None = None,
                      asset_exists: Callable[[str], bool] | None = None,
                      checksum_of: Callable[[str], str | None] | None = None,
                      min_geometry_strength: float = DEFAULT_MIN_GEOMETRY_STRENGTH,
                      fps_tolerance: float = 1e-3) -> SpatialValidation:
    """Reject before admission. Every fault is a code + message + where."""
    faults: list[SpatialFault] = []
    checked: list[str] = []

    def fault(code: FaultCode, msg: str, where: str = "") -> None:
        faults.append(SpatialFault(code, msg, where))

    # 1/2 coordinate space + units (the constructor already refuses unknowns;
    # a raw mapping is checked here so the caller gets a fault, not a trace)
    if not isinstance(m, SpatialSceneManifest):
        try:
            man = SpatialSceneManifest.from_dict(m)
        except (KeyError, ValueError, TypeError) as exc:
            msg = str(exc)
            code = FaultCode.MISSING_UNITS if "units" in msg else FaultCode.UNKNOWN_COORDINATE_SPACE
            if "tone" in msg or "frame" in msg:
                code = FaultCode.INVALID_FRAME_RANGE if "frame" in msg else FaultCode.UNKNOWN_COORDINATE_SPACE
            fault(code, f"manifest does not parse: {msg}", "manifest")
            return SpatialValidation(tuple(faults), ("parse",))
    else:
        man = m
    checked += ["coordinate_space", "units"]

    # 3 frame range
    tb = man.timebase
    if tb.fps <= 0:
        fault(FaultCode.INVALID_FRAME_RANGE, f"fps must be > 0, got {tb.fps}", "timebase.fps")
    if tb.start_frame < 0 or tb.end_frame < tb.start_frame:
        fault(FaultCode.INVALID_FRAME_RANGE, f"frames {tb.start_frame}..{tb.end_frame} invalid", "timebase")
    elif tb.fps > 0:
        expect = tb.frame_count / tb.fps
        if abs(expect - tb.duration_seconds) > (1.0 / tb.fps) + 1e-6:
            fault(FaultCode.INVALID_FRAME_RANGE,
                  f"duration {tb.duration_seconds}s disagrees with {tb.frame_count} frames @ {tb.fps} fps "
                  f"(= {expect:.4f}s)", "timebase.duration_seconds")
    checked.append("frame_range")

    # 4 frame rate
    if expected_fps is not None and abs(float(expected_fps) - tb.fps) > fps_tolerance:
        fault(FaultCode.FRAME_RATE_MISMATCH, f"manifest fps {tb.fps} != production fps {expected_fps}", "timebase.fps")
    checked.append("frame_rate")

    # 5/6 assets + checksums
    uris: list[tuple[str, str | None, str]] = [(man.camera.track_uri, man.camera.checksum, "camera.track_uri")]
    if man.camera.intrinsics_uri:
        uris.append((man.camera.intrinsics_uri, None, "camera.intrinsics_uri"))
    for e in man.entities:
        for u in e.uris:
            uris.append((u, e.checksum, f"entities[{e.entity_id}]"))
    if man.simulation.enabled and man.simulation.cache_uri:
        uris.append((man.simulation.cache_uri, man.simulation.checksum, "simulation.cache_uri"))
    elif man.simulation.enabled and not man.simulation.cache_uri:
        fault(FaultCode.MISSING_ASSET, "simulation enabled without a cache_uri", "simulation")
    for uri, chk, where in uris:
        if asset_exists is not None and not asset_exists(uri):
            fault(FaultCode.MISSING_ASSET, f"asset not available: {uri}", where)
            continue
        if chk is not None:
            if not chk.startswith("sha256:") or len(chk) != 7 + 64:
                fault(FaultCode.CHECKSUM_FAILURE, f"checksum malformed: {chk}", where)
            elif checksum_of is not None:
                actual = checksum_of(uri)
                if actual is not None and actual != chk:
                    fault(FaultCode.CHECKSUM_FAILURE, f"checksum mismatch for {uri}", where)
    checked += ["assets", "checksums"]

    # 7 entity ids
    ids = [e.entity_id for e in man.entities]
    if len(set(ids)) != len(ids):
        fault(FaultCode.UNRESOLVED_ENTITY, "duplicate entity ids", "entities")
    if known_entity_ids is not None:
        known = set(known_entity_ids)
        for e in man.entities:
            if e.entity_id not in known:
                fault(FaultCode.UNRESOLVED_ENTITY, f"entity {e.entity_id!r} is not in the continuity bible", f"entities[{e.entity_id}]")
    for e in man.entities:
        if e.entity_type == "character" and not e.identity_reference_ids:
            fault(FaultCode.UNRESOLVED_ENTITY, f"character {e.entity_id!r} has no identity references", f"entities[{e.entity_id}]")
    checked.append("entities")

    # 8 resolution
    c = man.conditioning
    if c.width is not None and c.height is not None and (c.width, c.height) != (man.render.width, man.render.height):
        fault(FaultCode.RESOLUTION_MISMATCH,
              f"conditioning {c.width}x{c.height} != render {man.render.width}x{man.render.height}", "conditioning")
    intr = man.camera.intrinsics
    if intr is not None and (intr.width, intr.height) != (man.render.width, man.render.height):
        fault(FaultCode.RESOLUTION_MISMATCH,
              f"camera intrinsics {intr.width}x{intr.height} != render {man.render.width}x{man.render.height}", "camera.intrinsics")
    checked.append("resolution")

    # 9 cross-run / revision contamination
    if expected_run_id is not None and man.run_id != expected_run_id:
        fault(FaultCode.CROSS_RUN_CONTAMINATION, f"manifest run {man.run_id!r} != run {expected_run_id!r}", "run_id")
    if expected_revision is not None and man.artifact_revision != expected_revision:
        fault(FaultCode.CROSS_RUN_CONTAMINATION,
              f"manifest revision {man.artifact_revision} != expected {expected_revision}", "artifact_revision")
    if expected_snapshot_id is not None and man.provenance.generation_snapshot_id != expected_snapshot_id:
        fault(FaultCode.CROSS_RUN_CONTAMINATION, "provenance.generation_snapshot_id belongs to another run", "provenance")
    checked.append("provenance")

    # tier / pass coherence + geometry floor (honesty checks)
    passes = set(c.requested_passes)
    if man.tier_profile.inference is InferenceTier.TOKEN_ROUTING and passes & TIER2_PASSES:
        fault(FaultCode.TIER_PASS_MISMATCH, "dense passes requested under inference tier 1 (token routing)", "conditioning")
    if man.tier_profile.inference is InferenceTier.DENSE_CONDITIONING and not (passes & TIER2_PASSES):
        fault(FaultCode.TIER_PASS_MISMATCH, "inference tier 2 without any dense pass", "conditioning")
    if not c.geometry_disabled_by_operator and c.geometry_strength < min_geometry_strength:
        fault(FaultCode.GEOMETRY_FLOOR_VIOLATION,
              f"geometry_strength {c.geometry_strength} < floor {min_geometry_strength} and geometry not explicitly disabled",
              "conditioning.geometry_strength")
    checked += ["tier_passes", "geometry_floor"]
    return SpatialValidation(tuple(faults), tuple(checked))


# --------------------------------------------------------------------------- #
# Fold 1 -> Fold 2 payload
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ConditioningRequest:
    """What the geometric-inference layer consumes for ONE segment: the frame
    grid it must align to, the passes to render from the authoritative camera,
    the resolution, the strengths, and whether containment is hard."""
    run_id: str
    segment_id: str
    manifest_digest: str
    frames: tuple[int, ...]
    timestamps: tuple[float, ...]
    fps: float
    width: int
    height: int
    camera_track_uri: str
    passes: tuple[ConditioningPass, ...]
    geometry_strength: float
    identity_strength: float
    entity_ids: tuple[str, ...]
    inference_tier: InferenceTier
    hard_containment: bool
    output_uri: str
    coordinate_system: CoordinateSystem = CANONICAL

    @classmethod
    def from_manifest(cls, m: SpatialSceneManifest) -> "ConditioningRequest":
        passes = tuple(m.conditioning.requested_passes)
        hard = m.tier_profile.inference >= InferenceTier.DENSE_CONDITIONING and bool(set(passes) & HARD_CONTAINMENT_PASSES)
        return cls(
            run_id=m.run_id, segment_id=m.segment_id, manifest_digest=m.digest,
            frames=m.timebase.frames(), timestamps=m.timebase.timestamps(), fps=m.timebase.fps,
            width=m.render.width, height=m.render.height, camera_track_uri=m.camera.track_uri,
            passes=passes, geometry_strength=m.conditioning.geometry_strength,
            identity_strength=m.conditioning.identity_strength,
            entity_ids=tuple(e.entity_id for e in m.entities), inference_tier=m.tier_profile.inference,
            hard_containment=hard, output_uri=m.conditioning.output_uri, coordinate_system=m.coordinate_system,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "segment_id": self.segment_id, "manifest_digest": self.manifest_digest,
            "frames": list(self.frames), "timestamps": list(self.timestamps), "fps": self.fps,
            "width": self.width, "height": self.height, "camera_track_uri": self.camera_track_uri,
            "passes": [p.value for p in self.passes], "geometry_strength": self.geometry_strength,
            "identity_strength": self.identity_strength, "entity_ids": list(self.entity_ids),
            "inference_tier": int(self.inference_tier), "hard_containment": self.hard_containment,
            "guarantee": ("hard geometric boundaries via masks/validation" if self.hard_containment
                          else "approximate layout only — not a geometric guarantee"),
            "output_uri": self.output_uri, "coordinate_system": self.coordinate_system.to_dict(),
        }


def frame_alignment_report(request: ConditioningRequest, produced_frames: Sequence[int],
                           produced_timestamps: Sequence[float] | None = None, *,
                           tolerance_s: float | None = None) -> dict[str, Any]:
    """Prove a conditioning pass (or a render) lines up frame-for-frame with
    the manifest's timebase. Missing, extra and time-shifted frames are listed."""
    tol = tolerance_s if tolerance_s is not None else 0.5 / request.fps
    want = list(request.frames)
    got = list(produced_frames)
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    shifted: list[dict[str, Any]] = []
    if produced_timestamps is not None:
        ts = {f: t for f, t in zip(got, produced_timestamps)}
        for f, t in zip(request.frames, request.timestamps):
            if f in ts and abs(ts[f] - t) > tol:
                shifted.append({"frame": f, "expected_s": t, "actual_s": ts[f]})
    aligned = not missing and not extra and not shifted and got == want
    return {"aligned": aligned, "expected": len(want), "produced": len(got), "missing": missing,
            "extra": extra, "shifted": shifted, "order_ok": got == want, "tolerance_s": tol}


# --------------------------------------------------------------------------- #
# tone (Fold 3)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ToneProfile:
    version: str
    tone: float
    render_tier: RenderTier
    controls: Mapping[str, Any]
    cfg_schedule: tuple[tuple[float, float], ...]          # (t in [0,1], cfg)
    geometry_schedule: tuple[tuple[float, float], ...]     # (t, geometry strength)
    positive_style: str
    negative_style: str
    geometry_floor: float
    geometry_disabled: bool

    @property
    def profile_id(self) -> str:
        return f"{self.version}:tone{self.tone:.1f}:r{int(self.render_tier)}"

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "profile_id": self.profile_id, "tone": self.tone,
                "render_tier": int(self.render_tier), "controls": dict(self.controls),
                "cfg_schedule": [list(p) for p in self.cfg_schedule],
                "geometry_schedule": [list(p) for p in self.geometry_schedule],
                "positive_style": self.positive_style, "negative_style": self.negative_style,
                "geometry_floor": self.geometry_floor, "geometry_disabled": self.geometry_disabled}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def tone_profile(tone: float, *, render_tier: RenderTier = RenderTier.STATIC_STYLE,
                 base_geometry_strength: float = 0.9,
                 geometry_floor: float = DEFAULT_MIN_GEOMETRY_STRENGTH,
                 geometry_disabled: bool = False, steps: int = 8) -> ToneProfile:
    """The 0–10 tone tensor → rendering controls. 0 = photoreal/PBR,
    5 = balanced, 10 = graphic/vector. Style changes; geometry conditioning
    retains at least ``geometry_floor`` unless ``geometry_disabled`` was set
    by the operator (and then the profile SAYS so)."""
    from hugpy_oracle.tone_scale import to_unit
    t = to_unit(tone)   # the one conversion point; refuses out-of-range
    controls = {
        "cfg": round(_lerp(5.5, 8.5, t), 3),
        "style_strength": round(_lerp(0.15, 0.95, t), 3),
        "lighting_complexity": round(_lerp(1.0, 0.15, t), 3),      # path-traced .. flat
        "shading": "pbr" if t < 0.35 else ("stylized" if t < 0.75 else "flat"),
        "edge_treatment": round(_lerp(0.0, 1.0, t), 3),            # none .. clean vector lines
        "texture_detail": round(_lerp(1.0, 0.2, t), 3),
        "motion_blur": round(_lerp(0.8, 0.1, t), 3),
        "subsurface_scattering": t < 0.35 and render_tier is RenderTier.NEURAL_RENDER,
        "ambient_occlusion": round(_lerp(1.0, 0.3, t), 3),
        "temporal_consistency": round(_lerp(0.9, 0.7, t), 3),
        "denoise_strength": round(_lerp(0.55, 0.8, t), 3),
        "color_response": "photographic" if t < 0.35 else ("filmic" if t < 0.75 else "graphic"),
    }
    geo = 0.0 if geometry_disabled else max(geometry_floor, base_geometry_strength - 0.25 * t)
    controls["geometry_strength"] = round(geo, 3)
    controls["identity_strength"] = round(_lerp(0.9, 0.75, t), 3)
    positive = ("photorealistic, physically based lighting, natural skin and fabric, lens response"
                if t < 0.35 else
                "stylized illustration, controlled shading, coherent palette" if t < 0.75 else
                "clean vector lines, flat graphic shading, simplified materials, bold shapes")
    negative = ("cartoon, flat shading, outlines, oversaturated" if t < 0.35 else
                "photographic noise, heavy texture grain" if t >= 0.75 else "")
    if render_tier is RenderTier.STATIC_STYLE:
        cfg_schedule = ((0.0, controls["cfg"]), (1.0, controls["cfg"]))
        geo_schedule = ((0.0, geo), (1.0, geo))
    else:
        # dynamic: geometry strongest early (structure), style/CFG rising late (appearance)
        cfg_schedule = tuple((round(i / (steps - 1), 4),
                              round(_lerp(controls["cfg"] * 0.8, controls["cfg"] * 1.1, i / (steps - 1)), 3))
                             for i in range(steps))
        geo_schedule = tuple((round(i / (steps - 1), 4),
                              round(max(geometry_floor if not geometry_disabled else 0.0,
                                        _lerp(geo, geo * 0.7, i / (steps - 1))), 3))
                             for i in range(steps))
    return ToneProfile(TONE_PROFILE_VERSION, float(tone), render_tier, controls, cfg_schedule, geo_schedule,
                       positive, negative, geometry_floor, geometry_disabled)


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def manifest_json_schema() -> dict[str, Any]:
    """Hand-rolled JSON Schema for the control envelope (the repo convention)."""
    def enum(vals):
        return {"type": "string", "enum": list(vals)}
    num = {"type": "number"}
    integer = {"type": "integer"}
    string = {"type": "string"}
    uri = {"type": "string", "pattern": r"^[a-z][a-z0-9+.-]*://"}
    checksum = {"type": ["string", "null"], "pattern": r"^sha256:[0-9a-f]{64}$"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SpatialSceneManifest", "type": "object",
        "required": ["schema_version", "run_id", "segment_id", "artifact_revision", "tier_profile", "timebase",
                     "coordinate_system", "camera", "entities", "conditioning", "style", "render", "provenance"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION}, "run_id": string, "segment_id": string,
            "artifact_revision": {"type": "integer", "minimum": 0},
            "tier_profile": {"type": "object", "properties": {
                "capture": {"type": "integer", "enum": [1, 2, 3]}, "inference": {"type": "integer", "enum": [1, 2, 3]},
                "render": {"type": "integer", "enum": [1, 2, 3]}}, "required": ["capture", "inference", "render"]},
            "timebase": {"type": "object", "required": ["fps", "start_frame", "end_frame", "duration_seconds"],
                         "properties": {"fps": {"type": "number", "exclusiveMinimum": 0}, "start_frame": integer,
                                        "end_frame": integer, "duration_seconds": num}},
            "coordinate_system": {"type": "object",
                                  "required": ["handedness", "up_axis", "forward_axis", "world_units", "matrix_order", "quaternion_order"],
                                  "properties": {"handedness": enum(_HANDEDNESS), "up_axis": enum(_AXES),
                                                 "forward_axis": enum(_AXES), "world_units": enum(_UNITS_TO_M),
                                                 "matrix_order": enum(_MATRIX_ORDER), "quaternion_order": enum(_QUAT_ORDER),
                                                 "depth_convention": enum(_DEPTH)}},
            "camera": {"type": "object", "required": ["track_uri"],
                       "properties": {"track_uri": uri, "intrinsics_uri": {"type": ["string", "null"]},
                                      "near_meters": num, "far_meters": num, "checksum": checksum}},
            "entities": {"type": "array", "items": {"type": "object", "required": ["entity_id", "entity_type"],
                                                     "properties": {"entity_id": string, "entity_type": string,
                                                                    "mesh_uri": {"type": ["string", "null"]},
                                                                    "rig_uri": {"type": ["string", "null"]},
                                                                    "animation_uri": {"type": ["string", "null"]},
                                                                    "identity_reference_ids": {"type": "array", "items": string},
                                                                    "checksum": checksum}}},
            "simulation": {"type": "object"},
            "conditioning": {"type": "object", "required": ["requested_passes", "output_uri"],
                             "properties": {"requested_passes": {"type": "array", "items": enum([p.value for p in ConditioningPass])},
                                            "output_uri": uri, "geometry_strength": {"type": "number", "minimum": 0, "maximum": 1},
                                            "identity_strength": {"type": "number", "minimum": 0, "maximum": 1}}},
            "style": {"type": "object", "properties": {"tone": {"type": "number", "minimum": 0, "maximum": 10},
                                                       "profile_id": string, "cfg_schedule_id": {"type": ["string", "null"]}}},
            "render": {"type": "object", "required": ["width", "height", "seed"],
                       "properties": {"width": integer, "height": integer, "seed": integer, "model_route": string}},
            "provenance": {"type": "object", "required": ["generation_snapshot_id", "screenplay_revision",
                                                          "shot_plan_revision", "continuity_revision"]},
            "fallbacks": {"type": "array"},
        },
        "additionalProperties": True,
    }
