"""k119 (or-k7) — spatial adherence evaluators for the ``spatial:<seg>`` gate.

Tier 3 does not eliminate distortion; it is MEASURED (``spatial.DriftThresholds``).
This module is the measuring instrument: pure numpy, no GPU, no catalog, no
disk. Every evaluator takes the locked :class:`~.spatial.SpatialSceneManifest`
and a mapping of PRODUCED observations (arrays the render/analysis workers
wrote — frames, depth, normals, flow, camera poses, entity bounds) and returns
ONE metric dict::

    {"metric": "reprojection_px", "value": 3.1, "threshold": 6.0, "ok": True,
     "code": None, "evidence": {...}, "skipped": None}

``ok`` is ``None`` and ``skipped`` names the missing input when the observation
set cannot answer the question — an instrument that was never pointed at the
scene reports "not measured", never "passed".

THRESHOLD → CODE. A failing metric names WHAT to regenerate, via the three
spatial repair codes ``repair_controller`` already maps to bounded subgraphs:

    reprojection / silhouette / depth / normals / flow  → GEOMETRY_DRIFT
    camera translation or rotation off the locked track → CAMERA_PATH_MISMATCH
    simulation contacts contradicted                    → COLLISION_VIOLATION

OBSERVATION KEYS (all optional; arrays are anything ``numpy.asarray`` reads)

    landmarks_expected / landmarks_observed   (N,2) or (F,N,2) pixels
    points_world + poses_observed             (N,3) + (F,4,4) world→camera,
                                              used to DERIVE landmarks_expected
                                              through the manifest intrinsics
    silhouette_expected / silhouette_observed (H,W) or (F,H,W) bool/0-1
    depth_expected / depth_observed           (H,W) or (F,H,W) metric metres
    normals_expected / normals_observed       (...,3) unit or unnormalized
    frames + flow                             (F,H,W[,C]) in [0,1] or [0,255]
                                              and (F-1,H,W,2) forward flow (dx,dy)
    flow_expected / flow_observed             (...,2) — normalized endpoint error
    poses_expected / poses_observed           (F,4,4) or (F,3) camera positions
                                              in the manifest coordinate system
    entity_bounds_observed                    {entity_id: (F,2,3) AABB min/max}
    contacts_expected                         [{"a","b","frame","kind"}] from the
                                              simulation; kind = touch|separate
    static_colliders                          {id: (2,3) AABB} — ground, walls

``spatial_rubrics()`` exposes name → callable so ``evaluation.py`` can register
the set as rubrics for the ``spatial:<seg>`` gate without importing numpy
itself; ``evaluate_spatial`` runs them all and folds the result into
``contracts.Check`` rows plus the distinct repair codes, in a stable order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hugpy_oracle.contracts import Check, CheckKind, RepairCode
from hugpy_oracle.spatial import (
    CameraIntrinsics,
    CoordinateSystem,
    DriftThresholds,
    SpatialSceneManifest,
)

Observations = Mapping[str, Any]
Evaluator = Callable[..., dict[str, Any]]

#: Metric name → the repair code its failure emits. Closed: a metric that is
#: not in this table cannot fail the gate, only report.
METRIC_CODES: dict[str, RepairCode] = {
    "reprojection_px": RepairCode.GEOMETRY_DRIFT,
    "silhouette_iou": RepairCode.GEOMETRY_DRIFT,
    "depth_rel_error": RepairCode.GEOMETRY_DRIFT,
    "normal_angle_deg": RepairCode.GEOMETRY_DRIFT,
    "flow_warp_error": RepairCode.GEOMETRY_DRIFT,
    "camera_drift_m": RepairCode.CAMERA_PATH_MISMATCH,
    "collision": RepairCode.COLLISION_VIOLATION,
}

#: Camera rotation off the locked track, in degrees, that counts as a path
#: mismatch on its own even when translation is within threshold. Not a
#: ``DriftThresholds`` field; the contract there is metric-metres.
CAMERA_ROTATION_DEG_MAX: float = 2.0

#: Penetration / separation slack for contact checks, metres.
CONTACT_TOLERANCE_M: float = 0.02


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _thresholds(manifest: SpatialSceneManifest | Mapping[str, Any] | None,
                thresholds: DriftThresholds | None) -> DriftThresholds:
    if thresholds is not None:
        return thresholds
    if isinstance(manifest, SpatialSceneManifest):
        return manifest.thresholds
    if isinstance(manifest, Mapping) and manifest.get("thresholds"):
        return DriftThresholds(**dict(manifest["thresholds"]))
    return DriftThresholds()


def _scale_to_m(manifest: Any) -> float:
    cs = getattr(manifest, "coordinate_system", None)
    if isinstance(cs, CoordinateSystem):
        return cs.scale_to_m
    if isinstance(manifest, Mapping) and manifest.get("coordinate_system"):
        return CoordinateSystem.from_dict(manifest["coordinate_system"]).scale_to_m
    return 1.0


def _intrinsics(manifest: Any) -> CameraIntrinsics | None:
    cam = getattr(manifest, "camera", None)
    intr = getattr(cam, "intrinsics", None)
    if isinstance(intr, CameraIntrinsics):
        return intr
    if isinstance(manifest, Mapping):
        d = (manifest.get("camera") or {}).get("intrinsics")
        if d:
            return CameraIntrinsics.from_dict(d)
    return None


def _arr(obs: Observations, key: str, dtype: Any = np.float64) -> np.ndarray | None:
    value = obs.get(key)
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def _metric(name: str, value: float | None, threshold: float | None, ok: bool | None,
            evidence: Mapping[str, Any] | None = None, skipped: str | None = None,
            *, higher_is_better: bool = False) -> dict[str, Any]:
    code = METRIC_CODES.get(name) if ok is False else None
    return {"metric": name,
            "value": None if value is None else float(value),
            "threshold": None if threshold is None else float(threshold),
            "ok": ok, "code": code, "higher_is_better": higher_is_better,
            "evidence": dict(evidence or {}), "skipped": skipped}


def _skip(name: str, threshold: float | None, reason: str,
          higher_is_better: bool = False) -> dict[str, Any]:
    return _metric(name, None, threshold, None, skipped=reason,
                   higher_is_better=higher_is_better)


def _per_frame(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values).ravel()]


def _worst_frames(per_frame: Sequence[float], n: int = 3, largest: bool = True) -> list[int]:
    order = sorted(range(len(per_frame)), key=lambda i: per_frame[i], reverse=largest)
    return order[:n]


def project_points(intrinsics: CameraIntrinsics, points_cam: np.ndarray) -> np.ndarray:
    """Pinhole projection of CAMERA-SPACE points (N,3) → pixels (N,2), matching
    ``CameraIntrinsics.project``; points behind the camera come back NaN."""
    p = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    depth = -p[:, 2]
    out = np.full((p.shape[0], 2), np.nan)
    front = depth > 1e-9
    out[front, 0] = intrinsics.fx * (p[front, 0] / depth[front]) + intrinsics.cx
    out[front, 1] = intrinsics.cy - intrinsics.fy * (p[front, 1] / depth[front])
    return out


def transform_points(pose_w2c: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    """Apply a (4,4) world→camera pose to (N,3) world points."""
    p = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    h = np.concatenate([p, np.ones((p.shape[0], 1))], axis=1)
    return (np.asarray(pose_w2c, dtype=np.float64) @ h.T).T[:, :3]


def pose_positions(poses: np.ndarray) -> np.ndarray:
    """Camera CENTRES (F,3) from (F,4,4) world→camera poses or (F,3) already."""
    a = np.asarray(poses, dtype=np.float64)
    if a.ndim == 2 and a.shape[1] == 3:
        return a
    if a.ndim == 3 and a.shape[1:] == (4, 4):
        rot = a[:, :3, :3]
        t = a[:, :3, 3]
        return -np.einsum("fji,fj->fi", rot, t)      # C = -R^T t
    if a.ndim == 2 and a.shape == (4, 4):
        return pose_positions(a[None])
    raise ValueError(f"poses must be (F,4,4) or (F,3), got shape {a.shape}")


def pose_rotations_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Per-frame rotation angle between two (F,4,4) pose stacks, degrees; None
    when either side carries positions only."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 3 or b.ndim != 3:
        return None
    rel = np.einsum("fij,fkj->fik", a[:, :3, :3], b[:, :3, :3])   # Ra Rb^T
    tr = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


# --------------------------------------------------------------------------- #
# evaluators — each (manifest, observations, thresholds=None) → metric dict
# --------------------------------------------------------------------------- #


def reprojection_error(manifest: Any, obs: Observations,
                       thresholds: DriftThresholds | None = None) -> dict[str, Any]:
    """Mean pixel distance between where the locked geometry says each
    landmark lands and where the render put it. Expected landmarks are taken
    verbatim or DERIVED from ``points_world`` + ``poses_observed`` through the
    manifest intrinsics (the render is judged against its own camera, so a
    camera fault is not double-counted here)."""
    th = _thresholds(manifest, thresholds).landmark_reprojection_px
    observed = _arr(obs, "landmarks_observed")
    expected = _arr(obs, "landmarks_expected")
    if expected is None and obs.get("points_world") is not None:
        intr = _intrinsics(manifest)
        poses = _arr(obs, "poses_observed")
        if intr is None:
            return _skip("reprojection_px", th, "manifest carries no intrinsics")
        if poses is None:
            return _skip("reprojection_px", th, "points_world needs poses_observed")
        if poses.ndim == 2:
            poses = poses[None]
        pts = _arr(obs, "points_world") * _scale_to_m(manifest)
        expected = np.stack([project_points(intr, transform_points(p, pts)) for p in poses])
    if observed is None or expected is None:
        return _skip("reprojection_px", th, "needs landmarks_expected and landmarks_observed")
    if expected.ndim == 2:
        expected = expected[None]
    if observed.ndim == 2:
        observed = observed[None]
    if expected.shape != observed.shape:
        return _skip("reprojection_px", th,
                     f"landmark shapes differ {expected.shape} vs {observed.shape}")
    d = np.linalg.norm(observed - expected, axis=-1)             # (F,N)
    valid = np.isfinite(d)
    if not valid.any():
        return _skip("reprojection_px", th, "no finite landmark pairs")
    per_frame = np.array([float(np.nanmean(row)) if np.isfinite(row).any() else np.nan
                          for row in d])
    mean = float(np.nanmean(d))
    worst = float(np.nanmax(d))
    ok = mean <= th
    return _metric("reprojection_px", mean, th, ok, {
        "max_px": worst, "frames": int(d.shape[0]), "landmarks": int(d.shape[1]),
        "per_frame_px": _per_frame(np.nan_to_num(per_frame, nan=-1.0)),
        "worst_frames": _worst_frames(list(np.nan_to_num(per_frame, nan=-1.0))),
        "landmarks_over": int(np.sum(np.nan_to_num(d, nan=0.0) > th))})


def _mask(a: np.ndarray) -> np.ndarray:
    return np.asarray(a) > 0.5


def silhouette_iou(manifest: Any, obs: Observations,
                   thresholds: DriftThresholds | None = None) -> dict[str, Any]:
    """Intersection over union of the entity silhouette the render drew and
    the one the geometry pass projected. Mean over frames; empty-vs-empty
    frames count as IoU 1 (nothing was supposed to be there, nothing was)."""
    th = _thresholds(manifest, thresholds).silhouette_iou_min
    e = obs.get("silhouette_expected")
    o = obs.get("silhouette_observed")
    if e is None or o is None:
        return _skip("silhouette_iou", th, "needs silhouette_expected and silhouette_observed",
                     higher_is_better=True)
    e, o = _mask(e), _mask(o)
    if e.ndim == 2:
        e, o = e[None], o[None]
    if e.shape != o.shape:
        return _skip("silhouette_iou", th, f"mask shapes differ {e.shape} vs {o.shape}",
                     higher_is_better=True)
    inter = np.logical_and(e, o).reshape(e.shape[0], -1).sum(axis=1)
    union = np.logical_or(e, o).reshape(e.shape[0], -1).sum(axis=1)
    per_frame = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    mean = float(per_frame.mean())
    return _metric("silhouette_iou", mean, th, mean >= th, {
        "min_iou": float(per_frame.min()), "frames": int(e.shape[0]),
        "per_frame_iou": _per_frame(per_frame),
        "worst_frames": _worst_frames(list(per_frame), largest=False),
        "expected_area_px": int(e.sum()), "observed_area_px": int(o.sum())},
        higher_is_better=True)


def depth_consistency(manifest: Any, obs: Observations,
                      thresholds: DriftThresholds | None = None) -> dict[str, Any]:
    """Median relative depth error |d_obs − d_exp| / d_exp over pixels where
    the expected depth is valid (> 0, finite). Median, not mean: a handful of
    edge pixels at a depth discontinuity is not geometry drift."""
    th = _thresholds(manifest, thresholds).depth_rel_error_max
    e = _arr(obs, "depth_expected")
    o = _arr(obs, "depth_observed")
    if e is None or o is None:
        return _skip("depth_rel_error", th, "needs depth_expected and depth_observed")
    if e.shape != o.shape:
        return _skip("depth_rel_error", th, f"depth shapes differ {e.shape} vs {o.shape}")
    valid = np.isfinite(e) & np.isfinite(o) & (e > 1e-6)
    if not valid.any():
        return _skip("depth_rel_error", th, "no valid expected depth")
    rel = np.zeros_like(e)
    rel[valid] = np.abs(o[valid] - e[valid]) / e[valid]
    frames = rel if rel.ndim == 3 else rel[None]
    vframes = valid if valid.ndim == 3 else valid[None]
    per_frame = np.array([float(np.median(f[v])) if v.any() else 0.0
                          for f, v in zip(frames, vframes)])
    median = float(np.median(rel[valid]))
    return _metric("depth_rel_error", median, th, median <= th, {
        "p90": float(np.percentile(rel[valid], 90)), "valid_px": int(valid.sum()),
        "per_frame_median": _per_frame(per_frame),
        "worst_frames": _worst_frames(list(per_frame)),
        "scale_bias": float(np.median(o[valid] / e[valid]))})


def normal_consistency(manifest: Any, obs: Observations,
                       thresholds: DriftThresholds | None = None) -> dict[str, Any]:
    """Mean angle (degrees) between expected and observed surface normals,
    over pixels where both are non-degenerate."""
    th = _thresholds(manifest, thresholds).normal_angle_deg_max
    e = _arr(obs, "normals_expected")
    o = _arr(obs, "normals_observed")
    if e is None or o is None:
        return _skip("normal_angle_deg", th, "needs normals_expected and normals_observed")
    if e.shape != o.shape or e.shape[-1] != 3:
        return _skip("normal_angle_deg", th, f"normal shapes differ/invalid {e.shape} vs {o.shape}")
    ne = np.linalg.norm(e, axis=-1)
    no = np.linalg.norm(o, axis=-1)
    valid = (ne > 1e-9) & (no > 1e-9)
    if not valid.any():
        return _skip("normal_angle_deg", th, "no valid normals")
    cos = np.sum(e * o, axis=-1) / np.maximum(ne * no, 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    mean = float(ang[valid].mean())
    flipped = float(np.mean(cos[valid] < 0.0))
    return _metric("normal_angle_deg", mean, th, mean <= th, {
        "p90_deg": float(np.percentile(ang[valid], 90)), "valid_px": int(valid.sum()),
        "flipped_fraction": flipped})


def _warp_back(frame_next: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``frame_next`` at (x+dx, y+dy) with bilinear interpolation.
    Returns (warped, valid) where valid marks samples inside the image."""
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    x = xs + flow[..., 0]
    y = ys + flow[..., 1]
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x0 = np.clip(np.floor(x).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = np.clip(x - x0, 0.0, 1.0)[..., None]
    wy = np.clip(y - y0, 0.0, 1.0)[..., None]
    img = frame_next if frame_next.ndim == 3 else frame_next[..., None]
    top = img[y0, x0] * (1 - wx) + img[y0, x1] * wx
    bot = img[y1, x0] * (1 - wx) + img[y1, x1] * wx
    return top * (1 - wy) + bot * wy, valid


def flow_warp_error(manifest: Any, obs: Observations,
                    thresholds: DriftThresholds | None = None) -> dict[str, Any]:
    """Temporal geometry consistency. With ``frames`` + ``flow``: warp frame
    t+1 back onto t along the forward flow and take the mean absolute
    photometric residual on a [0,1] scale. With ``flow_expected`` /
    ``flow_observed`` only: mean endpoint error normalized by the image
    diagonal (so the same threshold reads for both forms)."""
    th = _thresholds(manifest, thresholds).flow_warp_error_max
    frames = _arr(obs, "frames")
    flow = _arr(obs, "flow")
    if frames is not None and flow is not None:
        if frames.ndim not in (3, 4) or flow.ndim != 4 or flow.shape[-1] != 2:
            return _skip("flow_warp_error", th, "frames must be (F,H,W[,C]) and flow (F-1,H,W,2)")
        if flow.shape[0] != frames.shape[0] - 1 or flow.shape[1:3] != frames.shape[1:3]:
            return _skip("flow_warp_error", th,
                         f"flow {flow.shape} does not pair with frames {frames.shape}")
        scale = 255.0 if frames.max() > 1.0 else 1.0
        f = frames / scale
        per_frame = []
        for t in range(flow.shape[0]):
            warped, valid = _warp_back(f[t + 1], flow[t])
            cur = f[t] if f[t].ndim == 3 else f[t][..., None]
            resid = np.abs(cur - warped).mean(axis=-1)
            per_frame.append(float(resid[valid].mean()) if valid.any() else 0.0)
        pf = np.array(per_frame)
        mean = float(pf.mean())
        return _metric("flow_warp_error", mean, th, mean <= th, {
            "form": "photometric", "pairs": int(flow.shape[0]),
            "max": float(pf.max()), "per_pair": _per_frame(pf),
            "worst_pairs": _worst_frames(per_frame)})
    fe = _arr(obs, "flow_expected")
    fo = _arr(obs, "flow_observed")
    if fe is None or fo is None:
        return _skip("flow_warp_error", th,
                     "needs frames+flow or flow_expected+flow_observed")
    if fe.shape != fo.shape or fe.shape[-1] != 2:
        return _skip("flow_warp_error", th, f"flow shapes differ/invalid {fe.shape} vs {fo.shape}")
    h, w = fe.shape[-3], fe.shape[-2]
    diag = math.hypot(h, w)
    epe = np.linalg.norm(fo - fe, axis=-1) / diag
    frames_epe = epe if epe.ndim == 3 else epe[None]
    pf = frames_epe.reshape(frames_epe.shape[0], -1).mean(axis=1)
    mean = float(epe.mean())
    return _metric("flow_warp_error", mean, th, mean <= th, {
        "form": "endpoint", "pairs": int(frames_epe.shape[0]), "max": float(pf.max()),
        "mean_epe_px": float(mean * diag), "per_pair": _per_frame(pf),
        "worst_pairs": _worst_frames(list(pf))})


def camera_drift(manifest: Any, obs: Observations,
                 thresholds: DriftThresholds | None = None) -> dict[str, Any]:
    """Max camera-centre distance (metres) between the rendered camera path and
    the locked track, plus the max rotation angle. The track is authoritative;
    both sides are in the manifest coordinate system and scaled to metres."""
    th = _thresholds(manifest, thresholds).camera_drift_m_max
    e = obs.get("poses_expected")
    o = obs.get("poses_observed")
    if e is None or o is None:
        return _skip("camera_drift_m", th, "needs poses_expected and poses_observed")
    e = np.asarray(e, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)
    try:
        pe, po = pose_positions(e), pose_positions(o)
    except ValueError as exc:
        return _skip("camera_drift_m", th, str(exc))
    if pe.shape != po.shape:
        return _skip("camera_drift_m", th, f"pose counts differ {pe.shape[0]} vs {po.shape[0]}")
    scale = _scale_to_m(manifest)
    dist = np.linalg.norm(po - pe, axis=1) * scale
    rot = pose_rotations_deg(e, o)
    max_t = float(dist.max())
    max_r = float(rot.max()) if rot is not None else None
    ok = max_t <= th and (max_r is None or max_r <= CAMERA_ROTATION_DEG_MAX)
    first_over = int(np.argmax(dist > th)) if (dist > th).any() else None
    return _metric("camera_drift_m", max_t, th, ok, {
        "mean_m": float(dist.mean()), "frames": int(dist.shape[0]),
        "max_rotation_deg": max_r, "rotation_threshold_deg": CAMERA_ROTATION_DEG_MAX,
        "first_frame_over": first_over, "per_frame_m": _per_frame(dist),
        "worst_frames": _worst_frames(list(dist)),
        "final_offset_m": float(dist[-1])})


def _aabb_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Signed separation between two AABBs (2,3): positive = gap along the
    most-separated axis, negative = penetration depth (least overlap axis)."""
    lo = np.maximum(a[0], b[0])
    hi = np.minimum(a[1], b[1])
    overlap = hi - lo                      # per-axis; negative means separated
    # separated: the most negative axis is the gap; overlapping everywhere:
    # the least overlap is the penetration depth. Both read as -min(overlap).
    return float(-overlap.min())


def collision_check(manifest: Any, obs: Observations,
                    thresholds: DriftThresholds | None = None,
                    *, tolerance_m: float = CONTACT_TOLERANCE_M) -> dict[str, Any]:
    """Does the render honour the simulation's contacts? For each expected
    contact ``{"a","b","frame","kind"}``: ``touch``/``rest`` means the two
    AABBs must meet within ``tolerance_m`` (no gap, no penetration beyond it);
    ``separate`` means they must not interpenetrate beyond it. ``b`` may name
    a static collider. With no ``contacts_expected`` every entity pair is
    checked for interpenetration on every frame (nothing should pass through
    anything). Value = number of violations; threshold 0."""
    del thresholds
    bounds = obs.get("entity_bounds_observed") or {}
    statics = obs.get("static_colliders") or {}
    if not bounds:
        return _skip("collision", 0.0, "needs entity_bounds_observed")
    scale = _scale_to_m(manifest)
    boxes = {k: np.asarray(v, dtype=np.float64) * scale for k, v in bounds.items()}
    fixed = {k: np.asarray(v, dtype=np.float64) * scale for k, v in statics.items()}
    for k, v in boxes.items():
        if v.ndim != 3 or v.shape[1:] != (2, 3):
            return _skip("collision", 0.0, f"entity_bounds_observed[{k!r}] must be (F,2,3)")

    def box_at(name: str, frame: int) -> np.ndarray | None:
        if name in boxes:
            b = boxes[name]
            return b[frame] if 0 <= frame < b.shape[0] else None
        if name in fixed:
            return fixed[name]
        return None

    violations: list[dict[str, Any]] = []
    checked = 0
    contacts = obs.get("contacts_expected")
    if contacts:
        for c in contacts:
            a, b, frame = str(c["a"]), str(c["b"]), int(c.get("frame", 0))
            kind = str(c.get("kind", "touch"))
            ba, bb = box_at(a, frame), box_at(b, frame)
            if ba is None or bb is None:
                violations.append({"a": a, "b": b, "frame": frame, "kind": kind,
                                   "reason": "entity missing from observations"})
                continue
            checked += 1
            gap = _aabb_gap(ba, bb)
            if kind in ("touch", "rest") and abs(gap) > tolerance_m:
                violations.append({"a": a, "b": b, "frame": frame, "kind": kind,
                                   "gap_m": gap,
                                   "reason": "penetration" if gap < 0 else "floating"})
            elif kind == "separate" and gap < -tolerance_m:
                violations.append({"a": a, "b": b, "frame": frame, "kind": kind,
                                   "gap_m": gap, "reason": "penetration"})
    else:
        names = sorted(boxes)
        frames = min(boxes[n].shape[0] for n in names)
        for i, a in enumerate(names):
            for b in names[i + 1:] + sorted(fixed):
                for f in range(frames):
                    checked += 1
                    gap = _aabb_gap(boxes[a][f], box_at(b, f))
                    if gap < -tolerance_m:
                        violations.append({"a": a, "b": b, "frame": f, "kind": "separate",
                                           "gap_m": gap, "reason": "penetration"})
    n = len(violations)
    return _metric("collision", float(n), 0.0, n == 0, {
        "checked": checked, "tolerance_m": tolerance_m,
        "violations": violations[:20],
        "frames": sorted({v["frame"] for v in violations}),
        "entities": sorted({v["a"] for v in violations} | {v["b"] for v in violations})})


# --------------------------------------------------------------------------- #
# registry + report
# --------------------------------------------------------------------------- #


def spatial_rubrics() -> dict[str, Evaluator]:
    """name → evaluator, in gate order. ``evaluation.py`` registers these as
    the rubrics of the ``spatial:<seg>`` gate; each is called as
    ``fn(manifest, observations, thresholds=None)`` and returns a metric dict."""
    return {
        "reprojection_px": reprojection_error,
        "silhouette_iou": silhouette_iou,
        "depth_rel_error": depth_consistency,
        "normal_angle_deg": normal_consistency,
        "flow_warp_error": flow_warp_error,
        "camera_drift_m": camera_drift,
        "collision": collision_check,
    }


_KINDS = {"flow_warp_error": CheckKind.TEMPORAL}


def metric_to_check(metric: Mapping[str, Any]) -> Check:
    """Fold a metric dict into a ``contracts.Check`` row. A skipped metric is a
    non-passing row whose detail says it was not measured — the gate can
    decide whether "unmeasured" blocks, but it cannot mistake it for a pass."""
    name = str(metric["metric"])
    skipped = metric.get("skipped")
    ok = metric.get("ok")
    ev = metric.get("evidence") or {}
    if skipped:
        detail = f"not measured: {skipped}"
    else:
        cmp = ">=" if metric.get("higher_is_better") else "<="
        detail = (f"{name}={metric['value']:.4g} {cmp} {metric['threshold']:.4g}: "
                  f"{'ok' if ok else 'FAIL'}")
        if metric.get("code"):
            detail += f" → {metric['code'].value}"
        if ev.get("worst_frames"):
            detail += f"; worst frames {ev['worst_frames']}"
        if ev.get("worst_pairs"):
            detail += f"; worst pairs {ev['worst_pairs']}"
    return Check(name=f"spatial.{name}", kind=_KINDS.get(name, CheckKind.TECHNICAL),
                 value=metric.get("value"), threshold=metric.get("threshold"),
                 passed=bool(ok), detail=detail)


@dataclass(frozen=True, slots=True)
class SpatialEvalReport:
    segment_id: str
    metrics: tuple[dict[str, Any], ...]
    thresholds: DriftThresholds = field(default_factory=DriftThresholds)

    @property
    def codes(self) -> tuple[RepairCode, ...]:
        """Distinct repair codes, in gate order — first one is the route."""
        seen: list[RepairCode] = []
        for m in self.metrics:
            c = m.get("code")
            if c is not None and c not in seen:
                seen.append(c)
        return tuple(seen)

    @property
    def measured(self) -> tuple[str, ...]:
        return tuple(m["metric"] for m in self.metrics if not m.get("skipped"))

    @property
    def skipped(self) -> tuple[str, ...]:
        return tuple(m["metric"] for m in self.metrics if m.get("skipped"))

    @property
    def ok(self) -> bool:
        """True when every MEASURED metric passed and at least one was measured."""
        return bool(self.measured) and not self.codes

    @property
    def failures(self) -> tuple[dict[str, Any], ...]:
        return tuple(m for m in self.metrics if m.get("ok") is False)

    def checks(self) -> tuple[Check, ...]:
        return tuple(metric_to_check(m) for m in self.metrics)

    def to_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, "ok": self.ok,
                "codes": [c.value for c in self.codes],
                "measured": list(self.measured), "skipped": list(self.skipped),
                "thresholds": self.thresholds.to_dict(),
                "metrics": [{**m, "code": m["code"].value if m.get("code") else None}
                            for m in self.metrics]}


def evaluate_spatial(manifest: SpatialSceneManifest | Mapping[str, Any], observations: Observations,
                     *, thresholds: DriftThresholds | None = None,
                     rubrics: Mapping[str, Evaluator] | None = None) -> SpatialEvalReport:
    """Run every spatial rubric over one segment's observations."""
    th = _thresholds(manifest, thresholds)
    seg = (manifest.segment_id if isinstance(manifest, SpatialSceneManifest)
           else str((manifest or {}).get("segment_id", "?")))
    out = tuple(fn(manifest, observations, th) for fn in (rubrics or spatial_rubrics()).values())
    return SpatialEvalReport(segment_id=seg, metrics=out, thresholds=th)


# --------------------------------------------------------------------------- #
# synthetic scenes — Track D fixtures and the unit tests share ONE generator
# --------------------------------------------------------------------------- #


def synthetic_scene(*, frames: int = 8, size: tuple[int, int] = (48, 64),
                    seed: int = 0) -> dict[str, Any]:
    """A small analytic scene: a cube of landmarks in front of a dollying
    camera, its silhouette, depth, normals, flow and a resting contact. All
    ``*_observed`` arrays equal ``*_expected`` — a perfect render. Perturb
    with :func:`perturb` to manufacture each fault."""
    h, w = size
    rng = np.random.default_rng(seed)
    intr = CameraIntrinsics(fx=60.0, fy=60.0, cx=w / 2, cy=h / 2, width=w, height=h)
    # unit cube corners 4 m in front of the camera, plus a few random interior points
    corners = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    interior = rng.uniform(-0.4, 0.4, size=(8, 3))
    pts = np.concatenate([corners, interior]) + np.array([0.0, 0.0, -4.0])
    poses = []
    for f in range(frames):
        p = np.eye(4)
        p[0, 3] = -0.02 * f              # camera tracks right 2 cm per frame
        poses.append(p)
    poses = np.stack(poses)
    landmarks = np.stack([project_points(intr, transform_points(p, pts)) for p in poses])
    ys, xs = np.mgrid[0:h, 0:w]
    sil = np.zeros((frames, h, w), dtype=bool)
    depth = np.full((frames, h, w), 6.0)
    for f in range(frames):
        lm = landmarks[f]
        u0, u1 = lm[:, 0].min(), lm[:, 0].max()
        v0, v1 = lm[:, 1].min(), lm[:, 1].max()
        box = (xs >= u0) & (xs <= u1) & (ys >= v0) & (ys <= v1)
        sil[f] = box
        depth[f][box] = 3.5
    normals = np.zeros((frames, h, w, 3))
    normals[..., 2] = 1.0
    normals[sil] = np.array([0.0, 0.0, 1.0])
    normals[~sil] = np.array([0.0, 1.0, 0.0])
    texture = 0.05 * np.sin(xs / 3.0)[None].repeat(frames, 0)
    frames_img = (np.where(sil, 0.8, 0.2) + texture)[..., None]
    flow = np.zeros((frames - 1, h, w, 2))
    shift = landmarks[1:, 0, 0] - landmarks[:-1, 0, 0]            # horizontal image shift
    flow[..., 0] = shift[:, None, None]
    bounds = np.zeros((frames, 2, 3))
    bounds[:, 0] = [-0.5, 0.0, -4.5]
    bounds[:, 1] = [0.5, 1.0, -3.5]
    return {
        "intrinsics": intr,
        "points_world": pts,
        "poses_expected": poses, "poses_observed": poses.copy(),
        "landmarks_expected": landmarks, "landmarks_observed": landmarks.copy(),
        "silhouette_expected": sil, "silhouette_observed": sil.copy(),
        "depth_expected": depth, "depth_observed": depth.copy(),
        "normals_expected": normals, "normals_observed": normals.copy(),
        "frames": frames_img, "flow": flow,
        "flow_expected": flow, "flow_observed": flow.copy(),
        "entity_bounds_observed": {"hero": bounds},
        "static_colliders": {"ground": np.array([[-10.0, -1.0, -10.0], [10.0, 0.0, 10.0]])},
        "contacts_expected": [{"a": "hero", "b": "ground", "frame": f, "kind": "rest"}
                              for f in range(frames)],
    }


#: The perturbations Track D applies — name → what it does to the scene.
PERTURBATIONS: dict[str, str] = {
    "none": "perfect render",
    "landmark_shift": "every landmark displaced by `magnitude` px",
    "silhouette_erode": "observed silhouette shifted `magnitude` px right",
    "depth_scale": "observed depth scaled by 1+`magnitude`",
    "normal_tilt": "observed normals tilted by `magnitude` degrees",
    "flow_shift": "observed flow wrong by `magnitude` px",
    "camera_offset": "observed camera path offset by `magnitude` m",
    "camera_yaw": "observed camera yawed by `magnitude` degrees",
    "sink_into_ground": "hero sinks `magnitude` m into the ground",
    "float_off_ground": "hero floats `magnitude` m above the ground",
}


def perturb(scene: Mapping[str, Any], kind: str, magnitude: float) -> dict[str, Any]:
    """A copy of ``scene`` with ONE fault injected into the observed side."""
    s = dict(scene)
    if kind == "none":
        return s
    if kind == "landmark_shift":
        s["landmarks_observed"] = scene["landmarks_expected"] + np.array([magnitude, 0.0])
    elif kind == "silhouette_erode":
        k = int(round(magnitude))
        sil = np.asarray(scene["silhouette_expected"])
        s["silhouette_observed"] = np.roll(sil, k, axis=-1)
    elif kind == "depth_scale":
        s["depth_observed"] = np.asarray(scene["depth_expected"]) * (1.0 + magnitude)
    elif kind == "normal_tilt":
        n = np.asarray(scene["normals_expected"]).copy()
        a = math.radians(magnitude)          # about X: tilts every normal in the scene
        rot = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        s["normals_observed"] = n @ rot.T
    elif kind == "flow_shift":
        fl = np.asarray(scene["flow"]).copy()
        fl[..., 0] += magnitude
        s["flow"] = fl
        s["flow_observed"] = fl
    elif kind == "camera_offset":
        p = np.asarray(scene["poses_expected"]).copy()
        p[:, 1, 3] -= magnitude            # world→camera: t = -R C, so C_y += magnitude
        s["poses_observed"] = p
    elif kind == "camera_yaw":
        p = np.asarray(scene["poses_expected"]).copy()
        a = math.radians(magnitude)
        rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        p[:, :3, :3] = rot @ p[:, :3, :3]
        s["poses_observed"] = p
    elif kind in ("sink_into_ground", "float_off_ground"):
        b = np.asarray(scene["entity_bounds_observed"]["hero"]).copy()
        b[..., 1] += -magnitude if kind == "sink_into_ground" else magnitude
        s["entity_bounds_observed"] = {**scene["entity_bounds_observed"], "hero": b}
    else:
        raise KeyError(f"unknown perturbation {kind!r}; known: {sorted(PERTURBATIONS)}")
    return s


def synthetic_manifest_stub(intrinsics: CameraIntrinsics | None = None,
                            thresholds: DriftThresholds | None = None,
                            world_units: str = "meters") -> dict[str, Any]:
    """The minimum manifest MAPPING the evaluators read (segment id, units,
    intrinsics, thresholds) — tests that need a full validated
    ``SpatialSceneManifest`` build one through ``spatial``; the evaluators
    accept either."""
    intr = intrinsics or CameraIntrinsics(60.0, 60.0, 32.0, 24.0, 64, 48)
    return {"segment_id": "synthetic", "coordinate_system": CoordinateSystem(world_units=world_units).to_dict(),
            "camera": {"track_uri": "mem://track", "intrinsics": intr.to_dict()},
            "thresholds": (thresholds or DriftThresholds()).to_dict()}


def run_track_d_case(case: Any, *, frames: int = 8) -> dict[str, Any]:
    """Execute one Track D benchmark case: parse its ``input_text`` JSON
    ``{"perturbation": ..., "magnitude": ..., "expect_code": ...}``, build the
    synthetic scene, evaluate, and report whether the emitted code matches."""
    import json
    spec = json.loads(case.input_text)
    scene = perturb(synthetic_scene(frames=frames), spec["perturbation"], float(spec.get("magnitude", 0.0)))
    manifest = synthetic_manifest_stub(scene["intrinsics"])
    report = evaluate_spatial(manifest, scene)
    expected = spec.get("expect_code")
    got = report.codes[0].value if report.codes else None
    return {"case_id": case.case_id, "expected_code": expected, "emitted_code": got,
            "passed": got == expected, "codes": [c.value for c in report.codes],
            "report": report.to_dict()}


__all__ = [
    "CAMERA_ROTATION_DEG_MAX", "CONTACT_TOLERANCE_M", "METRIC_CODES", "PERTURBATIONS",
    "SpatialEvalReport", "camera_drift", "collision_check", "depth_consistency",
    "evaluate_spatial", "flow_warp_error", "metric_to_check", "normal_consistency",
    "perturb", "pose_positions", "pose_rotations_deg", "project_points",
    "reprojection_error", "run_track_d_case", "silhouette_iou", "spatial_rubrics",
    "synthetic_manifest_stub", "synthetic_scene", "transform_points",
]
