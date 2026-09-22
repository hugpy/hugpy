"""Live ``SpatialSceneManifest`` producers — Fold 1 (or-k8).

Until now every ``SpatialSceneManifest`` in the oracle was hand-built in a
test. This module derives one from a real scene source, stdlib only:

* :func:`from_gltf` — glTF 2.0, JSON (``.gltf``) or GLB container (``.glb``).
  Node TRS / matrix transforms become entity placements, ``cameras`` become
  ``CameraSpec`` + per-frame keyframes (animation channels are sampled on the
  production frame grid), buffers are read from data URIs, the GLB BIN chunk
  or a sibling file. glTF is metres / +Y up / right-handed by spec.
* :func:`from_usd` — the USDA *text* subset the oracle actually needs:
  ``def`` prims with ``xformOp:translate`` / ``rotateXYZ`` / ``scale`` (static
  or ``.timeSamples``), ``Camera`` prims (``focalLength``, apertures,
  ``clippingRange``) and layer metadata (``metersPerUnit``, ``upAxis``,
  ``startTimeCode`` / ``endTimeCode`` / ``timeCodesPerSecond``). USDC binary
  crate files are refused with an explicit :class:`NotImplementedError`; a
  fake parse would be a fabricated scene.
* :func:`from_pose_track` — generic mocap / physics JSON: a list of per-frame
  dicts carrying entity positions and an optional camera.

Every producer returns a :class:`SpatialSource`: the manifest *plus* the
sampled camera track and entity trajectories (the manifest itself only carries
URIs, by design), and the provenance of the source file (tool, path, sha256).
The manifest is run through :func:`spatial.validate_manifest` before it is
returned; faults raise :class:`SpatialSourceError` naming every fault, so an
invalid scene never reaches admission.

URIs minted here are *fragment URIs into the source file*
(``scene.gltf#camera/3``, ``scene.usda#/World/Hero``): the asset IS the source
file, and :func:`source_asset_exists` / :func:`source_checksum_of` are the
matching ``validate_manifest`` seams.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from hugpy_oracle import spatial as sp

__all__ = [
    "CameraKeyframe",
    "EntityTrack",
    "SpatialSource",
    "SpatialSourceError",
    "from_gltf",
    "from_pose_track",
    "from_usd",
    "load_source",
    "source_asset_exists",
    "source_checksum_of",
    "sha256_of_file",
]

SOURCE_SCHEMA_VERSION = "1.0"
DEFAULT_FPS = 24.0
DEFAULT_RENDER = sp.RenderSpec(width=1280, height=720, seed=0)
_DEFAULT_DENSE = (sp.ConditioningPass.DEPTH, sp.ConditioningPass.SILHOUETTE, sp.ConditioningPass.OCCLUSION)
_DEFAULT_SPARSE = (sp.ConditioningPass.BBOX, sp.ConditioningPass.KEYPOINTS)
_CLOTH_HAIR_TAGS = frozenset({"cloth", "hair", "cape", "cloak", "dress", "skirt", "scarf", "veil", "flag",
                              "curtain", "fur"})


class SpatialSourceError(ValueError):
    """The source could not become an admissible manifest. ``faults`` carries
    the validator's findings when validation (not parsing) was the problem."""

    def __init__(self, message: str, faults: Sequence[sp.SpatialFault] = ()) -> None:
        super().__init__(message)
        self.faults = tuple(faults)


# --------------------------------------------------------------------------- #
# small math (column-major 4x4 as 16 floats, glTF convention)
# --------------------------------------------------------------------------- #

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]     # xyzw


def _identity() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _mat_mul(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """column-major: out = a @ b."""
    out = [0.0] * 16
    for col in range(4):
        for row in range(4):
            out[col * 4 + row] = sum(a[k * 4 + row] * b[col * 4 + k] for k in range(4))
    return out


def _trs_matrix(t: Vec3, r: Quat, s: Vec3) -> list[float]:
    x, y, z, w = r
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z
    m = _identity()
    m[0] = (1 - 2 * (yy + zz)) * s[0]
    m[1] = (2 * (xy + wz)) * s[0]
    m[2] = (2 * (xz - wy)) * s[0]
    m[4] = (2 * (xy - wz)) * s[1]
    m[5] = (1 - 2 * (xx + zz)) * s[1]
    m[6] = (2 * (yz + wx)) * s[1]
    m[8] = (2 * (xz + wy)) * s[2]
    m[9] = (2 * (yz - wx)) * s[2]
    m[10] = (1 - 2 * (xx + yy)) * s[2]
    m[12], m[13], m[14] = t
    return m


def _mat_translation(m: Sequence[float]) -> Vec3:
    return (float(m[12]), float(m[13]), float(m[14]))


def _mat_forward(m: Sequence[float]) -> Vec3:
    """World-space direction of the local −Z axis (camera look direction)."""
    fx, fy, fz = -m[8], -m[9], -m[10]
    n = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    return (fx / n, fy / n, fz / n)


def _euler_xyz_to_quat(rx: float, ry: float, rz: float) -> Quat:
    """Degrees, applied X then Y then Z (USD ``rotateXYZ``)."""
    ax, ay, az = math.radians(rx) / 2, math.radians(ry) / 2, math.radians(rz) / 2
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    return (sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz)


def _lerp(a: Sequence[float], b: Sequence[float], u: float) -> tuple[float, ...]:
    return tuple(x + (y - x) * u for x, y in zip(a, b))


def _slerp(a: Sequence[float], b: Sequence[float], u: float) -> Quat:
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b = tuple(-y for y in b)
        dot = -dot
    if dot > 0.9995:
        q = _lerp(a, b, u)
        n = math.sqrt(sum(x * x for x in q)) or 1.0
        return tuple(x / n for x in q)  # type: ignore[return-value]
    th = math.acos(max(-1.0, min(1.0, dot)))
    s = math.sin(th)
    wa, wb = math.sin((1 - u) * th) / s, math.sin(u * th) / s
    return tuple(wa * x + wb * y for x, y in zip(a, b))  # type: ignore[return-value]


def _sample(times: Sequence[float], values: Sequence[Sequence[float]], t: float, *,
            interpolation: str = "LINEAR", rotation: bool = False) -> tuple[float, ...]:
    if not times:
        raise ValueError("empty keyframe track")
    if t <= times[0]:
        return tuple(values[0])
    if t >= times[-1]:
        return tuple(values[-1])
    lo, hi = 0, len(times) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if times[mid] <= t:
            lo = mid
        else:
            hi = mid
    if interpolation == "STEP":
        return tuple(values[lo])
    u = (t - times[lo]) / ((times[hi] - times[lo]) or 1.0)
    return _slerp(values[lo], values[hi], u) if rotation else _lerp(values[lo], values[hi], u)


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# --------------------------------------------------------------------------- #
# result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CameraKeyframe:
    frame: int
    t: float
    position: Vec3
    forward: Vec3                 # unit look direction in the source's world frame
    yfov_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"frame": self.frame, "t": self.t, "position": list(self.position),
                "forward": list(self.forward), "yfov_deg": self.yfov_deg}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CameraKeyframe":
        return cls(int(d["frame"]), float(d["t"]), tuple(float(x) for x in d["position"]),  # type: ignore[arg-type]
                   tuple(float(x) for x in d.get("forward") or (0.0, 0.0, -1.0)),  # type: ignore[arg-type]
                   d.get("yfov_deg"))


@dataclass(frozen=True, slots=True)
class EntityTrack:
    entity_id: str
    entity_type: str
    positions: tuple[Vec3, ...]           # one per frame of the timebase (world, source units -> metres)
    tags: tuple[str, ...] = ()
    extent_m: float | None = None         # rough bounding radius when the source declares one

    @property
    def path_length_m(self) -> float:
        return sum(_dist(a, b) for a, b in zip(self.positions, self.positions[1:]))

    @property
    def displacement_m(self) -> float:
        return _dist(self.positions[0], self.positions[-1]) if len(self.positions) > 1 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"entity_id": self.entity_id, "entity_type": self.entity_type,
                "positions": [list(p) for p in self.positions], "tags": list(self.tags),
                "extent_m": self.extent_m}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EntityTrack":
        return cls(d["entity_id"], d.get("entity_type", "prop"),
                   tuple(tuple(float(x) for x in p) for p in d.get("positions") or ()),  # type: ignore[arg-type]
                   tuple(d.get("tags") or ()), d.get("extent_m"))


@dataclass(frozen=True, slots=True)
class SpatialSource:
    """A validated manifest together with what the manifest only points at."""
    manifest: sp.SpatialSceneManifest
    camera_track: tuple[CameraKeyframe, ...]
    entity_tracks: tuple[EntityTrack, ...]
    source_tool: str                      # gltf | glb | usda | pose_track
    source_path: str | None
    source_sha256: str
    validation: sp.SpatialValidation
    notes: tuple[str, ...] = ()
    schema_version: str = SOURCE_SCHEMA_VERSION

    @property
    def ref(self) -> str:
        """The string a ``SegmentSpec.spatial_ref`` carries: tool, path and the
        sha256 of the SOURCE (for a path-less source, of its serialised form)."""
        return f"{self.source_tool}:{self.source_path or '-'}#{self.source_sha256}"

    def track_for(self, entity_id: str) -> EntityTrack | None:
        return next((t for t in self.entity_tracks if t.entity_id == entity_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "manifest": self.manifest.to_dict(),
                "camera_track": [k.to_dict() for k in self.camera_track],
                "entity_tracks": [t.to_dict() for t in self.entity_tracks],
                "source": {"tool": self.source_tool, "path": self.source_path, "sha256": self.source_sha256},
                "validation": self.validation.to_dict(), "notes": list(self.notes), "ref": self.ref}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SpatialSource":
        src = d.get("source") or {}
        man = sp.SpatialSceneManifest.from_dict(d["manifest"])
        val = sp.validate_manifest(man)
        return cls(man, tuple(CameraKeyframe.from_dict(k) for k in d.get("camera_track") or ()),
                   tuple(EntityTrack.from_dict(t) for t in d.get("entity_tracks") or ()),
                   src.get("tool", "unknown"), src.get("path"), src.get("sha256", ""), val,
                   tuple(d.get("notes") or ()), d.get("schema_version", SOURCE_SCHEMA_VERSION))


# --------------------------------------------------------------------------- #
# provenance seams
# --------------------------------------------------------------------------- #


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _strip_fragment(uri: str) -> str:
    return uri.split("#", 1)[0]


def source_asset_exists(uri: str) -> bool:
    """``validate_manifest(asset_exists=...)`` seam for fragment URIs: the
    asset exists when the file before the ``#`` does. ``data:`` and
    ``inline:`` URIs always exist."""
    base = _strip_fragment(uri)
    if base.startswith(("data:", "inline:")):
        return True
    return os.path.isfile(base)


def source_checksum_of(uri: str) -> str | None:
    base = _strip_fragment(uri)
    if base.startswith(("data:", "inline:")) or not os.path.isfile(base):
        return None
    return sha256_of_file(base)


def _timebase(duration_s: float, fps: float, *, start_frame: int = 0) -> sp.Timebase:
    if fps <= 0:
        raise SpatialSourceError(f"fps must be > 0, got {fps}")
    frames = max(1, int(round(max(0.0, duration_s) * fps)) + 1)
    end = start_frame + frames - 1
    # duration is the span between first and last frame (what the source says);
    # validate_manifest tolerates the one-frame fencepost against frame_count/fps
    return sp.Timebase(fps=float(fps), start_frame=start_frame, end_frame=end,
                       duration_seconds=(frames - 1) / float(fps))


def _provenance(tool: str, path: str | None, sha: str, conversions: Iterable[str],
                snapshot_id: str | None) -> sp.ProvenanceSpec:
    history = (f"source_tool={tool}", f"source_path={path or '-'}", f"source_sha256={sha}", *conversions)
    return sp.ProvenanceSpec(generation_snapshot_id=snapshot_id or f"source:{sha}",
                             screenplay_revision=0, shot_plan_revision=0, continuity_revision=0,
                             registry_version=None, conversion_history=tuple(history))


def _intrinsics(yfov_rad: float | None, render: sp.RenderSpec) -> sp.CameraIntrinsics | None:
    if yfov_rad is None or yfov_rad <= 0:
        return None
    f = (render.height / 2.0) / math.tan(yfov_rad / 2.0)
    return sp.CameraIntrinsics(fx=f, fy=f, cx=render.width / 2.0, cy=render.height / 2.0,
                               width=render.width, height=render.height)


def _finish(*, run_id: str, segment_id: str, tool: str, path: str | None, sha: str,
            coord: sp.CoordinateSystem, timebase: sp.Timebase, camera: sp.CameraSpec,
            entities: Sequence[sp.EntitySpec], camera_track: Sequence[CameraKeyframe],
            entity_tracks: Sequence[EntityTrack], render: sp.RenderSpec, conversions: Sequence[str],
            snapshot_id: str | None, notes: Sequence[str], passes: Sequence[sp.ConditioningPass] | None,
            output_uri: str | None, known_entity_ids: Iterable[str] | None,
            expected_fps: float | None, tone: float,
            capture: sp.CaptureTier = sp.CaptureTier.STATIC_RIG) -> SpatialSource:
    dense = any(e.mesh_uri for e in entities)
    if passes is None:
        passes = _DEFAULT_DENSE if dense else _DEFAULT_SPARSE
    passes = tuple(passes)
    inference = (sp.InferenceTier.DENSE_CONDITIONING if set(passes) & sp.TIER2_PASSES
                 else sp.InferenceTier.TOKEN_ROUTING)
    manifest = sp.SpatialSceneManifest(
        run_id=run_id, segment_id=segment_id, artifact_revision=0,
        tier_profile=sp.TierProfile(capture=capture, inference=inference),
        timebase=timebase, coordinate_system=coord, camera=camera, entities=tuple(entities),
        conditioning=sp.ConditioningSpec(requested_passes=passes,
                                         output_uri=output_uri or f"conditioning://{run_id}/{segment_id}",
                                         width=render.width, height=render.height),
        style=sp.StyleSpec(tone=tone), render=render,
        provenance=_provenance(tool, path, sha, conversions, snapshot_id))
    validation = sp.validate_manifest(manifest, known_entity_ids=known_entity_ids,
                                      asset_exists=source_asset_exists, checksum_of=source_checksum_of,
                                      expected_fps=expected_fps)
    if not validation.ok:
        lines = "; ".join(f"{f.code.value}@{f.where}: {f.message}" for f in validation.faults)
        raise SpatialSourceError(f"{tool} source {path or '-'} does not produce an admissible manifest: {lines}",
                                 validation.faults)
    return SpatialSource(manifest, tuple(camera_track), tuple(entity_tracks), tool, path, sha, validation,
                         tuple(notes))


# --------------------------------------------------------------------------- #
# glTF 2.0
# --------------------------------------------------------------------------- #

_GLB_MAGIC = 0x46546C67
_CT_FMT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
_TYPE_N = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _read_gltf_container(path: str) -> tuple[dict[str, Any], bytes | None, bytes]:
    with open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) >= 12 and struct.unpack_from("<I", raw, 0)[0] == _GLB_MAGIC:
        version, length = struct.unpack_from("<II", raw, 4)
        if version != 2:
            raise SpatialSourceError(f"GLB version {version} unsupported (need 2): {path}")
        off, doc, blob = 12, None, None
        while off + 8 <= min(length, len(raw)):
            clen, ctype = struct.unpack_from("<II", raw, off)
            data = raw[off + 8: off + 8 + clen]
            if ctype == 0x4E4F534A:          # JSON
                doc = json.loads(data.decode("utf-8"))
            elif ctype == 0x004E4942:        # BIN
                blob = data
            off += 8 + clen
        if doc is None:
            raise SpatialSourceError(f"GLB without a JSON chunk: {path}")
        return doc, blob, raw
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpatialSourceError(f"not a glTF JSON / GLB file: {path} ({exc})") from exc
    return doc, None, raw


def _gltf_buffer(doc: Mapping[str, Any], index: int, blob: bytes | None, base_dir: str,
                 cache: dict[int, bytes]) -> bytes:
    if index in cache:
        return cache[index]
    buf = doc["buffers"][index]
    uri = buf.get("uri")
    if uri is None:
        if blob is None:
            raise SpatialSourceError(f"buffer {index} has no uri and no GLB BIN chunk")
        data = blob
    elif uri.startswith("data:"):
        data = base64.b64decode(uri.split(",", 1)[1])
    else:
        with open(os.path.join(base_dir, uri), "rb") as fh:
            data = fh.read()
    cache[index] = data
    return data


def _gltf_accessor(doc: Mapping[str, Any], index: int, blob: bytes | None, base_dir: str,
                   cache: dict[int, bytes]) -> list[tuple[float, ...]]:
    acc = doc["accessors"][index]
    if "bufferView" not in acc:
        return [tuple([0.0] * _TYPE_N[acc["type"]])] * int(acc["count"])
    view = doc["bufferViews"][acc["bufferView"]]
    data = _gltf_buffer(doc, view["buffer"], blob, base_dir, cache)
    n = _TYPE_N[acc["type"]]
    fmt = _CT_FMT[acc["componentType"]]
    size = struct.calcsize(fmt)
    stride = view.get("byteStride") or n * size
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    out = []
    for i in range(int(acc["count"])):
        off = start + i * stride
        vals = struct.unpack_from("<" + fmt * n, data, off)
        if acc.get("normalized") and fmt != "f":
            top = float(2 ** (8 * size - (0 if fmt.isupper() else 1)) - 1)
            vals = tuple(max(-1.0, v / top) for v in vals)
        out.append(tuple(float(v) for v in vals))
    return out


def _node_local(node: Mapping[str, Any], anim: Mapping[str, Any] | None, t: float) -> list[float]:
    if "matrix" in node and not anim:
        return [float(x) for x in node["matrix"]]
    tr = tuple(node.get("translation", (0.0, 0.0, 0.0)))
    rot = tuple(node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    sc = tuple(node.get("scale", (1.0, 1.0, 1.0)))
    if anim:
        if "translation" in anim:
            tr = _sample(*anim["translation"][:2], t, interpolation=anim["translation"][2])
        if "rotation" in anim:
            rot = _sample(*anim["rotation"][:2], t, interpolation=anim["rotation"][2], rotation=True)
        if "scale" in anim:
            sc = _sample(*anim["scale"][:2], t, interpolation=anim["scale"][2])
    return _trs_matrix(tr, rot, sc)  # type: ignore[arg-type]


def _gltf_entity_type(node: Mapping[str, Any], doc: Mapping[str, Any]) -> str:
    extras = node.get("extras") or {}
    if extras.get("entity_type"):
        return str(extras["entity_type"])
    if "skin" in node:
        return "character"
    ext = node.get("extensions") or {}
    if "KHR_lights_punctual" in ext:
        return "light"
    if "camera" in node:
        return "camera_rig"
    return "prop"


def _tags_of(name: str, extras: Mapping[str, Any]) -> tuple[str, ...]:
    tags = extras.get("tags") or ()
    if isinstance(tags, str):
        tags = [tags]
    out = [str(t).lower() for t in tags]
    for word in re.findall(r"[A-Za-z]+", name or ""):
        if word.lower() in _CLOTH_HAIR_TAGS:
            out.append(word.lower())
    return tuple(dict.fromkeys(out))


def from_gltf(path: str, *, run_id: str, segment_id: str, fps: float = DEFAULT_FPS,
              render: sp.RenderSpec = DEFAULT_RENDER, camera: int | str | None = None,
              animation: int | str | None = None, identity_refs: Mapping[str, Sequence[str]] | None = None,
              passes: Sequence[sp.ConditioningPass] | None = None, output_uri: str | None = None,
              known_entity_ids: Iterable[str] | None = None, snapshot_id: str | None = None,
              tone: float = 5.0, duration_s: float | None = None) -> SpatialSource:
    """Derive a validated Fold-1 manifest from a glTF 2.0 / GLB file.

    ``camera`` selects a camera node (index into ``cameras`` or node name);
    default: the first node that references a camera. Without any camera the
    source is refused — a manifest with an invented camera would be a lie.
    ``identity_refs`` maps entity id -> identity reference ids for nodes the
    file marks as characters (``skin`` or ``extras.entity_type``)."""
    doc, blob, raw = _read_gltf_container(path)
    sha = _sha256_bytes(raw)
    tool = "glb" if blob is not None or raw[:4] == b"glTF" else "gltf"
    base_dir = os.path.dirname(os.path.abspath(path))
    nodes: list[Mapping[str, Any]] = list(doc.get("nodes") or [])
    if not nodes:
        raise SpatialSourceError(f"glTF has no nodes: {path}")
    cams = list(doc.get("cameras") or [])
    notes: list[str] = []
    cache: dict[int, bytes] = {}

    # parents
    parent: dict[int, int] = {}
    for i, n in enumerate(nodes):
        for c in n.get("children") or ():
            parent[int(c)] = i

    # animation channels -> node -> path -> (times, values, interpolation)
    anims = list(doc.get("animations") or [])
    chosen: Mapping[str, Any] | None = None
    if anims:
        if isinstance(animation, int):
            chosen = anims[animation]
        elif isinstance(animation, str):
            chosen = next((a for a in anims if a.get("name") == animation), None)
            if chosen is None:
                raise SpatialSourceError(f"no animation named {animation!r} in {path}")
        else:
            chosen = anims[0]
            if len(anims) > 1:
                notes.append(f"{len(anims)} animations; sampled the first ({chosen.get('name') or 0})")
    tracks: dict[int, dict[str, tuple[list[float], list[tuple[float, ...]], str]]] = {}
    anim_end = 0.0
    if chosen:
        for ch in chosen.get("channels") or ():
            tgt = ch.get("target") or {}
            if "node" not in tgt or tgt.get("path") not in ("translation", "rotation", "scale"):
                continue
            smp = chosen["samplers"][ch["sampler"]]
            times = [v[0] for v in _gltf_accessor(doc, smp["input"], blob, base_dir, cache)]
            values = _gltf_accessor(doc, smp["output"], blob, base_dir, cache)
            interp = smp.get("interpolation", "LINEAR")
            if interp == "CUBICSPLINE":
                values = values[1::3]   # keep the vertex values, drop tangents
                interp = "LINEAR"
                notes.append("CUBICSPLINE sampled linearly between vertices")
            tracks.setdefault(int(tgt["node"]), {})[tgt["path"]] = (times, values, interp)
            if times:
                anim_end = max(anim_end, times[-1])
    if duration_s is None:
        duration_s = anim_end
    timebase = _timebase(duration_s, fps)

    # units: glTF is metres by spec; honour an explicit asset.extras override
    extras_asset = (doc.get("asset") or {}).get("extras") or {}
    units = str(extras_asset.get("units") or "meters")
    coord = sp.CoordinateSystem(handedness="right", up_axis="Y", forward_axis="Z", world_units=units)
    scale_m = coord.scale_to_m
    conversions = [f"gltf(Y-up,{units},right)->manifest"]

    def world(i: int, t: float, memo: dict[int, list[float]]) -> list[float]:
        if i in memo:
            return memo[i]
        local = _node_local(nodes[i], tracks.get(i), t)
        m = _mat_mul(world(parent[i], t, memo), local) if i in parent else local
        memo[i] = m
        return m

    # camera node
    cam_node: int | None = None
    if isinstance(camera, str):
        cam_node = next((i for i, n in enumerate(nodes) if n.get("name") == camera and "camera" in n), None)
    elif isinstance(camera, int):
        cam_node = next((i for i, n in enumerate(nodes) if n.get("camera") == camera), None)
    else:
        cam_node = next((i for i, n in enumerate(nodes) if "camera" in n), None)
    if cam_node is None:
        raise SpatialSourceError(f"glTF has no usable camera node (camera={camera!r}): {path}; "
                                 f"a manifest needs an authoritative camera track")
    cam_def = cams[int(nodes[cam_node]["camera"])] if cams else {}
    persp = cam_def.get("perspective") or {}
    yfov = float(persp["yfov"]) if "yfov" in persp else None
    near = float(persp.get("znear", (cam_def.get("orthographic") or {}).get("znear", 0.1)))
    far = float(persp.get("zfar", (cam_def.get("orthographic") or {}).get("zfar", 100.0)))

    frames = timebase.frames()
    times = timebase.timestamps()
    cam_track: list[CameraKeyframe] = []
    positions: dict[int, list[Vec3]] = {i: [] for i in range(len(nodes))}
    for f, t in zip(frames, times):
        memo: dict[int, list[float]] = {}
        for i in range(len(nodes)):
            m = world(i, t, memo)
            tx, ty, tz = _mat_translation(m)
            positions[i].append((tx * scale_m, ty * scale_m, tz * scale_m))
        cm = world(cam_node, t, memo)
        cam_track.append(CameraKeyframe(f, t, positions[cam_node][-1], _mat_forward(cm),
                                        math.degrees(yfov) if yfov else None))

    ident = {k: tuple(v) for k, v in (identity_refs or {}).items()}
    entities: list[sp.EntitySpec] = []
    etracks: list[EntityTrack] = []
    seen: set[str] = set()
    for i, n in enumerate(nodes):
        if i == cam_node:
            continue
        etype = _gltf_entity_type(n, doc)
        if etype == "camera_rig" and "mesh" not in n:
            continue       # secondary cameras are not scene entities
        name = str(n.get("name") or f"node{i}")
        eid = name if name not in seen else f"{name}#{i}"
        seen.add(eid)
        extras = n.get("extras") or {}
        refs = ident.get(eid) or ident.get(name) or tuple(extras.get("identity_refs") or ())
        mesh_uri = f"{path}#mesh/{n['mesh']}" if "mesh" in n else None
        rig_uri = f"{path}#skin/{n['skin']}" if "skin" in n else None
        anim_uri = f"{path}#animation/{anims.index(chosen)}" if chosen and i in tracks else None
        entities.append(sp.EntitySpec(entity_id=eid, entity_type=etype, mesh_uri=mesh_uri, rig_uri=rig_uri,
                                      animation_uri=anim_uri, identity_reference_ids=tuple(refs),
                                      checksum=sha, source_format=tool))
        etracks.append(EntityTrack(eid, etype, tuple(positions[i]), _tags_of(name, extras),
                                   extras.get("extent_m")))
    camera_spec = sp.CameraSpec(track_uri=f"{path}#camera/{cam_node}", intrinsics=_intrinsics(yfov, render),
                                near_meters=near * scale_m, far_meters=far * scale_m, checksum=sha)
    return _finish(run_id=run_id, segment_id=segment_id, tool=tool, path=path, sha=sha, coord=coord,
                   timebase=timebase, camera=camera_spec, entities=entities, camera_track=cam_track,
                   entity_tracks=etracks, render=render, conversions=conversions, snapshot_id=snapshot_id,
                   notes=notes, passes=passes, output_uri=output_uri, known_entity_ids=known_entity_ids,
                   expected_fps=None, tone=tone)


# --------------------------------------------------------------------------- #
# USD (USDA text subset)
# --------------------------------------------------------------------------- #

_USDC_MAGIC = b"PXR-USDC"
_DEF_RE = re.compile(r'\bdef\s+(?:(\w+)\s+)?"([^"]+)"\s*(\([^)]*\))?\s*\{', re.S)
_ATTR_RE = re.compile(r'^\s*(?:custom\s+)?(?:uniform\s+)?([\w\[\]]+)\s+([\w:.]+)\s*=\s*(.+?)\s*$', re.M)
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_TUPLE_RE = re.compile(r"\(\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*(?:,\s*(" + _NUM + r"))?\s*\)")
_TIME_SAMPLE_RE = re.compile(r"(" + _NUM + r")\s*:\s*(\([^)]*\)|" + _NUM + r")")


@dataclass
class _Prim:
    type: str
    name: str
    path: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["_Prim"] = field(default_factory=list)
    parent: "_Prim | None" = None


def _match_brace(text: str, open_idx: int) -> int:
    depth, i, in_str = 0, open_idx, False
    while i < len(text):
        ch = text[i]
        if ch == '"' and text[i - 1] != "\\":
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    raise SpatialSourceError("unbalanced braces in USDA")


def _parse_prims(text: str, parent_path: str, parent: _Prim | None) -> list[_Prim]:
    prims: list[_Prim] = []
    pos = 0
    while True:
        m = _DEF_RE.search(text, pos)
        if not m:
            break
        open_idx = m.end() - 1
        close_idx = _match_brace(text, open_idx)
        body = text[open_idx + 1: close_idx]
        prim = _Prim(m.group(1) or "Xform", m.group(2), f"{parent_path}/{m.group(2)}", parent=parent)
        # own attributes = body with child defs removed
        own = body
        child_spans: list[tuple[int, int]] = []
        p2 = 0
        while True:
            cm = _DEF_RE.search(body, p2)
            if not cm:
                break
            c_open = cm.end() - 1
            c_close = _match_brace(body, c_open)
            child_spans.append((cm.start(), c_close + 1))
            p2 = c_close + 1
        for a, b in reversed(child_spans):
            own = own[:a] + own[b:]
        # attributes, including multi-line timeSamples blocks
        for am in re.finditer(r'^\s*(?:custom\s+)?(?:uniform\s+)?([\w\[\]]+)\s+([\w:.]+)\s*=\s*', own, re.M):
            start = am.end()
            rest = own[start:]
            if rest.lstrip().startswith("{"):
                b_open = start + rest.index("{")
                b_close = _match_brace(own, b_open)
                value = own[b_open: b_close + 1]
            elif rest.lstrip().startswith("["):
                value = rest[: rest.index("]") + 1]
            else:
                value = rest.split("\n", 1)[0].strip()
            prim.attrs[am.group(2)] = value.strip()
        prim.children = _parse_prims(body, prim.path, prim)
        prims.append(prim)
        pos = close_idx + 1
    return prims


def _vec(s: str) -> tuple[float, ...]:
    m = _TUPLE_RE.search(s)
    if not m:
        raise SpatialSourceError(f"expected a tuple, got {s!r}")
    return tuple(float(g) for g in m.groups() if g is not None)


def _usd_samples(value: str, scalar: bool = False) -> tuple[list[float], list[tuple[float, ...]]]:
    times: list[float] = []
    vals: list[tuple[float, ...]] = []
    for tm in _TIME_SAMPLE_RE.finditer(value):
        times.append(float(tm.group(1)))
        v = tm.group(2)
        vals.append((float(v),) if not v.startswith("(") else _vec(v))
    return times, vals


def _usd_op(prim: _Prim, op: str, t: float, default: tuple[float, ...]) -> tuple[float, ...]:
    key = f"xformOp:{op}"
    if key + ".timeSamples" in prim.attrs:
        times, vals = _usd_samples(prim.attrs[key + ".timeSamples"])
        if times:
            return _sample(times, vals, t)
    if key in prim.attrs:
        v = prim.attrs[key]
        return _vec(v) if "(" in v else (float(v),) * len(default)
    return default


def _usd_local(prim: _Prim, t: float) -> list[float]:
    tr = _usd_op(prim, "translate", t, (0.0, 0.0, 0.0))
    sc = _usd_op(prim, "scale", t, (1.0, 1.0, 1.0))
    rot: Quat = (0.0, 0.0, 0.0, 1.0)
    if "xformOp:rotateXYZ" in prim.attrs or "xformOp:rotateXYZ.timeSamples" in prim.attrs:
        e = _usd_op(prim, "rotateXYZ", t, (0.0, 0.0, 0.0))
        rot = _euler_xyz_to_quat(*e)
    elif "xformOp:orient" in prim.attrs:       # quatf (w, x, y, z) in USD
        w, x, y, z = _vec4(prim.attrs["xformOp:orient"])
        rot = (x, y, z, w)
    for axis in "XYZ":
        key = f"xformOp:rotate{axis}"
        if key in prim.attrs or key + ".timeSamples" in prim.attrs:
            deg = _usd_op(prim, f"rotate{axis}", t, (0.0,))[0]
            rot = _euler_xyz_to_quat(*(deg if a == axis else 0.0 for a in "XYZ"))
    return _trs_matrix(tr[:3], rot, sc[:3])  # type: ignore[arg-type]


def _vec4(s: str) -> tuple[float, float, float, float]:
    nums = re.findall(_NUM, s)
    if len(nums) < 4:
        raise SpatialSourceError(f"expected 4 numbers, got {s!r}")
    return tuple(float(n) for n in nums[:4])  # type: ignore[return-value]


def _usd_string(v: str | None) -> str | None:
    if v is None:
        return None
    m = re.search(r'"([^"]*)"', v)
    return m.group(1) if m else v.strip()


def _usd_string_list(v: str | None) -> tuple[str, ...]:
    return tuple(re.findall(r'"([^"]*)"', v or ""))


def _flatten(prims: list[_Prim]) -> list[_Prim]:
    out: list[_Prim] = []
    for p in prims:
        out.append(p)
        out.extend(_flatten(p.children))
    return out


def from_usd(path: str, *, run_id: str, segment_id: str, fps: float | None = None,
             render: sp.RenderSpec = DEFAULT_RENDER, camera: str | None = None,
             identity_refs: Mapping[str, Sequence[str]] | None = None,
             passes: Sequence[sp.ConditioningPass] | None = None, output_uri: str | None = None,
             known_entity_ids: Iterable[str] | None = None, snapshot_id: str | None = None,
             tone: float = 5.0) -> SpatialSource:
    """Derive a validated Fold-1 manifest from a USDA text layer.

    Limitation (by design, not by accident): USDC binary crate files are NOT
    parsed. The crate format needs the pxr toolchain; a stdlib reimplementation
    would be a partial decoder pretending to be a scene. ``NotImplementedError``
    names the file and the remedy (export ``.usda`` / run ``usdcat``)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw.startswith(_USDC_MAGIC):
        raise NotImplementedError(
            f"{path} is a USDC binary crate file; the stdlib USD reader only handles USDA text. "
            f"Export the layer as .usda (usdcat --usdFormat usda) and point the goal at that.")
    if path.lower().endswith(".usdz") or raw[:2] == b"PK":
        raise NotImplementedError(f"{path} is a USDZ package; unpack it and point the goal at the .usda layer.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SpatialSourceError(f"{path} is not UTF-8 USDA text") from exc
    if not text.lstrip().startswith("#usda"):
        raise SpatialSourceError(f"{path} lacks the #usda header; refusing to guess the format")
    sha = _sha256_bytes(raw)

    # layer metadata (the first parenthesised block after the header)
    meta: dict[str, str] = {}
    head = text.lstrip().split("\n", 1)[1] if "\n" in text.lstrip() else ""
    mm = re.match(r"\s*\((.*?)\n\)", head, re.S)
    if mm:
        for line in mm.group(1).splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    mpu = float(meta.get("metersPerUnit", "1") or 1)
    up = _usd_string(meta.get("upAxis")) or "Y"
    tcps = float(meta.get("timeCodesPerSecond") or meta.get("framesPerSecond") or (fps or DEFAULT_FPS))
    fps_eff = float(fps or tcps)
    start_tc = float(meta.get("startTimeCode", "0") or 0)
    end_tc = float(meta.get("endTimeCode", str(start_tc)) or start_tc)
    units = {1.0: "meters", 0.01: "centimeters", 0.001: "millimeters", 0.0254: "inches", 0.3048: "feet"}.get(mpu)
    notes: list[str] = []
    if units is None:
        notes.append(f"metersPerUnit={mpu} has no named unit; positions scaled to metres explicitly")
        scale_m, units_name = mpu, "meters"
    else:
        scale_m, units_name = mpu, units
    coord = sp.CoordinateSystem(handedness="right", up_axis=up.upper(), forward_axis=("-Y" if up.upper() == "Z" else "-Z"),
                                world_units=units_name)

    prims = _parse_prims(text, "", None)
    flat = _flatten(prims)
    if not flat:
        raise SpatialSourceError(f"USDA has no prims: {path}")
    cams = [p for p in flat if p.type == "Camera"]
    if camera is not None:
        cams = [p for p in cams if p.name == camera or p.path == camera]
    if not cams:
        raise SpatialSourceError(f"USDA has no Camera prim (camera={camera!r}): {path}; "
                                 f"a manifest needs an authoritative camera track")
    cam = cams[0]
    if len(cams) > 1:
        notes.append(f"{len(cams)} cameras; used {cam.path}")

    duration_s = max(0.0, (end_tc - start_tc) / tcps)
    timebase = _timebase(duration_s, fps_eff)

    def world(p: _Prim, tc: float) -> list[float]:
        local = _usd_local(p, tc)
        return _mat_mul(world(p.parent, tc), local) if p.parent is not None else local

    frames, times = timebase.frames(), timebase.timestamps()
    focal = float(cam.attrs.get("focalLength", "50") or 50)
    v_ap = float(cam.attrs.get("verticalAperture", "15.2908") or 15.2908)
    yfov = 2 * math.atan((v_ap / 2) / focal) if focal > 0 else None
    clip = _vec(cam.attrs["clippingRange"]) if "clippingRange" in cam.attrs else (0.1, 1000.0)

    cam_track: list[CameraKeyframe] = []
    positions: dict[str, list[Vec3]] = {p.path: [] for p in flat}
    for f, t in zip(frames, times):
        tc = start_tc + t * tcps
        for p in flat:
            m = world(p, tc)
            x, y, z = _mat_translation(m)
            positions[p.path].append((x * scale_m, y * scale_m, z * scale_m))
        cam_track.append(CameraKeyframe(f, t, positions[cam.path][-1], _mat_forward(world(cam, tc)),
                                        math.degrees(yfov) if yfov else None))

    ident = {k: tuple(v) for k, v in (identity_refs or {}).items()}
    entities: list[sp.EntitySpec] = []
    etracks: list[EntityTrack] = []
    for p in flat:
        if p.type == "Camera" or p.type in ("Scope", "Material", "Shader"):
            continue
        has_geom = p.type in ("Mesh", "Sphere", "Cube", "Cylinder", "Cone", "Capsule", "Points") or \
            any(c.type == "Mesh" for c in p.children)
        if p.type == "Xform" and not has_geom and not any(k.startswith("xformOp:") for k in p.attrs) \
                and "entity_type" not in p.attrs:
            continue       # pure grouping scope
        if p.parent is not None and p.type == "Mesh" and p.parent.type == "Xform" and \
                ("entity_type" in p.parent.attrs or any(k.startswith("xformOp:") for k in p.parent.attrs)):
            continue       # mesh child of a placed Xform: the Xform is the entity
        etype = _usd_string(p.attrs.get("entity_type")) or (
            "character" if p.type == "SkelRoot" else "light" if p.type.endswith("Light") else "prop")
        eid = _usd_string(p.attrs.get("entity_id")) or p.name
        refs = ident.get(eid) or ident.get(p.path) or _usd_string_list(p.attrs.get("identity_refs"))
        animated = any(k.endswith(".timeSamples") for k in p.attrs)
        entities.append(sp.EntitySpec(
            entity_id=eid, entity_type=etype, mesh_uri=f"{path}#{p.path}" if has_geom else None,
            rig_uri=f"{path}#{p.path}/skel" if p.type == "SkelRoot" else None,
            animation_uri=f"{path}#{p.path}/timeSamples" if animated else None,
            identity_reference_ids=tuple(refs), checksum=sha, source_format="usd"))
        tags = tuple(t.lower() for t in _usd_string_list(p.attrs.get("tags")))
        etracks.append(EntityTrack(eid, etype, tuple(positions[p.path]), tags or _tags_of(p.name, {}),
                                   float(p.attrs["extent_m"]) if "extent_m" in p.attrs else None))
    camera_spec = sp.CameraSpec(track_uri=f"{path}#{cam.path}", intrinsics=_intrinsics(yfov, render),
                                near_meters=clip[0] * scale_m, far_meters=clip[1] * scale_m, checksum=sha)
    conversions = [f"usda({up}-up,{units_name},right)->manifest"]
    return _finish(run_id=run_id, segment_id=segment_id, tool="usda", path=path, sha=sha, coord=coord,
                   timebase=timebase, camera=camera_spec, entities=entities, camera_track=cam_track,
                   entity_tracks=etracks, render=render, conversions=conversions, snapshot_id=snapshot_id,
                   notes=notes, passes=passes, output_uri=output_uri, known_entity_ids=known_entity_ids,
                   expected_fps=None, tone=tone)


# --------------------------------------------------------------------------- #
# generic pose track (mocap / physics JSON)
# --------------------------------------------------------------------------- #


def from_pose_track(frames: Sequence[Mapping[str, Any]], *, run_id: str, segment_id: str,
                    fps: float = DEFAULT_FPS, render: sp.RenderSpec = DEFAULT_RENDER,
                    source_tool: str = "pose_track", source_path: str | None = None,
                    units: str = "meters", up_axis: str = "Y",
                    entity_types: Mapping[str, str] | None = None,
                    identity_refs: Mapping[str, Sequence[str]] | None = None,
                    tags: Mapping[str, Sequence[str]] | None = None,
                    passes: Sequence[sp.ConditioningPass] | None = None, output_uri: str | None = None,
                    known_entity_ids: Iterable[str] | None = None, snapshot_id: str | None = None,
                    tone: float = 5.0) -> SpatialSource:
    """Derive a validated Fold-1 manifest from per-frame pose samples.

    Each frame is a dict: ``{"frame": n | "t": seconds, "entities": {id:
    {"position": [x, y, z], "type"?: ..., "tags"?: [...]}}, "camera"?:
    {"position": [...], "forward"? | "look_at"?: [...], "yfov_deg"?: f}}``.
    Frames that omit an entity carry its last known position forward (noted).
    Without a camera in ANY frame the track is refused."""
    if not frames:
        raise SpatialSourceError("pose track is empty")
    blob = json.dumps(list(frames), sort_keys=True, default=str).encode("utf-8")
    sha = _sha256_bytes(blob)
    coord = sp.CoordinateSystem(handedness="right", up_axis=up_axis, forward_axis=("-Y" if up_axis == "Z" else "-Z"),
                                world_units=units)
    scale_m = coord.scale_to_m
    notes: list[str] = []

    # frame grid
    keyed: list[tuple[float, Mapping[str, Any]]] = []
    for i, fr in enumerate(frames):
        if "t" in fr:
            t = float(fr["t"])
        elif "frame" in fr:
            t = float(fr["frame"]) / fps
        else:
            t = i / fps
        keyed.append((t, fr))
    keyed.sort(key=lambda kv: kv[0])
    t0 = keyed[0][0]
    duration = keyed[-1][0] - t0
    timebase = _timebase(duration, fps)

    ids: list[str] = []
    for _t, fr in keyed:
        for eid in (fr.get("entities") or {}):
            if eid not in ids:
                ids.append(eid)
    if not ids:
        raise SpatialSourceError("pose track names no entities")
    if not any(fr.get("camera") for _t, fr in keyed):
        raise SpatialSourceError("pose track has no camera in any frame; a manifest needs an authoritative camera")

    # per-entity sample lists -> resample onto the grid
    e_times: dict[str, list[float]] = {e: [] for e in ids}
    e_vals: dict[str, list[tuple[float, ...]]] = {e: [] for e in ids}
    c_times: list[float] = []
    c_pos: list[tuple[float, ...]] = []
    c_fwd: list[tuple[float, ...]] = []
    c_fov: float | None = None
    for t, fr in keyed:
        for eid, rec in (fr.get("entities") or {}).items():
            pos = rec.get("position") if isinstance(rec, Mapping) else rec
            e_times[eid].append(t - t0)
            e_vals[eid].append(tuple(float(x) * scale_m for x in pos[:3]))
        cam = fr.get("camera")
        if cam:
            p = tuple(float(x) * scale_m for x in cam["position"][:3])
            if cam.get("forward"):
                fwd = tuple(float(x) for x in cam["forward"][:3])
            elif cam.get("look_at"):
                la = tuple(float(x) * scale_m for x in cam["look_at"][:3])
                fwd = tuple(b - a for a, b in zip(p, la))
            else:
                fwd = (0.0, 0.0, -1.0)
            n = math.sqrt(sum(x * x for x in fwd)) or 1.0
            c_times.append(t - t0)
            c_pos.append(p)
            c_fwd.append(tuple(x / n for x in fwd))
            if cam.get("yfov_deg") is not None:
                c_fov = float(cam["yfov_deg"])
    missing = [e for e in ids if len(e_times[e]) < len(keyed)]
    if missing:
        notes.append(f"entities sampled in fewer frames than the track (held): {missing}")

    frames_grid, times = timebase.frames(), timebase.timestamps()
    cam_track = tuple(CameraKeyframe(f, t, _sample(c_times, c_pos, t), _sample(c_times, c_fwd, t), c_fov)  # type: ignore[arg-type]
                      for f, t in zip(frames_grid, times))
    types = dict(entity_types or {})
    ident = {k: tuple(v) for k, v in (identity_refs or {}).items()}
    tagmap = {k: tuple(v) for k, v in (tags or {}).items()}
    entities: list[sp.EntitySpec] = []
    etracks: list[EntityTrack] = []
    for eid in ids:
        first = next(fr["entities"][eid] for _t, fr in keyed if eid in (fr.get("entities") or {}))
        rec = first if isinstance(first, Mapping) else {}
        etype = types.get(eid) or str(rec.get("type") or "prop")
        etags = tagmap.get(eid) or tuple(str(t).lower() for t in (rec.get("tags") or ()))
        refs = ident.get(eid) or tuple(rec.get("identity_refs") or ())
        entities.append(sp.EntitySpec(entity_id=eid, entity_type=etype,
                                      animation_uri=f"inline:{source_tool}/{sha}#{eid}",
                                      identity_reference_ids=refs, checksum=sha, source_format=source_tool))
        etracks.append(EntityTrack(eid, etype, tuple(_sample(e_times[eid], e_vals[eid], t) for t in times),  # type: ignore[arg-type]
                                   etags, rec.get("extent_m")))
    yfov = math.radians(c_fov) if c_fov else None
    camera_spec = sp.CameraSpec(track_uri=f"inline:{source_tool}/{sha}#camera", intrinsics=_intrinsics(yfov, render),
                                checksum=sha)
    return _finish(run_id=run_id, segment_id=segment_id, tool=source_tool, path=source_path, sha=sha,
                   coord=coord, timebase=timebase, camera=camera_spec, entities=entities, camera_track=cam_track,
                   entity_tracks=etracks, render=render, conversions=[f"{source_tool}({up_axis}-up,{units})->manifest"],
                   snapshot_id=snapshot_id, notes=notes, passes=passes, output_uri=output_uri,
                   known_entity_ids=known_entity_ids, expected_fps=None, tone=tone,
                   capture=sp.CaptureTier.REALTIME_MOCAP)


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #


def load_source(path: str, *, run_id: str, segment_id: str, **kw: Any) -> SpatialSource:
    """Pick the producer by extension / magic. ``.json`` is read as a pose
    track (a list of frames, or ``{"frames": [...]}``)."""
    low = path.lower()
    if low.endswith((".gltf", ".glb")):
        return from_gltf(path, run_id=run_id, segment_id=segment_id, **kw)
    if low.endswith((".usd", ".usda", ".usdc", ".usdz")):
        return from_usd(path, run_id=run_id, segment_id=segment_id, **kw)
    if low.endswith(".json"):
        with open(path, "rb") as fh:
            raw = fh.read()
        doc = json.loads(raw.decode("utf-8"))
        if isinstance(doc, Mapping) and "asset" in doc and "nodes" in doc:
            return from_gltf(path, run_id=run_id, segment_id=segment_id, **kw)
        frames = doc.get("frames") if isinstance(doc, Mapping) else doc
        meta = {k: doc[k] for k in ("fps", "units", "up_axis", "source_tool") if isinstance(doc, Mapping) and k in doc}
        return from_pose_track(frames, run_id=run_id, segment_id=segment_id, source_path=path, **{**meta, **kw})
    with open(path, "rb") as fh:
        head = fh.read(8)
    if head[:4] == b"glTF":
        return from_gltf(path, run_id=run_id, segment_id=segment_id, **kw)
    if head.startswith(_USDC_MAGIC) or head.startswith(b"#usda"):
        return from_usd(path, run_id=run_id, segment_id=segment_id, **kw)
    raise SpatialSourceError(f"unrecognised spatial source: {path}")
