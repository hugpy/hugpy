"""Identity FROM-VIDEO job schema (k94) — the durable, JSON-safe intent for "a video of
a character in -> one bindable identity profile (reference views + 3D GLB) per detected
character out", in ONE chained job.

This is the clownworld MO (``characters3d.ts``): a single ``video_characters_glb`` job to
the ``IDENTITY_RENDER_URL`` service (char360 extraction chained into one Hunyuan3D GLB per
detected character), relayed by ``runners/identity_from_video.py``. Central has NO GPU and
never runs char360 / Hunyuan — the runner is a thin HTTP client over the shared
``runners/identity_render_client`` helpers.

House style mirrors ``identity_video_extract_schema``: a frozen, JSON-safe, validate-at-
construction spec built ONLY via ``make_identity_from_video``; the bus rehydrates it
through ``identity_from_video_from_dict`` (reconstruct + RE-VALIDATE). A raise inside the
factory is local to construction (caller error at the boundary); the RUNNER never raises
for an expected failure. No pathlib anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from hugpy_video.intel.media_schema import MediaRef

# The happy-path mesh params — IDENTICAL to clownworld's defaults (textured; octree 256
# because the texture bake's UV-unwrap time scales with face count — 380 was ~24 min per
# character — and the texture carries the visual detail). Accepted optionally on the
# route, never required.
DEFAULT_MESH_PARAMS: Dict[str, Any] = {"texture": True, "octree_resolution": 256}

# The ``mesh_params`` keys the service's MeshParams accepts for this kind. Unknown keys
# are dropped (never ride the bus as a silent no-op); values are not range-checked here
# (the service's pydantic model is the authority) beyond basic typing.
MESH_PARAM_KEYS = ("texture", "octree_resolution", "seed", "num_inference_steps")


@dataclass(frozen=True)
class IdentityFromVideoSpec:
    """Frozen, JSON-safe currency of an ``identity_from_video`` bus job.

        source        the source VIDEO clip (a ``MediaRef`` of kind ``"video"``). Its uri is
                      an absolute path forwarded to the service as ``video_path`` (ae +
                      central share the mount; no base64 inflation).
        name          the requested identity NAME. The first detected character becomes
                      profile ``slugify(name)``; extra characters become ``<slug>-2``,
                      ``<slug>-3`` … (the runner owns that rule).
        mesh_params   the service ``mesh_params`` block; defaults to DEFAULT_MESH_PARAMS.
        identity_id   the correlation id handed to the service (its JobCreateRequest
                      REQUIRES one) — the route passes the base slug.
    """
    source: MediaRef
    name: str
    mesh_params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_MESH_PARAMS))
    identity_id: str = ""


def _clean_mesh_params(raw) -> Dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"mesh_params must be a dict or None; got {type(raw).__name__}")
    out: Dict[str, Any] = dict(DEFAULT_MESH_PARAMS)
    for k in MESH_PARAM_KEYS:
        if k not in raw:
            continue
        v = raw[k]
        if k == "texture":
            out[k] = bool(v)
        else:
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"mesh_params.{k} must be a positive integer; got {v!r}")
            out[k] = v
    return out


def make_identity_from_video(
    *,
    source: MediaRef,
    name: str,
    mesh_params=None,
    identity_id=None,
) -> IdentityFromVideoSpec:
    """Validate every field and build the frozen ``IdentityFromVideoSpec``. Raises
    ``ValueError`` LOCALLY on a structural violation (never across the bus)."""
    if not isinstance(source, MediaRef):
        raise ValueError(f"source must be a MediaRef; got {type(source).__name__}")
    if source.kind != "video":
        raise ValueError(
            f"identity_from_video source must be a video; got kind={source.kind!r}")
    if not (isinstance(name, str) and name.strip()):
        raise ValueError(f"name must be a non-empty string; got {name!r}")
    name = name.strip()
    if identity_id is not None and not isinstance(identity_id, str):
        raise ValueError(
            f"identity_id must be a string or None; got {type(identity_id).__name__}")
    return IdentityFromVideoSpec(
        source=source,
        name=name,
        mesh_params=_clean_mesh_params(mesh_params),
        identity_id=(identity_id or "").strip(),
    )


def identity_from_video_from_dict(d: dict) -> IdentityFromVideoSpec:
    """Rebuild an ``IdentityFromVideoSpec`` from its ``asdict`` form THROUGH the
    validating factory (deserialize-then-revalidate). Registered in
    ``media_bus.SPEC_DESERIALIZERS`` under ``"identity_from_video"``."""
    from hugpy_video.intel.media_schema import make_media_ref

    raw_source = d.get("source")
    if not isinstance(raw_source, dict):
        raise ValueError(
            f"identity_from_video spec is missing a 'source' MediaRef; got {raw_source!r}")
    return make_identity_from_video(
        source=make_media_ref(**raw_source),
        name=d["name"],
        mesh_params=d.get("mesh_params") or {},
        identity_id=d.get("identity_id"),
    )
