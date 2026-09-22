"""Identity VIDEO-EXTRACT (char360) RELAY job schema — the durable, JSON-safe intent
for "extract per-character 360° view-sets from a source video on the remote GPU render
service, then write them back into identity profiles".

This is char360's CENTRAL-side currency (feature: CHAR360-FEATURE-PLAN.md, slice S3). It
is the INVERSE of ``identity_reconstruction`` (which *generates* a turnaround from
reference photos): a video clip is relayed to the standalone ``IDENTITY_RENDER_URL``
service (which grew the ``video_extract`` job kind in S2), which runs scene-detect ->
YOLO track -> insightface embed -> agglomerative cluster -> yaw-bin, and returns a
per-character view-set + a face-descriptor manifest. Central has NO GPU and NEVER imports
char360/cv2/insightface — the runner (``runners/identity_video_extract_relay.py``) is a
thin HTTP client, mirroring ``runners/identity_render_relay.py``.

House style mirrors ``frame_schema`` / ``identity_reconstruction_schema`` /
``IdentityMeshSpec``: a frozen, JSON-safe, validate-at-construction spec built ONLY via
``make_identity_video_extract``; the bus rehydrates it through
``identity_video_extract_from_dict`` (reconstruct + RE-VALIDATE). Every field is a
primitive / string tuple / plain dict so ``asdict`` -> ``json`` round-trips cleanly with
zero enum/dataclass ceremony.

A raise inside the factory / rehydrator is FINE — it is local to construction and never
crosses a module boundary (house discipline: a structurally-invalid spec is caller error
caught at the boundary). A raise inside the RUNNER is NOT — every expected failure there
is error-as-data (``JobResult(ok=False, JobError(...))``).

No pathlib anywhere. os.path only (none needed here — pure data).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from hugpy_video.intel.media_schema import MediaRef

# The write-back target sentinel: ``"create"`` mints a NEW identity profile per detected
# character; anything else is treated as an EXISTING profile SLUG to append the extracted
# view-set to. (Per CHAR360-FEATURE-PLAN §4b: create | add. The "scan/match" path is S3b
# and is deliberately OUT of scope here — a matched slug simply arrives as ``target``.)
CREATE_TARGET = "create"

# The REVIEW sentinel (CHARACTER-GROUPS-PLAN S1): a THIRD, non-committing target. Unlike
# ``create`` / a slug, ``review`` runs char360 and RETURNS the per-character grouped views
# to the caller WITHOUT writing ANY identity profile — the curation step the edit/move/merge
# UI needs before anything is committed. Structurally it is just another non-empty target
# string (the factory needs no special case); the RUNNER and the route branch on it. Kept a
# reserved word (not a legal slug — ``slugify`` would collapse it, but no profile is ever
# created named "review" through the review path, so it can never collide with a real slug).
REVIEW_TARGET = "review"

# The knobs the service's ``Char360Params`` accepts (mirrors
# identity_render_service/identity_render/api_models.py:Char360Params + char360's
# Char360Spec / CLI defaults). Kept as a bare allow-list so the central side forwards ONLY
# the fields the service understands and NEVER pins a default it does not own — an omitted
# field reproduces the pipeline's one-shot default (stride 8, yolov8m, min_h_frac 0.15,
# cluster_dist 0.55, min_faces 4). Central does NOT re-validate the values (the service's
# pydantic model is the authority); it only drops unknown keys so a typo can never ride
# across the bus as a silent no-op.
CHAR360_PARAM_KEYS = (
    "stride", "yolo_model", "min_h_frac", "cluster_dist", "min_faces",
)


@dataclass(frozen=True)
class IdentityVideoExtractSpec:
    """Frozen, JSON-safe currency of an ``identity_video_extract`` bus job.

        source           the source VIDEO clip to extract from (a ``MediaRef`` whose
                         ``kind`` MUST be ``"video"`` — validated at construction, mirroring
                         ``frame_schema``). Its ``uri`` is an absolute path; the runner
                         forwards it to the service as ``video_path`` (ae + central share
                         the ``/mnt/llm_storage`` mount, so a hundreds-of-MB clip need not
                         be base64-inflated through the request body).
        target           ``"create"`` (mint a NEW profile per detected character),
                         ``"review"`` (CHARACTER-GROUPS-PLAN S1 — run char360 and RETURN the
                         grouped views WITHOUT writing any profile), or an EXISTING profile
                         SLUG (append each character's view-set to it).
        char360_params   OPTIONAL passthrough knobs for the service's ``Char360Params``
                         (stride / yolo_model / min_h_frac / cluster_dist / min_faces). Kept
                         a PLAIN dict on the central side (never a typed sub-spec) — the
                         relay forwards it verbatim and the service validates it. Only the
                         known keys survive construction; unknown keys are dropped.
        identity_id      OPTIONAL correlation id handed to the service (its
                         ``JobCreateRequest`` REQUIRES an ``identity_id``). When ``target``
                         is a slug the route passes that slug; when ``target`` is
                         ``"create"`` the route synthesizes one. Empty/None here means the
                         RUNNER synthesizes a safe fallback id so the service POST never
                         fails on a missing correlation id.
    """
    source: MediaRef
    target: str
    char360_params: Dict[str, Any] = field(default_factory=dict)
    identity_id: Optional[str] = None


def make_identity_video_extract(
    *,
    source: MediaRef,
    target: str,
    char360_params: Optional[Dict[str, Any]] = None,
    identity_id: Optional[str] = None,
) -> IdentityVideoExtractSpec:
    """Validate every field and build the frozen ``IdentityVideoExtractSpec``.

    Raises ``ValueError``/``TypeError`` LOCALLY on any structural violation (house
    discipline: a structurally-invalid spec is caller error caught at the boundary, never
    carried across the bus). Runtime policy failures (the render service unconfigured, the
    target slug naming no profile) are NOT validated here — they surface as errors-as-data
    from the relay runner / store.
    """
    # source: a MediaRef of kind "video" (mirrors frame_schema.make_frame_extract). A raw
    # dict is a caller error — the route rehydrates it via make_media_ref before this.
    if not isinstance(source, MediaRef):
        raise ValueError(
            f"source must be a MediaRef; got {type(source).__name__}")
    if source.kind != "video":
        raise ValueError(
            f"identity_video_extract source must be a video; got kind={source.kind!r}")

    # target: a non-empty string — either the literal "create" or an existing profile slug.
    # (The store, not this factory, is the authority on whether a slug names a live profile;
    # here we only guard structure.)
    if not (isinstance(target, str) and target.strip()):
        raise ValueError(
            f"target must be a non-empty string ('create' or a profile slug); got {target!r}")
    target = target.strip()

    # char360_params: OPTIONAL plain dict. None -> {}. Coerce/keep only the keys the service
    # understands (CHAR360_PARAM_KEYS); drop the rest so a typo never rides across the bus as
    # a silent no-op. Values are NOT range-checked here (the service's pydantic model owns
    # that) — but a non-dict is a structural caller error.
    if char360_params is None:
        char360_params = {}
    if not isinstance(char360_params, dict):
        raise ValueError(
            f"char360_params must be a dict or None; got {type(char360_params).__name__}")
    clean_params: Dict[str, Any] = {
        k: char360_params[k] for k in CHAR360_PARAM_KEYS if k in char360_params
    }

    # identity_id: OPTIONAL correlation id. None/blank -> None (the runner synthesizes a
    # fallback so the service POST never fails on a missing id). A non-string is a caller
    # error; a blank string normalizes to None.
    if identity_id is not None:
        if not isinstance(identity_id, str):
            raise ValueError(
                f"identity_id must be a string or None; got {type(identity_id).__name__}")
        identity_id = identity_id.strip() or None

    return IdentityVideoExtractSpec(
        source=source,
        target=target,
        char360_params=clean_params,
        identity_id=identity_id,
    )


def identity_video_extract_from_dict(d: dict) -> IdentityVideoExtractSpec:
    """Rebuild an ``IdentityVideoExtractSpec`` from its ``asdict`` form THROUGH the
    validating factory (deserialize-then-revalidate, like every other bus spec). Registered
    in ``media_bus.SPEC_DESERIALIZERS`` under ``"identity_video_extract"``.

    ``asdict`` turns the nested ``MediaRef`` into a plain dict over JSON, so rehydrate it
    via ``make_media_ref`` first (mirrors how the frame_extract route / every source-bearing
    spec reconstructs its MediaRef). ``char360_params`` round-trips as a plain dict."""
    # Lazy-free import at module top is fine — media_schema is dependency-light and already
    # imported for the MediaRef type annotation above.
    from hugpy_video.intel.media_schema import make_media_ref

    raw_source = d.get("source")
    if not isinstance(raw_source, dict):
        raise ValueError(
            f"identity_video_extract spec is missing a 'source' MediaRef; got {raw_source!r}")
    source = make_media_ref(**raw_source)

    return make_identity_video_extract(
        source=source,
        target=d["target"],
        char360_params=d.get("char360_params") or {},
        identity_id=d.get("identity_id"),
    )
