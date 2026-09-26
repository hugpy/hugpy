### routes/video_routes.py
"""HTTP surface (Phase 3a) for the Video Intelligence crop feature.

Additive, all-JSON routes over the already-verified headless backbone
(`abstract_hugpy_dev.video_intel`). This module only translates HTTP <-> the
backbone; every invariant (metadata resolution, axis validity, single-writer
job state) lives in the backbone and is reused here, never re-implemented.

Frozen contract (a frontend is being built to the same contract in parallel):
    POST /video/ingest              {"path": "<abspath under /uploads>"} -> MediaRef
    POST /video/jobs/crop           {"source": <MediaRef>, "spatial"?, "temporal"?}
                                    -> {"job_id": ...}
    POST /video/jobs/frame_extract  {"source": <MediaRef>, "fps", "quality", "fmt", ...}
                                    -> {"job_id": ...}
    POST /video/jobs/audio_extract  {"source": <MediaRef>, "fmt"?: "wav"}
                                    -> {"job_id": ...}
    POST /video/jobs/generate_image {"parts": [...], "model_id", ...} -> {"job_id": ...}
    GET  /video/presets             -> {"presets": [ {"id","name","description",
                                    "mode","model_key","defaults":{...},
                                    "recommended"} ]}
    POST /video/presets/<id>/apply  -> auto-pick a GPU worker + assign + warm the
                                    preset's model -> {"ok","worker","model_key",
                                    "mode","defaults":{...},"warming"}
    GET  /video/jobs/<job_id>       -> {"job_id","status","result"}
    GET  /video/media?handle=       -> raw file bytes (source image OR job result)

Mirrors upload_routes' blueprint idiom: `get_bp(...)` mints the Blueprint (the
same shared helper the other *_bp modules use, re-exported through
flask_app.app.functions). Imported directly here so this module stays
self-contained and can be registered on a minimal standalone app for headless
verification without booting the full wsgi stack.
"""
from __future__ import annotations

import dataclasses
import json
import mimetypes
import os
import secrets
import time

from flask import request, jsonify, send_file, Response, stream_with_context

from abstract_flask import get_bp

from hugpy_platform.constants import UPLOADS_HOME, DEFAULT_ROOT
from hugpy_video.intel import media_store, media_bus, identity_profiles, shot_intent
from hugpy_video.intel.placement import PlacementSnapshot, job_placement
from hugpy_video.intel.media_schema import make_media_ref
from hugpy_video.intel.crop_schema import SpatialRegion, TemporalRegion, make_crop
from hugpy_video.intel.frame_schema import make_frame_extract
from hugpy_video.intel.audio_schema import make_audio_extract
from hugpy_video.intel.gen_schema import GenPromptPart, make_generate_image
from hugpy_video.intel.scene_schema import make_generate_scene
from hugpy_video.intel.movie_schema import GoalInterval, make_movie
from hugpy_video.intel.studio.job import make_studio_i2v
from hugpy_video.intel.studio.tester import (
    CATEGORY_KIND as _TESTER_CATEGORY_KIND,
    make_studio_tester,
    mint_battery_dir as _mint_tester_battery_dir,
)
from hugpy_video.intel.studio_movie_schema import (
    _DEFAULT_CONTEXT_FRAMES as _DEFAULT_MOVIE_CONTEXT_FRAMES,
    StudioMovieGoal,
    make_studio_movie,
)
from hugpy_video.intel.identity_reconstruction_schema import (
    DEFAULT_VIEWS as _DEFAULT_RECON_VIEWS,
    make_identity_reconstruction,
    make_identity_mesh,
    MESH_VIEW_NAMES as _MESH_VIEW_NAMES,
)
from hugpy_video.intel.identity_video_extract_schema import make_identity_video_extract
from hugpy_video.intel.identity_from_video_schema import make_identity_from_video
from hugpy_video.intel.chains import (
    resolve_video_parts,
    resolve_video_parts_scene,
    resolve_video_parts_movie,
)
from hugpy_engine.utils.no_think import (
    apply_no_think as _apply_no_think,
    strip_think as no_think,
    with_no_think as _with_no_think,
)
from hugpy_video import studio_assist_log as _assist_log

video_bp, logger = get_bp("video_bp", __name__)


def _log_assist(**fields) -> None:
    """Record ONE studio-assist attempt to the live log. STRICTLY a side-effect:
    ``studio_assist_log.append`` is already total, and this wrapper adds a second
    belt so a logging bug can never move a generate response. See
    comms/studio_assist_log.py (mirrors the eviction-telemetry store)."""
    try:
        _assist_log.append(**fields)
    except Exception:  # noqa: BLE001 — telemetry never breaks the serve path
        logger.debug("studio-assist log emit failed", exc_info=True)


# --------------------------------------------------------------------------- #
# k9 attribution — WHO is enqueueing this video job. Resolved HERE, in the Flask
# request context (media_bus is Flask-free), and threaded onto the job so it
# surfaces in /llm/jobs. Mirrors chat's streaming._resolve_request_principal:
# operator session first (the stronger, first-party identity), then a video-share
# principal (share:<key_id>) for an outside party on a share link, else None
# (unattributed — every self-hosted/open-mode call). Best-effort: attribution
# must never fail an enqueue.
# --------------------------------------------------------------------------- #
def _request_principal():
    try:
        from hugpy_server.app.operator_auth import principal_role, principal_username
        role = principal_role()
        if role == "operator":
            return "operator"
        if role == "member":
            # A member is attributed BY NAME (2026-08-06). "operator" would be a
            # lie now that the two are different tiers, and /llm/jobs is where an
            # admin reads who spent the GPU.
            username = principal_username()
            return f"user:{username}" if username else "member"
    except Exception:  # noqa: BLE001
        pass
    try:
        from hugpy_server.app.video_auth import _video_share_principal
        p = _video_share_principal(request)
        if p:
            return p
    except Exception:  # noqa: BLE001
        pass
    return None


# --------------------------------------------------------------------------- #
# 2026-08-06 ARTIFACT OWNERSHIP — the VIEWER of this request.
#
# The /video gate (video_auth) answers "may this caller use the studio at all".
# It cannot answer "whose clips are these", which is why an anonymous-to-hugpy
# member used to see (and stream) every other account's renders through
# /video/studio/clips, /video/jobs and /video/media?handle=. These helpers are
# the second half: every listing filters on the viewer, and every per-id read
# checks it.
#
# _viewer() -> (unscoped, username):
#   * (True,  <name|None>) — sees the WHOLE catalog, and may narrow it with
#     ?owner=<name>. That is: an admin session, the operator-token M2M path,
#     open mode (self-hosted single-operator), AND a caller with no account at
#     all that nevertheless reached this route.
#   * (False, "alice")     — a MEMBER: sees ONLY rows whose owner is "alice".
# NULL-owner (legacy/unattributed) rows match no member, so they are visible
# only to an unscoped viewer.
#
# WHY "no account" IS UNSCOPED RATHER THAN DENIED — read before changing.
# Authentication for this surface is the GATE's job (video_auth), not this
# module's. By the time a route body runs, the gate has already admitted the
# request as ONE of: an operator token, open mode, a member/admin session, or a
# VIDEO-SHARE credential (``hpv_…``) — the share link being an accountless
# principal BY DESIGN. Refusing "no account" here would silently kill the share
# feature (a share guest could no longer stream the clip the link exists to
# show) and would break every embedding that mounts these blueprints without a
# gate. So this layer answers only the question the gate cannot: "which of the
# catalog's rows belong to the LOGGED-IN MEMBER making this call". A share
# credential therefore sees exactly what it saw before this slice — unchanged,
# and unchangeable from the member side, since minting one is operator-only
# (/keys/video-share is in operator_auth._SENSITIVE). Scoping share links to a
# single artifact is a separate slice; it belongs in the share-key store, not
# in a fail-open/fail-closed flip here.
# --------------------------------------------------------------------------- #
def _viewer():
    try:
        from hugpy_server.app.operator_auth import principal_role, principal_username
        role = principal_role()
        username = principal_username()
    except Exception:  # noqa: BLE001 — an auth-layer hiccup must not open the
        # per-member scoping OR break the surface: scope to nobody-in-particular
        # exactly as an accountless caller, which the gate has already vetted.
        return (True, None)
    if role == "member":
        return (False, username)
    return (True, username if role == "operator" else None)


def _caller_username():
    """The owner string to stamp on artifacts this request creates (None for an
    operator-token / open-mode / share-link caller — such jobs are unattributed
    and therefore visible only to an unscoped viewer afterwards)."""
    _unscoped, username = _viewer()
    return username


def _owner_filter():
    """The owner a LISTING must be scoped to, as ``(scoped, owner)``.

    ``(False, None)``   -> unscoped (the whole catalog).
    ``(True, "alice")`` -> scope to that owner (a member, or an unscoped
                           viewer's explicit ``?owner=`` narrowing).
    ``(True, None)``    -> scope to NOBODY: a MEMBER whose username could not be
                           resolved. An empty listing is the honest answer —
                           never a silent fall-through to the whole catalog."""
    unscoped, username = _viewer()
    if unscoped:
        requested = (request.args.get("owner") or "").strip()
        return (True, requested) if requested else (False, None)
    return (True, username)


def _may_view_job(job_owner) -> bool:
    """May this request read the artifact owned by ``job_owner``? An unscoped
    viewer: always. A member: only their own — which makes a NULL-owner
    (legacy/unattributed) artifact invisible to every member."""
    unscoped, username = _viewer()
    if unscoped:
        return True
    if not job_owner or not username:
        return False
    return job_owner == username


def _forbidden_artifact():
    """The one deny shape for a per-id artifact read the viewer does not own.
    403 (not 404): job ids are uuid4 hex, so there is nothing to enumerate, and
    an honest refusal beats a lie the console would render as 'deleted'."""
    return jsonify({"error": "forbidden: artifact belongs to another account"}), 403


# --------------------------------------------------------------------------- #
# t172 — PUBLIC-BY-DEFAULT, OPT-IN PRIVATE, PER USER for GENERATED MEDIA (movies/
# renders + image/video gen outputs). The EXACT model the identity-profile owner
# gate (_owns_profile/_may_view_profile/_deny_profile) carries, specialized to a
# media job's own ``owner`` + ``private`` (media_bus.visibility_of). It supersedes
# the pre-t172 strict own-only scope (_may_view_job/_owner_filter) for the media
# surface:
#
#   owner    the creating username, stamped by _video_enqueue via _caller_username():
#            "alice" | None (operator-token / open-mode / share-link / no-account —
#            unattributed, read as operator-owned).
#   private  DEFAULT False == PUBLIC (visible to everyone the /video gate admits —
#            this PRESERVES the accepted public movie listing, board t172); True ==
#            visible ONLY to the owner + the operator/admin tier.
#
# VISIBILITY (read):  public -> anyone admitted;  private -> owner or operator/admin.
# MUTATION  (write):  owner or operator/admin ONLY (cancel/pause/resume/archive/
#           toggle) — a share guest may VIEW public media but never mutate it, nor
#           see/consume another account's private media.
# MIGRATION (non-destructive): a legacy owner-less / private-less row reads as
#           owner=None + private=False -> PUBLIC, operator-mutable only. Nothing hidden.
# --------------------------------------------------------------------------- #
def _owns_media(job_owner) -> bool:
    """Is THIS request the media artifact's owner, or the operator/admin tier? The
    MUTATION predicate + the private-view predicate (the media mirror of
    ``_owns_profile``). The operator tier (operator token / open mode / admin /
    valid hp_ key) owns EVERY artifact including legacy NULL-owner rows; any other
    caller must be the exact username stored as ``owner`` (media owner is a
    username, stamped by _caller_username — NOT a principal). A video-share guest /
    accountless caller is NOT an owner (unlike _may_view_job, which admits them as
    an unscoped VIEWER of PUBLIC media)."""
    try:
        from hugpy_server.app.operator_auth import operator_authenticated
        if operator_authenticated():
            return True
    except Exception:  # noqa: BLE001 — an auth-layer hiccup must never GRANT ownership
        pass
    _unscoped, username = _viewer()
    if not username or not job_owner:
        return False
    return job_owner == username


def _may_view_media(job_owner, private) -> bool:
    """Read visibility for one media artifact (mirror of ``_may_view_profile``): a
    PUBLIC artifact (``private`` false — the default and every legacy row) is
    visible to everyone the /video gate admitted; a PRIVATE one only to its owner +
    the operator/admin tier."""
    if not private:
        return True
    return _owns_media(job_owner)


def _deny_media(job_owner, private):
    """The deny response for a caller that may not MUTATE (or, for a private
    artifact, may not even VIEW) this media — mirror of ``_deny_profile`` + the t172
    spec: 403 (``_forbidden_artifact``) when the caller can still SEE it (a foreign
    PUBLIC artifact, the honest 'not yours'); 404 (existence hidden) for a foreign
    PRIVATE one. Job ids are uuid4 hex (unguessable), so the 404 is defense-in-depth
    parity with the profile gate rather than an anti-enumeration necessity."""
    if _may_view_media(job_owner, private):
        return _forbidden_artifact()
    return jsonify({"error": "media not found"}), 404


def _media_view_scope():
    """t172 media-library READ scope for a listing, as ``(mode, value)``:

      ("all",    None)     operator / admin / open mode / share link / no account —
                           the WHOLE catalog (an unscoped viewer may narrow with
                           ?owner=<name> -> ("owner", name)).
      ("owner",  name)     an unscoped viewer's explicit ?owner= narrowing (strict).
      ("viewer", username) a MEMBER — PUBLIC rows (any owner) + their OWN private.
                           ``username`` may be "" (a member whose name did not
                           resolve): the store then scopes to PUBLIC-only.

    Replaces the pre-t172 strict own-only ``_owner_filter`` on the media surface —
    a member now sees every public render, not only their own (public-by-default),
    the listing mirror of ``_may_view_media``."""
    unscoped, username = _viewer()
    if unscoped:
        requested = (request.args.get("owner") or "").strip()
        return ("owner", requested) if requested else ("all", None)
    return ("viewer", username or "")


def _media_scope_sql():
    """The t172 media-library scope as a ``(where_clause, params)`` fragment to AND
    into a raw media_jobs SELECT (the clips / projects listings, which query the bus
    DB directly rather than via list_jobs). ``("", ())`` for the unscoped (all)
    case. PUBLIC rows are ``COALESCE(private,0)=0`` (a legacy/NULL row is PUBLIC —
    the non-destructive migration); a member also matches their OWN rows by owner."""
    mode, value = _media_view_scope()
    if mode == "all":
        return "", ()
    if mode == "owner":                         # admin ?owner= narrowing (strict own)
        return "owner = ?", (value,)
    return "(COALESCE(private,0)=0 OR owner = ?)", (value,)   # member: public + own


def _video_enqueue(name, spec, private=None):
    """media_bus.enqueue with the request principal stamped for attribution (k9),
    the OWNER stamped for authorization (2026-08-06), and the t172 PRIVATE
    visibility flag stamped for public/private scoping. Every /video enqueue route
    funnels through this, so all three ride onto every job uniformly: the principal
    surfaces the job's origin on /llm/jobs, the owner is what the listing/serve
    routes filter on, and ``private`` is the media record's visibility (the sibling
    of owner). An operator-token / open-mode / share-link enqueue stores owner=NULL.

    ``private`` (t172) defaults to reading the ``private`` flag off the enqueue
    request body — public-by-default: absent / non-boolean coerces to False ==
    PUBLIC. A caller may pass it explicitly to override (e.g. a runner re-enqueue
    that propagates the parent's visibility)."""
    if private is None:
        body = request.get_json(silent=True) or {}
        raw = body.get("private", False)
        private = raw if isinstance(raw, bool) else False
    return media_bus.enqueue(name, spec, principal=_request_principal(),
                             owner=_caller_username(), private=bool(private))


@video_bp.errorhandler(media_bus.MediaJobsBusy)
def _media_jobs_busy(exc):
    """Turn a media-jobs store LOCK timeout (SQLITE_BUSY, exhausted retries) into a
    clean, retryable JSON 503 carrying the REAL sqlite reason — never an unhandled
    HTML 500. Every /video/jobs/* enqueue funnels through media_bus.enqueue, so
    this one handler covers them all. The client may re-submit the identical
    request (idempotent: nothing was persisted)."""
    logger.error("video enqueue: media_jobs store locked (%s) after %d attempt(s) "
                 "over %.2fs: %s", exc.op, exc.attempts, exc.seconds, exc.reason)
    return jsonify({
        "error": "media job store is temporarily locked; please retry",
        "reason": exc.reason,
        "op": exc.op,
        "attempts": exc.attempts,
        "waited_seconds": round(exc.seconds, 3),
        "retryable": True,
    }), 503


# --------------------------------------------------------------------------- #
# storage jail — same realpath-under-roots check as media_store._is_within,
# replicated here so a route never touches a path outside the storage roots.
# --------------------------------------------------------------------------- #
def _is_within(path: str, root: str) -> bool:
    if not root:
        return False
    rp = os.path.realpath(path)
    rr = os.path.realpath(root)
    try:
        return os.path.commonpath([rp, rr]) == rr
    except ValueError:
        return False


def _jail_resolve(handle):
    """Resolve a caller-supplied path to a realpath under UPLOADS_HOME or
    DEFAULT_ROOT. Returns the resolved realpath, or None if it escapes the jail
    (or is missing/ill-typed) — the single seam that keeps these routes from
    becoming an arbitrary-file-read/write."""
    if not handle or not isinstance(handle, str):
        return None
    rp = os.path.realpath(handle)
    if _is_within(rp, UPLOADS_HOME) or _is_within(rp, DEFAULT_ROOT):
        return rp
    return None


def _resolve_asset_uri(asset_id):
    """Resolve a media-catalog ``asset_id`` to the ``uri`` (abs path) of the produced
    ref that carries it — the B2 chain lookup so the console can hand the studio a
    prior tier's output BY ID (``source_asset_id``) rather than a path. Scans the
    media-bus job store (the same durable catalog ``/video/studio/clips`` reads) for a
    job result whose ``outputs[*].asset_id`` matches, newest first, and returns that
    output's uri. Read-only ``mode=ro`` connection (mirrors the clips-list route); the
    subsequent jail + ffprobe validation in the caller still guards the returned path.
    Returns the uri string, or None (unknown id / no catalog / transient lock)."""
    if not asset_id or not isinstance(asset_id, str):
        return None
    import json as _json
    import sqlite3
    try:
        conn = sqlite3.connect(
            f"file:{media_bus.DB_PATH}?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            rows = conn.execute(
                "SELECT result_json FROM media_jobs "
                "WHERE result_json IS NOT NULL ORDER BY updated DESC LIMIT 1000"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    for (result_json,) in rows:
        if not result_json:
            continue
        try:
            res = _json.loads(result_json)
        except (ValueError, TypeError):
            continue
        for o in (res.get("outputs") or []):
            if (isinstance(o, dict) and o.get("asset_id") == asset_id
                    and isinstance(o.get("uri"), str) and o["uri"]):
                return o["uri"]
    return None


def _autofit_vram_budget(raw):
    """A BLANK/absent/null ``vram_budget_gb`` means AUTOFIT (return ``None``): the studio
    render sizes the routing budget to the SERVING WORKER's GPU CAPACITY at render time,
    rather than a low guess that is guaranteed to fail (operator doctrine 2026-07-12:
    "if a model needs 14GB and it's blank, just do 14, otherwise a fail is 100% likely").
    An EXPLICIT number is the manual override — passed through untouched (a bad value still
    400s in the validating factory). None threads through the spec to render_clip, which
    resolves it against capacity and lets the reservation engine evict to free that room;
    unreadable VRAM REFUSES (operator 2026-07-27) instead of degrading to a synthetic
    default."""
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    return raw


def _ingest_image_references(raws):
    """Jail-resolve + ``media_store.ingest(kind_hint="image")``-classify a list of raw
    reference-image paths into media-store URIs — the ONE code path a studio spec's id_lock
    references (movie-level OR the S2-movie per-goal view refs) must pass through, because
    the renderer consumes media_store URIs, not raw paths. Errors-as-data: returns
    ``(uris, None)`` on success, or ``(None, (payload, status))`` on the FIRST bad path (a
    non-string, a jail escape -> 400, a missing file -> 404, an unreadable/non-image file ->
    400) so the caller returns the 4xx verbatim. Factored out of the movie-level reference
    loop so the per-goal bank frames get IDENTICAL jail + classify treatment (no duplication,
    no drift)."""
    uris = []
    for raw in raws:
        if not isinstance(raw, str) or not raw.strip():
            return None, ({"error": "each reference_image must be a non-empty path"}, 400)
        rp = _jail_resolve(raw)
        if rp is None:
            return None, ({"error": "reference_image outside storage jail"}, 400)
        if not os.path.isfile(rp):
            return None, ({"error": "reference_image not found"}, 404)
        try:
            iref = media_store.ingest(rp, kind_hint="image", owner=_caller_username())
        except Exception as exc:  # unreadable / not a real image = bad input
            return None, ({"error": f"reference_image is not a readable media file: {exc}"}, 400)
        if iref.kind != "image":
            return None, ({"error": f"reference_image is not an image (classified as {iref.kind})"}, 400)
        uris.append(iref.uri)
    return uris, None


# --------------------------------------------------------------------------- #
# 1) POST /video/ingest — resolve metadata ONCE, mint a MediaRef
# --------------------------------------------------------------------------- #
@video_bp.route("/video/ingest", methods=["POST"])
def video_ingest():
    body = request.get_json(silent=True) or {}
    path = body.get("path")
    resolved = _jail_resolve(path)
    if resolved is None:
        return jsonify({"error": f"path missing or outside storage jail: {path!r}"}), 400
    try:
        ref = media_store.ingest(resolved, owner=_caller_username())
    except Exception as exc:  # ingest raises locally (FileNotFound/Value/Runtime)
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
    return jsonify(dataclasses.asdict(ref)), 200


# --------------------------------------------------------------------------- #
# 2) POST /video/jobs/crop — validate + enqueue a crop job
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/crop", methods=["POST"])
def video_crop():
    body = request.get_json(silent=True) or {}
    source_d = body.get("source")
    if not isinstance(source_d, dict):
        return jsonify({"error": "missing or invalid 'source' MediaRef"}), 400
    sp = body.get("spatial")
    tp = body.get("temporal")
    try:
        source = make_media_ref(**source_d)
        spatial = SpatialRegion(**sp) if sp is not None else None
        temporal = TemporalRegion(**tp) if tp is not None else None
        spec = make_crop(source=source, spatial=spatial, temporal=temporal)
    except (ValueError, TypeError) as exc:  # invalid axis combo / bad fields = 400
        return jsonify({"error": str(exc)}), 400
    job_id = _video_enqueue("crop", spec)
    return jsonify({"job_id": job_id}), 200


# --------------------------------------------------------------------------- #
# 2b) POST /video/jobs/frame_extract — validate + enqueue a frame-extract job
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/frame_extract", methods=["POST"])
def video_frame_extract():
    body = request.get_json(silent=True) or {}
    source_d = body.get("source")
    if not isinstance(source_d, dict):
        return jsonify({"error": "missing or invalid 'source' MediaRef"}), 400
    win = body.get("window")
    try:
        source = make_media_ref(**source_d)
        window = TemporalRegion(**win) if win is not None else None
        spec = make_frame_extract(
            source=source,
            fps=body.get("fps"),
            quality=body.get("quality"),
            fmt=body.get("fmt"),
            window=window,
            max_frames=body.get("max_frames"),
        )
    except (ValueError, TypeError) as exc:  # bad fields / axis combo = 400
        return jsonify({"error": str(exc)}), 400
    job_id = _video_enqueue("frame_extract", spec)
    return jsonify({"job_id": job_id}), 200


# --------------------------------------------------------------------------- #
# 2b') POST /video/jobs/audio_extract — validate + enqueue an audio-extract job
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/audio_extract", methods=["POST"])
def video_audio_extract():
    body = request.get_json(silent=True) or {}
    source_d = body.get("source")
    if not isinstance(source_d, dict):
        return jsonify({"error": "missing or invalid 'source' MediaRef"}), 400
    try:
        source = make_media_ref(**source_d)
        spec = make_audio_extract(
            source=source,
            fmt=body.get("fmt", "wav"),
        )
    except (ValueError, TypeError) as exc:  # bad fields / non-video source = 400
        return jsonify({"error": str(exc)}), 400
    job_id = _video_enqueue("audio_extract", spec)
    return jsonify({"job_id": job_id}), 200


# --------------------------------------------------------------------------- #
# 2c) POST /video/jobs/generate_image — validate + resolve video parts + enqueue
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/generate_image", methods=["POST"])
def video_generate_image():
    body = request.get_json(silent=True) or {}
    parts_in = body.get("parts")
    if not isinstance(parts_in, list) or not parts_in:
        return jsonify({"error": "missing or empty 'parts' list"}), 400
    try:
        parts = []
        for pd in parts_in:
            if not isinstance(pd, dict):
                raise ValueError("each part must be an object")
            media_d = pd.get("media")
            media = make_media_ref(**media_d) if isinstance(media_d, dict) else None
            parts.append(GenPromptPart(
                kind=pd.get("kind"),
                text=pd.get("text"),
                media=media,
            ))
        spec = make_generate_image(
            parts=tuple(parts),
            model_id=body.get("model_id"),
            width=body.get("width"),
            height=body.get("height"),
            steps=body.get("steps"),
            guidance=body.get("guidance"),
            seed=body.get("seed"),
            negative=body.get("negative"),
            strength=body.get("strength"),   # img2img (additive, optional)
            project=body.get("project"),     # auto-archive NAME (optional)
        )
    except (ValueError, TypeError) as exc:  # bad fields / part combo = 400
        return jsonify({"error": str(exc)}), 400

    # Resolve any VIDEO parts to image frames BEFORE enqueue (Phase 7 chain), so
    # the runner never sees a video part. Extraction failure -> 400 (see chains).
    try:
        resolved = resolve_video_parts(spec)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = _video_enqueue("generate_image", resolved)
    return jsonify({"job_id": job_id}), 200


# --------------------------------------------------------------------------- #
# 2d) POST /video/jobs/generate_scene — one query -> N frames (+ optional mp4)
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/generate_scene", methods=["POST"])
def video_generate_scene():
    body = request.get_json(silent=True) or {}
    parts_in = body.get("parts")
    if not isinstance(parts_in, list) or not parts_in:
        return jsonify({"error": "missing or empty 'parts' list"}), 400
    try:
        parts = []
        for pd in parts_in:
            if not isinstance(pd, dict):
                raise ValueError("each part must be an object")
            media_d = pd.get("media")
            media = make_media_ref(**media_d) if isinstance(media_d, dict) else None
            parts.append(GenPromptPart(
                kind=pd.get("kind"),
                text=pd.get("text"),
                media=media,
            ))
        spec = make_generate_scene(
            parts=tuple(parts),
            model_id=body.get("model_id"),
            width=body.get("width"),
            height=body.get("height"),
            steps=body.get("steps"),
            guidance=body.get("guidance"),
            n_frames=body.get("n_frames"),
            fps=body.get("fps"),
            assemble=body.get("assemble"),
            seed=body.get("seed"),
            motion=body.get("motion"),
            negative=body.get("negative"),
            # img2img additive knobs (optional; absent -> factory defaults:
            # strength None -> runner 0.45; chain defaults True).
            strength=body.get("strength"),
            chain=body.get("chain", True),
            project=body.get("project"),     # auto-archive NAME (optional)
        )
    except (ValueError, TypeError) as exc:  # bad fields / part combo / frame_cap = 400
        return jsonify({"error": str(exc)}), 400

    # Resolve any VIDEO parts to image frames BEFORE enqueue (chain twin), so the
    # runner never sees a video part. Extraction failure -> 400 (see chains).
    try:
        resolved = resolve_video_parts_scene(spec)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = _video_enqueue("generate_scene", resolved)
    return jsonify({"job_id": job_id}), 200


# --------------------------------------------------------------------------- #
# 2d') POST /video/jobs/generate_movie — a GOAL TIMELINE -> a stitched movie
# --------------------------------------------------------------------------- #
# Mirrors generate_scene: parse body -> build GoalIntervals -> make_movie
# (validates contiguity/ranges) -> resolve any video goal refs -> enqueue. The
# scene-template fields (model/size/steps/…) are shared by every segment; `goals`
# is the ordered, contiguous, non-overlapping timeline; the director knobs turn on
# optional per-segment vision scoring + retry.
@video_bp.route("/video/jobs/generate_movie", methods=["POST"])
def video_generate_movie():
    body = request.get_json(silent=True) or {}
    goals_in = body.get("goals")
    if not isinstance(goals_in, list) or not goals_in:
        return jsonify({"error": "missing or empty 'goals' list"}), 400
    try:
        goals = []
        for gd in goals_in:
            if not isinstance(gd, dict):
                raise ValueError("each goal must be an object")
            ref_d = gd.get("ref")
            ref = make_media_ref(**ref_d) if isinstance(ref_d, dict) else None
            goals.append(GoalInterval(
                start_frame=gd.get("start_frame"),
                end_frame=gd.get("end_frame"),
                prompt=gd.get("prompt"),
                ref=ref,
                # per-goal PROMPT-COMPONENT overrides (k92) — absent = None = inherit
                # the movie-level knob; make_movie validates each when present.
                model_id=gd.get("model_id"),
                width=gd.get("width"),
                height=gd.get("height"),
                steps=gd.get("steps"),
                guidance=gd.get("guidance"),
                seed=gd.get("seed"),
                negative=gd.get("negative"),
                strength=gd.get("strength"),
                chain=gd.get("chain"),
                motion=gd.get("motion"),
            ))
        spec = make_movie(
            goals=tuple(goals),
            model_id=body.get("model_id"),
            width=body.get("width"),
            height=body.get("height"),
            steps=body.get("steps"),
            guidance=body.get("guidance"),
            fps=body.get("fps"),
            assemble=body.get("assemble"),
            seed=body.get("seed"),
            negative=body.get("negative"),
            strength=body.get("strength"),
            chain=body.get("chain", True),
            project=body.get("project"),
            # director knobs (optional; absent -> factory defaults)
            vision_enabled=body.get("vision_enabled", False),
            score_threshold=body.get("score_threshold", 60),
            max_attempts_per_segment=body.get("max_attempts_per_segment", 1),
            judge_model_id=body.get("judge_model_id"),
            time_budget_s=body.get("time_budget_s"),
        )
    except (ValueError, TypeError) as exc:  # bad fields / contiguity / frame_cap = 400
        return jsonify({"error": str(exc)}), 400

    # Resolve any VIDEO goal refs to a representative still BEFORE enqueue, so the
    # runner never sees a video ref. Extraction failure -> 400 (see chains).
    try:
        resolved = resolve_video_parts_movie(spec)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = _video_enqueue("generate_movie", resolved)
    return jsonify({"job_id": job_id}), 200


@video_bp.route("/video/jobs/performance", methods=["POST"])
def video_performance():
    """Submit an Oracle performance through the same attributed media bus as Studio."""
    from hugpy_oracle.relay.performance_relay import validate_performance_spec

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "performance body must be a JSON object"}), 400
    private = body.get("private", False)
    if not isinstance(private, bool):
        return jsonify({"error": "private must be a boolean"}), 400
    try:
        spec = validate_performance_spec({k: v for k, v in body.items() if k != "private"})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    job_id = _video_enqueue("video_performance", spec, private=private)
    return jsonify({"job_id": job_id}), 200


@video_bp.route("/video/performance/probe", methods=["GET"])
def video_performance_probe():
    """Show which Oracle stages this server can actually serve."""
    from hugpy_oracle.relay.performance_relay import probe
    return jsonify(probe()), 200


@video_bp.route("/video/identity-render/probe", methods=["GET"])
def video_identity_render_probe():
    """Read the external identity service's readiness without exposing its secret."""
    from hugpy_video.intel.runners.identity_render_client import service_config, auth_headers

    url, token = service_config()
    if not url or not token:
        return jsonify({"configured": False, "reachable": False,
                        "reason": "Set IDENTITY_RENDER_URL and IDENTITY_RENDER_TOKEN on the server."}), 200
    try:
        import requests
        response = requests.get(f"{url}/health", headers=auth_headers(token), timeout=2)
        if response.status_code != 200:
            return jsonify({"configured": True, "reachable": False,
                            "reason": f"identity service health returned HTTP {response.status_code}"}), 200
        body = response.json()
        if not isinstance(body, dict) or body.get("ok") is not True:
            raise ValueError("invalid health response")
        return jsonify({"configured": True, "reachable": True,
                        "version": body.get("version"),
                        "capabilities": body.get("capabilities") or {}}), 200
    except (requests.RequestException, ValueError):
        return jsonify({"configured": True, "reachable": False,
                        "reason": "identity service health is unavailable"}), 200


# --------------------------------------------------------------------------- #
# 2d'') POST /video/studio/i2v — a studio image-to-video clip via the cinema
#        studio spine (B2). Mirrors the movie/scene routes: parse body -> build
#        the validated StudioI2VSpec -> media_bus.enqueue -> {job_id}. The job
#        runs through the studio's own router->manifest->runner->content-addressed
#        clip path (produce_clip) and its output is cataloged in the media store.
#        Query it exactly like any other media job: GET /video/jobs/<job_id>.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/studio/i2v", methods=["POST"])
def video_studio_i2v():
    body = request.get_json(silent=True) or {}
    # resolution may arrive nested ({"resolution": {"width","height","fps"}}) or as
    # flat top-level keys — accept both (nested wins, mirrors the frontend contract).
    res = body.get("resolution") if isinstance(body.get("resolution"), dict) else {}
    width = res.get("width", body.get("width"))
    height = res.get("height", body.get("height"))
    fps = res.get("fps", body.get("fps"))
    # sane studio default so an empty POST still produces a clip (synthetic spine).
    # 832x480 (R_480P), not 512x512. A square 512 default was a GUARANTEED FAIL for
    # two whole capabilities: it is outside EVERY id-capable and EVERY v2v-capable
    # model's envelope on this fleet (Wan-VACE maxes at 480p), so an id_lock request
    # at the default either refused with no_capable_model or — worse — fell through
    # to the i2v runner, which does not read reference_images, and silently rendered
    # the wrong person. A default that cannot succeed is a failure promise
    # (defaults-are-promises); this one is the geometry the studio actually serves.
    width = 832 if width is None else width
    height = 480 if height is None else height
    fps = 24 if fps is None else fps
    # capability defaults to "i2v" (backward-compat). SHAPE is validated inside
    # make_studio_i2v; VIABILITY is validated right here, first, before anything else
    # in this handler does any work — see the block immediately below.
    capability = body.get("capability", "i2v")

    # ----------------------------------------------------------------------- #
    # CAPABILITY GATE — refuse a dead capability HERE, at the boundary, before a
    # job_id exists (2026-07-27).
    #
    # WHY THIS BLOCK IS THE FIRST THING THE HANDLER DOES. ``presets.capability_verdict``
    # was written to be consumed at the route layer — its own docstring said the route
    # "refuse[s] at the BOUNDARY (an honest 400 naming what IS served) instead of
    # enqueuing a job that dies in a runner three layers down" — and then NO route
    # called it. ``studio/job.py::_VALID_CAPABILITIES`` is still every member of the
    # Capability enum, so the boundary admitted all 16. MEASURED on this route
    # 2026-07-27, before this block existed:
    #
    #     audio    -> 200 {"job_id": ...}      lipsync  -> 200 {"job_id": ...}
    #     restore  -> 200 {"job_id": ...}      stream   -> 200 {"job_id": ...}
    #     keyframe -> 200 {"job_id": ...}      inpaint/outpaint/retake -> 200
    #
    # Every one of those burned a media-bus queue slot, showed the caller a job id to
    # poll, and then failed (or, worse for the four VACE-shaped ones, SUCCEEDED at
    # rendering something else — a plain full restyle wearing the requested capability
    # name). The router refuses the same set (CapabilityRouter.resolve's capability
    # gate), but the router only runs once a job has been ADMITTED, and "the user
    # discovers this after a job is accepted, queued and started" is the exact failure
    # this slice exists to delete. So: same verdict, one hop earlier, no queue slot.
    #
    # THE WORDING IS THE REGISTRY'S, verbatim. ``verdict.refusal`` already names the
    # measured blocker AND the fleet menu of what IS renderable; re-phrasing it here
    # would give the console and the log two different explanations of one fact.
    #
    # NOTHING WORKING IS NARROWED. The gate only fires for capabilities NO ratified
    # RenderPreset covers — 8 of 16 today (audio, inpaint, keyframe, lipsync, outpaint,
    # restore, retake, stream). t2v / i2v / v2v / id_lock / motion / upres / interp /
    # assemble pass through untouched and land on exactly the code path they did
    # before; tests/test_video_presets_route.py asserts both halves against the live
    # test client.
    #
    # AN UNKNOWN capability string is answered here too, rather than falling through
    # to make_studio_i2v's ValueError, so the caller gets the SAME shape of answer
    # (a 400 that names the alternatives) whether they asked for a typo or for
    # something real that this fleet cannot do.
    # ----------------------------------------------------------------------- #
    from hugpy_video.intel.studio import presets as _render_presets
    from hugpy_video.intel.studio.enums import Capability as _Capability
    try:
        _cap_enum = _Capability(capability)
    except (ValueError, TypeError):
        return jsonify({
            "error": f"unknown capability {capability!r}; this fleet renders: "
                     f"{_render_presets.available_menu()}",
            "capability": capability,
            "available": _render_presets.available_menu(),
        }), 400
    _verdict = _render_presets.capability_verdict(_cap_enum)
    if not _verdict.servable:
        return jsonify({
            "error": _verdict.refusal,
            "capability": _cap_enum.value,
            "reason": _verdict.reason,
            "available": _render_presets.available_menu(),
        }), 400
    # a start_image, if supplied, must resolve inside the storage jail (never an
    # arbitrary-file read) — same seam as /video/ingest. T2V is TEXT-ONLY, so a
    # start_image is meaningless for it: we DELIBERATELY IGNORE it (drop to None),
    # never jail-resolve or reject it — a t2v clip is a pure function of prompt +
    # seed + geometry. i2v (the default) is unaffected.
    start_image = body.get("start_image")
    if capability == "t2v":
        start_image = None
    elif start_image is not None:
        start_image = _jail_resolve(start_image)
        if start_image is None:
            return jsonify({"error": "start_image outside storage jail"}), 400
    # source_video (B2 movie->studio chain): the prior tier's clip this studio job
    # extends. Accept EITHER an absolute "source_video" path (jail-resolved like
    # start_image) OR a "source_asset_id" resolved to its uri via the media catalog.
    # An i2v job with a source but no start_image extends the clip from its LAST FRAME
    # (the runner does the extraction); t2v is text-only, so a source is meaningless
    # and DELIBERATELY DROPPED. A non-video / nonexistent / jail-escaping target is a
    # clean 4xx here rather than a deferred runner failure.
    source_video = body.get("source_video")
    source_asset_id = body.get("source_asset_id")
    if capability == "t2v":
        source_video = None
    else:
        if source_video is None and source_asset_id:
            source_video = _resolve_asset_uri(source_asset_id)
            if source_video is None:
                return jsonify(
                    {"error": f"source_asset_id not found in catalog: {source_asset_id!r}"}), 404
        if source_video is not None:
            resolved_sv = _jail_resolve(source_video)
            if resolved_sv is None:
                return jsonify({"error": "source_video outside storage jail"}), 400
            if not os.path.isfile(resolved_sv):
                return jsonify({"error": "source_video not found"}), 404
            # Authoritative video check: ffprobe-classify via media_store (probe wins).
            try:
                sref = media_store.ingest(resolved_sv, kind_hint="video", owner=_caller_username())
            except Exception as exc:  # unreadable / no A/V stream / jail = bad input
                return jsonify(
                    {"error": f"source_video is not a readable media file: {exc}"}), 400
            if sref.kind != "video":
                return jsonify(
                    {"error": f"source_video is not a video (classified as {sref.kind})"}), 400
            source_video = sref.uri
    # IDENTITY LOCK (id_lock): reference image(s) of the subject preserved across the
    # render (Wan VACE reference-to-video). Each is jail-resolved + ffprobe/PIL-classified
    # as an IMAGE (a non-image / jail-escaping / missing target is a clean 4xx here, not a
    # deferred runner failure). Up to 4 (all consumed by diffusers 0.39). Permitted for
    # the VACE capabilities (id_lock / v2v — the runners that consume them); rejected for
    # any other capability so a non-VACE runner can never silently ignore them.
    _REF_CAPS = {"id_lock", "v2v"}
    # UNIFIED IDENTITY (2026-07-12): an enqueue may name a saved identity profile
    # instead of raw reference_images; the profile's curated set is canonical. Its
    # OWN bounded block (a concurrent vram_budget_gb edit lives elsewhere in this
    # route — keep these seams from entangling).
    reference_images_in, _prof_err = _reference_images_from_body(body)
    if _prof_err is not None:
        _pl, _st = _prof_err
        return jsonify(_pl), _st
    # NOTE (2026-07-16): canonical may now hold 8 views, but _reference_images_from_body
    # already narrows a profile-resolved set to the RENDER cap (4) before returning, so the
    # >4 check below can never fire on an identity's own DNA. It still guards a caller's RAW
    # reference_images list, where >4 stays a clean caller error.
    resolved_refs: list = []
    if reference_images_in is not None:
        if not isinstance(reference_images_in, list):
            return jsonify({"error": "reference_images must be a list of paths"}), 400
        if capability not in _REF_CAPS:
            return jsonify({"error": "reference_images require capability id_lock or v2v; "
                                     f"got {capability!r}"}), 400
        if len(reference_images_in) > 4:
            return jsonify({"error": "at most 4 reference_images are accepted"}), 400
        for raw in reference_images_in:
            if not isinstance(raw, str) or not raw.strip():
                return jsonify({"error": "each reference_image must be a non-empty path"}), 400
            rp = _jail_resolve(raw)
            if rp is None:
                return jsonify({"error": "reference_image outside storage jail"}), 400
            if not os.path.isfile(rp):
                return jsonify({"error": "reference_image not found"}), 404
            try:
                iref = media_store.ingest(rp, kind_hint="image", owner=_caller_username())
            except Exception as exc:  # unreadable / not a real image = bad input
                return jsonify(
                    {"error": f"reference_image is not a readable media file: {exc}"}), 400
            if iref.kind != "image":
                return jsonify(
                    {"error": f"reference_image is not an image (classified as {iref.kind})"}), 400
            resolved_refs.append(iref.uri)
    # ROUTE RULE: capability id_lock REQUIRES >=1 reference image.
    if capability == "id_lock" and not resolved_refs:
        return jsonify(
            {"error": "capability id_lock requires at least one reference_image"}), 400

    # VACE control still (pose|depth|sketch) — a single image + its kind, jail-resolved
    # and image-classified. Valid for TWO capabilities, and it means something different
    # in each:
    #   * id_lock — OPTIONAL composition blocking alongside the identity references;
    #   * motion  — REQUIRED, and it IS the capability (see the block below).
    #
    # ⚠ WIDENED FROM id_lock-ONLY, 2026-07-27, and this is the fix that made `motion`
    # a real capability instead of a phantom. Until today the gate here was
    # `capability != "id_lock"`, which meant the four capabilities the old
    # clip-control-480p preset advertised (motion/inpaint/outpaint/retake) 400'd the
    # instant a control was supplied — and, with no control, sailed through to render a
    # PLAIN FULL RESTYLE identical to v2v. The preset table advertised four controlled
    # edits and the route could serve none of them.
    #
    # ``control_image`` is a genuinely distinct VACE branch (wan_vace.py:534-547: the
    # still is loaded, resized and repeated across num_frames as the pipeline's
    # `video=` control channel — not v2v's decoded source frames, not id_lock's
    # reference latents), so motion was made REACHABLE. inpaint/outpaint/retake were
    # not: no mask, expanded canvas or frame-range input exists anywhere in the spine,
    # so they became honest refusals at the capability gate above rather than routes.
    _CONTROL_CAPS = {"id_lock", "motion"}
    control_image = body.get("control_image")
    control_kind = body.get("control_kind")
    if (control_image is not None or control_kind is not None) \
            and capability not in _CONTROL_CAPS:
        return jsonify(
            {"error": "control_image/control_kind are only valid with capability "
                      f"id_lock or motion; got {capability!r}"}), 400
    # ROUTE RULE: capability motion REQUIRES a control_image — the exact mirror of the
    # id_lock/reference_images rule below, and for the same reason. A motion request
    # with no control carries NO conditioning this runner can tell apart from a plain
    # t2v/v2v render: wan_vace would either take the v2v branch (if a source clip rode
    # along) or refuse deep in preflight with SOURCE_MISSING after the job had already
    # been admitted and queued. Naming it here costs the caller one round trip.
    if capability == "motion" and control_image is None:
        return jsonify(
            {"error": "capability motion requires a control_image (pose|depth|sketch) — "
                      "it is the structural control that makes this a motion render "
                      "rather than a plain restyle; for a prompt-only clip use t2v, "
                      "for restyling an existing clip use v2v"}), 400
    # ...and it must be the ONLY control channel in play. wan_vace picks exactly one
    # (`if vace_context_frames: elif source_video: elif control_image:` at :498-547), so
    # a request carrying BOTH a source clip and a control still would silently drop the
    # control and render a full restyle — the phantom shape this whole slice exists to
    # delete, reappearing one layer down. Refuse it by name instead.
    if control_image is not None and source_video is not None:
        return jsonify(
            {"error": "control_image and source_video cannot be combined: the VACE "
                      "control channel takes ONE input and source_video wins, so the "
                      "control still would be silently ignored and you would get a "
                      "plain restyle. Send a control_image (motion) or a source_video "
                      "(v2v), not both"}), 400
    if control_image is not None:
        if not isinstance(control_image, str) or not control_image.strip():
            return jsonify({"error": "control_image must be a non-empty path"}), 400
        if control_kind not in ("pose", "depth", "sketch"):
            return jsonify({"error": "control_kind must be one of pose|depth|sketch"}), 400
        cp = _jail_resolve(control_image)
        if cp is None:
            return jsonify({"error": "control_image outside storage jail"}), 400
        if not os.path.isfile(cp):
            return jsonify({"error": "control_image not found"}), 404
        try:
            cref = media_store.ingest(cp, kind_hint="image", owner=_caller_username())
        except Exception as exc:
            return jsonify(
                {"error": f"control_image is not a readable media file: {exc}"}), 400
        if cref.kind != "image":
            return jsonify(
                {"error": f"control_image is not an image (classified as {cref.kind})"}), 400
        control_image = cref.uri
    elif control_kind is not None:
        return jsonify({"error": "control_kind requires a control_image"}), 400
    # SAMPLER OVERRIDES (route passthrough): optional "steps"/"cfg" numbers that PIN the
    # denoise settings (explicit values ALWAYS win over the bound model's family
    # default). Validate ranges HERE for a clean 400 with a precise message (steps 1-100,
    # cfg 0-20); make_studio_i2v re-checks the same bounds for non-route callers. An
    # integral float steps (e.g. 30.0 from a JS number input) is accepted as 30.
    steps = body.get("steps")
    if isinstance(steps, float) and steps.is_integer():
        steps = int(steps)
    if steps is not None and (not isinstance(steps, int) or isinstance(steps, bool)
                              or not (1 <= steps <= 100)):
        return jsonify({"error": "steps must be an integer in [1, 100]"}), 400
    cfg = body.get("cfg")
    if cfg is not None and (not isinstance(cfg, (int, float)) or isinstance(cfg, bool)
                            or not (0 <= cfg <= 20)):
        return jsonify({"error": "cfg must be a number in [0, 20]"}), 400
    # DIRECT MODEL CHOICE (pin): optional "model_id". Threaded into the spec -> the
    # CapabilityRequest pin. The router binds THAT model or returns a clear Err-as-data
    # (PINNED_MODEL_UNAVAILABLE / a sharpened gate reason) that rides back on the job —
    # never a silent fallback. Shape (non-empty string) is checked in make_studio_i2v.
    model_id = body.get("model_id")
    # CLIP LENGTH (2026-07-27): the caller's requested frame count. THE LAST HOP of a
    # chain that two review rounds found broken one layer lower each time — the field
    # existed on RenderManifest, then on StudioI2VSpec, then produce_clip grew the
    # parameter, and each round the route still dropped the body key silently. Absent /
    # null -> None -> the model's default length. Shape is validated here (a typo guard,
    # NOT a ceiling: an over-large request CLAMPS to the model's real max with a recorded
    # reason at render time, because the model is not bound yet at spec-build time).
    requested_frames = body.get("requested_frames")
    if isinstance(requested_frames, float) and requested_frames.is_integer():
        requested_frames = int(requested_frames)
    if requested_frames is not None and (
            not isinstance(requested_frames, int) or isinstance(requested_frames, bool)
            or not (1 <= requested_frames <= 10_000)):
        return jsonify({"error": "requested_frames must be an integer in [1, 10000]"}), 400
    try:
        spec = make_studio_i2v(
            capability=capability,
            width=width,
            height=height,
            fps=fps,
            # AUTOFIT: blank/absent/null -> None (size to the serving worker's free VRAM);
            # an explicit number is the manual override (unchanged).
            vram_budget_gb=_autofit_vram_budget(body.get("vram_budget_gb")),
            seed=body.get("seed", 0),
            out_root=body.get("out_root"),
            start_image=start_image,
            # C-prompt: accept "negative_prompt" (canonical) with backward-compat to
            # the older "negative" key; "prompt" carries the positive text prompt.
            negative=body.get("negative_prompt", body.get("negative")),
            prompt=body.get("prompt"),
            project=body.get("project"),     # auto-archive NAME (optional)
            # B2 chain: the validated abs path of the prior tier's clip (or None).
            source_video=source_video,
            # Sampler overrides + model pin (validated above / shape-checked in factory).
            steps=steps,
            cfg=cfg,
            model_id=model_id,
            requested_frames=requested_frames,
            # IDENTITY LOCK (id_lock): the validated reference image uris + optional VACE
            # control still (both already jail-resolved + image-classified above).
            reference_images=tuple(resolved_refs),
            control_image=control_image,
            control_kind=control_kind,
        )
    except (ValueError, TypeError) as exc:  # bad geometry / capability / overrides = 400
        return jsonify({"error": str(exc)}), 400
    job_id = _video_enqueue("studio_i2v", spec)
    return jsonify({"job_id": job_id}), 200


@video_bp.route("/video/studio/tester", methods=["POST"])
def video_studio_tester():
    """Studio TESTER — sweep ONE prompt across EVERY servable model of a category's
    type, recording a model-battery run-dir (one row per model).

    Operator-gated (see operator_auth._SENSITIVE): a full sweep is many GPU
    generations, so it runs as a BACKGROUND media-bus job (name "studio_tester") —
    the request enqueues and returns immediately with the job id and the battery
    run-dir; it does NOT block for the sweep. Per-model results stream live to the
    studio-assist log (correlated by the job id as run_id).

    Body:  {"category": "image"|"scene"|"clip"|"movie", "prompt": str,
            "models"?: [str], "start_image"?: str, "width"?, "height"?, "fps"?,
            "seed"?, "steps"?, "cfg"?, "requested_frames"?, "negative"?,
            "include_synthetic"?: bool, "run_label"?: str}
    200 -> {"job_id", "battery_run_dir", "category", "kind", "poll"}
    400 -> {"error"} on a bad category / empty prompt / malformed field.
    """
    body = request.get_json(silent=True) or {}
    models_in = body.get("models")
    if models_in is not None and not isinstance(models_in, list):
        return jsonify({"error": "'models' must be a list of model ids"}), 400
    try:
        spec = make_studio_tester(
            category=body.get("category"),
            prompt=body.get("prompt"),
            models=(tuple(models_in) if models_in else ()),
            out_root=body.get("out_root"),
            width=body.get("width", 768),
            height=body.get("height", 768),
            fps=body.get("fps", 16),
            seed=body.get("seed", 0),
            start_image=body.get("start_image"),
            steps=body.get("steps"),
            cfg=body.get("cfg"),
            requested_frames=body.get("requested_frames"),
            negative=body.get("negative"),
            include_synthetic=bool(body.get("include_synthetic", False)),
            run_label=body.get("run_label"),
        )
    except (ValueError, TypeError) as exc:  # bad category / prompt / field = 400
        return jsonify({"error": str(exc)}), 400

    # Pre-mint the battery run-dir HERE so the response can carry the exact path
    # (the worker records into it). None when battery recording is disabled — the
    # runner then mints a per-job run. Best-effort: a mint failure never blocks
    # the enqueue.
    battery_dir = None
    try:
        battery_dir = _mint_tester_battery_dir(f"tester:{spec.category}:{secrets.token_hex(6)}")
    except Exception:  # noqa: BLE001 — telemetry must never break the enqueue
        logger.debug("tester: battery pre-mint failed (non-fatal)", exc_info=True)
    if battery_dir:
        spec = dataclasses.replace(spec, battery_dir=battery_dir)

    job_id = _video_enqueue("studio_tester", spec)
    return jsonify({
        "job_id": job_id,
        "battery_run_dir": battery_dir,
        "category": spec.category,
        "kind": _TESTER_CATEGORY_KIND.get(spec.category),
        "poll": f"/video/jobs/{job_id}",
    }), 200


# --------------------------------------------------------------------------- #
# 2d''') POST /video/studio/movie — a STUDIO MOVIE (an ordered strip of studio
#        clips conjoined at splice points, like an NLE row) via the studio spine.
#        Mirrors /video/studio/i2v: parse body -> build the validated
#        StudioMovieSpec (a take-tree of segment NODES) -> media_bus.enqueue ->
#        {job_id}. The job runs through runners/studio_movie.py, which renders each
#        segment INLINE through the same produce_clip spine and stitches the strip
#        (non-destructive trims honored at concat). Query it exactly like any other
#        media job: GET /video/jobs/<job_id>.
#
#        Ergonomics: a goal's segment_id auto-fills to "seg_NN" and its
#        parent_segment_id auto-chains to the previous node when omitted, so a
#        minimal body ({"goals":[{"prompt":...},{"prompt":..., "branch_frame":10}]})
#        forms a valid LINEAR chain. An explicitly-supplied id/parent is passed
#        through and re-validated by make_studio_movie (a broken chain -> 400).
# --------------------------------------------------------------------------- #
@video_bp.route("/video/studio/movie", methods=["POST"])
def video_studio_movie():
    body = request.get_json(silent=True) or {}
    # resolution nested ({"resolution": {...}}) or flat top-level keys (nested wins),
    # mirroring the i2v route. Sane studio defaults so a minimal POST still renders.
    res = body.get("resolution") if isinstance(body.get("resolution"), dict) else {}
    width = res.get("width", body.get("width"))
    height = res.get("height", body.get("height"))
    fps = res.get("fps", body.get("fps"))
    # 832x480 (R_480P), not 512x512. A square 512 default was a GUARANTEED FAIL for
    # two whole capabilities: it is outside EVERY id-capable and EVERY v2v-capable
    # model's envelope on this fleet (Wan-VACE maxes at 480p), so an id_lock request
    # at the default either refused with no_capable_model or — worse — fell through
    # to the i2v runner, which does not read reference_images, and silently rendered
    # the wrong person. A default that cannot succeed is a failure promise
    # (defaults-are-promises); this one is the geometry the studio actually serves.
    width = 832 if width is None else width
    height = 480 if height is None else height
    fps = 24 if fps is None else fps

    goals_in = body.get("goals")
    if not isinstance(goals_in, list) or not goals_in:
        return jsonify({"error": "goals must be a non-empty list of segment nodes"}), 400

    # Build the take-tree nodes, auto-filling segment_id + the linear parent chain
    # when omitted (an explicit value is passed through + re-validated in the factory).
    goals = []
    prev_id = None
    for i, g in enumerate(goals_in):
        if not isinstance(g, dict):
            return jsonify({"error": f"goals[{i}] must be an object"}), 400
        seg_id = g.get("segment_id") or f"seg_{i:02d}"
        if i == 0:
            parent = g.get("parent_segment_id")  # None expected; factory enforces root
        else:
            parent = g.get("parent_segment_id", prev_id)
        goals.append(StudioMovieGoal(
            segment_id=seg_id,
            prompt=g.get("prompt"),
            parent_segment_id=parent,
            branch_frame=g.get("branch_frame"),
            negative=g.get("negative"),
            seed=g.get("seed"),
            model_id=g.get("model_id"),
            steps=g.get("steps"),
            cfg=g.get("cfg"),
            # JOINT MODE + context frames (VACE-extend splice motion-carry). joint_mode
            # defaults to "still" (absent/null/"" -> "still"); a bad non-empty value / a
            # root vace_extend / an out-of-range context_frames is a clean 400 via
            # make_studio_movie's validation below.
            joint_mode=(g.get("joint_mode") or "still"),
            context_frames=g.get("context_frames"),
            # CLIP LENGTH (2026-08-13): optional per-segment frames; None ->
            # the movie-level default below -> the bound model's default.
            frames=g.get("frames"),
        ))
        prev_id = seg_id

    # SEGMENT 0 conditioning still (optional): a jail-resolved + image-classified
    # start_image path, OR a start_image_asset_id resolved via the media catalog.
    # A jail-escaping / nonexistent / non-image target is a clean 4xx here rather
    # than a deferred runner failure. When absent, segment 0 renders t2v.
    start_ref = None
    start_image = body.get("start_image")
    start_asset_id = body.get("start_image_asset_id")
    if start_image is None and start_asset_id:
        start_image = _resolve_asset_uri(start_asset_id)
        if start_image is None:
            return jsonify(
                {"error": f"start_image_asset_id not found in catalog: {start_asset_id!r}"}), 404
    if start_image is not None:
        rp = _jail_resolve(start_image)
        if rp is None:
            return jsonify({"error": "start_image outside storage jail"}), 400
        if not os.path.isfile(rp):
            return jsonify({"error": "start_image not found"}), 404
        try:
            start_ref = media_store.ingest(rp, kind_hint="image", owner=_caller_username())
        except Exception as exc:  # unreadable / not a real image = bad input
            return jsonify(
                {"error": f"start_image is not a readable media file: {exc}"}), 400
        if start_ref.kind != "image":
            return jsonify(
                {"error": f"start_image is not an image (classified as {start_ref.kind})"}), 400

    # IDENTITY LOCK (id_lock): movie-level subject reference image(s). When present the movie
    # is an IDENTITY MOVIE — the runner renders EVERY segment capability id_lock (Wan-VACE
    # reference-to-video) so the locked subject carries across scene changes. Accept EITHER a
    # list of jailed "reference_images" paths OR "reference_image_asset_ids" (resolved via the
    # media catalog). Each is jail-resolved + ffprobe/PIL-classified as an IMAGE (a non-image /
    # jail-escaping / missing target is a clean 4xx here, not a deferred runner failure). Up to
    # 4 (all consumed by diffusers 0.39). Mirrors /video/studio/i2v's reference handling.
    # UNIFIED IDENTITY (2026-07-12): accept identity_profile:<slug> (canonical) as an
    # alternative to raw reference_images / reference_image_asset_ids. Own bounded block.
    reference_images_in, _prof_err = _reference_images_from_body(body)
    if _prof_err is not None:
        _pl, _st = _prof_err
        return jsonify(_pl), _st
    # NOTE (2026-07-16): an 8-view canonical is already narrowed to the RENDER cap (4) by
    # _reference_images_from_body, so the >4 check below only ever guards a raw caller list.
    reference_asset_ids = body.get("reference_image_asset_ids")
    if reference_images_in is None and isinstance(reference_asset_ids, list):
        reference_images_in = []
        for aid in reference_asset_ids:
            uri = _resolve_asset_uri(aid)
            if uri is None:
                return jsonify(
                    {"error": f"reference_image_asset_id not found in catalog: {aid!r}"}), 404
            reference_images_in.append(uri)
    resolved_refs: list = []
    if reference_images_in is not None:
        if not isinstance(reference_images_in, list):
            return jsonify({"error": "reference_images must be a list of paths"}), 400
        if len(reference_images_in) > 4:
            return jsonify({"error": "at most 4 reference_images are accepted"}), 400
        # jail-resolve + image-classify -> media_store URIs (the ONE ingest path; see
        # _ingest_image_references). A bad path is a clean 4xx here, not a runner failure.
        resolved_refs, _ref_err = _ingest_image_references(reference_images_in)
        if _ref_err is not None:
            _pl, _st = _ref_err
            return jsonify(_pl), _st

    # PER-GOAL VIEW (IDENTITY-3D-CONTINUITY-PLAN.md S2-movie + S3): let EACH segment of an
    # identity movie condition on a DIFFERENT turntable VIEW of the SAME identity, so a
    # ``cut`` into a new scene ("beach" -> "volleyball") holds the character while turning
    # the camera per shot. The MOVIE-LEVEL DNA (``resolved_refs``) stays canonical (resolved
    # above, unchanged) — this only OVERRIDES a goal's own reference set when a view resolves
    # to ring frames; a goal with no view is left untouched (its schema ``reference_images``
    # stays None -> the runner inherits the movie-level set, byte-identical to today).
    #
    # Per goal the view is chosen by precedence:
    #   1. explicit ``view`` on the goal (semantic name or {azimuth_deg})  -> "explicit" ;
    #   2. else DERIVED from the goal's prompt text (S3 keyword pass)       -> "derived"  ;
    #   3. else no view                                                     -> "none" (inherit).
    # DEGRADE-TO-INHERIT (defaults-are-promises): only an INVALID *explicit* view is a clean
    # 400 (naming the segment). A valid view on an identity with NO turntable ring, or on a
    # NON-identity movie (no slug / no movie-level refs), simply inherits — never an error.
    # The per-goal bank paths go through the SAME _ingest_image_references media_store path
    # the movie-level refs use (the renderer consumes URIs, not raw paths).
    slug = body.get("identity_profile")
    is_identity_movie = bool(resolved_refs) and isinstance(slug, str) and bool(slug.strip())
    if is_identity_movie:
        profile = identity_profiles.get_profile(slug.strip())  # re-fetch; validated above
        # bank_views resolves the version itself (id-or-name, else active), so mirror the
        # movie-level precedence by handing it the body's identity_version straight through
        # (already validated by _reference_images_from_body). Compute the ring ONCE.
        bank = identity_profiles.bank_views(
            profile, version_id=body.get("identity_version")) if profile else []
        for gi, (g_in, goal) in enumerate(zip(goals_in, goals)):
            view_hint = g_in.get("view")
            view_source = "none"
            azimuth_deg = None
            if view_hint is not None:
                azimuth_deg, view_err = identity_profiles.azimuth_for_view(view_hint)
                if view_err is not None:
                    return jsonify(
                        {"error": f"goal {goal.segment_id!r}: {view_err}"}), 400
                view_source = "explicit"
            else:
                derived = shot_intent.derive_view_from_prompt(goal.prompt)
                if derived is not None:
                    azimuth_deg, _ = identity_profiles.azimuth_for_view(derived)
                    view_source = "derived"
            if azimuth_deg is None:
                logger.info("movie per-goal view: segment=%s view_source=none", goal.segment_id)
                continue
            if not bank:
                # identity has no turntable ring -> inherit the movie-level DNA (never an error)
                logger.info("movie per-goal view: segment=%s view_source=%s no-ring -> inherit",
                            goal.segment_id, view_source)
                continue
            # K is the RENDER cap (4), not the canonical/storage cap: these frames go
            # straight into ONE segment's id_lock conditioning. Was MAX_CANONICAL_IMAGES
            # back when both numbers were 4; pinned to MAX_RENDER_REFS on 2026-07-16 when
            # canonical widened to 8, so a per-goal view still conditions on 4 frames.
            picked = identity_profiles.nearest_bank_views(
                bank, azimuth_deg, identity_profiles.MAX_RENDER_REFS)
            goal_uris, _g_err = _ingest_image_references([b["path"] for b in picked])
            if _g_err is not None:
                _pl, _st = _g_err
                return jsonify(_pl), _st
            goals[gi] = dataclasses.replace(goal, reference_images=tuple(goal_uris))
            logger.info("movie per-goal view: segment=%s view_source=%s azimuth=%.1f n_refs=%d",
                        goal.segment_id, view_source, azimuth_deg, len(goal_uris))

    try:
        spec = make_studio_movie(
            goals=tuple(goals),
            width=width,
            height=height,
            fps=fps,
            # AUTOFIT: blank/absent/null -> None (each segment sizes to the serving worker's
            # free VRAM at render time); an explicit number is the manual override.
            vram_budget_gb=_autofit_vram_budget(body.get("vram_budget_gb")),
            seed=body.get("seed", 0),
            # C-prompt: "negative_prompt" (canonical) with back-compat to "negative".
            negative=body.get("negative_prompt", body.get("negative")),
            model_id=body.get("model_id"),
            steps=body.get("steps"),
            cfg=body.get("cfg"),
            # SESSION NAME (k91): the operator's own name for this work. OPTIONAL —
            # absent/blank -> None, and the composer warns-but-allows on submit. Kept
            # distinct from ``project`` because ``project`` slugifies into the on-disk
            # movie DIR, so it cannot be renamed after a segment has landed; the title
            # can. "name" is accepted as an alias because that is what the field is
            # labelled in the UI.
            title=body.get("title", body.get("name")),
            project=body.get("project"),
            out_root=body.get("out_root"),
            start_image=start_ref,
            time_budget_s=body.get("time_budget_s"),
            # Movie-level default trailing-frame count for vace_extend splices (a node may
            # override via its own context_frames). Absent -> the schema default (8).
            context_frames=body.get("context_frames", _DEFAULT_MOVIE_CONTEXT_FRAMES),
            # CLIP LENGTH movie-level default (2026-08-13): every segment a goal
            # doesn't override renders this many frames (clamped + 4k+1-snapped
            # by the spine); absent -> each bound model's default (81 real).
            frames=body.get("frames"),
            # IDENTITY LOCK: the validated reference image uris (jail-resolved + image-classified
            # above). Non-empty -> an identity movie (every segment renders id_lock).
            reference_images=tuple(resolved_refs),
            # k120 slice 2 — PRODUCER CONTINUITY REFRESH: rewrite each non-root
            # segment's prompt from what the previous segment ACTUALLY rendered.
            continuity_refresh=bool(body.get("continuity_refresh", False)),
            continuity_model=(body.get("continuity_model") or None),
        )
    except (ValueError, TypeError) as exc:  # bad node / geometry / chain = 400
        return jsonify({"error": str(exc)}), 400

    # ----------------------------------------------------------------------- #
    # TAKE-TREE CAPABILITY PREFLIGHT (k58) — refuse HERE, over the WHOLE movie,
    # before a job_id exists.
    #
    # THE FAILURE THIS DELETES (operator movie eb9dee56, 2026-07-31): a 2-segment
    # movie pinned ``wan2.1-t2v-1.3b``; segment 0 (t2v) rendered on the 3090, then
    # segment 1 (a "still" splice -> capability i2v) died mid-movie with
    # ``pinned_model_unavailable`` — a refusal whose own text listed what WAS
    # available. Real GPU minutes bought a failure that was knowable at submit: a
    # movie's per-segment capabilities are a pure function of this spec.
    #
    # So the whole tree is walked now (``studio.movie_plan.preflight_movie``, the same
    # module the runner derives each segment's capability from — it cannot drift from
    # what will actually be asked for), and the refusal is PER SEGMENT: which segment,
    # which capability it needs, what the named model does serve, and what IS available.
    # A movie that passes here can no longer fail for a capability reason — only for a
    # runtime one (OOM, weights, a lost worker).
    #
    # NOT REFUSED: a movie-level pin that serves only SOME segments. That is a legal,
    # useful request (ruling 1) — the pin binds the segments it serves and the others
    # resolve their own capable model, attributed per segment in movie.json and the
    # stage log. Only an EXPLICIT per-segment ``model_id`` that cannot serve its own
    # segment is a refusal, because an explicit choice is never substituted.
    # ----------------------------------------------------------------------- #
    from hugpy_video.intel.studio.movie_plan import preflight_movie
    _seg_problems = preflight_movie(spec)
    if _seg_problems:
        _first = _seg_problems[0]
        return jsonify({
            "error": (f"this movie cannot be rendered as asked: "
                      f"{_first.get('detail')}"
                      + (f" (and {len(_seg_problems) - 1} more segment(s))"
                         if len(_seg_problems) > 1 else "")),
            "code": "movie_capability_preflight_failed",
            "segments": _seg_problems,
        }), 400

    # COORDINATION PREFLIGHT (k121) — the words-vs-knobs half of the same idea.
    # The capability preflight above proves the movie CAN render; this one asks
    # whether it renders the film the prompts describe. Same submit-time timing,
    # same per-segment refusal shape, and the full review rides the response
    # either way so "reviewed and fine" is visible, not merely implied.
    from hugpy_server.app.routes.video_coordination import (
        COORDINATION_KEY as _COORD_KEY,
        movie_coordination,
    )
    _coord_refusal, _coord_report = movie_coordination(spec, body)
    if _coord_refusal is not None:
        return jsonify(_coord_refusal), 400

    job_id = _video_enqueue("generate_studio_movie", spec)

    # SESSION PERSISTENCE (k91). Stamp the submitted spec into the movie dir the
    # instant the job exists — BEFORE any runner claims it — so this movie is
    # resumable from the moment it is asked for. That window is not theoretical: the
    # run this feature exists for (a 14-segment Cinema movie, 2026-08-06) died during
    # SEGMENT 0, which is exactly the state that has no ``movie.json`` yet. Written
    # only from here so the runner's own rewrite (which repeats it, deliberately, for
    # enqueue paths that are not this route) is a refresh and never the first record.
    # Best-effort by construction — ``write_movie_spec`` swallows its own IO errors,
    # and a job that is already enqueued must never 500 over a sidecar.
    try:
        from hugpy_video.intel.runners.studio_movie import movie_root_for, write_movie_spec
        write_movie_spec(movie_root_for(spec, job_id), spec, job_id)
    except Exception:  # noqa: BLE001 — the job is real either way
        logger.warning("studio movie %s: submit-time spec.json write failed", job_id,
                       exc_info=True)

    return jsonify({"job_id": job_id, _COORD_KEY: _coord_report}), 200


# --------------------------------------------------------------------------- #
# 2d''''') MOVIE SESSIONS (k91) — a movie DIR is a resumable SESSION, and these three
#     routes are the whole of its lifecycle surface:
#
#       GET  /video/studio/movies                 — every session + what it has rendered
#       POST /video/studio/movie/<movie_id>/pause  — stop the live job, keep the work
#       POST /video/studio/movie/<movie_id>/resume — re-enqueue the persisted spec
#
#     WHY THESE EXIST. A studio movie is ONE bus job that can run for hours, and until
#     now the only handle on it was that job id. When the job ended — cancelled, timed
#     out, reaped, or simply lost with the browser tab — the rendered segments stayed on
#     disk and became unreachable: real GPU hours with no way to see or continue them.
#     The two sidecars ``runners/studio_movie.py`` writes (``spec.json`` = what was
#     ASKED for, ``movie.json`` = what has RENDERED) make the DIR the durable identity
#     instead of the job, and these routes read/act on that.
#
#     ID = THE DIR LEAF. A movie's id is the name of its directory under the studio
#     movies root — the slugified ``project`` when the caller named one, else the
#     originating job id (``movie_root_for``, the single definition both sides share).
#     It therefore stays stable across a resume, which mints a NEW job id; keying these
#     routes on the job id would have made every resume a new, unrelated session.
#
#     SCOPE: only movies under the DEFAULT studio-movies root are listed/actionable. A
#     movie submitted with an explicit ``out_root`` lives outside it by the caller's own
#     choice and is not a session here — an honest 404 rather than a filesystem walk of
#     wherever a body pointed.
#
#     AUTH: the blanket /video gate (video_auth) already covers all three. No extra
#     operator check, matching the neighbouring surface: POST /video/studio/movie
#     (enqueue) and POST /video/jobs/<id>/cancel are on the same footing, and resume is
#     literally a re-enqueue of a spec this surface already accepted. The routes that DO
#     add ``operator_authenticated`` (/to-editor, /mlt/render) do so because they write
#     into the operator's real editing tree; these do not.
# --------------------------------------------------------------------------- #
def _movie_media_url(path):
    """A ``/video/media`` fetch url for an absolute media path, percent-encoded.

    The path is a QUERY VALUE, so it is quoted rather than interpolated raw: a movie
    dir's leaf is a caller-supplied project name (slugified, but slugs are not
    guaranteed url-safe forever), and an unencoded ``&`` or space would silently
    truncate the handle and turn a playable clip into a 404. ``safe="/"`` keeps the
    separators readable, which matters when an operator is reading these out of a
    session listing."""
    from urllib.parse import quote
    return "/video/media?handle=" + quote(str(path), safe="/")


def _movie_session_dir(movie_id):
    """Resolve a movie SESSION id to its absolute dir under the studio-movies root, or
    None when it is not a real session (ill-typed, path-escaping, or absent).

    The single jail seam for all three session routes: ``movie_id`` is a DIR LEAF, so
    anything carrying a separator or a parent ref is refused BEFORE it touches the
    filesystem, and the realpath is then re-checked under the root (a symlinked leaf
    cannot escape). Returning None for "does not exist" as well as "not allowed" is
    deliberate — the caller answers both with the same 404, so this never becomes a
    probe for what lives outside the tree."""
    from hugpy_video.intel.runners.studio_movie import STUDIO_MOVIE_ROOT
    if not movie_id or not isinstance(movie_id, str):
        return None
    if os.sep in movie_id or "/" in movie_id or movie_id in (".", "..") or "\0" in movie_id:
        return None
    path = os.path.join(STUDIO_MOVIE_ROOT, movie_id)
    if not _is_within(path, STUDIO_MOVIE_ROOT) or not os.path.isdir(path):
        return None
    return os.path.realpath(path)


# The bus statuses that mean "a job is still going to do something". Named once and
# shared by the single-job probe and the batch snapshot, so the two can never answer
# "is this session live" differently.
_MOVIE_INFLIGHT_STATES = ("queued", "claimed", "running", "cancelling")


def _movie_live_status(job_id):
    """The bus status of a session's CURRENT job (``None`` when it has none / is
    unknown), plus whether that status is IN-FLIGHT. Used to reconcile the manifest's
    coarse status against reality: the manifest is written BY the runner, so a run that
    died without a terminal write leaves ``status: "running"`` behind forever, and a
    session list that repeats that claim is the same lie about live work this slice
    exists to end.

    The SINGLE-job probe, used by pause/resume. The session LISTING resolves a whole
    page in one query instead — see ``_movie_job_statuses``."""
    if not job_id or not isinstance(job_id, str):
        return None, False
    try:
        view = media_bus.get(job_id)
    except Exception:  # noqa: BLE001 — a session route never 5xxes on a status read
        return None, False
    status = view.get("status") if isinstance(view, dict) else None
    return status, status in _MOVIE_INFLIGHT_STATES


def _read_movie_sidecars(movie_dir):
    """``(manifest, envelope)`` for a movie dir — the two sidecars, read ONCE. Both are
    already total (a missing/corrupt file reads as None), so a dir mid-write or one that
    predates a sidecar simply contributes less information, never an error."""
    from hugpy_video.intel.runners.studio_movie import read_movie_manifest, read_movie_spec
    return read_movie_manifest(movie_dir) or {}, read_movie_spec(movie_dir)


def _movie_session_job_id(manifest, envelope):
    """The session's CURRENT bus job id.

    The spec envelope's wins: it is rewritten on every submit AND every resume, while
    the manifest's is only as fresh as the last segment the runner finished — so after a
    resume the manifest still names the job that was paused. Falls back to the manifest
    for a dir whose spec sidecar predates resumable sessions."""
    job_id = envelope.get("job_id") if isinstance(envelope, dict) else None
    if not job_id and isinstance(manifest, dict):
        job_id = manifest.get("job_id")
    return job_id


def _movie_job_statuses(job_ids):
    """``{job_id: status}`` for a whole page of sessions in ONE read-only query.

    A per-row ``media_bus.get`` would be the exact shape k57 had to undo on
    ``GET /video/jobs``: a listing that opens one connection per row serializes behind
    whatever the renderers are doing to that DB, and the panel this feeds polls every
    few seconds. Unknown ids are simply absent from the map (the caller reads a missing
    entry as "no live job"), and any sqlite trouble yields an EMPTY map rather than an
    error — a session list that loses its status reconciliation is degraded, not
    broken."""
    ids = [j for j in job_ids if isinstance(j, str) and j]
    if not ids:
        return {}
    import sqlite3
    try:
        media_bus._ensure_db()
    except Exception:  # noqa: BLE001 — mirrors the clips listing's pre-read migration
        pass
    try:
        conn = sqlite3.connect(f"file:{media_bus.DB_PATH}?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            rows = conn.execute(
                "SELECT job_id, status FROM media_jobs WHERE job_id IN "
                "(" + ",".join("?" * len(ids)) + ")", ids).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return {}
    return {jid: status for jid, status in rows}


def _movie_job_visibilities(job_ids):
    """``{job_id: (owner, private)}`` for a whole page of sessions in ONE read-only
    query — the t172 visibility counterpart of ``_movie_job_statuses`` (same batch
    discipline: never one read per row on this few-second poll). A job_id ABSENT from
    the map (no bus row — reaped / expired / a session that outlived its job) is read
    by the caller as UNKNOWN -> PUBLIC/visible, which PRESERVES both the accepted
    public movie listing (board t172) and the "a session outlives the job that made
    it" invariant. sqlite trouble yields an EMPTY map (every movie reads as public/
    visible), degraded not broken."""
    ids = [j for j in job_ids if isinstance(j, str) and j]
    if not ids:
        return {}
    import sqlite3
    try:
        media_bus._ensure_db()
    except Exception:  # noqa: BLE001 — mirrors _movie_job_statuses' pre-read migration
        pass
    try:
        conn = sqlite3.connect(f"file:{media_bus.DB_PATH}?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            rows = conn.execute(
                "SELECT job_id, owner, private FROM media_jobs WHERE job_id IN "
                "(" + ",".join("?" * len(ids)) + ")", ids).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return {}
    return {jid: (owner, bool(private)) for jid, owner, private in rows}


def _movie_session_summary(movie_id, movie_dir, manifest, envelope, job_statuses=None):
    """Project ONE movie dir (plus its already-read sidecars) into the session summary
    the console lists.

    Total by construction — every read below degrades to a null/empty rather than
    raising, because this runs once per dir on a root with hundreds of them and ONE
    corrupt sidecar must not blank the whole list. The sidecars are passed IN rather
    than read here so the caller can read each dir exactly once and then resolve every
    bus status in a single query (see ``_movie_job_statuses``).

    The per-segment rows carry a ``media`` url for every clip that is actually ON DISK
    RIGHT NOW, which is what makes completed segments watchable AS THEY LAND rather than
    only after assembly: the runner rewrites ``movie.json`` after every single segment,
    so a poll of this endpoint mid-render already sees segment N's clip while segment
    N+1 is still denoising. The url is ``/video/media?handle=`` — the existing jailed
    byte-serving route (a segment clip lives under the media-store root, so it is inside
    that route's jail and carries a .mp4 name for the mime). No new serving route: one
    jail, one place to get it wrong."""
    from hugpy_video.intel.runners.studio_movie import (
        MOVIE_STATUS_DONE,
        MOVIE_STATUS_PARTIAL,
        MOVIE_STATUS_PAUSED,
        MOVIE_STATUS_RUNNING,
    )
    manifest = manifest if isinstance(manifest, dict) else {}
    spec_d = envelope.get("spec") if isinstance(envelope, dict) else None
    spec_d = spec_d if isinstance(spec_d, dict) else {}

    job_id = _movie_session_job_id(manifest, envelope)
    if job_statuses is None:
        job_status, live = _movie_live_status(job_id)
    else:
        job_status = job_statuses.get(job_id)
        live = job_status in _MOVIE_INFLIGHT_STATES

    # STATUS — the manifest's own claim, reconciled against the bus. Order matters:
    #   * "done" and "paused" WIN over anything the bus says. Each is written by exactly
    #     one writer making a claim only it can make — the runner assembles, pause
    #     parks — and neither is un-made by a bus row. A finished movie whose (stale or
    #     re-queued) row still reads in-flight is finished; a paused one whose
    #     cooperative cancel is still draining as "cancelling" is paused, because that
    #     drain IS the pause happening.
    #   * otherwise a job actually in flight IS running.
    #   * otherwise a "running" manifest with no live job means the run STOPPED without
    #     a terminal write -> "partial": some segments, no assembly. Repeating the
    #     manifest's "running" there would be the same lie about live work this whole
    #     slice exists to end.
    raw_status = manifest.get("status")
    if raw_status in (MOVIE_STATUS_DONE, MOVIE_STATUS_PAUSED):
        status = raw_status
    elif live:
        status = MOVIE_STATUS_RUNNING
    elif raw_status == MOVIE_STATUS_RUNNING or raw_status is None:
        status = MOVIE_STATUS_PARTIAL
    else:
        status = raw_status

    segs_raw = manifest.get("segments")
    segs_raw = segs_raw if isinstance(segs_raw, list) else []
    goals = spec_d.get("goals")
    goals = goals if isinstance(goals, list) else []

    segments = []
    for i, s in enumerate(segs_raw):
        if not isinstance(s, dict):
            continue
        clip_path = s.get("clip_path")
        available = bool(isinstance(clip_path, str) and clip_path
                         and os.path.isfile(clip_path))
        segments.append({
            "index": s.get("index", i),
            "segment_id": s.get("segment_id"),
            "prompt": s.get("prompt"),
            "status": s.get("status"),
            "resumed": s.get("resumed"),
            "frames": s.get("frames"),
            "duration_s": s.get("duration_s"),
            "error": s.get("error"),
            # CLIP AVAILABILITY — the honest "can I watch this right now" bit. False
            # for a segment that failed, and for one whose record exists but whose
            # bytes are gone; the url is omitted entirely in that case rather than
            # handed over as a link that 404s.
            "clip_available": available,
            "media": (_movie_media_url(clip_path) if available else None),
        })

    # Segments the spec ASKS for that have no record yet (the not-yet-started tail).
    # Listed as "pending" so the UI can render the full strip — a 14-segment movie
    # showing 2 rows mid-render reads as a 2-segment movie.
    for i in range(len(segments), len(goals)):
        g = goals[i] if isinstance(goals[i], dict) else {}
        segments.append({
            "index": i, "segment_id": g.get("segment_id"), "prompt": g.get("prompt"),
            "status": "pending", "resumed": None, "frames": None, "duration_s": None,
            "error": None, "clip_available": False, "media": None,
        })

    total = manifest.get("segments_total")
    if not isinstance(total, int) or total <= 0:
        total = len(goals) or len(segments)
    completed = manifest.get("segments_completed")
    if not isinstance(completed, int):
        completed = len([s for s in segments if s["status"] in ("done", "resumed")])

    # The assembled movie, when one has been stitched AND still exists on disk.
    assembly = manifest.get("assembly") if isinstance(manifest.get("assembly"), dict) else {}
    movie_media = None
    if assembly.get("movie"):
        movie_path = os.path.join(movie_dir, str(assembly["movie"]))
        if os.path.isfile(movie_path):
            movie_media = _movie_media_url(movie_path)

    return {
        "movie_id": movie_id,
        # TITLE: the manifest's (written per segment) else the spec's — a movie paused
        # before its first segment has a spec but no manifest, and it must still show
        # its name. None means UNNAMED, which the composer warns about.
        "title": manifest.get("title") or spec_d.get("title"),
        "project": manifest.get("project") or spec_d.get("project"),
        "job_id": job_id,
        "job_status": job_status,
        "status": status,
        "segments_completed": completed,
        "segments_total": total,
        "segments": segments,
        # RESUMABLE = there is a persisted spec to re-enqueue and the work is neither
        # finished nor already in flight. The UI's resume affordance keys on this, so
        # it is computed here rather than re-derived per client.
        "resumable": bool(spec_d) and status not in (MOVIE_STATUS_DONE,
                                                     MOVIE_STATUS_RUNNING),
        "updated": manifest.get("updated_at"),
        "width": manifest.get("width") or spec_d.get("width"),
        "height": manifest.get("height") or spec_d.get("height"),
        "fps": manifest.get("fps") or spec_d.get("fps"),
        "id_lock": bool(manifest.get("id_lock") or spec_d.get("reference_images")),
        "movie": movie_media,
    }


@video_bp.route("/video/studio/movies", methods=["GET"])
def video_studio_movies():
    """Every movie SESSION on the studio-movies root, newest activity first.

    A FILESYSTEM walk, not a bus query, and deliberately so: the bus row is the thing
    that goes away (cancelled, reaped, expired from the /llm/jobs retention window)
    while the rendered segments are the thing that persists. Listing from the dirs is
    what makes a session outlive the job that made it — the entire point.
    """
    from hugpy_video.intel.runners.studio_movie import STUDIO_MOVIE_ROOT

    try:
        limit = int(request.args.get("limit", 100))
    except (TypeError, ValueError):
        limit = 100
    limit = max(1, min(limit, 500))

    try:
        entries = [e for e in os.scandir(STUDIO_MOVIE_ROOT) if e.is_dir()]
    except OSError:
        # No studio-movies root yet (a box that has never rendered a movie) is an
        # EMPTY list, not an error — same posture as the clips listing.
        entries = []

    # Order by dir mtime BEFORE reading any sidecar, so a root with hundreds of dirs
    # only pays the json reads for the page it returns. mtime moves whenever the runner
    # rewrites movie.json (i.e. after every segment), so it tracks real activity.
    def _mtime(e):
        try:
            return e.stat().st_mtime
        except OSError:
            return 0.0

    entries.sort(key=_mtime, reverse=True)

    # Read each page dir's sidecars ONCE, then resolve every bus status in ONE query.
    # The per-row alternative is the shape k57 had to undo on GET /video/jobs: a
    # listing that hits the DB once per row serializes behind live renderers, and this
    # is a panel that polls every few seconds.
    sessions = []
    for e in entries[:limit]:
        movie_dir = _movie_session_dir(e.name)
        if movie_dir is None:
            continue
        try:
            manifest, envelope = _read_movie_sidecars(movie_dir)
        except Exception:  # noqa: BLE001 — one bad dir never blanks the listing
            logger.debug("movie sidecar read failed for %s", e.name, exc_info=True)
            continue
        sessions.append((e.name, movie_dir, manifest, envelope))

    session_job_ids = [_movie_session_job_id(m, env) for _n, _d, m, env in sessions]
    statuses = _movie_job_statuses(session_job_ids)
    # t172 VISIBILITY: hide a PRIVATE movie from a caller who is neither its owner nor
    # the operator/admin tier. A session whose bus row is UNKNOWN (reaped / expired /
    # outlived its job) is treated as PUBLIC/visible — preserving both the accepted
    # public movie listing and the session-outlives-job invariant. Batched (one query).
    visibilities = _movie_job_visibilities(session_job_ids)

    movies = []
    for name, movie_dir, manifest, envelope in sessions:
        ov = visibilities.get(_movie_session_job_id(manifest, envelope))
        if ov is not None and not _may_view_media(ov[0], ov[1]):
            continue  # a PRIVATE movie this caller may not view — hidden (existence)
        try:
            movies.append(_movie_session_summary(
                name, movie_dir, manifest, envelope, statuses))
        except Exception:  # noqa: BLE001 — one bad dir never blanks the listing
            logger.debug("movie session summary failed for %s", name, exc_info=True)

    return jsonify({"movies": movies}), 200


@video_bp.route("/video/studio/movie/<movie_id>/pause", methods=["POST"])
def video_studio_movie_pause(movie_id):
    """Pause a movie session: cooperatively cancel its live job, then record PAUSED.

    Pause is cancel PLUS a promise. The cancel is the existing cooperative path
    (``media_bus.cancel`` — a queued job dies outright, a running one stops between
    segments and its in-flight worker render gets the relayed cancel), so nothing new
    can wedge here. The promise is the manifest write: without it a paused movie is
    indistinguishable from one the operator gave up on, and "which of these is waiting
    on me" is the question the session list has to answer.

    The status is written EVEN IF there was nothing to cancel (an already-dead job, a
    session whose runner exited hours ago). Refusing in that case would be pedantically
    correct and practically wrong — the operator's intent is "leave this parked", and a
    session that cannot be parked because its job already ended is precisely the case
    this exists for. ``cancelled`` in the response says which it was.
    """
    from hugpy_video.intel.runners.studio_movie import MOVIE_STATUS_PAUSED, mark_movie_status
    movie_dir = _movie_session_dir(movie_id)
    if movie_dir is None:
        return jsonify({"error": f"unknown movie session: {movie_id!r}"}), 404

    job_id = _movie_session_job_id(*_read_movie_sidecars(movie_dir))

    # t172 MUTATION gate: only the movie's owner or the operator/admin tier may pause
    # it — the SAME visibility_of + _owns_media + _deny_media triad POST /video/jobs/
    # <id>/cancel uses (a share guest / other member is refused: 403 for a public
    # movie, 404 for a private one). A reaped bus row (found=False) skips, exactly as
    # cancel does; a session with no live bus job is a manifest-only park to protect.
    if job_id:
        found, job_owner, job_private = media_bus.visibility_of(job_id)
        if found and not _owns_media(job_owner):
            return _deny_media(job_owner, job_private)

    cancelled = False
    job_status = None
    if job_id:
        try:
            outcome = media_bus.cancel(job_id)
            cancelled = bool(outcome.get("cancelled"))
            job_status = outcome.get("status")
        except Exception:  # noqa: BLE001 — a failed cancel still parks the session
            logger.warning("movie %s: cancel of job %s failed", movie_id, job_id,
                           exc_info=True)

    manifest = mark_movie_status(movie_dir, MOVIE_STATUS_PAUSED, job_id=job_id)
    if manifest is None:
        # The cancel already landed, so the WORK is stopped; only the record failed.
        # Say exactly that rather than implying nothing happened.
        return jsonify({
            "error": "the movie was stopped but its manifest could not be written; "
                     "it will list as partial rather than paused",
            "movie_id": movie_id, "job_id": job_id, "cancelled": cancelled,
        }), 500

    return jsonify({
        "ok": True,
        "movie_id": movie_id,
        "job_id": job_id,
        "status": MOVIE_STATUS_PAUSED,
        # False = there was no live job to stop (already terminal/unknown). The
        # session is parked either way.
        "cancelled": cancelled,
        "job_status": job_status,
        "segments_completed": manifest.get("segments_completed"),
        "segments_total": manifest.get("segments_total"),
    }), 200


@video_bp.route("/video/studio/movie/<movie_id>/resume", methods=["POST"])
def video_studio_movie_resume(movie_id):
    """Resume a movie session: re-enqueue its persisted spec as a NEW bus job.

    There is no checkpoint format and no partial-render restart. Resume re-enqueues the
    SAME spec, and ``produce_clip``'s content addressing does the rest: a segment whose
    inputs hash to a clip already on the shared store comes back ``resumed=True`` without
    touching a GPU, so the run walks the completed prefix in seconds and picks up at the
    first segment that never finished. THE CLIPS ARE THE CHECKPOINT — which is why this
    is a handful of lines and why it cannot desynchronize from what actually rendered.

    Re-validated on the way through: ``studio_movie_from_dict`` rebuilds the spec through
    the SAME validating factory the submit route uses, so a hand-edited or
    version-skewed sidecar is a clean 400, never a malformed spec on the bus.

    A session with a job already IN FLIGHT is a 409, not a second job. Two runners over
    one movie dir would race on ``movie.json`` and interleave their segment records; the
    honest answer is "pause it first".
    """
    from hugpy_video.intel.runners.studio_movie import (
        MOVIE_STATUS_RUNNING,
        mark_movie_status,
        movie_root_for,
        read_movie_spec,
        write_movie_spec,
    )
    from hugpy_video.intel.studio_movie_schema import studio_movie_from_dict

    movie_dir = _movie_session_dir(movie_id)
    if movie_dir is None:
        return jsonify({"error": f"unknown movie session: {movie_id!r}"}), 404

    envelope = read_movie_spec(movie_dir)
    if not isinstance(envelope, dict) or not isinstance(envelope.get("spec"), dict):
        # A dir from before spec.json existed, or an unreadable one. The segments are
        # still there and still watchable — only the re-enqueue is impossible, and the
        # message says why rather than pretending the session is gone.
        return jsonify({
            "error": "this movie has no persisted spec.json, so it cannot be "
                     "re-enqueued (it predates resumable sessions); its rendered "
                     "segments are still listed and playable",
            "movie_id": movie_id,
        }), 409

    prior_job_id = envelope.get("job_id")

    # t172 MUTATION gate: only the prior render's owner or the operator/admin tier
    # may resume it — same visibility_of + _owns_media + _deny_media triad as pause /
    # jobs cancel (403 for a foreign public movie, 404 for a foreign private one).
    # A reaped bus row (found=False) skips, matching the sibling cancel route.
    if prior_job_id:
        found, prior_owner, prior_private = media_bus.visibility_of(prior_job_id)
        if found and not _owns_media(prior_owner):
            return _deny_media(prior_owner, prior_private)

    _prior_status, live = _movie_live_status(prior_job_id)
    if live:
        return jsonify({
            "error": f"movie {movie_id!r} is already running as job {prior_job_id} — "
                     f"pause it before resuming",
            "movie_id": movie_id, "job_id": prior_job_id, "status": _prior_status,
        }), 409

    # PIN THE SESSION DIR before rebuilding. Without this the resumed spec resolves its
    # movie dir from the NEW job id (for an unnamed movie the leaf IS the job id), so
    # every resume would render into a FRESH directory — finding none of the completed
    # segments, re-rendering the whole movie, and reporting success while doing it. The
    # pin is what makes the re-enqueue a resume rather than a restart. Set on the DICT
    # so it goes through the validating factory below (session_id is checked there as a
    # bare directory leaf), never spliced onto a built spec.
    spec_d = dict(envelope["spec"])
    spec_d["session_id"] = movie_id
    try:
        spec = studio_movie_from_dict(spec_d)
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify({
            "error": f"the persisted spec for {movie_id!r} is not a valid movie spec: {exc}",
            "movie_id": movie_id,
        }), 400

    # WHERE WILL THIS ACTUALLY RENDER? Asked through ``movie_root_for`` — the ONE
    # definition the runner itself uses — rather than assuming ``movie_dir``, and asked
    # BEFORE the enqueue so a mismatch refuses instead of leaving a job that will render
    # into the wrong place. The probe id is arbitrary precisely BECAUSE the pin is set:
    # if the leaf still varied with the job id, this check would catch it.
    resumed_root = movie_root_for(spec, "resume-probe")
    if os.path.realpath(resumed_root) != movie_dir:
        # Unreachable given the pin; kept because the failure it guards is SILENT — a
        # full re-render that looks like a successful resume, visible only as an
        # unexplained GPU bill.
        logger.error("movie %s: resume would resolve to %s, not the session dir %s",
                     movie_id, resumed_root, movie_dir)
        return jsonify({
            "error": f"resume for {movie_id!r} would render into a different directory "
                     f"than the session's own; refusing rather than re-rendering it",
            "movie_id": movie_id,
        }), 500

    job_id = _video_enqueue("generate_studio_movie", spec)

    # Rewrite the sidecars with the NEW job id, so a later pause cancels the job that is
    # actually about to run rather than the terminal one it replaced.
    write_movie_spec(resumed_root, spec, job_id)
    mark_movie_status(resumed_root, MOVIE_STATUS_RUNNING, job_id=job_id,
                      title=spec.title, project=spec.project,
                      segments_total=len(spec.goals))

    return jsonify({
        "ok": True,
        "movie_id": movie_id,
        "job_id": job_id,
        "previous_job_id": prior_job_id,
        "status": MOVIE_STATUS_RUNNING,
        # Every segment already on the shared store will come back resumed=True; the
        # count is what the console shows as "N of M already rendered".
        "segments_total": len(spec.goals),
        "poll": f"/video/jobs/{job_id}",
    }), 200


# --------------------------------------------------------------------------- #
# 2d'''') POST /video/mlt/render — headless Kdenlive/MLT render (k22).
#     The operator authors a project in Kdenlive on Windows against the Samba studio
#     share, saves the .kdenlive into the WRITABLE edits/ subtree, and this route
#     enqueues an ``mlt_render`` media_bus job that path-maps the project + renders it
#     server-side with melt, writing the output back under edits/renders/. Query it
#     exactly like any other media job: GET /video/jobs/<job_id>.
#
#     CONSOLE-OPERATOR ONLY: the blanket /video gate already admitted an operator OR a
#     share-link guest, but this WRITES INTO the operator's real editing tree (and reads
#     an arbitrary project file), so — like /to-editor — the body re-checks
#     operator_authenticated() and 403s a share guest. Defense in depth.
#
#     JAIL: the project MUST live under the studio tree (the runner re-checks); the output
#     is always resolved under edits/renders/ by the runner. A path outside the jail 400s
#     here for a fast, honest rejection.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/mlt/render", methods=["POST"])
def video_mlt_render():
    from hugpy_server.app.operator_auth import operator_authenticated
    if not operator_authenticated():
        return jsonify({"error": "operator session required"}), 403

    from hugpy_video.intel.mlt_render_schema import make_mlt_render
    from hugpy_video.intel.runners.mlt_render import STUDIO_ROOT

    body = request.get_json(silent=True) or {}
    project_path = body.get("project_path")
    if not isinstance(project_path, str) or not project_path.strip():
        return jsonify({"error": "project_path is required (absolute path to a "
                        ".kdenlive/.mlt project under the studio share)"}), 400

    # Jail the project under the studio tree (fast 400; the runner re-checks authoritatively).
    rp = os.path.realpath(project_path.strip())
    if not _is_within(rp, STUDIO_ROOT):
        return jsonify({"error": "project_path is outside the studio jail"}), 400
    if not os.path.isfile(rp):
        return jsonify({"error": "project_path not found"}), 404

    try:
        spec = make_mlt_render(
            project_path=rp,
            output_rel=body.get("output_rel"),
            width=body.get("width"),
            height=body.get("height"),
            fps=body.get("fps"),
            profile=body.get("profile"),
            vcodec=body.get("vcodec", "libx264"),
            acodec=body.get("acodec", "aac"),
            container=body.get("container", "mp4"),
            vb=body.get("vb"),
            drive_letter=body.get("drive_letter"),
        )
    except (ValueError, TypeError) as exc:  # structural spec error = 400
        return jsonify({"error": str(exc)}), 400
    job_id = _video_enqueue("mlt_render", spec)
    return jsonify({"job_id": job_id}), 200


# --------------------------------------------------------------------------- #
# 2e) GET /video/presets — curated "ideal default loads" for scene generation
# --------------------------------------------------------------------------- #
# Thin idiom (mirrors prompt_routes' GET /prompt/tasks): import the static
# registry, dump it as JSON. No side effects — a preset is just a named bundle
# of a model_key + coherence mode + per-frame defaults the UI pre-fills.
@video_bp.route("/video/presets", methods=["GET"])
def video_presets():
    from hugpy_video.intel.presets import available_presets
    return jsonify({"presets": [p.to_dict() for p in available_presets()]}), 200


# --------------------------------------------------------------------------- #
# 2e-bis) POST /video/prompt/assist — LLM-backed helper for the studio's prompt
#     input. Two modes:
#       detail    expand/enrich the caller's DRAFT image prompt into a richer,
#                 more vivid diffusion prompt (preserves the draft's subject
#                 and intent — "draft" is required).
#       generate  write a full, original image-generation prompt from
#                 scratch; "draft", if given, is used only as a loose theme.
#
#     Routes through the exact SAME internal chat plane as /chat/stream and
#     /prompt — managers.dispatch.execute_prompt via resolve() (see
#     ../functions/chat/streaming.py:94 execute_chat_stream and
#     ../routes/discord_routes.py:326 _generate_candidate for the two other
#     callers of this same one-shot pattern). On THIS central
#     (HUGPY_NO_LOCAL_SERVING=true, see managers/serve/policy.py) resolve()'s
#     DelegatingRunner sends the completion to a live GPU worker; it never
#     loads a model in-process here. No live worker / unreachable -> a clean
#     502 via the same _friendly_stream_error mapping /chat/stream uses on a
#     stream failure — never a 500, never a silent local load.
# --------------------------------------------------------------------------- #
# Default assist model. The operator asked for flux2-klein (2026-07-13), BUT on
# THIS dev central flux2 does not resolve for inference: execute_prompt (and even
# /v1/chat/completions) only know the 11 serve-configured models — the other ~100
# manifest/discovered models (flux2-klein, the HunyuanVideo rewriter, etc.) are
# catalog-only and 404 with "Unknown model_key". Defaulting to flux2 would make
# every default call fail, so per the defaults-are-promises doctrine the default
# is the best chat model that ACTUALLY resolves here: Qwen2.5-3B-Instruct-GGUF
# (verified 200 + good prompt enrichment). A caller can still pass any key via
# body["model"] — and once flux2 is serve-configured on the fleet this should
# switch back to it. keeper 2026-07-13 (flux2 non-resolution flagged to operator).
#
# SUPERSEDED 2026-07-27 (operator: "the flux2-uncensored should be the default for the
# text generation in video, this should have thinking turned off and routed through the
# computron"). The 2026-07-13 reasoning above was CORRECT WHEN WRITTEN and is now stale:
# flux2-klein was catalog-only then. Verified live today instead of assumed —
#   GET /llm/workers -> computron slot flux2-klein-9b-uncensored-text-encoder
#                       serving=True, vram=5,515,509,760 (5.14 GiB), ngl=27/36
#   POST /video/prompt/assist -> HTTP 200 in 8.5s, a clean 4-sentence prompt
# Weights have been on computron since 2026-07-18 (~53 GB, every quant; the q4_k_m that
# fits the 8 GiB 4060 is 5.03 GB) and computron is the ONLY worker holding it — ae carries
# the image-to-image variant — so "routed through computron" is a consequence of the
# fleet, not a routing rule anyone has to enforce.
#
# ⚠ IT IS A REASONING MODEL, so this default is only safe together with the no-think
# seam (utils/no_think.py). Without it the whole 200-token budget goes to <think> and
# the caller gets a monologue instead of a prompt (measured — see that module).
#
# ⚠ DO NOT "fix" this to Flux-Uncensored-V2. That row is ALSO tagged text-generation in
# the catalog but is a Flux IMAGE LoRA — the same mis-classification class as 41f908d.
_DEFAULT_PROMPT_ASSIST_MODEL = "flux2-klein-9b-uncensored-text-encoder"
# Reliability fallback (2026-08-05): the preferred model above is a flux2 text-
# encoder that gets evicted/reloaded as the fleet churns, so a live prompt-assist
# call can hit it mid-eviction and hard-502 (operator-reported "cross-origin"
# symptom that was really a 502). When the primary fails with a WORKER error, we
# retry ONCE with a small always-serve-configured chat model so prompt-generate
# keeps working through the churn. Only if the fallback ALSO fails do we 502.
# (2026-08-28: was the 3B, which the operator then blocked fleet-wide — a
# blocked fallback turned every primary hiccup into a hard refusal. 7B is the
# smallest unblocked chain member.)
_PROMPT_ASSIST_FALLBACK_MODEL = "Qwen2.5-7B-Instruct-GGUF"

_PROMPT_ASSIST_SYSTEM = (
    "You are an expert image-prompt engineer. Return ONLY the final prompt "
    "text — no preamble, no quotes, no explanation. Make it vivid, specific, "
    "and suitable for a diffusion image model."
)

# Context-aware framing (operator 2026-07-13 "yes" to context-aware generation).
# The caller may pass context.kind so a MOVIE/CLIP (video) gets motion/camera
# phrasing while an IMAGE/SCENE (still) keeps diffusion phrasing. No context ->
# the original still-image behavior, so old callers are unaffected.
_PROMPT_ASSIST_KINDS = ("image", "scene", "movie", "clip")
_PROMPT_ASSIST_VIDEO_KINDS = frozenset({"movie", "clip"})

# STUDIO-SPREAD-SPEC §1a/§1b added two modes. detail/generate are UNCHANGED —
# both new modes branch out of the handler before any of their validation runs.
_ASSIST_MODES = ("detail", "generate", "spread", "negative")

# SPREAD GENERATOR (spec §3, "start official, benchmark the rest"). A spread is
# ONE call that must hold a whole timeline plus a style bible plus locked
# identity data in context and emit structured JSON — a materially harder job
# than enriching a single draft, which is why it does not inherit the
# detail/generate default. Overridable per call via body["model"]; the spec's
# benchmark protocol decides any promotion from here.
_DEFAULT_SPREAD_MODEL = "Qwen2.5-7B-Instruct-GGUF"

# A spread writes N paragraphs in one reply, so the 200-token ceiling the
# single-prompt modes use would truncate it mid-segment — and a truncated JSON
# object is an unparseable one. Scaled per target row with a floor.
_SPREAD_TOKENS_PER_SEGMENT = 320
_SPREAD_TOKENS_MIN = 700
_SPREAD_TOKENS_MAX = 4000
_NEGATIVE_MAX_TOKENS = 200


# MEDIA PRETEXT (KEEPER-TASK k93 §C) — the sampling/describing/caching lives in
# video_assist_media.py (Flask-free); these three shims are the route's glue.
from hugpy_server.app.routes import video_assist_media as _assist_media


def _assist_media_describe(media, execute_prompt):
    """Describe ``media`` through the route's OWN execute_prompt seam (the one
    tests patch, the one /chat/stream uses) driven synchronously via _await_sync,
    jailed by _jail_resolve, owned by the caller. Raises MediaError."""
    def _execute(**kw):
        return _await_sync(execute_prompt(**kw))
    return _assist_media.describe_media(
        media, jail_resolve=_jail_resolve, execute=_execute,
        owner=_caller_username())


def _assist_media_log_fields(media_info):
    """Extra assist-log fields so the operator can read what the model was told
    (``/video/prompt/assist/log``). Empty dict when no media rode the request, so
    the record is unchanged for every caller that doesn't use it."""
    if not media_info:
        return {}
    return {"media_uri": media_info["uri"], "media_kind": media_info["kind"],
            "media_model": media_info["model"], "media_cached": media_info["cached"],
            "media_frames": len(media_info["frames"]),
            "media_pretext": media_info["pretext"]}


def _assist_media_response_fields(media_info):
    if not media_info:
        return {}
    return {"media": {"uri": media_info["uri"], "kind": media_info["kind"],
                      "model": media_info["model"], "cached": media_info["cached"],
                      "frames": media_info["frames"],
                      "pretext": media_info["pretext"]}}


def _assist_framing(kind):
    """Return (system_prompt, medium_noun) for the requested kind. Video kinds
    ask for motion/camera; everything else (incl. None) keeps still-image
    phrasing identical to the pre-context behavior."""
    if kind in _PROMPT_ASSIST_VIDEO_KINDS:
        return (
            "You are an expert video-prompt engineer. Return ONLY the final "
            "prompt text — no preamble, no quotes, no explanation. Make it "
            "vivid and specific, describing subject, motion, camera movement, "
            "and mood, suitable for a text-to-video model.",
            "video-generation prompt",
        )
    return (_PROMPT_ASSIST_SYSTEM, "image-generation prompt")


from hugpy_platform.async_runtime import await_sync as _await_sync


from hugpy_platform.results import result_text as _prompt_assist_result_text


def _studio_no_think(raw: str):
    """Studio-generate's reading of a no-think reply: ``(text, reasoning,
    from_reasoning)``.

    ``no_think`` (strip_think) is a STRIP FUNCTION, not a request the model must
    obey (operator 2026-07-31: *"no_think isn't a request. it's a function that
    strips the <think>…</think> from the actual response and returns the think as
    a dict var"*). A reasoning model that keeps its whole answer inside
    ``<think>`` has NOT failed — its content is right there in the reasoning dict
    var. For studio GENERATE (an image/video prompt, where any description beats
    a hard block) we therefore SALVAGE that reasoning as the prompt instead of
    refusing a callable model, and flag ``from_reasoning`` so the caller/UI can
    say where it came from. Only a reply with neither prose NOR reasoning is a
    genuine empty.

    Deliberately LOCAL to studio generate. The shared ``finalize_no_think`` stays
    strict (empty prose → honest error) because its other callers — the discord
    DEFER gate and the movie-keyframe verdict judge — must NEVER read the
    monologue as the answer: a verdict regex matching inside the reasoning would
    invert the decision.
    """
    prose, reasoning = no_think(raw)
    if prose:
        return prose, reasoning, False
    if reasoning:
        return reasoning, reasoning, True
    return "", "", False


# --------------------------------------------------------------------------- #
# NO-THINK — the package-wide seam now lives in utils/no_think.py (operator
# 2026-07-29: "the expectation of a model to adhere to no_think should be
# circumvented for any execution that requires this stipulation, package wide").
# The two-halves rationale, the live measurement that proved a strip-only fix
# useless, and the unclosed-<think> handling are documented there. This route
# uses the primitives directly (rather than execute_prompt_no_think) because it
# owns its own JSON envelope and error copy.
# --------------------------------------------------------------------------- #
# ---- assist TICKETS (2026-08-28): a cold model load measured 6m12s on the
# live fleet, and a synchronous browser call can't survive that — the client
# aborts (ClientGone), the load restarts on the next click, and the UI
# livelocks never seeing a prompt. `async:true` detaches the generation from
# the HTTP connection: POST returns a ticket immediately, the SAME handler
# body runs in a daemon thread, and GET .../ticket/<id> serves the finished
# response. Tickets are in-process (one gunicorn worker) and pruned by age.
import threading as _assist_threading

_ASSIST_TICKETS: dict[str, dict] = {}
_ASSIST_TICKETS_LOCK = _assist_threading.Lock()
_ASSIST_TICKET_TTL_S = 1800.0


def _assist_tickets_prune() -> None:
    now = time.monotonic()
    with _ASSIST_TICKETS_LOCK:
        for key in [k for k, v in _ASSIST_TICKETS.items()
                    if now - v["at"] > _ASSIST_TICKET_TTL_S]:
            _ASSIST_TICKETS.pop(key, None)


@video_bp.route("/video/prompt/assist", methods=["POST"])
def video_prompt_assist():
    body = request.get_json(silent=True) or {}
    if not body.pop("async", False):
        return _video_prompt_assist_impl(body)

    from flask import current_app

    _assist_tickets_prune()
    ticket = secrets.token_hex(16)
    with _ASSIST_TICKETS_LOCK:
        _ASSIST_TICKETS[ticket] = {"at": time.monotonic(), "pending": True}
    app = current_app._get_current_object()

    def _run() -> None:
        try:
            with app.app_context():
                resp = _video_prompt_assist_impl(body)
        except Exception as exc:  # noqa: BLE001 — a ticket must resolve, never hang
            logger.exception("prompt/assist ticket failed")
            payload, status = json.dumps({"error": str(exc)}), 500
        else:
            r, status = resp if isinstance(resp, tuple) else (resp, 200)
            payload = r.get_data(as_text=True)
        with _ASSIST_TICKETS_LOCK:
            _ASSIST_TICKETS[ticket] = {
                "at": time.monotonic(), "pending": False,
                "body": payload, "status": status,
            }

    _assist_threading.Thread(
        target=_run, daemon=True, name=f"assist-{ticket[:8]}",
    ).start()
    return jsonify({"ticket": ticket, "pending": True}), 202


@video_bp.route("/video/prompt/assist/ticket/<ticket>", methods=["GET"])
def video_prompt_assist_ticket(ticket: str):
    with _ASSIST_TICKETS_LOCK:
        row = _ASSIST_TICKETS.get(ticket)
    if row is None:
        return jsonify({"error": "unknown or expired ticket"}), 404
    if row["pending"]:
        return jsonify({"pending": True}), 200
    return Response(row["body"], status=row["status"],
                    mimetype="application/json")


def _video_prompt_assist_impl(body: dict):
    mode = body.get("mode")
    if mode not in _ASSIST_MODES:
        return jsonify({"error": "mode must be one of "
                        + "|".join(_ASSIST_MODES)}), 400

    # SPREAD and NEGATIVE are whole-request shapes of their own (STUDIO-SPREAD-SPEC
    # §1a/§1b) — a spread has no "draft" at all, and a negative is an exclusion list
    # rather than prose. They branch BEFORE the draft/kind validation below so the
    # detail/generate path stays byte-identical to what it was.
    if mode == "spread":
        return _assist_spread(body)
    if mode == "negative":
        return _assist_negative(body)

    draft = body.get("draft")
    if draft is not None and not isinstance(draft, str):
        return jsonify({"error": "draft must be a string"}), 400
    draft = draft.strip() if draft else ""
    if mode == "detail" and not draft:
        return jsonify({"error": 'draft is required for mode "detail"'}), 400

    model_key = body.get("model") or _DEFAULT_PROMPT_ASSIST_MODEL
    if not isinstance(model_key, str) or not model_key.strip():
        return jsonify({"error": "model must be a non-empty string"}), 400

    # Optional context so the assist is aware of WHAT is being generated
    # (still image vs. a video clip/movie). Absent/empty -> still-image
    # phrasing identical to the pre-context behavior (back-compat).
    context = body.get("context") or {}
    if not isinstance(context, dict):
        return jsonify({"error": "context must be an object"}), 400
    kind = context.get("kind")
    if kind is not None and kind not in _PROMPT_ASSIST_KINDS:
        return jsonify({
            "error": "context.kind must be one of " + "|".join(_PROMPT_ASSIST_KINDS)
        }), 400
    hint = context.get("hint")
    if hint is not None and not isinstance(hint, str):
        return jsonify({"error": "context.hint must be a string"}), 400
    hint = hint.strip() if hint else ""

    # MEDIA PRETEXT (k93 §C): an optional context.media {uri, mime, label} is
    # described by a vision model and prepended to the prompt. Validated here,
    # next to context.kind, so a malformed block is the same 400 style; absent
    # -> media_info stays None and every line below is byte-identical to before.
    try:
        media = _assist_media.validate_media(context)
    except _assist_media.MediaError as exc:
        return jsonify({"error": str(exc)}), exc.status

    system_prompt, medium = _assist_framing(kind)

    if mode == "detail":
        user = (
            f"Expand this draft {medium} into a richer, more vivid, more "
            "specific prompt. Preserve the subject and intent of the draft "
            f"— add detail, don't replace it.\n\nDraft prompt: {draft}"
        )
    elif draft:
        user = (f'Write one compelling, original {medium} using '
                f'"{draft}" as a loose theme.')
    else:
        user = f"Write one compelling, original {medium} of your choosing."

    # RANDOMIZED STEERING for "generate" (operator 2026-07-27: "i need generate
    # in the /video (the llm generate) to randomize the prompt"). The instruction
    # above is IDENTICAL on every call, and a small instruct model asked the same
    # question returns the same answer — so Generate produced the same handful of
    # prompts. Temperature alone doesn't fix that: it jitters wording while the
    # model walks to the same attractor. Changing the QUESTION does. A draft, if
    # given, keeps the SUBJECT axis (see prompt_seeds.steering_axes) so steering
    # colours the shot without overwriting what the operator asked for. "detail"
    # (Enhance) is deliberately NOT steered — its contract is to preserve the draft.
    if mode == "generate":
        from hugpy_video.intel.prompt_seeds import steering_clause
        user += "\n\n" + steering_clause(kind, has_draft=bool(draft))

    if hint:
        user += f"\n\nAdditional context to honor: {hint}"

    # TYPED CONTEXT (SPEC §1c). ``hint`` stays free-form (above, unchanged); the
    # STRUCTURED row state — this segment, its neighbours, the locked identity —
    # rides typed fields the backend renders into the preface, so joint modes
    # arrive as sentences and an identity arrives with its do-not-invent list
    # instead of a bare name the model will happily make a wardrobe up for.
    # Absent typed fields render to "" and this is a no-op: a caller that sends
    # only {kind, hint} gets the byte-identical message it got before.
    from hugpy_video.intel import prompt_spread
    try:
        typed_ctx = prompt_spread.validate_context(context)
    except prompt_spread.SpreadError as exc:
        return jsonify({"error": str(exc)}), 400
    preface = prompt_spread.render_context_preface(typed_ctx)
    if preface:
        user += "\n\n" + preface

    # Late imports (mirrors prompt_routes/discord_routes) — dodges circulars
    # and keeps this module app-boot cheap when chat's plane isn't touched.
    from hugpy_engine.dispatch.dispatch import execute_prompt
    from hugpy_server.app.functions.chat.streaming import _friendly_stream_error

    # MEDIA PRETEXT (k93 §C): describe the attached media through the SAME
    # execute_prompt seam (vision task), then put the description in FRONT of
    # the instruction. Cached per uri, so generate→enhance on one clip costs
    # one describe. Failures are honest: 400/404 for a bad uri, 502 when the
    # vision plane is down — never a silent "ignored your video".
    media_info = None
    if media is not None:
        try:
            media_info = _assist_media_describe(media, execute_prompt)
        except _assist_media.MediaError as exc:
            _log_assist(run_id=_assist_log.new_run_id(), mode=mode, kind=kind,
                        model_requested=model_key, media_uri=media["uri"],
                        outcome=(_assist_log.OUTCOME_WORKER_ERROR if exc.status >= 500
                                 else _assist_log.OUTCOME_RESOLVE_ERROR),
                        error=str(exc))
            return jsonify({"error": str(exc)}), exc.status
        user = _assist_media.prepend_pretext(user, media_info["pretext"])
    media_log = _assist_media_log_fields(media_info)

    # NO-THINK, half 1 of 2: suppress the monologue at GENERATION. The user's own draft
    # rides through untouched — the directive is appended, never substituted.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _with_no_think(user)},
    ]
    # STUDIO-ASSIST LOG (operator, 2026-07-31): the detail/generate path runs its
    # own generation (not through _assist_execute), so it mints its own run_id and
    # records the terminal outcome directly. There is no downstream parse — a
    # non-empty prompt IS served — so this handler is the only place its record is
    # emitted.
    _run_id = _assist_log.new_run_id()
    _started = time.monotonic()

    def _elapsed_ms():
        return int((time.monotonic() - _started) * 1000)

    try:
        result = _await_sync(execute_prompt(
            model_key=model_key,
            messages=messages,
            task="text-generation",
            max_new_tokens=200,
        ))
    except (KeyError, ValueError, TypeError, FileNotFoundError) as exc:
        # resolve()/builder validation errors (e.g. unknown model_key, or the
        # model doesn't support text-generation) — the caller's to fix, same
        # envelope /prompt uses for the identical exception set.
        _log_assist(run_id=_run_id, mode=mode, kind=kind, **media_log,
                    model_requested=model_key,
                    outcome=_assist_log.OUTCOME_RESOLVE_ERROR,
                    error=str(exc).strip("'\""), elapsed_ms=_elapsed_ms())
        return jsonify({"error": str(exc).strip("'\"")}), 400
    except Exception as exc:
        # No live worker for this model / worker unreachable / mid-eviction.
        # Before hard-502ing, fall back ONCE to a reliable always-serve-
        # configured chat model so prompt-assist keeps working while the
        # preferred model churns (2026-08-05 fix — see _PROMPT_ASSIST_FALLBACK_
        # MODEL). Only 502 if the fallback ALSO fails, or if the primary already
        # WAS the fallback.
        if model_key != _PROMPT_ASSIST_FALLBACK_MODEL:
            logger.warning("prompt/assist: primary model %s failed (%s) — "
                           "retrying on fallback %s", model_key,
                           type(exc).__name__, _PROMPT_ASSIST_FALLBACK_MODEL)
            try:
                result = _await_sync(execute_prompt(
                    model_key=_PROMPT_ASSIST_FALLBACK_MODEL,
                    messages=messages,
                    task="text-generation",
                    max_new_tokens=200,
                ))
                model_key = _PROMPT_ASSIST_FALLBACK_MODEL
            except Exception as exc2:
                logger.exception("prompt/assist failed (primary + fallback)")
                _log_assist(run_id=_run_id, mode=mode, kind=kind, **media_log,
                            model_requested=_PROMPT_ASSIST_FALLBACK_MODEL,
                            outcome=_assist_log.OUTCOME_WORKER_ERROR,
                            error=_friendly_stream_error(exc2),
                            elapsed_ms=_elapsed_ms())
                return jsonify({"error": _friendly_stream_error(exc2)}), 502
        else:
            # Actionable message via the same mapper /chat/stream uses, never a
            # raw traceback, never a 500.
            logger.exception("prompt/assist failed")
            _log_assist(run_id=_run_id, mode=mode, kind=kind, **media_log,
                        model_requested=model_key,
                        outcome=_assist_log.OUTCOME_WORKER_ERROR,
                        error=_friendly_stream_error(exc), elapsed_ms=_elapsed_ms())
            return jsonify({"error": _friendly_stream_error(exc)}), 502

    ok = result.get("ok", True) if isinstance(result, dict) else getattr(result, "ok", True)
    raw = _prompt_assist_result_text(result).strip()
    _resolved = _assist_resolved_model(result)
    if not ok or not raw:
        err = result.get("error") if isinstance(result, dict) else getattr(result, "error", None)
        err = err or "assist produced no text"
        _log_assist(run_id=_run_id, mode=mode, kind=kind, **media_log,
                    model_requested=model_key, model_resolved=_resolved, raw=raw,
                    outcome=_assist_log.classify_execute_error(502, err),
                    error=err, elapsed_ms=_elapsed_ms())
        return jsonify({"error": err}), 502

    # NO-THINK, half 2 of 2: strip defensively, because the model is caller-selectable
    # and the next one chosen may ignore the directive. The reasoning is not thrown away
    # — it rides its own key, so it can be shown or ignored but never mistaken for the
    # prompt (operator: "or even sends it out as a dict var of its own").
    text, reasoning, from_reasoning = _studio_no_think(raw)
    if not text:
        # Neither prose NOR reasoning — a genuinely empty reply. Say so honestly
        # rather than handing back an empty prompt box; the caller can retry or
        # pick another model.
        err = ("the assistant returned nothing — neither a prompt nor "
               f"reasoning came back from {model_key!r}; retry, or choose "
               "a different text generator")
        _log_assist(run_id=_run_id, mode=mode, kind=kind, **media_log,
                    model_requested=model_key, model_resolved=_resolved, raw=raw,
                    reasoning=reasoning, from_reasoning=from_reasoning,
                    outcome=_assist_log.OUTCOME_EMPTY, error=err,
                    elapsed_ms=_elapsed_ms())
        return jsonify({
            "error": err,
            "model": model_key,
            "reasoning": reasoning,
        }), 502

    _log_assist(run_id=_run_id, mode=mode, kind=kind, **media_log,
                model_requested=model_key, model_resolved=_resolved, raw=raw,
                text=text, reasoning=reasoning, from_reasoning=from_reasoning,
                outcome=_assist_log.OUTCOME_SERVED, elapsed_ms=_elapsed_ms())
    # COORDINATION REVIEW (k121): the single-row path gets the same check the
    # spread does, over the SAME typed context the writer was shown — what the
    # model was told about the join is exactly what the knobs are checked against.
    from hugpy_server.app.routes.video_coordination import (
        COORDINATION_KEY as _COORD_KEY,
        assist_coordination,
    )
    _coord = assist_coordination(typed_ctx, text)

    # PROVENANCE (SPEC §1f, the honest half): say which model was ASKED FOR and
    # which one actually answered. The 35B incident showed generated text
    # displayed as though it came from a model that had failed to load; a caller
    # that can compare these two can never be fooled that way again. ``model`` is
    # unchanged (the requested key) so no existing consumer moves.
    return jsonify({"prompt": text, "model": model_key, "kind": kind,
                    "reasoning": reasoning, "thinking_suppressed": True,
                    "from_reasoning": from_reasoning,
                    _COORD_KEY: _coord,
                    "model_requested": model_key,
                    "model_resolved": _resolved,
                    **_assist_media_response_fields(media_info)}), 200


# --------------------------------------------------------------------------- #
# STUDIO-ASSIST LIVE LOG (operator directive, 2026-07-31).
#
# "a live log in the studio ui showing what each generate attempt actually
# returned — the raw model reply, what was stripped, and the outcome — so they
# can self-diagnose without asking the keeper each time."
#
# These two routes are the console's read side of comms/studio_assist_log.py.
# They are the exact shape of the eviction-telemetry routes (bounded backfill +
# replay-then-live SSE, tailed by sqlite rowid so the stream is correct across
# gunicorn workers). Auth is NOT re-checked here: both routes live on the /video
# surface, so the blanket video gate (video_auth.install_video_gate) already
# requires the SAME console session /video/prompt/assist itself requires — the
# UI's existing creds work unchanged, and there is no new credential to loosen.
# --------------------------------------------------------------------------- #
_LOG_REPLAY_LIMIT = 100        # replay depth for a fresh SSE subscriber
_LOG_POLL_S = 0.5              # cursor poll — sub-second is "real time" for a human
_LOG_HEARTBEAT_S = 15.0        # keep an idle proxied pipe warm
_LOG_STREAM_MAX_S = 3600.0     # a forgotten tab must not pin a thread forever
_LOG_BACKFILL_RAW_CAP = 20000  # bound a huge reply on the PAGE LOAD only (the store
                               # and the SSE stream keep the full untruncated reply)


@video_bp.route("/video/prompt/assist/log", methods=["GET"])
def video_prompt_assist_log():
    """Bounded history of studio-assist attempts — the panel's page-load backfill.

    ``limit`` (default 200, max 2000) newest records, oldest-first for direct
    rendering. ``since`` is an epoch-seconds floor; ``after_id`` is the stream
    cursor form. ``raw`` is capped for the page load only (see the SSE route for
    the untruncated stream)."""
    try:
        limit = int(request.args.get("limit") or 200)
    except (TypeError, ValueError):
        limit = 200
    limit = max(1, min(limit, 2000))
    since = request.args.get("since")
    after = request.args.get("after_id")
    try:
        since_ts = float(since) if since not in (None, "") else None
    except (TypeError, ValueError):
        since_ts = None
    try:
        after_id = int(after) if after not in (None, "") else None
    except (TypeError, ValueError):
        after_id = None
    store = _assist_log.get_store()
    records = store.recent(limit=limit, since_ts=since_ts, after_id=after_id,
                           raw_cap=_LOG_BACKFILL_RAW_CAP)
    return jsonify({"events": records, "count": len(records),
                    "cursor": (records[-1].get("_id") if records else after_id or 0)})


@video_bp.route("/video/prompt/assist/log/stream", methods=["GET"])
def video_prompt_assist_log_stream():
    """SSE: the last ~100 attempts, then live.

    Tails the shared sqlite table by rowid, which is what makes this correct
    across gunicorn workers. Emits a ``: heartbeat`` comment when idle so a proxy
    does not reap the connection, and returns after ``_LOG_STREAM_MAX_S`` so a
    forgotten tab cannot pin a thread — EventSource reconnects and replays from
    the cursor, so the operator sees no gap. The full untruncated ``raw`` rides
    the stream (only the page-load backfill bounds it)."""
    try:
        replay = int(request.args.get("replay") or _LOG_REPLAY_LIMIT)
    except (TypeError, ValueError):
        replay = _LOG_REPLAY_LIMIT
    replay = max(0, min(replay, 1000))

    def sse(payload: dict) -> bytes:
        return f"data: {json.dumps(payload, default=str)}\n\n".encode("utf-8")

    def generate():
        store = _assist_log.get_store()
        cursor = 0
        try:
            backlog = store.recent(limit=replay) if replay else []
        except Exception:  # noqa: BLE001 — an unreadable store still streams live
            backlog = []
        for rec in backlog:
            cursor = max(cursor, int(rec.get("_id") or 0))
            yield sse(rec)
        if not cursor:
            try:
                cursor = store.max_id()
            except Exception:  # noqa: BLE001
                cursor = 0
        # Tell the client it is attached even when nothing has been generated yet.
        yield sse({"stage": "stream.ready", "ts": time.time(), "cursor": cursor})
        last_beat = time.time()
        deadline = time.time() + _LOG_STREAM_MAX_S
        while time.time() < deadline:
            try:
                fresh = store.recent(limit=200, after_id=cursor)
            except Exception:  # noqa: BLE001 — a transient store fault is not a
                fresh = []      # reason to drop the operator's stream
            for rec in fresh:
                cursor = max(cursor, int(rec.get("_id") or 0))
                yield sse(rec)
            if fresh:
                last_beat = time.time()
            elif time.time() - last_beat >= _LOG_HEARTBEAT_S:
                last_beat = time.time()
                yield b": heartbeat\n\n"
            time.sleep(_LOG_POLL_S)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
        direct_passthrough=True,
    )


# --------------------------------------------------------------------------- #
# STUDIO SPREAD (STUDIO-SPREAD-SPEC §1a) + NEGATIVE (§1b) + INTENT (§1d).
#
# The build the spec calls for, in three pieces:
#   * mode="spread"   ONE generator call for a whole movie, holding unselected
#                     rows as locked context. Never N sequential prompt calls —
#                     N calls is not a slower spread, it is a DIFFERENT and worse
#                     product (six rows, six unrelated worlds), which is exactly
#                     the defect this replaces.
#   * mode="negative" a negative prompt is an exclusion list, not prose; reusing
#                     the scene system prompt returns a poem.
#   * /video/prompt/intent  the 3B router that decides whether what the user
#                     typed is a scene or a direction.
#
# The heavy lifting (validation, preface rendering, JSON parsing) lives in
# video_intel/prompt_spread.py and video_intel/prompt_intent.py so it is
# testable without a Flask app and so this module stays a routing layer.
# --------------------------------------------------------------------------- #
def _assist_resolved_model(result):
    """Which model ACTUALLY answered (§1f). ``TaskResult.model_key`` is set by the
    runner that produced the result, so it differs from the requested key exactly
    when something resolved elsewhere. None when the result doesn't say."""
    if isinstance(result, dict):
        return result.get("model_key") or result.get("model") or None
    return getattr(result, "model_key", None) or None


def _assist_execute(model_key, messages, max_new_tokens,
                    _log_mode=None, _log_kind=None, **extra):
    """Run one assist generation through the shared chat plane.

    Returns ``(payload, None)`` on success or ``(None, (body, status))`` on
    failure, using the SAME error mapping the detail/generate path uses (400 for
    a caller-fixable resolve error, 502 for a fleet/worker failure — never a 500,
    never a silent local load). Both halves of the no-think seam are applied
    here, so every new mode gets them without repeating the reasoning.

    STUDIO-ASSIST LOG (operator, 2026-07-31): this is the generation choke point,
    so it mints the ``run_id`` and records the attempt. A FAILURE is terminal and
    logged here with the classified outcome (resolve_error / worker_error /
    empty), carrying the raw reply when one came back. A SUCCESS is PROVISIONAL —
    the mode handler that parses the reply decides served-vs-parse_error, and
    emits the terminal record under the same ``run_id`` (carried in the payload).
    ``_log_mode`` / ``_log_kind`` label the record and are stripped BEFORE the
    executor call so they never reach ``execute_prompt``.
    """
    from hugpy_engine.dispatch.dispatch import execute_prompt
    from hugpy_server.app.functions.chat.streaming import _friendly_stream_error
    run_id = _assist_log.new_run_id()
    started = time.monotonic()

    def _elapsed_ms():
        return int((time.monotonic() - started) * 1000)

    try:
        result = _await_sync(execute_prompt(
            model_key=model_key,
            messages=_apply_no_think(messages),   # NO-THINK half 1
            task="text-generation",
            max_new_tokens=max_new_tokens,
            **extra,
        ))
    except (KeyError, ValueError, TypeError, FileNotFoundError) as exc:
        _log_assist(run_id=run_id, mode=_log_mode, kind=_log_kind,
                    model_requested=model_key,
                    outcome=_assist_log.OUTCOME_RESOLVE_ERROR,
                    error=str(exc).strip("'\""), elapsed_ms=_elapsed_ms())
        return None, ({"error": str(exc).strip("'\""),
                       "model_requested": model_key, "run_id": run_id}, 400)
    except Exception as exc:
        logger.exception("prompt/assist failed")
        _log_assist(run_id=run_id, mode=_log_mode, kind=_log_kind,
                    model_requested=model_key,
                    outcome=_assist_log.OUTCOME_WORKER_ERROR,
                    error=_friendly_stream_error(exc), elapsed_ms=_elapsed_ms())
        return None, ({"error": _friendly_stream_error(exc),
                       "model_requested": model_key, "run_id": run_id}, 502)

    ok = result.get("ok", True) if isinstance(result, dict) else getattr(result, "ok", True)
    raw = _prompt_assist_result_text(result).strip()
    resolved = _assist_resolved_model(result)
    if not ok or not raw:
        err = result.get("error") if isinstance(result, dict) else getattr(result, "error", None)
        err = err or "assist produced no text"
        _log_assist(run_id=run_id, mode=_log_mode, kind=_log_kind,
                    model_requested=model_key, model_resolved=resolved, raw=raw,
                    outcome=_assist_log.classify_execute_error(502, err),
                    error=err, elapsed_ms=_elapsed_ms())
        return None, ({"error": err,
                       "model_requested": model_key,
                       "model_resolved": resolved, "run_id": run_id}, 502)

    text, reasoning, from_reasoning = _studio_no_think(raw)   # NO-THINK half 2
    if not text:
        err = ("the assistant returned nothing — neither output nor "
               f"reasoning came back from {model_key!r}; retry, or choose "
               "a different text generator")
        _log_assist(run_id=run_id, mode=_log_mode, kind=_log_kind,
                    model_requested=model_key, model_resolved=resolved, raw=raw,
                    reasoning=reasoning, from_reasoning=from_reasoning,
                    outcome=_assist_log.OUTCOME_EMPTY, error=err,
                    elapsed_ms=_elapsed_ms())
        return None, ({
            "error": err,
            "model": model_key,
            "model_requested": model_key,
            "model_resolved": resolved,
            "reasoning": reasoning,
            "run_id": run_id,
        }, 502)
    # SUCCESS — provisional. The mode handler emits the terminal record (served
    # or parse_error) under this run_id, with the raw reply already captured.
    return {"text": text, "reasoning": reasoning, "raw": raw,
            "from_reasoning": from_reasoning,
            "model_resolved": resolved,
            "run_id": run_id, "elapsed_ms": _elapsed_ms(),
            "model_requested": model_key,
            "log_mode": _log_mode, "log_kind": _log_kind}, None


def _assist_spread(body):
    """``mode="spread"`` — one call, N segment replacements, locked rows untouched."""
    from hugpy_video.intel import prompt_spread

    model_key = body.get("model") or _DEFAULT_SPREAD_MODEL
    if not isinstance(model_key, str) or not model_key.strip():
        return jsonify({"error": "model must be a non-empty string"}), 400
    model_key = model_key.strip()

    try:
        req = prompt_spread.build_spread_request(body)
    except prompt_spread.SpreadError as exc:
        return jsonify({"error": str(exc)}), 400

    n = len(req.target_segments)
    budget = max(_SPREAD_TOKENS_MIN,
                 min(_SPREAD_TOKENS_MAX, n * _SPREAD_TOKENS_PER_SEGMENT))

    messages = prompt_spread.build_spread_messages(req)

    # MEDIA PRETEXT (k93 §C) — same contract as detail/generate: validated next
    # to the rest of context, described once (cached per uri), prepended to the
    # ONE generator call's user message. Absent -> messages untouched.
    media_info = None
    try:
        media = _assist_media.validate_media(body.get("context") or {})
        if media is not None:
            from hugpy_engine.dispatch.dispatch import execute_prompt
            media_info = _assist_media_describe(media, execute_prompt)
    except _assist_media.MediaError as exc:
        _log_assist(run_id=_assist_log.new_run_id(), mode="spread",
                    model_requested=model_key,
                    media_uri=((body.get("context") or {}).get("media") or {}).get("uri"),
                    outcome=(_assist_log.OUTCOME_WORKER_ERROR if exc.status >= 500
                             else _assist_log.OUTCOME_RESOLVE_ERROR),
                    error=str(exc))
        return jsonify({"error": str(exc)}), exc.status
    if media_info is not None:
        messages[-1]["content"] = _assist_media.prepend_pretext(
            messages[-1]["content"], media_info["pretext"])
    media_log = _assist_media_log_fields(media_info)

    payload, err = _assist_execute(model_key, messages, budget, _log_mode="spread")
    if err is not None:
        body_out, status = err
        return jsonify(body_out), status

    try:
        # Pass the whole req (not just target_ids): the parser assembles each
        # result row's operation/negative/directions from the structure the
        # backend already holds, so the model only has to supply the prose.
        parsed = prompt_spread.parse_spread_reply(payload["text"], req)
    except prompt_spread.SpreadParseError as exc:
        # HONEST 502 (spec §1e/§1f). The raw, think-stripped reply rides along so
        # the failure is diagnosable — an invented segment here would be worse
        # than no segment, because it would silently become the user's movie.
        # TERMINAL LOG RECORD: this is the "did not return the JSON object the
        # spread contract requires" case — capture the FULL raw reply so the
        # operator can read exactly what the model sent.
        _log_assist(run_id=payload["run_id"], mode="spread", **media_log,
                    model_requested=model_key,
                    model_resolved=payload["model_resolved"],
                    raw=payload["raw"], text=payload["text"],
                    reasoning=payload["reasoning"],
                    from_reasoning=payload["from_reasoning"],
                    outcome=_assist_log.OUTCOME_PARSE_ERROR, error=str(exc),
                    elapsed_ms=payload.get("elapsed_ms"))
        return jsonify({
            "error": str(exc),
            "model": model_key,
            "model_requested": model_key,
            "model_resolved": payload["model_resolved"],
            "raw": exc.raw[:4000],
            "reasoning": payload["reasoning"],
        }), 502

    _log_assist(run_id=payload["run_id"], mode="spread", **media_log,
                model_requested=model_key,
                model_resolved=payload["model_resolved"], raw=payload["raw"],
                text=payload["text"], reasoning=payload["reasoning"],
                from_reasoning=payload["from_reasoning"],
                outcome=_assist_log.OUTCOME_SERVED,
                elapsed_ms=payload.get("elapsed_ms"))
    # COORDINATION REVIEW (k121). The spread already tells the writer what each
    # join MEANS; this turns the knob the writer was told about. Sets the
    # ratchet-safe ones onto the result rows (``segments[i].knobs``) and attaches
    # the report. Additive: a caller that ignores both keys is unchanged.
    from hugpy_server.app.routes.video_coordination import (
        COORDINATION_KEY as _COORD_KEY,
        spread_coordination,
    )
    parsed = spread_coordination(req, parsed)

    return jsonify({
        "mode": "spread",
        "segments": parsed["segments"],
        "missing_segments": parsed["missing_segments"],
        "warnings": parsed["warnings"],
        "invented_identity_attributes": parsed["invented_identity_attributes"],
        _COORD_KEY: parsed.get(_COORD_KEY),
        # Echo the steering set + the seed that produced it so the caller can PIN
        # a spread it liked and re-spread a subset into the same world later.
        "steering": req.steering,
        "steering_seed": req.steering_seed,
        "model": model_key,
        "model_requested": model_key,
        "model_resolved": payload["model_resolved"],
        "reasoning": payload["reasoning"],
        "thinking_suppressed": True,
        **_assist_media_response_fields(media_info),
    }), 200


def _assist_negative(body):
    """``mode="negative"`` — an artifact/quality exclusion list (§1b)."""
    from hugpy_video.intel import prompt_spread

    model_key = body.get("model") or _DEFAULT_PROMPT_ASSIST_MODEL
    if not isinstance(model_key, str) or not model_key.strip():
        return jsonify({"error": "model must be a non-empty string"}), 400
    model_key = model_key.strip()

    draft = body.get("draft")
    if draft is not None and not isinstance(draft, str):
        return jsonify({"error": "draft must be a string"}), 400
    subject = body.get("subject") or body.get("prompt")
    if subject is not None and not isinstance(subject, str):
        return jsonify({"error": "subject must be a string"}), 400

    context = body.get("context") or {}
    if not isinstance(context, dict):
        return jsonify({"error": "context must be an object"}), 400
    hint = context.get("hint")
    if hint is not None and not isinstance(hint, str):
        return jsonify({"error": "context.hint must be a string"}), 400
    try:
        typed_ctx = prompt_spread.validate_context(context)
    except prompt_spread.SpreadError as exc:
        return jsonify({"error": str(exc)}), 400

    # The shot being negated is whatever the caller can tell us about it: an
    # explicit subject, else the current segment's own prompt.
    if not subject:
        subject = (typed_ctx.get("segment") or {}).get("prompt") or ""

    messages = prompt_spread.build_negative_messages(
        (draft or "").strip(), typed_ctx, hint=(hint or "").strip(),
        subject=subject.strip())
    payload, err = _assist_execute(model_key, messages, _NEGATIVE_MAX_TOKENS,
                                   _log_mode="negative")
    if err is not None:
        body_out, status = err
        return jsonify(body_out), status

    # A negative exclusion list needs no further parse — a non-empty reply IS the
    # result, so the attempt is served the moment _assist_execute returned it.
    _log_assist(run_id=payload["run_id"], mode="negative",
                model_requested=model_key,
                model_resolved=payload["model_resolved"], raw=payload["raw"],
                text=payload["text"], reasoning=payload["reasoning"],
                from_reasoning=payload["from_reasoning"],
                outcome=_assist_log.OUTCOME_SERVED,
                elapsed_ms=payload.get("elapsed_ms"))
    # ``prompt`` carries the text for every assist mode (no existing key moves);
    # ``negative`` is the same string under the name this mode's caller wants.
    return jsonify({
        "mode": "negative",
        "prompt": payload["text"],
        "negative": payload["text"],
        "model": model_key,
        "model_requested": model_key,
        "model_resolved": payload["model_resolved"],
        "reasoning": payload["reasoning"],
        "thinking_suppressed": True,
    }), 200


# --------------------------------------------------------------------------- #
# POST /video/producer/plan — the CINEMA PRODUCER (KEEPER-TASK k120, slice 1).
#
# ONE call: premise → full structured film plan → per-segment
# {prompt, negative, seconds, joint}. The cinema composer populates its rows
# from this instead of the operator hand-counting segments and prompting each.
# Lengths ride as SECONDS (fps is movie-level; the composer owns the
# seconds→frames conversion, and the server clamps/snaps at submit anyway).
#
# Contracts, in the family's own idioms:
#   * Model: body["model"] wins, else the k109 routing-matrix pick for
#     screenplay.complete (resolve_authoring_model — lazy import, same reason
#     as script_first_routes._sf), else _DEFAULT_SPREAD_MODEL. NEVER the silent
#     3B task-default (k53 §2 spirit; operator: "the 3B is no good").
#   * Output is JSON (unlike spread's labelled blocks) because the plan is
#     TYPED — seconds and joint per segment — parsed with
#     utils.json_scavenge.extract_json_object on the think-stripped text, with
#     exactly ONE repair retry echoing why (screenplay._author's two-attempt
#     contract), then an honest 502 carrying the raw reply.
#   * Segment 0 is forced joint="cut" (studio_movie_schema hard-refuses a
#     non-cut first joint); later segments default to "vace_extend" so N+1
#     extends N's closing frame — the continuity chain the operator asked for.
# --------------------------------------------------------------------------- #
_PRODUCER_TOKENS_PER_SEGMENT = 340
_PRODUCER_TOKENS_MIN = 900
_PRODUCER_TOKENS_MAX = 4000
_PRODUCER_MAX_SEGMENTS = 64
_PRODUCER_VALID_JOINTS = ("cut", "still", "vace_extend")
_PRODUCER_DEFAULT_SECONDS = 3.4

_PRODUCER_SYSTEM = (
    "You are a film producer and director planning an AI-generated short film "
    "for an image-to-video pipeline. You break a premise into sequential "
    "segments (shots) and write each segment's generation prompt. Reply with "
    "ONLY one JSON object — no preamble, no markdown fences, no commentary."
)

# k120 landmine (STUDIO min_rank_floor): the writer chain's resolve fallthrough
# bottoms out at the 3B, which the operator ruled "no good" for script work.
# The registry cannot express a rank floor (a group's member order IS its order
# of operations, k119 — there is no per-stage floor field), so the STUDIO
# dispatch route enforces it here: a plan authored by a below-floor chain
# member is refused err-as-data (503, retryable) — queueing beats shipping a
# bad screenplay. Rank = 1-based position in the writer group's member order.
_STUDIO_WRITER_GROUP = "hugpy-agent-brains"
_STUDIO_MIN_RANK_FLOOR = 3


def _studio_floor_violation(resolved_model):
    """Error text when ``resolved_model`` sits below STUDIO's rank floor in the
    writer chain, else None. Fail-open on registry trouble and on models
    outside the group: the floor guards the KNOWN low-rank fallthrough, not an
    explicitly pinned outside model (a caller's pin is their choice)."""
    if not resolved_model:
        return None
    try:
        from hugpy_fleet.central import priority_groups
        group = priority_groups.get_group(_STUDIO_WRITER_GROUP)
        if not group:
            return None
        members = priority_groups.expand_members(group)
    except Exception:  # noqa: BLE001 — registry trouble must not kill the plan
        return None
    for rank, member in enumerate(members or [], start=1):
        # expand_members yields (member_key, source) tuples; tolerate bare
        # strings too so a future shape change fails matched, not raised.
        key = member[0] if isinstance(member, (tuple, list)) else member
        try:
            matched = priority_groups.keys_match(key, resolved_model)
        except Exception:  # noqa: BLE001
            matched = key == resolved_model
        if matched:
            if rank > _STUDIO_MIN_RANK_FLOOR:
                return (
                    f"script authoring routed to {resolved_model!r} — rank "
                    f"{rank} in the {_STUDIO_WRITER_GROUP!r} chain, below "
                    f"STUDIO's min_rank_floor of {_STUDIO_MIN_RANK_FLOOR} "
                    f"(k120: low-rank members are not acceptable for script "
                    f"work, pinned or not). Retry when a higher-rank member "
                    f'is servable, or pin an above-floor "model".'
                )
            return None
    return None


def _producer_messages(premise, segment_count, target_seconds, global_negative):
    schema = (
        '{"title": "...", "logline": "...", '
        '"style": {"setting": "...", "cast": "...", "visual_style": "...", '
        '"camera_language": "..."}, '
        '"segments": [{"prompt": "...", "negative": "...", '
        '"seconds": 3.4, "joint": "cut"}]}'
    )
    rules = [
        "Return ONLY the JSON object, matching this shape exactly: " + schema,
        "Each segment prompt must be a SELF-CONTAINED visual description "
        "(subject, action, setting, camera, lighting, present tense) — the "
        "video model sees one segment at a time and remembers nothing.",
        "Keep characters, wardrobe, setting and light continuous from segment "
        "to segment by RESTATING them; never write 'the same man as before'.",
        'joint says how a segment attaches to the previous one: "vace_extend" '
        "continues directly from the previous shot's last frame (same scene, "
        'continuous motion); "cut" starts a new shot. The FIRST segment must '
        'be "cut".',
        "seconds is the segment's duration; typical shots run 2–8 seconds.",
    ]
    if segment_count:
        rules.append(f"Use exactly {int(segment_count)} segments.")
    else:
        rules.append("Choose the segment count the story needs.")
    if target_seconds:
        rules.append(
            f"The whole film should total roughly {float(target_seconds):g} seconds."
        )
    if global_negative:
        rules.append(
            "A movie-level negative prompt already covers: "
            + str(global_negative)[:400]
            + " — per-segment negatives should only add SEGMENT-SPECIFIC exclusions."
        )
    user = "Premise:\n" + premise + "\n\nRules:\n- " + "\n- ".join(rules)
    return [
        {"role": "system", "content": _PRODUCER_SYSTEM},
        {"role": "user", "content": user},
    ]


def _producer_parse(text):
    """(plan, "") on success, (None, why) on failure — never raises."""
    from hugpy_engine.utils.json_scavenge import extract_json_object

    obj = extract_json_object(text or "")
    if not isinstance(obj, dict):
        return None, "no JSON object found in the reply"
    segs = obj.get("segments")
    if not isinstance(segs, list) or not segs:
        return None, 'the JSON object has no non-empty "segments" array'
    return obj, ""


@video_bp.route("/video/producer/plan", methods=["POST"])
def video_producer_plan():
    body = request.get_json(silent=True) or {}
    premise = str(body.get("premise") or "").strip()
    if not premise:
        return jsonify({"error": "premise is required"}), 400

    seg_count = body.get("segment_count")
    try:
        seg_count = int(seg_count) if seg_count not in (None, "", 0) else None
    except (TypeError, ValueError):
        return jsonify({"error": "segment_count must be an integer"}), 400
    if seg_count is not None and not 1 <= seg_count <= _PRODUCER_MAX_SEGMENTS:
        return jsonify({"error": f"segment_count must be 1..{_PRODUCER_MAX_SEGMENTS}"}), 400
    target_seconds = body.get("target_seconds")
    try:
        target_seconds = float(target_seconds) if target_seconds not in (None, "") else None
    except (TypeError, ValueError):
        return jsonify({"error": "target_seconds must be a number"}), 400

    # Pool: request tag > active task-template. An activated template fences
    # its workers to its pool, so a pool-less plan request would be refused
    # for the exact models the template pinned — inherit the active pool.
    pool = str(body.get("pool") or "").strip() or None
    if pool is None:
        try:
            from hugpy_fleet.central.task_templates import all_templates

            pool = next(
                (t["id"] for t in all_templates() if t.get("active")), None,
            )
        except Exception as exc:  # noqa: BLE001 — pool discovery must not kill the plan
            logger.warning("producer: active-template pool lookup failed (%s)", exc)

    # Model: request pin > routing matrix (screenplay.complete) > spread default.
    model_key = str(body.get("model") or "").strip()
    route_reason = "request"
    if not model_key:
        try:
            from hugpy_oracle.script_first import resolve_authoring_model

            choice = resolve_authoring_model("screenplay")
            model_key = str(choice.get("requested_model") or "").strip()
            if model_key:
                route_reason = f"{choice.get('source')}: {choice.get('reason')}"
        except Exception as exc:  # noqa: BLE001 — matrix trouble must not kill the plan
            logger.warning("producer: routing matrix unavailable (%s)", exc)
    if not model_key:
        model_key = _DEFAULT_SPREAD_MODEL
        route_reason = "producer default (spread model)"

    # Floor check BEFORE dispatch: a below-floor chain member (the 3B and
    # anything after it) is never even invoked for script work — refusing
    # after generation would still have burned GPU on an answer we discard
    # (operator 2026-08-28: "don't use 3B"). The post-execution check below
    # stays as the backstop for the resolver's INTERNAL fallthrough.
    floor_err = _studio_floor_violation(model_key)
    if floor_err:
        logger.warning("producer: floor refusal pre-dispatch (%s)", floor_err)
        return jsonify({
            "error": floor_err,
            "model_requested": model_key,
            "route_reason": route_reason,
        }), 503

    budget = max(
        _PRODUCER_TOKENS_MIN,
        min(_PRODUCER_TOKENS_MAX, (seg_count or 10) * _PRODUCER_TOKENS_PER_SEGMENT),
    )
    messages = _producer_messages(
        premise, seg_count, target_seconds, body.get("global_negative"),
    )
    payload, err = _assist_execute(
        model_key, messages, budget, _log_mode="producer", _log_kind="plan",
        pool=pool,
    )
    if err:
        return err
    plan, why = _producer_parse(payload["text"])
    if plan is None:
        # Exactly one repair attempt (screenplay._author's contract): echo the
        # reply and the reason, ask for the JSON alone. Then 502 honestly.
        repair = messages + [
            {"role": "assistant", "content": (payload.get("raw") or "")[:4000]},
            {
                "role": "user",
                "content": (
                    "Your reply could not be used: " + why + ". Return ONLY the "
                    "corrected JSON object described above — nothing else."
                ),
            },
        ]
        payload2, err2 = _assist_execute(
            model_key, repair, budget, _log_mode="producer", _log_kind="plan-repair",
            pool=pool,
        )
        if err2:
            return err2
        plan, why = _producer_parse(payload2["text"])
        if plan is None:
            logger.error("producer: unparseable after repair (%s)", why)
            return jsonify({
                "error": "producer reply unparseable",
                "detail": why,
                "raw": (payload2.get("raw") or "")[:4000],
            }), 502
        payload = payload2

    floor_err = _studio_floor_violation(payload.get("model_resolved"))
    if floor_err:
        logger.warning("producer: floor refusal (%s)", floor_err)
        return jsonify({
            "error": floor_err,
            "model_requested": model_key,
            "model_resolved": payload.get("model_resolved"),
            "route_reason": route_reason,
            "run_id": payload.get("run_id"),
        }), 503

    warnings: list = []
    segments = []
    for i, seg in enumerate(plan.get("segments", [])[:_PRODUCER_MAX_SEGMENTS]):
        if not isinstance(seg, dict):
            warnings.append(f"segment {i + 1}: not an object — dropped")
            continue
        prompt = str(seg.get("prompt") or seg.get("text") or "").strip()
        if not prompt:
            warnings.append(f"segment {i + 1}: empty prompt — dropped")
            continue
        try:
            seconds = float(seg.get("seconds") or 0)
        except (TypeError, ValueError):
            seconds = 0
        if not 0.2 <= seconds <= 120:
            if seg.get("seconds") not in (None, ""):
                warnings.append(
                    f"segment {i + 1}: seconds {seg.get('seconds')!r} out of range — "
                    f"defaulted to {_PRODUCER_DEFAULT_SECONDS}"
                )
            seconds = _PRODUCER_DEFAULT_SECONDS
        joint = str(seg.get("joint") or "").strip().lower()
        if joint not in _PRODUCER_VALID_JOINTS:
            joint = "vace_extend"
        segments.append({
            "segment_id": f"seg{len(segments) + 1:02d}",
            "prompt": prompt,
            "negative": str(seg.get("negative") or "").strip(),
            "seconds": round(seconds, 2),
            "joint": joint,
        })
    if not segments:
        return jsonify({
            "error": "the plan contained no usable segments",
            "raw": (payload.get("raw") or "")[:4000],
        }), 502
    # studio_movie_schema hard-refuses a non-cut joint on segment 0 — there is
    # no previous shot to carry a frame from.
    segments[0]["joint"] = "cut"

    style = plan.get("style")
    return jsonify({
        "mode": "producer",
        "title": str(plan.get("title") or "").strip(),
        "logline": str(plan.get("logline") or "").strip(),
        "style": style if isinstance(style, dict) else {},
        "segments": segments,
        "total_seconds": round(sum(s["seconds"] for s in segments), 2),
        "warnings": warnings,
        "model": model_key,
        "model_requested": model_key,
        "model_resolved": payload["model_resolved"],
        "route_reason": route_reason,
        "reasoning": payload["reasoning"],
        "thinking_suppressed": True,
    }), 200


# --------------------------------------------------------------------------- #
# POST /video/prompt/intent — the 3B intent router (spec §1d).
#
# WHY ITS OWN ROUTE RATHER THAN A MODE ON /assist (keeper's call, per the spec's
# "or a mode on the assist route — your call"):
#   * It does not generate. Every /assist mode returns a prompt and 502s when it
#     cannot; this returns a CLASSIFICATION and must never fail the caller — its
#     whole failure contract is inverted (degrade to "ambiguous", always 200).
#   * Different model role, different budget (100 tokens, temp 0) and a cache;
#     folding it into /assist would put a "sometimes cached, never errors" branch
#     inside a handler whose contract is "always generates, errors honestly".
#   * It is called on BLUR — far more often than a generation, and from fields
#     that may never be generated at all.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/prompt/intent", methods=["POST"])
def video_prompt_intent():
    from hugpy_video.intel import prompt_intent

    body = request.get_json(silent=True) or {}
    text = body.get("text")
    if text is None:
        text = body.get("draft")
    if text is not None and not isinstance(text, str):
        return jsonify({"error": "text must be a string"}), 400

    scope = body.get("scope") or "segment"
    if scope not in ("segment", "movie"):
        return jsonify({"error": 'scope must be "segment" or "movie"'}), 400

    model = body.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        return jsonify({"error": "model must be a non-empty string"}), 400

    # classify_intent NEVER raises and never blocks: blank short-circuits without
    # a model call, and any router failure degrades to "ambiguous" so the UI
    # offers both actions. A classifier outage must not make a text box unusable.
    return jsonify(prompt_intent.classify_intent(text, scope=scope, model=model)), 200


# --------------------------------------------------------------------------- #
# GPU-worker auto-pick — reuses the worker registry helpers (workers_for_model /
# _has_usable_gpu / _worker_fit) that the /llm/workers/<id>/assign path uses, so
# a preset lands on the same class of worker an operator would pick by hand.
# --------------------------------------------------------------------------- #
def _pick_gpu_worker(model_key):
    """Choose an online, approved, GPU-capable worker to warm ``model_key`` on.

    Preference order (best first): a worker that already carries the model
    (assigned or loaded — no reload), then one where it fits VRAM outright, then
    any where it at least fits (VRAM+RAM), then the most free VRAM. Returns None
    when no GPU worker is eligible (the caller maps that to a 409 NoGpuWorker).
    """
    from hugpy_fleet.central.workers import worker_store, _has_usable_gpu, _engine_unusable
    from hugpy_server.app.routes.worker_routes import _worker_fit

    # Workers already serving this model (assigned OR loaded), stale beats ok —
    # reusing one avoids a multi-GB reload. online_only=False so a briefly-stale
    # assignee still counts as "already has it". Wildcard catches are excluded:
    # a "take all comers" box (worker_wildcard opt-in) is ELIGIBLE for the
    # model but does not HAVE it, so counting it "warm" would fake the
    # avoid-a-reload preference this set exists for.
    warm_ids = {w["id"] for w in
                worker_store.workers_for_model(model_key, online_only=False)
                if not w.get("_wildcard_catch")}

    eligible = []
    for w in worker_store.all():
        # Same admission/engine/liveness gates workers_for_model applies, plus a
        # hard GPU requirement (a preset's "recommended: gpu" is load-bearing).
        if w.get("admission") != "approved":
            continue
        if _engine_unusable(w):
            continue
        if w.get("status") != "online":
            continue
        if not _has_usable_gpu(w):
            continue
        eligible.append(w)
    if not eligible:
        return None

    def _rank(w):
        fit = _worker_fit(model_key, w)   # fit/gpu_resident None for unsizable models
        return (
            0 if w["id"] in warm_ids else 1,
            0 if fit.get("gpu_resident") else 1,
            0 if fit.get("fit") is not False else 1,
            -(fit.get("vram_free") or 0),
            w.get("id", ""),
        )

    eligible.sort(key=_rank)
    return eligible[0]


# --------------------------------------------------------------------------- #
# 2f) POST /video/presets/<preset_id>/apply — validate + auto-pick a GPU worker,
#     assign + background-warm the model, return the gen defaults for the UI.
#     Guards mirror workers_assign: catalog membership (404) + central-holds-
#     files (409); adds preset-exists (404) and no-GPU-worker (409) on top.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/presets/<preset_id>/apply", methods=["POST"])
def video_preset_apply(preset_id):
    from hugpy_video.intel.presets import get_preset

    preset = get_preset(preset_id)
    if preset is None:
        return jsonify({"ok": False, "error": {
            "code": "UnknownPreset",
            "message": f"no video preset {preset_id!r}"}}), 404

    model_key = preset.model_key

    # Catalog membership — same source (get_models_dict) workers_assign checks.
    from hugpy_engine.config.models.models_config import get_models_dict
    if model_key not in get_models_dict(dict_return=True):
        return jsonify({"ok": False, "error": {
            "code": "UnknownModel",
            "message": f"preset model {model_key!r} is not in the catalog"}}), 404

    # Item-4 invariant: central must hold the files, or the worker silently pulls
    # from HF at internet speed. Reuse the exact guard workers_assign uses.
    from hugpy_server.app.routes.worker_routes import _central_missing_reason
    missing = _central_missing_reason(model_key)
    if missing:
        return jsonify({"ok": False, "error": {
            "code": "CentralMissing",
            "message": (f"central does not have {model_key!r} on disk ({missing}) "
                        "— download it on the Models tab first; workers provision "
                        "from central")}}), 409

    # Auto-pick a GPU-capable worker (presets are "recommended: gpu").
    worker = _pick_gpu_worker(model_key)
    if worker is None:
        return jsonify({"ok": False, "error": {
            "code": "NoGpuWorker",
            "message": ("no online GPU-capable worker is available to warm this "
                        "preset — bring a GPU worker online or assign manually")}}), 409

    # Apply = ALLOCATE only: assign the preset's model to the picked worker. It
    # is NOT loaded here — it loads when a generation calls it (operator
    # 2026-09-25: no phantom loads of never-called models).
    from hugpy_fleet.central.workers import assign_model
    # Auto-picked worker → an automated ("autoplace") designation: transient,
    # pruned when idle.
    assigned = assign_model(worker["id"], model_key, source="autoplace")
    if assigned is None:
        # Raced: the worker vanished between pick and assign.
        return jsonify({"ok": False, "error": {
            "code": "NoGpuWorker",
            "message": "the selected worker is no longer available — retry"}}), 409

    return jsonify({
        "ok": True,
        "worker": {"name": assigned.get("name"), "id": assigned.get("id")},
        "model_key": model_key,
        "mode": preset.mode,
        "defaults": preset.defaults(),
        "warming": False,   # allocation only — loads on the first generation
    }), 200


# --------------------------------------------------------------------------- #
# 2g) GET /movie/presets — curated MOVIE TEMPLATES for the Movie Maker tab
# --------------------------------------------------------------------------- #
# Movie twin of GET /video/presets: import the static registry, dump it as JSON.
# No side effects — a movie template is just a named bundle of a model_key + the
# scene-template settings + a goal timeline (contiguous half-open intervals that
# tile [0, total)) the Movie Maker tab pre-fills into its goal editor.
@video_bp.route("/movie/presets", methods=["GET"])
def movie_presets():
    from hugpy_video.intel.presets import available_movie_presets
    return jsonify({"presets": [p.to_dict() for p in available_movie_presets()]}), 200


# --------------------------------------------------------------------------- #
# 2h) POST /movie/presets/<preset_id>/apply — return the directly-POSTable
#     generate_movie body for this template (unknown id -> 404). Read-only/open,
#     same auth posture as GET /movie/presets: unlike the video apply this does
#     NOT touch the worker plane — a movie template just pre-fills the goal editor
#     (its `request` sub-object is a curated /video/jobs/generate_movie body).
# --------------------------------------------------------------------------- #
@video_bp.route("/movie/presets/<preset_id>/apply", methods=["POST"])
def movie_preset_apply(preset_id):
    from hugpy_video.intel.presets import get_movie_preset

    preset = get_movie_preset(preset_id)
    if preset is None:
        return jsonify({"ok": False, "error": {
            "code": "UnknownPreset",
            "message": f"no movie preset {preset_id!r}"}}), 404

    return jsonify(preset.apply()), 200


# --------------------------------------------------------------------------- #
# 2i) GET /video/studio/presets — curated STUDIO clip presets for the Studio Clips
#     station. Studio twin of GET /video/presets + GET /movie/presets: import the
#     static registry, dump it as JSON. No side effects — a studio preset is a named
#     bundle of a capability ("i2v"/"t2v") + geometry + a routing vram_budget_gb the
#     station pre-fills into its generate affordance. The model is NOT pinned here;
#     the studio router resolves capability + resolution + budget at run time.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/studio/presets", methods=["GET"])
def studio_presets():
    from hugpy_video.intel.studio_presets import available_studio_presets
    return jsonify({"presets": [p.to_dict() for p in available_studio_presets()]}), 200


# --------------------------------------------------------------------------- #
# 2j) POST /video/studio/presets/<preset_id>/apply — return the directly-POSTable
#     /video/studio/i2v body for this preset (unknown id -> 404). Read-only/open,
#     same posture as the movie apply: unlike the video-preset apply this does NOT
#     touch the worker plane — a studio preset just pre-fills the generate affordance
#     (its `request` sub-object is a curated /video/studio/i2v body).
# --------------------------------------------------------------------------- #
@video_bp.route("/video/studio/presets/<preset_id>/apply", methods=["POST"])
def studio_preset_apply(preset_id):
    from hugpy_video.intel.studio_presets import get_studio_preset

    preset = get_studio_preset(preset_id)
    if preset is None:
        return jsonify({"ok": False, "error": {
            "code": "UnknownPreset",
            "message": f"no studio preset {preset_id!r}"}}), 404

    return jsonify(preset.apply()), 200


# --------------------------------------------------------------------------- #
# 2k) GET /video/render/presets — WHAT THIS FLEET CAN ACTUALLY RENDER TODAY.
#
#     Fourth and LAST preset surface on this blueprint, and the only one grounded in
#     measurement rather than intent. The other three answer "what did we curate?";
#     this one answers "what will come back as pixels?" — the eight ratified
#     ``RenderPreset`` rows in video_intel/studio/presets.py, each a PROVEN TUPLE of
#     (capability, model, precision, geometry, frame budget) measured on the live
#     fleet 2026-07-27 (CAPABILITY-VIABILITY-MAP.md, MODEL-POOL-INVENTORY.md).
#
#     WHY A NEW PATH RATHER THAN EXTENDING AN OLD ONE. /video/presets (scene "ideal
#     default loads"), /movie/presets (Movie Maker goal timelines) and
#     /video/studio/presets (the Studio Clips station's prefill bundles) are three
#     FROZEN wire contracts with live console callers and their own registries
#     (video_intel/presets.py, video_intel/studio_presets.py). Folding a different
#     table into any of them would break a pinned shape. So this is additive and
#     namespaced under /video/render/ — "render presets", the RenderPreset table.
#
#     HONEST ABOUT THE GAPS TOO. The list is only half the answer: `unavailable`
#     carries every ``Capability`` the enum declares and NO preset covers, with the
#     measured blocker and the full refusal text — straight from the registry's own
#     ``capability_verdict``, never re-worded here. That is the whole point of the
#     table: a caller learns "studio cannot render 'audio' on this fleet: the studio
#     clip contract has NO audio track at all …" at DISCOVERY time instead of
#     enqueueing a job that dies in a runner three layers down.
#
#     Derived at REQUEST time from the registry — there is deliberately no cached or
#     hardcoded copy in the route layer, so proving a ninth preset is a one-file
#     change (tests/test_video_presets_route.py pins this by swapping the registry
#     accessor and demanding the response follow).
#
#     Read-only, no side effects, no worker/catalog touch. Auth is the blanket
#     /video gate (video_auth.install_video_gate): a console session OR a video-share
#     credential, exactly like every sibling preset GET — nothing new, nothing looser.
# --------------------------------------------------------------------------- #
def _render_preset_placement(p):
    """The two VRAM numbers a caller needs BEFORE it fills in a budget field, derived
    (never restated) from the SAME code the router and the runner read.

    Returns ``(envelope_gb, need_gib, fits_render_box)``.

    THE TWO NUMBERS ARE DIFFERENT QUESTIONS AND THE CONSOLE NEEDS BOTH:

      * ``envelope_gb`` — the registry's declared VRAM cost for THIS preset's model at
        THIS preset's precision (``ModelConfig.vram.as_map()[precision]``). This is the
        one and only number ``vram_budget_gb`` is ever compared against: router.py:368
        does ``cfg.vram.fits(req.vram_budget_gb)``, which is ``gb <= budget``. So a
        caller that wants THIS preset's binding must send a budget >= this, and a
        console that wants to stop offering budgets that cannot bind anything real must
        read it from here rather than mirror a constant by hand. (The studio console did
        mirror one — ``STUDIO_REAL_FLOOR_GB = 6`` — and it had drifted below every real
        row in this table.)
      * ``need_gib`` — the TOTAL whole-on-GPU footprint the render actually costs at
        this preset's geometry and default frame budget (DiT + text encoder + VAE +
        denoise workspace), from ``runners.wan_i2v._placement_need_gib`` — the same
        function ``_should_place_whole_on_gpu`` calls. It is NOT a budget input; it is
        provenance, and it is what makes ``fits_render_box`` meaningful: clip-i2v-480p's
        18.0 GB envelope looks affordable while its real need (29.20 GiB) exceeds the
        only render box this fleet has, which is exactly why that row is proven=False.

    NEVER RAISES. This is a read-only discovery route; a registry row that has moved on,
    a model with no measured footprint, or an ffmpeg preset with no geometry at all all
    answer ``None`` rather than 500 a menu. ``_placement_need_gib`` is pure arithmetic
    (no torch/diffusers — wan_i2v keeps the heavy stack lazy inside the runner), so this
    import costs nothing but the module load.
    """
    envelope_gb = None
    need_gib = None
    fits = None
    try:
        from hugpy_video.intel.studio.registry import MODEL_REGISTRY
        cfg = MODEL_REGISTRY.get(p.model_id)
        if cfg is not None:
            envelope_gb = cfg.vram.as_map().get(p.precision)
    except Exception:  # pragma: no cover - a discovery route never fails on provenance
        envelope_gb = None
    try:
        if p.width and p.height and p.default_frames:
            from hugpy_video.intel.studio.runners.wan_i2v import _placement_need_gib
            from hugpy_video.intel.studio import presets as _rp
            need_gib = _placement_need_gib(
                p.model_id, p.precision, p.width, p.height, p.default_frames,
            )
            if need_gib is not None:
                fits = need_gib <= _rp.RENDER_BOX_VRAM_GIB
    except Exception:  # pragma: no cover
        need_gib = None
        fits = None
    return envelope_gb, need_gib, fits


def _render_preset_row(p):
    """Wire shape for one ``RenderPreset``. A projection of the frozen dataclass —
    every value is read off ``p``, none is restated here.

    Keys follow the sibling preset surfaces where they overlap (``id``, not the
    dataclass's ``preset_id``) and the registry's own naming where they do not
    (``title``, ``evidence``). ``capability`` is the PRIMARY one a caller asks for;
    ``capabilities`` is the full set — every ratified row serves exactly one today,
    and the plural key is kept because the row that served FOUR (clip-control-480p)
    turned out to serve one and render a fifth thing, which is the lesson this shape
    should keep visible. ``geometry`` is the registry's own human string — "source"
    on the two ffmpeg enhance presets, which have no geometry of their own (a 0 there
    would be a lie with a shape)."""
    envelope_gb, need_gib, fits_box = _render_preset_placement(p)
    return {
        "id": p.preset_id,
        "title": p.title,
        "description": p.description,
        "capability": p.capability.value,
        "capabilities": [c.value for c in p.capabilities],
        "model": p.model_id,
        "framework": p.framework.value,
        "task": p.task.value,
        "precision": p.precision.value,
        "geometry": p.geometry,
        "width": p.width,
        "height": p.height,
        "fps": p.fps,
        "default_frames": p.default_frames,
        "max_frames": p.max_frames,
        "inputs": list(p.inputs),
        # PROVEN says this exact path has produced pixels on this fleet, and it is
        # CHECKED (tests/studio/test_presets.py proves it against the clip store, and
        # against the rule that a composite can never out-prove what it composes).
        # False on three rows today — clip-i2v-480p (0 of 47 landed Wan clips came
        # from any 14B row), movie-480p (its `still` joint binds that same 14B), and
        # clip-motion-480p (a real VACE branch the route only unlocked on 2026-07-27,
        # so nothing has come through it yet). Surfacing it is the point: a caller may
        # prefer a proven path, and now can tell which those are.
        "proven": p.proven,
        "evidence": p.evidence,
        # MOVIE ONLY; empty tuples on every single-clip preset, kept in the shape so
        # the row is uniform and a client never branches on key presence.
        "composes": list(p.composes),
        "joints": list(p.joints),
        # ── PLACEMENT (added 2026-07-29 so a console stops mirroring these by hand) ──
        # See _render_preset_placement for what each one answers and which line of the
        # router/runner it is derived from. All three may be null: the ffmpeg enhance
        # rows have no geometry to price, and a model with no measured footprint has no
        # derivable need. Null means "not derivable here", never 0.
        #
        # vram_envelope_gb is THE number a caller's `vram_budget_gb` is compared against
        # (router.py: cfg.vram.fits(budget) is `gb <= budget`), so it is the MINIMUM
        # budget that can bind this preset's model at this precision.
        "vram_envelope_gb": envelope_gb,
        # vram_need_gib is the whole-on-GPU footprint at this row's geometry + default
        # frames — provenance for fits_render_box, NOT a budget input. Feeding it into a
        # budget field would inflate the budget past cheaper real models and let a bigger
        # one win the router's fit test, which is the opposite of what a caller wants.
        "vram_need_gib": need_gib,
        "fits_render_box": fits_box,
    }


@video_bp.route("/video/render/presets", methods=["GET"])
def video_render_presets():
    from hugpy_video.intel.studio import presets as render_presets

    # Sorted by capability value so the refusal block is byte-stable across calls —
    # the same reason ``presets.available_menu`` sorts (a payload that reorders itself
    # looks like a change to anyone diffing it). The preset list keeps the registry's
    # RATIFIED order (clips, then movie, then enhance), which is not alphabetical and
    # is meaningful, so it is passed through untouched.
    unavailable = []
    for cap in sorted(render_presets.unservable_capabilities(), key=lambda c: c.value):
        verdict = render_presets.capability_verdict(cap)
        unavailable.append({
            "capability": verdict.capability.value,
            "reason": verdict.reason,
            "refusal": verdict.refusal,
        })

    return jsonify({
        "presets": [_render_preset_row(p) for p in render_presets.all_presets()],
        "unavailable": unavailable,
        # The one-line menu the refusals quote, surfaced on its own so a client can
        # show "what IS available" without parsing it back out of a refusal string.
        "menu": render_presets.available_menu(),
        # Frame-budget arithmetic a caller needs BEFORE spending 6 minutes of denoise:
        # every Wan pipeline requires num_frames == 4k+1 (the latent VAE compresses
        # time 4:1), and the runner silently snaps down to the nearest such value.
        "frame_cadence": render_presets.WAN_FRAME_CADENCE,
        # WHERE these numbers were sized. Not a routing input — provenance, so a
        # reader can tell which box the VRAM figures in `evidence` refer to.
        "render_box": render_presets.RENDER_BOX,
        # The capacity every row's `fits_render_box` was tested against, published
        # beside it so a caller can re-derive the verdict instead of trusting a bool.
        "render_box_vram_gib": render_presets.RENDER_BOX_VRAM_GIB,
    }), 200


# --------------------------------------------------------------------------- #
# 2l) GET /video/prompt/assist/models — WHICH TEXT GENERATORS THIS FLEET CAN
#     ACTUALLY RUN, for the studio's Enhance/Generate picker.
#
#     Same doctrine as /video/render/presets: offer only what completes. A picker
#     built from the raw catalog would be a menu of failures — most rows are
#     catalog-only and 404 on use, and the catalog additionally MIS-TAGS at least
#     one image model as text-generation (see _ASSIST_MISTAGGED below). So a row is
#     offered here only when a live worker actually holds it.
#
#     `serving` distinguishes "seated right now, answers in ~1s" from "present on a
#     worker, first call pays a cold load". That difference is minutes, so the UI
#     should show it rather than let a user wonder if Enhance hung.
# --------------------------------------------------------------------------- #
# Catalog rows that DECLARE text-generation but are not text models. Keyed with the
# evidence, because the next person will otherwise "fix" the omission.
_ASSIST_MISTAGGED = {
    "Flux-Uncensored-V2": ("a Flux IMAGE LoRA mis-tagged text-generation in the "
                           "catalog — same class as 41f908d (a video diffusion model "
                           "advertising itself as a chat model)"),
}


@video_bp.route("/video/prompt/assist/models", methods=["GET"])
def video_prompt_assist_models():
    from hugpy_fleet.central.workers import worker_store
    from hugpy_engine.config.models.models_config import get_models_dict

    catalog = get_models_dict(dict_return=True) or {}

    # Which models a LIVE worker holds, and which are seated right now. Only online
    # workers count: a model on an offline box is not a thing a user can pick today.
    # Per-key hottest residency state across ONLINE workers (canonical
    # STATE-MODEL.md #4): serving (answering) > loaded (in VRAM idle) >
    # hot (on the worker hot drive, pays a t_load) > cold (central only, pays a
    # download+load). "cold" is RESERVED for not-on-hot-drive — a seated/on-disk
    # model must never read cold.
    _RANK = {"cold": 0, "hot": 1, "loaded": 2, "serving": 3}
    held: dict[str, str] = {}

    def _mark(key: object, st: str) -> None:
        k = str(key or "")
        if k and (k not in held or _RANK[st] > _RANK[held[k]]):
            held[k] = st

    try:
        for w in worker_store.all():
            if str(getattr(w, "status", None) or (w.get("status") if isinstance(w, dict) else "")) != "online":
                continue
            row = w if isinstance(w, dict) else getattr(w, "__dict__", {}) or {}
            local = {str(x) for x in (row.get("models_local") or ())}
            for key in (row.get("models") or ()):
                _mark(key, "hot" if str(key) in local else "cold")
            for alloc in (row.get("allocations") or ()):
                key = alloc.get("model_key")
                if alloc.get("serving"):
                    _mark(key, "serving")
                elif alloc.get("healthy") or alloc.get("materialized"):
                    _mark(key, "loaded")
                else:                       # allocated but not measured-resident => at least on drive
                    _mark(key, "hot")
    except Exception:                       # noqa: BLE001 — a picker must never 5xx
        logger.exception("assist model discovery: worker enumeration failed")

    models, excluded = [], []
    for key, state in sorted(held.items()):
        cfg = catalog.get(key)
        if cfg is None:
            continue                        # held but not in the catalog: not offerable
        tasks = list((cfg.get("tasks") if isinstance(cfg, dict) else None) or ())
        if "text-generation" not in tasks:
            continue
        why = _ASSIST_MISTAGGED.get(key)
        if why:
            excluded.append({"model": key, "reason": why})
            continue
        # Operator-blocked models are not offerable: listing one lets a stored
        # browser pick keep resending a key the resolver will always refuse.
        try:
            from hugpy_fleet.central.blocklist import block_reason, is_blocked

            if is_blocked(key):
                excluded.append({
                    "model": key,
                    "reason": block_reason(key) or "blocked by the operator",
                })
                continue
        except Exception:               # noqa: BLE001 — a picker must never 5xx
            logger.exception("assist model discovery: blocklist check failed")
        models.append({
            "model": key,
            "state": state,                 # canonical serving|loaded|hot|cold (STATE-MODEL.md #4)
            "serving": state == "serving",  # SERVING = actively answering
            "framework": (cfg.get("framework") if isinstance(cfg, dict) else None),
            "default": key == _DEFAULT_PROMPT_ASSIST_MODEL,
        })

    return jsonify({
        "models": models,
        "default": _DEFAULT_PROMPT_ASSIST_MODEL,
        # Surfaced, not silently dropped: a row we deliberately refuse to offer and
        # why. Silent omission is how a mis-tag survives for months.
        "excluded": excluded,
        # Every generator served through this route runs with thinking suppressed
        # (see utils/no_think.py) — the picker can say so instead of implying that
        # choosing a reasoning model will fill the prompt box with a monologue.
        "thinking_suppressed": True,
    }), 200


# --------------------------------------------------------------------------- #
# 2z) GET /video/jobs — bus-wide LISTING for the console-wide "Active Processes"
#     view. In-flight media-bus jobs by default (queued/claimed/running/
#     cancelling); ?all=1 appends recent terminal rows (bounded). Each row carries
#     its parsed `progress` (incl. the awaiting_capacity HOLD marker) and a
#     `placement` object — {source:"reservation"|"template", host, worker_id, gpu,
#     process, reserved_bytes} (omit-when-unset) — so the console can show WHERE a
#     run physically executes (e.g. "ae · cuda:0 · P-studio"). Read-only; never
#     5xxes (a bus/placement hiccup degrades to fewer rows / no placement).
#
#     Each row also carries what the Active panel needs to be INFORMATIVE (k57):
#     `progress_ratio` (0..1, null when unmeasurable) + `progress_detail`
#     (segment i/N, step i/N, fraction), a `stage_log` TAIL (+ `stage_log_total`;
#     terminal rows keep the full timeline), the verbatim `failure` envelope
#     (code/message/stage/retryable), and `progressed_at`. Rows abandoned by a dead
#     process (no movement for HUGPY_MEDIA_BUS_STALE_SECONDS, default 6h) are
#     hidden; ?stale=1 shows them, flagged `stale` + `stale_for_s`.
#
#     NOTE: this bare-path route MUST be registered on the blueprint so it wins
#     over the SPA catch-all (`@app.route("/<path:asset>")`): Werkzeug ranks a
#     static rule above a <path:> converter regardless of registration order, so
#     GET /video/jobs resolves here, not to index.html. (The per-id sibling
#     /video/jobs/<job_id> below is a distinct, more-specific rule.)
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs", methods=["GET"])
def video_jobs_list():
    def _flag(name):
        return (request.args.get(name) or "").strip().lower() in (
            "1", "true", "yes", "on")

    include_terminal = _flag("all")
    include_stale = _flag("stale")
    try:
        limit = int(request.args.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    # VISIBILITY (t172): public-by-default. A member sees every PUBLIC job (any
    # owner) PLUS their OWN private jobs; an admin/operator/open-mode/share/no-
    # account caller sees the whole bus (and may narrow with ?owner=<name>). This
    # supersedes the pre-t172 strict own-only scope — the media mirror of the
    # identity-profiles listing, preserving the accepted public movie listing.
    mode, value = _media_view_scope()
    scope_kwargs = {}
    if mode == "owner":
        scope_kwargs["owner"] = value
    elif mode == "viewer":
        scope_kwargs["viewer"] = value
    jobs = []
    try:
        rows = media_bus.list_jobs(include_terminal=include_terminal, limit=limit,
                                   include_stale=include_stale, **scope_kwargs)
    except Exception:  # noqa: BLE001 — the listing never 5xxes
        logger.debug("video jobs listing failed", exc_info=True)
        rows = []
    # ONE placement snapshot for the whole page (k57). The per-row job_placement
    # this replaces consulted the reservation store per row — and that read took a
    # WRITE lock (its lapsed-lease sweep) on a store live renderers heartbeat into,
    # plus a fresh measured.json read per row. With renders in flight the listing
    # serialized behind those locks and hung past 60s, starving the Active panel of
    # the feed that carries its logs, progress and errors.
    try:
        snapshot = PlacementSnapshot()
    except Exception:  # noqa: BLE001 — placement is observability, never a 5xx
        logger.debug("placement snapshot failed", exc_info=True)
        snapshot = None
    for row in rows:
        if snapshot is not None:
            pl = snapshot.placement_for(row.get("job_id"), row.get("name"))
            if pl:
                row["placement"] = pl
        jobs.append(row)
    return jsonify({"jobs": jobs}), 200


# --------------------------------------------------------------------------- #
# 3) GET /video/jobs/<job_id> — read-only job view (unknown id -> null view)
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/<job_id>", methods=["GET"])
def video_job_status(job_id):
    # media_bus.get returns {"job_id","name":null,"status":null,"result":null,
    # "progress":null} for an unknown id, so the poller can distinguish
    # "not yet / unknown" from a real status. Enriched with a `placement` object
    # (same helper as GET /video/jobs / /video/studio/clips) when known — WHERE the
    # run executes — set only when present, so a light/unknown job is unaffected.
    #
    # OWNERSHIP (2026-08-06): reading ONE job by id is the same disclosure as
    # listing it (the view carries the spec's prompts and the result's paths), so
    # it obeys the same rule — owner or admin. The probe runs BEFORE the full
    # read so a refused caller never pays for (or touches) a megabyte result blob.
    found, job_owner, job_private = media_bus.visibility_of(job_id)
    if found and not _may_view_media(job_owner, job_private):
        return _deny_media(job_owner, job_private)
    view = media_bus.get(job_id)
    try:
        pl = job_placement(job_id, view.get("name") if isinstance(view, dict) else None)
        if pl and isinstance(view, dict):
            view["placement"] = pl
    except Exception:  # noqa: BLE001 — enrichment never breaks the status read
        pass
    return jsonify(view), 200


# --------------------------------------------------------------------------- #
# 3b) POST /video/jobs/<job_id>/cancel — cooperative cancel
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/<job_id>/cancel", methods=["POST"])
def video_job_cancel(job_id):
    # queued jobs die outright; a running scene stops BETWEEN frames (mid-frame
    # inference is never interrupted). Idempotent — cancelling a terminal or
    # unknown job reports cancelled=False.
    #
    # OWNERSHIP (2026-08-06): cancelling is a MUTATION of someone's render, so it
    # follows the same owner-or-admin rule as reading it. (Cancelling your OWN
    # job stays a member action — this route is deliberately not in the console's
    # operator-only inventory.)
    found, job_owner, job_private = media_bus.visibility_of(job_id)
    if found and not _owns_media(job_owner):
        return _deny_media(job_owner, job_private)
    return jsonify(media_bus.cancel(job_id)), 200


# --------------------------------------------------------------------------- #
# 3c) POST /video/jobs/<job_id>/private — t172 owner-only VISIBILITY toggle for a
#     generated-media job (a movie / render / image-video gen OUTPUT). Body
#     {"private": bool}: flip the media record PUBLIC<->PRIVATE. Owner or the
#     operator/admin tier ONLY (mirror of the identity-profile toggle) — a share
#     guest may VIEW public media but never change its visibility, and a foreign
#     PRIVATE job 404s (existence hidden). Returns {job_id, private}. This is the
#     per-item toggle the media library uses; a bare gen enqueue sets the initial
#     visibility via the `private` flag (see _video_enqueue).
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/<job_id>/private", methods=["POST"])
def video_job_set_private(job_id):
    found, job_owner, job_private = media_bus.visibility_of(job_id)
    if not found:
        return jsonify({"error": "media not found"}), 404
    if not _owns_media(job_owner):
        return _deny_media(job_owner, job_private)
    body = request.get_json(silent=True) or {}
    if "private" not in body or not isinstance(body.get("private"), bool):
        return jsonify({"error": "private (boolean) is required"}), 400
    result = media_bus.set_media_private(job_id, body["private"])
    if not result.get("found"):  # lost a race with a concurrent reap/quarantine
        return jsonify({"error": "media not found"}), 404
    return jsonify({"job_id": job_id, "private": result["private"]}), 200


# --------------------------------------------------------------------------- #
# 4) GET /video/media?handle=<abspath> — serve raw bytes for a MediaRef uri
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# RAW-PATH OWNERSHIP (2026-08-06) for GET /video/media?handle=<abspath>.
#
# The jail (_jail_resolve) answers "is this path inside our storage roots" — it
# has never answered "is this path YOURS", so any caller who could pass the
# /video gate could stream any other account's clip, frame or upload by path.
# These helpers are that second question, in two cheap layers:
#
#   1. UPLOADS NAMESPACE — a member's uploads now land in UPLOADS_HOME/<username>/
#      (upload_routes), so the first path segment IS the owner. Reserved dirs
#      (.sessions, generated) are NOT namespaces and fall through to layer 2.
#   2. THE JOB CATALOG — "does a job I OWN reference this path?", asked as a
#      LIKE over the requesting member's OWN rows only (owner = ? is indexed),
#      so it never scans the whole catalog. Matches the exact path and, for
#      sidecars (manifest.json, extracted frames), anything under the same
#      directory as a path the job references.
#
# Answer cached per (viewer, path) for 5 minutes — a <video> element re-fetches
# the same handle on every seek/range request.
#
# ADMIN sees everything, as everywhere else. A path that neither layer can
# attribute (a legacy flat upload, a pre-ownership artifact) is ADMIN-ONLY.
# --------------------------------------------------------------------------- #
_UPLOAD_RESERVED_DIRS = frozenset({".sessions", "generated"})
_PATH_OWNER_TTL = 300.0
_PATH_OWNER_MAX = 1024
_PATH_OWNER_CACHE: dict = {}


def _uploads_namespace(resolved):
    """The username namespace a path sits in under UPLOADS_HOME, or None (not an
    upload, a flat legacy upload, or a reserved dir)."""
    if not _is_within(resolved, UPLOADS_HOME):
        return None
    try:
        rel = os.path.relpath(resolved, os.path.realpath(UPLOADS_HOME))
    except ValueError:
        return None
    parts = rel.split(os.sep)
    if len(parts) < 2:
        return None                       # a flat file directly in UPLOADS_HOME
    if parts[0] in _UPLOAD_RESERVED_DIRS:
        return None
    return parts[0]


def _my_upload_namespace(username):
    """This viewer's namespace directory name — derived by the SAME helper the
    upload writer uses (operator_auth.upload_namespace), so the two can never
    drift on the mapping from username to directory."""
    try:
        from hugpy_server.app.operator_auth import upload_namespace
        return upload_namespace(username)
    except Exception:  # noqa: BLE001 — fail closed
        return None


def _like_escape(value: str) -> str:
    """Escape LIKE metacharacters so a path containing % or _ can never
    over-match (an over-match here would GRANT access)."""
    return (value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_"))


def _member_owns_path(username: str, resolved: str) -> bool:
    """True iff a media job OWNED BY ``username`` references this path (or the
    directory it lives in). Scoped by ``owner = ?`` so the scan is bounded to
    that member's own rows via the owner index."""
    import sqlite3
    key = (username, resolved)
    now = time.time()
    hit = _PATH_OWNER_CACHE.get(key)
    if hit and hit[1] > now:
        return hit[0]
    exact = f"%{_like_escape(resolved)}%"
    parent = f"%{_like_escape(os.path.dirname(resolved) + os.sep)}%"
    allowed = False
    try:
        media_bus._ensure_db()
    except Exception:  # noqa: BLE001
        pass
    try:
        conn = sqlite3.connect(
            f"file:{media_bus.DB_PATH}?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            row = conn.execute(
                "SELECT 1 FROM media_jobs WHERE owner = ? AND ("
                "  result_json LIKE ? ESCAPE '\\' OR spec_json LIKE ? ESCAPE '\\'"
                "  OR result_json LIKE ? ESCAPE '\\' OR spec_json LIKE ? ESCAPE '\\'"
                ") LIMIT 1",
                (username, exact, exact, parent, parent),
            ).fetchone()
            allowed = row is not None
        finally:
            conn.close()
    except sqlite3.Error:
        # A locked/absent DB must not silently GRANT — fail closed.
        logger.debug("media handle ownership lookup failed", exc_info=True)
        allowed = False
    if len(_PATH_OWNER_CACHE) > _PATH_OWNER_MAX:
        _PATH_OWNER_CACHE.clear()
    _PATH_OWNER_CACHE[key] = (allowed, now + _PATH_OWNER_TTL)
    return allowed


def _public_job_references_path(resolved: str) -> bool:
    """True iff a PUBLIC media job's OUTPUT references this path (or its directory).
    The t172 public-visibility counterpart of ``_member_owns_path``, scoped to
    ``COALESCE(private,0)=0`` (a legacy/NULL row is PUBLIC) and to ``result_json``
    (the produced OUTPUT) ONLY — NOT ``spec_json`` — so a public render exposes the
    public artifact it produced, never the (possibly private) input SOURCES it
    consumed. Cached like _member_owns_path (key namespaced so it never collides
    with a member entry). Uploads are resolved BEFORE this in _may_serve_handle, so
    a member's private upload path never reaches here."""
    import sqlite3
    key = ("__public__", resolved)
    now = time.time()
    hit = _PATH_OWNER_CACHE.get(key)
    if hit and hit[1] > now:
        return hit[0]
    exact = f"%{_like_escape(resolved)}%"
    parent = f"%{_like_escape(os.path.dirname(resolved) + os.sep)}%"
    allowed = False
    try:
        media_bus._ensure_db()
    except Exception:  # noqa: BLE001
        pass
    try:
        conn = sqlite3.connect(
            f"file:{media_bus.DB_PATH}?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            row = conn.execute(
                "SELECT 1 FROM media_jobs WHERE COALESCE(private,0)=0 AND ("
                "  result_json LIKE ? ESCAPE '\\' OR result_json LIKE ? ESCAPE '\\'"
                ") LIMIT 1",
                (exact, parent),
            ).fetchone()
            allowed = row is not None
        finally:
            conn.close()
    except sqlite3.Error:
        # A locked/absent DB must not silently GRANT — fail closed.
        logger.debug("media handle public lookup failed", exc_info=True)
        allowed = False
    if len(_PATH_OWNER_CACHE) > _PATH_OWNER_MAX:
        _PATH_OWNER_CACHE.clear()
    _PATH_OWNER_CACHE[key] = (allowed, now + _PATH_OWNER_TTL)
    return allowed


def _may_serve_handle(resolved: str) -> bool:
    unscoped, username = _viewer()
    if unscoped:
        return True            # admin / operator-token / open mode / share link
    # A MEMBER. Their OWN uploads (namespaced) + their own jobs' paths, as before;
    # PLUS (t172) any path a PUBLIC media job PRODUCED — public-by-default, so a
    # member can stream another account's public movie/clip by uri (cross-station
    # library playback), exactly as the by-id serve routes now allow. Uploads are
    # resolved FIRST and never fall through to the public probe, so this never
    # widens access to a member's PRIVATE uploads.
    ns = _uploads_namespace(resolved)
    if ns is not None:
        return username is not None and ns == _my_upload_namespace(username)
    if username and _member_owns_path(username, resolved):
        return True
    return _public_job_references_path(resolved)


@video_bp.route("/video/media", methods=["GET"])
def video_media():
    handle = request.args.get("handle")
    resolved = _jail_resolve(handle)
    if resolved is None or not os.path.isfile(resolved):
        return jsonify({"error": "not found"}), 404
    # OWNERSHIP, after the jail (2026-08-06). The jail says the path is inside
    # our storage; this says it is the caller's to read.
    if not _may_serve_handle(resolved):
        return jsonify({"error": "forbidden: artifact belongs to another account"}), 403
    mime, _ = mimetypes.guess_type(resolved)
    if mime is None:
        # A studio clip can be served here BY URI (cross-station library playback) from
        # an extensionless media-store path — guess_type then returns None and the
        # generic octet-stream fallback makes the console <video> show a gray unknown
        # mime. A file under the studio clips dir is always an mp4, so prefer that.
        from hugpy_video.intel.studio.job import DEFAULT_CLIPS_ROOT
        if _is_within(os.path.realpath(resolved), DEFAULT_CLIPS_ROOT):
            mime = "video/mp4"
    return send_file(
        resolved,
        mimetype=mime or "application/octet-stream",
        conditional=True,
    )


# --------------------------------------------------------------------------- #
# 5) GET /video/studio/clip/<job_id> — STREAM a produced studio clip (slice #3).
#     Convenience twin of /video/media for studio i2v: resolve the clip path from
#     the media-bus job result (the content-addressed clip cataloged by the studio
#     runner) BY JOB ID, so the console viewer plays a clip without ever handling a
#     filesystem path. Range-aware (send_file conditional=True) so an HTML5 <video>
#     can seek. Path-traversal guard: the resolved realpath MUST live under the
#     studio clips dir — a job whose output escapes that tree (or any non-studio /
#     non-done job) is refused, so this can only ever serve a cataloged studio clip.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/studio/clips", methods=["GET"])
def video_studio_clips():
    # DURABLE recent-clips list for the console viewer (slice #3). Sources the media
    # CATALOG (the media-bus job store) rather than the comms /llm/jobs view, which
    # only retains terminal rows for ~600s — so a clip produced an hour ago still
    # lists here. Read-only projection: job_id + status + the first output's display
    # metadata (asset_id/geometry/duration). The clip bytes are NEVER referenced by
    # path in the response — playback is by job_id through /video/studio/clip/<id>,
    # which owns the jail. Reads media_bus's own DB_PATH via a read-only connection
    # (mirrors media_bus.get's read); it does not mutate the bus or its schema.
    #
    # `archived_at IS NULL` excludes ARCHIVED clips (POST .../archive) — this is
    # the fix for "removed clips just reappear": this list IS the catalog (it is
    # DB-driven, not a filesystem walk), so a clip hidden here via media_bus.archive
    # stays hidden across every poll, unlike the old client-only "remove" the UI
    # used to do (which this same query's next 6s tick silently resurrected).
    import json as _json
    import sqlite3

    try:
        limit = int(request.args.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, 200))

    # Guarantee the schema (progress_json / archived_at / stage_log_json columns)
    # exists before the read-only SELECT below — a RO connection cannot ALTER, so on
    # a fresh process that has not yet run a write path the column would be missing
    # and the whole SELECT would raise (empty list). _ensure_db is idempotent + cheap.
    try:
        media_bus._ensure_db()
    except Exception:  # noqa: BLE001 — never let migration break the listing
        pass

    # VISIBILITY (t172): public-by-default. A member's poll returns every PUBLIC
    # clip (any owner) PLUS their OWN private clips; an admin's query is unscoped
    # (or narrowed by ?owner=<name>). `COALESCE(private,0)=0` treats a legacy/NULL
    # row as PUBLIC (the non-destructive migration). Supersedes the strict own-only
    # `AND owner = ?` scope — a member sees every public render, not only theirs.
    scope_clause, scope_params = _media_scope_sql()

    clips = []
    try:
        conn = sqlite3.connect(
            f"file:{media_bus.DB_PATH}?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            base_sql = (
                "SELECT job_id, status, result_json, created, updated, progress_json, "
                "stage_log_json "
                "FROM media_jobs WHERE name='studio_i2v' AND archived_at IS NULL ")
            if scope_clause:
                rows = conn.execute(
                    base_sql + f"AND {scope_clause} ORDER BY updated DESC LIMIT ?",
                    (*scope_params, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    base_sql + "ORDER BY updated DESC LIMIT ?", (limit,),
                ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        # No DB yet / transient lock (or a not-yet-migrated archived_at column on
        # a DB no write path has touched this process) -> an empty list is the
        # honest answer, same posture as before this feature.
        rows = []

    # One snapshot for the page, not one reservation-store round trip per clip —
    # same k57 fix as GET /video/jobs (a per-row placement read took a write lock
    # renderers contend for; a 200-clip page paid for it 200 times).
    try:
        snapshot = PlacementSnapshot()
    except Exception:  # noqa: BLE001 — placement never breaks the clips listing
        logger.debug("placement snapshot failed", exc_info=True)
        snapshot = None

    for job_id, status, result_json, created, updated, progress_json, stage_log_json in rows:
        out = None
        res = None
        if result_json:
            try:
                res = _json.loads(result_json)
                outputs = res.get("outputs") or []
                if outputs and isinstance(outputs[0], dict):
                    o = outputs[0]
                    out = {
                        "asset_id": o.get("asset_id"),
                        # The clip's abs path — so the console can build a full video
                        # MediaRef for the Session Library + "Send to Studio" chain
                        # (B2). It is the same uri the movie/scene job results already
                        # expose on /video/jobs/<id>; playback still goes by job_id
                        # through /video/studio/clip/<id>, which owns the jail.
                        "uri": o.get("uri"),
                        "mime": o.get("mime"),
                        "width": o.get("width"),
                        "height": o.get("height"),
                        "duration_s": o.get("duration_s"),
                    }
            except (ValueError, TypeError):
                out = None
        # Additive honesty (Active Processes): the live progress blob (incl. the
        # awaiting_capacity HOLD marker) + a placement object (WHERE the render
        # executes). Existing fields are untouched. `progress` is null unless the
        # runner has written one; `placement` is omitted unless known.
        progress = None
        if progress_json:
            try:
                progress = _json.loads(progress_json)
            except (ValueError, TypeError):
                progress = None
        # STAGE TIMELINE + terminal FAILURE summary + stall basis (the exhaustive
        # per-process telemetry). RETAINED through terminal, so a failed/cancelled
        # row still carries its full timeline + the exact failing stage/code/message.
        # Additive — every existing key above is untouched.
        stage_log = media_bus._load_stage_log(stage_log_json)
        clip = {
            "job_id": job_id,
            "status": status,
            "playable": bool(status == "done" and out),
            "created": created,
            "updated": updated,
            "output": out,
            "progress": progress,
            "stage_log": stage_log,
            "failure": media_bus.build_failure_summary(res, stage_log),
            "last_movement_ts": media_bus._last_movement_ts(stage_log, updated),
            "current_stage": media_bus._current_stage(stage_log),
        }
        if snapshot is not None:
            pl = snapshot.placement_for(job_id, "studio_i2v")
            if pl:
                clip["placement"] = pl
        clips.append(clip)

    return jsonify({"clips": clips}), 200


@video_bp.route("/video/studio/clip/<job_id>", methods=["GET"])
def video_studio_clip(job_id):
    # Canonical clips root — imported LAZILY (mirrors the runner's lazy studio-spine
    # imports) so this module stays app-boot cheap and drift-free with studio.job.
    from hugpy_video.intel.studio.job import DEFAULT_CLIPS_ROOT

    # Archived clips are HIDDEN from GET /video/studio/clips but their bytes are
    # NEVER deleted (never-delete doctrine) — a direct fetch by id gets an HONEST
    # 410 naming "archived", not a bare 404 that reads as "this never existed".
    # OWNERSHIP (2026-08-06): streaming a clip by id is the artifact itself, so
    # it is owner-or-admin. Checked BEFORE the archived probe/full read, so a
    # foreign clip cannot even be distinguished as archived.
    found, clip_owner, clip_private = media_bus.visibility_of(job_id)
    if found and not _may_view_media(clip_owner, clip_private):
        return _deny_media(clip_owner, clip_private)

    if media_bus.is_archived(job_id):
        return jsonify({"error": "clip archived", "archived": True}), 410

    view = media_bus.get(job_id)              # {"status","result",...} — unknown -> nulls
    result = view.get("result") if isinstance(view, dict) else None
    if not (isinstance(result, dict) and result.get("ok")):
        # unknown / queued / running / failed / cancelled — no playable clip yet
        return jsonify({"error": "no completed studio clip for that job"}), 404

    outputs = result.get("outputs") or []
    first = outputs[0] if (outputs and isinstance(outputs[0], dict)) else {}
    uri = first.get("uri")
    if not uri or not isinstance(uri, str):
        return jsonify({"error": "job result carries no clip uri"}), 404

    # Jail: only serve a file that really lives under the studio clips tree. This is
    # the single seam that keeps a crafted/rehomed uri from becoming an arbitrary
    # file read — it is checked on the REALPATH, so symlinks/.. can't escape.
    resolved = os.path.realpath(uri)
    if not _is_within(resolved, DEFAULT_CLIPS_ROOT) or not os.path.isfile(resolved):
        return jsonify({"error": "clip not found"}), 404

    # EXPLICIT Content-Type — NEVER filename guessing. A content-addressed studio clip
    # can be served from a uri WITHOUT a .mp4 extension (media-store ingest names some
    # assets by id), and send_file's extension guess then yields no/an unknown type, so
    # the console <video> shows a gray "unknown mime" error. Source the mime from the
    # catalog record (outputs[].mime, already "video/mp4"); fall back to "video/mp4" for
    # anything under the studio clips dir (every file the jail above admits is a clip).
    mime = first.get("mime")
    if not (isinstance(mime, str) and mime.strip()):
        mime = "video/mp4"
    return send_file(
        resolved,
        mimetype=mime,
        conditional=True,   # HTTP Range + conditional requests => <video> can seek
    )


# --------------------------------------------------------------------------- #
# 5b) GET /video/studio/clip/<job_id>/detail — the exact CREATION PARAMETERS of a
#     studio render, for a list-row "why did this pass/fail" expander. LAZY (fetched
#     on expand, not on the 6s list poll) so the list stays lean, and JAILED like the
#     stream route (the manifest.json is read only from beside a clip that really lives
#     under the studio clips tree). Two data sources, used-not-invented:
#       * DONE  -> the content-addressed manifest.json beside the clip (the TRUE render
#                  params: model_id, precision, resolution, seed, sampler steps/cfg/shift,
#                  prompt/negative, source_video, content_hash) + the output geometry.
#       * FAILED/CANCELLED (or any non-done) -> the job's requested spec + the error
#                  {code, message, retryable} from the job result. No clip, so no manifest.
#     The requested SPEC (from the bus row's spec_json) rides along in every case so the
#     UI can show "asked for X, got Y".
# --------------------------------------------------------------------------- #
@video_bp.route("/video/studio/clip/<job_id>/detail", methods=["GET"])
def video_studio_clip_detail(job_id):
    import json as _json
    import sqlite3
    from hugpy_video.intel.studio.job import DEFAULT_CLIPS_ROOT

    # OWNERSHIP (2026-08-06): the detail view carries the render's prompts,
    # source paths and manifest — the most disclosive read on this surface. Same
    # owner-or-admin rule as the stream route, checked first.
    found, clip_owner, clip_private = media_bus.visibility_of(job_id)
    if found and not _may_view_media(clip_owner, clip_private):
        return _deny_media(clip_owner, clip_private)

    # Same honest 410 as the stream route: an archived clip's row/manifest still
    # exist (never-delete), but the expander should say "archived", not 404.
    if media_bus.is_archived(job_id):
        return jsonify({"error": "clip archived", "archived": True}), 410

    # Read the bus row read-only (mirrors /video/studio/clips) — spec_json carries the
    # REQUESTED params, result_json the outcome. Unknown id -> 404 (nothing to detail).
    # Ensure the schema (stage_log_json) before the RO read (see /video/studio/clips).
    try:
        media_bus._ensure_db()
    except Exception:  # noqa: BLE001
        pass

    row = None
    try:
        conn = sqlite3.connect(
            f"file:{media_bus.DB_PATH}?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            row = conn.execute(
                "SELECT status, spec_json, result_json, created, updated, stage_log_json "
                "FROM media_jobs WHERE job_id=? AND name='studio_i2v'",
                (job_id,),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        row = None
    if row is None:
        return jsonify({"error": "no studio job for that id"}), 404
    status, spec_json, result_json, created, updated, stage_log_json = row
    stage_log = media_bus._load_stage_log(stage_log_json)

    # Curated view of the REQUESTED spec (drop out_root — an internal path). Everything
    # else is a creation parameter worth showing.
    spec_view = None
    if spec_json:
        try:
            s = _json.loads(spec_json)
            spec_view = {k: s.get(k) for k in (
                "capability", "width", "height", "fps", "vram_budget_gb", "seed",
                "prompt", "negative", "steps", "cfg", "model_id",
                "start_image", "source_video")}
        except (ValueError, TypeError):
            spec_view = None

    result = None
    if result_json:
        try:
            result = _json.loads(result_json)
        except (ValueError, TypeError):
            result = None

    error_view = None
    manifest_view = None
    if isinstance(result, dict):
        if result.get("ok"):
            outputs = result.get("outputs") or []
            first = outputs[0] if (outputs and isinstance(outputs[0], dict)) else {}
            uri = first.get("uri")
            if isinstance(uri, str) and uri:
                resolved = os.path.realpath(uri)
                # Same jail as the stream route: only read a manifest beside a clip that
                # really lives under the studio clips tree (checked on the realpath).
                if _is_within(resolved, DEFAULT_CLIPS_ROOT) and os.path.isfile(resolved):
                    mpath = os.path.join(os.path.dirname(resolved), "manifest.json")
                    if _is_within(os.path.realpath(mpath), DEFAULT_CLIPS_ROOT) \
                            and os.path.isfile(mpath):
                        try:
                            with open(mpath, "r", encoding="utf-8") as fh:
                                m = _json.load(fh)
                            manifest_view = _studio_manifest_view(m, resolved, first)
                        except (OSError, ValueError, TypeError):
                            manifest_view = None
        else:
            err = result.get("error") or {}
            if isinstance(err, dict):
                # `stage` = the recorded failing stage (where it broke), from the
                # retained timeline — the whole point of the exhaustive view.
                _fail = media_bus.build_failure_summary(result, stage_log)
                error_view = {
                    "code": err.get("code"),
                    "message": err.get("message"),
                    "retryable": bool(err.get("retryable", False)),
                    "stage": _fail.get("stage") if _fail else None,
                }

    # SOURCE discriminator (coordinator addendum): which record the render params come
    # from. "manifest" = the content-addressed manifest.json beside a produced clip — the
    # TRUE, RESOLVED params (model bound, sampler steps/cfg actually used). "job_record" =
    # only the media-bus row exists: a FAILED / CANCELLED / still-running job wrote NO
    # manifest (only successful renders write the clip dir), so `spec` carries the
    # REQUESTED params (unresolved — e.g. `steps` null means "model default", which was
    # never bound) and `error` carries the failure {code,message,retryable}. The UI reads
    # this to label job_record params as REQUESTED and to explain the row that most needs
    # it. We surface only what the record HOLDS — no resolved sampler value is invented
    # for a job that failed before it bound a model.
    source = "manifest" if manifest_view is not None else "job_record"
    return jsonify({
        "job_id": job_id,
        "status": status,
        "source": source,
        "playable": bool(status == "done" and manifest_view is not None),
        "spec": spec_view,
        "manifest": manifest_view,
        "error": error_view,
        # STAGE TIMELINE (retained through terminal) — the full sequence this render
        # moved through, so a failed/cancelled row in the Library is inspectable with
        # exactly what it did + where it stopped. Additive.
        "stage_log": stage_log,
        "current_stage": media_bus._current_stage(stage_log),
        "last_movement_ts": media_bus._last_movement_ts(stage_log, updated),
        # Bus row timestamps (epoch seconds) — when the job was enqueued / last updated.
        # Present in every case (the row always carries them); the UI shows them on a
        # failed/cancelled row alongside the requested params + error.
        "created": created,
        "updated": updated,
    }), 200


# --------------------------------------------------------------------------- #
# 5c) POST /video/studio/clip/<job_id>/archive + /unarchive — the fix for
#     "removed clips just reappear". GET /video/studio/clips (2j... above) is
#     DB-driven, not a filesystem walk, so a real "remove" has to be a mark THAT
#     query excludes — which is exactly what media_bus.archive does (see its
#     header note). The clip's row and bytes on disk are NEVER touched; archiving
#     only flips archived_at, so the never-delete doctrine holds trivially (there
#     is no delete path to guard against). Idempotent by choice, not 409: a
#     retried/double-clicked archive (the UI archives optimistically, see
#     StudioPlane's Session-Library remove) must read as "already gone", never as
#     a failure that would restore the row and show an error for nothing.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/studio/clip/<job_id>/archive", methods=["POST"])
def video_studio_clip_archive(job_id):
    # OWNERSHIP (2026-08-06): hiding a clip from the library is a write on
    # someone's artifact — owner or admin, same rule as reading it.
    found, clip_owner, clip_private = media_bus.visibility_of(job_id)
    if found and not _owns_media(clip_owner):
        return _deny_media(clip_owner, clip_private)
    result = media_bus.archive(job_id)
    if not result["found"]:
        return jsonify({"ok": False, "error": "no studio job for that id"}), 404
    return jsonify({
        "ok": True,
        "job_id": job_id,
        "archived": True,
        "already": result["already"],
        "archived_at": result["archived_at"],
    }), 200


@video_bp.route("/video/studio/clip/<job_id>/unarchive", methods=["POST"])
def video_studio_clip_unarchive(job_id):
    # The honest counterpart — cheap to add, and it makes archive a REVERSIBLE
    # hide rather than a one-way trapdoor. Same ownership rule as archive.
    found, clip_owner, clip_private = media_bus.visibility_of(job_id)
    if found and not _owns_media(clip_owner):
        return _deny_media(clip_owner, clip_private)
    result = media_bus.unarchive(job_id)
    if not result["found"]:
        return jsonify({"ok": False, "error": "no studio job for that id"}), 404
    return jsonify({
        "ok": True,
        "job_id": job_id,
        "archived": False,
        "already": result["already"],
    }), 200


# --------------------------------------------------------------------------- #
# 5d) POST /video/studio/clip/<job_id>/to-editor — Studio "Send to Editor" (k12).
#     A ONE-CLICK handoff of a produced clip, in a Filmora-native MP4, into a
#     stable inbox (studio.job.EDITOR_INBOX_ROOT) the operator's LAN Windows
#     workstation (Filmora desktop) picks up over a host-side Samba export.
#
#     CONSOLE-OPERATOR ONLY. The blanket /video gate (video_auth.install_video_gate)
#     already admitted EITHER an operator session OR a video-share credential by the
#     time this body runs — but this action WRITES INTO THE OPERATOR'S REAL EDITING
#     FOLDER, so a share-link guest (who CAN pass that gate) must not reach it. The
#     body re-checks operator_authenticated() and 403s otherwise: defense in depth,
#     independent of the blanket gate (mirrors _request_principal's lazy import).
#
#     The clip is resolved BY JOB ID exactly like GET /video/studio/clip/<id>
#     (archived -> 410, no completed clip -> 404, jail on the realpath under the
#     studio clips tree). The FORMAT GUARANTEE (ffprobe -> remux-or-transcode) lives
#     in the backbone (studio.editor_handoff), never inline here, per this module's
#     house rule. RE-SEND is NON-DESTRUCTIVE: unique_path appends _1, _2 … rather
#     than clobbering a prior copy the operator may have open in Filmora mid-edit.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/studio/clip/<job_id>/to-editor", methods=["POST"])
def video_studio_clip_to_editor(job_id):
    # Cheapest check first: CONSOLE OPERATOR ONLY (a share guest is refused here
    # even though it passed the blanket /video gate). Same lazy import idiom as
    # _request_principal() at the top of this module.
    from hugpy_server.app.operator_auth import operator_authenticated
    if not operator_authenticated():
        return jsonify({"error": "operator session required"}), 403

    import json as _json
    import sqlite3
    from hugpy_video.intel.studio.job import DEFAULT_CLIPS_ROOT, EDITOR_INBOX_ROOT
    from hugpy_video.intel.studio.editor_handoff import editor_filename, send_to_editor
    from hugpy_platform.utils import unique_path

    # Resolve the clip exactly like GET /video/studio/clip/<job_id>: an archived
    # clip's bytes still exist (never-delete) -> honest 410, not a bare 404; any
    # non-done/failed job -> no playable clip; the jail on the realpath keeps a
    # crafted/rehomed uri from becoming an arbitrary file read.
    if media_bus.is_archived(job_id):
        return jsonify({"error": "clip archived", "archived": True}), 410

    view = media_bus.get(job_id)
    result = view.get("result") if isinstance(view, dict) else None
    if not (isinstance(result, dict) and result.get("ok")):
        return jsonify({"error": "no completed studio clip for that job"}), 404

    outputs = result.get("outputs") or []
    first = outputs[0] if (outputs and isinstance(outputs[0], dict)) else {}
    uri = first.get("uri")
    if not uri or not isinstance(uri, str):
        return jsonify({"error": "job result carries no clip uri"}), 404

    resolved = os.path.realpath(uri)
    if not _is_within(resolved, DEFAULT_CLIPS_ROOT) or not os.path.isfile(resolved):
        return jsonify({"error": "clip not found"}), 404

    # Filename slug from the render's prompt/project. spec_json carries them; there
    # is no dedicated media_bus getter, so read the bus row read-only, exactly like
    # GET /video/studio/clip/<id>/detail. Best-effort — a slug of "clip" is a fine
    # fallback if the row/spec can't be read.
    title = None
    try:
        conn = sqlite3.connect(
            f"file:{media_bus.DB_PATH}?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            row = conn.execute(
                "SELECT spec_json FROM media_jobs WHERE job_id=? AND name='studio_i2v'",
                (job_id,),
            ).fetchone()
        finally:
            conn.close()
        if row and row[0]:
            s = _json.loads(row[0])
            title = s.get("prompt") or s.get("project")
    except (sqlite3.Error, ValueError, TypeError):
        title = None

    filename = editor_filename(title, job_id)
    os.makedirs(EDITOR_INBOX_ROOT, exist_ok=True)
    dest = unique_path(os.path.join(EDITOR_INBOX_ROOT, filename))

    outcome = send_to_editor(resolved, dest)
    if not outcome.ok:
        return jsonify({
            "error": "could not prepare the clip for the editor",
            "code": outcome.code,
            "message": outcome.message,
        }), 500

    return jsonify({
        "ok": True,
        "filename": os.path.basename(outcome.dest),
        "path": outcome.dest,
        # "copy" (remux) vs "transcode" — surfaced so the console can say which ran.
        "mode": outcome.mode,
    }), 200


def _studio_manifest_view(m: dict, clip_path: str, output: dict) -> dict:
    """Compact projection of a render manifest.json for the clip-detail expander. The
    clip DIR name IS the content_hash (content-addressed storage), so we surface it from
    the path rather than recomputing. ``output`` (the cataloged MediaRef) supplies the
    real duration; frames is DERIVED (duration * fps) for a CFR clip — not invented."""
    ladder = m.get("resolution_ladder") or []
    res = ladder[0] if (ladder and isinstance(ladder[0], (list, tuple))
                        and len(ladder[0]) == 3) else None
    resolution = ({"width": res[0], "height": res[1], "fps": res[2]}
                  if res is not None else None)
    seeds = m.get("seeds") or {}
    duration_s = output.get("duration_s") if isinstance(output, dict) else None
    frames = None
    if resolution is not None and isinstance(duration_s, (int, float)):
        frames = int(round(duration_s * resolution["fps"]))
    return {
        "content_hash": os.path.basename(os.path.dirname(clip_path)),
        "model_id": m.get("model_id"),
        "framework": m.get("framework"),
        "task": m.get("task"),
        "capability": m.get("capability"),
        "precision": m.get("precision"),
        "determinism_class": m.get("determinism_class"),
        "resolution": resolution,
        "duration_s": duration_s,
        "frames": frames,
        "seed": seeds.get("global_seed"),
        "sampler": m.get("sampler"),   # {sampler, scheduler, steps, cfg, shift, sigmas}
        "prompt": m.get("prompt"),
        "negative_prompt": m.get("negative_prompt"),
        "source_video": m.get("source_video"),
    }


# --------------------------------------------------------------------------- #
# 5c) GET /video/projects — the distinct known auto-archive PROJECT names.
#     Read-only projection over the media-bus job store: scans EVERY job's stored
#     spec_json (studio_i2v, generate_movie, generate_scene, generate_image) for
#     distinct non-empty "project" values — the optional human archive NAME threaded
#     through each enqueue route — and returns them sorted case-insensitively. Uses
#     the same jailed mode=ro connection idiom as /video/studio/clips (no writes, no
#     schema touch). A frontend project-picker reads this to offer known names; a
#     job that carried no project contributes nothing (no empty entry).
# --------------------------------------------------------------------------- #
@video_bp.route("/video/projects", methods=["GET"])
def video_projects():
    import json as _json
    import sqlite3

    # Distinct EXACT names (a set); the case-insensitive ordering is the SORT, not the
    # de-dup, so "Alpha" and "alpha" would both list (they are distinct strings) — the
    # frontend contract only pins the shape {"projects": [...]} and the sort.
    # VISIBILITY (t172): the project-name vocabulary a member sees is built from
    # every PUBLIC job (any owner) PLUS their OWN private jobs — public-by-default,
    # matching the clips/jobs listings. Admin sees every project name (and may
    # narrow with ?owner=). A legacy/NULL row is PUBLIC (COALESCE(private,0)=0).
    scope_clause, scope_params = _media_scope_sql()

    names: set = set()
    try:
        conn = sqlite3.connect(
            f"file:{media_bus.DB_PATH}?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            if scope_clause:
                rows = conn.execute(
                    "SELECT spec_json FROM media_jobs "
                    f"WHERE spec_json IS NOT NULL AND {scope_clause}", scope_params,
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT spec_json FROM media_jobs WHERE spec_json IS NOT NULL"
                ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        # No DB yet / transient lock -> an empty list is the honest answer.
        rows = []

    for (spec_json,) in rows:
        if not spec_json:
            continue
        try:
            spec = _json.loads(spec_json)
        except (ValueError, TypeError):
            continue
        if not isinstance(spec, dict):
            continue
        p = spec.get("project")
        if isinstance(p, str):
            p = p.strip()
            if p:
                names.add(p)

    return jsonify({"projects": sorted(names, key=str.lower)}), 200


# --------------------------------------------------------------------------- #
# IDENTITY PROFILES (studio stage (a)) — a NAMED, DURABLE library item:
#   {name, reference_images (1..4), created_at, notes?}
# DOCTRINE (STUDIO-ROADMAP.md "IDENTITY PROFILES"): a profile is the durable form
# of "the reference set IS the identity" — a character's curated reference DNA
# saved ONCE and associated anywhere (single clips, movies, stills) instead of
# being re-supplied per request. This is the LIBRARY-ITEM surface; stage (b)
# (turnaround generation from a profile + the re-edit loop that promotes an
# approved rendering to the canonical reference) layers on top of this store next.
#
# These routes only translate HTTP <-> video_intel.identity_profiles (the store
# owns the single-writer/atomic-write invariant, reused from api_keys). Reference
# images are validated HERE with the studio movie/i2v route's EXACT jail + ingest
# + image-classify pass, then stored as given (durable uploads/store paths).
# Errors-as-data throughout — a bad path is a clean 4xx, never a deferred failure.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# t171 — PER-ACCOUNT OWNERSHIP + PUBLIC/PRIVATE VISIBILITY for identity profiles
# (2026-09-14 security review, finding S4). The /video gate answers "may this
# caller use the studio"; it cannot answer "whose identity profile is this". The
# store carried no owner at all, so any principal past the gate could list, read,
# delete or patch ANY account's profile. These helpers are the second half — the
# SAME owner model the movie owner-gate uses (_request_principal + the operator
# tier), specialized to the profile record's own ``owner`` + ``private`` fields.
#
#   owner    stamped at create from _request_principal(): "operator" |
#            "user:<name>" | "share:<id>" | None (open-mode / anonymous, and every
#            legacy owner-less profile — all read as operator-owned below).
#   private  DEFAULT False == PUBLIC (visible to everyone the /video gate admits);
#            True == visible ONLY to the owner + the operator/admin tier.
#
# VISIBILITY (read):  public -> anyone;  private -> owner or operator/admin.
# MUTATION  (write):  owner or operator/admin ONLY (create/delete/patch/toggle/…),
#           public or private alike — a video-share guest may VIEW public profiles
#           but never mutate one, nor see/consume another account's private one.
# MIGRATION (non-destructive): a legacy profile emits owner=None + private=False
#           -> PUBLIC and mutable only by the operator tier. Nothing is hidden.
# --------------------------------------------------------------------------- #
def _owns_profile(owner) -> bool:
    """Is THIS request the profile's owner, or the operator/admin tier? The
    operator tier (operator token / open mode / admin / valid hp_ key) owns EVERY
    profile including legacy NULL-owner rows; any other caller must present the
    exact principal stored as ``owner`` (a member "user:<name>", a share
    "share:<id>"). The single MUTATION predicate and the private-view predicate."""
    try:
        from hugpy_server.app.operator_auth import operator_authenticated
        if operator_authenticated():
            return True
    except Exception:  # noqa: BLE001 — an auth-layer hiccup must never GRANT ownership
        pass
    me = _request_principal()
    if not me or not owner:
        return False
    return owner == me


def _may_view_profile(profile) -> bool:
    """Read visibility for one profile dict: a PUBLIC profile (``private`` false —
    the default, and every legacy profile) is visible to everyone the /video gate
    admitted; a PRIVATE one only to its owner + the operator/admin tier."""
    if not isinstance(profile, dict):
        return False
    if not profile.get("private"):
        return True
    return _owns_profile(profile.get("owner"))


def _deny_profile(profile):
    """The deny response for a caller that may not MUTATE (or, for a private
    profile, may not even VIEW) this profile. 403 when the caller can still SEE it
    — a public profile whose existence is not secret, the honest 'not yours' the
    movie owner-gate's _forbidden_artifact returns; 404 (existence hidden) when the
    caller may not view it at all. The split matters because identity slugs are
    GUESSABLE (derived from display names) — unlike the uuid4 job ids the movie 403
    was written for, a 403 on a foreign PRIVATE profile would itself leak that the
    name exists (the enumeration S4 forbids), so a private-foreign profile 404s as
    if it did not exist."""
    if _may_view_profile(profile):
        return _forbidden_artifact()
    return jsonify({"error": "identity profile not found"}), 404


def _parse_private_flag(body):
    """Parse an optional ``private`` create/toggle flag. ``(value, None)`` (default
    False == PUBLIC) or ``(None, (payload, status))`` for a non-boolean."""
    raw = body.get("private", False)
    if raw is None:
        raw = False
    if not isinstance(raw, bool):
        return None, ({"error": "private must be a boolean"}, 400)
    return raw, None


def _validate_profile_reference_images(raws):
    """Jail-resolve + image-classify a list of reference-image paths EXACTLY as the
    studio routes do (``_jail_resolve`` -> ``media_store.ingest(kind_hint="image")``
    -> ``kind == "image"``). Returns ``(resolved_abs_paths, None)`` on success or
    ``(None, (error_payload, status))`` so the caller returns the clean 4xx. 1..4
    images required — the same envelope the movie route enforces."""
    if not isinstance(raws, list) or not raws:
        return None, ({"error": "reference_images must be a non-empty list of paths"}, 400)
    if len(raws) > identity_profiles.MAX_SOURCE_IMAGES:
        return None, ({
            "error": f"at most {identity_profiles.MAX_SOURCE_IMAGES} reference_images are accepted"
        }, 400)
    resolved: list = []
    for raw in raws:
        if not isinstance(raw, str) or not raw.strip():
            return None, ({"error": "each reference_image must be a non-empty path"}, 400)
        rp = _jail_resolve(raw)
        if rp is None:
            return None, ({"error": "reference_image outside storage jail"}, 400)
        if not os.path.isfile(rp):
            return None, ({"error": "reference_image not found"}, 404)
        try:
            iref = media_store.ingest(rp, kind_hint="image", owner=_caller_username())
        except Exception as exc:  # unreadable / not a real image = bad input
            return None, ({"error": f"reference_image is not a readable media file: {exc}"}, 400)
        if iref.kind != "image":
            return None, ({
                "error": f"reference_image is not an image (classified as {iref.kind})"
            }, 400)
        resolved.append(iref.uri)
    return resolved, None


def _reference_images_from_body(body):
    """Resolve an optional ``identity_profile`` slug to its canonical reference set,
    for the studio i2v / movie enqueue bodies.

    UNIFIED IDENTITY (operator 2026-07-12): an identity profile IS the identity. A
    studio enqueue may carry ``identity_profile: "<slug>"`` instead of a raw
    ``reference_images`` list. The saved profile's reference set is CANONICAL: when
    BOTH are present the profile WINS — a named, curated identity outranks ad-hoc
    paths (and a later edit to the profile re-resolves through the same slug). Raw
    ``reference_images`` stays accepted for the unsaved / backward-compat case.

    PROMOTED CANONICAL preference (stage (b)): once the operator promotes reconstruction
    views to the profile's ``canonical`` ref set (POST .../canonical), that APPROVED DNA
    outranks the raw uploaded ``reference_images`` — the profile resolves to ``canonical``
    when it is non-empty, falling back to ``reference_images`` otherwise. Same single
    code path either way.

    VERSION-AWARE DNA (VERSIONS slice): an identity now holds N named versions, each with
    its own promoted ``canonical``. The resolved DNA is the ACTIVE version's canonical by
    default; an optional ``identity_version`` (a version_id OR its name, e.g.
    "textured-01") names a specific version. Resolution precedence is:
    active/named version's canonical -> profile-level ``canonical`` -> ``reference_images``
    — so an un-versioned profile, an un-versioned caller, or an empty-canonical version all
    degrade to exactly the behavior above. A stale/unknown/archived ``identity_version`` is
    a clean 404.

    Returns ``(reference_images_or_None, None)`` on success, or
    ``(None, (error_payload, status))`` when a given slug names no profile — a stale
    slug is a clean 4xx, never a silent empty identity. The returned list still flows
    through the SAME jail + ingest + image-classify validation the routes already run
    (the profile's stored paths are re-checked, one code path)."""
    slug = body.get("identity_profile")
    if slug is None:
        return body.get("reference_images"), None
    if not isinstance(slug, str) or not slug.strip():
        return None, ({"error": "identity_profile must be a slug string"}, 400)
    profile = identity_profiles.get_profile(slug.strip())
    if profile is None:
        return None, ({"error": f"identity_profile {slug!r} not found"}, 404)
    # t171: a PRIVATE identity may only be RESOLVED/consumed (movie / i2v /
    # reconstruction DNA) by its owner or the operator/admin tier. A PUBLIC profile
    # (the default, and every legacy one) resolves for everyone — zero regression.
    # A private profile the caller may not view is reported as not-found (same 404
    # shape) so a render enqueue can never leak that the slug exists.
    if not _may_view_profile(profile):
        return None, ({"error": f"identity_profile {slug!r} not found"}, 404)

    # VERSION-AWARE DNA (IDENTITY-VERSIONS-SLICE.md slice 2): the id_lock reference set is
    # the ACTIVE version's canonical by default; an explicit ``identity_version`` names a
    # specific one — matched by its version_id OR its name (e.g. "textured-01"). A stale /
    # unknown / archived version is a clean 404, never a silent wrong-identity (the public
    # ``versions`` list already omits archived versions, so those are unreachable here).
    # Precedence for the resolved DNA:
    #     chosen version's canonical -> profile-level canonical -> reference_images
    # so a pre-versions profile, an un-versioned caller, or a version with an empty
    # canonical all degrade to EXACTLY today's behavior (no regression on first load).
    versions = [v for v in (profile.get("versions") or []) if isinstance(v, dict)]
    req_version = body.get("identity_version")
    chosen_version = None
    if req_version is not None:
        if not isinstance(req_version, str) or not req_version.strip():
            return None, ({"error": "identity_version must be a version id or name string"}, 400)
        needle = req_version.strip()
        chosen_version = next(
            (v for v in versions
             if v.get("version_id") == needle or v.get("name") == needle),
            None,
        )
        if chosen_version is None:
            return None, ({"error": f"identity_version {req_version!r} not found for "
                                    f"identity_profile {slug!r}"}, 404)
    else:
        active_id = profile.get("active_version")
        if active_id:
            chosen_version = next(
                (v for v in versions if v.get("version_id") == active_id), None)

    version_canonical = (
        [p for p in (chosen_version.get("canonical") or []) if isinstance(p, str)]
        if chosen_version else []
    )
    profile_canonical = [p for p in (profile.get("canonical") or []) if isinstance(p, str)]
    canonical = version_canonical or profile_canonical
    canonical_default = canonical if canonical else list(profile.get("reference_images") or [])
    # Provenance: did the DNA come from the canonical RING (angle-ordered frames), or from
    # the raw ``reference_images`` uploads (unordered photos)? Only a ring may be angle-
    # strided down to the render cap — see the narrowing block at the end of this function.
    from_canonical_ring = bool(canonical)

    # VIEW-AWARE DNA (IDENTITY-3D-CONTINUITY-PLAN.md S2): an optional ``identity_view`` hint
    # — a semantic name ("back", "left-profile", …) OR an ``{azimuth_deg, elevation_deg?}``
    # object — selects the K angle-nearest frames from the identity's turntable RING instead
    # of the flat cardinals, so a "from behind" shot conditions on back-view frames. The bank
    # is a pure computed read over the chosen version's turntable reconstruction (the ring the
    # identity already rendered — no new state). Precedence + graceful degrade:
    #   * NO hint                       -> canonical_default (byte-identical to before) ;
    #   * hint + a turntable bank exists -> the K angle-nearest bank frames (angle-spread) ;
    #   * hint but NO bank (versionless / clay-only / legacy profile) -> canonical_default.
    # So a hintless call is zero-regression and a hinted call on an identity without a ring
    # still yields the working canonical set (defaults-are-promises). K matches the RENDER cap
    # (``MAX_RENDER_REFS``, i.e. up to 4 — what one id_lock render can consume; the canonical
    # STORAGE cap is 8 since 2026-07-16). An invalid hint is a clean 400.
    view_hint = body.get("identity_view")
    view_source = "canonical-default"
    chosen = canonical_default
    if view_hint is not None:
        azimuth_deg, view_err = identity_profiles.azimuth_for_view(view_hint)
        if view_err is not None:
            return None, ({"error": view_err}, 400)
        chosen_version_id = chosen_version.get("version_id") if chosen_version else None
        bank = identity_profiles.bank_views(profile, version_id=chosen_version_id)
        if bank:
            # K = the RENDER cap (4). These frames become one render's id_lock refs, so this
            # tracks MAX_RENDER_REFS, NOT the canonical/storage cap (8 since 2026-07-16).
            # The two were the same number (4) when this was written.
            picked = identity_profiles.nearest_bank_views(
                bank, azimuth_deg, identity_profiles.MAX_RENDER_REFS)
            chosen = [b["path"] for b in picked]
            view_source = "explicit-view"
        # else: no turntable ring for this version -> fall through on canonical_default.

    # A saved profile keeps positional slots for sources that never materialized
    # (recorded in missing_references) or that have since gone stale on disk. Drop
    # any reference that no longer exists so ONE bad path can't 404 the whole
    # identity downstream — the profile IS the identity, complete or not
    # (operator 2026-07-12). If this empties the set, the caller's non-empty
    # validation returns a clean 400 rather than a per-path "not found".
    chosen = [r for r in chosen if isinstance(r, str) and os.path.isfile(r)]

    # CANONICAL 8 vs RENDER 4 (2026-07-16) — narrowed HERE, once, for every caller.
    # The canonical STORAGE cap widened to 8 (the 45° ring) but ONE id_lock/VACE render
    # still consumes at most 4 refs (a model constraint: each ref becomes a VACE reference
    # latent). Every consumer of this resolver feeds a render channel that hard-rejects >4
    # (/video/enqueue and /video/studio/movie 400; identity_reconstruction_schema and
    # studio.job RAISE), so an 8-view identity would otherwise fail against its OWN approved
    # DNA. This function is the ONE place that knows the refs came from a canonical RING, so
    # it is the honest place to narrow: callers stay unchanged and can't forget.
    # Striding (not [:4]) keeps the sample angle-spread — an 8-view set yields the 4
    # cardinals (0/90/180/270), byte-identical DNA to a 4-view profile today, instead of the
    # lopsided front+right half-turn [:4] would take. <=4 sets pass through untouched, so
    # every profile on disk right now resolves EXACTLY as before (zero regression).
    # The explicit-view path already asked the bank for MAX_RENDER_REFS, so it is a no-op
    # there; this backstops the canonical-default path.
    # SCOPED TO A CANONICAL RING ON PURPOSE: when a profile has NO canonical, this resolver
    # falls back to the raw ``reference_images`` uploads (up to 12) — unordered photos, not
    # ring frames. Those must NOT be strided (there is no angle to spread across, and their
    # >4 handling is each caller's existing [:4]/400 contract). Only narrow what actually
    # came from the canonical ring, so the legacy upload path is untouched.
    n_before = len(chosen)
    if from_canonical_ring:
        chosen = identity_profiles.render_refs_from_canonical(chosen)
    # Record which DNA path served the resolve so a live enqueue is auditable (the return
    # SHAPE is unchanged; this is observability only). ``view_source`` is the S2 honesty
    # flag: explicit-view == the turntable ring served it; canonical-default == today's set.
    logger.info("identity DNA resolved: slug=%s view_source=%s n_refs=%d (from %d canonical)",
                slug.strip(), view_source, len(chosen), n_before)
    return list(chosen), None


@video_bp.route("/video/identity-profiles", methods=["GET"])
def video_identity_profiles_list():
    # Active (non-archived) profiles, newest first, SCOPED to what the caller may
    # SEE (t171): every PUBLIC profile plus — for a member/operator — their own /
    # (operator tier) all PRIVATE ones. A video-share guest sees PUBLIC only. The
    # store's projection folds slug + owner + private into each row, so the filter
    # reads them straight off the wire shape.
    visible = [p for p in identity_profiles.list_profiles() if _may_view_profile(p)]
    return jsonify({"profiles": visible}), 200


@video_bp.route("/video/identity-profiles", methods=["POST"])
def video_identity_profiles_create():
    # {name, reference_images[1..4], notes?} -> create. Slug derives from name; a
    # collision with an existing ACTIVE profile is a 409 (never a silent overwrite).
    body = request.get_json(silent=True) or {}
    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        return jsonify({"error": "name is required"}), 400
    resolved, err = _validate_profile_reference_images(body.get("reference_images"))
    if err is not None:
        payload, status = err
        return jsonify(payload), status
    notes = body.get("notes")
    if notes is not None and not isinstance(notes, str):
        return jsonify({"error": "notes must be a string"}), 400
    private, perr = _parse_private_flag(body)  # t171: optional, default False == PUBLIC
    if perr is not None:
        payload, status = perr
        return jsonify(payload), status
    try:
        # t171: stamp the CREATOR as owner (the same identity the movie owner-gate
        # uses), so only they + the operator tier may later mutate it.
        profile = identity_profiles.create_profile(
            name, resolved, notes=notes or "",
            owner=_request_principal(), private=private)
    except identity_profiles.ProfileError as exc:  # dup slug / bad shape = errors-as-data
        status = 409 if exc.code == "duplicate" else 400
        return jsonify({"error": str(exc), "code": exc.code}), status
    return jsonify({"profile": profile}), 201


@video_bp.route("/video/identity-profiles/<slug>", methods=["GET"])
def video_identity_profile_get(slug):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    # t171: a PRIVATE profile is visible only to its owner + the operator/admin
    # tier. Hide a foreign private profile as not-found (404, not a 403 that would
    # confirm the guessable slug); PUBLIC profiles fall through to everyone.
    if not _may_view_profile(profile):
        return jsonify({"error": "identity profile not found"}), 404
    # ANGLE BANK summary (IDENTITY-3D-CONTINUITY-PLAN.md S1): surface, per version, WHAT
    # angles the identity actually carries (count + degrees_per_frame + azimuth range) so
    # the operator/UI can see the ring a view hint can select from. Purely ADDITIVE — a
    # new ``views`` key on each version object, computed read-only from the turntable
    # reconstruction; no existing key is removed or renamed. Scoped to this single-profile
    # GET (not the list) so the cheap list endpoint stays untaxed.
    for v in profile.get("versions") or []:
        if isinstance(v, dict):
            v["views"] = identity_profiles.views_summary(profile, version_id=v.get("version_id"))
    return jsonify({"profile": profile}), 200


@video_bp.route("/video/identity-profiles/<slug>", methods=["DELETE"])
def video_identity_profile_delete(slug):
    # ARCHIVE semantics (never-delete doctrine): the store moves the entry under a
    # `_deleted` key with a timestamp rather than erasing it. Idempotent — deleting
    # an unknown/already-archived slug is a clean 404 no-op.
    #
    # t171: only the owner or the operator/admin tier may delete. Fetch + authorize
    # BEFORE the archive. A foreign PUBLIC profile gets an honest 403; a foreign
    # PRIVATE one gets the SAME 404 (archived:false) an unknown slug gets, so its
    # existence is never leaked.
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found", "archived": False}), 404
    if not _owns_profile(profile.get("owner")):
        if _may_view_profile(profile):
            return _forbidden_artifact()
        return jsonify({"error": "identity profile not found", "archived": False}), 404
    archived = identity_profiles.delete_profile(slug)
    if archived is None:
        return jsonify({"error": "identity profile not found", "archived": False}), 404
    return jsonify({"ok": True, "archived": True, "slug": slug}), 200


# --------------------------------------------------------------------------- #
# PATCH /video/identity-profiles/<slug> — edit an existing profile's DISPLAY
# fields + reference set. {name?, notes?, reference_images?}, all optional (a
# true partial update — an omitted key is left untouched, matching
# identity_profiles.update_profile's **kwargs contract below). Any given
# `reference_images` runs the SAME jail + ingest + image-classify validation as
# POST create (`_validate_profile_reference_images`), so a PATCH can never leave
# a profile pointing at an unreadable/non-image/jail-escaping path.
#
# RENAMING KEEPS THE SLUG STABLE — `name` only ever changes the stored display
# string, never the `<slug>` this route (and every identity_profile:<slug>
# reference in a saved template/spec/enqueue body) keys on. Re-slugging on
# rename would silently break every one of those references; see
# identity_profiles.update_profile's docstring for the full rationale.
#
# An identity keeps >=1 reference image always: a given `reference_images` must
# be non-empty (the store rejects an empty list exactly like create does) — omit
# the key entirely to leave the current set untouched. Existence is checked
# BEFORE any (potentially expensive) reference validation runs, so an unknown
# slug 404s fast without jail-resolving/ingesting images that will never be
# stored; the store's own None-return is still honored afterward as a defensive
# recheck against a concurrent archive racing this same request.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>", methods=["PATCH"])
def video_identity_profile_update(slug):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    if not _owns_profile(profile.get("owner")):  # t171: owner/operator only
        return _deny_profile(profile)

    body = request.get_json(silent=True) or {}
    kwargs: dict = {}

    if "name" in body:
        name = body.get("name")
        if not isinstance(name, str) or not name.strip():
            return jsonify({"error": "name is required"}), 400
        kwargs["name"] = name

    if "notes" in body:
        notes = body.get("notes")
        if notes is not None and not isinstance(notes, str):
            return jsonify({"error": "notes must be a string"}), 400
        kwargs["notes"] = notes or ""

    if "reference_images" in body:
        resolved, err = _validate_profile_reference_images(body.get("reference_images"))
        if err is not None:
            payload, status = err
            return jsonify(payload), status
        # Wire key stays ``reference_images`` (UI + tests read it); the store's
        # param is ``source_images`` (internal rename). Map here, or update_profile
        # raises "unexpected keyword argument 'reference_images'" (the PATCH 500).
        kwargs["source_images"] = resolved

    try:
        profile = identity_profiles.update_profile(slug, **kwargs)
    except identity_profiles.ProfileError as exc:  # errors-as-data, never a 500
        return jsonify({"error": str(exc), "code": exc.code}), 400
    if profile is None:  # lost a race with a concurrent archive
        return jsonify({"error": "identity profile not found"}), 404
    return jsonify({"profile": profile}), 200


# --------------------------------------------------------------------------- #
# POST /video/identity-profiles/<slug>/private — t171 owner-only VISIBILITY toggle.
# Body {"private": bool}: flip a profile PUBLIC<->PRIVATE. Owner or the operator/
# admin tier ONLY — a video-share guest may VIEW a public profile but can never
# change its visibility, and a foreign PRIVATE one 404s (existence hidden). A
# PUBLIC profile is visible to all; PRIVATE only to owner + operator. Returns
# {profile} carrying the new ``private`` state.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/private", methods=["POST"])
def video_identity_profile_set_private(slug):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    if not _owns_profile(profile.get("owner")):
        return _deny_profile(profile)
    body = request.get_json(silent=True) or {}
    if "private" not in body or not isinstance(body.get("private"), bool):
        return jsonify({"error": "private (boolean) is required"}), 400
    updated = identity_profiles.set_profile_private(slug, body["private"])
    if updated is None:  # lost a race with a concurrent archive
        return jsonify({"error": "identity profile not found"}), 404
    return jsonify({"profile": updated}), 200


# --------------------------------------------------------------------------- #
# POST /video/identity-profiles/<slug>/reconstruction — STAGE (b): generate an
# identity-locked TURNAROUND of the character (one still per view) from its reference
# images + description, stored in the identity's dir for approval.
#
# Body (all optional): {prompt?: str, views?: [str], seed?: int, mode?: "sheet"|"turntable"}
#   prompt  extra description woven into every view prompt; default = the profile's notes.
#   views   the view names to render; default ["front","three_quarter","profile","back"].
#           A SINGLE-view request (["front"]) is valid — the cheap one-render check.
#           IGNORED in turntable mode (the orbit clip's frames define the set).
#   seed    base render seed (default 0; all views share it — the view differs by prompt).
#   mode    "sheet" (default — N independent view-stills) or "turntable" (ONE 360° orbit
#           clip, every frame kept as an angular degree-view the UI scrubs to rotate).
#
# Enqueues ONE orchestrator job (identity_reconstruction) and returns {job_id, recon_id};
# its runner renders each view behind the swap seam, then attaches the produced stills to
# the profile. Poll GET /video/jobs/<job_id>; on done, re-read the profile (GET .../<slug>)
# for the new ``reconstructions`` entry keyed by recon_id.
# --------------------------------------------------------------------------- #
# NOTE (keeper 2026-07-14): the OLD sheet/turntable-only reconstruction handler was
# CONSOLIDATED into the single angle-ring-aware handler below
# (video_identity_profile_reconstruction). Its @video_bp.route decorator is removed so
# exactly ONE rule serves this path — Werkzeug matched the first-registered rule, which
# shadowed the newer handler and rejected mode="angle-ring". Body retained (never-delete);
# full prior file at video_routes.py.bak-keeper-20260714.
def _retired_video_identity_profile_reconstructions(slug):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    body = request.get_json(silent=True) or {}

    # Resolve the profile's reference set through the SAME resolver the studio enqueue
    # uses (canonical-preferred) + the SAME jail + ingest + image-classify validation
    # (single code path). A promoted canonical set outranks the raw uploads here too.
    refs_in, perr = _reference_images_from_body({"identity_profile": slug})
    if perr is not None:
        payload, status = perr
        return jsonify(payload), status
    resolved_refs, verr = _validate_profile_reference_images(refs_in)
    if verr is not None:
        payload, status = verr
        return jsonify(payload), status

    # prompt: default to the profile's notes (the description the profile carries).
    prompt = body.get("prompt")
    if prompt is None:
        prompt = profile.get("notes") or ""
    elif not isinstance(prompt, str):
        return jsonify({"error": "prompt must be a string"}), 400

    # views: default to the canonical turnaround set; a non-empty list of non-empty
    # strings (a single view is valid — verify one render cheaply first).
    views = body.get("views")
    if views is None:
        views = list(_DEFAULT_RECON_VIEWS)
    if not isinstance(views, list) or not views:
        return jsonify({"error": "views must be a non-empty list of view names"}), 400
    for v in views:
        if not isinstance(v, str) or not v.strip():
            return jsonify({"error": "each view must be a non-empty string"}), 400

    seed = body.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        return jsonify({"error": "seed must be an int"}), 400

    # mode: default "sheet" (the existing N-independent-view-stills path). "turntable"
    # renders ONE 360° orbit clip and keeps every frame as a scrubbable degree-view
    # (``views`` from the body are ignored in turntable mode — the orbit defines the set).
    mode = body.get("mode", "sheet")
    if mode not in ("sheet", "turntable"):
        return jsonify({"error": 'mode must be "sheet" or "turntable"'}), 400

    recon_id = "recon_" + secrets.token_hex(8)
    try:
        spec = make_identity_reconstruction(
            slug=slug,
            recon_id=recon_id,
            # builder param is source_images (internal rename); passing
            # reference_images= raises "unexpected keyword argument".
            source_images=tuple(resolved_refs),
            views=tuple(views),
            base_prompt=prompt,
            seed=seed,
            mode=mode,
            # geometry (<=480p id_lock ceiling) + autofit VRAM are the schema defaults.
        )
    except (ValueError, TypeError) as exc:  # bad fields = 400
        return jsonify({"error": str(exc)}), 400
    job_id = _video_enqueue("identity_reconstruction", spec)
    return jsonify({"job_id": job_id, "recon_id": recon_id}), 200


# --------------------------------------------------------------------------- #
# POST /video/identity-profiles/<slug>/canonical — STAGE (b) approve: promote chosen
# reconstruction views into the profile's ``canonical`` reference set (the approved
# character DNA the resolver then PREFERS over the raw uploads). Body:
# {recon_id: str, views: [int]} — the view indices of that reconstruction to promote
# (at most 4; canonical feeds the id_lock reference channel). Returns {profile} with
# ``canonical`` populated. An unknown recon_id / out-of-range index is a clean 400.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/canonical", methods=["POST"])
def video_identity_profile_canonical(slug):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    if not _owns_profile(profile.get("owner")):  # t171: owner/operator only
        return _deny_profile(profile)
    body = request.get_json(silent=True) or {}
    recon_id = body.get("recon_id")
    if not isinstance(recon_id, str) or not recon_id.strip():
        return jsonify({"error": "recon_id is required"}), 400
    views = body.get("views")
    if not isinstance(views, list) or not views:
        return jsonify({"error": "views must be a non-empty list of view indices"}), 400
    for v in views:
        if not isinstance(v, int) or isinstance(v, bool):
            return jsonify({"error": "each view must be an integer index"}), 400
    try:
        profile = identity_profiles.promote_reconstruction_views(slug, recon_id, views)
    except identity_profiles.ProfileError as exc:  # bad index / unknown recon = 400
        return jsonify({"error": str(exc), "code": exc.code}), 400
    if profile is None:  # lost a race with a concurrent archive
        return jsonify({"error": "identity profile not found"}), 404
    return jsonify({"profile": profile}), 200

# --------------------------------------------------------------------------- #
# 5d) POST /video/identity-profiles/<slug>/reconstruction — STAGE (b) update:
#     Support "mode": "angle-ring" in the base reconstruction handler.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/reconstruction", methods=["POST"])
def video_identity_profile_reconstruction(slug):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    if not _owns_profile(profile.get("owner")):  # t171: owner/operator only
        return _deny_profile(profile)
    body = request.get_json(silent=True) or {}

    refs_in, perr = _reference_images_from_body({"identity_profile": slug})
    if perr is not None:
        payload, status = perr
        return jsonify(payload), status
    # id_lock reference channel accepts at most 4 (_MAX_CANONICAL_IMAGES). A profile may
    # hold up to 12 SOURCE images (and, since 2026-07-16, up to 8 CANONICAL views), so
    # narrow the resolver's canonical-preferred, existence-filtered set to 4 rather than
    # 400-ing with "at most 4 ... accepted".
    # A canonical RING is already narrowed to 4 (ring-strided, so the 4 cardinals rather
    # than a lopsided half-turn) inside _reference_images_from_body. This [:4] therefore
    # only ever trims a raw 12-image UPLOAD set — unordered photos with no angle to spread
    # across — so it keeps its original first-4 behavior, unchanged.
    refs_in = list(refs_in)[:4]
    if not refs_in:
        return jsonify({"error": "Profile has no valid reference images"}), 400
    resolved_refs, verr = _validate_profile_reference_images(refs_in)
    if verr is not None:
        payload, status = verr
        return jsonify(payload), status

    prompt = body.get("prompt")
    if prompt is None:
        prompt = profile.get("notes") or ""
    elif not isinstance(prompt, str):
        return jsonify({"error": "prompt must be a string"}), 400

    # mode: "sheet" | "turntable" | "angle-ring"
    mode = body.get("mode", "sheet")
    if mode not in ("sheet", "turntable", "angle-ring"):
        return jsonify({"error": 'mode must be "sheet", "turntable", or "angle-ring"'}), 400

    # Extract angle-ring parameters when active
    angle_step_deg = body.get("angle_step_deg", 10)
    elevations_deg = body.get("elevations_deg", [0])

    if mode == "angle-ring":
        # 36 views at 10 deg step is default
        if not isinstance(angle_step_deg, int) or angle_step_deg <= 0:
            return jsonify({"error": "angle_step_deg must be a positive integer"}), 400
        if not isinstance(elevations_deg, list) or not elevations_deg:
            return jsonify({"error": "elevations_deg must be a non-empty list of integers"}), 400
        views = [f"angle_{deg}" for deg in range(0, 360, angle_step_deg)]
    else:
        views = body.get("views")
        if views is None:
            views = list(_DEFAULT_RECON_VIEWS)
        if not isinstance(views, list) or not views:
            return jsonify({"error": "views must be a non-empty list"}), 400

    seed = body.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        return jsonify({"error": "seed must be an int"}), 400

    # CLEANUP-PROMPT slice (C4 — the reachability wire): same body-or-gen_settings
    # precedence as the /generate route's vision_model/cleanup_prompt resolution — an
    # EXPLICIT request-body ``negative_prompt`` wins; else the identity's PERSISTED
    # gen_settings.negative_prompt (the Advanced-panel field); else "" (today's exact
    # call, defaults-are-promises). NOTE: make_identity_reconstruction is currently the
    # bare ``**kwargs`` passthrough (identity_reconstruction_schema.py:570), so this
    # value reaches IdentityReconstructionSpec.negative_prompt directly via the
    # dataclass constructor, unvalidated by the dead factory at :119 — acceptable here
    # since the field is a plain string with default "" (out of scope to fix the shadow).
    _gen_settings = profile.get("gen_settings") or {}
    negative_prompt = body.get("negative_prompt")
    if negative_prompt in (None, ""):
        negative_prompt = _gen_settings.get("negative_prompt", "")

    recon_id = "recon_" + secrets.token_hex(8)
    try:
        spec = make_identity_reconstruction(
            slug=slug,
            recon_id=recon_id,
            # builder param is source_images (internal rename); passing
            # reference_images= raises "unexpected keyword argument".
            source_images=tuple(resolved_refs),
            views=tuple(views),
            base_prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            mode=mode,
            angle_step_deg=angle_step_deg,
            elevations_deg=elevations_deg,
        )
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
        
    job_id = _video_enqueue("identity_reconstruction", spec)
    return jsonify({"job_id": job_id, "recon_id": recon_id}), 200


# --------------------------------------------------------------------------- #
# 5e) PATCH /video/identity-profiles/<slug>/reconstruction/<recon_id>/views/<view_id>
#     Approve or reject a specific angle tile. Updates the status locally in the
#     identity profile record.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/reconstruction/<recon_id>/views/<view_id>", methods=["PATCH"])
def video_identity_profile_view_status(slug, recon_id, view_id):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    if not _owns_profile(profile.get("owner")):  # t171: owner/operator only
        return _deny_profile(profile)
        
    body = request.get_json(silent=True) or {}
    status = body.get("status")
    if status not in ("approved", "rejected"):
        return jsonify({"error": "status must be 'approved' or 'rejected'"}), 400

    try:
        profile = identity_profiles.update_reconstruction_view_status(
            slug=slug,
            recon_id=recon_id,
            view_id=view_id,
            status=status
        )
    except identity_profiles.ProfileError as exc:
        return jsonify({"error": str(exc), "code": exc.code}), 400
        
    return jsonify({"profile": profile}), 200


# --------------------------------------------------------------------------- #
# 5f) POST /video/identity-profiles/<slug>/reconstruction/<recon_id>/views/<view_id>/regenerate
#     Regenerate a single angle conditioned on nearby approved angle neighbors.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/reconstruction/<recon_id>/views/<view_id>/regenerate", methods=["POST"])
def video_identity_profile_view_regenerate(slug, recon_id, view_id):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    if not _owns_profile(profile.get("owner")):  # t171: owner/operator only
        return _deny_profile(profile)
        
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt") or (profile.get("notes") or "")
    seed = body.get("seed", secrets.randbelow(1000000))
    use_neighbors = bool(body.get("use_nearest_approved_neighbors", True))

    try:
        # Fetch the active reconstruction configuration to preserve overall context
        spec = identity_profiles.make_single_view_regeneration_spec(
            slug=slug,
            recon_id=recon_id,
            view_id=view_id,
            prompt=prompt,
            seed=seed,
            use_neighbors=use_neighbors
        )
    except identity_profiles.ProfileError as exc:
        return jsonify({"error": str(exc), "code": exc.code}), 400

    job_id = _video_enqueue("identity_view_regenerate", spec)
    return jsonify({"job_id": job_id, "recon_id": recon_id}), 200


# --------------------------------------------------------------------------- #
# Shared: build the cardinal-view -> path map for a mesh build, JAILED to the
# profile's OWN reference/canonical images. Central has no GPU — the relay runner
# reads exactly these paths — so the "default front = canonical[0]-on-disk else the
# first existing reference" rule + the path jail live in ONE place, shared by the
# per-reconstruction mesh route (5g) and the one-click /generate template.
#   Returns (view_map, candidates, None) on success, or (None, None, (payload, status))
#   when an explicit view path is not one of the profile's own images (a clean 400) —
#   the caller then returns ``jsonify(payload), status``.
#   ``candidates`` is the ordered list of ALL existing-on-disk source reference images
#   — populated ONLY when the caller did NOT explicitly assign ``views.front`` (an
#   explicit front assignment disables fleet-VLM auto-selection: candidates == []).
#   The relay runner (bus job context, never the request handler — a vision call is
#   ~5-10s/image and there can be up to 12) uses this to ask the fleet vision amenity
#   which candidate shows the character's FULL BODY and swap it in as front (keeper
#   2026-07-14: luigi ref_00 was a cropped waist-up portrait -> cut-off-legs mesh).
# --------------------------------------------------------------------------- #
def _resolve_profile_mesh_views(profile, body_views):
    canonical = [p for p in (profile.get("canonical") or []) if isinstance(p, str)]
    references = [p for p in (profile.get("reference_images") or []) if isinstance(p, str)]
    allowed = set(canonical) | set(references)  # the jail: only the profile's own images

    def _first_existing(paths):
        for p in paths:
            if os.path.isfile(p):
                return p
        return None

    # Default FRONT = the first existing SOURCE reference image — NEVER canonical.
    # canonical is GENERATION DNA and (via auto-promote) usually holds renders of the
    # PREVIOUS mesh's turntable; feeding it back into mesh reconstruction creates a
    # feedback loop that faithfully re-meshes the prior mesh, artifacts and all (bit
    # live 2026-07-14: luigi's 2nd generate re-meshed his 1st mesh's background slab
    # because front defaulted to canonical[0] — rembg 'applied' but the subject WAS the
    # old render). Reconstruction eats ORIGINAL photos; canonical is only reachable via
    # an EXPLICIT body views assignment below.
    existing_references = [p for p in references if os.path.isfile(p)]
    front = existing_references[0] if existing_references else None
    view_map: dict = {}
    if front is not None:
        view_map["front"] = front
    # Candidates for fleet-VLM auto-selection: every OTHER existing source reference,
    # in order (the current default front stays first-tried / the fallback). Cleared
    # below the moment an explicit front is assigned.
    candidates: list = list(existing_references)

    # Optional explicit assignment: {views: {front|right|back|left: <path>}}. Each path
    # must be one of the profile's own images (the jail) — an arbitrary path is a clean
    # 400, never accepted.
    if isinstance(body_views, dict):
        for vname, vpath in body_views.items():
            if vname not in _MESH_VIEW_NAMES:
                return None, None, ({"error": f"unknown view {vname!r}; expected one of "
                                        f"{list(_MESH_VIEW_NAMES)}"}, 400)
            if not isinstance(vpath, str) or vpath not in allowed:
                return None, None, ({"error": f"view {vname!r} must reference one of this "
                                        "profile's own reference/canonical images"}, 400)
            view_map[vname] = vpath
            if vname == "front":
                # An EXPLICIT front assignment is an operator override — auto-selection
                # never second-guesses it.
                candidates = []

    if "front" not in view_map:
        return None, None, ({"error": "profile has no usable reference image for the mesh "
                                "front view"}, 400)
    return view_map, candidates, None


# --------------------------------------------------------------------------- #
# 5g) POST /video/identity-profiles/<slug>/reconstruction/<recon_id>/mesh
#     Trigger a ComfyUI mesh pipeline job (Hunyuan3D) utilizing selected, approved
#     anchor views.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/reconstruction/<recon_id>/mesh", methods=["POST"])
def video_identity_profile_build_mesh(slug, recon_id):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    if not _owns_profile(profile.get("owner")):  # t171: owner/operator only
        return _deny_profile(profile)

    body = request.get_json(silent=True) or {}

    # The mesh (+ optional turntable) is built from the identity's OWN reference/canonical
    # images — NEVER arbitrary paths (the shared jail below). (The old contract's
    # `views: [viewIds]` list is IGNORED, not rejected — the default front mapping applies
    # — so an older client never hard-400s here.)
    view_map, view_candidates, verr = _resolve_profile_mesh_views(profile, body.get("views"))
    if verr is not None:
        payload, status = verr
        return jsonify(payload), status

    # Optional knobs — malformed optional params fall back to their defaults (a bad tuning
    # value should not hard-fail the build; make_identity_mesh re-validates regardless).
    def _pos_int(d, name, default):
        v = d.get(name, default)
        return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else default

    seed = body.get("seed", 12345)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        seed = 12345
    tt = body.get("turntable") if isinstance(body.get("turntable"), dict) else {}
    elev = tt.get("elevation_deg", 8.0)
    if not isinstance(elev, (int, float)) or isinstance(elev, bool):
        elev = 8.0

    # PER-IDENTITY VISION MODEL — same precedence as the one-click /generate route
    # (request-body ``vision_model`` > the identity's persisted gen_settings.vision_model
    # > None == fleet default). A surgical per-reconstruction mesh build runs the SAME
    # front-select step, so it honors the identity's chosen VL model too. Blank/None here
    # keeps today's behavior exactly (no ``model`` sent -> the 3B).
    _gen_settings = profile.get("gen_settings") or {}
    vision_model = body.get("vision_model")
    if vision_model in (None, ""):
        vision_model = _gen_settings.get("vision_model")
    if vision_model in (None, ""):
        # AUTO default (operator 2026-07-15): prefer a 7B VL when the fleet has one —
        # the 3B mislabeled a waist-up ref as full-body. None when no 7B is installed
        # (== fleet-default 3B); the relay degrades a failing 7B call back to the 3B.
        vision_model = identity_profiles.preferred_identity_vision_model()

    # CLEANUP / NEGATIVE prompt — SAME body-or-gen_settings precedence as /generate
    # (C4). This surgical per-reconstruction re-mesh runs the SAME T-pose front render
    # + studio path, so it must honor the identity's persisted cleanup/negative too —
    # else a re-mesh silently drops the "no object on her back" instruction. Blank =
    # today's exact behavior.
    cleanup_prompt = body.get("cleanup_prompt")
    if cleanup_prompt in (None, ""):
        cleanup_prompt = _gen_settings.get("cleanup_prompt", "")
    negative_prompt = body.get("negative_prompt")
    if negative_prompt in (None, ""):
        negative_prompt = _gen_settings.get("negative_prompt", "")

    try:
        spec = make_identity_mesh(
            slug=slug,
            recon_id=recon_id,
            view_sources=tuple(view_map.items()),
            seed=seed,
            num_inference_steps=_pos_int(body, "num_inference_steps", 30),
            octree_resolution=_pos_int(body, "octree_resolution", 380),
            texture=bool(body.get("texture", False)),
            chain_turntable=bool(body.get("chain_turntable", True)),
            frame_count=_pos_int(tt, "frame_count", 72),
            fps=_pos_int(tt, "fps", 24),
            width=_pos_int(tt, "width", 768),
            height=_pos_int(tt, "height", 768),
            elevation_deg=elev,
            transparent=bool(tt.get("transparent", False)),
            view_candidates=view_candidates,
            vision_model=vision_model,
            cleanup_prompt=cleanup_prompt,
            negative_prompt=negative_prompt,
        )
    except (ValueError, TypeError) as exc:  # bad fields = 400
        return jsonify({"error": str(exc)}), 400

    # Seed the mesh state to "queued" so the GET mesh-status route + UI reflect the
    # in-flight build immediately (best-effort — a clean no-op if this recon_id has no
    # attached reconstruction yet; the relay attaches + records the terminal state).
    try:
        identity_profiles.set_mesh_state(slug, recon_id, {"status": "queued", "error": None})
    except identity_profiles.ProfileError:
        pass

    job_id = _video_enqueue("identity_mesh_build", spec)
    return jsonify({"job_id": job_id, "recon_id": recon_id}), 200


# --------------------------------------------------------------------------- #
# 5h) GET /video/identity-profiles/<slug>/reconstruction/<recon_id>/mesh
#     Retrieve the generation status, active GLB path, and preview assets.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/reconstruction/<recon_id>/mesh", methods=["GET"])
def video_identity_profile_mesh_status(slug, recon_id):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    if not _may_view_profile(profile):  # t171 read-gate: private -> owner/operator only
        return jsonify({"error": "identity profile not found"}), 404

    mesh_state = identity_profiles.get_mesh_state(slug, recon_id)
    if mesh_state is None:
        return jsonify({"status": "none"}), 200

    return jsonify(mesh_state), 200


# --------------------------------------------------------------------------- #
# 5g-vx) POST /video/identity-profiles/video-extract
#     char360 VIDEO -> per-character 360° view-sets, written back into identity profiles
#     (CHAR360-FEATURE-PLAN S3). This is its OWN relay job (identity_video_extract, mirrors
#     identity_mesh_build) — it relays a source clip to the standalone GPU render service
#     (which grew the video_extract kind in S2), polls it, downloads the per-character
#     view-sets, and writes them back. It is NOT the local identity_reconstruction job.
#
#     Body:
#       source          a MediaRef-shaped dict for the source VIDEO clip (the same shape
#                       the frame_extract route accepts — rehydrated via make_media_ref).
#                       Its uri must be an absolute path inside the storage jail; the runner
#                       forwards it to the service as video_path (ae + central share the
#                       mount, so a large clip is never base64-inflated through the body).
#       target          "create" (mint a NEW profile per detected character), "review"
#                       (CHARACTER-GROUPS-PLAN S1 — run char360 and RETURN the grouped views
#                       for curation, writing NO profile), or an EXISTING profile slug
#                       (append each character's view-set to it). Required.
#       char360_params? optional passthrough knobs for the service's Char360Params
#                       (stride / yolo_model / min_h_frac / cluster_dist / min_faces);
#                       unknown keys are dropped by the spec factory.
#     Returns {job_id, target} 200; a bad source/target is a clean 400; an unknown target
#     slug is a 404 (checked up front, mirroring the mesh route's profile guard).
#
#     REVIEW-mode RESULT CONTRACT (S1): the terminal result carries the grouped manifest —
#     GET /video/jobs/<job_id> -> result.groups =
#       {"n_characters": int,
#        "groups": [{"char": str, "face_centroid": [float]|null,
#                    "views": [{"url": <abs jailed path handle>, "yaw": float|null,
#                               "bin": int|null, "score": float|null}]}]}
#     Each view "url" is the persisted crop's media HANDLE (an absolute path under the
#     storage jail); the UI renders it via mediaBytesUrl(url) — the SAME GET /video/media
#     ?handle= route the profile canonical views use. No profile is created or appended.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/video-extract", methods=["POST"])
def video_identity_profile_video_extract():
    body = request.get_json(silent=True) or {}

    # source: a MediaRef-shaped dict (mirrors the frame_extract route — the console builds
    # the ref via POST /video/ingest, then hands us the ref dict). Rehydrate + validate it,
    # then jail its uri so the runner can never be pointed at an arbitrary file to relay.
    source_d = body.get("source")
    if not isinstance(source_d, dict):
        return jsonify({"error": "missing or invalid 'source' MediaRef (a video)"}), 400
    try:
        source = make_media_ref(**source_d)
    except (ValueError, TypeError) as exc:  # bad ref shape / non-abs uri = 400
        return jsonify({"error": f"invalid source MediaRef: {exc}"}), 400
    if source.kind != "video":
        return jsonify({"error": f"source must be a video; got kind={source.kind!r}"}), 400
    if _jail_resolve(source.uri) is None:
        return jsonify({"error": "source video is outside the storage jail"}), 400

    target = body.get("target")
    if not isinstance(target, str) or not target.strip():
        return jsonify({"error": "target is required ('create' or an existing profile slug)"}), 400
    target = target.strip()

    # An ADD target must name a LIVE profile — a clean 404 up front (rather than after a
    # full extract), mirroring the mesh route's get_profile guard. The correlation id handed
    # to the service is the slug (add) or a synthesized id (create/review — the runner
    # synthesizes one when identity_id is blank, but pass an explicit honest one here too).
    #
    # "review" (CHARACTER-GROUPS-PLAN S1) is NON-COMMITTING: like "create" it names no
    # existing profile, so it is NOT profile-checked here — it runs char360 and returns the
    # grouped views for curation WITHOUT writing anything. The grouped manifest rides the
    # job's terminal result and is read via GET /video/jobs/<job_id> -> result.groups.
    if target in ("create", "review"):
        identity_id = None  # the runner synthesizes videoextract-<job_id>
    else:
        # t171: appending a character view-set to an EXISTING profile is a MUTATION
        # of it — owner or operator/admin only. Foreign PUBLIC -> honest 403;
        # foreign PRIVATE -> the same {target} not-found 404 (existence hidden).
        target_profile = identity_profiles.get_profile(target)
        if target_profile is None:
            return jsonify({"error": f"identity profile {target!r} not found"}), 404
        if not _owns_profile(target_profile.get("owner")):
            if _may_view_profile(target_profile):
                return _forbidden_artifact()
            return jsonify({"error": f"identity profile {target!r} not found"}), 404
        identity_id = target

    char360_params = body.get("char360_params")
    if char360_params is not None and not isinstance(char360_params, dict):
        return jsonify({"error": "char360_params must be an object"}), 400

    try:
        spec = make_identity_video_extract(
            source=source,
            target=target,
            char360_params=char360_params,
            identity_id=identity_id,
        )
    except (ValueError, TypeError) as exc:  # bad fields = 400
        return jsonify({"error": str(exc)}), 400

    job_id = _video_enqueue("identity_video_extract", spec)
    return jsonify({"job_id": job_id, "target": target}), 200


# --------------------------------------------------------------------------- #
# 5h) POST /video/identity-profiles/from-groups
#     CHARACTER-GROUPS-PLAN S3 — commit the curated char360 groups (S1's REVIEW
#     manifest, edited client-side by S2) into identity profiles. ONE profile is
#     created per submitted group, through the EXACT SAME validation + copy path
#     as the single-profile create route above
#     (_validate_profile_reference_images -> identity_profiles.create_profile):
#     no re-invented staging, no new jail rule. The reference-image entries are
#     the jailed crop handles S1 persisted under
#     <IDENTITIES_HOME>/_char360_extracts/<job_id>/char_NN/<file> (servable via
#     GET /video/media?handle=), but any jail-valid image path is accepted —
#     this route does not care how a handle was produced.
#
#     Body: {"groups": [{"name"?: str, "reference_images": [<handle>, ...]}]}
#     A missing/blank name is derived as "Character N" (1-based index over the
#     submitted list) so every group still gets a stable, url-safe default slug.
#
#     Returns 200 ALWAYS (a bad group is errors-as-data, never a batch failure) —
#       {"results": [{"name": str, "ok": bool, "slug"?: str, "error"?: str}]}
#     — one result per input group, ORDER-PRESERVING. A validation failure (bad/
#     missing/non-image reference, jail escape, dup slug, ...) records
#     ok:false + error for THAT group only; the rest of the batch still commits.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/from-groups", methods=["POST"])
def video_identity_profiles_from_groups():
    body = request.get_json(silent=True) or {}
    groups = body.get("groups")
    if not isinstance(groups, list) or not groups:
        return jsonify({"error": "groups must be a non-empty list"}), 400
    private, perr = _parse_private_flag(body)  # t171: optional top-level, default PUBLIC
    if perr is not None:
        payload, status = perr
        return jsonify(payload), status
    owner = _request_principal()  # t171: stamp the creator on every group's profile

    results = []
    for idx, group in enumerate(groups):
        default_name = f"Character {idx + 1}"
        if not isinstance(group, dict):
            results.append({"name": default_name, "ok": False, "error": "group must be an object"})
            continue
        raw_name = group.get("name")
        name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else default_name

        resolved, err = _validate_profile_reference_images(group.get("reference_images"))
        if err is not None:
            payload, _status = err
            results.append({"name": name, "ok": False, "error": payload.get("error", "invalid reference_images")})
            continue

        try:
            profile = identity_profiles.create_profile(
                name, resolved, owner=owner, private=private)
        except identity_profiles.ProfileError as exc:  # dup slug / bad shape = errors-as-data
            results.append({"name": name, "ok": False, "error": str(exc)})
            continue

        results.append({"name": name, "ok": True, "slug": profile["slug"]})

    return jsonify({"results": results}), 200


# --------------------------------------------------------------------------- #
# k94 — ONE PATH: a video (or a few photos) of a character in -> a bindable identity
#       (reference views + canonical + 3D GLB) out. No knobs on the happy path.
#
# POST /video/identity-profiles/from-video
#     Body: {"source": <MediaRef video>, "name": "…", "mesh_params"?: {texture?,
#            octree_resolution?}}  (mesh_params defaults {texture:true,
#            octree_resolution:256} — clownworld's; accepted optionally, never required)
#     -> 202 {"job_id", "name", "slug", "kind": "identity_from_video"}
#     ONE chained ``video_characters_glb`` job on the render service (char360 + one
#     Hunyuan3D GLB per detected character), relayed by runners/identity_from_video.py.
#     On done the runner creates/refreshes ONE profile per character: ``slug`` for the
#     first, ``<slug>-2``, ``<slug>-3`` … for extras; reference images = the char360
#     view crops (front first); GLB + mesh state attached via the existing helpers.
#     Poll GET /video/jobs/<job_id>: ``progress`` carries the service's stage / progress
#     / log_tail; the terminal ``result.identities`` lists the profiles created.
#     An unconfigured/unreachable render service fails the JOB as data (the route
#     still 202s); the source uri is jailed exactly like /video-extract.
#
# POST /video/identity-profiles/from-images
#     Body: {"sources": [<MediaRef image> | <path>, …], "name": "…", "mesh_params"?}
#     -> 202 {"job_id", "recon_id", "slug", "profile", "kind": "identity_mesh_build"}
#     = create the profile (the SAME validation + copy path as POST
#     /video/identity-profiles) + the one-click full-identity build
#     (``_enqueue_full_identity``: mesh -> turntable -> auto-promoted canonical, front
#     auto-selected by the fleet-VLM step) chained server-side in ONE request. A
#     duplicate name is a 409 (nothing enqueued), like the create route.
# --------------------------------------------------------------------------- #
def _name_and_slug_from_body(body):
    """``(name, slug, error_tuple_or_None)`` for the k94 create routes."""
    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        return None, None, ({"error": "name is required"}, 400)
    name = name.strip()
    slug = identity_profiles.slugify(name)
    if not slug:
        return None, None, ({"error": "name has no url-safe characters"}, 400)
    return name, slug, None


@video_bp.route("/video/identity-profiles/from-video", methods=["POST"])
def video_identity_profiles_from_video():
    body = request.get_json(silent=True) or {}

    # source: a MediaRef-shaped dict for a VIDEO, rehydrated + jailed exactly like the
    # /video-extract route (the runner forwards its uri as video_path — never an
    # arbitrary file).
    source_d = body.get("source")
    if not isinstance(source_d, dict):
        return jsonify({"error": "missing or invalid 'source' MediaRef (a video)"}), 400
    try:
        source = make_media_ref(**source_d)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"invalid source MediaRef: {exc}"}), 400
    if source.kind != "video":
        return jsonify({"error": f"source must be a video; got kind={source.kind!r}"}), 400
    if _jail_resolve(source.uri) is None:
        return jsonify({"error": "source video is outside the storage jail"}), 400

    name, slug, nerr = _name_and_slug_from_body(body)
    if nerr is not None:
        payload, status = nerr
        return jsonify(payload), status

    mesh_params = body.get("mesh_params")
    if mesh_params is not None and not isinstance(mesh_params, dict):
        return jsonify({"error": "mesh_params must be an object"}), 400

    try:
        spec = make_identity_from_video(
            source=source, name=name, mesh_params=mesh_params, identity_id=slug)
    except (ValueError, TypeError) as exc:  # bad fields = 400
        return jsonify({"error": str(exc)}), 400

    job_id = _video_enqueue("identity_from_video", spec)
    return jsonify({"job_id": job_id, "name": name, "slug": slug,
                    "kind": "identity_from_video"}), 202


@video_bp.route("/video/identity-profiles/from-images", methods=["POST"])
def video_identity_profiles_from_images():
    body = request.get_json(silent=True) or {}

    name, _slug, nerr = _name_and_slug_from_body(body)
    if nerr is not None:
        payload, status = nerr
        return jsonify(payload), status

    # sources: MediaRef-shaped dicts (kind image) or bare paths — either way every path
    # runs the SAME jail + ingest + image-classify validation as POST create.
    sources = body.get("sources")
    if not isinstance(sources, list) or not sources:
        return jsonify({"error": "sources must be a non-empty list of image MediaRefs"}), 400
    raw_paths: list = []
    for s in sources:
        if isinstance(s, dict):
            try:
                ref = make_media_ref(**s)
            except (ValueError, TypeError) as exc:
                return jsonify({"error": f"invalid source MediaRef: {exc}"}), 400
            if ref.kind != "image":
                return jsonify({"error": f"every source must be an image; got kind={ref.kind!r}"}), 400
            raw_paths.append(ref.uri)
        elif isinstance(s, str):
            raw_paths.append(s)
        else:
            return jsonify({"error": "each source must be a MediaRef object or a path"}), 400
    resolved, err = _validate_profile_reference_images(raw_paths)
    if err is not None:
        payload, status = err
        return jsonify(payload), status

    mesh_params = body.get("mesh_params")
    if mesh_params is not None and not isinstance(mesh_params, dict):
        return jsonify({"error": "mesh_params must be an object"}), 400
    mesh_params = dict(mesh_params or {})

    private, perr = _parse_private_flag(body)  # t171: optional, default False == PUBLIC
    if perr is not None:
        payload, status = perr
        return jsonify(payload), status
    try:
        profile = identity_profiles.create_profile(
            name, resolved, notes="", owner=_request_principal(), private=private)
    except identity_profiles.ProfileError as exc:  # dup slug / bad shape = errors-as-data
        status = 409 if exc.code == "duplicate" else 400
        return jsonify({"error": str(exc), "code": exc.code}), status

    # The one-click full-identity build, on the just-created profile. The happy path is
    # clownworld's mesh_params (textured, octree 256); everything else is the /generate
    # default (turntable chained, canonical auto-promoted, fleet-VLM front select).
    gen_body: dict = {
        "texture": bool(mesh_params.get("texture", True)),
        "octree_resolution": mesh_params.get("octree_resolution", 256),
    }
    resp, status = _enqueue_full_identity(profile["slug"], profile, gen_body)
    if status != 200:
        # The profile exists (a valid, bindable reference set) but the build could not be
        # enqueued — say so honestly rather than hiding the created profile.
        resp = dict(resp)
        resp["profile"] = profile
        resp["slug"] = profile["slug"]
        return jsonify(resp), status
    resp = dict(resp)
    resp.update({"slug": profile["slug"], "profile": profile, "kind": "identity_mesh_build"})
    return jsonify(resp), 202


# --------------------------------------------------------------------------- #
# 5i) POST /video/identity-profiles/<slug>/generate
#     ONE-CLICK FULL IDENTITY GENERATION — the template that turns a saved identity
#     PROFILE into a complete 3D identity in a single action: a Hunyuan3D mesh, a
#     true-360° Blender turntable, and (by default) auto-promoted CANONICAL reference
#     angles. Usable on ANY profile that carries 1..12 reference images — NO prior
#     reconstruction / character-sheet / angle-ring approval is required (it mints a
#     fresh recon_id and drives the whole chain off the profile's own refs).
#
#     Body — all optional; the DEFAULTS are the happy path (a bare {} is the intended
#     call):
#       views?      {front|right|back|left: <path>} — each an EXPLICIT override; every
#                   path must be one of THIS profile's own reference/canonical images
#                   (the shared jail — an arbitrary path is a clean 400). Omitted views
#                   fall back to the default front (canonical[0]-on-disk else the first
#                   existing reference image), exactly like the 5g mesh route.
#       texture?    bool (default TRUE) — bake a texture onto the mesh (the template's happy path).
#       turntable?  {frame_count?, fps?, width?, height?, elevation_deg?, transparent?}
#                   — orbit render knobs (defaults 72 / 24 / 768 / 768 / 8.0 / False).
#       auto_promote? bool (default True) — promote the 4 cardinal turntable frames to
#                   canonical AFTER the render, but ONLY when canonical is still empty
#                   (a curated canonical set is never clobbered; the relay re-reads it).
#     Always chains the turntable (the "full identity" always renders mesh -> 360°).
#     Returns {job_id, recon_id} 200; 404 for an unknown profile.
# --------------------------------------------------------------------------- #
_POSE_CHOICES = ("none", "t-pose")

# T-pose render geometry — the Wan-VACE id_lock ceiling is 480p (a 512² id_lock render
# fails ``no_capable_model``); a square 480×480 portrait matches the movie/reconstruction
# id_lock default, so the T-pose still snaps to it. The capability probe builds a spec at
# THIS geometry so the routing decision it makes matches the render the relay will run.
# Shared with runners/identity_render_relay._render_pose_front (kept identical there).
_TPOSE_RENDER_W, _TPOSE_RENDER_H, _TPOSE_RENDER_FPS = 480, 480, 16


def _pose_stage_capable(slug: str) -> bool:
    """Whether the T-pose pose-normalization RENDER STAGE can actually run on this
    deployment RIGHT NOW.

    Slice 5 (IDENTITY-VERSIONS-SLICE.md) is the render stage: the relay renders ONE
    id_lock T-pose STILL on the Wan-VACE path (studio worker on ae, ~6GB free on the
    3090) and meshes THAT instead of the crossed-arm source photo. Whether that render
    will land on a GPU worker — versus silently falling to central's GPU-less in-process
    path, where an id_lock render can only fail — is the EXACT same question the studio
    delegation layer already answers for every VACE render: ``should_delegate(spec)``
    returns True iff (a) a studio worker is RESOLVABLE (``HUGPY_STUDIO_WORKER`` set, or
    the registry-based resolver once studio models are first-class rows) AND (b) the
    request binds a REAL (non-synthetic) VACE model at the worker's autofit budget. That
    predicate reads the in-process worker registry + the capability router — no network
    probe — so it is cheap enough to run on the request path (defaults-are-promises: we
    only advertise "capable" when the render will really delegate to a GPU that owns the
    model).

    We reuse that decision instead of hand-rolling a ``comfy.id_lock`` capability check:
    the T-pose still is a Wan-VACE reference-to-video render, NOT a ComfyUI-IPAdapter
    still, so ``comfy.id_lock`` is the wrong signal — the right signal is "will an
    id_lock studio render delegate to a worker that serves the VACE model". We build a
    REPRESENTATIVE id_lock ``studio_i2v`` spec at the reconstruction id_lock ceiling
    (480×480, the same geometry the relay's T-pose render will use) and ask
    ``should_delegate`` about it.

    FAIL-CLOSED: any exception — an import failure, a registry read hiccup, a router
    error — degrades to False so a ``pose: "t-pose"`` request falls back to the normal
    front (honest not-capable notice) rather than enqueuing a build that would fail its
    pose render. ``slug`` is accepted for signature stability (a future per-identity gate
    could consult it) but the capability is deployment-wide today.

    Kept a FUNCTION so the probe lives in ONE place and a test can exercise the capable
    branch by monkeypatching it (or by pointing HUGPY_STUDIO_WORKER at a real registry
    row)."""
    try:
        # Lazy imports — the studio spine + registry are heavy and must not load at
        # module import time (this route module imports at app boot). Mirrors the
        # relay's lazy-import discipline.
        from hugpy_video.intel.runners.studio_i2v import should_delegate
        from hugpy_video.intel.studio.job import make_studio_i2v

        # A representative id_lock still spec at the VACE 480p id_lock ceiling — the SAME
        # geometry the relay's T-pose render uses (see _render_pose_front). A single
        # placeholder reference image is enough for the routing decision (should_delegate
        # reads the worker registry + the capability router; it does not open the file).
        probe = make_studio_i2v(
            capability="id_lock",
            width=_TPOSE_RENDER_W,
            height=_TPOSE_RENDER_H,
            fps=_TPOSE_RENDER_FPS,
            vram_budget_gb=None,        # autofit — exactly what the real render uses
            seed=0,
            prompt="capability probe",
            reference_images=("__probe__",),
        )
        return bool(should_delegate(probe))
    except Exception:  # noqa: BLE001 — fail-closed: any probe failure => not capable
        logger.debug("pose-stage capability probe failed for %s", slug, exc_info=True)
        return False


@video_bp.route("/video/identity-profiles/<slug>/generate", methods=["POST"])
def video_identity_profile_generate(slug):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    if not _owns_profile(profile.get("owner")):  # t171: owner/operator only
        return _deny_profile(profile)

    body = request.get_json(silent=True) or {}
    resp, status = _enqueue_full_identity(slug, profile, body)
    return jsonify(resp), status


def _enqueue_full_identity(slug, profile, body):
    """The ONE-CLICK full-identity enqueue (the /generate body above), factored (k94) so
    POST /video/identity-profiles/from-images can chain "create profile" + this in ONE
    request without copying it. Returns ``(payload_dict, http_status)``; the caller
    jsonifies. ``body`` follows the /generate contract exactly (all optional), plus an
    additive optional ``octree_resolution`` (positive int; default 380 as before)."""
    # Cardinal view map, JAILED to the profile's own images (shared with the 5g route).
    view_map, view_candidates, verr = _resolve_profile_mesh_views(profile, body.get("views"))
    if verr is not None:
        payload, status = verr
        return payload, status

    # POSE NORMALIZATION (IDENTITY-VERSIONS-SLICE.md slice 3): optional pose, validated
    # none|t-pose. The t-pose RENDER STAGE (render an id_lock T-pose still and mesh THAT to
    # clear crossed-arm occlusion) is slice 5 — gated here behind a capability check.
    #   * pose="none" (or absent) -> today's behavior EXACTLY (no extra response fields).
    #   * pose="t-pose" + capable  -> the spec carries pose so the relay renders the T-pose
    #                                 front; response notes it applied.
    #   * pose="t-pose" + NOT capable (today) -> the build STILL proceeds off the normal
    #     front (honest fallback), and the response carries a structured not-capable notice
    #     so the caller knows the normalization was not applied. A malformed pose is a 400.
    pose_req = body.get("pose", "none")
    if pose_req is None:
        pose_req = "none"
    if not isinstance(pose_req, str) or pose_req not in _POSE_CHOICES:
        return {"error": f"pose must be one of {list(_POSE_CHOICES)}"}, 400
    effective_pose = "none"
    pose_notice = None
    if pose_req == "t-pose":
        if _pose_stage_capable(slug):
            effective_pose = "t-pose"
            pose_notice = {"requested": "t-pose", "applied": True, "capable": True}
        else:
            pose_notice = {
                "requested": "t-pose",
                "applied": False,
                "capable": False,
                "code": "pose_stage_unavailable",
                "message": ("the T-pose normalization render stage is not yet available "
                            "on this deployment; generated from the source pose instead"),
            }

    # A fresh, self-describing reconstruction id — the whole chain (mesh + turntable +
    # canonical) hangs off it; no prior reconstruction is needed.
    recon_id = "identity_" + secrets.token_hex(8)

    # Optional knobs — a malformed optional value falls back to its default (a bad tuning
    # value should not hard-fail the build; make_identity_mesh re-validates regardless).
    def _pos_int(d, name, default):
        v = d.get(name, default)
        return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else default

    tt = body.get("turntable") if isinstance(body.get("turntable"), dict) else {}
    elev = tt.get("elevation_deg", 8.0)
    if not isinstance(elev, (int, float)) or isinstance(elev, bool):
        elev = 8.0

    # PER-IDENTITY VISION MODEL (operator-requested): the VL model the mesh relay's
    # FRONT-SELECT step uses to pick the full-body reference before meshing. Precedence:
    # an EXPLICIT request-body ``vision_model`` wins; else the identity's PERSISTED
    # ``gen_settings.vision_model`` (what the Settings tab saves); else None (== the
    # fleet-default VL model — the relay sends no ``model`` field, byte-identical to before
    # this setting existed). ``profile`` is already the PUBLIC shape here, so its
    # ``gen_settings`` is the full defaulted block; ``make_identity_mesh`` normalizes
    # ""/whitespace to None. This is why a BARE one-click still honors the identity's saved
    # choice even when the UI omits the field from the body.
    _gen_settings = profile.get("gen_settings") or {}
    vision_model = body.get("vision_model")
    if vision_model in (None, ""):
        vision_model = _gen_settings.get("vision_model")
    if vision_model in (None, ""):
        # AUTO default (operator 2026-07-15): prefer a 7B VL when the fleet has one —
        # the 3B mislabeled a waist-up ref as full-body. None when no 7B is installed
        # (== fleet-default 3B); the relay degrades a failing 7B call back to the 3B.
        vision_model = identity_profiles.preferred_identity_vision_model()

    # CLEANUP-PROMPT slice (C4 — the reachability wire): same precedence pattern as
    # vision_model above — an EXPLICIT request-body value wins; else the identity's
    # PERSISTED gen_settings value (what the new Advanced-panel field saves); else ""
    # (== today's exact render, defaults-are-promises). make_identity_mesh coerces
    # None -> "" regardless, so a bare one-click still honors the profile's saved
    # cleanup/negative steer even when the UI body omits the fields.
    cleanup_prompt = body.get("cleanup_prompt")
    if cleanup_prompt in (None, ""):
        cleanup_prompt = _gen_settings.get("cleanup_prompt", "")
    negative_prompt = body.get("negative_prompt")
    if negative_prompt in (None, ""):
        negative_prompt = _gen_settings.get("negative_prompt", "")

    try:
        spec = make_identity_mesh(
            slug=slug,
            recon_id=recon_id,
            view_sources=tuple(view_map.items()),
            # TEXTURE DEFAULTS TRUE on the one-click template (operator 2026-07-14 night:
            # only the explicitly-textured run "is textured correctly of the 3" — the UI
            # sends a bare body, so the default IS the promise; the paint path is proven
            # and costs ~1-2 extra minutes). ``"texture": false`` opts a run out; the
            # per-reconstruction 5g mesh route keeps texture opt-in for surgical builds.
            texture=bool(body.get("texture", True)),
            chain_turntable=True,   # the full-identity template always renders the 360°
            auto_promote=bool(body.get("auto_promote", True)),
            # k94 (additive): the from-images path passes clownworld's leaner 256 so the
            # texture bake doesn't take ~24 min/character; a bare /generate keeps 380.
            octree_resolution=_pos_int(body, "octree_resolution", 380),
            frame_count=_pos_int(tt, "frame_count", 72),
            fps=_pos_int(tt, "fps", 24),
            width=_pos_int(tt, "width", 768),
            height=_pos_int(tt, "height", 768),
            elevation_deg=elev,
            transparent=bool(tt.get("transparent", False)),
            view_candidates=view_candidates,
            pose=effective_pose,
            vision_model=vision_model,
            cleanup_prompt=cleanup_prompt,
            negative_prompt=negative_prompt,
        )
    except (ValueError, TypeError) as exc:  # bad fields = 400
        return {"error": str(exc)}, 400

    # Seed mesh state to "queued" so GET .../reconstruction/<recon_id>/mesh + the UI
    # reflect the in-flight build immediately (set_mesh_state creates the record for this
    # brand-new recon_id — the mesh-first flow — then the relay records the terminal state).
    try:
        identity_profiles.set_mesh_state(slug, recon_id, {"status": "queued", "error": None})
    except identity_profiles.ProfileError:
        pass

    job_id = _video_enqueue("identity_mesh_build", spec)
    resp = {"job_id": job_id, "recon_id": recon_id}
    # Only surface a ``pose`` block when t-pose was explicitly requested — a bare click
    # (pose="none") keeps the exact {job_id, recon_id} shape it has today.
    if pose_notice is not None:
        resp["pose"] = pose_notice
    return resp, 200


# --------------------------------------------------------------------------- #
# 5j) PATCH /video/identity-profiles/<slug>/settings — VERSIONS slice: persist the
#     per-identity generation defaults the left-column Settings tab edits and a
#     bare /generate click honors. Body IS the partial gen_settings object itself
#     (NOT nested under a "gen_settings" key — matches identityProfileSettingsUrl's
#     client, which PATCHes `fields` directly). A true partial merge — an omitted
#     key is left untouched (identity_profiles.set_gen_settings's contract); the
#     store rejects an unknown key, a wrong-typed value, an out-of-enum pose, or a
#     front_ref outside the profile's own reference images, all as a clean 400
#     (ProfileError, never a 500). Unknown slug -> 404, same message/shape as the
#     other identity-profile routes.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/settings", methods=["PATCH"])
def video_identity_profile_settings(slug):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    if not _owns_profile(profile.get("owner")):  # t171: owner/operator only
        return _deny_profile(profile)

    body = request.get_json(silent=True) or {}
    try:
        profile = identity_profiles.set_gen_settings(slug, body)
    except identity_profiles.ProfileError as exc:  # unknown key / bad type = clean 400
        return jsonify({"error": str(exc), "code": exc.code}), 400
    if profile is None:  # lost a race with a concurrent archive
        return jsonify({"error": "identity profile not found"}), 404
    return jsonify({"profile": profile}), 200


# --------------------------------------------------------------------------- #
# 5k) POST /video/identity-profiles/<slug>/versions/<version_id>/activate —
#     VERSIONS slice: point the identity's ACTIVE version at <version_id> (the
#     id_lock DNA source future generations resolve to by default, via
#     _reference_images_from_body). No body. 404 when the slug is unknown OR
#     version_id names no active (non-archived) version of it —
#     set_active_version returns None for either case; the slug is checked first
#     so an unknown slug gets the same message every other identity-profile route
#     gives it, and an unknown/archived version_id is called out by name.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/versions/<version_id>/activate", methods=["POST"])
def video_identity_profile_activate_version(slug, version_id):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    if not _owns_profile(profile.get("owner")):  # t171: owner/operator only
        return _deny_profile(profile)

    profile = identity_profiles.set_active_version(slug, version_id)
    if profile is None:
        return jsonify({"error": f"version {version_id!r} not found"}), 404
    return jsonify({"profile": profile}), 200


# --------------------------------------------------------------------------- #
# 5l) PATCH /video/identity-profiles/<slug>/versions/<version_id> — VERSIONS
#     slice: rename/annotate one version. Body {name?, notes?}, both optional (a
#     true partial update — an omitted key is left untouched, mirrors the
#     PATCH /<slug> profile-edit route's **kwargs idiom above). A blank name is a
#     clean 400 (checked here exactly like the profile-edit route, before ever
#     calling the store — update_version also guards it, belt-and-suspenders).
#     404 when the slug or version_id is unknown/archived.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/versions/<version_id>", methods=["PATCH"])
def video_identity_profile_update_version(slug, version_id):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    if not _owns_profile(profile.get("owner")):  # t171: owner/operator only
        return _deny_profile(profile)

    body = request.get_json(silent=True) or {}
    kwargs: dict = {}

    if "name" in body:
        name = body.get("name")
        if not isinstance(name, str) or not name.strip():
            return jsonify({"error": "name is required"}), 400
        kwargs["name"] = name

    if "notes" in body:
        notes = body.get("notes")
        if notes is not None and not isinstance(notes, str):
            return jsonify({"error": "notes must be a string"}), 400
        kwargs["notes"] = notes or ""

    try:
        profile = identity_profiles.update_version(slug, version_id, **kwargs)
    except identity_profiles.ProfileError as exc:  # errors-as-data, never a 500
        return jsonify({"error": str(exc), "code": exc.code}), 400
    if profile is None:
        return jsonify({"error": f"version {version_id!r} not found"}), 404
    return jsonify({"profile": profile}), 200


# --------------------------------------------------------------------------- #
# 5m) DELETE /video/identity-profiles/<slug>/versions/<version_id> — VERSIONS
#     slice: ARCHIVE a version (never-delete: flagged, bytes kept, dropped from
#     the wire list). The store REFUSES the clay base (the geometric ground
#     truth) and the currently ACTIVE version — either surfaces here as the
#     store's own ProfileError message/code turned into a clean 400 (the check
#     lives in identity_profiles.archive_version; this route does not duplicate
#     it). 404 when the slug or version_id is unknown/already archived.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/versions/<version_id>", methods=["DELETE"])
def video_identity_profile_archive_version(slug, version_id):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    if not _owns_profile(profile.get("owner")):  # t171: owner/operator only
        return _deny_profile(profile)

    try:
        profile = identity_profiles.archive_version(slug, version_id)
    except identity_profiles.ProfileError as exc:  # base/active refusal = clean 400
        return jsonify({"error": str(exc), "code": exc.code}), 400
    if profile is None:
        return jsonify({"error": f"version {version_id!r} not found"}), 404
    return jsonify({"profile": profile}), 200
