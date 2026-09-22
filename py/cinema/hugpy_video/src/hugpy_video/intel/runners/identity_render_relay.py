"""Identity 3D MESH (+ turntable) bus runner — a RELAY to a remote GPU render service.

Central has NO GPU: it must never import torch / hy3dgen / a diffusion pipeline for a
mesh build. This runner is a thin HTTP CLIENT to the ``IDENTITY_RENDER_URL`` service
(a sibling agent builds that service to the FIXED contract mirrored below). It:

  1. Reads the identity's cardinal reference images (jailed by the route to the
     profile's OWN refs) and base64-encodes them.
  2. POSTs a job to ``<IDENTITY_RENDER_URL>/jobs`` (kind ``mesh_and_turntable`` when the
     spec chains a turntable, else ``mesh_build``), authenticating with the
     ``X-Identity-Render-Token`` header.
  3. POLLs ``GET /jobs/<id>`` every ~5s until done / error / a generous deadline, honoring
     a cooperative cancel (relayed as a best-effort ``DELETE /jobs/<id>``).
  4. On done, DOWNLOADS every produced file and PERSISTS it under the identity's own dir
     (``<IDENTITIES_HOME>/<slug>/mesh/<recon_id>/``) via ``identity_profiles`` path
     helpers — the GLB + mesh json at the mesh root, the turntable frames + mp4 under
     ``turntable/``.
  5. ATTACHES the turntable frames as a ``reconstructions`` entry (reusing
     ``attach_reconstruction`` with ``replace=True`` so the EXISTING promote route works
     on the turntable output) and RECORDS mesh state (``set_mesh_state``) so the GET
     mesh-status route surfaces status + GLB + mp4.

Pure ``(IdentityMeshSpec, job_id) -> JobResult`` (map §6): EVERY expected failure — an
unconfigured service, an unreachable host, a 401, a render error, a timeout — is DATA
(``JobResult(ok=False, JobError(...))``), never a raise. Only a genuine programmer error
raises, and ``media_bus.run_claimed`` is the one place that catches that.

Heavy imports (requests, identity_profiles, media_schema, media_bus) are LAZY — done
inside the runner — so importing this module (which ``runners/__init__`` does at boot)
stays cheap and can never break app boot. No pathlib anywhere; os.path only.

Remote service contract (FIXED — the service is built to exactly this):
  * ``GET  /health``                     -> 200 {ok, service, capabilities}
  * ``POST /jobs``                       -> 202 {job_id}
  * ``GET  /jobs/<id>``                  -> {job_id, status, error?, files?,
                                           stage?, progress?, log_tail?, updated?}
       stage (str), progress (0..1 float), log_tail (last <=40 lines, newest-last),
       updated (float epoch) are ADDITIVE live-progress fields — they MAY be absent
       (an older service); the relay mirrors whatever it sees into GET /llm/jobs and
       degrades gracefully when they are missing.
  * ``GET  /jobs/<id>/files/<path>``     -> raw bytes
  * ``DELETE /jobs/<id>``                -> best-effort cleanup
Auth: header ``X-Identity-Render-Token: <IDENTITY_RENDER_TOKEN>`` (missing/wrong -> 401).

FLEET-VLM FRONT AUTO-SELECTION (keeper 2026-07-14): the route only ever defaults
``front`` to the profile's FIRST source reference image — if that photo is a cropped
waist-up portrait, the mesh comes out with cut-off legs (happened live: luigi ref_00).
When the route did NOT see an explicit ``views.front`` override, it hands this runner
``spec.view_candidates`` (every existing source ref, in order) instead of guessing.
HERE — inside the bus job, never the HTTP handler, because each vision call is
~5-10s and there can be up to 12 candidates — ``_select_front_view`` asks hugpy's own
``/ml/vision`` amenity, in order, "does this show the character's full body?" and the
first "yes" becomes front. Any failure (network, non-200, ambiguous text, the
``IDENTITY_FRONT_AUTOSELECT`` kill-switch) is a soft skip — the default front from the
route is always a safe fallback, so selection NEVER fails the job.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import secrets
import time

from hugpy_video.intel.result_schema import JobError, JobResult
from hugpy_video.intel.runners import identity_render_client as _client

logger = logging.getLogger(__name__)

# HTTP budget / poll cadence / deadline now live in identity_render_client (k94 —
# factored out so the three identity relays share ONE submit/poll/download/persist
# path). The old module-level names stay bound as aliases for any reader/test.
_HTTP_TIMEOUT_S = _client.HTTP_TIMEOUT_S
_POLL_INTERVAL_S = _client.POLL_INTERVAL_S
_POLL_DEADLINE_S = _client.POLL_DEADLINE_S

# ---- fleet-VLM front auto-selection (module docstring) ------------------------- #
# Per-candidate HTTP budget as a (connect, read) tuple — a vision call is ~5-10s, so
# the read side stays generous; the connect side stays short (localhost — a down
# amenity fails fast, not a hang). Overridable via env for tests / tuning.
_VISION_TIMEOUT_S = (
    float(os.getenv("IDENTITY_FRONT_VISION_CONNECT_TIMEOUT_S", "10") or "10"),
    float(os.getenv("IDENTITY_FRONT_VISION_READ_TIMEOUT_S", "60") or "60"),
)
_VISION_MAX_CANDIDATES = 12
_VISION_PROMPT = (
    "Look at the character in this image. Does the image show the character's ENTIRE "
    "body from head to feet, with the feet fully visible (not cropped by the image "
    "edge)? Answer with exactly one word: yes or no."
)


def _autoselect_enabled() -> bool:
    """The ``IDENTITY_FRONT_AUTOSELECT`` kill-switch — default ON; "off"/"0"/"false"
    (any case) disables fleet-VLM front selection entirely."""
    v = (os.getenv("IDENTITY_FRONT_AUTOSELECT", "") or "").strip().lower()
    return v not in ("off", "0", "false")


def _reply_says_yes(text) -> bool:
    """A clear, unambiguous "yes" as the FIRST word of the vision reply (tolerating
    leading punctuation/markdown like ``**Yes.**`` or ``"Yes,"``) — a longer hedge that
    merely mentions "yes" later ("no, but yes the head is visible...") does not count."""
    if not isinstance(text, str) or not text.strip():
        return False
    cleaned = re.sub(r"^[^a-zA-Z]+", "", text.strip())
    if not cleaned:
        return False
    first_word = cleaned.split(None, 1)[0].rstrip(".,!:;\"'*_")
    return first_word.lower() == "yes"


def _select_front_view(candidates, slug: str, requests_mod, model=None):
    """Ask hugpy's own fleet vision amenity (``POST /ml/vision``), IN ORDER (capped at
    ``_VISION_MAX_CANDIDATES``), which candidate reference image shows the character's
    FULL BODY. Returns ``(chosen_path_or_None, checked_count)`` — the first candidate
    with a clear "yes" wins. EVERY failure (unreadable file, unreachable amenity, a
    non-200, a non-JSON/not-ok reply, an ambiguous answer) is a soft skip logged at
    info: this never raises, so a flaky vision call can never fail the mesh job.

    ``model`` is the per-identity VISION MODEL (``IdentityMeshSpec.vision_model``, resolved
    by the route from the identity's gen_settings). When set (a non-empty key), it is
    passed as ``"model"`` in the /ml/vision body so THIS front-select runs on that VL model
    (e.g. a 7B). When None/empty, NO ``model`` field is sent — byte-identical to before this
    setting existed, so /ml/vision falls back to ``DEFAULT_VISION_MODEL`` (the 3B). A bad or
    slow chosen model still cannot fail the job: the soft-skip fallthrough already keeps the
    route's default front on any miss."""
    base = (os.getenv("HUGPY_CENTRAL_URL", "") or "http://127.0.0.1:7002").strip().rstrip("/")
    url = f"{base}/ml/vision"
    checked = 0
    for path in candidates[:_VISION_MAX_CANDIDATES]:
        checked += 1
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            logger.info("identity mesh front-select: candidate unreadable (%s)", path)
            continue
        b64 = base64.b64encode(raw).decode("ascii")
        # Only include ``model`` when one was chosen (per-identity setting or the auto-7B
        # preference) — an absent field keeps the default-VL (3B) path byte-identical to
        # today (defaults-are-promises). A FAILED with-model call retries once WITHOUT the
        # model, so a 7B that can't load right now degrades to the 3B's judgment instead
        # of no judgment at all ("7B if available" — availability is proven per call).
        reply = None
        for try_model in ((model, None) if model else (None,)):
            body = {"image_b64": b64, "prompt": _VISION_PROMPT}
            if try_model:
                body["model"] = try_model
            try:
                resp = requests_mod.post(url, json=body, timeout=_VISION_TIMEOUT_S)
            except requests_mod.RequestException as exc:
                logger.info("identity mesh front-select: /ml/vision unreachable for %s "
                            "(model=%s: %s)", path, try_model or "default", exc)
                continue
            if resp.status_code != 200:
                logger.info("identity mesh front-select: /ml/vision HTTP %s for %s "
                            "(model=%s)", resp.status_code, path, try_model or "default")
                continue
            try:
                parsed = resp.json()
            except ValueError:
                logger.info("identity mesh front-select: non-JSON /ml/vision reply for %s", path)
                continue
            if not isinstance(parsed, dict) or parsed.get("ok") is False:
                logger.info("identity mesh front-select: /ml/vision reported not-ok for %s "
                            "(model=%s)", path, try_model or "default")
                continue
            reply = parsed
            break
        if reply is None:
            continue
        if _reply_says_yes(reply.get("text")):
            logger.info("identity mesh front-select: %s qualifies as full-body (profile %s)",
                        path, slug)
            return path, checked
    return None, checked


# ---- T-POSE POSE-NORMALIZATION STAGE (IDENTITY-VERSIONS-SLICE.md slice 5) ---------- #
# Hunyuan3D meshes the INPUT pose: luigi's crossed arms occlude the torso and leave "the
# unknown below them" (operator's words). The fix is 2D pose normalization BEFORE meshing
# — render ONE identity-locked STILL of the character standing in a clean T-pose (arms
# outstretched, full body, facing camera), and mesh THAT as the front. The still is
# produced on the EXISTING Wan-VACE id_lock render path (``_render_identity_view``, the
# reconstruction runner's SWAP SEAM) — this is that machinery's second life, exactly as
# the slice doc anticipated.
#
# Geometry: the Wan-VACE id_lock ceiling is 480p (a 512² id_lock render fails
# ``no_capable_model``); a square 480×480 @ 16fps matches the movie/reconstruction
# id_lock default. KEEP THESE IDENTICAL to video_routes._TPOSE_RENDER_* so the route's
# capability probe (which builds an id_lock spec at this geometry to decide "will this
# delegate to a GPU worker") matches the render actually run here.
_TPOSE_RENDER_W, _TPOSE_RENDER_H, _TPOSE_RENDER_FPS = 480, 480, 16
# The pose-normalization prompt. Front-facing, arms out, full body head-to-feet — the
# ambiguity-clearing stance. No profile DESCRIPTION is woven in: the reference images
# already carry the identity (id_lock), and an extra description would only fight the
# T-pose stance we are trying to force. A plain neutral background helps the service-side
# rembg (background removal happens on the render service, applied to the front it meshes).
#
# CLEANUP-PROMPT slice (operator-requested 2026-07-15): this is the BASE stance. A cleanup
# clause (``IdentityMeshSpec.cleanup_prompt`` — a positive-worded AVOID instruction like
# "no object on her back, clean bare back") is APPENDED to it by ``_render_pose_front`` when
# non-empty. That relaxes the "no description woven in" policy for CLEANUP only: removing a
# prop does NOT fight the stance (unlike an identity description would). Empty cleanup_prompt
# -> this exact constant, byte-identical to today. ``_tpose_prompt`` centralizes the assembly.
_TPOSE_PROMPT = (
    "the same person standing in a T-pose, arms outstretched horizontally to the sides, "
    "full body visible from head to feet, feet fully in frame, facing the camera, "
    "neutral standing stance, plain neutral background, even lighting"
)


def _tpose_prompt(cleanup_prompt: str = "") -> str:
    """Assemble the effective T-pose front-render prompt: ``_TPOSE_PROMPT`` verbatim, plus
    ``", " + cleanup_prompt`` when a non-empty cleanup clause is supplied.

    An empty (or whitespace-only) ``cleanup_prompt`` returns the constant UNCHANGED — the
    byte-identical-to-today guarantee (defaults-are-promises). The cleanup clause is a
    positive-worded AVOID instruction (a prop-removal), NOT an identity description, so it
    is allowed to ride the stance prompt (see the ``_TPOSE_PROMPT`` note above)."""
    clause = (cleanup_prompt or "").strip()
    return f"{_TPOSE_PROMPT}, {clause}" if clause else _TPOSE_PROMPT


def _render_pose_front(refs, seed: int, slug: str, job_id: str, should_cancel=None,
                       cleanup_prompt: str = "", negative_prompt: str = ""):
    """Render ONE identity-locked T-pose STILL from the profile's reference images and
    return its abs path, or ``None`` on ANY failure.

    Reuses ``runners/identity_reconstruction._render_identity_view`` — the working
    Wan-VACE id_lock render seam (delegated to the studio GPU worker on ae) — prompted
    for a clean, full-body T-pose. VRAM contention resolves via the normal studio LRU-
    yield (eviction is unblocked as of 0.1.178): this path routes through the standard
    studio delegation, so we do NOT build a bespoke evictor.

    CLEANUP-PROMPT slice: ``cleanup_prompt`` (a positive-worded avoid instruction from
    ``IdentityMeshSpec.cleanup_prompt``) is woven into the effective T-pose prompt via
    ``_tpose_prompt`` when non-empty; ``negative_prompt`` (from
    ``IdentityMeshSpec.negative_prompt``) is forwarded as a TRUE negative to the studio
    Wan-VACE render. BOTH default "" -> the effective prompt is the constant verbatim and
    the negative is "" — byte-identical to today's render call.

    HONEST DEGRADE (the ``IdentityMeshSpec.pose`` contract): a failed pose render must
    NEVER fail the mesh job. Every failure — the studio worker offline, a render error,
    a timeout, an import hiccup — returns ``None`` here; the caller then falls back to
    the normal front-select flow. ``_render_identity_view`` already converts an unroutable
    / errored render into a ``None`` (it never raises for expected failures), and we wrap
    the whole thing so even a genuine import/programmer slip degrades rather than killing
    a build whose mesh + turntable would otherwise succeed."""
    try:
        # Lazy import — the reconstruction runner pulls in the studio spine; keep this
        # module boot-cheap (runners/__init__ imports it at boot).
        from hugpy_video.intel.runners.identity_reconstruction import _render_identity_view

        still = _render_identity_view(
            refs, _tpose_prompt(cleanup_prompt), seed,
            width=_TPOSE_RENDER_W, height=_TPOSE_RENDER_H, fps=_TPOSE_RENDER_FPS,
            # A DISTINCT render_id (mirrors the reconstruction runner's per-render ids) so
            # the worker never dedupes this against another render as one "exists" job.
            render_id=f"{job_id}:{slug}:tpose_front",
            should_cancel=should_cancel,
            # CLEANUP-PROMPT slice: forward the TRUE negative to the studio Wan-VACE path.
            # "" (default) -> the studio path's own default negative_prompt="", so today's
            # exact call is reproduced (see _render_identity_view / run_produce_clip).
            negative_prompt=negative_prompt,
        )
        if not still:
            logger.info("identity mesh T-pose stage: render produced no still for %s "
                        "(falling back to front-select)", slug)
            return None
        return still
    except Exception:  # noqa: BLE001 — honest degrade: a pose-render failure never fails the job
        logger.info("identity mesh T-pose stage: render raised for %s (falling back to "
                    "front-select)", slug, exc_info=True)
        return None


# Persist helpers (k94): the shared client owns them; aliases keep the old names.
_dest_for = _client.dest_for
_atomic_write_bytes = _client.atomic_write_bytes


def run_identity_mesh_build(spec, job_id: str) -> JobResult:
    """Relay ``spec`` to the remote render service and persist the result. Returns
    ``JobResult(ok=True, outputs=(turntable mp4,))`` on success (the GLB is recorded in
    the profile's mesh state, which the GET mesh route reads — a GLB is not a MediaRef
    kind). Every expected failure is a clean error-as-data."""
    # Lazy imports keep this module boot-cheap (runners/__init__ imports it at boot).
    from hugpy_video.intel import identity_profiles
    from hugpy_video.intel.media_bus import is_cancelling, set_progress
    from hugpy_video.intel.media_schema import make_media_ref

    slug = spec.slug
    recon_id = spec.recon_id

    def _set_state(patch: dict) -> None:
        try:
            identity_profiles.set_mesh_state(slug, recon_id, patch)
        except Exception:  # noqa: BLE001 — mesh-state is a best-effort mirror; never fail the job on it
            logger.debug("set_mesh_state failed for %s/%s", slug, recon_id, exc_info=True)

    def _fail(code: str, message: str, retryable: bool) -> JobResult:
        _set_state({"status": "error", "error": message})
        return JobResult(job_id=job_id, ok=False,
                         error=JobError(code=code, message=message, retryable=retryable))

    url, token = _client.service_config()
    if not url or not token:
        nc = _client.not_configured_error("mesh builds")
        return _fail(nc.code, nc.message, retryable=nc.retryable)

    import requests  # lazy — present (2.34.2); keeps the module boot-cheap

    headers = _client.auth_headers(token)

    # Cancel probe shared by the T-pose stage (relays the mesh job's cancel down to the
    # id_lock render, mirroring identity_reconstruction's should_cancel). is_cancelling is
    # already imported at the top of this function (used by the poll loop below).
    should_cancel = lambda: is_cancelling(job_id)  # noqa: E731

    view_sources = spec.view_sources or ()
    candidates = tuple(getattr(spec, "view_candidates", ()) or ())

    # ---- T-POSE POSE-NORMALIZATION STAGE (slice 5) — runs BEFORE front-select -------- #
    # When the spec requests ``pose == "t-pose"``, render an identity-locked T-pose STILL
    # (arms out, full body, no crossed-arm occlusion) and use THAT as the mesh front — it
    # OVERRIDES front-select entirely (a clean full-body T-pose is a better front than any
    # source photo we could pick). On ANY failure we fall through to the existing
    # front-select flow: a failed pose render NEVER fails the mesh job (the honest-degrade
    # contract documented on IdentityMeshSpec.pose). The outcome is recorded in mesh state
    # as ``pose_stage`` so the GET mesh-status route surfaces what actually happened.
    #
    # The reference images the id_lock render conditions on are the profile's own refs —
    # every ``view_sources`` path is one, and ``view_candidates`` (when present) is the
    # profile's other refs; we hand the render the DISTINCT set of both (front first) so a
    # cropped waist-up "front" isn't the sole conditioning image.
    pose_requested = (getattr(spec, "pose", "none") or "none") == "t-pose"
    pose_stage = None
    tpose_front = None
    if pose_requested:
        pose_refs: list[str] = []
        for _n, _p in view_sources:
            if isinstance(_p, str) and _p and _p not in pose_refs:
                pose_refs.append(_p)
        for _p in candidates:
            if isinstance(_p, str) and _p and _p not in pose_refs:
                pose_refs.append(_p)
        if not pose_refs:
            pose_stage = {"requested": "t-pose", "applied": False,
                          "reason": "no reference images to condition the T-pose render on"}
        else:
            tpose_front = _render_pose_front(
                pose_refs, spec.seed, slug, job_id, should_cancel=should_cancel,
                # CLEANUP-PROMPT slice: the spec's render-steer channels reach the T-pose
                # render here. Both default "" on the spec -> _render_pose_front renders the
                # exact constant + "" negative (byte-identical to today). getattr keeps an
                # OLD spec (pre-cleanup, no field) working.
                cleanup_prompt=getattr(spec, "cleanup_prompt", "") or "",
                negative_prompt=getattr(spec, "negative_prompt", "") or "")
            if tpose_front:
                pose_stage = {"requested": "t-pose", "applied": True,
                              "rendered_front": tpose_front,
                              "reason": "rendered an id_lock T-pose still; using it as the mesh front"}
                # The rendered T-pose still BECOMES the front and DISABLES front-select
                # (candidates cleared) — there is nothing better to pick than a clean
                # full-body T-pose we just rendered from the identity's own DNA.
                view_sources = tuple(
                    ("front", tpose_front) if n == "front" else (n, p)
                    for n, p in view_sources)
                if not any(n == "front" for n, _ in view_sources):
                    view_sources = (("front", tpose_front),) + tuple(view_sources)
                candidates = ()
            else:
                pose_stage = {"requested": "t-pose", "applied": False,
                              "reason": "T-pose render failed or unavailable; fell back to "
                                        "front-select on the source photos"}

    # ---- FLEET-VLM FRONT AUTO-SELECTION (module docstring) --------------------------- #
    # The route only hands >=2 view_candidates when the caller did NOT explicitly assign
    # a front — an explicit assignment is an operator override that auto-selection never
    # second-guesses. Every branch records an honest ``front_selection`` outcome; a VLM
    # miss/error NEVER fails the job — it just keeps the route's default front.
    # (When the T-pose stage above applied, ``candidates`` is now empty, so front-select
    # short-circuits to the "explicit" branch on the just-rendered T-pose front.)
    default_front = next((p for n, p in view_sources if n == "front"), None)
    front_selection = {"mode": "explicit", "chosen": default_front, "checked": 0,
                       "full_body": None}
    if tpose_front:
        # The T-pose still is the front — record it as the selection mode so mesh state
        # tells the whole story (front came from the T-pose stage, not a source photo).
        front_selection["mode"] = "t-pose"
        front_selection["full_body"] = True
    elif not candidates:
        front_selection["mode"] = "explicit"
    elif len(candidates) < 2:
        # Only one usable reference on the whole profile — nothing to choose between.
        front_selection["mode"] = "default"
    elif not _autoselect_enabled():
        front_selection["mode"] = "disabled"
        logger.info("identity mesh front-select: IDENTITY_FRONT_AUTOSELECT disabled for %s", slug)
    else:
        # spec.vision_model (None when the identity uses the fleet default) selects which
        # VL model the front-select vision calls run on — see _select_front_view.
        chosen, checked = _select_front_view(
            candidates, slug, requests, getattr(spec, "vision_model", None))
        front_selection["mode"] = "vlm"
        front_selection["checked"] = checked
        if chosen is not None:
            front_selection["chosen"] = chosen
            front_selection["full_body"] = True
            view_sources = tuple(
                (n, chosen) if n == "front" else (n, p) for n, p in view_sources)
        else:
            front_selection["full_body"] = False
            logger.info("identity mesh front-select: no full-body candidate found for %s "
                        "(checked %d of %d) — keeping default front",
                        slug, checked, len(candidates))

    # ---- read + base64 the assigned view images (jailed to the profile's own refs) ----
    views_b64: dict[str, str] = {}
    for pair in (view_sources or ()):
        try:
            name, path = pair[0], pair[1]
        except (IndexError, TypeError):
            continue
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            logger.warning("identity mesh: view %r image unreadable (%s)", name, path)
            continue
        views_b64[name] = base64.b64encode(raw).decode("ascii")
    if "front" not in views_b64:
        return _fail(
            "bad_views",
            f"no readable 'front' reference image for the mesh build of profile {slug!r}",
            retryable=False)

    kind = "mesh_and_turntable" if spec.chain_turntable else "mesh_build"
    payload = {
        "kind": kind,
        "identity_id": slug,
        "views": views_b64,
        "mesh_params": {
            "seed": spec.seed,
            "num_inference_steps": spec.num_inference_steps,
            "octree_resolution": spec.octree_resolution,
            "texture": bool(spec.texture),
        },
        "turntable_params": {
            "frame_count": spec.frame_count,
            "fps": spec.fps,
            "width": spec.width,
            "height": spec.height,
            "elevation_deg": spec.elevation_deg,
            "transparent": bool(spec.transparent),
        },
    }

    running_patch = {"status": "running", "error": None, "job_id": job_id,
                     "front_selection": front_selection}
    if pose_stage is not None:
        running_patch["pose_stage"] = pose_stage
    _set_state(running_patch)

    # ---- POST the job (shared client — k94) ----
    remote_id, err = _client.submit_job(requests, url, headers, payload)
    if err is not None:
        return _fail(err.code, err.message, retryable=err.retryable)

    def _delete_remote() -> None:
        _client.delete_remote(requests, url, headers, remote_id)

    # ---- poll until done / error / cancel / deadline (shared client — k94) ----
    # The live stage/progress/log_tail mirror into the media bus (and thence GET
    # /llm/jobs) rides inside poll_job; a user cancel records mesh state first.
    files, err = _client.poll_job(
        requests, url, headers, remote_id, job_id,
        label=f"identity mesh build for profile {slug!r}",
        is_cancelling=is_cancelling, set_progress=set_progress,
        progress_source="identity_render",
        on_cancel=lambda: _set_state({"status": "cancelled", "error": None}))
    if err is not None:
        if err.code == "cancelled":
            # mesh state was already set to "cancelled" by on_cancel — not an error state.
            return JobResult(job_id=job_id, ok=False, error=JobError(
                code="cancelled", message=err.message, retryable=False))
        return _fail(err.code, err.message, retryable=err.retryable)

    # ---- download every produced file + persist under the identity dir ----
    mesh_dir = os.path.join(identity_profiles._identity_dir(slug), "mesh", recon_id)
    turntable_dir = os.path.join(mesh_dir, "turntable")
    persisted = _client.persist_mesh_files(
        requests, url, headers, remote_id, files, mesh_dir, turntable_dir)
    glb_path = persisted.glb_path
    mesh_json_path = persisted.mesh_json_path
    video_path = persisted.video_path
    frame_paths: list[str] = list(persisted.frame_paths)

    if glb_path is None:
        return _fail("render_no_glb",
                     f"the render service reported done for profile {slug!r} but produced "
                     "no .glb mesh file",
                     retryable=True)

    # ---- attach the turntable frames as a reconstruction entry (promotable) ----
    # Reuse attach_reconstruction with replace=True so the promote route — which finds a
    # record by recon_id — sees THESE frames (and the list never grows a duplicate id).
    # The frames carry angular order via degrees_per_frame + their sorted order so the UI
    # can scrub. attach failure never fails the job; mesh state is the durable record.
    attached_rec = None
    if frame_paths:
        frame_paths.sort()  # frame_0000.png, frame_0001.png … == angular order
        n = len(frame_paths)
        try:
            attached_rec = identity_profiles.attach_reconstruction(
                slug, recon_id, frame_paths,
                spec={"job_id": job_id, "mode": "turntable", "frame_count": n,
                      "degrees_per_frame": (round(360.0 / n, 2) if n else None),
                      "source": "identity_render_relay", "glb_path": glb_path},
                replace=True,
            )
        except Exception:  # noqa: BLE001
            logger.warning("identity mesh: attach_reconstruction failed for %s/%s",
                           slug, recon_id, exc_info=True)

    # ---- optional AUTO-PROMOTE: seed canonical from the 45°-spaced turntable frames ----
    # The ONE-CLICK full-identity template (POST .../generate) sets spec.auto_promote so a
    # single action goes mesh -> turntable -> CANONICAL. Guardrails (all deliberate):
    #   * only when a turntable actually attached (a record with N views exists);
    #   * only when N >= 4 — a partial canonical would be a false promise, so skip entirely;
    #   * promote the frames NEAREST each SEMANTIC_VIEWS azimuth (0/45/90/135/180/225/270/
    #     315°) via ``identity_profiles.canonical_frame_indices`` — SELECTION over frames the
    #     orbit already rendered, indexing the attached reconstruction's own view list (what
    #     promote validates). Nothing is re-rendered.
    # WIDENED 4 -> 8 (operator 2026-07-16: "45 degree shots"). Selection is by DEGREES, not
    # by index arithmetic: the old ``[0, N//4, N//2, 3N//4]`` shortcut only told the truth on
    # a ring whose frame count divides evenly, whereas the azimuth walk reads
    # ``degrees_per_frame`` off the record and is correct on any ring (a coarse ring simply
    # yields fewer than 8 deduped views instead of duplicate bytes).
    # NOTE (S5): the operator also asked for 2 AERIAL views (overhead / -overhead). They are
    # NOT promoted here and must not be faked from this ring — the turntable is
    # single-elevation (every frame's elevation_deg is 0.0). They need the mesh-render path.
    # LATEST-WINS (operator RESCINDED the never-clobber rule, 2026-07-14 night: "it is
    # less important now that we have a real base project as a whole to go from"):
    # auto-promote REPLACES any existing canonical with the newest generation's cardinals.
    # This is provenance-safe because mesh reconstruction never reads canonical (the
    # feedback-loop fix); the manual Approve button still allows picking other angles,
    # and ``auto_promote: false`` on the request opts a run out entirely.
    # A promotion failure NEVER fails the job (the mesh + turntable already succeeded) — it
    # is recorded in mesh state as ``auto_promote_error`` and the job still returns ok.
    auto_promote_extra: dict = {}
    promoted_canonical: list = []  # the canonical this run seeded (empty if none) -> the minted version's DNA
    # The AZIMUTH of each promoted view (2026-07-16), positionally aligned with
    # promoted_canonical — carried onto the minted version so the ACTIVE version's DNA (what
    # the resolver actually serves) knows its own angles. Empty = no angle provenance.
    promoted_angles: list = []
    if getattr(spec, "auto_promote", False) and attached_rec is not None:
        try:
            rec_views = list(attached_rec.get("views") or [])
            nv = len(rec_views)
            if nv < 4:
                logger.info("identity mesh: auto-promote skipped for %s/%s — only %d "
                            "turntable frames (need >=4 for a canonical set)",
                            slug, recon_id, nv)
            else:
                # degrees_per_frame off the record; None -> canonical_frame_indices derives
                # 360/N itself (the same fallback bank_views uses for older records).
                chosen_idx = identity_profiles.canonical_frame_indices(
                    nv, attached_rec.get("degrees_per_frame"))
                logger.info("identity mesh: auto-promote selecting %d canonical views for "
                            "%s/%s from a %d-frame ring (indices=%s)",
                            len(chosen_idx), slug, recon_id, nv, chosen_idx)
                promoted = identity_profiles.promote_reconstruction_views(slug, recon_id, chosen_idx)
                auto_promote_extra["auto_promoted"] = True
                if isinstance(promoted, dict):
                    promoted_canonical = [
                        p for p in (promoted.get("canonical") or []) if isinstance(p, str)]
                    # promote_reconstruction_views persisted the angles alongside the paths;
                    # read them back rather than recomputing, so the version can never
                    # disagree with the profile about what angle a view is.
                    promoted_angles = list(promoted.get("canonical_angles") or [])
        except Exception as exc:  # noqa: BLE001 — a promote failure never fails the job
            logger.warning("identity mesh: auto-promote failed for %s/%s",
                           slug, recon_id, exc_info=True)
            auto_promote_extra["auto_promote_error"] = str(exc)

    # ---- VERSION MINT (IDENTITY-VERSIONS-SLICE.md slice 3) ----
    # A successful build lands as a NEW append-only VERSION and becomes ACTIVE (latest-wins
    # applies WITHIN a version's canonical; versions themselves never overwrite one another).
    # ``kind`` is "textured" when this run baked a texture, else "clay" — and the FIRST clay
    # version minted for an identity is pinned as the geometric BASE (identity_profiles.
    # _auto_version_name owns that rule). The version's ``canonical`` is the set auto-promote
    # just seeded (empty for a surgical run that did not auto-promote — the version still
    # records the recon + kind, and the resolver falls back to the profile canonical /
    # reference set). mint_version dedupes by recon_id, so a bus RETRY of this same build
    # updates the version in place rather than duplicating it. A mint failure NEVER fails the
    # job (the mesh + turntable already succeeded) — it is recorded as ``version_error``.
    version_extra: dict = {}
    try:
        kind = "textured" if bool(getattr(spec, "texture", False)) else "clay"
        # canonical_angles=None (not []) when this run promoted nothing, so mint_version
        # reads it as "no angle information" exactly like every pre-2026-07-16 caller.
        minted = identity_profiles.mint_version(
            slug, recon_id, kind, promoted_canonical,
            canonical_angles=promoted_angles or None)
        if isinstance(minted, dict):
            version_extra["version_id"] = minted.get("version_id")
            version_extra["version_name"] = minted.get("name")
            version_extra["version_kind"] = minted.get("kind")
    except Exception as exc:  # noqa: BLE001 — a version mint failure never fails the job
        logger.warning("identity mesh: version mint failed for %s/%s",
                       slug, recon_id, exc_info=True)
        version_extra["version_error"] = str(exc)

    # ---- record terminal mesh state (the GET mesh-status route reads this) ----
    # front_selection is re-included here (not just in the "running" patch above)
    # because attach_reconstruction(replace=True), just above, REPLACES the whole
    # reconstruction record when a turntable attached — including its "mesh" sub-dict —
    # so anything set only before that attach would otherwise vanish from the terminal
    # state a caller actually reads.
    terminal_patch = {
        "status": "done",
        "error": None,
        "glb_path": glb_path,
        "video_path": video_path,
        "mesh_json_path": mesh_json_path,
        "frame_count": len(frame_paths),
        "textured": bool(spec.texture),
        "job_id": job_id,
        "front_selection": front_selection,
        **auto_promote_extra,
        **version_extra,
    }
    # pose_stage re-included here (not just the "running" patch above) for the same reason
    # front_selection is: attach_reconstruction(replace=True) rewrites the reconstruction
    # record, so a field set only before that attach would vanish from the terminal state
    # a caller actually reads.
    if pose_stage is not None:
        terminal_patch["pose_stage"] = pose_stage
    _set_state(terminal_patch)

    _delete_remote()  # best-effort remote cleanup after a successful download

    # ---- JobResult outputs: the turntable mp4 (a video MediaRef). The GLB is NOT a
    # MediaRef kind (image/audio/video only), so it is surfaced via mesh state above. ----
    outputs: list = []
    if video_path:
        try:
            outputs.append(make_media_ref(
                asset_id=secrets.token_hex(8), kind="video", uri=video_path,
                mime="video/mp4"))
        except Exception:  # noqa: BLE001 — a MediaRef we can't build never fails the job
            logger.debug("identity mesh: could not build mp4 MediaRef", exc_info=True)
    return JobResult(job_id=job_id, ok=True, outputs=tuple(outputs))
