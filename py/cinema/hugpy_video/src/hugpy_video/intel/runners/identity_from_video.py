"""Identity FROM-VIDEO bus runner (k94) — ONE chained job, the clownworld MO.

Input: a video of a character. Output: ONE bindable identity profile per detected
character — reference images (the char360 view crops, front first), a canonical view set,
and a 3D GLB (+ turntable when the service produced one) attached through the SAME store
helpers the mesh relay uses, so the existing mesh-status / promote / bind routes just
work on the result.

It is a thin HTTP RELAY to the ``IDENTITY_RENDER_URL`` service (central has no GPU and
never runs char360 / Hunyuan3D), over the shared ``identity_render_client`` helpers — the
exact submit / poll / download / persist path ``identity_render_relay`` and
``identity_video_extract_relay`` use (factored, not copied).

The service job (identical payload to clownworld's ``characters3d.ts``):

    POST /jobs  {"kind": "video_characters_glb", "identity_id": "<slug>",
                 "video_path": "<abs path>", "mesh_params": {"texture": true,
                 "octree_resolution": 256}}

Done-job file layout (documented contract — clownworld reads exactly this):
    characters3d_result.json   {n_characters, n_meshed, characters:[{char, glb, n_views,
                                views_used:{front: <basename>}, error?}]}
    char360_result.json        the char360 manifest: characters:[{char, views:[{file:
                                "char_NN/view_MM_….png", yaw, bin, score}], face_centroid}]
    char_NN/identity.glb       one GLB per meshed character
    char_NN/<front basename>   the front view the mesh used (``char_NN.png`` in the task)
    char_NN/<view files>       the char360 view crops
  (+ optionally char_NN/*.mp4 and char_NN/frames/*.png when the service renders a
   turntable for the kind — persisted + attached when present, never required.)

Per detected character (manifest order), the runner:
  1. downloads the front + view crops into the jailed staging dir
     ``<IDENTITIES_HOME>/_char360_extracts/<job_id>/<char>/`` (servable via /video/media);
  2. creates (or REFRESHES — an existing active profile of that slug gets its reference
     set replaced) the profile: character 0 is ``slugify(name)``, character k is
     ``slugify(f"{name} {k+1}")`` -> ``<slug>-2``, ``<slug>-3`` …; reference images =
     the crops capped at ``MAX_SOURCE_IMAGES``, front FIRST (the mesh routes' default
     front = the first reference);
  3. attaches the full view set as a ``video_extract`` reconstruction (the angle bank),
     promotes up to 8 views nearest the semantic azimuths to ``canonical``;
  4. persists the GLB (+ mesh json / mp4 / frames) under ``<slug>/mesh/<recon_id>/`` via
     ``dest_for``; attaches turntable frames (when any) with ``replace=True``; mints a
     version (latest-wins ACTIVE) and records terminal mesh state so GET
     ``.../reconstruction/<recon_id>/mesh`` reports ``done`` + ``glb_path``.

Every expected failure — unconfigured/unreachable service, 401, a render error, a
timeout, no characters, nothing written back — is DATA (``JobResult(ok=False)``). A
per-character write-back problem never kills the other characters. The terminal
``JobResult.identities`` carries ``{name, n_characters, profiles:[{char, slug, name,
glb, n_views, canonical, error?}]}`` so the UI can jump to what was created.
"""
from __future__ import annotations

import json
import logging
import os

from hugpy_video.intel.result_schema import JobError, JobResult
from hugpy_video.intel.runners import identity_render_client as _client

logger = logging.getLogger(__name__)

_SUMMARY_NAME = "characters3d_result.json"
_MANIFEST_NAME = "char360_result.json"
_MESH_EXTS = (".glb", ".json", ".mp4")


def char_slug(name: str, index: int, slugify) -> tuple[str, str]:
    """``(display_name, slug)`` for the ``index``-th detected character of a request
    named ``name``: the first is the name itself, extras are ``"<name> 2"`` -> ``<slug>-2``.
    Pure so the route/tests can predict the slugs."""
    display = name if index == 0 else f"{name} {index + 1}"
    return display, slugify(display)


def _nearest_view_indices(yaws: list, semantic_views: dict, angular_distance) -> list[int]:
    """Indices of the views nearest each semantic azimuth (deduped, in semantic order),
    using each view's REAL yaw (char360 bins by measured yaw, unlike a uniform ring).
    Views with an unknown yaw are never picked. Empty when no yaw is known."""
    known = [(i, float(y)) for i, y in enumerate(yaws)
             if isinstance(y, (int, float)) and not isinstance(y, bool)]
    if not known:
        return []
    out: list[int] = []
    for target in semantic_views.values():
        best = min(known, key=lambda iy: angular_distance(iy[1] % 360.0, float(target)))
        if best[0] not in out:
            out.append(best[0])
    return out


def run_identity_from_video(spec, job_id: str) -> JobResult:
    """Relay ``spec`` as ONE ``video_characters_glb`` job, then turn every detected
    character into a bindable identity profile. See the module docstring."""
    # Lazy imports keep this module boot-cheap (runners/__init__ imports it at boot) and
    # keep char360/cv2/hy3dgen off the central side entirely — this runner only RELAYS.
    from hugpy_video.intel import identity_profiles
    from hugpy_video.intel.media_bus import is_cancelling, principal_of, set_progress, visibility_of

    # t172 / S4: the INITIATING principal + visibility, read off THIS relay job's own
    # bus row, so an auto-created identity profile is OWNED by the member who asked for
    # the build (not owner=None -> operator-owned, the flagged gap) and inherits their
    # public/private choice. A legacy/unattributed job -> (None, public), i.e. the prior
    # behavior. Best-effort: an attribution hiccup must never fail the build.
    try:
        _init_owner = principal_of(job_id)
        _init_private = visibility_of(job_id)[2]
    except Exception:  # noqa: BLE001
        _init_owner, _init_private = None, False

    def _fail(code: str, message: str, retryable: bool) -> JobResult:
        return JobResult(job_id=job_id, ok=False,
                         error=JobError(code=code, message=message, retryable=retryable))

    url, token = _client.service_config()
    if not url or not token:
        nc = _client.not_configured_error("video-to-identity builds (char360 + Hunyuan3D)")
        return _fail(nc.code, nc.message, retryable=nc.retryable)

    import requests  # lazy — present; keeps the module boot-cheap

    headers = _client.auth_headers(token)
    name = spec.name
    base_slug = identity_profiles.slugify(name)
    identity_id = (getattr(spec, "identity_id", "") or "").strip() or base_slug or f"fromvideo-{job_id}"
    label = f"identity build from video for {name!r}"

    payload = {
        "kind": "video_characters_glb",
        "identity_id": identity_id,
        # video_path (NOT video_b64): ae + central share the mount, so a clip that may be
        # hundreds of MB is never base64-inflated through the request body.
        "video_path": spec.source.uri,
        "mesh_params": dict(spec.mesh_params or {}),
    }

    remote_id, err = _client.submit_job(requests, url, headers, payload)
    if err is not None:
        return _fail(err.code, err.message, retryable=err.retryable)

    def _delete_remote() -> None:
        _client.delete_remote(requests, url, headers, remote_id)

    files, err = _client.poll_job(
        requests, url, headers, remote_id, job_id,
        label=label, is_cancelling=is_cancelling, set_progress=set_progress,
        progress_source="identity_from_video")
    if err is not None:
        return _fail(err.code, err.message, retryable=err.retryable)
    files = [f for f in (files or []) if isinstance(f, str) and f.strip()]

    # ---- manifests: the summary (per-character glb + front) and the char360 manifest
    #      (per-character view crops). Either may be absent on an older service; the
    #      file list itself is the last-resort source of truth (char_NN/identity.glb).
    def _json_file(fname: str):
        if fname not in files:
            return None
        raw = _client.download_file(requests, url, headers, remote_id, fname)
        if raw is None:
            return None
        try:
            doc = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return doc if isinstance(doc, dict) else None

    summary = _json_file(_SUMMARY_NAME) or {}
    manifest = _json_file(_MANIFEST_NAME) or {}

    summary_by_char: dict = {}
    order: list[str] = []
    for e in (summary.get("characters") or []):
        if isinstance(e, dict) and isinstance(e.get("char"), str) and e["char"].strip():
            summary_by_char[e["char"]] = e
            if e["char"] not in order:
                order.append(e["char"])
    manifest_by_char: dict = {}
    for e in (manifest.get("characters") or []):
        if isinstance(e, dict) and isinstance(e.get("char"), str) and e["char"].strip():
            manifest_by_char[e["char"]] = e
            if e["char"] not in order:
                order.append(e["char"])
    for f in files:  # last resort: a char dir with a GLB but no manifest entry
        parts = f.replace("\\", "/").split("/")
        if len(parts) >= 2 and parts[0].startswith("char_") and parts[0] not in order \
                and f.lower().endswith(".glb"):
            order.append(parts[0])

    if not order:
        _delete_remote()
        n = summary.get("n_characters", manifest.get("n_characters"))
        return _fail("no_characters",
                     f"the video-to-identity build found no characters in the source video "
                     f"(n_characters={n!r})",
                     retryable=False)

    stage_root = os.path.join(identity_profiles.IDENTITIES_HOME, "_char360_extracts", job_id)
    results: list[dict] = []
    wrote_any = False

    for idx, char_id in enumerate(order):
        if is_cancelling(job_id):  # cooperative cancel between characters
            _delete_remote()
            return _fail("cancelled", f"{label} cancelled by user", retryable=False)

        display, slug = char_slug(name, idx, identity_profiles.slugify)
        entry: dict = {"char": char_id, "slug": slug, "name": display, "glb": False,
                       "n_views": 0, "canonical": 0}
        results.append(entry)
        s_entry = summary_by_char.get(char_id) or {}
        m_entry = manifest_by_char.get(char_id) or {}
        char_files = [f for f in files if f.replace("\\", "/").startswith(f"{char_id}/")]

        # ---- 1. the view crops (+ the front the mesh used), into the staging jail ----
        view_records: list[dict] = []
        seen: set = set()

        def _stage(rel: str):
            safe = _client.safe_rel(rel)
            if not safe or safe in seen:
                return None
            data = _client.download_file(requests, url, headers, remote_id, rel)
            if data is None:
                return None
            dest = os.path.join(stage_root, safe)
            try:
                _client.atomic_write_bytes(dest, data)
            except OSError:
                logger.warning("identity from-video: could not persist %r -> %s", rel, dest)
                return None
            seen.add(safe)
            return dest

        front_rel = None
        vu = s_entry.get("views_used")
        if isinstance(vu, dict) and isinstance(vu.get("front"), str) and vu["front"].strip():
            front_rel = f"{char_id}/{os.path.basename(vu['front'])}"
        front_path = _stage(front_rel) if front_rel else None
        if front_path:
            view_records.append({"url": front_path, "yaw": 0.0, "front": True})

        for v in (m_entry.get("views") or []):
            if not isinstance(v, dict) or not isinstance(v.get("file"), str):
                continue
            dest = _stage(v["file"])
            if dest is None:
                continue
            view_records.append({"url": dest, "yaw": v.get("yaw"), "bin": v.get("bin"),
                                 "score": v.get("score")})
        view_paths = [r["url"] for r in view_records]
        entry["n_views"] = len(view_paths)

        # ---- 2. the GLB (+ mesh json / mp4 / turntable frames) for this character ----
        mesh_names = [
            f for f in char_files
            if f.lower().endswith(_MESH_EXTS) or "/frames/" in f.replace("\\", "/")]
        recon_id = f"fromvideo_{job_id}_{char_id}"
        mesh_dir = os.path.join(identity_profiles._identity_dir(slug), "mesh", recon_id)
        turntable_dir = os.path.join(mesh_dir, "turntable")
        persisted = _client.persist_mesh_files(
            requests, url, headers, remote_id, mesh_names, mesh_dir, turntable_dir,
            strip_prefix=f"{char_id}/")
        if persisted.glb_path is None and s_entry.get("error"):
            entry["error"] = str(s_entry.get("error"))

        if not view_paths and persisted.glb_path is None:
            entry["error"] = entry.get("error") or "no views and no mesh for this character"
            continue

        # ---- 3. create / refresh the profile ----
        refs = view_paths[:identity_profiles.MAX_SOURCE_IMAGES]
        try:
            existing = identity_profiles.get_profile(slug)
            if existing is None:
                if not refs:
                    entry["error"] = "character meshed but produced no view crops to seed a profile"
                    continue
                prof = identity_profiles.create_profile(
                    display, list(refs),
                    notes=f"Created from video ({char_id}, {len(view_paths)} views) — job {job_id}.",
                    owner=_init_owner, private=_init_private)
            else:
                prof = identity_profiles.update_profile(slug, source_images=list(refs)) if refs else existing
            if not isinstance(prof, dict) or not prof.get("slug"):
                entry["error"] = "profile store returned no slug"
                continue
        except identity_profiles.ProfileError as exc:
            entry["error"] = f"{exc}"
            continue
        except Exception as exc:  # noqa: BLE001 — never let one character kill the job
            logger.warning("identity from-video: profile write raised for %s", char_id, exc_info=True)
            entry["error"] = f"{type(exc).__name__}: {exc}"
            continue
        wrote_any = True

        # ---- 3b. the view set as a video_extract reconstruction + canonical promote ----
        attached = None
        if view_paths:
            try:
                attached = identity_profiles.attach_reconstruction(
                    slug, recon_id, list(view_paths),
                    spec={"source": "identity_from_video", "mode": "video_extract",
                          "frame_count": len(view_paths),
                          "degrees_per_frame": round(360.0 / len(view_paths), 2),
                          "job_id": job_id, "char": char_id,
                          "face_centroid": m_entry.get("face_centroid")},
                    replace=True)
            except Exception:  # noqa: BLE001
                logger.warning("identity from-video: attach views failed for %s/%s",
                               slug, recon_id, exc_info=True)
        promoted_canonical: list = []
        if attached is not None:
            try:
                idxs = _nearest_view_indices(
                    [r.get("yaw") for r in view_records], identity_profiles.SEMANTIC_VIEWS,
                    identity_profiles._angular_distance)
                if not idxs:
                    idxs = list(range(min(len(view_paths), 8)))
                idxs = idxs[:identity_profiles.MAX_SOURCE_IMAGES]
                promoted = identity_profiles.promote_reconstruction_views(slug, recon_id, idxs)
                if isinstance(promoted, dict):
                    promoted_canonical = [p for p in (promoted.get("canonical") or [])
                                          if isinstance(p, str)]
                    entry["canonical"] = len(promoted_canonical)
            except Exception:  # noqa: BLE001 — a promote failure never fails the character
                logger.warning("identity from-video: canonical promote failed for %s", slug,
                               exc_info=True)

        # ---- 4. mesh state + (optional) turntable + version ----
        if persisted.glb_path is None:
            entry["error"] = entry.get("error") or "the service produced no GLB for this character"
            continue
        entry["glb"] = True
        if persisted.frame_paths:
            try:
                n = len(persisted.frame_paths)
                identity_profiles.attach_reconstruction(
                    slug, recon_id, list(persisted.frame_paths),
                    spec={"job_id": job_id, "mode": "turntable", "frame_count": n,
                          "degrees_per_frame": round(360.0 / n, 2),
                          "source": "identity_from_video", "glb_path": persisted.glb_path},
                    replace=True)
            except Exception:  # noqa: BLE001
                logger.warning("identity from-video: attach turntable failed for %s/%s",
                               slug, recon_id, exc_info=True)
        textured = bool((spec.mesh_params or {}).get("texture", True))
        version_extra: dict = {}
        try:
            minted = identity_profiles.mint_version(
                slug, recon_id, "textured" if textured else "clay", promoted_canonical)
            if isinstance(minted, dict):
                version_extra["version_id"] = minted.get("version_id")
                version_extra["version_name"] = minted.get("name")
        except Exception as exc:  # noqa: BLE001
            logger.warning("identity from-video: version mint failed for %s", slug, exc_info=True)
            version_extra["version_error"] = str(exc)
        try:
            identity_profiles.set_mesh_state(slug, recon_id, {
                "status": "done", "error": None,
                "glb_path": persisted.glb_path,
                "video_path": persisted.video_path,
                "mesh_json_path": persisted.mesh_json_path,
                "frame_count": len(persisted.frame_paths),
                "textured": textured,
                "job_id": job_id,
                "source": "identity_from_video",
                "front_selection": {"mode": "char360", "chosen": front_path,
                                    "checked": 0, "full_body": None},
                **version_extra,
            })
        except Exception:  # noqa: BLE001 — mesh state is a best-effort mirror
            logger.warning("identity from-video: set_mesh_state failed for %s/%s",
                           slug, recon_id, exc_info=True)

    _delete_remote()  # best-effort remote cleanup — everything we need is local now

    if not wrote_any:
        detail = "; ".join(f"{e['char']}: {e.get('error')}" for e in results if e.get("error"))
        return _fail("write_back_failed",
                     f"{label} wrote back no characters" + (f" ({detail})" if detail else ""),
                     retryable=False)

    logger.info("identity from-video done for %r: %s", name,
                [(e["slug"], e["glb"], e["n_views"]) for e in results])
    return JobResult(job_id=job_id, ok=True,
                     identities={"name": name, "n_characters": len(order), "profiles": results})
