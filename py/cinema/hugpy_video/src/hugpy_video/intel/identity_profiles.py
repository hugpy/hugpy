"""Identity-profile store — the DURABLE form of "the reference set IS the identity".

An identity profile is a NAMED library item: ``{name, source_images (1..4),
created_at, notes?}``. The operator's vision (STUDIO-ROADMAP.md, "IDENTITY
PROFILES") is that a character's DNA — its curated reference set — is created
ONCE and associated anywhere (single clips, movies, stills) instead of being
re-supplied per request. This module is stage (a): the library item itself.
Stage (b) — turnaround generation from a profile + the re-edit loop that promotes
an approved rendering to the canonical reference — comes next and layers on top
of this store; nothing here forecloses it (a future ``canonical`` ref set is just
another key on the entry).

Storage is a first-class directory of its own — the Identities feature OWNS its
reference images instead of merely pointing at ephemeral ``uploads/`` paths that
the session-scoped upload reaper (upload_routes._wipe_session) is free to erase::

    <IDENTITIES_HOME>/
      identity_profiles.json          # the registry (source of truth), {profiles,_deleted}
      <slug>/
        profile.json                  # human-readable denormalized MIRROR of the entry
        ref_00.<ext>  ref_01.<ext> …  # the identity's reference images, COPIED IN
        _superseded/<ts>/…            # refs replaced by an update (never erased)
      _deleted/
        <slug>@<ts>/                  # archived identity dirs (moved here on delete)

PERSISTENCE INVARIANT (operator 2026-07-13 — "identities are persistent, their
reference images must never be reaped"): IDENTITIES_HOME is a SIBLING of
UPLOADS_HOME (both under DEFAULT_ROOT), NEVER a child of it. The upload reaper is
jailed to UPLOADS_HOME (upload_routes._within_uploads), so it structurally cannot
reach an identity's copied images. This module also NEVER registers a copied ref
with any upload-session tracking (it never calls _touch_session, never writes
under UPLOADS_HOME/.sessions/): an identity's pixels live ONLY under
``IDENTITIES_HOME/<slug>/`` and answer to nothing but this store.

On create the validated source paths (which the route has already jail-resolved +
image-classified) are COPIED into the identity's dir; the entry's
``source_images`` then point at the identity-owned copies (still under
DEFAULT_ROOT, so the media_store jail still accepts them). The registry lives with
the other durable registries and survives restarts; the copies survive the reaper.

Single-writer discipline is the api_keys._save idiom REUSED VERBATIM: a
process-wide lock plus a unique-per-write temp file (pid + token) renamed onto
the target with os.replace as the sole atomicity point — so two concurrent
writers never race between open() and replace() (that exact race bit a batch of
/v1 auths on 2026-07-11). The same temp-name+os.replace atomicity guards every
copied ref and every profile.json write. Never-delete doctrine: a delete ARCHIVES
the entry under ``_deleted`` AND MOVES the identity's dir under ``_deleted/`` —
bytes are never erased, only relocated; an update MOVES superseded refs under
``_superseded/`` rather than overwriting them.

Backward-compat: the FIRST load after this change, when the new registry is absent
but the legacy ``PROJECTS_HOME/identity_profiles.json`` exists, migrates each
active profile into its own dir (COPYING referenced images, never moving), leaving
the legacy registry and the original uploads UNTOUCHED (reversible). A missing
source never crashes and never drops the identity — the entry is kept, the source
recorded in ``missing_references``. Guarded by new-registry-absence, so it runs at
most once and is idempotent.
"""
from __future__ import annotations

import os
import re
import json
import secrets
import shutil
import threading
import time
import unicodedata
from typing import Any, Optional

from hugpy_platform.constants import IDENTITIES_HOME, PROJECTS_HOME, UPLOADS_HOME

# json is imported lazily inside _load/_save to keep the module import cheap and
# to mirror how the route modules defer json/sqlite until first use.

# video_intel/identity_profiles.py (Append to existing file)

from hugpy_video.intel.identity_reconstruction_schema import (
    IdentitySingleViewRegenSpec,
    IdentityMeshSpec,
)

_LOCK = threading.Lock()

MAX_SOURCE_IMAGES = 12

# CANONICAL CAP — the number of views the STORE may hold as an identity's approved DNA.
# Widened 4 -> 8 (operator, 2026-07-16: "45 degree shots") so the canonical set is the full
# 45°-spaced azimuth ring (``SEMANTIC_VIEWS``) rather than the 4 cardinals.
#
# ⚠ THIS IS *NOT* THE RENDER CAP. Two different numbers were conflated at 4 until now:
#   * STORAGE  (this) — how many approved views an identity may HOLD. Now 8.
#   * RENDER  (``identity_reconstruction_schema._MAX_CANONICAL_IMAGES``, ``studio.job.
#     _MAX_REFERENCE_IMAGES``, both still 4) — how many refs ONE id_lock/VACE render may
#     CONSUME. That 4 is a MODEL constraint (diffusers 0.39 prepends each reference as a
#     VACE reference latent; more = more conditioning latents + VRAM), NOT a bookkeeping
#     choice, so it deliberately does NOT move with this one.
# Consumers that feed the render channel must therefore SELECT a <=4 subset of canonical
# (see ``render_refs_from_canonical``); they must never pass the whole set through.
MAX_CANONICAL_IMAGES = 8

# The render-channel cap, mirrored here so the store can enforce the subset rule in one
# place. Kept in lockstep with ``identity_reconstruction_schema._MAX_CANONICAL_IMAGES``.
MAX_RENDER_REFS = 4

# --------------------------------------------------------------------------- #
# VERSIONS slice (operator-directed 2026-07-14; IDENTITY-VERSIONS-SLICE.md).
#
# An identity holds ONE base (the clay mesh minted on the first clay generate,
# name "base") + N append-only VERSIONS — named render-sets minted by the mesh
# RELAY on each successful generate. Never-delete lives here too: a version is
# ARCHIVED (flagged, dropped from the wire list), its pixels never erased.
#
# ``gen_settings`` are the per-identity generation defaults the left-column
# Settings tab persists and the Generate click prefills. ALWAYS present on the
# wire (defaulted) so the UI can rely on the shape. Additive keys only — nothing
# below removes or renames an existing profile field.
# --------------------------------------------------------------------------- #
VERSION_KINDS = ("clay", "textured", "styled")

# The canonical happy-path generation settings (defaults-are-promises: a bare
# Generate click == these). Merged over any stored partial in ``_public`` so the
# wire shape is complete even for a profile that has never had settings PATCHed.
_DEFAULT_GEN_SETTINGS: dict[str, Any] = {
    "texture": True,
    "pose": "none",          # "none" | "t-pose"
    "frame_count": 72,
    "fps": 24,
    "width": 768,
    "height": 768,
    "auto_promote": True,
    "front_ref": None,       # abs path (one of the profile's own refs) | null
    "remove_background": True,
    # The VL model the identity 3D-imaging pipeline's FRONT-SELECT step uses to pick
    # the full-body reference before meshing (relay: _select_front_view -> POST
    # /ml/vision). None (default) == the fleet-default VL model (the 3B) — the relay
    # sends NO model field, byte-identical to before this setting existed (zero
    # regression; defaults-are-promises). A non-null value is a model key the fleet
    # advertises as image-text-to-text capable (e.g. a 7B), validated LIVE in
    # set_gen_settings so it can never point at a non-existent / non-VL model.
    "vision_model": None,    # image-text-to-text model key | null (== fleet default)
    # CLEANUP-PROMPT slice (operator-requested 2026-07-15): remembered per-identity
    # render-steer channels so the operator does NOT re-type the cleanup instruction on
    # every generate (profile-IS-identity doctrine). Both default "" -> a bare Generate
    # click renders byte-identically to today (defaults-are-promises).
    #   cleanup_prompt   a positive-worded AVOID instruction (e.g. "no object on her back,
    #                    clean bare back") woven into the T-pose front render prompt.
    #   negative_prompt  a TRUE negative forwarded to the studio Wan-VACE render.
    "cleanup_prompt": "",    # positive-avoid instruction woven into the front render | ""
    "negative_prompt": "",   # true negative forwarded to the studio render | ""
}
_POSE_CHOICES = ("none", "t-pose")

# --------------------------------------------------------------------------- #
# CONSENT / AUTHORIZATION slice (k97, oracle authority gate).
#
# An identity profile is a real person's DNA often enough that reproducing it is
# a rights question, not a rendering question (oracle architecture invariant 7 +
# §11: "Real-person likeness and voice replication require typed authorization").
# So a profile may carry an ``authorization`` block::
#
#     "authorization": {"likeness": {"granted": true,
#                                    "evidence": "release-2026-08-14.pdf",
#                                    "granted_at": "2026-08-14T10:00:00+00:00"}}
#
# ABSENT MEANS UNAUTHORIZED, and nothing migrates: every profile on disk today
# has no block and is therefore not authorized — which is the honest reading, not
# a regression. Grandfathering existing profiles into "granted" would fabricate
# consent nobody gave. ``granted`` is only ever true alongside EVIDENCE a human
# can go read; the store refuses to record a bare "yes".
# --------------------------------------------------------------------------- #
AUTHORIZATION_KINDS = ("likeness", "voice")


def _authorization_block(entry: dict[str, Any]) -> dict[str, Any]:
    """The stored consent block, normalized to the wire shape. Unknown kinds and
    malformed rows are DROPPED rather than echoed: a half-parsed consent record
    must never read as a grant."""
    raw = entry.get("authorization")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for kind in AUTHORIZATION_KINDS:
        row = raw.get(kind)
        if not isinstance(row, dict):
            continue
        out[kind] = {
            "granted": bool(row.get("granted")),
            "evidence": str(row.get("evidence") or ""),
            "granted_at": row.get("granted_at"),
        }
    return out


def _authority_kind(kind: Any) -> str:
    """Accept either a plain string or an ``oracle.contracts.AuthorityKind``
    (a str-Enum) without importing the oracle package — video_intel must not
    take a dependency on the layer above it."""
    return str(getattr(kind, "value", kind) or "").strip().lower()


class ProfileError(ValueError):
    """Bad-input on the store contract (empty name, no/too-many refs, dup slug).

    A ValueError subclass so the route's existing ``except (ValueError, TypeError)``
    idiom catches it as a clean 4xx; ``code`` lets the route pick the precise
    status (409 for a duplicate, 400 for the rest) — errors-as-data, never a raw
    500 crossing the HTTP boundary."""

    def __init__(self, message: str, code: str = "invalid_profile") -> None:
        super().__init__(message)
        self.code = code


def _store_path() -> str:
    # Call-time resolution (not import-time) mirrors api_keys._store_path: it reads
    # the module global IDENTITIES_HOME so a test that rebinds it to a temp dir lands
    # the whole store there, never in the real identities tree. (Env isolation does
    # not work — constants' get_env_value reads the .env file — so the direct module
    # rebind is the honest lever, exactly as the store test already does.)
    return os.path.join(IDENTITIES_HOME, "identity_profiles.json")


def _legacy_store_path() -> str:
    # The pre-migration location: a single buried JSON under PROJECTS_HOME whose
    # entries merely pointed at ephemeral uploads/ paths. Read-only from here on —
    # the migration COPIES out of it and never mutates or deletes it (reversibility).
    return os.path.join(PROJECTS_HOME, "identity_profiles.json")


def _identity_dir(slug: str) -> str:
    # The per-identity folder that OWNS this profile's reference images. A sibling
    # of the registry under IDENTITIES_HOME, itself a sibling of UPLOADS_HOME — see
    # the module docstring's PERSISTENCE INVARIANT: the upload reaper cannot reach here.
    return os.path.join(IDENTITIES_HOME, slug)


def _ext_for(src: str) -> str:
    """The lowercased extension to give a copied ref (source ext preserved). A
    source with no extension gets ``.img`` so the copy always has a stable name."""
    ext = os.path.splitext(src)[1].lower()
    return ext if ext else ".img"


def _atomic_copy(src: str, dest: str) -> None:
    """Copy *src* -> *dest* with the store's atomicity idiom: copy to a unique temp
    name IN the destination dir, then os.replace onto the final name (the sole
    atomicity point). Raises on a missing/unreadable source (callers tolerate it)."""
    tmp = f"{dest}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dest)


def _materialize_refs(dir_path: str, sources: list[str]) -> tuple[list[str], list[str]]:
    """COPY each source in order into ``dir_path/ref_NN.<ext>`` and return
    ``(source_images, missing_references)``.

    Order is preserved and the ref number is the list POSITION, so a rendered set
    stays aligned even when a middle source is absent. A missing/unreadable source
    NEVER crashes: nothing is copied for it, its ORIGINAL path is retained in the
    returned reference set, and it is recorded in ``missing_references`` (so a UI
    can flag a broken ref and a future re-copy has the original handle). The
    never-delete doctrine holds — this only ever writes new ``ref_NN`` files."""
    os.makedirs(dir_path, exist_ok=True)
    refs: list[str] = []
    missing: list[str] = []
    for i, src in enumerate(sources):
        dest = os.path.join(dir_path, f"ref_{i:02d}{_ext_for(src)}")
        if os.path.isfile(src):
            try:
                _atomic_copy(src, dest)
                refs.append(dest)
                continue
            except OSError:
                pass  # fall through to the missing-source path
        refs.append(src)      # keep the original handle for a broken/absent source
        missing.append(src)
    return refs, missing


def _write_profile_json(slug: str, entry: dict[str, Any]) -> None:
    """(Re)write the human-readable denormalized MIRROR of an entry at
    ``<slug>/profile.json``. The registry is the source of truth; this is a
    convenience view regenerated on every create/update. Same atomic temp+replace."""
    dir_path = _identity_dir(slug)
    os.makedirs(dir_path, exist_ok=True)
    doc = {
        "slug": slug,
        "name": entry.get("name", ""),
        "created_at": entry.get("created_at"),
        "notes": entry.get("notes", ""),
        "owner": entry.get("owner"),            # t171 ownership (mirror)
        "private": bool(entry.get("private")),  # t171 visibility (mirror)
        "source_images": list(entry.get("source_images") or []),
        "missing_references": list(entry.get("missing_references") or []),
        # stage (b) additive keys: the generated turnaround renderings awaiting
        # approval, and any views the operator has promoted to the canonical set.
        "reconstructions": list(entry.get("reconstructions") or []),
        "canonical": list(entry.get("canonical") or []),
        # k97 consent block — mirrored so the rights record is readable on disk
        # next to the pixels it governs. Empty ({}) means nothing is authorized.
        "authorization": _authorization_block(entry),
    }
    path = os.path.join(dir_path, "profile.json")
    tmp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _supersede_existing_refs(dir_path: str) -> None:
    """MOVE the identity's current ``ref_NN.*`` files aside into
    ``<dir>/_superseded/<ts>/`` before a new set is copied in — never-delete: a
    replaced reference set is relocated, never overwritten or erased. No-op when the
    dir is absent or holds no ref files. (Only the top-level ``ref_*`` files move;
    prior ``_superseded`` archives and ``profile.json`` stay put.)"""
    if not os.path.isdir(dir_path):
        return
    existing = [
        n for n in os.listdir(dir_path)
        if n.startswith("ref_") and not n.endswith(".tmp")
        and os.path.isfile(os.path.join(dir_path, n))
    ]
    if not existing:
        return
    dest_dir = os.path.join(dir_path, "_superseded", f"{time.time()}")
    os.makedirs(dest_dir, exist_ok=True)
    for n in existing:
        shutil.move(os.path.join(dir_path, n), os.path.join(dest_dir, n))


def _materialize_refs_staged(dir_path: str, sources: list[str]) -> tuple[list[str], list[str]]:
    """Two-stage COPY-commit variant of ``_materialize_refs`` for UPDATES, where an
    incoming path may BE one of the identity's OWN current ``ref_NN`` files (the UI reads
    the owned copies back and re-submits them when editing the set). The old update flow
    superseded the current refs FIRST and then copied from ``sources`` — so a
    self-referential source had already been MOVED under ``_superseded/`` and was recorded
    as MISSING, desyncing the registry (it named ``ref_01..ref_11`` while disk held
    ``ref_00..ref_10``, and every path failed ``os.path.isfile`` downstream).

    Here every readable source is COPIED to a ``_stage_NN`` name (extension preserved)
    BEFORE anything is superseded — so a self-reference is captured while it still exists;
    only THEN is the current set relocated (never-delete, via ``_supersede_existing_refs``)
    and the staged files committed to their final position-numbered ``ref_NN`` names. A
    missing/unreadable source behaves exactly as in ``_materialize_refs`` (its ORIGINAL
    path is retained in the returned set and recorded in ``missing``)."""
    os.makedirs(dir_path, exist_ok=True)
    # STAGE 1 — copy each readable source to a staging name (distinct from ref_*), so
    # NOTHING is superseded until every source is safely captured off its origin.
    staged: dict[int, tuple[str, str]] = {}   # position -> (stage_path, ext)
    originals: dict[int, str] = {}            # position -> original path (missing source)
    missing: list[str] = []
    for i, src in enumerate(sources):
        ext = _ext_for(src)
        if os.path.isfile(src):
            stage = os.path.join(dir_path, f"_stage_{i:02d}{ext}")
            try:
                _atomic_copy(src, stage)
                staged[i] = (stage, ext)
                continue
            except OSError:
                pass
        originals[i] = src
        missing.append(src)
    # STAGE 2 — relocate the current ref_* set now that every new source is staged.
    _supersede_existing_refs(dir_path)
    # STAGE 3 — commit staged files to their final ref_NN names (position preserved).
    refs: list[str] = []
    for i in range(len(sources)):
        if i in staged:
            stage, ext = staged[i]
            dest = os.path.join(dir_path, f"ref_{i:02d}{ext}")
            os.replace(stage, dest)
            refs.append(dest)
        else:
            refs.append(originals[i])
    return refs, missing


# --------------------------------------------------------------------------- #
# stage (b) — reconstruction (turnaround) + canonical helpers. Reuse the store's
# atomic-copy / supersede / atomic-json idioms VERBATIM so the generated stills
# and the promoted canonical set answer to nothing but this store, exactly like a
# profile's reference images do.
# --------------------------------------------------------------------------- #
def _reconstruction_root(slug: str) -> str:
    """``<slug>/reconstruction`` — the folder holding every recon bundle for an
    identity (one ``<recon_id>/`` subdir per generated turnaround set)."""
    return os.path.join(_identity_dir(slug), "reconstruction")


def _reconstruction_dir(slug: str, recon_id: str) -> str:
    """``<slug>/reconstruction/<recon_id>`` — one generated turnaround set's bundle
    (its ``views/`` stills + ``manifest.json``)."""
    return os.path.join(_reconstruction_root(slug), recon_id)


def _canonical_dir(slug: str) -> str:
    """``<slug>/canonical`` — the promoted canonical reference set (``ref_NN.<ext>``),
    a SIBLING key of the identity's own ``ref_NN`` uploads. The module docstring
    (stage (b)) anticipates this: "a future ``canonical`` ref set is just another key
    on the entry"."""
    return os.path.join(_identity_dir(slug), "canonical")


def _atomic_write_json(path: str, doc: dict[str, Any]) -> None:
    """Write *doc* as pretty JSON to *path* with the store's atomicity idiom (unique
    temp name in the dest dir + os.replace). Mirrors ``_write_profile_json``'s write."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _supersede_existing_dir(target_dir: str) -> None:
    """MOVE an existing *target_dir* aside under ``<parent>/_superseded/<ts>/<base>``
    before a fresh set is written in its place — never-delete: a replaced recon
    bundle or canonical set is relocated, never overwritten or erased. No-op when the
    dir is absent. Mirrors ``_supersede_existing_refs`` (which moves individual ref
    files) but at whole-directory granularity."""
    if not os.path.isdir(target_dir):
        return
    parent = os.path.dirname(target_dir)
    base = os.path.basename(target_dir)
    dest = os.path.join(parent, "_superseded", f"{time.time()}", base)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(target_dir, dest)


def _materialize_views(dir_path: str, sources: list[str]) -> tuple[list[str], list[str]]:
    """COPY each produced still in order into ``dir_path/view_NN.png`` and return
    ``(views, missing)``. Order is preserved and the view number is the list POSITION.
    A missing/unreadable source NEVER crashes (mirrors ``_materialize_refs``): nothing
    is copied for it, its ORIGINAL path is retained in the returned list, and it is
    recorded in ``missing``. Stills come out of the frame-extract job as PNG, so every
    owned copy is normalized to ``.png``."""
    os.makedirs(dir_path, exist_ok=True)
    views: list[str] = []
    missing: list[str] = []
    for i, src in enumerate(sources):
        dest = os.path.join(dir_path, f"view_{i:02d}.png")
        if isinstance(src, str) and os.path.isfile(src):
            try:
                _atomic_copy(src, dest)
                views.append(dest)
                continue
            except OSError:
                pass  # fall through to the missing-source path
        views.append(src)     # keep the original handle for a broken/absent source
        missing.append(src)
    return views, missing


def _archive_identity_dir(slug: str, deleted_at: float) -> None:
    """MOVE ``<slug>/`` under ``_deleted/<slug>@<deleted_at>/`` (correlating with the
    registry archive key) so a delete relocates the identity's owned pixels rather
    than erasing them. Best-effort and tolerant: a missing dir or an FS error must
    never fail the delete (the registry archive is the durable record of record)."""
    src = _identity_dir(slug)
    if not os.path.isdir(src):
        return
    try:
        graveyard = os.path.join(IDENTITIES_HOME, "_deleted")
        os.makedirs(graveyard, exist_ok=True)
        shutil.move(src, os.path.join(graveyard, f"{slug}@{deleted_at}"))
    except OSError:
        pass  # never erase, and never let a relocation hiccup fail the archive


def _empty() -> dict[str, Any]:
    return {"profiles": {}, "_deleted": {}}


def _migrate_legacy() -> Optional[dict[str, Any]]:
    """One-time, idempotent, reversible migration from the legacy single-file store.

    Runs ONLY when the new registry is absent AND the legacy
    ``PROJECTS_HOME/identity_profiles.json`` exists (the caller guards on
    new-registry-absence, so this fires at most once). For each ACTIVE profile it
    creates ``<slug>/`` and COPIES (never moves) each referenced image into
    ``ref_NN.<ext>``, rewriting source_images to the copies and recording any
    missing sources in ``missing_references`` — a missing source neither crashes
    nor drops the identity (an all-missing entry is kept with an empty-copied set,
    its original paths retained, and the full missing list). ``_deleted`` entries
    carry over as-is (archived dirs are not materialized). The legacy registry and
    every original uploads/ file are left UNTOUCHED (reversibility). Returns the new
    registry (already written atomically), or None when there is nothing to migrate."""
    if os.path.exists(_store_path()):
        return None  # already migrated — never run twice
    legacy_path = _legacy_store_path()
    if not os.path.exists(legacy_path):
        return None
    try:
        with open(legacy_path, "r", encoding="utf-8") as f:
            legacy = json.load(f)
    except (OSError, ValueError):
        return None  # a corrupt legacy file is not worth crashing the feature over
    if not isinstance(legacy, dict):
        return None

    profiles = legacy.get("profiles") if isinstance(legacy.get("profiles"), dict) else {}
    deleted = legacy.get("_deleted") if isinstance(legacy.get("_deleted"), dict) else {}
    new_data: dict[str, Any] = {"profiles": {}, "_deleted": dict(deleted)}
    for slug, entry in profiles.items():
        if not isinstance(entry, dict):
            continue
        new_entry = dict(entry)
        sources = [r for r in (entry.get("source_images") or []) if isinstance(r, str)]
        refs, missing = _materialize_refs(_identity_dir(slug), sources)
        new_entry["source_images"] = refs
        if missing:
            new_entry["missing_references"] = missing
        else:
            new_entry.pop("missing_references", None)
        new_data["profiles"][slug] = new_entry
        _write_profile_json(slug, new_entry)
    _save(new_data)
    return new_data


def _backfill_versions(data: dict[str, Any]) -> bool:
    """One-time, idempotent, in-place VERSIONS backfill for identities that predate the
    versions slice (IDENTITY-VERSIONS-SLICE.md, build order 1).

    For each ACTIVE profile that carries NO ``versions`` key yet, seed the append-only
    version list from what the profile already holds — the design's migration verbatim:
    "existing identities get their newest recon as base/clay + current canonical as
    version-01":

      * base (clay): minted from the NEWEST reconstruction — its ``recon_id`` anchors the
        geometric ground truth. Flagged ``base: True`` (never archivable, like a first
        clay minted live). Created ONLY when the profile has >=1 reconstruction.
      * version-01: the profile's CURRENT promoted ``canonical`` set becomes a named
        render-set (kind ``"textured"`` — the promoted cardinals are rendered DNA).
        Created ONLY when ``canonical`` is non-empty, and made ACTIVE so the version-aware
        resolver returns EXACTLY today's DNA (no behavioral regression on first load).
      * ``active_version`` points at version-01 when present, else the base, else None.

    A never-generated profile (no reconstruction AND no canonical) is still marked
    migrated — it gets ``versions: []`` + ``active_version: None`` so this never re-runs
    for it. Idempotency is guarded by ``"versions" not in entry``: once an entry carries
    the key (even an empty list) it is NEVER re-seeded. ``_deleted`` archives are left
    untouched. Mutates ``data`` in place and returns True iff any entry changed (the
    caller then persists via the single _save below)."""
    changed = False
    for entry in (data.get("profiles") or {}).values():
        if not isinstance(entry, dict) or "versions" in entry:
            continue
        recons = [r for r in (entry.get("reconstructions") or []) if isinstance(r, dict)]
        canonical = [p for p in (entry.get("canonical") or []) if isinstance(p, str)]
        # reconstructions append newest-LAST (attach_reconstruction appends), so the last
        # element is the newest recon — the one whose mesh IS the identity's current clay.
        newest_recon = recons[-1] if recons else None
        versions: list[dict[str, Any]] = []
        base_id: Optional[str] = None
        v01_id: Optional[str] = None
        if newest_recon is not None:
            base_id = "ver_" + secrets.token_hex(8)
            versions.append({
                "version_id": base_id,
                "name": "base",
                "kind": "clay",
                "recon_id": newest_recon.get("recon_id"),
                "created_at": (newest_recon.get("created_at")
                               or entry.get("created_at") or time.time()),
                "canonical": [],  # clay = geometry; textured DNA rides on version-01
                "notes": "",
                "base": True,  # the geometric ground truth — never archivable
            })
        if canonical:
            v01_id = "ver_" + secrets.token_hex(8)
            versions.append({
                "version_id": v01_id,
                "name": "version-01",
                "kind": "textured",
                "recon_id": (newest_recon.get("recon_id")
                             if newest_recon is not None else None),
                "created_at": entry.get("created_at") or time.time(),
                "canonical": list(canonical),
                "notes": "",
            })
        entry["versions"] = versions
        # ACTIVE preserves today's resolver output: version-01 (== current canonical) when
        # present, else the clay base, else nothing (a never-generated profile).
        entry["active_version"] = v01_id or base_id
        changed = True
    if changed:
        _save(data)
    return changed


def _load() -> dict[str, Any]:
    path = _store_path()
    if not os.path.exists(path):
        migrated = _migrate_legacy()
        if migrated is not None:
            _backfill_versions(migrated)  # seed versions on the freshly-migrated registry
            return migrated
        return _empty()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        # A corrupt/half-written file must not wedge the feature — the honest
        # answer is an empty store (the next _save re-writes it cleanly).
        return _empty()
    if not isinstance(data, dict):
        return _empty()
    data.setdefault("profiles", {})
    data.setdefault("_deleted", {})
    _backfill_versions(data)  # seed versions[] for any identity that predates the slice
    return data


def _save(data: dict[str, Any]) -> None:
    path = _store_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Temp name UNIQUE PER WRITE (pid + token): several gunicorn processes may
    # write here, and two writers sharing one "<path>.tmp" race between open()
    # and os.replace() — the loser's replace() dies FileNotFoundError. pid+token
    # keeps every write atomic AND collision-free; os.replace is the atomicity
    # point. (Lifted from api_keys._save, which learned this the hard way.)
    tmp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def slugify(name: str) -> str:
    """A stable, filesystem-safe, url-safe slug from a display name. NFKD-fold to
    ascii, lowercase, non-alphanumerics -> single hyphens, trim. The slug is the
    profile's identity in the store + the DELETE route path segment."""
    if not isinstance(name, str):
        return ""
    norm = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", norm).strip("-").lower()
    return slug


def _public(slug: str, entry: dict[str, Any]) -> dict[str, Any]:
    """The wire shape for one profile — the slug folded in alongside the stored
    fields so a list row is self-describing (the caller keys deletes on it)."""
    out = {
        "slug": slug,
        "name": entry.get("name", ""),
        # WIRE key is ``reference_images`` (what the UI schema + video_routes'
        # identity resolver + tests read), sourced from the internal
        # ``source_images`` storage field. Keep these two names in sync: emitting
        # ``source_images`` on the wire silently blanks the UI's reference set and
        # makes the movie/i2v identity resolver see an empty identity.
        "reference_images": list(entry.get("source_images") or []),
        "created_at": entry.get("created_at"),
        "notes": entry.get("notes", ""),
        # t171 — the owner principal + public/private flag, ALWAYS on the wire so
        # the route's ownership/visibility gate + the UI can read them. A legacy
        # owner-less profile emits owner=None (operator-owned) + private=False
        # (PUBLIC): visible to all, mutable only by the operator/admin tier.
        "owner": entry.get("owner"),
        "private": bool(entry.get("private")),
    }
    # Additive: only surfaced when some source could not be copied in, so a broken
    # reference is honestly visible (a UI can show a badge). Absent on the happy path.
    missing = entry.get("missing_references")
    if missing:
        out["missing_references"] = list(missing)
    # Stage (b) additive keys — ALWAYS present (default empty) so a UI can rely on the
    # shape: ``reconstructions`` is the ordered list of generated turnaround sets
    # awaiting approval; ``canonical`` is the promoted reference set (empty until the
    # operator promotes recon views). Neither removes/renames an existing key.
    out["reconstructions"] = [
        _recon_manifest(r) for r in (entry.get("reconstructions") or [])
        if isinstance(r, dict)
    ]
    out["canonical"] = list(entry.get("canonical") or [])
    # ANGLE PROVENANCE (2026-07-16) — ``canonical_angles``: the azimuth in degrees of each
    # ``canonical`` entry, POSITIONALLY aligned (``canonical_angles[i]`` describes
    # ``canonical[i]``); an element may be ``null`` when that view's angle is unknown.
    # Shape choice (deliberate): ``canonical`` STAYS a flat list of path strings and this
    # rides alongside as a PARALLEL list. Changing canonical's element type to objects would
    # break every existing consumer at once (the id_lock resolver, the Generate bar, Studio,
    # the fixed 7-key version wire contract) for zero gain.
    # BACK-COMPAT: OMITTED entirely for the profiles on disk today, which were promoted
    # before angles were recorded and have no stored angle. Absent means "angles unknown" —
    # a consumer must degrade to unlabeled ordinals, exactly as it behaves today. It is
    # never faked: guessing 0.0 would read as "front" and be a lie.
    canonical_angles = entry.get("canonical_angles")
    if canonical_angles and len(canonical_angles) == len(out["canonical"]):
        out["canonical_angles"] = list(canonical_angles)
    canonical_missing = entry.get("canonical_missing")
    if canonical_missing:
        out["canonical_missing"] = list(canonical_missing)
    # VERSIONS slice additive keys — ALWAYS present (defaulted) so the UI can rely
    # on the shape. ``versions`` omits archived entries (never-delete: they stay in
    # storage, just off the wire); ``active_version`` is the id_lock DNA pointer;
    # ``gen_settings`` is the full happy-path defaults merged over any stored partial.
    out["versions"] = [
        _public_version(v) for v in (entry.get("versions") or [])
        if isinstance(v, dict) and not v.get("archived")
    ]
    out["active_version"] = entry.get("active_version")
    out["gen_settings"] = _merged_gen_settings(entry.get("gen_settings"))
    # CONSENT slice (k97) — ALWAYS present so a reader can rely on the shape; an
    # EMPTY dict is the honest default and means "nothing is authorized". Never
    # defaulted to granted, never inferred from the profile existing.
    out["authorization"] = _authorization_block(entry)
    return out


def _public_version(v: dict[str, Any]) -> dict[str, Any]:
    """The FIXED wire shape for one version — the seven contract keys (a sibling UI is
    built to these), plus ``canonical_angles`` when — and only when — this version carries
    real angle provenance. The internal-only ``base``/``archived`` flags stay OFF the wire;
    the base version is identifiable by ``name == "base"`` + ``kind == "clay"``, and
    archived versions are omitted from the list entirely.

    ``canonical_angles`` (2026-07-16) is ADDITIVE and OPTIONAL: the per-view azimuth in
    degrees, positionally aligned with ``canonical`` (an element may be ``null``). It is
    omitted for every version minted before angles were recorded — i.e. everything on disk
    today keeps EXACTLY the seven-key shape, byte-for-byte, so the sibling UI built to those
    keys is unaffected. A reader must treat "absent" as "angles unknown" and fall back to
    unlabeled ordinals; it must never assume 0.0."""
    out = {
        "version_id": v.get("version_id"),
        "name": v.get("name", ""),
        "kind": v.get("kind", "clay"),
        "recon_id": v.get("recon_id"),
        "created_at": v.get("created_at"),
        "canonical": list(v.get("canonical") or []),
        "notes": v.get("notes", ""),
    }
    angles = v.get("canonical_angles")
    # Length guard: if the two ever drift apart, drop the angles rather than mislabel a view.
    if angles and len(angles) == len(out["canonical"]):
        out["canonical_angles"] = list(angles)
    return out


def _merged_gen_settings(stored: Any) -> dict[str, Any]:
    """The full happy-path defaults with any stored partial layered on top — only
    KNOWN keys survive (a stale/unknown stored key is dropped from the wire), so the
    shape is always exactly the contract's keys (all of ``_DEFAULT_GEN_SETTINGS``)."""
    gs = dict(_DEFAULT_GEN_SETTINGS)
    if isinstance(stored, dict):
        for k in _DEFAULT_GEN_SETTINGS:
            if k in stored:
                gs[k] = stored[k]
    return gs


def list_profiles() -> list[dict[str, Any]]:
    """Active (non-archived) profiles, newest first."""
    with _LOCK:
        data = _load()
    out = [_public(slug, entry) for slug, entry in data["profiles"].items()]
    out.sort(key=lambda p: p.get("created_at") or 0, reverse=True)
    return out


def get_profile(slug: str) -> Optional[dict[str, Any]]:
    """One active profile by slug, or None if unknown/archived."""
    if not slug or not isinstance(slug, str):
        return None
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
    if entry is None:
        return None
    return _public(slug, entry)


def profile_authorized(slug: str, kind: str = "likeness") -> bool:
    """Is this identity authorized for ``kind`` ("likeness" | "voice")?

    True ONLY for an active profile carrying an explicit ``granted`` row WITH
    evidence. Every other answer — unknown slug, archived profile, absent block,
    granted-without-evidence, unknown kind — is False. This is the accessor the
    oracle authority gate consults; it is deliberately incapable of returning
    True by omission."""
    k = _authority_kind(kind)
    if k not in AUTHORIZATION_KINDS:
        return False
    profile = get_profile(slug)
    if profile is None:
        return False
    row = (profile.get("authorization") or {}).get(k)
    if not isinstance(row, dict):
        return False
    return bool(row.get("granted")) and bool(str(row.get("evidence") or "").strip())


def set_profile_authorization(
    slug: str,
    kind: str,
    *,
    granted: bool,
    evidence: str = "",
    granted_at: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Record (or revoke) consent for one kind on an ACTIVE profile, in place.

    Granting REQUIRES evidence — a pointer to the release/contract/ticket a human
    can go read (§11: consent is pointed at, never inferred). Revoking
    (``granted=False``) keeps the evidence string as history: the row flips to
    not-granted rather than vanishing, so the record of what was once claimed
    survives (never-delete doctrine). Returns the updated public shape, or None
    when the slug names no active profile."""
    k = _authority_kind(kind)
    if k not in AUTHORIZATION_KINDS:
        raise ProfileError(
            f"authorization kind must be one of {list(AUTHORIZATION_KINDS)}, "
            f"got {kind!r}", code="invalid_profile")
    evidence = (evidence or "").strip()
    if granted and not evidence:
        raise ProfileError(
            f"granting {k} authorization requires evidence (a release/contract/"
            f"ticket reference) — consent is never inferred",
            code="invalid_profile")
    if not slug or not isinstance(slug, str):
        return None

    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        entry = dict(entry)  # edit a copy — nothing is written until _save
        block = dict(entry.get("authorization") or {})
        prior = block.get(k) if isinstance(block.get(k), dict) else {}
        block[k] = {
            "granted": bool(granted),
            "evidence": evidence or str(prior.get("evidence") or ""),
            "granted_at": granted_at or (
                time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())),
        }
        entry["authorization"] = block
        _write_profile_json(slug, entry)  # regenerate the mirror
        data["profiles"][slug] = entry
        _save(data)
    return _public(slug, entry)


# --------------------------------------------------------------------------- #
# OWNERSHIP + VISIBILITY slice (t171 — 2026-09-14 security review, finding S4).
#
# Until now a profile carried NO owner, so any principal past the /video gate
# could list / read / delete / patch ANY account's profile. Two additive fields
# fix that WITHOUT hiding or freezing anything already on disk:
#
#   owner    the creating principal, stamped by the route from the SAME identity
#            the movie owner-gate uses (_request_principal): "operator" |
#            "user:<name>" | "share:<id>" | None. None == unattributed, read by
#            the route as operator-owned.
#   private  DEFAULT False == PUBLIC (visible to everyone the /video gate lets in);
#            True == visible only to the owner + the operator/admin tier.
#
# The STORE only persists + echoes these (create_profile, set_profile_private,
# _public, the profile.json mirror); it does NOT authorize — read/write scoping
# lives in the route layer next to the request context. MIGRATION is a no-op by
# construction: a legacy entry has neither key, so it reads as owner=None +
# private=False -> PUBLIC and operator-owned. Nothing is rewritten, hidden, or
# made immutable to current studio use. ``profile_authorized`` (the likeness/voice
# CONSENT gate) is a different question entirely and is left untouched.
# --------------------------------------------------------------------------- #
def set_profile_private(slug: str, private: bool) -> Optional[dict[str, Any]]:
    """Flip an ACTIVE profile's ``private`` visibility flag in place (the owner-only
    PUBLIC<->PRIVATE toggle; the ROUTE enforces owner/operator authorization before
    calling here). Mirrors set_profile_authorization's edit-a-copy + regenerate-
    mirror + single-writer _save idiom. Returns the updated public shape, or None
    when the slug names no active profile (unknown or archived)."""
    if not slug or not isinstance(slug, str):
        return None
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        entry = dict(entry)  # edit a copy — nothing is written until _save
        entry["private"] = bool(private)
        _write_profile_json(slug, entry)  # regenerate the on-disk mirror
        data["profiles"][slug] = entry
        _save(data)
    return _public(slug, entry)


def create_profile(
    name: str,
    source_images: list[str],
    notes: str = "",
    owner: Optional[str] = None,
    private: bool = False,
) -> dict[str, Any]:
    """Create a profile from a display name + a set of ALREADY-VALIDATED reference
    image paths (the route jail-resolves + ffprobe-classifies them as images
    first, exactly like the movie route; this store trusts what it is handed and
    persists it). Slug is derived from the name; a collision with an existing
    ACTIVE slug raises ProfileError(code="duplicate") -> the route's 409. The
    validated sources are COPIED into the identity's own dir (``<slug>/ref_NN.<ext>``)
    and the entry then points at those identity-owned copies, 1..4 in order (order
    is meaningful — an id_lock hash keys on it). A source gone missing does NOT crash
    the create: it is skipped, its original path retained, recorded in
    ``missing_references``. This is the escape from the upload reaper — the pixels
    now live under IDENTITIES_HOME, not the ephemeral uploads/ tree.

    OWNERSHIP (t171): ``owner`` is the creating principal (stamped by the route
    from the same identity the movie owner-gate uses) and ``private`` (default
    False == PUBLIC) is the visibility flag. Both are persisted verbatim; the
    store does not itself authorize — the route layer scopes reads/writes on them.
    ``owner=None`` (the default, and every legacy owner-less profile) reads as
    operator-owned; ``private`` absent/False reads as PUBLIC, so nothing on disk
    today is hidden or made immutable by this change (non-destructive)."""
    display = (name or "").strip()
    if not display:
        raise ProfileError("name is required", code="invalid_profile")
    if not isinstance(source_images, list) or not source_images:
        raise ProfileError("at least one reference_image is required", code="invalid_profile")
    # Change MAX_source_images to MAX_SOURCE_IMAGES here:
    if len(source_images) > MAX_SOURCE_IMAGES:
        raise ProfileError(
            f"at most {MAX_SOURCE_IMAGES} source_images are accepted",
            code="invalid_profile",
        )
    refs: list[str] = []
    for raw in source_images:
        if not isinstance(raw, str) or not raw.strip():
            raise ProfileError("each reference_image must be a non-empty path", code="invalid_profile")
        refs.append(raw)
    slug = slugify(display)
    if not slug:
        raise ProfileError("name has no url-safe characters", code="invalid_profile")

    entry: dict[str, Any] = {
        "name": display,
        "source_images": refs,  # replaced by the identity-owned copies below
        "created_at": time.time(),
        "notes": (notes or "").strip(),
        # t171 per-account ownership + public/private visibility (security review
        # S4). ``owner`` is the creator's principal (the route passes
        # _request_principal(): "operator" | "user:<name>" | "share:<id>" | None);
        # ``private`` defaults False == PUBLIC. Persisted here; the ROUTE enforces.
        "owner": owner,
        "private": bool(private),
    }
    with _LOCK:
        data = _load()
        if slug in data["profiles"]:
            raise ProfileError(f"a profile named {display!r} already exists", code="duplicate")
        # COPY the validated sources into the identity's own dir; the entry now
        # points at the identity-owned copies (under IDENTITIES_HOME, still within
        # DEFAULT_ROOT -> media_store jail OK) and so survives the upload reaper.
        owned, missing = _materialize_refs(_identity_dir(slug), refs)
        entry["source_images"] = owned
        if missing:
            entry["missing_references"] = missing
        _write_profile_json(slug, entry)
        data["profiles"][slug] = entry
        _save(data)
    return _public(slug, entry)


def update_profile(
    slug: str,
    *,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    source_images: Optional[list[str]] = None,
) -> Optional[dict[str, Any]]:
    """Edit an EXISTING active profile IN PLACE. A true partial update: each of
    ``name``/``notes``/``source_images`` left at its default (``None``) is a
    no-op for that field — the route only forwards the keys actually present in
    the PATCH body, so this never needs a sentinel to distinguish "omitted" from
    "cleared".

    SLUG STABILITY (why a rename never re-slugs): the slug is this profile's
    identity everywhere OUTSIDE this store — it is the ``<slug>`` segment in the
    route path, and every ``identity_profile:<slug>`` reference a saved template,
    movie spec, or enqueue body carries. Re-deriving the slug from a new display
    name on rename would silently strand every one of those references (a
    template built against "mira" would 404 the moment someone renamed the
    display name to "Mira Prime"). So ``name`` is DISPLAY-ONLY here: the dict key
    in ``data["profiles"]`` never changes, only the stored ``name`` string does.

    ``source_images``, when given, must be a non-empty 1..MAX_source_images
    list (the route has already jail-resolved + image-classified it exactly like
    POST create) — an identity is never left pointing at zero references. Pass it
    as ``None`` (the default) to leave the current reference set untouched. When a
    new set IS given it is COPIED into the identity's dir renumbered from ref_00,
    and the superseded ref files are MOVED under ``<slug>/_superseded/<ts>/`` — the
    prior pixels are never erased, only relocated (never-delete doctrine). The
    ``profile.json`` mirror is regenerated on every update.

    Returns the updated public shape, or ``None`` if the slug names no ACTIVE
    profile (unknown or archived) — the route turns that into the same 404
    get/delete already give an unknown slug. Archives nothing: this mutates the
    entry in place, so ``created_at`` and the archive history are untouched."""
    if not slug or not isinstance(slug, str):
        return None

    new_name: Optional[str] = None
    if name is not None:
        new_name = name.strip()
        if not new_name:
            raise ProfileError("name is required", code="invalid_profile")

    new_refs: Optional[list[str]] = None
    if source_images is not None:
        if not isinstance(source_images, list) or not source_images:
            raise ProfileError("at least one reference_image is required", code="invalid_profile")
        # Change MAX_source_images to MAX_SOURCE_IMAGES here:
        if len(source_images) > MAX_SOURCE_IMAGES:
            raise ProfileError(
                f"at most {MAX_SOURCE_IMAGES} source_images are accepted",
                code="invalid_profile",
            )
        new_refs = []
        for raw in source_images:
            if not isinstance(raw, str) or not raw.strip():
                raise ProfileError("each reference_image must be a non-empty path", code="invalid_profile")
            new_refs.append(raw)

    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        entry = dict(entry)  # edit a copy — nothing is written until _save
        if new_name is not None:
            entry["name"] = new_name
        if notes is not None:
            entry["notes"] = notes.strip()
        if new_refs is not None:
            # Two-stage COPY-commit: stage the new set to _stage_NN BEFORE superseding the
            # old ref_* files, so re-submitting the identity's OWN owned paths (the UI reads
            # them back on edit) can't desync — a self-referential source is copied while it
            # still exists. _materialize_refs_staged supersedes then commits internally.
            # (Was: supersede-then-copy, which moved the source out from under the copy and
            # recorded every ref as missing → registry named ref_01..ref_11, disk had ref_00..)
            owned, missing = _materialize_refs_staged(_identity_dir(slug), new_refs)
            entry["source_images"] = owned
            if missing:
                entry["missing_references"] = missing
            else:
                entry.pop("missing_references", None)
        _write_profile_json(slug, entry)  # regenerate the mirror on every update
        data["profiles"][slug] = entry
        _save(data)
    return _public(slug, entry)


def delete_profile(slug: str) -> Optional[dict[str, Any]]:
    """ARCHIVE (never erase) a profile. The registry entry moves under ``_deleted``
    keyed ``<slug>@<deleted_at>`` with a ``deleted_at`` stamp AND the identity's own
    dir is MOVED under ``_deleted/<slug>@<deleted_at>/`` — so a name can be reused,
    the history is preserved, and the reference pixels are relocated rather than
    erased (never-delete doctrine). Returns the archived public shape, or None if the
    slug was not an active profile (idempotent — deleting an unknown/already-archived
    slug is a clean no-op)."""
    if not slug or not isinstance(slug, str):
        return None
    with _LOCK:
        data = _load()
        entry = data["profiles"].pop(slug, None)
        if entry is None:
            return None
        deleted_at = time.time()
        entry = dict(entry)
        entry["deleted_at"] = deleted_at
        data["_deleted"][f"{slug}@{deleted_at}"] = entry
        _save(data)
        # Relocate (never rm) the identity's owned bytes under _deleted/, correlated
        # with the registry archive key by the same deleted_at stamp. Best-effort.
        _archive_identity_dir(slug, deleted_at)
    return _public(slug, entry)


# --------------------------------------------------------------------------- #
# STAGE (b) — identity RECONSTRUCTION (turnaround generation) + canonical promote.
#
# The module docstring's stage (b): "turnaround generation from a profile + the
# re-edit loop that promotes an approved rendering to the canonical reference".
# These store functions own the DURABLE side of that loop; the actual rendering
# (the Wan VACE id_lock render + frame-extract, behind the runner's swap seam)
# hands the produced stills to ``attach_reconstruction`` on completion.
#
#   attach_reconstruction     — persist a generated turnaround set (its stills copied
#                               in, a manifest written, the entry's ``reconstructions``
#                               list appended) for approval.
#   promote_reconstruction_views — copy chosen recon views into the ``canonical`` ref
#                               set on the entry (the approved reference DNA).
#   list_reconstructions / get_reconstruction — UI read helpers.
#
# All follow the store's never-delete + atomic idioms: an existing recon bundle or
# canonical set is SUPERSEDED (moved aside), never overwritten; every copy + json
# write is atomic; the registry mutation is single-writer under ``_LOCK``.
# --------------------------------------------------------------------------- #
def _recon_manifest(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize ONE stored reconstruction record into the wire/manifest shape the UI
    reads. Additive + backward-compat only: an OLD record (attached before turntables)
    carries no ``mode`` -> defaulted to ``"sheet"`` so the shape is always self-describing.
    A turntable record's ``frame_count``/``degrees_per_frame`` (and the ordered ``views``
    holding the orbit frames in angular order) ride through untouched. Never drops or
    renames a key — the sheet path's shape is unchanged.

    The recognized-mode tuple is widened to include ``"angle-ring"`` (a pre-existing latent
    bug — it was silently downgraded to "sheet" on read) and ``"video_extract"`` (char360
    video-derived sets, CHAR360-FEATURE-PLAN S3) so those modes ROUND-TRIP on read instead
    of being flattened to "sheet". Kept in lock-step with attach_reconstruction's write-side
    gate below."""
    out = dict(record)
    if out.get("mode") not in ("sheet", "turntable", "angle-ring", "video_extract"):
        out["mode"] = "sheet"
    return out


def attach_reconstruction(
    slug: str,
    recon_id: str,
    view_paths: list[str],
    *,
    spec: Optional[dict[str, Any]] = None,
    replace: bool = False,
) -> Optional[dict[str, Any]]:
    """Persist a generated turnaround set for *slug* under ``recon_id``.

    Creates ``<slug>/reconstruction/<recon_id>/views/`` and ``_atomic_copy``s each
    produced still (order preserved) into ``view_NN.png``, writes ``manifest.json``
    (recon_id, created_at, ordered views, plus the render provenance carried in
    *spec*: job_id / prompt / seed / per-view prompts / view names), and APPENDS the
    record to the entry's ``reconstructions`` list under ``_LOCK`` (regenerating
    ``profile.json``). Never overwrites an existing recon dir — it is SUPERSEDED
    (moved aside) first, so a re-run under a colliding id never erases prior pixels.

    A missing/unreadable produced still does NOT crash: it is skipped, its original
    path retained in the record, and recorded under ``missing_views`` (mirrors
    ``_materialize_refs``). Returns the stored reconstruction record, or ``None`` if
    the slug names no ACTIVE profile (the caller — a bus runner — surfaces that as a
    clean error-as-data; the profile may have been archived mid-render)."""
    if not slug or not isinstance(slug, str):
        return None
    if not recon_id or not isinstance(recon_id, str) or not recon_id.strip():
        raise ProfileError("recon_id is required", code="invalid_profile")
    if not isinstance(view_paths, list) or not view_paths:
        raise ProfileError("view_paths must be a non-empty list of still paths",
                           code="invalid_profile")

    recon_dir = _reconstruction_dir(slug, recon_id)
    views_dir = os.path.join(recon_dir, "views")
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        # Never overwrite an existing recon bundle — relocate it (never-delete).
        _supersede_existing_dir(recon_dir)
        owned, missing = _materialize_views(views_dir, view_paths)
        record: dict[str, Any] = {
            "recon_id": recon_id,
            "created_at": time.time(),
            "views": owned,
        }
        if missing:
            record["missing_views"] = missing
        meta = spec if isinstance(spec, dict) else {}
        for key in ("job_id", "prompt", "seed", "prompts", "view_names",
                    # turntable additive provenance (sheet records omit these):
                    "frame_count", "degrees_per_frame", "orbit_prompt",
                    # char360 video-extract additive provenance (CHAR360-FEATURE-PLAN S3):
                    # ``char`` = the detected character id; ``face_centroid`` = that
                    # character's normed face embedding (or None) — S3b's face-descriptor
                    # match path reads it off the record. Kept additively so no existing
                    # key is renamed or dropped.
                    "char", "face_centroid"):
            if key in meta:
                record[key] = meta[key]
        # MODE always stored so the manifest is self-describing; absent meta => "sheet"
        # (the existing N-independent-view-stills path). For a turntable, ``views`` holds
        # the orbit clip's frames in ANGULAR order and ``frame_count`` its length. The gate
        # is widened to keep ``"angle-ring"`` (a pre-existing latent bug — it was wrongly
        # downgraded to "sheet") and ``"video_extract"`` (char360, CHAR360-FEATURE-PLAN S3)
        # rather than SILENTLY DOWNGRADING them. Kept in lock-step with _recon_manifest's
        # read-side normalizer above. (RECON_MODES + the reconstruction route validators are
        # deliberately NOT touched — this relay job never builds an IdentityReconstructionSpec.)
        mode = meta.get("mode")
        record["mode"] = (
            mode if mode in ("sheet", "turntable", "angle-ring", "video_extract") else "sheet")
        # manifest.json: the on-disk denormalized mirror of the record (same atomic write).
        _atomic_write_json(os.path.join(recon_dir, "manifest.json"), record)

        entry = dict(entry)  # edit a copy — nothing is written until _save
        recons = list(entry.get("reconstructions") or [])
        # replace=True (used by the mesh RELAY, which re-attaches under an EXISTING
        # recon_id): overwrite the first record sharing this recon_id IN PLACE so the
        # promote route — which finds a record by recon_id — sees THIS set, and the
        # list never grows a second same-id record. Default (append) is unchanged for
        # the reconstruction runner, which always mints a fresh recon_id.
        replaced = False
        if replace:
            for i, r in enumerate(recons):
                if isinstance(r, dict) and r.get("recon_id") == recon_id:
                    recons[i] = record
                    replaced = True
                    break
        if not replaced:
            recons.append(record)
        entry["reconstructions"] = recons
        _write_profile_json(slug, entry)  # regenerate the mirror
        data["profiles"][slug] = entry
        _save(data)
    return dict(record)


def canonical_angles_for_indices(record: dict[str, Any], idxs: list[int]) -> list[Optional[float]]:
    """The azimuth of each promoted ring index — the ANGLE PROVENANCE that ``canonical``
    historically threw away (2026-07-16).

    Before this, ``promote_reconstruction_views`` copied the frames at ``chosen_indices``
    and stored ONLY the destination paths, so ``ref_02`` meant "the third view someone
    promoted", never "180°". With 8 views that loss is worse than with 4: the operator's
    intent is to pick the canonical view whose ANGLE matches the shot, which is impossible
    if the angle isn't on the wire.

    Azimuth is computed with the SAME convention as ``bank_views``
    (``index * degrees_per_frame % 360``, ``degrees_per_frame`` off the record, else
    ``360/len(views)``) so the bank and canonical can never disagree — degrees are the
    canonical truth (see the ANGLE BANK section docstring).

    Returns one entry per index, POSITIONALLY aligned with the promoted paths. An entry is
    ``None`` when the angle is genuinely unknowable — a NON-turntable record (a ``sheet``
    recon has no orbit, so its views carry no azimuth) or a bad index. ``None`` is honest:
    it means "no angle", never a guessed 0.0 (which would read as "front" and be a lie)."""
    if not isinstance(record, dict) or record.get("mode") != "turntable":
        # sheet / video_extract / clay records carry no orbit -> no azimuth to report.
        return [None] * len(idxs)
    views = [v for v in (record.get("views") or []) if isinstance(v, str)]
    if not views:
        return [None] * len(idxs)
    dpf = record.get("degrees_per_frame")
    try:
        dpf = float(dpf)
        if dpf <= 0:
            raise ValueError
    except (TypeError, ValueError):
        dpf = 360.0 / len(views)
    out: list[Optional[float]] = []
    for i in idxs:
        out.append(round((i * dpf) % 360.0, 4) if 0 <= i < len(views) else None)
    return out


def promote_reconstruction_views(
    slug: str,
    recon_id: str,
    chosen_indices: list[int],
) -> Optional[dict[str, Any]]:
    """Promote chosen views of reconstruction ``recon_id`` into the entry's
    ``canonical`` reference set — the approved character DNA.

    Copies the views at ``chosen_indices`` (into ``<slug>/canonical/ref_NN.<ext>`` via
    ``_materialize_refs``, so a promoted set is renumbered from ``ref_00`` exactly like
    an uploaded set) and points the entry's ``canonical`` key at those identity-owned
    copies. Any existing canonical set is SUPERSEDED first (never-delete). At most
    ``MAX_source_images`` may be promoted (canonical feeds the id_lock reference
    channel, capped at 4). ``profile.json`` is regenerated.

    Errors-as-data on the store contract (``ProfileError`` -> the route's 4xx): a bad
    index shape, an out-of-range index, an unknown ``recon_id``, or too many views.
    Returns the updated public profile shape (its ``canonical`` now populated), or
    ``None`` if the slug names no ACTIVE profile (the route's 404)."""
    if not slug or not isinstance(slug, str):
        return None
    if not recon_id or not isinstance(recon_id, str) or not recon_id.strip():
        raise ProfileError("recon_id is required", code="invalid_profile")
    if not isinstance(chosen_indices, list) or not chosen_indices:
        raise ProfileError("views must be a non-empty list of view indices",
                           code="invalid_profile")
    idxs: list[int] = []
    for raw in chosen_indices:
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            raise ProfileError("each view index must be a non-negative int",
                               code="invalid_profile")
        idxs.append(raw)
    if len(idxs) > MAX_CANONICAL_IMAGES:
        raise ProfileError(
            f"at most {MAX_CANONICAL_IMAGES} views may be promoted to canonical",
            code="invalid_profile")

    canon_dir = _canonical_dir(slug)
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        record = next(
            (r for r in (entry.get("reconstructions") or [])
             if isinstance(r, dict) and r.get("recon_id") == recon_id),
            None,
        )
        if record is None:
            raise ProfileError(f"reconstruction {recon_id!r} not found",
                               code="invalid_profile")
        views = list(record.get("views") or [])
        chosen_sources: list[str] = []
        for i in idxs:
            if i >= len(views):
                raise ProfileError(
                    f"view index {i} is out of range (this reconstruction has "
                    f"{len(views)} views)", code="invalid_profile")
            chosen_sources.append(views[i])
        # ANGLE PROVENANCE (2026-07-16): capture each promoted view's azimuth BEFORE the
        # copy, while we still know which ring index it came from — the information this
        # function used to destroy. Computed off the SAME record, so it cannot drift from
        # the bank. ``_materialize_refs`` keeps its output POSITIONALLY aligned with its
        # input (a missing source keeps its original handle rather than being dropped), so
        # angles[i] describes owned[i] for every i.
        angles = canonical_angles_for_indices(record, idxs)
        # Never overwrite an existing canonical set — relocate it (never-delete).
        _supersede_existing_dir(canon_dir)
        owned, missing = _materialize_refs(canon_dir, chosen_sources)
        entry = dict(entry)  # edit a copy — nothing is written until _save
        entry["canonical"] = owned
        # PARALLEL key, not a change to ``canonical``'s element type: ``canonical`` stays a
        # flat list of path strings forever (every consumer + the fixed 7-key version wire
        # contract depends on that). Written ONLY when at least one real angle is known, so
        # a sheet-recon promote adds no misleading all-null key.
        if len(angles) == len(owned) and any(a is not None for a in angles):
            entry["canonical_angles"] = angles
        else:
            entry.pop("canonical_angles", None)  # stale angles must never outlive their set
        if missing:
            entry["canonical_missing"] = missing
        else:
            entry.pop("canonical_missing", None)
        _write_profile_json(slug, entry)  # regenerate the mirror
        data["profiles"][slug] = entry
        _save(data)
    return _public(slug, entry)


def list_reconstructions(slug: str) -> Optional[list[dict[str, Any]]]:
    """The ordered list of generated turnaround sets for *slug* (newest last, as
    attached), or ``None`` if the slug names no active profile."""
    if not slug or not isinstance(slug, str):
        return None
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
    if entry is None:
        return None
    return [_recon_manifest(r) for r in (entry.get("reconstructions") or [])
            if isinstance(r, dict)]


def get_reconstruction(slug: str, recon_id: str) -> Optional[dict[str, Any]]:
    """One reconstruction record by id, or ``None`` if the slug/recon_id is unknown."""
    recons = list_reconstructions(slug)
    if recons is None:
        return None
    return next((r for r in recons if r.get("recon_id") == recon_id), None)


def update_reconstruction_view_status(slug: str, recon_id: str, view_id: str, status: str) -> dict:
    """Updates the explicit approval status of a single angle tile."""
    if status not in ("approved", "rejected"):
        raise ProfileError("validation", "Status must be 'approved' or 'rejected'")
        
    profile = get_profile(slug)
    if not profile:
        raise ProfileError("not_found", f"Profile {slug} not found")

    recons = profile.get("reconstructions", [])
    target_recon = next((r for r in recons if r.get("recon_id") == recon_id), None)
    if not target_recon:
        raise ProfileError("not_found", f"Reconstruction {recon_id} not found")

    views = target_recon.get("views", [])
    target_view = next((v for v in views if isinstance(v, dict) and v.get("viewId") == view_id), None)
    if not target_view:
        raise ProfileError("not_found", f"View {view_id} not found")

    target_view["status"] = status
    
    # Save the updated profile
    _save_profile(slug, profile)
    return profile

def make_single_view_regeneration_spec(slug: str, recon_id: str, view_id: str, prompt: str, seed: int, use_neighbors: bool) -> IdentitySingleViewRegenSpec:
    """Prepares the regeneration spec, extracting nearest approved neighbors for conditioning."""
    profile = get_profile(slug)
    if not profile:
        raise ProfileError("not_found", f"Profile {slug} not found")
        
    # (In a full implementation, you would scan target_recon["views"] to find 
    # the closest azimuthDeg neighbors with status == "approved" and extract their imageUris)
    neighbor_uris = [] 
    
    return IdentitySingleViewRegenSpec(
        slug=slug,
        recon_id=recon_id,
        view_id=view_id,
        prompt=prompt,
        seed=seed,
        use_neighbors=use_neighbors,
        neighbor_images=tuple(neighbor_uris)
    )

def make_mesh_reconstruction_spec(slug: str, recon_id: str, view_ids: list, backend: str, workflow: str, output_format: str) -> IdentityMeshSpec:
    """Locks the active views and queues the mesh build job."""
    profile = get_profile(slug)
    if not profile:
        raise ProfileError("not_found", f"Profile {slug} not found")
        
    recons = profile.get("reconstructions", [])
    target_recon = next((r for r in recons if r.get("recon_id") == recon_id), None)
    if not target_recon:
        raise ProfileError("not_found", f"Reconstruction {recon_id} not found")

    # Initialize the mesh block to "queued" so the UI disables the build button
    if "mesh" not in target_recon:
        target_recon["mesh"] = {}
    target_recon["mesh"]["status"] = "queued"
    target_recon["mesh"]["error"] = None
    
    _save_profile(slug, profile)
    
    return IdentityMeshSpec(
        slug=slug,
        recon_id=recon_id,
        view_ids=tuple(view_ids),
        backend=backend,
        workflow=workflow,
        output_format=output_format
    )

def get_mesh_state(slug: str, recon_id: str) -> dict:
    """Read-only fetch of the mesh build status."""
    profile = get_profile(slug)
    if not profile:
        return None

    recons = profile.get("reconstructions", [])
    target_recon = next((r for r in recons if r.get("recon_id") == recon_id), None)
    if not target_recon:
        return None

    return target_recon.get("mesh")


def set_mesh_state(slug: str, recon_id: str, patch: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Merge *patch* into reconstruction ``recon_id``'s ``mesh`` state block (created if
    absent) and persist. This is the DURABLE WRITE side of the mesh-build lifecycle that
    ``get_mesh_state`` reads: the route seeds ``{"status": "queued"}`` on enqueue, and the
    relay runner records ``{"status": "running"}`` → ``{"status": "done", "glb_path": …,
    "video_path": …, "mesh_json_path": …}`` (or ``{"status": "error", "error": …}``).

    Uses the store's REAL single-writer + atomic idiom (``_LOCK`` / ``_load`` / ``_save`` /
    ``_write_profile_json``) — NOT the unwired ``_save_profile`` stub the older
    ``make_mesh_reconstruction_spec`` references. Returns the updated ``mesh`` dict, or
    ``None`` if the slug/recon_id names no active reconstruction (a best-effort caller — a
    bus runner — tolerates that; the profile may have been archived or the reconstruction
    not yet attached)."""
    if not slug or not isinstance(slug, str):
        return None
    if not recon_id or not isinstance(recon_id, str):
        return None
    if not isinstance(patch, dict):
        raise ProfileError("mesh state patch must be a dict", code="invalid_profile")
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        recons = list(entry.get("reconstructions") or [])
        target_idx = next(
            (i for i, r in enumerate(recons)
             if isinstance(r, dict) and r.get("recon_id") == recon_id),
            None,
        )
        if target_idx is None:
            # MESH-FIRST flow (v0 photos→mesh): there IS no prior reconstruction to hang
            # the mesh state on — the old silent ``return None`` made a successful build
            # invisible to GET .../mesh even though the GLB persisted (keeper 2026-07-14,
            # first live build). Create a minimal record in attach_reconstruction's shape
            # (views empty until a turntable attach replaces it) instead of no-opping.
            recons.append({
                "recon_id": recon_id,
                "created_at": time.time(),
                "views": [],
                "mode": "mesh",
            })
            target_idx = len(recons) - 1
        record = dict(recons[target_idx])
        mesh = dict(record.get("mesh") or {})
        mesh.update(patch)
        record["mesh"] = mesh
        recons[target_idx] = record
        entry = dict(entry)  # edit a copy — nothing is written until _save
        entry["reconstructions"] = recons
        _write_profile_json(slug, entry)  # regenerate the mirror
        data["profiles"][slug] = entry
        _save(data)
    return mesh


# --------------------------------------------------------------------------- #
# VERSIONS slice — append-only version list + active pointer + gen_settings.
#
# The mesh RELAY (runners/identity_render_relay.py) owns the mint moment: on a
# successful build it calls ``mint_version`` with the auto-promoted cardinals as the
# version's canonical, and the new version becomes ACTIVE (latest-wins). The routes
# own activate / rename / archive + the Settings tab (``set_gen_settings``). Every
# mutation is single-writer under ``_LOCK`` with the store's atomic-write idiom, and
# never-delete holds — an archived version is flagged, never dropped from storage.
# --------------------------------------------------------------------------- #
def _auto_version_name(versions: list, kind: str) -> tuple[str, bool]:
    """The auto-name for a freshly minted version + whether it is the BASE.

    The FIRST ``clay`` version ever minted for a profile is the base (name ``"base"``,
    the geometric ground truth). Every other version is ``"<kind>-NN"`` where NN counts
    ALL prior versions of that kind (archived included, so a name is never reused).
    Returns ``(name, is_base)``."""
    has_base = any(isinstance(v, dict) and v.get("base") for v in versions)
    if kind == "clay" and not has_base:
        return "base", True
    nn = 1 + sum(1 for v in versions if isinstance(v, dict) and v.get("kind") == kind)
    return f"{kind}-{nn:02d}", False


def mint_version(
    slug: str,
    recon_id: str,
    kind: str,
    canonical: list[str],
    name: Optional[str] = None,
    canonical_angles: Optional[list[Optional[float]]] = None,
) -> Optional[dict[str, Any]]:
    """Mint (or, on a re-run of the SAME ``recon_id``, update in place) a VERSION and
    make it ACTIVE (latest-wins).

    ``kind`` is one of ``VERSION_KINDS`` (``clay``/``textured``/``styled``). ``canonical``
    is the version's own promoted cardinal set (the id_lock DNA a caller gets when this
    version is active). When ``name`` is ``None`` an auto-name is derived
    (``_auto_version_name``) — the first clay is pinned as the ``base``. Dedupe by
    ``recon_id``: a version already carrying this recon_id (e.g. a bus retry) is updated
    in place rather than duplicated, preserving its ``version_id``/``name``/base flag.

    ``canonical_angles`` (optional, 2026-07-16) is the per-view AZIMUTH provenance,
    positionally aligned with ``canonical`` (an element may be ``None`` = angle unknown).
    Default ``None`` = "no angle information", which is exactly how every version minted
    before this existed behaves — so all pre-existing callers are unaffected. Stored (and
    put on the wire) only when it aligns with ``canonical`` and carries a real angle;
    otherwise it is dropped rather than allowed to mislabel views.

    Append-only + never-delete: an existing version is never removed here. Returns the
    minted/updated version's PUBLIC shape, or ``None`` if the slug names no active
    profile (a best-effort caller — the relay — tolerates that)."""
    if not slug or not isinstance(slug, str):
        return None
    if not recon_id or not isinstance(recon_id, str) or not recon_id.strip():
        raise ProfileError("recon_id is required", code="invalid_profile")
    if kind not in VERSION_KINDS:
        raise ProfileError(f"kind must be one of {list(VERSION_KINDS)}", code="invalid_profile")
    canon = [p for p in (canonical or []) if isinstance(p, str)]
    # Angles ride along ONLY when they describe this exact set (same length) and carry at
    # least one real value; anything else is dropped so a version can never mislabel its
    # own views. ``None`` (every pre-2026-07-16 caller) simply means "no angles".
    canon_angles: Optional[list[Optional[float]]] = None
    if (canonical_angles is not None
            and len(canonical_angles) == len(canon)
            and any(a is not None for a in canonical_angles)):
        canon_angles = list(canonical_angles)
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        entry = dict(entry)  # edit a copy — nothing is written until _save
        versions = list(entry.get("versions") or [])
        existing_idx = next(
            (i for i, v in enumerate(versions)
             if isinstance(v, dict) and v.get("recon_id") == recon_id and not v.get("archived")),
            None,
        )
        if existing_idx is not None:
            version = dict(versions[existing_idx])
            version["kind"] = kind
            version["canonical"] = canon
            # Keep angles in lockstep with the set they describe: a re-run that supplies
            # none must CLEAR any stale angles rather than leave them pointing at old views.
            if canon_angles is not None:
                version["canonical_angles"] = canon_angles
            else:
                version.pop("canonical_angles", None)
            if name is not None:
                version["name"] = name
            versions[existing_idx] = version
        else:
            auto_name, is_base = _auto_version_name(versions, kind)
            version = {
                "version_id": "ver_" + secrets.token_hex(8),
                "name": name if name is not None else auto_name,
                "kind": kind,
                "recon_id": recon_id,
                "created_at": time.time(),
                "canonical": canon,
                "notes": "",
            }
            if canon_angles is not None:
                version["canonical_angles"] = canon_angles
            if is_base and name is None:
                version["base"] = True  # the geometric ground truth — never archivable
            versions.append(version)
        entry["versions"] = versions
        entry["active_version"] = version["version_id"]  # latest-wins
        _write_profile_json(slug, entry)
        data["profiles"][slug] = entry
        _save(data)
    return _public_version(version)


def set_active_version(slug: str, version_id: str) -> Optional[dict[str, Any]]:
    """Point the identity's ACTIVE version at ``version_id`` (the id_lock DNA source).
    Returns the updated PUBLIC profile, or ``None`` if the slug names no active profile
    OR ``version_id`` names no active (non-archived) version of it — the route turns
    either into a 404."""
    if not slug or not isinstance(slug, str) or not version_id or not isinstance(version_id, str):
        return None
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        versions = list(entry.get("versions") or [])
        match = next(
            (v for v in versions
             if isinstance(v, dict) and v.get("version_id") == version_id and not v.get("archived")),
            None,
        )
        if match is None:
            return None
        entry = dict(entry)
        entry["active_version"] = version_id
        _write_profile_json(slug, entry)
        data["profiles"][slug] = entry
        _save(data)
    return _public(slug, entry)


def update_version(
    slug: str,
    version_id: str,
    *,
    name: Optional[str] = None,
    notes: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Partial edit of a version's DISPLAY fields (``name`` / ``notes``) in place — an
    argument left at ``None`` is a no-op for that field (the route only forwards keys
    actually present). Returns the updated PUBLIC profile, or ``None`` if the slug/version
    is unknown or archived (route 404). A blank ``name`` is a ProfileError -> the route's
    400."""
    if not slug or not isinstance(slug, str) or not version_id or not isinstance(version_id, str):
        return None
    new_name: Optional[str] = None
    if name is not None:
        new_name = name.strip()
        if not new_name:
            raise ProfileError("name is required", code="invalid_profile")
    new_notes: Optional[str] = None
    if notes is not None:
        if not isinstance(notes, str):
            raise ProfileError("notes must be a string", code="invalid_profile")
        new_notes = notes.strip()
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        versions = list(entry.get("versions") or [])
        idx = next(
            (i for i, v in enumerate(versions)
             if isinstance(v, dict) and v.get("version_id") == version_id and not v.get("archived")),
            None,
        )
        if idx is None:
            return None
        version = dict(versions[idx])
        if new_name is not None:
            version["name"] = new_name
        if new_notes is not None:
            version["notes"] = new_notes
        versions[idx] = version
        entry = dict(entry)
        entry["versions"] = versions
        _write_profile_json(slug, entry)
        data["profiles"][slug] = entry
        _save(data)
    return _public(slug, entry)


def archive_version(slug: str, version_id: str) -> Optional[dict[str, Any]]:
    """ARCHIVE a version (never-delete: flag it ``archived``, keep its bytes, drop it
    from the wire list). REFUSED for the clay BASE (the geometric ground truth) and for
    the currently ACTIVE version — either is a ProfileError -> the route's 400 with a
    clear message. Returns the updated PUBLIC profile, or ``None`` if the slug/version is
    unknown / already archived (route 404)."""
    if not slug or not isinstance(slug, str) or not version_id or not isinstance(version_id, str):
        return None
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        versions = list(entry.get("versions") or [])
        idx = next(
            (i for i, v in enumerate(versions)
             if isinstance(v, dict) and v.get("version_id") == version_id and not v.get("archived")),
            None,
        )
        if idx is None:
            return None
        if versions[idx].get("base"):
            raise ProfileError(
                "the clay base version cannot be archived (it is the identity's geometric "
                "ground truth)", code="invalid_profile")
        if entry.get("active_version") == version_id:
            raise ProfileError(
                "the active version cannot be archived — activate another version first",
                code="invalid_profile")
        version = dict(versions[idx])
        version["archived"] = True
        version["archived_at"] = time.time()
        versions[idx] = version
        entry = dict(entry)
        entry["versions"] = versions
        _write_profile_json(slug, entry)
        data["profiles"][slug] = entry
        _save(data)
    return _public(slug, entry)


def _valid_vision_model_keys() -> set[str]:
    """The set of model keys the FLEET currently advertises as ``image-text-to-text``
    capable — the ONLY keys a per-identity ``gen_settings.vision_model`` may name.

    Resolved from the LIVE vision registry (never a hardcode) so the setting can never
    point at a model that has been renamed away or that isn't a VL model. The vision
    registry is the UNION of ``image-text-to-text`` AND ``text-to-image`` models, so we
    filter to the image-text-to-text capability specifically: the front-select step is an
    image->text ask, and a pure text-to-image generator key would route to the wrong
    engine.

    Imported LAZILY (inside this call), NEVER at module import time: this store module is
    imported at app boot, and pulling the vision registry up to boot would make a cheap
    store import expensive. A registry-import failure is a fail-CLOSED empty set — a
    non-null vision_model then fails validation with a clear message rather than being
    silently accepted unvalidatable (the None/"" "fleet default" path never reaches here,
    so clearing/keeping the default still works on a stripped install)."""
    try:
        from hugpy_media.vision.vision_coder import VISION_MODELS_REGISTRY
    except Exception:  # noqa: BLE001 — no registry available -> nothing validates as a VL key
        return set()
    return {
        k for k, cfg in VISION_MODELS_REGISTRY.items()
        if "image-text-to-text" in (getattr(cfg, "tasks", None) or ())
    }


def preferred_identity_vision_model() -> Optional[str]:
    """The AUTO default VL model for identity 3D-imaging front-select: a 7B when the
    fleet has one, else None (== the fleet-default VL, the 3B).

    Operator 2026-07-15: "7B should be default if it is available" — the 3B mislabeled
    a waist-up reference as full-body, so the bigger judge is the better default for
    exactly this call. Availability = the live image-text-to-text registry carries a
    Qwen2.5-VL-7B key (installed models only); sorted() makes the pick deterministic
    when several vendor-prefixed copies exist. The relay retries WITHOUT a model on a
    failed 7B call, so an unloadable 7B degrades to the 3B, never to no judgment."""
    sevens = sorted(k for k in _valid_vision_model_keys() if "Qwen2.5-VL-7B" in k)
    return sevens[0] if sevens else None


def set_gen_settings(slug: str, partial: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Merge a PARTIAL gen_settings update into the identity's stored settings. Only the
    contract's known keys are accepted (an unknown key is a ProfileError -> the route's
    400); every value is type-checked, ``pose`` is enum-checked, ``front_ref`` (when
    non-null) is JAILED to the profile's OWN reference images, and ``vision_model`` (when
    non-null) is validated against the LIVE image-text-to-text registry so it can never
    name a non-existent / non-VL model. Returns the updated PUBLIC profile (its
    ``gen_settings`` reflecting the merge), or ``None`` if the slug names no active
    profile (route 404)."""
    if not isinstance(partial, dict):
        raise ProfileError("gen_settings must be an object", code="invalid_profile")
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        own = {p for p in (entry.get("source_images") or []) if isinstance(p, str)}
        cleaned: dict[str, Any] = {}
        for k, v in partial.items():
            if k not in _DEFAULT_GEN_SETTINGS:
                raise ProfileError(f"unknown gen_settings key {k!r}", code="invalid_profile")
            if k in ("texture", "auto_promote", "remove_background"):
                if not isinstance(v, bool):
                    raise ProfileError(f"{k} must be a bool", code="invalid_profile")
            elif k == "pose":
                if v not in _POSE_CHOICES:
                    raise ProfileError(
                        f"pose must be one of {list(_POSE_CHOICES)}", code="invalid_profile")
            elif k in ("frame_count", "fps", "width", "height"):
                if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                    raise ProfileError(f"{k} must be a positive int", code="invalid_profile")
            elif k == "front_ref":
                if v is not None and (not isinstance(v, str) or v not in own):
                    raise ProfileError(
                        "front_ref must be one of the profile's own reference images",
                        code="invalid_profile")
            elif k in ("cleanup_prompt", "negative_prompt"):
                # CLEANUP-PROMPT slice: plain string render-steer channels. None/"" ==
                # "no steer" (byte-identical to today), stored as "". Any non-string is a
                # ProfileError -> the route's 400. Whitespace is preserved verbatim (the
                # relay/reconstruction wiring strips as needed) beyond the None->"" coerce.
                if v is None:
                    v = ""
                elif not isinstance(v, str):
                    raise ProfileError(f"{k} must be a string", code="invalid_profile")
            elif k == "vision_model":
                # None/"" == "use the fleet-default VL model" (the 3B): a zero-regression
                # CLEAR, always accepted and stored as None. A non-empty value must be a
                # model key the fleet advertises as image-text-to-text capable (checked
                # against the LIVE registry, lazily). Anything else is a ProfileError ->
                # the route's 400, so the setting can never point at a non-VL / unknown
                # model. Validated LAST so the lazy registry import only happens when a
                # real key is actually being set.
                if v in (None, ""):
                    v = None
                elif not isinstance(v, str) or v not in _valid_vision_model_keys():
                    raise ProfileError(
                        "vision_model must be null/empty (use the fleet-default VL model) "
                        "or a model key the fleet advertises as image-text-to-text capable",
                        code="invalid_profile")
            cleaned[k] = v
        entry = dict(entry)
        gs = dict(entry.get("gen_settings") or {})
        gs.update(cleaned)
        entry["gen_settings"] = gs
        _write_profile_json(slug, entry)
        data["profiles"][slug] = entry
        _save(data)
    return _public(slug, entry)


# --------------------------------------------------------------------------- #
# ANGLE BANK (IDENTITY-3D-CONTINUITY-PLAN.md S1+S2) — the turntable ring as a
# QUERYABLE, read-only asset over EXISTING state. No new persisted manifest, no
# migration, no writes: the bank is a COMPUTED read over what the mesh relay already
# rendered and ``attach_reconstruction`` already stored.
#
# Where the bank lives: each version carries a ``recon_id``; the entry's
# ``reconstructions`` list holds a record for that id with ``mode == "turntable"``,
# an ordered ``views`` list (``view_00.png`` … ``view_71.png`` copied under
# ``<slug>/reconstruction/<recon_id>/views/``), plus ``frame_count`` and
# ``degrees_per_frame`` (luigi: 72 frames @ 5.0°/frame). So an angle bank is simply
# that record read back with an azimuth attached to each frame by INDEX.
#
# AZIMUTH / ROTATION CONVENTION — the single source of truth (documented ONCE here):
#   * ``azimuth_deg`` is CANONICAL. Semantic names (front/back/…-profile/…) are only a
#     convenience map over degrees (``SEMANTIC_VIEWS``); when the two ever disagree,
#     degrees win. This keeps the labels re-pinnable without touching stored bytes.
#   * Frame 0 == azimuth 0° == FRONT (turntable convention: the orbit starts head-on).
#   * Azimuth rises with the frame index in the render's orbit direction:
#         azimuth_deg(N) = (N * degrees_per_frame) % 360
#     so the ring reads 0° → … → 355° over 72 frames and WRAPS (frame 71 == 355°, one
#     step short of a full turn back to front). ``degrees_per_frame`` comes off the
#     record; on an older record that predates that field it is derived as
#     ``360.0 / len(views)`` (a full turn spread evenly over the frames rendered).
#   * The orbit turns the subject so increasing azimuth first brings the subject's
#     LEFT side toward camera — hence 90° == left-profile, 180° == back, 270° ==
#     right-profile. VERIFIED 2026-07-16: the operator eyeballed the labeled 8-view rows
#     on two textured identities and confirmed "0 is front and 90 degrees is stage left"
#     (stage left == the performer's left == the camera's right, so at 90° the camera
#     sees the subject's own LEFT side). This SUPERSEDES the prior never-eyeballed guess
#     (90° == right-profile), which was mirrored. Only ``SEMANTIC_VIEWS`` changed — the
#     azimuth degrees were canonical and correct all along, so no stored data, no azimuth
#     math, and no bank index changes were touched. 0°/180° are the mirror's fixed points.
#   * ``elevation_deg`` is recorded as 0.0 for every ring frame: the turntable is
#     single-elevation. Real elevations (overhead / low-angle) are a later slice (S5,
#     on-demand novel views from the mesh); they are deliberately NOT synthesized here.
#
# Angular nearness always WRAPS: 10° and 350° are 20° apart, never 340° (see
# ``_angular_distance``). Selection spreads OUTWARD by increasing wrap-distance, so a
# hint never returns k copies of the single nearest frame — it returns the k distinct
# frames straddling the target.
# --------------------------------------------------------------------------- #

# name -> azimuth_deg. A convenience map over the canonical degrees (see the section
# docstring's convention). Names are matched case-insensitively after trimming. Only
# the azimuthal views the single-elevation ring can actually serve are listed — a pure
# elevation ask (e.g. "overhead") is intentionally absent and resolves to a clean error
# rather than a wrong frame, until the mesh-render slice (S5) can serve elevations.
# HANDEDNESS RESOLVED 2026-07-16 (operator eyeballed the labeled 8-view rows on two
# textured identities — Luigi + keeper-mesh-live — verbatim: "the views are accurate 0 is
# front and 90 degrees is stage left").
#
# STAGE LEFT is the PERFORMER's left, i.e. the camera's/viewer's RIGHT. So at azimuth 90°
# the camera sees the subject's OWN LEFT side => 90° is the LEFT profile, not the right.
# The prior assignment (90° == right-profile) was the documented-but-never-eyeballed guess
# recorded in the section docstring; the real orbit turns the other way. This is a MIRROR,
# not an offset: 0° front and 180° back are FIXED POINTS (confirmed — "0 is front"), and
# only the sided names swap across them (45<->315, 90<->270, 135<->225).
#
# Per the convention, azimuth degrees are canonical, so this is the ONLY place that
# changes: no stored bytes, no azimuth math, no bank/frame indices, no re-render. The
# ring frame at 90° is the same pixels it always was — it is now named correctly.
#
# NB this also corrects `shot_intent.py`, which maps prompt text -> a view NAME: a prompt
# asking for the subject's right side had been silently resolving to the mirrored frame.
SEMANTIC_VIEWS: dict[str, float] = {
    "front": 0.0,
    "three-quarter-left": 45.0,
    "left-profile": 90.0,
    "back-left": 135.0,
    "back": 180.0,
    "back-right": 225.0,
    "right-profile": 270.0,
    "three-quarter-right": 315.0,
}


def _angular_distance(a: float, b: float) -> float:
    """Wrap-aware angular distance in degrees, always in ``[0, 180]``. Python's ``%``
    returns the sign of the divisor, so ``(a - b) % 360`` is already non-negative; the
    ``min(d, 360 - d)`` is what makes 10° and 350° read as 20° apart (the short way
    round the circle), never 340° — the whole reason a hint at 350° selects frames near
    0° rather than marching the long way back through 180°."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def canonical_frame_indices(view_count: int, degrees_per_frame: Optional[float] = None) -> list[int]:
    """The ring-frame INDICES that back the canonical set — one per ``SEMANTIC_VIEWS``
    azimuth, chosen by NEAREST ANGLE (operator 2026-07-16: "45 degree shots").

    Selection is by DEGREES, never by name or by an index shortcut: for each target azimuth
    in ``SEMANTIC_VIEWS`` this walks the ring and keeps the frame whose computed azimuth
    (``index * degrees_per_frame % 360``, the section's convention) is angularly nearest
    (``_angular_distance``, so 350° is 10° from 0°, not 350°). On the live rings (72 frames
    @ 5°/frame) every 45° target lands EXACTLY on a rendered frame — 0/9/18/27/36/45/54/63 —
    so this is a pure SELECTION over frames that already exist. Nothing is re-rendered,
    interpolated, or synthesized.

    Why nearest-angle and not ``N//8``: an index shortcut silently lies on any ring whose
    frame count isn't a multiple of 8 (or whose ``degrees_per_frame`` doesn't close a full
    turn). Nearest-angle degrades honestly — on a sparse ring two targets may resolve to the
    same frame, so the result is DEDUPED and stays in ascending ring order. That means a
    coarse ring simply yields FEWER than 8 canonical entries rather than duplicate bytes.

    ``degrees_per_frame`` defaults to a full turn spread over the frames (``360/view_count``)
    for older records lacking the field — the same derivation ``bank_views`` uses. Returns
    ``[]`` for a non-positive ``view_count``.

    NOTE (S5): the 2 AERIAL views the operator also asked for (overhead / -overhead) are NOT
    here and must not be faked from this ring — the turntable is single-elevation
    (``elevation_deg`` is 0.0 for every frame). They need the mesh-render path (S5).
    """
    if view_count <= 0:
        return []
    try:
        dpf = float(degrees_per_frame)  # type: ignore[arg-type]
        if dpf <= 0:
            raise ValueError
    except (TypeError, ValueError):
        dpf = 360.0 / view_count
    picked: list[int] = []
    for target in SEMANTIC_VIEWS.values():
        best = min(
            range(view_count),
            key=lambda i: (_angular_distance(target, (i * dpf) % 360.0), i),
        )
        if best not in picked:
            picked.append(best)  # dedupe: a sparse ring may map 2 targets onto 1 frame
    return sorted(picked)


def render_refs_from_canonical(canonical: list[str]) -> list[str]:
    """The <=``MAX_RENDER_REFS`` subset of a canonical set that ONE id_lock/VACE render may
    consume — the seam that lets the STORE hold 8 while the RENDER channel still takes 4.

    Canonical is stored in ascending ring order (front, 45°, 90°, …), so the first 4 entries
    of an 8-view set would be a LOPSIDED half-turn (0/45/90/135 — the subject's front and
    right side only), which is worse identity DNA than today's 4 cardinals. Instead this
    takes an evenly-STRIDED sample across the ring, which for an 8-view set yields exactly
    the 4 cardinals (0°/90°/180°/270°) — i.e. byte-identical DNA to what a 4-view profile
    feeds the renderer today. A 4-view set (every profile on disk right now) strides by 1
    and passes through UNCHANGED, so existing identities render exactly as before.

    Shorter sets pass through untouched. This never raises — an over-long set is sampled,
    never rejected, because the caller is a render path, not a validator."""
    refs = [p for p in (canonical or []) if isinstance(p, str)]
    n = len(refs)
    if n <= MAX_RENDER_REFS:
        return refs
    # Evenly spaced picks across the ring: for n=8 -> indices 0,2,4,6 == the cardinals.
    stride = n / float(MAX_RENDER_REFS)
    out: list[str] = []
    for j in range(MAX_RENDER_REFS):
        idx = int(j * stride)
        if idx < n and refs[idx] not in out:
            out.append(refs[idx])
    return out


def _resolve_version(profile: dict[str, Any], version_id: Optional[str]) -> Optional[dict[str, Any]]:
    """Pick the version whose bank a caller means — MIRRORING the studio resolver's
    precedence (``video_routes._reference_images_from_body``): ``version_id`` names a
    specific version by its id OR its name (e.g. "textured-01"); ``None`` falls back to
    the profile's ACTIVE version. Returns the version dict from the PUBLIC profile shape,
    or ``None`` when nothing matches (a versionless profile, or an unknown id/name) — the
    caller then reports an empty bank and degrades, never crashes."""
    versions = [v for v in (profile.get("versions") or []) if isinstance(v, dict)]
    if version_id is None:
        active_id = profile.get("active_version")
        if not active_id:
            return None
        return next((v for v in versions if v.get("version_id") == active_id), None)
    return next(
        (v for v in versions
         if v.get("version_id") == version_id or v.get("name") == version_id),
        None,
    )


def _turntable_recon(profile: dict[str, Any], version: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The turntable reconstruction record backing *version*'s bank, or ``None``. Maps
    ``version.recon_id`` onto the entry's ``reconstructions`` list and requires
    ``mode == "turntable"`` — a clay/mesh-only or sheet recon carries no orbit ring and
    yields no bank (the caller falls back to the flat canonical set)."""
    if not isinstance(version, dict):
        return None
    recon_id = version.get("recon_id")
    if not recon_id:
        return None
    return next(
        (r for r in (profile.get("reconstructions") or [])
         if isinstance(r, dict) and r.get("recon_id") == recon_id and r.get("mode") == "turntable"),
        None,
    )


def bank_views(profile: dict[str, Any], *, version_id: Optional[str] = None) -> list[dict[str, Any]]:
    """The angle bank for a version's turntable ring — a pure, read-only COMPUTED view
    over existing store state (no ``_LOCK``, no writes, no new persisted manifest).

    *profile* is the PUBLIC shape (``get_profile``/``_public``). The chosen version is the
    ACTIVE one when ``version_id`` is ``None``, else the version named by id-or-name
    (``_resolve_version`` mirrors the studio resolver's precedence). Returns, in ANGULAR
    (frame) order::

        [{"index", "azimuth_deg", "elevation_deg", "path", "source": "turntable"}, …]

    with ``azimuth_deg = (index * degrees_per_frame) % 360`` per the section's convention
    (``degrees_per_frame`` off the record, or ``360/len(views)`` on an older record that
    lacks it), and ``elevation_deg`` fixed at 0.0 (single-elevation ring).

    Returns ``[]`` when the chosen version has no turntable reconstruction — a versionless
    profile, an unknown version, a clay/sheet-only recon — so the caller can fall back to
    today's flat canonical set. Frames whose file is missing on disk are FILTERED OUT
    (``os.path.isfile``), but the azimuth of every surviving frame is computed from its
    ORIGINAL ring index, so a gap never rotates the remaining angles."""
    version = _resolve_version(profile, version_id)
    recon = _turntable_recon(profile, version)
    if recon is None:
        return []
    # A turntable record's ``views`` is an ordered list of PNG paths (the orbit frames
    # copied in by ``_materialize_views``); non-strings are defensively skipped.
    views = [v for v in (recon.get("views") or []) if isinstance(v, str)]
    if not views:
        return []
    dpf = recon.get("degrees_per_frame")
    try:
        dpf = float(dpf)
        if dpf <= 0:
            raise ValueError
    except (TypeError, ValueError):
        dpf = 360.0 / len(views)  # older record: spread a full turn over the frames it has
    bank: list[dict[str, Any]] = []
    for index, path in enumerate(views):
        if not os.path.isfile(path):
            continue  # a dropped frame must not shift the angles of the survivors
        bank.append({
            "index": index,
            "azimuth_deg": round((index * dpf) % 360.0, 4),
            "elevation_deg": 0.0,
            "path": path,
            "source": "turntable",
        })
    return bank


def nearest_bank_views(bank: list[dict[str, Any]], azimuth_deg: float, k: int) -> list[dict[str, Any]]:
    """The ``k`` bank frames nearest *azimuth_deg*, wrap-aware and angle-SPREAD.

    Sorts the bank by wrap-aware angular distance to the target (``_angular_distance``,
    so 350° is near 0°) and takes the closest ``k`` DISTINCT frames — because every ring
    frame is a distinct angle, the result naturally straddles the target (350° → 350°,
    345°, 355°, 0° …) rather than returning ``k`` copies of the single nearest. The tie
    break is azimuth then path, purely for determinism. Paths are de-duplicated
    defensively. Returns ``min(k, len(bank))`` entries; ``[]`` for an empty bank or
    ``k <= 0``."""
    if not bank or k <= 0:
        return []
    ordered = sorted(
        bank,
        key=lambda v: (_angular_distance(azimuth_deg, v["azimuth_deg"]), v["azimuth_deg"], v["path"]),
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for v in ordered:
        p = v["path"]
        if p in seen:
            continue
        seen.add(p)
        out.append(v)
        if len(out) >= k:
            break
    return out


def azimuth_for_view(hint: Any) -> tuple[Optional[float], Optional[str]]:
    """Resolve a view HINT to a canonical azimuth in ``[0, 360)``.

    Accepts either a SEMANTIC string (looked up case-insensitively in ``SEMANTIC_VIEWS``)
    or a DICT ``{"azimuth_deg": <number>, "elevation_deg"?: <number>}``. Elevation is
    accepted for forward-compatibility but does NOT affect selection today — the ring is
    single-elevation, so nearness is azimuth-only (see S5 for real elevations).

    Errors-as-data (never raises): returns ``(azimuth_deg, None)`` on success or
    ``(None, message)`` on a bad hint — an unknown semantic name, a dict missing a numeric
    ``azimuth_deg``, or a wrong type. The route turns the message into a clean 400. The
    returned azimuth is wrapped into ``[0, 360)`` so an out-of-range dict value (e.g.
    370°) is normalized rather than rejected."""
    if isinstance(hint, str):
        name = hint.strip().lower()
        if name in SEMANTIC_VIEWS:
            return SEMANTIC_VIEWS[name] % 360.0, None
        return None, (f"unknown identity_view {hint!r}; expected one of "
                      f"{sorted(SEMANTIC_VIEWS)} or an object with a numeric azimuth_deg")
    if isinstance(hint, dict):
        az = hint.get("azimuth_deg")
        if isinstance(az, bool) or not isinstance(az, (int, float)):
            return None, "identity_view object must carry a numeric azimuth_deg"
        elev = hint.get("elevation_deg")
        if elev is not None and (isinstance(elev, bool) or not isinstance(elev, (int, float))):
            return None, "identity_view elevation_deg must be a number when present"
        return float(az) % 360.0, None
    return None, "identity_view must be a semantic name string or an {azimuth_deg} object"


def views_summary(profile: dict[str, Any], *, version_id: Optional[str] = None) -> dict[str, Any]:
    """A compact, read-only summary of a version's angle ring — count + degrees_per_frame
    + azimuth range — for the operator/UI to see WHAT angles an identity carries without
    shipping every frame path. Computed from the same bank as ``bank_views`` (so ``count``
    reflects frames that actually exist on disk), with ``degrees_per_frame``/``frame_count``
    read straight off the record. A version with no turntable ring returns the zeroed shape
    (``count`` 0, ranges ``None``, ``source`` ``None``) rather than being omitted, so a
    caller can rely on the key always being present (defaults-are-promises)."""
    version = _resolve_version(profile, version_id)
    recon = _turntable_recon(profile, version)
    if recon is None:
        return {"count": 0, "frame_count": 0, "degrees_per_frame": None,
                "azimuth_min": None, "azimuth_max": None, "source": None}
    views = [v for v in (recon.get("views") or []) if isinstance(v, str)]
    dpf = recon.get("degrees_per_frame")
    try:
        dpf = float(dpf)
        if dpf <= 0:
            raise ValueError
    except (TypeError, ValueError):
        dpf = round(360.0 / len(views), 6) if views else None
    bank = bank_views(profile, version_id=version_id)
    azimuths = [b["azimuth_deg"] for b in bank]
    return {
        "count": len(bank),
        "frame_count": recon.get("frame_count") or len(views),
        "degrees_per_frame": dpf,
        "azimuth_min": min(azimuths) if azimuths else None,
        "azimuth_max": max(azimuths) if azimuths else None,
        "source": "turntable",
    }
