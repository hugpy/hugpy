"""ANGLE BANK slice (IDENTITY-3D-CONTINUITY-PLAN.md S1+S2).

Exercises view-aware DNA resolution: the identity's turntable RING (already on disk
as ordered ``view_NN.png`` frames under a ``mode:"turntable"`` reconstruction) is read
back as a queryable angle bank, and an optional ``identity_view`` hint selects the
K angle-nearest frames from it — WITHOUT any new persisted state, migration, or write.

Isolation idiom is lifted verbatim from test_identity_versions.py: rebind the store
module globals (IDENTITIES_HOME/PROJECTS_HOME) to temp dirs under the real DEFAULT_ROOT,
jail reference images under the real UPLOADS_HOME, point the media bus at a temp DB, and
build a minimal Flask app with the video blueprint.

Locks (each check runs independently so one failure never masks the rest):
  [bank]
   1. bank_views computes correct azimuths from a synthetic 72-frame turntable recon
      (index N -> N*degrees_per_frame, elevation 0.0, angular order, source turntable).
  [selection]
   2. nearest_bank_views is WRAP-aware and angle-spread: a hint at 350° picks frames
      straddling 0°/340° (proves 350↔0 is 10° apart, not 350°), distinct, no dupes.
  [semantic]
   3. azimuth_for_view maps semantic names -> azimuth (case-insensitive), accepts an
      {azimuth_deg} object (wrapping out-of-range), and errors-as-data on a bad hint.
  [resolver]
   4. _reference_images_from_body WITH a view hint returns bank frames near the hint;
      WITHOUT a hint returns EXACTLY the canonical set (asserted equal to the no-hint path).
   5. an invalid identity_view is a clean 400 tuple (the resolver's error idiom).
   6. a versionless / no-turntable profile WITH a hint degrades to the canonical set
      (no crash, byte-identical to the hintless fallback).
  [route]
   7. GET /video/identity-profiles/<slug> surfaces a per-version ``views`` summary
      (count + degrees_per_frame + azimuth range) — additive, never removing a key.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python -m pytest tests/test_identity_views.py -q
  venv/bin/python tests/test_identity_views.py
"""
from __future__ import annotations

import atexit
import logging
import os
import shutil
import sqlite3
import sys
import tempfile

logging.disable(logging.INFO)

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import importlib  # noqa: E402

from flask import Flask  # noqa: E402

from hugpy_video.intel import identity_profiles
from hugpy_video.intel import media_bus
from hugpy_platform.constants import UPLOADS_HOME, DEFAULT_ROOT

# STORE isolation — rebind the module globals the store path helpers read (env isolation
# does not work: constants read the .env file, not os.environ). IDENTITIES_HOME must sit
# under the real DEFAULT_ROOT so the identity-owned ref copies still pass the route jail.
_TMP_IDENTITIES = tempfile.mkdtemp(prefix="hugpy-idview-store-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
identity_profiles.IDENTITIES_HOME = _TMP_IDENTITIES
_TMP_PROJECTS = tempfile.mkdtemp(prefix="hugpy-idview-projects-")
identity_profiles.PROJECTS_HOME = _TMP_PROJECTS

# JAIL: reference/frame images must resolve under the real UPLOADS_HOME.
_TMP_UPLOADS = tempfile.mkdtemp(prefix="hugpy-idview-uploads-", dir=UPLOADS_HOME)

# media bus -> temp DB so nothing touches the real catalog.
_TMP_DB = tempfile.mkstemp(prefix="idview-bus-", suffix=".db")[1]
media_bus.DB_PATH = _TMP_DB
media_bus._initialized = False
with sqlite3.connect(_TMP_DB) as _c:
    _c.execute(
        "CREATE TABLE IF NOT EXISTS media_jobs ("
        "job_id TEXT PRIMARY KEY, name TEXT, status TEXT, spec_json TEXT, "
        "result_json TEXT, claim_token TEXT, created REAL, updated REAL, "
        "progress_json TEXT)")

vr = importlib.import_module("hugpy_server.app.routes.video_routes")

app = Flask(__name__)
app.register_blueprint(vr.video_bp)
client = app.test_client()


@atexit.register
def _cleanup() -> None:
    for d in (_TMP_IDENTITIES, _TMP_PROJECTS, _TMP_UPLOADS):
        shutil.rmtree(d, ignore_errors=True)
    try:
        os.remove(_TMP_DB)
    except OSError:
        pass


def _make_png(path: str, color=(120, 120, 120)) -> None:
    from PIL import Image

    Image.new("RGB", (16, 16), color).save(path)


_IMG_A = os.path.join(_TMP_UPLOADS, "view_a.png")
_IMG_B = os.path.join(_TMP_UPLOADS, "view_b.png")
_make_png(_IMG_A, (200, 40, 40))
_make_png(_IMG_B, (40, 200, 40))


def _fresh_profile(name: str) -> dict:
    """A plain profile (no reconstruction, no version) via the store — its refs copied
    into IDENTITIES_HOME so they exist on disk for the resolver's existence filter."""
    return identity_profiles.create_profile(name, [_IMG_A, _IMG_B], notes="")


def _turntable_profile(name: str, frame_count: int = 72, dpf: float = 5.0):
    """Build a profile whose ACTIVE version is backed by a synthetic ``frame_count``-frame
    turntable reconstruction (degrees_per_frame ``dpf``), returning
    ``(slug, recon_id, ordered_view_paths, canonical)``.

    The frames are real (empty-but-present) files so ``attach_reconstruction`` copies them
    in and the bank's ``os.path.isfile`` filter keeps them; the version's canonical is the
    profile's own first ref (an existing path) so the hintless resolve has a known set."""
    p = _fresh_profile(name)
    slug = p["slug"]
    frame_dir = tempfile.mkdtemp(prefix="tt-src-", dir=_TMP_UPLOADS)
    sources = []
    for i in range(frame_count):
        fp = os.path.join(frame_dir, f"frame_{i:04d}.png")
        # A present file is all the bank's existence filter checks — a byte is enough.
        with open(fp, "wb") as f:
            f.write(b"\x89PNG\r\n")
        sources.append(fp)
    recon_id = "recon_tt_" + slug
    identity_profiles.attach_reconstruction(
        slug, recon_id, sources,
        spec={"mode": "turntable", "degrees_per_frame": dpf, "frame_count": frame_count},
    )
    owned = identity_profiles.get_profile(slug)["reference_images"]
    canonical = [owned[0]]
    identity_profiles.mint_version(slug, recon_id, "textured", canonical)
    # The stored (copied-in) ring frames, in angular order — index N == view_NN.png.
    rec = identity_profiles.get_reconstruction(slug, recon_id)
    return slug, recon_id, list(rec["views"]), canonical


# --------------------------------------------------------------------------- #
# [1] bank_views — azimuths computed from index; angular order; source/elevation
# --------------------------------------------------------------------------- #
def test_bank_views_computes_azimuths_from_index():
    slug, recon_id, views, _canon = _turntable_profile("Bank One")
    profile = identity_profiles.get_profile(slug)
    bank = identity_profiles.bank_views(profile)

    assert len(bank) == 72, len(bank)
    # index N -> azimuth N*5 (mod 360); frame 0 == front == 0°, frame 71 == 355°.
    assert bank[0]["azimuth_deg"] == 0.0
    assert bank[18]["azimuth_deg"] == 90.0      # right-profile per the documented convention
    assert bank[36]["azimuth_deg"] == 180.0     # back
    assert bank[54]["azimuth_deg"] == 270.0     # left-profile
    assert bank[71]["azimuth_deg"] == 355.0     # one 5° step short of a full turn
    # angular order (strictly increasing over the single wrap-free ring) + shape contract.
    assert [b["index"] for b in bank] == list(range(72))
    assert all(b["elevation_deg"] == 0.0 for b in bank)      # single-elevation ring
    assert all(b["source"] == "turntable" for b in bank)
    assert all(os.path.isfile(b["path"]) for b in bank)      # existence-filtered
    assert [b["path"] for b in bank] == views                # angular-order paths preserved


# --------------------------------------------------------------------------- #
# [2] nearest_bank_views — WRAP-aware + angle-spread, no duplicate frames
# --------------------------------------------------------------------------- #
def test_nearest_bank_views_is_wrap_aware_and_spread():
    slug, _recon_id, _views, _canon = _turntable_profile("Bank Two")
    profile = identity_profiles.get_profile(slug)
    bank = identity_profiles.bank_views(profile)

    picked = identity_profiles.nearest_bank_views(bank, 350.0, 4)
    az = [p["azimuth_deg"] for p in picked]
    # 350° is 10° from 0°, NOT 350° — a wrap-blind distance would never reach frame 0.
    assert 0.0 in az, az
    # every pick is within one step-cluster of the target the SHORT way round the circle.
    assert all(identity_profiles._angular_distance(350.0, a) <= 10.0 for a in az), az
    # concrete spread for this bank: {350, 345, 355, 0} — straddles the target, no dupes.
    assert set(az) == {350.0, 345.0, 355.0, 0.0}, az
    assert len({p["path"] for p in picked}) == 4              # distinct frames, never k copies
    assert len(picked) == 4

    # k larger than the bank is clamped; k<=0 / empty bank are empty.
    assert len(identity_profiles.nearest_bank_views(bank, 90.0, 1000)) == 72
    assert identity_profiles.nearest_bank_views(bank, 90.0, 0) == []
    assert identity_profiles.nearest_bank_views([], 90.0, 4) == []


# --------------------------------------------------------------------------- #
# [3] azimuth_for_view — semantic map + object form + errors-as-data
# --------------------------------------------------------------------------- #
def test_azimuth_for_view_semantic_and_object():
    # HANDEDNESS (operator-verified 2026-07-16, two textured identities): "0 is front
    # and 90 degrees is stage left" — stage left == the performer's left == the camera's
    # RIGHT, so at 90° the camera sees the subject's own LEFT side. These expectations
    # were MIRRORED from the original never-eyeballed guess (which had 90° ==
    # right-profile). 0°/180° are the mirror's fixed points. Degrees stayed canonical
    # throughout; only the sided NAMES moved. Do not "correct" this back.
    for name, want in (("front", 0.0), ("left-profile", 90.0), ("back", 180.0),
                       ("right-profile", 270.0), ("three-quarter-left", 45.0),
                       ("three-quarter-right", 315.0)):
        az, err = identity_profiles.azimuth_for_view(name)
        assert err is None and az == want, (name, az, err)
    # case-insensitive + trimmed.
    assert identity_profiles.azimuth_for_view("  Back ") == (180.0, None)
    # object form; an out-of-range azimuth wraps into [0, 360).
    assert identity_profiles.azimuth_for_view({"azimuth_deg": 123.0}) == (123.0, None)
    assert identity_profiles.azimuth_for_view({"azimuth_deg": 370, "elevation_deg": 15}) == (10.0, None)
    # errors-as-data (never raises): unknown name, object missing azimuth, wrong type.
    for bad in ("sideways", "overhead", {"elevation_deg": 5}, {"azimuth_deg": "x"}, 123, None):
        az, err = identity_profiles.azimuth_for_view(bad)
        assert az is None and isinstance(err, str) and err, (bad, az, err)


# --------------------------------------------------------------------------- #
# [4] resolver — a hint returns bank frames; NO hint returns EXACTLY the canonical set
# --------------------------------------------------------------------------- #
def test_reference_images_from_body_view_hint_vs_canonical():
    slug, recon_id, views, canonical = _turntable_profile("Resolve Four")

    # NO hint -> the active version's canonical set, existence-filtered (today's behavior).
    no_hint, err = vr._reference_images_from_body({"identity_profile": slug})
    assert err is None and no_hint == canonical, (no_hint, err)

    # WITH a "back" (180°) hint -> the 4 angle-nearest RING frames, not the cardinals.
    hinted, err = vr._reference_images_from_body(
        {"identity_profile": slug, "identity_view": "back"})
    assert err is None, err
    assert len(hinted) == 4
    # 180° == index 36; nearest-4 by (distance, azimuth) == indices {36, 35, 37, 34}.
    assert set(hinted) == {views[36], views[35], views[37], views[34]}, hinted
    # the hint demonstrably changed the DNA away from the flat canonical set.
    assert set(hinted) != set(no_hint)

    # an object hint resolves the same ring by degrees (front == 0°).
    front, err = vr._reference_images_from_body(
        {"identity_profile": slug, "identity_view": {"azimuth_deg": 0.0}})
    assert err is None and set(front) == {views[0], views[1], views[71], views[2]}, front


# --------------------------------------------------------------------------- #
# [5] resolver — an invalid identity_view is a clean 400 tuple
# --------------------------------------------------------------------------- #
def test_reference_images_from_body_invalid_view_is_400():
    slug, _recon_id, _views, _canon = _turntable_profile("Resolve Five")
    refs, err = vr._reference_images_from_body(
        {"identity_profile": slug, "identity_view": "sideways"})
    assert refs is None and err is not None and err[1] == 400, err
    # a malformed object hint is also a clean 400 (never a crash).
    refs, err = vr._reference_images_from_body(
        {"identity_profile": slug, "identity_view": {"elevation_deg": 10}})
    assert refs is None and err is not None and err[1] == 400, err


# --------------------------------------------------------------------------- #
# [6] resolver — a versionless / no-turntable profile with a hint DEGRADES to canonical
# --------------------------------------------------------------------------- #
def test_reference_images_from_body_no_bank_degrades():
    p = _fresh_profile("Resolve Six")  # no reconstruction, no version, no ring
    slug = p["slug"]
    owned = set(identity_profiles.get_profile(slug)["reference_images"])

    # a hint on an identity WITHOUT a ring must not crash and must yield today's set.
    hinted, err = vr._reference_images_from_body(
        {"identity_profile": slug, "identity_view": "back"})
    assert err is None, err
    assert set(hinted) == owned, (hinted, owned)
    # byte-identical to the hintless fallback.
    no_hint, err2 = vr._reference_images_from_body({"identity_profile": slug})
    assert err2 is None and set(no_hint) == set(hinted)


# --------------------------------------------------------------------------- #
# [7] GET route — per-version ``views`` summary (additive)
# --------------------------------------------------------------------------- #
def test_get_profile_surfaces_views_summary():
    slug, _recon_id, _views, _canon = _turntable_profile("Summary Seven")
    r = client.get(f"/video/identity-profiles/{slug}")
    assert r.status_code == 200, (r.status_code, r.get_json())
    prof = r.get_json()["profile"]
    versions = prof["versions"]
    assert versions, prof
    summary = versions[0]["views"]
    assert summary["count"] == 72
    assert summary["degrees_per_frame"] == 5.0
    assert summary["frame_count"] == 72
    assert summary["azimuth_min"] == 0.0 and summary["azimuth_max"] == 355.0
    assert summary["source"] == "turntable"
    # additive: the pre-existing version contract keys are still present.
    for key in ("version_id", "name", "kind", "recon_id", "created_at", "canonical", "notes"):
        assert key in versions[0], key


# --------------------------------------------------------------------------- #
# [8] CANONICAL 8-VIEW SELECTION (operator 2026-07-16: "45 degree shots").
#     canonical_frame_indices selects the ring frames NEAREST each SEMANTIC_VIEWS
#     azimuth — a SELECTION over already-rendered frames, never a re-render.
# --------------------------------------------------------------------------- #
def test_canonical_frame_indices_hits_every_45_degree_view():
    # The live ring shape (verified against /video/identity-profiles 2026-07-16: every
    # turntable record is 72 frames @ 5.0°/frame). Each 45° target lands EXACTLY on a
    # rendered frame -> indices 0,9,18,27,36,45,54,63.
    idx = identity_profiles.canonical_frame_indices(72, 5.0)
    assert idx == [0, 9, 18, 27, 36, 45, 54, 63], idx
    assert len(idx) == identity_profiles.MAX_CANONICAL_IMAGES == 8, idx
    # Selected BY DEGREES: the chosen frames' azimuths are exactly the semantic targets.
    azimuths = sorted((i * 5.0) % 360.0 for i in idx)
    assert azimuths == sorted(identity_profiles.SEMANTIC_VIEWS.values()), azimuths
    # Ascending ring order (canonical is stored front -> 45° -> 90° -> …).
    assert idx == sorted(idx), idx


def test_canonical_frame_indices_degrades_on_sparse_rings_without_duplicates():
    # A ring coarser than 45° cannot serve 8 distinct angles. It must yield FEWER deduped
    # views (honest degrade), never duplicate frames padded out to 8.
    for n, dpf in ((4, 90.0), (3, 120.0), (5, 72.0), (1, 360.0)):
        idx = identity_profiles.canonical_frame_indices(n, dpf)
        assert len(idx) == len(set(idx)), (n, idx)      # no duplicates
        assert len(idx) <= min(8, n), (n, idx)
        assert all(0 <= i < n for i in idx), (n, idx)
    # An 8-frame @45° ring maps one target per frame, exactly.
    assert identity_profiles.canonical_frame_indices(8, 45.0) == list(range(8))
    # degrees_per_frame missing (older record) -> derived as 360/N, same as bank_views.
    assert identity_profiles.canonical_frame_indices(72, None) == [0, 9, 18, 27, 36, 45, 54, 63]
    assert identity_profiles.canonical_frame_indices(0, 5.0) == []


def test_canonical_selection_is_selection_not_rerender():
    """The 8 promoted canonical paths must be frames the ring ALREADY rendered."""
    slug, recon_id, views, _canon = _turntable_profile("Eight Select")
    idx = identity_profiles.canonical_frame_indices(len(views), 5.0)
    promoted = identity_profiles.promote_reconstruction_views(slug, recon_id, idx)
    canon = promoted["canonical"]
    assert len(canon) == 8, canon
    # Every canonical file exists, and is a COPY of the corresponding ring frame (the
    # promote path renumbers to ref_NN under <slug>/canonical/ — identity-owned bytes).
    assert all(os.path.isfile(p) for p in canon), canon
    assert len(set(canon)) == 8, canon              # 8 distinct files, no duplicates
    assert all(os.path.basename(p).startswith("ref_") for p in canon), canon


# --------------------------------------------------------------------------- #
# [9] BACK-COMPAT + the CANONICAL(8) vs RENDER(4) seam. An 8-view identity must
#     still drive a render channel that hard-rejects >4, and a 4-view profile
#     (every profile on disk today) must resolve EXACTLY as before.
# --------------------------------------------------------------------------- #
def test_render_refs_from_canonical_strides_8_to_the_cardinals():
    eight = [f"ref_{i:02d}.png" for i in range(8)]   # 0/45/90/135/180/225/270/315°
    got = identity_profiles.render_refs_from_canonical(eight)
    # Ring-strided -> the 4 CARDINALS (0°/90°/180°/270°), not the lopsided first-4
    # half-turn (0/45/90/135 = front + right side only).
    assert got == ["ref_00.png", "ref_02.png", "ref_04.png", "ref_06.png"], got
    assert len(got) <= identity_profiles.MAX_RENDER_REFS == 4, got
    # A 4-view set (today's on-disk profiles) passes through UNCHANGED — zero regression.
    four = [f"ref_{i:02d}.png" for i in range(4)]
    assert identity_profiles.render_refs_from_canonical(four) == four
    # Shorter sets and empties are untouched; never raises on an over-long set.
    assert identity_profiles.render_refs_from_canonical(["a"]) == ["a"]
    assert identity_profiles.render_refs_from_canonical([]) == []


def test_resolver_narrows_an_8_view_canonical_to_the_render_cap():
    """An 8-view identity must NOT 400 against its own DNA: the resolver narrows to 4."""
    slug, recon_id, views, _canon = _turntable_profile("Eight Resolve")
    idx = identity_profiles.canonical_frame_indices(len(views), 5.0)
    promoted = identity_profiles.promote_reconstruction_views(slug, recon_id, idx)
    # The resolver prefers the ACTIVE VERSION's canonical over the profile-level set, so
    # mint a version carrying the promoted 8 (what the relay's auto-promote + mint does).
    identity_profiles.mint_version(slug, recon_id, "textured", promoted["canonical"])
    # The store still HOLDS 8 (the operator's ask) ...
    prof = identity_profiles.get_profile(slug)
    assert len(prof["canonical"]) == 8, prof["canonical"]
    # ... but the render-channel resolver hands back at most 4 (the VACE ref-latent cap).
    refs, err = vr._reference_images_from_body({"identity_profile": slug})
    assert err is None, err
    assert len(refs) == identity_profiles.MAX_RENDER_REFS == 4, refs
    assert len(set(refs)) == len(refs), refs         # distinct frames, angle-spread
    assert all(p in prof["canonical"] for p in refs), (refs, prof["canonical"])


def test_resolver_4_view_canonical_is_unchanged():
    """BACK-COMPAT: a stored 4-entry canonical (every profile on disk today) resolves to
    exactly those 4 paths, in order — the narrowing seam is a no-op below the cap."""
    slug, recon_id, views, _canon = _turntable_profile("Four Legacy")
    promoted = identity_profiles.promote_reconstruction_views(slug, recon_id, [0, 18, 36, 54])
    # The resolver prefers the ACTIVE VERSION's canonical, so put the 4-view set there —
    # this mirrors a profile stored before 2026-07-16 (4 views, cardinal angles).
    identity_profiles.mint_version(slug, recon_id, "textured", promoted["canonical"])
    prof = identity_profiles.get_profile(slug)
    assert len(prof["canonical"]) == 4, prof["canonical"]
    refs, err = vr._reference_images_from_body({"identity_profile": slug})
    assert err is None, err
    assert refs == prof["canonical"], (refs, prof["canonical"])


# --------------------------------------------------------------------------- #
# [10] ANGLE PROVENANCE (2026-07-16). Promoting used to DESTROY the angle: only the
#      destination paths were stored, so "ref_02" meant "the third view someone
#      promoted", never "180°". canonical_angles carries it, positionally aligned.
# --------------------------------------------------------------------------- #
def test_promote_persists_canonical_angles_aligned_with_paths():
    slug, recon_id, views, _canon = _turntable_profile("Angle Provenance")
    idx = identity_profiles.canonical_frame_indices(len(views), 5.0)
    promoted = identity_profiles.promote_reconstruction_views(slug, recon_id, idx)

    angles = promoted["canonical_angles"]
    # One angle per canonical path, POSITIONALLY aligned, and they ARE the 45° ring.
    assert len(angles) == len(promoted["canonical"]) == 8, (angles, promoted["canonical"])
    assert angles == [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0], angles
    # Degrees are the canonical truth: every angle matches a SEMANTIC_VIEWS target.
    assert set(angles) == set(identity_profiles.SEMANTIC_VIEWS.values()), angles
    # It survives a round-trip through the store's public shape (not just the return).
    prof = identity_profiles.get_profile(slug)
    assert prof["canonical_angles"] == angles, prof.get("canonical_angles")


def test_canonical_angles_absent_for_legacy_and_sheet_recons():
    """BACK-COMPAT + honesty: no angle is INVENTED where none is knowable."""
    # A profile that never promoted has no canonical and no angles key at all.
    fresh = _fresh_profile("No Angles")
    assert "canonical_angles" not in fresh, fresh.get("canonical_angles")
    # A SHEET recon carries no orbit -> no azimuth. Promoting from it must NOT write a
    # misleading all-null (or worse, all-0.0 == "front") angle key.
    slug = fresh["slug"]
    src = fresh["reference_images"][0]
    identity_profiles.attach_reconstruction(slug, "recon_sheet", [src], spec={"mode": "sheet"})
    promoted = identity_profiles.promote_reconstruction_views(slug, "recon_sheet", [0])
    assert promoted["canonical"], promoted["canonical"]
    assert "canonical_angles" not in promoted, promoted.get("canonical_angles")
    # The angle helper itself reports None (unknown), never a guessed 0.0.
    assert identity_profiles.canonical_angles_for_indices(
        {"mode": "sheet", "views": ["a"]}, [0]) == [None]


def test_public_version_shape_is_additive_only():
    """The version wire contract's 7 keys are untouched for everything on disk today."""
    seven = {"version_id", "name", "kind", "recon_id", "created_at", "canonical", "notes"}
    # No stored angles (every version minted before 2026-07-16) -> EXACTLY the 7 keys.
    legacy = identity_profiles._public_version(
        {"version_id": "v1", "name": "textured-01", "kind": "textured", "recon_id": "r",
         "created_at": 1.0, "canonical": ["a.png", "b.png", "c.png", "d.png"], "notes": ""})
    assert set(legacy.keys()) == seven, sorted(legacy.keys())
    # With angles -> the same 7 PLUS one additive key. Nothing renamed, nothing dropped.
    withang = identity_profiles._public_version(
        {"version_id": "v2", "name": "textured-02", "kind": "textured", "recon_id": "r2",
         "created_at": 2.0, "canonical": ["a.png", "b.png"], "notes": "",
         "canonical_angles": [0.0, 180.0]})
    assert set(withang.keys()) == seven | {"canonical_angles"}, sorted(withang.keys())
    assert withang["canonical_angles"] == [0.0, 180.0]
    # DRIFT GUARD: angles that don't describe this set are DROPPED, never mislabelled.
    drifted = identity_profiles._public_version(
        {"version_id": "v3", "name": "x", "kind": "clay", "recon_id": "r3",
         "created_at": 3.0, "canonical": ["a.png", "b.png"], "notes": "",
         "canonical_angles": [0.0, 45.0, 90.0]})
    assert "canonical_angles" not in drifted, drifted.get("canonical_angles")


def test_mint_version_carries_angles_and_defaults_to_none():
    """The ACTIVE version's DNA (what the resolver serves) knows its own angles; and a
    caller that passes no angles behaves exactly as before this key existed."""
    slug, recon_id, views, _canon = _turntable_profile("Mint Angles")
    idx = identity_profiles.canonical_frame_indices(len(views), 5.0)
    promoted = identity_profiles.promote_reconstruction_views(slug, recon_id, idx)
    minted = identity_profiles.mint_version(
        slug, recon_id, "textured", promoted["canonical"],
        canonical_angles=promoted["canonical_angles"])
    assert minted["canonical_angles"] == [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0], minted
    # A pre-existing caller (positional args only, no angles) -> no key, no crash.
    plain = identity_profiles.mint_version(slug, recon_id, "textured", promoted["canonical"])
    assert "canonical_angles" not in plain, plain.get("canonical_angles")


CHECKS = [
    ("bank_views computes azimuths from index (0/90/180/270/355), order + shape",
     test_bank_views_computes_azimuths_from_index),
    ("nearest_bank_views wrap-aware + angle-spread; distinct frames; clamps",
     test_nearest_bank_views_is_wrap_aware_and_spread),
    ("azimuth_for_view semantic map + {azimuth_deg} object + errors-as-data",
     test_azimuth_for_view_semantic_and_object),
    ("resolver: hint -> ring frames; NO hint -> exactly the canonical set",
     test_reference_images_from_body_view_hint_vs_canonical),
    ("resolver: invalid identity_view -> clean 400 tuple",
     test_reference_images_from_body_invalid_view_is_400),
    ("resolver: versionless / no-ring profile + hint degrades to canonical",
     test_reference_images_from_body_no_bank_degrades),
    ("GET /identity-profiles/<slug> surfaces per-version views summary (additive)",
     test_get_profile_surfaces_views_summary),
    ("canonical_frame_indices hits every 45° SEMANTIC_VIEWS azimuth on a 72@5° ring",
     test_canonical_frame_indices_hits_every_45_degree_view),
    ("canonical_frame_indices degrades on sparse rings (fewer views, no duplicates)",
     test_canonical_frame_indices_degrades_on_sparse_rings_without_duplicates),
    ("canonical promote of the 8 selected indices = already-rendered frames, no re-render",
     test_canonical_selection_is_selection_not_rerender),
    ("render_refs_from_canonical strides 8 -> the 4 cardinals; <=4 passes through",
     test_render_refs_from_canonical_strides_8_to_the_cardinals),
    ("resolver narrows an 8-view canonical to the render cap (no 400 vs own DNA)",
     test_resolver_narrows_an_8_view_canonical_to_the_render_cap),
    ("BACK-COMPAT: a stored 4-view canonical resolves unchanged",
     test_resolver_4_view_canonical_is_unchanged),
    ("promote persists canonical_angles, positionally aligned with the paths",
     test_promote_persists_canonical_angles_aligned_with_paths),
    ("no angle is invented: absent for legacy profiles and sheet recons",
     test_canonical_angles_absent_for_legacy_and_sheet_recons),
    ("_public_version stays additive: 7 keys without angles, 8 with; drift drops",
     test_public_version_shape_is_additive_only),
    ("mint_version carries angles; a pre-existing caller is unaffected",
     test_mint_version_carries_angles_and_defaults_to_none),
]


def main() -> int:
    passed = failed = 0
    for i, (name, fn) in enumerate(CHECKS, 1):
        try:
            fn()
        except Exception as exc:  # surface EVERY divergence, not just the first
            failed += 1
            print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"[{i}] PASS  {name}")
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
