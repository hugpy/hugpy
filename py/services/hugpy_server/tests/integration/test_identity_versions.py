"""IDENTITY VERSIONS slice (backend, IDENTITY-VERSIONS-SLICE.md build order 1-3).

Exercises the version layer that rides on top of the identity-profiles store +
its studio resolver + the one-click /generate route, WITHOUT touching the real
registry (same isolation idiom as test_identity_profiles.py: rebind the store
module globals to temp dirs, jail reference images under the real UPLOADS_HOME,
point the media bus at a temp DB).

Locks (each check runs independently so one failure never masks the rest):
  [store/mint]
   1. mint_version: the FIRST clay is the pinned ``base``; a textured run mints
      ``textured-NN``; every mint becomes ACTIVE (latest-wins); a re-mint under the
      SAME recon_id updates in place (dedupe), never a second version.
   2. archive_version refuses the base + the active version; archives an eligible
      version (off the wire, bytes/flag kept — never-delete).
   3. set_gen_settings merges known keys, enum-checks pose, jails front_ref, rejects
      an unknown key; the wire always carries the full defaulted gen_settings shape.
  [migration]
   4. An identity that predates the slice (recon + canonical, no ``versions`` key) is
      backfilled on first load: newest recon -> clay ``base``; current canonical ->
      ``version-01`` (textured, ACTIVE). Idempotent — a second load never re-seeds.
   5. A never-generated identity is marked migrated with ``versions: []`` /
      ``active_version: None`` (so the backfill never re-runs for it).
  [resolver]
   6. _reference_images_from_body returns the ACTIVE version's canonical; an explicit
      ``identity_version`` (by name OR id) selects a specific version; an unknown
      version is a clean 404; a versionless profile degrades to today's behavior.
  [route: /generate pose]
   7. pose="none" (or absent) keeps the exact {job_id, recon_id} response; an invalid
      pose is a 400; pose="t-pose" while the stage is NOT capable falls back (200 +
      structured not-capable notice) and enqueues a spec with pose="none"; with the
      capability monkeypatched on, the spec carries pose="t-pose".

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python -m pytest tests/test_identity_versions.py -q
  venv/bin/python tests/test_identity_versions.py
"""
from __future__ import annotations

import atexit
import json
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
_TMP_IDENTITIES = tempfile.mkdtemp(prefix="hugpy-idver-store-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
identity_profiles.IDENTITIES_HOME = _TMP_IDENTITIES
_TMP_PROJECTS = tempfile.mkdtemp(prefix="hugpy-idver-projects-")
identity_profiles.PROJECTS_HOME = _TMP_PROJECTS

# JAIL: reference images must resolve under the real UPLOADS_HOME.
_TMP_UPLOADS = tempfile.mkdtemp(prefix="hugpy-idver-uploads-", dir=UPLOADS_HOME)

# media bus -> temp DB so /generate enqueues never touch the real catalog.
_TMP_DB = tempfile.mkstemp(prefix="idver-bus-", suffix=".db")[1]
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


def _make_png(path: str, color=(180, 90, 40)) -> None:
    from PIL import Image

    Image.new("RGB", (64, 64), color).save(path)


_IMG_A = os.path.join(_TMP_UPLOADS, "ver_a.png")
_IMG_B = os.path.join(_TMP_UPLOADS, "ver_b.png")
_make_png(_IMG_A, (200, 40, 40))
_make_png(_IMG_B, (40, 200, 40))


def _fresh_profile(name: str) -> dict:
    """Create a profile via the store (its refs copied into IDENTITIES_HOME) and return
    the public shape. Each test uses a unique name so the shared temp store never
    collides across checks."""
    return identity_profiles.create_profile(name, [_IMG_A, _IMG_B], notes="")


def _spec_json_for(job_id: str) -> dict:
    """The enqueued spec dict for a job, read straight from the temp bus DB."""
    with sqlite3.connect(_TMP_DB) as c:
        row = c.execute("SELECT spec_json FROM media_jobs WHERE job_id=?", (job_id,)).fetchone()
    assert row is not None, f"no job row for {job_id}"
    return json.loads(row[0])


# --------------------------------------------------------------------------- #
# [1] mint_version — base / textured-NN / active / dedupe
# --------------------------------------------------------------------------- #
def test_mint_version_base_then_textured_active_and_dedupe():
    p = _fresh_profile("Mint One")
    slug = p["slug"]
    assert p["versions"] == [] and p["active_version"] is None

    # First CLAY mint -> the pinned base, and it is ACTIVE.
    base = identity_profiles.mint_version(slug, "recon_clay", "clay", [])
    assert base["name"] == "base" and base["kind"] == "clay"
    prof = identity_profiles.get_profile(slug)
    assert prof["active_version"] == base["version_id"]
    assert [v["name"] for v in prof["versions"]] == ["base"]

    # A textured run -> textured-01, becomes ACTIVE, base preserved.
    tex = identity_profiles.mint_version(slug, "recon_tex1", "textured", [_IMG_A])
    assert tex["name"] == "textured-01" and tex["kind"] == "textured"
    prof = identity_profiles.get_profile(slug)
    assert prof["active_version"] == tex["version_id"]
    assert [v["name"] for v in prof["versions"]] == ["base", "textured-01"]

    # Re-mint under the SAME recon_id (a bus retry) UPDATES in place — no duplicate,
    # version_id preserved.
    again = identity_profiles.mint_version(slug, "recon_tex1", "textured", [_IMG_A, _IMG_B])
    assert again["version_id"] == tex["version_id"]
    prof = identity_profiles.get_profile(slug)
    assert [v["name"] for v in prof["versions"]] == ["base", "textured-01"]  # still 2
    tv = next(v for v in prof["versions"] if v["version_id"] == tex["version_id"])
    assert tv["canonical"] == [_IMG_A, _IMG_B]  # canonical updated by the re-mint


# --------------------------------------------------------------------------- #
# [2] archive_version — refuses base + active; archives an eligible version
# --------------------------------------------------------------------------- #
def test_archive_version_guards_and_never_delete():
    p = _fresh_profile("Arch Two")
    slug = p["slug"]
    base = identity_profiles.mint_version(slug, "r_base", "clay", [])
    v1 = identity_profiles.mint_version(slug, "r_t1", "textured", [_IMG_A])
    v2 = identity_profiles.mint_version(slug, "r_t2", "textured", [_IMG_B])  # v2 now active

    # base is never archivable.
    try:
        identity_profiles.archive_version(slug, base["version_id"])
        assert False, "expected ProfileError archiving the base"
    except identity_profiles.ProfileError:
        pass
    # the ACTIVE version (v2) is not archivable while active.
    try:
        identity_profiles.archive_version(slug, v2["version_id"])
        assert False, "expected ProfileError archiving the active version"
    except identity_profiles.ProfileError:
        pass

    # v1 (non-base, non-active) archives: off the wire, but bytes/flag kept in storage.
    out = identity_profiles.archive_version(slug, v1["version_id"])
    assert out is not None
    live = {v["version_id"] for v in out["versions"]}
    assert v1["version_id"] not in live               # dropped from the wire list
    assert {base["version_id"], v2["version_id"]} <= live
    raw = json.load(open(identity_profiles._store_path()))
    stored = raw["profiles"][slug]["versions"]
    archived = next(v for v in stored if v["version_id"] == v1["version_id"])
    assert archived.get("archived") is True           # flagged, not erased


# --------------------------------------------------------------------------- #
# [3] gen_settings — defaults on the wire; merge/validate; unknown key rejected
# --------------------------------------------------------------------------- #
def test_gen_settings_defaults_merge_and_validation():
    p = _fresh_profile("Settings Three")
    slug = p["slug"]
    # The full defaulted shape is ALWAYS on the wire (defaults-are-promises).
    gs = p["gen_settings"]
    assert gs["texture"] is True and gs["pose"] == "none" and gs["remove_background"] is True
    assert set(gs) == set(identity_profiles._DEFAULT_GEN_SETTINGS)

    out = identity_profiles.set_gen_settings(slug, {"texture": False, "pose": "t-pose"})
    assert out["gen_settings"]["texture"] is False
    assert out["gen_settings"]["pose"] == "t-pose"
    assert out["gen_settings"]["frame_count"] == 72  # untouched default still present

    # enum-checked pose, jailed front_ref, unknown key -> ProfileError (route 400).
    for bad in ({"pose": "crouch"}, {"front_ref": "/etc/passwd"}, {"bogus": 1}):
        try:
            identity_profiles.set_gen_settings(slug, bad)
            assert False, f"expected ProfileError for {bad}"
        except identity_profiles.ProfileError:
            pass

    # front_ref pointing at one of the profile's OWN owned refs is accepted.
    own = identity_profiles.get_profile(slug)["reference_images"][0]
    ok = identity_profiles.set_gen_settings(slug, {"front_ref": own})
    assert ok["gen_settings"]["front_ref"] == own


# --------------------------------------------------------------------------- #
# [4] MIGRATION — pre-versions identity backfilled on first load; idempotent
# --------------------------------------------------------------------------- #
def test_migration_backfills_base_and_version01_idempotent():
    slug = "legacy-luigi"
    entry = {
        "name": "Legacy Luigi",
        "source_images": [_IMG_A],
        "created_at": 100.0,
        "notes": "",
        "reconstructions": [
            {"recon_id": "recon_old", "created_at": 50.0, "views": [_IMG_A]},
            {"recon_id": "recon_new", "created_at": 90.0, "views": [_IMG_A, _IMG_B]},
        ],
        "canonical": [_IMG_A, _IMG_B],
        # NOTE: deliberately NO "versions" key — this is the pre-slice shape.
    }
    # Write the store file directly (bypassing create_profile) to simulate an identity
    # persisted before the versions slice existed.
    identity_profiles._save({"profiles": {slug: entry}, "_deleted": {}})

    prof = identity_profiles.get_profile(slug)  # first load -> backfill
    names = [v["name"] for v in prof["versions"]]
    assert names == ["base", "version-01"], names
    base = next(v for v in prof["versions"] if v["name"] == "base")
    v01 = next(v for v in prof["versions"] if v["name"] == "version-01")
    assert base["kind"] == "clay" and base["recon_id"] == "recon_new"  # NEWEST recon
    assert v01["kind"] == "textured" and v01["canonical"] == [_IMG_A, _IMG_B]
    assert prof["active_version"] == v01["version_id"]  # version-01 is ACTIVE

    # Persisted (the backfill saved) + base flagged in storage.
    raw = json.load(open(identity_profiles._store_path()))
    stored = raw["profiles"][slug]
    assert "versions" in stored and len(stored["versions"]) == 2
    stored_base = next(v for v in stored["versions"] if v["name"] == "base")
    assert stored_base.get("base") is True

    # Idempotent: a second load must NOT re-seed (same version_ids).
    ids_before = {v["version_id"] for v in stored["versions"]}
    prof2 = identity_profiles.get_profile(slug)
    ids_after = {v["version_id"] for v in prof2["versions"]}
    assert ids_before == ids_after, (ids_before, ids_after)


# --------------------------------------------------------------------------- #
# [5] MIGRATION — a never-generated identity is marked migrated (empty versions)
# --------------------------------------------------------------------------- #
def test_migration_never_generated_marks_empty():
    slug = "legacy-empty"
    entry = {"name": "Legacy Empty", "source_images": [_IMG_A], "created_at": 100.0,
             "notes": "", "reconstructions": [], "canonical": []}  # no versions key
    identity_profiles._save({"profiles": {slug: entry}, "_deleted": {}})

    prof = identity_profiles.get_profile(slug)
    assert prof["versions"] == [] and prof["active_version"] is None
    raw = json.load(open(identity_profiles._store_path()))
    assert "versions" in raw["profiles"][slug]  # marked migrated so it never re-runs


# --------------------------------------------------------------------------- #
# [6] RESOLVER — version-aware DNA + identity_version param + fallback
# --------------------------------------------------------------------------- #
def test_resolver_version_aware_and_identity_version_param():
    p = _fresh_profile("Resolve Six")
    slug = p["slug"]
    owned = identity_profiles.get_profile(slug)["reference_images"]  # real on-disk paths
    a, b = owned[0], owned[1]

    # Two versions with DISTINCT canonical sets; v_b is minted last -> ACTIVE.
    v_a = identity_profiles.mint_version(slug, "r_a", "textured", [a])
    v_b = identity_profiles.mint_version(slug, "r_b", "textured", [b])

    # Default resolve -> ACTIVE version's canonical (v_b -> [b]).
    refs, err = vr._reference_images_from_body({"identity_profile": slug})
    assert err is None and refs == [b], (refs, err)

    # identity_version by NAME selects a specific version.
    refs, err = vr._reference_images_from_body(
        {"identity_profile": slug, "identity_version": v_a["name"]})
    assert err is None and refs == [a], (refs, err)

    # identity_version by ID also works.
    refs, err = vr._reference_images_from_body(
        {"identity_profile": slug, "identity_version": v_b["version_id"]})
    assert err is None and refs == [b], (refs, err)

    # Unknown version -> clean 404 (never a silent wrong-identity).
    refs, err = vr._reference_images_from_body(
        {"identity_profile": slug, "identity_version": "textured-99"})
    assert refs is None and err is not None and err[1] == 404, err

    # Backward-compat: a versionless profile falls back to reference_images.
    p2 = _fresh_profile("Resolve Six B")
    refs, err = vr._reference_images_from_body({"identity_profile": p2["slug"]})
    assert err is None and set(refs) == set(p2["reference_images"]), (refs, err)


# --------------------------------------------------------------------------- #
# [7] ROUTE /generate — pose param validation + capability gate + spec plumbing
# --------------------------------------------------------------------------- #
def test_generate_pose_none_default_and_invalid():
    p = _fresh_profile("Pose Seven")
    slug = p["slug"]

    # bare click (no pose) -> today's exact shape, spec pose defaults "none".
    r = client.post(f"/video/identity-profiles/{slug}/generate", json={})
    assert r.status_code == 200, (r.status_code, r.get_json())
    j = r.get_json()
    assert set(j) == {"job_id", "recon_id"}, j
    assert _spec_json_for(j["job_id"]).get("pose", "none") == "none"

    # invalid pose -> 400.
    bad = client.post(f"/video/identity-profiles/{slug}/generate", json={"pose": "crouch"})
    assert bad.status_code == 400, (bad.status_code, bad.get_json())


def test_generate_pose_tpose_not_capable_falls_back():
    p = _fresh_profile("Pose Eight")
    slug = p["slug"]
    # The T-pose render stage is slice 5 — _pose_stage_capable is False today.
    assert vr._pose_stage_capable(slug) is False
    r = client.post(f"/video/identity-profiles/{slug}/generate", json={"pose": "t-pose"})
    assert r.status_code == 200, (r.status_code, r.get_json())
    j = r.get_json()
    assert j["pose"]["applied"] is False and j["pose"]["capable"] is False
    assert j["pose"]["code"] == "pose_stage_unavailable"
    # honest fallback: the enqueued spec meshes the NORMAL front (pose="none").
    assert _spec_json_for(j["job_id"]).get("pose", "none") == "none"


def test_generate_pose_tpose_capable_passes_through():
    p = _fresh_profile("Pose Nine")
    slug = p["slug"]
    orig = vr._pose_stage_capable
    vr._pose_stage_capable = lambda _slug: True  # slice-5 capable branch
    try:
        r = client.post(f"/video/identity-profiles/{slug}/generate", json={"pose": "t-pose"})
        assert r.status_code == 200, (r.status_code, r.get_json())
        j = r.get_json()
        assert j["pose"]["applied"] is True and j["pose"]["capable"] is True
        assert _spec_json_for(j["job_id"])["pose"] == "t-pose"  # spec carries it through
    finally:
        vr._pose_stage_capable = orig


CHECKS = [
    ("mint_version: first clay=base, textured-NN, active, dedupe by recon_id",
     test_mint_version_base_then_textured_active_and_dedupe),
    ("archive_version refuses base+active; archives eligible (flagged, kept)",
     test_archive_version_guards_and_never_delete),
    ("gen_settings full defaults on wire; merge/enum/jail/unknown-key",
     test_gen_settings_defaults_merge_and_validation),
    ("migration backfills base + version-01 from recon+canonical; idempotent",
     test_migration_backfills_base_and_version01_idempotent),
    ("migration marks a never-generated identity with empty versions",
     test_migration_never_generated_marks_empty),
    ("resolver returns ACTIVE version canonical; identity_version by name/id; 404; fallback",
     test_resolver_version_aware_and_identity_version_param),
    ("/generate pose=none keeps shape; invalid pose -> 400",
     test_generate_pose_none_default_and_invalid),
    ("/generate pose=t-pose not-capable -> 200 fallback + structured notice; spec pose=none",
     test_generate_pose_tpose_not_capable_falls_back),
    ("/generate pose=t-pose capable -> spec carries pose=t-pose",
     test_generate_pose_tpose_capable_passes_through),
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
