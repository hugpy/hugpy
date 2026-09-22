"""IDENTITY VERSIONS HTTP surface — the four routes the identity-versions UI calls
that were wired store-side (video_intel/identity_profiles.py) but never given a
Flask route (405 on the wire): PATCH settings, POST activate, PATCH/DELETE a
version. This file exercises routes/video_routes.py's translation of those four
verbs, WITHOUT touching the real registry (same isolation idiom as
test_identity_profiles.py / test_identity_versions.py: rebind the store module
globals to temp dirs, jail reference images under the real UPLOADS_HOME).

Locks (each check runs independently — a fresh, uniquely-named profile per test —
so one failure never masks the rest):
  [1] PATCH .../settings: a valid partial merges into gen_settings and returns
      {profile} with the full defaulted shape; an unknown key -> 400; a
      wrong-typed known key -> 400; unknown slug -> 404.
  [2] POST .../versions/<id>/activate: flips active_version to the named
      (currently-inactive) version; unknown slug -> 404; unknown version_id on a
      real slug -> 404.
  [3] PATCH .../versions/<id>: renames a version in place ({profile} response,
      the version's name updated); a blank name -> 400; unknown slug -> 404;
      unknown version_id -> 404.
  [4] DELETE .../versions/<id>: refuses the clay BASE (400) and the currently
      ACTIVE version (400); archives an eligible (non-base, non-active) version
      (200 -> it drops off the public versions list, never-delete); unknown
      slug -> 404; unknown version_id -> 404.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python -m pytest tests/test_identity_version_routes.py -q
  venv/bin/python tests/test_identity_version_routes.py
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

# STORE isolation — rebind the module globals the store's path helpers read (env
# isolation does NOT work here: constants' get_env_value reads the .env file, not
# os.environ). IDENTITIES_HOME must sit under the real DEFAULT_ROOT so the
# identity-owned ref copies still pass the route's storage jail.
_TMP_IDENTITIES = tempfile.mkdtemp(prefix="hugpy-idverroutes-store-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
identity_profiles.IDENTITIES_HOME = _TMP_IDENTITIES
_TMP_PROJECTS = tempfile.mkdtemp(prefix="hugpy-idverroutes-projects-")
identity_profiles.PROJECTS_HOME = _TMP_PROJECTS

# JAIL: reference images must resolve under the real UPLOADS_HOME.
_TMP_UPLOADS = tempfile.mkdtemp(prefix="hugpy-idverroutes-uploads-", dir=UPLOADS_HOME)

# media bus -> temp DB (unused by these 4 routes, but importing video_routes wires
# the blueprint, and sibling identity route tests always isolate the bus too).
_TMP_DB = tempfile.mkstemp(prefix="idverroutes-bus-", suffix=".db")[1]
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


_IMG_A = os.path.join(_TMP_UPLOADS, "vr_a.png")
_IMG_B = os.path.join(_TMP_UPLOADS, "vr_b.png")
_make_png(_IMG_A, (200, 40, 40))
_make_png(_IMG_B, (40, 200, 40))


def _fresh_profile(name: str) -> dict:
    """Create a profile via the store (its refs copied into IDENTITIES_HOME) and
    return the public shape. Each test uses a unique name so the shared temp store
    never collides across checks."""
    return identity_profiles.create_profile(name, [_IMG_A, _IMG_B], notes="")


# --------------------------------------------------------------------------- #
# [1] PATCH .../settings
# --------------------------------------------------------------------------- #
def test_settings_patch_merges_validates_and_404s():
    p = _fresh_profile("Settings One")
    slug = p["slug"]

    r = client.patch(
        f"/video/identity-profiles/{slug}/settings",
        json={"frame_count": 48, "pose": "t-pose"},
    )
    assert r.status_code == 200, (r.status_code, r.get_json())
    gs = r.get_json()["profile"]["gen_settings"]
    assert gs["frame_count"] == 48 and gs["pose"] == "t-pose", gs
    # untouched keys stay at their contract defaults (a true partial merge).
    assert gs["fps"] == 24 and gs["texture"] is True, gs

    # unknown key -> clean 400, never a 500.
    bad = client.patch(f"/video/identity-profiles/{slug}/settings", json={"bogus": 1})
    assert bad.status_code == 400, (bad.status_code, bad.get_json())
    assert "error" in bad.get_json(), bad.get_json()

    # wrong-typed known key -> clean 400.
    bad_type = client.patch(
        f"/video/identity-profiles/{slug}/settings", json={"frame_count": "many"}
    )
    assert bad_type.status_code == 400, (bad_type.status_code, bad_type.get_json())

    # unknown slug -> 404.
    r404 = client.patch("/video/identity-profiles/nobody/settings", json={"fps": 30})
    assert r404.status_code == 404, r404.status_code
    assert "error" in r404.get_json(), r404.get_json()


# --------------------------------------------------------------------------- #
# [2] POST .../versions/<id>/activate
# --------------------------------------------------------------------------- #
def test_activate_version_switches_active_and_404s():
    p = _fresh_profile("Activate One")
    slug = p["slug"]
    base = identity_profiles.mint_version(slug, "recon_clay_act", "clay", [])
    tex = identity_profiles.mint_version(slug, "recon_tex_act", "textured", [_IMG_A])
    # latest mint (tex) is active now — activating base should flip it back.
    prof = identity_profiles.get_profile(slug)
    assert prof["active_version"] == tex["version_id"], prof

    r = client.post(f"/video/identity-profiles/{slug}/versions/{base['version_id']}/activate")
    assert r.status_code == 200, (r.status_code, r.get_json())
    assert r.get_json()["profile"]["active_version"] == base["version_id"], r.get_json()

    # unknown slug -> 404.
    r404_slug = client.post(f"/video/identity-profiles/nobody/versions/{base['version_id']}/activate")
    assert r404_slug.status_code == 404, r404_slug.status_code

    # unknown version_id on a real slug -> 404.
    r404_ver = client.post(f"/video/identity-profiles/{slug}/versions/ver_nope/activate")
    assert r404_ver.status_code == 404, r404_ver.status_code
    assert "error" in r404_ver.get_json(), r404_ver.get_json()


# --------------------------------------------------------------------------- #
# [3] PATCH .../versions/<id>
# --------------------------------------------------------------------------- #
def test_rename_version_happy_validation_and_404s():
    p = _fresh_profile("Rename One")
    slug = p["slug"]
    base = identity_profiles.mint_version(slug, "recon_clay_ren", "clay", [])

    r = client.patch(
        f"/video/identity-profiles/{slug}/versions/{base['version_id']}",
        json={"name": "Custom Base Name"},
    )
    assert r.status_code == 200, (r.status_code, r.get_json())
    versions = r.get_json()["profile"]["versions"]
    v = next(x for x in versions if x["version_id"] == base["version_id"])
    assert v["name"] == "Custom Base Name", v

    # blank name -> clean 400.
    blank = client.patch(
        f"/video/identity-profiles/{slug}/versions/{base['version_id']}", json={"name": "  "}
    )
    assert blank.status_code == 400, (blank.status_code, blank.get_json())

    # unknown slug -> 404.
    r404_slug = client.patch(
        f"/video/identity-profiles/nobody/versions/{base['version_id']}", json={"name": "x"}
    )
    assert r404_slug.status_code == 404, r404_slug.status_code

    # unknown version_id -> 404.
    r404_ver = client.patch(
        f"/video/identity-profiles/{slug}/versions/ver_nope", json={"name": "x"}
    )
    assert r404_ver.status_code == 404, r404_ver.status_code


# --------------------------------------------------------------------------- #
# [4] DELETE .../versions/<id> — base/active refusal (400) + eligible archive (200)
# --------------------------------------------------------------------------- #
def test_archive_version_refusals_and_success_and_404s():
    p = _fresh_profile("Archive One")
    slug = p["slug"]
    base = identity_profiles.mint_version(slug, "recon_clay_arc", "clay", [])
    tex1 = identity_profiles.mint_version(slug, "recon_tex1_arc", "textured", [_IMG_A])
    tex2 = identity_profiles.mint_version(slug, "recon_tex2_arc", "textured", [_IMG_B])
    # tex2 is the latest mint -> currently ACTIVE; tex1 is neither base nor active.
    prof = identity_profiles.get_profile(slug)
    assert prof["active_version"] == tex2["version_id"], prof

    # refuse the clay BASE.
    r_base = client.delete(f"/video/identity-profiles/{slug}/versions/{base['version_id']}")
    assert r_base.status_code == 400, (r_base.status_code, r_base.get_json())
    assert "error" in r_base.get_json(), r_base.get_json()

    # refuse the currently ACTIVE version.
    r_active = client.delete(f"/video/identity-profiles/{slug}/versions/{tex2['version_id']}")
    assert r_active.status_code == 400, (r_active.status_code, r_active.get_json())

    # archive an eligible (non-base, non-active) version -> 200, dropped from the wire.
    r_ok = client.delete(f"/video/identity-profiles/{slug}/versions/{tex1['version_id']}")
    assert r_ok.status_code == 200, (r_ok.status_code, r_ok.get_json())
    remaining_ids = {v["version_id"] for v in r_ok.get_json()["profile"]["versions"]}
    assert tex1["version_id"] not in remaining_ids, remaining_ids
    assert base["version_id"] in remaining_ids and tex2["version_id"] in remaining_ids

    # unknown slug -> 404.
    r404_slug = client.delete(f"/video/identity-profiles/nobody/versions/{tex1['version_id']}")
    assert r404_slug.status_code == 404, r404_slug.status_code

    # unknown version_id -> 404.
    r404_ver = client.delete(f"/video/identity-profiles/{slug}/versions/ver_nope")
    assert r404_ver.status_code == 404, r404_ver.status_code


CHECKS = [
    ("PATCH settings merges/validates/404s", test_settings_patch_merges_validates_and_404s),
    ("POST activate switches active_version + 404s", test_activate_version_switches_active_and_404s),
    ("PATCH version renames + validates + 404s", test_rename_version_happy_validation_and_404s),
    ("DELETE version refuses base/active, archives eligible, 404s",
     test_archive_version_refusals_and_success_and_404s),
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
