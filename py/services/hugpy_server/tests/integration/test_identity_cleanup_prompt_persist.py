"""IDENTITY CLEANUP-PROMPT (C1 persistence) — the per-profile ``cleanup_prompt`` /
``negative_prompt`` gen_settings (operator-requested 2026-07-15; profile-IS-identity: the
operator does not re-type the cleanup instruction on every generate).

Locks the store side WITHOUT a GPU / network:
  * set_gen_settings ACCEPTS both keys as plain strings and STORES them.
  * a non-string value is a ProfileError -> the route's 400 (via PATCH too).
  * None coerces to "" (a clear), never an error.
  * a BARE profile yields "" for BOTH on the wire (byte-identical to today —
    defaults-are-promises).

Isolation mirrors test_identity_vision_setting.py exactly (rebind the store module globals
to temp dirs — env isolation does NOT work since constants read the .env file — jail refs
under the real UPLOADS_HOME, point the media bus at a temp DB). Run ALONE — the identity
test family cross-pollutes via the import-time IDENTITIES_HOME rebind.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python -m pytest tests/test_identity_cleanup_prompt_persist.py -q
  venv/bin/python tests/test_identity_cleanup_prompt_persist.py
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

# --------------------------------------------------------------------------- #
# STORE + BUS isolation (mirrors test_identity_vision_setting.py exactly).
# --------------------------------------------------------------------------- #
_TMP_IDENTITIES = tempfile.mkdtemp(prefix="hugpy-cleanup-store-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
identity_profiles.IDENTITIES_HOME = _TMP_IDENTITIES
_TMP_PROJECTS = tempfile.mkdtemp(prefix="hugpy-cleanup-projects-")
identity_profiles.PROJECTS_HOME = _TMP_PROJECTS
_TMP_UPLOADS = tempfile.mkdtemp(prefix="hugpy-cleanup-uploads-", dir=UPLOADS_HOME)

_TMP_DB = tempfile.mkstemp(prefix="cleanup-bus-", suffix=".db")[1]
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


_IMG = os.path.join(_TMP_UPLOADS, "hero.png")
_make_png(_IMG, (200, 40, 40))


def _create_profile(name: str) -> str:
    r = client.post("/video/identity-profiles",
                    json={"name": name, "reference_images": [_IMG], "notes": "a knight"})
    assert r.status_code == 201, (r.status_code, r.get_json())
    return r.get_json()["profile"]["slug"]


# --------------------------------------------------------------------------- #
# STORE: set_gen_settings accepts + stores the two cleanup channels.
# --------------------------------------------------------------------------- #
def test_set_gen_settings_accepts_and_stores_both():
    slug = _create_profile("Cleanup Store")
    prof = identity_profiles.set_gen_settings(
        slug, {"cleanup_prompt": "no object on her back, clean bare back",
               "negative_prompt": "backpack, symbols"})
    assert prof is not None
    gs = prof["gen_settings"]
    assert gs["cleanup_prompt"] == "no object on her back, clean bare back", gs
    assert gs["negative_prompt"] == "backpack, symbols", gs


def test_set_gen_settings_none_clears_to_empty():
    slug = _create_profile("Cleanup Clear")
    identity_profiles.set_gen_settings(slug, {"cleanup_prompt": "x", "negative_prompt": "y"})
    prof = identity_profiles.set_gen_settings(
        slug, {"cleanup_prompt": None, "negative_prompt": None})
    gs = prof["gen_settings"]
    assert gs["cleanup_prompt"] == "" and gs["negative_prompt"] == "", gs


def test_set_gen_settings_rejects_non_string():
    slug = _create_profile("Cleanup Reject")
    for bad in ({"cleanup_prompt": 5}, {"negative_prompt": ["x"]}):
        try:
            identity_profiles.set_gen_settings(slug, bad)
        except identity_profiles.ProfileError as exc:
            assert exc.code == "invalid_profile", exc.code
        else:
            raise AssertionError(f"expected ProfileError for {bad!r}")
    # and the PATCH route surfaces it as a clean 400
    r = client.patch(f"/video/identity-profiles/{slug}/settings",
                     json={"cleanup_prompt": 5})
    assert r.status_code == 400, (r.status_code, r.get_json())


def test_bare_profile_defaults_empty_zero_regression():
    # A brand-new profile carries "" for BOTH on the wire, so a bare Generate is byte-
    # identical to before this setting existed (defaults-are-promises).
    slug = _create_profile("Cleanup Default")
    gs = client.get(f"/video/identity-profiles/{slug}").get_json()["profile"]["gen_settings"]
    assert "cleanup_prompt" in gs and "negative_prompt" in gs, gs
    assert gs["cleanup_prompt"] == "" and gs["negative_prompt"] == "", gs


def test_patch_route_stores_both():
    slug = _create_profile("Cleanup Patch")
    r = client.patch(f"/video/identity-profiles/{slug}/settings",
                     json={"cleanup_prompt": "no logos", "negative_prompt": "text"})
    assert r.status_code == 200, (r.status_code, r.get_json())
    gs = r.get_json()["profile"]["gen_settings"]
    assert gs["cleanup_prompt"] == "no logos" and gs["negative_prompt"] == "text", gs


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} CLEANUP-PROMPT PERSIST CHECKS PASSED")
