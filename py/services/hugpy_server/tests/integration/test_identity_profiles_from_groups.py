"""CHARACTER-GROUPS-PLAN S3 — POST /video/identity-profiles/from-groups.

Commits curated char360 groups (S1 REVIEW manifest, edited client-side by S2)
into identity profiles: ONE profile per submitted group, through the SAME
validation + copy path the single-profile create route uses
(_validate_profile_reference_images -> identity_profiles.create_profile).

Isolation mirrors tests/test_identity_profiles.py exactly: IDENTITIES_HOME +
PROJECTS_HOME rebound to temp dirs (direct module rebind — env isolation does
not work here since constants' get_env_value reads the .env file, not
os.environ), reference images staged under a temp subdir of the REAL
UPLOADS_HOME so the storage jail + media_store.ingest classify them honestly.

Locks:
  [1] two groups -> two profiles created with the derived/explicit names +
      slugs returned, references landed (byte-identical copies) in each
      profile's own dir.
  [2] a group with an invalid reference (jail-escape) records ok:false + error
      for THAT group only — the other group in the same batch still commits.
  [3] a missing/blank name derives "Character N" (1-based index).
  [4] groups missing/empty -> clean 400, no profile created.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python -m pytest tests/test_identity_profiles_from_groups.py -q
"""
from __future__ import annotations

import atexit
import importlib
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

import pytest  # noqa: E402
from flask import Flask  # noqa: E402

from hugpy_video.intel import identity_profiles
from hugpy_video.intel import media_bus
from hugpy_platform.constants import UPLOADS_HOME, DEFAULT_ROOT

# STORE isolation — same lever test_identity_profiles.py uses: rebind the module
# globals the store's path helpers read so the whole registry lands in temp dirs.
_TMP_IDENTITIES = tempfile.mkdtemp(prefix="hugpy-fromgroups-identity-store-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
identity_profiles.IDENTITIES_HOME = _TMP_IDENTITIES
_TMP_PROJECTS = tempfile.mkdtemp(prefix="hugpy-fromgroups-identity-projects-")
identity_profiles.PROJECTS_HOME = _TMP_PROJECTS

# JAIL: reference images must resolve under the REAL UPLOADS_HOME.
_TMP_UPLOADS = tempfile.mkdtemp(prefix="hugpy-fromgroups-uploads-", dir=UPLOADS_HOME)

# media_bus isn't hit by this route, but importing video_routes wires it up —
# point it at a temp DB so nothing touches the real dev catalog (mirrors
# test_identity_profiles.py's isolation).
_TMP_DB = tempfile.mkstemp(prefix="fromgroups-bus-", suffix=".db")[1]
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


def _same_bytes(a: str, b: str) -> bool:
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


_CHAR_A0 = os.path.join(_TMP_UPLOADS, "char_a_00.png")
_CHAR_A1 = os.path.join(_TMP_UPLOADS, "char_a_01.png")
_CHAR_B0 = os.path.join(_TMP_UPLOADS, "char_b_00.png")
_make_png(_CHAR_A0, (200, 40, 40))
_make_png(_CHAR_A1, (190, 50, 50))
_make_png(_CHAR_B0, (40, 200, 40))


def _owned(slug: str, index: int, src: str) -> str:
    ext = os.path.splitext(src)[1].lower() or ".img"
    return os.path.join(_TMP_IDENTITIES, slug, f"ref_{index:02d}{ext}")


# --------------------------------------------------------------------------- #
# CROSS-FILE ISOLATION — this file and test_identity_video_extract_relay.py EACH rebind
# identity_profiles.IDENTITIES_HOME/PROJECTS_HOME + media_bus.DB_PATH to their OWN
# tempfile.mkdtemp store in their import preamble (above). Under pytest both modules import
# at COLLECTION time, so whichever file imports last wins those shared globals and the
# other file's tests silently run against the WRONG temp store. Re-assert THIS file's
# bindings before every test so file import/collection order never matters.
#
# media_bus._initialized guards a lazy one-time schema migration keyed off whatever
# DB_PATH happened to be bound when it last ran (see media_bus._ensure_db). If the OTHER
# file's globals rebound DB_PATH to ITS db and that db already got migrated, _initialized
# is left True — so when we rebind DB_PATH back to OUR db here, an already-True
# _initialized would make _ensure_db() a no-op and skip OUR db's migration. Our db WAS
# already schema-created by this file's own preamble (CREATE TABLE up front), so nothing
# breaks today, but future ALTER-TABLE migrations added to _ensure_db would silently never
# reach it. Clear _initialized here too so _ensure_db() (called at the top of every
# media_bus function) always re-verifies/re-migrates the CURRENTLY bound db — the ALTER
# TABLE calls are idempotent (they swallow "duplicate column" errors), so re-running them
# against an already-migrated db is a harmless no-op.
@pytest.fixture(autouse=True)
def _rebind_isolation_globals():
    identity_profiles.IDENTITIES_HOME = _TMP_IDENTITIES
    identity_profiles.PROJECTS_HOME = _TMP_PROJECTS
    media_bus.DB_PATH = _TMP_DB
    media_bus._initialized = False
    yield


# --------------------------------------------------------------------------- #
# [1] two groups -> two profiles, references landed in each profile's own dir.
# --------------------------------------------------------------------------- #
def test_two_groups_create_two_profiles():
    r = client.post(
        "/video/identity-profiles/from-groups",
        json={
            "groups": [
                {"name": "Hero", "reference_images": [_CHAR_A0, _CHAR_A1]},
                {"name": "Sidekick", "reference_images": [_CHAR_B0]},
            ]
        },
    )
    assert r.status_code == 200, (r.status_code, r.get_json())
    body = r.get_json()
    results = body["results"]
    assert len(results) == 2, results

    assert results[0]["ok"] is True, results[0]
    assert results[0]["name"] == "Hero", results[0]
    assert results[0]["slug"] == "hero", results[0]
    owned_a = [_owned("hero", 0, _CHAR_A0), _owned("hero", 1, _CHAR_A1)]
    for o, src in zip(owned_a, (_CHAR_A0, _CHAR_A1)):
        assert os.path.isfile(o) and _same_bytes(o, src), o

    assert results[1]["ok"] is True, results[1]
    assert results[1]["name"] == "Sidekick", results[1]
    assert results[1]["slug"] == "sidekick", results[1]
    owned_b = [_owned("sidekick", 0, _CHAR_B0)]
    assert os.path.isfile(owned_b[0]) and _same_bytes(owned_b[0], _CHAR_B0), owned_b

    # both durably readable via the normal profile GET
    hero = client.get("/video/identity-profiles/hero").get_json()["profile"]
    assert hero["reference_images"] == owned_a, hero
    kick = client.get("/video/identity-profiles/sidekick").get_json()["profile"]
    assert kick["reference_images"] == owned_b, kick


# --------------------------------------------------------------------------- #
# [2] one bad group (jail-escape ref) errors-as-data; sibling group still commits.
# --------------------------------------------------------------------------- #
def test_bad_group_does_not_abort_batch():
    r = client.post(
        "/video/identity-profiles/from-groups",
        json={
            "groups": [
                {"name": "Broken", "reference_images": ["/etc/passwd"]},
                {"name": "Survivor", "reference_images": [_CHAR_B0]},
            ]
        },
    )
    assert r.status_code == 200, (r.status_code, r.get_json())
    results = r.get_json()["results"]
    assert len(results) == 2, results

    assert results[0]["ok"] is False, results[0]
    assert results[0]["name"] == "Broken", results[0]
    assert "slug" not in results[0], results[0]
    assert "jail" in results[0]["error"], results[0]

    assert results[1]["ok"] is True, results[1]
    assert results[1]["slug"] == "survivor", results[1]
    assert client.get("/video/identity-profiles/survivor").status_code == 200


# --------------------------------------------------------------------------- #
# [3] missing/blank name -> derived "Character N" (1-based index).
# --------------------------------------------------------------------------- #
def test_missing_name_derives_character_n():
    r = client.post(
        "/video/identity-profiles/from-groups",
        json={
            "groups": [
                {"reference_images": [_CHAR_A0]},
                {"name": "   ", "reference_images": [_CHAR_B0]},
            ]
        },
    )
    assert r.status_code == 200, (r.status_code, r.get_json())
    results = r.get_json()["results"]
    assert results[0]["ok"] is True, results[0]
    assert results[0]["name"] == "Character 1", results[0]
    assert results[0]["slug"] == "character-1", results[0]
    assert results[1]["ok"] is True, results[1]
    assert results[1]["name"] == "Character 2", results[1]
    assert results[1]["slug"] == "character-2", results[1]


# --------------------------------------------------------------------------- #
# [4] groups missing/empty -> clean 400, nothing created.
# --------------------------------------------------------------------------- #
def test_groups_missing_or_empty_400():
    r = client.post("/video/identity-profiles/from-groups", json={})
    assert r.status_code == 400, (r.status_code, r.get_json())
    assert "error" in r.get_json(), r.get_json()

    r2 = client.post("/video/identity-profiles/from-groups", json={"groups": []})
    assert r2.status_code == 400, (r2.status_code, r2.get_json())


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
