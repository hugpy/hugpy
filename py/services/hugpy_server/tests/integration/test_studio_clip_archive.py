"""Studio clip ARCHIVE / UNARCHIVE — the fix for "removed clips just reappear".

Root cause (verified before this slice): GET /video/studio/clips is DB-driven (a
media_jobs SELECT filtered on name='studio_i2v', not a filesystem walk — see that
route's header note). It had ONLY read routes, so any UI "remove" could only ever
mutate CLIENT state; the list's ~6s poll (studioShared.tsx useStudioClips) put the
clip straight back the next tick because the server never learned it had been
"removed".

This suite covers the server-side fix, POST .../archive + .../unarchive:

  1. archive() marks the bus row (archived_at) — it does NOT touch the row's clip
     bytes on disk (never-delete doctrine holds trivially: there is no delete path
     to guard against, only a DB-column mark the list query excludes).
  2. GET /video/studio/clips excludes an archived clip but keeps listing an
     unarchived control clip (proves the WHERE clause, not just "list went empty").
  3. GET /video/studio/clip/<id> and .../detail answer an archived clip with an
     HONEST 410 naming "archived" — not a bare 404 that reads as "never existed".
  4. unarchive() reverses it: the clip rejoins the list, the serve/detail routes
     serve it again, the bytes were never moved so there is nothing to "restore".
  5. Both verbs are IDEMPOTENT (house choice, stated here rather than 409): a
     retried/double-clicked archive must read as "already gone", not a failure —
     the archived_at timestamp is never bumped by a second archive call.
  6. archive/unarchive are scoped to name='studio_i2v' — a job id that exists
     under a different job name answers 404 (this bus carries every job kind;
     archive is a studio-clips-library concept only).

Script style matches the sibling suites (plain python, numbered PASS/FAIL,
nonzero exit iff any FAILED). pytest is NOT installed in this venv.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_clip_archive.py
"""
from __future__ import annotations

import atexit
import dataclasses
import importlib
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import time

logging.disable(logging.INFO)
os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")
os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="studio-archive-test-"))

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from flask import Flask  # noqa: E402

from hugpy_video.intel import media_bus
from hugpy_video.intel.studio.env import StudioEnv
from hugpy_video.intel.studio.job import DEFAULT_CLIPS_ROOT, make_studio_i2v
from hugpy_video.intel.studio.produce import produce_clip
from hugpy_video.intel.studio.enums import Capability
from hugpy_video.intel.studio.schemas import CapabilityRequest, Resolution

_FFMPEG = shutil.which("ffmpeg") is not None

# Isolate the media bus DB so we own the rows the archive routes read/write. Unlike
# test_studio_clip_serve.py's manual CREATE TABLE, this suite goes through
# media_bus._ensure_db() itself (the normal init path) so the archived_at migration
# runs exactly as it would in production — proving the migration, not routing
# around it.
_TMP_DB = tempfile.mkstemp(prefix="studio-archive-bus-", suffix=".db")[1]
media_bus.DB_PATH = _TMP_DB
media_bus._initialized = False
media_bus._ensure_db()

os.makedirs(DEFAULT_CLIPS_ROOT, exist_ok=True)
_WORK = tempfile.mkdtemp(prefix="archive-", dir=DEFAULT_CLIPS_ROOT)  # inside the jail


@atexit.register
def _cleanup():
    shutil.rmtree(_WORK, ignore_errors=True)
    try:
        os.remove(_TMP_DB)
    except OSError:
        pass


vr = importlib.import_module("hugpy_server.app.routes.video_routes")
app = Flask(__name__)
app.register_blueprint(vr.video_bp)
client = app.test_client()


def _env() -> StudioEnv:
    return StudioEnv(
        output_root=_WORK, weights_root="/w", manifest_root="/m",
        master_colorspace="rec709", master_fps=12, max_vram_gb=24.0,
        loudness_target_lufs=-14.0, allow_unpinned=True)


def _insert_job(job_id, name, status, result, spec):
    now = time.time()
    with sqlite3.connect(_TMP_DB) as c:
        c.execute(
            "INSERT OR REPLACE INTO media_jobs (job_id, name, status, spec_json, "
            "result_json, claim_token, created, updated, progress_json, archived_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (job_id, name, status,
             json.dumps(spec) if spec is not None else None,
             json.dumps(result) if result is not None else None,
             None, now, now, None, None))


# Two real synthetic clips (clip.mp4 under the jail) — ARCHIVE_ID gets archived by
# the checks below; CONTROL_ID stays untouched throughout, proving the list filter
# excludes the archived clip specifically rather than emptying out entirely.
ARCHIVE_ID = "archive-me"
CONTROL_ID = "leave-me-listed"
_ARCHIVE_URI = None

if _FFMPEG:
    req = CapabilityRequest(capability=Capability.I2V,
                            target_resolution=Resolution(320, 180, 12),
                            vram_budget_gb=0.5)
    _spec = dataclasses.asdict(make_studio_i2v(
        capability="i2v", width=320, height=180, fps=12, vram_budget_gb=0.5, seed=0,
        out_root=_WORK, prompt="synthetic archive-test clip"))

    def _ok_result(uri):
        out = {"asset_id": "asset-archive", "uri": uri, "kind": "video",
               "mime": "video/mp4", "width": 320, "height": 180, "duration_s": 2.0}
        return {"job_id": "x", "ok": True, "outputs": [out], "error": None}

    _res_a = produce_clip(req, env=_env(), out_root=_WORK)
    assert _res_a.is_ok(), _res_a
    _ARCHIVE_URI = _res_a.unwrap().path
    _insert_job(ARCHIVE_ID, "studio_i2v", "done", _ok_result(_ARCHIVE_URI), _spec)

    _res_b = produce_clip(req, env=_env(), out_root=_WORK)
    assert _res_b.is_ok(), _res_b
    _insert_job(CONTROL_ID, "studio_i2v", "done",
                _ok_result(_res_b.unwrap().path), _spec)
else:
    # No ffmpeg — insert rows carrying a synthetic (non-existent) uri so the
    # archive/list/idempotency checks (which don't touch bytes) still run; the
    # serve-route 410-vs-200 body checks below degrade gracefully (see each check).
    _spec = dataclasses.asdict(make_studio_i2v(
        capability="i2v", width=320, height=180, fps=12, vram_budget_gb=0.5, seed=0,
        out_root=_WORK, prompt="synthetic archive-test clip (no ffmpeg)"))
    _fake_uri = os.path.join(_WORK, "no-ffmpeg-stub", "clip.mp4")

    def _ok_result(uri):
        out = {"asset_id": "asset-archive", "uri": uri, "kind": "video",
               "mime": "video/mp4", "width": 320, "height": 180, "duration_s": 2.0}
        return {"job_id": "x", "ok": True, "outputs": [out], "error": None}

    _insert_job(ARCHIVE_ID, "studio_i2v", "done", _ok_result(_fake_uri), _spec)
    _insert_job(CONTROL_ID, "studio_i2v", "done", _ok_result(_fake_uri), _spec)

# A job that exists in the bus but under a DIFFERENT name — archive/unarchive must
# scope to name='studio_i2v' and refuse this one as "not found" (404), same as a
# wholly unknown id.
_insert_job("not-a-studio-job", "crop", "done", {"job_id": "x", "ok": True,
            "outputs": [], "error": None}, {})


def _list_ids():
    r = client.get("/video/studio/clips")
    assert r.status_code == 200, r.status_code
    return {c["job_id"] for c in r.get_json()["clips"]}


# --------------------------------------------------------------------------- #
# [1] Archiving an unknown job id -> 404, ok:False.
# --------------------------------------------------------------------------- #
def test_archive_unknown_404():
    r = client.post("/video/studio/clip/does-not-exist/archive")
    assert r.status_code == 404, (r.status_code, r.get_json())
    assert r.get_json()["ok"] is False


# --------------------------------------------------------------------------- #
# [2] Archiving a job under a NON-studio_i2v name is refused (scope), also 404.
# --------------------------------------------------------------------------- #
def test_archive_wrong_job_kind_404():
    r = client.post("/video/studio/clip/not-a-studio-job/archive")
    assert r.status_code == 404, (r.status_code, r.get_json())


# --------------------------------------------------------------------------- #
# [3] Both clips list before archiving (baseline).
# --------------------------------------------------------------------------- #
def test_both_listed_before_archive():
    ids = _list_ids()
    assert ARCHIVE_ID in ids, ids
    assert CONTROL_ID in ids, ids


# --------------------------------------------------------------------------- #
# [4] Archive the target clip -> 200, archived:True, already:False, a real
#     archived_at timestamp.
# --------------------------------------------------------------------------- #
_first_archived_at = None


def test_archive_ok():
    global _first_archived_at
    r = client.post(f"/video/studio/clip/{ARCHIVE_ID}/archive")
    assert r.status_code == 200, (r.status_code, r.get_json())
    body = r.get_json()
    assert body["ok"] is True, body
    assert body["archived"] is True, body
    assert body["already"] is False, body
    assert isinstance(body["archived_at"], (int, float)), body
    _first_archived_at = body["archived_at"]


# --------------------------------------------------------------------------- #
# [5] THE FIX: the archived clip drops out of the list; the control clip (never
#     archived) stays listed — proves the WHERE clause targets the right row.
# --------------------------------------------------------------------------- #
def test_archived_clip_excluded_control_stays():
    ids = _list_ids()
    assert ARCHIVE_ID not in ids, ids
    assert CONTROL_ID in ids, ids


# --------------------------------------------------------------------------- #
# [6] Never-delete: the clip's bytes are UNTOUCHED at their ORIGINAL path (this
#     catalog is DB-driven, so archiving is a column flip, not a filesystem move —
#     see media_bus.archive's docstring for why that satisfies the doctrine).
# --------------------------------------------------------------------------- #
def test_bytes_survive_at_original_path():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping byte-survival check)")
        return
    assert os.path.isfile(_ARCHIVE_URI), _ARCHIVE_URI
    assert os.path.getsize(_ARCHIVE_URI) > 0, _ARCHIVE_URI


# --------------------------------------------------------------------------- #
# [7] GET the archived clip by id -> honest 410 naming "archived" (not 404).
# --------------------------------------------------------------------------- #
def test_serve_archived_clip_410():
    r = client.get(f"/video/studio/clip/{ARCHIVE_ID}")
    assert r.status_code == 410, (r.status_code, r.data)
    body = r.get_json()
    assert body["archived"] is True, body


# --------------------------------------------------------------------------- #
# [8] DETAIL of the archived clip -> the same honest 410.
# --------------------------------------------------------------------------- #
def test_detail_archived_clip_410():
    r = client.get(f"/video/studio/clip/{ARCHIVE_ID}/detail")
    assert r.status_code == 410, (r.status_code, r.get_json())
    assert r.get_json()["archived"] is True


# --------------------------------------------------------------------------- #
# [9] Double-archive is a clean NO-OP (house choice: idempotent 200, not 409) —
#     already:True, and archived_at is NOT bumped by the second call.
# --------------------------------------------------------------------------- #
def test_double_archive_is_idempotent_noop():
    r = client.post(f"/video/studio/clip/{ARCHIVE_ID}/archive")
    assert r.status_code == 200, (r.status_code, r.get_json())
    body = r.get_json()
    assert body["ok"] is True, body
    assert body["already"] is True, body
    assert body["archived_at"] == _first_archived_at, (body["archived_at"], _first_archived_at)


# --------------------------------------------------------------------------- #
# [10] Unarchiving an unknown / wrong-kind id also answers 404 (mirrors archive).
# --------------------------------------------------------------------------- #
def test_unarchive_unknown_404():
    r = client.post("/video/studio/clip/does-not-exist/unarchive")
    assert r.status_code == 404, (r.status_code, r.get_json())


# --------------------------------------------------------------------------- #
# [11] Unarchive restores the clip: 200, archived:False; it rejoins the list;
#      the serve route answers 200 again (bytes were never moved).
# --------------------------------------------------------------------------- #
def test_unarchive_restores():
    r = client.post(f"/video/studio/clip/{ARCHIVE_ID}/unarchive")
    assert r.status_code == 200, (r.status_code, r.get_json())
    body = r.get_json()
    assert body["ok"] is True, body
    assert body["archived"] is False, body
    assert body["already"] is False, body

    ids = _list_ids()
    assert ARCHIVE_ID in ids, ids

    if _FFMPEG:
        r2 = client.get(f"/video/studio/clip/{ARCHIVE_ID}")
        assert r2.status_code == 200, (r2.status_code, r2.data)


# --------------------------------------------------------------------------- #
# [12] Double-unarchive is also an idempotent no-op (already:True).
# --------------------------------------------------------------------------- #
def test_double_unarchive_is_idempotent_noop():
    r = client.post(f"/video/studio/clip/{ARCHIVE_ID}/unarchive")
    assert r.status_code == 200, (r.status_code, r.get_json())
    body = r.get_json()
    assert body["ok"] is True, body
    assert body["already"] is True, body


CHECKS = [
    ("archive an unknown job id -> 404", test_archive_unknown_404),
    ("archive a non-studio_i2v job id -> 404 (scope)", test_archive_wrong_job_kind_404),
    ("both clips list before archiving (baseline)", test_both_listed_before_archive),
    ("archive the target clip -> 200, archived_at minted", test_archive_ok),
    ("archived clip excluded from list; control clip stays listed",
     test_archived_clip_excluded_control_stays),
    ("never-delete: clip bytes survive at their original path",
     test_bytes_survive_at_original_path),
    ("GET an archived clip -> honest 410 (not 404)", test_serve_archived_clip_410),
    ("DETAIL of an archived clip -> honest 410", test_detail_archived_clip_410),
    ("double-archive is an idempotent no-op (house choice, not 409)",
     test_double_archive_is_idempotent_noop),
    ("unarchive an unknown job id -> 404", test_unarchive_unknown_404),
    ("unarchive restores: relists + re-servable, bytes untouched",
     test_unarchive_restores),
    ("double-unarchive is also an idempotent no-op", test_double_unarchive_is_idempotent_noop),
]


def main() -> int:
    passed = failed = 0
    for i, (name, fn) in enumerate(CHECKS, 1):
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"[{i}] PASS  {name}")
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
