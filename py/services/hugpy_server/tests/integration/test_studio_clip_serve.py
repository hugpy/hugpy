"""Studio clip SERVE mimetype + clip DETAIL endpoint (coordinator addenda, this slice).

Addendum 1 — GET /video/studio/clip/<id> and GET /video/media must send an EXPLICIT
Content-Type, never filename guessing: a content-addressed clip can be served from a
uri WITHOUT a .mp4 extension, and the guess then yields octet-stream -> the console
<video> shows a gray "unknown mime" error. The serve routes source the mime from the
catalog (outputs[].mime) and fall back to video/mp4 for anything under the studio
clips dir.

Addendum 2 — GET /video/studio/clip/<id>/detail returns the exact CREATION PARAMETERS:
the content-addressed manifest for a DONE clip (model_id, sampler steps/cfg/shift,
seed, resolution, prompt, content_hash) + the requested spec; the error + spec for a
FAILED job.

Script style matches the other studio suites (plain python, numbered PASS/FAIL,
nonzero exit iff any FAILED). pytest is NOT installed in this venv.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_clip_serve.py
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
os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="studio-serve-test-"))

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

# Isolate the media bus DB so we own the rows the serve/detail routes read.
_TMP_DB = tempfile.mkstemp(prefix="studio-serve-bus-", suffix=".db")[1]
media_bus.DB_PATH = _TMP_DB
media_bus._initialized = False
with sqlite3.connect(_TMP_DB) as _c:
    _c.execute(
        "CREATE TABLE IF NOT EXISTS media_jobs (job_id TEXT PRIMARY KEY, name TEXT, "
        "status TEXT, spec_json TEXT, result_json TEXT, claim_token TEXT, "
        "created REAL, updated REAL, progress_json TEXT)")

os.makedirs(DEFAULT_CLIPS_ROOT, exist_ok=True)
_WORK = tempfile.mkdtemp(prefix="serve-", dir=DEFAULT_CLIPS_ROOT)  # inside the jail


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


def _insert_job(job_id, status, result, spec):
    now = time.time()
    with sqlite3.connect(_TMP_DB) as c:
        c.execute(
            "INSERT OR REPLACE INTO media_jobs (job_id, name, status, spec_json, "
            "result_json, claim_token, created, updated, progress_json) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (job_id, "studio_i2v", status,
             json.dumps(spec) if spec is not None else None,
             json.dumps(result) if result is not None else None,
             None, now, now, None))


# Build a real synthetic clip once (clip.mp4 + manifest.json under the jail), then an
# EXTENSIONLESS copy beside it — the scenario the mimetype fix targets.
_CLIP_MP4 = None
_CLIP_NOEXT = None
if _FFMPEG:
    req = CapabilityRequest(capability=Capability.I2V,
                            target_resolution=Resolution(320, 180, 12),
                            vram_budget_gb=0.5)
    _res = produce_clip(req, env=_env(), out_root=_WORK)
    assert _res.is_ok(), _res
    _CLIP_MP4 = _res.unwrap().path
    _CLIP_NOEXT = os.path.join(os.path.dirname(_CLIP_MP4), "clipfile")  # NO extension
    shutil.copyfile(_CLIP_MP4, _CLIP_NOEXT)

    _spec = dataclasses.asdict(make_studio_i2v(
        capability="i2v", width=320, height=180, fps=12, vram_budget_gb=0.5, seed=0,
        out_root=_WORK, prompt="a test prompt"))

    def _ok_result(uri, with_mime):
        out = {"asset_id": "asset-serve", "uri": uri, "kind": "video",
               "width": 320, "height": 180, "duration_s": 2.0}
        if with_mime:
            out["mime"] = "video/mp4"
        return {"job_id": "x", "ok": True, "outputs": [out], "error": None}

    _insert_job("done-mp4", "done", _ok_result(_CLIP_MP4, True), _spec)
    _insert_job("done-noext-mime", "done", _ok_result(_CLIP_NOEXT, True), _spec)
    _insert_job("done-noext-nomime", "done", _ok_result(_CLIP_NOEXT, False), _spec)

# A FAILED job (no clip) — detail must surface the error + the requested spec.
_insert_job(
    "failed-job", "failed",
    {"job_id": "x", "ok": False, "outputs": [],
     "error": {"code": "vram_exceeded", "message": "no model fit the budget",
               "retryable": False}},
    dataclasses.asdict(make_studio_i2v(
        capability="i2v", width=1280, height=720, fps=16, vram_budget_gb=2.0, seed=0,
        out_root=_WORK, prompt="wanted big", steps=40, cfg=6.0)))

# A CANCELLED-before-start job — detail also falls back to the job record: the cancel is
# the error, the requested spec explains what was asked, no manifest exists.
_insert_job(
    "cancelled-job", "cancelled",
    {"job_id": "x", "ok": False, "outputs": [],
     "error": {"code": "cancelled", "message": "cancelled before it started",
               "retryable": False}},
    dataclasses.asdict(make_studio_i2v(
        capability="t2v", width=832, height=480, fps=16, vram_budget_gb=6.0, seed=7,
        out_root=_WORK, prompt="never ran")))


def _ct(resp):
    return (resp.headers.get("Content-Type") or "").split(";")[0].strip()


# --------------------------------------------------------------------------- #
# [1] Serve a clip whose uri HAS a .mp4 extension -> video/mp4 (baseline).
# --------------------------------------------------------------------------- #
def test_serve_mp4_extension():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping)")
        return
    r = client.get("/video/studio/clip/done-mp4")
    assert r.status_code == 200, r.status_code
    assert _ct(r) == "video/mp4", _ct(r)


# --------------------------------------------------------------------------- #
# [2] Serve a clip whose uri has NO extension, catalog mime present -> video/mp4.
# --------------------------------------------------------------------------- #
def test_serve_no_extension_catalog_mime():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping)")
        return
    r = client.get("/video/studio/clip/done-noext-mime")
    assert r.status_code == 200, r.status_code
    assert _ct(r) == "video/mp4", _ct(r)


# --------------------------------------------------------------------------- #
# [3] THE BUG FIX: no extension AND no catalog mime -> fall back to video/mp4
#     (never octet-stream / an unknown type the <video> renders gray).
# --------------------------------------------------------------------------- #
def test_serve_no_extension_no_mime_falls_back():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping)")
        return
    r = client.get("/video/studio/clip/done-noext-nomime")
    assert r.status_code == 200, r.status_code
    assert _ct(r) == "video/mp4", f"expected video/mp4, got {_ct(r)!r}"


# --------------------------------------------------------------------------- #
# [4] /video/media hardening: an extensionless clip served BY URI (cross-station
#     library playback) also gets video/mp4 (not octet-stream).
# --------------------------------------------------------------------------- #
def test_video_media_extensionless_clip():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping)")
        return
    r = client.get(f"/video/media?handle={_CLIP_NOEXT}")
    assert r.status_code == 200, r.status_code
    assert _ct(r) == "video/mp4", f"expected video/mp4, got {_ct(r)!r}"


# --------------------------------------------------------------------------- #
# [5] DETAIL of a DONE clip: the manifest (true render params) + the spec.
# --------------------------------------------------------------------------- #
def test_detail_done_clip():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping)")
        return
    r = client.get("/video/studio/clip/done-noext-mime/detail")
    assert r.status_code == 200, (r.status_code, r.get_json())
    body = r.get_json()
    assert body["status"] == "done", body
    m = body.get("manifest")
    assert m is not None, body
    assert m["model_id"] == "synthetic-i2v", m
    assert m["content_hash"], m
    assert m["sampler"]["steps"] == 1, m["sampler"]        # synthetic placeholder
    assert m["resolution"] == {"width": 320, "height": 180, "fps": 12}, m["resolution"]
    assert m["frames"] is not None, m                       # derived from duration*fps
    assert body["error"] is None, body
    # the requested spec rides along too
    assert body["spec"]["capability"] == "i2v", body["spec"]
    assert "out_root" not in body["spec"], body["spec"]     # internal path redacted
    # SOURCE discriminator: a DONE clip's params come from the content-addressed manifest.
    assert body["source"] == "manifest", body
    # Bus timestamps ride along in every case (epoch seconds).
    assert body["created"] is not None and body["updated"] is not None, body


# --------------------------------------------------------------------------- #
# [6] DETAIL of a FAILED job: error {code,message,retryable} + the requested spec,
#     manifest null (no clip was produced).
# --------------------------------------------------------------------------- #
def test_detail_failed_job():
    r = client.get("/video/studio/clip/failed-job/detail")
    assert r.status_code == 200, (r.status_code, r.get_json())
    body = r.get_json()
    assert body["status"] == "failed", body
    assert body["manifest"] is None, body
    # SOURCE discriminator (coordinator addendum): a FAILED job wrote no manifest, so the
    # detail falls back to the media-bus JOB RECORD — the params are REQUESTED, not
    # resolved, and this marker lets the UI label them so.
    assert body["source"] == "job_record", body
    assert body["error"]["code"] == "vram_exceeded", body["error"]
    assert body["error"]["retryable"] is False, body["error"]
    # the requested spec explains WHAT was asked for (incl. the override + budget)
    assert body["spec"]["steps"] == 40 and body["spec"]["cfg"] == 6.0, body["spec"]
    assert body["spec"]["vram_budget_gb"] == 2.0, body["spec"]
    # Bus timestamps present on the failed row too (for the expander's "when" line).
    assert body["created"] is not None and body["updated"] is not None, body


# --------------------------------------------------------------------------- #
# [7] DETAIL of a CANCELLED job: falls back to the job record (source job_record),
#     surfacing the cancel error + the requested spec, manifest null.
# --------------------------------------------------------------------------- #
def test_detail_cancelled_job():
    r = client.get("/video/studio/clip/cancelled-job/detail")
    assert r.status_code == 200, (r.status_code, r.get_json())
    body = r.get_json()
    assert body["status"] == "cancelled", body
    assert body["source"] == "job_record", body
    assert body["manifest"] is None, body
    assert body["error"]["code"] == "cancelled", body["error"]
    assert body["spec"]["capability"] == "t2v", body["spec"]


# --------------------------------------------------------------------------- #
# [8] DETAIL of an unknown id -> 404.
# --------------------------------------------------------------------------- #
def test_detail_unknown_404():
    r = client.get("/video/studio/clip/does-not-exist/detail")
    assert r.status_code == 404, r.status_code


CHECKS = [
    ("serve .mp4 uri -> Content-Type video/mp4", test_serve_mp4_extension),
    ("serve extensionless uri + catalog mime -> video/mp4", test_serve_no_extension_catalog_mime),
    ("serve extensionless uri, NO catalog mime -> video/mp4 fallback (the bug fix)",
     test_serve_no_extension_no_mime_falls_back),
    ("/video/media extensionless studio clip -> video/mp4", test_video_media_extensionless_clip),
    ("detail of a done clip: manifest params + spec (source manifest)", test_detail_done_clip),
    ("detail of a failed job: error + spec, manifest null (source job_record)",
     test_detail_failed_job),
    ("detail of a cancelled job: error + spec, manifest null (source job_record)",
     test_detail_cancelled_job),
    ("detail of an unknown id -> 404", test_detail_unknown_404),
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
