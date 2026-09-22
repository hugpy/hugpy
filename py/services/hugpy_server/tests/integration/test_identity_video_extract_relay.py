"""IDENTITY VIDEO-EXTRACT (char360) RELAY — the central bus runner that relays a source
video to a REMOTE GPU render service (central has no GPU + never runs char360), polls it,
downloads the per-character view-sets, and writes them back into identity profiles
(CHAR360-FEATURE-PLAN S3).

Exercises video_intel/runners/identity_video_extract_relay.py end-to-end WITHOUT a GPU and
WITHOUT any network beyond localhost, AND the two write-back mode gates in
video_intel/identity_profiles.py (attach_reconstruction + _recon_manifest) that S3 widened
so ``video_extract`` (and the latent ``angle-ring``) modes round-trip instead of being
silently downgraded to ``"sheet"``:
  * a threaded http.server stands in for the IDENTITY_RENDER_URL service, honoring the FIXED
    S2 contract (token-gated /jobs, GET /jobs/<id> -> done + files incl. char360_result.json,
    file download, DELETE cleanup), driven by a SCRIPT so each test controls the manifest;
  * the identities STORE + the media BUS are redirected to temp dirs/DB exactly as
    test_identity_render_relay.py does (module-global rebind — env isolation does NOT work
    since constants read the .env file), so nothing touches the real dev trees.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/test_identity_video_extract_relay.py
  # or: venv/bin/python -m pytest tests/test_identity_video_extract_relay.py -q
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
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.disable(logging.INFO)

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import importlib  # noqa: E402

import pytest  # noqa: E402
from flask import Flask  # noqa: E402

from hugpy_video.intel import identity_profiles
from hugpy_video.intel import media_bus
from hugpy_video.intel.runners import identity_video_extract_relay
from hugpy_video.intel.identity_video_extract_schema import make_identity_video_extract
from hugpy_video.intel.media_schema import make_media_ref
from hugpy_platform.constants import UPLOADS_HOME, DEFAULT_ROOT

# --------------------------------------------------------------------------- #
# STORE + BUS isolation (mirrors test_identity_render_relay.py exactly).
# --------------------------------------------------------------------------- #
_TMP_IDENTITIES = tempfile.mkdtemp(prefix="hugpy-vx-store-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
identity_profiles.IDENTITIES_HOME = _TMP_IDENTITIES
_TMP_PROJECTS = tempfile.mkdtemp(prefix="hugpy-vx-projects-")
identity_profiles.PROJECTS_HOME = _TMP_PROJECTS
_TMP_UPLOADS = tempfile.mkdtemp(prefix="hugpy-vx-uploads-", dir=UPLOADS_HOME)

_TMP_DB = tempfile.mkstemp(prefix="vx-bus-", suffix=".db")[1]
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


# --------------------------------------------------------------------------- #
# The MOCK render service (localhost, threaded). Honors the FIXED S2 contract for a
# ``video_extract`` job. A SCRIPT controls the terminal status + the manifest so each
# test drives a different shape (multi-char / no-char / error).
# --------------------------------------------------------------------------- #
_TOKEN = "test-render-token-vx"

# Mutable per-test knobs (set before each POST-driven run):
_STATUS = {"value": "done"}          # the terminal status GET reports ("done" | "error")
_ERROR = {"value": None}             # the error string when status == "error"
_MANIFEST = {"value": None}          # the char360_result.json dict the service "produced"
_INCLUDE_MANIFEST_FILE = {"value": True}   # whether char360_result.json is in the files list
_RECEIVED: list[dict] = []           # captured POST /jobs payloads, for assertions


def _png_bytes(color=(120, 40, 200)) -> bytes:
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


def _files_for_manifest(manifest: dict) -> list[str]:
    """The flat file list a done video_extract job reports: char360_result.json (optional
    per the include-flag) + every character's per-view file (job-relative char_NN/<file>)."""
    files: list[str] = []
    if _INCLUDE_MANIFEST_FILE["value"]:
        files.append("char360_result.json")
    for ch in (manifest.get("characters") or []):
        for v in (ch.get("views") or []):
            files.append(v["file"])
    return files


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence the server
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self) -> bool:
        return self.headers.get("X-Identity-Render-Token") == _TOKEN

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"ok": True, "service": "identity-render",
                                    "capabilities": {"video_extract": {"ready": True}}})
        if not self._authed():
            return self._json(401, {"error": "bad token"})
        if self.path.startswith("/jobs/") and "/files/" in self.path:
            fname = self.path.split("/files/", 1)[1]
            if fname == "char360_result.json":
                data = json.dumps(_MANIFEST["value"] or {}).encode()
            else:
                # a per-view PNG; colour it off the file name so bytes differ per view.
                data = _png_bytes((hash(fname) % 255, 60, 90))
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path.startswith("/jobs/"):
            job = self.path.rsplit("/", 1)[1]
            status = _STATUS["value"]
            resp = {"job_id": job, "status": status}
            if status == "error":
                resp["error"] = _ERROR["value"] or "boom"
            elif status == "done":
                resp["files"] = _files_for_manifest(_MANIFEST["value"] or {})
            return self._json(200, resp)
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._authed():
            return self._json(401, {"error": "bad token"})
        if self.path == "/jobs":
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            _RECEIVED.append(payload)
            return self._json(202, {"job_id": "remote-vx-1"})
        return self._json(404, {"error": "not found"})

    def do_DELETE(self):
        if not self._authed():
            return self._json(401, {"error": "bad token"})
        return self._json(200, {"ok": True})


_SERVER = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
_PORT = _SERVER.server_address[1]
_THREAD = threading.Thread(target=_SERVER.serve_forever, daemon=True)
_THREAD.start()

os.environ["IDENTITY_RENDER_URL"] = f"http://127.0.0.1:{_PORT}"
os.environ["IDENTITY_RENDER_TOKEN"] = _TOKEN
# Poll fast so the tests are instant (the mock reports terminal on the first poll anyway).
os.environ["IDENTITY_RENDER_POLL_INTERVAL_S"] = "0"


@atexit.register
def _cleanup() -> None:
    try:
        _SERVER.shutdown()
    except Exception:
        pass
    for d in (_TMP_IDENTITIES, _TMP_PROJECTS, _TMP_UPLOADS):
        shutil.rmtree(d, ignore_errors=True)
    try:
        os.remove(_TMP_DB)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_png(path: str, color=(180, 90, 40)) -> None:
    from PIL import Image
    Image.new("RGB", (64, 64), color).save(path)


_IMG = os.path.join(_TMP_UPLOADS, "seed.png")
_make_png(_IMG, (200, 40, 40))

# A real video uri only needs to be an absolute path for make_media_ref (the runner forwards
# it as video_path; the mock never actually reads the clip). Put it under the uploads jail so
# the ROUTE's jail check also passes.
_VIDEO_PATH = os.path.join(_TMP_UPLOADS, "clip.mp4")
with open(_VIDEO_PATH, "wb") as _f:
    _f.write(b"\x00\x00\x00\x18ftypmp42FAKE-MP4-BYTES")


def _video_ref():
    return make_media_ref(asset_id="clip1", kind="video", uri=_VIDEO_PATH, mime="video/mp4")


def _reset_service(status="done", error=None, manifest=None, include_manifest_file=True):
    _STATUS["value"] = status
    _ERROR["value"] = error
    _MANIFEST["value"] = manifest
    _INCLUDE_MANIFEST_FILE["value"] = include_manifest_file
    _RECEIVED.clear()


def _char(char_id: str, n_views: int, centroid):
    """A manifest character with n_views binned views (flat char_NN/<file> names)."""
    return {
        "char": char_id,
        "views": [
            {"bin": b, "file": f"{char_id}/view_{b:02d}_bin{b:02d}.png",
             "yaw": float(b * 30), "yaw_source": "face_pose", "score": 0.9}
            for b in range(n_views)
        ],
        "bins_filled": list(range(n_views)),
        "bins_missing": [],
        "face_centroid": centroid,
    }


def _manifest(characters, identity_id="vx", n=None):
    return {
        "ok": True, "kind": "video_extract", "identity_id": identity_id,
        "source_video": _VIDEO_PATH,
        "n_characters": (len(characters) if n is None else n),
        "bins_deg": 30, "characters": characters,
    }


def _create_profile(name: str) -> str:
    r = client.post("/video/identity-profiles",
                    json={"name": name, "reference_images": [_IMG], "notes": "seed"})
    assert r.status_code == 201, (r.status_code, r.get_json())
    return r.get_json()["profile"]["slug"]


def _drain_bus(job_id: str) -> dict:
    processed = media_bus.work_once("test-worker")
    assert processed == job_id, (processed, job_id)
    return media_bus.get(job_id)


# --------------------------------------------------------------------------- #
# CROSS-FILE ISOLATION — this file and test_identity_profiles_from_groups.py EACH rebind
# identity_profiles.IDENTITIES_HOME/PROJECTS_HOME + media_bus.DB_PATH to their OWN
# tempfile.mkdtemp store in their import preamble (above). Under pytest both modules import
# at COLLECTION time, so whichever file imports last wins those shared globals and the
# other file's tests silently run against the WRONG temp store. Re-assert THIS file's
# bindings before every test so file import/collection order never matters.
#
# media_bus._initialized guards a lazy one-time schema migration keyed off whatever
# DB_PATH happened to be bound when it last ran (see media_bus._ensure_db). If some other
# test file's globals rebound DB_PATH to ITS db and that db already got migrated,
# _initialized is left True — so when we rebind DB_PATH back to OUR db here, an
# already-True _initialized would make _ensure_db() a no-op and skip OUR db's migration.
# Our db WAS already schema-created by this file's own preamble (CREATE TABLE up front),
# so nothing breaks today, but future ALTER-TABLE migrations added to _ensure_db would
# silently never reach it. Clear _initialized here too so _ensure_db() (called at the top
# of every media_bus function) always re-verifies/re-migrates the CURRENTLY bound db —
# the ALTER TABLE calls are idempotent (they swallow "duplicate column" errors), so
# re-running them against an already-migrated db is a harmless no-op.
@pytest.fixture(autouse=True)
def _rebind_isolation_globals():
    identity_profiles.IDENTITIES_HOME = _TMP_IDENTITIES
    identity_profiles.PROJECTS_HOME = _TMP_PROJECTS
    media_bus.DB_PATH = _TMP_DB
    media_bus._initialized = False
    yield


# --------------------------------------------------------------------------- #
# [1] CREATE path — a 2-character clip mints TWO new profiles, each with a
#     video_extract reconstruction; the POST carried video_path (not b64).
# --------------------------------------------------------------------------- #
def test_create_mints_profile_per_character():
    man = _manifest([_char("char_00", 4, [0.1, 0.2, 0.3]),
                     _char("char_01", 3, [0.4, 0.5, 0.6])])
    _reset_service(manifest=man)

    spec = make_identity_video_extract(source=_video_ref(), target="create")
    job_id = media_bus.enqueue("identity_video_extract", spec)
    view = _drain_bus(job_id)
    assert view["status"] == "done", view
    assert view["result"]["ok"] is True, view["result"]

    # The service got a well-formed video_extract job, with the clip as video_path.
    assert _RECEIVED, "render service received no job"
    sent = _RECEIVED[-1]
    assert sent["kind"] == "video_extract", sent
    assert sent["video_path"] == _VIDEO_PATH, sent
    assert "video_b64" not in sent or not sent.get("video_b64"), sent
    assert sent["identity_id"], sent   # a correlation id was synthesized for create

    # TWO new profiles now exist (video-char-*), each with a video_extract recon carrying the
    # right frame_count + degrees_per_frame + face_centroid + char.
    profs = [p for p in identity_profiles.list_profiles()
             if p["name"].startswith("video-char-char_")]
    assert len(profs) == 2, [p["name"] for p in profs]
    by_char = {}
    for p in profs:
        recs = p.get("reconstructions") or []
        assert len(recs) == 1, p
        rec = recs[0]
        assert rec["mode"] == "video_extract", rec       # NOT silently downgraded to "sheet"
        by_char[rec["char"]] = rec
    assert set(by_char) == {"char_00", "char_01"}, by_char
    assert by_char["char_00"]["frame_count"] == 4, by_char["char_00"]
    assert by_char["char_00"]["degrees_per_frame"] == round(360.0 / 4, 2), by_char["char_00"]
    assert by_char["char_00"]["face_centroid"] == [0.1, 0.2, 0.3], by_char["char_00"]
    assert by_char["char_01"]["frame_count"] == 3, by_char["char_01"]
    assert len(by_char["char_00"]["views"]) == 4, by_char["char_00"]


# --------------------------------------------------------------------------- #
# [2] ADD path — an existing profile gains a video_extract reconstruction (APPEND,
#     not replace: a prior sheet recon is preserved).
# --------------------------------------------------------------------------- #
def test_add_appends_reconstruction_to_existing_profile():
    slug = _create_profile("Add Target")
    # a prior sheet reconstruction exists (should be preserved, never clobbered).
    s0 = os.path.join(_TMP_UPLOADS, "prior_sheet.png")
    _make_png(s0, (9, 9, 9))
    identity_profiles.attach_reconstruction(slug, "recon_prior", [s0],
                                            spec={"job_id": "p", "mode": "sheet"})

    man = _manifest([_char("char_00", 4, [1.0, 0.0, 0.0])], identity_id=slug)
    _reset_service(manifest=man)

    # Snapshot the create-profiles BEFORE the ADD (this module shares one store across the
    # whole run, so a prior create test may have minted some — the ADD must add NONE).
    before_created = {p["slug"] for p in identity_profiles.list_profiles()
                      if p["name"].startswith("video-char-")}

    spec = make_identity_video_extract(source=_video_ref(), target=slug, identity_id=slug)
    job_id = media_bus.enqueue("identity_video_extract", spec)
    view = _drain_bus(job_id)
    assert view["status"] == "done" and view["result"]["ok"] is True, view

    prof = client.get(f"/video/identity-profiles/{slug}").get_json()["profile"]
    recs = prof["reconstructions"]
    modes = sorted(r["mode"] for r in recs)
    assert modes == ["sheet", "video_extract"], modes    # prior preserved + new appended
    vx = next(r for r in recs if r["mode"] == "video_extract")
    assert vx["char"] == "char_00" and vx["face_centroid"] == [1.0, 0.0, 0.0], vx
    assert vx["frame_count"] == 4, vx

    # The ADD minted NO new create-profile (it appended to the named slug only).
    after_created = {p["slug"] for p in identity_profiles.list_profiles()
                     if p["name"].startswith("video-char-")}
    assert after_created == before_created, sorted(after_created - before_created)


# --------------------------------------------------------------------------- #
# [3] The ROUTE enqueues + jails: a MediaRef-dict source (create) enqueues; a
#     non-video source, an out-of-jail uri, and an unknown add-slug are clean 4xx.
# --------------------------------------------------------------------------- #
def test_route_enqueues_and_validates():
    man = _manifest([_char("char_00", 4, None)])
    _reset_service(manifest=man)

    import dataclasses
    src = dataclasses.asdict(_video_ref())

    # happy: create target enqueues + runs to done.
    r = client.post("/video/identity-profiles/video-extract",
                    json={"source": src, "target": "create"})
    assert r.status_code == 200, (r.status_code, r.get_json())
    body = r.get_json()
    assert body["target"] == "create" and isinstance(body["job_id"], str), body
    view = _drain_bus(body["job_id"])
    assert view["status"] == "done" and view["result"]["ok"] is True, view

    # a non-video source -> 400.
    img_src = dataclasses.asdict(make_media_ref(
        asset_id="i", kind="image", uri=_IMG, mime="image/png"))
    r400 = client.post("/video/identity-profiles/video-extract",
                       json={"source": img_src, "target": "create"})
    assert r400.status_code == 400, (r400.status_code, r400.get_json())

    # an out-of-jail source uri -> 400.
    esc = dict(src); esc["uri"] = "/etc/passwd"
    resc = client.post("/video/identity-profiles/video-extract",
                       json={"source": esc, "target": "create"})
    assert resc.status_code == 400, (resc.status_code, resc.get_json())

    # an unknown ADD slug -> 404 (checked up front, before any extract).
    r404 = client.post("/video/identity-profiles/video-extract",
                       json={"source": src, "target": "no-such-slug"})
    assert r404.status_code == 404, (r404.status_code, r404.get_json())

    # a missing target -> 400.
    rnotgt = client.post("/video/identity-profiles/video-extract", json={"source": src})
    assert rnotgt.status_code == 400, (rnotgt.status_code, rnotgt.get_json())


# --------------------------------------------------------------------------- #
# [4] not_configured — a missing IDENTITY_RENDER_URL is a clean error-as-data,
#     never a raise through the bus.
# --------------------------------------------------------------------------- #
def test_not_configured_is_error_as_data():
    saved = os.environ.pop("IDENTITY_RENDER_URL", None)
    try:
        spec = make_identity_video_extract(source=_video_ref(), target="create")
        res = identity_video_extract_relay.run_identity_video_extract(spec, "job-x")
        assert res.ok is False, res
        assert res.error is not None and res.error.code == "not_configured", res.error
    finally:
        if saved is not None:
            os.environ["IDENTITY_RENDER_URL"] = saved


# --------------------------------------------------------------------------- #
# [5] 401 — a wrong token is a clean render_unauthorized error-as-data.
# --------------------------------------------------------------------------- #
def test_bad_token_is_error_as_data():
    saved = os.environ.get("IDENTITY_RENDER_TOKEN")
    os.environ["IDENTITY_RENDER_TOKEN"] = "WRONG-TOKEN"
    try:
        spec = make_identity_video_extract(source=_video_ref(), target="create")
        res = identity_video_extract_relay.run_identity_video_extract(spec, "job-401")
        assert res.ok is False and res.error.code == "render_unauthorized", res.error
    finally:
        if saved is not None:
            os.environ["IDENTITY_RENDER_TOKEN"] = saved


# --------------------------------------------------------------------------- #
# [6] unreachable — a down host is a clean render_unreachable error-as-data.
# --------------------------------------------------------------------------- #
def test_unreachable_is_error_as_data():
    saved = os.environ.get("IDENTITY_RENDER_URL")
    # a closed port on localhost -> connection refused fast.
    os.environ["IDENTITY_RENDER_URL"] = "http://127.0.0.1:1"
    try:
        spec = make_identity_video_extract(source=_video_ref(), target="create")
        res = identity_video_extract_relay.run_identity_video_extract(spec, "job-unreach")
        assert res.ok is False and res.error.code == "render_unreachable", res.error
    finally:
        if saved is not None:
            os.environ["IDENTITY_RENDER_URL"] = saved


# --------------------------------------------------------------------------- #
# [7] status=error — the service reporting an error is a clean render_failed.
# --------------------------------------------------------------------------- #
def test_service_error_is_error_as_data():
    _reset_service(status="error", error="yolo weights missing")
    spec = make_identity_video_extract(source=_video_ref(), target="create")
    res = identity_video_extract_relay.run_identity_video_extract(spec, "job-err")
    assert res.ok is False and res.error.code == "render_failed", res.error
    assert "yolo weights missing" in res.error.message, res.error


# --------------------------------------------------------------------------- #
# [8] no characters — an extract that finds nobody is a clean no_characters
#     error-as-data; nothing is written back.
# --------------------------------------------------------------------------- #
def test_no_characters_is_error_as_data():
    before = len(identity_profiles.list_profiles())
    _reset_service(manifest=_manifest([], n=0))
    spec = make_identity_video_extract(source=_video_ref(), target="create")
    res = identity_video_extract_relay.run_identity_video_extract(spec, "job-empty")
    assert res.ok is False and res.error.code == "no_characters", res.error
    assert len(identity_profiles.list_profiles()) == before, "no profile should be minted"


# --------------------------------------------------------------------------- #
# [9] WRITE-BACK MODE FIX (the S3 gate widening) — a direct attach with
#     mode="video_extract" persists that mode (NOT "sheet") and carries char +
#     face_centroid; and the latent "angle-ring" mode now round-trips too.
# --------------------------------------------------------------------------- #
def test_write_back_mode_and_face_centroid_persist():
    slug = _create_profile("Mode Fix")
    s0 = os.path.join(_TMP_UPLOADS, "mf0.png")
    _make_png(s0, (3, 4, 5))

    rec = identity_profiles.attach_reconstruction(
        slug, "recon_vx", [s0],
        spec={"source": "identity_video_extract_relay", "mode": "video_extract",
              "frame_count": 1, "degrees_per_frame": 360.0, "job_id": "j",
              "char": "char_00", "face_centroid": [0.7, 0.1, 0.2]})
    # the RETURNED record keeps the mode (not downgraded) + the new provenance keys.
    assert rec["mode"] == "video_extract", rec
    assert rec["char"] == "char_00", rec
    assert rec["face_centroid"] == [0.7, 0.1, 0.2], rec

    # it ALSO round-trips on READ (get_reconstruction / the profile manifest normalizer).
    got = identity_profiles.get_reconstruction(slug, "recon_vx")
    assert got["mode"] == "video_extract", got
    assert got["char"] == "char_00" and got["face_centroid"] == [0.7, 0.1, 0.2], got
    prof = client.get(f"/video/identity-profiles/{slug}").get_json()["profile"]
    r0 = next(r for r in prof["reconstructions"] if r["recon_id"] == "recon_vx")
    assert r0["mode"] == "video_extract" and r0["face_centroid"] == [0.7, 0.1, 0.2], r0

    # the latent angle-ring bug is fixed in the SAME gate: it no longer downgrades to sheet.
    s1 = os.path.join(_TMP_UPLOADS, "mf1.png")
    _make_png(s1, (6, 7, 8))
    ar = identity_profiles.attach_reconstruction(
        slug, "recon_ar", [s1], spec={"job_id": "j2", "mode": "angle-ring"})
    assert ar["mode"] == "angle-ring", ar
    assert identity_profiles.get_reconstruction(slug, "recon_ar")["mode"] == "angle-ring"

    # and a plain (no-mode) record still defaults to "sheet" — unchanged behavior.
    s2 = os.path.join(_TMP_UPLOADS, "mf2.png")
    _make_png(s2, (1, 1, 1))
    sh = identity_profiles.attach_reconstruction(slug, "recon_sheet2", [s2], spec={"job_id": "j3"})
    assert sh["mode"] == "sheet", sh


# --------------------------------------------------------------------------- #
# [10] REVIEW path (CHARACTER-GROUPS-PLAN S1) — target="review" runs char360, downloads
#      the per-character crops into the storage jail, and returns the grouped views on
#      JobResult.groups WITHOUT writing ANY identity profile. Retrieved via the same
#      GET /video/jobs plumbing (media_bus.get -> result.groups).
# --------------------------------------------------------------------------- #
def test_review_returns_groups_and_writes_no_profile():
    man = _manifest([_char("char_00", 4, [0.1, 0.2, 0.3]),
                     _char("char_01", 3, [0.4, 0.5, 0.6])])
    _reset_service(manifest=man)

    before = {p["slug"] for p in identity_profiles.list_profiles()}

    spec = make_identity_video_extract(source=_video_ref(), target="review")
    job_id = media_bus.enqueue("identity_video_extract", spec)
    view = _drain_bus(job_id)
    assert view["status"] == "done", view
    assert view["result"]["ok"] is True, view["result"]

    # The grouped manifest rides the terminal result.
    grouped = view["result"]["groups"]
    assert grouped is not None, view["result"]
    assert grouped["n_characters"] == 2, grouped
    gs = grouped["groups"]
    assert [g["char"] for g in gs] == ["char_00", "char_01"], gs
    assert gs[0]["face_centroid"] == [0.1, 0.2, 0.3], gs[0]
    # per-view shape: url (a persisted jailed handle that EXISTS) + yaw/bin/score.
    v0 = gs[0]["views"]
    assert len(v0) == 4, v0
    assert len(gs[1]["views"]) == 3, gs[1]
    for i, vv in enumerate(v0):
        assert set(vv) == {"url", "yaw", "bin", "score"}, vv
        assert isinstance(vv["url"], str) and os.path.isabs(vv["url"]), vv
        assert os.path.isfile(vv["url"]), vv          # the crop was really persisted
        assert vv["url"].startswith(_TMP_IDENTITIES), vv  # under the storage jail
        assert vv["bin"] == i and vv["yaw"] == float(i * 30), vv
        assert vv["score"] == 0.9, vv

    # NO profile was created or appended — review is non-committing.
    after = {p["slug"] for p in identity_profiles.list_profiles()}
    assert after == before, sorted(after - before)


# --------------------------------------------------------------------------- #
# [11] REVIEW via the ROUTE — POST target="review" enqueues (no profile lookup), runs to
#      done, and the grouped manifest is retrievable through GET /video/jobs plumbing.
# --------------------------------------------------------------------------- #
def test_review_route_enqueues_and_returns_groups():
    man = _manifest([_char("char_00", 2, [1.0, 0.0, 0.0])])
    _reset_service(manifest=man)

    before = {p["slug"] for p in identity_profiles.list_profiles()}

    import dataclasses
    src = dataclasses.asdict(_video_ref())
    r = client.post("/video/identity-profiles/video-extract",
                    json={"source": src, "target": "review"})
    assert r.status_code == 200, (r.status_code, r.get_json())
    body = r.get_json()
    assert body["target"] == "review" and isinstance(body["job_id"], str), body

    view = _drain_bus(body["job_id"])
    assert view["status"] == "done" and view["result"]["ok"] is True, view
    grouped = view["result"]["groups"]
    assert grouped["n_characters"] == 1, grouped
    assert grouped["groups"][0]["char"] == "char_00", grouped

    # Route review path mints NO profile (it never resolves target as a slug).
    after = {p["slug"] for p in identity_profiles.list_profiles()}
    assert after == before, sorted(after - before)


# --------------------------------------------------------------------------- #
# [12] REVIEW with no characters is still a clean no_characters error-as-data (the shared
#      early guard), and nothing is written.
# --------------------------------------------------------------------------- #
def test_review_no_characters_is_error_as_data():
    before = len(identity_profiles.list_profiles())
    _reset_service(manifest=_manifest([], n=0))
    spec = make_identity_video_extract(source=_video_ref(), target="review")
    res = identity_video_extract_relay.run_identity_video_extract(spec, "job-review-empty")
    assert res.ok is False and res.error.code == "no_characters", res.error
    assert res.groups is None, res
    assert len(identity_profiles.list_profiles()) == before, "review must write nothing"


CHECKS = [
    ("create: a 2-char clip mints a profile per character (video_extract recon)",
     test_create_mints_profile_per_character),
    ("add: an existing profile gains a video_extract recon (append, prior preserved)",
     test_add_appends_reconstruction_to_existing_profile),
    ("route: enqueues a MediaRef-dict source; jails non-video / escape / unknown-slug",
     test_route_enqueues_and_validates),
    ("not_configured -> clean error-as-data (never raises)", test_not_configured_is_error_as_data),
    ("401 -> render_unauthorized error-as-data", test_bad_token_is_error_as_data),
    ("unreachable host -> render_unreachable error-as-data", test_unreachable_is_error_as_data),
    ("service status=error -> render_failed error-as-data", test_service_error_is_error_as_data),
    ("no characters found -> no_characters error-as-data; nothing written", test_no_characters_is_error_as_data),
    ("write-back: video_extract mode + char + face_centroid persist; angle-ring round-trips",
     test_write_back_mode_and_face_centroid_persist),
    ("review: 2-char clip returns grouped views (url/yaw/bin/score) + writes NO profile",
     test_review_returns_groups_and_writes_no_profile),
    ("review route: POST target=review enqueues + returns groups; mints no profile",
     test_review_route_enqueues_and_returns_groups),
    ("review: no characters -> no_characters error-as-data; nothing written",
     test_review_no_characters_is_error_as_data),
]


def main() -> int:
    passed = failed = 0
    for i, (name, fn) in enumerate(CHECKS, 1):
        try:
            fn()
        except Exception as exc:  # surface EVERY divergence, not just the first
            failed += 1
            import traceback
            print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"[{i}] PASS  {name}")
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
