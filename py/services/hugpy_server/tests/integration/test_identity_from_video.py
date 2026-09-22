"""k94 — IDENTITY FROM VIDEO: the ONE chained ``video_characters_glb`` relay
(runners/identity_from_video.py) + the two one-path routes
(POST /video/identity-profiles/from-video, POST /video/identity-profiles/from-images).

Exercised WITHOUT a GPU and WITHOUT any network beyond localhost, against a threaded
http.server standing in for the IDENTITY_RENDER_URL service (the render service itself is
DOWN as of 2026-08-20 — these tests pin the DOCUMENTED contract clownworld's
characters3d.ts reads). The identities STORE + the media BUS are redirected to temp
dirs/DB exactly as test_identity_video_extract_relay.py does.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python -m pytest tests/test_identity_from_video.py -q
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
from hugpy_video.intel.runners import identity_from_video as ifv
from hugpy_video.intel.runners import identity_render_client as client
from hugpy_video.intel.identity_from_video_schema import (
    make_identity_from_video,
    identity_from_video_from_dict,
    DEFAULT_MESH_PARAMS,
)
from hugpy_video.intel.media_schema import make_media_ref
from hugpy_platform.constants import UPLOADS_HOME, DEFAULT_ROOT

# --------------------------------------------------------------------------- #
# STORE + BUS isolation (mirrors the other identity relay tests).
# --------------------------------------------------------------------------- #
_TMP_IDENTITIES = tempfile.mkdtemp(prefix="hugpy-ifv-store-",
                                   dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
identity_profiles.IDENTITIES_HOME = _TMP_IDENTITIES
_TMP_PROJECTS = tempfile.mkdtemp(prefix="hugpy-ifv-projects-")
identity_profiles.PROJECTS_HOME = _TMP_PROJECTS
_TMP_UPLOADS = tempfile.mkdtemp(prefix="hugpy-ifv-uploads-", dir=UPLOADS_HOME)

_TMP_DB = tempfile.mkstemp(prefix="ifv-bus-", suffix=".db")[1]
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
app_client = app.test_client()


# --------------------------------------------------------------------------- #
# The MOCK render service — honors the FIXED contract for a ``video_characters_glb``
# job. A SCRIPT controls the terminal status + the two manifests + the file list.
# --------------------------------------------------------------------------- #
_TOKEN = "test-render-token-ifv"
_STATUS = {"value": "done"}
_ERROR = {"value": None}
_SUMMARY = {"value": None}      # characters3d_result.json
_MANIFEST = {"value": None}     # char360_result.json
_EXTRA_FILES = {"value": []}    # extra job-relative files (e.g. a turntable)
_RECEIVED: list[dict] = []
_POLLS = {"n": 0}

_GLB_MAGIC = b"glTF" + b"\x02\x00\x00\x00" + b"\x00" * 24


def _png_bytes(color=(120, 40, 200)) -> bytes:
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


def _files_list() -> list[str]:
    files: list[str] = []
    if _SUMMARY["value"] is not None:
        files.append("characters3d_result.json")
    if _MANIFEST["value"] is not None:
        files.append("char360_result.json")
    for ch in ((_MANIFEST["value"] or {}).get("characters") or []):
        for v in (ch.get("views") or []):
            files.append(v["file"])
    for e in ((_SUMMARY["value"] or {}).get("characters") or []):
        if e.get("glb"):
            files.append(f"{e['char']}/identity.glb")
            files.append(f"{e['char']}/mesh.json")
        front = (e.get("views_used") or {}).get("front")
        if front:
            files.append(f"{e['char']}/{front}")
    files.extend(_EXTRA_FILES["value"])
    # dedupe, keep order
    out, seen = [], set()
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _raw(self, data: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authed(self) -> bool:
        return self.headers.get("X-Identity-Render-Token") == _TOKEN

    def do_GET(self):
        if not self._authed():
            return self._json(401, {"error": "bad token"})
        if self.path.startswith("/jobs/") and "/files/" in self.path:
            fname = self.path.split("/files/", 1)[1]
            if fname == "characters3d_result.json":
                return self._raw(json.dumps(_SUMMARY["value"] or {}).encode())
            if fname == "char360_result.json":
                return self._raw(json.dumps(_MANIFEST["value"] or {}).encode())
            if fname.endswith(".glb"):
                return self._raw(_GLB_MAGIC + fname.encode())
            if fname.endswith(".json"):
                return self._raw(json.dumps({"mesh": fname}).encode())
            if fname.endswith(".mp4"):
                return self._raw(b"\x00\x00\x00\x18ftypmp42" + fname.encode())
            return self._raw(_png_bytes((hash(fname) % 255, 60, 90)))
        if self.path.startswith("/jobs/"):
            _POLLS["n"] += 1
            job = self.path.rsplit("/", 1)[1]
            status = _STATUS["value"]
            resp = {"job_id": job, "status": status, "stage": "mesh",
                    "progress": 0.5, "log_tail": ["char_00: meshing"]}
            if status == "error":
                resp["error"] = _ERROR["value"] or "boom"
            elif status == "done":
                resp["files"] = _files_list()
            return self._json(200, resp)
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._authed():
            return self._json(401, {"error": "bad token"})
        if self.path == "/jobs":
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            _RECEIVED.append(payload)
            return self._json(202, {"job_id": "remote-ifv-1"})
        return self._json(404, {"error": "not found"})

    def do_DELETE(self):
        if not self._authed():
            return self._json(401, {"error": "bad token"})
        return self._json(200, {"ok": True})


_SERVER = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
_PORT = _SERVER.server_address[1]
_URL = f"http://127.0.0.1:{_PORT}"
threading.Thread(target=_SERVER.serve_forever, daemon=True).start()


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


_IMG_A = os.path.join(_TMP_UPLOADS, "a.png")
_IMG_B = os.path.join(_TMP_UPLOADS, "b.png")
_make_png(_IMG_A, (200, 40, 40))
_make_png(_IMG_B, (40, 200, 40))

_VIDEO_PATH = os.path.join(_TMP_UPLOADS, "clip.mp4")
with open(_VIDEO_PATH, "wb") as _f:
    _f.write(b"\x00\x00\x00\x18ftypmp42FAKE-MP4-BYTES")


def _video_ref():
    return make_media_ref(asset_id="clip1", kind="video", uri=_VIDEO_PATH, mime="video/mp4")


def _char_manifest(char_id: str, n_views: int, centroid=None):
    return {
        "char": char_id,
        "views": [
            {"bin": b, "file": f"{char_id}/view_{b:02d}_yaw{b * 45:+04d}.png",
             "yaw": float(b * 45), "yaw_source": "face_pose", "score": 0.9}
            for b in range(n_views)
        ],
        "bins_filled": list(range(n_views)), "bins_missing": [],
        "face_centroid": centroid,
    }


def _char_summary(char_id: str, n_views: int, glb=True, error=None):
    e = {"char": char_id, "n_views": n_views,
         "views_used": {"front": f"view_00_yaw+000.nobg.png"}}
    if glb:
        e["glb"] = f"{char_id}/identity.glb"
    if error:
        e["error"] = error
    return e


def _reset(status="done", error=None, chars=(("char_00", 8), ("char_01", 3)),
           glb_for=None, extra_files=()):
    _STATUS["value"] = status
    _ERROR["value"] = error
    _RECEIVED.clear()
    _POLLS["n"] = 0
    _EXTRA_FILES["value"] = list(extra_files)
    glb_for = set(c for c, _ in chars) if glb_for is None else set(glb_for)
    _MANIFEST["value"] = {"ok": True, "kind": "video_characters_glb", "n_characters": len(chars),
                          "characters": [_char_manifest(c, n) for c, n in chars]}
    _SUMMARY["value"] = {"n_characters": len(chars), "n_meshed": len(glb_for),
                         "characters": [_char_summary(c, n, glb=(c in glb_for),
                                                      error=None if c in glb_for else "mesh failed")
                                        for c, n in chars]}


def _drain(job_id: str) -> dict:
    processed = media_bus.work_once("test-worker")
    assert processed == job_id, (processed, job_id)
    return media_bus.get(job_id)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # Rebind the shared globals before EVERY test (other identity test files rebind the
    # same module globals in their import preambles — collection order must not matter).
    identity_profiles.IDENTITIES_HOME = _TMP_IDENTITIES
    identity_profiles.PROJECTS_HOME = _TMP_PROJECTS
    media_bus.DB_PATH = _TMP_DB
    media_bus._initialized = False
    monkeypatch.setenv("IDENTITY_RENDER_URL", _URL)
    monkeypatch.setenv("IDENTITY_RENDER_TOKEN", _TOKEN)
    monkeypatch.setattr(client, "POLL_INTERVAL_S", 0.0)
    yield


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
def test_spec_defaults_and_roundtrip():
    spec = make_identity_from_video(source=_video_ref(), name="  Hero  ")
    assert spec.name == "Hero"
    assert spec.mesh_params == DEFAULT_MESH_PARAMS == {"texture": True, "octree_resolution": 256}
    spec2 = make_identity_from_video(source=_video_ref(), name="Hero",
                                     mesh_params={"texture": False, "octree_resolution": 380,
                                                  "bogus": 1})
    assert spec2.mesh_params == {"texture": False, "octree_resolution": 380}
    from dataclasses import asdict
    again = identity_from_video_from_dict(json.loads(json.dumps(asdict(spec2))))
    assert again == spec2
    with pytest.raises(ValueError):
        make_identity_from_video(source=make_media_ref(asset_id="i", kind="image",
                                                       uri=_IMG_A, mime="image/png"),
                                 name="x")
    with pytest.raises(ValueError):
        make_identity_from_video(source=_video_ref(), name="")
    with pytest.raises(ValueError):
        make_identity_from_video(source=_video_ref(), name="x",
                                 mesh_params={"octree_resolution": -1})


def test_char_slug_rule():
    sl = identity_profiles.slugify
    assert ifv.char_slug("Big Hero", 0, sl) == ("Big Hero", "big-hero")
    assert ifv.char_slug("Big Hero", 1, sl) == ("Big Hero 2", "big-hero-2")
    assert ifv.char_slug("Big Hero", 2, sl) == ("Big Hero 3", "big-hero-3")


# --------------------------------------------------------------------------- #
# the happy path: payload shape, persist layout, one profile per character
# --------------------------------------------------------------------------- #
def test_payload_persist_and_profiles_per_character():
    _reset()
    spec = make_identity_from_video(source=_video_ref(), name="Relay Hero", identity_id="relay-hero")
    job_id = media_bus.enqueue("identity_from_video", spec)
    view = _drain(job_id)
    assert view["status"] == "done", view
    assert view["result"]["ok"] is True, view["result"]

    # PAYLOAD — byte-for-byte clownworld's shape.
    assert len(_RECEIVED) == 1
    sent = _RECEIVED[0]
    assert sent == {"kind": "video_characters_glb", "identity_id": "relay-hero",
                    "video_path": _VIDEO_PATH,
                    "mesh_params": {"texture": True, "octree_resolution": 256}}, sent

    # RESULT manifest
    ids = view["result"]["identities"]
    assert ids["name"] == "Relay Hero" and ids["n_characters"] == 2
    slugs = [p["slug"] for p in ids["profiles"]]
    assert slugs == ["relay-hero", "relay-hero-2"], ids
    assert all(p["glb"] for p in ids["profiles"]), ids
    assert [p["n_views"] for p in ids["profiles"]] == [9, 4]  # front + 8 / front + 3

    # PROFILES — one per character, refs = crops (front first), capped at MAX.
    p1 = identity_profiles.get_profile("relay-hero")
    p2 = identity_profiles.get_profile("relay-hero-2")
    assert p1 is not None and p2 is not None
    assert p1["name"] == "Relay Hero" and p2["name"] == "Relay Hero 2"
    assert 1 <= len(p1["reference_images"]) <= identity_profiles.MAX_SOURCE_IMAGES
    assert len(p1["reference_images"]) == 9
    # the first reference is the FRONT the mesh used (the routes' default front)
    assert os.path.isfile(p1["reference_images"][0])
    # canonical was promoted from the yaw-binned crops (8 semantic azimuths on char_00)
    assert len(p1["canonical"]) == 8, p1["canonical"]
    assert len(p2["canonical"]) == 3, p2["canonical"]

    # MESH — persisted under <slug>/mesh/<recon_id>/, mesh state done + glb_path, and a
    # version minted + ACTIVE so the existing mesh-status / bind routes just work.
    for slug in ("relay-hero", "relay-hero-2"):
        prof = identity_profiles.get_profile(slug)
        recs = prof["reconstructions"]
        assert len(recs) == 1, recs
        rec = recs[0]
        assert rec["recon_id"].startswith(f"fromvideo_{job_id}_char_")
        assert rec["mode"] == "video_extract"
        mesh = identity_profiles.get_mesh_state(slug, rec["recon_id"])
        assert mesh["status"] == "done", mesh
        assert mesh["glb_path"].endswith("identity.glb") and os.path.isfile(mesh["glb_path"])
        assert mesh["glb_path"].startswith(os.path.join(_TMP_IDENTITIES, slug, "mesh", rec["recon_id"]))
        assert mesh["mesh_json_path"].endswith("mesh.json")
        assert mesh["textured"] is True
        with open(mesh["glb_path"], "rb") as f:
            assert f.read(4) == b"glTF"
        assert prof["active_version"] is not None
        assert prof["versions"][0]["recon_id"] == rec["recon_id"]
        assert prof["versions"][0]["kind"] == "textured"
        # the GET mesh-status route reads the same record
        r = app_client.get(f"/video/identity-profiles/{slug}/reconstruction/{rec['recon_id']}/mesh")
        assert r.status_code == 200 and r.get_json()["status"] == "done"

    # staging crops live under the jailed IDENTITIES_HOME/_char360_extracts/<job_id>/
    assert os.path.isdir(os.path.join(_TMP_IDENTITIES, "_char360_extracts", job_id, "char_00"))


def test_turntable_files_are_attached_when_the_service_emits_them():
    _reset(chars=(("char_00", 4),),
           extra_files=["char_00/turntable.mp4"] + [f"char_00/frames/frame_{i:04d}.png" for i in range(8)])
    spec = make_identity_from_video(source=_video_ref(), name="Spinner")
    view = _drain(media_bus.enqueue("identity_from_video", spec))
    assert view["result"]["ok"] is True, view["result"]
    prof = identity_profiles.get_profile("spinner")
    rec = prof["reconstructions"][0]
    assert rec["mode"] == "turntable" and len(rec["views"]) == 8
    mesh = identity_profiles.get_mesh_state("spinner", rec["recon_id"])
    assert mesh["video_path"].endswith("turntable.mp4") and os.path.isfile(mesh["video_path"])
    assert mesh["frame_count"] == 8
    assert "turntable" in os.path.relpath(mesh["video_path"], _TMP_IDENTITIES)
    # canonical was promoted from the crops BEFORE the turntable replaced the view record
    assert len(prof["canonical"]) == 4


def test_rerun_refreshes_the_same_profile_instead_of_failing_on_duplicate():
    _reset(chars=(("char_00", 5),))
    spec = make_identity_from_video(source=_video_ref(), name="Twice")
    v1 = _drain(media_bus.enqueue("identity_from_video", spec))
    assert v1["result"]["ok"] is True
    first_refs = identity_profiles.get_profile("twice")["reference_images"]
    _reset(chars=(("char_00", 6),))
    v2 = _drain(media_bus.enqueue("identity_from_video", spec))
    assert v2["result"]["ok"] is True, v2["result"]
    prof = identity_profiles.get_profile("twice")
    assert prof["reference_images"] != first_refs
    assert len(prof["reference_images"]) == 7
    assert len([p for p in identity_profiles.list_profiles() if p["slug"].startswith("twice")]) == 1


def test_character_without_glb_still_gets_a_profile_but_reports_error():
    _reset(chars=(("char_00", 4), ("char_01", 4)), glb_for=("char_00",))
    spec = make_identity_from_video(source=_video_ref(), name="Partial")
    view = _drain(media_bus.enqueue("identity_from_video", spec))
    assert view["result"]["ok"] is True
    profs = {p["slug"]: p for p in view["result"]["identities"]["profiles"]}
    assert profs["partial"]["glb"] is True and "error" not in profs["partial"]
    assert profs["partial-2"]["glb"] is False and profs["partial-2"]["error"]
    assert identity_profiles.get_profile("partial-2") is not None   # views-only profile
    assert identity_profiles.get_profile("partial-2")["active_version"] is None


# --------------------------------------------------------------------------- #
# honest failures — DATA, never a raise
# --------------------------------------------------------------------------- #
def test_not_configured_is_error_as_data(monkeypatch):
    monkeypatch.delenv("IDENTITY_RENDER_URL")
    spec = make_identity_from_video(source=_video_ref(), name="Nope")
    view = _drain(media_bus.enqueue("identity_from_video", spec))
    assert view["status"] == "failed"
    assert view["result"]["error"]["code"] == "not_configured"
    assert "IDENTITY_RENDER_URL" in view["result"]["error"]["message"]
    assert identity_profiles.get_profile("nope") is None


def test_unreachable_service_is_error_as_data(monkeypatch):
    # a closed port — exactly the live condition on 2026-08-20 (9750 closed on the host)
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    monkeypatch.setenv("IDENTITY_RENDER_URL", f"http://127.0.0.1:{port}")
    spec = make_identity_from_video(source=_video_ref(), name="Down")
    view = _drain(media_bus.enqueue("identity_from_video", spec))
    assert view["status"] == "failed"
    assert view["result"]["error"]["code"] == "render_unreachable"
    assert view["result"]["error"]["retryable"] is True
    assert identity_profiles.get_profile("down") is None


def test_service_error_and_no_characters():
    _reset(status="error", error="cuda OOM")
    view = _drain(media_bus.enqueue("identity_from_video",
                                    make_identity_from_video(source=_video_ref(), name="Oom")))
    assert view["result"]["error"]["code"] == "render_failed"
    assert "cuda OOM" in view["result"]["error"]["message"]

    _reset(chars=())
    view = _drain(media_bus.enqueue("identity_from_video",
                                    make_identity_from_video(source=_video_ref(), name="Empty")))
    assert view["result"]["error"]["code"] == "no_characters"
    assert identity_profiles.get_profile("empty") is None


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
def test_from_video_route_validates_and_202s_even_when_service_is_down(monkeypatch):
    src = {"asset_id": "clip1", "kind": "video", "uri": _VIDEO_PATH, "mime": "video/mp4"}
    r = app_client.post("/video/identity-profiles/from-video", json={"source": src})
    assert r.status_code == 400 and "name" in r.get_json()["error"]
    r = app_client.post("/video/identity-profiles/from-video", json={"name": "x"})
    assert r.status_code == 400
    r = app_client.post("/video/identity-profiles/from-video",
                        json={"name": "x", "source": {**src, "uri": "/etc/passwd"}})
    assert r.status_code == 400 and "jail" in r.get_json()["error"]
    r = app_client.post("/video/identity-profiles/from-video",
                        json={"name": "x", "source": {**src, "kind": "image", "mime": "image/png"}})
    assert r.status_code == 400
    r = app_client.post("/video/identity-profiles/from-video",
                        json={"name": "x", "source": src, "mesh_params": "no"})
    assert r.status_code == 400

    # service DOWN: the route still 202s; the JOB fails honestly.
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    monkeypatch.setenv("IDENTITY_RENDER_URL", f"http://127.0.0.1:{port}")
    r = app_client.post("/video/identity-profiles/from-video",
                        json={"name": "Route Hero", "source": src,
                              "mesh_params": {"octree_resolution": 128}})
    assert r.status_code == 202, r.get_json()
    body = r.get_json()
    assert body["slug"] == "route-hero" and body["kind"] == "identity_from_video"
    view = _drain(body["job_id"])
    assert view["name"] == "identity_from_video"
    assert view["result"]["error"]["code"] == "render_unreachable"
    # the optional mesh_params rode onto the spec
    spec = json.loads(sqlite3.connect(_TMP_DB).execute(
        "SELECT spec_json FROM media_jobs WHERE job_id=?", (body["job_id"],)).fetchone()[0])
    assert spec["mesh_params"] == {"texture": True, "octree_resolution": 128}
    assert spec["identity_id"] == "route-hero"


def test_from_images_route_creates_profile_and_chains_mesh_build():
    r = app_client.post("/video/identity-profiles/from-images",
                        json={"name": "Photo Hero", "sources": [
                            {"asset_id": "a", "kind": "image", "uri": _IMG_A, "mime": "image/png"},
                            _IMG_B]})
    assert r.status_code == 202, r.get_json()
    body = r.get_json()
    assert body["slug"] == "photo-hero" and body["kind"] == "identity_mesh_build"
    assert body["profile"]["slug"] == "photo-hero"
    assert len(body["profile"]["reference_images"]) == 2
    assert body["recon_id"] and body["job_id"]
    # ONE enqueue: the mesh build, seeded "queued" on the minted recon
    assert media_bus.get(body["job_id"])["name"] == "identity_mesh_build"
    assert identity_profiles.get_mesh_state("photo-hero", body["recon_id"])["status"] == "queued"
    spec = json.loads(sqlite3.connect(_TMP_DB).execute(
        "SELECT spec_json FROM media_jobs WHERE job_id=?", (body["job_id"],)).fetchone()[0])
    assert spec["texture"] is True and spec["octree_resolution"] == 256
    assert spec["chain_turntable"] is True and spec["auto_promote"] is True
    # the fleet-VLM front auto-select gets every ref as a candidate (no explicit front)
    assert len(spec["view_candidates"]) == 2

    # duplicate name -> 409, nothing enqueued
    r = app_client.post("/video/identity-profiles/from-images",
                        json={"name": "Photo Hero", "sources": [_IMG_A]})
    assert r.status_code == 409
    # validation
    r = app_client.post("/video/identity-profiles/from-images", json={"name": "X", "sources": []})
    assert r.status_code == 400
    r = app_client.post("/video/identity-profiles/from-images",
                        json={"name": "X", "sources": ["/etc/passwd"]})
    assert r.status_code == 400
    r = app_client.post("/video/identity-profiles/from-images",
                        json={"name": "X", "sources": [
                            {"asset_id": "v", "kind": "video", "uri": _VIDEO_PATH, "mime": "video/mp4"}]})
    assert r.status_code == 400


def test_generate_route_unchanged_shape():
    # /generate still answers {job_id, recon_id} 200 after the helper factoring.
    r = app_client.post("/video/identity-profiles",
                        json={"name": "Gen Hero", "reference_images": [_IMG_A]})
    assert r.status_code == 201
    r = app_client.post("/video/identity-profiles/gen-hero/generate", json={})
    assert r.status_code == 200, r.get_json()
    assert set(r.get_json()) == {"job_id", "recon_id"}
    r = app_client.post("/video/identity-profiles/gen-hero/generate", json={"pose": "nope"})
    assert r.status_code == 400
    r = app_client.post("/video/identity-profiles/nobody/generate", json={})
    assert r.status_code == 404
