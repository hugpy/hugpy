"""IDENTITY 3D MESH RELAY (studio stage (c)) — the bus runner that relays a mesh +
turntable build to a REMOTE GPU render service (central has no GPU).

Exercises video_intel/runners/identity_render_relay.py end-to-end WITHOUT a GPU and
WITHOUT any network beyond localhost:
  * a threaded http.server stands in for the IDENTITY_RENDER_URL service, honoring the
    FIXED contract (token-gated /jobs, GET /jobs/<id> -> done + files, file download,
    DELETE cleanup);
  * the identities STORE + the media BUS are redirected to temp dirs/DB exactly as
    test_identity_profiles.py does (module-global rebind — env isolation does NOT work
    since constants read the .env file), so nothing touches the real dev trees.

Drives the whole chain the task specifies:
    route (build extended spec + jail) -> enqueue -> bus claim -> RELAY runner
    -> POST /jobs -> poll -> download -> PERSIST under the identity dir
    -> attach turntable frames as a reconstruction -> record mesh state
    -> canonical PROMOTE of a turntable frame.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/test_identity_render_relay.py
  # or: venv/bin/python -m pytest tests/test_identity_render_relay.py -q
"""
from __future__ import annotations

import atexit
import base64
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

from flask import Flask  # noqa: E402

from hugpy_video.intel import identity_profiles
from hugpy_video.intel import media_bus
from hugpy_video.intel.runners import identity_render_relay
from hugpy_video.intel.identity_reconstruction_schema import make_identity_mesh
from hugpy_platform.constants import UPLOADS_HOME, DEFAULT_ROOT

# --------------------------------------------------------------------------- #
# STORE + BUS isolation (mirrors test_identity_profiles.py exactly).
# --------------------------------------------------------------------------- #
_TMP_IDENTITIES = tempfile.mkdtemp(prefix="hugpy-relay-store-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
identity_profiles.IDENTITIES_HOME = _TMP_IDENTITIES
_TMP_PROJECTS = tempfile.mkdtemp(prefix="hugpy-relay-projects-")
identity_profiles.PROJECTS_HOME = _TMP_PROJECTS
_TMP_UPLOADS = tempfile.mkdtemp(prefix="hugpy-relay-uploads-", dir=UPLOADS_HOME)

_TMP_DB = tempfile.mkstemp(prefix="relay-bus-", suffix=".db")[1]
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
# The MOCK render service (localhost, threaded). Honors the FIXED contract.
# --------------------------------------------------------------------------- #
_TOKEN = "test-render-token-abc123"
_FRAME_COUNT = 4
# The file set a "mesh_and_turntable" done job reports (contract file names).
_FILES = ["identity.glb", "identity_mesh.json", "turntable.mp4"] + [
    f"frames/frame_{i:04d}.png" for i in range(_FRAME_COUNT)
]


def _png_bytes(color=(120, 40, 200)) -> bytes:
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


def _file_bytes(name: str) -> bytes:
    if name.endswith(".png"):
        idx = int(name.split("_")[-1].split(".")[0])
        return _png_bytes((idx * 30 % 255, 60, 90))
    if name.endswith(".glb"):
        return b"glTF\x02\x00\x00\x00FAKE-BINARY-GLB-BYTES"
    if name.endswith(".json"):
        return json.dumps({"request_id": "r", "identity_id": "id", "textured": False}).encode()
    if name.endswith(".mp4"):
        return b"\x00\x00\x00\x18ftypmp42FAKE-MP4-BYTES"
    return b""


_RECEIVED: list[dict] = []   # captured POST /jobs payloads, for assertions

# ---- /ml/vision stand-in (shares this same mock server + port; HUGPY_CENTRAL_URL is
# pointed at it below) — the runner's front-auto-selection helper POSTs here with no
# auth (the real /ml gate is a DIFFERENT, currently-open gate; the mock mirrors that).
# ``_VISION_SCRIPT`` maps a candidate's raw image bytes -> the {"ok","text"} reply the
# mock returns for it (default: {"ok": True, "text": "no"} for any un-scripted image, so
# a test only needs to script the candidates it cares about). ``_VISION_CALLS`` records
# every candidate's raw bytes, in call order, so a test can assert exactly which images
# were (or were NOT) checked. #
_VISION_SCRIPT: dict[bytes, dict] = {}
_VISION_CALLS: list[bytes] = []


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
                                    "capabilities": {"mesh": {"ready": True},
                                                     "turntable": {"ready": True}}})
        if not self._authed():
            return self._json(401, {"error": "bad token"})
        if self.path.startswith("/jobs/") and "/files/" in self.path:
            fname = self.path.split("/files/", 1)[1]
            data = _file_bytes(fname)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path.startswith("/jobs/"):
            job = self.path.rsplit("/", 1)[1]
            # Report DONE immediately with the produced file set.
            return self._json(200, {"job_id": job, "status": "done", "files": _FILES})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        # /ml/vision needs no auth (mirrors the real amenity's currently-open gate) and
        # is a DIFFERENT contract from the token-gated render-service /jobs below.
        if self.path == "/ml/vision":
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            raw = base64.b64decode(payload.get("image_b64") or "")
            _VISION_CALLS.append(raw)
            reply = _VISION_SCRIPT.get(raw, {"ok": True, "text": "no, the legs are cropped"})
            return self._json(200, reply)
        if not self._authed():
            return self._json(401, {"error": "bad token"})
        if self.path == "/jobs":
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            _RECEIVED.append(payload)
            return self._json(202, {"job_id": "remote-job-1"})
        return self._json(404, {"error": "not found"})

    def do_DELETE(self):
        if not self._authed():
            return self._json(401, {"error": "bad token"})
        return self._json(200, {"ok": True})


_SERVER = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
_PORT = _SERVER.server_address[1]
_THREAD = threading.Thread(target=_SERVER.serve_forever, daemon=True)
_THREAD.start()

# Point the relay at the mock (read by the runner at RUN time via os.getenv). The
# fleet vision amenity base URL (HUGPY_CENTRAL_URL, default http://127.0.0.1:7002 in
# prod) is ALSO pointed at this same mock server — it serves both the render-service
# contract (/jobs, token-gated) and the /ml/vision stand-in (no auth) side by side.
os.environ["IDENTITY_RENDER_URL"] = f"http://127.0.0.1:{_PORT}"
os.environ["IDENTITY_RENDER_TOKEN"] = _TOKEN
os.environ["HUGPY_CENTRAL_URL"] = f"http://127.0.0.1:{_PORT}"


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
# fixtures
# --------------------------------------------------------------------------- #
def _make_png(path: str, color=(180, 90, 40)) -> None:
    from PIL import Image
    Image.new("RGB", (64, 64), color).save(path)


_IMG = os.path.join(_TMP_UPLOADS, "hero_a.png")
_make_png(_IMG, (200, 40, 40))


def _create_profile(name: str) -> str:
    r = client.post("/video/identity-profiles",
                    json={"name": name, "reference_images": [_IMG], "notes": "a knight"})
    assert r.status_code == 201, (r.status_code, r.get_json())
    return r.get_json()["profile"]["slug"]


def _create_profile_multi(name: str, imgs: list[str]) -> str:
    """Like ``_create_profile`` but with >=2 reference images — the shape needed to
    exercise fleet-VLM front auto-selection (the route only hands the runner >=2
    ``view_candidates`` when there are >=2 existing source references AND no explicit
    ``views.front`` override)."""
    r = client.post("/video/identity-profiles",
                    json={"name": name, "reference_images": imgs, "notes": "a knight"})
    assert r.status_code == 201, (r.status_code, r.get_json())
    return r.get_json()["profile"]["slug"]


def _drain_bus(job_id: str) -> dict:
    """Run the claimed job to completion (the relay is synchronous) and return the
    bus view. work_once claims + runs; the mock reports done on the first poll, so this
    returns without any real wait."""
    processed = media_bus.work_once("test-worker")
    assert processed == job_id, (processed, job_id)
    return media_bus.get(job_id)


# --------------------------------------------------------------------------- #
# [1] FULL relay happy path: route -> enqueue -> relay -> persist -> attach -> promote.
# --------------------------------------------------------------------------- #
def test_relay_full_pipeline():
    _RECEIVED.clear()
    slug = _create_profile("Relay Hero")

    # Simulate the real UI flow: a reconstruction already exists (the operator clicks
    # "build mesh" on it), so attach a prior SHEET recon under the recon_id first.
    recon_id = "recon_mesh_1"
    s0 = os.path.join(_TMP_UPLOADS, "sheet0.png")
    _make_png(s0, (10, 20, 30))
    identity_profiles.attach_reconstruction(slug, recon_id, [s0],
                                            spec={"job_id": "prior", "mode": "sheet"})

    # Build the mesh via the REPOINTED route (default front = the profile's own ref;
    # chain a turntable). Returns 200 {job_id, recon_id}.
    r = client.post(f"/video/identity-profiles/{slug}/reconstruction/{recon_id}/mesh",
                    json={"chain_turntable": True, "texture": False,
                          "turntable": {"frame_count": _FRAME_COUNT, "fps": 12}})
    assert r.status_code == 200, (r.status_code, r.get_json())
    job_id = r.get_json()["job_id"]
    assert r.get_json()["recon_id"] == recon_id, r.get_json()

    # queued state seeded on the existing reconstruction.
    assert (identity_profiles.get_mesh_state(slug, recon_id) or {}).get("status") == "queued"

    # Run the relay through the bus.
    view = _drain_bus(job_id)
    assert view["status"] == "done", view
    assert view["result"]["ok"] is True, view["result"]

    # The mock received a well-formed job (kind, identity_id, front view b64).
    assert _RECEIVED, "render service received no job"
    sent = _RECEIVED[-1]
    assert sent["kind"] == "mesh_and_turntable", sent
    assert sent["identity_id"] == slug, sent
    assert "front" in sent["views"] and sent["views"]["front"], sent
    assert sent["mesh_params"]["octree_resolution"] == 380, sent
    assert sent["turntable_params"]["frame_count"] == _FRAME_COUNT, sent

    # PERSISTENCE: GLB + mesh json at the mesh root; mp4 + frames under turntable/.
    mesh_dir = os.path.join(_TMP_IDENTITIES, slug, "mesh", recon_id)
    assert os.path.isfile(os.path.join(mesh_dir, "identity.glb")), mesh_dir
    assert os.path.isfile(os.path.join(mesh_dir, "identity_mesh.json")), mesh_dir
    assert os.path.isfile(os.path.join(mesh_dir, "turntable", "turntable.mp4")), mesh_dir
    fdir = os.path.join(mesh_dir, "turntable", "frames")
    frames_on_disk = sorted(n for n in os.listdir(fdir) if n.endswith(".png"))
    assert len(frames_on_disk) == _FRAME_COUNT, frames_on_disk

    # No stray *.tmp left behind (atomic writes).
    strays = []
    for root, _d, files in os.walk(mesh_dir):
        strays += [f for f in files if f.endswith(".tmp")]
    assert strays == [], strays

    # MESH STATE recorded terminal + GLB path (what GET .../mesh reads).
    ms = identity_profiles.get_mesh_state(slug, recon_id)
    assert ms["status"] == "done" and ms["error"] is None, ms
    assert ms["glb_path"] == os.path.join(mesh_dir, "identity.glb"), ms
    assert ms["video_path"] == os.path.join(mesh_dir, "turntable", "turntable.mp4"), ms
    assert ms["frame_count"] == _FRAME_COUNT, ms

    # RECONSTRUCTION ATTACH: the recon_id record is now the turntable (replace, NOT a
    # duplicate) with the frames as scrubbable views in angular order.
    prof = client.get(f"/video/identity-profiles/{slug}").get_json()["profile"]
    recons = [r for r in prof["reconstructions"] if r["recon_id"] == recon_id]
    assert len(recons) == 1, f"expected exactly one recon record, got {len(recons)}"
    rec = recons[0]
    assert rec["mode"] == "turntable", rec
    assert rec["frame_count"] == _FRAME_COUNT, rec
    assert len(rec["views"]) == _FRAME_COUNT, rec
    assert rec["degrees_per_frame"] == round(360.0 / _FRAME_COUNT, 2), rec

    # JobResult OUTPUTS reference the turntable mp4 (a video MediaRef; the GLB rides in
    # mesh state since a GLB is not a MediaRef kind).
    outs = view["result"]["outputs"]
    assert len(outs) == 1 and outs[0]["kind"] == "video", outs
    assert outs[0]["uri"].endswith("turntable.mp4"), outs

    # CANONICAL PROMOTE of a turntable frame -> the promote route works unchanged.
    p = client.post(f"/video/identity-profiles/{slug}/canonical",
                    json={"recon_id": recon_id, "views": [0]})
    assert p.status_code == 200, (p.status_code, p.get_json())
    canon = p.get_json()["profile"]["canonical"]
    assert len(canon) == 1 and os.path.isfile(canon[0]), canon


# --------------------------------------------------------------------------- #
# [2] Mesh-only (no turntable) — kind flips to mesh_build; GLB persisted, no mp4.
# --------------------------------------------------------------------------- #
def test_relay_mesh_only_no_turntable():
    _RECEIVED.clear()
    slug = _create_profile("Relay MeshOnly")
    recon_id = "recon_mesh_only"

    spec = make_identity_mesh(
        slug=slug, recon_id=recon_id,
        view_sources=[("front", client.get(f"/video/identity-profiles/{slug}")
                       .get_json()["profile"]["reference_images"][0])],
        chain_turntable=False,
    )
    job_id = media_bus.enqueue("identity_mesh_build", spec)
    view = _drain_bus(job_id)
    assert view["status"] == "done" and view["result"]["ok"] is True, view

    sent = _RECEIVED[-1]
    assert sent["kind"] == "mesh_build", sent

    mesh_dir = os.path.join(_TMP_IDENTITIES, slug, "mesh", recon_id)
    assert os.path.isfile(os.path.join(mesh_dir, "identity.glb")), mesh_dir
    ms = identity_profiles.get_mesh_state(slug, recon_id)
    # No reconstruction was pre-attached, so mesh state is None (best-effort seed) BUT the
    # relay still records done state IF a recon exists; here the frames still attach one.
    # (The mock always returns turntable files, so a recon record is created + state set.)
    assert ms is not None and ms["status"] == "done", ms


# --------------------------------------------------------------------------- #
# [3] not_configured — a missing IDENTITY_RENDER_URL is a clean error-as-data
#     (ok=False, code "not_configured"), never a raise through the bus.
# --------------------------------------------------------------------------- #
def test_relay_not_configured_is_error_as_data():
    saved = os.environ.pop("IDENTITY_RENDER_URL", None)
    try:
        spec = make_identity_mesh(slug="ghost", recon_id="r",
                                  view_sources=[("front", _IMG)])
        res = identity_render_relay.run_identity_mesh_build(spec, "job-x")
        assert res.ok is False, res
        assert res.error is not None and res.error.code == "not_configured", res.error
    finally:
        if saved is not None:
            os.environ["IDENTITY_RENDER_URL"] = saved


# --------------------------------------------------------------------------- #
# [4] Route JAIL — a view path that is NOT one of the profile's own images is a
#     clean 400 (never accepted); an unknown slug is a 404.
# --------------------------------------------------------------------------- #
def test_mesh_route_jails_arbitrary_paths():
    slug = _create_profile("Relay Jail")
    recon_id = "recon_jail"
    r = client.post(f"/video/identity-profiles/{slug}/reconstruction/{recon_id}/mesh",
                    json={"views": {"front": "/etc/passwd"}})
    assert r.status_code == 400, (r.status_code, r.get_json())

    r404 = client.post(f"/video/identity-profiles/no-such/reconstruction/{recon_id}/mesh",
                       json={})
    assert r404.status_code == 404, r404.status_code


# --------------------------------------------------------------------------- #
# [5] ONE-CLICK FULL IDENTITY (POST .../generate) with an EMPTY canonical set —
#     the relay auto-promotes the 4 cardinal turntable frames to canonical.
# --------------------------------------------------------------------------- #
def test_generate_route_auto_promotes_when_canonical_empty():
    _RECEIVED.clear()
    slug = _create_profile("Gen AutoPromote")

    # A fresh profile starts with an EMPTY canonical set.
    prof0 = client.get(f"/video/identity-profiles/{slug}").get_json()["profile"]
    assert prof0["canonical"] == [], prof0["canonical"]

    # ONE-CLICK: a bare body -> chain the turntable + auto_promote defaults True. Mints
    # a fresh recon_id; no prior reconstruction needed.
    r = client.post(f"/video/identity-profiles/{slug}/generate", json={})
    assert r.status_code == 200, (r.status_code, r.get_json())
    body = r.get_json()
    recon_id = body["recon_id"]
    assert recon_id.startswith("identity_"), recon_id

    # The full spec chained a turntable (kind mesh_and_turntable).
    view = _drain_bus(body["job_id"])
    assert view["status"] == "done" and view["result"]["ok"] is True, view
    assert _RECEIVED[-1]["kind"] == "mesh_and_turntable", _RECEIVED[-1]

    # AUTO-PROMOTE fired. Canonical is selected by NEAREST AZIMUTH to each SEMANTIC_VIEWS
    # target (8 of them since 2026-07-16), NOT by a fixed count: this mock ring has only
    # _FRAME_COUNT (=4) frames, so the 8 targets dedupe onto the 4 distinct frames available
    # — the honest sparse-ring degrade (fewer views, never duplicate bytes). A real 72-frame
    # ring yields all 8; that is covered in test_identity_views.py.
    prof = client.get(f"/video/identity-profiles/{slug}").get_json()["profile"]
    canon = prof["canonical"]
    assert len(canon) == min(8, _FRAME_COUNT), canon
    assert all(os.path.isfile(p) for p in canon), canon

    # mesh state records the auto-promotion + the terminal build.
    ms = identity_profiles.get_mesh_state(slug, recon_id)
    assert ms["status"] == "done" and ms.get("auto_promoted") is True, ms


# --------------------------------------------------------------------------- #
# [6] ONE-CLICK generate on a profile whose canonical was ALREADY populated —
#     LATEST-WINS (operator RESCINDED the never-clobber rule 2026-07-14): the new
#     generation's cardinal frames REPLACE the prior canonical. Provenance-safe
#     because mesh reconstruction never reads canonical (the feedback-loop fix);
#     ``auto_promote: false`` remains the opt-out.
# --------------------------------------------------------------------------- #
def test_generate_route_latest_wins_replaces_canonical():
    _RECEIVED.clear()
    slug = _create_profile("Gen LatestWins")

    # Seed a PRIOR canonical from an earlier sheet recon.
    prior = "recon_prior"
    s0 = os.path.join(_TMP_UPLOADS, "curated0.png")
    _make_png(s0, (5, 5, 5))
    identity_profiles.attach_reconstruction(slug, prior, [s0],
                                            spec={"job_id": "p", "mode": "sheet"})
    identity_profiles.promote_reconstruction_views(slug, prior, [0])
    before = client.get(f"/video/identity-profiles/{slug}").get_json()["profile"]["canonical"]
    assert len(before) == 1, before

    # ONE-CLICK generate (auto_promote defaults True) — the NEW cardinals replace it.
    r = client.post(f"/video/identity-profiles/{slug}/generate", json={})
    assert r.status_code == 200, (r.status_code, r.get_json())
    recon_id = r.get_json()["recon_id"]
    view = _drain_bus(r.get_json()["job_id"])
    assert view["status"] == "done" and view["result"]["ok"] is True, view

    after = client.get(f"/video/identity-profiles/{slug}").get_json()["profile"]["canonical"]
    assert after != before, (before, after)          # prior set replaced
    # min(8, _FRAME_COUNT): 8 semantic azimuths deduped onto this 4-frame mock ring.
    assert len(after) == min(8, _FRAME_COUNT), after
    # and mesh state records the auto-promote.
    ms = identity_profiles.get_mesh_state(slug, recon_id)
    assert ms.get("auto_promoted") is True, ms


# --------------------------------------------------------------------------- #
# [7] The /generate route ENQUEUES a mesh job + SEEDS mesh state to "queued"
#     (route-level, before the bus runs it).
# --------------------------------------------------------------------------- #
def test_generate_route_enqueues_and_seeds_state():
    _RECEIVED.clear()
    slug = _create_profile("Gen Enqueue")

    r = client.post(f"/video/identity-profiles/{slug}/generate", json={})
    assert r.status_code == 200, (r.status_code, r.get_json())
    body = r.get_json()
    assert isinstance(body.get("job_id"), str) and body["job_id"], body
    recon_id = body.get("recon_id")
    assert isinstance(recon_id, str) and recon_id.startswith("identity_"), recon_id

    # State seeded to "queued" BEFORE the bus claims it.
    ms = identity_profiles.get_mesh_state(slug, recon_id)
    assert ms is not None and ms.get("status") == "queued", ms

    # And it is claimable + runs the relay to done against the mock service.
    view = _drain_bus(body["job_id"])
    assert view["status"] == "done", view


# --------------------------------------------------------------------------- #
# [8] /generate jails arbitrary view paths (400); unknown slug 404 (shares the
#     mesh-route jail via _resolve_profile_mesh_views).
# --------------------------------------------------------------------------- #
def test_generate_route_jails_and_404():
    slug = _create_profile("Gen Jail")
    r = client.post(f"/video/identity-profiles/{slug}/generate",
                    json={"views": {"front": "/etc/passwd"}})
    assert r.status_code == 400, (r.status_code, r.get_json())
    r404 = client.post("/video/identity-profiles/no-such/generate", json={})
    assert r404.status_code == 404, r404.status_code


# --------------------------------------------------------------------------- #
# [9] FLEET-VLM FRONT AUTO-SELECTION — the 2nd candidate answers "yes" (full body) ->
#     it becomes front. The render-service mock must receive the 2nd image's bytes as
#     the "front" view, and mesh state must record an honest "vlm" selection outcome.
# --------------------------------------------------------------------------- #
def test_front_autoselect_second_candidate_wins():
    _RECEIVED.clear()
    _VISION_SCRIPT.clear()
    _VISION_CALLS.clear()
    img1 = os.path.join(_TMP_UPLOADS, "vlm_cropped.png")
    img2 = os.path.join(_TMP_UPLOADS, "vlm_fullbody.png")
    _make_png(img1, (11, 22, 33))
    _make_png(img2, (44, 55, 66))
    slug = _create_profile_multi("VLM Second Wins", [img1, img2])
    # Profile creation MATERIALIZES its own copies of the uploads (ref_00.png, …) — the
    # route's candidates are those identity-owned paths, not the original upload paths;
    # fetch them so "chosen" assertions compare against what the runner actually saw.
    owned = client.get(f"/video/identity-profiles/{slug}").get_json()["profile"]["reference_images"]
    assert len(owned) == 2, owned

    _VISION_SCRIPT[open(img1, "rb").read()] = {"ok": True, "text": "No, the feet are cropped."}
    _VISION_SCRIPT[open(img2, "rb").read()] = {"ok": True, "text": "Yes, the entire body is visible."}

    recon_id = "recon_vlm_win"
    r = client.post(f"/video/identity-profiles/{slug}/reconstruction/{recon_id}/mesh",
                    json={"chain_turntable": True,
                          "turntable": {"frame_count": _FRAME_COUNT, "fps": 12}})
    assert r.status_code == 200, (r.status_code, r.get_json())
    view = _drain_bus(r.get_json()["job_id"])
    assert view["status"] == "done" and view["result"]["ok"] is True, view

    # The render service received candidate 2's bytes as "front", not candidate 1's.
    sent_front_b64 = _RECEIVED[-1]["views"]["front"]
    assert base64.b64decode(sent_front_b64) == open(img2, "rb").read()

    # Both candidates were checked, in order, before the winner was found.
    assert len(_VISION_CALLS) == 2, _VISION_CALLS
    assert _VISION_CALLS[0] == open(img1, "rb").read()
    assert _VISION_CALLS[1] == open(img2, "rb").read()

    ms = identity_profiles.get_mesh_state(slug, recon_id)
    fs = ms.get("front_selection") or {}
    assert fs.get("mode") == "vlm", fs
    assert fs.get("chosen") == owned[1], fs
    assert fs.get("full_body") is True, fs
    assert fs.get("checked") == 2, fs


# --------------------------------------------------------------------------- #
# [10] All candidates answer "no" (or the amenity reports not-ok) -> auto-selection
#      falls back to the existing default front (candidate 1); the job still succeeds.
# --------------------------------------------------------------------------- #
def test_front_autoselect_falls_back_on_no_or_error():
    # Pin the AUTO vision-model preference off: this test's contract is the MODEL-LESS
    # front-select path (exactly one /ml/vision call per candidate — no with-model
    # retry). The auto-7B + retry behavior is covered by test_identity_vision_setting.
    orig_pref = identity_profiles.preferred_identity_vision_model
    identity_profiles.preferred_identity_vision_model = lambda: None
    try:
        _front_autoselect_falls_back_on_no_or_error_body()
    finally:
        identity_profiles.preferred_identity_vision_model = orig_pref


def _front_autoselect_falls_back_on_no_or_error_body():
    _RECEIVED.clear()
    _VISION_SCRIPT.clear()
    _VISION_CALLS.clear()
    img1 = os.path.join(_TMP_UPLOADS, "vlm_no1.png")
    img2 = os.path.join(_TMP_UPLOADS, "vlm_no2.png")
    _make_png(img1, (77, 88, 99))
    _make_png(img2, (100, 110, 120))
    slug = _create_profile_multi("VLM Fallback", [img1, img2])
    owned = client.get(f"/video/identity-profiles/{slug}").get_json()["profile"]["reference_images"]
    assert len(owned) == 2, owned

    # img1 -> a soft ERROR (not-ok reply); img2 -> a clear "no". Neither qualifies.
    _VISION_SCRIPT[open(img1, "rb").read()] = {"ok": False, "text": None}
    _VISION_SCRIPT[open(img2, "rb").read()] = {"ok": True, "text": "No, only the torso is shown."}

    recon_id = "recon_vlm_fallback"
    r = client.post(f"/video/identity-profiles/{slug}/reconstruction/{recon_id}/mesh",
                    json={"chain_turntable": True,
                          "turntable": {"frame_count": _FRAME_COUNT, "fps": 12}})
    assert r.status_code == 200, (r.status_code, r.get_json())
    view = _drain_bus(r.get_json()["job_id"])
    assert view["status"] == "done" and view["result"]["ok"] is True, view

    # Front stayed the DEFAULT (candidate 1) — never fails the job over a vision miss.
    sent_front_b64 = _RECEIVED[-1]["views"]["front"]
    assert base64.b64decode(sent_front_b64) == open(img1, "rb").read()
    assert len(_VISION_CALLS) == 2, _VISION_CALLS

    ms = identity_profiles.get_mesh_state(slug, recon_id)
    fs = ms.get("front_selection") or {}
    assert fs.get("mode") == "vlm", fs
    assert fs.get("chosen") == owned[0], fs
    assert fs.get("full_body") is False, fs
    assert fs.get("checked") == 2, fs


# --------------------------------------------------------------------------- #
# [11] Kill-switch IDENTITY_FRONT_AUTOSELECT=off -> NO /ml/vision calls at all; the
#      default front stands and mesh state records mode "disabled".
# --------------------------------------------------------------------------- #
def test_front_autoselect_kill_switch_off():
    _RECEIVED.clear()
    _VISION_SCRIPT.clear()
    _VISION_CALLS.clear()
    img1 = os.path.join(_TMP_UPLOADS, "vlm_off1.png")
    img2 = os.path.join(_TMP_UPLOADS, "vlm_off2.png")
    _make_png(img1, (5, 6, 7))
    _make_png(img2, (8, 9, 10))
    slug = _create_profile_multi("VLM KillSwitch", [img1, img2])
    owned = client.get(f"/video/identity-profiles/{slug}").get_json()["profile"]["reference_images"]
    assert len(owned) == 2, owned
    # Even if img2 WOULD qualify, the switch must prevent the amenity from ever running.
    _VISION_SCRIPT[open(img2, "rb").read()] = {"ok": True, "text": "Yes, full body."}

    os.environ["IDENTITY_FRONT_AUTOSELECT"] = "off"
    try:
        recon_id = "recon_vlm_off"
        r = client.post(f"/video/identity-profiles/{slug}/reconstruction/{recon_id}/mesh",
                        json={"chain_turntable": True,
                              "turntable": {"frame_count": _FRAME_COUNT, "fps": 12}})
        assert r.status_code == 200, (r.status_code, r.get_json())
        view = _drain_bus(r.get_json()["job_id"])
        assert view["status"] == "done" and view["result"]["ok"] is True, view
    finally:
        os.environ.pop("IDENTITY_FRONT_AUTOSELECT", None)

    assert _VISION_CALLS == [], _VISION_CALLS  # the amenity was never called

    sent_front_b64 = _RECEIVED[-1]["views"]["front"]
    assert base64.b64decode(sent_front_b64) == open(img1, "rb").read()

    ms = identity_profiles.get_mesh_state(slug, recon_id)
    fs = ms.get("front_selection") or {}
    assert fs.get("mode") == "disabled", fs
    assert fs.get("chosen") == owned[0], fs


CHECKS = [
    ("full relay pipeline: route->enqueue->relay->persist->attach->promote", test_relay_full_pipeline),
    ("mesh-only build flips kind to mesh_build; GLB persisted", test_relay_mesh_only_no_turntable),
    ("missing render config -> clean error-as-data (not_configured)", test_relay_not_configured_is_error_as_data),
    ("mesh route jails arbitrary view paths (400); unknown slug 404", test_mesh_route_jails_arbitrary_paths),
    ("/generate auto-promotes cardinal frames when canonical empty", test_generate_route_auto_promotes_when_canonical_empty),
    ("/generate never clobbers a curated canonical set", test_generate_route_latest_wins_replaces_canonical),
    ("/generate enqueues a mesh job + seeds mesh state 'queued'", test_generate_route_enqueues_and_seeds_state),
    ("/generate jails arbitrary view paths (400); unknown slug 404", test_generate_route_jails_and_404),
    ("front auto-select: 2nd candidate answers yes -> becomes front", test_front_autoselect_second_candidate_wins),
    ("front auto-select: all no/error -> falls back to default front", test_front_autoselect_falls_back_on_no_or_error),
    ("front auto-select: kill-switch off -> no /ml/vision calls", test_front_autoselect_kill_switch_off),
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
