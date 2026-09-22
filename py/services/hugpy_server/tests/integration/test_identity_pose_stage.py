"""T-POSE POSE-NORMALIZATION RENDER STAGE (IDENTITY-VERSIONS-SLICE.md slice 5).

Hunyuan3D meshes the INPUT pose: crossed arms occlude the torso and leave "the unknown
below them" (operator's words about luigi). The fix is 2D pose normalization BEFORE
meshing — render ONE identity-locked T-pose STILL (arms out, full body, facing camera)
on the Wan-VACE id_lock path and mesh THAT as the front. This test locks the whole
contract WITHOUT a GPU and WITHOUT any network beyond localhost:

  (a) CAPABLE + SUCCESS — when the pose-render seam (identity_render_relay._render_pose_front)
      returns a still, the relay uses it as the mesh FRONT (the POST /jobs body's front
      view is the rendered T-pose, NOT the source photo), records front_selection.mode
      "t-pose" and pose_stage.applied True, and the job still succeeds end-to-end.
  (b) RENDER FAILURE — when the seam returns None (worker offline / render error /
      timeout), the relay FALLS BACK to the existing front-select flow, the job STILL
      succeeds (honest degrade — a failed pose render never fails the mesh job), and
      mesh state records pose_stage.applied False with a reason.
  (c) POSE "none" — a build with pose="none" (or absent) NEVER attempts a pose render
      (the seam is asserted un-called) and carries NO pose_stage in mesh state — the
      relay is byte-identical to a pre-slice-5 build.
  (d) ROUTE not-capable notice — with _pose_stage_capable monkeypatched False (today's
      real fleet posture unless a studio worker is up), POST /generate with pose="t-pose"
      still returns 200, the enqueued spec carries pose="none" (the build proceeds off the
      normal front), and the response's structured `pose` notice says applied False.
      With it monkeypatched True, the spec carries pose="t-pose" and the notice says
      applied True.

Isolation mirrors test_identity_render_relay.py EXACTLY (a threaded http.server stands in
for the render service AND /ml/vision; the store + media bus are rebound to temp dirs/DB —
env isolation does NOT work since constants read the .env file). Each check is independent
so one failure never masks the rest.

Run (both as pytest and as a script; run ALONE — the identity test family cross-pollutes
when co-run via an import-time IDENTITIES_HOME rebind):
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python -m pytest tests/test_identity_pose_stage.py -q
  venv/bin/python tests/test_identity_pose_stage.py
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

from flask import Flask  # noqa: E402

from hugpy_video.intel import identity_profiles
from hugpy_video.intel import media_bus
from hugpy_video.intel.runners import identity_render_relay
from hugpy_video.intel.identity_reconstruction_schema import make_identity_mesh
from hugpy_platform.constants import UPLOADS_HOME, DEFAULT_ROOT

# --------------------------------------------------------------------------- #
# STORE + BUS isolation (mirrors test_identity_render_relay.py exactly).
# --------------------------------------------------------------------------- #
_TMP_IDENTITIES = tempfile.mkdtemp(prefix="hugpy-pose-store-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
identity_profiles.IDENTITIES_HOME = _TMP_IDENTITIES
_TMP_PROJECTS = tempfile.mkdtemp(prefix="hugpy-pose-projects-")
identity_profiles.PROJECTS_HOME = _TMP_PROJECTS
_TMP_UPLOADS = tempfile.mkdtemp(prefix="hugpy-pose-uploads-", dir=UPLOADS_HOME)

_TMP_DB = tempfile.mkstemp(prefix="pose-bus-", suffix=".db")[1]
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
# The MOCK render service (localhost, threaded) — the FIXED contract, plus a
# /ml/vision stand-in that always answers "no" (so front-select, when it runs,
# keeps the default front — irrelevant to the pose-stage assertions here).
# --------------------------------------------------------------------------- #
_TOKEN = "pose-render-token-xyz"
_FRAME_COUNT = 4
_FILES = ["identity.glb", "identity_mesh.json", "turntable.mp4"] + [
    f"frames/frame_{i:04d}.png" for i in range(_FRAME_COUNT)]
_RECEIVED: list[dict] = []   # captured POST /jobs payloads, for assertions


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
        return json.dumps({"request_id": "r", "identity_id": "id"}).encode()
    if name.endswith(".mp4"):
        return b"\x00\x00\x00\x18ftypmp42FAKE-MP4-BYTES"
    return b""


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
                                    "capabilities": {"mesh": {"ready": True}}})
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
            return self._json(200, {"job_id": job, "status": "done", "files": _FILES})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        # /ml/vision — no auth (mirrors the real amenity's open gate); always "no" so a
        # front-select that DOES run keeps the default front (not what these tests check).
        if self.path == "/ml/vision":
            n = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(n)
            return self._json(200, {"ok": True, "text": "no, the legs are cropped"})
        if not self._authed():
            return self._json(401, {"error": "bad token"})
        if self.path == "/jobs":
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            _RECEIVED.append(payload)
            return self._json(202, {"job_id": "remote-job-pose-1"})
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


_IMG = os.path.join(_TMP_UPLOADS, "hero_pose.png")
_make_png(_IMG, (200, 40, 40))


def _create_profile(name: str) -> str:
    r = client.post("/video/identity-profiles",
                    json={"name": name, "reference_images": [_IMG], "notes": "a knight"})
    assert r.status_code == 201, (r.status_code, r.get_json())
    return r.get_json()["profile"]["slug"]


def _drain_bus(job_id: str) -> dict:
    """Run the claimed job to completion (the relay is synchronous; the mock reports
    done on the first poll) and return the bus view."""
    processed = media_bus.work_once("test-worker")
    assert processed == job_id, (processed, job_id)
    return media_bus.get(job_id)


def _front_ref(slug: str) -> str:
    return client.get(f"/video/identity-profiles/{slug}") \
        .get_json()["profile"]["reference_images"][0]


class _PoseStub:
    """Monkeypatch stand-in for identity_render_relay._render_pose_front. Records every
    call and returns a configured path (a rendered T-pose still) or None (a render
    failure). Installed/removed around a single check so tests never leak state."""

    def __init__(self, returns):
        self.returns = returns
        self.calls: list[tuple] = []

    def __call__(self, refs, seed, slug, job_id, should_cancel=None, **kwargs):
        # **kwargs tolerates the CLEANUP-PROMPT slice's additive cleanup_prompt /
        # negative_prompt kwargs (the relay now forwards them) without this pose-stage
        # stub needing to care — the cleanup wiring is asserted in its own test file.
        self.calls.append((tuple(refs), seed, slug, job_id))
        return self.returns


def _install_pose_stub(returns):
    stub = _PoseStub(returns)
    orig = identity_render_relay._render_pose_front
    identity_render_relay._render_pose_front = stub
    return stub, orig


# --------------------------------------------------------------------------- #
# (a) CAPABLE + SUCCESS — the rendered T-pose still becomes the mesh FRONT.
# --------------------------------------------------------------------------- #
def test_pose_success_front_replaced_mode_tpose():
    _RECEIVED.clear()
    slug = _create_profile("Pose Success")
    recon_id = "recon_pose_ok"

    # The rendered T-pose still (a real file so the relay can read + b64 it as the front).
    tpose_still = os.path.join(_TMP_UPLOADS, "tpose_ok.png")
    _make_png(tpose_still, (5, 200, 30))
    tpose_bytes = open(tpose_still, "rb").read()

    stub, orig = _install_pose_stub(tpose_still)
    try:
        spec = make_identity_mesh(
            slug=slug, recon_id=recon_id,
            view_sources=[("front", _front_ref(slug))],
            chain_turntable=False, pose="t-pose",
        )
        job_id = media_bus.enqueue("identity_mesh_build", spec)
        view = _drain_bus(job_id)
    finally:
        identity_render_relay._render_pose_front = orig

    # The job SUCCEEDED end-to-end.
    assert view["status"] == "done" and view["result"]["ok"] is True, view
    # The pose seam was called exactly once, conditioned on the profile's refs.
    assert len(stub.calls) == 1, stub.calls
    assert _front_ref(slug) in stub.calls[0][0], stub.calls

    # The POST /jobs FRONT view is the RENDERED T-pose still, NOT the source photo.
    import base64
    sent = _RECEIVED[-1]
    assert base64.b64decode(sent["views"]["front"]) == tpose_bytes, \
        "the mesh front should be the rendered T-pose still, not the source photo"

    # Mesh state records the pose stage applied + front_selection mode "t-pose".
    ms = identity_profiles.get_mesh_state(slug, recon_id)
    assert ms["status"] == "done", ms
    assert ms["pose_stage"]["requested"] == "t-pose", ms
    assert ms["pose_stage"]["applied"] is True, ms
    assert ms["pose_stage"]["rendered_front"] == tpose_still, ms
    assert ms["front_selection"]["mode"] == "t-pose", ms


# --------------------------------------------------------------------------- #
# (b) RENDER FAILURE — falls back to front-select; the job STILL succeeds.
# --------------------------------------------------------------------------- #
def test_pose_render_failure_falls_back_job_succeeds():
    _RECEIVED.clear()
    slug = _create_profile("Pose Fallback")
    recon_id = "recon_pose_fail"
    front = _front_ref(slug)
    front_bytes = open(front, "rb").read()

    # The pose seam returns None (worker offline / render error / timeout).
    stub, orig = _install_pose_stub(None)
    try:
        spec = make_identity_mesh(
            slug=slug, recon_id=recon_id,
            view_sources=[("front", front)],
            chain_turntable=False, pose="t-pose",
        )
        job_id = media_bus.enqueue("identity_mesh_build", spec)
        view = _drain_bus(job_id)
    finally:
        identity_render_relay._render_pose_front = orig

    # HONEST DEGRADE: a failed pose render NEVER fails the mesh job.
    assert view["status"] == "done" and view["result"]["ok"] is True, view
    assert len(stub.calls) == 1, stub.calls

    # The mesh front fell back to the NORMAL source photo (front-select default).
    import base64
    sent = _RECEIVED[-1]
    assert base64.b64decode(sent["views"]["front"]) == front_bytes, \
        "on a failed pose render the front should be the source photo (front-select)"

    # Mesh state records the pose stage requested but NOT applied, with a reason.
    ms = identity_profiles.get_mesh_state(slug, recon_id)
    assert ms["status"] == "done", ms
    assert ms["pose_stage"]["requested"] == "t-pose", ms
    assert ms["pose_stage"]["applied"] is False, ms
    assert isinstance(ms["pose_stage"].get("reason"), str) and ms["pose_stage"]["reason"], ms
    # front_selection did NOT flip to "t-pose".
    assert ms["front_selection"]["mode"] != "t-pose", ms


# --------------------------------------------------------------------------- #
# (c) POSE "none" — NO render attempted; byte-identical to a pre-slice-5 build.
# --------------------------------------------------------------------------- #
def test_pose_none_no_render_no_pose_stage():
    _RECEIVED.clear()
    slug = _create_profile("Pose None")
    recon_id = "recon_pose_none"

    # A stub that would RAISE if ever called — proves pose="none" never touches the seam.
    def _boom(*a, **k):
        raise AssertionError("pose render must NOT be attempted for pose='none'")

    orig = identity_render_relay._render_pose_front
    identity_render_relay._render_pose_front = _boom
    try:
        spec = make_identity_mesh(
            slug=slug, recon_id=recon_id,
            view_sources=[("front", _front_ref(slug))],
            chain_turntable=False,  # pose defaults to "none"
        )
        assert spec.pose == "none", spec.pose
        job_id = media_bus.enqueue("identity_mesh_build", spec)
        view = _drain_bus(job_id)
    finally:
        identity_render_relay._render_pose_front = orig

    assert view["status"] == "done" and view["result"]["ok"] is True, view
    # No pose_stage in mesh state at all — a pre-slice-5 build carries none.
    ms = identity_profiles.get_mesh_state(slug, recon_id)
    assert "pose_stage" not in ms, ms
    # front_selection is the ordinary "explicit" (one ref, no candidates) — unchanged.
    assert ms["front_selection"]["mode"] != "t-pose", ms


# --------------------------------------------------------------------------- #
# (d) ROUTE not-capable / capable notice (POST /generate). media_bus.enqueue mocked
#     so no bus job actually runs — we only assert the route's pose gating + notice.
# --------------------------------------------------------------------------- #
def _capture_generate(slug: str, body: dict):
    captured = {}

    def _fake_enqueue(name, spec, **kwargs):   # principal/owner attribution
        captured["name"] = name
        captured["spec"] = spec
        return "job-fake-pose"

    orig = media_bus.enqueue
    media_bus.enqueue = _fake_enqueue
    try:
        r = client.post(f"/video/identity-profiles/{slug}/generate", json=body)
    finally:
        media_bus.enqueue = orig
    return r, captured.get("spec")


def test_route_not_capable_notice_falls_back():
    slug = _create_profile("Route NotCapable")
    orig = vr._pose_stage_capable
    vr._pose_stage_capable = lambda _slug: False
    try:
        r, spec = _capture_generate(slug, {"pose": "t-pose"})
    finally:
        vr._pose_stage_capable = orig
    assert r.status_code == 200, (r.status_code, r.get_json())
    # The build proceeds off the NORMAL front — the enqueued spec's pose is "none".
    assert spec.pose == "none", spec.pose
    # The response carries the structured not-capable notice.
    pose = r.get_json().get("pose")
    assert pose and pose["requested"] == "t-pose", pose
    assert pose["applied"] is False and pose["capable"] is False, pose
    assert pose.get("code") == "pose_stage_unavailable", pose
    assert isinstance(pose.get("message"), str) and pose["message"], pose


def test_route_capable_notice_applies():
    slug = _create_profile("Route Capable")
    orig = vr._pose_stage_capable
    vr._pose_stage_capable = lambda _slug: True
    try:
        r, spec = _capture_generate(slug, {"pose": "t-pose"})
    finally:
        vr._pose_stage_capable = orig
    assert r.status_code == 200, (r.status_code, r.get_json())
    # Capable -> the spec carries pose="t-pose" so the relay renders the T-pose front.
    assert spec.pose == "t-pose", spec.pose
    pose = r.get_json().get("pose")
    assert pose and pose["applied"] is True and pose["capable"] is True, pose


def test_route_pose_none_is_byte_identical():
    """A bare click (pose absent / "none") keeps the exact {job_id, recon_id} shape — NO
    pose block in the response, spec pose "none" — regardless of capability."""
    slug = _create_profile("Route PoseNone")
    r, spec = _capture_generate(slug, {})
    assert r.status_code == 200, (r.status_code, r.get_json())
    assert spec.pose == "none", spec.pose
    assert "pose" not in r.get_json(), r.get_json()


def test_route_invalid_pose_is_400():
    slug = _create_profile("Route BadPose")
    r, _ = _capture_generate(slug, {"pose": "crouch"})
    assert r.status_code == 400, (r.status_code, r.get_json())


CHECKS = [
    ("(a) capable+success -> front replaced, mode 't-pose'", test_pose_success_front_replaced_mode_tpose),
    ("(b) render failure -> falls back, job succeeds, applied False", test_pose_render_failure_falls_back_job_succeeds),
    ("(c) pose 'none' -> no render attempted, no pose_stage", test_pose_none_no_render_no_pose_stage),
    ("(d) route not-capable -> notice applied False, spec pose none", test_route_not_capable_notice_falls_back),
    ("(d) route capable -> notice applied True, spec pose t-pose", test_route_capable_notice_applies),
    ("(d) route pose 'none' -> byte-identical, no pose block", test_route_pose_none_is_byte_identical),
    ("(d) route invalid pose -> 400", test_route_invalid_pose_is_400),
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
