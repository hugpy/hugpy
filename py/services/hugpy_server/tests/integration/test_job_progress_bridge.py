"""Live per-stage progress + rolling log tail + HONEST stalled on /llm/jobs.

The goal (keeper 2026-07-14): a wedged identity render must read as STUCK on GET
/llm/jobs — carrying its stage, a rolling log tail, and an honest ``stalled``
flag — instead of green-active with ``progress: 0.0`` / ``message: ""``. The
seam is:

    identity_render_relay poll  ->  media_bus.set_progress(job_id, blob)
      ->  job_bridge.on_progress  ->  comms.JobStore.update  ->  GET /llm/jobs

This suite exercises BOTH halves:

  * PURE comms units (no server): the Job surface carries stage/log_tail;
    ``stalled`` is COMPUTED fresh (active + forward-progress silence), not a stale
    stored bool; ``progressed_at`` advances on real movement (progress/stage/token)
    but NOT on a log-tail-only write.
  * INTEGRATION (a localhost mock render service, no GPU / no real network): the
    RELAY reads a poll body's stage/progress/log_tail into the media bus, and it
    reaches the comms Job — and it DEGRADES gracefully when the service omits the
    additive fields (an older service).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hugpy_control.jobs import Job, JobStore, _compute_stalled, LOG_TAIL_CAP
from hugpy_video.intel import job_bridge

logging.disable(logging.INFO)


# =========================================================================== #
# PART A — pure comms units (no server).
# =========================================================================== #
def test_to_dict_carries_stage_and_log_tail():
    j = Job(id="a1", status="processing", stage="meshing",
            log_tail=["boot", "loading model", "step 3/10"])
    d = j.to_dict()
    assert d["stage"] == "meshing", d
    assert d["log_tail"] == ["boot", "loading model", "step 3/10"], d
    assert "progressed_at" in d, d
    # Backward-compatible defaults: a plain chat job carries "" / [].
    d0 = Job(id="a0").to_dict()
    assert d0["stage"] == "" and d0["log_tail"] == [], d0


def test_log_tail_capped_at_serialize():
    lines = [f"line {i}" for i in range(200)]
    d = Job(id="a2", status="processing", log_tail=lines).to_dict()
    assert len(d["log_tail"]) == LOG_TAIL_CAP, len(d["log_tail"])
    # Newest-last: the cap keeps the TAIL (the most recent lines).
    assert d["log_tail"][-1] == "line 199", d["log_tail"][-1]
    assert d["log_tail"][0] == f"line {200 - LOG_TAIL_CAP}", d["log_tail"][0]


def test_stalled_true_when_active_and_progress_silent():
    j = Job(id="a3", status="processing")
    j.progressed_at = time.time() - 200          # silent well past the 90s default
    assert j.to_dict()["stalled"] is True, j.to_dict()


def test_stalled_false_when_fresh():
    j = Job(id="a4", status="processing")
    j.progressed_at = time.time()                # just moved
    assert j.to_dict()["stalled"] is False, j.to_dict()


def test_stalled_false_when_terminal():
    j = Job(id="a5", status="done")
    j.progressed_at = time.time() - 5000         # old, but DONE is never "stalled"
    assert j.to_dict()["stalled"] is False, j.to_dict()


def test_stalled_false_when_pending():
    # A job waiting its turn in the queue is starved, not wedged.
    j = Job(id="a5b", status="pending")
    j.progressed_at = time.time() - 5000
    assert j.to_dict()["stalled"] is False, j.to_dict()


def test_stalled_env_threshold_override(monkeypatch):
    j = Job(id="a6", status="streaming")
    j.progressed_at = time.time() - 30
    monkeypatch.setenv("HUGPY_JOB_STALL_SECONDS", "10")   # 30s silence now exceeds 10s
    assert j.to_dict()["stalled"] is True, j.to_dict()
    monkeypatch.setenv("HUGPY_JOB_STALL_SECONDS", "120")  # ...but not 120s
    assert j.to_dict()["stalled"] is False, j.to_dict()


def test_stalled_explicit_bool_still_honored():
    # A download writer that set self.stalled=True keeps reading stalled even
    # though progressed_at is fresh (the OR half of the honest computation).
    j = Job(id="a7", status="processing", stalled=True)
    j.progressed_at = time.time()
    assert j.to_dict()["stalled"] is True, j.to_dict()


def test_compute_stalled_is_fail_open():
    now = time.time()
    assert _compute_stalled("processing", None, now) is False          # no ts
    assert _compute_stalled("processing", "garbage", now) is False     # bad ts
    assert _compute_stalled("processing", now - 5000, now) is True     # old + active


def test_update_advances_progressed_at_on_movement_not_on_logs():
    store = JobStore()                                   # mirror-less (no arg)
    store.create(id="u1", kind="download", status="processing")
    store.update("u1", progress=0.2)
    p1 = store.get("u1").progressed_at
    time.sleep(0.01)

    # A log-tail-only write is NOT forward progress -> the stall clock holds.
    store.update("u1", log_tail=["still trying...", "still trying..."])
    assert store.get("u1").progressed_at == p1, "log-only write bumped the clock"

    # A numeric progress advance IS movement.
    time.sleep(0.01)
    store.update("u1", progress=0.5)
    p2 = store.get("u1").progressed_at
    assert p2 > p1, (p1, p2)

    # A stage change IS movement (progress unchanged).
    time.sleep(0.01)
    store.update("u1", stage="finalizing")
    assert store.get("u1").progressed_at > p2

    # A non-advancing progress (same value) does NOT reset the clock.
    p3 = store.get("u1").progressed_at
    time.sleep(0.01)
    store.update("u1", progress=0.5)
    assert store.get("u1").progressed_at == p3, "flat progress bumped the clock"


def test_update_caps_log_tail_on_write():
    store = JobStore()
    store.create(id="u2", kind="media", status="processing")
    store.update("u2", log_tail=[f"l{i}" for i in range(500)])
    assert len(store.get("u2").log_tail) == LOG_TAIL_CAP


def test_on_output_bumps_progressed_at():
    store = JobStore()
    store.create(id="u3", kind="chat", status="processing")
    store.get("u3").progressed_at = time.time() - 500     # pretend it went quiet
    store.on_output("u3", 1)                              # a token arrives
    assert store.get("u3").to_dict()["stalled"] is False, "a token left it stalled"


def test_on_progress_bridge_stamps_comms_job(monkeypatch):
    """job_bridge.on_progress pulls stage/progress/log_tail out of a runner's blob
    and mirrors them (plus a short message) into the comms Job."""
    store = JobStore()
    store.create(id="b1", kind="identity_mesh_build", status="processing")
    monkeypatch.setattr(job_bridge, "_store", lambda: store)

    job_bridge.on_progress("b1", {
        "source": "identity_render", "remote_updated": 123.0,
        "stage": "turntable", "progress": 0.42,
        "log_tail": ["render frame 3", "render frame 4"],
    })
    d = store.get("b1").to_dict()
    assert d["stage"] == "turntable", d
    assert abs(d["progress"] - 0.42) < 1e-6, d
    assert d["log_tail"] == ["render frame 3", "render frame 4"], d
    assert d["message"] == "turntable 42%", d


def test_on_progress_graceful_on_empty_or_unknown(monkeypatch):
    store = JobStore()
    store.create(id="b2", kind="identity_mesh_build", status="processing",
                 message="prior")
    monkeypatch.setattr(job_bridge, "_store", lambda: store)

    # A blob with no recognized keys mirrors NOTHING (message left intact).
    job_bridge.on_progress("b2", {"source": "identity_render", "cpu": 0.9})
    assert store.get("b2").message == "prior", store.get("b2").message
    assert store.get("b2").stage == "", store.get("b2").stage

    # A non-dict, and an unknown id, are safe no-ops (never raise).
    job_bridge.on_progress("b2", None)                # type: ignore[arg-type]
    job_bridge.on_progress("no-such-id", {"stage": "x", "progress": 0.1})


# =========================================================================== #
# PART B — integration: the RELAY reads a poll body into the media bus, and it
# reaches the comms Job (and degrades when the service omits the fields).
#
# Store + bus isolation mirrors test_identity_render_relay.py (module-global
# rebind — env isolation does not work since constants read the .env file).
# The comms store is a fresh per-module JobStore handed to the bridge, so the
# test IS the owner process and snapshot() reads the local record.
# =========================================================================== #
_TOKEN = "prog-render-token-xyz"
_FRAME_COUNT = 4
_FILES = ["identity.glb", "identity_mesh.json", "turntable.mp4"] + [
    f"frames/frame_{i:04d}.png" for i in range(_FRAME_COUNT)
]

# Per-remote-id poll counter + the body served on the FIRST poll. A test sets
# "running_body" to a running-with-progress dict (or None to jump straight to
# done) and clears "poll_counts" before draining.
_SVC = {"poll_counts": {}, "running_body": None}


def _png_bytes(color=(120, 40, 200)) -> bytes:
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


def _file_bytes(name: str) -> bytes:
    if name.endswith(".png"):
        return _png_bytes()
    if name.endswith(".glb"):
        return b"glTF\x02\x00\x00\x00FAKE"
    if name.endswith(".json"):
        return json.dumps({"identity_id": "id"}).encode()
    if name.endswith(".mp4"):
        return b"\x00\x00\x00\x18ftypmp42FAKE"
    return b""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
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
        if not self._authed():
            return self._json(401, {"error": "bad token"})
        if self.path.startswith("/jobs/") and "/files/" in self.path:
            data = _file_bytes(self.path.split("/files/", 1)[1])
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path.startswith("/jobs/"):
            job = self.path.rsplit("/", 1)[1]
            n = _SVC["poll_counts"].get(job, 0) + 1
            _SVC["poll_counts"][job] = n
            # First poll -> the (optional) running body; thereafter -> done.
            if n == 1 and _SVC["running_body"] is not None:
                return self._json(200, {"job_id": job, "status": "running",
                                        **_SVC["running_body"]})
            return self._json(200, {"job_id": job, "status": "done", "files": _FILES})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._authed():
            return self._json(401, {"error": "bad token"})
        if self.path == "/jobs":
            n = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(n)
            return self._json(202, {"job_id": "remote-prog-1"})
        return self._json(404, {"error": "not found"})

    def do_DELETE(self):
        return self._json(200, {"ok": True})


def _make_png(path: str) -> None:
    from PIL import Image
    Image.new("RGB", (64, 64), (200, 40, 40)).save(path)


@pytest.fixture(scope="module")
def render_service():
    """Isolated identities/projects/uploads trees, a throwaway media-bus DB, a
    fresh comms JobStore behind the bridge, a fast poll interval, and the
    localhost mock render service. No explicit front views -> the route/runner
    would try the fleet-VLM amenity; the tests build the spec DIRECTLY with an
    explicit front (view_sources) so no vision call happens (candidates<2 ->
    mode "default"), keeping this network-free."""
    from hugpy_video.intel import identity_profiles
    from hugpy_video.intel import media_bus
    from hugpy_video.intel.runners import identity_render_client, identity_render_relay
    from hugpy_platform.constants import UPLOADS_HOME, DEFAULT_ROOT

    mp = pytest.MonkeyPatch()
    scratch = os.path.join(DEFAULT_ROOT, "video_intel", "_scratch")
    os.makedirs(scratch, exist_ok=True)
    os.makedirs(UPLOADS_HOME, exist_ok=True)
    tmp_identities = tempfile.mkdtemp(prefix="hugpy-prog-store-", dir=scratch)
    tmp_projects = tempfile.mkdtemp(prefix="hugpy-prog-projects-")
    tmp_uploads = tempfile.mkdtemp(prefix="hugpy-prog-uploads-", dir=UPLOADS_HOME)
    mp.setattr(identity_profiles, "IDENTITIES_HOME", tmp_identities)
    mp.setattr(identity_profiles, "PROJECTS_HOME", tmp_projects)

    tmp_db = tempfile.mkstemp(prefix="prog-bus-", suffix=".db")[1]
    mp.setattr(media_bus, "DB_PATH", tmp_db)
    mp.setattr(media_bus, "_initialized", False)
    with sqlite3.connect(tmp_db) as _c:
        _c.execute(
            "CREATE TABLE IF NOT EXISTS media_jobs ("
            "job_id TEXT PRIMARY KEY, name TEXT, status TEXT, spec_json TEXT, "
            "result_json TEXT, claim_token TEXT, created REAL, updated REAL, "
            "progress_json TEXT)")

    # Purely per-process comms store: the test IS the owner process.
    store = JobStore()
    mp.setattr(job_bridge, "_store", lambda: store)

    # Shrink the poll interval so the multi-poll (running -> done) integration
    # never actually sleeps 5s.
    mp.setattr(identity_render_client, "POLL_INTERVAL_S", 0.01)
    mp.setattr(identity_render_relay, "_POLL_INTERVAL_S", 0.01, raising=False)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    mp.setenv("IDENTITY_RENDER_URL", f"http://127.0.0.1:{port}")
    mp.setenv("IDENTITY_RENDER_TOKEN", _TOKEN)

    img = os.path.join(tmp_uploads, "front.png")
    _make_png(img)

    try:
        yield {"store": store, "img": img, "identity_profiles": identity_profiles,
               "media_bus": media_bus}
    finally:
        server.shutdown()
        mp.undo()
        for d in (tmp_identities, tmp_projects, tmp_uploads):
            shutil.rmtree(d, ignore_errors=True)
        try:
            os.remove(tmp_db)
        except OSError:
            pass


def _profile_and_spec(svc, slug_seed: str):
    """Create a profile (so persistence has a home) + a mesh spec whose front is
    the profile's own materialized reference. Returns (slug, recon_id, spec)."""
    from hugpy_video.intel.identity_reconstruction_schema import make_identity_mesh
    identity_profiles = svc["identity_profiles"]
    prof = identity_profiles.create_profile(name=slug_seed, source_images=[svc["img"]])
    slug = prof["slug"]
    ref0 = identity_profiles.get_profile(slug)["reference_images"][0]
    recon_id = "recon_prog_1"
    spec = make_identity_mesh(
        slug=slug, recon_id=recon_id,
        view_sources=[("front", ref0)],
        chain_turntable=True,
        frame_count=_FRAME_COUNT,
    )
    return slug, recon_id, spec


@pytest.fixture
def set_progress_calls(render_service, monkeypatch):
    """Wrap media_bus.set_progress to record every blob the RELAY hands it, while
    still running the real thing (so the bridge into comms fires). The runner
    imports set_progress lazily at call time, so patching the module attribute
    takes effect for the very next run."""
    media_bus = render_service["media_bus"]
    calls: list = []
    orig = media_bus.set_progress

    def _spy(job_id, progress):
        calls.append((job_id, dict(progress)))
        return orig(job_id, progress)

    monkeypatch.setattr(media_bus, "set_progress", _spy)
    return calls


def test_relay_stamps_pbody_progress_into_llm_jobs(render_service, set_progress_calls):
    """The relay reads a poll body's stage/progress/log_tail/updated into the
    media bus, and it reaches the comms Job that GET /llm/jobs serializes."""
    media_bus = render_service["media_bus"]
    _SVC["poll_counts"].clear()
    _SVC["running_body"] = {"stage": "meshing", "progress": 0.4,
                            "log_tail": ["load hy3dgen", "octree pass 1", "octree pass 2"],
                            "updated": 1720000000.0}
    _slug, _recon, spec = _profile_and_spec(render_service, "Prog Stamp")
    job_id = media_bus.enqueue("identity_mesh_build", spec)
    processed = media_bus.work_once("prog-worker")
    assert processed == job_id, (processed, job_id)

    # The RELAY built the blob from the poll body (proves it reads the fields).
    calls = set_progress_calls
    assert calls, "the relay never stamped progress"
    _jid, blob = calls[0]
    assert blob["stage"] == "meshing", blob
    assert blob["progress"] == 0.4, blob
    assert blob["log_tail"] == ["load hy3dgen", "octree pass 1", "octree pass 2"], blob
    assert blob["source"] == "identity_render", blob
    assert blob["remote_updated"] == 1720000000.0, blob

    # And it REACHED the comms Job (terminal-retained; stage/log_tail persist
    # past the done write, which only sets status + the artifact uri message).
    rows = render_service["store"].snapshot(live_only=False)
    row = next((r for r in rows if r["id"] == job_id), None)
    assert row is not None, [r["id"] for r in rows]
    assert row["stage"] == "meshing", row
    assert row["log_tail"] == ["load hy3dgen", "octree pass 1", "octree pass 2"], row
    assert row["status"] == "done", row


def test_relay_degrades_when_service_omits_progress_fields(render_service, set_progress_calls):
    """An older render service that omits stage/progress/log_tail: the relay
    stamps NOTHING (no blob), never raises, and the job still completes."""
    media_bus = render_service["media_bus"]
    _SVC["poll_counts"].clear()
    # A running poll with NO additive fields (older service shape).
    _SVC["running_body"] = {}
    _slug, _recon, spec = _profile_and_spec(render_service, "Prog Degrade")
    job_id = media_bus.enqueue("identity_mesh_build", spec)
    processed = media_bus.work_once("prog-worker-2")
    assert processed == job_id, (processed, job_id)

    assert set_progress_calls == [], \
        f"relay stamped despite no progress fields: {set_progress_calls}"
    view = media_bus.get(job_id)
    assert view["status"] == "done", view
    rows = render_service["store"].snapshot(live_only=False)
    row = next((r for r in rows if r["id"] == job_id), None)
    assert row is not None and row["status"] == "done", row
    assert row["stage"] == "", row          # nothing stamped -> default
    assert row["log_tail"] == [], row
