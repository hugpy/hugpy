"""Studio "Send to Editor" (k12) — backend tests.

Covers the new POST /video/studio/clip/<job_id>/to-editor route + its backbone
(video_intel/studio/editor_handoff.py):

  * AUTH (defense in depth) — a VIDEO-SHARE principal passes the blanket /video
    gate but is 403'd by the route's OWN operator_authenticated() check; a valid
    console SESSION and the OPERATOR-TOKEN path are both allowed through (they
    then reach clip resolution -> 404 on an unknown id, proving they got PAST auth).
  * BRANCH DECISION — plan_handoff() on fabricated ffprobe JSON: already
    h264/yuv420p/mp4 -> "copy"; wrong pix_fmt / codec / container / no-video ->
    "transcode". Plus a REAL ffmpeg round-trip of each branch (guarded on ffmpeg).
  * FILENAME SANITIZATION — a genuinely nasty title -> a clean, Windows-legal,
    collision-suffixed filename.
  * 404 / 410 — unknown or incomplete clip -> 404; an archived clip -> 410.
  * FULL ROUTE E2E (guarded on ffmpeg) — an operator sends a real done clip: 200
    with {ok, filename, path, mode:"copy"}, the file lands in the inbox, and a
    RE-SEND is non-destructive (unique_path numeric suffix, prior copy intact).

Written in the studio script style (plain asserts, __main__ guard, numbered PASS
lines) so it runs BOTH as `venv/bin/python tests/test_video_to_editor.py` and via
`pytest tests/test_video_to_editor.py -q`. The ffmpeg-backed checks self-SKIP if
ffmpeg/ffprobe are absent; everything else always runs.

Isolation: media_bus.DB_PATH is repointed to a PRIVATE temp sqlite db (the
_private_bus idiom from test_video_progress_archive) so the real job store is
never touched; the clips/inbox roots are monkeypatched to temp dirs.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logging.disable(logging.INFO)  # silence registry chatter on import

# Deterministic env BEFORE the app modules import.
os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")
os.environ.setdefault(
    "PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-to-editor-test-"))
os.environ.pop("HUGPY_OPERATOR_TOKEN", None)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flask import Flask  # noqa: E402

oa = importlib.import_module("hugpy_server.app.operator_auth")
va = importlib.import_module("hugpy_server.app.video_auth")
vr = importlib.import_module("hugpy_server.app.routes.video_routes")

from hugpy_video.intel import media_bus
from hugpy_video.intel.studio import job as job_mod
from hugpy_video.intel.studio.job import make_studio_i2v
from hugpy_video.intel.studio.editor_handoff import (
    HandoffResult,
    editor_filename,
    plan_handoff,
    send_to_editor,
    _ffprobe,
)

_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _ApiPrefixMiddleware:
    """Mirror wsgi_app.ApiPrefixMiddleware / test_video_gate: strip a leading /api
    BEFORE routing so /api/video/... routes to the bare /video/... rule."""
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path == "/api" or path.startswith("/api/"):
            environ["PATH_INFO"] = path[len("/api"):] or "/"
        return self.wsgi_app(environ, start_response)


def _build_gated_app():
    """The real video_bp behind BOTH production gates (operator + video) and the
    /api-strip middleware — so a request reaches the route body exactly as it does
    in production (the blanket /video gate having already run)."""
    app = Flask(__name__)
    app.register_blueprint(vr.video_bp)
    oa.install_operator_gate(app)
    va.install_video_gate(app)
    app.wsgi_app = _ApiPrefixMiddleware(app.wsgi_app)
    return app


# One gated app/client reused across tests — the gates read env + module internals
# at REQUEST time, so per-test env/monkeypatch changes take effect without a rebuild
# (and we never re-register the blueprint on a second app).
_CLIENT = _build_gated_app().test_client()


def _private_bus():
    """Repoint media_bus.DB_PATH to a private temp db + reset the init flag."""
    tmpdir = tempfile.mkdtemp(prefix="hugpy_to_editor_bus_")
    media_bus.DB_PATH = os.path.join(tmpdir, "media_jobs.db")
    media_bus._initialized = False
    return media_bus, tmpdir


def _as_operator_session():
    """External mode, a valid console session, no share. Caller restores via the
    returned (orig_sess, orig_share) and its own finally."""
    os.environ["HUGPY_AUTH_MODE"] = "external"
    os.environ.pop("HUGPY_OPERATOR_TOKEN", None)
    oa._SESSION_CACHE.clear()
    orig_sess, orig_share = oa._validate_session_external, va._video_share_principal
    oa._validate_session_external = lambda: True
    va._video_share_principal = lambda request: None
    return orig_sess, orig_share


def _restore(orig_sess, orig_share):
    oa._validate_session_external = orig_sess
    va._video_share_principal = orig_share


def _make_clip(path: str, pix_fmt: str = "yuv420p") -> None:
    """A real 1s 128x96 H.264 clip via lavfi testsrc (mirrors the studio suites).
    pix_fmt controls the branch: yuv420p is Filmora-native (copy), yuv444p is not
    (transcode)."""
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i",
         "testsrc=duration=1:size=128x96:rate=8",
         "-c:v", "libx264", "-pix_fmt", pix_fmt, path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def _video_stream(probe: dict) -> dict:
    return next(s for s in probe["streams"] if s.get("codec_type") == "video")


# --------------------------------------------------------------------------- #
# 1) AUTH — defense in depth
# --------------------------------------------------------------------------- #
def test_auth_share_principal_denied():
    """A video-share principal satisfies the blanket /video gate but the route's
    own operator_authenticated() check must STILL 403 it."""
    _private_bus()
    os.environ["HUGPY_AUTH_MODE"] = "external"
    os.environ.pop("HUGPY_OPERATOR_TOKEN", None)
    oa._SESSION_CACHE.clear()
    orig_sess, orig_share = oa._validate_session_external, va._video_share_principal
    try:
        oa._validate_session_external = lambda: False              # no console session
        va._video_share_principal = lambda request: {"scope": "video"}  # a share guest
        r = _CLIENT.post("/api/video/studio/clip/deadbeef/to-editor")
        assert r.status_code == 403, (r.status_code, r.get_data(as_text=True))
        assert "operator" in (r.get_json() or {}).get("error", "").lower()
    finally:
        _restore(orig_sess, orig_share)
    print("[1] PASS  share principal 403'd by the route despite passing the /video gate")


def test_auth_console_session_allowed():
    """A valid console session passes the route's check -> reaches clip resolution
    (unknown id -> 404, NOT 403), proving it got past auth."""
    _private_bus()
    orig_sess, orig_share = _as_operator_session()
    try:
        r = _CLIENT.post("/api/video/studio/clip/unknownjob/to-editor")
        assert r.status_code == 404, (r.status_code, r.get_data(as_text=True))
        assert "no completed studio clip" in (r.get_json() or {}).get("error", "")
    finally:
        _restore(orig_sess, orig_share)
    print("[2] PASS  console session allowed (past auth -> 404 on unknown clip)")


def test_auth_operator_token_allowed():
    """The HUGPY_OPERATOR_TOKEN automation path is accepted; without it the blanket
    gate denies (401) before the route runs."""
    _private_bus()
    os.environ["HUGPY_AUTH_MODE"] = "external"
    os.environ["HUGPY_OPERATOR_TOKEN"] = "s3cret"
    oa._SESSION_CACHE.clear()
    orig_sess, orig_share = oa._validate_session_external, va._video_share_principal
    try:
        oa._validate_session_external = lambda: False
        va._video_share_principal = lambda request: None
        assert _CLIENT.post("/api/video/studio/clip/x/to-editor").status_code == 401
        r = _CLIENT.post("/api/video/studio/clip/x/to-editor",
                         headers={"X-Operator-Token": "s3cret"})
        assert r.status_code == 404, (r.status_code, r.get_data(as_text=True))
    finally:
        os.environ.pop("HUGPY_OPERATOR_TOKEN", None)
        _restore(orig_sess, orig_share)
    print("[3] PASS  operator token accepted (401 without, 404 with)")


# --------------------------------------------------------------------------- #
# 2) BRANCH DECISION — fabricated ffprobe JSON (no ffmpeg needed)
# --------------------------------------------------------------------------- #
def test_branch_decision():
    native = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p"}],
    }
    assert plan_handoff(native) == "copy"
    # wrong pix_fmt
    assert plan_handoff({**native, "streams": [
        {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv444p"}]}) == "transcode"
    # wrong codec
    assert plan_handoff({**native, "streams": [
        {"codec_type": "video", "codec_name": "vp9", "pix_fmt": "yuv420p"}]}) == "transcode"
    # wrong container
    assert plan_handoff({
        "format": {"format_name": "matroska,webm"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p"}],
    }) == "transcode"
    # no video stream
    assert plan_handoff({
        "format": {"format_name": "mov,mp4,m4a"},
        "streams": [{"codec_type": "audio", "codec_name": "aac"}],
    }) == "transcode"
    # empty / missing
    assert plan_handoff({}) == "transcode"
    print("[4] PASS  plan_handoff: copy only for h264/yuv420p/mp4, transcode otherwise")


# --------------------------------------------------------------------------- #
# 3) FILENAME SANITIZATION
# --------------------------------------------------------------------------- #
def test_filename_sanitized():
    nasty = 'a red: lantern <swaying> / "night" \\wind? |cinematic*\t\n'
    fn = editor_filename(nasty, "a7fd8435d6244b9a964b6c4b3db014af")
    for ch in '<>:"/\\|?*':
        assert ch not in fn, (ch, fn)
    assert all(ord(c) >= 32 for c in fn), fn          # no control chars
    assert fn.endswith("_a7fd8435.mp4"), fn           # shortId suffix + ext
    assert not fn[0] in "._- ", fn                    # clean leading char
    assert len(fn) < 120, len(fn)
    # empty / blank / None title -> the "clip" fallback slug
    assert editor_filename(None, "deadbeefcafef00d") == "clip_deadbeef.mp4"
    assert editor_filename("   ", "deadbeefcafef00d") == "clip_deadbeef.mp4"
    # a very long prompt stays bounded (slug capped at 60 chars pre-suffix)
    long_fn = editor_filename("word " * 100, "12345678")
    assert len(long_fn) < 120, len(long_fn)
    print("[5] PASS  editor_filename: Windows-legal, suffixed, bounded, fallback-safe")


# --------------------------------------------------------------------------- #
# 4) 404 / 410 on the route
# --------------------------------------------------------------------------- #
def test_incomplete_clip_404():
    """A queued (never-run) studio job has no result -> 404, not a 200/500."""
    _private_bus()
    jid = media_bus.enqueue(
        "studio_i2v", make_studio_i2v(width=64, height=64, fps=8, prompt="pending"))
    orig_sess, orig_share = _as_operator_session()
    try:
        r = _CLIENT.post(f"/api/video/studio/clip/{jid}/to-editor")
        assert r.status_code == 404, (r.status_code, r.get_data(as_text=True))
        assert "no completed studio clip" in (r.get_json() or {}).get("error", "")
    finally:
        _restore(orig_sess, orig_share)
    print("[6] PASS  incomplete (queued) clip -> 404")


def test_archived_clip_410():
    """An archived clip's bytes survive (never-delete) -> honest 410, not 404."""
    _private_bus()
    jid = media_bus.enqueue(
        "studio_i2v", make_studio_i2v(width=64, height=64, fps=8, prompt="archived"))
    media_bus.archive(jid)
    orig_sess, orig_share = _as_operator_session()
    try:
        r = _CLIENT.post(f"/api/video/studio/clip/{jid}/to-editor")
        assert r.status_code == 410, (r.status_code, r.get_data(as_text=True))
        assert (r.get_json() or {}).get("archived") is True
    finally:
        _restore(orig_sess, orig_share)
    print("[7] PASS  archived clip -> 410")


# --------------------------------------------------------------------------- #
# 5) REAL ffmpeg round-trips (guarded)
# --------------------------------------------------------------------------- #
def test_send_to_editor_copy_roundtrip():
    if not _FFMPEG:
        print("[8] SKIP  ffmpeg/ffprobe not available")
        return
    d = tempfile.mkdtemp(prefix="hugpy_handoff_copy_")
    src = os.path.join(d, "clip.mp4")
    _make_clip(src, pix_fmt="yuv420p")            # already Filmora-native
    dest = os.path.join(d, "out.mp4")
    res = send_to_editor(src, dest)
    assert isinstance(res, HandoffResult) and res.ok, res
    assert res.mode == "copy", res
    assert os.path.isfile(dest) and os.path.getsize(dest) > 0
    v = _video_stream(_ffprobe(dest))
    assert v["codec_name"] == "h264" and v["pix_fmt"] == "yuv420p", v
    print("[8] PASS  copy branch remuxes an h264/yuv420p/mp4 losslessly")


def test_send_to_editor_transcode_roundtrip():
    if not _FFMPEG:
        print("[9] SKIP  ffmpeg/ffprobe not available")
        return
    d = tempfile.mkdtemp(prefix="hugpy_handoff_transcode_")
    src = os.path.join(d, "clip.mp4")
    _make_clip(src, pix_fmt="yuv444p")            # h264 but NOT yuv420p -> transcode
    assert plan_handoff(_ffprobe(src)) == "transcode"
    dest = os.path.join(d, "out.mp4")
    res = send_to_editor(src, dest)
    assert res.ok and res.mode == "transcode", res
    v = _video_stream(_ffprobe(dest))
    assert v["codec_name"] == "h264" and v["pix_fmt"] == "yuv420p", v
    print("[9] PASS  transcode branch re-encodes a non-native clip to h264/yuv420p")


def test_send_to_editor_missing_input():
    res = send_to_editor("/no/such/clip.mp4", "/tmp/never-written.mp4")
    assert not res.ok and res.code == "missing_input", res
    assert not os.path.exists("/tmp/never-written.mp4")
    print("[10] PASS  missing input -> HandoffResult(ok=False, missing_input), no output")


# --------------------------------------------------------------------------- #
# 6) FULL ROUTE E2E — operator sends a real done clip (guarded)
# --------------------------------------------------------------------------- #
def test_route_success_and_resend():
    if not _FFMPEG:
        print("[11] SKIP  ffmpeg/ffprobe not available")
        return
    _private_bus()
    tmp = tempfile.mkdtemp(prefix="hugpy_to_editor_e2e_")
    clips_root = os.path.join(tmp, "clips")
    inbox_root = os.path.join(tmp, "editor-inbox")
    clip_dir = os.path.join(clips_root, "content_hash_abc")
    os.makedirs(clip_dir)
    clip = os.path.join(clip_dir, "clip.mp4")
    _make_clip(clip, pix_fmt="yuv420p")

    jid = media_bus.enqueue(
        "studio_i2v",
        make_studio_i2v(width=128, height=96, fps=8, prompt="a red lantern swaying"))
    # Mark it done with an ok result pointing at the real clip (mirrors a runner's
    # terminal write; media_bus.get() json.loads result_json for the route).
    result_json = json.dumps({
        "job_id": jid, "ok": True,
        "outputs": [{"uri": clip, "mime": "video/mp4"}], "error": None})
    conn = media_bus._connect()
    try:
        conn.execute("UPDATE media_jobs SET status='done', result_json=? WHERE job_id=?",
                     (result_json, jid))
    finally:
        conn.close()

    orig_clips, orig_inbox = job_mod.DEFAULT_CLIPS_ROOT, job_mod.EDITOR_INBOX_ROOT
    job_mod.DEFAULT_CLIPS_ROOT = clips_root
    job_mod.EDITOR_INBOX_ROOT = inbox_root
    os.environ["HUGPY_AUTH_MODE"] = "external"
    os.environ["HUGPY_OPERATOR_TOKEN"] = "s3cret"
    oa._SESSION_CACHE.clear()
    orig_sess, orig_share = oa._validate_session_external, va._video_share_principal
    try:
        oa._validate_session_external = lambda: False
        va._video_share_principal = lambda request: None
        hdr = {"X-Operator-Token": "s3cret"}

        r1 = _CLIENT.post(f"/api/video/studio/clip/{jid}/to-editor", headers=hdr)
        assert r1.status_code == 200, (r1.status_code, r1.get_data(as_text=True))
        b1 = r1.get_json()
        assert b1["ok"] is True and b1["mode"] == "copy", b1
        assert b1["filename"].endswith(f"_{jid[:8]}.mp4"), b1
        assert b1["filename"].startswith("a_red_lantern_swaying"), b1
        assert os.path.realpath(b1["path"]).startswith(os.path.realpath(inbox_root))
        assert os.path.isfile(b1["path"]) and os.path.getsize(b1["path"]) > 0

        # RE-SEND: unique_path appends a numeric suffix; the first copy is untouched.
        r2 = _CLIENT.post(f"/api/video/studio/clip/{jid}/to-editor", headers=hdr)
        assert r2.status_code == 200, (r2.status_code, r2.get_data(as_text=True))
        b2 = r2.get_json()
        assert b2["filename"] != b1["filename"], (b1, b2)
        assert "_1" in b2["filename"], b2
        assert os.path.isfile(b1["path"]) and os.path.isfile(b2["path"])
    finally:
        os.environ.pop("HUGPY_OPERATOR_TOKEN", None)
        job_mod.DEFAULT_CLIPS_ROOT = orig_clips
        job_mod.EDITOR_INBOX_ROOT = orig_inbox
        _restore(orig_sess, orig_share)
    print("[11] PASS  route 200 (copy) + non-destructive re-send (_1 suffix)")


# --------------------------------------------------------------------------- #
def _run_all():
    test_auth_share_principal_denied()
    test_auth_console_session_allowed()
    test_auth_operator_token_allowed()
    test_branch_decision()
    test_filename_sanitized()
    test_incomplete_clip_404()
    test_archived_clip_410()
    test_send_to_editor_copy_roundtrip()
    test_send_to_editor_transcode_roundtrip()
    test_send_to_editor_missing_input()
    test_route_success_and_resend()
    print("\nALL send-to-editor backend checks passed")


if __name__ == "__main__":
    _run_all()
