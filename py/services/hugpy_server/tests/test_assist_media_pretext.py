"""KEEPER-TASK k93 §C — /video/prompt/assist ``context.media`` pretext.

Driven through the REAL route with the ONE execute_prompt seam patched (the
functions.imports binding — patching managers.dispatch would run live inference,
see tests/test_prompt_spread.py). The fake executor answers BOTH the vision
describe calls (task=image-text-to-text) and the text generation, and records
every call so the test can assert what the model was told.

Covers:
  * validation — malformed media blocks 400 in the context.<field> style
  * absent media — the user message is byte-identical to a no-media call
  * image pretext — one description, "Reference image: …" prepended
  * video pretext — ≤4 evenly spaced frames through the real frame-extract
    runner (a tiny synthetic clip rendered with ffmpeg; skipped if no ffmpeg),
    "Reference video: …" prepended, description logged, response carries media
  * cache — a second call on the same uri makes NO further vision calls
  * spread — the same contract on mode="spread"
"""
from __future__ import annotations

import importlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

logging.disable(logging.INFO)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_video import studio_assist_log as SAL
from hugpy_platform.constants import DEFAULT_ROOT
from hugpy_server.app.routes import video_assist_media as VAM

GEN_REPLY = "A lone diver under blue ice, shafts of light, slow push-in."
VISION_REPLY = "<think>looking</think>A red square on a black field, flat lighting, static framing."


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
@pytest.fixture()
def harness():
    from flask import Flask
    vr = importlib.import_module("hugpy_server.app.routes.video_routes")
    imports_mod = importlib.import_module("hugpy_engine.dispatch.dispatch")
    app = Flask(__name__)
    app.register_blueprint(vr.video_bp)
    calls = []

    def fake_execute_prompt(*a, **kw):
        calls.append(kw)
        if kw.get("task") == "image-text-to-text":
            return {"ok": True, "text": VISION_REPLY, "model_key": "vision-x"}
        return {"ok": True, "text": GEN_REPLY, "model_key": "gen-x"}

    orig = imports_mod.execute_prompt
    imports_mod.execute_prompt = fake_execute_prompt
    d = tempfile.mkdtemp()
    SAL.set_store(SAL.StudioAssistStore(path=os.path.join(d, "c.db")))
    SAL.reset_for_tests()
    VAM.cache_clear()
    try:
        yield vr, app.test_client(), calls
    finally:
        imports_mod.execute_prompt = orig
        SAL.set_store(None)
        VAM.cache_clear()
        shutil.rmtree(d, ignore_errors=True)


def _ffmpeg():
    try:
        from hugpy_platform.binaries import resolve_bin
        return resolve_bin("ffmpeg") or shutil.which("ffmpeg")
    except Exception:
        return shutil.which("ffmpeg")


@pytest.fixture(scope="module")
def media_dir():
    """A jailed dir (under DEFAULT_ROOT) holding a 2s synthetic clip + a png."""
    base = os.path.join(DEFAULT_ROOT, "video_intel", "_test_assist_media")
    os.makedirs(base, exist_ok=True)
    d = tempfile.mkdtemp(dir=base)
    ff = _ffmpeg()
    paths = {"dir": d, "video": None, "image": None}
    if ff:
        vid = os.path.join(d, "clip.mp4")
        img = os.path.join(d, "still.png")
        r1 = subprocess.run([ff, "-y", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=2",
                             "-r", "10", "-pix_fmt", "yuv420p", vid],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        r2 = subprocess.run([ff, "-y", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1",
                             "-frames:v", "1", img],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r1.returncode == 0 and os.path.isfile(vid):
            paths["video"] = vid
        if r2.returncode == 0 and os.path.isfile(img):
            paths["image"] = img
    yield paths
    shutil.rmtree(d, ignore_errors=True)


def _gen(client, ctx=None, **extra):
    body = {"mode": "generate", "draft": "a diver", "model": "m"}
    if ctx is not None:
        body["context"] = ctx
    body.update(extra)
    return client.post("/video/prompt/assist", json=body)


def _user_msgs(calls):
    return [c["messages"][-1]["content"] for c in calls if "messages" in c]


# --------------------------------------------------------------------------- #
# 1. validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("media, frag", [
    ("clip.mp4", "context.media must be an object"),
    ({}, "context.media.uri must be a non-empty string"),
    ({"uri": "x"}, "context.media.mime must be a non-empty string"),
    ({"uri": "x", "mime": 5}, "context.media.mime must be a non-empty string"),
    ({"uri": "x", "mime": "audio/wav"}, "context.media.mime must be video/* or image/*"),
    ({"uri": "x", "mime": "video/mp4", "label": 3}, "context.media.label must be a string"),
])
def test_malformed_media_is_400(harness, media, frag):
    vr, client, calls = harness
    r = _gen(client, {"kind": "scene", "media": media})
    assert r.status_code == 400, r.get_json()
    assert frag in r.get_json()["error"]
    assert calls == []          # rejected before any model call


def test_uri_outside_jail_is_400_and_missing_is_404(harness):
    vr, client, calls = harness
    r = _gen(client, {"media": {"uri": "/etc/passwd", "mime": "video/mp4"}})
    assert r.status_code == 400 and "jail" in r.get_json()["error"]
    r = _gen(client, {"media": {"uri": os.path.join(DEFAULT_ROOT, "nope-k93.mp4"),
                                "mime": "video/mp4"}})
    assert r.status_code == 404
    assert calls == []


def test_kind_validation_unchanged(harness):
    vr, client, _ = harness
    r = _gen(client, {"kind": "poem"})
    assert r.status_code == 400
    assert r.get_json()["error"].startswith("context.kind must be one of")


# --------------------------------------------------------------------------- #
# 2. absent media -> byte-identical
# --------------------------------------------------------------------------- #
def test_absent_media_is_byte_identical(harness):
    vr, client, calls = harness
    r1 = _gen(client, {"kind": "scene", "hint": "cold"})
    r2 = _gen(client, {"kind": "scene", "hint": "cold", "media": None})
    assert r1.status_code == r2.status_code == 200
    m1, m2 = _user_msgs(calls)
    # Generate is randomly steered per call, so compare everything but the
    # steering clause: the message must not carry any pretext.
    assert "Reference video" not in m1 and "Reference image" not in m1
    assert m1.split("\n\n")[0] == m2.split("\n\n")[0]
    assert "media" not in r1.get_json() and "media" not in r2.get_json()


# --------------------------------------------------------------------------- #
# 3. image / video pretext, cache, log, response
# --------------------------------------------------------------------------- #
def test_image_pretext_prepended_once(harness, media_dir):
    if not media_dir["image"]:
        pytest.skip("ffmpeg unavailable — cannot render the synthetic still")
    vr, client, calls = harness
    ctx = {"kind": "scene", "media": {"uri": media_dir["image"], "mime": "image/png",
                                       "label": "ref still"}}
    r = _gen(client, ctx)
    assert r.status_code == 200, r.get_json()
    vision = [c for c in calls if c.get("task") == "image-text-to-text"]
    assert len(vision) == 1
    assert vision[0]["file"] == os.path.realpath(media_dir["image"])
    user = _user_msgs(calls)[0]
    assert user.startswith('Reference image ("ref still"): A red square on a black field')
    assert "<think>" not in user          # the describe reply is think-stripped
    assert "Write one compelling" in user  # instruction still follows the pretext
    body = r.get_json()
    assert body["media"]["kind"] == "image" and body["media"]["cached"] is False
    assert body["media"]["pretext"].startswith("Reference image")


def test_video_pretext_samples_frames_caches_and_logs(harness, media_dir):
    if not media_dir["video"]:
        pytest.skip("ffmpeg unavailable — cannot render the synthetic clip")
    vr, client, calls = harness
    ctx = {"kind": "scene", "media": {"uri": media_dir["video"], "mime": "video/mp4"}}

    r = _gen(client, ctx)
    assert r.status_code == 200, r.get_json()
    vision = [c for c in calls if c.get("task") == "image-text-to-text"]
    assert 1 <= len(vision) <= VAM.MAX_FRAMES
    assert all(os.path.isfile(c["file"]) for c in vision)
    assert all("model_key" in c and c["model_key"] for c in vision)
    user = _user_msgs(calls)[0]
    assert user.startswith(f"Reference video: {len(vision)} frame")
    assert "Frame 1 (t=0s)" in user
    body = r.get_json()
    media = body["media"]
    assert media["kind"] == "video" and media["cached"] is False
    assert len(media["frames"]) == len(vision)
    ts = [f["t"] for f in media["frames"]]
    assert ts == sorted(ts) and ts[0] == 0.0
    if len(ts) > 1:
        gaps = [round(b - a, 2) for a, b in zip(ts, ts[1:])]
        assert max(gaps) - min(gaps) < 0.02      # evenly spaced

    # the operator can read what the model was told
    recs = SAL.get_store().recent()
    served = [x for x in recs if x.get("outcome") == SAL.OUTCOME_SERVED]
    assert served and served[-1]["media_pretext"].startswith("Reference video")
    assert served[-1]["media_frames"] == len(vision)
    assert served[-1]["media_cached"] is False

    # CACHE: enhance on the same clip -> zero new vision calls
    n_before = len(calls)
    r2 = client.post("/video/prompt/assist", json={
        "mode": "detail", "draft": "a diver", "model": "m", "context": ctx})
    assert r2.status_code == 200, r2.get_json()
    new_calls = calls[n_before:]
    assert [c for c in new_calls if c.get("task") == "image-text-to-text"] == []
    assert r2.get_json()["media"]["cached"] is True
    assert new_calls[-1]["messages"][-1]["content"].startswith("Reference video:")
    assert "Expand this draft" in new_calls[-1]["messages"][-1]["content"]


def test_mime_kind_mismatch_is_400(harness, media_dir):
    if not media_dir["image"]:
        pytest.skip("ffmpeg unavailable")
    vr, client, calls = harness
    r = _gen(client, {"media": {"uri": media_dir["image"], "mime": "video/mp4"}})
    assert r.status_code == 400
    assert "says video but the file is image" in r.get_json()["error"]


def test_vision_failure_is_honest_502(harness, media_dir):
    if not media_dir["image"]:
        pytest.skip("ffmpeg unavailable")
    vr, client, calls = harness
    imports_mod = importlib.import_module("hugpy_engine.dispatch.dispatch")

    def boom(*a, **kw):
        if kw.get("task") == "image-text-to-text":
            raise RuntimeError("no live worker")
        return {"ok": True, "text": GEN_REPLY}
    imports_mod.execute_prompt = boom
    r = _gen(client, {"media": {"uri": media_dir["image"], "mime": "image/png"}})
    assert r.status_code == 502
    assert "vision model" in r.get_json()["error"]
    recs = SAL.get_store().recent()
    assert recs and recs[-1]["outcome"] == SAL.OUTCOME_WORKER_ERROR
    assert recs[-1]["media_uri"] == media_dir["image"]


# --------------------------------------------------------------------------- #
# 4. spread carries the same contract
# --------------------------------------------------------------------------- #
def test_spread_accepts_media(harness, media_dir):
    if not media_dir["image"]:
        pytest.skip("ffmpeg unavailable")
    vr, client, calls = harness
    import json
    ids = ("segment-0", "segment-1")
    reply = json.dumps({"segments": [
        {"segment_id": sid, "operation": "generate_from_direction",
         "prompt": f"A shot for {sid}.", "negative": "blurry",
         "continuity_note": "", "directions_used": [0]} for sid in ids],
        "invented_identity_attributes": [], "warnings": []})
    imports_mod = importlib.import_module("hugpy_engine.dispatch.dispatch")

    def fake(*a, **kw):
        calls.append(kw)
        if kw.get("task") == "image-text-to-text":
            return {"ok": True, "text": VISION_REPLY}
        return {"ok": True, "text": reply}
    imports_mod.execute_prompt = fake

    body = {
        "mode": "spread",
        "movie_query": "a diver finds something under the ice",
        "style_bible": {"world": "arctic"},
        "fixed_segments": [],
        "target_segments": [{"segment_id": sid, "direction": "colder",
                             "joint_mode": "vace_extend", "index": i}
                            for i, sid in enumerate(ids)],
        "global_negative": ["watermark"],
        "steering_seed": 1,
        "context": {"media": {"uri": media_dir["image"], "mime": "image/png"}},
    }
    r = client.post("/video/prompt/assist", json=body)
    assert r.status_code == 200, r.get_json()
    user = _user_msgs(calls)[0]
    assert user.startswith("Reference image: ")
    assert "THE TIMELINE:" in user
    assert r.get_json()["media"]["kind"] == "image"

    # malformed on spread -> same 400 style
    body["context"] = {"media": {"uri": 1, "mime": "image/png"}}
    r = client.post("/video/prompt/assist", json=body)
    assert r.status_code == 400
    assert "context.media.uri" in r.get_json()["error"]
