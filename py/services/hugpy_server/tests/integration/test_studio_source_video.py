"""B2 movie->studio chain — the studio's VIDEO input (``source_video``).

Locks the source-clip slice as executable checks, in the same script style as
``test_studio_prompt.py`` / ``test_studio_t2v.py`` (plain python, ``__main__``
guard, numbered ``[n] PASS`` / ``[n] FAIL`` lines, nonzero exit iff any check
FAILED, every check independently run so a failing one never masks the rest).
pytest is NOT installed in this venv, so there are no fixtures.

Invariant under test: a studio i2v job can CONSUME a prior tier's output (a
movie/scene mp4). ``source_video`` threads end to end —
route -> ``make_studio_i2v`` -> bus adapter -> ``produce_clip`` ->
``make_render_manifest`` -> ``RenderManifest`` (part of ``content_hash``) -> the
i2v runner, which EXTENDS the clip from its LAST FRAME (ffmpeg) when no
``start_image`` is given. The route also accepts a ``source_asset_id`` resolved to
its uri via the media catalog, and rejects a non-video / nonexistent /
jail-escaping target with a clean 4xx. t2v is text-only: a source is carried but
never alters a frame (dropped at the route).

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_source_video.py
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict

logging.disable(logging.INFO)  # silence the models_config registry chatter

os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import importlib  # noqa: E402

from flask import Flask  # noqa: E402

from hugpy_platform.constants import DEFAULT_ROOT
from hugpy_video.intel import media_bus
from hugpy_video.intel.runners.studio_i2v import run_studio_i2v
from hugpy_video.intel.studio.artifacts import Artifact
from hugpy_video.intel.studio.enums import Capability, Task
from hugpy_video.intel.studio.env import StudioEnv
from hugpy_video.intel.studio.job import make_studio_i2v, studio_i2v_from_dict
from hugpy_video.intel.studio.manifest import (
    make_render_manifest,
    render_manifest_from_dict,
    render_manifest_to_dict,
)
from hugpy_video.intel.studio.produce import produce_clip
from hugpy_video.intel.studio.router import CapabilityRouter
from hugpy_video.intel.studio.schemas import (
    CapabilityRequest,
    Resolution,
    SamplerConfig,
    SeedBundle,
)

_FFMPEG = shutil.which("ffmpeg") is not None
_FFPROBE = shutil.which("ffprobe") is not None

R_TINY = Resolution(320, 180, 12)

# --------------------------------------------------------------------------- #
# Isolation: point the media bus at a TEMP DB so route enqueues + the catalog
# lookup never touch the real /mnt/llm_storage/video_intel/media_jobs.db. A real
# work dir UNDER DEFAULT_ROOT holds the tiny fixtures (the route jails to it).
# --------------------------------------------------------------------------- #
_TMP_DB = tempfile.mkstemp(prefix="srcvid-bus-", suffix=".db")[1]
media_bus.DB_PATH = _TMP_DB
media_bus._initialized = False  # force _ensure_db to re-init against the temp DB
with sqlite3.connect(_TMP_DB) as _c:
    _c.execute(
        "CREATE TABLE IF NOT EXISTS media_jobs ("
        "job_id TEXT PRIMARY KEY, name TEXT, status TEXT, spec_json TEXT, "
        "result_json TEXT, claim_token TEXT, created REAL, updated REAL, "
        "progress_json TEXT)")

_WORK = tempfile.mkdtemp(prefix="studio-srcvid-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))  # inside the jail
_SRC_MP4 = os.path.join(_WORK, "source_movie.mp4")
_NOT_A_VIDEO = os.path.join(_WORK, "not_a_video.txt")
_OUTSIDE_MP4 = tempfile.mkstemp(prefix="outside-", suffix=".mp4")[1]  # OUTSIDE the jail
_SEEDED_ASSET_ID = "seeded" + uuid.uuid4().hex

vr = importlib.import_module("hugpy_server.app.routes.video_routes")
app = Flask(__name__)
app.register_blueprint(vr.video_bp)
client = app.test_client()


def _make_mp4(path: str, *, dur: int = 2, w: int = 160, h: int = 120, rate: int = 8) -> None:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i",
         f"testsrc=duration={dur}:size={w}x{h}:rate={rate}",
         "-pix_fmt", "yuv420p", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def _setup_fixtures() -> None:
    """A real tiny mp4 + a real non-video file inside the jail, a real mp4 OUTSIDE
    the jail, and a seeded catalog row (asset_id -> the in-jail mp4's uri)."""
    if _FFMPEG:
        _make_mp4(_SRC_MP4)
        _make_mp4(_OUTSIDE_MP4)
    with open(_NOT_A_VIDEO, "w", encoding="utf-8") as fh:
        fh.write("this is plainly not a video\n")
    # Seed the catalog: a completed job whose output carries _SEEDED_ASSET_ID + the
    # in-jail mp4 as its uri, so /video/studio/i2v can resolve source_asset_id -> uri.
    result_json = json.dumps({
        "job_id": "seed-job", "ok": True,
        "outputs": [{
            "asset_id": _SEEDED_ASSET_ID, "kind": "video", "uri": _SRC_MP4,
            "mime": "video/mp4", "width": 160, "height": 120, "duration_s": 2.0,
        }],
    })
    with sqlite3.connect(_TMP_DB) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO media_jobs "
            "(job_id, name, status, spec_json, result_json, created, updated) "
            "VALUES (?,?,?,?,?,?,?)",
            ("seed-job", "generate_movie", "done", "{}", result_json,
             time.time(), time.time()))


def _teardown_fixtures() -> None:
    shutil.rmtree(_WORK, ignore_errors=True)
    for p in (_TMP_DB, _OUTSIDE_MP4):
        try:
            os.remove(p)
        except OSError:
            pass


def _studio_env(master_fps: int = 12) -> StudioEnv:
    return StudioEnv(
        output_root="/out", weights_root="/weights", manifest_root="/manifests",
        master_colorspace="rec709", master_fps=master_fps, max_vram_gb=24.0,
        loudness_target_lufs=-14.0, allow_unpinned=True)


def _synth_binding():
    return CapabilityRouter().resolve(CapabilityRequest(
        capability=Capability.I2V, target_resolution=R_TINY,
        vram_budget_gb=0.5)).unwrap()


def _mk_manifest(source_video: str, *, capability=Capability.I2V):
    b = CapabilityRouter().resolve(CapabilityRequest(
        capability=capability, target_resolution=R_TINY, vram_budget_gb=0.5)).unwrap()
    return make_render_manifest(
        render_id="rsv", capability=capability, binding=b,
        seeds=SeedBundle(global_seed=3, stage_seeds=(("base", 3),)),
        sampler=SamplerConfig(sampler="euler", scheduler="normal", steps=1, cfg=1.0),
        resolution_ladder=(R_TINY,), env=_studio_env(), source_video=source_video)


# --------------------------------------------------------------------------- #
# [1] Spec: make_studio_i2v carries source_video; asdict/from_dict round-trips it;
#     an empty string is rejected; an absent key rehydrates to None (back-compat).
# --------------------------------------------------------------------------- #
def test_spec_roundtrips_source_video():
    spec = make_studio_i2v(width=64, height=64, fps=8, source_video="/mnt/x/movie.mp4")
    assert spec.source_video == "/mnt/x/movie.mp4", "spec must carry source_video"
    d = asdict(spec)
    assert d["source_video"] == "/mnt/x/movie.mp4", "asdict(spec) must carry source_video"
    spec2 = studio_i2v_from_dict(d)
    assert spec2.source_video == spec.source_video, "from_dict must preserve source_video"

    d.pop("source_video", None)
    spec3 = studio_i2v_from_dict(d)
    assert spec3.source_video is None, "a spec dict without source_video rehydrates to None"

    try:
        make_studio_i2v(width=64, height=64, fps=8, source_video="")
    except ValueError:
        pass
    else:
        raise AssertionError("an empty-string source_video must be rejected")

    # None (no source) stays valid — the common i2v/t2v case.
    assert make_studio_i2v(width=64, height=64, fps=8).source_video is None


# --------------------------------------------------------------------------- #
# [2] Manifest: source_video is in the content_hash (a DIFFERENT source -> a
#     DIFFERENT hash; the SAME source -> the SAME hash); to_dict/from_dict
#     round-trips it (and the hash); an absent key rehydrates to "" (back-compat).
# --------------------------------------------------------------------------- #
def test_manifest_hash_keys_on_source_video():
    h_none = _mk_manifest("").content_hash()
    h_a = _mk_manifest("/movies/a.mp4").content_hash()
    h_b = _mk_manifest("/movies/b.mp4").content_hash()
    assert h_a != h_none, "a source_video must change the content_hash vs. no source"
    assert h_a != h_b, "a DIFFERENT source_video must change the content_hash"
    assert _mk_manifest("/movies/a.mp4").content_hash() == h_a, (
        "the SAME source_video must hash equal (determinism preserved)")

    m = _mk_manifest("/movies/a.mp4")
    d = render_manifest_to_dict(m)
    assert d["source_video"] == "/movies/a.mp4", "to_dict must serialize source_video"
    m2 = render_manifest_from_dict(d)
    assert m2.source_video == "/movies/a.mp4", "from_dict must rehydrate source_video"
    assert m2.content_hash() == m.content_hash(), (
        "to_dict->from_dict must preserve content_hash with source_video")

    d.pop("source_video", None)  # a manifest serialized before the field existed
    m3 = render_manifest_from_dict(d)
    assert m3.source_video == "", "a pre-source_video manifest dict must rehydrate to ''"


# --------------------------------------------------------------------------- #
# [3] Route: POST source_video=<real tiny mp4 inside the jail> -> 200 {job_id}.
# --------------------------------------------------------------------------- #
def test_route_source_video_real_mp4_200():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping route real-mp4 check)")
        return
    r = client.post("/video/studio/i2v", json={
        "capability": "i2v", "resolution": {"width": 320, "height": 180, "fps": 12},
        "vram_budget_gb": 0.5, "seed": 0, "source_video": _SRC_MP4})
    assert r.status_code == 200, (r.status_code, r.get_json())
    assert isinstance(r.get_json().get("job_id"), str), r.get_json()


# --------------------------------------------------------------------------- #
# [4] Route: a nonexistent path UNDER the jail -> 404 (not found).
# --------------------------------------------------------------------------- #
def test_route_source_video_nonexistent_404():
    ghost = os.path.join(_WORK, "no-such-clip.mp4")
    r = client.post("/video/studio/i2v", json={
        "capability": "i2v", "source_video": ghost})
    assert r.status_code == 404, (r.status_code, r.get_json())


# --------------------------------------------------------------------------- #
# [5] Route: a path ESCAPING the storage jail -> 400 (never an arbitrary read).
# --------------------------------------------------------------------------- #
def test_route_source_video_jail_escape_400():
    r = client.post("/video/studio/i2v", json={
        "capability": "i2v", "source_video": _OUTSIDE_MP4})
    assert r.status_code == 400, (r.status_code, r.get_json())
    r2 = client.post("/video/studio/i2v", json={
        "capability": "i2v", "source_video": "/etc/passwd"})
    assert r2.status_code == 400, (r2.status_code, r2.get_json())


# --------------------------------------------------------------------------- #
# [6] Route: an in-jail file that is NOT a video -> 400 (probe-classified).
# --------------------------------------------------------------------------- #
def test_route_source_video_not_a_video_400():
    if not _FFPROBE:
        print("      (ffprobe unavailable — skipping non-video classify check)")
        return
    r = client.post("/video/studio/i2v", json={
        "capability": "i2v", "source_video": _NOT_A_VIDEO})
    assert r.status_code == 400, (r.status_code, r.get_json())


# --------------------------------------------------------------------------- #
# [7] Route: source_asset_id resolves via the media catalog to its uri -> 200;
#     an unknown asset id -> 404.
# --------------------------------------------------------------------------- #
def test_route_source_asset_id_resolves_and_unknown_404():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping source_asset_id check)")
        return
    r = client.post("/video/studio/i2v", json={
        "capability": "i2v", "source_asset_id": _SEEDED_ASSET_ID})
    assert r.status_code == 200, (r.status_code, r.get_json())
    assert isinstance(r.get_json().get("job_id"), str), r.get_json()

    r2 = client.post("/video/studio/i2v", json={
        "capability": "i2v", "source_asset_id": "no-such-asset-" + uuid.uuid4().hex})
    assert r2.status_code == 404, (r2.status_code, r2.get_json())


# --------------------------------------------------------------------------- #
# [8] Route: t2v is TEXT-ONLY — a source is DROPPED before validation, so even a
#     jail-escaping source_video does not 400 a t2v enqueue (it is ignored).
# --------------------------------------------------------------------------- #
def test_route_t2v_ignores_source_video():
    r = client.post("/video/studio/i2v", json={
        "capability": "t2v", "source_video": _OUTSIDE_MP4,
        "prompt": "a lighthouse sweeping fog"})
    assert r.status_code == 200, (
        f"t2v must IGNORE (not reject) a source_video; got {r.status_code} {r.get_json()}")


# --------------------------------------------------------------------------- #
# [9] Produce path: produce_clip with a real source_video + NO start_image ->
#     Ok(Artifact); the last-frame extraction ran (source_lastframe.png sidecar) and
#     the manifest.json records source_video; a re-run RESUMES (deterministic).
# --------------------------------------------------------------------------- #
def test_produce_clip_extends_from_source_video():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping produce extend check)")
        return
    env = _studio_env(12)
    req = CapabilityRequest(
        capability=Capability.I2V, target_resolution=R_TINY, vram_budget_gb=0.5)
    out_root = tempfile.mkdtemp(prefix="studio-srcvid-out-")
    try:
        res = produce_clip(req, env=env, out_root=out_root, source_video=_SRC_MP4)
        assert res.is_ok(), f"produce_clip with a source_video must be Ok(Artifact); got {res}"
        art = res.unwrap()
        assert isinstance(art, Artifact)
        assert os.path.isfile(art.path) and os.path.getsize(art.path) > 0, "clip non-empty"

        clip_dir = os.path.dirname(art.path)
        # extraction actually ran: the last-frame conditioning still is on disk.
        assert os.path.isfile(os.path.join(clip_dir, "source_lastframe.png")), (
            "the i2v last-frame extraction must have produced source_lastframe.png")
        # the manifest sidecar records the source (provenance + it is in the hash).
        with open(os.path.join(clip_dir, "manifest.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        assert man["source_video"] == _SRC_MP4, (
            f"manifest.json must record source_video; got {man.get('source_video')!r}")

        # a DIFFERENT (no) source addresses a DIFFERENT clip dir (source is in the hash).
        res_nosrc = produce_clip(req, env=env, out_root=out_root)
        assert res_nosrc.is_ok()
        assert os.path.dirname(res_nosrc.unwrap().path) != clip_dir, (
            "a source_video must produce a different content-addressed clip dir")

        # determinism: an identical extend RESUMES the same clip (source in the hash,
        # last-frame extraction deterministic).
        res2 = produce_clip(req, env=env, out_root=out_root, source_video=_SRC_MP4)
        assert res2.is_ok() and res2.unwrap().resumed is True, (
            "an identical source_video extend must resume the existing clip")
        assert res2.unwrap().content_hash == art.content_hash
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [10] Produce path: t2v CARRIES source_video in the manifest but never alters a
#      frame — no extraction (manifest.task != I2V), and no source_lastframe.png.
# --------------------------------------------------------------------------- #
def test_produce_clip_t2v_carries_but_ignores_source():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping t2v carry check)")
        return
    env = _studio_env(12)
    req = CapabilityRequest(
        capability=Capability.T2V, target_resolution=R_TINY, vram_budget_gb=0.5)
    out_root = tempfile.mkdtemp(prefix="studio-srcvid-t2v-")
    try:
        res = produce_clip(req, env=env, out_root=out_root, source_video=_SRC_MP4,
                           prompt="a drone over a city grid")
        assert res.is_ok(), f"t2v produce_clip must be Ok; got {res}"
        clip_dir = os.path.dirname(res.unwrap().path)
        with open(os.path.join(clip_dir, "manifest.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        assert man["source_video"] == _SRC_MP4, "t2v manifest must CARRY the source_video"
        assert man["task"] == "t2v", man["task"]
        assert not os.path.isfile(os.path.join(clip_dir, "source_lastframe.png")), (
            "t2v must NOT extract a last frame (text-only)")
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [11] Bus adapter: run_studio_i2v with a source_video spec (no start_image) ->
#      JobResult(ok=True) carrying a video clip ref; the clip's manifest records
#      source_video (the spec->adapter->produce->manifest thread, end to end).
# --------------------------------------------------------------------------- #
def test_run_studio_i2v_source_video_ok():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping bus-adapter extend check)")
        return
    out_root = tempfile.mkdtemp(prefix="studio-srcvid-bus-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    spec = make_studio_i2v(
        capability="i2v", width=320, height=180, fps=12, vram_budget_gb=0.5, seed=0,
        out_root=out_root, source_video=_SRC_MP4)
    assert spec.source_video == _SRC_MP4, "make_studio_i2v must carry source_video"

    orig = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    try:
        result = run_studio_i2v(spec, job_id="srcvid-bus-1")
        assert result.ok is True, f"a source_video bus job must be ok=True; got {result}"
        assert result.error is None, result.error
        assert len(result.outputs) == 1, result.outputs
        ref = result.outputs[0]
        assert getattr(ref, "kind", None) == "video", f"ref must be a video; got {ref}"
        clip_dir = os.path.dirname(ref.uri)
        with open(os.path.join(clip_dir, "manifest.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        assert man["source_video"] == _SRC_MP4, (
            "the produced clip's manifest must record source_video end to end")
    finally:
        media_bus.is_cancelling = orig
        shutil.rmtree(out_root, ignore_errors=True)


CHECKS = [
    ("spec: make_studio_i2v carries source_video; round-trip + empty-reject + back-compat",
     test_spec_roundtrips_source_video),
    ("manifest: source_video in content_hash (diff->diff, same->same) + round-trip",
     test_manifest_hash_keys_on_source_video),
    ("route: POST source_video=<real tiny mp4> -> 200 {job_id}",
     test_route_source_video_real_mp4_200),
    ("route: nonexistent path under the jail -> 404",
     test_route_source_video_nonexistent_404),
    ("route: path escaping the storage jail -> 400",
     test_route_source_video_jail_escape_400),
    ("route: an in-jail non-video file -> 400 (probe-classified)",
     test_route_source_video_not_a_video_400),
    ("route: source_asset_id resolves via the catalog -> 200; unknown -> 404",
     test_route_source_asset_id_resolves_and_unknown_404),
    ("route: t2v IGNORES (never rejects) a source_video",
     test_route_t2v_ignores_source_video),
    ("produce: i2v extends from source_video's LAST FRAME + manifest records it + resume",
     test_produce_clip_extends_from_source_video),
    ("produce: t2v CARRIES source_video but extracts no last frame (text-only)",
     test_produce_clip_t2v_carries_but_ignores_source),
    ("bus adapter: run_studio_i2v(source_video spec) -> ok + video ref + manifest records it",
     test_run_studio_i2v_source_video_ok),
]


def main() -> int:
    _setup_fixtures()
    passed = 0
    failed = 0
    try:
        for i, (name, fn) in enumerate(CHECKS, 1):
            try:
                fn()
            except Exception as exc:  # surface EVERY divergence, not just the first
                failed += 1
                print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
            else:
                passed += 1
                print(f"[{i}] PASS  {name}")
    finally:
        _teardown_fixtures()
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
