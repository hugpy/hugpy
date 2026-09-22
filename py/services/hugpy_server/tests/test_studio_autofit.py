"""Studio AUTOFIT budget — conformance.

Operator doctrine (2026-07-12): a BLANK vram budget must NOT be a guaranteed-fail low
guess. "Why can it not default to what's needed? If a model needs 14GB and it's blank,
why would it fail trying 6GB — just do 14, otherwise a fail is 100% likely." A blank
budget means AUTOFIT: size the routing budget to the SERVING WORKER's GPU CAPACITY.

⚠ CAPACITY, NOT FREE (operator ruling 2026-07-27: "it needs to evict like everything
else, not assess what it thinks it should set a budget for"). Sizing to momentary FREE
VRAM made the bound MODEL a function of whatever the LLM residents held that second —
measured on ae: 3.30 / 5.84 / 7.39 / 6.67 GB minutes apart, crossing the 6.00 GB
cheapest-real-model floor in both directions, so the same request rendered real video or
a synthetic noise blob depending on the second. Capacity states what the card can hold;
the reservation engine evicts to free it, or refuses honestly.

What is under test:
  * ROUTE sentinel (_autofit_vram_budget): blank/absent/null -> None (autofit); an explicit
    number is the manual override (passthrough); a bad value still 400s in the factory.
  * SPEC threading: make_studio_i2v / make_studio_movie accept None (autofit) as legal,
    still reject bad numbers, and None round-trips through asdict -> from_dict.
  * AUTOFIT RESOLUTION (_resolve_autofit / _autofit_from_worker): a fake registry row with
    known GPU CAPACITY -> effective = total - margin (10% or 2GB, whichever larger);
    host-only URL match; no worker / no VRAM data -> "unresolved" with NO fallback number
    (the render refuses); an EXPLICIT budget bypasses the worker lookup entirely.
  * render_clip STAMPS the resolved (effective_budget_gb, budget_source), REFUSES an
    unsizable render (gpu_unavailable) instead of writing a synthetic blob, and treats
    the synthetic prover as OPT-IN (STUDIO_ALLOW_SYNTHETIC=1).
  * MOVIE: an id-movie with a None budget -> every segment carries the autofit budget
    (>= the VACE floor) + budget_source in movie.json; an EXPLICIT budget is byte-identical
    to today (floored, budget_source "explicit").

The worker registry the autofit resolver reads is the ``hugpy_engine.placement`` seam
(the fleet installs its registry there at composition time); each check installs a
fake registry with known GPU CAPACITY rows.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict

import pytest

from hugpy_engine import placement
from hugpy_video.intel import media_bus, media_store
from hugpy_video.intel.runners import studio_i2v as S
from hugpy_video.intel.runners.studio_movie import run_generate_studio_movie
from hugpy_video.intel.studio.job import make_studio_i2v, studio_i2v_from_dict
from hugpy_video.intel.studio_movie_schema import (
    StudioMovieGoal,
    make_studio_movie,
    studio_movie_from_dict,
)
from hugpy_server.app.routes.video_routes import _autofit_vram_budget

try:
    from PIL import Image
    _PIL = True
except Exception:  # noqa: BLE001
    _PIL = False

_FFMPEG = shutil.which("ffmpeg") is not None
_FFPROBE = shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not _FFMPEG, reason="ffmpeg unavailable")
needs_media_tools = pytest.mark.skipif(not (_FFMPEG and _FFPROBE and _PIL),
                                       reason="ffmpeg/ffprobe/PIL unavailable")

_GIB = 1024 ** 3
_ENV_KEYS = ("HUGPY_STUDIO_WORKER", "HUGPY_STUDIO_FORCE_REMOTE",
             "HUGPY_STUDIO_POLL_INTERVAL_S", "HUGPY_STUDIO_KICKOFF_RETRY_WINDOW_S",
             "HUGPY_STUDIO_KICKOFF_RETRY_INTERVAL_S", "STUDIO_ALLOW_SYNTHETIC")


@pytest.fixture(scope="module", autouse=True)
def _throwaway_media_bus():
    """LIVE-DB SAFETY (mirrors the offload/movie suites): repoint the bus at a
    throwaway db so is_cancelling / set_progress / enqueue never touch a live central."""
    tmp = tempfile.mkdtemp(prefix="hugpy_autofit_test_")
    orig_path, orig_init = media_bus.DB_PATH, media_bus._initialized
    media_bus.DB_PATH = os.path.join(tmp, "media_jobs.db")
    media_bus._initialized = False
    yield
    media_bus.DB_PATH, media_bus._initialized = orig_path, orig_init
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(autouse=True)
def _studio_env(monkeypatch):
    """Unpinned studio, no worker/remote/synthetic env leaking in from the shell,
    and the cancel probe answers 'no' without a job row."""
    monkeypatch.setenv("STUDIO_ALLOW_UNPINNED", "1")
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(media_bus, "is_cancelling", lambda job_id: False)


class _FakeRegistry:
    def __init__(self, rows):
        self._rows = list(rows)

    def list_workers(self, online_only=False):
        return list(self._rows)


class _ExplodingRegistry:
    def list_workers(self, online_only=False):
        raise AssertionError("explicit budget must not read the worker store")


@pytest.fixture
def fake_workers(monkeypatch):
    """Install a fake worker registry returning ``rows`` on the placement seam
    (conftest restores the seam after the test)."""
    def _install(rows):
        placement.set_worker_registry(_FakeRegistry(rows))
    yield _install
    placement.set_worker_registry(None)


def _worker_row(name, url, total_gib, free_gib=0.5):
    """A registry row whose GPU has ``total_gib`` CAPACITY.

    ``free_gib`` defaults to 0.5 ON PURPOSE — under every real model's floor. Autofit
    sizes to CAPACITY (operator ruling 2026-07-27), so a regression to the old
    ``memory_free`` basis makes every budget here collapse to a synthetic-binding number
    instead of the expected one, and these tests fail loudly rather than silently."""
    return {"name": name, "url": url,
            "gpus": [{"index": 0, "memory_total": int(total_gib * _GIB),
                      "memory_free": int(free_gib * _GIB)}]}


def _i2v(budget, **kw):
    return make_studio_i2v(capability="i2v", width=256, height=256, fps=8,
                           vram_budget_gb=budget, seed=0, **kw)


# --------------------------------------------------------------------------- #
# (A) ROUTE SENTINEL + factory
# --------------------------------------------------------------------------- #
def test_route_sentinel():
    # blank / absent / null -> None (autofit)
    assert _autofit_vram_budget(None) is None
    assert _autofit_vram_budget("") is None
    assert _autofit_vram_budget("   ") is None
    # explicit number -> passthrough (manual override)
    assert _autofit_vram_budget(8.0) == 8.0
    assert _autofit_vram_budget(6) == 6
    # a bad non-empty value passes through so the FACTORY 400s it (not silently coerced)
    assert _autofit_vram_budget("bad") == "bad"


def test_factory_accepts_none_i2v():
    assert _i2v(None).vram_budget_gb is None
    # explicit still works + coerces to float
    sp2 = _i2v(8)
    assert sp2.vram_budget_gb == 8.0 and isinstance(sp2.vram_budget_gb, float)


@pytest.mark.parametrize("bad", ["bad", 0, -1, [1]])
def test_factory_rejects_bad_i2v(bad):
    with pytest.raises((ValueError, TypeError)):
        _i2v(bad)


def test_route_composition_i2v():
    # the exact route composition: make_studio_i2v(vram_budget_gb=_autofit_vram_budget(raw))
    assert _i2v(_autofit_vram_budget("")).vram_budget_gb is None
    with pytest.raises((ValueError, TypeError)):
        _i2v(_autofit_vram_budget("bad"))    # a bad budget string must still 400


def test_factory_none_movie():
    g = (StudioMovieGoal(segment_id="s0", prompt="a"),)
    sp = make_studio_movie(goals=g, width=256, height=256, fps=8, vram_budget_gb=None)
    assert sp.vram_budget_gb is None
    for bad in ("bad", 0, -1):
        with pytest.raises((ValueError, TypeError)):
            make_studio_movie(goals=g, width=256, height=256, fps=8, vram_budget_gb=bad)


def test_none_roundtrips():
    back = studio_i2v_from_dict(json.loads(json.dumps(asdict(_i2v(None)))))
    assert back.vram_budget_gb is None
    # explicit round-trips too
    assert studio_i2v_from_dict(json.loads(json.dumps(asdict(_i2v(8))))).vram_budget_gb == 8.0
    # movie round-trip
    g = (StudioMovieGoal(segment_id="s0", prompt="a"),)
    msp = make_studio_movie(goals=g, width=256, height=256, fps=8, vram_budget_gb=None)
    mback = studio_movie_from_dict(json.loads(json.dumps(asdict(msp))))
    assert mback.vram_budget_gb is None


# --------------------------------------------------------------------------- #
# (B) AUTOFIT RESOLUTION
# --------------------------------------------------------------------------- #
def test_explicit_bypasses_lookup(monkeypatch):
    """An explicit budget must NEVER touch the worker store."""
    placement.set_worker_registry(_ExplodingRegistry())
    try:
        monkeypatch.setenv("HUGPY_STUDIO_WORKER", "http://10.9.9.9:9100")
        spec, eff, src = S._resolve_autofit(_i2v(8.0))
        assert src == "explicit" and eff == 8.0 and spec.vram_budget_gb == 8.0
    finally:
        placement.set_worker_registry(None)


def test_autofit_sizes_to_worker_capacity(fake_workers, monkeypatch):
    fake_workers([_worker_row("ae", "http://10.9.9.9:7003", 24.0)])
    # HUGPY_STUDIO_WORKER port (9100) differs from the registry row port (7003) —
    # host-only match must still resolve.
    monkeypatch.setenv("HUGPY_STUDIO_WORKER", "http://10.9.9.9:9100")
    spec, eff, src = S._resolve_autofit(_i2v(None))
    # 24 GiB CAPACITY, margin = max(2.4, 2.0) = 2.4 -> 21.6
    assert abs(eff - 21.6) < 1e-6, eff
    assert src == "autofit:ae", src
    assert abs(spec.vram_budget_gb - 21.6) < 1e-6


@pytest.mark.parametrize("total_gib,expected", [
    (40.0, 36.0),   # 10% (4.0) dominates
    (12.0, 10.0),   # 2GB floor (2.0 > 1.2) dominates
])
def test_margin_math(fake_workers, monkeypatch, total_gib, expected):
    fake_workers([_worker_row("box", "http://1.1.1.1:9100", total_gib)])
    monkeypatch.setenv("HUGPY_STUDIO_WORKER", "http://1.1.1.1:9100")
    _sp, eff, _src = S._resolve_autofit(_i2v(None))
    assert abs(eff - expected) < 1e-6, f"expected {expected}; got {eff}"


def test_no_host_match_unresolved(fake_workers, monkeypatch):
    # worker row exists but a DIFFERENT host -> no match -> UNRESOLVED (no fallback number)
    fake_workers([_worker_row("ae", "http://10.0.0.1:9100", 24.0)])
    monkeypatch.setenv("HUGPY_STUDIO_WORKER", "http://192.168.5.5:9100")
    spec, eff, src = S._resolve_autofit(_i2v(None))
    assert eff is None and src == "unresolved", (eff, src)
    # the spec is returned UNCHANGED — no invented budget rides onward
    assert spec.vram_budget_gb is None


def test_no_worker_env_unresolved(fake_workers):
    fake_workers([_worker_row("ae", "http://10.0.0.1:9100", 24.0)])
    _sp, eff, src = S._resolve_autofit(_i2v(None))     # no HUGPY_STUDIO_WORKER
    assert eff is None and src == "unresolved", (eff, src)


def test_no_vram_data_unresolved(fake_workers, monkeypatch):
    # matched box, but it reports no gpus / no VRAM -> UNRESOLVED
    fake_workers([{"name": "gpuless", "url": "http://10.9.9.9:9100", "gpus": []}])
    monkeypatch.setenv("HUGPY_STUDIO_WORKER", "http://10.9.9.9:9100")
    _sp, eff, src = S._resolve_autofit(_i2v(None))
    assert eff is None and src == "unresolved", (eff, src)


# --------------------------------------------------------------------------- #
# (C) render_clip STAMPS the resolved budget (incl. in-process GPU-less central)
# --------------------------------------------------------------------------- #
def test_render_clip_unresolved_refuses(tmp_path):
    """THE REGRESSION FOR THE 2026-07-27 RULING.

    Blank budget + no resolvable studio worker used to become 0.5 -> synthetic -> a
    104KB noise blob written as a finished clip with a share URL. It must now REFUSE.
    """
    spec = _i2v(None, out_root=str(tmp_path))     # no worker -> nothing can size this
    outcome = S.render_clip(spec, render_id="autofit-inproc")
    assert outcome.ok is False, "an unsizable render must refuse, never write a blob"
    assert outcome.error is not None and outcome.error.code == "gpu_unavailable", outcome.error
    assert outcome.budget_source == "unresolved"
    assert outcome.effective_budget_gb is None
    assert outcome.path is None


def test_render_clip_synthetic_is_opt_in(tmp_path):
    """A sub-real EXPLICIT budget binds the synthetic prover, so it must refuse unless
    STUDIO_ALLOW_SYNTHETIC=1 — and must still work when the operator opts in."""
    spec = make_studio_i2v(capability="i2v", width=256, height=256, fps=8,
                           vram_budget_gb=0.5, seed=1, out_root=str(tmp_path))
    outcome = S.render_clip(spec, render_id="autofit-synth-off")
    assert outcome.ok is False, "synthetic must not render while opt-in is off"
    assert outcome.error is not None and outcome.error.code == "no_capable_model", outcome.error
    # the refusal must NAME the budget so the operator can act on it
    assert "0.50" in (outcome.error.message or ""), outcome.error.message


@needs_ffmpeg
def test_render_clip_explicit_stamp(tmp_path, monkeypatch):
    # 0.5 binds the synthetic prover, and synthetic is opt-in since 2026-07-27. This
    # check is about the STAMP, not the tier, so it opts in explicitly — which also
    # covers that the opt-in actually works.
    monkeypatch.setenv("STUDIO_ALLOW_SYNTHETIC", "1")
    spec = make_studio_i2v(capability="i2v", width=256, height=256, fps=8,
                           vram_budget_gb=0.5, seed=1, out_root=str(tmp_path))
    outcome = S.render_clip(spec, render_id="autofit-explicit")
    assert outcome.ok is True, outcome.error
    assert outcome.budget_source == "explicit"
    assert outcome.effective_budget_gb == 0.5


# --------------------------------------------------------------------------- #
# (D) MOVIE autofit — a compact fake worker so an id-movie completes on a fake GPU box
# --------------------------------------------------------------------------- #
_FPS, _W, _H = 12, 320, 180
_SEG_FRAMES = _FPS * 2


class _FakeWorker:
    """Minimal studio-worker HTTP mock (mirrors test_studio_movie_offload): captures each
    delegated spec, renders a solid-gray clip on the SHARED store, scripts a running->done
    poll. Content-addresses by spec so a re-run resumes."""

    def __init__(self):
        self.posts = []
        self.renders = {}
        self._seen = {}

    @staticmethod
    def _key(spec):
        return (spec["out_root"], spec.get("prompt"), spec["seed"],
                spec["width"], spec["height"], spec["fps"], spec["capability"])

    def post(self, url, payload, timeout):
        if url.endswith("/studio/render"):
            rid, spec = payload["job_id"], payload["spec"]
            self.posts.append((rid, spec))
            key = self._key(spec)
            resumed = key in self._seen
            if resumed:
                clip = self._seen[key]
            else:
                clip = os.path.join(spec["out_root"], "_worker", "clip.mp4")
                _build_gray_clip(clip, _SEG_FRAMES, spec["width"], spec["height"], spec["fps"])
                self._seen[key] = clip
            self.renders[rid] = {"polls": 0, "done": {
                "ok": True, "path": clip, "content_hash": f"wc-{abs(hash(key))}",
                "frames": _SEG_FRAMES, "width": spec["width"], "height": spec["height"],
                "duration_s": _SEG_FRAMES / spec["fps"], "resumed": resumed}}
            return 202, {"ok": True, "accepted": "started", "pkg_version": S._pkg_version()}
        return 200, {"cancelled": True}

    def get(self, url, timeout):
        rid = url.rsplit("/", 1)[-1]
        r = self.renders.get(rid)
        if r is None:
            return 200, {"status": "unknown", "result": None}
        r["polls"] += 1
        frames = [{"status": "running", "progress": {"phase": "r"}},
                  {"status": "done", "result": r["done"]}]
        return 200, dict(frames[min(r["polls"] - 1, len(frames) - 1)])


def _build_gray_clip(dst, n_frames, w, h, fps):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fdir = tempfile.mkdtemp(prefix=".wframes-", dir=os.path.dirname(dst))
    try:
        for n in range(n_frames):
            Image.new("RGB", (w, h), (128, 128, 128)).save(os.path.join(fdir, f"f_{n:04d}.png"))
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(fdir, "f_%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), dst],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    finally:
        shutil.rmtree(fdir, ignore_errors=True)


@pytest.fixture
def movie_env(monkeypatch, fake_workers, tmp_path):
    """A fake studio worker on the HTTP seam + a registry row for its host (24 GiB
    capacity -> autofit 21.6) + fast poll/retry timings. The media store's storage
    jail is pointed at ``tmp_path`` so the movie's clips ingest without touching the
    operator's real store."""
    monkeypatch.setattr(media_store, "DEFAULT_ROOT", str(tmp_path))
    fake = _FakeWorker()
    monkeypatch.setattr(S, "_http_post_json", fake.post)
    monkeypatch.setattr(S, "_http_get_json", fake.get)
    fake_workers([_worker_row("ae", "http://10.9.9.9:7003", 24.0)])
    monkeypatch.setenv("HUGPY_STUDIO_WORKER", "http://10.9.9.9:9100")   # host matches the row
    monkeypatch.setenv("HUGPY_STUDIO_POLL_INTERVAL_S", "0.01")
    monkeypatch.setenv("HUGPY_STUDIO_KICKOFF_RETRY_WINDOW_S", "0.2")
    monkeypatch.setenv("HUGPY_STUDIO_KICKOFF_RETRY_INTERVAL_S", "0.02")
    return fake


def _refs(work, n):
    refs = []
    for i in range(n):
        rp = os.path.join(work, f"ref_{i}.png")
        Image.new("RGB", (64, 64), (30 + i * 40, 20, 10)).save(rp)
        refs.append(rp)
    return tuple(refs)


@needs_media_tools
def test_id_movie_autofit(movie_env, tmp_path):
    work = str(tmp_path)
    goals = (StudioMovieGoal(segment_id="s0", prompt="her on the beach"),
             StudioMovieGoal(segment_id="s1", prompt="playing volleyball",
                             parent_segment_id="s0", joint_mode="cut", seed=22))
    spec = make_studio_movie(goals=goals, width=_W, height=_H, fps=_FPS,
                             vram_budget_gb=None,   # BLANK -> autofit
                             seed=21, out_root=work, reference_images=_refs(work, 2))
    res = run_generate_studio_movie(spec, job_id="idm-autofit")
    assert res.ok is True, f"autofit id-movie must complete; got {res.error}"
    # Every delegated segment carried the AUTOFIT budget (21.6), well above the VACE floor.
    assert len(movie_env.posts) == 2, f"both segments must delegate; got {len(movie_env.posts)}"
    for rid, dspec in movie_env.posts:
        assert dspec["capability"] == "id_lock", rid
        assert abs(dspec["vram_budget_gb"] - 21.6) < 1e-6, (rid, dspec["vram_budget_gb"])
        assert dspec["vram_budget_gb"] >= 6.0, "autofit budget must clear the VACE floor"
    # movie.json / seg records carry the RESOLVED budget + budget_source.
    for seg in res.movie["segments"]:
        assert seg["budget_source"] == "autofit:ae", seg.get("budget_source")
        assert abs(seg["vram_budget_gb"] - 21.6) < 1e-6, seg["vram_budget_gb"]
        assert seg["vram_budget_gb"] >= 6.0


@needs_media_tools
def test_id_movie_explicit_regression(movie_env, tmp_path):
    """A live worker row exists, but an EXPLICIT budget must BYPASS the lookup entirely."""
    work = str(tmp_path)
    goals = (StudioMovieGoal(segment_id="s0", prompt="her on the beach"),)
    spec = make_studio_movie(goals=goals, width=_W, height=_H, fps=_FPS,
                             vram_budget_gb=8.0,    # EXPLICIT -> unchanged (floored to >=6)
                             seed=7, out_root=work, reference_images=_refs(work, 1))
    res = run_generate_studio_movie(spec, job_id="idm-explicit")
    assert res.ok is True, res.error
    (_rid, dspec), = movie_env.posts
    # Byte-identical to today: explicit 8.0 floored to max(8,6)=8.0, source "explicit".
    assert dspec["vram_budget_gb"] == 8.0
    seg = res.movie["segments"][0]
    assert seg["budget_source"] == "explicit", seg.get("budget_source")
    assert seg["vram_budget_gb"] == 8.0
