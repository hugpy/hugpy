"""Studio worker PLACEMENT — conformance.

Operator ruling (2026-09-24): a studio (video) render is placed LIKE AN LLM — hugpy's
worker registry picks the GPU worker (online + advertises studio render + holds the bound
model's weights on disk + feasible VRAM). It must NOT depend on the central env var
``HUGPY_STUDIO_WORKER`` to reach a GPU (that var is retained only as an explicit operator
OVERRIDE). When nothing is feasible a REAL-model render gets a NAMED refusal — never a
silent fall-back to central's GPU-less in-process path (the ``deps_missing`` incident this
fixes). A SYNTHETIC render still runs in-process, unchanged.

What is under test:
  * resolver PICKS the feasible worker (render-capable, holds the model, usable VRAM);
  * resolver HONORS the ``HUGPY_STUDIO_WORKER`` override without consulting the registry;
  * NAMED refusals: no render-capable worker / model files on no worker / no feasible VRAM;
  * ``render_clip`` REFUSES a real-model render with no worker (never in-process), while a
    SYNTHETIC render is NOT placement-refused (stays in-process);
  * ``should_delegate`` (the routes' capability probe reuses it) AGREES with the resolver;
  * the worker-side presence signal (``studio.presence.present_model_ids``) reads the disk;
  * the capacity budget is ONE source shared with autofit.

The worker registry the resolver reads is the ``hugpy_engine.placement`` seam; each check
installs a fake registry with known rows and RESTORES it (autouse fixture) so no patch
leaks into an unrelated test.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from hugpy_engine import placement
from hugpy_video.intel import media_bus
from hugpy_video.intel.runners import studio_i2v as S
from hugpy_video.intel.runners import studio_placement as SP
from hugpy_video.intel.studio.job import make_studio_i2v
from hugpy_video.intel.studio.presence import present_model_ids, weights_roots

_GIB = 1024 ** 3
_ENV_KEYS = ("HUGPY_STUDIO_WORKER", "HUGPY_STUDIO_FORCE_REMOTE",
             "STUDIO_WEIGHTS_ROOT", "STUDIO_WEIGHTS_HOT_ROOT")
_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not _FFMPEG, reason="ffmpeg unavailable")

# A real Wan model that binds at a modest budget, and its on-disk weight layout.
_REAL_MODEL = "wan2.1-t2v-1.3b"
_REAL_WEIGHT_URI = "Wan-AI/Wan2.1-T2V-1.3B"


class _FakeRegistry:
    def __init__(self, rows):
        self._rows = list(rows)

    def list_workers(self, online_only=True):
        return list(self._rows)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No studio env leaking in, unpinned registry ok, cancel probe says 'no', and the
    placement seam is RESET after every test (so a fake registry never leaks)."""
    monkeypatch.setenv("STUDIO_ALLOW_UNPINNED", "1")
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(media_bus, "is_cancelling", lambda job_id: False)
    yield
    placement.set_worker_registry(None)


def _install(rows):
    placement.set_worker_registry(_FakeRegistry(rows))


def _row(name, url, total_gib, models, render=True, free_gib=0.5):
    return {
        "name": name, "url": url,
        "gpus": [{"index": 0, "memory_total": int(total_gib * _GIB),
                  "memory_free": int(free_gib * _GIB)}],
        "studio": {"render": render, "models": list(models),
                   "weights_root": "/mnt/llm_storage"},
    }


def _real(budget=8.0, **kw):
    return make_studio_i2v(capability="t2v", width=832, height=480, fps=16,
                           vram_budget_gb=budget, seed=0, prompt="a shot", **kw)


def _synth(**kw):
    # i2v at a sub-real budget binds the synthetic prover (resolves_to_real_model False).
    return make_studio_i2v(capability="i2v", width=64, height=64, fps=8,
                           vram_budget_gb=0.5, seed=1, **kw)


# --------------------------------------------------------------------------- #
# (A) resolver picks the feasible worker
# --------------------------------------------------------------------------- #
def test_picks_feasible_worker():
    _install([_row("ae", "http://10.0.0.9:7003", 24.0, [_REAL_MODEL])])
    url, refusal = SP.resolve_worker(_real())
    assert refusal is None
    assert url == "http://10.0.0.9:7003"


def test_prefers_larger_capacity():
    # Both hold the model + are render-capable; the best VRAM fit wins (stable pick).
    _install([_row("small", "http://10.0.0.5:7003", 12.0, [_REAL_MODEL]),
              _row("big", "http://10.0.0.9:7003", 48.0, [_REAL_MODEL])])
    url, refusal = SP.resolve_worker(_real())
    assert refusal is None and url == "http://10.0.0.9:7003"


# --------------------------------------------------------------------------- #
# (B) explicit override
# --------------------------------------------------------------------------- #
def test_env_override_wins_without_registry(monkeypatch):
    # An exploding registry proves the override never consults placement.
    class _Boom:
        def list_workers(self, online_only=True):
            raise AssertionError("override must not read the worker registry")
    placement.set_worker_registry(_Boom())
    monkeypatch.setenv("HUGPY_STUDIO_WORKER", "http://1.2.3.4:9100/")
    url, refusal = SP.resolve_worker(_real())
    assert refusal is None and url == "http://1.2.3.4:9100"   # trailing slash stripped


# --------------------------------------------------------------------------- #
# (C) named refusals
# --------------------------------------------------------------------------- #
def test_refuse_no_render_capable_worker():
    # A worker exists but does NOT advertise studio render.
    _install([_row("nogpu", "http://10.0.0.9:7003", 24.0, [_REAL_MODEL], render=False)])
    url, refusal = SP.resolve_worker(_real())
    assert url == ""
    assert refusal is not None and refusal.code == "no_studio_worker"
    assert refusal.retryable is True
    assert "render" in refusal.message


def test_refuse_model_not_on_worker():
    # Render-capable + usable VRAM, but the bound model's weights are on NO worker.
    _install([_row("ae", "http://10.0.0.9:7003", 24.0, [])])
    url, refusal = SP.resolve_worker(_real())
    assert url == ""
    assert refusal is not None and refusal.code == "studio_model_not_on_worker"
    assert refusal.retryable is False        # deterministic: needs operator placement
    assert _REAL_MODEL in refusal.message


def test_refuse_no_feasible_vram():
    # Render-capable + holds the model, but reports NO usable VRAM (no gpus).
    _install([{"name": "ae", "url": "http://10.0.0.9:7003", "gpus": [],
               "studio": {"render": True, "models": [_REAL_MODEL]}}])
    url, refusal = SP.resolve_worker(_real())
    assert url == ""
    assert refusal is not None and refusal.code == "no_feasible_studio_worker"


def test_refuse_no_workers_at_all():
    _install([])
    url, refusal = SP.resolve_worker(_real())
    assert url == "" and refusal is not None and refusal.code == "no_studio_worker"


# --------------------------------------------------------------------------- #
# (D) render_clip: real refuses (never in-process), synthetic is not placement-refused
# --------------------------------------------------------------------------- #
def test_render_clip_real_refuses_not_in_process(tmp_path):
    """THE LIVE-SYMPTOM REGRESSION: a real-model render with no studio worker must NOT
    run on the GPU-less control plane (deps_missing) — it must refuse, named."""
    _install([])                                   # no worker at all
    outcome = S.render_clip(_real(out_root=str(tmp_path)), render_id="real-noworker")
    assert outcome.ok is False
    assert outcome.error is not None and outcome.error.code == "no_studio_worker"
    assert outcome.error.retryable is True
    assert outcome.path is None                    # nothing rendered in-process


def test_render_clip_real_missing_files_refuses(tmp_path):
    _install([_row("ae", "http://10.0.0.9:7003", 24.0, [])])   # render-capable, no files
    outcome = S.render_clip(_real(out_root=str(tmp_path)), render_id="real-nofiles")
    assert outcome.ok is False
    assert outcome.error.code == "studio_model_not_on_worker"
    assert outcome.error.retryable is False
    assert outcome.path is None


def test_synthetic_is_not_placement_real(tmp_path):
    """A synthetic render is never treated as a real placement target — the decision
    that keeps it in-process (ffmpeg-free assertion)."""
    _install([])
    assert S.resolves_to_real_model(_synth(out_root=str(tmp_path))) is False
    assert S.should_delegate(_synth(out_root=str(tmp_path))) is False


@needs_ffmpeg
def test_synthetic_renders_in_process(tmp_path, monkeypatch):
    """With the opt-in, a synthetic render still completes IN-PROCESS with NO worker —
    placement never refuses it."""
    monkeypatch.setenv("STUDIO_ALLOW_SYNTHETIC", "1")
    _install([])
    outcome = S.render_clip(_synth(out_root=str(tmp_path)), render_id="synth-inproc")
    assert outcome.error is None or outcome.error.code not in (
        "no_studio_worker", "studio_worker_unreachable", "no_feasible_studio_worker",
        "studio_model_not_on_worker"), outcome.error
    assert outcome.ok is True, outcome.error


# --------------------------------------------------------------------------- #
# (E) should_delegate agrees with the resolver (the routes' capability probe reuses it)
# --------------------------------------------------------------------------- #
def test_should_delegate_agrees_with_resolver():
    real = _real()
    # feasible worker present -> both say GO
    _install([_row("ae", "http://10.0.0.9:7003", 24.0, [_REAL_MODEL])])
    assert SP.resolve_worker(real)[0] != ""
    assert S.should_delegate(real) is True
    # no feasible worker -> both say NO
    _install([_row("ae", "http://10.0.0.9:7003", 24.0, [])])
    assert SP.resolve_worker(real)[0] == ""
    assert S.should_delegate(real) is False


def test_should_delegate_override(monkeypatch):
    monkeypatch.setenv("HUGPY_STUDIO_WORKER", "http://127.0.0.1:9100")
    assert S.should_delegate(_real()) is True         # override + real model
    assert S.should_delegate(_synth()) is False       # synthetic never delegates


# --------------------------------------------------------------------------- #
# (F) worker-side signal helpers + disk presence
# --------------------------------------------------------------------------- #
def test_worker_signal_parsers():
    w = _row("ae", "http://x:1", 24.0, [_REAL_MODEL])
    assert SP.worker_advertises_studio_render(w) is True
    assert _REAL_MODEL in SP.worker_studio_models(w)
    assert SP.worker_advertises_studio_render({"studio": {"render": False}}) is False
    assert SP.worker_advertises_studio_render({}) is False
    assert SP.worker_studio_models({}) == frozenset()


def test_capacity_budget_matches_autofit_margin():
    # 24 GiB capacity, margin = max(2.4, 2.0) = 2.4 -> 21.6 (the autofit number).
    w = _row("ae", "http://x:1", 24.0, [_REAL_MODEL])
    assert abs(SP.worker_capacity_budget_gb(w) - 21.6) < 1e-6
    # a box that reports no VRAM -> None (caller refuses)
    assert SP.worker_capacity_budget_gb({"gpus": []}) is None


def test_presence_reads_disk(tmp_path, monkeypatch):
    root = str(tmp_path)
    model_dir = os.path.join(root, "Wan-AI", "Wan2.1-T2V-1.3B")
    os.makedirs(model_dir)
    # no marker yet -> not present
    assert _REAL_MODEL not in present_model_ids((root,))
    with open(os.path.join(model_dir, "model_index.json"), "w") as fh:
        fh.write("{}")
    assert _REAL_MODEL in present_model_ids((root,))
    # env-driven roots
    monkeypatch.setenv("STUDIO_WEIGHTS_ROOT", root)
    assert weights_roots() == (root,)
    assert _REAL_MODEL in present_model_ids()


def test_presence_no_roots_is_empty():
    assert present_model_ids(()) == ()


def test_studio_weights_root_is_manifest_single_source(tmp_path):
    """``job.studio_weights_root()`` is the SINGLE source of the shared weights root: the
    exact ``weights_root`` ``resolve_studio_env`` stamps into the render manifest. Central
    advertises this to workers so a worker's presence probe checks the SAME root the render
    loads from."""
    from hugpy_video.intel.studio.job import resolve_studio_env, studio_weights_root
    root = studio_weights_root()
    assert isinstance(root, str) and root
    env = resolve_studio_env(str(tmp_path), master_fps=16)
    assert env.weights_root == root


def test_studio_weights_root_honors_env(monkeypatch):
    """An explicit STUDIO_WEIGHTS_ROOT (the shared mount) wins over the DEFAULT_ROOT
    default."""
    from hugpy_video.intel.studio.job import studio_weights_root
    monkeypatch.setenv("STUDIO_WEIGHTS_ROOT", "/mnt/16T/llm_storage/video_intel/studio/weights")
    assert studio_weights_root() == "/mnt/16T/llm_storage/video_intel/studio/weights"
