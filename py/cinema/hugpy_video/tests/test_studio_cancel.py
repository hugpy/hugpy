"""Task 1 — single-shot mid-render cancel conformance.

Locks the cooperative-cancel slice as executable checks, in the same script style
as ``test_studio_conformance.py`` / ``test_studio_prompt.py`` (plain python,
``__main__`` guard, numbered ``[n] PASS`` / ``[n] FAIL`` lines, nonzero exit iff
any check FAILED, every check independently run so a failing one never masks the
rest). pytest is NOT installed in this venv, so there are no fixtures.

Invariant under test: a RUNNING studio i2v job is stoppable mid-render via a
caller-supplied ``should_cancel`` probe threaded DOWN the produce path
(bus adapter -> produce_clip -> runner). A cancel aborts BEFORE the atomic
os.replace, so NO clip lands at the content-addressed path — resume/idempotency
stay intact (a re-run regenerates). The studio spine stays media_bus-free; only
the bus adapter (video_intel/runners/studio_i2v.py) sources the probe.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_cancel.py
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import time

logging.disable(logging.INFO)

os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_video.intel.studio.enums import Capability
from hugpy_video.intel.studio.env import StudioEnv
from hugpy_video.intel.studio.errors import ErrorCode, StageError
from hugpy_video.intel.studio.job import make_studio_i2v
from hugpy_video.intel.studio.produce import produce_clip
from hugpy_video.intel.studio.schemas import CapabilityRequest, Resolution
from hugpy_video.intel import media_bus
from hugpy_video.intel.runners.studio_i2v import run_studio_i2v

_FFMPEG = shutil.which("ffmpeg") is not None


def _studio_env(master_fps: int = 12) -> StudioEnv:
    return StudioEnv(
        output_root="/out",
        weights_root="/weights",
        manifest_root="/manifests",
        master_colorspace="rec709",
        master_fps=master_fps,
        max_vram_gb=24.0,
        loudness_target_lufs=-14.0,
        allow_unpinned=True,
    )


def _synth_request(*, res=Resolution(320, 180, 12), budget=0.5) -> CapabilityRequest:
    """CAP-I2V at a VRAM budget too small for any real model (min real i2v = 8GB),
    so the router deterministically binds the tiny synthetic-i2v model (fps*2 = 24
    frames — plenty of headroom to cancel after the 2nd)."""
    return CapabilityRequest(
        capability=Capability.I2V, target_resolution=res, vram_budget_gb=budget)


def _clip_files_under(root: str) -> list[str]:
    """Every clip.mp4 anywhere under ``root`` — the abort-before-replace guarantee
    is proven by this list being EMPTY after a cancelled render."""
    found = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn == "clip.mp4":
                found.append(os.path.join(dirpath, fn))
    return found


# --------------------------------------------------------------------------- #
# (i) produce_clip cancelled after the 2nd frame -> Err(cancelled), NO clip
# --------------------------------------------------------------------------- #
def test_produce_clip_cancel_mid_render_leaves_no_clip():
    env = _studio_env()
    req = _synth_request()
    out_root = tempfile.mkdtemp(prefix="studio-cancel-")

    # should_cancel is polled at the TOP of each frame: call 1 (frame 0) -> False,
    # call 2 (frame 1) -> False, call 3 (frame 2) -> True => abort after 2 frames.
    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        return calls["n"] > 2

    try:
        res = produce_clip(req, env=env, out_root=out_root, should_cancel=should_cancel)
        assert res.is_err(), f"a mid-render cancel must return Err, not Ok; got {res}"
        assert isinstance(res.error, StageError), "Err payload must be a StageError value"
        assert res.error.code == ErrorCode.CANCELLED, (
            f"cancel must map to ErrorCode.CANCELLED; got {res.error.code}")
        assert res.error.code.value == "cancelled", (
            "the code string must be 'cancelled' (media_bus terminal vocab)")
        # aborted AFTER 2 frames (the 3rd poll tripped it), BEFORE the atomic replace
        assert calls["n"] == 3, f"should_cancel must be polled per frame; got {calls['n']}"
        # THE guarantee: no clip.mp4 landed anywhere under out_root
        leftovers = _clip_files_under(out_root)
        assert leftovers == [], (
            f"a cancelled render must leave NO clip.mp4 (abort before os.replace); "
            f"found {leftovers}")
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (ii) regression: should_cancel=None (and always-False) -> a normal clip
# --------------------------------------------------------------------------- #
def test_produce_clip_no_cancel_regression():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping no-cancel regression clip check)")
        return
    env = _studio_env()
    req = _synth_request()

    # (a) should_cancel omitted (None default) -> historical behavior, a real clip
    out_a = tempfile.mkdtemp(prefix="studio-nocancel-none-")
    try:
        res = produce_clip(req, env=env, out_root=out_a)
        assert res.is_ok(), f"produce_clip without should_cancel must yield Ok; got {res}"
        art = res.unwrap()
        assert os.path.isfile(art.path) and os.path.getsize(art.path) > 0, "clip mp4 non-empty"
        assert os.path.basename(art.path) == "clip.mp4"
    finally:
        shutil.rmtree(out_a, ignore_errors=True)

    # (b) should_cancel present but ALWAYS FALSE -> still a real clip (never trips)
    out_b = tempfile.mkdtemp(prefix="studio-nocancel-false-")
    try:
        res = produce_clip(req, env=env, out_root=out_b, should_cancel=lambda: False)
        assert res.is_ok(), f"an always-False should_cancel must yield Ok; got {res}"
        art = res.unwrap()
        assert os.path.isfile(art.path) and os.path.getsize(art.path) > 0, "clip mp4 non-empty"
    finally:
        shutil.rmtree(out_b, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (iii) run_studio_i2v (bus adapter) with is_cancelling monkeypatched True ->
#       JobResult(ok=False, error.code=="cancelled", retryable=False), NO clip
# --------------------------------------------------------------------------- #
def test_run_studio_i2v_cancel_maps_to_job_error():
    spec = make_studio_i2v(
        width=320, height=180, fps=12, vram_budget_gb=0.5, seed=0,
        out_root=tempfile.mkdtemp(prefix="studio-bus-cancel-"))

    orig = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: True   # cancel is pending from frame 0
    try:
        result = run_studio_i2v(spec, job_id="cancel-job-1")
        assert result.ok is False, f"a cancelled bus job must be ok=False; got {result}"
        assert result.error is not None, "a cancelled job must carry a JobError"
        assert result.error.code == "cancelled", (
            f"the bus adapter must map CANCELLED -> code 'cancelled'; got {result.error.code}")
        assert result.error.retryable is False, (
            "a cancel is intentional and must NOT be retryable")
        assert result.outputs == (), "a cancelled job must produce no outputs"
        leftovers = _clip_files_under(spec.out_root)
        assert leftovers == [], (
            f"a cancelled bus job must leave NO clip.mp4; found {leftovers}")
    finally:
        media_bus.is_cancelling = orig
        shutil.rmtree(spec.out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (iv) DELEGATED cancel is retried until DELIVERED. A cancel POST that raises
#      used to still set cancel_sent=True, so one transient blip on the worker
#      link dropped the user's cancel forever and the render ran to its 30-min
#      budget while the console said "cancelling". The worker link is known to
#      blip, so the one-shot was a real loss of intent.
# --------------------------------------------------------------------------- #
def test_delegated_cancel_retries_until_delivered():
    from hugpy_video.intel.runners import studio_i2v as s

    posts = []          # every cancel POST attempt, in order
    attempts = {"n": 0}

    def fake_post(url, payload, timeout):
        if url.endswith("/studio/cancel/rid-1"):
            posts.append(url)
            attempts["n"] += 1
            if attempts["n"] < 3:        # first TWO attempts blip
                raise OSError("connection refused")
            return 200, {"ok": True}
        return 200, {"job_id": "rid-1", "accepted": "running"}   # kick-off

    def fake_get(url, timeout):
        # Settle only AFTER the cancel actually lands, so the loop must retry.
        # A honored cancel comes back as the worker's ERROR-AS-DATA terminal
        # (status 'error' + an Err(CANCELLED) result), NOT status 'cancelled' —
        # the delegation loop's terminals are exactly ('done', 'error'), so any
        # other string reads as "still running" and polls to the budget.
        if attempts["n"] >= 3:
            return 200, {"status": "error",
                         "result": {"ok": False,
                                    "error": {"code": "cancelled",
                                              "message": "cancelled",
                                              "retryable": False}}}
        return 200, {"status": "running"}

    # BOUND the delegation. Without this the OLD one-shot behavior doesn't fail the
    # assertion — it HANGS: the cancel is dropped, the worker never settles, and the
    # loop polls until the 30-min render budget. A short budget turns that into a
    # fast, legible failure instead of a wedged test run.
    orig_env = {k: os.environ.get(k) for k in
                ("HUGPY_STUDIO_DELEGATE_TIMEOUT_S", "HUGPY_STUDIO_OVERALL_CAP_S",
                 "HUGPY_STUDIO_POLL_INTERVAL_S")}
    os.environ["HUGPY_STUDIO_DELEGATE_TIMEOUT_S"] = "3"
    os.environ["HUGPY_STUDIO_OVERALL_CAP_S"] = "3"
    os.environ["HUGPY_STUDIO_POLL_INTERVAL_S"] = "0.01"

    orig_post, orig_get, orig_sleep = s._http_post_json, s._http_get_json, time.sleep
    s._http_post_json, s._http_get_json = fake_post, fake_get
    time.sleep = lambda _s: None          # don't actually wait out the poll ticks
    try:
        spec = make_studio_i2v(width=320, height=180, fps=12, vram_budget_gb=0.5,
                               seed=0, out_root=tempfile.mkdtemp(prefix="studio-delegcancel-"))
        outcome = s._delegate_to_worker(
            "http://worker.invalid", spec, "rid-1",
            should_cancel=lambda: True, progress_sink=lambda _b: None)
    finally:
        s._http_post_json, s._http_get_json = orig_post, orig_get
        time.sleep = orig_sleep
        for k, v in orig_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(spec.out_root, ignore_errors=True)

    assert attempts["n"] >= 3, (
        "a cancel POST that raised must be RETRIED on the next poll tick; "
        f"only {attempts['n']} attempt(s) were made — the intent was dropped")
    assert outcome.ok is False, f"a cancelled delegation must be ok=False; got {outcome}"
    assert outcome.error is not None and outcome.error.code == "cancelled", (
        f"a delivered cancel must settle as code 'cancelled'; got {outcome.error}")


CHECKS = [
    ("produce_clip cancelled after 2nd frame -> Err(cancelled) + NO clip.mp4",
     test_produce_clip_cancel_mid_render_leaves_no_clip),
    ("regression: should_cancel None / always-False -> a normal clip",
     test_produce_clip_no_cancel_regression),
    ("run_studio_i2v + is_cancelling=True -> JobResult(ok=False, 'cancelled', not retryable)",
     test_run_studio_i2v_cancel_maps_to_job_error),
    ("delegated cancel POST that blips is RETRIED until delivered (not dropped)",
     test_delegated_cancel_retries_until_delivered),
]


def main() -> int:
    passed = 0
    failed = 0
    for i, (name, fn) in enumerate(CHECKS, 1):
        try:
            fn()
        except Exception as exc:  # surface EVERY divergence, not just the first
            failed += 1
            print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"[{i}] PASS  {name}")
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
