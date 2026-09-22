"""IDENTITY CLEANUP-PROMPT (C2 + C3) — the render-prompt wiring (operator-requested
2026-07-15). NO GPU, NO network: every render seam is mocked and the assertions are on
the PROMPT / NEGATIVE actually handed down.

C2 — T-pose front render (identity_render_relay):
  * ``_tpose_prompt("")`` == ``_TPOSE_PROMPT`` VERBATIM (byte-identical to today).
  * ``_tpose_prompt("no object on her back")`` == ``_TPOSE_PROMPT + ", no object on her
    back"`` (a cleanup clause rides the stance; it does not fight it).
  * ``_render_pose_front(..., cleanup_prompt=, negative_prompt=)`` passes the assembled
    prompt AND the negative down into ``_render_identity_view`` (mocked to capture).
  * empty cleanup + empty negative -> the constant + "" negative (regression-safe).

C3 — reconstruction render seam (identity_reconstruction._render_identity_view):
  * ``_render_identity_view(..., negative_prompt=)`` FORWARDS the negative into the studio
    call (``make_studio_i2v(negative=...)``, mocked to capture the kwarg).
  * empty negative reaches the studio call as "" (== today's byte-identical render, since
    the studio path already defaults negative_prompt="").

This file mocks at the LAZY-IMPORT SOURCE modules (the runners import their seams inside
the function body via ``from .identity_reconstruction import _render_identity_view`` /
``from ..studio.job import make_studio_i2v``), so patching the source-module attribute is
what the runtime import resolves to.

Run (as pytest and as a script; run ALONE — the identity test family cross-pollutes):
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python -m pytest tests/test_identity_cleanup_prompt_wiring.py -q
  venv/bin/python tests/test_identity_cleanup_prompt_wiring.py
"""
from __future__ import annotations

import logging
import os
import sys

logging.disable(logging.INFO)

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_video.intel.runners import identity_render_relay
from hugpy_video.intel.runners import identity_reconstruction


# --------------------------------------------------------------------------- #
# C2 (i): _tpose_prompt assembly — the byte-identical-empty guarantee.
# --------------------------------------------------------------------------- #
def test_tpose_prompt_empty_is_constant_verbatim():
    # THE regression guarantee: an empty (or whitespace-only) cleanup clause reproduces the
    # exact _TPOSE_PROMPT constant — byte-for-byte, no trailing comma, no whitespace.
    assert identity_render_relay._tpose_prompt("") == identity_render_relay._TPOSE_PROMPT
    assert identity_render_relay._tpose_prompt() == identity_render_relay._TPOSE_PROMPT
    assert identity_render_relay._tpose_prompt("   ") == identity_render_relay._TPOSE_PROMPT
    assert identity_render_relay._tpose_prompt(None) == identity_render_relay._TPOSE_PROMPT


def test_tpose_prompt_appends_cleanup_clause():
    cleanup = "no object on her back, clean bare back"
    expected = f"{identity_render_relay._TPOSE_PROMPT}, {cleanup}"
    assert identity_render_relay._tpose_prompt(cleanup) == expected
    # surrounding whitespace on the clause is stripped (no double-space / dangling space)
    assert identity_render_relay._tpose_prompt("  " + cleanup + "  ") == expected


# --------------------------------------------------------------------------- #
# C2 (ii): _render_pose_front hands the assembled prompt + negative to the seam.
# --------------------------------------------------------------------------- #
class _ViewCapture:
    """Stand-in for identity_reconstruction._render_identity_view — records the prompt and
    every kwarg, returns a configured still path (or None)."""

    def __init__(self, returns="/rendered/tpose.png"):
        self.returns = returns
        self.calls: list[dict] = []

    def __call__(self, refs, prompt, seed, **kwargs):
        self.calls.append({"refs": tuple(refs), "prompt": prompt, "seed": seed, **kwargs})
        return self.returns


def _patch_view(returns="/rendered/tpose.png"):
    cap = _ViewCapture(returns)
    orig = identity_reconstruction._render_identity_view
    identity_reconstruction._render_identity_view = cap
    return cap, orig


def test_pose_front_forwards_cleanup_and_negative():
    cap, orig = _patch_view()
    try:
        out = identity_render_relay._render_pose_front(
            ["/ref/a.png"], seed=7, slug="s", job_id="j1",
            cleanup_prompt="no ball on back", negative_prompt="backpack, symbols")
    finally:
        identity_reconstruction._render_identity_view = orig
    assert out == "/rendered/tpose.png"
    assert len(cap.calls) == 1, cap.calls
    call = cap.calls[0]
    # the assembled cleanup prompt reached the seam
    assert call["prompt"] == (
        identity_render_relay._TPOSE_PROMPT + ", no ball on back"), call["prompt"]
    # the negative was forwarded through as a kwarg
    assert call["negative_prompt"] == "backpack, symbols", call


def test_pose_front_empty_is_constant_and_empty_negative():
    # THE byte-identical-empty guarantee at the pose-render seam: empty cleanup + empty
    # negative -> the exact constant prompt and negative_prompt="".
    cap, orig = _patch_view()
    try:
        identity_render_relay._render_pose_front(
            ["/ref/a.png"], seed=1, slug="s", job_id="j2")  # both default ""
    finally:
        identity_reconstruction._render_identity_view = orig
    call = cap.calls[0]
    assert call["prompt"] == identity_render_relay._TPOSE_PROMPT, call["prompt"]
    assert call["negative_prompt"] == "", call


def test_pose_front_degrades_to_none_on_seam_failure():
    # HONEST DEGRADE unchanged: a seam that returns None -> _render_pose_front returns None
    # (the caller falls back to front-select; the mesh job never fails on the pose stage).
    cap, orig = _patch_view(returns=None)
    try:
        out = identity_render_relay._render_pose_front(
            ["/ref/a.png"], seed=1, slug="s", job_id="j3", cleanup_prompt="x")
    finally:
        identity_reconstruction._render_identity_view = orig
    assert out is None


def test_pose_front_degrades_to_none_on_seam_raise():
    # Even a genuine raise inside the seam degrades to None (the try/except wrap).
    def _boom(*a, **k):
        raise RuntimeError("studio exploded")
    orig = identity_reconstruction._render_identity_view
    identity_reconstruction._render_identity_view = _boom
    try:
        out = identity_render_relay._render_pose_front(
            ["/ref/a.png"], seed=1, slug="s", job_id="j4", negative_prompt="y")
    finally:
        identity_reconstruction._render_identity_view = orig
    assert out is None


# --------------------------------------------------------------------------- #
# C3: _render_identity_view forwards negative into the studio call.
# --------------------------------------------------------------------------- #
class _StudioSpecCapture:
    """Stand-in for studio.job.make_studio_i2v — records every kwarg (esp. ``negative``)
    and returns a trivial sentinel spec object (render_clip is also mocked, so the spec is
    never actually used to render)."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        return object()  # opaque sentinel — render_clip is mocked below


def _patch_studio(clip_ok=False):
    """Patch make_studio_i2v (capture) + render_clip (short-circuit) at their SOURCE
    modules — _render_identity_view lazy-imports them from ..studio.job / .studio_i2v.
    Returns (capture, restore_fn)."""
    from hugpy_video.intel.studio import job as studio_job
    from hugpy_video.intel.runners import studio_i2v

    cap = _StudioSpecCapture()
    orig_make = studio_job.make_studio_i2v
    studio_job.make_studio_i2v = cap

    class _Outcome:
        ok = clip_ok
        path = None
        error = None
    orig_render = studio_i2v.render_clip
    studio_i2v.render_clip = lambda spec, *, render_id, should_cancel=None, **kw: _Outcome()

    def _restore():
        studio_job.make_studio_i2v = orig_make
        studio_i2v.render_clip = orig_render
    return cap, _restore


def test_render_identity_view_forwards_negative():
    cap, restore = _patch_studio()
    try:
        # render_clip returns ok=False -> _render_identity_view returns None (we only care
        # that make_studio_i2v was called with the forwarded negative).
        out = identity_reconstruction._render_identity_view(
            ["/ref/a.png"], "front view", 3,
            width=480, height=480, fps=16, render_id="rid",
            negative_prompt="backpack, symbols")
    finally:
        restore()
    assert out is None  # render short-circuited
    assert len(cap.calls) == 1, cap.calls
    assert cap.calls[0]["negative"] == "backpack, symbols", cap.calls[0]
    assert cap.calls[0]["prompt"] == "front view", cap.calls[0]


def test_render_identity_view_empty_negative_is_empty_string():
    # THE byte-identical-empty guarantee at the reconstruction seam: an unset negative
    # reaches make_studio_i2v as "" (== today's render — make_studio_i2v was previously
    # called with NO negative -> None -> run_produce_clip's `None or ""` == ""; now it is
    # called with "" -> run_produce_clip's `"" or ""` == "". Same "" into produce_clip).
    cap, restore = _patch_studio()
    try:
        identity_reconstruction._render_identity_view(
            ["/ref/a.png"], "back view", 3,
            width=480, height=480, fps=16, render_id="rid")  # negative_prompt defaults ""
    finally:
        restore()
    assert cap.calls[0]["negative"] == "", cap.calls[0]


def test_render_identity_turntable_forwards_negative():
    # The turntable reconstruction render honors the negative too.
    cap, restore = _patch_studio()
    try:
        out = identity_reconstruction._render_identity_turntable(
            ["/ref/a.png"], "orbit prompt", 3,
            width=480, height=480, fps=16, render_id="rid", max_frames=240,
            negative_prompt="prop")
    finally:
        restore()
    assert out is None
    assert cap.calls[0]["negative"] == "prop", cap.calls[0]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} CLEANUP-PROMPT WIRING CHECKS PASSED")
