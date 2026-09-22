"""Randomized steering for the LLM prompt-assist "generate" mode.

Operator, 2026-07-27: *"i need generate in the /video (the llm generate) to
randomize the prompt"*. ``POST /video/prompt/assist`` with ``mode="generate"``
sent ONE fixed instruction every call, so a small instruct model returned the
same handful of prompts. These lock the fix: the brief must actually differ,
a caller's draft must survive it, and Enhance must not be steered at all.

Run: venv/bin/python -m pytest tests/test_prompt_seeds.py -q
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_video.intel import prompt_seeds as PS


def test_every_axis_is_present_for_a_still():
    axes = PS.steering_axes("image", rng=random.Random(1))
    assert set(axes) == {"subject", "setting", "light", "mood", "palette", "shot"}


def test_video_kinds_add_motion():
    """A t2v/i2v prompt that names no movement wastes the medium."""
    for kind in ("movie", "clip"):
        assert "motion" in PS.steering_axes(kind, rng=random.Random(2)), kind
    assert "motion" not in PS.steering_axes("image", rng=random.Random(2))


def test_a_draft_keeps_its_subject():
    """THE CONTRACT: steering may colour the shot, never replace what the
    operator asked for. With a draft, the SUBJECT axis is dropped entirely."""
    axes = PS.steering_axes("movie", has_draft=True, rng=random.Random(3))
    assert "subject" not in axes
    assert "setting" in axes and "motion" in axes      # the rest still steer


def test_successive_briefs_actually_differ():
    """The whole point. 40 draws must not collapse onto a few briefs."""
    seen = {PS.steering_clause("movie") for _ in range(40)}
    assert len(seen) >= 35, f"only {len(seen)} distinct briefs in 40 draws"


def test_the_noticeable_axes_never_repeat_back_to_back():
    """Uniform sampling over 12 subjects repeats often — observed twice in the
    first three live calls, which READS as 'it didn't randomize'. Subject and
    setting must never repeat consecutively."""
    subs, sets_ = [], []
    for _ in range(30):
        a = PS.steering_axes("movie")
        subs.append(a["subject"])
        sets_.append(a["setting"])
    for i in range(1, len(subs)):
        assert subs[i] != subs[i - 1], f"subject repeated at draw {i}"
        assert sets_[i] != sets_[i - 1], f"setting repeated at draw {i}"


def test_the_clause_reads_as_instructions_not_a_template():
    """The LLM must WRITE the prompt; these are constraints handed to it. If the
    clause ever became the prompt itself, output would read as slot-filled."""
    clause = PS.steering_clause("movie", rng=random.Random(4))
    assert "flowing prose, not a list" in clause
    assert clause.count("\n- ") == 7          # 6 still axes + motion


def test_the_banks_are_deep_enough_to_matter():
    """Guards against someone thinning the banks until Generate repeats again."""
    assert PS.combinations("image") > 1_000_000
    assert PS.combinations("movie") > PS.combinations("image")
    # A draft drops one axis, so the space shrinks but stays large.
    assert PS.combinations("movie", has_draft=True) > 1_000_000


def test_injected_rng_is_deterministic():
    """Reproducible for tests without freezing production behaviour."""
    a = PS.steering_axes("image", rng=random.Random(7))
    b = PS.steering_axes("image", rng=random.Random(7))
    # subject/setting carry anti-repeat state, so compare the pure axes.
    for axis in ("light", "mood", "palette", "shot"):
        assert a[axis] == b[axis], axis


# ── the synthetic gate must refuse NOISE, not "last-resort" ─────────────────
def test_the_synthetic_gate_refuses_provers_not_ffmpeg_transforms():
    """REGRESSION: the 2026-07-27 opt-in gate killed UPRES and INTERPOLATE.

    `models_seed` uses `synthetic=True` for TWO different things:
      * Framework.SYNTHETIC — output is noise from seed + geometry (the provers).
      * Framework.FFMPEG / CODEFORMER — REAL transforms of REAL pixels, flagged
        only so the premium RIFE/LTX rows outrank them ("LAST-RESORT").

    Gating on the flag refused the second group too. Their premium runners return
    Err unconditionally (weights/deps missing), so the ffmpeg row IS the working
    path — and blocking it took two live product surfaces to zero.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hugpy_video.intel.studio.registry import MODEL_REGISTRY

    def refused(model_id):
        cfg = MODEL_REGISTRY.get(model_id)
        assert cfg is not None, model_id
        return getattr(cfg.family, "value", cfg.family) == "synthetic"

    # noise -> refuse
    assert refused("synthetic-i2v")
    assert refused("synthetic-t2v")
    # real transforms flagged last-resort -> MUST NOT refuse
    assert not refused("ffmpeg-minterpolate")
    assert not refused("ffmpeg-lanczos-upscale")
    # and a real model obviously not
    assert not refused("wan2.1-t2v-1.3b")


def test_the_two_meanings_of_synthetic_still_diverge():
    """Proves the flag really is ambiguous, so the gate can never go back to it.
    If someone 'tidies' models_seed so synthetic==Framework.SYNTHETIC, this fails
    loudly and the gate can be simplified deliberately rather than by accident."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hugpy_video.intel.studio.registry import MODEL_REGISTRY
    ffmpeg = MODEL_REGISTRY.get("ffmpeg-minterpolate")
    assert ffmpeg.synthetic is True, "the ranking flag"
    assert getattr(ffmpeg.family, "value", ffmpeg.family) != "synthetic", "but not noise"
