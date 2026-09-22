"""Units B + C of METHOD-vigilant-inference.md on the live recipe:
the prompt compiler decides candidates/angles per segment inside
``run_performance``, and judge verdicts are written to the reliability ledger
against the model that produced the artifact.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_performance_vigilant.py -q
"""
from __future__ import annotations

import logging
import os
import sys

logging.disable(logging.INFO)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
for p in (_SRC, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402

from hugpy_oracle import performance as perf
from hugpy_oracle import selection
from hugpy_oracle.contracts import QualityProfile
from hugpy_oracle.prompt_compiler import compile_context
from dataclasses import replace  # noqa: E402

import test_oracle_performance as base  # noqa: E402  (shared fakes + goal builder)


class SpyFakes(base.Fakes):
    """Records every keyframe prompt and attributes each produced ref to a
    fake model, the way ``_live_gen_image`` does via ``remember_producer``."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.prompts: list[str] = []
        self.model_for_seed = kw.pop("model_for_seed", None) or (lambda seed: "flux" if seed % 2 == 0 else "sdxl")

    def gen_image(self, prompt, identity_refs, seed):
        self.prompts.append(prompt)
        ref = super().gen_image(prompt, identity_refs, seed)
        selection.remember_producer(ref, perf.KEYFRAME_CAPABILITY, self.model_for_seed(seed))
        return ref


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    """Bind the process selector to a fresh ledger for this test only."""
    path = os.path.join(str(tmp_path), "ledger.sqlite")
    led = selection.ReliabilityLedger(path)
    sel = selection.Selector(ledger=led, get_view=lambda c: None, get_matrix=lambda: None)
    monkeypatch.setattr(selection, "_PROCESS_SELECTOR", sel)
    yield led
    led.close()
    monkeypatch.setattr(selection, "_PROCESS_SELECTOR", None)


def test_segment_context_adapts_a_locked_spec(tmp_path):
    fakes = base.Fakes()
    result = perf.run_performance(base.performance_goal(), seams=fakes.seams(tmp_path))
    assert result.ok
    spec = result.segments[0]
    ctx = perf.segment_context(spec, base.performance_goal())
    assert ctx["segment_id"] == spec.segment_id
    assert ctx["scene"] == spec.prompt
    assert ctx["identity_constraints"]  # identity refs ride as invariant constraints
    assert ctx["duration_s"] == pytest.approx(spec.duration_s)
    plan = compile_context(ctx, goal=base.goal_spec())
    assert plan.segment_id == spec.segment_id and plan.candidates >= 1


def test_first_candidate_is_the_locked_prompt_and_later_ones_carry_angles(tmp_path, ledger):
    # verdict list is consumed ACROSS segments: segment 1 gets NO, NO, YES;
    # every later segment passes on its first candidate.
    fakes = SpyFakes(keyframe_verdicts=[
        {"verdict": "NO", "score": 20, "why": "wrong"},
        {"verdict": "NO", "score": 25, "why": "wrong"},
        {"verdict": "YES", "score": 90, "why": "ok"},
    ])
    result = perf.run_performance(base.performance_goal(), seams=fakes.seams(tmp_path))
    assert result.ok, result.gap
    first = result.segments[0]
    assert fakes.calls["gen_image"] == 3 + (len(result.segments) - 1)
    taken = fakes.prompts[:3]
    assert taken[0] == first.prompt, "candidate 0 must be the locked prompt verbatim"
    assert taken[1].startswith(first.prompt) and "PRIORITY]" in taken[1]
    assert taken[2].startswith(first.prompt) and "PRIORITY]" in taken[2]
    for spec, prompt in zip(result.segments[1:], fakes.prompts[3:]):
        assert prompt == spec.prompt  # siblings: their own locked prompt, untouched


def test_judge_verdicts_are_ledgered_against_the_producing_model(tmp_path, ledger):
    fakes = SpyFakes(keyframe_verdicts=[
        {"verdict": "NO", "score": 20, "why": "identity off"},
        {"verdict": "YES", "score": 90, "why": "ok"},
    ])
    result = perf.run_performance(base.performance_goal(), seams=fakes.seams(tmp_path))
    assert result.ok, result.gap
    rows = ledger.recent(perf.KEYFRAME_CAPABILITY, limit=100)
    assert len(rows) == fakes.calls["gen_image"], "one verdict per produced keyframe"
    assert {r["model_id"] for r in rows} <= {"flux", "sdxl"}
    passed = [r for r in rows if r["hard_pass"] == 1]
    failed = [r for r in rows if r["hard_pass"] == 0]
    assert len(failed) == 1 and len(passed) == len(result.segments)
    # the failed one is attributed to the model that produced that seed
    assert failed[0]["model_id"] in ("flux", "sdxl")
    assert sum(ledger.stats(perf.KEYFRAME_CAPABILITY, m).n for m in ("flux", "sdxl")) == len(rows)


def test_difficulty_can_raise_candidates_above_the_seam_floor(tmp_path, ledger):
    """BEST profile + action-heavy blocking: the compiler's multiplicity
    exceeds keyframe_candidates=1, so two rejections no longer sink the shot."""
    fakes = SpyFakes(keyframe_verdicts=[
        {"verdict": "NO", "score": 10, "why": "no"},
        {"verdict": "NO", "score": 10, "why": "no"},
        {"verdict": "YES", "score": 95, "why": "ok"},
    ])
    goal = base.performance_goal(
        goal=replace(base.goal_spec(), quality=QualityProfile.BEST),
        blocking="Ana hurls the lantern; Ben sprints and catches it as he passes behind the cart",
        camera={"movement": "handheld"},
    )
    result = perf.run_performance(goal, seams=fakes.seams(tmp_path, keyframe_candidates=1))
    assert result.ok, result.gap
    spec = result.segments[0]
    plan = compile_context(perf.segment_context(spec, goal), goal=goal.goal, max_candidates=5)
    assert plan.candidates >= 3, plan.reasons
    assert {v.angle for v in plan.variants} >= {"identity", "physics", "camera"}
    # floor was 1: the old code would have tried exactly one keyframe for segment 1 and failed
    assert fakes.calls["gen_image"] == 3 + (len(result.segments) - 1)


def test_compiler_is_off_the_critical_path(tmp_path, monkeypatch):
    """A compiler fault is a limitation note, never a failed build."""
    import hugpy_oracle.prompt_compiler as pc

    def boom(*a, **k):
        raise RuntimeError("compiler down")

    monkeypatch.setattr(pc, "compile_context", boom)
    fakes = base.Fakes()
    result = perf.run_performance(base.performance_goal(), seams=fakes.seams(tmp_path))
    assert result.ok
    assert any("prompt compiler unavailable" in l for l in result.limitations)


def test_tts_candidates_are_ledgered_against_the_producing_voice_model(tmp_path, ledger):
    class TtsSpy(base.Fakes):
        def synth(self, line, voice, seed):
            out = super().synth(line, voice, seed)
            ref = out[0] if isinstance(out, tuple) else out
            selection.remember_producer(ref, perf.TTS_CAPABILITY, "chatterbox-fake")
            return out

    fakes = TtsSpy()
    result = perf.run_performance(base.performance_goal(), seams=fakes.seams(tmp_path))
    assert result.ok, result.gap
    rows = ledger.recent(perf.TTS_CAPABILITY, limit=200)
    assert len(rows) == fakes.calls["synth"], "one verdict per TTS candidate"
    assert {r["model_id"] for r in rows} == {"chatterbox-fake"}
    assert all(r["hard_pass"] is not None for r in rows)
    assert ledger.stats(perf.TTS_CAPABILITY, "chatterbox-fake").n == len(rows)
