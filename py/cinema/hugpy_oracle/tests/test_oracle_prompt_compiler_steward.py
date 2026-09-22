"""k113b/k113c — prompt compiler (context / length / multiplicity per prompt)
and steward (self-check, self-balance).

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_prompt_compiler_steward.py -q
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

logging.disable(logging.INFO)
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from hugpy_oracle.contracts import BudgetHints, Eligibility, GoalSpec, QualityProfile, RepairCode
from hugpy_oracle.prompt_compiler import (
    compile_context,
    difficulty_score,
    extract_signals,
    render_prompt,
)
from hugpy_oracle.selection import ReliabilityLedger, SelectionPolicy, Selector, select
from hugpy_oracle.steward import Steward, StewardPolicy, rank_agreement


# --------------------------------------------------------------------------- #
# prompt compiler
# --------------------------------------------------------------------------- #

STATIC = {
    "segment_id": "seg_01", "characters": ["Alex"],
    "scene": "INT. KITCHEN - DAY. Alex stands at the counter, reading a letter.",
    "state_before": "Alex: calm, blue shirt, letter unopened on counter.",
    "state_after": "Alex: letter open in left hand, expression shifts to worry.",
    "camera": {"shot": "medium close-up", "lens": "50mm", "movement": "static"},
    "identity_constraints": "Alex: ref_alex_front, ref_alex_profile; short dark hair, scar on left brow.",
    "negative_constraints": "no extra fingers, no text overlays, no second person",
    "duration_s": 4.0, "tone": 2.0,
}

ACTION = {
    "segment_id": "seg_07", "characters": ["Alex", "Sam"],
    "scene": "EXT. ALLEY - NIGHT. Sam throws the bottle; Alex catches it as he runs past the dumpster.",
    "blocking": "Sam hurls the bottle in an arc; Alex sprints left-to-right, catches it chest height, "
                "passes behind the dumpster and re-emerges.",
    "props": ["bottle", "dumpster"],
    "camera": {"movement": "handheld tracking, whip pan to Sam", "lens": "35mm"},
    "state_before": "Bottle in Sam's right hand. Alex 6m camera-left.",
    "state_after": "Bottle in Alex's hands. Alex camera-right of dumpster. Sam's arm still extended.",
    "identity_constraints": "Alex: ref_alex_*; Sam: ref_sam_*; Sam wears a long coat.",
    "negative_constraints": "no bottle duplication, no limb intersection with dumpster",
    "dialogue": "Sam: Catch! Alex: Got it — now run.",
    "duration_s": 6.0, "tone": 1.0,
}


def test_static_shot_is_easy_and_gets_one_prompt():
    sig = extract_signals(STATIC)
    assert sig.camera_motion == "static" and sig.props_with_momentum == 0 and sig.characters == 1
    d, why = difficulty_score(sig)
    assert d < 0.2
    plan = compile_context(STATIC, model_context_tokens=4096, eligible_models=3)
    assert plan.candidates == 1 and plan.variants[0].angle == "identity"


def test_action_shot_is_hard_and_multiplies_across_angles_and_models():
    sig = extract_signals(ACTION)
    assert sig.characters == 2 and sig.props_with_momentum >= 1
    assert sig.camera_motion == "complex" and sig.occlusion and sig.moving_characters >= 1
    d, why = difficulty_score(sig)
    assert d >= 0.6
    assert any("NO spatial manifest" in w for w in why)
    plan = compile_context(ACTION, goal=GoalSpec(objective="o", raw_prompt="p", quality=QualityProfile.BEST),
                           model_context_tokens=4096, eligible_models=3)
    assert plan.candidates >= 3
    angles = [v.angle for v in plan.variants]
    assert "physics" in angles and "camera" in angles and angles[0] == "identity"
    assert plan.variants[1].spread_model is True and plan.variants[0].spread_model is False
    assert any("no spatial manifest" in r for r in plan.reasons)


def test_spatial_manifest_removes_the_unconstrained_penalty():
    seg = dict(ACTION, spatial_manifest={"timebase": {"fps": 24}, "camera": {"track_uri": "x"},
                                         "entities": [{"entity_id": "alex"}], "tier_profile": {"capture": 1}})
    d_with, why = difficulty_score(extract_signals(seg))
    d_without, _ = difficulty_score(extract_signals(ACTION))
    assert d_with < d_without
    plan = compile_context(seg, model_context_tokens=4096)
    assert any(s.name == "spatial" and s.content for s in plan.sections)


def test_length_scales_with_difficulty_and_profile():
    easy = compile_context(STATIC, model_context_tokens=4096)
    hard = compile_context(ACTION, model_context_tokens=4096)
    assert hard.target_tokens > easy.target_tokens
    preview = compile_context(ACTION, goal=GoalSpec(objective="o", raw_prompt="p", quality=QualityProfile.PREVIEW),
                              model_context_tokens=4096)
    assert preview.target_tokens < hard.target_tokens and preview.candidates == 1
    assert hard.target_tokens <= int(4096 * 0.35)


def test_budget_caps_multiplicity():
    g = GoalSpec(objective="o", raw_prompt="p", quality=QualityProfile.BEST,
                 budget=BudgetHints(max_seconds=40))
    plan = compile_context(ACTION, goal=g, model_context_tokens=4096, eligible_models=3)
    assert plan.candidates == 1


def test_invariant_sections_are_never_truncated_and_render_carries_angle():
    seg = dict(ACTION, scene=ACTION["scene"] * 60, production_design="x " * 2000)
    plan = compile_context(seg, model_context_tokens=1024)
    inv = {s.name for s in plan.sections if s.priority == 0}
    assert {"identity_constraints", "negative_constraints", "state_after"} <= inv
    text = render_prompt(plan, plan.variants[0])
    assert text.startswith("[IDENTITY PRIORITY]")
    assert ACTION["negative_constraints"] in text
    assert ACTION["state_after"] in text
    assert "…" in text  # something lower-priority was truncated
    assert len(text) // 4 <= plan.target_tokens * 1.5


def test_missing_required_section_is_reported_not_invented():
    seg = dict(STATIC)
    del seg["state_after"]
    plan = compile_context(seg, model_context_tokens=2048)
    assert any("state_after" in r and "EMPTY" in r for r in plan.reasons)


def test_plan_serializes():
    import json
    plan = compile_context(ACTION, model_context_tokens=2048)
    json.dumps(plan.to_dict())


# --------------------------------------------------------------------------- #
# steward
# --------------------------------------------------------------------------- #


@dataclass
class FakeView:
    name: str
    model_ids: tuple
    eligibility: Eligibility = Eligibility(eligible=True)


@pytest.fixture()
def ledger(tmp_path):
    l = ReliabilityLedger(os.path.join(str(tmp_path), "ledger.sqlite"))
    yield l
    l.close()


def test_rank_agreement_basics():
    assert rank_agreement([(0.1, 0), (0.5, 1), (0.9, 1), (0.3, 0)]) > 0.8
    assert rank_agreement([(0.9, 0), (0.1, 1), (0.5, 0), (0.2, 1)]) < -0.5
    assert rank_agreement([(0.5, 1), (0.5, 1), (0.5, 1)]) is None


def test_clean_ledger_reports_ok_not_silence(ledger):
    rep = Steward(ledger).check()
    kinds = [f.kind for f in rep.findings]
    assert "matrix_stale" in kinds  # no matrix loaded is said out loud
    assert rep.summary


def test_streak_alarm_with_dominant_code(ledger):
    for _ in range(5):
        ledger.record("image.generate", "flux", ok=True, hard_pass=False, repair_code=RepairCode.IDENTITY_DRIFT)
    rep = Steward(ledger).check()
    streak = [f for f in rep.findings if f.kind == "streak"]
    assert streak and streak[0].severity == "alarm" and "identity_drift" in streak[0].message
    assert not rep.ok


def test_starvation_raises_exploration(ledger):
    for _ in range(45):
        ledger.record("image.generate", "flux", ok=True, hard_pass=True, score=0.8)
    sel = Selector(ledger=ledger, get_view=lambda c: FakeView(c, ("flux", "sdxl")), get_matrix=lambda: None,
                   policy=SelectionPolicy(explore_every=12))
    st = Steward(ledger, selector=sel, eligible_models={"image.generate": ("flux", "sdxl")})
    rep = st.check()
    assert any(f.kind == "starvation" and "sdxl" in f.message for f in rep.findings)
    assert sel.policy.explore_every < 12
    assert rep.to_dict()["policy_changed"] is True


def test_uncalibrated_scores_shift_weight_to_ledger(ledger):
    # high predicted scores fail, low predicted scores pass -> negative agreement
    for i in range(20):
        hi = i % 2 == 0
        ledger.record("audio.tts", "cb", ok=True, hard_pass=not hi, score=0.9 if hi else 0.2)
    sel = Selector(ledger=ledger, get_view=lambda c: FakeView(c, ("cb",)), get_matrix=lambda: None)
    w0 = sel.policy.w_matrix_quality
    rep = Steward(ledger, selector=sel).check()
    cal = [f for f in rep.findings if f.kind == "calibration"]
    assert cal and cal[0].severity == "warn"
    assert sel.policy.w_matrix_quality < w0
    assert sel.policy.w_ledger_pass > SelectionPolicy().w_ledger_pass
    # bounded: a second check moves at most one more step
    Steward(ledger, selector=sel).check()
    assert sel.policy.w_matrix_quality >= w0 - 2 * StewardPolicy().weight_step - 1e-9


def test_selector_explores_runner_up_periodically(ledger):
    view = FakeView("image.generate", ("flux", "sdxl"))
    pol = SelectionPolicy(explore_every=5, explore_margin=1.0, spread_candidates=False)
    # seed evidence so both are near; then count who gets picked across calls
    for _ in range(4):
        ledger.record("image.generate", "flux", ok=True, hard_pass=True)
    ledger.record("image.generate", "sdxl", ok=True, hard_pass=True)  # total 5 -> explore tick
    d = select("image.generate", view=view, ledger=ledger, policy=pol)
    # sdxl was tried 0 calls ago (< explore_every) so exploration must NOT fire on a fresh runner-up
    assert d.explored is False
    for _ in range(5):
        ledger.record("image.generate", "flux", ok=True, hard_pass=True)
    d2 = select("image.generate", view=view, ledger=ledger, policy=pol)  # total 10, sdxl stale 5
    assert d2.explored is True and d2.model_id == "sdxl"
    assert "explore" in d2.rationale


# --------------------------------------------------------------------------- #
# k115: calibration weighted by judge-panel confidence
# --------------------------------------------------------------------------- #


def test_rank_agreement_weights_discount_low_confidence_rows():
    # predictive rows at full confidence, contradicting rows at near-zero confidence
    good = [(0.9, 1), (0.8, 1), (0.7, 1), (0.2, 0), (0.1, 0), (0.3, 0)]
    bad = [(0.95, 0), (0.05, 1), (0.9, 0), (0.1, 1)]
    pairs = good + bad
    unweighted = rank_agreement(pairs)
    weighted = rank_agreement(pairs, [1.0] * len(good) + [0.05] * len(bad))
    assert weighted > unweighted
    assert rank_agreement(pairs, [1.0] * len(pairs)) == pytest.approx(unweighted)
    # zero-weight rows are dropped entirely
    assert rank_agreement(pairs, [1.0] * len(good) + [0.0] * len(bad)) == pytest.approx(
        rank_agreement(good))
    with pytest.raises(ValueError):
        rank_agreement(pairs, [1.0])


def test_steward_calibration_uses_row_confidence(ledger, monkeypatch):
    # 20 judged rows: 14 predictive, 6 contradicting; the contradicting ones came
    # from a split panel (confidence 0.5) once the ledger can say so.
    rows = []
    for i in range(20):
        contradict = i < 6
        hi = i % 2 == 0
        hard_pass = (not hi) if contradict else hi
        ledger.record("audio.tts", "cb", ok=True, hard_pass=hard_pass, score=0.9 if hi else 0.2)
        rows.append(contradict)
    plain = Steward(ledger)._calibration("audio.tts", ledger.recent("audio.tts"))[0]

    real_recent = ledger.recent

    def with_conf(cap=None, *, limit=500):
        out = real_recent(cap, limit=limit)
        for r in out:            # newest first: seq 1..6 are the contradicting ones
            r["confidence"] = 0.1 if r["seq"] <= 6 else 1.0
        return out
    monkeypatch.setattr(ledger, "recent", with_conf)
    st = Steward(ledger)
    rho, finding = st._calibration("audio.tts", ledger.recent("audio.tts"))
    assert rho > plain
    assert finding.evidence["n"] == 20 and finding.evidence["effective_n"] == pytest.approx(14.6)
    assert "confidence-weighted n 14.6" in finding.message
