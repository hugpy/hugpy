"""k109 — the script-first model evaluation: case suites, the two scoring
layers, the judge's self-refusal, the routing matrix and the run dir.

Every test is OFFLINE: the dispatch, the route, the judge and the fleet
telemetry are all module-level seams that are monkeypatched here, so no worker,
no GPU and no registry are touched.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_benchmark.py -q
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

logging.disable(logging.INFO)  # silence the models_config registry chatter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from hugpy_oracle import benchmark, benchmark_cases, routing_matrix
from hugpy_oracle.benchmark_cases import FIXTURE_SCREENPLAY, OPERATIONS, case as get_case
from hugpy_oracle.screenplay import (
    AuthoringGap,
    PlotSpec,
    Screenplay,
    build_continuity,
    build_shot_plan,
    chain_breaks,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRoute:
    def __init__(self, execution="execute", model_ids=("m1", "m2"),
                 model_id="m1", task="text-generation", reasons=()):
        self.execution = execution
        self.model_ids = tuple(model_ids)
        self.model_id = model_id
        self.task = task
        self.reasons = tuple(reasons)
        self.capability = "text.chat"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No fleet, ever. Each test overrides what it actually cares about."""
    monkeypatch.setattr(benchmark, "_run_bounded",
                        lambda fn, deadline_s, label: fn())
    monkeypatch.setattr(benchmark, "_vram_snapshot", lambda: None)
    monkeypatch.setattr(benchmark, "_nvml_reader", lambda: None)  # or-k10: no GPU in tests
    monkeypatch.setattr(benchmark, "_selected_worker", lambda m, t: "worker-a")
    monkeypatch.setattr(benchmark, "_load_state", lambda m, w: "loaded")
    monkeypatch.setattr(benchmark, "_registry_version", lambda: "rv-test")
    monkeypatch.setattr(benchmark, "_capability_view", lambda *a, **k: None)
    monkeypatch.setattr(benchmark, "_resolve_route",
                        lambda *a, **k: FakeRoute())
    monkeypatch.setattr(benchmark, "_no_think", lambda p: p)
    monkeypatch.setattr(benchmark, "_strip_think", lambda t: t)


def replies(*texts, usage=None):
    """A fake dispatch that answers with ``texts`` in order (last one repeats),
    recording every body it was handed."""
    seen: list[dict] = []
    queue = list(texts)

    def _dispatch(task, body):
        seen.append(dict(body, _task=task))
        text = queue.pop(0) if len(queue) > 1 else (queue[0] if queue else "")
        payload = {"text": text, "ok": True, "model_key": body.get("model_key"),
                   "finish_reason": "stop"}
        if usage:
            payload["usage"] = dict(usage)
        return payload

    _dispatch.seen = seen  # type: ignore[attr-defined]
    return _dispatch


# ---------------------------------------------------------------------------
# Canned outputs
# ---------------------------------------------------------------------------

GOOD_A1 = {
    "title": "THE LAST BUS",
    "logline": "A woman carries a cake across a night town.",
    "characters": ["RUTH", "DRIVER"],
    "scenes": [
        {"scene_id": "s1", "heading": "INT. NIGHT BUS - NIGHT",
         "location": "NIGHT BUS", "time_of_day": "NIGHT",
         "action": "RUTH sits with a cake box on her knees.",
         "present_at_open": ["RUTH", "DRIVER"], "props": ["cake box"],
         "dialogue": [{"line_id": "l1", "speaker": "RUTH",
                       "text": "Just to the depot. I know the way back."}],
         "transition": "CUT TO:", "story_time_s": 0.0},
        {"scene_id": "s2", "heading": "EXT. DEPOT FORECOURT - NIGHT",
         "location": "DEPOT FORECOURT", "time_of_day": "NIGHT",
         "action": "The bus pulls out without her. RUTH holds the box level.",
         "present_at_open": ["RUTH"], "props": ["cake box"],
         "dialogue": [{"line_id": "l2", "speaker": "RUTH",
                       "text": "It only has to last until morning."}],
         "transition": "FADE OUT.", "story_time_s": 600.0},
    ],
}

# Same shape, but the supplied dialogue was REWORDED and the second scene is
# gone — the two failures preservation and completeness exist to catch.
BAD_A1 = {
    "title": "THE LAST BUS",
    "characters": ["RUTH"],
    "scenes": [
        {"scene_id": "s1", "heading": "INT. A BUS - NIGHT",
         "location": "A BUS", "time_of_day": "NIGHT", "action": "",
         "present_at_open": ["RUTH"],
         "dialogue": [{"line_id": "l1", "speaker": "RUTH",
                       "text": "Take me to the depot please."}],
         "transition": "CUT TO:", "story_time_s": 0.0},
    ],
}

GOOD_B = {
    "premise": "A ferry rider crosses all day to avoid a decision.",
    "genre": "drama", "tone": "quiet", "ending": "She stays ashore.",
    "pacing": "slow", "notes": "",
    "characters": [
        {"name": "IRIS", "goal": "delay the decision",
         "conflict": "the last crossing ends the day", "arc": "chooses"},
        {"name": "the ticket inspector", "goal": "keep the peace",
         "conflict": "the rules say otherwise", "arc": "speaks at last"},
    ],
    "beats": [
        {"beat_id": "b1", "summary": "IRIS boards the ferry off season.",
         "characters": ["IRIS"], "causes": [], "turning_point": False},
        {"beat_id": "b2", "summary": "the ticket inspector says nothing.",
         "characters": ["the ticket inspector"], "causes": ["b1"],
         "turning_point": False},
        {"beat_id": "b3", "summary": "IRIS is recognized by a passenger.",
         "characters": ["IRIS"], "causes": ["b2"], "turning_point": True},
        {"beat_id": "b4", "summary": "the last crossing empties.",
         "characters": ["IRIS", "the ticket inspector"], "causes": ["b3"],
         "turning_point": False},
    ],
}

# Dangling cause + orphan character: two of PlotSpec's three refusals.
BAD_B = {
    "premise": "x", "genre": "drama", "tone": "quiet", "ending": "x",
    "characters": [{"name": "IRIS", "goal": "g", "conflict": "c", "arc": "a"},
                   {"name": "GHOST", "goal": "g", "conflict": "c", "arc": "a"}],
    "beats": [{"beat_id": "b1", "summary": "s", "characters": ["IRIS"],
               "causes": ["b9"]}],
}


def _breakdown(scene_ids=("s1", "s2", "s3")):
    return {"scenes": [
        {"scene_id": sid, "interior_exterior": "INT", "location": "LIGHTHOUSE",
         "time_of_day": "NIGHT", "cast": ["MARA"], "props": ["logbook"],
         "wardrobe": ["oilskin coat"], "sound": ["surf"]}
        for sid in scene_ids]}


def _shot_ids():
    return build_shot_plan(FIXTURE_SCREENPLAY).segment_ids


# ---------------------------------------------------------------------------
# 1. The case suites
# ---------------------------------------------------------------------------


def test_every_track_has_at_least_six_cases():
    for track in ("A", "B", "C"):
        assert len(benchmark_cases.SUITES[track]) >= 6, track


def test_track_b_covers_the_docs_six_input_conditions():
    conditions = " ".join(c.condition for c in benchmark_cases.TRACK_B).lower()
    for phrase in ("detailed partial premise", "sparse notes",
                   "characters without a plot", "setting without characters",
                   "minimal input", "no meaningful prior narrative input"):
        assert phrase in conditions, phrase


def test_track_c_covers_six_distinct_operations():
    ops = {c.operation for c in benchmark_cases.TRACK_C}
    assert len(ops) >= 6
    assert {"breakdown.script", "continuity.extract", "shotlist.build",
            "storyboard.prompts", "segment.prompts",
            "assembly.plan"} <= ops


def test_every_case_declares_a_known_operation_and_a_checklist():
    for case in benchmark_cases.ALL_CASES:
        assert case.operation in OPERATIONS, case.case_id
        assert case.expectations, case.case_id
        assert any(e.layer == "deterministic" for e in case.expectations)


def test_fixture_screenplay_is_valid_and_its_continuity_chain_is_sound():
    assert isinstance(FIXTURE_SCREENPLAY, Screenplay)
    assert chain_breaks(build_continuity(FIXTURE_SCREENPLAY)) == ()


def test_cases_for_selects_tracks_and_limits():
    picked = benchmark_cases.cases_for("AB", limit_per_track=2)
    assert [c.case_id for c in picked] == ["A1-partial", "A2-disconnected",
                                           "B1-detailed-premise",
                                           "B2-sparse-notes"]


# ---------------------------------------------------------------------------
# 2. Deterministic scoring — Track A
# ---------------------------------------------------------------------------


def test_track_a_good_output_scores_valid_and_fully_preserved():
    case = get_case("A1-partial")
    raw = json.dumps(GOOD_A1)
    score = benchmark.score_case(case, Screenplay.from_dict(GOOD_A1), raw)
    assert score.valid is True
    assert score.preservation == 1.0
    assert score.contradiction_rate == 0.0
    assert score.completeness == 1.0
    assert score.score > 90


def test_track_a_reworded_line_is_not_preserved():
    case = get_case("A1-partial")
    raw = json.dumps(BAD_A1)
    score = benchmark.score_case(case, Screenplay.from_dict(BAD_A1), raw)
    assert score.preservation is not None and score.preservation < 0.5
    assert score.completeness < 1.0
    assert score.score < 70


def test_track_a_authoring_gap_scores_invalid_with_its_errors():
    case = get_case("A1-partial")
    gap = AuthoringGap(errors=("scene 's2' is at story time 1.0s, BEFORE 's1'",),
                       raw="not json", stage="screenplay",
                       code="AUTHORING_INVALID", attempts=2)
    score = benchmark.score_case(case, gap, gap.raw)
    assert score.valid is False
    assert score.error_count == 1
    assert score.contradiction_rate > 0.0     # the error IS a contradiction


def test_track_a_constraints_are_checked_and_counted():
    case = get_case("A4-constrained")
    obeying = json.loads(json.dumps(GOOD_A1))
    obeying["scenes"][0]["heading"] = "INT. SERVER ROOM - NIGHT"
    obeying["scenes"][0]["location"] = "SERVER ROOM"
    obeying["scenes"][0]["dialogue"][0]["text"] = \
        "That is not a fan. That is somebody in the building."
    score = benchmark.score_case(case, Screenplay.from_dict(obeying),
                                 json.dumps(obeying))
    assert score.constraint_adherence == 1.0

    breaking = json.loads(json.dumps(obeying))
    breaking["scenes"][1]["action"] += " She finds a gun in the rack."
    breaking["scenes"][1]["transition"] = "CUT TO:"
    score2 = benchmark.score_case(case, Screenplay.from_dict(breaking),
                                  json.dumps(breaking))
    assert score2.constraint_adherence is not None
    assert score2.constraint_adherence < 1.0
    keys = {c.key: c.passed for c in score2.checks}
    assert keys["constraint:forbidden_term=gun"] is False
    assert keys["constraint:requires_transition=FADE OUT."] is False


def test_track_a_contradiction_pairs_fire_only_inside_one_scene():
    case = get_case("A6-contradiction")
    both = json.loads(json.dumps(GOOD_A1))
    both["scenes"][0]["action"] = ("The radio is smashed on the floor. NELL "
                                   "raises the radio and calls the coastguard.")
    score = benchmark.score_case(case, Screenplay.from_dict(both),
                                 json.dumps(both))
    assert score.contradiction_rate > 0.0
    resolved = json.loads(json.dumps(GOOD_A1))
    resolved["scenes"][0]["action"] = "The radio is smashed on the floor."
    score2 = benchmark.score_case(case, Screenplay.from_dict(resolved),
                                  json.dumps(resolved))
    assert score2.contradiction_rate == 0.0


# ---------------------------------------------------------------------------
# 3. Deterministic scoring — Track B
# ---------------------------------------------------------------------------


def test_track_b_good_plot_validates_and_preserves_the_notes():
    case = get_case("B2-sparse-notes")
    score = benchmark.score_case(case, PlotSpec.from_dict(GOOD_B),
                                 json.dumps(GOOD_B))
    assert score.valid is True
    assert score.preservation == 1.0
    assert score.completeness == 1.0
    assert score.score > 90


def test_track_b_dangling_cause_is_a_causal_failure():
    case = get_case("B2-sparse-notes")
    with pytest.raises(Exception) as excinfo:
        PlotSpec.from_dict(BAD_B)
    gap = AuthoringGap(errors=tuple(excinfo.value.errors), raw=json.dumps(BAD_B),
                       stage="plot", code="AUTHORING_INVALID", attempts=2)
    score = benchmark.score_case(case, gap, gap.raw)
    assert score.valid is False
    assert score.error_count >= 2
    assert score.contradiction_rate > 0.0
    assert {c.key for c in score.checks} >= {"validates", "causal_logic",
                                             "complete_artifact"}


# ---------------------------------------------------------------------------
# 4. Deterministic scoring — Track C
# ---------------------------------------------------------------------------


def test_track_c_breakdown_covering_every_scene_is_valid():
    case = get_case("C1-breakdown")
    obj = _breakdown()
    score = benchmark.score_case(case, obj, json.dumps(obj))
    assert score.valid is True
    assert score.preservation == 1.0
    assert score.completeness == 1.0
    assert score.contradiction_rate == 0.0


def test_track_c_missing_scene_lowers_coverage_and_invalidates():
    case = get_case("C1-breakdown")
    obj = _breakdown(("s1", "s2"))
    score = benchmark.score_case(case, obj, json.dumps(obj))
    assert score.valid is False
    assert score.preservation == pytest.approx(2 / 3)
    assert any("no scenes row covers" in e for e in score.errors)


def test_track_c_hallucinated_id_is_a_contradiction():
    case = get_case("C1-breakdown")
    obj = _breakdown(("s1", "s2", "s3", "s9"))
    score = benchmark.score_case(case, obj, json.dumps(obj))
    assert score.contradiction_rate > 0.0
    assert any("s9" in e for e in score.errors)


def test_track_c_continuity_accuracy_is_measured_against_the_derived_bible():
    case = get_case("C2-continuity")
    bible = build_continuity(FIXTURE_SCREENPLAY)
    right = {"segments": [
        {"segment_id": e.segment_id,
         "state_before": {"location": e.state_before["location"],
                          "time_of_day": e.state_before["time_of_day"],
                          "present": list(e.state_before["present"])},
         "state_after": {"location": e.state_after["location"],
                         "time_of_day": e.state_after["time_of_day"],
                         "present": list(e.state_after["present"])}}
        for e in bible.entries]}
    assert benchmark.score_case(case, right, json.dumps(right)).accuracy == 1.0

    wrong = json.loads(json.dumps(right))
    wrong["segments"][1]["state_before"]["present"] = ["NOBODY"]
    score = benchmark.score_case(case, wrong, json.dumps(wrong))
    assert score.accuracy is not None and score.accuracy < 1.0


def test_track_c_assembly_plan_partition_is_measured():
    case = get_case("C6-assembly")
    ids = _shot_ids()
    gapless = {"timeline": [
        {"segment_id": sid, "start_s": i * 4.0, "end_s": (i + 1) * 4.0,
         "transition": "CUT TO:"} for i, sid in enumerate(ids)]}
    assert benchmark.score_case(case, gapless,
                                json.dumps(gapless)).accuracy == 1.0
    gappy = json.loads(json.dumps(gapless))
    gappy["timeline"][1]["start_s"] += 9.0
    gappy["timeline"][1]["end_s"] += 9.0
    score = benchmark.score_case(case, gappy, json.dumps(gappy))
    assert score.accuracy is not None and score.accuracy < 1.0


def test_track_c_segment_prompt_leak_is_a_contradiction():
    case = get_case("C5-segment-prompts")
    ids = _shot_ids()
    leaky = {"segments": [
        {"segment_id": sid, "duration_s": 4.0,
         "prompt": ("MARA at the desk in the LIGHTHOUSE KEEPER'S ROOM, then "
                    "the CLIFF PATH as before")}
        for sid in ids]}
    score = benchmark.score_case(case, leaky, json.dumps(leaky))
    assert score.contradiction_rate > 0.0
    keys = {c.key: c.passed for c in score.checks}
    assert keys["no_cross_segment_leak"] is False
    assert keys["constraint:forbidden_term=as before"] is False


def test_workflow_errors_reports_every_problem_at_once():
    spec = OPERATIONS["breakdown.script"]
    problems = benchmark.workflow_errors(
        spec, {"scenes": [{"scene_id": "s1"}]}, ("s1", "s2"))
    assert len(problems) > 3
    assert any("missing 'cast'" in p for p in problems)
    assert any("s2" in p for p in problems)


# ---------------------------------------------------------------------------
# 5. The judge
# ---------------------------------------------------------------------------


def test_judge_refuses_to_be_the_candidate(monkeypatch):
    monkeypatch.setattr(benchmark, "_resolve_route",
                        lambda *a, **k: FakeRoute(model_ids=("solo",),
                                                  model_id="solo"))
    called: list = []
    monkeypatch.setattr(benchmark, "_dispatch",
                        lambda task, body: called.append(body))
    score = benchmark.judge_attempt(get_case("B5-minimal"), "some output",
                                    "solo")
    assert score.refused is True
    assert score.available is False
    assert score.score is None
    assert called == []          # nothing was dispatched at all


def test_judge_picks_an_independent_model_and_parses_the_verdict(monkeypatch):
    dispatch = replies("VERDICT=YES; SCORE=82; WHY=coherent and complete.")
    monkeypatch.setattr(benchmark, "_dispatch", dispatch)
    score = benchmark.judge_attempt(get_case("B5-minimal"), "some output", "m1")
    assert score.judge_model == "m2"
    assert score.verdict == "YES"
    assert score.score == 82.0
    assert score.available is True
    assert dispatch.seen[0]["model_key"] == "m2"


def test_judge_without_a_route_is_unavailable_not_refused(monkeypatch):
    monkeypatch.setattr(benchmark, "_resolve_route",
                        lambda *a, **k: FakeRoute(execution="gap",
                                                  reasons=("no text model",)))
    score = benchmark.judge_attempt(get_case("B5-minimal"), "out", "m1")
    assert score.available is False and score.refused is False
    assert "no text model" in score.detail


def test_judge_fault_degrades_to_unavailable(monkeypatch):
    def boom(task, body):
        raise RuntimeError("judge exploded")
    monkeypatch.setattr(benchmark, "_dispatch", boom)
    score = benchmark.judge_attempt(get_case("B5-minimal"), "out", "m1")
    assert score.available is False
    assert "judge exploded" in score.detail


def test_judge_prompt_carries_the_track_rubric_and_reply_format():
    prompt = benchmark.build_judge_prompt(get_case("A1-partial"), "OUTPUT")
    assert "narrative coherence" in prompt
    assert "VERDICT=YES|NO; SCORE=0-100" in prompt


# ---------------------------------------------------------------------------
# 6. One attempt, end to end (fake dispatch)
# ---------------------------------------------------------------------------


def test_run_case_track_a_records_quality_and_performance(monkeypatch):
    dispatch = replies(json.dumps(GOOD_A1),
                       usage={"prompt_tokens": 900, "completion_tokens": 400})
    monkeypatch.setattr(benchmark, "_dispatch", dispatch)
    attempt, raw = benchmark.run_case(get_case("A1-partial"), "m1",
                                      config=benchmark.RunConfig(judge=False))
    assert attempt.ok is True
    assert attempt.deterministic.valid is True
    assert attempt.perf.latency_s is not None
    assert attempt.perf.prompt_tokens == 900
    assert attempt.perf.completion_tokens == 400
    assert attempt.perf.worker == "worker-a"
    assert attempt.perf.load_state == "loaded"
    assert attempt.registry_version is None or attempt.registry_version == "rv-test"
    assert raw.startswith("{")


def test_run_case_uses_one_bounded_repair_like_k110(monkeypatch):
    dispatch = replies("sorry, I cannot", json.dumps(GOOD_A1))
    monkeypatch.setattr(benchmark, "_dispatch", dispatch)
    attempt, _raw = benchmark.run_case(get_case("A1-partial"), "m1",
                                       config=benchmark.RunConfig(judge=False))
    assert attempt.deterministic.valid is True
    assert attempt.perf.dispatch_calls == 2
    assert "YOUR PREVIOUS REPLY WAS REJECTED" in dispatch.seen[1]["prompt"]


def test_run_case_without_a_route_is_an_honest_gap(monkeypatch):
    monkeypatch.setattr(benchmark, "_resolve_route",
                        lambda *a, **k: FakeRoute(execution="gap",
                                                  reasons=("nothing eligible",)))
    attempt, raw = benchmark.run_case(get_case("B5-minimal"), "m1")
    assert attempt.failure and attempt.failure.startswith("no-route")
    assert attempt.gap_code == "CAPABILITY_GAP"
    assert attempt.ok is False and raw == ""


def test_run_case_timeout_is_recorded_as_a_timeout(monkeypatch):
    def timeout(task, body):
        raise TimeoutError("dispatch:text.chat exceeded 240.0s")
    monkeypatch.setattr(benchmark, "_dispatch", timeout)
    attempt, _raw = benchmark.run_case(get_case("B5-minimal"), "m1")
    assert attempt.failure == "timeout"
    assert attempt.gap_code == "LLM_ERROR"


# ---------------------------------------------------------------------------
# 7. Mode plumbing: ceiling vs normalized
# ---------------------------------------------------------------------------


def test_mode_params_normalized_is_a_constant():
    params, source = benchmark.mode_params("normalized")
    assert params == benchmark.NORMALIZED_PARAMS
    assert source == "normalized:constant"


def test_mode_params_ceiling_reads_the_catalog_limit_when_there_is_one():
    class View:
        limits = {"context_tokens": 32768}
    params, source = benchmark.mode_params("ceiling", "m1", View())
    assert params["context_tokens"] == 32768
    assert "catalog.limits" in source
    params2, source2 = benchmark.mode_params("ceiling", "m1", None)
    assert params2["context_tokens"] == benchmark.CEILING_PARAMS["context_tokens"]
    assert source2.startswith("ceiling:default")


def test_mode_reaches_the_dispatch_body(monkeypatch):
    dispatch = replies(json.dumps(GOOD_B))
    monkeypatch.setattr(benchmark, "_dispatch", dispatch)
    for mode, expected in (("normalized", benchmark.NORMALIZED_PARAMS),
                           ("ceiling", benchmark.CEILING_PARAMS)):
        attempt, _ = benchmark.run_case(
            get_case("B2-sparse-notes"), "m1",
            config=benchmark.RunConfig(mode=mode, judge=False))
        body = dispatch.seen[-1]
        assert body["max_new_tokens"] == expected["max_new_tokens"]
        assert body["temperature"] == expected["temperature"]
        assert "context_tokens" not in body          # recorded, never sent
        assert attempt.perf.params["context_tokens"] == expected["context_tokens"]
        assert attempt.perf.params["context_tokens_enforced"] is False


def test_bad_mode_is_refused():
    with pytest.raises(ValueError):
        benchmark.RunConfig(mode="turbo")


# ---------------------------------------------------------------------------
# 8. The sweep: abort, repeats, run dir
# ---------------------------------------------------------------------------


def test_sweep_drops_a_model_after_two_consecutive_timeouts(monkeypatch, tmp_path):
    def timeout(task, body):
        raise TimeoutError("dispatch timed out")
    monkeypatch.setattr(benchmark, "_dispatch", timeout)
    run = benchmark.run_sweep(
        ["slowpoke"], run_dir=str(tmp_path),
        config=benchmark.RunConfig(tracks="B", limit_per_track=4, judge=False))
    assert "slowpoke" in run.aborted
    assert len(run.attempts) == benchmark.TIMEOUT_ABORT_STREAK
    assert all(a.failure == "timeout" for a in run.attempts)


def test_sweep_writes_a_reproducible_run_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "_dispatch", replies(json.dumps(GOOD_B)))
    run = benchmark.run_sweep(
        ["m1", "m2"], run_dir=str(tmp_path),
        config=benchmark.RunConfig(tracks="B", limit_per_track=2, judge=False))
    for name in ("cases.json", "environment.json", "attempts.jsonl",
                 "scores.json", "routing_matrix.json", "leaderboard.md",
                 "run.log"):
        assert os.path.isfile(os.path.join(str(tmp_path), name)), name
    rows = [json.loads(l) for l in
            open(os.path.join(str(tmp_path), "attempts.jsonl"))]
    assert len(rows) == 4 == len(run.attempts)
    env = json.load(open(os.path.join(str(tmp_path), "environment.json")))
    assert env["registry_version"] == "rv-test"
    assert env["models"] == ["m1", "m2"]
    assert env["config"]["mode"] == "normalized"
    assert env["vram_reserve_gib"] == benchmark.DEFAULT_VRAM_RESERVE_GIB
    assert os.path.isdir(os.path.join(str(tmp_path), "raw"))


def test_repeats_produce_variance_in_the_summary(monkeypatch, tmp_path):
    good, bad = json.dumps(GOOD_B), json.dumps(BAD_B)
    dispatch = replies(good, good, bad, bad)   # repair round then a bad pair
    monkeypatch.setattr(benchmark, "_dispatch", dispatch)
    run = benchmark.run_sweep(
        ["m1"], run_dir=str(tmp_path),
        config=benchmark.RunConfig(tracks="B", limit_per_track=1, repeats=2,
                                   judge=False))
    assert len(run.attempts) == 2
    stats = routing_matrix.summarize(run.rows)[0]
    assert stats["attempts"] == 2
    assert stats["deterministic_stdev"] is not None


def test_new_run_dir_claims_a_timestamped_directory(tmp_path):
    first = benchmark.new_run_dir(str(tmp_path))
    second = benchmark.new_run_dir(str(tmp_path))
    assert first != second
    assert os.path.basename(first).startswith("oracle-")
    assert os.path.isdir(os.path.join(first, "raw"))


def test_discover_models_reports_the_catalogs_reasons(monkeypatch):
    class Gap:
        class eligibility:
            eligible = False
            reasons = ("no text model is loadable on this fleet",)
        model_ids = ()
    monkeypatch.setattr(benchmark, "_capability_view", lambda *a, **k: Gap())
    models, reasons = benchmark.discover_models()
    assert models == ()
    assert reasons == ("no text model is loadable on this fleet",)


# ---------------------------------------------------------------------------
# 9. The routing matrix
# ---------------------------------------------------------------------------


def _row(operation, model, ok, det, judge, latency, failure=None):
    return {"operation": operation, "model": model, "ok": ok, "case_id": "x",
            "track": "B", "mode": "normalized", "failure": failure,
            "deterministic": {"score": det, "preservation": 1.0,
                              "contradiction_rate": 0.0, "completeness": 1.0,
                              "constraint_adherence": None, "accuracy": None},
            "judge": {"score": judge, "available": judge is not None,
                      "judge_model": "judge-x"},
            "perf": {"latency_s": latency, "tokens_per_s": 30.0,
                     "vram_used_delta_bytes": None}}


def test_matrix_orders_by_success_then_quality_then_latency():
    rows = [_row("plot.construct", "fast-but-wrong", False, 20.0, None, 2.0,
                 "dispatch_error"),
            _row("plot.construct", "good", True, 90.0, 80.0, 40.0),
            _row("plot.construct", "okay", True, 70.0, 60.0, 5.0)]
    matrix = routing_matrix.derive_matrix(rows, registry_version="rv-test",
                                          mode="normalized", run_id="r1")
    entry = matrix.entry("plot.construct")
    assert entry.primary == "good"
    assert entry.fallback == "okay"
    assert [c.model for c in entry.candidates][-1] == "fast-but-wrong"
    assert entry.evidence["primary"]["quality"] == 85.0
    assert entry.evidence["fallback"]["latency_s"] == 5.0


def test_matrix_never_promotes_a_model_that_never_succeeded():
    rows = [_row("plot.construct", "a", False, 10.0, None, 1.0, "timeout"),
            _row("plot.construct", "b", False, 12.0, None, 1.0, "timeout")]
    entry = routing_matrix.derive_matrix(rows).entry("plot.construct")
    assert entry.primary is None and entry.fallback is None
    assert "NO route" in entry.note


def test_matrix_flags_a_single_model_route_as_a_single_point_of_failure():
    rows = [_row("plot.construct", "a", True, 80.0, None, 1.0),
            _row("plot.construct", "b", False, 10.0, None, 1.0, "timeout")]
    entry = routing_matrix.derive_matrix(rows).entry("plot.construct")
    assert entry.primary == "a" and entry.fallback is None
    assert "single point of failure" in entry.note


def test_matrix_roundtrips_through_json():
    rows = [_row("plot.construct", "a", True, 80.0, 70.0, 3.0),
            _row("screenplay.complete", "b", True, 60.0, None, 9.0)]
    matrix = routing_matrix.derive_matrix(rows, registry_version="rv-test",
                                          mode="ceiling", run_id="r2")
    encoded = json.dumps(matrix.to_dict(), sort_keys=True, default=str)
    restored = routing_matrix.RoutingMatrix.from_dict(json.loads(encoded))
    assert json.dumps(restored.to_dict(), sort_keys=True, default=str) == encoded
    assert restored.operations == matrix.operations
    assert restored.registry_version == "rv-test"


def test_best_route_reads_a_saved_matrix_from_disk(tmp_path, monkeypatch):
    rows = [_row("plot.construct", "a", True, 80.0, 70.0, 3.0),
            _row("plot.construct", "b", True, 60.0, 50.0, 4.0)]
    matrix = routing_matrix.derive_matrix(rows, registry_version="rv-test",
                                          mode="normalized", run_id="r3")
    path = os.path.join(str(tmp_path), "routing_matrix.json")
    assert routing_matrix.save_matrix(matrix, path) is True
    choice = routing_matrix.best_route("plot.construct", path=path)
    assert choice.primary == "a" and choice.fallback == "b"
    assert choice.registry_version == "rv-test"
    assert choice.evidence["primary"]["deterministic"] == 80.0

    monkeypatch.setenv(routing_matrix.MATRIX_PATH_ENV, path)
    assert routing_matrix.best_route("plot.construct").primary == "a"
    assert routing_matrix.best_route("nope.nothing") is None


def test_best_route_without_a_matrix_is_none(monkeypatch):
    monkeypatch.delenv(routing_matrix.MATRIX_PATH_ENV, raising=False)
    assert routing_matrix.best_route("plot.construct") is None


# ---------------------------------------------------------------------------
# k114's follow-up, landed: ``load_latest_matrix`` — find the newest
# ``oracle-*`` run dir's matrix and verify its ``registry_version`` against
# the live catalog before handing it back. Never a stale route.
# ---------------------------------------------------------------------------


def _one_model_matrix(model: str, registry_version: str,
                      operation: str = "plot.construct") -> "routing_matrix.RoutingMatrix":
    rows = [_row(operation, model, True, 80.0, 70.0, 3.0)]
    return routing_matrix.derive_matrix(rows, registry_version=registry_version,
                                        mode="normalized", run_id="rX")


def test_load_latest_matrix_finds_the_newest_run_dir_and_verifies_version(
        tmp_path):
    root = str(tmp_path)
    old_dir = os.path.join(root, "oracle-20260101-0000")
    new_dir = os.path.join(root, "oracle-20260102-0000")
    os.makedirs(old_dir)
    os.makedirs(new_dir)
    old_path = os.path.join(old_dir, "routing_matrix.json")
    new_path = os.path.join(new_dir, "routing_matrix.json")
    routing_matrix.save_matrix(_one_model_matrix("old-model", "rv-x"), old_path)
    routing_matrix.save_matrix(_one_model_matrix("new-model", "rv-x"), new_path)
    # the newer FILE's mtime, not the alphabetically-later dir name, decides
    # "latest" — set explicitly so the test does not depend on clock
    # resolution between two writes microseconds apart.
    now = os.path.getmtime(new_path)
    os.utime(old_path, (now - 100.0, now - 100.0))
    os.utime(new_path, (now, now))

    matrix, reason = routing_matrix.load_latest_matrix(
        root=root, live_registry_version=lambda: "rv-x")
    assert matrix is not None
    assert matrix.entry("plot.construct").primary == "new-model"
    assert "verified" in reason


def test_load_latest_matrix_refuses_a_stale_registry_version(tmp_path):
    root = str(tmp_path)
    run_dir = os.path.join(root, "oracle-20260101-0000")
    os.makedirs(run_dir)
    routing_matrix.save_matrix(
        _one_model_matrix("m", "rv-old"),
        os.path.join(run_dir, "routing_matrix.json"))

    matrix, reason = routing_matrix.load_latest_matrix(
        root=root, live_registry_version=lambda: "rv-new")
    assert matrix is None
    assert "rv-old" in reason and "rv-new" in reason
    assert "not honoured" in reason


def test_load_latest_matrix_with_no_run_dirs_is_none_with_a_reason(tmp_path):
    matrix, reason = routing_matrix.load_latest_matrix(root=str(tmp_path))
    assert matrix is None
    assert "no oracle-*" in reason


def test_load_latest_matrix_ignores_a_run_dir_with_no_matrix_file(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "oracle-20260101-0000"))   # no matrix here
    matrix, reason = routing_matrix.load_latest_matrix(root=root)
    assert matrix is None
    assert "no oracle-*" in reason


def test_load_latest_matrix_never_guesses_when_the_live_version_is_unreadable(
        tmp_path):
    root = str(tmp_path)
    run_dir = os.path.join(root, "oracle-20260101-0000")
    os.makedirs(run_dir)
    routing_matrix.save_matrix(
        _one_model_matrix("m", "rv-1"),
        os.path.join(run_dir, "routing_matrix.json"))

    matrix, reason = routing_matrix.load_latest_matrix(
        root=root, live_registry_version=lambda: None)
    assert matrix is None
    assert "could not read the live catalog" in reason


def test_load_latest_matrix_honours_the_run_root_env_var(tmp_path, monkeypatch):
    root = str(tmp_path)
    run_dir = os.path.join(root, "oracle-20260101-0000")
    os.makedirs(run_dir)
    routing_matrix.save_matrix(
        _one_model_matrix("m", "rv-1"),
        os.path.join(run_dir, "routing_matrix.json"))
    monkeypatch.setenv(routing_matrix.RUN_ROOT_ENV, root)

    matrix, reason = routing_matrix.load_latest_matrix(
        live_registry_version=lambda: "rv-1")
    assert matrix is not None and "verified" in reason


def test_composite_is_derived_and_never_used_for_ranking():
    slow_and_good = {"deterministic_mean": 75.0, "judge_mean": 75.0,
                     "latency_mean_s": 300.0}
    fast_and_worse = {"deterministic_mean": 60.0, "judge_mean": 60.0,
                      "latency_mean_s": 1.0}
    # The composite likes the fast one; the RANKING must not.
    assert routing_matrix.composite_of(fast_and_worse) > \
        routing_matrix.composite_of(slow_and_good)
    rows = [_row("plot.construct", "slow-good", True, 75.0, 75.0, 300.0),
            _row("plot.construct", "fast-bad", True, 60.0, 60.0, 1.0)]
    entry = routing_matrix.derive_matrix(rows).entry("plot.construct")
    assert entry.primary == "slow-good"   # quality wins the RANKING


def test_leaderboard_renders_separate_quality_and_performance_tables():
    rows = [_row("plot.construct", "a", True, 80.0, 70.0, 3.0),
            _row("plot.construct", "b", False, 10.0, None, 9.0, "timeout")]
    matrix = routing_matrix.derive_matrix(rows, registry_version="rv-test",
                                          mode="normalized", run_id="r4",
                                          run_dir="/tmp/x")
    text = routing_matrix.render_leaderboard(matrix, rows)
    assert "## Routing matrix" in text
    assert "## Quality (no performance in this table)" in text
    assert "## Performance (no quality in this table)" in text
    assert "## Composite (derived AFTER the two tables above)" in text
    assert routing_matrix.FORMULA_NOTE in text
    assert "rv-test" in text and "`a`" in text


def test_leaderboard_of_an_empty_run_says_so():
    matrix = routing_matrix.derive_matrix([], run_id="empty")
    text = routing_matrix.render_leaderboard(matrix, [])
    assert "No attempts were recorded" in text


# ---------------------------------------------------------------------------
# 10. Contracts this benchmark leans on (drift alarms)
# ---------------------------------------------------------------------------


def test_authoring_contract_is_k110s_own():
    from hugpy_oracle import screenplay
    assert callable(getattr(screenplay, "_author", None)), (
        "benchmark.author_completion reuses k110's bounded author loop; if it "
        "was renamed, the benchmark stops measuring the pipeline's behaviour")


def test_payload_accepts_a_plain_mapping_and_an_object():
    assert benchmark._payload({"text": "hi"}) == {"text": "hi"}
    assert benchmark._payload("hi")["text"] == "hi"


def test_pinned_judge_is_used_when_eligible_and_not_the_candidate(monkeypatch):
    dispatch = replies("VERDICT=NO; SCORE=31; WHY=thin.")
    monkeypatch.setattr(benchmark, "_dispatch", dispatch)
    score = benchmark.judge_attempt(get_case("B5-minimal"), "out", "m1",
                                    preferred_judge="m2")
    assert score.judge_model == "m2"
    assert score.score == 31.0
    assert "operator-pinned" in score.detail


def test_pinned_judge_that_is_the_candidate_is_refused_not_swapped(monkeypatch):
    called: list = []
    monkeypatch.setattr(benchmark, "_dispatch",
                        lambda task, body: called.append(body))
    score = benchmark.judge_attempt(get_case("B5-minimal"), "out", "m2",
                                    preferred_judge="m2")
    assert score.refused is True and score.available is False
    assert called == []


def test_unpinned_judge_falls_back_when_the_pin_is_not_eligible():
    model, why = benchmark.pick_judge("m1", FakeRoute(), preferred="ghost")
    assert model == "m2" and "not in the eligible set" in why


# ---------------------------------------------------------------------------
# or-k10 — peak-VRAM sampler
# ---------------------------------------------------------------------------


def _central_seam(values):
    """A fake ``_vram_snapshot`` that walks ``values`` (last value repeats)."""
    it = iter(values)
    last = {"v": values[-1]}

    def snap():
        try:
            v = next(it)
        except StopIteration:
            v = last["v"]
        if v is None:
            return None
        return {"workers": [{"id": "worker-a", "vram_total": 100,
                             "vram_used": v, "vram_free": 100 - v}]}
    return snap


def test_vram_sampler_noop_when_no_seam_answers():
    s = benchmark._VramSampler("worker-a", sample_ms=5, nvml=lambda: None,
                               snapshot=lambda: None)
    with s:
        time.sleep(0.03)
    assert s.perf_fields() == {"vram_peak_bytes": None,
                               "vram_sample_count": 0, "vram_sampler": None}
    assert s._thread is None


def test_vram_sampler_disabled_by_zero_cadence():
    calls = {"n": 0}

    def nvml():
        calls["n"] += 1
        return 7
    s = benchmark._VramSampler("worker-a", sample_ms=0, nvml=nvml)
    with s:
        time.sleep(0.02)
    assert calls["n"] == 0
    assert s.perf_fields()["vram_sampler"] is None


def test_vram_sampler_prefers_nvml_and_records_peak():
    seq = iter([10, 50, 30, 20])
    nvml = lambda: next(seq, 20)  # noqa: E731
    s = benchmark._VramSampler("worker-a", sample_ms=5, nvml=nvml,
                               snapshot=_central_seam([999]))
    with s:
        time.sleep(0.06)
    f = s.perf_fields()
    assert f["vram_sampler"] == "nvml"
    assert f["vram_peak_bytes"] == 50
    assert f["vram_sample_count"] >= 3


def test_vram_sampler_falls_back_to_central_scoped_to_worker():
    snap = _central_seam([10, 80, 40])
    s = benchmark._VramSampler("worker-a", sample_ms=5, nvml=lambda: None,
                               snapshot=snap)
    with s:
        time.sleep(0.06)
    f = s.perf_fields()
    assert f["vram_sampler"] == "central"
    assert f["vram_peak_bytes"] == 80
    assert f["vram_sample_count"] >= 2


def test_vram_sampler_uses_module_seam_when_unspecified(monkeypatch):
    monkeypatch.setattr(benchmark, "_nvml_reader", lambda: None)
    monkeypatch.setattr(benchmark, "_vram_snapshot", _central_seam([3, 9]))
    s = benchmark._VramSampler("worker-a", sample_ms=5).start()
    time.sleep(0.03)
    s.stop()
    assert s.source == "central" and s.peak_bytes == 9


def test_vram_sampler_skips_failed_samples():
    seq = iter([5, None, 12, None])
    s = benchmark._VramSampler(None, sample_ms=5, nvml=lambda: next(seq, None))
    with s:
        time.sleep(0.05)
    assert s.peak_bytes == 12
    assert s.sample_count == 2


def test_perf_record_carries_peak_fields_and_reports_summarize_them():
    perf = benchmark.PerfRecord(model="m", vram_used_delta_bytes=4,
                                vram_peak_bytes=99, vram_sample_count=3,
                                vram_sampler="central")
    d = perf.to_dict()
    assert d["vram_used_delta_bytes"] == 4
    assert d["vram_peak_bytes"] == 99
    assert d["vram_sample_count"] == 3
    assert d["vram_sampler"] == "central"
    rows = [{"model": "m", "perf": d},
            {"model": "m", "perf": dict(d, vram_peak_bytes=120,
                                        vram_sampler="nvml")},
            {"model": "n", "perf": {"vram_peak_bytes": None,
                                    "vram_sample_count": 0}}]
    peaks = benchmark.vram_peak_per_model(rows)
    assert peaks["m"]["vram_peak_bytes"] == 120
    assert peaks["m"]["vram_sampler"] == "nvml"
    assert peaks["m"]["vram_sample_count"] == 6
    assert peaks["m"]["attempts"] == 2
    assert peaks["n"]["vram_peak_bytes"] is None


def test_run_config_sample_ms_round_trips():
    cfg = benchmark.RunConfig(vram_sample_ms=50)
    assert cfg.to_dict()["vram_sample_ms"] == 50
    assert benchmark.RunConfig().vram_sample_ms == benchmark.DEFAULT_VRAM_SAMPLE_MS
    assert benchmark.StationaryConfig().to_dict()["vram_sample_ms"] == \
        benchmark.DEFAULT_VRAM_SAMPLE_MS


def test_run_case_records_peak_alongside_delta(monkeypatch):
    monkeypatch.setattr(benchmark, "_dispatch", replies(json.dumps(GOOD_A1)))
    monkeypatch.setattr(benchmark, "_vram_snapshot",
                        _central_seam([10, 10, 70, 30, 30]))
    attempt, _ = benchmark.run_case(
        get_case("A1-partial"), "m1",
        config=benchmark.RunConfig(judge=False, vram_sample_ms=1))
    perf = attempt.perf.to_dict()
    assert perf["vram_sampler"] == "central"
    assert perf["vram_peak_bytes"] >= 30
    assert perf["vram_sample_count"] >= 2
    assert isinstance(perf["vram_used_delta_bytes"], int)


def test_run_case_peak_is_none_when_seam_is_silent(monkeypatch):
    monkeypatch.setattr(benchmark, "_dispatch", replies(json.dumps(GOOD_A1)))
    attempt, _ = benchmark.run_case(get_case("A1-partial"), "m1",
                                    config=benchmark.RunConfig(judge=False))
    perf = attempt.perf.to_dict()
    assert perf["vram_used_delta_bytes"] is None
    assert perf["vram_peak_bytes"] is None
    assert perf["vram_sample_count"] == 0
    assert perf["vram_sampler"] is None
