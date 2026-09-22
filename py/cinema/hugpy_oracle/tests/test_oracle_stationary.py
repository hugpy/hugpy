"""k109b — the stationary-prompt full-fleet sweep, offline.

Every test here runs with NO fleet: no worker, no GPU, no registry read, no
clock dependency and no network. The scenario is pure data, the point map is
pure data, the validators are pure functions, and the two live seams the sweep
uses (dispatch and the fleet's VRAM meter) are module-level and monkeypatched.

What is asserted, in the order the deliverable asks for it:

  * the scenario's digests are STABLE and per-piece;
  * the point map covers all sixteen lifecycle steps, honestly, with the
    NO_CANDIDATES points naming what is missing;
  * verdict classification;
  * the silent-wav and blank-image content guards;
  * resumability — a journal skips completed cells and re-runs dispatch faults;
  * the matrix extension keeps k109's loader and ``best_route`` working.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_stationary.py -q
"""
from __future__ import annotations

import json
import logging
import math
import os
import struct
import sys
import wave

logging.disable(logging.INFO)  # silence the models_config registry chatter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from hugpy_oracle import benchmark, benchmark_cases, routing_matrix, stationary_scenario as scenario


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wav(path: str, amplitude: int, seconds: float = 0.5,
               rate: int = 16000) -> str:
    """A 16-bit PCM wav of a 440Hz tone at ``amplitude``. amplitude=0 is the
    digital-silence failure this sweep's audio guard exists to catch."""
    frames = int(rate * seconds)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(
            struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 440 * i / rate)))
            for i in range(frames)))
    return path


def _cell(**kwargs) -> benchmark.Cell:
    base = dict(point_id="p03-plot", step=3, operation="plot.construct",
                model="m1", capability="text.chat", verdict="capable",
                stage="llm")
    base.update(kwargs)
    return benchmark.Cell(**base)


# ---------------------------------------------------------------------------
# The scenario: digests, shape, derivation
# ---------------------------------------------------------------------------


def test_scenario_digest_is_stable_across_calls():
    """Two calls in one process agree. Nothing in the digest reads a clock, an
    environment variable or an unordered set."""
    assert scenario.scenario_digest() == scenario.scenario_digest()
    assert scenario.scenario_digest().startswith("sha256:")


def test_scenario_digest_is_stable_across_a_fresh_import():
    """A reimport in a clean module namespace produces the SAME digest — which
    is what makes it safe to compare a run from today against one from a week
    ago."""
    import importlib
    fresh = importlib.reload(scenario)
    assert fresh.scenario_digest() == scenario.scenario_digest()


def test_scenario_digest_moves_when_a_piece_moves():
    """The digest is a function of the CONTENT. Computed here over a mutated
    copy of the parts rather than by editing the module, so the test proves
    sensitivity without leaving the real scenario changed."""
    parts = scenario.scenario_parts()
    baseline = scenario._sha256(scenario._canonical(parts))
    parts["tone"] = parts["tone"] + " and warm"
    assert scenario._sha256(scenario._canonical(parts)) != baseline


def test_part_digests_cover_every_piece_of_the_brief():
    parts = scenario.scenario_parts()
    digests = scenario.part_digests()
    assert set(digests) == set(parts)
    assert all(v.startswith("sha256:") for v in digests.values())


def test_scenario_is_two_characters_three_lines_two_locations():
    """The brief the operator asked for, asserted rather than described."""
    play = scenario.SCENARIO_SCREENPLAY
    assert set(play.characters) == {"NIA", "TEODOR"}
    assert len(scenario.CHARACTER_SHEETS) == 2
    lines = [ln for scene in play.scenes for ln in scene.dialogue]
    assert len(lines) == 3
    assert {scene.location for scene in play.scenes} == {"HARBOUR WALL",
                                                         "PILOT HUT"}


def test_hero_shot_exists_in_the_scenarios_own_derived_shot_plan():
    """The single shot every image, video and VLM candidate works on must be a
    shot the pipeline would really produce. A hero shot that is not in the
    scenario's own shot plan is a scenario bug, and it fails HERE rather than
    at 3am inside a six-hour sweep."""
    ok, detail = scenario.shot_request_is_derivable()
    assert ok, detail


def test_fixed_segment_durations_match_the_derived_shot_ids():
    """Step 16 plans over fixed durations; step 7 and 12 cover derived shot
    ids. If those two id sets ever disagree the sweep is asking two different
    films about one brief."""
    assert scenario.segment_duration_ids() == scenario.derived_shot_ids()


def test_derived_continuity_covers_every_scene():
    bible = scenario.derived_continuity()
    assert [entry.segment_id for entry in bible.entries] == \
        list(scenario.SCENARIO_SCREENPLAY.scene_ids)


# ---------------------------------------------------------------------------
# The lifecycle point map — sixteen steps, honestly
# ---------------------------------------------------------------------------


def test_point_map_covers_all_sixteen_lifecycle_steps():
    steps = sorted({p.step for p in scenario.LIFECYCLE_POINTS})
    assert steps == list(range(1, 17))


def test_gap_points_name_the_missing_capability_and_claim_no_operation():
    """The gap IS the data — an unnamed gap is not evidence."""
    gaps = scenario.gap_points()
    assert gaps, "the honest answer for this fleet is not 'no gaps'"
    for point in gaps:
        assert point.missing_capability
        assert not point.operations
        assert point.note.strip()


def test_a_gap_point_without_a_named_capability_is_refused():
    with pytest.raises(ValueError, match="must NAME the missing capability"):
        scenario.LifecyclePoint(step=8, point_id="p-bad", name="x", kind="gap")


def test_a_measured_point_must_declare_an_operation():
    with pytest.raises(ValueError, match="must declare at least one operation"):
        scenario.LifecyclePoint(step=3, point_id="p-bad", name="x", kind="llm")


def test_the_spatial_folds_and_tier3_are_the_no_candidate_points():
    """The four points this fleet genuinely cannot serve, by step number."""
    assert sorted(p.step for p in scenario.gap_points()) == [8, 10, 11, 13]


def test_pipeline_points_are_not_reported_as_gaps():
    """Steps 1, 2 and 9 are executed by this codebase. Saying 'no model is
    capable of the immutable run snapshot' would be a category error, so they
    are a separate KIND and carry no missing_capability."""
    pipeline = scenario.points_for_kind("pipeline")
    assert sorted(p.step for p in pipeline) == [1, 2, 9]
    assert all(not p.missing_capability for p in pipeline)


def test_every_operation_has_exactly_one_owning_point():
    """A routing-matrix key with two owners cannot produce a per-point verdict.
    The module asserts this at import; this test is what fails if that
    assertion is ever removed."""
    owners = scenario.POINT_FOR_OPERATION
    assert len(owners) == len(scenario.STATIONARY_OPERATIONS)
    assert set(owners) == set(scenario.STATIONARY_OPERATIONS)


def test_the_eight_llm_operations_are_the_briefed_ones():
    llm_ops = tuple(op for p in scenario.points_for_kind("llm")
                    for op in p.operations)
    assert set(llm_ops) == {
        "plot.construct", "screenplay.complete", "continuity.bible",
        "screenplay.breakdown", "shots.design", "segment.compile-prompt",
        "correction.notes", "postproduction.plan"}


# ---------------------------------------------------------------------------
# The reference frames and their answer key
# ---------------------------------------------------------------------------


def test_reference_frames_key_is_derived_from_the_violation_list():
    """expected_verdict is not typed twice. A frame whose two halves disagree
    cannot be constructed."""
    with pytest.raises(ValueError, match="disagrees with violations"):
        scenario.ReferenceFrame(frame_id="x", prompt="p", violations=("a",),
                                expected_verdict="YES")


def test_reference_frames_have_both_compliant_and_planted_members():
    """A judge that fails everything is not strict, it is useless — the
    compliant frames are what catch that, so there must be some."""
    compliant = [f for f in scenario.REFERENCE_FRAMES if not f.violations]
    planted = [f for f in scenario.REFERENCE_FRAMES if f.violations]
    assert len(compliant) >= 2
    assert len(planted) >= 4
    assert all(f.rationale for f in scenario.REFERENCE_FRAMES)


def test_the_reference_frame_key_limitation_is_stated_in_the_module():
    """The key is derived from the render PROMPT, not from a human reading the
    pixels. That is a heuristic and the module has to say so — this test is the
    thing that fails if somebody quietly deletes the disclosure."""
    basis = scenario.REFERENCE_FRAME_KEY_BASIS.lower()
    assert "prompt" in basis and "not from a human" in basis
    assert "grounding" in basis


# ---------------------------------------------------------------------------
# The preamble — the same brief for every model
# ---------------------------------------------------------------------------


def test_preamble_briefs_every_llm_operation():
    for point in scenario.points_for_kind("llm"):
        for operation in point.operations:
            text = scenario.stationary_preamble(operation)
            assert scenario.TONE in text
            assert "NIA" in text and "TEODOR" in text
            assert "cf1" in text          # the continuity fact set


def test_preamble_refuses_an_unbriefed_operation():
    """A silent empty preamble would let a new operation be swept against a
    different question than its peers, and nothing would say so."""
    with pytest.raises(KeyError, match="no stationary preamble"):
        scenario.stationary_preamble("plot.invent")


def test_correction_preamble_carries_the_one_fixed_rejection_report():
    text = scenario.stationary_preamble("correction.notes")
    assert scenario.REJECTED_VALIDATION_REPORT in text
    assert scenario.SHOT_REQUEST.spec_text in text


def test_postproduction_preamble_carries_the_fixed_durations_and_target():
    text = scenario.stationary_preamble("postproduction.plan")
    assert scenario.DELIVERY_TARGET in text
    for segment_id, duration in scenario.SEGMENT_DURATIONS_S:
        assert f"{segment_id}: {duration}s" in text


# ---------------------------------------------------------------------------
# The two stationary validators
# ---------------------------------------------------------------------------

_GOOD_CORRECTIONS = {"corrections": [
    {"check": "adherence.time_of_day", "locked_value": "overcast dusk, no sun",
     "correction": "render under flat overcast dusk light", "source": "shot_spec"},
    {"check": "adherence.wardrobe", "locked_value": "yellow foul-weather jacket",
     "correction": "NIA wears the bright yellow jacket", "source": "character_sheet"},
    {"check": "adherence.prop", "locked_value": "white dive slate on a lanyard",
     "correction": "restore the white dive slate", "source": "character_sheet"},
    {"check": "tone", "locked_value": "desaturated blue-grey, no flare",
     "correction": "desaturate and remove the flare", "source": "tone"},
]}


def test_correction_notes_accepts_one_note_per_failing_check():
    errors, facts = scenario.validate_correction_notes(_GOOD_CORRECTIONS)
    assert errors == ()
    assert facts["accuracy"] == 1.0
    assert facts["locked_value_cited"] == 4


def test_correction_notes_refuses_a_missing_failing_check():
    payload = {"corrections": _GOOD_CORRECTIONS["corrections"][:2]}
    errors, facts = scenario.validate_correction_notes(payload)
    assert any("adherence.prop" in e for e in errors)
    assert facts["accuracy"] < 1.0


def test_correction_notes_refuses_a_note_for_a_check_that_passed():
    payload = {"corrections": _GOOD_CORRECTIONS["corrections"] + [
        {"check": "adherence.cast_count", "locked_value": "two",
         "correction": "keep two people", "source": "shot_spec"}]}
    errors, _facts = scenario.validate_correction_notes(payload)
    assert any("which PASSED" in e for e in errors)


def test_correction_notes_refuses_chaining_off_the_rejected_attempt():
    """Step 15's whole rule: correct FROM the locked spec, never from the
    rejected result or the prompt that produced it."""
    payload = {"corrections": [dict(_GOOD_CORRECTIONS["corrections"][0],
                                    correction="same as before but at dusk")]}
    errors, facts = scenario.validate_correction_notes(payload)
    assert facts["chained"] == 1
    assert any("refer to the rejected attempt" in e for e in errors)


def _timeline(rows, export=True):
    payload = {"timeline": rows}
    if export:
        payload["export"] = {"container": "mp4", "fps": 24}
    return payload


_GOOD_TIMELINE = [
    {"segment_id": "s1-1", "start_s": 0.0, "end_s": 5.0, "transition": "CUT TO:"},
    {"segment_id": "s1-2", "start_s": 5.0, "end_s": 9.0, "transition": "CUT TO:"},
    {"segment_id": "s2-1", "start_s": 9.0, "end_s": 15.0, "transition": "CUT TO:"},
    {"segment_id": "s3-1", "start_s": 15.0, "end_s": 22.0, "transition": "FADE OUT."},
]


def test_timeline_accepts_a_gapless_partition_of_the_fixed_durations():
    errors, facts = scenario.validate_timeline(_timeline(_GOOD_TIMELINE))
    assert errors == ()
    assert facts["accuracy"] == 1.0 and facts["export_present"] is True


def test_timeline_detects_a_gap():
    rows = [dict(r) for r in _GOOD_TIMELINE]
    rows[1]["start_s"], rows[1]["end_s"] = 6.0, 10.0
    errors, facts = scenario.validate_timeline(_timeline(rows))
    assert facts["gaps"] >= 1
    assert any("gap(s) between windows" in e for e in errors)


def test_timeline_detects_an_overlap():
    rows = [dict(r) for r in _GOOD_TIMELINE]
    rows[1]["start_s"], rows[1]["end_s"] = 4.0, 8.0
    errors, facts = scenario.validate_timeline(_timeline(rows))
    assert facts["overlaps"] >= 1
    assert any("overlapping window" in e for e in errors)


def test_timeline_refuses_a_missing_export_block():
    errors, facts = scenario.validate_timeline(
        _timeline(_GOOD_TIMELINE, export=False))
    assert facts["export_present"] is False
    assert any("export" in e for e in errors)


def test_timeline_flags_an_invented_segment_id():
    rows = _GOOD_TIMELINE + [{"segment_id": "s9-9", "start_s": 22.0,
                              "end_s": 24.0, "transition": "CUT TO:"}]
    errors, facts = scenario.validate_timeline(_timeline(rows))
    assert facts["invented"] == 1
    assert any("invented segment id" in e for e in errors)


def test_prompt_chaining_check_names_the_markers_it_fired_on():
    hits, markers = scenario.check_no_prompt_chaining(
        ["a clean prompt", "continuing from the previous clip"])
    assert hits == 1
    assert "previous clip" in markers


# ---------------------------------------------------------------------------
# Verdict classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs,expected", [
    (dict(produced=True, validated=True, structured=True, refused=False),
     "capable"),
    (dict(produced=True, validated=False, structured=True, refused=False),
     "partial"),
    (dict(produced=True, validated=False, structured=False, refused=False),
     "incapable"),
    (dict(produced=False, validated=False, structured=False, refused=False),
     "incapable"),
    (dict(produced=True, validated=False, structured=False, refused=True),
     "refused"),
    (dict(produced=True, validated=True, structured=True, refused=True),
     "refused"),
    (dict(produced=True, validated=True, structured=True, refused=False,
          no_candidates=True), "NO_CANDIDATES"),
])
def test_verdict_classification(kwargs, expected):
    assert benchmark.classify_verdict(**kwargs) == expected


def test_a_cell_cannot_carry_an_unknown_verdict():
    with pytest.raises(ValueError, match="is not one of"):
        _cell(verdict="probably-fine")


def test_refusal_detection_only_reads_the_head_of_a_reply():
    """A valid screenplay whose DIALOGUE contains 'I cannot' is not a refusal.
    The window is what keeps that true."""
    assert benchmark.looks_like_refusal("I'm sorry, but I cannot create that.")
    tail = ("{" + "x" * 900 + "} NIA: I cannot assist with that, he said.")
    assert not benchmark.looks_like_refusal(tail)


def test_no_candidates_cell_names_the_missing_capability():
    point = scenario.POINTS_BY_ID["p10-fold1-capture"]
    cell = benchmark.no_candidates_cell(point, "nothing captures spatial data")
    assert cell.verdict == "NO_CANDIDATES"
    assert cell.model == "(none)"
    assert cell.evidence["missing_capability"] == list(point.missing_capability)
    assert cell.to_dict()["ok"] is False


# ---------------------------------------------------------------------------
# Content guards — a silent wav and a blank frame both score zero
# ---------------------------------------------------------------------------


def test_a_silent_wav_fails_the_content_guard(tmp_path):
    """The 2026-08-21 tts-silence fault: a valid, right-duration wav holding
    digital silence. Existence is not substance for audio."""
    path = _write_wav(str(tmp_path / "silent.wav"), amplitude=0)
    ok, detail = benchmark.audio_carries_sound(path)
    assert ok is False
    assert "peak 0/32767" in detail
    assert "silence floor" in detail


def test_a_loud_wav_passes_the_content_guard(tmp_path):
    path = _write_wav(str(tmp_path / "loud.wav"), amplitude=20000)
    ok, detail = benchmark.audio_carries_sound(path)
    assert ok is True
    assert "peak" in detail


def test_an_unmeasurable_file_is_never_called_silent(tmp_path):
    """A format this cannot read is reported as unmeasurable, not as silence —
    a guard that guessed would fail every model on a fleet that emits mp3."""
    path = str(tmp_path / "not-a-wav.wav")
    with open(path, "wb") as handle:
        handle.write(b"definitely not RIFF")
    assert benchmark.wav_levels(path) is None
    ok, detail = benchmark.audio_carries_sound(path)
    assert ok is True and "unmeasurable" in detail


def test_a_tts_cell_with_a_silent_wav_scores_zero_with_empty_output(tmp_path,
                                                                    monkeypatch):
    """End to end for the guard: the sweep's own TTS cell, with the dispatch
    stubbed to return a silent wav, must come back incapable/EMPTY_OUTPUT and
    must NOT be rescued by a correct duration."""
    path = _write_wav(str(tmp_path / "tts.wav"), amplitude=0, seconds=2.32)
    monkeypatch.setattr(benchmark, "_selected_worker", lambda m, t: "w")
    monkeypatch.setattr(benchmark, "_load_state", lambda m, w: "loaded")
    monkeypatch.setattr(benchmark, "_vram_snapshot", lambda: None)
    monkeypatch.setattr(benchmark, "_run_bounded",
                        lambda fn, deadline, label: fn())
    monkeypatch.setattr(benchmark, "_dispatch", lambda task, body: {
        "ok": True, "audio": [{"path": path, "duration_s": 2.32,
                               "sample_rate": 16000}]})
    point = scenario.POINTS_BY_ID["p16a-speak"]
    cell, _raw = benchmark.run_tts_cell(
        point, "silent-model", config=benchmark.StationaryConfig(),
        run_dir=str(tmp_path))
    assert cell.verdict == "incapable"
    assert cell.gap_code == "EMPTY_OUTPUT"
    assert cell.deterministic.valid is False
    assert any(c.key == "carries_sound" and not c.passed
               for c in cell.deterministic.checks)


def test_a_flat_image_fails_the_blank_guard(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    flat = str(tmp_path / "flat.png")
    Image.new("RGB", (64, 64), (18, 18, 18)).save(flat)
    ok, detail = benchmark.image_carries_content(flat)
    assert ok is False and "stdev 0.0" in detail


def test_a_textured_image_passes_the_blank_guard(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    noisy = str(tmp_path / "noisy.png")
    image = Image.new("L", (64, 64))
    image.putdata([(i * 7) % 256 for i in range(64 * 64)])
    image.save(noisy)
    ok, detail = benchmark.image_carries_content(noisy)
    assert ok is True and "stdev" in detail


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------


def test_journal_skips_completed_cells_and_the_last_row_wins(tmp_path):
    run_dir = str(tmp_path / "oracle-x")
    os.makedirs(run_dir)
    path = os.path.join(run_dir, benchmark.CELLS_FILE)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(_cell(verdict="incapable").to_dict()) + "\n")
        handle.write(json.dumps(_cell(verdict="capable").to_dict()) + "\n")
        handle.write("\n")                       # blank lines are tolerated
        handle.write("{not json\n")              # so is a torn write
    journal = benchmark.load_journal(run_dir)
    assert list(journal) == ["p03-plot|plot.construct|m1"]
    assert journal["p03-plot|plot.construct|m1"]["verdict"] == "capable"


def test_retry_failed_drops_dispatch_faults_but_keeps_findings(tmp_path):
    """A timeout is the fleet; a model that answered badly is a FINDING, and
    re-rolling findings until they improve is not benchmarking."""
    run_dir = str(tmp_path / "oracle-y")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, benchmark.CELLS_FILE), "w",
              encoding="utf-8") as handle:
        handle.write(json.dumps(_cell(
            model="timed-out", verdict="incapable",
            failure="timeout").to_dict()) + "\n")
        handle.write(json.dumps(_cell(
            model="answered-prose", verdict="incapable").to_dict()) + "\n")
    kept = benchmark.load_journal(run_dir, retry_failed=True)
    assert "p03-plot|plot.construct|answered-prose" in kept
    assert "p03-plot|plot.construct|timed-out" not in kept
    assert len(benchmark.load_journal(run_dir)) == 2


def test_a_journalled_cell_round_trips_through_the_resume_rehydration():
    cell = _cell(verdict="partial", note="two validator errors",
                 deterministic=benchmark.DeterministicScore(
                     valid=False, error_count=2, completeness=0.5),
                 perf=benchmark.PerfRecord(model="m1", latency_s=12.5,
                                           vram_used_delta_bytes=1 << 30))
    back = benchmark._cell_from_row(cell.to_dict())
    assert back.key == cell.key
    assert back.verdict == "partial"
    assert back.deterministic.score == cell.deterministic.score
    assert back.perf.latency_s == 12.5
    assert back.perf.vram_used_delta_bytes == 1 << 30


def test_resume_dir_refuses_a_run_id_that_does_not_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="--resume takes the run id"):
        benchmark.resume_dir("oracle-not-here", str(tmp_path))
    made = str(tmp_path / "oracle-here")
    os.makedirs(made)
    assert benchmark.resume_dir("oracle-here", str(tmp_path)) == made


def test_resume_command_is_a_command_an_operator_can_paste():
    command = benchmark.resume_command("oracle-20260821-0231-k109b")
    assert "sweep --resume oracle-20260821-0231-k109b" in command


# ---------------------------------------------------------------------------
# The matrix extension stays compatible with k109 / k114b
# ---------------------------------------------------------------------------


def test_stationary_rows_derive_a_matrix_with_the_scenario_stamped():
    rows = [_cell(model="strong", verdict="capable",
                  deterministic=benchmark.DeterministicScore(valid=True),
                  perf=benchmark.PerfRecord(model="strong", latency_s=5.0)).to_dict(),
            _cell(model="weak", verdict="partial",
                  deterministic=benchmark.DeterministicScore(valid=False),
                  perf=benchmark.PerfRecord(model="weak", latency_s=2.0)).to_dict()]
    matrix = routing_matrix.derive_matrix(
        rows, registry_version="rv", run_id="r", mode="stationary",
        scenario_version=scenario.SCENARIO_VERSION,
        scenario_digest=scenario.scenario_digest())
    entry = matrix.entry("plot.construct")
    assert entry.primary == "strong"
    assert matrix.scenario_version == scenario.SCENARIO_VERSION
    assert matrix.to_dict()["scenario_digest"] == scenario.scenario_digest()


def test_a_k109_matrix_file_without_a_scenario_still_loads_and_routes():
    """k114b's ``load_latest_matrix``/``best_route`` must keep reading the
    files k109 already wrote. The new fields are defaulted, never required."""
    k109_shape = {
        "schema": routing_matrix.SCHEMA_VERSION, "registry_version": "rv-old",
        "mode": "normalized", "run_id": "oracle-20260821-0115-pilot",
        "entries": {"plot.construct": {
            "operation": "plot.construct", "primary": "old-model",
            "fallback": None, "note": "n", "candidates": []}}}
    matrix = routing_matrix.RoutingMatrix.from_dict(k109_shape)
    assert matrix.scenario_version == ""
    choice = routing_matrix.best_route("plot.construct", matrix=matrix)
    assert choice.primary == "old-model"
    assert choice.registry_version == "rv-old"
    assert choice.scenario_version == ""


def test_a_stationary_matrix_survives_a_save_load_round_trip(tmp_path):
    rows = [_cell(model="m1",
                  deterministic=benchmark.DeterministicScore(valid=True),
                  perf=benchmark.PerfRecord(model="m1", latency_s=1.0)).to_dict()]
    matrix = routing_matrix.derive_matrix(
        rows, registry_version="rv", run_id="r", mode="stationary",
        scenario_version="salt-line/1", scenario_digest="sha256:abc")
    path = str(tmp_path / "routing_matrix.json")
    assert routing_matrix.save_matrix(matrix, path)
    reloaded = routing_matrix.load_matrix(path)
    assert reloaded.scenario_version == "salt-line/1"
    assert reloaded.scenario_digest == "sha256:abc"
    assert reloaded.entry("plot.construct").primary == "m1"


def test_the_k109_operations_are_untouched_by_the_additive_merge():
    """The stationary wave ADDS matrix keys. A collision would silently
    redefine a k109 key's shape and rescore every old run against the wrong
    validator, so the merge asserts — and so does this."""
    for name in ("screenplay.complete", "plot.construct", "breakdown.script",
                 "continuity.extract", "shotlist.build", "storyboard.prompts",
                 "segment.prompts", "assembly.plan"):
        assert name in benchmark_cases.OPERATIONS
    for name in benchmark_cases.STATIONARY_OPERATION_SPECS:
        assert benchmark_cases.OPERATIONS[name] is \
            benchmark_cases.STATIONARY_OPERATION_SPECS[name]


def test_a_prompt_built_without_a_preamble_is_byte_identical_to_k109s():
    """The preamble is additive. Omitted, k109's own runs must reproduce
    exactly — otherwise every historical row was measured against a prompt
    that no longer exists."""
    case = benchmark_cases.case("C1-breakdown")
    assert benchmark.build_prompt(case) == benchmark.build_workflow_prompt(case)
    assert not benchmark.build_prompt(case).startswith("THE FILM:")
    # and the Track C source really does default to k109's own fixture
    assert benchmark_cases.FIXTURE_SCREENPLAY.title in benchmark.build_prompt(case)


def test_a_prompt_built_with_the_preamble_carries_the_whole_brief():
    case = benchmark_cases.stationary_case_for("shots.design")
    prompt = benchmark.build_prompt(
        case, scenario.stationary_preamble("shots.design"),
        scenario.SCENARIO_SCREENPLAY)
    assert prompt.startswith("THE FILM: SALT LINE")
    assert "NIA" in prompt and scenario.TONE in prompt
    for shot_id in scenario.derived_shot_ids():
        assert shot_id in prompt


def test_every_stationary_case_maps_to_a_briefed_operation():
    for case in benchmark_cases.STATIONARY_CASES:
        assert case.operation in scenario.STATIONARY_BRIEFED_OPERATIONS
        assert scenario.stationary_preamble(case.operation)
    with pytest.raises(KeyError, match="no stationary case"):
        benchmark_cases.stationary_case_for("clip.render")


# ---------------------------------------------------------------------------
# The per-point reports
# ---------------------------------------------------------------------------


def _report_rows():
    points = [p.to_dict() for p in scenario.LIFECYCLE_POINTS]
    rows = [
        _cell(model="strong", verdict="capable",
              deterministic=benchmark.DeterministicScore(valid=True),
              perf=benchmark.PerfRecord(model="strong", latency_s=4.0)).to_dict(),
        _cell(model="middling", verdict="partial",
              perf=benchmark.PerfRecord(model="middling", latency_s=2.0)).to_dict(),
        benchmark.no_candidates_cell(
            scenario.POINTS_BY_ID["p08-spatial-feasibility"],
            "no geometry solver on this fleet").to_dict(),
    ]
    return rows, points


def test_summarize_points_ranks_the_capable_models_and_keeps_the_gaps():
    rows, _points = _report_rows()
    stats = {s["point_id"]: s for s in routing_matrix.summarize_points(rows)}
    assert stats["p03-plot"]["winner"] == "strong"
    assert stats["p03-plot"]["verdicts"]["partial"] == 1
    gap = stats["p08-spatial-feasibility"]
    assert gap["no_candidates"] is True and gap["winner"] is None
    assert gap["notes"] == ["no geometry solver on this fleet"]


def test_the_capability_grid_separates_gaps_from_model_free_steps():
    rows, points = _report_rows()
    grid = routing_matrix.render_capability_grid(
        rows, points, scenario_version="salt-line/1")
    assert "## Points with NO candidate on this fleet" in grid
    assert "## Model-free pipeline steps" in grid
    assert "p08-spatial-feasibility" in grid
    assert "`strong`" in grid
    # the gap row's model placeholder never becomes a grid row
    assert "| `(none)` |" not in grid


def test_the_per_point_leaderboard_renders_a_no_candidates_point_honestly():
    rows, points = _report_rows()
    report = routing_matrix.render_point_leaderboards(
        rows, points, scenario_version="salt-line/1")
    assert "Step 8" in report
    assert "**NO_CANDIDATES.** no geometry solver on this fleet" in report
    assert "`strong`" in report


def test_reports_render_from_rows_alone_with_no_fleet_and_no_benchmark_run():
    """The reports are PURE: rows in, markdown out. That is what makes
    ``--report-only`` able to rebuild everything from an old cells.jsonl on a
    box with the fleet switched off."""
    rows, points = _report_rows()
    assert routing_matrix.render_capability_grid(rows, points).strip()
    assert routing_matrix.render_point_leaderboards(rows, points).strip()
    assert routing_matrix.summarize_points([])==[]


# ---------------------------------------------------------------------------
# Fleet-degradation handling (the sweep runs while another agent repairs a
# worker, so these two distinctions are load-bearing rather than decorative)
# ---------------------------------------------------------------------------


def test_a_busy_worker_is_retryable_but_is_not_an_outage():
    """Measured live 2026-08-21: a probe that timed out left its model loading
    on the worker, and the immediate retry came back ``WorkerBusyError``.
    Busy means ALIVE. Pausing a six-hour sweep because three big models queued
    behind each other would be a bug wearing a safety feature's clothes."""
    busy = "WorkerBusyError: worker is already serving a request"
    assert benchmark.is_transient(busy) is True
    assert benchmark._fleet_unreachable(busy) is False


def test_transport_death_is_an_outage_and_a_plain_timeout_is_not():
    dead = "ConnectionRefusedError: connection refused to worker ae"
    assert benchmark._fleet_unreachable(dead) is True
    assert benchmark.is_transient(dead) is True
    slow = "timeout: k109b-probe:big-model did not answer within 120.0s"
    assert benchmark._fleet_unreachable(slow) is False
    assert benchmark.is_transient(slow) is False


def test_roster_is_ordered_resident_first_and_stably(monkeypatch):
    """The order is reproducible given the same fleet state, and the state each
    model was sorted on is returned so it can be checked afterwards."""
    states = {"cold-a": "cold", "hot-a": "loaded", "cold-b": "cold",
              "hot-b": "loaded", "warming": "loading"}
    monkeypatch.setattr(benchmark, "_selected_worker", lambda m, t: "w")
    monkeypatch.setattr(benchmark, "_load_state", lambda m, w: states[m])
    ordered, seen = benchmark.order_by_residency(list(states))
    assert ordered == ("hot-a", "hot-b", "warming", "cold-a", "cold-b")
    assert seen == states
    # stable: same input, same fleet, same answer
    assert benchmark.order_by_residency(list(states))[0] == ordered


def test_an_unreadable_load_state_sorts_mid_pack_and_never_raises(monkeypatch):
    def boom(model, worker):
        raise RuntimeError("the fleet meter is down")
    monkeypatch.setattr(benchmark, "_selected_worker", lambda m, t: "w")
    monkeypatch.setattr(benchmark, "_load_state", boom)
    ordered, seen = benchmark.order_by_residency(["a", "b"])
    assert ordered == ("a", "b")
    assert set(seen.values()) == {"unknown"}


# ---------------------------------------------------------------------------
# The human-confirmed frame key — the wave's most important honesty seam
# ---------------------------------------------------------------------------


def _stub_vlm(monkeypatch, replies):
    """Drive run_vlm_cell with canned VLM answers, keyed by frame id."""
    monkeypatch.setattr(benchmark, "_selected_worker", lambda m, t: "w")
    monkeypatch.setattr(benchmark, "_load_state", lambda m, w: "loaded")
    monkeypatch.setattr(benchmark, "_vram_snapshot", lambda: None)

    def ask(model, image_path, prompt, deadline_s=180.0):
        frame_id = os.path.splitext(os.path.basename(image_path))[0]
        text = replies[frame_id]
        from hugpy_oracle.evaluation import parse_judge_verdict
        parsed = parse_judge_verdict(text)
        return {"ok": True, "text": text, "verdict": parsed["verdict"],
                "score": parsed["score"], "why": parsed["why"],
                "latency_s": 1.0, "error": None}
    monkeypatch.setattr(benchmark, "ask_vlm", ask)


def _frames(tmp_path, ids):
    directory = tmp_path / "frames"
    directory.mkdir(exist_ok=True)
    out = {}
    for frame_id in ids:
        path = directory / f"{frame_id}.png"
        path.write_bytes(b"not really a png")
        out[frame_id] = str(path)
    return out


_CONFIRMATION = {
    "basis": "human inspection of the pixels",
    "frames": {
        "rf3-wrong-time": {"expected_verdict": "NO",
                           "violations": ["time_of_day"],
                           "cues": ["sun", "blue sky"]},
        "rf4-third-person": {"expected_verdict": "NO",
                             "violations": ["prop"],
                             "cues": ["slate"]},
    },
}


def test_the_confirmed_key_overrides_the_prompt_derived_one(tmp_path,
                                                            monkeypatch):
    """Live on 2026-08-21 the renderer did not land the third-person violation,
    so the PROMPT-derived key for rf4 was wrong about what is in the picture.
    A run dir that carries a human read of the pixels must win."""
    ids = ["rf3-wrong-time", "rf4-third-person"]
    _stub_vlm(monkeypatch, {
        "rf3-wrong-time": "VERDICT=NO; SCORE=20; WHY=bright sun and blue sky "
                          "on a wet stone harbour wall at dusk with two "
                          "figures, no green buoy",
        "rf4-third-person": "VERDICT=NO; SCORE=30; WHY=no white dive slate on "
                            "the man in yellow on the wet stone harbour wall "
                            "at dusk, two people, no green buoy",
    })
    point = scenario.POINTS_BY_ID["p14-validate"]
    cell, _raw = benchmark.run_vlm_cell(
        point, "judge-1", _frames(tmp_path, ids),
        config=benchmark.StationaryConfig(), confirmation=_CONFIRMATION)
    assert cell.evidence["key_source"] == "human-confirmed"
    assert cell.evidence["violation_hit"] == 1.0
    assert cell.verdict == "capable"
    by_frame = {a["frame_id"]: a for a in cell.evidence["answers"]}
    # the prompt key called rf4 a cast_count fault; the confirmed key does not
    assert by_frame["rf4-third-person"]["violations"] == ["prop"]


def test_without_a_confirmation_file_the_key_source_is_named_as_the_prompt():
    """Never silently either way — the row says which key graded it."""
    assert benchmark.load_frame_confirmation("/no/such/run/dir") == {}


def test_a_judge_that_answers_no_to_everything_does_not_score_capable(
        tmp_path, monkeypatch):
    """When every confirmed verdict is NO, agreement stops discriminating.
    ``names_the_actual_fault`` is what still separates a judge that looked from
    one that did not."""
    ids = ["rf3-wrong-time", "rf4-third-person"]
    _stub_vlm(monkeypatch, {
        "rf3-wrong-time": "VERDICT=NO; SCORE=10; WHY=it does not adhere",
        "rf4-third-person": "VERDICT=NO; SCORE=10; WHY=it does not adhere",
    })
    point = scenario.POINTS_BY_ID["p14-validate"]
    cell, _raw = benchmark.run_vlm_cell(
        point, "lazy-judge", _frames(tmp_path, ids),
        config=benchmark.StationaryConfig(), confirmation=_CONFIRMATION)
    assert cell.evidence["agreement"] == 1.0        # perfect on the key...
    assert cell.evidence["violation_hit"] == 0.0    # ...and it named nothing
    assert cell.evidence["key_discriminates"] is False
    assert cell.verdict != "capable"
    assert any(c.key == "agrees_with_key" and "WARNING" in c.detail
               for c in cell.deterministic.checks)


def test_a_malformed_confirmation_file_degrades_to_the_prompt_key(tmp_path):
    path = tmp_path / "frames"
    path.mkdir()
    (path / "human_confirmation.json").write_text("{not json")
    assert benchmark.load_frame_confirmation(str(tmp_path)) == {}
    (path / "human_confirmation.json").write_text('{"basis": "no frames key"}')
    assert benchmark.load_frame_confirmation(str(tmp_path)) == {}
