"""k104 — audio-first shot windows and the sibling SegmentSpec compiler.

Everything here is offline and deterministic: the compiler is pure over locked
artifacts and an injected ``prompt_writer``, so Stage 8's timing rules and
Stage 14's shape are exercised with no model, no worker and no GPU.

Locks:
  [1] Stage 8: windows follow the LINE TIMINGS. Padding only consumes silence
      that already exists, a long line splits at MEASURED pauses or not at all,
      a too-short window merges rather than stretches, and a window that cannot
      merge is emitted short instead of padded into a lie.
  [2] Stage 14 / invariant 9, structurally: the writer is called with the same
      locked context for every index, no SegmentSpec exists while any prompt is
      written, ``parents`` holds only lock-side digests, and a writer that
      reaches for a previous spec finds nothing to reach for.
  [3] the compiler refuses inputs the lock did not lock — by digest.
  [4] the emitted PlanGraph passes ``sibling_check`` and the k103 validator
      reports no SIBLING_VIOLATION, while a hand-built chained graph does.
  [5] sequential and parallel execution read the SAME graph: identical edges,
      identical node coverage, different batching.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_segments.py -q
"""
from __future__ import annotations

import dataclasses
import gc
import logging
import os
import sys

import pytest

logging.disable(logging.INFO)  # silence the models_config registry chatter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_oracle.audio_master import AudioMaster, LineTiming, WordTiming
from hugpy_oracle.contracts import ArtifactKind, GoalSpec
from hugpy_oracle.plan import (
    Edge,
    FrozenParams,
    NodeKind,
    PlanGraph,
    PlanNode,
    Port,
    SEGMENT_PARAM,
    goal_digest,
    sibling_check,
)
from hugpy_oracle.production import (
    ContinuityBible,
    ContinuityState,
    ProductionLock,
    ShotPlan,
    ShotPlanEntry,
)
from hugpy_oracle.segments import (
    JOINT_MODES,
    JOINT_MODE_PLAIN,
    LOCK_NODE_ID,
    SEGMENT_CAPABILITY,
    CompileRefused,
    LockedContext,
    LockedSegmentBrief,
    SegmentSpec,
    SiblingViolation,
    assert_siblings,
    build_locked_context,
    compile_segments,
    default_prompt_writer,
    execution_order,
    render_dependencies,
    segment_node_id,
    segment_seed,
    shot_plan_from_windows,
    shot_windows_from_audio,
    sibling_violations_in,
)
from hugpy_oracle.validator import ErrorCode, validate

from test_oracle_production import (  # noqa: E402
    make_bible, make_lock, make_master, make_plan, make_snapshot,
)

DIALOGUE = {"l1": "You came back.", "l2": "I never left, not really.",
            "l3": "Then take the coat off."}
GOAL = GoalSpec(objective="two characters, three lines",
                raw_prompt="two characters, three lines")


def master_of(*timings: LineTiming, total: float | None = None) -> AudioMaster:
    """An AudioMaster over hand-built k102 timings — no synthesis, no ASR."""
    last = max((t.next_start_s for t in timings), default=0.0)
    return AudioMaster(
        timeline_digest="timeline-abc", line_timings=tuple(timings),
        tracks=tuple((t.line_id, f"/a/{t.line_id}.wav") for t in timings),
        total_seconds=total if total is not None else last).lock()


def compile_fixture(**kw):
    """The three-line scene from the production fixtures, compiled."""
    snap, master, bible, plan = (make_snapshot(), make_master(), make_bible(),
                                 make_plan())
    lock = ProductionLock.lock(snap, audio_master=master, continuity=bible,
                               shot_plan=plan, locked_at="2026-08-20T10:05:00Z")
    call = dict(snapshot=snap, audio_master=master, continuity=bible,
                shot_plan=plan, tone=0.35, dialogue=DIALOGUE)
    call.update(kw)
    return lock, compile_segments(lock, **call)


# ===========================================================================
# [1] Stage 8 — audio-first shot windows
# ===========================================================================


def test_windows_follow_the_line_timings_exactly():
    master = master_of(LineTiming("l1", 0.0, 2.0, pause_after_s=0.5),
                       LineTiming("l2", 2.5, 5.0, pause_after_s=0.0))
    assert shot_windows_from_audio(master, min_shot_s=1.0, max_shot_s=8.0) == (
        (0.0, 2.0, ("l1",)), (2.5, 5.0, ("l2",)))


def test_padding_only_consumes_silence_that_already_exists():
    # l1 holds a 0.5 s pause, l2 holds none: padding takes 0.2 s after l1 and
    # nothing after l2, and nothing before l2 because l1's pause already owns
    # that silence. No window can eat a neighbour's audio.
    master = master_of(LineTiming("l1", 0.0, 2.0, pause_after_s=0.5),
                       LineTiming("l2", 2.5, 5.0, pause_after_s=0.0))
    windows = shot_windows_from_audio(master, min_shot_s=1.0, max_shot_s=8.0,
                                      pad_s=0.2)
    assert windows == ((0.0, 2.2, ("l1",)), (2.5, 5.0, ("l2",)))
    assert windows[0][1] <= windows[1][0]


def test_padding_is_capped_by_the_available_pause_not_by_the_request():
    master = master_of(LineTiming("l1", 0.0, 2.0, pause_after_s=0.1),
                       LineTiming("l2", 2.1, 4.0, pause_after_s=0.0))
    windows = shot_windows_from_audio(master, min_shot_s=1.0, max_shot_s=8.0,
                                      pad_s=5.0)
    assert windows[0] == (0.0, 2.1, ("l1",))     # 0.1 s available, 0.1 s taken


def test_padding_uses_the_lead_in_silence_of_the_first_line():
    master = master_of(LineTiming("l1", 1.0, 3.0, pause_after_s=0.0))
    assert shot_windows_from_audio(master, min_shot_s=1.0, max_shot_s=8.0,
                                   pad_s=0.4)[0][0] == 0.6


def test_a_long_line_splits_at_its_measured_word_pauses():
    words = (WordTiming("a", 0.0, 3.0), WordTiming("b", 3.5, 6.0),
             WordTiming("c", 6.5, 9.0), WordTiming("d", 9.5, 12.0))
    master = master_of(LineTiming("l1", 0.0, 12.0, words=words))
    windows = shot_windows_from_audio(master, min_shot_s=1.0, max_shot_s=5.0)
    # Cuts land on pause MIDPOINTS (3.25 / 6.25 / 9.25), never on a clock.
    assert windows == ((0.0, 3.25, ("l1",)), (3.25, 6.25, ("l1",)),
                       (6.25, 9.25, ("l1",)), (9.25, 12.0, ("l1",)))
    assert all(end - start <= 5.0 + 1e-6 for start, end, _ in windows)


def test_every_piece_of_a_split_line_still_names_that_line():
    words = (WordTiming("a", 0.0, 3.0), WordTiming("b", 3.5, 9.0))
    master = master_of(LineTiming("l1", 0.0, 9.0, words=words))
    windows = shot_windows_from_audio(master, min_shot_s=1.0, max_shot_s=5.0)
    assert len(windows) == 2
    assert {ids for _s, _e, ids in windows} == {("l1",)}


def test_a_long_line_with_no_measured_words_is_not_split():
    # This fleet's whisper path returns no word times (k102's record). Inventing
    # a cut point mid-word is exactly the arbitrary duration Stage 8 prohibits,
    # so the long window is emitted honestly and k107 gets to complain.
    master = master_of(LineTiming("l1", 0.0, 12.0))
    assert shot_windows_from_audio(master, min_shot_s=1.0, max_shot_s=5.0) == (
        (0.0, 12.0, ("l1",)),)


def test_a_long_line_whose_only_pause_is_out_of_range_is_not_split():
    words = (WordTiming("a", 0.0, 11.0), WordTiming("b", 11.5, 12.0))
    master = master_of(LineTiming("l1", 0.0, 12.0, words=words))
    assert shot_windows_from_audio(master, min_shot_s=1.0, max_shot_s=5.0) == (
        (0.0, 12.0, ("l1",)),)


def test_a_zero_width_word_boundary_is_not_a_pause():
    words = (WordTiming("a", 0.0, 6.0), WordTiming("b", 6.0, 12.0))
    master = master_of(LineTiming("l1", 0.0, 12.0, words=words))
    assert shot_windows_from_audio(master, min_shot_s=1.0, max_shot_s=8.0) == (
        (0.0, 12.0, ("l1",)),)


def test_a_short_window_merges_forward_into_the_next_line():
    master = master_of(LineTiming("l1", 0.0, 0.5, pause_after_s=0.0),
                       LineTiming("l2", 0.5, 3.0, pause_after_s=0.0))
    assert shot_windows_from_audio(master, min_shot_s=1.0, max_shot_s=8.0) == (
        (0.0, 3.0, ("l1", "l2")),)


def test_a_short_final_window_merges_backward():
    master = master_of(LineTiming("l1", 0.0, 3.0, pause_after_s=0.0),
                       LineTiming("l2", 3.0, 3.5, pause_after_s=0.0))
    assert shot_windows_from_audio(master, min_shot_s=1.0, max_shot_s=8.0) == (
        (0.0, 3.5, ("l1", "l2")),)


def test_a_short_window_that_cannot_merge_is_emitted_short_not_stretched():
    # Merging would break the ceiling, so the honest answer is a short shot —
    # k107's SHOT_TOO_SHORT, not silent padding that hides it.
    master = master_of(LineTiming("l1", 0.0, 1.0, pause_after_s=0.0),
                       LineTiming("l2", 1.0, 3.0, pause_after_s=0.0))
    windows = shot_windows_from_audio(master, min_shot_s=2.0, max_shot_s=2.5)
    assert windows == ((0.0, 1.0, ("l1",)), (1.0, 3.0, ("l2",)))
    assert windows[0][1] - windows[0][0] < 2.0


def test_windows_cover_every_line_exactly_once_and_never_overlap():
    master = make_master()
    windows = shot_windows_from_audio(master, min_shot_s=0.5, max_shot_s=8.0,
                                      pad_s=0.1)
    covered = [line for _s, _e, ids in windows for line in ids]
    assert sorted(covered) == sorted(master.line_ids)
    assert len(covered) == len(set(covered))
    assert all(a[1] <= b[0] + 1e-6 for a, b in zip(windows, windows[1:]))


def test_windows_of_an_empty_master_are_empty():
    assert shot_windows_from_audio(
        AudioMaster(timeline_digest="t", line_timings=(), tracks=(),
                    total_seconds=0.0).lock()) == ()


@pytest.mark.parametrize("kwargs", [
    {"min_shot_s": 0.0}, {"min_shot_s": -1.0},
    {"min_shot_s": 4.0, "max_shot_s": 2.0}, {"pad_s": -0.1},
])
def test_window_bounds_are_validated(kwargs):
    with pytest.raises(ValueError):
        shot_windows_from_audio(make_master(), **kwargs)


def test_shot_windows_refuses_something_that_is_not_an_audio_master():
    with pytest.raises(TypeError, match="AudioMaster"):
        shot_windows_from_audio(object())          # type: ignore[arg-type]


def test_shot_plan_from_windows_builds_a_lockable_plan():
    master = make_master()
    plan = shot_plan_from_windows(
        shot_windows_from_audio(master, min_shot_s=0.5, max_shot_s=8.0),
        rubric=("ana is recognizable",), camera={"shot_size": "medium"})
    assert plan.segment_ids == ("s1", "s2", "s3")
    assert plan.line_ids == master.line_ids
    assert plan.overlaps() == ()


# ===========================================================================
# [2] Stage 14 — sibling compilation
# ===========================================================================


def test_compile_yields_one_sibling_spec_per_shot():
    lock, specs = compile_fixture()
    assert [s.segment_id for s in specs] == ["s1", "s2", "s3"]
    assert [s.index for s in specs] == [0, 1, 2]
    assert {s.lock_digest for s in specs} == {lock.digest}
    for spec in specs:
        assert set(spec.parents) <= set(lock.parent_digests)
        assert spec.parents == lock.parent_digests
        assert spec.joint_mode == "cut"
        assert spec.tone == 0.35
        assert spec.prompt.strip()


def test_no_spec_names_another_spec_as_a_parent():
    _lock, specs = compile_fixture()
    digests = {s.digest for s in specs}
    for spec in specs:
        assert digests.isdisjoint(spec.parents)
    assert sibling_violations_in(specs) == ()
    assert_siblings(specs)                     # does not raise


def test_specs_carry_their_own_locked_material_and_nothing_elses():
    _lock, specs = compile_fixture()
    for spec in specs:
        assert spec.continuity.segment_id == spec.segment_id
        assert spec.shot.segment_id == spec.segment_id
        assert spec.rubric == spec.shot.rubric
        assert spec.audio_window == spec.shot.window
        assert spec.spatial_ref is None        # Fold artifacts are Wave 5


def test_compilation_is_deterministic():
    lock_a, specs_a = compile_fixture()
    lock_b, specs_b = compile_fixture()
    assert lock_a.digest == lock_b.digest
    assert [s.digest for s in specs_a] == [s.digest for s in specs_b]


def test_seeds_derive_from_the_lock_and_the_salt_is_the_repair_dial():
    lock, specs = compile_fixture()
    assert [s.seed_base for s in specs] == [
        segment_seed(lock.digest, sid, 0) for sid in ("s1", "s2", "s3")]
    assert all(0 <= s.seed_base < 2 ** 32 for s in specs)
    _lock, salted = compile_fixture(seed_salt=7)
    assert [s.seed_base for s in salted] != [s.seed_base for s in specs]


def test_specs_round_trip_without_loss():
    _lock, specs = compile_fixture()
    for spec in specs:
        again = SegmentSpec.from_dict(spec.to_dict())
        assert again == spec
        assert again.digest == spec.digest


# --- the writer seam -------------------------------------------------------


def test_the_writer_gets_the_same_locked_context_object_for_every_index():
    seen: list[tuple[int, int]] = []

    def writer(context, index):
        seen.append((id(context), index))
        return f"shot {index}"

    _lock, specs = compile_fixture(prompt_writer=writer)
    assert [i for _c, i in seen] == [0, 1, 2]
    assert len({c for c, _i in seen}) == 1          # ONE context, three calls
    assert [s.prompt for s in specs] == ["shot 0", "shot 1", "shot 2"]


def test_the_locked_context_has_no_field_that_could_hold_a_written_prompt():
    # This is the enforcement, not a convention: invariant 9's prohibited input
    # has nowhere to live.
    brief_fields = {f.name for f in dataclasses.fields(LockedSegmentBrief)}
    assert brief_fields.isdisjoint({"prompt", "negative_prompt", "spec_digest",
                                    "render", "clip", "seed_base", "spec"})
    context_fields = {f.name for f in dataclasses.fields(LockedContext)}
    assert context_fields.isdisjoint({"specs", "prompts", "previous",
                                      "segment_specs", "written"})


def test_a_writer_reaching_for_a_previous_specs_prompt_cannot_find_one():
    def malicious(context, index):
        with pytest.raises(AttributeError):
            _ = context.segments[index - 1].prompt      # no such field
        with pytest.raises(AttributeError):
            _ = context.specs                           # no such collection
        return f"shot {index}"

    _lock, specs = compile_fixture(prompt_writer=malicious)
    assert len(specs) == 3


def test_reaching_backwards_by_index_is_refused_outright():
    def backwards(context, index):
        with pytest.raises(IndexError, match="negative indices are refused"):
            context.brief(-1)
        if index > 0:
            # Reaching back DOES yield the neighbour — but as LOCKED material,
            # which is the required relationship, not the prohibited one.
            earlier = context.brief(index - 1)
            assert isinstance(earlier, LockedSegmentBrief)
            assert "prompt" not in earlier.to_dict()
        return f"shot {index}"

    _lock, specs = compile_fixture(prompt_writer=backwards)
    assert len(specs) == 3


def test_no_segment_spec_exists_anywhere_while_prompts_are_being_written():
    # Two-phase compilation, proven rather than asserted in a docstring: during
    # phase 1 there is no SegmentSpec for this lock alive in the interpreter,
    # so even a writer that walked the heap would find nothing to chain to.
    lock = make_lock()
    alive: list[int] = []

    def counting(context, index):
        gc.collect()
        alive.append(sum(1 for o in gc.get_objects()
                         if type(o) is SegmentSpec
                         and o.lock_digest == context.lock_digest))
        return f"shot {index}"

    specs = compile_segments(lock, snapshot=make_snapshot(),
                             audio_master=make_master(), continuity=make_bible(),
                             shot_plan=make_plan(), tone=0.5,
                             prompt_writer=counting)
    assert alive == [0, 0, 0]
    assert len(specs) == 3


def test_a_writer_that_returns_nothing_usable_is_refused():
    for bad in ("", "   ", None, 42):
        with pytest.raises(CompileRefused, match="non-empty string"):
            compile_fixture(prompt_writer=lambda c, i, _b=bad: _b)


def test_the_default_writer_is_a_deterministic_template():
    lock, specs = compile_fixture()
    context = build_locked_context(
        lock, snapshot=make_snapshot(), audio_master=make_master(),
        continuity=make_bible(), shot_plan=make_plan(), tone=0.35,
        dialogue=DIALOGUE)
    assert default_prompt_writer(context, 0) == specs[0].prompt
    assert default_prompt_writer(context, 0) == default_prompt_writer(context, 0)
    assert "kitchen" in specs[0].prompt
    assert JOINT_MODE_PLAIN["cut"] in specs[0].prompt


def test_the_preface_renders_the_whole_locked_film_with_this_shot_marked():
    lock, _specs = compile_fixture()
    context = build_locked_context(
        lock, snapshot=make_snapshot(), audio_master=make_master(),
        continuity=make_bible(), shot_plan=make_plan(), tone=0.35,
        dialogue=DIALOGUE, beats=("open", "turn", "close"))
    text = context.preface(1)
    assert ">> [1] s2" in text and "   [0] s1" in text     # every row visible
    assert "You came back." in text                       # locked dialogue
    assert JOINT_MODE_PLAIN["cut"] in text                # sentences, not tokens
    assert "DO NOT INCLUDE: gore" in text                 # snapshot exclusions
    assert "beat: turn" in text
    assert "prompt:" not in text                          # nothing written yet


# --- the compiler's own refusals -------------------------------------------


@pytest.mark.parametrize("swap", ["snapshot", "continuity", "audio_master",
                                  "shot_plan"])
def test_compile_refuses_an_artifact_the_lock_never_saw(swap):
    lock = make_lock()
    call = dict(snapshot=make_snapshot(), audio_master=make_master(),
                continuity=make_bible(), shot_plan=make_plan(), tone=0.5)
    call[swap] = {
        "snapshot": lambda: make_snapshot(deliverable="something else"),
        "continuity": lambda: ContinuityBible(entries=(ContinuityState("s1"),
                                                       ContinuityState("s2"),
                                                       ContinuityState("s3"))),
        "audio_master": lambda: dataclasses.replace(make_master(),
                                                    candidates_considered=99),
        "shot_plan": lambda: ShotPlan(entries=(
            ShotPlanEntry("s1", ("l1", "l2", "l3"), 0.0, 6.0, rubric=("x",)),)),
    }[swap]()
    with pytest.raises(CompileRefused, match=swap):
        compile_segments(lock, **call)


def test_compile_refuses_an_identity_the_lock_never_authorized():
    lock = make_lock()
    with pytest.raises(CompileRefused, match="never authorized"):
        compile_segments(lock, snapshot=make_snapshot(),
                         audio_master=make_master(), continuity=make_bible(),
                         shot_plan=make_plan(), tone=0.5,
                         identity_refs=("identity_profile:stranger",))


def test_compile_refuses_a_tone_outside_the_profile_range():
    lock = make_lock()
    with pytest.raises(CompileRefused, match="tone"):
        compile_segments(lock, snapshot=make_snapshot(),
                         audio_master=make_master(), continuity=make_bible(),
                         shot_plan=make_plan(), tone=1.5)


def test_compile_refuses_a_per_segment_argument_that_does_not_line_up():
    lock = make_lock()
    with pytest.raises(CompileRefused, match="unknown segment"):
        compile_fixture(joint_modes={"s9": "cut"})
    with pytest.raises(CompileRefused, match="entries for"):
        compile_fixture(scene_refs=("only one",))
    assert lock.revision == 0


# --- joint modes -----------------------------------------------------------


def test_joint_modes_ride_through_and_index_zero_is_always_a_cut():
    _lock, specs = compile_fixture(
        joint_modes={"s1": "vace_extend", "s2": "still", "s3": "vace_extend"})
    # Segment 0 has no previous shot to carry anything from, so 'cut' wins.
    assert [s.joint_mode for s in specs] == ["cut", "still", "vace_extend"]
    assert [s.needs_previous_frames for s in specs] == [False, True, True]


def test_a_spec_at_index_zero_cannot_be_constructed_with_a_carrying_join():
    _lock, specs = compile_fixture()
    with pytest.raises(ValueError, match="no previous shot"):
        dataclasses.replace(specs[0], joint_mode="still")


def test_render_dependencies_are_reported_but_never_become_plan_edges():
    lock, specs = compile_fixture(joint_modes={"s2": "still",
                                               "s3": "vace_extend"})
    assert render_dependencies(specs) == (("s2", "s1"), ("s3", "s2"))
    graph = to_graph(specs, lock)
    # The frame handoff is an ARTIFACT dependency the orchestrator resolves at
    # execution; it is not prompt lineage, so it never chains the specs.
    assert all(edge.src_node == LOCK_NODE_ID for edge in graph.edges)
    assert sibling_check(graph, graph.segment_node_ids())


# ===========================================================================
# [3] SegmentSpec's own structural refusals
# ===========================================================================


def test_a_spec_without_an_acceptance_rubric_is_refused():
    _lock, specs = compile_fixture()
    with pytest.raises(ValueError, match="no acceptance rubric"):
        dataclasses.replace(specs[0], rubric=())


def test_a_spec_without_parents_is_refused():
    _lock, specs = compile_fixture()
    with pytest.raises(ValueError, match="no parents"):
        dataclasses.replace(specs[0], parents=())


def test_a_spec_carrying_another_segments_continuity_is_refused():
    _lock, specs = compile_fixture()
    with pytest.raises(ValueError, match="how a chain starts"):
        dataclasses.replace(specs[0], continuity=specs[1].continuity)


def test_a_spec_carrying_another_segments_shot_entry_is_refused():
    _lock, specs = compile_fixture()
    with pytest.raises(ValueError, match="shot plan entry of"):
        dataclasses.replace(specs[0], shot=specs[1].shot)


@pytest.mark.parametrize("change", [
    {"tone": 1.4}, {"seed_base": -1}, {"seed_base": 2 ** 32},
    {"joint_mode": "morph"}, {"prompt": "  "}, {"index": -1},
])
def test_spec_fields_are_validated_at_construction(change):
    _lock, specs = compile_fixture()
    with pytest.raises((ValueError, TypeError)):
        dataclasses.replace(specs[1], **change)


# ===========================================================================
# assert_siblings — the check over a compiled set
# ===========================================================================


def test_assert_siblings_catches_a_spec_that_names_a_sibling():
    _lock, specs = compile_fixture()
    chained = dataclasses.replace(specs[1],
                                  parents=specs[1].parents + (specs[0].digest,))
    with pytest.raises(SiblingViolation) as excinfo:
        assert_siblings((specs[0], chained, specs[2]))
    assert excinfo.value.pairs == (("s2", "s1"),)
    assert "S1 -> S2 -> S3" in str(excinfo.value)


def test_assert_siblings_catches_two_productions_spliced_together():
    _lock, specs = compile_fixture()
    other = dataclasses.replace(specs[2], lock_digest="a-different-lock")
    with pytest.raises(SiblingViolation, match="different production locks"):
        assert_siblings((specs[0], specs[1], other))


def test_assert_siblings_catches_a_parent_from_outside_the_lock():
    lock, specs = compile_fixture()
    stray = dataclasses.replace(specs[0],
                                parents=specs[0].parents + ("some-other-digest",))
    with pytest.raises(SiblingViolation, match="not locked artifacts"):
        assert_siblings((stray,), lock=lock)
    assert_siblings((stray,))          # without the lock, only the chain check


def test_assert_siblings_on_an_empty_set_is_a_no_op():
    assert assert_siblings(()) is None


# ===========================================================================
# [4] The emitted PlanGraph
# ===========================================================================


def to_graph(specs, lock=None, **kw) -> PlanGraph:
    from hugpy_oracle.segments import to_plan_graph
    return to_plan_graph(specs, lock, **kw)


def test_the_graph_is_a_locked_parent_fanning_out_to_sibling_tasks():
    lock, specs = compile_fixture()
    graph = to_graph(specs, lock, goal=GOAL)
    assert graph.node_ids == (LOCK_NODE_ID, "segment:s1", "segment:s2",
                              "segment:s3")
    assert graph.goal_digest == goal_digest(GOAL)
    assert graph.roots() == (LOCK_NODE_ID,)
    assert set(graph.leaves()) == {"segment:s1", "segment:s2", "segment:s3"}
    assert len(graph.edges) == 3
    assert all(e.src_node == LOCK_NODE_ID and e.dst_port == "lock"
               for e in graph.edges)
    assert graph.node(LOCK_NODE_ID).kind is NodeKind.GATE
    assert graph.node(LOCK_NODE_ID).capability is None


def test_segment_nodes_are_tagged_and_carry_their_spec():
    lock, specs = compile_fixture()
    graph = to_graph(specs, lock)
    assert graph.segment_node_ids() == ("segment:s1", "segment:s2", "segment:s3")
    node = graph.node(segment_node_id("s2"))
    assert node.params[SEGMENT_PARAM] is True and node.is_segment
    assert node.capability == SEGMENT_CAPABILITY
    assert node.params["spec_digest"] == specs[1].digest
    assert node.params["prompt"] == specs[1].prompt
    assert node.params["seed_base"] == specs[1].seed_base
    assert tuple(node.params["line_ids"]) == ("l2",)
    # Stage 9's rubric rides in as plan-time acceptance tests, so k107 reads
    # the repair-code mapping off the graph.
    assert tuple(t.threshold for t in node.acceptance) == specs[1].rubric


def test_the_lock_node_records_every_locked_digest():
    lock, specs = compile_fixture()
    params = to_graph(specs, lock).node(LOCK_NODE_ID).params
    assert params["lock_digest"] == lock.digest
    assert params["audio_master_digest"] == lock.audio_master_digest
    assert params["shot_plan_digest"] == lock.shot_plan_digest
    assert params["screenplay_digest"] is None
    assert tuple(params["parents"]) == lock.parent_digests


def test_the_graph_can_be_rebuilt_from_the_specs_alone():
    lock, specs = compile_fixture()
    without = to_graph(specs)
    assert without.goal_digest == lock.digest      # pinned to the lock, honestly
    assert without.structure_digest() != to_graph(specs, lock).structure_digest()
    assert without.segment_node_ids() == ("segment:s1", "segment:s2", "segment:s3")


def test_the_emitted_graph_passes_sibling_check():
    lock, specs = compile_fixture()
    graph = to_graph(specs, lock)
    assert sibling_check(graph, graph.segment_node_ids())


def test_the_validator_finds_no_sibling_violation_in_the_emitted_graph():
    lock, specs = compile_fixture()
    report = validate(to_graph(specs, lock, goal=GOAL), {}, GOAL)
    codes = {e.code for e in report.errors}
    assert ErrorCode.SIBLING_VIOLATION not in codes
    # The capability is genuinely absent from an empty catalog view; that is a
    # true finding about this fleet, and not what this test is about.
    assert ErrorCode.UNKNOWN_CAPABILITY in codes


def test_a_hand_built_chained_graph_does_violate():
    # The prohibited relationship, written out by hand so the check is proven
    # to fire rather than merely proven not to.
    def seg(node_id):
        return PlanNode(node_id=node_id, kind=NodeKind.TASK,
                        capability=SEGMENT_CAPABILITY,
                        inputs=(Port("prev", ArtifactKind.VIDEO, required=False),),
                        outputs=(Port("clip", ArtifactKind.VIDEO),),
                        params=FrozenParams({SEGMENT_PARAM: True}))
    chained = PlanGraph(
        graph_id="chained", goal_digest=goal_digest(GOAL),
        nodes=(seg("segment:s1"), seg("segment:s2"), seg("segment:s3")),
        edges=(Edge("segment:s1", "clip", "segment:s2", "prev"),
               Edge("segment:s2", "clip", "segment:s3", "prev")))
    assert not sibling_check(chained, chained.segment_node_ids())
    report = validate(chained, {}, GOAL)
    assert ErrorCode.SIBLING_VIOLATION in {e.code for e in report.errors}


def test_to_plan_graph_refuses_to_emit_a_chained_graph():
    _lock, specs = compile_fixture()
    chained = dataclasses.replace(specs[1],
                                  parents=specs[1].parents + (specs[0].digest,))
    with pytest.raises(SiblingViolation):
        to_graph((specs[0], chained, specs[2]))


def test_to_plan_graph_needs_at_least_one_spec():
    with pytest.raises(CompileRefused, match="at least one"):
        to_graph(())


# ===========================================================================
# [5] Sequential vs parallel — same dependencies either way
# ===========================================================================


def test_sequential_and_parallel_read_the_same_graph():
    lock, specs = compile_fixture()
    graph_seq = to_graph(specs, lock)
    graph_par = to_graph(specs, lock)
    sequential = execution_order(specs, "sequential", graph=graph_seq)
    parallel = execution_order(specs, "parallel", graph=graph_par)
    # Doc Stage 14: "Segment execution may be sequential or parallel without
    # changing this dependency structure."
    assert set(graph_seq.edges) == set(graph_par.edges)
    assert graph_seq.structure_digest() == graph_par.structure_digest()
    assert sequential != parallel
    assert {n for batch in sequential for n in batch} == \
           {n for batch in parallel for n in batch}


def test_sequential_is_one_node_per_batch_in_topological_order():
    lock, specs = compile_fixture()
    assert execution_order(specs, "sequential", lock=lock) == (
        (LOCK_NODE_ID,), ("segment:s1",), ("segment:s2",), ("segment:s3",))


def test_parallel_is_one_batch_per_dependency_level():
    lock, specs = compile_fixture()
    assert execution_order(specs, "parallel", lock=lock) == (
        (LOCK_NODE_ID,), ("segment:s1", "segment:s2", "segment:s3"))


def test_the_lock_is_always_the_first_thing_in_either_order():
    lock, specs = compile_fixture()
    for mode in ("sequential", "parallel"):
        assert execution_order(specs, mode, lock=lock)[0] == (LOCK_NODE_ID,)


def test_execution_order_refuses_an_unknown_mode():
    lock, specs = compile_fixture()
    with pytest.raises(ValueError, match="sequential"):
        execution_order(specs, "whenever", lock=lock)


# ===========================================================================
# Vocabulary sync — the ONE test here that imports the studio side.
# ===========================================================================


def test_joint_modes_mirror_the_studio_spread():
    """Mirrored, not imported (``video_intel`` builds the model registry at
    import time). Drift is therefore a failing test, not a stale comment."""
    from hugpy_video.intel.prompt_spread import VALID_JOINT_MODES
    assert set(JOINT_MODES) == set(VALID_JOINT_MODES)
    assert set(JOINT_MODE_PLAIN) == set(VALID_JOINT_MODES)


def test_the_segment_capability_is_a_real_catalog_name():
    from hugpy_oracle import catalog
    assert SEGMENT_CAPABILITY in set(catalog.STUDIO_CAPABILITY_NAME.values())
