"""k104 — GenerationSnapshot, the continuity/shot-plan shells, and the Stage 11
production-lock transition.

Everything here is offline and deterministic: these are contracts, not an
engine, so no catalog, no workers, no GPU, no clock. The one test that imports
the studio side is the vocabulary keep-in-sync test at the bottom, and it says
so.

Locks:
  [1] Stage 4: a snapshot carries only prompts that existed BEFORE the run, and
      a RunPromptLedger refuses one minted during it — by digest, not by
      convention (invariant 9).
  [2] wire shape: every artifact round-trips through to_dict/from_dict without
      loss and digests are stable, order-independent and clock-free.
  [3] structural refusals at construction: an unknown camera key, a shot that
      ends before it starts, a shot list out of timeline order, two continuity
      states for one segment.
  [4] Stage 11: lock() validates what it locks — an unlocked AudioMaster, a
      dropped line, an overlapping or overrunning window, a missing continuity
      state, an identity the snapshot never carried, two registry versions.
  [5] Stage 10: a post-lock material change is revision N+1 with a recorded
      reason, never an edit.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_production.py -q
"""
from __future__ import annotations

import dataclasses
import logging
import os
import sys

import pytest

logging.disable(logging.INFO)  # silence the models_config registry chatter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_oracle.audio_master import AudioMaster, LineTiming
from hugpy_oracle.production import (
    CAMERA_MOVES,
    CAMERA_VIEWS,
    ContinuityBible,
    ContinuityState,
    GenerationSnapshot,
    LockRefused,
    ProductionLock,
    RunPromptLedger,
    RunPromptRefused,
    SHOT_SIZES,
    ShotPlan,
    ShotPlanEntry,
    prompt_digest,
)


# ===========================================================================
# Fixtures — a three-line scene, built from k102's dataclasses directly.
# No synthesis, no ASR, no fan-out: this task consumes an AudioMaster, it does
# not produce one, so the fixture is the artifact and nothing else.
# ===========================================================================


def make_master(*, locked: bool = True, registry_version=None) -> AudioMaster:
    timings = (
        LineTiming("l1", 0.0, 2.0, pause_after_s=0.5),
        LineTiming("l2", 2.5, 5.0, pause_after_s=0.25),
        LineTiming("l3", 5.25, 6.0, pause_after_s=0.0),
    )
    master = AudioMaster(timeline_digest="timeline-abc",
                         line_timings=timings,
                         tracks=(("l1", "/a/1.wav"), ("l2", "/a/2.wav"),
                                 ("l3", "/a/3.wav")),
                         total_seconds=6.0, candidates_considered=9,
                         registry_version=registry_version)
    return master.lock() if locked else master


def make_plan() -> ShotPlan:
    return ShotPlan(entries=(
        ShotPlanEntry("s1", ("l1",), 0.0, 2.5,
                      camera={"view": "front", "shot_size": "medium"},
                      blocking="ana at the sink", lighting="window key",
                      rubric=("ana is recognizable", "the coat is on")),
        ShotPlanEntry("s2", ("l2",), 2.5, 5.25,
                      camera={"view": "right-profile", "movement": "dolly_in"},
                      rubric=("bo enters frame left",)),
        ShotPlanEntry("s3", ("l3",), 5.25, 6.0,
                      camera={"shot_size": "close"},
                      rubric=("the coat is off",)),
    ))


def make_bible() -> ContinuityBible:
    return ContinuityBible(
        entries=(ContinuityState("s1", {"coat": "on"}, {"coat": "on"}),
                 ContinuityState("s2", {"coat": "on"}, {"coat": "off"}),
                 ContinuityState("s3", {"coat": "off"}, {"coat": "off"})),
        characters=("ana", "bo"), wardrobe=("wool coat",), props=("mug",),
        locations=("kitchen",), notes="late afternoon, overcast")


def make_snapshot(**kw) -> GenerationSnapshot:
    base = dict(raw_request_ref="mct:request/42",
                prompts_before_run=("two characters, three lines",),
                operator_refs=("file:/refs/kitchen.jpg",),
                identity_refs=("identity_profile:ana", "identity_profile:bo"),
                voice_refs=("voice_profile:ana",),
                deliverable="a 15 second scene", exclusions=("gore",),
                created_at="2026-08-20T10:00:00Z")
    base.update(kw)
    return GenerationSnapshot(**base)


def make_lock(**kw) -> ProductionLock:
    return ProductionLock.lock(make_snapshot(), audio_master=make_master(),
                               continuity=make_bible(), shot_plan=make_plan(),
                               locked_at="2026-08-20T10:05:00Z", **kw)


# ===========================================================================
# [1] Stage 4 — the immutable generation snapshot
# ===========================================================================


def test_snapshot_requires_a_raw_request_ref():
    with pytest.raises(ValueError, match="raw_request_ref"):
        GenerationSnapshot(raw_request_ref="  ", deliverable="a clip")


def test_snapshot_requires_a_deliverable():
    # Stage 4 lists the requested deliverable as snapshot content; without it
    # the snapshot cannot gate anything downstream.
    with pytest.raises(ValueError, match="deliverable"):
        GenerationSnapshot(raw_request_ref="mct:request/1")


def test_snapshot_round_trips_and_digests_are_stable():
    snap = make_snapshot()
    again = GenerationSnapshot.from_dict(snap.to_dict())
    assert again == snap
    assert again.digest == snap.digest
    # No clock inside: the same facts twice are the same artifact.
    assert make_snapshot().digest == snap.digest


def test_snapshot_refs_are_deduplicated_in_field_order():
    snap = make_snapshot(operator_refs=("a", "b", "a"))
    assert snap.operator_refs == ("a", "b")
    assert snap.refs[:2] == ("a", "b")
    assert "identity_profile:ana" in snap.refs


def test_snapshot_refuses_a_bare_string_where_a_sequence_belongs():
    with pytest.raises(TypeError, match="missing-brackets"):
        make_snapshot(operator_refs="file:/one.jpg")


def test_snapshot_refuses_an_empty_reference_entry():
    with pytest.raises(ValueError, match="empty entry"):
        make_snapshot(acquisition_refs=("ok", "   "))


def test_prompt_digests_are_namespaced_and_stable():
    snap = make_snapshot(prompts_before_run=("alpha", "beta"))
    assert snap.prompt_digests == (prompt_digest("alpha"), prompt_digest("beta"))
    # Namespaced under a {"prompt": …} envelope, so a prompt digest can never
    # be confused with a raw sha256 of the same text.
    import hashlib
    assert prompt_digest("alpha") != hashlib.sha256(b"alpha").hexdigest()


def test_assert_not_run_prompt_passes_an_unknown_digest():
    digest = prompt_digest("written before the run")
    assert GenerationSnapshot.assert_not_run_prompt(digest, ()) == digest


def test_assert_not_run_prompt_refuses_a_run_minted_digest():
    digest = prompt_digest("minted mid-run")
    with pytest.raises(RunPromptRefused) as excinfo:
        GenerationSnapshot.assert_not_run_prompt(digest, {digest})
    assert excinfo.value.prompt_digest == digest
    assert "invariant 9" in str(excinfo.value)


def test_ledger_records_prompts_and_reads_back_by_text_or_digest():
    ledger = RunPromptLedger()
    digest = ledger.record("a segment prompt written during the run")
    assert len(ledger) == 1
    assert "a segment prompt written during the run" in ledger
    assert digest in ledger                      # 64-hex reads as a digest
    assert "something else entirely" not in ledger
    assert ledger.digests == (digest,)
    assert tuple(ledger) == (digest,)


def test_ledger_seeds_from_prompts_and_digests_and_records_in_bulk():
    ledger = RunPromptLedger(prompts=("one",), digests=(prompt_digest("two"),))
    assert len(ledger) == 2
    ledger.record_all(("three", "four"))
    assert len(ledger) == 4
    assert ledger.digests == tuple(sorted(ledger.digests))   # sorted view


def test_ledger_refuses_and_asserts_admissibility():
    ledger = RunPromptLedger(prompts=("mid-run",))
    assert ledger.refuses("mid-run")
    assert not ledger.refuses("pre-run")
    assert ledger.assert_admissible("pre-run") == prompt_digest("pre-run")
    with pytest.raises(RunPromptRefused):
        ledger.assert_admissible("mid-run")


def test_with_prompt_refuses_a_run_minted_prompt_and_never_mutates():
    snap = make_snapshot()
    ledger = RunPromptLedger()
    ledger.record("the writer's output for segment 2")
    with pytest.raises(RunPromptRefused):
        snap.with_prompt("the writer's output for segment 2", ledger=ledger)
    # A pre-run prompt is accepted and produces a NEW snapshot (invariant 4).
    grown = snap.with_prompt("also asked for before the run", ledger=ledger)
    assert grown is not snap
    assert snap.prompts_before_run == ("two characters, three lines",)
    assert len(grown.prompts_before_run) == 2
    assert grown.digest != snap.digest


def test_assert_pre_run_catches_a_snapshot_that_grew_during_the_run():
    ledger = RunPromptLedger(prompts=("segment 2's prompt",))
    clean = make_snapshot()
    assert clean.assert_pre_run(ledger) is clean
    dirty = dataclasses.replace(
        clean, prompts_before_run=clean.prompts_before_run + ("segment 2's prompt",))
    with pytest.raises(RunPromptRefused):
        dirty.assert_pre_run(ledger)
    # A plain iterable of digests works too — the ledger is a convenience, not
    # a required type.
    with pytest.raises(RunPromptRefused):
        dirty.assert_pre_run([prompt_digest("segment 2's prompt")])


# ===========================================================================
# [2]/[3] Stage 7 — continuity
# ===========================================================================


def test_continuity_state_freezes_its_mappings_and_stays_hashable():
    state = ContinuityState("s1", {"coat": "on", "props": ["mug"]},
                            {"coat": "off", "props": ["mug"]})
    assert hash(state)                      # frozen + FrozenParams => hashable
    assert state.state_before["props"] == ("mug",)
    with pytest.raises(TypeError):
        state.state_before["coat"] = "off"  # type: ignore[index]


def test_continuity_state_reports_what_changed():
    state = ContinuityState("s2", {"coat": "on", "hour": 17},
                            {"coat": "off", "hour": 17, "mug": "empty"})
    assert state.changed_keys == ("coat", "mug")


def test_continuity_state_round_trips():
    state = ContinuityState("s1", {"a": 1}, {"a": 2})
    assert ContinuityState.from_dict(state.to_dict()) == state


def test_bible_refuses_two_states_for_one_segment():
    with pytest.raises(ValueError, match="two continuity states"):
        ContinuityBible(entries=(ContinuityState("s1"), ContinuityState("s1")))


def test_bible_reports_missing_segments_in_the_order_asked():
    bible = make_bible()
    assert bible.missing(("s1", "s2", "s3")) == ()
    assert bible.missing(("s9", "s1", "s7")) == ("s9", "s7")
    assert bible.state("s2").changed_keys == ("coat",)
    with pytest.raises(KeyError):
        bible.state("nope")


def test_bible_round_trips_and_digests():
    bible = make_bible()
    assert ContinuityBible.from_dict(bible.to_dict()) == bible
    assert ContinuityBible.from_dict(bible.to_dict()).digest == bible.digest


# ===========================================================================
# [3] Stage 9 — the shot plan
# ===========================================================================


def test_shot_entry_refuses_an_unknown_camera_key():
    # A typo'd direction that silently vanished is worse than one that refuses.
    with pytest.raises(ValueError, match="unknown key"):
        ShotPlanEntry("s1", camera={"lense_mm": 35}, rubric=("x",))


@pytest.mark.parametrize("camera", [
    {"view": "over-the-moon"},
    {"shot_size": "enormous"},
    {"movement": "teleport"},
])
def test_shot_entry_refuses_a_value_outside_a_closed_vocabulary(camera):
    with pytest.raises(ValueError, match="vocabulary"):
        ShotPlanEntry("s1", camera=camera, rubric=("x",))


def test_shot_entry_accepts_the_real_vocabularies():
    entry = ShotPlanEntry("s1", camera={"view": sorted(CAMERA_VIEWS)[0],
                                        "shot_size": sorted(SHOT_SIZES)[0],
                                        "movement": sorted(CAMERA_MOVES)[0],
                                        "lens_mm": 35.0, "eyeline": "to camera"},
                          rubric=("x",))
    assert entry.camera["lens_mm"] == 35.0


@pytest.mark.parametrize("lens", [0, -1, "35mm", True])
def test_shot_entry_refuses_a_nonsense_lens(lens):
    with pytest.raises(ValueError, match="lens_mm"):
        ShotPlanEntry("s1", camera={"lens_mm": lens}, rubric=("x",))


def test_shot_entry_refuses_a_window_that_ends_before_it_starts():
    with pytest.raises(ValueError, match="ends before it starts"):
        ShotPlanEntry("s1", start_s=4.0, end_s=1.0)


def test_shot_entry_exposes_its_window_and_duration():
    entry = make_plan().entry("s2")
    assert entry.duration_s == 2.75
    assert entry.window == (2.5, 5.25, ("l2",))


def test_shot_plan_refuses_duplicate_segments_and_out_of_order_entries():
    with pytest.raises(ValueError, match="two entries"):
        ShotPlan(entries=(ShotPlanEntry("s1"), ShotPlanEntry("s1")))
    with pytest.raises(ValueError, match="out of timeline order"):
        ShotPlan(entries=(ShotPlanEntry("s1", start_s=5.0, end_s=6.0),
                          ShotPlanEntry("s2", start_s=1.0, end_s=2.0)))


def test_shot_plan_finds_overlaps_but_does_not_refuse_them():
    # Two angles on one line is a legitimate artifact; only the LOCK refuses it.
    plan = ShotPlan(entries=(ShotPlanEntry("s1", start_s=0.0, end_s=3.0),
                             ShotPlanEntry("s2", start_s=2.0, end_s=4.0)))
    assert plan.overlaps() == (("s1", "s2"),)
    assert make_plan().overlaps() == ()


def test_shot_plan_reading_and_round_trip():
    plan = make_plan()
    assert plan.segment_ids == ("s1", "s2", "s3")
    assert plan.line_ids == ("l1", "l2", "l3")
    assert plan.total_seconds == 6.0
    assert ShotPlan.from_dict(plan.to_dict()) == plan


# ===========================================================================
# [4] Stage 11 — the lock transition
# ===========================================================================


def test_lock_requires_a_locked_audio_master():
    with pytest.raises(LockRefused, match="audio master is not locked"):
        ProductionLock.lock(make_snapshot(),
                            audio_master=make_master(locked=False),
                            continuity=make_bible(), shot_plan=make_plan())


def test_lock_records_the_digests_of_exactly_what_it_locked():
    snap, master, bible, plan = (make_snapshot(), make_master(), make_bible(),
                                 make_plan())
    lock = ProductionLock.lock(snap, audio_master=master, continuity=bible,
                               shot_plan=plan)
    assert lock.snapshot_digest == snap.digest
    assert lock.audio_master_digest == master.digest
    assert lock.continuity_digest == bible.digest
    assert lock.shot_plan_digest == plan.digest
    assert lock.screenplay_digest is None       # k110's artifact, honestly absent
    assert lock.revision == 0 and lock.parent_revision is None


def test_lock_is_clock_free_so_the_same_production_locks_to_one_digest():
    assert make_lock().digest == make_lock().digest


def test_parent_digests_are_the_lock_and_what_it_locked_only():
    lock = make_lock()
    assert lock.parent_digests[0] == lock.digest
    assert set(lock.parent_digests) == {
        lock.digest, lock.snapshot_digest, lock.continuity_digest,
        lock.audio_master_digest, lock.shot_plan_digest}
    with_script = make_lock(screenplay_digest="screenplay-1")
    assert "screenplay-1" in with_script.parent_digests
    assert len(with_script.parent_digests) == 6
    # Deduplicated, so a degenerate lock cannot inflate a spec's lineage.
    degenerate = ProductionLock(snapshot_digest="d", screenplay_digest=None,
                                continuity_digest="d", audio_master_digest="d",
                                shot_plan_digest="d")
    assert degenerate.parent_digests == (degenerate.digest, "d")


def test_lock_refuses_a_shot_naming_a_line_the_master_does_not_have():
    plan = ShotPlan(entries=(ShotPlanEntry("s1", ("l1", "l9"), 0.0, 6.0,
                                           rubric=("x",)),))
    with pytest.raises(LockRefused, match="stale"):
        ProductionLock.lock(make_snapshot(), audio_master=make_master(),
                            continuity=ContinuityBible(
                                entries=(ContinuityState("s1"),)),
                            shot_plan=plan)


def test_lock_refuses_a_spoken_line_no_shot_covers():
    plan = ShotPlan(entries=(ShotPlanEntry("s1", ("l1",), 0.0, 6.0,
                                           rubric=("x",)),))
    with pytest.raises(LockRefused, match="no picture behind it"):
        ProductionLock.lock(make_snapshot(), audio_master=make_master(),
                            continuity=ContinuityBible(
                                entries=(ContinuityState("s1"),)),
                            shot_plan=plan)


def test_lock_refuses_overlapping_shot_windows():
    plan = ShotPlan(entries=(
        ShotPlanEntry("s1", ("l1", "l2"), 0.0, 5.25, rubric=("x",)),
        ShotPlanEntry("s2", ("l3",), 4.0, 6.0, rubric=("x",))))
    bible = ContinuityBible(entries=(ContinuityState("s1"), ContinuityState("s2")))
    with pytest.raises(LockRefused, match="overlap"):
        ProductionLock.lock(make_snapshot(), audio_master=make_master(),
                            continuity=bible, shot_plan=plan)


def test_lock_refuses_a_shot_running_past_the_end_of_the_audio():
    plan = ShotPlan(entries=(ShotPlanEntry("s1", ("l1", "l2", "l3"), 0.0, 9.0,
                                           rubric=("x",)),))
    with pytest.raises(LockRefused, match="end after the audio master"):
        ProductionLock.lock(make_snapshot(), audio_master=make_master(),
                            continuity=ContinuityBible(
                                entries=(ContinuityState("s1"),)),
                            shot_plan=plan)


def test_lock_refuses_a_segment_with_no_continuity_state():
    thin = ContinuityBible(entries=(ContinuityState("s1"), ContinuityState("s2")))
    with pytest.raises(LockRefused, match=r"\['s3'\]"):
        ProductionLock.lock(make_snapshot(), audio_master=make_master(),
                            continuity=thin, shot_plan=make_plan())


def test_lock_refuses_an_identity_the_snapshot_never_carried():
    with pytest.raises(LockRefused, match="not in the generation snapshot"):
        make_lock(identity_refs=("identity_profile:stranger",))


def test_lock_defaults_identity_refs_to_the_snapshot_and_accepts_a_subset():
    assert make_lock().identity_refs == ("identity_profile:ana",
                                         "identity_profile:bo")
    assert make_lock(identity_refs=("identity_profile:bo",)).identity_refs == \
        ("identity_profile:bo",)


def test_lock_adopts_the_audio_masters_registry_version_when_it_has_none():
    lock = ProductionLock.lock(make_snapshot(), continuity=make_bible(),
                               shot_plan=make_plan(),
                               audio_master=make_master(registry_version="reg-7"))
    assert lock.registry_version == "reg-7"


def test_lock_refuses_two_disagreeing_registry_versions():
    with pytest.raises(LockRefused, match="two routing-registry versions"):
        ProductionLock.lock(make_snapshot(registry_version="reg-1"),
                            continuity=make_bible(), shot_plan=make_plan(),
                            audio_master=make_master(registry_version="reg-7"))


def test_lock_refuses_a_snapshot_prompt_minted_during_the_run():
    ledger = RunPromptLedger(prompts=("two characters, three lines",))
    with pytest.raises(RunPromptRefused):
        ProductionLock.lock(make_snapshot(), audio_master=make_master(),
                            continuity=make_bible(), shot_plan=make_plan(),
                            run_prompts=ledger)


@pytest.mark.parametrize("kwargs", [
    {"audio_master": object()},
    {"continuity": object()},
    {"shot_plan": object()},
])
def test_lock_refuses_the_wrong_types_at_the_boundary(kwargs):
    call = dict(audio_master=make_master(), continuity=make_bible(),
                shot_plan=make_plan())
    call.update(kwargs)
    with pytest.raises(TypeError):
        ProductionLock.lock(make_snapshot(), **call)


def test_lock_refuses_an_empty_shot_plan():
    with pytest.raises(LockRefused, match="shot plan is empty"):
        ProductionLock.lock(make_snapshot(), audio_master=make_master(),
                            continuity=make_bible(), shot_plan=ShotPlan())


# ===========================================================================
# [5] Stage 10 — post-lock revisions
# ===========================================================================


def test_revise_bumps_the_revision_and_records_its_parent_and_reason():
    lock = make_lock()
    revised = lock.revise("the dolly path collides with the counter")
    assert revised.revision == 1
    assert revised.parent_revision == 0
    assert revised.revision_reason == "the dolly path collides with the counter"
    assert revised.digest != lock.digest
    assert lock.revision == 0                 # the original is untouched
    twice = revised.revise("re-lit after the camera moved")
    assert (twice.revision, twice.parent_revision) == (2, 1)


def test_revise_can_swap_a_locked_artifact_digest():
    lock = make_lock()
    revised = lock.revise("shot plan reblocked", shot_plan_digest="new-plan")
    assert revised.shot_plan_digest == "new-plan"
    assert revised.snapshot_digest == lock.snapshot_digest
    assert "new-plan" in revised.parent_digests


def test_revise_refuses_an_unexplained_change():
    with pytest.raises(LockRefused, match="needs a reason"):
        make_lock().revise("   ")


def test_revise_refuses_a_field_that_is_not_revisable():
    with pytest.raises(LockRefused, match="cannot change"):
        make_lock().revise("why", revision=9)


def test_a_revision_without_a_reason_cannot_even_be_constructed():
    lock = make_lock()
    with pytest.raises(ValueError, match="unaudited production change"):
        dataclasses.replace(lock, revision=1, parent_revision=0)


def test_lock_refuses_a_parent_revision_that_does_not_precede_it():
    with pytest.raises(ValueError, match="must precede"):
        dataclasses.replace(make_lock(), revision=1, parent_revision=1,
                            revision_reason="why")


def test_lock_round_trips_including_its_revision_history():
    revised = make_lock().revise("re-timed after the audio repair")
    assert ProductionLock.from_dict(revised.to_dict()) == revised
    assert ProductionLock.from_dict(revised.to_dict()).digest == revised.digest


# ===========================================================================
# Vocabulary sync — the ONE test here that imports the studio side.
# ===========================================================================


def test_camera_views_mirror_the_identity_bank():
    """``CAMERA_VIEWS`` is mirrored, not imported, because importing
    ``video_intel`` builds the model registry at import time. Mirrored means
    drift is possible, so the drift is a failing test rather than a comment."""
    from hugpy_video.intel.identity_profiles import SEMANTIC_VIEWS
    assert CAMERA_VIEWS == frozenset(SEMANTIC_VIEWS)


def test_camera_view_from_prompt_wraps_the_deterministic_shot_intent_pass():
    from hugpy_oracle.production import camera_view_from_prompt
    assert camera_view_from_prompt("she walks away down the hall") == "back"
    assert camera_view_from_prompt("a left profile shot") == "left-profile"
    assert camera_view_from_prompt("they talk") is None
    assert camera_view_from_prompt("") is None
