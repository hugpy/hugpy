"""k114 — the script-first RUN manager: persistence, lifecycle, invariant 9.

Offline and deterministic. The run root is a pytest ``tmp_path``, the LLM is a
function returning canned JSON, the fleet/route/catalog reads are injected
lambdas and the segment dispatch is a recording stub — so nothing here touches
a catalog, a registry, a worker, a GPU, a network or a clock it does not own.

Locks:
  [1] lifecycle: create -> author/edit -> derive -> lock -> compile -> attempt
      -> promote, each step persisted and reloadable by run id alone.
  [2] doc Stage 4: the snapshot carries only prompts persisted BEFORE the run
      started; one persisted after is VISIBLY excluded, with its reason, and
      never reaches ``prompts_before_run``.
  [3] the 422 paths: an authoring gap is never a coerced artifact, and an
      operator edit is held to the same constructor with EVERY error reported.
  [4] doc Stage 11/10: a lock refuses what k104/k110 refuse (and refuses at all
      without an AudioMaster, honestly, rather than inventing one); after the
      lock an edit is refused and only ``revise`` with a reason moves.
  [5] doc Stage 14: every segment names the SAME lock-side parents and no
      segment names a sibling; regenerating one segment leaves every sibling's
      digest untouched and reuses the FROZEN spec at a new seed.
  [6] the promotion rule: a promoted output can seed a NEW run and is refused
      re-entry into its own — by digest, inside k104's own code.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_script_first.py -q
"""
from __future__ import annotations

import json
import logging
import os
import sys

import pytest

logging.disable(logging.INFO)  # silence the models_config registry chatter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_oracle import script_first as sf
from hugpy_oracle.audio_master import AudioMaster, Line, LineTiming
from hugpy_oracle.production import prompt_digest
from hugpy_oracle.screenplay import AuthoringGap, PlotSpec, Screenplay, Scene


# ---------------------------------------------------------------------------
# Fixtures — everything injected, nothing live
# ---------------------------------------------------------------------------

REGISTRY = "sha256:testregistry00"

FLEET = {
    "registry_version": REGISTRY,
    "capabilities": [
        {"name": "text.chat", "eligible": True, "model_ids": ["fixture-llm"],
         "reasons": []},
        {"name": "audio.tts", "eligible": False, "model_ids": [],
         "reasons": ["no worker seats text-to-speech"]},
        {"name": "image.generate", "eligible": True,
         "model_ids": ["sd-turbo"], "reasons": []},
    ],
    "hardware": {"workers": [{"name": "fixture-box", "gpus": []}]},
}

ROUTE = {"capability": "text.chat", "execution": "execute",
         "model_id": "fixture-llm", "model_rationale": "only-eligible",
         "reasons": []}


def fleet():
    return dict(FLEET)


def route():
    return dict(ROUTE)


def catalog_view():
    """An EMPTY catalog view. The validator reports UNKNOWN_CAPABILITY against
    it, which is a TRUE finding about a fleet with no catalog — these tests
    assert the report is carried verbatim, never that it is clean."""
    return {}


def make_run(tmp_path, **kwargs):
    kwargs.setdefault("deliverable", "a two-shot short")
    kwargs.setdefault("requirements", "Ana leaves the house at dusk.")
    kwargs.setdefault("sources", [
        {"prompt_id": "p1", "text": "a woman leaves a house",
         "persisted_at": "2020-01-01T00:00:00+00:00"}])
    return sf.ScriptFirstRun.create(root=str(tmp_path), fleet=fleet,
                                    route=route, **kwargs)


def screenplay_obj(*, dialogue=True):
    line = Line(line_id="l1", speaker="ANA", text="I am leaving.")
    s1 = Scene(scene_id="s1", heading="INT. KITCHEN - DAY", location="KITCHEN",
               time_of_day="DAY", action="Ana stares at the kettle.",
               present_at_open=("ANA",), props=("kettle",),
               dialogue=((line,) if dialogue else ()))
    s2 = Scene(scene_id="s2", heading="EXT. STREET - NIGHT", location="STREET",
               time_of_day="NIGHT", action="Ana walks away from the house.",
               present_at_open=("ANA",), story_time_s=600.0)
    return Screenplay(title="Kettle", scenes=(s1, s2),
                      logline="A woman leaves.")


def master_for(play):
    timeline = play.to_dialogue_timeline(locked=True)
    timings = tuple(LineTiming(line_id=l.line_id, start_s=float(i) * 3.0,
                               end_s=float(i) * 3.0 + 2.0, pause_after_s=0.5)
                    for i, l in enumerate(play.lines))
    tracks = tuple((l.line_id, f"/fixtures/{l.line_id}.wav") for l in play.lines)
    return AudioMaster(timeline_digest=timeline.digest, line_timings=timings,
                       tracks=tracks, total_seconds=12.0,
                       candidates_considered=1, locked=True)


def locked_run(tmp_path, **kwargs):
    run = make_run(tmp_path, **kwargs)
    play = screenplay_obj()
    run.put_artifact("screenplay", play.to_dict())
    run.lock_run(audio_master=master_for(play).to_dict())
    return run


def recording_dispatch(calls):
    def dispatch(spec, *, kind, seed, settings=None):
        calls.append({"segment_id": spec.segment_id, "kind": kind,
                      "seed": seed, "prompt": spec.prompt,
                      "spec_digest": spec.digest})
        return {"ok": True, "kind": kind, "capability": "image.generate",
                "model_id": "sd-turbo", "seed": seed, "prompt": spec.prompt,
                "params": {"seed": seed, "steps": 4},
                "artifacts": [{"kind": "image", "uri": f"/out/{seed}.png"}],
                "receipt": {"model_id": "sd-turbo", "duration_s": 1.0},
                "gap": None}
    return dispatch


def plot_llm(reply):
    def llm(prompt):
        return reply
    return llm


VALID_PLOT = {
    "premise": "A woman decides to leave the house she grew up in.",
    "summary": "Ana packs, hesitates at the kettle, and walks out at dusk.",
    "beginning": "Ana stands in the kitchen she has always known.",
    "middle": "The kettle boils; she does not pour it.",
    "ending": "Ana walks down the street without looking back.",
    "characters": [{"name": "ANA", "goal": "leave", "conflict": "attachment",
                    "arc": "hesitation to resolve",
                    "description": "a woman in her thirties"}],
    "beats": [
        {"beat_id": "b1", "summary": "Ana stands in the kitchen",
         "characters": ["ANA"], "location": "KITCHEN", "time_of_day": "DAY"},
        {"beat_id": "b2", "summary": "Ana walks out", "characters": ["ANA"],
         "causes": ["b1"], "turning_point": True, "location": "STREET",
         "time_of_day": "NIGHT"},
    ],
    "tone": "quiet", "genre": "drama", "pacing": "slow",
    "world_rules": ["nobody follows her"],
}


# ---------------------------------------------------------------------------
# [1] lifecycle + persistence
# ---------------------------------------------------------------------------


def test_create_persists_a_run_readable_by_id_alone(tmp_path):
    run = make_run(tmp_path)
    again = sf.ScriptFirstRun.load(run.run_id, str(tmp_path))
    assert again.state["snapshot_digest"] == run.state["snapshot_digest"]
    assert os.path.isfile(sf.state_path(run.run_id, str(tmp_path)))


def test_run_dir_mirrors_the_performance_run_convention(tmp_path):
    run = make_run(tmp_path)
    assert sf.run_dir(run.run_id, str(tmp_path)).endswith(
        os.path.join("runs", "script_first", run.run_id))
    assert sf.state_path(run.run_id, str(tmp_path)).endswith("state.json")


def test_unknown_run_is_a_404_shaped_refusal(tmp_path):
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        sf.ScriptFirstRun.load("sf-nope", str(tmp_path))
    assert excinfo.value.code == "RUN_NOT_FOUND"
    assert excinfo.value.http_status == 404


def test_a_journal_at_another_state_version_is_not_read(tmp_path):
    run = make_run(tmp_path)
    path = sf.state_path(run.run_id, str(tmp_path))
    payload = json.load(open(path))
    payload["version"] = sf.STATE_VERSION + 7
    json.dump(payload, open(path, "w"))
    with pytest.raises(sf.ScriptFirstRefused):
        sf.ScriptFirstRun.load(run.run_id, str(tmp_path))


def test_list_runs_is_newest_first_and_summarised(tmp_path):
    a = make_run(tmp_path, created_at="2026-01-01T00:00:00+00:00")
    b = make_run(tmp_path, created_at="2026-06-01T00:00:00+00:00")
    ids = [r["run_id"] for r in sf.ScriptFirstRun.list_runs(str(tmp_path))]
    assert ids.index(b.run_id) < ids.index(a.run_id)
    summary = sf.ScriptFirstRun.list_runs(str(tmp_path))[0]
    assert set(("run_id", "locked", "registry_version", "segments")) <= set(summary)


def test_the_run_state_carries_the_frozen_model_and_hardware_view(tmp_path):
    run = make_run(tmp_path)
    models = run.state["models"]
    assert models["fleet"]["registry_version"] == REGISTRY
    assert models["authoring_route"]["model_id"] == "fixture-llm"
    assert models["authoring_route"]["model_rationale"] == "only-eligible"
    assert run.registry_version == REGISTRY


def test_the_snapshot_carries_the_registry_version_it_was_taken_under(tmp_path):
    assert make_run(tmp_path).snapshot.registry_version == REGISTRY


def test_limitations_are_never_empty(tmp_path):
    assert make_run(tmp_path).limitations()


# ---------------------------------------------------------------------------
# [2] doc Stage 4 — the immutable snapshot
# ---------------------------------------------------------------------------


def test_a_prompt_persisted_after_the_run_started_is_excluded_and_visible(tmp_path):
    run = make_run(tmp_path, created_at="2026-08-21T00:00:00+00:00", sources=[
        {"prompt_id": "before", "text": "a woman leaves a house",
         "persisted_at": "2026-08-20T00:00:00+00:00"},
        {"prompt_id": "after", "text": "minted after the run began",
         "persisted_at": "2026-08-21T00:00:01+00:00"}])
    assert run.snapshot.prompts_before_run == ("a woman leaves a house",)
    rows = {r["prompt_id"]: r for r in run.state["sources"]}
    assert rows["after"]["included"] is False
    assert "AFTER this run started" in rows["after"]["exclusion_reason"]
    # visible, not dropped: the excluded source is still in the run
    assert rows["after"]["text"] == "minted after the run began"


def test_a_source_whose_claimed_hash_disagrees_is_refused_before_any_snapshot(tmp_path):
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        make_run(tmp_path, sources=[
            {"prompt_id": "p1", "text": "a woman leaves a house",
             "hash": "0" * 64}])
    assert excinfo.value.code == "SOURCE_DIGEST_MISMATCH"
    assert excinfo.value.http_status == 422
    assert not sf.ScriptFirstRun.list_runs(str(tmp_path))


def test_a_matching_claimed_hash_is_accepted(tmp_path):
    text = "a woman leaves a house"
    run = make_run(tmp_path, sources=[{"prompt_id": "p1", "text": text,
                                       "hash": prompt_digest(text)}])
    assert run.snapshot.prompts_before_run == (text,)


def test_every_source_problem_is_reported_at_once(tmp_path):
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        make_run(tmp_path, sources=[{"prompt_id": "a", "text": ""},
                                    {"prompt_id": "b", "text": "  "}])
    assert len(excinfo.value.errors) == 2


def test_a_deliverable_is_required_because_a_snapshot_gates_nothing_without_one(tmp_path):
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        make_run(tmp_path, deliverable="")
    assert excinfo.value.code == "SOURCE_INVALID"


def test_references_reach_the_snapshot_under_their_own_kinds(tmp_path):
    run = make_run(tmp_path, references={
        "operator": ["upload:kitchen.png"], "identity": ["identity_profile:ana"],
        "voice": ["voice_profile:ana"], "exclusions": ["no blood"]})
    snap = run.snapshot
    assert snap.identity_refs == ("identity_profile:ana",)
    assert snap.voice_refs == ("voice_profile:ana",)
    assert snap.operator_refs == ("upload:kitchen.png",)
    assert snap.exclusions == ("no blood",)


def test_the_same_prompt_twice_is_one_snapshot_entry(tmp_path):
    run = make_run(tmp_path, sources=[
        {"prompt_id": "a", "text": "one text"},
        {"prompt_id": "b", "text": "one text"}])
    assert run.snapshot.prompts_before_run == ("one text",)


# ---------------------------------------------------------------------------
# [3] authoring + the 422 paths
# ---------------------------------------------------------------------------


def test_authoring_a_plot_with_a_good_reply_stores_the_artifact(tmp_path):
    run = make_run(tmp_path)
    entry = run.author("plot", llm=plot_llm(json.dumps(VALID_PLOT)))
    assert entry["provenance"] == "authored"
    assert entry["digest"] == run.plot().digest
    assert run.plot().premise.startswith("A woman decides")


def test_an_authoring_gap_is_a_422_and_never_a_coerced_artifact(tmp_path):
    run = make_run(tmp_path)
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.author("plot", llm=plot_llm('{"premise": "x"}'))
    exc = excinfo.value
    assert exc.code == "AUTHORING_GAP"
    assert exc.http_status == 422
    assert exc.detail["gap"]["raw"]
    assert exc.errors
    assert run.plot() is None                     # nothing was coerced
    # the gap is PERSISTED so the screen can show it after a reload
    reloaded = sf.ScriptFirstRun.load(run.run_id, str(tmp_path))
    assert reloaded.state["artifacts"]["plot"]["gap"]["code"]


def test_a_capability_gap_from_bind_llm_is_the_same_typed_shape(tmp_path):
    run = make_run(tmp_path)
    gap = AuthoringGap(errors=("no text model is eligible",), stage="bind",
                       code="CAPABILITY_GAP")
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.author("plot", llm=gap)
    assert excinfo.value.detail["gap"]["code"] == "CAPABILITY_GAP"


def test_a_screenplay_cannot_be_authored_before_its_plot(tmp_path):
    run = make_run(tmp_path)
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.author("screenplay", llm=plot_llm("{}"))
    assert excinfo.value.code == "ARTIFACT_MISSING"


def test_only_plot_and_screenplay_are_model_authored(tmp_path):
    run = make_run(tmp_path)
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.author("continuity", llm=plot_llm("{}"))
    assert excinfo.value.code == "STAGE_UNKNOWN"


def test_an_operator_edit_goes_through_the_same_constructor(tmp_path):
    run = make_run(tmp_path)
    entry = run.put_artifact("plot", VALID_PLOT)
    assert entry["provenance"] == "operator_edit"
    assert run.plot().digest == PlotSpec.from_dict(VALID_PLOT).digest


def test_an_invalid_operator_edit_reports_every_problem_at_once(tmp_path):
    run = make_run(tmp_path)
    broken = json.loads(json.dumps(VALID_PLOT))
    broken["beats"][1]["causes"] = ["b9"]          # a cause that does not exist
    broken["characters"].append({"name": "GHOST", "goal": "g", "conflict": "c",
                                 "arc": "a", "description": "d"})
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.put_artifact("plot", broken)
    exc = excinfo.value
    assert exc.code == "ARTIFACT_INVALID"
    assert exc.http_status == 422
    assert len(exc.errors) >= 2                    # not just the first problem
    assert any("b9" in e for e in exc.errors)


def test_a_non_object_edit_is_refused_rather_than_coerced(tmp_path):
    run = make_run(tmp_path)
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.put_artifact("plot", ["not", "an", "object"])
    assert excinfo.value.code == "ARTIFACT_INVALID"


def test_editing_the_screenplay_drops_what_was_derived_from_it(tmp_path):
    run = make_run(tmp_path)
    play = screenplay_obj()
    run.put_artifact("screenplay", play.to_dict())
    run.build_preproduction()
    assert run.continuity() is not None
    edited = play.to_dict()
    edited["logline"] = "A woman leaves, at last."
    run.put_artifact("screenplay", edited)
    assert run.continuity() is None                # stale, dropped, not reused
    assert run.shot_draft() is None


def test_provenance_is_not_authorable_by_an_editor(tmp_path):
    run = make_run(tmp_path)
    run.put_artifact("plot", VALID_PLOT)
    edited = screenplay_obj().to_dict()
    edited["plot_digest"] = "0" * 64
    run.put_artifact("screenplay", edited)
    assert run.screenplay().plot_digest == run.plot().digest


# ---------------------------------------------------------------------------
# [4] doc Stage 11 / Stage 10 — the lock and the revision
# ---------------------------------------------------------------------------


def test_preproduction_is_derivable_before_any_lock(tmp_path):
    run = make_run(tmp_path)
    run.put_artifact("screenplay", screenplay_obj().to_dict())
    built = run.build_preproduction()
    assert built["shot_plan"]["digest"] and built["continuity"]["digest"]
    assert run.shot_draft().audio_first is False   # no master yet, and it says so
    assert any("estimate" in l for l in run.limitations())


def test_a_lock_without_an_audio_master_refuses_honestly_and_names_the_seat(tmp_path):
    run = make_run(tmp_path)
    run.put_artifact("screenplay", screenplay_obj().to_dict())
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.lock_run()
    exc = excinfo.value
    assert exc.code == "AUDIO_MASTER_MISSING"
    assert exc.http_status == 409
    assert exc.detail["capability"] == "audio.tts"
    assert "chatterbox" in exc.detail["requirement"]
    assert exc.detail["eligible"] is False
    assert not run.is_locked


def test_a_lock_needs_a_screenplay(tmp_path):
    run = make_run(tmp_path)
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.lock_run()
    assert excinfo.value.code == "ARTIFACT_MISSING"


def test_the_lock_closes_over_every_artifact_and_locks_the_screenplay(tmp_path):
    run = locked_run(tmp_path)
    lock = run.lock()
    assert run.is_locked
    assert run.screenplay().locked is True
    assert lock.screenplay_digest == run.screenplay().digest
    assert lock.continuity_digest == run.continuity().digest
    assert lock.audio_master_digest == run.audio_master().digest
    assert lock.shot_plan_digest == run.shot_draft().plan.digest
    assert lock.registry_version == REGISTRY
    assert lock.digest in lock.parent_digests


def test_a_master_from_a_different_draft_is_refused_by_digest(tmp_path):
    run = make_run(tmp_path)
    run.put_artifact("screenplay", screenplay_obj().to_dict())
    other = screenplay_obj()
    other = Screenplay(title="Other", scenes=other.scenes, logline="different")
    stale = master_for(other)
    stale = AudioMaster(timeline_digest="0" * 64,
                        line_timings=stale.line_timings, tracks=stale.tracks,
                        total_seconds=stale.total_seconds, locked=True)
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.lock_run(audio_master=stale.to_dict())
    assert excinfo.value.code == "LOCK_REFUSED"
    assert not run.is_locked


def test_an_unlocked_audio_master_is_refused(tmp_path):
    run = make_run(tmp_path)
    play = screenplay_obj()
    run.put_artifact("screenplay", play.to_dict())
    draft = master_for(play).to_dict()
    draft["locked"] = False
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.lock_run(audio_master=draft)
    assert excinfo.value.code == "LOCK_REFUSED"
    assert "not locked" in "\n".join(excinfo.value.errors)


def test_after_the_lock_an_artifact_edit_is_refused_and_names_revise(tmp_path):
    run = locked_run(tmp_path)
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.put_artifact("plot", VALID_PLOT)
    exc = excinfo.value
    assert exc.code == "ALREADY_LOCKED"
    assert "/revise" in exc.message


def test_a_revision_without_a_reason_is_refused(tmp_path):
    run = locked_run(tmp_path)
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.revise("   ")
    assert excinfo.value.code == "REVISION_REASON_MISSING"


def test_a_revision_is_n_plus_one_with_the_reason_recorded(tmp_path):
    run = locked_run(tmp_path)
    before = run.lock().digest
    run.revise("the kitchen shot needs a longer window")
    lock = run.lock()
    assert lock.revision == 1
    assert lock.parent_revision == 0
    assert lock.revision_reason == "the kitchen shot needs a longer window"
    assert lock.digest != before
    history = run.state["lock_history"]
    assert [h["revision"] for h in history] == [0, 1]


def test_revising_drops_the_previous_revisions_segments(tmp_path):
    run = locked_run(tmp_path)
    run.compile(catalog_view=catalog_view)
    assert run.state["segments"]
    run.revise("re-cut the opening")
    assert run.state["segments"] is None           # never mixed across versions


def test_revising_an_artifact_folds_its_new_digest_into_the_revision(tmp_path):
    run = locked_run(tmp_path)
    play = run.screenplay()
    edited = play.to_dict()
    edited["logline"] = "A woman leaves, at last."
    run.revise("the logline was wrong",
               artifacts={"screenplay": edited})
    assert run.lock().screenplay_digest == run.screenplay().digest
    assert run.lock().revision == 1


def test_compiling_before_the_lock_is_refused(tmp_path):
    run = make_run(tmp_path)
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.compile(catalog_view=catalog_view)
    assert excinfo.value.code == "NOT_LOCKED"


# ---------------------------------------------------------------------------
# [5] doc Stage 14 — siblings, not a chain
# ---------------------------------------------------------------------------


def test_every_segment_names_the_same_lock_side_parents_and_no_sibling(tmp_path):
    run = locked_run(tmp_path)
    entry = run.compile(catalog_view=catalog_view)
    specs = run.specs()
    assert len(specs) >= 2
    lock = run.lock()
    parents = set(lock.parent_digests)
    ids = {s.segment_id for s in specs}
    for spec in specs:
        assert set(spec.parents) == parents
        assert not (set(spec.parents) & {s.digest for s in specs})
        assert spec.lock_digest == lock.digest
    assert entry["sibling_shape"]["parent_digest"] == lock.digest
    assert set(entry["sibling_shape"]["children"]) == ids


def test_the_response_carries_prompt_seed_parents_and_the_validator_report(tmp_path):
    run = locked_run(tmp_path)
    entry = run.compile(catalog_view=catalog_view)
    row = entry["specs"][0]
    assert row["prompt"] and row["seed_base"] and row["parents"]
    assert row["window"] and row["rubric"]
    assert "ok" in entry["validation"]
    assert entry["graph"]["nodes"]
    assert "production_lock" in entry["graph"]["nodes"]


def test_parallel_and_sequential_are_the_same_graph_batched_differently(tmp_path):
    run = locked_run(tmp_path)
    entry = run.compile(catalog_view=catalog_view)
    orders = entry["execution_order"]
    flat_seq = [n for batch in orders["sequential"] for n in batch]
    flat_par = [n for batch in orders["parallel"] for n in batch]
    assert sorted(flat_seq) == sorted(flat_par)
    assert len(orders["sequential"]) >= len(orders["parallel"])


def test_every_minted_prompt_lands_in_the_run_ledger(tmp_path):
    run = locked_run(tmp_path)
    run.compile(catalog_view=catalog_view)
    ledger = run.ledger
    for spec in run.specs():
        assert spec.prompt in ledger
    assert len(ledger) == len(run.specs())


def test_a_minted_segment_prompt_can_never_re_enter_this_snapshot(tmp_path):
    run = locked_run(tmp_path)
    run.compile(catalog_view=catalog_view)
    minted = run.specs()[0].prompt
    assert run.accepts_source(minted) is False
    assert run.accepts_source("something nobody minted here") is True


def test_compiling_twice_is_deterministic(tmp_path):
    run = locked_run(tmp_path)
    first = run.compile(catalog_view=catalog_view)
    second = run.compile(catalog_view=catalog_view)
    assert [r["digest"] for r in first["specs"]] == \
           [r["digest"] for r in second["specs"]]


# -- regeneration ------------------------------------------------------------


def test_regenerating_one_segment_leaves_every_sibling_digest_untouched(tmp_path):
    run = locked_run(tmp_path)
    run.compile(catalog_view=catalog_view)
    calls = []
    dispatch = recording_dispatch(calls)
    target = run.specs()[0].segment_id
    before = dict(run.digests()["segments"])
    run.generate_segment(target, dispatch=dispatch)
    attempt = run.generate_segment(target, dispatch=dispatch)
    after = dict(run.digests()["segments"])
    assert before == after
    assert attempt["siblings_unchanged"] is True
    assert attempt["siblings_before"] == attempt["siblings_after"]
    # and no sibling was dispatched
    assert {c["segment_id"] for c in calls} == {target}


def test_a_regeneration_reuses_the_frozen_spec_at_the_next_seed(tmp_path):
    run = locked_run(tmp_path)
    run.compile(catalog_view=catalog_view)
    calls = []
    target = run.specs()[0]
    a1 = run.generate_segment(target.segment_id,
                              dispatch=recording_dispatch(calls))
    a2 = run.generate_segment(target.segment_id,
                              dispatch=recording_dispatch(calls))
    assert a1["spec_digest"] == a2["spec_digest"] == target.digest
    assert a1["prompt"] == a2["prompt"] == target.prompt
    assert a1["seed"] == target.seed_base
    assert a2["seed"] == target.seed_base + 1
    assert (a1["attempt"], a2["attempt"]) == (1, 2)


def test_an_attempt_records_the_exact_prompt_model_seed_and_params(tmp_path):
    run = locked_run(tmp_path)
    run.compile(catalog_view=catalog_view)
    spec = run.specs()[0]
    attempt = run.generate_segment(spec.segment_id,
                                   dispatch=recording_dispatch([]))
    assert attempt["prompt"] == spec.prompt
    assert attempt["model_id"] == "sd-turbo"
    assert attempt["params"]["seed"] == spec.seed_base
    assert attempt["lock_digest"] == run.lock().digest
    assert attempt["registry_version"] == REGISTRY
    assert attempt["parents"] == list(spec.parents)
    assert attempt["receipt"]["model_id"] == "sd-turbo"


def test_a_dispatch_gap_is_recorded_as_an_attempt_not_swallowed(tmp_path):
    run = locked_run(tmp_path)
    run.compile(catalog_view=catalog_view)

    def gapping(spec, *, kind, seed, settings=None):
        return {"ok": False, "kind": kind, "capability": sf.CLIP_CAPABILITY,
                "model_id": None, "seed": seed, "prompt": spec.prompt,
                "params": {}, "artifacts": [], "receipt": None,
                "gap": {"code": "CAPABILITY_GAP",
                        "capability": sf.CLIP_CAPABILITY,
                        "reasons": ["video.* resolves deferred by design"],
                        "requirement": sf.SEAM_REQUIREMENTS[sf.CLIP_CAPABILITY]}}

    spec = run.specs()[0]
    attempt = run.generate_segment(spec.segment_id, kind="clip",
                                   dispatch=gapping)
    assert attempt["ok"] is False
    assert attempt["gap"]["capability"] == sf.CLIP_CAPABILITY
    assert "studio" in attempt["gap"]["requirement"]
    assert len(run.attempts(spec.segment_id)) == 1


def test_a_raising_dispatch_becomes_receipt_data_not_a_500(tmp_path):
    run = locked_run(tmp_path)
    run.compile(catalog_view=catalog_view)

    def explode(spec, *, kind, seed, settings=None):
        raise RuntimeError("the worker went away")

    spec = run.specs()[0]
    attempt = run.generate_segment(spec.segment_id, dispatch=explode)
    assert attempt["ok"] is False
    assert "the worker went away" in attempt["gap"]["reasons"][0]


def test_generating_before_compiling_is_refused(tmp_path):
    run = locked_run(tmp_path)
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.generate_segment("s1-1", dispatch=recording_dispatch([]))
    assert excinfo.value.code == "SEGMENTS_MISSING"


def test_an_unknown_segment_is_a_404_shaped_refusal(tmp_path):
    run = locked_run(tmp_path)
    run.compile(catalog_view=catalog_view)
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.generate_segment("nope", dispatch=recording_dispatch([]))
    assert excinfo.value.code == "SEGMENT_UNKNOWN"
    assert excinfo.value.http_status == 404


def test_attempts_survive_a_reload(tmp_path):
    run = locked_run(tmp_path)
    run.compile(catalog_view=catalog_view)
    spec = run.specs()[0]
    run.generate_segment(spec.segment_id, dispatch=recording_dispatch([]))
    again = sf.ScriptFirstRun.load(run.run_id, str(tmp_path))
    assert len(again.attempts(spec.segment_id)) == 1


# ---------------------------------------------------------------------------
# [6] promotion — a NEW run only
# ---------------------------------------------------------------------------


def test_a_promoted_prompt_is_refused_re_entry_into_its_own_run(tmp_path):
    run = locked_run(tmp_path)
    run.compile(catalog_view=catalog_view)
    spec = run.specs()[0]
    record = run.promote(segment_id=spec.segment_id, note="the good one")
    assert record["digest"] == prompt_digest(spec.prompt)
    assert record["usable_in"] == "a NEW run only"
    assert "generated during this run" in record["refused_here"]
    assert run.accepts_source(record["text"]) is False


def test_a_promoted_prompt_seeds_a_new_run(tmp_path):
    run = locked_run(tmp_path)
    run.compile(catalog_view=catalog_view)
    record = run.promote(segment_id=run.specs()[0].segment_id)
    fresh = sf.ScriptFirstRun.create(
        deliverable="the sequel", root=str(tmp_path), fleet=fleet, route=route,
        sources=[{"prompt_id": record["source_id"], "text": record["text"],
                  "hash": record["digest"], "origin": "promoted"}])
    assert record["text"] in fresh.snapshot.prompts_before_run
    # the NEW run's ledger is empty — nothing was minted here yet, which is
    # exactly why the same text that its origin run refuses is admissible here.
    assert len(fresh.ledger) == 0
    assert fresh.accepts_source(record["text"]) is True
    assert run.accepts_source(record["text"]) is False


def test_a_promoted_source_is_persisted_outside_any_run(tmp_path):
    run = locked_run(tmp_path)
    run.compile(catalog_view=catalog_view)
    record = run.promote(segment_id=run.specs()[0].segment_id)
    listed = sf.list_promoted_sources(str(tmp_path))
    assert [s["source_id"] for s in listed] == [record["source_id"]]
    assert sf.load_promoted_source(record["source_id"],
                                   str(tmp_path))["digest"] == record["digest"]
    # and it does NOT show up as a run
    assert record["source_id"] not in [r["run_id"] for r in
                                       sf.ScriptFirstRun.list_runs(str(tmp_path))]


def test_promoting_free_text_works_and_is_still_run_scoped(tmp_path):
    run = make_run(tmp_path)
    record = run.promote(text="a kettle that never boils", note="an idea")
    assert run.accepts_source("a kettle that never boils") is False
    assert record["origin"]["run_id"] == run.run_id


def test_promoting_nothing_is_refused(tmp_path):
    run = make_run(tmp_path)
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.promote()
    assert excinfo.value.code == "PROMOTE_REFUSED"


def test_promoting_an_unknown_attempt_is_refused(tmp_path):
    run = locked_run(tmp_path)
    run.compile(catalog_view=catalog_view)
    with pytest.raises(sf.ScriptFirstRefused) as excinfo:
        run.promote(segment_id=run.specs()[0].segment_id, attempt=99)
    assert excinfo.value.code == "SEGMENT_UNKNOWN"


def test_a_promotion_carries_its_origin_receipt(tmp_path):
    run = locked_run(tmp_path)
    run.compile(catalog_view=catalog_view)
    spec = run.specs()[0]
    run.generate_segment(spec.segment_id, dispatch=recording_dispatch([]))
    record = run.promote(segment_id=spec.segment_id, attempt=1)
    origin = record["origin"]
    assert origin["run_id"] == run.run_id
    assert origin["segment_id"] == spec.segment_id
    assert origin["spec_digest"] == spec.digest
    assert origin["lock_digest"] == run.lock().digest
    assert origin["attempt"] == 1
    assert origin["artifacts"]


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------


def test_every_refusal_code_has_a_status(tmp_path):
    assert set(sf.REFUSAL_CODES) == set(sf.REFUSAL_STATUS)


def test_an_unknown_refusal_code_cannot_be_constructed():
    with pytest.raises(ValueError):
        sf.ScriptFirstRefused("NOT_A_CODE", "nope")


def test_the_module_does_not_import_video_intel_or_the_catalog_at_module_level():
    import ast
    path = os.path.join(_SRC, "hugpy_oracle", "script_first.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    top = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            top.append(node.module or "")
    assert not [m for m in top if "video_intel" in m or "catalog" in m
                or "router" in m or "runtime" in m], top


def test_the_run_state_is_json_serialisable(tmp_path):
    run = locked_run(tmp_path)
    run.compile(catalog_view=catalog_view)
    run.generate_segment(run.specs()[0].segment_id,
                         dispatch=recording_dispatch([]))
    json.dumps(run.state)                          # must not raise


def test_the_authoring_deadline_is_passed_to_the_binding(tmp_path, monkeypatch):
    """Authoring is the one synchronous oracle call whose honest cost is
    minutes; the deadline is widened by the caller, never removed."""
    seen = {}

    def fake_bind(**kwargs):
        seen.update(kwargs)
        return plot_llm(json.dumps(VALID_PLOT))

    monkeypatch.setattr(sf, "bind_llm", fake_bind)
    run = make_run(tmp_path)
    run.author("plot", deadline_s=600)
    assert seen["deadline_s"] == 600.0
    assert "plot" in seen["objective"]


def test_a_gapped_stage_is_not_summarised_as_an_artifact(tmp_path):
    run = make_run(tmp_path)
    with pytest.raises(sf.ScriptFirstRefused):
        run.author("plot", llm=plot_llm("not json at all"))
    summary = run.summary()
    assert "plot" not in summary["stages"]
    assert summary["gapped_stages"] == ["plot"]
    run.put_artifact("screenplay", screenplay_obj().to_dict())
    assert run.summary()["stages"] == ["screenplay"]
