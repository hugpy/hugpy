"""k114 — HTTP for the script-first pipeline: /video/script/*.

The blueprint is registered on a BARE Flask app (no app factory, no
``video_auth`` gate, no worker daemon) and every live seam on
``oracle.script_first`` is monkeypatched, so these tests exercise the wire
shape and the status-code mapping without a catalog, a registry, a model or a
GPU. What is asserted here is the thing a route can get wrong that the module
cannot: the status code, the body shape, and that a refusal is never softened
into a 200.

Locks:
  [1] the lifecycle over HTTP: 201 create, 200 get/list, and the full run state
      on every mutation so a UI never has to re-fetch to stay honest.
  [2] the refusal mapping: 422 for anything the artifact constructors reject
      (authoring gap, bad edit, bad source hash), 409 for a run that is in the
      wrong state (no audio master, already locked, not locked), 404 for a run
      or segment that does not exist.
  [3] the segment surface: the compile response carries the sibling shape and
      the validator report; a generate is an ATTEMPT with its receipt, and a
      gapped attempt is still a recorded attempt rather than an error.
  [4] promotion: 201 with the actual refusal text, and the promoted source is
      offered to a NEW run by id without its text being trusted from the body.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_script_first_routes.py -q
"""
from __future__ import annotations

import json
import logging
import os
import sys

import pytest

logging.disable(logging.INFO)

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from flask import Flask  # noqa: E402

from hugpy_oracle import script_first as sf
from hugpy_oracle.audio_master import AudioMaster, Line, LineTiming
from hugpy_oracle.screenplay import Scene, Screenplay
from hugpy_server.app.routes import script_first_routes as routes

BASE = routes.BASE
REGISTRY = "sha256:testregistry00"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A bare app with only this blueprint, and every live seam replaced."""
    monkeypatch.setenv(sf.RUN_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(sf, "live_fleet_view", lambda: {
        "registry_version": REGISTRY,
        "capabilities": [
            {"name": "text.chat", "eligible": True,
             "model_ids": ["fixture-llm"], "reasons": []},
            {"name": "audio.tts", "eligible": False, "model_ids": [],
             "reasons": ["no worker seats text-to-speech"]}],
        "hardware": {"workers": []}})
    monkeypatch.setattr(sf, "live_authoring_route", lambda: {
        "capability": "text.chat", "execution": "execute",
        "model_id": "fixture-llm", "model_rationale": "only-eligible",
        "reasons": []})
    monkeypatch.setattr(sf, "live_catalog_view", lambda: {})
    monkeypatch.setattr(sf, "live_segment_dispatch",
                        lambda spec, *, kind, seed, settings=None: {
                            "ok": True, "kind": kind,
                            "capability": "image.generate",
                            "model_id": "sd-turbo", "seed": seed,
                            "prompt": spec.prompt,
                            "params": {"seed": seed},
                            "artifacts": [{"kind": "image",
                                           "uri": f"/out/{seed}.png"}],
                            "receipt": {"model_id": "sd-turbo"}, "gap": None})
    app = Flask(__name__)
    app.register_blueprint(routes.script_first_bp)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def post(client, path, payload=None):
    return client.post(path, data=json.dumps(payload or {}),
                       content_type="application/json")


def put(client, path, payload):
    return client.put(path, data=json.dumps(payload),
                      content_type="application/json")


def screenplay_dict():
    line = Line(line_id="l1", speaker="ANA", text="I am leaving.")
    s1 = Scene(scene_id="s1", heading="INT. KITCHEN - DAY", location="KITCHEN",
               time_of_day="DAY", action="Ana stares at the kettle.",
               present_at_open=("ANA",), props=("kettle",), dialogue=(line,))
    s2 = Scene(scene_id="s2", heading="EXT. STREET - NIGHT", location="STREET",
               time_of_day="NIGHT", action="Ana walks away from the house.",
               present_at_open=("ANA",), story_time_s=600.0)
    return Screenplay(title="Kettle", scenes=(s1, s2),
                      logline="A woman leaves.").to_dict()


def master_dict():
    play = Screenplay.from_dict(screenplay_dict())
    timeline = play.to_dialogue_timeline(locked=True)
    timings = tuple(LineTiming(line_id=l.line_id, start_s=float(i) * 3.0,
                               end_s=float(i) * 3.0 + 2.0, pause_after_s=0.5)
                    for i, l in enumerate(play.lines))
    tracks = tuple((l.line_id, f"/fixtures/{l.line_id}.wav") for l in play.lines)
    return AudioMaster(timeline_digest=timeline.digest, line_timings=timings,
                       tracks=tracks, total_seconds=12.0, locked=True).to_dict()


def created(client, **body):
    body.setdefault("deliverable", "a two-shot short")
    body.setdefault("requirements", "Ana leaves the house at dusk.")
    body.setdefault("sources", [{"prompt_id": "p1",
                                 "text": "a woman leaves a house",
                                 "persisted_at": "2020-01-01T00:00:00+00:00"}])
    res = post(client, f"{BASE}/runs", body)
    assert res.status_code == 201, res.get_json()
    return res.get_json()["run_id"]


def locked(client):
    run_id = created(client)
    assert put(client, f"{BASE}/runs/{run_id}/screenplay",
               screenplay_dict()).status_code == 200
    res = post(client, f"{BASE}/runs/{run_id}/lock",
               {"audio_master": master_dict()})
    assert res.status_code == 200, res.get_json()
    return run_id


# ---------------------------------------------------------------------------
# [1] lifecycle
# ---------------------------------------------------------------------------


def test_create_returns_201_with_the_run_and_its_snapshot(client):
    res = post(client, f"{BASE}/runs", {
        "deliverable": "a two-shot short",
        "requirements": "Ana leaves at dusk.",
        "sources": [{"prompt_id": "p1", "text": "a woman leaves a house"}]})
    assert res.status_code == 201
    body = res.get_json()
    assert body["ok"] is True and body["run_id"].startswith("sf-")
    run = body["run"]
    assert run["snapshot"]["prompts_before_run"] == ["a woman leaves a house"]
    assert run["snapshot_digest"]
    assert run["models"]["fleet"]["registry_version"] == REGISTRY
    assert run["models"]["authoring_route"]["model_id"] == "fixture-llm"
    assert run["locked"] is False
    assert run["limitations"]


def test_list_and_get_round_trip(client):
    run_id = created(client)
    listing = client.get(f"{BASE}/runs").get_json()
    assert listing["ok"] and listing["count"] == 1
    assert listing["runs"][0]["run_id"] == run_id
    one = client.get(f"{BASE}/runs/{run_id}")
    assert one.status_code == 200
    assert one.get_json()["run"]["run_id"] == run_id


def test_an_unknown_run_is_404(client):
    res = client.get(f"{BASE}/runs/sf-nope")
    assert res.status_code == 404
    assert res.get_json()["code"] == "RUN_NOT_FOUND"
    assert res.get_json()["ok"] is False


def test_a_post_start_source_is_excluded_and_shown(client, monkeypatch):
    res = post(client, f"{BASE}/runs", {
        "deliverable": "a short",
        "sources": [{"prompt_id": "before", "text": "before the run",
                     "persisted_at": "2020-01-01T00:00:00+00:00"},
                    {"prompt_id": "after", "text": "after the run",
                     "persisted_at": "2999-01-01T00:00:00+00:00"}]})
    run = res.get_json()["run"]
    rows = {r["prompt_id"]: r for r in run["sources"]}
    assert rows["after"]["included"] is False
    assert rows["after"]["exclusion_reason"]
    assert run["snapshot"]["prompts_before_run"] == ["before the run"]


def test_a_bad_source_hash_is_422(client):
    res = post(client, f"{BASE}/runs", {
        "deliverable": "a short",
        "sources": [{"prompt_id": "p1", "text": "x", "hash": "0" * 64}]})
    assert res.status_code == 422
    assert res.get_json()["code"] == "SOURCE_DIGEST_MISMATCH"


def test_a_missing_deliverable_is_422(client):
    res = post(client, f"{BASE}/runs", {"sources": []})
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# [2] authoring / editing over HTTP
# ---------------------------------------------------------------------------


def test_an_authoring_gap_is_422_with_the_errors_and_the_raw_reply(client,
                                                                   monkeypatch):
    monkeypatch.setattr(sf, "bind_llm", lambda *a, **k: (lambda p: "not json"))
    run_id = created(client)
    res = post(client, f"{BASE}/runs/{run_id}/plot", {})
    assert res.status_code == 422
    body = res.get_json()
    assert body["code"] == "AUTHORING_GAP"
    assert body["errors"]
    assert body["detail"]["gap"]["raw"] == "not json"
    assert body["detail"]["gap"]["code"] in ("AUTHORING_UNPARSED",
                                             "AUTHORING_INVALID")


def test_no_eligible_text_model_is_the_same_422_shape(client, monkeypatch):
    from hugpy_oracle.screenplay import AuthoringGap
    monkeypatch.setattr(sf, "bind_llm", lambda *a, **k: AuthoringGap(
        errors=("capability 'text.chat' is not executable on this fleet",),
        stage="bind", code="CAPABILITY_GAP"))
    run_id = created(client)
    res = post(client, f"{BASE}/runs/{run_id}/plot", {})
    assert res.status_code == 422
    assert res.get_json()["detail"]["gap"]["code"] == "CAPABILITY_GAP"


# ---------------------------------------------------------------------------
# k114's follow-up, landed: authoring routes through k109's matrix, honestly
# ---------------------------------------------------------------------------

_VALID_PLOT = {
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


def _matrix_row(operation, model):
    return {"operation": operation, "model": model, "ok": True, "track": "B",
            "mode": "normalized", "failure": None,
            "deterministic": {"score": 82.0, "preservation": 1.0,
                              "contradiction_rate": 0.0, "completeness": 1.0,
                              "constraint_adherence": None, "accuracy": None},
            "judge": {"score": 85.0, "available": True,
                      "judge_model": "judge-x"},
            "perf": {"latency_s": 24.3, "tokens_per_s": None,
                     "vram_used_delta_bytes": None}}


def test_plot_authoring_routes_through_the_k109_matrix_when_available(
        client, monkeypatch):
    matrix = sf.routing_matrix.derive_matrix(
        [_matrix_row("plot.construct", "Qwen3.8_4B_Distilled_GGUF")],
        registry_version=REGISTRY, mode="normalized", run_id="oracle-pilot")
    monkeypatch.setattr(sf.routing_matrix, "load_latest_matrix",
                        lambda *a, **k: (matrix, "stub: verified"))
    seen: dict = {}

    def fake_bind(**kwargs):
        seen.update(kwargs)
        return lambda prompt: json.dumps(_VALID_PLOT)

    monkeypatch.setattr(sf, "bind_llm", fake_bind)
    run_id = created(client)
    res = post(client, f"{BASE}/runs/{run_id}/plot", {})
    assert res.status_code == 200
    assert seen["requested_model"] == "Qwen3.8_4B_Distilled_GGUF"
    body = res.get_json()
    choice = body["artifact"]["model_choice"]
    assert choice["source"] == "k109-matrix"
    assert choice["requested_model"] == "Qwen3.8_4B_Distilled_GGUF"
    assert "k109 routing matrix primary" in choice["reason"]
    assert body["run"]["models"]["last_authoring_choice"] == choice
    assert "Qwen3.8_4B_Distilled_GGUF" in body["artifact"]["note"]


def test_plot_authoring_falls_back_honestly_when_the_matrix_is_stale(
        client, monkeypatch):
    # ``load_latest_matrix`` itself is what refuses a registry_version
    # mismatch (routing_matrix's own tests cover the mismatch logic); this
    # is the shape it hands back to ``author()`` when it does.
    monkeypatch.setattr(
        sf.routing_matrix, "load_latest_matrix",
        lambda *a, **k: (None, "registry_version mismatch: not honoured"))
    seen: dict = {}

    def fake_bind(**kwargs):
        seen.update(kwargs)
        return lambda prompt: json.dumps(_VALID_PLOT)

    monkeypatch.setattr(sf, "bind_llm", fake_bind)
    run_id = created(client)
    res = post(client, f"{BASE}/runs/{run_id}/plot", {})
    assert res.status_code == 200
    assert seen["requested_model"] is None      # never a stale route
    choice = res.get_json()["artifact"]["model_choice"]
    assert choice["source"] == "catalog-default"
    assert choice["requested_model"] is None
    assert "not honoured" in choice["reason"]


def test_plot_authoring_falls_back_honestly_when_no_matrix_exists(
        client, monkeypatch):
    monkeypatch.setattr(sf.routing_matrix, "load_latest_matrix",
                        lambda *a, **k: (None, "no oracle-* run dir found"))
    seen: dict = {}

    def fake_bind(**kwargs):
        seen.update(kwargs)
        return lambda prompt: json.dumps(_VALID_PLOT)

    monkeypatch.setattr(sf, "bind_llm", fake_bind)
    run_id = created(client)
    res = post(client, f"{BASE}/runs/{run_id}/plot", {})
    assert res.status_code == 200
    assert seen["requested_model"] is None
    choice = res.get_json()["artifact"]["model_choice"]
    assert choice["source"] == "catalog-default"
    assert "no oracle-* run dir found" in choice["reason"]


def test_putting_a_valid_screenplay_is_200_and_returns_the_whole_run(client):
    run_id = created(client)
    res = put(client, f"{BASE}/runs/{run_id}/screenplay", screenplay_dict())
    assert res.status_code == 200
    body = res.get_json()
    assert body["artifact"]["provenance"] == "operator_edit"
    assert body["run"]["digests"]["screenplay"] == body["artifact"]["digest"]


def test_putting_an_invalid_screenplay_is_422_listing_every_problem(client):
    run_id = created(client)
    broken = screenplay_dict()
    broken["scenes"][0]["heading"] = "the kitchen"
    broken["scenes"][1]["scene_id"] = "s1"
    res = put(client, f"{BASE}/runs/{run_id}/screenplay", broken)
    assert res.status_code == 422
    body = res.get_json()
    assert body["code"] == "ARTIFACT_INVALID"
    assert body["errors"]


def test_an_unknown_stage_is_404(client):
    run_id = created(client)
    res = post(client, f"{BASE}/runs/sf-nope/plot", {})
    assert res.status_code == 404


def test_preproduction_derives_both_viewers_before_any_lock(client):
    run_id = created(client)
    put(client, f"{BASE}/runs/{run_id}/screenplay", screenplay_dict())
    res = post(client, f"{BASE}/runs/{run_id}/preproduction", {})
    assert res.status_code == 200
    body = res.get_json()
    assert body["artifacts"]["continuity"]["digest"]
    assert body["artifacts"]["shot_plan"]["digest"]
    assert body["run"]["locked"] is False


# ---------------------------------------------------------------------------
# [2b] the lock / revise surface
# ---------------------------------------------------------------------------


def test_locking_without_an_audio_master_is_409_naming_the_seat(client):
    run_id = created(client)
    put(client, f"{BASE}/runs/{run_id}/screenplay", screenplay_dict())
    res = post(client, f"{BASE}/runs/{run_id}/lock", {})
    assert res.status_code == 409
    body = res.get_json()
    assert body["code"] == "AUDIO_MASTER_MISSING"
    assert body["detail"]["capability"] == "audio.tts"
    assert "chatterbox" in body["detail"]["requirement"]


def test_locking_with_a_master_is_200_and_the_run_reads_locked(client):
    run_id = locked(client)
    run = client.get(f"{BASE}/runs/{run_id}").get_json()["run"]
    assert run["locked"] is True
    assert run["digests"]["production_lock"]
    assert run["lock"]["parent_digests"]
    assert run["lock_history"][0]["reason"] == "initial lock"


def test_editing_after_the_lock_is_409(client):
    run_id = locked(client)
    res = put(client, f"{BASE}/runs/{run_id}/screenplay", screenplay_dict())
    assert res.status_code == 409
    assert res.get_json()["code"] == "ALREADY_LOCKED"
    assert "/revise" in res.get_json()["message"]


def test_a_revision_without_a_reason_is_422(client):
    run_id = locked(client)
    res = post(client, f"{BASE}/runs/{run_id}/revise", {"reason": ""})
    assert res.status_code == 422
    assert res.get_json()["code"] == "REVISION_REASON_MISSING"


def test_a_revision_with_a_reason_is_200_and_bumps_the_revision(client):
    run_id = locked(client)
    res = post(client, f"{BASE}/runs/{run_id}/revise",
               {"reason": "the opening shot is too short"})
    assert res.status_code == 200
    lock = res.get_json()["lock"]["payload"]
    assert lock["revision"] == 1
    assert lock["revision_reason"] == "the opening shot is too short"


def test_compiling_before_the_lock_is_409(client):
    run_id = created(client)
    res = post(client, f"{BASE}/runs/{run_id}/segments", {})
    assert res.status_code == 409
    assert res.get_json()["code"] == "NOT_LOCKED"


# ---------------------------------------------------------------------------
# [3] segments
# ---------------------------------------------------------------------------


def test_compiling_returns_the_sibling_shape_and_the_validator_report(client):
    run_id = locked(client)
    res = post(client, f"{BASE}/runs/{run_id}/segments", {})
    assert res.status_code == 200
    segments = res.get_json()["segments"]
    assert len(segments["specs"]) >= 2
    shape = segments["sibling_shape"]
    assert shape["parent"] == "production_lock"
    assert shape["parent_digest"] == segments["lock_digest"]
    parents = set(segments["parent_digests"])
    child_digests = {r["digest"] for r in segments["specs"]}
    for row in segments["specs"]:
        assert set(row["parents"]) == parents
        assert not (set(row["parents"]) & child_digests)
        assert row["prompt"] and row["seed_base"]
    assert "ok" in segments["validation"]
    assert "production_lock" in segments["graph"]["nodes"]


def test_get_segments_returns_specs_and_attempts(client):
    run_id = locked(client)
    post(client, f"{BASE}/runs/{run_id}/segments", {})
    body = client.get(f"{BASE}/runs/{run_id}/segments").get_json()
    assert body["segments"]["specs"]
    assert body["attempts"] == {}


def test_generating_one_segment_records_an_attempt_and_leaves_siblings_alone(client):
    run_id = locked(client)
    segments = post(client, f"{BASE}/runs/{run_id}/segments",
                    {}).get_json()["segments"]
    target = segments["specs"][0]["segment_id"]
    before = {r["segment_id"]: r["digest"] for r in segments["specs"]}
    res = post(client, f"{BASE}/runs/{run_id}/segments/{target}/generate", {})
    assert res.status_code == 200
    attempt = res.get_json()["attempt"]
    assert attempt["attempt"] == 1
    assert attempt["model_id"] == "sd-turbo"
    assert attempt["siblings_unchanged"] is True
    after = {r["segment_id"]: r["digest"]
             for r in res.get_json()["run"]["segments"]["specs"]}
    assert before == after


def test_regenerating_is_attempt_two_at_the_next_seed(client):
    run_id = locked(client)
    segments = post(client, f"{BASE}/runs/{run_id}/segments",
                    {}).get_json()["segments"]
    target = segments["specs"][0]
    path = f"{BASE}/runs/{run_id}/segments/{target['segment_id']}/generate"
    a1 = post(client, path, {}).get_json()["attempt"]
    a2 = post(client, path, {}).get_json()["attempt"]
    assert (a1["seed"], a2["seed"]) == (target["seed_base"],
                                        target["seed_base"] + 1)
    assert a1["spec_digest"] == a2["spec_digest"] == target["digest"]


def test_a_gapped_clip_attempt_is_a_recorded_attempt_not_an_error(client,
                                                                  monkeypatch):
    monkeypatch.setattr(sf, "live_segment_dispatch",
                        lambda spec, *, kind, seed, settings=None: {
                            "ok": False, "kind": kind,
                            "capability": sf.CLIP_CAPABILITY,
                            "model_id": None, "seed": seed,
                            "prompt": spec.prompt, "params": {},
                            "artifacts": [], "receipt": None,
                            "gap": {"code": "CAPABILITY_GAP",
                                    "capability": sf.CLIP_CAPABILITY,
                                    "reasons": ["route is 'deferred'"],
                                    "requirement": sf.SEAM_REQUIREMENTS[
                                        sf.CLIP_CAPABILITY]}})
    run_id = locked(client)
    segments = post(client, f"{BASE}/runs/{run_id}/segments",
                    {}).get_json()["segments"]
    target = segments["specs"][0]["segment_id"]
    res = post(client, f"{BASE}/runs/{run_id}/segments/{target}/generate",
               {"kind": "clip"})
    assert res.status_code == 200                  # an attempt, not a failure
    body = res.get_json()
    assert body["ok"] is False
    assert body["attempt"]["gap"]["capability"] == sf.CLIP_CAPABILITY
    assert "studio" in body["attempt"]["gap"]["requirement"]


def test_generating_an_unknown_segment_is_404(client):
    run_id = locked(client)
    post(client, f"{BASE}/runs/{run_id}/segments", {})
    res = post(client, f"{BASE}/runs/{run_id}/segments/nope/generate", {})
    assert res.status_code == 404
    assert res.get_json()["code"] == "SEGMENT_UNKNOWN"


def test_generating_before_compiling_is_409(client):
    run_id = locked(client)
    res = post(client, f"{BASE}/runs/{run_id}/segments/s1-1/generate", {})
    assert res.status_code == 409
    assert res.get_json()["code"] == "SEGMENTS_MISSING"


# ---------------------------------------------------------------------------
# [4] promotion
# ---------------------------------------------------------------------------


def test_promoting_is_201_and_carries_the_actual_refusal(client):
    run_id = locked(client)
    segments = post(client, f"{BASE}/runs/{run_id}/segments",
                    {}).get_json()["segments"]
    target = segments["specs"][0]["segment_id"]
    res = post(client, f"{BASE}/runs/{run_id}/promote",
               {"segment_id": target, "note": "the good one"})
    assert res.status_code == 201
    body = res.get_json()
    assert body["source"]["usable_in"] == "a NEW run only"
    assert "generated during this run" in body["source"]["refused_here"]
    assert "NEW run" in body["next"]


def test_a_promoted_source_is_listed_and_seeds_a_new_run_by_id(client):
    run_id = locked(client)
    post(client, f"{BASE}/runs/{run_id}/segments", {})
    promoted = post(client, f"{BASE}/runs/{run_id}/promote",
                    {"text": "a kettle that never boils"}).get_json()["source"]
    listing = client.get(f"{BASE}/sources").get_json()
    assert [s["source_id"] for s in listing["sources"]] == [promoted["source_id"]]
    # the NEW run names it by id ONLY; the text is read from the source file
    res = post(client, f"{BASE}/runs", {
        "deliverable": "the sequel",
        "sources": [{"source_id": promoted["source_id"],
                     "text": "", "hash": "a lie"}]})
    assert res.status_code == 201
    assert res.get_json()["run"]["snapshot"]["prompts_before_run"] == \
        ["a kettle that never boils"]


def test_promoting_nothing_is_409(client):
    run_id = created(client)
    res = post(client, f"{BASE}/runs/{run_id}/promote", {})
    assert res.status_code == 409
    assert res.get_json()["code"] == "PROMOTE_REFUSED"


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------


def test_every_route_is_under_the_video_prefix_so_it_inherits_the_video_gate():
    app = Flask(__name__)
    app.register_blueprint(routes.script_first_bp)
    rules = [str(r) for r in app.url_map.iter_rules()
             if r.endpoint != "static"]
    assert rules
    assert all(r.startswith("/video/") for r in rules), rules


def test_a_body_that_is_not_an_object_does_not_500(client):
    run_id = created(client)
    res = client.put(f"{BASE}/runs/{run_id}/plot", data="[1, 2, 3]",
                     content_type="application/json")
    assert res.status_code == 422


def test_an_unexpected_exception_is_a_500_naming_itself(client, monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("the disk is on fire")
    monkeypatch.setattr(sf.ScriptFirstRun, "list_runs", staticmethod(explode))
    res = client.get(f"{BASE}/runs")
    assert res.status_code == 500
    assert res.get_json()["code"] == "UNEXPECTED"
    assert "the disk is on fire" in res.get_json()["message"]


def test_a_refusal_body_carries_every_error_in_the_error_alias(client):
    """The console's shared transport keeps only `parsed.error` from a non-2xx
    body. If the validator errors are not in there they never reach a screen."""
    run_id = created(client)
    broken = screenplay_dict()
    broken["scenes"][0]["heading"] = "the kitchen"
    body = put(client, f"{BASE}/runs/{run_id}/screenplay", broken).get_json()
    assert body["error"].startswith("ARTIFACT_INVALID:")
    for problem in body["errors"]:
        assert problem in body["error"]


def test_a_refusal_is_journalled_on_the_run_it_was_about(client, monkeypatch):
    monkeypatch.setattr(sf, "bind_llm", lambda *a, **k: (lambda p: "not json"))
    run_id = created(client)
    post(client, f"{BASE}/runs/{run_id}/plot", {})
    run = client.get(f"{BASE}/runs/{run_id}").get_json()["run"]
    assert run["last_refusal"]["code"] == "AUTHORING_GAP"
    assert run["last_refusal"]["errors"]
    assert run["last_refusal"]["detail"]["gap"]["raw"] == "not json"


def test_a_journalled_refusal_is_cleared_by_the_next_success(client, monkeypatch):
    monkeypatch.setattr(sf, "bind_llm", lambda *a, **k: (lambda p: "not json"))
    run_id = created(client)
    post(client, f"{BASE}/runs/{run_id}/plot", {})
    assert client.get(f"{BASE}/runs/{run_id}").get_json()["run"]["last_refusal"]
    put(client, f"{BASE}/runs/{run_id}/screenplay", screenplay_dict())
    assert client.get(f"{BASE}/runs/{run_id}").get_json()["run"]["last_refusal"] is None
