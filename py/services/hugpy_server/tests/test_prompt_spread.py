"""STUDIO SPREAD — mode="spread" / mode="negative" / typed context (SPEC §1a-1e).

What these lock down, in the order the spec argues them:

  §1a  a spread is ONE generator call for N segments. The old per-row Generate
       drew a fresh steering set per call, so six rows produced six unrelated
       worlds; the fix is not "fewer calls", it is ONE call with ONE shared
       world. The call-count assertion below IS the feature.
  §1a  a LOCKED row is untouchable. If a model returns a segment the user did
       not select, it is DROPPED with a warning — otherwise the selection
       checkbox is a lie and Generate silently eats work the user kept.
  §1b  a negative prompt is an exclusion list, not prose. Reusing the scene
       system prompt returns a poem.
  §1c  structured row state rides TYPED fields, not the free-form hint, and
       joint modes reach the model as SENTENCES — a model shown the token
       "vace_extend" guesses what it means.
  §1e  constrained JSON in, defensive parse out: unparseable => an honest 502
       carrying the raw text, never a fabricated segment.

⚠ EXECUTOR MOCKING. ``from abstract_hugpy_dev.managers import dispatch`` returns
the INNER dispatch.dispatch module, not the package — patching that attribute
patches the wrong object and the test quietly runs REAL inference. The route
reaches ``execute_prompt`` through ``flask_app.app.functions.imports``, so that
is the module patched here (see ``_patch_executor``); tests/test_no_think.py
documents the same landmine for the dispatch-side seam.

Run (pytest or as a script):
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python -m pytest tests/test_prompt_spread.py -q
"""
from __future__ import annotations

import json
import logging
import os
import sys

logging.disable(logging.INFO)

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import importlib  # noqa: E402

import pytest  # noqa: E402
from flask import Flask  # noqa: E402

from hugpy_video.intel import prompt_spread as PS
from hugpy_video.intel import prompt_seeds as SEEDS

vr = importlib.import_module("hugpy_server.app.routes.video_routes")
_IMPORTS_MOD = importlib.import_module("hugpy_engine.dispatch.dispatch")

app = Flask(__name__)
app.register_blueprint(vr.video_bp)
client = app.test_client()


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
class _Calls(list):
    """Records every execute_prompt kwargs dict. len() is the call count."""

    @property
    def messages(self):
        assert self, "the executor was never called"
        return self[-1]["messages"]

    @property
    def user(self):
        return self.messages[-1]["content"]

    @property
    def system(self):
        return self.messages[0]["content"]


def _patch_executor(monkeypatch, reply_text, ok=True, error=None, raises=None):
    """Stub the ONE executor the assist route actually calls.

    ``_assist_execute`` does ``from ..functions.imports import execute_prompt``,
    which reads the attribute off that package at call time — so patching it
    there is what the route sees. Patching ``managers.dispatch`` instead would
    leave the real function in place and run live inference."""
    calls = _Calls()

    def fake_execute_prompt(*args, **kwargs):
        calls.append(kwargs)
        if raises is not None:
            raise raises
        return {"ok": ok, "text": reply_text, "error": error,
                "model_key": "resolved-model-x"}

    monkeypatch.setattr(_IMPORTS_MOD, "execute_prompt", fake_execute_prompt)
    return calls


def _segments_reply(ids, extra=None):
    payload = {"segments": [
        {"segment_id": sid, "operation": "generate_from_direction",
         "prompt": f"A shot for {sid}.", "negative": "blurry, watermark",
         "continuity_note": f"follows on from before {sid}",
         "directions_used": [0]}
        for sid in ids
    ], "invented_identity_attributes": [], "warnings": []}
    if extra:
        payload["segments"].extend(extra)
    return json.dumps(payload)


def _spread_body(n_targets=4, n_fixed=2, **over):
    body = {
        "mode": "spread",
        "movie_query": "a diver finds something under the ice",
        "style_bible": {"world": "arctic research station",
                        "subject": "a lone diver",
                        "visual_style": "grainy 16mm"},
        "fixed_segments": [
            {"segment_id": f"segment-locked-{i}", "prompt": f"locked shot {i}",
             "joint_mode": "still", "index": i}
            for i in range(n_fixed)
        ],
        "target_segments": [
            {"segment_id": f"segment-{i}", "direction": "make it colder",
             "joint_mode": "vace_extend", "index": n_fixed + i}
            for i in range(n_targets)
        ],
        "global_negative": ["watermark", "text overlay"],
        "steering_seed": 184392,
    }
    body.update(over)
    return body


# --------------------------------------------------------------------------- #
# §1a — REQUEST VALIDATION
# --------------------------------------------------------------------------- #
def test_spread_requires_target_segments():
    r = client.post("/video/prompt/assist", json={"mode": "spread",
                                                  "movie_query": "x"})
    assert r.status_code == 400
    assert "target_segments" in r.get_json()["error"]


def test_spread_rejects_duplicate_segment_ids():
    body = _spread_body(n_targets=1, n_fixed=0)
    body["fixed_segments"] = [{"segment_id": "segment-0", "prompt": "p"}]
    r = client.post("/video/prompt/assist", json=body)
    assert r.status_code == 400
    assert "duplicate" in r.get_json()["error"]


def test_spread_rejects_a_bad_joint_mode():
    body = _spread_body(n_targets=1, n_fixed=0)
    body["target_segments"][0]["joint_mode"] = "crossfade"
    r = client.post("/video/prompt/assist", json=body)
    assert r.status_code == 400
    assert "joint_mode" in r.get_json()["error"]


def test_spread_rejects_a_bad_global_negative():
    body = _spread_body(n_targets=1, n_fixed=0, global_negative=[{"nope": 1}])
    r = client.post("/video/prompt/assist", json=body)
    assert r.status_code == 400


def test_unknown_mode_is_still_a_400():
    r = client.post("/video/prompt/assist", json={"mode": "sideways"})
    assert r.status_code == 400
    assert "spread" in r.get_json()["error"]


# --------------------------------------------------------------------------- #
# §1a — ONE CALL PER SPREAD (the feature itself)
# --------------------------------------------------------------------------- #
def test_a_spread_of_four_segments_is_exactly_one_generator_call(monkeypatch):
    ids = [f"segment-{i}" for i in range(4)]
    calls = _patch_executor(monkeypatch, _segments_reply(ids))
    r = client.post("/video/prompt/assist", json=_spread_body(4))
    assert r.status_code == 200, r.get_json()
    assert len(calls) == 1, f"{len(calls)} calls for 4 segments — that is N rolls"
    assert len(r.get_json()["segments"]) == 4


def test_the_whole_timeline_including_locked_rows_is_in_the_one_brief(monkeypatch):
    ids = [f"segment-{i}" for i in range(2)]
    calls = _patch_executor(monkeypatch, _segments_reply(ids))
    client.post("/video/prompt/assist", json=_spread_body(2, n_fixed=2))
    user = calls.user
    assert "LOCKED — do not rewrite" in user
    assert "REGENERATE" in user
    assert "locked shot 0" in user and "locked shot 1" in user
    assert "SEGMENT 1 of 4" in user      # timeline-ordered, locked included


def test_one_shared_steering_set_and_a_per_segment_beat(monkeypatch):
    """The coherence mechanism: ONE world, only the beat varies."""
    ids = [f"segment-{i}" for i in range(3)]
    calls = _patch_executor(monkeypatch, _segments_reply(ids))
    r = client.post("/video/prompt/assist", json=_spread_body(3, n_fixed=0))
    user = calls.user
    assert user.count("SHARED WORLD") == 1
    beats = [ln for ln in user.splitlines() if "beat for this shot:" in ln]
    assert len(beats) == 3 and len(set(beats)) == 3, beats
    # the steering set is echoed back so a spread can be pinned + replayed
    assert set(r.get_json()["steering"]) >= {"subject", "setting", "mood"}
    assert r.get_json()["steering_seed"] == 184392


def test_the_same_seed_rebuilds_the_same_world():
    a = SEEDS.spread_axes("movie", seed=99)
    b = SEEDS.spread_axes("movie", seed=99)
    c = SEEDS.spread_axes("movie", seed=100)
    assert a == b, "a seeded spread must be replayable"
    assert a != c


def test_beats_are_ordered_and_distinct_across_a_short_movie():
    beats = [SEEDS.beat_for_index(i, 4) for i in range(4)]
    assert len(set(beats)) == 4, beats


# --------------------------------------------------------------------------- #
# §1a — LOCKED ROWS ARE UNTOUCHABLE
# --------------------------------------------------------------------------- #
def test_a_returned_locked_segment_is_dropped_not_applied(monkeypatch):
    ids = [f"segment-{i}" for i in range(2)]
    rogue = {"segment_id": "segment-locked-0", "operation": "generate",
             "prompt": "I rewrote the row you locked", "negative": "",
             "continuity_note": ""}
    _patch_executor(monkeypatch, _segments_reply(ids, extra=[rogue]))
    r = client.post("/video/prompt/assist", json=_spread_body(2, n_fixed=2))
    data = r.get_json()
    returned = {s["segment_id"] for s in data["segments"]}
    assert returned == set(ids), returned
    assert "segment-locked-0" not in returned
    assert any("not selected" in w for w in data["warnings"]), data["warnings"]


def test_a_target_the_model_skipped_is_reported_not_invented(monkeypatch):
    _patch_executor(monkeypatch, _segments_reply(["segment-0"]))
    r = client.post("/video/prompt/assist", json=_spread_body(3, n_fixed=0))
    data = r.get_json()
    assert [s["segment_id"] for s in data["segments"]] == ["segment-0"]
    assert data["missing_segments"] == ["segment-1", "segment-2"]
    assert any("did not write" in w for w in data["warnings"])


# --------------------------------------------------------------------------- #
# §1e — PARSING: fences, think blocks, and honest failure
# --------------------------------------------------------------------------- #
def test_a_fenced_reply_is_still_parsed(monkeypatch):
    body = "```json\n" + _segments_reply(["segment-0"]) + "\n```"
    _patch_executor(monkeypatch, body)
    r = client.post("/video/prompt/assist", json=_spread_body(1, n_fixed=0))
    assert r.status_code == 200
    assert r.get_json()["segments"][0]["prompt"].startswith("A shot for")


def test_think_is_stripped_before_the_json_is_scavenged(monkeypatch):
    reply = ("<think>Let me consider {this} carefully</think>"
             + _segments_reply(["segment-0"]))
    calls = _patch_executor(monkeypatch, reply)
    r = client.post("/video/prompt/assist", json=_spread_body(1, n_fixed=0))
    data = r.get_json()
    assert r.status_code == 200, data
    assert data["segments"][0]["segment_id"] == "segment-0"
    assert data["reasoning"] == "Let me consider {this} carefully"
    assert data["thinking_suppressed"] is True
    # half 1 of the seam rode the query too
    assert "/no_think" in calls.user


def test_a_preamble_that_cannot_cover_the_shots_is_an_honest_502(monkeypatch):
    # A single throwaway line for TWO selected shots is not a coherent 2-scene
    # reply: one paragraph cannot cover two shots, so it fails honestly (with the
    # raw) rather than pasting "Sure! Here are your shots" into a shot. This is
    # the guard that keeps the tolerant divvy from fabricating.
    _patch_executor(monkeypatch, "Sure! Here are some lovely shots for you.")
    r = client.post("/video/prompt/assist", json=_spread_body(2, n_fixed=0))
    assert r.status_code == 502
    data = r.get_json()
    assert "segments" not in data, "a parse failure must never fabricate segments"
    assert "lovely shots" in data["raw"]


def test_two_unlabelled_paragraphs_are_divvied_into_the_two_shots(monkeypatch):
    # THE OPERATOR'S CASE (2026-07-31): "it was asked to generate 2 scenes ... it
    # should do that, which it probably did. it's the parser on this end that is
    # no good." The model wrote two coherent paragraphs with NO JSON and NO
    # labels; the old parser threw them away demanding a JSON envelope. Now they
    # are divvied onto the two selected shots in timeline order.
    reply = ("A lone figure crosses a rain-slick rooftop at dawn, silver light "
             "on wet tiles.\n\n"
             "The camera pushes in as sparks scatter and the city wakes below.")
    _patch_executor(monkeypatch, reply)
    r = client.post("/video/prompt/assist", json=_spread_body(2, n_fixed=0))
    assert r.status_code == 200, r.get_json()
    segs = r.get_json()["segments"]
    assert [s["segment_id"] for s in segs] == ["segment-0", "segment-1"]
    assert segs[0]["prompt"].startswith("A lone figure")
    assert segs[1]["prompt"].startswith("The camera pushes in")


def test_segments_json_kept_INSIDE_a_think_block_is_now_recovered(monkeypatch):
    # THE WIN: wazimondo~Qwen3.6-35B (the default spread generator) is a
    # reasoning model that wraps its whole answer in <think>. Before, that was a
    # hard 502 "only reasoning and no output". Now the JSON inside the think
    # block is salvaged and parses to real segments.
    inner = _segments_reply(["segment-0"])
    _patch_executor(monkeypatch, f"<think>{inner}</think>")
    r = client.post("/video/prompt/assist", json=_spread_body(1, n_fixed=0))
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["segments"][0]["segment_id"] == "segment-0"


def test_a_worker_failure_is_a_502_not_a_500(monkeypatch):
    _patch_executor(monkeypatch, "", raises=RuntimeError("no live worker"))
    r = client.post("/video/prompt/assist", json=_spread_body(1, n_fixed=0))
    assert r.status_code == 502


def test_an_unknown_model_key_is_a_400(monkeypatch):
    _patch_executor(monkeypatch, "", raises=KeyError("Unknown model_key"))
    r = client.post("/video/prompt/assist", json=_spread_body(1, n_fixed=0))
    assert r.status_code == 400


def test_spread_uses_the_spec_generator_by_default_and_honours_an_override(monkeypatch):
    calls = _patch_executor(monkeypatch, _segments_reply(["segment-0"]))
    client.post("/video/prompt/assist", json=_spread_body(1, n_fixed=0))
    assert calls[-1]["model_key"] == "Qwen2.5-7B-Instruct-GGUF"
    client.post("/video/prompt/assist",
                json=_spread_body(1, n_fixed=0, model="some-other-model"))
    assert calls[-1]["model_key"] == "some-other-model"


def test_provenance_names_requested_and_resolved(monkeypatch):
    _patch_executor(monkeypatch, _segments_reply(["segment-0"]))
    r = client.post("/video/prompt/assist", json=_spread_body(1, n_fixed=0))
    data = r.get_json()
    assert data["model_requested"] == "Qwen2.5-7B-Instruct-GGUF"
    assert data["model_resolved"] == "resolved-model-x"


# --------------------------------------------------------------------------- #
# §1b — NEGATIVE MODE FRAMING
# --------------------------------------------------------------------------- #
def test_negative_mode_uses_its_own_exclusion_framing(monkeypatch):
    calls = _patch_executor(monkeypatch, "deformed hands, flicker, watermark")
    r = client.post("/video/prompt/assist", json={
        "mode": "negative", "subject": "a diver under the ice",
        "context": {"kind": "movie"}})
    assert r.status_code == 200, r.get_json()
    system = calls.system
    assert "EXCLUSION LIST" in system
    assert "comma-separated" in system
    # it must NOT be the scene-prose framing — that returns a poem
    assert "expert image-prompt engineer" not in system
    assert "expert video-prompt engineer" not in system
    data = r.get_json()
    assert data["negative"] == "deformed hands, flicker, watermark"
    assert data["prompt"] == data["negative"]     # no existing key moved


def test_negative_mode_folds_in_the_existing_negative_and_the_shot(monkeypatch):
    calls = _patch_executor(monkeypatch, "blurry, jpeg artifacts")
    client.post("/video/prompt/assist", json={
        "mode": "negative", "draft": "blurry",
        "context": {"kind": "movie",
                    "segment": {"prompt": "a diver under the ice"}}})
    user = calls.user
    assert "a diver under the ice" in user
    assert "blurry" in user


def test_negative_mode_strips_think(monkeypatch):
    _patch_executor(monkeypatch, "<think>hmm</think>deformed hands, flicker")
    r = client.post("/video/prompt/assist", json={"mode": "negative"})
    data = r.get_json()
    assert data["negative"] == "deformed hands, flicker"
    assert data["reasoning"] == "hmm"


# --------------------------------------------------------------------------- #
# §1c — TYPED CONTEXT
# --------------------------------------------------------------------------- #
def test_joint_modes_reach_the_model_in_plain_language(monkeypatch):
    calls = _patch_executor(monkeypatch, "an enriched prompt")
    client.post("/video/prompt/assist", json={
        "mode": "detail", "draft": "a diver",
        "context": {"kind": "movie",
                    "segment": {"joint_mode": "vace_extend", "branch_frame": 37},
                    "previous_segment": {"prompt": "the descent",
                                         "joint_mode": "still"},
                    "next_segment": {"prompt": "the surface", "joint_mode": "cut"}}})
    user = calls.user
    assert "vace_extend" not in user, "the raw token must never reach the model"
    assert "carrying its motion" in user                    # vace_extend
    assert "no motion is carried" in user                   # still
    assert "hard cut" in user                               # cut
    assert "frame 37" in user
    assert "THE SHOT BEFORE THIS ONE" in user and "THE SHOT AFTER THIS ONE" in user


def test_typed_context_is_validated_not_swallowed():
    r = client.post("/video/prompt/assist", json={
        "mode": "detail", "draft": "a diver",
        "context": {"segment": {"branch_frame": -3}}})
    assert r.status_code == 400
    assert "branch_frame" in r.get_json()["error"]


def test_hint_stays_free_form_and_still_works(monkeypatch):
    calls = _patch_executor(monkeypatch, "an enriched prompt")
    client.post("/video/prompt/assist", json={
        "mode": "detail", "draft": "a diver",
        "context": {"kind": "movie", "hint": "keep it under the ice"}})
    assert "keep it under the ice" in calls.user


def test_detail_without_typed_context_is_unchanged(monkeypatch):
    """Back-compat: a caller that sends only {kind, hint} must get the message it
    always got — no preface, no new sections."""
    calls = _patch_executor(monkeypatch, "an enriched prompt")
    r = client.post("/video/prompt/assist", json={
        "mode": "detail", "draft": "a diver", "context": {"kind": "movie"}})
    user = calls.user
    for marker in ("THE SHOT BEFORE THIS ONE", "THIS SHOT (current state)",
                   "LOCKED IDENTITY"):
        assert marker not in user
    assert r.get_json()["prompt"] == "an enriched prompt"


# --------------------------------------------------------------------------- #
# §1e — LOCKED IDENTITY
# --------------------------------------------------------------------------- #
def test_identity_block_carries_the_do_not_invent_list(monkeypatch):
    calls = _patch_executor(monkeypatch, _segments_reply(["segment-0"]))
    body = _spread_body(1, n_fixed=0)
    body["context"] = {"kind": "movie", "identity_profile": {
        "slug": "alex", "name": "Alex", "notes": "a station diver",
        "reference_images": ["/x/a.png", "/x/b.png"]}}
    r = client.post("/video/prompt/assist", json=body)
    assert r.status_code == 200, r.get_json()
    user = calls.user
    assert "LOCKED IDENTITY" in user
    assert "Alex" in user and "a station diver" in user
    assert "DO NOT INVENT" in user
    for attr in ("age", "gender", "clothing", "ethnicity"):
        assert attr in user
    assert "reference images on file: 2" in user


def test_identity_accepts_the_specs_own_shape():
    block = PS.render_identity_block(PS._validate_identity({
        "identity_id": "alex", "name": "Alex",
        "reference_asset_ids": ["identity-alex-front"],
        "locked_description": "Existing identity profile",
        "do_not_invent": ["age", "gender", "clothing", "ethnicity"]}))
    assert "identity alex" in block
    assert "Existing identity profile" in block


def test_identity_without_a_name_is_rejected():
    with pytest.raises(PS.SpreadError):
        PS.validate_context({"identity_profile": {"slug": "alex"}})


# --------------------------------------------------------------------------- #
# the shared JSON scavenger (one definition, three callers)
# --------------------------------------------------------------------------- #
def test_scavenger_is_shared_not_a_third_copy():
    from hugpy_engine.utils import json_scavenge
    from hugpy_curation.review import judge
    assert judge._extract_json('noise {"a": 1} tail') == {"a": 1}
    assert json_scavenge.extract_json_object("no json here") is None
    assert json_scavenge.extract_json_array('```json\n[1,2]\n```') == [1, 2]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
