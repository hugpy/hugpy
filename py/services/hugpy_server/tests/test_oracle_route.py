"""k90b — POST /oracle/route: the intent-inference table, route resolution over
the k90a catalog (providers stubbed — no GPU/network/workers), execution through
a monkeypatched dispatch seam, the mandatory deterministic Scorecard, and the
gap/deferred wire shapes.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_route.py -q
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys

logging.disable(logging.INFO)  # silence the models_config registry chatter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from hugpy_oracle import catalog, router, runtime, scorecard
from hugpy_oracle.contracts import (
    ArtifactKind,
    FailureClass,
    GoalSpec,
    InputKind,
    InputRef,
    RepairCode,
)


def _goal(prompt="hello", inputs=(), capability=None):
    return GoalSpec(objective=prompt, raw_prompt=prompt,
                    inputs=tuple(inputs), capability=capability)


def _ref(kind, ref="x"):
    return InputRef(kind=InputKind(kind), ref=ref)


_ROWS = {
    "whisper-x":  {"tasks": ["automatic-speech-recognition"],
                   "framework": "transformers"},
    "qwen-chat":  {"tasks": ["text-generation"], "framework": "gguf"},
    "qwen-vl":    {"tasks": ["image-text-to-text"], "framework": "gguf"},
    "sdxl":       {"tasks": ["text-to-image"], "framework": "transformers"},
    "vit-cls":    {"tasks": ["image-classification"], "framework": "transformers"},
}


def _stub_catalog(monkeypatch, rows=_ROWS, workers=({"id": "w1"},), capable=True,
                  central=True, blocked=frozenset()):
    monkeypatch.setattr(catalog, "_legacy_registry_rows", lambda: dict(rows))
    monkeypatch.setattr(catalog, "_online_workers",
                        lambda: None if workers is None else list(workers))
    monkeypatch.setattr(catalog, "_worker_task_capable", lambda w, t: bool(capable))
    monkeypatch.setattr(catalog, "_central_task_available", lambda t: central)
    monkeypatch.setattr(catalog, "_blocked_model_keys", lambda: set(blocked))


# ---------------------------------------------------------------------------
# The mapping table: every executable capability's task maps back through
# LEGACY_TASK_CAPABILITY — the router and the catalog can never disagree.
# ---------------------------------------------------------------------------


def test_capability_task_table_is_inverse_consistent():
    for cap, task in router.CAPABILITY_TASK.items():
        if cap in catalog.SPEECH_CAPABILITY_TASK:
            # k98's speech family (folded in via **catalog.SPEECH_CAPABILITY_TASK)
            # is a deliberately separate mapping — its tasks ("text-to-speech",
            # and "automatic-speech-recognition" reused for the word_timestamps
            # variant) do not invert back through LEGACY_TASK_CAPABILITY by
            # design (catalog.py's module docstring explains why). Check it
            # against its own source table instead.
            assert catalog.SPEECH_CAPABILITY_TASK[cap] == task
            continue
        assert catalog.LEGACY_TASK_CAPABILITY.get(task) == cap
    # and every legacy + speech capability is dispatchable
    assert set(router.CAPABILITY_TASK) == (
        set(catalog.LEGACY_TASK_CAPABILITY.values())
        | set(catalog.SPEECH_CAPABILITY_TASK))


# ---------------------------------------------------------------------------
# Intent inference — the deterministic table.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt,inputs,expected", [
    ("what is in this picture?", [_ref("image")], "image.understand"),
    ("describe the scene", [_ref("image")], "image.understand"),
    ("make it a watercolor painting", [_ref("image")], "image.transform"),
    ("restyle this photo as pixel art", [_ref("image")], "image.transform"),
    # transform verb BUT question-like -> understand wins
    ("what would make it a watercolor?", [_ref("image")], "image.understand"),
    ("get the words out", [_ref("audio")], "audio.transcribe"),
    ("transcribe", [_ref("video")], "audio.transcribe"),
    ("what does this page say", [_ref("url", "https://x.test")], "web.fetch"),
    ("summarize this report for me", [], "text.summarize"),
    ("tl;dr the following", [], "text.summarize"),
    ("hello there", [], "text.chat"),
])
def test_inference_table(prompt, inputs, expected):
    cap, why = router.infer_capability(_goal(prompt, inputs))
    assert cap == expected, why
    assert why  # the reason is recorded, never blank


def test_explicit_capability_wins_over_inference(monkeypatch):
    _stub_catalog(monkeypatch)
    goal = _goal("summarize this please", capability="text.chat")
    route = router.resolve_route(goal)
    assert route.capability == "text.chat"
    assert route.inferred is False
    assert route.inference_reason == "explicit capability in request"


# ---------------------------------------------------------------------------
# resolve_route — model choice + branches.
# ---------------------------------------------------------------------------


def test_resolve_route_only_eligible_model(monkeypatch):
    _stub_catalog(monkeypatch)
    route = router.resolve_route(_goal("hi", [_ref("audio", "/tmp/a.wav")]))
    assert route.execution == "execute"
    assert route.capability == "audio.transcribe"
    assert route.task == "automatic-speech-recognition"
    assert route.model_id == "whisper-x"
    assert route.model_rationale == "only-eligible"
    assert route.inferred is True


def test_resolve_route_requested_model_wins(monkeypatch):
    _stub_catalog(monkeypatch)
    route = router.resolve_route(_goal("hi"), requested_model="qwen-chat")
    assert route.model_id == "qwen-chat"
    assert route.model_rationale == "requested"


def test_resolve_route_requested_model_outside_capability_refuses(monkeypatch):
    _stub_catalog(monkeypatch)
    with pytest.raises(router.RouteRefusal):
        router.resolve_route(_goal("hi"), requested_model="whisper-x")


def test_resolve_route_default_when_multiple_models_and_selection_silent(monkeypatch):
    """TODO-4: with >1 eligible model the evidence selector decides; the
    legacy TASK_DEFAULTS branch is only the fallback when it has no opinion."""
    rows = dict(_ROWS, **{"qwen-chat-2": {"tasks": ["text-generation"],
                                          "framework": "gguf"}})
    _stub_catalog(monkeypatch, rows=rows)
    monkeypatch.setattr(router, "_task_default_model", lambda t: "qwen-chat-2")
    monkeypatch.setattr(router, "_select_model", lambda goal, cap, view: (None, "", ("selection disabled",)))
    route = router.resolve_route(_goal("hi"))
    assert route.model_id == "qwen-chat-2"
    assert route.model_rationale == "default"
    assert "selection disabled" in route.reasons


def test_resolve_route_uses_evidence_selection_when_multiple_models(monkeypatch):
    rows = dict(_ROWS, **{"qwen-chat-2": {"tasks": ["text-generation"],
                                          "framework": "gguf"}})
    _stub_catalog(monkeypatch, rows=rows)
    monkeypatch.setattr(router, "_task_default_model", lambda t: "qwen-chat-2")
    from hugpy_oracle import selection
    sel = selection.Selector(ledger=None, get_matrix=lambda: None)
    monkeypatch.setattr(selection, "_PROCESS_SELECTOR", sel)
    route = router.resolve_route(_goal("hi"))
    assert route.model_id in route.model_ids
    assert route.model_rationale.startswith("selected:")
    assert any(r.startswith("selection: ") for r in route.reasons)
    monkeypatch.setattr(selection, "_PROCESS_SELECTOR", None)


def test_resolve_route_unknown_capability_is_gap_not_keyerror(monkeypatch):
    _stub_catalog(monkeypatch)
    route = router.resolve_route(_goal("hi", capability="quantum.flux"))
    assert route.execution == "gap"
    assert any("unknown capability" in r for r in route.reasons)


def test_resolve_route_ineligible_echoes_catalog_reasons(monkeypatch):
    _stub_catalog(monkeypatch, workers=(), central=False)
    route = router.resolve_route(_goal("hi", capability="audio.transcribe"))
    assert route.execution == "gap"
    joined = " | ".join(route.reasons)
    assert "no online worker registered" in joined
    assert "central cannot serve" in joined


def test_resolve_route_video_is_deferred_with_menu(monkeypatch):
    _stub_catalog(monkeypatch)
    monkeypatch.setattr(router, "_studio_menu", lambda: "t2v (clip-t2v-480p)")
    route = router.resolve_route(_goal("hi", capability="video.generate.t2v"))
    assert route.execution == "deferred"
    assert route.alternatives == "t2v (clip-t2v-480p)"
    assert route.reasons  # the studio verdict/advisory rides along


def test_resolve_route_deterministic_amenity(monkeypatch):
    _stub_catalog(monkeypatch, rows={})
    route = router.resolve_route(_goal("hi", capability="doc.extract"))
    assert route.execution == "execute"
    assert route.model_id is None
    assert route.model_rationale == "deterministic-local"
    assert route.placement == "local"


# ---------------------------------------------------------------------------
# k98b — speech capabilities stitched through CAPABILITY_TASK /
# capability_params. audio.tts and audio.speaker_similarity resolve through
# the router with a typed, specific diagnosis rather than falling into an
# "unknown capability" / router-drift gap; audio.transcribe.word_timestamps
# tracks its base capability's real eligibility now that the whisper
# passthrough chain (catalog._word_timestamps_wired) is closed.
# ---------------------------------------------------------------------------


def test_resolve_route_audio_tts_names_missing_worker_seat_not_unknown(monkeypatch):
    _stub_catalog(monkeypatch)  # default rows carry no chatterbox-marker row
    route = router.resolve_route(_goal("say hello", capability="audio.tts"))
    assert route.execution == "gap"
    joined = " | ".join(route.reasons).lower()
    # A typed, specific diagnosis — never the bare "unknown capability" text
    # an unmapped/misspelled name would get (test_resolve_route_unknown_
    # capability_is_gap_not_keyerror), and never a KeyError/drift alarm now
    # that CAPABILITY_TASK carries the SPEECH_CAPABILITY_TASK rows.
    assert "unknown capability" not in joined
    assert "worker" in joined


def test_resolve_route_audio_speaker_similarity_is_a_declared_gap(monkeypatch):
    _stub_catalog(monkeypatch)
    route = router.resolve_route(_goal("compare voices",
                                       capability="audio.speaker_similarity"))
    assert route.execution == "gap"
    joined = " | ".join(route.reasons).lower()
    assert "unknown capability" not in joined
    assert "speaker-embedding" in joined
    # Deliberately absent from CAPABILITY_TASK/SPEECH_CAPABILITY_TASK (catalog.py:
    # "inventing one would be the exact phantom this module exists to delete") —
    # confirm the router never fabricates a dispatch task for it.
    assert "audio.speaker_similarity" not in router.CAPABILITY_TASK


def test_resolve_route_word_timestamps_tracks_audio_transcribe_when_eligible(monkeypatch):
    _stub_catalog(monkeypatch)  # whisper-x row + capable worker -> eligible
    base = router.resolve_route(_goal("x", capability="audio.transcribe"))
    route = router.resolve_route(
        _goal("x", capability="audio.transcribe.word_timestamps"))
    assert base.execution == "execute"
    assert route.execution == "execute"
    assert route.task == "automatic-speech-recognition"
    # The one fixed param the capability implies — catalog.capability_params,
    # merged onto the decision for whoever builds the actual dispatch kwargs.
    assert route.dispatch_params == {"word_timestamps": True}


def test_resolve_route_word_timestamps_tracks_audio_transcribe_when_ineligible(monkeypatch):
    _stub_catalog(monkeypatch, workers=(), central=False)
    base = router.resolve_route(_goal("x", capability="audio.transcribe"))
    route = router.resolve_route(
        _goal("x", capability="audio.transcribe.word_timestamps"))
    assert base.execution == "gap"
    assert route.execution == "gap"


def test_route_audio_tts_wire_shape_is_typed_ineligible(monkeypatch):
    """POST /oracle/route with an audio.tts goal: a 422 gap response, but one
    that names the missing worker seat rather than reading as an unmapped
    capability. NOTE: the wire's scorecard.repair_code is still the generic
    RepairCode.CAPABILITY_GAP — that classification lives in scorecard.py,
    outside this task's file scope — but the reasons carried on route/
    scorecard are the specific, typed diagnosis, never a bare label."""
    client = _client(monkeypatch)
    resp = client.post("/oracle/route",
                       json={"prompt": "say hello", "capability": "audio.tts"})
    assert resp.status_code == 422
    body = resp.get_json()
    assert body["ok"] is False
    assert body["route"]["execution"] == "gap"
    joined = " | ".join(body["route"]["reasons"]).lower()
    assert "unknown capability" not in joined
    assert "worker" in joined
    assert body["scorecard"]["repair_code"] == "capability_gap"


# ---------------------------------------------------------------------------
# runtime.execute_route — dispatch monkeypatched, no GPU/network.
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, **payload):
        self._payload = payload

    def model_dump(self):
        return dict(self._payload)


def _exec_route(monkeypatch, capability="text.chat", **goal_kw):
    _stub_catalog(monkeypatch)
    goal = _goal(goal_kw.pop("prompt", "hello"), goal_kw.pop("inputs", ()),
                 capability=capability)
    return goal, router.resolve_route(goal)


def _patch_dispatch(monkeypatch, fn):
    monkeypatch.setattr(runtime, "_dispatch", fn)
    monkeypatch.setattr(runtime, "_normalized_kwargs",
                        lambda task, body: dict(body, task=task))


def test_execute_route_happy_path(monkeypatch):
    goal, route = _exec_route(monkeypatch)
    seen = {}

    def fake_dispatch(kwargs):
        seen.update(kwargs)
        return _Result(ok=True, text="the answer", model_key="qwen-chat")

    _patch_dispatch(monkeypatch, fake_dispatch)
    artifacts, receipt = runtime.execute_route(goal, route)

    assert seen["task"] == "text-generation"
    assert seen["model_key"] == "qwen-chat"
    assert seen["prompt"] == "hello"

    assert len(artifacts) == 1
    art = artifacts[0]
    assert art["kind"] == "text" and art["text"] == "the answer"
    assert art["sha256"] == hashlib.sha256(b"the answer").hexdigest()

    assert receipt.capability == "text.chat"
    assert receipt.model_id == "qwen-chat"
    assert receipt.failure is None
    assert receipt.retries == 0
    assert receipt.duration_s >= 0
    assert receipt.started_at and receipt.ended_at
    assert receipt.request_dict()["task"] == "text-generation"
    assert [a.uri for a in receipt.artifacts] == [art["uri"]]


def test_execute_route_retries_once_on_worker_unavailable(monkeypatch):
    goal, route = _exec_route(monkeypatch)
    calls = {"n": 0}

    def flaky(kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("connection refused")
        return _Result(ok=True, text="second try")

    _patch_dispatch(monkeypatch, flaky)
    artifacts, receipt = runtime.execute_route(goal, route)
    assert calls["n"] == 2
    assert receipt.retries == 1
    assert receipt.failure is None
    assert artifacts[0]["text"] == "second try"
    assert any("retried once" in w for w in receipt.warnings)


def test_execute_route_worker_unavailable_after_retry(monkeypatch):
    goal, route = _exec_route(monkeypatch)

    def dead(kwargs):
        raise ConnectionError("worker unreachable")

    _patch_dispatch(monkeypatch, dead)
    artifacts, receipt = runtime.execute_route(goal, route)
    assert artifacts == []
    assert receipt.retries == 1
    assert receipt.failure is FailureClass.WORKER_UNAVAILABLE
    assert receipt.log_excerpt


def test_execute_route_timeout_is_not_retried(monkeypatch):
    goal, route = _exec_route(monkeypatch)
    calls = {"n": 0}

    def slow(kwargs):
        calls["n"] += 1
        raise TimeoutError("request timed out after 300s")

    _patch_dispatch(monkeypatch, slow)
    _, receipt = runtime.execute_route(goal, route)
    assert calls["n"] == 1
    assert receipt.retries == 0
    assert receipt.failure is FailureClass.TIMEOUT


def test_execute_route_runner_not_ok_is_runner_error(monkeypatch):
    goal, route = _exec_route(monkeypatch)
    _patch_dispatch(monkeypatch,
                    lambda kwargs: _Result(ok=False, error="model exploded"))
    artifacts, receipt = runtime.execute_route(goal, route)
    assert receipt.failure is FailureClass.RUNNER_ERROR
    assert "model exploded" in " ".join(receipt.log_excerpt)


def test_build_request_body_shapes(monkeypatch):
    _stub_catalog(monkeypatch)
    # image.understand: file + prompt
    goal = _goal("what is this?", [_ref("image", "/tmp/x.png")],
                 capability="image.understand")
    body = runtime.build_request_body(goal, router.resolve_route(goal))
    assert body["file"] == "/tmp/x.png" and body["prompt"] == "what is this?"
    # similarity needs two texts
    goal = _goal("compare", [_ref("text", "a")], capability="text.similarity")
    route = router.RouteDecision(capability="text.similarity",
                                 execution="execute", task="sentence-similarity")
    with pytest.raises(runtime.GoalShapeError):
        runtime.build_request_body(goal, route)
    # missing image input is a typed shape error, pre-dispatch
    goal = _goal("classify", capability="image.classify")
    route = router.RouteDecision(capability="image.classify",
                                 execution="execute", task="image-classification")
    with pytest.raises(runtime.GoalShapeError):
        runtime.build_request_body(goal, route)


# ---------------------------------------------------------------------------
# Scorecard — deterministic technical checks.
# ---------------------------------------------------------------------------


def _executed(monkeypatch, dispatch):
    goal, route = _exec_route(monkeypatch)
    _patch_dispatch(monkeypatch, dispatch)
    artifacts, receipt = runtime.execute_route(goal, route)
    return goal, route, artifacts, receipt


def test_scorecard_happy_path(monkeypatch):
    goal, route, arts, receipt = _executed(
        monkeypatch, lambda k: _Result(ok=True, text="fine"))
    card = scorecard.build_technical_scorecard(goal, route, arts, receipt)
    assert card.hard_pass is True
    assert card.repair_code is None
    assert card.judge_results == ()          # k90c seam: present and empty
    assert "judge_results" in card.to_dict()
    assert {c.name for c in card.checks} == {"execution", "empty_output",
                                             "format", "decode"}


def test_scorecard_empty_artifact_fails_empty_output(monkeypatch):
    goal, route, arts, receipt = _executed(
        monkeypatch, lambda k: _Result(ok=True, text="   "))
    assert len(arts) == 1                     # the blank artifact IS produced
    card = scorecard.build_technical_scorecard(goal, route, arts, receipt)
    assert card.hard_pass is False
    assert card.repair_code is RepairCode.EMPTY_OUTPUT
    failing = {c.name for c in card.checks if not c.passed}
    assert failing == {"empty_output"}


def test_scorecard_worker_unavailable_from_receipt(monkeypatch):
    def dead(kwargs):
        raise ConnectionError("no route to host")
    goal, route, arts, receipt = _executed(monkeypatch, dead)
    card = scorecard.build_technical_scorecard(goal, route, arts, receipt)
    assert card.hard_pass is False
    assert card.repair_code is RepairCode.WORKER_UNAVAILABLE
    execution = [c for c in card.checks if c.name == "execution"][0]
    assert execution.value == "worker_unavailable"


def test_scorecard_format_mismatch():
    route = router.RouteDecision(
        capability="text.chat", execution="execute", task="text-generation",
        produces=(ArtifactKind.TEXT,))
    receipt = _receipt_for(route)
    arts = [{"kind": "json", "uri": "inline:json/x", "sha256": "0" * 64,
             "data": {"x": 1}}]
    card = scorecard.build_technical_scorecard(
        _goal("hi"), route, arts, receipt)
    assert card.hard_pass is False
    assert card.repair_code is RepairCode.FORMAT_MISMATCH


def test_scorecard_missing_file_is_decode_failed(tmp_path):
    route = router.RouteDecision(
        capability="image.generate", execution="execute", task="text-to-image",
        produces=(ArtifactKind.IMAGE,))
    receipt = _receipt_for(route)
    # one real (undecodable) file and the check must name it
    bad = tmp_path / "img.png"
    bad.write_bytes(b"not a png at all")
    arts = [{"kind": "image", "uri": str(bad), "sha256": "0" * 64}]
    card = scorecard.build_technical_scorecard(_goal("hi"), route, arts, receipt)
    try:
        import PIL  # noqa: F401
        assert card.repair_code is RepairCode.DECODE_FAILED
        assert card.hard_pass is False
    except ImportError:
        # PIL absent: degradation is named, size-only check passes
        decode = [c for c in card.checks if c.name == "decode"][0]
        assert "PIL unavailable" in decode.detail


def _receipt_for(route):
    from hugpy_oracle.contracts import ExecutionReceipt
    return ExecutionReceipt(
        request=ExecutionReceipt.normalize_request({"task": route.task}),
        capability=route.capability, model_id="m", worker=None,
        started_at="2026-08-05T00:00:00+00:00",
        ended_at="2026-08-05T00:00:01+00:00", duration_s=1.0)


def test_gap_and_deferred_scorecards_are_typed():
    gap = router.RouteDecision(capability="audio.transcribe", execution="gap",
                               reasons=("no model registered",))
    card = scorecard.build_gap_scorecard(gap)
    assert card.hard_pass is False
    assert card.repair_code is RepairCode.CAPABILITY_GAP
    assert "no model registered" in card.checks[0].detail

    deferred = router.RouteDecision(capability="video.generate.t2v",
                                    execution="deferred", model_id="wan-t2v")
    card = scorecard.build_deferred_scorecard(deferred)
    assert card.hard_pass is False
    assert card.repair_code is None
    assert card.judge_results == ()


# ---------------------------------------------------------------------------
# The route, over a bare Flask app.
# ---------------------------------------------------------------------------


def _client(monkeypatch, dispatch=None):
    from flask import Flask
    from hugpy_server.app.routes.oracle_routes import oracle_bp
    _stub_catalog(monkeypatch)
    if dispatch is not None:
        _patch_dispatch(monkeypatch, dispatch)
    app = Flask("oracle-route-test")
    app.register_blueprint(oracle_bp)
    return app.test_client()


def test_route_executes_and_carries_all_five_parts(monkeypatch):
    client = _client(monkeypatch,
                     dispatch=lambda k: _Result(ok=True, text="routed!"))
    resp = client.post("/oracle/route", json={"prompt": "hello oracle"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    for part in ("goal", "route", "artifacts", "receipt", "scorecard"):
        assert part in body, part
    assert body["route"]["capability"] == "text.chat"
    assert body["route"]["model_rationale"] == "only-eligible"
    assert body["artifacts"][0]["text"] == "routed!"
    assert body["scorecard"]["hard_pass"] is True
    assert body["scorecard"]["judge_results"] == []   # k90c seam on the wire
    assert body["receipt"]["capability"] == "text.chat"


def test_route_video_deferred_shape(monkeypatch):
    monkeypatch.setattr(router, "_studio_menu", lambda: "menu-here")
    client = _client(monkeypatch)
    resp = client.post("/oracle/route",
                       json={"prompt": "make a clip",
                             "capability": "video.generate.t2v"})
    assert resp.status_code == 202
    body = resp.get_json()
    assert body["execution"] == "deferred"
    assert body["routed"] == "video.generate.t2v"
    assert body["reason"]
    assert "binding" in body and "alternatives" in body
    assert body["scorecard"]["hard_pass"] is False


def test_route_capability_gap_shape(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post("/oracle/route",
                       json={"prompt": "x", "capability": "no.such.capability"})
    assert resp.status_code == 422
    body = resp.get_json()
    assert body["ok"] is False
    assert body["scorecard"]["repair_code"] == "capability_gap"
    assert body["route"]["execution"] == "gap"


def test_route_unmapped_task_string_is_gap_not_keyerror(monkeypatch):
    client = _client(monkeypatch)
    for task in ("text-to-video", "quantum-flux-sorting"):
        resp = client.post("/oracle/route",
                           json={"prompt": "x", "capability": task})
        assert resp.status_code == 422, task
        body = resp.get_json()
        assert body["scorecard"]["repair_code"] == "capability_gap"


def test_route_legacy_task_string_folds_to_capability(monkeypatch):
    client = _client(monkeypatch,
                     dispatch=lambda k: _Result(ok=True, text="hi"))
    resp = client.post("/oracle/route",
                       json={"prompt": "x", "capability": "text-generation"})
    assert resp.status_code == 200
    assert resp.get_json()["route"]["capability"] == "text.chat"


def test_route_requested_model_mismatch_is_400(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post("/oracle/route",
                       json={"prompt": "x", "model_id": "whisper-x"})
    assert resp.status_code == 400
    assert "does not serve" in resp.get_json()["error"]


def test_route_needs_prompt_or_capability(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post("/oracle/route", json={})
    assert resp.status_code == 400
    resp = client.post("/oracle/route", json={"inputs": []})
    assert resp.status_code == 400


def test_route_bad_input_kind_is_400(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post("/oracle/route",
                       json={"prompt": "x",
                             "inputs": [{"kind": "hologram", "uri": "y"}]})
    assert resp.status_code == 400
    assert "kind" in resp.get_json()["error"]


def test_route_missing_required_input_is_400(monkeypatch):
    client = _client(monkeypatch, dispatch=lambda k: _Result(ok=True, text="t"))
    resp = client.post("/oracle/route",
                       json={"prompt": "classify",
                             "capability": "image.classify"})
    assert resp.status_code == 400
    assert "image" in resp.get_json()["error"]


# ---------------------------------------------------------------------------
# k97 — the authority gate on the router and on the wire.
# ---------------------------------------------------------------------------

_MIRA = "identity_profile:mira"


def _rights(kind="likeness", subject=_MIRA):
    """A RightsManifest body, in the shape POST /oracle/route accepts."""
    return {"authorizations": [{"kind": kind, "subject": subject,
                                "evidence": "release-2026-08-14.pdf",
                                "granted_by": "operator"}]}


def test_resolve_route_refuses_an_unauthorized_identity_request(monkeypatch):
    from hugpy_oracle.contracts import AuthorityKind
    _stub_catalog(monkeypatch)
    route = router.resolve_route(_goal(f"restyle {_MIRA} as a knight"))
    assert route.execution == "refused"
    assert route.authority is not None
    assert route.authority.missing == ((AuthorityKind.LIKENESS, _MIRA),)
    # the refusal names the subject, so the operator knows which release to get
    assert any(_MIRA in r for r in route.reasons)
    # …and it never leaked a model or a dispatch task
    assert route.model_id is None and route.task is None


def test_resolve_route_gate_runs_before_the_catalog_is_even_read(monkeypatch):
    """The gate is Stage 1: an unauthorized request must not learn a route, so
    the capability lookup must not have run."""
    _stub_catalog(monkeypatch)
    seen = []
    monkeypatch.setattr(router, "_get_capability",
                        lambda name: seen.append(name))
    route = router.resolve_route(
        _goal("go", capability="video.generate.id_lock",
              inputs=[_ref("text", _MIRA)]))
    assert route.execution == "refused"
    assert seen == []


def test_resolve_route_proceeds_when_the_manifest_covers_it(monkeypatch):
    from hugpy_oracle.contracts import Authorization, AuthorityKind, GoalSpec, RightsManifest
    _stub_catalog(monkeypatch)
    prompt = f"what is {_MIRA} wearing in this shot?"
    goal = GoalSpec(
        objective=prompt, raw_prompt=prompt,
        inputs=(_ref("image", "/tmp/a.png"),),
        rights=RightsManifest(authorizations=(
            Authorization(kind=AuthorityKind.LIKENESS, subject=_MIRA,
                          evidence="release-2026-08-14.pdf"),)))
    route = router.resolve_route(goal)
    assert route.execution == "execute"
    assert route.capability == "image.understand"
    assert route.authority is not None and route.authority.ok is True
    assert route.to_dict()["authority"]["ok"] is True


def test_ordinary_routes_carry_an_empty_authority_record(monkeypatch):
    _stub_catalog(monkeypatch)
    route = router.resolve_route(_goal("hello there"))
    assert route.execution == "execute"
    assert route.authority.ok is True and route.authority.required == ()
    assert route.to_dict()["authority"]["missing"] == []


def test_route_authority_refusal_is_typed_403_and_never_dispatches(monkeypatch):
    calls = []
    client = _client(monkeypatch,
                     dispatch=lambda k: calls.append(k) or _Result(ok=True, text="x"))
    resp = client.post("/oracle/route",
                       json={"prompt": "make her say the line",
                             "identity_profile": "mira"})
    assert resp.status_code == 403
    body = resp.get_json()
    assert body["ok"] is False
    assert body["missing_authority"] == [{"kind": "likeness", "subject": _MIRA}]
    assert body["route"]["execution"] == "refused"
    assert body["receipt"]["failure"] == FailureClass.REFUSED.value
    assert body["scorecard"]["repair_code"] == RepairCode.SOURCE_AUTHORITY_MISSING.value
    assert body["scorecard"]["hard_pass"] is False
    assert body["planner_mode"] == "local_only"
    # the whole point: nothing executed
    assert calls == []


def test_route_with_authorization_proceeds_to_the_normal_path(monkeypatch):
    client = _client(monkeypatch,
                     dispatch=lambda k: _Result(ok=True, text="routed!"))
    resp = client.post("/oracle/route",
                       json={"prompt": "tell me about her",
                             "identity_profile": "mira",
                             "rights": _rights()})
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert body["ok"] is True
    assert body["artifacts"][0]["text"] == "routed!"
    assert body["goal"]["rights"]["authorizations"][0]["subject"] == _MIRA
    assert body["route"]["authority"]["ok"] is True


def test_route_echoes_planner_mode_on_every_response(monkeypatch):
    client = _client(monkeypatch,
                     dispatch=lambda k: _Result(ok=True, text="ok"))
    monkeypatch.setattr(router, "_studio_menu", lambda: "menu")
    cases = [
        ({"prompt": "hello"}, 200),
        ({"prompt": "clip", "capability": "video.generate.t2v"}, 202),
        ({"prompt": "x", "capability": "no.such.capability"}, 422),
        ({"prompt": "x", "capability": "quantum-flux-sorting"}, 422),
        ({"prompt": "x", "model_id": "whisper-x"}, 400),
        ({}, 400),
        ({"prompt": "go", "identity_profile": "mira"}, 403),
    ]
    for body, status in cases:
        resp = client.post("/oracle/route", json=body)
        assert resp.status_code == status, (body, resp.get_json())
        assert resp.get_json()["planner_mode"] == "local_only", body


def test_route_accepts_an_explicit_frontier_planner_mode(monkeypatch):
    client = _client(monkeypatch, dispatch=lambda k: _Result(ok=True, text="ok"))
    resp = client.post("/oracle/route",
                       json={"prompt": "hello", "planner_mode": "frontier"})
    assert resp.status_code == 200
    assert resp.get_json()["planner_mode"] == "frontier"
    resp = client.post("/oracle/route",
                       json={"prompt": "hello", "planner_mode": "telepathy"})
    assert resp.status_code == 400
    assert "telepathy" in resp.get_json()["error"]


def test_route_malformed_rights_block_is_a_clean_400(monkeypatch):
    client = _client(monkeypatch, dispatch=lambda k: _Result(ok=True, text="ok"))
    for rights in ({"authorizations": [{"kind": "likeness"}]},          # no subject
                   {"authorizations": [{"kind": "telepathy",
                                        "subject": "x", "evidence": "y"}]},
                   {"authorizations": [{"kind": "likeness", "subject": _MIRA}]}):
        resp = client.post("/oracle/route",
                           json={"prompt": "go", "rights": rights})
        assert resp.status_code == 400, rights
        assert resp.get_json()["ok"] is False


def test_route_uses_consent_recorded_on_the_identity_profile(monkeypatch, tmp_path):
    """The release the operator filed ONCE on the identity is an authorization —
    the caller does not re-state it on every request. Absent/unevidenced stays
    refused, so the store can only ever open the gate with evidence."""
    from hugpy_video.intel import identity_profiles

    monkeypatch.setattr(identity_profiles, "IDENTITIES_HOME",
                        str(tmp_path / "identities"))
    monkeypatch.setattr(identity_profiles, "PROJECTS_HOME", str(tmp_path / "projects"))
    src = tmp_path / "ref.png"
    src.write_bytes(b"reference-bytes")
    slug = identity_profiles.create_profile("Mira", [str(src)])["slug"]

    client = _client(monkeypatch, dispatch=lambda k: _Result(ok=True, text="ok"))
    body = {"prompt": "tell me about her", "identity_profile": slug}

    # no consent on file -> refused, by name
    resp = client.post("/oracle/route", json=body)
    assert resp.status_code == 403
    assert resp.get_json()["missing_authority"] == [
        {"kind": "likeness", "subject": f"identity_profile:{slug}"}]

    # consent filed with evidence -> the same request proceeds
    identity_profiles.set_profile_authorization(
        slug, "likeness", granted=True, evidence="release-2026-08-14.pdf")
    resp = client.post("/oracle/route", json=body)
    assert resp.status_code == 200, resp.get_json()
    granted = resp.get_json()["goal"]["rights"]["authorizations"][0]
    assert granted["granted_by"] == f"identity_profile:{slug}"
    assert granted["evidence"] == "release-2026-08-14.pdf"

    # revoked -> refused again
    identity_profiles.set_profile_authorization(slug, "likeness", granted=False)
    assert client.post("/oracle/route", json=body).status_code == 403


# ---------------------------------------------------------------------------
# k101b — the route ENDS, honestly: a bounded wait, one canonical subject ref,
# and a refusal receipt that carries its own classification.
# ---------------------------------------------------------------------------


def test_route_dispatch_timeout_is_a_typed_504(monkeypatch):
    """The fleet stalls (a busy worker, a cold load holding for up to 25
    minutes) — the oracle stops waiting at ITS deadline and answers with the
    same typed shape as every other failure, instead of riding the hold until
    gunicorn drops the connection and the caller gets nothing at all."""
    import threading
    import time as _time

    release = threading.Event()
    client = _client(monkeypatch, dispatch=lambda k: release.wait(30))
    monkeypatch.setattr(runtime, "sync_deadline_s", lambda goal=None: 0.4)
    monkeypatch.setattr(runtime, "_selected_worker",
                        lambda model, task, pool: "a-brain")
    try:
        t0 = _time.monotonic()
        resp = client.post("/oracle/route", json={"prompt": "hello oracle"})
        elapsed = _time.monotonic() - t0
    finally:
        release.set()

    assert resp.status_code == 504
    assert elapsed < 3.0, elapsed          # ~the deadline, not ~the cold hold
    body = resp.get_json()
    assert body["ok"] is False
    assert body["execution"] == "timeout"
    assert body["planner_mode"] == "local_only"
    # the same five parts every other answer carries
    for part in ("goal", "route", "receipt", "scorecard", "reason"):
        assert part in body, part
    assert body["receipt"]["failure"] == FailureClass.TIMEOUT.value
    assert body["scorecard"]["repair_code"] == RepairCode.TIMEOUT.value
    assert body["scorecard"]["hard_pass"] is False
    # the reason names what the wait was holding on
    assert "qwen-chat" in body["reason"]
    assert "a-brain" in body["reason"]


def test_route_budget_hint_rides_on_the_goal(monkeypatch):
    """``budget.max_seconds`` is the first source of the synchronous deadline
    (runtime.sync_deadline_s), so the route must actually parse it."""
    client = _client(monkeypatch, dispatch=lambda k: _Result(ok=True, text="ok"))
    resp = client.post("/oracle/route",
                       json={"prompt": "hello", "budget": {"max_seconds": 20}})
    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json()["goal"]["budget"]["max_seconds"] == 20
    # a malformed budget is a typed 400, never a traceback
    assert client.post("/oracle/route",
                       json={"prompt": "hello", "budget": {"max_seconds": -1}}
                       ).status_code == 400
    # …and a non-object budget is simply not a budget
    assert client.post("/oracle/route",
                       json={"prompt": "hello", "budget": [20]}
                       ).status_code == 200


@pytest.mark.parametrize("spelling", ["mira", "identity_profile:mira",
                                      " identity_profile:mira "])
def test_route_identity_ref_is_prefixed_exactly_once(monkeypatch, spelling):
    """Both spellings the callers actually send land on ONE canonical subject.
    Before k101b an already-prefixed value was prefixed AGAIN, and the refusal
    named ``identity_profile:identity_profile`` — a person who does not exist,
    with the real slug lost."""
    client = _client(monkeypatch,
                     dispatch=lambda k: _Result(ok=True, text="x"))
    resp = client.post("/oracle/route",
                       json={"prompt": "make her say the line",
                             "identity_profile": spelling})
    assert resp.status_code == 403
    body = resp.get_json()
    assert body["missing_authority"] == [{"kind": "likeness", "subject": _MIRA}]
    refs = [i["ref"] for i in body["goal"]["inputs"]]
    assert refs == [_MIRA]


def test_route_voice_profile_ref_is_prefixed_exactly_once(monkeypatch):
    client = _client(monkeypatch,
                     dispatch=lambda k: _Result(ok=True, text="x"))
    for spelling in ("mira", "voice_profile:mira"):
        resp = client.post("/oracle/route",
                           json={"prompt": "say the line",
                                 "voice_profile": spelling})
        assert resp.status_code == 403, spelling
        body = resp.get_json()
        assert body["missing_authority"] == [
            {"kind": "voice", "subject": "voice_profile:mira"}], spelling


def test_route_identity_field_keeps_a_voice_prefix_it_was_given(monkeypatch):
    """A ``voice_profile:`` ref passed in the identity field is a VOICE
    subject; silently relabelling it would be the same class of lie as the
    double prefix."""
    client = _client(monkeypatch,
                     dispatch=lambda k: _Result(ok=True, text="x"))
    resp = client.post("/oracle/route",
                       json={"prompt": "say the line",
                             "identity_profile": "voice_profile:mira"})
    assert resp.status_code == 403
    assert resp.get_json()["missing_authority"] == [
        {"kind": "voice", "subject": "voice_profile:mira"}]


def test_route_refusal_receipt_carries_its_own_failure_class(monkeypatch):
    """The 403 body's receipt IS ``authority.refusal_receipt`` — classified
    REFUSED, not a null failure next to a scorecard that says otherwise. (The
    serialized field is ``receipt.failure``; ``ExecutionReceipt`` has no
    ``failure_class`` key, so a reader looking for that name reads None off a
    receipt that is in fact classified.)"""
    client = _client(monkeypatch, dispatch=lambda k: _Result(ok=True, text="x"))
    resp = client.post("/oracle/route",
                       json={"prompt": "make her say the line",
                             "identity_profile": "mira"})
    assert resp.status_code == 403
    receipt = resp.get_json()["receipt"]
    assert receipt["failure"] == FailureClass.REFUSED.value == "refused"
    assert "failure_class" not in receipt
    assert receipt["capability"] == "text.chat"
    assert receipt["model_id"] == ""        # the gate came before a model pick
    assert receipt["duration_s"] == 0.0
    assert receipt["log_excerpt"], "a refusal receipt states its reason"
    assert resp.get_json()["scorecard"]["repair_code"] == (
        RepairCode.SOURCE_AUTHORITY_MISSING.value)


# ---------------------------------------------------------------------------
# k105 — registry_version: top-level on every response, stamped on every
# receipt this route builds.
# ---------------------------------------------------------------------------


def test_route_success_response_carries_top_level_registry_version(monkeypatch):
    client = _client(monkeypatch, dispatch=lambda k: _Result(ok=True, text="hi"))
    expected = catalog.registry_version()
    assert expected is not None
    resp = client.post("/oracle/route", json={"prompt": "hello oracle"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["registry_version"] == expected
    assert body["receipt"]["registry_version"] == expected


def test_route_refusal_response_and_receipt_carry_registry_version(monkeypatch):
    client = _client(monkeypatch, dispatch=lambda k: _Result(ok=True, text="x"))
    expected = catalog.registry_version()
    resp = client.post("/oracle/route",
                       json={"prompt": "make her say the line",
                             "identity_profile": "mira"})
    assert resp.status_code == 403
    body = resp.get_json()
    assert body["registry_version"] == expected
    assert body["receipt"]["registry_version"] == expected


def test_route_capability_gap_response_carries_registry_version(monkeypatch):
    client = _client(monkeypatch)
    expected = catalog.registry_version()
    resp = client.post("/oracle/route",
                       json={"prompt": "x", "capability": "no.such.capability"})
    assert resp.status_code == 422
    assert resp.get_json()["registry_version"] == expected


def test_route_unmapped_task_gap_response_carries_registry_version(monkeypatch):
    client = _client(monkeypatch)
    expected = catalog.registry_version()
    resp = client.post("/oracle/route",
                       json={"prompt": "x", "capability": "quantum-flux-sorting"})
    assert resp.status_code == 422
    assert resp.get_json()["registry_version"] == expected


def test_route_deferred_response_carries_registry_version(monkeypatch):
    monkeypatch.setattr(router, "_studio_menu", lambda: "menu-here")
    client = _client(monkeypatch)
    expected = catalog.registry_version()
    resp = client.post("/oracle/route",
                       json={"prompt": "make a clip",
                             "capability": "video.generate.t2v"})
    assert resp.status_code == 202
    assert resp.get_json()["registry_version"] == expected


def test_route_timeout_response_and_receipt_carry_registry_version(monkeypatch):
    import threading

    release = threading.Event()
    client = _client(monkeypatch, dispatch=lambda k: release.wait(30))
    monkeypatch.setattr(runtime, "sync_deadline_s", lambda goal=None: 0.4)
    monkeypatch.setattr(runtime, "_selected_worker",
                        lambda model, task, pool: "a-brain")
    expected = catalog.registry_version()
    try:
        resp = client.post("/oracle/route", json={"prompt": "hello oracle"})
    finally:
        release.set()
    assert resp.status_code == 504
    body = resp.get_json()
    assert body["registry_version"] == expected
    assert body["receipt"]["registry_version"] == expected


def test_route_answers_with_null_registry_version_when_the_catalog_faults(monkeypatch):
    """A catalog fault must not crash the route — the response still goes
    out, top-level AND on the receipt, both honestly null (never a guess)."""
    from hugpy_server.app.routes import oracle_routes

    client = _client(monkeypatch, dispatch=lambda k: _Result(ok=True, text="hi"))
    monkeypatch.setattr(oracle_routes, "_safe_registry_version", lambda: None)
    resp = client.post("/oracle/route", json={"prompt": "hello oracle"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["registry_version"] is None
    assert body["receipt"]["registry_version"] is None


def test_safe_registry_version_returns_none_on_a_catalog_fault(monkeypatch):
    """The helper itself never propagates a catalog exception."""
    from hugpy_server.app.routes import oracle_routes

    def boom():
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(catalog, "registry_version", boom)
    assert oracle_routes._safe_registry_version() is None


def test_capabilities_route_carries_top_level_registry_version(monkeypatch):
    client = _client(monkeypatch)
    expected = catalog.registry_version()
    resp = client.get("/oracle/capabilities")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["registry_version"] == expected
    # every entry already carried it (k101); the top-level key is the SAME
    # snapshot, not a second, possibly-different read.
    assert {c["registry_version"] for c in body["capabilities"]} == {expected}


def test_capabilities_route_filter_carries_registry_version(monkeypatch):
    client = _client(monkeypatch)
    expected = catalog.registry_version()
    resp = client.get("/oracle/capabilities?capability=audio.transcribe")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["registry_version"] == expected
    assert body["capabilities"][0]["registry_version"] == expected


def test_capabilities_route_unknown_name_still_carries_registry_version(monkeypatch):
    client = _client(monkeypatch)
    expected = catalog.registry_version()
    resp = client.get("/oracle/capabilities?capability=no.such.capability")
    assert resp.status_code == 404
    assert resp.get_json()["registry_version"] == expected


def test_capabilities_route_answers_with_null_registry_version_on_a_catalog_fault(
        monkeypatch):
    """A ``list_capabilities()`` fault (registry unreadable) is a typed 500,
    not a crash — and it carries an honest ``registry_version: null`` rather
    than a guess."""
    _stub_catalog(monkeypatch)

    def boom():
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(catalog, "registry_version", boom)
    from flask import Flask
    from hugpy_server.app.routes.oracle_routes import oracle_bp
    app = Flask("oracle-route-test-fault")
    app.register_blueprint(oracle_bp)
    client = app.test_client()

    resp = client.get("/oracle/capabilities")
    assert resp.status_code == 500
    body = resp.get_json()
    assert body["ok"] is False
    assert body["registry_version"] is None


# ---------------------------------------------------------------------------
# k113 — planner mode GATES (POLICY-rights-consent-disclosure §3.1-3.2)
# ---------------------------------------------------------------------------


def test_route_helper_reports_the_effective_planner_mode(monkeypatch):
    from hugpy_server.app.routes import oracle_routes
    from hugpy_oracle.plan import FRONTIER_ENABLED_ENV
    monkeypatch.delenv(FRONTIER_ENABLED_ENV, raising=False)
    assert oracle_routes._planner_mode({"planner_mode": "frontier"}) == "local_only"
    assert oracle_routes._planner_mode({"planner_mode": "telepathy"}) == "local_only"
    assert oracle_routes._planner_mode({}) == "local_only"
    monkeypatch.setenv(FRONTIER_ENABLED_ENV, "1")
    assert oracle_routes._planner_mode({"planner_mode": "frontier"}) == "frontier"


def test_route_refuses_a_frontier_capability_under_local_only(monkeypatch):
    from hugpy_oracle.plan import FRONTIER_ENABLED_ENV
    monkeypatch.setenv(FRONTIER_ENABLED_ENV, "1")
    client = _client(monkeypatch, dispatch=lambda k: _Result(ok=True, text="ok"))
    resp = client.post("/oracle/route",
                       json={"prompt": "plan", "capability": "frontier.plan"})
    assert resp.status_code == 403, resp.get_json()
    body = resp.get_json()
    assert body["planner_mode"] == "local_only"
    assert body["missing_authority"] == [{"kind": "network", "subject": "frontier.plan"}]
    assert body["receipt"]["failure"] == "refused"
    assert body["route"]["authority"]["fallback"] is None


def test_route_fallback_offer_rides_on_the_refusal(monkeypatch):
    client = _client(monkeypatch, dispatch=lambda k: _Result(ok=True, text="ok"))
    resp = client.post("/oracle/route", json={"prompt": "go", "identity_profile": "mira"})
    assert resp.status_code == 403
    body = resp.get_json()
    auth = body["route"]["authority"]
    assert auth["outcome"] == "fallback_offered"
    assert auth["fallback"]["capability"] == "text.chat"
    assert "mira" not in auth["fallback"]["disclosure"].lower()
    assert "apply_fallback" in body["scorecard"]["recommended_repair"]
