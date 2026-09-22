"""k90c — evaluator kernel + bounded repair: rubric selection, verdict parsing,
per-quality thresholds, judge-unavailable degradation (the fleet's vision plane
is DOWN today — that path is load-bearing), the repair policy table, and the
route-level integration with dispatch + judge monkeypatched.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_evaluation.py -q
"""
from __future__ import annotations

import base64
import logging
import os
import sys

logging.disable(logging.INFO)  # silence the models_config registry chatter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from hugpy_oracle import catalog, evaluation, repair, router, runtime, scorecard
from hugpy_oracle.contracts import (
    CheckKind,
    ExecutionReceipt,
    GoalSpec,
    InputKind,
    InputRef,
    QualityProfile,
    RepairCode,
    Scorecard,
)

# 1x1 red PNG — a REAL decodable image so the technical decode check passes.
_PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8Dw"
    "HwAFAAH/q842iQAAAABJRU5ErkJggg==")


def _goal(prompt="hello", inputs=(), capability=None,
          quality=QualityProfile.BALANCED):
    return GoalSpec(objective=prompt, raw_prompt=prompt, inputs=tuple(inputs),
                    capability=capability, quality=quality)


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


def _stub_catalog(monkeypatch, rows=_ROWS):
    monkeypatch.setattr(catalog, "_legacy_registry_rows", lambda: dict(rows))
    monkeypatch.setattr(catalog, "_online_workers", lambda: [{"id": "w1"}])
    monkeypatch.setattr(catalog, "_worker_task_capable", lambda w, t: True)
    monkeypatch.setattr(catalog, "_central_task_available", lambda t: True)
    monkeypatch.setattr(catalog, "_blocked_model_keys", lambda: set())


def _receipt(capability="image.generate", failure=None):
    return ExecutionReceipt(
        request=ExecutionReceipt.normalize_request({"task": "t"}),
        capability=capability, model_id="m", worker=None,
        started_at="2026-08-05T00:00:00+00:00",
        ended_at="2026-08-05T00:00:01+00:00", duration_s=1.0, failure=failure)


def _passing_card():
    return Scorecard(hard_pass=True)


def _image_artifact(tmp_path):
    p = os.path.join(str(tmp_path), "out.png")
    with open(p, "wb") as fh:
        fh.write(_PNG_1PX)
    return {"kind": "image", "uri": p, "sha256": "0" * 64}


def _image_route(model_id="sdxl", model_ids=("sdxl",)):
    return router.RouteDecision(
        capability="image.generate", execution="execute", task="text-to-image",
        model_id=model_id, model_ids=tuple(model_ids))


class _Result:
    def __init__(self, **payload):
        self._payload = payload

    def model_dump(self):
        return dict(self._payload)

    def __getattr__(self, name):
        try:
            return self._payload[name]
        except KeyError:
            raise AttributeError(name)


def _judge_reply(monkeypatch, text):
    """Judge answers with ``text`` (or raises, when text is an exception)."""
    calls = []

    def fake(task, body):
        calls.append((task, dict(body)))
        if isinstance(text, BaseException):
            raise text
        return _Result(ok=True, text=text)

    monkeypatch.setattr(evaluation, "_judge_dispatch", fake)
    return calls


# ---------------------------------------------------------------------------
# Rubric selection per capability.
# ---------------------------------------------------------------------------


def test_rubric_table_image_and_summarize():
    for cap in ("image.generate", "image.transform"):
        rubric = evaluation.RUBRICS[cap]
        assert rubric.kind is CheckKind.INTENT
        assert rubric.judge_capability == "image.understand"
        assert rubric.judged_artifact == "image"
    rubric = evaluation.RUBRICS["text.summarize"]
    assert rubric.kind is CheckKind.SEMANTIC
    assert rubric.judge_capability == "text.chat"
    assert rubric.judged_artifact == "text"
    assert evaluation.DEFAULT_EVALUATED == frozenset(
        {"image.generate", "image.transform", "text.summarize"})


@pytest.mark.parametrize("cap", [
    "text.embed", "text.similarity", "image.detect", "image.classify",
    "image.depth", "image.segment", "audio.transcribe", "image.understand",
    "text.chat", "text.keywords", "doc.extract", "web.fetch",
])
def test_analytic_capabilities_have_no_rubric(cap):
    assert cap not in evaluation.RUBRICS
    # and evaluate() is the identity for them — no judge call is even attempted
    route = router.RouteDecision(capability=cap, execution="execute", task="t")
    card = _passing_card()
    out = evaluation.evaluate(_goal(), route, [{"kind": "text", "uri": "u",
                                                "text": "x"}], _receipt(cap), card)
    assert out is card


# ---------------------------------------------------------------------------
# Verdict parsing — lifted movie discipline.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,verdict,score,why", [
    ("VERDICT=YES; SCORE=88; WHY=matches the goal.", "YES", 88, "matches the goal"),
    ("verdict: no; score: 12; why: wrong subject", "NO", 12, "wrong subject"),
    ("VERDICT=YES; SCORE=250; WHY=x", "YES", 100, "x"),      # clamped
    ("Sure! YES it does.", "YES", None, ""),                  # bare-word fallback
    ("complete garbage lacking fields", None, None, ""),      # malformed -> unscored
    ("", None, None, ""),
])
def test_parse_judge_verdict(text, verdict, score, why):
    out = evaluation.parse_judge_verdict(text)
    assert out["verdict"] == verdict
    assert out["score"] == score
    assert out["why"] == why


# ---------------------------------------------------------------------------
# evaluate() — thresholds per quality, degradation, disagreement.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quality,threshold,passes", [
    (QualityProfile.PREVIEW, 40, True),    # score 50 clears preview's 40
    (QualityProfile.BALANCED, 60, False),  # ...but not balanced's 60
    (QualityProfile.BEST, 75, False),      # ...nor best's 75
])
def test_threshold_per_quality(monkeypatch, tmp_path, quality, threshold, passes):
    _stub_catalog(monkeypatch)
    assert evaluation.THRESHOLDS[quality] == threshold
    _judge_reply(monkeypatch, "VERDICT=YES; SCORE=50; WHY=middling")
    goal = _goal("a red square", quality=quality)
    card = evaluation.evaluate(goal, _image_route(), [_image_artifact(tmp_path)],
                               _receipt(), _passing_card())
    assert card.hard_pass is passes
    assert card.judge_results[0].score == 50.0
    if passes:
        assert card.repair_code is None
    else:
        assert card.repair_code is RepairCode.INTENT_MISMATCH
        assert str(threshold) in card.diagnosis
        assert card.recommended_repair


def test_judge_model_comes_from_catalog_not_hardcoded(monkeypatch, tmp_path):
    _stub_catalog(monkeypatch)
    calls = _judge_reply(monkeypatch, "VERDICT=YES; SCORE=90; WHY=good")
    card = evaluation.evaluate(_goal("a red square"), _image_route(),
                               [_image_artifact(tmp_path)], _receipt(),
                               _passing_card())
    assert card.hard_pass is True
    task, body = calls[0]
    assert task == "image-text-to-text"          # image.understand's task
    assert body["model_key"] == "qwen-vl"        # the catalog's eligible VLM
    assert body["file"].endswith("out.png")
    assert card.judge_results[0].judge == "image_intent:qwen-vl"


def test_judge_error_degrades_hard_pass_unaffected(monkeypatch, tmp_path):
    """The live-fleet state: both vision backends broken. A judge raise must
    record {judge, verdict: "unavailable", rationale: <error tail>} and leave
    hard_pass alone (movie: 'unscored, keep')."""
    _stub_catalog(monkeypatch)
    _judge_reply(monkeypatch, ConnectionError("computron GGUF VL 500s with images"))
    card = evaluation.evaluate(_goal("a red square"), _image_route(),
                               [_image_artifact(tmp_path)], _receipt(),
                               _passing_card())
    assert card.hard_pass is True
    assert card.repair_code is None
    (jr,) = card.judge_results
    assert jr.verdict == "unavailable"
    assert "computron GGUF VL 500s" in jr.rationale


def test_judge_not_ok_and_garbage_degrade(monkeypatch, tmp_path):
    _stub_catalog(monkeypatch)
    # not-ok result
    _judge_reply(monkeypatch, None)
    monkeypatch.setattr(evaluation, "_judge_dispatch",
                        lambda task, body: _Result(ok=False, error="load error"))
    card = evaluation.evaluate(_goal("x"), _image_route(),
                               [_image_artifact(tmp_path)], _receipt(),
                               _passing_card())
    assert card.hard_pass is True
    assert card.judge_results[0].verdict == "unavailable"
    assert "load error" in card.judge_results[0].rationale
    # garbage reply -> unscored, keep
    _judge_reply(monkeypatch, "mumble mumble maybe perhaps")
    card = evaluation.evaluate(_goal("x"), _image_route(),
                               [_image_artifact(tmp_path)], _receipt(),
                               _passing_card())
    assert card.hard_pass is True
    assert card.judge_results[0].verdict == "unscored"


def test_no_eligible_judge_capability_degrades(monkeypatch, tmp_path):
    # a fleet with no VLM at all: judge route resolves to a gap -> unavailable
    rows = {k: v for k, v in _ROWS.items() if k != "qwen-vl"}
    _stub_catalog(monkeypatch, rows=rows)
    card = evaluation.evaluate(_goal("x"), _image_route(),
                               [_image_artifact(tmp_path)], _receipt(),
                               _passing_card())
    assert card.hard_pass is True
    (jr,) = card.judge_results
    assert jr.verdict == "unavailable"
    assert "image.understand" in jr.rationale


def test_failed_execution_and_missing_artifact_skip_the_judge(monkeypatch, tmp_path):
    _stub_catalog(monkeypatch)
    calls = _judge_reply(monkeypatch, "VERDICT=YES; SCORE=99; WHY=x")
    from hugpy_oracle.contracts import FailureClass
    # failed execution: nothing to judge
    card = evaluation.evaluate(
        _goal("x"), _image_route(), [],
        _receipt(failure=FailureClass.WORKER_UNAVAILABLE),
        Scorecard(hard_pass=False, repair_code=RepairCode.WORKER_UNAVAILABLE))
    assert card.judge_results == ()
    # no image artifact on disk: rubric has no evidence
    card = evaluation.evaluate(
        _goal("x"), _image_route(),
        [{"kind": "image", "uri": os.path.join(str(tmp_path), "gone.png")}],
        _receipt(), _passing_card())
    assert card.judge_results == ()
    assert calls == []


def test_verdict_no_without_score_fails_and_disagreement_recorded(monkeypatch, tmp_path):
    _stub_catalog(monkeypatch)
    # NO with no score -> explicit judge fail
    _judge_reply(monkeypatch, "VERDICT=NO; WHY=wrong subject entirely")
    card = evaluation.evaluate(_goal("x"), _image_route(),
                               [_image_artifact(tmp_path)], _receipt(),
                               _passing_card())
    assert card.hard_pass is False
    assert card.repair_code is RepairCode.INTENT_MISMATCH
    # NO but score above bar -> score wins, disagreement recorded
    _judge_reply(monkeypatch, "VERDICT=NO; SCORE=80; WHY=odd")
    card = evaluation.evaluate(_goal("x"), _image_route(),
                               [_image_artifact(tmp_path)], _receipt(),
                               _passing_card())
    assert card.hard_pass is True
    assert card.disagreements and "score wins" in card.disagreements[0]


def test_summarize_rubric_judges_via_text_chat(monkeypatch):
    _stub_catalog(monkeypatch)
    calls = _judge_reply(monkeypatch, "VERDICT=YES; SCORE=85; WHY=faithful")
    goal = _goal("summarize this", inputs=[_ref("text", "a long source text")],
                 capability="text.summarize")
    route = router.RouteDecision(capability="text.summarize",
                                 execution="execute", task="text-summarization")
    arts = [{"kind": "text", "uri": "inline:text/x", "text": "a short summary"}]
    card = evaluation.evaluate(goal, route, arts,
                               _receipt("text.summarize"), _passing_card())
    assert card.hard_pass is True
    task, body = calls[0]
    assert task == "text-generation"             # text.chat's task
    assert body["model_key"] == "qwen-chat"
    assert "a long source text" in body["prompt"]
    assert "a short summary" in body["prompt"]
    assert card.judge_results[0].judge == "summary_faithfulness:qwen-chat"


# ---------------------------------------------------------------------------
# Repair policy — attempt_repair mapping.
# ---------------------------------------------------------------------------


def _failing(code):
    return Scorecard(hard_pass=False, repair_code=code)


@pytest.mark.parametrize("code", [RepairCode.WORKER_UNAVAILABLE, RepairCode.TIMEOUT])
def test_repair_infra_failure_retries_next_model(code):
    route = _image_route(model_id="sdxl", model_ids=("sdxl", "flux"))
    d = repair.attempt_repair(_goal(), route, _failing(code))
    assert d.action == "retry_next_model"
    assert d.model_id == "flux"                 # the next ELIGIBLE model
    assert code.value in d.rationale


@pytest.mark.parametrize("code", [RepairCode.WORKER_UNAVAILABLE, RepairCode.TIMEOUT])
def test_repair_no_eligible_alternative_is_honest_none(code):
    route = _image_route(model_id="sdxl", model_ids=("sdxl",))
    d = repair.attempt_repair(_goal(), route, _failing(code))
    assert d.action == "none"
    assert "no eligible alternative" in d.rationale


@pytest.mark.parametrize("code", [RepairCode.EMPTY_OUTPUT, RepairCode.DECODE_FAILED])
def test_repair_empty_or_undecodable_retries_same(code):
    d = repair.attempt_repair(_goal(), _image_route(), _failing(code))
    assert d.action == "retry_same"


def test_repair_intent_mismatch_reseeds_image_generate_only():
    goal = _goal("a red square")
    d = repair.attempt_repair(goal, _image_route(),
                              _failing(RepairCode.INTENT_MISMATCH))
    assert d.action == "reseed"
    assert d.seed == repair.bumped_seed(goal) and isinstance(d.seed, int)
    # ...but not other judged capabilities
    route = router.RouteDecision(capability="text.summarize",
                                 execution="execute", task="text-summarization")
    d = repair.attempt_repair(goal, route, _failing(RepairCode.INTENT_MISMATCH))
    assert d.action == "none"


def test_repair_none_cases():
    d = repair.attempt_repair(_goal(), _image_route(), Scorecard(hard_pass=True))
    assert d.action == "none"
    d = repair.attempt_repair(_goal(), _image_route(),
                              Scorecard(hard_pass=False))   # no code
    assert d.action == "none"
    d = repair.attempt_repair(_goal(), _image_route(),
                              _failing(RepairCode.CAPABILITY_GAP))
    assert d.action == "none"


def test_execute_repair_reseed_passes_seed_and_annotates(monkeypatch, tmp_path):
    _stub_catalog(monkeypatch)
    seen = {}

    def fake_dispatch(kwargs):
        seen.update(kwargs)
        return _Result(ok=True, images=[{"path": _image_artifact(tmp_path)["uri"]}])

    monkeypatch.setattr(runtime, "_dispatch", fake_dispatch)
    monkeypatch.setattr(runtime, "_normalized_kwargs",
                        lambda task, body: dict(body, task=task))
    goal = _goal("a red square", capability="image.generate")
    route = router.resolve_route(goal)
    d = repair.attempt_repair(goal, route, _failing(RepairCode.INTENT_MISMATCH))
    arts, receipt, route2 = repair.execute_repair(goal, route, d)
    assert seen["seed"] == d.seed
    assert arts and arts[0]["kind"] == "image"
    assert any("repair attempt (reseed)" in w for w in receipt.warnings)
    assert route2 is route


def test_execute_repair_next_model_swaps_route(monkeypatch):
    _stub_catalog(monkeypatch)
    seen = {}
    monkeypatch.setattr(runtime, "_dispatch",
                        lambda kwargs: (seen.update(kwargs),
                                        _Result(ok=True, text="ok"))[1])
    monkeypatch.setattr(runtime, "_normalized_kwargs",
                        lambda task, body: dict(body, task=task))
    goal = _goal("hi", capability="text.chat")
    route = router.RouteDecision(
        capability="text.chat", execution="execute", task="text-generation",
        model_id="qwen-chat", model_ids=("qwen-chat", "qwen-chat-2"))
    d = repair.attempt_repair(goal, route, _failing(RepairCode.TIMEOUT))
    _, receipt, route2 = repair.execute_repair(goal, route, d)
    assert seen["model_key"] == "qwen-chat-2"
    assert route2.model_id == "qwen-chat-2"
    assert route2.model_rationale == "repair:next-eligible"
    assert receipt.model_id == "qwen-chat-2"


# ---------------------------------------------------------------------------
# Route-level integration — dispatch + judge monkeypatched.
# ---------------------------------------------------------------------------


def _client(monkeypatch, dispatch, judge_replies):
    """Flask test client with generation dispatch and judge scripted. The
    judge pops replies off ``judge_replies`` in order."""
    from flask import Flask
    from hugpy_server.app.routes.oracle_routes import oracle_bp
    _stub_catalog(monkeypatch)
    monkeypatch.setattr(runtime, "_dispatch", dispatch)
    monkeypatch.setattr(runtime, "_normalized_kwargs",
                        lambda task, body: dict(body, task=task))
    judge_calls = []

    def fake_judge(task, body):
        judge_calls.append((task, dict(body)))
        reply = judge_replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return _Result(ok=True, text=reply)

    monkeypatch.setattr(evaluation, "_judge_dispatch", fake_judge)
    app = Flask("oracle-eval-test")
    app.register_blueprint(oracle_bp)
    return app.test_client(), judge_calls


def _gen_dispatch(tmp_path, log):
    """image.generate fake: every call renders a fresh real 1x1 png."""
    def fake(kwargs):
        log.append(dict(kwargs))
        p = os.path.join(str(tmp_path), f"gen_{len(log)}.png")
        with open(p, "wb") as fh:
            fh.write(_PNG_1PX)
        return _Result(ok=True, images=[{"path": p}])
    return fake


def test_route_healthy_pass_carries_judge_results(monkeypatch, tmp_path):
    gen_log = []
    client, judge_calls = _client(
        monkeypatch, _gen_dispatch(tmp_path, gen_log),
        ["VERDICT=YES; SCORE=92; WHY=exactly the goal"])
    resp = client.post("/oracle/route",
                       json={"prompt": "a red square",
                             "capability": "image.generate"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["scorecard"]["hard_pass"] is True
    (jr,) = body["scorecard"]["judge_results"]
    assert jr["verdict"] == "YES" and jr["score"] == 92.0
    assert len(gen_log) == 1 and len(judge_calls) == 1
    assert "receipts" not in body and "repair" not in body   # no repair ran


def test_route_judged_fail_reseed_then_pass(monkeypatch, tmp_path):
    gen_log = []
    client, judge_calls = _client(
        monkeypatch, _gen_dispatch(tmp_path, gen_log),
        ["VERDICT=NO; SCORE=20; WHY=wrong subject",
         "VERDICT=YES; SCORE=90; WHY=fixed"])
    resp = client.post("/oracle/route",
                       json={"prompt": "a red square",
                             "capability": "image.generate"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert len(gen_log) == 2                         # exactly ONE repair
    assert "seed" not in gen_log[0]
    assert isinstance(gen_log[1]["seed"], int)       # the bumped seed
    assert body["repair"]["action"] == "reseed"
    assert len(body["receipts"]) == 2                # both attempts kept
    assert body["scorecard"]["hard_pass"] is True
    assert "bounded repair" in body["scorecard"]["diagnosis"]
    assert any("repair attempt" in w for w in body["receipt"]["warnings"])
    assert len(judge_calls) == 2                     # re-scored after repair


def test_route_judged_fail_reseed_still_fail_is_honest(monkeypatch, tmp_path):
    gen_log = []
    client, _ = _client(
        monkeypatch, _gen_dispatch(tmp_path, gen_log),
        ["VERDICT=NO; SCORE=15; WHY=wrong",
         "VERDICT=NO; SCORE=25; WHY=still wrong"])
    resp = client.post("/oracle/route",
                       json={"prompt": "a red square",
                             "capability": "image.generate"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(gen_log) == 2                         # ONE repair, no loop
    assert body["scorecard"]["hard_pass"] is False
    assert body["scorecard"]["repair_code"] == "intent_mismatch"
    assert "bounded repair" in body["scorecard"]["diagnosis"]
    assert len(body["receipts"]) == 2


def test_route_judge_unavailable_still_answers_200_pass(monkeypatch, tmp_path):
    """The degraded live path: generation works, both vision judges are down."""
    gen_log = []
    client, _ = _client(monkeypatch, _gen_dispatch(tmp_path, gen_log),
                        [ConnectionError("MiniCPM load error")])
    resp = client.post("/oracle/route",
                       json={"prompt": "a red square",
                             "capability": "image.generate"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["scorecard"]["hard_pass"] is True
    (jr,) = body["scorecard"]["judge_results"]
    assert jr["verdict"] == "unavailable"
    assert "MiniCPM" in jr["rationale"]
    assert len(gen_log) == 1                         # no repair on unavailable


def test_route_evaluate_defaults_and_opt_out(monkeypatch, tmp_path):
    # text.chat is NOT evaluated by default
    client, judge_calls = _client(monkeypatch,
                                  lambda k: _Result(ok=True, text="hi"), [])
    resp = client.post("/oracle/route", json={"prompt": "hello",
                                              "capability": "text.chat"})
    assert resp.status_code == 200
    assert resp.get_json()["scorecard"]["judge_results"] == []
    assert judge_calls == []
    # image.generate IS by default — and evaluate:false turns it off
    gen_log = []
    client, judge_calls = _client(monkeypatch, _gen_dispatch(tmp_path, gen_log),
                                  ["VERDICT=NO; SCORE=1; WHY=x"])
    resp = client.post("/oracle/route",
                       json={"prompt": "a red square",
                             "capability": "image.generate",
                             "evaluate": False})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["scorecard"]["judge_results"] == []
    assert judge_calls == [] and len(gen_log) == 1


def test_route_repair_false_reports_first_attempt(monkeypatch, tmp_path):
    gen_log = []
    client, _ = _client(monkeypatch, _gen_dispatch(tmp_path, gen_log),
                        ["VERDICT=NO; SCORE=10; WHY=wrong"])
    resp = client.post("/oracle/route",
                       json={"prompt": "a red square",
                             "capability": "image.generate",
                             "repair": False})
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(gen_log) == 1                         # repair suppressed
    assert body["scorecard"]["hard_pass"] is False
    assert body["scorecard"]["repair_code"] == "intent_mismatch"
    assert "receipts" not in body


def test_route_technical_failure_still_repairs_without_evaluate(monkeypatch):
    """text.chat (evaluate off) whose worker dies: repair defaults ON because a
    technical check failed; WORKER_UNAVAILABLE with no alternative -> honest
    'none' decision on the wire, single receipt."""
    def dead(kwargs):
        raise ConnectionError("connection refused")
    client, _ = _client(monkeypatch, dead, [])
    resp = client.post("/oracle/route", json={"prompt": "hello",
                                              "capability": "text.chat"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["scorecard"]["hard_pass"] is False
    assert body["repair"]["action"] == "none"        # only-eligible qwen-chat
    assert "no eligible alternative" in body["repair"]["rationale"]
    assert "receipts" not in body


# ---------------------------------------------------------------------------
# k115 (or-k6): independent judge panel.
# ---------------------------------------------------------------------------

from hugpy_oracle.contracts import JudgeResult


def _jr(judge, verdict, score=None):
    return JudgeResult(judge=f"image_intent:{judge}", verdict=verdict, score=score)


@pytest.mark.parametrize("model_id, family", [
    ("Qwen/Qwen2.5-7B-Instruct-Q4_K_M", "qwen"),
    ("qwen-vl", "qwen"),
    ("qwen2-chat", "qwen"),
    ("meta-llama/Llama-3-8b", "llama"),
    ("llava-1.5-13b", "llava"),
    ("sdxl", "sdxl"),
    ("whisper-x", "whisper"),
    ("gemma2:9b", "gemma"),
    (None, ""),
])
def test_model_family_key(model_id, family):
    assert evaluation.model_family(model_id) == family


def test_fold_unanimous_split_single_none():
    p = evaluation.fold_verdicts([_jr("a", "YES", 90), _jr("b", "YES", 70)])
    assert p.confidence == 1.0 and p.verdict == "YES" and p.score == 80
    assert p.disagreements == () and p.limitations == ()

    p = evaluation.fold_verdicts([_jr("a", "YES", 90), _jr("b", "NO", 20)])
    assert p.confidence == 0.5 and p.verdict == "SPLIT"
    assert len(p.disagreements) == 2 and "image_intent:b: NO (20)" in p.disagreements

    p = evaluation.fold_verdicts([_jr("a", "YES", 90), _jr("b", "YES", 80), _jr("c", "NO", 10)])
    assert abs(p.confidence - 2 / 3) < 1e-3 and p.verdict == "YES"
    assert p.disagreements == ("image_intent:c: NO (10)",)

    p = evaluation.fold_verdicts([_jr("a", "YES", 90), _jr("b", "unavailable")])
    assert p.confidence == evaluation.SINGLE_JUDGE_CONFIDENCE
    assert p.limitations == ("single_judge",) and p.verdict == "YES"

    p = evaluation.fold_verdicts([_jr("a", "unavailable"), _jr("b", "unscored")])
    assert p.confidence == 0.0 and p.limitations == ("no_judge",)
    assert p.verdict == "unscored"


def _route(cap, model_id, model_ids):
    return router.RouteDecision(capability=cap, execution="execute", task="t",
                                model_id=model_id, model_ids=tuple(model_ids))


def test_resolve_judge_routes_excludes_generator_prior_picks_and_family(monkeypatch):
    from hugpy_oracle import selection
    pool = ("sdxl-vl", "qwen-vl", "qwen2-vl-7b", "llava-1.5", "gemma-vision")
    monkeypatch.setattr(evaluation, "_resolve_judge_route_excluding",
                        lambda cap, ex: _route(cap, "qwen-vl", pool))
    monkeypatch.setattr(selection, "requested_model_for",
                        lambda goal, cap, **kw: (None, None))
    monkeypatch.setattr(router, "resolve_route",
                        lambda goal, requested=None: _route(goal.capability, requested, pool))
    routes = evaluation._resolve_judge_routes("image.understand", ("sdxl",), 3)
    picked = [r.model_id for r in routes]
    # qwen2-vl-7b is the same family as qwen-vl; sdxl-vl is the generator's family
    assert picked == ["qwen-vl", "llava-1.5", "gemma-vision"]


def test_resolve_judge_routes_single_family_pool_yields_one(monkeypatch):
    from hugpy_oracle import selection
    pool = ("qwen-vl", "qwen2-vl-2b", "Qwen/Qwen2.5-VL-72B")
    monkeypatch.setattr(evaluation, "_resolve_judge_route_excluding",
                        lambda cap, ex: _route(cap, "qwen-vl", pool))
    monkeypatch.setattr(selection, "requested_model_for",
                        lambda goal, cap, **kw: ("qwen2-vl-2b", {}))
    monkeypatch.setattr(router, "resolve_route",
                        lambda goal, requested=None: _route(goal.capability, requested, pool))
    routes = evaluation._resolve_judge_routes("image.understand", ("sdxl",), 2)
    assert [r.model_id for r in routes] == ["qwen-vl"]


def test_resolve_judge_routes_refuses_generator_as_only_judge(monkeypatch):
    monkeypatch.setattr(evaluation, "_resolve_judge_route_excluding",
                        lambda cap, ex: _route(cap, "sdxl", ("sdxl",)))
    routes = evaluation._resolve_judge_routes("image.understand", ("sdxl",), 2)
    assert [r.model_id for r in routes] == ["sdxl"]  # run_judge turns this into the refusal


def _panel_routes(monkeypatch, *model_ids):
    monkeypatch.setattr(
        evaluation, "_resolve_judge_routes",
        lambda cap, ex, n: [_route(cap, m, model_ids) for m in model_ids][:n])


def _judge_by_model(monkeypatch, replies):
    calls = []

    def fake(task, body):
        calls.append(dict(body))
        return _Result(ok=True, text=replies[body["model_key"]])

    monkeypatch.setattr(evaluation, "_judge_dispatch", fake)
    return calls


def test_run_judges_two_independent_unanimous(monkeypatch, tmp_path):
    _panel_routes(monkeypatch, "qwen-vl", "llava-1.5")
    calls = _judge_by_model(monkeypatch, {"qwen-vl": "VERDICT=YES; SCORE=90; WHY=a",
                                          "llava-1.5": "VERDICT=YES; SCORE=70; WHY=b"})
    panel = evaluation.run_judges(evaluation.RUBRICS["image.generate"], _goal("x"),
                                  [_image_artifact(tmp_path)], generator_model="sdxl")
    assert [c["model_key"] for c in calls] == ["qwen-vl", "llava-1.5"]
    assert panel.confidence == 1.0 and panel.score == 80 and panel.limitations == ()
    card = evaluation.evaluate(_goal("x"), _image_route(), [_image_artifact(tmp_path)],
                               _receipt(), _passing_card())
    assert card.hard_pass and card.confidence == 1.0 and card.disagreements == ()
    assert len(card.judge_results) == 2


def test_evaluate_split_panel_records_disagreements_and_half_confidence(monkeypatch, tmp_path):
    _panel_routes(monkeypatch, "qwen-vl", "llava-1.5")
    _judge_by_model(monkeypatch, {"qwen-vl": "VERDICT=YES; SCORE=90; WHY=fine",
                                  "llava-1.5": "VERDICT=NO; SCORE=10; WHY=wrong subject"})
    card = evaluation.evaluate(_goal("x"), _image_route(), [_image_artifact(tmp_path)],
                               _receipt(), _passing_card())
    assert card.confidence == 0.5
    assert any(d.startswith("image_intent:llava-1.5: NO") for d in card.disagreements)
    assert any(d.startswith("image_intent:qwen-vl: YES") for d in card.disagreements)
    # mean 50 < balanced 60 -> fails on the panel, not on one voice
    assert card.hard_pass is False and card.repair_code is RepairCode.INTENT_MISMATCH
    assert "agreement 0.50" in card.diagnosis


def test_evaluate_single_independent_judge_is_named_not_faked(monkeypatch, tmp_path):
    _panel_routes(monkeypatch, "qwen-vl")
    _judge_by_model(monkeypatch, {"qwen-vl": "VERDICT=YES; SCORE=90; WHY=fine"})
    card = evaluation.evaluate(_goal("x"), _image_route(), [_image_artifact(tmp_path)],
                               _receipt(), _passing_card())
    assert card.hard_pass is True
    assert card.confidence == evaluation.SINGLE_JUDGE_CONFIDENCE
    assert "limitation:single_judge" in card.disagreements
    assert len(card.judge_results) == 1


def test_evaluate_second_judge_down_degrades_to_single(monkeypatch, tmp_path):
    _panel_routes(monkeypatch, "qwen-vl", "llava-1.5")

    def fake(task, body):
        if body["model_key"] == "llava-1.5":
            raise ConnectionError("vision plane down")
        return _Result(ok=True, text="VERDICT=YES; SCORE=90; WHY=fine")
    monkeypatch.setattr(evaluation, "_judge_dispatch", fake)
    card = evaluation.evaluate(_goal("x"), _image_route(), [_image_artifact(tmp_path)],
                               _receipt(), _passing_card())
    assert card.hard_pass is True and card.confidence == evaluation.SINGLE_JUDGE_CONFIDENCE
    assert [j.verdict for j in card.judge_results] == ["YES", "unavailable"]
    assert "limitation:single_judge" in card.disagreements


def test_evaluate_forwards_panel_confidence_to_ledger_when_supported(monkeypatch, tmp_path):
    from hugpy_oracle import selection
    seen = []
    monkeypatch.setattr(selection, "note_verdict",
                        lambda cap, model, *, hard_pass, confidence=1.0:
                        seen.append((cap, model, hard_pass, confidence)))
    _panel_routes(monkeypatch, "qwen-vl", "llava-1.5")
    _judge_by_model(monkeypatch, {"qwen-vl": "VERDICT=YES; SCORE=90; WHY=a",
                                  "llava-1.5": "VERDICT=YES; SCORE=80; WHY=b"})
    evaluation.evaluate(_goal("x"), _image_route(), [_image_artifact(tmp_path)],
                        _receipt(), _passing_card())
    assert seen == [("image.generate", "sdxl", True, 1.0)]


def test_run_judge_still_single_and_backward_compatible(monkeypatch, tmp_path):
    _stub_catalog(monkeypatch)
    calls = _judge_reply(monkeypatch, "VERDICT=YES; SCORE=90; WHY=good")
    res = evaluation.run_judge(evaluation.RUBRICS["image.generate"], _goal("x"),
                               [_image_artifact(tmp_path)], generator_model="sdxl")
    assert isinstance(res, JudgeResult) and res.verdict == "YES" and len(calls) == 1
