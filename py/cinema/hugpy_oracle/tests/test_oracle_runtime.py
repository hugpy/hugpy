"""k101b — oracle runtime: the bounded dispatch deadline, honest failure
logging, k98b's capability params, and the two speech request bodies.

The point of the deadline tests: the fleet's own retry ladder is deliberately
patient (``managers/resolvers/remote.py`` re-holds a cold/busy worker for up to
``HUGPY_COLD_HOLD_MAX_S`` — 25 minutes on the dev unit), which is right for a
job and a lie for a synchronous HTTP request. The oracle must stop waiting and
ANSWER, with a typed TIMEOUT receipt naming what it was holding on (doc
invariant 12), and the abandoned worker thread must never be able to write into
a response that already went out.

No GPU, no network, no workers: every dispatch seam is monkeypatched.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_runtime.py -q
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time

logging.disable(logging.INFO)  # silence the models_config registry chatter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from hugpy_oracle import runtime, scorecard
from hugpy_oracle.authority import AuthorityDecision
from hugpy_oracle.contracts import (
    ArtifactKind,
    AuthorityKind,
    BudgetHints,
    FailureClass,
    GoalSpec,
    InputKind,
    InputRef,
    RepairCode,
)
from hugpy_oracle.router import RouteDecision

_LOGGER = "hugpy_oracle.runtime"


class _Result:
    def __init__(self, **payload):
        self._payload = payload

    def model_dump(self):
        return dict(self._payload)


def _goal(prompt="summarize the quarterly report", inputs=(), capability=None,
          budget=None, rights=None):
    return GoalSpec(objective=prompt, raw_prompt=prompt, inputs=tuple(inputs),
                    capability=capability, rights=rights,
                    budget=budget or BudgetHints())


def _route(capability="text.summarize", task="text-summarization",
           model_id="flan-t5-large", placement="auto", params=None,
           authority=None):
    return RouteDecision(
        capability=capability, execution="execute", source="tasks", task=task,
        model_id=model_id, model_rationale="default", placement=placement,
        produces=(ArtifactKind.TEXT,), dispatch_params=dict(params or {}),
        authority=authority)


def _patch_dispatch(monkeypatch, fn):
    monkeypatch.setattr(runtime, "_dispatch", fn)
    monkeypatch.setattr(runtime, "_normalized_kwargs",
                        lambda task, body: dict(body, task=task))
    # The "what was I holding on?" lookup must never reach the live fleet from
    # a test; the real seam is exercised separately below.
    monkeypatch.setattr(runtime, "_selected_worker",
                        lambda model, task, pool: "a-brain")


# ---------------------------------------------------------------------------
# sync_deadline_s — where the bound comes from.
# ---------------------------------------------------------------------------


def test_sync_deadline_defaults_to_sixty_seconds(monkeypatch):
    monkeypatch.delenv(runtime.SYNC_DEADLINE_ENV, raising=False)
    assert runtime.sync_deadline_s(_goal()) == runtime.DEFAULT_SYNC_DEADLINE_S
    assert runtime.sync_deadline_s(None) == runtime.DEFAULT_SYNC_DEADLINE_S


def test_sync_deadline_reads_the_env_and_clamps_it(monkeypatch):
    monkeypatch.setenv(runtime.SYNC_DEADLINE_ENV, "30")
    assert runtime.sync_deadline_s(_goal()) == 30.0
    # Below the floor nothing on this fleet can answer; above the ceiling the
    # HTTP request is fiction (gunicorn's own --timeout is 120s here).
    monkeypatch.setenv(runtime.SYNC_DEADLINE_ENV, "0.5")
    assert runtime.sync_deadline_s(_goal()) == runtime.MIN_SYNC_DEADLINE_S
    monkeypatch.setenv(runtime.SYNC_DEADLINE_ENV, "99999")
    assert runtime.sync_deadline_s(_goal()) == runtime.MAX_SYNC_DEADLINE_S
    # An unparseable setting degrades to the default; it never crashes a route.
    monkeypatch.setenv(runtime.SYNC_DEADLINE_ENV, "soon")
    assert runtime.sync_deadline_s(_goal()) == runtime.DEFAULT_SYNC_DEADLINE_S


def test_the_goal_budget_hint_wins_over_the_env(monkeypatch):
    monkeypatch.setenv(runtime.SYNC_DEADLINE_ENV, "300")
    goal = _goal(budget=BudgetHints(max_seconds=20))
    assert runtime.sync_deadline_s(goal) == 20.0


# ---------------------------------------------------------------------------
# run_bounded + the one-shot handoff.
# ---------------------------------------------------------------------------


def test_run_bounded_returns_a_fast_result():
    assert runtime.run_bounded(lambda: "quick", 5.0, "test") == "quick"


def test_run_bounded_reraises_in_the_calling_thread():
    def boom():
        raise ConnectionError("connection refused")

    with pytest.raises(ConnectionError):
        runtime.run_bounded(boom, 5.0, "test")


def test_run_bounded_raises_dispatch_timeout_on_expiry():
    release = threading.Event()
    try:
        t0 = time.monotonic()
        with pytest.raises(runtime.DispatchTimeout):
            runtime.run_bounded(lambda: release.wait(30), 0.3, "test")
        assert time.monotonic() - t0 < 1.0
    finally:
        release.set()


def test_run_bounded_refuses_a_spent_deadline():
    calls = []
    with pytest.raises(runtime.DispatchTimeout):
        runtime.run_bounded(lambda: calls.append(1), 0.0, "test")
    assert calls == []  # no thread is even started once the budget is gone


def test_the_one_shot_slot_can_be_read_once_and_then_closes():
    slot = runtime._Handoff()
    assert slot.deliver("returned", 1) is True
    assert slot.deliver("returned", 2) is False   # single slot, first write wins
    assert slot.claim() == ("returned", 1)
    assert slot.claim() is None                   # taken once, never twice
    assert slot.closed is True
    # The whole race this exists for: a worker finishing AFTER the deadline.
    assert slot.deliver("returned", 3) is False


# ---------------------------------------------------------------------------
# execute_route under the deadline.
# ---------------------------------------------------------------------------


def test_execute_route_times_out_and_ends_honestly(monkeypatch):
    """A dispatch that outlives the deadline becomes a typed TIMEOUT receipt in
    about one deadline — not a hung connection."""
    goal, route = _goal(), _route(placement="a-brain")
    release, started, late = threading.Event(), threading.Event(), {}

    def stalled(kwargs):
        started.set()
        release.wait(30)          # the fleet's cold hold, in miniature
        late["finished"] = True
        return _Result(ok=True, text="too late to matter")

    _patch_dispatch(monkeypatch, stalled)
    try:
        t0 = time.monotonic()
        artifacts, receipt = runtime.execute_route(goal, route, deadline_s=1.0)
        elapsed = time.monotonic() - t0
    finally:
        release.set()

    assert started.is_set()
    assert 1.0 <= elapsed <= 1.5, elapsed
    assert receipt.failure is FailureClass.TIMEOUT
    assert artifacts == []
    assert receipt.duration_s >= 1.0

    # The reason names what the wait was holding on — a timeout that cannot say
    # WHICH model on WHICH worker is half a diagnosis.
    reason = " ".join(receipt.log_excerpt)
    assert "flan-t5-large" in reason
    assert "a-brain" in reason
    assert "text-summarization" in reason

    card = scorecard.build_technical_scorecard(goal, route, artifacts, receipt)
    assert card.hard_pass is False
    assert card.repair_code is RepairCode.TIMEOUT

    # …and the orphaned thread, when it finally finishes, changes nothing.
    for _ in range(200):
        if late.get("finished"):
            break
        time.sleep(0.01)
    assert late.get("finished") is True
    assert receipt.failure is FailureClass.TIMEOUT
    assert artifacts == []
    assert receipt.artifacts == ()


def test_execute_route_never_retries_a_timeout(monkeypatch):
    """WORKER_UNAVAILABLE is retried once; a timeout is not — a second wait
    would double the very hang the bound exists to end."""
    release, calls = threading.Event(), []

    def stalled(kwargs):
        calls.append(kwargs)
        release.wait(30)
        return _Result(ok=True, text="late")

    _patch_dispatch(monkeypatch, stalled)
    try:
        _arts, receipt = runtime.execute_route(_goal(), _route(), deadline_s=0.4)
    finally:
        release.set()
    assert len(calls) == 1
    assert receipt.retries == 0
    assert receipt.failure is FailureClass.TIMEOUT


def test_execute_route_logs_the_timeout(monkeypatch, caplog):
    release = threading.Event()
    _patch_dispatch(monkeypatch, lambda kwargs: release.wait(30))
    try:
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            runtime.execute_route(_goal(), _route(), deadline_s=0.4)
    finally:
        release.set()
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "failure_class=timeout" in logged
    assert "text.summarize" in logged


def test_a_fast_dispatch_is_untouched_by_the_deadline(monkeypatch):
    seen = {}

    def quick(kwargs):
        seen.update(kwargs)
        return _Result(ok=True, text="the answer", model_key="flan-t5-large")

    _patch_dispatch(monkeypatch, quick)
    artifacts, receipt = runtime.execute_route(_goal(), _route(), deadline_s=5.0)
    assert receipt.failure is None
    assert artifacts[0]["text"] == "the answer"
    assert seen["task"] == "text-summarization"
    assert receipt.warnings == ()


def test_execute_route_logs_a_classified_failure(monkeypatch, caplog):
    """No silent receipting: a classified failure is receipt data AND a log
    line, or a stalled fleet stays invisible until a caller complains."""
    def boom(kwargs):
        raise RuntimeError("runner exploded while decoding")

    _patch_dispatch(monkeypatch, boom)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        artifacts, receipt = runtime.execute_route(_goal(), _route(),
                                                   deadline_s=5.0)
    assert artifacts == []
    assert receipt.failure is FailureClass.RUNNER_ERROR
    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert records, "a classified failure must be logged"
    logged = " ".join(r.getMessage() for r in records)
    assert "failure_class=runner_error" in logged
    assert "text.summarize" in logged
    assert "flan-t5-large" in logged
    assert "runner exploded" in logged


def test_a_worker_unavailable_failure_is_logged_once_after_its_retry(monkeypatch):
    def dead(kwargs):
        raise ConnectionError("worker unreachable")

    _patch_dispatch(monkeypatch, dead)
    _arts, receipt = runtime.execute_route(_goal(), _route(), deadline_s=5.0)
    assert receipt.retries == 1
    assert receipt.failure is FailureClass.WORKER_UNAVAILABLE


# ---------------------------------------------------------------------------
# _dispatch_target — what the expired wait was holding on.
# ---------------------------------------------------------------------------


def test_dispatch_target_prefers_the_placement_pin(monkeypatch):
    monkeypatch.setattr(runtime, "_selected_worker",
                        lambda *a: pytest.fail("a pinned route needs no lookup"))
    target = runtime._dispatch_target(_route(placement="ae"), {})
    assert "'ae'" in target and "flan-t5-large" in target


def test_dispatch_target_falls_back_to_the_live_selection(monkeypatch):
    monkeypatch.setattr(runtime, "_selected_worker",
                        lambda model, task, pool: "a-brain")
    assert "a-brain" in runtime._dispatch_target(_route(), {})


def test_dispatch_target_says_unknown_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(runtime, "_selected_worker",
                        lambda model, task, pool: None)
    route = _route(model_id=None)
    target = runtime._dispatch_target(route, {})
    assert "unknown" in target


def test_a_broken_worker_lookup_never_costs_the_diagnosis(monkeypatch):
    def broken(model, task, pool):
        raise RuntimeError("registry read failed")

    monkeypatch.setattr(runtime, "_selected_worker", broken)
    target = runtime._dispatch_target(_route(), {})
    assert "flan-t5-large" in target
    assert "worker unknown" in target


# ---------------------------------------------------------------------------
# k98b's dispatch_params: capability-defining, so the catalog wins.
# ---------------------------------------------------------------------------


def _wt_goal():
    return _goal("transcribe with timings",
                 inputs=[InputRef(kind=InputKind.AUDIO, ref="/tmp/a.wav")],
                 capability="audio.transcribe.word_timestamps")


def _wt_route(params=None):
    return _route(capability="audio.transcribe.word_timestamps",
                  task="automatic-speech-recognition", model_id="whisper-x",
                  params={"word_timestamps": True} if params is None else params)


def test_dispatch_params_are_merged_into_the_dispatch_kwargs(monkeypatch):
    monkeypatch.setattr(runtime, "_normalized_kwargs",
                        lambda task, body: dict(body, task=task))
    _body, kwargs, warnings = runtime.dispatch_kwargs(_wt_goal(), _wt_route())
    assert kwargs["word_timestamps"] is True
    assert kwargs["file"] == "/tmp/a.wav"
    assert kwargs["model_key"] == "whisper-x"
    assert warnings == []


def test_capability_params_win_over_caller_kwargs(monkeypatch):
    """catalog params are capability-DEFINING, not preferences: a
    word_timestamps run dispatched with word_timestamps=False would be a
    different capability wearing this one's name. The override is recorded."""
    monkeypatch.setattr(runtime, "_normalized_kwargs",
                        lambda task, body: dict(body, task=task))
    _body, kwargs, warnings = runtime.dispatch_kwargs(
        _wt_goal(), _wt_route(), overrides={"word_timestamps": False,
                                            "temperature": 0.1})
    assert kwargs["word_timestamps"] is True          # the catalog wins
    assert kwargs["temperature"] == 0.1               # everything else does not
    assert warnings and "word_timestamps" in warnings[0]


def test_the_override_warning_reaches_the_receipt(monkeypatch):
    _patch_dispatch(monkeypatch, lambda kwargs: _Result(ok=True, text="words"))
    _arts, receipt = runtime.execute_route(
        _wt_goal(), _wt_route(), overrides={"word_timestamps": False},
        deadline_s=5.0)
    assert receipt.request_dict()["word_timestamps"] is True
    assert any("word_timestamps" in w for w in receipt.warnings)


def test_a_route_without_capability_params_is_unchanged(monkeypatch):
    monkeypatch.setattr(runtime, "_normalized_kwargs",
                        lambda task, body: dict(body, task=task))
    _body, kwargs, warnings = runtime.dispatch_kwargs(_goal(), _route())
    assert warnings == []
    assert "word_timestamps" not in kwargs


# ---------------------------------------------------------------------------
# build_request_body — the two speech capabilities (k98).
# ---------------------------------------------------------------------------


def test_build_request_body_word_timestamps():
    body = runtime.build_request_body(_wt_goal(), _wt_route())
    assert body == {"file": "/tmp/a.wav", "word_timestamps": True,
                    "model_key": "whisper-x"}


def test_word_timestamps_without_audio_is_a_typed_shape_error():
    goal = _goal("transcribe", capability="audio.transcribe.word_timestamps")
    with pytest.raises(runtime.GoalShapeError) as exc:
        runtime.build_request_body(goal, _wt_route())
    assert "audio" in str(exc.value)


def _tts_route(authority=None, model_id="chatterbox"):
    return _route(capability="audio.tts", task="text-to-speech",
                  model_id=model_id, authority=authority)


def test_build_request_body_audio_tts_speaks_the_prompt():
    goal = _goal("say hello to the room", capability="audio.tts")
    body = runtime.build_request_body(goal, _tts_route())
    assert body["text"] == "say hello to the room"
    assert "reference_audio" not in body      # the default voice, honestly
    assert body["model_key"] == "chatterbox"


def test_build_request_body_audio_tts_prefers_the_text_inputs():
    goal = _goal("read these lines", capability="audio.tts",
                 inputs=[InputRef(kind=InputKind.TEXT, ref="line one"),
                         InputRef(kind=InputKind.TEXT, ref="line two")])
    body = runtime.build_request_body(goal, _tts_route())
    assert body["text"] == "line one\n\nline two"


def test_build_request_body_audio_tts_needs_a_line():
    goal = _goal("say it", capability="audio.tts",
                 inputs=[InputRef(kind=InputKind.TEXT, ref="   ")])
    with pytest.raises(runtime.GoalShapeError) as exc:
        runtime.build_request_body(goal, _tts_route())
    assert "line to speak" in str(exc.value)


def test_audio_tts_marks_a_reference_authorized_only_when_the_gate_cleared_it():
    ref = InputRef(kind=InputKind.AUDIO, ref="/tmp/voice.wav", label="voice ref")
    goal = _goal("say the line", capability="audio.tts", inputs=[ref])
    cleared = AuthorityDecision(
        ok=True, reason="granted",
        required=((AuthorityKind.VOICE, "voice_profile:mira"),))
    body = runtime.build_request_body(goal, _tts_route(authority=cleared))
    assert body["reference_audio"] == "/tmp/voice.wav"
    assert body["authorized"] is True


def test_audio_tts_never_claims_authorization_the_gate_did_not_give():
    """An audio input the gate did not read as a voice reference arrives
    unauthorized, and the TTS runner refuses it. A silent fall back to the
    default voice — or a clone nobody cleared — would both be worse."""
    ref = InputRef(kind=InputKind.AUDIO, ref="/tmp/bed.wav", label="music bed")
    goal = _goal("say the line", capability="audio.tts", inputs=[ref])
    nothing_required = AuthorityDecision(ok=True, reason="no typed authority "
                                         "required", required=())
    for authority in (None, nothing_required):
        body = runtime.build_request_body(goal, _tts_route(authority=authority))
        assert body["reference_audio"] == "/tmp/bed.wav"
        assert body["authorized"] is False


# ---------------------------------------------------------------------------
# k105 — registry_version stamped on every receipt execute_route builds.
# ---------------------------------------------------------------------------


def test_catalog_registry_version_returns_none_on_a_catalog_fault(monkeypatch, caplog):
    """The lazy seam never lets a broken catalog crash the caller — it reads
    ``None`` and says why in the log, exactly like every other 'ineligible >
    faked' guard in this module."""
    from hugpy_oracle import catalog as oracle_catalog

    def boom():
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(oracle_catalog, "registry_version", boom)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert runtime._catalog_registry_version() is None
    assert any("registry_version" in r.getMessage() for r in caplog.records)


def test_execute_route_stamps_the_given_registry_version(monkeypatch):
    """A caller (the route) that already read the catalog hands the version
    straight through — no second catalog read inside execute_route."""
    seen = {}

    def quick(kwargs):
        return _Result(ok=True, text="hi")

    _patch_dispatch(monkeypatch, quick)

    def must_not_be_called():
        seen["called"] = True
        return "sha256:should-not-be-used-00"

    monkeypatch.setattr(runtime, "_catalog_registry_version", must_not_be_called)
    _artifacts, receipt = runtime.execute_route(
        _goal(), _route(), registry_version="sha256:deadbeefcafef00d")
    assert receipt.registry_version == "sha256:deadbeefcafef00d"
    assert "called" not in seen


def test_execute_route_falls_back_to_catalog_registry_version_when_not_given(monkeypatch):
    monkeypatch.setattr(runtime, "_catalog_registry_version",
                        lambda: "sha256:fallback0000001")
    _patch_dispatch(monkeypatch, lambda kwargs: _Result(ok=True, text="hi"))
    _artifacts, receipt = runtime.execute_route(_goal(), _route())
    assert receipt.registry_version == "sha256:fallback0000001"


def test_execute_route_never_crashes_when_the_catalog_cannot_be_read(monkeypatch):
    """A catalog fault answers with an honest ``None`` on the receipt, never a
    500 in place of a finished execution."""
    monkeypatch.setattr(runtime, "_catalog_registry_version", lambda: None)
    _patch_dispatch(monkeypatch, lambda kwargs: _Result(ok=True, text="hi"))
    _artifacts, receipt = runtime.execute_route(_goal(), _route())
    assert receipt.registry_version is None


def test_timeout_receipt_carries_the_registry_version(monkeypatch):
    release = threading.Event()
    _patch_dispatch(monkeypatch, lambda kwargs: release.wait(30))
    try:
        _artifacts, receipt = runtime.execute_route(
            _goal(), _route(), deadline_s=0.4,
            registry_version="sha256:timeoutversion01")
    finally:
        release.set()
    assert receipt.failure is FailureClass.TIMEOUT
    assert receipt.registry_version == "sha256:timeoutversion01"


def test_a_classified_failure_receipt_carries_the_registry_version(monkeypatch):
    def boom(kwargs):
        raise RuntimeError("kaboom")

    _patch_dispatch(monkeypatch, boom)
    _artifacts, receipt = runtime.execute_route(
        _goal(), _route(), registry_version="sha256:failureversion1")
    assert receipt.failure is FailureClass.RUNNER_ERROR
    assert receipt.registry_version == "sha256:failureversion1"
