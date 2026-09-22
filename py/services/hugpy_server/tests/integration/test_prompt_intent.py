"""INTENT ROUTER — POST /video/prompt/intent (STUDIO-SPREAD-SPEC §1d).

Fork 1 was settled AGAINST a word-count heuristic, and these lock the four
properties that make the router safe to put in front of a text box the user is
still typing in:

  1. BLANK SHORT-CIRCUITS with no model call at all — the commonest case must not
     cost a GPU round trip or a spinner.
  2. LOW CONFIDENCE IS "ambiguous" — below 0.80 the UI shows BOTH actions rather
     than pre-arming the wrong one. The router's own guess still rides along
     (``detected_intent``) so the UI can hint without committing.
  3. FAILURE DEGRADES, NEVER 500s. No worker, junk reply, unknown intent — all of
     them return 200 with "ambiguous". A classifier outage must not make a text
     field unusable.
  4. CACHED BY INPUT HASH. Classification happens on blur; a user who blurs,
     edits nothing, and blurs again must not pay twice.

⚠ EXECUTOR MOCKING. ``classify_intent`` reaches the fleet through
``utils.no_think.execute_prompt_no_think``, which does ``from
..managers.dispatch import execute_prompt``. That reads the sys.modules ENTRY —
patching the ``dispatch`` ATTRIBUTE gives you the inner dispatch.dispatch module
and quietly runs REAL inference. ``_patch_dispatch`` below mirrors
tests/test_no_think.py exactly.

Run (pytest or as a script):
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python -m pytest tests/test_prompt_intent.py -q
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

from hugpy_video.intel import prompt_intent as PI

vr = importlib.import_module("hugpy_server.app.routes.video_routes")
app = Flask(__name__)
app.register_blueprint(vr.video_bp)
client = app.test_client()

_SCENE = "A woman enters a red room."
_DIRECTION = "keep her wardrobe unchanged, but make the framing wider and colder"


@pytest.fixture(autouse=True)
def _clean_cache():
    PI.cache_clear()
    yield
    PI.cache_clear()


def _executor(intent="direction", confidence=0.94, text=None, ok=True,
              raises=None, calls=None):
    """A fake in ``execute_prompt_no_think``'s RETURN shape (the seam's dict)."""
    def fake(*args, **kwargs):
        if calls is not None:
            calls.append(kwargs)
        if raises is not None:
            raise raises
        body = text if text is not None else json.dumps(
            {"intent": intent, "confidence": confidence})
        return {"ok": ok, "text": body, "reasoning": "",
                "thinking_suppressed": True, "error": None}
    return fake


def _patch_dispatch(monkeypatch, fake):
    """See the module docstring — patch the sys.modules ENTRY, not the attribute."""
    monkeypatch.setattr(
        sys.modules["hugpy_engine.dispatch"], "execute_prompt", fake)


# --------------------------------------------------------------------------- #
# 1. BLANK SHORT-CIRCUIT — no model call
# --------------------------------------------------------------------------- #
def test_blank_short_circuits_without_a_model_call():
    def explode(*a, **k):
        raise AssertionError("the router was called for an empty field")

    out = PI.classify_intent("   ", executor=explode)
    assert out["intent"] == "empty"
    assert out["operation"] == "generate"
    assert out["confidence"] == 1.0
    assert out["model"] is None


def test_blank_short_circuits_through_the_route(monkeypatch):
    _patch_dispatch(monkeypatch, _executor(raises=AssertionError("called!")))
    r = client.post("/video/prompt/intent", json={"text": ""})
    assert r.status_code == 200
    assert r.get_json()["intent"] == "empty"


def test_none_text_is_empty_too():
    assert PI.classify_intent(None)["intent"] == "empty"


# --------------------------------------------------------------------------- #
# 2. CLASSIFICATION + THE CONFIDENCE FLOOR
# --------------------------------------------------------------------------- #
def test_a_direction_routes_to_generate_from_direction():
    out = PI.classify_intent(_DIRECTION,
                             executor=_executor("direction", 0.94))
    assert out["intent"] == "direction"
    assert out["operation"] == "generate_from_direction"
    assert out["confidence"] == 0.94
    assert out["degraded"] is False


def test_a_scene_routes_to_enhance():
    out = PI.classify_intent(_SCENE, executor=_executor("scene_prompt", 0.91))
    assert out["intent"] == "scene_prompt"
    assert out["operation"] == "enhance_scene"


def test_low_confidence_becomes_ambiguous_and_offers_no_default_action():
    out = PI.classify_intent("make it red", executor=_executor("direction", 0.62))
    assert out["intent"] == "ambiguous"
    assert out["operation"] is None, "ambiguous must not pre-arm a button"
    assert out["detected_intent"] == "direction"   # the UI may still hint
    assert "both actions" in out["reason"]


def test_the_floor_is_the_spec_floor():
    assert PI.CONFIDENCE_FLOOR == 0.80
    assert PI.classify_intent("x", executor=_executor("direction", 0.80))["intent"] \
        == "direction"
    assert PI.classify_intent("y", executor=_executor("direction", 0.79))["intent"] \
        == "ambiguous"


# --------------------------------------------------------------------------- #
# 3. HONEST DEGRADE — never a 500, never a block
# --------------------------------------------------------------------------- #
def test_router_exception_degrades_to_ambiguous():
    out = PI.classify_intent(_SCENE,
                             executor=_executor(raises=RuntimeError("no worker")))
    assert out["intent"] == "ambiguous"
    assert out["degraded"] is True
    assert out["operation"] is None


def test_junk_output_degrades_to_ambiguous():
    out = PI.classify_intent(_SCENE, executor=_executor(text="I think it's a scene!"))
    assert out["intent"] == "ambiguous"
    assert out["degraded"] is True


def test_an_unknown_intent_value_degrades():
    out = PI.classify_intent(
        _SCENE, executor=_executor(text='{"intent": "vibes", "confidence": 0.99}'))
    assert out["intent"] == "ambiguous"
    assert out["degraded"] is True


def test_a_failed_generation_degrades():
    out = PI.classify_intent(_SCENE, executor=_executor(text="", ok=False))
    assert out["intent"] == "ambiguous"
    assert out["degraded"] is True


def test_the_route_degrades_with_a_200_not_a_500(monkeypatch):
    _patch_dispatch(monkeypatch, _executor(raises=RuntimeError("no live worker")))
    r = client.post("/video/prompt/intent", json={"text": _SCENE})
    assert r.status_code == 200
    body = r.get_json()
    assert body["intent"] == "ambiguous" and body["degraded"] is True


# --------------------------------------------------------------------------- #
# 4. CACHE BY INPUT HASH
# --------------------------------------------------------------------------- #
def test_a_repeat_classification_is_served_from_cache():
    calls = []
    ex = _executor("direction", 0.94, calls=calls)
    first = PI.classify_intent(_DIRECTION, executor=ex)
    second = PI.classify_intent(_DIRECTION, executor=ex)
    assert len(calls) == 1, "the router was asked the same question twice"
    assert first["cached"] is False and second["cached"] is True
    assert second["intent"] == first["intent"]
    assert PI.cache_stats()["hits"] == 1


def test_different_text_is_a_different_cache_entry():
    calls = []
    ex = _executor("direction", 0.94, calls=calls)
    PI.classify_intent(_DIRECTION, executor=ex)
    PI.classify_intent(_SCENE, executor=ex)
    assert len(calls) == 2


def test_a_degraded_answer_is_not_cached():
    """A classifier outage must not poison the field for the rest of the process."""
    calls = []
    PI.classify_intent(_SCENE, executor=_executor(raises=RuntimeError("down")))
    out = PI.classify_intent(_SCENE, executor=_executor("scene_prompt", 0.95,
                                                        calls=calls))
    assert len(calls) == 1
    assert out["intent"] == "scene_prompt"


# --------------------------------------------------------------------------- #
# the fleet call itself: temp 0, <=100 tokens, no-think, 3B router
# --------------------------------------------------------------------------- #
def test_the_router_call_is_temp_zero_capped_and_no_think(monkeypatch):
    seen = {}

    def fake_execute_prompt(*args, **kwargs):
        seen.update(kwargs)
        return {"ok": True, "text": '{"intent": "scene_prompt", "confidence": 0.9}'}

    _patch_dispatch(monkeypatch, fake_execute_prompt)
    out = PI.classify_intent(_SCENE)          # real seam, faked dispatch
    assert out["intent"] == "scene_prompt"
    assert seen["model_key"] == "Qwen2.5-3B-Instruct-GGUF"
    assert seen["temperature"] == 0.0
    assert seen["max_new_tokens"] == 100
    assert "/no_think" in seen["messages"][-1]["content"]


def test_think_blocks_are_stripped_before_the_json_is_read(monkeypatch):
    _patch_dispatch(monkeypatch, lambda *a, **k: {
        "ok": True,
        "text": '<think>hmm {maybe} a scene?</think>'
                '{"intent": "scene_prompt", "confidence": 0.93}'})
    out = PI.classify_intent(_SCENE)
    assert out["intent"] == "scene_prompt"
    assert out["confidence"] == 0.93


def test_a_caller_may_override_the_router_model():
    calls = []
    PI.classify_intent(_SCENE, model="some-other-3b",
                       executor=_executor("scene_prompt", 0.9, calls=calls))
    assert calls[0]["model_key"] == "some-other-3b"


# --------------------------------------------------------------------------- #
# route surface
# --------------------------------------------------------------------------- #
def test_route_returns_the_classification(monkeypatch):
    _patch_dispatch(monkeypatch, _executor("direction", 0.95))
    r = client.post("/video/prompt/intent",
                    json={"text": _DIRECTION, "scope": "segment"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["intent"] == "direction"
    assert body["scope"] == "segment"
    assert body["operation"] == "generate_from_direction"
    assert body["confidence"] == 0.95


def test_route_rejects_a_bad_scope():
    r = client.post("/video/prompt/intent", json={"text": "x", "scope": "galaxy"})
    assert r.status_code == 400


def test_route_rejects_non_string_text():
    r = client.post("/video/prompt/intent", json={"text": {"nope": 1}})
    assert r.status_code == 400


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
