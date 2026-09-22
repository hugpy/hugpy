"""INTENT ROUTER — is what the user typed a SCENE or a DIRECTION? (SPEC §1d)

THE PROBLEM, AND WHY WORD COUNT IS NOT THE ANSWER
-------------------------------------------------
A studio prompt field receives two completely different kinds of text:

    scene prompt : "A woman enters a red room."          -> enhance / normalize
    direction    : "keep her wardrobe unchanged, but      -> generate a new prompt
                    make the framing wider and colder"      from this instruction

The obvious heuristic — short means scene, long means direction — is BACKWARDS on
both of those examples (fork 1, settled in the spec: no word-count heuristic).
The distinction is grammatical and semantic, not dimensional: a scene DESCRIBES a
frame; a direction INSTRUCTS a change. A 3B instruct model gets this right; a
length threshold cannot.

WHAT THIS IS ALLOWED TO DO
--------------------------
Nothing destructive. The classification is a VISIBLE, OVERRIDABLE DEFAULT that
selects which button the UI pre-arms. A router mistake must never overwrite user
work, so:

  * blank input SHORT-CIRCUITS to ``empty`` with NO model call at all — the
    single most common case must not cost a GPU round trip or a spinner;
  * confidence below :data:`CONFIDENCE_FLOOR` returns ``ambiguous``, and the UI
    shows BOTH actions rather than guessing;
  * ANY failure — no worker, unparseable reply, model missing — degrades to
    ``ambiguous`` too. This route never 500s and never blocks the user, because
    a classifier outage must not make a text box unusable.

CACHING
-------
Classify on blur / action-invoke, never per keystroke — but a user who blurs a
field, edits nothing, and blurs it again would still pay twice, and the composer
has many fields. So results are memoized by a hash of (model, scope, text) in a
small process-local LRU. Deliberately in-memory: it is a latency nicety over a
pure function of the input, not state worth persisting or sharing across gunicorn
workers.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ROUTER_MODEL",
    "CONFIDENCE_FLOOR",
    "ROUTER_SYSTEM",
    "VALID_INTENTS",
    "OPERATION_FOR_INTENT",
    "classify_intent",
    "cache_clear",
    "cache_stats",
]

#: SPEC §3. Small, fast, and — unlike the assist default — NOT a reasoning model;
#: a router that thinks for 200 tokens before answering defeats the purpose.
ROUTER_MODEL = "Qwen2.5-3B-Instruct-GGUF"

#: Below this the answer is not trustworthy enough to pre-arm a button (§1d.4).
CONFIDENCE_FLOOR = 0.80

VALID_INTENTS = ("empty", "direction", "scene_prompt", "ambiguous")

#: What the UI should DO for each classification. ``ambiguous`` maps to None on
#: purpose — there is no default action, which is exactly what "show both" means.
OPERATION_FOR_INTENT: Dict[str, Optional[str]] = {
    "empty": "generate",
    "scene_prompt": "enhance_scene",
    "direction": "generate_from_direction",
    "ambiguous": None,
}

ROUTER_SYSTEM = (
    "You classify one piece of text typed into a film shot's prompt field.\n"
    "\n"
    'A "scene_prompt" DESCRIBES what is in the shot — a place, people, an '
    "action, a look. Example: \"A woman enters a red room.\" (short, but a "
    "scene).\n"
    'A "direction" INSTRUCTS a change to the shot rather than describing it — '
    "it tells the writer what to do, keep, remove, or adjust. Example: \"keep "
    "her wardrobe unchanged, but make the framing wider and colder\" (long, but "
    "a direction).\n"
    "Length is irrelevant. Decide by whether the text DESCRIBES a frame or "
    "INSTRUCTS a change.\n"
    "\n"
    "Reply with ONLY this JSON object and nothing else:\n"
    '{"intent": "direction" or "scene_prompt", "confidence": 0.0 to 1.0}\n'
    "Use a confidence below 0.8 when the text is genuinely both or neither."
)

_MAX_TOKENS = 100
_CACHE_MAX = 256
_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_cache_lock = Lock()
_hits = 0
_misses = 0


def _cache_key(text: str, scope: str, model: str) -> str:
    blob = "\x00".join((model, scope, text))
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()


def cache_clear() -> None:
    global _hits, _misses
    with _cache_lock:
        _cache.clear()
        _hits = _misses = 0


def cache_stats() -> Dict[str, int]:
    with _cache_lock:
        return {"size": len(_cache), "hits": _hits, "misses": _misses}


def _result(intent: str, confidence: float, scope: str, model: Optional[str],
            *, cached: bool = False, degraded: bool = False,
            reason: str = "", detected: Optional[str] = None) -> Dict[str, Any]:
    out = {
        "intent": intent,
        "scope": scope,
        "operation": OPERATION_FOR_INTENT.get(intent),
        "confidence": round(float(confidence), 3),
        "model": model,
        "cached": cached,
        "degraded": degraded,
    }
    if reason:
        out["reason"] = reason
    # What the router ACTUALLY said before the confidence floor demoted it —
    # so a UI can show "probably a direction" while still offering both.
    if detected and detected != intent:
        out["detected_intent"] = detected
    return out


def classify_intent(text: Any, scope: str = "segment",
                    model: Optional[str] = None,
                    executor=None) -> Dict[str, Any]:
    """Classify ``text``. Always returns a result dict — never raises.

    ``executor`` is injectable for tests; production uses the package no-think
    seam (``utils.no_think.execute_prompt_no_think``), so nothing new crosses the
    wire and any ``<think>`` the router emits is stripped before the JSON
    scavenger ever sees it.
    """
    global _hits, _misses
    model_key = (model or ROUTER_MODEL).strip() or ROUTER_MODEL
    body = text.strip() if isinstance(text, str) else ""

    # 1) Blank short-circuits WITHOUT a model call (§1d.1). Confidence is 1.0
    #    because "the field is empty" is not an inference.
    if not body:
        return _result("empty", 1.0, scope, None)

    key = _cache_key(body, scope, model_key)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            _hits += 1
            return dict(hit, cached=True)
        _misses += 1

    if executor is None:
        from hugpy_engine.utils.no_think import execute_prompt_no_think
        executor = execute_prompt_no_think

    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": "Classify this text:\n\n" + body},
    ]
    try:
        # temperature 0 + a hard token ceiling (§1d): the router must be
        # deterministic and cheap. Both are EXISTING ChatRequest fields, so this
        # is not a wire change.
        reply = executor(
            model_key=model_key,
            messages=messages,
            task="text-generation",
            max_new_tokens=_MAX_TOKENS,
            temperature=0.0,
            do_sample=False,
        )
    except Exception as exc:  # noqa: BLE001 — a classifier outage is never fatal
        logger.warning("intent router unavailable (%s): %s", model_key, exc)
        return _result("ambiguous", 0.0, scope, model_key, degraded=True,
                       reason="the intent router was unavailable")

    if not isinstance(reply, dict) or not reply.get("ok") or not reply.get("text"):
        err = (reply or {}).get("error") if isinstance(reply, dict) else None
        return _result("ambiguous", 0.0, scope, model_key, degraded=True,
                       reason=err or "the intent router returned nothing usable")

    from hugpy_engine.utils.json_scavenge import extract_json_object
    parsed = extract_json_object(reply["text"])
    if not isinstance(parsed, dict):
        return _result("ambiguous", 0.0, scope, model_key, degraded=True,
                       reason="the intent router did not return JSON")

    detected = parsed.get("intent")
    if detected not in VALID_INTENTS:
        return _result("ambiguous", 0.0, scope, model_key, degraded=True,
                       reason=f"the intent router returned an unknown intent "
                              f"{detected!r}")
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    intent = detected
    reason = ""
    if intent != "ambiguous" and confidence < CONFIDENCE_FLOOR:
        # Not wrong — merely not sure enough to pre-arm one button (§1d.4).
        intent = "ambiguous"
        reason = (f"the router was only {confidence:.0%} sure — both actions are "
                  "offered")

    out = _result(intent, confidence, scope, model_key, reason=reason,
                  detected=detected)
    with _cache_lock:
        _cache[key] = dict(out)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return out
