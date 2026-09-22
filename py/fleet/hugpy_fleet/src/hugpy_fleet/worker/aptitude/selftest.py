"""Worker-side aptitude SELF-TEST — scaffolded, and it ships DARK.

The studio-aptitude bench (``evaluations/studio-aptitude/``) proved the right
mechanical scorer but the WRONG delivery: an external sweep that cold-loads
models and polls central is exactly the load the operator's 2026-07-29 ruling
halted. The same 35 mechanical points are worth having as a continuous signal —
but only if getting them costs the pool nothing.

So every gate here is a refusal by default:

  * **OFF unless the operator says otherwise.** ``HUGPY_WORKER_SELFTEST=on`` is
    the only thing that makes this module do anything. Unset/anything-else =>
    ``maybe_run`` returns a reason and makes zero calls. It is an operator lever,
    per the standing doctrine that a default must be a success path — and a
    default that spends GPU time on self-measurement is not one.

  * **RESIDENT ONLY.** A candidate must already be in ``loaded_models``. This
    module never loads, never warms, never provisions, never evicts. If nothing
    is resident there is nothing to test, and that is a correct outcome.

  * **IDLE ONLY.** A model that served within the idle window is doing real
    work; the self-test yields to it. Measurement never competes with traffic.

  * **ONE case per interval.** Not a suite, not a sweep — one short prompt, at
    most once per ``HUGPY_WORKER_SELFTEST_INTERVAL_S`` (default 30 min), rotating
    through the cases so coverage accumulates over hours instead of arriving as
    a burst.

  * **MECHANICAL ONLY.** The judge half of the bench (65 of the 100 points) is
    an LLM call and is deliberately NOT ported. Nothing here asks a model for an
    opinion about another model. Only the 35 points code can decide.

  * **INJECTED CALLER.** ``maybe_run`` takes the function that talks to the
    model. This module imports no runner, no dispatch, no torch — so when the
    lever is off (the shipped state) it costs one env lookup.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Optional

from hugpy_fleet.worker.aptitude import cases as _cases
from hugpy_fleet.worker.aptitude import parse as _parse
from hugpy_fleet.worker.aptitude import score as _score

DEFAULT_INTERVAL_S = 1800.0
# A resident that answered within this window is WORKING — leave it alone.
DEFAULT_IDLE_S = 300.0


def enabled() -> bool:
    return (os.environ.get("HUGPY_WORKER_SELFTEST") or "").strip().lower() in (
        "on", "1", "true", "yes",
    )


def interval_s() -> float:
    try:
        return max(60.0, float(os.environ.get("HUGPY_WORKER_SELFTEST_INTERVAL_S") or DEFAULT_INTERVAL_S))
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_S


def idle_s() -> float:
    try:
        return max(0.0, float(os.environ.get("HUGPY_WORKER_SELFTEST_IDLE_S") or DEFAULT_IDLE_S))
    except (TypeError, ValueError):
        return DEFAULT_IDLE_S


# ── the rotation ────────────────────────────────────────────────────────────
def _units() -> list:
    """Flat rotation of (suite, case) units. Built once per call from the pure
    case data — cheap, and it keeps no global that a test would have to reset."""
    units = []
    for c in _cases.SCENE_CASES:
        units.append(("scene", c))
    for cid, text, expected in _cases.ROUTING_CASES:
        units.append(("routing", {"id": cid, "text": text, "expected": expected}))
    for c in _cases.NEGATIVE_CASES:
        units.append(("negative", c))
    return units


def _system_for(suite: str) -> str:
    return {
        "scene": _cases.SCENE_SYSTEM,
        "routing": _cases.ROUTING_SYSTEM,
        "negative": _cases.NEGATIVE_SYSTEM,
    }[suite]


def _user_for(suite: str, case: dict) -> str:
    if suite == "scene":
        import json
        return json.dumps(case["payload"], ensure_ascii=False)
    if suite == "routing":
        return case["text"]
    return case["prompt"]


def score_unit(suite: str, case: dict, resp: dict) -> dict:
    """Score one reply MECHANICALLY. Pure: no model, no network, no clock.

    Returns a compact row for the aggregate — the scored points plus just enough
    detail to explain them. The full generated text is NOT kept: the aggregate
    is a rolling health file, not a transcript store."""
    text = (resp or {}).get("text", "") or ""
    obj, how = _parse.extract_json(text)
    if suite == "scene":
        sc = _score.score_scene(case, obj, how, resp or {})
        mech = sc["mechanical"]
        return {
            "suite": "scene", "case_id": case["id"],
            "mech_points": mech["total"], "mech_max": _score.MECH_MAX,
            "parts": {k: v for k, v in mech.items() if k != "total"},
            "directions_matched": sc["detail"]["directions_matched"],
            "directions_total": sc["detail"]["directions_total"],
            "inventions": len(sc["detail"]["inventions"]),
            "words": sc["detail"]["words"],
            "json_how": how,
        }
    if suite == "routing":
        sr = _score.score_routing_one(
            case["expected"], obj, how, resp or {},
            partial_ok=case["id"] in _cases.ROUTING_PARTIAL_OK)
        return {
            "suite": "routing", "case_id": case["id"],
            # a routing case is pass/fail — normalize onto the same 35-pt scale
            # so one number in the aggregate is comparable across suites.
            "mech_points": round(sr["correct"] * _score.MECH_MAX, 2),
            "mech_max": _score.MECH_MAX,
            "expected": sr["expected"], "got": sr["got"],
            "json_ok": sr["json_ok"], "json_how": how,
        }
    sn = _score.score_negative(obj, how, resp or {})
    mech = sn["mechanical"]
    return {
        "suite": "negative", "case_id": case["id"],
        # score_negative is on a /100 scale; project onto /35 for comparability.
        "mech_points": round(mech["total"] * _score.MECH_MAX / 100.0, 2),
        "mech_max": _score.MECH_MAX,
        "parts": {k: v for k, v in mech.items() if k != "total"},
        "terms": sn["terms"], "prose_detected": sn["prose_detected"],
        "json_how": how,
    }


# ── the gate ────────────────────────────────────────────────────────────────
class SelfTestRunner:
    def __init__(self) -> None:
        self._last_run = 0.0
        self._cursor = 0

    def pick_candidate(self, loaded_models, last_served: Optional[dict] = None,
                       loading=None, now: Optional[float] = None) -> Optional[str]:
        """The first resident that is IDLE, or None.

        ``last_served`` is the aggregate's own per-model ``last_served_at`` map —
        already-known data, so deciding "idle" costs no probe. A model still
        loading is never a candidate: touching it would race the load."""
        now = float(now if now is not None else time.time())
        busy = {str(k) for k in (loading or [])}
        window = idle_s()
        for key in (loaded_models or []):
            key = str(key)
            if key in busy:
                continue
            ts = (last_served or {}).get(key)
            if ts and (now - float(ts)) < window:
                continue          # serving real traffic — yield to it
            return key
        return None

    def maybe_run(self, loaded_models, call: Callable[..., dict], *,
                  last_served: Optional[dict] = None, loading=None,
                  now: Optional[float] = None) -> dict:
        """Run at most ONE case, or explain why it didn't.

        Always returns ``{"ran": bool, "reason": str, ...}`` and never raises —
        this is called from the heartbeat tick, where an exception would cost a
        beat (and a missed beat drops the box off the fleet)."""
        now = float(now if now is not None else time.time())
        if not enabled():
            return {"ran": False, "reason": "disabled"}
        if (now - self._last_run) < interval_s():
            return {"ran": False, "reason": "interval"}
        try:
            model_key = self.pick_candidate(loaded_models, last_served, loading, now)
            if not model_key:
                return {"ran": False, "reason": "no idle resident"}
            units = _units()
            suite, case = units[self._cursor % len(units)]
            self._cursor = (self._cursor + 1) % len(units)
            # Claim the slot BEFORE the call: a slow reply must not let the next
            # beat start a second one.
            self._last_run = now
            started = time.time()
            resp = call(model_key=model_key,
                        system=_system_for(suite),
                        user=_user_for(suite, case))
            if not isinstance(resp, dict):
                resp = {"text": str(resp or "")}
            resp.setdefault("elapsed_s", round(time.time() - started, 3))
            row = score_unit(suite, case, resp)
            row["at"] = now
            row["elapsed_s"] = resp.get("elapsed_s")
            return {"ran": True, "reason": "ok", "model_key": model_key, "score": row}
        except Exception as exc:  # noqa: BLE001 — a self-test NEVER breaks a beat
            return {"ran": False, "reason": f"error: {type(exc).__name__}: {exc}"}


_RUNNER: Optional[SelfTestRunner] = None


def get_runner() -> SelfTestRunner:
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = SelfTestRunner()
    return _RUNNER


def reset_runner() -> SelfTestRunner:
    global _RUNNER
    _RUNNER = SelfTestRunner()
    return _RUNNER
