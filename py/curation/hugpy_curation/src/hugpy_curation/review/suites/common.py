"""Shared plumbing for the task grading suites.

Checkers here are deterministic string predicates over a model's reply.  The
image suites build on them; nothing in this module imports PIL or numpy so the
registry stays importable on a box without them.
"""
from __future__ import annotations

import json
import re
import time

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

# Queueing, not failure: the dispatcher decides when capacity frees.  Retried
# until the caller's lane deadline (the bounded client raises LaneTimeout).
_BUSY = ("worker_busy", "model_busy", "concurrency limit")
# Central declining to START another cold load (503 + Retry-After).  Retried a
# BOUNDED number of times, then reported as exhausted - never looped on.
_CAPACITY = ("cold_load_capacity", "ColdHoldCapacity", "cold_hold")
CAPACITY_RETRIES = 3
CAPACITY_BACKOFF_S = (5.0, 10.0, 20.0)


class LaneAbort(Exception):
    """Base for run-loop control flow that must escape a suite call."""


class LaneTimeout(LaneAbort):
    def __init__(self, phase, elapsed_s, budget_s):
        super().__init__(f"timeout after {elapsed_s:.1f}s in {phase} (budget {budget_s:.0f}s)")
        self.phase, self.elapsed_s, self.budget_s = phase, elapsed_s, budget_s


class LaneCancelled(LaneAbort):
    pass


def retrying_request(client, path, body, fleet_error, sleep=time.sleep):
    """POST ``body`` to ``path``; returns ``(response, error)``.

    Busy refusals queue (bounded by the lane deadline the client enforces);
    ``cold_load_capacity`` retries at most CAPACITY_RETRIES times with backoff,
    then returns ``cold_load_capacity exhausted``.  LaneAbort propagates.
    """
    # A bounded run client supplies a cancel/deadline-aware sleep.
    sleep = getattr(client, "sleep", None) if callable(getattr(client, "sleep", None)) else sleep
    capacity_tries = 0
    while True:
        try:
            return client.request(path, "POST", body, timeout=None), None
        except LaneAbort:
            raise
        except fleet_error as exc:
            message = str(exc)
            if any(marker in message for marker in _CAPACITY):
                if capacity_tries >= CAPACITY_RETRIES:
                    return None, (f"cold_load_capacity exhausted after {capacity_tries} retries: "
                                  f"{type(exc).__name__}: {message}")
                sleep(CAPACITY_BACKOFF_S[min(capacity_tries, len(CAPACITY_BACKOFF_S) - 1)])
                capacity_tries += 1
                continue
            if not any(marker in message for marker in _BUSY):
                return None, f"{type(exc).__name__}: {message}"
            sleep(2)
        except Exception as exc:  # transport faults are call errors, not grades
            return None, f"{type(exc).__name__}: {exc}"


# ------------------------------------------------------ failure analysis ----
MODEL_LEVEL = ("hard_load_failure", "held")          # skip the model everywhere
LANE_LEVEL = ("vram_fit", "refused")                  # skip this lane only
_UNREACHABLE = ("WorkerUnreachable", "unreachable", "URLError", "Connection refused",
                "ConnectError", "RemoteDisconnected", "ConnectionResetError", "Name or service not known")


def _find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find(value, key)
            if found is not None:
                return found
    return None


def _json_tail(text):
    start = (text or "").find("{")
    if start < 0:
        return None
    try:
        return json.loads(text[start:])
    except ValueError:
        return None


def classify(error):
    """``(class, evidence)`` for an error string from a suite call.

    class is one of hard_load_failure / held / vram_fit / refused /
    cold_load_capacity / unreachable / timeout / http_error / error; evidence
    keeps the error verbatim plus any structured fields the reply carried
    (load_failure.class, loader_stderr first line, request_id, diagnostics).
    """
    text = str(error or "")
    low = text.lower()
    payload = _json_tail(text)
    load_failure = _find(payload, "load_failure") if payload is not None else None
    lf_class = load_failure.get("class") if isinstance(load_failure, dict) else None
    evidence = {"error": text[:4000]}
    stderr = (load_failure or {}).get("loader_stderr") if isinstance(load_failure, dict) else None
    stderr = stderr or (_find(payload, "loader_stderr") if payload is not None else None)
    if stderr:
        evidence["loader_stderr"] = str(stderr)[:2000]
        evidence["loader_stderr_first_line"] = next((l for l in str(stderr).splitlines() if l.strip()), "")
    for key in ("request_id", "diagnostics"):
        value = _find(payload, key) if payload is not None else None
        if value is not None:
            evidence[key] = value
    if lf_class:
        evidence["load_failure_class"] = lf_class
    if lf_class == "hard_load_failure" or "hard_load_failure" in low or "hard load failure" in low:
        return "hard_load_failure", evidence
    if "faulty_model" in low or ("admission" in low and "held" in low) or " is held" in low:
        return "held", evidence
    if "cold_load_capacity" in low or "coldholdcapacity" in low:
        return "cold_load_capacity", evidence
    if lf_class == "vram_fit" or "vram_fit" in low or "does not fit" in low or "misconfigured" in low:
        return "vram_fit", evidence
    if lf_class == "unreachable" or any(m.lower() in low for m in _UNREACHABLE):
        return "unreachable", evidence
    if "requested worker" in low:
        return "refused", evidence
    if "timeout" in low or "timed out" in low:
        return "timeout", evidence
    if "http " in low:
        return "http_error", evidence
    return "error", evidence


def numbers(text):
    """Integers mentioned in ``text`` (digits and small number words), in order."""
    out = []
    for token in re.findall(r"-?\d+|[a-z]+", (text or "").lower()):
        if token.lstrip("-").isdigit():
            out.append(int(token))
        elif token in _NUMBER_WORDS:
            out.append(_NUMBER_WORDS[token])
    return out


def number_is(n):
    """The reply's first number (digit or word) equals ``n``."""
    def check(text):
        found = numbers(text)
        return bool(found) and found[0] == n
    check.rule = f"first number == {n}"
    check.answer = str(n)
    return check


def choice(correct, options, ignore=()):
    """Of ``options``, the one mentioned FIRST in the reply is ``correct``.

    Stops a reply that lists every option ("red, blue or yellow") from passing
    the way a plain substring test would.  ``ignore`` phrases (typically the
    question's own referent, e.g. "blue circle") are blanked out first so an
    echo of the question does not count as an answer.
    """
    options = tuple(dict.fromkeys((correct,) + tuple(options)))
    patterns = {o: re.compile(r"\b" + re.escape(o) + r"s?\b") for o in options}

    def check(text):
        low = (text or "").lower()
        for phrase in ignore:
            low = low.replace(phrase.lower(), " ")
        hits = [(m.start(), o) for o, p in patterns.items() for m in [p.search(low)] if m]
        return bool(hits) and min(hits)[1] == correct
    check.rule = f"first of {list(options)} mentioned == {correct!r}"
    check.answer = correct
    return check


def contains_normalized(phrase):
    """Letters/digits of ``phrase`` appear in order in the reply (case/space-insensitive)."""
    want = re.sub(r"[^a-z0-9]", "", phrase.lower())

    def check(text):
        return want in re.sub(r"[^a-z0-9]", "", (text or "").lower())
    check.rule = f"contains {phrase!r} (normalized)"
    check.answer = phrase
    return check


# The call-ledger fields a graded reply carries (2026-09-23): central's relay
# stamps ``timings.call`` onto every reply (/v1/chat/completions and /ml/*,
# resolvers.remote._stamp_done) with the numbers it recorded in model_calls for
# THIS call. A benchmark row copies them so it can be matched to that relay row
# (request_id — measure once) and states the serving stamp and the generation
# split instead of re-deriving them from a wall clock.
SERVED_FIELDS = ("request_id", "prompt_s", "generation_s", "gen_tokens", "gen_basis",
                 "served_worker", "served_quant", "served_alloc_mode")


def served_fields(response):
    """{request_id, prompt_s, generation_s, gen_tokens, gen_basis, served_*}
    from a reply's ``timings.call`` (absent keys omitted; {} for an older
    central). ``served_*`` is what hugpy served — the lane's own quant/alloc
    stay the lane's."""
    t = response.get("timings") if isinstance(response, dict) else None
    c = t.get("call") if isinstance(t, dict) else None
    if not isinstance(c, dict):
        return {}
    out = {k: c.get(k) for k in ("request_id", "prompt_s", "generation_s", "gen_tokens", "gen_basis")}
    out.update(served_worker=c.get("worker"), served_quant=c.get("quant"),
               served_alloc_mode=c.get("alloc_mode"))
    return {k: v for k, v in out.items() if v not in (None, "")}
