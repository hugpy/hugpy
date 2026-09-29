"""model_liveness — "is model M alive on worker W?" WITHOUT spending inference.

Operator ruling (2026-09-29, after ~216 "Reply with: OK" smokes a day landed on
the Coder-Next slot): liveness is DERIVED from the call logs and the GPU loads,
in this order, and a real generation is the last resort, never a habit.

  1. the calls record — central's ``/api/llm/calls`` (one JSON line per
     request, pending + done rows keyed by id). The most recent PROCESSED call
     for that model (any caller) inside ``window_s`` with status ``done`` means
     alive: no probe. Its ``duration_ms`` is the latency.
  2. the worker's GPU state — central's ``/api/llm/workers``: the worker is
     online with a fresh heartbeat and the model is resident (a healthy slot
     holding ``model_key``, or listed in ``loaded_models``). Resident + fresh
     heartbeat = loadable/alive without any inference. A busy slot is
     ``generating``, an idle one ``idle``. The worker's own
     ``model_call_stats[model].last_call`` inside the window counts as a
     processed call too (source ``gpu`` — it is the worker's record).
  3. only when NEITHER exists (never called in the window and not resident)
     may the caller fall back to one real generation — and only when it says so
     (``allow_probe=True``: an explicit operator action or a first use, never a
     timer), ``max_tokens`` 1, never while central's queue holds a job for that
     model. The probe is a NORMAL call (``User-Agent: hugpy-grade/1``,
     ``X-Hugpy-Client-Process: grade``) so it lands in the calls record with
     its pending/done stamps: that record IS the persisted verdict every
     consumer reads through step 1. A second sender inside the window sees the
     graded row and never re-probes.

Contract (every field always present)::

    {"alive": True|False|None, "source": "calls"|"gpu"|"probe"|"unknown",
     "last_call_ts": float|None, "latency_ms": int|None,
     "resident": bool|None, "slot_state": "idle"|"generating"|"unhealthy"|"absent"|None,
     "worker": str, "model": str, "why": str}

``alive`` is None when nothing is known (no record, not resident, no probe
allowed or probe deferred). Pure function of the ``http`` callable so tests run
without a network: ``http(method, url, body=None, headers=None, timeout=...)``
-> ``(status, json_or_text)`` — the shape of ``pkg_promote._http``.
"""
from __future__ import annotations

import os
import time
from typing import Callable

LIVENESS_WINDOW_S = float(os.environ.get("HUGPY_LIVENESS_WINDOW_S", "600"))   # >= 10 minutes
HEARTBEAT_FRESH_S = 120.0
CALLS_LIMIT = 600
PROBE_TIMEOUT_S = 120.0
PROBE_UA = "hugpy-grade/1"
PROBE_PROCESS = "grade"
# The probe's whole point is one processed call in the record; the reply text is
# irrelevant, so it asks for the smallest possible generation.
PROBE_PROMPT = "OK"
PROBE_MAX_TOKENS = 1

Http = Callable[..., tuple]


def _empty(worker: str, model: str, why: str) -> dict:
    return {"alive": None, "source": "unknown", "last_call_ts": None, "latency_ms": None,
            "resident": None, "slot_state": None, "worker": worker, "model": model, "why": why}


# ------------------------------------------------------------------ 1. calls record

def latest_processed_call(rows: list, model: str, worker: str | None, since: float) -> dict | None:
    """Newest ``done`` row for ``model`` at/after ``since``. Rows carrying a
    worker must match ``worker`` (a worker-pinned question); rows without one
    (older log lines) count for any worker. Pending rows never count: a job
    that has not been processed proves nothing."""
    best = None
    for r in rows or []:
        if not isinstance(r, dict) or r.get("model") != model or r.get("status") != "done":
            continue
        ts = r.get("ended_ts") or r.get("ts") or r.get("started_ts") or 0
        if float(ts) < since:
            continue
        if worker and r.get("worker") and r.get("worker") != worker:
            continue
        if best is None or float(ts) > float(best.get("ended_ts") or best.get("ts") or 0):
            best = r
    return best


def from_calls(http: Http, central: str, worker: str, model: str, window_s: float,
               now: float, headers: dict | None = None) -> dict | None:
    st, body = http("GET", f"{central}/api/llm/calls?limit={CALLS_LIMIT}", headers=headers)
    rows = (body or {}).get("calls") if st == 200 and isinstance(body, dict) else None
    hit = latest_processed_call(rows or [], model, worker, now - window_s)
    if not hit:
        return None
    ts = float(hit.get("ended_ts") or hit.get("ts") or hit.get("started_ts") or 0)
    lat = hit.get("duration_ms")
    return {"alive": True, "source": "calls", "last_call_ts": ts,
            "latency_ms": int(lat) if isinstance(lat, (int, float)) else None,
            "resident": None, "slot_state": None, "worker": worker, "model": model,
            "why": f"processed call {hit.get('id')} {int(now - ts)}s ago "
                   f"(ua={hit.get('ua') or '?'}, {lat or '?'} ms)"}


# ------------------------------------------------------------------ 2. GPU state

def slot_state_for(w: dict, model: str) -> tuple[bool, str]:
    """(resident, slot_state) from one worker row of ``/api/llm/workers``."""
    for s in w.get("slots") or []:
        if not isinstance(s, dict) or s.get("model_key") != model:
            continue
        if not s.get("healthy", s.get("child_pid") is not None):
            return False, "unhealthy"
        return True, "generating" if s.get("busy") else "idle"
    if model in (w.get("loaded_models") or []):
        return True, "idle"                          # resident, no slot detail (non-slot engine)
    return False, "absent"


def from_gpu(http: Http, central: str, worker: str, model: str, window_s: float, now: float,
             headers: dict | None = None) -> dict:
    st, body = http("GET", f"{central}/api/llm/workers", headers=headers)
    ws = body if isinstance(body, list) else (body or {}).get("workers") if isinstance(body, dict) else None
    if st != 200 or not isinstance(ws, list):
        return _empty(worker, model, f"worker listing unreadable (http {st})")
    w = next((x for x in ws if isinstance(x, dict) and x.get("name") == worker), None)
    if w is None:
        return _empty(worker, model, "worker not in central's listing")
    age = now - float(w.get("last_seen") or 0)
    fresh = w.get("status") == "online" and age < HEARTBEAT_FRESH_S
    resident, slot = slot_state_for(w, model)
    stats = (w.get("model_call_stats") or {}).get(model) or {}
    last_call = stats.get("last_call")
    tok_s = stats.get("tok_s_last") or stats.get("tok_s_ewma")
    out = {"alive": None, "source": "gpu", "last_call_ts": float(last_call) if last_call else None,
           "latency_ms": None, "resident": resident, "slot_state": slot, "worker": worker,
           "model": model, "why": ""}
    if not fresh:
        out.update(alive=False, why=f"worker {w.get('status')}, heartbeat {int(age)}s old")
        return out
    if last_call and now - float(last_call) < window_s:
        out.update(alive=True, why=f"worker record: processed call {int(now - float(last_call))}s ago"
                                   + (f", {tok_s} tok/s" if tok_s else ""))
        return out
    if resident:
        out.update(alive=True, why=f"resident on {worker} ({slot}), heartbeat {int(age)}s old")
        return out
    out.update(alive=False if slot == "unhealthy" else None,
               why=f"not resident on {worker} ({slot})")
    return out


# ------------------------------------------------------------------ 3. probe (last resort)

def queue_holds(http: Http, central: str, model: str, headers: dict | None = None) -> bool | None:
    """True when central's queue has an active/waiting job for ``model``;
    None when the queue cannot be read (then no probe: unknown is not idle)."""
    st, body = http("GET", f"{central}/api/llm/queue", headers=headers)
    if st != 200 or not isinstance(body, dict):
        return None
    for key in ("active", "waiting"):
        for j in body.get(key) or []:
            if isinstance(j, dict) and (j.get("model") == model or j.get("model_key") == model):
                return True
    return False


def probe(http: Http, central: str, worker: str, model: str, headers: dict | None = None,
          now: float | None = None) -> dict:
    """ONE real generation, ``max_tokens`` 1, recorded by central as a normal
    call from ``grade`` — the persisted verdict. Only reached via
    ``allow_probe=True`` and an idle queue."""
    now = time.time() if now is None else now
    hdrs = {"User-Agent": PROBE_UA, "X-Hugpy-Client-Process": PROBE_PROCESS,
            "X-Hugpy-Client-Task": f"grade:{worker}:{model}",
            "X-Hugpy-Client-Platform": "model-liveness", **(headers or {})}
    t0 = time.monotonic()
    st, body = http("POST", f"{central}/v1/chat/completions",
                    {"model": model, "max_tokens": PROBE_MAX_TOKENS, "temperature": 0,
                     "messages": [{"role": "user", "content": PROBE_PROMPT}],
                     "alloc": {"worker": worker}},
                    headers=hdrs, timeout=PROBE_TIMEOUT_S)
    ms = int((time.monotonic() - t0) * 1000)
    ok = st == 200 and isinstance(body, dict) and bool(body.get("choices"))
    out = {"alive": ok, "source": "probe", "last_call_ts": now, "latency_ms": ms,
           "resident": None, "slot_state": None, "worker": worker, "model": model,
           "why": f"graded: HTTP {st} in {ms} ms"}
    if not ok:
        out["error"] = (str(body) if not isinstance(body, dict) else str(body))[-500:]
        out["transient"] = st in (0, 429, 502, 503, 504)
    return out


# ------------------------------------------------------------------ the helper

def model_liveness(http: Http, central: str, worker: str, model: str, *, allow_probe: bool = False,
                   window_s: float | None = None, headers: dict | None = None,
                   now: float | None = None) -> dict:
    """See the module doc. ``allow_probe`` is the ONLY way a generation happens,
    and even then only when steps 1-2 know nothing and the queue is idle."""
    central = central.rstrip("/")
    window_s = LIVENESS_WINDOW_S if window_s is None else float(window_s)
    now = time.time() if now is None else now
    hit = from_calls(http, central, worker, model, window_s, now, headers)
    if hit:
        return hit
    gpu = from_gpu(http, central, worker, model, window_s, now, headers)
    if gpu["alive"] is not None:
        return gpu
    if not allow_probe:
        gpu["why"] += "; no processed call in the window; probe not allowed on this path"
        return gpu
    held = queue_holds(http, central, model, headers)
    if held is not False:
        gpu["why"] += ("; queue holds a job for this model — probe deferred" if held
                       else "; queue unreadable — probe deferred")
        return gpu
    p = probe(http, central, worker, model, headers, now)
    p.update(resident=gpu["resident"], slot_state=gpu["slot_state"])
    return p
