"""Tiered cognitive grading plus fixed-allocation fleet benchmarking."""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
import os
import logging
from urllib.parse import quote

from hugpy_curation.review.suites.common import (
    LaneAbort, LaneCancelled, LaneTimeout, classify, retrying_request)
from hugpy_platform.no_think import strip_think


_log = logging.getLogger(__name__)


class FleetError(Exception):
    pass

_DEFAULT_TIMEOUT = object()


class Client:
    """Small central-loopback JSON client; intentionally owned by HugPy."""
    def __init__(self, base, key="", operator_token="", timeout=60):
        self.base, self.key, self.operator_token, self.timeout = base.rstrip("/"), key, operator_token, timeout

    def request(self, path, method="GET", body=None, timeout=_DEFAULT_TIMEOUT):
        headers = {"Accept": "application/json"}
        if self.key: headers["Authorization"] = "Bearer " + self.key
        if self.operator_token: headers["X-Operator-Token"] = self.operator_token
        data = None if body is None else json.dumps(body).encode()
        if data is not None: headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            request_timeout = self.timeout if timeout is _DEFAULT_TIMEOUT else timeout
            with urllib.request.urlopen(request, timeout=request_timeout) as response: return json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:
                detail = ""
            suffix = (": " + detail) if detail else ""
            raise FleetError("HTTP %s at %s%s" % (exc.code, path, suffix)) from None


class BenchmarkControl:
    def __init__(self):
        self._all, self._lock = threading.Event(), threading.RLock()
        self._cancelled = {name: set() for name in ("worker", "model", "quant", "call")}
    def is_set(self): return self._all.is_set()
    def set(self): self._all.set()
    def cancel(self, scope="execution", worker_id=None, model=None, quant=None, call=None):
        if scope == "execution": self.set(); return
        key = {"worker": worker_id, "model": (worker_id, model), "quant": (worker_id, model, quant),
               "call": (worker_id, model, quant, call)}[scope]
        if not key: raise ValueError("cancellation requires an identifier")
        with self._lock: self._cancelled[scope].add(key)
    def cancelled(self, worker_id=None, model=None, quant=None, call=None):
        with self._lock:
            return self.is_set() or worker_id in self._cancelled["worker"] or (worker_id, model) in self._cancelled["model"] or (worker_id, model, quant) in self._cancelled["quant"] or (worker_id, model, quant, call) in self._cancelled["call"]


def eligible(worker):
    return worker.get("status") == "online" and not worker.get("unreachable") and worker.get("admission") == "approved" and worker.get("serve_mode") != "off"


def response_rows(payload, key):
    value = payload if isinstance(payload, list) else payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(value, list): raise FleetError("invalid %s response" % key)
    return value


# Control-plane setup fetches (the worker roster, the model catalog) the run
# cannot start or resume without. Central may be WARMING UP: a restart-resume
# runs seconds after /health first answers — before the first gunicorn worker is
# free to serve /llm/workers — and a restart's own loopback call can briefly time
# out, refuse the connection, or 502. Those are transient central states, not a
# grading failure: ride them out with bounded backoff and give up (raising) only
# after the whole window, so ONE warm-up blip can never kill a run.
CONTROL_RETRY_TOTAL_S = float(os.environ.get("HUGPY_BENCH_CONTROL_RETRY_S") or 300.0)
CONTROL_BACKOFF_S = (1.0, 2.0, 5.0, 10.0, 15.0, 30.0)
_TRANSIENT_HTTP = (500, 502, 503, 504)
_TRANSIENT_MARKERS = ("timed out", "timeout", "temporarily unavailable",
                      "connection refused", "connection reset", "remotedisconnected",
                      "bad gateway", "gateway time-out", "service unavailable",
                      "name or service not known", "warming up", "max retries")


def _transient_control_error(exc):
    """True for a control-plane fault that is central WARMING/BLIPPING (a bounded
    retry rides it out), False for a hard no (a genuine 4xx, a bad response)."""
    low = str(exc).lower()
    if isinstance(exc, FleetError):
        match = re.search(r"http (\d+)", low)
        return bool(match and int(match.group(1)) in _TRANSIENT_HTTP)
    if isinstance(exc, (TimeoutError, ConnectionError, urllib.error.URLError, OSError)):
        return True
    return any(marker in low for marker in _TRANSIENT_MARKERS)


def retrying_control_request(client, path, key, timeout=_DEFAULT_TIMEOUT,
                             total_s=None, sleep=time.sleep, log=None):
    """GET a control-plane list (``/llm/workers``, ``/models?verbose=1``) that
    the run cannot proceed without, riding out a warming/blipping central with
    bounded backoff. Returns the parsed list (``response_rows``). Raises the last
    error only after ``total_s`` of PERSISTENT transient failure (a genuinely
    down central, not a warm-up blip); a hard error (4xx, malformed reply) raises
    at once."""
    total_s = CONTROL_RETRY_TOTAL_S if total_s is None else total_s
    deadline = time.monotonic() + total_s
    attempt = 0
    while True:
        try:
            return response_rows(client.request(path, timeout=timeout), key)
        except Exception as exc:  # noqa: BLE001 — re-raised below when not transient/persistent
            if not _transient_control_error(exc) or time.monotonic() >= deadline:
                raise
            wait = CONTROL_BACKOFF_S[min(attempt, len(CONTROL_BACKOFF_S) - 1)]
            wait = min(wait, max(0.0, deadline - time.monotonic()))
            if log is not None:
                log("notice", {"phase": "control-plane", "path": path,
                               "error": f"{type(exc).__name__}: {exc}", "retry_in_s": round(wait, 1),
                               "message": f"central not ready for {path} "
                                          f"({type(exc).__name__}: {exc}); retrying in {wait:.0f}s"})
            sleep(wait)
            attempt += 1


def verbose_catalog(client, log=None):
    return retrying_control_request(client, "/models?verbose=1", "models", log=log)


def model_id(model):
    return model.get("model_key") or model.get("id") or model.get("name")


def _ints(s): return [int(x) for x in re.findall(r"-?\d+", s or "")]


def _r(rule, fn, answer=None):
    """Attach the human-readable expectation (``rule``) and the expected
    response itself (``answer``) to a checker (``describe`` / ``expected_answer``)."""
    fn.rule = rule
    fn.answer = answer
    return fn


def _last(n): return _r(f"last number == {n}", lambda s: _ints(s)[-1:] == [n], str(n))
def _has(word): return _r(f"contains {word!r} (case-insensitive)", lambda s: word.lower() in (s or "").lower(), word)
def _exact(value): return _r(f"reply == {value!r} exactly", lambda s: (s or "").strip() == value, value)


def _json_eq(expected):
    def check(value):
        match = re.search(r"\{.*\}", value or "", re.S)
        try: return bool(match) and json.loads(match.group(0)) == expected
        except ValueError: return False
    return _r(f"JSON object == {json.dumps(expected)}", check, json.dumps(expected))


def describe(checker):
    """What a suite item expects (its checker's ``rule``), or None."""
    return getattr(checker, "rule", None)


def expected_answer(checker):
    """The expected response text for a suite item, or None when the item is
    graded by a rule with no single answer (images)."""
    return getattr(checker, "answer", None)


ACTUAL_MAX = 20000   # a whole reply; a graded response is never shown as a truncated blob


def _actual(response):
    """The model's verbatim output for the history, bounded."""
    out = response.get("output") if isinstance(response, dict) else response
    if out is None and isinstance(response, dict):
        out = response.get("answer")
    if out is not None and not isinstance(out, str):
        out = json.dumps(out, default=str) if isinstance(out, (dict, list)) else repr(out)
    return out if out is None or len(out) <= ACTUAL_MAX else out[:ACTUAL_MAX] + "…[truncated]"


def _why(passed, error, expected, actual):
    if passed:
        return None
    if error:
        return f"call failed: {error}"
    if actual in (None, ""):
        return "empty reply" + (f"; expected {expected}" if expected else "")
    return (f"expected {expected}" if expected else "checker rejected the reply") + \
        f"; got {actual!r}"


TASKS_TIERED = {
    "math": (("easy", "Compute 15 + 22. Reply with only the final number.", _last(37)),
             ("medium", "Compute 17 * 23. Reply with only the final number.", _last(391)),
             ("hard", "Solve for x: 3x + 12 = 27. Reply with only the number.", _last(5))),
    "wordprob": (("easy", "John has 5 apples. He buys 3 more. How many? Reply number only.", _last(8)),
                 ("medium", "A store had 48 apples. Sold 19 in the morning, 12 in the afternoon. Left? Reply number only.", _last(17)),
                 ("hard", "Train A leaves at 60mph. 2 hours later Train B leaves at 80mph. Hours until they meet? Reply number only.", _last(6))),
    "factual": (("easy", "What planet is known as the Red Planet? Reply with only the planet name.", _has("mars")),
                ("medium", "What is the chemical symbol for gold? Reply with only the symbol.", _has("au")),
                ("hard", "Who discovered penicillin? Reply with only the last name.", _has("fleming"))),
    "format_primes": (("easy", "List the first three prime numbers separated by commas and nothing else.", _r("first 3 integers == [2, 3, 5]", lambda s: _ints(s)[:3] == [2, 3, 5], "2, 3, 5")),
                      ("medium", "List the first five prime numbers separated by commas and nothing else.", _r("first 5 integers == [2, 3, 5, 7, 11]", lambda s: _ints(s)[:5] == [2, 3, 5, 7, 11], "2, 3, 5, 7, 11")),
                      ("hard", "List the first five prime numbers in reverse order separated by commas and nothing else.", _r("first 5 integers == [11, 7, 5, 3, 2]", lambda s: _ints(s)[:5] == [11, 7, 5, 3, 2], "11, 7, 5, 3, 2"))),
    "logic": (("easy", "If all bloops are razzies, and all razzies are lazzies, are all bloops lazzies? Answer yes or no.", _has("yes")),
              ("medium", "A is taller than B. C is shorter than B. Who is the shortest? Reply only with the letter.", _r("reply == 'c' (case-insensitive)", lambda s: (s or "").strip().lower() == "c", "C")),
              ("hard", "Can a 3-gallon jug and a 5-gallon jug measure exactly 4 gallons? Answer yes or no.", _has("yes"))),
    "exact_instruction": (("easy", "Reply with exactly the single word: BANANA", _exact("BANANA")),
                          ("medium", "Reply with exactly the single word: BANANA, but in lowercase.", _exact("banana")),
                          ("hard", "Reply with exactly the string: [BANANA_123] and absolutely nothing else.", _exact("[BANANA_123]"))),
    "coding": (("easy", "Write a Python one-liner using sum() to total a list named xs. Reply with only code.", _r("code contains sum(xs)", lambda s: "sum(xs)" in re.sub(r"\s+", "", s or ""), "total = sum(xs)")),
               ("medium", "Write a Python list comprehension returning only even numbers from list xs. Reply with only code.", _r("code contains %2==0", lambda s: "%2==0" in re.sub(r"\s+", "", s or ""), "[x for x in xs if x % 2 == 0]")),
               ("hard", "Write a recursive Python lambda named fib for Fibonacci. Reply with only code.", _r("code has lambda, -1 and -2", lambda s: "lambda" in s and "-1" in s and "-2" in s, "fib = lambda n: n if n < 2 else fib(n-1) + fib(n-2)"))),
    "json": (("easy", 'Output only a JSON object with key "a" set to 1.', _json_eq({"a": 1})),
             ("medium", 'Output only a JSON object with keys "a" set to 1 and "b" set to 2.', _json_eq({"a": 1, "b": 2})),
             ("hard", 'Output only a JSON object with key "a" containing a list of 1 and 2.', _json_eq({"a": [1, 2]}))),
    "letters": (("easy", "How many times does the letter e appear in the word tree? Reply number only.", _last(2)),
                ("medium", "How many times does the letter r appear in the word strawberry? Reply number only.", _last(3)),
                ("hard", "How many times does the letter s appear in the word mississippi? Reply number only.", _last(4))),
}


def _content(response):
    try: return response["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError): return ""


def _serving(client, model):
    try: return client.request("/llm/serving/" + quote(model, safe="")) or {}
    except Exception: return {}


def _quants(model, serving, worker_id):
    choices = serving.get("available_gguf_detail") or model.get("gguf_variants_detail") or []
    out = []
    for item in choices:
        # Discovery deliberately exposes incomplete shard groups so an operator
        # can see and reclaim the litter.  They are not runnable variants.  A
        # lone ``-00002-of-00002`` must never become a benchmark lane.
        if isinstance(item, dict) and item.get("complete") is False:
            continue
        name = (item.get("file") or item.get("filename") or item.get("quant")) if isinstance(item, dict) else item
        size = (item.get("bytes") or item.get("size_bytes")) if isinstance(item, dict) else None
        if name and not any(row["quant"] == name for row in out): out.append({"quant": name, "size_bytes": size})
    if not out:
        for name in serving.get("available_gguf") or model.get("gguf_variants") or []:
            if isinstance(name, str): out.append({"quant": name, "size_bytes": None})
    fallback = (serving.get("gguf_file_by_worker") or {}).get(worker_id) or serving.get("effective_gguf") or model.get("effective_gguf") or model.get("quant")
    return out or [{"quant": fallback or "runtime default", "size_bytes": model.get("effective_bytes") or model.get("size_bytes")}]


def _cap(worker, *keys):
    return next((worker[k] for k in keys if isinstance(worker.get(k), (int, float))), None)


def _gb(n):
    return f"{n / 1e9:.1f} GB" if isinstance(n, (int, float)) else "unknown"


def _lane_path(lane):
    """Absolute path of the lane's GGUF on central's store, or None."""
    model, quant = lane["model_record"], lane.get("quant")
    dest = model.get("destination") or (model.get("extra") or {}).get("dir")
    if quant in (None, "", "runtime default"):
        quant = model.get("effective_gguf")
    if not dest or not quant or not str(quant).lower().endswith(".gguf"):
        return None
    path = os.path.join(dest, quant)
    return path if os.path.exists(path) else None


def _lane_moe_detail(lane):
    """(MoE detail, where it was read) for the lane's own quant, or (None, None).

    The catalog's ``moe`` field describes the effective GGUF only; any other quant
    is read from its own header (hugpy_engine.spill's cached reader), never
    extrapolated from a different file."""
    model, quant = lane["model_record"], lane.get("quant")
    spec = model.get("moe") if isinstance(model.get("moe"), dict) else None
    if spec and spec.get("is_moe") and quant in (None, "", "runtime default", model.get("effective_gguf")):
        return spec, f"catalog moe detail of {model.get('effective_gguf')}"
    path = _lane_path(lane)
    if not path:
        return None, None
    try:
        from hugpy_engine.spill import gguf_moe_detail
        detail = gguf_moe_detail(path) or {}
    except Exception:  # noqa: BLE001 — unreadable header: priced as dense, stated below
        return None, None
    return (detail, f"GGUF header of {os.path.basename(path)}") if detail.get("is_moe") else (None, None)


def _lane_n_cpu_moe(lane):
    """(n_cpu_moe, source) the load would carry, or (None, None).

    Precedence mirrors the wire: the worker's persisted spill for this model,
    then the per-model serve override (``/llm/serving`` resolves the owner-
    qualified key), then the split the resident seat was actually launched with."""
    joined = lane["joined"]
    for value, source in (((joined.get("alloc") or {}).get("n_cpu_moe"), "worker spill"),
                          ((joined.get("serve_override") or {}).get("n_cpu_moe"), "serve override"),
                          ((joined.get("seat") or {}).get("n_cpu_moe"), "resident seat")):
        if value in (None, ""):
            continue
        try:
            return int(value), source
        except (TypeError, ValueError):
            continue
    return None, None


def _lane_ctx(lane):
    """(ctx, source) the load is priced at: the seat's measured ctx, else the
    served ctx_size central resolved."""
    joined = lane["joined"]
    for value, source in (((joined.get("allocation") or {}).get("ctx"), "seat ctx"),
                          ((joined.get("seat") or {}).get("ctx"), "seat ctx"),
                          (joined.get("serve_ctx_size"), "served ctx_size")):
        if isinstance(value, int) and value > 0:
            return value, source
    return None, None


def _effective_need(lane, mode, extras, size):
    """What loading this lane in ``mode`` actually puts on each device.

    Returns ``(vram_need, ram_need, basis)``. Uses the engine's own pricing —
    ``spill.moe_split_need`` for an expert split (n_cpu_moe from spill/serve
    override/seat), ``spill.vision_projector_bytes`` for the mmproj,
    ``spill.vram_ctx_reserve_bytes`` for the KV/context reserve and
    ``alloc_modes.bnb_effective_bytes`` for 4-bit — so the planner and placement
    agree. ``basis`` lists every number used."""
    model = lane["model_record"]
    gguf = str(model.get("framework") or "").lower() in ("gguf", "llama_cpp")
    parts = []
    if extras.get("bnb_4bit"):
        try:
            from hugpy_engine.alloc_modes import bnb_effective_bytes, BNB_4BIT_SIZE_RATIO
            weights = bnb_effective_bytes(size)
            parts.append(f"bnb 4-bit {_gb(weights)} = {_gb(size)} x {BNB_4BIT_SIZE_RATIO}")
        except Exception:  # noqa: BLE001
            weights = size
            parts.append(f"bnb 4-bit pricing unavailable; full size {_gb(size)}")
    else:
        weights = size
    if mode == "ram_only" or not gguf:
        parts.insert(0, f"weights {_gb(weights)}")
        need = weights
        return (None if mode == "ram_only" else need), (need if mode == "ram_only" else None), "; ".join(parts)
    path = _lane_path(lane)
    mmproj = 0
    try:
        from hugpy_engine.spill import vision_projector_bytes
        mmproj = int(vision_projector_bytes(path)) if path else int(model.get("mmproj_bytes") or 0)
    except Exception:  # noqa: BLE001
        mmproj = int(model.get("mmproj_bytes") or 0)
    kv, kv_note = 0, "KV reserve not priced (no GGUF path on central)"
    ctx, ctx_src = _lane_ctx(lane)
    if path:
        try:
            from hugpy_engine.spill import vram_ctx_reserve_bytes
            kv, kv_src, kv_detail = vram_ctx_reserve_bytes(path, ctx)
            kv = int(kv)
            kv_note = (f"KV/context reserve {_gb(kv)} ({kv_src}, ctx {kv_detail.get('ctx') or ctx}"
                       f"{' from ' + ctx_src if ctx_src else ''})")
        except Exception as exc:  # noqa: BLE001
            kv, kv_note = 0, f"KV reserve not priced ({exc})"
    detail, detail_src = _lane_moe_detail(lane)
    ncm, ncm_src = _lane_n_cpu_moe(lane)
    if mode == "explicit" and ncm is None:
        try:
            from hugpy_engine.spill import MOE_ALL_LAYERS
            ncm, ncm_src = MOE_ALL_LAYERS, "explicit-mode default (all expert layers)"
        except Exception:  # noqa: BLE001
            pass
    split = None
    if detail and ncm:
        try:
            from hugpy_engine.spill import moe_split_need
            norm = dict(detail)
            norm["expert_bytes_by_layer"] = {int(k): int(v) for k, v in (detail.get("expert_bytes_by_layer") or {}).items()}
            split = moe_split_need(norm, ncm)
        except Exception:  # noqa: BLE001
            split = None
    if split:
        vram = int(split["gpu_bytes"]) + mmproj + kv
        ram = int(split["cpu_bytes"])
        parts[:0] = [f"override n_cpu_moe={ncm} ({ncm_src}; {detail_src})",
                     f"GPU weights {_gb(split['gpu_bytes'])} (non-expert + experts of layers >= {split['layers_on_cpu']})",
                     f"mmproj {_gb(mmproj)}", kv_note,
                     f"CPU experts {_gb(ram)} ({split['layers_on_cpu']} layers)"]
        return vram, ram, "; ".join(parts)
    vram = (weights or 0) + mmproj + kv if weights else None
    why_dense = ("dense model" if not detail else
                 f"n_cpu_moe={ncm} ({ncm_src}): no experts to CPU" if ncm == 0 else
                 "MoE with no n_cpu_moe in worker spill, serve override or resident seat: experts on GPU")
    parts[:0] = [f"{why_dense}", f"weights {_gb(weights)}", f"mmproj {_gb(mmproj)}", kv_note]
    return vram, None, "; ".join(parts)


def _variations(lane):
    model, joined, worker = lane["model_record"], lane["joined"], lane["worker_record"]
    size = lane.get("size_bytes") or model.get("effective_bytes") or model.get("size_bytes")
    vram, ram = _cap(worker, "max_vram_bytes", "vram_total"), _cap(worker, "max_ram_bytes", "ram_total")
    candidates = [("standard", "gpu_only", {}), ("standard", "ram_only", {})]
    if joined.get("bnb_4bit") is not None or model.get("is_4bit_capable") or model.get("bnb_capable"):
        candidates += [("4-bit", "gpu_only", {"bnb_4bit": True}), ("4-bit", "ram_only", {"bnb_4bit": True})]
    if joined.get("moe") is not None or model.get("is_moe_capable") or model.get("moe"):
        candidates.append(("standard", "explicit", {"moe": True}))
    wname = worker.get("name") or worker.get("id") or lane.get("worker")
    rows = []
    for precision, mode, extras in candidates:
        vneed, rneed, basis = _effective_need(lane, mode, extras, size) if size else (None, None, "size unknown")
        over = []
        if vneed is not None and vram is not None and vneed > vram:
            over.append(f"VRAM need {_gb(vneed)} ({vneed} bytes) exceeds {wname} VRAM capacity {_gb(vram)} ({vram} bytes)")
        if rneed is not None and ram is not None and rneed > ram:
            over.append(f"RAM need {_gb(rneed)} ({rneed} bytes) exceeds {wname} RAM capacity {_gb(ram)} ({ram} bytes)")
        runnable = not over
        checked = [f"VRAM {_gb(vneed)} of {_gb(vram)}" if vneed is not None else None,
                   f"RAM {_gb(rneed)} of {_gb(ram)}" if rneed is not None else None]
        reason = None if runnable else "; ".join(over) + f" — {mode} priced as: {basis}; checked {', '.join(c for c in checked if c)}"
        rows.append((precision, mode, extras, runnable, reason, vneed if vneed is not None else rneed))
    return rows


def _lanes(client, workers, models):
    """A lane per (model, eligible worker, quant), for EVERY selected eligible
    worker — designated or not (operator ruling 2026-09-24: testing runs on all
    SELECTED workers; the benchmark reaches a non-designated seat through the
    per-request ``alloc.worker`` pin, which serves without a designation). The
    worker's own per-model join row (bnb_4bit / moe / alloc / seat) is used when
    it has one; a worker with no row for this model gets an empty ``joined`` so
    the capacity check prices it as an autofit dense load."""
    eligible_workers = [w for w in workers if eligible(w)]
    lanes = []
    for model in models:
        mid = model_id(model); serving = _serving(client, mid)
        joins = {}
        for row in model.get("workers") or []:
            for key in (row.get("worker_id"), row.get("worker")):
                if key is not None and key not in joins:
                    joins[key] = row
        for worker in eligible_workers:
            wid, wname = worker.get("id"), worker.get("name")
            joined = joins.get(wid) or joins.get(wname) or {}
            # The per-model serve override (n_cpu_moe, n_gpu_layers) and served
            # ctx ride the lane's join row so the constraint check prices the
            # load placement will actually perform (kept out of public()).
            joined = {**joined, "serve_override": serving.get("override") or {},
                      "serve_ctx_size": serving.get("ctx_size")}
            for quant in _quants(model, serving, wid):
                lanes.append({"model": mid, "model_record": model, "joined": joined, "worker_record": worker,
                              "worker": wname or wid, "worker_id": wid, **quant})
    return lanes


def _select_quant(client, lane):
    if lane["quant"] == "runtime default": return
    path = "/llm/serving/" + quote(lane["model"], safe="")
    serving = client.request(path); pins = dict(serving.get("gguf_file_by_worker") or {})
    if pins.get(lane["worker_id"]) != lane["quant"]:
        pins[lane["worker_id"]] = lane["quant"]; client.request(path, "POST", {"gguf_file_by_worker": pins})


def _cold_reset(client, lane):
    """Leave the selected worker cold; do not provision from the grader.

    Cold preparation is exactly an in-memory evict followed by removal of the
    worker-local cached copy.  The next ordinary inference call is solely
    responsible for triggering the normal central-to-worker download and load.
    """
    base = "/llm/workers/" + quote(str(lane["worker_id"]), safe="")
    # A benchmark reset is an explicit destructive reset of this one seat.  It
    # must not be vetoed by residency policy; the assignment itself survives.
    body = {"model_key": lane["model"], "force": True}
    evicted = client.request(base + "/evict", "POST", body)
    if isinstance(evicted, dict) and evicted.get("ok") is False:
        raise RuntimeError(evicted.get("reason") or "worker eviction failed")
    removed = client.request(base + "/cache-evict", "POST",
                             {"model_key": lane["model"]})
    if isinstance(removed, dict) and removed.get("ok") is False:
        raise RuntimeError(removed.get("reason") or "worker cache eviction failed")
    return {"evict": evicted, "cache_evict": removed}


def _body(lane, variation, prompt, tokens):
    _precision, mode, extras = variation[:3]
    return {"model": lane["model"], "messages": [{"role": "user", "content": prompt + " /no_think"}],
            "max_tokens": tokens, "max_chunks": 1, "temperature": 0,
            "alloc": {"worker": lane["worker"], "alloc_mode": mode, "benchmark": True, **extras}}


def _call(client, lane, variation, prompt, tokens):
    started = time.time()
    # Inference is a queued operation with no fixed read deadline here: busy /
    # capacity refusals queue (retrying_request), and the run loop's bounded
    # client enforces the lane's wall-clock budget instead.
    response, error = retrying_request(client, "/v1/chat/completions",
                                       _body(lane, variation, prompt, tokens), FleetError)
    answer = _content(response) if response is not None else ""
    elapsed = time.time() - started; usage = response.get("usage") if isinstance(response, dict) else None
    generated = usage.get("completion_tokens") if isinstance(usage, dict) else None
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    # Older workers/relays answer correctly but omit OpenAI usage/timing fields.
    # Keep telemetry useful by estimating tokens from the text in that case;
    # native usage remains preferred whenever it is present.
    if not isinstance(generated, (int, float)) or generated <= 0:
        generated = max(1, round(len(answer) / 4)) if answer else None
    if not isinstance(prompt_tokens, (int, float)) or prompt_tokens <= 0:
        prompt_tokens = max(1, round(len(prompt) / 4))
    speed = generated / elapsed if isinstance(generated, (int, float)) and generated > 0 and elapsed > 0 else None
    return answer, error, round(elapsed, 4), speed, prompt_tokens, generated


def _hot_judge(workers, catalog):
    """A vision model ALREADY resident on an eligible worker, or None.

    Used only for the optional imagegen judge; never selects anything that
    would need a cold load.
    """
    vision = {}
    for row in catalog:
        tasks = [row.get("primary_task")] + list(row.get("tasks") or [])
        if "image-text-to-text" not in tasks: continue
        if row.get("framework") == "gguf" and not row.get("mmproj_bytes"): continue
        vision[model_id(row)] = row
    for worker in workers:
        if not eligible(worker): continue
        for loaded in worker.get("loaded_models") or []:
            if loaded in vision:
                return {"model": loaded, "worker": worker.get("name") or worker.get("id")}
    return None


JUDGE_TOKENS = 512
JUDGE_SYSTEM = (
    "You grade ONE reply from a language model under test. Give two INDEPENDENT verdicts. "
    "correct: the reply contains the right answer to the task in substance; ignore wording, extra text, "
    "chain-of-thought, or formatting. A reply that never states the right answer is not correct. "
    "format_ok: the reply obeys the task's output constraints (e.g. 'reply with only the number', "
    "'nothing else', an exact string, a bare JSON object). A correct answer buried in extra text is "
    "correct=true, format_ok=false. Use the automatic check as evidence, not as the answer. "
    "Return ONLY a JSON object: {\"correct\": true|false, \"format_ok\": true|false, \"reason\": \"<25 words\"}"
)


JUDGE_LADDER_EXHAUSTED = "no judge: brain ladder exhausted (model under test is the brain)"


def _brain_ladder():
    """The agent-brain ladder agents resolve their default brain from: the
    ``HUGPY_AGENT_BRAINS`` csv in order, else the single canonical
    ``DEFAULT_AGENT_BRAIN`` (coder-next). Same env and default the sentinel's
    case runs and every hugpy agent inherit."""
    from hugpy_platform.constants import DEFAULT_AGENT_BRAIN
    ladder = [b.strip() for b in (os.environ.get("HUGPY_AGENT_BRAINS") or "").split(",") if b.strip()]
    return ladder or [DEFAULT_AGENT_BRAIN]


def _brain_tail(key):
    """Bare tail of a model key: catalog ``Org~Name`` and bare-name forms name
    the same model (mirrors hugpy_agent.gateway.brain_matches_key)."""
    return (key or "").strip().split("~")[-1]


def _brain_judge(exclude=None):
    """The judge = the hugpy agent DEFAULT BRAIN, resolved exactly as agents do:
    the first brain-ladder entry that is not ``exclude`` (the model under test).
    Named BY KEY so central orchestrates placement/loading — never a worker pin,
    never a residency requirement. When the model under test IS the ladder entry
    the walk advances to the next; an exhausted ladder means there is no judge
    (``{"exhausted": <reason>}``)."""
    tail = _brain_tail(exclude)
    for brain in _brain_ladder():
        if _brain_tail(brain) == tail: continue
        return {"model": brain}
    return {"exhausted": JUDGE_LADDER_EXHAUSTED}


def _balanced_json_objects(text):
    """Yield each top-level balanced ``{...}`` span in ``text`` (brace-aware and
    string-aware), so a verdict survives surrounding prose, code fences and
    braces that appear inside JSON string values."""
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]; start = None


def _judge_parse(text):
    # Judge models are thinking models: strip <think>...</think> first (they may
    # ignore /no_think), then take the FIRST balanced JSON object that actually
    # carries a verdict, tolerating prose and code fences around it.
    prose, _reasoning = strip_think(text or "")
    def flag(v):
        if isinstance(v, bool): return v
        if isinstance(v, str): return v.strip().lower() in ("true", "yes", "1")
        return None
    for span in _balanced_json_objects(prose):
        try: data = json.loads(span)
        except ValueError: continue
        if not isinstance(data, dict) or "correct" not in data: continue
        correct = flag(data.get("correct"))
        if correct is None: continue
        fmt = flag(data.get("format_ok"))
        return {"correct": correct, "format_ok": fmt if fmt is not None else correct,
                "reason": str(data.get("reason") or "")}
    return None


def judge_reply(client, target, prompt, expected, actual, check_pass, tokens=JUDGE_TOKENS):
    """Ask the agent DEFAULT BRAIN (``target`` = {model}) whether ``actual``
    answers ``prompt`` correctly and whether it obeys the format.  The brain is
    named BY KEY and routed through central's ordinary serving path — central
    orchestrates placement and any cold load; the grader never picks a worker.
    Returns {model, correct, format_ok, reason, answer, error}; ``correct`` is
    None when the judge failed or answered unparseably."""
    if isinstance(prompt, dict): prompt = prompt.get("text") or prompt.get("prompt") or json.dumps(prompt, default=str)
    user = (f"TASK PROMPT:\n{prompt}\n\nEXPECTED (automatic check rule): {expected or 'n/a'}\n"
            f"AUTOMATIC CHECK RESULT: {'PASS' if check_pass else 'FAIL'}\n\nMODEL REPLY (verbatim):\n"
            f"{(actual or '')[:ACTUAL_MAX]}\n\nJSON verdict only.")
    body = {"model": target["model"], "max_tokens": tokens, "max_chunks": 1, "temperature": 0,
            "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user", "content": user + " /no_think"}],
            "alloc": {"benchmark": True}}
    started = time.time()
    response, error = retrying_request(client, "/v1/chat/completions", body, FleetError)
    answer = _content(response) if response is not None else ""
    parsed = _judge_parse(answer) if not error else None
    out = {"model": target["model"], "answer": answer[:600],
           "raw": answer, "elapsed_s": round(time.time() - started, 4), "error": error or None,
           "correct": None, "format_ok": None, "reason": None}
    if parsed: out.update(parsed)
    elif not error: out["error"] = "judge reply unparseable (no JSON verdict)"
    return out


# ------------------------------------------------------- phase 2: judging ----
# Operator ruling 2026-09-24: grading is two phases. PHASE 1 collects and
# persists every raw output; PHASE 2 loads the judge ONCE, after the whole lot,
# and grades the stored outputs. Bounded in-loop retry rides out a judge that
# cannot yet be served (a serving refusal), then leaves the item unjudged WITH
# the real reason for a later judge-now / restart-resume to retry — never a
# silent substitution.
JUDGE_RETRY_ROUNDS = int(os.environ.get("HUGPY_BENCH_JUDGE_ROUNDS") or 3)
JUDGE_RETRY_BACKOFF_S = float(os.environ.get("HUGPY_BENCH_JUDGE_BACKOFF_S") or 30.0)


def _judge_pending(results):
    """Every collected item still awaiting a brain-judge verdict
    (``item['judge']['status'] == 'pending'``), as ``(row, item)`` pairs. The
    resident-VL image-judge categories (``judge_*``) are graded in phase 1 and
    never appear here."""
    out = []
    for row in results or ():
        if not isinstance(row, dict) or row.get("status") != "complete":
            continue
        detail = row.get("detail")
        if not isinstance(detail, dict):
            continue
        for name, entry in detail.items():
            if str(name).startswith("judge_") or not isinstance(entry, dict):
                continue
            for item in entry.get("history") or ():
                if isinstance(item, dict) and isinstance(item.get("judge"), dict) \
                        and item["judge"].get("status") == "pending":
                    out.append((row, item))
    return out


def _grade_from_detail(detail):
    """Recompute ``(score, format_score, revised_n)`` from a result row's
    ``detail`` after phase-2 judging revised its per-item verdicts. ``max`` is
    fixed at collection (it counts the graded items plus any resident-VL judge
    categories); only the achieved score moves, and the ``judge_*`` categories
    keep their phase-1 tiers untouched."""
    score = 0
    for name, entry in detail.items():
        if not isinstance(entry, dict):
            continue
        if str(name).startswith("judge_"):
            score += entry.get("tier", 0) or 0
            continue
        if "history" not in entry:
            continue
        entry["tier"] = sum(1 for i in entry["history"] if i.get("pass"))
        entry["format"] = sum(1 for i in entry["history"] if i.get("format") is True)
        entry["revised"] = sum(1 for i in entry["history"] if i.get("revised"))
        score += entry["tier"]
    graded = [v for k, v in detail.items() if isinstance(v, dict) and not str(k).startswith("judge_")]
    format_score = sum(v.get("format") or 0 for v in graded)
    revised_n = sum(v.get("revised") or 0 for v in graded)
    return score, format_score, revised_n


def _apply_judged_row(row):
    """Refresh a collected result row's score/grade/format and judge summary from
    its (now judge-revised) ``detail``. Idempotent."""
    detail = row.get("detail") or {}
    score, format_score, revised_n = _grade_from_detail(detail)
    row["score"] = score
    if row.get("max") is not None:
        row["grade"] = f"{score}/{row['max']}"
    row["format_score"] = format_score
    row["revised"] = revised_n
    if row.get("format_max"):
        row["format_grade"] = f"{format_score}/{row['format_max']}"
    pending = models = 0
    brain = None
    for name, entry in detail.items():
        if str(name).startswith("judge_") or not isinstance(entry, dict):
            continue
        for item in entry.get("history") or ():
            j = item.get("judge") if isinstance(item, dict) else None
            if not isinstance(j, dict):
                continue
            if j.get("model"):
                brain = brain or j.get("model")
            if j.get("status") == "pending":
                pending += 1
    summary = row.get("judge") if isinstance(row.get("judge"), dict) else {}
    summary.update({"status": "pending" if pending else "judged",
                    "revised": revised_n, "pending": pending})
    if brain:
        summary["model"] = brain
    row["judge"] = summary
    return row


def judge_collected(client, results, report, tokens=JUDGE_TOKENS, stop=None,
                    rounds=None, sleep=time.sleep, backoff_s=None):
    """PHASE 2. Grade every collected output still awaiting the brain judge.

    Scheduled after the whole collection lot is done (review_routes) so the judge
    — the agent DEFAULT BRAIN, resolved BY KEY per row (never the model under
    test, never pinned, placed by central) — never contends with the make-room
    evictions of the models under test. Resumable and idempotent: an item already
    judged is skipped; an item whose judge cannot be served is left ``pending``
    WITH the real refusal reason and retried for ``rounds`` bounded passes
    (``backoff_s`` between), then left unjudged for a later judge-now /
    restart-resume. A row whose judge ladder is exhausted (the model under test
    IS the brain) is recorded ``unavailable`` — terminal, no judge exists. After
    a row's items change, its score/grade is recomputed and re-reported
    (``judge-result``) so the DB grade updates. Never substitutes a different
    judge. Returns a summary dict."""
    rounds = JUDGE_RETRY_ROUNDS if rounds is None else rounds
    backoff_s = JUDGE_RETRY_BACKOFF_S if backoff_s is None else backoff_s
    rows = [r for r in (results or ()) if isinstance(r, dict) and r.get("status") == "complete"
            and isinstance(r.get("detail"), dict)]
    total = len(_judge_pending(rows))
    revised_total = refused = 0

    def cancelled():
        return stop is not None and stop.is_set()

    def progress():
        pend = len(_judge_pending(rows))
        report("judge-progress", {"phase": "judging", "total": total,
                                  "judged": total - pend, "pending": pend,
                                  "revised": revised_total, "refused": refused})

    progress()
    for attempt in range(max(1, rounds)):
        if cancelled():
            break
        pending_pairs = _judge_pending(rows)
        if not pending_pairs:
            break
        if attempt:
            sleep(backoff_s)
            if cancelled():
                break
        by_row = {}
        for row, item in pending_pairs:
            by_row.setdefault(id(row), (row, []))[1].append(item)
        for row, items in by_row.values():
            if cancelled():
                break
            brain = _brain_judge(exclude=row.get("model"))
            changed = False
            for item in items:
                if cancelled():
                    break
                if (item.get("judge") or {}).get("status") != "pending":
                    # Already judged this pass — the alloc variations of one model
                    # share a single graded ``detail`` object, so a shared item is
                    # judged once; this row still needs its own DB grade refreshed.
                    changed = True
                    continue
                if not brain.get("model"):
                    item["judge"] = {"model": None, "correct": None, "format_ok": None,
                                     "reason": None, "error": brain["exhausted"],
                                     "status": "unavailable"}
                    changed = True
                    continue
                verdict = judge_reply(client, brain, item.get("prompt"), item.get("expected"),
                                      item.get("actual"), item.get("check_pass"), tokens)
                if verdict["correct"] is not None:
                    item["pass"] = bool(verdict["correct"])
                    item["format"] = bool(verdict["format_ok"])
                    item["revised"] = item["pass"] != bool(item.get("check_pass"))
                    kept = {k: verdict[k] for k in
                            ("model", "correct", "format_ok", "reason", "error", "raw") if k in verdict}
                    kept["status"] = "judged"
                    item["judge"] = kept
                    if not item["pass"] and verdict.get("reason"):
                        item["why"] = f"judge: {verdict['reason']}"
                    elif item["revised"]:
                        item["why"] = f"judge revised the check: {verdict.get('reason') or 'correct in substance'}"
                    elif item["pass"]:
                        item.pop("why", None)
                    if item["revised"]:
                        revised_total += 1
                    changed = True
                else:
                    # A serving refusal / timeout / unparseable reply is NOT a
                    # verdict: leave the item pending WITH the real reason so it is
                    # retried this run's remaining rounds and by judge-now / resume.
                    prior = item.get("judge") if isinstance(item.get("judge"), dict) else {}
                    item["judge"] = {"model": brain["model"], "correct": None, "format_ok": None,
                                     "reason": None, "error": verdict.get("error"),
                                     "status": "pending",
                                     "attempts": int(prior.get("attempts") or 0) + 1}
                    refused += 1
                    changed = True
            if changed:
                _apply_judged_row(row)
                report("judge-result", row)
                progress()
    pending_pairs = _judge_pending(rows)
    summary = {"phase": "judging", "total": total, "judged": total - len(pending_pairs),
               "pending": len(pending_pairs), "revised": revised_total, "refused": refused,
               "unjudged": [{"model": r.get("model"), "worker": r.get("worker"), "quant": r.get("quant"),
                             "reason": (i.get("judge") or {}).get("error")} for r, i in pending_pairs]}
    report("judge-summary", summary)
    if pending_pairs:
        report("notice", f"HugPy-native judging finished with {len(pending_pairs)} item(s) unjudged "
                         f"(judge not served); retry with judge-now or a restart-resume")
    else:
        report("notice", "HugPy-native judging complete")
    return summary


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float)) and v > 0]
    return sum(values) / len(values) if values else None


# ----------------------------------------------------------------- budgets ----
# Every lane is wall-clock bounded so one bad model can never stall a run.
# Precedence: DEFAULT_BUDGETS < HUGPY_BENCH_* env < run_capacity_benchmark(budgets=...)
# (the /llm/benchmark/run body's "budgets" object).
DEFAULT_BUDGETS = {
    "cold_load_cap_s": 300.0,   # seat call (cold load) cap when no measured load rate
    "cold_load_min_s": 60.0,    # floor for a rate-derived cold-load budget
    "cold_load_max_s": 1800.0,  # sanity cap for a rate-derived budget (large model, known link)
    "call_s": 120.0,            # any single graded / probe call
    "model_s": 600.0,           # everything for one model (all workers x variations) — a floor:
                                # never below the loop's own computed need (model_budget)
    "model_max_s": 3600.0,      # cap on that computed per-model budget
    "resume_hours": 24.0,       # resume=True skips lanes graded within this window
}
BUDGET_ENV = {
    "cold_load_cap_s": "HUGPY_BENCH_COLD_LOAD_S", "cold_load_min_s": "HUGPY_BENCH_COLD_LOAD_MIN_S",
    "cold_load_max_s": "HUGPY_BENCH_COLD_LOAD_MAX_S",
    "call_s": "HUGPY_BENCH_CALL_S", "model_s": "HUGPY_BENCH_MODEL_S",
    "model_max_s": "HUGPY_BENCH_MODEL_MAX_S",
    "resume_hours": "HUGPY_BENCH_RESUME_HOURS",
}
UNREACHABLE_STRIKES = 2


def bench_budgets(overrides=None):
    out = dict(DEFAULT_BUDGETS)
    for key, env in BUDGET_ENV.items():
        try:
            if os.environ.get(env): out[key] = float(os.environ[env])
        except ValueError:
            pass
    for key, value in (overrides or {}).items():
        if key in out and isinstance(value, (int, float)) and value > 0: out[key] = float(value)
    return out


def _cold_budget(lane, budgets, recorded_rate=None):
    """Rate-derived cold-load budget (2x bytes / measured rate + 30s) when the
    worker reports a load rate (heartbeat ``load_bytes_per_s``) or one is
    RECORDED for it (``cold_store.worker_rate``: bytes/s of its last measured
    central->worker transfer), else the fixed cap. ``cold_load_cap_s`` applies
    ONLY when no rate is known; a known rate gets 2x bytes/rate + 30 s (floor
    ``cold_load_min_s``) under the generous ``cold_load_max_s`` sanity cap, so a
    legitimately large model on a known link never times out for being large."""
    worker = lane.get("worker_record") or {}
    rate = _cap(worker, "load_bytes_per_s", "measured_load_bps")
    if not rate and isinstance(recorded_rate, (int, float)) and recorded_rate > 0:
        rate = recorded_rate
    size = lane.get("size_bytes") or (lane.get("model_record") or {}).get("effective_bytes")
    if rate and size:
        return max(budgets["cold_load_min_s"], min(budgets["cold_load_max_s"], 2.0 * size / rate + 30.0))
    return budgets["cold_load_cap_s"]


def model_budget(budgets, cold_budget, n_items):
    """Effective per-model budget: ``model_s`` never undercuts what the loop
    itself computed (cold budget + one ``call_s`` per graded item + 30 s),
    capped by ``model_max_s``."""
    need = cold_budget + n_items * budgets["call_s"] + 30.0
    return max(budgets["model_s"], min(budgets["model_max_s"], need))


class _BoundedClient:
    """Wraps the central client for ONE lane: every request is wall-clock
    capped (phase cap and the model deadline) and abandoned on cancel.

    The HTTP call runs on a daemon thread with its own socket timeout; the
    caller polls cancel/deadline every 0.2s, so neither a hung socket nor a
    queue wait can hold the run past the budget.
    """
    tick = 0.2

    def __init__(self, client, stop, cancel_key, deadline):
        self.client, self.stop, self.cancel_key, self.deadline = client, stop, cancel_key, deadline
        self.phase, self.cap = "call", DEFAULT_BUDGETS["call_s"]

    def set_phase(self, phase, cap):
        self.phase, self.cap = phase, cap

    def cancelled(self):
        if self.stop.is_set(): return True
        check = getattr(self.stop, "cancelled", None)
        return bool(check and check(**self.cancel_key))

    def _budget(self, started):
        return min(self.cap, self.deadline - started)

    def sleep(self, seconds):
        started = time.monotonic(); budget = self._budget(started)
        while time.monotonic() - started < seconds:
            if self.cancelled(): raise LaneCancelled("cancelled while waiting")
            if time.monotonic() - started >= budget: raise LaneTimeout(self.phase, time.monotonic() - started, budget)
            time.sleep(min(self.tick, seconds))

    def request(self, path, method="GET", body=None, timeout=None):
        started = time.monotonic(); budget = self._budget(started)
        if self.cancelled(): raise LaneCancelled("cancelled")
        if budget <= 0: raise LaneTimeout(self.phase + " (model budget exhausted)", 0.0, 0.0)
        box = {}

        def run():
            try: box["value"] = self.client.request(path, method, body, timeout=budget + 5)
            except BaseException as exc: box["error"] = exc  # noqa: BLE001 - re-raised below
        worker = threading.Thread(target=run, name="benchmark-call", daemon=True); worker.start()
        while True:
            worker.join(self.tick)
            if not worker.is_alive(): break
            if self.cancelled(): raise LaneCancelled("cancelled during " + self.phase)
            elapsed = time.monotonic() - started
            if elapsed >= budget: raise LaneTimeout(self.phase, elapsed, budget)
        if "error" in box: raise box["error"]
        return box["value"]


def _alloc_key(config, mode):
    return mode if config in ("full", "standard") else f"{config}:{mode}"


def result_key(row):
    """Identity of one benchmark row inside a run: (model, quant, alloc, worker).
    A resumed run skips every key already recorded before the restart."""
    return (row.get("model"), row.get("quant") or "",
            _alloc_key(row.get("config") or "standard", row.get("alloc_mode") or ""), row.get("worker"))


# ------------------------------------------------------------ cold loads ----
# Operator 2026-09-23: the cold reset exists ONLY to record, once, the time to
# get a model from central onto a worker and loaded. Once that triple
# (model, worker, quant) has a recorded cold load, a run reuses it and goes
# straight to hot-load + graded calls; ``force_cold`` re-measures.
COLD_SPLIT_KEYS = ("transfer_s", "transfer_bytes", "load_s", "bytes_per_s")


MAX_PLAUSIBLE_BPS = 5e9      # faster than any disk->VRAM path in this fleet


def _recorded_cold(cold_store, model, quant, worker, size_bytes=None):
    """The MEASURED cold load for (model, quant, worker), or None (never raises).
    Only a value measured by the benchmark's cold path counts (``measured_at``
    = the row's ``cold_measured_at``); legacy EMA ``cold_load_s`` values are
    not measurements. A value faster than ``MAX_PLAUSIBLE_BPS`` for the model's
    bytes is discarded as bogus (logged) — the lane re-measures."""
    if cold_store is None:
        return None
    try:
        rec = cold_store.get(model, quant or "", worker)
    except Exception:
        return None
    cold = rec.get("cold_load_s") if isinstance(rec, dict) else None
    if not (isinstance(cold, (int, float)) and cold > 0 and rec.get("measured_at")):
        return None
    size = rec.get("transfer_bytes") or size_bytes
    if isinstance(size, (int, float)) and size > 0 and cold < size / MAX_PLAUSIBLE_BPS:
        _log.warning("benchmark: recorded cold load %.2fs for %s/%s on %s is implausible for %d bytes "
                     "(< bytes / %.0e B/s) — discarded, re-measuring", cold, model, quant, worker,
                     int(size), MAX_PLAUSIBLE_BPS)
        return None
    return rec


def _worker_rate(cold_store, worker):
    if cold_store is None or not hasattr(cold_store, "worker_rate"):
        return None
    try:
        rate = cold_store.worker_rate(worker)
    except Exception:
        return None
    return rate if isinstance(rate, (int, float)) and rate > 0 else None


def _cold_label(rec):
    at = rec.get("measured_at")
    day = time.strftime("%Y-%m-%d", time.localtime(at)) if isinstance(at, (int, float)) else "date unknown"
    return f"recorded {rec['cold_load_s']:.0f} s ({day})"


def _ledger_transfer(client, lane, since):
    """The central->worker transfer of this seat from central's own transfer
    ledger (``GET /llm/transfers``): the newest COMPLETE entry for this model on
    this worker that started after the seat began. ``{}`` when none."""
    from urllib.parse import urlencode
    q = urlencode({"model": lane.get("model") or "", "worker": lane.get("worker_id") or lane.get("worker") or ""})
    try:
        body = client.request("/llm/transfers?" + q, timeout=10) or {}
    except Exception:
        return {}
    rows = body.get("transfers") if isinstance(body, dict) else body if isinstance(body, list) else None
    tail = str(lane.get("model") or "").split("~")[-1]
    names = {lane.get("worker_id"), lane.get("worker")} - {None}
    hits = [t for t in rows or [] if isinstance(t, dict)
            and str(t.get("model_key") or "").split("~")[-1] == tail
            and (t.get("worker") in names or t.get("worker_id") in names)
            and t.get("status") == "complete" and t.get("finished_at")
            and float(t.get("started_at") or 0) >= since - 1.0]
    if not hits:
        return {}
    t = max(hits, key=lambda t: float(t.get("started_at") or 0))
    secs = float(t["finished_at"]) - float(t["started_at"])
    nbytes = t.get("bytes_served") or t.get("total_bytes")
    out = {"transfer_source": "ledger"}
    if secs > 0:
        out["transfer_s"] = round(secs, 3)
    if isinstance(nbytes, (int, float)) and nbytes > 0:
        out["transfer_bytes"] = int(nbytes)
    return out


def _cold_split(client, lane, since):
    """transfer/load split of a measured cold seat. The transfer comes from
    central's transfer ledger first (``transfer_source: ledger``), else from the
    eviction telemetry's ``provision.done`` (``transfer_source: events``); the
    load from ``load.done``. Events of the lane's worker win; ``{}`` when
    neither source has anything (older worker, telemetry off)."""
    ledger = _ledger_transfer(client, lane, since)
    try:
        body = client.request("/llm/evictions?limit=500&since=%.3f" % (since - 1.0), timeout=10) or {}
    except Exception:
        body = {}
    events = body.get("events") if isinstance(body, dict) else body if isinstance(body, list) else None
    tail = str(lane.get("model") or "").split("~")[-1]
    mine = [e for e in events or [] if isinstance(e, dict)
            and str(e.get("model_key") or "").split("~")[-1] == tail
            and e.get("stage") in ("provision.done", "load.done")]
    ours = [e for e in mine if e.get("worker_id") in {lane.get("worker_id"), lane.get("worker")}]
    out = dict(ledger)
    for ev in ours or mine:
        ms = ev.get("duration_ms")
        if not isinstance(ms, (int, float)) or ms <= 0:
            continue
        if ev["stage"] == "provision.done":
            if ledger.get("transfer_s"):
                continue
            out["transfer_source"] = "events"
            out["transfer_s"] = round(ms / 1000.0, 3)
            if isinstance(ev.get("bytes"), (int, float)) and ev["bytes"] > 0:
                out["transfer_bytes"] = int(ev["bytes"])
        else:
            out["load_s"] = round(ms / 1000.0, 3)
    if out.get("transfer_bytes") and out.get("transfer_s"):
        out["bytes_per_s"] = round(out["transfer_bytes"] / out["transfer_s"], 1)
    return out


def _recent_grades(client, hours):
    """{(model, quant, alloc, worker): graded_at} graded within ``hours`` (best effort)."""
    try:
        rows = client.request("/llm/model-metrics2?limit=5000", timeout=30) or {}
    except Exception:
        return {}
    cutoff, out = time.time() - hours * 3600, {}
    for row in rows.get("rows") or [] if isinstance(rows, dict) else []:
        at = row.get("graded_at")
        if row.get("grade") is not None and isinstance(at, (int, float)) and at >= cutoff:
            out[(row.get("model_name"), row.get("quant") or "", row.get("alloc_mode") or "", row.get("worker"))] = at
    return out


def _last_load_report(client, worker_id, model):
    if not worker_id: return None
    try:
        for worker in response_rows(client.request("/llm/workers", timeout=10), "workers"):
            if worker.get("id") == worker_id or worker.get("name") == worker_id:
                return (worker.get("load_reports") or {}).get(model)
    except Exception:
        return None
    return None


def run_capacity_benchmark(client, workers, tokens, stop, report, model_ids=None, worker_ids=None,
                           suite=None, with_judge=False, budgets=None, resume=False, force=False,
                           done=None, cold_store=None, force_cold=False, defer_judge=False):
    """Grade once per precision and benchmark every physically valid allocation.

    Each model is graded by the suite registered for its task
    (``suites.suite_for_model``); ``suite`` (a suite name) forces one suite for
    every selected model.  Models whose task has no suite are skipped with a
    recorded reason rather than graded by the text suite.

    Bounded: every lane runs under ``bench_budgets(budgets)`` (seat/cold-load
    cap, per-call cap, per-model cap); a lane that does not produce a grade
    records a row with ``failure_class`` (timeout / hard_load_failure / held /
    vram_fit / refused / unreachable / cold_load_capacity / http_error / error /
    no_suite / no_lane / cancelled) and ``evidence``, then the run moves on.  Hard load
    failures and held models skip the model everywhere; a worker failing
    UNREACHABLE_STRIKES times in a row is skipped for the rest of the run.
    ``resume`` skips lanes graded within ``resume_hours`` unless ``force``.
    ``done`` (``result_key`` tuples) are rows THIS run already recorded before a
    central restart: skipped silently (counted as completed, no new row).
    ``cold_store`` (``get(model, quant, worker)`` / ``worker_rate(worker)``)
    holds recorded cold loads: a pinned lane with one skips the cold reset and
    reuses it (``cold_source: recorded <date>``); without one (or with
    ``force_cold``) the lane resets, measures and its rows carry
    ``cold_source: measured`` + the transfer/load split for the caller to record.
    Order: already-hot models first, then smallest first.

    ``defer_judge`` (operator ruling 2026-09-24, the two-phase split): PHASE 1
    (collect). When true, the deterministic ``check_pass`` is still computed for
    every item and its raw output/timings persisted as the run goes, but the
    brain (agent-default) text judge is NOT called inline — each graded item is
    left ``judge.status == 'pending'`` and PHASE 2 (``judge_collected``) grades
    the whole lot afterwards, so a judge load never contends with the make-room
    evictions of the models under test. False = the historical inline behaviour.
    The optional resident-VL image judge (``with_judge``) is unaffected: it only
    uses an already-hot model and so never causes that contention.
    """
    from .suites import suite_by_name, suite_for_model
    b = bench_budgets(budgets)
    forced = suite_by_name(suite) if suite else None
    full_catalog = catalog = verbose_catalog(client, log=report)
    if model_ids: catalog = [m for m in catalog if model_id(m) in set(model_ids)]
    if worker_ids:
        wanted = set(worker_ids); workers = [w for w in workers if w.get("id") in wanted or w.get("name") in wanted]
    hot = {m for w in workers if eligible(w) for m in (w.get("loaded_models") or [])}
    recent = _recent_grades(client, b["resume_hours"]) if resume and not force else {}
    done = set(done or ())

    def failure_row(base, cls, reason, evidence=None, **extra):
        return {**base, "matrix": True, "ok": False, "status": "skipped" if cls in ("no_suite", "resume") else "failed",
                "failure_class": cls, "grade": None, "reason": reason, "evidence": evidence or {},
                "error": reason, "tok_s": "N/A", "detail": {}, "finished": time.time(), **extra}

    def record_failure(base, cls, reason, evidence=None, **extra):
        row = failure_row(base, cls, reason, evidence, **extra)
        results.append(row); report("result", row)
        if cls not in ("resume",):
            # The immutable call ledger is what per-model failure views read.
            report("call", {**base, "task": f"benchmark:{cls}", "timestamp": time.time(),
                            "elapsed_s": extra.get("elapsed_s"), "tok_s": "N/A", "tok_s_avg": "N/A",
                            "ctx_in": None, "ctx_out": None, "caller": "orchestrator", "output": None,
                            "grade": "ERROR", "passed": False, "error": reason,
                            "failure_class": cls, "reason": reason})
        return row

    results = []
    suite_of, gradable = {}, []
    for model in catalog:
        chosen = forced or suite_for_model(model)
        if chosen is None:
            task = model.get("primary_task") or ",".join(model.get("tasks") or []) or "unknown"
            if result_key({"model": model_id(model), "worker": "central"}) in done:
                continue
            record_failure({"model": model_id(model), "worker": "central", "worker_id": None, "task": task,
                            "grade_suite": None}, "no_suite", f"no suite for task {task}",
                           persist=False)
            continue
        suite_of[model_id(model)] = chosen; gradable.append(model)
    size_of = lambda m: m.get("effective_bytes") or m.get("size_bytes") or float("inf")
    gradable.sort(key=lambda m: (model_id(m) not in hot and not m.get("hot"), size_of(m)))
    rank = {model_id(m): i for i, m in enumerate(gradable)}
    judge_target = _hot_judge(workers, full_catalog) if any(s.judge for s in suite_of.values()) else None
    lanes = _lanes(client, workers, [m for m in gradable if suite_of[model_id(m)].pinned])
    for model in gradable:
        if not suite_of[model_id(model)].pinned:
            lanes.append({"model": model_id(model), "model_record": model, "joined": {}, "worker_record": {},
                          "worker": "hugpy-placed", "worker_id": None, "quant": "runtime default",
                          "size_bytes": model.get("effective_bytes") or model.get("size_bytes")})
    lanes.sort(key=lambda lane: (rank[lane["model"]], lane["model"] not in (lane["worker_record"].get("loaded_models") or [])))
    public = lambda lane: {k: v for k, v in lane.items() if k not in {"model_record", "joined", "worker_record", "size_bytes"}}
    # A gradable model with NO lane never reaches the loop, so say exactly why
    # here instead of letting the run end "complete" with nothing graded.
    laned = {lane["model"] for lane in lanes}
    eligible_workers = [w.get("name") or w.get("id") for w in workers if eligible(w)]
    for model in gradable:
        mid = model_id(model)
        if mid in laned: continue
        joined = model.get("workers") or []
        # Lanes are now built for every SELECTED eligible worker regardless of
        # designation, so the only way a gradable model has no lane is that the
        # selection contains no eligible worker at all — say exactly that.
        if not eligible_workers:
            reason = ("no eligible worker among the selected worker(s) — a worker must be online, "
                      "admission-approved, reachable and have serving on; nothing can serve this model "
                      "for the test")
        else:
            reason = (f"no runnable quant/worker pair resolved on the {len(eligible_workers)} eligible "
                      f"worker(s) {eligible_workers}")
        admission = model.get("admission") if isinstance(model.get("admission"), dict) else {}
        record_failure({"model": mid, "worker": "central", "worker_id": None, "quant": "runtime default",
                        "config": "standard", "alloc_mode": "auto", "task": model.get("primary_task"),
                        "grade_suite": suite_of[mid].name}, "no_lane", reason,
                       {"eligible_workers": eligible_workers,
                        "joined": [{k: j.get(k) for k in ("worker", "designated", "loaded", "status")} for j in joined],
                        "admission": {k: admission.get(k) for k in ("status", "reason", "integrity")},
                        "hot": bool(model.get("hot"))})

    def lane_variations(lane):
        chosen = suite_of[lane["model"]]
        if not chosen.pinned:
            yield "standard", "auto", {}, True, None, lane.get("size_bytes"); return
        for precision, mode, extras, runnable, reason, runtime_bytes in _variations(lane):
            if chosen.modes and mode not in chosen.modes:
                runnable, reason = False, f"{chosen.name} grades {'/'.join(chosen.modes)} allocations only"
            yield precision, mode, extras, runnable, reason, runtime_bytes

    cold_of, cold_label = {}, {}
    for lane in lanes:
        if not suite_of[lane["model"]].pinned:
            cold_of[id(lane)], cold_label[id(lane)] = None, "n/a (hugpy-placed)"
            continue
        rec = None if force_cold else _recorded_cold(
            cold_store, lane["model"], lane["quant"], lane["worker"],
            lane.get("size_bytes") or (lane.get("model_record") or {}).get("effective_bytes"))
        cold_of[id(lane)] = rec
        cold_label[id(lane)] = _cold_label(rec) if rec else (
            "will measure (force_cold)" if force_cold else "will measure")
    budget_of = {}
    for lane in lanes:
        grader_ = suite_of[lane["model"]]
        n_items = sum(len(tiers) for tiers in (getattr(grader_, "tasks", None) or {}).values())
        rate = _worker_rate(cold_store, lane["worker"]) if grader_.pinned else None
        need = model_budget(b, _cold_budget(lane, b, rate), n_items)
        budget_of[lane["model"]] = max(budget_of.get(lane["model"], 0.0), need)
    planned = []
    for lane in lanes:
        for precision, mode, _extras, runnable, reason, runtime_bytes in lane_variations(lane):
            planned.append({**public(lane), "cold": cold_label[id(lane)],
                            "budget_model_s": round(budget_of[lane["model"]], 1),
                            "config": precision, "alloc_mode": mode, "runnable": runnable,
                            "status": "N/A" if runnable else "hardware_constraint_failed", "constraint_reason": reason,
                            "disk_bytes": lane.get("size_bytes"), "runtime_bytes": runtime_bytes,
                            "grade_suite": suite_of[lane["model"]].name, "grade": "N/A", "tok_s": "N/A"})
    total = sum(row["runnable"] for row in planned)
    budget_total = sum(budget_of.get(model_id(m), b["model_s"]) for m in gradable)
    report("plan", {"total": len(planned), "runnable": total, "workers": len({r["worker_id"] for r in planned}),
                    "rows": planned, "budgets": b, "budget_total_s": budget_total})
    completed = 0
    model_deadline, dead_models, worker_strikes, dead_workers = {}, {}, {}, {}

    def remaining_budget():
        now = time.monotonic()
        started = sum(max(0.0, d - now) for m, d in model_deadline.items() if m not in dead_models)
        return round(started + sum(budget_of.get(model_id(m), b["model_s"]) for m in gradable
                                   if model_id(m) not in model_deadline), 1)

    def progress(plan):
        report("progress", {"completed": completed, "total": total, "percent": round(100 * completed / max(1, total), 2),
                            "remaining_budget_s": remaining_budget(), **plan})

    def strike(worker, cls):
        """Count consecutive unreachable failures; True when the worker is now dead."""
        if cls != "unreachable":
            worker_strikes[worker] = 0; return False
        worker_strikes[worker] = worker_strikes.get(worker, 0) + 1
        if worker_strikes[worker] >= UNREACHABLE_STRIKES and worker != "hugpy-placed":
            dead_workers[worker] = f"worker {worker} unreachable {worker_strikes[worker]}x in a row"
            return True
        return False

    for lane in lanes:
        if stop.is_set(): break
        grader = suite_of[lane["model"]]; mid, wname = lane["model"], lane["worker"]
        suite_fields = {"grade_suite": grader.name, "grade_task": grader.grade_task}
        variations = list(lane_variations(lane))

        def lane_plan(variation):
            precision, mode, _e, _r, _reason, runtime_bytes = variation
            return {**public(lane), "config": precision, "alloc_mode": mode,
                    "disk_bytes": lane.get("size_bytes"), "runtime_bytes": runtime_bytes, **suite_fields}

        def skip_rest(cls, reason, evidence=None, runnable_only=True):
            nonlocal completed
            for variation in variations:
                if runnable_only and not variation[3]: continue
                record_failure(lane_plan(variation), cls, reason, evidence); completed += 1
            progress(lane_plan(variations[0]) if variations else {})

        if mid in dead_models:
            cls, reason, evidence = dead_models[mid]; skip_rest(cls, f"skipped: {reason}", evidence); continue
        if wname in dead_workers:
            skip_rest("unreachable", "skipped: " + dead_workers[wname]); continue
        model_s = budget_of.get(mid, b["model_s"])
        deadline = model_deadline.setdefault(mid, time.monotonic() + model_s)
        if time.monotonic() >= deadline:
            skip_rest("timeout", f"skipped: model budget {model_s:.0f}s exhausted"); continue
        if done and variations and all(result_key(lane_plan(v)) in done for v in variations):
            completed += sum(1 for v in variations if v[3]); progress(lane_plan(variations[0])); continue
        grades = {}; cold_s = None; cold_info = {}
        cold_budget = _cold_budget(lane, b, _worker_rate(cold_store, wname))
        recorded = cold_of.get(id(lane))
        if recorded:
            cold_s = recorded["cold_load_s"]
            cold_info = {"cold_source": _cold_label(recorded), "cold_measured": False,
                         **{k: recorded[k] for k in COLD_SPLIT_KEYS if recorded.get(k) is not None}}
        if grader.pinned:
            try:
                _select_quant(client, lane)
            except Exception as exc:
                report("notice", {**public(lane), "phase": "select-quant", "error": str(exc)})
            if recorded:
                report("notice", {**public(lane), "phase": "cold-reset", "error": None,
                                  "skipped": "cold load " + cold_info["cold_source"]})
            else:
                try:
                    _cold_reset(client, lane)
                    report("notice", {**public(lane), "phase": "cold-reset", "error": None})
                except Exception as exc:
                    # Continue to the normal seat so an older worker can still be
                    # benchmarked, but expose that the cold reset was unavailable.
                    report("notice", {**public(lane), "phase": "cold-reset", "error": str(exc)})
        for index, variation in enumerate(variations):
            precision, mode, _extras, runnable, reason, runtime_bytes = variation
            plan = {k: v for k, v in lane_plan(variation).items() if grader.name != "hugpy-native-v2" or k not in suite_fields}
            if done and result_key(plan) in done:
                if runnable: completed += 1; progress(plan)
                continue
            if not runnable:
                result = {**plan, "matrix": True, "ok": False, "status": "hardware_constraint_failed",
                          "constraint_reason": reason, "error": reason, "grade": "N/A", "tok_s": "N/A", "detail": {}}
                results.append(result); report("result", result); continue
            if stop.is_set() or (hasattr(stop, "cancelled") and stop.cancelled(worker_id=lane["worker_id"], model=mid, quant=lane["quant"], call=precision + ":" + mode)): break
            if (mid, lane["quant"], _alloc_key(precision, mode), wname) in recent:
                record_failure(plan, "resume", f"skipped: graded within {b['resume_hours']:g}h (resume)", persist=False)
                completed += 1; progress(plan); continue
            bounded = _BoundedClient(client, stop, {"worker_id": lane["worker_id"], "model": mid, "quant": lane["quant"],
                                                    "call": precision + ":" + mode}, deadline)
            lane_started = time.monotonic()

            def fail(cls, why, evidence=None, **extra):
                nonlocal completed
                row = record_failure({**plan, **suite_fields}, cls, why, evidence,
                                     elapsed_s=round(time.monotonic() - lane_started, 4), **extra)
                completed += 1; progress(plan); return row

            try:
                phase = "cold-load" if cold_s is None else "hot-load"
                bounded.set_phase(phase, cold_budget)
                seat_wall = time.time()
                seat_started = time.monotonic(); seat = grader.call(bounded, lane, variation, grader.seat, 1); seat_error = seat["error"]
                seat_s = round(time.monotonic() - seat_started, 4)
                report("notice", {**plan, "phase": phase, "elapsed_s": seat_s, "error": seat_error})
                if seat_error:
                    cls, evidence = classify(seat_error)
                    if cls in ("hard_load_failure", "held"):
                        first = evidence.get("loader_stderr_first_line")
                        why = f"{cls} on {wname}" + (f" — {first}" if first else "")
                        dead_models[mid] = (cls, why, evidence)
                    elif strike(wname, cls):
                        pass
                    fail(cls, f"{phase} failed ({cls}): {seat_error}", evidence, phase=phase)
                    if mid in dead_models or wname in dead_workers: break
                    continue
                strike(wname, "ok")
                if cold_s is None:
                    cold_s, hot_s = seat_s, seat_s
                    if grader.pinned:
                        cold_info = {"cold_source": "measured", "cold_measured": True,
                                     **_cold_split(client, lane, seat_wall)}
                        heartbeat_rate = _cap(lane.get("worker_record") or {}, "load_bytes_per_s")
                        if heartbeat_rate: cold_info["worker_load_bytes_per_s"] = heartbeat_rate
                else: hot_s = seat_s
                bounded.set_phase("call", b["call_s"])
                abort = None
                if precision not in grades:
                    detail, answers, calls, judged = {}, {}, [], {}
                    text_judge = _brain_judge(exclude=mid)
                    for category, tiers in grader.tasks.items():
                        history, judge_history = [], []
                        for tier, prompt, checker in tiers:
                            bounded.set_phase(f"call {category} ({tier})", b["call_s"])
                            response = grader.call(bounded, lane, variation, prompt, tokens)
                            error, speed = response["error"], response.get("tok_s")
                            if error:
                                cls, evidence = classify(error)
                                if cls in ("hard_load_failure", "held"):
                                    dead_models[mid] = (cls, f"{cls} on {wname}", evidence); abort = (cls, error, evidence)
                                elif strike(wname, cls):
                                    abort = (cls, error, evidence)
                            else:
                                strike(wname, "ok")
                            check_pass = bool(not error and grader.score(response, checker))
                            expected, actual = describe(checker), _actual(response)
                            # Two independent verdicts.  The automatic check is the first
                            # read; the agent DEFAULT BRAIN, named by key and routed by
                            # central, then decides substance (correct) and instruction
                            # compliance (format) separately and may revise it.  When the
                            # model under test IS the brain the ladder is exhausted: no
                            # judge, recorded explicitly on the call.
                            passed, format_ok, verdict, revised = check_pass, (check_pass if not error else None), None, False
                            if not error and isinstance(response.get("answer"), str):
                                if not text_judge.get("model"):
                                    verdict = {"model": None, "correct": None, "format_ok": None,
                                               "reason": None, "error": text_judge["exhausted"],
                                               "status": "unavailable"}
                                elif defer_judge:
                                    # PHASE 1 (collect): the raw output + the
                                    # deterministic check are recorded now; the
                                    # brain judge runs in PHASE 2 (judge_collected)
                                    # after the whole lot, so no judge load contends
                                    # with the models under test. Left PENDING.
                                    verdict = {"model": text_judge["model"], "correct": None,
                                               "format_ok": None, "reason": None, "error": None,
                                               "status": "pending"}
                                else:
                                    bounded.set_phase(f"judge {category} ({tier})", b["call_s"])
                                    verdict = judge_reply(bounded, text_judge, prompt, expected, actual, check_pass)
                                    verdict["status"] = "judged" if verdict["correct"] is not None else "error"
                                    if verdict["correct"] is not None:
                                        passed, format_ok = bool(verdict["correct"]), bool(verdict["format_ok"])
                                        revised = passed != check_pass
                            why = _why(passed, error, expected, actual)
                            if verdict and verdict.get("reason") and not passed:
                                why = f"judge: {verdict['reason']}"
                            elif revised and verdict:
                                why = f"judge revised the check: {verdict.get('reason') or 'correct in substance'}"
                            item = {"tier": tier, "pass": passed, "check_pass": check_pass, "format": format_ok,
                                    "revised": revised, "prompt": prompt if isinstance(prompt, str) else str(prompt.get("text") or prompt.get("prompt") or prompt) if isinstance(prompt, dict) else str(prompt),
                                    "expected": expected, "expected_answer": expected_answer(checker), "actual": actual,
                                    **({"why": why} if why else {}),
                                    **({"judge": {k: verdict[k] for k in ("model", "correct", "format_ok", "reason", "error", "raw", "status") if k in verdict}}
                                       if verdict else {})}
                            history.append(item)
                            call = {**plan, "task": f"{category} ({tier})", "timestamp": time.time(),
                                    "elapsed_s": response["elapsed_s"], "tok_s": speed or "N/A",
                                    "tok_s_avg": speed or "N/A", "ctx_in": response.get("ctx_in"), "ctx_out": response.get("ctx_out"),
                                    "caller": "orchestrator", "output": response.get("output"),
                                    "grade": "ERROR" if error else "PASS" if passed else "FAIL", "passed": passed, "error": error or "N/A",
                                    "check_pass": check_pass, "format": format_ok, "revised": revised, "prompt": item["prompt"],
                                    "expected": expected, "expected_answer": item["expected_answer"], "actual": actual,
                                    **({"why": why} if why else {}),
                                    **({"judge": item["judge"]} if verdict else {})}
                            if grader.name != "hugpy-native-v2":
                                call["check"] = getattr(checker, "rule", None)
                                for key in ("s_per_image", "images_per_s", "image_bytes"):
                                    if key in response: call[key] = response[key]
                            if grader.judge and judge_target and not error:
                                verdict = grader.judge(bounded, judge_target, response, prompt)
                                call["judge"] = verdict; judge_history.append({"tier": tier, "pass": bool(verdict["pass"])})
                            calls.append(call); report("call", call)
                            if stop.is_set() or abort: break
                        detail[category] = {"tier": sum(x["pass"] for x in history), "max": 3,
                                            "format": sum(1 for x in history if x.get("format") is True),
                                            "revised": sum(1 for x in history if x.get("revised")), "history": history}
                        answers[category] = history
                        if judge_history:
                            judged["judge_" + category] = {"tier": sum(x["pass"] for x in judge_history), "max": 3,
                                                           "history": judge_history, "model": judge_target["model"]}
                        if stop.is_set() or abort: break
                    if abort:
                        cls, error, evidence = abort
                        fail(cls, f"graded call failed ({cls}): {error}", evidence, phase="call")
                        break
                    answered = [c for c in calls if c.get("grade") in ("PASS", "FAIL")]
                    if not answered:
                        # No item was answered: there is NO grade. A 0/N here would be a
                        # lie about aptitude; the reasons are the recorded call errors.
                        errs = [c.get("error") for c in calls if c.get("error") not in (None, "N/A", "")]
                        first = errs[0] if errs else "no graded call was made"
                        cls0, ev0 = classify(first) if errs else ("error", {})
                        fail(cls0 if errs else "no_answers",
                             f"no item answered ({len(calls)} call(s), {len(errs)} error(s)); first: {first}",
                             {**(ev0 or {}), "errors": errs}, phase="call")
                        break
                    depths = {k: v["tier"] for k, v in detail.items()}
                    score, maximum = sum(depths.values()), grader.max
                    if with_judge and judged:
                        detail.update(judged); score += sum(v["tier"] for v in judged.values()); maximum += 3 * len(judged)
                    format_score = sum(v.get("format") or 0 for v in detail.values() if isinstance(v, dict))
                    revised_n = sum(v.get("revised") or 0 for v in detail.values() if isinstance(v, dict))
                    grades[precision] = {"detail": detail, "answers": answers, "calls": calls, "score": score, "max": maximum,
                                         "format_score": format_score, "format_max": maximum, "revised": revised_n,
                                         "task_best": max(depths, key=depths.get), "task_worst": min(depths, key=depths.get),
                                         "judge": ({"status": ("pending" if defer_judge else "judged"),
                                                    "model": text_judge["model"], "revised": revised_n}
                                                   if text_judge.get("model") else
                                                   {"status": "unavailable",
                                                    "reason": text_judge["exhausted"]})}
                    if judged: grades[precision]["judge"].update({"counted": bool(with_judge), "vision_model": judge_target["model"], "detail": judged})
                grade = grades.get(precision, {"detail": {}, "answers": {}, "calls": [], "score": 0, "max": grader.max, "task_best": "N/A", "task_worst": "N/A"})
                graded_calls = grade.get("calls", [])
                if grader.speed is not None:
                    bounded.set_phase("speed probe", b["call_s"])
                    probe = grader.call(bounded, lane, variation, grader.speed, tokens)
                    speed_error, inference_s, speed = probe["error"], probe["elapsed_s"], probe.get("tok_s")
                else:
                    # No separate probe (image generation): the graded calls are the throughput sample.
                    speed_error, speed = None, None
                    inference_s = _mean([c.get("elapsed_s") for c in graded_calls if c.get("grade") != "ERROR"])
            except LaneTimeout as exc:
                fail("timeout", str(exc), {"phase": exc.phase, "elapsed_s": round(exc.elapsed_s, 3), "budget_s": exc.budget_s,
                                           "load_report": _last_load_report(client, lane["worker_id"], mid)},
                     phase=exc.phase)
                continue
            except LaneCancelled as exc:
                fail("cancelled", f"cancelled: {exc}")
                break
            # Some relays return the summary without usage metadata. The
            # completed cognitive calls above still provide a valid measured
            # throughput for this allocation, so use their mean as the row's
            # tok/s rather than publishing N/A.
            call_speeds = [c.get("tok_s") for c in graded_calls
                           if isinstance(c.get("tok_s"), (int, float)) and c.get("tok_s") > 0]
            if speed is None and call_speeds:
                speed = sum(call_speeds) / len(call_speeds)
            result = {**plan, **grade, "matrix": True, "ok": not speed_error,
                      "status": "complete" if not speed_error else "error", "grade": f"{grade['score']}/{grade['max']}",
                      "format_grade": (f"{grade['format_score']}/{grade['format_max']}" if grade.get("format_max") else None),
                      "cold_s": cold_s, **cold_info, "cold_shared": True, "hot_load_s": hot_s, "inference_s": inference_s,
                      "tok_s": speed or "N/A", "tok_s_avg": speed or "N/A", "metrics_complete": speed is not None,
                      "error": speed_error or "N/A", "finished": time.time()}
            if grader.name != "hugpy-native-v2":
                per_image = _mean([c.get("s_per_image") for c in graded_calls])
                if per_image is not None:
                    result.update(s_per_image=round(per_image, 4), images_per_s=round(1 / per_image, 6),
                                  image_bytes_avg=_mean([c.get("image_bytes") for c in graded_calls]),
                                  metrics_complete=True)
            results.append(result); completed += 1; report("result", result)
            progress(plan)
    judge_calls = [c for r in results for c in (r.get("calls") or [])
                   if isinstance(c, dict) and isinstance(c.get("judge"), dict)]
    judge_failures = sum(1 for c in judge_calls if c["judge"].get("error"))
    report("summary", {"execution": {"configurations": len(results)}, "workers": [], "models": [], "quants": [],
                       "dead_models": {m: v[1] for m, v in dead_models.items()}, "dead_workers": dead_workers,
                       "failures": sum(1 for r in results if r.get("failure_class")),
                       "judge_calls": len(judge_calls), "judge_failures": judge_failures})
    report("notice", "HugPy-native benchmark complete")
