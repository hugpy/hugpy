"""Tiered cognitive grading plus fixed-allocation fleet benchmarking."""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import quote


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
                detail = exc.read().decode("utf-8", "replace")[:4000]
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


def verbose_catalog(client):
    return response_rows(client.request("/models?verbose=1"), "models")


def model_id(model):
    return model.get("model_key") or model.get("id") or model.get("name")


def _ints(s): return [int(x) for x in re.findall(r"-?\d+", s or "")]
def _last(n): return lambda s: _ints(s)[-1:] == [n]
def _has(word): return lambda s: word.lower() in (s or "").lower()
def _exact(value): return lambda s: (s or "").strip() == value


def _json_eq(expected):
    def check(value):
        match = re.search(r"\{.*\}", value or "", re.S)
        try: return bool(match) and json.loads(match.group(0)) == expected
        except ValueError: return False
    return check


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
    "format_primes": (("easy", "List the first three prime numbers separated by commas and nothing else.", lambda s: _ints(s)[:3] == [2, 3, 5]),
                      ("medium", "List the first five prime numbers separated by commas and nothing else.", lambda s: _ints(s)[:5] == [2, 3, 5, 7, 11]),
                      ("hard", "List the first five prime numbers in reverse order separated by commas and nothing else.", lambda s: _ints(s)[:5] == [11, 7, 5, 3, 2])),
    "logic": (("easy", "If all bloops are razzies, and all razzies are lazzies, are all bloops lazzies? Answer yes or no.", _has("yes")),
              ("medium", "A is taller than B. C is shorter than B. Who is the shortest? Reply only with the letter.", lambda s: (s or "").strip().lower() == "c"),
              ("hard", "Can a 3-gallon jug and a 5-gallon jug measure exactly 4 gallons? Answer yes or no.", _has("yes"))),
    "exact_instruction": (("easy", "Reply with exactly the single word: BANANA", _exact("BANANA")),
                          ("medium", "Reply with exactly the single word: BANANA, but in lowercase.", _exact("banana")),
                          ("hard", "Reply with exactly the string: [BANANA_123] and absolutely nothing else.", _exact("[BANANA_123]"))),
    "coding": (("easy", "Write a Python one-liner using sum() to total a list named xs. Reply with only code.", lambda s: "sum(xs)" in re.sub(r"\s+", "", s or "")),
               ("medium", "Write a Python list comprehension returning only even numbers from list xs. Reply with only code.", lambda s: "%2==0" in re.sub(r"\s+", "", s or "")),
               ("hard", "Write a recursive Python lambda named fib for Fibonacci. Reply with only code.", lambda s: "lambda" in s and "-1" in s and "-2" in s)),
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


def _variations(lane):
    model, joined, worker = lane["model_record"], lane["joined"], lane["worker_record"]
    size = lane.get("size_bytes") or model.get("effective_bytes") or model.get("size_bytes")
    vram, ram = _cap(worker, "max_vram_bytes", "vram_total"), _cap(worker, "max_ram_bytes", "ram_total")
    candidates = [("standard", "gpu_only", {}, size, vram), ("standard", "ram_only", {}, size, ram)]
    if joined.get("bnb_4bit") is not None or model.get("is_4bit_capable") or model.get("bnb_capable"):
        four = int(size * .55) if size else None
        candidates += [("4-bit", "gpu_only", {"bnb_4bit": True}, four, vram), ("4-bit", "ram_only", {"bnb_4bit": True}, four, ram)]
    if joined.get("moe") is not None or model.get("is_moe_capable") or model.get("moe"):
        spec = model.get("moe") if isinstance(model.get("moe"), dict) else {}
        total = (model.get("moe_explicit_vram") or spec.get("non_expert_bytes") or 0) + (model.get("moe_explicit_ram") or spec.get("expert_bytes") or 0)
        candidates.append(("standard", "explicit", {"moe": True}, total or size, vram))
    rows = []
    for precision, mode, extras, required, capacity in candidates:
        runnable = required is None or capacity is None or required <= capacity
        kind = "RAM" if mode == "ram_only" else "VRAM"
        reason = None if runnable else f"Size {required} bytes exceeds {kind} capacity {capacity} bytes"
        rows.append((precision, mode, extras, runnable, reason, required))
    return rows


def _lanes(client, workers, models):
    by_id, by_name, lanes = {w.get("id"): w for w in workers}, {w.get("name"): w for w in workers}, []
    for model in models:
        mid = model_id(model); serving = _serving(client, mid)
        for joined in model.get("workers") or []:
            worker = by_id.get(joined.get("worker_id")) or by_name.get(joined.get("worker"))
            if not worker or not eligible(worker) or not joined.get("designated"): continue
            for quant in _quants(model, serving, worker.get("id")):
                lanes.append({"model": mid, "model_record": model, "joined": joined, "worker_record": worker,
                              "worker": worker.get("name") or worker.get("id"), "worker_id": worker.get("id"), **quant})
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
            "alloc": {"worker": lane["worker"], "alloc_mode": mode, **extras}}


def _call(client, lane, variation, prompt, tokens):
    started = time.time(); response = None; error = None; answer = ""
    while True:
        try:
            # Inference is an intentionally unbounded queued operation.  The
            # client default is only for metadata/control-plane requests; a
            # fixed 60s read deadline turns a legitimate cold load or queue
            # wait into a destructive false benchmark failure.
            response = client.request("/v1/chat/completions", "POST",
                                      _body(lane, variation, prompt, tokens),
                                      timeout=None)
            answer = _content(response)
            break
        except FleetError as exc:
            message = str(exc)
            if "worker_busy" not in message and "concurrency limit" not in message:
                error = f"{type(exc).__name__}: {message}"; break
            # Treat admission pressure as queueing, not a failed model call.
            # The worker/central dispatcher decides when capacity is free.
            time.sleep(2)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"; break
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


def run_capacity_benchmark(client, workers, tokens, stop, report, model_ids=None, worker_ids=None):
    """Grade once per precision and benchmark every physically valid allocation."""
    catalog = verbose_catalog(client)
    if model_ids: catalog = [m for m in catalog if model_id(m) in set(model_ids)]
    if worker_ids:
        wanted = set(worker_ids); workers = [w for w in workers if w.get("id") in wanted or w.get("name") in wanted]
    lanes = _lanes(client, workers, catalog); planned = []
    public = lambda lane: {k: v for k, v in lane.items() if k not in {"model_record", "joined", "worker_record", "size_bytes"}}
    for lane in lanes:
        for precision, mode, _extras, runnable, reason, runtime_bytes in _variations(lane):
            planned.append({**public(lane), "config": precision, "alloc_mode": mode, "runnable": runnable,
                            "status": "N/A" if runnable else "hardware_constraint_failed", "constraint_reason": reason,
                            "disk_bytes": lane.get("size_bytes"), "runtime_bytes": runtime_bytes,
                            "grade": "N/A", "tok_s": "N/A"})
    total = sum(row["runnable"] for row in planned)
    report("plan", {"total": len(planned), "runnable": total, "workers": len({r["worker_id"] for r in planned}), "rows": planned})
    completed, results = 0, []
    for lane in lanes:
        if stop.is_set(): break
        _select_quant(client, lane); grades = {}; cold_s = None
        try:
            _cold_reset(client, lane)
            report("notice", {**public(lane), "phase": "cold-reset", "error": None})
        except Exception as exc:
            # Continue to the normal seat so an older worker can still be
            # benchmarked, but expose that the cold reset was unavailable.
            report("notice", {**public(lane), "phase": "cold-reset", "error": str(exc)})
        for variation in _variations(lane):
            precision, mode, _extras, runnable, reason, runtime_bytes = variation
            plan = {**public(lane), "config": precision, "alloc_mode": mode,
                    "disk_bytes": lane.get("size_bytes"), "runtime_bytes": runtime_bytes}
            if not runnable:
                result = {**plan, "matrix": True, "ok": False, "status": "hardware_constraint_failed",
                          "constraint_reason": reason, "error": reason, "grade": "N/A", "tok_s": "N/A", "detail": {}}
                results.append(result); report("result", result); continue
            if stop.is_set() or (hasattr(stop, "cancelled") and stop.cancelled(worker_id=lane["worker_id"], model=lane["model"], quant=lane["quant"], call=precision + ":" + mode)): break
            seat_started = time.monotonic(); _answer, seat_error, _elapsed, _speed, _ctx_in, _ctx_out = _call(client, lane, variation, "Reply only: ready", 1)
            seat_s = round(time.monotonic() - seat_started, 4)
            if cold_s is None: cold_s, hot_s, phase = seat_s, seat_s, "cold-load"
            else: hot_s, phase = seat_s, "hot-load"
            report("notice", {**plan, "phase": phase, "elapsed_s": seat_s, "error": seat_error})
            if precision not in grades and not seat_error:
                detail, answers, calls = {}, {}, []
                for category, tiers in TASKS_TIERED.items():
                    history = []
                    for tier, prompt, checker in tiers:
                        answer, error, elapsed, speed, ctx_in, ctx_out = _call(client, lane, variation, prompt, tokens)
                        passed = not error and bool(checker(answer)); history.append({"tier": tier, "pass": passed})
                        call = {**plan, "task": f"{category} ({tier})", "timestamp": time.time(),
                                "elapsed_s": elapsed, "tok_s": speed or "N/A",
                                "tok_s_avg": speed or "N/A", "ctx_in": ctx_in, "ctx_out": ctx_out,
                                "caller": "orchestrator", "output": answer,
                                "grade": "ERROR" if error else "PASS" if passed else "FAIL", "passed": passed, "error": error or "N/A"}
                        calls.append(call); report("call", call)
                        if stop.is_set(): break
                    detail[category] = {"tier": sum(x["pass"] for x in history), "max": 3, "history": history}; answers[category] = history
                depths = {k: v["tier"] for k, v in detail.items()}
                grades[precision] = {"detail": detail, "answers": answers, "calls": calls, "score": sum(depths.values()), "max": 27,
                                     "task_best": max(depths, key=depths.get), "task_worst": min(depths, key=depths.get)}
            grade = grades.get(precision, {"detail": {}, "answers": {}, "calls": [], "score": 0, "max": 27, "task_best": "N/A", "task_worst": "N/A"})
            _answer, speed_error, inference_s, speed, _ctx_in, _ctx_out = _call(client, lane, variation,
                "Generate a standard 100 word summary of theoretical optics and microfabrication parameters.", tokens)
            # Some relays return the summary without usage metadata. The
            # completed cognitive calls above still provide a valid measured
            # throughput for this allocation, so use their mean as the row's
            # tok/s rather than publishing N/A.
            call_speeds = [c.get("tok_s") for c in grade.get("calls", [])
                           if isinstance(c.get("tok_s"), (int, float)) and c.get("tok_s") > 0]
            if speed is None and call_speeds:
                speed = sum(call_speeds) / len(call_speeds)
            result = {**plan, **grade, "matrix": True, "ok": not seat_error and not speed_error,
                      "status": "complete" if not seat_error and not speed_error else "error", "grade": f"{grade['score']}/{grade['max']}",
                      "cold_s": cold_s, "cold_shared": True, "hot_load_s": hot_s, "inference_s": inference_s,
                      "tok_s": speed or "N/A", "tok_s_avg": speed or "N/A", "metrics_complete": speed is not None,
                      "error": seat_error or speed_error or "N/A", "finished": time.time()}
            results.append(result); completed += 1; report("result", result)
            report("progress", {"completed": completed, "total": total, "percent": round(100 * completed / max(1, total), 2), **plan})
    report("summary", {"execution": {"configurations": len(results)}, "workers": [], "models": [], "quants": []})
    report("notice", "HugPy-native benchmark complete")
