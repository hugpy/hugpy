"""Per-worker TEST FIRE: randomly call every text-generation model on one worker.

Operator ask (2026-09-29): "a test button where it starts to randomly call
every model on that worker which does text gen". One background job per worker
walks that worker's text-generation models in a fresh random order each round
and sends each one a tiny prompt through the NORMAL /v1 intake — the same
``_completion_kwargs`` + ``_v1_events`` path an OpenAI client (and the
fire_keys.py harness) hits — with the request pinned to the worker via the
existing ``alloc.worker`` steer, so placement resolves to that box or fails
naming why (never silently reroutes).

Routes (bare and under /api, like the rest of the worker surface):

    POST /llm/workers/<id>/test-fire                    {rounds?, concurrency?, max_tokens?, seed?}
                                                        -> {job_id, ...}   409 if one is already running
    GET  /llm/workers/<id>/test-fire                    the worker's current/last job (or 404)
    GET  /llm/workers/<id>/test-fire/<job_id>           {running, round, done_calls, total_planned,
                                                         results[], summary{ok, failed, per_model}}
    POST /llm/workers/<id>/test-fire/<job_id>/stop      stop after in-flight calls finish
                                                        (in-flight jobs get the authoritative cancel)

Model SELECTION reads the catalog rows the console already renders
(``get_models_dict(dict_return=True)`` -> ``tasks`` / ``primary_task``): a
model is text-gen when it carries ``text-generation`` / ``text2text-generation``
AND no vision / image / video / audio / embedding / rerank task. Operator-
blocked and archive-marked keys are skipped. The candidate set is the worker's
designations plus what it holds locally (``models``, ``models_local`` and the
storage rows), i.e. "that worker's models" as the row shows them.

Results are kept in memory only: a ring of the last ``RESULT_RING`` calls per
job plus per-model last status; finished jobs stay readable until
``MAX_FINISHED`` newer ones displace them.
"""
from __future__ import annotations

import random
import threading
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional

from flask import jsonify, request

from abstract_flask import get_bp

test_fire_bp, logger = get_bp("test_fire_bp", __name__)

# ── selection vocabulary ─────────────────────────────────────────────────────
TEXT_GEN_TASKS = frozenset({"text-generation", "text2text-generation"})
# Any of these on a row makes it NOT a plain text-gen target for this button
# (vision-language, image/video generation, audio, embeddings, rerank, parts).
NON_TEXT_TASKS = frozenset({
    "image-text-to-text", "visual-question-answering", "image-to-text",
    "text-to-image", "image-to-image", "image-to-video", "text-to-video",
    "automatic-speech-recognition", "text-to-speech", "audio-classification",
    "feature-extraction", "sentence-similarity", "keyword-extraction",
    "embedding", "rerank", "reranking",
    "depth-estimation", "object-detection", "image-classification",
    "image-segmentation", "pipeline-component", "adapter", "needs-classification",
})

PROMPTS = (
    "Reply with one short sentence: what is 2+2?",
    "Name one primary color.",
    "Say hello in one word.",
    "What is the capital of France? Answer briefly.",
    "Finish this: the sky is",
    "Give a one-word synonym for 'fast'.",
    "Count from 1 to 3.",
    "What color is grass? One word.",
)

RESULT_RING = 200
MAX_FINISHED = 20
DEFAULT_MAX_TOKENS = 32
MAX_MAX_TOKENS = 512
MAX_CONCURRENCY = 8
MAX_ROUNDS = 1000


def model_tasks(row: dict) -> set:
    """The task vocabulary the harness/assortment use: ``tasks`` ∪ ``primary_task``."""
    tasks = set(t for t in (row.get("tasks") or []) if isinstance(t, str))
    pt = row.get("primary_task")
    if isinstance(pt, str) and pt:
        tasks.add(pt)
    return tasks


def is_text_gen(row: dict) -> bool:
    """Plain text generation: has a text-gen task and no vision/image/audio/
    embedding task (a ``['image-text-to-text', 'text-generation']`` VL row is
    NOT a target here — it is exercised by the vision tooling)."""
    if not isinstance(row, dict):
        return False
    tasks = model_tasks(row)
    return bool(tasks & TEXT_GEN_TASKS) and not (tasks & NON_TEXT_TASKS)


def worker_model_keys(worker: dict) -> list:
    """Ordered, de-duplicated candidate keys: designations + local files +
    storage rows — what the worker row lists as 'its' models."""
    seen: "OrderedDict[str, None]" = OrderedDict()
    for k in (worker.get("models") or []):
        if isinstance(k, str) and k:
            seen.setdefault(k, None)
    for k in (worker.get("models_local") or []):
        if isinstance(k, str) and k:
            seen.setdefault(k, None)
    storage = worker.get("storage") or {}
    for r in (storage.get("models") or []) if isinstance(storage, dict) else []:
        k = r.get("model_key") if isinstance(r, dict) else None
        if isinstance(k, str) and k:
            seen.setdefault(k, None)
    return list(seen.keys())


def _catalog_row(catalog: dict, key: str) -> Optional[dict]:
    """Registry lookup tolerant of the bare/qualified spellings (``owner~X`` vs
    ``X``) the worker roster carries; exact key first."""
    row = catalog.get(key)
    if isinstance(row, dict):
        return row
    tail = key.split("~", 1)[1] if "~" in key else None
    if tail and isinstance(catalog.get(tail), dict):
        return catalog[tail]
    for k, r in catalog.items():
        if isinstance(r, dict) and (r.get("model_key") == key or k.split("~", 1)[-1] == key):
            return r
    return None


def _key_forms(key: str) -> set:
    """Spellings one model id can wear across the catalog / a worker's store:
    ``owner/X`` (hub id), ``owner~X`` (qualified key), ``X`` (bare key)."""
    k = str(key or "").strip().rstrip("/")
    if not k:
        return set()
    forms = {k, k.lower()}
    for sep in ("/", "~"):
        if sep in k:
            owner, tail = k.split(sep, 1)
            forms |= {tail, tail.lower(), f"{owner}~{tail}", f"{owner}/{tail}",
                      f"{owner}~{tail}".lower(), f"{owner}/{tail}".lower()}
    return forms


def adapter_base_on_worker(base_model: Optional[str], worker: dict) -> bool:
    """True when the adapter's ``base_model`` (hub id) is among the keys this
    worker's store / designations carry, under any spelling."""
    want = _key_forms(base_model)
    if not want:
        return False
    for key in worker_model_keys(worker):
        if _key_forms(key) & want:
            return True
    return False


def unserveable_reason(row: dict) -> Optional[str]:
    """The catalog's own ``extra.serveable=False`` verdict (e.g. the engine's
    "PEFT adapter (base '…') — … NOT in this store" refusal), else None."""
    extra = row.get("extra")
    if isinstance(extra, dict) and extra.get("serveable") is False:
        return str(extra.get("unserveable_reason") or "marked unserveable")
    return None


def select_text_gen_models(worker: dict, catalog: dict,
                           blocked: Iterable[str] = (),
                           archived: Iterable[str] = ()) -> tuple:
    """(selected, skipped) for one worker. ``selected`` rows are
    ``{model_key, framework, tasks}``; ``skipped`` rows carry a ``reason``.

    Rows that cannot load deterministically are skipped up front so the strip
    can say "skipped: base missing" instead of painting a red failure: the
    catalog's ``extra.serveable=False`` verdict, and a PEFT/LoRA adapter row
    (``base_model`` set) whose base is not in THIS worker's store."""
    blocked = set(blocked or ())
    archived = set(archived or ())
    selected, skipped = [], []
    for key in worker_model_keys(worker):
        if key in blocked:
            skipped.append({"model_key": key, "reason": "operator-blocked"})
            continue
        if key in archived:
            skipped.append({"model_key": key, "reason": "archive-marked"})
            continue
        row = _catalog_row(catalog, key)
        if row is None:
            skipped.append({"model_key": key, "reason": "not in catalog"})
            continue
        if row.get("blocked"):
            skipped.append({"model_key": key, "reason": "operator-blocked"})
            continue
        if not is_text_gen(row):
            skipped.append({"model_key": key,
                            "reason": "not text-gen: " + ",".join(sorted(model_tasks(row))) or "no task"})
            continue
        why = unserveable_reason(row)
        if why:
            skipped.append({"model_key": key, "reason": "unserveable: " + why,
                            "kind": "base missing" if "PEFT adapter" in why else "unserveable"})
            continue
        base = row.get("base_model")
        if isinstance(base, str) and base.strip() and not adapter_base_on_worker(base, worker):
            skipped.append({"model_key": key, "kind": "base missing",
                            "reason": f"adapter base {base!r} is not in this worker's store"})
            continue
        selected.append({"model_key": key,
                         "framework": row.get("framework"),
                         "tasks": sorted(model_tasks(row) & TEXT_GEN_TASKS)})
    return selected, skipped


# ── error classification (mirrors the harness's structured_error markers) ────
_ERROR_KINDS = (
    ("LoadRefusal", "loadrefusal"),
    ("fit_failure", "fit_failure"),
    ("fit_failure", "does not fit"),
    ("worker_busy", "worker_busy"),
    ("model_busy", "model_busy"),
    ("requested_worker", "requested worker"),
    ("context_overflow", "context length"),
    ("capacity", "admission cap"),
    ("timeout", "timed out"),
    ("timeout", "timeout"),
)


def classify_error(message: Optional[str]) -> Optional[str]:
    if not message:
        return None
    low = str(message).lower()
    for kind, marker in _ERROR_KINDS:
        if marker in low:
            return kind
    return "error"


# ── bookkeeping ──────────────────────────────────────────────────────────────
class TestFireJob:
    """In-memory state for one test-fire run (thread-safe snapshot/record)."""

    def __init__(self, worker_id: str, worker_name: str, models: list, *,
                 rounds: int = 1, concurrency: int = 1,
                 max_tokens: int = DEFAULT_MAX_TOKENS, seed: Optional[int] = None,
                 skipped: Optional[list] = None):
        self.job_id = uuid.uuid4().hex[:12]
        self.worker_id = worker_id
        self.worker_name = worker_name
        self.models = [m["model_key"] if isinstance(m, dict) else str(m) for m in models]
        self.rounds = int(rounds)            # 0 = until stopped
        self.concurrency = int(concurrency)
        self.max_tokens = int(max_tokens)
        self.seed = seed
        self.skipped = list(skipped or [])
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.running = False
        self.round = 0
        self.done_calls = 0
        self.ok = 0
        self.failed = 0
        self.stop_reason: Optional[str] = None
        self.results: deque = deque(maxlen=RESULT_RING)
        self.per_model: dict = {k: {"ok": 0, "failed": 0, "last_status": None,
                                    "last_error": None, "last_latency_s": None,
                                    "last_tok_s": None, "last_at": None}
                                for k in self.models}
        self.in_flight: dict = {}   # request_id -> model_key
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    # ----- control
    @property
    def total_planned(self) -> Optional[int]:
        return None if self.rounds == 0 else self.rounds * len(self.models)

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def request_stop(self, reason: str = "operator") -> None:
        with self._lock:
            if self.stop_reason is None:
                self.stop_reason = reason
        self._stop.set()

    # ----- recording
    def mark_in_flight(self, request_id: Optional[str], model_key: str) -> None:
        if request_id:
            with self._lock:
                self.in_flight[request_id] = model_key

    def record(self, result: dict) -> None:
        """Append one call result and fold it into the counters/per-model view."""
        key = result.get("model_key")
        rid = result.get("request_id")
        with self._lock:
            if rid:
                self.in_flight.pop(rid, None)
            self.results.append(result)
            self.done_calls += 1
            pm = self.per_model.setdefault(key, {"ok": 0, "failed": 0, "last_status": None,
                                                 "last_error": None, "last_latency_s": None,
                                                 "last_tok_s": None, "last_at": None})
            if result.get("ok"):
                self.ok += 1
                pm["ok"] += 1
            else:
                self.failed += 1
                pm["failed"] += 1
            pm["last_status"] = "ok" if result.get("ok") else "fail"
            pm["last_error"] = result.get("error")
            pm["last_error_kind"] = result.get("error_kind")
            pm["last_latency_s"] = result.get("latency_s")
            pm["last_tok_s"] = result.get("tok_s")
            pm["last_at"] = result.get("started")

    def snapshot(self, limit: int = RESULT_RING) -> dict:
        with self._lock:
            results = list(self.results)[-limit:]
            per_model = {k: dict(v) for k, v in self.per_model.items()}
            in_flight = dict(self.in_flight)
        return {
            "job_id": self.job_id,
            "worker_id": self.worker_id,
            "worker": self.worker_name,
            "running": self.running,
            "round": self.round,
            "rounds": self.rounds,
            "concurrency": self.concurrency,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "models": list(self.models),
            "skipped": list(self.skipped),
            "done_calls": self.done_calls,
            "total_planned": self.total_planned,
            "in_flight": [{"request_id": r, "model_key": k} for r, k in in_flight.items()],
            "stop_requested": self.stop_requested(),
            "stop_reason": self.stop_reason,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "results": results,
            "summary": {"ok": self.ok, "failed": self.failed,
                        "total": self.done_calls, "per_model": per_model},
        }


# ── the run loop (pure: the call function is injected) ───────────────────────
def run_job(job: TestFireJob, call_fn: Callable[..., dict],
            prompts: tuple = PROMPTS) -> None:
    """Round after round, every model once in a fresh random order; ``call_fn``
    (model_key, prompt, max_tokens, job) -> result dict. Stops when the round
    budget is spent, on ``request_stop`` (after in-flight calls finish), or when
    there are no models."""
    rng = random.Random(job.seed)
    job.started = time.time()
    job.running = True
    try:
        if not job.models:
            job.request_stop("no text-gen models")
            return
        with ThreadPoolExecutor(max_workers=max(1, job.concurrency)) as pool:
            while not job.stop_requested():
                if job.rounds and job.round >= job.rounds:
                    break
                job.round += 1
                order = list(job.models)
                rng.shuffle(order)
                pending = []
                for key in order:
                    if job.stop_requested():
                        break
                    prompt = rng.choice(prompts)
                    fut = pool.submit(_guarded_call, call_fn, key, prompt, job)
                    pending.append(fut)
                    # concurrency=1 degenerates to strictly sequential calls so
                    # a warm slot is never piled on (worker_busy contract).
                    if len(pending) >= max(1, job.concurrency):
                        pending.pop(0).result()
                for fut in pending:
                    fut.result()
    except Exception as exc:  # noqa: BLE001 — the loop must always finish
        logger.exception("test-fire %s: run loop failed", job.job_id)
        job.request_stop(f"loop error: {type(exc).__name__}: {exc}")
    finally:
        job.running = False
        job.finished = time.time()
        if job.stop_reason is None:
            job.stop_reason = "complete"


def _guarded_call(call_fn, key, prompt, job) -> dict:
    t0 = time.time()
    try:
        res = call_fn(key, prompt, job.max_tokens, job)
    except Exception as exc:  # noqa: BLE001
        res = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
               "error_kind": classify_error(f"{exc}") or "exception"}
    res = dict(res or {})
    res.setdefault("model_key", key)
    res.setdefault("prompt", prompt)
    res.setdefault("started", t0)
    res.setdefault("latency_s", round(time.time() - t0, 3))
    res.setdefault("ok", False)
    res.setdefault("round", job.round)
    if not res["ok"] and res.get("error_kind") is None:
        res["error_kind"] = classify_error(res.get("error"))
    job.record(res)
    return res


# ── the real call: the /v1 intake, in-process ────────────────────────────────
def fire_one(model_key: str, prompt: str, max_tokens: int, job: TestFireJob) -> dict:
    """One tiny non-streaming completion through the /v1 intake (same request
    shape the harness POSTs), pinned to the job's worker via ``alloc.worker``.
    Drains the same event stream the non-streaming route drains."""
    from hugpy_server.app.routes.v1_helpers import _completion_kwargs
    from hugpy_server.app.routes.v1_routes import _v1_events
    from hugpy_server.app.functions.chat.streaming import chat_iter_sync

    payload = {
        "model": model_key,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "stream": False,
        "temperature": 0,
        "alloc": {"worker": job.worker_id},
        "user": f"test-fire:{job.job_id}",
    }
    prompt_kwargs = _completion_kwargs(payload)
    prompt_kwargs["caller"] = "test-fire"
    request_id = prompt_kwargs.get("request_id")
    job.mark_in_flight(request_id, model_key)

    started = time.time()
    text_parts: list = []
    error_message = None
    usage = None
    timings = None
    hugpy_meta = None
    worker_seen = None
    try:
        for ev in chat_iter_sync(_v1_events(prompt_kwargs, payload)):
            t = getattr(ev, "type", None)
            if t == "token":
                text_parts.append(getattr(ev, "text", "") or "")
            elif t == "done":
                usage = getattr(ev, "usage", None)
                timings = getattr(ev, "timings", None)
                hugpy_meta = getattr(ev, "hugpy", None)
            elif t == "status":
                worker_seen = (getattr(ev, "worker_name", None)
                               or getattr(ev, "served_by", None) or worker_seen)
            elif t == "error":
                error_message = getattr(ev, "message", None) or "error"
    except Exception as exc:  # noqa: BLE001 — a raised relay error is a result, not a crash
        error_message = f"{type(exc).__name__}: {exc}"
    latency = round(time.time() - started, 3)

    content = "".join(text_parts)
    if "[error:" in content and not error_message:
        error_message = content[:2000]
    ok = error_message is None and bool(content.strip())
    if not ok and error_message is None:
        error_message = "empty completion"

    tokens = None
    if isinstance(usage, dict):
        tokens = usage.get("completion_tokens")
    elif usage is not None:
        tokens = getattr(usage, "completion_tokens", None)
    tok_s = _tok_per_s(timings, tokens, latency)
    if isinstance(hugpy_meta, dict):
        worker_seen = hugpy_meta.get("worker") or hugpy_meta.get("served_by") or worker_seen

    return {
        "model_key": model_key,
        "request_id": request_id,
        "prompt": prompt,
        "started": started,
        "latency_s": latency,
        "ok": ok,
        "status": "ok" if ok else "fail",
        "tokens": tokens,
        "tok_s": tok_s,
        "worker": worker_seen,
        "content80": content[:80] if content else None,
        "error": None if ok else str(error_message)[:2000],
        "error_kind": None if ok else classify_error(error_message),
    }


def _tok_per_s(timings, tokens, latency_s) -> Optional[float]:
    if isinstance(timings, dict):
        call = timings.get("call")
        if isinstance(call, dict) and call.get("tok_per_s") is not None:
            try:
                return round(float(call["tok_per_s"]), 2)
            except (TypeError, ValueError):
                pass
        if timings.get("predicted_per_second") is not None:
            try:
                return round(float(timings["predicted_per_second"]), 2)
            except (TypeError, ValueError):
                pass
    if tokens and latency_s and latency_s > 0:
        try:
            return round(float(tokens) / float(latency_s), 2)
        except (TypeError, ValueError):
            pass
    return None


# ── registry ─────────────────────────────────────────────────────────────────
_JOBS: "OrderedDict[str, TestFireJob]" = OrderedDict()
_BY_WORKER: dict = {}
_REG_LOCK = threading.Lock()


def _running_job_for(worker_id: str) -> Optional[TestFireJob]:
    with _REG_LOCK:
        jid = _BY_WORKER.get(worker_id)
        job = _JOBS.get(jid) if jid else None
    if job is not None and (job.running or (job.thread is not None and job.thread.is_alive())):
        return job
    return None


def _latest_job_for(worker_id: str) -> Optional[TestFireJob]:
    with _REG_LOCK:
        jid = _BY_WORKER.get(worker_id)
        return _JOBS.get(jid) if jid else None


def _register(job: TestFireJob) -> None:
    with _REG_LOCK:
        _JOBS[job.job_id] = job
        _BY_WORKER[job.worker_id] = job.job_id
        # Drop the oldest FINISHED jobs beyond the retention window.
        finished = [j for j in _JOBS.values() if not j.running and j.thread is not None
                    and not j.thread.is_alive()]
        while len(_JOBS) > MAX_FINISHED and finished:
            old = finished.pop(0)
            if _BY_WORKER.get(old.worker_id) == old.job_id and old.job_id != job.job_id:
                # keep the worker's pointer only while it points at a live job
                _BY_WORKER.pop(old.worker_id, None)
            _JOBS.pop(old.job_id, None)


def start_job(worker_id: str, worker_name: str, models: list, *, rounds=1,
              concurrency=1, max_tokens=DEFAULT_MAX_TOKENS, seed=None,
              skipped=None, call_fn: Callable[..., dict] = fire_one,
              spawn: bool = True) -> TestFireJob:
    """Create + register + (by default) spawn the run thread. ``spawn=False``
    leaves the job registered but not started (tests drive ``run_job``)."""
    job = TestFireJob(worker_id, worker_name, models, rounds=rounds,
                      concurrency=concurrency, max_tokens=max_tokens, seed=seed,
                      skipped=skipped)
    _register(job)
    if spawn:
        job.running = True   # visible as running before the thread's first tick
        job.thread = threading.Thread(target=run_job, args=(job, call_fn),
                                      name=f"test-fire-{job.job_id}", daemon=True)
        job.thread.start()
    return job


def reset_registry() -> None:
    """Tests only."""
    with _REG_LOCK:
        _JOBS.clear()
        _BY_WORKER.clear()


# ── collaborators (module attrs so tests can fake them in place) ─────────────
def _get_worker(worker_id: str) -> Optional[dict]:
    from hugpy_fleet.central.workers import get_worker
    return get_worker(worker_id)


def _catalog() -> dict:
    from hugpy_engine.config.models.models_config import get_models_dict
    d = get_models_dict(dict_return=True)
    return d if isinstance(d, dict) else {}


def _blocked_keys() -> set:
    try:
        from hugpy_fleet.central.blocklist import blocked_keys
        return set(blocked_keys() or ())
    except Exception:  # noqa: BLE001 — the blocklist is advisory here
        return set()


def _archived_keys() -> set:
    try:
        from hugpy_fleet.central.archive_gate import archived_keys
        return set(archived_keys() or ())
    except Exception:  # noqa: BLE001
        return set()


def _cancel_request(request_id: str, reason: str) -> bool:
    try:
        from hugpy_control.jobs import job_store
        res = job_store.cancel_authoritative(request_id, reason)
        return bool((res or {}).get("cancelled"))
    except Exception:  # noqa: BLE001 — best-effort; the loop still stops
        logger.debug("test-fire: cancel %s failed", request_id, exc_info=True)
        return False


# ── body validation ──────────────────────────────────────────────────────────
def _int_field(body: dict, name: str, default: int, lo: int, hi: int) -> int:
    v = body.get(name, default)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError(f"{name} must be an integer")
    try:
        iv = int(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer")
    if iv < lo or iv > hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return iv


def parse_start_body(body) -> dict:
    """{rounds, concurrency, max_tokens, seed} with defaults + bounds."""
    body = body if isinstance(body, dict) else {}
    out = {
        "rounds": _int_field(body, "rounds", 1, 0, MAX_ROUNDS),
        "concurrency": _int_field(body, "concurrency", 1, 1, MAX_CONCURRENCY),
        "max_tokens": _int_field(body, "max_tokens", DEFAULT_MAX_TOKENS, 1, MAX_MAX_TOKENS),
        "seed": None,
    }
    if body.get("seed") is not None:
        out["seed"] = _int_field(body, "seed", 0, -(2 ** 31), 2 ** 31 - 1)
    return out


# ── routes ───────────────────────────────────────────────────────────────────
@test_fire_bp.route("/llm/workers/<worker_id>/test-fire", methods=["POST"])
def test_fire_start(worker_id):
    worker = _get_worker(worker_id)
    if not worker:
        return jsonify({"ok": False, "error": "unknown worker"}), 404
    try:
        opts = parse_start_body(request.get_json(silent=True))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    running = _running_job_for(worker_id)
    if running is not None:
        return jsonify({"ok": False, "error": "test-fire already running on this worker",
                        "job_id": running.job_id}), 409
    selected, skipped = select_text_gen_models(worker, _catalog(),
                                               blocked=_blocked_keys(),
                                               archived=_archived_keys())
    if not selected:
        return jsonify({"ok": False, "error": "no text-generation models on this worker",
                        "skipped": skipped}), 400
    job = start_job(worker_id, worker.get("name") or worker_id, selected,
                    rounds=opts["rounds"], concurrency=opts["concurrency"],
                    max_tokens=opts["max_tokens"], seed=opts["seed"], skipped=skipped)
    logger.info("test-fire %s: worker=%s models=%d rounds=%s concurrency=%d max_tokens=%d",
                job.job_id, job.worker_name, len(job.models), job.rounds or "∞",
                job.concurrency, job.max_tokens)
    return jsonify({"ok": True, "job_id": job.job_id, "worker_id": worker_id,
                    "models": job.models, "skipped": skipped,
                    "rounds": job.rounds, "concurrency": job.concurrency,
                    "max_tokens": job.max_tokens, "total_planned": job.total_planned}), 202


@test_fire_bp.route("/llm/workers/<worker_id>/test-fire", methods=["GET"])
def test_fire_current(worker_id):
    job = _latest_job_for(worker_id)
    if job is None:
        return jsonify({"ok": False, "error": "no test-fire job for this worker"}), 404
    return jsonify(job.snapshot())


@test_fire_bp.route("/llm/workers/<worker_id>/test-fire/<job_id>", methods=["GET"])
def test_fire_status(worker_id, job_id):
    with _REG_LOCK:
        job = _JOBS.get(job_id)
    if job is None or job.worker_id != worker_id:
        return jsonify({"ok": False, "error": "unknown test-fire job"}), 404
    try:
        limit = int(request.args.get("limit", RESULT_RING))
    except (TypeError, ValueError):
        limit = RESULT_RING
    return jsonify(job.snapshot(limit=max(1, min(limit, RESULT_RING))))


@test_fire_bp.route("/llm/workers/<worker_id>/test-fire/<job_id>/stop", methods=["POST"])
def test_fire_stop(worker_id, job_id):
    with _REG_LOCK:
        job = _JOBS.get(job_id)
    if job is None or job.worker_id != worker_id:
        return jsonify({"ok": False, "error": "unknown test-fire job"}), 404
    was_running = job.running
    job.request_stop("operator")
    with job._lock:
        in_flight = list(job.in_flight.keys())
    cancelled = [rid for rid in in_flight if _cancel_request(rid, "test-fire stop")]
    return jsonify({"ok": True, "job_id": job.job_id, "was_running": was_running,
                    "in_flight": len(in_flight), "cancelled": cancelled})


__all__ = [
    "test_fire_bp", "TEXT_GEN_TASKS", "NON_TEXT_TASKS", "PROMPTS",
    "is_text_gen", "model_tasks", "worker_model_keys", "select_text_gen_models",
    "adapter_base_on_worker", "unserveable_reason",
    "classify_error", "TestFireJob", "run_job", "fire_one", "start_job",
    "parse_start_body", "reset_registry",
]
