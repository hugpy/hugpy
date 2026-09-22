### routes/review_routes.py
"""Model review over HTTP — the on-demand half of the reviewer.

  GET  /llm/review/criteria            saved criteria
  PUT  /llm/review/criteria/<name>     create/update one
  POST /llm/review/screen              {hub_ids:[...]} metadata verdict, no download
  POST /llm/review/run                 {criteria|hub_ids} full pipeline (background)
  GET  /llm/review/runs                run history
  GET  /llm/review/results             recorded reviews (?criteria=&best=1)
  POST /llm/review/ingest              a worker box pushes a finished run in

CENTRAL'S DB IS THE SOURCE OF TRUTH. The pipeline runs where the GPU is (ae),
writing its rows to that box's local sqlite; without /ingest those rows never
reach the DB these read routes serve and the console shows an empty
leaderboard. /ingest is the landing zone — see review/push.py for the sending
half. Flow is one-way, worker -> central, always.

Screening is synchronous — it's metadata only and returns in seconds. A full
run downloads weights and loads them on the GPU, so it goes to a background
thread and the caller polls /runs; holding a request open for the length of a
20 GB download is how you get a gateway timeout and an orphaned staging dir.
"""
import json
import os
import tempfile

from abstract_flask import get_bp
from flask import abort, jsonify, request

review_bp, logger = get_bp("review_bp", __name__)

_RUNNING: dict = {}          # criteria name -> {"run_id":…, "started":…}

# Capacity benchmark state belongs to central/HugPy.  Execution is delegated
# through a named adapter (initially hugpy-agent), but the console owns the run,
# progress, result stream and report location.
_BENCHMARK = {"status": "idle", "events": [], "results": [], "calls": [],
              "summary": {},
              "plan": {"rows": [], "total": 0, "runnable": 0},
              "progress": {"completed": 0, "total": 0, "percent": 0}}
_BENCHMARK_LOCK = None
_BENCHMARK_STATE = os.path.expanduser(os.environ.get(
    "HUGPY_BENCHMARK_STATE", "~/.local/state/hugpy/benchmark.json"))


def _benchmark_save():
    """Atomically persist the public run so gunicorn workers/restarts agree."""
    parent = os.path.dirname(_BENCHMARK_STATE)
    os.makedirs(parent, mode=0o700, exist_ok=True)
    # The file is mode 0600. Persist the lease token so every gunicorn worker
    # can validate adapter heartbeats, but never expose it from _public().
    public = {k: v for k, v in _BENCHMARK.items()
              if k not in {"thread", "control"}}
    fd, tmp = tempfile.mkstemp(prefix=".benchmark-", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(public, stream, sort_keys=True)
        os.replace(tmp, _BENCHMARK_STATE)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _benchmark_reload():
    try:
        with open(_BENCHMARK_STATE, encoding="utf-8") as stream:
            saved = json.load(stream)
    except (OSError, ValueError):
        return
    if saved.get("updated", 0) > _BENCHMARK.get("updated", 0):
        internals = {k: _BENCHMARK.get(k) for k in ("thread", "control", "lease_token")}
        _BENCHMARK.clear(); _BENCHMARK.update(saved)
        _BENCHMARK.update({k: v for k, v in internals.items() if v is not None})


def _benchmark_updated(now=None):
    import time as _time
    _BENCHMARK["updated"] = now or _time.time()
    _benchmark_save()


def _benchmark_lock():
    global _BENCHMARK_LOCK
    if _BENCHMARK_LOCK is None:
        import threading
        _BENCHMARK_LOCK = threading.RLock()
    return _BENCHMARK_LOCK


def _benchmark_public():
    with _benchmark_lock():
        _benchmark_reload()
        row = {k: v for k, v in _BENCHMARK.items()
               if k not in {"thread", "control", "lease_token"}}
        row["events"] = list(row.get("events", []))[-200:]
        row["results"] = list(row.get("results", []))
        return row


def _metric_number(value):
    """A real finite metric or None; spreadsheet sentinels never enter SQL."""
    import math
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _persist_benchmark_call(row):
    """Append an executed cognitive call to the immutable model_calls ledger."""
    if not isinstance(row, dict) or not row.get("model") or not row.get("worker"):
        return False
    try:
        from hugpy_server.app.routes.metrics_routes import _live_db
        config = str(row.get("config") or "standard")
        mode = str(row.get("alloc_mode") or "")
        alloc = mode if config == "standard" else "%s:%s" % (config, mode)
        state = json.dumps({"caller": row.get("caller") or "orchestrator",
                            "grade": row.get("grade"), "passed": row.get("passed"),
                            "error": row.get("error")}, sort_keys=True)
        query = """
          INSERT INTO model_calls
            (model_name,worker,quant,alloc_mode,tok_per_s,prompt_tokens,
             completion_tokens,elapsed_s,task,request_id,state)
          VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
        """
        params = (str(row["model"]), str(row["worker"]), str(row.get("quant") or ""), alloc,
                  _metric_number(row.get("tok_s_avg") or row.get("tok_s")), row.get("ctx_in"),
                  row.get("ctx_out"), _metric_number(row.get("elapsed_s")), row.get("task"),
                  row.get("request_id"), state)
        with _live_db().cursor() as cursor:
            cursor.execute(query, params)
        return True
    except Exception:
        logger.warning("benchmark call persistence failed for %s/%s",
                       row.get("model"), row.get("task"), exc_info=True)
        return False
def _persist_benchmark_result(row):
    if not isinstance(row, dict) or not row.get("model") or not row.get("worker"):
        return False
    try:
        from hugpy_server.app.routes.metrics_routes import _live_db
        config = str(row.get("config") or "full")
        mode = str(row.get("alloc_mode") or "")
        alloc = mode if config in ("full", "standard") else "%s:%s" % (config, mode)
        cold = _metric_number(row.get("cold_s"))
        # Loading and grading are separate measurements. A lane cold-loads
        # once; every later allocation variation records its own hot reload.
        hot = _metric_number(row.get("hot_load_s"))
        tok = _metric_number(row.get("tok_s_avg"))
        if tok is None:
            tok = _metric_number(row.get("tok_s"))
        # A transport/auth failure is not a grading result. Keep telemetry that
        # exists, but never replace a previously valid score with 0/N/A.
        valid_grade = (row.get("status") == "complete" and
                       row.get("error") in (None, "N/A", ""))
        score, maximum = _metric_number(row.get("score")), _metric_number(row.get("max"))
        grade = 100.0 * score / maximum if valid_grade and score is not None and maximum else None
        detail = json.dumps(row.get("detail") or {}, sort_keys=True) if valid_grade else None
        query = """
          INSERT INTO model_metrics
            (model_name, quant, alloc_mode, worker, cold_load_s, hot_load_s,
             tok_per_s, tok_per_s_avg, n_samples, task, updated_at,
             grade, grade_suite, grade_detail, graded_at)
          VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'text-generation',now(),
                  %s,'hugpy-native-v2',%s,
                  CASE WHEN %s::double precision IS NULL THEN NULL ELSE now() END)
          ON CONFLICT (model_name, quant, alloc_mode, worker) DO UPDATE SET
            cold_load_s=COALESCE(EXCLUDED.cold_load_s,model_metrics.cold_load_s),
            hot_load_s=COALESCE(EXCLUDED.hot_load_s,model_metrics.hot_load_s),
            tok_per_s=COALESCE(EXCLUDED.tok_per_s,model_metrics.tok_per_s),
            tok_per_s_avg=CASE WHEN EXCLUDED.tok_per_s IS NULL THEN model_metrics.tok_per_s_avg
              ELSE (COALESCE(model_metrics.tok_per_s_avg,0)*model_metrics.n_samples
                    +EXCLUDED.tok_per_s)/(model_metrics.n_samples+1) END,
            n_samples=model_metrics.n_samples+CASE WHEN EXCLUDED.tok_per_s IS NULL THEN 0 ELSE 1 END,
            task=EXCLUDED.task, updated_at=now(),
            grade=COALESCE(EXCLUDED.grade,model_metrics.grade),
            grade_suite=COALESCE(EXCLUDED.grade_suite,model_metrics.grade_suite),
            grade_detail=COALESCE(EXCLUDED.grade_detail,model_metrics.grade_detail),
            graded_at=COALESCE(EXCLUDED.graded_at,model_metrics.graded_at)
        """
        sample = 1 if tok is not None else 0
        params = (str(row["model"]), str(row.get("quant") or ""), alloc,
                  str(row["worker"]), cold, hot, tok, tok, sample,
                  grade, detail, grade)
        with _live_db().cursor() as cursor:
            cursor.execute(query, params)
        return True
    except Exception:
        logger.warning("benchmark metric persistence failed for %s/%s",
                       row.get("model"), row.get("quant"), exc_info=True)
        return False


@review_bp.route("/llm/benchmark/lock", methods=["POST"])
def benchmark_lock():
    """Exclusive lease used by `hugpy-agent console` test mode."""
    import secrets
    import time as _time
    body = request.get_json(silent=True) or {}
    action = body.get("action")
    with _benchmark_lock():
        _benchmark_reload()
        if action == "acquire":
            if _BENCHMARK.get("status") == "running":
                return jsonify(_benchmark_public()), 409
            token = secrets.token_urlsafe(24)
            _BENCHMARK.clear()
            _BENCHMARK.update(status="running", run_id="agent-" + secrets.token_hex(6),
                              executor="hugpy-agent-console", started=_time.time(),
                              finished=None, exclusive=True, lease_token=token,
                              events=[], results=[], calls=[], summary={},
                              plan={"rows": [], "total": 0, "runnable": 0},
                              progress={"completed": 0, "total": 0, "percent": 0})
            _BENCHMARK["heartbeat"] = _time.time()
            _benchmark_updated()
            public = _benchmark_public(); public["lease_token"] = token
            return jsonify(public), 201
        if action in {"heartbeat", "event"}:
            if not _BENCHMARK.get("lease_token") or not secrets.compare_digest(
                    str(body.get("lease_token") or ""), _BENCHMARK["lease_token"]):
                abort(403, description="invalid benchmark lease")
            _BENCHMARK["heartbeat"] = _time.time()
            if action == "event":
                kind, value = body.get("kind"), body.get("value")
                if kind == "result" and isinstance(value, dict):
                    _BENCHMARK["results"].append(value)
                    _persist_benchmark_result(value)
                elif kind == "call" and isinstance(value, dict):
                    _BENCHMARK["calls"].append(value)
                    _persist_benchmark_call(value)
                elif kind in {"plan", "progress", "summary"} and isinstance(value, dict):
                    _BENCHMARK[kind] = value
                else:
                    _BENCHMARK["events"].append(
                        {"at": _time.time(), "kind": kind, "value": value,
                         "message": value if isinstance(value, str) else None})
            _benchmark_updated()
            return jsonify({"ok": True, "run_id": _BENCHMARK.get("run_id")})
        if action == "release":
            if not _BENCHMARK.get("lease_token") or not secrets.compare_digest(
                    str(body.get("lease_token") or ""), _BENCHMARK["lease_token"]):
                abort(403, description="invalid benchmark lease")
            _BENCHMARK.update(status=body.get("status", "complete"),
                              finished=_time.time(), lease_token=None)
            _benchmark_updated()
            return jsonify(_benchmark_public())
    abort(400, description="action must be acquire, heartbeat, event, or release")


@review_bp.route("/llm/benchmark/status", methods=["GET"])
def benchmark_status():
    """Current capacity benchmark, including live per-quant results."""
    import time as _time
    with _benchmark_lock():
        _benchmark_reload()
        local_thread = _BENCHMARK.get("thread")
        locally_alive = local_thread is not None and local_thread.is_alive()
        if (_BENCHMARK.get("status") == "running" and not locally_alive and
                _time.time() - float(_BENCHMARK.get("heartbeat") or 0) > 45):
            _BENCHMARK.update(status="interrupted", finished=_time.time(),
                              error="executor heartbeat expired")
            _benchmark_updated()
    return jsonify(_benchmark_public())


@review_bp.route("/llm/benchmark/run", methods=["POST"])
def benchmark_run():
    """Start the HugPy-native per-worker model test from the console.

    The adapter makes ordinary central chat-completion calls with the existing
    per-request worker pin. HugPy remains solely responsible for selection,
    placement, provisioning, loading, dispatch, and eviction.
    """
    import threading
    import time as _time
    import uuid

    body = request.get_json(silent=True) or {}
    executor = body.get("executor", "hugpy-central")
    if executor != "hugpy-central":
        abort(400, description="executor must be hugpy-central")
    try:
        tokens = max(8, min(2048, int(body.get("tokens", 128))))
    except (TypeError, ValueError):
        abort(400, description="tokens must be an integer")
    model_ids = [str(value) for value in (body.get("models") or []) if value]
    worker_ids = [str(value) for value in (body.get("workers") or []) if value]

    with _benchmark_lock():
        if _BENCHMARK.get("status") == "running":
            return jsonify(_benchmark_public()), 409
        run_id = uuid.uuid4().hex[:12]
        _BENCHMARK.clear()
        _BENCHMARK.update({"run_id": run_id, "status": "running",
                           "executor": executor, "started": _time.time(),
                           "finished": None, "tokens": tokens,
                           "heartbeat": _time.time(),
                           "models": model_ids, "workers": worker_ids,
                           "events": [], "results": [], "calls": [],
                           "summary": {},
                           "plan": {"rows": [], "total": 0, "runnable": 0},
                           "progress": {"completed": 0, "total": 0, "percent": 0},
                           "report": None})
        _benchmark_updated()

    def report(kind, value):
        with _benchmark_lock():
            _BENCHMARK["heartbeat"] = _time.time()
            if kind == "result" and isinstance(value, dict):
                _BENCHMARK["results"].append(value)
                _persist_benchmark_result(value)
            elif kind == "call" and isinstance(value, dict):
                _BENCHMARK["calls"].append(value)
                _persist_benchmark_call(value)
            elif kind == "plan" and isinstance(value, dict):
                _BENCHMARK["plan"] = value
                _BENCHMARK["progress"]["total"] = value.get("runnable", 0)
            elif kind == "progress" and isinstance(value, dict):
                _BENCHMARK["progress"] = value
            elif kind == "summary" and isinstance(value, dict):
                _BENCHMARK["summary"] = value
            else:
                _BENCHMARK["events"].append(
                    {"at": _time.time(), "kind": kind, "value": value,
                     "message": value if isinstance(value, str) else None})
                # The agent reports the durable report path in its last notice.
                marker = "Capacity benchmark complete: "
                if isinstance(value, str) and value.startswith(marker):
                    _BENCHMARK["report"] = value[len(marker):]
            _benchmark_updated()

    def work():
        try:
            from hugpy_curation.review.fleet_grading import (
                BenchmarkControl,
                Client,
                response_rows,
                run_capacity_benchmark,
            )
            base = (os.environ.get("HUGPY_BENCHMARK_API") or
                    "http://127.0.0.1:7002").rstrip("/")
            client = Client(base,
                            key=os.environ.get("HUGPY_BENCHMARK_API_KEY", ""),
                            operator_token=os.environ.get("HUGPY_OPERATOR_TOKEN", ""),
                            timeout=60)
            workers = response_rows(client.request("/llm/workers"), "workers")
            control = BenchmarkControl()
            with _benchmark_lock():
                _BENCHMARK["control"] = control
            run_capacity_benchmark(client, workers, tokens, control, report,
                                   model_ids=model_ids or None,
                                   worker_ids=worker_ids or None)
            # Do not call an all-error/timeout run successful merely because
            # the worker loop returned. A model that answered normally should
            # have at least one completed allocation result.
            with _benchmark_lock():
                completed_results = [r for r in _BENCHMARK.get("results", [])
                                     if isinstance(r, dict) and r.get("status") == "complete"]
            status, error = ("complete", None) if completed_results else (
                "failed", "No allocation completed successfully; see cold-load/seat errors")
        except Exception as exc:  # background failure must remain visible
            logger.exception("capacity benchmark failed")
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        with _benchmark_lock():
            _BENCHMARK.update(status=status, error=error, finished=_time.time())
            _benchmark_updated()

    thread = threading.Thread(target=work, name=f"benchmark-{run_id}", daemon=True)
    with _benchmark_lock():
        _BENCHMARK["thread"] = thread
    thread.start()
    return jsonify(_benchmark_public()), 202


@review_bp.route("/llm/benchmark/cancel", methods=["POST"])
def benchmark_cancel():
    """Cancel/skip one call, quant, model, worker, or the whole execution."""
    body = request.get_json(silent=True) or {}
    scope = body.get("scope", "execution")
    if scope not in {"call", "quant", "model", "worker", "execution"}:
        abort(400, description="scope must be call, quant, model, worker, or execution")
    with _benchmark_lock():
        control = _BENCHMARK.get("control")
        if _BENCHMARK.get("status") != "running" or control is None:
            return jsonify({"status": "not_running"}), 409
        try:
            control.cancel(scope=scope, worker_id=body.get("worker_id"),
                           model=body.get("model"), quant=body.get("quant"),
                           call=body.get("call"))
        except ValueError as exc:
            abort(400, description=str(exc))
        _BENCHMARK["events"].append({"at": __import__("time").time(),
            "kind": "cancel", "message": "cancelled %s" % scope,
            "scope": scope, "target": {k: body.get(k) for k in
                ("worker_id", "model", "quant", "call") if body.get(k)}})
    return jsonify(_benchmark_public())


def _crit(name_or_payload):
    from hugpy_curation.review.criteria import ReviewCriteria, load_criteria
    if isinstance(name_or_payload, str):
        return load_criteria(name_or_payload)
    return ReviewCriteria.from_dict(name_or_payload)


@review_bp.route("/llm/review/criteria", methods=["GET"])
def review_criteria_list():
    from hugpy_curation.review.criteria import list_criteria, load_criteria
    out = []
    for name in list_criteria():
        try:
            out.append(load_criteria(name).to_dict())
        except Exception as exc:
            out.append({"name": name, "error": str(exc)})
    return jsonify(out)


@review_bp.route("/llm/review/criteria/<name>", methods=["PUT"])
def review_criteria_put(name):
    from hugpy_curation.review.criteria import ReviewCriteria, save_criteria
    payload = request.get_json(silent=True) or {}
    payload["name"] = name
    crit = ReviewCriteria.from_dict(payload)
    save_criteria(crit)
    return jsonify(crit.to_dict())


@review_bp.route("/llm/review/screen", methods=["POST"])
def review_screen():
    """Metadata-only verdict for specific repos. Fast, fetches no weights."""
    payload = request.get_json(silent=True) or {}
    hub_ids = payload.get("hub_ids") or ([payload["hub_id"]]
                                         if payload.get("hub_id") else [])
    if not hub_ids:
        abort(400, description="hub_ids is required")
    if len(hub_ids) > 50:
        abort(400, description="at most 50 hub_ids per request")

    crit = _crit(payload.get("criteria") or payload.get("criteria_name") or {"name": "adhoc"})
    from hugpy_curation.review.screen import screen, _hf_api
    api = _hf_api()
    results = []
    for hub_id in hub_ids:
        try:
            results.append(screen(hub_id, crit, api=api).to_dict())
        except Exception as exc:
            results.append({"hub_id": hub_id, "passed": False,
                            "reasons": [f"{type(exc).__name__}: {exc}"]})
    return jsonify({"criteria": crit.name, "results": results})


@review_bp.route("/llm/review/run", methods=["POST"])
def review_run():
    """Full pipeline in the background. One run per criteria at a time — two
    concurrent runs would fight over the GPU and the disk cap."""
    import threading
    import time as _time

    payload = request.get_json(silent=True) or {}
    name = payload.get("criteria") or payload.get("criteria_name")
    if not name:
        abort(400, description="criteria is required")
    crit = _crit(name)
    hub_ids = payload.get("hub_ids") or None
    force = bool(payload.get("force"))

    live = _RUNNING.get(crit.name)
    if live and live.get("thread") and live["thread"].is_alive():
        return jsonify({"status": "already_running", **{
            k: v for k, v in live.items() if k != "thread"}}), 409

    from hugpy_curation.review import pipeline

    def _work():
        try:
            pipeline.run(crit, hub_ids=hub_ids, force=force,
                         log=lambda m: logger.info("[review] %s", m))
        except Exception as exc:
            logger.exception("review run for %s failed: %s", crit.name, exc)

    t = threading.Thread(target=_work, name=f"review-{crit.name}", daemon=True)
    _RUNNING[crit.name] = {"started": _time.time(), "thread": t}
    t.start()
    return jsonify({"status": "started", "criteria": crit.name}), 202


@review_bp.route("/llm/review/runs", methods=["GET"])
def review_runs():
    from hugpy_curation.review import store
    return jsonify(store.runs(request.args.get("criteria"),
                              limit=request.args.get("limit", 20, type=int)))


@review_bp.route("/llm/review/status", methods=["GET"])
def review_status():
    """One GET for the console settings page (2026-08-13): every saved
    criteria (with its ``enabled`` switch) plus whether an API-started run is
    in flight for it right now. Timer-started runs live in another process —
    they show up in /runs, not here; ``running`` covers only this process's
    background threads (the console's own "Run now")."""
    from hugpy_curation.review.criteria import list_criteria, load_criteria
    crits = []
    for name in list_criteria():
        try:
            c = load_criteria(name).to_dict()
        except Exception as exc:
            c = {"name": name, "error": str(exc)}
        live = _RUNNING.get(name)
        running = bool(live and live.get("thread") and live["thread"].is_alive())
        c["running"] = running
        if running:
            c["running_since"] = live.get("started")
        crits.append(c)
    return jsonify({"criteria": crits})


# ── ingest: a worker box hands central its finished run ───────────────────
# Cap one POST. push.py chunks at 250; anything far above that is a confused or
# hostile caller, and the run header rides every chunk so truncating the tail
# would silently lose results rather than fail honestly.
MAX_INGEST_RESULTS = 1000


def _ingest_authorized() -> bool:
    """Ingest gate: the SAME credential /llm/evictions/ingest uses.

    Modelled on eviction_routes._worker_authorized — delegates to
    worker_routes._enrollment_ok, so a worker presents the enrollment bearer it
    already holds for register/heartbeat and revoking that worker stops its
    review pushes at the same instant it stops its heartbeat. No new secret to
    rotate. An operator token is accepted too (manual replay from a laptop,
    `python -m abstract_hugpy_dev.review push`).

    Fails CLOSED if the gate cannot be imported: an unauthenticated write
    endpoint is not an acceptable degradation."""
    try:
        from hugpy_server.app.operator_auth import operator_authenticated
        if operator_authenticated():
            return True
    except Exception:                       # noqa: BLE001
        pass                                # fall through to the worker gate
    # STRICTER than register/heartbeat on purpose (keeper, 2026-07-29): the
    # gradual-rollout "no token -> allow" that _enrollment_ok grants would make
    # this WRITE endpoint publicly writable through the internet-facing origin
    # (a probe proved it). A present, valid enrollment token is required
    # always; the fleet holds per-box tokens since the same day.
    try:
        from hugpy_server.app.routes.worker_routes import _bearer_token
        from hugpy_fleet.central.enrollment_tokens import verify_enrollment_token
        tok = _bearer_token()
        return tok is not None and bool(verify_enrollment_token(tok))
    except Exception:                       # noqa: BLE001
        logger.warning("review ingest: enrollment gate unavailable — refusing")
        return False


@review_bp.route("/llm/review/ingest", methods=["POST"])
def review_ingest():
    """Accept one pushed run + its results from the box that produced them.

    Body::

        {"host": "ae",
         "criteria": "nightly",
         "run": {"run_id": 12, "criteria": "nightly", "started_at": …,
                 "finished_at": …, "screened": …, "passed": …,
                 "downloaded": …, "smoked": …, "error": null},
         "results": [{"run_id": 12, "criteria": …, "hub_id": …, "stage": …,
                      "passed": …, "score": …, "verdict": …,
                      "payload": {…}, "reviewed_at": …}, …]}

    IDEMPOTENT. Rows are keyed on (host, run_id) and
    (host, run_id, hub_id, stage), so a retried or chunk-replayed push updates
    in place — a worker that never sees our 200 can resend forever without
    duplicating a single row.

    NEVER 5xx OVER DATA. A malformed row is counted in ``rejected`` and the
    rest of the batch still lands; a store fault answers 200 with everything
    counted rejected. The only non-2xx answers are 401 (unauthenticated) and
    400 (the envelope itself is not the documented shape) — because a worker
    that reads a 5xx as "central is broken, retry harder" is how a telemetry
    channel turns into a storm."""
    if not _ingest_authorized():
        return jsonify({"error": "Worker enrollment or operator token required."}), 401
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "expected a JSON object"}), 400
    host = (body.get("host") or "").strip() if isinstance(body.get("host"), str) else ""
    if not host:
        return jsonify({"error": "host is required — rows are keyed on it"}), 400
    run = body.get("run")
    results = body.get("results")
    if run is not None and not isinstance(run, dict):
        return jsonify({"error": "run must be an object"}), 400
    if results is None:
        results = []
    if not isinstance(results, list):
        return jsonify({"error": "results must be a list"}), 400
    if len(results) > MAX_INGEST_RESULTS:
        return jsonify({"error": f"at most {MAX_INGEST_RESULTS} results per "
                                 f"request; send more chunks"}), 400

    from hugpy_curation.review import store

    run_row = None
    run_ok = False
    if run:
        try:
            run_row = store.ingest_run(host, run)
            run_ok = run_row is not None
        except Exception as exc:            # noqa: BLE001 — never 5xx at a worker
            logger.warning("review ingest: run upsert failed for %s: %s", host, exc)

    accepted = rejected = 0
    try:
        accepted, rejected = store.ingest_results(host, results)
    except Exception as exc:                # noqa: BLE001
        logger.warning("review ingest: result upsert failed for %s: %s", host, exc)
        rejected = len(results)

    if not run_ok and run:
        rejected += 1                       # the run header itself was unusable
    return jsonify({"ok": True, "host": host, "run_id": run_row,
                    "accepted": accepted, "rejected": rejected})


# ── k120: dossiers ────────────────────────────────────────────────────────
# The review row carries a compact SUMMARY (discovery_dossier/store.summary);
# the full dossier — card digest, every quant, every mention, every sample — is
# a file. These two routes are the split: /dossiers lists what the console
# renders in the leaderboard, /dossier serves ONE expanded, which is how an
# operator actually reads them.


@review_bp.route("/llm/review/dossiers", methods=["GET"])
def review_dossiers():
    """Every dossier for one card, newest first, as compact summaries."""
    criteria = request.args.get("criteria")
    if not criteria:
        abort(400, description="criteria is required")
    limit = request.args.get("limit", 20, type=int)
    from hugpy_curation.dossier import store as dstore
    out = []
    for path in dstore.list_for(criteria)[:max(1, limit)]:
        dossier = dstore.load_path(path)
        if dossier is not None:
            out.append(dstore.summary(dossier, path))
    return jsonify({"criteria": criteria, "dossiers": out})


@review_bp.route("/llm/review/dossier", methods=["GET"])
def review_dossier():
    """One full dossier. 404 with the reason rather than an empty object — a
    console showing a blank panel cannot tell "never reviewed" from "the file
    is gone"."""
    criteria = request.args.get("criteria")
    hub_id = request.args.get("hub_id")
    if not criteria or not hub_id:
        abort(400, description="criteria and hub_id are required")
    from hugpy_curation.dossier import store as dstore
    dossier = dstore.load(criteria, hub_id)
    if dossier is None:
        return jsonify({"error": f"no dossier on this box for {hub_id} under "
                                 f"criteria {criteria!r}",
                        "path": dstore.path_for(criteria, hub_id)}), 404
    return jsonify(dossier.to_dict())


@review_bp.route("/llm/review/radar", methods=["GET"])
def review_radar():
    """The gem radar's last output for one card — models the cards are not
    asking about that the cached community pulls kept naming."""
    criteria = request.args.get("criteria")
    if not criteria:
        abort(400, description="criteria is required")
    from hugpy_curation.dossier import store as dstore
    return jsonify(dstore.load_radar(criteria) or
                   {"criteria": criteria, "hits": [],
                    "detail": "no radar pass has run for this card "
                              "(set radar: true to enable it)"})


@review_bp.route("/llm/review/results", methods=["GET"])
def review_results():
    from hugpy_curation.review import store
    criteria = request.args.get("criteria")
    limit = request.args.get("limit", 50, type=int)
    if request.args.get("best") not in (None, "", "0"):
        if not criteria:
            abort(400, description="best=1 requires criteria")
        return jsonify(store.leaderboard(criteria, limit=limit))
    return jsonify(store.recent(criteria, limit=limit,
                                stage=request.args.get("stage")))
