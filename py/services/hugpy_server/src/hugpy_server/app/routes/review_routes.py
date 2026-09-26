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
# A run is ACTIVE while collecting (running / resuming — central restarted
# mid-collection and is continuing the same run_id) or JUDGING (phase 2, the
# scheduled judge pass over the collected outputs). Busy checks (pkg_promote's
# quiet-central wait, the admission runner, the single-run 409) treat all the
# same; the restart-resume election tells the phases apart by ``status``.
_BENCHMARK_ACTIVE = ("running", "resuming", "judging")
_BENCHMARK_RESUME_MAX = int(os.environ.get("HUGPY_BENCHMARK_RESUME_MAX", "3") or 3)
_RESTART_REASON = "restarted after central restart"


def _benchmark_state_path():
    """``$HUGPY_BENCHMARK_STATE`` or ``PROJECTS_HOME/benchmark_state.json`` —
    the durable run state a restarted central resumes from."""
    env = (os.environ.get("HUGPY_BENCHMARK_STATE") or "").strip()
    if env:
        return os.path.expanduser(env)
    try:
        from hugpy_platform.constants import PROJECTS_HOME
        return os.path.join(str(PROJECTS_HOME), "benchmark_state.json")
    except Exception:  # noqa: BLE001 — per-user durable fallback
        return os.path.expanduser("~/.local/state/hugpy/benchmark_state.json")


_BENCHMARK_STATE = _benchmark_state_path()
# Where runs were persisted before 2026-09-23; read only while the new file
# does not exist yet, so the run in flight at the cutover restart is resumed.
_BENCHMARK_LEGACY_STATE = os.path.expanduser("~/.local/state/hugpy/benchmark.json")


def _benchmark_runs_dir():
    """Every run, durable: ``<state dir>/benchmark_runs/<run_id>.json``. The
    single state file only ever holds the CURRENT run, so without this a new
    run overwrote the previous run's calls and responses."""
    return os.path.join(os.path.dirname(_BENCHMARK_STATE), "benchmark_runs")


def _benchmark_save():
    """Atomically persist the whole run (plan rows, results, calls, params,
    budgets, owner) on every update, so gunicorn workers agree and a restarted
    central can resume it. The file is 0600: it carries the adapter lease token
    (every gunicorn worker validates heartbeats with it; _public() hides it).
    The same snapshot is also written under its run_id so past runs stay
    readable (``GET /llm/benchmark/runs``)."""
    from hugpy_platform.atomic_json import save_json
    public = {k: v for k, v in _BENCHMARK.items()
              if k not in {"thread", "control"}}
    try:
        snapshot = json.loads(json.dumps(public, default=str))
        save_json(_BENCHMARK_STATE, snapshot)
        os.chmod(_BENCHMARK_STATE, 0o600)
    except OSError:
        logger.warning("benchmark state not persisted to %s", _BENCHMARK_STATE, exc_info=True)
        return
    run_id = snapshot.get("run_id")
    if run_id:
        try:
            os.makedirs(_benchmark_runs_dir(), exist_ok=True)
            path = os.path.join(_benchmark_runs_dir(), f"{run_id}.json")
            save_json(path, {k: v for k, v in snapshot.items() if k != "lease_token"})
        except OSError:
            logger.warning("benchmark run %s not archived under %s", run_id, _benchmark_runs_dir(), exc_info=True)


def _benchmark_run_summary(row):
    err = row.get("error")
    return {"run_id": row.get("run_id"), "status": row.get("status"), "started": row.get("started"),
            "finished": row.get("finished"), "updated": row.get("updated"), "models": row.get("models") or [],
            "workers": row.get("workers") or [], "results": len(row.get("results") or []),
            "calls": len(row.get("calls") or []),
            "error": err.get("headline") if isinstance(err, dict) else err}


@review_bp.route("/llm/benchmark/runs", methods=["GET"])
def benchmark_runs():
    """Every archived run, newest first: run_id, status, started/finished,
    models, result and call counts, error headline."""
    out, errors = [], []
    d = _benchmark_runs_dir()
    try:
        names = [n for n in os.listdir(d) if n.endswith(".json")]
    except FileNotFoundError:
        return jsonify({"runs": [], "dir": d, "reason": f"no archived runs: {d} does not exist yet"})
    for name in names:
        try:
            with open(os.path.join(d, name), encoding="utf-8") as stream:
                out.append(_benchmark_run_summary(json.load(stream)))
        except (OSError, ValueError) as exc:
            errors.append({"file": name, "error": f"{type(exc).__name__}: {exc}"})
    out.sort(key=lambda r: r.get("started") or 0, reverse=True)
    return jsonify({"runs": out, "dir": d, "errors": errors})


@review_bp.route("/llm/benchmark/runs/<run_id>", methods=["GET"])
def benchmark_run_get(run_id):
    """One archived run, whole (every result, every call with its full record)."""
    import re as _re
    if not _re.fullmatch(r"[0-9a-f]{6,32}", run_id or ""):
        return jsonify({"error": f"run id {run_id!r} is not a hex run id"}), 400
    path = os.path.join(_benchmark_runs_dir(), f"{run_id}.json")
    try:
        with open(path, encoding="utf-8") as stream:
            row = json.load(stream)
    except FileNotFoundError:
        return jsonify({"error": f"no archived run {run_id}: {path} does not exist"}), 404
    except (OSError, ValueError) as exc:
        return jsonify({"error": f"archived run {run_id} unreadable at {path}: {type(exc).__name__}: {exc}"}), 500
    row.pop("lease_token", None)
    return jsonify(row)


def _benchmark_read_state():
    for path in (_BENCHMARK_STATE, _BENCHMARK_LEGACY_STATE):
        if path != _BENCHMARK_STATE and os.path.exists(_BENCHMARK_STATE):
            break
        try:
            with open(path, encoding="utf-8") as stream:
                saved = json.load(stream)
        except (OSError, ValueError):
            continue
        if isinstance(saved, dict):
            return saved
    return None


def _benchmark_reload():
    saved = _benchmark_read_state()
    if not saved:
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
        row["events"] = list(row.get("events", []))   # whole; never a tail
        row["results"] = list(row.get("results", []))
        return row


def _proc_start(pid):
    """Kernel start time of ``pid`` (jiffies since boot), or None if it is gone."""
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as stream:
            return int(stream.read().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError, TypeError):
        return None


def _benchmark_owner():
    import socket
    return {"host": socket.gethostname(), "pid": os.getpid(), "start": _proc_start(os.getpid())}


def _benchmark_owner_alive(owner):
    """True while the process that runs the benchmark thread still exists
    (same host, same pid, same start time — a recycled pid is not the owner).
    A run with no recorded owner is judged by its heartbeat instead."""
    import socket
    if not isinstance(owner, dict) or not owner.get("pid"):
        return None
    if owner.get("host") != socket.gethostname():
        return True                     # another host's process: not ours to judge
    start = _proc_start(owner["pid"])
    return start is not None and (owner.get("start") is None or start == owner.get("start"))


_PROCESS_STARTED = __import__("time").time()


def _benchmark_orphaned(state):
    """A hugpy-central run whose owning process is gone. Runs persisted before
    owners were recorded count as orphaned when their last heartbeat predates
    this server process."""
    if state.get("executor") != "hugpy-central" or state.get("status") not in _BENCHMARK_ACTIVE:
        return False
    local = state.get("thread")
    if local is not None and local.is_alive():
        return False
    alive = _benchmark_owner_alive(state.get("owner"))
    if alive is None:
        return float(state.get("heartbeat") or 0) < _PROCESS_STARTED
    return not alive


class _ColdStore:
    """Recorded cold loads, read back by the benchmark loop
    (``fleet_grading.run_capacity_benchmark(cold_store=...)``). Lives on the
    registry's ``model_metrics`` rows: ``cold_load_s`` (total seat time incl.
    the central->worker transfer) plus the split columns this module adds
    (``transfer_s``, ``transfer_bytes``, ``load_s``, ``bytes_per_s``,
    ``cold_measured_at``). Every read is best effort: no DB = "not recorded"."""

    def get(self, model, quant, worker):
        """Only a cold load MEASURED by the benchmark's cold path (the row has
        ``cold_measured_at``) counts; legacy EMA ``cold_load_s`` values do not
        and the lane re-measures (overwriting them)."""
        if not _cold_columns():
            return None
        from hugpy_server.app.routes.metrics_routes import _live_db
        names = list(dict.fromkeys([model, str(model).split("~")[-1]]))
        with _live_db().cursor() as cur:
            cur.execute("SELECT cold_load_s, extract(epoch from cold_measured_at), transfer_s, "
                        "transfer_bytes, load_s, bytes_per_s FROM model_metrics "
                        "WHERE model_name = ANY(%s) AND quant = %s AND worker = %s "
                        "AND cold_load_s > 0 AND cold_measured_at IS NOT NULL "
                        "ORDER BY cold_measured_at DESC LIMIT 1",
                        (names, quant or "", worker))
            row = cur.fetchone()
        if not row:
            return None
        out = {"cold_load_s": float(row[0]), "measured_at": float(row[1])}
        for key, value in zip(("transfer_s", "transfer_bytes", "load_s", "bytes_per_s"), row[2:]):
            if value is not None:
                out[key] = float(value)
        return out

    def worker_rate(self, worker):
        if not _cold_columns():
            return None
        from hugpy_server.app.routes.metrics_routes import _live_db
        with _live_db().cursor() as cur:
            cur.execute("SELECT bytes_per_s FROM model_metrics WHERE worker = %s AND bytes_per_s > 0 "
                        "ORDER BY cold_measured_at DESC NULLS LAST LIMIT 1", (worker,))
            row = cur.fetchone()
        return float(row[0]) if row and row[0] else None


_COLD_COLUMNS = None
_COLD_DDL = (
    "ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS transfer_s DOUBLE PRECISION",
    "ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS transfer_bytes BIGINT",
    "ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS load_s DOUBLE PRECISION",
    "ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS bytes_per_s DOUBLE PRECISION",
    "ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS cold_measured_at TIMESTAMPTZ",
)


def _cold_columns():
    """Whether model_metrics carries the cold-load split columns (added once per
    process, idempotently; the same DDL is in hugpy_engine's MetricsQueries
    MIGRATIONS). False = record/read cold_load_s only."""
    global _COLD_COLUMNS
    if _COLD_COLUMNS is None:
        try:
            from hugpy_server.app.routes.metrics_routes import _live_db
            with _live_db().cursor() as cur:
                try:
                    for ddl in _COLD_DDL:
                        cur.execute(ddl)
                except Exception:  # noqa: BLE001 — not the owner: use them if present
                    logger.info("model_metrics cold columns not added here", exc_info=True)
                cur.execute("SELECT count(*) FROM information_schema.columns WHERE table_name = "
                            "'model_metrics' AND column_name IN ('transfer_s', 'transfer_bytes', "
                            "'load_s', 'bytes_per_s', 'cold_measured_at')")
                _COLD_COLUMNS = cur.fetchone()[0] == 5
        except Exception:  # noqa: BLE001
            logger.warning("model_metrics cold columns unavailable", exc_info=True)
            return False
    return _COLD_COLUMNS


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
        # THE WHOLE CALL RECORD, verbatim and unbounded: prompt, expected,
        # expected_answer, output/actual, check_pass, format, revised, judge,
        # why, failure_class, reason, evidence, timings.  The ledger is the
        # source every view reads; an allowlist here is a data loss.
        state = {k: v for k, v in row.items() if k not in ("model", "worker")}
        state.setdefault("caller", "orchestrator")
        state["benchmark"] = True
        state = json.dumps(state, sort_keys=True, default=str)
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
    if row.get("persist") is False:     # e.g. resume / no-suite skips: nothing measured
        return False
    try:
        from hugpy_server.app.routes.metrics_routes import _live_db
        config = str(row.get("config") or "full")
        mode = str(row.get("alloc_mode") or "")
        alloc = mode if config in ("full", "standard") else "%s:%s" % (config, mode)
        # A reused (recorded) cold load is not a new measurement: never re-stamp it.
        cold = None if row.get("cold_measured") is False else _metric_number(row.get("cold_s"))
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
        # PHASE 1 (collect) persists every raw output (grade_detail) as the run
        # goes, but the GRADE appears only once judged (operator ruling
        # 2026-09-24): a row still awaiting the brain judge is written with its
        # measurements + raw outputs and a NULL grade/graded_at, which PHASE 2
        # (_persist_benchmark_grade) fills in. A row with no pending judge (the
        # historical inline path, an exhausted ladder, a non-judged suite) keeps
        # its grade here.
        awaiting_judge = (isinstance(row.get("judge"), dict)
                          and row["judge"].get("status") == "pending")
        grade = (100.0 * score / maximum if valid_grade and not awaiting_judge
                 and score is not None and maximum else None)
        detail = json.dumps(row.get("detail") or {}, sort_keys=True) if valid_grade else None
        query = """
          INSERT INTO model_metrics
            (model_name, quant, alloc_mode, worker, cold_load_s, hot_load_s,
             tok_per_s, tok_per_s_avg, n_samples, task, updated_at,
             grade, grade_suite, grade_detail, graded_at)
          VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now(),
                  %s,%s,%s,
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
        # Suite-aware (hugpy_curation.review.suites): rows from the vision /
        # imagegen suites name themselves; text rows keep the historical values.
        task = str(row.get("grade_task") or "text-generation")
        suite = str(row.get("grade_suite") or "hugpy-native-v2")
        params = (str(row["model"]), str(row.get("quant") or ""), alloc,
                  str(row["worker"]), cold, hot, tok, tok, sample, task,
                  grade, suite, detail, grade)
        with _live_db().cursor() as cursor:
            cursor.execute(query, params)
        if row.get("cold_measured") is True and cold is not None:
            _persist_cold_split(row, cold)
        return True
    except Exception:
        logger.warning("benchmark metric persistence failed for %s/%s",
                       row.get("model"), row.get("quant"), exc_info=True)
        return False


def _persist_cold_split(row, cold):
    """A MEASURED cold load belongs to (model, quant, worker), whatever the
    allocation: stamp it + the transfer/load split on every row of the triple
    so the next run reads it back instead of resetting the worker again."""
    if not _cold_columns():
        return False
    from hugpy_server.app.routes.metrics_routes import _live_db
    transfer_bytes = row.get("transfer_bytes")
    with _live_db().cursor() as cursor:
        cursor.execute(
            "UPDATE model_metrics SET cold_load_s = %s, transfer_s = %s, transfer_bytes = %s, "
            "load_s = %s, bytes_per_s = %s, cold_measured_at = now() "
            "WHERE model_name = %s AND quant = %s AND worker = %s",
            (cold, _metric_number(row.get("transfer_s")),
             int(transfer_bytes) if isinstance(transfer_bytes, (int, float)) else None,
             _metric_number(row.get("load_s")), _metric_number(row.get("bytes_per_s")),
             str(row["model"]), str(row.get("quant") or ""), str(row["worker"])))
    return True


def _persist_benchmark_grade(row):
    """PHASE 2 (judge): update ONLY the grade columns of the collected result's
    existing model_metrics row — ``grade``, ``grade_detail`` (the full per-item
    history with the judge verdicts) and ``graded_at``. The load/throughput terms
    (cold/hot/tok_per_s/n_samples) were measured in phase 1 and are never
    re-stamped here, so the DB grade tracks the judge without disturbing the
    measurements. model_calls is append-only (its rows record the call as it ran)
    and is left untouched: the judge verdict is a later grading step, recorded on
    the rollup row's grade_detail."""
    if not isinstance(row, dict) or not row.get("model") or not row.get("worker"):
        return False
    if row.get("persist") is False:
        return False
    try:
        from hugpy_server.app.routes.metrics_routes import _live_db
        config = str(row.get("config") or "full")
        mode = str(row.get("alloc_mode") or "")
        alloc = mode if config in ("full", "standard") else "%s:%s" % (config, mode)
        valid_grade = (row.get("status") == "complete" and row.get("error") in (None, "N/A", ""))
        score, maximum = _metric_number(row.get("score")), _metric_number(row.get("max"))
        grade = 100.0 * score / maximum if valid_grade and score is not None and maximum else None
        if grade is None:
            return False
        detail = json.dumps(row.get("detail") or {}, sort_keys=True)
        with _live_db().cursor() as cursor:
            cursor.execute(
                "UPDATE model_metrics SET grade = %s, grade_detail = %s, graded_at = now(), "
                "updated_at = now() WHERE model_name = %s AND quant = %s AND alloc_mode = %s "
                "AND worker = %s",
                (grade, detail, str(row["model"]), str(row.get("quant") or ""), alloc,
                 str(row["worker"])))
        return True
    except Exception:
        logger.warning("benchmark grade update failed for %s/%s",
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
            if _BENCHMARK.get("status") in _BENCHMARK_ACTIVE:
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
        owner_alive = _benchmark_owner_alive(_BENCHMARK.get("owner"))
        orphaned = _benchmark_orphaned(_BENCHMARK)   # central restarted under it: resume below
        if not orphaned:
            if (_BENCHMARK.get("status") in _BENCHMARK_ACTIVE and not locally_alive
                    and not owner_alive and
                    _time.time() - float(_BENCHMARK.get("heartbeat") or 0) > 45):
                _BENCHMARK.update(status="interrupted", finished=_time.time(),
                                  error="executor heartbeat expired")
                _benchmark_updated()
    if orphaned:
        benchmark_resume_orphaned()
    return jsonify(_benchmark_public())


def _benchmark_done_keys(results):
    """result_key of every row recorded before the restart (a cancelled row is
    not done: that lane re-runs)."""
    from hugpy_curation.review.fleet_grading import result_key
    return {result_key(r) for r in results or []
            if isinstance(r, dict) and r.get("failure_class") != "cancelled"}


def _benchmark_inflight(state, done):
    """The lane that was mid-flight at the restart: the newest call / progress
    row whose key has no result yet."""
    from hugpy_curation.review.fleet_grading import result_key
    for row in list(reversed(state.get("calls") or [])) + [state.get("progress") or {}]:
        if isinstance(row, dict) and row.get("model") and row.get("worker"):
            key = result_key(row)
            if key not in done:
                return key
            return None
    return None


def _benchmark_client():
    """The loopback Client the benchmark thread uses (also for a post-restart
    restore that runs in a gunicorn worker, not the dead run thread)."""
    from hugpy_curation.review.fleet_grading import Client
    base = (os.environ.get("HUGPY_BENCHMARK_API") or "http://127.0.0.1:7002").rstrip("/")
    return Client(base, key=os.environ.get("HUGPY_BENCHMARK_API_KEY", ""),
                  operator_token=os.environ.get("HUGPY_OPERATOR_TOKEN", ""), timeout=60)


def _benchmark_workers(client, log=None):
    """The worker roster the benchmark thread runs against.

    Prefer central's OWN in-process data: a self-HTTP GET of /llm/workers right
    after a restart-resume (the run thread starts the moment /health answers) can
    block on the very gunicorn workers that would answer it, and that timeout
    used to kill the whole run (incident 2026-09-24). When the in-process read is
    unavailable, fall back to a BOUNDED-RETRY loopback fetch that rides out a
    warming central instead of dying on one blip."""
    try:
        from hugpy_server.app.routes.worker_routes import workers_payload
        rows = workers_payload()
        if isinstance(rows, list):
            return rows
        logger.info("in-process worker roster returned %s; using loopback fetch", type(rows).__name__)
    except Exception:  # noqa: BLE001 — fall back to the resilient loopback fetch
        logger.info("in-process worker roster unavailable; using loopback fetch", exc_info=True)
    from hugpy_curation.review.fleet_grading import retrying_control_request
    return retrying_control_request(client, "/llm/workers", "workers", log=log)


def _restore_persisted_state(reason):
    """Restore worker state from the run's PERSISTED snapshot, for a run that
    ended terminally (cancelled / interrupted) at a central restart — the thread
    whose try/finally would have restored it is gone. Idempotent: skips when
    there is no snapshot or a restore is already recorded. Records the restore
    report under the run's ``state`` and an event, exactly like the live path.
    Never raises."""
    import time as _time
    with _benchmark_lock():
        _benchmark_reload()
        st = _BENCHMARK.get("state") or {}
        snap = st.get("snapshot")
        if not snap or st.get("restore") is not None:
            return
    try:
        from hugpy_curation.review import worker_state
        report = worker_state.restore(_benchmark_client(), snap)
    except Exception as exc:  # noqa: BLE001 — restore must never break the resume election
        logger.exception("post-restart worker-state restore failed")
        report = {"workers": [], "error": f"restore raised: {type(exc).__name__}: {exc}"}
    with _benchmark_lock():
        _benchmark_reload()
        _BENCHMARK.setdefault("state", {})["restore"] = report
        _BENCHMARK.setdefault("events", []).append(
            {"at": _time.time(), "kind": "state-restore", "value": report,
             "message": f"worker state restored after {reason}"})
        _benchmark_updated()


def benchmark_resume_orphaned(reason="central restarted"):
    """Resume a hugpy-central run whose owning process died (a central restart:
    package promotion, crash). Elected by a non-blocking flock beside the state
    file so exactly one gunicorn worker resumes it. The run keeps its run_id;
    lanes already recorded are skipped (``done``); the lane in flight restarts
    from scratch with ``restart_reason``. Not resumed: an operator-cancelled run
    (``cancel_requested``) and a run already resumed ``_BENCHMARK_RESUME_MAX``
    times — both end with the reason recorded. Returns what it did or None."""
    import fcntl
    import time as _time
    try:
        lock = open(_BENCHMARK_STATE + ".resume.lock", "a")
    except OSError:
        return None
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return None                 # another worker is resuming it
        terminal = None
        with _benchmark_lock():
            _benchmark_reload()
            state = _BENCHMARK
            if not _benchmark_orphaned(state):
                return None
            now = _time.time()
            # Which phase was interrupted decides how it resumes: a run
            # interrupted while JUDGING resumes PHASE 2 only (re-judge the pending
            # outputs), never re-running collection.
            prev_status = state.get("status")
            interrupted_at = state.get("heartbeat") or state.get("updated")
            state["interrupted_at"] = interrupted_at
            if state.get("cancel_requested"):
                state.update(status="cancelled", finished=now, error=(
                    "cancelled by the operator before the central restart; not resumed"))
                _benchmark_updated()
                terminal = "cancelled"
            else:
                count = int(state.get("resume_count") or 0)
                if count >= _BENCHMARK_RESUME_MAX:
                    state.update(status="interrupted", finished=now, error=(
                        f"not resumed: interrupted by {count + 1} central restarts "
                        f"(max {_BENCHMARK_RESUME_MAX} resumes)"))
                    _benchmark_updated()
                    terminal = "interrupted"
        # A run that ends terminally without a resume still had its state backed
        # up; the process that would have restored it died at the restart, so
        # restore here (outside the lock — restore does network I/O) from the
        # persisted snapshot (operator ruling 2026-09-24: restore on cancel /
        # failure / exception / restart).
        if terminal is not None:
            _restore_persisted_state(terminal)
            return terminal
        with _benchmark_lock():
            _benchmark_reload()
            state = _BENCHMARK
            now = _time.time()
            params = dict(state.get("params") or {})
            params.setdefault("tokens", state.get("tokens") or 128)
            params.setdefault("models", state.get("models") or [])
            params.setdefault("workers", state.get("workers") or [])
            run_id = state.get("run_id")
            if prev_status == "judging":
                # PHASE 2 was interrupted: resume judging only. The collected
                # outputs are already persisted; the judge grades the ones still
                # pending and skips those already judged.
                resume_phase, done = "judge", None
                state.update(status="judging", resume_count=count + 1, heartbeat=now,
                             owner=_benchmark_owner(), finished=None, error=None,
                             resumed_from={"run_id": run_id, "interrupted_at": interrupted_at,
                                           "reason": reason, "resume": count + 1, "phase": "judge",
                                           "checkpoint": state.get("checkpoint")})
                state.setdefault("events", []).append(
                    {"at": now, "kind": "resume", "message": (
                        f"{reason}: resuming judging for run {run_id} — already-judged "
                        f"rows are skipped")})
            else:
                resume_phase = "collect"
                done = _benchmark_done_keys(state.get("results"))
                inflight = _benchmark_inflight(state, done)
                restarted = dict(state.get("restarted_lanes") or {})
                if inflight:
                    restarted["|".join(str(k) for k in inflight)] = _RESTART_REASON
                state.update(status="resuming", resume_count=count + 1, restarted_lanes=restarted,
                             heartbeat=now, owner=_benchmark_owner(), finished=None, error=None,
                             resumed_from={"run_id": run_id, "interrupted_at": interrupted_at,
                                           "reason": reason, "resume": count + 1,
                                           "completed_rows": len(done),
                                           "restarted_lane": list(inflight) if inflight else None,
                                           "checkpoint": state.get("checkpoint")})
                state.setdefault("events", []).append(
                    {"at": now, "kind": "resume", "message": (
                        f"{reason}: resuming run {run_id} — {len(done)} rows already "
                        f"recorded are skipped" + (f"; lane {'/'.join(str(k) for k in inflight)} "
                                                   f"{_RESTART_REASON}" if inflight else ""))})
            _benchmark_updated()
        _start_benchmark_thread(run_id, params, done=done, phase=resume_phase)
        logger.warning("benchmark %s resumed after a central restart (phase=%s, %d rows done)",
                       run_id, resume_phase, len(done or ()))
        return "resumed"
    finally:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock.close()


def start_benchmark_resume(base=None, wait_s=120.0):
    """Server-start hook (wsgi_app): once this server answers /health, resume a
    run a restart orphaned. Background + bounded; never raises."""
    import threading
    import time as _time
    import urllib.request

    def run():
        url = (base or os.environ.get("HUGPY_BENCHMARK_API") or "http://127.0.0.1:7002").rstrip("/")
        deadline = _time.monotonic() + wait_s
        while _time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url + "/health", timeout=5):
                    break
            except Exception:  # noqa: BLE001 — not up yet
                _time.sleep(2)
        try:
            benchmark_resume_orphaned()
        except Exception:  # noqa: BLE001
            logger.exception("benchmark resume on start failed")

    saved = _benchmark_read_state() or {}
    if saved.get("status") not in _BENCHMARK_ACTIVE or saved.get("executor") != "hugpy-central":
        return False
    threading.Thread(target=run, name="benchmark-resume", daemon=True).start()
    return True


@review_bp.route("/llm/benchmark/run", methods=["POST"])
def benchmark_run():
    """Start the HugPy-native per-worker model test from the console.

    The adapter makes ordinary central chat-completion calls with the existing
    per-request worker pin. HugPy remains solely responsible for selection,
    placement, provisioning, loading, dispatch, and eviction.

    Body: models, workers, tokens, suite, with_judge, budgets, resume, force,
    force_cold (re-measure cold loads even where one is recorded).
    """
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
    # Optional: force one grading suite (e.g. "hugpy-vision-v1" for VL GGUFs
    # whose primary_task is text-generation); omitted = per-model by task.
    suite = str(body["suite"]) if body.get("suite") else None
    with_judge = bool(body.get("with_judge"))
    # Bounded-run knobs (fleet_grading.DEFAULT_BUDGETS keys); resume skips
    # lanes graded within budgets.resume_hours unless force.
    budgets = body.get("budgets") if isinstance(body.get("budgets"), dict) else None
    resume, force = bool(body.get("resume")), bool(body.get("force"))
    force_cold = bool(body.get("force_cold"))
    if suite:
        try:
            from hugpy_curation.review.suites import suite_by_name
            suite_by_name(suite)
        except KeyError as exc:
            abort(400, description=str(exc).strip("'\""))
    # Everything a restarted central needs to continue this run (persisted).
    params = {"tokens": tokens, "models": model_ids, "workers": worker_ids, "suite": suite,
              "with_judge": with_judge, "budgets": budgets, "resume": resume, "force": force,
              "force_cold": force_cold}

    with _benchmark_lock():
        _benchmark_reload()
        if _BENCHMARK.get("status") in _BENCHMARK_ACTIVE:
            return jsonify(_benchmark_public()), 409
        run_id = uuid.uuid4().hex[:12]
        _BENCHMARK.clear()
        _BENCHMARK.update({"run_id": run_id, "status": "running",
                           "executor": executor, "started": _time.time(),
                           "finished": None, "tokens": tokens,
                           "heartbeat": _time.time(), "owner": _benchmark_owner(),
                           "params": params,
                           "models": model_ids, "workers": worker_ids,
                           "events": [], "results": [], "calls": [],
                           "summary": {},
                           "plan": {"rows": [], "total": 0, "runnable": 0},
                           "progress": {"completed": 0, "total": 0, "percent": 0},
                           "report": None})
        _benchmark_updated()
    _start_benchmark_thread(run_id, params)
    return jsonify(_benchmark_public()), 202


@review_bp.route("/llm/benchmark/judge", methods=["POST"])
def benchmark_judge():
    """Operator-triggered PHASE 2 for the CURRENT run (finished or partial):
    (re)judge every collected output still pending.

    Refuses (409) while a run is still collecting or judging, and when there is
    nothing collected to judge. The judge is the agent default brain, resolved by
    key and placed by central — never pinned, never a different judge; an item
    whose judge cannot be served stays unjudged with its recorded reason."""
    import time as _time
    with _benchmark_lock():
        _benchmark_reload()
        status = _BENCHMARK.get("status")
        if status in _BENCHMARK_ACTIVE:
            return jsonify({"status": status, "run_id": _BENCHMARK.get("run_id"),
                            "reason": f"run is {status}; judging is already scheduled"}), 409
        run_id = _BENCHMARK.get("run_id")
        results = [r for r in (_BENCHMARK.get("results") or []) if isinstance(r, dict)]
        if not run_id or not results:
            return jsonify({"status": status, "run_id": run_id,
                            "reason": "no collected run to judge"}), 409
        from hugpy_curation.review.fleet_grading import _judge_pending
        pending = len(_judge_pending(results))
        params = dict(_BENCHMARK.get("params") or {})
        _BENCHMARK.update(status="judging", finished=None, error=None,
                          heartbeat=_time.time(), owner=_benchmark_owner())
        _BENCHMARK.setdefault("events", []).append(
            {"at": _time.time(), "kind": "judge-now",
             "message": f"operator started judging {pending} pending item(s) for run {run_id}"})
        _benchmark_updated()
    _start_benchmark_thread(run_id, params, phase="judge")
    return jsonify(_benchmark_public()), 202


def _benchmark_failure_summary(results, plan, params, started_at=None):
    """Why a run produced no completed allocation, from its own rows: the
    failure classes with counts, the first recorded reason per class, and the
    plan size.  Structured (dict) so the console can render each part."""
    by_class = {}
    for r in results:
        cls = r.get("failure_class") or ("hardware_constraint" if r.get("status") == "hardware_constraint_failed" else None) or r.get("status") or "unknown"
        entry = by_class.setdefault(cls, {"count": 0, "models": [], "first_reason": None, "evidence": None})
        entry["count"] += 1
        if r.get("model") and r["model"] not in entry["models"]:
            entry["models"].append(r["model"])
        if entry["first_reason"] is None:
            entry["first_reason"] = r.get("reason") or r.get("error")
            entry["evidence"] = r.get("evidence") or None
    planned, runnable = plan.get("total", 0), plan.get("runnable", 0)
    requested = (params or {}).get("models") or []
    if not results and not planned:
        headline = (f"nothing was graded: {len(requested) or 'all'} model(s) requested, 0 lanes planned "
                    f"(no worker/model pair could be graded)")
    elif not results:
        headline = f"nothing was graded: {planned} lane(s) planned, {runnable} runnable, 0 rows recorded"
    else:
        parts = ", ".join(f"{v['count']} {k}" for k, v in by_class.items())
        headline = f"0 of {len(results)} allocation(s) completed: {parts}"
    import time as _time
    elapsed = round(_time.time() - started_at, 1) if isinstance(started_at, (int, float)) else None
    if elapsed is not None:
        headline += f" — {elapsed} s spent"
    return {"headline": headline, "planned": planned, "runnable": runnable, "requested_models": requested,
            "elapsed_s": elapsed, "failure_classes": by_class}


def _start_benchmark_thread(run_id, params, done=None, phase="collect"):
    """Run the benchmark ``run_id`` on a daemon thread of THIS process. Every
    report is persisted (``_benchmark_updated``) so a restart can resume from it.

    ``phase`` picks which half runs (operator ruling 2026-09-24, the two-phase
    split): ``collect`` runs PHASE 1 (collect+persist every raw output, deferring
    the judge) with ``done`` = result keys already recorded, then schedules PHASE
    2 automatically once the lot is done; ``judge`` runs PHASE 2 ONLY (judge the
    already-collected outputs) — used by the restart-resume of a run interrupted
    while judging, and by the operator's judge-now."""
    import threading
    import time as _time

    def report(kind, value):
        with _benchmark_lock():
            if _BENCHMARK.get("run_id") != run_id:
                return
            _BENCHMARK["heartbeat"] = _time.time()
            if _BENCHMARK.get("status") == "resuming":
                _BENCHMARK["status"] = "running"
            if kind == "result" and isinstance(value, dict):
                results = _BENCHMARK.setdefault("results", [])
                if _BENCHMARK.get("resumed_from"):
                    from hugpy_curation.review.fleet_grading import result_key
                    key = result_key(value)
                    why = (_BENCHMARK.get("restarted_lanes") or {}).get("|".join(str(k) for k in key))
                    if why:
                        value = {**value, "restart_reason": why}
                    # a re-run lane (cancelled at the restart) replaces its old row
                    results[:] = [r for r in results if not (isinstance(r, dict) and result_key(r) == key)]
                results.append(value)
                _persist_benchmark_result(value)
            elif kind == "call" and isinstance(value, dict):
                _BENCHMARK.setdefault("calls", []).append(value)
                _persist_benchmark_call(value)
            elif kind == "judge-result" and isinstance(value, dict):
                # PHASE 2: a re-graded result. Replace the collected row in place
                # (by result_key) and update ONLY its DB grade columns.
                from hugpy_curation.review.fleet_grading import result_key
                key = result_key(value)
                results = _BENCHMARK.setdefault("results", [])
                results[:] = [value if (isinstance(r, dict) and result_key(r) == key) else r
                              for r in results]
                _persist_benchmark_grade(value)
            elif kind == "judge-progress" and isinstance(value, dict):
                _BENCHMARK["judge_progress"] = value
            elif kind == "judge-summary" and isinstance(value, dict):
                _BENCHMARK["judge_summary"] = value
            elif kind == "plan" and isinstance(value, dict):
                _BENCHMARK["plan"] = value
                _BENCHMARK.setdefault("progress", {})["total"] = value.get("runnable", 0)
            elif kind == "progress" and isinstance(value, dict):
                _BENCHMARK["progress"] = value
            elif kind == "summary" and isinstance(value, dict):
                _BENCHMARK["summary"] = value
            elif kind in ("state-snapshot", "state-restore") and isinstance(value, dict):
                # Worker-state backup/restore rides its own top-level field so a
                # restarted central resumes from the SNAPSHOT (never re-snapshots
                # the already-mutated live state) and restores exactly once. Also
                # mirrored into events so the activity log shows it whole.
                _BENCHMARK.setdefault("state", {})[
                    "snapshot" if kind == "state-snapshot" else "restore"] = value
                _BENCHMARK.setdefault("events", []).append(
                    {"at": _time.time(), "kind": kind, "value": value})
            else:
                _BENCHMARK.setdefault("events", []).append(
                    {"at": _time.time(), "kind": kind, "value": value,
                     "message": value if isinstance(value, str) else None})
                # The agent reports the durable report path in its last notice.
                marker = "Capacity benchmark complete: "
                if isinstance(value, str) and value.startswith(marker):
                    _BENCHMARK["report"] = value[len(marker):]
            _benchmark_updated()

    def work():
        status, error = "failed", "benchmark did not run"
        client = snap = None
        try:
            from hugpy_curation.review.fleet_grading import (
                BenchmarkControl,
                Client,
                judge_collected,
                run_capacity_benchmark,
            )
            from hugpy_curation.review import worker_state
            base = (os.environ.get("HUGPY_BENCHMARK_API") or
                    "http://127.0.0.1:7002").rstrip("/")
            client = Client(base,
                            key=os.environ.get("HUGPY_BENCHMARK_API_KEY", ""),
                            operator_token=os.environ.get("HUGPY_OPERATOR_TOKEN", ""),
                            timeout=60)
            control = BenchmarkControl()
            with _benchmark_lock():
                _BENCHMARK["control"] = control
            if phase == "judge":
                # PHASE 2 ONLY (a judging run resumed after a restart, or the
                # operator's judge-now): grade the already-collected outputs. No
                # worker-state snapshot/restore — collection already restored it,
                # and the judge is placed by central's ordinary path.
                with _benchmark_lock():
                    collected = [r for r in _BENCHMARK.get("results", []) if isinstance(r, dict)]
                judge_collected(client, collected, report, stop=control)
                status, error = "complete", None
            else:
                # In-process first, then a bounded-retry loopback fetch: a transient
                # here (central warming up after the promotion that resumed this run)
                # must never kill the run — it retries, then records honestly.
                workers = _benchmark_workers(client, log=report)
                with _benchmark_lock():
                    # A resumed run restores from the SNAPSHOT persisted before the
                    # restart — never re-snapshot the state the interrupted run left
                    # mutated (operator ruling 2026-09-24).
                    snap = (_BENCHMARK.get("state") or {}).get("snapshot")
                if not snap:
                    snap = worker_state.snapshot(client, params.get("workers") or None,
                                                 params.get("models") or None)
                    report("state-snapshot", snap)
                extra = {}
                for key in ("suite", "budgets"):
                    if params.get(key):
                        extra[key] = params[key]
                for key in ("with_judge", "resume", "force", "force_cold"):
                    if params.get(key):
                        extra[key] = True
                if done:
                    extra["done"] = done
                try:
                    # PHASE 1 (collect): defer_judge=True — every raw output and the
                    # deterministic check are persisted as the run goes; the brain
                    # judge does NOT run inline.
                    run_capacity_benchmark(client, workers, params.get("tokens") or 128, control, report,
                                           model_ids=params.get("models") or None,
                                           worker_ids=params.get("workers") or None,
                                           cold_store=_ColdStore(), defer_judge=True, **extra)
                finally:
                    # Restore every involved worker's backed-up state after
                    # COLLECTION (completion, cancel, failure, exception) — before
                    # judging, so PHASE 2 grades with the models under test unloaded
                    # (no contention) and a restart during judging never leaks
                    # residency. A crash that kills the process before this runs is
                    # covered by the resume path, which restores from the snapshot.
                    if client is not None and snap is not None:
                        try:
                            report("state-restore", worker_state.restore(client, snap))
                        except Exception as exc2:  # noqa: BLE001 — restore must never mask the run
                            logger.exception("benchmark worker-state restore failed")
                            report("state-restore", {"workers": [],
                                                     "error": f"restore raised: {type(exc2).__name__}: {exc2}"})
                # Do not call an all-error/timeout run successful merely because
                # the worker loop returned. A model that answered normally should
                # have at least one completed allocation result.
                with _benchmark_lock():
                    all_results = [r for r in _BENCHMARK.get("results", []) if isinstance(r, dict)]
                    plan = dict(_BENCHMARK.get("plan") or {})
                completed_results = [r for r in all_results if r.get("status") == "complete"]
                if completed_results:
                    # PHASE 2 is scheduled automatically once the collection lot is
                    # done, unless the operator cancelled the run (its collected rows
                    # stay unjudged for a later judge-now).
                    if not control.is_set():
                        with _benchmark_lock():
                            if _BENCHMARK.get("run_id") == run_id:
                                _BENCHMARK["status"] = "judging"
                                _benchmark_updated()
                        judge_collected(client, all_results, report, stop=control)
                    status, error = "complete", None
                else:
                    # The reason is the recorded rows themselves, never a canned sentence.
                    with _benchmark_lock():
                        started_at = _BENCHMARK.get("started")
                    status, error = "failed", _benchmark_failure_summary(all_results, plan, params, started_at)
        except Exception as exc:  # background failure must remain visible
            logger.exception("capacity benchmark failed")
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        with _benchmark_lock():
            if _BENCHMARK.get("run_id") == run_id:
                _BENCHMARK.update(status=status, error=error, finished=_time.time())
                _benchmark_updated()

    thread = threading.Thread(target=work, name=f"benchmark-{run_id}", daemon=True)
    with _benchmark_lock():
        _BENCHMARK["thread"] = thread
    thread.start()
    return thread


@review_bp.route("/llm/benchmark/checkpoint", methods=["POST"])
def benchmark_checkpoint():
    """Persist the running benchmark NOW and mark the checkpoint (pkg_promote
    asks before it restarts central after the quiet-wait expired). The state
    file already holds every result row; this makes the intent explicit so
    the resumed run shows ``resumed_from.checkpoint``."""
    import time as _time
    body = request.get_json(silent=True) or {}
    with _benchmark_lock():
        _benchmark_reload()
        if _BENCHMARK.get("status") not in _BENCHMARK_ACTIVE:
            return jsonify({"checkpointed": False, "status": _BENCHMARK.get("status"),
                            "run_id": _BENCHMARK.get("run_id")})
        mark = {"at": _time.time(), "reason": str(body.get("reason") or "checkpoint requested"),
                "rows": len(_BENCHMARK.get("results") or []),
                "progress": _BENCHMARK.get("progress")}
        _BENCHMARK["checkpoint"] = mark
        _BENCHMARK.setdefault("events", []).append(
            {"at": mark["at"], "kind": "checkpoint", "message": "checkpoint: " + mark["reason"]})
        _benchmark_updated()
        return jsonify({"checkpointed": True, "run_id": _BENCHMARK.get("run_id"),
                        "status": _BENCHMARK.get("status"), "progress": _BENCHMARK.get("progress"),
                        "rows": mark["rows"], "state_file": _BENCHMARK_STATE})


@review_bp.route("/llm/benchmark/cancel", methods=["POST"])
def benchmark_cancel():
    """Cancel/skip one call, quant, model, worker, or the whole execution.
    An execution cancel is persisted (``cancel_requested``): a run cancelled
    before a central restart is never resumed by it."""
    import time as _time
    body = request.get_json(silent=True) or {}
    scope = body.get("scope", "execution")
    if scope not in {"call", "quant", "model", "worker", "execution"}:
        abort(400, description="scope must be call, quant, model, worker, or execution")
    with _benchmark_lock():
        control = _BENCHMARK.get("control")
        if _BENCHMARK.get("status") not in _BENCHMARK_ACTIVE or control is None:
            return jsonify({"status": "not_running"}), 409
        try:
            control.cancel(scope=scope, worker_id=body.get("worker_id"),
                           model=body.get("model"), quant=body.get("quant"),
                           call=body.get("call"))
        except ValueError as exc:
            abort(400, description=str(exc))
        if scope == "execution":
            _BENCHMARK.update(cancel_requested=True, cancelled_at=_time.time())
        _BENCHMARK.setdefault("events", []).append({"at": _time.time(),
            "kind": "cancel", "message": "cancelled %s" % scope,
            "scope": scope, "target": {k: body.get(k) for k in
                ("worker_id", "model", "quant", "call") if body.get(k)}})
        _benchmark_updated()
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
