"""COMPUTE-ACTIVITY METRICS — the Metrics panel's read surfaces.

    GET /llm/model-metrics      the OLD EMA snapshot (load_metrics + call_metrics)
                                 — db-toolserver PgMetricsStore. BROKEN (role
                                 lacks CREATE) and the wrong/old schema; kept
                                 only so this route doesn't disappear out from
                                 under any lingering caller. Do not build on it.
    GET /llm/model-metrics2     the LIVE per-(model x quant x alloc_mode x
                                 worker) card, straight off db-hugpy's
                                 ``model_metrics`` table (t146). THIS is what
                                 the console panel reads now.
    GET /llm/compute-actions    the durable, append-only compute-action log
    POST /llm/model-grade       (t147) persist an APTITUDE grade onto a
                                 model's ``model_metrics`` rows so it shows in
                                 #metrics, not only in a grader's JSON/markdown.
                                 OPERATOR-ONLY (operator_auth._SENSITIVE) — the
                                 one mutation on this blueprint.

The three GETs are MEMBER-READABLE — the console reads them exactly like every
other panel read (GET /llm/evictions, GET /llm/model-groups).

NEVER 500. A store read that faults reports an EMPTY payload with the error
attached (same posture as GET /llm/model-groups): a broken metrics PAGE must
not look like a broken metrics FEATURE, and it certainly must not surface as a
worker/console-visible error. Reads are cheap and idempotent, so the worst a
fault can do is show the operator "no data yet, here's why". Faults ARE
logged (warning), never silently swallowed.
"""
from flask import jsonify, request

from abstract_flask import get_bp

metrics_bp, logger = get_bp("metrics_bp", __name__)

# Lazy, process-cached handle onto the model-index registry DB (db-hugpy).
# ``imports/src/model_index/`` is solcatcher-owned (REGISTRY-DB-INDEX.md,
# 2026-09-10, not writable by the hugpy service user) — this route only READS
# the ``model_metrics`` table their schema already defines, through their own
# ``DatabaseClient`` (fork-guarded, autocommit, one connection per PID) rather
# than opening a second ad hoc connection per request. The SELECT is inlined
# here (not added to their ``query_registry.py``) for the same ownership
# reason; the column list matches ``MetricsQueries.CREATE_TABLE`` verbatim.
_LIVE_METRICS_COLS = ("model_name", "quant", "alloc_mode", "worker",
                     "cold_load_s", "hot_load_s", "tok_per_s", "tok_per_s_avg",
                     "upload_time_s", "media_bytes", "n_samples",
                     "temperature", "task", "updated_at",
                     "grade", "grade_suite", "grade_detail", "graded_at")
_live_metrics_db = None


def _live_db():
    """The cached ``DatabaseClient`` for this worker process, or ``None`` if
    the model-index client can't even be imported (old wheel, missing dep)."""
    global _live_metrics_db
    if _live_metrics_db is None:
        from hugpy_engine.model_index.client import DatabaseClient
        _live_metrics_db = DatabaseClient()
    return _live_metrics_db


@metrics_bp.route("/llm/model-metrics", methods=["GET"])
def model_metrics_snapshot():
    """The EMA snapshot the summary charts draw from.

    ``{load_metrics: [...], call_metrics: [...], generated_at}`` — every
    per-(model, variant, worker_card, temperature) load row and every per-model
    call row, straight off the store. Member-readable; never 500 (empty lists +
    an ``error`` field on any store fault)."""
    import time
    try:
        from hugpy_fleet.central.model_metrics import model_metrics_store as store
        load_rows, call_rows = _read_ema(store)
        try:
            by_task = store.all_calls_by_task()
        except Exception:  # noqa: BLE001 — additive; never fail the snapshot
            by_task = []
        return jsonify({"load_metrics": load_rows, "call_metrics": call_rows,
                        "calls_by_task": by_task, "generated_at": time.time()})
    except Exception as exc:  # noqa: BLE001 — a read surface must not 500
        logger.warning("GET /llm/model-metrics failed: %s", exc, exc_info=True)
        return jsonify({"load_metrics": [], "call_metrics": [],
                        "calls_by_task": [],
                        "generated_at": time.time(), "error": str(exc)}), 200


def _read_ema(store):
    """The two EMA tables as row-dicts, via the store's backend-agnostic dump
    methods (``all_loads``/``all_calls``).

    Both the SQLite ``ModelMetricsStore`` and the Postgres ``PgMetricsStore``
    (the DB-arm cutover, 2026-09-01) expose these method-compatibly and each
    returns ``[]`` on any fault — so this never pokes ``_connect``/``_ensure``
    (which only existed on the SQLite store and left the panel empty on PG).
    ``([], [])`` if the store is disabled or unreachable — same fail-open."""
    try:
        return store.all_loads(), store.all_calls()
    except Exception:  # noqa: BLE001 — a read surface must not fault the snapshot
        return [], []


@metrics_bp.route("/llm/model-metrics2", methods=["GET"])
def model_metrics_live():
    """The LIVE fleet metrics sheet — every ``model_metrics`` row (one per
    model x quant x alloc_mode x worker) off db-hugpy, newest-updated first.

    Query: ``limit`` (default 500, max 5000). Member-readable; never 500
    (empty ``rows`` + an ``error`` field on any fault — off/unreachable DB,
    missing table, import failure). This is the panel's real data source
    (t146); the sibling ``/llm/model-metrics`` above is the old, broken one."""
    import time
    try:
        limit = max(1, min(int(request.args.get("limit") or 500), 5000))
    except (TypeError, ValueError):
        limit = 500
    try:
        from hugpy_engine.model_index.client import enabled
        if not enabled():
            return jsonify({"rows": [], "count": 0, "generated_at": time.time(),
                            "error": "registry DB disabled "
                                     "(HUGPY_REGISTRY_DB != pg)"}), 200
        query = """
            SELECT model_name, quant, alloc_mode, worker, cold_load_s,
                   hot_load_s, tok_per_s, tok_per_s_avg, upload_time_s,
                   media_bytes, n_samples, temperature, task,
                   extract(epoch from updated_at),
                   grade, grade_suite, grade_detail, extract(epoch from graded_at)
            FROM model_metrics
            ORDER BY updated_at DESC
            LIMIT %s
        """
        with _live_db().cursor() as cur:
            cur.execute(query, (limit,))
            rows = [dict(zip(_LIVE_METRICS_COLS, r)) for r in cur.fetchall()]
        return jsonify({"rows": rows, "count": len(rows),
                        "generated_at": time.time()})
    except Exception as exc:  # noqa: BLE001 — a read surface must not 500
        logger.warning("GET /llm/model-metrics2 failed: %s", exc, exc_info=True)
        try:
            _live_db().mark_unavailable(exc, "reading the live metrics sheet")
        except Exception:  # noqa: BLE001
            pass
        return jsonify({"rows": [], "count": 0, "generated_at": time.time(),
                        "error": str(exc)}), 200


def _grade_from_body(body: dict):
    """(model, grade 0..100, suite, detail, quant, worker) from a POST body.

    Accepts either an explicit ``grade`` (0..100) or the station
    ``model_grader.py`` JSON verbatim (``{model, score, total, pct, results}``
    — so ``curl -d @gradings/<model>.json`` persists a grading as-is).
    Raises ValueError with a caller-facing message on a bad body."""
    if not isinstance(body, dict):
        raise ValueError("JSON object body required")
    model = (body.get("model") or body.get("model_key")
             or body.get("model_name") or "").strip()
    if not model:
        raise ValueError("model is required")
    grade = body.get("grade")
    if grade is None:
        grade = body.get("pct")
    if grade is None and body.get("total"):
        try:
            grade = 100.0 * float(body.get("score") or 0) / float(body["total"])
        except (TypeError, ValueError, ZeroDivisionError):
            grade = None
    try:
        grade = float(grade)
    except (TypeError, ValueError):
        raise ValueError("grade (0..100) or score/total is required")
    if not (0.0 <= grade <= 100.0):
        raise ValueError("grade must be within 0..100")
    suite = body.get("suite") or body.get("grade_suite") or "text-aptitude"
    detail = body.get("detail")
    if detail is None and isinstance(body.get("results"), list):
        # the grader's per-task rows, kept compact: {task: ok}
        detail = {str(r.get("task")): bool(r.get("ok"))
                  for r in body["results"] if isinstance(r, dict)}
    return (model, round(grade, 2), str(suite), detail,
            str(body.get("quant") or ""), str(body.get("worker") or ""))


@metrics_bp.route("/llm/model-grade", methods=["POST"])
def model_grade_record():
    """Persist one model's APTITUDE grade onto its live ``model_metrics``
    rows (t147). Operator-only (``_SENSITIVE``).

    Body: ``{model, grade}`` (+ optional ``suite``, ``detail``, ``quant``,
    ``worker``) or the station grader's JSON as-is. 400 on a bad body, 503
    when the registry DB is off/unreachable (a write that did not happen must
    not report 200 — the grader's own file is then the only record)."""
    import time
    try:
        model, grade, suite, detail, quant, worker = _grade_from_body(
            request.get_json(silent=True))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    try:
        from hugpy_engine.model_index import enabled, record_grade
        if not enabled():
            return jsonify({"ok": False, "model": model, "grade": grade,
                            "error": "registry DB disabled "
                                     "(HUGPY_REGISTRY_DB != pg)"}), 503
        ok = bool(record_grade(model, grade, suite=suite, detail=detail,
                               quant=quant, worker=worker))
    except Exception as exc:  # noqa: BLE001 — report, never 500
        logger.warning("POST /llm/model-grade failed for %s: %s", model, exc,
                       exc_info=True)
        return jsonify({"ok": False, "model": model, "grade": grade,
                        "error": str(exc)}), 503
    if not ok:
        return jsonify({"ok": False, "model": model, "grade": grade,
                        "error": "registry DB unavailable"}), 503
    logger.info("model-grade: %s = %.2f (%s)", model, grade, suite)
    return jsonify({"ok": True, "model": model, "grade": grade,
                    "suite": suite, "generated_at": time.time()})


@metrics_bp.route("/llm/compute-actions", methods=["GET"])
def compute_actions_log():
    """The durable compute-action log, NEWEST FIRST, bounded.

    Query: ``limit`` (default 200, max 5000), ``since`` (epoch-seconds floor),
    ``action`` (load|call|evict|provision|fit_fail|member_select),
    ``model``, ``worker`` (name or full card). Member-readable; never 500
    (empty list + ``error`` on a store fault)."""
    try:
        limit = int(request.args.get("limit") or 200)
    except (TypeError, ValueError):
        limit = 200
    since = request.args.get("since")
    try:
        since_ts = float(since) if since not in (None, "") else None
    except (TypeError, ValueError):
        since_ts = None
    action = (request.args.get("action") or "").strip() or None
    model = (request.args.get("model") or "").strip() or None
    worker = (request.args.get("worker") or "").strip() or None
    try:
        from hugpy_fleet.central.model_metrics import model_metrics_store as store
        rows = store.recent_actions(limit=limit, since_ts=since_ts,
                                    action=action, model=model, worker=worker)
        return jsonify({"actions": rows, "count": len(rows)})
    except Exception as exc:  # noqa: BLE001 — a read surface must not 500
        logger.warning("GET /llm/compute-actions failed: %s", exc, exc_info=True)
        return jsonify({"actions": [], "count": 0, "error": str(exc)}), 200
