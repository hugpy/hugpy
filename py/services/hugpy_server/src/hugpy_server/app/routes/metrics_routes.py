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
    GET /llm/compute-actions/<id>   ONE row, detail whole (+ its log text);
                                 ``?format=text`` -> the log as text/plain
    GET /llm/logs/<ref>, GET /llm/logs?ref=<ref>
                                 ANY log by its log_ref, WHOLE, as text/plain
                                 (``?format=json`` -> {log_ref, source, bytes,
                                 text}); ``?tail=N`` is the only way to get
                                 less than everything. Refs: compute_actions#N,
                                 admission:<job_id>, benchmark:events,
                                 request:<request_id> / memory:<pid>:<rid>
                                 (routing diagnostics), or a per-load log file
                                 path under PROJECTS_HOME/logs.
    GET /llm/benchmark/events   every benchmark event of the current run, whole
                                 (the status read keeps only the last 200)
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
import json
import os

from flask import Response, jsonify, request

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

    Query: ``limit`` (default 500, max 5000); optional ``model`` (one model,
    every alias form: ``owner~name`` and the bare tail) and ``suite``
    (``grade_suite``, e.g. ``integrity``) filters — additive, absent = all
    rows as before. Member-readable; never 500
    (empty ``rows`` + an ``error`` field on any fault — off/unreachable DB,
    missing table, import failure). This is the panel's real data source
    (t146); the sibling ``/llm/model-metrics`` above is the old, broken one."""
    import time
    try:
        limit = max(1, min(int(request.args.get("limit") or 500), 5000))
    except (TypeError, ValueError):
        limit = 500
    model = (request.args.get("model") or "").strip()
    suite = (request.args.get("suite") or "").strip()
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
            {where}
            ORDER BY updated_at DESC
            LIMIT %s
        """
        where, params = [], []
        if model:
            # same alias forms the grade writer uses (name_forms): the key as
            # given plus its owner-stripped tail; also any owner~<tail> form.
            tail = model.split("~")[-1]
            where.append("(model_name = ANY(%s) OR model_name LIKE %s)")
            params += [list(dict.fromkeys([model, tail])),
                       "%~" + tail.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")]
        if suite:
            where.append("grade_suite = %s")
            params.append(suite)
        query = query.replace("{where}", ("WHERE " + " AND ".join(where)) if where else "")
        with _live_db().cursor() as cur:
            cur.execute(query, (*params, limit))
            rows = [dict(zip(_LIVE_METRICS_COLS, r)) for r in cur.fetchall()]
        # THROUGHPUT (2026-09-23): the headline tok/s is the MEAN over every
        # recorded call (sum tokens / sum seconds) with n and spread, joined
        # onto each card; the card's own tok_per_s / tok_per_s_avg are the
        # placement EMAs and are NOT the model's tok/s.
        by_model = attach_throughput(rows, call_stats(model), call_stats_error(model))
        return jsonify({"rows": rows, "count": len(rows), "throughput_by_model": by_model,
                        "generated_at": time.time()})
    except Exception as exc:  # noqa: BLE001 — a read surface must not 500
        logger.warning("GET /llm/model-metrics2 failed: %s", exc, exc_info=True)
        try:
            _live_db().mark_unavailable(exc, "reading the live metrics sheet")
        except Exception:  # noqa: BLE001
            pass
        return jsonify({"rows": [], "count": 0, "generated_at": time.time(),
                        "error": str(exc)}), 200


NO_CALLS = {"n_calls": 0, "mean_tok_s": None}
_STATS_CACHE: dict = {}
# key -> the real fault text when the call-ledger read failed (so an empty
# result is reported as a failed read, never as "no calls").
_STATS_ERR: dict = {}


def no_calls(model=None, worker=None, quant="", alloc="", err=None) -> dict:
    """The explicit no-calls state, built from what was asked and what the
    ledger read returned: either the read fault, or the absence scoped to
    model / worker / quant / alloc."""
    out = dict(NO_CALLS)
    if err:
        out["reason"] = f"call ledger (model_calls) read failed: {err}"
        out["error"] = err
        return out
    scope = " on ".join(x for x in (str(model or "any model"), str(worker or "")) if x)
    cell = "/".join(x for x in (quant or "", alloc or "") if x)
    out["reason"] = f"no calls recorded in model_calls for {scope}" + (f" ({cell})" if cell else "")
    return out


def call_stats_error(model: str = ""):
    """The fault text of the last call_stats read for ``model`` ('' = all), or None."""
    return _STATS_ERR.get(model or "*")
_STATS_TTL_S = 8.0


def call_stats(model: str = "") -> list:
    """Throughput stats off the per-call ledger (model_calls): one row per
    (model, worker, quant, alloc) + a per-model rollup (worker None). Cached
    briefly; [] when the registry DB is off (every consumer then says "no
    calls recorded" rather than showing an EMA)."""
    import time
    key = model or "*"
    hit = _STATS_CACHE.get(key)
    if hit and time.monotonic() - hit[0] < _STATS_TTL_S:
        return hit[1]
    err = None
    try:
        from hugpy_engine.model_index import enabled, fetch_call_stats
        t0 = time.time()
        stats = fetch_call_stats(model) or []
        if not enabled():
            err = "registry DB disabled (HUGPY_REGISTRY_DB != pg)"
        elif not stats:
            # fetch_call_stats fails open to [] — tell a failed read from a
            # real absence by the fault it recorded during THIS read.
            from hugpy_engine.model_index import last_db_error
            le = last_db_error()
            if le and le.get("at", 0) >= t0:
                err = f"{le.get('error')} (while {le.get('doing')})"
    except Exception as exc:  # noqa: BLE001
        logger.warning("call stats read failed: %s", exc)
        stats, err = [], f"{type(exc).__name__}: {exc}"
    _STATS_ERR[key] = err
    _STATS_CACHE[key] = (time.monotonic(), stats)
    return stats


_TP_KEYS = ("n_calls", "n_rated", "mean_tok_s", "p50", "p90", "min", "max", "first_at",
            "last_at", "tokens", "seconds", "completion_tokens", "n_no_tokens",
            "n_no_window", "n_engine", "n_stream", "n_wall", "n_bench", "n_bench_twins",
            "n_no_quant", "n_no_alloc", "reason")


def _tp_why(st: dict) -> "str | None":
    """Why a stats block has no mean — from its own counts (never generic)."""
    if st.get("mean_tok_s") is not None:
        return None
    n, twins = int(st.get("n_calls") or 0), int(st.get("n_bench_twins") or 0)
    if not n and twins:
        return (f"{twins} graded call(s) in this cell are counted once, on their relay "
                f"row (the serving path's measurement) — see the worker / unstamped totals")
    return st.get("reason") or "no calls recorded"


def _tp(st: dict) -> dict:
    """One throughput block: Σ tokens-in-window / Σ generation seconds over
    every counted call, with n (all calls) and n_rated (calls that had a
    generation window), the spread, the basis counts and — when there is no
    mean — why. Model/worker blocks nest ``unstamped`` (and ``by_worker``)."""
    out = {k: st.get(k) for k in _TP_KEYS}
    n, nr = int(st.get("n_calls") or 0), int(st.get("n_rated") or 0)
    out["label"] = (f"Σtok/Σgen-s over {nr} of {n} call{'s' if n != 1 else ''}" if n
                    else "no calls counted")
    why = _tp_why(st)
    if why:
        out["reason"] = why
    if isinstance(st.get("unstamped"), dict):
        u = st["unstamped"]
        out["unstamped"] = {k: u.get(k) for k in _TP_KEYS}
        missing = [f"{u.get('n_no_quant') or 0} without quant", f"{u.get('n_no_alloc') or 0} without alloc"]
        out["unstamped"]["label"] = (f"+{u.get('n_calls') or 0} calls without quant/alloc stamp "
                                     f"({', '.join(missing)})") if u.get("n_calls") else u.get("reason")
    if isinstance(st.get("by_worker"), dict):
        out["by_worker"] = {w: _tp(ws) for w, ws in st["by_worker"].items()}
    return out


def _name_tail(name) -> str:
    return str(name or "").split("~")[-1]


def attach_throughput(rows: list, stats: list, err=None) -> dict:
    """Stamp ``throughput`` (Σtok/Σgen-s over all calls, or the explicit
    no-calls state with its reason) onto each metrics card; returns
    {model_name: rollup} where the rollup nests ``by_worker`` and the
    ``unstamped`` bucket (calls without a quant/alloc stamp — counted in the
    model and worker totals, never dropped). Model names match across alias
    forms (owner~name / name)."""
    cell, model = {}, {}
    for st in stats or []:
        if st.get("worker") is None:
            model[st.get("model_name")] = _tp(st)
        else:
            cell[(_name_tail(st.get("model_name")), st.get("worker"), st.get("quant") or "",
                  st.get("alloc_mode") or "")] = _tp(st)
    by_tail = {_name_tail(m): v for m, v in model.items()}
    for r in rows or []:
        k = (r.get("model_name"), r.get("worker"), r.get("quant") or "", r.get("alloc_mode") or "")
        hit = cell.get((_name_tail(k[0]),) + k[1:])
        if hit is None:
            hit = no_calls(*k, err=err)
            roll = by_tail.get(_name_tail(k[0]))
            if roll and not err:
                u = (roll.get("unstamped") or {})
                hit["reason"] += (f"; the model has {roll.get('n_calls') or 0} counted call(s)"
                                  + (f", {u.get('n_calls')} of them without a quant/alloc stamp"
                                     if u.get("n_calls") else ""))
        r["throughput"] = hit
    return model


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
        from hugpy_engine.model_index import last_db_error
        le = last_db_error() or {}
        return jsonify({"ok": False, "model": model, "grade": grade,
                        "error": (f"record_grade({model!r}, {grade}) did not write: "
                                  + (f"{le.get('error')} (while {le.get('doing')})" if le
                                     else "the model_index service returned False with no recorded DB fault"))}), 503
    logger.info("model-grade: %s = %.2f (%s)", model, grade, suite)
    return jsonify({"ok": True, "model": model, "grade": grade,
                    "suite": suite, "generated_at": time.time()})


@metrics_bp.route("/llm/diagnostics/<request_id>", methods=["GET"])
def routing_diagnostics_get(request_id):
    """The stored routing-refusal record for one request (2026-09-23): the
    structured object every /v1 refusal is rendered from. 404 when unknown."""
    from hugpy_engine.routing_diagnostics import lookup, miss_reason
    diag = lookup(request_id)
    if diag is None:
        return jsonify({"request_id": request_id, "found": False,
                        "error": f"no routing-refusal record for request {request_id!r}: "
                                 f"{miss_reason(request_id)}"}), 404
    return jsonify(diag)


@metrics_bp.route("/llm/models/<path:model_key>/failures", methods=["GET"])
def model_failures(model_key):
    """A model's failed loads + routing refusals, newest first (``?limit=``,
    default 50, max 5000). Rows are compute_actions rows (action=load
    outcome=fail / action=call outcome=refused) with their detail."""
    try:
        limit = max(1, min(5000, int(request.args.get("limit") or 50)))
    except (TypeError, ValueError):
        limit = 50
    try:
        from hugpy_fleet.central import model_metrics as _mm
        loads = _mm.recent_actions(_mm.model_metrics_store, limit=limit, action="load",
                                   model=model_key, outcome="fail")
        refusals = _mm.recent_actions(_mm.model_metrics_store, limit=limit, action="call",
                                      model=model_key, outcome="refused")
        rows = sorted(list(loads) + list(refusals),
                      key=lambda r: (r.get("ts") or 0, r.get("id") or 0), reverse=True)[:limit]
        return jsonify({"model": model_key, "failures": rows, "count": len(rows)})
    except Exception as exc:  # noqa: BLE001 — a read surface must not 500
        logger.warning("GET /llm/models/%s/failures failed: %s", model_key, exc, exc_info=True)
        return jsonify({"model": model_key, "failures": [], "count": 0, "error": str(exc)}), 200


@metrics_bp.route("/llm/compute-actions", methods=["GET"])
def compute_actions_log():
    """The durable compute-action log, NEWEST FIRST, bounded.

    Query: ``limit`` (default 200, max 5000), ``since`` (epoch-seconds floor),
    ``action`` (load|call|evict|provision|fit_fail|member_select),
    ``model``, ``worker`` (name or full card), ``outcome`` (e.g. ``fail``,
    ``loaded``). Member-readable; never 500
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
    outcome = (request.args.get("outcome") or "").strip() or None
    try:
        from hugpy_fleet.central import model_metrics as _mm
        # outcome (2026-09-23): e.g. ?action=load&outcome=fail&model=K lists a
        # model's failed loads with their load_failure detail. The module
        # helper filters on every backend (Postgres store included).
        rows = _mm.recent_actions(_mm.model_metrics_store, limit=limit,
                                  since_ts=since_ts, action=action, model=model,
                                  worker=worker, outcome=outcome)
        return jsonify({"actions": rows, "count": len(rows)})
    except Exception as exc:  # noqa: BLE001 — a read surface must not 500
        logger.warning("GET /llm/compute-actions failed: %s", exc, exc_info=True)
        return jsonify({"actions": [], "count": 0, "error": str(exc)}), 200


# ── grading-suite registry (2026-09-23) ──────────────────────────────────────
# The console must never hardcode a suite: which tasks a grade row covers, how
# many tiers each has and the score's denominator come from the suite that
# produced it. This read serves hugpy_curation.review.suites (plus the legacy
# 9-task single-tier suite old rows were recorded under) with each item's
# prompt and expected answer, so a PASS/FAIL marker can say what was asked and
# what counted as right. Member-readable; never 500.
LEGACY_SUITES = {
    "fleet-capacity-v1": {"task": "text-generation", "tiers": 1, "legacy": True,
                          "tasks": ["math", "wordprob", "factual", "format_primes", "logic",
                                    "exact_instruction", "coding", "json", "letters"]},
}


def _expected_of(checker) -> str:
    """The checker's stated rule, else the literal values its closure tests
    against (the text suite's lambdas), else an explicit 'not recorded'."""
    rule = getattr(checker, "rule", None)
    if rule:
        return str(rule)
    vals = []
    for cell in (getattr(checker, "__closure__", None) or ()):
        try:
            v = cell.cell_contents
        except ValueError:
            continue
        if isinstance(v, (str, int, float)) or (isinstance(v, (tuple, list)) and all(
                isinstance(x, (str, int, float)) for x in v)):
            vals.append(repr(v))
        elif callable(v):
            inner = _expected_of(v)
            if inner and not inner.startswith("("):
                vals.append(inner)
    return ("checks " + ", ".join(vals)) if vals else "(checker states no rule)"


def _prompt_of(spec) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        text = spec.get("text") or spec.get("prompt") or ""
        extra = " + a generated test image" if spec.get("image") is not None else ""
        subject = f" (subject: {spec['subject']})" if spec.get("subject") else ""
        return f"{text}{subject}{extra}"
    return str(spec)


def suite_registry() -> list:
    out = []
    try:
        from hugpy_curation.review.suites import SUITES, SUITES_BY_NAME
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"suite registry unavailable: {exc}"}]
    by_task = {}
    for task, s in SUITES.items():
        by_task.setdefault(s.name, []).append(task)
    for name, s in SUITES_BY_NAME.items():
        entry = {"name": name, "task": s.grade_task, "serves_tasks": by_task.get(name, []),
                 "tiers": 3, "legacy": False, "tasks": [], "items": {}}
        try:
            for tname, tiers in s.tasks.items():
                entry["tasks"].append(tname)
                entry["items"][tname] = [{"tier": t[0], "prompt": _prompt_of(t[1]),
                                          "expected": _expected_of(t[2]),
                                          "expected_answer": getattr(t[2], "answer", None)} for t in tiers]
            entry["tiers"] = max((len(v) for v in entry["items"].values()), default=3)
            entry["max"] = sum(len(v) for v in entry["items"].values())
        except Exception as exc:  # noqa: BLE001 — a suite whose module fails still lists
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["max"] = None
        out.append(entry)
    for name, meta in LEGACY_SUITES.items():
        out.append({"name": name, **meta, "max": meta["tiers"] * len(meta["tasks"]), "items": {}})
    return out


@metrics_bp.route("/llm/benchmark/suites", methods=["GET"])
def benchmark_suites():
    """Every grading suite: name, task, ordered tasks, tiers, max, and per item
    the prompt + expected answer. The console derives matrix columns and score
    denominators from this — never from constants."""
    import time
    try:
        return jsonify({"suites": suite_registry(), "generated_at": time.time()})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"suites": [], "error": str(exc), "generated_at": time.time()}), 200


# ── whole logs (2026-09-23) ──────────────────────────────────────────────────
# The operator's rule: "everything that says logs needs to be logs. not
# paraphrasing, canned feedback, truncated text." These reads relay a log
# WHOLE, from its actual source; ``?tail=N`` is the only way to get less.
def _tail(text: str) -> str:
    raw = request.args.get("tail")
    if raw in (None, ""):
        return text
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return text
    return "\n".join(text.split("\n")[-n:]) if n > 0 else text


def _log_reply(ref: str, source: str, text, extra: dict = None, status: int = 200):
    """text/plain by default, JSON on ``?format=json``. Empty logs say what
    was read and that it held zero bytes — never a blank."""
    text = "" if text is None else str(text)
    full_bytes = len(text.encode("utf-8"))
    body = _tail(text)
    fmt = (request.args.get("format") or "").strip().lower()
    if fmt == "json":
        out = {"log_ref": ref, "source": source, "bytes": full_bytes, "text": body}
        if request.args.get("tail"):
            out["tail"] = request.args.get("tail")
        if extra:
            out.update(extra)
        return jsonify(out), status
    if not text:
        body = f"(empty log: read {source} for {ref}; bytes=0)\n"
    return Response(body, status=status, mimetype="text/plain",
                    headers={"X-Log-Ref": ref.encode("ascii", "replace").decode("ascii"),
                             "X-Log-Source": source.encode("ascii", "replace").decode("ascii"),
                             "X-Log-Bytes": str(full_bytes)})


def _action_log_text(row: dict) -> tuple:
    """``(source, text)`` — the log a compute_actions row carries, whole: the
    loader stderr, else the error message, else the stored record itself."""
    d = row.get("detail")
    rid = row.get("id")
    if isinstance(d, dict):
        for k in ("loader_stderr", "message", "error", "reason"):
            if d.get(k):
                return f"compute_actions#{rid} detail.{k}", str(d[k])
        return f"compute_actions#{rid} detail (whole record)", json.dumps(d, indent=2, default=str)
    raw = row.get("detail_json")
    if raw:
        return f"compute_actions#{rid} detail_json (unparsed)", str(raw)
    return f"compute_actions#{rid} detail_json", ""


@metrics_bp.route("/llm/compute-actions/<int:action_id>", methods=["GET"])
def compute_action_one(action_id):
    """ONE compute_actions row, detail whole, plus the log it carries
    ({log_ref, text_source, bytes, text}). ``?format=text`` -> text/plain."""
    try:
        from hugpy_fleet.central import model_metrics as _mm
        row = _mm.get_action(action_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("GET /llm/compute-actions/%s failed: %s", action_id, exc, exc_info=True)
        return jsonify({"id": action_id, "found": False,
                        "error": f"{type(exc).__name__}: {exc}"}), 200
    ref = f"compute_actions#{action_id}"
    if row is None:
        return jsonify({"id": action_id, "found": False, "log_ref": ref,
                        "error": f"no compute_actions row with id {action_id}"}), 404
    src, text = _action_log_text(row)
    d = row.get("detail") if isinstance(row.get("detail"), dict) else {}
    if (request.args.get("format") or "").strip().lower() == "text":
        return _log_reply(ref, src, text)
    return jsonify({**row, "found": True, "log_ref": ref,
                    "file_log_ref": d.get("log_ref"), "text_source": src,
                    "bytes": len(text.encode("utf-8")), "text": _tail(text)})


def _benchmark_events(with_run: bool = False):
    from hugpy_server.app.routes import review_routes as _rv
    with _rv._benchmark_lock():
        _rv._benchmark_reload()
        events = list(_rv._BENCHMARK.get("events") or [])
        run_id = _rv._BENCHMARK.get("run_id")
    return (events, run_id) if with_run else events


def _events_text(events: list) -> str:
    import time as _time
    lines = []
    for e in events:
        at = e.get("at")
        stamp = (_time.strftime("%Y-%m-%dT%H:%M:%S", _time.localtime(at))
                 if isinstance(at, (int, float)) else str(at))
        msg = e.get("message")
        if msg is None:
            msg = json.dumps(e.get("value"), default=str)
        lines.append(f"{stamp} [{e.get('kind')}] {msg}")
    return "\n".join(lines)


@metrics_bp.route("/llm/benchmark/events", methods=["GET"])
def benchmark_events():
    """Every event of the current/last benchmark run, whole (the status read
    trims to the last 200). ``?tail=N`` for the last N; ``?format=text``."""
    try:
        events, run_id = _benchmark_events(with_run=True)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"events": [], "count": 0, "error": f"{type(exc).__name__}: {exc}"}), 200
    if (request.args.get("format") or "").strip().lower() == "text":
        return _log_reply("benchmark:events", "benchmark state events[]", _events_text(events))
    total = len(events)
    raw = request.args.get("tail")
    try:
        n = int(raw) if raw not in (None, "") else 0
    except (TypeError, ValueError):
        n = 0
    return jsonify({"run_id": run_id, "events": events[-n:] if n > 0 else events, "count": total})


def _file_log(ref: str):
    """A per-load log file under PROJECTS_HOME/logs, read whole. Returns
    ``(source, text)`` or None when it is not a local, readable file there."""
    try:
        from hugpy_platform.constants import PROJECTS_HOME
    except Exception:  # noqa: BLE001
        PROJECTS_HOME = os.environ.get("PROJECTS_HOME")
    if not PROJECTS_HOME:
        return None
    root = os.path.realpath(os.path.join(str(PROJECTS_HOME), "logs"))
    real = os.path.realpath(ref)
    if not (real == root or real.startswith(root + os.sep)) or not os.path.isfile(real):
        return None
    with open(real, "r", encoding="utf-8", errors="replace") as fh:
        return f"file {real}", fh.read()


def _resolve_log(ref: str):
    """``(source, text, extra, status)`` for any log_ref, whole."""
    ref = (ref or "").strip()
    low = ref.lower()
    # compute_actions#N / compute_actions:N / bare N
    for pfx in ("compute_actions#", "compute_actions:", "compute-actions:", "ca:"):
        if low.startswith(pfx):
            ref = ref[len(pfx):]
            low = ref
            break
    else:
        pfx = None
    if pfx is not None or ref.isdigit():
        from hugpy_fleet.central import model_metrics as _mm
        row = _mm.get_action(ref)
        if row is None:
            return (f"compute_actions#{ref}", "", {"found": False,
                    "error": f"no compute_actions row with id {ref}"}, 404)
        src, text = _action_log_text(row)
        d = row.get("detail") if isinstance(row.get("detail"), dict) else {}
        return src, text, {"found": True, "row": row, "file_log_ref": d.get("log_ref")}, 200
    if low.startswith("admission:"):
        from hugpy_storage.admission import admission_queue
        jid = ref.split(":", 1)[1]
        job = admission_queue.get(jid)
        if job is None:
            return (f"admission_jobs id={jid}", "", {"found": False,
                    "error": f"no admission job {jid!r}"}, 404)
        return (f"admission_jobs.log id={jid}", job.get("log") or "",
                {"found": True, "job": {k: v for k, v in job.items() if k != "log"}}, 200)
    if low.startswith("benchmark:"):
        return "benchmark state events[]", _events_text(_benchmark_events()), {"found": True}, 200
    if low.startswith("request:") or low.startswith("memory:"):
        from hugpy_engine.routing_diagnostics import lookup, miss_reason
        rid = ref.split(":", 1)[1] if low.startswith("request:") else ref.split(":", 2)[-1]
        diag = lookup(rid)
        if diag is None:
            return (f"routing diagnostics request={rid}", "", {"found": False,
                    "error": miss_reason(rid)}, 404)
        return (f"routing diagnostics request={rid}", json.dumps(diag, indent=2, default=str),
                {"found": True}, 200)
    # a per-load log file (slot agent): local file first, else the load-fail
    # row that carried its whole text to central
    got = _file_log(ref)
    if got is not None:
        return got[0], got[1], {"found": True}, 200
    from hugpy_fleet.central import model_metrics as _mm
    for r in _mm.recent_actions(_mm.model_metrics_store, limit=5000, action="load",
                                outcome="fail"):
        d = r.get("detail") if isinstance(r.get("detail"), dict) else {}
        if d.get("log_ref") == ref:
            src = (f"compute_actions#{r.get('id')} detail.loader_stderr "
                   f"(the file lives on worker {str(r.get('worker_card') or '?').split(':')[0]})")
            return src, d.get("loader_stderr") or "", {"found": True, "row": r}, 200
    return (ref, "", {"found": False,
            "error": f"unknown log_ref {ref!r}: not a compute_actions id, admission:/"
                     f"benchmark:/request: ref, a readable file under PROJECTS_HOME/logs, "
                     f"nor the log_ref of any of the last 5000 load-fail rows"}, 404)


@metrics_bp.route("/llm/logs", methods=["GET"])
@metrics_bp.route("/llm/logs/<path:ref>", methods=["GET"])
def log_by_ref(ref=None):
    """ANY log by its log_ref, WHOLE (see the module docstring)."""
    ref = ref or request.args.get("ref") or ""
    if not ref:
        return jsonify({"error": "pass a log_ref: /llm/logs/<ref> or /llm/logs?ref=<ref>"}), 400
    try:
        src, text, extra, status = _resolve_log(ref)
    except Exception as exc:  # noqa: BLE001 — a read surface must not 500
        logger.warning("GET /llm/logs %s failed: %s", ref, exc, exc_info=True)
        src, text, extra, status = (ref, "", {"found": False,
                                              "error": f"{type(exc).__name__}: {exc}"}, 200)
    if status != 200 and (request.args.get("format") or "").strip().lower() != "json":
        return Response(f"{extra.get('error')}\n", status=status, mimetype="text/plain")
    return _log_reply(ref, src, text, extra, status)
