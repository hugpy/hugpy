"""Test-fire HISTORY (operator 2026-10-02): every run and every call result
persisted, so a worker's past test fires stay viewable (the console's history
strip + popup) and survive central restarts — the job registry in
test_fire_routes is in-memory only.

Two tables in the hugpy DB:
  test_fire_runs     one row per job: worker, options, counts, start/finish, stop reason
  test_fire_results  one row per call: model, ok, latency, tokens, tok/s, error kind

``analyze()`` is the INFERENCE over that data (the calibrate idea applied to
test fires): per model on a worker, the baseline drawn from its stored ok
calls (median latency / tok/s), its success rate, its error kinds, and a
verdict for the latest run against that baseline — healthy / flaky / failing /
regressed / untested. Measured history is the oracle; the thresholds below are
the only judgement.

Fail-open everywhere (like central.presence): no DSN / no driver / a DB error
means the run still happens, it just is not remembered."""
from __future__ import annotations

import json
import logging
import os
import statistics
import threading
from typing import Optional

log = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS test_fire_runs (
    job_id       TEXT PRIMARY KEY,
    worker_id    TEXT NOT NULL,
    worker_name  TEXT,
    created      DOUBLE PRECISION NOT NULL,
    started      DOUBLE PRECISION,
    finished     DOUBLE PRECISION,
    rounds       INTEGER, concurrency INTEGER, max_tokens INTEGER,
    models       JSONB NOT NULL DEFAULT '[]'::jsonb,
    skipped      JSONB NOT NULL DEFAULT '[]'::jsonb,
    ok           INTEGER NOT NULL DEFAULT 0,
    failed       INTEGER NOT NULL DEFAULT 0,
    stop_reason  TEXT
);
CREATE INDEX IF NOT EXISTS test_fire_runs_worker ON test_fire_runs (worker_id, created DESC);
CREATE TABLE IF NOT EXISTS test_fire_results (
    id          BIGSERIAL PRIMARY KEY,
    job_id      TEXT NOT NULL,
    worker_id   TEXT NOT NULL,
    model_key   TEXT NOT NULL,
    started     DOUBLE PRECISION,
    ok          BOOLEAN NOT NULL,
    latency_s   DOUBLE PRECISION,
    tokens      INTEGER,
    tok_s       DOUBLE PRECISION,
    error       TEXT,
    error_kind  TEXT,
    served_by   TEXT,
    content80   TEXT,
    prompt      TEXT
);
CREATE INDEX IF NOT EXISTS test_fire_results_job ON test_fire_results (job_id, id);
CREATE INDEX IF NOT EXISTS test_fire_results_pair ON test_fire_results (worker_id, model_key, started DESC);
ALTER TABLE test_fire_runs ADD COLUMN IF NOT EXISTS round INTEGER;
ALTER TABLE test_fire_runs ADD COLUMN IF NOT EXISTS current_model TEXT;
ALTER TABLE test_fire_runs ADD COLUMN IF NOT EXISTS resumed_from TEXT;
"""

# verdict thresholds (analyze)
HEALTHY_RATE = 0.95          # success rate at/above which a model is healthy
FAILING_STREAK = 3           # this many latest calls all failed -> failing
REGRESS_TOK_S = 0.70         # latest run's median tok/s below 70% of baseline -> regressed
REGRESS_LATENCY = 1.50       # ... or median latency above 150% of baseline
MIN_BASELINE = 3             # ok calls needed before a baseline is drawn

_lock = threading.Lock()
_conn = None
_conn_pid = None
_ddl_done = False
_warned = False


def _dsn() -> Optional[str]:
    v = (os.environ.get("HUGPY_REGISTRY_PG_DSN") or "").strip()
    if v:
        return v
    try:
        from hugpy_engine.model_index.client import enabled, resolve_dsn
    except ImportError:
        return None
    return resolve_dsn() if enabled() else None


def _run(fn):
    """One short autocommit statement on this module's own connection
    (lock_timeout 5 s); None on any failure."""
    global _conn, _conn_pid, _ddl_done, _warned
    dsn = _dsn()
    if not dsn:
        return None
    with _lock:
        try:
            import psycopg
            if _conn is None or _conn.closed or _conn_pid != os.getpid():
                _conn = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
                _conn_pid = os.getpid()
                with _conn.cursor() as cur:
                    cur.execute("SET lock_timeout = '5s'")
            with _conn.cursor() as cur:
                if not _ddl_done:
                    cur.execute(DDL)
                    _ddl_done = True
                return fn(cur)
        except Exception as exc:  # noqa: BLE001
            if not _warned:
                log.warning("test-fire history unavailable (%s: %s) — runs are not remembered",
                            type(exc).__name__, exc)
                _warned = True
            try:
                if _conn is not None:
                    _conn.close()
            except Exception:  # noqa: BLE001
                pass
            _conn = None
            return None


def _rows(cur) -> list:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# ── writes (called by the job) ───────────────────────────────────────────────
def save_run(job) -> bool:
    """Upsert the run row from the live job (start, progress and finish)."""
    def op(cur):
        cur.execute(
            "INSERT INTO test_fire_runs (job_id, worker_id, worker_name, created, started, finished,"
            " rounds, concurrency, max_tokens, models, skipped, ok, failed, stop_reason,"
            " round, current_model, resumed_from)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s,%s)"
            " ON CONFLICT (job_id) DO UPDATE SET started=EXCLUDED.started, finished=EXCLUDED.finished,"
            " ok=EXCLUDED.ok, failed=EXCLUDED.failed, stop_reason=EXCLUDED.stop_reason,"
            " skipped=EXCLUDED.skipped, round=EXCLUDED.round, current_model=EXCLUDED.current_model",
            (job.job_id, job.worker_id, job.worker_name, job.created, job.started, job.finished,
             job.rounds, job.concurrency, job.max_tokens, json.dumps(list(job.models)),
             json.dumps(list(job.skipped), default=str), job.ok, job.failed, job.stop_reason,
             getattr(job, "round", None), getattr(job, "current_model", None),
             getattr(job, "resumed_from", None)))
        return True
    return bool(_run(op))


def save_result(job, result: dict) -> bool:
    def op(cur):
        cur.execute(
            "INSERT INTO test_fire_results (job_id, worker_id, model_key, started, ok, latency_s, tokens,"
            " tok_s, error, error_kind, served_by, content80, prompt)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (job.job_id, job.worker_id, str(result.get("model_key")), result.get("started"),
             bool(result.get("ok")), result.get("latency_s"), result.get("tokens"), result.get("tok_s"),
             result.get("error"), result.get("error_kind"), result.get("worker"),
             result.get("content80"), result.get("prompt")))
        return True
    return bool(_run(op))


# ── reads ────────────────────────────────────────────────────────────────────
def list_runs(worker_id: str, limit: int = 50) -> list:
    def op(cur):
        cur.execute("SELECT job_id, worker_id, worker_name, created, started, finished, rounds, concurrency,"
                    " max_tokens, models, skipped, ok, failed, stop_reason, round, current_model, resumed_from"
                    " FROM test_fire_runs"
                    " WHERE worker_id = %s ORDER BY created DESC LIMIT %s", (worker_id, int(limit)))
        return _rows(cur)
    return _run(op) or []


def get_run(job_id: str) -> Optional[dict]:
    def op(cur):
        cur.execute("SELECT * FROM test_fire_runs WHERE job_id = %s", (job_id,))
        runs = _rows(cur)
        if not runs:
            return None
        cur.execute("SELECT model_key, started, ok, latency_s, tokens, tok_s, error, error_kind,"
                    " served_by, content80, prompt FROM test_fire_results WHERE job_id = %s ORDER BY id",
                    (job_id,))
        return dict(runs[0], results=_rows(cur))
    return _run(op)


def resume_plan(run: dict, results: list) -> Optional[dict]:
    """PURE: how to pick an unfinished run back up (operator 2026-10-02: "a
    resume option that starts with the last model that was run if incomplete —
    start that model's run over and proceed"). Returns ``{first_order,
    rounds}`` — the rest of the interrupted round, the in-flight model first,
    then the rounds still owed (0 = until stopped) — or None when nothing is
    left. Assumes the round-robin order run_job uses (every model once per round)."""
    models = list(run.get("models") or [])
    if not models:
        return None
    n = len(models)
    done = len(results or [])
    rounds = int(run.get("rounds") or 0)
    full = done // n
    if rounds and full >= rounds:
        return None
    in_round = [r.get("model_key") for r in (results or [])[full * n:]]
    remaining = [m for m in models if m not in in_round]
    cur = run.get("current_model")
    if cur in remaining:
        remaining = [cur] + [m for m in remaining if m != cur]
    if not remaining:
        return None
    return {"first_order": remaining, "rounds": (0 if not rounds else rounds - full),
            "interrupted_model": cur, "completed_rounds": full}


def _pair_results(worker_id: str, limit_runs: int) -> "tuple[list, list]":
    def op(cur):
        cur.execute("SELECT job_id, created FROM test_fire_runs WHERE worker_id = %s"
                    " ORDER BY created DESC LIMIT %s", (worker_id, int(limit_runs)))
        runs = _rows(cur)
        if not runs:
            return runs, []
        cur.execute("SELECT job_id, model_key, started, ok, latency_s, tok_s, error_kind FROM test_fire_results"
                    " WHERE job_id = ANY(%s) ORDER BY started", ([r["job_id"] for r in runs],))
        return runs, _rows(cur)
    return _run(op) or ([], [])


def _median(xs):
    xs = [float(x) for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def verdict_for(calls: list, latest_job: Optional[str]) -> dict:
    """PURE: the per-model inference over its stored calls (oldest first).

    baseline = medians over the ok calls BEFORE the latest run (MIN_BASELINE
    needed); latest = medians over the latest run's ok calls. Verdict order:
    untested -> failing (last FAILING_STREAK all failed) -> regressed (latest
    slower than baseline past the thresholds) -> flaky (success < HEALTHY_RATE)
    -> healthy."""
    n = len(calls)
    oks = [c for c in calls if c.get("ok")]
    kinds: dict = {}
    for c in calls:
        if not c.get("ok"):
            k = c.get("error_kind") or "error"
            kinds[k] = kinds.get(k, 0) + 1
    prior_ok = [c for c in oks if c.get("job_id") != latest_job]
    latest_ok = [c for c in oks if c.get("job_id") == latest_job]
    base = None
    if len(prior_ok) >= MIN_BASELINE:
        base = {"latency_s": _median(c.get("latency_s") for c in prior_ok),
                "tok_s": _median(c.get("tok_s") for c in prior_ok), "samples": len(prior_ok)}
    latest = None
    if latest_ok:
        latest = {"latency_s": _median(c.get("latency_s") for c in latest_ok),
                  "tok_s": _median(c.get("tok_s") for c in latest_ok), "samples": len(latest_ok)}
    rate = (len(oks) / n) if n else None
    out = {"calls": n, "ok": len(oks), "failed": n - len(oks),
           "success_rate": (round(rate, 3) if rate is not None else None),
           "error_kinds": kinds, "baseline": base, "latest": latest,
           "last_ok_at": (oks[-1].get("started") if oks else None), "why": None}
    if not n:
        out.update(verdict="untested", why="no stored calls")
        return out
    tail = calls[-FAILING_STREAK:]
    if len(tail) >= min(FAILING_STREAK, n) and all(not c.get("ok") for c in tail):
        out.update(verdict="failing", why=f"last {len(tail)} call(s) failed"
                   + (f" ({max(kinds, key=kinds.get)})" if kinds else ""))
        return out
    if base and latest:
        slow_tok = (base["tok_s"] and latest["tok_s"] is not None
                    and latest["tok_s"] < base["tok_s"] * REGRESS_TOK_S)
        slow_lat = (base["latency_s"] and latest["latency_s"] is not None
                    and latest["latency_s"] > base["latency_s"] * REGRESS_LATENCY)
        if slow_tok or slow_lat:
            parts = []
            if slow_tok:
                parts.append(f"tok/s {latest['tok_s']} vs baseline {base['tok_s']}")
            if slow_lat:
                parts.append(f"latency {latest['latency_s']}s vs baseline {base['latency_s']}s")
            out.update(verdict="regressed", why="; ".join(parts))
            return out
    if rate is not None and rate < HEALTHY_RATE:
        out.update(verdict="flaky", why=f"{round(rate * 100)}% ok over {n} call(s)")
        return out
    out.update(verdict="healthy", why=f"{round((rate or 0) * 100)}% ok over {n} call(s)")
    return out


def analyze(worker_id: str, limit_runs: int = 20) -> dict:
    """Per-model inference over the worker's last ``limit_runs`` stored runs."""
    runs, results = _pair_results(worker_id, limit_runs)
    latest_job = runs[0]["job_id"] if runs else None
    by_model: dict = {}
    for r in results:
        by_model.setdefault(r["model_key"], []).append(r)
    models = {mk: verdict_for(calls, latest_job) for mk, calls in sorted(by_model.items())}
    counts: dict = {}
    for v in models.values():
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    return {"worker_id": worker_id, "runs": len(runs), "latest_job": latest_job,
            "models": models, "verdicts": counts,
            "thresholds": {"healthy_rate": HEALTHY_RATE, "failing_streak": FAILING_STREAK,
                           "regress_tok_s": REGRESS_TOK_S, "regress_latency": REGRESS_LATENCY,
                           "min_baseline": MIN_BASELINE}}
