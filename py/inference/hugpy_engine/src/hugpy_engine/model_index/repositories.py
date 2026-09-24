"""Repositories — one class per table, one explicit method per QueryRegistry
operation (explicit-wiring ruling 2026-09-10, modeled on solcatcher's
repos/*/repository classes). No SQL here; it all lives in query_registry.

Repositories are THIN: parameters in, rows out. Fallbacks, locking and the
enabled() gate belong to the service; connections to the DatabaseClient.
"""
from __future__ import annotations

import json
from typing import Optional

from hugpy_engine.model_index.client import DatabaseClient
from hugpy_engine.model_index.query_registry import (
    CallQueries,
    DiscoveryQueries,
    MetricsQueries,
    QuantQueries,
)


class DiscoveryRepository:
    def __init__(self, db: DatabaseClient):
        self.db = db

    def create_table(self, cur) -> None:
        cur.execute(DiscoveryQueries.CREATE_TABLE)
        for q in DiscoveryQueries.CREATE_INDEXES:
            cur.execute(q)

    def delete_absent(self, cur, names: list) -> None:
        cur.execute(DiscoveryQueries.DELETE_ABSENT, (names,))

    def upsert_row(self, cur, name: str, row: dict) -> None:
        cur.execute(DiscoveryQueries.UPSERT_ROW,
                    (name, json.dumps(row), row.get("hub_id"),
                     row.get("framework")))

    def fetch_all(self, cur) -> dict:
        cur.execute(DiscoveryQueries.FETCH_ALL)
        return {name: row for name, row in cur.fetchall()}


class QuantsRepository:
    def __init__(self, db: DatabaseClient):
        self.db = db

    def create_table(self, cur) -> None:
        cur.execute(QuantQueries.CREATE_TABLE)
        for q in QuantQueries.CREATE_INDEXES:
            cur.execute(q)

    def delete_absent_models(self, cur, names: list) -> None:
        cur.execute(QuantQueries.DELETE_ABSENT_MODELS, (names,))

    def replace_for_model(self, cur, name: str, quants) -> None:
        cur.execute(QuantQueries.DELETE_FOR_MODEL, (name,))
        if not isinstance(quants, list):
            return
        for q in quants:
            if not isinstance(q, dict) or not q.get("file"):
                continue
            cur.execute(QuantQueries.UPSERT_VARIANT,
                        (name, q.get("file"), q.get("quant"),
                         q.get("bytes"), q.get("shards")))


class MetricsRepository:
    _FETCH_COLS = ("quant", "alloc_mode", "worker", "cold_load_s",
                   "hot_load_s", "tok_per_s", "tok_per_s_avg",
                   "upload_time_s", "media_bytes", "n_samples",
                   "temperature", "task", "updated_at",
                   "grade", "grade_suite", "graded_at")

    def __init__(self, db: DatabaseClient):
        self.db = db

    def create_table(self, cur) -> None:
        cur.execute(MetricsQueries.CREATE_TABLE)
        for q in MetricsQueries.CREATE_INDEXES:
            cur.execute(q)
        for q in MetricsQueries.MIGRATIONS:
            cur.execute(q)

    def upsert_grade(self, cur, *, name_forms, grade, suite, detail,
                     quant="", worker="") -> int:
        """Stamp a grade across the model's rows (see UPDATE_GRADE); seed a
        stub row when it has none. Returns the number of rows carrying the
        grade afterwards (>= 1)."""
        if suite == "integrity":
            # Integrity is a verdict, not an aptitude score: its own row,
            # never the serving rows (see MetricsQueries.UPSERT_INTEGRITY).
            cur.execute(MetricsQueries.UPSERT_INTEGRITY,
                        {"m": name_forms[0], "g": grade, "d": detail})
            return 1
        params = {"names": list(name_forms), "g": grade, "s": suite,
                  "d": detail, "q": quant or ""}
        cur.execute(MetricsQueries.UPDATE_GRADE, params)
        n = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        if n == 0:
            cur.execute(MetricsQueries.INSERT_GRADE_STUB,
                        {"m": name_forms[0], "q": quant or "",
                         "w": worker or "", "g": grade, "s": suite,
                         "d": detail})
            n = 1
        return n

    def record_cold_load(self, cur, *, name_forms, worker, cold_load_s,
                         quant="") -> int:
        """A measured cold load onto every row the model has on ``worker``
        (see UPDATE_COLD_LOAD); a stub sample row when it has none."""
        cur.execute(MetricsQueries.UPDATE_COLD_LOAD,
                    {"names": list(name_forms), "w": worker,
                     "cold": cold_load_s})
        n = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        if n == 0:
            self.upsert_sample(cur, model_name=name_forms[0], quant=quant,
                               alloc_mode="", worker=worker,
                               cold_load_s=cold_load_s, hot_load_s=None,
                               tok_per_s=None, upload_time_s=None,
                               media_bytes=None, temperature=None, task=None)
            n = 1
        return n

    def upsert_sample(self, cur, *, model_name, quant, alloc_mode, worker,
                      cold_load_s, hot_load_s, tok_per_s, upload_time_s,
                      media_bytes, temperature, task) -> None:
        cur.execute(MetricsQueries.UPSERT_SAMPLE,
                    {"m": model_name, "q": quant or "", "a": alloc_mode or "",
                     "w": worker, "cold": cold_load_s, "hot": hot_load_s,
                     "tok": tok_per_s, "up": upload_time_s, "mb": media_bytes,
                     "temp": temperature, "task": task})

    def fetch_by_model(self, cur, name_forms: list) -> list:
        cur.execute(MetricsQueries.FETCH_BY_MODEL, (name_forms,))
        return [dict(zip(self._FETCH_COLS, r)) for r in cur.fetchall()]


class CallsRepository:
    _FETCH_COLS = ("ts", "worker", "quant", "alloc_mode", "tok_per_s",
                   "prompt_tokens", "completion_tokens", "elapsed_s", "task",
                   "request_id", "state", "prompt_s", "generation_s",
                   "gen_tokens")

    def __init__(self, db: DatabaseClient):
        self.db = db

    def create_table(self, cur) -> None:
        cur.execute(CallQueries.CREATE_TABLE)
        for q in CallQueries.CREATE_INDEXES:
            cur.execute(q)
        for q in CallQueries.MIGRATIONS:
            cur.execute(q)

    def insert_call(self, cur, *, model_name, worker, quant, alloc_mode,
                    tok_per_s, prompt_tokens, completion_tokens, elapsed_s,
                    task, request_id, state, prompt_s=None,
                    generation_s=None, gen_tokens=None) -> None:
        cur.execute(CallQueries.INSERT_CALL,
                    (model_name, worker, quant or "", alloc_mode or "",
                     tok_per_s, prompt_tokens, completion_tokens, elapsed_s,
                     task, request_id,
                     json.dumps(state, default=str) if state is not None else None,
                     prompt_s, generation_s, gen_tokens))

    def fetch_by_model(self, cur, name_forms: list, limit: int) -> list:
        cur.execute(CallQueries.FETCH_BY_MODEL, (name_forms, limit))
        return [dict(zip(self._FETCH_COLS, r)) for r in cur.fetchall()]

    # Per-level numbers of one CALL_STATS output row (see the query's comment).
    _STAT_COLS = ("n_calls", "n_rated", "tokens", "seconds", "p50", "p90",
                  "min", "max", "first_at", "last_at", "completion_tokens",
                  "n_no_tokens", "n_no_window", "n_engine", "n_stream",
                  "n_wall", "n_bench", "n_bench_twins", "n_no_quant",
                  "n_no_alloc")

    @staticmethod
    def _stat(vals: tuple) -> dict:
        out = {}
        for k, v in zip(CallsRepository._STAT_COLS, vals):
            if k.startswith("n_"):
                out[k] = int(v or 0)
            elif k in ("tokens", "seconds", "completion_tokens"):
                out[k] = float(v or 0)
            else:
                out[k] = float(v) if v is not None else None
        t, sec = out["tokens"], out["seconds"]
        # THE mean: Σ tokens-in-window / Σ generation seconds (never a mean of
        # per-call rates, never an EMA). None + why when nothing is rateable.
        out["mean_tok_s"] = (t / sec) if t > 0 and sec > 0 else None
        if out["mean_tok_s"] is None:
            if not out["n_calls"]:
                out["reason"] = "no calls recorded"
            else:
                why = []
                if out["n_no_tokens"]:
                    why.append(f"{out['n_no_tokens']} without completion tokens")
                if out["n_no_window"]:
                    why.append(f"{out['n_no_window']} with no generation window "
                               "(1-token replies / zero generation seconds)")
                out["reason"] = (f"{out['n_calls']} call(s), none rateable: "
                                 + "; ".join(why or ["no tokens/seconds recorded"]))
        return out

    def call_stats(self, cur, name_forms: Optional[list] = None) -> list:
        """Throughput per (model, worker, quant, alloc) cell and per model,
        from the model_calls ledger (CallQueries.CALL_STATS).

        Returns one dict per CELL (``level`` 'cell') and one per MODEL
        (``level`` 'model', ``worker`` None — the shape older consumers read).
        The model row nests ``by_worker`` {worker: stats} and ``unstamped``
        (rows with quant '' or alloc_mode '', counted in the model and worker
        totals and reported as their own bucket, never dropped); each
        by_worker entry nests its own ``unstamped``. Every stats dict carries
        n_calls, n_rated, tokens, seconds, mean_tok_s (= tokens / seconds),
        p50/p90/min/max of per-call tok/s, first_at/last_at and the counts that
        explain any absence. With ``name_forms`` every alias form of the model
        is folded under ``name_forms[0]``."""
        if name_forms:
            q = (CallQueries.CALL_STATS.replace("{model_expr}", "%s::text")
                 .replace("{model_filter}", "AND c.model_name = ANY(%s)"))
            cur.execute(q, (name_forms[0], list(name_forms)))
        else:
            cur.execute(CallQueries.CALL_STATS.replace("{model_expr}", "c.model_name")
                        .replace("{model_filter}", ""))
        cells, models, workers, u_model, u_worker = [], {}, {}, {}, {}
        for row in cur.fetchall():
            m, w, qn, am, u, g_w, g_q, g_u = row[:8]
            st = self._stat(tuple(row[8:]))
            if not g_q and not g_w:                           # (m, w, q, a)
                cells.append({"model_name": m, "worker": w, "quant": qn,
                              "alloc_mode": am, "level": "cell",
                              "unstamped": (qn or "") == "" or (am or "") == "", **st})
            elif not g_u:                                     # unstamped buckets
                if u:
                    (u_worker.setdefault(m, {}).__setitem__(w, st) if not g_w
                     else u_model.__setitem__(m, st))
            elif not g_w:                                     # (m, w)
                workers.setdefault(m, {})[w] = st
            else:                                             # (m)
                models[m] = st
        none = {"n_calls": 0, "mean_tok_s": None, "reason": "every call carries a quant and alloc stamp"}
        out = list(cells)
        for m, st in models.items():
            by_worker = {}
            for w, ws in (workers.get(m) or {}).items():
                by_worker[w] = {**ws, "worker": w,
                                "unstamped": (u_worker.get(m) or {}).get(w) or dict(none)}
            out.append({"model_name": m, "worker": None, "quant": None,
                        "alloc_mode": None, "level": "model", **st,
                        "unstamped": u_model.get(m) or dict(none),
                        "by_worker": by_worker})
        return out

    def worker_averages(self, cur, name_forms: list) -> list:
        cur.execute(CallQueries.WORKER_AVERAGES_BY_MODEL, (name_forms,))
        return [{"worker": w,
                 "avg_tok_s": float(a) if a is not None else None,
                 "n_calls": n, "last_ts": t}
                for (w, a, n, t) in cur.fetchall()]
