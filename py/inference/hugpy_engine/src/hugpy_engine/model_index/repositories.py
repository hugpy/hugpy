"""Repositories — one class per table, one explicit method per QueryRegistry
operation (explicit-wiring ruling 2026-09-10, modeled on solcatcher's
repos/*/repository classes). No SQL here; it all lives in query_registry.

Repositories are THIN: parameters in, rows out. Fallbacks, locking and the
enabled() gate belong to the service; connections to the DatabaseClient.
"""
from __future__ import annotations

import json

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
                   "request_id")

    def __init__(self, db: DatabaseClient):
        self.db = db

    def create_table(self, cur) -> None:
        cur.execute(CallQueries.CREATE_TABLE)
        for q in CallQueries.CREATE_INDEXES:
            cur.execute(q)

    def insert_call(self, cur, *, model_name, worker, quant, alloc_mode,
                    tok_per_s, prompt_tokens, completion_tokens, elapsed_s,
                    task, request_id, state) -> None:
        cur.execute(CallQueries.INSERT_CALL,
                    (model_name, worker, quant or "", alloc_mode or "",
                     tok_per_s, prompt_tokens, completion_tokens, elapsed_s,
                     task, request_id,
                     json.dumps(state) if state is not None else None))

    def fetch_by_model(self, cur, name_forms: list, limit: int) -> list:
        cur.execute(CallQueries.FETCH_BY_MODEL, (name_forms, limit))
        return [dict(zip(self._FETCH_COLS, r)) for r in cur.fetchall()]

    def worker_averages(self, cur, name_forms: list) -> list:
        cur.execute(CallQueries.WORKER_AVERAGES_BY_MODEL, (name_forms,))
        return [{"worker": w,
                 "avg_tok_s": float(a) if a is not None else None,
                 "n_calls": n, "last_ts": t}
                for (w, a, n, t) in cur.fetchall()]
