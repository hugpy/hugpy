"""ModelIndexService — the business layer over the repositories (explicit
wiring 2026-09-10, modeled on solcatcher's repos/*/service). Owns: the
enabled() gate, locking, transactions, alias forms, the empty-report guard,
and fail-open degradation. Callers use the module-level functions in
``__init__``; nothing outside this package touches a repository directly.
"""
from __future__ import annotations

import logging

from hugpy_engine.model_index.client import DatabaseClient, enabled
from hugpy_engine.model_index.repositories import (
    CallsRepository,
    DiscoveryRepository,
    MetricsRepository,
    QuantsRepository,
)

logger = logging.getLogger("abstract_hugpy_dev.model_index")


def name_forms(model_name: str) -> list:
    """Alias forms one model may be recorded under: the given key and its
    owner-stripped tail ('Jackrong~X' and 'X' are one model — the serve path
    records under the RESOLVED key, which may drop the owner prefix)."""
    forms = [str(model_name)]
    tail = str(model_name).split("~")[-1]
    if tail and tail not in forms:
        forms.append(tail)
    return forms


class ModelIndexService:

    def __init__(self, db: "DatabaseClient | None" = None):
        self.db = db or DatabaseClient()
        self.discovery = DiscoveryRepository(self.db)
        self.quants = QuantsRepository(self.db)
        self.metrics = MetricsRepository(self.db)
        self.calls = CallsRepository(self.db)
        self._started_pid = None

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self, cur) -> None:
        """Create every table/index — once per process, on the first call."""
        import os
        if self._started_pid == os.getpid():
            return
        self.discovery.create_table(cur)
        self.quants.create_table(cur)
        self.metrics.create_table(cur)
        self.calls.create_table(cur)
        self._started_pid = os.getpid()

    # ── discovery report ──────────────────────────────────────────────────
    def save_discovery(self, report: dict) -> bool:
        """Persist a walk's full report in ONE transaction (delete-absent +
        upsert-present — rebuildable-index semantics); refresh model_quants
        from each row's marker `quants`. An EMPTY report never truncates
        (degraded-mount caution — the remedy is 'skip this write', never
        'keep stale rows')."""
        if not enabled() or not isinstance(report, dict):
            return False
        rows = {str(k): v for k, v in report.items() if isinstance(v, dict)}
        if not rows:
            return False
        try:
            with self.db.lock:
                with self.db.transaction(), self.db.cursor() as cur:
                    self.start(cur)
                    names = list(rows)
                    self.discovery.delete_absent(cur, names)
                    self.quants.delete_absent_models(cur, names)
                    for name, row in rows.items():
                        self.discovery.upsert_row(cur, name, row)
                        self.quants.replace_for_model(cur, name,
                                                      row.get("quants"))
            logger.info("registry DB: saved discovery report (%d rows)",
                        len(rows))
            return True
        except Exception as exc:  # noqa: BLE001 — never break the walk
            self.db.mark_unavailable(exc, "saving the discovery report")
            return False

    def load_discovery(self) -> "dict | None":
        """The report as a dict, or None (off/empty/unreachable — the caller
        falls back to the JSON file; empty is None on purpose: before the
        first walk writes, JSON remains authoritative)."""
        if not enabled():
            return None
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    self.start(cur)
                    rows = self.discovery.fetch_all(cur)
            return rows or None
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "loading the discovery report")
            return None

    # ── metrics card + call ledger ────────────────────────────────────────
    def record_metric(self, model_name: str, worker: str, *, quant: str = "",
                      alloc_mode: str = "", tok_per_s=None, cold_load_s=None,
                      hot_load_s=None, upload_time_s=None, media_bytes=None,
                      temperature=None, task=None) -> bool:
        if not enabled() or not model_name or not worker:
            return False
        try:
            with self.db.lock:
                with self.db.transaction(), self.db.cursor() as cur:
                    self.start(cur)
                    self.metrics.upsert_sample(
                        cur, model_name=model_name, quant=quant,
                        alloc_mode=alloc_mode, worker=worker,
                        cold_load_s=cold_load_s, hot_load_s=hot_load_s,
                        tok_per_s=tok_per_s, upload_time_s=upload_time_s,
                        media_bytes=media_bytes, temperature=temperature,
                        task=task)
            return True
        except Exception as exc:  # noqa: BLE001 — never fail a serve
            self.db.mark_unavailable(exc, "recording a model metric")
            return False

    def record_grade(self, model_name: str, grade, *, suite=None,
                     detail=None, quant: str = "", worker: str = "") -> bool:
        """Persist an APTITUDE grade (0..100) for ``model_name`` onto its
        metrics rows (t147) — every alias form, every quant/alloc/worker row
        (or just ``quant`` when given); a stub row when it was never served.
        ``detail`` (dict/list -> JSON text) is the suite's per-task record.
        Fail-open like every write here: False on off/invalid/unreachable."""
        if not enabled() or not model_name:
            return False
        try:
            g = float(grade)
        except (TypeError, ValueError):
            return False
        if not (0.0 <= g <= 100.0):
            return False
        if detail is not None and not isinstance(detail, str):
            import json
            try:
                detail = json.dumps(detail, default=str)[:20000]
            except Exception:  # noqa: BLE001
                detail = None
        try:
            with self.db.lock:
                with self.db.transaction(), self.db.cursor() as cur:
                    self.start(cur)
                    self.metrics.upsert_grade(
                        cur, name_forms=name_forms(model_name), grade=g,
                        suite=(str(suite) if suite else None),
                        detail=detail, quant=quant or "", worker=worker or "")
            return True
        except Exception as exc:  # noqa: BLE001 — never fail a grader
            self.db.mark_unavailable(exc, "recording a model grade")
            return False

    def record_cold_load(self, model_name: str, worker: str, cold_load_s,
                         *, quant: str = "") -> bool:
        """A measured COLD load (seconds, from the worker's calibration
        wire) onto ``model_name``'s rows on ``worker`` — the cold_load_s
        column the serve seam leaves NULL. Fail-open."""
        if not enabled() or not model_name or not worker:
            return False
        try:
            c = float(cold_load_s)
        except (TypeError, ValueError):
            return False
        if not (c > 0.0):
            return False
        try:
            with self.db.lock:
                with self.db.transaction(), self.db.cursor() as cur:
                    self.start(cur)
                    self.metrics.record_cold_load(
                        cur, name_forms=name_forms(model_name),
                        worker=str(worker), cold_load_s=c, quant=quant or "")
            return True
        except Exception as exc:  # noqa: BLE001 — never fail a heartbeat
            self.db.mark_unavailable(exc, "recording a cold load")
            return False

    def record_call(self, model_name: str, worker: str, *, quant: str = "",
                    alloc_mode: str = "", tok_per_s=None, prompt_tokens=None,
                    completion_tokens=None, elapsed_s=None, task=None,
                    request_id=None, state=None) -> bool:
        if not enabled() or not model_name or not worker:
            return False
        try:
            with self.db.lock:
                with self.db.transaction(), self.db.cursor() as cur:
                    self.start(cur)
                    self.calls.insert_call(
                        cur, model_name=model_name, worker=worker,
                        quant=quant, alloc_mode=alloc_mode,
                        tok_per_s=tok_per_s, prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        elapsed_s=elapsed_s, task=task,
                        request_id=request_id, state=state)
            return True
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "recording a call")
            return False

    def fetch_model_metrics(self, model_name: str) -> list:
        if not enabled() or not model_name:
            return []
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    self.start(cur)
                    return self.metrics.fetch_by_model(
                        cur, name_forms(model_name))
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "fetching model metrics")
            return []

    def fetch_model_calls(self, model_name: str, limit: int = 200) -> list:
        if not enabled() or not model_name:
            return []
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    self.start(cur)
                    return self.calls.fetch_by_model(
                        cur, name_forms(model_name),
                        max(1, min(int(limit or 200), 2000)))
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "fetching model calls")
            return []

    def fetch_worker_averages(self, model_name: str) -> list:
        if not enabled() or not model_name:
            return []
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    self.start(cur)
                    return self.calls.worker_averages(
                        cur, name_forms(model_name))
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "fetching worker averages")
            return []
