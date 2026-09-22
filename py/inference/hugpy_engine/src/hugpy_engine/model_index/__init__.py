"""MODEL INDEX — Postgres-backed registry store (design:
directions/REGISTRY-DB-INDEX.md; explicit-wiring layout 2026-09-10, modeled on
solcatcher's src/db/repositories/repos):

    query_registry.py   EVERY SQL statement, named — nothing inlines SQL
    client.py           DatabaseClient: DSN, fork-guard, transactions, fail-open
    repositories.py     one class per table, one method per query
    service.py          ModelIndexService: gate, locking, alias forms, guards
    this file           explicit exports + the module-level singleton the
                        call sites use (same function API as the original
                        single-file module — hooks in get_module.py,
                        models_config.py, workers.py, worker_routes.py are
                        unchanged)

ENV-GATED, INERT BY DEFAULT: HUGPY_REGISTRY_DB=pg (central's hugpy-api.env).
Every failure degrades to the JSON path and logs once.
"""
from __future__ import annotations

# ============================================================
# CORE EXPORTS
# ============================================================

from hugpy_engine.model_index.client import DatabaseClient, enabled, resolve_dsn
from hugpy_engine.model_index.service import ModelIndexService, name_forms

# ============================================================
# QUERY REGISTRIES
# ============================================================

from hugpy_engine.model_index.query_registry import (
    CallQueries,
    DiscoveryQueries,
    MetricsQueries,
    QuantQueries,
)

# ============================================================
# REPOSITORIES
# ============================================================

from hugpy_engine.model_index.repositories import (
    CallsRepository,
    DiscoveryRepository,
    MetricsRepository,
    QuantsRepository,
)

# ============================================================
# MODULE-LEVEL SINGLETON — the API every call site uses
# ============================================================

_service = ModelIndexService()

save_discovery = _service.save_discovery
load_discovery = _service.load_discovery
record_metric = _service.record_metric
record_call = _service.record_call
record_grade = _service.record_grade
record_cold_load = _service.record_cold_load
fetch_model_metrics = _service.fetch_model_metrics
fetch_model_calls = _service.fetch_model_calls
fetch_worker_averages = _service.fetch_worker_averages

__all__ = [
    "DatabaseClient", "ModelIndexService", "enabled", "resolve_dsn",
    "name_forms",
    "DiscoveryQueries", "QuantQueries", "MetricsQueries", "CallQueries",
    "DiscoveryRepository", "QuantsRepository", "MetricsRepository",
    "CallsRepository",
    "save_discovery", "load_discovery", "record_metric", "record_call",
    "record_grade", "record_cold_load",
    "fetch_model_metrics", "fetch_model_calls", "fetch_worker_averages",
]
