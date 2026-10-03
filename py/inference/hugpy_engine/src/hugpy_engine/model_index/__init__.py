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
    PAIR_KNOB_KEYS,
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
record_call_if_absent = _service.record_call_if_absent
fetch_call_stats = _service.fetch_call_stats
record_grade = _service.record_grade
record_cold_load = _service.record_cold_load
fetch_model_metrics = _service.fetch_model_metrics
fetch_model_calls = _service.fetch_model_calls
fetch_worker_averages = _service.fetch_worker_averages
sync_worker_settings = _service.sync_worker_settings
fetch_models_for_display = _service.fetch_models_for_display
fetch_worker_settings = _service.fetch_worker_settings
resolve_model_id = _service.resolve_model_id
write_pair_knobs = _service.write_pair_knobs
read_pair_knobs = _service.read_pair_knobs
pair_knobs_by_worker = _service.pair_knobs_by_worker
set_pair_assigned = _service.set_pair_assigned
fetch_assigned_by_worker = _service.fetch_assigned_by_worker
set_pair_pinned = _service.set_pair_pinned
record_worker_presence = _service.record_worker_presence
fetch_all_serving_settings = _service.fetch_all_serving_settings
set_model_serving_settings = _service.set_model_serving_settings
fetch_worker_presence = _service.fetch_worker_presence
fetch_allocation_candidates = _service.fetch_allocation_candidates
fetch_presence_catalog = _service.fetch_presence_catalog
fetch_pairs_by_worker = _service.fetch_pairs_by_worker


def last_db_error():
    """The service client's last recorded fault ({doing, error, at}) or None."""
    return _service.db.last_error


__all__ = [
    "DatabaseClient", "ModelIndexService", "enabled", "resolve_dsn",
    "name_forms",
    "DiscoveryQueries", "QuantQueries", "MetricsQueries", "CallQueries",
    "DiscoveryRepository", "QuantsRepository", "MetricsRepository",
    "CallsRepository",
    "save_discovery", "load_discovery", "record_metric", "record_call", "record_call_if_absent",
    "record_grade", "record_cold_load",
    "fetch_model_metrics", "fetch_model_calls", "fetch_worker_averages", "fetch_call_stats",
    "sync_worker_settings",
    "fetch_models_for_display",
    "fetch_worker_settings",
    "resolve_model_id", "write_pair_knobs", "read_pair_knobs", "pair_knobs_by_worker", "PAIR_KNOB_KEYS", "set_pair_assigned", "fetch_assigned_by_worker",
    "set_pair_pinned", "fetch_pairs_by_worker",
    "record_worker_presence", "fetch_worker_presence", "fetch_allocation_candidates", "fetch_presence_catalog",
    "fetch_all_serving_settings", "set_model_serving_settings",
    "last_db_error",
]
