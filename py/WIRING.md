# Composition wiring ledger

Everything an upper layer must install at startup so the seams below it have
real implementations. `hugpy_server.wsgi_app` (central) and the fleet worker
entry points (worker boxes) are the composition roots. Each row was reported
by the package agent that created the seam; keep this file current.

| Seam (lower package) | Installer to call | Implementation (upper package) | Who wires it |
|---|---|---|---|
| `hugpy_engine.placement` (WorkerRegistry, WorkerTransport, EvictionLedger, Blocklist, ModelMetrics, PriorityGroups) | `hugpy_fleet.central.placement.install()` | fleet adapters over `central.workers/worker_http/evictions/blocklist/model_metrics/priority_groups` | server startup |
| `hugpy_engine.tasks` runner table | `hugpy_engine.tasks.load_entry_points()` or `hugpy_media.plugin.register()` / `hugpy_video.plugin.register()` | media, video (entry-point group `hugpy_engine.tasks`) | server startup; fleet worker boot (optional, lazy) |
| `hugpy_storage.catalog_source` (CatalogSource) | `hugpy_engine.catalog_bridge.install()` | engine registry adapter | engine (lazy from facade); server/fleet worker call it explicitly |
| `hugpy_storage.providers.set_serve_path_hook` | `hugpy_engine.catalog_bridge.install()` | `hugpy_engine.serve.hot_cache.use` | engine |
| `hugpy_storage.providers.set_budget_gate / set_transfer_telemetry / set_executor_registrar` | fleet worker boot **and every slot child** | `worker.budget.evict_to_fit`, `central.evictions`, `worker.agent.register_executor` | fleet |
| `hugpy_storage.providers.set_footprint_selector` | server startup (`hugpy_server.wiring.install_all`) | `hugpy_storage.format_select.effective_bytes` (relocated from the server; storage could make it the default) | server |
| `hugpy_storage.hf_token.add_token_listener(fn)` | server startup (`hugpy_server.wiring.install_hf_token_listener`) | `hugpy_server.app.functions.imports.utils.constants.rebuild_hf_api` | server |
| `hugpy_control.bus` topic `catalog.changed` (`TOPIC_CATALOG_CHANGED`) | subscribe in `hugpy_engine.catalog_bridge.install()` | published by storage after download/wipe/promote | engine |
| `hugpy_media.hooks.set_process_hooks(...)` | fleet worker boot (lazy, when media importable) | `hugpy_fleet.worker.pid_registry.record_foreign_call/end_foreign_call` | fleet |
| `hugpy_video.hooks.PromptCoordinator` + `hugpy_video.jobs.register_job("video_performance")` | `hugpy_oracle.install_hooks()` | `hugpy_oracle.relay.hooks.OraclePromptCoordinator`, `run_video_performance` | server startup |
| `hugpy_oracle.providers.set_dossier_source` | `hugpy_curation.install_providers()` (or `hugpy-curation install-providers`) | `hugpy_curation.dossier.oracle_source.DossierStoreSource` | server startup |
| `hugpy_curation.providers.set_doctrine_source` (`latest()`) | `hugpy_curation.install_providers(doctrine_source=...)`; falls back to the source installed in `hugpy_oracle.providers` | fleet doctrine adapter | server startup (one doctrine source serves oracle and curation) |
| `hugpy_oracle.providers.set_doctrine_source / set_task_capability_gate / set_load_state_source` | server startup | fleet adapters (`hugpy_fleet.doctrine` latest, `central.workers._task_capable`, `central.workers.load_state_for_model`) | server |
| `hugpy_oracle.runtime.execute_route` pool | `oracle_routes` puts the API-key-resolved `pool` into the body | server | server |

Open items recorded by agents:

- `hugpy_oracle.interim_ledger.MediaBusSource` reads video's `media_jobs.db` read-only; a `hugpy_video.jobs` bulk-export API would remove the cross-package file read.
- `format_select.py` now lives at `hugpy_storage.format_select` (relocated 2026-09-22); `hugpy_storage.providers.set_footprint_selector` could default to `effective_bytes` directly (storage-owned change, not yet made).
- `worker_store_isolation` is vendored at `py/services/hugpy_server/tests/integration/worker_store_isolation.py` (a copy of fleet's); the server conftest puts `tests/` and `tests/integration/` on `sys.path`.
- The server composition root is `hugpy_server.wiring.install_all(app)`, called from `hugpy_server.wsgi_app.get_hugpy_flask` and reported at `app.extensions["hugpy_wiring"]`; it also wires the control bus (`wire_cancel`/`wire_job_events`/`wire_settings_events`) and the fleet eviction store sink.
