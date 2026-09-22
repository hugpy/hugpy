# Registry → rebuildable DB index (operator decision, 2026-09-10)

**Decision (operator-approved):** keep `hugpy.json` markers as the per-model
source of truth co-located with the weights; replace the aggregate JSON
registries (`projects/model_discovery.json`, `projects/model_manifest.json`)
with a **database index that is rebuilt from markers** — droppable and
regenerable at any time, never a second truth.

## Why

The 2026-09-10 incident chain showed the JSON registries' failure modes:

- **Phantom persistence.** The walk registered org dirs (`gguf/<org>`),
  downloader staging (`<repo>.tmp-<pid>`), `_archive/` and `incoming/` rows;
  the merge ("re-adopt prior location — don't throw it away") and the shrink
  guard then kept them alive after the walk bugs were fixed. Rows had to be
  removed by hand-editing JSON.
- **Count wobble.** The console model total moved with every download
  start/finish because staging dirs entered and left the walk.
- **Swallowed models.** `gguf/ponpoke` registered as a model and pruned the
  walk, hiding `flux2-klein-9b-uncensored-text-encoder` from the catalog
  entirely (fixed in `paths.py::is_model_dir` — org-level guard).
- **Whole-file rewrites** race between central, the worker agent and the
  downloader; there is no row-level write.

## Shape

- **Markers (`hugpy.json`)** — identity, capability flags, and now the
  **`quants` manifest** (`[{file, quant, bytes, shards}]`, mmproj excluded,
  shard sets collapsed), stamped at `write_hugpy_marker` time and kept true by
  `sync_marker_quants` on every discovery walk (2026-09-10). Markers travel
  with the files, survive central loss, and make the store self-describing.
- **DB index** — SQLite beside the comms mirror
  (`llm_storage/comms/`, same pattern the download queue already uses) or the
  toolserver Postgres. Tables ≈ models(hub_id, name, framework, tasks,
  folder, bytes, status, marker_json, walked_at), quants(model, file, quant,
  bytes, shards). `rebuild()` = walk markers → truncate + reinsert. All
  readers (`get_models_dict`, /api/models, quant choices, storage proposals)
  move to the index; the JSONs become export artifacts or die.
- **No shrink guard needed**: a degraded-mount walk is detected the same way,
  but the remedy is "don't rebuild now", not "never delete rows".

## Prerequisite already done

Discovery hardening shipped 2026-09-10 (src + both installed wheels):
org-level guard in `is_model_dir`, `.tmp-` exclusion, `_archive`/`incoming`/
`dedupes` excluded from the walk, quant manifest on markers. A future
`backfill_markers` sweep (exists in `hugpy_marker.py`) will stamp the last
unstamped dirs so the index can be built from markers alone.

## Phase 1 — SHIPPED 2026-09-10

- Database `hugpy` (owner: user `hugpy`) on the toolserver's Postgres server —
  same infra, separate database, toolserver's tables untouched.
- `imports/src/model_index.py`: env-gated (`HUGPY_REGISTRY_DB=pg` +
  `HUGPY_REGISTRY_PG_DSN`, URI form — a space-separated DSN breaks shell
  sourcing of the env file), autocommit + `conn.transaction()` (psycopg3's
  `with conn:` CLOSES the connection — do not reintroduce it).
- Dual-write: `discover_models` saves the report to the DB after the shrink
  guard (a refused degraded walk never reaches the DB; an empty report never
  truncates). JSON remains the fallback/export.
- DB-first read: `models_config._load_discovery_report` — DB when enabled and
  populated, else the JSON path byte-identical to before. Workers (no flag,
  no psycopg) are untouched.
- `model_quants` fills from marker `quants` riding the report rows; it grows
  as markers get stamped/synced (backfill_markers sweep = the remaining step
  to full coverage).

## Next phases (operator: "most everything for hugpy should be converted to db")

1. `model_manifest.json` (download registry) → table, same dual-write pattern.
2. Marker backfill sweep so model_quants covers the whole store.
3. Runtime settings / overrides / priority groups stores → tables.
4. Retire the JSON files once each reader is DB-first and burned in.

## Metrics card — SHIPPED 2026-09-10 (same session)

Modeled on the operator's `llm_storage/assets/metrics.xlsx_0.ods`:
- `model_metrics` — rollup card, one row per (model × quant × alloc_mode ×
  worker): cold/hot load s (fed later by load probes/battery), last +
  running-average tok/s, n_samples, upload_time_s, media_bytes, task.
- `model_calls` — append-only per-call ledger (every serve: tok/s,
  prompt/completion tokens, elapsed, request_id, config-state snapshot).
- Recorder hook: `workers.record_serve_metrics` (the one tok/s writer) —
  captures worker/alloc/moe/4-bit inside the store lock, writes outside it.
- Read: `GET /api/llm/models/<key>/metrics` → {metrics, worker_averages,
  calls}; alias-tolerant (`Jackrong~X` ≡ `X`).
- Seeded from the fleet's live model_tok_stats (31 rows).
- Gotcha fixed: PID fork-guard on the PG connection (gunicorn forks after a
  boot-time connect; a shared socket = "SSL bad record mac").

## DB code layout — STANDARD (operator ruling, 2026-09-10)

All hugpy DB creation and calls follow the explicit repository pattern of
solcatcher's `src/db/repositories/repos` — implemented for the model index at
`imports/src/model_index/` and REQUIRED for every future table:

    query_registry.py   EVERY SQL statement, named constants grouped per
                        repository (CREATE_TABLE + CREATE_INDEXES first,
                        then the named operations). Nothing inlines SQL.
    client.py           DatabaseClient — DSN resolution, PID fork-guard,
                        psycopg3 autocommit + transaction(), fail-open
                        mark_unavailable(). The only file that imports psycopg.
    repositories.py     One class per table, one explicit method per query.
                        Thin: parameters in, rows out.
    service.py          The business layer — enabled() gate, locking,
                        transactions, alias forms, guards, degradation.
    __init__.py         Explicit exports + a module-level singleton exposing
                        the plain-function API call sites use.
