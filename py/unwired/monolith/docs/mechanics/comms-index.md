<!-- MECHANICS DOC — comms-index. Paths relative to
src/abstract_hugpy_dev/src/abstract_hugpy_dev/ unless noted. -->

# Comms substrate + model registry — mechanics

**Scope (paths this doc covers):** `comms/*` (bus, jobs, heartbeat_db, evictions,
evict_policy, principals, priority_groups, model_metadata, model_metrics,
model_status_cache, blocklist, calibration, feeds, todo_keeper[_daemon]),
`imports/src/model_index/*` (client, query_registry, repositories, service),
`flask_app/app/functions/imports/utils/workers.py` (worker_store + placement).
**One-liner:** the stdlib-only control-plane substrate — message bus, job
tracking, the worker registry/router, operator policy stores, and fleet
telemetry — that every transport imports, plus a Postgres model-metrics/
registry layer that is quietly, only-partly replacing one corner of it.
**Owner process(es):** central `7002_hugpy_api` owns almost all of this
(JobStore, Bus, WorkerStore, the feeds refresher thread, calibration/metrics/
metadata stores). `worker_agent` writes heartbeat_db rows, relays eviction
telemetry, and produces calibration samples. `hugpy-todo-keeper.service` is a
separate standing daemon that talks to central over HTTP, not a library import.

## 1. Purpose & responsibilities
- One job schema (`Job`/`JobStore`, F5) and one control/lifecycle bus
  (`Bus`/`BusMessage`, F1) for every transport, importable without cycles —
  `comms/__init__.py:1-16` is explicit that this package is stdlib-only and
  imports nothing else from `abstract_hugpy_dev`.
- The worker fleet registry: persistence, admission, model assignment, and the
  routing-rank/placement decision for "which worker serves this request"
  (`WorkerStore`, in `flask_app/.../workers.py`, not `comms/`).
- Several operator-facing runtime-policy stores riding one shared F4 idiom
  (`comms/settings.py`, `settings.json`, fcntl read-modify-write): `blocklist.py`,
  `priority_groups.py`, `evict_policy.py`.
- Fleet telemetry (`evictions.py`, `calibration.py`) and **two independently
  wired** metrics/registry pipelines: legacy `model_metrics.py` (SQLite,
  silently upgradeable to an external Postgres store) and the new
  `imports/src/model_index` registry (Postgres db `hugpy`, explicit repos/SQL).
- Does **not** gate or veto a serving decision from telemetry — `evictions.py`
  is observation-only by construction (its own docstring: "if this module were
  deleted the behavior... would be byte-identical"). Does **not** carry the
  token/data plane (`bus.py` is control-plane only). Does **not** replace legacy
  per-file identity stores — `principals.py` wraps them, never supersedes them.

## 2. Key modules
| file | responsibility |
|---|---|
| `comms/bus.py` | F1 typed frozen pub/sub (`BusMessage`); control and lifecycle share one bus |
| `comms/jobs.py` | F5 `Job`/`JobStore` — one schema for chat/download/media jobs; cancel races, cross-process mirror |
| `comms/shared.py` | (support) `SqliteMirror` — cross-process job mirror + download claim queue; used by `jobs.py`, `evictions.py` |
| `comms/settings.py` | F4 runtime settings store (`settings.json`); backs blocklist/priority_groups/evict_policy |
| `comms/principals.py` | F2 `Principal` identity wrapper over the legacy credential stores + `hpp_` tokens |
| `comms/blocklist.py` | operator model block/unblock (pool-wide) + per-(model,worker) pair blocks |
| `comms/priority_groups.py` | explicit ordered fallback lists; outranks derived model-groups; supplies the placement `workers` fallback |
| `comms/evict_policy.py` | fleet-wide `least_reaping` drop-pass switch (parity between central's preview and a worker's auto-evict) |
| `comms/evictions.py` | serve-pipeline telemetry stream (provision/resolve/load/evict events) + durable `EvictionStore` + worker→central `EvictionRelay` |
| `comms/calibration.py` | learned VRAM/RSS/load-time correction factors (measured vs. predicted) |
| `comms/model_metadata.py` | fetch-once HF/Civitai per-repo metadata cache (SQLite) |
| `comms/model_status_cache.py` | per-process TTL + single-flight memo in front of the model install-status filesystem walk |
| `comms/model_metrics.py` | **legacy** load/serve/call EMA store — SQLite by default, swapped for `abstract_toolserver` Postgres at import time |
| `comms/heartbeat_db.py` | worker heartbeat dual-write to Postgres (db `hugpy`), reusing model_index's `DatabaseClient` |
| `comms/feeds.py` | materializes console feeds (workers/slots/queue/jobs/...) into Postgres (db `hugpy`) for the push channel |
| `comms/todo_keeper.py` | pure todo.add/todo.tidy contract: parse task → build prompt → parse model reply |
| `comms/todo_keeper_daemon.py` | the I/O half — enroll/heartbeat/poll/report loop against central's `/agent/*` HTTP surface |
| `imports/src/model_index/client.py` | `DatabaseClient` — DSN resolution, fork-guard, the ONE place model_index touches psycopg |
| `imports/src/model_index/query_registry.py` | every SQL statement, named (`Discovery`/`Quant`/`Metrics`/`Call` queries) |
| `imports/src/model_index/repositories.py` | one class per table, thin params-in/rows-out |
| `imports/src/model_index/service.py` | `ModelIndexService` — the `enabled()` gate, locking, alias forms, fail-open |
| `flask_app/.../workers.py` | `WorkerStore` (registry/eligibility/ranking) + `_match_keys`/`placement_policy`/`_pref_index` placement helpers |

## 3. Entry points
- `comms/__init__.py:17-115` — the "import comms" surface: re-exports
  `Job`/`JobStore`/`Bus`/`BusMessage`/`wire_cancel`/`wire_job_events`,
  `Principal`/`PrincipalStore`/`allowed`, `SettingsStore`, blocklist verbs,
  calibration, and `ModelMetricsStore`/`model_metrics_store`. **Not** re-exported**:
  `heartbeat_db`, `feeds`, `priority_groups`, `evictions`, `model_status_cache`,
  `model_metadata` — callers import those modules directly.
- `comms.bus.wire_cancel` / `wire_job_events` (`bus.py:182,210`) — called once per
  process entrypoint (flask app factory, worker agent main) to wire Job
  transitions onto the Bus and `control.cancel` back onto `JobStore.cancel`.
- HTTP: flask routes call `worker_store` (`workers.py:4681`) for `/llm/workers*`,
  `job_store` (`jobs.py:1051`) for `/llm/jobs`+`/jobs`, `evictions.recent()`/
  `EvictionStore` (`evictions.py:724,988`) for `/llm/evictions*`, and
  `/agent/register`+heartbeat+tasks+result (agent_nodes.py, adjacent, out of
  scope) for the todo-keeper's HTTP counterpart.
- `python -m abstract_hugpy_dev.comms.todo_keeper_daemon` —
  `hugpy-todo-keeper.service`'s `ExecStart` (see §9 — the one entry point here
  that does **not** run from the src checkout).
- `imports.src.model_index` module functions (`record_metric`, `record_call`,
  `fetch_model_metrics`, `fetch_model_calls`, `fetch_worker_averages`,
  `save_discovery`, `load_discovery` — `__init__.py:47-53`) — imported directly
  by `get_module.py`, `models_config.py`, `workers.py`, `worker_routes.py`.

## 4. Data flow (the spine)
1. **Job lifecycle** (chat/download/media, any transport): `JobStore.create()`
   (`jobs.py:391`) stamps a `Job`, fires `on_change` → the Bus wiring publishes
   `job.created`/`job.status`/`job.done` (`bus.py:219-231`), and mirrors to
   `shared.SqliteMirror` so a cancel POST landing on gunicorn worker B can flag
   a job process A owns. `cancel_authoritative()` (`jobs.py:636-707`) either
   **relays** (a live local stream tears itself down and writes the terminal
   status) or **force-terminals** the row directly (no live owner anywhere) —
   never both, and never reports `cancelled:true` for an unknown id.
2. **Worker pick for a model** (`WorkerStore.pick_for_model`, `workers.py:4279`):
   operator BLOCK gate → alias-tolerant eligibility (`workers_for_model`,
   `:3984`; an ALLOCATED model is a HARD scope per the 2026-08-28 ruling) →
   package-version soft gate → k56 ordered `worker_prefs` HARD scope
   (`placement_policy` → `managers/serve/overrides.py:833`, falling back to
   `comms.priority_groups.workers_for_key` when the model carries no prefs of
   its own, `priority_groups.py:282-305`) → `_routing_rank` sort (designation >
   measured-resident > allocated > ⭐star/GPU/least-recently-picked,
   `workers.py:2721-2734`) → polite (no-evict) admission walk. `candidates_for_model`
   (`:4621`) shares the identical scope/rank so a mid-flight reroute is never a
   re-decision.
3. **Serve-pipeline telemetry**: `emit_eviction_event(stage, **fields)`
   (`evictions.py:624`) → local ring + registered sinks, every step
   independently guarded (never raises into the load/evict path) → on a worker,
   `EvictionRelay` (`:1033`) batches to central's `/llm/evictions/ingest` every
   0.5s or 20 events; on central, `install_store_sink` (`:1005`) appends
   straight into the durable `EvictionStore` (SQLite, `:809`). The same event
   additionally taps `_tap_compute_action` (`:665-668`, stage→action map
   `:676-685`) into `model_metrics_store.append_action` — the one place eviction
   telemetry and the metrics pipeline touch.
4. **The two metrics/registry pipelines** (see §8 for the split's live health):
   (a) `managers/resolvers/remote.py:239,258` call `model_metrics_store.record_call`/
   `record_load` → either the local SQLite `ModelMetricsStore` (`model_metrics.py:112`)
   or, when importable, `abstract_toolserver.metrics.PgMetricsStore` (swapped in
   at import time, `model_metrics.py:637-642`) writing Postgres db `toolserver`.
   (b) `imports/src/model_index`'s `record_metric`/`record_call`
   (`__init__.py:49-50` → `service.py:98,120`) write Postgres db `hugpy` through
   `repositories.py`, using only the named SQL in `query_registry.py`, gated by
   `HUGPY_REGISTRY_DB=pg` (`client.py:17-20`).
5. **Worker heartbeat / console feeds**: `heartbeat_db.upsert()` (`:98`)
   dual-writes each beat to Postgres table `hugpy_worker_heartbeat` (db `hugpy`,
   reusing `imports.src.model_index.client.DatabaseClient` with
   `dsn_resolver=dsn` pinned to `HUGPY_REGISTRY_PG_DSN`, `:64-69`) beside the
   existing HTTP beat; `feeds.py`'s leader-elected refresh loop (`:156`,
   `pg_try_advisory_lock`) materializes `/llm/workers`, `/llm/slots`,
   `/llm/queue`, etc. into table `hugpy_feed` (same db `hugpy`, default
   `DatabaseClient()` resolution, `:88-93`), digest-gated so an unchanged
   payload never re-writes.

## 5. State, persistence & invariants
- **JSON, fcntl-locked read-modify-write** (the recurring idiom): `workers.json`
  (`workers.py:131-133`, beside the model manifest), `principals.json`
  (`principals.py:156`), `settings.json` (`settings.py:27-32`, backs
  `blocklist`/`priority_groups`/`evict_policy` plus `discord.*`/`personalities`/
  `delegation`/`gates`). Each: reload → mutate → atomic `os.replace` under an
  exclusive lock, a short (2-3s) read cache, and a non-empty-but-unparseable
  file is treated as CORRUPTION — `workers.py:3027-3061` logs and **re-raises**
  rather than silently "healing" to `{}` and wiping the fleet on the next write.
- **SQLite, WAL, self-disabling stores**, all under `$PROJECTS_HOME` (resolves
  to the **cold** `llm_storage` mount on this box unless `HUGPY_*_DB`
  overrides — confirm against PROVISIONS.md before assuming that's fine for a
  given file): `model_metadata.db`, `calibration.db`, `model_metrics.db`
  (legacy fallback), the comms mirror db (`HUGPY_COMMS_DB`; jobs' cross-process
  claim queue + cancel flags), and `eviction_events` (same db file by default,
  `evictions.py:782-795`). All follow "N consecutive failures → disable loudly,
  degrade to the non-DB behavior" — **except** `model_metrics.py`'s own
  `ModelMetricsStore._ensure()`, which disables silently (§8).
- `model_status_cache.py` converges across gunicorn's 3 worker processes
  through a small **local** epoch file (`$TMPDIR/hugpy-model-status.epoch`,
  `:146-155`), not the shared store — deliberate, so the cache-busting read
  never adds a virtiofs round-trip of its own.
- **Postgres, two separate databases**: db `hugpy` (via
  `imports/src/model_index/client.py`'s `DatabaseClient`, reused verbatim by
  `heartbeat_db.py` and `feeds.py`) holds `discovery_models`, `model_quants`,
  `model_metrics`, `model_calls`, `hugpy_worker_heartbeat`, `hugpy_feed`; db
  `toolserver` (via the separate `abstract_toolserver` pip package,
  `SOLCATCHER_POSTGRESQL_*` creds) holds the LEGACY `load_metrics`/
  `call_metrics`/`call_metrics_by_task`/`compute_actions` tables
  `model_metrics.py` can swap onto.
- **Invariants**: first-terminal-status-wins for a `Job` (`jobs.py:492-496`); a
  blocked model is never a routing candidate anywhere, checked before any
  per-worker work (`workers.py:3995-4001`); at most one ENABLED priority group
  may claim a given model key, enforced on write (`priority_groups.py:371-380`);
  an ordered `worker_prefs` list is a HARD scope — a model carrying one never
  lands off it, even onto the wider fleet (`workers.py:2814-2827`).

## 6. Cross-subsystem edges
- **→ `managers/serve/overrides.py`** ([[serving-core]]): `placement_policy()`/
  `resolve_polite()` are the real per-model prefs/polite store; `workers.py`'s
  `placement_policy`/`_polite_on` (`:2772-2799`) are guarded re-exports, and
  `overrides.placement_policy` (`:833-874`) is what actually calls
  `priority_groups.workers_for_key` as the fallback (`:866-868`).
- **→ `managers/eviction.py`** ([[serving-core]]): `evict_policy.least_reaping()`
  reads the fleet-wide drop-pass switch whose default `managers/eviction.py`
  owns (`evict_policy.py:67`); `evictions.py`'s telemetry instruments (never
  gates) `managers.dispatch.dispatch`'s headroom pass.
- **→ `abstract_toolserver`** (external pip package,
  `/srv/hugpy/venv/lib/python3.12/site-packages/abstract_toolserver/`, **not**
  part of this src tree): `model_metrics.py:639` imports
  `abstract_toolserver.metrics.PgMetricsStore` — the only dependency in this
  doc's scope on code outside `abstract_hugpy_dev` entirely. The same package's
  MCP tool surface (`mcp__toolserver__db_*`/`metrics_*`) reads the same db
  `toolserver` tables from a different process.
- **← `flask_app/app/routes/worker_routes.py`**: heartbeat ingest calls
  `record_loads_from_calibration` (`model_metrics.py:645`) and
  `calibration_store.record_many` (`calibration.py:204`).
- **← `worker_agent`** ([[worker-fleet]]): produces calibration samples, relays
  eviction telemetry (`install_relay`, `evictions.py:1119`), dual-writes
  heartbeats.
- Sibling docs: [[worker-fleet]] (worker_agent's own enroll/admit/heartbeat
  half), [[serving-core]] (`managers/eviction.py`, `managers/serve/overrides.py`),
  [[api-routes]] (the flask routes that are "dumb consumers" of `worker_store`/
  `job_store`, per `workers.py`'s own docstring).

## 7. Key contracts / types
- `BusMessage` (`bus.py:43-61`) — frozen dataclass: `topic, id, ts, source,
  principal, channel, target, job_id, payload`; topic matches exact or
  `prefix.*`.
- `Job` (`jobs.py:200-267`) — lifecycle `pending → processing → streaming →
  {done,cancelled,failed,expired,discarded}` (`CANONICAL_STATUSES`/
  `TERMINAL_STATUSES`, `:49-60`); legacy `queued/running/completed/waiting/active`
  normalize on read/write.
- Priority-group record (`priority_groups.py:43-55`): `{id, name,
  members:[ordered model keys], workers:[ordered worker ids], enabled,
  created_at, updated_at, by}`; a member may nest via a `"group:<id>"`
  reference, expanded in place (`:217-256`).
- Block record (`blocklist.py`): global `{blocked, by, ts, note?}` under NS
  `models.blocked` (`:45`, unblock leaves an `operator_unblocked` tombstone,
  `:122-157`); per-(model×worker) `{blocked, why, ts, by}` under NS
  `models.blocked_workers` (`:204-246`).
- `placement_policy(model_key) -> (prefs: list[str], polite: bool,
  polite_by_worker: dict)` (`managers/serve/overrides.py:833`, re-exported
  `workers.py:2772`).
- Calibration sample wire fields (`calibration.py:75-79`): `model_key, engine,
  verdict, ctx_pct, needs_weights_bytes, needs_kv_bytes, need_total_bytes,
  n_gpu_layers, total_layers, vram_bytes, rss_bytes, load_seconds, device, ok,
  ts` — only `verdict='full'` rows feed the correction ratio.
- todo-keeper wire contract v1 (`todo_keeper.py:9-38`): dispatch
  `{"task":{"kind":"todo.add"|"todo.tidy","v":1,"vm",instruction/items}}`;
  result string `{"kind","v":1,"items":[...],"mode":"additive"|"proposal"}`.
- `model_metrics`/`model_calls` (Postgres db `hugpy`, `query_registry.py:73-192`):
  the rollup card (`model_name,quant,alloc_mode,worker` primary key, EMA
  `tok_per_s_avg`) vs. the append-only per-call ledger.

## 8. Gotchas, tech-debt & review findings

⚠ **The legacy/new split fails over invisibly.** `model_metrics.py:637-642`
swaps the module-level `model_metrics_store` singleton for
`abstract_toolserver.metrics.PgMetricsStore` at **import time**, inside a bare
`except Exception: pass` — no logging. Since this runs once per process and
every caller does `from .model_metrics import model_metrics_store` (a frozen
reference), whichever store wins at that one moment is what the process uses
for its entire life. `abstract_toolserver/metrics.py`'s `_ensure()`
(site-packages, `:81-145`) — the gate every read/write in that store passes
through — is *also* a bare `except Exception: return False`/`None`/`[]`, and
there is not one `log`/`logger` call anywhere in that 492-line module. A DB
fault at any layer (import, connect, first query) degrades every metric
silently to "no data yet," indistinguishable from a quiet fleet.

⚠ **Confirmed live during this review.** Direct read-only inspection
(2026-09-14 ~06:37 CDT) of db `toolserver` shows `load_metrics`/`call_metrics`/
`call_metrics_by_task`/`compute_actions` (19/22/16/14,636 rows respectively)
have taken **no writes since 2026-09-08 04:14:27** — 6 days stale. The local
SQLite fallback (`$PROJECTS_HOME/model_metrics.db`, on the **cold**
`llm_storage` mount — exactly the "WAL-on-virtiofs" problem the Postgres
cutover's own docstring says it was built to retire) shows scattered writes
through **2026-09-13 06:13** and is *also* silent since, including across a
fresh central restart at 2026-09-14 06:24 with 13+ minutes of confirmed active
model-resolution traffic in the log. Both metrics backends are currently
receiving nothing, and per the finding above nothing anywhere logs that fact.

ℹ **Not a missing-CREATE-privilege problem, contrary to what the DB-arm
docstring's framing might suggest.** `pg_namespace.nspacl` for `toolserver`'s
`public` schema explicitly grants role `toolserver` (the role central's
`hugpy.env`-dotenv fallback authenticates as, per `TOOLSERVER_ENV_FILE` in the
`7002_hugpy_api.service` unit) `=UC` (USAGE+CREATE), and
`information_schema.role_table_grants` shows both `toolserver` and `hugpy`
hold full SELECT/INSERT/UPDATE/DELETE on all four legacy tables. Because of
the swallow above, the actual 2026-09-08 trigger is not recoverable from
static code — worth a live follow-up (candidates: `managers/resolvers/remote.py
:239,258` simply not being hit under the current traffic shape, or a transient
fault the one time this process's singleton was decided, which then stuck for
the process's whole life).

△ **Three metrics/registry consumers, two Postgres databases, inconsistent
enable-gating.** `imports/src/model_index` (db `hugpy`) is gated by
`HUGPY_REGISTRY_DB=pg` (`client.py:17-20`). `heartbeat_db.py` (`:64-69,77`) and
`feeds.py` (`:88-93,96-114`) both reuse the *same* `DatabaseClient` class
against the *same* db `hugpy`, but neither checks that flag — they gate purely
on DSN presence (`heartbeat_db.dsn()`) or a bare try/except. `model_metrics.py`'s
legacy cutover (`:637-642`) is a third, independently-configured path into a
*different* database (`toolserver`) via a wholly separate pip package. All
three happen to agree today (every relevant env var is set on this box), but
nothing enforces that they can't diverge.

△ **`model_metrics.py`'s own local store degrades more quietly than its
siblings.** `ModelMetricsStore._ensure()` (`model_metrics.py:224-226`)
self-disables with a bare `except Exception: self._disabled = True; return
False` — no log call at all. Contrast `comms/calibration.py`'s `_note_failure`
(`:187-195`) and `comms/model_metadata.py`'s `_note_failure` (`:293-303`), both
of which log a warning per failure and an `error` when they self-disable. The
one store both metrics pipelines ultimately funnel through is also the one
with the least observability.

△ **`hugpy-todo-keeper.service` runs the pip wheel, not src — unlike
everything else in this doc.** Verified: its unit has no `PYTHONPATH` override,
and `python -m abstract_hugpy_dev.comms.todo_keeper_daemon` resolves
`abstract_hugpy_dev` from `/srv/hugpy/venv/.../site-packages` (installed
version **0.1.251**) while `src/abstract_hugpy_dev/pyproject.toml` is at
**0.1.257**. A `comms/todo_keeper.py`/`todo_keeper_daemon.py` src edit is not
live on `systemctl restart 7002_hugpy_api`; it needs a normal wheel publish +
`pip install -U` + this unit's own restart. (The two copies are currently
byte-size-identical, so there is no known live drift today — but the next edit
here will silently miss the daemon unless that's remembered.)

ℹ **Priority-group `workers` fallback, traced end to end.**
`managers/serve/overrides.placement_policy()` (`:833-874`) reads a model's own
`worker_prefs` first; only when that list is empty does it call
`comms.priority_groups.workers_for_key(model_key)` (`priority_groups.py:282-305`),
which returns the ordered `workers` allocation of the ENABLED group claiming
the model (alias-tolerant, nested-group-aware via `expand_members`/
`effective_workers`, `:228-279`), or `[]`. Per-model prefs always outrank the
group's; `workers.py`'s `_prefs_scope` (`:2814-2827`) then treats a non-empty
result as a HARD scope — refusing (with a logged reason) rather than spilling
onto the wider fleet when no listed worker is eligible.

ℹ **`workers.json`'s corruption handling is stricter than it needs to assume
its siblings match.** `_read_unlocked` (`workers.py:3027-3061`) explicitly
refuses to treat a non-empty-but-unparseable registry as an empty one (raises,
logs `error`), because this file is rewritten often enough that "heal to
empty" would have wiped the fleet at least once historically. `principals.py`
and `settings.py` were not checked for the same discipline in this pass — worth
a follow-up before assuming parity.

## 9. Deploy/run boundary
- Central (`7002_hugpy_api.service`) sets
  `PYTHONPATH=/srv/hugpy/src/abstract_hugpy_dev/src` (verified in the unit
  file) — `import abstract_hugpy_dev` resolves to the **src checkout**, so
  every file in this doc's scope under `comms/`, `imports/src/model_index/`,
  and `flask_app/.../workers.py` is live on `systemctl restart 7002_hugpy_api`,
  no wheel rebuild.
- **`hugpy-todo-keeper.service` is the exception** (see §8): no `PYTHONPATH`
  override, so it runs the pip-installed wheel (0.1.251) rather than src
  (0.1.257) — its own restart, plus a wheel publish, is required to pick up a
  src edit here.
- `abstract_toolserver` (the legacy metrics Postgres backend) is a wholly
  separate pip package/repo; its release cadence and install location
  (`/srv/hugpy/venv`) are independent of `abstract_hugpy_dev`'s deploy loop
  entirely — a fix to `PgMetricsStore._ensure()`'s silent swallow (§8) would
  ship there, not here.
- Workers (`worker_agent`) run the pip wheel per the fleet's normal
  self-update mechanism (README §GPU worker fleet & sharding) — a `comms/` src
  edit reaches `heartbeat_db`/`evictions`/`calibration` on the worker side only
  after a wheel publish + the worker's own `pip install -U` + re-exec.
