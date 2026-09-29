# PostgreSQL worker registry cutover

The central `WorkerStore` can use the existing Hugpy PostgreSQL registry as its
authoritative worker-state store. This includes each worker's latest storage
survey and its `store` / `counts_toward_budget` classification. Routine reads
load that persisted record; the worker still refreshes its disk survey every
60 seconds to detect changes made outside Hugpy. An actual reap request keeps
its fresh worker-side guards.

## Cutover

1. Stop central API writers so no older process can keep writing `workers.json`
   during the import.
2. Back up the current `workers.json` and PostgreSQL database.
3. Confirm `HUGPY_REGISTRY_DB=pg`, then set
   `HUGPY_WORKER_REGISTRY_BACKEND=pg` in the central API environment. Restart
   all central API processes.
4. The first registry access creates `hugpy_worker_registry` and imports the
   current `workers.json` once. PostgreSQL then holds one row per worker;
   subsequent worker and operator mutations use transactions and write only
   the worker rows that changed.
5. Confirm the worker count and IDs through the workers API, and confirm that
   a heartbeat and an operator change remain visible across API processes.

An explicitly constructed `WorkerStore(path=...)` continues to use the file
backend. This keeps isolated tests and tools from writing the live database.

## Recovery

If PostgreSQL is unavailable after cutover, registry mutations fail instead of
silently writing a divergent JSON file. Restore database service or export the
database payload to `workers.json` before deliberately switching back to the
file backend. The original JSON file is an import snapshot, not a live mirror.

The derived storage view is materialized into each PostgreSQL worker row when
its inputs change. It expires after at most 60 seconds so time-based pull
liveness, fleet policy, and model-size changes are eventually reflected even
without an operator action. Status reads reuse the stored view until then.
Reap approval always recomputes its guards from current worker state.

The model allocation view is also materialized per worker. Heartbeat telemetry
does not invalidate it; designation, allocation, memory capacity, or model
switch changes do. Its one-hour expiry covers model-file changes made outside
Hugpy. Ordinary workers API reads use the PostgreSQL copy and do not walk model
files to rebuild every assigned model's planned split.
