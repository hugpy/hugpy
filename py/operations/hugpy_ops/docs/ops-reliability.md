# Operations: sentinel, chaos, provisioner, keepers — mechanics

**Scope:** `hugpy_ops/sentinel/`, `hugpy_ops/chaos/`, `hugpy_ops/provisioner.py`,
`hugpy_ops/keeper.py`, `hugpy_ops/todo_keeper.py` + `todo_keeper_daemon.py`.
This is the operations half of the monolith's `curation-reliability.md`;
`discovery_dossier/` + `review/` are documented by `hugpy-curation`,
`downloader/` by `hugpy-storage`, `fleet_doctrine/` by `hugpy-fleet`.
**One-liner:** bounded operational consumers of the public APIs — a probe
that opens cases (sentinel), an exerciser that measures placement (chaos), a
scanner that enqueues declared-but-missing weights (provisioner), and two
model-driven keepers (terminal keeper, todo-keeper node).
**Owner process(es):** `sentinel/` is its own systemd timer
(`deploy/hugpy-sentinel.timer`), a pure HTTP client of central. `chaos/` is a
manual operator CLI, never a daemon. `provisioner.py` runs by hand
(`hugpy-provisioner`) and inside every sentinel pass. `todo_keeper_daemon.py`
is a long-lived unit (`deploy/hugpy-todo-keeper.service`). `keeper.py` is an
interactive terminal session. **None of these modules are imported by
normal server startup** (`hugpy_server` lists `hugpy_ops` only as an
optional dependency and never imports it).

## 1. Purpose & responsibilities

- **Probe** — `sentinel/` watches central's read surfaces for bound-exceeded
  conditions (stalled jobs, offline workers, version skew, lost capabilities,
  hard-fail streaks, missing weights, a cold pilot light), opens deduplicated
  cases, spawns one bounded diagnosis agent per new case.
- **Exercise** — `chaos/` randomly (or, via `sweep`, deterministically) fires
  real generations across model × worker × alloc-mode × ctx% to compare
  predicted vs. measured placement, for a sibling learner (out of scope).
- **Fetch-declared** — `provisioner.py` detects declared-but-missing weights
  across the engine registry (comfy + tasks rows) and the studio zoo and
  enqueues them on `hugpy_storage`'s download queue. Detection + enqueue +
  provenance only; there is no download engine here.
- **Keep** — `keeper.py` is a REPL in which a central-served model keeps a
  machine (or an LXD instance) through a shell; `todo_keeper*` is the P3
  agent node answering the console's todo drawer (`todo.add` / `todo.tidy`).
- All of them mutate fleet state only through explicit, reversible, gated
  paths: the sentinel's remedy whitelist (default OFF), chaos's
  snapshot/apply/restore over the operator-gated `/assign`, the provisioner's
  enqueue (cancelable job), the keeper's bounded action loop.

## 2. Key modules (file → responsibility)

| subsystem | file | responsibility |
|---|---|---|
| sentinel | `checks.py` | detection — pure functions over central's 3 HTTP surfaces + local history; `check_workers` attaches `hugpy_ops.versions` to a skew case |
| sentinel | `cases.py` | SQLite case store, one OPEN case per fingerprint (partial unique index = structural dedupe) |
| sentinel | `runner.py` | one pass: detect → open/touch cases → spawn diagnosis agent, or the `weight_missing` fast path |
| sentinel | `remedies.py` | the whitelist — typed, reversible, gated (`remedies` / `downloads`), prod-worker-excluded |
| sentinel | `settings.py` | env knobs; central via `hugpy_platform.central_base_url`, state root via `hugpy_platform.hugpy_state_dir()` |
| sentinel | `__main__.py` | `hugpy-sentinel run-once | status | record-scorecard` |
| chaos | `runner.py` | random exerciser: draw → price → apply → fire → measure → restore; `hugpy-chaos` |
| chaos | `sweep.py` | k7 deterministic offload speed-cliff sweep (`hugpy-chaos sweep`) |
| chaos | `assortment.py` | enumerate the live cube, draw seeded combos, feasibility; alloc-mode vocabulary from `hugpy_engine.alloc_modes` |
| chaos | `alloc.py` | snapshot/apply/restore a model's per-worker spill, verified |
| chaos | `client.py` | the one HTTP seam onto central (stdlib-only, injectable); base URL via `hugpy_platform.central_base_url` |
| chaos | `observe.py` | build the predicted (pre-fire) / measured (post-fire) observation halves |
| chaos | `schema.py` | `chaos-obs/1` — the observation contract shared with the (out-of-scope) learner |
| (root) | `provisioner.py` | `Want` detection over `Registries` (engine `MODELS`, studio `MODEL_REGISTRY`, storage `resolve_model_dir`), ComfyUI-Manager catalog resolution, `enqueue` on `hugpy_storage.downloader.queue.enqueue_download` |
| (root) | `keeper.py` | stdlib-only terminal keeper; action loop bounded by `--max-steps`, per-command 300 s timeout, observation truncation |
| (root) | `todo_keeper.py` | the PURE todo-keeper contract (parse task → prompt → parse reply → envelope); refuses corrupt payloads |
| (root) | `todo_keeper_daemon.py` | enroll once (token persisted 0600 under `hugpy_platform.config_dir()`), heartbeat, pull, report; 409 = already recorded |
| (root) | `versions.py` | `hugpy-fleet` / `hugpy-server` installed versions via `importlib.metadata`, lazily |

## 3. Entry points

- **`hugpy-sentinel`** — `hugpy-sentinel.timer` (`OnBootSec=2m`,
  `OnUnitActiveSec=10m`) → oneshot `.service` → `hugpy-sentinel run-once`;
  plus `status` / `record-scorecard` (`sentinel/__main__.py`).
- **`hugpy-chaos`** — manual only: `hugpy-chaos [--dry-run] [--rounds N] ...`
  or `hugpy-chaos sweep [--plan] [--top-n N] ...` (`chaos/runner.py:main`
  dispatches the subcommand). No systemd unit references chaos.
- **`hugpy-provisioner`** — `hugpy-provisioner [--apply] [--floor-gb N]
  [--json]` (default DRY RUN) and `hugpy-provisioner catalog [--type X]
  [--missing]`; `provisioner.wants()` is also called by every sentinel pass
  (`checks.default_weight_wants`).
- **`hugpy-keeper`** — `hugpy-keeper --model <id> [--exec lxc:<name>]
  [--bridge <id>]`; `python3 hugpy_ops/keeper.py` also works without the
  ecosystem installed (inline `central_base_url` mirror).
- **`hugpy-todo-keeper`** — `hugpy-todo-keeper.service` (`Type=simple`,
  `Restart=always`) → `todo_keeper_daemon.main`.
- The thin `hugpy` meta CLI dispatches `hugpy keeper|sentinel|chaos|provision`
  to the same `main` functions.

## 4. Data flow

**A. Sentinel: detect → case → diagnose/remedy**
1. `checks.detect()`: 3 GETs (`/llm/jobs?live=0`, `/llm/workers`,
   `/oracle/capabilities`) + local scorecard history + the provisioner scan →
   rule functions → `Anomaly` list.
2. `runner.run_once()`: each anomaly through `CaseStore.open_or_touch()` —
   the partial UNIQUE index on `fingerprint WHERE state != 'closed'` makes
   dedupe a DB guarantee.
3. New non-`weight_missing` cases: `spawn_agent_for_case()` runs
   `hugpy-agent case <brief> --case-dir <dir>` bounded (900 s default,
   document-only policy), parses the JSON report → `documented` / `escalated`.
4. `weight_missing` cases fast-path: `remedies.execute(enqueue_download)`
   POSTs `{central}/llm/repos/download` — the console's own add-models
   enqueue — no agent spawned.
5. Other remedies (unload/relaunch, chat cancel) stay behind
   `HUGPY_SENTINEL_REMEDIES` (default OFF); downloads have their own gate
   (`HUGPY_SENTINEL_DOWNLOADS`, default ON); worker `ae` is structurally
   excluded from remedies in both `eligible()` and `execute()`.

**B. Chaos: random exercise / deterministic sweep**
1. `draw_combo()` picks a servable model, an already-assigned worker, a
   framework-gated alloc mode, a ctx% — seeded, reproducible.
2. `observe.build_predicted()` prices the combo via `/models/<key>/meta`
   before firing; infeasible combos are skipped, never fired.
3. `alloc.snapshot()` → `alloc.apply()` (the **only** mutating call,
   operator-gated `/assign`) → `client.chat_stream()` fires a small real
   generation over the console's public path → `observe.build_measured()`.
4. `alloc.restore()` writes the prior spill back and verifies byte-identical;
   every trial only touches an already-assigned pair, so restore is a
   write-back, never an unassign.
5. `sweep` ranks top-N used GGUF models, forces each cold between grid points
   (`/unload`), times generations per VRAM share, detects the tok/s cliff.
   Observations land in `<models_root()>/comms/chaos/observations.jsonl`
   (`HUGPY_CHAOS_OUT_DIR` overrides); the markdown report defaults to
   `<out-dir>/OFFLOAD-CLIFF-<date>.md`.

**C. Provisioner: scan → want → enqueue**
1. `wants(root, registries)` runs three independent scans (one failing never
   hides the others): `comfy_wants` (curated `framework == "comfy"` rows +
   the §5b id_lock assets, ComfyUI-Manager catalog-first resolution),
   `studio_wants` (studio zoo rows whose weights dir holds no bytes),
   `tasks_wants` (remaining curated rows through storage's read-through
   resolver so a legacy layout is never re-wanted).
2. A `Want` is `resolved` only when its hub id is PROVEN by the registry row /
   manifest / catalog — never guessed; non-HF sources are surfaced in the note.
3. `enqueue(want)` refuses explicitly (`unresolved-source`, `duplicate`
   against the live job mirror, `disk-floor`) or calls
   `hugpy_storage.downloader.queue.enqueue_download` with
   `transport="provisioner"` — the same queue, daemon, dedupe and `/jobs` view
   as a human-clicked download.

**D. Keepers**
- `keeper.run_agent_turn()` streams the model, runs at most `--max-steps`
  fenced `action` commands (each `subprocess.run` capped at 300 s, output
  truncated to `KEEPER_OBS_LIMIT`), and returns the first reply that carries
  no action block.
- `todo_keeper_daemon.TodoKeeperNode`: enroll once (`POST /agent/register`,
  token persisted 0600), heartbeat, pull with a durable monotonic cursor,
  answer through `todo_keeper.handle_task` (pure), report; a 409 on report
  means "already recorded" and advances the cursor.

## 5. State, persistence & invariants

- **sentinel/**: SQLite case store (`$HUGPY_SENTINEL_DIR/cases.db`), `cases.md`
  ledger (append-only), `history.jsonl` (pruned past 256 KB). Default root
  `hugpy_platform.hugpy_state_dir()/sentinel`; the unit pins
  `HUGPY_SENTINEL_DIR`.
- **chaos/**: append-only `observations.jsonl` (`chaos-obs/1`) + per-run
  manifest JSON under `<models_root()>/comms/chaos/`. No DB; HTTP only, and
  only `/assign` + `/unload` mutate anything, always restored.
- **provisioner**: no state of its own — it reads registries and the job
  mirror (`hugpy_control.jobs.job_store`) and writes only through the queue.
- **todo-keeper**: `{node_id, token, cursor}` at
  `hugpy_platform.config_dir()/todo-keeper.json` (0600, dir 0700;
  `HUGPY_TODO_KEEPER_STATE` overrides). A corrupt file is refused, never
  clobbered.
- **keeper**: none (conversation lives in the process).

## 6. Cross-package edges (all through public surfaces)

- `sentinel/` → `provisioner.py` → `hugpy_storage`: `checks.default_weight_wants()`
  calls `provisioner.wants()`; the fast path POSTs `{central}/llm/repos/download`
  over HTTP; `provisioner.enqueue()` calls `enqueue_download` directly when run
  by hand.
- `provisioner.py` → `hugpy_engine.config.models.models_config.MODELS`,
  `hugpy_video.intel.studio.MODEL_REGISTRY`, `hugpy_storage.model_paths.resolve_model_dir`,
  `hugpy_control.jobs.job_store` — all public names, all loaded lazily inside
  the scan, all replaceable through `provisioner.Registries` for tests.
- `chaos/` → `hugpy_engine.alloc_modes`: one shared alloc-mode vocabulary,
  never redefined locally.
- Everything → `hugpy_platform`: `central_base_url`, `hugpy_state_dir`,
  `config_dir`, `models_root`.
- `sentinel/`, `chaos/`, `keeper.py`, `todo_keeper_daemon.py` talk to central
  over plain HTTP only — none import `hugpy_server`.

## 7. Key contracts / types

- `sentinel.cases.Anomaly{fingerprint, kind, severity, evidence}` / `Case`
  (states `open → agent_running → documented | escalated → remedied | closed`)
  / `sentinel.remedies.Remedy{name, method, url_template, applies_to,
  reversible, required_params, gate}`.
- `chaos.schema` — `chaos-obs/1`: `{combo, predicted, measured, restore}` +
  optional `sweep`; `blank_observation()` / `validate_observation()`.
- `provisioner.Want{registry, name, reason, dest, hub_id, filename, include,
  framework, est_bytes, note}` — `resolved` iff `hub_id` is provable;
  `provisioner.Registries{curated, studio, resolver, catalog, folders,
  studio_weights_root}`.
- `todo_keeper` contract v1: dispatch `{"task": {"kind": "todo.add"|"todo.tidy",
  "v": 1, ...}}`, result JSON string `{"kind", "v": 1, "items", "mode":
  "additive"|"proposal"}`; the node never mints `id`/`by`/`ts`.

## 8. Gotchas & open items

1. The sentinel's `worker_version` case reports the `hugpy-fleet` /
   `hugpy-server` versions installed **beside the sentinel**, which need not
   be what central runs; central's `version_ok` stays the authority.
2. `chaos` `--env-file` still defaults to the station share path
   (`/srv/share/projects/hugpy/d-env/env`) when `HUGPY_ENV_FILE` is unset —
   a deployment convention, not a platform path.
3. The provisioner's studio dest is the studio weights root
   (`STUDIO_WEIGHTS_ROOT` or `<DEFAULT_ROOT>/video_intel/studio/weights`) but
   the transfer plane lands in its own layout; the note on each studio want
   spells the symlink to make afterwards.
4. "card" means a GPU-bearing worker throughout `chaos/` (not a review
   criteria card, not a README card digest).

## 9. Deploy/run boundary

Each tool is a console script of the `hugpy-ops` wheel installed into a venv
on the box that runs it; nothing here rides central's live `src` checkout.
`deploy/hugpy-sentinel.{service,timer}` and `deploy/hugpy-todo-keeper.service`
are the units (edit `ExecStart`'s venv path per box); `deploy/bin/*` are
superseded wrappers kept for uninstalled checkouts.
