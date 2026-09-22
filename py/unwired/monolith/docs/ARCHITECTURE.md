# ARCHITECTURE.md — the `abstract_hugpy_dev` package

Deep map of the importable package. Load this when working *inside* the package;
for host layout / services / deploy see `/srv/hugpy/README.md`, `SERVICES.md`, and
`/srv/hugpy/CLAUDE.md` (the session router). Paths below are relative to
`/srv/hugpy` unless noted.

## Package identity

- Import name: **`abstract_hugpy_dev`** (src-layout).
- Importable root: `src/abstract_hugpy_dev/src/abstract_hugpy_dev/` — note the triple
  nesting (host dir / repo / `src/` / package).
- Version: single source of truth in `src/abstract_hugpy_dev/pyproject.toml`;
  `__init__.py` reads it back from installed metadata with a hardcoded fallback (which
  has lagged real releases before — don't trust the fallback constant).
- What it is (pyproject): self-hosted LLM console — model registry & downloads,
  streaming chat, OpenAI-compatible `/v1` API with on-site keys, and a GPU worker fleet
  with cross-machine RPC sharding.

## Entry points

Console scripts (pyproject `[project.scripts]`):

- **`hugpy`** → `abstract_hugpy_dev.cli:main` (`cli.py`). Subcommands:
  - `serve` → `_serve()` builds the Flask app (`flask_app/wsgi_app.py`) and serves it via
    gunicorn/waitress (dev-server fallback). This is **central** (:7002) — API + web console
    in one process.
  - `worker` → `worker_agent/agent.py` (also `python -m abstract_hugpy_dev.worker_agent`).
  - `bot` → `bot/bot.py` (`HugpyBot`).
  - `keeper` → `keeper.py`. Plus `chat`, `install-engine`, `install-deps`, `reclassify-images`.
- **`hugpy-downloader`** → `downloader/daemon.py:main` (backs `hugpy-downloader-dev.service`).
- Module entries with `__main__.py`: `worker_agent`, `gguf_worker`, `review`, `sentinel`,
  `downloader`, `phone_brick`, `chaos`.
- `comms/` is a **library substrate** (imported by bot/keeper/worker/daemons), not a process.

`central.py` holds the single source of truth for the central base URL every arm dials
(default `http://127.0.0.1:7002`).

## Subsystems

Paths below sit under `src/abstract_hugpy_dev/src/abstract_hugpy_dev/`.

### API / console
- `flask_app/` — Flask app package. `wsgi_app.py` = WSGI entry; `app/` = the real app.
- `flask_app/app/routes/` — ~35 blueprints: `v1_routes.py` (OpenAI `/v1`), `chat_routes.py`,
  `worker_routes.py`, `fleet_routes.py`, `metrics_routes.py`, `oracle_routes.py`,
  `video_routes.py`, `review_routes.py`, `comms_routes.py`, `discord_routes.py`,
  `phone_brick_routes.py`, `pypi_routes.py`, …
- `flask_app/app/` — auth layers: `member_auth.py`, `operator_auth.py`, `video_auth.py`; `endpoints_*.py`.
- `console_dist/` — **prebuilt** web UI bundle shipped in the wheel (NOT source; edit `src/react/ui/`).

### Model serving / registry (largest subsystem)
- `managers/` — serving & generation managers:
  - `serve/` — `serve.py`, `supervisor.py`, `slot_agent.py`, `slots.py`, `model_cache.py`,
    `hot_cache.py`, `policy.py`, `profiles.py`.
  - directly under `managers/`: `eviction.py`, `spill.py`, `alloc_modes.py`, `draft_models.py`.
  - `dispatch/`, `generate/` (`coder.py`, `generate_runner.py` — `generate_runner2.py` is DEAD, no importers),
    `fleet/`, `chat_context/`,
    `resolvers/`, and media managers `comfy/`, `imagegen/`, `video_gen/`, `vision/`,
    `whisper_model/`, `tts/`, `summarizers/`, `embed/`, `llama/`.
- `engine/` — native llama.cpp provisioning (`build.py`, `fetch.py`, `resolve.py`).
- `model_sync.py` — pull whole model dirs from a central node.
- `provisioner.py` — detect declared-but-missing weights across registries and enqueue downloads.
- `model_battery.py` — per-session render/generation battery recorder.
- `_platform/` — `hardware.py`, `binaries.py`, `paths.py`, `procutil.py`, `async_runtime.py`, `client_liveness.py`.

### Worker fleet
- `worker_agent/` — full standalone GPU worker (`agent.py`, `__main__.py`, `provision.py`,
  `budget.py`, `flex.py`, `studio_render.py`, `comfy_watchdog.py`, `aptitude/`, `deploy/`, `bootstrap.sh`).
- `gguf_worker/` — slim GGUF-only worker for constrained boxes (Termux/ARM).
- `phone_brick/` — phone-as-worker (`orchestrator.py`, `worker.py`, `rpc_backend.py`,
  `consensus.py`, `detector.py`, `registration.py`).

### Comms / bot / keeper
- `comms/` — shared substrate: `bus.py` (typed frozen message/control bus), `jobs.py`,
  `heartbeat_db.py`, `evictions.py`/`evict_policy.py`, `principals.py`, `priority_groups.py`,
  model metadata/metrics/status caches, `blocklist.py`, `calibration.py`, `feeds.py`,
  `todo_keeper.py` (pure) + `todo_keeper_daemon.py` (stdlib-only systemd daemon).
- `bot/` — Discord arm over HTTP: `bot.py`, `cogs/`, `hugpy_client.py`, `streamer.py`, `config.py`, `prefs.py`.
- `keeper.py` — terminal machine-warden REPL (health, services, shell).

### Video / creative pipeline
- `video_intel/` — headless video-intelligence backbone: frozen schemas (`media_schema.py`,
  `job_schema.py`, `scene_schema.py`, `movie_schema.py`, identity/crop/frame/audio), `media_bus.py`,
  `media_store.py`, `runners/`, `studio/`, `reservation/`.
- `oracle/` — declarative eliminate-then-rank planning: `pipeline(s).py`, `dag_runtime.py`,
  `router.py`/`routing_matrix.py`, `selection.py`, `validator.py`, `steward.py`, `prompt_compiler.py`,
  `screenplay.py`, `storyboard.py`, `production.py`, `postproduction.py`, `spatial*.py`, `scorecard.py`.
- `discovery_dossier/` — research/score HF models into dossiers.
- `review/` — search→screen→download→smoke→judge pipeline for candidate models.

### Reliability / ops
- `sentinel/` — bounded per-case health/diagnostic agent runner.
- `chaos/` — sweep models×cards×alloc-modes×ctx to exercise the fleet.
- `fleet_doctrine/` — fleet policy/diagnosis (`doctrine.py`, `doctor.py`).
- `downloader/` — model-download daemon in its own process (`daemon.py`, `engine.py`, `queue.py`, `presence.py`).

### Shared / compat
- `utils/` — `no_think.py`, `json_scavenge.py`, `text/`, `pdfs/`, `seo/`.
- `imports/` — centralized dependency/config re-export layer (star-imported by `__init__`).
- `_compat_pydantic.py` — pydantic shim for no-pydantic-core platforms (phone/ARM).

## Per-subsystem mechanics docs

Deeper, self-contained deep dives live in `docs/mechanics/` (within this repo,
`src/abstract_hugpy_dev/`), one doc per subsystem, from a 2026-09-14 code-review pass.
Each follows `_TEMPLATE.md`: purpose, key modules, entry points, data flow, state &
invariants, cross-subsystem edges, contracts, review findings (§8), deploy boundary (§9).
Read the one doc for the area you're touching rather than the tree; the orientation stack is
`/srv/hugpy/CLAUDE.md` → this file → these mechanics docs → the code.

| mechanics doc | subsystem it covers (paths under `src/abstract_hugpy_dev/src/abstract_hugpy_dev/` unless noted) |
|---|---|
| `docs/mechanics/api-routes.md` | Central API & console — `flask_app/` (`wsgi_app.py`, `app/routes/*`, `member_auth`/`operator_auth`/`video_auth`, `endpoints_*`) |
| `docs/mechanics/serving-core.md` | Model serving / slots / placement — `managers/serve/*`, `managers/{eviction,spill,alloc_modes,draft_models}.py`, `managers/dispatch/`, `managers/resolvers/` |
| `docs/mechanics/engine-generation.md` | Engine + generation lanes — `engine/`, `managers/generate/`, `managers/llama/`, `provisioner.py`, `model_sync.py`, `model_battery.py` |
| `docs/mechanics/media-gen.md` | Image/video/audio/vision generation — `managers/{comfy,imagegen,video_gen,vision,whisper_model,tts,summarizers,embed}/` |
| `docs/mechanics/worker-fleet.md` | Worker agents + enroll/admit/assign — `worker_agent/`, `gguf_worker/`, `phone_brick/` (+ central `/llm/workers/*`) |
| `docs/mechanics/comms-index.md` | Comms substrate + model registry DB — `comms/*`, `imports/src/model_index/*`, `flask_app/app/functions/imports/utils/workers.py` |
| `docs/mechanics/video-oracle.md` | Video-intelligence + oracle planning — `video_intel/`, `oracle/` |
| `docs/mechanics/curation-reliability.md` | Model curation + reliability daemons — `discovery_dossier/`, `review/`, `sentinel/`, `chaos/`, `fleet_doctrine/`, `downloader/` |
| `docs/mechanics/cli-agents-platform.md` | CLI/bot/keeper + platform/utils + py side-pkgs — `cli.py`, `bot/`, `keeper.py`, `_platform/`, `utils/`, `imports/`, `src/py/*` |
| `docs/mechanics/frontends.md` | Web/desktop UIs — `src/react/*` (console, agents/media/video UIs, `ui_shared/`) and `src/station-app/` |

Index: `docs/mechanics/README.md`. Consolidated review findings across all ten: `docs/mechanics/FINDINGS.md`.

## Tests

- `src/abstract_hugpy_dev/tests/` — ~275 flat `test_*.py` + `tests/comms/`; shared `conftest.py`.
- Run with **pytest** from the repo root (`cd src/abstract_hugpy_dev`). No custom pytest config in
  `pyproject.toml` (only an `ocr`-extra dep), so pytest uses defaults over `tests/`.
- Daemons' systemd units: `src/abstract_hugpy_dev/deploy/` (`hugpy-steward.*`, `hugpy-sentinel.*`,
  `hugpy-review@.*`, `hugpy-todo-keeper.service`, `hugpy-downloader-dev.service`, `deploy/user/`).
- Package doctrine notes: `src/abstract_hugpy_dev/deploy/{ALLOCATION-MODES,SENTINEL,STATE-MODEL,WORKER-BOOT-PREWARM,WORKER-WILDCARD}.md`.

## Stale / noise inside this repo (skip; do not edit as live)

Cleaned up 2026-09-14 — `venv/`, `backups/`, and stray logs were moved to
`/srv/hugpy/archive/noise-cleanup-20260914/` (see its `MANIFEST.md`; restore with `mv`).
All are git-ignored (`.gitignore`), so if regenerated, skip them again:

- `venv/` (~6.5 GB), `.venv/` — real virtualenvs; if you need a dev env here, recreate it.
- `backups/` — dated duplicate copies of `bot/`, `comms/`, `managers/`, `console_dist/`. Dead.
- Stray logs (`err.log`, `hugpy_flask.log`, `smoke.log`) and `*.bak-remount-*`, `*.save` files.
- `build/`, `dist/*.whl`+`*.tar.gz`, `*.egg-info/` — build artifacts, **owned by `solcatcher`**
  (the pyit publish pipeline regenerates them each publish); left in place, git-ignored, harmless.
- Possible duplicate impls to disambiguate before editing: `managers/generate/generate_runner.py`
  vs `generate_runner2.py`; `oracle/pipeline.py` vs `pipelines.py`; `oracle/repair.py` vs
  `repair_controller.py`; `oracle/spatial.py` vs `spatial_eval.py`/`spatial_sources.py`;
  top-level `phone_brick/` vs `managers/phone_brick_orchestrator/`.
