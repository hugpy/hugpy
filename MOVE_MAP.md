# Hugpy extraction move map

This tracks what has already left `abstract_hugpy_dev` and what still needs to
move. The detailed ownership list is
[`py/partition.toml`](py/partition.toml); the architectural reasoning is in
[`PARTITION.md`](PARTITION.md).

## Status legend

- **Done** — standalone source and packaging already exist; no implementation
  remains to extract from the monolith.
- **Boundary** — standalone public API exists, but its implementation still
  calls into the monolith.
- **Skeleton** — destination package exists, but most runtime source remains in
  the monolith.
- **Unmoved** — destination is planned but has not been created.
- **Retire** — remove after importers move; do not publish as a package.

## Current state

| Status | Distribution | Current location | Remaining work |
|---|---|---|---|
| **Done** | `abstract-identity` | `py/cinema/abstract_identity` | Keep its existing library/service boundary. Only replace old monolith references with its public client/API. |
| **Done** | `hugpy-agent` | `py/inference/hugpy_agent` | Keep independent. Move only the central `/agent/*` HTTP adapter to `hugpy-server`. |
| **Done** | `abstract-toolserver` | `py/tools/abstract_toolserver` | Keep independent; consume through its library/service contract. |
| **Done** | `abstract-apply` | `py/tools/abstract_apply` | Keep as an ecosystem consumer. |
| **Boundary** | `hugpy-engine` | `py/hugpy_engine` now; target `py/inference/hugpy_engine` | Public catalog/allocation/engine/runtime/query API is extracted. Move the implementation listed below, then delete `LegacyHugpyBackend`. |
| **Skeleton** | `hugpy-video` | `py/cinema/hugpy_video` | Config/package skeleton exists. Move `video_intel`, video runners and schemas. |
| **Unmoved** | All other target distributions | See the batches below | Create package, move owned source/tests, replace imports, build wheel. |

The temporary `abstract_hugpy_dev/core` facade has already been removed. The
affected chat and `/v1` adapters import `hugpy_engine` explicitly.

## Destination tree

```text
py/
├── foundation/
│   ├── hugpy_platform/       UNMOVED
│   └── hugpy_control/        UNMOVED
├── inference/
│   ├── hugpy_engine/         BOUNDARY (currently py/hugpy_engine)
│   ├── hugpy_media/          UNMOVED
│   └── hugpy_agent/          DONE
├── storage/
│   └── hugpy_storage/        UNMOVED
├── fleet/
│   └── hugpy_fleet/          UNMOVED
├── cinema/
│   ├── abstract_identity/    DONE
│   ├── hugpy_video/          SKELETON
│   └── hugpy_oracle/         UNMOVED
├── curation/
│   └── hugpy_curation/       UNMOVED
├── operations/
│   └── hugpy_ops/            UNMOVED
├── integrations/
│   └── hugpy_discord/        UNMOVED
├── services/
│   └── hugpy_server/         DONE
├── meta/
│   └── hugpy/                UNMOVED
├── tools/
│   ├── abstract_apply/       DONE
│   ├── abstract_toolserver/  DONE
│   └── hugpy-station/        KEEP SEPARATE
└── unwired/                  RETIRED/QUARANTINED CODE ONLY
```

## Remaining move batches

### 1. `hugpy-platform` — unmoved

Move:

- `_platform/`
- `_compat_pydantic.py`
- `central.py`
- `imports/src/constants/`
- `imports/src/_compat.py`
- `imports/src/except_utils.py`
- `imports/src/module_imports.py`
- the genuinely generic functions from `imports/src/utils.py`

Before cutting: remove any imports from these files back into model, manager, or
Flask code. The resulting base must be stdlib-first and domain-free.

### 2. `hugpy-control` — unmoved

Move:

- `comms/bus.py`
- `comms/calllog.py`
- `comms/feeds.py`
- `comms/jobs.py`
- `comms/principals.py`
- `comms/settings.py`
- `comms/shared.py`
- `comms/task_templates.py`

Before cutting: inject storage paths and remove the current imports from control
state into engine/fleet implementations. Domain-specific `comms` files go to
their owners below.

### 3. `hugpy-engine` implementation — boundary exists, implementation unmoved

Move behind the existing API:

- `engine/`
- `managers/alloc_modes.py`, `spill.py`, `eviction.py`, `draft_models.py`
- `managers/chat_context/`
- `managers/dispatch/`
- `managers/generate/` except retired files
- `managers/llama/`
- `managers/resolvers/`
- `managers/serve/`
- `imports/config/`
- `imports/apis/call_api.py`, `get_module.py`, `serve/`, `systemd_units.py`
- `imports/src/model_index/`
- model classification/election/adapter helpers
- inference chat/event/model/runner/task/metadata schemas
- `utils/json_scavenge.py`, `utils/no_think.py`

Cut blockers:

- engine currently imports media, storage, fleet, server, control, Oracle and
  video code;
- replace those with runner registration, catalog/storage protocols, placement
  providers and event callbacks;
- after the last monolith call is gone, delete
  `hugpy_engine.backends.LegacyHugpyBackend` and its optional monolith extra.

### 4. `hugpy-storage` — unmoved

Move:

- `downloader/`
- `model_sync.py`
- `provisioner.py`
- `comms/hf_metadata.py`
- `comms/model_metadata.py`
- `comms/model_physical.py`
- `comms/model_status_cache.py`
- download, Hugging Face, reclassify and reconcile API helpers

Cut blocker: storage and engine currently import each other. Storage must own
artifact transfer and emit catalog invalidation; engine must own discovery and
consume that event without storage importing engine internals.

### 5. `hugpy-media` — unmoved

Move:

- `managers/comfy/`
- `managers/embed/`
- `managers/imagegen/`
- `managers/keywords/`
- `managers/summarizers/`
- `managers/tts/`
- `managers/vision/`
- `managers/vision_analysis/`
- `managers/whisper_model/`
- media-specific schemas and chunking
- `utils/pdfs/`, `utils/seo/`, `utils/text/`
- `model_battery.py`

Cut blockers: media currently imports fleet, storage, video and meta helpers.
Convert runners to explicit engine plugins and put heavy dependencies behind
extras/lazy imports.

### 6. `hugpy-video` implementation — skeleton exists, implementation unmoved

Move:

- `video_intel/`
- `managers/video/`
- `managers/video_gen/`
- `imports/src/schemas/video_schemas.py`
- `comms/studio_assist_log.py`

Cut blockers: replace imports of Flask/server helpers with injected services.
Remove all four video-to-Oracle imports; video executes immutable specs and
publishes results, while Oracle owns planning.

### 7. `hugpy-oracle` — unmoved

Move all of `oracle/`.

Cut blockers: Oracle currently imports server, fleet, media, curation and meta
code. Keep its pure planning/evaluation core; express runtime needs through
`hugpy-engine`, `hugpy-video`, injected stores and public protocols.

### 8. `hugpy-fleet` — unmoved

Move:

- `worker_agent/`
- `gguf_worker/`
- `phone_brick/`
- `fleet_doctrine/` and `fleet_runbook.json`
- `managers/fleet/`, `managers/phone_brick_orchestrator/`
- PID/token reporting helpers
- fleet-specific `comms` files: agent nodes, blocklist, calibration, eviction
  policy/events, heartbeat DB, model metrics and priority groups

Cut blockers: the full worker imports engine implementation heavily. Retain that
dependency, but load media/video worker capabilities through optional plugins.
Move the central and worker wire DTOs together so their contracts cannot drift.

### 9. `hugpy-curation` — unmoved

Move:

- `discovery_dossier/`
- `review/`

Cut blockers: replace direct fleet/server/control imports with public clients or
injected services. Review downloads must go through `hugpy-storage` rather than
its current second download path.

### 10. `hugpy-ops` — unmoved

Move:

- `chaos/`
- `sentinel/`
- `keeper.py`
- `comms/todo_keeper.py`
- `comms/todo_keeper_daemon.py`

Cut blocker: these tools must become leaf consumers of engine, fleet, curation
and control APIs. Normal server import must never start or import them.

### 11. `hugpy-discord` — unmoved

Move all of `bot/`.

Cut blocker: remove its one direct engine import. The bot should remain an HTTP
client of a configured Hugpy server and carry only the Discord optional
dependency.

### 12. `hugpy-server` — unmoved

Move:

- `flask_app/`
- `console_dist/` only as generated package data, built from `react/`

Move last because the server is the composition root. Route modules become thin
HTTP adapters over the extracted packages. Keep API keys, member/operator auth,
SSE, middleware, route prefixes and static mounts here.

React build ownership:

- `react/ui` → `/`
- `react/agents_ui` → `/fleet`
- `react/media_intelligence_ui` → `/media`
- `react/video_intelligence_ui` → `/video`

### 13. `hugpy` meta distribution — unmoved

Move:

- `cli.py`
- `hpy.py`
- final version/export surface from `__init__.py`
- install-profile/task-extra declarations now in `managers/task_deps.py` and the
  monolith `pyproject.toml`

This package only dispatches commands and defines extras. It must not contain
model, route, worker or media implementations.

## Retire rather than move

These paths must not become packages:

- the `imports/**/__init__.py` and `imports.py` wildcard-export chain;
- `managers/__init__.py` and `managers/imports.py`;
- `utils/__init__.py` and `utils/imports.py` wildcard shims;
- `imports/src/init_imports.py` and `standalone_utils.py` after their symbols have
  explicit owners;
- `console_dist.bak-shardpage/` and the backup fleet runbook;
- `get_vids.py`;
- confirmed dead `managers/falconsai/`, `generate_runner2.py`, and
  `coder_guff.py`.

Put any retained historical copy under `py/unwired/`; nothing there is imported,
built or published.

## Repository assets outside the import package

The split also owns the surrounding tests, service files and documentation:

| Current path | Destination owner |
|---|---|
| `tests/` | Move each test with the implementation it exercises. Cross-package HTTP and installed-wheel tests go to `hugpy-server`; final ecosystem smoke tests go to the `hugpy` meta package. |
| `deploy/hugpy-downloader-dev.service` | `hugpy-storage` |
| `deploy/hugpy-review@.*` and `deploy/user/hugpy-review@.*` | `hugpy-curation` |
| `deploy/hugpy-sentinel.*`, `hugpy-todo-keeper.service`, `bin/hugpy-chaos`, `bin/hugpy-todo-keeper` | `hugpy-ops` |
| `deploy/hugpy-steward.*` | `hugpy-oracle` |
| `deploy/ALLOCATION-MODES.md`, `STATE-MODEL.md`, `WORKER-BOOT-PREWARM.md`, `WORKER-WILDCARD.md` | Split between `hugpy-engine` and `hugpy-fleet`; retain one canonical copy per subject. |
| `docs/mechanics/engine-generation.md`, `serving-core.md` | `hugpy-engine` |
| `docs/mechanics/media-gen.md` | `hugpy-media` |
| `docs/mechanics/worker-fleet.md`, `comms-index.md` | Split between `hugpy-fleet`, `hugpy-control`, and `hugpy-storage`. |
| `docs/mechanics/video-oracle.md` | Split between `hugpy-video` and `hugpy-oracle`. |
| `docs/mechanics/curation-reliability.md` | Split between `hugpy-curation` and `hugpy-ops`. |
| `docs/mechanics/api-routes.md`, `frontends.md` | `hugpy-server` |
| `docs/mechanics/cli-agents-platform.md`, top-level `README.md` | Split between `hugpy-platform`, `hugpy-discord`, and the `hugpy` meta package. |
| `directions/REGISTRY-DB-INDEX.md` | `hugpy-engine` model-catalog documentation. |
| root `pyproject.toml` | Replace with the thin `hugpy` meta distribution after dependency extras move to their owners. |
| `setup.py` | Retire; every destination uses `pyproject.toml`. |
| `build/`, `dist/`, `*.egg-info`, `.pytest_cache` | Discard and regenerate per distribution. |

There are currently 310 test modules. Their move should be import-driven rather
than filename-only: first relocate tests whose primary imported implementation
has an owner, then keep multi-owner tests as server or ecosystem integration
tests.

## Cut order and completion gates

| Order | Cut | Complete when |
|---:|---|---|
| 1 | Platform | Wheel imports with stdlib/base dependencies and no monolith path. |
| 2 | Control | Stores accept configured roots; no engine/fleet imports. |
| 3 | Engine | Prompt-to-reply tests pass from installed wheel; legacy backend deleted. |
| 4 | Storage | Daemon/download/sync tests pass; engine-storage cycle removed. |
| 5 | Media | Each task registers as a plugin; CPU-only import passes. |
| 6 | Video | Job bus and synthetic runner tests pass without Oracle/server imports. |
| 7 | Oracle | Pure plan/DAG/selection tests pass against video contracts. |
| 8 | Fleet | Central/worker contract tests pass; optional GPU/media imports remain lazy. |
| 9 | Curation | Dossier/review pipeline uses storage and public engine APIs. |
| 10 | Ops + Discord | Both are leaf packages with independent entry points. |
| 11 | Server | Route contract suite passes with package fakes and built React mounts. |
| 12 | Meta | `hugpy` CLI dispatches installed package entry points. |
| 13 | Delete monolith | No package imports `abstract_hugpy_dev`; all wheels build and install together. |

Run `python py/validate_partition.py` after changing this map. It currently
checks that all top-level source areas are assigned, all declared source paths
exist, and the proposed package dependency graph is acyclic.
