# Hugpy extraction move map — end state

The monolith `abstract_hugpy_dev` has been fully partitioned. This is the
closing ledger: what became a distribution, what was retired, and where the
non-code assets went. Ownership detail is [`py/partition.toml`](py/partition.toml)
(167 move entries, 45 retirements); the architecture is
[`PARTITION.md`](PARTITION.md) (revision 2); registries and release order are
[`ECOSYSTEM.md`](ECOSYSTEM.md); startup seams are [`py/WIRING.md`](py/WIRING.md).

Gates (all green on 2026-09-22):

```bash
python py/validate_partition.py            # manifest valid, acyclic
python py/validate_partition.py --edges    # 0 forbidden import edges
.venv/bin/python py/tooling/check_imports.py   # 0 dangling imports per package
./local_install.sh --venv /tmp/hugpy-local-venv --check   # fresh editable install + alias check
```

## Distributions

| Status | Distribution | Import name | Location | Depends on |
|---|---|---|---|---|
| **Done** | `hugpy-platform` | `hugpy_platform` | `py/foundation/hugpy_platform` | – |
| **Done** | `hugpy-control` | `hugpy_control` | `py/foundation/hugpy_control` | platform |
| **Done** | `hugpy-storage` | `hugpy_storage` | `py/storage/hugpy_storage` | platform, control |
| **Done** | `hugpy-engine` | `hugpy_engine` | `py/inference/hugpy_engine` | platform, control, storage |
| **Done** | `hugpy-media` | `hugpy_media` | `py/inference/hugpy_media` | platform, storage, engine |
| **Done** | `hugpy-video` | `hugpy_video` | `py/cinema/hugpy_video` | platform, control, engine, media |
| **Done** | `hugpy-oracle` | `hugpy_oracle` | `py/cinema/hugpy_oracle` | platform, control, engine, media, video |
| **Done** | `hugpy-fleet` | `hugpy_fleet` | `py/fleet/hugpy_fleet` | platform, control, storage, engine |
| **Done** | `hugpy-curation` | `hugpy_curation` | `py/curation/hugpy_curation` | platform, control, storage, engine, oracle |
| **Done** | `hugpy-ops` | `hugpy_ops` | `py/operations/hugpy_ops` | platform, control, storage, engine, video, fleet, curation |
| **Done** | `hugpy-discord` | `hugpy_discord` | `py/integrations/hugpy_discord` | platform |
| **Done** | `hugpy-server` | `hugpy_server` | `py/services/hugpy_server` | everything above except ops/discord (optional) |
| **Done** | `hugpy` (meta) | `hugpy` | `py/meta/hugpy` | platform (+ extras pull the rest) |
| **Compat** | `abstract_hugpy_dev` 0.1.267.dev0 | `abstract_hugpy_dev` | `py/compat/abstract_hugpy_dev` | all 13; dev-only alias, never published |

Every package builds a wheel, ships `LICENSE`, and its tests pass strict (the
monolith name blocked by `conftest.py`). Cross-package placement and task
execution go through the seams in `hugpy_engine.placement` / `hugpy_engine.tasks`;
`hugpy_server.wiring.install_all()` wires them at startup.

## Independent packages (unchanged boundary)

| Distribution | Location | Note |
|---|---|---|
| `abstract-identity` | `py/cinema/abstract_identity` | own repo `hugpy/abstract-identity`, MIT |
| `hugpy-agent` | `py/inference/hugpy_agent` | own repo `hugpy/hugpy-agent` |
| `abstract-claude`, `abstract-gpt` | `py/inference/abstract_*` | not exported to the monorepo |
| `abstract-toolserver`, `abstract-apply` | `py/tools/*` | not exported to the monorepo |
| `hugpy-station` | `py/tools/hugpy-station` | private repo; holds deploy tooling and credentials, never exported |

## React consoles

`react/` is an npm workspace (`@hugpy/ui-shared`, `@hugpy/ui`, `@hugpy/agents-ui`,
`@hugpy/media-intelligence-ui`, `@hugpy/video-intelligence-ui`). Built bundles
are copied into `hugpy_server` package data by
`py/services/hugpy_server/tools/build_console.py` according to
`console_manifest.json` (see `BUILD_CONSOLE.md`). Publishing is tag-triggered
(`.github/workflows/npm-publish.yml`) and gated on an explicit go.

## Retired (never a distribution)

All in `py/partition.toml` `[[retire]]`; readable copies in
[`py/unwired/monolith/`](py/unwired/monolith/README.md):

- the wildcard re-export layer: `__init__.py`, `imports/**`, `managers/__init__.py`,
  `managers/imports.py`, `utils/__init__.py`, `utils/imports.py`, `comms/__init__.py`,
  every `*/imports.py` aggregator, `imports/src/standalone_utils.py`;
- dead or superseded code: `get_vids.py`, `managers/falconsai/`,
  `managers/generate/coder_guff.py`, `generate_runner2.py`, `worker_agent/get_size.py`,
  `managers/get_pids.py` (copy in `py/unwired/fleet/`);
- `.bak-*` snapshots of routes/workers/operator_auth and `console_dist.bak-shardpage`;
- nine files that moved with their package and were then dissolved during the
  explicit-import cleanup (listed at the end of the retire block).

The compat shell answers for every retired aggregator path as a lazy namespace
(`py/compat/abstract_hugpy_dev/RELOCATIONS.md`: 503 aliased modules, 55
aggregators, 10 modules with no alias).

## Where the non-code assets went

| Asset | Now |
|---|---|
| systemd units `hugpy-steward.*` | `py/cinema/hugpy_oracle/deploy/` |
| `hugpy-review@.*` (+ user units) | `py/curation/hugpy_curation/deploy/` |
| `hugpy-sentinel.*`, `hugpy-todo-keeper.service`, `SENTINEL.md`, `bin/` | `py/operations/hugpy_ops/deploy/` |
| `deploy/STATE-MODEL.md`, `WORKER-BOOT-PREWARM.md`, `WORKER-WILDCARD.md` | `py/fleet/hugpy_fleet/docs/` |
| `deploy/ALLOCATION-MODES.md` | `py/inference/hugpy_engine/docs/` |
| `directions/REGISTRY-DB-INDEX.md` | `py/storage/hugpy_storage/docs/` |
| `ARCHITECTURE.md`, `docs/mechanics/` (monolith-era maps) | `py/unwired/monolith/docs/` |
| `tests/` (multi-package) | `py/services/hugpy_server/tests/integration/`; single-package tests sit with their owner |
| last wheel, relocation map, surface probe | `py/compat/abstract_hugpy_dev/tools/inputs/` (generator inputs) |
| full git history (checkpoint `7c19ce7`) | `py/unwired/archive/abstract_hugpy_dev-pre-partition.bundle` (gitignored, 20 MB) |

The `abstract_hugpy_dev/` directory no longer exists in the workspace.
