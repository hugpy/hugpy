# hugpy-ops

`hugpy_ops` — the operational consumers of the Hugpy public APIs, extracted
from `abstract_hugpy_dev` as part of the partition (`py/partition.toml`,
`PARTITION.md`). Nothing in this package is imported by normal server
startup; every tool is its own console script:

| Script | Module | What it does |
|---|---|---|
| `hugpy-sentinel` | `hugpy_ops.sentinel` | one detect -> case -> diagnose pass over central's HTTP read surfaces (`run-once`, `status`, `record-scorecard`) |
| `hugpy-chaos` | `hugpy_ops.chaos` | chaos-and-learn exerciser (`hugpy-chaos`) and the k7 offload speed-cliff sweep (`hugpy-chaos sweep`) |
| `hugpy-keeper` | `hugpy_ops.keeper` | stationary terminal REPL in which a served model keeps a machine or an LXD instance |
| `hugpy-todo-keeper` | `hugpy_ops.todo_keeper_daemon` | the enrolling todo-keeper agent node (`todo.add` / `todo.tidy`) |
| `hugpy-provisioner` | `hugpy_ops.provisioner` | declared-but-missing weights across the engine, studio and comfy registries -> `hugpy_storage` download queue |

Allowed ecosystem dependencies: `hugpy_platform`, `hugpy_control`,
`hugpy_storage`, `hugpy_engine`, `hugpy_video`, `hugpy_fleet`,
`hugpy_curation`. The central URL and every state root come from
`hugpy_platform` (`central_base_url`, `hugpy_state_dir`, `config_dir`,
`models_root`).

Deploy units and the operator notes live in `deploy/`; mechanics in `docs/`.

```sh
cd py/operations/hugpy_ops && ../../../.venv/bin/python -m pytest -q --timeout=120
```
