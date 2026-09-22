# Retired monolith remnants

What was left in `abstract_hugpy_dev/src/abstract_hugpy_dev/` after every owned
module moved to its `hugpy_*` package (partition rev 2, 2026-09-22). Nothing
here is importable or wired; it is kept for reading only. The full pre-partition
history is in `../archive/abstract_hugpy_dev-pre-partition.bundle`.

| Path | What it was | Why it has no home |
|---|---|---|
| `__init__.py`, `imports/**`, `managers/__init__.py`, `managers/imports.py`, `utils/__init__.py`, `utils/imports.py`, `comms/__init__.py` | the wildcard re-export layer (`from .x import *` chains) | retired by `py/partition.toml`; the compat shell `py/compat/abstract_hugpy_dev` synthesises these as lazy namespaces |
| `imports/src/standalone_utils.py` | duplicate helpers used only by the aggregator layer | retired; owners import the real helpers |
| `get_vids.py` | scratch downloader script | dead code |
| `managers/falconsai/**` | abandoned Falconsai summariser experiment | dead code |
| `managers/generate/coder_guff.py`, `generate_runner2.py` | superseded generation runners | replaced by `hugpy_engine.generate` |
| `_scripts/get_all_imports.py`, `get_lines.py`, `get_lines_2.py` | one-off analysis scripts that sat beside the package in `src/` | not part of the package |
| `docs/ARCHITECTURE.md`, `docs/mechanics/*.md` | the monolith's own package map and the 2026-09-14 mechanics review | paths refer to the pre-partition tree; `PARTITION.md` is the current map |

Docs that still applied moved to their owners: `ALLOCATION-MODES.md` to
`py/inference/hugpy_engine/docs/`, `REGISTRY-DB-INDEX.md` to
`py/storage/hugpy_storage/docs/`, the worker state docs to
`py/fleet/hugpy_fleet/docs/`, the systemd units to the packages' `deploy/`.
The monolith's remaining `tests/` (scratch `evictions_change/`, a two-line
`test.py`) were not kept; `conftest.py` and `worker_store_isolation.py` live on
in `py/services/hugpy_server/tests/`.
