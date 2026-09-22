# Extraction guide (working contract for every package agent)

This is the shared, binding procedure for finishing one Hugpy package after
its files have been physically moved out of `abstract_hugpy_dev`. Read it
fully before touching code. The manifest `py/partition.toml` is the single
source of truth for ownership and allowed imports; `PARTITION.md` explains the
architecture.

## 1. Environment

```bash
cd /home/op/Documents/hugpy/trimming
source .venv/bin/activate            # every hugpy-* package is installed -e
# (historical) export PYTHONPATH=abstract_hugpy_dev/src was the transitional mode; the tree is retired,
# the compat shell py/compat/abstract_hugpy_dev now answers for old import paths
export HUGPY_ALLOW_MONOLITH=1              # lifts the conftest block during transition
```

Strict mode (the completion gate) is the same commands **without** those two
variables: the monolith must be unimportable and the package must still pass.

Tools (all read the manifest; run from the workspace root):

| Tool | Purpose |
|---|---|
| `python py/validate_partition.py --edges --package <id>` | Every import edge the manifest forbids for your package, with file:line. This is your to-do list. |
| `.venv/bin/python py/tooling/explicit_imports.py <file-or-dir> --write` | Replaces `from X import *` with explicit imports resolved to the true defining module; also re-sources explicit imports from retired aggregator modules. Ran with `PYTHONPATH=abstract_hugpy_dev/src` while the tree existed; the compat shell serves the same map now. Leaves a `# TODO(partition): unresolved` comment for names it cannot place. |
| `.venv/bin/python py/tooling/check_imports.py <id>` | Dangling imports (module or name that does not exist). Must be zero. |
| `python py/tooling/relocate.py <monolith-relative-path>` | Re-home a file after you change its owner/destination in `partition.toml`. Rewrites references workspace-wide. |
| `python py/tooling/extract_package.py <id> --scaffold-only` | Regenerate `tests/test_import_policy.py` after a manifest change. |

## 2. Rules (from PARTITION.md, enforced by `tests/test_import_policy.py`)

1. No wildcard imports anywhere in `src/`.
2. No import of `abstract_hugpy_dev`, ever, in `src/` or `tests/`.
3. Only the ecosystem packages listed in your `depends` may be imported.
   `optional_depends` may be imported only lazily (inside functions or
   `try:` blocks), never at module import time.
4. `import <package>` must succeed with every optional dependency and the
   monolith blocked (`test_package_imports_without_monolith`). Heavy third-party
   stacks (torch, diffusers, whisper, llama_cpp, cv2, discord, flask for
   non-server packages) are imported lazily.
5. State has one owner. Never open another package's state files; call its
   public API or use an injected protocol.
6. HTTP is an adapter. Core behaviour must be callable without Flask. Flask
   lives only in `hugpy_server`.
7. Tests move with the implementation and must pass in strict mode.
8. Public API is exported deliberately from the package `__init__` via
   `__all__`; keep `__init__` light (no heavy imports).

## 3. How to remove a forbidden edge (pick the *first* that applies)

1. **Wrong owner.** The imported thing (or the importing thing) simply belongs
   elsewhere. Fix `partition.toml`, run `relocate.py`, regenerate the policy
   test. Prefer moving *down* (toward platform) over moving up.
2. **Re-export through a retired aggregator.** Run `explicit_imports.py`; the
   true origin is usually already an allowed package.
3. **Lower layer reaching up** (e.g. engine importing fleet's worker
   registry, video importing server helpers). Define a small `typing.Protocol`
   in the lower package plus a module-level provider registry with a safe
   default (no-op / in-memory), and have the upper package register its
   implementation at composition time. Naming convention:
   `hugpy_<pkg>/providers.py` exposing `get_<thing>()` / `set_<thing>()`.
   The composition root (`hugpy_server.wsgi_app` or a package `cli`) wires
   real implementations. Document the protocol in the module docstring.
4. **Genuinely shared low-level helper** (pure function, no domain): move it
   to `hugpy_platform` (stdlib-first) and import it from there.
5. **Dead or unwired code.** If nothing reachable calls it, delete it. Do
   not keep dead code alive by shimming; git history is the record.

Never solve an edge by adding the dependency to `partition.toml` unless the
architecture note in PARTITION.md is genuinely wrong; if you believe it is,
state why in your report and keep the graph acyclic
(`python py/validate_partition.py` must stay green).

## 4. Cross-package protocol homes (agreed, do not duplicate)

| Concern | Protocol module | Implemented by | Wired by |
|---|---|---|---|
| Worker placement / registry queries (which worker serves what, worker HTTP) | `hugpy_engine.placement` | `hugpy_fleet.central.placement` | `hugpy_server` |
| Eviction ledger, blocklist, model metrics, priority groups seen by the engine | `hugpy_engine.placement` (same registry, separate Protocols) | `hugpy_fleet.central.*` | `hugpy_server` |
| Task runner registry (media/video runners plugging into the engine) | `hugpy_engine.tasks` (`register_runner`, `runner_for`) | `hugpy_media.plugin`, `hugpy_video.plugin` | `hugpy_server`, worker entry points |
| Catalog invalidation after downloads | `hugpy_control.bus` topic `TOPIC_CATALOG_CHANGED` | published by `hugpy_storage` | consumed by `hugpy_engine` |
| Download requests from upper layers | `hugpy_storage.download_models` / `hugpy_storage.downloader.queue` public functions | storage | callers import storage directly (allowed for engine, media, fleet, curation, ops, server) |
| GGUF/marker file inspection | `hugpy_storage.hugpy_marker`, `hugpy_storage.gguf_inspect` | storage | engine's `spill` imports from storage |
| Media extraction used by Oracle | `hugpy_media.extract` | media | oracle imports media directly |
| Central base URL, env values, app dirs | `hugpy_platform` | platform | everyone |

If you must add a protocol not in this table, add a row here in your report.

## 5. Ownership of edits (parallel agents)

- You may edit anything under your own package root.
- You may **add** files to another package only when this guide's table says
  that package hosts the protocol, and only new files (never rewrite theirs).
- You may edit another package's existing file only to *remove* code you are
  moving into yours, leaving a one-line import re-export so callers keep
  working, and you must say so in your report.
- Do not edit the monolith (`abstract_hugpy_dev/`) except to delete files that
  became unused; never add code there.
- Do not touch `py/tooling/`, `partition.toml` dependency lists, or another
  agent's tests unless told to.

## 6. Tests

- `pytest <your package root>` in transitional mode first, then strict mode.
- Script-style tests (module-level `fail = 0` counters, `if __name__ ==
  "__main__": sys.exit(...)`) do nothing under pytest. Convert the ones that
  exercise your package into real `def test_*` functions (`assert`), keeping
  the intent; delete any that require live services and cannot be faked, and
  list them in your report.
- Tests that need the monolith or another non-dependency package are
  integration tests: move them to `py/services/hugpy_server/tests/integration/`
  (they are collected later by the server agent), do not delete them.
- Add at least one behavioural smoke test per public entry point you create
  (protocol default, plugin registration, CLI `--help`).

## 7. Report format (what you return)

1. Forbidden edges before / after (`validate_partition.py --edges --package`).
2. Dangling imports after (`check_imports.py`).
3. Test results: transitional and strict, with counts, and the list of tests
   deleted/moved and why.
4. Files you added outside your package (protocol implementations).
5. Manifest changes you made (owner/destination) and why.
6. Anything left unresolved with the exact file:line and the reason.
