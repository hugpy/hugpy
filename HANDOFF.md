# Handoff — Hugpy partition, updated 2026-09-22 (session 2)

Start here. Companion documents: `PARTITION.md` (architecture, rev 2),
`py/partition.toml` (ownership manifest), `py/EXTRACTION_GUIDE.md`,
`py/WIRING.md` (startup seams), `ECOSYSTEM.md` (registries and release plan),
`MOVE_MAP.md` (closing ledger), `LOCAL_INSTALL.md` (checkout install).

## 1. State of the workspace

- The monolith `abstract_hugpy_dev/` **no longer exists in the workspace**.
  Its full history is `py/unwired/archive/abstract_hugpy_dev-pre-partition.bundle`
  (checkpoint `7c19ce7`, gitignored, 20 MB; also on the host at
  `/srv/hugpy/src/abstract_hugpy_dev`). Retired remnants and monolith-era docs
  are in `py/unwired/monolith/`.
- 13 `hugpy-*` distributions plus the compat shell
  `py/compat/abstract_hugpy_dev` (0.1.267.dev0, dev-only, never published):
  a meta-path finder aliasing 503 old module paths to the same objects in
  their new homes, 55 retired aggregators as lazy namespaces, 657 top-level
  names re-exported; `abstract-hugpy-dev-check` proves it; `tools/generate.py`
  regenerates it from `tools/inputs/` (last wheel, monolith map, surface probe)
  without needing the monolith tree.
- `./local_install.sh [--venv X] [--extras server] [--check]` installs
  everything editable in one pip run (stdlib `py/local_install.py`); verified
  against a fresh venv. CI has a `local-install` job and the compat package in
  the matrix.
- React: `react/` is an npm workspace; tag-triggered
  `.github/workflows/npm-publish.yml`; `hugpy_server` consumes bundles via
  `console_manifest.json` + `tools/build_console.py` (`BUILD_CONSOLE.md`).
  Package `files` now include `dist` so future tarballs ship built bundles
  (today's published versions do not, so `--from-npm` cannot work until the
  next release). `react/video_intelligence_ui` now has a package-lock.

## 2. Gates (all green at the end of this session)

| Gate | Result |
|---|---|
| `python py/validate_partition.py` / `--edges` | valid, 167 moves, 45 retirements, 0 forbidden edges |
| `py/tooling/check_imports.py` | 0 dangling in all 13 |
| strict tests: platform 41, control 26, storage 127, engine 331, media 66, video 476, oracle 1221, fleet 1072, curation 108, ops 171, discord 13, meta 102, compat 16 | pass |
| server strict | 1885 passed, 0 failed, 35 xfailed, 21 skipped |
| `./local_install.sh --venv /tmp/hugpy-local-venv --check` | OK |

Server failures were triaged against the monolith checkpoint: 19 were real
partition regressions (fixed; two were code bugs: `describe_disk_error`
contract in `hugpy_storage.model_presence`, and `EngineCatalogSource.refresh`
walking the store on every `catalog.changed` and dropping the physical table);
100 failed before the partition too, of which 62 were test-order coupling
(fixed by collection-time isolation in the server conftest), 3 need `melt`
(skipif), 35 are stale and carry `xfail(strict=False)`; the ledger is
`py/services/hugpy_server/tests/integration/STALE.md`.

## 3. Next actions, in order

1. Push the monorepo again from the host (the workspace commit of this
   session is the source): `git archive HEAD | tar -x -C /tmp/hugpy-monorepo`,
   delete `py/tools`, `py/inference/abstract_claude`, `abstract_gpt`,
   `py/cinema/abstract_identity`, `py/inference/hugpy_agent`; commit;
   `git bundle create`; scp to 192.168.1.100; on the host clone the bundle and
   `git push https://github.com/hugpy/hugpy.git main`. Then check
   `gh run list -R hugpy/hugpy` (first CI run never verified).
2. Optional cleanups: fix the 35 stale server tests (the 21 studio ones only
   need `_setup_fixtures()` called from a module fixture); add vitest to
   `react/media_intelligence_ui`; add `CHANGELOG.md` per React package;
   remove the empty `react/agents_ui/ui_shared/`.
3. Only on explicit go: PyPI 0.2.0 line bottom-up (`ECOSYSTEM.md` D.3),
   bridge release retiring `abstract_hugpy_dev` on PyPI (D.4), npm republish
   via tags (D.5; needs the `npm-publish` GitHub environment and `NPM_TOKEN`).
4. Live host switch last: on the host, from the workspace checkout,
   `./local_install.sh --venv /srv/hugpy/venv --extras server --check`; move
   the systemd units from the package `deploy/` dirs (oracle, curation, ops);
   restart. **Never touch the Hugging Face org `hugpy-ai`.**

## 4. Known open items

- `hugpy_oracle.interim_ledger.MediaBusSource` reads video's `media_jobs.db`
  read-only; a `hugpy_video.jobs` bulk-export API would remove it.
- `hugpy_fleet.worker.agent` imports Flask at module level.
- `hugpy_ops.chaos` defaults the operator token file to the station share path.
- `py/inference/hugpy_agent` local `origin` still points at a broken
  AbstractEndeavors URL; the new remote is `https://github.com/hugpy/hugpy-agent.git`.
- The system miniconda still has monolith 0.1.266 in site-packages; never run
  package tests with it (`local_install.py` refuses non-venv interpreters).
- `@hugpy/console` and `@hugpy/vm-mgr` have no source in this workspace.
