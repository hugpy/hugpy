# Handoff — Hugpy partition, updated 2026-09-22 (session 3)

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
- The monorepo `github.com/hugpy/hugpy` carries the workspace export
  (everything except `py/tools`, `py/inference/abstract_claude`,
  `py/inference/abstract_gpt`, `py/cinema/abstract_identity`,
  `py/inference/hugpy_agent`). Re-push procedure: `git archive HEAD` into
  `/tmp/hugpy-monorepo`, delete those five, tar, scp to 192.168.1.100, and on
  the host overlay it with `rsync -a --delete --exclude .git` onto the clone
  `/tmp/hugpy-monorepo-push` (pull first), commit, `git push`. Host commits so
  far: `dc28670` (session 2 first push), `5141d72`, `e334744`, `7e17756`,
  `a19f16f` (session 3).
- The compat shell binds stdlib re-exports name by name
  (`_relocations.reexport_stdlib`); names the running interpreter lacks (on 3.12:
  `NoDefault`, `ReadOnly`, `TypeIs`, `get_protocol_members`, `is_protocol`) go to
  `_SURFACE_UNAVAILABLE` and the check reports them as UNAVAILABLE, not missing.
  Verified on 3.12 (`/usr/bin/python3.12`, venv `/tmp/hugpy-312`) and 3.13.
- React: `react/` is an npm workspace; tag-triggered
  `.github/workflows/npm-publish.yml`; `hugpy_server` consumes bundles via
  `console_manifest.json` + `tools/build_console.py` (`BUILD_CONSOLE.md`).
  Package `files` now include `dist` so future tarballs ship built bundles
  (today's published versions do not, so `--from-npm` cannot work until the
  next release). `react/video_intelligence_ui` now has a package-lock.
  `react/media_intelligence_ui` runs vitest (`npm test`, 74 tests); every
  package has a `CHANGELOG.md`; CI runs `npm test --if-present` per package.

## 2. Gates (all green at the end of this session)

| Gate | Result |
|---|---|
| `python py/validate_partition.py` / `--edges` | valid, 167 moves, 45 retirements, 0 forbidden edges |
| `py/tooling/check_imports.py` | 0 dangling in all 13 |
| strict tests: platform 41, control 26, storage 127, engine 331, media 66, video 476, oracle 1221, fleet 1072, curation 108, ops 171, discord 13, meta 102, compat 16 | pass |
| server strict | 1908 passed, 0 failed, 12 xfailed, 21 skipped |
| `./local_install.sh --venv /tmp/hugpy-local-venv --check` | OK on 3.13 and 3.12 |
| `react/media_intelligence_ui` `npm test` | 74 passed |
| GitHub CI on `hugpy/hugpy` main | run 35793948920 (host commit `a19f16f`): all 20 jobs green |

Server failures were triaged against the monolith checkpoint: 19 were real
partition regressions (fixed; two were code bugs: `describe_disk_error`
contract in `hugpy_storage.model_presence`, and `EngineCatalogSource.refresh`
walking the store on every `catalog.changed` and dropping the physical table);
100 failed before the partition too, of which 62 were test-order coupling
(fixed by collection-time isolation in the server conftest), 3 need `melt`
(skipif), 35 were stale and carried `xfail(strict=False)`. Session 3 fixed 23
of those (the 21 script-style studio tests via a module-scoped autouse fixture
calling `_setup_fixtures()`, plus `test_cold_hold_cap` wording and the
`studio_tester` fake enqueue signature); 12 remain xfailed and are listed in
`py/services/hugpy_server/tests/integration/STALE.md` (6 identity-profile
tests that predate versioned profiles, 5 preset/pin tests bound to the removed
`wan2.2-t2v-a14b` seed row, 1 compute-actions error-reason assertion).

First two CI runs on the monorepo failed and were triaged: run 1 (session 2
push) because the validator then still checked source paths against the
monolith tree; run 2 because the per-package `pip install -e` loop reached
`hugpy_control` before `hugpy_platform` and fell through to PyPI, and because
the compat shell did `from _typing import NoDefault`, which only exists on
3.13; run 3 because the runner image has no ffmpeg (video, server), the MoE
split test relied on the box's RAM, and the unmocked fleet sizing test needs a
model store; run 4 because the studio scratch dir under a fresh storage root
did not exist, and because `_warmable_subset` swallowed the first un-fittable
skip note while `monotonic()` was still below the cooldown (a real bug on any
freshly booted box, fixed in `hugpy_server.app.routes.worker_routes`). CI now
installs ffmpeg, the store-dependent test skips without a store, and the
video conftest creates the scratch dir.

## 3. Next actions, in order

1. Nothing is blocking. The monorepo on GitHub matches workspace `eef460e`
   (plus this handoff commit). If you change the workspace, re-push with the
   procedure in section 1 and watch `gh run list -R hugpy/hugpy` from the host.
2. Optional cleanups left: the 12 remaining stale server tests (see STALE.md;
   the preset/pin ones need a decision on whether `wan2.2-t2v-a14b` returns to
   the studio seed or the presets retarget); `hugpy_fleet.worker.agent`'s
   module-level Flask import.
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
