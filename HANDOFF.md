# Handoff — Hugpy partition, updated 2026-09-22 (session 4)

Start here. Companion documents: `PARTITION.md` (architecture, rev 2),
`py/partition.toml` (ownership manifest), `py/EXTRACTION_GUIDE.md`,
`py/WIRING.md` (startup seams), `ECOSYSTEM.md` (registries and release plan),
`LOCAL_INSTALL.md` (checkout install).

## 1. State of the workspace

- **The previous chapter is archived off-tree, not carried.** Everything
  pre-partition now lives on the host at
  `/mnt/16T_toshiba/llm_storage/ARCHIVE/HUGPY_OLD` (README inside): the
  monolith history bundle (`7c19ce7`, sha256 `ed741af4…`), the retired
  remnants and parked scripts that were `py/unwired/`, `MOVE_MAP.md` (the
  closing ledger), the server `STALE.md` triage, the session-3 handoff, and the
  host's dirty monolith checkout (`/srv/pyit/dev/abstract_hugpy_dev`, HEAD
  `c46c5a5` + 1087 uncommitted changes). Nothing in the tree references any of
  it any more. The 166 superseded Station installer builds (7.3 GB under
  `py/tools/hugpy-station/old/`) were dropped outright, not archived, on the
  user's call.
- 13 `hugpy-*` distributions plus the compat shell `py/compat/abstract_hugpy_dev`
  (dev-only, never published; `tools/generate.py` regenerates it from
  `tools/inputs/` and no longer looks for unwired copies).
- `hugpy_fleet` now declares `flask>=3` (the worker agent is a Flask app and
  `hugpy-worker` is its console script; the dependency was never listed).
- The 12 xfailed server tests were deleted, not fixed; `test_identity_profiles`
  creates `mira` once in a module-scoped autouse fixture, which the surviving
  list/get/duplicate checks read back. The 3 `melt` skips remain (environment).
- `local_install.sh` produces a bare venv (no pytest). Run package tests the way
  CI does: from the package dir, `.venv/bin/python -m pytest -q --timeout=120`.
- Monorepo `github.com/hugpy/hugpy` matches workspace `ffa7fbe` (plus this
  handoff commit). Re-push procedure unchanged: `git archive HEAD` into
  `/tmp/hugpy-monorepo`, delete `py/tools`, `py/inference/{abstract_claude,
  abstract_gpt,hugpy_agent}`, `py/cinema/abstract_identity`, tar, scp to
  192.168.1.100, overlay with `rsync -a --delete --exclude .git` onto
  `/tmp/hugpy-monorepo-push` (pull first), commit, push. Host commits: `dc28670`,
  `5141d72`, `e334744`, `7e17756`, `a19f16f` (sessions 2–3), `1119398` (session 4).

## 2. Gates (all green at the end of this session)

| Gate | Result |
|---|---|
| `python py/validate_partition.py` / `--edges` | valid, 167 moves, 45 retirements, 0 forbidden edges |
| `py/tooling/check_imports.py` | 0 dangling in all 13 |
| fleet strict | 1072 passed, 4 skipped |
| server strict | 1908 passed, 0 failed, 21 skipped (no xfails left) |
| `./local_install.sh --venv /tmp/hugpy-local-venv --check` | OK (3.13) |
| `abstract-hugpy-dev-check` | OK on 3.13 and 3.12 |
| host `./local_install.sh --venv /srv/hugpy/venv-0.2 --extras server,ops --check` | OK (python 3.12.3) |
| GitHub CI on `hugpy/hugpy` main | run 35797052968 (host commit `1119398`): all 20 jobs green |

## 3. Host cutover — staged, one command left

The user has said the live host services are odds and ends and may be down
while the new packages go up. Everything is prepared; the final step was
blocked by the assistant's permission classifier (production deploy) and is
the user's to run:

```bash
ssh solcatcher@192.168.1.100 bash /tmp/hugpy-cutover.sh
```

What is already in place on the host:

- `/srv/hugpy/src/hugpy` — full workspace export (owned by `hugpy`, Station
  `old/` builds removed).
- `/srv/hugpy/venv-0.2` — fresh python3.12 venv, `hugpy[server,ops]` + the 13
  packages installed editable from that checkout, `--check` OK. Every console
  script the units need is present (`hugpy`, `hugpy-bot`, `hugpy-downloader`,
  `hugpy-todo-keeper`, `gunicorn`).
- `/tmp/hugpy-cutover.sh` — backs up the four unit files to
  `/srv/hugpy/backups/units-monolith-2026-09-22/`, rewrites the two
  `ExecStart`s that invoked the monolith by module path (`hugpy-bot`,
  `hugpy-todo-keeper`; the API and downloader commands are unchanged), stops
  `7002_hugpy_api`/`hugpy-bot`/`hugpy-todo-keeper`, moves `/srv/hugpy/venv` to
  `venv.monolith-0.1.266` and **symlinks** `/srv/hugpy/venv -> venv-0.2` (a
  rename would break the venv's baked-in shebangs), reloads, starts, and prints
  unit state, `/health`, and any journal errors. Rollback steps are in the
  script header.
- `hugpy-downloader.service` was already inactive before this session; the
  script leaves it that way (start it by hand if wanted).
- `board-mirror@` and `journal-watch` are stdlib-only scripts and keep working
  through the symlink. Station, toolserver (7004), demo, showroom and exec units
  use their own interpreters and are untouched.
- The monolith's on-disk checkout `/srv/pyit/dev/abstract_hugpy_dev` (symlinked
  from `/srv/hugpy/src/abstract_hugpy_dev`) is still there; it is archived in
  full, so it can be deleted once the cutover has settled.

## 4. Next actions, in order

1. Run the cutover (section 3) and watch `journalctl -u 7002_hugpy_api -f`.
   If anything misbehaves, roll back per the script header and report.
2. After it settles: delete `/srv/pyit/dev/abstract_hugpy_dev`, the symlink,
   `/srv/hugpy/venv.monolith-0.1.266`, and the stray `/srv/hugpy/src (Copy)` /
   `src.zip` (554 MB) — all previous-chapter material. Adopt the packages'
   `deploy/` units (oracle steward, curation review, ops sentinel) if those
   services are wanted on the host.
3. Only on explicit go: PyPI 0.2.0 line bottom-up (`ECOSYSTEM.md` D.3), bridge
   release retiring `abstract_hugpy_dev` on PyPI (D.4), npm republish via tags
   (D.5; needs the `npm-publish` GitHub environment and `NPM_TOKEN`).
   **Never touch the Hugging Face org `hugpy-ai`.**

## 5. Known open items

- `hugpy_oracle.interim_ledger.MediaBusSource` reads video's `media_jobs.db`
  read-only; a `hugpy_video.jobs` bulk-export API would remove it.
- `hugpy_ops.chaos` defaults the operator token file to the station share path.
- `hugpy-downloader` has no argument parser: `--help` starts the daemon.
- `py/inference/hugpy_agent` local `origin` still points at a broken
  AbstractEndeavors URL; the new remote is `https://github.com/hugpy/hugpy-agent.git`.
- The system miniconda still has monolith 0.1.266 in site-packages; never run
  package tests with it (`local_install.py` refuses non-venv interpreters).
- `@hugpy/console` and `@hugpy/vm-mgr` have no source in this workspace.
- The preset `max-quality-t2v` still targets 1280x720 t2v with no catalog model
  satisfying it (the `wan2.2-t2v-a14b` seed row was removed 2026-08-13); the
  tests that pinned it are gone, the preset itself is untouched.
