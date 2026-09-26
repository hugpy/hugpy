# hugpy-drift-check — the build-consistency gate

`hugpy-drift-check` (also `hugpy drift ...`) is a mechanical gate modelled on
the station's `station-drift-check.sh`. It proves that four things are on the
**same build** — the dev checkout, the packages installed in this interpreter,
the fleet's central and workers, and PyPI — and fails loudly when they are not.

The rule it enforces:

> **Nothing the system runs may be hand-authored somewhere that does not ship.**

Since the workspace moved to one git-derived lockstep version (every in-tree
`hugpy-*` distribution reads `X.Y.Z` at a tag and `X.Y.Z.devN+g<sha>[.d<date>]`
between tags), "same build" is a question `git` and `importlib.metadata` can
answer without trusting anyone's word. This tool asks it.

```
hugpy-drift-check                              # all sections, table, exit code
hugpy-drift-check --sections A,B --allow-dirty # what a developer runs mid-edit
hugpy-drift-check --sections C --central http://10.0.0.5:7002
hugpy-drift-check --json                       # the same report as JSON
hugpy-drift-check --quiet --notify-url https://hooks.example/drift
hugpy-drift-check --install-timer --dry-run    # print the systemd units
```

It is stdlib-only. `hugpy_platform.buildinfo` is imported lazily when the
installed platform ships it; without it the check degrades to plain
`importlib.metadata` + `git` and still runs every section.

## Sections and what each proves

Every section emits rows `section / subject / status / detail`. A status is
one of:

| status  | meaning |
|---------|---------|
| `ok`    | verified in sync |
| `DRIFT` | verified **not** in sync — this is the finding the gate exists for |
| `error` | could not verify (central unreachable, no git, PyPI down) |
| `info`  | context, or a comparison that is not applicable here |

### A `checkout` — the workspace vs its origin

Locates the workspace (`--workspace PATH`, else the editable install's source
per buildinfo / `direct_url.json`, else the git root of the current
directory). Then:

- `HEAD` — info: sha, branch, path.
- `tree` — modified **tracked** files. Untracked files are never drift (they
  do not ship, so nothing that ships can run them). Dirty is `DRIFT` unless
  `--allow-dirty` (then `info`).
- `origin/<branch>` — ahead/behind the upstream ref. `--fetch` runs
  `git fetch --quiet origin` first; without it the comparison is against the
  last fetch (the detail says which). Ahead (unpushed commits) or behind
  (origin moved on) is `DRIFT`. Detached HEAD or no `origin/<branch>` ref is
  `info`.

Proves: what is checked out is what the repository of record has.

### B `installed` — this interpreter vs the checkout

For each workspace distribution installed in **this** interpreter
(`buildinfo.WORKSPACE_DISTRIBUTIONS`, else the built-in list of 14):

- an editable install's source must be at the workspace HEAD, else `DRIFT`;
  if its recorded metadata was stamped at an older commit the row stays `ok`
  but says "reinstall to refresh";
- a non-editable install stamped `+g<sha>` from a commit other than the
  workspace HEAD is `DRIFT` ("built from g…");
- a version with no stamped sha and no tag (`0.0.0+unknown`) is `DRIFT`:
  "no git identity at build time". (`0.0.1.devN+unknown.g<sha>` — an untagged
  checkout — still carries its sha and is fine.)
- `lockstep` — all installed workspace distributions must share one version,
  else `DRIFT` listing which distributions sit at which version.

Proves: the code this interpreter imports is one build and it is the checkout.

### C `fleet` — central and every worker

GETs `<central>/api/health` and `<central>/api/llm/workers`. Central defaults
to `HUGPY_BASE_URL` (or its aliases `HUGPY_CENTRAL`, `HUGPY_URL`,
`WORKER_CENTRAL_URL`), else `http://127.0.0.1:7002`; `--central` overrides.
Central calls wait up to 60 s (`--timeout` / `HUGPY_DRIFT_TIMEOUT`): right
after a restart `/api/llm/workers` can take a minute while workers re-register,
and a short timeout would report a healthy fleet as `error` (exit 2).
A bearer token comes from `--token`, else `HUGPY_TOKEN` / `HUGPY_API_KEY`
(the two read surfaces are usually open).

- `central` — `/api/health` must carry `build` (`{version, sha, dirty, …}`).
  Present and equal to this interpreter's build: `ok`; different: `DRIFT`;
  absent: `error` — a central that cannot say which build it runs cannot be
  verified (this is what a pre-buildinfo central looks like).
- `worker <name>` — when the row carries `environment_digest.build`, its
  version and sha are compared to central's build (`ok` / `DRIFT`). When it
  does not, `pkg_version` is compared to `required_pkg_version`:
  `version_ok is None` (central pins nothing) is `info`; unequal is `DRIFT`.
  A worker whose `pkg_version` is a `0.1.<n>` monolith release, or that
  reports no version, is `DRIFT`: "monolith / no build identity".
- Central unreachable: `error` rows, never a traceback.

Proves: every process answering requests is the build central runs, which is
the build in this interpreter.

### D `pypi` — the newest tag vs what is published

Reads the newest tag in the checkout (`git describe --tags --abbrev=0`,
leading `v` stripped) and, for each workspace distribution,
`https://pypi.org/pypi/<name>/json` (10 s timeout).

- tag newer than PyPI: `DRIFT` — "unpublished release";
- PyPI newer than tag: `DRIFT` — "checkout behind release";
- equal: `ok`; not on PyPI: `info`; no tag in the checkout: `info` (the PyPI
  versions are listed for reference);
- PyPI unreachable: `error`. `--no-pypi` skips the section (one `info` row).

Proves: the release the world can install is the release the repository
declares.

## Exit codes

| exit | verdict      | when |
|------|--------------|------|
| 0    | `IN SYNC`    | no `DRIFT` and no `error` rows (`info` is fine) |
| 1    | `DRIFT`      | at least one `DRIFT` row (even if errors are also present) |
| 2    | `UNVERIFIED` | no drift found, but at least one section could not verify |

The exit code is the contract: wire it into a release script exactly as
`build-release.sh` wires `station-drift-check.sh` — run it first, refuse to
build when it is non-zero.

## Reading the table

```
SECTION      SUBJECT           STATUS  DETAIL
-----------  ----------------  ------  ------
A checkout   HEAD              info    952c2d8aa882 on consistency  (/home/u/hugpy-ws)
A checkout   tree              ok      clean (no modified tracked files)
A checkout   origin/consistency  ok    in sync at 952c2d8aa882 (vs last fetch)
B installed  hugpy-platform    ok      0.0.1.dev11+unknown.g952c2d8aa: source at workspace HEAD
B installed  lockstep          ok      14 distribution(s) at 0.0.1.dev11+unknown.g952c2d8aa
C fleet      central           error   no build identity in /api/health — central predates buildinfo; …
C fleet      worker aeb        DRIFT   monolith / no build identity (pkg_version=0.1.266) [online]

drift-check: 16 ok, 1 drift, 1 error, 3 info  ->  DRIFT  (exit 1)
```

- `SECTION` is the letter and name; `SUBJECT` is what was compared; `DRIFT`
  is upper-cased so it is visible in a scroll-back; `DETAIL` always names both
  sides of a failed comparison.
- The last line is the verdict with the counts and the exit code.
- `--quiet` prints the table only when the verdict is not `IN SYNC` (for
  timers and cron: silence means good). `--json` prints the same report as
  one document (`ok`, `exit_code`, `verdict`, `counts`, `rows`, plus the
  central URL, workspace path and this interpreter's build identity).
- `--notify-url URL` POSTs that JSON document when the verdict is not
  `IN SYNC`; a failed POST is reported on stderr and never changes the exit
  code.

## The timer

```
sudo hugpy-drift-check --install-timer --central http://127.0.0.1:7002 \
     --notify-url https://hooks.example/drift --env-file /etc/hugpy/drift.env
hugpy-drift-check --install-timer --user --on-calendar hourly   # per-user units
hugpy-drift-check --install-timer --dry-run                      # print, write nothing
```

`--install-timer` renders `hugpy-drift-check.service` and
`hugpy-drift-check.timer` from the templates shipped in the distribution
(`hugpy_ops/drift_units/`), writes them to `/etc/systemd/system` (or
`~/.config/systemd/user` with `--user`), then runs `systemctl [--user]
daemon-reload` and `enable --now hugpy-drift-check.timer`.

- `ExecStart` is the absolute `hugpy-drift-check` console script next to the
  interpreter that ran the installer (falling back to
  `<python> -m hugpy_ops.drift`), with `--quiet --fetch` and whichever of
  `--central`, `--notify-url`, `--workspace`, `--allow-dirty`, `--no-pypi`, `--timeout`,
  `--sections` you passed.
- `--env-file PATH` becomes `EnvironmentFile=-PATH` (put `HUGPY_BASE_URL=` and
  `HUGPY_TOKEN=` there rather than on the command line); `--token` becomes an
  `Environment=HUGPY_TOKEN=` line.
- `--on-calendar` is any systemd `OnCalendar=` spec (default `daily`); the
  timer is `Persistent=true` with five minutes of jitter.
- Do not edit the written units by hand — that is exactly the drift this tool
  exists to catch. Re-run the installer with different flags instead.

Inspect with `systemctl [--user] list-timers 'hugpy-drift-check*'` and
`journalctl [--user] -u hugpy-drift-check`.

## Related commands

- `hugpy build` / `hugpy build --json` — this interpreter's build identity
  (`hugpy_platform.buildinfo.identity_line()` / `build_info()`).
- `hugpy version` — the installed distributions, headed by the identity line.
- `GET /api/health` on central carries `build`; each `GET /api/llm/workers`
  row carries `environment_digest.build` — the two sides section C compares.
