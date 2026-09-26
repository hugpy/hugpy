# Consistency: one version, three surfaces

How the fourteen `hugpy-*` distributions stay the same code on PyPI, in every
developer checkout and on every fleet worker, and how that is checked.

## The rule

> **Nothing the system serves or executes may be hand-authored somewhere that
> does not ship.**

That is the station drift doctrine (`station-drift-check.sh`), applied to the
Python side. Convention does not hold it; the checks below do, mechanically. A
live hotfix is allowed; shipping while one is outstanding is not.

## The three surfaces and the one version they share

| Surface | What runs there | Where its version comes from |
|---|---|---|
| PyPI | the wheels `pip install "hugpy[server]"` resolves | the git tag `vX.Y.Z` the CI built from |
| Dev checkouts | `./local_install.sh` editable installs of this tree | the same tag, or `X.Y.Z.devN+gSHA` between tags |
| Fleet workers | `hugpy-fleet` on every box central manages | central's own installed version (`required_pkg_version`) |

There is exactly one version number for all 14 distributions at any commit.
Nobody types it: `py/validate_partition.py --versions` fails any pyproject that
carries a static `version =` or any `__init__.py` that spells one out.

## How versions are derived

Every in-tree pyproject has `dynamic = ["version"]` and

```toml
[tool.setuptools_scm]
root = "../../.."            # the workspace: py/<layer>/<pkg> -> repo root
fallback_version = "0.0.0+unknown"
```

setuptools-scm reads the workspace git history at build (or editable-install)
time:

| State of the checkout | Version every distribution reports |
|---|---|
| exactly at tag `v0.2.0` | `0.2.0` |
| 7 commits after `v0.2.0`, clean | `0.2.1.dev7+g1a2b3c4` |
| same, with uncommitted tracked changes | `0.2.1.dev7+g1a2b3c4.d20260922` |
| no tags reachable at all (this tree before its first release) | `0.0.1.dev<N>+unknown.g1a2b3c4` |
| no git metadata (tarball, sdist without PKG-INFO) | `0.0.0+unknown` |

`__version__` in every package is `importlib.metadata.version(<dist>)`, with
`"0.0.0+unknown"` as the only literal allowed. An editable install freezes the
version at install time; re-run `./local_install.sh` after tagging if you need
the installed metadata to say the new number.

## The release flow

1. Merge to `main`. CI (`.github/workflows/ci.yml`) is green: manifest, import
   edges, `--versions`, the per-package matrix, `local-install`, and `lockstep`
   (all 14 editable-installed distributions report one identical version that
   is not `0.0.0+unknown`).
2. On a clean, up-to-date `main`: `./release.sh X.Y.Z --dry-run`, then
   `./release.sh X.Y.Z`. The script refuses on a dirty tree, a non-default or
   stale branch, an existing tag, a red `--versions`, or a red
   `hugpy-drift-check --sections A,B`. It then runs
   `git tag -a vX.Y.Z -m "hugpy X.Y.Z"` and `git push origin vX.Y.Z`.
3. The tag triggers `.github/workflows/pypi-publish.yml`: build sdist+wheel for
   every `[[package]]` in `py/partition.toml` (cut order, derived at run time),
   assert every artifact's version equals the tag, `twine check`, publish
   through PyPI trusted publishing (environment `pypi`, no stored token), then
   create the GitHub Release `vX.Y.Z` with the artifacts attached.
4. Central adopts; workers converge (next section).

If PyPI rejects the trusted-publisher exchange after CI has built and checked
the archives, the release is not complete. An operator can download that run's
`dist` artifact, upload the same files with an account-scoped PyPI token, and
verify the wheel and sdist at every exact-version PyPI JSON endpoint. Only then
attach those archives to the GitHub Release. Do not rebuild an already-published
version from another commit or assume that a pushed tag was published.

Nothing edits a version anywhere in that flow. The first partitioned release is
`0.2.0`: PyPI `hugpy` already exists at 0.1.181 (the retired monolith) and
0.2.0 sorts above it.

A brand-new distribution name needs to exist on PyPI before its normal trusted
publisher can be registered. PyPI allows a given (owner, repository, workflow,
environment) trusted publisher as a *pending* publisher for only one not-yet-existing
project, so the name is created once by hand (local-only tag `vX.Y.Za0`,
`python -m build`, `twine upload` of the a0 placeholder with an account-scoped
token, tag deleted), after which the ordinary publisher is added on the project
page and the tag flow above takes over. An operator can also create the name by
uploading the first final release from CI's validated artifacts with the same
account-scoped token, then register the publisher for later releases.

## How central adopts and workers converge

```bash
# on central
pip install -U "hugpy[server]==X.Y.Z"
systemctl restart <the hugpy service>
```

Central's `required_pkg_version` is **its own installed `hugpy-fleet`
version**, read fresh from `importlib.metadata` on every heartbeat reply
(`hugpy_fleet.central.workers.required_pkg_version`). There is no pin file, no
env var and no cache: the instant central runs X.Y.Z it advertises X.Y.Z. Each
worker compares that to its own version on its next heartbeat, installs the
advertised version from central's PEP 503 index
(`<central>/api/llm/pip/simple`, populated by `py/build_wheels.py --publish`,
added as an extra index alongside PyPI when it holds the release) and re-execs.

Only a clean tagged version is ever advertised: a version containing `+`
(`0.2.1.dev7+g1a2b3c4`, `0.0.0+unknown`) makes `required_pkg_version` return
`None`, which means "not managing versions". An editable dev central therefore
never pushes a dev build to the fleet, and a broken checkout never pushes
`0.0.0+unknown`.

## Releasing through central (no PyPI in the loop)

Central already hands workers their model files; it hands them their wheels
the same way. The wheel directory `hugpy_fleet.central.workers.pkg_index_dir()`
(`$HUGPY_PKG_INDEX_DIR`, default `<state dir>/pip_index`) is served as a PEP 503
index at `<central>/api/llm/pip/simple/<name>/`. Publishing a release there
is one command on the central box, from a checkout of the tag:

```bash
git fetch --tags && git checkout vX.Y.Z
python py/build_wheels.py --expect-tag vX.Y.Z --publish "$HUGPY_PKG_INDEX_DIR"
```

`py/build_wheels.py` is the same builder CI's `pypi-publish.yml` runs: every
`[[package]]` in cut order, versions read back from the artifacts, all
identical and equal to the tag, or nothing is published. Artifacts are
immutable on the index (a differing file with the same name is refused).

From then on:

* `required-version`, register and heartbeat replies carry `pkg_index_url`
  whenever the index holds **every** lockstep wheel of the required version
  (`pkg_index_has`); a half-published index is never advertised, so workers
  fall back to PyPI rather than strand on a missing sibling.
* the worker's converge adds it as `pip --extra-index-url`: the pinned
  `hugpy-*` wheels resolve from central, third-party dependencies still come
  from PyPI, and an explicit `--pkg-index` (a WireGuard-only box with no egress)
  keeps its `--index-url` meaning.
* `bootstrap.sh` (`<central>/api/llm/workers/install.sh`) reads the same key
  from `required-version`, so a bare box enrolls from central's wheels too.
* drift section D counts a tag that central's index serves as published, and
  reports PyPI's version alongside for reference.

PyPI remains the public surface (`pip install hugpy` on a box that knows no
central) and the tag flow above still publishes there; central's index is how
the fleet itself converges, with or without PyPI.

## Publishing to GitHub (a pipeline step, not a manual act)

Pushing the source trees to GitHub is a STEP of the pkg_src pipeline, run by the
same watcher that builds versions and promotes the fleet. `pkg_publish.py`
(driven from `pkg_src.py`) commits and pushes the configured trees when a
promotion the watcher judged reaches `healthy` (pkg_promote's `rollout_tick`).
It is a scaffold, **OFF by default**, and inert until an operator writes the
config and turns it on.

```bash
cp py/tooling/pkg_publish.example.toml /srv/hugpy/etc/pkg_publish.toml
# edit /srv/hugpy/etc/pkg_publish.toml:  enabled = true,  mode = "dry-run"
python3 py/tooling/pkg_src.py publish --config <NAME> --dry-run   # preview by hand
# review the pkg_src.publishes rows, then set  mode = "push"
```

Roll out in two moves. With `enabled = true, mode = "dry-run"` the step stages,
secret-scans and commits **locally** on every `promotion -> healthy`, but never
pushes; with `mode = "push"` it also pushes each repo's branch. The config file
(`$PKG_PUBLISH_CONFIG`, else `/srv/hugpy/etc/pkg_publish.toml`; absent =>
disabled) lists each `[[repo]]` (path, https url, branch, include_untracked,
extra_gitignore), the token env var + env file, and the commit author. Publishing
is guarded so it can never block or fail a promotion, runs one-at-a-time under a
flock, and every attempt is recorded per repo in `pkg_src.publishes` (promotion,
config, release, repo, branch, commit sha, pushed, mode, secret-scan, status,
scrubbed error, timings).

Per repo, in order: a rebase/merge in progress, a detached HEAD or a wrong-branch
checkout is refused (`dirty_state`); the remote branch is fetched over HTTPS and,
if it holds commits the local branch lacks, the repo is **skipped, never
force-pushed** (`diverged`); the tree is staged honoring `.gitignore`; then a
**secret gate** scans the staged added content (gitleaks if installed, else a
grep ruleset — private keys, `sk-`/`sk-ant-`/`hf_`/`ghp_`/`github_pat_`/`pypi-`/
`npm_` tokens, `postgres://user:pass@`, `*_TOKEN|*_KEY|*_SECRET|PASSWORD=`
literals, `.env`/`.pypirc`/`.npmrc`/`.git-credentials`/`htpasswd`/`.secret`/
`.auth` filenames, WireGuard `PrivateKey`, files > 10 MB) — any hit unstages
everything and blocks the push (`blocked_secret`, recording file:line+rule, never
the value); nothing staged is `clean`; otherwise it commits `<config> ->
<release>` (a `Published-By: pkg_src promotion <id>` trailer, no Co-Authored-By)
and pushes (or stops at the local commit in dry-run). The token is never written
to git config, a remote URL, a log or the DB: it is read from the env file at run
time, handed to git only through a `GIT_ASKPASS` helper, and scrubbed from any
captured output.

## The drift check

`hugpy-drift-check` (from `hugpy-ops`) is read-only and exits 0 in sync,
1 on drift, 2 on error. `--sections` selects what to compare:

| Section | Compares | Where it runs |
|---|---|---|
| A | the git checkout: clean tracked tree, no unpushed commits, HEAD level with the remote | dev box, before a release (`release.sh`) |
| B | the installed distributions vs. the checkout: all 14 present, one identical version, editable paths resolve here | CI `lockstep` job, dev box |
| C | the fleet vs. central: central's `/api/health` build identity, then every worker's heartbeat build (version + sha) against it; a worker still on the monolith, or without a build identity, is drift; an offline worker is info (not part of the running fleet); `version_ok = null` (central pins nothing) is info | central |
| D | the newest git tag vs. PyPI: a tag newer than PyPI is an unpublished release unless central's index serves it, PyPI newer than the tag is a checkout behind a release, not on PyPI at all is info (ok when central's index serves the tag) | dev box, central |

`release.sh` gates on `A,B`; CI runs `B` (green until `hugpy-ops` ships the
command, enforcing from then on); the nightly timer shipped with `hugpy-ops`
(`hugpy-drift-check --install-timer [--user] [--central URL] [--notify-url URL]`
writes and enables `hugpy-drift-check.timer`) runs `A,B,C,D` with `--quiet --fetch`. Runtime state (registries, caches, logs, model files) is never
compared: only authored and derived artifacts are drift.

## A fresh box from PyPI

| Command | What lands |
|---|---|
| `pip install hugpy` | the `hugpy`/`hpy` commands and `hugpy-platform`, nothing heavier |
| `pip install "hugpy[server]"` | central: `hugpy-server`, `hugpy-fleet`, `hugpy-engine[gguf]`, media, video, oracle, curation, storage, control, discord bot, gunicorn |
| `hugpy install-deps` | a worker box's pip extras: `hugpy[gpu-worker]` by default, `--cpu` for `hugpy[cpu-worker]`, `--profile auto` to detect; `--version X.Y.Z` pins |
| `hugpy install-engine [--cuda]` | the native llama.cpp `llama-server`/`rpc-server` binaries |
| `hugpy build` / `hugpy version` | the build identity of this install (version, sha, dirty, editable) and every distribution's version |
| `hugpy drift --install-timer` | the nightly drift check as a systemd timer (`--user` for a user unit; `--dry-run` prints the units) |

All of it resolves from PyPI at one version; `pip install "hugpy[worker]==X.Y.Z"`
pins a box to a release and central's heartbeat keeps it there afterwards.

## FAQ

**Why lockstep instead of per-package versions?** The 14 packages are one
acyclic graph with seams that change on both sides at once
(`hugpy_engine.placement`, `hugpy_video.hooks`, the wire schemas). One tag per
release means one number to compare across PyPI, checkouts and workers, and
`pip install "hugpy[server]==X.Y.Z"` names a complete, tested set. Per-package
tags would make "is this fleet consistent?" a 14-way question.

**What about the external packages?** `[[external_package]]` entries in
`py/partition.toml` (`hugpy-agent`, `hugpy-station`, `abstract-identity`,
`abstract-claude`, `abstract-gpt`, `abstract-toolserver`, `abstract-apply`)
have their own repositories, cadences and licenses. They are not built by
`pypi-publish.yml`, not checked by `--versions`, and not part of the lockstep.
`py/compat/abstract_hugpy_dev` is static-versioned (0.1.267.dev0) on purpose:
it is a dev-only alias shell and is never published.

**The check is red. Now what?**

| Message | Do |
|---|---|
| `--versions`: static `version =` / `__version__ = "…"` literal | delete the literal; the version is the tag |
| `lockstep`: distributions report different versions | one package was installed from PyPI or an old wheel: re-run `./local_install.sh --check` |
| anything reports `0.0.0+unknown` | no git metadata: clone with history (`fetch-depth: 0` in CI), or install from a wheel |
| section A: dirty / unpushed / behind | commit and push, or `git pull`; never tag what the remote does not have |
| section C: running != installed | restart the service (the install already happened) |
| section D: a worker is behind | wait one heartbeat; if it persists, the worker cannot reach central's index or runs a standalone agent (see `gguf_worker` docs) |
| section D: a worker is *ahead* | central was not upgraded; adopt the release on central first |
| `pypi-publish`: built version != tag | the tag is not on the commit you think, or the tag name is not `vX.Y.Z`; delete the tag, fix, re-tag |
