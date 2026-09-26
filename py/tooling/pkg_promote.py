#!/usr/bin/env python3
"""pkg_promote — take a known-good configuration to the fleet, and let the fleet judge it.

A configuration (``pkg_src.configs``) pins one DB version per package. The fleet,
however, installs ONE lockstep version for all 13 in-tree distributions
(``py/partition.toml`` ``[[package]]``), and central only advertises clean
(no ``+local``) versions. Promotion bridges the two:

  * **Version scheme: ``<base>.post<N>``.** ``base`` is the newest release among
    the members' ``pkg_version`` (``0.2.1`` for a mix of ``0.2.0`` / ``0.2.1+m3``
    trees); ``N`` is the next number not already used by a promotion, a wheel in
    the index dir or a git tag. Why post: it is a FINAL release to PEP 440, so it
    sorts ``0.2.1 < 0.2.1.post1 < 0.2.2.dev0 < 0.2.2``, it passes
    ``required_pkg_version`` (no ``+``), ``build_wheels._TAG_RE`` and the
    constraints/``pkg_index_has`` logic unchanged, and pip never treats it as a
    pre-release (``.devN``/``aN`` are skipped by resolvers unless named exactly,
    and ``X.Y.(Z+1).devN`` is what setuptools-scm already calls an untagged dev
    checkout, so it would collide in meaning). The same lockstep member set always
    gets the same version back (a re-promote is idempotent).
  * **Reuse.** When the lockstep members are exactly a release tag's trees (a
    ``tag:vX`` configuration with the identical member set, or every label == a
    tagged ``X``), nothing is renumbered: version ``X``, the tag's wheels, and
    only a (re)pin.
  * **Build.** Every lockstep member is materialised from the DB
    (``pkg_src.materialise``, never the working tree or git) and built with
    ``python -m build --wheel`` exactly like ``py/build_wheels.py``, with
    ``SETUPTOOLS_SCM_PRETEND_VERSION_FOR_<DIST>=<version>`` and a
    ``SOURCE_DATE_EPOCH`` fixed by the member set, so a rebuild is byte-identical
    and ``build_wheels.publish`` (immutable, idempotent) can be re-run. Wheels are
    verified with ``build_wheels.verify`` (one wheel per dist, all at the version,
    no local version) and every ``hugpy-*`` Requires-Dist must name a dist of the
    set and admit the version.

The pin. ``required_pkg_version`` is central's OWN installed ``hugpy-fleet``
version (hugpy_fleet.central.workers; there is no pin file — the
``HUGPY_REQUIRED_PKG_VERSION`` line in hugpy-api.env is read by nothing). So
moving the fleet = installing the promoted wheels into central's venv and
restarting central (the ``~/hugpy-central-adopt.sh`` shape, minus git):

    sudo -n -u hugpy /srv/hugpy/venv/bin/python -m pip install -U \\
         --upgrade-strategy only-if-needed --find-links <index> hugpy-*==V ...
    sudo -n -u hugpy /srv/hugpy/venv/bin/python -m pip install --no-deps \\
         --no-index --force-reinstall --find-links <index> hugpy-*==V ...
    sudo -n /usr/local/sbin/hugpy-gate run svc.control -p verb=restart -p unit=7002_hugpy_api.service
      (fallback: sudo -n -u hugpy sudo -n /usr/bin/systemctl restart 7002_hugpy_api.service)

Quiet central first (2026-09-23). The restart kills whatever central runs, so
a pin that would restart it first asks ``/llm/benchmark/status`` (running or
resuming) and ``/llm/admission`` (queued/running jobs). Busy -> no restart: the
row stays ``published`` with ``note = "waiting: <what>"`` (+ ``pin_log.quiet_wait``)
and the watcher's next tick retries the SAME row. Bounded by
``HUGPY_PROMOTE_QUIET_WAIT_S`` (default 1800 s) from the first wait; past it the
benchmark is checkpointed (``POST /llm/benchmark/checkpoint``) and the pin
proceeds, noted ``proceeded after quiet-wait expiry; ...``. ``--now`` /
``HUGPY_PROMOTE_NOW=1`` skip the wait. Central resumes the benchmark and
re-queues admission jobs on start, so an expired wait loses at most the lane
in flight.

No root beyond the audited hugpy-gate action (vm_mgr: ``(hugpy) NOPASSWD: ALL`` +
hugpy-gate; hugpy's own sudoers allows restarting 7002). NOTE: this replaces
central's EDITABLE installs of the 13 dists with the promoted wheels — central
then runs exactly the promoted bytes, not the live tree (re-run the adopt
script's editable step to go back to the tree). ``hugpy-downloader`` is not in
the gate's allowlist and keeps its old in-memory code until restarted by root.
Skipped when central already runs V. Workers converge on their next heartbeat
(``pip install -c constraints.txt`` from central's index).

A per-worker canary is NOT possible with today's mechanism: ``POST
/llm/workers/<id>/update {version}`` makes the worker pip-install under central's
constraints.txt (which pins the OLD version -> pip conflict), and even when it
installs, the restarted agent's register reply carries the old
``required_pkg_version`` and ``_self_update_if_needed`` converges it straight
back. So the rollout is fleet-wide and every online worker is judged.

Rollout (operator ruling: the workers decide). A pin opens a rollout on the
promotion row (``rolling_out``). ``rollout_tick`` (idempotent, called by the
watcher) judges it:

  * central: ``/api/health`` answers and ``required-version`` == V within
    ``CENTRAL_GRACE_S`` of the pin, else FAIL;
  * each worker ONLINE at pin time (baseline snapshot): back online with a fresh
    heartbeat (< ``HEARTBEAT_FRESH_S``), ``pkg_version`` == V (and, for a reuse-tag
    promotion, pkg_drift's verdict not ``drift`` when that module is present), then one tiny real
    inference through central ``/v1/chat/completions`` pinned to it with
    ``alloc.worker`` on a model that was HOT on it at pin time (max_tokens 4).
    No hot model at pin time = smoke skipped (recorded; never cold-downloads).
    one smoke attempt is made per worker. A failed/empty response is retained
    as ``unjudged`` and is never retried by the minute watcher;
  * workers offline at pin time (or never seen) are info, never a failure.

All judged online workers pass -> ``healthy``. Any FAIL -> automatic rollback:
re-pin the previous ``healthy`` promotion's version (its wheels are still on the
index; else the pin recorded before this rollout), mark the row ``rolled_back``
with ``fleet_bad`` and the per-worker evidence, and open a ``kind='rollback'``
row that is judged too but never rolled back further (``unhealthy`` needs the
operator). ``auto_promote`` never re-pushes a config (or the same member set)
whose promotion is ``fleet_bad``. Test-based ``configs.known_good`` is untouched.
No API key for the smoke call (``PKG_PROMOTE_API_KEY`` / ``HUGPY_API_KEY`` /
``~/.config/pkg_src/api_key``) -> the rollout ends ``unjudged``, not healthy.

Manual rollback = promote the previous known-good configuration
(``api_promote(conn, "<previous>", pin=True)``): its version is recomputed to the
same number, its wheels are already on the index, and only the pin moves.

    pkg_promote.py plan CONFIG            dry run (JSON)
    pkg_promote.py apply CONFIG [--pin] [--now]
                                          build + publish [+ move central's pin; --now skips
                                          the quiet-central wait]
    pkg_promote.py tick                   one rollout_tick
    pkg_promote.py ls                     promotions, newest first

DB: --dsn / $PKG_SRC_DSN, default "dbname=hugpy" (peer auth on ae).
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from datetime import datetime
from email.parser import Parser
from pathlib import Path

from psycopg import sql
from psycopg.types.json import Jsonb

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pkg_src as S  # noqa: E402  (sibling: DB mirror, versions, configs)
import build_wheels as BW  # noqa: E402  (py/build_wheels.py: the one release builder)

try:                                                   # PEP 440 ordering + specifiers
    from packaging.requirements import Requirement
    from packaging.version import InvalidVersion, Version
except ImportError:                                    # pragma: no cover
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.version import InvalidVersion, Version

SCHEMA = S.SCHEMA
OWNER_ROLE = "vm_mgr"
DEFAULT_INDEX_DIR = os.environ.get("HUGPY_PKG_INDEX_DIR",
                                   "/mnt/16T_toshiba/llm_storage/projects/pip_index")
DEFAULT_CENTRAL = os.environ.get("PKG_PROMOTE_CENTRAL", "http://127.0.0.1:7002")
CENTRAL_VENV = Path(os.environ.get("PKG_PROMOTE_CENTRAL_VENV", "/srv/hugpy/venv"))
CENTRAL_USER = "hugpy"
CENTRAL_UNIT = "7002_hugpy_api.service"
HUGPY_GATE = "/usr/local/sbin/hugpy-gate"

ROLLOUT_WINDOW_S = 15 * 60        # every online worker must pass within this of the pin
CENTRAL_GRACE_S = 5 * 60          # central must answer at the new version within this
HEARTBEAT_FRESH_S = 120           # a heartbeat older than this is not "back online"
SMOKE_TIMEOUT_S = 300             # one smoke call (may reload a model from local disk)
OPEN = ("rolling_out",)
# Quiet central (2026-09-23: promotion 11's restart killed a running benchmark).
# A pin that restarts central first waits for no benchmark / admission work,
# bounded: past this window it checkpoints the benchmark and proceeds anyway.
QUIET_WAIT_S = float(os.environ.get("HUGPY_PROMOTE_QUIET_WAIT_S", 30 * 60) or 0)
NOW_ENV = "HUGPY_PROMOTE_NOW"          # truthy = skip the quiet-central wait (operator --now)


def _env_now() -> bool:
    return (os.environ.get(NOW_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def log(msg: str) -> None:
    print(f"pkg_promote: {msg}", flush=True)


# ------------------------------------------------------------------------- schema

def ensure_promotions(cur) -> None:
    """pkg_src.promotions, created as vm_mgr so it is owned like the other tables."""
    cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f"{SCHEMA}.promotions",))
    if cur.fetchone()[0]:
        return
    cur.execute("SELECT current_user = %s OR pg_has_role(%s, 'MEMBER')", (OWNER_ROLE, OWNER_ROLE))
    as_owner = cur.fetchone()[0]
    if as_owner:
        cur.execute(sql.SQL("SET ROLE {r}").format(r=sql.Identifier(OWNER_ROLE)))
    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {s}.promotions (
            id              bigserial PRIMARY KEY,
            kind            text NOT NULL DEFAULT 'promote' CHECK (kind IN ('promote', 'rollback')),
            config          text,
            config_id       bigint REFERENCES {s}.configs(id) ON DELETE SET NULL,
            version         text NOT NULL,
            mode            text NOT NULL CHECK (mode IN ('build', 'reuse-tag')),
            members         jsonb NOT NULL,
            wheels          jsonb,
            index_dir       text,
            source_epoch    bigint,
            status          text NOT NULL DEFAULT 'published' CHECK (status IN
                            ('published', 'rolling_out', 'healthy', 'rolled_back',
                             'unhealthy', 'unjudged', 'rollback_failed', 'superseded')),
            previous_version text,
            rollback_of     bigint REFERENCES {s}.promotions(id),
            rolled_back_to  text,
            fleet_bad       boolean NOT NULL DEFAULT false,
            baseline        jsonb,
            verdict         jsonb,
            pin_log         jsonb,
            by              text,
            note            text,
            created_at      timestamptz NOT NULL DEFAULT now(),
            published_at    timestamptz,
            pinned_at       timestamptz,
            deadline        timestamptz,
            judged_at       timestamptz
        );
        CREATE INDEX IF NOT EXISTS promotions_version ON {s}.promotions (version);
        CREATE INDEX IF NOT EXISTS promotions_open ON {s}.promotions (status)
            WHERE status = 'rolling_out';
    """).format(s=sql.Identifier(SCHEMA)))
    if as_owner:
        cur.execute("RESET ROLE")


# ---------------------------------------------------------------------- the plan

def lockstep_packages() -> list[dict]:
    """partition.toml [[package]] entries (cut order), each + its package dir name."""
    return [p | {"package": Path(p["destination"]).name} for p in BW.packages()]


def config_row(cur, name: str) -> tuple[int, bool, str, str | None]:
    cur.execute(sql.SQL("SELECT id, known_good, source, git_head FROM {s}.configs WHERE name = %s")
                .format(s=sql.Identifier(SCHEMA)), (name,))
    row = cur.fetchone()
    if row is None:
        raise SystemExit(f"no configuration {name!r} (see: pkg_src.py config ls)")
    return row


def lockstep_members(cur, config: str) -> tuple[dict[str, int], dict[str, int]]:
    """({package: version_id} of the 13 lockstep dists, {other live members}).
    Retired packages (pkg_src.retired, e.g. abstract_hugpy_dev) are never promoted."""
    members = S.live_members(cur, S.config_members(cur, config))
    names = [p["package"] for p in lockstep_packages()]
    missing = [n for n in names if n not in members]
    if missing:
        raise SystemExit(f"config {config}: lockstep package(s) not pinned: {', '.join(missing)}")
    return ({n: members[n] for n in names},
            {k: v for k, v in members.items() if k not in names})


def version_rows(cur, ids) -> dict[int, dict]:
    cur.execute(sql.SQL("SELECT id, package, label, pkg_version, created_at FROM {s}.versions "
                        "WHERE id = ANY(%s)").format(s=sql.Identifier(SCHEMA)), (list(ids),))
    return {r[0]: dict(package=r[1], label=r[2], pkg_version=r[3], created_at=r[4])
            for r in cur.fetchall()}


def _v(s: str | None) -> Version | None:
    try:
        return Version(s) if s else None
    except InvalidVersion:
        return None


def git_tags() -> set[str]:
    try:
        out = S._git(S.git_root(S.PY_ROOT), "tag", "-l", "v[0-9]*")
    except Exception:                                  # noqa: BLE001 — no git: no tags known
        return set()
    return {t.strip()[1:] for t in out.splitlines() if t.strip()}


def tag_epoch(version: str) -> int | None:
    try:
        return int(S._git(S.git_root(S.PY_ROOT), "log", "-1", "--format=%ct", f"v{version}").strip())
    except Exception:                                  # noqa: BLE001
        return None


def reused_tag(cur, members: dict[str, int], rows: dict[int, dict]) -> str | None:
    """Release version X when the lockstep members ARE tag vX's trees, else None."""
    cur.execute(sql.SQL("SELECT c.source, jsonb_object_agg(m.package, m.version_id) "
                        "FROM {s}.configs c JOIN {s}.config_members m ON m.config_id = c.id "
                        "WHERE c.source LIKE 'tag:v%%' GROUP BY c.id ORDER BY c.id DESC")
                .format(s=sql.Identifier(SCHEMA)))
    for source, tag_members in cur.fetchall():
        if all(tag_members.get(p) == v for p, v in members.items()):
            return source[len("tag:v"):]
    labels = {rows[v]["label"] for v in members.values()}
    if len(labels) == 1:
        (label,) = labels
        if "+" not in label and label in git_tags():
            return label
    return None


def index_versions(index_dir: Path) -> dict[str, set[str]]:
    """{version: {normalised dist}} of the wheels in ``index_dir``."""
    out: dict[str, set[str]] = {}
    try:
        names = os.listdir(index_dir)
    except OSError:
        return out
    for fn in names:
        if fn.endswith(".whl"):
            parts = fn[:-4].split("-")
            if len(parts) >= 2:
                out.setdefault(parts[1], set()).add(S.norm_dist(parts[0]))
    return out


def index_has(index_dir: Path, version: str) -> bool:
    """Same rule as central's pkg_index_has: every lockstep wheel of ``version``."""
    want = {S.norm_dist(p["distribution"]) for p in lockstep_packages()}
    return want <= index_versions(index_dir).get(version, set())


def allocate_version(cur, members: dict[str, int], base: str, index_dir: Path) -> str:
    """``base.postN``: reuse the number a previous promotion of this exact member set
    got; otherwise the next N not used by a promotion, an index wheel or a git tag."""
    cur.execute(sql.SQL("SELECT version FROM {s}.promotions WHERE mode = 'build' AND members = %s "
                        "ORDER BY id DESC LIMIT 1").format(s=sql.Identifier(SCHEMA)),
                (Jsonb(members),))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(sql.SQL("SELECT version FROM {s}.promotions").format(s=sql.Identifier(SCHEMA)))
    used = {r[0] for r in cur.fetchall()} | set(index_versions(index_dir)) | git_tags()
    top = 0
    for u in used:
        v = _v(u)
        if v is not None and v.base_version == base and v.post is not None:
            top = max(top, v.post)
    return f"{base}.post{top + 1}"


def plan_promotion(cur, config: str, index_dir: Path) -> dict:
    """Everything a promotion would do, without doing it."""
    cid, good, source, head = config_row(cur, config)
    members, others = lockstep_members(cur, config)
    rows = version_rows(cur, members.values())
    tag = reused_tag(cur, members, rows)
    bases = [v for v in (_v(r["pkg_version"]) for r in rows.values()) if v is not None]
    if not bases:
        raise SystemExit(f"config {config}: no member carries a release version")
    base = max(bases).base_version
    if tag:
        mode, version, epoch = "reuse-tag", tag, tag_epoch(tag)
    else:
        mode, version = "build", allocate_version(cur, members, base, index_dir)
        epoch = int(max(r["created_at"] for r in rows.values()).timestamp())
    have = index_has(index_dir, version)
    dists = [p["distribution"] for p in lockstep_packages()]
    wheels = [f"{BW.file_stem(d)}{version}-py3-none-any.whl" for d in dists]
    return {
        "config": config, "config_id": cid, "known_good": good, "source": source,
        "mode": mode, "version": version, "base": base, "source_date_epoch": epoch,
        "members": members,
        "labels": {p: rows[v]["label"] for p, v in members.items()},
        "not_lockstep": sorted(others),
        "dists": dists, "wheels": wheels,
        "index_dir": str(index_dir), "index_has_version": have,
        "index_writable": os.access(index_dir, os.W_OK),
        "build_needed": not have,
    }


# --------------------------------------------------------------------- build path

def build_python() -> str:
    """Interpreter with the ``build`` package (isolated builds, same as build_wheels)."""
    cands = [os.environ.get("PKG_PROMOTE_PYTHON"), sys.executable]
    env = S.current_env()
    if env is not None:
        cands.append(str(env / "venv" / "bin" / "python"))
    cands.append(str(CENTRAL_VENV / "bin" / "python"))
    for c in filter(None, cands):
        if Path(c).exists() and subprocess.run([c, "-c", "import build"],
                                               capture_output=True).returncode == 0:
            return c
    raise SystemExit("no interpreter with the `build` package — set PKG_PROMOTE_PYTHON")


def build_members(conn, members: dict[str, int], version: str, out: Path, epoch: int | None,
                  python: str | None = None) -> list[Path]:
    """Materialise ``members`` from the DB and build one wheel per lockstep dist at
    ``version`` into ``out``. Returns the verified wheels (build_wheels.verify)."""
    python = python or build_python()
    pkgs = lockstep_packages()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pkg_promote_src_") as tmp:
        root = Path(tmp) / "py"
        with conn.cursor() as cur:
            dirs = S.materialise(cur, members, root)
        console_fresh_or_rebuild(dirs.values())
        for f in root.rglob("*"):                       # modes as version identity sees them:
            if f.is_file() and not f.is_symlink():      # exec bit only (umask-independent)
                os.chmod(f, 0o755 if S.is_exec(f.stat().st_mode) else 0o644)
        env ={k: v for k, v in os.environ.items() if not k.startswith("SETUPTOOLS_SCM_PRETEND")}
        if epoch:
            env["SOURCE_DATE_EPOCH"] = str(epoch)
        for p in pkgs:
            d = dirs.get(p["package"])
            if d is None:
                raise SystemExit(f"{p['package']}: version {members[p['package']]} has no files")
            name = S.dist_name(d)
            if S.norm_dist(name or "") != S.norm_dist(p["distribution"]):
                raise SystemExit(f"{d}: pyproject name {name!r} != partition {p['distribution']!r}")
            key = re.sub(r"[-_.]+", "_", name).upper()
            penv = env | {f"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_{key}": version}
            r = subprocess.run([python, "-m", "build", "--wheel", "--outdir", str(out), str(d)],
                               env=penv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               text=True)
            if r.returncode:
                sys.stderr.write(r.stderr[-4000:])
                raise SystemExit(f"building {p['distribution']} failed (rc {r.returncode})")
            log(f"built {p['distribution']} {version}")
    _, files = BW.verify(out, pkgs, version, sdist=False, strict=True)
    check_pins(files, version)
    return files


def console_fresh_or_rebuild(member_dirs) -> None:
    """The console bundle ships as tracked files; the React source lives in the
    live tree. When the hugpy-server member being built carries the LIVE tree's
    bundle (same index.html bytes) and that bundle is STALE against react/ui/src
    (a UI edit nobody rebuilt), SELF-HEAL: this is the fleet deploy path and dev
    is live, so an edit must reach the fleet without a human running npm. We
    rebuild the live bundle from react/ui/src (build_console.py --from-source
    --only /) and copy the fresh bundle into each carrier member dir so the wheel
    ships it. If the rebuild fails, the promotion FAILS LOUDLY with the real npm
    output (the watcher logs the SystemExit) instead of retry-spinning on "stale".
    The staleness check itself stays a hard guard on the manual/PyPI path
    (build_wheels.console_stale_reason). A member whose bundle differs from the
    live one (an older version rebuilt for rollback) is not comparable to today's
    React source and is not judged."""
    live_index = BW.CONSOLE_DIST / "index.html"
    if not live_index.is_file():
        return
    carriers = [Path(d) for d in member_dirs
                if (Path(d) / "src" / "hugpy_server" / "console_dist" / "index.html").is_file()
                and sha256(Path(d) / "src" / "hugpy_server" / "console_dist" / "index.html")
                == sha256(live_index)]
    if not carriers:
        return
    stale = BW.console_stale_reason()
    if not stale:
        return
    rebuild_console(stale)                              # raises SystemExit on failure
    fresh = BW.CONSOLE_DIST
    for d in carriers:
        dest = d / "src" / "hugpy_server" / "console_dist"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(fresh, dest)


def rebuild_console(stale: str) -> None:
    """Run build_console.py --from-source --only / on the LIVE tree to refresh the
    console bundle (reusing the existing node_modules; the deploy user owns write
    access to react/ui and console_dist by ACL). Loud on failure: SystemExit with
    the real npm/webpack output, which the watcher prints to its journal."""
    log(f"console self-heal: {stale}")
    tool = BW.CONSOLE_TOOL
    if not tool.is_file():
        raise SystemExit(f"hugpy-server: console is stale and {tool} is missing: {stale}")
    # npm lives in the deploy user's ~/.local/bin, node in /usr/local/bin; neither
    # is guaranteed on a systemd user unit's PATH, so name them explicitly.
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(p for p in (str(Path.home() / ".local" / "bin"),
                                              "/usr/local/bin", "/usr/bin",
                                              env.get("PATH", "")) if p)
    cmd = [sys.executable, str(tool), "--from-source", "--skip-install", "--only", "/"]
    log(f"console self-heal: + {' '.join(cmd)}")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode:
        tail = ((r.stdout or "") + (r.stderr or ""))[-4000:]
        raise SystemExit(f"hugpy-server: console_dist rebuild FAILED (rc {r.returncode}) — deploy "
                         f"blocked until the React build succeeds. was: {stale}\n{tail}")
    log("console self-heal: rebuilt console_dist from react/ui/src")


def check_pins(files: list[Path], version: str) -> None:
    """Every wheel says Version: ``version`` and each hugpy-* Requires-Dist names a
    dist of the set whose specifier admits ``version``."""
    dists = {S.norm_dist(p["distribution"]) for p in lockstep_packages()}
    problems = []
    for f in files:
        with zipfile.ZipFile(f) as z:
            meta_name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
            meta = Parser().parsestr(z.read(meta_name).decode())
        if meta["Version"] != version:
            problems.append(f"{f.name}: METADATA Version {meta['Version']} != {version}")
        for req in meta.get_all("Requires-Dist") or []:
            r = Requirement(req)
            n = S.norm_dist(r.name)
            if not n.startswith("hugpy"):
                continue
            if n not in dists:
                problems.append(f"{f.name}: requires {r.name}, not a lockstep dist")
            elif r.specifier and not r.specifier.contains(version, prereleases=True):
                problems.append(f"{f.name}: {req} excludes {version}")
    if problems:
        raise SystemExit("inter-package pins do not resolve:\n  " + "\n  ".join(problems))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def publish_wheels(files: list[Path], index_dir: Path) -> list[str]:
    """build_wheels.publish (immutable, idempotent). Without write access (the vm_mgr
    service), copy as the ``hugpy`` user, which owns the index's existing wheels."""
    if os.access(index_dir, os.W_OK):
        return [p.name for p in BW.publish(files, index_dir)]
    copied = []
    for f in files:
        dst = index_dir / f.name
        if dst.exists():
            if sha256(f) == sha256(dst):
                continue
            raise SystemExit(f"{dst} exists with different bytes; an artifact for a version "
                             f"is immutable — remove it by hand to replace it")
        os.chmod(f, 0o644)
        os.chmod(f.parent, 0o755)
        subprocess.run(["sudo", "-n", "-u", CENTRAL_USER, "cp", "--preserve=timestamps",
                        str(f), str(dst)], check=True)
        copied.append(f.name)
    return copied


def build_and_publish(conn, plan: dict, index_dir: Path, python: str | None = None) -> dict:
    """The build+publish half of a promotion — NO known-good check (callers do it).
    Returns {filename: sha256} of the version's wheels as they sit in the index."""
    version = plan["version"]
    with tempfile.TemporaryDirectory(prefix="pkg_promote_whl_") as tmp:
        files = build_members(conn, plan["members"], version, Path(tmp), plan["source_date_epoch"],
                              python)
        copied = publish_wheels(files, index_dir)
        log(f"published {len(copied)} new wheel(s) ({len(files) - len(copied)} already present) "
            f"-> {index_dir}")
    return index_wheels(index_dir, version)


def index_wheels(index_dir: Path, version: str) -> dict[str, str]:
    out = {}
    for p in lockstep_packages():
        f = index_dir / f"{BW.file_stem(p['distribution'])}{version}-py3-none-any.whl"
        if f.exists():
            out[f.name] = sha256(f)
    return out


# ------------------------------------------------------------------ fleet access

def _http(method: str, url: str, body: dict | None = None, headers: dict | None = None,
          timeout: float = 15) -> tuple[int, dict | str | None]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            status = r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read().decode("utf-8", "replace"), e.code
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        return 0, f"{type(e).__name__}: {e}"
    try:
        return status, json.loads(raw) if raw else None
    except ValueError:
        return status, raw[-2000:]


def api_key() -> str | None:
    k = os.environ.get("PKG_PROMOTE_API_KEY") or os.environ.get("HUGPY_API_KEY")
    if k:
        return k.strip()
    f = Path(os.environ.get("PKG_PROMOTE_API_KEY_FILE", Path.home() / ".config/pkg_src/api_key"))
    try:
        return f.read_text().strip() or None
    except OSError:
        return None


class LiveFleet:
    """The real central + fleet. Every production side effect lives in ``pin``."""

    def __init__(self, central: str = DEFAULT_CENTRAL):
        self.central = central.rstrip("/")

    def required_version(self) -> str | None:
        st, body = _http("GET", f"{self.central}/api/llm/workers/required-version")
        return body.get("required_pkg_version") if st == 200 and isinstance(body, dict) else None

    def health(self) -> dict:
        st, body = _http("GET", f"{self.central}/api/health")
        build = (body or {}).get("build") if isinstance(body, dict) else None
        return {"up": st == 200, "http": st, "version": (build or {}).get("version"),
                "required": self.required_version() if st == 200 else None}

    def workers(self) -> list[dict] | None:
        st, body = _http("GET", f"{self.central}/api/llm/workers")
        if st != 200:
            return None
        ws = body if isinstance(body, list) else (body or {}).get("workers") or []
        return [{"id": w.get("id"), "name": w.get("name"), "status": w.get("status"),
                 "pkg_version": w.get("pkg_version"), "version_ok": w.get("version_ok"),
                 "last_seen": w.get("last_seen"), "hot": list(w.get("loaded_models") or [])}
                for w in ws]

    def smoke(self, worker: str, model: str) -> dict:
        key = api_key()
        if not key:
            return {"ok": False, "transient": True, "no_key": True, "error": "no API key"}
        t0 = time.monotonic()
        st, body = _http("POST", f"{self.central}/v1/chat/completions",
                         {"model": model, "max_tokens": 4, "temperature": 0,
                          "messages": [{"role": "user", "content": "Reply with: OK"}],
                          "alloc": {"worker": worker}},
                         headers={
                             "Authorization": f"Bearer {key}",
                             "X-Hugpy-Client-Process": "pkg-src-watch/pkg-promote",
                             "X-Hugpy-Client-Pid": str(os.getpid()),
                             "X-Hugpy-Client-User": getpass.getuser(),
                             "X-Hugpy-Client-Request": str(uuid.uuid4()),
                             "X-Hugpy-Client-Task": f"worker-rollout-smoke:{worker}:{model}",
                             "X-Hugpy-Client-Platform": "worker-package-rollout",
                         }, timeout=SMOKE_TIMEOUT_S)
        secs = round(time.monotonic() - t0, 1)
        ok = st == 200 and isinstance(body, dict) and bool(body.get("choices"))
        out = {"ok": ok, "http": st, "seconds": secs, "model": model,
               "transient": st in (0, 429, 502, 503, 504)}
        if not ok:
            out["error"] = (json.dumps(body) if isinstance(body, dict) else str(body))[-500:]
        return out

    def _headers(self) -> dict:
        key = api_key()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def busy(self) -> dict | None:
        """What a central restart would kill right now, or None when quiet:
        a running/resuming benchmark, or admission jobs queued/running (they
        start benchmarks). ``{"reason": str, "benchmark": run_id?, ...}``."""
        reasons, out = [], {}
        st, body = _http("GET", f"{self.central}/api/llm/benchmark/status", headers=self._headers())
        if st != 200 or not isinstance(body, dict):
            # FAIL SAFE (incident 2026-09-24): the run-state could not be read
            # (central warming up, a transient 502). A restart is exactly what
            # kills a running benchmark, so an UNKNOWN state must DEFER the restart
            # — never proceed as if quiet. The bounded quiet-wait still lets the
            # pin through if central stays unreadable past the window.
            return {"reason": f"benchmark run-state unreadable (http {st}); "
                              f"deferring restart until central answers",
                    "status_unreadable": True}
        if body.get("status") in ("running", "resuming"):
            prog = body.get("progress") or {}
            out["benchmark"] = body.get("run_id")
            reasons.append(f"benchmark {body.get('run_id')} {body.get('status')} "
                           f"({prog.get('completed', 0)}/{prog.get('total', 0)})")
        st, body = _http("GET", f"{self.central}/api/llm/admission?status=pending",
                         headers=self._headers(), timeout=60)
        queue = (body or {}).get("queue") if st == 200 and isinstance(body, dict) else None
        if queue:
            running = [j for j in queue if j.get("status") == "running"]
            out["admission"] = [j.get("id") for j in queue]
            reasons.append(f"admission {len(running)} running / {len(queue) - len(running)} queued"
                           + (f" (job {running[0].get('id')} {running[0].get('model_key')})" if running else ""))
        if not reasons:
            return None
        return out | {"reason": "; ".join(reasons)}

    def checkpoint(self, reason: str) -> dict:
        st, body = _http("POST", f"{self.central}/api/llm/benchmark/checkpoint", {"reason": reason},
                         headers=self._headers())
        return body if st == 200 and isinstance(body, dict) else {"checkpointed": False, "http": st,
                                                                  "error": str(body)[:300]}

    def restart_needed(self, version: str) -> bool:
        installed = self.central_installed()
        return not (installed and all(v == version for v in installed.values()))

    def central_installed(self) -> dict[str, str | None]:
        code = ("import importlib.metadata as m, json, sys\n"
                "out = {}\n"
                "for n in sys.argv[1:]:\n"
                "    try: out[n] = m.version(n)\n"
                "    except m.PackageNotFoundError: out[n] = None\n"
                "print(json.dumps(out))")
        r = subprocess.run([str(CENTRAL_VENV / "bin" / "python"), "-c", code,
                            *[p["distribution"] for p in lockstep_packages()]],
                           capture_output=True, text=True)
        return json.loads(r.stdout) if r.returncode == 0 else {}

    def pin_commands(self, version: str, index_dir: Path) -> list[list[str]]:
        specs = [f"{p['distribution']}=={version}" for p in lockstep_packages()]
        pip = ["sudo", "-n", "-u", CENTRAL_USER, "env", f"HOME=/srv/{CENTRAL_USER}",
               str(CENTRAL_VENV / "bin" / "python"), "-m", "pip", "install", "-q",
               "--disable-pip-version-check"]
        return [pip + ["-U", "--upgrade-strategy", "only-if-needed", "--find-links",
                       str(index_dir), *specs],
                pip + ["--no-deps", "--no-index", "--force-reinstall", "--find-links",
                       str(index_dir), *specs],
                ["sudo", "-n", HUGPY_GATE, "run", "svc.control", "-p", "verb=restart",
                 "-p", f"unit={CENTRAL_UNIT}"]]

    def pin(self, version: str, index_dir: Path) -> dict:
        """Make central run (and so require) ``version``. PRODUCTION CHANGE."""
        if not index_has(index_dir, version):
            return {"ok": False, "error": f"index {index_dir} lacks lockstep wheels of {version}"}
        installed = self.central_installed()
        steps = []
        if installed and all(v == version for v in installed.values()):
            steps.append({"step": "install", "skipped": f"central already runs {version}"})
            return {"ok": True, "steps": steps, "restarted": False}
        cmds = self.pin_commands(version, index_dir)
        for name, cmd in zip(("resolve", "exact-bytes"), cmds[:2]):
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            steps.append({"step": name, "rc": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]})
            if r.returncode:
                return {"ok": False, "steps": steps}
        r = subprocess.run(cmds[2], capture_output=True, text=True, timeout=300)
        steps.append({"step": "restart(gate)", "rc": r.returncode,
                      "tail": (r.stdout + r.stderr)[-800:]})
        if r.returncode:
            r = subprocess.run(["sudo", "-n", "-u", CENTRAL_USER, "sudo", "-n",
                                "/usr/bin/systemctl", "restart", CENTRAL_UNIT],
                               capture_output=True, text=True, timeout=300)
            steps.append({"step": "restart(hugpy sudoers)", "rc": r.returncode,
                          "tail": (r.stdout + r.stderr)[-800:]})
            if r.returncode:
                return {"ok": False, "steps": steps}
        for _ in range(60):                              # up + advertising the new pin
            time.sleep(3)
            if self.required_version() == version:
                return {"ok": True, "steps": steps, "restarted": True}
        return {"ok": False, "steps": steps, "error": f"central not advertising {version} after restart"}


# ------------------------------------------------------------------------ the API

def _row(cur, pid: int) -> dict:
    cur.execute(sql.SQL("SELECT * FROM {s}.promotions WHERE id = %s")
                .format(s=sql.Identifier(SCHEMA)), (pid,))
    cols = [d.name for d in cur.description]
    r = cur.fetchone()
    return {c: (v.isoformat() if isinstance(v, datetime) else v) for c, v in zip(cols, r)}


def open_rollout(cur) -> dict | None:
    cur.execute(sql.SQL("SELECT id FROM {s}.promotions WHERE status = 'rolling_out' "
                        "ORDER BY id LIMIT 1").format(s=sql.Identifier(SCHEMA)))
    r = cur.fetchone()
    return _row(cur, r[0]) if r else None


def fleet_bad(cur, config: str, members: dict[str, int]) -> int | None:
    cur.execute(sql.SQL("SELECT id FROM {s}.promotions WHERE fleet_bad AND kind = 'promote' "
                        "AND (config = %s OR members = %s) ORDER BY id DESC LIMIT 1")
                .format(s=sql.Identifier(SCHEMA)), (config, Jsonb(members)))
    r = cur.fetchone()
    return r[0] if r else None


def baseline(fleet) -> dict:
    """Who is online at pin time, what is hot, and which hot model ANSWERED before the pin.
    Only that model is the smoke target afterwards: a model that was already broken (too big
    for the box, bad file) must never count as the new version's regression."""
    ws = fleet.workers() or []
    now = time.time()
    out = {}
    for w in ws:
        online = w["status"] == "online" and now - float(w.get("last_seen") or 0) < HEARTBEAT_FRESH_S
        smoke_model, tried = None, []
        for m in (w["hot"] or [])[:3] if online else []:
            r = fleet.smoke(w["name"], m)
            tried.append({"model": m, "ok": r["ok"], **({} if r["ok"] else {"error": str(r.get("error"))[:300]})})
            if r["ok"] or r.get("no_key"):             # no key: judged 'unjudged' later, as before
                smoke_model = m
                break
        out[w["name"]] = {"online": online, "hot": w["hot"], "pkg_version": w["pkg_version"],
                          "smoke_model": smoke_model, "pre_pin_smoke": tried}
    return out


def api_promote(conn, config: str, apply: bool = False, pin: bool = False,
                index_dir=DEFAULT_INDEX_DIR, central: str = DEFAULT_CENTRAL,
                by: str | None = None, fleet=None, python: str | None = None,
                now: bool = False) -> dict:
    """Promote ``config`` (the dev tree as recorded) to the fleet as ONE lockstep version.

    Not gated on ``configs.known_good`` (that gates PyPI uploads). Default: dry run — the plan
    (version, dists, wheels to build/publish, current pin -> new pin, commands).
    ``apply``: build from the DB tree + publish into ``index_dir``, record the
    promotion. ``pin`` (implies apply): move central's required version (the
    production change; see the module doc for the exact mechanism) and open a
    rollout that ``rollout_tick`` judges from the workers' health.
    Rollback = promote the previous known-good configuration with ``pin=True``.
    A pin waits for a quiet central (``quiet_gate``; ``now`` skips the wait);
    a waiting promotion row is reused by the next attempt, not duplicated."""
    fleet = fleet or LiveFleet(central)
    index_dir = Path(index_dir)
    by = by or getpass.getuser()
    with conn.cursor() as cur:
        ensure_promotions(cur)
        conn.commit()
        plan = plan_promotion(cur, config, index_dir)
        # Dev IS live: the fleet runs what /srv/hugpy/src/hugpy/py holds, so a fleet
        # promotion is not gated on configs.known_good — known-good gates PyPI
        # uploads only (operator ruling 2026-09-25).
        current = fleet.required_version()
        plan |= {"current_pin": current, "new_pin": plan["version"],
                 "pin_changes": current != plan["version"],
                 "fleet_bad_promotion": fleet_bad(cur, config, plan["members"]),
                 "open_rollout": (open_rollout(cur) or {}).get("id")}
        if isinstance(fleet, LiveFleet):
            plan["pin_commands"] = [" ".join(c) for c in fleet.pin_commands(plan["version"], index_dir)]
    if not (apply or pin):
        return plan | {"dry_run": True}
    if pin and plan["open_rollout"]:
        raise SystemExit(f"rollout {plan['open_rollout']} is still open — wait for rollout_tick "
                         f"to judge it")
    if pin:
        waiting = waiting_promotion(conn, plan["version"])
        if waiting:
            log(f"promotion {waiting}: {config} -> {plan['version']} retries its pin (was waiting)")
            return plan | {"dry_run": False, "promotion": waiting,
                           "wheels_sha256": index_wheels(index_dir, plan["version"])} \
                | start_rollout(conn, waiting, fleet, index_dir, now=now)
    wheels = index_wheels(index_dir, plan["version"])
    if plan["build_needed"]:
        if not os.access(index_dir, os.W_OK) and shutil.which("sudo") is None:
            raise SystemExit(f"{index_dir} is not writable by {by}")
        wheels = build_and_publish(conn, plan, index_dir, python)
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            INSERT INTO {s}.promotions (config, config_id, version, mode, members, wheels, index_dir,
                                        source_epoch, previous_version, by, published_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now()) RETURNING id""")
            .format(s=sql.Identifier(SCHEMA)),
            (config, plan["config_id"], plan["version"], plan["mode"], Jsonb(plan["members"]),
             Jsonb(wheels), str(index_dir), plan["source_date_epoch"], current, by))
        pid = cur.fetchone()[0]
    conn.commit()
    log(f"promotion {pid}: {config} -> {plan['version']} ({plan['mode']}, {len(wheels)} wheels)")
    out = plan | {"dry_run": False, "promotion": pid, "wheels_sha256": wheels}
    if pin:
        out |= start_rollout(conn, pid, fleet, index_dir, now=now)
    return out


def waiting_promotion(conn, version: str) -> int | None:
    """The newest unpinned promotion of ``version`` still waiting for a quiet central."""
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT id FROM {s}.promotions WHERE version = %s AND kind = 'promote' "
                            "AND status = 'published' AND pinned_at IS NULL "
                            "AND pin_log ? 'quiet_wait' ORDER BY id DESC LIMIT 1")
                    .format(s=sql.Identifier(SCHEMA)), (version,))
        r = cur.fetchone()
    return r[0] if r else None


def quiet_gate(row: dict, fleet, now: bool = False, wait_s: float | None = None,
               clock=time.time) -> dict:
    """May the pin restart central now? ``{"go": bool, "note": str|None,
    "quiet_wait": {...}}``. Busy (benchmark running/resuming, admission jobs
    queued/running) -> wait, recorded on the row, retried by the next watcher
    tick; bounded by ``wait_s`` (``HUGPY_PROMOTE_QUIET_WAIT_S``, default 30 min)
    from the first wait, after which the benchmark is checkpointed and the pin
    proceeds. ``now`` / ``HUGPY_PROMOTE_NOW`` skip the wait. A pin that would
    not restart central (already runs the version) never waits."""
    wait_s = QUIET_WAIT_S if wait_s is None else wait_s
    prior = (row.get("pin_log") or {}).get("quiet_wait") if isinstance(row.get("pin_log"), dict) else None
    if now or _env_now():
        return {"go": True, "note": "quiet-central wait skipped (--now)" if prior else None,
                "quiet_wait": (prior or {}) | {"skipped": "now"}}
    needed = getattr(fleet, "restart_needed", None)
    if needed is not None and not needed(row["version"]):
        return {"go": True, "note": None, "quiet_wait": prior}
    check = getattr(fleet, "busy", None)
    busy = check() if check is not None else None
    t = clock()
    if not busy:
        waited = int(t - float(prior["since"])) if prior and prior.get("since") is not None else 0
        return {"go": True, "note": f"central quiet after {waited}s wait" if prior else None,
                "quiet_wait": (prior | {"quiet_at": t}) if prior else None}
    since = float(prior["since"]) if prior and prior.get("since") is not None else t
    state = {"since": since, "last": t, "busy": busy, "window_s": wait_s,
             "checks": int((prior or {}).get("checks") or 0) + 1}
    if t - since < wait_s:
        return {"go": False, "note": f"waiting: {busy['reason']} (quiet-wait {int(t - since)}s of "
                                     f"{int(wait_s)}s)", "quiet_wait": state}
    what = []
    if busy.get("benchmark"):
        ck = fleet.checkpoint(f"promotion {row.get('id')} ({row['version']}) restarts central: "
                              f"quiet-wait {int(wait_s)}s expired") if hasattr(fleet, "checkpoint") else {}
        state["checkpoint"] = ck
        what.append(f"benchmark {busy['benchmark']} "
                    + ("checkpointed" if ck.get("checkpointed") else
                       f"NOT checkpointed ({ck.get('error') or ck.get('status') or 'no answer'})"))
    if busy.get("admission"):
        what.append(f"admission jobs {', '.join(map(str, busy['admission'][:5]))} re-queue on restart")
    return {"go": True, "note": "proceeded after quiet-wait expiry; " + "; ".join(what or [busy["reason"]]),
            "quiet_wait": state | {"expired_at": t}}


def start_rollout(conn, pid: int, fleet, index_dir: Path, now: bool = False) -> dict:
    """Snapshot the fleet, move the pin, open the rollout on promotion ``pid``
    — once central is quiet (``quiet_gate``); while it is busy the row stays
    ``published`` with a ``waiting: ...`` note and this returns ``waiting``."""
    with conn.cursor() as cur:
        row = _row(cur, pid)
    gate = quiet_gate(row, fleet, now=now)
    pin_log = dict(row.get("pin_log") or {}) if isinstance(row.get("pin_log"), dict) else {}
    if gate["quiet_wait"] is not None:
        pin_log["quiet_wait"] = gate["quiet_wait"]
    if not gate["go"]:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("UPDATE {s}.promotions SET pin_log = %s, note = %s WHERE id = %s")
                        .format(s=sql.Identifier(SCHEMA)), (Jsonb(pin_log), gate["note"], pid))
        conn.commit()
        log(f"promotion {pid}: {gate['note']}")
        return {"pinned": False, "waiting": gate["note"], "quiet_wait": gate["quiet_wait"]}
    if gate["note"]:
        log(f"promotion {pid}: {gate['note']}")
    base = baseline(fleet)
    res = fleet.pin(row["version"], index_dir)
    if gate["quiet_wait"] is not None:
        res = dict(res) | {"quiet_wait": gate["quiet_wait"]}
    note = gate["note"]
    with conn.cursor() as cur:
        if res.get("ok"):
            cur.execute(sql.SQL("UPDATE {s}.promotions SET status = 'rolling_out', pinned_at = now(), "
                                "deadline = now() + make_interval(secs => %s), baseline = %s, "
                                "pin_log = %s, verdict = '{{}}', note = %s WHERE id = %s")
                        .format(s=sql.Identifier(SCHEMA)),
                        (ROLLOUT_WINDOW_S, Jsonb(base), Jsonb(res), note, pid))
        else:
            cur.execute(sql.SQL("UPDATE {s}.promotions SET pin_log = %s, note = %s WHERE id = %s")
                        .format(s=sql.Identifier(SCHEMA)),
                        (Jsonb(res), "pin failed" + (f"; {note}" if note else ""), pid))
    conn.commit()
    log(f"promotion {pid}: pin {row['version']} {'ok — rollout open' if res.get('ok') else 'FAILED'}")
    return {"pinned": bool(res.get("ok")), "pin_log": res, "baseline": base, "note": note}


def auto_promote(conn, config: str, index_dir=DEFAULT_INDEX_DIR, central: str = DEFAULT_CENTRAL,
                 fleet=None, python: str | None = None, now: bool = False) -> dict:
    """Hook for "config just became known-good": build + publish + pin + open a
    rollout. Refuses a config the fleet already rejected (fleet_bad), and waits
    (returns skipped) while another rollout is open. Idempotent: a config whose
    version is already the pin and was judged healthy is a no-op."""
    fleet = fleet or LiveFleet(central)
    with conn.cursor() as cur:
        ensure_promotions(cur)
        conn.commit()
        members, _ = lockstep_members(cur, config)
        bad = fleet_bad(cur, config, members)
        if bad:
            log(f"auto: {config} refused — promotion {bad} was rolled back by the fleet")
            return {"config": config, "skipped": "fleet_bad", "promotion": bad}
        busy = open_rollout(cur)
        if busy:
            log(f"auto: {config} waits — rollout {busy['id']} ({busy['version']}) still open")
            return {"config": config, "skipped": "rollout_open", "promotion": busy["id"]}
    plan = api_promote(conn, config, index_dir=index_dir, central=central, fleet=fleet)
    with conn.cursor() as cur:
        # judged already (unjudged = converged but no smoke key): never re-open every tick
        cur.execute(sql.SQL("SELECT id FROM {s}.promotions WHERE version = %s "
                            "AND status IN ('healthy', 'unjudged') ORDER BY id DESC LIMIT 1")
                    .format(s=sql.Identifier(SCHEMA)), (plan["version"],))
        healthy = cur.fetchone()
    if healthy and not plan["pin_changes"]:
        log(f"auto: {config} = {plan['version']} is already the judged pin")
        return {"config": config, "skipped": "already_pinned", "promotion": healthy[0]}
    return api_promote(conn, config, pin=True, index_dir=index_dir, central=central,
                       by="auto", fleet=fleet, python=python, now=now)


# ------------------------------------------------------------------ rollout judge

def _drift_verdicts(conn, central: str, config: str | None) -> dict[str, str]:
    """{worker: verdict} from tooling/pkg_drift.py (read-only) when it is present.
    pkg_drift maps member trees to RELEASE TAGS, so it only speaks for reuse-tag
    promotions; a ``.postN`` build is judged on pkg_version == target alone."""
    try:
        import pkg_drift
    except ImportError:
        return {}
    try:
        rep = pkg_drift.api_drift(conn, config=config, central=central)
    except (Exception, SystemExit) as e:               # noqa: BLE001 — optional signal
        log(f"pkg_drift unavailable: {type(e).__name__}: {e}")
        return {}
    return {w.get("worker") or w.get("name"): w.get("verdict")
            for w in (rep.get("workers") or []) if isinstance(w, dict)}


def judge_worker(fleet, name: str, target: str, base: dict, live: dict | None, prev: dict,
                 drift: str | None, now: float, past_deadline: bool) -> dict:
    """One worker's verdict: pass | fail | pending | info (+ evidence)."""
    v = dict(prev)
    if not base.get("online"):
        return {"verdict": "info", "why": "offline at pin time"}
    if v.get("verdict") in ("pass", "fail"):
        return v
    fresh = bool(live) and live["status"] == "online" \
        and now - float(live.get("last_seen") or 0) < HEARTBEAT_FRESH_S
    converged = bool(live) and live.get("pkg_version") == target and drift != "drift"
    v.update(online=fresh, pkg_version=(live or {}).get("pkg_version"), drift=drift)
    if fresh and converged:
        model = base.get("smoke_model")
        if not model:
            v.update(verdict="pass", smoke="skipped: no hot model answered before the pin")
            return v
        # The watcher runs every minute. Repeating a model request after a
        # timeout/empty answer cannot make the already-converged package more
        # correct, and can hammer a model whose provider/session is unavailable.
        # Persist the first result in the rollout verdict and stop probing.
        previous_smoke = v.get("smoke")
        if isinstance(previous_smoke, dict):
            if previous_smoke.get("ok"):
                v["verdict"] = "pass"
            else:
                detail = previous_smoke.get("error") or f"HTTP {previous_smoke.get('http', 'unknown')}"
                v.update(verdict="unjudged", why=f"single smoke attempt did not pass; not retried: {detail}")
            return v
        r = fleet.smoke(name, model)
        v["smoke"] = r
        if r["ok"]:
            v["verdict"] = "pass"
            return v
        if r.get("no_key"):
            v["no_key"] = True
        detail = r.get("error") or f"HTTP {r.get('http', 'unknown')}"
        v.update(verdict="unjudged", why=f"single smoke attempt did not pass; not retried: {detail}")
        return v
    if past_deadline:
        if v.get("no_key"):
            v.update(verdict="unjudged", why="converged, but no API key for the smoke call")
        else:
            v.update(verdict="fail", why=("not back online" if not fresh else
                                          f"not converged to {target}" if not converged else
                                          "smoke never passed") + f" within {ROLLOUT_WINDOW_S}s")
        return v
    v["verdict"] = "pending"
    return v


def rollout_tick(conn, central: str = DEFAULT_CENTRAL, fleet=None,
                 index_dir=DEFAULT_INDEX_DIR) -> dict:
    """Judge the open rollout once (idempotent; the watcher calls it periodically).
    Returns {} when nothing is open, else the rollout's state after this tick."""
    fleet = fleet or LiveFleet(central)
    with conn.cursor() as cur:
        ensure_promotions(cur)
        conn.commit()
        row = open_rollout(cur)
        if row is None:
            return {}
        cur.execute("SELECT extract(epoch FROM now() - %s::timestamptz), now() > %s::timestamptz",
                    (row["pinned_at"], row["deadline"]))
        age, past = cur.fetchone()
    age = float(age)
    pid, target = row["id"], row["version"]
    verdict = dict(row["verdict"] or {})
    health = fleet.health()
    central_ok = health["up"] and health.get("required") == target
    verdict["_central"] = health | {"ok": central_ok}
    failures = []
    if not central_ok and age > CENTRAL_GRACE_S:
        failures.append(f"central not serving {target} {int(age)}s after the pin ({health})")
    if central_ok:
        live = {w["name"]: w for w in fleet.workers() or []}
        drift = (_drift_verdicts(conn, central, row["config"])
                 if row["mode"] == "reuse-tag" and row["config"] else {})
        now = time.time()
        for name, base in (row["baseline"] or {}).items():
            v = judge_worker(fleet, name, target, base, live.get(name), verdict.get(name, {}),
                             drift.get(name), now, bool(past))
            verdict[name] = v
            if v["verdict"] == "fail":
                failures.append(f"{name}: {v.get('why')}")
    workers = {k: v for k, v in verdict.items() if not k.startswith("_")}
    states = [v["verdict"] for v in workers.values()]
    summary = " ".join(f"{k}={v['verdict']}" for k, v in sorted(workers.items())) or "no workers"
    if failures:
        # Operator: "we dont need auto rollbacks, we need reasons why it failed ... we are
        # trying to push the new to working". The pin stays; the evidence is the product.
        status = "unhealthy"
        verdict["_failures"] = failures
    elif "pending" in states or not central_ok:
        status = "rolling_out"
    elif "pass" in states and "unjudged" not in states:
        status = "healthy"
    else:
        status = "unjudged"                             # nobody online, or no smoke key
    with conn.cursor() as cur:
        cur.execute(sql.SQL("UPDATE {s}.promotions SET verdict = %s WHERE id = %s")
                    .format(s=sql.Identifier(SCHEMA)), (Jsonb(verdict), pid))
        if status in ("healthy", "unjudged", "unhealthy"):
            cur.execute(sql.SQL("UPDATE {s}.promotions SET status = %s, judged_at = now() "
                                "WHERE id = %s").format(s=sql.Identifier(SCHEMA)), (status, pid))
    conn.commit()
    log(f"rollout {pid} {target}: {status}  [{summary}]"
        + (f"  failures: {'; '.join(failures)}" if failures else ""))
    return {"promotion": pid, "version": target, "status": status, "failures": failures,
            "verdict": verdict}


def rollback(conn, row: dict, failures: list[str], fleet, index_dir: Path) -> dict:
    """Re-pin the previous healthy version; mark ``row`` rolled_back + fleet_bad."""
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT version FROM {s}.promotions WHERE status = 'healthy' "
                            "AND version <> %s ORDER BY judged_at DESC, id DESC LIMIT 1")
                    .format(s=sql.Identifier(SCHEMA)), (row["version"],))
        r = cur.fetchone()
    target = r[0] if r else row["previous_version"]
    if not target or target == row["version"] or not index_has(index_dir, target):
        with conn.cursor() as cur:
            cur.execute(sql.SQL("UPDATE {s}.promotions SET status = 'rollback_failed', fleet_bad = "
                                "true, judged_at = now(), note = %s WHERE id = %s")
                        .format(s=sql.Identifier(SCHEMA)),
                        (f"no rollback target on the index (previous {target!r}); "
                         f"failures: {'; '.join(failures)}", row["id"]))
        conn.commit()
        log(f"rollout {row['id']}: ROLLBACK IMPOSSIBLE (no healthy version on the index) — operator")
        return {"rolled_back_to": None}
    with conn.cursor() as cur:
        cur.execute(sql.SQL("UPDATE {s}.promotions SET status = 'rolled_back', fleet_bad = true, "
                            "rolled_back_to = %s, judged_at = now(), note = %s WHERE id = %s")
                    .format(s=sql.Identifier(SCHEMA)),
                    (target, "; ".join(failures), row["id"]))
        cur.execute(sql.SQL("""
            INSERT INTO {s}.promotions (kind, config, version, mode, members, wheels, index_dir,
                                        previous_version, rollback_of, by, published_at)
            SELECT 'rollback', p.config, p.version, p.mode, p.members, p.wheels, p.index_dir,
                   %s, %s, 'rollout_tick', p.published_at
            FROM {s}.promotions p WHERE p.version = %s ORDER BY p.id DESC LIMIT 1
            RETURNING id""").format(s=sql.Identifier(SCHEMA)),
            (row["version"], row["id"], target))
        got = cur.fetchone()
        if got is None:                                 # target was never promoted (adopt script)
            cur.execute(sql.SQL("""
                INSERT INTO {s}.promotions (kind, version, mode, members, wheels, index_dir,
                                            previous_version, rollback_of, by)
                VALUES ('rollback', %s, 'reuse-tag', '{{}}', %s, %s, %s, %s, 'rollout_tick')
                RETURNING id""").format(s=sql.Identifier(SCHEMA)),
                (target, Jsonb(index_wheels(index_dir, target)), str(index_dir), row["version"],
                 row["id"]))
            got = cur.fetchone()
    conn.commit()
    log(f"rollout {row['id']}: ROLLED BACK {row['version']} -> {target} (rollback row {got[0]})")
    res = start_rollout(conn, got[0], fleet, index_dir)
    return {"rolled_back_to": target, "rollback_promotion": got[0], **res}


def api_promotions(conn, limit: int = 20) -> list[dict]:
    with conn.cursor() as cur:
        ensure_promotions(cur)
        conn.commit()
        cur.execute(sql.SQL("SELECT id FROM {s}.promotions ORDER BY id DESC LIMIT %s")
                    .format(s=sql.Identifier(SCHEMA)), (limit,))
        ids = [r[0] for r in cur.fetchall()]
        return [{k: v for k, v in _row(cur, i).items() if k not in ("members", "baseline", "pin_log")}
                for i in ids]


def main(argv=None) -> int:
    import psycopg
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dsn", default=S.DEFAULT_DSN)
    ap.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    ap.add_argument("--central", default=DEFAULT_CENTRAL)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan").add_argument("config")
    x = sub.add_parser("apply"); x.add_argument("config"); x.add_argument("--pin", action="store_true")
    x.add_argument("--now", action="store_true",
                   help="restart central even while a benchmark/admission job runs (skip the quiet wait)")
    sub.add_parser("tick")
    sub.add_parser("ls")
    a = ap.parse_args(argv)
    with psycopg.connect(a.dsn) as conn:
        if a.cmd == "plan":
            return S.print_json(api_promote(conn, a.config, index_dir=a.index_dir, central=a.central))
        if a.cmd == "apply":
            return S.print_json(api_promote(conn, a.config, apply=True, pin=a.pin,
                                            index_dir=a.index_dir, central=a.central, now=a.now))
        if a.cmd == "tick":
            return S.print_json(rollout_tick(conn, a.central, index_dir=a.index_dir))
        return S.print_json(api_promotions(conn))


if __name__ == "__main__":
    sys.exit(main())
