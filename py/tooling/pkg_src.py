#!/usr/bin/env python3
"""pkg_src — mirror every package under py/ into Postgres, one table per package.

Files on disk are the source of truth; the tables are a complete mirror. Each
package directory (py/<group>/<package>/ holding a pyproject.toml) gets a table
``pkg_src.<package>`` with ONE ROW PER FILE — .py, pyproject.toml, README,
templates, JSON, binary assets — so ``build`` can recreate the whole package
from the table alone. ``pkg_src.packages`` catalogs them; ``pkg_src.sync_runs``
records every sync.

The file set is git's view of the package dir (tracked + untracked-not-ignored),
so build/, dist/, *.egg-info and __pycache__ never land in a table.

    pkg_src.py status                     disk vs table, per package
    pkg_src.py sync [PKG ...] [--dry-run] upsert changed files, drop removed ones
    pkg_src.py build PKG --out DIR        recreate the package from its table
    pkg_src.py verify [PKG ...]           build to a temp dir, compare to disk
    pkg_src.py ls PKG                     one line per row
    pkg_src.py cat PKG REL_PATH           print one file from the table
    pkg_src.py watch [--interval S]       poll the tree, sync packages as they change

Versions. Every distinct tree a sync sees is kept as an immutable version of the
package (``pkg_src.versions`` + ``version_files``, contents deduped by sha256 in
``pkg_src.blobs``). The identity is a sha over (path, content, exec bit) only, so
a git tag and the same checkout on disk (664 vs 644) are the same version. The
label is the RELEASED (PyPI) version with an internal micro number as a 4th
segment: the release tree itself is ``0.2.1`` and each later change counts up
under it — ``0.2.1.1``, ``0.2.1.2`` … — resetting to ``.1`` when the next PyPI
release advances the base to ``0.2.2`` (see ``next_micro``). A 3-part version is
a release workers run; a 4-part version is an internal, never-published revision. A configuration pins one version per package (``pkg_src.configs``
+ ``config_members``, by version id); versions and configurations can be marked
known-good and restored.

    pkg_src.py versions [PKG] [--good]            history, newest first
    pkg_src.py good PKG REF [--unset] [--note N]  mark a package version known-good
    pkg_src.py build PKG --version REF --out DIR  recreate a past version
    pkg_src.py restore PKG REF [--apply]          put a version back into the tree
    pkg_src.py config save NAME [--good] [--note N]    pin the current tree
    pkg_src.py config import-tag TAG [--good]     pin a git tag (release baseline)
    pkg_src.py config ls | show NAME | diff A B
    pkg_src.py config good NAME [--unset]
    pkg_src.py config build NAME --out DIR        materialise every package
    pkg_src.py config restore NAME [--apply]      revert the tree to a configuration

Verification (the test env: immutable, hashed venvs under $PKG_SRC_TESTENV/envs; with
$PKG_SRC_SANDBOX every command runs as user pkgtest in a throwaway systemd sandbox):

    Every verify of a member set containing hugpy_fleet also runs the WORKER PROBE: the
    real worker booted for 30 s with an import/spawn audit hook in every python process
    (result.worker_probe, table worker_probe). Fails on a retired package imported or
    spawned, or a workspace ImportError; reports static-vs-runtime package sets.

    pkg_src.py retire PKG [--undo] [--note N]     deprecated package: history kept, never
                                                  tested / known-good / promoted
    pkg_src.py testenv build                      build a fresh env from scratch, switch to it
    pkg_src.py testenv ls | path
    pkg_src.py job verify PKG@REF … | --config N  queue: install those versions, run CI checks
    pkg_src.py job restore PKG@REF … | --config N queue: put versions back into the tree
    pkg_src.py job ls | job show ID

The watcher runs queued jobs one at a time. ``good`` / ``config good`` refuse
unless a verify job passed on exactly that version / configuration, in a test env
that matched its recorded fingerprint.

Code intelligence (pkg_graph.py: every .py blob parsed once into py_symbols /
py_imports / py_calls; graphs resolved per member set):

    pkg_src.py analyze [--config N | --job ID]    duplicates, collisions, missing/undeclared
                                                  deps, cycles, unreferenced, test coverage
    pkg_src.py dups [--callers]                   duplicate-function groups (+ who calls which copy)
    pkg_src.py trace MODULE:QUALNAME [--job ID]   tests that ran it + static callers/callees
    pkg_src.py trace --test NODEID [--job ID]     every workspace function that test executed
    pkg_src.py symbols NAME_LIKE [--kind K]       find functions/classes/variables

REF = current | good (latest known-good) | version id | label | hash prefix.
Restores are dry runs unless --apply; --apply first saves the tree as config
``pre-restore-<utc>`` so every revert is itself revertable.

DB: --dsn / $PKG_SRC_DSN, default "dbname=hugpy" (peer auth on ae).
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pkg_graph  # noqa: E402  (sibling module: symbol index + dependency graph)

SCHEMA = "pkg_src"
PY_ROOT = Path(__file__).resolve().parent.parent          # .../hugpy/py
DEFAULT_DSN = os.environ.get("PKG_SRC_DSN", "dbname=hugpy")


# --------------------------------------------------------------------------- disk

def discover(py_root: Path) -> dict[str, Path]:
    """{package: dir} for every py/<group>/<package>/pyproject.toml."""
    out: dict[str, Path] = {}
    for pp in sorted(py_root.glob("*/*/pyproject.toml")):
        name = pp.parent.name
        if name in out:
            raise SystemExit(f"duplicate package dir name {name!r}: {out[name]} and {pp.parent}")
        out[name] = pp.parent
    return out


def _git(repo: Path, *args: str) -> str:
    # The repo is owned by `hugpy`; allow it for this call only (no config change).
    return subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(repo), *args],
        check=True, capture_output=True, text=True).stdout


@functools.lru_cache(maxsize=None)
def git_root(path: Path) -> Path:
    return Path(_git(path, "rev-parse", "--show-toplevel").strip())


def package_files(pkg_dir: Path) -> list[str]:
    """Package-relative paths git considers part of the package, that exist."""
    repo = git_root(pkg_dir)
    rel = pkg_dir.relative_to(repo)
    listed = _git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", str(rel))
    out = set()
    for p in filter(None, listed.split("\0")):
        full = repo / p
        if full.is_file() or full.is_symlink():
            out.add(str(Path(p).relative_to(rel)))
    return sorted(out)


def read_file(path: Path) -> dict:
    """One row's worth of facts about a file."""
    st = path.lstat()
    mode = stat.S_IMODE(st.st_mode)
    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
    if stat.S_ISLNK(st.st_mode):
        target = os.readlink(path)
        return dict(kind="symlink", is_binary=False, content_text=None, content_bytes=None,
                    link_target=target, sha256=hashlib.sha256(b"link:" + target.encode()).hexdigest(),
                    size_bytes=len(target), file_mode=mode, file_mtime=mtime)
    return facts_from_bytes(path.name, path.read_bytes(), mode, mtime)


def facts_from_bytes(name: str, data: bytes, mode: int, mtime=None) -> dict:
    text = None
    if b"\0" not in data:                     # Postgres text can't hold NUL
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            pass
    suffix = Path(name).suffix.lower().lstrip(".")
    return dict(kind=suffix or name.lower(), is_binary=text is None,
                content_text=text, content_bytes=None if text is not None else data,
                link_target=None, sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data), file_mode=mode, file_mtime=mtime)


def tree_sha(rows) -> str:
    """Order-independent digest of (path, mode, content) for a whole package."""
    h = hashlib.sha256()
    for rel, mode, sha in sorted(rows):
        h.update(f"{rel}\0{mode:o}\0{sha}\n".encode())
    return h.hexdigest()


def is_exec(mode: int) -> bool:
    return bool(mode & 0o111)


def version_sha(facts: dict[str, dict]) -> str:
    """Identity of a package version: path + content + exec bit (umask-independent)."""
    h = hashlib.sha256()
    for rel in sorted(facts):
        f = facts[rel]
        h.update(f"{rel}\0{'x' if is_exec(f['file_mode']) else '-'}\0{f['sha256']}\n".encode())
    return h.hexdigest()


def pyproject_version(pkg_dir: Path) -> str | None:
    m = re.search(r'^\s*version\s*=\s*"([^"]+)"', (pkg_dir / "pyproject.toml").read_text(), re.M)
    return m.group(1) if m else None


def tag_version(repo: Path, ref: str = "HEAD") -> str | None:
    """Nearest release tag at/below ``ref`` (setuptools-scm lockstep version), sans 'v'."""
    try:
        return _git(repo, "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*", ref).strip()[1:]
    except subprocess.CalledProcessError:
        return None


def base_version(pkg_dir: Path) -> str | None:
    """Static pyproject version if the package pins one, else the workspace tag."""
    return pyproject_version(pkg_dir) or tag_version(git_root(pkg_dir))


# ----------------------------------------------------------------------------- db

def ensure_schema(cur) -> None:
    cur.execute(sql.SQL("""
        CREATE SCHEMA IF NOT EXISTS {s};
        CREATE TABLE IF NOT EXISTS {s}.packages (
            package      text PRIMARY KEY,
            group_dir    text NOT NULL,
            repo_path    text NOT NULL,
            version      text,
            file_count   integer NOT NULL,
            total_bytes  bigint NOT NULL,
            tree_sha256  text NOT NULL,
            git_head     text,
            synced_at    timestamptz NOT NULL DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS {s}.sync_runs (
            id           bigserial PRIMARY KEY,
            package      text NOT NULL,
            added        integer NOT NULL,
            changed      integer NOT NULL,
            removed      integer NOT NULL,
            tree_sha256  text NOT NULL,
            git_head     text,
            ran_at       timestamptz NOT NULL DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS {s}.blobs (
            sha256         text PRIMARY KEY,
            is_binary      boolean NOT NULL,
            content_text   text,
            content_bytes  bytea,
            link_target    text,
            size_bytes     bigint NOT NULL
        );
        CREATE TABLE IF NOT EXISTS {s}.versions (
            id           bigserial PRIMARY KEY,
            package      text NOT NULL,
            version_sha  text NOT NULL,
            label        text NOT NULL,
            pkg_version  text,
            file_count   integer NOT NULL,
            total_bytes  bigint NOT NULL,
            git_head     text,
            source       text NOT NULL,
            created_at   timestamptz NOT NULL DEFAULT now(),
            known_good   boolean NOT NULL DEFAULT false,
            good_note    text,
            good_at      timestamptz,
            UNIQUE (package, version_sha)
        );
        CREATE TABLE IF NOT EXISTS {s}.version_files (
            version_id   bigint NOT NULL REFERENCES {s}.versions(id) ON DELETE CASCADE,
            rel_path     text NOT NULL,
            sha256       text NOT NULL REFERENCES {s}.blobs(sha256),
            file_mode    integer NOT NULL,
            PRIMARY KEY (version_id, rel_path)
        );
        CREATE TABLE IF NOT EXISTS {s}.configs (
            id           bigserial PRIMARY KEY,
            name         text NOT NULL UNIQUE,
            source       text NOT NULL,
            git_head     text,
            note         text,
            created_at   timestamptz NOT NULL DEFAULT now(),
            known_good   boolean NOT NULL DEFAULT false,
            good_at      timestamptz
        );
        CREATE TABLE IF NOT EXISTS {s}.config_members (
            config_id    bigint NOT NULL REFERENCES {s}.configs(id) ON DELETE CASCADE,
            package      text NOT NULL,
            version_id   bigint NOT NULL REFERENCES {s}.versions(id),
            PRIMARY KEY (config_id, package)
        );
        ALTER TABLE {s}.packages ADD COLUMN IF NOT EXISTS version_id bigint
            REFERENCES {s}.versions(id);
        ALTER TABLE {s}.versions ADD COLUMN IF NOT EXISTS micro integer;
        CREATE TABLE IF NOT EXISTS {s}.jobs (
            id           bigserial PRIMARY KEY,
            kind         text NOT NULL CHECK (kind IN ('verify', 'restore')),
            args         jsonb NOT NULL,
            status       text NOT NULL DEFAULT 'queued'
                         CHECK (status IN ('queued', 'running', 'pass', 'fail', 'error', 'done')),
            requested_by text,
            created_at   timestamptz NOT NULL DEFAULT now(),
            started_at   timestamptz,
            finished_at  timestamptz,
            result       jsonb,
            log          text
        );
        CREATE TABLE IF NOT EXISTS {s}.test_results (
            job_id       bigint NOT NULL REFERENCES {s}.jobs(id) ON DELETE CASCADE,
            package      text NOT NULL,
            version_id   bigint NOT NULL REFERENCES {s}.versions(id),
            step         text NOT NULL,
            ok           boolean NOT NULL,
            seconds      real NOT NULL,
            output_tail  text,
            PRIMARY KEY (job_id, package, step)
        );
        CREATE TABLE IF NOT EXISTS {s}.retired (
            package      text PRIMARY KEY,
            retired_at   timestamptz NOT NULL DEFAULT now(),
            note         text
        );
        CREATE TABLE IF NOT EXISTS {s}.worker_probe (
            job_id       bigint NOT NULL REFERENCES {s}.jobs(id) ON DELETE CASCADE,
            pid          integer NOT NULL,
            process      text NOT NULL,
            module       text NOT NULL,
            dist         text,
            package      text,
            PRIMARY KEY (job_id, pid, module)
        );
        CREATE TABLE IF NOT EXISTS {s}.testenvs (
            id           bigserial PRIMARY KEY,
            path         text NOT NULL UNIQUE,
            sha          text NOT NULL,
            python       text NOT NULL,
            dists        text NOT NULL,
            built_for    jsonb,
            reason       text,
            built_at     timestamptz NOT NULL DEFAULT now()
        );
    """).format(s=sql.Identifier(SCHEMA)))


def record_version(cur, package: str, facts: dict[str, dict], pkg_version: str | None,
                   git_head: str | None, source: str) -> tuple[int, str, bool]:
    """Store ``facts`` ({rel: read_file()-shaped dict}) as a version of ``package``.
    Returns (id, label, created). An identical tree reuses the existing version."""
    vsha = version_sha(facts)
    cur.execute(sql.SQL("SELECT id, label FROM {s}.versions WHERE package = %s AND version_sha = %s")
                .format(s=sql.Identifier(SCHEMA)), (package, vsha))
    row = cur.fetchone()
    if row:
        return row[0], row[1], False
    cur.executemany(sql.SQL(
        "INSERT INTO {s}.blobs (sha256, is_binary, content_text, content_bytes, link_target, "
        "size_bytes) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (sha256) DO NOTHING")
        .format(s=sql.Identifier(SCHEMA)),
        [(f["sha256"], f["is_binary"], f["content_text"], f["content_bytes"], f["link_target"],
          f["size_bytes"]) for f in {f["sha256"]: f for f in facts.values()}.values()])
    micro, label = next_micro(cur, package, pkg_version, source)
    cur.execute(sql.SQL(
        "INSERT INTO {s}.versions (package, version_sha, label, micro, pkg_version, file_count, "
        "total_bytes, git_head, source) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id")
        .format(s=sql.Identifier(SCHEMA)),
        (package, vsha, label, micro, pkg_version, len(facts),
         sum(f["size_bytes"] for f in facts.values()), git_head, source))
    vid = cur.fetchone()[0]
    cur.executemany(sql.SQL("INSERT INTO {s}.version_files (version_id, rel_path, sha256, file_mode) "
                            "VALUES (%s,%s,%s,%s)").format(s=sql.Identifier(SCHEMA)),
                    [(vid, rel, f["sha256"], f["file_mode"]) for rel, f in facts.items()])
    return vid, label, True


def next_micro(cur, package: str, base: str | None, source: str) -> tuple[int, str]:
    """Micro number + label for a new version of ``package`` on release ``base``.

    ``base`` is the released (PyPI) version — PyPI dictates it, and every source
    change after that release is an internal micro version counted UNDER it as a
    4th release segment: the tree of a release tag itself is micro 0, labelled
    exactly ``base`` (``0.2.1``); every other tree counts up from 1: ``0.2.1.1``,
    ``0.2.1.2`` … The counter is scoped to ``(package, base)``, so it resets to
    ``.1`` the moment a new release advances ``base`` to ``0.2.2``. This label is
    already its PEP 440 normal form (no leading zeros), so the DB string equals
    what a build/``packaging`` comparison produces — nothing to reconcile. A
    4-segment version is ALWAYS an internal, unpublished revision; workers only
    ever run the 3-part release, so a worker on ``0.2.1.N`` is by definition
    drift. Rows recorded before this scheme keep their old ``0.2.1+m<N>`` labels;
    nothing references a label (configs pin ``version_id``), so the two coexist."""
    base = base or "0"
    cur.execute(sql.SQL("SELECT coalesce(max(micro), 0), bool_or(micro = 0) FROM {s}.versions "
                        "WHERE package = %s AND pkg_version IS NOT DISTINCT FROM %s")
                .format(s=sql.Identifier(SCHEMA)), (package, None if base == "0" else base))
    top, has_zero = cur.fetchone()
    if source.startswith("tag:") and not has_zero:
        return 0, base
    return top + 1, f"{base}.{top + 1}"


def relabel_micro(conn) -> None:
    """One-off: give versions recorded before micro numbering their micro label."""
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT id, package, pkg_version, source FROM {s}.versions "
                            "WHERE micro IS NULL ORDER BY id").format(s=sql.Identifier(SCHEMA)))
        for vid, pkg, base, source in cur.fetchall():
            micro, label = next_micro(cur, pkg, base, source)
            cur.execute(sql.SQL("UPDATE {s}.versions SET micro = %s, label = %s WHERE id = %s")
                        .format(s=sql.Identifier(SCHEMA)), (micro, label, vid))
    conn.commit()


def backfill_versions(conn) -> None:
    """First run after versioning landed: the mirror tables already hold a synced
    tree per package — keep it as that package's first version."""
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT package, version, git_head FROM {s}.packages "
                            "WHERE version_id IS NULL").format(s=sql.Identifier(SCHEMA)))
        for name, pkg_version, head in cur.fetchall():
            if not table_exists(cur, name):
                continue
            cur.execute(sql.SQL("SELECT {c} FROM {t}").format(
                c=sql.SQL(", ").join(map(sql.Identifier, COLS)), t=sql.Identifier(SCHEMA, name)))
            facts = {r[0]: dict(zip(COLS[1:], r[1:])) for r in cur.fetchall()}
            for f in facts.values():
                if f["content_bytes"] is not None:
                    f["content_bytes"] = bytes(f["content_bytes"])
            repo = git_root(PY_ROOT)
            vid, _, _ = record_version(cur, name, facts,
                                       pkg_version or (tag_version(repo, head) if head else None),
                                       head, "backfill")
            cur.execute(sql.SQL("UPDATE {s}.versions v SET created_at = p.synced_at FROM "
                                "{s}.packages p WHERE v.id = %s AND p.package = %s")
                        .format(s=sql.Identifier(SCHEMA)), (vid, name))
            cur.execute(sql.SQL("UPDATE {s}.packages SET version_id = %s WHERE package = %s")
                        .format(s=sql.Identifier(SCHEMA)), (vid, name))
    conn.commit()


def resolve_version(cur, package: str, ref: str) -> int:
    s = sql.Identifier(SCHEMA)
    if ref == "current":
        cur.execute(sql.SQL("SELECT version_id FROM {s}.packages WHERE package = %s").format(s=s),
                    (package,))
    elif ref == "good":
        cur.execute(sql.SQL("SELECT id FROM {s}.versions WHERE package = %s AND known_good "
                            "ORDER BY good_at DESC, id DESC LIMIT 1").format(s=s), (package,))
    elif ref.isdigit():
        cur.execute(sql.SQL("SELECT id FROM {s}.versions WHERE package = %s AND id = %s")
                    .format(s=s), (package, int(ref)))
    else:
        h = ref.split("+t", 1)[1] if "+t" in ref else ref
        cur.execute(sql.SQL("SELECT id FROM {s}.versions WHERE package = %s AND "
                            "(label = %s OR version_sha LIKE %s)").format(s=s),
                    (package, ref, h + "%"))
    rows = cur.fetchall()
    if len(rows) != 1 or rows[0][0] is None:
        raise SystemExit(f"{package}: version ref {ref!r} matched {len(rows)} versions")
    return rows[0][0]


def version_files(cur, vid: int) -> dict[str, tuple]:
    """{rel: (sha256, mode, is_binary, text, bytes, link)} for version ``vid``."""
    cur.execute(sql.SQL("SELECT f.rel_path, f.sha256, f.file_mode, b.is_binary, b.content_text, "
                        "b.content_bytes, b.link_target FROM {s}.version_files f "
                        "JOIN {s}.blobs b USING (sha256) WHERE f.version_id = %s")
                .format(s=sql.Identifier(SCHEMA)), (vid,))
    return {r[0]: r[1:] for r in cur.fetchall()}


def write_entry(dest: Path, sha, mode, is_bin, text, data, link) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    if link is not None:
        os.symlink(link, dest)
    else:
        dest.write_bytes(bytes(data) if is_bin else text.encode("utf-8"))
        os.chmod(dest, mode)


def ensure_table(cur, package: str) -> None:
    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {t} (
            rel_path       text PRIMARY KEY,
            kind           text NOT NULL,
            is_binary      boolean NOT NULL,
            content_text   text,
            content_bytes  bytea,
            link_target    text,
            sha256         text NOT NULL,
            size_bytes     bigint NOT NULL,
            file_mode      integer NOT NULL,
            file_mtime     timestamptz,
            synced_at      timestamptz NOT NULL DEFAULT now(),
            CHECK (link_target IS NOT NULL
                   OR (is_binary AND content_bytes IS NOT NULL)
                   OR (NOT is_binary AND content_text IS NOT NULL))
        )
    """).format(t=sql.Identifier(SCHEMA, package)))


def table_exists(cur, package: str) -> bool:
    cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f'{SCHEMA}."{package}"',))
    return cur.fetchone()[0]


def table_index(cur, package: str) -> dict[str, tuple[str, int]]:
    """{rel_path: (sha256, file_mode)} currently in the table."""
    if not table_exists(cur, package):
        return {}
    cur.execute(sql.SQL("SELECT rel_path, sha256, file_mode FROM {t}")
                .format(t=sql.Identifier(SCHEMA, package)))
    return {r: (s, m) for r, s, m in cur.fetchall()}


COLS = ("rel_path", "kind", "is_binary", "content_text", "content_bytes", "link_target",
        "sha256", "size_bytes", "file_mode", "file_mtime")


# ----------------------------------------------------------------------- commands

def cmd_sync(conn, pkgs: dict[str, Path], names: list[str], dry_run: bool,
             source: str = "sync") -> int:
    with conn.cursor() as cur:
        ensure_schema(cur)
        for name in names:
            pkg_dir = pkgs[name]
            files = package_files(pkg_dir)
            disk = {rel: read_file(pkg_dir / rel) for rel in files}
            have = table_index(cur, name)
            added = [r for r in disk if r not in have]
            changed = [r for r in disk if r in have
                       and have[r] != (disk[r]["sha256"], disk[r]["file_mode"])]
            removed = [r for r in have if r not in disk]
            tsha = tree_sha((r, d["file_mode"], d["sha256"]) for r, d in disk.items())
            head = _git(git_root(pkg_dir), "rev-parse", "HEAD").strip()
            print(f"{name:20} +{len(added):<4} ~{len(changed):<4} -{len(removed):<4} "
                  f"{len(disk)} files  {tsha[:12]}")
            for tag, lst in (("+", added), ("~", changed), ("-", removed)):
                for r in lst if (dry_run or len(lst) <= 20) and have else []:
                    print(f"    {tag} {r}")
            if dry_run or (not (added or changed or removed) and (have or not disk)):
                continue
            ensure_table(cur, name)
            t = sql.Identifier(SCHEMA, name)
            upsert = sql.SQL(
                "INSERT INTO {t} ({cols}) VALUES ({vals}) ON CONFLICT (rel_path) DO UPDATE SET "
                "{sets}, synced_at = now()").format(
                t=t, cols=sql.SQL(", ").join(map(sql.Identifier, COLS)),
                vals=sql.SQL(", ").join(sql.Placeholder() * len(COLS)),
                sets=sql.SQL(", ").join(sql.SQL("{c} = EXCLUDED.{c}").format(c=sql.Identifier(c))
                                        for c in COLS[1:]))
            cur.executemany(upsert, [tuple([r] + [disk[r][c] for c in COLS[1:]])
                                     for r in added + changed])
            if removed:
                cur.execute(sql.SQL("DELETE FROM {t} WHERE rel_path = ANY(%s)").format(t=t),
                            (removed,))
            pyver = base_version(pkg_dir)
            vid, label, created = record_version(cur, name, disk, pyver, head, source)
            print(f"    version {label} ({'new' if created else 'seen before'}, id {vid})")
            cur.execute(sql.SQL("""
                INSERT INTO {s}.packages (package, group_dir, repo_path, version, file_count,
                                          total_bytes, tree_sha256, git_head, synced_at,
                                          version_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now(), %s)
                ON CONFLICT (package) DO UPDATE SET group_dir = EXCLUDED.group_dir,
                    repo_path = EXCLUDED.repo_path, version = EXCLUDED.version,
                    file_count = EXCLUDED.file_count, total_bytes = EXCLUDED.total_bytes,
                    tree_sha256 = EXCLUDED.tree_sha256, git_head = EXCLUDED.git_head,
                    synced_at = now(), version_id = EXCLUDED.version_id
            """).format(s=sql.Identifier(SCHEMA)),
                (name, pkg_dir.parent.name, str(pkg_dir.relative_to(git_root(pkg_dir))),
                 pyver, len(disk), sum(d["size_bytes"] for d in disk.values()), tsha, head, vid))
            cur.execute(sql.SQL("INSERT INTO {s}.sync_runs (package, added, changed, removed, "
                                "tree_sha256, git_head) VALUES (%s,%s,%s,%s,%s,%s)")
                        .format(s=sql.Identifier(SCHEMA)),
                        (name, len(added), len(changed), len(removed), tsha, head))
            conn.commit()
    if not dry_run:
        pkg_graph.index_blobs(conn)
    return 0


def build(conn, name: str, out: Path, ref: str | None = None) -> int:
    """Write every row of pkg_src.<name> (or of version ``ref``) under ``out``.
    Returns the file count."""
    with conn.cursor() as cur:
        if ref is not None:
            files = version_files(cur, resolve_version(cur, name, ref))
            for rel, entry in files.items():
                write_entry(out / rel, *entry)
            return len(files)
        if not table_exists(cur, name):
            raise SystemExit(f"no table {SCHEMA}.{name} — run: sync {name}")
        cur.execute(sql.SQL("SELECT rel_path, is_binary, content_text, content_bytes, "
                            "link_target, file_mode FROM {t} ORDER BY rel_path")
                    .format(t=sql.Identifier(SCHEMA, name)))
        n = 0
        for rel, is_bin, text, data, link, mode in cur:
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if link is not None:
                os.symlink(link, dest)
            else:
                dest.write_bytes(bytes(data) if is_bin else text.encode("utf-8"))
                os.chmod(dest, mode)
            n += 1
    return n


def prepare_out(out: Path, force: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not force:
            raise SystemExit(f"{out} is not empty (use --force to replace it)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def cmd_build(conn, name: str, out: Path, force: bool, ref: str | None = None) -> int:
    prepare_out(out, force)
    print(f"{name}: wrote {build(conn, name, out, ref)} files to {out}")
    return 0


def cmd_verify(conn, pkgs: dict[str, Path], names: list[str]) -> int:
    bad = 0
    for name in names:
        pkg_dir = pkgs[name]
        disk = {r: read_file(pkg_dir / r) for r in package_files(pkg_dir)}
        with tempfile.TemporaryDirectory(prefix=f"pkg_src_{name}_") as tmp:
            try:
                build(conn, name, Path(tmp))
            except SystemExit as e:
                print(f"{name:20} MISSING  {e}"); bad += 1; continue
            rebuilt = {str(p.relative_to(tmp)): read_file(p)
                       for p in Path(tmp).rglob("*") if p.is_file() or p.is_symlink()}
        dsha = tree_sha((r, d["file_mode"], d["sha256"]) for r, d in disk.items())
        bsha = tree_sha((r, d["file_mode"], d["sha256"]) for r, d in rebuilt.items())
        if dsha == bsha:
            print(f"{name:20} OK       {len(rebuilt)} files rebuilt byte-identical  {bsha[:12]}")
            continue
        bad += 1
        only_disk = sorted(set(disk) - set(rebuilt))
        only_tbl = sorted(set(rebuilt) - set(disk))
        diff = sorted(r for r in set(disk) & set(rebuilt)
                      if (disk[r]["sha256"], disk[r]["file_mode"])
                      != (rebuilt[r]["sha256"], rebuilt[r]["file_mode"]))
        print(f"{name:20} DIFFERS  disk-only {len(only_disk)}, table-only {len(only_tbl)}, "
              f"content/mode {len(diff)}  (run: sync {name})")
        for tag, lst in (("disk-only", only_disk), ("table-only", only_tbl), ("differs", diff)):
            for r in lst[:10]:
                print(f"    {tag:10} {r}")
    return 1 if bad else 0


def cmd_status(conn, pkgs: dict[str, Path]) -> int:
    with conn.cursor() as cur:
        ensure_schema(cur)
        cur.execute(sql.SQL("SELECT package, file_count, tree_sha256, synced_at FROM {s}.packages")
                    .format(s=sql.Identifier(SCHEMA)))
        cat = {p: (n, s, t) for p, n, s, t in cur.fetchall()}
    print(f"{'package':20} {'group':13} {'files':>5}  {'state':9} last sync")
    for name, pkg_dir in pkgs.items():
        files = package_files(pkg_dir)
        dsha = tree_sha((r, d["file_mode"], d["sha256"])
                        for r, d in ((r, read_file(pkg_dir / r)) for r in files))
        if name not in cat:
            state, when = "not synced", "-"
        else:
            state = "in sync" if cat[name][1] == dsha else "stale"
            when = cat[name][2].astimezone().strftime("%Y-%m-%d %H:%M")
        print(f"{name:20} {pkg_dir.parent.name:13} {len(files):>5}  {state:9} {when}")
    return 0


def cmd_ls(conn, name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT rel_path, kind, is_binary, size_bytes, file_mode, sha256 "
                            "FROM {t} ORDER BY rel_path").format(t=sql.Identifier(SCHEMA, name)))
        for rel, kind, is_bin, size, mode, sha in cur:
            print(f"{mode:04o} {size:>9} {'bin' if is_bin else 'txt'} {kind:8} {sha[:10]}  {rel}")
    return 0


def cmd_cat(conn, name: str, rel: str) -> int:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT is_binary, content_text, content_bytes, link_target "
                            "FROM {t} WHERE rel_path = %s").format(t=sql.Identifier(SCHEMA, name)),
                    (rel,))
        row = cur.fetchone()
    if row is None:
        raise SystemExit(f"{rel} not in {SCHEMA}.{name}")
    is_bin, text, data, link = row
    if link is not None:
        print(f"-> {link}")
    elif is_bin:
        sys.stdout.buffer.write(bytes(data))
    else:
        sys.stdout.write(text)
    return 0


# ---------------------------------------------------------------------- versions

def cmd_versions(conn, names: list[str], good_only: bool) -> int:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            SELECT v.id, v.package, v.label, v.file_count, v.source, v.created_at, v.known_good,
                   v.good_note, v.id = p.version_id
            FROM {s}.versions v LEFT JOIN {s}.packages p USING (package)
            WHERE v.package = ANY(%s) AND (%s OR v.known_good)
            ORDER BY v.package, v.id DESC""").format(s=sql.Identifier(SCHEMA)),
            (names, not good_only))
        print(f"{'id':>5} {'package':20} {'label':24} {'files':>5}  {'created':16} flags")
        for vid, pkg, label, n, src, at, good, note, cur_ in cur.fetchall():
            flags = " ".join(f for f in ("CURRENT" if cur_ else "", "GOOD" if good else "", src) if f)
            print(f"{vid:>5} {pkg:20} {label:24} {n:>5}  "
                  f"{at.astimezone():%Y-%m-%d %H:%M} {flags}{'  — ' + note if note else ''}")
    return 0


def cmd_good(conn, name: str, ref: str, unset: bool, note: str | None) -> int:
    r = api_mark_good(conn, package=name, ref=ref, note=note, unset=unset)
    print(f"{name} {r['label']}: {'not ' if unset else ''}known-good"
          + (f" (tested by job {r['job']})" if r.get("job") else ""))
    return 0


def restore_plan(cur, pkg_dir: Path, vid: int):
    """(writes, deletes, target) to turn ``pkg_dir`` into version ``vid``."""
    target = version_files(cur, vid)
    disk = {r: read_file(pkg_dir / r) for r in package_files(pkg_dir)}
    writes = sorted(r for r, (sha, mode, *_) in target.items()
                    if r not in disk or disk[r]["sha256"] != sha
                    or is_exec(disk[r]["file_mode"]) != is_exec(mode))
    deletes = sorted(set(disk) - set(target))
    return writes, deletes, target


def restore_packages(conn, pkgs: dict[str, Path], plan: dict[str, int], apply: bool,
                     reason: str) -> int:
    """Put each {package: version_id} back into the tree (dry run unless ``apply``)."""
    todo = {}
    with conn.cursor() as cur:
        for name, vid in plan.items():
            w, d, target = restore_plan(cur, pkgs[name], vid)
            if w or d:
                todo[name] = (w, d, target)
            print(f"{name:20} {'unchanged' if not (w or d) else f'write {len(w)}, delete {len(d)}'}")
            for tag, lst in (("write", w), ("delete", d)):
                for r in lst:
                    print(f"    {tag:6} {r}")
    if not todo:
        print("tree already matches")
        return 0
    if not apply:
        print("dry run — re-run with --apply to change the tree")
        return 0
    snap = f"pre-restore-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    save_config(conn, pkgs, snap, good=False, note=f"automatic, before restoring {reason}")
    for name, (w, d, target) in todo.items():
        for r in w:
            write_entry(pkgs[name] / r, *target[r])
        for r in d:
            (pkgs[name] / r).unlink()
    cmd_sync(conn, pkgs, list(todo), False, source=f"restore:{reason}")
    print(f"restored {reason}; previous tree saved as config {snap}")
    return 0


# ------------------------------------------------------------------ configurations

def config_id(cur, name: str) -> int:
    cur.execute(sql.SQL("SELECT id FROM {s}.configs WHERE name = %s")
                .format(s=sql.Identifier(SCHEMA)), (name,))
    row = cur.fetchone()
    if row is None:
        raise SystemExit(f"no configuration {name!r} (see: config ls)")
    return row[0]


def config_members(cur, name: str) -> dict[str, int]:
    cur.execute(sql.SQL("SELECT package, version_id FROM {s}.config_members WHERE config_id = %s")
                .format(s=sql.Identifier(SCHEMA)), (config_id(cur, name),))
    return dict(cur.fetchall())


def insert_config(cur, name: str, members: dict[str, int], source: str, head: str | None,
                  good: bool, note: str | None) -> None:
    s = sql.Identifier(SCHEMA)
    cur.execute(sql.SQL("SELECT 1 FROM {s}.configs WHERE name = %s").format(s=s), (name,))
    if cur.fetchone():
        raise SystemExit(f"configuration {name!r} already exists")
    cur.execute(sql.SQL("INSERT INTO {s}.configs (name, source, git_head, note, known_good, "
                        "good_at) VALUES (%s,%s,%s,%s,%s, CASE WHEN %s THEN now() END) "
                        "RETURNING id").format(s=s), (name, source, head, note, good, good))
    cid = cur.fetchone()[0]
    cur.executemany(sql.SQL("INSERT INTO {s}.config_members (config_id, package, version_id) "
                            "VALUES (%s,%s,%s)").format(s=s),
                    [(cid, p, v) for p, v in sorted(members.items())])


def save_config(conn, pkgs: dict[str, Path], name: str, good: bool, note: str | None) -> int:
    cmd_sync(conn, pkgs, list(pkgs), False)          # pin what is on disk right now
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT package, version_id FROM {s}.packages WHERE package = ANY(%s)")
                    .format(s=sql.Identifier(SCHEMA)), (list(pkgs),))
        members = dict(cur.fetchall())
        repo = git_root(next(iter(pkgs.values())))
        head = _git(repo, "rev-parse", "HEAD").strip()
        dirty = len(_git(repo, "status", "--porcelain", "--", "py").splitlines())
        insert_config(cur, name, members, f"tree@{head[:10]}{f'+{dirty}dirty' if dirty else ''}",
                      head, good, note)
    conn.commit()
    print(f"config {name}: {len(members)} packages pinned{' (known-good)' if good else ''}")
    return 0


def git_package_facts(repo: Path, tag: str, rel: Path) -> dict[str, dict]:
    """read_file()-shaped facts for every file of ``rel`` at git ref ``tag``."""
    ls = _git(repo, "ls-tree", "-r", "-z", tag, "--", str(rel))
    entries = []
    for item in filter(None, ls.split("\0")):
        meta, path = item.split("\t", 1)
        mode, kind, obj = meta.split()
        if kind == "blob":
            entries.append((str(Path(path).relative_to(rel)), int(mode, 8), obj))
    if not entries:
        return {}
    out = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repo), "cat-file", "--batch"],
                         input="".join(f"{o}\n" for _, _, o in entries).encode(),
                         check=True, capture_output=True).stdout
    facts, pos = {}, 0
    for rel_path, mode, _ in entries:
        nl = out.index(b"\n", pos)
        size = int(out[pos:nl].split()[2])
        data = out[nl + 1:nl + 1 + size]
        pos = nl + 1 + size + 1
        if mode == 0o120000:
            target = data.decode()
            facts[rel_path] = dict(kind="symlink", is_binary=False, content_text=None,
                                   content_bytes=None, link_target=target,
                                   sha256=hashlib.sha256(b"link:" + target.encode()).hexdigest(),
                                   size_bytes=len(target), file_mode=0o777, file_mtime=None)
        else:
            facts[rel_path] = facts_from_bytes(rel_path, data, mode & 0o777)
    return facts


def cmd_import_tag(conn, pkgs: dict[str, Path], tag: str, name: str | None, good: bool,
                   note: str | None) -> int:
    repo = git_root(next(iter(pkgs.values())))
    commit = _git(repo, "rev-list", "-n1", tag).strip()
    members = {}
    with conn.cursor() as cur:
        for pkg, pkg_dir in pkgs.items():
            rel = pkg_dir.relative_to(repo)
            facts = git_package_facts(repo, tag, rel)
            if not facts:
                print(f"{pkg:20} absent at {tag}")
                continue
            pp = facts.get("pyproject.toml")
            m = pp and pp["content_text"] and re.search(
                r'^\s*version\s*=\s*"([^"]+)"', pp["content_text"], re.M)
            vid, label, created = record_version(cur, pkg, facts,
                                                 m.group(1) if m else tag_version(repo, commit),
                                                 commit, f"tag:{tag}")
            members[pkg] = vid
            print(f"{pkg:20} {label:24} {'new' if created else 'seen before'}")
        insert_config(cur, name or tag, members, f"tag:{tag}", commit, good, note)
    conn.commit()
    print(f"config {name or tag}: {len(members)} packages pinned{' (known-good)' if good else ''}")
    return 0


def cmd_config_ls(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            SELECT c.name, c.source, c.created_at, c.known_good, c.note, count(m.package)
            FROM {s}.configs c LEFT JOIN {s}.config_members m ON m.config_id = c.id
            GROUP BY c.id ORDER BY c.created_at DESC""").format(s=sql.Identifier(SCHEMA)))
        print(f"{'name':32} {'pkgs':>4}  {'created':16} {'good':4}  source / note")
        for name, src, at, good, note, n in cur.fetchall():
            print(f"{name:32} {n:>4}  {at.astimezone():%Y-%m-%d %H:%M} {'GOOD' if good else '    '}"
                  f"  {src}{'  — ' + note if note else ''}")
    return 0


def cmd_config_show(conn, name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            SELECT m.package, v.label, v.known_good, v.id = p.version_id
            FROM {s}.config_members m JOIN {s}.versions v ON v.id = m.version_id
            LEFT JOIN {s}.packages p ON p.package = m.package
            WHERE m.config_id = %s ORDER BY m.package""").format(s=sql.Identifier(SCHEMA)),
            (config_id(cur, name),))
        for pkg, label, good, same in cur.fetchall():
            print(f"{pkg:20} {label:24} {'GOOD' if good else '    '} "
                  f"{'= current' if same else '≠ current'}")
    return 0


def cmd_config_diff(conn, a: str, b: str) -> int:
    with conn.cursor() as cur:
        ma, mb = config_members(cur, a), config_members(cur, b)
        for pkg in sorted(set(ma) | set(mb)):
            va, vb = ma.get(pkg), mb.get(pkg)
            if va == vb:
                continue
            fa = version_files(cur, va) if va else {}
            fb = version_files(cur, vb) if vb else {}
            chg = [r for r in set(fa) & set(fb) if fa[r][:2] != fb[r][:2]]
            print(f"{pkg:20} +{len(set(fb) - set(fa))} ~{len(chg)} -{len(set(fa) - set(fb))}")
            for tag, lst in (("+", set(fb) - set(fa)), ("~", chg), ("-", set(fa) - set(fb))):
                for r in sorted(lst):
                    print(f"    {tag} {r}")
    return 0


def cmd_config_good(conn, name: str, unset: bool, note: str | None = None) -> int:
    r = api_mark_good(conn, config=name, note=note, unset=unset)
    print(f"config {name}: {'not ' if unset else ''}known-good"
          + (f" (tested by job {r['job']})" if r.get("job") else ""))
    return 0


def cmd_config_build(conn, pkgs: dict[str, Path], name: str, out: Path, force: bool) -> int:
    prepare_out(out, force)
    repo = git_root(next(iter(pkgs.values())))
    with conn.cursor() as cur:
        members = config_members(cur, name)
        for pkg, vid in sorted(members.items()):
            files = version_files(cur, vid)
            dest = out / (pkgs[pkg].relative_to(repo / "py") if pkg in pkgs else pkg)
            for rel, entry in files.items():
                write_entry(dest / rel, *entry)
            print(f"{pkg:20} {len(files)} files -> {dest}")
    return 0


# ---------------------------------------------------------------------- test env
#
# The test env is a series of immutable venvs under $PKG_SRC_TESTENV/envs/<utc>,
# each built from scratch and recorded in pkg_src.testenvs (python + third-party
# dists, hashed); envs/current is the one jobs use. Envs are never patched: when a
# member set needs other third-party deps, or an env no longer matches its recorded
# hash, a new env is built next to it and becomes current once it checks out.
#
# A verify job materialises the exact versions from the DB — never the working
# tree — and runs what CI runs per package: wheel build, pytest --timeout=120,
# import. With $PKG_SRC_SANDBOX (the root-owned pkg-test-run launcher) every test-
# env command runs as user pkgtest in a throwaway systemd sandbox: private /tmp,
# read-only system, no /home, a per-job HOME. Known-good requires a passing job
# whose env matched its recorded hash.

TESTENV = Path(os.environ.get("PKG_SRC_TESTENV",
                              Path.home() / ".local/share/pkg_src/testenv"))
TESTENV_PYTHON = os.environ.get("PKG_SRC_TESTENV_PYTHON", "/usr/bin/python3.12")
SANDBOX = os.environ.get("PKG_SRC_SANDBOX", "")      # e.g. /usr/local/libexec/pkg-test-run
TEST_TOOLS = ["pip", "build", "wheel", "setuptools", "setuptools-scm", "pytest", "pytest-timeout",
              "pytest-cov", "coverage"]
PYTEST_BUDGET = 1800            # whole-suite wall clock per package (per-test: 120 s, as CI)
FINGERPRINT = ("import importlib.metadata as m, json, sys\n"
               "json.dump({'python': sys.version.split()[0], 'dists': sorted({"
               "d.metadata['Name'] + '==' + d.version for d in m.distributions()})}, "
               "open(sys.argv[1], 'w'))")


def run_step(cmd: list[str], cwd: Path | None = None, env: dict | None = None,
             timeout: float = 1800) -> tuple[bool, float, str]:
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        ok, out = p.returncode == 0, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired as e:
        ok, out = False, f"TIMEOUT after {timeout}s\n{(e.stdout or b'')[-4000:]!r}"
    return ok, round(time.monotonic() - t0, 1), out[-6000:]


def tenv_run(cmd: list, cwd: Path | None = None, extra: dict | None = None,
             timeout: float = 1800) -> tuple[bool, float, str]:
    """Run a test-env command: sandboxed as pkgtest when $PKG_SRC_SANDBOX is set."""
    cmd = [str(c) for c in cmd]
    extra = {k: str(v) for k, v in (extra or {}).items()}
    if not SANDBOX:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("HUGPY", "PIP_", "VIRTUAL_ENV", "PG"))} | extra
        return run_step(cmd, cwd=cwd, env=env, timeout=timeout)
    wrap = ["sudo", "-n", SANDBOX, "--timeout", str(int(timeout)), "--cwd", str(cwd or TESTENV)]
    for k, v in extra.items():
        wrap += ["-E", f"{k}={v}"]
    # a shell parent, as under CI: the command must not be the unit's main process
    # (hugpy_fleet's restart tests assert their parent is not the systemd manager)
    shell = ["/bin/sh", "-c", '"$@"; rc=$?; exit $rc', "pkg-test"]
    return run_step([*wrap, "--", *shell, *cmd], timeout=timeout + 60)


def tenv_rmtree(path: Path) -> None:
    if SANDBOX and path.exists():                    # what the sandbox wrote belongs to pkgtest
        tenv_run(["rm", "-rf", path], timeout=600)
    shutil.rmtree(path, ignore_errors=True)


def base_env(venv: Path, home: Path) -> dict[str, str]:
    return {"PATH": f"{venv / 'bin'}:/usr/local/bin:/usr/bin:/bin", "PYTHONNOUSERSITE": "1",
            "HOME": str(home), "XDG_CACHE_HOME": str(home / ".cache"),
            "PIP_CACHE_DIR": str(TESTENV / "pip-cache"), "PIP_DISABLE_PIP_VERSION_CHECK": "1"}


def norm_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def dist_name(pkg_dir: Path) -> str | None:
    m = re.search(r'^\s*name\s*=\s*"([^"]+)"', (pkg_dir / "pyproject.toml").read_text(), re.M)
    return m.group(1) if m else None


def materialise(cur, members: dict[str, int], root: Path) -> dict[str, Path]:
    """Write each {package: version_id} to root/<group>/<package>; returns the dirs."""
    cur.execute(sql.SQL("SELECT package, group_dir FROM {s}.packages")
                .format(s=sql.Identifier(SCHEMA)))
    groups = dict(cur.fetchall())
    dirs = {}
    for pkg, vid in members.items():
        d = root / groups.get(pkg, "_") / pkg
        files = version_files(cur, vid)
        if not files:
            continue
        for rel, entry in files.items():
            write_entry(d / rel, *entry)
        dirs[pkg] = d
    return dirs


def scm_env(cur, members: dict[str, int], dirs: dict[str, Path]) -> dict[str, str]:
    """setuptools-scm has no git in a materialised tree: hand each dist its label."""
    env = {}
    for pkg, d in dirs.items():
        cur.execute(sql.SQL("SELECT label FROM {s}.versions WHERE id = %s")
                    .format(s=sql.Identifier(SCHEMA)), (members[pkg],))
        label = cur.fetchone()[0]
        name = dist_name(d)
        if name:
            env[f"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_{re.sub(r'[-_.]+', '_', name).upper()}"] = label
    return env


def workspace_dists(dirs: dict[str, Path]) -> set[str]:
    return {norm_dist(n) for n in map(dist_name, dirs.values()) if n}


def env_fingerprint(venv: Path, home: Path, workspace: set[str]) -> dict | None:
    """python version + every third-party dist (workspace packages excluded), hashed."""
    out = home / "env-fingerprint.json"
    out.unlink(missing_ok=True)
    ok, _, _ = tenv_run([venv / "bin" / "python", "-c", FINGERPRINT, out],
                        cwd=home, extra=base_env(venv, home), timeout=120)
    if not ok or not out.exists():
        return None
    data = json.loads(out.read_text())
    dists = [d for d in data["dists"] if norm_dist(d.split("==")[0]) not in workspace]
    freeze = "\n".join(dists)
    return {"python": data["python"], "freeze": freeze, "count": len(dists),
            "sha": hashlib.sha256(f"python {data['python']}\n{freeze}".encode()).hexdigest()}


def testenv_row(cur, env_dir: Path) -> tuple[int, str] | None:
    cur.execute(sql.SQL("SELECT id, sha FROM {s}.testenvs WHERE path = %s")
                .format(s=sql.Identifier(SCHEMA)), (str(env_dir),))
    return cur.fetchone()


def current_env() -> Path | None:
    cur = TESTENV / "envs" / "current"
    return cur.resolve() if cur.exists() else None


def testenv_build(conn, members: dict[str, int], reason: str, log=print) -> Path | None:
    """Build a new env from scratch with the third-party deps of ``members``, record its
    fingerprint and make it current. Older envs stay on disk and in the table."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = TESTENV / "envs" / stamp
    venv, home = d / "venv", d / "home"
    home.mkdir(parents=True, exist_ok=True)
    with conn.cursor() as cur:
        dirs = materialise(cur, members, d / "src")
        env = base_env(venv, home) | scm_env(cur, members, dirs)
    workspace = workspace_dists(dirs)
    py = venv / "bin" / "python"
    for step, cmd, timeout in (
            ("venv", [TESTENV_PYTHON, "-m", "venv", venv], 600),
            ("tools", [py, "-m", "pip", "install", "-q", *TEST_TOOLS], 1800),
            ("deps", [py, "-m", "pip", "install", "-q", *dirs.values()], 7200),
            # third-party only: every job installs its own members editable
            ("drop-workspace", [py, "-m", "pip", "uninstall", "-y", "-q", *sorted(workspace)], 600),
            ("check", [py, "-m", "pip", "check"], 300)):
        ok, secs, out = tenv_run(cmd, cwd=d, extra=env, timeout=timeout)
        log(f"testenv {stamp}: {step} {'ok' if ok else 'FAILED'} ({secs}s)")
        if not ok:
            log(out)
            tenv_rmtree(d)
            return None
    tenv_rmtree(d / "src")
    fp = env_fingerprint(venv, home, workspace)
    if fp is None:
        log(f"testenv {stamp}: fingerprint FAILED")
        tenv_rmtree(d)
        return None
    with conn.cursor() as cur:
        cur.execute(sql.SQL("INSERT INTO {s}.testenvs (path, sha, python, dists, built_for, reason) "
                            "VALUES (%s,%s,%s,%s,%s,%s)").format(s=sql.Identifier(SCHEMA)),
                    (str(d), fp["sha"], fp["python"], fp["freeze"], Jsonb(members), reason))
    conn.commit()
    link = TESTENV / "envs" / ".current.new"
    link.unlink(missing_ok=True)
    link.symlink_to(stamp)
    os.replace(link, TESTENV / "envs" / "current")
    log(f"testenv {stamp}: current (python {fp['python']}, {fp['count']} dists, {fp['sha'][:12]})")
    return d


def api_testenvs(conn) -> list[dict]:
    cur_env = current_env()
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT id, path, sha, python, reason, built_at, "
                            "array_length(string_to_array(dists, E'\\n'), 1) "
                            "FROM {s}.testenvs ORDER BY id DESC").format(s=sql.Identifier(SCHEMA)))
        return [{"id": i, "path": p, "sha": s[:12], "python": py, "dists": n, "reason": r,
                 "built_at": b.isoformat(), "current": cur_env is not None and Path(p) == cur_env,
                 "on_disk": Path(p).exists()}
                for i, p, s, py, r, b, n in cur.fetchall()]


# Worker import probe: boot the real worker (it is what the fleet runs) with an audit hook
# in EVERY python process it starts (sitecustomize on PYTHONPATH), so what it actually
# imports and spawns is evidence, not a static guess. Unreachable central, own ports.
WORKER_ENTRY = "hugpy_fleet.worker"
PROBE_SECS = 30
PROBE_PORTS = {"WORKER_PORT": "19399", "SLOT_PORT_BASE": "18101", "SLOT_ADVERTISE": "127.0.0.1"}
PROBE_SITE = r"""# pkg_src worker probe: record every import and spawned command of this process
import os, sys, json, time, threading, atexit
_dir = os.environ.get("PKG_PROBE_DIR")
if _dir:
    _seen, _spawns, _owners = set(), [], None

    def _hook(ev, args):
        if ev == "import":
            _seen.add(args[0])
        elif ev in ("subprocess.Popen", "os.exec", "os.posix_spawn"):
            a = args[1]
            _spawns.append([str(x) for x in (a if isinstance(a, (list, tuple)) else [a])])
    sys.addaudithook(_hook)

    def _dump():
        global _owners
        try:
            if _owners is None:
                import importlib.metadata as md
                _owners = md.packages_distributions()
            mods = {m: (_owners.get(m.split(".")[0]) or [None])[0]
                    for m in list(_seen) if m in sys.modules}
            tmp = os.path.join(_dir, f".{os.getpid()}.tmp")
            with open(tmp, "w") as f:
                json.dump({"pid": os.getpid(), "argv": sys.argv, "modules": mods,
                           "spawns": _spawns}, f)
            os.replace(tmp, os.path.join(_dir, f"{os.getpid()}.json"))
        except Exception:
            pass

    def _loop():
        while True:
            time.sleep(2)
            _dump()
    threading.Thread(target=_loop, daemon=True, name="pkg-probe").start()
    atexit.register(_dump)
"""


def static_worker_packages(conn, gid: int) -> set[str]:
    """Packages reachable by imports from the worker entry in graph ``gid``."""
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            WITH RECURSIVE reach(m) AS (
                SELECT module FROM {s}.graph_modules WHERE graph_id = %s AND module = %s
                UNION
                SELECT i.dst_module FROM {s}.graph_imports i JOIN reach r ON i.src_module = r.m
                WHERE i.graph_id = %s AND i.dst_package IS NOT NULL)
            SELECT DISTINCT g.package FROM reach r
            JOIN {s}.graph_modules g ON g.graph_id = %s AND g.module = r.m""")
                    .format(s=sql.Identifier(SCHEMA)), (gid, WORKER_ENTRY, gid, gid))
        return {r[0] for r in cur.fetchall()}


def retired_refs(dirs: dict[str, Path], packages, retired: set[str]) -> list[str]:
    """file:line in live (non-test) source that still names a retired package."""
    if not retired:
        return []
    pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(retired))) + r")\b")
    hits = []
    for pkg in sorted(packages):
        root = dirs[pkg]
        for f in sorted((root / "src").rglob("*.py")) if (root / "src").is_dir() else []:
            for n, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if pat.search(line):
                    hits.append(f"{f.relative_to(root.parent)}:{n}: {line.strip()[:160]}")
    return hits


def worker_probe(conn, job_id: int, gid: int, work: Path, venv: Path, env: dict,
                 dirs: dict[str, Path]) -> dict:
    t0 = time.monotonic()
    site, out = work / "probe" / "site", work / "probe" / "out"
    site.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (site / "sitecustomize.py").write_text(PROBE_SITE)
    log = work / "probe" / "worker.log"
    penv = env | PROBE_PORTS | {"PYTHONPATH": site, "PKG_PROBE_DIR": out, "PROBE_LOG": log}
    tenv_run(["sh", "-c", f'timeout -s TERM -k 5 {PROBE_SECS} "$@" > "$PROBE_LOG" 2>&1; exit 0',
              "probe", venv / "bin" / "python", "-m", WORKER_ENTRY,
              "--central", "http://127.0.0.1:9", "--name", f"probe-{job_id}",
              "--port", PROBE_PORTS["WORKER_PORT"]],
             cwd=work / "home", extra=penv, timeout=PROBE_SECS + 60)
    with conn.cursor() as cur:
        retired = retired_packages(cur)
    by_dist = {norm_dist(n): p for p, n in ((p, dist_name(d)) for p, d in dirs.items()) if n}
    procs = [json.loads(f.read_text()) for f in sorted(out.glob("*.json"))]
    rows, runtime, dists, spawns, bad = [], set(), {}, [], []
    for pr in procs:
        proc = " ".join(pr["argv"])[:300]
        for mod, dist in pr["modules"].items():
            top = mod.split(".")[0]
            pkg = top if top in dirs else by_dist.get(norm_dist(dist)) if dist else None
            rows.append((job_id, pr["pid"], proc, mod, dist, pkg))
            if pkg:
                runtime.add(pkg)
            if dist:
                dists[dist] = dists.get(dist, 0) + 1
            if top in retired:
                bad.append(f"imported retired module {mod} (pid {pr['pid']})")
        for cmd in pr["spawns"]:
            s = " ".join(cmd)
            spawns.append(s[:300])
            if any(re.search(rf"\b{re.escape(r)}\b", s) for r in retired):
                bad.append(f"spawned retired package: {s[:300]}")
    text = log.read_text(errors="replace") if log.exists() else ""
    ws_tops = set(dirs) | retired
    errors = [ln.strip()[:300] for ln in text.splitlines()
              if re.search(r"ModuleNotFoundError|ImportError|No module named", ln)]
    missing = {m.split(".")[0] for m in re.findall(r"No module named '([\w.]+)'", text)}
    fatal = [f"missing workspace module: {m}" for m in sorted(missing & ws_tops)]
    started = any(m.startswith(WORKER_ENTRY) for pr in procs for m in pr["modules"])
    with conn.cursor() as cur:
        cur.executemany(sql.SQL("INSERT INTO {s}.worker_probe (job_id, pid, process, module, dist, "
                                "package) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING")
                        .format(s=sql.Identifier(SCHEMA)), rows)
    conn.commit()
    static = static_worker_packages(conn, gid)
    ok = started and not bad and not fatal
    return {"ok": ok, "seconds": round(time.monotonic() - t0, 1), "started": started,
            "processes": len(procs), "modules": len(rows),
            "workspace_packages": sorted(runtime), "static_packages": sorted(static),
            "static_only": sorted(static - runtime), "runtime_only": sorted(runtime - static),
            "dists": dict(sorted(dists.items(), key=lambda kv: -kv[1])),
            "retired_hits": bad, "import_errors": fatal, "optional_import_errors": errors[:20],
            "retired_refs_in_source": retired_refs(dirs, runtime | static, retired)[:100],
            "spawns": spawns[:50], **({} if ok else {"log_tail": text[-3000:]})}


def run_verify(conn, job_id: int, members: dict[str, int], test: list[str]) -> dict:
    """Install ``members`` into the current test env and run the CI checks for ``test``.

    Mirrors .github/workflows/ci.yml: every member installed editable (from the
    DB-materialised tree), then per tested package: wheel build, pytest
    --timeout=120 (under coverage with per-test contexts -> pkg_src.test_trace),
    and finally the built wheels installed non-editable and imported."""
    import fcntl
    TESTENV.mkdir(parents=True, exist_ok=True)
    lock = open(TESTENV / "lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)                  # one job owns the env at a time
    result: dict = {"members": members, "tested": test, "env": {}}
    gid = pkg_graph.build_graph(conn, members)
    result["graph"] = gid
    work = TESTENV / "work" / f"job-{job_id}"
    tenv_rmtree(work)
    home = work / "home"
    home.mkdir(parents=True)
    (work / "cov").mkdir()
    with conn.cursor() as cur:
        dirs = materialise(cur, members, work / "py")
        scm = scm_env(cur, members, dirs)
    workspace = workspace_dists(dirs)
    editable = [a for d in dirs.values() for a in ("-e", str(d))]

    def install(venv: Path) -> dict:
        env = base_env(venv, home) | scm
        py = venv / "bin" / "python"
        steps = {}
        ok, secs, out = tenv_run([py, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
                                  *editable], cwd=work, extra=env)
        steps["install"] = {"ok": ok, "seconds": secs, **({} if ok else {"tail": out})}
        if ok:
            ok, secs, out = tenv_run([py, "-m", "pip", "check"], cwd=work, extra=env, timeout=300)
            steps["pip_check"] = {"ok": ok, "seconds": secs, **({} if ok else {"tail": out})}
        if ok:
            fp = env_fingerprint(venv, home, workspace)
            with conn.cursor() as cur:
                row = testenv_row(cur, venv.parent)
            ok = bool(fp and row and fp["sha"] == row[1])
            result["testenv"] = {"id": row and row[0], "path": str(venv.parent),
                                 "sha": fp and fp["sha"], "recorded_sha": row and row[1], "ok": ok}
            steps["testenv"] = {"ok": ok, "seconds": 0, **({} if ok else {
                "tail": "env does not match its recorded fingerprint (or is unrecorded)"})}
        return steps

    env_root = current_env()
    if env_root is None:
        env_root = testenv_build(conn, members, f"job {job_id}: no env yet")
    if env_root is None:
        result["env"]["testenv_build"] = {"ok": False, "seconds": 0, "tail": "see job log"}
        return result | {"status": "error"}
    steps = install(env_root / "venv")
    failed = next((k for k, v in steps.items() if not v["ok"]), None)
    if failed in ("pip_check", "testenv"):            # never patch an env: build a new one
        why = "member deps changed" if failed == "pip_check" else "env drifted from its fingerprint"
        rebuilt = testenv_build(conn, members, f"job {job_id}: {why}")
        if rebuilt is not None:
            env_root = rebuilt
            steps = install(env_root / "venv") | {"rebuilt": {"ok": True, "seconds": 0, "why": why}}
    result["env"].update(steps)
    venv = env_root / "venv"
    py = venv / "bin" / "python"
    env = base_env(venv, home) | scm
    env_ok = all(v["ok"] for v in result["env"].values())
    all_ok = env_ok
    rows, traced = [], {}
    wheels = work / "wheels"

    def record(pkg, step, ok, secs, out):
        nonlocal all_ok
        all_ok &= ok
        rows.append((job_id, pkg, members[pkg], step, ok, secs, out[-3000:]))
        print(f"job {job_id}: {pkg:20} {step:7} {'ok' if ok else 'FAIL'} {secs}s", flush=True)

    tested = [p for p in test if p in dirs] if env_ok else []
    for pkg in tested:
        d = dirs[pkg]
        record(pkg, "build", *tenv_run([py, "-m", "build", "--wheel", "--no-isolation",
                                        "--outdir", wheels], cwd=d, extra=env, timeout=900))
        if (d / "tests").is_dir():
            cov = work / "cov" / f"{pkg}.coverage"
            ok, secs, out = tenv_run(
                [py, "-m", "pytest", "-q", "--timeout=120", "-p", "no:cacheprovider",
                 f"--cov={work / 'py'}", "--cov-context=test", "--cov-report="],
                cwd=d, extra=env | {"COVERAGE_FILE": cov}, timeout=PYTEST_BUDGET)
            record(pkg, "pytest", ok, secs, out)
            if cov.exists():
                try:
                    traced[pkg] = pkg_graph.load_coverage_trace(conn, job_id, pkg, str(cov), gid)
                except Exception as e:                # tracing must never fail the test verdict
                    traced[pkg] = f"trace error: {type(e).__name__}: {e}"
    if tested:                                        # CI's last step: the wheel itself imports
        built = sorted(wheels.glob("*.whl"))
        ok, secs, out = tenv_run([py, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
                                  *built], cwd=work, extra=env)
        result["env"]["wheel_install"] = {"ok": ok, "seconds": secs, **({} if ok else {"tail": out})}
        all_ok &= ok
        for pkg in tested:
            record(pkg, "import", *tenv_run([py, "-c", f"import {pkg}"], cwd=home, extra=env,
                                            timeout=300))
    if env_ok and "hugpy_fleet" in dirs:              # what the fleet runs, observed
        try:
            probe = worker_probe(conn, job_id, gid, work, venv, env, dirs)
        except Exception as e:
            probe = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        result["worker_probe"] = probe
        all_ok &= probe["ok"]
        print(f"job {job_id}: worker probe {'ok' if probe['ok'] else 'FAIL'}", flush=True)
    with conn.cursor() as cur:
        cur.executemany(sql.SQL("INSERT INTO {s}.test_results (job_id, package, version_id, step, "
                                "ok, seconds, output_tail) VALUES (%s,%s,%s,%s,%s,%s,%s)")
                        .format(s=sql.Identifier(SCHEMA)), rows)
    conn.commit()
    result["trace_rows"] = traced
    try:
        result["analysis"] = analysis_report(conn, gid, members, job_id=job_id, tested=tested)
    except Exception as e:
        result["analysis"] = {"error": f"{type(e).__name__}: {e}"}
    if all_ok:
        tenv_rmtree(work)                             # keep failing trees for inspection
    else:
        result["work_dir"] = str(work)
    return result | {"status": "pass" if all_ok else "fail"}


def analysis_report(conn, gid: int, members: dict[str, int], job_id: int | None = None,
                    tested: list[str] | None = None, top: int = 25) -> dict:
    """Everything inferable from the symbol index + graph (and the job's test trace)."""
    with conn.cursor() as cur:
        dups = pkg_graph.duplicates(cur, gid)
        coll = pkg_graph.name_collisions(cur, gid)
        dead = pkg_graph.unreferenced(cur, gid)
        probs = pkg_graph.dependency_problems(cur, gid, members)
        cur.execute(sql.SQL("SELECT stats FROM {s}.graphs WHERE id = %s")
                    .format(s=sql.Identifier(SCHEMA)), (gid,))
        out = {"graph": gid, "stats": cur.fetchone()[0],
               "duplicate_groups": len(dups),
               "duplicate_functions": sum(d["copies"] for d in dups),
               "cross_package_duplicate_groups": sum(1 for d in dups if len(d["packages"]) > 1),
               "top_duplicates": dups[:top],
               "name_collisions": len(coll), "top_name_collisions": coll[:top],
               "unreferenced_functions": len(dead), "unreferenced_sample": dead[:top],
               **probs}
        if job_id is not None and tested:
            out |= pkg_graph.trace_report(cur, job_id, gid, tested)
    return out


# --------------------------------------------------------------------------- jobs

def current_members(cur) -> dict[str, int]:
    cur.execute(sql.SQL("SELECT package, version_id FROM {s}.packages WHERE version_id IS NOT NULL")
                .format(s=sql.Identifier(SCHEMA)))
    return live_members(cur, dict(cur.fetchall()))


def retired_packages(cur) -> set[str]:
    cur.execute(sql.SQL("SELECT package FROM {s}.retired").format(s=sql.Identifier(SCHEMA)))
    return {r[0] for r in cur.fetchall()}


def live_members(cur, members: dict[str, int]) -> dict[str, int]:
    """Drop retired packages: still versioned (history), never tested, pinned or promoted."""
    gone = retired_packages(cur)
    return {p: v for p, v in members.items() if p not in gone}


def api_retire(conn, package: str, note: str | None = None, undo: bool = False) -> dict:
    with conn.cursor() as cur:
        if undo:
            cur.execute(sql.SQL("DELETE FROM {s}.retired WHERE package = %s")
                        .format(s=sql.Identifier(SCHEMA)), (package,))
        else:
            cur.execute(sql.SQL("INSERT INTO {s}.retired (package, note) VALUES (%s, %s) "
                                "ON CONFLICT (package) DO UPDATE SET note = EXCLUDED.note")
                        .format(s=sql.Identifier(SCHEMA)), (package, note))
        gone = sorted(retired_packages(cur))
    conn.commit()
    return {"package": package, "retired": not undo, "all_retired": gone}


def parse_overrides(spec) -> dict[str, str]:
    """'pkg@ref,pkg2@ref2' | {pkg: ref} → {pkg: ref}; a bare 'pkg' means pkg@current."""
    if isinstance(spec, dict):
        return spec
    out = {}
    for item in filter(None, (x.strip() for x in (spec or "").split(","))):
        pkg, _, ref = item.partition("@")
        out[pkg] = ref or "current"
    return out


def api_request(conn, kind: str, config: str | None = None, packages=None,
                requested_by: str | None = None) -> dict:
    """Queue a verify/restore job. Versions are resolved NOW, so later edits to the
    tree don't change what the job tests or restores."""
    with conn.cursor() as cur:
        if config:
            members = live_members(cur, config_members(cur, config))
            test = sorted(members)
        else:
            over = parse_overrides(packages)
            if not over:
                raise SystemExit("give a config or packages ('pkg@ref,…')")
            gone = retired_packages(cur) & over.keys()
            if gone:
                raise SystemExit(f"retired package(s): {', '.join(sorted(gone))}")
            resolved = {p: resolve_version(cur, p, r) for p, r in over.items()}
            members = current_members(cur) | resolved if kind == "verify" else resolved
            test = sorted(resolved)
        args = {"config": config, "members": members, "test": test}
        cur.execute(sql.SQL("INSERT INTO {s}.jobs (kind, args, requested_by) VALUES (%s, %s, %s) "
                            "RETURNING id").format(s=sql.Identifier(SCHEMA)),
                    (kind, Jsonb(args), requested_by))
        jid = cur.fetchone()[0]
    conn.commit()
    return {"job": jid, "kind": kind, "status": "queued", **args}


def api_job(conn, job_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT id, kind, args, status, requested_by, created_at, started_at, "
                            "finished_at, result FROM {s}.jobs WHERE id = %s")
                    .format(s=sql.Identifier(SCHEMA)), (job_id,))
        row = cur.fetchone()
        if row is None:
            raise SystemExit(f"no job {job_id}")
        keys = ("job", "kind", "args", "status", "requested_by", "created_at", "started_at",
                "finished_at", "result")
        out = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in zip(keys, row)}
        cur.execute(sql.SQL("SELECT r.package, v.label, r.step, r.ok, r.seconds, "
                            "CASE WHEN r.ok THEN NULL ELSE r.output_tail END "
                            "FROM {s}.test_results r JOIN {s}.versions v ON v.id = r.version_id "
                            "WHERE r.job_id = %s ORDER BY r.package, r.step")
                    .format(s=sql.Identifier(SCHEMA)), (job_id,))
        out["tests"] = [dict(zip(("package", "label", "step", "ok", "seconds", "tail"), r))
                        for r in cur.fetchall()]
    return out


def api_jobs(conn, limit: int = 20) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT id, kind, status, args->>'config', args->'test', created_at, "
                            "finished_at FROM {s}.jobs ORDER BY id DESC LIMIT %s")
                    .format(s=sql.Identifier(SCHEMA)), (limit,))
        return [dict(job=i, kind=k, status=st, config=c, test=t, created_at=a.isoformat(),
                     finished_at=f.isoformat() if f else None)
                for i, k, st, c, t, a, f in cur.fetchall()]


def passing_job_for_version(cur, vid: int) -> int | None:
    """Latest passing verify job that actually tested version ``vid``."""
    cur.execute(sql.SQL("""
        SELECT j.id FROM {s}.jobs j
        WHERE j.kind = 'verify' AND j.status = 'pass'
          AND (j.result->'testenv'->>'ok')::boolean     -- ran on an env matching its record
          AND EXISTS (SELECT 1 FROM {s}.test_results r WHERE r.job_id = j.id AND r.version_id = %s)
        ORDER BY j.id DESC LIMIT 1""").format(s=sql.Identifier(SCHEMA)), (vid,))
    row = cur.fetchone()
    return row[0] if row else None


def passing_job_for_members(cur, members: dict[str, int]) -> int | None:
    """Latest passing verify job that installed exactly ``members`` and tested all of them."""
    cur.execute(sql.SQL("""
        SELECT id FROM {s}.jobs
        WHERE kind = 'verify' AND status = 'pass' AND args->'members' = %s
          AND (result->'testenv'->>'ok')::boolean
          AND (SELECT count(DISTINCT package) FROM {s}.test_results r WHERE r.job_id = jobs.id)
              >= %s
        ORDER BY id DESC LIMIT 1""").format(s=sql.Identifier(SCHEMA)),
        (Jsonb({k: v for k, v in members.items()}),
         sum(1 for v in members.values() if v is not None) - empty_members(cur, members)))
    row = cur.fetchone()
    return row[0] if row else None


def empty_members(cur, members: dict[str, int]) -> int:
    cur.execute(sql.SQL("SELECT count(*) FROM {s}.versions WHERE id = ANY(%s) AND file_count = 0")
                .format(s=sql.Identifier(SCHEMA)), (list(members.values()),))
    return cur.fetchone()[0]


def api_mark_good(conn, package: str | None = None, ref: str = "current",
                  config: str | None = None, note: str | None = None, unset: bool = False) -> dict:
    """Flag a package version or a configuration known-good. Refused unless a verify
    job passed on exactly that version / that configuration's full member set."""
    s = sql.Identifier(SCHEMA)
    with conn.cursor() as cur:
        if config:
            cid = config_id(cur, config)
            job = None if unset else passing_job_for_members(
                cur, live_members(cur, config_members(cur, config)))
            if not unset and job is None:
                raise SystemExit(f"config {config} has no passing verify job — "
                                 f"run: pkg_src.py job verify --config {config}")
            cur.execute(sql.SQL("UPDATE {s}.configs SET known_good = %s, good_at = CASE WHEN %s "
                                "THEN now() END, note = coalesce(%s, note) WHERE id = %s")
                        .format(s=s), (not unset, not unset, note, cid))
            conn.commit()
            return {"config": config, "known_good": not unset, "job": job}
        vid = resolve_version(cur, package, ref)
        job = None if unset else passing_job_for_version(cur, vid)
        if not unset and job is None:
            raise SystemExit(f"{package}@{ref} has no passing verify job — "
                             f"run: pkg_src.py job verify {package}@{ref}")
        cur.execute(sql.SQL("UPDATE {s}.versions SET known_good = %s, good_note = %s, "
                            "good_at = CASE WHEN %s THEN now() END WHERE id = %s RETURNING label")
                    .format(s=s), (not unset, note, not unset, vid))
        label = cur.fetchone()[0]
    conn.commit()
    return {"package": package, "label": label, "known_good": not unset, "job": job}


def api_versions(conn, package: str | None = None, good_only: bool = False,
                 limit: int = 50) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            SELECT v.id, v.package, v.label, v.file_count, v.source, v.created_at, v.known_good,
                   v.good_note, v.id = p.version_id,
                   (SELECT j.status FROM {s}.test_results r JOIN {s}.jobs j ON j.id = r.job_id
                    WHERE r.version_id = v.id ORDER BY j.id DESC LIMIT 1)
            FROM {s}.versions v LEFT JOIN {s}.packages p USING (package)
            WHERE (%s::text IS NULL OR v.package = %s) AND (NOT %s OR v.known_good)
            ORDER BY v.id DESC LIMIT %s""").format(s=sql.Identifier(SCHEMA)),
            (package, package, good_only, limit))
        return [dict(id=i, package=p, label=l, files=n, source=src, created_at=a.isoformat(),
                     known_good=g, good_note=gn, current=bool(c), last_test=t)
                for i, p, l, n, src, a, g, gn, c, t in cur.fetchall()]


def api_configs(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            SELECT c.name, c.source, c.created_at, c.known_good, c.note,
                   json_object_agg(m.package, v.label ORDER BY m.package)
            FROM {s}.configs c JOIN {s}.config_members m ON m.config_id = c.id
            JOIN {s}.versions v ON v.id = m.version_id
            GROUP BY c.id ORDER BY c.created_at DESC""").format(s=sql.Identifier(SCHEMA)))
        return [dict(name=n, source=src, created_at=a.isoformat(), known_good=g, note=note,
                     members=m) for n, src, a, g, note, m in cur.fetchall()]


def api_config_save(conn, name: str, note: str | None = None) -> dict:
    """Pin the versions the mirror currently holds (the watcher keeps it seconds behind disk)."""
    with conn.cursor() as cur:
        members = current_members(cur)
        cur.execute(sql.SQL("SELECT git_head FROM {s}.packages ORDER BY synced_at DESC LIMIT 1")
                    .format(s=sql.Identifier(SCHEMA)))
        head = (cur.fetchone() or [None])[0]
        insert_config(cur, name, members, "mirror", head, False, note)
    conn.commit()
    return {"config": name, "packages": len(members)}


def members_for(conn, config: str | None = None, job: int | None = None) -> dict[str, int]:
    with conn.cursor() as cur:
        if config:
            return config_members(cur, config)
        if job:
            cur.execute(sql.SQL("SELECT args->'members' FROM {s}.jobs WHERE id = %s")
                        .format(s=sql.Identifier(SCHEMA)), (job,))
            row = cur.fetchone()
            if row is None:
                raise SystemExit(f"no job {job}")
            return {k: int(v) for k, v in row[0].items()}
        return current_members(cur)


def api_analysis(conn, config: str | None = None, job: int | None = None) -> dict:
    """Duplicates, collisions, dependency problems, matrix — for the current tree,
    a configuration, or a job's member set (plus that job's test coverage)."""
    members = members_for(conn, config, job)
    gid = pkg_graph.build_graph(conn, members)
    tested = None
    if job:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT DISTINCT package FROM {s}.test_trace WHERE job_id = %s")
                        .format(s=sql.Identifier(SCHEMA)), (job,))
            tested = [r[0] for r in cur.fetchall()]
    return analysis_report(conn, gid, members, job_id=job, tested=tested)


def api_duplicates(conn, config: str | None = None, job: int | None = None,
                   with_callers: bool = False) -> list[dict]:
    members = members_for(conn, config, job)
    gid = pkg_graph.build_graph(conn, members)
    with conn.cursor() as cur:
        return pkg_graph.duplicate_calls(cur, gid) if with_callers else pkg_graph.duplicates(cur, gid)


def split_fn(function: str) -> tuple[str, str]:
    if ":" not in function:
        raise SystemExit("function must be 'module:qualname', e.g. hugpy_fleet.worker.agent:main")
    return tuple(function.split(":", 1))


def api_trace(conn, function: str | None = None, test: str | None = None,
              job: int | None = None, config: str | None = None, depth: int = 4) -> dict:
    """Traceability. function='module:qualname' → tests that executed it (dynamic, from
    the latest traced job or ``job``), its static callers and callees. test='pytest
    node id' (+ job) → every workspace function that test executed."""
    with conn.cursor() as cur:
        if test:
            if not job:
                cur.execute(sql.SQL("SELECT max(job_id) FROM {s}.test_trace WHERE test = %s")
                            .format(s=sql.Identifier(SCHEMA)), (test,))
                job = cur.fetchone()[0]
            return {"test": test, "job": job,
                    "executed": pkg_graph.test_functions(cur, job, test) if job else []}
    if not function:
        raise SystemExit("give function='module:qualname' or test='node id'")
    mod, q = split_fn(function)
    members = members_for(conn, config, job)
    gid = pkg_graph.build_graph(conn, members)
    with conn.cursor() as cur:
        return {"function": function, "graph": gid,
                "tested_by": pkg_graph.function_tests(cur, mod, q, job),
                "callers": pkg_graph.callers(cur, gid, mod, q, depth),
                "callees": pkg_graph.callees(cur, gid, mod, q, depth)}


def api_symbols(conn, name: str, kind: str | None = None, config: str | None = None,
                limit: int = 100) -> list[dict]:
    """Find functions/classes/variables by name (SQL LIKE pattern) in the current tree
    (or a configuration), with location, signature and body hashes."""
    gid = pkg_graph.build_graph(conn, members_for(conn, config))
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            SELECT m.package, m.module, s.qualname, s.kind, s.lineno, s.args, s.doc,
                   left(s.norm_sha, 12), m.rel_path
            FROM {s}.graph_modules m JOIN {s}.py_symbols s ON s.sha256 = m.sha256
            WHERE m.graph_id = %s AND s.name LIKE %s AND (%s::text IS NULL OR s.kind = %s)
            ORDER BY m.module, s.lineno LIMIT %s""").format(s=sql.Identifier(SCHEMA)),
            (gid, name, kind, kind, limit))
        keys = ("package", "module", "qualname", "kind", "line", "args", "doc", "norm_sha", "file")
        return [dict(zip(keys, r)) for r in cur.fetchall()]


def run_job(dsn: str, job_id: int) -> int:
    """Execute one claimed job (the watcher spawns this in a child process)."""
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT kind, args FROM {s}.jobs WHERE id = %s")
                        .format(s=sql.Identifier(SCHEMA)), (job_id,))
            kind, args = cur.fetchone()
        members = {k: int(v) for k, v in args["members"].items()}
        try:
            if kind == "verify":
                result = run_verify(conn, job_id, members, args["test"])
            else:
                pkgs = discover(PY_ROOT)
                restore_packages(conn, pkgs, {p: v for p, v in members.items() if p in pkgs},
                                 True, f"config {args['config']}" if args.get("config")
                                 else f"job {job_id}")
                result = {"status": "done"}
        except (Exception, SystemExit) as e:
            result = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        with conn.cursor() as cur:
            cur.execute(sql.SQL("UPDATE {s}.jobs SET status = %s, result = %s, finished_at = now() "
                                "WHERE id = %s").format(s=sql.Identifier(SCHEMA)),
                        (result["status"], Jsonb(result), job_id))
        conn.commit()
    print(f"job {job_id}: {result['status']}", flush=True)
    return 0 if result["status"] in ("pass", "done") else 1


def claim_job(conn) -> int | None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            UPDATE {s}.jobs SET status = 'running', started_at = now()
            WHERE id = (SELECT id FROM {s}.jobs WHERE status = 'queued'
                        ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED)
            RETURNING id""").format(s=sql.Identifier(SCHEMA)))
        row = cur.fetchone()
    conn.commit()
    return row[0] if row else None


# ------------------------------------------------------------------------- watch

def tree_signature(repo: Path, pkgs: dict[str, Path]) -> dict[str, tuple]:
    """Cheap per-package fingerprint: (path, mtime_ns, size, mode) of git's file set."""
    listed = _git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "py")
    prefixes = sorted(((str(d.relative_to(repo)) + "/", n) for n, d in pkgs.items()),
                      key=lambda t: -len(t[0]))
    sig: dict[str, list] = {n: [] for n in pkgs}
    for p in filter(None, listed.split("\0")):
        for pre, name in prefixes:
            if p.startswith(pre):
                try:
                    st = os.lstat(repo / p)
                except FileNotFoundError:
                    break
                sig[name].append((p, st.st_mtime_ns, st.st_size, st.st_mode))
                break
    return {n: tuple(sorted(v)) for n, v in sig.items()}


PROMOTE_EVERY = 60              # seconds between promote steps in the watcher


def promote_step(dsn: str) -> None:
    """Deploy the newest known-good configuration and judge the open rollout (pkg_promote):
    central adopts the build, workers follow it on heartbeat, the workers' health decides
    rollback. Every call is idempotent; failures are logged, never fatal to the watcher."""
    import pkg_promote
    with psycopg.connect(dsn) as conn:
        tick = pkg_promote.rollout_tick(conn)
        # The GitHub commit+push step of this pipeline: when a rollout is judged
        # healthy on this tick, publish the source trees (a hard no-op unless an
        # operator wrote pkg_publish.toml with enabled=true). Publishing must
        # NEVER block or fail a promotion, so it is fully guarded here.
        if tick.get("status") == "healthy" and tick.get("promotion"):
            try:
                import pkg_publish
                pkg_publish.publish_after_promotion(conn, tick["promotion"])
            except (Exception, SystemExit) as e:
                print(f"publish: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT name FROM {s}.configs WHERE known_good "
                                "ORDER BY good_at DESC NULLS LAST, id DESC LIMIT 1")
                        .format(s=sql.Identifier(SCHEMA)))
            row = cur.fetchone()
        if row:
            pkg_promote.auto_promote(conn, row[0])


def cmd_watch(dsn: str, py_root: Path, interval: float, settle: float) -> int:
    pkgs = discover(py_root)
    repo = git_root(next(iter(pkgs.values())))
    print(f"watching {len(pkgs)} packages under {py_root} every {interval}s", flush=True)
    with psycopg.connect(dsn) as conn:                 # catch up on anything missed
        cmd_sync(conn, pkgs, list(pkgs), False, source="watch")
    with psycopg.connect(dsn) as conn:                 # a job that died with the last watcher
        with conn.cursor() as cur:
            cur.execute(sql.SQL("UPDATE {s}.jobs SET status = 'error', finished_at = now(), "
                                "result = '{{\"status\": \"error\", \"error\": \"watcher restarted\"}}' "
                                "WHERE status = 'running'").format(s=sql.Identifier(SCHEMA)))
        conn.commit()
    last = tree_signature(repo, pkgs)
    pending: dict[str, float] = {}
    child: subprocess.Popen | None = None
    last_promote = 0.0
    while True:
        time.sleep(interval)
        if time.monotonic() - last_promote >= PROMOTE_EVERY:
            last_promote = time.monotonic()
            try:
                promote_step(dsn)
            except (Exception, SystemExit) as e:
                print(f"promote: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        try:
            if child is not None and child.poll() is not None:
                child = None
            if child is None:
                with psycopg.connect(dsn) as conn:
                    jid = claim_job(conn)
                if jid is not None:
                    print(f"job {jid}: starting", flush=True)
                    child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                              "--dsn", dsn, "job", "run", str(jid)])
            new_pkgs = discover(py_root)
            if new_pkgs.keys() != pkgs.keys():
                pending.update({n: time.monotonic() for n in new_pkgs.keys() - pkgs.keys()})
                pkgs = new_pkgs
            sig = tree_signature(repo, pkgs)
            now = time.monotonic()
            for n in pkgs:
                if sig[n] != last.get(n):
                    pending[n] = now                   # (re)start its settle clock
            last = sig
            ready = [n for n, t in pending.items() if now - t >= settle]
            if ready:
                with psycopg.connect(dsn) as conn:
                    cmd_sync(conn, pkgs, sorted(ready), False, source="watch")
                for n in ready:
                    pending.pop(n, None)
                sys.stdout.flush()
        except Exception as e:                         # keep watching through DB/git hiccups
            print(f"watch: {type(e).__name__}: {e}", file=sys.stderr, flush=True)


def print_json(obj) -> int:
    import json
    print(json.dumps(obj, indent=2, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--py-root", type=Path, default=PY_ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    s = sub.add_parser("sync"); s.add_argument("packages", nargs="*"); s.add_argument("--dry-run", action="store_true")
    b = sub.add_parser("build"); b.add_argument("package"); b.add_argument("--out", type=Path, required=True)
    b.add_argument("--force", action="store_true"); b.add_argument("--version", dest="ref")
    v = sub.add_parser("verify"); v.add_argument("packages", nargs="*")
    l = sub.add_parser("ls"); l.add_argument("package")
    c = sub.add_parser("cat"); c.add_argument("package"); c.add_argument("rel_path")
    w = sub.add_parser("watch"); w.add_argument("--interval", type=float, default=3.0)
    w.add_argument("--settle", type=float, default=5.0, help="quiet seconds before a package syncs")
    vs = sub.add_parser("versions"); vs.add_argument("packages", nargs="*")
    vs.add_argument("--good", action="store_true")
    g = sub.add_parser("good"); g.add_argument("package"); g.add_argument("ref")
    g.add_argument("--unset", action="store_true"); g.add_argument("--note")
    r = sub.add_parser("restore"); r.add_argument("package"); r.add_argument("ref")
    r.add_argument("--apply", action="store_true")
    cf = sub.add_parser("config").add_subparsers(dest="ccmd", required=True)
    x = cf.add_parser("save"); x.add_argument("name"); x.add_argument("--good", action="store_true")
    x.add_argument("--note")
    x = cf.add_parser("import-tag"); x.add_argument("tag"); x.add_argument("--name")
    x.add_argument("--good", action="store_true"); x.add_argument("--note")
    cf.add_parser("ls")
    x = cf.add_parser("show"); x.add_argument("name")
    x = cf.add_parser("diff"); x.add_argument("a"); x.add_argument("b")
    x = cf.add_parser("good"); x.add_argument("name"); x.add_argument("--unset", action="store_true")
    x.add_argument("--note")
    x = cf.add_parser("build"); x.add_argument("name"); x.add_argument("--out", type=Path, required=True)
    x.add_argument("--force", action="store_true")
    x = cf.add_parser("restore"); x.add_argument("name"); x.add_argument("--apply", action="store_true")
    x = cf.add_parser("verify"); x.add_argument("name")
    x = sub.add_parser("retire", help="retire a package: kept as history, never tested/promoted")
    x.add_argument("package"); x.add_argument("--note"); x.add_argument("--undo", action="store_true")
    te = sub.add_parser("testenv").add_subparsers(dest="tcmd", required=True)
    te.add_parser("build"); te.add_parser("ls"); te.add_parser("path")
    jb = sub.add_parser("job").add_subparsers(dest="jcmd", required=True)
    x = jb.add_parser("verify", help="queue: install + test PKG@REF … (others at current) or --config")
    x.add_argument("packages", nargs="*"); x.add_argument("--config")
    x = jb.add_parser("restore", help="queue: put PKG@REF … or --config back into the tree")
    x.add_argument("packages", nargs="*"); x.add_argument("--config")
    x = jb.add_parser("show"); x.add_argument("id", type=int)
    x = jb.add_parser("ls"); x.add_argument("--limit", type=int, default=20)
    x = jb.add_parser("run", help="(internal) execute a claimed job"); x.add_argument("id", type=int)
    x = sub.add_parser("analyze", help="duplicates, collisions, dependency problems (JSON)")
    x.add_argument("--config"); x.add_argument("--job", type=int)
    x = sub.add_parser("dups", help="duplicate-function groups; --callers: who calls which copy")
    x.add_argument("--config"); x.add_argument("--job", type=int)
    x.add_argument("--callers", action="store_true")
    x = sub.add_parser("trace", help="function 'module:qualname' or --test NODEID")
    x.add_argument("function", nargs="?"); x.add_argument("--test"); x.add_argument("--job", type=int)
    x.add_argument("--config"); x.add_argument("--depth", type=int, default=4)
    x = sub.add_parser("symbols", help="find symbols by name (SQL LIKE)")
    x.add_argument("name"); x.add_argument("--kind"); x.add_argument("--config")
    x = sub.add_parser("publish", help="commit+push the source trees (pkg_publish step; OFF "
                                       "unless pkg_publish.toml enabled=true)")
    grp = x.add_mutually_exclusive_group()
    grp.add_argument("--promotion", type=int); grp.add_argument("--config")
    x.add_argument("--config-file", dest="config_file")
    x.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "job" and a.jcmd == "run":
        return run_job(a.dsn, a.id)
    if a.cmd == "watch":
        return cmd_watch(a.dsn, a.py_root, a.interval, a.settle)
    if a.cmd == "publish":
        import pkg_publish
        with psycopg.connect(a.dsn) as conn:
            return print_json(pkg_publish.publish(conn, promotion_id=a.promotion, config=a.config,
                                                  config_path=a.config_file, dry_run=a.dry_run))

    pkgs = discover(a.py_root)

    def pick(names):
        unknown = [n for n in names if n not in pkgs]
        if unknown:
            raise SystemExit(f"unknown package(s): {', '.join(unknown)}; have: {', '.join(pkgs)}")
        return names or list(pkgs)

    with psycopg.connect(a.dsn) as conn:
        with conn.cursor() as cur:
            ensure_schema(cur)
            pkg_graph.ensure_graph_schema(cur)
        conn.commit()
        backfill_versions(conn)
        relabel_micro(conn)
        if a.cmd == "status":
            return cmd_status(conn, pkgs)
        if a.cmd == "sync":
            return cmd_sync(conn, pkgs, pick(a.packages), a.dry_run)
        if a.cmd == "build":
            pick([a.package]); return cmd_build(conn, a.package, a.out, a.force, a.ref)
        if a.cmd == "verify":
            return cmd_verify(conn, pkgs, pick(a.packages))
        if a.cmd == "ls":
            return cmd_ls(conn, a.package)
        if a.cmd == "cat":
            return cmd_cat(conn, a.package, a.rel_path)
        if a.cmd == "versions":
            return cmd_versions(conn, pick(a.packages), a.good)
        if a.cmd == "good":
            pick([a.package]); return cmd_good(conn, a.package, a.ref, a.unset, a.note)
        if a.cmd == "restore":
            pick([a.package])
            with conn.cursor() as cur:
                vid = resolve_version(cur, a.package, a.ref)
            return restore_packages(conn, pkgs, {a.package: vid}, a.apply, f"{a.package}@{a.ref}")
        if a.cmd == "analyze":
            return print_json(api_analysis(conn, a.config, a.job))
        if a.cmd == "dups":
            return print_json(api_duplicates(conn, a.config, a.job, a.callers))
        if a.cmd == "trace":
            return print_json(api_trace(conn, a.function, a.test, a.job, a.config, a.depth))
        if a.cmd == "symbols":
            return print_json(api_symbols(conn, a.name, a.kind, a.config))
        if a.cmd == "retire":
            return print_json(api_retire(conn, a.package, a.note, a.undo))
        if a.cmd == "testenv":
            if a.tcmd == "path":
                print(TESTENV); return 0
            if a.tcmd == "ls":
                return print_json(api_testenvs(conn))
            with conn.cursor() as cur:
                members = current_members(cur)
            return 0 if testenv_build(conn, members, "cli") else 1
        if a.cmd == "job":
            if a.jcmd in ("verify", "restore"):
                spec = ",".join(a.packages)
                return print_json(api_request(conn, a.jcmd, config=a.config, packages=spec,
                                              requested_by="cli"))
            if a.jcmd == "show":
                return print_json(api_job(conn, a.id))
            if a.jcmd == "ls":
                return print_json(api_jobs(conn, a.limit))
        if a.cmd == "config":
            if a.ccmd == "save":
                return save_config(conn, pkgs, a.name, a.good, a.note)
            if a.ccmd == "import-tag":
                return cmd_import_tag(conn, pkgs, a.tag, a.name, a.good, a.note)
            if a.ccmd == "ls":
                return cmd_config_ls(conn)
            if a.ccmd == "show":
                return cmd_config_show(conn, a.name)
            if a.ccmd == "diff":
                return cmd_config_diff(conn, a.a, a.b)
            if a.ccmd == "good":
                return cmd_config_good(conn, a.name, a.unset, a.note)
            if a.ccmd == "verify":
                return print_json(api_request(conn, "verify", config=a.name, requested_by="cli"))
            if a.ccmd == "build":
                return cmd_config_build(conn, pkgs, a.name, a.out, a.force)
            if a.ccmd == "restore":
                with conn.cursor() as cur:
                    plan = {p: v for p, v in config_members(cur, a.name).items() if p in pkgs}
                return restore_packages(conn, pkgs, plan, a.apply, f"config {a.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
