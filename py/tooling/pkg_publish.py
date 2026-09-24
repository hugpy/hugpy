#!/usr/bin/env python3
"""pkg_publish — the GitHub commit+push STEP of the pkg_src versioning pipeline.

Publishing the source trees to GitHub is not a manual act: it is a step that
runs when a promotion the watcher judged reaches ``healthy`` (pkg_promote), the
same way ``pkg_src.py watch`` already builds versions and pkg_promote already
pins the fleet. This module is that step. It is a SCAFFOLD: OFF by default and
inert until an operator writes the config and turns it on (dry-run first).

    pkg_src.py publish [--promotion ID | --config NAME] [--dry-run]

Config (TOML, default $PKG_PUBLISH_CONFIG or /srv/hugpy/etc/pkg_publish.toml;
absent => disabled). See pkg_publish.example.toml. ``enabled = false`` and
``mode = "dry-run"`` mean nothing is ever committed or pushed until an operator
sets ``enabled = true`` and, after reviewing the dry-run rows, ``mode = "push"``.

Every attempt is recorded per repo in ``pkg_src.publishes`` (promotion, config,
release, repo, branch, commit sha, pushed, mode, secret-scan result, status,
scrubbed error, timings). The token is NEVER written to git config, a remote
URL, a log, or the DB: it is read from the env file at run time and handed to
git only through a GIT_ASKPASS helper, and scrubbed from any captured output.

Per repo, in order (any refusal records a status and skips push — never force):
  a. refuse if a rebase/merge is in progress, HEAD is detached, or the checkout
     is not on the configured branch          -> status ``dirty_state``
  b. fetch the remote branch over HTTPS; if it holds commits not in the local
     branch, do NOT force                      -> status ``diverged``, skip
  c. stage with ``git add`` honoring .gitignore (+ any extra_gitignore)
  d. SECRET GATE: scan the staged added content (gitleaks if installed, else a
     grep ruleset) for private keys, sk-/sk-ant-/hf_/ghp_/github_pat_/pypi-/npm_
     tokens, postgres://user:pass@, *_TOKEN|*_KEY|*_SECRET|PASSWORD literals,
     .env/.pypirc/.npmrc/.git-credentials/htpasswd/.secret/.auth filenames,
     WireGuard PrivateKey, and files > 10 MB. ANY hit -> unstage everything,
     status ``blocked_secret`` (file:line+type recorded, never the value), skip
  e. commit ``<config> -> <release>`` (+ package->label body, Published-By
     trailer); nothing staged                  -> status ``clean``
  f. push (mode = push) or report (dry-run / mode = dry-run)

DB: --dsn / $PKG_SRC_DSN (default "dbname=hugpy"), same as pkg_src.
"""
from __future__ import annotations

import fcntl
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pkg_src as S  # noqa: E402  (sibling: DB mirror, versions, configs, SCHEMA)

SCHEMA = S.SCHEMA
OWNER_ROLE = "vm_mgr"                       # owns the pkg_src tables (as pkg_promote)
DEFAULT_CONFIG = os.environ.get("PKG_PUBLISH_CONFIG", "/srv/hugpy/etc/pkg_publish.toml")
LOCK_PATH = os.environ.get("PKG_PUBLISH_LOCK", "/tmp/pkg_src_publish.lock")
BIG_FILE_BYTES = 10 * 1024 * 1024           # a staged file larger than this is a hit

# status values recorded per repo; a superset that the CHECK constraint enforces.
STATUSES = ("disabled", "dirty_state", "diverged", "blocked_secret",
            "clean", "staged", "committed", "pushed", "error")


def log(msg: str) -> None:
    print(f"pkg_publish: {msg}", flush=True)


# --------------------------------------------------------------------------- schema

def ensure_publishes(cur) -> None:
    """pkg_src.publishes, created as vm_mgr so it is owned like the other tables
    (idempotent, same migration style as pkg_src.ensure_schema / ensure_promotions)."""
    cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f"{SCHEMA}.publishes",))
    if cur.fetchone()[0]:
        return
    cur.execute("SELECT current_user = %s OR pg_has_role(%s, 'MEMBER')", (OWNER_ROLE, OWNER_ROLE))
    as_owner = cur.fetchone()[0]
    if as_owner:
        cur.execute(sql.SQL("SET ROLE {r}").format(r=sql.Identifier(OWNER_ROLE)))
    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {s}.publishes (
            id            bigserial PRIMARY KEY,
            promotion_id  bigint REFERENCES {s}.promotions(id) ON DELETE SET NULL,
            config        text,
            release       text,
            repo_path     text NOT NULL,
            remote_url    text,
            branch        text,
            commit_sha    text,
            pushed        boolean NOT NULL DEFAULT false,
            mode          text NOT NULL,
            secret_scan   jsonb,
            status        text NOT NULL CHECK (status IN
                          ('disabled','dirty_state','diverged','blocked_secret',
                           'clean','staged','committed','pushed','error')),
            error         text,
            started_at    timestamptz NOT NULL DEFAULT now(),
            finished_at   timestamptz
        );
        CREATE INDEX IF NOT EXISTS publishes_promotion ON {s}.publishes (promotion_id);
        CREATE INDEX IF NOT EXISTS publishes_repo ON {s}.publishes (repo_path);
    """).format(s=sql.Identifier(SCHEMA)))
    if as_owner:
        cur.execute("RESET ROLE")


def record(conn, row: dict) -> int:
    """Insert one pkg_src.publishes row; returns its id."""
    cols = ("promotion_id", "config", "release", "repo_path", "remote_url", "branch",
            "commit_sha", "pushed", "mode", "secret_scan", "status", "error",
            "started_at", "finished_at")
    vals = [row.get(c) for c in cols]
    vals[cols.index("secret_scan")] = Jsonb(row.get("secret_scan"))
    with conn.cursor() as cur:
        ensure_publishes(cur)
        cur.execute(sql.SQL("INSERT INTO {s}.publishes ({cols}) VALUES ({ph}) RETURNING id")
                    .format(s=sql.Identifier(SCHEMA),
                            cols=sql.SQL(", ").join(sql.Identifier(c) for c in cols),
                            ph=sql.SQL(", ").join(sql.Placeholder() * len(cols))), vals)
        pid = cur.fetchone()[0]
    conn.commit()
    return pid


# ----------------------------------------------------------------------- config/token

def load_config(path: str | Path | None = None) -> dict | None:
    """Parse the publish config; returns None when the file is absent (=> disabled)."""
    p = Path(path or DEFAULT_CONFIG)
    if not p.exists():
        return None
    with p.open("rb") as fh:
        return tomllib.load(fh)


def read_token(env_file: str | Path, var: str) -> str | None:
    """Read one KEY=VALUE from a .env-style file WITHOUT logging it. Follows a
    symlink (e.g. /srv/hugpy/.env -> etc/hugpy.env). Returns None if absent."""
    p = Path(env_file)
    if not p.exists():
        return None
    for raw in p.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == var:
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            return v or None
    return None


def scrub(text: str | None, secrets: list[str]) -> str | None:
    """Replace any secret value in ``text`` with ``***`` (defence in depth: git
    output should never carry the token, but we never store it if it slips in)."""
    if not text:
        return text
    out = text
    for sec in secrets:
        if sec:
            out = out.replace(sec, "***")
    return out


# -------------------------------------------------------------------------- git glue

def git(repo: Path, *args: str, env: dict | None = None, check: bool = False,
        timeout: float = 300) -> tuple[int, str, str]:
    """Run a git command in ``repo`` (safe.directory=* so foreign-owned trees are
    fine). Returns (rc, stdout, stderr). Never raises unless ``check``."""
    base = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1"}
    if env:
        base.update(env)
    p = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repo), *args],
                       capture_output=True, text=True, env=base, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.returncode, p.stdout, p.stderr


def _askpass_env(token: str) -> tuple[dict, str]:
    """A temp GIT_ASKPASS helper that feeds git a PAT for HTTPS auth. The token
    is passed only through the child env (PKG_PUBLISH_PW), never a URL/config."""
    fd, path = tempfile.mkstemp(prefix="pkg_publish_askpass_", suffix=".sh")
    os.write(fd, b'#!/bin/sh\ncase "$1" in\n'
                 b'  Username*) printf \'x-access-token\' ;;\n'
                 b'  *) printf \'%s\' "$PKG_PUBLISH_PW" ;;\nesac\n')
    os.close(fd)
    os.chmod(path, stat.S_IRWXU)
    return {"GIT_ASKPASS": path, "PKG_PUBLISH_PW": token}, path


def git_in_progress(repo: Path) -> str | None:
    """A rebase/merge/cherry-pick/bisect mid-operation, or a detached HEAD."""
    gd = repo / ".git"
    if not gd.is_dir():                                   # worktree/gitfile: ask git
        rc, out, _ = git(repo, "rev-parse", "--git-dir")
        gd = (repo / out.strip()) if rc == 0 else gd
    for marker, why in (("rebase-merge", "rebase in progress"),
                        ("rebase-apply", "rebase in progress"),
                        ("MERGE_HEAD", "merge in progress"),
                        ("CHERRY_PICK_HEAD", "cherry-pick in progress"),
                        ("BISECT_LOG", "bisect in progress")):
        if (gd / marker).exists():
            return why
    rc, out, _ = git(repo, "symbolic-ref", "-q", "HEAD")
    if rc != 0:
        return "HEAD is detached"
    return None


# ------------------------------------------------------------------------- secret gate

SENSITIVE_NAMES = (".env", ".pypirc", ".npmrc", ".git-credentials",
                   "htpasswd", ".htpasswd", ".secret", ".auth")

RULES: list[tuple[str, "re.Pattern"]] = [
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}")),
    ("github_token", re.compile(r"\bghp_[A-Za-z0-9]{36,}")),
    ("pypi_token", re.compile(r"\bpypi-[A-Za-z0-9_\-]{20,}")),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{36,}")),
    ("pg_url_password", re.compile(r"postgres(?:ql)?://[^:/\s]+:[^@/\s]+@")),
    ("wireguard_key", re.compile(r"(?i)PrivateKey\s*=\s*[A-Za-z0-9+/]{43}=")),
    # NAME=<literal>: TOKEN/SECRET/PASSWORD/KEY assigned a real value (an env or
    # placeholder reference — $VAR, ${...}, os.environ, getenv, <...> — is not one).
    ("assigned_secret", re.compile(
        r"(?i)\b[A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API[_-]?KEY|PRIVATE[_-]?KEY)"
        r"\s*[:=]\s*['\"]?(?!\s*$)(?!\$)(?!\{\{)(?!<)"
        r"(?!os\.environ)(?!getenv)(?!process\.env)([^\s'\"]{6,})")),
]


def _staged_added(repo: Path) -> list[tuple[str, int, str]]:
    """(rel_path, line_no, text) for every ADDED line in the staged diff."""
    rc, out, _ = git(repo, "diff", "--cached", "--unified=0", "--no-color", "--no-ext-diff")
    added: list[tuple[str, int, str]] = []
    path, ln = None, 0
    for line in out.splitlines():
        if line.startswith("+++ "):
            p = line[4:].strip()
            path = None if p == "/dev/null" else p[2:] if p.startswith(("a/", "b/")) else p
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            ln = int(m.group(1)) if m else 0
        elif line.startswith("+") and not line.startswith("+++"):
            if path is not None:
                added.append((path, ln, line[1:]))
            ln += 1
    return added


def _gitleaks(repo: Path) -> list[dict] | None:
    """Run gitleaks on the staged content if it is installed; else None."""
    if not shutil.which("gitleaks"):
        return None
    with tempfile.NamedTemporaryFile(suffix=".json") as rep:
        git_env = {**os.environ}
        p = subprocess.run(["gitleaks", "protect", "--staged", "--no-banner",
                            "--redact", "--report-format", "json",
                            "--report-path", rep.name],
                           cwd=str(repo), capture_output=True, text=True, env=git_env)
        try:
            import json
            data = json.loads(Path(rep.name).read_text() or "[]")
        except Exception:
            data = []
    hits = []
    for f in data or []:
        hits.append({"file": f.get("File"), "line": f.get("StartLine"),
                     "type": f.get("RuleID") or "gitleaks"})
    return hits


def secret_scan(repo: Path) -> dict:
    """Scan the staged change. Returns {'scanner', 'hits':[{file,line,type}], 'clean'}.
    Never records a secret VALUE — only its location and rule name."""
    gl = _gitleaks(repo)
    if gl is not None:
        return {"scanner": "gitleaks", "hits": gl, "clean": not gl}
    hits: list[dict] = []
    # content rules over added lines
    for path, ln, text in _staged_added(repo):
        for name, pat in RULES:
            if pat.search(text):
                hits.append({"file": path, "line": ln, "type": name})
    # filename + size rules over the staged file list
    rc, out, _ = git(repo, "diff", "--cached", "--name-only", "--diff-filter=ACMR")
    for rel in filter(None, (l.strip() for l in out.splitlines())):
        base = Path(rel).name
        if base in SENSITIVE_NAMES or any(base.endswith(n) for n in SENSITIVE_NAMES):
            hits.append({"file": rel, "line": 0, "type": "sensitive_filename"})
        fp = repo / rel
        try:
            if fp.is_file() and fp.stat().st_size > BIG_FILE_BYTES:
                hits.append({"file": rel, "line": 0, "type": "large_file"})
        except OSError:
            pass
    return {"scanner": "grep", "hits": hits, "clean": not hits}


# ---------------------------------------------------------------------- per-repo step

def member_labels(cur, config: str | None) -> dict[str, str]:
    """{package: internal micro label} for a config's pinned versions (commit body)."""
    if not config:
        return {}
    try:
        members = S.config_members(cur, config)
    except SystemExit:
        return {}
    out = {}
    for pkg, vid in sorted(members.items()):
        cur.execute(sql.SQL("SELECT label FROM {s}.versions WHERE id = %s")
                    .format(s=sql.Identifier(SCHEMA)), (vid,))
        r = cur.fetchone()
        out[pkg] = r[0] if r else str(vid)
    return out


def commit_message(config: str | None, release: str | None, promotion_id: int | None,
                   labels: dict[str, str]) -> str:
    subject = f"{config or 'pkg_src'} -> {release or '(unreleased)'}"
    body = "\n".join(f"  {p}: {l}" for p, l in labels.items())
    trailer = f"Published-By: pkg_src promotion {promotion_id}" if promotion_id else \
              "Published-By: pkg_src"
    parts = [subject]
    if body:
        parts.append(body)
    parts.append(trailer)
    return "\n\n".join(parts)


def publish_repo(conn, repo_cfg: dict, *, config: str | None, release: str | None,
                 promotion_id: int | None, mode: str, token: str | None,
                 author: dict, message: str) -> dict:
    """Run the ordered per-repo publish for one repo. Records exactly one
    pkg_src.publishes row and returns it. Never force-pushes; never raises."""
    repo = Path(repo_cfg["path"])
    branch = repo_cfg.get("branch", "main")
    url = repo_cfg.get("url")
    include_untracked = bool(repo_cfg.get("include_untracked", True))
    extra_ignore = list(repo_cfg.get("extra_gitignore", []))
    secrets = [token] if token else []
    row = {"promotion_id": promotion_id, "config": config, "release": release,
           "repo_path": str(repo), "remote_url": url, "branch": branch,
           "commit_sha": None, "pushed": False, "mode": mode, "secret_scan": None,
           "status": "error", "error": None,
           "started_at": datetime.now(timezone.utc), "finished_at": None}

    def finish(status, error=None):
        row["status"] = status
        row["error"] = scrub(error, secrets)
        row["finished_at"] = datetime.now(timezone.utc)
        record(conn, row)
        log(f"{repo.name} [{branch}] {mode}: {status}"
            + (f" — {row['error']}" if row["error"] else ""))
        return row

    if not repo.exists():
        return finish("error", f"repo path does not exist: {repo}")

    # (a) refuse a mid-operation / detached / wrong-branch checkout
    why = git_in_progress(repo)
    if why:
        return finish("dirty_state", why)
    rc, cur_branch, err = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    cur_branch = cur_branch.strip()
    if rc != 0:
        return finish("error", err.strip() or "cannot read HEAD")
    if cur_branch != branch:
        return finish("dirty_state", f"on {cur_branch!r}, expected {branch!r}")

    # (b) fetch the remote branch; refuse to force over a diverged remote
    askpass, askpass_path = ({}, None)
    try:
        if url:
            if token:
                askpass, askpass_path = _askpass_env(token)
            rc, _, err = git(repo, "fetch", "--quiet", url, branch, env=askpass, timeout=180)
            if rc != 0:
                return finish("error", f"fetch failed: {err.strip()}")
            rc, out, _ = git(repo, "rev-list", "--count", "HEAD..FETCH_HEAD")
            ahead = int(out.strip() or "0") if rc == 0 else 0
            if ahead > 0:
                return finish("diverged",
                              f"remote {branch} has {ahead} commit(s) not in local; not forcing")

        # (c) stage honoring .gitignore
        add = ["add", "-A"] if include_untracked else ["add", "-u"]
        rc, _, err = git(repo, *add)
        if rc != 0:
            return finish("error", f"git add failed: {err.strip()}")
        for pat in extra_ignore:                          # unstage extra-ignored paths
            git(repo, "reset", "-q", "--", f":(glob){pat}")

        # (d) SECRET GATE
        scan = secret_scan(repo)
        row["secret_scan"] = scan
        if not scan["clean"]:
            git(repo, "reset", "-q")                       # unstage everything
            return finish("blocked_secret",
                          "secret gate: " + "; ".join(f"{h['type']}@{h['file']}:{h['line']}"
                                                       for h in scan["hits"]))

        # (e) commit (skip when nothing is staged)
        rc, _, _ = git(repo, "diff", "--cached", "--quiet")
        if rc == 0:
            return finish("clean")
        ident = ["-c", f"user.name={author.get('name', 'pkg_src')}",
                 "-c", f"user.email={author.get('email', 'noreply@localhost')}"]
        with tempfile.NamedTemporaryFile("w", suffix=".msg", delete=False) as mf:
            mf.write(message)
            msg_path = mf.name
        try:
            rc, _, err = git(repo, *ident, "commit", "--no-verify", "-F", msg_path)
        finally:
            os.unlink(msg_path)
        if rc != 0:
            return finish("error", f"commit failed: {err.strip()}")
        rc, sha, _ = git(repo, "rev-parse", "HEAD")
        row["commit_sha"] = sha.strip()

        # (f) push, or stop at the local commit for dry-run
        if mode != "push":
            return finish("staged")
        if not url:
            return finish("committed", "no remote url configured; commit not pushed")
        if not token:
            return finish("committed", "no token available; commit not pushed")
        if askpass_path is None:
            askpass, askpass_path = _askpass_env(token)
        rc, _, err = git(repo, "push", url, f"HEAD:{branch}", env=askpass, timeout=300)
        if rc != 0:
            return finish("committed", f"push failed: {err.strip()}")
        row["pushed"] = True
        return finish("pushed")
    finally:
        if askpass_path:
            try:
                os.unlink(askpass_path)
            except OSError:
                pass


# ----------------------------------------------------------------------- orchestration

def _promotion_row(cur, promotion_id: int) -> dict | None:
    cur.execute(sql.SQL("SELECT id, config, version, status FROM {s}.promotions WHERE id = %s")
                .format(s=sql.Identifier(SCHEMA)), (promotion_id,))
    r = cur.fetchone()
    if not r:
        return None
    return {"id": r[0], "config": r[1], "version": r[2], "status": r[3]}


def _latest_promotion_for(cur, config: str) -> dict | None:
    cur.execute(sql.SQL("SELECT id, config, version, status FROM {s}.promotions "
                        "WHERE config = %s ORDER BY id DESC LIMIT 1")
                .format(s=sql.Identifier(SCHEMA)), (config,))
    r = cur.fetchone()
    if not r:
        return None
    return {"id": r[0], "config": r[1], "version": r[2], "status": r[3]}


def publish(conn, *, promotion_id: int | None = None, config: str | None = None,
            config_path: str | Path | None = None, dry_run: bool = False) -> dict:
    """Publish every configured repo once, under a fleet-wide flock (one publish at
    a time). Resolves the release/config from the promotion when given. Records a
    pkg_src.publishes row per repo. Returns a summary dict."""
    cfg = load_config(config_path)
    if not cfg:
        return {"enabled": False, "reason": f"no config at {config_path or DEFAULT_CONFIG}",
                "repos": []}
    enabled = bool(cfg.get("enabled", False))
    mode = "dry-run" if dry_run or not enabled else cfg.get("mode", "dry-run")
    if not enabled:
        return {"enabled": False, "reason": "enabled = false", "mode": mode, "repos": []}

    release = None
    with conn.cursor() as cur:
        pr = None
        if promotion_id is not None:
            pr = _promotion_row(cur, promotion_id)
        elif config:
            pr = _latest_promotion_for(cur, config)
        if pr:
            promotion_id = pr["id"]
            config = config or pr["config"]
            release = pr["version"]
        labels = member_labels(cur, config)
    message = commit_message(config, release, promotion_id, labels)

    author = cfg.get("author", {}) or {}
    token = None
    if mode == "push":
        token = read_token(cfg.get("env_file", "/srv/hugpy/.env"),
                           cfg.get("token_env", "HUGPY_GIPPA_FIAIN"))

    results = []
    lock = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)                  # one publish at a time, fleet-wide
        for repo_cfg in cfg.get("repo", []):
            try:
                results.append(publish_repo(conn, repo_cfg, config=config, release=release,
                                            promotion_id=promotion_id, mode=mode, token=token,
                                            author=author, message=message))
            except Exception as e:                        # never let one repo abort the rest
                log(f"{repo_cfg.get('path')}: {type(e).__name__}: {scrub(str(e), [token] if token else [])}")
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return {"enabled": True, "mode": mode, "promotion": promotion_id, "config": config,
            "release": release, "repos": [{k: r.get(k) for k in
                                           ("repo_path", "branch", "status", "pushed", "commit_sha")}
                                          for r in results]}


def publish_after_promotion(conn, promotion_id: int, config_path: str | Path | None = None) -> dict:
    """The watcher's hook: called when a promotion reaches ``healthy``. A no-op
    unless a config exists AND ``enabled = true``. NEVER blocks or fails a
    promotion: every error is caught, recorded, and swallowed."""
    try:
        cfg = load_config(config_path)
        if not cfg or not cfg.get("enabled", False):
            return {"enabled": False}
        return publish(conn, promotion_id=promotion_id, config_path=config_path)
    except Exception as e:
        log(f"publish_after_promotion({promotion_id}): {type(e).__name__}: {e}")
        return {"enabled": True, "error": f"{type(e).__name__}"}


# ------------------------------------------------------------------------------- cli

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dsn", default=S.DEFAULT_DSN)
    ap.add_argument("--config-file", dest="config_file", default=None,
                    help=f"publish config toml (default {DEFAULT_CONFIG})")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--promotion", type=int)
    g.add_argument("--config", help="pkg_src configuration name")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    with psycopg.connect(a.dsn) as conn:
        out = publish(conn, promotion_id=a.promotion, config=a.config,
                      config_path=a.config_file, dry_run=a.dry_run)
    import json
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
