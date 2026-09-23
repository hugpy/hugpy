"""hugpy-drift-check — prove that the checkout, the installed packages, the
fleet and PyPI are all on ONE build; fail loudly when they are not.

The standing rule this enforces (the same one station-drift-check.sh holds for
the station .deb):

    *** NOTHING THE SYSTEM RUNS MAY BE HAND-AUTHORED SOMEWHERE THAT DOES NOT
        SHIP. ***

Convention does not hold it. This does, mechanically. Since the 2026-09-22
versioning change every in-tree hugpy-* distribution carries the workspace's
git identity in its version (``X.Y.Z.devN+g<sha>[.dirty]``), so "same build"
is a question git and ``importlib.metadata`` can answer without trusting
anyone's word.

Sections, each producing rows ``{section, subject, status, detail}`` with
``status`` one of ``ok`` / ``drift`` / ``error`` / ``info``:

  A  checkout   the workspace HEAD vs ``origin/<branch>`` (ahead/behind = drift)
                and the working tree (dirty = drift unless ``--allow-dirty``)
  B  installed  every workspace distribution in THIS interpreter: one lockstep
                version, every editable source at the workspace HEAD, no
                ``+unknown`` build (= no git identity at build time)
  C  fleet      central's ``/api/health`` build vs each ``/api/llm/workers`` row:
                ``environment_digest.build`` when the worker reports one, else
                ``pkg_version`` vs ``required_pkg_version``; a monolith
                ``0.1.x`` worker or one without a build identity is drift
  D  pypi       the newest git tag vs PyPI's latest release for each workspace
                distribution (tag ahead = unpublished release; PyPI ahead =
                checkout behind release; absent from PyPI = info)

Exit codes: 0 everything ok (info rows allowed), 1 drift, 2 only errors (the
check could not verify — central unreachable, git missing, PyPI down).

Genuine runtime STATE is deliberately not checked: model stores, job stores,
logs, env files. Only AUTHORED and DERIVED artifacts are drift.

Stdlib only. ``hugpy_platform.buildinfo`` is imported lazily when present and
its absence degrades to plain ``importlib.metadata`` + ``git`` lookups.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import os
import re
import shutil
import string
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Optional

OK, DRIFT, ERROR, INFO = "ok", "drift", "error", "info"

SECTIONS = {"A": "checkout", "B": "installed", "C": "fleet", "D": "pypi"}
_SECTION_BY_NAME = {v: k for k, v in SECTIONS.items()}

# Mirrors hugpy_platform.buildinfo.WORKSPACE_DISTRIBUTIONS; used only when
# buildinfo is not importable (an older hugpy-platform, or none at all).
FALLBACK_DISTRIBUTIONS = (
    "hugpy-platform", "hugpy-control", "hugpy-storage", "hugpy-engine",
    "hugpy-media", "hugpy-video", "hugpy-oracle", "hugpy-fleet",
    "hugpy-curation", "hugpy-ops", "hugpy-discord", "hugpy-server", "hugpy",
)

DEFAULT_CENTRAL = "http://127.0.0.1:7002"
CENTRAL_ENV_VARS = ("HUGPY_BASE_URL", "HUGPY_CENTRAL", "HUGPY_URL", "WORKER_CENTRAL_URL")
TOKEN_ENV_VARS = ("HUGPY_TOKEN", "HUGPY_API_KEY", "HUGPY_KEY")
PYPI_JSON = "https://pypi.org/pypi/{name}/json"
HTTP_TIMEOUT = 10.0

# The retired monolith (abstract_hugpy_dev) published 0.1.<n> releases; a
# worker still reporting one of those runs code that never carried a git
# identity. Workspace versions between tags look like 0.1.devN+g<sha>, which
# this deliberately does not match.
_MONOLITH_VERSION = re.compile(r"^0\.1\.\d+(\.post\d+)?$")
# setuptools-scm local segments: ``+g<sha>[.d<date>]`` (tagged history) or
# ``+unknown.g<sha>`` (no tag reachable — the sha is still authoritative);
# ``0.0.0+unknown`` alone means the build had no git metadata at all.
_LOCAL_SHA = re.compile(r"[+.]g([0-9a-fA-F]{7,40})(?=\.|$)")
_UNITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drift_units")

_SUMMARY_WORD = {0: "IN SYNC", 1: "DRIFT", 2: "UNVERIFIED"}


# --------------------------------------------------------------------------- #
# report model
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    section: str
    subject: str
    status: str
    detail: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    rows: list[Row] = field(default_factory=list)
    central: Optional[str] = None
    workspace: Optional[str] = None
    build: Optional[dict] = None
    generated_at: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())

    def add(self, section: str, subject: str, status: str, detail: str = "") -> Row:
        row = Row(section, subject, status, detail)
        self.rows.append(row)
        return row

    def extend(self, rows: Iterable[Row]) -> None:
        self.rows.extend(rows)

    @property
    def counts(self) -> dict[str, int]:
        out = {OK: 0, DRIFT: 0, ERROR: 0, INFO: 0}
        for r in self.rows:
            out[r.status] = out.get(r.status, 0) + 1
        return out

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def exit_code(self) -> int:
        """0 = nothing drifted and nothing failed to verify; 1 = drift
        (even when errors are also present — drift is the stronger verdict);
        2 = no drift found but at least one section could not verify."""
        counts = self.counts
        if counts[DRIFT]:
            return 1
        if counts[ERROR]:
            return 2
        return 0

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "verdict": _SUMMARY_WORD[self.exit_code],
            "generated_at": self.generated_at,
            "central": self.central,
            "workspace": self.workspace,
            "build": self.build,
            "counts": self.counts,
            "rows": [r.as_dict() for r in self.rows],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=False)

    def table(self) -> str:
        """Aligned text: one row per line, a summary line last."""
        header = ("SECTION", "SUBJECT", "STATUS", "DETAIL")
        cells = [(f"{r.section} {SECTIONS.get(r.section, '')}".strip(), r.subject,
                  r.status.upper() if r.status == DRIFT else r.status, r.detail)
                 for r in self.rows]
        widths = [len(h) for h in header]
        for row in cells:
            for i in range(3):
                widths[i] = max(widths[i], len(row[i]))
        lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(header)).rstrip()]
        lines.append("  ".join("-" * widths[i] for i in range(4)).rstrip())
        for row in cells:
            lines.append("  ".join(c.ljust(widths[i]) if i < 3 else c
                                   for i, c in enumerate(row)).rstrip())
        c = self.counts
        lines.append("")
        lines.append(f"drift-check: {c[OK]} ok, {c[DRIFT]} drift, {c[ERROR]} error, "
                     f"{c[INFO]} info  ->  {_SUMMARY_WORD[self.exit_code]}"
                     f"  (exit {self.exit_code})")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# lazy buildinfo + fallbacks
# --------------------------------------------------------------------------- #
def _buildinfo():
    """``hugpy_platform.buildinfo`` when importable, else None. Never raises."""
    try:
        return importlib.import_module("hugpy_platform.buildinfo")
    except Exception:  # noqa: BLE001 — older platform, or no platform at all
        return None


def workspace_distributions() -> tuple[str, ...]:
    bi = _buildinfo()
    names = getattr(bi, "WORKSPACE_DISTRIBUTIONS", None) if bi else None
    return tuple(names) if names else FALLBACK_DISTRIBUTIONS


def _record_fallback(name: str) -> Optional[dict]:
    """``{name, version, editable, source}`` from importlib.metadata alone
    (PEP 610 direct_url.json says whether the install is editable and from
    where)."""
    from importlib import metadata

    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001
        return None
    editable, source = False, None
    try:
        raw = dist.read_text("direct_url.json")
        if raw:
            direct = json.loads(raw)
            editable = bool((direct.get("dir_info") or {}).get("editable"))
            url = direct.get("url") or ""
            if url.startswith("file://"):
                source = url[len("file://"):]
    except Exception:  # noqa: BLE001
        pass
    return {"name": name, "version": dist.version, "editable": editable, "source": source}


def distribution_record(name: str) -> Optional[dict]:
    bi = _buildinfo()
    fn = getattr(bi, "distribution_record", None) if bi else None
    if fn:
        try:
            rec = fn(name)
            if rec is not None:
                return dict(rec)
            return None
        except Exception:  # noqa: BLE001 — degrade to metadata
            pass
    return _record_fallback(name)


def local_build_identity() -> Optional[dict]:
    """This interpreter's build identity per buildinfo, else None."""
    bi = _buildinfo()
    fn = getattr(bi, "build_identity", None) if bi else None
    if not fn:
        return None
    try:
        return dict(fn())
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# git (a thin, never-raising wrapper)
# --------------------------------------------------------------------------- #
def git(path: str, *args: str, timeout: float = 30.0, strip: bool = True) -> Optional[str]:
    """stdout of ``git -C path args`` (stripped unless ``strip=False``), or
    None on any failure."""
    if not shutil.which("git"):
        return None
    try:
        proc = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", path, *args],
            capture_output=True, text=True, timeout=timeout, check=False,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() if strip else proc.stdout


def git_root(path: str) -> Optional[str]:
    return git(path, "rev-parse", "--show-toplevel") or None


def git_head(path: str) -> Optional[str]:
    return git(path, "rev-parse", "HEAD") or None


def git_branch(path: str) -> Optional[str]:
    """Current branch name, None when detached."""
    out = git(path, "symbolic-ref", "--short", "-q", "HEAD")
    return out or None


def git_dirty_files(path: str) -> Optional[list[str]]:
    """Modified TRACKED paths (untracked files are not drift: they do not
    ship and cannot be run by anything that ships)."""
    # Not stripped: porcelain lines are "XY path" and X may be a space.
    out = git(path, "status", "--porcelain", "--untracked-files=no", strip=False)
    if out is None:
        return None
    return [line[3:] for line in out.splitlines() if len(line) > 3]


def git_ahead_behind(path: str, upstream: str) -> Optional[tuple[int, int]]:
    """``(ahead, behind)`` of HEAD relative to ``upstream``; None when the
    upstream ref does not exist."""
    out = git(path, "rev-list", "--left-right", "--count", f"{upstream}...HEAD")
    if not out:
        return None
    try:
        behind, ahead = (int(x) for x in out.split())
    except ValueError:
        return None
    return ahead, behind


def git_latest_tag(path: str) -> Optional[str]:
    return git(path, "describe", "--tags", "--abbrev=0") or None


def _short(sha: Optional[str], n: int = 12) -> str:
    return (sha or "")[:n] or "?"


def _same_sha(a: Optional[str], b: Optional[str]) -> bool:
    """Full-vs-abbreviated tolerant equality."""
    if not a or not b:
        return False
    a, b = a.lower(), b.lower()
    return a.startswith(b) or b.startswith(a)


def locate_workspace(explicit: Optional[str] = None) -> Optional[str]:
    """``--workspace``, else the editable source buildinfo reports, else the
    git root of the current directory. None when no checkout is findable."""
    if explicit:
        root = git_root(explicit)
        return root or os.path.abspath(explicit)
    ident = local_build_identity()
    source = (ident or {}).get("source")
    if ident and ident.get("editable") and source and os.path.isdir(source):
        root = git_root(source)
        if root:
            return root
    # any editable workspace distribution's source
    for name in workspace_distributions():
        rec = distribution_record(name)
        if rec and rec.get("editable") and rec.get("source") and os.path.isdir(rec["source"]):
            root = git_root(rec["source"])
            if root:
                return root
    return git_root(os.getcwd())


# --------------------------------------------------------------------------- #
# HTTP (injectable: tests monkeypatch ``fetch_json``)
# --------------------------------------------------------------------------- #
def fetch_json(url: str, token: Optional[str] = None, timeout: float = HTTP_TIMEOUT) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "hugpy-drift-check"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict, timeout: float = HTTP_TIMEOUT) -> int:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "hugpy-drift-check"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status)


def central_url(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.rstrip("/")
    for name in CENTRAL_ENV_VARS:
        val = os.environ.get(name)
        if val:
            return val.rstrip("/")
    return DEFAULT_CENTRAL


def central_token(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit.strip() or None
    for name in TOKEN_ENV_VARS:
        val = (os.environ.get(name) or "").strip()
        if val:
            return val
    return None


# --------------------------------------------------------------------------- #
# version parsing (stdlib, PEP 440 enough for release/pre/post/dev ordering)
# --------------------------------------------------------------------------- #
_VERSION_RE = re.compile(
    r"^v?(?P<release>\d+(?:\.\d+)*)"
    r"(?:(?P<pre_l>a|b|rc)(?P<pre_n>\d+))?"
    r"(?:\.post(?P<post>\d+))?"
    r"(?:\.dev(?P<dev>\d+))?"
    r"(?:\+(?P<local>.*))?$"
)
_PRE_ORDER = {"a": 0, "b": 1, "rc": 2}


def version_key(version: str) -> Optional[tuple]:
    """Sortable key; None when the string is not a version we understand."""
    m = _VERSION_RE.match((version or "").strip())
    if not m:
        return None
    release = tuple(int(x) for x in m.group("release").split("."))
    release = release + (0,) * (6 - len(release))
    pre = (_PRE_ORDER[m.group("pre_l")], int(m.group("pre_n"))) if m.group("pre_l") else (3, 0)
    post = int(m.group("post")) if m.group("post") else -1
    dev = int(m.group("dev")) if m.group("dev") else float("inf")
    return (release, pre, post, dev)


def version_cmp(a: str, b: str) -> Optional[int]:
    ka, kb = version_key(a), version_key(b)
    if ka is None or kb is None:
        return None
    return (ka > kb) - (ka < kb)


def strip_v(tag: Optional[str]) -> Optional[str]:
    if not tag:
        return None
    return tag[1:] if tag[:1] in ("v", "V") and tag[1:2].isdigit() else tag


# --------------------------------------------------------------------------- #
# section A — checkout
# --------------------------------------------------------------------------- #
def check_checkout(workspace: Optional[str], allow_dirty: bool = False,
                   fetch: bool = False) -> list[Row]:
    S = "A"
    rows: list[Row] = []
    if not workspace or not os.path.isdir(workspace):
        rows.append(Row(S, "workspace", INFO,
                        "no checkout found (non-editable install?) — pass --workspace PATH"))
        return rows
    if not shutil.which("git"):
        rows.append(Row(S, "workspace", ERROR, "git is not on PATH; cannot verify the checkout"))
        return rows
    head = git_head(workspace)
    if not head:
        rows.append(Row(S, "workspace", ERROR, f"{workspace} is not a git checkout"))
        return rows
    branch = git_branch(workspace)
    rows.append(Row(S, "HEAD", INFO,
                    f"{_short(head)} on {branch or 'detached HEAD'}  ({workspace})"))

    dirty = git_dirty_files(workspace)
    if dirty is None:
        rows.append(Row(S, "tree", ERROR, "git status failed"))
    elif not dirty:
        rows.append(Row(S, "tree", OK, "clean (no modified tracked files)"))
    else:
        sample = ", ".join(dirty[:4]) + (" …" if len(dirty) > 4 else "")
        if allow_dirty:
            rows.append(Row(S, "tree", INFO, f"{len(dirty)} modified tracked file(s), allowed: {sample}"))
        else:
            rows.append(Row(S, "tree", DRIFT,
                            f"{len(dirty)} modified tracked file(s) not committed: {sample}"))

    if not branch:
        rows.append(Row(S, "origin", INFO, "detached HEAD; no upstream to compare"))
        return rows
    upstream = f"origin/{branch}"
    if fetch:
        if git(workspace, "fetch", "--quiet", "origin", timeout=120) is None:
            rows.append(Row(S, upstream, ERROR, "git fetch origin failed"))
            return rows
    ab = git_ahead_behind(workspace, upstream)
    if ab is None:
        rows.append(Row(S, upstream, INFO,
                        f"no {upstream} ref (branch never pushed, or no origin remote)"))
        return rows
    ahead, behind = ab
    fetched = "after fetch" if fetch else "vs last fetch"
    if ahead == 0 and behind == 0:
        rows.append(Row(S, upstream, OK, f"in sync at {_short(head)} ({fetched})"))
    else:
        parts = []
        if ahead:
            parts.append(f"{ahead} unpushed commit(s)")
        if behind:
            parts.append(f"{behind} commit(s) behind origin")
        rows.append(Row(S, upstream, DRIFT, ", ".join(parts) + f" ({fetched})"))
    return rows


def _stamped_sha(version: str) -> Optional[str]:
    """The ``g<sha>`` setuptools-scm stamped into ``version``'s local segment
    (lower-case, abbreviated as stamped), or None: a release exactly at a tag
    carries none, and so does a build without git metadata (``+unknown``)."""
    m = _LOCAL_SHA.search(version or "")
    return m.group(1).lower() if m else None


def _is_release(version: str) -> bool:
    """``X.Y.Z`` with no dev/local segment: a build exactly at a tag."""
    return bool(version) and "+" not in version and ".dev" not in version


# --------------------------------------------------------------------------- #
# section B — installed
# --------------------------------------------------------------------------- #
def check_installed(workspace: Optional[str]) -> list[Row]:
    S = "B"
    rows: list[Row] = []
    ws_head = git_head(workspace) if workspace else None
    versions: dict[str, list[str]] = {}
    installed = 0
    for name in workspace_distributions():
        rec = distribution_record(name)
        if rec is None:
            rows.append(Row(S, name, INFO, "not installed"))
            continue
        installed += 1
        version = str(rec.get("version") or "")
        versions.setdefault(version, []).append(name)
        editable = bool(rec.get("editable"))
        source = rec.get("source")
        where = f"editable from {source}" if editable else "installed (non-editable)"
        if not _stamped_sha(version) and not _is_release(version):
            rows.append(Row(S, name, DRIFT,
                            f"{version or '?'}: no git identity at build time ({where})"))
            continue
        if editable:
            src_head = git_head(source) if source and os.path.isdir(source) else None
            if not src_head:
                rows.append(Row(S, name, ERROR, f"{version}: editable source {source} has no git HEAD"))
            elif ws_head and not _same_sha(src_head, ws_head):
                rows.append(Row(S, name, DRIFT,
                                f"{version}: editable source HEAD {_short(src_head)} != "
                                f"workspace HEAD {_short(ws_head)}"))
            elif not ws_head:
                rows.append(Row(S, name, INFO, f"{version}: {where} (no workspace HEAD to compare)"))
            else:
                note = ""
                stamped = _stamped_sha(version)
                if stamped and not _same_sha(stamped, ws_head):
                    note = f"; metadata built at g{stamped[:7]}, reinstall to refresh"
                rows.append(Row(S, name, OK, f"{version}: source at workspace HEAD{note}"))
        else:
            stamped = _stamped_sha(version)
            if stamped and ws_head and not _same_sha(stamped, ws_head):
                rows.append(Row(S, name, DRIFT,
                                f"{version}: built from g{stamped[:7]}, workspace HEAD is "
                                f"{_short(ws_head)}"))
            else:
                rows.append(Row(S, name, OK, f"{version}: {where}"))

    if installed == 0:
        rows.append(Row(S, "lockstep", INFO, "no workspace distribution installed in this interpreter"))
    elif len(versions) == 1:
        (only,) = versions
        rows.append(Row(S, "lockstep", OK, f"{installed} distribution(s) at {only}"))
    else:
        desc = "; ".join(f"{v or '?'} ({', '.join(names)})"
                         for v, names in sorted(versions.items(), key=lambda kv: -len(kv[1])))
        rows.append(Row(S, "lockstep", DRIFT, f"mixed versions in one interpreter: {desc}"))
    return rows


# --------------------------------------------------------------------------- #
# section C — fleet
# --------------------------------------------------------------------------- #
def _build_desc(build: Optional[dict]) -> str:
    if not build:
        return "?"
    v = build.get("version") or "?"
    sha = build.get("sha")
    tail = f" g{_short(sha, 7)}" if sha else ""
    return f"{v}{tail}{' dirty' if build.get('dirty') else ''}"


def _same_build(a: Optional[dict], b: Optional[dict]) -> bool:
    if not a or not b:
        return False
    if (a.get("version") or "") != (b.get("version") or ""):
        return False
    sa, sb = a.get("sha"), b.get("sha")
    if sa and sb:
        return _same_sha(sa, sb)
    return True


def check_fleet(central: str, token: Optional[str] = None,
                local_build: Optional[dict] = None) -> list[Row]:
    S = "C"
    rows: list[Row] = []
    try:
        health = fetch_json(f"{central}/api/health", token)
    except Exception as exc:  # noqa: BLE001 — unreachable is a finding, not a crash
        rows.append(Row(S, "central", ERROR, f"{central}/api/health unreachable: {_exc(exc)}"))
        return rows
    if not isinstance(health, dict):
        rows.append(Row(S, "central", ERROR, f"{central}/api/health returned no object"))
        return rows

    central_build = health.get("build") if isinstance(health.get("build"), dict) else None
    if central_build is None:
        rows.append(Row(S, "central", ERROR,
                        "no build identity in /api/health — central predates buildinfo; "
                        "its build cannot be verified"))
    elif local_build:
        if _same_build(central_build, local_build):
            rows.append(Row(S, "central", OK, f"{_build_desc(central_build)} = this interpreter"))
        else:
            rows.append(Row(S, "central", DRIFT,
                            f"central {_build_desc(central_build)} != this interpreter "
                            f"{_build_desc(local_build)}"))
    else:
        rows.append(Row(S, "central", INFO,
                        f"{_build_desc(central_build)} (no local buildinfo to compare against)"))

    try:
        workers = fetch_json(f"{central}/api/llm/workers", token)
    except Exception as exc:  # noqa: BLE001
        rows.append(Row(S, "workers", ERROR, f"{central}/api/llm/workers unreachable: {_exc(exc)}"))
        return rows
    if isinstance(workers, dict):
        workers = workers.get("workers") or workers.get("data") or []
    if not isinstance(workers, list):
        rows.append(Row(S, "workers", ERROR, "/api/llm/workers returned no list"))
        return rows
    if not workers:
        rows.append(Row(S, "workers", INFO, "central has no enrolled workers"))
        return rows

    reference = central_build or local_build
    for w in workers:
        if not isinstance(w, dict):
            continue
        rows.append(_worker_row(w, reference, central_build is not None))
    return rows


def _worker_row(w: dict, reference: Optional[dict], central_has_build: bool) -> Row:
    S = "C"
    name = f"worker {w.get('name') or '?'}"
    status = w.get("status") or "?"
    tag = f"[{status}{', unreachable' if w.get('unreachable') else ''}]"
    digest = w.get("environment_digest") if isinstance(w.get("environment_digest"), dict) else {}
    build = digest.get("build") if isinstance(digest.get("build"), dict) else None
    pkg_version = w.get("pkg_version")

    if build:
        if reference is None:
            return Row(S, name, INFO, f"{_build_desc(build)} {tag}; nothing to compare against")
        if _same_build(build, reference):
            return Row(S, name, OK, f"{_build_desc(build)} {tag}")
        against = "central" if central_has_build else "this interpreter"
        return Row(S, name, DRIFT,
                   f"{_build_desc(build)} != {against} {_build_desc(reference)} {tag}")

    if not pkg_version or _MONOLITH_VERSION.match(str(pkg_version)):
        return Row(S, name, DRIFT,
                   f"monolith / no build identity (pkg_version={pkg_version or 'none'}) {tag}")

    required = w.get("required_pkg_version")
    version_ok = w.get("version_ok")
    if version_ok is None or required in (None, ""):
        return Row(S, name, INFO,
                   f"pkg_version {pkg_version}, central pins nothing (required_pkg_version=None) {tag}")
    if version_ok is True or str(required) == str(pkg_version):
        return Row(S, name, OK, f"pkg_version {pkg_version} = required {required} {tag}")
    return Row(S, name, DRIFT, f"pkg_version {pkg_version} != required {required} {tag}")


def _exc(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    reason = getattr(exc, "reason", None)
    return f"{type(exc).__name__}: {reason or exc}"


# --------------------------------------------------------------------------- #
# section D — pypi
# --------------------------------------------------------------------------- #
def pypi_latest(name: str) -> Optional[str]:
    """PyPI's latest version for ``name``; None when the project is absent."""
    try:
        data = fetch_json(PYPI_JSON.format(name=name), None, HTTP_TIMEOUT)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    return (data.get("info") or {}).get("version")


def check_pypi(workspace: Optional[str]) -> list[Row]:
    S = "D"
    rows: list[Row] = []
    tag = strip_v(git_latest_tag(workspace)) if workspace else None
    if not tag:
        rows.append(Row(S, "tag", INFO,
                        "no release tag in the checkout (git describe --tags); "
                        "PyPI versions listed for reference only"
                        if workspace else "no checkout; cannot read release tags"))
    else:
        rows.append(Row(S, "tag", INFO, f"newest release tag {tag}"))

    for name in workspace_distributions():
        try:
            latest = pypi_latest(name)
        except Exception as exc:  # noqa: BLE001
            rows.append(Row(S, name, ERROR, f"pypi lookup failed: {_exc(exc)}"))
            continue
        if latest is None:
            rows.append(Row(S, name, INFO, "not on PyPI"))
            continue
        if not tag:
            rows.append(Row(S, name, INFO, f"PyPI {latest} (no local tag to compare)"))
            continue
        cmp = version_cmp(tag, latest)
        if cmp is None:
            rows.append(Row(S, name, ERROR, f"cannot compare tag {tag} with PyPI {latest}"))
        elif cmp == 0:
            rows.append(Row(S, name, OK, f"tag {tag} = PyPI {latest}"))
        elif cmp > 0:
            rows.append(Row(S, name, DRIFT, f"unpublished release: tag {tag} > PyPI {latest}"))
        else:
            rows.append(Row(S, name, DRIFT, f"checkout behind release: PyPI {latest} > tag {tag}"))
    return rows


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def parse_sections(spec: Optional[str]) -> list[str]:
    if not spec or spec.strip().lower() == "all":
        return list(SECTIONS)
    out: list[str] = []
    for part in re.split(r"[,\s]+", spec.strip()):
        if not part:
            continue
        key = part.upper() if len(part) == 1 else _SECTION_BY_NAME.get(part.lower())
        if key not in SECTIONS:
            raise ValueError(f"unknown section {part!r}; choose from "
                             f"{', '.join(f'{k}={v}' for k, v in SECTIONS.items())}")
        if key not in out:
            out.append(key)
    return out


def run(sections: Iterable[str] = "ABCD", *, workspace: Optional[str] = None,
        central: Optional[str] = None, token: Optional[str] = None,
        allow_dirty: bool = False, fetch: bool = False, pypi: bool = True) -> Report:
    wanted = [s.upper() for s in sections]
    ws = locate_workspace(workspace)
    report = Report(central=central_url(central), workspace=ws, build=local_build_identity())
    if "A" in wanted:
        report.extend(check_checkout(ws, allow_dirty=allow_dirty, fetch=fetch))
    if "B" in wanted:
        report.extend(check_installed(ws))
    if "C" in wanted:
        report.extend(check_fleet(report.central, central_token(token), report.build))
    if "D" in wanted:
        if pypi:
            report.extend(check_pypi(ws))
        else:
            report.add("D", "pypi", INFO, "skipped (--no-pypi)")
    return report


# --------------------------------------------------------------------------- #
# systemd timer
# --------------------------------------------------------------------------- #
def console_script_path() -> str:
    """Absolute ``hugpy-drift-check`` next to this interpreter, else a
    ``python -m hugpy_ops.drift`` fallback (both absolute)."""
    bindir = os.path.dirname(os.path.abspath(sys.executable))
    for cand in ("hugpy-drift-check", "hugpy-drift-check.exe"):
        p = os.path.join(bindir, cand)
        if os.path.isfile(p):
            return p
    return f"{os.path.abspath(sys.executable)} -m hugpy_ops.drift"


def _read_unit_template(name: str) -> str:
    with open(os.path.join(_UNITS_DIR, name), encoding="utf-8") as fh:
        return fh.read()


def render_units(*, exec_start: str, on_calendar: str = "daily",
                 environment: Iterable[str] = (), env_file: Optional[str] = None,
                 user_mode: bool = False, working_directory: Optional[str] = None) -> dict[str, str]:
    """``{unit filename: text}`` for the service and the timer."""
    env_lines = [f"Environment={line}" for line in environment]
    if env_file:
        env_lines.append(f"EnvironmentFile=-{os.path.abspath(env_file)}")
    service = string.Template(_read_unit_template("hugpy-drift-check.service")).substitute(
        exec_start=exec_start,
        environment="\n".join(env_lines) if env_lines else "# (no Environment= lines given)",
        wanted_by="default.target" if user_mode else "multi-user.target",
        working_directory=f"WorkingDirectory={working_directory}\n" if working_directory else "",
    )
    timer = string.Template(_read_unit_template("hugpy-drift-check.timer")).substitute(
        on_calendar=on_calendar,
        wanted_by="timers.target",
    )
    return {"hugpy-drift-check.service": service, "hugpy-drift-check.timer": timer}


def install_timer(args: argparse.Namespace, runner: Callable = subprocess.run) -> int:
    exec_start = [console_script_path(), "--quiet", "--fetch"]
    environment: list[str] = []
    if args.central:
        exec_start += ["--central", args.central]
    if args.notify_url:
        exec_start += ["--notify-url", args.notify_url]
    if args.workspace:
        exec_start += ["--workspace", os.path.abspath(args.workspace)]
    if args.allow_dirty:
        exec_start.append("--allow-dirty")
    if args.no_pypi:
        exec_start.append("--no-pypi")
    chosen = parse_sections(args.sections)
    if chosen != list(SECTIONS):
        exec_start += ["--sections", ",".join(chosen)]
    if args.token:
        environment.append(f"HUGPY_TOKEN={args.token}")
    units = render_units(exec_start=" ".join(exec_start), on_calendar=args.on_calendar or "daily",
                         environment=environment, env_file=args.env_file, user_mode=args.user,
                         working_directory=os.path.abspath(args.workspace) if args.workspace else None)
    if args.user:
        unit_dir = os.path.join(os.path.expanduser("~"), ".config", "systemd", "user")
        ctl = ["systemctl", "--user"]
    else:
        unit_dir = "/etc/systemd/system"
        ctl = ["systemctl"]
    commands = [ctl + ["daemon-reload"], ctl + ["enable", "--now", "hugpy-drift-check.timer"]]

    if args.dry_run:
        for fname, text in units.items():
            print(f"# ---- {os.path.join(unit_dir, fname)} ----")
            print(text.rstrip("\n"))
            print()
        print("# would run:")
        for cmd in commands:
            print("#   " + " ".join(cmd))
        return 0

    try:
        os.makedirs(unit_dir, exist_ok=True)
        for fname, text in units.items():
            with open(os.path.join(unit_dir, fname), "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"wrote {os.path.join(unit_dir, fname)}")
    except OSError as exc:
        print(f"hugpy-drift-check: cannot write units to {unit_dir}: {exc}"
              f"{'' if args.user else ' (run as root, or pass --user)'}", file=sys.stderr)
        return 2
    for cmd in commands:
        proc = runner(cmd)
        rc = getattr(proc, "returncode", proc if isinstance(proc, int) else 0)
        if rc:
            print(f"hugpy-drift-check: {' '.join(cmd)} failed (exit {rc})", file=sys.stderr)
            return 2
    print("hugpy-drift-check.timer enabled; "
          f"inspect with: {' '.join(ctl)} list-timers hugpy-drift-check*")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hugpy-drift-check",
        description="prove checkout, installed packages, fleet and PyPI are one build; "
                    "exit 0 in sync, 1 drift, 2 could not verify",
        epilog="sections: A=checkout  B=installed  C=fleet  D=pypi",
    )
    p.add_argument("--sections", default="A,B,C,D",
                   help="comma-separated letters or names (default: all)")
    p.add_argument("--json", action="store_true", help="print the JSON report instead of the table")
    p.add_argument("--central", help=f"central base URL (default: HUGPY_BASE_URL or {DEFAULT_CENTRAL})")
    p.add_argument("--token", help="bearer token for central (default: HUGPY_TOKEN / HUGPY_API_KEY)")
    p.add_argument("--workspace", help="workspace checkout (default: the editable install's source, "
                                       "else the git root of the current directory)")
    p.add_argument("--allow-dirty", action="store_true", help="a dirty tree is info, not drift")
    p.add_argument("--fetch", action="store_true", help="git fetch origin before comparing")
    p.add_argument("--no-pypi", action="store_true", help="skip section D (no network to pypi.org)")
    p.add_argument("--quiet", action="store_true", help="print the table only when not in sync")
    p.add_argument("--notify-url", help="POST the JSON report here when not in sync")
    t = p.add_argument_group("timer", "install a systemd timer that runs this check")
    t.add_argument("--install-timer", action="store_true",
                   help="write hugpy-drift-check.service/.timer and enable the timer")
    t.add_argument("--user", action="store_true", help="user units (~/.config/systemd/user) + systemctl --user")
    t.add_argument("--on-calendar", default="daily", help="systemd OnCalendar= (default: daily)")
    t.add_argument("--env-file", help="EnvironmentFile= for the service (central URL, token)")
    t.add_argument("--dry-run", action="store_true", help="print the units, write nothing")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        sections = parse_sections(args.sections)
    except ValueError as exc:
        print(f"hugpy-drift-check: {exc}", file=sys.stderr)
        return 2
    if args.install_timer:
        return install_timer(args)

    try:
        report = run(sections, workspace=args.workspace, central=args.central, token=args.token,
                     allow_dirty=args.allow_dirty, fetch=args.fetch, pypi=not args.no_pypi)
    except KeyboardInterrupt:
        return 130

    if args.json:
        print(report.to_json())
    elif not (args.quiet and report.ok):
        print(report.table())

    if args.notify_url and not report.ok:
        try:
            status = post_json(args.notify_url, report.as_dict())
            print(f"hugpy-drift-check: notified {args.notify_url} (HTTP {status})", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — notification failure never masks the verdict
            print(f"hugpy-drift-check: notify {args.notify_url} failed: {_exc(exc)}", file=sys.stderr)
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
