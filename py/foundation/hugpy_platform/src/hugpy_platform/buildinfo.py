"""Build identity: exactly which build is this process running?

Every hugpy process (central, worker agents, the CLI) answers the same three
questions from the same code: *which version* (the git-derived workspace
version every in-tree distribution shares), *which commit* (sha + dirty flag,
either read live from the editable checkout or parsed out of the version's
local segment for a wheel install), and *from where* (editable source path or
site-packages). Central surfaces the compact form on ``/health`` and the full
document on ``/build``; a worker folds the compact form into its heartbeat.

Discipline, same as ``environment_report``: stdlib only, nothing here raises,
every git call is a child process with a 5 s timeout, and an answer that is
not known is ``None`` — never a guess. Versions come from ``importlib.metadata``
(what is actually importable from this interpreter), never from a literal.

Version strings. setuptools-scm emits ``X.Y.Z`` at a tag and
``X.Y.Z.devN+g<sha>[.d<yyyymmdd>]`` between tags (``.d<date>`` means the tree
was dirty when built; a checkout without tags reads ``0.0.1.devN+unknown.g<sha>``
and one built without git metadata at all is ``0.0.0+unknown``). ``parse_version``
pulls the sha and dirty flag back out of that local segment.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time

#: The 13 in-tree distributions, in partition order (PARTITION.md).
WORKSPACE_DISTRIBUTIONS: "tuple[str, ...]" = (
    "hugpy-platform",
    "hugpy-control",
    "hugpy-storage",
    "hugpy-engine",
    "hugpy-media",
    "hugpy-video",
    "hugpy-oracle",
    "hugpy-fleet",
    "hugpy-curation",
    "hugpy-ops",
    "hugpy-discord",
    "hugpy-server",
    "hugpy",
)

#: Distributions that ride alongside the workspace but are versioned on their
#: own: the portable agent client, the two independent services, and the
#: ``abstract_hugpy_dev`` compatibility shell (importlib normalises the
#: underscore name to this dash form).
EXTERNAL_DISTRIBUTIONS: "tuple[str, ...]" = (
    "hugpy-agent",
    "abstract-identity",
    "abstract-toolserver",
    "abstract-hugpy-dev",
)

#: Which distribution's version names the build, first installed wins. Fleet
#: first because the worker is the process most often asked; the server for
#: central; the platform is always present when anything hugpy is.
IDENTITY_PRECEDENCE: "tuple[str, ...]" = ("hugpy-fleet", "hugpy-server", "hugpy-platform")

#: Every git child process gets this long and no longer.
GIT_TIMEOUT_S: float = 5.0

_SHA_COMPONENT = re.compile(r"^g([0-9a-fA-F]{7,40})$")
_DIRTY_COMPONENT = re.compile(r"^d\d{8}$")


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def normalize_dist(name: str) -> str:
    """PEP 503 normalisation (``abstract_hugpy_dev`` -> ``abstract-hugpy-dev``)."""
    return re.sub(r"[-_.]+", "-", (name or "").strip()).lower()


def _direct_url(dist: object) -> "dict | None":
    """The PEP 610 ``direct_url.json`` of an installed distribution, or None
    (an index install has none)."""
    try:
        text = dist.read_text("direct_url.json")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return None
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001
        return None
    return parsed if isinstance(parsed, dict) else None


def _file_url_to_path(url: str) -> "str | None":
    if not isinstance(url, str) or not url.startswith("file:"):
        return None
    try:
        from urllib.parse import unquote, urlparse
        parsed = urlparse(url)
        path = unquote(parsed.path or "")
        if os.name == "nt" and re.match(r"^/[A-Za-z]:", path):
            path = path[1:]
        return path or None
    except Exception:  # noqa: BLE001
        return None


def distribution_record(name: str) -> "dict | None":
    """``{"name", "version", "editable", "source"}`` for an installed
    distribution; None when it is not installed (or metadata is unreadable).

    ``editable`` and ``source`` come from ``direct_url.json``: an editable
    install records ``dir_info.editable`` and a ``file://`` url naming the
    project directory; a wheel from an index has neither."""
    try:
        from importlib.metadata import PackageNotFoundError, distribution
    except Exception:  # noqa: BLE001
        return None
    try:
        dist = distribution(name)
    except PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001
        return None
    try:
        version = str(dist.version) if dist.version is not None else None
    except Exception:  # noqa: BLE001
        version = None
    editable = False
    source: "str | None" = None
    direct = _direct_url(dist)
    if direct:
        dir_info = direct.get("dir_info")
        if isinstance(dir_info, dict):
            editable = bool(dir_info.get("editable"))
        source = _file_url_to_path(str(direct.get("url") or ""))
    return {"name": normalize_dist(name), "version": version,
            "editable": editable, "source": source}


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------


def _git(args: "list[str]", cwd: str) -> "str | None":
    """stdout of ``git -C cwd <args>`` stripped, or None on any failure
    (git missing, not a repo, timeout, non-zero exit)."""
    try:
        proc = subprocess.run(
            ["git", "-C", cwd, *args], capture_output=True, text=True,
            timeout=GIT_TIMEOUT_S, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except Exception:  # noqa: BLE001 — FileNotFoundError, TimeoutExpired, ...
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip()


def git_identity(path: "str | None") -> "dict | None":
    """``{"sha", "short", "branch", "dirty", "describe", "tag", "root"}`` for
    the git repository containing ``path``; None when there is none (or git
    is not installed). ``branch`` is None on a detached HEAD, ``tag`` is the
    nearest reachable tag (None in an untagged repository)."""
    if not path:
        return None
    try:
        cwd = os.path.abspath(os.path.expanduser(str(path)))
        if not os.path.isdir(cwd):
            cwd = os.path.dirname(cwd)
        if not os.path.isdir(cwd):
            return None
    except Exception:  # noqa: BLE001
        return None
    root = _git(["rev-parse", "--show-toplevel"], cwd)
    if not root:
        return None
    sha = _git(["rev-parse", "HEAD"], cwd) or None
    short = _git(["rev-parse", "--short", "HEAD"], cwd) or None
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd) or None
    if branch == "HEAD":
        branch = None
    # Tracked modifications only: the same question ``git describe --dirty``
    # answers, and what setuptools-scm stamps as ``.d<date>``.
    status = _git(["status", "--porcelain", "--untracked-files=no"], cwd)
    dirty = bool(status) if status is not None else None
    describe = _git(["describe", "--tags", "--always", "--dirty"], cwd) or None
    tag = _git(["describe", "--tags", "--abbrev=0"], cwd) or None
    return {"sha": sha, "short": short, "branch": branch, "dirty": dirty,
            "describe": describe, "tag": tag, "root": root}


# ---------------------------------------------------------------------------
# Version strings
# ---------------------------------------------------------------------------


def parse_version(version: "str | None") -> "dict":
    """``{"sha", "dirty", "unknown"}`` recovered from a setuptools-scm version.

    ``sha`` is the ``g<sha>`` local component (abbreviated, as stamped) or
    None; ``dirty`` is True when a ``.d<yyyymmdd>`` component is present, False
    when a sha was stamped without one, None when nothing is known;
    ``unknown`` is True when the local segment says ``unknown`` — the build
    had no git metadata (``fallback_version``) or no tag to measure from."""
    out: "dict" = {"sha": None, "dirty": None, "unknown": False}
    if not version or "+" not in str(version):
        return out
    local = str(version).split("+", 1)[1]
    for component in local.split("."):
        match = _SHA_COMPONENT.match(component)
        if match and out["sha"] is None:
            out["sha"] = match.group(1).lower()
        elif _DIRTY_COMPONENT.match(component):
            out["dirty"] = True
        elif component == "unknown":
            out["unknown"] = True
    if out["sha"] is not None and out["dirty"] is None:
        out["dirty"] = False
    return out


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def _identity_record() -> "dict | None":
    for name in IDENTITY_PRECEDENCE:
        record = distribution_record(name)
        if record is not None:
            return record
    return None


def _identity_from_record(record: "dict | None") -> "dict":
    identity: "dict" = {"version": None, "sha": None, "dirty": None,
                        "editable": False, "source": None, "distribution": None}
    if not record:
        return identity
    identity["version"] = record.get("version")
    identity["editable"] = bool(record.get("editable"))
    identity["source"] = record.get("source")
    identity["distribution"] = record.get("name")
    git = git_identity(record.get("source")) if identity["editable"] else None
    if git and git.get("sha"):
        identity["sha"] = git.get("sha")
        identity["dirty"] = git.get("dirty")
    else:
        parsed = parse_version(identity["version"])
        identity["sha"] = parsed["sha"]
        identity["dirty"] = parsed["dirty"]
    return identity


def build_identity() -> "dict":
    """The compact identity that rides heartbeats and ``/health``:
    ``{"version", "sha", "dirty", "editable", "source", "distribution"}``.

    The version is the first installed of ``IDENTITY_PRECEDENCE``. For an
    editable install the sha and dirty flag are read live from the checkout
    (what is *running*, including uncommitted edits); for a wheel they are
    what setuptools-scm stamped into the version. Never raises."""
    try:
        return _identity_from_record(_identity_record())
    except Exception as exc:  # noqa: BLE001
        return {"version": None, "sha": None, "dirty": None, "editable": False,
                "source": None, "distribution": None,
                "error": f"{type(exc).__name__}: {exc}"}


def lockstep(distributions: "dict") -> "dict":
    """``{"ok", "versions": {version: [names]}}`` over the INSTALLED workspace
    distributions: ok when they all share one version (an empty install is
    trivially ok — there is nothing to disagree)."""
    versions: "dict[str, list[str]]" = {}
    for name in WORKSPACE_DISTRIBUTIONS:
        record = distributions.get(name)
        if not record:
            continue
        versions.setdefault(str(record.get("version")), []).append(name)
    return {"ok": len(versions) <= 1, "versions": versions}


def build_info() -> "dict":
    """The full build document served by ``GET /build``. Never raises."""
    try:
        distributions = {name: distribution_record(name)
                         for name in WORKSPACE_DISTRIBUTIONS + EXTERNAL_DISTRIBUTIONS}
        identity = _identity_from_record(_identity_record())
        workspace = None
        for name in WORKSPACE_DISTRIBUTIONS:
            record = distributions.get(name)
            if record and record.get("editable") and record.get("source"):
                workspace = git_identity(record["source"])
                if workspace:
                    break
        try:
            hostname = socket.gethostname()
        except Exception:  # noqa: BLE001
            hostname = None
        return {
            "identity": identity,
            "distributions": distributions,
            "lockstep": lockstep(distributions),
            "workspace": workspace,
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "hostname": hostname,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    except Exception as exc:  # noqa: BLE001
        return {"identity": build_identity(), "distributions": {}, "lockstep":
                {"ok": False, "versions": {}}, "workspace": None,
                "python": sys.version.split()[0], "executable": sys.executable,
                "hostname": None,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "error": f"{type(exc).__name__}: {exc}"}


def identity_line(identity: "dict | None" = None) -> str:
    """One human line: ``hugpy-fleet 0.2.1.dev3+g952c2d8 (editable /path, dirty)``."""
    try:
        ident = identity if identity is not None else build_identity()
        name = ident.get("distribution") or "hugpy"
        version = ident.get("version") or "unknown"
        notes: "list[str]" = []
        if ident.get("editable"):
            notes.append(f"editable {ident.get('source') or '?'}")
        elif ident.get("sha") and ident["sha"] not in str(version):
            notes.append(f"g{ident['sha'][:7]}")
        if ident.get("dirty"):
            notes.append("dirty")
        line = f"{name} {version}"
        return f"{line} ({', '.join(notes)})" if notes else line
    except Exception:  # noqa: BLE001
        return "hugpy unknown"


def main(argv: "list[str] | None" = None) -> int:
    """``python -m hugpy_platform.buildinfo [--json]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--json" in args:
        print(json.dumps(build_info(), indent=2, sort_keys=True))
    else:
        print(identity_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
