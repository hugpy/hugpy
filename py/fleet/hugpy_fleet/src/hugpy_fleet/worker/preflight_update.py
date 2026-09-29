"""Converge an installed worker venv before importing the worker package.

This file is copied beside ``worker.env`` by the installer and run as a
systemd ``ExecStartPre``. It intentionally uses only the standard library so
it can repair a worker whose installed Hugpy modules fail during import.
"""
from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

_BACKOFF_S = 300
_VERSION_RE = re.compile(r"^[A-Za-z0-9.!+_-]+$")


def _get(url: str, timeout: float = 20.0) -> bytes:
    req = Request(url, headers={"Accept": "application/json, text/plain"})
    with urlopen(req, timeout=timeout) as response:
        return response.read()


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_names() -> set[str]:
    found = set()
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            found.add(_norm(name))
    return found


def _state_path() -> Path:
    home = Path(os.environ.get("HUGPY_HOME") or (Path.home() / ".hugpy"))
    return home / "state" / "preflight-package-update.json"


def _recently_attempted(path: Path, target: str) -> bool:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        return (state.get("target") == target and
                time.time() - float(state.get("at", 0)) < _BACKOFF_S)
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _record_attempt(path: Path, target: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"target": target, "at": time.time()}),
                   encoding="utf-8")
    os.replace(tmp, path)


def _trusted_host(url: str) -> str | None:
    parts = urlsplit(url)
    if parts.scheme != "http" or not parts.hostname:
        return None
    if parts.hostname in ("127.0.0.1", "localhost", "::1"):
        return None
    return parts.netloc


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--central", required=True)
    args = parser.parse_args()
    central = args.central.rstrip("/")
    try:
        info = json.loads(_get(central + "/api/llm/workers/required-version"))
        target = str(info.get("required_pkg_version") or "").strip()
        if not target or not _VERSION_RE.fullmatch(target):
            print("worker preflight: central has no valid package pin; skipping")
            return 0
        constraints_url = central + "/api/llm/workers/constraints.txt"
        lines = [line.strip() for line in _get(constraints_url, timeout=30.0)
                 .decode("utf-8", "replace").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    except Exception as exc:  # central may be starting or temporarily unreachable
        print(f"worker preflight: cannot read central package pin: {exc}; skipping")
        return 0

    installed = _installed_names()
    pins: list[str] = []
    seen = set()
    for line in lines:
        name, sep, _version = line.partition("==")
        normalized = _norm(name.strip())
        if sep and (normalized in installed or normalized == "hugpy-fleet"):
            if normalized not in seen:
                pins.append(f"{name.strip()}=={target}")
                seen.add(normalized)
    if "hugpy-fleet" not in seen:
        pins.append(f"hugpy-fleet=={target}")
    if not lines:
        print("worker preflight: central has no lockstep constraints; skipping")
        return 0

    try:
        if all(metadata.version(spec.split("==", 1)[0]) == target for spec in pins):
            return 0
    except metadata.PackageNotFoundError:
        pass
    except Exception as exc:
        print(f"worker preflight: cannot inspect installed versions: {exc}; skipping")
        return 0

    state_path = _state_path()
    if _recently_attempted(state_path, target):
        print(f"worker preflight: update to {target} was recently attempted; backing off")
        return 0
    _record_attempt(state_path, target)

    cmd = [sys.executable, "-m", "pip", "install", "-U",
           "--upgrade-strategy", "only-if-needed"]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt",
                                     delete=False) as fh:
        fh.write("\n".join(lines) + "\n")
        constraint_file = fh.name
    cmd += ["-c", constraint_file]
    explicit_index = os.environ.get("WORKER_PKG_INDEX", "").strip()
    pkg_index = str(info.get("pkg_index_url") or "").strip()
    if explicit_index:
        cmd += ["--index-url", explicit_index]
        index_urls = [explicit_index]
    else:
        index_urls = []
        if pkg_index:
            cmd += ["--extra-index-url", pkg_index]
            index_urls.append(pkg_index)
    for index_url in index_urls:
        trusted = _trusted_host(index_url)
        if trusted:
            cmd += ["--trusted-host", trusted]
    cmd += pins
    print(f"worker preflight: converging {len(pins)} installed Hugpy package(s) to {target}")
    try:
        result = subprocess.run(cmd, check=False, timeout=600)
    except Exception as exc:
        print(f"worker preflight: pip update failed: {exc}; worker will still try to start")
        return 0
    finally:
        try:
            os.unlink(constraint_file)
        except OSError:
            pass
    if result.returncode:
        print(f"worker preflight: pip update exited {result.returncode}; worker will still try to start")
    else:
        try:
            state_path.unlink()
        except OSError:
            pass
        print(f"worker preflight: packages installed at {target}; starting worker")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
