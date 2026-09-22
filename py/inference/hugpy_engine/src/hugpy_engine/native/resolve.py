"""Locate the native llama.cpp executables — one resolver for the whole package.

Replaces the per-module ``LLAMA_CPP_DIR`` / ``LLAMA_SERVER_BIN`` defaults that
used to disagree (``/srv/abstractendeavors/...`` vs ``~/.local/share/hugpy/...``).
Resolution order, for each of ``llama-server`` / ``rpc-server`` / ``llama-cli``:

    1. explicit env override (e.g. ``LLAMA_SERVER_BIN``)
    2. an env-pinned engine dir (``HUGPY_ENGINE_DIR``/``LLAMA_CPP_DIR``, incl. a
       ``build/bin`` subdir, matching the cmake layout)
    3. the persisted install record (see :func:`persist_install`) — ``hugpy
       install-engine`` unpacks into a PER-USER data dir, so a worker service
       running under a different user/HOME could never re-find it and the
       operator reinstalled forever (k94, computron 2026-08-06); the install now
       records where it landed and resolution honours that record
    4. the per-user default engine dir (``paths.engine_dir()``)
    5. ``PATH`` (``shutil.which``, ``.exe`` on Windows)

Returns ``None`` when absent — callers fall back to the in-process runner and/or
tell the user to run ``hugpy install-engine``.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import List, Optional, Tuple

from hugpy_platform.platform_facade import IS_WINDOWS, env_value
from hugpy_platform.binaries import candidate_names, resolve_bin
from hugpy_platform.app_dirs import engine_dir

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Persisted install record                                                    #
# --------------------------------------------------------------------------- #
# Lives in the worker agent's own settings file — the /ops/config store
# (worker_agent.agent._settings_path: ``<WORKER_ID_FILE>.settings.json``) — so
# the box keeps ONE persisted config instead of growing another file format.
# The path derivation is duplicated here because importing the agent would drag
# the whole worker stack into every engine resolve.
_SETTINGS_KEY = "engine"


def _settings_path() -> str:
    from hugpy_platform import app_dirs as _hp
    return _hp.worker_id_file() + ".settings.json"


def persisted_engine() -> dict:
    """The recorded install (``{"dir": ..., "server_bin": ...}``), or ``{}``."""
    try:
        with open(_settings_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    rec = data.get(_SETTINGS_KEY) if isinstance(data, dict) else None
    return rec if isinstance(rec, dict) else {}


def persist_install(dir_path: str, server_path: str) -> Optional[str]:
    """Record where ``hugpy install-engine`` landed the engine.

    Read-modify-write of the settings file — unrelated keys (slot_count,
    residency, …) are preserved, and the write is the same atomic tmp+replace
    as the agent's ``_save_settings``. Returns the settings path, or ``None``
    when unwritable (non-fatal: the install still works for this user).
    """
    path = _settings_path()
    try:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        data[_SETTINGS_KEY] = {"dir": dir_path, "server_bin": server_path}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        os.replace(tmp, path)
        return path
    except OSError as exc:
        logger.warning("could not persist engine location to %s: %s", path, exc)
        return None


# A stale record is IGNORED, never fatal — but the reason must be loggable
# without spamming every 15s heartbeat resolve, so each stale value warns once
# per process.
_stale_warned: set = set()


def _warn_stale(field: str, value: str) -> None:
    if value in _stale_warned:
        return
    _stale_warned.add(value)
    logger.warning("persisted engine %s %r no longer exists — ignoring it "
                   "(re-run `hugpy install-engine` to refresh %s)",
                   field, value, _settings_path())


# --------------------------------------------------------------------------- #
# Binary resolution                                                           #
# --------------------------------------------------------------------------- #
def _dir_candidates(root: str) -> List[str]:
    # Prebuilt release zips unpack their binaries at the top level or under bin/;
    # a from-source cmake build puts them in build/bin/. Search all three.
    return [
        root,
        os.path.join(root, "bin"),
        os.path.join(root, "build", "bin"),
    ]


def _env_pinned_dir() -> Optional[str]:
    return env_value("HUGPY_ENGINE_DIR") or env_value("LLAMA_CPP_DIR")


def _find_under(root: str, name: str) -> Optional[str]:
    for d in _dir_candidates(root):
        for n in candidate_names(name):
            p = os.path.join(d, n)
            if os.path.isfile(p) and os.access(p, os.F_OK if IS_WINDOWS else os.X_OK):
                return p
    return None


def _resolve_info(name: str, *env_keys: str) -> Tuple[Optional[str], Optional[str]]:
    """``(path, source)`` with source in env|config|data_dir|path; ``(None,
    None)`` when the binary is absent everywhere."""
    for key in env_keys:
        override = env_value(key)
        if override and os.path.isfile(override):
            return override, "env"
    env_pinned = _env_pinned_dir()
    if env_pinned:
        found = _find_under(engine_dir(), name)     # engine_dir() == the pin
        if found:
            return found, "env"
    rec = persisted_engine()
    exact = rec.get("server_bin") if name == "llama-server" else None
    if isinstance(exact, str) and exact:
        if os.path.isfile(exact):
            return exact, "config"
        _warn_stale("server_bin", exact)
    pdir = rec.get("dir")
    if isinstance(pdir, str) and pdir and pdir != engine_dir():
        if os.path.isdir(pdir):
            found = _find_under(pdir, name)
            if found:
                return found, "config"
        else:
            _warn_stale("dir", pdir)
    if not env_pinned:
        found = _find_under(engine_dir(), name)
        if found:
            return found, "data_dir"
    found = resolve_bin(name)
    return (found, "path") if found else (None, None)


def _resolve(name: str, *env_keys: str) -> Optional[str]:
    return _resolve_info(name, *env_keys)[0]


def server_bin() -> Optional[str]:
    """Path to ``llama-server`` (the HTTP inference server), or ``None``."""
    return _resolve("llama-server", "LLAMA_SERVER_BIN")


def rpc_bin() -> Optional[str]:
    """Path to ``rpc-server`` (the cross-machine shard backend), or ``None``."""
    return _resolve("rpc-server", "WORKER_RPC_BIN", "LLAMA_RPC_BIN")


def cli_bin() -> Optional[str]:
    """Path to ``llama-cli``, or ``None``."""
    return _resolve("llama-cli", "LLAMA_CLI_BIN")


def have_native_engine() -> bool:
    return server_bin() is not None


# --------------------------------------------------------------------------- #
# Shared-library path for spawned native binaries                             #
# --------------------------------------------------------------------------- #
# A from-source ``llama-server`` links against sibling ``.so`` files (libllama,
# libggml, libggml-cuda …) that live NEXT to the binary — not in a system lib
# dir. When the agent spawns a slot child (or a native --mmproj/--rpc server)
# the child must find those on its loader path or it dies with
# "libggml.so: cannot open shared object file". On ae (2026-07-06) this was
# patched by hand as a unit-level LD_LIBRARY_PATH; deriving it from the engine
# dir in code makes the fix travel with the package instead.
def managed_engine_root(bin_path: str) -> Optional[str]:
    """The hugpy-managed engine root containing ``bin_path`` — the env-pinned
    dir, the persisted install dir, or the per-user default — or ``None`` for
    a system/PATH binary (whose libs the loader already resolves)."""
    real = os.path.realpath(bin_path)
    roots = [engine_dir()]
    pdir = persisted_engine().get("dir")
    if isinstance(pdir, str) and pdir:
        roots.append(pdir)
    for root in roots:
        rr = os.path.realpath(root)
        if real == rr or real.startswith(rr + os.sep):
            return root
    return None


def _lib_dir_candidates(root: str) -> List[str]:
    # The engine dir itself + the usual binary/lib locations: a prebuilt release
    # zip unpacks .so at the top level or under lib/; a from-source cmake build
    # co-locates them with the binaries under build/bin (and sometimes build/lib).
    cands = [
        root,
        os.path.join(root, "lib"),
        os.path.join(root, "bin"),
        os.path.join(root, "build", "bin"),
        os.path.join(root, "build", "lib"),
    ]
    # Any other ``lib`` dir shallowly under root or build/ (cmake variants).
    for base in (root, os.path.join(root, "build")):
        try:
            for name in os.listdir(base):
                cands.append(os.path.join(base, name, "lib"))
        except OSError:
            pass
    out: List[str] = []
    seen = set()
    for d in cands:
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def engine_lib_dirs(bin_path: Optional[str] = None) -> List[str]:
    """Existing directories that may hold the native llama.cpp shared libs.

    Without ``bin_path`` this keeps the historical behavior: dirs only when the
    engine dir is env-pinned (``HUGPY_ENGINE_DIR``/``LLAMA_CPP_DIR``) — we do
    NOT guess a default, so a box with a system/PATH llama-server (its libs
    already resolvable) is left untouched.

    With ``bin_path`` (the binary about to be spawned) it returns the lib dirs
    of the hugpy-managed root that binary lives under — the default data-dir
    install and the persisted-config location included — and ``[]`` for a
    system/PATH binary, preserving the dont-touch rule. Only dirs that exist
    are returned.
    """
    if bin_path is not None:
        root = managed_engine_root(bin_path)
        return _lib_dir_candidates(root) if root else []
    if not _env_pinned_dir():
        return []
    return _lib_dir_candidates(engine_dir())


def ld_library_path_with_engine(current: Optional[str] = None,
                                bin_path: Optional[str] = None) -> Optional[str]:
    """Prepend the engine lib dirs (:func:`engine_lib_dirs`) to an
    ``LD_LIBRARY_PATH`` value, skipping any already present.

    ``bin_path`` additionally pulls in the lib dirs of the managed root the
    binary being spawned was resolved from, so a default-location or
    persisted-config install works without an env pin; env-pinned dirs are
    still always included (unchanged behavior).

    Returns ``current`` unchanged when there is nothing to add (no managed
    binary or engine-dir override, dirs already present, or non-Linux where
    LD_LIBRARY_PATH is inert). None-safe: a ``None`` input with dirs to add
    yields just the new dirs joined.
    """
    import sys
    if not sys.platform.startswith("linux"):
        return current
    dirs = engine_lib_dirs()
    if bin_path:
        for d in engine_lib_dirs(bin_path):
            if d not in dirs:
                dirs.append(d)
    if not dirs:
        return current
    parts = [p for p in (current or "").split(os.pathsep) if p]
    have = set(parts)
    new = [d for d in dirs if d not in have]
    if not new:
        return current
    return os.pathsep.join(new + parts)


# --------------------------------------------------------------------------- #
# Native-engine health snapshot                                               #
# --------------------------------------------------------------------------- #
# "Installed but unusable" was invisible from central: /health showed only the
# python binding, while a resolvable binary whose exec dies on missing sibling
# libs silently fell back to the projector-less python server (the vision
# wedge). native_engine_status() is the remote-visibility fix (k94).
_SPAWN_PROBE_TIMEOUT_S = 10.0
_SPAWN_PROBE_TTL_S = 300.0
_spawn_probe_cache: dict = {}          # bin path -> (monotonic_at, ok, error)


def _spawn_probe(bin_path: str) -> Tuple[bool, Optional[str]]:
    """One-shot ``llama-server --version``: an ``isfile``'d binary can still die
    at exec, so this proves the loader actually starts it. Runs with the same
    LD_LIBRARY_PATH derivation the real spawn sites use."""
    env = dict(os.environ)
    ld = ld_library_path_with_engine(env.get("LD_LIBRARY_PATH"), bin_path=bin_path)
    if ld:
        env["LD_LIBRARY_PATH"] = ld
    try:
        proc = subprocess.run([bin_path, "--version"], capture_output=True,
                              text=True, timeout=_SPAWN_PROBE_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return False, f"--version probe exceeded {_SPAWN_PROBE_TIMEOUT_S:.0f}s"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    blob = (proc.stderr or "") + "\n" + (proc.stdout or "")
    # llama.cpp prints ``version: <N> (<hash>)`` — accept it even on a nonzero
    # rc (older builds exit oddly on --version); a loader death (rc 127,
    # "cannot open shared object file") never prints it.
    if proc.returncode == 0 or "version" in blob.lower():
        return True, None
    tail = " ".join(blob.split())[-300:]
    return False, f"exit {proc.returncode}: {tail}".strip().rstrip(":")


def native_engine_status(probe: bool = True) -> dict:
    """The worker /health ``native_engine`` snapshot: where (and whether) the
    native llama-server resolves, and whether it actually spawns. ``spawn_ok``
    is TTL-cached per binary so /health and the heartbeat stay fast; ``None``
    means not probed (binary absent, or ``probe=False``)."""
    path, source = _resolve_info("llama-server", "LLAMA_SERVER_BIN")
    out = {"found": path is not None, "path": path, "source": source,
           "spawn_ok": None, "error": None}
    if not path or not probe:
        return out
    now = time.monotonic()
    hit = _spawn_probe_cache.get(path)
    if hit is None or now - hit[0] >= _SPAWN_PROBE_TTL_S:
        ok, err = _spawn_probe(path)
        hit = (now, ok, err)
        _spawn_probe_cache[path] = hit
    out["spawn_ok"], out["error"] = hit[1], hit[2]
    return out
