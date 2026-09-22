"""Per-model dependency PROFILES — isolated venvs for slot children (stage 1).

Operator problem (2026-07-12): a model that needs EXTRA pip deps (the canonical
example is ``optimum``) must be installable WITHOUT version-conflict risk to the
shared worker venv. Two versions of one package cannot share a process — that
physics note is the whole doctrine — so isolation happens at the PROCESS seam: a
profiled model's SLOT CHILD launches from the profile's own venv, never the
agent's.

A **profile** = a named venv + a declarative manifest ``{name, packages:[...]}``,
materialized on the worker at ``<worker_root>/envs/<name>/`` (a plain
``python -m venv`` + ``pip install`` of the manifest). The BASE package
``abstract_hugpy_dev`` is deliberately NOT installed there — the venv serves
binaries/deps to a slot child (the ``llama_cpp.server`` python fallback child, or
a profile-shipped binary on the child's PATH), NOT the agent. **The agent's own
process NEVER imports from a profile venv**; it only computes the child's
interpreter / PATH and hands them to the slot at the spawn seam.

Design shape (matches the tree's idioms):

  * Pure mechanics + a small registry. This module holds NO operator settings —
    exactly like ``slots.set_eviction_policy``, the agent registers a resolver
    (``set_model_resolver``) that reads its own ``_RUNTIME_SETTINGS`` so all
    settings-reading stays in one place and this module never imports the agent
    (no circular import; importable by the agent AND by the runner seam).
  * Materialization is a background job (the agent kicks ``materialize_all`` at
    boot with its ``register_executor`` so a restart shuts the pool down
    cleanly). It is SLOW (pip), so it runs off the request path; a profile only
    routes/seats once it is ``ready``.
  * Failure is DATA, never a crash: every outcome is stamped into
    ``<dir>/profile.json`` (ok/error + the manifest hash + an error string) and
    surfaced in the heartbeat, so a bad manifest never brings the agent down.

Stage 1 wires the SLOT-child seam (the ``llama_cpp.server`` python child is the
real, tested consumer; a profile-shipped native binary rides the child PATH).
In-process transformers/diffusers models ignore profiles in stage 1 — moving
them to runner children (the first ``optimum`` consumer) is stage 3.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time

logger = logging.getLogger("abstract_hugpy_dev.profiles")

# Slug-safe profile name: starts alnum, then alnum / dot / underscore / hyphen.
# Bounds the on-disk directory name (no traversal, no spaces).
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# Profiles whose materialization is queued/running RIGHT NOW — so the heartbeat
# reports 'materializing' the instant a kick lands, before the pip finishes.
_INFLIGHT: "set[str]" = set()
_INFLIGHT_LOCK = threading.Lock()

# The most recent materialization pool (kept as a strong ref so it isn't GC'd
# while its tasks run; the agent also registers it so a restart shuts it down).
_POOL = None

# Registered by the agent: model_key -> {'name','state','bin','error'} | None.
# Kept here (not read from the agent) so this module never imports the agent.
_RESOLVER = None


# --------------------------------------------------------------------------- #
# naming / paths                                                              #
# --------------------------------------------------------------------------- #
def slug_ok(name) -> bool:
    """True iff ``name`` is a safe profile identifier (directory-name safe)."""
    return isinstance(name, str) and bool(_SLUG_RE.match(name))


def worker_root() -> str:
    """The box-local worker home the tree already uses.

    The installer (``worker_agent/install.py``) sets ``HUGPY_ENGINE_DIR`` to
    ``~/hugpy-worker/engine``; ``~/hugpy-worker`` is that canonical root. We
    honor an explicit ``HUGPY_WORKER_ROOT`` override first, then DERIVE the root
    from ``HUGPY_ENGINE_DIR`` when it points inside a ``hugpy-worker`` tree (so a
    custom install location is respected), and finally fall back to
    ``~/hugpy-worker`` — never a hardcoded absolute path.
    """
    override = os.environ.get("HUGPY_WORKER_ROOT")
    if override:
        return os.path.expanduser(override)
    engine = os.environ.get("HUGPY_ENGINE_DIR")
    if engine:
        parent = os.path.dirname(os.path.normpath(os.path.expanduser(engine)))
        if os.path.basename(parent) == "hugpy-worker":
            return parent
    return os.path.join(os.path.expanduser("~"), "hugpy-worker")


def profiles_root() -> str:
    """Where all profile venvs live: ``<worker_root>/envs``."""
    return os.path.join(worker_root(), "envs")


def profile_dir(name: str) -> str:
    return os.path.join(profiles_root(), name)


def profile_bin_dir(name: str) -> str:
    """The venv's executable dir (``bin`` on POSIX, ``Scripts`` on Windows) —
    this is what gets PREPENDED to the slot child's PATH."""
    sub = "Scripts" if os.name == "nt" else "bin"
    return os.path.join(profile_dir(name), sub)


def profile_python(name: str) -> str:
    """The venv's python interpreter — what a python-launched slot child
    (``python -m llama_cpp.server``) is spawned from instead of the agent's."""
    exe = "python.exe" if os.name == "nt" else "python"
    return os.path.join(profile_bin_dir(name), exe)


def _state_path(name: str) -> str:
    return os.path.join(profile_dir(name), "profile.json")


def manifest_hash(packages) -> str:
    """Stable short hash of the (normalized) package manifest. A change here is
    what triggers a re-materialize; a byte-identical manifest is a no-op."""
    pkgs = [p.strip() for p in (packages or []) if isinstance(p, str) and p.strip()]
    return hashlib.sha256(json.dumps(pkgs).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# on-disk state (profile.json) — the honest record the heartbeat reads         #
# --------------------------------------------------------------------------- #
def read_state(name: str) -> "dict | None":
    try:
        with open(_state_path(name), "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def _write_state(name: str, *, ok: bool, hash: str, packages, error=None) -> None:
    d = profile_dir(name)
    os.makedirs(d, exist_ok=True)
    payload = {
        "name": name,
        "ok": bool(ok),
        "hash": hash,
        "packages": list(packages or []),
        "materialized_at": time.time(),
        "error": error,
    }
    tmp = _state_path(name) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, _state_path(name))


def _mark_inflight(name: str) -> None:
    with _INFLIGHT_LOCK:
        _INFLIGHT.add(name)


def _clear_inflight(name: str) -> None:
    with _INFLIGHT_LOCK:
        _INFLIGHT.discard(name)


def _is_inflight(name: str) -> bool:
    with _INFLIGHT_LOCK:
        return name in _INFLIGHT


# --------------------------------------------------------------------------- #
# materialization                                                             #
# --------------------------------------------------------------------------- #
def _run(cmd: "list[str]", timeout: float = 1800.0) -> None:
    """Run a venv/pip subprocess, raising on non-zero. This is the SINGLE seam
    tests fake to exercise the lifecycle without a real pip. Output is captured
    (never inherits the agent's stdout) and the tail rides the error record."""
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(
            f"command failed (rc={proc.returncode}): {' '.join(cmd)}\n{tail}")


def materialize(name: str, packages) -> dict:
    """(Re)create the profile venv and install its manifest. Idempotent by
    manifest hash. **Never raises** — the outcome is recorded as DATA in
    ``profile.json`` (ok/error) so the heartbeat can report it and the agent is
    never crashed by a bad manifest.

    Returns ``{'state': 'ready'|'error', ...}``.
    """
    if not slug_ok(name):
        logger.warning("profiles: refusing invalid profile name %r", name)
        return {"state": "error", "error": f"invalid profile name {name!r}"}
    pkgs = [p.strip() for p in (packages or []) if isinstance(p, str) and p.strip()]
    want = manifest_hash(pkgs)
    st = read_state(name)
    if (st and st.get("ok") and st.get("hash") == want
            and os.path.isdir(profile_dir(name))):
        return {"state": "ready", "hash": want}         # idempotent no-op

    _mark_inflight(name)
    try:
        os.makedirs(profiles_root(), exist_ok=True)
        # Fresh venv (--clear rebuilds cleanly on a manifest change). The BASE
        # package abstract_hugpy_dev is deliberately NOT installed here — the
        # venv serves the slot child's deps/binaries, not the agent.
        _run([sys.executable, "-m", "venv", "--clear", profile_dir(name)])
        if pkgs:
            _run([profile_python(name), "-m", "pip", "install", "--upgrade", *pkgs])
        _write_state(name, ok=True, hash=want, packages=pkgs, error=None)
        logger.info("profiles: materialized %r (%d package(s))", name, len(pkgs))
        return {"state": "ready", "hash": want}
    except Exception as exc:   # noqa: BLE001 — failure must be data, never a crash
        err = f"{type(exc).__name__}: {exc}"
        logger.warning("profiles: materialization of %r failed: %s", name, err)
        _write_state(name, ok=False, hash=want, packages=pkgs, error=err)
        return {"state": "error", "error": err}
    finally:
        _clear_inflight(name)


def state_for(name: str, packages) -> str:
    """Coarse routing/heartbeat state for one profile: ``ready`` |
    ``materializing`` | ``error``. ``ready`` means the on-disk venv matches the
    CURRENT manifest hash and its last pip succeeded — the only state in which a
    profiled model may seat."""
    want = manifest_hash(packages)
    if _is_inflight(name):
        return "materializing"
    st = read_state(name)
    if st and st.get("hash") == want:
        return "ready" if st.get("ok") else "error"
    # Declared but not materialized for THIS manifest yet (new / changed) — a
    # background kick is pending; report it as materializing (don't route).
    return "materializing"


def report(declared: dict) -> dict:
    """Heartbeat truth: ``{name: {state, error?}}`` for every DECLARED profile.
    Errors carry their message so the console shows why a profile is stuck."""
    out: dict = {}
    for name, spec in (declared or {}).items():
        packages = (spec or {}).get("packages") or []
        state = state_for(name, packages)
        row = {"state": state}
        if state == "error":
            row["error"] = (read_state(name) or {}).get("error")
        out[name] = row
    return out


def materialize_all(declared: dict, *, register=None):
    """Kick a background job to materialize every declared profile that is not
    already ``ready`` for its current manifest. Registered (via ``register`` —
    the agent's ``register_executor``) so a restart shuts the pool down cleanly.

    Idempotent: a ready profile is a no-op; a changed manifest re-materializes.
    Returns the pool (or None when nothing was pending) so callers/tests can
    join it; the agent ignores the return.
    """
    declared = declared or {}
    pending = []
    for name, spec in declared.items():
        if not slug_ok(name):
            logger.warning("profiles: skipping invalid profile name %r", name)
            continue
        packages = (spec or {}).get("packages") or []
        if state_for(name, packages) != "ready":
            pending.append((name, packages))
            _mark_inflight(name)      # report 'materializing' the instant we kick
    if not pending:
        return None

    from concurrent.futures import ThreadPoolExecutor
    global _POOL
    if _POOL is not None:             # retire a prior batch (defensive)
        try:
            _POOL.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass
    # max_workers=1: materialize serially (pip installs are heavy; a single
    # lane keeps the box calm and the cache uncontended). Slow but off the
    # request path — a profile only routes once ready.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="profile-materialize")
    _POOL = pool
    if register is not None:
        try:
            register(pool)
        except Exception:  # noqa: BLE001 — registration is best-effort
            pass
    for name, packages in pending:
        pool.submit(materialize, name, packages)
    # Let the pool retire its worker once the batch drains; already-submitted
    # tasks still run to completion (shutdown(wait=False) blocks no new submits).
    pool.shutdown(wait=False)
    return pool


# --------------------------------------------------------------------------- #
# resolver registry (the runner seam consumes this; the agent registers it)    #
# --------------------------------------------------------------------------- #
def set_model_resolver(fn) -> None:
    """Register the agent's ``model_key -> profile-decision`` resolver. ``fn``
    returns ``{'name','state','bin','error'}`` for an attributed model, or None
    for an unattributed one. Mirrors ``slots.set_eviction_policy`` so all
    operator-settings reading stays in the agent."""
    global _RESOLVER
    _RESOLVER = fn


def resolve_model(model_key: str) -> "dict | None":
    """The runner seam's single entrypoint: the profile decision for
    ``model_key``, or None when no profile is attributed (base behavior). A
    broken/absent resolver returns None (never breaks serving)."""
    if _RESOLVER is None:
        return None
    try:
        return _RESOLVER(model_key)
    except Exception:  # noqa: BLE001 — a broken resolver must never break serving
        return None


# --------------------------------------------------------------------------- #
# slot-child spawn helpers (used at the seam in slot_agent.py)                 #
# --------------------------------------------------------------------------- #
def child_python(profile_bin: "str | None", default_python: str) -> str:
    """Interpreter a PYTHON-launched slot child (``python -m llama_cpp.server``)
    should run from. With a profile bin dir → that venv's python (must exist, or
    we raise errors-as-data rather than silently using the shared venv, which
    would reintroduce the conflict class). No profile → the agent's default."""
    if not profile_bin:
        return default_python
    exe = "python.exe" if os.name == "nt" else "python"
    cand = os.path.join(profile_bin, exe)
    if os.path.isfile(cand):
        return cand
    raise RuntimeError(
        f"profile venv python missing at {cand!r} — the profile is not "
        "materialized (or its venv was removed); refusing to fall back to the "
        "shared interpreter")


def child_env(env: dict, profile_bin: "str | None") -> dict:
    """The child's environment with the profile venv activated: its bin dir
    PREPENDED to PATH (so a profile-shipped binary wins) + the standard
    ``VIRTUAL_ENV`` marker, ``PYTHONHOME`` dropped. A no-op (returns ``env``
    unchanged) without a profile or if the bin dir is absent."""
    if profile_bin and os.path.isdir(profile_bin):
        env = dict(env)
        env["PATH"] = profile_bin + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = os.path.dirname(profile_bin)
        env.pop("PYTHONHOME", None)
    return env
