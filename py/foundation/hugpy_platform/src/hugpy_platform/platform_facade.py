"""Cross-platform foundation for hugpy.

hugpy began life as a Linux-only service and grew a lot of POSIX assumptions:
systemd units, ``/srv`` and ``/etc`` paths, ``nvidia-smi`` + ``/proc/meminfo``
probing, ``os.killpg`` process groups. This package is the seam that lets the
same code run on Windows, macOS, and Linux:

    paths     — per-OS data/config/cache/engine dirs (XDG / AppData / Library)
    binaries  — resolve an executable name to a path, ``.exe`` on Windows
    procutil  — spawn detached, terminate a process tree, re-exec — portably
    hardware  — RAM + GPU probes that degrade to ``None`` off Linux/NVIDIA

Everything honours the existing env-var overrides (via ``env_value`` here, which
consults the process environment and then a ``.env`` file through
``abstract_essentials.get_env_value``) so deployments that already set
``LLAMA_CPP_DIR``/``DEFAULT_ROOT``/etc. keep working.
"""
from __future__ import annotations

import os
import sys

try:  # abstract_essentials is a declared dependency; degrade to os.environ only.
    from abstract_essentials.env_utils import get_env_value as _file_env_value
except Exception:  # pragma: no cover - only when the dependency is absent
    _file_env_value = None

IS_WINDOWS = sys.platform.startswith("win")
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

EXE_SUFFIX = ".exe" if IS_WINDOWS else ""


def _sanitize_env_str(raw: str) -> str:
    """Strip an inline ``# comment`` and surrounding whitespace/quotes from a
    ``.env``-style value.

    ``abstract_security``/``abstract_essentials`` reads ``.env`` files line by
    line and hands back everything after the ``=`` verbatim, comment included
    (a shell/dotenv reader would normally treat ``KEY=val   # note`` as a
    comment, but this one does not). A trailing inline comment on a path
    value like ``HUGPY_ENGINE_DIR=/mnt/.../engine   # native build dir`` used
    to get baked straight into ``os.makedirs`` calls, producing a garbage
    directory whose name was the literal comment text (the computron
    incident this function exists to prevent).

    Rules, in order:
      1. If the value is wrapped in a single or double quote pair, unwrap it
         and return the inner text as-is — a quoted value owns any ``#`` it
         contains (e.g. ``KEY="a#b"`` -> ``a#b``), including quoted values
         that themselves contain a trailing ``# comment`` inside the quotes.
      2. Otherwise, an unquoted value is cut at the first whitespace-then-``#``
         (`` #``) — i.e. a ``#`` must be preceded by whitespace to start a
         comment, so a bare ``#`` glued to non-space text (rare, but avoids
         mangling something like a URL fragment) is left alone.
      3. Trailing whitespace is stripped in both cases.
    """
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    # unquoted: cut at the first "whitespace + #" (inline comment marker)
    idx = None
    for i in range(1, len(s)):
        if s[i] == "#" and s[i - 1].isspace():
            idx = i - 1
            break
    if idx is not None:
        s = s[:idx]
    return s.strip()


def env_value(name: str):
    """Resolve a config override the way the rest of hugpy does.

    Consults ``abstract_essentials.get_env_value`` (process environment first,
    then a ``.env`` file found from the working directory / home / ``~/.envy_all``),
    falling back to ``os.environ``. Returns ``None`` when unset/empty so callers
    can ``or`` in a default.

    Precedence: a value in the PROCESS ENVIRONMENT wins over the ``.env`` file
    (that is ``abstract_essentials.get_env_value``'s own rule, so a systemd
    ``Environment=`` line or a ``FLAG=x cmd`` prefix always takes effect). Because
    the file is free-form text (not shell-parsed), every value — file or
    ``os.environ`` — is run through ``_sanitize_env_str`` before being returned,
    so a stray inline comment or accidental quoting can never leak into a
    path/config value (see ``_sanitize_env_str`` for the computron incident this
    closes).
    """
    val = None
    if _file_env_value is not None:
        try:
            val = _file_env_value(name)
        except Exception:
            val = None
    if val is None:
        val = os.environ.get(name)
    if isinstance(val, str):
        val = _sanitize_env_str(val)
        return val or None
    return val
