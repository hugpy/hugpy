"""Physical presence of a model on disk — storage's own completeness rule.

"Is this directory a usable, COMPLETE copy of the model?" is a question about
bytes at rest, so it is answered here, from the on-disk truth plus the few
routing facts a registry row carries (framework, pinned filename, include
patterns, tasks). The engine's ``config.main.model_looks_downloaded`` is the
same rule (this is its home now; the engine may re-export it); provision,
the console status feed and the read-through resolver all call THIS one so
"installed" can never mean two different things on the same box.

Also here: the small disk-error helpers a transfer failure needs to explain
itself (``errno_name`` / ``disk_stats`` / ``describe_disk_error``) and the
whole-directory size walk (``dir_size_bytes``).

``cfg`` may be an attribute-style config (the engine's ModelConfig), a plain
dict row, or None; only ``framework``, ``filename``, ``include``,
``primary_task`` and ``tasks`` are read.
"""
from __future__ import annotations

import glob
import os
import threading
from typing import Any, Optional

_WEIGHT_FLOOR = 1024 * 1024        # below this a *.safetensors is an LFS pointer stub


def _field(cfg: Any, name: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _is_mmproj(name: str) -> bool:
    try:
        from hugpy_platform.utils import is_mmproj_file
        return bool(is_mmproj_file(name))
    except Exception:  # noqa: BLE001
        low = os.path.basename(str(name or "")).lower()
        return "mmproj" in low or "mm-proj" in low or "mm_proj" in low


def _find_mmproj(path: str) -> Optional[str]:
    try:
        from hugpy_platform.utils import find_mmproj
        return find_mmproj(path)
    except Exception:  # noqa: BLE001
        for root, _dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".gguf") and _is_mmproj(f):
                    return os.path.join(root, f)
        return None


def _rglob(path: str, pattern: str, recursive: bool = False) -> list[str]:
    if recursive:
        return glob.glob(os.path.join(path, "**", pattern), recursive=True)
    return glob.glob(os.path.join(path, pattern))


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def gguf_files(path: str) -> list[str]:
    """Every model .gguf under ``path`` (recursive; shards nest in subdirs),
    projector sidecars excluded. Sorted for determinism."""
    if not path or not os.path.isdir(path):
        return []
    found = _rglob(path, "*.gguf", recursive=True) + _rglob(path, "*.GGUF", recursive=True)
    return sorted({g for g in found if not _is_mmproj(g)})


def find_gguf_file(path: str, cfg: Any = None, prefer: Optional[str] = None) -> Optional[str]:
    """The GGUF a load would open, by storage's rule: a designation (``prefer``
    then ``cfg.filename``) matched by exact basename or substring (a quant tag),
    else the single file, else the lexically-first one. The engine's runtime
    election (quant ranking, shard-set completeness) is a serving decision and
    lives in ``hugpy_engine.gguf_election``; presence only needs *a* file."""
    ggufs = gguf_files(path)
    if not ggufs:
        return None
    for want in (prefer, _field(cfg, "filename")):
        if not want or _is_mmproj(want):
            continue
        base = os.path.basename(str(want)).lower()
        for g in ggufs:
            if os.path.basename(g).lower() == base:
                return g
        hits = sorted(g for g in ggufs if base in os.path.basename(g).lower())
        if hits:
            return hits[0]
    return ggufs[0]


def _wants_vision(cfg: Any) -> bool:
    if _field(cfg, "primary_task") == "image-text-to-text":
        return True
    if "image-text-to-text" in (_field(cfg, "tasks") or []):
        return True
    inc = _field(cfg, "include") or []
    if isinstance(inc, str):
        inc = [inc]
    return any(_is_mmproj(x) for x in inc)


def model_looks_downloaded(path: str, cfg: Any = None) -> bool:
    """Lightweight completeness check that never mistakes a partial HF / LFS
    pointer directory for a usable model dir.

    * comfy: the single checkpoint file (``cfg.filename``) is present.
    * gguf: some model .gguf over 1 MiB (pin honoured when present; any
      complete quant otherwise); a VISION gguf must also have its mmproj
      projector beside it or llama.cpp silently loads text-only.
    * transformers / diffusers: a config.json with >1 MiB safetensors (or the
      classic expected files); a model_index.json pipeline with real weights
      in its component dirs; or a config-less custom repo with real weights.
    """
    if _field(cfg, "framework") == "comfy":
        fn = _field(cfg, "filename", "") or ""
        return bool(fn) and os.path.isfile(os.path.join(path or "", fn))

    if not path or not os.path.isdir(path):
        return False

    if _field(cfg, "framework") == "gguf":
        g = find_gguf_file(path, cfg)
        if not (g and os.path.exists(g) and _size(g) > _WEIGHT_FLOOR):
            return False
        if _wants_vision(cfg) and not _find_mmproj(path):
            return False
        return True

    if not os.path.isfile(os.path.join(path, "config.json")):
        if os.path.exists(os.path.join(path, "model_index.json")):
            weights = _rglob(path, "*.safetensors", recursive=True) \
                or _rglob(path, "*.bin", recursive=True)
            if not weights:
                return False
            return all(_size(f) > _WEIGHT_FLOOR for f in weights)
        weights = _rglob(path, "*.safetensors") or _rglob(path, "*.bin")
        if weights and all(_size(f) > _WEIGHT_FLOOR for f in weights):
            return True
        return False

    safetensor_files = _rglob(path, "*.safetensors")
    if safetensor_files:
        return all(_size(f) >= _WEIGHT_FLOOR for f in safetensor_files)

    expected_any = (
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "processor_config.json",
    )
    return any(os.path.exists(os.path.join(path, name)) for name in expected_any)


def pinned_filename_present(directory: str, cfg: Any) -> Optional[bool]:
    """For a GGUF with a pinned ``filename``: is that EXACT quant present? A
    mismatch is a WARNING (the model still serves off another complete quant),
    not "incomplete". None when there is no pin to check."""
    fn = _field(cfg, "filename")
    fw = (_field(cfg, "framework") or "")
    if str(fw).lower() != "gguf" or not fn:
        return None
    base = os.path.basename(str(fn)).lower()
    try:
        for root, _dirs, files in os.walk(directory or ""):
            for f in files:
                b = f.lower()
                if b == base or base in b:
                    return True
    except OSError:
        return False
    return False


# ── directory size (cached by dir mtime) ─────────────────────────────────────
_SIZE_CACHE: dict = {}
_SIZE_LOCK = threading.Lock()


def dir_size_bytes(path: Optional[str]) -> Optional[int]:
    """On-disk footprint of ``path`` (every file, every format), cached by the
    directory's mtime so a listing stays cheap. None when not a directory."""
    if not path or not os.path.isdir(path):
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    with _SIZE_LOCK:
        hit = _SIZE_CACHE.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            total += _size(os.path.join(root, f))
    with _SIZE_LOCK:
        _SIZE_CACHE[path] = (mtime, total)
    return total


def walk_listing(root: str) -> list[tuple[str, int]]:
    """``[(relpath, size)]`` for every regular file under ``root``, dot-dirs
    (``.cache``, ``.git``, …) pruned — the transfer-manifest view of a dir."""
    out: list[tuple[str, int]] = []
    if not root or not os.path.isdir(root):
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for f in filenames:
            if f.startswith("."):
                continue
            full = os.path.join(dirpath, f)
            if os.path.islink(full) and not os.path.isfile(full):
                continue
            out.append((os.path.relpath(full, root), _size(full)))
    return out


# ── disk-error explanation ───────────────────────────────────────────────────
def errno_name(exc: BaseException) -> str:
    """``ENOSPC`` from an OSError, "" from anything else. The symbolic name is
    the part an operator can act on."""
    try:
        import errno as _errno
        num = getattr(exc, "errno", None)
        if num is None:
            return ""
        return _errno.errorcode.get(int(num), "") or ""
    except Exception:  # noqa: BLE001
        return ""


def _fmt_bytes(n: Optional[int]) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "unknown"
    if n < 1024:
        return f"{n} B"
    v = float(n)
    for unit in ("KB", "MB", "GB", "TB"):
        v /= 1024.0
        if v < 1024 or unit == "TB":
            return f"{v:.1f} {unit}".replace(".0 ", " ")
    return f"{v:.1f} TB"


def disk_stats(path: Optional[str]) -> dict:
    """``{disk_free_bytes, disk_total_bytes, disk_mount}`` for the filesystem
    ``path`` lives (or would be created) on; walks up to the nearest existing
    ancestor. ``{}`` rather than raising, always."""
    try:
        p = os.path.abspath(str(path or "."))
        for _ in range(64):
            if os.path.exists(p):
                break
            parent = os.path.dirname(p)
            if not parent or parent == p:
                break
            p = parent
        st = os.statvfs(p)
        return {
            "disk_free_bytes": int(st.f_bavail) * int(st.f_frsize),
            "disk_total_bytes": int(st.f_blocks) * int(st.f_frsize),
            "disk_mount": p,
        }
    except Exception:  # noqa: BLE001
        return {}


def describe_disk_error(exc: BaseException, dest_path: Optional[str] = None,
                        stats: Optional[dict] = None) -> str:
    """One operator-grade line for an OS-level transfer failure, e.g.
    ``disk full (ENOSPC) on /mnt/storage — 0 B free of 938 GB``.

    Only the errnos an operator can act on get a sentence: ENOSPC / EDQUOT
    (with free/total when known) and EROFS / EACCES / EPERM ("cannot write to
    <where> (<ERRNO>)"). Everything else returns "" so callers fall back to
    str(exc) without a special case. Contract carried over verbatim from the
    pre-partition ``comms.evictions.describe_disk_error``."""
    try:
        name = errno_name(exc)
        if not name:
            return ""
        st = stats if isinstance(stats, dict) else disk_stats(dest_path)
        where = st.get("disk_mount") or dest_path or "the destination volume"
        if name == "ENOSPC":
            head = f"disk full (ENOSPC) on {where}"
        elif name == "EDQUOT":
            head = f"disk quota exceeded (EDQUOT) on {where}"
        elif name in ("EROFS", "EACCES", "EPERM"):
            return f"cannot write to {where} ({name})"
        else:
            return ""
        free = st.get("disk_free_bytes")
        total = st.get("disk_total_bytes")
        if free is None or total is None:
            return head
        return f"{head} — {_fmt_bytes(free)} free of {_fmt_bytes(total)}"
    except Exception:  # noqa: BLE001
        return ""


__all__ = [
    "model_looks_downloaded", "find_gguf_file", "gguf_files",
    "pinned_filename_present", "dir_size_bytes", "walk_listing",
    "errno_name", "disk_stats", "describe_disk_error",
]
