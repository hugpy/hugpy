"""SSD hot-cache for GGUF models.

Canonical models live on the large/slow HDD under MODELS_HOME. This promotes the
*hot set* onto a fast SSD mount (HUGPY_MODEL_CACHE) and serves loads from there;
the HDD stays the source of truth. The cache is bounded (LRU-evicted) so it never
mirrors the whole library — only the models actually in use.

Why warm-then-load (not copy-inline): a cold HDD->SSD copy of a 45 GiB model takes
minutes, which can't fit inside a synchronous load request (gunicorn --timeout).
So `use()` never blocks: it serves the SSD copy if present, otherwise kicks a
*background* warm and returns the HDD path for this load. Warm a model once and
subsequent loads are NVMe-fast.
"""
import os
import re
import glob
import shutil
import threading
import time
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = os.environ.get("HUGPY_MODEL_CACHE", "/var/cache/hugpy-models")
CACHE_MAX_BYTES = int(float(os.environ.get("HUGPY_MODEL_CACHE_MAX_GIB", "450")) * (1 << 30))

_warming: set = set()                 # srcs being warmed right now (single-flight)
_warming_lock = threading.Lock()


def enabled() -> bool:
    return os.path.isdir(CACHE_DIR) and os.access(CACHE_DIR, os.W_OK)


def _models_home() -> str:
    try:
        from hugpy_platform.constants import MODELS_HOME
        return MODELS_HOME
    except Exception:
        return "/mnt/llm_storage/models"


def file_set(src: str) -> list:
    """The files needed to load `src`: the gguf itself, all shards if it's a split
    model (…-00001-of-00004.gguf), and a sibling mmproj projector if present."""
    out = [src]
    d = os.path.dirname(src)
    base = os.path.basename(src)
    m = re.search(r"-\d{5}-of-(\d{5})\.gguf$", base)
    if m:
        out = glob.glob(os.path.join(d, f"{base[:m.start()]}-*-of-{m.group(1)}.gguf"))
    out += glob.glob(os.path.join(d, "*mmproj*.gguf")) + glob.glob(os.path.join(d, "*mmproj*.GGUF"))
    return sorted({f for f in out if os.path.isfile(f)})


def cache_path(src: str) -> str:
    """Mirror src under CACHE_DIR (MODELS_HOME-relative; falls back to basename)."""
    home = _models_home()
    try:
        rel = os.path.relpath(src, home)
        if rel.startswith(".."):
            raise ValueError
    except Exception:
        rel = os.path.basename(src)
    return os.path.join(CACHE_DIR, rel)


def is_complete(src: str) -> bool:
    """True if every file `src` needs is in the cache with a matching size."""
    for f in file_set(src):
        c = cache_path(f)
        try:
            if not (os.path.isfile(c) and os.path.getsize(c) == os.path.getsize(f)):
                return False
        except OSError:
            return False
    return True


def _entry_dir(src: str) -> str:
    """The cache dir that holds this model's gguf(s) — the eviction unit."""
    return os.path.dirname(cache_path(src))


def touch(src: str) -> None:
    try:
        now = time.time()
        for f in file_set(src):
            c = cache_path(f)
            if os.path.exists(c):
                os.utime(c, (now, now))
    except Exception:
        pass


def _cache_used_bytes() -> int:
    total = 0
    for root, _d, files in os.walk(CACHE_DIR):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _free_bytes() -> int:
    try:
        st = os.statvfs(CACHE_DIR)
        return st.f_bavail * st.f_frsize
    except OSError:
        return 0


def _leaf_model_dirs() -> list:
    """Cache dirs that directly contain .gguf files (the evictable entries),
    each with its newest mtime (LRU key)."""
    entries = []
    for root, _d, files in os.walk(CACHE_DIR):
        ggufs = [f for f in files if f.lower().endswith(".gguf")]
        if ggufs:
            mt = max(os.path.getmtime(os.path.join(root, f)) for f in ggufs)
            sz = sum(os.path.getsize(os.path.join(root, f)) for f in files)
            entries.append((mt, root, sz))
    return entries


def evict_for(need_bytes: int, keep_dir: str = "") -> None:
    """Delete least-recently-used cache entries until `need_bytes` fits under the
    budget AND on the filesystem. Never evicts keep_dir (the one being warmed)."""
    def over_budget():
        return _cache_used_bytes() + need_bytes > CACHE_MAX_BYTES or _free_bytes() < need_bytes
    if not over_budget():
        return
    for mt, d, sz in sorted(_leaf_model_dirs()):           # oldest first
        if d == keep_dir or keep_dir.startswith(d + os.sep):
            continue
        try:
            shutil.rmtree(d)
            logger.info("model_cache: evicted %s (%.1f GiB, idle since %s)",
                        d, sz / (1 << 30), time.strftime("%Y-%m-%d %H:%M", time.localtime(mt)))
        except OSError as exc:
            logger.warning("model_cache: evict failed for %s: %s", d, exc)
        if not over_budget():
            break


def warm(src: str) -> str | None:
    """Copy src (+ shards + mmproj) into the cache, LRU-evicting first. Verified
    (size-checked) and atomic per file (.part -> rename). Single-flight per src.
    Returns the cached gguf path, or None on failure/disabled."""
    if not enabled():
        return None
    files = file_set(src)
    if not files:
        return None
    with _warming_lock:
        if src in _warming:
            return None                                    # another thread has it
        _warming.add(src)
    try:
        need = sum(os.path.getsize(f) for f in files)
        evict_for(need, keep_dir=_entry_dir(src))
        for f in files:
            dst = cache_path(f)
            if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(f):
                continue                                   # already cached, complete
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".part"
            logger.info("model_cache: warming %s -> %s (%.1f GiB)",
                        f, dst, os.path.getsize(f) / (1 << 30))
            shutil.copyfile(f, tmp)
            if os.path.getsize(tmp) != os.path.getsize(f):
                os.remove(tmp)
                raise IOError(f"size mismatch copying {f}")
            os.replace(tmp, dst)
        logger.info("model_cache: warmed %s", src)
        return cache_path(src)
    except Exception as exc:
        logger.warning("model_cache: warm failed for %s: %s", src, exc)
        return None
    finally:
        with _warming_lock:
            _warming.discard(src)


def _warm_async(src: str) -> None:
    with _warming_lock:
        if src in _warming:
            return
    threading.Thread(target=warm, args=(src,), daemon=True, name="model-cache-warm").start()


def use(src: str) -> str:
    """Resolve the path a loader should open. SSD copy if warm (and touch it for
    LRU); otherwise kick a background warm and return the HDD path for this load.
    Never blocks — a cold load runs off the HDD while the cache fills for next time."""
    if not src or not enabled():
        return src
    try:
        if is_complete(src):
            touch(src)
            return cache_path(src)
        _warm_async(src)
    except Exception as exc:
        logger.warning("model_cache: use() fell back to HDD for %s: %s", src, exc)
    return src


def status() -> dict:
    """Cache overview for the console."""
    if not enabled():
        return {"enabled": False}
    entries = [{"dir": os.path.relpath(d, CACHE_DIR), "bytes": sz, "mtime": mt}
               for mt, d, sz in sorted(_leaf_model_dirs(), reverse=True)]
    return {"enabled": True, "dir": CACHE_DIR, "max_bytes": CACHE_MAX_BYTES,
            "used_bytes": _cache_used_bytes(), "free_bytes": _free_bytes(),
            "warming": sorted(_warming), "entries": entries}
