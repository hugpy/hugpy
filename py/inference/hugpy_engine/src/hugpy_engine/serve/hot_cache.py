"""HOT-CACHE TIER — an automatic LRU cache of the MAIN model catalog on a
worker's fast box-local NVMe.

Operator doctrine (verbatim): "anything that is called should be on the hot
drive; when space gets tight and one is needed then the fifo should go by the
time since called and space afforded by deletion."

Shape of the mechanism
----------------------
The canonical models live on the big/shared array under ``MODELS_HOME``
(e.g. ``/mnt/llm_storage/models`` — the fleet's source of truth). This tier
promotes the *hot set* — the models actually being called — onto a box-local
NVMe (``HUGPY_HOT_CACHE_ROOT``) and serves loads from there. The shared array
stays the SOURCE OF TRUTH and is NEVER written or deleted by this module.

Read-through, never blocking
    ``use(shared_path)`` returns the path a loader should open. If a COMPLETE
    hot copy exists it returns the hot path (and stamps "last called"); else it
    kicks a *background* promotion and returns the shared path UNCHANGED for
    this (cold) load. A cold ~100 GiB HDD->NVMe copy is minutes of sustained IO
    and can NEVER sit inside a synchronous load request, so promotion is always
    async. Warm a model once and subsequent calls hit NVMe.

Env-gated, zero behaviour change when unset
    Unset ``HUGPY_HOT_CACHE_ROOT`` -> ``use()`` returns its argument byte-for-
    byte. A box without the env, or central, behaves exactly as before.

Concurrency / correctness
    A SINGLE promoter thread drains a queue (one big copy at a time — the shared
    array can't sustain many). Each file copies to a ``.part`` temp then
    atomically renames, and the completeness gate requires EVERY file present
    with a matching size, so a partial/in-flight copy is never resolvable.

Eviction = the operator's FIFO-by-time-since-called
    When a promotion needs room beyond the budget, evict least-recently-CALLED
    hot entries first, weighing the space each frees, until the incoming model
    fits. If it cannot fit even after evicting every eligible entry, SKIP the
    promotion (log it) — the actual load already ran off the shared array, so a
    load NEVER fails for lack of hot space.

Anti-thrash (churn is the normal mode here)
    Most models the 3090 serves are large relative to the budget (two or three
    fill the drive) and rotate in and out constantly. A model that JUST served
    must not be displaced by a first-time/stale caller — otherwise two big
    models alternating would copy+evict ~100 GiB per call. So an entry idle for
    less than ``HUGPY_HOT_CACHE_MIN_RESIDENCY_S`` is NOT an eviction candidate;
    if the only room is behind such a fresh entry, the promotion is skipped and
    that model keeps serving from the shared array until the activity pattern
    genuinely shifts.

This is the GENERAL mechanism for the main catalog. The studio render path has
a sibling ``STUDIO_WEIGHTS_HOT_ROOT`` (passive: use a hot dir if an operator
placed one there); see the report note on unifying the two.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import queue
import re
import shutil
import threading
import time

from hugpy_engine.serve import chunksum_verify as _cv

logger = logging.getLogger(__name__)

# Eviction telemetry (operator directive 2026-07-28). The hot tier is DISK
# eviction, and it belongs on the same live stream as VRAM eviction — "the
# entire process" includes the drive copy going away. Stdlib-only and
# best-effort: an unavailable emitter leaves this module byte-identical.
from hugpy_engine.placement import get_eviction_ledger as _ledger  # noqa: E402


def _evt_emit(stage: str, **fields) -> None:
    """Best-effort eviction-telemetry emit. Never raises — the promoter thread
    must not die, and a promotion must not change outcome, over a telemetry
    fault."""
    try:
        _ledger().emit(stage, **fields)
    except Exception:  # noqa: BLE001
        pass


def _verify_staged(src: str, staged: str) -> tuple[str, str]:
    """Content-verify a staged .part against its shared-store source's sidecar.

    Split out from _promote so it is directly drivable in tests and so a
    verification bug can never take down the promoter thread: an unexpected
    exception degrades to UNVERIFIED (copy proceeds, as it did before this
    gate existed) rather than turning the hot tier into a hard dependency on
    the sidecar machinery. Only a POSITIVE corruption proof stops a promote.
    """
    try:
        return _cv.verify_against_source(staged, src)
    except Exception as exc:  # noqa: BLE001 — never fail a promote on our own bug
        logger.warning("hot_cache: verification error for %s (%s) — treating as "
                       "unverified", src, exc)
        return _cv.UNVERIFIED, f"verifier error: {exc}"

# --------------------------------------------------------------------------- #
# Env knobs (read live so a systemd drop-in edit + restart takes effect; unset
# ROOT == disabled == byte-identical behaviour).
# --------------------------------------------------------------------------- #
_ENV_ROOT = "HUGPY_HOT_CACHE_ROOT"
_ENV_GIB = "HUGPY_HOT_CACHE_GIB"
_ENV_MIN_RESIDENCY = "HUGPY_HOT_CACHE_MIN_RESIDENCY_S"

_DEFAULT_GIB = 225.0
_DEFAULT_MIN_RESIDENCY_S = 1800.0
_INDEX_NAME = ".hot_cache_index.json"

GiB = 1 << 30


def _root() -> str:
    return (os.environ.get(_ENV_ROOT) or "").strip()


def _own_budget_bytes() -> int:
    """The hot tier's OWN declared budget (HUGPY_HOT_CACHE_GIB), unclamped."""
    try:
        return int(float(os.environ.get(_ENV_GIB, _DEFAULT_GIB)) * GiB)
    except (TypeError, ValueError):
        return int(_DEFAULT_GIB * GiB)


def _store_disk_cap_gib() -> float | None:
    """Central's disk_cache_gib for this box, projected into env by the worker
    agent's _adopt_storage_inputs (read LIVE). None when unset."""
    raw = os.environ.get("_HUGPY_CENTRAL_DISK_CACHE_GIB")
    if raw in (None, ""):
        return None
    try:
        v = float(raw)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _same_drive(a: str, b: str) -> bool:
    """True when paths a and b live on the same physical drive (same st_dev),
    walking up to the nearest existing parent. Mirrors budget._drive_id's signal
    so the hot tier and the pull gate agree on 'same drive'."""
    def _dev(p: str):
        p = p or ""
        for _ in range(64):
            try:
                return os.stat(p).st_dev
            except OSError:
                parent = os.path.dirname(p)
                if not parent or parent == p:
                    return None
                p = parent
        return None
    da, db = _dev(a), _dev(b)
    return da is not None and da == db


def _budget_bytes() -> int:
    """The EFFECTIVE hot-tier budget (slice 4, min-wins). The tier's own
    HUGPY_HOT_CACHE_GIB, floored by central's disk_cache_gib WHEN the hot root
    shares the store root's drive — so a promote/admit can never write bytes past
    the effective cap the pull gate would refuse (the ae 400-vs-1500 same-dir
    overcrowding). Different drive → the tiers don't contend → own budget stands.
    Unset central term or unresolvable roots → own budget, unchanged."""
    own = _own_budget_bytes()
    central_gib = _store_disk_cap_gib()
    if central_gib is None:
        return own
    root = _root()
    store = _models_home()
    if not root or not store:
        return own
    try:
        if not _same_drive(root, store):
            return own                       # separate drive — no contention
    except Exception:  # noqa: BLE001 — a drive probe must never break a load
        return own
    return min(own, int(central_gib * GiB))


def _min_residency_s() -> float:
    try:
        return float(os.environ.get(_ENV_MIN_RESIDENCY, _DEFAULT_MIN_RESIDENCY_S))
    except (TypeError, ValueError):
        return _DEFAULT_MIN_RESIDENCY_S


def _models_home() -> str:
    try:
        from hugpy_platform.constants import MODELS_HOME
        return str(MODELS_HOME)
    except Exception:  # noqa: BLE001
        return "/mnt/llm_storage/models"


def _paths_overlap(a: str, b: str) -> bool:
    """True when realpath(a) and realpath(b) are the same dir OR one contains the
    other. Component-aware (commonpath), so /x/models and /x/models2 do NOT
    overlap but /x/models and /x/models/gguf do."""
    if not a or not b:
        return False
    try:
        ra, rb = os.path.realpath(a), os.path.realpath(b)
        if ra == rb:
            return True
        common = os.path.commonpath([ra, rb])
        return common == ra or common == rb
    except (ValueError, OSError):
        return False


def _degenerate_root() -> bool:
    """True when the hot root OVERLAPS the canonical model store (MODELS_HOME) —
    a 'cache of itself' (ae: HUGPY_HOT_CACHE_ROOT == MODEL_HOME).

    THE ae 2026-07-17 DATA-LOSS ROOT CAUSE. When the hot root is the store root,
    hot_path(f)==f (identity), _rebuild_index adopts EVERY canonical model dir as
    a hot 'entry', and _make_room's LRU eviction then rmtree's real model weights
    out of MODELS_HOME to hit the (min-wins-floored) budget — silently deleting
    the store it was meant to accelerate. models_local fell 65→0 as slice-4's
    400 GiB floor turned this eviction aggressive against 672 GiB present.

    In this configuration the tier CANNOT promote or evict without operating on
    the canonical store, so both are disabled (enabled()→False). Serving is
    unaffected: use() already returns under-root paths unchanged (the files are
    already 'hot'), so loads still run straight off MODELS_HOME. The operator's
    remedy is a hot root on a SEPARATE dir/drive; until then the tier is inert
    rather than destructive."""
    root = _root()
    if not root:
        return False
    return _paths_overlap(root, _models_home())


def enabled() -> bool:
    """True only when a hot root is configured, usable (exists + writable), AND
    NOT degenerate (does not overlap MODELS_HOME). A misconfigured or degenerate
    root disables the tier rather than raising into a load — critically, a
    degenerate root must never promote/evict, or it deletes the canonical store
    (the ae data-loss incident; see _degenerate_root)."""
    root = _root()
    if not root:
        return False
    if _degenerate_root():
        return False
    try:
        os.makedirs(root, exist_ok=True)
        return os.path.isdir(root) and os.access(root, os.W_OK)
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Path mapping. rel() mirrors the shared layout under the hot root so a model
# lands at <root>/<family>/<task>/<owner>/<repo>/... — identical to its shared
# subtree, which makes the mapping stable and human-legible.
# --------------------------------------------------------------------------- #
def _rel(path: str) -> str:
    home = _models_home()
    try:
        rel = os.path.relpath(path, home)
        if rel.startswith(".."):
            raise ValueError
        return rel
    except (ValueError, OSError):
        return os.path.basename(path.rstrip(os.sep))


def hot_path(shared_path: str) -> str:
    """The hot-root mirror of a shared file OR dir."""
    return os.path.join(_root(), _rel(shared_path))


def _under_root(path: str) -> bool:
    root = _root()
    if not root:
        return False
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except (ValueError, OSError):
        return False


def _entry_src_dir(shared_path: str) -> str:
    """The shared dir that is the EVICTION UNIT for this model. For a resolved
    GGUF file it's the containing dir (holds the quant + shards + mmproj); for a
    transformers/diffusers model it's the model dir itself."""
    if os.path.isdir(shared_path):
        return os.path.abspath(shared_path)
    return os.path.abspath(os.path.dirname(shared_path))


def _entry_key(shared_path: str) -> str:
    """Stable index key + display name = the MODELS_HOME-relative entry dir."""
    return _rel(_entry_src_dir(shared_path))


# --------------------------------------------------------------------------- #
# File set: the exact bytes a call needs on the hot drive.
#   * GGUF file input  -> the quant, every shard of a split model, + mmproj.
#   * model-dir input  -> every real file under the dir (transformers/diffusers).
# --------------------------------------------------------------------------- #
def _gguf_file_set(src: str) -> list[str]:
    out = [src]
    d = os.path.dirname(src)
    base = os.path.basename(src)
    m = re.search(r"-\d{5}-of-(\d{5})\.gguf$", base, re.IGNORECASE)
    if m:
        stem = base[: m.start()]
        out = glob.glob(os.path.join(d, f"{stem}-*-of-{m.group(1)}.gguf")) + \
            glob.glob(os.path.join(d, f"{stem}-*-of-{m.group(1)}.GGUF"))
    out += glob.glob(os.path.join(d, "*mmproj*.gguf")) + glob.glob(os.path.join(d, "*mmproj*.GGUF"))
    return sorted({f for f in out if os.path.isfile(f)})


def _is_bookkeeping(name: str) -> bool:
    """Transfer bookkeeping / staging remnants — NOT model content.

    ``.chunksums-*.json`` sidecars are verification metadata that belong beside
    the SOURCE; copying them to the hot drive spends the weight budget on
    bookkeeping and makes them look like files needing verification (they have
    no sidecar of their own -> a pointless "UNVERIFIED" line per promote).
    ``.part``/``.state.json`` are a crashed pull's leftovers: promoting them
    would carry a wedge onto the hot drive — the exact class of artifact that
    misled this incident for 32h. Mirrors central's own exclusion list
    (worker_routes.py: ``".chunksums-" in name or name.endswith((".part", …))``)
    and reconcile.py's, so the three agree on what counts as real weight.
    """
    low = name.lower()
    return ".chunksums-" in low or low.endswith((".part", ".state.json"))


def _dir_file_set(d: str) -> list[str]:
    out: list[str] = []
    for root, _sub, files in os.walk(d):
        for f in files:
            if _is_bookkeeping(f):
                continue
            p = os.path.join(root, f)
            if os.path.isfile(p) and not os.path.islink(p):
                out.append(p)
    return sorted(out)


def _file_set(shared_path: str) -> list[str]:
    if os.path.isdir(shared_path):
        return _dir_file_set(shared_path)
    return _gguf_file_set(shared_path)


def _sizes(paths: list[str]) -> int:
    total = 0
    for p in paths:
        try:
            total += os.path.getsize(p)
        except OSError:
            pass
    return total


def is_complete(shared_path: str) -> bool:
    """True iff every file the call needs is on the hot drive with a matching
    size. The size match is the completeness gate: a truncated / in-flight copy
    (the ``.part`` never renames until it verifies) reads as incomplete and
    transparently falls back to the shared array.

    NOTE the size match here is a RESOLUTION gate, not a trust gate — content is
    proven at promote time (_promote -> _verify_staged), which is the only place
    that has the source beside the copy. Re-hashing every file on every call
    would put minutes of IO inside a load."""
    return not incomplete_reason(shared_path)


def incomplete_reason(shared_path: str) -> str:
    """Why the hot copy isn't usable — "" when it IS complete.

    Exists because the incident's cost was diagnostic, not mechanical: a hot
    tree of ``.part`` files read as a bare False, the loader silently fell back,
    and the failure eventually surfaced as diffusers hunting a legacy ``.bin``
    that was never the problem. Naming the real state (staging file present,
    size mismatch, absent) is what turns 32h into 32 seconds.
    """
    files = _file_set(shared_path)
    if not files:
        return "no source files in the shared store"
    for f in files:
        hp = hot_path(f)
        try:
            if os.path.isfile(hp):
                if os.path.getsize(hp) == os.path.getsize(f):
                    continue
                return (f"{os.path.basename(f)}: hot copy is "
                        f"{os.path.getsize(hp)}B vs {os.path.getsize(f)}B in the "
                        f"shared store (incomplete copy)")
            if os.path.isfile(hp + ".part"):
                st = os.stat(hp + ".part")
                return (f"{os.path.basename(f)}: transfer never finished — a "
                        f"staging .part ({st.st_size}B, {(time.time() - st.st_mtime) / 3600:.1f}h "
                        f"old) is present but was never promoted. This model is "
                        f"NOT usable from the hot cache; serving from the shared "
                        f"store instead")
            return f"{os.path.basename(f)}: absent from the hot cache"
        except OSError as exc:
            return f"{os.path.basename(f)}: {exc}"
    return ""


# --------------------------------------------------------------------------- #
# Persistent index. entries[<rel_key>] = {model_key, bytes, last_called,
# promoted_at, kind}. Written atomically; rebuilt by scanning the hot root if
# missing/corrupt.
# --------------------------------------------------------------------------- #
_INDEX: dict = {"version": 1, "entries": {}}
_INDEX_LOADED = False
_INDEX_LOCK = threading.RLock()


def _index_file() -> str:
    return os.path.join(_root(), _INDEX_NAME)


def _load_index() -> None:
    global _INDEX, _INDEX_LOADED
    with _INDEX_LOCK:
        if _INDEX_LOADED:
            return
        path = _index_file()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and isinstance(data.get("entries"), dict):
                _INDEX = {"version": 1, "entries": data["entries"]}
                _INDEX_LOADED = True
                return
            raise ValueError("index shape")
        except (OSError, ValueError, json.JSONDecodeError):
            _INDEX = {"version": 1, "entries": {}}
            _rebuild_index_locked()
            _INDEX_LOADED = True


def _rebuild_index_locked() -> None:
    """Reconstruct the index from what is on the hot root. An entry = a leaf dir
    holding real weight files; last_called/promoted_at seed from newest mtime
    (best available truth) so recency survives an index loss."""
    root = _root()
    if not root or not os.path.isdir(root):
        return
    entries: dict = {}
    for dirpath, _sub, files in os.walk(root):
        real = [f for f in files if not f.startswith(".") and
                os.path.isfile(os.path.join(dirpath, f)) and
                not os.path.islink(os.path.join(dirpath, f))]
        if not real:
            continue
        # Only leaf-ish dirs that directly hold files become entries; a parent
        # that merely contains subdirs is not itself an entry.
        rel = os.path.relpath(dirpath, root)
        if rel == ".":
            continue
        try:
            newest = max(os.path.getmtime(os.path.join(dirpath, f)) for f in real)
            sz = sum(os.path.getsize(os.path.join(dirpath, f)) for f in real)
        except OSError:
            continue
        entries[rel] = {"model_key": rel, "bytes": int(sz),
                        "last_called": float(newest), "promoted_at": float(newest),
                        "kind": "dir"}
    _INDEX["entries"] = entries
    _save_index_locked()
    if entries:
        logger.info("hot_cache: rebuilt index from disk (%d entries, %.1f GiB)",
                    len(entries), sum(e["bytes"] for e in entries.values()) / GiB)


def _save_index_locked() -> None:
    path = _index_file()
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_INDEX, fh)
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("hot_cache: index save failed: %s", exc)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _index_used_bytes() -> int:
    with _INDEX_LOCK:
        return sum(int(e.get("bytes", 0) or 0) for e in _INDEX["entries"].values())


def _stamp_called(key: str, bytes_hint: int = 0, kind: str = "dir") -> None:
    now = time.time()
    with _INDEX_LOCK:
        e = _INDEX["entries"].get(key)
        if e is None:
            e = {"model_key": key, "bytes": int(bytes_hint or 0),
                 "last_called": now, "promoted_at": now, "kind": kind}
            _INDEX["entries"][key] = e
        else:
            e["last_called"] = now
            if bytes_hint:
                e["bytes"] = int(bytes_hint)
        _save_index_locked()


# --------------------------------------------------------------------------- #
# Free space + pin awareness.
# --------------------------------------------------------------------------- #
def _free_bytes() -> int:
    try:
        st = os.statvfs(_root())
        return st.f_bavail * st.f_frsize
    except OSError:
        return 0


# NOTE (operator ruling, 2026-07-25): 📌 PIN HAS NOTHING TO DO WITH EVICTION.
# Pin means exactly two things and no more:
#   1) this model is ALLOCATED to this worker, and
#   2) that allocation SURVIVES a restart ("or nuclear war") — which is its
#      actual purpose: routing persistence.
# It is not a residency lock, not a hot-tier reservation, not a priority. The
# hot tier is an LRU CACHE of the shared store, so its eviction order is
# least-recently-called, full stop. There was previously an `_is_pinned()`
# helper here used as the PRIMARY eviction sort key ("unpinned before pinned"),
# which quietly made pin a protection: on ae that froze ~321 GiB of the 600 GiB
# hot budget behind 15 pinned models of which only 2 had EVER been called,
# while genuinely-hot models competed for the remainder. Deleted, not softened
# — a weaker version of the same idea is the same bug.


# --------------------------------------------------------------------------- #
# Eviction — the operator's FIFO-by-time-since-called, with the anti-thrash
# residency guard. Runs ONLY inside the single promoter thread, so it never
# races a concurrent promotion of the same entry.
# --------------------------------------------------------------------------- #
def _evict_locked(key: str) -> int:
    """Delete a hot entry dir + drop its index row. Returns bytes freed. NEVER
    touches the shared store (only paths under the hot root)."""
    hp = os.path.join(_root(), key)
    freed = int(_INDEX["entries"].get(key, {}).get("bytes", 0) or 0)
    try:
        if _under_root(hp) and os.path.isdir(hp):
            shutil.rmtree(hp)
    except OSError as exc:
        logger.warning("hot_cache: evict failed for %s: %s", key, exc)
    _INDEX["entries"].pop(key, None)
    return freed


def evict_path(shared_path: str) -> int:
    """MRU / TARGETED eviction: drop the hot entry for `shared_path` NOW, whatever
    its last-called age — the inverse of _make_room's least-recently-called LRU.
    A batch caller that is DONE with a model wants its bytes back immediately so
    the NEXT promotion doesn't shed an OLDER model it still needs. Safe:
    _evict_locked only rmtree's under the hot root, never the shared store.
    Returns bytes freed (0 if the model wasn't hot)."""
    try:
        key = _entry_key(shared_path)
    except Exception:
        return 0
    with _INDEX_LOCK:
        freed = _evict_locked(key) if key in _INDEX["entries"] else 0
        if freed:
            _save_index_locked()
    return freed


def evict_model(model_key: str, cfg=None) -> int:
    """evict_path addressed by model_key: resolve the model to its shared-store
    file, then drop its hot entry. Falls back to matching the index by entry
    name when the file can't be resolved. Best-effort; never raises."""
    try:
        from hugpy_engine.serve.serve import _model_file_for
        if cfg is None:
            try:
                from hugpy_engine.config.main import get_model_config
                cfg = get_model_config(model_key)
            except Exception:
                cfg = None
        src = _model_file_for(model_key, cfg)
        if src:
            freed = evict_path(src)
            if freed:
                return freed
    except Exception:
        pass
    with _INDEX_LOCK:                                   # fallback: match the index
        freed = 0
        for k in list(_INDEX["entries"].keys()):
            if k == model_key or os.path.basename(k) == model_key:
                freed += _evict_locked(k)
        if freed:
            _save_index_locked()
        return freed


def _make_room(need: int, keep_key: str) -> bool:
    """Evict least-recently-CALLED entries until `need` fits under the budget AND
    on the filesystem.

    ONLY WHEN THE LIMIT IS HIT (operator, 2026-07-25): ``fits()`` is checked
    first and returns immediately when the incoming model already has room —
    nothing is evicted just because the cache is large. A hot drive sitting at
    580 of 600 GiB is doing its job; it only sheds when a call needs the space.
    When a model that ISN'T on the hot drive is called and there is no room, the
    oldest-called entries go until it fits.

    Anti-thrash: an entry idle < min_residency is not a candidate. 📌pin is NOT
    consulted — it is routing persistence, not a residency lock; 🔒static is the
    only protection. Returns True if room was made, False if the model can't fit
    even after evicting every eligible entry (then the caller SKIPS promotion)."""
    budget = _budget_bytes()
    residency = _min_residency_s()
    now = time.time()

    def fits() -> bool:
        return (_index_used_bytes() + need <= budget) and (_free_bytes() >= need)

    with _INDEX_LOCK:
        if fits():
            return True
        # ── eviction telemetry (operator directive 2026-07-28) ──────────────
        # DISK eviction is part of "the entire process": an operator watching a
        # model go cold wants to see the hot-NVMe copy reaped, not just the VRAM
        # seat. Same stream, tagged tier="hot_cache", its own run scope (this
        # pass has no VRAM headroom pass to belong to unless one is already open
        # on this thread, in which case run_scope nests under it). Opened only
        # past the fits() early return, so a cache with room emits nothing.
        try:
            _tel_scope = _ledger().run_scope()
        except Exception:  # noqa: BLE001
            _tel_scope = None
        if _tel_scope is not None:
            _tel_scope.__enter__()
        try:
            return _make_room_locked(need, keep_key, budget, residency, now, fits)
        finally:
            if _tel_scope is not None:
                try:
                    _tel_scope.__exit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass


def _make_room_locked(need: int, keep_key: str, budget: int, residency: float,
                      now: float, fits) -> bool:
    """The eviction walk itself — called with ``_INDEX_LOCK`` already held.

    Split out of ``_make_room`` only so the telemetry run scope can wrap it
    without re-indenting the walk. Behavior is unchanged."""
    _evt_emit("headroom.start", trigger="hot_cache",
              incoming_model=keep_key, tier="hot_cache",
              need_bytes=int(need), budget_bytes=int(budget))
    _evt_emit("fit.fail", incoming_model=keep_key, tier="hot_cache",
              need_bytes=int(need), free_bytes=int(_free_bytes()),
              used_bytes=int(_index_used_bytes()))
    _tel_evicted: list = []
    candidates = [(k, v) for k, v in _INDEX["entries"].items() if k != keep_key]
    # ── THE SHARED EVICT ORDER (spec assets/evictionflow.html, box 2) ────
    # The hot tier is the THIRD site that must agree with central's preview
    # and the worker's auto-evict, so it imports the same key rather than
    # spelling its own. Its old key was bare ``last_called`` ascending; the
    # shared key's ②/③/④ (idle longest, fewest calls, stable model_key) is a
    # strict refinement of that — same primary order, with real tiebreaks
    # where it previously relied on dict insertion order.
    #
    # Device = the hot DRIVE: as on the storage path, key ① ("pref ==
    # other device first") names VRAM or RAM, neither of which is a disk,
    # so it is a constant here and the order is honest ②/③/④.
    #
    # WALK ONLY — no drop pass. This loop evicts INCREMENTALLY and re-tests
    # ``fits()`` after each real delete (free bytes move for reasons this
    # index does not model), so a precomputed victim set would be a lie.
    # The ORDER is what has to be shared; the stopping rule is this site's.
    from hugpy_engine import eviction as _ev
    # Anti-thrash: the hot tier's OWN residency window, which is the exact
    # shape enacted proposal 2 reuses in ``eviction._partition`` — kept here
    # (rather than delegated) because this window is keyed on last_called,
    # not on a load time this index does not record.
    eligible = [(k, v) for (k, v) in candidates
                if (now - float(v.get("last_called", 0) or 0)) >= residency]
    fresh = len(candidates) - len(eligible)
    # The anti-thrash window is this tier's protection clause; a skipped entry
    # streams with it so the console can show WHY the drive did not shed.
    _eligible_keys = {k for (k, _v) in eligible}
    for _k, _v in candidates:
        if _k in _eligible_keys:
            continue
        _evt_emit("candidate.skip", model_key=_k, tier="hot_cache",
                  incoming_model=keep_key,
                  reason="within min-residency window (anti-thrash)",
                  idle_s=int(now - float(_v.get("last_called", 0) or 0)),
                  bytes=int(_v.get("bytes") or 0))
    eligible.sort(key=lambda kv: _ev.sort_key(
        _ev.EvictUnit(model_key=kv[0],
                     bytes=int(kv[1].get("bytes") or 0) or None,
                     last_call=float(kv[1].get("last_called", 0) or 0),
                     calls=int(kv[1].get("calls") or 0)),
        "disk", now))
    for k, v in eligible:
        idle = now - float(v.get("last_called", 0) or 0)
        _evt_emit("evict.start", model_key=k, tier="hot_cache",
                  incoming_model=keep_key, idle_s=int(idle))
        _t0 = time.time()
        freed = _evict_locked(k)
        logger.info("hot_cache: evicted %s (%.1f GiB, idle %.0f min) to make "
                    "room for %s", k, freed / GiB, idle / 60.0, keep_key)
        _tel_evicted.append(k)
        _evt_emit("evict.done", model_key=k, tier="hot_cache",
                  incoming_model=keep_key, freed_bytes=int(freed),
                  idle_s=int(idle),
                  duration_ms=int((time.time() - _t0) * 1000))
        if fits():
            _save_index_locked()
            _evt_emit("headroom.done", incoming_model=keep_key,
                      tier="hot_cache", evicted=list(_tel_evicted),
                      outcome="fit")
            return True
    _save_index_locked()
    if not fits():
        # SKIP promote is a REFUSAL of the hot copy, not of the load — the model
        # still serves from the shared array. The stream says "refused" about the
        # promotion so the console does not read it as a failed request.
        _evt_emit("headroom.done", incoming_model=keep_key, tier="hot_cache",
                  evicted=list(_tel_evicted), outcome="refused",
                  reason=("recent entries protected by the anti-thrash window"
                          if fresh else
                          "does not fit even after evicting every eligible entry"),
                  note="promotion skipped; the model serves from the shared store")
        if fresh:
            logger.info("hot_cache: SKIP promote %s — need %.1f GiB but only "
                        "%.1f GiB freeable; %d recent entr%s within the "
                        "%.0f-min residency window are protected (anti-thrash)",
                        keep_key, need / GiB, _free_bytes() / GiB, fresh,
                        "y" if fresh == 1 else "ies", residency / 60.0)
        else:
            logger.info("hot_cache: SKIP promote %s — need %.1f GiB, does not "
                        "fit even after evicting all evictable entries",
                        keep_key, need / GiB)
        return False
    _evt_emit("headroom.done", incoming_model=keep_key, tier="hot_cache",
              evicted=list(_tel_evicted), outcome="fit")
    return True


# --------------------------------------------------------------------------- #
# Promotion — single promoter thread draining a queue. Atomic per file
# (.part -> rename); completeness requires every file, so partials never serve.
# --------------------------------------------------------------------------- #
_QUEUE: "queue.Queue[str]" = queue.Queue()
_QUEUED: set = set()          # entry keys queued or in-flight (dedup)
_INFLIGHT: str | None = None
_STATE_LOCK = threading.Lock()
_PROMOTER: threading.Thread | None = None


def _promote(shared_path: str) -> None:
    """Copy a model's file set into the hot cache, evicting LRU first. Returns
    nothing; logs outcomes. Abandons cleanly (removes the .part) if the source
    changes/vanishes mid-copy."""
    # SAFETY (ae data-loss guard): never promote/evict when the hot root overlaps
    # MODELS_HOME — _make_room would rmtree canonical model dirs. use() already
    # gates on enabled() (False here), but the promoter thread is a second entry
    # point, so re-check at the choke point. See _degenerate_root.
    if not enabled():
        return
    key = _entry_key(shared_path)
    files = _file_set(shared_path)
    if not files:
        return
    if is_complete(shared_path):
        _stamp_called(key, bytes_hint=_sizes(files))
        return
    need = _sizes(files)
    if need <= 0:
        return
    budget = _budget_bytes()
    if need > budget:
        logger.info("hot_cache: SKIP promote %s — %.1f GiB exceeds the whole "
                    "%.1f GiB budget", key, need / GiB, budget / GiB)
        return
    if not _make_room(need, keep_key=key):
        return
    try:
        for f in files:
            dst = hot_path(f)
            try:
                if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(f):
                    continue                                   # already present
            except OSError:
                pass
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".part"
            src_size = os.path.getsize(f)
            logger.info("hot_cache: promoting %s -> %s (%.1f GiB)",
                        f, dst, src_size / GiB)
            shutil.copyfile(f, tmp)
            if os.path.getsize(tmp) != src_size:
                os.remove(tmp)
                raise IOError(f"size mismatch copying {f} (source changed mid-copy?)")
            # CONTENT gate, not a length gate. A size check is exactly what let
            # the 2026-07-15 sd-turbo corruption through: the staged vae was
            # 167335342 bytes — the CORRECT size — but the bytes were wrong, so
            # it promoted, then surfaced 32h later as diffusers complaining
            # about a missing legacy .bin (an error naming the wrong file
            # entirely). Verify against the shared store's chunksums sidecar.
            verdict, detail = _verify_staged(f, tmp)
            if verdict == _cv.CORRUPT:
                # Do NOT promote, and do NOT keep the bad bytes: leaving the
                # .part is what wedged ae for 32h and misled the next reader.
                # Dropping it makes the next promote a clean, succeeding retry.
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise IOError(
                    f"CORRUPT TRANSFER of {f} -> {dst}: {detail}. Refusing to "
                    f"promote; staged copy discarded so the next call re-copies. "
                    f"The shared-store source is unchanged.")
            if verdict == _cv.UNVERIFIED:
                # Absent/stale sidecar = no evidence either way. Copy proceeds
                # (see chunksum_verify: blocking here would break every file
                # that has no sidecar), but we say so rather than implying the
                # bytes were checked.
                logger.info("hot_cache: %s promoted UNVERIFIED (%s)",
                            os.path.basename(f), detail)
            os.replace(tmp, dst)
        if is_complete(shared_path):
            _stamp_called(key, bytes_hint=need)
            logger.info("hot_cache: promoted %s (%.1f GiB) — next call is NVMe-hot",
                        key, need / GiB)
        else:
            logger.warning("hot_cache: promote %s finished incomplete — leaving "
                           "shared-served", key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("hot_cache: promote failed for %s: %s", key, exc)


def _promoter_loop() -> None:
    global _INFLIGHT
    while True:
        shared_path = _QUEUE.get()
        key = _entry_key(shared_path)
        with _STATE_LOCK:
            _INFLIGHT = key
        try:
            _promote(shared_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("hot_cache: promoter error for %s: %s", key, exc)
        finally:
            with _STATE_LOCK:
                _INFLIGHT = None
                _QUEUED.discard(key)
            _QUEUE.task_done()


def _ensure_promoter() -> None:
    global _PROMOTER
    with _STATE_LOCK:
        if _PROMOTER is not None and _PROMOTER.is_alive():
            return
        _PROMOTER = threading.Thread(target=_promoter_loop, name="hot-cache-promoter",
                                     daemon=True)
        _PROMOTER.start()


def _enqueue(shared_path: str) -> None:
    key = _entry_key(shared_path)
    with _STATE_LOCK:
        if key in _QUEUED:
            return                                            # already queued/in-flight
        _QUEUED.add(key)
    _ensure_promoter()
    _QUEUE.put(shared_path)


# --------------------------------------------------------------------------- #
# Public read-through resolution.
# --------------------------------------------------------------------------- #
def use(shared_path: str, promote: bool = True) -> str:
    """Resolve the path a loader should open. Returns the HOT copy when complete
    (and stamps it "called" for LRU); otherwise schedules an async promotion and
    returns ``shared_path`` UNCHANGED so the cold load runs off the shared array
    while the hot copy fills for next time. NEVER blocks, NEVER raises, NEVER
    writes/deletes the shared store. Disabled (env unset) -> returns its
    argument unchanged."""
    if not shared_path or not enabled():
        return shared_path
    if _under_root(shared_path):
        return shared_path                                    # already a hot path
    try:
        _load_index()
        why = incomplete_reason(shared_path)
        if not why:
            _stamp_called(_entry_key(shared_path), bytes_hint=_sizes(_file_set(shared_path)))
            return hot_path(shared_path)
        # Say WHY we're serving cold. A wedged .part used to be indistinguishable
        # from "not promoted yet" in the logs, which is how a 32h-stale staging
        # file stayed invisible until a loader misreported it.
        logger.info("hot_cache: serving %s from the shared store — %s",
                    _entry_key(shared_path), why)
        if promote:
            _enqueue(shared_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("hot_cache: use() fell back to shared for %s: %s", shared_path, exc)
    return shared_path


# --------------------------------------------------------------------------- #
# Stale .part surfacing. A wedged staging file is junk that MISLEADS the next
# reader — the sd-turbo tree sat as .part for 32h and the eventual error blamed
# a missing .bin. We SURFACE them (heartbeat/status) rather than auto-deleting:
# a .part younger than the threshold may be an in-flight copy on another
# thread/process, and this module must never race a live transfer. Reaping is
# left to the promoter, which already discards its own .part on a failed verify
# and re-copies from scratch — so a surfaced stale .part is diagnostic, not a
# leak. Scanning is confined to the hot root (_under_root); the shared store's
# real weights are never touched.
# --------------------------------------------------------------------------- #
_STALE_PART_S = 3600.0        # 1h: far beyond any legitimate single-file copy


def stale_parts(older_than_s: float = _STALE_PART_S) -> list[dict]:
    """Wedged ``.part`` staging files under the hot root, oldest first."""
    root = _root()
    if not root or not os.path.isdir(root):
        return []
    now = time.time()
    out: list[dict] = []
    for dirpath, _sub, files in os.walk(root):
        for f in files:
            if not f.endswith(".part"):
                continue
            p = os.path.join(dirpath, f)
            if not _under_root(p):
                continue
            try:
                st = os.stat(p)
            except OSError:
                continue
            age = now - st.st_mtime
            if age < older_than_s:
                continue
            out.append({"path": p, "rel": os.path.relpath(p, root),
                        "bytes": int(st.st_size), "age_s": round(age, 1)})
    out.sort(key=lambda e: e["age_s"], reverse=True)
    return out


def status() -> dict:
    """Honest hot-cache overview for the worker storage view / console."""
    if not enabled():
        # Name the degenerate case explicitly (hot root == store) so the
        # heartbeat shows WHY the tier is off rather than a bare disabled — this
        # is the ae misconfiguration that caused data loss before the guard.
        if _degenerate_root():
            return {"enabled": False, "disabled_reason": "degenerate_root",
                    "root": _root(), "models_home": _models_home(),
                    "detail": ("hot root overlaps MODELS_HOME — promotion/eviction "
                               "disabled so the tier cannot delete the canonical "
                               "store; point HUGPY_HOT_CACHE_ROOT at a separate "
                               "dir/drive to enable hot-caching")}
        return {"enabled": False}
    try:
        _load_index()
        with _INDEX_LOCK:
            entries = [
                {"model_key": e.get("model_key", k), "bytes": int(e.get("bytes", 0) or 0),
                 "last_called": float(e.get("last_called", 0) or 0),
                 "promoted_at": float(e.get("promoted_at", 0) or 0),
                 "kind": e.get("kind", "dir")}
                for k, e in _INDEX["entries"].items()
            ]
        entries.sort(key=lambda m: m["last_called"], reverse=True)
        with _STATE_LOCK:
            promoting = _INFLIGHT
            queued = sorted(_QUEUED - ({_INFLIGHT} if _INFLIGHT else set()))
        # Surfaced so a wedged staging file is VISIBLE on the heartbeat instead
        # of waiting 32h to reappear as a misleading loader error.
        try:
            stale = stale_parts()
        except OSError:
            stale = []
        return {
            "enabled": True,
            "stale_parts": stale,
            "root": _root(),
            "budget_bytes": _budget_bytes(),          # EFFECTIVE (min-wins)
            # Visibility (slice 4): the tier's own declared budget vs the central
            # disk_cache_gib floor, so the operator can see WHY budget_bytes is
            # what it is when the hot root shares the store drive.
            "budget_own_bytes": _own_budget_bytes(),
            "budget_central_disk_cache_gib": _store_disk_cap_gib(),
            "used_bytes": _index_used_bytes(),
            "free_bytes": _free_bytes(),
            "min_residency_s": _min_residency_s(),
            "promoting": promoting,
            "queued": queued,
            "entries": entries,
        }
    except Exception as exc:  # noqa: BLE001
        return {"enabled": True, "root": _root(), "error": str(exc)}
