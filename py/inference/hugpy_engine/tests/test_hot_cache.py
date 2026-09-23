"""HOT-CACHE TIER regression — the box-local NVMe LRU of the MAIN catalog.

Covers the operator's doctrine end to end, on tmpdirs (no real NVMe needed):
  * unset env  -> byte-identical behaviour (use() returns its argument).
  * read-through resolution prefers a COMPLETE hot copy; an incomplete/partial
    (size-mismatched) hot copy is ignored -> serves the shared path.
  * promote-on-call is async (background promoter thread) + atomic (.part ->
    rename, no leftover temp) and produces a complete, resolvable hot copy.
  * eviction order = least-recently-CALLED first, freeing enough space; budget
    respected. PURE LRU — 📌pin is not an eviction input (operator, 2026-07-25:
    pin is routing persistence only; 🔒static residency is the sole protection).
  * ANTI-THRASH: two big models alternating do NOT churn — the second does NOT
    evict the first within the residency window, but DOES once it expires.
  * gguf file-set semantics (quant + shards + mmproj) promote as one entry.
  * the shared store is NEVER written or deleted.
  * the JSON index rebuilds from a disk scan when missing/corrupt.

Runs like the other tests here: venv/bin/python tests/test_hot_cache.py
"""
import logging
# (module-level logging.disable removed: it leaked into every later test under pytest)

import os
import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib

hc = importlib.import_module("hugpy_engine.serve.hot_cache")

# Real implementations, restored by _reset() so a monkeypatch never leaks between
# cases (a small budget from one test must not silently disable the next).
_REAL_BUDGET = hc._budget_bytes
_REAL_FREE = hc._free_bytes

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


def test_hot_cache():
    """Converted from the script-style suite: every `check` is an assert."""


    # --------------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------------- #
    def _mk(root, rel, files):
        """Create a model dir <root>/<rel> holding {name: size_bytes}."""
        d = os.path.join(root, rel)
        for name, size in files.items():
            p = os.path.join(d, name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(b"x" * size)
        return d


    def _snapshot(root):
        out = {}
        for dp, _sub, fs in os.walk(root):
            for f in fs:
                p = os.path.join(dp, f)
                st = os.stat(p)
                out[os.path.relpath(p, root)] = st.st_size
        return out


    def _has_part(root):
        for _dp, _sub, fs in os.walk(root):
            if any(f.endswith(".part") for f in fs):
                return True
        return False


    def _reset(budget=None, residency=0.0, freeb=10 ** 15):
        """Fresh shared+hot tmp roots + clean module state + deterministic knobs."""
        tmp = tempfile.mkdtemp(prefix="hotcache-")
        shared = os.path.join(tmp, "shared")
        hot = os.path.join(tmp, "hot")
        os.makedirs(shared, exist_ok=True)
        os.makedirs(hot, exist_ok=True)
        os.environ["HUGPY_HOT_CACHE_ROOT"] = hot
        hc._models_home = lambda: shared
        hc._min_residency_s = lambda: residency
        hc._free_bytes = (lambda: freeb) if freeb is not None else _REAL_FREE
        hc._budget_bytes = (lambda: budget) if budget is not None else _REAL_BUDGET
        with hc._INDEX_LOCK:
            hc._INDEX = {"version": 1, "entries": {}}
            hc._INDEX_LOADED = False
        with hc._STATE_LOCK:
            hc._QUEUED.clear()
        return tmp, shared, hot


    def _promote_sync(path):
        """use() (enqueues) then block on the single promoter draining the queue."""
        hc.use(path)
        hc._QUEUE.join()


    # --------------------------------------------------------------------------- #
    # 1. unset env -> byte-identical
    # --------------------------------------------------------------------------- #
    os.environ.pop("HUGPY_HOT_CACHE_ROOT", None)
    p = "/mnt/llm_storage/models/fam/task/owner/repo/model.gguf"
    check("unset env: disabled", hc.enabled() is False)
    check("unset env: use() returns argument unchanged", hc.use(p) == p)
    check("unset env: status disabled", hc.status() == {"enabled": False})


    # --------------------------------------------------------------------------- #
    # 2. enabled + read-through prefers a complete hot copy
    # --------------------------------------------------------------------------- #
    tmp, shared, hot = _reset()
    d = _mk(shared, "fam/txt/acme/m1", {"config.json": 10, "model.safetensors": 2000})
    check("enabled when root set", hc.enabled() is True)
    # cold: not in hot yet -> returns shared, schedules promotion
    r = hc.use(d)
    check("cold call serves the shared dir", r == d)
    hc._QUEUE.join()
    check("promotion produced a complete hot copy", hc.is_complete(d) is True)
    check("no .part temp left after atomic promote", _has_part(hot) is False)
    # warm: now prefers the hot copy
    r2 = hc.use(d)
    check("warm call serves the hot dir", r2 == hc.hot_path(d))
    key = hc._entry_key(d)
    check("index records the entry", key in hc._INDEX["entries"])
    check("index last_called stamped", hc._INDEX["entries"][key]["last_called"] > 0)
    shutil.rmtree(tmp, ignore_errors=True)


    # --------------------------------------------------------------------------- #
    # 3. incomplete / partial hot copy is ignored
    # --------------------------------------------------------------------------- #
    tmp, shared, hot = _reset()
    d = _mk(shared, "fam/txt/acme/m2", {"config.json": 10, "model.safetensors": 2000})
    _promote_sync(d)
    check("partial: promoted complete first", hc.is_complete(d) is True)
    # corrupt the hot copy: truncate one file (size mismatch)
    victim = os.path.join(hc.hot_path(d), "model.safetensors")
    with open(victim, "wb") as fh:
        fh.write(b"x" * 5)
    check("partial: size-mismatched hot copy reads incomplete", hc.is_complete(d) is False)
    # promote=False: the default would kick the async promoter at the incomplete
    # copy, racing the delete + is_complete read below (flaked on CI).
    check("partial: incomplete hot copy falls back to shared", hc.use(d, promote=False) == d)
    # a wholly-missing file also reads incomplete
    os.remove(victim)
    check("partial: missing hot file reads incomplete", hc.is_complete(d) is False)
    shutil.rmtree(tmp, ignore_errors=True)


    # --------------------------------------------------------------------------- #
    # 4. promote-on-call is async + atomic; shared store untouched
    # --------------------------------------------------------------------------- #
    tmp, shared, hot = _reset()
    d = _mk(shared, "fam/txt/acme/m3", {"config.json": 10, "a.safetensors": 1500,
                                        "b/b.safetensors": 1500})
    before = _snapshot(shared)
    first = hc.use(d)                          # returns immediately (async)
    check("async: cold use() does not block-copy (returns shared)", first == d)
    hc._QUEUE.join()
    check("async: hot copy complete after promoter drains", hc.is_complete(d) is True)
    check("async: nested-file layout mirrored", os.path.isfile(os.path.join(hc.hot_path(d), "b", "b.safetensors")))
    check("async: no .part temp survives", _has_part(hot) is False)
    check("shared store byte-identical after promote", _snapshot(shared) == before)
    shutil.rmtree(tmp, ignore_errors=True)


    # --------------------------------------------------------------------------- #
    # 5. eviction: least-recently-CALLED first, frees enough, budget respected
    # --------------------------------------------------------------------------- #
    # Each model = 1000 bytes; budget = 2500 holds two, a third forces one eviction.
    tmp, shared, hot = _reset(budget=2500, residency=0.0)
    a = _mk(shared, "fam/txt/acme/A", {"w.safetensors": 1000})
    b = _mk(shared, "fam/txt/acme/B", {"w.safetensors": 1000})
    c = _mk(shared, "fam/txt/acme/C", {"w.safetensors": 1000})
    _promote_sync(a)
    _promote_sync(b)
    check("evict: A and B both hot", hc.is_complete(a) and hc.is_complete(b))
    # Make A the least-recently-called (older last_called than B).
    ka, kb = hc._entry_key(a), hc._entry_key(b)
    with hc._INDEX_LOCK:
        hc._INDEX["entries"][ka]["last_called"] = 100.0
        hc._INDEX["entries"][kb]["last_called"] = 200.0
        hc._save_index_locked()
    before = _snapshot(shared)
    _promote_sync(c)                           # needs room -> evicts LRU (A)
    check("evict: C promoted", hc.is_complete(c) is True)
    check("evict: LRU (A) evicted", hc.is_complete(a) is False)
    check("evict: more-recent (B) kept", hc.is_complete(b) is True)
    check("evict: A dropped from index", hc._entry_key(a) not in hc._INDEX["entries"])
    check("budget respected after eviction", hc._index_used_bytes() <= 2500)
    check("evict: shared store untouched", _snapshot(shared) == before)
    shutil.rmtree(tmp, ignore_errors=True)


    # --------------------------------------------------------------------------- #
    # 6. eviction is PURE LRU — 📌pin is NOT an input (operator ruling 2026-07-25)
    # --------------------------------------------------------------------------- #
    # This case previously asserted the OPPOSITE ("pinned entries evict LAST"), and
    # an `_is_pinned()` helper existed solely to make that true. The operator ruled
    # that out: _"pin routing has nothing to do with eviction... the only eviction
    # protection is 'static' residency"_. Pin means exactly two things — this model
    # is ALLOCATED to this worker, and that allocation SURVIVES a restart. It is
    # routing persistence, never a resource lock.
    #
    # Why it mattered: on ae the pinned-last order froze ~321 GiB of the 600 GiB hot
    # NVMe budget behind 15 pinned models of which only 2 had EVER been called,
    # while genuinely-hot models fought over the remainder. The fast drive was being
    # spent on weight nobody asked for.
    #
    # So: A is the LRU and is ALSO pinned. Pure LRU must evict A anyway.
    tmp, shared, hot = _reset(budget=2500, residency=0.0)
    a = _mk(shared, "fam/txt/acme/PA", {"w.safetensors": 1000})
    b = _mk(shared, "fam/txt/acme/PB", {"w.safetensors": 1000})
    c = _mk(shared, "fam/txt/acme/PC", {"w.safetensors": 1000})
    _promote_sync(a)
    _promote_sync(b)
    ka, kb = hc._entry_key(a), hc._entry_key(b)
    with hc._INDEX_LOCK:
        hc._INDEX["entries"][ka]["last_called"] = 100.0   # A is the LRU
        hc._INDEX["entries"][kb]["last_called"] = 200.0
        hc._save_index_locked()
    # No pin monkeypatch: there is no longer a pin hook to patch. Even if the worker
    # considered A pinned, eviction must not consult it.
    _promote_sync(c)
    check("pure-LRU: the LRU (A) is evicted even though it is pinned",
          hc.is_complete(a) is False)
    check("pure-LRU: the more-recently-called (B) is kept", hc.is_complete(b) is True)
    check("pure-LRU: C promoted", hc.is_complete(c) is True)
    check("pure-LRU: no pin hook survives in the eviction path",
          not hasattr(hc, "_is_pinned"))
    shutil.rmtree(tmp, ignore_errors=True)


    # --------------------------------------------------------------------------- #
    # 7. ANTI-THRASH: two big models alternating do not churn within residency
    # --------------------------------------------------------------------------- #
    # Budget fits exactly ONE model (1500 > 1000, < 2000). Large residency window.
    tmp, shared, hot = _reset(budget=1500, residency=10000.0)
    a = _mk(shared, "fam/txt/acme/HA", {"w.safetensors": 1000})
    b = _mk(shared, "fam/txt/acme/HB", {"w.safetensors": 1000})
    _promote_sync(a)
    check("anti-thrash: A hot", hc.is_complete(a) is True)
    # B called while A is fresh -> must evict A to fit, but A is within residency ->
    # promotion is SKIPPED; A stays hot, B serves from shared.
    r = hc.use(b)
    hc._QUEUE.join()
    check("anti-thrash: B call serves shared", r == b)
    check("anti-thrash: A NOT evicted within residency", hc.is_complete(a) is True)
    check("anti-thrash: B NOT promoted within residency", hc.is_complete(b) is False)
    # Now A's activity genuinely ages out of the residency window.
    with hc._INDEX_LOCK:
        hc._INDEX["entries"][hc._entry_key(a)]["last_called"] = 1.0   # long idle
        hc._save_index_locked()
    _promote_sync(b)
    check("anti-thrash: after residency expires, B promoted", hc.is_complete(b) is True)
    check("anti-thrash: stale A now evicted", hc.is_complete(a) is False)
    shutil.rmtree(tmp, ignore_errors=True)


    # --------------------------------------------------------------------------- #
    # 8. gguf file-set: quant + shards + mmproj promote as ONE entry
    # --------------------------------------------------------------------------- #
    tmp, shared, hot = _reset()
    gdir = _mk(shared, "fam/gguf/acme/G", {
        "model-00001-of-00002.gguf": 800,
        "model-00002-of-00002.gguf": 800,
        "mmproj-model.gguf": 300,
        "README.md": 50,                       # not part of the gguf file set
    })
    gguf = os.path.join(gdir, "model-00001-of-00002.gguf")
    fs = hc._file_set(gguf)
    check("gguf file_set: 2 shards + mmproj (README excluded)",
          sorted(os.path.basename(f) for f in fs) ==
          ["mmproj-model.gguf", "model-00001-of-00002.gguf", "model-00002-of-00002.gguf"])
    _promote_sync(gguf)
    check("gguf: resolved from the file resolves to a complete hot copy", hc.is_complete(gguf) is True)
    check("gguf: shard 2 promoted", os.path.isfile(os.path.join(hc.hot_path(gdir), "model-00002-of-00002.gguf")))
    check("gguf: mmproj promoted", os.path.isfile(os.path.join(hc.hot_path(gdir), "mmproj-model.gguf")))
    check("gguf: warm call returns hot gguf path", hc.use(gguf) == hc.hot_path(gguf))
    check("gguf: entry keyed by dir, not file", hc._entry_key(gguf) == hc._rel(gdir))
    shutil.rmtree(tmp, ignore_errors=True)


    # --------------------------------------------------------------------------- #
    # 9. index rebuilds from disk scan when missing / corrupt
    # --------------------------------------------------------------------------- #
    tmp, shared, hot = _reset()
    d = _mk(shared, "fam/txt/acme/RB", {"config.json": 10, "w.safetensors": 2000})
    _promote_sync(d)
    key = hc._entry_key(d)
    # Corrupt the index file and drop the in-memory copy.
    with open(hc._index_file(), "w", encoding="utf-8") as fh:
        fh.write("{ not json")
    with hc._INDEX_LOCK:
        hc._INDEX = {"version": 1, "entries": {}}
        hc._INDEX_LOADED = False
    hc._load_index()
    check("rebuild: corrupt index recovered from disk scan", key in hc._INDEX["entries"])
    check("rebuild: entry bytes recomputed", hc._INDEX["entries"][key]["bytes"] == 2010)
    # and it is usable again
    check("rebuild: hot copy still resolves after rebuild", hc.use(d) == hc.hot_path(d))
    shutil.rmtree(tmp, ignore_errors=True)


    # --------------------------------------------------------------------------- #
    # 10. status() observability shape
    # --------------------------------------------------------------------------- #
    tmp, shared, hot = _reset(budget=10 * (1 << 30))
    d = _mk(shared, "fam/txt/acme/ST", {"config.json": 10, "w.safetensors": 2000})
    _promote_sync(d)
    st = hc.status()
    check("status: enabled", st.get("enabled") is True)
    check("status: reports root", st.get("root") == hot)
    check("status: budget + used present", "budget_bytes" in st and st.get("used_bytes") == 2010)
    check("status: entry carries last_called", st["entries"] and "last_called" in st["entries"][0])
    shutil.rmtree(tmp, ignore_errors=True)


    print(f"\nALL HOT-CACHE CHECKS PASSED ({ok})")
