"""Worker-side ROLLING AGGREGATE — the worker aggregates its own facts.

OPERATOR RULING (2026-07-29): *"API calls to workers are overloading the hugpy
pool quite consistently. Have the workers agg their own datas as much as is
reasonable for the package health"* — *"a rolling json for central to pick up
upon read."*

The failure class this closes is the ``ae-goes-deaf`` one: central (or a
benchmark sweep) fanning per-worker detail polls at the pool, whose 503-storm
starves the heartbeat until the box reads as offline. **Measurement must never
be the load source.** So:

  * **Observe, never act.** Nothing in this module loads a model, touches a
    runner, evicts, or opens a GPU handle. Every number it stores is a number
    the worker ALREADY computed for its own work (a served request it just
    finished, a calibration sample the beat already drained, the RAM/VRAM split
    the beat already sampled). There is **no new measurement path** here — the
    measured-truth helpers landed in 0.1.224 stay the single source.

  * **No new thread.** Updates piggyback on work in flight (``/infer`` finishing,
    a heartbeat tick). The only periodic act is a debounced *flush* of an
    in-memory dict to one small file — no model is touched to produce it.

  * **ROLLING, not a log.** Every collection is bounded: models are capped and
    evicted least-recently-served-first, per-model event history is a short
    ring, and the latency reservoir is a fixed window. The file cannot grow
    without bound, so it can never become the disk problem it was meant to
    diagnose.

  * **Atomic.** Writes go to ``<file>.tmp`` + ``os.replace`` — a reader (the
    ``GET /ops/aggregate`` route, or central through the relay) either sees the
    previous complete document or the next one, never a torn one.

  * **Never raises into a caller.** Every public entry point is wrapped: this is
    telemetry, and telemetry that can break a served request or skip a beat is
    worse than no telemetry. A broken aggregator degrades to silence.

Read path: central pulls **on read** (``GET /llm/workers/<id>/aggregate`` relays
to the worker's ``GET /ops/aggregate``), behind a short central-side cache. The
heartbeat carries only a COMPACT summary — counts plus a digest — so the beat
stays small and central can tell "changed / unchanged" without a fetch.

Off switch: ``HUGPY_WORKER_AGGREGATE=off``.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any, Optional

SCHEMA_VERSION = 1

# ── bounds (this is a ROLLING aggregate; every one of these is a hard cap) ──
MAX_MODELS = 64          # distinct model rows kept; LRU-by-last_touched evicted
MAX_EVENTS = 20          # per-model recent serve/load ring
MAX_LATENCY_SAMPLES = 128  # per-model reservoir p95 is computed from
MAX_SELFTEST = 10        # per-model self-test scores kept
MAX_ERROR_CHARS = 2000   # last_error is VERBATIM but bounded

_DEFAULT_FLUSH_S = 10.0


def _enabled() -> bool:
    return (os.environ.get("HUGPY_WORKER_AGGREGATE") or "").strip().lower() not in (
        "off", "0", "false", "no",
    )


def aggregate_path() -> str:
    """Path of the rolling aggregate file.

    Lives in the worker's existing state dir convention (``_platform.paths``'s
    ``data_dir()``, the same root ``managers/serve/supervisor`` and the pid
    registry use) — no new location is invented. ``HUGPY_WORKER_AGGREGATE_PATH``
    overrides it (tests, and an operator who wants it on another volume)."""
    override = (os.environ.get("HUGPY_WORKER_AGGREGATE_PATH") or "").strip()
    if override:
        return override
    try:
        from hugpy_platform.app_dirs import data_dir
        base = data_dir()
    except Exception:  # noqa: BLE001 — never fail on a path lookup
        base = os.path.join(os.path.expanduser("~"), ".local", "share", "hugpy")
    d = os.path.join(base, "aggregate")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return os.path.join(d, "worker_aggregate.json")


def _flush_interval_s() -> float:
    try:
        return max(1.0, float(os.environ.get("HUGPY_WORKER_AGGREGATE_FLUSH_S") or _DEFAULT_FLUSH_S))
    except (TypeError, ValueError):
        return _DEFAULT_FLUSH_S


def _clip(text: Any) -> Optional[str]:
    """Verbatim, but bounded. The operator's standing want is the REAL error
    text (a summarized error has repeatedly cost a diagnosis), so we keep the
    head and tail and mark the elision rather than paraphrasing anything."""
    if text is None:
        return None
    s = str(text)
    if len(s) <= MAX_ERROR_CHARS:
        return s
    keep = MAX_ERROR_CHARS // 2 - 20
    return s[:keep] + "\n…[elided]…\n" + s[-keep:]


def _p95(samples: list) -> Optional[float]:
    """p95 over the bounded reservoir. Nearest-rank on a sorted copy — no
    dependency, and exact for the window we actually keep."""
    if not samples:
        return None
    xs = sorted(samples)
    idx = int(round(0.95 * (len(xs) - 1)))
    return round(float(xs[idx]), 3)


class WorkerAggregate:
    """In-memory rolling aggregate + its atomic file projection.

    The dict is authoritative; the file is a projection flushed on a debounce.
    A crash therefore loses at most one flush interval of telemetry, which is
    the correct trade: never fsync on the serving path.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._path = path
        self._started_at = time.time()
        self._models: dict[str, dict] = {}
        self._process: dict = {}
        self._counters = {
            "requests": 0, "ok": 0, "fail": 0,
            "loads": 0, "load_failures": 0, "selftests": 0,
        }
        self._last_flush = 0.0
        self._dirty = False
        self._digest = ""
        self._doc_bytes = 0
        self._mtime = 0.0
        # loading-transition tracker (beat-cadence cold-load timing fallback)
        self._loading_since: dict[str, float] = {}

    # ── path ────────────────────────────────────────────────────────────
    @property
    def path(self) -> str:
        if self._path is None:
            self._path = aggregate_path()
        return self._path

    # ── model row ───────────────────────────────────────────────────────
    def _row(self, model_key: str, now: float) -> dict:
        row = self._models.get(model_key)
        if row is None:
            row = {
                "model_key": model_key,
                "requests": 0, "ok": 0, "fail": 0,
                "tokens_out": 0,
                "latency_ms": {"n": 0, "mean": None, "p95": None, "min": None, "max": None},
                "last_error": None, "last_error_at": None,
                "first_served_at": None, "last_served_at": None,
                "loads": {"count": 0, "failures": 0, "last_seconds": None,
                          "mean_seconds": None, "last_at": None,
                          "last_error": None, "last_error_at": None},
                "events": [],
                "selftest": None,
                "_lat": [],          # bounded reservoir (stripped from the doc)
                "_lat_sum": 0.0,
                "_load_sum": 0.0,
                "_touched": now,
            }
            self._models[model_key] = row
            self._evict_models()
        row["_touched"] = now
        return row

    def _evict_models(self) -> None:
        """Bound the model table. Least-recently-TOUCHED goes first — the same
        contention-LRU shape residency uses, so the rows that survive are the
        ones an operator is actually looking at."""
        if len(self._models) <= MAX_MODELS:
            return
        victims = sorted(self._models.items(), key=lambda kv: kv[1].get("_touched") or 0.0)
        for key, _ in victims[: len(self._models) - MAX_MODELS]:
            self._models.pop(key, None)

    @staticmethod
    def _push(row: dict, event: dict) -> None:
        events = row["events"]
        events.append(event)
        if len(events) > MAX_EVENTS:
            del events[: len(events) - MAX_EVENTS]

    # ── recording (all piggyback on work the worker already did) ────────
    def record_serve(self, model_key: Optional[str], *, ok: bool,
                     latency_ms: Optional[float] = None,
                     tokens_out: Optional[int] = None,
                     error: Any = None, task: Optional[str] = None,
                     at: Optional[float] = None) -> None:
        """One finished request. Called from the /infer paths AFTER the response
        is produced — it adds arithmetic, not work."""
        if not _enabled():
            return
        try:
            now = float(at if at is not None else time.time())
            key = str(model_key or "<unknown>")
            with self._lock:
                row = self._row(key, now)
                row["requests"] += 1
                self._counters["requests"] += 1
                if ok:
                    row["ok"] += 1
                    self._counters["ok"] += 1
                else:
                    row["fail"] += 1
                    self._counters["fail"] += 1
                    row["last_error"] = _clip(error)
                    row["last_error_at"] = now
                if tokens_out:
                    row["tokens_out"] += int(tokens_out)
                if latency_ms is not None:
                    ms = float(latency_ms)
                    lat = row["_lat"]
                    lat.append(ms)
                    row["_lat_sum"] += ms
                    if len(lat) > MAX_LATENCY_SAMPLES:
                        row["_lat_sum"] -= lat.pop(0)
                    stat = row["latency_ms"]
                    stat["n"] = row["requests"]
                    stat["mean"] = round(row["_lat_sum"] / max(1, len(lat)), 3)
                    stat["p95"] = _p95(lat)
                    stat["min"] = round(min(lat), 3)
                    stat["max"] = round(max(lat), 3)
                if row["first_served_at"] is None:
                    row["first_served_at"] = now
                row["last_served_at"] = now
                ev = {"kind": "serve", "at": now, "ok": bool(ok)}
                if latency_ms is not None:
                    ev["ms"] = round(float(latency_ms), 3)
                if tokens_out:
                    ev["tokens"] = int(tokens_out)
                if task:
                    ev["task"] = str(task)
                if not ok:
                    ev["error"] = _clip(error)
                self._push(row, ev)
                self._dirty = True
            self.maybe_flush()
        except Exception:  # noqa: BLE001 — telemetry NEVER breaks a served request
            pass

    def record_load(self, model_key: Optional[str], *, seconds: Optional[float] = None,
                    ok: bool = True, error: Any = None,
                    device: Optional[str] = None, engine: Optional[str] = None,
                    at: Optional[float] = None) -> None:
        """A load event. Sourced from data the worker already produced — the
        calibration samples the beat drains (precise ``load_seconds``) and the
        loading→loaded transition the beat already observes (beat-cadence)."""
        if not _enabled():
            return
        try:
            now = float(at if at is not None else time.time())
            key = str(model_key or "<unknown>")
            with self._lock:
                row = self._row(key, now)
                loads = row["loads"]
                if ok:
                    loads["count"] += 1
                    self._counters["loads"] += 1
                    if seconds is not None:
                        loads["last_seconds"] = round(float(seconds), 3)
                        row["_load_sum"] += float(seconds)
                        loads["mean_seconds"] = round(row["_load_sum"] / max(1, loads["count"]), 3)
                else:
                    loads["failures"] += 1
                    self._counters["load_failures"] += 1
                    loads["last_error"] = _clip(error)
                    loads["last_error_at"] = now
                loads["last_at"] = now
                ev = {"kind": "load", "at": now, "ok": bool(ok)}
                if seconds is not None:
                    ev["seconds"] = round(float(seconds), 3)
                if device:
                    ev["device"] = str(device)
                if engine:
                    ev["engine"] = str(engine)
                if not ok:
                    ev["error"] = _clip(error)
                self._push(row, ev)
                self._dirty = True
            self.maybe_flush()
        except Exception:  # noqa: BLE001
            pass

    def ingest_calibration_samples(self, samples) -> int:
        """Turn the calibration rows the BEAT ALREADY DRAINED into load events.

        This is the whole point of reusing them: ``load_seconds``, ``vram_bytes``
        and ``rss_bytes`` are already-measured truth from 0.1.224's helpers, and
        the beat computed them whether or not this aggregate exists. Zero new
        measurement, precise cold-load durations."""
        if not _enabled() or not samples:
            return 0
        n = 0
        try:
            for s in samples:
                if not isinstance(s, dict):
                    continue
                self.record_load(
                    s.get("model_key"),
                    seconds=s.get("load_seconds"),
                    ok=bool(s.get("ok", True)),
                    error=s.get("verdict") if not s.get("ok", True) else None,
                    device=s.get("device"), engine=s.get("engine"),
                    at=s.get("ts"),
                )
                n += 1
        except Exception:  # noqa: BLE001
            pass
        return n

    def observe_loading(self, loading, loaded, at: Optional[float] = None) -> None:
        """Derive cold-load events from the loading→loaded transition the beat
        already reports. Beat-cadence granularity (a calibration sample, when
        present, carries the precise figure and simply records a second, more
        accurate event). Never probes anything to find this out."""
        if not _enabled():
            return
        try:
            now = float(at if at is not None else time.time())
            loading_set = {str(k) for k in (loading or [])}
            loaded_set = {str(k) for k in (loaded or [])}
            with self._lock:
                for key in loading_set:
                    self._loading_since.setdefault(key, now)
                done = [k for k in list(self._loading_since)
                        if k not in loading_set]
                # bound the tracker the same way everything else is bounded
                if len(self._loading_since) > MAX_MODELS:
                    for k in list(self._loading_since)[: len(self._loading_since) - MAX_MODELS]:
                        self._loading_since.pop(k, None)
            for key in done:
                started = self._loading_since.pop(key, None)
                if started is None:
                    continue
                if key in loaded_set:
                    self.record_load(key, seconds=max(0.0, now - started), ok=True, at=now)
                else:
                    # left "loading" without becoming resident — a failed load.
                    # The verbatim cause rides the serve row's last_error when a
                    # request drove it; here we can only state the fact.
                    self.record_load(
                        key, seconds=max(0.0, now - started), ok=False,
                        error="left loading without becoming resident", at=now)
        except Exception:  # noqa: BLE001
            pass

    def record_process_health(self, **facts) -> None:
        """Process health from the beat's OWN samples (rss/ram/vram splits).

        The caller passes numbers it already has — this function must never be
        given a callable that would go measure something."""
        if not _enabled():
            return
        try:
            clean = {k: v for k, v in facts.items() if v is not None}
            if not clean:
                return
            clean["at"] = time.time()
            with self._lock:
                self._process = clean
                self._dirty = True
        except Exception:  # noqa: BLE001
            pass

    def record_selftest(self, model_key: str, score: dict, at: Optional[float] = None) -> None:
        """Append a MECHANICAL aptitude score (see ``worker_agent.aptitude``).
        Bounded history; no judge, no opinion, no load."""
        if not _enabled():
            return
        try:
            now = float(at if at is not None else time.time())
            with self._lock:
                row = self._row(str(model_key), now)
                st = row.get("selftest") or {"runs": 0, "history": [], "last_at": None}
                st["runs"] = int(st.get("runs", 0)) + 1
                st["last_at"] = now
                st["last"] = score
                hist = st.setdefault("history", [])
                hist.append({"at": now, **{k: score.get(k) for k in
                                           ("case_id", "mech_points", "mech_max") if k in score}})
                if len(hist) > MAX_SELFTEST:
                    del hist[: len(hist) - MAX_SELFTEST]
                row["selftest"] = st
                self._counters["selftests"] += 1
                self._dirty = True
        except Exception:  # noqa: BLE001
            pass

    # ── document + flush ────────────────────────────────────────────────
    def document(self) -> dict:
        """The full aggregate as a plain JSON-safe dict. Private ``_``-prefixed
        working fields (the latency reservoir, running sums) are stripped: they
        are how the rolling stats are computed, not facts anyone reads."""
        with self._lock:
            now = time.time()
            models = {}
            for key, row in self._models.items():
                models[key] = {k: v for k, v in row.items() if not k.startswith("_")}
            return {
                "schema_version": SCHEMA_VERSION,
                "generated_at": now,
                "started_at": self._started_at,
                "uptime_s": round(now - self._started_at, 3),
                "counters": dict(self._counters),
                "models": models,
                "process": dict(self._process),
                "selftest": {
                    "enabled": _selftest_enabled(),
                    "runs": self._counters.get("selftests", 0),
                },
                "bounds": {
                    "max_models": MAX_MODELS, "max_events": MAX_EVENTS,
                    "max_latency_samples": MAX_LATENCY_SAMPLES,
                    "max_selftest": MAX_SELFTEST,
                },
            }

    def flush(self) -> bool:
        """Write the document ATOMICALLY (tmp + os.replace).

        A reader never sees a torn file: ``os.replace`` is atomic within a
        filesystem, and the tmp file is created beside the target so it always
        is. Returns True on a successful write."""
        try:
            doc = self.document()
            blob = json.dumps(doc, ensure_ascii=False, separators=(",", ":"), default=str)
            path = self.path
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(blob)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass          # fsync is a nicety here, not a correctness need
            os.replace(tmp, path)
            with self._lock:
                self._digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
                self._doc_bytes = len(blob)
                self._mtime = time.time()
                self._last_flush = self._mtime
                self._dirty = False
            return True
        except Exception:  # noqa: BLE001 — a failed flush must never propagate
            try:
                os.unlink(self.path + ".tmp")
            except OSError:
                pass
            return False

    def maybe_flush(self, force: bool = False) -> bool:
        """Debounced flush. The serving path calls this on every request; it
        writes at most once per interval, so a burst of 500 requests costs one
        file write, not 500."""
        try:
            with self._lock:
                if not self._dirty:
                    return False
                if not force and (time.time() - self._last_flush) < _flush_interval_s():
                    return False
            return self.flush()
        except Exception:  # noqa: BLE001
            return False

    # ── heartbeat rider ─────────────────────────────────────────────────
    def heartbeat_summary(self) -> dict:
        """COMPACT summary for the beat — counts plus a digest, NOT the file.

        Central uses the digest/mtime to decide whether a pull would even show
        it anything new. Keeping the document off the beat is the entire point:
        the beat must stay small enough that it is never the thing that starves.
        """
        try:
            with self._lock:
                last_served = max(
                    [r.get("last_served_at") or 0.0 for r in self._models.values()] or [0.0])
                return {
                    "schema_version": SCHEMA_VERSION,
                    "digest": self._digest or None,
                    "mtime": self._mtime or None,
                    "bytes": self._doc_bytes or None,
                    "models": len(self._models),
                    "requests": self._counters["requests"],
                    "ok": self._counters["ok"],
                    "fail": self._counters["fail"],
                    "loads": self._counters["loads"],
                    "load_failures": self._counters["load_failures"],
                    "last_served_at": last_served or None,
                    "selftest_runs": self._counters.get("selftests", 0),
                    "path": self.path,
                }
        except Exception:  # noqa: BLE001
            return {"schema_version": SCHEMA_VERSION}

    def read_file(self) -> Optional[dict]:
        """Read the flushed document back (what ``GET /ops/aggregate`` serves).
        Falls back to the live in-memory document if the file is missing or
        unreadable — a reader should get facts, not a 404 over a flush race."""
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def reset(self) -> None:
        with self._lock:
            self._models.clear()
            self._process = {}
            self._loading_since.clear()
            for k in self._counters:
                self._counters[k] = 0
            self._started_at = time.time()
            self._last_flush = 0.0
            self._dirty = False
            self._digest = ""
            self._doc_bytes = 0
            self._mtime = 0.0


# ── process-wide singleton ──────────────────────────────────────────────────
_AGG: Optional[WorkerAggregate] = None
_AGG_LOCK = threading.Lock()


def get_aggregate() -> WorkerAggregate:
    global _AGG
    if _AGG is None:
        with _AGG_LOCK:
            if _AGG is None:
                _AGG = WorkerAggregate()
    return _AGG


def reset_aggregate(path: Optional[str] = None) -> WorkerAggregate:
    """Test seam: replace the singleton (optionally pointing at a tmp path)."""
    global _AGG
    with _AGG_LOCK:
        _AGG = WorkerAggregate(path)
    return _AGG


def _selftest_enabled() -> bool:
    """The aptitude self-test ships DARK. Only an explicit operator env turns it
    on — see ``worker_agent.aptitude.selftest``."""
    return (os.environ.get("HUGPY_WORKER_SELFTEST") or "").strip().lower() in ("on", "1", "true", "yes")


# ── module-level conveniences (what agent.py calls) ─────────────────────────
def record_serve(*args, **kwargs) -> None:
    get_aggregate().record_serve(*args, **kwargs)


def record_load(*args, **kwargs) -> None:
    get_aggregate().record_load(*args, **kwargs)


def heartbeat_summary() -> dict:
    return get_aggregate().heartbeat_summary()


def tokens_out_of(result: Any) -> Optional[int]:
    """Completion-token count out of a result envelope, if it states one.

    Tolerant by design and NEVER a fallback estimate: an invented token count is
    worse than none, so an envelope that does not report usage contributes
    nothing to ``tokens_out`` rather than a guessed number."""
    try:
        if not isinstance(result, dict):
            return None
        usage = result.get("usage")
        if isinstance(usage, dict):
            for k in ("completion_tokens", "output_tokens", "tokens_out", "eval_count"):
                v = usage.get(k)
                if isinstance(v, (int, float)) and v >= 0:
                    return int(v)
        for k in ("completion_tokens", "tokens_out", "output_tokens"):
            v = result.get(k)
            if isinstance(v, (int, float)) and v >= 0:
                return int(v)
    except Exception:  # noqa: BLE001
        pass
    return None
