"""t28 — load-and-learn calibration store (central, SQLite, stdlib-only).

The accuracy stack (0.1.189-0.1.191) produces BOTH halves of a learning signal
on every load: the PREDICTION (``_incoming_need_detail`` -> weights x1.15 +
GQA-aware KV@ctx%) and the MEASURED TRUTH (per-process VRAM / RSS / load
duration in the serving contract). This store closes the loop: workers ship
compact ``calibration_sample`` rows on the heartbeat wire, central persists
them, aggregates a per-model median measured/predicted ratio, and derives a
CONSERVATIVE, clamped correction factor so the static x1.15 fudge becomes a
learned, per-model number with honest provenance.

Design (mirrors ``comms/shared.py`` SqliteMirror — the proven cross-process
idiom): one short-lived connection per op, WAL, best-effort (a store failure
degrades to the static x1.15, never breaks a heartbeat), self-disabling after
MAX_FAILURES. UNLIKE the jobs mirror this DB is DURABLE by default (learned
corrections should survive a restart) — ``HUGPY_CALIBRATION_DB`` if set, else
``$PROJECTS_HOME/calibration.db``.

Only ROBUST RATIOS here (median + spread gate + clamp). Template derivation over
ctx%/quant/engine features (sklearn regression) is the FUTURE stage (t30) — the
sample schema carries the raw features so that stays possible, but this module
adds no sklearn dependency.

Doctrine [[defaults-are-promises]]: the learned path may only make predictions
MORE accurate. A correction is published ONLY when there are >= MIN_SAMPLES
usable observations AND their spread is sane, and it is ALWAYS clamped to
``[CLAMP_LO, CLAMP_HI]``. The master switch ``HUGPY_CALIBRATION`` (default on)
makes the whole layer inert when ``off`` — ``corrections()`` returns ``{}`` so
neither the worker reply nor the central preflight consult the learned number.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import statistics
import threading
import time
from typing import Any, Optional

from hugpy_control.shared import retry_on_emfile

logger = logging.getLogger(__name__)

MAX_FAILURES = 5

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calibration_samples (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id           TEXT,
    model_key           TEXT NOT NULL,
    engine              TEXT,
    -- placement verdict the observation was taken under: full|partial|cpu|
    -- refuse|unknown. ONLY 'full' rows (weights+kv fully GPU-resident) feed the
    -- ratio — a partial/cpu row measures a fraction of `need`, and a refuse row
    -- never loaded, so both would skew the fudge if counted.
    verdict             TEXT,
    ctx_pct             INTEGER,
    needs_weights_bytes INTEGER,
    needs_kv_bytes      INTEGER,
    need_total_bytes    INTEGER,   -- the PREDICTION (== admission `needs_bytes`)
    n_gpu_layers        INTEGER,
    total_layers        INTEGER,
    vram_bytes          INTEGER,   -- the MEASURED truth (per-process nvidia-smi)
    rss_bytes           INTEGER,
    load_seconds        REAL,
    device              TEXT,
    ok                  INTEGER NOT NULL DEFAULT 1,
    ts                  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calib_model ON calibration_samples(model_key, ts);
"""

# Columns accepted from a worker sample (extra keys ignored — additive-safe).
_SAMPLE_COLS = (
    "model_key", "engine", "verdict", "ctx_pct", "needs_weights_bytes",
    "needs_kv_bytes", "need_total_bytes", "n_gpu_layers", "total_layers",
    "vram_bytes", "rss_bytes", "load_seconds", "device", "ok", "ts",
)


# ── tunables (env-overridable so tests + the operator can retune live) ────────
def _enabled() -> bool:
    """Master switch. Default ON. ``off``/``0``/``false``/``no`` makes the whole
    learned layer inert (corrections() -> {}); the static x1.15 stands."""
    return (os.environ.get("HUGPY_CALIBRATION") or "on").strip().lower() not in (
        "0", "off", "false", "no", "")


def _min_samples() -> int:
    try:
        return max(1, int(os.environ.get("HUGPY_CALIBRATION_MIN_SAMPLES", "3")))
    except ValueError:
        return 3


def _max_spread() -> float:
    """Max relative-MAD spread for a correction to be trusted. Above this the
    observations are too noisy — fall back to static."""
    try:
        return max(0.0, float(os.environ.get("HUGPY_CALIBRATION_MAX_SPREAD", "0.35")))
    except ValueError:
        return 0.35


def _clamp_band() -> "tuple[float, float]":
    try:
        lo = float(os.environ.get("HUGPY_CALIBRATION_CLAMP_LO", "0.8"))
    except ValueError:
        lo = 0.8
    try:
        hi = float(os.environ.get("HUGPY_CALIBRATION_CLAMP_HI", "1.5"))
    except ValueError:
        hi = 1.5
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _window() -> int:
    """Newest-N usable observations per model that feed the aggregate (also the
    per-model row-retention cap)."""
    try:
        return max(1, int(os.environ.get("HUGPY_CALIBRATION_WINDOW", "200")))
    except ValueError:
        return 200


def clamp_correction(value: float) -> float:
    lo, hi = _clamp_band()
    return max(lo, min(hi, float(value)))


from hugpy_control.shared import project_db_path

def default_db_path() -> str:
    return project_db_path("HUGPY_CALIBRATION_DB", "calibration.db")


class CalibrationStore:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or default_db_path()
        self._failures = 0
        self._disabled = False
        self._init_lock = threading.Lock()
        self._initialized = False

    # -- plumbing ------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        # Retry the store-open past the restart-burst EMFILE (see
        # comms.shared.retry_on_emfile) before running the handle-local PRAGMAs.
        from hugpy_control.shared import connect_wal
        return connect_wal(self.path, retry=retry_on_emfile)

    def _ensure(self) -> bool:
        from hugpy_control.shared import ensure_store_schema
        return ensure_store_schema(self, _SCHEMA, script=True)

    def _note_failure(self, op: str, exc: Exception) -> None:
        self._failures += 1
        if self._failures >= MAX_FAILURES and not self._disabled:
            self._disabled = True
            logger.error("calibration store DISABLED after %d failures "
                         "(last: %s during %s) — need-pricing falls back to the "
                         "static x1.15 until restart", self._failures, exc, op)
        else:
            logger.warning("calibration store %s failed: %s", op, exc)

    def _ok(self) -> None:
        self._failures = 0

    # -- writes --------------------------------------------------------------
    def record(self, worker_id: Optional[str], sample: dict[str, Any]) -> None:
        self.record_many(worker_id, [sample])

    def record_many(self, worker_id: Optional[str],
                    samples: "list[dict[str, Any]] | None") -> int:
        """Persist worker calibration samples. Best-effort: a bad row is skipped,
        a store error degrades to a no-op. Returns the count actually written."""
        if not samples or not self._ensure():
            return 0
        rows = []
        now = time.time()
        for s in samples:
            if not isinstance(s, dict) or not s.get("model_key"):
                continue
            rows.append((
                worker_id,
                str(s.get("model_key")),
                _s(s.get("engine")),
                _s(s.get("verdict")),
                _i(s.get("ctx_pct")),
                _i(s.get("needs_weights_bytes")),
                _i(s.get("needs_kv_bytes")),
                _i(s.get("need_total_bytes")),
                _i(s.get("n_gpu_layers")),
                _i(s.get("total_layers")),
                _i(s.get("vram_bytes")),
                _i(s.get("rss_bytes")),
                _f(s.get("load_seconds")),
                _s(s.get("device")),
                0 if s.get("ok") is False else 1,
                _f(s.get("ts")) or now,
            ))
        if not rows:
            return 0
        try:
            with self._connect() as conn:
                conn.executemany(
                    "INSERT INTO calibration_samples "
                    "(worker_id, model_key, engine, verdict, ctx_pct, "
                    " needs_weights_bytes, needs_kv_bytes, need_total_bytes, "
                    " n_gpu_layers, total_layers, vram_bytes, rss_bytes, "
                    " load_seconds, device, ok, ts) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            self._ok()
            # Opportunistic retention: cap the per-model row count so the store
            # can't grow without bound on a long-lived central.
            for mk in {r[1] for r in rows}:
                self._trim_model(mk)
            return len(rows)
        except Exception as exc:  # noqa: BLE001
            self._note_failure("record_many", exc)
            return 0

    def _trim_model(self, model_key: str) -> None:
        """Keep only the newest ``_window()`` rows for a model."""
        cap = _window()
        try:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM calibration_samples WHERE model_key=? AND id NOT IN "
                    "(SELECT id FROM calibration_samples WHERE model_key=? "
                    " ORDER BY ts DESC LIMIT ?)",
                    (model_key, model_key, cap))
        except Exception as exc:  # noqa: BLE001 — retention is housekeeping, never fatal
            logger.debug("calibration trim(%s) failed: %s", model_key, exc)

    # -- aggregation ---------------------------------------------------------
    def _usable_ratios(self, conn, model_key: str) -> "list[float]":
        """Newest-window ratios (measured VRAM / predicted need) for the rows that
        legitimately calibrate the fudge: a SUCCESSFUL, verdict='full' load with a
        positive measured VRAM and a positive prediction. Partial/cpu/refuse rows
        are deliberately excluded (they don't measure the full need)."""
        cur = conn.execute(
            "SELECT need_total_bytes, vram_bytes FROM calibration_samples "
            "WHERE model_key=? AND ok=1 AND verdict='full' "
            "  AND vram_bytes IS NOT NULL AND vram_bytes>0 "
            "  AND need_total_bytes IS NOT NULL AND need_total_bytes>0 "
            "ORDER BY ts DESC LIMIT ?", (model_key, _window()))
        out = []
        for need, vram in cur.fetchall():
            try:
                out.append(float(vram) / float(need))
            except (TypeError, ValueError, ZeroDivisionError):
                continue
        return out

    @staticmethod
    def _spread(ratios: "list[float]", median: float) -> float:
        """Relative median absolute deviation — 0.0 == perfectly consistent."""
        if not ratios or median <= 0:
            return float("inf")
        mad = statistics.median([abs(r - median) for r in ratios])
        return mad / median

    def aggregate(self, model_key: str) -> "dict | None":
        """Per-model aggregate: sample counts, median ratio, spread, and the
        gate-passing clamped correction (None when not enough / too noisy).
        Read-only; None on a store error or a model with no usable rows."""
        if not self._ensure():
            return None
        try:
            with self._connect() as conn:
                ratios = self._usable_ratios(conn, model_key)
                total = conn.execute(
                    "SELECT COUNT(*), MAX(ts) FROM calibration_samples "
                    "WHERE model_key=?", (model_key,)).fetchone()
            self._ok()
        except Exception as exc:  # noqa: BLE001
            self._note_failure("aggregate", exc)
            return None
        sample_count = int((total or [0])[0] or 0)
        last_ts = (total or [0, None])[1]
        usable = len(ratios)
        if usable == 0:
            return {"model_key": model_key, "sample_count": sample_count,
                    "usable_count": 0, "median_ratio": None, "spread": None,
                    "correction": None, "gated": False, "last_ts": last_ts}
        median = statistics.median(ratios)
        spread = self._spread(ratios, median)
        gated = (usable >= _min_samples() and spread <= _max_spread())
        correction = clamp_correction(median) if gated else None
        return {"model_key": model_key, "sample_count": sample_count,
                "usable_count": usable, "median_ratio": round(median, 4),
                "spread": (round(spread, 4) if spread != float("inf") else None),
                "correction": (round(correction, 4) if correction is not None else None),
                "gated": bool(gated), "last_ts": last_ts}

    def _models(self, conn) -> "list[str]":
        return [r[0] for r in conn.execute(
            "SELECT DISTINCT model_key FROM calibration_samples").fetchall()]

    def corrections(self, model_keys: "list[str] | None" = None) -> "dict[str, dict]":
        """The published, gate-passing corrections keyed by model_key —
        ``{model_key: {"correction", "median_ratio", "spread", "count"}}``.

        Returns ``{}`` when the master switch is off (the whole layer inert) so
        neither the worker reply nor the central preflight ever consult a learned
        number while disabled. ``model_keys`` restricts to a relevant set (the
        worker's models); None = every model with data."""
        if not _enabled() or not self._ensure():
            return {}
        try:
            with self._connect() as conn:
                keys = model_keys if model_keys is not None else self._models(conn)
        except Exception as exc:  # noqa: BLE001
            self._note_failure("corrections", exc)
            return {}
        out: dict[str, dict] = {}
        for mk in keys:
            agg = self.aggregate(mk)
            if agg and agg.get("gated") and agg.get("correction") is not None:
                out[mk] = {"correction": agg["correction"],
                           "median_ratio": agg["median_ratio"],
                           "spread": agg["spread"],
                           "count": agg["usable_count"]}
        return out

    def correction_for(self, model_key: str) -> "float | None":
        """The clamped learned correction for one model, or None (static stands).
        Honors the master switch."""
        return (self.corrections([model_key]) or {}).get(model_key, {}).get("correction")

    def table(self) -> "list[dict]":
        """Per-model calibration rows for ``GET /llm/calibration`` — every model
        with samples, gated or not (introspection shows what WOULD be learned even
        when the switch is off). Sorted by usable_count desc."""
        if not self._ensure():
            return []
        try:
            with self._connect() as conn:
                models = self._models(conn)
        except Exception as exc:  # noqa: BLE001
            self._note_failure("table", exc)
            return []
        rows = [self.aggregate(mk) for mk in models]
        rows = [r for r in rows if r]
        rows.sort(key=lambda r: (r.get("usable_count") or 0), reverse=True)
        return rows


def _i(v: Any) -> "int | None":
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _f(v: Any) -> "float | None":
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _s(v: Any) -> "str | None":
    return None if v is None else str(v)


# One process-wide store (central). Best-effort throughout; safe to import from
# any surface (stdlib-only, no cycles).
calibration_store = CalibrationStore()


def record_samples(worker_id: Optional[str],
                   samples: "list[dict] | None") -> int:
    return calibration_store.record_many(worker_id, samples)


def corrections_for(model_keys: "list[str] | None" = None) -> "dict[str, dict]":
    return calibration_store.corrections(model_keys)


def calibration_table() -> "list[dict]":
    return calibration_store.table()
