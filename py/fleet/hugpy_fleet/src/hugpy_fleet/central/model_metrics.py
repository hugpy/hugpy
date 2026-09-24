"""MODEL METRICS (2026-08-28) — measured load/serve timings per (model,
variant, worker card), logged as a side effect of every real load and call.
Measure, don't bench: rows exist only because work actually happened.

Operator concept: groups rank their members, but rank alone can't answer
"which member/worker pair answers FASTEST right now?" — that needs

    total_time_to_output = time_to_download + time_to_evict_to_fit
                         + time_to_upload + time_to_output

where upload time and tok/s differ per variant (split | moe | gpu_only_4bit
| ram_only_4bit), per card, and hot vs cold. This store holds the measured
terms as EMAs; the decision-time terms (download presence, evict-to-fit dry
run) are the scheduler's, passed into ``estimate_total_time``. Stage
transitions that displace a model the next stage needs again charge the
displaced model's hot re-upload too — pass it as ``displaced_reupload_s``.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Optional

from hugpy_control.shared import retry_on_emfile

VARIANTS = ("split", "moe", "gpu_only_4bit", "ram_only_4bit")
TEMPERATURES = ("loaded", "unloaded")   # LOADED-at-pick (STATE-MODEL.md #3); never "hot"/"cold" (that named a VRAM condition)

# EMA weight for a new sample. High enough to track a card whose behavior
# drifts (thermals, contention), low enough that one anomalous load doesn't
# repaint the picture.
EMA_ALPHA = float(os.environ.get("HUGPY_MODEL_METRICS_EMA_ALPHA", "0.3") or 0.3)

# The durable compute-action log is NOT pruned by default (2026-09-23): it is
# the log the console reads, the DB has no space shortage, and deleting rows
# destroyed exactly the history an operator comes back for. A positive
# HUGPY_COMPUTE_ACTIONS_MAX_ROWS opts back into the amortized prune; 0/unset
# keeps every row.
try:
    ACTIONS_MAX_ROWS = int(os.environ.get("HUGPY_COMPUTE_ACTIONS_MAX_ROWS", "0") or 0)
except ValueError:
    ACTIONS_MAX_ROWS = 0
ACTIONS_PRUNE_EVERY = 500

# The action vocabulary the durable log speaks. NOT the eviction STAGE names —
# those are the wire contract with the fleet; these are the coarser buckets the
# Metrics panel groups by. ``emit_eviction_event``'s tap maps stage -> action.
ACTION_LOAD = "load"
ACTION_CALL = "call"
ACTION_EVICT = "evict"
ACTION_PROVISION = "provision"
ACTION_FIT_FAIL = "fit_fail"
ACTION_MEMBER_SELECT = "member_select"


def _ema(prev: Optional[float], sample: float, n_prev: int) -> float:
    # First sample IS the estimate; after that, standard exponential blend.
    if prev is None or n_prev <= 0:
        return float(sample)
    return (1.0 - EMA_ALPHA) * float(prev) + EMA_ALPHA * float(sample)


def derive_variant(n_gpu_layers: Any, total_layers: Optional[int] = None,
                   *, moe_capable: bool = False) -> Optional[str]:
    """Derive the VARIANTS member from what the serving contract actually did,
    so the enum can't drift from reality — callers should never pass a variant
    string by hand.

    Inputs mirror the serving contract's own vocabulary: ``n_gpu_layers`` as
    llama.cpp uses it (-1 = all layers on GPU, 0 or the chaos sweep's "off" =
    none), ``total_layers`` when known, ``moe_capable`` from model_physical's
    capability pair. MoE is checked first (MoE fit math is its own regime —
    expert bytes are mmap-eligible, dense partial math doesn't apply). The
    "_4bit" in the gpu/ram-only names is the operator's deployment vocabulary
    (those tiers run quantized on this fleet), not re-derived here. Returns
    None when the split is unknowable — an unknown variant is unrecorded, not
    guessed.
    """
    if moe_capable:
        return "moe"
    if n_gpu_layers in ("off", 0):
        return "ram_only_4bit"
    if n_gpu_layers is None:
        return None
    try:
        n = int(n_gpu_layers)
    except (TypeError, ValueError):
        return None
    if n == -1 or (total_layers is not None and n >= int(total_layers)):
        return "gpu_only_4bit"
    if total_layers is None:
        # A positive layer count with no total is ambiguous (could be all of
        # them) — unrecorded beats misfiled.
        return None
    return "split"


from hugpy_control.shared import project_db_path

def default_db_path() -> str:
    return project_db_path("HUGPY_MODEL_METRICS_DB", "model_metrics.db")


class ModelMetricsStore:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or default_db_path()
        self._disabled = False
        self._init_lock = threading.Lock()
        self._initialized = False
        # Amortized-prune counter for the durable compute-action log.
        self._action_appends = 0

    # -- plumbing ------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        # Retry the store-open past the restart-burst EMFILE (see
        # comms.shared.retry_on_emfile) before running the handle-local PRAGMAs.
        from hugpy_control.shared import connect_wal
        return connect_wal(self.path, retry=retry_on_emfile)

    def _ensure(self) -> bool:
        if self._disabled:
            return False
        if self._initialized:
            return True
        with self._init_lock:
            if self._initialized:
                return True
            try:
                conn = self._connect()
                try:
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS load_metrics ("
                        " model TEXT NOT NULL,"
                        " variant TEXT NOT NULL,"
                        " worker_card TEXT NOT NULL,"
                        " temperature TEXT NOT NULL,"
                        " upload_time_s REAL,"
                        " tok_per_s REAL,"
                        " n_samples INTEGER NOT NULL DEFAULT 0,"
                        " updated_at REAL NOT NULL,"
                        " PRIMARY KEY (model, variant, worker_card,"
                        "              temperature))")
                    # Canon migration (STATE-MODEL.md #3): legacy temperature
                    # values hot/cold named a VRAM condition "hot" — rebucket
                    # to loaded/unloaded (keyed on LOADED-at-pick). Idempotent.
                    conn.execute("UPDATE OR IGNORE load_metrics"
                                 " SET temperature='loaded' WHERE temperature='hot'")
                    conn.execute("UPDATE OR IGNORE load_metrics"
                                 " SET temperature='unloaded' WHERE temperature='cold'")
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS call_metrics ("
                        " model TEXT PRIMARY KEY,"
                        " avg_tok_output_per_call REAL,"
                        " n_calls INTEGER NOT NULL DEFAULT 0,"
                        " updated_at REAL NOT NULL)")
                    # Per-TASK breakdown (2026-09-01) — the same call EMA keyed
                    # by (model, task), so the Metrics panel can show a tok/s
                    # column per task (text-generation, image-text-to-text,
                    # text-to-image, ...). Central is both router and record
                    # keeper, so `task` is in scope at the record site (resolve's
                    # own arg) — no wire change. Additive: the model-only
                    # call_metrics above is untouched; a caller with no task
                    # simply doesn't write here.
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS call_metrics_by_task ("
                        " model TEXT NOT NULL,"
                        " task TEXT NOT NULL,"
                        " avg_tok_output_per_call REAL,"
                        " avg_compute_s REAL,"       # wall-clock per call, EMA
                        " n_calls INTEGER NOT NULL DEFAULT 0,"
                        " updated_at REAL NOT NULL,"
                        " PRIMARY KEY (model, task))")
                    # avg_compute_s added 2026-09-01 for time-aware allocation —
                    # tolerate a pre-existing table from before the column landed.
                    try:
                        conn.execute("ALTER TABLE call_metrics_by_task"
                                     " ADD COLUMN avg_compute_s REAL")
                    except Exception:  # noqa: BLE001 — already present
                        pass
                    # The durable compute-action LOG (2026-09-01) — append-only,
                    # one row per load/call/evict/provision/... with full detail.
                    # Distinct from the EMA rows above: those are the current
                    # estimate, this is the history the Metrics panel reads. The
                    # rowid is the natural newest-first / paging cursor, same
                    # pattern as eviction_events. Columns are only the ones the
                    # readers filter/order by; everything else rides in
                    # detail_json, so a new field needs no migration.
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS compute_actions ("
                        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                        " ts REAL NOT NULL,"
                        " action TEXT NOT NULL,"
                        " model TEXT,"
                        " variant TEXT,"
                        " worker_card TEXT,"
                        " duration_s REAL,"
                        " tok_per_s REAL,"
                        " tokens INTEGER,"
                        " bytes INTEGER,"
                        " outcome TEXT,"
                        " detail_json TEXT)")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS ix_compute_actions_ts"
                        " ON compute_actions(ts)")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS ix_compute_actions_action"
                        " ON compute_actions(action)")
                    conn.commit()
                finally:
                    conn.close()
                self._initialized = True
                return True
            except Exception:  # noqa: BLE001 — metrics must never break serving
                self._disabled = True
                return False

    # -- writes --------------------------------------------------------------
    def record_load(self, model: str, variant: str, worker_card: str,
                    temperature: str, *,
                    upload_time_s: Optional[float] = None,
                    tok_per_s: Optional[float] = None) -> bool:
        """Fold one real load/serve observation into the EMAs.

        Either measurement may be absent (a load that never generated has no
        tok/s yet); the absent one keeps its previous estimate.
        """
        if variant not in VARIANTS or temperature not in TEMPERATURES:
            return False
        if not (model or "").strip() or not (worker_card or "").strip():
            return False
        if not self._ensure():
            return False
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT upload_time_s, tok_per_s, n_samples"
                    " FROM load_metrics WHERE model=? AND variant=?"
                    " AND worker_card=? AND temperature=?",
                    (model, variant, worker_card, temperature)).fetchone()
                prev_up, prev_tok, n = row if row else (None, None, 0)
                new_up = (_ema(prev_up, upload_time_s, n)
                          if upload_time_s is not None else prev_up)
                new_tok = (_ema(prev_tok, tok_per_s, n)
                           if tok_per_s is not None else prev_tok)
                conn.execute(
                    "INSERT INTO load_metrics"
                    " (model, variant, worker_card, temperature,"
                    "  upload_time_s, tok_per_s, n_samples, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(model, variant, worker_card, temperature)"
                    " DO UPDATE SET upload_time_s=excluded.upload_time_s,"
                    "  tok_per_s=excluded.tok_per_s,"
                    "  n_samples=excluded.n_samples,"
                    "  updated_at=excluded.updated_at",
                    (model, variant, worker_card, temperature,
                     new_up, new_tok, n + 1, time.time()))
                conn.commit()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return False
        # Durable log: one 'load' row carrying THIS observation's perf numbers
        # (the EMA rows above are the running estimate; the log is the history
        # the Metrics panel reads). Guarded inside append_action — a log fault
        # never fails the record that already committed.
        self.append_action(
            ACTION_LOAD, model=model, variant=variant, worker_card=worker_card,
            duration_s=upload_time_s, tok_per_s=tok_per_s, outcome=temperature)
        return True

    def record_call(self, model: str, tok_output: float,
                    *, task: Optional[str] = None,
                    compute_s: Optional[float] = None,
                    call: Optional[Dict[str, Any]] = None) -> bool:
        """Fold one completed call into the call EMAs.

        ``task`` (the router's pipeline task, in scope at the record site because
        central both routes and keeps records) additionally folds the same
        observation into the per-(model, task) EMA, so the Metrics panel can show
        a tok-output column per task. ``compute_s`` (the call's wall-clock, from
        the relay's elapsed_s) EMAs an average compute-time per (model, task) —
        the term a time-aware allocator prices a worker/model choice against.
        Absent task -> only the model-level row moves, byte-identical to before.
        """
        if not (model or "").strip():
            return False
        if not self._ensure():
            return False
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT avg_tok_output_per_call, n_calls"
                    " FROM call_metrics WHERE model=?", (model,)).fetchone()
                prev, n = row if row else (None, 0)
                conn.execute(
                    "INSERT INTO call_metrics"
                    " (model, avg_tok_output_per_call, n_calls, updated_at)"
                    " VALUES (?,?,?,?)"
                    " ON CONFLICT(model) DO UPDATE SET"
                    "  avg_tok_output_per_call=excluded.avg_tok_output_per_call,"
                    "  n_calls=excluded.n_calls, updated_at=excluded.updated_at",
                    (model, _ema(prev, tok_output, n), n + 1, time.time()))
                t = (task or "").strip()
                if t:
                    trow = conn.execute(
                        "SELECT avg_tok_output_per_call, avg_compute_s, n_calls"
                        " FROM call_metrics_by_task WHERE model=? AND task=?",
                        (model, t)).fetchone()
                    tprev, tprev_c, tn = trow if trow else (None, None, 0)
                    new_c = (_ema(tprev_c, compute_s, tn)
                             if compute_s is not None else tprev_c)
                    conn.execute(
                        "INSERT INTO call_metrics_by_task"
                        " (model, task, avg_tok_output_per_call, avg_compute_s,"
                        "  n_calls, updated_at)"
                        " VALUES (?,?,?,?,?,?)"
                        " ON CONFLICT(model, task) DO UPDATE SET"
                        "  avg_tok_output_per_call=excluded.avg_tok_output_per_call,"
                        "  avg_compute_s=excluded.avg_compute_s,"
                        "  n_calls=excluded.n_calls, updated_at=excluded.updated_at",
                        (model, t, _ema(tprev, tok_output, tn), new_c,
                         tn + 1, time.time()))
                conn.commit()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return False
        # Durable log: one 'call' row carrying THIS call's own numbers. ``call``
        # (resolvers.remote.per_call_row) supplies worker, wall duration,
        # tokens/duration and the prompt/gen split; without it the row still
        # carries the token count and task, as before.
        c = call if isinstance(call, dict) else {}
        detail = dict(c.get("detail") or {})
        if (task or "").strip():
            detail.setdefault("task", task)
        self.append_action(ACTION_CALL, model=model,
                           worker_card=c.get("worker_card"),
                           duration_s=c.get("duration_s", compute_s if call is not None else None),
                           tok_per_s=c.get("tok_per_s"),
                           tokens=int(tok_output)
                           if tok_output is not None else None,
                           outcome=c.get("outcome"),
                           detail=detail or None)
        return True

    # -- durable compute-action log -----------------------------------------
    def append_action(self, action: str, *,
                      model: Optional[str] = None,
                      variant: Optional[str] = None,
                      worker_card: Optional[str] = None,
                      duration_s: Optional[float] = None,
                      tok_per_s: Optional[float] = None,
                      tokens: Optional[int] = None,
                      bytes: Optional[int] = None,  # noqa: A002 — column name
                      outcome: Optional[str] = None,
                      detail: Optional[Dict[str, Any]] = None,
                      ts: Optional[float] = None) -> bool:
        """Append ONE row to the durable compute-action log. BEST EFFORT, ALWAYS.

        This runs on the hot compute path (it is tapped from the total eviction
        emit and from record_load/record_call), so it mirrors the store's
        fail-open discipline to the letter: a bad field, a wedged file, a
        disabled store — none of them may raise, block, or change a compute
        outcome. It returns True/False for tests and callers who care, and
        swallows everything else.

        ``action`` is the only required field; ``detail`` is any extra JSON the
        stage carried (it is stored as a string, never trusted to serialize).
        """
        act = (action or "").strip()
        if not act:
            return False
        if not self._ensure():
            return False
        try:
            detail_json = None
            if detail:
                try:
                    detail_json = json.dumps(detail, default=str)
                except Exception:  # noqa: BLE001 — a bad detail is not a reason
                    detail_json = None  # …to drop the whole row
            row = (
                float(ts) if ts is not None else time.time(),
                act,
                str(model) if model is not None else None,
                str(variant) if variant is not None else None,
                str(worker_card) if worker_card is not None else None,
                float(duration_s) if duration_s is not None else None,
                float(tok_per_s) if tok_per_s is not None else None,
                int(tokens) if tokens is not None else None,
                int(bytes) if bytes is not None else None,
                str(outcome) if outcome is not None else None,
                detail_json,
            )
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO compute_actions"
                    " (ts, action, model, variant, worker_card, duration_s,"
                    "  tok_per_s, tokens, bytes, outcome, detail_json)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)", row)
                conn.commit()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001 — logging must NEVER break serving
            return False
        # Amortized prune, outside the write's try so a prune fault can't fail
        # the append that already landed.
        self._action_appends += 1
        if self._action_appends >= ACTIONS_PRUNE_EVERY:
            self._action_appends = 0
            self._prune_actions()
        return True

    def _prune_actions(self) -> None:
        """Keep the newest ``ACTIONS_MAX_ROWS`` rows — only when an operator
        set a positive cap (default: keep everything). Best-effort."""
        if ACTIONS_MAX_ROWS <= 0 or not self._ensure():
            return
        try:
            conn = self._connect()
            try:
                conn.execute(
                    "DELETE FROM compute_actions WHERE id <= ("
                    "  SELECT MAX(id) - ? FROM compute_actions)",
                    (ACTIONS_MAX_ROWS,))
                conn.commit()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return

    def recent_actions(self, limit: int = 200, *,
                       since_ts: Optional[float] = None,
                       action: Optional[str] = None,
                       model: Optional[str] = None,
                       worker: Optional[str] = None,
                       outcome: Optional[str] = None) -> list:
        """The durable log, NEWEST FIRST, bounded. Best-effort — returns [] on
        any fault so a broken store shows an empty panel, not a broken one.

        ``worker`` matches the worker NAME (the part before ``:`` in a card,
        e.g. ``ae`` matches ``ae:0``) as well as the full card, so the panel's
        per-worker filter reads naturally.
        """
        if not self._ensure():
            return []
        clauses, args = [], []
        if since_ts is not None:
            clauses.append("ts >= ?")
            args.append(float(since_ts))
        if action:
            clauses.append("action = ?")
            args.append(str(action))
        if model:
            clauses.append("model = ?")
            args.append(str(model))
        if outcome:
            clauses.append("outcome = ?")
            args.append(str(outcome))
        if worker:
            w = str(worker)
            clauses.append("(worker_card = ? OR worker_card LIKE ?)")
            args.extend([w, w + ":%"])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        lim = max(1, min(int(limit or 200), 5000))
        try:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT id, ts, action, model, variant, worker_card,"
                    " duration_s, tok_per_s, tokens, bytes, outcome, detail_json"
                    " FROM compute_actions" + where +
                    " ORDER BY id DESC LIMIT ?", (*args, lim))
                got = cur.fetchall()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return []
        out = []
        for r in got:
            detail = None
            if r[11]:
                try:
                    detail = json.loads(r[11])
                except Exception:  # noqa: BLE001
                    detail = None
            out.append({
                "id": int(r[0]), "ts": r[1], "action": r[2], "model": r[3],
                "variant": r[4], "worker_card": r[5], "duration_s": r[6],
                "tok_per_s": r[7], "tokens": r[8], "bytes": r[9],
                "outcome": r[10], "detail": detail,
            })
        return out

    def get_action(self, action_id: int) -> Optional[Dict[str, Any]]:
        """ONE compute_actions row by id, detail whole. None when absent."""
        if not self._ensure():
            return None
        try:
            conn = self._connect()
            try:
                r = conn.execute(
                    "SELECT id, ts, action, model, variant, worker_card,"
                    " duration_s, tok_per_s, tokens, bytes, outcome, detail_json"
                    " FROM compute_actions WHERE id = ?", (int(action_id),)).fetchone()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return None
        if not r:
            return None
        return _action_row(tuple(r))

    # -- reads ---------------------------------------------------------------
    def get_load(self, model: str, variant: str, worker_card: str,
                 temperature: str) -> Optional[Dict[str, Any]]:
        if not self._ensure():
            return None
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT upload_time_s, tok_per_s, n_samples, updated_at"
                    " FROM load_metrics WHERE model=? AND variant=?"
                    " AND worker_card=? AND temperature=?",
                    (model, variant, worker_card, temperature)).fetchone()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return None
        if not row:
            return None
        return {"upload_time_s": row[0], "tok_per_s": row[1],
                "n_samples": row[2], "updated_at": row[3]}

    def get_call(self, model: str) -> Optional[Dict[str, Any]]:
        if not self._ensure():
            return None
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT avg_tok_output_per_call, n_calls, updated_at"
                    " FROM call_metrics WHERE model=?", (model,)).fetchone()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return None
        if not row:
            return None
        return {"avg_tok_output_per_call": row[0], "n_calls": row[1],
                "updated_at": row[2]}

    def all_calls_by_task(self) -> list:
        """Every per-(model, task) call EMA row, for the Metrics panel's
        tok-output-per-task pivot. Empty list on any fault (never raises)."""
        if not self._ensure():
            return []
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT model, task, avg_tok_output_per_call, avg_compute_s,"
                    " n_calls, updated_at FROM call_metrics_by_task"
                    " ORDER BY model, task").fetchall()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return []
        return [{"model": r[0], "task": r[1],
                 "avg_tok_output_per_call": r[2], "avg_compute_s": r[3],
                 "n_calls": r[4], "updated_at": r[5]} for r in rows]

    def all_loads(self) -> list:
        """Every load_metrics row, for the Metrics panel's cold-load sheet.
        Empty list on any fault. Mirrors PgMetricsStore.all_loads so the read
        path is backend-agnostic."""
        if not self._ensure():
            return []
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT model, variant, worker_card, temperature,"
                    " upload_time_s, tok_per_s, n_samples, updated_at FROM load_metrics"
                    " ORDER BY model, variant, worker_card, temperature").fetchall()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return []
        return [{"model": r[0], "variant": r[1], "worker_card": r[2],
                 "temperature": r[3], "upload_time_s": r[4], "tok_per_s": r[5],
                 "n_samples": r[6], "updated_at": r[7]} for r in rows]

    def all_calls(self) -> list:
        """Every model-level call_metrics EMA row, busiest first. [] on fault."""
        if not self._ensure():
            return []
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT model, avg_tok_output_per_call, n_calls, updated_at"
                    " FROM call_metrics ORDER BY n_calls DESC").fetchall()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return []
        return [{"model": r[0], "avg_tok_output_per_call": r[1],
                 "n_calls": r[2], "updated_at": r[3]} for r in rows]

    # -- scoring -------------------------------------------------------------
    def estimate_total_time(self, model: str, variant: str, worker_card: str,
                            *, temperature: str = "loaded",
                            time_to_download_s: float = 0.0,
                            time_to_evict_s: float = 0.0,
                            displaced_reupload_s: float = 0.0,
                            ) -> Optional[float]:
        """Estimate total_time_to_output for one (member, worker) pair.

        The measured terms come from this store; the decision-time terms are
        the caller's: download is 0 when weights are cached on the worker's
        host, evict comes from the fit-check dry run — which MUST use the
        accounting of the loader that will actually run the member (the slot
        guard over-estimates GGUFs that llama-server's partial offload fits) —
        and ``displaced_reupload_s`` is the hot re-upload of a model this
        transition displaces but the next stage needs again.

        Returns None when the pair has no measurements yet — an unmeasured
        pair is unranked, not free.
        """
        load = self.get_load(model, variant, worker_card, temperature)
        call = self.get_call(model)
        if not load or load["upload_time_s"] is None or not load["tok_per_s"]:
            return None
        if not call or call["avg_tok_output_per_call"] is None:
            return None
        return (float(time_to_download_s) + float(time_to_evict_s)
                + float(load["upload_time_s"])
                + float(call["avg_tok_output_per_call"]) / float(load["tok_per_s"])
                + float(displaced_reupload_s))


model_metrics_store = ModelMetricsStore()

# ── DB arm cutover (2026-09-01) ──────────────────────────────────────────────
# Metrics move onto the toolserver's Postgres (the fleet's DB arm) — no more
# SQLite-on-virtiofs, where WAL checkpoints rewrite the file and reset its times.
# abstract_toolserver.metrics.PgMetricsStore is a method-compatible drop-in (same
# EMA, same four tables incl. call_metrics_by_task.avg_compute_s + compute_actions,
# same "metrics must never break serving" fail-open). Its import is dependency-light
# (the package __init__ is lazy), so it does NOT pull the toolset's flask/geopandas/
# PyQt6/playwright chain into central. The SQLite class above stays as a FALLBACK
# only when the package/DB isn't importable, so import never breaks; in normal
# operation this is Postgres-only. HUGPY_METRICS_BACKEND=sqlite forces the legacy
# file (instant rollback, no code edit). Central runs from the share tree, so a
# restart picks this up — no wheel publish needed.
if os.environ.get("HUGPY_METRICS_BACKEND", "").strip().lower() != "sqlite":
    # The toolserver store reads HUGPY_COMPUTE_ACTIONS_MAX_ROWS at import with a
    # 20000 default (and treats 0 as "default"), pruning the log central writes.
    # Keep every row unless an operator set a cap explicitly (2026-09-23).
    if not os.environ.get("HUGPY_COMPUTE_ACTIONS_MAX_ROWS"):
        os.environ["HUGPY_COMPUTE_ACTIONS_MAX_ROWS"] = str(10 ** 15)
    try:
        from abstract_toolserver.metrics import PgMetricsStore as _PgMetricsStore
        model_metrics_store = _PgMetricsStore()
    except Exception:  # noqa: BLE001 — toolserver absent -> keep SQLite, never break
        pass


def record_loads_from_calibration(worker_name, samples) -> int:
    """The upload_time_s producer — the cold-load half ``_record_model_metrics``
    deliberately leaves to the load path. Folds worker-shipped calibration
    samples that carry a measured ``load_seconds`` into the cold-load EMA.

    The calibration wire is the SINGLE producer (the k119 follow-up's open
    question, resolved as: merge, don't duplicate — calibration_samples already
    had the ``load_seconds`` column and the heartbeat wire; a second reporter
    would just drift). ``worker_name`` maps to the card exactly as the
    completion seam does (``"<name>:0"``); variant is derived, never passed by
    hand. Skips, never guesses: no load_seconds, unknown variant, or a failed
    load leave the EMA untouched. Returns how many samples were folded.
    Fail-open per telemetry doctrine — this must never fail a heartbeat."""
    if not worker_name or not samples:
        return 0
    card = f"{worker_name}:0"
    folded = 0
    for s in samples:
        try:
            if not isinstance(s, dict) or not s.get("ok"):
                continue
            load_s = s.get("load_seconds")
            if not isinstance(load_s, (int, float)) or load_s <= 0:
                continue
            variant = derive_variant(
                s.get("n_gpu_layers"), s.get("total_layers"),
                moe_capable=bool(s.get("moe_capable")))
            if variant is None:
                continue
            if model_metrics_store.record_load(
                    str(s.get("model_key") or ""), variant, card, "unloaded",
                    upload_time_s=float(load_s)):
                folded += 1
        except Exception:  # noqa: BLE001 — one bad sample must not drop the rest
            continue
    return folded


# ── Failed-load log (2026-09-23) ─────────────────────────────────────────────
# Every failed model load lands as ONE compute_actions row: action="load",
# outcome="fail", model, variant (the GGUF file the loader opened), worker_card,
# duration_s (the attempt), and detail = the worker's structured load_failure
# ({class, loader_stderr (whole), log_ref, path}) + message + phase/source. Before this
# nothing recorded a failed load; the why (e.g. Echo-Mini's check_tensor_dims
# wrong shape) survived only in the worker journal. Callers: the /v1 relay's
# worker-error path (remote.py, per-run) and set_load_report (the warm/probe
# catch-all, deduped by the report's ts). Fail-open like every metrics write.
ACTION_LOAD_FAIL_OUTCOME = "fail"
_LOAD_FAIL_SEEN: "set" = set()
_LOAD_FAIL_SEEN_MAX = 4096
_LOAD_FAIL_LOCK = threading.Lock()


def _action_row(r) -> Dict[str, Any]:
    """A compute_actions tuple/mapping (12 columns) -> the reader dict."""
    if isinstance(r, dict):
        r = (r["id"], r["ts"], r["action"], r["model"], r["variant"],
             r["worker_card"], r["duration_s"], r["tok_per_s"], r["tokens"],
             r["bytes"], r["outcome"], r["detail_json"])
    detail, raw = None, r[11]
    if raw:
        try:
            detail = json.loads(raw)
        except Exception:  # noqa: BLE001 — keep the stored text itself
            detail = None
    return {"id": int(r[0]), "ts": r[1], "action": r[2], "model": r[3],
            "variant": r[4], "worker_card": r[5], "duration_s": r[6],
            "tok_per_s": r[7], "tokens": r[8], "bytes": r[9],
            "outcome": r[10], "detail": detail, "detail_json": raw}


def get_action(action_id, store: Any = None) -> Optional[Dict[str, Any]]:
    """ONE compute_actions row by id with its detail WHOLE (and the stored
    detail_json text verbatim). Works on both backends: the store's own
    ``get_action`` when it has one, else a direct SELECT through the
    toolserver's Postgres connection. None when absent. Never raises."""
    try:
        aid = int(str(action_id).split("#")[-1])
    except (TypeError, ValueError):
        return None
    store = store if store is not None else model_metrics_store
    fn = getattr(store, "get_action", None)
    if callable(fn):
        try:
            return fn(aid)
        except Exception:  # noqa: BLE001
            return None
    try:
        from abstract_toolserver import metrics as _pg
        if not _pg._ensure():
            return None
        with _pg._conn() as c, c.cursor() as cur:
            cur.execute(
                "SELECT id, ts, action, model, variant, worker_card, duration_s,"
                " tok_per_s, tokens, bytes, outcome, detail_json FROM compute_actions"
                " WHERE id = %s", (aid,))
            r = cur.fetchone()
        return _action_row(r) if r else None
    except Exception:  # noqa: BLE001
        return None


def recent_actions(store: Any = None, limit: int = 200, *,
                   since_ts: Optional[float] = None, action: Optional[str] = None,
                   model: Optional[str] = None, worker: Optional[str] = None,
                   outcome: Optional[str] = None) -> list:
    """``store.recent_actions`` with an ``outcome`` filter on EVERY backend: a
    store whose reader predates the filter (the toolserver Postgres store)
    is over-fetched (max window) and filtered here. Never raises."""
    store = store if store is not None else model_metrics_store
    kw = dict(since_ts=since_ts, action=action, model=model, worker=worker)
    if outcome:
        try:
            return store.recent_actions(limit=limit, outcome=outcome, **kw)
        except TypeError:
            rows = store.recent_actions(limit=5000, **kw) or []
            lim = max(1, min(int(limit or 200), 5000))
            return [r for r in rows if r.get("outcome") == outcome][:lim]
    return store.recent_actions(limit=limit, **kw)


def _variant_of(load_failure: Optional[dict], variant: Optional[str]) -> Optional[str]:
    if variant:
        return str(variant)
    path = (load_failure or {}).get("path")
    return os.path.basename(str(path)) if path else None


def record_load_failure(model: str, worker_card: str, *,
                        load_failure: Optional[dict] = None,
                        message: Optional[str] = None,
                        variant: Optional[str] = None,
                        duration_s: Optional[float] = None,
                        phase: Optional[str] = None,
                        source: Optional[str] = None,
                        report_ts: Optional[float] = None,
                        store: Any = None) -> bool:
    """Append ONE ``load``/``fail`` row. ``report_ts`` (a load report's own
    stamp) dedupes: the same (model, worker_card, report_ts) is recorded once,
    across beats AND restarts (the durable log is checked, not only memory).
    Returns True when a row landed. Never raises."""
    try:
        if not model or not worker_card:
            return False
        store = store if store is not None else model_metrics_store
        key = None
        if report_ts is not None:
            key = (str(model), str(worker_card), round(float(report_ts), 3))
            with _LOAD_FAIL_LOCK:
                if key in _LOAD_FAIL_SEEN:
                    return False
            try:
                prior = recent_actions(store, limit=200, action="load",
                                       model=str(model), worker=str(worker_card),
                                       outcome=ACTION_LOAD_FAIL_OUTCOME)
            except Exception:  # noqa: BLE001 — dedupe is best-effort
                prior = []
            for r in prior or ():
                d = r.get("detail") or {}
                rts = d.get("report_ts") if isinstance(d, dict) else None
                if rts is not None and round(float(rts), 3) == key[2] \
                        and r.get("worker_card") == str(worker_card):
                    with _LOAD_FAIL_LOCK:
                        _LOAD_FAIL_SEEN.add(key)
                    return False
        lf = dict(load_failure) if isinstance(load_failure, dict) else {}
        if not lf.get("class"):
            lf["class"] = "other"
        lf.setdefault("loader_stderr", None)
        lf.setdefault("path", None)
        if message:
            lf["message"] = str(message)       # whole — no cap (2026-09-23)
        if phase:
            lf["phase"] = str(phase)
        if source:
            lf["source"] = str(source)
        if report_ts is not None:
            lf["report_ts"] = float(report_ts)
        ok = bool(store.append_action(
            ACTION_LOAD, model=str(model), variant=_variant_of(lf, variant),
            worker_card=str(worker_card),
            duration_s=float(duration_s) if duration_s is not None else None,
            outcome=ACTION_LOAD_FAIL_OUTCOME, detail=lf))
        if ok and key is not None:
            with _LOAD_FAIL_LOCK:
                if len(_LOAD_FAIL_SEEN) >= _LOAD_FAIL_SEEN_MAX:
                    _LOAD_FAIL_SEEN.clear()
                _LOAD_FAIL_SEEN.add(key)
        return ok
    except Exception:  # noqa: BLE001 — metrics must never break serving
        return False


def record_load_report_failure(worker: Optional[dict], model: str,
                               report: Optional[dict]) -> bool:
    """The load_reports catch-all: a FAILED warm/probe report becomes one
    load/fail row, deduped by the report's ``ts``. A "not local" probe verdict
    is not a load attempt (the probe never loads absent weights) — skipped."""
    try:
        if not isinstance(report, dict) or report.get("ok", True) is not False:
            return False
        err = str(report.get("error") or "")
        if err.startswith("not local"):
            return False
        lf = report.get("load_failure")
        if not isinstance(lf, dict):
            try:        # an older worker: classify its text the engine's way
                from hugpy_engine.serve.load_failure import classify_text, _stderr_from_text
                lf = {"class": classify_text(err), "loader_stderr": _stderr_from_text(err),
                      "path": None}
            except Exception:  # noqa: BLE001
                lf = {"class": "other", "loader_stderr": None, "path": None}
        w = worker or {}
        card = f"{w.get('name') or w.get('id') or 'unknown'}:0"
        ts = report.get("ts")
        dur = None
        if isinstance(ts, (int, float)):
            dur = max(0.0, time.time() - float(ts))
        return record_load_failure(
            model, card, load_failure=lf, message=err or None,
            variant=report.get("gguf_file"), duration_s=dur,
            phase=report.get("phase") or "cold", source="load_report",
            report_ts=ts if isinstance(ts, (int, float)) else None)
    except Exception:  # noqa: BLE001
        return False
