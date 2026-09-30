"""Client-declared session state (SESSION-LEASE-20260929).

Central used to infer a caller's state from socket liveness and job ages, so a
long prefill read as stuck and a crashed client's call sat pending forever.
Clients (hugpy-agent and the harnesses it launches) now DECLARE it:

  * identity on every call — the ``X-Hugpy-Client-*`` headers calllog already
    parses (Session/Turn/Request/Process/Pid/User/Task/Platform). ``note_job``
    binds each job id to (session, turn, client request) at job start.
  * a lease — ``POST /llm/sessions/<sid>/lease`` every ~10 s while a turn is in
    flight, TTL ~30 s, body ``{state, event, turn_id, request_ids, ttl, ...}``.
  * transitions — the same route with ``event`` = ``turn_done`` (state idle)
    or ``session_closed`` (state closed).

The table (one row per session): session_id, client_process, pid, user, host,
platform, client, state (active|waiting|idle|abandoned|closed), last_lease_at,
ttl, current_turn, lease_requests (the client's in-flight ids), last_error,
created_at, updated_at. A side table maps job id -> (session, turn, client
request) so a client can cancel by ITS request id and the queue can hide jobs
of a session that is idle/closed/abandoned.

Rules:
  * a session whose lease LAPSES (last_lease_at + ttl < now while active/
    waiting) turns ``abandoned`` and its live jobs are cancelled through the
    authoritative cancel path (JobStore.cancel_authoritative);
  * a session that never leased (an old client) is NEVER abandoned by lease
    logic — the absence of a lease is not evidence of death;
  * ``lease_fresh(job_id)`` protects a job from central's own idle/wedge
    heuristics (long prefill) while its session's lease is fresh;
  * idle/closed transitions cancel the session's live jobs the client no
    longer lists (they are orphans by the client's own word);
  * closed/abandoned/idle rows are kept RETENTION_S (24 h), then pruned.

Storage: one small SQLite file shared by every gunicorn worker process
(connection per call — fork-safe), ``HUGPY_SESSIONS_DB`` overrides the path,
``off`` disables the feature (every read is empty, every write a no-op).
Sweeps are EVENT-DRIVEN (lease POSTs, session/queue/jobs views, job starts) —
no timer thread, same doctrine as JobStore.expire_pending_orphans.
Every public function is best-effort and never raises into serving.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from typing import Callable, Iterable, Optional

log = logging.getLogger(__name__)

STATES = ("active", "waiting", "idle", "abandoned", "closed")
LIVE_STATES = ("active", "waiting")
QUIET_STATES = ("idle", "closed", "abandoned")     # never shown as pending
RETENTION_S = 24 * 3600.0
DEFAULT_TTL = 30.0
TTL_MIN, TTL_MAX = 5.0, 600.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS client_sessions (
  session_id TEXT PRIMARY KEY,
  client_process TEXT, pid TEXT, user TEXT, host TEXT, platform TEXT, client TEXT,
  state TEXT NOT NULL,
  last_lease_at REAL, ttl REAL,
  current_turn TEXT, lease_requests TEXT,
  last_error TEXT, last_event TEXT,
  created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS client_session_jobs (
  job_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL, turn_id TEXT, client_request TEXT,
  created_at REAL NOT NULL, ended_at REAL
);
CREATE INDEX IF NOT EXISTS csj_session ON client_session_jobs(session_id);
CREATE INDEX IF NOT EXISTS csj_request ON client_session_jobs(client_request);
"""

_INIT_LOCK = threading.Lock()
_INITED: set = set()


def db_path() -> Optional[str]:
    env = (os.environ.get("HUGPY_SESSIONS_DB") or "").strip()
    if env.lower() in ("off", "none", "0", "disabled"):
        return None
    if env:
        return env
    try:
        from hugpy_control.shared import default_db_path
        base = os.path.dirname(default_db_path()) or "/tmp"
    except Exception:  # noqa: BLE001
        base = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    return os.path.join(base, f"hugpy-sessions-{os.getuid()}.db")


def _connect() -> Optional[sqlite3.Connection]:
    p = db_path()
    if not p:
        return None
    conn = sqlite3.connect(p, timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    if p not in _INITED:
        with _INIT_LOCK:
            if p not in _INITED:
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                except sqlite3.DatabaseError:
                    pass
                conn.executescript(_SCHEMA)
                _INITED.add(p)
    return conn


def _clean(v, n: int = 200) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s[:n] or None


def _ttl(v) -> float:
    try:
        t = float(v)
    except (TypeError, ValueError):
        return DEFAULT_TTL
    return min(TTL_MAX, max(TTL_MIN, t))


def _row(r: sqlite3.Row, jobs: Optional[list] = None, now: Optional[float] = None) -> dict:
    d = dict(r)
    try:
        d["lease_requests"] = json.loads(d.get("lease_requests") or "[]")
    except ValueError:
        d["lease_requests"] = []
    now = time.time() if now is None else now
    la = d.get("last_lease_at")
    d["lease_age_s"] = round(now - la, 1) if la else None
    d["lease_fresh"] = bool(la and d["state"] in LIVE_STATES
                            and now - la <= float(d.get("ttl") or DEFAULT_TTL))
    if jobs is not None:
        d["in_flight"] = jobs
    return d


# ── the job store seam ───────────────────────────────────────────────────
def _job_status(job_id: str) -> Optional[str]:
    try:
        from hugpy_control.jobs import job_store, normalize_status
        d = job_store.get_dict(job_id)
        return normalize_status(d.get("status")) if d else None
    except Exception:  # noqa: BLE001
        return None


_TERMINAL = ("done", "failed", "cancelled", "expired", "interrupted", "error")


def _is_live(job_id: str) -> bool:
    st = _job_status(job_id)
    return st is not None and st not in _TERMINAL


def _default_cancel(job_id: str, reason: str) -> dict:
    from hugpy_control.jobs import job_store
    return job_store.cancel_authoritative(job_id, reason)


# ── writes ───────────────────────────────────────────────────────────────
def note_job(job_id: str, ctx: dict, now: Optional[float] = None) -> None:
    """Bind a new job to the session that declared itself on the request
    (calllog start row). Creates the session row for a client that never
    leases — such a session is tracked but never abandoned by lease logic."""
    sid = _clean((ctx or {}).get("client_session"))
    if not job_id or not sid:
        return
    now = time.time() if now is None else now
    try:
        conn = _connect()
        if conn is None:
            return
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO client_session_jobs(job_id, session_id, turn_id,"
                " client_request, created_at, ended_at) VALUES (?,?,?,?,?,NULL)",
                (job_id, sid, _clean(ctx.get("client_turn")),
                 _clean(ctx.get("client_request")), now))
            cur = conn.execute("SELECT state FROM client_sessions WHERE session_id=?", (sid,))
            got = cur.fetchone()
            if got is None:
                conn.execute(
                    "INSERT INTO client_sessions(session_id, client_process, pid, user,"
                    " host, platform, client, state, last_lease_at, ttl, current_turn,"
                    " lease_requests, created_at, updated_at, last_event)"
                    " VALUES (?,?,?,?,?,?,?,?,NULL,?,?,'[]',?,?,'job')",
                    (sid, _clean(ctx.get("client_process"), 160), _clean(ctx.get("client_pid"), 24),
                     _clean(ctx.get("client_user"), 160), _clean(ctx.get("client")),
                     _clean(ctx.get("client_platform"), 100), None, "active", DEFAULT_TTL,
                     _clean(ctx.get("client_turn")), now, now))
            else:
                # A new call is the client's own word that it is working again.
                conn.execute(
                    "UPDATE client_sessions SET state=CASE WHEN state IN ('idle','abandoned')"
                    " THEN 'active' ELSE state END, current_turn=COALESCE(?, current_turn),"
                    " updated_at=? WHERE session_id=? AND state!='closed'",
                    (_clean(ctx.get("client_turn")), now, sid))
    except Exception:  # noqa: BLE001
        log.debug("sessions.note_job failed", exc_info=True)


def note_job_end(job_id: str, now: Optional[float] = None) -> None:
    try:
        conn = _connect()
        if conn is None or not job_id:
            return
        with conn:
            conn.execute("UPDATE client_session_jobs SET ended_at=? WHERE job_id=?"
                         " AND ended_at IS NULL",
                         (time.time() if now is None else now, job_id))
    except Exception:  # noqa: BLE001
        log.debug("sessions.note_job_end failed", exc_info=True)


def lease(session_id: str, body: dict, now: Optional[float] = None,
          cancel: Callable[[str, str], dict] = None,
          is_live: Callable[[str], bool] = None) -> dict:
    """Renew / transition one session. Returns the session row plus the job
    ids this call cancelled (orphans the client no longer lists)."""
    sid = _clean(session_id)
    if not sid:
        return {"ok": False, "error": "session id required"}
    body = body if isinstance(body, dict) else {}
    now = time.time() if now is None else now
    event = _clean(body.get("event"), 40) or "lease"
    state = _clean(body.get("state"), 20) or "active"
    if event == "session_closed":
        state = "closed"
    elif event == "turn_done" and state not in ("waiting", "active"):
        state = "idle"
    if state not in ("active", "waiting", "idle", "closed"):
        state = "active"
    listed = [str(x)[:200] for x in (body.get("request_ids") or []) if x][:256]
    turn = _clean(body.get("turn_id"))
    conn = _connect()
    if conn is None:
        return {"ok": False, "error": "sessions disabled"}
    with conn:
        conn.execute(
            "INSERT INTO client_sessions(session_id, client_process, pid, user, host,"
            " platform, client, state, last_lease_at, ttl, current_turn, lease_requests,"
            " last_error, last_event, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL,?,?,?)"
            " ON CONFLICT(session_id) DO UPDATE SET"
            " client_process=COALESCE(excluded.client_process, client_process),"
            " pid=COALESCE(excluded.pid, pid), user=COALESCE(excluded.user, user),"
            " host=COALESCE(excluded.host, host), platform=COALESCE(excluded.platform, platform),"
            " client=COALESCE(excluded.client, client), state=excluded.state,"
            " last_lease_at=excluded.last_lease_at, ttl=excluded.ttl,"
            " current_turn=COALESCE(excluded.current_turn, current_turn),"
            " lease_requests=excluded.lease_requests, last_error=NULL,"
            " last_event=excluded.last_event, updated_at=excluded.updated_at",
            (sid, _clean(body.get("client_process"), 160), _clean(body.get("pid"), 24),
             _clean(body.get("user"), 160), _clean(body.get("host")),
             _clean(body.get("platform"), 100), _clean(body.get("client"), 64), state, now,
             _ttl(body.get("ttl")), turn, json.dumps(listed), event, now, now))
    cancelled: list = []
    if state in ("idle", "closed") or event == "turn_done":
        # The client's own word: whatever it does not list any more is orphaned.
        scope_turn = turn if (event == "turn_done" and state not in ("idle", "closed")) else None
        cancelled = _cancel_orphans(conn, sid, set(listed), scope_turn,
                                    f"client session {state} ({event})",
                                    cancel or _default_cancel, is_live or _is_live, now)
    sweep(now=now, cancel=cancel, is_live=is_live)
    return {"ok": True, "session": get(sid, now=now, is_live=is_live), "cancelled": cancelled}


def _cancel_orphans(conn, sid, keep: set, turn: Optional[str], reason: str,
                    cancel, is_live, now) -> list:
    q = ("SELECT job_id, client_request FROM client_session_jobs"
         " WHERE session_id=? AND ended_at IS NULL")
    args: list = [sid]
    if turn:
        q += " AND turn_id=?"
        args.append(turn)
    out = []
    for r in conn.execute(q, args).fetchall():
        jid, creq = r["job_id"], r["client_request"]
        if creq and creq in keep:
            continue
        if not is_live(jid):
            conn.execute("UPDATE client_session_jobs SET ended_at=? WHERE job_id=?", (now, jid))
            continue
        try:
            res = cancel(jid, reason) or {}
        except Exception:  # noqa: BLE001
            res = {}
        if res.get("cancelled"):
            out.append(jid)
    return out


def sweep(now: Optional[float] = None, cancel: Callable[[str, str], dict] = None,
          is_live: Callable[[str], bool] = None) -> list:
    """Lapse check + retention. Returns the job ids cancelled because their
    session's lease lapsed. Never raises."""
    now = time.time() if now is None else now
    cancel = cancel or _default_cancel
    is_live = is_live or _is_live
    out: list = []
    try:
        conn = _connect()
        if conn is None:
            return out
        lapsed = conn.execute(
            "SELECT session_id, last_lease_at, ttl FROM client_sessions"
            " WHERE state IN ('active','waiting') AND last_lease_at IS NOT NULL"
            " AND last_lease_at + COALESCE(ttl, ?) < ?", (DEFAULT_TTL, now)).fetchall()
        for r in lapsed:
            sid = r["session_id"]
            age = int(now - r["last_lease_at"])
            with conn:
                cur = conn.execute(
                    "UPDATE client_sessions SET state='abandoned', last_error=?, updated_at=?"
                    " WHERE session_id=? AND state IN ('active','waiting')"
                    " AND last_lease_at=?",
                    (f"lease lapsed: no renewal for {age}s (ttl {int(r['ttl'] or DEFAULT_TTL)}s)",
                     now, sid, r["last_lease_at"]))
            if cur.rowcount:
                out += _cancel_orphans(conn, sid, set(), None,
                                       f"client session abandoned (lease lapsed {age}s)",
                                       cancel, is_live, now)
        cutoff = now - RETENTION_S
        with conn:
            conn.execute(
                "DELETE FROM client_session_jobs WHERE session_id IN (SELECT session_id"
                " FROM client_sessions WHERE state IN ('closed','abandoned','idle')"
                " AND updated_at < ?)", (cutoff,))
            conn.execute("DELETE FROM client_sessions WHERE state IN"
                         " ('closed','abandoned','idle') AND updated_at < ?", (cutoff,))
            conn.execute("DELETE FROM client_session_jobs WHERE ended_at IS NOT NULL"
                         " AND ended_at < ?", (cutoff,))
    except Exception:  # noqa: BLE001
        log.debug("sessions.sweep failed", exc_info=True)
    return out


# ── reads ────────────────────────────────────────────────────────────────
def _jobs_of(conn, sid: str, is_live) -> list:
    rows = conn.execute(
        "SELECT job_id, turn_id, client_request, created_at FROM client_session_jobs"
        " WHERE session_id=? AND ended_at IS NULL ORDER BY created_at", (sid,)).fetchall()
    return [{"job_id": r["job_id"], "turn_id": r["turn_id"],
             "client_request": r["client_request"], "created_at": r["created_at"],
             "status": _job_status(r["job_id"])}
            for r in rows if is_live(r["job_id"])]


def get(session_id: str, now: Optional[float] = None, is_live=None) -> Optional[dict]:
    try:
        conn = _connect()
        if conn is None:
            return None
        r = conn.execute("SELECT * FROM client_sessions WHERE session_id=?",
                         (session_id,)).fetchone()
        if r is None:
            return None
        return _row(r, _jobs_of(conn, session_id, is_live or _is_live), now)
    except Exception:  # noqa: BLE001
        log.debug("sessions.get failed", exc_info=True)
        return None


def list_sessions(state: Optional[str] = None, now: Optional[float] = None,
                  is_live=None) -> list:
    try:
        conn = _connect()
        if conn is None:
            return []
        q, args = "SELECT * FROM client_sessions", []
        if state:
            q += " WHERE state=?"
            args.append(state)
        q += " ORDER BY updated_at DESC LIMIT 500"
        il = is_live or _is_live
        return [_row(r, _jobs_of(conn, r["session_id"], il), now)
                for r in conn.execute(q, args).fetchall()]
    except Exception:  # noqa: BLE001
        log.debug("sessions.list failed", exc_info=True)
        return []


def job_sessions(job_ids: Iterable[str], now: Optional[float] = None) -> dict:
    """{job_id: {session_id, state, lease_fresh, turn_id, client_request}} for
    the ids that belong to a declared session."""
    ids = [j for j in job_ids if j]
    if not ids:
        return {}
    try:
        conn = _connect()
        if conn is None:
            return {}
        now = time.time() if now is None else now
        out = {}
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            q = ("SELECT j.job_id, j.turn_id, j.client_request, s.* FROM client_session_jobs j"
                 " JOIN client_sessions s ON s.session_id=j.session_id WHERE j.job_id IN (%s)"
                 % ",".join("?" * len(chunk)))
            for r in conn.execute(q, chunk).fetchall():
                la = r["last_lease_at"]
                out[r["job_id"]] = {
                    "session_id": r["session_id"], "state": r["state"],
                    "turn_id": r["turn_id"], "client_request": r["client_request"],
                    "client_process": r["client_process"],
                    "lease_age_s": round(now - la, 1) if la else None,
                    "lease_fresh": bool(la and r["state"] in LIVE_STATES
                                        and now - la <= float(r["ttl"] or DEFAULT_TTL)),
                }
        return out
    except Exception:  # noqa: BLE001
        log.debug("sessions.job_sessions failed", exc_info=True)
        return {}


def fresh_job_ids(now: Optional[float] = None) -> set:
    """Every unfinished job bound to a session whose lease is fresh — one
    query, for the job store's wedge-expiry guard."""
    now = time.time() if now is None else now
    try:
        conn = _connect()
        if conn is None:
            return set()
        rows = conn.execute(
            "SELECT j.job_id FROM client_session_jobs j JOIN client_sessions s"
            " ON s.session_id=j.session_id WHERE j.ended_at IS NULL"
            " AND s.state IN ('active','waiting') AND s.last_lease_at IS NOT NULL"
            " AND s.last_lease_at + COALESCE(s.ttl, ?) >= ?", (DEFAULT_TTL, now)).fetchall()
        return {r["job_id"] for r in rows}
    except Exception:  # noqa: BLE001
        return set()


def lease_fresh(job_id: str, now: Optional[float] = None) -> bool:
    """True while the job's session holds a fresh lease — central must not
    retire it on idle-socket / no-progress heuristics (long prefill)."""
    return bool(job_sessions([job_id], now).get(job_id, {}).get("lease_fresh"))


def resolve_job_id(some_id: str) -> Optional[str]:
    """A client's own request id -> the job id central minted for it (most
    recent live binding first). None when the id is not a client request."""
    try:
        conn = _connect()
        if conn is None or not some_id:
            return None
        r = conn.execute(
            "SELECT job_id FROM client_session_jobs WHERE client_request=?"
            " ORDER BY (ended_at IS NULL) DESC, created_at DESC LIMIT 1",
            (some_id,)).fetchone()
        return r["job_id"] if r else None
    except Exception:  # noqa: BLE001
        return None
