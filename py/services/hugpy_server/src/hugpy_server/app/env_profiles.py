"""PER-MODEL ENVIRONMENTS — the DB side (operator 2026-10-02: "multiple envs
should be able to be established and attributed to a model's runtime … the
helper/hugpy-brain model is a good fit for this charge").

A profile is a named venv recipe: pinned ``packages`` + a ``base`` ("worker" =
an overlay on the worker's own venv, "isolated" = a fresh one); workers
materialize the APPROVED profiles attributed to their models
(hugpy_engine.serve.profiles) and report back state + the frozen lock.

    env_profiles          name, packages, base, note, status (proposed|approved), author
    env_profile_workers   per (profile, worker): state, base, lock, error, materialized_at, test

Materializing is a pip install on a worker box, so a profile hugpy-brain
creates starts PROPOSED and files a help ticket; only an APPROVED profile is
relayed to workers. An operator-created profile is approved on creation.
Fail-open (small_pg)."""
from __future__ import annotations

import json
import re
import time
from typing import Optional

BASES = ("worker", "isolated")
STATUSES = ("proposed", "approved")
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_PKG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-\[\],]*((==|>=|<=|~=|!=|<|>)[A-Za-z0-9.*+!_\-]+(,(==|>=|<=|~=|!=|<|>)[A-Za-z0-9.*+!_\-]+)*)?$")

DDL = """
CREATE TABLE IF NOT EXISTS env_profiles (
    name        TEXT PRIMARY KEY,
    packages    JSONB NOT NULL DEFAULT '[]'::jsonb,
    base        TEXT NOT NULL DEFAULT 'worker',
    note        TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'proposed',
    created_by  TEXT,
    by_kind     TEXT,
    created_at  DOUBLE PRECISION NOT NULL,
    updated_at  DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS env_profile_workers (
    name            TEXT NOT NULL,
    worker_id       TEXT NOT NULL,
    state           TEXT,
    base            TEXT,
    lock            JSONB NOT NULL DEFAULT '[]'::jsonb,
    error           TEXT,
    materialized_at DOUBLE PRECISION,
    test            JSONB,
    updated_at      DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (name, worker_id)
);
"""

_RUNNER = None


def _run(fn):
    global _RUNNER
    if _RUNNER is None:
        from hugpy_server.app.small_pg import Runner
        _RUNNER = Runner("env profiles", DDL)
    return _RUNNER.run(fn)


def _rows(cur) -> list:
    from hugpy_server.app.small_pg import rows
    return rows(cur)


def validate(name, packages, base) -> "tuple[str, list, str]":
    """(name, packages, base) cleaned; ValueError on a bad name / pin / base.
    Packages are pip requirement specifiers — no URLs, paths or options."""
    name = str(name or "").strip()
    if not _SLUG.match(name):
        raise ValueError(f"profile name {name!r} must be slug-safe (letters, digits, . _ -; max 64)")
    pkgs = [str(p).strip() for p in (packages or []) if str(p).strip()]
    bad = [p for p in pkgs if not _PKG.match(p)]
    if bad:
        raise ValueError(f"not plain pip requirement specifiers: {bad}")
    base = str(base or "worker").strip().lower()
    if base not in BASES:
        raise ValueError(f"base must be one of {list(BASES)}")
    return name, pkgs, base


def put(name, packages, base="worker", note="", *, by: str, by_kind: str = "operator") -> Optional[dict]:
    """Create or replace a profile. An operator's write is approved; an agent's
    (re)write goes back to PROPOSED (a changed recipe is a new install)."""
    name, pkgs, base = validate(name, packages, base)
    status = "approved" if by_kind == "operator" else "proposed"
    now = time.time()

    def op(cur):
        cur.execute("INSERT INTO env_profiles (name, packages, base, note, status, created_by, by_kind, created_at, updated_at)"
                    " VALUES (%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (name) DO UPDATE SET"
                    " packages=EXCLUDED.packages, base=EXCLUDED.base, note=EXCLUDED.note, status=EXCLUDED.status,"
                    " by_kind=EXCLUDED.by_kind, updated_at=EXCLUDED.updated_at",
                    (name, json.dumps(pkgs), base, str(note or "")[:2000], status, by, by_kind, now, now))
        return True
    return get(name) if _run(op) else None


def approve(name: str, *, by: str) -> Optional[dict]:
    def op(cur):
        cur.execute("UPDATE env_profiles SET status='approved', updated_at=%s WHERE name=%s", (time.time(), name))
        return cur.rowcount
    return get(name) if _run(op) else None


def get(name: str) -> Optional[dict]:
    def op(cur):
        cur.execute("SELECT * FROM env_profiles WHERE name=%s", (name,))
        r = _rows(cur)
        if not r:
            return None
        out = r[0]
        cur.execute("SELECT worker_id, state, base, lock, error, materialized_at, test, updated_at"
                    " FROM env_profile_workers WHERE name=%s ORDER BY worker_id", (name,))
        out["workers"] = _rows(cur)
        return out
    return _run(op)


def list_all() -> list:
    def op(cur):
        cur.execute("SELECT * FROM env_profiles ORDER BY name")
        profs = _rows(cur)
        cur.execute("SELECT name, worker_id, state, base, lock, error, materialized_at, test, updated_at"
                    " FROM env_profile_workers")
        by = {}
        for w in _rows(cur):
            by.setdefault(w.pop("name"), []).append(w)
        for p in profs:
            p["workers"] = by.get(p["name"], [])
        return profs
    return _run(op) or []


def approved_specs(names) -> dict:
    """{name: {packages, base}} for the APPROVED profiles among ``names`` (the
    relay to a worker declares only these)."""
    names = [n for n in (names or []) if n]
    if not names:
        return {}

    def op(cur):
        cur.execute("SELECT name, packages, base FROM env_profiles WHERE status='approved' AND name = ANY(%s)",
                    (names,))
        return {r[0]: {"packages": list(r[1] or []), "base": r[2]} for r in cur.fetchall()}
    return _run(op) or {}


def record_worker_report(worker_id: str, report: dict) -> int:
    """Store a worker's ``{name: {state, base?, lock?, error?, materialized_at?}}``
    (hugpy_engine.serve.profiles.report) — the heartbeat's profile truth."""
    if not isinstance(report, dict) or not report:
        return 0
    now = time.time()

    def op(cur):
        n = 0
        for name, row in report.items():
            if not isinstance(row, dict):
                continue
            cur.execute("INSERT INTO env_profile_workers (name, worker_id, state, base, lock, error, materialized_at, updated_at)"
                        " VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s) ON CONFLICT (name, worker_id) DO UPDATE SET"
                        " state=EXCLUDED.state, base=COALESCE(EXCLUDED.base, env_profile_workers.base),"
                        " lock=CASE WHEN EXCLUDED.state='ready' THEN EXCLUDED.lock ELSE env_profile_workers.lock END,"
                        " error=EXCLUDED.error,"
                        " materialized_at=COALESCE(EXCLUDED.materialized_at, env_profile_workers.materialized_at),"
                        " updated_at=EXCLUDED.updated_at",
                        (str(name), str(worker_id), row.get("state"), row.get("base"),
                         json.dumps(row.get("lock") or []), row.get("error"), row.get("materialized_at"), now))
            n += 1
        return n
    return _run(op) or 0


def record_test(name: str, worker_id: str, result: dict) -> bool:
    def op(cur):
        cur.execute("INSERT INTO env_profile_workers (name, worker_id, test, updated_at) VALUES (%s,%s,%s::jsonb,%s)"
                    " ON CONFLICT (name, worker_id) DO UPDATE SET test=EXCLUDED.test, updated_at=EXCLUDED.updated_at",
                    (name, worker_id, json.dumps(result, default=str), time.time()))
        return True
    return bool(_run(op))
