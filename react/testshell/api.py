#!/usr/bin/env python3
"""hugpy testshell API: the UI reflects the DB, the knobs turn the DB.

Reads one row per model from the `model_full` view (DDL: model_full.sql).
Writes exactly two things:
  * operator knobs      model_workers.user_settings   (POST .../knobs)
  * computed-once facts models.weights, model_workers.plan  (compute.py)
The computed facts are filled in lazily: a background pass runs at startup
and every ENSURE_S seconds, and a knob write recomputes that pair at once.
Serves ./dist as the SPA when built.

  HUGPY_TESTSHELL_DSN   libpq DSN (default: dbname=hugpy, local socket)
  HUGPY_TESTSHELL_PORT  listen port (default 7013)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from datetime import date, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute  # noqa: E402

try:  # /srv/hugpy/venv has psycopg 3; system python has psycopg2
    import psycopg as _pg
    from psycopg.types.json import Jsonb as J
    def _connect(dsn): return _pg.connect(dsn, autocommit=True)
except ImportError:  # pragma: no cover
    import psycopg2 as _pg
    from psycopg2.extras import Json as J
    def _connect(dsn):
        c = _pg.connect(dsn); c.autocommit = True; return c

DSN = os.environ.get("HUGPY_TESTSHELL_DSN", "dbname=hugpy")
PORT = int(os.environ.get("HUGPY_TESTSHELL_PORT", "7013"))
ENSURE_S = int(os.environ.get("HUGPY_TESTSHELL_ENSURE_S", "60"))
DIST = Path(__file__).resolve().parent / "dist"

MODELS_SQL = """
SELECT id, name, hub_id, framework, spec, weights, serving, quants, workers, live, updated_at
FROM model_full {where} ORDER BY name
"""
WORKERS_SQL = """
SELECT r.worker_id, r.payload->>'name' AS name, r.payload->>'url' AS url,
       r.payload->>'gpu' AS gpu, r.payload->>'pkg_version' AS pkg_version, r.revision, r.updated_at,
       (SELECT sum((g->>'memory_total')::bigint) FROM jsonb_array_elements(r.payload->'gpus') g) AS gpu_total,
       (r.payload->>'ram_total')::bigint AS ram_total
FROM hugpy_worker_registry r ORDER BY 2
"""
LIVENESS_SQL = "SELECT payload FROM hugpy_feed WHERE feed = 'liveness'"
REGISTRY_SQL = "SELECT worker_id, payload FROM hugpy_worker_registry"
MODEL_RAW_SQL = """
SELECT m.id, m.name, m.framework, m.attributes, m.weights,
       COALESCE((SELECT json_agg(json_build_object('file', q.file, 'bytes', q.bytes)) FROM model_quants q WHERE q.model_id = m.id), '[]'::json) AS quants
FROM models m {where} ORDER BY m.id
"""
PAIRS_SQL = "SELECT worker_id, user_settings, plan, fits, assigned FROM model_workers WHERE model_id = %s"
TABLES_SQL = """
SELECT c.relname, c.reltuples::bigint FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind = 'r' ORDER BY 1
"""


def _json_default(o):
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    return str(o)


def _rows(cur, sql, params=()):
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _as_json(v):
    return json.loads(v) if isinstance(v, str) else v


# ── reads ────────────────────────────────────────────────────────────────────
def load_workers(cur):
    live = {}
    for r in _rows(cur, LIVENESS_SQL):
        for w in _as_json(r["payload"]) or []:
            live[w.get("id")] = w
    cur.execute("SELECT worker_id, gpu_total, ram_total, gpu_limit, ram_limit, vram_reserve, ram_reserve, gpu_budget, ram_budget, rev, last_change, observed_at FROM worker_budgets")
    cols = [d[0] for d in cur.description]
    budgets = {row[0]: dict(zip(cols, row)) for row in cur.fetchall()}
    out = []
    for r in _rows(cur, WORKERS_SQL):
        l = live.get(r["worker_id"], {})
        r.update(status=l.get("status"), last_seen=l.get("last_seen"),
                 loaded_models=l.get("loaded_models") or [], gpus=l.get("gpus") or [],
                 budget=budgets.get(r["worker_id"]))
        out.append(r)
    return out


def load_models(cur, model_id=None):
    where, params = ("WHERE id = %s", (model_id,)) if model_id is not None else ("", ())
    out = []
    for m in _rows(cur, MODELS_SQL.format(where=where), params):
        for k in ("spec", "weights", "serving", "quants", "workers", "live"):
            m[k] = _as_json(m[k])
        out.append(m)
    return out


# ── computed-once facts ──────────────────────────────────────────────────────
_ensure_lock = threading.Lock()
_ensure_state = {"running": False, "last_run": None, "last_took_s": None, "specs": 0, "plans": 0,
                 "errors": {}}


def _registry(cur):
    by = {}
    for r in _rows(cur, REGISTRY_SQL):
        p = _as_json(r["payload"]) or {}
        by[r["worker_id"]] = p
        if p.get("name"):
            by.setdefault(p["name"], p)
    return by


FACTS_UPSERT = """
INSERT INTO model_quant_facts (model_id, file, quant, path, size_bytes, is_moe, expert_bytes, non_expert_bytes,
                               expert_count, expert_used_count, kv_geo, kv_cost, bnb_4bit, sig, error, version, computed_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (model_id, file) DO UPDATE SET quant = EXCLUDED.quant, path = EXCLUDED.path, size_bytes = EXCLUDED.size_bytes,
  is_moe = EXCLUDED.is_moe, expert_bytes = EXCLUDED.expert_bytes, non_expert_bytes = EXCLUDED.non_expert_bytes,
  expert_count = EXCLUDED.expert_count, expert_used_count = EXCLUDED.expert_used_count, kv_geo = EXCLUDED.kv_geo,
  kv_cost = EXCLUDED.kv_cost,
  bnb_4bit = EXCLUDED.bnb_4bit, sig = EXCLUDED.sig, error = EXCLUDED.error, version = EXCLUDED.version, computed_at = now()
"""
VERDICT_UPSERT = """
INSERT INTO model_worker_quants (model_id, worker_id, file, fits, modes, memory, auto, moe_offered, moe, inputs, version, computed_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (model_id, worker_id, file) DO UPDATE SET fits = EXCLUDED.fits, modes = EXCLUDED.modes, memory = EXCLUDED.memory,
  auto = EXCLUDED.auto, moe_offered = EXCLUDED.moe_offered, moe = EXCLUDED.moe, inputs = EXCLUDED.inputs,
  version = EXCLUDED.version, computed_at = now()
"""


BUDGET_UPSERT = """
INSERT INTO worker_budgets (worker_id, name, gpu_total, ram_total, gpu_limit, ram_limit, vram_reserve, ram_reserve,
                            gpu_budget, ram_budget, pkg_version, rev, last_change, observed_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (worker_id) DO UPDATE SET name = EXCLUDED.name, gpu_total = EXCLUDED.gpu_total, ram_total = EXCLUDED.ram_total,
  gpu_limit = EXCLUDED.gpu_limit, ram_limit = EXCLUDED.ram_limit, vram_reserve = EXCLUDED.vram_reserve, ram_reserve = EXCLUDED.ram_reserve,
  gpu_budget = EXCLUDED.gpu_budget, ram_budget = EXCLUDED.ram_budget, pkg_version = EXCLUDED.pkg_version,
  rev = EXCLUDED.rev, last_change = EXCLUDED.last_change, observed_at = now()
"""
_EARMARK_KEYS = ("gpu_total", "ram_total", "gpu_limit", "ram_limit", "vram_reserve", "ram_reserve", "gpu_budget", "ram_budget", "pkg_version")


def refresh_worker_budgets(cur, reg):
    """Keep worker_budgets (the EARMARK) true to the registry. rev bumps on any
    change of totals / limits / reserves / budgets; last_change records whether a
    budget EXPANDED or REDUCED (either may be true). An offline worker (no
    reading) keeps its stored earmark. Returns {worker_id: earmark row}."""
    cur.execute("SELECT worker_id, name, gpu_total, ram_total, gpu_limit, ram_limit, vram_reserve, ram_reserve, gpu_budget, ram_budget,"
                " pkg_version, rev, last_change FROM worker_budgets")
    have = {r[0]: dict(zip(("worker_id", "name") + _EARMARK_KEYS + ("rev", "last_change"), r)) for r in cur.fetchall()}
    out = {}
    for wid, payload in reg.items():
        if not isinstance(payload, dict) or (payload.get("id") and payload.get("id") != wid):
            continue                             # the registry map also aliases workers by name; one earmark per worker id
        totals = compute.worker_totals(payload)
        old = have.get(wid)
        if totals.get("gpu_total") is None and totals.get("ram_total") is None and old:
            out[wid] = old                       # offline: absence of a reading, not a change
            continue
        e = compute.worker_earmark(totals)
        e["name"] = (payload or {}).get("name") or wid
        if old and all(old.get(k) == e.get(k) for k in _EARMARK_KEYS):
            out[wid] = old
            continue
        rev = (old["rev"] + 1) if old else 1
        change = None
        if old:
            def up(k): return (e.get(k) or 0) > (old.get(k) or 0)
            def down(k): return (e.get(k) or 0) < (old.get(k) or 0)
            change = {"at": time.time(), "expanded": up("gpu_budget") or up("ram_budget"),
                      "reduced": down("gpu_budget") or down("ram_budget"),
                      "prev": {"gpu_budget": old.get("gpu_budget"), "ram_budget": old.get("ram_budget"), "rev": old["rev"]}}
        cur.execute(BUDGET_UPSERT, (wid, e["name"], e.get("gpu_total"), e.get("ram_total"), e.get("gpu_limit"), e.get("ram_limit"),
                                    e.get("vram_reserve"), e.get("ram_reserve"), e.get("gpu_budget"), e.get("ram_budget"),
                                    e.get("pkg_version"), rev, J(change) if change else None))
        e.update(worker_id=wid, rev=rev, last_change=change)
        out[wid] = e
    return out


def _prior_unfeasible(cur, model_id, worker_id):
    """fits=false verdicts of a pair, keyed by file (to carry over a reduction)."""
    cur.execute("SELECT file, fits, modes, memory, auto, moe_offered, moe, budget_rev, kept FROM model_worker_quants"
                " WHERE model_id = %s AND worker_id = %s AND fits = false", (model_id, worker_id))
    out = {}
    for r in cur.fetchall():
        out[r[0]] = {"file": r[0], "feasible": {"fits": False, "ok": False, "modes": _as_json(r[2]) or [],
                                                 "why": "kept: unfeasible under a larger budget; this budget is smaller"},
                     "memory": _as_json(r[3]) or {}, "auto": _as_json(r[4]), "moe": _as_json(r[6]) or {"offered": False},
                     "budget_rev": r[7], "kept_prev": _as_json(r[8])}
    return out


def _write_facts(cur, model_id, weights):
    variants = (weights or {}).get("variants") or {}
    cur.execute("DELETE FROM model_quant_facts WHERE model_id = %s AND NOT (file = ANY(%s))", (model_id, list(variants)))
    for key, v in variants.items():
        moe = v.get("moe") or {}
        cur.execute(FACTS_UPSERT, (model_id, key, v.get("quant"), v.get("file") or v.get("dir"), v.get("size_bytes"),
                                   bool(moe.get("is_moe")), moe.get("expert_bytes"), moe.get("non_expert_bytes"),
                                   moe.get("expert_count"), moe.get("expert_used_count"), J(v.get("kv_geo") or {}),
                                   J(v["kv_cost"]) if v.get("kv_cost") else None, J(v.get("bnb_4bit") or {}), J(v.get("sig") or {}), v.get("error"), compute.PLAN_VERSION))


def _write_verdicts(cur, model_id, worker_id, plan):
    bq = (plan or {}).get("by_quant") or {}
    cur.execute("DELETE FROM model_worker_quants WHERE model_id = %s AND worker_id = %s AND NOT (file = ANY(%s))",
                (model_id, worker_id, list(bq)))
    for key, q in bq.items():
        cur.execute(VERDICT_UPSERT, (model_id, worker_id, key, q["feasible"].get("fits"), J(q["feasible"].get("modes") or []),
                                     J(q.get("memory") or {}), J(q.get("auto")), bool((q.get("moe") or {}).get("offered")),
                                     J(q.get("moe") or {}), J(plan.get("inputs") or {}), compute.PLAN_VERSION))
        cur.execute("UPDATE model_worker_quants SET budget_rev = %s, kept = %s WHERE model_id = %s AND worker_id = %s AND file = %s",
                    ((plan.get("inputs") or {}).get("budget_rev"), J(q["kept"]) if q.get("kept") else None, model_id, worker_id, key))


def _selected_verdict(cur, model_id, worker_id, knobs):
    """The per-quant verdict the knobs pick (gguf_file basename, else the default variant)."""
    cur.execute("SELECT w.plan->>'default_variant', v.file, v.moe, v.moe_offered FROM model_workers w"
                " LEFT JOIN model_worker_quants v ON v.model_id = w.model_id AND v.worker_id = w.worker_id"
                " WHERE w.model_id = %s AND w.worker_id = %s", (model_id, worker_id))
    rows = cur.fetchall()
    if not rows or rows[0][1] is None:
        return None
    by = {r[1]: {"file": r[1], "moe": _as_json(r[2]) or {}, "moe_offered": bool(r[3])} for r in rows}
    want = os.path.basename(str((knobs or {}).get("gguf_file") or ""))
    return by.get(want) or by.get(rows[0][0] or "") or next(iter(by.values()))


def ensure_model(cur, model_id, reg=None, force=False, earmarks=None):
    """Ingest the marker into models.weights/hub + model_quant_facts, then fill
    every pair's plan + model_worker_quants when missing or when the EARMARK
    (worker budget rev) or a weights file changed. On a pure budget REDUCTION the
    pair's fits=false verdicts are carried over untouched (operator rule).
    Returns (specs_written, plans_written)."""
    reg = reg if reg is not None else _registry(cur)
    earmarks = earmarks if earmarks is not None else refresh_worker_budgets(cur, reg)
    rows = _rows(cur, MODEL_RAW_SQL.format(where="WHERE m.id = %s"), (model_id,))
    if not rows:
        return 0, 0
    m = rows[0]
    m["attributes"] = _as_json(m["attributes"]) or {}
    m["weights"] = None if force else _as_json(m["weights"])
    m["quants"] = _as_json(m["quants"])
    specs = plans = 0
    if compute.weights_stale(m):
        m["weights"] = compute.compute_weights(m)
        hub = m["weights"].pop("hub", None)          # the full Hub record lives in its own column
        cur.execute("UPDATE models SET weights = %s, hub = COALESCE(%s, hub) WHERE id = %s", (J(m["weights"]), J(hub) if hub else None, model_id))
        _write_facts(cur, model_id, m["weights"])
        specs += 1
    for w in _rows(cur, PAIRS_SQL, (model_id,)):
        plan = None if force else _as_json(w["plan"])
        em = earmarks.get(w["worker_id"]) or {}
        inputs = compute.plan_inputs(m, em)
        if compute.plan_stale(plan, inputs):
            old_in = (plan or {}).get("inputs") or {}
            keep = None
            ch = em.get("last_change") or {}
            same_files = old_in.get("variant_sigs") == inputs.get("variant_sigs") and (plan or {}).get("version") == compute.PLAN_VERSION
            if (not force and same_files and old_in.get("budget_rev") is not None and old_in.get("budget_rev") != inputs.get("budget_rev")
                    and ch.get("reduced") and not ch.get("expanded")):
                keep = _prior_unfeasible(cur, model_id, w["worker_id"])      # reduction only: unfeasible stays
            full = compute.compute_plan(m, inputs, earmark=em, keep=keep)
            _write_verdicts(cur, model_id, w["worker_id"], full)
            summary = {k: v for k, v in full.items() if k != "by_quant"}
            summary["quants"] = list((full.get("by_quant") or {}).keys())
            cur.execute("UPDATE model_workers SET plan = %s, fits = %s WHERE model_id = %s AND worker_id = %s",
                        (J(summary), compute.plan_fits(full), model_id, w["worker_id"]))
            plans += 1
        knobs = _as_json(w["user_settings"]) or {}
        # QUANT LIST (operator 2026-10-02): verdicts changed → re-walk the pair's
        # ordered quant list and move the derived gguf_file if a different quant
        # now fits first (same fit-walk as the knob write — one truth).
        if isinstance(knobs.get("quants"), list) and knobs["quants"]:
            from hugpy_engine.model_index.query_registry import quant_fit_walk
            cur.execute("SELECT file, kv_cost->>'ctx_train' FROM model_quant_facts WHERE model_id=%s", (model_id,))
            facts = {r[0]: (int(r[1]) if r[1] else None) for r in cur.fetchall()}
            cur.execute("SELECT file, memory FROM model_worker_quants WHERE model_id=%s AND worker_id=%s", (model_id, w["worker_id"]))
            vmem = {r[0]: (_as_json(r[1]) or {}) for r in cur.fetchall()}
            chosen, _ = quant_fit_walk(knobs["quants"], vmem, ctx_pct=knobs.get("ctx_pct"), trained_by_file=facts,
                                       kv_cache_type=knobs.get("kv_cache_type") or "f16", bnb_on=knobs.get("bnb_4bit") is True)
            want = chosen or knobs["quants"][0]
            if knobs.get("gguf_file") != want:
                # placement knobs tuned for the previous quant → auto for the new one
                from hugpy_engine.model_index.query_registry import PLACEMENT_KNOB_KEYS
                drop = list(PLACEMENT_KNOB_KEYS) + ["tuned_for"] if (knobs.get("tuned_for") and knobs["tuned_for"] != want) else []
                cur.execute("UPDATE model_workers SET user_settings = (user_settings || %s) - %s::text[], updated_at = now()"
                            " WHERE model_id = %s AND worker_id = %s", (J({"gguf_file": want}), drop, model_id, w["worker_id"]))
                knobs["gguf_file"] = want
                for k in drop:
                    knobs.pop(k, None)
        # a `moe` knob on a pair whose SELECTED quant does not offer MoE is residue: drop it
        if knobs.get("moe") is not None:
            sel = _selected_verdict(cur, model_id, w["worker_id"], knobs) or {}
            if not sel.get("moe_offered"):
                cur.execute("UPDATE model_workers SET user_settings = user_settings - 'moe', updated_at = now()"
                            " WHERE model_id = %s AND worker_id = %s", (model_id, w["worker_id"]))
    return specs, plans


def ensure_all():
    if not _ensure_lock.acquire(blocking=False):
        return
    t0 = time.time()
    _ensure_state.update(running=True)
    specs = plans = 0
    try:
        with _connect(DSN) as conn, conn.cursor() as cur:
            reg = _registry(cur)
            ems = refresh_worker_budgets(cur, reg)      # the EARMARK, refreshed once per pass
            ids = [r["id"] for r in _rows(cur, "SELECT id FROM models ORDER BY id")]
        for mid in ids:
            try:
                with _connect(DSN) as conn, conn.cursor() as cur:
                    s, p = ensure_model(cur, mid, reg, earmarks=ems)
                specs += s; plans += p
                _ensure_state["errors"].pop(mid, None)
            except Exception as exc:  # noqa: BLE001 — one bad model must not stop the pass
                _ensure_state["errors"][mid] = f"{type(exc).__name__}: {exc}"
    finally:
        _ensure_state.update(running=False, last_run=time.time(), last_took_s=round(time.time() - t0, 1),
                             specs=_ensure_state["specs"] + specs, plans=_ensure_state["plans"] + plans)
        _ensure_lock.release()
    sys.stderr.write(f"ensure: {specs} specs, {plans} plans in {_ensure_state['last_took_s']}s, "
                     f"{len(_ensure_state['errors'])} errors\n")


def _ensure_loop():
    while True:
        try:
            ensure_all()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        time.sleep(ENSURE_S)


# ── http ─────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "hugpy-testshell/0.2"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=_json_default).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        return json.loads(raw.decode() or "{}")

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            if path.startswith("/api/"):
                return self._api_get(path[len("/api"):])
            return self._static(path)
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        try:
            if path.startswith("/api/"):
                return self._api_post(path[len("/api"):], self._body())
            self._send(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _api_get(self, path):
        t0 = time.time()
        with _connect(DSN) as conn, conn.cursor() as cur:
            if path == "/health":
                cur.execute("SELECT current_database(), now()")
                db, now = cur.fetchone()
                return self._send(200, {"ok": True, "db": db, "now": now, "tables": _rows(cur, TABLES_SQL),
                                        "ensure": _ensure_state, "knob_keys": compute.KNOB_KEYS,
                                        "alloc_modes": compute.ALLOC_MODES})
            if path == "/workers":
                return self._send(200, load_workers(cur))
            if path == "/models":
                models = load_models(cur)
                return self._send(200, {"models": models, "workers": load_workers(cur), "count": len(models),
                                        "ensure": _ensure_state, "alloc_modes": compute.ALLOC_MODES,
                                        "knob_keys": compute.KNOB_KEYS,
                                        "took_ms": round((time.time() - t0) * 1000)})
            if path.startswith("/models/"):
                mid = _int(path.rsplit("/", 1)[1])
                if mid is None:
                    return self._send(400, {"error": "model id must be an integer"})
                rows = load_models(cur, mid)
                return self._send(200, rows[0]) if rows else self._send(404, {"error": "no such model"})
        self._send(404, {"error": f"unknown endpoint {path}"})

    def _api_post(self, path, body):
        parts = path.strip("/").split("/")
        with _connect(DSN) as conn, conn.cursor() as cur:
            # POST /api/models/<id>/recompute
            if len(parts) == 3 and parts[0] == "models" and parts[2] == "recompute":
                mid = _int(parts[1])
                specs, plans = ensure_model(cur, mid, force=True)
                rows = load_models(cur, mid)
                return self._send(200, {"model": rows[0] if rows else None, "specs": specs, "plans": plans})
            # POST /api/models/<id>/workers/<worker_id>/knobs  {"set": {...}, "unset": [...]}
            if len(parts) == 5 and parts[0] == "models" and parts[2] == "workers" and parts[4] == "knobs":
                mid, wid = _int(parts[1]), parts[3]
                reg = _registry(cur)
                if mid is None or wid not in reg:
                    return self._send(400, {"error": "unknown model id or worker id"})
                to_set = body.get("set") or {}
                to_unset = [str(k) for k in (body.get("unset") or [])]
                bad = [k for k in list(to_set) + to_unset if k not in compute.KNOB_KEYS]
                if bad:
                    return self._send(400, {"error": f"unknown knob(s) {bad}; allowed: {list(compute.KNOB_KEYS)}"})
                if to_set.get("moe") is True:
                    cur.execute("SELECT user_settings FROM model_workers WHERE model_id = %s AND worker_id = %s", (mid, wid))
                    r = cur.fetchone()
                    knobs_now = dict(_as_json(r[0]) or {}) if r else {}
                    knobs_now.update(to_set)
                    for k in to_unset: knobs_now.pop(k, None)
                    sel = _selected_verdict(cur, mid, wid, knobs_now) or {}
                    if not sel.get("moe_offered"):
                        return self._send(409, {"error": "MoE is not offered for this quant on this worker — "
                                                         + ((sel.get("moe") or {}).get("why") or "no verdict computed yet")})
                # Knobs ride an EXISTING pair only. A pair row exists when the
                # model is assigned to (or live on) the worker; a knob written
                # for any other worker would be residue (operator, 2026-10-01).
                cur.execute(
                    "UPDATE model_workers SET"
                    "   user_settings = (COALESCE(user_settings, '{}'::jsonb) || %s::jsonb) - %s::text[],"
                    "   updated_at = now()"
                    " WHERE model_id = %s AND worker_id = %s",
                    (J(to_set), to_unset, mid, wid))
                if cur.rowcount == 0:
                    return self._send(409, {"error": "model is not assigned to this worker — assign it first; "
                                                     "knobs on an unassigned pair would be residue"})
                specs, plans = ensure_model(cur, mid, reg)
                rows = load_models(cur, mid)
                return self._send(200, {"model": rows[0] if rows else None, "plans": plans})
            if path == "/ensure":
                threading.Thread(target=ensure_all, daemon=True).start()
                return self._send(202, {"started": not _ensure_state["running"]})
        self._send(404, {"error": f"unknown endpoint {path}"})

    def _static(self, path):
        if not DIST.is_dir():
            return self._send(503, {"error": "dist/ not built; run `npm run build` or use the vite dev server"})
        target = (DIST / path.lstrip("/")).resolve()
        if not str(target).startswith(str(DIST)) or not target.is_file():
            target = DIST / "index.html"
        ctype = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
                 ".svg": "image/svg+xml", ".json": "application/json"}.get(target.suffix, "application/octet-stream")
        self._send(200, target.read_bytes(), ctype)


def _int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    with _connect(DSN) as c, c.cursor() as cur:
        cur.execute("SELECT count(*), count(weights) FROM models")
        n, done = cur.fetchone()
        print(f"hugpy testshell api :{PORT}  dsn={DSN!r}  models={n}  weights filled={done}", flush=True)
    threading.Thread(target=_ensure_loop, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
