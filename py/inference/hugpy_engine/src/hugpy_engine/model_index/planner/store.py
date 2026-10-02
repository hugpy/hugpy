"""hugpy PLANNER — DB side (moved from react/testshell/api.py 2026-10-02).

Writes the computed-once facts: worker_budgets (the EARMARK), models.weights,
model_quant_facts, model_workers.plan and model_worker_quants (the verdicts),
and re-walks a pair's `quants` ladder when its verdicts change.
Pure compute lives in .compute. No HTTP here: a host process (the testshell
today) runs ensure_all() on its own cadence and exposes routes.

  HUGPY_PLANNER_DSN   libpq DSN (falls back to HUGPY_TESTSHELL_DSN, then dbname=hugpy)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback  # noqa: F401  (kept for hosts that log through this module)
from datetime import date, datetime
from decimal import Decimal

from . import compute

try:  # /srv/hugpy/venv has psycopg 3; system python has psycopg2
    import psycopg as _pg
    from psycopg.types.json import Jsonb as J
    def _connect(dsn): return _pg.connect(dsn, autocommit=True)
except ImportError:  # pragma: no cover
    import psycopg2 as _pg
    from psycopg2.extras import Json as J
    def _connect(dsn):
        c = _pg.connect(dsn); c.autocommit = True; return c

DSN = os.environ.get("HUGPY_PLANNER_DSN") or os.environ.get("HUGPY_TESTSHELL_DSN", "dbname=hugpy")

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


def ensure_all(dsn=None):
    dsn = dsn or DSN
    if not _ensure_lock.acquire(blocking=False):
        return
    t0 = time.time()
    _ensure_state.update(running=True)
    specs = plans = 0
    try:
        with _connect(dsn) as conn, conn.cursor() as cur:
            reg = _registry(cur)
            ems = refresh_worker_budgets(cur, reg)      # the EARMARK, refreshed once per pass
            ids = [r["id"] for r in _rows(cur, "SELECT id FROM models ORDER BY id")]
        for mid in ids:
            try:
                with _connect(dsn) as conn, conn.cursor() as cur:
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


# ── discovery (operator 2026-10-02: "a targeted discovery would be ideal") ──
# DISCOVERY is the hugpy action that re-reads model files. Targeted = ONE model:
# resync its marker's quant list from disk, re-stamp its weights facts (forced:
# the operator asked), mirror the quants into model_quants, then recompute its
# facts + every pair's verdicts. The full pass runs the change-driven version
# over every model at the end of POST /models/discover.
def _model_dir(cur, model_id):
    cur.execute("SELECT name, attributes->>'dir' FROM models WHERE id = %s", (model_id,))
    r = cur.fetchone()
    return (r[0], r[1]) if r else (None, None)


def _ensure_marker(cur, model_id, d):
    """ASSURE A hugpy.json (operator 2026-10-02: "that's what the discovery
    should do, assure a hugpy.json"). Returns (marker, created). A dir without
    one gets its identity stamped from the model's DB row through the same
    writer downloads use (source "discovery"); the facts are stamped after."""
    from hugpy_storage import hugpy_marker as hm
    m = hm.read_hugpy_marker(d)
    if isinstance(m, dict):
        return m, False
    cur.execute("SELECT name, hub_id, framework, attributes FROM models WHERE id = %s", (model_id,))
    r = cur.fetchone()
    if not r:
        return None, False
    name, hub_id, framework, attrs = r[0], r[1], r[2], (_as_json(r[3]) or {})
    hm.write_hugpy_marker(d, hub_id=hub_id or attrs.get("hub_id") or name, name=name,
                          framework=framework or attrs.get("framework"),
                          tasks=attrs.get("tasks"), primary_task=attrs.get("primary_task"),
                          filename=attrs.get("filename"), source="discovery")
    return hm.read_hugpy_marker(d), True


def _sync_model_quants(cur, name, marker):
    from hugpy_engine.model_index.repositories import QuantsRepository
    QuantsRepository(None).replace_for_model(cur, name, (marker or {}).get("quants"))


def discover_model(cur, model_id, reg=None, earmarks=None):
    """Targeted discovery for one model. Returns a summary dict (never raises
    for a model-level problem; the problem is in `error`)."""
    from hugpy_storage import hugpy_marker as hm
    t0 = time.time()
    name, d = _model_dir(cur, model_id)
    out = {"model_id": model_id, "name": name, "dir": d, "error": None}
    if name is None:
        out["error"] = "no such model"
        return out
    if not d or not os.path.isdir(d):
        out["error"] = "model dir not on disk"
        return out
    try:
        m, out["marker_created"] = _ensure_marker(cur, model_id, d)
    except Exception as exc:  # noqa: BLE001
        m, out["marker_error"] = None, f"{type(exc).__name__}: {exc}"
    if not isinstance(m, dict):
        out["error"] = "could not write a hugpy.json in the model dir" + (f" ({out.get('marker_error')})" if out.get("marker_error") else "")
        return out
    try:
        m = hm.sync_marker_quants(d, marker=m, write=True) or m
    except Exception as exc:  # noqa: BLE001
        out["quants_error"] = f"{type(exc).__name__}: {exc}"
    try:
        m = hm.stamp_marker_weights_facts(d, marker=m, write=True, force=True) or m
    except Exception as exc:  # noqa: BLE001
        out["facts_error"] = f"{type(exc).__name__}: {exc}"
    _sync_model_quants(cur, name, m)
    out["specs"], out["plans"] = ensure_model(cur, model_id, reg, force=True, earmarks=earmarks)
    files = ((m.get(hm.WEIGHTS_FACTS_KEY) or {}).get("files")) or {}
    out["quants"] = [q.get("file") for q in (m.get("quants") or []) if isinstance(q, dict)]
    out["facts"] = {k: {"is_moe": bool(v.get("is_moe")), "size_bytes": v.get("size_bytes"),
                        "shards": v.get("shards"), "error": v.get("error")} for k, v in files.items()}
    cur.execute("SELECT w.worker_id, w.assigned, count(v.file), bool_or(v.fits), bool_or(v.moe_offered),"
                " bool_or(v.memory->'explicit' ? 'band') FROM model_workers w"
                " LEFT JOIN model_worker_quants v ON v.model_id = w.model_id AND v.worker_id = w.worker_id"
                " WHERE w.model_id = %s GROUP BY 1, 2", (model_id,))
    out["workers"] = {r[0]: {"assigned": r[1], "verdicts": r[2], "fits": r[3], "moe_offered": r[4], "explicit_band": r[5]}
                      for r in cur.fetchall()}
    if not files:
        out["error"] = "no weights file found in the model dir"
    out["took_s"] = round(time.time() - t0, 2)
    return out


def discover_all(dsn=None):
    """End-of-walk planner pass for the full discovery: re-stamp every model's
    weights facts CHANGE-DRIVEN (a stat per file; headers only for a changed or
    missing file), then ensure every model — forced for a model whose stored
    weights carry an error, so the walk retries it now instead of hourly."""
    from hugpy_storage import hugpy_marker as hm
    dsn = dsn or DSN
    t0 = time.time()
    res = {"models": 0, "markers_created": 0, "stamped": 0, "specs": 0, "plans": 0, "no_dir": 0, "errors": {}}
    with _connect(dsn) as conn, conn.cursor() as cur:
        reg = _registry(cur)
        ems = refresh_worker_budgets(cur, reg)
        rows = _rows(cur, "SELECT id, name, attributes->>'dir' AS dir, (weights->>'error') IS NOT NULL AS err FROM models ORDER BY id")
    for r in rows:
        res["models"] += 1
        try:
            with _connect(dsn) as conn, conn.cursor() as cur:
                d = r["dir"]
                if not d or not os.path.isdir(d):
                    res["no_dir"] += 1
                if d and os.path.isdir(d):
                    m, created = _ensure_marker(cur, r["id"], d)
                    res["markers_created"] += int(bool(created))
                    if isinstance(m, dict):
                        m = hm.sync_marker_quants(d, marker=m, write=True) or m
                        if hm.stamp_marker_weights_facts(d, marker=m, write=True) is not None:
                            res["stamped"] += 1
                            m = hm.read_hugpy_marker(d) or m
                        _sync_model_quants(cur, r["name"], m)
                s, p = ensure_model(cur, r["id"], reg, force=bool(r["err"]), earmarks=ems)
                res["specs"] += s; res["plans"] += p
        except Exception as exc:  # noqa: BLE001 — one bad model must not stop the pass
            res["errors"][r["id"]] = f"{type(exc).__name__}: {exc}"
    res["took_s"] = round(time.time() - t0, 1)
    return res
