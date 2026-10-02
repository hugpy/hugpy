"""CALIBRATE RUN — measured ground truth for one (model, worker) pair.

WHY (operator 2026-10-02): every sizing bug of the day was a formula
disagreeing with reality (hybrid KV priced 4x, VL configs read as
geometry-less, the DB context never reaching the gate, evicted weights left on
the card). A prediction is only checkable against a measurement, so the
operator gets a controlled load that records both:

  1. PRECHECK  the card must be idle of other evictable models (a calibration
               measures ONE model). ``evict_others`` clears on-demand residents
               first through the worker's own /ops/evict; static / protected
               residents are never touched — a busy card refuses instead.
  2. PREDICT   the worker's dry-run admission (/fit-preview) — what the load
               gate prices: weights x margin, KV at the effective ctx.
  3. LOAD      one pinned one-token chat through central's own /v1 (the real
               routing -> admission -> seat path, with the DB pair knobs).
  4. MEASURE   /ops/residents (the model's own VRAM as the worker attributes
               it) + /ops/vram-holders (device used).
  5. UNLOAD    the worker's /ops/evict, then measure again: what the eviction
               left on the card (the stranded-weights leak).
  6. RECORD    a row in model_worker_calibration (predicted vs measured).

Nothing here edits code or knobs. The ticket / help-agent loop reads these rows.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

SELF_BASE = os.environ.get("HUGPY_SELF_BASE", "http://127.0.0.1:7002")
LOAD_TIMEOUT_S = float(os.environ.get("HUGPY_CALIBRATE_LOAD_TIMEOUT_S", "900"))
SETTLE_S = float(os.environ.get("HUGPY_CALIBRATE_SETTLE_S", "3"))
# |measured - predicted| / measured above this is a DISAGREEMENT worth a ticket
TOLERANCE = float(os.environ.get("HUGPY_CALIBRATE_TOLERANCE", "0.10"))
# what an unload may leave behind on the card before it is called a leak
LEAK_BYTES = int(float(os.environ.get("HUGPY_CALIBRATE_LEAK_MIB", "512")) * 2 ** 20)

DDL = """
CREATE TABLE IF NOT EXISTS model_worker_calibration (
    id            BIGSERIAL PRIMARY KEY,
    model_id      BIGINT,
    model_key     TEXT NOT NULL,
    worker_id     TEXT NOT NULL,
    worker_name   TEXT,
    file          TEXT,
    gpu           TEXT,
    bnb_4bit      BOOLEAN NOT NULL DEFAULT false,
    ctx           INTEGER,
    ctx_source    TEXT,
    predicted     JSONB,
    measured      JSONB,
    verdict       TEXT,
    detail        JSONB,
    at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS model_worker_calibration_pair
    ON model_worker_calibration (model_key, worker_id, at DESC);
"""

_JOBS: dict = {}
_LOCK = threading.Lock()
_BUSY: set = set()          # worker ids with a calibration in flight (one per card)


def _now() -> float:
    return time.time()


def _job_update(jid: str, **kw) -> None:
    with _LOCK:
        j = _JOBS.get(jid)
        if j is not None:
            j.update(kw)
            j["updated_at"] = _now()


def _step(jid: str, text: str) -> None:
    with _LOCK:
        j = _JOBS.get(jid)
        if j is not None:
            j.setdefault("steps", []).append({"at": _now(), "text": text})
            j["updated_at"] = _now()


def get_job(jid: str) -> Optional[dict]:
    with _LOCK:
        j = _JOBS.get(jid)
        return json.loads(json.dumps(j, default=str)) if j else None


def _worker_get(worker: dict, path: str, params: Optional[dict] = None, timeout: float = 20):
    from hugpy_fleet.central import worker_http
    r = worker_http.get(worker, path, params=params, read_timeout=timeout)
    return r.json()


def _worker_post(worker: dict, path: str, body: dict, timeout: float = 120):
    from hugpy_fleet.central import worker_http
    r = worker_http.post(worker, path, json=body, read_timeout=timeout)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"error": f"HTTP {r.status_code} with no JSON"}


def _residents(worker: dict) -> list:
    d = _worker_get(worker, "/ops/residents") or {}
    return [r for r in d.get("residents") or [] if isinstance(r, dict)]


def _device(worker: dict) -> dict:
    d = _worker_get(worker, "/ops/vram-holders") or {}
    cards = d.get("cards") or []
    return {"used": d.get("vram_used_bytes"), "free": d.get("vram_free_bytes"),
            "total": d.get("vram_total_bytes"),
            "cards": [{"index": c.get("index"), "used": c.get("mem_used_bytes"),
                       "free": c.get("mem_free_bytes")} for c in cards],
            "holders": [{"kind": h.get("kind"), "model_key": h.get("model_key"),
                         "vram_bytes": h.get("vram_bytes"), "pid": h.get("pid")}
                        for h in d.get("holders") or []]}


def _gpu_name(worker: dict) -> Optional[str]:
    try:
        gpus = worker.get("gpus") or []
        if gpus and isinstance(gpus[0], dict):
            return gpus[0].get("name")
        return worker.get("gpu")
    except Exception:  # noqa: BLE001
        return None


def _chat(model_key: str, worker_name: str) -> dict:
    body = json.dumps({"model": model_key, "max_tokens": 1,
                       "messages": [{"role": "user", "content": "Say OK."}],
                       "alloc": {"worker": worker_name}}).encode()
    req = urllib.request.Request(SELF_BASE.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = _now()
    try:
        with urllib.request.urlopen(req, timeout=LOAD_TIMEOUT_S) as r:
            d = json.load(r)
    except urllib.error.HTTPError as exc:
        try:
            d = json.loads(exc.read().decode() or "{}")
        except Exception:  # noqa: BLE001
            d = {"error": f"HTTP {exc.code}"}
    except Exception as exc:  # noqa: BLE001
        d = {"error": f"{type(exc).__name__}: {exc}"}
    d["_took_s"] = round(_now() - t0, 2)
    return d


def _pct(a, b) -> Optional[float]:
    try:
        return round((float(a) - float(b)) / float(b) * 100.0, 1) if b else None
    except (TypeError, ValueError):
        return None


def _record(row: dict) -> Optional[int]:
    from hugpy_engine.model_index import enabled, resolve_model_id
    if not enabled():
        return None
    from hugpy_engine.model_index.client import resolve_dsn
    from hugpy_engine.model_index.planner.store import J, _connect
    mid = resolve_model_id(row["model_key"])
    with _connect(resolve_dsn()) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        cur.execute(
            "INSERT INTO model_worker_calibration (model_id, model_key, worker_id, worker_name, file, gpu,"
            " bnb_4bit, ctx, ctx_source, predicted, measured, verdict, detail)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            (mid, row["model_key"], row["worker_id"], row.get("worker_name"), row.get("file"),
             row.get("gpu"), bool(row.get("bnb")), row.get("ctx"), row.get("ctx_source"),
             J(row.get("predicted") or {}), J(row.get("measured") or {}), row.get("verdict"),
             J(row.get("detail") or {})))
        return cur.fetchone()[0]


def history(model_key: str, worker_id: Optional[str] = None, limit: int = 20) -> list:
    from hugpy_engine.model_index.client import resolve_dsn
    from hugpy_engine.model_index.planner.store import _connect, _rows
    with _connect(resolve_dsn()) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        q = ("SELECT id, model_key, worker_id, worker_name, file, gpu, bnb_4bit, ctx, ctx_source,"
             " predicted, measured, verdict, detail, at::text AS at FROM model_worker_calibration"
             " WHERE model_key = %s" + (" AND worker_id = %s" if worker_id else "")
             + " ORDER BY at DESC LIMIT %s")
        return _rows(cur, q, (model_key, worker_id, int(limit)) if worker_id else (model_key, int(limit)))


def _run(jid: str, worker: dict, model_key: str, bnb: bool, evict_others: bool) -> None:
    wid = str(worker.get("id"))
    name = worker.get("name") or wid
    try:
        # 1. PRECHECK — one model on the card
        res = _residents(worker)
        if any(r.get("model_key") == model_key for r in res):
            raise RuntimeError(f"{model_key} is already loaded on {name} — unload it first, so the "
                               "baseline is measured without it")
        others = [r for r in res if r.get("model_key") != model_key]
        if others:
            if not evict_others:
                raise RuntimeError("card not idle: " + ", ".join(
                    f"{r.get('model_key')} ({r.get('residency')}, {round((r.get('vram_bytes') or 0) / 2**30, 1)} GiB)"
                    for r in others) + " — run with evict_others to clear on-demand residents first")
            for r in others:
                if str(r.get("residency") or "").lower() == "static":
                    raise RuntimeError(f"card holds a STATIC resident {r.get('model_key')} — a calibration "
                                       "never evicts static models; unload it yourself first")
            for r in others:
                _step(jid, f"evicting on-demand resident {r.get('model_key')} for a clean baseline")
                st, body = _worker_post(worker, "/ops/evict", {"model_key": r.get("model_key")})
                if st >= 400 or (isinstance(body, dict) and body.get("ok") is False):
                    raise RuntimeError(f"could not evict {r.get('model_key')}: {body}")
            time.sleep(SETTLE_S)
        # 2. PREDICT
        _step(jid, "asking the load gate for its dry-run prediction")
        pred = _worker_get(worker, "/fit-preview/" + urllib.request.quote(model_key, safe=""),
                           params={"bnb": "1"} if bnb else None) or {}
        x = pred.get("ctx") or {}
        nd = pred.get("need_detail") or {}
        # 3. BASELINE
        base = _device(worker)
        _step(jid, f"baseline: {round((base.get('used') or 0) / 2**30, 2)} GiB used on the card")
        # 4. LOAD — one pinned one-token chat through the real path
        _step(jid, f"loading via a pinned one-token chat on {name}")
        chat = _chat(model_key, name)
        answered = isinstance(chat, dict) and "choices" in chat
        time.sleep(SETTLE_S)
        loaded = _device(worker)
        res1 = _residents(worker)
        mine = next((r for r in res1 if r.get("model_key") == model_key), None)
        if not answered:
            err = chat.get("error") if isinstance(chat, dict) else chat
            raise RuntimeError(f"the load did not answer: {json.dumps(err, default=str)[:600]}")
        # 4b. RECORD THE MARGIN (2026-10-02) — a full load measured alone on an
        # idle card IS the weights measurement the gate prices from; the worker
        # nets out KV x sequences with the heartbeat learner's own arithmetic.
        # A 4-bit load records its OWN margin (bnb_4bit: the worker prices it
        # against the 4-bit file figure, under <key>#bnb-4bit); never for a
        # planned partial (only part of the weights are on the card).
        margin_rec = None
        d_used = (loaded.get("used") - base.get("used")) \
            if (loaded.get("used") is not None and base.get("used") is not None) else None
        if pred.get("action") == "proceed" and d_used and d_used > 0:
            _step(jid, "recording the measured weights margin on the worker")
            st_m, margin_rec = _worker_post(worker, "/ops/weights-margin",
                                            {"model_key": model_key, "delta_bytes": int(d_used),
                                             "ctx": x.get("resolved"), "bnb_4bit": bool(bnb)})
            if st_m >= 400 or not (isinstance(margin_rec, dict) and margin_rec.get("ok")):
                _step(jid, f"margin not recorded: {(margin_rec or {}).get('error') if isinstance(margin_rec, dict) else margin_rec}")
        # 5. UNLOAD — the worker's own eviction, then what it left behind
        _step(jid, "unloading through the worker's eviction path")
        st, ev = _worker_post(worker, "/ops/evict", {"model_key": model_key})
        time.sleep(SETTLE_S)
        after = _device(worker)
        used0, used1, used2 = base.get("used"), loaded.get("used"), after.get("used")
        measured = {
            "resident_vram_bytes": (mine or {}).get("vram_bytes"),
            "host_mode": (mine or {}).get("host_mode"),
            "device_used_delta_bytes": (used1 - used0) if (used0 is not None and used1 is not None) else None,
            "leftover_after_unload_bytes": (used2 - used0) if (used0 is not None and used2 is not None) else None,
            "evict_status": st, "evict": ev, "load_took_s": chat.get("_took_s"),
            "margin": margin_rec,
        }
        load_bytes = measured["resident_vram_bytes"] or measured["device_used_delta_bytes"]
        predicted = {"gate_need_bytes": pred.get("need_bytes"), "gate_weights_bytes": pred.get("weights_bytes"),
                     "gate_kv_bytes": pred.get("kv_bytes"), "gate_action": pred.get("action"),
                     "weights_margin": nd.get("weights_margin"), "weights_margin_source": nd.get("weights_margin_source"),
                     "calibration_correction": nd.get("calibration_correction"),
                     "weights_file_bytes": nd.get("weights_file_bytes")}
        err_pct = _pct(predicted["gate_need_bytes"], load_bytes)
        leak = measured["leftover_after_unload_bytes"]
        verdict = ("leak" if (leak is not None and leak > LEAK_BYTES)
                   else "disagree" if (err_pct is not None and abs(err_pct) > TOLERANCE * 100)
                   else "agree" if err_pct is not None else "unmeasured")
        row = {"model_key": model_key, "worker_id": wid, "worker_name": name,
               "file": nd.get("weights_file"), "gpu": _gpu_name(worker), "bnb": bnb,
               "ctx": x.get("resolved"), "ctx_source": x.get("source"),
               "predicted": predicted, "measured": measured, "verdict": verdict,
               "detail": {"gate_error_pct": err_pct, "tolerance_pct": TOLERANCE * 100,
                          "leak_threshold_bytes": LEAK_BYTES, "baseline": base, "loaded": loaded,
                          "after_unload": after, "preview": pred, "chat_status": chat.get("hugpy_status")}}
        try:
            row["id"] = _record(row)
        except Exception as exc:  # noqa: BLE001 — the result is still returned
            row["record_error"] = f"{type(exc).__name__}: {exc}"
        _step(jid, f"done: {verdict}" + (f" (gate {err_pct:+.1f}% vs measured)" if err_pct is not None else ""))
        _job_update(jid, status="done", result=row)
        logger.info("calibration %s on %s: %s gate_err=%s%% leftover=%s", model_key, name, verdict, err_pct, leak)
    except Exception as exc:  # noqa: BLE001 — a calibration reports, never raises
        _step(jid, f"stopped: {exc}")
        _job_update(jid, status="error", error=str(exc))
        logger.warning("calibration %s on %s stopped: %s", model_key, name, exc)
    finally:
        with _LOCK:
            _BUSY.discard(wid)


def start(worker: dict, model_key: str, *, bnb: bool = False, evict_others: bool = False) -> dict:
    wid = str(worker.get("id"))
    with _LOCK:
        if wid in _BUSY:
            return {"error": "a calibration is already running on this worker"}
        _BUSY.add(wid)
        jid = "cal-" + uuid.uuid4().hex[:12]
        _JOBS[jid] = {"job_id": jid, "model_key": model_key, "worker_id": wid,
                      "worker_name": worker.get("name"), "status": "running",
                      "bnb": bnb, "evict_others": evict_others,
                      "started_at": _now(), "updated_at": _now(), "steps": []}
    threading.Thread(target=_run, args=(jid, worker, model_key, bnb, evict_others),
                     name=f"calibrate-{model_key[:24]}", daemon=True).start()
    return get_job(jid)
