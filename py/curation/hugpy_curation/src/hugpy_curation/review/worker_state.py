"""Back up and restore each involved worker's allocation state around a run.

Operator ruling (2026-09-24): the capacity benchmark runs on ALL SELECTED
workers, designated or not.  The ONLY allocation-aware step it may impose is to
back up each involved worker's state BEFORE the test and restore it AFTER — on
completion, cancel, failure, or a central restart mid-run.  "State" is exactly:

  1. the model allocations to the worker — designations/assignments, pins,
     including designation source/meta;
  2. every per-model variable the benchmark may alter — per-worker quant choice
     (``gguf_file_by_worker``), and the per-(worker, model) moe / bnb / spill
     levers, plus the per-worker wildcard / limits;
  3. the worker's hard-drive occupancy — which model files were on its drive
     (``models_local``).

The snapshot is a plain JSON-able dict so review_routes can persist it WITH the
run and a restarted central can restore from it (never re-snapshotting the
already-mutated live state).  ``restore`` re-reads the live state and issues only
the writes needed to converge back to the snapshot, so a run that changed
nothing restores nothing — and it records, per worker, exactly what it did and
every failure with its real reason (metrics/grading contract: explicit absences,
no truncation).

Everything here goes through central's own HTTP routes (the ``fleet_grading``
Client): GET /llm/workers, GET /llm/serving, and the per-worker POST routes
(/assign, /unassign, /pin, /moe, /bnb, /limits, /wildcard, /evict, /cache-evict,
/fetch) plus POST /llm/serving/<model>.  Nothing is written to a worker box or a
venv directly; disk re-provision uses the FETCH-ONLY verb (/fetch — download to
the worker's drive without loading VRAM), the normal hugpy provisioning path,
never rsync/HF ad hoc.  Restore only puts prior models back ON DISK; it never
pins or protects (any later eviction is natural and fine).
"""
from __future__ import annotations

import time
from urllib.parse import quote

from hugpy_curation.review.fleet_grading import eligible, model_id, response_rows, verbose_catalog


def _wkey(worker):
    return worker.get("id") or worker.get("name")


def _gb(n):
    try:
        return f"{float(n) / 1e9:.1f} GB"
    except (TypeError, ValueError):
        return "unknown"


def _designation_pins(worker):
    """{model_key: bool} pinned state, from the public ``designations`` rows."""
    out = {}
    for row in worker.get("designations") or []:
        mk = row.get("model_key")
        if mk is not None:
            out[mk] = bool(row.get("pinned"))
    return out


def _designation_sources(worker):
    """{model_key: source} for every designated model (provenance, for the record)."""
    out = {}
    for row in worker.get("designations") or []:
        mk = row.get("model_key")
        if mk is not None:
            out[mk] = row.get("source")
    return out


def _serving_gguf_map(client, model_ids):
    """{model_key: gguf_file_by_worker dict} in ONE GET /llm/serving, filtered to
    ``model_ids`` when given (else every model with an override).  Best effort:
    an unreadable listing yields {} and the run still snapshots the rest."""
    try:
        rows = client.request("/llm/serving")
    except Exception:  # noqa: BLE001 — a missing listing is not a snapshot failure
        return {}, "GET /llm/serving failed"
    if not isinstance(rows, list):
        return {}, "GET /llm/serving returned no list"
    wanted = set(model_ids) if model_ids else None
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row.get("key") or row.get("model_key") or row.get("model")
        if key is None or (wanted is not None and key not in wanted):
            continue
        override = row.get("override") if isinstance(row.get("override"), dict) else {}
        out[key] = dict(override.get("gguf_file_by_worker") or {})
    return out, None


def snapshot(client, worker_ids=None, model_ids=None):
    """Capture the involved workers' state → a JSON-able dict.

    ``worker_ids`` (ids or names; None = every eligible worker) and ``model_ids``
    (None = every catalog model) scope exactly what the run will touch, mirroring
    ``run_capacity_benchmark``'s own eligible+selection filtering.
    """
    at = time.time()
    errors = []
    try:
        all_workers = response_rows(client.request("/llm/workers"), "workers")
    except Exception as exc:  # noqa: BLE001
        return {"at": at, "error": f"GET /llm/workers failed: {type(exc).__name__}: {exc}",
                "workers": [], "models": [], "per_worker": {}, "serving": {},
                "scope": {"worker_ids": worker_ids, "model_ids": model_ids}}
    wanted_workers = set(worker_ids) if worker_ids else None
    involved = [w for w in all_workers if eligible(w)
                and (wanted_workers is None or _wkey(w) in wanted_workers
                     or w.get("name") in wanted_workers or w.get("id") in wanted_workers)]

    # Catalog read (best effort, once): the model scope for a full run AND a
    # fallback per-model size (effective_bytes / size_bytes) for disk fit math.
    catalog_sizes = {}
    catalog_models = None
    try:
        catalog_models = verbose_catalog(client)
    except Exception as exc:  # noqa: BLE001 — degrade; storage rows still give sizes
        errors.append(f"catalog read failed: {type(exc).__name__}: {exc}")
    for m in catalog_models or []:
        mk = model_id(m)
        size = m.get("effective_bytes") or m.get("size_bytes")
        if mk and isinstance(size, (int, float)) and size > 0:
            catalog_sizes[mk] = int(size)

    # Model scope: the explicit list, else every model central knows (so a full
    # run's snapshot covers everything the benchmark could touch).
    if model_ids:
        scope_models = [str(m) for m in model_ids]
    elif catalog_models is not None:
        scope_models = [mk for mk in (model_id(m) for m in catalog_models) if mk]
    else:
        scope_models = None

    serving_map, serving_err = _serving_gguf_map(client, scope_models)
    if serving_err:
        errors.append(serving_err)

    per_worker = {}
    for w in involved:
        wid = _wkey(w)
        # DISK-TRUTH = every model on record on this drive, NOT scoped to the
        # test (operator ruling 2026-09-24: all prior models must be back on the
        # drive after the test). ``models_local`` is central's disk-truth list.
        models_local = list(w.get("models_local") or [])
        storage = w.get("storage") if isinstance(w.get("storage"), dict) else {}
        # Per-model on-disk sizes: the worker's own storage survey first (actual
        # bytes on the drive), then the catalog. So a missing prior model can be
        # sized for the fit check, and a current extra sized for freeing.
        model_sizes = {}
        for row in (storage.get("models") or []):
            if isinstance(row, dict) and row.get("model_key") and isinstance(row.get("bytes"), (int, float)):
                model_sizes[row["model_key"]] = int(row["bytes"])
        for mk in models_local:
            if mk not in model_sizes and mk in catalog_sizes:
                model_sizes[mk] = catalog_sizes[mk]
        per_worker[wid] = {
            "worker": w.get("name") or wid,
            "worker_id": w.get("id"),
            # (1) allocations: designation membership, pins, provenance, spill.
            "models": sorted(set(w.get("models") or [])),
            "pins": _designation_pins(w),
            "sources": _designation_sources(w),
            "designation_meta": dict(w.get("designation_meta") or {}),
            "spill_by_model": dict(w.get("spill_by_model") or {}),
            # (2) per-model / per-worker levers the benchmark may alter.
            "moe_by_model": dict(w.get("moe_by_model") or {}),
            "bnb_by_model": dict(w.get("bnb_by_model") or {}),
            "limits": dict(w.get("limits") or {}),
            "wildcard": bool(w.get("wildcard")),
            # (3) disk occupancy: the FULL prior model set on the drive, the
            # capacity/free the worker reports, and per-model sizes for fit math.
            "models_local": models_local,
            "on_disk": models_local,
            "model_sizes": model_sizes,
            "disk_free": storage.get("disk_free"),
            "disk_total": storage.get("disk_total"),
        }

    return {
        "at": at,
        "scope": {"worker_ids": worker_ids, "model_ids": model_ids},
        "workers": [pw["worker"] for pw in per_worker.values()],
        "worker_ids": list(per_worker.keys()),
        "models": scope_models or [],
        "serving": serving_map,
        "per_worker": per_worker,
        "errors": errors,
    }


def _post(client, path, body):
    """POST that returns (ok, detail): errors are data, never exceptions."""
    try:
        res = client.request(path, "POST", body)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    if isinstance(res, dict) and res.get("ok") is False:
        err = res.get("error")
        if isinstance(err, dict):
            err = err.get("message") or err.get("code") or str(err)
        return False, err or "worker refused the operation"
    return True, res


def restore(client, snap):
    """Converge each involved worker back to ``snap``; return a per-worker report.

    Re-reads the live state and writes ONLY the differences, so a run that
    changed nothing restores nothing.  Report per worker::

        {worker, worker_id, restored:[...], redownloaded:[...], removed:[...],
         failures:[{item, error}]}

    ``restored`` names each converged item; ``redownloaded`` names models re-
    provisioned to disk (they were on the drive before and the test evicted
    them); ``removed`` names test-downloaded models deleted only to fit a prior
    model back (not attempted automatically — see the note in the report);
    ``failures`` carries every write that did not take, with its real reason.
    """
    if not isinstance(snap, dict) or not snap.get("per_worker"):
        return {"at": time.time(), "workers": [],
                "error": "no snapshot to restore from" if isinstance(snap, dict) else "snapshot is not a dict"}

    # Live state to diff against.
    try:
        now_workers = {(_wkey(w)): w for w in response_rows(client.request("/llm/workers"), "workers")}
        by_name = {w.get("name"): w for w in now_workers.values()}
    except Exception as exc:  # noqa: BLE001
        return {"at": time.time(), "workers": [],
                "error": f"GET /llm/workers failed at restore: {type(exc).__name__}: {exc}"}
    now_serving, _serr = _serving_gguf_map(client, list(snap.get("serving") or {}) or None)

    reports = []
    for wid, snap_w in snap["per_worker"].items():
        w = now_workers.get(wid) or by_name.get(snap_w.get("worker"))
        rep = {"worker": snap_w.get("worker"), "worker_id": snap_w.get("worker_id"),
               "restored": [], "redownloaded": [], "removed": [], "failures": []}
        base = "/llm/workers/" + quote(str(wid), safe="")
        if w is None:
            rep["failures"].append({"item": "worker", "error": "worker not in the registry at restore time"})
            reports.append(rep)
            continue

        # (1a) designation membership.
        snap_models = set(snap_w.get("models") or [])
        now_models = set(w.get("models") or [])
        for mk in sorted(now_models - snap_models):
            # A designation that appeared during the test (the benchmark itself
            # never adds one; defensive) — remove it.
            ok, detail = _post(client, base + "/unassign", {"model_key": mk})
            (rep["restored"].append(f"unassigned added designation {mk}") if ok
             else rep["failures"].append({"item": f"unassign {mk}", "error": str(detail)}))
        for mk in sorted(snap_models - now_models):
            spill = (snap_w.get("spill_by_model") or {}).get(mk)
            ok, detail = _post(client, base + "/assign",
                               {"model_key": mk, **({"spill": spill} if spill else {})})
            (rep["restored"].append(f"re-designated {mk}") if ok
             else rep["failures"].append({"item": f"assign {mk}", "error": str(detail)}))

        # (1b) pins.
        now_pins = _designation_pins(w)
        for mk, want in (snap_w.get("pins") or {}).items():
            if bool(now_pins.get(mk)) != bool(want):
                ok, detail = _post(client, base + "/pin", {"model_key": mk, "pinned": bool(want)})
                (rep["restored"].append(f"pin {mk}={'on' if want else 'off'}") if ok
                 else rep["failures"].append({"item": f"pin {mk}", "error": str(detail)}))

        # (1c) spill (placement contract) for models still designated.
        now_spill = dict(w.get("spill_by_model") or {})
        for mk in snap_models & now_models:
            want = (snap_w.get("spill_by_model") or {}).get(mk) or {}
            if (now_spill.get(mk) or {}) != want:
                ok, detail = _post(client, base + "/assign", {"model_key": mk, "spill": want})
                (rep["restored"].append(f"spill {mk}") if ok
                 else rep["failures"].append({"item": f"spill {mk}", "error": str(detail)}))

        # (2a) moe / (2b) bnb per (worker, model).
        now_moe = dict(w.get("moe_by_model") or {})
        want_moe = dict(snap_w.get("moe_by_model") or {})
        for mk in set(now_moe) | set(want_moe):
            if now_moe.get(mk) != want_moe.get(mk):
                ok, detail = _post(client, base + "/moe", {"model_key": mk, "value": want_moe.get(mk)})
                (rep["restored"].append(f"moe {mk}={want_moe.get(mk)}") if ok
                 else rep["failures"].append({"item": f"moe {mk}", "error": str(detail)}))
        now_bnb = dict(w.get("bnb_by_model") or {})
        want_bnb = dict(snap_w.get("bnb_by_model") or {})
        for mk in set(now_bnb) | set(want_bnb):
            if bool(now_bnb.get(mk)) != bool(want_bnb.get(mk)):
                ok, detail = _post(client, base + "/bnb", {"model_key": mk, "enabled": bool(want_bnb.get(mk))})
                (rep["restored"].append(f"bnb {mk}={'on' if want_bnb.get(mk) else 'off'}") if ok
                 else rep["failures"].append({"item": f"bnb {mk}", "error": str(detail)}))

        # (2c) per-worker limits + wildcard.
        if (dict(w.get("limits") or {})) != (dict(snap_w.get("limits") or {})):
            ok, detail = _post(client, base + "/limits", {"limits": dict(snap_w.get("limits") or {})})
            (rep["restored"].append("limits") if ok
             else rep["failures"].append({"item": "limits", "error": str(detail)}))
        if bool(w.get("wildcard")) != bool(snap_w.get("wildcard")):
            ok, detail = _post(client, base + "/wildcard", {"enabled": bool(snap_w.get("wildcard"))})
            (rep["restored"].append(f"wildcard={'on' if snap_w.get('wildcard') else 'off'}") if ok
             else rep["failures"].append({"item": "wildcard", "error": str(detail)}))

        # (3) disk occupancy (operator ruling 2026-09-24): every model on record
        # on this drive BEFORE the test must be back on it after. Re-provision any
        # missing prior model via the FETCH-ONLY verb (/fetch — download to disk,
        # never load into VRAM; restore only needs the files back, and any later
        # eviction is natural). Extras the test accumulated are LEFT in place; the
        # one caveat is a full-ish drive with no room for a prior model — then,
        # and only then, free space by removing ONLY extras (never a prior model,
        # never central/shared storage), fewest removals (largest extras first),
        # and only when the extras can actually cover the shortfall.
        prior = set(snap_w.get("on_disk") or [])
        now_local = set(w.get("models_local") or [])
        missing = [mk for mk in prior if mk not in now_local]
        storage_now = w.get("storage") if isinstance(w.get("storage"), dict) else {}
        now_rows = {r.get("model_key"): r for r in (storage_now.get("models") or [])
                    if isinstance(r, dict) and r.get("model_key")}
        snap_sizes = snap_w.get("model_sizes") or {}

        def _size(mk, _rows=now_rows, _snap=snap_sizes):
            for v in ((_rows.get(mk) or {}).get("bytes"), _snap.get(mk)):
                if isinstance(v, (int, float)) and v > 0:
                    return int(v)
            return None

        def _fetch(mk):
            # Fetch-only: download to the worker's DISK without loading it into
            # VRAM (operator ruling 2026-09-24 — restore only needs prior models
            # back on disk; any later disk eviction is natural, so restore never
            # pins or protects). Fire-and-forget: _post already turns an
            # {ok:false} / HTTP error (incl. an old agent's 501 "unsupported")
            # into a recorded failure, and we report the fetch STARTED, not its
            # completion.
            return _post(client, base + "/fetch", {"model_key": mk})

        # Removable extras: on the drive now, NOT a prior model, in the worker's
        # OWN reapable store, not protected/pinned (cache-evict refuses shared/
        # central copies). Largest first so the fewest removals cover a shortfall.
        removable = []
        for mk in now_local - prior:
            row = now_rows.get(mk) or {}
            if row.get("store") == "reapable" and not row.get("protected") and not row.get("pinned"):
                sz = _size(mk)
                if sz:
                    removable.append((mk, sz))
        removable.sort(key=lambda t: t[1], reverse=True)
        removed = set()
        free_est = float(storage_now.get("disk_free")) if isinstance(storage_now.get("disk_free"), (int, float)) else None

        # Biggest missing prior model first: handle the tightest fit while the
        # most extras are still available to free.
        for mk in sorted(missing, key=lambda m: (_size(m) or 0), reverse=True):
            need = _size(mk)
            if free_est is None or need is None:
                ok, detail = _fetch(mk)
                if ok:
                    rep["redownloaded"].append(mk)
                    rep["restored"].append(
                        f"re-provisioned {mk} (fit not checked: "
                        f"{'free space' if free_est is None else 'size'} unknown)")
                else:
                    rep["failures"].append({"item": f"re-provision {mk}", "error": str(detail)})
                continue
            if free_est < need:
                deficit = need - free_est
                avail = [(m, b) for (m, b) in removable if m not in removed]
                chosen, acc = [], 0
                for (m, b) in avail:   # already largest-first
                    chosen.append((m, b)); acc += b
                    if acc >= deficit:
                        break
                if acc < deficit:
                    # Extras cannot cover it — delete NOTHING for this model.
                    rep["failures"].append({"item": f"re-provision {mk}", "error": (
                        f"not enough room to restore prior model {mk}: need {_gb(need)} ({need} B), "
                        f"free {_gb(free_est)} ({int(free_est)} B), removable extras {_gb(acc)} "
                        f"({acc} B) across {len(avail)} extra model(s)")})
                    continue
                freed = 0
                for (m, b) in chosen:
                    _post(client, base + "/evict", {"model_key": m, "force": True})  # best-effort VRAM drop
                    ok, detail = _post(client, base + "/cache-evict", {"model_key": m})
                    if ok:
                        got = detail.get("freed_bytes") if isinstance(detail, dict) else None
                        got = int(got) if isinstance(got, (int, float)) and got > 0 else b
                        freed += got; removed.add(m); free_est += got
                        rep["removed"].append(m)
                        rep["restored"].append(f"removed {m} ({_gb(b)}) to fit prior model {mk}")
                    else:
                        rep["failures"].append({"item": f"cache-evict {m}", "error": str(detail)})
                if free_est < need:
                    rep["failures"].append({"item": f"re-provision {mk}", "error": (
                        f"freed {_gb(freed)} but still short: need {_gb(need)}, "
                        f"free {_gb(free_est)} after removals")})
                    continue
            ok, detail = _fetch(mk)
            if ok:
                rep["redownloaded"].append(mk)
                if isinstance(free_est, (int, float)) and need:
                    free_est -= need
            else:
                rep["failures"].append({"item": f"re-provision {mk}", "error": str(detail)})
        reports.append(rep)

    # per-model gguf_file_by_worker (the quant pin the benchmark's _select_quant
    # writes) — one POST per model whose map drifted, restoring the full original.
    rep_by_wid = {r["worker_id"]: r for r in reports}
    rep_by_name = {r["worker"]: r for r in reports}
    for mk, orig in (snap.get("serving") or {}).items():
        orig = dict(orig or {})
        cur = dict((now_serving or {}).get(mk) or {})
        if orig == cur:
            continue
        ok, detail = _post(client, "/llm/serving/" + quote(str(mk), safe=""),
                           {"gguf_file_by_worker": orig})
        # Attribute the change to each worker whose pin actually differed.
        changed = [wk for wk in set(orig) | set(cur) if orig.get(wk) != cur.get(wk)]
        for wk in changed or [None]:
            rep = rep_by_wid.get(wk) or rep_by_name.get(wk)
            if rep is None:
                continue
            (rep["restored"].append(f"quant pin for {mk}") if ok
             else rep["failures"].append({"item": f"gguf pin {mk}", "error": str(detail)}))
        if not ok and not changed:
            # No involved-worker attribution — surface it once at the top level.
            reports.append({"worker": "central", "worker_id": None, "restored": [], "redownloaded": [],
                            "removed": [], "failures": [{"item": f"gguf pin {mk}", "error": str(detail)}]})

    return {"at": time.time(), "workers": reports,
            "note": ("disk re-provision uses the fetch-only verb (/fetch — download to disk, no VRAM "
                     "load; nothing is pinned or protected); test-accumulated extras are kept unless a "
                     "full drive leaves no room for a prior model, in which case the fewest extras "
                     "(largest first) are removed to fit it — never a prior model or central storage; "
                     "a shortfall extras cannot cover deletes nothing and is reported")}
