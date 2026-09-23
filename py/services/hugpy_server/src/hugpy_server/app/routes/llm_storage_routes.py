import json
import threading
import time as _time

import logging
import os
import shutil

from abstract_flask import get_bp
from flask import abort, jsonify, request
from hugpy_engine.config.main import get_model_config
from hugpy_engine.config.models.models_config import get_models_dict
from hugpy_engine.manifest import upsert_model
from hugpy_engine.wire.config_schemas import settings
from hugpy_engine.wire.request_schemas import HFRepoDownloadRequest
from hugpy_fleet.central.peers import list_peers
from hugpy_storage.console.cancelable_downloads import update_model_sizes, update_model_status
from hugpy_storage.downloader.engine import invalidate_model_status_cache
from hugpy_storage.downloader.queue import (
    get_download,
    cancel_download,
    enqueue_download,
    list_downloads,
    retry_download,
)
from hugpy_storage.model_paths import route_destination
# Registry prune (hide a not-installed "ghost" model) + media-chat allow-flag.
# Explicit imports so they work regardless of the functions star-export.
from hugpy_engine.config.models.models_config import (
    prune_model,
    set_model_media,
    media_state,
    media_states,
    media_default_state,
    set_media_default,
    refresh_registry,
)
# The enqueue must be able to tell the operator whether the job LANDED — see
# _enqueued_response. Explicit, not via the star-export, so a refactor of
# functions/__init__ can never silently drop the honesty back out of the reply.
from hugpy_storage.downloader.queue import queue_depth, queue_healthy
from hugpy_storage.downloader.presence import downloader_alive
# Build identity (WP2): which build is central running. Module import, so a
# test can monkeypatch ``_buildinfo.build_identity`` and see the route follow.
from hugpy_platform import buildinfo as _buildinfo

llm_bp, logger = get_bp("llm_bp", __name__)

for name in ("httpx", "httpcore", "huggingface_hub", "filelock", "urllib3"):
    logging.getLogger(name).setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────
def _guarded_build(fn):
    """``fn()`` or ``{"error": ...}`` — the health and build routes must never
    5xx because a metadata read or a repository probe went wrong."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


@llm_bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "storage_root": str(settings.storage_root),
        "manifest_path": str(settings.manifest_path),
        "build": _guarded_build(_buildinfo.build_identity),
    })


@llm_bp.route("/build", methods=["GET"])
def build():
    """The full build document: identity, every workspace/external
    distribution, the lockstep check, the editable checkout's state."""
    doc = _guarded_build(_buildinfo.build_info)
    if "identity" not in doc:
        doc = {"identity": None, "build": doc, "error": doc.get("error")}
    return jsonify(doc)


@llm_bp.route("/llm/peers", methods=["GET"])
def peers():
    return jsonify(list_peers())


# The two size annotators MOVED to functions/downloads/model_physical.py, which
# owns deriving a model's physical state. They were route privates, and a route
# private cannot be a WRITE point — the download hook and the /models/discover
# repair sweep have to derive the same numbers the listing shows, through the
# same code, or the persisted record and the response drift. Re-exported here
# under their old names so nothing that reached for them has to move.
from hugpy_storage.console.model_physical import (
    annotate_gguf_size as _annotate_gguf_size,
    annotate_size as _annotate_size,
    refresh_fields,
    rebuild_physical,
)


@llm_bp.route("/models", methods=["GET"])
def list_models():
    manifest = get_models_dict(dict_return=True)
    media_default = media_default_state()
    # Operator BLOCK set (guarded — a listing must never 500 over the blocklist).
    try:
        from hugpy_fleet.central.blocklist import blocked_keys, block_info
        _blocked = blocked_keys()
    except Exception:  # noqa: BLE001
        _blocked, block_info = set(), (lambda _k: None)
    # Media-chat flags for the WHOLE manifest in one store read — media_state()
    # per model re-read media_models.json ~107 times (isfile + open + read each,
    # every one a virtiofs round-trip). Same rule as the physical state below:
    # nothing on this loop may touch the filesystem per model.
    _media = media_states(manifest.keys() | {
        (m.get("model_key") or k) for k, m in manifest.items()})
    output = []
    for key, model in manifest.items():
        model = update_model_status(model)
        mk = model.get("model_key") or key
        # Operator BLOCK state (additive): ⛔ blocked from the serving pool. The
        # console renders the chip + block/unblock control off this flag; the
        # full record (by/ts/note) rides `block` for the tooltip.
        model["blocked"] = (mk in _blocked) or (key in _blocked)
        if model["blocked"]:
            model["block"] = block_info(mk) or block_info(key)
        else:
            # get_models_dict returns the cached manifest dicts (mutated in place
            # like model["media"] below), so a stale `block` record from a prior
            # blocked read must be cleared on unblock — the `blocked` bool alone
            # is not enough.
            model.pop("block", None)
        # Whether this model is offered in the media-intelligence chat dropdown.
        model["media"] = _media[mk]
        # Whether this model is THE preselected default for the media chat.
        # Exactly one model carries media_default=True (or none, if unset).
        model["media_default"] = (mk == media_default)
        # The size half of the model's PERSISTED physical state: the GGUF
        # effective quant (the one that serves, never the all-quants dir sum)
        # plus a size for EVERY model in ANY disposition, so the picker shows
        # what you're committing before you commit it. Same numbers the two
        # annotators produced — derived once, at the events that change them,
        # not re-walked out of the store on every GET.
        update_model_sizes(model, mk)
        output.append(model)

    # HOT / COLD residency (operator ask 2026-09-12): a model is HOT where it
    # sits on a worker's hot store (its `models_local`), COLD when it's a known
    # model that isn't staged hot anywhere. The catalog row is worker-agnostic,
    # so stamp it here off ONE registry read (cached heartbeat facts, no worker
    # I/O) rather than making every caller re-derive it from /llm/workers.
    #   hot          -> bool (staged on >=1 worker)
    #   hot_workers  -> [worker name, ...] holding it hot
    # Matching tolerates the model_key / name / hub_id / org~repo spellings a
    # worker may report models_local under. Never 5xxes the listing.
    try:
        from hugpy_fleet.central.workers import list_workers
        _hot = {}
        for w in list_workers():
            wname = w.get("name") or w.get("id")
            for mk in (w.get("models_local") or []):
                _hot.setdefault(mk, []).append(wname)
        for m in output:
            hid = m.get("hub_id") or ""
            cands = {m.get("model_key"), m.get("name"), hid}
            if "/" in hid:
                cands.add(hid.replace("/", "~"))
            hw = sorted({wn for c in cands if c for wn in _hot.get(c, [])})
            m["hot_workers"] = hw
            m["hot"] = bool(hw)
    except Exception:  # noqa: BLE001 — residency stamp must never 500 the listing
        logger.exception("hot/cold residency stamp failed")

    # VERBOSE view (operator ask 2026-07-29): join each model with the per-worker
    # serving facts the Workers panel renders — Memory, Alloc, 4-bit, MoE, Seat,
    # Residency, 📌 — so ONE call relays everything known about a model in the
    # pool. `?verbose=1` for the whole roster; same join per key on
    # GET /models/<key>. Read-only: relays the registry rows verbatim, derives
    # nothing new (measured stays measured, planned stays planned).
    if request.args.get("verbose") in ("1", "true", "yes"):
        try:
            _joins = _verbose_worker_join({m.get("model_key") for m in output})
            for m in output:
                m["workers"] = _joins.get(m.get("model_key"), [])
        except Exception:  # noqa: BLE001 — the join must never 500 the listing
            logger.exception("verbose worker join failed")

    return jsonify(output)


def _verbose_worker_join(model_keys: set) -> dict:
    """model_key -> [per-worker serving row]. One registry read for the whole
    roster; every field is the worker's own report, relayed under the name the
    console column uses."""
    from hugpy_fleet.central.workers import list_workers
    out: dict = {mk: [] for mk in model_keys if mk}
    for w in list_workers():
        wname, wid = w.get("name"), w.get("id")
        designated = set(w.get("models") or [])
        spill_by = w.get("spill_by_model") or {}
        alloc_modes = w.get("model_alloc_modes") or {}
        bnb_by = w.get("bnb_by_model") or {}
        moe_by = w.get("moe_by_model") or {}
        loaded = set(w.get("loaded_models") or [])
        allocs = {a.get("model_key"): a for a in (w.get("allocations") or [])
                  if isinstance(a, dict)}
        seats = {s.get("model_key"): s for s in (w.get("slots") or [])
                 if isinstance(s, dict) and s.get("model_key")}
        storage_rows = {r.get("model_key"): r
                        for r in ((w.get("storage") or {}).get("models") or [])
                        if isinstance(r, dict)}
        star = w.get("boot_prewarm")
        for mk in out:
            touched = (mk in designated or mk in loaded or mk in allocs
                       or mk in seats or mk in storage_rows)
            if not touched:
                continue
            alloc = allocs.get(mk) or {}
            seat = seats.get(mk) or {}
            st = storage_rows.get(mk) or {}
            out[mk].append({
                "worker": wname, "worker_id": wid, "status": w.get("status"),
                "designated": mk in designated,
                # Alloc — the designation's spill (mode, budgets, ngl, bands)
                "alloc": spill_by.get(mk) or {},
                "alloc_mode": alloc_modes.get(mk),
                # 4-bit / MoE levers as the worker reports them
                "bnb_4bit": bnb_by.get(mk),
                "moe": moe_by.get(mk),
                # Memory — measured residency (vram/weight bytes, gpu_pct, lane)
                "allocation": alloc or None,
                "loaded": mk in loaded,
                "serving": alloc.get("serving"),
                # Seat — the slot child holding this model, if any
                "seat": ({"slot_id": seat.get("slot_id"),
                          "healthy": seat.get("healthy"),
                          "busy": seat.get("busy"),
                          "n_gpu_layers": seat.get("n_gpu_layers"),
                          "n_cpu_moe": seat.get("n_cpu_moe"),
                          "ctx": seat.get("ctx"),
                          "last_load_error": seat.get("last_load_error")}
                         if seat else None),
                # Residency / 📌 — the storage report's disposition
                "pinned": st.get("pinned"),
                "protected": st.get("protected"),
                "assigned": st.get("assigned"),
                "on_disk_bytes": st.get("bytes"),
                "provisioning": st.get("provisioning"),
                "keep_warm_star": (star == mk) or None,
            })
    return out


# ── Disk discovery (the console's "Discover models" button) ───────────────
# The discovery report (MODELS_DISCOVERY_PATH) is the persisted half of the
# registry; a walk taken while the storage mount was degraded can shrink it,
# so models "disappear" from /models while their files still sit on disk
# (observed 2026-07-04: report at 2 entries vs 108 model dirs). Nothing
# re-walks at runtime — download completion only re-READS the report — so
# this route is the recovery path. The walk enriches from hub metadata and
# can take minutes, hence the background thread + poll shape: POST to start,
# GET for state, re-fetch /models when running goes false.
#
# State lives in a FILE next to the discovery report, not a module global:
# the API runs gunicorn --workers 3, so per-process state answers the poll
# wrong 2/3 of the time (the comms-mirror lesson). A stale "running" left by
# a killed worker expires via _DISCOVER_STALE_S.
_discover_lock = threading.Lock()
_DISCOVER_STALE_S = 30 * 60


def _discover_state_path() -> str:
    from hugpy_platform.constants import MODELS_DISCOVERY_PATH
    return str(MODELS_DISCOVERY_PATH) + ".state.json"


def _read_discover_state() -> dict:
    state = {"running": False, "started_at": None, "finished_at": None,
             "found": None, "error": None}
    try:
        with open(_discover_state_path(), "r", encoding="utf-8") as fh:
            state.update(json.load(fh))
    except (OSError, ValueError):
        pass
    # Self-heal: a worker that died mid-sweep leaves running=true forever.
    if state.get("running") and state.get("started_at") and \
            _time.time() - state["started_at"] > _DISCOVER_STALE_S:
        state.update(running=False,
                     error="sweep did not finish (worker restarted?)")
    return state


def _write_discover_state(state: dict) -> None:
    path = _discover_state_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
    os.replace(tmp, path)


def _run_discovery(state: dict):
    try:
        refresh_registry(run_discovery=True)   # walk + save report + in-place update
        manifest = get_models_dict(dict_return=True)
        state["found"] = len(manifest)
        # THE REPAIR PASS for the persisted physical state. The store is SHARED
        # and MUTABLE — another box writes weights, an operator `mv`s a
        # directory, the reaper deletes — and none of that fires one of our
        # events, so a persisted size or status can go stale with nobody to tell
        # us. This sweep re-derives and rewrites every row. It belongs here
        # because /models/discover is ALREADY the "re-read the disk, the catalog
        # drifted" recovery route, it already runs on a background thread where
        # a full walk is affordable, and refresh_registry above has just dropped
        # the table — so the console's next /models is correct AND warm instead
        # of paying the whole walk inside one request.
        state["physical"] = rebuild_physical(manifest, source="discover")
    except Exception as exc:  # noqa: BLE001 — state must always resolve
        logger.warning("model discovery sweep failed: %s", exc)
        state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        state.update(running=False, finished_at=_time.time())
        try:
            _write_discover_state(state)
        except OSError as exc:
            logger.warning("could not persist discovery state: %s", exc)


@llm_bp.route("/models/discover", methods=["POST"])
def discover_models_start():
    with _discover_lock:
        state = _read_discover_state()
        if state["running"]:
            return jsonify({**state, "started": False}), 409
        state.update(running=True, started_at=_time.time(),
                     finished_at=None, found=None, error=None)
        _write_discover_state(state)
    threading.Thread(target=_run_discovery, args=(state,),
                     name="models-discover", daemon=True).start()
    return jsonify({**state, "started": True}), 202


@llm_bp.route("/models/discover", methods=["GET"])
def discover_models_state():
    return jsonify(_read_discover_state())


# ── Store reconcile (the flattening migration) ────────────────────────────
# Flattens the store to models/<runtime>/<owner>/<repo>: moves each repo's
# COMPLETE copy into the flat path, merges complements (an mmproj twin),
# ARCHIVES losers + .part orphans (never deletes), updates the registry +
# markers. MONITOR-FIRST: body {"apply": false} (the default) returns the full
# plan and touches NOTHING; {"apply": true} executes. Operator-token gated like
# the other mutating store ops. The full JSON report is also written next to the
# discovery report (reconcile_report[.dry].json) for the keeper to review.
@llm_bp.route("/models/reconcile", methods=["POST"])
def reconcile_store_route():
    body = request.get_json(silent=True) or {}
    apply = bool(body.get("apply", False))
    from hugpy_engine.apis.reconcile import reconcile_store
    from hugpy_platform.constants import MODELS_DISCOVERY_PATH
    suffix = "reconcile_report.json" if apply else "reconcile_report.dry.json"
    report_path = os.path.join(os.path.dirname(str(MODELS_DISCOVERY_PATH)), suffix)
    report = reconcile_store(apply=apply, report_path=report_path)
    report["report_path"] = report_path
    if apply:
        # An applied reconcile MOVES weights between layouts, so every persisted
        # destination and every persisted size is suspect. No model_key here on
        # purpose: the blast radius is the whole store, so the whole table goes
        # (over-dropping costs re-derivation; under-dropping would keep pointing
        # at the old dirs). reconcile's own _persist_registry only calls
        # refresh_registry when it had registry rows to write — a move-only plan
        # would otherwise leave the records pointing at the old dirs.
        invalidate_model_status_cache("store reconciled")
    return jsonify(report), (200 if apply else 202)


# ── Image-task re-classification (the k61 one-shot) ───────────────────────
# Re-derives each model's task from its OWN directory (a diffusers
# model_index.json pipeline class, or an adapter-only dir) and re-stamps the
# sidecar + the discovery row where they disagree. Same MONITOR-FIRST posture as
# reconcile: {"apply": false} (the default) reports what WOULD change and touches
# nothing. This exists so the fleet's wrong stamps — flux2 marked
# image-to-image-only, an image LoRA left null and read as an LLM — are corrected
# by code on every box instead of by hand on one.
@llm_bp.route("/models/reclassify-images", methods=["POST"])
def reclassify_images_route():
    body = request.get_json(silent=True) or {}
    apply = bool(body.get("apply", False))
    from hugpy_engine.apis.reclassify import reclassify_images
    report = reclassify_images(apply=apply)
    if apply and report["changed"]:
        # Tasks changed => every derived registry row and every cached status
        # that keyed on the old task is stale.
        refresh_registry(run_discovery=False)
        invalidate_model_status_cache("image tasks reclassified")
    return jsonify(report), (200 if apply else 202)


@llm_bp.route("/models/<model_key>", methods=["GET"])
def get_model(model_key):
    manifest = get_models_dict(dict_return=True)
    logger.info(manifest)
    if model_key not in manifest:
        abort(404, description="Unknown model key.")
    model = manifest[model_key]
    # The single-model detail read is the EXPLICIT refresh path: it always
    # derives LIVE (one model is ~10^2 filesystem calls, not ~10^4) and REWRITES
    # the persisted record the listings read, so "open the row" is how an
    # operator forces a re-read of a shared, mutable store — and it repairs the
    # listing for everyone else at the same time.
    detail = {"key": model_key, **model, **refresh_fields(model, model_key)}
    # The single-model read is ALWAYS verbose (operator ask 2026-07-29): the
    # per-worker serving facts (alloc/4-bit/MoE/seat/residency/📌) ride every
    # detail fetch — one call, everything known about the model in the pool.
    try:
        mk = model.get("model_key") or model_key
        detail["workers"] = _verbose_worker_join({mk}).get(mk, [])
    except Exception:  # noqa: BLE001 — the join must never 500 the detail read
        logger.exception("verbose worker join failed for %s", model_key)
    return jsonify(detail)


# ──────────────────────────────────────────────────────────────────────────
# Downloads: ENQUEUE / READ / CANCEL only.
#
# The API does not download. It creates a queued job of kind "download" and the
# hugpy-downloader-dev daemon claims and runs it (abstract_hugpy_dev/downloader/).
# No transfer child parented to a gunicorn worker, no monitor/watch threads, no
# per-second store walk and no HF network call on a request path — those are what
# starved the pool that also serves /llm/workers/<id>/heartbeat and made every
# worker read `offline` during a download.
#
# There is deliberately NO in-process fallback when the daemon is down: falling
# back would quietly resurrect the exact bug. The job stays visibly queued and
# says so (queue.annotate_waiting).
# ──────────────────────────────────────────────────────────────────────────
def _enqueued_response(job, **extra):
    """The enqueue reply, with the two facts a caller needs to know it WORKED:
    the job id (already there) and whether the job is actually ON the daemon's
    queue. Before k119 a POST returned a cheerful `queued` row even when the
    shared mirror was quarantined and no daemon could ever see it — the console
    showed "queued" forever and the request was silently expired 30 minutes
    later. ``queued`` is False + ``queue_error`` is set when that happens, so the
    UI can shout instead of spinning."""
    d = job.to_legacy_dict()
    depth = queue_depth()
    healthy = depth >= 0 and queue_healthy()
    d.update(extra)
    d["queued"] = bool(healthy)
    # Position is 1-based and includes this job (it was just enqueued), so "1"
    # means "next up" — what an operator expects a queue position to mean.
    d["queue_position"] = depth if depth >= 0 else None
    d["downloader_running"] = bool(downloader_alive())
    if not healthy:
        d["queue_error"] = (
            "The shared download queue is unreachable — this job was recorded "
            "but the downloader cannot claim it yet.")
    elif not d["downloader_running"]:
        d["queue_error"] = (
            "Queued, but the downloader service is not running "
            "(hugpy-downloader-dev). Start it and this job will be claimed.")
    return jsonify(d)


@llm_bp.route("/models/<model_key>/download", methods=["POST"])
def start_download(model_key):
    model = get_model_config(model_key,dict_return=True)
    if not model:
        abort(404, description="Unknown model key.")
    logger.info(model)
    body = request.get_json(silent=True) or {}
    job = enqueue_download(model_key, model,
                           total_bytes=body.get("total_bytes"))
    logger.info("enqueued download job %s for %s", job.id, model_key)
    return _enqueued_response(job)


@llm_bp.route("/jobs", methods=["GET"])
def list_jobs():
    # MIRROR-MERGED: live download rows are owned by the daemon process and exist
    # only in the shared mirror, and terminal ones must stay visible too (a job
    # that vanished at 100% instead of reading "completed" would be worse than
    # before). Legacy wire shape preserved: queued/running/completed, error as a
    # string — the console's ModelTable reads exactly this.
    #
    # k121 honesty pass BEFORE the read (operator ruling 2026-08-20): a stalled
    # queue must never render as a cheerful "queued".
    #   * expire_pending_orphans — never-claimed rows go terminal `expired`
    #     with the reason in `message` (parity with /llm/jobs, which already
    #     did this; this view didn't).
    #   * reap_dead_claims — a row claimed by a downloader that died and never
    #     restarted was IMMORTAL (claim skips it, expiry skips it); it now
    #     re-queues with an honest message once the heartbeat is long-dead.
    # Failure records (failed/expired) persist here until POST /jobs/<id>/discard.
    from hugpy_control.jobs import job_store
    from hugpy_storage.downloader.queue import merge_notes, reap_dead_claims
    try:
        job_store.expire_pending_orphans()
    except Exception:
        pass
    reap_dead_claims()
    return jsonify(merge_notes(list_downloads()))


@llm_bp.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    d = get_download(job_id)
    if d is None:
        abort(404, description="Unknown job ID.")
    return jsonify(d)


@llm_bp.route("/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    """Cancel CROSS-PROCESS: raises the shared cancel flag the daemon's store
    watcher is already listening on, and force-marks an owner-less row terminal
    so a cancel can never answer true while nothing changes."""
    res = cancel_download(job_id)
    if res.get("reason") == "unknown job":
        abort(404, description="Unknown job ID.")
    return jsonify(res)


@llm_bp.route("/jobs/<job_id>/retry", methods=["POST"])
def retry_job(job_id):
    """Re-queue a failed/cancelled download; the daemon picks it back up and
    resumes from the partial files on disk (same job id, same payload)."""
    res = retry_download(job_id)
    if res.get("reason") == "unknown job":
        abort(404, description="Unknown job ID.")
    return jsonify(res)


@llm_bp.route("/jobs/<job_id>/discard", methods=["POST"])
def discard_job(job_id):
    """ADMIN DISCARD (k121): the only way a persistent failure record
    (failed/expired) leaves the queue. Gated in operator_auth's mutation
    allowlist alongside cancel/retry."""
    from hugpy_storage.downloader.queue import discard_download
    res = discard_download(job_id)
    if res.get("reason") == "unknown job":
        abort(404, description="Unknown job ID.")
    return jsonify(res)


@llm_bp.route("/jobs/<job_id>/diagnose", methods=["POST"])
def diagnose_job(job_id):
    """THE INFERENCE TRIGGER for a failed/stalled download (k121): asks the
    hugpy VM's keeper (the same B seat as the help widget's direct line) to
    diagnose this specific record, grounded in the row itself + live queue/
    fleet facts. The diagnosis is PINNED to the record (job_notes) and rides
    every subsequent /jobs read until the record is discarded."""
    from hugpy_storage.downloader.queue import get_download, set_diagnosis
    from hugpy_server.app.keeper_line import fleet_grounding, keeper_ask
    d = get_download(job_id)
    if d is None:
        abort(404, description="Unknown job ID.")
    prompt = (
        "You are the hugpy fleet keeper. Diagnose this model-download problem "
        "for the operator: name the root cause and the ONE concrete action "
        "that fixes it. Be specific and short (a few sentences).\n\n"
        f"JOB RECORD:\n{json.dumps(d, default=str)[:4000]}\n\n"
        f"FLEET FACTS:\n{fleet_grounding(include_queue=False)}")
    res = keeper_ask(prompt)
    reply = res.get("reply") or ""
    stored = bool(reply) and set_diagnosis(job_id, reply)
    return jsonify({"ok": bool(reply), "diagnosis": reply,
                    "offline": bool(res.get("offline")), "stored": stored})


@llm_bp.route("/llm/repos/download", methods=["POST"])
def download_repo():
    """Acquire any Hugging Face repo by hub_id without a pre-registered manifest entry.

    If register=True, the model is added to the manifest so it appears in the
    registry browser on the next refresh.
    """
    body = HFRepoDownloadRequest(**(request.get_json(silent=True) or {}))
    model = {
        "name": body.name or body.hub_id.split("/")[-1],
        "hub_id": body.hub_id,
        "framework": body.framework,
        "task": body.task,
        "filename": body.filename,
        "include": body.include,
    }

    if body.register:
        model_key, _ = upsert_model(settings.manifest_path, model)
    else:
        from hugpy_engine.manifest import key_for_hub_id
        model_key = key_for_hub_id(body.hub_id)

    job = enqueue_download(model_key, model, total_bytes=body.total_bytes)
    logger.info("enqueued download job %s for %s (hub_id=%s)",
                job.id, model_key, body.hub_id)
    return _enqueued_response(job, model_key=model_key)


@llm_bp.route("/models/<model_key>", methods=["DELETE"])
def delete_model(model_key):
    manifest = get_models_dict(dict_return=True)
    if model_key not in manifest:
        abort(404, description="Unknown model key.")

    destination = route_destination(manifest.get(model_key))
    if not os.path.exists(destination):
        return jsonify({
            "deleted": False,
            "message": "Model is not installed.",
            "destination": str(destination),
        })

    shutil.rmtree(destination)
    # DELETE does not go through refresh_registry (the catalog row survives, only
    # the files go), so it must say so itself — otherwise the listings would keep
    # reporting "installed" for a model whose weights are gone.
    invalidate_model_status_cache(f"model deleted: {model_key}",
                                  model_key=model_key)
    return jsonify({"deleted": True, "destination": str(destination)})


@llm_bp.route("/models/<model_key>/prune", methods=["POST"])
def prune_model_route(model_key):
    """Remove a NOT-installed model's registry entry (a "ghost" row).

    Distinct from DELETE, which only removes downloaded files. Prune hides the
    catalog row itself (persisted in pruned_models.json) so it stops cluttering
    the listing. Refuses to prune a model that still has files on disk — Delete
    those first, so prune never silently orphans real data."""
    manifest = get_models_dict(dict_return=True)
    if model_key not in manifest:
        abort(404, description="Unknown model key.")

    destination = route_destination(manifest.get(model_key))
    if destination and os.path.exists(destination):
        return jsonify({
            "pruned": False,
            "message": "Model has files on disk — delete them before pruning.",
            "destination": str(destination),
        }), 409

    result = prune_model(model_key)
    # Prune only hides a not-installed row (it refuses when files exist), so the
    # STATUS is unchanged — but it is a mutating store op and the memo is keyed
    # by routing identity, so drop it rather than reason about whether a pruned
    # key can come back. One re-walk is the entire cost.
    invalidate_model_status_cache(f"model pruned: {model_key}",
                                  model_key=model_key)
    return jsonify(result)


@llm_bp.route("/models/<model_key>/media", methods=["POST"])
def set_model_media_route(model_key):
    """Toggle whether a model is offered in the media-intelligence chat dropdown.

    Body: {"enabled": bool}. Curated default models start enabled; the store only
    keeps deviations from that default (see set_model_media)."""
    manifest = get_models_dict(dict_return=True)
    if model_key not in manifest:
        abort(404, description="Unknown model key.")
    body = request.get_json(silent=True) or {}
    enabled = body.get("enabled", body.get("media", True))
    return jsonify(set_model_media(model_key, enabled))


@llm_bp.route("/models/<model_key>/media-default", methods=["POST"])
def set_model_media_default_route(model_key):
    """Set (or clear) the single default media-chat model — the one the media
    chat dropdown preselects.

    Body: {"default": bool} (defaults to True). default=True makes this model THE
    default, replacing any previous one; default=False clears it only if this
    model is the current default. Single global value, persisted server-side
    (media_default.json) so every client agrees.

    Setting a model as default does NOT require it to be media-enabled."""
    manifest = get_models_dict(dict_return=True)
    if model_key not in manifest:
        abort(404, description="Unknown model key.")
    body = request.get_json(silent=True) or {}
    is_default = body.get("default", body.get("enabled", True))
    return jsonify(set_media_default(model_key, is_default))
