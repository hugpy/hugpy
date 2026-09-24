"""Control-plane routes over the comms foundations (F2 principals, F4
settings, F5 unified jobs). Operator-gated writes are listed in
operator_auth._SENSITIVE — the gate lives there, not here.

Design: these are THIN adapters. All behavior lives in comms.principals /
comms.settings / comms.jobs; a route parses HTTP and calls one method. If a
route grows logic, it's in the wrong file.
"""
from __future__ import annotations

import json
import logging
import os
import time

from flask import Blueprint, jsonify, request, abort

logger = logging.getLogger(__name__)

comms_bp = Blueprint("comms_bp", __name__)


# ---------------------------------------------------------------------------
# Audit log — UTIL-02's requirement, useful for every operator action.
# JSONL next to the other state files; also published on the bus.
# ---------------------------------------------------------------------------
def audit(action: str, detail: dict) -> None:
    entry = {"ts": time.time(), "action": action, "detail": detail,
             "remote": request.remote_addr if request else None}
    try:
        from hugpy_control.bus import bus
        bus.publish("audit", payload=entry)
    except Exception:
        pass
    try:
        base = (os.environ.get("PROJECTS_HOME") or "").strip()
        if not base:
            from hugpy_platform.constants import PROJECTS_HOME as _PH
            base = str(_PH)
        with open(os.path.join(base, "audit.log"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        logger.warning("audit write failed for %s", action, exc_info=True)


# ---------------------------------------------------------------------------
# F5 — unified jobs view (CON-01 backend): every transport, canonical states,
# merged across gunicorn workers via the mirror. /llm/queue keeps its legacy
# shape; THIS is the one new surfaces should read.
# ---------------------------------------------------------------------------
@comms_bp.route("/llm/snapshot", methods=["GET"])
def llm_snapshot():
    """FEEDS-DB-20260910: every console feed in one read, from the materialized rows."""
    from flask import current_app
    from hugpy_fleet.central import feeds
    feeds.ensure_refresher(current_app._get_current_object())
    want = [f for f in (request.args.get("feeds") or "").split(",") if f] or None
    data = feeds.read_all(want)
    return jsonify({
        "ts": time.time(), "leader": feeds.is_leader(), "db": feeds.db_ok(),
        "versions": {f: e["version"] for f, e in data.items()},
        "updated": {f: e["updated"] for f, e in data.items()},
        "feeds": {f: e["payload"] for f, e in data.items()},
    })


@comms_bp.route("/llm/events", methods=["GET"])
def llm_events():
    """FEEDS-DB-20260910: SSE — full snapshot on connect, then only changed feeds."""
    from flask import current_app, Response
    from hugpy_fleet.central import feeds
    feeds.ensure_refresher(current_app._get_current_object())
    want = [f for f in (request.args.get("feeds") or "").split(",") if f] or None

    def gen():
        data = feeds.read_all(want)
        seen = {f: e["version"] for f, e in data.items()}
        yield "event: snapshot\ndata: " + json.dumps({
            "ts": time.time(), "versions": seen,
            "feeds": {f: e["payload"] for f, e in data.items()}}, default=str) + "\n\n"
        started = time.time()
        last_beat = started
        while time.time() - started < 3600:
            time.sleep(1.0)
            try:
                versions = feeds.read_versions()
            except Exception:  # noqa: BLE001
                continue
            changed = [f for f, v in versions.items() if (want is None or f in want) and seen.get(f) != v]
            if changed:
                fresh = feeds.read_all(changed)
                for f, e in fresh.items():
                    seen[f] = e["version"]
                    yield "event: feed\ndata: " + json.dumps(
                        {"feed": f, "version": e["version"], "updated": e["updated"], "payload": e["payload"]},
                        default=str) + "\n\n"
                last_beat = time.time()
            elif time.time() - last_beat > 15:
                yield ": keepalive\n\n"
                last_beat = time.time()

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


@comms_bp.route("/llm/calls", methods=["GET"])
def llm_calls():
    """CALL-LOG-20260910: who called what, from where, when — merged from the call log."""
    from hugpy_control import calllog
    try:
        limit = max(1, min(2000, int(request.args.get("limit") or 300)))
    except ValueError:
        limit = 300
    try:
        since = float(request.args.get("since")) if request.args.get("since") else None
    except ValueError:
        since = None
    rows = calllog.read(limit=limit, since=since)
    return jsonify({"calls": rows, "count": len(rows), "path": calllog.path()})


@comms_bp.route("/llm/jobs", methods=["GET"])
def llm_jobs():
    from hugpy_control.jobs import job_store
    kind = (request.args.get("kind") or "").strip() or None
    transport = (request.args.get("transport") or "").strip() or None
    live = (request.args.get("live") or "1").strip() not in ("0", "false")
    # Pending-orphan expiry is EVENT-DRIVEN (slice 9, defect 2): this view is a
    # natural trigger — no timer thread. Retire never-dispatched pending jobs
    # BEFORE the snapshot so the operator never sees an immortal pending row.
    # Aged on progressed_at inside the store, so reading this view can never
    # RESET the clock (the resurrection bug stays fixed).
    try:
        job_store.expire_pending_orphans()
    except Exception:
        pass
    rows = job_store.snapshot(kinds={kind} if kind else None,
                              live_only=live)
    if transport:
        rows = [r for r in rows if r.get("transport") == transport]
    return jsonify({"jobs": rows, "counts": job_store.counts()})


# ---------------------------------------------------------------------------
# Cancel fan-out (B1): one control-plane cancel across BOTH planes on the shared
# job id. comms.JobStore.cancel handles chat/download (and raises the mirror
# flag); for a media job (transport == "media") we ALSO reach the media_bus so
# the execution plane actually stops (queued -> terminal 'cancelled', running ->
# cooperative 'cancelling'). The media_bus import lives HERE (the flask route),
# never in comms/, so comms stays free of any video_intel coupling.
# ---------------------------------------------------------------------------
@comms_bp.route("/llm/jobs/<job_id>/cancel", methods=["POST"])
def llm_job_cancel(job_id):
    from hugpy_control.jobs import job_store
    body = request.get_json(silent=True) or {}
    reason = str(body.get("reason") or "")
    # AUTHORITATIVE, HONEST cancel (slice 9, defect 1): a live owner is relayed;
    # an owner-less job (pending/worker-null, or a mirror-only immortal row) is
    # force-marked terminal in the store — persisted, surviving restarts. The
    # `mode` says which happened, and a cancel that changes NOTHING returns
    # cancelled:false (no more lying cancelled:true on a stuck job).
    res = job_store.cancel_authoritative(job_id, reason)
    ok = bool(res.get("cancelled"))
    mode = res.get("mode")
    job = job_store.get(job_id)
    transport = getattr(job, "transport", None) if job is not None else None
    comms_status = res.get("status")
    if comms_status is None and job is not None:
        comms_status = job.to_dict()["status"]
    # Reach the execution plane UNCONDITIONALLY. media_bus.cancel is a safe no-op
    # for a non-media id (unknown -> {cancelled:False, status:None}, a pure read),
    # so calling it always also covers a QUEUED media job — which has no local
    # JobStore record (on_enqueue is mirror-only) and therefore no readable
    # transport from job_store.get; the gated version silently skipped it. The
    # media_bus import lives HERE (flask route), never in comms/, so comms stays
    # free of any video_intel coupling.
    from hugpy_video.intel import media_bus
    mb = media_bus.cancel(job_id)
    mb_known = mb.get("status") is not None or bool(mb.get("cancelled"))
    if mb_known and transport is None:
        transport = "media"  # queued media, inferred from the execution plane
    if mb.get("cancelled") and not ok:
        # The comms plane found nothing but the media plane did — report honestly.
        ok = True
        mode = mode if mode not in (None, "noop") else "media"
    cancelled = ok
    status = mb.get("status") if mb_known else comms_status
    audit("job.cancel", {"job_id": job_id, "transport": transport,
                         "cancelled": cancelled, "mode": mode, "reason": reason})
    return jsonify({"job_id": job_id, "cancelled": cancelled, "mode": mode,
                    "status": status, "transport": transport})


# ---------------------------------------------------------------------------
# F2 — principals (operator-gated writes; see _SENSITIVE)
# ---------------------------------------------------------------------------
@comms_bp.route("/auth/principals", methods=["GET"])
def principals_list():
    from hugpy_control.principals import principal_store
    return jsonify({"principals": [p.to_dict()
                                   for p in principal_store.all()]})


@comms_bp.route("/auth/principals", methods=["POST"])
def principals_create():
    from hugpy_control.principals import principal_store
    body = request.get_json(silent=True) or {}
    try:
        p = principal_store.create(
            kind=str(body.get("kind") or "user"),
            name=str(body.get("name") or ""),
            groups=list(body.get("groups") or []),
            expires_in=body.get("expires_in"))
    except ValueError as exc:
        abort(400, description=str(exc))
    token = None
    if body.get("issue_token", True):
        token = principal_store.issue_token(
            p.id, expires_in=body.get("token_expires_in"))
    audit("principal.create", {"id": p.id, "kind": p.kind, "name": p.name,
                               "groups": p.groups})
    # token plaintext is returned ONCE, here.
    return jsonify({**p.to_dict(), "token": token})


@comms_bp.route("/auth/principals/<principal_id>", methods=["DELETE"])
def principals_revoke(principal_id):
    from hugpy_control.principals import principal_store
    ok = principal_store.revoke(principal_id)
    if not ok:
        abort(404, description="Unknown principal id.")
    audit("principal.revoke", {"id": principal_id})
    return jsonify({"revoked": True, "id": principal_id})


@comms_bp.route("/auth/principals/<principal_id>/token", methods=["POST"])
def principals_issue_token(principal_id):
    from hugpy_control.principals import principal_store
    body = request.get_json(silent=True) or {}
    token = principal_store.issue_token(principal_id,
                                        expires_in=body.get("expires_in"))
    if token is None:
        abort(404, description="Unknown principal id.")
    audit("principal.token", {"id": principal_id})
    return jsonify({"id": principal_id, "token": token})


@comms_bp.route("/auth/discord-link", methods=["POST"])
def discord_link():
    """DISC-05 handshake: the bot relays a user's /link <token> here. The
    token proves possession of a principal; the snowflake binds to it.
    Deliberately NOT operator-gated — the token IS the credential."""
    from hugpy_control.principals import principal_store
    body = request.get_json(silent=True) or {}
    token = str(body.get("token") or "")
    discord_user_id = str(body.get("discord_user_id") or "")
    if not token or not discord_user_id:
        abort(400, description="token and discord_user_id required")
    p = principal_store.link_discord(token, discord_user_id)
    if p is None:
        # Same response either way would be kinder to attackers; be explicit
        # for the legitimate user instead — tokens are 160-bit random.
        abort(401, description="invalid or expired principal token")
    audit("principal.discord-link", {"id": p.id,
                                     "discord_user_id": discord_user_id})
    return jsonify({"linked": True, "principal": p.to_dict()})


@comms_bp.route("/auth/whoami", methods=["GET"])
def whoami():
    """Resolve the caller's principal from whatever credential they sent —
    the one surface every client can use to learn who the system thinks
    they are."""
    from hugpy_control.principals import principal_store
    auth = request.headers.get("Authorization", "")
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else None
    if bearer and bearer.startswith("hpp_"):
        p = principal_store.resolve_token(bearer)
        if p is not None:
            return jsonify({"principal": p.to_dict(), "via": "principal-token"})
    try:
        from hugpy_server.app.operator_auth import operator_authenticated
        if operator_authenticated():
            return jsonify({"principal":
                            principal_store.resolve_operator().to_dict(),
                            "via": "operator"})
    except Exception:
        pass
    if bearer:
        try:
            from hugpy_server.app.functions.imports.utils.api_keys import verify_api_key, key_id_for_token
            if verify_api_key(bearer):
                kid = key_id_for_token(bearer)
                return jsonify({"principal": principal_store
                                .resolve_api_key(kid or "?").to_dict(),
                                "via": "api-key"})
        except Exception:
            pass
    return jsonify({"principal": None, "via": None})


# ---------------------------------------------------------------------------
# F4 — settings control API (CON-08 surface). GET open (UIs read it);
# writes operator-gated via _SENSITIVE.
# ---------------------------------------------------------------------------
@comms_bp.route("/settings", methods=["GET"])
def settings_namespaces():
    from hugpy_control.settings import settings_store
    return jsonify({"namespaces": settings_store.namespaces()})


@comms_bp.route("/settings/<ns>", methods=["GET"])
def settings_ns(ns):
    from hugpy_control.settings import settings_store
    return jsonify({"ns": ns, "values": settings_store.all(ns)})


@comms_bp.route("/settings/<ns>/<path:key>", methods=["GET"])
def settings_get(ns, key):
    from hugpy_control.settings import settings_store
    return jsonify({"ns": ns, "key": key,
                    "value": settings_store.get(ns, key)})


@comms_bp.route("/settings/<ns>/<path:key>", methods=["POST", "PUT"])
def settings_set(ns, key):
    from hugpy_control.settings import settings_store
    body = request.get_json(silent=True)
    if body is None or "value" not in body:
        abort(400, description='body must be {"value": ...} '
                               '(or {"merge": {...}} for dict patch)')
    if isinstance(body.get("merge"), dict):
        value = settings_store.merge(ns, key, body["merge"])
    else:
        value = settings_store.set(ns, key, body["value"])
    audit("settings.set", {"ns": ns, "key": key})
    return jsonify({"ns": ns, "key": key, "value": value})


@comms_bp.route("/settings/<ns>/<path:key>", methods=["DELETE"])
def settings_delete(ns, key):
    from hugpy_control.settings import settings_store
    existed = settings_store.delete(ns, key)
    audit("settings.delete", {"ns": ns, "key": key})
    return jsonify({"ns": ns, "key": key, "deleted": existed})


# ---------------------------------------------------------------------------
# Bot M2M pref write — deliberately open, same trust tier as the discord
# inbox/outbox routes the bot already uses. Users set their own default
# model via /model in Discord; the value lands in the F4 settings store so
# the console sees and can override it (CON-08).
# ---------------------------------------------------------------------------
@comms_bp.route("/discord/prefs", methods=["POST"])
def discord_prefs_set():
    from hugpy_control.settings import settings_store
    body = request.get_json(silent=True) or {}
    user_id = str(body.get("user_id") or "")
    if not user_id:
        abort(400, description="user_id required")
    model_key = body.get("model_key")
    value = settings_store.merge("discord.users", user_id,
                                 {"model": model_key})
    return jsonify({"user_id": user_id, "value": value})


# ---------------------------------------------------------------------------
# F3.1 — model metadata (CON-03 backend): one endpoint every picker reads.
# ---------------------------------------------------------------------------
@comms_bp.route("/models/<path:model_key>/meta", methods=["GET"])
def model_meta_route(model_key):
    from hugpy_engine.config.main import get_model_config
    from hugpy_engine.config.models.model_meta import model_meta
    # get_model_config RAISES KeyError for an unknown key (the `is None` guard
    # below only catches a returned-None). A picker polling this with a stale
    # key must get a clean 404, never a 500. keeper 2026-07-13 (see serving_get).
    try:
        cfg = get_model_config(model_key)
    except KeyError:
        cfg = None
    if cfg is None:
        abort(404, description="Unknown model key.")
    vram = None
    raw = (request.args.get("vram_gib") or "").strip()
    if raw:
        try:
            vram = int(float(raw) * (1024 ** 3))
        except ValueError:
            abort(400, description="vram_gib must be a number")
    worker_id = (request.args.get("worker") or "").strip()
    if worker_id and vram is None:
        # Annotate against a live worker's free VRAM (the roadmap's ask).
        try:
            from hugpy_fleet.central.workers import list_workers
            for w in list_workers():
                if w.get("id") == worker_id or w.get("name") == worker_id:
                    vram = w.get("vram_free")
                    break
        except Exception:
            pass
    # ?gguf=<basename|quant> sizes + recommends for THAT variant (the per-worker
    # quant dropdown's fit readout); absent = the effective quant, as before.
    sel = (request.args.get("gguf") or "").strip() or None
    return jsonify(model_meta(cfg, vram_bytes=vram, select_gguf=sel))
