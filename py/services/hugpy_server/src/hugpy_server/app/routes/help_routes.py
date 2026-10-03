"""OPERATOR help-agent query path — the console Help button's backend.

    POST /llm/help/sessions                 {prompt, context?, backend?:"local"}
                                            -> 201 transcript
    GET  /llm/help/sessions                 -> {sessions:[{id,title,backend,status,...}]}
    GET  /llm/help/sessions/<id>?since=N    -> transcript (records from index N)
    POST /llm/help/sessions/<id>/messages   {text, context?} -> transcript
    GET  /llm/help/sessions/<id>/stream?since=N
                                            -> text/event-stream of transcript records
    POST /llm/help/sessions/<id>/stop       -> transcript (interrupts the turn)
    GET  /llm/help/logs?source=central|worker|compute-actions|failures|diagnostics|audit|workers
                        &lines=&worker=&model=&since=&request_id=&errors=1
                                            -> bounded read-only readout
    GET  /llm/help/status                   -> {backends:[...available...], enabled}

GATE: OPERATOR ONLY, always. The agent behind this path can read logs and EDIT
CODE, so this gate is strict — ``operator_auth.operator_authenticated()`` with
NO ``HUGPY_AGENT_OPEN`` waiver (that flag is a fleet-view testing convenience;
it never opens a code-editing surface). Members get 403, anonymous 401. The
same paths are also listed in ``operator_auth._SENSITIVE`` so the central
before_request gate covers them independently. ``HUGPY_HELP_AGENT=0`` turns
the whole surface off (503).

Transcripts: ``$PROJECTS_HOME/help_sessions/<id>.jsonl`` (append-only).
"""
from __future__ import annotations

import json
import os
import time

from flask import Response, jsonify, request, stream_with_context

from abstract_flask import get_bp
from hugpy_server.app import help_agent

help_bp, logger = get_bp("help_bp", __name__)

STREAM_MAX_S = float(os.environ.get("HUGPY_HELP_STREAM_MAX_S", "900"))
STREAM_POLL_S = float(os.environ.get("HUGPY_HELP_STREAM_POLL_S", "1.0"))


def _enabled() -> bool:
    return (os.environ.get("HUGPY_HELP_AGENT", "1") or "").strip().lower() not in (
        "0", "false", "no", "off")


def _no_session(sid):
    """404 body naming what was looked for and why it was not found."""
    try:
        store = help_agent.service().store
        if not help_agent.ID_RX.match(sid or ""):
            why = f"id does not match the session id format {help_agent.ID_RX.pattern!r}"
        else:
            why = f"no transcript file {store.path(sid)}"
    except Exception as exc:  # noqa: BLE001
        why = f"session store lookup failed: {type(exc).__name__}: {exc}"
    return jsonify({"error": f"no help session {sid!r}: {why}", "session": sid}), 404


def _gate():
    """None when the caller is an operator; a (response, code) otherwise.
    Fails closed if the auth module cannot load."""
    try:
        from hugpy_server.app.operator_auth import (
            ROLE_MEMBER, operator_authenticated, principal_role)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": "operator authentication required: operator_auth failed to "
                                 f"import ({type(exc).__name__}: {exc})",
                        "gate": "operator"}), 401
    try:
        if operator_authenticated():
            if not _enabled():
                return jsonify({"error": "help agent is disabled on this deployment "
                                         "(HUGPY_HELP_AGENT=0)"}), 503
            return None
        role = principal_role()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": "operator authentication required: the auth check raised "
                                 f"{type(exc).__name__}: {exc}",
                        "gate": "operator"}), 401
    if role == ROLE_MEMBER:
        return jsonify({"error": "the hugpy help agent is operator-only "
                                 "(it can read logs and edit code)",
                        "gate": "operator"}), 403
    return jsonify({"error": f"operator authentication required: this session's role is {role!r}",
                    "gate": "operator", "role": role}), 401


def _who() -> str:
    try:
        from hugpy_server.app.operator_auth import principal_username
        return principal_username() or "operator"
    except Exception:  # noqa: BLE001
        return "operator"


def _since() -> int:
    try:
        return max(0, int(request.args.get("since") or 0))
    except (TypeError, ValueError):
        return 0


@help_bp.route("/llm/help/status", methods=["GET"])
def help_status():
    g = _gate()
    if g is not None:
        return g
    _chosen, tried = help_agent.select_backend()
    return jsonify({"enabled": True, "backends": tried,
                    "sessions_dir": help_agent.HelpStore().root})


@help_bp.route("/llm/help/sessions", methods=["GET"])
def help_sessions_list():
    g = _gate()
    if g is not None:
        return g
    return jsonify({"sessions": help_agent.service().store.list()})


@help_bp.route("/llm/help/sessions", methods=["POST"])
def help_sessions_start():
    g = _gate()
    if g is not None:
        return g
    body = request.get_json(silent=True) or {}
    prefer = "local" if str(body.get("backend") or "").lower() in ("local", "local-model") else ""
    try:
        t = help_agent.service().start(body.get("prompt") or "", body.get("context"),
                                       by=_who(), prefer=prefer)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    logger.info("help session %s started by %s via %s", t["id"], _who(), t.get("backend"))
    return jsonify(t), 201


@help_bp.route("/llm/help/sessions/<sid>", methods=["GET"])
def help_session_get(sid):
    g = _gate()
    if g is not None:
        return g
    try:
        return jsonify(help_agent.service().transcript(sid, since=_since()))
    except KeyError:
        return _no_session(sid)


@help_bp.route("/llm/help/sessions/<sid>/messages", methods=["POST"])
def help_session_send(sid):
    g = _gate()
    if g is not None:
        return g
    body = request.get_json(silent=True) or {}
    try:
        t = help_agent.service().send(sid, body.get("text") or "", body.get("context"),
                                      by=_who())
    except KeyError:
        return _no_session(sid)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except help_agent.BackendError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify(t)


@help_bp.route("/llm/help/sessions/<sid>/stop", methods=["POST"])
def help_session_stop(sid):
    g = _gate()
    if g is not None:
        return g
    try:
        return jsonify(help_agent.service().stop(sid))
    except KeyError:
        return _no_session(sid)


@help_bp.route("/llm/help/sessions/<sid>/stream", methods=["GET"])
def help_session_stream(sid):
    """SSE: every transcript record from ``since`` on, as it lands; one
    ``status`` frame per change; ends when the session goes idle/error/stopped
    (after at least one poll) or after HUGPY_HELP_STREAM_MAX_S."""
    g = _gate()
    if g is not None:
        return g
    svc = help_agent.service()
    if not svc.store.exists(sid):
        return _no_session(sid)
    start_at = _since()

    def gen():
        cursor = start_at
        last_status = None
        deadline = time.time() + STREAM_MAX_S
        idle_polls = 0
        while time.time() < deadline:
            t = svc.transcript(sid, since=cursor)
            for rec in t["records"]:
                yield f"id: {rec['i']}\ndata: {json.dumps(rec, default=str)}\n\n"
                cursor = rec["i"] + 1
            if t["status"] != last_status:
                last_status = t["status"]
                yield f"event: status\ndata: {json.dumps({'status': last_status, 'count': t['count']})}\n\n"
            if last_status != "busy":
                idle_polls += 1
                if idle_polls >= 2:
                    return
            else:
                idle_polls = 0
                yield ": keepalive\n\n"
            time.sleep(STREAM_POLL_S)

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@help_bp.route("/llm/help/logs", methods=["GET"])
def help_logs():
    g = _gate()
    if g is not None:
        return g
    a = request.args
    since = a.get("since")
    try:
        since_f = float(since) if since not in (None, "") else None
    except (TypeError, ValueError):
        since_f = None
    return jsonify(help_agent.read_logs(
        a.get("source") or "", lines=a.get("lines") or 200, worker=a.get("worker") or "",
        model=a.get("model") or "", since=since_f, request_id=a.get("request_id") or "",
        errors=(a.get("errors") or "").lower() in ("1", "true", "yes")))


# ── help tickets (operator 2026-10-02): pre-existing approvals ───────────────
def _ticket_prompt(t: dict) -> str:
    d = t.get("detail") or {}
    return (f"A {t.get('kind')} ticket was filed: {t.get('title')}.\n"
            f"model={t.get('model_key')} worker={t.get('worker_name') or t.get('worker_id')} "
            f"verdict={d.get('verdict')} file={d.get('file')} gpu={d.get('gpu')} ctx={d.get('ctx')} "
            f"4-bit={d.get('bnb')}\npredicted={json.dumps(d.get('predicted'), default=str)[:1500]}\n"
            f"measured={json.dumps(d.get('measured'), default=str)[:1500]}\n"
            "Explain the most likely cause of the disagreement between the load gate's prediction "
            "and the measurement, and what (if anything) should change.")


@help_bp.route("/llm/help/tickets", methods=["GET"])
def help_tickets_list():
    """{tickets, pending, keeper} — ?status=pending|acted|sent|dismissed (default all)."""
    g = _gate()
    if g is not None:
        return g
    from hugpy_server.app import help_tickets
    from hugpy_server.app.routes.keeper_help_routes import keeper_available
    status = request.args.get("status") or None
    return jsonify({"tickets": help_tickets.list_tickets(status, 200),
                    "pending": help_tickets.pending_count(), "keeper": keeper_available()})


@help_bp.route("/llm/help/tickets/<int:tid>/act", methods=["POST"])
def help_ticket_act(tid):
    """The operator's choice on one ticket. Body {action}:
      calibrate — run the calibration again (same model, worker, 4-bit flag)
      keeper    — file it on the keeper bridge (pending approval) when one exists
      discuss   — open a help session primed with the ticket
      dismiss   — close it"""
    g = _gate()
    if g is not None:
        return g
    from hugpy_server.app import help_tickets
    t = help_tickets.get_ticket(tid)
    if not t:
        return jsonify({"error": f"no help ticket {tid}"}), 404
    action = str((request.get_json(silent=True) or {}).get("action") or "").lower()
    d = t.get("detail") or {}
    if action == "dismiss":
        return jsonify(help_tickets.set_status(tid, "dismissed", {"by": _who()}))
    if action == "approve" and t.get("kind") == "env-profile":
        from hugpy_server.app import env_profiles as ep
        prof = ep.approve(d.get("name"), by=_who())
        if not prof:
            return jsonify({"error": f"no environment {d.get('name')!r}"}), 404
        out = help_tickets.set_status(tid, "acted", {"action": "approve", "env_profile": d.get("name"),
                                                     "by": _who()})
        return jsonify({"ticket": out, "env_profile": prof})
    if action == "calibrate":
        from hugpy_server.app import calibration_run
        from hugpy_server.app.routes.worker_routes import get_worker
        worker = get_worker(t.get("worker_id")) if t.get("worker_id") else None
        if worker is None:
            return jsonify({"error": f"worker {t.get('worker_id')!r} is not in the registry"}), 404
        job = calibration_run.start(worker, t["model_key"], bnb=bool(d.get("bnb")), evict_others=False)
        if job.get("error"):
            return jsonify({"error": job["error"], "job": job}), 409
        out = help_tickets.set_status(tid, "acted", {"action": "calibrate", "job_id": job.get("job_id"),
                                                     "by": _who()})
        return jsonify({"ticket": out, "job": job})
    if action == "keeper":
        from hugpy_server.app.routes.keeper_help_routes import file_keeper_report
        msg, bridge = file_keeper_report(t["title"], proposed="look into this calibration finding",
                                         context=json.dumps(d, default=str)[:4000],
                                         username=_who(), source="help-ticket")
        if bridge is None:
            return jsonify({"error": "no keeper is available on this deployment"}), 409
        if not msg:
            return jsonify({"error": "could not file it on the keeper bridge"}), 503
        out = help_tickets.set_status(tid, "sent", {"action": "keeper", "bridge_id": bridge["id"],
                                                    "message_id": msg.get("id"), "by": _who()})
        return jsonify({"ticket": out, "keeper": {"bridge_id": bridge["id"], "message_id": msg.get("id")}})
    if action == "discuss":
        try:
            sess = help_agent.service().start(_ticket_prompt(t), {"ticket": t.get("id")}, by=_who())
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        out = help_tickets.set_status(tid, "acted", {"action": "discuss", "session": sess.get("id"),
                                                     "by": _who()})
        return jsonify({"ticket": out, "session": sess})
    return jsonify({"error": "action must be one of calibrate, keeper, discuss, dismiss "
                             "(approve for an env-profile ticket)"}), 400
