"""Member-callable HELP surface: file a request/proposed action for the Keeper.

WHY
---
The floating Help widget (``react/ui_shared/help/helpWidget.js``, mounted on all
four dev.hugpy.ai surfaces) answers questions by streaming ``/chat/stream``. That
is READ-ONLY talk. The moment a member wants something DONE — a fix applied, a
setting changed, anything with a side effect — the widget must not act. It files
a ticket instead, and an operator decides.

This module is that filing seam, and nothing else:

    POST /keeper/help/report   {description, proposed_action?, page_url?, context?}
        -> 201 {"ok": true, "ticket_id": <bridge message id>, "bridge_id": ...,
                "status": "pending"}

The report is appended to the Keeper's Discord bridge as an OUTBOUND message with
``status="pending"`` — the exact record shape ``defer_mode="defer"`` produces for
a model/keeper draft. That means it lands in the console's EXISTING approval flow
with no new review UI: an operator sees it in the bridge transcript and either
``POST /discord/bridges/<id>/approve`` (which sends it into the Discord channel)
or ``/reject`` (which discards it). Until then it stays pending. This route never
enqueues outbound itself — writing a "sent" row or calling ``enqueue_outbound``
would be exactly the console→channel send vector that was deliberately removed
from ``discord_routes.discord_bridge_send`` in 2026-07-09.

AUTH
----
Member OR operator, via ``operator_auth.member_authenticated()`` — the same rule
``member_auth`` applies to the studio/media plane, checked HERE because
``/keeper/*`` is outside that gate's surface regex. Anonymous gets a JSON 401 so
the widget can show its sign-in prompt. Members are allowed deliberately: filing
a request is a product action; only APPROVING it is operator-only.

BRIDGE SELECTION (and the config knob this introduces)
------------------------------------------------------
``_resolve_help_bridge()``, first match wins:

  1. ``HUGPY_HELP_BRIDGE_ID`` — an explicit bridge id. THE knob introduced by
     this module; set it to pin help tickets to one channel.
  2. the first bridge whose ``defer_mode`` is ``"defer"`` (the "hold everything
     for the operator" bridges — semantically the approval queue).
  3. the first bridge whose ``keeper_target`` matches ``HUGPY_HELP_KEEPER_TARGET``
     (default ``session:hugpy-keeper``) — this deployment's designated keeper
     channel. (As of 2026-08-06 no bridge is in ``defer`` mode, so this is the
     branch that actually fires here.)
  4. the first ``brain="keeper"`` bridge, as a last resort.

Bridges with ``log_mode="none"`` are skipped throughout: an ephemeral bridge
retains nothing, and a ticket that isn't retained can never be approved.
"""
import os

from flask import request, jsonify

from abstract_flask import get_bp
from hugpy_server.app.functions.imports.utils.discord_bindings import (
    list_bridges,
    get_bridge,
    append_bridge_message,
)

keeper_help_bp, logger = get_bp("keeper_help_bp", __name__)

# The designated keeper channel's bridge marker when no explicit id is pinned.
_DEFAULT_KEEPER_TARGET = "session:hugpy-keeper"
# Truncation guards — a ticket is a human-readable Discord message, not a log
# sink. Discord's own hard cap is 2000 chars; the console renders the transcript.
_MAX_FIELD = 4000
_MAX_CONTEXT = 4000


def _retains(bridge) -> bool:
    """False for an ephemeral bridge — nothing appended to it survives, so a
    pending ticket there could never reach an operator."""
    return bool(bridge) and (bridge.get("log_mode", "open") != "none")


def _resolve_help_bridge():
    """The bridge help tickets are filed against, or None. See the module
    docstring for the (documented, env-overridable) precedence."""
    pinned = (os.getenv("HUGPY_HELP_BRIDGE_ID") or "").strip()
    if pinned:
        bridge = get_bridge(pinned)
        if _retains(bridge):
            return bridge
        logger.warning("HUGPY_HELP_BRIDGE_ID=%s is unknown or ephemeral — falling back",
                       pinned)
    try:
        bridges = list_bridges() or []
    except Exception:  # noqa: BLE001 — a store hiccup must not 500 the widget
        logger.warning("help report: bridge listing failed", exc_info=True)
        return None
    usable = [b for b in bridges if _retains(b)]
    for b in usable:
        if b.get("defer_mode") == "defer":
            return b
    target = (os.getenv("HUGPY_HELP_KEEPER_TARGET") or _DEFAULT_KEEPER_TARGET).strip()
    for b in usable:
        if b.get("keeper_target") == target:
            return b
    for b in usable:
        if b.get("brain") == "keeper":
            return b
    return None


def _clip(value, limit=_MAX_FIELD) -> str:
    text = (value or "")
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


def _require_member():
    """None when the caller may file a ticket; a (response, 401) tuple otherwise.
    Fails closed if the gate module is unavailable (same posture as
    agent_routes._require_member_strict)."""
    try:
        from hugpy_server.app.operator_auth import member_authenticated
    except Exception:  # noqa: BLE001
        return jsonify({"error": "authentication required"}), 401
    if not member_authenticated():
        return jsonify({"error": "authentication required"}), 401
    return None


def _username() -> str:
    try:
        from hugpy_server.app.operator_auth import principal_username
        return principal_username() or ""
    except Exception:  # noqa: BLE001
        return ""


@keeper_help_bp.route("/keeper/help/ask", methods=["POST"])
def keeper_help_ask():
    """THE DIRECT LINE (operator ruling 2026-08-20): the help widget's Ask tab
    talks to the hugpy VM's KEEPER — the same B seat as the station console and
    the bridge watcher — grounded in live fleet facts (workers, downloader,
    queue incl. persistent failure records, store space) plus B's own MCT
    ledger state. No canned answers and no substitute model: if the keeper's
    brain is unreachable the caller gets ``offline: true`` and B's
    deterministic state readout, clearly labelled.

    Body: {"text": str, "history": [{role, content}, ...]?}
    Reply: {"ok": bool, "reply": str, "offline": bool}
    """
    gate = _require_member()
    if gate is not None:
        return gate
    from hugpy_server.app.keeper_line import fleet_grounding, keeper_ask
    body = request.get_json(silent=True) or {}
    text = _clip(body.get("text"), _MAX_CONTEXT)
    if not text:
        return jsonify({"error": "text is required"}), 400
    history = body.get("history") if isinstance(body.get("history"), list) else []
    grounded = (
        "You are the hugpy fleet keeper answering the console help line. "
        "Resolve the user's issue: name the cause and the concrete action. "
        "If the fix needs an operator's hands, say exactly what to run/click "
        "and suggest filing it via the help widget's 'Request a fix' tab.\n\n"
        f"LIVE FLEET FACTS:\n{fleet_grounding()}\n\n"
        f"USER ({_username() or 'member'}):\n{text}")
    res = keeper_ask(grounded, history=history)
    return jsonify({"ok": bool(res.get("reply")),
                    "reply": res.get("reply") or "",
                    "offline": bool(res.get("offline"))})


@keeper_help_bp.route("/keeper/help/report", methods=["POST"])
def keeper_help_report():
    """MEMBER or OPERATOR: file a help request / proposed action for approval.

    Body: ``{description (required), proposed_action?, page_url?, context?}``.
    Returns 201 ``{ok, ticket_id, bridge_id, status:"pending", submitted_by}``.
    The widget shows "submitted for approval" off ``ticket_id``.

    The widget NEVER executes anything — this route only RECORDS. Nothing here
    reaches Discord until an operator approves the pending message in the
    console (``POST /discord/bridges/<bridge_id>/approve``)."""
    denied = _require_member()
    if denied is not None:
        return denied

    body = request.get_json(silent=True) or {}
    description = _clip(body.get("description"))
    if not description:
        return jsonify({"error": "description is required"}), 400
    proposed = _clip(body.get("proposed_action"))
    page_url = _clip(body.get("page_url"), 512)
    context = _clip(body.get("context"), _MAX_CONTEXT)
    username = _username() or "operator"

    bridge = _resolve_help_bridge()
    if not bridge:
        logger.error("help report: no keeper bridge available (set HUGPY_HELP_BRIDGE_ID)")
        return jsonify({
            "error": "no keeper approval channel is configured on this deployment "
                     "(set HUGPY_HELP_BRIDGE_ID)",
        }), 503

    lines = [
        "**Help request (pending operator approval)**",
        f"from: {username}",
    ]
    if page_url:
        lines.append(f"page: {page_url}")
    lines.append("")
    lines.append(description)
    if proposed:
        lines.append("")
        lines.append("proposed action:")
        lines.append(proposed)
    if context:
        lines.append("")
        lines.append("page context:")
        lines.append(context)
    content = "\n".join(lines)

    msg = append_bridge_message(
        bridge["id"],
        direction="out",
        source="help-widget",
        content=content,
        author=username,
        # ALWAYS pending, regardless of the bridge's own defer_mode: a member's
        # proposed action is operator-approved by construction, never auto-sent.
        status="pending",
    )
    if not msg:
        return jsonify({"error": "could not file the request"}), 503
    logger.info("help report filed: bridge=%s msg=%s by=%s",
                bridge["id"], msg.get("id"), username)
    return jsonify({
        "ok": True,
        "ticket_id": msg.get("id"),
        "bridge_id": bridge["id"],
        "status": msg.get("status", "pending"),
        "submitted_by": username,
        "ts": msg.get("ts"),
    }), 201


@keeper_help_bp.route("/keeper/help/report", methods=["GET"])
def keeper_help_report_list():
    """MEMBER or OPERATOR: the caller's own filed tickets and their current
    status (pending / sent / rejected), so the widget can show what happened to
    a request after the operator acted. Operators see every widget-filed ticket
    on the resolved bridge."""
    denied = _require_member()
    if denied is not None:
        return denied
    bridge = _resolve_help_bridge()
    if not bridge:
        return jsonify({"tickets": []})
    try:
        from hugpy_server.app.functions.imports.utils.discord_bindings import get_bridge_messages
        messages = get_bridge_messages(bridge["id"], 0.0) or []
    except Exception:  # noqa: BLE001
        logger.warning("help report list failed", exc_info=True)
        return jsonify({"tickets": []})
    try:
        from hugpy_server.app.operator_auth import operator_authenticated
        is_operator = bool(operator_authenticated())
    except Exception:  # noqa: BLE001
        is_operator = False
    mine = _username()
    tickets = [
        {"ticket_id": m.get("id"), "status": m.get("status"),
         "ts": m.get("ts"), "content": m.get("content"),
         "author": m.get("author")}
        for m in messages
        if m.get("source") == "help-widget"
        and (is_operator or (mine and m.get("author") == mine))
    ]
    tickets.sort(key=lambda t: t.get("ts") or 0, reverse=True)
    return jsonify({"bridge_id": bridge["id"], "tickets": tickets[:50]})
