"""HTTP surface for model <-> Discord bindings (the bot's "mobile arm").

Two audiences, like worker_routes / phone_brick_routes:

  * the console UI (human-driven):
        GET    /discord/bindings
        POST   /discord/bindings        {"model_key", "channel_id"?, "user_id"?, "label"?}
        DELETE /discord/bindings/<id>
        POST   /discord/outbox          {"content", "channel_id"?|"user_id"?|"binding_id"?}
  * the hugpy bot (machine-to-machine, polls central):
        GET    /discord/resolve?channel_id=&user_id=   -> {"model_key": ... | null}
        POST   /discord/outbox/drain                   -> {"messages": [...]}
        POST   /discord/channels   {"channels":[{id,name,guild,guild_id}]}  (bot reports)
        GET    /discord/channels   -> {"channels":[...], "channels_at": ts}  (UI dropdown)
  * the /fleet screen (MEMBER, human-driven; 2026-08-06):
        POST   /discord/bot-links          {"kind","label"?,"scopes"?,"ttl_days"?}
        GET    /discord/bot-links          the caller's own bot keys (operator: all)
        DELETE /discord/bot-links/<key_id> revoke one (operator, or its creator)
    A per-account scoped hugpy API key + the config a member's OWN Discord bot
    needs. Member-gated in this module (never in operator_auth._SENSITIVE) —
    see the block above ``_BOT_LINK_CREATED_BY`` for what it does and does NOT
    hand out.

All state lives in functions.imports.utils.discord_bindings; this module only
translates HTTP <-> that store. The blueprint is discovered + mounted bare
(/discord/...) via routes/__init__, and dual-mounted under /api in
wsgi_app.get_hugpy_flask (the /api prefix is stripped by nginx in prod and by
ApiPrefixMiddleware on bare gunicorn), exactly like the GPU worker routes.
"""
from pydantic import BaseModel
from flask import request, jsonify, abort

from abstract_flask import get_bp
from hugpy_engine.config.models.models_config import get_models_dict
from hugpy_server.app.functions.imports.utils.discord_bindings import (
    list_bindings,
    add_binding,
    remove_binding,
    resolve_model,
    enqueue_outbound,
    drain_outbound,
    set_channels,
    get_channels,
    set_users,
    get_users,
    add_bridge,
    list_bridges,
    get_bridge,
    remove_bridge,
    bridge_for_channel,
    append_bridge_message,
    get_bridge_messages,
    update_bridge_message,
    clear_bridge_messages,
    add_session,
    list_sessions,
    revoke_session,
    delete_session,
    prune_sessions,
    session_by_token,
)
import os
import time

discord_bp, logger = get_bp("discord_bp", __name__)


class BindingRequest(BaseModel):
    model_key: str
    channel_id: str | None = None
    user_id: str | None = None
    label: str | None = None


class OutboundRequest(BaseModel):
    content: str
    channel_id: str | None = None
    user_id: str | None = None
    binding_id: str | None = None


@discord_bp.route("/discord/bindings", methods=["GET"])
def discord_bindings_list():
    return jsonify({"bindings": list_bindings()})


@discord_bp.route("/discord/bindings", methods=["POST"])
def discord_bindings_create():
    body = BindingRequest(**(request.get_json(silent=True) or {}))
    if body.model_key not in get_models_dict(dict_return=True):
        abort(404, description="Unknown model key.")
    if not body.channel_id and not body.user_id:
        abort(400, description="Provide a channel_id and/or a user_id.")
    try:
        binding = add_binding(
            model_key=body.model_key,
            channel_id=body.channel_id,
            user_id=body.user_id,
            label=body.label,
        )
    except ValueError as exc:
        abort(400, description=str(exc))
    return jsonify(binding), 201


@discord_bp.route("/discord/bindings/<binding_id>", methods=["DELETE"])
def discord_bindings_delete(binding_id):
    if not remove_binding(binding_id):
        abort(404, description="Unknown binding id.")
    return jsonify({"ok": True, "id": binding_id})


@discord_bp.route("/discord/resolve", methods=["GET"])
def discord_resolve():
    """The bot asks central which model an inbound (channel, user) should hit."""
    model_key = resolve_model(
        channel_id=request.args.get("channel_id"),
        user_id=request.args.get("user_id"),
    )
    return jsonify({"model_key": model_key})


@discord_bp.route("/discord/outbox", methods=["POST"])
def discord_outbox_enqueue():
    """Queue a model-originated message to be pushed into its Discord target."""
    body = OutboundRequest(**(request.get_json(silent=True) or {}))
    if not (body.content or "").strip():
        abort(400, description="content is required.")
    try:
        msg = enqueue_outbound(
            content=body.content,
            channel_id=body.channel_id,
            user_id=body.user_id,
            binding_id=body.binding_id,
        )
    except ValueError as exc:
        abort(400, description=str(exc))
    return jsonify(msg), 201


@discord_bp.route("/discord/outbox/drain", methods=["POST"])
def discord_outbox_drain():
    """The bot polls this and delivers each returned message to Discord."""
    return jsonify({"messages": drain_outbound()})


@discord_bp.route("/discord/channels", methods=["GET"])
def discord_channels_list():
    """The console UI reads this to populate the channel dropdown."""
    return jsonify(get_channels())


@discord_bp.route("/discord/channels", methods=["POST"])
def discord_channels_report():
    """The bot reports the text channels it can currently see."""
    body = request.get_json(silent=True) or {}
    channels = body.get("channels")
    if not isinstance(channels, list):
        abort(400, description="channels must be a list.")
    return jsonify(set_channels(channels))


@discord_bp.route("/discord/users", methods=["GET"])
def discord_users_list():
    """The console UI reads this to populate the user dropdown."""
    return jsonify(get_users())


@discord_bp.route("/discord/users", methods=["POST"])
def discord_users_report():
    """The bot reports the guild members it can see (needs members intent)."""
    body = request.get_json(silent=True) or {}
    users = body.get("users")
    if not isinstance(users, list):
        abort(400, description="users must be a list.")
    return jsonify(set_users(users))


# ── bridges: a console session <-> a Discord channel ──────────────────────
class BridgeRequest(BaseModel):
    binding_id: str | None = None         # resolve channel/model from an existing binding
    channel_id: str | None = None         # …or specify directly
    model_key: str | None = None
    user_id: str | None = None
    directive: str | None = None
    defer_mode: str = "auto"              # auto | defer | directive
    brain: str = "model"                  # model (auto-reply) | keeper (keeper drives)
    keeper_target: str | None = None      # informational: which keeper is wired here
    log_mode: str = "open"                # open (retain transcript) | none (ephemeral)


@discord_bp.route("/discord/bridges", methods=["GET"])
def discord_bridges_list():
    return jsonify({"bridges": list_bridges()})


@discord_bp.route("/discord/bridges", methods=["POST"])
def discord_bridges_create():
    body = BridgeRequest(**(request.get_json(silent=True) or {}))
    channel_id, model_key, user_id = body.channel_id, body.model_key, body.user_id
    if body.binding_id:
        match = next((b for b in list_bindings() if b.get("id") == body.binding_id), None)
        if not match:
            abort(404, description="Unknown binding id.")
        channel_id = channel_id or match.get("channel_id")
        user_id = user_id or match.get("user_id")
        model_key = model_key or match.get("model_key")
    if not channel_id:
        abort(400, description="Provide a binding_id or a channel_id.")
    try:
        bridge = add_bridge(channel_id=channel_id, model_key=model_key, user_id=user_id,
                            directive=body.directive, defer_mode=body.defer_mode,
                            brain=body.brain, keeper_target=body.keeper_target,
                            log_mode=body.log_mode)
    except ValueError as exc:
        abort(400, description=str(exc))
    return jsonify(bridge), 201


@discord_bp.route("/discord/bridges/<bridge_id>", methods=["DELETE"])
def discord_bridges_delete(bridge_id):
    if not remove_bridge(bridge_id):
        abort(404, description="Unknown bridge id.")
    return jsonify({"ok": True, "id": bridge_id})


@discord_bp.route("/discord/bridges/<bridge_id>/messages", methods=["GET"])
def discord_bridge_messages(bridge_id):
    """The console polls this for the merged transcript (since a timestamp)."""
    if not get_bridge(bridge_id):
        abort(404, description="Unknown bridge id.")
    try:
        since = float(request.args.get("since", "0") or 0)
    except ValueError:
        since = 0.0
    return jsonify({"messages": get_bridge_messages(bridge_id, since)})


@discord_bp.route("/discord/bridges/<bridge_id>/messages", methods=["DELETE"])
def discord_bridge_clear_messages(bridge_id):
    """Clear a bridge's console-side transcript. Operator-gated (see
    operator_auth._SENSITIVE). Does NOT delete anything from the Discord channel
    — only the history stored here (which also resets the model reply context
    for model-brained bridges)."""
    if not get_bridge(bridge_id):
        abort(404, description="Unknown bridge id.")
    cleared = clear_bridge_messages(bridge_id)
    return jsonify({"ok": True, "id": bridge_id, "cleared": cleared or 0})


@discord_bp.route("/discord/bridges/<bridge_id>/send", methods=["POST"])
def discord_bridge_send(bridge_id):
    """DISABLED 2026-07-09 (operator): console→channel send was an outbound-send
    vector reachable by anyone past the auth gate (not just the operator), so the
    console can no longer push arbitrary text into a Discord channel. Legitimate
    outbound still flows via /keeper-reply (keeper-driven, defer-gated) and the
    scoped session token (/discord/session/<token>/send). To restore: drop the
    abort and un-archive the body below."""
    if not get_bridge(bridge_id):
        abort(404, description="Unknown bridge id.")
    abort(403, description="Console→channel send is disabled (removed as an "
                           "outbound-send vector). Use a keeper or a session token.")
    # ── ARCHIVED original body (kept per the no-delete rule) ──────────────
    # body = request.get_json(silent=True) or {}
    # content = (body.get("content") or "").strip()
    # if not content:
    #     abort(400, description="content is required.")
    # source = body.get("source") or "console"
    # msg = append_bridge_message(bridge_id, direction="out", source=source,
    #                             content=content, author=body.get("author"))
    # enqueue_outbound(content=content, channel_id=bridge.get("channel_id"),
    #                  user_id=bridge.get("user_id"))
    # return jsonify(msg or {}), 201


# ── candidate generation (the bridge's "brain") ───────────────────────────
from hugpy_platform.async_runtime import await_sync as _await_sync


from hugpy_platform.results import result_text as _result_text


_DIRECTIVE_DECIDE = (
    "\n\nWhen you have drafted your reply: if the guidance above indicates you "
    "should let the operator review it before it is sent (e.g. uncertainty, or a "
    "sensitive / out-of-scope topic), begin the reply with 'DEFER: '. Otherwise "
    "reply normally."
)


def _generate_candidate(bridge: dict, current_inbound: str = "") -> str:
    """A model reply built from the bridge's directive + recent sent transcript.
    Returns '' on any failure so the caller falls back to manual operation.

    current_inbound is used for no-logs (ephemeral) bridges, whose transcript is
    never stored: the current turn can't be read back, so it is passed in and
    appended as the sole user turn (history is empty). For logged bridges the
    inbound is already in the transcript, so current_inbound is left blank."""
    model_key = bridge.get("model_key")
    if not model_key:
        return ""
    directive = (bridge.get("directive") or
                 "You are relaying a Discord channel on the operator's behalf. Reply concisely.")
    if bridge.get("defer_mode") == "directive":
        directive = directive + _DIRECTIVE_DECIDE
    msgs = [{"role": "system", "content": directive}]
    for m in get_bridge_messages(bridge["id"]):
        if m.get("status") != "sent":
            continue  # skip pending/rejected — not part of the real conversation
        role = "user" if m.get("direction") == "in" else "assistant"
        msgs.append({"role": role, "content": m.get("content") or ""})
    if current_inbound:
        msgs.append({"role": "user", "content": current_inbound})
    if len(msgs) > 13:                       # cap history (system + last 12 turns)
        msgs = [msgs[0]] + msgs[-12:]
    # NO-THINK (utils/no_think.py). This reply is PARSED and then TRANSMITTED:
    # _route_bridge_reply gates on candidate.upper().startswith("DEFER"), so a
    # <think> prelude silently defeats the defer gate — a model that CHOSE to
    # escalate would be auto-sent instead. And whatever survives goes verbatim to
    # a live Discord channel. Nobody is watching tokens arrive here, so reasoning
    # is never legitimately shown; it is stripped and logged, not sent.
    from hugpy_engine.utils.no_think import apply_no_think, strip_think
    from hugpy_engine.dispatch.dispatch import execute_prompt
    result = _await_sync(execute_prompt(model_key=model_key,
                                        messages=apply_no_think(msgs),
                                        task="text-generation"))
    candidate, reasoning = strip_think((_result_text(result) or "").strip())
    if reasoning and not candidate:
        # Nothing but thinking — return '' so the caller falls back to manual
        # operation rather than sending a monologue to the channel.
        logger.warning("bridge %s: model returned only reasoning, no candidate",
                       bridge.get("id"))
    return candidate


@discord_bp.route("/discord/inbox", methods=["POST"])
def discord_inbox():
    """The bot posts inbound channel messages here. For a bridged channel we
    record the message, generate a candidate reply from the bridge's directive,
    then apply defer_mode: auto-send, hold for the operator, or let the model
    decide (directive mode)."""
    body = request.get_json(silent=True) or {}
    channel_id = body.get("channel_id")
    content = (body.get("content") or "").strip()
    attachments = body.get("attachments") or []
    bridge = bridge_for_channel(channel_id) if channel_id else None
    if not bridge:
        return jsonify({"bridged": False})
    if not content and not attachments:
        # nothing to record (an image-only message still has attachments)
        return jsonify({"bridged": True, "bridge_id": bridge["id"],
                        "defer_mode": bridge.get("defer_mode"), "action": "none"})

    append_bridge_message(bridge["id"], direction="in", source="discord",
                          content=content, author=body.get("author"),
                          attachments=attachments)

    # keeper-brained bridges: an attached keeper process polls the transcript
    # and drives replies itself (via /keeper-reply), so central only records the
    # inbound turn here — no auto-candidate.
    if bridge.get("brain", "model") == "keeper":
        return jsonify({"bridged": True, "bridge_id": bridge["id"],
                        "brain": "keeper", "defer_mode": bridge.get("defer_mode"),
                        "action": "await_keeper"})

    # For a no-logs bridge the inbound was not retained, so hand it to the model
    # directly; logged bridges already have it in the transcript.
    ephemeral = bridge.get("log_mode", "open") == "none"
    action = "none"
    try:
        candidate = _generate_candidate(bridge, current_inbound=content if ephemeral else "")
    except Exception:
        logger.exception("bridge candidate generation failed")
        candidate = ""

    action = _route_bridge_reply(bridge, candidate, source="model")
    return jsonify({"bridged": True, "bridge_id": bridge["id"],
                    "defer_mode": bridge.get("defer_mode"), "action": action})


def _route_bridge_reply(bridge: dict, candidate: str, *, source: str) -> str:
    """Apply a bridge's defer_mode to an outbound reply: hold it for operator
    approval (pending) or send it (enqueue to Discord). Shared by the model
    auto-candidate path and the keeper-reply path so both gate identically.
    Returns the action taken: 'none' | 'pending' | 'sent'."""
    candidate = (candidate or "").strip()
    if not candidate:
        return "none"
    mode = bridge.get("defer_mode", "auto")
    defer = (mode == "defer")
    if mode == "directive" and candidate.upper().startswith("DEFER"):
        defer = True
        candidate = candidate.split(":", 1)[1].strip() if ":" in candidate else candidate
    if not candidate:
        return "none"
    if defer:
        append_bridge_message(bridge["id"], direction="out", source=source,
                              content=candidate, status="pending")
        return "pending"
    append_bridge_message(bridge["id"], direction="out", source=source,
                          content=candidate, status="sent")
    enqueue_outbound(content=candidate, channel_id=bridge.get("channel_id"),
                     user_id=bridge.get("user_id"))
    return "sent"


@discord_bp.route("/discord/bridges/<bridge_id>/keeper-reply", methods=["POST"])
def discord_bridge_keeper_reply(bridge_id):
    """A keeper submits a drafted reply for a keeper-brained bridge. The bridge's
    defer_mode decides what happens: user-strict (defer) holds it as a pending
    candidate for operator approval in the console; keeper-choice (directive)
    lets the keeper send directly or escalate with a 'DEFER:' prefix; auto sends.
    The keeper never reaches Discord unreviewed under user-strict."""
    bridge = get_bridge(bridge_id)
    if not bridge:
        abort(404, description="Unknown bridge id.")
    body = request.get_json(silent=True) or {}
    content = (body.get("content") or "").strip()
    if not content:
        abort(400, description="content is required.")
    action = _route_bridge_reply(bridge, content, source="keeper")
    return jsonify({"bridge_id": bridge_id, "defer_mode": bridge.get("defer_mode"),
                    "action": action}), 201


@discord_bp.route("/discord/bridges/<bridge_id>/approve", methods=["POST"])
def discord_bridge_approve(bridge_id):
    """Approve a pending candidate AS GENERATED and send it to Discord. Editing
    the candidate before send was removed 2026-07-09 (operator): it let anyone
    past the auth gate push arbitrary text into a channel. Reviewers can now only
    approve the model's exact draft or reject it. (Archived: the request's
    `content` field is intentionally ignored.)"""
    bridge = get_bridge(bridge_id)
    if not bridge:
        abort(404, description="Unknown bridge id.")
    body = request.get_json(silent=True) or {}
    msg_id = body.get("message_id")
    if not msg_id:
        abort(400, description="message_id is required.")
    # NB: body.get("content") is deliberately NOT applied — send-as-generated only.
    msg = update_bridge_message(bridge_id, msg_id, status="sent")
    if not msg:
        abort(404, description="Unknown message id.")
    enqueue_outbound(content=msg["content"], channel_id=bridge.get("channel_id"),
                     user_id=bridge.get("user_id"))
    return jsonify(msg)


@discord_bp.route("/discord/bridges/<bridge_id>/reject", methods=["POST"])
def discord_bridge_reject(bridge_id):
    """Discard a pending candidate without sending it."""
    if not get_bridge(bridge_id):
        abort(404, description="Unknown bridge id.")
    msg_id = (request.get_json(silent=True) or {}).get("message_id")
    if not msg_id:
        abort(400, description="message_id is required.")
    msg = update_bridge_message(bridge_id, msg_id, status="rejected")
    if not msg:
        abort(404, description="Unknown message id.")
    return jsonify(msg)


# ── comms sessions: scoped bearer endpoints for terminal agents ────────────
# The operator mints a session bound to one channel (POST /discord/sessions —
# operator-gated) and hands the returned token to a terminal/agent session.
# The holder gets exactly two verbs on /discord/session/<token>/…: read the
# channel transcript and send into the channel. Revocable, optionally
# expiring, and the store only keeps the token's sha256. NOTE: the token
# rides in the URL path for paste-ability — it will appear in front-proxy
# access logs, which are operator-controlled here; rotate via revoke+mint.

class SessionMintRequest(BaseModel):
    channel_id: str
    label: str | None = None
    ttl_hours: float | None = None
    author: str | None = None


# Discord hard-caps a message at 2000 chars; refuse instead of letting the
# bot's delivery fail silently later.
_SESSION_MSG_LIMIT = 1900


@discord_bp.route("/discord/sessions", methods=["POST"])   # operator-gated
def discord_sessions_mint():
    body = SessionMintRequest(**(request.get_json(silent=True) or {}))
    # Inbound needs the channel bridged; reuse the existing bridge or create a
    # pure relay (keeper brain — never auto-generates a model reply).
    bridge = bridge_for_channel(body.channel_id)
    if not bridge:
        bridge = add_bridge(channel_id=body.channel_id, brain="keeper",
                            keeper_target=f"session:{(body.label or '').strip() or 'terminal'}")
    try:
        token, session = add_session(channel_id=body.channel_id,
                                     label=body.label or "",
                                     ttl_hours=body.ttl_hours,
                                     author=body.author)
    except ValueError as exc:
        abort(400, description=str(exc))
    return jsonify({"token": token,           # shown exactly once
                    "session": session,
                    "bridge_id": bridge["id"],
                    "endpoint": f"/discord/session/{token}"}), 201


@discord_bp.route("/discord/sessions", methods=["GET"])    # operator-gated
def discord_sessions_list():
    return jsonify({"sessions": list_sessions()})


@discord_bp.route("/discord/sessions/<session_id>", methods=["DELETE"])  # operator-gated
def discord_sessions_revoke(session_id):
    """``?purge=1`` (2026-08-13) removes the ROW from the ledger instead of
    marking it revoked — the UI's dead-weight cleanup. Purging a still-live
    session is revoke-and-erase in one step (the token's hash goes with the
    row, so the bearer stops verifying)."""
    purge = (request.args.get("purge") or "").strip() in ("1", "true", "yes")
    if purge:
        if not delete_session(session_id):
            abort(404, description="Unknown session id.")
        return jsonify({"ok": True, "id": session_id, "purged": True})
    if not revoke_session(session_id):
        abort(404, description="Unknown or already-revoked session id.")
    return jsonify({"ok": True, "id": session_id})


@discord_bp.route("/discord/sessions/prune", methods=["POST"])  # operator-gated
def discord_sessions_prune():
    """Sweep every revoked/expired session row from the ledger — the bulk
    "clear the clutter" counterpart of ``?purge=1`` (2026-08-13). Live
    sessions are never touched. Returns {pruned: n}. Gated in operator_auth
    (its own _SENSITIVE entry — the bare ^/discord/sessions$ regex does not
    cover this subpath)."""
    return jsonify({"ok": True, "pruned": prune_sessions()})


def _session_or_404(token: str) -> dict:
    s = session_by_token(token)
    if not s:
        abort(404)  # bad, revoked and expired are indistinguishable — no oracle
    return s


@discord_bp.route("/discord/session/<token>", methods=["GET"])
def discord_session_info(token):
    """Self-describing so the endpoint alone is enough to hand to an agent."""
    s = _session_or_404(token)
    chans = (get_channels() or {}).get("channels") or []
    name = next((c.get("name") for c in chans
                 if str(c.get("id")) == str(s["channel_id"])), None)
    return jsonify({
        "channel_id": s["channel_id"],
        "channel": name,
        "label": s.get("label") or "",
        "bridged": bool(bridge_for_channel(s["channel_id"])),
        "usage": {
            "poll": "GET …/messages?since=<ts float from last message> → {\"messages\":[…]}",
            "send": f"POST …/send {{\"content\":\"…\"}} (≤{_SESSION_MSG_LIMIT} chars) → bot delivers within ~8s",
        },
    })


@discord_bp.route("/discord/session/<token>/messages", methods=["GET"])
def discord_session_messages(token):
    s = _session_or_404(token)
    bridge = bridge_for_channel(s["channel_id"])
    if not bridge:
        return jsonify({"messages": [],
                        "warning": "channel bridge removed — inbound relay is off"})
    try:
        since = float(request.args.get("since", "0") or 0)
    except ValueError:
        since = 0.0
    return jsonify({"messages": get_bridge_messages(bridge["id"], since)})


def _validate_options(raw):
    """Validate an optional `options` payload for an interactive escalation.

    Returns None when absent (byte-for-byte the old plain-text behaviour), or a
    normalised list of 1..5 stripped label strings. Aborts 400 on any violation.
    Discord allows at most 5 buttons per action row; the >5 (select-menu) case is
    a bot-render concern, so this scoped session verb caps at 5 to keep an
    escalation answerable as a single button row."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        abort(400, description="options must be a list of strings.")
    if not raw:
        abort(400, description="options must be a non-empty list (omit it for a plain message).")
    if len(raw) > 5:
        abort(400, description="at most 5 options.")
    clean = []
    for opt in raw:
        if not isinstance(opt, str):
            abort(400, description="each option must be a string.")
        label = opt.strip()
        if not label:
            abort(400, description="options must not be empty.")
        if len(label) > 80:
            abort(400, description="each option must be at most 80 chars.")
        clean.append(label)
    return clean


# ── MEMBER bot links: per-account credentials for running a Discord bot ────
#
# 2026-08-06. The /fleet screen ("Your fleet") already lets a signed-in member
# mint an agent install link and pull the fleet-console package. These two
# blocks are the Discord half of the same product question — "give me what I
# need to run a bot against this deployment, as ME".
#
# WHAT IS ACTUALLY GENERATED (one artifact, two targets)
# -----------------------------------------------------
# Both kinds mint the SAME underlying artifact: a scoped, owner-recorded hugpy
# API key (api_keys.create_api_key) plus the config a bot process needs to use
# it. They differ only in the TARGET they are rendered for:
#
#   kind="discord-bot"        your OWN Discord bot application talking to this
#                             deployment — the key + the OpenAI-compatible /v1
#                             base URL, usable from discord.py / discord.js.
#   kind="hugpy-discord-bot"  the hugpy bot arm itself (abstract_hugpy_dev/bot,
#                             `hugpy bot`) — the key as HUGPY_API_KEY plus the
#                             .env / run lines that arm reads (see
#                             bot/hugpy_client._default_api_key and bot/config).
#
# WHAT IS DELIBERATELY *NOT* GENERATED
# ------------------------------------
#   * the shared DISCORD_TOKEN. It is the operator's bot-account credential for
#     the whole guild; it is never handed to a member, and it is not even
#     readable here (the Flask unit does not load the tree's .env — see the
#     docs' troubleshooting note). A member brings their own bot token from the
#     Discord Developer Portal; the response says so explicitly.
#   * a channel-scoped comms session (POST /discord/sessions). That mint stays
#     OPERATOR-ONLY on purpose: a session is bound to a channel_id the caller
#     names, and nothing in this deployment can prove a member controls the
#     channel behind an id. A member-mintable session would therefore let any
#     member read the transcript of — and send into — any channel the operator's
#     bot can see, which is an escalation, not a product. Members get their own
#     bot's credentials instead.
#   * any scope beyond the product plane. A non-operator's request is CLAMPED to
#     _MEMBER_BOT_SCOPES ("v1", "ml") and refused loudly if it asks for more —
#     never "full", never the fleet-enrolling "agent-register". Identical rule
#     and rationale to agent_routes._MEMBER_LINK_SCOPES.
#
# GATING. Not in operator_auth._SENSITIVE, for the same reason the install-link
# routes are not: this is a MEMBER product action, and the rule that has to be
# expressed is "operator OR the member who created this record", which a
# (methods, path) rule cannot say. It is enforced HERE by _require_member_bot,
# fails CLOSED for anonymous (401), and honors no testing waiver — this surface
# mints credentials.
#
# OWNERSHIP. The creating username is recorded on the minted key as
#   created_by = f"{_BOT_LINK_CREATED_BY}:{username}"
# and the kind as the key's `name`, so GET can return exactly the caller's own
# records and DELETE can refuse someone else's. An operator sees every bot-link
# key. A non-operator caller with no resolvable username is refused rather than
# handed an UNATTRIBUTED credential (same posture as install_link_create).

_BOT_LINK_CREATED_BY = "discord-bot-link"
_MEMBER_BOT_SCOPES = ("v1", "ml")
_BOT_LINK_KINDS = ("discord-bot", "hugpy-discord-bot")
# Human names, used in the response and matched by the /fleet help prompts.
_BOT_LINK_TITLES = {
    "discord-bot": "the discord bot",
    "hugpy-discord-bot": "the hugpy discord bot",
}
# View Channels | Send Messages | Embed Links | Attach Files | Read History.
# The minimum a relay bot needs; deliberately no moderation/manage bits.
_BOT_INVITE_PERMISSIONS = 1024 + 2048 + 16384 + 32768 + 65536   # 117760


from hugpy_server.app.auth_common import require_member_strict as _require_member_bot


def _bot_caller_is_operator() -> bool:
    try:
        from hugpy_server.app.operator_auth import operator_authenticated
        return bool(operator_authenticated())
    except Exception:  # noqa: BLE001 — fail closed (treat as non-operator)
        return False


from hugpy_server.app.auth_common import caller_username as _bot_caller_username


def _public_api_base() -> str:
    """This deployment's public API base. Delegates to
    ``agent_routes._install_public_base`` on purpose: ONE function decides what
    the public base is (HUGPY_PUBLIC_BASE if set, else the forwarded
    proto/host + the /api mount). A second copy here is exactly how the two
    would drift — the install-link mint already hit that bug once."""
    from hugpy_server.app.routes.agent_routes import _install_public_base
    return _install_public_base()


def _discord_client_id() -> str:
    """The hugpy bot application's PUBLIC OAuth client (application) id, or "".

    Read from the environment only — never derived from DISCORD_TOKEN, which
    this process does not load and must not touch. Unset today on this
    deployment, so the invite URL is omitted and the caller is told which
    variable to set instead of being handed a broken link."""
    for var in ("HUGPY_DISCORD_CLIENT_ID", "DISCORD_CLIENT_ID",
                "DISCORD_APPLICATION_ID"):
        value = (os.getenv(var) or "").strip()
        if value:
            return value
    return ""


def _bot_invite_url() -> str:
    client_id = _discord_client_id()
    if not client_id:
        return ""
    return ("https://discord.com/api/oauth2/authorize"
            f"?client_id={client_id}"
            f"&permissions={_BOT_INVITE_PERMISSIONS}"
            "&scope=bot%20applications.commands")


def _bot_link_config(kind: str, raw_key: str, api_base: str) -> dict:
    """The ready-to-paste config for a kind. Built HERE, never hand-assembled by
    a consumer — the same "one owner for this string shape" rule
    agent_routes._install_commands follows for the install one-liners."""
    if kind == "hugpy-discord-bot":
        env = "\n".join([
            "# .env for your own hugpy-discord bot instance",
            "# DISCORD_TOKEN is YOURS — create a bot application at",
            "# https://discord.com/developers/applications and paste its token.",
            "DISCORD_TOKEN=<your own discord bot token>",
            f"HUGPY_BASE_URL={api_base}",
            f"HUGPY_API_KEY={raw_key}",
        ])
        commands = {
            "install": "python3 -m venv ~/hugpy-bot/venv\n"
                       "~/hugpy-bot/venv/bin/pip install --upgrade abstract_hugpy",
            "run": "~/hugpy-bot/venv/bin/hugpy bot --env ./.env",
        }
    else:  # discord-bot
        env = "\n".join([
            "# .env for your own Discord bot talking to hugpy",
            "# DISCORD_TOKEN is YOURS — create a bot application at",
            "# https://discord.com/developers/applications and paste its token.",
            "DISCORD_TOKEN=<your own discord bot token>",
            f"HUGPY_BASE_URL={api_base}",
            f"HUGPY_API_KEY={raw_key}",
            f"OPENAI_BASE_URL={api_base.rsplit('/api', 1)[0]}/v1",
            f"OPENAI_API_KEY={raw_key}",
        ])
        commands = {
            "verify": f"curl -fsS -H 'X-API-Key: {raw_key}' {api_base}/health",
        }
    return {"env": env, "commands": commands}


class BotLinkRequest(BaseModel):
    kind: str
    label: str | None = None
    scopes: list[str] | None = None
    ttl_days: float | None = None


@discord_bp.route("/discord/bot-links", methods=["POST"])
def discord_bot_link_create():
    """MEMBER or OPERATOR: generate the credentials + config for running a
    Discord bot against this deployment, scoped to the caller's account.

    Body: ``{kind: "discord-bot"|"hugpy-discord-bot", label?, scopes?, ttl_days?}``.
    Returns 201 with the raw key ONCE (``api_key``) plus ``env`` / ``commands``
    / ``invite_url`` / ``notes``. The key is individually revocable
    (``DELETE /discord/bot-links/<key_id>``) and never carries operator rights:
    an api key of ANY scope is refused by operator_auth, which accepts only
    HUGPY_OPERATOR_TOKEN or a central session."""
    _require_member_bot()
    body = BotLinkRequest(**(request.get_json(silent=True) or {}))
    kind = (body.kind or "").strip()
    if kind not in _BOT_LINK_KINDS:
        abort(400, description=f"'kind' must be one of {list(_BOT_LINK_KINDS)}.")
    is_operator = _bot_caller_is_operator()
    owner = None if is_operator else _bot_caller_username()
    if not is_operator and not owner:
        # Passed member_authenticated() but no username resolved — refuse to
        # mint an UNATTRIBUTED credential from a non-operator caller.
        abort(401, description="Authentication required for this route.")
    label = (body.label or "").strip() or _BOT_LINK_TITLES[kind]
    scopes = body.scopes
    if not is_operator and scopes:
        bad = [s for s in scopes if str(s) not in _MEMBER_BOT_SCOPES]
        if bad:
            # Refused loudly, never silently downgraded: a caller must never
            # believe it holds a scope it was not given.
            abort(403, description=(
                f"scope(s) {bad} are operator-only; a member bot key may "
                f"request {list(_MEMBER_BOT_SCOPES)}."))
    if not scopes:
        scopes = ["v1"]
    expires_at = None
    if body.ttl_days:
        try:
            expires_at = time.time() + float(body.ttl_days) * 86400.0
        except (TypeError, ValueError):
            abort(400, description="'ttl_days' must be a number.")

    from hugpy_server.app.functions.imports.utils.api_keys import create_api_key
    try:
        key = create_api_key(
            name=kind,
            label=label,
            scopes=scopes,
            created_by=f"{_BOT_LINK_CREATED_BY}:{owner or 'operator'}",
            expires_at=expires_at,
        )
    except ValueError as exc:
        abort(400, description=str(exc))

    api_base = _public_api_base()
    raw_key = key.pop("key")          # shown exactly once, by this response
    config = _bot_link_config(kind, raw_key, api_base)
    invite_url = _bot_invite_url()
    notes = [
        "This key is scoped to your account and is shown once. Store it in the "
        "bot's .env; you can revoke it any time.",
        "The DISCORD_TOKEN is not issued here. Create your own bot application "
        "at discord.com/developers/applications — hugpy never hands out the "
        "shared bot token.",
    ]
    if not invite_url:
        notes.append(
            "No OAuth invite link is available: this deployment has no Discord "
            "application id configured (set HUGPY_DISCORD_CLIENT_ID on the API "
            "service). Invite your own bot from its own application page in "
            "the Discord Developer Portal instead.")
    logger.info("bot link minted: kind=%s key=%s owner=%s scopes=%s",
                kind, key.get("id"), owner or "operator", scopes)
    return jsonify({
        "kind": kind,
        "title": _BOT_LINK_TITLES[kind],
        "key_id": key.get("id"),
        "label": label,
        "owner": owner,
        "scopes": key.get("scopes"),
        "prefix": key.get("prefix"),
        "created_at": key.get("created_at"),
        "expires_at": key.get("expires_at"),
        "api_key": raw_key,
        "api_base": api_base,
        "invite_url": invite_url,
        "client_id_configured": bool(invite_url),
        "env": config["env"],
        "commands": config["commands"],
        "notes": notes,
    }), 201


def _bot_link_rows(username, is_operator: bool) -> list:
    """The bot-link keys visible to this caller, newest first, WITHOUT any
    secret (list_api_keys never returns a hash, and the raw key exists only in
    the mint response)."""
    from hugpy_server.app.functions.imports.utils.api_keys import list_api_keys
    mine = f"{_BOT_LINK_CREATED_BY}:{username}" if username else None
    rows = []
    for rec in list_api_keys():
        created_by = rec.get("created_by") or ""
        if not created_by.startswith(f"{_BOT_LINK_CREATED_BY}:"):
            continue
        if not is_operator and created_by != mine:
            continue
        kind = rec.get("name") or ""
        rows.append({
            "key_id": rec.get("id"),
            "kind": kind,
            "title": _BOT_LINK_TITLES.get(kind, kind),
            "label": rec.get("label") or "",
            "owner": created_by.split(":", 1)[1] or None,
            "scopes": rec.get("scopes") or [],
            "prefix": rec.get("prefix"),
            "created_at": rec.get("created_at"),
            "expires_at": rec.get("expires_at"),
            "status": ("expired" if rec.get("expired")
                       else "disabled" if rec.get("disabled") else "active"),
        })
    return rows


@discord_bp.route("/discord/bot-links", methods=["GET"])
def discord_bot_link_list():
    """MEMBER or OPERATOR: the bot keys generated from this surface. A member
    sees ONLY their own; an operator sees every bot-link key. Raw keys are never
    returned — they exist exactly once, in the mint response."""
    _require_member_bot()
    is_operator = _bot_caller_is_operator()
    username = _bot_caller_username()
    if not is_operator and not username:
        abort(401, description="Authentication required for this route.")
    return jsonify({"links": _bot_link_rows(username, is_operator)})


@discord_bp.route("/discord/bot-links/<key_id>", methods=["DELETE"])
def discord_bot_link_revoke(key_id):
    """OPERATOR (any) or the MEMBER who generated it: revoke the bot key. A
    member revoking someone else's key gets 403; an unknown id is 404 either
    way. Only keys minted from THIS surface are addressable here — the general
    console key ledger stays operator-only at /keys."""
    _require_member_bot()
    is_operator = _bot_caller_is_operator()
    username = _bot_caller_username()
    if not is_operator and not username:
        abort(401, description="Authentication required for this route.")
    rows = _bot_link_rows(username, is_operator=True)
    match = next((r for r in rows if r.get("key_id") == key_id), None)
    if not match:
        abort(404, description="Unknown bot link.")
    if not is_operator and match.get("owner") != username:
        abort(403, description="This bot link belongs to another account.")
    from hugpy_server.app.functions.imports.utils.api_keys import revoke_api_key
    if not revoke_api_key(key_id):
        abort(404, description="Unknown bot link.")
    logger.info("bot link revoked: key=%s by=%s", key_id, username or "operator")
    return jsonify({"ok": True, "key_id": key_id})


@discord_bp.route("/discord/session/<token>/send", methods=["POST"])
def discord_session_send(token):
    s = _session_or_404(token)
    body = request.get_json(silent=True) or {}
    content = (body.get("content") or "").strip()
    if not content:
        abort(400, description="content is required.")
    if len(content) > _SESSION_MSG_LIMIT:
        abort(413, description=f"content exceeds {_SESSION_MSG_LIMIT} chars — split it.")
    options = _validate_options(body.get("options"))
    author = (body.get("author") or s.get("author") or s.get("label") or "session").strip()
    bridge = bridge_for_channel(s["channel_id"])
    msg = None
    if bridge:  # record in the transcript so other session holders see it
        kwargs = {"direction": "out", "source": "session",
                  "content": content, "author": author}
        if options is not None:
            # transcript-only nicety; the store may drop unknown kwargs — harmless.
            kwargs["options"] = options
        msg = append_bridge_message(bridge["id"], **kwargs)
    enqueue_outbound(content=content, channel_id=s["channel_id"], options=options)
    return jsonify({"ok": True, "message": msg}), 201
