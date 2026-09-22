"""Eviction telemetry — central ingest, history, and the live SSE stream.

WHY (operator directive, 2026-07-28): the operator must be able to VISUALIZE the
eviction process in REAL TIME. Workers evict; central is where a human is
looking. This module is the seam between the two.

    POST /llm/evictions/ingest    worker batches in (enrollment-gated, same
                                  credential as register/heartbeat)
    GET  /llm/evictions           bounded history for a page load / backfill
    GET  /llm/evictions/stream    SSE: replay the last ~100, then live

WHY A SQLITE POLL AND NOT AN IN-PROCESS PUBSUB
    Dev central runs gunicorn with 3 workers x 8 threads. An ingest POST is
    load-balanced onto whichever process the OS picked, while the SSE client is
    held open on another. An in-process queue would show that client a third of
    the fleet's events and look, convincingly, like a flaky relay. The shared
    comms sqlite file is the rendezvous every process already has, and its
    autoincrement rowid is a natural stream cursor: a reader polls
    "rowid > cursor" every ~0.5s and gets exactly what it hasn't seen, no matter
    who wrote it. Same file, same pragmas, same EMFILE hardening as the comms
    mirror — see comms/evictions.py and comms/shared.py.

STRICTLY OBSERVATIONAL. Nothing in this module can gate, delay or veto an
eviction. Ingest ALWAYS answers 200 with a count once the caller is
authenticated: a store fault costs the console history, and must never make a
worker think its telemetry channel is a problem worth retrying into.
"""
import json
import time

from flask import Response, jsonify, request, stream_with_context

from abstract_flask import get_bp
from hugpy_fleet.central import evictions as evictions_mod

eviction_bp, logger = get_bp("eviction_bp", __name__)

# Replay depth for a fresh SSE subscriber — enough that a card the operator just
# saw scroll by is still there when they open the panel, small enough that
# attaching is instant.
REPLAY_LIMIT = 100
# Cursor poll interval. Sub-second is what "real time" means for a human
# watching; the query is an indexed "rowid > ?" against a WAL-mode file.
POLL_S = 0.5
# Proxies (nginx in front of dev) drop an idle upstream response. A comment line
# is a no-op to EventSource and keeps the pipe warm.
HEARTBEAT_S = 15.0
# Safety cap — a browser tab left open for days must not pin a gunicorn thread
# forever. The client's EventSource reconnects automatically on close, so this
# is invisible in the UI.
STREAM_MAX_S = 3600.0
MAX_BATCH = 500


def _worker_authorized() -> bool:
    """Ingest gate: the SAME enrollment credential register/heartbeat use.

    Deliberately not a new scheme. A worker already holds exactly one central
    credential; making it hold a second one for telemetry would be a second
    thing to rotate and get wrong. Revocation bites here the instant it bites
    the heartbeat (a revoked worker's telemetry stops with it). If the gate
    cannot be imported we fail CLOSED — an unauthenticated write endpoint is
    not an acceptable degradation.

    STRICTER than register/heartbeat on purpose (keeper, 2026-07-29): those
    keep the gradual-rollout allowance (no token -> allow while
    HUGPY_WORKER_ENROLL_REQUIRED is off), but this central's public origin
    proxies straight to Flask, so a tokenless WRITE endpoint here is writable
    by the whole internet — a probe proved it. Ingest therefore requires a
    PRESENT, VALID token always. The fleet was enrolled with per-box tokens
    the same day, so no live worker regresses."""
    try:
        from hugpy_server.app.routes.worker_routes import _bearer_token
        from hugpy_fleet.central.enrollment_tokens import verify_enrollment_token
        tok = _bearer_token()
        return tok is not None and bool(verify_enrollment_token(tok))
    except Exception:  # noqa: BLE001
        logger.warning("eviction ingest: enrollment gate unavailable — refusing")
        return False


def _operator_or_worker() -> bool:
    """Read gate. The console reads these through an operator session; a worker
    may read its own stream for debugging. Fails OPEN to the enrollment rule
    only — never wider than the heartbeat endpoint."""
    try:
        from hugpy_server.app.operator_auth import operator_authenticated
        if operator_authenticated():
            return True
    except Exception:  # noqa: BLE001
        pass
    # A valid API key may READ telemetry (operator ask 2026-07-29: bench
    # clients associate eviction events with their own call windows). Reads
    # only — ingest keeps the strict worker-enrollment gate above; a key that
    # can already drive generations learns nothing new from seeing evictions.
    try:
        from hugpy_server.app.routes.worker_routes import _bearer_token
        from hugpy_server.app.functions.imports.utils.api_keys import key_id_for_token
        tok = _bearer_token()
        if tok and key_id_for_token(tok):
            return True
    except Exception:  # noqa: BLE001
        pass
    return _worker_authorized()


@eviction_bp.route("/llm/evictions/ingest", methods=["POST"])
def evictions_ingest():
    """Accept a batch of eviction-telemetry events from a worker.

    Body: ``{"events": [ {...}, ... ]}``. Events are stored verbatim (the full
    dict rides in the row's JSON body), so a worker running a newer release can
    add fields to a stage without a central migration — forward compatibility
    matters here because worker-side emitters reach the fleet on a release
    cadence central does not control.

    Always 200 on an authenticated call, even when the store is down: the count
    tells the truth, and a worker must never treat telemetry as a failure worth
    escalating."""
    if not _worker_authorized():
        return jsonify({"error": "Worker enrollment token invalid or required."}), 401
    body = request.get_json(silent=True) or {}
    events = body.get("events")
    if not isinstance(events, list):
        return jsonify({"error": "expected {'events': [...]}"}), 400
    if len(events) > MAX_BATCH:
        # Truncate rather than reject: a relay that fell behind should still
        # deliver what it can. Newest events matter most, so keep the tail.
        events = events[-MAX_BATCH:]
    clean = [e for e in events if isinstance(e, dict) and e.get("stage")]
    stored = 0
    try:
        stored = evictions_mod.get_store().append(clean)
    except Exception as exc:  # noqa: BLE001 — never surface a store fault to a worker
        logger.warning("eviction ingest: store append failed (%s) — %d event(s) "
                       "dropped; the fleet is unaffected", exc, len(clean))
    return jsonify({"ok": True, "received": len(clean), "stored": stored})


@eviction_bp.route("/llm/evictions", methods=["GET"])
def evictions_recent():
    """Bounded history — the console's page-load backfill.

    ``limit`` (default 200, max 2000) newest events, oldest-first for direct
    rendering. ``since`` is an epoch-seconds floor; ``after_id`` is the stream
    cursor form."""
    if not _operator_or_worker():
        return jsonify({"error": "not authorized"}), 401
    try:
        limit = int(request.args.get("limit") or 200)
    except (TypeError, ValueError):
        limit = 200
    limit = max(1, min(limit, 2000))
    since = request.args.get("since")
    after = request.args.get("after_id")
    try:
        since_ts = float(since) if since not in (None, "") else None
    except (TypeError, ValueError):
        since_ts = None
    try:
        after_id = int(after) if after not in (None, "") else None
    except (TypeError, ValueError):
        after_id = None
    store = evictions_mod.get_store()
    events = store.recent(limit=limit, since_ts=since_ts, after_id=after_id)
    return jsonify({"events": events, "count": len(events),
                    "cursor": (events[-1].get("_id") if events else after_id or 0)})


@eviction_bp.route("/llm/evictions/stream", methods=["GET"])
def evictions_stream():
    """SSE: the last ~100 events, then live.

    Tails the shared sqlite table by rowid, which is what makes this correct
    across gunicorn workers (see the module docstring). Emits a `: heartbeat`
    comment when idle so a proxy does not reap the connection, and returns after
    STREAM_MAX_S so a forgotten tab cannot pin a thread — EventSource reconnects
    on its own, and the reconnect replays from the cursor, so the operator sees
    no gap.

    k59: also takes a slot from the fast-read reserve. STREAM_MAX_S bounds how
    long ONE stream holds a thread; it does nothing about how MANY do. Past the
    reserve's cap this returns an honest 503 + Retry-After rather than taking
    the last thread and making an unrelated endpoint time out instead."""
    from hugpy_fleet.central import pool_guard

    if not _operator_or_worker():
        return jsonify({"error": "not authorized"}), 401
    try:
        replay = int(request.args.get("replay") or REPLAY_LIMIT)
    except (TypeError, ValueError):
        replay = REPLAY_LIMIT
    replay = max(0, min(replay, 1000))

    def sse(payload: dict) -> bytes:
        return f"data: {json.dumps(payload, default=str)}\n\n".encode("utf-8")

    try:
        slot = pool_guard.stream_slot()
        slot.__enter__()
    except pool_guard.StreamCapacityExceeded as exc:
        return (jsonify(exc.as_error()), 503,
                {"Retry-After": str(exc.retry_after)})

    def generate():
        store = evictions_mod.get_store()
        cursor = 0
        try:
            backlog = store.recent(limit=replay) if replay else []
        except Exception:  # noqa: BLE001 — an unreadable store still streams live
            backlog = []
        for ev in backlog:
            cursor = max(cursor, int(ev.get("_id") or 0))
            yield sse(ev)
        if not cursor:
            # Empty (or unreadable) history: start from the current head so we
            # stream only what happens from now on, rather than re-sending a
            # backlog the client just declined.
            try:
                cursor = store.max_id()
            except Exception:  # noqa: BLE001
                cursor = 0
        # Tell the client it is attached even when nothing has ever been
        # evicted, so the UI can show "live" instead of "connecting" forever.
        yield sse({"stage": "stream.ready", "ts": time.time(), "cursor": cursor})
        last_beat = time.time()
        deadline = time.time() + STREAM_MAX_S
        while time.time() < deadline:
            try:
                fresh = store.recent(limit=200, after_id=cursor)
            except Exception:  # noqa: BLE001 — a transient store fault is not
                fresh = []      # a reason to drop the operator's stream
            for ev in fresh:
                cursor = max(cursor, int(ev.get("_id") or 0))
                yield sse(ev)
            if fresh:
                last_beat = time.time()
            elif time.time() - last_beat >= HEARTBEAT_S:
                last_beat = time.time()
                yield b": heartbeat\n\n"
            time.sleep(POLL_S)

    def guarded():
        """Hold the reserve slot for the generator's whole life. Flask closes
        the generator when the client disconnects (GeneratorExit), so the
        finally runs on a walked-away tab too — a slot that leaked on
        disconnect would shrink the reserve one abandoned tab at a time."""
        try:
            yield from generate()
        finally:
            slot.__exit__(None, None, None)

    return Response(
        stream_with_context(guarded()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
        direct_passthrough=True,
    )
