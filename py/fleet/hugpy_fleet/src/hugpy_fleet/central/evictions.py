"""Serve-pipeline telemetry — every stage of serving a model, as an event.

Named for evictions because evictions were the first thing it carried; it now
carries the WHOLE path a request walks — provision, resolve, load, and the
eviction pass that may happen in the middle. The module/route/stage names stay
as they are: they are a wire contract with the worker fleet and the console,
and workers only pick up a rename on a release.


WHY THIS EXISTS (operator directive, 2026-07-28): "the operator must be able to
VISUALIZE the eviction process in REAL TIME". Before this module the only trace
of an eviction was scattered ``logger.info`` lines on whichever box did the
evicting — unreadable across a fleet, unstreamable to a console, and silent
about the question the doctrine actually cares about: *why was that resident
protected?* This module turns the eviction path into a stream of typed events
and carries them to central so a human can watch a headroom pass happen.

OBSERVATION ONLY — the load path's contract is unchanged.
    Nothing here may gate, delay, or veto an eviction. Every public entry point
    is wrapped so that ANY failure (a full ring, a dead relay, a broken sink, a
    sqlite fault) is swallowed and the eviction proceeds exactly as it would
    have without telemetry. The eviction path calls into telemetry; telemetry
    never calls back into the eviction path. If this module were deleted the
    behavior of ``ensure_headroom_for_load`` would be byte-identical.

FLASK-FREE by construction. ``managers.dispatch.dispatch`` is shared code that
runs on bare central, on a worker, and in tests; it may not grow a web
dependency. Only stdlib is imported at module scope (``requests`` is imported
lazily, inside the relay thread, and only on a worker that configured one).

THE SHAPE OF THE STREAM
    One ``run_id`` (uuid4 hex) correlates every event of ONE serve attempt —
    including the headroom pass inside it — so the console can render the
    attempt as a single card:

        provision.start  {model_key, source, dest_path}
        provision.fail   {model_key, source, error_class, errno_name, detail,
                          disk_free_bytes, disk_total_bytes, human}
        provision.done   {model_key, source, bytes, duration_ms}
        resolve.fail     {model_key, resolved_path, reason}
        load.start       {model_key, engine}
        load.done        {model_key, engine, duration_ms}
        load.fail        {model_key, engine, error}

        headroom.start   {incoming_model, need_bytes?, trigger}
        fit.fail         {incoming_model, deficit_bytes?, free_bytes?}
        candidate.skip   {model_key, reason}          <- the protection story
        evict.start      {model_key, tier}
        evict.done       {model_key, tier, freed_bytes?, duration_ms}
        evict.fail       {model_key, tier, error}
        reclaim.done     {}
        makeroom.verdict {action, reason, evicted[]}
        headroom.done    {incoming_model, evicted[], outcome}

    ``tier`` names HOW the weights were held, not who held them:
    "in-process" | "slot-child" | "hot_cache" | "reservation" | "comfy".

    Disk eviction (the hot-NVMe LRU cache) rides the same stream tagged
    tier="hot_cache" — it is part of "the entire process", and an operator
    watching a model go cold wants to see the disk copy reaped too.

SEQ is per-process and monotonic. It is NOT a global order across the fleet:
two workers both start at 1. Correlation is (worker_id, seq); ordering for
display is ``ts``. Central assigns its own rowid on ingest, and that rowid —
not seq — is the SSE cursor.

TRANSPORT
    emit() -> local ring (bounded, drop-oldest) -> registered sinks.
    On a WORKER a relay sink batches into a background flusher that POSTs to
    central ``/llm/evictions/ingest`` (<=1s latency: flush every 0.5s or at 20
    events). On CENTRAL the sink is the sqlite store directly. Drop-oldest on a
    full ring is deliberate: a wedged relay must cost nothing but old events.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sqlite3
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable, Deque, Iterable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage vocabulary. The keeper owns nomenclature (doctrine): these strings are
# the wire contract shared by dispatch, the worker agent, the central store and
# the console. Add stages here rather than inventing them at a call site.
# ---------------------------------------------------------------------------
#
# TWO FAMILIES OF STAGE SHARE THIS STREAM.
#
#   the SERVE pipeline   provision.* -> resolve.* -> load.*
#   the EVICTION pass    headroom.* / fit.* / candidate.* / evict.* / ...
#
# They were one stream from the moment the 2026-07-28 ENOSPC incident showed
# what a half-stream costs. computron's disk was 100% full; provisioning died
# with Errno 28 before a single byte of the model existed, so the eviction feed
# said — accurately — "trigger load / no candidates walked", and the operator
# read that as "eviction is fine" and went looking in the wrong place. The
# feed was not wrong, it was SHORT: it began after the point of failure. A
# request that dies during provisioning must say so in the same card, under the
# same run_id, as one that dies during eviction, because the operator's actual
# question is never "how did eviction go" — it is "where did my request die and
# why". Everything upstream of eviction now answers that in-band.
STAGE_PROVISION_START = "provision.start"
STAGE_PROVISION_FAIL = "provision.fail"
STAGE_PROVISION_DONE = "provision.done"
STAGE_RESOLVE_FAIL = "resolve.fail"
STAGE_LOAD_START = "load.start"
STAGE_LOAD_DONE = "load.done"
STAGE_LOAD_FAIL = "load.fail"

SERVE_STAGES = (
    STAGE_PROVISION_START, STAGE_PROVISION_FAIL, STAGE_PROVISION_DONE,
    STAGE_RESOLVE_FAIL,
    STAGE_LOAD_START, STAGE_LOAD_DONE, STAGE_LOAD_FAIL,
)

# Where the weights were coming FROM on a provision.* event. Ordered as the
# provisioner actually tries them.
SOURCE_LOCAL = "local"
SOURCE_CENTRAL_TRANSFER = "central-transfer"
SOURCE_ARCHIVE = "archive"
SOURCE_HF = "hf"
SOURCES = (SOURCE_LOCAL, SOURCE_CENTRAL_TRANSFER, SOURCE_ARCHIVE, SOURCE_HF)

STAGE_HEADROOM_START = "headroom.start"
STAGE_FIT_FAIL = "fit.fail"
STAGE_CANDIDATE_SKIP = "candidate.skip"
STAGE_EVICT_START = "evict.start"
STAGE_EVICT_DONE = "evict.done"
STAGE_EVICT_FAIL = "evict.fail"
STAGE_RECLAIM_DONE = "reclaim.done"
STAGE_MAKEROOM_VERDICT = "makeroom.verdict"
STAGE_HEADROOM_DONE = "headroom.done"

EVICTION_STAGES = (
    STAGE_HEADROOM_START, STAGE_FIT_FAIL, STAGE_CANDIDATE_SKIP,
    STAGE_EVICT_START, STAGE_EVICT_DONE, STAGE_EVICT_FAIL,
    STAGE_RECLAIM_DONE, STAGE_MAKEROOM_VERDICT, STAGE_HEADROOM_DONE,
)

# A THIRD FAMILY: MODEL GROUPS (operator ruling 2026-07-28).
#
# A group picks WHICH ITERATION of a base model serves a request — the
# transformers repo or the GGUF, and which rung of the GGUF's quant ladder. That
# choice happens upstream of provisioning, so without these stages the feed's
# first line would be ``provision.start`` for a model key the operator never
# typed, with nothing anywhere saying why THAT one. Same lesson as the ENOSPC
# incident that merged the serve and eviction families: the feed must not begin
# after the decision the operator is asking about.
#
# ``member.select`` is emitted once per served request when a group was
# consulted; ``member.skip`` once per iteration that lost, each carrying the
# one-line reason the pipeline recorded ("transformers member excluded: fits
# only as 4-bit (quality)"). Both are OBSERVATION ONLY, like everything here.
#
# ``headroom.start`` additionally grows an optional ``group`` field
# {group_key, tick} when a group's ticked standard is what DEMANDED the pass —
# which is how an operator tells "this eviction happened because the speed tick
# refused to spill" from an ordinary contention eviction. A field, not a stage:
# the headroom pass is the same pass either way.
STAGE_MEMBER_SELECT = "member.select"
STAGE_MEMBER_SKIP = "member.skip"

GROUP_STAGES = (STAGE_MEMBER_SELECT, STAGE_MEMBER_SKIP)

# The full vocabulary, in the order a request travels: a group picks the member,
# then that member is provisioned, resolved and loaded, with an eviction pass
# possible in the middle. Kept as ``STAGES`` for anything already importing it.
STAGES = GROUP_STAGES + SERVE_STAGES + EVICTION_STAGES

# Residency tiers — HOW the evicted weights were held.
TIER_IN_PROCESS = "in-process"
TIER_SLOT_CHILD = "slot-child"
TIER_HOT_CACHE = "hot_cache"
TIER_RESERVATION = "reservation"
TIER_COMFY = "comfy"

# Ring size. ~2000 events is minutes of a busy box and a few hundred KB; it is
# a display buffer, not a log — the journal line emitted alongside every event
# is the durable record.
RING_MAX = 2000

_RING: Deque[dict] = deque(maxlen=RING_MAX)
_RING_LOCK = threading.Lock()
_SEQ = 0
_SEQ_LOCK = threading.Lock()
_SINKS: list[Callable[[dict], None]] = []
_SINKS_LOCK = threading.Lock()
_WORKER_ID: Optional[str] = None


def new_run_id() -> str:
    """A correlation id for ONE headroom pass. Every event of that pass carries
    it, which is what lets the console render a pass as a single card."""
    return uuid.uuid4().hex


# The ambient run_id, per THREAD.
#
# A headroom pass is not one function: dispatch's yield loop calls into the
# worker's cross-tier make-room hook, which calls into the slot pool, which
# calls the /ops/evict verb — all on the SAME thread, all part of one pass, none
# of them able to receive a run_id parameter without threading an argument
# through a dozen signatures that have nothing to do with telemetry. A
# thread-local ambient id is the small primitive that correlates them: dispatch
# opens the scope, everything underneath inherits it for free, and a call
# arriving with no scope open simply emits an uncorrelated event.
_LOCAL = threading.local()


def current_run_id() -> str:
    return getattr(_LOCAL, "run_id", "") or ""


class run_scope:
    """Context manager binding an ambient run_id for this thread.

    Re-entrant by save/restore rather than by refusing to nest, so a nested
    pass (a sweep that triggers a load) restores its parent's id on exit
    instead of clearing it."""

    def __init__(self, run_id: Optional[str] = None) -> None:
        self.run_id = run_id or new_run_id()
        self._prev = ""

    def __enter__(self) -> str:
        self._prev = current_run_id()
        _LOCAL.run_id = self.run_id
        return self.run_id

    def __exit__(self, *exc) -> None:
        _LOCAL.run_id = self._prev
        return None


class serve_scope(run_scope):
    """Open a correlation scope AT REQUEST-SERVE ENTRY, if one isn't open.

    The eviction pass opens its own scope because it is self-contained. The
    serve pipeline is not: provisioning, resolution and load are three
    different modules that a request passes through in sequence, and a scope
    opened by the eviction pass in the middle of that sequence would put the
    eviction card in a different card from the provisioning that preceded it.
    So serve entry opens the scope and everything downstream — including
    ``ensure_headroom_for_load`` — inherits it.

    JOINS rather than nests when a scope is already open (unlike ``run_scope``,
    which always mints a fresh id). A serve that is already inside somebody's
    pass is part of that pass; minting a new id there is exactly the split this
    class exists to prevent."""

    def __init__(self, run_id: Optional[str] = None) -> None:
        existing = current_run_id()
        super().__init__(run_id or existing or new_run_id())


class group_scope:
    """Ambient "this pass was demanded by a model group", per THREAD.

    Same primitive, same reason, as ``run_scope``: the headroom pass is not one
    function, and threading a group_key through ``ensure_headroom_for_load`` ->
    make-room hook -> slot pool -> /ops/evict would put a routing concept into a
    dozen signatures that have nothing to do with routing. The member selector
    opens this scope around the serve it provoked; ``headroom.start`` picks the
    field up for free, and a pass opened with no scope simply has no ``group``.

    Save/restore rather than refuse-to-nest, exactly like ``run_scope``.

    PHASE 1 SCOPE: in-process only. Central picks the member and routes; the
    WORKER's headroom pass is a different process and will not see this until
    phase 2 puts the demand on the wire (see MODEL-GROUPS-SPEC §8). Nothing
    depends on it being there — the field is optional and its absence renders as
    an ordinary contention eviction, which is what it is from the worker's side.
    """

    def __init__(self, group_key: str, tick: Optional[str] = None) -> None:
        self.group = {"group_key": str(group_key), "tick": tick} if group_key else None
        self._prev: Any = None

    def __enter__(self):
        self._prev = getattr(_LOCAL, "group", None)
        _LOCAL.group = self.group
        return self.group

    def __exit__(self, *exc) -> None:
        _LOCAL.group = self._prev
        return None


def current_group() -> Optional[dict]:
    """The ambient group demand, or None. Total — never raises."""
    try:
        g = getattr(_LOCAL, "group", None)
        return dict(g) if isinstance(g, dict) else None
    except Exception:  # noqa: BLE001 — telemetry never raises
        return None


# Disk-failure helpers are storage's (one owner): errno_name, disk_stats and
# describe_disk_error live in hugpy_storage.model_presence and are re-exported
# here for the telemetry callers that always imported them from this module.
from hugpy_storage.model_presence import (  # noqa: E402
    _fmt_bytes,
    describe_disk_error,
    disk_stats,
    errno_name,
)



def _never_raises(fn):
    """Belt to the emitter's braces.

    ``emit_eviction_event`` is already total, but these convenience wrappers do
    work of their own before they call it — stringifying an exception, reading
    statvfs — and the call sites are on the serve path, where a telemetry
    traceback would convert an honest ENOSPC into a mystery. Nothing above this
    line is allowed to fail louder than a debug log."""
    import functools

    @functools.wraps(fn)
    def _wrapped(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception:  # noqa: BLE001
            logger.debug("serve telemetry: %s failed", fn.__name__, exc_info=True)
            return None
    return _wrapped


@_never_raises
def emit_member_select(group_key: str, model_key: str,
                       reason: Optional[str] = None,
                       as_: Optional[str] = None,
                       ticks: Optional[dict] = None, **extra: Any):
    """``member.select`` — THIS iteration of the group is serving, and why.

    ``as_`` is the chosen ladder rung ("q4_k_m") for a GGUF member, absent for
    anything else. Trailing underscore because ``as`` is a keyword; it goes on
    the wire as ``as``, which is what the console renders."""
    return emit_eviction_event(STAGE_MEMBER_SELECT, group_key=group_key,
                               model_key=model_key, reason=reason,
                               **{"as": as_}, ticks=ticks, **extra)


@_never_raises
def emit_member_skip(group_key: str, model_key: str, reason: str, **extra: Any):
    """``member.skip`` — this iteration lost, and the one-line reason it lost.

    The reason is the operator-facing sentence, not a code: "transformers member
    excluded: fits only as 4-bit (quality)". A skip with no reason is worse than
    no skip at all, so callers always pass one."""
    return emit_eviction_event(STAGE_MEMBER_SKIP, group_key=group_key,
                               model_key=model_key, reason=reason, **extra)


@_never_raises
def emit_provision_start(model_key: str, source: str,
                         dest_path: Optional[str] = None, **extra: Any):
    """``provision.start`` — we are about to try ``source`` for ``model_key``."""
    return emit_eviction_event(STAGE_PROVISION_START, model_key=model_key,
                               source=source, dest_path=dest_path, **extra)


@_never_raises
def emit_provision_fail(model_key: str, source: str,
                        exc: Optional[BaseException] = None,
                        dest_path: Optional[str] = None,
                        detail: Optional[str] = None, **extra: Any):
    """``provision.fail`` — ``source`` did not deliver, and WHY.

    On any errno-bearing OSError this attaches the destination filesystem's
    free/total. That is the whole point of the stage: the operator should never
    again have to ssh into a worker to learn that its disk was full."""
    fields: dict = {"model_key": model_key, "source": source}
    if exc is not None:
        fields["error_class"] = type(exc).__name__
        en = errno_name(exc)
        if en:
            fields["errno_name"] = en
        fields.setdefault("detail", detail or str(exc) or type(exc).__name__)
    elif detail:
        fields["detail"] = detail
    if dest_path:
        fields["dest_path"] = dest_path
    # Disk facts whenever the error smells like the filesystem, or the caller
    # asked for them by naming a destination.
    if dest_path and (exc is None or errno_name(exc)):
        st = disk_stats(dest_path)
        fields.update(st)
        if exc is not None:
            human = describe_disk_error(exc, dest_path, st)
            if human:
                fields["human"] = human
    fields.update(extra)
    return emit_eviction_event(STAGE_PROVISION_FAIL, **fields)


@_never_raises
def emit_provision_done(model_key: str, source: str,
                        bytes_: Optional[int] = None,
                        duration_ms: Optional[int] = None, **extra: Any):
    """``provision.done`` — the weights are on this box's disk."""
    return emit_eviction_event(STAGE_PROVISION_DONE, model_key=model_key,
                               source=source, bytes=bytes_,
                               duration_ms=duration_ms, **extra)


@_never_raises
def emit_resolve_fail(model_key: str, resolved_path: Optional[str],
                      reason: str, **extra: Any):
    """``resolve.fail`` — provisioning claimed success but there is no usable
    GGUF at the path we would hand to llama.cpp. Spawning anyway is the SIGILL
    the refusal exists to prevent; this stage makes the refusal visible."""
    return emit_eviction_event(STAGE_RESOLVE_FAIL, model_key=model_key,
                               resolved_path=resolved_path, reason=reason,
                               **extra)


@_never_raises
def emit_load_start(model_key: str, engine: Optional[str] = None, **extra: Any):
    return emit_eviction_event(STAGE_LOAD_START, model_key=model_key,
                               engine=engine, **extra)


@_never_raises
def emit_load_done(model_key: str, engine: Optional[str] = None,
                   duration_ms: Optional[int] = None, **extra: Any):
    return emit_eviction_event(STAGE_LOAD_DONE, model_key=model_key,
                               engine=engine, duration_ms=duration_ms, **extra)


@_never_raises
def emit_load_fail(model_key: str, engine: Optional[str] = None,
                   exc: Optional[BaseException] = None,
                   error: Optional[str] = None, **extra: Any):
    if error is None and exc is not None:
        error = f"{type(exc).__name__}: {exc}"
    return emit_eviction_event(STAGE_LOAD_FAIL, model_key=model_key,
                               engine=engine, error=error, **extra)


def set_worker_id(worker_id: Optional[str]) -> None:
    """Stamp events with the worker's REGISTERED id (the one central knows it
    by), rather than the hostname fallback. Called by the worker agent once its
    registration has settled; harmless to call again."""
    global _WORKER_ID
    if worker_id:
        _WORKER_ID = str(worker_id)


def worker_id() -> str:
    """The id stamped on every event: the registered worker id when one has
    been set, else HUGPY_WORKER_NAME, else the hostname. Never empty — an
    unattributable event is worse than a coarsely attributed one."""
    if _WORKER_ID:
        return _WORKER_ID
    env = (os.environ.get("HUGPY_WORKER_NAME") or "").strip()
    if env:
        return env
    try:
        return socket.gethostname() or "unknown"
    except Exception:  # noqa: BLE001 — telemetry never raises
        return "unknown"


def register_sink(fn: Callable[[dict], None]) -> None:
    """Attach a delivery sink: ``fn(event_dict)``. A sink MUST NOT block and
    MUST NOT raise (it is called on the eviction path); it is wrapped anyway.
    Central registers the sqlite store; a worker registers the relay buffer."""
    with _SINKS_LOCK:
        if fn not in _SINKS:
            _SINKS.append(fn)


def clear_sinks() -> None:
    """Detach every sink (tests; process teardown)."""
    with _SINKS_LOCK:
        _SINKS.clear()


def _next_seq() -> int:
    global _SEQ
    with _SEQ_LOCK:
        _SEQ += 1
        return _SEQ


def _kv_line(ev: dict) -> str:
    """The journal rendering. "The entire process should be logged" means the
    structured stream AND the journal — a box whose relay is down must still
    tell the whole story to ``journalctl``. One line, key=value, stable order
    (stage first), so it greps."""
    lead = ("run_id", "worker_id", "incoming_model", "model_key", "source",
            "tier", "engine", "errno_name", "reason", "action", "outcome")
    parts = [f"stage={ev.get('stage')}"]
    for k in lead:
        v = ev.get(k)
        if v not in (None, ""):
            parts.append(f"{k}={v}")
    for k in sorted(ev):
        if k in lead or k in ("stage", "ts", "seq"):
            continue
        v = ev[k]
        if v in (None, ""):
            continue
        if isinstance(v, (list, tuple, dict)):
            v = json.dumps(v, separators=(",", ":"), default=str)
        parts.append(f"{k}={v}")
    return " ".join(parts)


def build_event(stage: str, **fields: Any) -> dict:
    """The event dict, fully stamped. Split out from ``emit`` so tests (and the
    relay's replay path) can construct one without publishing it."""
    ev = {
        "stage": str(stage),
        "ts": time.time(),
        "seq": _next_seq(),
        "worker_id": worker_id(),
    }
    for k, v in fields.items():
        if v is None:
            continue                     # absent beats null on the wire
        ev[k] = v
    # Inherit the ambient pass id when the caller didn't name one — this is how
    # make-room / slot / hot-cache events land in the same card as the dispatch
    # pass that provoked them.
    if not ev.get("run_id"):
        rid = current_run_id()
        if rid:
            ev["run_id"] = rid
        else:
            ev.pop("run_id", None)
    return ev


def emit_eviction_event(stage: str, **fields: Any) -> Optional[dict]:
    """Publish one eviction-telemetry event. BEST EFFORT, ALWAYS.

    Appends to the local ring, writes the journal line, and hands the event to
    every registered sink. Every one of those three is independently guarded:
    a sink that throws does not stop the ring, and nothing that happens in here
    reaches the caller. Returns the event (handy in tests) or None if even the
    build failed.

    ``run_id`` should be passed by callers inside a headroom pass so their
    events group; a bare event without one still streams, it just renders as
    its own card."""
    try:
        ev = build_event(stage, **fields)
    except Exception:  # noqa: BLE001 — a bad field must never break a load
        logger.debug("eviction telemetry: build failed", exc_info=True)
        return None
    try:
        with _RING_LOCK:
            _RING.append(ev)
    except Exception:  # noqa: BLE001
        pass
    try:
        logger.info("eviction %s", _kv_line(ev))
    except Exception:  # noqa: BLE001
        pass
    try:
        with _SINKS_LOCK:
            sinks = list(_SINKS)
    except Exception:  # noqa: BLE001
        sinks = []
    for sink in sinks:
        try:
            sink(ev)
        except Exception:  # noqa: BLE001 — a broken sink is a telemetry bug,
            logger.debug("eviction telemetry sink failed", exc_info=True)
    # ADDITIVE TAP (2026-09-01): also fold this event into the durable
    # compute-action log the Metrics panel reads. Guarded to the same standard
    # as everything above — a log fault must be INVISIBLE to the compute path,
    # so the whole thing is one swallow-everything try that never touches ``ev``
    # or the return value. See ``_tap_compute_action``.
    try:
        _tap_compute_action(ev)
    except Exception:  # noqa: BLE001 — logging must never break a load/evict
        logger.debug("compute-action log tap failed", exc_info=True)
    return ev


# Which durable-log ``action`` each telemetry STAGE folds into. Stages absent
# from this map do not produce a log row (start/heartbeat/verdict noise the
# panel doesn't chart); the ones present are the coarse buckets the Metrics
# panel groups by. Kept beside the emit path so a new charted stage is one line.
_STAGE_TO_ACTION = {
    STAGE_LOAD_DONE: "load",
    STAGE_LOAD_FAIL: "load",
    STAGE_EVICT_DONE: "evict",
    STAGE_EVICT_FAIL: "evict",
    STAGE_PROVISION_DONE: "provision",
    STAGE_PROVISION_FAIL: "provision",
    STAGE_FIT_FAIL: "fit_fail",
    STAGE_MEMBER_SELECT: "member_select",
}


def _tap_compute_action(ev: dict) -> None:
    """Fold ONE emitted telemetry event into the durable compute-action log.

    Best-effort by construction and by its single caller's guard: it maps the
    stage to a coarse action, pulls the model/worker/bytes/outcome fields the
    panel filters on, and hands the rest to ``append_action`` as ``detail`` —
    which is itself fail-open. Stages not in ``_STAGE_TO_ACTION`` are skipped
    silently. The store import is lazy so this module stays flask/DB-free at
    import time (it runs on bare central, on workers, and in tests).
    """
    action = _STAGE_TO_ACTION.get(ev.get("stage"))
    if not action:
        return
    from hugpy_fleet.central.model_metrics import model_metrics_store
    model = ev.get("model_key") or ev.get("incoming_model")
    worker = ev.get("worker_id")
    # duration_ms -> seconds when a stage carried a timing.
    dur_ms = ev.get("duration_ms")
    duration_s = (float(dur_ms) / 1000.0) if isinstance(dur_ms, (int, float)) else None
    # An outcome word for the panel's status column: an explicit outcome, else
    # "fail" for the .fail stages, else "ok".
    outcome = ev.get("outcome")
    if not outcome:
        outcome = "fail" if str(ev.get("stage") or "").endswith(".fail") else "ok"
    # Everything not already promoted to a column rides in detail, minus the
    # stamps the panel doesn't need per-row.
    detail = {k: v for k, v in ev.items()
              if k not in ("stage", "ts", "seq", "model_key", "incoming_model",
                           "worker_id", "duration_ms", "outcome", "bytes")}
    detail["stage"] = ev.get("stage")
    model_metrics_store.append_action(
        action, model=model, worker_card=worker, duration_s=duration_s,
        bytes=ev.get("bytes"), outcome=outcome, detail=detail,
        ts=ev.get("ts"))


def recent(limit: int = 200) -> list[dict]:
    """The local process ring, oldest-first. This is the PROCESS-local view —
    on central the durable/cross-process view is ``EvictionStore.recent``."""
    with _RING_LOCK:
        items = list(_RING)
    if limit and limit > 0:
        items = items[-limit:]
    return items


def reset_for_tests() -> None:
    """Drop the ring, the sinks and the seq counter. Tests only."""
    global _SEQ, _WORKER_ID
    with _RING_LOCK:
        _RING.clear()
    clear_sinks()
    with _SEQ_LOCK:
        _SEQ = 0
    _WORKER_ID = None


# ---------------------------------------------------------------------------
# Central store — the durable, cross-process view.
#
# gunicorn runs 3 workers x 8 threads here. An ingest POST lands on process B
# while the SSE stream is held by process A, so an in-process pubsub would
# silently show a client only a third of the fleet's events. The shared sqlite
# file is the rendezvous, and its ROWID is the stream cursor: a reader polls
# "rowid > cursor" and gets exactly what it has not seen, regardless of which
# gunicorn worker wrote it. Same file, same pragmas, same EMFILE hardening as
# the comms mirror (see comms/shared.py) — this is that pattern, not a new one.
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS eviction_events (
    rowid_alias INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    seq         INTEGER,
    worker_id   TEXT,
    run_id      TEXT,
    stage       TEXT    NOT NULL,
    model_key   TEXT,
    tier        TEXT,
    body        TEXT    NOT NULL
)
"""
_INDEXES = (
    "CREATE INDEX IF NOT EXISTS ix_evev_ts ON eviction_events(ts)",
    "CREATE INDEX IF NOT EXISTS ix_evev_run ON eviction_events(run_id)",
)

# Bounded history. ~10k rows is hours of fleet eviction and a couple of MB;
# pruning is amortized (every PRUNE_EVERY appends) so ingest stays cheap.
MAX_ROWS = 10000
PRUNE_EVERY = 200
MAX_FAILURES = 5


def default_db_path() -> str:
    """The comms db. Deliberately the SAME resolution as
    ``comms.shared.default_db_path`` — one HUGPY_COMMS_DB per service, one file
    for the whole control plane. Imported lazily so this module stays usable
    when comms.shared is mid-edit or unavailable."""
    try:
        from hugpy_control.shared import default_db_path as _p
        return _p()
    except Exception:  # noqa: BLE001 — mirror the fallback rather than fail
        env = (os.environ.get("HUGPY_COMMS_DB") or "").strip()
        if env:
            return env
        base = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
        return os.path.join(base, f"hugpy-comms-{os.getuid()}.db")


def _retry_on_emfile(fn):
    """Use the comms mirror's EMFILE burst hardening when it is importable
    (restart bursts on the virtiofs mount momentarily return EMFILE — see the
    incident note in comms/shared.py); otherwise call straight through."""
    try:
        from hugpy_control.shared import retry_on_emfile
        return retry_on_emfile(fn)
    except ImportError:
        return fn()


class EvictionStore:
    """Bounded sqlite history of eviction events, shared across processes.

    Every method is best-effort: a store fault degrades the console to "no
    history" and is never allowed to surface into an ingest response or an
    eviction. After MAX_FAILURES consecutive faults the store disables itself
    loudly rather than taxing every ingest with a doomed write — the same
    self-disable the comms mirror uses."""

    def __init__(self, path: Optional[str] = None,
                 max_rows: int = MAX_ROWS) -> None:
        self.path = path or default_db_path()
        self.max_rows = max_rows
        self._failures = 0
        self._disabled = False
        # HUGPY_COMMS_DB honors the same disable sentinels the jobs mirror does
        # ("off"/"none"/"0"/"disabled"). shared.default_db_path returns the env
        # VERBATIM, so without this check a test run with HUGPY_COMMS_DB=off
        # creates a sqlite file literally named `off` in the CWD (it did, on
        # 2026-07-28) instead of disabling the store.
        if str(self.path).strip().lower() in ("off", "none", "0", "disabled"):
            self._disabled = True
        self._initialized = False
        self._init_lock = threading.Lock()
        self._appends = 0

    # -- plumbing ----------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = _retry_on_emfile(lambda: sqlite3.connect(self.path, timeout=2.0))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=2000")
        return conn

    def _ensure(self) -> bool:
        if self._disabled:
            return False
        if self._initialized:
            return True
        with self._init_lock:
            if self._initialized:
                return True
            try:
                d = os.path.dirname(self.path)
                if d:
                    os.makedirs(d, exist_ok=True)
                with self._connect() as conn:
                    conn.execute(_SCHEMA)
                    for stmt in _INDEXES:
                        conn.execute(stmt)
                self._initialized = True
                return True
            except Exception as exc:  # noqa: BLE001
                self._note_failure("init", exc)
                return False

    def _note_failure(self, op: str, exc: BaseException) -> None:
        self._failures += 1
        if self._failures >= MAX_FAILURES and not self._disabled:
            self._disabled = True
            logger.error("eviction telemetry store DISABLED after %d failures "
                         "(last: %s during %s) — the console loses eviction "
                         "history until restart; evictions are unaffected",
                         self._failures, exc, op)
        else:
            logger.warning("eviction telemetry store %s failed: %s", op, exc)

    # -- write -------------------------------------------------------------
    def append(self, events: Iterable[dict]) -> int:
        """Persist a batch. Returns how many rows landed (0 on any fault).

        The full event is stored as JSON in ``body``; the columns are only the
        ones we filter/order by. That keeps the schema stable as stages gain
        fields — a new field needs no migration, it just rides in the body."""
        rows = []
        for ev in events or ():
            if not isinstance(ev, dict) or not ev.get("stage"):
                continue
            try:
                rows.append((
                    float(ev.get("ts") or time.time()),
                    int(ev.get("seq") or 0),
                    str(ev.get("worker_id") or ""),
                    str(ev.get("run_id") or ""),
                    str(ev.get("stage")),
                    str(ev.get("model_key") or ""),
                    str(ev.get("tier") or ""),
                    json.dumps(ev, default=str),
                ))
            except Exception:  # noqa: BLE001 — skip the bad row, keep the batch
                continue
        if not rows or not self._ensure():
            return 0
        try:
            with self._connect() as conn:
                conn.executemany(
                    "INSERT INTO eviction_events "
                    "(ts, seq, worker_id, run_id, stage, model_key, tier, body) "
                    "VALUES (?,?,?,?,?,?,?,?)", rows)
            self._failures = 0
        except Exception as exc:  # noqa: BLE001
            self._note_failure("append", exc)
            return 0
        self._appends += len(rows)
        if self._appends >= PRUNE_EVERY:
            self._appends = 0
            self.prune()
        return len(rows)

    def prune(self) -> None:
        """Keep the newest ``max_rows``. Amortized, best-effort, never fatal."""
        if not self._ensure():
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM eviction_events WHERE rowid_alias <= ("
                    "  SELECT MAX(rowid_alias) - ? FROM eviction_events)",
                    (self.max_rows,))
        except Exception as exc:  # noqa: BLE001
            self._note_failure("prune", exc)

    # -- read --------------------------------------------------------------
    def recent(self, limit: int = 200, since_ts: Optional[float] = None,
               after_id: Optional[int] = None) -> list[dict]:
        """Newest-last list of events, each carrying its store id as ``_id``.

        ``after_id`` is the STREAM cursor (rowid); ``since_ts`` is the
        human-facing "last N seconds" filter. The query selects the newest
        matching rows then reverses, so a ``limit`` always yields the LATEST
        events rather than the oldest ones."""
        if not self._ensure():
            return []
        clauses, args = [], []
        if after_id is not None:
            clauses.append("rowid_alias > ?")
            args.append(int(after_id))
        if since_ts is not None:
            clauses.append("ts >= ?")
            args.append(float(since_ts))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        lim = max(1, min(int(limit or 200), 5000))
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT rowid_alias, body FROM eviction_events" + where +
                    " ORDER BY rowid_alias DESC LIMIT ?", (*args, lim))
                got = cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            self._note_failure("recent", exc)
            return []
        out = []
        for rid, body in reversed(got):
            try:
                ev = json.loads(body)
            except Exception:  # noqa: BLE001
                continue
            ev["_id"] = int(rid)
            out.append(ev)
        return out

    def max_id(self) -> int:
        """The current head cursor — where a live-only stream starts."""
        if not self._ensure():
            return 0
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT MAX(rowid_alias) FROM eviction_events").fetchone()
            return int(row[0]) if row and row[0] else 0
        except Exception as exc:  # noqa: BLE001
            self._note_failure("max_id", exc)
            return 0


_STORE: Optional[EvictionStore] = None
_STORE_LOCK = threading.Lock()


def get_store() -> EvictionStore:
    """The process-wide store singleton (central side)."""
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = EvictionStore()
    return _STORE


def set_store(store: Optional[EvictionStore]) -> None:
    """Swap the singleton (tests point it at a tmp file)."""
    global _STORE
    with _STORE_LOCK:
        _STORE = store


def install_store_sink() -> None:
    """CENTRAL: locally emitted events go straight to the durable store.

    Central does no local LLM serving (HUGPY_NO_LOCAL_SERVING), so it rarely
    evicts — but the studio/video reservation path does, and those events must
    appear in the same stream as the fleet's."""
    def _sink(ev: dict) -> None:
        get_store().append([ev])
    register_sink(_sink)


# ---------------------------------------------------------------------------
# Worker relay — buffered, batched, non-blocking.
#
# Workers run on other boxes; central must see their events or the console is
# blind to the only boxes that actually evict. The relay is a bounded buffer
# plus one daemon thread: emit() only appends (microseconds, never network),
# and the thread POSTs batches. "Real time" for a human watching is sub-second,
# so it flushes every FLUSH_INTERVAL_S or as soon as BATCH_MAX events are
# queued, whichever comes first.
# ---------------------------------------------------------------------------

FLUSH_INTERVAL_S = 0.5
BATCH_MAX = 20
RELAY_BUFFER_MAX = 2000
INGEST_PATH = "/llm/evictions/ingest"


class EvictionRelay:
    """Worker-side buffered relay to central's ingest endpoint.

    Drop-oldest when the buffer fills: a central that is down or slow must cost
    a worker nothing but stale events. Failures are logged sparsely (first, then
    every 100th) — a fleet-wide central outage must not itself become the log
    storm that starves heartbeats (see the 2026-07-27 stat-storm incident)."""

    def __init__(self, post: Callable[[list[dict]], None],
                 interval: float = FLUSH_INTERVAL_S,
                 batch_max: int = BATCH_MAX,
                 buffer_max: int = RELAY_BUFFER_MAX) -> None:
        self._post = post
        self._interval = interval
        self._batch_max = batch_max
        self._buf: Deque[dict] = deque(maxlen=buffer_max)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.dropped = 0
        self.sent = 0
        self.failures = 0

    def offer(self, ev: dict) -> None:
        """The sink. Append-only; never blocks, never talks to the network."""
        with self._lock:
            if len(self._buf) == self._buf.maxlen:
                self.dropped += 1
            self._buf.append(ev)
            ready = len(self._buf) >= self._batch_max
        if ready:
            self._wake.set()

    def drain(self, limit: Optional[int] = None) -> list[dict]:
        with self._lock:
            n = len(self._buf) if limit is None else min(limit, len(self._buf))
            return [self._buf.popleft() for _ in range(n)]

    def flush_once(self) -> int:
        """POST whatever is buffered. Returns the count sent (0 on failure).

        A failed batch is NOT re-queued: re-queueing under a sustained central
        outage turns the buffer into a replay loop that never drains and starves
        fresh events. Telemetry is lossy on purpose — the journal holds the
        durable copy."""
        batch = self.drain(self._batch_max * 5)
        if not batch:
            return 0
        try:
            self._post(batch)
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            if self.failures == 1 or self.failures % 100 == 0:
                logger.warning("eviction telemetry relay: %d failed flush(es), "
                               "last: %s (events dropped; evictions unaffected)",
                               self.failures, exc)
            return 0
        self.failures = 0
        self.sent += len(batch)
        return len(batch)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self._interval)
            self._wake.clear()
            try:
                self.flush_once()
            except Exception:  # noqa: BLE001 — the thread must never die
                logger.debug("eviction relay loop error", exc_info=True)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="eviction-telemetry-relay", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()


_RELAY: Optional[EvictionRelay] = None


def install_relay(post: Callable[[list[dict]], None]) -> EvictionRelay:
    """WORKER: buffer emitted events and batch them to central.

    ``post(batch)`` is supplied by the worker agent so this module needs no
    knowledge of central's base url or auth — that already lives in the agent's
    central client, and duplicating it here would be a second thing to get
    wrong."""
    global _RELAY
    if _RELAY is not None:
        return _RELAY
    relay = EvictionRelay(post)
    relay.start()
    register_sink(relay.offer)
    _RELAY = relay
    return relay


def relay_stats() -> dict:
    if _RELAY is None:
        return {"installed": False}
    return {"installed": True, "sent": _RELAY.sent,
            "dropped": _RELAY.dropped, "failures": _RELAY.failures}
