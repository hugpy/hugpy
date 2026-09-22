"""Runner dispatch — dumb consumer of Resolution.

All routing logic lives in model_resolver.resolve(). This module owns
two things and only two things:

    1. A per-process instance cache keyed by (model_key, task).
    2. An execute_prompt entry point that turns request kwargs into
       a result by handing off to resolve() and the runner.

It does not:
    - Decide which builder to call.
    - Decide which runner class to instantiate.
    - Validate that model+task are compatible.
    - Default task to cfg.primary_task.

If you find yourself adding any of that here, stop and add it to
model_resolver.resolve() instead. That's the whole point.

Why a per-process cache:
    Loading a 14B model takes seconds; doing it on every request is
    obviously wrong. Per-(model_key, task) caching means the same
    model can host two task-runners (e.g. text-generation + code-
    generation on one llama.cpp instance) and each gets its own
    runner wrapper, but inner singletons (REGISTRY for DeepCoder,
    get_llama_runner for llama.cpp) still de-dup the heavy state.
"""

from __future__ import annotations
import contextvars
import inspect
import asyncio
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from hugpy_engine.resolvers import resolve
from hugpy_engine.schemas.event_schemas import DoneEvent, ErrorEvent, TokenEvent
from hugpy_engine.schemas.runner_schemas import Runner
from hugpy_engine.schemas.task_schemas import Resolution

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# k96 NO-MAKEROOM (request flag "no_makeroom": true — the agent-brain no-evict
# guarantee, operator ruling 2026-08-06).
#
# Semantics: serving this request must never cost the fleet a resident model.
# A WARM model serves exactly as today. A COLD load runs the whole admission
# POLITELY (the k56 no_evict rules): the in-process contention yield is
# skipped, the cross-tier make-room hook may only admit into genuinely free
# room (its polite branch — it never evicts and refuses honestly), the slot
# pool neither ceiling-evicts nor bumps a seat, and an unfittable load FAILS
# FAST with LoadRefusal ("won't fit … refusing without evicting") — a
# capacity-class error the agent's brain ladder walks down.
#
# Transport: execute_prompt/execute_prompt_stream read the key from
# prompt_kwargs and bind this contextvar for the pass; the RELAY translates
# the flag onto the spill's ``no_evict`` (version-gated per worker), so on a
# worker the per-request HUGPY_NO_EVICT env carries the same guarantee across
# threads. Flag absent -> everything below is byte-identical to before.
# ---------------------------------------------------------------------------
_NO_MAKEROOM: contextvars.ContextVar = contextvars.ContextVar(
    "hugpy_no_makeroom", default=False)


def no_makeroom_active() -> bool:
    """True while the current request carries "no_makeroom": true — read by
    the headroom pass below, the slot pool's seat path, and the worker's
    make-room hook so every evicting choke point asks the same question."""
    try:
        return bool(_NO_MAKEROOM.get())
    except Exception:  # noqa: BLE001 — a broken flag read is not a policy
        return False


# ---------------------------------------------------------------------------
# Per-process instance cache — keyed by (model_key, task) per the contract
# in Resolution.cache_key.
# ---------------------------------------------------------------------------

_INSTANCES: Dict[Tuple[str, str], Runner] = {}
_INSTANCES_LOCK = threading.Lock()

# Models whose runner build (weight load) is IN FLIGHT right now — read by the
# worker heartbeat so the console can attribute "heating" (load in progress)
# distinctly from "serving" (resident) and "cold" (assigned, not loaded).
# Separate lock: heartbeats must read this while a slow build holds
# _INSTANCES_LOCK for minutes.
_BUILDING: set = set()
_BUILDING_LOCK = threading.Lock()


def loading_model_keys() -> List[str]:
    """model_keys whose runner/weights are currently loading in this process."""
    with _BUILDING_LOCK:
        return sorted(_BUILDING)


# ---------------------------------------------------------------------------
# Residency policy — CONTENTION-based LRU (operator doctrine, locked 2026-07-11).
#
# Doctrine: "The likelihood of one model being queried having another
# consecutive query is high — this is the M.O. of hugpy. Resources should
# facilitate keeping models hot; models should spend the least possible time
# 'loading'." So an on-demand model loads on call and STAYS resident until
# another load actually NEEDS its resources; then the least-recently-used
# on-demand resident yields. static/pinned residents NEVER yield.
#
# 2026-07-11 DRIFT CORRECTION: this used to be a CLOCK — the worker's residency
# sweep evicted any on-demand resident after on_demand_ttl_s of no requests
# (default 900s), so a model that had just answered a chat was gone minutes
# later and the next chat paid a full reload. That contradicted the doctrine and
# the console's own residency tooltip ("holds its slot until another model needs
# the seat"). The idle clock is now OPT-IN (worker_agent._residency_sweep_once:
# it runs only when the operator explicitly sets on_demand_ttl_s); the DEFAULT
# trigger is memory contention — ensure_headroom_for_load(), below.
# ---------------------------------------------------------------------------

# Per-model last-request time — the LRU key. touch_model() fires on every served
# request (execute_prompt / execute_prompt_stream); the least-recently touched
# on-demand resident is the first to yield under contention.
_LAST_USED: Dict[str, float] = {}


def touch_model(model_key: str) -> None:
    import time as _time
    _LAST_USED[model_key] = _time.time()


def last_used_snapshot() -> Dict[str, float]:
    return dict(_LAST_USED)


class LoadRefusal(Exception):
    """A GPU load that can't fit even after evicting every permissible resident
    (slice 10). Raised by ensure_headroom_for_load's cross-tier make-room BEFORE
    any CUDA allocation, carrying the typed reason so the caller surfaces an
    honest 'won't fit' instead of an admit-then-OOM crash."""
    def __init__(self, reason: dict):
        self.reason = reason or {}
        super().__init__(self.reason.get("reason") or "won't fit on GPU")


# ---------------------------------------------------------------------------
# Contention hooks — registered by the worker agent; None on bare central.
#
# dispatch is shared code and must not import worker_agent (the dependency runs
# the other way). So, mirroring managers.serve.slots.set_residency_lookup, the
# worker registers the box-specific policy here at startup. Unset -> contention
# eviction is a no-op: central does no local serving, and tests register their
# own fakes. dispatch owns the LRU MECHANISM; the worker owns the fit MEASUREMENT.
# ---------------------------------------------------------------------------
_FIT_CHECK = None        # (model_key) -> bool: does loading it fit in current headroom?
_EVICTABLE = None        # (model_key) -> bool: is it a yieldable on-demand in-process resident?
_POST_EVICT = None       # () -> None: reclaim host RAM / CUDA cache after an eviction
# (model_key) -> dict: CROSS-TIER VRAM make-room (slice 10). The in-process LRU
# yield below only sees _INSTANCES residents; a SLOT CHILD (subprocess) squatting
# the GPU is invisible to it. The worker registers a make-room hook that sees ALL
# residents (in-process + slot + comfy) from the pid-registry measured truth and
# evicts the minimum permissible set through the /ops/evict verb — closing the
# blind admit-then-OOM the incident exposed. None -> the old in-process-only path.
_MAKE_ROOM = None
# (model_key) -> str|None: WHY a candidate was rejected by _EVICTABLE ("static",
# "actively-replying", "slot-backed", ...). Eviction telemetry only — the yield
# DECISION is still _EVICTABLE's alone. Registered separately (rather than by
# widening _EVICTABLE's return) so the predicate's contract is untouched and an
# unregistered reason hook simply reports "not-evictable".
_EVICT_REASON = None
_CONTENTION_LOCK = threading.Lock()


# --- eviction telemetry (observation only; never gates a load) --------------
# The operator must be able to WATCH a headroom pass happen (directive
# 2026-07-28). Every stage below emits a structured event AND a journal line.
# comms.evictions is stdlib-only and flask-free, so importing it here keeps
# dispatch importable on bare central and in tests exactly as before; if the
# import fails for any reason, _emit degrades to a no-op and the eviction path
# is byte-identical to its pre-telemetry self.
from hugpy_engine.placement import get_eviction_ledger as _ledger  # noqa: E402

STAGE_CANDIDATE_SKIP = "candidate.skip"


def _emit(stage: str, **fields) -> None:
    """Best-effort telemetry. Swallows everything — a load must never fail,
    slow down, or change outcome because an event could not be published."""
    try:
        _ledger().emit(stage, **fields)
    except Exception:                        # noqa: BLE001
        pass


def _current_group() -> "Optional[dict]":
    """The ambient model-group demand, or None. ``getattr`` rather than a direct
    call so a peer running a comms module that predates model groups degrades to
    "no group" instead of an AttributeError on the eviction path."""
    try:
        fn = getattr(_ledger(), "current_group", None)
        return fn() if fn is not None else None
    except Exception:                        # noqa: BLE001
        return None


def _new_run_id() -> str:
    try:
        return str(_ledger().new_run_id() or "")
    except Exception:                        # noqa: BLE001
        return ""


class _run_scope:
    """Bind the pass's run_id for this thread so the worker's cross-tier
    make-room hook, the slot pool and the hot-cache reaper all emit INTO this
    pass without dispatch having to thread an argument through their signatures.
    Degrades to a bare no-op context when telemetry is unavailable."""

    def __init__(self, run_id: str) -> None:
        self._inner = None
        if run_id:
            try:
                self._inner = _ledger().run_scope(run_id)
            except Exception:                # noqa: BLE001
                self._inner = None

    def __enter__(self):
        if self._inner is not None:
            try:
                self._inner.__enter__()
            except Exception:                # noqa: BLE001
                self._inner = None
        return self

    def __exit__(self, *exc):
        if self._inner is not None:
            try:
                self._inner.__exit__(*exc)
            except Exception:                # noqa: BLE001
                pass
        return None


def set_fit_check(fn) -> None:
    """Register the headroom fit-guard: ``fn(model_key) -> bool``, True when the
    model fits in current headroom without yielding a resident. None disables
    contention eviction (the bare-central / untested default)."""
    global _FIT_CHECK
    _FIT_CHECK = fn


def set_evictable(fn) -> None:
    """Register the yield predicate: ``fn(model_key) -> bool``, True for an
    on-demand in-process resident that MAY yield (not static, not pinned, no
    active gate permits, not slot-backed)."""
    global _EVICTABLE
    _EVICTABLE = fn


def set_post_evict_hook(fn) -> None:
    """Register a post-eviction reclaim (gc + malloc_trim + cuda empty_cache) so
    the next headroom re-check sees the freed memory. None -> skipped."""
    global _POST_EVICT
    _POST_EVICT = fn


def set_make_room(fn) -> None:
    """Register the CROSS-TIER VRAM make-room (slice 10): ``fn(model_key) -> dict``
    with {"action": "proceed"|"evicted"|"refuse", "reason": {...}|None}. Called
    after the in-process LRU yield so a SLOT-CHILD squatter (invisible to the
    in-process path) is also evicted, and an unfittable load is REFUSED before any
    CUDA allocation. None -> the historical in-process-only contention path."""
    global _MAKE_ROOM
    _MAKE_ROOM = fn


def set_evict_reason(fn) -> None:
    """Register the telemetry-only skip-reason lookup: ``fn(model_key) -> str``,
    the reason ``_EVICTABLE`` rejected that candidate ("static",
    "actively-replying", "slot-backed", ...). Purely observational — it is read
    only to LABEL a skip the predicate already decided, never to decide one.
    Unregistered -> skips report the generic "not-evictable"."""
    global _EVICT_REASON
    _EVICT_REASON = fn


def _skip_reason(model_key: str) -> str:
    """The label for a rejected candidate. Never raises: an unreadable reason
    degrades to the generic string rather than disturbing the yield loop."""
    if _EVICT_REASON is None:
        return "not-evictable"
    try:
        return str(_EVICT_REASON(model_key) or "not-evictable")
    except Exception:                        # noqa: BLE001
        return "not-evictable"


def _next_lru_evictable(exclude: str, run_id: str = "") -> Optional[str]:
    """The least-recently-used yieldable on-demand in-process resident, or None.

    Candidates are the model_keys currently holding a runner in ``_INSTANCES``,
    minus the model being loaded, minus everything the registered ``_EVICTABLE``
    predicate rejects. Those rejections are ONLY: 🔒static (the one residency
    lock), actively-answering (an in-flight generation), and slot-backed (weights
    in another process, so dropping the proxy frees nothing). 📌 pin is NOT among
    them — pin is routing DESIGNATION only (survives restarts) and has no bearing
    on residency or eviction (operator ruling 2026-07-15/07-17); a pinned model
    yields to contention like any other. Ordered by ``_LAST_USED`` ascending —
    the coldest yields first. A never-touched resident (warmed by the filler,
    never requested) sorts oldest and yields first, which is exactly right."""
    if _EVICTABLE is None:
        return None
    residents = {mk for (mk, _task) in loaded_model_keys()}
    residents.discard(exclude)
    # Telemetry (2026-07-28): report the REJECTED residents too, with WHY. The
    # winner alone is not the story the doctrine cares about — "what was
    # protected, and under which clause" is. Sorted so a run's skip rows render
    # in a stable order rather than set-iteration order.
    cands = []
    for mk in sorted(residents):
        if _EVICTABLE(mk):
            cands.append(mk)
        else:
            _emit(STAGE_CANDIDATE_SKIP,
                  run_id=run_id, model_key=mk, tier="in-process",
                  incoming_model=exclude, reason=_skip_reason(mk))
    if not cands:
        return None
    cands.sort(key=lambda mk: _LAST_USED.get(mk, 0.0))
    return cands[0]


def ensure_headroom_for_load(model_key: str, trigger: str = "load") -> List[str]:
    """CONTENTION eviction — the default (clock-free) residency trigger.

    Called right before a NEW runner is built (see ``_get_or_build_runner``).
    While the registered fit-guard says the incoming model does NOT fit, yield
    the least-recently-used on-demand resident one at a time — re-checking
    headroom after each (the post-evict reclaim hook trims the freed host arena /
    CUDA cache so the re-check actually sees the room) — until it fits or nothing
    is left to yield.

    A model with an in-flight generation (gate permits) is skipped, never ripped
    out from under it — the next LRU candidate is chosen, and the busy one
    becomes evictable only once its permits release. If nothing is yieldable, we
    return and the load proceeds (or fails) EXACTLY as it does today: contention
    only ADDS room, it never changes the too-big error envelope.

    No-op on bare central / when no fit-guard is registered, and on a POLITE
    load (k56, spill ``no_evict``): a model that may spend only genuinely free
    headroom yields nothing and evicts nobody. Returns the list of yielded
    model_keys (for logging + tests).

    k96 ``no_makeroom`` (request flag, bound via the module contextvar): same
    politeness per request, plus fail-fast — where an unflagged short pass
    would proceed-unfit, a no_makeroom pass raises LoadRefusal ("refusing
    without evicting") so the caller gets a capacity-class error instead of a
    wedged cold load. A warm model never reaches this pass at all.

    CROSS-TIER (slice 10): after the in-process LRU yield, a registered make-room
    hook (set_make_room) runs a SECOND pass that also evicts SLOT-CHILD squatters
    (subprocess VRAM the in-process path cannot see) and REFUSES an unfittable
    load before any CUDA allocation (raising LoadRefusal). Without the hook the
    behavior is byte-identical to before.

    TELEMETRY (2026-07-28): the whole pass runs inside one ``run_id`` scope so
    every stage — fit failures, protected residents and WHY, each yield, the
    cross-tier verdict — streams to the console as one correlated card. Purely
    observational: see ``comms.evictions``."""
    run_id = _new_run_id()
    with _run_scope(run_id):
        return _headroom_pass(model_key, trigger, run_id)


def _headroom_pass(model_key: str, trigger: str, run_id: str) -> List[str]:
    """The pass itself. Split out only so the public entry point can hold the
    telemetry run scope open across it (including the raising paths) without
    re-indenting the eviction logic."""
    evicted: List[str] = []
    outcome = "fit"
    # ``group`` is present only when a MODEL GROUP's ticked standard is what
    # demanded this pass (member selector opened an evictions.group_scope) —
    # that is how an operator tells "the speed tick refused to spill" from an
    # ordinary contention eviction. Absent otherwise, and absence is the norm:
    # a plain contention pass has no group and renders exactly as before.
    _emit("headroom.start", run_id=run_id, incoming_model=model_key,
          trigger=trigger, group=_current_group())
    # k56 POLITE LOAD: this pass exists to TAKE room from residents, so a load
    # flagged no_evict skips the in-process yield entirely. The cross-tier hook
    # below still runs — it is the honest-refusal path, and it reads the same
    # flag to decide between "admit into free room" and "refuse without
    # touching anyone". Unflagged loads are untouched.
    #
    # k96 "no_makeroom": the per-REQUEST twin of the per-model no_evict flag
    # (agent-brain calls send it on every chat). Same politeness, one extra
    # promise: where an unflagged short pass would proceed-unfit (the
    # admit-then-OOM shape) this one FAILS FAST below — the caller wants a
    # capacity-class refusal it can walk its brain ladder on, never a wedged
    # cold load. A warm model never reaches this pass (its runner is cached).
    try:
        from hugpy_engine.spill import no_evict_env
        _polite = no_evict_env()
    except Exception:  # noqa: BLE001 — an unreadable flag is not a policy
        _polite = False
    _no_makeroom = no_makeroom_active()
    _polite = _polite or _no_makeroom
    if _polite:
        logger.info("polite load (no_evict%s): skipping the in-process contention "
                    "yield for %s — it may only take genuinely free room",
                    "/no_makeroom" if _no_makeroom else "", model_key)
        _emit("candidate.skip", run_id=run_id, incoming_model=model_key,
              tier="in-process", reason="polite load (no_evict) — never evicts")
    if _FIT_CHECK is not None and not _polite:
        with _CONTENTION_LOCK:
            while not _FIT_CHECK(model_key):
                _emit("fit.fail", run_id=run_id, incoming_model=model_key)
                cand = _next_lru_evictable(exclude=model_key, run_id=run_id)
                if cand is None:
                    break                    # nothing to yield in-process -> below
                logger.info("contention evict: yielding LRU on-demand resident %s to "
                            "make room for %s (doctrine 2026-07-11: keep models hot, "
                            "yield only under memory pressure)", cand, model_key)
                _emit("evict.start", run_id=run_id, model_key=cand,
                      tier="in-process", incoming_model=model_key)
                _t0 = time.time()
                try:
                    evict(cand)
                except Exception as _exc:    # noqa: BLE001 — one bad evict must not wedge the load
                    logger.warning("contention evict of %s failed", cand, exc_info=True)
                    _emit("evict.fail", run_id=run_id, model_key=cand,
                          tier="in-process", incoming_model=model_key,
                          error=str(_exc))
                    break
                _emit("evict.done", run_id=run_id, model_key=cand,
                      tier="in-process", incoming_model=model_key,
                      duration_ms=int((time.time() - _t0) * 1000))
                evicted.append(cand)
                if _POST_EVICT is not None:
                    try:
                        _POST_EVICT()        # trim so the next fit re-check sees the freed room
                    except Exception:        # noqa: BLE001
                        logger.warning("post-evict reclaim failed", exc_info=True)
                    else:
                        _emit("reclaim.done", run_id=run_id,
                              incoming_model=model_key)

    # CROSS-TIER make-room + honest refusal (slice 10). The in-process yield above
    # cannot see a slot child; this hook sees ALL residents and evicts the minimum
    # permissible set (slot children included), then REFUSES if still short.
    if _MAKE_ROOM is not None:
        try:
            verdict = _MAKE_ROOM(model_key)
        except LoadRefusal as _ref:
            _emit("makeroom.verdict", run_id=run_id, incoming_model=model_key,
                  action="refuse", reason=getattr(_ref, "reason", None))
            _emit("headroom.done", run_id=run_id, incoming_model=model_key,
                  evicted=list(evicted), outcome="refused")
            raise
        except Exception:                    # noqa: BLE001 — a broken hook never blocks a load
            logger.warning("cross-tier make-room failed for %s", model_key,
                           exc_info=True)
            verdict = None
        if isinstance(verdict, dict):
            _emit("makeroom.verdict", run_id=run_id, incoming_model=model_key,
                  action=verdict.get("action"), reason=verdict.get("reason"),
                  evicted=list(verdict.get("evicted") or []),
                  freed_bytes=verdict.get("freed_bytes"),
                  note=verdict.get("note"))
            evicted.extend(v for v in (verdict.get("evicted") or [])
                           if v not in evicted)
            # action == "partial": the full weights don't fit, but the worker
            # admitted an honest GGUF PARTIAL offload (autofit's hybrid). It is an
            # ADMIT, not a refusal — the in-process llama_cpp load then reads the
            # pinned n_gpu_layers via spill.gguf_gpu_layers (set on the served
            # path by the same admission), so we neither raise nor re-price.
            if verdict.get("action") == "refuse":
                _emit("headroom.done", run_id=run_id, incoming_model=model_key,
                      evicted=list(evicted), outcome="refused")
                raise LoadRefusal(verdict.get("reason") or
                                  {"reason": "won't fit on GPU", "model_key": model_key})
            if verdict.get("action") == "partial":
                outcome = "partial"
    # "proceeded-unfit": the in-process yield ran out of permissible residents
    # and no make-room hook is registered, so the load goes ahead against a
    # headroom the fit-guard last said was short. Naming it is the point — that
    # is precisely the admit-then-OOM shape the operator wants visible.
    if outcome == "fit" and _FIT_CHECK is not None and _MAKE_ROOM is None:
        try:
            if not _FIT_CHECK(model_key):
                outcome = "proceeded-unfit"
        except Exception:                    # noqa: BLE001 — unmeasurable -> report fit
            pass
        if outcome == "proceeded-unfit" and _no_makeroom:
            # k96: a no_makeroom request must never ride the admit-then-OOM
            # shape — with nothing evictable ON PURPOSE and the fit-guard
            # saying short, refuse NOW with the capacity-class wording the
            # agent's brain ladder walks on.
            _emit("headroom.done", run_id=run_id, incoming_model=model_key,
                  evicted=list(evicted), outcome="refused")
            raise LoadRefusal({
                "reason": ("won't fit on GPU: %s is not resident and free "
                           "headroom is short — no_makeroom forbids evicting "
                           "residents, refusing without evicting" % model_key),
                "model_key": model_key, "no_makeroom": True})
    _emit("headroom.done", run_id=run_id, incoming_model=model_key,
          evicted=list(evicted), outcome=outcome)
    return evicted


def _get_or_build_runner(res: Resolution) -> Runner:
    """Cache-coherent runner lookup. Double-checked locking under the cache lock."""
    cached = _INSTANCES.get(res.cache_key)
    if cached is not None:
        return cached

    # Contention-based residency (doctrine 2026-07-11): a NEW in-process load may
    # need room. Yield the LRU on-demand resident(s) BEFORE we build — done here,
    # OUTSIDE _INSTANCES_LOCK, because evict() takes that lock too (holding it
    # would deadlock). No-op on bare central (no fit-guard registered).
    ensure_headroom_for_load(res.model_key)

    with _INSTANCES_LOCK:
        cached = _INSTANCES.get(res.cache_key)
        if cached is not None:
            return cached

        logger.info(
            "instantiating runner: model=%s task=%s class=%s framework=%s",
            res.model_key, res.task, res.runner_cls.__name__, res.framework,
        )
        with _BUILDING_LOCK:
            _BUILDING.add(res.model_key)
        try:
            instance = res.runner_cls(res.cfg)
        finally:
            with _BUILDING_LOCK:
                _BUILDING.discard(res.model_key)
        _INSTANCES[res.cache_key] = instance
        return instance


# ---------------------------------------------------------------------------
# Argument normalization — flexible positional input -> kwargs dict.
# ---------------------------------------------------------------------------

def infer_arg_name(arg: Any) -> Optional[str]:
    if arg is None:
        return None
    if isinstance(arg, bool):
        return "do_sample"
    if isinstance(arg, int):
        return "max_new_tokens"
    if isinstance(arg, float):
        return "temperature"
    if isinstance(arg, list):
        return "messages"
    if isinstance(arg, str):
        if os.path.exists(arg):
            return "file"
        lowered = arg.lower()
        looks_like_model = (
            "/" in arg
            or "_gguf" in lowered
            or any(tag in lowered for tag in ("qwen", "llama", "mistral", "gpt"))
        )
        return "model_key" if looks_like_model else "messages"
    return None


def normalize_prompt_kwargs(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Convert flexible input into builder-compatible kwargs.

    Explicit kwargs win over inferred positional args. A second float
    becomes top_p (since temperature is already set).
    """
    prompt_kwargs = dict(kwargs)

    for arg in args:
        guessed_key = infer_arg_name(arg)
        if guessed_key is None:
            raise TypeError(f"Could not infer argument type for positional arg: {arg!r}")

        if guessed_key in prompt_kwargs:
            if guessed_key == "temperature" and "top_p" not in prompt_kwargs:
                prompt_kwargs["top_p"] = arg
            continue

        prompt_kwargs[guessed_key] = arg

    return prompt_kwargs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def runner_for(
    model_key: Optional[str] = None,
    *,
    task: Optional[str] = None,
) -> Runner:
    """Get a runner by model_key, task, or both.

    Both are passed through resolve() — so the same (model_key, task)
    pair always lands on the same cached runner, whether you came in
    here or through execute_prompt.
    """
    if model_key is None and task is None:
        raise ValueError("runner_for requires at least one of model_key or task")

    res = resolve({"model_key": model_key, "task": task})
    return _get_or_build_runner(res)


def execute_prompt(*args: Any, **kwargs: Any):
    """One-shot request -> result. Sync entrypoint; awaits inside if needed.

    ``no_makeroom`` (k96): a truthy prompt_kwarg binds the no-makeroom
    contextvar for the pass — the cold-load admission then runs politely and
    refuses instead of evicting (see the module-level note)."""
    prompt_kwargs = normalize_prompt_kwargs(*args, **kwargs)
    token = _NO_MAKEROOM.set(bool(prompt_kwargs.get("no_makeroom")))
    try:
        res = resolve(prompt_kwargs)
        req = res.builder(prompt_kwargs, res.model_key)
        runner = _get_or_build_runner(res)
        touch_model(res.model_key)   # residency idle clock
        return runner.run(req=req)
    finally:
        # The try body can resume in a DIFFERENT Context (async/greenlet hop),
        # where reset(token) raises "created in a different Context" and 500s
        # the whole request (live incident 2026-08-06, first cold load after
        # k96). Same-context nesting still uses the token; cross-context falls
        # back to restoring the default explicitly.
        try:
            _NO_MAKEROOM.reset(token)
        except ValueError:
            _NO_MAKEROOM.set(False)


async def stream_runner(runner, req, cancel_event=None):
    """Drive any runner to a stream of events — the shared streaming primitive.

    Runners that implement a real async-generator `stream` get streamed through
    (cancel_event forwarded when the signature accepts it). Everyone else is run
    once and emitted as a single token + done. The caller never has to know
    which kind it got.

    This is factored out of execute_prompt_stream so other places that hold a
    runner + built req — notably the DelegatingRunner's local-fallback branch —
    reuse the exact same stream-or-wrap logic instead of duplicating it.
    """
    stream = getattr(runner, "stream", None)
    if stream is not None:
        # Pass cancel_event only if the runner's stream() accepts it.
        try:
            accepts_cancel = "cancel_event" in inspect.signature(stream).parameters
        except (TypeError, ValueError):
            accepts_cancel = False
        produced = stream(req, cancel_event=cancel_event) if (accepts_cancel and cancel_event is not None) else stream(req)
        if hasattr(produced, "__aiter__"):          # real streamer
            async for event in produced:
                yield event
            return
        if inspect.isawaitable(produced):           # coroutine-shaped stream(); don't leak it
            produced.close()

    # one-shot path — the universal verb every runner implements
    result = runner.run(req=req)
    if inspect.isawaitable(result):
        result = await result

    if getattr(result, "ok", True):
        yield TokenEvent(request_id=req.request_id,
                         text=getattr(result, "text", "") or str(result))
        yield DoneEvent(request_id=req.request_id, input_tokens=0,
                        output_chunks=1,
                        finish_reason=getattr(result, "finish_reason", None) or "stop",
                        # ChatResult already defines usage; surface it when the
                        # runner filled it in (None otherwise — additive).
                        usage=getattr(result, "usage", None),
                        # Same additive contract for the engine's measured
                        # decode rate: TaskResult is extra="allow", so a runner
                        # that attached `timings` carries it here. None from any
                        # runner that didn't. RECORDING ONLY.
                        timings=getattr(result, "timings", None))
    else:
        yield ErrorEvent(request_id=req.request_id,
                         message=getattr(result, "error", None) or "run failed")


async def execute_prompt_stream(*args, cancel_event=None, **kwargs):
    """Single resolve→builder→runner pass, yielded as events.

    The primitive: resolve the request, build it, stream the runner once.
    Chat continuation/seam handling lives one layer up in execute_chat_stream;
    this stays a single pass so it composes (and so a remote relay sees one
    pass, not a nested continuation loop).

    ``cancel_event`` (an asyncio.Event) is forwarded to runners that accept it
    (llama.cpp, summarizer, DeepCoder), so a caller can stop generation
    mid-stream.

    ``no_makeroom`` (k96): a truthy prompt_kwarg binds the no-makeroom
    contextvar across the pass (build AND stream — a lazy slot seat happens on
    first runner access), so the admission runs politely and refuses instead
    of evicting (see the module-level note)."""
    prompt_kwargs = normalize_prompt_kwargs(*args, **kwargs)
    token = _NO_MAKEROOM.set(bool(prompt_kwargs.get("no_makeroom")))
    try:
        res = resolve(prompt_kwargs)
        req = res.builder(prompt_kwargs, res.model_key)
        runner = _get_or_build_runner(res)
        touch_model(res.model_key)   # residency idle clock
        async for event in stream_runner(runner, req, cancel_event=cancel_event):
            yield event
    finally:
        # The try body can resume in a DIFFERENT Context (async/greenlet hop),
        # where reset(token) raises "created in a different Context" and 500s
        # the whole request (live incident 2026-08-06, first cold load after
        # k96). Same-context nesting still uses the token; cross-context falls
        # back to restoring the default explicitly.
        try:
            _NO_MAKEROOM.reset(token)
        except ValueError:
            _NO_MAKEROOM.set(False)
# ---------------------------------------------------------------------------
# Chat continuation engine — one shared implementation.
#
# Hoisted out of the worker agent so local chat and worker chat behave
# identically: both drive this. It wraps the single-pass execute_prompt_stream
# with auto-continuation past the token cap + seam dedup. Because the primitive
# stays single-pass, a remote relay sees a *completed* response (finish=stop)
# and this loop terminates after one pass — no double-continuation.
# ---------------------------------------------------------------------------
from hugpy_engine.schemas.event_schemas import StatusEvent

# How many continuation passes before giving up (runaway guard). Env-overridable;
# the WORKER_* names are honored too so existing worker deployments keep tuning.
_MAX_CONTINUATIONS = int(os.environ.get("HUGPY_MAX_CONTINUATIONS",
                         os.environ.get("WORKER_MAX_CONTINUATIONS", "20")))
# At a seam the model often re-emits the tail of the previous part; drop an
# overlap up to this many chars.
_SEAM_WINDOW = int(os.environ.get("HUGPY_SEAM_WINDOW",
                   os.environ.get("WORKER_SEAM_WINDOW", "400")))
# finish_reasons that mean "ran out of room" -> continue.
_CONTINUE_ON = {"max_tokens", "length"}
# Per-pass output ceiling. This continuation loop delivers totals larger than
# any single pass, so a runner never needs more than one cap's worth at a time.
# Forwarding a huge max_new_tokens straight through also breaks workers on an
# OLDER abstract_hugpy_dev build that *raises* on over-cap instead of clamping
# (the local coder clamps; old remote workers don't). Clamping the per-pass
# value here makes the gateway resilient to that version skew. Matches the
# DeepCoder default cap; override (lower) if a worker runs a smaller cap.
_PER_PASS_MAX_TOKENS = int(os.environ.get("HUGPY_PER_PASS_MAX_TOKENS", "16000"))


def _overlap_len(prev_tail: str, seg: str) -> int:
    """Longest suffix of prev_tail that is also a prefix of seg (seam dedup).

    Exact match — verbatim repetition is by far the common case at a seam.
    """
    maxk = min(len(prev_tail), len(seg))
    for k in range(maxk, 0, -1):
        if prev_tail.endswith(seg[:k]):
            return k
    return 0


async def execute_chat_stream(*args, cancel_event=None, **kwargs):
    """Chat streaming with auto-continuation + seam-dedup.

    Drives execute_prompt_stream (one resolve→build→run pass) repeatedly: when a
    pass stops because it hit the token cap (finish_reason in _CONTINUE_ON), the
    partial answer is appended as an assistant turn and generation continues, so
    a response longer than any single token allowance still completes.

    Yields StreamEvents: TokenEvent for text, StatusEvent between continuation
    segments (and any provisioning/status passthrough from a worker), and a
    single terminal DoneEvent — or ErrorEvent. ``cancel_event`` stops it between
    and during passes.
    """
    prompt_kwargs = normalize_prompt_kwargs(*args, **kwargs)

    rid = prompt_kwargs.get("request_id")
    if not rid:
        import uuid
        rid = uuid.uuid4().hex
    prompt_kwargs["request_id"] = rid  # stable id across all passes

    # Normalize to a messages list so we can append assistant partials.
    messages = prompt_kwargs.get("messages")
    if not messages:
        messages = [{"role": "user", "content": prompt_kwargs.get("prompt", "")}]
    base = {k: v for k, v in prompt_kwargs.items() if k not in ("messages", "prompt")}

    # Clamp the per-pass output budget before any pass (local or worker relay):
    # continuation below covers totals beyond one pass, and this keeps an
    # over-cap value from ever reaching a worker that would raise on it.
    _mnt = base.get("max_new_tokens")
    if isinstance(_mnt, int) and _mnt > _PER_PASS_MAX_TOKENS:
        base["max_new_tokens"] = _PER_PASS_MAX_TOKENS

    # Caller-supplied continuation budget (ChatRequest.max_chunks). This loop
    # is where "Continue exactly where you left off" passes are minted, so an
    # explicit max_chunks MUST bound the TOTAL number of passes here — the
    # runner-level unbounded loop honoring req.max_chunks isn't enough, because
    # a bounded (max_new_tokens) pass that ends finish=length re-enters THIS
    # loop. max_chunks=1 therefore means: one pass, no auto-continuation (the
    # OpenAI /v1 max_tokens contract). Absent/invalid -> today's ceiling.
    _mc = base.get("max_chunks")
    try:
        _mc = int(_mc) if _mc is not None else None
    except (TypeError, ValueError):
        _mc = None
    total_passes = _MAX_CONTINUATIONS + 1
    if _mc and _mc > 0:
        total_passes = min(total_passes, _mc)

    full_text = ""
    # Aggregate token accounting across continuation passes: each inner pass's
    # done event may carry a usage dict (engine-reported or tokenizer-counted
    # by the runner); the request's real cost is their key-wise sum. Stays None
    # when no pass reported — consumers (/v1 usage object) must degrade.
    usage_totals: Optional[dict] = None
    # Engine-measured decode rate across continuation passes. NOT summed, unlike
    # usage above: tok/s is a RATE and adding two rates is meaningless (two
    # passes at 115 tok/s is 115 tok/s, not 230). Every pass re-measures the same
    # steady-state speed on the same placement, so last-reported wins — it is
    # both representative and the freshest. None when no pass reported.
    timings_last: Optional[dict] = None

    def _merge_usage(part):
        nonlocal usage_totals
        if not isinstance(part, dict) or not part:
            return
        totals = dict(usage_totals or {})
        for k, v in part.items():
            if isinstance(v, int):
                totals[k] = (totals.get(k) or 0) + v
        usage_totals = totals or None

    for attempt in range(total_passes):
        if cancel_event is not None and cancel_event.is_set():
            yield DoneEvent(request_id=rid, input_tokens=0, output_chunks=1,
                            finish_reason="cancelled")
            return
        if attempt > 0:
            yield StatusEvent(type="status", request_id=rid, stage="generate",
                              message=f"continuing (part {attempt + 1})…",
                              segment=attempt + 1)

        # Seam dedup: on a continuation pass, buffer the head of the segment
        # until we have _SEAM_WINDOW chars (or the pass ends), strip any overlap
        # with what we already emitted, then stream the rest live.
        is_cont = attempt > 0
        prev_tail = full_text[-_SEAM_WINDOW:] if is_cont else ""
        buffering = is_cont
        head = ""
        seg_text = ""
        finish = "stop"
        errored = False

        async for event in execute_prompt_stream(messages=messages,
                                                 cancel_event=cancel_event, **base):
            etype = getattr(event, "type", None)
            if etype == "token":
                text = getattr(event, "text", "") or ""
                seg_text += text
                if buffering:
                    head += text
                    if len(head) < _SEAM_WINDOW:
                        continue
                    k = _overlap_len(prev_tail, head)
                    emit, head, buffering = head[k:], "", False
                    if emit:
                        full_text += emit
                        yield TokenEvent(request_id=rid, text=emit)
                elif text:
                    full_text += text
                    yield TokenEvent(request_id=rid, text=text)
            elif etype == "done":
                finish = getattr(event, "finish_reason", None) or "stop"
                _merge_usage(getattr(event, "usage", None))
                _t = getattr(event, "timings", None)
                if isinstance(_t, dict) and _t:
                    timings_last = _t
            elif etype == "error":
                # A pass that dies after text already streamed shouldn't turn a
                # partially-delivered answer into "[Error: ...]" in the chat.
                # This happens for real: a rambling model (e.g. a text-encoder
                # repack that never stops thinking) trips the engine mid-stream
                # (context overrun, decode assert) and the server aborts the
                # response body. End gracefully: an honest "truncated" status +
                # a normal done, with the failure logged. Only an error with
                # NOTHING delivered is surfaced as an error.
                if full_text.strip():
                    logger.warning(
                        "pass %s failed (%s); ending %s gracefully with %d "
                        "chars already streamed", attempt + 1,
                        getattr(event, "message", None) or "run failed",
                        rid, len(full_text))
                    yield StatusEvent(type="status", request_id=rid,
                                      stage="generate",
                                      message="engine stream ended early — "
                                              "answer truncated")
                    yield DoneEvent(request_id=rid, input_tokens=0,
                                    output_chunks=1, finish_reason="stop",
                                    usage=usage_totals, timings=timings_last)
                else:
                    yield ErrorEvent(request_id=rid,
                                     message=getattr(event, "message", None) or "run failed")
                errored = True
                break
            else:
                # status / provisioning passthrough (e.g. relayed from a worker)
                yield event

        if errored:
            return

        # Pass ended while still buffering (short segment): flush remainder
        # minus the seam overlap.
        if buffering:
            k = _overlap_len(prev_tail, head)
            emit = head[k:]
            if emit:
                full_text += emit
                yield TokenEvent(request_id=rid, text=emit)

        if finish not in _CONTINUE_ON:
            yield DoneEvent(request_id=rid, input_tokens=0, output_chunks=1,
                            finish_reason=finish, usage=usage_totals, timings=timings_last)
            return
        if not seg_text.strip():
            # Hit the cap but produced nothing usable — stop to avoid a loop.
            yield DoneEvent(request_id=rid, input_tokens=0, output_chunks=1,
                            finish_reason="stop", usage=usage_totals, timings=timings_last)
            return

        # Continue: append the partial assistant turn and re-prompt to keep going.
        messages = messages + [
            {"role": "assistant", "content": seg_text},
            {"role": "user", "content": "Continue exactly where you left off. "
                                        "Do not repeat any previous text."},
        ]

    # Exhausted the continuation budget.
    yield DoneEvent(request_id=rid, input_tokens=0, output_chunks=1,
                    finish_reason="max_tokens", usage=usage_totals, timings=timings_last)


# ---------------------------------------------------------------------------
# Inspection / lifecycle — single definition each, no duplicates.
# ---------------------------------------------------------------------------

def loaded_model_keys() -> List[Tuple[str, str]]:
    """Which (model_key, task) pairs currently have a runner instantiated."""
    with _INSTANCES_LOCK:
        return sorted(_INSTANCES.keys())


# Model dirs are immutable once pulled, so walk each once and memoize. The key
# is (path, chosen_gguf): WHICH quant serves is operator-mutable (the gguf_file
# override), so a plain path key held the stale weight_bytes forever after a
# re-pin — keying on the chosen artifact makes an override change miss the
# cache naturally (set_override already drops the physical record; this cache
# re-derives on the next read because the chosen basename changed).
_DISK_DETAIL_CACHE: Dict[tuple, dict] = {}
_WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt", ".onnx")


def _dir_size_detail(path: str, model_key: Optional[str] = None,
                     cfg: Optional[dict] = None) -> dict:
    """Recursively size a model dir: total on-disk bytes + weight-file bytes.

    ``model_bytes`` is the WHOLE-DIR total (the row's on-disk size, contract
    unchanged). ``weight_bytes`` is what the framework will actually pull into
    memory — and for gguf/llama_cpp that is the SINGLE designated/elected quant
    (+ mmproj, shards summed), NOT every quant in the dir: a 24-quant repo used
    to report its full ~53GB litter as the expected-VRAM proxy. Resolved via
    ``effective_load_requirement`` when (model_key, cfg) are supplied; non-GGUF
    (or no model identity) keeps the recursive weight-ext sum exactly as
    before. Cached; returns {} for a missing/unreadable dir (caller degrades)."""
    chosen = None
    eff_weight = None
    framework = ""
    if isinstance(cfg, dict):
        framework = str(cfg.get("framework") or "").lower()
    elif cfg is not None:
        framework = str(getattr(cfg, "framework", "") or "").lower()
    if model_key and framework in ("gguf", "llama_cpp"):
        try:
            from hugpy_engine.serve.overrides import effective_load_requirement
            req = effective_load_requirement(model_key, path, cfg) or {}
            chosen = req.get("chosen_gguf")
            eff_weight = req.get("weights_bytes")
        except Exception:  # noqa: BLE001 — sizing refinement is best-effort
            chosen = eff_weight = None
    cache_key = (path, chosen)
    cached = _DISK_DETAIL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    total = weight = 0
    try:
        for root, _dirs, files in os.walk(path):
            for fn in files:
                try:
                    sz = os.path.getsize(os.path.join(root, fn))
                except OSError:
                    continue
                total += sz
                if fn.lower().endswith(_WEIGHT_EXTS):
                    weight += sz
    except OSError:
        return {}
    if eff_weight:
        weight = int(eff_weight)             # the artifact that loads, not the litter
    out: dict = {}
    if total:
        out["model_bytes"] = total          # frontend renders this as the row's size
    if weight:
        out["weight_bytes"] = weight         # expected-VRAM proxy
    if out:
        _DISK_DETAIL_CACHE[cache_key] = out
    return out


def loaded_disk_detail() -> dict:
    """Per-loaded-model on-disk size for EVERY framework (transformers, diffusers,
    llama_cpp) — keyed by model_key.

    ``loaded_runner_detail`` only sizes in-process GGUF runners, so non-GGUF
    serving rows carried no size at all. This walks each loaded model's dir
    (resolved the same way the puller/loader do, via ``route_destination``) so
    every serving row gets a size server-side — no per-browser computation.
    GGUF rows are refined afterward by ``loaded_runner_detail`` (exact file
    bytes + layer split), which overlays this."""
    out: dict = {}
    try:
        from hugpy_storage.model_paths import route_destination
        from hugpy_engine.config.main import get_model_config
    except Exception:
        return out
    for (mk, _task) in loaded_model_keys():
        if mk in out:
            continue
        try:
            cfg = get_model_config(mk, dict_return=True)
            path = route_destination(cfg)
        except Exception:
            continue
        d = _dir_size_detail(path, model_key=mk, cfg=cfg)
        if d:
            out[mk] = d
    return out


def evict(model_key: str, task: Optional[str] = None) -> bool:
    """Drop runner(s) from the cache AND free the model's weights.

    If task is None, all task-variants for that model_key are dropped.
    Returns True if anything was evicted.

    Popping the wrapper alone is not enough: the llama.cpp singleton
    (_LLAMA_INSTANCES) holds the loaded weights, so without the cascade the
    VRAM/RAM stayed pinned after "unload" until the process died.
    """
    with _INSTANCES_LOCK:
        if task is not None:
            dropped = _INSTANCES.pop((model_key, task), None) is not None
        else:
            to_drop = [k for k in list(_INSTANCES) if k[0] == model_key]
            for k in to_drop:
                _INSTANCES.pop(k, None)
            dropped = bool(to_drop)
    try:
        from hugpy_engine.llama.runners.get import evict_llama_runner
        heavy = evict_llama_runner(model_key)
    except Exception:
        heavy = False
    return dropped or heavy


def clear() -> None:
    """Drop all cached runners (and their loaded weights)."""
    with _INSTANCES_LOCK:
        _INSTANCES.clear()
    try:
        from hugpy_engine.llama.runners.get import clear_llama_runners
        clear_llama_runners()
    except Exception:
        pass


def supported_task_keys() -> List[Tuple[str, str]]:
    # (framework, task) pairs a runner is registered for — engine chat runners
    # plus whatever media/video plugged in (hugpy_engine.tasks).
    from hugpy_engine.resolvers.categories import FRAMEWORK_RUNNERS
    return sorted(FRAMEWORK_RUNNERS)
