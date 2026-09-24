"""ComfyUI idle-VRAM watchdog — a stale comfy process never squats the card (k54).

THE INCIDENT (operator, 2026-07-31). A ComfyUI pid held 2874 MiB for 61 HOURS
after its last render, with the comfy queue empty the whole time. Nothing
reclaimed it: comfy is an EXTERNAL, adopted process, so it is out of hugpy's
allocations (0.1.137) and every eviction path deliberately skips it ("comfy has
its own headroom path"). That exclusion is right for *contention* — the worker
does not own comfy's residency policy — but it left NO path at all for the
opposite case: comfy holding VRAM it is provably not using. This module is that
path, and it closes the known ``comfy-process-vram-evict-gap`` (eviction blind
to out-of-band process VRAM).

THE MECHANISM, verified live: ``POST /free {"unload_models": true,
"free_memory": true}`` dropped the same process from 2874 MiB to 378 MiB — the
bare CUDA context floor — with the server still healthy and ready for the next
job. It is comfy's OWN API, so this is never a PID kill: the worker asks, comfy
releases. An adopted external process is NEVER SIGKILLed automatically (it may
be mid-render despite appearances); a /free that doesn't take is SURFACED
(log + telemetry), not escalated.

THE IDLE PREDICATE — all four clauses, ANDed:
  1. the comfy process holds VRAM ABOVE the ~400 MiB CUDA-context floor
     (at/below the floor there is nothing to reclaim);
  2. no active/registered comfy call in the pid registry's foreign-call table
     (the call-time attribution the runner records at dispatch);
  3. ComfyUI's own ``/queue`` reports ``queue_running`` AND ``queue_pending``
     empty — comfy's truth, not ours;
  4. that state has PERSISTED past an idle TTL (default 10 min, env
     ``HUGPY_COMFY_IDLE_FREE_S``).

Clause 4 is a DEBOUNCE against racing an about-to-start render — not a
protection class. Freshness is RANK, never a veto (no-timeblock-on-eviction):
under CONTENTION, when an LLM load actually needs the bytes, ``reclaim()`` drops
clause 4 entirely and frees on 1-3 alone (minimize-loading doctrine: contention
beats idleness). Only the FIRST clause is inviolable — you cannot reclaim what
isn't held.

DEGRADE-TO-NO-OP, like every other worker-side measurement here. Any clause we
cannot PROVE reads as "not idle": an unreachable ``/queue``, an unmeasurable
VRAM figure, a missing pid registry — all leave comfy alone. Freeing a busy
comfy would kill a render; leaving a squatter costs VRAM the next beat retries
for. The asymmetry is deliberate.

SELF-CONTAINED + INJECTABLE. This module imports nothing from ``agent.py``
(whose import pulls torch and the whole runner stack) — every box-touching
capability (the comfy VRAM read, the base URL, the /free call, the clock, the
settle sleep) is a constructor arg, so every path is unit-testable with no GPU
and no ComfyUI. The worker wires the real probes at boot.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_MIB = 1024 * 1024

# How long the idle state must PERSIST before the watchdog frees — the debounce
# against racing a render that is about to start (a queued job can appear
# between two beats). Not a protection class: `reclaim()` ignores it entirely.
_DEFAULT_IDLE_TTL_S = 600.0

# The CUDA context floor. A live ComfyUI server that has unloaded everything
# still holds its bare context (378 MiB measured on ae's 3090); anything at or
# under this is NOT a squatter and there is nothing to reclaim. Deliberately a
# round number just above the measurement — the reclaim is worth doing only for
# real model bytes.
_DEFAULT_FLOOR_MIB = 400

# How long to let comfy actually hand the memory back before we re-measure. The
# /free response returns before the allocator has finished releasing, so a
# same-instant re-read would report a false "it didn't work".
_DEFAULT_SETTLE_S = 2.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


def idle_ttl_s() -> float:
    """Idle debounce before the watchdog frees, ``HUGPY_COMFY_IDLE_FREE_S``
    (seconds). 0 / negative means "free as soon as the other clauses hold"."""
    return _env_float("HUGPY_COMFY_IDLE_FREE_S", _DEFAULT_IDLE_TTL_S)


def context_floor_bytes() -> int:
    """The bare-CUDA-context floor in bytes, ``HUGPY_COMFY_VRAM_FLOOR_MIB``."""
    return int(_env_float("HUGPY_COMFY_VRAM_FLOOR_MIB", _DEFAULT_FLOOR_MIB) * _MIB)


def settle_s() -> float:
    """Post-/free settle before re-measuring, ``HUGPY_COMFY_FREE_SETTLE_S``."""
    return _env_float("HUGPY_COMFY_FREE_SETTLE_S", _DEFAULT_SETTLE_S)


def enabled() -> bool:
    """The watchdog kill switch (``HUGPY_COMFY_IDLE_FREE=0`` disables it).
    Default ON: a stale comfy squatter is the operator's directive to remove,
    and the whole predicate degrades to a no-op on a box with no comfy."""
    v = (os.environ.get("HUGPY_COMFY_IDLE_FREE") or "").strip().lower()
    return v not in ("0", "false", "no", "off")


def queue_state(url: str, client=None, timeout: float = 3.0) -> "Optional[dict]":
    """``{"running": int, "pending": int}`` from ComfyUI's ``GET /queue``, or
    ``None`` when we cannot read it (comfy down, a non-200, an unparseable body).

    ``None`` is NOT "idle" — the caller treats an unreadable queue as busy. This
    is comfy's OWN account of its work; we never infer emptiness from our side of
    the wire alone."""
    if not url:
        return None
    try:
        import httpx
    except Exception:  # noqa: BLE001 — no httpx on this box: can't prove idle
        return None
    own = client is None
    if own:
        client = httpx.Client(timeout=timeout)
    try:
        r = client.get(url.rstrip("/") + "/queue")
        if r.status_code != 200:
            return None
        body = r.json() or {}
        return {"running": len(body.get("queue_running") or []),
                "pending": len(body.get("queue_pending") or [])}
    except Exception:  # noqa: BLE001 — unreachable / parse error: can't prove idle
        return None
    finally:
        if own:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass


def active_comfy_call() -> "Optional[dict]":
    """The pid registry's live comfy foreign-call record, or ``None``.

    A call recorded at dispatch (``comfy_runner._reg_comfy_call``) means THIS
    worker has a generation in flight against comfy — an absolute veto on
    freeing. A registry that can't be read raises nothing; it returns a sentinel
    the caller reads as "cannot prove there is no call"."""
    from hugpy_fleet.worker import pid_registry
    return pid_registry.active_foreign_call("comfy")


class _Unknown:
    """Sentinel: the call table could not be read (≠ 'no call')."""


UNKNOWN = _Unknown()


class ComfyIdleWatchdog:
    """The idle-comfy reclaimer. One per worker; holds only the idle CLOCK.

    Probes (all injectable, all called with no arguments unless noted):
      * ``vram_probe(fresh: bool = False)`` -> comfy's VRAM in bytes, or None
        when unmeasurable. ``fresh=True`` must bypass any nvidia-smi cache (the
        post-/free re-measure would otherwise read the pre-/free value).
      * ``url_probe()``   -> the adopted comfy base URL.
      * ``free_call()``   -> ``(ok: bool, note: str)`` — comfy's POST /free.
      * ``queue_probe(url)`` -> ``{"running", "pending"}`` | None.
      * ``call_probe()``  -> the active foreign call dict | None | UNKNOWN.
      * ``emit(stage, **fields)`` -> eviction telemetry (best-effort).
    """

    def __init__(self, vram_probe: Callable[..., "Optional[int]"],
                 url_probe: Callable[[], str],
                 free_call: Callable[[], "tuple[bool, str]"],
                 queue_probe: "Optional[Callable[[str], Optional[dict]]]" = None,
                 call_probe: "Optional[Callable[[], object]]" = None,
                 emit: "Optional[Callable[..., None]]" = None,
                 clock: "Optional[Callable[[], float]]" = None,
                 sleep: "Optional[Callable[[float], None]]" = None) -> None:
        self._vram_probe = vram_probe
        self._url_probe = url_probe
        self._free_call = free_call
        self._queue_probe = queue_probe or queue_state
        self._call_probe = call_probe or self._default_call_probe
        self._emit = emit or _default_emit
        self._clock = clock or time.time
        self._sleep = sleep or time.sleep
        # When the CURRENT unbroken idle streak began, or None when comfy is not
        # (provably) idle. Reset on every non-idle observation, which is what
        # makes the TTL a persistence test rather than a stopwatch on the box.
        self._idle_since: "Optional[float]" = None
        self._last_free_at: float = 0.0

    # -- observation ---------------------------------------------------------
    @staticmethod
    def _default_call_probe() -> object:
        try:
            return active_comfy_call()
        except Exception:  # noqa: BLE001 — unreadable table: never assume "no call"
            return UNKNOWN

    def observe(self, fresh: bool = False) -> dict:
        """Evaluate clauses 1-3 (VRAM above floor / no registered call / comfy's
        queue empty) and maintain the idle clock. Returns
        ``{"idle", "why", "vram_bytes", "queue", "idle_since", "idle_for_s"}``.

        ``why`` is the clause that decided it — the honest line for the log and
        the telemetry, never a bare "not idle"."""
        floor = context_floor_bytes()
        try:
            # ``fresh`` is optional in the probe contract: a plain
            # zero-argument probe (the common test shape) is accepted too.
            try:
                vram = self._vram_probe(fresh=fresh)
            except TypeError:
                vram = self._vram_probe()
        except Exception:  # noqa: BLE001 — unmeasurable: leave comfy alone
            vram = None
        now = self._clock()
        if vram is None:
            return self._not_idle(now, "comfy VRAM unmeasurable (no GPU / no comfy proc)",
                                  None, None)
        if int(vram) <= floor:
            # At the context floor there is nothing to reclaim. Not a failure —
            # the wanted steady state.
            return self._not_idle(
                now, f"at/below the CUDA context floor ({int(vram) // _MIB} MiB "
                     f"<= {floor // _MIB} MiB) — nothing to reclaim", int(vram), None)
        call = self._call_probe()
        if call is UNKNOWN:
            return self._not_idle(now, "comfy call table unreadable — cannot prove idle",
                                  int(vram), None)
        if call:
            mk = (call or {}).get("model_key") or "?"
            return self._not_idle(now, f"a comfy call is in flight ({mk})",
                                  int(vram), None)
        q = self._queue_probe(self._url_probe())
        if q is None:
            return self._not_idle(now, "comfy /queue unreadable — cannot prove idle",
                                  int(vram), None)
        if int(q.get("running") or 0) or int(q.get("pending") or 0):
            return self._not_idle(
                now, f"comfy queue busy (running={q.get('running')}, "
                     f"pending={q.get('pending')})", int(vram), q)
        if self._idle_since is None:
            self._idle_since = now
        return {"idle": True, "why": "no call, empty queue, VRAM above the context floor",
                "vram_bytes": int(vram), "queue": q,
                "idle_since": self._idle_since,
                "idle_for_s": max(0.0, now - self._idle_since)}

    def _not_idle(self, now: float, why: str, vram: "Optional[int]",
                  queue: "Optional[dict]") -> dict:
        self._idle_since = None
        return {"idle": False, "why": why, "vram_bytes": vram, "queue": queue,
                "idle_since": None, "idle_for_s": 0.0}

    # -- the two entry points ------------------------------------------------
    def tick(self, free_vram_bytes: "Optional[int]" = None,
             total_vram_bytes: "Optional[int]" = None) -> dict:
        """The IDLE-TTL path — the residency beat's watchdog pass.

        Frees only when the idle predicate holds AND has persisted past
        ``idle_ttl_s()``. A no-op (and silent — no telemetry at all) on a box
        where comfy holds nothing, which is every box without comfy."""
        if not enabled():
            return {"action": "skip", "reason": "watchdog disabled "
                                                "(HUGPY_COMFY_IDLE_FREE=0)"}
        obs = self.observe()
        if not obs["idle"]:
            return {"action": "skip", "reason": obs["why"],
                    "vram_bytes": obs.get("vram_bytes")}
        ttl = idle_ttl_s()
        if obs["idle_for_s"] < ttl:
            return {"action": "wait", "reason": (
                f"idle for {obs['idle_for_s']:.0f}s of the {ttl:.0f}s debounce"),
                "vram_bytes": obs.get("vram_bytes"),
                "idle_for_s": obs["idle_for_s"]}
        logger.info(
            "comfy idle watchdog: ComfyUI has held %s with an empty queue and no "
            "registered call for %.0fs (TTL %.0fs) — asking comfy to free it "
            "(POST /free; never a kill)",
            _human(obs["vram_bytes"]), obs["idle_for_s"], ttl)
        return self._do_free(obs, trigger="comfy-idle", incoming_model=None,
                             free_vram_bytes=free_vram_bytes,
                             total_vram_bytes=total_vram_bytes,
                             note=f"idle {obs['idle_for_s']:.0f}s >= TTL {ttl:.0f}s")

    def reclaim(self, incoming_model: "Optional[str]" = None,
                need_bytes: "Optional[int]" = None,
                trigger: str = "contention") -> dict:
        """The CONTENTION path — an LLM load / headroom sweep needs the bytes.

        Same predicate MINUS the TTL: freshness is rank, not a veto, and a
        resident that nothing is using loses to a load that is (minimize-loading
        doctrine). Clauses 1-3 still bind absolutely — an in-flight comfy call or
        a non-empty comfy queue is never overridden by contention, because that
        would kill a render, and no LLM load is worth that."""
        if not enabled():
            return {"action": "skip", "reason": "watchdog disabled "
                                                "(HUGPY_COMFY_IDLE_FREE=0)"}
        obs = self.observe()
        if not obs["idle"]:
            return {"action": "skip", "reason": obs["why"],
                    "vram_bytes": obs.get("vram_bytes")}
        logger.info(
            "comfy reclaim under contention: ComfyUI holds %s idle (empty queue, "
            "no registered call) and %s needs room — asking comfy to free it "
            "(TTL waived: contention beats idleness)",
            _human(obs["vram_bytes"]), incoming_model or "a pending load")
        return self._do_free(obs, trigger=trigger, incoming_model=incoming_model,
                             need_bytes=need_bytes,
                             note="contention — idle TTL waived")

    # -- the free itself -----------------------------------------------------
    def _do_free(self, obs: dict, trigger: str, incoming_model: "Optional[str]",
                 need_bytes: "Optional[int]" = None,
                 free_vram_bytes: "Optional[int]" = None,
                 total_vram_bytes: "Optional[int]" = None,
                 note: "Optional[str]" = None) -> dict:
        """POST /free, settle, RE-MEASURE, report. The re-measure is the point:
        "comfy accepted /free" is not evidence that VRAM came back, and the
        operator's question is about bytes, not HTTP codes."""
        before = int(obs.get("vram_bytes") or 0)
        started = self._clock()
        self._emit("headroom.start", trigger=trigger, incoming_model=incoming_model,
                   need_bytes=need_bytes, free_bytes=free_vram_bytes,
                   total_bytes=total_vram_bytes, note=note)
        self._emit("evict.start", model_key="comfy", tier="comfy", trigger=trigger,
                   incoming_model=incoming_model, vram_bytes=before,
                   idle_for_s=int(obs.get("idle_for_s") or 0))
        try:
            ok, why = self._free_call()
        except Exception as exc:  # noqa: BLE001 — a broken /free never breaks a beat
            ok, why = False, f"{type(exc).__name__}: {exc}"
        dur_ms = int((self._clock() - started) * 1000)
        if not ok:
            # SURFACE, never escalate: the worker does not own comfy's process, so
            # a failed /free is reported and retried next beat — it is NEVER
            # followed by a SIGKILL of an adopted external process.
            logger.warning("comfy idle watchdog: /free did not take (%s) — "
                           "surfacing it; an adopted external process is never "
                           "killed automatically", why)
            self._emit("evict.fail", model_key="comfy", tier="comfy",
                       trigger=trigger, incoming_model=incoming_model,
                       duration_ms=dur_ms, error=str(why))
            self._emit("headroom.done", trigger=trigger, incoming_model=incoming_model,
                       evicted=[], outcome="proceeded-unfit", note=str(why))
            return {"action": "failed", "reason": why, "before_bytes": before,
                    "freed_bytes": 0}
        try:
            self._sleep(settle_s())
        except Exception:  # noqa: BLE001
            pass
        after = self.observe(fresh=True).get("vram_bytes")
        self._last_free_at = self._clock()
        # A fresh re-measure that can't read comfy at all means the process holds
        # no GPU memory any more — the whole point. Treat it as fully freed.
        after_i = int(after) if after is not None else 0
        freed = max(0, before - after_i)
        dur_ms = int((self._clock() - started) * 1000)
        if after_i > context_floor_bytes():
            # /free was accepted and bytes did NOT come back to the floor. Real,
            # reportable, and still not a kill: comfy may be holding something we
            # cannot see. Surface it and let the next beat try again.
            logger.warning(
                "comfy idle watchdog: /free accepted but ComfyUI still holds %s "
                "(freed %s, floor %s) — surfacing; not killing the process",
                _human(after_i), _human(freed), _human(context_floor_bytes()))
            self._emit("evict.fail", model_key="comfy", tier="comfy",
                       trigger=trigger, incoming_model=incoming_model,
                       freed_bytes=freed, duration_ms=dur_ms,
                       error=(f"/free accepted but {_human(after_i)} still held "
                              f"(above the {_human(context_floor_bytes())} floor)"))
            self._emit("headroom.done", trigger=trigger, incoming_model=incoming_model,
                       evicted=[], outcome="proceeded-unfit")
            return {"action": "partial", "reason": "VRAM stayed above the floor",
                    "before_bytes": before, "after_bytes": after_i,
                    "freed_bytes": freed}
        logger.info("comfy idle watchdog: reclaimed %s from ComfyUI (%s -> %s, "
                    "server still up)", _human(freed), _human(before),
                    _human(after_i))
        self._emit("evict.done", model_key="comfy", tier="comfy", trigger=trigger,
                   incoming_model=incoming_model, freed_bytes=freed,
                   duration_ms=dur_ms)
        self._emit("reclaim.done", incoming_model=incoming_model)
        self._emit("headroom.done", trigger=trigger, incoming_model=incoming_model,
                   evicted=["comfy"], outcome="fit", freed_bytes=freed)
        return {"action": "freed", "reason": why, "before_bytes": before,
                "after_bytes": after_i, "freed_bytes": freed}


def _default_emit(stage: str, **fields) -> None:
    """Best-effort telemetry emit — the same contract every other emitter here
    honours: observation only, a failure is swallowed."""
    try:
        from hugpy_fleet.central import evictions as _evt
        _evt.emit_eviction_event(stage, **fields)
    except Exception:  # noqa: BLE001 — telemetry never disturbs a reclaim
        pass


from hugpy_platform.formatting import human_bytes as _human
