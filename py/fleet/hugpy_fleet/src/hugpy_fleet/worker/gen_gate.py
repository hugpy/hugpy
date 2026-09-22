"""Per-model in-process generation gate (worker-side concurrency hardening).

The failure class this closes (real incident, computron 2026-07-11): a batch
client fired concurrent ``/v1/chat/completions`` at a model served IN-PROCESS by
llama-cpp-python. A llama.cpp ``Llama`` context is NOT concurrency-safe — two
threads decoding the same context race in native code and the whole worker
process SEGV/ABRTs, killing every other in-flight stream on the box. There was
no generation lock: ``dispatch._INSTANCES_LOCK`` guards instance CREATION only,
not the generate/stream itself.

This module is that lock. It serializes entry into an in-process runner PER
MODEL, held for the ENTIRE generation including the full life of a streamed
response, with a bounded wait so callers never pile up unboundedly — on timeout
we return an honest, structured busy error (never a silent hang, never a crash).

What is and isn't gated:
  * IN-PROCESS runners (llama-cpp-python, in-process transformers) — GATED. Their
    true safe concurrency is 1, so the default per-model limit is 1.
  * SLOT-relayed requests (the worker proxying to its own llama-server /
    llama_cpp.server child) — NOT gated. The child process has a real scheduler
    and handles parallel requests itself; serializing them here would only add
    latency. ``should_gate`` returns False for a model currently seated in a slot.

v0 scope: the gate is per-WORKER-PROCESS (module-global). A worker agent runs a
single Flask process, so that is exactly the right granularity — every request
that reaches this box's in-process runner passes through this one registry.

Env knobs (read lazily so a re-exec / test can retune):
  * ``HUGPY_INPROCESS_MAX_CONCURRENCY`` — safe concurrent entrants per in-process
    model. Default 1 (llama.cpp / transformers truth). Advertised to central in
    the heartbeat as ``serving_limits.in_process_max_concurrency``.
  * ``HUGPY_WORKER_GEN_GATE_TIMEOUT_S`` — acquire wait. Default 0 (unbounded
    queue); a positive value opts into a timeout.
  * ``HUGPY_WORKER_GEN_GATE`` — set ``off``/``0``/``false`` to disable gating
    entirely (emergency escape hatch; the crash returns with it off).
"""
from __future__ import annotations

import os
import threading
import time
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (read lazily — a /ops/config re-exec or a test can change it).
# ---------------------------------------------------------------------------
def concurrency_limit() -> int:
    """Safe concurrent entrants into ONE in-process model runner.

    Default 1: a llama.cpp ``Llama`` context and an in-process transformers model
    are single-threaded for generation. An operator who KNOWS a runner is
    reentrant can raise it, but the honest default is the one that never crashes.
    """
    try:
        n = int(os.environ.get("HUGPY_INPROCESS_MAX_CONCURRENCY", "1"))
    except (TypeError, ValueError):
        n = 1
    return max(1, n)


def gate_timeout_s() -> float:
    try:
        t = float(os.environ.get("HUGPY_WORKER_GEN_GATE_TIMEOUT_S", "0"))
    except (TypeError, ValueError):
        t = 0.0
    return max(0.0, t)


def _gate_disabled() -> bool:
    return os.environ.get("HUGPY_WORKER_GEN_GATE", "").strip().lower() in (
        "off", "0", "false", "no",
    )


# ---------------------------------------------------------------------------
# Honest busy error — matches the worker's structured error envelope
# ({"ok": False, "error": {"code", "message", ...}}), the shape the /ops/* and
# /infer error paths already return, so central + direct clients parse it the
# same way.
# ---------------------------------------------------------------------------
class ModelBusy(Exception):
    """Raised when the per-model gate can't be acquired within the bounded wait.

    Carries the data central/clients need to degrade honestly: which model, how
    many requests are already in the runner, and how long we waited.
    """

    def __init__(self, model_key: Optional[str], in_flight: int, waited_s: float):
        self.model_key = model_key
        self.in_flight = int(in_flight)
        self.waited_s = round(float(waited_s), 3)
        super().__init__(
            f"model '{model_key}' is busy: {in_flight} request(s) already in the "
            f"in-process runner (llama.cpp/transformers serialize per model); "
            f"waited {self.waited_s}s")

    def as_error(self, worker: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """The JSON body for a 503 response (worker error-envelope idiom)."""
        body: Dict[str, Any] = {
            "ok": False,
            "error": {
                "code": "model_busy",
                "message": str(self),
                "model_key": self.model_key,
                "in_flight": self.in_flight,
                "waited_s": self.waited_s,
            },
        }
        if worker:
            body["worker"] = worker
        return body


class ModelCancelled(Exception):
    """A request was cancelled while waiting for its model gate."""


# ---------------------------------------------------------------------------
# The gate registry — one bounded gate per model_key.
# ---------------------------------------------------------------------------
class _Gate:
    """A per-model bounded-concurrency gate with an honest active-count.

    ``BoundedSemaphore`` enforces the limit and gives a native bounded-wait
    acquire; ``_active`` is our own truthful count of entrants currently inside
    the runner (reported in a ModelBusy so the caller can say how busy it is).
    """

    def __init__(self, limit: int):
        self._sem = threading.BoundedSemaphore(limit)
        self._lock = threading.Lock()
        self._active = 0
        self.limit = limit

    def acquire(self, model_key: Optional[str], timeout_s: float,
                cancel_event: Optional[threading.Event] = None) -> None:
        start = time.monotonic()
        deadline = start + timeout_s if timeout_s > 0 else None
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise ModelCancelled(model_key)
            wait = .5 if deadline is None else max(0.0, min(.5, deadline - time.monotonic()))
            if self._sem.acquire(timeout=wait):
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise ModelBusy(model_key, self.active(), time.monotonic() - start)
        with self._lock:
            self._active += 1

    def release(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1
        # Only release the semaphore for a real prior acquire (the token guards
        # against a double-release, so this never over-releases past the bound).
        self._sem.release()

    def active(self) -> int:
        with self._lock:
            return self._active


_gates: Dict[str, _Gate] = {}
_reg_lock = threading.Lock()


def _gate_for(model_key: str) -> _Gate:
    with _reg_lock:
        g = _gates.get(model_key)
        if g is None:
            g = _Gate(concurrency_limit())
            _gates[model_key] = g
        return g


def in_flight(model_key: str) -> int:
    """Requests currently executing inside ``model_key``'s in-process runner."""
    with _reg_lock:
        g = _gates.get(model_key)
    return g.active() if g is not None else 0


def total_in_flight() -> int:
    """Total in-process generations currently executing across ALL models.

    The sum of every gate's active entrant count — i.e. how many native
    generate/stream calls are in flight this instant. The worker's restart path
    uses this to DRAIN before exiting: a restart waits (bounded) for this to hit
    0 so it never tears a native llama.cpp/transformers call out from under a
    live request. Lock-safe against concurrent acquire/release.
    """
    with _reg_lock:
        gates = list(_gates.values())
    return sum(g.active() for g in gates)


# ---------------------------------------------------------------------------
# Slot awareness — a model seated in a slot child is NOT gated here.
# ---------------------------------------------------------------------------
def should_gate(model_key: Optional[str]) -> bool:
    """Whether a request for ``model_key`` must pass through the in-process gate.

    False when gating is disabled, or when the model is currently slot-backed
    (its generation runs in a separate llama-server child that schedules itself).
    Otherwise True — and it FAILS TOWARD gating: any doubt about slot residency
    serializes (safe, a touch slower) rather than risking concurrent native
    entry (a crash). ``model_key`` None/empty is gated under a shared fallback
    key — an un-keyed request could still be in-process.
    """
    if _gate_disabled():
        return False
    if not model_key:
        return True
    try:
        # Heavy import (drags llama/torch); lazy + defensive. A slot-backed model
        # leaves a HOLLOW in-process runner that only proxies to the child, so it
        # is safe to run concurrently — the child serializes/parallelizes itself.
        from hugpy_engine.llama.runners.get import slot_backed_model_keys
        slot_keys = slot_backed_model_keys() or set()
    except Exception:  # noqa: BLE001 — can't tell -> gate (fail safe)
        return True
    if model_key in slot_keys:
        return False
    tail = str(model_key).split("/")[-1]
    if any(tail == str(k).split("/")[-1] for k in slot_keys):
        return False
    return True


# ---------------------------------------------------------------------------
# Public acquire surface. Two shapes for two call sites:
#   * a Token (acquire now, release later) for the streaming path, where the
#     hold must span a generator handed back to Flask; and
#   * a context manager for the one-shot path.
# Both raise ModelBusy on a bounded-wait timeout.
# ---------------------------------------------------------------------------
class Token:
    """A held (or no-op) gate reservation. ``release()`` is idempotent."""

    __slots__ = ("_gate", "_released")

    def __init__(self, gate: Optional[_Gate]):
        self._gate = gate
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._gate is not None:
            try:
                self._gate.release()
            except Exception:  # noqa: BLE001 — release must never raise into a finally
                logger.exception("gen_gate: release failed")

    # Also usable as a context manager for symmetry.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


_NULL_TOKEN = Token(None)


def _model_key_of(payload: Dict[str, Any]) -> Optional[str]:
    """The gate key for a request payload. llama.cpp keys one Llama context per
    model, so model_key identifies the shared native state to serialize on. Fall
    back to task (then a shared sentinel) so an un-keyed request is still gated —
    it could reach an in-process runner too."""
    mk = payload.get("model_key")
    if mk:
        return str(mk)
    task = payload.get("task")
    return f"__task__:{task}" if task else "__ungated_default__"


def acquire_for_payload(payload: Dict[str, Any], *,
                        timeout_s: Optional[float] = None,
                        cancel_event: Optional[threading.Event] = None) -> Token:
    """Bounded-acquire the gate for this request payload; return a Token.

    Returns the shared no-op token when the model is slot-backed or gating is
    disabled (nothing to serialize). Raises ModelBusy on timeout. The caller
    MUST call ``token.release()`` when the generation (incl. the whole stream)
    finishes — do it in a ``finally``.
    """
    model_key = _model_key_of(payload)
    if not should_gate(model_key):
        return _NULL_TOKEN
    gate = _gate_for(model_key)
    gate.acquire(model_key, gate_timeout_s() if timeout_s is None else timeout_s,
                 cancel_event=cancel_event)
    return Token(gate)


class gate_for_payload:
    """Context-manager form of :func:`acquire_for_payload` for the one-shot path.

    ``with gate_for_payload(payload): result = _run_once(payload)`` — the gate is
    held for the whole synchronous run and released on exit. Raises ModelBusy on
    a bounded-wait timeout (the caller maps it to a 503).
    """

    def __init__(self, payload: Dict[str, Any], *, timeout_s: Optional[float] = None):
        self._payload = payload
        self._timeout_s = timeout_s
        self._token: Optional[Token] = None

    def __enter__(self) -> Token:
        self._token = acquire_for_payload(self._payload, timeout_s=self._timeout_s)
        return self._token

    def __exit__(self, *exc):
        if self._token is not None:
            self._token.release()
        return False
