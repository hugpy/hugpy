"""The reservation engine — acquire / hold / release around a heavy video run.

``acquire(job_name, spec, run_id)`` is called by ``media_bus.run_claimed`` right
before a heavy runner dispatches. It:

  1. loads the task template (measured overlay applied) → the run's PEAK GPU need;
  2. resolves the TARGET worker/card (today: ``ae``, the one video GPU);
  3. records an ACTIVE claim in the registry BEFORE making room, so central
     admission-respect (``fleet_snapshot``) immediately treats the reserved bytes
     as not-free for other placements;
  4. drives PROACTIVE make-room through the EXISTING worker verbs — a ComfyUI
     flush first (``/ops/evict`` a comfy-attributed resident → the worker's own
     ``_comfy_free_models`` / ``set_comfy_headroom_hook`` path), then the eviction
     engine (``/ops/evict`` the on-demand residents largest-first, ``force=false``
     so the WORKER's gate keeps 🔒static / actively-replying / queued-ahead
     residents safe) — polling live free-VRAM within a bounded deadline;
  5. on success, starts a lease REFRESHER (heartbeat) and returns a handle held
     for the whole run; on timeout-while-short it RELEASES the claim and raises
     ``ReservationRefused`` (honest refusal — never an admit-then-OOM, never a
     new protected tier, never a deadlock against one it can't clear).

``release(run_id)`` (called on ANY terminal run path — done/failed/cancelled/
abort/crash-via-lease-expiry) stops the refresher and terminals the claim.

Everything is BEST-EFFORT and fail-OPEN on infrastructure problems: if the store
is down, the fleet is unreadable, or the peak is unknown, a render PROCEEDS
UNRESERVED exactly as it does today. The engine only ever REFUSES when it can
measure a real shortfall it could not clear — the one case where proceeding
would OOM.
"""
from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from hugpy_video.intel.reservation.registry import reservation_registry
from hugpy_video.intel.reservation.templates import ReservationTemplate, load_template

logger = logging.getLogger(__name__)

# Eviction telemetry (operator directive 2026-07-28). The reservation flush is
# the one eviction path CENTRAL itself drives — central does no local serving,
# but it clears the video card for a render. Those evictions belong on the same
# live stream as the fleet's, tagged tier="reservation". Best-effort throughout:
# a telemetry fault must never refuse or delay a render.
# Delivered through the ``hugpy_engine.placement`` eviction-ledger seam: the
# fleet installs its telemetry sink at composition time; the null default
# records nothing (single-box / tests).


# One headroom pass = one ``run_id`` on every event it emits (the console
# renders the flush->evict sequence as one card): ``_run_scope()`` binds it
# through the ledger's own ``run_scope`` (the fleet's, when installed).


def _ledger():
    from hugpy_engine.placement import get_eviction_ledger
    return get_eviction_ledger()


def _run_scope():
    try:
        return _ledger().run_scope()
    except Exception:  # noqa: BLE001 - a ledger without scopes: no grouping
        return contextlib.nullcontext(None)


def _evt_emit(stage: str, **fields) -> None:
    try:
        _ledger().emit(stage, **fields)
    except Exception:  # noqa: BLE001
        pass


_BYTES_PER_GIB = 1024 ** 3


class ReservationRefused(Exception):
    """The card could not be cleared to the run's peak within the deadline (the
    shortfall is all protected residents, or make-room stalled). Carries a typed
    reason so the dispatch path surfaces an honest 'GPU unavailable' terminal
    instead of dispatching a render that would OOM."""
    def __init__(self, reason: Dict[str, Any]):
        self.reason = reason or {}
        super().__init__(self.reason.get("reason") or "GPU reservation refused")


# ── tunables (env-overridable) ───────────────────────────────────────────────
def _makeroom_timeout_s() -> float:
    try:
        return max(0.0, float(os.environ.get("HUGPY_RESERVATION_MAKEROOM_TIMEOUT_S", "90")))
    except ValueError:
        return 90.0


def _poll_s() -> float:
    try:
        return max(0.2, float(os.environ.get("HUGPY_RESERVATION_POLL_S", "3")))
    except ValueError:
        return 3.0


def _settle_s() -> float:
    """Pause after an evict so CUDA/host frees settle before the next fit re-read."""
    try:
        return max(0.0, float(os.environ.get("HUGPY_RESERVATION_SETTLE_S", "1.5")))
    except ValueError:
        return 1.5


def _enabled() -> bool:
    return (os.environ.get("HUGPY_RESERVATIONS") or "on").strip().lower() not in (
        "0", "off", "false", "no", "")


def _refuse_enabled() -> bool:
    """Whether a make-room shortfall HARD-REFUSES the run (gpu_unavailable) vs.
    proceeds best-effort.

    DEFAULT OFF — a safety promise ([[defaults-are-promises]]). The seeded peaks
    are the WHOLE-GPU envelope (e.g. Wan ~20 GB), but a studio render with a blank
    budget AUTOFITS and OFFLOADS to whatever VRAM is free (§3.4 stage 1), so it
    does NOT strictly need the envelope — refusing on it would block a render that
    would have succeeded offloaded. So by default the engine does the VALUABLE
    part (proactive make-room + honest accounting) and PROCEEDS, leaving the actual
    fit to the render's autofit + the WORKER's own admission gate (the authority,
    which already handles offload). Turn ON once p7's ``measured.json`` supplies
    real per-(model,geometry,precision) peaks — then an honest refusal is a
    refusal against a TRUE need, not the envelope."""
    return (os.environ.get("HUGPY_RESERVATION_REFUSE") or "off").strip().lower() in (
        "1", "on", "true", "yes")


# ── fleet reads (lazy — the engine stays boot-cheap; a bare/worker context degrades) ─
def _list_workers() -> List[Dict[str, Any]]:
    try:
        from hugpy_engine.placement import get_worker_registry
        return [dict(w) for w in (get_worker_registry().list_workers(online_only=False) or ())]
    except Exception:  # noqa: BLE001
        return []


def _url_host(url: str) -> str:
    from urllib.parse import urlparse
    if not url:
        return ""
    u = url if "://" in url else "http://" + url
    try:
        return (urlparse(u).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return ""


def _delegation_base(template: ReservationTemplate) -> str:
    """The delegation target URL for this template's task (host-matched to a
    registry worker below). Studio tasks → HUGPY_STUDIO_WORKER; identity tasks →
    IDENTITY_RENDER_URL; dispatch-plane tasks have no fixed base."""
    if template.delegation == "studio_worker":
        return (os.environ.get("HUGPY_STUDIO_WORKER") or "").strip()
    if template.delegation == "identity_render":
        return (os.environ.get("IDENTITY_RENDER_URL") or "").strip()
    return ""


def _has_gpu(w: Dict[str, Any]) -> Optional[int]:
    gpus = [g for g in (w.get("gpus") or []) if isinstance(g, dict)]
    totals = [g.get("memory_total") for g in gpus
              if isinstance(g.get("memory_total"), (int, float)) and g.get("memory_total") > 0]
    return int(max(totals)) if totals else None


def _resolve_target(template: ReservationTemplate
                    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """(worker_id, worker_dict) for the card this run reserves, or (None, None).

    Prefer a host-match to the delegation target (studio/identity env); else the
    single online GPU-bearing worker with the most VRAM (today: ``ae``, the one
    video GPU §0). Only ONLINE workers are eligible."""
    workers = [w for w in _list_workers() if w.get("status") == "online"]
    if not workers:
        return None, None
    base = _delegation_base(template)
    host = _url_host(base)
    if host:
        for w in workers:
            if _url_host(w.get("url") or "") == host and w.get("id"):
                return w["id"], w
    # Fallback: the biggest online GPU box (the video GPU).
    gpu_workers = [(w, _has_gpu(w)) for w in workers]
    gpu_workers = [(w, t) for (w, t) in gpu_workers if t]
    if not gpu_workers:
        return None, None
    gpu_workers.sort(key=lambda wt: -wt[1])
    w = gpu_workers[0][0]
    return w.get("id"), w


def _refresh_worker(worker_id: str) -> Optional[Dict[str, Any]]:
    for w in _list_workers():
        if w.get("id") == worker_id:
            return w
    return None


def _free_vram(worker: Optional[Dict[str, Any]]) -> Optional[int]:
    """Physical free VRAM on the target card — the LARGEST single GPU's free bytes
    (a render binds ONE device; never the multi-GPU sum). None when unmeasurable."""
    if not worker:
        return None
    gpus = [g for g in (worker.get("gpus") or []) if isinstance(g, dict)]
    frees = [g.get("memory_free") for g in gpus
             if isinstance(g.get("memory_free"), (int, float)) and g.get("memory_free") > 0]
    return int(max(frees)) if frees else None


def _pid_models(worker: Dict[str, Any]) -> List[Dict[str, Any]]:
    pr = worker.get("pid_registry")
    if isinstance(pr, dict):
        models = pr.get("models")
        if isinstance(models, list):
            return [m for m in models if isinstance(m, dict)]
    return []


def _planned_gpu_bytes(worker: Dict[str, Any], model_key: str) -> int:
    """What central ITSELF priced this model's GPU share at when it placed it.

    Central issues every placement and already computes ``planned_split`` for
    every designated model (72 on ae). When the per-pid VRAM attribution is
    missing, this is the honest fallback size — not a guess, but the number
    central used to decide the placement in the first place."""
    ps = worker.get("planned_split")
    if not isinstance(ps, dict):
        return 0
    row = ps.get(model_key)
    if not isinstance(row, dict):
        return 0
    for field in ("gpu_bytes", "size_bytes"):
        try:
            v = int(row.get(field) or 0)
        except (TypeError, ValueError):
            v = 0
        if v > 0:
            return v
    return 0


def _gpu_residents(worker: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``{model_key: {"bytes": int, "host_mode": str}}`` for everything central
    knows is ON the card. THE ONE resident enumeration in this module.

    ⚠ IT MUST STAY THE ONLY ONE. The first cut of this fix (ec4b8b1) taught the
    ACCOUNTING path (``_evictable_bytes``) about live slot occupants and left the
    ACTION path (``_evict_candidates``) walking ``pid_registry`` alone. The result
    was strictly worse than before the fix: admission started believing 21 GiB was
    reclaimable, so ``force_admit_safe`` released a render that make-room then
    could not free a single byte for — an admit-then-OOM, which is the exact class
    ``vram-admission-no-evict`` exists to forbid. A resident that cannot be NAMED
    is a resident that must not be COUNTED, so both paths consume this dict.

    THE CENTRAL-SIDE HALF OF THE k30 INVISIBILITY FIX (2026-07-27). The worker's
    own evict planner (``agent._vram_residents``) unions the pid registry with
    every LIVE SLOT OCCUPANT, joining the child's real VRAM — because a slot
    child holds the card whether or not the registry has attributed it yet.
    Central never got that union: it read ``pid_registry.models`` alone and
    SKIPPED every row whose ``model_key`` is null, so a slot child holding the
    whole card counted as ZERO evictable bytes.

    What that cost, measured 2026-07-27: llama-server pid 2261542 held 21.26 of
    23.56 GiB on ae. Central saw `allocations[].vram_bytes = None`, computed
    `evictable = 0`, held the operator's movie segment in `awaiting_capacity`,
    then force-admitted it anyway (``force_admit_safe`` only asks whether another
    *reservation* is active, and an LLM never files one) straight into an OOM.

    Nothing here is new telemetry. Every input already rides the heartbeat:
      * ``pid_registry.models``  — attributed model→pid→VRAM rows
      * ``slots[]``              — the live occupant's ``model_key``
      * ``planned_split``        — central's own price for that placement
    The bug was never that central could not know; it was that it did not join.
    """
    out: Dict[str, int] = {}
    if not worker:
        return out
    # SKIP LEDGER (operator ask 2026-07-27: "see what gets skipped and why").
    # Every byte on the card that this function does NOT return is a byte the
    # eviction planner believes it cannot reclaim — which is exactly how 21 GiB
    # became invisible. Name each omission and its reason instead of silently
    # `continue`-ing past it, so the next 21 GiB shows up in a log line and not
    # in an OOM traceback.
    skipped: List[str] = []
    for m in _pid_models(worker):
        mk = m.get("model_key")
        try:
            vb = int(m.get("vram_bytes") or 0)
        except (TypeError, ValueError):
            vb = 0
        if not mk:
            # cuda_context lump / idle comfy: genuinely no model to target by key.
            if vb > 0:
                skipped.append(
                    f"pid={m.get('pid')} {vb} B host_mode={m.get('host_mode') or '?'} "
                    f"— no model_key (unattributed; not targetable by /ops/evict)")
            continue
        if vb <= 0:
            skipped.append(f"{mk} — attributed but vram_bytes={m.get('vram_bytes')!r}")
            continue
        key = str(mk)
        if vb > int((out.get(key) or {}).get("bytes") or 0):
            out[key] = {"bytes": vb, "host_mode": str(m.get("host_mode") or "")}
    # Union in LIVE slot occupants the registry has not attributed. A slot with a
    # model_key claim IS a resource allocation; its child holds the VRAM whether
    # or not the per-pid join has caught up (fresh re-exec, swept record, or a
    # child the reconcile pass tagged as an anonymous cuda_context lump).
    for s in (worker.get("slots") or []):
        if not isinstance(s, dict):
            continue
        mk = s.get("model_key")
        if not mk:
            continue
        key = str(mk)
        if out.get(key):
            continue                       # already attributed with real bytes
        try:
            vb = int(s.get("expected_bytes") or 0)
        except (TypeError, ValueError):
            vb = 0
        # expected_bytes when the slot reports it, else central's own price.
        sized = vb if vb > 0 else _planned_gpu_bytes(worker, key)
        if sized <= 0:
            skipped.append(f"{key} — live slot occupant, but no expected_bytes and "
                           f"no planned_split price (size unknown)")
            continue
        # A slot child is an LLM/diffusers seat, never the comfy process.
        out[key] = {"bytes": sized, "host_mode": "slot"}
    if skipped:
        logger.info("gpu residents on %s: counted %d (%d B) · SKIPPED %d → %s",
                    worker.get("name") or worker.get("id"), len(out),
                    sum(r["bytes"] for r in out.values()), len(skipped),
                    "; ".join(skipped[:6]))
    return out


def _comfy_resident_keys(worker: Dict[str, Any]) -> List[str]:
    """Comfy-attributed residents central can flush via /ops/evict (→ comfy /free).
    An IDLE comfy holds a null model_key (nothing central can target by key — that
    residual case is the worker-side set_comfy_headroom_hook, release-bound)."""
    out = []
    for m in _pid_models(worker):
        if str(m.get("host_mode")) == "comfy" and m.get("model_key") \
                and int(m.get("vram_bytes") or 0) > 0:
            out.append(str(m["model_key"]))
    return out


def _evict_candidates(worker: Dict[str, Any], tried: set) -> List[str]:
    """On-demand LLM/diffusers residents central may ASK the worker to evict,
    largest-first (frees the most, fewest calls). ``force=false`` on the relay
    means the WORKER is the authority on what is actually permissible — a static /
    replying / queued-ahead resident is refused there, never here. cuda_context /
    comfy(null-key) entries carry no model_key and are skipped.

    ⚠ READS THE SAME UNION AS THE ACCOUNTING PATH — do not narrow this back to
    ``_pid_models``. It walked the registry alone until 2026-07-27 while
    ``_evictable_bytes`` had already been taught about live slot occupants, and
    that split is worse than either half: admission believed 21 GiB was
    reclaimable and make-room then freed nothing, turning a correctly-QUEUED
    render into an admit-then-OOM. Counted and evictable must be the same set."""
    rows = []
    for mk, rec in _gpu_residents(worker).items():
        if mk in tried:
            continue
        if str(rec.get("host_mode")) == "comfy":
            continue  # handled by the comfy-flush pass
        vb = int(rec.get("bytes") or 0)
        if vb <= 0:
            continue
        rows.append((str(mk), vb))
    if not rows:
        # Fallback for a worker that doesn't report pid_registry: loaded_detail.
        ld = worker.get("loaded_detail")
        if isinstance(ld, dict):
            for mk, det in ld.items():
                if mk in tried or not isinstance(det, dict):
                    continue
                vb = int(det.get("model_bytes") or det.get("weight_bytes") or 0)
                rows.append((str(mk), vb))
    rows.sort(key=lambda r: -r[1])
    return [mk for mk, _ in rows]


def _evict(worker: Dict[str, Any], model_key: str, force: bool = False) -> Dict[str, Any]:
    """Relay a targeted eviction to the worker's control agent (/ops/evict). The
    worker picks the mechanism by host_mode (comfy /free, slot SIGTERM, in-process
    ref-drop) and enforces the protection gate when force=false. Best-effort:
    a transport error reads as 'not evicted' and the loop moves on."""
    # Telemetry (2026-07-28): the reservation flush is a real eviction of a real
    # card and belongs on the operator's live stream like any other, tagged
    # tier="reservation". Emitted from CENTRAL (which drives this relay), so
    # `target_worker` names the box whose VRAM is actually being freed — the
    # event's own worker_id is central.
    _evt_emit("evict.start", model_key=model_key, tier="reservation",
              target_worker=worker.get("id") or worker.get("name"), force=force)
    _t0 = time.time()
    try:
        # Worker HTTP goes through the placement transport seam (the fleet
        # installs it; the null default raises -> "not evicted", never a render
        # refusal). A non-dict reply is reported as-is instead of guessed at.
        from hugpy_engine.placement import get_worker_transport
        reply = get_worker_transport().post_json(
            worker, "/ops/evict",
            {"model_key": model_key, "force": bool(force)}, timeout=45.0)
        out = reply if isinstance(reply, dict) else {
            "evicted": False, "reason": f"unexpected reply {type(reply).__name__}"}
    except Exception as exc:  # noqa: BLE001
        out = {"evicted": False, "reason": f"{type(exc).__name__}: {exc}"}
    _ms = int((time.time() - _t0) * 1000)
    _tw = worker.get("id") or worker.get("name")
    if out.get("evicted"):
        _evt_emit("evict.done", model_key=model_key, tier="reservation",
                  target_worker=_tw, duration_ms=_ms,
                  freed_bytes=out.get("vram_freed"))
    else:
        # NOT necessarily an error: force=false means the worker's own gate may
        # have PROTECTED it (static / actively replying). That is a skip, and the
        # operator wants to read it as one — a refusal to evict is the doctrine
        # working, not a fault.
        _evt_emit("candidate.skip", model_key=model_key, tier="reservation",
                  target_worker=_tw, duration_ms=_ms,
                  reason=str(out.get("reason") or "worker gate declined"))
    return out


def _evictable_bytes(worker: Optional[Dict[str, Any]]) -> int:
    """VRAM (bytes) that safe make-room COULD reclaim on this card — the comfy +
    on-demand residents the engine would ask the worker to evict (force=false).

    Deliberately OPTIMISTIC: it counts every keyed resident's VRAM, even ones the
    worker's gate may ultimately protect (static / replying / queued-ahead). That
    bias is on purpose — an admission PROBE over-estimating headroom only ever
    ADMITS a run that then falls to acquire()'s best-effort + the worker gate (no
    worse than today), whereas under-estimating would HOLD a render that would
    have succeeded. Reservation-caused shortfall is accounted separately (via
    ``reserved_bytes``), so this figure is purely the physical make-room ceiling."""
    if not worker:
        return 0
    # Reads the UNION (registry ∪ live slot occupants), not the registry alone —
    # see _gpu_residents. Before 2026-07-27 this walked pid_registry.models and
    # `continue`d past every null model_key, so a slot child holding 21 of 23.5
    # GiB contributed ZERO and the card read as un-freeable.
    return sum(r["bytes"] for r in _gpu_residents(worker).values())


def _refusal_reason(worker: Optional[Dict[str, Any]], peak: int,
                    free: Optional[int], evicted: List[str]) -> Dict[str, Any]:
    remaining = []
    if worker:
        # SAME UNION as the accounting + action paths. Walking _pid_models here
        # produced the self-contradictory refusal the 2026-07-23 incident already
        # named once ("evicted 0 idle ... 0 protected still hold the card") — a
        # refusal that lists no remaining residents while 21 GiB of slot child
        # holds the card is not a report, it is a riddle.
        for mk, rec in _gpu_residents(worker).items():
            vb = int(rec.get("bytes") or 0)
            if mk not in evicted and vb > 0:
                remaining.append({"model_key": str(mk),
                                  "vram_bytes": vb,
                                  "host_mode": str(rec.get("host_mode") or "")})
    return {
        "reason": "GPU reservation could not clear the card to the run's peak "
                  "(remaining residents are protected: static / actively replying "
                  "/ queued ahead)",
        "peak_bytes": int(peak),
        "free_bytes": (int(free) if free is not None else None),
        "short_by_bytes": (int(peak - free) if free is not None else None),
        "worker_id": (worker or {}).get("id"),
        "evicted": list(evicted),
        "remaining_residents": remaining,
    }


# ── make-room orchestration ──────────────────────────────────────────────────
def _ensure_headroom(worker_id: str, worker: Dict[str, Any], peak: Optional[int]
                     ) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """Drive the card to ``peak`` free bytes via the existing verbs. Returns
    (ok, evicted, refusal_reason). Fails OPEN (ok=True) when peak/free are
    unmeasurable — never blocks a render we cannot size."""
    evicted: List[str] = []
    if peak is None:
        return True, evicted, None
    free = _free_vram(worker)
    if free is None:
        return True, evicted, None            # unmeasurable → proceed best-effort
    if free >= peak:
        return True, evicted, None            # already fits — no eviction (shared card)

    # Telemetry run scope: opened only past the three early returns above, so a
    # reservation that fits emits nothing. Everything _ensure_headroom calls
    # (including _evict) inherits this run_id, so the console renders the whole
    # flush→evict sequence as one card.
    _scope = _run_scope()
    if _scope is not None:
        _scope.__enter__()
    try:
        return _ensure_headroom_inner(worker_id, worker, peak, evicted, free)
    finally:
        if _scope is not None:
            try:
                _scope.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass


def _ensure_headroom_inner(worker_id: str, worker: Dict[str, Any], peak: int,
                           evicted: List[str], free: Optional[int]
                           ) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """The make-room drive itself. Split out only so the telemetry run scope can
    wrap every return path; the logic is unchanged."""
    _evt_emit("headroom.start", trigger="reservation", tier="reservation",
              target_worker=worker.get("id") or worker.get("name"),
              need_bytes=peak, free_bytes=free)
    _evt_emit("fit.fail", tier="reservation", need_bytes=peak, free_bytes=free)

    def _done(ok: bool, ev: List[str], reason):
        _evt_emit("headroom.done", trigger="reservation", evicted=list(ev),
                  outcome=("fit" if ok else "refused"), reason=reason,
                  target_worker=worker.get("id") or worker.get("name"))
        return ok, ev, reason

    deadline = time.time() + _makeroom_timeout_s()
    tried: set = set()

    # 1) ComfyUI flush FIRST — the cheap 2-7 GB out-of-band win.
    for mk in _comfy_resident_keys(worker):
        res = _evict(worker, mk, force=False)
        if res.get("evicted"):
            evicted.append(mk)
        tried.add(mk)
        fa = res.get("vram_free_after")
        if isinstance(fa, (int, float)) and fa >= peak:
            return _done(True, evicted, None)
    worker = _refresh_worker(worker_id) or worker
    free = _free_vram(worker)
    if free is not None and free >= peak:
        return _done(True, evicted, None)

    # 2) Eviction engine — on-demand residents largest-first, force=false so the
    #    worker's own gate protects static/replying/queued-ahead. Bounded wait.
    while time.time() < deadline:
        worker = _refresh_worker(worker_id) or worker
        free = _free_vram(worker)
        if free is not None and free >= peak:
            return _done(True, evicted, None)
        cands = [c for c in _evict_candidates(worker, tried)]
        if not cands:
            # Nothing left we may try. Give an in-flight resident a moment to free
            # (it may finish replying), then re-check; if still short → the whole
            # shortfall is protected → refuse honestly (never deadlock).
            time.sleep(_poll_s())
            worker = _refresh_worker(worker_id) or worker
            free = _free_vram(worker)
            if free is not None and free >= peak:
                return _done(True, evicted, None)
            return _done(False, evicted, _refusal_reason(worker, peak, free, evicted))
        cand = cands[0]
        res = _evict(worker, cand, force=False)
        tried.add(cand)
        if res.get("evicted"):
            evicted.append(cand)
            fa = res.get("vram_free_after")
            if isinstance(fa, (int, float)) and fa >= peak:
                return _done(True, evicted, None)
        # else: gated (protected) or no-op — 'tried' keeps us from spinning on it.
        time.sleep(_settle_s())

    # Deadline hit — one last honest re-read.
    worker = _refresh_worker(worker_id) or worker
    free = _free_vram(worker)
    if free is not None and free >= peak:
        return _done(True, evicted, None)
    return _done(False, evicted, _refusal_reason(worker, peak, free, evicted))


# ── lease refresher ──────────────────────────────────────────────────────────
class _Claim:
    """Handle for a held reservation. Owns a daemon refresher thread that heartbeats
    the lease until released; a crash that never releases lets the lease lapse and
    the registry self-expires the claim (orphan safety)."""
    def __init__(self, run_id: str, worker_id: Optional[str], peak: Optional[int]):
        self.run_id = run_id
        self.worker_id = worker_id
        self.peak = peak
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start_refresher(self) -> None:
        interval = max(5.0, reservation_registry.lease_ttl_s / 3.0)

        def _loop() -> None:
            while not self._stop.wait(interval):
                try:
                    reservation_registry.refresh(self.run_id)
                except Exception:  # noqa: BLE001 — a refresh miss just shortens the lease
                    pass

        t = threading.Thread(target=_loop,
                             name=f"reservation-refresh-{self.run_id[:8]}",
                             daemon=True)
        self._thread = t
        t.start()

    def stop(self) -> None:
        self._stop.set()


# Live refreshers by run_id, so release(run_id) can stop the thread even without
# the handle in hand (belt-and-suspenders against a lost handle).
_ACTIVE: Dict[str, _Claim] = {}
_ACTIVE_LOCK = threading.Lock()


# ── admission probe (advisory — the SCHEDULER's gate, not an authority) ───────
def admission_enabled() -> bool:
    """Public alias of the enable gate so the media-bus scheduler can take a pure
    FIFO fast-path (a transparent no-op) when the reservation layer is OFF."""
    return _enabled()


def can_admit(job_name: str, spec: Any = None,
              run_id: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Non-destructive fit PROBE: "would ``job_name``'s reservation fit on its
    target card NOW — counting active reserved bytes, after the make-room the
    engine could safely do — WITHOUT evicting anything and WITHOUT creating a
    claim?" Returns ``(admit, reason)``.

    admit=True  → safe for the scheduler to START this job now.
    admit=False → HOLD: the headroom this run needs is currently committed to
                  ANOTHER active reservation (a heavy run in flight); starting it
                  now would collide. ``reason`` carries the accounting.

    Fail-OPEN (admit=True, reason=None), same philosophy as ``acquire``: a light
    task (no template), the layer disabled, an unresolvable fleet, an unmeasured
    peak, or an unreadable free-VRAM all ADMIT — the probe never blocks a job it
    cannot honestly size, and acquire() + the worker gate remain the authority.

    The accounting is: ``prospective = free - reserved + evictable`` vs ``peak``.
      * ``free``      physical free VRAM on the target card (reflects real usage);
      * ``reserved``  bytes already claimed by OTHER active reservations — the
                      capacity an in-flight heavy run intends to occupy (its
                      allocation may not yet show in ``free``), so we subtract it
                      to serialize heavy runs behind one another;
      * ``evictable`` what safe make-room could physically reclaim (comfy +
                      on-demand residents) — added back optimistically.
    Note this is deliberately conservative for a SECOND heavy run (its peak is
    ~20 GB and ``reserved`` already carries the first run's ~20 GB on a 24 GB
    card, so ``prospective`` goes negative and it HOLDS) and generous for the
    first / a lone run (nothing reserved → make-room headroom carries it)."""
    if not _enabled():
        return True, None
    template = load_template(job_name)
    if template is None:
        return True, None                      # light task — never gated
    worker_id, worker = _resolve_target(template)
    if worker_id is None or worker is None:
        return True, None                      # can't see the fleet — fail open
    peak = template.peak_bytes()
    if peak is None:
        return True, None                      # unmeasured — best-effort, don't gate
    free = _free_vram(worker)
    if free is None:
        return True, None                      # unmeasurable — fail open
    try:
        reserved = int(reservation_registry.reserved_bytes(worker_id))
    except Exception:  # noqa: BLE001 — a store hiccup must never wrongly HOLD a render
        reserved = 0
    evictable = _evictable_bytes(worker)
    prospective = free - reserved + evictable
    admit = prospective >= peak
    reason: Optional[Dict[str, Any]] = None
    if not admit:
        reason = {
            "reason": "GPU capacity for this run is currently committed to an "
                      "active reservation (a heavy run in flight) — holding until "
                      "it releases",
            "worker_id": worker_id,
            "peak_bytes": int(peak),
            "free_bytes": int(free),
            "reserved_bytes": int(reserved),
            "evictable_bytes": int(evictable),
            "short_by_bytes": int(peak - prospective),
        }
    logger.info("admission %s for %s (run %s) on %s: peak=%s free=%s reserved=%s "
                "evictable=%s -> prospective=%s",
                "ADMIT" if admit else "HOLD", job_name, run_id or "-", worker_id,
                peak, free, reserved, evictable, prospective)
    return admit, reason


def force_admit_safe(job_name: str) -> bool:
    """True when the scheduler's starvation/deadlock guard may force-admit a held
    HEAD: a light task always; a heavy task only when its target card is
    physically CLEARABLE to the run's peak.

    ⚠ THIS USED TO ASK THE WRONG QUESTION (fixed 2026-07-27, operator: *"if it
    cannot evict, queue; when it can evict, evict and proceed"*). It returned
    ``reserved_bytes(worker_id) <= 0`` — "is another RESERVATION holding this
    card?" — which is not the same as "is this card free". An **LLM slot child
    never files a video reservation**, so a card with 21.26 of 23.56 GiB in use
    reported ``reserved_bytes == 0``, the guard read it as idle, and the
    scheduler force-admitted a correctly-held render straight into an OOM. The
    guard's own rationale ("it will OFFLOAD / the worker gate is the fit
    authority") did not hold: 1.01 GiB free cannot absorb a ~2 GiB contiguous
    allocation, and the worker gate cannot count an occupant central never
    attributed.

    Now: no competing reservation AND ``free + evictable >= peak``. Failing that,
    the head stays QUEUED (``awaiting_capacity``) until an occupant releases —
    which is the outcome the operator asked for and, empirically, a short wait:
    the 21 GiB holder freed itself within minutes.

    Still fail-OPEN on genuinely unreadable infra (no template / no target / no
    measurable free): an instrument failure must not wedge the queue forever.
    Unknown-because-unmeasured is a different thing from known-to-be-occupied,
    and only the latter is a reason to hold."""
    if not _enabled():
        return True
    template = load_template(job_name)
    if template is None:
        return True
    worker_id, worker = _resolve_target(template)
    if worker_id is None or worker is None:
        return True
    try:
        if int(reservation_registry.reserved_bytes(worker_id)) > 0:
            return False           # a heavy run already holds this card
    except Exception:  # noqa: BLE001
        return True
    peak = template.peak_bytes()
    free = _free_vram(worker)
    if peak is None or free is None:
        return True                # unmeasured — never wedge the queue on a blind spot
    clearable = int(free) + _evictable_bytes(worker)
    if clearable >= int(peak):
        return True
    logger.info("force-admit DENIED for %s on %s: clearable=%s (free=%s + "
                "evictable=%s) < peak=%s — holding in the queue until an "
                "occupant releases", job_name, worker_id, clearable, free,
                _evictable_bytes(worker), peak)
    return False


# ── public API (dispatch path only) ──────────────────────────────────────────
def acquire(job_name: str, spec: Any, run_id: str) -> Optional[_Claim]:
    """Pre-claim the card for a heavy video run. Returns a held claim, or None when
    the task is not reservable / the layer is disabled / infra is unreadable
    (proceed unreserved). Raises ReservationRefused when a real, measured shortfall
    could not be cleared within the deadline (surface an honest terminal)."""
    if not _enabled():
        return None
    template = load_template(job_name)
    if template is None:
        return None                       # light task — no reservation
    worker_id, worker = _resolve_target(template)
    peak = template.peak_bytes()
    if worker_id is None or worker is None:
        # Can't see the fleet — fail open (proceed unreserved), don't block a render
        # because central momentarily can't resolve the target.
        logger.info("reservation: no target worker resolved for %s (run %s) — "
                    "proceeding unreserved", job_name, run_id)
        return None

    gpu = template.gpu_affinity
    # Claim BEFORE make-room so admission-respect sees the reserved bytes immediately.
    reservation_registry.claim(run_id, worker_id, gpu, job_name, peak)
    try:
        ok, evicted, refusal = _ensure_headroom(worker_id, worker, peak)
    except Exception as exc:  # noqa: BLE001 — an engine bug must not wedge the render
        logger.warning("reservation make-room raised for %s (run %s) — proceeding "
                       "unreserved: %s", job_name, run_id, exc, exc_info=True)
        # Keep the claim (accounting) but don't refuse on our own bug; the worker's
        # own admission gate remains the backstop.
        claim = _Claim(run_id, worker_id, peak)
        claim.start_refresher()
        with _ACTIVE_LOCK:
            _ACTIVE[run_id] = claim
        return claim

    if evicted:
        reservation_registry.note_make_room(run_id, evicted)
    if not ok:
        if _refuse_enabled():
            # Honest refusal (opt-in) — release the claim so we don't hold phantom
            # bytes, then raise so the dispatch path terminals the run as
            # gpu_unavailable. Only sound once the peak is a MEASURED true need.
            reservation_registry.release(run_id, reason=(refusal or {}).get("reason"),
                                         state="released")
            logger.info("reservation REFUSED for %s (run %s): peak=%s free=%s "
                        "evicted=%s", job_name, run_id, peak,
                        (refusal or {}).get("free_bytes"), evicted)
            raise ReservationRefused(refusal or {"reason": "GPU reservation refused"})
        # DEFAULT (best-effort): make-room did what it safely could; the peak is
        # only the whole-GPU ENVELOPE, and the render autofits/offloads to the
        # remaining VRAM. PROCEED — hold the claim (accounting) and let the render's
        # autofit + the worker admission gate decide the real fit. Never blocks a
        # render on the envelope; never OOMs (the worker gate is the backstop).
        logger.info("reservation best-effort for %s (run %s): could not reach "
                    "envelope peak=%s (free=%s, evicted=%s) — proceeding; the "
                    "render autofits + the worker gate is the fit authority",
                    job_name, run_id, peak, (refusal or {}).get("free_bytes"),
                    evicted or "none")

    claim = _Claim(run_id, worker_id, peak)
    claim.start_refresher()
    with _ACTIVE_LOCK:
        _ACTIVE[run_id] = claim
    logger.info("reservation HELD for %s (run %s) on %s: peak=%s evicted=%s",
                job_name, run_id, worker_id, peak, evicted or "none")
    return claim


def release(run_id: str, reason: Optional[str] = None) -> None:
    """Release a run's claim on ANY terminal path (done/failed/cancelled/abort).
    Idempotent + best-effort — a double release or an unknown run is a clean no-op.
    Stops the refresher so the lease stops heartbeating."""
    if not run_id:
        return
    with _ACTIVE_LOCK:
        claim = _ACTIVE.pop(run_id, None)
    if claim is not None:
        try:
            claim.stop()
        except Exception:  # noqa: BLE001
            pass
    try:
        reservation_registry.release(run_id, reason=reason or "run terminal")
    except Exception:  # noqa: BLE001 — release is best-effort; the lease TTL is the backstop
        pass
