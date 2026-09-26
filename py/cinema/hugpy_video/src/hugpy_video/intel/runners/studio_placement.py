"""Studio worker PLACEMENT — central-side resolver for "which GPU worker should run
this studio render", the studio analogue of how an LLM call is placed.

Operator ruling (2026-09-24): *video generation must be handled like LLMs — hugpy's
placement chooses the GPU worker; a studio render must NOT depend on the central env var
``HUGPY_STUDIO_WORKER`` to reach a GPU worker.* This module is that placement. It reads
the SAME worker registry the engine's placement seam exposes
(``hugpy_engine.placement.get_worker_registry``) — the in-process registry central
already fills from worker heartbeats — and picks a worker that:

  * is ONLINE (``list_workers(online_only=True)``),
  * ADVERTISES studio render capability (heartbeat ``studio.render == True``; a GPU box
    with the studio spine installed — see ``hugpy_fleet.worker.agent._studio_capability``),
  * has the render's BOUND model's weights ON DISK (heartbeat ``studio.models`` carries
    the ``model_id`` — a real files-on-disk fact, so routing here never triggers a
    transfer), and
  * has FEASIBLE VRAM (its GPU CAPACITY minus the autofit margin is > 0 and, for an
    explicit budget, covers it via the router's own fit check).

RULINGS HONORED. Distribution is FEASIBLE BY DEFAULT: any feasible worker qualifies
(best VRAM fit first for a stable, capacity-honest pick — NOT a forced designation). The
worker OWNS its GPU service (central only routes; it never forces a download or a
placement). ``HUGPY_STUDIO_WORKER`` is RETAINED as an explicit OPERATOR OVERRIDE — when
set it wins outright (documented escape hatch), but it is no longer the REQUIREMENT.

WHEN NOTHING IS FEASIBLE this returns a NAMED, fact-built refusal (never a silent fall
back to central's GPU-less in-process path for a real model — the caller turns it into a
clean ``JobError``). The refusal names the exact miss: no render-capable worker online,
the model's weights on no worker, or no feasible VRAM.

No network here: presence + capacity are read off the registry rows the heartbeat
already delivered, so the same decision runs on the request path (``should_delegate`` /
the routes' capability probe) with no round-trip. No pathlib anywhere. os.path only.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Mapping

logger = logging.getLogger(__name__)

# Explicit operator override — the SAME env var, DEMOTED from "requirement" to "override".
_WORKER_ENV = "HUGPY_STUDIO_WORKER"

# Autofit margin math — the SINGLE source (studio_i2v._autofit_from_worker imports these),
# so the capacity a worker is JUDGED feasible on and the budget a render is SIZED to are
# the same number. Hold back the LARGER of 10% or 2GB (2GB dominates small cards, 10% big
# ones) for activations / allocator fragmentation / the OS on the card.
_AUTOFIT_MARGIN_FRACTION = 0.10
_AUTOFIT_MARGIN_FLOOR_GB = 2.0
# gpus[].memory_total is BYTES (nvidia-smi MiB * 1024*1024); GiB is the honest inverse.
_BYTES_PER_GIB = 1024 ** 3


@dataclass(frozen=True)
class StudioPlacementRefusal:
    """Why no worker could take a REAL-model render, built from recorded facts. The
    caller (``render_clip``) maps it verbatim into a bus ``JobError`` — a specific,
    named reason, never a generic 'it failed'."""
    code: str
    message: str
    retryable: bool


def _worker_admitted(worker: "Mapping") -> bool:
    """Parity with LLM routing (``workers.py`` uses ``admission == "approved"``
    everywhere it decides a worker may serve): only an APPROVED worker is a studio
    target. A row with no admission key (a bare fake / legacy record) is treated as
    approved so unit rows and pre-admission fleets route unchanged; an explicitly
    ``pending`` / ``blocked`` worker is skipped."""
    adm = worker.get("admission") if isinstance(worker, Mapping) else None
    return adm in (None, "approved")


def worker_advertises_studio_render(worker: "Mapping") -> bool:
    """True iff ``worker``'s heartbeat says it can run a studio render on its GPU
    (``studio.render``). A worker that never advertised it (no ``studio`` blob) is never
    a studio target — the honest absence, not an assumption."""
    blob = worker.get("studio") if isinstance(worker, Mapping) else None
    return isinstance(blob, dict) and bool(blob.get("render"))


def worker_studio_models(worker: "Mapping") -> "frozenset[str]":
    """The set of studio ``model_id``s whose weights are ON DISK on ``worker`` (heartbeat
    ``studio.models``). Empty when the worker advertises none — central then refuses to
    route a real model there rather than triggering a transfer."""
    blob = worker.get("studio") if isinstance(worker, Mapping) else None
    if not isinstance(blob, dict):
        return frozenset()
    models = blob.get("models")
    if isinstance(models, (list, tuple)):
        return frozenset(str(m) for m in models)
    return frozenset()


def worker_capacity_budget_gb(worker: "Mapping") -> "float | None":
    """A worker's autofit routing budget = its LARGEST single GPU's CAPACITY minus the
    safety margin, or None when it reports no usable VRAM. CAPACITY, not free (operator
    ruling 2026-07-27): the reservation engine evicts to make room; the budget states
    what the card can HOLD, not what happens to be free this second. A single render
    binds ONE device, so size to the largest single GPU, never the multi-GPU sum."""
    if not isinstance(worker, Mapping):
        return None
    gpus = [g for g in (worker.get("gpus") or []) if isinstance(g, dict)]
    totals = [g.get("memory_total") for g in gpus
              if isinstance(g.get("memory_total"), (int, float)) and g.get("memory_total") > 0]
    if not totals:
        return None
    total_gib = max(totals) / _BYTES_PER_GIB
    margin = max(total_gib * _AUTOFIT_MARGIN_FRACTION, _AUTOFIT_MARGIN_FLOOR_GB)
    effective = total_gib - margin
    return effective if effective > 0 else None


def bound_model_for_budget(spec, budget_gb: float) -> "tuple[str | None, bool | None]":
    """``(model_id, is_synthetic)`` the studio router binds for ``spec`` AT ``budget_gb``
    — or ``(None, None)`` when the router returns an Err (nothing fits / no capable model)
    or the bound id is unknown. Read-only probe (no load). ``budget_gb`` is used only when
    the spec's own budget is blank (autofit); an explicit spec budget is honored as-is so
    the same model binds here and at render time."""
    try:
        from hugpy_video.intel.runners.studio_i2v import build_capability_request
        from hugpy_video.intel.studio.registry import MODEL_REGISTRY
        from hugpy_video.intel.studio.router import CapabilityRouter

        probe = spec if spec.vram_budget_gb is not None else replace(spec, vram_budget_gb=budget_gb)
        res = CapabilityRouter().resolve(build_capability_request(probe))
        if res.is_err():
            return None, None
        model_id = res.unwrap().model_id
        cfg = MODEL_REGISTRY.get(model_id)
        if cfg is None:
            return None, None
        return model_id, bool(cfg.synthetic)
    except Exception:  # noqa: BLE001 — a probe failure is "cannot place", never a raise
        logger.debug("studio placement: model-bind probe failed", exc_info=True)
        return None, None


def _online_workers() -> "list[Mapping]":
    try:
        from hugpy_engine.placement import get_worker_registry
        rows = get_worker_registry().list_workers(online_only=True) or ()
    except Exception:  # noqa: BLE001 — no fleet installed -> no workers
        return []
    return [w for w in rows if isinstance(w, Mapping)]


def _env_override() -> str:
    return (os.environ.get(_WORKER_ENV) or "").strip().rstrip("/")


def resolve_worker(spec) -> "tuple[str, StudioPlacementRefusal | None]":
    """``(base_url, refusal)`` for the studio worker THIS render should delegate to.

    * ``HUGPY_STUDIO_WORKER`` set  -> that URL, no refusal (explicit operator override).
    * else PLACEMENT: the first feasible online worker (render-capable, holds the bound
      model on disk, usable VRAM), best VRAM fit first.
    * nothing feasible -> ``("", StudioPlacementRefusal(...))`` naming the exact miss.

    A ``("", None)`` is returned ONLY when a probe genuinely cannot classify (defensive);
    callers treat any empty URL for a real-model render as a refusal."""
    env = _env_override()
    if env:
        return env, None

    workers = _online_workers()
    render_workers = [w for w in workers
                      if _worker_admitted(w) and worker_advertises_studio_render(w)]
    if not render_workers:
        return "", StudioPlacementRefusal(
            code="no_studio_worker",
            message=(f"no online worker advertises studio render for capability "
                     f"{spec.capability!r} at {spec.width}x{spec.height} "
                     f"({len(workers)} worker(s) online, 0 render-capable); set "
                     f"HUGPY_STUDIO_WORKER to override"),
            retryable=True)

    missing_model: "str | None" = None
    feasible_capacity = 0
    for worker in sorted(render_workers, key=lambda w: worker_capacity_budget_gb(w) or 0.0,
                         reverse=True):
        cap = worker_capacity_budget_gb(worker)
        if cap is None:
            continue
        feasible_capacity += 1
        budget = float(spec.vram_budget_gb) if spec.vram_budget_gb is not None else cap
        model_id, synthetic = bound_model_for_budget(spec, budget)
        if model_id is None or synthetic:
            continue                       # router err / synthetic at this budget
        if model_id not in worker_studio_models(worker):
            missing_model = model_id
            continue                       # weights not on THIS worker -> no transfer
        url = str(worker.get("url") or "").strip().rstrip("/")
        if url:
            return url, None

    if missing_model is not None:
        return "", StudioPlacementRefusal(
            code="studio_model_not_on_worker",
            message=(f"studio model {missing_model!r} for capability {spec.capability!r} "
                     f"is on no online studio worker's disk (checked "
                     f"{len(render_workers)} render-capable worker(s), "
                     f"{feasible_capacity} with usable VRAM); place the weights on a "
                     f"worker or set HUGPY_STUDIO_WORKER"),
            retryable=False)
    return "", StudioPlacementRefusal(
        code="no_feasible_studio_worker",
        message=(f"no online studio worker can serve capability {spec.capability!r} at "
                 f"{spec.width}x{spec.height}, budget "
                 f"{'autofit' if spec.vram_budget_gb is None else spec.vram_budget_gb} "
                 f"({len(render_workers)} render-capable, {feasible_capacity} with usable "
                 f"VRAM, none bound a real model)"),
        retryable=True)


def resolve_worker_url(spec) -> str:
    """The base URL only (``""`` = none). The pluggable seam ``studio_i2v`` calls."""
    return resolve_worker(spec)[0]
