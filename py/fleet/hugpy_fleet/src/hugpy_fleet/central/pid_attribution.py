"""Central-side call-time attribution for RELAY-dispatched foreign GPU services.

A worker's ``pid_registry`` can attribute the services IT drives (its slot children,
its in-process torch models, its adopted ComfyUI — see ``worker_agent/pid_registry``).
But some GPU work is dispatched from CENTRAL to a separate service the worker never
initiated — the live case is the identity 3D render service (``identity_render_relay``),
which central POSTs a mesh job to. That service's process shows up in the hosting box's
``pid_registry.unattributed`` (hy3dgen ~5 GB, ``…/identity-render/venv/bin/python``) as an
anonymous squatter, even though the CALL that produced it — the ``identity_mesh_build``
media job — knew the identity slug at dispatch.

This module closes that gap on the READ side: before ``/llm/workers`` is serialized,
correlate each worker's ``pid_registry.unattributed`` entries that are RECOGNIZED as a
relay-dispatched foreign service (by nvidia-smi ``process_name`` markers) with the ACTIVE
relay jobs central dispatched (from the unified job store, ``GET /llm/jobs`` /
``comms.job_store``), and STAMP the job's model/slug + job_id onto the entry.

Design:
  * ``attribute_foreign_relay_procs`` is PURE (worker row + active-job list in, enriched
    row out) so every branch is unit-testable with fakes — no comms, no media_bus, no GPU.
  * ``_active_relay_jobs`` / ``enrich_workers_pid_registry`` are the impure glue the route
    calls: they read the live job store ONCE per ``/llm/workers`` request and best-effort
    augment each relay job with its slug. Every impure step is exception-swallowing — a
    correlation failure must NEVER break the worker list (same discipline as the worker's
    heartbeat isolation).
  * The service table ``FOREIGN_RELAY_SERVICES`` is the SEAM: add a dict to register the
    next relay-dispatched service (ssh/docker render pools, an ASR relay, …) without
    touching the correlation core.

Guardrail preserved: an unattributed entry that matches NO known relay service, or a
recognized service with NO active job, is left as an honest squatter / recognized-idle —
a truly foreign, call-less process still surfaces. Attribution stamps EXTRA keys onto the
entry (``service``/``host_mode``/``model_key``/``job_id``/``slug``/``label``/
``attribution``); it never drops the ``pid``/``name``/``mib`` a consumer already reads.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── the seam: relay-dispatched foreign GPU services central can attribute ──────
# Each entry maps nvidia-smi ``process_name`` MARKERS (case-insensitive substrings) to the
# unified-store job KIND(S) whose active instance explains that process, plus the
# ``host_mode`` label to stamp. ``slug_keys`` is the ordered list of active-job fields to
# read the human identity/model from (first present wins).
FOREIGN_RELAY_SERVICES: tuple = (
    {
        "service": "identity-render",
        "name_markers": ("identity-render", "identity_render", "hy3dgen"),
        "job_kinds": ("identity_mesh_build",),
        "host_mode": "identity-render",
        "slug_keys": ("slug", "model_key", "model"),
    },
)


def _match_service(name: str, services) -> Optional[dict]:
    """The first service whose markers appear in the process ``name`` (a nvidia-smi
    process path), or None. Empty name -> no match."""
    low = (name or "").lower()
    if not low:
        return None
    for svc in services:
        for m in svc.get("name_markers", ()):
            if m and m.lower() in low:
                return svc
    return None


def _pick(job: dict, keys) -> Optional[str]:
    """First truthy value among ``keys`` in ``job`` (the slug/model resolution order)."""
    for k in keys:
        v = job.get(k)
        if v:
            return v
    return None


def attribute_foreign_relay_procs(
        pid_registry: Optional[dict],
        active_jobs: Optional[List[dict]],
        services=FOREIGN_RELAY_SERVICES) -> Optional[dict]:
    """PURE: return a copy of ``pid_registry`` whose ``unattributed`` entries that are
    recognized as a relay foreign service are stamped with the correlating active job's
    model/slug + job_id.

    ``pid_registry`` : ``{"models":[…], "unattributed":[{pid,name,mib}, …]}`` (a worker
                        row's ``pid_registry`` heartbeat field) — any non-dict / missing
                        ``unattributed`` returns the input unchanged (degrade-safe).
    ``active_jobs``   : ``[{"kind","id","model","model_key"?,"slug"?}, …]`` — the live
                        relay jobs (a subset of ``GET /llm/jobs``).

    Stamping (per matched entry, in place within a COPY):
      * exactly one active job of the service's kind -> ``attribution="relay-job"`` with
        ``job_id`` + ``model_key`` (slug-first) + ``slug`` + ``model``;
      * multiple -> ``attribution="relay-job-ambiguous"``, ``job_id`` a list (we cannot
        say which mesh drives which pid without a target key — honest ambiguity);
      * none -> ``attribution="recognized-idle"`` (OUR service, just no active call).
    A matched entry always gains ``service`` + ``host_mode`` so a consumer can render it
    as attributed rather than an anonymous red squatter.
    """
    if not isinstance(pid_registry, dict):
        return pid_registry
    unattr = pid_registry.get("unattributed")
    if not isinstance(unattr, list) or not unattr:
        return pid_registry

    by_kind: Dict[str, List[dict]] = {}
    for j in (active_jobs or []):
        if not isinstance(j, dict):
            continue
        k = j.get("kind")
        if k:
            by_kind.setdefault(k, []).append(j)

    new_unattr: List[Any] = []
    changed = False
    for entry in unattr:
        if not isinstance(entry, dict):
            new_unattr.append(entry)
            continue
        svc = _match_service(str(entry.get("name") or ""), services)
        if svc is None:
            new_unattr.append(entry)          # genuine squatter — the wanted signal
            continue
        jobs: List[dict] = []
        for k in svc.get("job_kinds", ()):
            jobs.extend(by_kind.get(k, []))
        stamped = dict(entry)
        stamped["service"] = svc["service"]
        stamped["host_mode"] = svc["host_mode"]
        if len(jobs) == 1:
            j = jobs[0]
            slug = _pick(j, svc.get("slug_keys", ("slug", "model_key", "model")))
            stamped["job_id"] = j.get("id")
            stamped["model_key"] = slug
            stamped["slug"] = j.get("slug")
            stamped["model"] = j.get("model")
            stamped["label"] = None
            stamped["attribution"] = "relay-job"
        elif len(jobs) > 1:
            stamped["job_id"] = [j.get("id") for j in jobs]
            stamped["model_key"] = None
            stamped["label"] = "%s (ambiguous: %d active jobs)" % (
                svc["service"], len(jobs))
            stamped["attribution"] = "relay-job-ambiguous"
        else:
            stamped["job_id"] = None
            stamped["model_key"] = None
            stamped["label"] = "%s (idle/unknown job)" % svc["service"]
            stamped["attribution"] = "recognized-idle"
        new_unattr.append(stamped)
        changed = True

    if not changed:
        return pid_registry
    out = dict(pid_registry)
    out["unattributed"] = new_unattr
    return out


# ── impure glue (route-facing) ────────────────────────────────────────────────
def _relay_job_kinds(services=FOREIGN_RELAY_SERVICES) -> set:
    kinds: set = set()
    for svc in services:
        kinds.update(svc.get("job_kinds", ()))
    return kinds


def _augment_slug(job_id: str) -> Optional[str]:
    """Best-effort identity slug for a relay ``job_id`` from the media_bus spec.

    The unified job store row carries kind + job id but NOT the identity slug (the slug
    lives in the media_bus spec, keyed by the SAME job id). Read it read-only, exactly as
    ``video_routes`` opens the media DB. Any failure -> None (attribution still stamps the
    job id + kind; the slug is a bonus)."""
    try:
        import json
        import sqlite3
        from hugpy_video.intel import media_bus
        conn = sqlite3.connect(
            "file:%s?mode=ro" % media_bus.DB_PATH, uri=True, timeout=5.0)
        try:
            row = conn.execute(
                "SELECT spec_json FROM media_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        finally:
            conn.close()
        if not row or not row[0]:
            return None
        spec = json.loads(row[0])
        slug = spec.get("slug") if isinstance(spec, dict) else None
        return slug or None
    except Exception:  # noqa: BLE001 — slug is a best-effort enrichment
        return None


def _active_relay_jobs() -> List[dict]:
    """Live relay jobs from the unified store (``comms.job_store``), filtered to the
    registered relay kinds and best-effort augmented with their identity slug. Returns
    ``[]`` on any failure (the correlation then no-ops and rows pass through unchanged)."""
    kinds = _relay_job_kinds()
    if not kinds:
        return []
    try:
        from hugpy_control.jobs import job_store
        rows = job_store.snapshot(kinds=kinds, live_only=True)
    except Exception:  # noqa: BLE001 — no store / broken snapshot -> no correlation
        return []
    out: List[dict] = []
    for r in (rows or []):
        if not isinstance(r, dict):
            continue
        job = dict(r)
        if "slug" not in job or not job.get("slug"):
            slug = _augment_slug(job.get("id"))
            if slug:
                job["slug"] = slug
        out.append(job)
    return out


def enrich_workers_pid_registry(worker_rows: Optional[List[dict]]) -> None:
    """Mutate each worker row's ``pid_registry`` in place with relay-service attribution.

    Called by the ``GET /llm/workers`` serializer. Reads the active relay jobs ONCE
    (O(workers x activejobs) correlation is tiny). Fully best-effort: any failure logs at
    debug and leaves the rows untouched — the worker list must never 5xx on telemetry
    enrichment."""
    if not worker_rows:
        return
    try:
        # Always run the correlation (even with no active jobs): a recognized-but-
        # idle relay service — identity-render with no mesh job in flight — should
        # read as "recognized-idle" (OURS, just idle), NOT as an anonymous squatter.
        # The pure function returns the row UNCHANGED when nothing matches, so a
        # worker with only genuine squatters costs one cheap pass.
        active = _active_relay_jobs()
        for w in worker_rows:
            if not isinstance(w, dict):
                continue
            pr = w.get("pid_registry")
            enriched = attribute_foreign_relay_procs(pr, active)
            if enriched is not pr:
                w["pid_registry"] = enriched
    except Exception:  # noqa: BLE001 — enrichment never breaks the worker list
        logger.debug("pid_registry relay attribution failed", exc_info=True)
