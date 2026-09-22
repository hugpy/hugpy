"""p6 — the video GPU reservation engine.

Heavy video pipelines (studio i2v / movie, identity reconstruction / mesh /
char360, and the vision-judge movie) share ONE video GPU on the fleet today
(``ae``) with the LLM agent-brain. Left alone they collide mid-render: a Wan
denoise or a Hunyuan3D mesh load OOMs against the 17.7 GB brain squatting the
card (the recorded incident class — see ``dev/VIDEO-TASK-SEQUENCES.md`` §0/§4).

This package pre-claims the card for a whole run BEFORE dispatch, and drives
PROACTIVE make-room through the EXISTING eviction verbs so the run has the room
it needs instead of discovering the collision mid-inference.

Three parts (each its own module):
  * ``templates``  — per-task reservation templates as DATA (§5 schema) plus the
                     measured-overlay loader (built-in estimates OVERLAID by the
                     sibling p7 ``measured.json`` when present — measured wins).
  * ``registry``   — the central reservation registry: claims keyed by
                     (worker/gpu, run_id) with a lease TTL + heartbeat-refresh,
                     persisted via the comms-SQLite idiom. Read-only listing +
                     ``reserved_bytes`` for admission-respect.
  * ``engine``     — acquire/release orchestration: resolve template → peak →
                     target worker → make-room (comfy flush, then the eviction
                     verbs) → bounded wait → hold the claim (refreshing) OR an
                     honest refusal. Never a new protected tier; never a deadlock.

Boundaries this package keeps:
  * Claims are created/released ONLY by the video dispatch path (media_bus).
    There is NO public create route this slice — the operator's console only
    SEES the listing (``GET /llm/reservations``).
  * Protections are ABSOLUTE: make-room only ever asks the worker to evict via
    ``/ops/evict`` with ``force=false``, so the worker's own gate keeps 🔒static /
    actively-replying / queued-ahead / comfy-in-flight residents safe. A
    reservation that can't clear them WAITS then REFUSES honestly.
"""
from __future__ import annotations

from hugpy_video.intel.reservation.templates import (
    ReservationTemplate,
    Stage,
    load_template,
    is_reservable,
    reservable_tasks,
)
from hugpy_video.intel.reservation.registry import reservation_registry
from hugpy_video.intel.reservation.engine import (
    ReservationRefused,
    acquire,
    release,
    can_admit,
    force_admit_safe,
    admission_enabled,
)

__all__ = [
    "ReservationTemplate",
    "Stage",
    "load_template",
    "is_reservable",
    "reservable_tasks",
    "reservation_registry",
    "ReservationRefused",
    "acquire",
    "release",
    "can_admit",
    "force_admit_safe",
    "admission_enabled",
]
