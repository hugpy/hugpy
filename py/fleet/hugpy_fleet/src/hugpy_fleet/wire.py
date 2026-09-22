"""Wire DTOs shared by fleet-central and the workers.

Central (``hugpy_fleet.central.workers``) and the worker agents
(``hugpy_fleet.worker``, ``hugpy_fleet.gguf_worker``) exchange plain JSON
objects over HTTP. These dataclasses name the fields of the four contracts —
enrollment, heartbeat, assignment, operation — so the two sides cannot
drift, without forcing either side onto a validation stack. They are
additive: ``extra`` carries fields a newer peer sends that this build does
not know, and ``to_payload()`` omits ``None`` so an older peer never sees a
key it would refuse.

Keep these dependency-free (stdlib dataclasses only): the phone worker and
the GGUF worker import them on Termux-class boxes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

__all__ = [
    "WireDTO",
    "EnrollmentRequest",
    "WorkerRegistration",
    "WorkerHeartbeat",
    "ModelAssignment",
    "WorkerOperation",
    "OPERATION_VERBS",
]

# Verbs a worker exposes under ``/ops/<verb>``; central's operation DTO carries
# one of these. Kept as data so the console and the workers agree on the set.
OPERATION_VERBS = (
    "aggregate", "cache-evict", "config", "environment", "evict", "free-ram",
    "heartbeat-nudge", "pip", "reap-orphans", "restart", "update", "vram-holders",
)


@dataclass
class WireDTO:
    """Base: symmetric ``from_payload``/``to_payload`` with an ``extra`` bag."""

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Optional[Dict[str, Any]]):
        payload = dict(payload or {})
        known = {f.name for f in fields(cls)} - {"extra"}
        kwargs = {k: payload.pop(k) for k in list(payload) if k in known}
        return cls(**kwargs, extra=payload)

    def to_payload(self) -> Dict[str, Any]:
        out = {k: v for k, v in asdict(self).items() if k != "extra" and v is not None}
        out.update(self.extra or {})
        return out


@dataclass
class EnrollmentRequest(WireDTO):
    """``POST /llm/workers/register`` body. ``token`` is the plaintext
    enrollment token (``central/enrollment_tokens.py`` verifies the hash)."""

    name: str = ""
    url: Optional[str] = None
    token: Optional[str] = None
    worker_id: Optional[str] = None
    role: str = "worker"
    pool: Optional[str] = None
    gpus: List[Dict[str, Any]] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    pkg_version: Optional[str] = None
    engine_build: Optional[str] = None
    rpc_endpoint: Optional[str] = None
    free_ram: Optional[int] = None
    ram_total: Optional[int] = None
    engine: Optional[Dict[str, Any]] = None
    caps: Optional[Dict[str, Any]] = None
    env: Optional[Dict[str, Any]] = None
    serving_limits: Optional[Dict[str, Any]] = None
    slot_capable: Optional[bool] = None
    slot_incapable_reason: Optional[str] = None
    task_capabilities: Optional[Dict[str, bool]] = None


@dataclass
class WorkerRegistration(WireDTO):
    """Central's reply to enrollment: the registry row plus directives."""

    id: str = ""
    name: str = ""
    url: Optional[str] = None
    admission: str = "approved"
    pool: Optional[str] = None
    models: List[str] = field(default_factory=list)
    spill: Dict[str, Any] = field(default_factory=dict)
    boot_prewarm: Optional[str] = None
    required_pkg_version: Optional[str] = None
    pkg_index: Optional[str] = None


@dataclass
class WorkerHeartbeat(WireDTO):
    """``POST /llm/workers/<id>/heartbeat`` body — the worker's live state."""

    gpus: List[Dict[str, Any]] = field(default_factory=list)
    loaded_models: List[str] = field(default_factory=list)
    loading: List[str] = field(default_factory=list)
    models_local: List[str] = field(default_factory=list)
    provisioning: List[str] = field(default_factory=list)
    provision_progress: Optional[Dict[str, Any]] = None
    spill: Optional[Dict[str, Any]] = None
    url: Optional[str] = None
    pkg_version: Optional[str] = None
    engine_build: Optional[str] = None
    role: Optional[str] = None
    rpc_endpoint: Optional[str] = None
    free_ram: Optional[int] = None
    ram_total: Optional[int] = None
    free_ram_raw: Optional[int] = None
    ram_worker_bytes: Optional[int] = None
    ram_external_bytes: Optional[int] = None
    vram_attributed_bytes: Optional[int] = None
    vram_unattributed_bytes: Optional[int] = None
    disk: Optional[Dict[str, Any]] = None
    engine: Optional[Dict[str, Any]] = None
    pool: Optional[str] = None
    caps: Optional[Dict[str, Any]] = None
    env: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    comfy: Optional[Dict[str, Any]] = None
    loaded_detail: Optional[Dict[str, Any]] = None
    slots: Optional[List[Dict[str, Any]]] = None
    allocations: Optional[List[Dict[str, Any]]] = None
    pid_registry: Optional[Dict[str, Any]] = None
    storage: Optional[Dict[str, Any]] = None
    install: Optional[Dict[str, Any]] = None
    serving_limits: Optional[Dict[str, Any]] = None
    slot_capable: Optional[bool] = None
    slot_incapable_reason: Optional[str] = None
    task_capabilities: Optional[Dict[str, bool]] = None
    vram_evictions: Optional[Dict[str, Any]] = None
    vram_holders: Optional[Dict[str, Any]] = None
    aggregate: Optional[Dict[str, Any]] = None
    environment_digest: Optional[Dict[str, Any]] = None
    doctrine_status: Optional[Dict[str, Any]] = None


@dataclass
class ModelAssignment(WireDTO):
    """A designation: ``model_key`` on ``worker_id`` with its spill config
    (allocation mode, n_gpu_layers, ...). Central persists it; the worker
    receives the effective set in every register/heartbeat reply."""

    worker_id: str = ""
    model_key: str = ""
    spill: Dict[str, Any] = field(default_factory=dict)
    granted: bool = False


@dataclass
class WorkerOperation(WireDTO):
    """An operator verb relayed to a worker's ``/ops/<verb>`` endpoint."""

    verb: str = ""
    worker_id: Optional[str] = None
    model_key: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)

    def path(self) -> str:
        return f"/ops/{self.verb}"
