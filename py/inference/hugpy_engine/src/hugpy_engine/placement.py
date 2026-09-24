"""Placement seam: what the engine may ask about the fleet, without importing it.

The engine decides *how* to run a prompt; the fleet knows *where* workers are.
Historically the engine reached straight into the central worker registry,
worker HTTP breaker, eviction ledger, blocklist, model-metrics store and
priority groups. Those now live in ``hugpy_fleet`` and the engine must not
import them. Instead the engine calls the Protocols below through ``get_*()``
accessors, and the composition root (``hugpy_server`` or a worker entry point)
installs real implementations with ``set_*()`` at startup.

Every accessor returns a safe default when nothing has been installed:
"no fleet" — no remote workers, nothing blocked, no metrics, no eviction
ledger, a plain httpx transport with no circuit breaker. That keeps
``import hugpy_engine`` and single-box operation working without
``hugpy_fleet`` installed.

Implementations live in ``hugpy_fleet.central.placement``; see
``py/EXTRACTION_GUIDE.md`` section 4. Each Protocol lists, per method, the
fleet function it replaces so the provider is a thin adapter:

    WorkerRegistry    central.workers   (workers_for_model, shared model-key forms)
    WorkerTransport   central.worker_http (base_url, breaker_key, guard, note_ok,
                                          note_failure, breaker_snapshot,
                                          breaker_scope, async_client,
                                          TRANSPORT_ERRORS -> transport_errors)
    EvictionLedger    central.evictions (emit_eviction_event -> emit,
                                         emit_resolve_fail, disk_stats,
                                         new_run_id, run_scope)
    Blocklist         central.blocklist (blocked_keys, block_reason)
    ModelMetrics      central.model_metrics (derive_variant,
                                             model_metrics_store.record_call /
                                             record_load / get_call)
    PriorityGroups    central.priority_groups (workers_for_key)
"""

from __future__ import annotations

import contextlib
from typing import (
    Any,
    AsyncContextManager,
    ContextManager,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

__all__ = [
    "WorkerRegistry",
    "WorkerTransport",
    "EvictionLedger",
    "Blocklist",
    "ModelMetrics",
    "PriorityGroups",
    "NullWorkerRegistry",
    "NullWorkerTransport",
    "NullEvictionLedger",
    "NullBlocklist",
    "NullModelMetrics",
    "NullPriorityGroups",
    "get_worker_registry",
    "set_worker_registry",
    "get_worker_transport",
    "set_worker_transport",
    "get_eviction_ledger",
    "set_eviction_ledger",
    "get_blocklist",
    "set_blocklist",
    "get_model_metrics",
    "set_model_metrics",
    "get_priority_groups",
    "set_priority_groups",
    "reset_providers",
    "local_key_forms",
]


# ---------------------------------------------------------------------------
# Protocols. Keep them narrow: only what engine code actually calls. Extend
# here (with a default in the Null* class) rather than importing the fleet.
# ---------------------------------------------------------------------------

@runtime_checkable
class WorkerRegistry(Protocol):
    """Read-only view of enrolled workers as the engine needs it."""

    def list_workers(self, *, online_only: bool = True) -> Sequence[Mapping[str, Any]]: ...

    def get_worker(self, worker_id: str) -> Optional[Mapping[str, Any]]: ...

    def workers_for_model(self, model_key: str, *, online_only: bool = True,
                          ready_only: bool = True) -> Sequence[Mapping[str, Any]]:
        """Workers currently holding ``model_key`` (fleet: ``workers.workers_for_model``)."""
        ...

    def match_keys(self, model_key: str, candidates: Iterable[str]) -> Sequence[str]:
        """Which of ``candidates`` are the same model as ``model_key`` (aliases, quants)."""
        ...

    def key_forms(self, model_key: str) -> Sequence[str]:
        """Every spelling that names the same model: raw, lower-cased, the
        ``/``-tail and the ``~``-tail (fleet: ``workers._match_keys``)."""
        ...


@runtime_checkable
class WorkerTransport(Protocol):
    """HTTP calls to a worker; the engine never builds worker URLs itself.

    ``transport_errors`` is the tuple of exception types a relay treats as
    "worker unreachable" (fleet: ``worker_http.TRANSPORT_ERRORS``).
    """

    transport_errors: tuple

    def get_json(self, worker: Mapping[str, Any], path: str, *, timeout: float = 10.0) -> Any: ...

    def post_json(self, worker: Mapping[str, Any], path: str, payload: Any, *, timeout: float = 60.0) -> Any: ...

    def stream(self, worker: Mapping[str, Any], path: str, payload: Any, *, timeout: float = 600.0) -> Iterable[bytes]: ...

    def base_url(self, worker: Mapping[str, Any]) -> str: ...

    def breaker_key(self, worker: Mapping[str, Any]) -> str: ...

    def guard(self, key: str, *, url: str = "", force: bool = False) -> None:
        """Raise when the breaker for ``key`` is open; no-op otherwise."""
        ...

    def note_ok(self, key: str) -> None: ...

    def note_failure(self, key: str, exc: BaseException) -> None: ...

    def breaker_snapshot(self) -> Mapping[str, Mapping[str, Any]]: ...

    def breaker_scope(self, worker: Mapping[str, Any], *, force: bool = False) -> ContextManager[str]:
        """Guard on enter, note_ok/note_failure on exit; yields the breaker key."""
        ...

    def async_client(self, call: str = "relay") -> AsyncContextManager[Any]:
        """An ``httpx.AsyncClient`` (context manager) with the timeouts of ``call``
        (``"relay"``: streaming, ``"relay_long"``: one long body, ``"control"``)."""
        ...


@runtime_checkable
class EvictionLedger(Protocol):
    """Eviction/serve telemetry stream (fleet: ``central.evictions``)."""

    def record(self, event: Mapping[str, Any]) -> None: ...

    def recent(self, *, limit: int = 100) -> Sequence[Mapping[str, Any]]: ...

    def emit(self, stage: str, **fields: Any) -> Optional[Mapping[str, Any]]:
        """Publish one telemetry event (fleet: ``emit_eviction_event``). Never raises."""
        ...

    def emit_resolve_fail(self, model_key: str, resolved_path: Optional[str],
                          reason: str, **extra: Any) -> Optional[Mapping[str, Any]]: ...

    def disk_stats(self, path: Optional[str]) -> Mapping[str, Any]: ...

    def new_run_id(self) -> str: ...

    def run_scope(self, run_id: Optional[str] = None) -> ContextManager[str]:
        """Bind ``run_id`` (or a fresh one) for this thread so nested emits
        belong to one pass; yields the bound id."""
        ...

    def current_group(self) -> Optional[Mapping[str, Any]]:
        """The ambient model-group demand bound on this thread, or None
        (fleet: ``evictions.current_group``)."""
        ...


@runtime_checkable
class Blocklist(Protocol):
    def blocked_keys(self) -> Sequence[str]: ...

    def block_reason(self, model_key: str) -> Optional[str]: ...

    # Optional (2026-09-23): the post-download admission gate's refusal for a
    # HELD model. Callers use getattr — an implementation without it = "never held".
    # def admission_reason(self, model_key: str) -> Optional[str]: ...
    # Optional (2026-09-23): the operator's ARCHIVE mark refusal (hugpy.json
    # "archive"). Callers use getattr — absent = "never marked".
    # def archive_reason(self, model_key: str) -> Optional[str]: ...


@runtime_checkable
class ModelMetrics(Protocol):
    """Per-model serving metrics (fleet: ``central.model_metrics``)."""

    def derive_variant(self, n_gpu_layers: Any, total_layers: Optional[int] = None,
                       *, moe_capable: bool = False) -> Optional[str]:
        """The VARIANTS member for what the serving contract did (split / moe /
        gpu_only_4bit / ram_only_4bit), or None when underivable."""
        ...

    def record_call(self, model_key: str, tok_output: Optional[float] = None, *,
                    task: Optional[str] = None, compute_s: Optional[float] = None,
                    worker_id: Optional[str] = None, **fields: Any) -> bool: ...

    def record_load(self, model_key: str, variant: str, worker_card: str,
                    temperature: str, *, upload_time_s: Optional[float] = None,
                    tok_per_s: Optional[float] = None) -> bool: ...

    def get_call(self, model_key: str) -> Optional[Mapping[str, Any]]:
        """The call EMA row (``n_calls``, ...) or None."""
        ...

    def stats(self, model_key: str) -> Mapping[str, Any]: ...


@runtime_checkable
class PriorityGroups(Protocol):
    def workers_for_key(self, model_key: str) -> Sequence[str]: ...


# ---------------------------------------------------------------------------
# Null defaults: single-box / no-fleet behaviour.
# ---------------------------------------------------------------------------

from hugpy_platform.model_keys import model_key_forms as local_key_forms


class NullWorkerRegistry:
    def list_workers(self, *, online_only: bool = True):
        return ()

    def get_worker(self, worker_id: str):
        return None

    def workers_for_model(self, model_key: str, *, online_only: bool = True, ready_only: bool = True):
        return ()

    def match_keys(self, model_key: str, candidates):
        forms = local_key_forms(model_key)
        return tuple(c for c in candidates if c in forms or str(c).lower() in forms)

    def key_forms(self, model_key: str):
        return local_key_forms(model_key)


class NullWorkerTransport:
    """Plain httpx against ``worker["url"]``: no breaker, sane timeouts.

    Enough for a static placement.json peer or an explicit worker pin on a box
    without the fleet package; the fleet's transport adds the circuit breaker.
    """

    _TIMEOUTS = {"relay": 600.0, "relay_long": 3600.0, "control": 10.0}

    @property
    def transport_errors(self) -> tuple:
        try:
            import httpx
        except Exception:  # pragma: no cover - httpx is a declared dependency
            return (OSError,)
        return (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError,
                httpx.RemoteProtocolError, httpx.NetworkError)

    def base_url(self, worker) -> str:
        url = (worker or {}).get("url") if isinstance(worker, Mapping) else getattr(worker, "url", None)
        return str(url or "").rstrip("/")

    def breaker_key(self, worker) -> str:
        if isinstance(worker, Mapping):
            return str(worker.get("id") or worker.get("name") or self.base_url(worker))
        return str(worker)

    def guard(self, key: str, *, url: str = "", force: bool = False) -> None:
        return None

    def note_ok(self, key: str) -> None:
        return None

    def note_failure(self, key: str, exc: BaseException) -> None:
        return None

    def breaker_snapshot(self):
        return {}

    def breaker_scope(self, worker, *, force: bool = False):
        return contextlib.nullcontext(self.breaker_key(worker))

    def async_client(self, call: str = "relay"):
        import httpx

        read = self._TIMEOUTS.get(call, 600.0)
        return httpx.AsyncClient(timeout=httpx.Timeout(read, connect=3.0))

    def _sync_client(self, timeout: float):
        import httpx

        return httpx.Client(timeout=httpx.Timeout(timeout, connect=3.0))

    def get_json(self, worker, path: str, *, timeout: float = 10.0):
        with self._sync_client(timeout) as client:
            resp = client.get(self.base_url(worker) + "/" + path.lstrip("/"))
            resp.raise_for_status()
            return resp.json()

    def post_json(self, worker, path: str, payload, *, timeout: float = 60.0):
        with self._sync_client(timeout) as client:
            resp = client.post(self.base_url(worker) + "/" + path.lstrip("/"), json=payload)
            resp.raise_for_status()
            return resp.json()

    def stream(self, worker, path: str, payload, *, timeout: float = 600.0):
        with self._sync_client(timeout) as client:
            with client.stream("POST", self.base_url(worker) + "/" + path.lstrip("/"), json=payload) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_bytes():
                    yield chunk


class NullEvictionLedger:
    def record(self, event):
        return None

    def recent(self, *, limit: int = 100):
        return ()

    def emit(self, stage: str, **fields):
        return None

    def emit_resolve_fail(self, model_key, resolved_path, reason, **extra):
        return None

    def disk_stats(self, path):
        return {}

    def new_run_id(self) -> str:
        return ""

    def run_scope(self, run_id=None):
        return contextlib.nullcontext(run_id or "")

    def current_group(self):
        return None


class NullBlocklist:
    def blocked_keys(self):
        return ()

    def block_reason(self, model_key: str):
        return None

    def admission_reason(self, model_key: str):
        return None

    def archive_reason(self, model_key: str):
        return None


class NullModelMetrics:
    def derive_variant(self, n_gpu_layers, total_layers=None, *, moe_capable: bool = False):
        return None

    def record_call(self, model_key, tok_output=None, *, task=None, compute_s=None,
                    worker_id=None, **fields):
        return False

    def record_load(self, model_key, variant, worker_card, temperature, *,
                    upload_time_s=None, tok_per_s=None):
        return False

    def get_call(self, model_key: str):
        return None

    def stats(self, model_key: str):
        return {}


class NullPriorityGroups:
    def workers_for_key(self, model_key: str):
        return ()


_providers: dict[str, Any] = {}
_defaults: dict[str, Any] = {
    "worker_registry": NullWorkerRegistry(),
    "worker_transport": NullWorkerTransport(),
    "eviction_ledger": NullEvictionLedger(),
    "blocklist": NullBlocklist(),
    "model_metrics": NullModelMetrics(),
    "priority_groups": NullPriorityGroups(),
}


def _get(name: str):
    return _providers.get(name, _defaults[name])


def _set(name: str, impl) -> None:
    if impl is None:
        _providers.pop(name, None)
    else:
        _providers[name] = impl


def get_worker_registry() -> WorkerRegistry:
    return _get("worker_registry")


def set_worker_registry(impl: WorkerRegistry | None) -> None:
    _set("worker_registry", impl)


def get_worker_transport() -> WorkerTransport:
    return _get("worker_transport")


def set_worker_transport(impl: WorkerTransport | None) -> None:
    _set("worker_transport", impl)


def get_eviction_ledger() -> EvictionLedger:
    return _get("eviction_ledger")


def set_eviction_ledger(impl: EvictionLedger | None) -> None:
    _set("eviction_ledger", impl)


def get_blocklist() -> Blocklist:
    return _get("blocklist")


def set_blocklist(impl: Blocklist | None) -> None:
    _set("blocklist", impl)


def get_model_metrics() -> ModelMetrics:
    return _get("model_metrics")


def set_model_metrics(impl: ModelMetrics | None) -> None:
    _set("model_metrics", impl)


def get_priority_groups() -> PriorityGroups:
    return _get("priority_groups")


def set_priority_groups(impl: PriorityGroups | None) -> None:
    _set("priority_groups", impl)


def reset_providers() -> None:
    """Drop every installed provider (tests)."""
    _providers.clear()
