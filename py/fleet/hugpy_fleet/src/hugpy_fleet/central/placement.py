"""Fleet implementations of the engine's placement seam.

``hugpy_engine.placement`` declares what the engine may ask about the fleet
(``WorkerRegistry``, ``WorkerTransport``, ``EvictionLedger``, ``Blocklist``,
``ModelMetrics``, ``PriorityGroups``) and ships "no fleet" defaults. This
module adapts the fleet-central stores to those Protocols and wires them in.

Composition root (``hugpy_server.wsgi_app`` or a fleet CLI) calls::

    from hugpy_fleet.central import placement
    placement.install()        # at startup
    placement.uninstall()      # tests / shutdown

``install()`` returns the adapters it wired plus ``"oracle_providers"`` (the
``hugpy_oracle.providers`` shapes from ``central.oracle_adapters``, which the
server passes on). It also registers the *legacy* resolver providers the engine still
consults (``hugpy_engine.resolvers.remote.set_*``): worker pick/spill,
exact lookup, sharding placement, cap-aware candidates, no-worker
diagnostic, load-state hold, serve-metrics sink and the model-group member
selector. Historically ``central/workers.py`` did that as an import side
effect; importing fleet modules now has none.

Every adapter is a thin view over the owning module — no fleet state is
duplicated here — so any behaviour change belongs in ``workers.py``,
``worker_http.py``, ``evictions.py``, ``blocklist.py``, ``model_metrics.py``
or ``priority_groups.py``, not in this file.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from hugpy_engine import placement as _seam

__all__ = [
    "FleetWorkerRegistry",
    "FleetWorkerTransport",
    "FleetEvictionLedger",
    "FleetBlocklist",
    "FleetModelMetrics",
    "FleetPriorityGroups",
    "install",
    "uninstall",
    "installed",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WorkerRegistry
# ---------------------------------------------------------------------------

class FleetWorkerRegistry:
    """Read-only view of the central worker registry (``central/workers.py``).

    ``store`` defaults to the module-level ``workers.worker_store`` *at call
    time* (so tests that swap the singleton, and ``swap_worker_store``, are
    honoured); pass a ``WorkerStore`` to bind a specific registry.
    """

    def __init__(self, store: Any = None) -> None:
        self._store = store

    def _s(self):
        if self._store is not None:
            return self._store
        from hugpy_fleet.central import workers as W
        return W.worker_store

    def list_workers(self, *, online_only: bool = True) -> Sequence[Mapping[str, Any]]:
        from hugpy_fleet.central import workers as W
        rows = list(self._s().all() or [])
        if online_only:
            rows = [w for w in rows if W._is_online(w)]
        return tuple(rows)

    def get_worker(self, worker_id: str) -> Optional[Mapping[str, Any]]:
        if not worker_id:
            return None
        return self._s().get(str(worker_id))

    def workers_for_model(self, model_key: str, *, online_only: bool = True,
                          ready_only: bool = True) -> Sequence[Mapping[str, Any]]:
        """Workers that can serve ``model_key``.

        ``ready_only`` — the ranked, online, policy-filtered candidate list
        (what routing would actually use). Otherwise every registered worker
        that has the model assigned, granted or resident (``online_only``
        then decides whether offline rows are kept).
        """
        from hugpy_fleet.central import workers as W
        if not model_key:
            return ()
        if ready_only:
            return tuple(self._s().candidates_for_model(str(model_key)) or [])
        wanted = W._match_keys(str(model_key))
        out = []
        for w in self._s().all() or []:
            if online_only and not W._is_online(w):
                continue
            if W._resident_on(w, model_key, wanted) or W._allocated_on(w, model_key, wanted):
                out.append(w)
                continue
            names = set()
            for m in (w.get("models") or []):
                names |= W._match_keys(str(m))
            if names & wanted:
                out.append(w)
        return tuple(out)

    def key_forms(self, model_key: str) -> Sequence[str]:
        from hugpy_fleet.central import workers as W
        return tuple(sorted(W._match_keys(str(model_key or ""))))

    def match_keys(self, model_key: str, candidates: Iterable[str]) -> Sequence[str]:
        from hugpy_fleet.central import workers as W
        wanted = W._match_keys(str(model_key or ""))
        if not wanted:
            return ()
        return tuple(c for c in candidates if W._match_keys(str(c)) & wanted)


# ---------------------------------------------------------------------------
# WorkerTransport
# ---------------------------------------------------------------------------

class FleetWorkerTransport:
    """Central->worker HTTP through ``central/worker_http.py`` (split timeouts,
    circuit breaker). ``timeout`` narrows the READ budget of the named call
    class; connect timeouts stay the fleet's own. The breaker surface
    (``guard``/``note_ok``/``note_failure``/``breaker_scope``) is the module's."""

    def __init__(self, *, call: str = "control", stream_call: str = "relay") -> None:
        self.call = call
        self.stream_call = stream_call

    @staticmethod
    def _wh():
        from hugpy_fleet.central import worker_http
        return worker_http

    @property
    def transport_errors(self) -> tuple:
        return tuple(self._wh().TRANSPORT_ERRORS)

    def base_url(self, worker: Mapping[str, Any]) -> str:
        return self._wh().base_url(worker)

    def breaker_key(self, worker: Mapping[str, Any]) -> str:
        return self._wh().breaker_key(worker)

    def guard(self, key: str, *, url: str = "", force: bool = False) -> None:
        self._wh().guard(key, url=url, force=force)

    def note_ok(self, key: str) -> None:
        self._wh().note_ok(key)

    def note_failure(self, key: str, exc: BaseException) -> None:
        self._wh().note_failure(key, exc)

    def breaker_snapshot(self) -> Mapping[str, Mapping[str, Any]]:
        return self._wh().breaker_snapshot()

    def breaker_scope(self, worker: Mapping[str, Any], *, force: bool = False):
        return self._wh().breaker_scope(worker, force=force)

    def async_client(self, call: str = "relay"):
        return self._wh().async_client(call)

    @staticmethod
    def _decode(resp) -> Any:
        try:
            return resp.json()
        except Exception:  # noqa: BLE001 - non-JSON body: hand back the text
            return resp.text

    def get_json(self, worker: Mapping[str, Any], path: str, *, timeout: float = 10.0) -> Any:
        resp = self._wh().get(worker, path, call=self.call, read_timeout=timeout)
        return self._decode(resp)

    def post_json(self, worker: Mapping[str, Any], path: str, payload: Any, *, timeout: float = 60.0) -> Any:
        resp = self._wh().post(worker, path, call=self.call, read_timeout=timeout, json=payload)
        return self._decode(resp)

    def stream(self, worker: Mapping[str, Any], path: str, payload: Any, *, timeout: float = 600.0) -> Iterator[bytes]:
        with self._wh().stream("POST", worker, path, call=self.stream_call, json=payload) as resp:
            for chunk in resp.iter_bytes():
                if chunk:
                    yield chunk


# ---------------------------------------------------------------------------
# EvictionLedger
# ---------------------------------------------------------------------------

class FleetEvictionLedger:
    """Eviction/serve telemetry (``central/evictions.py``).

    ``record`` routes a ``{"stage": ..., **fields}`` event through the normal
    emit path (ring + sinks + relay); an event without a stage is stored as
    ``eviction.event``. ``recent`` prefers the durable cross-process store
    and falls back to this process's ring. ``emit``/``emit_resolve_fail``/
    ``disk_stats``/``new_run_id``/``run_scope`` are the module functions.
    """

    @staticmethod
    def _ev():
        from hugpy_fleet.central import evictions
        return evictions

    def record(self, event: Mapping[str, Any]) -> None:
        ev = dict(event or {})
        stage = str(ev.pop("stage", "") or "eviction.event")
        self._ev().emit_eviction_event(stage, **ev)

    def recent(self, *, limit: int = 100) -> Sequence[Mapping[str, Any]]:
        evictions = self._ev()
        try:
            rows = evictions.get_store().recent(limit=limit)
            if rows:
                return tuple(rows)
        except Exception:  # noqa: BLE001 - history must never raise into the engine
            pass
        return tuple(evictions.recent(limit=limit))

    def emit(self, stage: str, **fields: Any) -> Optional[Mapping[str, Any]]:
        return self._ev().emit_eviction_event(stage, **fields)

    def emit_resolve_fail(self, model_key: str, resolved_path: Optional[str],
                          reason: str, **extra: Any) -> Optional[Mapping[str, Any]]:
        return self._ev().emit_resolve_fail(model_key, resolved_path, reason, **extra)

    def disk_stats(self, path: Optional[str]) -> Mapping[str, Any]:
        return self._ev().disk_stats(path)

    def new_run_id(self) -> str:
        return self._ev().new_run_id()

    def run_scope(self, run_id: Optional[str] = None):
        return self._ev().run_scope(run_id)

    def current_group(self) -> Optional[Mapping[str, Any]]:
        return self._ev().current_group()


# ---------------------------------------------------------------------------
# Blocklist
# ---------------------------------------------------------------------------

class FleetBlocklist:
    def blocked_keys(self) -> Sequence[str]:
        from hugpy_fleet.central import blocklist
        return tuple(sorted(str(k) for k in blocklist.blocked_keys()))

    def block_reason(self, model_key: str) -> Optional[str]:
        from hugpy_fleet.central import blocklist
        return blocklist.block_reason(model_key)

    def admission_reason(self, model_key: str) -> Optional[str]:
        """Refusal text when the post-download admission HELD the model."""
        from hugpy_fleet.central import admission_gate
        return admission_gate.admission_reason(model_key)

    def archive_reason(self, model_key: str) -> Optional[str]:
        """Refusal text when the operator marked the model for archive
        (hugpy.json["archive"]; see central/archive_gate.py)."""
        from hugpy_fleet.central import archive_gate
        return archive_gate.archive_reason(model_key)


# ---------------------------------------------------------------------------
# ModelMetrics
# ---------------------------------------------------------------------------

class FleetModelMetrics:
    """The one metrics ledger (``central/model_metrics.py``; Postgres when the
    toolserver is present, SQLite otherwise). ``store`` defaults to the
    module singleton at call time. Every write is fail-open (metrics never
    break serving) and reports whether it landed."""

    def __init__(self, store: Any = None) -> None:
        self._store = store

    def _s(self):
        if self._store is not None:
            return self._store
        from hugpy_fleet.central import model_metrics
        return model_metrics.model_metrics_store

    def derive_variant(self, n_gpu_layers: Any, total_layers: Optional[int] = None,
                       *, moe_capable: bool = False) -> Optional[str]:
        from hugpy_fleet.central import model_metrics
        return model_metrics.derive_variant(n_gpu_layers, total_layers, moe_capable=moe_capable)

    def record_call(self, model_key: str, tok_output: Optional[float] = None, *,
                    task: Optional[str] = None, compute_s: Optional[float] = None,
                    worker_id: Optional[str] = None, **fields: Any) -> bool:
        if tok_output is None:
            tok_output = fields.get("tok_s")
        if tok_output is None:
            return False
        try:
            call = fields.get("call")
            if call is not None:
                try:
                    # THIS call's own numbers for its durable row (per_call_row).
                    return bool(self._s().record_call(str(model_key), float(tok_output),
                                                      task=task, compute_s=compute_s, call=call))
                except TypeError:   # a store that predates ``call=``
                    pass
            return bool(self._s().record_call(str(model_key), float(tok_output),
                                              task=task, compute_s=compute_s))
        except Exception as exc:  # noqa: BLE001 - metrics never break serving
            log.debug("record_call failed for %s: %s", model_key, exc)
            return False

    def record_load(self, model_key: str, variant: str, worker_card: str,
                    temperature: str, *, upload_time_s: Optional[float] = None,
                    tok_per_s: Optional[float] = None) -> bool:
        try:
            return bool(self._s().record_load(str(model_key), variant, worker_card, temperature,
                                              upload_time_s=upload_time_s, tok_per_s=tok_per_s))
        except Exception as exc:  # noqa: BLE001
            log.debug("record_load failed for %s: %s", model_key, exc)
            return False

    def record_load_failure(self, model_key: str, worker_card: str, **kw: Any) -> bool:
        """One load/fail compute_actions row (model_metrics.record_load_failure)."""
        try:
            from hugpy_fleet.central import model_metrics
            return bool(model_metrics.record_load_failure(
                str(model_key), worker_card, store=self._s(), **kw))
        except Exception as exc:  # noqa: BLE001
            log.debug("record_load_failure failed for %s: %s", model_key, exc)
            return False

    # ── routing refusals (2026-09-23): the structured diagnostics record of
    # every refusal lands in the same compute_actions log as the load
    # failures — action="call", outcome="refused", detail = the record.
    def record_refusal(self, diag: Mapping[str, Any]) -> Optional[int]:
        """Append the refusal row; returns its compute_actions id (or None)."""
        try:
            from hugpy_fleet.central import model_metrics
            store = self._s()
            model = ((diag.get("model") or {}).get("resolved")) or None
            rid = diag.get("request_id")
            if not store.append_action("call", model=model, outcome="refused",
                                       detail=dict(diag)):
                return None
            for r in model_metrics.recent_actions(store, limit=20, action="call",
                                                  model=model, outcome="refused"):
                if (r.get("detail") or {}).get("request_id") == rid:
                    return int(r["id"])
        except Exception as exc:  # noqa: BLE001 — never break a refusal
            log.debug("record_refusal failed: %s", exc)
        return None

    def find_refusal(self, request_id: str) -> Optional[Mapping[str, Any]]:
        try:
            from hugpy_fleet.central import model_metrics
            for r in model_metrics.recent_actions(self._s(), limit=5000, action="call",
                                                  outcome="refused"):
                d = r.get("detail") or {}
                if d.get("request_id") == request_id:
                    return {**d, "log_ref": d.get("log_ref") or f"compute_actions#{r['id']}"}
        except Exception as exc:  # noqa: BLE001
            log.debug("find_refusal failed: %s", exc)
        return None

    def get_call(self, model_key: str) -> Optional[Mapping[str, Any]]:
        try:
            return self._s().get_call(str(model_key))
        except Exception:  # noqa: BLE001
            return None

    def stats(self, model_key: str) -> Mapping[str, Any]:
        return dict(self.get_call(model_key) or {})


# ---------------------------------------------------------------------------
# PriorityGroups
# ---------------------------------------------------------------------------

class FleetPriorityGroups:
    def workers_for_key(self, model_key: str) -> Sequence[str]:
        from hugpy_fleet.central import priority_groups
        return tuple(priority_groups.workers_for_key(model_key) or [])


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------

_installed: Dict[str, Any] = {}


def _legacy_setters() -> Dict[str, Callable]:
    """The engine's pre-Protocol resolver seams, each optional."""
    out: Dict[str, Callable] = {}
    try:
        from hugpy_engine.resolvers import remote as R
    except Exception as exc:  # noqa: BLE001 — engine without the legacy seam
        log.info("legacy resolver seam unavailable: %s", exc)
        return out
    for name in ("set_worker_provider", "set_worker_lookup_provider",
                 "set_placement_provider", "set_worker_candidates_provider",
                 "set_no_worker_diagnostic", "set_no_worker_skips",
                 "set_load_state_provider",
                 "set_serve_metrics_sink", "set_member_selector"):
        fn = getattr(R, name, None)
        if callable(fn):
            out[name] = fn
    return out


def _install_legacy_providers() -> None:
    from hugpy_fleet.central import workers as W
    setters = _legacy_setters()

    def _try(name: str, *args: Any) -> None:
        fn = setters.get(name)
        if fn is None:
            return
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 — one seam must not sink the rest
            log.info("%s not registered: %s", name, exc)

    _try("set_worker_provider", W.pick_worker_for_model, W.spill_for)
    _try("set_worker_lookup_provider", W.lookup_worker)
    # Allocator-driven sharding: no-op until HUGPY_SHARD_MODELS opts a model in.
    _try("set_placement_provider", W.placement_for_model)
    _try("set_worker_candidates_provider", W.candidates_for_model)
    _try("set_no_worker_diagnostic", W.explain_no_worker)
    _try("set_no_worker_skips", W.no_worker_skips)
    _try("set_load_state_provider", W.load_state_for_model)
    _try("set_serve_metrics_sink", W.record_serve_metrics)
    # Registering is not enabling: the selector's own kill switch
    # (settings model_groups.enabled, default false) decides per call.
    try:
        from hugpy_fleet.central.model_groups import member_for_model
        _try("set_member_selector", member_for_model)
    except Exception as exc:  # noqa: BLE001
        log.info("model-group member selector not registered: %s", exc)


def _uninstall_legacy_providers() -> None:
    setters = _legacy_setters()
    for name, fn in setters.items():
        try:
            if name == "set_worker_provider":
                fn(None, None)
            else:
                fn(None)
        except Exception:  # noqa: BLE001
            pass


def install(*, registry: Any = None, transport: Any = None,
            eviction_ledger: Any = None, blocklist: Any = None,
            model_metrics: Any = None, priority_groups: Any = None,
            legacy_resolver_providers: bool = True) -> Dict[str, Any]:
    """Wire the fleet into the engine. Idempotent; returns what was installed.

    Keyword arguments override individual adapters (tests, embedding hosts).
    ``legacy_resolver_providers=False`` skips the ``hugpy_engine.resolvers``
    setters (single-box hosts that only want the Protocol seam).
    """
    adapters = {
        "worker_registry": registry or FleetWorkerRegistry(),
        "worker_transport": transport or FleetWorkerTransport(),
        "eviction_ledger": eviction_ledger or FleetEvictionLedger(),
        "blocklist": blocklist or FleetBlocklist(),
        "model_metrics": model_metrics or FleetModelMetrics(),
        "priority_groups": priority_groups or FleetPriorityGroups(),
    }
    _seam.set_worker_registry(adapters["worker_registry"])
    _seam.set_worker_transport(adapters["worker_transport"])
    _seam.set_eviction_ledger(adapters["eviction_ledger"])
    _seam.set_blocklist(adapters["blocklist"])
    _seam.set_model_metrics(adapters["model_metrics"])
    _seam.set_priority_groups(adapters["priority_groups"])
    if legacy_resolver_providers:
        _install_legacy_providers()
    _installed.clear()
    _installed.update(adapters)
    _installed["legacy_resolver_providers"] = bool(legacy_resolver_providers)
    # Fleet facts Oracle consumes (hugpy_oracle.providers.set_*): fleet cannot
    # import Oracle, so the server hands these over.
    from hugpy_fleet.central.oracle_adapters import oracle_providers
    _installed["oracle_providers"] = oracle_providers()
    return dict(_installed)


def uninstall() -> None:
    """Return the engine to its "no fleet" defaults."""
    _seam.set_worker_registry(None)
    _seam.set_worker_transport(None)
    _seam.set_eviction_ledger(None)
    _seam.set_blocklist(None)
    _seam.set_model_metrics(None)
    _seam.set_priority_groups(None)
    if _installed.get("legacy_resolver_providers"):
        _uninstall_legacy_providers()
    _installed.clear()


def installed() -> Dict[str, Any]:
    return dict(_installed)
