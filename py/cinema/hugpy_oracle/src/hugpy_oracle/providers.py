"""Provider seams: what the oracle may ask of packages it must not import.

The oracle decides *what* to run and *which* model should run it; it does
not own the fleet (``hugpy_fleet``), the discovery dossiers
(``hugpy_curation``) or the HTTP composition root (``hugpy_server``). Where
oracle code used to reach straight into those packages it now calls one of
the Protocols below through a ``get_*()`` accessor, and the composition root
installs the real implementation with ``set_*()`` at startup.

Every accessor returns a safe default when nothing has been installed:

* :class:`DossierSource`      -> no dossier store (``root()`` is ``None``,
  ``iter_dossiers()`` is empty). Implemented by ``hugpy_curation`` (the
  dossier store owner); the interim ledger reports ``source_unavailable``.
* :class:`DoctrineSource`     -> ``latest()`` is ``None`` ("central holds no
  doctrine"), so the doctrine probe reports UNKNOWN. Implemented by
  ``hugpy_fleet.doctrine``.
* :class:`TaskCapabilityGate` -> the data-only, legacy-permissive rule over a
  worker's heartbeat ``task_capabilities`` (a worker is skipped only when it
  AFFIRMATIVELY advertises the task as unavailable). ``hugpy_fleet`` installs
  its own ``workers._task_capable`` so the comfy/vision carve-outs stay in one
  place.
* :class:`LoadStateSource`    -> derived from the worker record the engine's
  :mod:`hugpy_engine.placement` registry returns (``loaded_models`` /
  ``loading`` / ``provisioning``). ``hugpy_fleet`` installs
  ``workers.load_state_for_model`` for the live ``/health``-probed view.

Worker rosters and the operator blocklist are NOT re-declared here: the
catalog and the benchmark read them through ``hugpy_engine.placement``
(``get_worker_registry()`` / ``get_blocklist()``), the agreed cross-package
home for placement queries (``py/EXTRACTION_GUIDE.md`` section 4).

Wiring: ``hugpy_server`` (or a package CLI) calls the ``set_*`` functions
once at composition time; tests call :func:`reset_providers`.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Protocol, runtime_checkable

__all__ = [
    "DossierRecord",
    "DossierSource",
    "DoctrineSource",
    "TaskCapabilityGate",
    "LoadStateSource",
    "NullDossierSource",
    "NullDoctrineSource",
    "DefaultTaskCapabilityGate",
    "RegistryLoadStateSource",
    "get_dossier_source",
    "set_dossier_source",
    "get_doctrine_source",
    "set_doctrine_source",
    "get_task_capability_gate",
    "set_task_capability_gate",
    "get_load_state_source",
    "set_load_state_source",
    "reset_providers",
]


# ---------------------------------------------------------------------------
# Protocols. Keep them narrow: only what oracle code actually calls.
# ---------------------------------------------------------------------------

#: One dossier as the interim ledger consumes it. Keys:
#:   ``criteria``  the search criteria bucket the dossier was built under
#:   ``hub_id``    the store's file stem (``org__repo``)
#:   ``payload``   the dossier JSON as a mapping
#:   ``path``      the store's own pointer to the record (artifact ref)
DossierRecord = Mapping[str, Any]


@runtime_checkable
class DossierSource(Protocol):
    """Read-only view of the discovery dossier store (owned by curation)."""

    def root(self) -> Optional[str]:
        """The store's root path for provenance, or ``None`` when absent."""
        ...

    def iter_dossiers(self) -> Iterable[DossierRecord]:
        """Every dossier the store holds, in store order."""
        ...


@runtime_checkable
class DoctrineSource(Protocol):
    """The fleet environment doctrine central currently holds."""

    def latest(self) -> Any:
        """The current doctrine record, or ``None`` when central holds none."""
        ...


@runtime_checkable
class TaskCapabilityGate(Protocol):
    """The fleet's capability-honesty rule for one worker heartbeat record."""

    def task_capable(self, worker: Mapping[str, Any], task: Optional[str]) -> bool: ...


@runtime_checkable
class LoadStateSource(Protocol):
    """Residency of ``model_key`` on ``worker_id`` as a compact status mapping
    (``healthy`` / ``in_progress`` / ``loading`` / ``pulling`` / ``on_disk``),
    or ``None`` when nobody can say."""

    def load_state(self, model_key: str, worker_id: str) -> Optional[Mapping[str, Any]]: ...


# ---------------------------------------------------------------------------
# Defaults: single-box / no-fleet / no-curation behaviour.
# ---------------------------------------------------------------------------

class NullDossierSource:
    def root(self):
        return None

    def iter_dossiers(self):
        return ()


class NullDoctrineSource:
    def latest(self):
        return None


class DefaultTaskCapabilityGate:
    """Affirmative-deny only, legacy-permissive: a worker without a
    ``task_capabilities`` block, or one that does not enumerate ``task``, is
    assumed capable; only ``task_capabilities[task] == False`` denies."""

    def task_capable(self, worker, task):
        if not task:
            return True
        caps = worker.get("task_capabilities") if isinstance(worker, Mapping) else None
        if not isinstance(caps, Mapping) or task not in caps:
            return True
        return bool(caps.get(task))


class RegistryLoadStateSource:
    """Load state read off the worker record the engine's placement registry
    returns. No network: heartbeat fields only."""

    def load_state(self, model_key, worker_id):
        try:
            from hugpy_engine.placement import get_worker_registry
            registry = get_worker_registry()
            worker = registry.get_worker(worker_id)
        except Exception:  # noqa: BLE001 — a registry fault is "cannot say"
            return None
        if not isinstance(worker, Mapping):
            return None

        def _member(field: str) -> bool:
            coll = worker.get(field) or ()
            try:
                return bool(registry.match_keys(model_key, [str(k) for k in coll]))
            except Exception:  # noqa: BLE001
                return model_key in coll

        loaded = _member("loaded_models")
        loading = _member("loading")
        pulling = _member("provisioning")
        return {
            "healthy": loaded,
            "on_disk": _member("models_local"),
            "loading": loading,
            "pulling": pulling,
            "in_progress": loading or pulling,
            "progress": None,
            "message": None,
            "error": None,
        }


_providers: dict[str, Any] = {}
_defaults: dict[str, Any] = {
    "dossier_source": NullDossierSource(),
    "doctrine_source": NullDoctrineSource(),
    "task_capability_gate": DefaultTaskCapabilityGate(),
    "load_state_source": RegistryLoadStateSource(),
}


def _get(name: str):
    return _providers.get(name, _defaults[name])


def _set(name: str, impl) -> None:
    if impl is None:
        _providers.pop(name, None)
    else:
        _providers[name] = impl


def get_dossier_source() -> DossierSource:
    return _get("dossier_source")


def set_dossier_source(impl: DossierSource | None) -> None:
    _set("dossier_source", impl)


def get_doctrine_source() -> DoctrineSource:
    return _get("doctrine_source")


def set_doctrine_source(impl: DoctrineSource | None) -> None:
    _set("doctrine_source", impl)


def get_task_capability_gate() -> TaskCapabilityGate:
    return _get("task_capability_gate")


def set_task_capability_gate(impl: TaskCapabilityGate | None) -> None:
    _set("task_capability_gate", impl)


def get_load_state_source() -> LoadStateSource:
    return _get("load_state_source")


def set_load_state_source(impl: LoadStateSource | None) -> None:
    _set("load_state_source", impl)


def reset_providers() -> None:
    """Drop every installed provider (tests)."""
    _providers.clear()
