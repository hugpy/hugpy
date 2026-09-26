"""Request resolution: model key -> (framework, task, builder, runner, placement)."""
from hugpy_engine.resolvers.model_resolver import (
    EXTERNAL_TASK_RUNNERS,
    Peer,
    ensure_staple_weights,
    external_runner_for,
    peer_for,
    resolve,
    resolve_model_key,
    validate_registry,
)
from hugpy_engine.resolvers.remote import (
    get_worker_provider,
    set_no_worker_diagnostic,
    set_no_worker_skips,
    set_placement_provider,
    set_worker_provider,
)

__all__ = [
    "EXTERNAL_TASK_RUNNERS", "Peer", "ensure_staple_weights", "external_runner_for",
    "get_worker_provider", "peer_for", "resolve", "resolve_model_key",
    "set_no_worker_diagnostic", "set_no_worker_skips", "set_placement_provider",
    "set_worker_provider", "validate_registry",
]
