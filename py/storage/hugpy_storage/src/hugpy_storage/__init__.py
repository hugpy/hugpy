"""hugpy_storage — bytes at rest and in transit.

The download queue and daemon (``downloader``), resumable central/HF transfer
(``provision``, ``model_sync``, ``download_models``), Hugging Face transport
and token (``huggingface_api``, ``hf_token``, ``model_metadata``), physical
inventory and its caches (``model_physical``, ``model_status_cache``,
``model_presence``), the on-disk model layout (``model_paths``,
``hugpy_marker``, ``gguf_inspect``) and the console-side helpers.

Seams to the layers above (installed by the composition root):

* :mod:`hugpy_storage.catalog_source` — the model registry Protocol
  (``set_catalog_source``); null default knows no models.
* :mod:`hugpy_storage.providers` — budget gate, transfer telemetry, executor
  registrar, serve-path hook, footprint selector.
* :mod:`hugpy_storage.events` — ``catalog.changed`` on the control bus after
  the inventory moves.

This ``__init__`` is deliberately light: nothing heavy (huggingface_hub,
requests, pydantic, sqlite stores) is imported here.
"""
from __future__ import annotations

from hugpy_storage.catalog_source import (
    CatalogSource,
    DictCatalogSource,
    NullCatalogSource,
    get_catalog_source,
    set_catalog_source,
)
from hugpy_storage.events import publish_catalog_changed
from hugpy_storage.providers import (
    set_budget_gate,
    set_executor_registrar,
    set_footprint_selector,
    set_serve_path_hook,
    set_transfer_telemetry,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # catalog seam
    "CatalogSource", "NullCatalogSource", "DictCatalogSource",
    "set_catalog_source", "get_catalog_source",
    # upper-layer hooks
    "set_budget_gate", "set_transfer_telemetry", "set_executor_registrar",
    "set_serve_path_hook", "set_footprint_selector",
    # events
    "publish_catalog_changed",
]
