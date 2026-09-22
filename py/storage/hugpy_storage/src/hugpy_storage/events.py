"""Catalog-invalidation events — storage tells the world the inventory moved.

After a download lands, a wipe removes weights, or a staged dir promotes onto
its final path, the ENGINE's registry (discovered from the on-disk roots) is
stale. Storage never imports the engine, so it publishes
``hugpy_control.bus.TOPIC_CATALOG_CHANGED`` and the engine subscribes
(PARTITION.md: "hugpy-storage changes those roots and emits invalidation
events"). The payload names what changed so a consumer can refresh one row
instead of re-walking the store.

    publish_catalog_changed("download completed", model_key="x", hub_id=...)

Publishing is best-effort and synchronous on the caller's thread (the bus is
in-process, non-blocking, bounded per subscriber).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from hugpy_control.bus import TOPIC_CATALOG_CHANGED, BusMessage, bus

logger = logging.getLogger("hugpy_storage.events")

SOURCE = "hugpy_storage"


def publish_catalog_changed(reason: str = "", *, model_key: Optional[str] = None,
                            hub_id: Optional[str] = None,
                            destination: Optional[str] = None,
                            change: str = "changed",
                            the_bus: Any = None, **extra: Any) -> Optional[BusMessage]:
    """Publish ``catalog.changed``. Returns the message, or None if the bus
    refused (never raises — invalidation must not break the operation)."""
    payload = {
        "reason": reason or "",
        "change": change,            # "download" | "wipe" | "promote" | "changed"
        "model_key": model_key,
        "hub_id": hub_id,
        "destination": destination,
        "ts": time.time(),
    }
    payload.update({k: v for k, v in extra.items() if v is not None})
    try:
        return (the_bus or bus).publish(TOPIC_CATALOG_CHANGED, source=SOURCE,
                                        payload=payload)
    except Exception:  # noqa: BLE001
        logger.debug("catalog.changed publish failed (%s)", reason, exc_info=True)
        return None


__all__ = ["publish_catalog_changed", "TOPIC_CATALOG_CHANGED", "SOURCE"]
