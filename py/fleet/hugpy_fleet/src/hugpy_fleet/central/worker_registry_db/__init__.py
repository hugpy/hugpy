"""PostgreSQL worker registry, enabled by HUGPY_WORKER_REGISTRY_BACKEND=pg."""

from .service import WorkerRegistryDB

__all__ = ["WorkerRegistryDB"]
