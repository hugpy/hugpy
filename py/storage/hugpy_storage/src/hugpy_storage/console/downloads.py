"""Minimal in-process download jobs (legacy console helper).

The real transfer plane is ``hugpy_storage.downloader`` (queue + daemon); this
keeps the tiny synchronous job table some console code still imports. The
model row comes from the installed catalog source; Flask-free.
"""
import threading
import uuid
from typing import Dict

from hugpy_storage.catalog_source import catalog_get, routing_of

# ---------------------------------------------------------------------------
# In-process job store
# ---------------------------------------------------------------------------

jobs: Dict[str, Dict] = {}
jobs_lock = threading.Lock()


def make_job(model_key: str) -> str:
    job_id = uuid.uuid4().hex[:10]
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "model_key": model_key,
            "status": "queued",
            "message": "",
        }
    return job_id


def download_model(model_key: str, model: dict | None = None) -> str:
    """Synchronous download of one registry model; returns its destination."""
    from hugpy_storage.download_models import download_one
    from hugpy_storage.model_paths import route_destination
    if model is None:
        cfg = catalog_get(model_key)
        if cfg is None:
            raise KeyError(f"Unknown model {model_key!r}")
        model = routing_of(cfg)
    download_one(model, model_key=model_key)
    return route_destination(model)


def run_download(job_id: str, model_key: str) -> None:
    with jobs_lock:
        jobs[job_id]["status"] = "running"
    try:
        dest = download_model(model_key)
        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["message"] = str(dest)
    except Exception as exc:
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = str(exc)
