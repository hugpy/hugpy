"""HTTP client for talking to a phone-brick worker.

Thin wrapper over the worker's ``/queue`` + ``/results`` protocol: push a task,
then poll until the matching completed job comes back. Uses ``urllib`` only, so
the orchestrator has no third-party HTTP dependency.
"""
from __future__ import annotations

import json
import time
import urllib.request

from hugpy_fleet.phone_brick.schemas import PhoneSpec, RunCancelled


def _post_json(url: str, body: dict, timeout: float) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout: float):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


class WorkerClient:
    """Reach one worker by its :class:`PhoneSpec`."""

    def __init__(self, phone: PhoneSpec):
        self.phone = phone
        self.base = f"http://{phone.host}:{phone.port}"

    def push(self, task: str, timeout: float = 5.0) -> str:
        """Queue a task; return the worker-assigned job id."""
        _, resp = _post_json(f"{self.base}/queue", {"task": task}, timeout)
        job_id = resp.get("id")
        if job_id is None:
            raise RuntimeError(f"worker push returned no id: {resp}")
        return job_id

    def status(self, timeout: float = 5.0) -> dict:
        _, resp = _get_json(f"{self.base}/status", timeout)
        return resp

    def drain_until(self, job_id: str, drain_timeout: float = 60.0,
                    poll_s: float = 2.0, request_timeout: float = 5.0,
                    cancel_check=None) -> dict | None:
        """Poll ``/results`` until the job with ``job_id`` appears, or time out.

        Returns the completed job dict, or ``None`` if the deadline passed. If
        ``cancel_check`` is given and returns truthy between polls, raise
        :class:`RunCancelled` so the run aborts promptly instead of waiting out
        the timeout.
        """
        deadline = time.time() + drain_timeout
        while time.time() < deadline:
            if cancel_check and cancel_check():
                raise RunCancelled("cancelled while waiting for phone")
            try:
                _, resp = _get_json(f"{self.base}/results", request_timeout)
            except Exception:  # noqa: BLE001 — transient; keep polling
                time.sleep(poll_s)
                continue
            if isinstance(resp, list):
                for job in resp:
                    if job.get("id") == job_id:
                        return job
            time.sleep(poll_s)
        return None
