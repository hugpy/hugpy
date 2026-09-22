"""Optional self-registration of a phone worker with the console.

When ``PHONE_BRICK_CENTRAL`` is set, a worker announces itself to the console's
phone-brick pool and heartbeats its live ``/status`` so the console UI can show
it online with its loaded model and queue depth — the same register+heartbeat
handshake the GPU worker agent uses, kept deliberately dependency-light
(``urllib`` only) so it still runs on a Termux phone.

This is entirely additive: with ``PHONE_BRICK_CENTRAL`` unset, nothing happens
and the worker behaves exactly as before (CLI orchestration keeps working).

Config (all via env):
    PHONE_BRICK_CENTRAL    base URL of the console the phone reaches, e.g.
                           http://10.8.0.1:7002  (direct over the VPN) or
                           https://hugpy.example/api  (through nginx). Required
                           to enable registration.
    PHONE_BRICK_NAME       display name in the console (default: hostname).
    PHONE_BRICK_COLOR      box colour for this phone's detections (default #58a6ff).
    PHONE_BRICK_HEARTBEAT  seconds between heartbeats (default 20).
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.request

_ID_FILE = os.path.expanduser("~/.phone-brick/phone_id")
_DEFAULT_HEARTBEAT_S = 20.0


def _post(url: str, body: dict, timeout: float = 5.0) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return resp.status, (json.loads(raw) if raw else {})


def _cached_id() -> str | None:
    try:
        with open(_ID_FILE, "r", encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _cache_id(phone_id: str) -> None:
    try:
        os.makedirs(os.path.dirname(_ID_FILE), exist_ok=True)
        with open(_ID_FILE, "w", encoding="utf-8") as fh:
            fh.write(phone_id)
    except OSError:
        pass  # a non-writable home just means we re-register on restart


class RegistrationAgent:
    """Registers a worker with the console and heartbeats its live status."""

    def __init__(self, worker, *, central: str, name: str, color: str,
                 port: int, interval_s: float = _DEFAULT_HEARTBEAT_S):
        # The pool routes live under /phone-brick; nginx and bare gunicorn both
        # accept this (the latter via the ApiPrefixMiddleware /api strip), so a
        # central of either ".../" or ".../api" works.
        self._base = central.rstrip("/") + "/phone-brick"
        self._worker = worker
        self._name = name
        self._color = color
        self._port = port
        self._interval = interval_s
        self._id = _cached_id()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "RegistrationAgent":
        self._thread.start()
        return self

    def _register(self) -> bool:
        try:
            _, resp = _post(f"{self._base}/register", {
                "name": self._name,
                "port": self._port,
                "color": self._color,
                "phone_id": self._id,   # keep our identity across restarts
            })
            self._id = resp.get("id") or self._id
            if self._id:
                _cache_id(self._id)
            print(f"[phone-brick] registered with console as {self._name} "
                  f"(id={self._id})")
            return True
        except Exception as exc:  # noqa: BLE001 — keep the worker running regardless
            print(f"[phone-brick] register failed: {type(exc).__name__}: {exc}")
            return False

    def _live(self) -> dict:
        """Trimmed /status the console shows for each phone."""
        s = self._worker.status()
        keys = ("model_loaded", "model_path", "queue_size", "processing",
                "completed_count", "classes", "classes_source", "shell_enabled")
        return {k: s.get(k) for k in keys}

    def _loop(self) -> None:
        if not self._id:
            self._register()
        while True:
            try:
                status, _ = _post(f"{self._base}/{self._id}/heartbeat",
                                  {"live": self._live(), "port": self._port})
                if status == 410:  # console forgot us → re-register
                    self._register()
            except urllib.error.HTTPError as exc:
                if exc.code == 410:
                    self._register()
                else:
                    print(f"[phone-brick] heartbeat HTTP {exc.code}")
            except Exception as exc:  # noqa: BLE001 — transient; try again next tick
                print(f"[phone-brick] heartbeat failed: {type(exc).__name__}: {exc}")
            time.sleep(self._interval)


def maybe_start(worker, port: int) -> RegistrationAgent | None:
    """Start a registration agent iff PHONE_BRICK_CENTRAL is configured."""
    central = os.environ.get("PHONE_BRICK_CENTRAL")
    if not central:
        return None
    agent = RegistrationAgent(
        worker,
        central=central,
        name=os.environ.get("PHONE_BRICK_NAME") or socket.gethostname(),
        color=os.environ.get("PHONE_BRICK_COLOR", "#58a6ff"),
        port=port,
        interval_s=float(os.environ.get("PHONE_BRICK_HEARTBEAT", _DEFAULT_HEARTBEAT_S)),
    )
    return agent.start()
