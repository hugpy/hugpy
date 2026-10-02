"""hugpy-link (2026-10-02): per-host presence daemon — the CONNECTION, apart
from the worker's heavy state heartbeat.

Every HUGPY_LINK_INTERVAL_S (5 s) it classifies the local worker process and
POSTs ``/api/llm/workers/<id>/presence`` to central:

  up          unit active, /health answered within HUGPY_LINK_BUSY_S (2 s)
  busy        unit active, /health slow or timed out
  restarting  unit activating/deactivating, or /health refused < 60 s after start
  down        unit failed/inactive, or /health refused for longer

STDLIB ONLY and deliberately outside the auto-publish converge path: it is
copied to the host (e.g. ~/hugpy-link/hugpy_link.py) and runs as its own user
unit, restarted only when this file changes — never because the worker or
central redeployed. Config (env):
  HUGPY_LINK_CENTRAL     central base URL (default WORKER_CENTRAL_URL / :7002)
  HUGPY_LINK_WORKER_URL  local worker (default http://127.0.0.1:${WORKER_PORT:-9200})
  HUGPY_LINK_WORKER_ID   worker id (default: learned from the worker's /health)
  HUGPY_LINK_UNIT        worker user unit (default hugpy-worker.service)
  WORKER_ENROLL_TOKEN    sent as Bearer, same gate as the heartbeat
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

LINK_VERSION = "1"
_START = time.time()


def _env_f(name, default):
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


INTERVAL_S = _env_f("HUGPY_LINK_INTERVAL_S", 5.0)
BUSY_S = _env_f("HUGPY_LINK_BUSY_S", 2.0)
HEALTH_TIMEOUT_S = _env_f("HUGPY_LINK_HEALTH_TIMEOUT_S", 4.0)
RESTART_GRACE_S = _env_f("HUGPY_LINK_RESTART_GRACE_S", 60.0)
CENTRAL = (os.environ.get("HUGPY_LINK_CENTRAL") or os.environ.get("WORKER_CENTRAL_URL")
           or "http://127.0.0.1:7002").rstrip("/")
WORKER_URL = (os.environ.get("HUGPY_LINK_WORKER_URL")
              or f"http://127.0.0.1:{os.environ.get('WORKER_PORT') or 9200}").rstrip("/")
UNIT = os.environ.get("HUGPY_LINK_UNIT") or "hugpy-worker.service"
TOKEN = (os.environ.get("WORKER_ENROLL_TOKEN") or "").strip()


def log(msg):
    print(f"hugpy-link: {msg}", file=sys.stderr, flush=True)


def unit_status(unit=UNIT):
    """(active_state, main_pid, seconds since the unit entered active) from
    ``systemctl --user show``; ("unknown", 0, None) when systemctl is unusable."""
    try:
        out = subprocess.run(
            ["systemctl", "--user", "show", unit, "-p", "ActiveState",
             "-p", "MainPID", "-p", "ActiveEnterTimestampMonotonic"],
            capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return "unknown", 0, None
    kv = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    try:
        pid = int(kv.get("MainPID") or 0)
    except ValueError:
        pid = 0
    since = None
    try:
        mono = int(kv.get("ActiveEnterTimestampMonotonic") or 0)
        if mono:
            since = max(0.0, time.monotonic() - mono / 1e6)
    except ValueError:
        pass
    return kv.get("ActiveState") or "unknown", pid, since


def probe_health(url=WORKER_URL, timeout=HEALTH_TIMEOUT_S):
    """('ok'|'timeout'|'refused'|'error', elapsed_ms, body-or-None)."""
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(url + "/health", timeout=timeout) as r:
            body = json.loads(r.read() or b"{}")
        return "ok", (time.monotonic() - t0) * 1000.0, body
    except (socket.timeout, TimeoutError):
        return "timeout", (time.monotonic() - t0) * 1000.0, None
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return "timeout", (time.monotonic() - t0) * 1000.0, None
        if isinstance(reason, ConnectionRefusedError):
            return "refused", (time.monotonic() - t0) * 1000.0, None
        return "error", (time.monotonic() - t0) * 1000.0, None
    except Exception:  # noqa: BLE001
        return "error", (time.monotonic() - t0) * 1000.0, None


def classify(active_state, since_s, health, health_ms, busy_s=BUSY_S, grace_s=RESTART_GRACE_S):
    """Pure: the worker_state for one tick."""
    if active_state in ("activating", "deactivating", "reloading"):
        return "restarting"
    if active_state in ("failed", "inactive"):
        return "down"
    if health == "ok":
        return "up" if health_ms <= busy_s * 1000.0 else "busy"
    if health == "timeout":
        return "busy"
    # refused / error: the process is not serving yet (or anymore)
    if since_s is not None and since_s < grace_s:
        return "restarting"
    return "down"


def worker_version():
    try:
        from importlib.metadata import version
        return version("hugpy-fleet")
    except Exception:  # noqa: BLE001
        return None


def post_presence(worker_id, body, central=CENTRAL, token=TOKEN, timeout=5.0):
    req = urllib.request.Request(
        f"{central}/api/llm/workers/{worker_id}/presence",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def main():
    worker_id = (os.environ.get("HUGPY_LINK_WORKER_ID") or "").strip() or None
    host = socket.gethostname()
    last_pid, version, last_note = None, None, None
    log(f"start: central={CENTRAL} worker={WORKER_URL} unit={UNIT} every {INTERVAL_S:g}s")
    while True:
        t0 = time.monotonic()
        active, pid, since = unit_status()
        health, ms, body = probe_health()
        if body and body.get("worker_id"):
            worker_id = worker_id or str(body["worker_id"])
        if pid != last_pid:
            last_pid, version = pid, worker_version()
        state = classify(active, since, health, ms)
        note = None
        if not worker_id:
            note = "worker id unknown yet (set HUGPY_LINK_WORKER_ID or wait for /health)"
        else:
            try:
                post_presence(worker_id, {
                    "worker_state": state, "version": version, "link_version": LINK_VERSION,
                    "uptime_s": round(time.time() - _START), "worker_pid": pid or None,
                    "health_ms": round(ms), "unit_state": active, "host": host})
            except urllib.error.HTTPError as exc:
                note = f"central answered HTTP {exc.code}"
            except Exception as exc:  # noqa: BLE001
                note = f"central unreachable ({type(exc).__name__})"
        if note != last_note:  # log transitions only, never every tick
            log(note or f"ok: worker {state}")
            last_note = note
        time.sleep(max(0.5, INTERVAL_S - (time.monotonic() - t0)))


if __name__ == "__main__":
    main()
