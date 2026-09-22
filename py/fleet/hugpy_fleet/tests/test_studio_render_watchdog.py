"""k17 — studio render deadlock watchdog: the render runs in a KILLABLE, timeout-
bounded child process, so a wedged fp32-VAE-decode / postprocess tail can never
freeze the worker (recorded 2026-07-18 hang required a manual restart).

Same plain-script style as the other studio tests (no pytest in this venv):
numbered ``[n] PASS`` / ``[n] FAIL`` lines, every check independent, nonzero exit
iff any FAILED. No GPU/torch needed — the real child target is swapped for
module-level FAKE targets (spawn-picklable) that model the child's behaviour
(returns a payload / hangs forever / respects cancel / crashes), so we exercise the
PARENT-side watchdog + cancel + result-plumbing purely with stdlib multiprocessing.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_render_watchdog.py
"""
from __future__ import annotations

import os
import sys
import threading
import time

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_fleet.worker import _studio_subproc as ss


# --------------------------------------------------------------------------- #
# FAKE child targets — module-level so multiprocessing 'spawn' can pickle them by
# reference. Signature matches _studio_subproc._child_main: (spec_dict, conn, ev).
# --------------------------------------------------------------------------- #
def ok_target(spec_dict, conn, cancel_event):
    conn.send({"ok": True, "path": "/shared/clip.mp4", "content_hash": "deadbeef",
               "frames": 81, "width": 480, "height": 480, "duration_s": 6.75,
               "resumed": False})
    conn.close()


def hang_target(spec_dict, conn, cancel_event):
    # Never sends, ignores cancel — models a native/CUDA wedge in the render tail.
    while True:
        time.sleep(0.2)


def cancel_respecting_target(spec_dict, conn, cancel_event):
    # Cooperative: abort as soon as the parent bridges the cancel, Err(cancelled).
    while not cancel_event.is_set():
        time.sleep(0.05)
    conn.send({"ok": False, "error": {
        "code": "cancelled", "message": "cancelled mid-denoise (interrupted)",
        "retryable": False}})
    conn.close()


def crash_target(spec_dict, conn, cancel_event):
    # Dies without ever sending a payload (models a segfault / OOM-kill).
    os._exit(1)


def _run_checks() -> int:
    failures = 0
    n = 0

    def check(cond: bool, label: str, detail: str = "") -> None:
        nonlocal failures, n
        n += 1
        if cond:
            print(f"[{n}] PASS {label}")
        else:
            failures += 1
            print(f"[{n}] FAIL {label}{(' — ' + detail) if detail else ''}")

    # 1) Happy path: the child's payload round-trips across the process boundary.
    out = ss.run_render_subprocess({}, None, 30.0, _target=ok_target, _poll_s=0.1)
    check(out.get("ok") is True and out.get("frames") == 81,
          "child payload round-trips verbatim", repr(out))

    # 2) Watchdog: a hung child is KILLED and the call returns an honest, bounded
    #    error — never blocks forever (the whole point of the fix).
    t0 = time.monotonic()
    out = ss.run_render_subprocess({}, None, 2.0, _target=hang_target, _poll_s=0.1)
    elapsed = time.monotonic() - t0
    check(out.get("ok") is False
          and out.get("error", {}).get("code") == "internal"
          and "timed out" in out.get("error", {}).get("message", ""),
          "wedged render times out with honest internal error", repr(out))
    check(2.0 <= elapsed < 20.0,
          "watchdog returns promptly after the timeout (bounded, not infinite)",
          f"elapsed={elapsed:.1f}s")
    check(out.get("error", {}).get("retryable") is True,
          "timeout error is retryable (central may re-dispatch)")

    # 3) Cooperative cancel: a cancel-aware child settles as cancelled once the
    #    parent bridges the job's threading.Event onto the mp cancel event.
    flag = threading.Event()
    threading.Timer(0.6, flag.set).start()
    out = ss.run_render_subprocess({}, flag, 30.0,
                                   _target=cancel_respecting_target, _poll_s=0.1)
    check(out.get("ok") is False
          and out.get("error", {}).get("code") == "cancelled",
          "cooperative cancel settles as cancelled", repr(out))

    # 4) Cancel of a child that IGNORES cancel: parent hard-kills after the grace
    #    window and settles as cancelled (worker never hangs on an unresponsive
    #    render). Shrink the grace so the test is fast.
    orig_grace = ss._CANCEL_GRACE_S
    ss._CANCEL_GRACE_S = 0.8
    try:
        flag = threading.Event()
        flag.set()  # cancel already requested when the render starts
        t0 = time.monotonic()
        out = ss.run_render_subprocess({}, flag, 30.0,
                                       _target=hang_target, _poll_s=0.1)
        elapsed = time.monotonic() - t0
    finally:
        ss._CANCEL_GRACE_S = orig_grace
    check(out.get("ok") is False
          and out.get("error", {}).get("code") == "cancelled"
          and elapsed < 20.0,
          "unresponsive child is killed after cancel grace", repr(out))

    # 5) Child crash without a payload -> honest 'exited without a result'.
    out = ss.run_render_subprocess({}, None, 30.0, _target=crash_target, _poll_s=0.1)
    check(out.get("ok") is False
          and "without a result" in out.get("error", {}).get("message", ""),
          "child crash surfaces as honest internal error", repr(out))

    # 6) Env knobs: timeout parsing + in-process escape hatch.
    os.environ.pop("HUGPY_STUDIO_RENDER_TIMEOUT_S", None)
    check(ss.render_timeout_s() == ss._DEFAULT_TIMEOUT_S,
          "timeout defaults when env unset")
    os.environ["HUGPY_STUDIO_RENDER_TIMEOUT_S"] = "900"
    check(ss.render_timeout_s() == 900.0, "timeout env override honored")
    os.environ["HUGPY_STUDIO_RENDER_TIMEOUT_S"] = "-5"
    check(ss.render_timeout_s() == ss._DEFAULT_TIMEOUT_S,
          "non-positive timeout falls back to default")
    os.environ.pop("HUGPY_STUDIO_RENDER_TIMEOUT_S", None)

    check(ss.render_inprocess_forced() is False,
          "in-process render off by default")
    os.environ["HUGPY_STUDIO_RENDER_INPROCESS"] = "1"
    check(ss.render_inprocess_forced() is True,
          "in-process escape hatch reads env")
    os.environ.pop("HUGPY_STUDIO_RENDER_INPROCESS", None)

    print(f"\n{n - failures}/{n} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_checks())
