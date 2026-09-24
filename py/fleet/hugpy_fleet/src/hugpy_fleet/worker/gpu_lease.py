"""gpu_lease — run a GPU batch job as an evictable pseudo-model.

The general-purpose supervisor for the "one card, two masters" problem: a
long checkpoint-and-resume batch job (an OCR corpus run, a render sweep) that
would otherwise either hold the card for days or not run at all. Wrapped in
this supervisor, the job leases the card FROM the hugpy worker's eviction
machinery instead of squatting on it:

  * it launches its child only after CLAIMING headroom — the worker evicts
    managed models that have gone idle (min-idle guarded, unforced) until the
    lease's declared VRAM need is free;
  * it registers the running child with the worker as an ``external``
    resident, so the child shows up in the pid registry / heartbeat / central
    console as a first-class, measured, evictable occupant;
  * when a real model load (or an image gen) needs the card, the worker's
    ordinary eviction verb reaches this supervisor's control URL — the
    supervisor SIGTERM->grace->SIGKILLs its child process group, replies only
    after the VRAM is actually free, and goes back to waiting;
  * it then re-claims on a poll loop and relaunches the child, over and over,
    until the child finally exits 0 — at which point the job is DONE and the
    supervisor unregisters and exits.

THE ADMISSION CONTRACT: the wrapped command must be resume-safe (idempotent
re-runs that skip completed work). Eviction is a kill; a job that cannot be
killed and relaunched without losing correctness must not be run under this.

Deliberately stdlib-only (urllib, http.server, subprocess): the supervisor
must run under ANY python (a miniconda OCR env, a bare system python), not
just the worker venv.

Usage:
  python -m hugpy_fleet.worker.gpu_lease \\
      --key ocr:bluebook --vram-gib 17 \\
      [--priority evictable|non-evictable] [--resume enabled|disabled] \\
      [--worker http://127.0.0.1:9200] [--control-port 9310] \\
      [--min-idle 300] [--poll 60] [--grace 15] [--max-failures 5] \\
      -- <command ...>

Adjust a RUNNING lease's policy (either form, no restart needed):
  python -m hugpy_fleet.worker.gpu_lease --adjust --key ocr:bluebook \\
      [--priority non-evictable] [--resume disabled] [--worker ...]

Policy (wildcard-process ruling, 2026-08-12):
  --priority evictable      (default) any demand path may pause the job —
                            model loads, image gens, other claims.
  --priority non-evictable  the external twin of static residency: only an
                            operator force-evict touches it.
  --resume enabled          (default) pause -> wait -> re-claim -> relaunch.
  --resume disabled         a pause ENDS the run: the supervisor unregisters
                            and exits 75 (EX_TEMPFAIL; set SuccessExitStatus=75
                            on a systemd unit that should treat that as clean).

Default --worker: ``HUGPY_LEASE_WORKER`` if set, else
``http://127.0.0.1:${WORKER_PORT:-9100}`` — the worker agent reads WORKER_PORT
too (its argparse default and studio_reserve share the convention), so a box
whose unit sets WORKER_PORT=9200 gets the right target with no flag. The flag
always wins.

Env overrides (flags win): HUGPY_LEASE_WORKER, HUGPY_LEASE_MIN_IDLE_S,
HUGPY_LEASE_POLL_S, HUGPY_LEASE_GRACE_S, HUGPY_LEASE_CONTROL_PORT.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("gpu_lease")

_MIB = 1024 * 1024


def _default_worker() -> str:
    """The worker agent this lease talks to. HUGPY_LEASE_WORKER wins; otherwise
    derive 127.0.0.1:${WORKER_PORT} — the SAME env the worker agent's own
    argparse default and studio_reserve.py read, so the lease follows the box's
    port without a hardcoded guess (9100 only as the last-resort fallback that
    matches the agent's own default)."""
    env = os.environ.get("HUGPY_LEASE_WORKER")
    if env and env.strip():
        return env.strip()
    port = (os.environ.get("WORKER_PORT") or "9100").strip() or "9100"
    return f"http://127.0.0.1:{port}"


# ── worker API (urllib — no requests dependency) ─────────────────────────────
def _post_json(url: str, body: dict, timeout: float = 30.0) -> "dict | None":
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except Exception as exc:  # noqa: BLE001 — worker down is a normal state here
        logger.debug("POST %s failed: %s", url, exc)
        return None


def _nvidia_free_bytes() -> "int | None":
    """Direct free-VRAM read for the worker-down fallback: the lease must not
    stall the batch forever just because the worker agent is stopped (a
    stopped worker serves no models, so the card is fair game if it's free)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        line = (out.stdout or "").strip().splitlines()
        return int(line[0].strip()) * _MIB if line else None
    except Exception:  # noqa: BLE001 — no nvidia-smi -> unmeasurable
        return None


def _gpu_pids() -> "dict[int, int]":
    """pid -> vram_bytes for every compute process on the card(s)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        pids: dict[int, int] = {}
        for ln in (out.stdout or "").strip().splitlines():
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 2:
                try:
                    pids[int(parts[0])] = int(parts[1]) * _MIB
                except ValueError:
                    continue
        return pids
    except Exception:  # noqa: BLE001
        return {}


def _ppid(pid: int) -> "int | None":
    try:
        with open(f"/proc/{pid}/stat") as f:
            # field 4 (1-based) after the parenthesised comm, which may
            # itself contain spaces — split on the LAST ')'.
            tail = f.read().rsplit(")", 1)[1].split()
            return int(tail[1])
    except Exception:  # noqa: BLE001
        return None


def _is_descendant(pid: int, ancestor: int, max_hops: int = 32) -> bool:
    cur = pid
    for _ in range(max_hops):
        if cur == ancestor:
            return True
        cur = _ppid(cur)
        if cur is None or cur <= 1:
            return False
    return False


class Lease:
    """The supervisor state machine: WAITING -> RUNNING -> (YIELDED ->
    WAITING)* -> DONE. One child process group at a time; all transitions
    under one lock so a /pause racing a natural exit stays coherent."""

    def __init__(self, args, cmd: "list[str]") -> None:
        self.args = args
        self.cmd = cmd
        self.lock = threading.RLock()
        self.state = "waiting"            # waiting | running | yielded | done
        self.child: "subprocess.Popen | None" = None
        self.control_url: "str | None" = None
        self.resume_now = threading.Event()
        self.launches = 0
        self.evictions = 0
        self.failures = 0                 # consecutive non-evict failures
        self.started_at = time.time()
        # wildcard-process policy (2026-08-12), both adjustable at runtime via
        # the control server's POST /set (which the worker's /ops/external/set
        # forwards to). evictable is ENFORCED worker-side (the eviction paths
        # skip a non-evictable lease); it's carried here so heartbeats and the
        # console agree. resume is enforced HERE: a paused resume-disabled
        # lease exits (EX_TEMPFAIL) instead of waiting to re-claim.
        self.evictable = (args.priority != "non-evictable")
        self.resume_enabled = (args.resume != "disabled")

    # ── worker calls ─────────────────────────────────────────────────────────
    def claim(self) -> "tuple[bool, str]":
        """(may_run, why). Ask the worker to evict idle models down to our
        VRAM need; fall back to a bare nvidia-smi free check when the worker
        is unreachable (a stopped worker serves nobody — free card is ours)."""
        need = self.args.vram_gib * 2**30
        body = {"model_key": self.args.key,
                "target_free_gib": self.args.vram_gib}
        if self.args.min_idle is not None:
            body["min_idle_s"] = self.args.min_idle
        res = _post_json(self.args.worker + "/ops/external/claim", body,
                         timeout=120.0)
        if res is not None:
            if res.get("reached"):
                return True, (f"claim reached: {res.get('free_after', 0)/2**30:.1f}"
                              f" GiB free (evicted {res.get('evicted') or []})")
            return False, ("claim short: "
                           f"{(res.get('free_after') or 0)/2**30:.1f} GiB free "
                           f"< {self.args.vram_gib:.1f} GiB target; "
                           f"skipped={res.get('skipped') or []}")
        free = _nvidia_free_bytes()
        if free is None:
            return False, "worker unreachable and nvidia-smi unmeasurable"
        if free >= need:
            return True, (f"worker unreachable; nvidia-smi fallback: "
                          f"{free/2**30:.1f} GiB free >= need")
        return False, (f"worker unreachable; nvidia-smi fallback: "
                       f"{free/2**30:.1f} GiB free < need")

    def register(self) -> None:
        """(Re-)register with the worker as an external resident. Called after
        every launch and on the heartbeat — idempotent on the worker, and the
        heartbeat re-register is what heals a worker restart (its registries
        are in-memory). Registers the GPU-holding descendant pid when one is
        visible (honest nvidia-smi VRAM join); the child pid otherwise."""
        with self.lock:
            child = self.child
            if child is None or child.poll() is not None:
                return
            pid = child.pid
        best, best_vram = None, -1
        for gp, vb in _gpu_pids().items():
            if vb > best_vram and _is_descendant(gp, pid):
                best, best_vram = gp, vb
        with self.lock:
            evictable, resume = self.evictable, self.resume_enabled
        _post_json(self.args.worker + "/ops/external/register", {
            "model_key": self.args.key,
            "pid": best or pid,
            "control_url": self.control_url,
            "vram_gib": self.args.vram_gib,
            "note": "gpu_lease: " + " ".join(self.cmd)[:200],
            "evictable": evictable,
            "resume": "enabled" if resume else "disabled",
        }, timeout=15.0)

    def unregister(self) -> None:
        _post_json(self.args.worker + "/ops/external/unregister",
                   {"model_key": self.args.key}, timeout=15.0)

    # ── child lifecycle ──────────────────────────────────────────────────────
    def launch(self) -> None:
        with self.lock:
            self.child = subprocess.Popen(self.cmd, start_new_session=True)
            self.state = "running"
            self.launches += 1
            logger.info("launched (#%d) pid=%d pgid=%d: %s", self.launches,
                        self.child.pid, self.child.pid, " ".join(self.cmd))
        self.register()

    def stop_child(self, reason: str) -> bool:
        """SIGTERM the child's process GROUP, grace-wait, SIGKILL leftovers.
        Returns True if a running child was stopped. Synchronous — /pause
        replies only after this returns, so the worker's post-evict free-VRAM
        read is real."""
        with self.lock:
            child = self.child
            if child is None or child.poll() is not None:
                return False
            self.state = "yielded"
            pgid = child.pid
        logger.info("pausing child pgid=%d (%s)", pgid, reason)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        deadline = time.time() + self.args.grace
        while time.time() < deadline:
            if child.poll() is not None:
                break
            time.sleep(0.5)
        if child.poll() is None:
            logger.warning("child ignored SIGTERM for %.0fs — SIGKILL pgid=%d",
                           self.args.grace, pgid)
        # KILL the whole group either way: the leader exiting does not prove
        # its GPU-holding descendants did, and a stray worker keeping VRAM
        # would make the eviction a lie.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        self.evictions += 1
        return True

    # ── control server (the URL _evict_model's external branch calls) ────────
    def serve_control(self) -> None:
        lease = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *a):  # noqa: N802 — quiet
                logger.debug("control: " + fmt, *a)

            def _reply(self, code: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                if self.path.rstrip("/") in ("", "/status"):
                    return self._reply(200, lease.status())
                return self._reply(404, {"ok": False})

            def _body(self) -> dict:
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    return json.loads(self.rfile.read(n).decode() or "{}")
                except Exception:  # noqa: BLE001 — bad body reads as empty
                    return {}

            def do_POST(self):  # noqa: N802
                path = self.path.rstrip("/")
                if path == "/pause":
                    was = lease.stop_child("evicted by worker")
                    return self._reply(200, {"ok": True, "paused": True,
                                             "was_running": was,
                                             **lease.status()})
                if path == "/resume":
                    lease.resume_now.set()
                    return self._reply(200, {"ok": True, "resuming": True})
                if path == "/set":
                    # live policy adjustment (worker /ops/external/set
                    # forwards here; direct curl works too). Takes effect on
                    # the next heartbeat/decision — no child restart.
                    body = self._body()
                    with lease.lock:
                        if "evictable" in body:
                            lease.evictable = bool(body["evictable"])
                        if body.get("resume") in ("enabled", "disabled"):
                            was_disabled = not lease.resume_enabled
                            lease.resume_enabled = (body["resume"] == "enabled")
                            # flipping a paused lease back to enabled should
                            # not wait out the poll timer
                            if was_disabled and lease.resume_enabled:
                                lease.resume_now.set()
                    lease.register()
                    return self._reply(200, {"ok": True, **lease.status()})
                return self._reply(404, {"ok": False})

        srv = ThreadingHTTPServer((self.args.control_host,
                                   self.args.control_port), Handler)
        self.control_url = (f"http://{self.args.control_host}:"
                            f"{srv.server_address[1]}")
        threading.Thread(target=srv.serve_forever, daemon=True,
                         name="lease-control").start()
        logger.info("control server on %s (pause/resume/status)",
                    self.control_url)

    def status(self) -> dict:
        with self.lock:
            child = self.child
            return {"key": self.args.key, "state": self.state,
                    "child_pid": child.pid if child else None,
                    "child_running": bool(child and child.poll() is None),
                    "launches": self.launches, "evictions": self.evictions,
                    "failures": self.failures,
                    "evictable": self.evictable,
                    "resume": "enabled" if self.resume_enabled else "disabled",
                    "uptime_s": round(time.time() - self.started_at, 1)}

    # ── the loop ─────────────────────────────────────────────────────────────
    def run(self) -> int:
        self.serve_control()

        def _bail(signum, frame):  # noqa: ARG001
            logger.info("signal %d — stopping child and unregistering", signum)
            self.stop_child(f"supervisor got signal {signum}")
            self.unregister()
            sys.exit(128 + signum)

        signal.signal(signal.SIGTERM, _bail)
        signal.signal(signal.SIGINT, _bail)

        # ADAPTIVE CLAIM PACING (2026-08-11): with the worker's min-idle
        # damper now 0 ("is it serving" is the only gate), thrash control
        # lives here, derived from observed contention instead of a fixed
        # model-idle rule. A resume that gets displaced quickly (< stable_s)
        # doubles the wait before the next claim, up to backoff cap; a run
        # that survives stable_s resets to the base poll. Under a busy chat
        # session the lease naturally goes quiet; on an idle card it retries
        # at full cadence.
        claim_wait = self.args.poll
        stable_s = 600.0

        while True:
            # WAIT for the card. First claim is immediate; retries poll.
            while True:
                with self.lock:
                    # resume policy also binds here: a lease that was already
                    # evicted and is waiting to re-claim when the operator
                    # disables resume stops waiting and exits. (A never-
                    # launched lease still gets its first launch — resume
                    # governs resumption, not admission.)
                    if self.evictions > 0 and not self.resume_enabled:
                        logger.info("resume DISABLED while waiting to re-claim"
                                    " — unregistering and exiting 75")
                        self.unregister()
                        return 75
                ok, why = self.claim()
                logger.info("claim: %s", why)
                if ok:
                    break
                with self.lock:
                    self.state = "waiting"
                self.resume_now.wait(timeout=claim_wait)
                self.resume_now.clear()

            self.launch()
            launched_at = time.time()

            # SUPERVISE until the child exits (naturally or paused away).
            next_beat = time.time() + self.args.heartbeat
            while True:
                with self.lock:
                    child = self.child
                if child is None or child.poll() is not None:
                    break
                if time.time() >= next_beat:
                    self.register()
                    next_beat = time.time() + self.args.heartbeat
                time.sleep(2)

            rc = child.returncode if child is not None else None
            with self.lock:
                yielded = (self.state == "yielded")

            # ORDER MATTERS: yielded is checked BEFORE rc==0. A SIGTERM'd job
            # that traps the signal may exit 0 mid-corpus; treating that as
            # done would silently truncate the batch. The cost of this order
            # on the true completion-races-pause case is one extra relaunch of
            # a resume-safe job, which then exits 0 with nothing to do.
            if yielded:
                with self.lock:
                    resume_enabled = self.resume_enabled
                if not resume_enabled:
                    # resume: disabled — an eviction is the END of this run,
                    # not a pause. EX_TEMPFAIL (75) so a systemd unit can
                    # distinguish "yielded by policy" from real failure
                    # (SuccessExitStatus=75 to treat it as clean).
                    logger.info("child paused away (eviction #%d) and resume "
                                "is DISABLED — unregistering and exiting 75",
                                self.evictions)
                    self.unregister()
                    return 75
                ran = time.time() - launched_at
                if ran < stable_s:
                    claim_wait = min(claim_wait * 2, 900.0)
                    logger.info("child paused away (eviction #%d) after only "
                                "%.0fs — card is contended, next claim in "
                                "%.0fs", self.evictions, ran, claim_wait)
                else:
                    claim_wait = self.args.poll
                    logger.info("child paused away (eviction #%d) after a "
                                "stable %.0fs run — waiting to re-claim",
                                self.evictions, ran)
                continue
            if rc == 0:
                logger.info("job COMPLETE after %d launch(es), %d eviction(s)"
                            " — unregistering", self.launches, self.evictions)
                with self.lock:
                    self.state = "done"
                self.unregister()
                return 0
            # "Consecutive" means consecutive: a child that ran healthily for
            # 10+ minutes before dying is a fresh incident, not an escalation
            # of a crash loop — without this reset, transient failures hours
            # apart would eventually kill a multi-day batch.
            if time.time() - launched_at > 600:
                self.failures = 0
            self.failures += 1
            logger.warning("child exited rc=%s (consecutive failure %d/%d)",
                           rc, self.failures, self.args.max_failures)
            if self.failures >= self.args.max_failures:
                logger.error("giving up after %d consecutive failures",
                             self.failures)
                self.unregister()
                return rc or 1
            time.sleep(min(300, 15 * self.failures))    # linear backoff


def _env_float(name: str, default: "float | None") -> "float | None":
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _adjust(argv: "list[str]") -> int:
    """--adjust mode: POST the worker's /ops/external/set for a live lease.
    The worker updates its registry (which its eviction paths enforce) and
    forwards the change to the lease supervisor's control URL, so both sides
    agree without restarting anything."""
    ap = argparse.ArgumentParser(
        prog="gpu_lease --adjust",
        description="Adjust a running lease's eviction/resume policy.")
    ap.add_argument("--adjust", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--key", required=True)
    ap.add_argument("--priority", choices=["evictable", "non-evictable"])
    ap.add_argument("--resume", choices=["enabled", "disabled"])
    ap.add_argument("--worker", default=_default_worker())
    args = ap.parse_args(argv)
    if args.priority is None and args.resume is None:
        ap.error("nothing to adjust: pass --priority and/or --resume")
    body: dict = {"model_key": args.key}
    if args.priority is not None:
        body["evictable"] = (args.priority != "non-evictable")
    if args.resume is not None:
        body["resume"] = args.resume
    res = _post_json(args.worker + "/ops/external/set", body, timeout=30.0)
    print(json.dumps(res if res is not None
                     else {"ok": False, "reason": "worker unreachable"},
                     indent=2))
    return 0 if (res or {}).get("ok") else 1


def main(argv: "list[str] | None" = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--adjust" in argv:
        return _adjust(argv)
    ap = argparse.ArgumentParser(
        prog="gpu_lease",
        description="Run a resume-safe GPU batch job as an evictable "
                    "pseudo-model of the hugpy worker.")
    ap.add_argument("--key", required=True,
                    help="model_key to register as (e.g. ocr:bluebook)")
    ap.add_argument("--vram-gib", type=float, required=True,
                    help="free VRAM (GiB) that must be claimable before launch")
    ap.add_argument("--priority", choices=["evictable", "non-evictable"],
                    default="evictable",
                    help="non-evictable = the external twin of static "
                         "residency: only an operator force-evict pauses it")
    ap.add_argument("--resume", choices=["enabled", "disabled"],
                    default="enabled",
                    help="disabled = an eviction ENDS the run (exit 75) "
                         "instead of waiting to re-claim the card")
    ap.add_argument("--worker", default=_default_worker(),
                    help="worker agent base URL (default: HUGPY_LEASE_WORKER "
                         "or http://127.0.0.1:${WORKER_PORT:-9100})")
    ap.add_argument("--control-host", default="127.0.0.1")
    ap.add_argument("--control-port", type=int,
                    default=int(_env_float("HUGPY_LEASE_CONTROL_PORT", 0) or 0),
                    help="0 = pick a free port")
    ap.add_argument("--min-idle", type=float,
                    default=_env_float("HUGPY_LEASE_MIN_IDLE_S", None),
                    help="only evict models idle >= this many seconds "
                         "(default: worker's HUGPY_EXTERNAL_MIN_IDLE_S, 0)")
    ap.add_argument("--poll", type=float,
                    default=_env_float("HUGPY_LEASE_POLL_S", 60.0),
                    help="seconds between claim retries while yielded")
    ap.add_argument("--grace", type=float,
                    default=_env_float("HUGPY_LEASE_GRACE_S", 15.0),
                    help="SIGTERM->SIGKILL grace for the child group")
    ap.add_argument("--heartbeat", type=float, default=45.0,
                    help="seconds between re-registers while running")
    ap.add_argument("--max-failures", type=int, default=5,
                    help="consecutive non-eviction child failures before "
                         "giving up")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- command to run")
    args = ap.parse_args(argv)

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given (put it after --)")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return Lease(args, cmd).run()


if __name__ == "__main__":
    sys.exit(main())
