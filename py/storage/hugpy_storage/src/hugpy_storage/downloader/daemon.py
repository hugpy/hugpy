"""``hugpy-downloader`` — the process that actually downloads models.

The console API enqueues; THIS claims and runs. Nothing about a transfer (the
spawned child, the progress walk, the stall killer, the HF size estimate, the
resume backoff) happens inside gunicorn any more, which is the entire point:
those threads shared a pool with ``/llm/workers/<id>/heartbeat``, and starving
it for 45s makes every worker read ``offline``.

The queue is the comms mirror that already existed for cross-process cancel —
no broker, no new IPC:

    API                                   daemon
    ───                                   ──────
    job_store.enqueue(kind="download")
      -> mirror row, status pending,      claim_next() compare-and-set under a
         claimed_by NULL, payload={model}    write lock -> exactly one runner
                                          job_store.create(id=<same id>) — the
                                            daemon becomes the OWNER
                                          run_download_job(...) to terminal
    POST /jobs/<id>/cancel
      -> mirror cancel flag               the store's watcher thread sees the
                                            flag, the engine kills the child

FAIL-OVER: a daemon that dies mid-transfer leaves rows claimed by a pid that no
longer exists. On startup we re-queue everything claimed by a previous owner
(``adopt_stale``); partial files stay on disk and the staging dir is adopted by
``download_one``, so a resumed transfer continues instead of refetching.

CONCURRENCY: HUGPY_DOWNLOADER_MAX_CONCURRENT (default 2) transfers at once, each
with its own monitor thread and its own spawned child — the same shape the API
had, just in a process where blocking costs nothing but a download slot.
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import sys
import threading
import time

from hugpy_control.jobs import job_store
from hugpy_storage.downloader import presence
from hugpy_storage.downloader.engine import DOWNLOAD_KIND, hf_token_present, run_download_job

logger = logging.getLogger("hugpy.downloader")

POLL_SECONDS = float(os.environ.get("HUGPY_DOWNLOADER_POLL_SECONDS", "1.5") or 1.5)
MAX_CONCURRENT = int(os.environ.get("HUGPY_DOWNLOADER_MAX_CONCURRENT", "2") or 2)
# How often an in-flight transfer narrates itself into the journal (k119). The
# old daemon logged only `polling (active=N)`: an operator could not tell a
# healthy 40GB transfer from a wedged one, nor a claimed job from a lost one.
PROGRESS_LOG_SECONDS = float(
    os.environ.get("HUGPY_DOWNLOADER_PROGRESS_LOG_SECONDS", "30") or 30)
# How often the idle heartbeat prints. It now carries QUEUE DEPTH and MIRROR
# HEALTH, because "polling (active=0)" was logged happily for 11 days while the
# mirror was quarantined and 65 download jobs rotted unclaimed.
IDLE_LOG_SECONDS = float(
    os.environ.get("HUGPY_DOWNLOADER_IDLE_LOG_SECONDS", "300") or 300)

_ADOPT_MESSAGE = "Re-queued after a downloader restart — resuming…"


def _human(n: int | float) -> str:
    """Bytes for a human, so a progress line is readable at a glance."""
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def owner_id() -> str:
    """Identity of THIS daemon on the shared queue. Host + pid, so a claim can
    be attributed, and a restart is always a different owner (which is what
    makes adopt_stale's "claimed by someone else" test correct)."""
    try:
        host = socket.gethostname()
    except OSError:
        host = "?"
    return f"{host}:{os.getpid()}"


class Downloader:
    def __init__(self, max_concurrent: int = MAX_CONCURRENT,
                 poll_seconds: float = POLL_SECONDS) -> None:
        self.owner = owner_id()
        self.max_concurrent = max(1, int(max_concurrent))
        self.poll_seconds = max(0.2, float(poll_seconds))
        self._active: dict[str, threading.Thread] = {}
        self._done: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # Latches so an outage is announced ONCE (and its recovery once), instead
        # of every 5 minutes forever or — as before — never.
        self._queue_down_since: float = 0.0

    # -- lifecycle -----------------------------------------------------------
    def stop(self, *_a) -> None:
        self._stop.set()

    def adopt(self) -> list[str]:
        """Take over work a dead predecessor left mid-flight."""
        adopted = job_store.adopt_stale((DOWNLOAD_KIND,), self.owner,
                                        message=_ADOPT_MESSAGE)
        if adopted:
            logger.warning("adopted %d download job(s) left by a previous "
                           "downloader: %s", len(adopted), ", ".join(adopted))
        return adopted

    # -- the queue -----------------------------------------------------------
    def _free_slots(self) -> int:
        with self._lock:
            for jid, t in list(self._active.items()):
                if not t.is_alive():
                    self._active.pop(jid, None)
            return self.max_concurrent - len(self._active)

    def _run_claimed(self, claimed: dict) -> None:
        """Become the OWNER of a claimed job and run it to a terminal state."""
        job_id = claimed.get("id")
        model_key = claimed.get("model_key") or ""
        payload = claimed.get("payload") or {}
        model = payload.get("model")
        total_bytes = payload.get("total_bytes") or claimed.get("total_bytes")

        if not model:
            # Nothing to download and no way to find out — fail it honestly
            # rather than leaving it to be claimed and dropped forever.
            job_store.create(model_key, id=job_id, kind=DOWNLOAD_KIND,
                             transport=claimed.get("transport"))
            job_store.finish(job_id, "failed",
                             error="queued job carries no model spec")
            return

        # Materialising the job LOCALLY is what makes this process its owner:
        # the engine's progress writes, the store's cancel watcher thread and
        # first-terminal-wins all operate on a local record.
        job_store.create(model_key, id=job_id, kind=DOWNLOAD_KIND,
                         transport=claimed.get("transport"),
                         model_name=claimed.get("model"),
                         payload=payload, total_bytes=total_bytes,
                         status="pending")
        hub_id = (model or {}).get("hub_id") or model_key
        logger.info("CLAIMED %s model_key=%s hub_id=%s total_bytes=%s",
                    job_id, model_key, hub_id, total_bytes or "?")
        started = time.time()
        # A per-claim progress ticker. `polling (active=N)` said nothing about
        # whether the transfer that IS running is moving — an operator watching
        # the journal could not tell a 40GB download from a wedged one. The
        # engine owns the job row; this thread only READS it and narrates.
        ticker = threading.Thread(
            target=self._narrate, args=(job_id, model_key), daemon=True,
            name=f"narrate-{str(job_id)[:8]}")
        ticker.start()
        try:
            status = run_download_job(job_id, model_key, model,
                                      total_bytes=total_bytes)
            job = job_store.get(job_id)
            err = getattr(job, "error", None) if job else None
            logger.info("%s %s (%s) after %.0fs%s",
                        "DONE" if status == "completed" else status.upper(),
                        job_id, model_key, time.time() - started,
                        f" — {err}" if err else "")
        except Exception as exc:  # noqa: BLE001 — a crash must not lose the job
            logger.exception("FAILED %s (%s) — download job crashed",
                             job_id, model_key)
            job_store.finish(job_id, "failed", error=exc)
        finally:
            self._done.add(job_id)

    def _narrate(self, job_id: str, model_key: str) -> None:
        """Log this transfer's progress every PROGRESS_LOG_SECONDS until it goes
        terminal, so the journal shows movement (or the absence of it) per job
        instead of only an idle heartbeat."""
        while job_id not in self._done and not self._stop.is_set():
            self._stop.wait(PROGRESS_LOG_SECONDS)
            job = job_store.get(job_id)
            if job is None or job_id in self._done:
                return
            pct = float(getattr(job, "progress", 0.0) or 0.0) * 100.0
            got = getattr(job, "downloaded_bytes", 0) or 0
            total = getattr(job, "total_bytes", 0) or 0
            bps = getattr(job, "bytes_per_second", None) or 0
            stalled = bool(getattr(job, "stalled", False))
            logger.info(
                "downloading %s (%s) %.1f%% %s/%s at %.1f MB/s%s",
                job_id, model_key, pct, _human(got),
                _human(total) if total else "?",
                bps / 1e6, "  [STALLED]" if stalled else "")

    # -- queue health ---------------------------------------------------------
    def _queue_health(self) -> dict:
        """Depth of the unclaimed download queue + the mirror's breaker state.

        This is the fact the daemon was missing. `claim_next` returning None is
        AMBIGUOUS — it means "nothing to do" and "I cannot see the queue at all"
        with the same value, and the second case is what the download outage
        actually was."""
        mirror = job_store.mirror
        if mirror is None:
            return {"queued": "?", "mirror": "absent", "ok": False}
        try:
            h = mirror.health()
        except Exception:  # noqa: BLE001 — health must never break the loop
            return {"queued": "?", "mirror": "unknown", "ok": True}
        if not h.get("ok"):
            return {"queued": "?",
                    "mirror": f"QUARANTINED retry_in={h.get('retry_in', 0):.0f}s"
                              f" last={h.get('last_error') or '?'}",
                    "ok": False}
        try:
            depth = mirror.claimable_count((DOWNLOAD_KIND,))
        except Exception:  # noqa: BLE001
            depth = "?"
        return {"queued": depth, "mirror": "ok", "ok": True}

    def _watch_queue_health(self) -> None:
        """Announce a queue outage ONCE, and its recovery once. The 2026-08-10
        outage logged a single ERROR and then eleven days of cheerful INFO."""
        h = self._queue_health()
        if not h["ok"] and not self._queue_down_since:
            self._queue_down_since = time.time()
            logger.error(
                "QUEUE UNREACHABLE (%s) — this daemon can claim NOTHING; every "
                "Add-model request will sit queued until it recovers. The mirror "
                "retries itself; check %s.",
                h["mirror"], os.environ.get("HUGPY_COMMS_DB") or "(default db)")
        elif h["ok"] and self._queue_down_since:
            down = time.time() - self._queue_down_since
            self._queue_down_since = 0.0
            logger.warning("QUEUE RECOVERED after %.0fs — resuming claims "
                           "(queued=%s)", down, h["queued"])

    def _dispatch(self) -> int:
        started = 0
        while not self._stop.is_set() and self._free_slots() > 0:
            claimed = job_store.claim_next((DOWNLOAD_KIND,), self.owner)
            if not claimed:
                break
            job_id = claimed.get("id")
            if not job_id:
                break
            t = threading.Thread(target=self._run_claimed, args=(claimed,),
                                 name=f"download-{str(job_id)[:8]}",
                                 daemon=True)
            with self._lock:
                self._active[job_id] = t
            t.start()
            started += 1
        return started

    def run(self) -> int:
        logger.info("hugpy downloader starting — owner=%s max_concurrent=%d "
                    "poll=%.1fs comms_db=%s hf_token=%s",
                    self.owner, self.max_concurrent, self.poll_seconds,
                    os.environ.get("HUGPY_COMMS_DB") or "(default)",
                    # Boolean only — a token value is never logged.
                    "present" if hf_token_present() else "ABSENT (gated/private "
                    "repos will fail; save one in the console)")
        if job_store.mirror is None:
            logger.error("no cross-process comms mirror (HUGPY_COMMS_DB=off or "
                         "unwritable) — there is no queue to serve. Refusing to "
                         "run as a no-op.")
            return 2
        self.adopt()
        presence.beat(self.owner)
        last_beat = time.time()
        idle_logged = 0.0
        while not self._stop.is_set():
            try:
                started = self._dispatch()
                now = time.time()
                if now - last_beat >= presence.BEAT_INTERVAL:
                    presence.beat(self.owner)
                    last_beat = now
                self._watch_queue_health()
                if started:
                    idle_logged = now
                elif now - idle_logged >= IDLE_LOG_SECONDS:
                    # A heartbeat in the journal, so an operator can see the
                    # daemon is polling and not wedged. Every 5 min, not every
                    # tick — this runs ~40x a minute. It now reports the QUEUE
                    # (depth + mirror health), because the old line said
                    # "polling (active=0)" for 11 days while the mirror was
                    # quarantined and every Add-model request rotted unclaimed.
                    with self._lock:
                        active = len(self._active)
                    h = self._queue_health()
                    logger.info("polling (active=%d queued=%s mirror=%s)",
                                active, h["queued"], h["mirror"])
                    idle_logged = now
            except Exception:  # noqa: BLE001 — the loop must never die
                logger.exception("downloader poll failed")
            self._stop.wait(self.poll_seconds)
        presence.clear()
        logger.info("hugpy downloader stopping")
        return 0


def main(argv: list[str] | None = None) -> int:
    # basicConfig alone is NOT enough: importing the package installs handlers on
    # the root logger (the tree logs to a file), and basicConfig is a no-op once
    # root has any handler — the daemon would run silently in journalctl, which
    # is the one place an operator looks. Attach our own stdout handler and set
    # the level explicitly.
    level = os.environ.get("HUGPY_DOWNLOADER_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    logging.getLogger("hugpy_storage.downloader").addHandler(handler)
    logging.getLogger("hugpy_storage.downloader").setLevel(level)
    for noisy in ("httpx", "httpcore", "huggingface_hub", "filelock", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.INFO)
    d = Downloader()
    # SIGTERM (systemd stop/restart) ends the poll loop. In-flight children die
    # with the cgroup; their rows are re-queued and RESUMED by the next daemon
    # (see adopt()), which is why a restart mid-transfer is safe.
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, d.stop)
        except (ValueError, OSError):
            pass
    return d.run()


if __name__ == "__main__":
    raise SystemExit(main())
