"""The model-transfer lifecycle — FLASK-FREE, so the downloader daemon can run
it without importing the web stack.

MOVED here (2026-07-28) from
``flask_app/app/functions/downloads/cancelable_downloads.py``, which keeps thin
re-exports for the route layer and the existing tests. This is a MOVE, not a
rewrite: every behaviour below is a hard-won fix and is preserved verbatim —

  * SPAWN, never fork (2026-07-27). ``mp.Process`` defaults to fork on Linux;
    forking a multi-threaded server copies locks in whatever state they were in,
    including the ``fcntl.flock``'d worker registry every heartbeat takes. That
    is how one download "pushed off all of the workers". Spawn starts a clean
    interpreter.
  * STAGED-BYTES progress (2026-07-12). Atomic provisioning downloads into a
    ``<dest>.tmp-<pid>`` sibling and renames on completion, so measuring `dest`
    alone reported 0% for the entire in-flight window.
  * The STALL KILLER: no new bytes for STALL_SECONDS -> kill the process group
    and resume; up to MAX_ATTEMPTS with backoff. HF keeps partial files, and the
    staging dir is adopted by name, so a resume continues rather than refetches.
  * A BOUNDED size estimate (2026-07-27). The estimate is a network round-trip
    that only turns an indeterminate bar into a percentage; it may never park a
    thread for the life of a download.

What is NEW here is only WHERE it runs: ``run_download_job`` is the synchronous
form the daemon calls on its own worker thread, and it is the one that owns the
job. ``start_cancellable_download`` remains as the fire-and-forget wrapper.
"""
from __future__ import annotations

import fnmatch
import logging
import multiprocessing as mp
import os
import tempfile
import threading
import time
from typing import Optional

from hugpy_platform.procutil import terminate_tree
from hugpy_control.jobs import Job, job_store
# Module-level (not call-local) so these are ONE seam: the spawned child
# re-imports this module anyway, and the tests that drive the monitor without a
# real store/network patch them here.
from hugpy_storage.download_models import download_one, record_downloaded_model, staged_bytes
from hugpy_storage.catalog_source import catalog_refresh
from hugpy_storage.events import publish_catalog_changed
from hugpy_storage.model_paths import route_destination, split_hub_id

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Tunables (env-overridable). A download that writes no new bytes for
# STALL_SECONDS is considered stalled and gets killed + resumed. Each download
# is attempted up to MAX_ATTEMPTS times; HF keeps partial files on disk so a
# resume picks up where the previous attempt stopped.
# ──────────────────────────────────────────────────────────────────────────
STALL_SECONDS = int(os.environ.get("HUGPY_DOWNLOAD_STALL_SECONDS", "180"))
MAX_ATTEMPTS = int(os.environ.get("HUGPY_DOWNLOAD_MAX_ATTEMPTS", "4"))
# How long a child may take to produce its FIRST byte before the stall killer
# treats it as wedged (k119). Distinct from STALL_SECONDS on purpose: a spawn
# child re-imports the package (registry walk included) before it ever opens a
# socket, and that boot has been measured past 180s on a loaded virtiofs box —
# the killer was ending transfers that had not yet had a chance to start.
STARTUP_GRACE_SECONDS = int(
    os.environ.get("HUGPY_DOWNLOAD_STARTUP_GRACE_SECONDS", "600"))

# How long the (network) size estimate may take before the download proceeds
# with an indeterminate progress bar. Deliberately short: the estimate is a
# NICETY, while the thread it occupies is shared with everything else.
_ESTIMATE_TIMEOUT_S = float(os.environ.get("HUGPY_HF_ESTIMATE_TIMEOUT_S", "8") or 8)

# The job kind this engine executes. One constant so the API's enqueue, the
# daemon's claim filter and the /jobs view can never disagree about the name.
DOWNLOAD_KIND = "download"


# ──────────────────────────────────────────────────────────────────────────
# Error hand-off across the process boundary — the download runs in a child
# process, so it writes its failure reason to a temp file the monitor reads.
# ──────────────────────────────────────────────────────────────────────────
def _error_path(job_id: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"hugpy-download-{job_id}.err")


def _write_error(job_id: str, msg: str) -> None:
    try:
        with open(_error_path(job_id), "w", encoding="utf-8") as fh:
            fh.write(msg[:2000])
    except OSError:
        pass


def _read_error(job_id: str) -> Optional[str]:
    try:
        with open(_error_path(job_id), "r", encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _clear_error(job_id: str) -> None:
    try:
        os.remove(_error_path(job_id))
    except OSError:
        pass


# ──────────────────────────────────────────────────────────────────────────
# TERMINAL vs RETRYABLE failures (k119)
# ──────────────────────────────────────────────────────────────────────────
# The resume loop below retries MAX_ATTEMPTS times with backoff. That is right
# for a dropped connection and WRONG for a gated repo, a bad token or a repo
# that does not exist: those fail identically four times, take ~1 minute of
# backoff to say so, and then surface the raw exception text
# ("GatedRepoError: 401 Client Error…"), which tells an operator nothing about
# what to DO. Classify them once, fail fast, and answer in an imperative.
#
# Matched on the exception's rendered "Type: message" string because the child
# process hands its failure across the boundary as text (see _write_error) —
# there is no exception object left to isinstance() by the time we decide.
_GATED_MARKERS = ("gatedrepoerror", "is a gated repo", "access to model",
                  "you must be authenticated", "awaiting a review",
                  "accept the conditions")
_AUTH_MARKERS = ("401 client error", "unauthorized", "invalid credentials",
                 "invalid user token", "repository not found")
_MISSING_MARKERS = ("repositorynotfounderror", "entrynotfounderror",
                    "revisionnotfounderror", "404 client error")
_DISK_MARKERS = ("no space left on device", "oserror: [errno 28]", "errno 28")


def classify_download_error(detail: str) -> Optional[tuple[str, str]]:
    """(reason_code, operator-facing message) for a TERMINAL failure, or None
    when the failure is the retryable kind the resume loop should keep at.

    reason codes: ``gated_repo`` | ``auth`` | ``not_found`` | ``no_space``."""
    d = (detail or "").lower()
    if not d:
        return None
    if any(m in d for m in _GATED_MARKERS):
        return ("gated_repo",
                "Gated repo: this model requires accepting its license on "
                "huggingface.co AND an HF token with access. Accept the terms on "
                "the model page, then save a token in the console (Settings → HF "
                "token) and retry — no restart needed.")
    if any(m in d for m in _AUTH_MARKERS):
        return ("auth",
                "Hugging Face rejected the credentials (401/404-as-private). Save "
                "a valid HF token in the console (Settings → HF token) and retry; "
                "a private or gated repo is invisible without one.")
    if any(m in d for m in _MISSING_MARKERS):
        return ("not_found",
                "Hugging Face has no such repo/file at that revision — check the "
                "hub id and the selected quant/filename.")
    if any(m in d for m in _DISK_MARKERS):
        return ("no_space",
                "Out of disk space on the model store — free space or evict a "
                "model, then retry.")
    return None


def invalidate_model_status_cache(reason: str = "", model_key: str = None) -> None:
    """Tell every process that a model's physical state just changed.

    THE one invalidation entry point, called from the events that actually move
    a model between not_installed/partial/installed. Two things happen:

      * the PERSISTED physical record is dropped — targeted when ``model_key``
        is known (delete / prune / a download reaching a terminal state), so the
        other rows stay warm; whole-table otherwise.
      * the ``central-holdings`` memo (comms/model_status_cache.py) is flushed —
        it answers a DIFFERENT question ("can central provide this model to a
        worker?") off the same presence facts, so it must hear the same events.

    NOTE the cross-process caveat now that downloads run in the daemon: both
    stores are persisted (comms/), so dropping the record here is visible to the
    API — but the API's own in-process memoisation of a model row expires on its
    own clock. The console's model list re-derives on the single-model detail
    read, which is the explicit refresh path.

    Best-effort by design: a cache that cannot be invalidated must never break
    the operation that changed the store."""
    try:
        from hugpy_storage.model_physical import forget_physical
        forget_physical(model_key, reason)
    except Exception:  # noqa: BLE001
        logger.debug("model physical-state invalidation failed (%s)", reason,
                     exc_info=True)
    try:
        from hugpy_storage.model_status_cache import invalidate_model_status
        invalidate_model_status(reason)
    except Exception:  # noqa: BLE001
        logger.debug("model-status cache invalidation failed (%s)", reason,
                     exc_info=True)
    # And say so on the control bus: the engine's catalog (and anything else
    # that derives from the physical inventory) hears every move here.
    publish_catalog_changed(reason, model_key=model_key)


def _estimate_total_bytes(model: dict) -> Optional[int]:
    """Sum the sizes of exactly the files this download will fetch, so the
    progress bar can show a real percentage. Respects filename (single GGUF),
    include patterns, or full repo. Returns None on any failure -> the bar
    falls back to indeterminate, which still works."""
    hub_id = model.get("hub_id")
    if not hub_id:
        return None
    from hugpy_platform import constants as _c
    repo_id, _ = split_hub_id(hub_id)
    # Per-repo metadata rides the permanent central HF cache (fetch-once —
    # comms/model_metadata.py): only the first estimate of a repo ever pings HF.
    from hugpy_storage.model_metadata import fetch_repo_info
    try:
        payload = fetch_repo_info(repo_id, files_metadata=True,
                                  api=getattr(_c, "hfApi", None))
    except Exception as exc:
        logger.info("size estimate failed for %s: %s", hub_id, exc)
        return None
    if payload is None:
        return None

    filename = model.get("filename")
    include = model.get("include")

    def will_download(path: str) -> bool:
        if filename:
            return path == filename or path.endswith("/" + filename)
        if include:
            pats = include if isinstance(include, list) else [include]
            return any(fnmatch.fnmatch(path, p) for p in pats)
        return True

    total = sum((s.get("size") or 0) for s in (payload.get("siblings") or [])
                if will_download(s.get("rfilename") or ""))
    return total or None


def _estimate_total_bytes_bounded(model: dict) -> Optional[int]:
    """_estimate_total_bytes with a hard wall-clock bound.

    The underlying HfApi call cannot be interrupted, so the worker thread it
    occupies is left to finish on its own (daemon, so it can never block
    shutdown) — what we bound is how long WE wait for it. Returns None on
    timeout: an indeterminate bar, which is strictly better than a stalled API.

    A bare daemon thread, NOT ThreadPoolExecutor: the executor's context manager
    JOINS its workers on __exit__, so `with … as ex` waits for the hung call
    anyway and the timeout buys nothing. Verified the hard way — a 2s bound over
    a 60s call still returned at 60s.
    """
    box: dict = {}

    def _run():
        try:
            box["v"] = _estimate_total_bytes(model)
        except Exception as exc:  # noqa: BLE001
            box["e"] = exc

    t = threading.Thread(target=_run, name="hf-estimate", daemon=True)
    t.start()
    t.join(_ESTIMATE_TIMEOUT_S)
    if t.is_alive():
        logger.warning(
            "HF size estimate exceeded %.0fs — continuing with an indeterminate "
            "progress bar (the download itself is unaffected)",
            _ESTIMATE_TIMEOUT_S)
        return None
    if "e" in box:
        logger.info("HF size estimate unavailable (%s)", box["e"])
        return None
    return box.get("v")


# ──────────────────────────────────────────────────────────────────────────
# Subprocess worker — module-level so it's spawn-safe. Captures the real
# failure reason (HF errors propagate out of download_one) into the error file,
# then re-raises so the process exits non-zero and the monitor sees the failure.
# ──────────────────────────────────────────────────────────────────────────
def hf_token_present() -> bool:
    """Is an HF token in force for THIS process? Boolean only — the value is
    never returned, logged, or put on the wire.

    The console-saved token (0600 file at ``HF_TOKEN_PATH``) is seeded into the
    env by ``imports.src.constants.constants`` at import, and the transfer child
    is SPAWNED, so it re-imports and re-reads that file — a token saved in the
    console reaches the very next transfer with no daemon restart. This function
    exists so the daemon can SAY which of the two worlds it is in when a gated
    repo fails."""
    try:
        from hugpy_platform.constants import read_stored_hf_token
        if read_stored_hf_token():
            return True
    except Exception:  # noqa: BLE001 — a diagnostic must never break a download
        pass
    return bool((os.environ.get("HF_TOKEN")
                 or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip())


def _download_worker(job_id: str, model_key: str, model: dict) -> None:
    os.setpgrp()
    try:
        download_one(model=model, model_key=model_key)   # writes hugpy.json via _stamp
        _clear_error(job_id)
    except Exception as exc:
        # Tag an auth-shaped failure with whether we were even AUTHENTICATED.
        # "GatedRepoError" with no token and "GatedRepoError" with a token are
        # different operator actions (save a token vs accept the license), and
        # the raw HF text does not distinguish them.
        detail = f"{type(exc).__name__}: {exc}"
        if classify_download_error(detail) and not hf_token_present():
            detail += " [no HF token configured on this host]"
        _write_error(job_id, detail)
        raise


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _progress_bytes(dest: str) -> int:
    """Bytes on disk for a download IN PROGRESS: the final `dest` (once
    promoted) PLUS any still-live `.tmp-<pid>` staging sibling(s) — atomic
    provisioning (imports/apis/download_models.py) lands new work in staging
    and only renames it onto `dest` on completion, so `_dir_bytes(dest)` alone
    always read 0 (0%) for the entire in-flight window (operator-felt
    regression, fixed 2026-07-12: a staging dir was seen growing 3.7GB->4.4GB
    in 3s while the console showed 0%). Safe to sum both — see
    download_models.staged_bytes' docstring for why this never double-counts."""
    return _dir_bytes(dest) + staged_bytes(dest)


def _is_cancelled(job_id: str) -> bool:
    """A cancel that this process must obey.

    Checks ``cancel_requested`` as well as a terminal ``cancelled`` status: the
    request can arrive by two routes now. A cancel POST landing on the API force-
    marks the mirror row and raises the shared flag; the store's watcher thread
    in THIS process turns that flag into a local cancel. Whichever of the two
    lands first, the resume loop must stop."""
    cur = job_store.get(job_id)
    if cur is None:
        return False
    return bool(cur.cancel_requested) or cur.status == "cancelled"


def _watch(proc, job_id: str, dest: str, total_bytes: Optional[int],
           baseline: int = 0) -> bool:
    """Sample progress every second while ``proc`` runs.

    Reports bytes/sec and percentage. Returns True if the transfer STALLED
    (no new bytes for STALL_SECONDS) — in which case the process group is
    killed so it can be resumed — or False if the process exited on its own.

    ``baseline`` is what was ALREADY on the destination when this job started
    (k119). Fetching one missing quant into a repo dir that already holds 26GB
    of other quants otherwise rendered as "99.9% — 26.1GB/6.3GB", because
    _progress_bytes measures the whole dir. The new file lands in the STAGING
    sibling and is only merged at promote time, so subtracting the starting dir
    size is exactly the bytes THIS transfer is responsible for. A resume's own
    partials live in staging, above the baseline, and stay counted.
    """
    last_bytes = _progress_bytes(dest)
    last_change = time.time()
    prev_bytes, prev_t = last_bytes, last_change

    while proc.is_alive():
        time.sleep(1.0)
        if _is_cancelled(job_id):
            return False
        now = time.time()
        got = _progress_bytes(dest)
        bps = max(got - prev_bytes, 0) / max(now - prev_t, 1e-6)
        prev_bytes, prev_t = got, now
        if got > last_bytes:
            last_bytes, last_change = got, now
        # THIS transfer's bytes, not the directory's (see `baseline`).
        mine = max(got - baseline, 0)
        pct = (mine / total_bytes) if total_bytes else 0.0
        job_store.update(job_id, progress=min(pct, 0.999),
                         downloaded_bytes=mine, bytes_per_second=bps,
                         stalled=False)

        # STARTUP GRACE (k119). The stall clock used to start the instant the
        # child was spawned — but a `spawn` child re-imports the whole package
        # (including the model-registry walk) before it opens a socket, which on
        # a loaded virtiofs box has been measured past three minutes. The killer
        # then fired before a single byte could exist, killed a child that was
        # merely booting, and did it again on every attempt: a download that
        # "constantly does not download" while looking busy. Until the FIRST
        # bytes of this transfer land, allow a longer window; once bytes are
        # flowing, STALL_SECONDS is the honest no-progress rule it always was.
        idle_budget = STALL_SECONDS if mine > 0 else STARTUP_GRACE_SECONDS
        if (now - last_change) >= idle_budget:
            job_store.update(job_id, stalled=True)
            terminate_tree(proc)
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────
# The run: spawn the worker under a monitor that auto-resumes a stalled/failed
# transfer with backoff, surfaces the real error, and resolves the terminal
# state. A cancel at any point stops the loop and kills the child.
# ──────────────────────────────────────────────────────────────────────────
def run_download_job(job_id: str, model_key: str, model: dict,
                     total_bytes: Optional[int] = None) -> str:
    """Run one download to a terminal state, SYNCHRONOUSLY, in the calling
    thread. The caller (the daemon's worker thread) must already hold the job in
    the local store — this function owns its lifecycle from here.

    Returns the terminal status: "completed" | "failed" | "cancelled".
    """
    dest = route_destination(model=model)
    # What is ALREADY there before this job starts. Subtracted from every
    # progress sample so a one-file fetch into a populated repo dir reports its
    # own bytes (k119) instead of the directory's.
    baseline = _dir_bytes(dest) if os.path.isdir(dest) else 0
    logger.info("download %s -> %s (baseline on disk: %d bytes)",
                model_key, dest, baseline)

    job_store.update(
        job_id, status="running", message="Downloading…",
        total_bytes=total_bytes, attempt=1, max_attempts=MAX_ATTEMPTS,
        stalled=False, error=None, _model=model,
    )

    def _spawn():
        _clear_error(job_id)
        # SPAWN, NOT FORK — see the module docstring. Cheap insurance that costs
        # one interpreter start (~0.3s) per attempt, irrelevant next to a
        # multi-GB transfer, and the child already takes only picklable args.
        ctx = mp.get_context("spawn")
        p = ctx.Process(target=_download_worker,
                        args=(job_id, model_key, model), daemon=True)
        p.start()
        job_store.update(job_id, _proc=p)
        return p

    def _cancelled_now(proc=None) -> bool:
        """Cancel check that also TAKES THE CHILD DOWN. _watch returning on a
        cancel used to be followed by proc.join(), which blocks until the
        transfer finishes on its own — the child has to be killed first."""
        if not _is_cancelled(job_id):
            return False
        if proc is not None and proc.is_alive():
            terminate_tree(proc)
        return True

    if total_bytes is None:
        # SIZE ESTIMATE IS A NETWORK CALL. Bounded (see
        # _estimate_total_bytes_bounded) so a slow/unreachable HF cannot park
        # this thread for the life of the download. It is off the API process
        # entirely now, but the bound stays: an unbounded wait would still hold
        # a daemon slot doing nothing.
        total_bytes = _estimate_total_bytes_bounded(model)
        if total_bytes:
            job_store.update(job_id, total_bytes=total_bytes)

    attempt = 1
    while True:
        if attempt > 1:
            job_store.update(
                job_id, attempt=attempt, status="running", stalled=False,
                message=f"Resuming (attempt {attempt}/{MAX_ATTEMPTS})…",
            )
        proc = _spawn()
        stalled = _watch(proc, job_id, dest, total_bytes, baseline)
        if _cancelled_now(proc):
            proc.join()
            return _finish_cancelled(job_id, model_key)
        proc.join()

        if _cancelled_now(proc):
            return _finish_cancelled(job_id, model_key)

        if not stalled and proc.exitcode == 0:
            job_store.update(
                job_id, status="completed", progress=1.0, stalled=False,
                downloaded_bytes=max(_progress_bytes(dest) - baseline, 0),
                error=None,
                bytes_per_second=None, message=f"Installed at {dest}",
            )
            try:
                record_downloaded_model(model, dest)
                # The registry is engine state: the installed catalog source
                # re-derives it from the discovery report (the engine's
                # refresh_registry(run_discovery=False)); the null source is a
                # no-op and the catalog.changed event below carries the news.
                catalog_refresh()
            except Exception as exc:
                logger.warning("post-download registry refresh failed: %s", exc)
            # not_installed/partial -> installed. The refresh already
            # invalidates, but say it explicitly here too: if the refresh above
            # raised, the listings must still stop reporting the old status.
            # Over-invalidating costs one re-walk; under-invalidating hides a
            # model the operator just downloaded.
            invalidate_model_status_cache(
                f"download completed: {model_key}", model_key=model_key)
            publish_catalog_changed("download completed", model_key=model_key,
                                    hub_id=(model or {}).get("hub_id"),
                                    destination=dest, change="download")
            return "completed"

        # Failed or stalled — figure out why, then resume or give up.
        detail = _read_error(job_id) or (
            f"stalled: no new data for {STALL_SECONDS}s"
            if stalled else f"worker exited with code {proc.exitcode}"
        )

        # TERMINAL failures short-circuit the resume loop. Retrying a gated repo
        # or a bad token four times just delays the same answer by a minute of
        # backoff and then prints raw HF exception text; the operator needs the
        # ACTION, immediately. `error_reason` is the typed code the console can
        # branch on; `error` stays the raw detail so nothing is hidden.
        verdict = classify_download_error(detail)
        if verdict is not None:
            reason, guidance = verdict
            logger.warning("download %s (%s) is terminal [%s]: %s",
                           job_id, model_key, reason, detail)
            job_store.update(
                job_id, status="failed", stalled=False, bytes_per_second=None,
                message=guidance, error=detail, error_reason=reason,
            )
            invalidate_model_status_cache(
                f"download failed ({reason}): {model_key}", model_key=model_key)
            return "failed"

        if attempt >= MAX_ATTEMPTS:
            job_store.update(
                job_id, status="failed", stalled=stalled, bytes_per_second=None,
                message="Download stalled." if stalled else "Download failed.",
                error=detail,
            )
            # Terminal too: a give-up leaves partial files at the destination,
            # which is a real not_installed -> partial move.
            invalidate_model_status_cache(
                f"download failed: {model_key}", model_key=model_key)
            return "failed"

        backoff = min(2 ** attempt, 30)
        job_store.update(
            job_id, status="running", stalled=stalled, error=detail,
            message=(f"{'Stalled' if stalled else 'Error'}; retrying in {backoff}s "
                     f"(attempt {attempt + 1}/{MAX_ATTEMPTS})…"),
        )
        for _ in range(backoff):
            if _cancelled_now():
                return _finish_cancelled(job_id, model_key)
            time.sleep(1.0)
        attempt += 1


def _finish_cancelled(job_id: str, model_key: str) -> str:
    """Teardown-side terminal write for a cancel. First-terminal-wins in the
    store makes this a no-op if the cancelling process already force-marked the
    row — the point is that the OWNER confirms the resources are released."""
    job_store.update(job_id, status="cancelled", message="Cancelled by user.",
                     stalled=False, bytes_per_second=None)
    # Cancel is terminal and leaves whatever landed on disk behind — the row's
    # status can move (not_installed -> partial).
    invalidate_model_status_cache(f"download cancelled: {model_key}",
                                  model_key=model_key)
    return "cancelled"


def start_cancellable_download(job: Job, model: dict,
                               total_bytes: Optional[int] = None) -> None:
    """Fire-and-forget wrapper: run_download_job on a daemon thread.

    IN-PROCESS by definition, so it is NOT what the API calls any more (that is
    the whole point of the daemon). Kept because the daemon itself has no use
    for a thread wrapper but the tests and any single-process embedding do.
    """
    threading.Thread(
        target=run_download_job,
        args=(job.id, job.model_key, model),
        kwargs={"total_bytes": total_bytes},
        name=f"download-{job.id[:8]}", daemon=True).start()
