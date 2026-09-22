"""Download progress-honesty + staging-orphan-reaper regression (2026-07-12).

The atomic-provisions change (hugpy_storage.download_models) lands an in-flight
download in a per-pid staging sibling ``<dest>.tmp-<pid>`` and only renames it
onto the final ``<dest>`` on completion (integrity fix — a partial pull can
never sit at a resolvable model path). The download-progress reader
(hugpy_storage.downloader.engine, re-exported by console.cancelable_downloads)
measured bytes at ``dest`` ONLY, so every in-flight download showed 0% until
the finishing rename.

This file exercises, without touching the real model store:
  * progress honesty  — the fixed `_progress_bytes` reads staging bytes while
    in flight and final-dest bytes once promoted, never both (rename-safe);
  * the orphan reaper — dead-pid + stale staging is removed, live-pid staging
    is left alone, young dead-pid staging survives the grace window;
  * adopt-on-resume   — a fresh run's staging dir adopts (renames onto
    itself) the newest dead-pid orphan for the SAME dest instead of
    re-fetching from zero, and ages out any other orphans past grace;
  * the discover-walk hook actually calls the reaper (wiring check, no real
    filesystem/network walk).
"""
import os
import shutil
import subprocess
import time

import pytest

from hugpy_storage import download_models as dm
from hugpy_engine.apis import get_module as gm
from hugpy_storage.console import cancelable_downloads as cd

MB = 1024 * 1024


def wfile(path, mb=2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"\0" * (mb * MB))


def dead_pid() -> int:
    """A pid that is GUARANTEED not alive: spawn a trivial child, wait for it
    (reaped), return its pid. Called only inside tests — nothing forks at import."""
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def age(path, seconds_ago):
    t = time.time() - seconds_ago
    os.utime(path, (t, t))


# ===========================================================================
# PROGRESS HONESTY — the actual regression site (_progress_bytes)
# ===========================================================================
def test_progress_honesty_reads_staging_then_dest(tmp_path):
    tmp = str(tmp_path)
    dest = os.path.join(tmp, "models", "transformers", "own", "in-flight-model")
    staged = dest + f".tmp-{os.getpid()}"
    wfile(os.path.join(staged, "model-00001.safetensors"), 3)
    wfile(os.path.join(staged, "model-00002.safetensors"), 2)
    staged_total = 5 * MB

    assert not os.path.exists(dest)                 # still mid-download
    # OLD read path (_dir_bytes(dest) alone) is the live 0% bug
    assert cd._dir_bytes(dest) == 0
    # FIXED read path reports the staging bytes
    assert cd._progress_bytes(dest) == staged_total
    assert dm.staged_bytes(dest) == staged_total

    # simulate _promote_staged's common-path rename (dest didn't exist -> os.rename)
    os.rename(staged, dest)
    assert not os.path.exists(staged)
    # progress reads the final dest, same total (no double count)
    assert cd._progress_bytes(dest) == staged_total
    assert dm.staged_bytes(dest) == 0

    # a second, freshly-arriving staging dir for a DIFFERENT in-flight attempt of
    # the same dest (e.g. a resume after promote already happened) must not be
    # double-counted against the now-complete dest
    staged2 = dest + ".tmp-999999"
    wfile(os.path.join(staged2, "extra.safetensors"), 1)
    assert cd._progress_bytes(dest) == staged_total + 1 * MB


# ===========================================================================
# ORPHAN REAPER — reap_orphaned_staging (store-wide sweep, hooked at discover)
# ===========================================================================
def test_orphan_reaper_respects_pid_liveness_and_grace(tmp_path):
    root = str(tmp_path / "reaper-store")
    models = os.path.join(root, "models", "transformers", "own")

    old_dead_dir = os.path.join(models, "repo-a") + f".tmp-{dead_pid()}"
    wfile(os.path.join(old_dead_dir, "half.bin"), 1)
    age(old_dead_dir, 3600)                          # 1h old — well past grace

    live_dir = os.path.join(models, "repo-b") + f".tmp-{os.getpid()}"
    wfile(os.path.join(live_dir, "half.bin"), 1)
    age(live_dir, 3600)                              # old mtime but LIVE pid

    young_dead_dir = os.path.join(models, "repo-c") + f".tmp-{dead_pid()}"
    wfile(os.path.join(young_dead_dir, "half.bin"), 1)
    # fresh mtime (just written) — within the grace window

    removed = dm.reap_orphaned_staging(root=root, grace_seconds=600)

    # dead-pid + stale (past grace) staging IS removed
    assert old_dead_dir in removed and not os.path.exists(old_dead_dir)
    # live-pid staging is NEVER touched (even with an old mtime)
    assert live_dir not in removed and os.path.exists(live_dir)
    # dead-pid + young (within grace) staging is KEPT
    assert young_dead_dir not in removed and os.path.exists(young_dead_dir)

    # age the young one out and re-sweep with a short grace -> now it goes too
    age(young_dead_dir, 5)
    removed2 = dm.reap_orphaned_staging(root=root, grace_seconds=1)
    assert young_dead_dir in removed2 and not os.path.exists(young_dead_dir)
    assert os.path.exists(live_dir)


# ===========================================================================
# ADOPT-ON-RESUME — _adopt_or_reap_staging (per-dest hook before a new pull)
# ===========================================================================
def test_adopt_on_resume_takes_newest_dead_orphan(tmp_path):
    tmp = str(tmp_path)
    adest = os.path.join(tmp, "models", "transformers", "own", "resumable-model")

    orphan_old = adest + f".tmp-{dead_pid()}"
    wfile(os.path.join(orphan_old, "shard1.bin"), 1)
    # left with a FRESH mtime here on purpose: it must survive the upcoming
    # grace_seconds=600 adopt call below (aged out only in the later, separate
    # short-grace sweep) — orphan_new is created after it so it's naturally the
    # newer of the two, no manual aging needed to pick the adoption winner.

    orphan_new = adest + f".tmp-{dead_pid()}"
    wfile(os.path.join(orphan_new, "shard1.bin"), 4)     # further along -> newest

    fresh_staged = dm._staging_dir(adest)                # this run's own pid
    result = dm._adopt_or_reap_staging(adest, fresh_staged, grace_seconds=600)

    assert result == fresh_staged
    # the NEWEST dead orphan was renamed onto the fresh staged path (adopted)
    assert os.path.isdir(fresh_staged)
    assert os.path.getsize(os.path.join(fresh_staged, "shard1.bin")) == 4 * MB
    # it WAS the rename, not a copy
    assert not os.path.exists(orphan_new)
    # the OLDER orphan is untouched (still within its own grace check here)
    assert os.path.exists(orphan_old)

    # re-run with a short grace: the older, non-adopted orphan should now age out
    age(orphan_old, 3600)
    result2 = dm._adopt_or_reap_staging(adest + "-other", adest + "-other.tmp-nope",
                                        grace_seconds=1)
    # no-op when there are no siblings for that dest
    assert result2 == adest + "-other.tmp-nope"
    # direct grace check on the untouched older orphan via the store-wide reaper
    removed4 = dm.reap_orphaned_staging(root=tmp, grace_seconds=1)
    assert orphan_old in removed4 and not os.path.exists(orphan_old)


def test_adopt_on_resume_never_touches_live_sibling(tmp_path):
    """A LIVE sibling must never be adopted or reaped, even if it's the "newest"."""
    ldest = os.path.join(str(tmp_path), "models", "transformers", "own",
                         "live-contended-model")
    liveproc = subprocess.Popen(["sleep", "20"])
    try:
        live_sibling = ldest + f".tmp-{liveproc.pid}"
        wfile(os.path.join(live_sibling, "shard1.bin"), 2)
        fresh2 = dm._staging_dir(ldest)
        result3 = dm._adopt_or_reap_staging(ldest, fresh2, grace_seconds=0)
        assert not os.path.exists(fresh2) and os.path.exists(live_sibling)
        assert result3 == fresh2
    finally:
        liveproc.terminate()
        liveproc.wait()
        shutil.rmtree(ldest + f".tmp-{liveproc.pid}", ignore_errors=True)


# ===========================================================================
# DISCOVER-WALK HOOK WIRING — reaper actually fires from discover_model(s)
# ===========================================================================
def test_discover_walk_hook_calls_reaper(monkeypatch):
    calls = []
    monkeypatch.setattr(dm, "reap_orphaned_staging",
                        lambda *a, **k: (calls.append((a, k)) or []))
    gm._reap_orphaned_staging_quiet()
    assert len(calls) == 1
