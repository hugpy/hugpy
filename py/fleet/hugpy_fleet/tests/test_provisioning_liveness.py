"""Provisioning must AGE ON PROGRESS, not on presence-in-a-list.

The defect (operator + keeper, 2026-07-16), verified live before this fix:

    computron  last_seen=4.9s     status=online   provisioning=0
    op         last_seen=7750.0s  status=offline  provisioning=4   <- STOPPED 2h+
    ae         last_seen=15.0s    status=online   provisioning=63  <- 0 bytes moving

Nothing ever cleared a `provisioning` entry whose owner died or whose pull
ended. The worker adds the key at KICK time and removes it in a `finally`; if
the process dies mid-pull that `finally` never runs, so central reported the
model as "⏳ pulling" FOREVER, and the console promised a transfer that was not
happening ("defaults are promises").

This is the SAME defect class as the orphan-job bug (state that ages on writes
which stop arriving when the writer dies). The recorded lesson, applied here:
age on PROGRESS, not on any-write / not on presence-in-a-list.

⚠ This codebase has TWICE shipped tests that asserted the bug, so these assert
BEHAVIOR through the PUBLIC read path (`_public_view` / `storage_proposal`) —
the same shape the console actually renders — not internal helpers.

Two things are deliberately pinned as SEPARATE properties, because conflating
them is the subtle bug:
  * REPORTING  — a dead pull must not read as in-flight.
  * PROTECTION — a dead pull must not hold phantom eviction protection, while a
    LIVE pull must keep it (deleting under a live write is the one truly
    destructive mistake available here).

Run: venv/bin/python -m pytest tests/test_provisioning_liveness.py -v
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.central import workers as W

GIB = 1 << 30


def _worker(*, last_seen_ago, provisioning, progress=None, models=None):
    """A worker record shaped like a real heartbeat row."""
    now = time.time()
    return {
        "id": "w", "name": "w", "url": "http://w",
        "last_seen": now - last_seen_ago,
        "provisioning": list(provisioning),
        "provision_progress": dict(progress or {}),
        "loaded_models": [], "loading": [],
        "config": {}, "limits": {}, "grants": {},
        "disk": {"free_bytes": 10 * GIB, "total_bytes": 500 * GIB},
        "model_last_picked": {},
        "storage": {
            "cache_used_bytes": 200 * GIB,
            "disk_free": 10 * GIB,          # < 50 GiB reserve -> over budget
            "models": models if models is not None else [],
        },
    }


def _row(key, gib=40, **flags):
    row = {"model_key": key, "bytes": int(gib * GIB), "protected": False,
           "why": "", "pinned": False, "loaded": False, "loading": False,
           "provisioning": False, "assigned": False}
    row.update(flags)
    return row


def _prot(worker, key):
    """The public storage row for `key` -> (protected, why)."""
    view = W.storage_proposal(worker)
    row = {m["model_key"]: m for m in view["models"]}[key]
    return row["protected"], row["why"]


# ── 1. the op case: offline worker -> nothing it claimed is in flight ───────
def test_offline_worker_reports_no_provisioning():
    """op: STOPPED for 2h+, central still listed 4 models as provisioning."""
    w = _worker(last_seen_ago=7750.0,
                provisioning=["a", "b", "c", "d"],
                # a FROZEN progress entry — the pull died at 7% and the last
                # snapshot replays verbatim forever. frac>0 must NOT save it.
                progress={"a": {"done_bytes": 1818660754, "total_bytes": 25185315616,
                                "frac": 0.0722, "progressed_at": time.time() - 7750.0}})
    assert W._public_view(w)["status"] == "offline"
    assert W._public_view(w)["provisioning"] == []


def test_a_frozen_frac_is_not_evidence_of_life():
    """The trap: op's dead pull sits at frac=0.0722 with 1.8GB done. A truthy
    frac proves bytes moved ONCE, never that they are moving NOW."""
    w = _worker(last_seen_ago=7750.0, provisioning=["a"],
                progress={"a": {"done_bytes": 1818660754, "total_bytes": 25185315616,
                                "frac": 0.0722}})
    assert W._live_provisioning(w) == set()


# ── 2. the ae case: online, entries, but ZERO bytes moving ─────────────────
def test_online_worker_with_no_progress_entries_is_not_in_flight():
    """ae: online 15s ago, 63 provisioning entries, 0 progress entries.

    WORKER_PROVISION_CONCURRENCY defaults to 1, so these are QUEUED behind the
    semaphore, not transferring. Queued != pulling."""
    w = _worker(last_seen_ago=15.0,
                provisioning=[f"m{i}" for i in range(63)], progress={})
    assert W._public_view(w)["status"] == "online"
    assert W._public_view(w)["provisioning"] == []


def test_online_worker_with_stalled_progress_is_not_in_flight():
    """Online agent, but THIS model's bytes stopped advancing long ago."""
    w = _worker(last_seen_ago=15.0, provisioning=["wedged"],
                progress={"wedged": {"done_bytes": 5 * GIB, "total_bytes": 50 * GIB,
                                     "frac": 0.1,
                                     "progressed_at": time.time() - 3600}})
    assert W._live_provisioning(w) == set()


# ── 3. the REAL case must survive (the whole point of the guard) ───────────
def test_online_worker_with_forward_progress_still_reports_pulling():
    w = _worker(last_seen_ago=5.0, provisioning=["real"],
                progress={"real": {"done_bytes": 9 * GIB, "total_bytes": 50 * GIB,
                                   "frac": 0.18, "progressed_at": time.time() - 2}})
    assert W._public_view(w)["provisioning"] == ["real"]


def test_a_just_started_pull_is_not_born_stale():
    """A pull that just kicked has moved no bytes yet; it must not be declared
    dead on arrival."""
    w = _worker(last_seen_ago=5.0, provisioning=["fresh"],
                progress={"fresh": {"done_bytes": 0, "total_bytes": 50 * GIB,
                                    "frac": 0.0, "progressed_at": time.time()}})
    assert W._live_provisioning(w) == {"fresh"}


@pytest.mark.parametrize("clock", [None, "garbage", {}])
def test_fail_safe_toward_live_on_a_missing_or_garbage_clock(clock):
    """Fail-SAFE: a false 'live' costs a stale pill; a false 'dead' could
    unprotect a real in-flight write. Never unprotect on a bad clock."""
    entry = {"done_bytes": 1 * GIB, "total_bytes": 50 * GIB, "frac": 0.02}
    if clock is not None:
        entry["progressed_at"] = clock
    w = _worker(last_seen_ago=5.0, provisioning=["x"], progress={"x": entry})
    assert W._live_provisioning(w) == {"x"}


# ── 4./5. the EVICTION GUARD — the subtle part ─────────────────────────────
def test_a_live_pull_keeps_its_eviction_protection():
    """Do NOT strip protection from a model that IS genuinely being pulled."""
    w = _worker(last_seen_ago=5.0, provisioning=["real"],
                progress={"real": {"done_bytes": 9 * GIB, "total_bytes": 50 * GIB,
                                   "frac": 0.18, "progressed_at": time.time() - 2}},
                models=[_row("real")])
    assert _prot(w, "real") == (True, "provisioning")
    assert W.storage_proposal(w)["proposed_evictions"] == []


def test_a_stale_entry_does_not_grant_phantom_eviction_protection():
    """The converse bug: a dead entry used to make a real, cold, reclaimable
    file un-evictable FOREVER, silently shrinking the reclaimable pool."""
    w = _worker(last_seen_ago=7750.0, provisioning=["ghost"],
                models=[_row("ghost")])
    protected, _ = _prot(w, "ghost")
    assert protected is False
    assert [e["model_key"] for e in W.storage_proposal(w)["proposed_evictions"]] == ["ghost"]


def test_worker_row_flag_alone_cannot_resurrect_protection():
    """The worker's own per-row provisioning flag rides the SAME dead snapshot
    as the stale list, so central must not honour it either."""
    w = _worker(last_seen_ago=7750.0, provisioning=[],
                models=[_row("ghost", provisioning=True)])
    assert _prot(w, "ghost")[0] is False


def test_other_guards_are_untouched():
    """Only PROVISIONING is liveness-gated. loaded/loading keep protecting;
    pinned/assigned are CANDIDATES (operator 2026-07-17: pin = allocation
    survives restarts, allocation = routing — "neither of those should have
    any bearing on the pull or eviction")."""
    w = _worker(last_seen_ago=5.0, provisioning=[],
                models=[_row("l", loaded=True), _row("h", loading=True),
                        _row("p", pinned=True), _row("a", assigned=True)])
    assert _prot(w, "l")[0] and _prot(w, "h")[0]
    assert not _prot(w, "p")[0] and not _prot(w, "a")[0]


# ── the central progress CLOCK (heartbeat) ────────────────────────────────
def test_heartbeat_only_bumps_the_clock_when_bytes_actually_advance(tmp_path, monkeypatch):
    """The orphan-job lesson in one test: the worker re-sends its whole progress
    map every heartbeat, so ARRIVAL of the field must not count as progress."""
    # k3 isolation: WorkerStore(path=tmp_path/...) isolates the worker rows,
    # but register(worker_id=...) ALSO reads the assignment-memory sidecar,
    # which independently derives its directory from settings.manifest_path
    # (a frozen, .env-file-resolved singleton — see
    # tests/worker_store_isolation.py). Redirect it into tmp_path too so this
    # test never touches the live /mnt/llm_storage/projects/ registry.
    monkeypatch.setattr(W.settings, "manifest_path", str(tmp_path / "model_manifest.json"))
    reg = W.WorkerStore(path=str(tmp_path / "workers.json"))
    reg.register(worker_id="w", name="w", url="http://w")

    reg.heartbeat("w", provisioning=["m"],
                  provision_progress={"m": {"done_bytes": 1 * GIB,
                                            "total_bytes": 50 * GIB}})
    t1 = reg.get("w")["provision_progress"]["m"]["progressed_at"]

    # Same bytes re-announced -> NOT progress; the clock must not move.
    time.sleep(0.02)
    reg.heartbeat("w", provisioning=["m"],
                  provision_progress={"m": {"done_bytes": 1 * GIB,
                                            "total_bytes": 50 * GIB}})
    t2 = reg.get("w")["provision_progress"]["m"]["progressed_at"]
    assert t2 == t1, "a re-announced identical snapshot must not count as progress"

    # Bytes advanced -> the clock moves.
    time.sleep(0.02)
    reg.heartbeat("w", provisioning=["m"],
                  provision_progress={"m": {"done_bytes": 2 * GIB,
                                            "total_bytes": 50 * GIB}})
    t3 = reg.get("w")["provision_progress"]["m"]["progressed_at"]
    assert t3 > t1, "advancing done_bytes must bump the clock"


def test_a_wedged_pull_turns_stale_without_any_writer_noticing(tmp_path,
                                                               monkeypatch):
    """End-to-end of the whole fix: a live agent keeps heart-beating (so it
    stays ONLINE) but its pull stops advancing. It must age out of in-flight on
    its own, with no daemon and no 'finished/failed' message ever arriving."""
    monkeypatch.setenv("HUGPY_PROVISION_STALL_SECONDS", "1")
    # k3 isolation: see the note in the previous test — redirect the
    # assignment-memory sidecar too, not just WorkerStore's own path.
    monkeypatch.setattr(W.settings, "manifest_path", str(tmp_path / "model_manifest.json"))
    reg = W.WorkerStore(path=str(tmp_path / "workers.json"))
    reg.register(worker_id="w", name="w", url="http://w")

    snap = {"m": {"done_bytes": 1 * GIB, "total_bytes": 50 * GIB}}
    reg.heartbeat("w", provisioning=["m"], provision_progress=snap)
    assert reg.get("w")["provisioning"] == ["m"], "a fresh pull reads as in-flight"

    time.sleep(1.1)
    # Still alive, still re-announcing the SAME frozen snapshot (the op case).
    row = reg.heartbeat("w", provisioning=["m"], provision_progress=snap)
    assert row["status"] == "online", "the agent is still alive"
    assert row["provisioning"] == [], "but its wedged pull is no longer in-flight"
