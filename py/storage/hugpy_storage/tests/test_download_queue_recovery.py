"""k119 — why "Add models" downloads constantly did not download.

THE OUTAGE, precisely. ``hugpy-downloader-dev`` was `active (running)` and
logged ``polling (active=0)`` every five minutes for eleven days, and claimed
nothing the entire time. It was not a queue-name mismatch: producer and consumer
agree on ``kind="download"`` (one constant, engine.DOWNLOAD_KIND). The daemon had
gone DEAF.

  1. ``SqliteMirror._note_failure`` set ``_disabled = True`` after MAX_FAILURES
     consecutive faults and never cleared it — "degrade to per-process until
     restart". The daemon opens ~40 SQLite connections a minute against a
     virtiofs mount that demonstrably throws transient `disk I/O error` /
     `database is locked` / `unable to open database file`. On 2026-08-10
     08:54:51-57 five consecutive blips latched the mirror off permanently.
     ``Downloader.run()`` checks the mirror ONCE, at startup, so nothing ever
     noticed. => the breaker must HALF-OPEN and heal itself.

  2. ``expire_pending_orphans`` retired every ``pending`` row with no ``worker``
     after 30 minutes, labelled "never dispatched — model unresolvable or no
     capable worker". A download NEVER has a worker: it is dispatched by a
     DAEMON taking a claim. So every queued download was deleted half an hour
     later under a message that pointed the operator at the wrong subsystem. All
     65 download rows in the dev comms DB read exactly that string.
     => claim-queue kinds age on their own long clock, with an honest message.

  3. ``claim_next`` answers None for BOTH "nothing queued" and "I cannot see the
     queue", which is what made (1) invisible. => a depth probe that can say
     "I don't know", and a heartbeat line that carries it.

Everything here runs in-process against a temp DB: two JobStore instances stand
in for the two processes, exactly as test_downloader_queue.py does.
"""
import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_control import shared as sh
from hugpy_control.jobs import JobStore
from hugpy_control.shared import SqliteMirror
from hugpy_storage.downloader.engine import DOWNLOAD_KIND, classify_download_error

KIND = DOWNLOAD_KIND


@pytest.fixture()
def mirror(tmp_path):
    return SqliteMirror(str(tmp_path / "comms.db"))


def _payload(key="tiny"):
    return {"model": {"hub_id": f"acme/{key}", "name": key}}


def _break(m, n=sh.MAX_FAILURES, op="claim_next"):
    """Feed the breaker n synthetic faults, as the mount does."""
    for _ in range(n):
        m._note_failure(op, sqlite3.OperationalError("disk I/O error"))


# ══ 1. the producer/consumer CONTRACT — enqueue shape == claim filter ══════
def test_enqueue_shape_matches_the_daemons_claim_filter(tmp_path):
    """The one test that would have caught a real kind/state mismatch. The API's
    enqueue must produce a row that the daemon's EXACT claim predicate selects:
    status 'pending', the download kind, unclaimed, not cancelled."""
    db = str(tmp_path / "comms.db")
    api = JobStore(mirror=SqliteMirror(db))
    daemon = JobStore(mirror=SqliteMirror(db))

    from hugpy_storage.downloader import queue as dlq
    dlq.job_store = api            # the API process enqueues
    try:
        job = dlq.enqueue_download("tiny", {"hub_id": "acme/tiny", "name": "t"})
    finally:
        dlq.job_store = __import__(
            "hugpy_control.jobs", fromlist=["job_store"]).job_store

    row = sqlite3.connect(db).execute(
        "SELECT kind, status, claimed_by, cancel_requested FROM jobs WHERE id=?",
        (job.id,)).fetchone()
    assert row[0] == KIND, "enqueue must use the kind the daemon claims"
    assert row[1] == "pending", "the claim filter only takes status='pending'"
    assert not row[2], "a fresh enqueue must be UNCLAIMED"
    assert row[3] == 0

    claimed = daemon.mirror.claim_next((KIND,), "daemon:1")
    assert claimed is not None and claimed["id"] == job.id
    assert (claimed.get("payload") or {}).get("model"), \
        "the daemon needs the model spec — it is a different process"


def test_the_claim_filter_and_the_depth_probe_agree(mirror):
    """claimable_count must count exactly the rows claim_next would take."""
    store = JobStore(mirror=mirror)
    for i in range(3):
        store.enqueue(f"m{i}", kind=KIND, payload=_payload(f"m{i}"))
    store.enqueue("other", kind="chat")
    assert mirror.claimable_count((KIND,)) == 3
    mirror.claim_next((KIND,), "daemon:1")
    assert mirror.claimable_count((KIND,)) == 2, \
        "a claimed row is no longer claimable and must not be counted"


def test_depth_says_I_DONT_KNOW_when_the_queue_is_unreachable(mirror):
    """-1, not 0. 'nothing queued' and 'I cannot see the queue' answering the
    same value is what let the outage read as an idle daemon for 11 days."""
    store = JobStore(mirror=mirror)
    store.enqueue("m", kind=KIND, payload=_payload())
    assert mirror.claimable_count((KIND,)) == 1
    _break(mirror)
    assert mirror.claimable_count((KIND,)) == -1


# ══ 2. the breaker HEALS — the root cause ═════════════════════════════════
def test_a_burst_of_faults_quarantines_but_does_not_kill_the_mirror(mirror):
    store = JobStore(mirror=mirror)
    store.enqueue("m", kind=KIND, payload=_payload())
    _break(mirror)
    assert mirror.health()["ok"] is False
    assert mirror.claim_next((KIND,), "d:1") is None, "quarantined = no claims"
    assert mirror.health()["retry_in"] > 0, "a quarantine must EXPIRE"


def test_the_quarantine_expires_and_the_daemon_claims_again(mirror, monkeypatch):
    """THE regression test. Before k119 this claim stayed None forever and the
    operator's download sat queued until the sweep deleted it."""
    monkeypatch.setattr(sh, "MIRROR_COOLDOWN_SECONDS", 0.05)
    m = SqliteMirror(mirror.path)
    m._cooldown = 0.05
    store = JobStore(mirror=m)
    job = store.enqueue("m", kind=KIND, payload=_payload())

    _break(m)
    assert m.claim_next((KIND,), "d:1") is None

    time.sleep(0.12)                      # cooldown elapses -> half-open probe
    claimed = m.claim_next((KIND,), "d:1")
    assert claimed is not None and claimed["id"] == job.id, \
        "the mirror must heal itself; a transient blip must not be permanent"
    assert m.health()["ok"] is True
    assert m.health()["recoveries"] == 1


def test_a_failed_probe_reopens_the_breaker_immediately(mirror, monkeypatch):
    """A probe that fails must not spend another MAX_FAILURES calls proving what
    it just showed, and its cooldown must back off."""
    monkeypatch.setattr(sh, "MIRROR_COOLDOWN_SECONDS", 0.05)
    m = SqliteMirror(mirror.path)
    m._cooldown = 0.05
    JobStore(mirror=m).enqueue("m", kind=KIND, payload=_payload())
    _break(m)
    first = m._cooldown
    time.sleep(0.12)
    m._ensure()                            # half-open
    assert m._probing is True
    _break(m, n=1)                         # the probe fails
    assert m.health()["ok"] is False
    assert m._cooldown > first, "a repeat outage must back off"


def test_the_cooldown_is_capped(mirror, monkeypatch):
    monkeypatch.setattr(sh, "MIRROR_COOLDOWN_MAX_SECONDS", 0.4)
    m = SqliteMirror(mirror.path)
    m._cooldown = 0.1
    for _ in range(20):
        m._probing = True
        _break(m, n=1)
    assert m._cooldown <= 0.4, "backoff must not grow without bound"


def test_recovery_is_logged_so_an_outage_is_never_silent(mirror, monkeypatch,
                                                         caplog):
    monkeypatch.setattr(sh, "MIRROR_COOLDOWN_SECONDS", 0.05)
    m = SqliteMirror(mirror.path)
    m._cooldown = 0.05
    JobStore(mirror=m).enqueue("m", kind=KIND, payload=_payload())
    with caplog.at_level("WARNING", logger="hugpy_control.shared"):
        _break(m)
        time.sleep(0.12)
        m.claim_next((KIND,), "d:1")
    text = caplog.text
    assert "QUARANTINED" in text, "the outage must announce itself"
    assert "RECOVERED" in text, \
        "and so must the recovery — one ERROR then 11 days of silence is the bug"


# ══ 3. a queued download is NOT an undispatched chat job ══════════════════
def test_a_queued_download_is_not_expired_as_no_capable_worker(mirror):
    """All 65 download rows on the dev box read 'no capable worker'. A download
    has no worker BY DESIGN — it is claimed by a daemon."""
    store = JobStore(mirror=mirror)
    job = store.enqueue("m", kind=KIND, payload=_payload())
    with sqlite3.connect(mirror.path) as c:      # age it past the 30-min cutoff
        c.execute("UPDATE jobs SET progressed_at=? WHERE id=?",
                  (time.time() - 3600, job.id))
    assert mirror.expire_pending_orphans() == [], \
        "a download queued for an hour is WAITING, not undispatched"


def test_a_chat_job_is_still_expired_on_the_short_clock(mirror):
    """The fix must not disarm the sweep it was carved out of."""
    store = JobStore(mirror=mirror)
    job = store.create("m", kind="chat")
    with sqlite3.connect(mirror.path) as c:
        c.execute("UPDATE jobs SET status='pending', progressed_at=? WHERE id=?",
                  (time.time() - 3600, job.id))
    assert mirror.expire_pending_orphans() == [job.id]


def test_an_ancient_unclaimed_download_expires_with_an_HONEST_message(mirror):
    store = JobStore(mirror=mirror)
    job = store.enqueue("m", kind=KIND, payload=_payload())
    with sqlite3.connect(mirror.path) as c:
        c.execute("UPDATE jobs SET progressed_at=? WHERE id=?",
                  (time.time() - sh._claim_queue_expiry_seconds() - 60, job.id))
    assert mirror.expire_pending_orphans() == [job.id]
    msg = store.get_dict(job.id)["message"]
    assert "downloader" in msg.lower(), msg
    assert "capable worker" not in msg, \
        "naming the wrong subsystem sent the operator hunting workers for weeks"


def test_a_claimed_download_is_never_swept(mirror):
    """A daemon holds it; the sweep must not race the transfer's first write."""
    store = JobStore(mirror=mirror)
    job = store.enqueue("m", kind=KIND, payload=_payload())
    mirror.claim_next((KIND,), "daemon:1")
    with sqlite3.connect(mirror.path) as c:
        c.execute("UPDATE jobs SET progressed_at=? WHERE id=?",
                  (time.time() - sh._claim_queue_expiry_seconds() - 60, job.id))
    assert mirror.expire_pending_orphans() == []


def test_the_local_store_sweep_agrees_with_the_mirror_sweep():
    """jobs.py holds the same policy for rows this process owns — the two
    implementations of one rule must not drift."""
    # create(), not enqueue(): enqueue deliberately keeps NO local row (a
    # leftover pending row in the API masks the daemon's live progress). These
    # are the rows a process OWNS — the ones the local sweep is about.
    store = JobStore()
    dl = store.create("m", kind=KIND)
    chat = store.create("c", kind="chat")
    for j in (dl, chat):
        j.status = "pending"
        j.progressed_at = time.time() - 3600
    expired = store.expire_pending_orphans()
    assert chat.id in expired
    assert dl.id not in expired, "same carve-out, both sides of the boundary"


# ══ 4. gated repos fail FAST and TYPED, never silently ════════════════════
@pytest.mark.parametrize("detail,reason", [
    ("GatedRepoError: 401 Client Error. Cannot access gated repo", "gated_repo"),
    ("HfHubHTTPError: You must be authenticated to access this", "gated_repo"),
    ("HTTPError: 401 Client Error: Unauthorized for url", "auth"),
    ("RepositoryNotFoundError: 404 Client Error", "not_found"),
    ("OSError: [Errno 28] No space left on device", "no_space"),
])
def test_terminal_failures_are_typed(detail, reason):
    got = classify_download_error(detail)
    assert got is not None, f"{detail!r} must be classified, not retried 4x"
    assert got[0] == reason


def test_a_gated_repo_tells_the_operator_what_to_DO():
    _, msg = classify_download_error("GatedRepoError: is a gated repo")
    low = msg.lower()
    assert "token" in low and ("licen" in low or "accept" in low), msg


@pytest.mark.parametrize("detail", [
    "ConnectionError: Connection reset by peer",
    "stalled: no new data for 180s",
    "worker exited with code -9",
    "",
])
def test_retryable_failures_are_left_to_the_resume_loop(detail):
    assert classify_download_error(detail) is None, \
        "a dropped connection must still be resumed, not failed on sight"


# ══ 5. "already present" must mean the file you ASKED for ════════════════
# The second half of the operator's complaint, visible only once the queue was
# fixed: 26 quants of unsloth/Qwen3.8-27B-GGUF were added from the Add-models
# tab, 3 files landed, and all 26 jobs read `completed`. `model_looks_downloaded`
# asks "does this dir hold a usable gguf" — right for SERVING (get_gguf_file
# deliberately ELECTS a quant when the designated one is absent), wrong for a
# DOWNLOAD, where it silently answered "yes, you already have it".
def _gguf(tmp_path, *names):
    d = tmp_path / "repo"
    d.mkdir(exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"\0" * 2_000_000)      # past the 1MB sanity floor
    return str(d)


def test_a_requested_quant_that_is_absent_is_NOT_already_present(tmp_path):
    from hugpy_storage.download_models import requested_file_missing
    d = _gguf(tmp_path, "M-UD-IQ1_S.gguf", "M-UD-Q2_K_XL.gguf")
    model = {"framework": "gguf", "filename": "M-UD-Q6_K.gguf"}
    assert requested_file_missing(model, d) is True, \
        "electing a DIFFERENT quant is a serving rule, not a download rule"


def test_a_requested_quant_that_is_present_is_not_refetched(tmp_path):
    from hugpy_storage.download_models import requested_file_missing
    d = _gguf(tmp_path, "M-UD-IQ1_S.gguf", "M-UD-Q6_K.gguf")
    assert requested_file_missing(
        {"framework": "gguf", "filename": "M-UD-Q6_K.gguf"}, d) is False


def test_a_quant_in_a_shard_subdir_counts_as_present(tmp_path):
    from hugpy_storage.download_models import requested_file_missing
    d = tmp_path / "repo" / "BF16"
    d.mkdir(parents=True)
    (d / "M-BF16-00001-of-00002.gguf").write_bytes(b"\0" * 2_000_000)
    assert requested_file_missing(
        {"framework": "gguf", "filename": "M-BF16-00001-of-00002.gguf"},
        str(tmp_path / "repo")) is False, "split ggufs nest; the walk must recurse"


@pytest.mark.parametrize("model", [
    {"framework": "transformers", "filename": None},
    {"framework": "gguf", "filename": None},            # snapshot/pattern pull
    {"framework": "gguf", "include": ["*Q4*"]},         # no single file to check
    {"framework": "diffusers"},
])
def test_every_other_download_shape_keeps_the_old_rule(tmp_path, model):
    """Deliberately narrow: only an EXPLICIT gguf filename changes behaviour."""
    from hugpy_storage.download_models import requested_file_missing
    assert requested_file_missing(model, _gguf(tmp_path, "x.gguf")) is False


def test_progress_measures_THIS_transfer_not_the_directory(tmp_path, monkeypatch):
    """Fetching one 6.3GB quant into a repo dir already holding 26GB rendered as
    "99.9% — 26.1GB/6.3GB" (observed live, 2026-08-21 02:39). _progress_bytes
    measures the whole dir; the baseline is what makes the bar honest."""
    from hugpy_storage.downloader import engine as eng

    dest = str(tmp_path / "repo")
    _gguf(tmp_path, "already-here.gguf")            # 2MB of OTHER quants
    baseline = eng._dir_bytes(dest)
    assert baseline > 0

    seen = {}
    monkeypatch.setattr(eng.job_store, "update",
                        lambda jid, **kw: seen.update(kw) or None)
    monkeypatch.setattr(eng, "_is_cancelled", lambda _jid: False)
    # dir grows by 1MB of OUR file while 2MB of someone else's sits there
    monkeypatch.setattr(eng, "_progress_bytes",
                        lambda _d: baseline + 1_000_000)

    class _Proc:
        def __init__(self): self.n = 0
        def is_alive(self): self.n += 1; return self.n <= 1

    eng._watch(_Proc(), "j1", dest, total_bytes=2_000_000, baseline=baseline)
    assert seen["downloaded_bytes"] == 1_000_000, \
        "the bar must count OUR bytes, not the directory's"
    assert 0.49 < seen["progress"] < 0.51, seen["progress"]


# ══ 6. the stall killer must not shoot the daemon, or a booting child ═════
def test_terminate_tree_never_signals_its_own_process_group(monkeypatch):
    """2026-08-21 02:47:15: the stall killer fired while a spawn child was still
    importing, so getpgid(child) was still the DAEMON's group — killpg SIGTERMed
    hugpy-downloader-dev itself ("hugpy downloader stopping" in the same second).
    """
    from hugpy_platform import procutil

    killed = {}
    monkeypatch.setattr(procutil.os, "getpgid", lambda _p: 4242)
    monkeypatch.setattr(procutil.os, "getpgrp", lambda: 4242)   # SAME group
    monkeypatch.setattr(procutil.os, "killpg",
                        lambda *a: killed.setdefault("pg", a))
    monkeypatch.setattr(procutil.os, "kill", lambda *a: killed.setdefault("pid", a))

    class _P: pid = 99
    procutil.terminate_tree(_P())
    assert "pg" not in killed, "killpg on our own group kills the caller"
    assert killed.get("pid") == (99, __import__("signal").SIGTERM)


def test_terminate_tree_still_kills_a_real_child_group(monkeypatch):
    from hugpy_platform import procutil
    killed = {}
    monkeypatch.setattr(procutil.os, "getpgid", lambda _p: 777)
    monkeypatch.setattr(procutil.os, "getpgrp", lambda: 4242)   # different
    monkeypatch.setattr(procutil.os, "killpg",
                        lambda *a: killed.setdefault("pg", a))

    class _P: pid = 99
    procutil.terminate_tree(_P())
    assert killed["pg"][0] == 777, "a child that DID setpgrp is still group-killed"


def test_a_booting_child_gets_a_startup_grace(tmp_path, monkeypatch):
    """No bytes yet != stalled. A spawn child re-imports the package (registry
    walk included) before it opens a socket; that boot has been measured past
    STALL_SECONDS, and the killer was ending transfers that had not started."""
    from hugpy_storage.downloader import engine as eng
    assert eng.STARTUP_GRACE_SECONDS > eng.STALL_SECONDS

    monkeypatch.setattr(eng.job_store, "update", lambda *a, **k: None)
    monkeypatch.setattr(eng, "_is_cancelled", lambda _j: False)
    monkeypatch.setattr(eng, "_progress_bytes", lambda _d: 0)   # not a byte yet
    monkeypatch.setattr(eng, "STALL_SECONDS", 0)                # instantly "stale"
    monkeypatch.setattr(eng, "STARTUP_GRACE_SECONDS", 10_000)
    killed = []
    monkeypatch.setattr(eng, "terminate_tree", lambda p: killed.append(p))

    class _Proc:
        def __init__(self): self.n = 0
        def is_alive(self): self.n += 1; return self.n <= 2

    assert eng._watch(_Proc(), "j", str(tmp_path), 1000) is False
    assert killed == [], "a child that has not produced a byte is BOOTING"


def test_a_transfer_that_stops_mid_flight_is_still_killed(tmp_path, monkeypatch):
    """The grace must not disarm the stall killer once bytes ARE flowing."""
    from hugpy_storage.downloader import engine as eng
    monkeypatch.setattr(eng.job_store, "update", lambda *a, **k: None)
    monkeypatch.setattr(eng, "_is_cancelled", lambda _j: False)
    monkeypatch.setattr(eng, "_progress_bytes", lambda _d: 500)  # bytes landed
    monkeypatch.setattr(eng, "STALL_SECONDS", 0)
    killed = []
    monkeypatch.setattr(eng, "terminate_tree", lambda p: killed.append(p))

    class _Proc:
        def is_alive(self): return True

    assert eng._watch(_Proc(), "j", str(tmp_path), 1000) is True
    assert len(killed) == 1


def test_the_typed_reason_reaches_the_wire(mirror):
    """error_reason must survive to_dict/to_legacy or the console cannot branch
    on it (and would print a raw 401 traceback at the operator)."""
    store = JobStore(mirror=mirror)
    job = store.create("m", kind=KIND)          # the DAEMON owns a running row
    store.update(job.id, status="failed", error="GatedRepoError: gated",
                 error_reason="gated_repo")
    assert store.get(job.id).to_legacy_dict()["error_reason"] == "gated_repo"
    assert "error_reason" not in store.create("c", kind="chat").to_dict(), \
        "omit-when-unset: no other row's wire shape changes"
