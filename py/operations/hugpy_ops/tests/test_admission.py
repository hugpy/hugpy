"""Post-download admission gate (hugpy_ops.admission): static fail -> held
without a benchmark; static_ok -> benchmark (queued behind a running one) ->
admitted with its grade; a load failure during the benchmark -> held with the
loader stderr; the record lands on hugpy.json; the seed's decisions; and the
whole job makes zero network attempts against a fake central."""
from __future__ import annotations

import json
import socket

import pytest

from hugpy_ops import admission as A
from hugpy_ops import model_audit as ma
from hugpy_storage import admission as sadm
from hugpy_storage import hugpy_marker as hm

KEY = "M-GGUF"


class FakeCentral:
    base = "http://central.invalid"

    def __init__(self, dest, *, busy_first=0, results=None, failures=None, task="text-generation",
                 key=KEY, run_status="complete", error=None, archive=None, running_forever=False):
        self.dest = dest
        self.task, self.key, self.run_status, self.error = task, key, run_status, error
        self.archive, self.running_forever = archive, running_forever
        self.cancelled = 0
        self.busy_left = busy_first
        self.results = results if results is not None else []
        self.failures = failures or []
        self.started = 0
        self.integrity = []
        self.polls = 0

    def catalog(self):
        return [{"model_key": self.key, "framework": "gguf", "destination": self.dest,
                 "hub_id": f"owner/{KEY}", "primary_task": self.task, "workers": []}]

    def workers(self):
        return [{"name": "aeb", "id": "w1", "status": "online", "gpu_total_bytes_known": 24 << 30}]

    def grade_rows(self):
        return {}

    def record_integrity(self, rep, ctx):
        self.integrity.append(rep.verdict)
        return {"ok": True}

    def benchmark_start(self, key, tokens=128):
        if self.busy_left:
            self.busy_left -= 1
            return 409, {"run_id": "other", "status": "running"}
        self.started += 1
        return 202, {"run_id": "r1", "status": "running"}

    def benchmark_status(self):
        self.polls += 1
        if self.polls < 2 or self.running_forever:
            return {"run_id": "r1", "status": "running", "results": []}
        return {"run_id": "r1", "status": self.run_status, "results": self.results, "error": self.error}

    def benchmark_run(self, run_id):
        return self.archive if self.archive and self.archive.get("run_id") == run_id else None

    def benchmark_cancel(self):
        self.cancelled += 1

    def load_failures(self, key, since):
        return self.failures


def _audit(verdict, why="w"):
    def audit(row, ctx):
        rep = ma.ModelReport(row["model_key"], row["framework"], row["hub_id"],
                             row["primary_task"], row["destination"])
        rep.verdict, rep.why = verdict, why
        rep.log.append(f"fake audit -> {verdict}")
        return rep
    return audit


def _job(dest):
    return {"id": "j1", "model_key": KEY, "directory": dest}


@pytest.fixture
def dest(tmp_path, monkeypatch):
    monkeypatch.setattr("hugpy_storage.model_metadata.model_metadata_store",
                        type("S", (), {"get_repo_info": lambda self, h: None})())
    monkeypatch.setattr("hugpy_storage.model_physical.forget_physical", lambda *a, **k: True)
    monkeypatch.setattr(ma, "load_overrides", lambda: ({}, "/dev/null"))

    def no_net(*a, **k):
        raise AssertionError("network attempted")
    monkeypatch.setattr(socket, "socket", no_net)
    d = tmp_path / KEY
    d.mkdir()
    hm.write_hugpy_marker(str(d), hub_id=f"owner/{KEY}", name=KEY, framework="gguf", source="download")
    return str(d)


def run(dest, central, verdict, run_kw=None, **kw):
    lines = []
    rec = A.run_job(_job(dest), central, log=lines.append, audit=_audit(verdict, **kw),
                    finalize=lambda rep, row, ctx: rep, sleep=lambda s: None, poll=0, **(run_kw or {}))
    return rec, lines


@pytest.fixture
def suites(monkeypatch):
    """Pin the task -> suite map (hugpy_curation.review.suites) for the runner."""
    table = {"text-generation": "hugpy-native-v2", "image-text-to-text": "hugpy-vision-v1",
             "text-to-image": "hugpy-imagegen-v1"}
    monkeypatch.setattr(A, "expected_suite", lambda row: (table.get(row.get("primary_task")), True))
    return table


@pytest.mark.parametrize("verdict", [ma.BROKEN, ma.FAULTY, ma.MISCONFIG, ma.NOT_DOWNLOADED])
def test_static_fail_is_held_with_reason_and_no_benchmark(dest, verdict):
    c = FakeCentral(dest)
    rec, lines = run(dest, c, verdict, why="tensor dims wrong")
    assert rec["status"] == "held" and rec["integrity"] == verdict
    assert rec["reason"] == f"{verdict}: tensor dims wrong"
    assert c.started == 0 and c.integrity == [verdict]


def test_static_ok_benchmarks_and_admits_with_grade(dest):
    c = FakeCentral(dest, busy_first=2, results=[
        {"model": KEY, "worker": "aeb", "quant": "Q4", "status": "complete", "error": "N/A",
         "score": 18, "max": 24},
        {"model": "other", "status": "complete", "error": "N/A", "score": 24, "max": 24}])
    rec, lines = run(dest, c, ma.STATIC_OK)
    assert rec["status"] == "admitted" and rec["grade"] == 75.0
    assert c.started == 1                                   # waited behind the 409s, then ran
    assert any("waiting behind it" in line for line in lines)


def test_load_failure_during_benchmark_is_held_with_loader_stderr(dest):
    fail = {"id": 7, "worker_card": "aeb:0", "action": "load", "outcome": "fail",
            "detail": {"class": "faulty_model",
                       "loader_stderr": "llama_model_load: error\ncheck_tensor_dims: tensor 'token_embd.weight' has wrong shape"}}
    c = FakeCentral(dest, results=[{"model": KEY, "worker": "aeb", "status": "error",
                                    "error": "load failed"}], failures=[fail])
    rec, _ = run(dest, c, ma.STATIC_OK)
    assert rec["status"] == "held"
    assert "check_tensor_dims" in rec["reason"] and "faulty_model" in rec["reason"]


def test_no_grade_is_pending_with_the_runs_own_reason(dest):
    rec, _ = run(dest, FakeCentral(dest, results=[]), ma.STATIC_OK)
    assert rec["status"] == "pending"
    assert "no grade" not in rec["reason"]                  # never a canned sentence
    assert rec["reason"] and rec.get("elapsed_s") is not None


def test_non_text_verdict_admits_only_when_no_suite_covers_the_task(dest, suites):
    c = FakeCentral(dest, task="audio-classification")
    rec, _ = run(dest, c, ma.UNSUPPORTED, why="comfy")
    assert rec["status"] == "admitted" and "no grading suite for task 'audio-classification'" in rec["reason"]
    assert c.started == 0


def test_stale_suite_mismatch_runs_the_tasks_suite(dest, suites):
    """comfy-dreamshaper-8: an old suite_mismatch verdict never admits when a
    suite (hugpy-imagegen-v1) now exists for the task — the benchmark runs."""
    c = FakeCentral(dest, task="text-to-image", results=[
        {"model": KEY, "worker": "central", "status": "complete", "error": "N/A", "score": 3, "max": 4,
         "grade": "3/4", "grade_suite": "hugpy-imagegen-v1"}])
    rec, _ = run(dest, c, ma.SUITE_MISMATCH)
    assert c.started == 1
    assert rec["status"] == "admitted" and rec["grade"] == 75.0 and "hugpy-imagegen-v1" in rec["reason"]


def test_grade_row_counts_even_when_the_run_failed(dest, suites):
    """Qwen2.5-Coder-1.5B: 23/27 recorded but the speed probe errored and the
    run ended failed — the grade is the recorded data; admitted."""
    c = FakeCentral(dest, run_status="failed", error={"headline": "0 of 1 allocation(s) completed"},
                    results=[{"model": KEY, "worker": "aeb", "quant": "Q4", "status": "error",
                              "error": "speed probe: timeout", "score": 23, "max": 27, "grade": "23/27"}])
    rec, _ = run(dest, c, ma.STATIC_OK)
    assert rec["status"] == "admitted" and rec["grade"] == round(100 * 23 / 27, 2)
    assert "run r1 ended failed" in rec["reason"]


def test_archived_run_rows_are_read_and_matched_by_owner_key(dest, suites):
    c = FakeCentral(dest, key="Owner~" + KEY, results=[], archive={
        "run_id": "r1", "status": "complete",
        "results": [{"model": KEY, "worker": "aeb", "status": "complete", "error": "N/A",
                     "score": 1, "max": 2}]})
    rec, _ = run(dest, c, ma.STATIC_OK)
    assert rec["status"] == "admitted" and rec["grade"] == 50.0


def test_run_still_running_at_the_deadline_is_pending_not_held(dest, suites):
    """Qwen2.5-VL-3B: the job must not end while its run is still running."""
    ticks = iter(range(0, 10_000, 5))
    c = FakeCentral(dest, running_forever=True)
    rec, _ = run(dest, c, ma.STATIC_OK, run_kw={"clock": lambda: next(ticks), "benchmark_timeout": 60})
    assert rec["status"] == "pending" and rec["failure_class"] == "timeout"
    assert "still running" in rec["reason"] and c.cancelled == 1


def test_run_level_error_without_a_model_row_is_pending(dest, suites):
    c = FakeCentral(dest, run_status="failed", results=[{"model": "other", "failure_class": "timeout",
                                                         "status": "failed", "reason": "x"}],
                    error={"headline": "0 of 1 allocation(s) completed: 1 timeout",
                           "failure_classes": {"timeout": {"count": 1, "models": ["other"],
                                                           "first_reason": "x", "evidence": None}}})
    rec, _ = run(dest, c, ma.STATIC_OK)
    assert rec["status"] == "pending" and "0 of 1 allocation(s) completed" in rec["reason"]


def test_no_lane_row_is_pending_blocked_on_placement_with_its_evidence(dest, suites):
    ev = {"designated_workers": [], "eligible_workers": ["aeb"], "joined": [], "admission": {}, "hot": False}
    reason = "model is not placed on any worker (no designation)"
    c = FakeCentral(dest, run_status="failed", results=[
        {"model": KEY, "worker": "central", "status": "failed", "failure_class": "no_lane",
         "reason": reason, "evidence": ev}])
    rec, _ = run(dest, c, ma.STATIC_OK)
    assert rec["status"] == "pending" and rec["reason"] == reason
    assert rec["blocked_on"] == "placement" and rec["evidence"] == ev and rec["failure_class"] == "no_lane"


def test_hard_load_failure_row_is_held_with_its_reason(dest, suites):
    c = FakeCentral(dest, run_status="failed", results=[
        {"model": KEY, "worker": "aeb", "status": "failed", "failure_class": "hard_load_failure",
         "reason": "llama_model_load: bad tensor", "evidence": {"worker": "aeb"}}])
    rec, _ = run(dest, c, ma.STATIC_OK)
    assert rec["status"] == "held" and rec["reason"] == "llama_model_load: bad tensor"


def test_process_writes_record_to_marker_and_finishes_job(dest, tmp_path, monkeypatch):
    q = sadm.AdmissionQueue(str(tmp_path / "q.sqlite"))
    job, _ = q.enqueue(KEY, "cap-1", directory=dest)
    q.claim_next("t")
    monkeypatch.setattr(ma, "audit_model", _audit(ma.FAULTY, "bad"))
    monkeypatch.setattr(ma, "finalize", lambda rep, row, ctx: rep)
    rec = A.process(job, queue=q, central=FakeCentral(dest))
    block = sadm.read_admission(dest)
    assert block["status"] == "held" and block["job"] == job["id"] and rec["status"] == "held"
    row = q.get(job["id"])
    assert row["status"] == sadm.Q_DONE and "audit verdict: faulty_model" in row["log"]


def test_seed_decisions_and_dry_run_counts(tmp_path, monkeypatch):
    monkeypatch.setattr("hugpy_storage.model_physical.forget_physical", lambda *a, **k: True)
    dirs = {}
    for k in ("a", "b", "c", "d", "e"):
        d = tmp_path / k
        d.mkdir()
        hm.write_hugpy_marker(str(d), hub_id=f"o/{k}", name=k, framework="gguf", source="backfill")
        dirs[k] = str(d)
    report = {"models": {
        "a": {"verdict": ma.STATIC_OK, "destination": dirs["a"], "primary_task": "text-generation"},
        "b": {"verdict": ma.BROKEN, "why": "short file", "destination": dirs["b"]},
        "c": {"verdict": ma.SUITE_MISMATCH, "destination": dirs["c"]},
        "d": {"verdict": ma.STATIC_OK, "destination": dirs["d"], "primary_task": "text-generation"},
        "e": {"verdict": ma.STATIC_OK, "destination": dirs["e"], "primary_task": "text-generation"},
        "gone": {"verdict": ma.NOT_DOWNLOADED, "destination": str(tmp_path / "nope")}}}
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report))
    grades = A.text_grades({"a": [{"grade": 80, "grade_suite": "hugpy-native-v2"},
                                  {"grade": 100, "grade_suite": "integrity"}],
                            "e": [{"grade": 0, "grade_suite": "hugpy-native-v2"}]})
    out = A.seed(str(path), grades=grades)
    by = {r["model_key"]: r for r in out["rows"]}
    assert by["a"]["status"] == "admitted" and by["a"]["grade"] == 80
    assert by["b"]["status"] == "held" and "short file" in by["b"]["reason"]
    assert by["c"]["status"] == "admitted"
    assert by["d"]["status"] == "pending"
    assert by["e"]["status"] == "held"
    assert by["gone"]["action"].startswith("skip")
    assert out["counts"]["admitted"] == 2 and out["counts"]["held"] == 3 and out["counts"]["pending"] == 1
    assert sadm.read_admission(dirs["a"]) is None             # dry-run wrote nothing
    A.seed(str(path), grades=grades, apply=True)
    assert sadm.read_admission(dirs["a"])["status"] == "admitted"


def test_real_audit_admission_job_makes_no_network_attempt(dest, tmp_path):
    """C: the static audit + the admission job on an installed fixture model,
    with socket.socket raising — zero network attempts (fake central)."""
    from test_model_audit import write_gguf
    write_gguf(str(tmp_path / KEY / "M-Q4.gguf"))
    rec = A.run_job(_job(dest), FakeCentral(dest, results=[
        {"model": KEY, "worker": "aeb", "status": "complete", "error": "N/A", "score": 1, "max": 2}]),
        log=lines_sink(), sleep=lambda s: None, poll=0)
    assert rec["integrity"] == ma.STATIC_OK and rec["status"] == "admitted" and rec["grade"] == 50.0


def lines_sink():
    out = []
    return out.append


# ── restart survival (2026-09-23: a promotion restart of central) ────────────

def test_running_job_is_requeued_with_attempt_on_runner_election(tmp_path):
    q = sadm.AdmissionQueue(str(tmp_path / "q.sqlite"))
    job, _ = q.enqueue(KEY, "2026-09-23T15:00:00Z", directory=str(tmp_path))
    assert q.claim_next("ae:111")["status"] == sadm.Q_RUNNING   # central dies here
    out = A.recover_orphans(q, "ae:222")
    row = q.get(job["id"])
    assert out["requeued"] == [job["id"]] and row["status"] == sadm.Q_QUEUED
    assert row["attempt"] == 1 and row["owner"] is None
    assert "re-queued: central restarted" in row["log"] and "(attempt 1)" in row["log"]
    # bounded: a job interrupted MAX_ATTEMPTS times is failed with the reason, not looped
    for _ in range(sadm.MAX_ATTEMPTS - 1):
        q.claim_next("x"); A.recover_orphans(q, "y")
    q.claim_next("x")
    out = A.recover_orphans(q, "y")
    row = q.get(job["id"])
    assert out["failed"] == [job["id"]] and row["status"] == sadm.Q_FAILED
    assert "not re-queued again" in row["result"]["reason"]


def test_runner_is_re_elected_on_start_and_recovers_before_claiming(tmp_path):
    import fcntl
    import threading

    class Q(sadm.AdmissionQueue):
        claimed = threading.Event()

        def claim_next(self, owner):          # never run a job in this test
            Q.claimed.set()
            return None

    q = Q(str(tmp_path / "q.sqlite"))
    job, _ = q.enqueue(KEY, "2026-09-23T15:00:00Z")
    sadm.AdmissionQueue.claim_next(q, "dead-runner")
    lock_path = str(tmp_path / "admission_runner.lock")
    t = threading.Thread(target=A._runner_loop, args=(q, lock_path, 3600.0), daemon=True)
    t.start()
    assert Q.claimed.wait(10)
    assert q.get(job["id"])["status"] == sadm.Q_QUEUED and q.get(job["id"])["attempt"] == 1
    with open(lock_path, "a") as fh:          # the elected runner holds the host lock
        with pytest.raises(OSError):
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    import os
    state = A.runner_state(q)                 # who runs jobs: host/pid, not a boolean
    assert state["elected"] and state["pid"] == os.getpid() and state["host"] == socket.gethostname()


def test_old_queue_file_gains_the_attempt_column(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.sqlite")
    conn = sqlite3.connect(path)
    conn.executescript(sadm._SCHEMA.replace("    attempt      INTEGER NOT NULL DEFAULT 0,\n", ""))
    conn.execute("INSERT INTO admission_jobs (id, model_key, captured_at, status, created_at) "
                 "VALUES ('j', 'M', 'c', 'running', 1)")
    conn.commit(); conn.close()
    q = sadm.AdmissionQueue(path)
    assert A.recover_orphans(q, "ae:1")["requeued"] == ["j"] and q.get("j")["attempt"] == 1
