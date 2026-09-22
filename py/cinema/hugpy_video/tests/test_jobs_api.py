"""hugpy_video.jobs: an upper layer registers a job + runner and the bus runs
it end to end (enqueue -> claim -> run -> terminal) in a private sqlite file,
with no oracle, server or fleet. This is the ``video_performance`` pattern.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from hugpy_video import jobs, state


@dataclass(frozen=True)
class EchoSpec:
    text: str
    n: int = 1


def _echo_from_dict(d: dict) -> EchoSpec:
    return EchoSpec(text=str(d["text"]), n=int(d.get("n", 1)))


def _run_echo(spec, job_id: str):
    from hugpy_video.intel.result_schema import JobResult
    assert isinstance(spec, EchoSpec), type(spec)
    jobs.set_progress(job_id, {"stage": "echo", "done": spec.n, "total": spec.n})
    return JobResult(job_id, ok=True, outputs=())


@pytest.fixture
def private_bus(tmp_path):
    from hugpy_video.intel import media_bus
    media_bus.DB_PATH = os.path.join(tmp_path, "jobs.db")
    media_bus._initialized = False
    media_bus._bridge = lambda *a, **k: None       # no control-plane job store here
    yield media_bus


def test_register_job_runs_through_the_bus(private_bus):
    spec = jobs.register_job("echo_test", spec_type=EchoSpec, runner_key=("test", "echo"),
                             queue="media", timeout_s=10, from_dict=_echo_from_dict,
                             runner=_run_echo)
    try:
        assert jobs.registered_jobs()["echo_test"] is spec
        job_id = jobs.enqueue("echo_test", EchoSpec("hi", 2))
        assert jobs.get(job_id)["status"] == "queued"
        assert jobs.work_once("t") == job_id
        row = jobs.get(job_id)
        assert row["status"] == "done", row
        assert row["result"]["ok"] is True
        assert jobs.list_jobs(include_terminal=True)[0]["job_id"] == job_id
        assert jobs.work_once("t") is None
    finally:
        jobs.unregister_job("echo_test")
    assert "echo_test" not in jobs.registered_jobs()
    with pytest.raises(KeyError):
        jobs.enqueue("echo_test", EchoSpec("x"))


def test_replace_false_keeps_existing(private_bus):
    a = jobs.register_job("echo_keep", spec_type=EchoSpec, runner_key=("test", "keep"),
                          queue="media", timeout_s=1, from_dict=_echo_from_dict, runner=_run_echo)
    try:
        b = jobs.register_job("echo_keep", spec_type=EchoSpec, runner_key=("test", "keep2"),
                              queue="gpu", timeout_s=2, from_dict=_echo_from_dict,
                              runner=_run_echo, replace=False)
        assert b is a
    finally:
        jobs.unregister_job("echo_keep")


def test_state_roots_are_injectable(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGPY_MEDIA_JOBS_DB", str(tmp_path / "x.db"))
    monkeypatch.setenv("HUGPY_RESERVATIONS_DB", str(tmp_path / "r.db"))
    state.reset_state()
    try:
        roots = state.get_state_roots()
        assert roots.media_jobs_db == str(tmp_path / "x.db")
        assert roots.reservations_db == str(tmp_path / "r.db")
        from hugpy_video.intel.reservation.registry import default_db_path
        assert default_db_path() == str(tmp_path / "r.db")
        over = state.configure_state(media_jobs_db="/tmp/over.db")
        assert over.media_jobs_db == "/tmp/over.db" and state.media_jobs_db_path() == "/tmp/over.db"
        with pytest.raises(TypeError):
            state.configure_state(nope=1)
    finally:
        state.reset_state()
