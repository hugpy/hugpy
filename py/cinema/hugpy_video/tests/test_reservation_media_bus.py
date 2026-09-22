"""p6 — the reservation seam INSIDE media_bus.run_claimed.

Asserts the dispatch wiring, with the reservation acquire/release stubbed:
  * a heavy run ACQUIRES before the runner and RELEASES on the terminal (happy);
  * an honest REFUSAL short-circuits the runner into a gpu_unavailable terminal
    (no runner call, no dangling claim to release);
  * a runner that RAISES still releases the claim (the finally);
  * a runner that returns CANCELLED (abort honored mid-run) still releases.

A throwaway job kind is registered so the REAL run_claimed path is exercised
end-to-end (claim → acquire → runner → terminal → release) without any GPU.

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_reservation_media_bus.py -q
"""
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from hugpy_video.intel import media_bus as MB
from hugpy_video.intel.job_schema import JobSpec
from hugpy_video.intel.result_schema import JobResult, JobError


@dataclass(frozen=True)
class _FakeSpec:
    tag: str = "x"


@pytest.fixture
def bus(tmp_path, monkeypatch):
    """A private media bus DB + a registered throwaway 'fake_heavy' job kind."""
    monkeypatch.setattr(MB, "DB_PATH", str(tmp_path / "media_jobs.db"))
    monkeypatch.setattr(MB, "_initialized", False)
    monkeypatch.setitem(MB.JOB_REGISTRY, "fake_heavy",
                        JobSpec("fake_heavy", _FakeSpec, ("fake", "heavy"), "gpu", 60))
    monkeypatch.setitem(MB.SPEC_DESERIALIZERS, "fake_heavy", lambda d: _FakeSpec(**d))
    yield


def _events_and_stubs(monkeypatch, runner, *, refusal=False):
    events = []

    def _acq(name, spec, job_id):
        events.append(("acquire", name))
        if refusal:
            return None, JobResult(
                job_id=job_id, ok=False,
                error=JobError(code="gpu_unavailable",
                               message="GPU reservation refused", retryable=True))
        return object(), None      # a held handle

    def _rel(job_id):
        events.append(("release", job_id))

    monkeypatch.setattr(MB, "_acquire_reservation", _acq)
    monkeypatch.setattr(MB, "_release_reservation", _rel)
    monkeypatch.setitem(MB.DISPATCH, ("fake", "heavy"), runner)
    return events


def test_happy_path_acquire_before_runner_release_after(bus, monkeypatch):
    def runner(spec, job_id):
        # by the time the runner fires, the reservation must already be held
        assert ("acquire", "fake_heavy") in events
        events.append(("runner", job_id))
        return JobResult(job_id=job_id, ok=True)

    events = _events_and_stubs(monkeypatch, runner)
    job_id = MB.enqueue("fake_heavy", _FakeSpec(tag="a"))
    MB.work_once()
    kinds = [e[0] for e in events]
    assert kinds == ["acquire", "runner", "release"]
    assert MB.get(job_id)["status"] == "done"


def test_refusal_short_circuits_runner_into_gpu_unavailable(bus, monkeypatch):
    def runner(spec, job_id):
        events.append(("runner", job_id))       # must NOT happen
        return JobResult(job_id=job_id, ok=True)

    events = _events_and_stubs(monkeypatch, runner, refusal=True)
    job_id = MB.enqueue("fake_heavy", _FakeSpec(tag="b"))
    MB.work_once()
    assert ("runner", job_id) not in events     # runner never ran
    assert ("acquire", "fake_heavy") in events
    # No handle was returned, so nothing is released (no phantom claim).
    assert not any(e[0] == "release" for e in events)
    view = MB.get(job_id)
    assert view["status"] == "failed"
    assert view["result"]["error"]["code"] == "gpu_unavailable"


def test_runner_raise_still_releases(bus, monkeypatch):
    def runner(spec, job_id):
        events.append(("runner", job_id))
        raise RuntimeError("boom mid-render")

    events = _events_and_stubs(monkeypatch, runner)
    job_id = MB.enqueue("fake_heavy", _FakeSpec(tag="c"))
    MB.work_once()
    assert ("release", job_id) in events        # finally released the claim
    view = MB.get(job_id)
    assert view["status"] == "failed"
    assert view["result"]["error"]["code"] == "internal"


def test_abort_cancelled_run_still_releases(bus, monkeypatch):
    def runner(spec, job_id):
        # Simulate a runner that honored a cancel between frames.
        events.append(("runner", job_id))
        return JobResult(job_id=job_id, ok=False,
                         error=JobError(code="cancelled",
                                        message="cancelled", retryable=False))

    events = _events_and_stubs(monkeypatch, runner)
    job_id = MB.enqueue("fake_heavy", _FakeSpec(tag="d"))
    MB.work_once()
    assert ("release", job_id) in events        # abort path releases the claim too
    assert MB.get(job_id)["status"] == "cancelled"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
