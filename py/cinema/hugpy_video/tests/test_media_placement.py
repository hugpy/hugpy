"""Placement projection + the bus-wide listing (GET /video/jobs backbone).

Covers the observability slice that makes in-flight work VISIBLE with placement:
  * media_bus.list_jobs — in-flight filter, FIFO order, awaiting_capacity
    passthrough, ?all terminal bound, limit clamp;
  * placement.job_placement — reservation join (active claim wins, with the
    template device/process overlay), template fallback for a reservable heavy
    task with no active claim, light-task locus fallback, unknown -> None.

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_media_placement.py -q
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from hugpy_video.intel import media_bus as MB
from hugpy_video.intel import placement as PL
from hugpy_video.intel.job_schema import JobSpec


@dataclass(frozen=True)
class _FakeSpec:
    tag: str = "x"


@pytest.fixture
def bus(tmp_path, monkeypatch):
    """A private media bus DB + a registered throwaway 'fake_job' kind so enqueue
    works without touching any real runner/spec."""
    monkeypatch.setattr(MB, "DB_PATH", str(tmp_path / "media_jobs.db"))
    monkeypatch.setattr(MB, "_initialized", False)
    monkeypatch.setitem(MB.JOB_REGISTRY, "fake_job",
                        JobSpec("fake_job", _FakeSpec, ("fake", "job"), "cpu", 60))
    monkeypatch.setitem(MB.SPEC_DESERIALIZERS, "fake_job", lambda d: _FakeSpec(**d))
    yield


def _set_status(job_id, status):
    conn = MB._connect()
    try:
        conn.execute("UPDATE media_jobs SET status=?, updated=? WHERE job_id=?",
                     (status, time.time(), job_id))
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# media_bus.list_jobs
# --------------------------------------------------------------------------- #
def test_list_jobs_in_flight_only_and_fifo(bus):
    a = MB.enqueue("fake_job", _FakeSpec(tag="a"))
    time.sleep(0.01)
    b = MB.enqueue("fake_job", _FakeSpec(tag="b"))
    time.sleep(0.01)
    c = MB.enqueue("fake_job", _FakeSpec(tag="c"))
    _set_status(c, "done")  # terminal -> excluded by default

    rows = MB.list_jobs()
    ids = [r["job_id"] for r in rows]
    assert ids == [a, b]                     # FIFO by created, terminal excluded
    assert all(r["name"] == "fake_job" for r in rows)
    assert rows[0]["status"] == "queued"
    assert rows[0]["progress"] is None


def test_list_jobs_awaiting_capacity_passthrough(bus):
    a = MB.enqueue("fake_job", _FakeSpec(tag="a"))
    marker = {"phase": "awaiting_capacity",
              "reason": {"shortfall_bytes": 123},
              "held_since": time.time(), "overtaken": 2}
    MB.set_progress(a, marker)

    rows = MB.list_jobs()
    row = next(r for r in rows if r["job_id"] == a)
    assert row["progress"]["phase"] == "awaiting_capacity"
    assert row["progress"]["overtaken"] == 2
    assert row["progress"]["reason"]["shortfall_bytes"] == 123


def test_list_jobs_all_appends_terminal_bounded(bus):
    live = MB.enqueue("fake_job", _FakeSpec(tag="live"))
    dones = []
    for i in range(4):
        j = MB.enqueue("fake_job", _FakeSpec(tag=f"d{i}"))
        _set_status(j, "done")
        dones.append(j)

    default_rows = MB.list_jobs()
    assert [r["job_id"] for r in default_rows] == [live]      # no terminals

    all_rows = MB.list_jobs(include_terminal=True, limit=2)
    statuses = [r["status"] for r in all_rows]
    # 1 in-flight (live, under the limit) + at most `limit` terminal rows.
    assert statuses.count("done") <= 2
    assert any(r["job_id"] == live for r in all_rows)


def test_list_jobs_limit_clamped(bus):
    for i in range(3):
        MB.enqueue("fake_job", _FakeSpec(tag=str(i)))
    assert len(MB.list_jobs(limit=0)) >= 1        # clamped up to >=1
    assert len(MB.list_jobs(limit=99999)) == 3    # clamped down, all returned


# --------------------------------------------------------------------------- #
# placement.job_placement
# --------------------------------------------------------------------------- #
def test_placement_template_fallback_reservable(bus):
    # studio_i2v is a reservable heavy task; with no active claim the placement is
    # the TEMPLATE hint derived from the representative (Wan denoise) stage.
    pl = PL.job_placement("no-such-run", "studio_i2v")
    assert pl["source"] == "template"
    assert pl["host"] == "ae"
    assert pl["gpu"] == "cuda:0"
    assert pl["process"] == "P-studio"
    assert "reserved_bytes" not in pl


def test_placement_light_task(bus):
    pl = PL.job_placement("no-such-run", "crop")
    assert pl == {"source": "template", "host": "central", "process": "ffmpeg"}


def test_placement_unknown_task_none(bus):
    assert PL.job_placement("no-such-run", "totally_unknown") is None
    assert PL.job_placement("no-such-run", None) is None


def test_placement_reservation_wins_with_overlay(bus, monkeypatch):
    from hugpy_video.intel.reservation import registry as REG

    # k57: placement reads the registry's READ-ONLY snapshot ({run_id: row}) once
    # per page instead of a per-row get() (whose lapsed-lease sweep took a write
    # lock live renderers contend for), so that is the seam to fake here.
    def _fake_snapshot():
        return {"run-42": {"run_id": "run-42", "state": "active",
                           "worker_id": "ae-worker-1", "gpu": "ae",
                           "task": "studio_i2v", "peak_bytes": 20 * 1024 ** 3}}

    monkeypatch.setattr(REG.reservation_registry, "active_snapshot", _fake_snapshot)
    pl = PL.job_placement("run-42", "studio_i2v")
    assert pl["source"] == "reservation"
    assert pl["worker_id"] == "ae-worker-1"
    assert pl["host"] == "ae"
    assert pl["reserved_bytes"] == 20 * 1024 ** 3
    # device + process come from the template overlay (registry carries only the
    # box-level affinity + worker).
    assert pl["gpu"] == "cuda:0"
    assert pl["process"] == "P-studio"


def test_placement_inactive_reservation_falls_back_to_template(bus, monkeypatch):
    from hugpy_video.intel.reservation import registry as REG
    monkeypatch.setattr(REG.reservation_registry, "get",
                        lambda run_id: {"state": "released", "task": "studio_i2v"})
    pl = PL.job_placement("run-42", "studio_i2v")
    assert pl["source"] == "template"       # released claim is not a placement


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
