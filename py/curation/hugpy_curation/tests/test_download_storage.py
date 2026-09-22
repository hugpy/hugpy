"""The review's download path goes through ``hugpy_storage`` — queue when a
daemon is alive, storage's synchronous transfer otherwise — with no transfer
code of its own. Exercised against a fake queue and a fake inline transfer."""

from __future__ import annotations

import pytest

from hugpy_curation.review import download as dl


class FakeQueue:
    """A :class:`DownloadQueue` whose job walks a scripted status sequence."""

    def __init__(self, statuses, alive=True, error=None):
        self.statuses = list(statuses)
        self.alive = alive
        self.error = error
        self.enqueued = []
        self.polls = 0

    def available(self):
        return self.alive

    def enqueue(self, model_key, model, total_bytes=None):
        self.enqueued.append((model_key, model, total_bytes))
        return "job-1"

    def get(self, job_id):
        assert job_id == "job-1"
        status = self.statuses[min(self.polls, len(self.statuses) - 1)]
        self.polls += 1
        if status is None:
            return None
        return {"id": job_id, "status": status, "error": self.error,
                "message": "Queued — waiting for downloader…"}


@pytest.fixture
def no_disk(monkeypatch):
    """``locate`` never touches storage's on-disk layout in these tests."""
    monkeypatch.setattr(dl, "locate", lambda model: "/models/" + model["hub_id"])


def test_model_spec_single_file_and_sharded_quant():
    key, model = dl.review_model_spec("org/repo", "Q4_K_M", ["repo-Q4_K_M.gguf"])
    assert key == "repo-Q4_K_M"
    assert model["filename"] == "repo-Q4_K_M.gguf"
    assert model["framework"] == "gguf" and "include" not in model

    key, model = dl.review_model_spec("org/repo", "Q8_0", ["a-Q8_0-00001-of-00002.gguf",
                                                            "a-Q8_0-00002-of-00002.gguf"])
    assert model["include"] == ["*Q8_0*.gguf"] and "filename" not in model


def test_daemon_alive_enqueues_and_polls_to_completion(no_disk):
    q = FakeQueue(["queued", "running", "running", "completed"])
    sleeps = []
    out = dl.download_quant("org/repo", "Q4_K_M", ["f.gguf"], queue=q,
                            inline=lambda *a: pytest.fail("inline must not run"),
                            sleep=sleeps.append, poll_seconds=0.5, log=lambda s: None)
    assert out == "/models/org/repo"
    assert q.enqueued == [("repo-Q4_K_M", {"hub_id": "org/repo", "framework": "gguf",
                                           "primary_task": "text-generation",
                                           "filename": "f.gguf"}, None)]
    assert sleeps == [0.5, 0.5, 0.5]          # one sleep per non-terminal poll


def test_daemon_failure_raises_with_the_jobs_error(no_disk):
    q = FakeQueue(["running", "failed"], error={"message": "HTTP 401 gated repo"})
    with pytest.raises(RuntimeError, match="download failed: HTTP 401 gated repo"):
        dl.download_quant("org/repo", "Q4_K_M", ["f.gguf"], queue=q,
                          sleep=lambda s: None, log=lambda s: None)


def test_daemon_job_vanishing_is_a_failure(no_disk):
    q = FakeQueue(["queued", None])
    with pytest.raises(RuntimeError, match="download lost"):
        dl.download_quant("org/repo", "Q4_K_M", ["f.gguf"], queue=q,
                          sleep=lambda s: None, log=lambda s: None)


def test_daemon_wait_times_out(no_disk, monkeypatch):
    q = FakeQueue(["running"])
    clock = iter([0.0, 0.0, 10.0, 20.0, 30.0])
    monkeypatch.setattr(dl.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError):
        dl.download_quant("org/repo", "Q4_K_M", ["f.gguf"], queue=q, timeout_seconds=5,
                          sleep=lambda s: None, log=lambda s: None)


def test_no_daemon_runs_storages_inline_transfer(no_disk):
    q = FakeQueue(["completed"], alive=False)
    calls = []
    out = dl.download_quant("org/repo", "Q4_K_M", ["f.gguf"], queue=q,
                            inline=lambda model, key: calls.append((key, model)),
                            log=lambda s: None)
    assert out == "/models/org/repo"
    assert q.enqueued == []
    assert calls == [("repo-Q4_K_M", {"hub_id": "org/repo", "framework": "gguf",
                                      "primary_task": "text-generation",
                                      "filename": "f.gguf"})]


def test_inline_failure_propagates(no_disk):
    q = FakeQueue([], alive=False)

    def boom(model, key):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        dl.download_quant("org/repo", "Q4_K_M", ["f.gguf"], queue=q, inline=boom,
                          log=lambda s: None)


def test_default_queue_is_storages_and_pipeline_delegates(monkeypatch):
    """The pipeline's ``_download`` is a thin call into ``download_quant`` and
    the default queue is the storage adapter (no job store opened at import)."""
    from hugpy_curation.review import pipeline

    assert isinstance(dl.StorageQueue(), dl.DownloadQueue)
    assert isinstance(FakeQueue([]), dl.DownloadQueue)
    seen = {}
    monkeypatch.setattr(dl, "download_quant",
                        lambda hub_id, quant, files, **kw: seen.update(
                            hub_id=hub_id, quant=quant, files=files) or "/dir")
    assert pipeline._download("org/repo", "Q4_K_M", ["f.gguf"]) == "/dir"
    assert seen == {"hub_id": "org/repo", "quant": "Q4_K_M", "files": ["f.gguf"]}


def test_storage_queue_reports_unavailable_when_presence_is_unreadable(monkeypatch):
    import hugpy_storage.downloader.presence as presence

    monkeypatch.setattr(presence, "downloader_alive", lambda: False)
    assert dl.StorageQueue().available() is False
