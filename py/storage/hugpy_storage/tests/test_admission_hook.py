"""Post-download admission — the storage half: the install hook enqueues ONE
job per install (idempotent per model_key + manifest.captured_at), marks the
model pending on its hugpy.json, never fires on a worker, and the marker
writer carries admission + captured Hub facts across re-stamps. Also the
vision-GGUF task rule and the purpose-gated repo-info read."""
from __future__ import annotations

import json
import os
import socket

import pytest

from hugpy_storage import admission as adm
from hugpy_storage import hugpy_marker as hm


class _NullStore:
    def get_repo_info(self, hub_id):
        return None


@pytest.fixture(autouse=True)
def _no_real_metadata_store(monkeypatch):
    """Never touch the operator's real metadata store from a test."""
    monkeypatch.setattr("hugpy_storage.model_metadata.model_metadata_store", _NullStore())


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.delenv("WORKER_CENTRAL_URL", raising=False)
    monkeypatch.delenv(adm.OFF_ENV, raising=False)
    monkeypatch.setattr("hugpy_storage.model_physical.forget_physical",
                        lambda *a, **k: True, raising=False)
    return adm.AdmissionQueue(str(tmp_path / "q.sqlite"))


def _install(tmp_path, name="M-GGUF", captured="2026-09-23T00:00:00+00:00"):
    d = tmp_path / "models" / "gguf" / "owner" / name
    d.mkdir(parents=True)
    (d / "m.gguf").write_bytes(b"GGUF" + b"\0" * 64)
    hm.write_hugpy_marker(str(d), hub_id=f"owner/{name}", name=name, framework="gguf",
                          tasks=["text-generation"], source="download",
                          manifest={"revision": "abc", "captured_at": captured,
                                    "source": "huggingface", "files": []})
    return str(d)


def test_stamp_enqueues_exactly_once_per_install(tmp_path, queue):
    d = _install(tmp_path)
    j1 = adm.on_install_complete(d, "owner_M-GGUF", queue=queue)
    j2 = adm.on_install_complete(d, "owner_M-GGUF", queue=queue)
    assert j1 and j2 and j1["id"] == j2["id"]
    assert len(queue.list()) == 1
    job = queue.list()[0]
    assert job["model_key"] == "M-GGUF"          # the marker's declared name, not the job key
    assert job["captured_at"] == "2026-09-23T00:00:00+00:00"
    block = adm.read_admission(d)
    assert block["status"] == "pending" and block["job"] == j1["id"]
    assert set(block) >= {"status", "reason", "integrity", "grade", "at", "job"}


def test_new_install_capture_is_a_new_job(tmp_path, queue):
    d = _install(tmp_path)
    adm.on_install_complete(d, None, queue=queue)
    marker = hm.read_hugpy_marker(d)
    marker["manifest"]["captured_at"] = "2026-09-24T00:00:00+00:00"
    hm._save_marker(d, marker)
    adm.on_install_complete(d, None, queue=queue)
    assert len(queue.list()) == 2


def test_hook_never_fires_on_a_worker(tmp_path, queue, monkeypatch):
    monkeypatch.setenv("WORKER_CENTRAL_URL", "http://central.invalid:7002")
    d = _install(tmp_path)
    assert adm.on_install_complete(d, None, queue=queue) is None
    assert queue.list() == []


def test_hook_never_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("WORKER_CENTRAL_URL", raising=False)
    bad = adm.AdmissionQueue(str(tmp_path / "file-not-dir"))
    (tmp_path / "file-not-dir").mkdir()          # sqlite cannot open a directory
    d = _install(tmp_path)
    assert adm.on_install_complete(d, None, queue=bad) is None


def test_claim_is_single_and_finish_records(tmp_path, queue):
    d = _install(tmp_path)
    adm.on_install_complete(d, None, queue=queue)
    job = queue.claim_next("runner-a")
    assert job and job["status"] == adm.Q_RUNNING
    assert queue.claim_next("runner-b") is None
    queue.finish(job["id"], adm.Q_DONE, result={"status": "admitted"}, log=["a", "b"])
    got = queue.get(job["id"])
    assert got["status"] == adm.Q_DONE and got["result"]["status"] == "admitted"


def test_restamp_carries_admission_and_manifest(tmp_path, queue):
    d = _install(tmp_path)
    adm.write_admission(d, adm.admission_record(adm.HELD, reason="faulty_model: x", integrity="faulty_model"))
    hm.write_hugpy_marker(d, hub_id="owner/M-GGUF", name="M-GGUF", framework="gguf",
                          tasks=["text-generation"], source="reclassify")
    m = hm.read_hugpy_marker(d)
    assert m["admission"]["status"] == "held"
    assert m["manifest"]["captured_at"] == "2026-09-23T00:00:00+00:00"


def test_request_admission_always_queues_and_marks_pending(tmp_path, queue):
    d = _install(tmp_path)
    adm.write_admission(d, adm.admission_record(adm.HELD, reason="x"))
    job = adm.request_admission("M-GGUF", d, queue=queue)
    assert job["source"] == "rerun" and adm.read_admission(d)["status"] == "pending"


def test_hub_meta_captured_once_from_local_store_only(tmp_path, monkeypatch):
    calls = []

    class Store:
        def get_repo_info(self, hub_id):
            calls.append(hub_id)
            return {"pipeline_tag": "text-generation", "license": "apache-2.0",
                    "safetensors_params": 7, "gated": False, "tags": ["x"]}

    monkeypatch.setattr("hugpy_storage.model_metadata.model_metadata_store", Store())

    def no_net(*a, **k):
        raise AssertionError("network attempted")
    monkeypatch.setattr(socket, "socket", no_net)
    d = _install(tmp_path, name="Hub-GGUF")
    m = hm.read_hugpy_marker(d)
    assert m["hub_meta"]["pipeline_tag"] == "text-generation"
    assert m["hub_meta"]["parameter_count"] == 7
    hm.write_hugpy_marker(d, hub_id="owner/Hub-GGUF", name="Hub-GGUF", source="reclassify")
    assert hm.read_hugpy_marker(d)["hub_meta"]["license"] == "apache-2.0"
    assert calls == ["owner/Hub-GGUF"]           # captured once; the re-stamp carried it
    assert hm.cached_hub_meta("comfy/some-checkpoint") is None


def test_vl_rule_marks_mmproj_gguf_image_text_to_text(tmp_path):
    d = tmp_path / "vl"
    (d / "mmproj").mkdir(parents=True)
    (d / "model.gguf").write_bytes(b"x")
    (d / "mmproj" / "mmproj-f16.gguf").write_bytes(b"x")
    tasks, primary = hm.vl_gguf_tasks(str(d), "gguf", ["text-generation"], "text-generation")
    assert primary == "image-text-to-text" and tasks == ["image-text-to-text", "text-generation"]
    # not a gguf / another task / no projector: unchanged
    assert hm.vl_gguf_tasks(str(d), "transformers", None, None) == (None, None)
    assert hm.vl_gguf_tasks(str(d), "gguf", ["pipeline-component"], "pipeline-component") == \
        (["pipeline-component"], "pipeline-component")
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "model.gguf").write_bytes(b"x")
    assert hm.vl_gguf_tasks(str(plain), "gguf", None, "text-generation") == (None, "text-generation")
    # the marker writer applies it at install
    hm.write_hugpy_marker(str(d), hub_id="o/vl", name="vl", framework="gguf",
                          tasks=["text-generation"], primary_task="text-generation", source="download")
    m = hm.read_hugpy_marker(str(d))
    assert m["primary_task"] == "image-text-to-text" and "text-generation" in m["tasks"]


def test_fetch_repo_info_without_a_fetch_purpose_never_fetches(tmp_path, monkeypatch):
    from hugpy_storage import model_metadata as mm
    store = mm.ModelMetadataStore(str(tmp_path / "meta.db"))
    monkeypatch.setattr(mm, "model_metadata_store", store)

    class Api:
        calls = 0

        def model_info(self, *a, **k):
            Api.calls += 1
            raise AssertionError("live Hub call")
    assert mm.fetch_repo_info("owner/repo", api=Api()) is None
    assert mm.fetch_repo_info("owner/repo", api=Api(), purpose="installed") is None
    assert Api.calls == 0
    store.put_repo_info("owner/repo", {"id": "owner/repo", "siblings": []})
    assert mm.fetch_repo_info("owner/repo", api=Api())["id"] == "owner/repo"
    assert Api.calls == 0
