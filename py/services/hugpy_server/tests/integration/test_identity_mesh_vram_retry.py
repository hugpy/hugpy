"""identity_mesh.run — the video_intel comfy station honours the one-shot
VRAM-class retry contract (k71). Split out of hugpy_media's
``test_comfy_vram_retry.py``: it needs ``hugpy_video`` (a non-dependency of
media) plus media's ``vram_retry`` classification, so it is an integration test.
"""
from __future__ import annotations

import types

import pytest

identity_mesh = pytest.importorskip("hugpy_video.intel.runners.identity_mesh")

SPEC = types.SimpleNamespace(slug="hero", recon_id="rc1", view_ids=("v0",))


@pytest.fixture(autouse=True)
def _no_settle(monkeypatch):
    monkeypatch.setenv("HUGPY_VRAM_RETRY_SETTLE_S", "0")
    monkeypatch.setattr(identity_mesh, "_resolve_view_uris",
                        lambda slug, rid, vids: {"v0": "/tmp/v0.png"})


def test_mesh_retryable_first_failure_retries_once(monkeypatch):
    calls = {"submit": 0, "cancel": []}
    mesh_oom = RuntimeError('ComfyUI execution error: {"exception_type": '
                            '"torch.OutOfMemoryError"}')
    mesh_oom.comfy_prompt_id = "mesh-first"
    script = [mesh_oom,
              RuntimeError("ComfyUI execution error: missing node "
                           "Hunyuan3DShapeGenerator")]

    def fake_submit(payload):
        calls["submit"] += 1
        raise script.pop(0)

    monkeypatch.setattr(identity_mesh, "_submit_and_wait", fake_submit)
    monkeypatch.setattr(identity_mesh, "_cancel_and_settle",
                        lambda pid, job: calls["cancel"].append(pid))

    out = identity_mesh.run(SPEC)
    assert calls["submit"] == 2, "retryable first failure -> exactly one retry"
    assert calls["cancel"] == ["mesh-first"], "first submission cancelled by its prompt id"
    assert out["ok"] is False and "missing node" in out["error"]["message"]


def test_mesh_non_retryable_does_not_retry(monkeypatch):
    calls = {"submit": 0, "cancel": []}

    def fake_submit_bad(payload):
        calls["submit"] += 1
        raise RuntimeError("ComfyUI execution error: value not in list")

    monkeypatch.setattr(identity_mesh, "_submit_and_wait", fake_submit_bad)
    monkeypatch.setattr(identity_mesh, "_cancel_and_settle",
                        lambda pid, job: calls["cancel"].append(pid))
    out = identity_mesh.run(SPEC)
    assert calls["submit"] == 1 and calls["cancel"] == [] and out["ok"] is False
