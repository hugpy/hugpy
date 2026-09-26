"""Worker studio-presence resolves against CENTRAL's shared weights root.

The bug (0.2.1.post24): the worker's presence probe read only its own ``os.environ`` for
the weights root, but ae-worker's process env sets NEITHER STUDIO_WEIGHTS_ROOT nor the hot
root — the render loads from CENTRAL's manifest root (the shared 16T mount) — so every
worker advertised ``models: []`` and every real render was refused
``studio_model_not_on_worker``.

The fix: central advertises its shared studio weights root (``job.studio_weights_root``) in
the register/heartbeat reply; the agent adopts it onto ``state.studio_weights_root`` and its
presence probe checks ``model_index.json`` readability under THAT root (the same one the
render uses). A worker that cannot read the root (e.g. computron, no shared studio weights)
still reports ``[]`` — honest per-worker presence, never a transfer trigger.

These checks exercise the worker agent's ``_studio_capability`` / ``_adopt_studio_weights_root``
directly. hugpy_video's studio spine is required for the presence read, so the test skips
cleanly where it is not installed.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("hugpy_video.intel.studio.presence")

from hugpy_fleet.worker import agent
from hugpy_fleet.worker import plugins


@pytest.fixture(autouse=True)
def _restore_studio_cap_cache():
    """The presence cache is a module global; snapshot + restore it so no leaked probe
    result crosses into another test."""
    saved = dict(agent._STUDIO_CAP_CACHE)
    yield
    agent._STUDIO_CAP_CACHE.clear()
    agent._STUDIO_CAP_CACHE.update(saved)


def _bust_cache():
    # A never-equal root sentinel forces _studio_capability to recompute this call.
    agent._STUDIO_CAP_CACHE.update(at=0.0, value=None, root=object())


def _weights_tree(base, org_name):
    d = os.path.join(base, *org_name.split("/"))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "model_index.json"), "w") as fh:
        fh.write("{}")
    return base


@pytest.fixture
def _renderable(monkeypatch):
    """A box that mounts the spine + has a GPU, with NO worker-local weights env (the ae
    condition that triggered the bug)."""
    monkeypatch.setattr(agent, "detect_gpus", lambda: [{"index": 0, "memory_total": 1}])
    monkeypatch.setattr(plugins, "video_present", lambda: True)
    monkeypatch.delenv("STUDIO_WEIGHTS_ROOT", raising=False)
    monkeypatch.delenv("STUDIO_WEIGHTS_HOT_ROOT", raising=False)


def test_presence_empty_without_central_root(_renderable, tmp_path):
    """The bug condition: renderable box, weights on disk, but no central root known yet
    and no worker env -> the probe can't see the shared weights -> models []."""
    _weights_tree(str(tmp_path / "weights"), "Wan-AI/Wan2.1-T2V-1.3B")
    state = types.SimpleNamespace(studio_weights_root=None)
    _bust_cache()
    cap = agent._studio_capability(state)
    assert cap is not None and cap["render"] is True
    assert cap["models"] == [], cap


def test_presence_uses_central_advertised_root(_renderable, tmp_path):
    """After adopting central's shared root, the SAME box advertises the model it can now
    read under that root — no worker-local env needed."""
    root = _weights_tree(str(tmp_path / "weights"), "Wan-AI/Wan2.1-T2V-1.3B")
    state = types.SimpleNamespace(studio_weights_root=None)

    # Central advertises the root in the reply; the agent adopts it.
    agent._adopt_studio_weights_root(state, {"studio_weights_root": root})
    assert state.studio_weights_root == root

    _bust_cache()
    cap = agent._studio_capability(state)
    assert cap["render"] is True
    assert "wan2.1-t2v-1.3b" in cap["models"], cap
    assert cap["weights_root"] == root


def test_presence_empty_when_root_unreadable(_renderable, tmp_path):
    """computron-like: central advertises a root this box has NO studio weights under ->
    still [] (never advertise a model it can't load)."""
    state = types.SimpleNamespace(studio_weights_root=str(tmp_path / "not-mounted"))
    _bust_cache()
    cap = agent._studio_capability(state)
    assert cap["render"] is True
    assert cap["models"] == [], cap


def test_adopt_ignores_blank_and_keeps_prior(_renderable, tmp_path):
    """A reply with no/blank studio_weights_root must not wipe a known root."""
    root = _weights_tree(str(tmp_path / "weights"), "Wan-AI/Wan2.1-T2V-1.3B")
    state = types.SimpleNamespace(studio_weights_root=root)
    agent._adopt_studio_weights_root(state, {})               # missing key
    assert state.studio_weights_root == root
    agent._adopt_studio_weights_root(state, {"studio_weights_root": ""})   # blank
    assert state.studio_weights_root == root


def test_cache_recomputes_when_central_root_changes(_renderable, tmp_path):
    """The 60s cache is keyed on the central root, so the first beat after central
    advertises it recomputes instead of serving a stale []."""
    root = _weights_tree(str(tmp_path / "weights"), "Wan-AI/Wan2.1-T2V-1.3B")
    state = types.SimpleNamespace(studio_weights_root=None)
    _bust_cache()
    assert agent._studio_capability(state)["models"] == []    # caches [] for root=None
    # adopt the root; the cache must NOT serve the stale [] (root changed None -> root)
    state.studio_weights_root = root
    cap = agent._studio_capability(state)                      # no manual bust
    assert "wan2.1-t2v-1.3b" in cap["models"], cap
