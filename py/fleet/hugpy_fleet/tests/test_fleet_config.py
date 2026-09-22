"""Fleet state roots come from hugpy_platform and are injectable."""

from __future__ import annotations

import os

from hugpy_fleet.central import config as C
from hugpy_fleet.central import workers as W


def test_defaults_derive_from_platform_manifest_dir():
    C.reset()
    assert C.settings.manifest_path
    assert C.state_dir() == (os.path.dirname(C.settings.manifest_path) or ".")
    assert C.state_path("workers.json") == W._default_workers_path()


def test_env_state_dir_override(monkeypatch, tmp_path):
    C.reset()
    monkeypatch.setenv("HUGPY_FLEET_STATE_DIR", str(tmp_path))
    assert C.state_dir() == str(tmp_path)
    assert W._default_workers_path() == str(tmp_path / "workers.json")
    assert W._assign_memory_path().startswith(str(tmp_path))
    monkeypatch.delenv("HUGPY_FLEET_STATE_DIR")
    assert C.state_dir() != str(tmp_path)


def test_configure_and_reset(tmp_path):
    try:
        C.configure(state_dir=str(tmp_path / "s"), manifest_path=str(tmp_path / "m.json"),
                    storage_root=str(tmp_path))
        assert C.state_dir() == str(tmp_path / "s")
        assert C.manifest_path() == str(tmp_path / "m.json")
        assert C.storage_root() == str(tmp_path)
        # attribute injection (the legacy ``settings.manifest_path = ...`` idiom)
        C.settings.state_dir_override = None
        C.settings.manifest_path = str(tmp_path / "x" / "manifest.json")
        assert C.state_dir() == str(tmp_path / "x")
    finally:
        C.reset()
    assert C.settings.state_dir_override is None
