"""hugpy-video CLI: --help, jobs list/registry against a private bus, selftest."""
from __future__ import annotations

import os

import pytest

from hugpy_video import cli


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "hugpy-video" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "jobs" in capsys.readouterr().out


def test_jobs_list_and_registry_on_private_bus(tmp_path, capsys):
    from hugpy_video.intel import media_bus
    media_bus.DB_PATH = os.path.join(tmp_path, "jobs.db")
    media_bus._initialized = False
    assert cli.main(["jobs", "list"]) == 0
    assert "no jobs" in capsys.readouterr().out
    assert cli.main(["jobs", "list", "--json"]) == 0
    assert capsys.readouterr().out.strip() == "[]"
    assert cli.main(["jobs", "registry"]) == 0
    out = capsys.readouterr().out
    assert "studio_i2v" in out and "ffmpeg/crop" in out


def test_state_prints_roots(capsys, monkeypatch, tmp_path):
    from hugpy_video import state
    monkeypatch.setenv("HUGPY_VIDEO_STATE_DIR", str(tmp_path))
    state.reset_state()
    assert cli.main(["state"]) == 0
    out = capsys.readouterr().out
    assert str(tmp_path) in out and "media_jobs_db" in out
    assert state.media_jobs_db_path() == os.path.join(str(tmp_path), "media_jobs.db")
    state.reset_state()


def test_selftest_passes():
    from hugpy_video.selftest import run_selftest
    assert run_selftest() == []
