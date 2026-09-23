"""``hugpy-oracle`` CLI: help, version, and a steward pass in both modes
with no server and no fleet."""

from __future__ import annotations

import json
from importlib.metadata import version

import pytest

from hugpy_oracle import cli


def test_help_and_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "steward" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    # the version is git-derived (lockstep), never a literal — compare with installed metadata
    assert f"hugpy-oracle {version('hugpy-oracle')}" in capsys.readouterr().out


def test_steward_local_pass_reports_on_an_empty_ledger(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HUGPY_API", raising=False)
    monkeypatch.delenv("ORACLE_ROUTING_MATRIX", raising=False)
    from hugpy_oracle import catalog
    monkeypatch.setattr(catalog, "list_capabilities", lambda *a, **k: [])   # no registry read
    rc = cli.main(["steward", "--ledger", str(tmp_path / "ledger.sqlite"), "--compact"])
    body = json.loads(capsys.readouterr().out.strip())
    assert rc == 0
    assert body["ok"] is True
    assert body["applied"] is False
    assert body["ledger_rows"] == 0
    assert body["ledger_path"] == str(tmp_path / "ledger.sqlite")
    assert body["findings"] and body["summary"]          # never silent
    assert not any(f["severity"] == "alarm" for f in body["findings"])


def test_steward_remote_mode_posts_to_the_api(monkeypatch, capsys):
    calls = []

    def fake_remote(base_url, *, apply, timeout=120.0):
        calls.append((base_url, apply))
        return {"ok": False, "findings": [{"kind": "streak", "severity": "alarm"}]}

    monkeypatch.setattr(cli, "run_steward_remote", fake_remote)
    monkeypatch.setenv("HUGPY_API", "http://127.0.0.1:7002")
    rc = cli.main(["steward", "--apply", "--compact"])
    assert rc == 2                                   # an alarm fails the unit
    assert calls == [("http://127.0.0.1:7002", True)]
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_transport_failure_is_exit_1_with_the_reason(monkeypatch, capsys):
    def boom(base_url, *, apply, timeout=120.0):
        raise OSError("connection refused")

    monkeypatch.setattr(cli, "run_steward_remote", boom)
    rc = cli.main(["steward", "--api", "http://127.0.0.1:1"])
    assert rc == 1
    assert "connection refused" in json.loads(capsys.readouterr().out)["error"]


def test_install_hooks_subcommand_reports_without_video_hooks(capsys):
    rc = cli.main(["install-hooks"])
    assert rc == 0
    body = json.loads(capsys.readouterr().out)
    assert set(body) == {"video_hooks", "prompt_coordinator", "performance_runner"}
