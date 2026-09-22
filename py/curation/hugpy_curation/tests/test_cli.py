"""``hugpy-curation``: ``--help`` on every subcommand, ``dossier list`` over a
temp store, ``install-providers`` wiring the oracle seam, and the ``review``
pass-through to the pipeline CLI."""

from __future__ import annotations

import json

import pytest

from hugpy_curation import cli


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("DOSSIER_DIR", str(tmp_path / "dossiers"))
    monkeypatch.setenv("REVIEW_CRITERIA_DIR", str(tmp_path / "criteria"))
    monkeypatch.setenv("REVIEW_DB", str(tmp_path / "reviews.db"))
    from hugpy_oracle.providers import reset_providers
    reset_providers()
    yield
    reset_providers()


@pytest.mark.parametrize("argv", [["--help"], ["review", "--help"], ["dossier", "--help"],
                                  ["dossier", "list", "--help"], ["install-providers", "--help"]])
def test_help_exits_zero(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out


def test_top_level_help_lists_the_subcommands(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for name in ("review", "dossier", "install-providers"):
        assert name in out


def test_version(capsys):
    from hugpy_curation import __version__
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_dossier_list_empty_and_populated(capsys, tmp_path):
    assert cli.main(["dossier", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []

    from hugpy_curation.dossier import store as dstore
    from hugpy_curation.dossier.dossier import ModelDossier
    dstore.save(ModelDossier(hub_id="org/alpha", criteria="nightly"))
    dstore.save(ModelDossier(hub_id="org/beta", criteria="vision"))

    assert cli.main(["dossier", "list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert sorted((r["criteria"], r["hub_id"]) for r in rows) == [
        ("nightly", "org/alpha"), ("vision", "org/beta")]
    assert cli.main(["dossier", "list", "vision"]) == 0
    text = capsys.readouterr().out
    assert "org/beta" in text and "org/alpha" not in text


def test_install_providers_wires_the_oracle(capsys, tmp_path):
    from hugpy_oracle import providers as op
    assert cli.main(["install-providers"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dossier_source"]["installed"] is True
    assert report["dossier_source"]["root"] == str(tmp_path / "dossiers")
    assert op.get_dossier_source().root() == str(tmp_path / "dossiers")


def test_review_passthrough_reaches_the_pipeline_cli(capsys):
    # `criteria list` on an empty criteria dir is the cheapest real subcommand.
    assert cli.main(["review", "criteria", "list"]) == 0
    assert "(no saved criteria)" in capsys.readouterr().out
    assert cli.main(["review", "criteria", "set", "nightly", "--set", "query=qwen3"]) == 0
    assert cli.main(["review", "criteria", "show", "nightly"]) == 0
    assert json.loads(capsys.readouterr().out.split("saved", 1)[-1].split("\n", 1)[1])["query"] == "qwen3"


def test_review_without_arguments_prints_the_pipeline_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["review"])
    assert exc.value.code == 0
    assert "screen" in capsys.readouterr().out
