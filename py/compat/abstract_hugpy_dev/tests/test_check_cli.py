"""``abstract-hugpy-dev-check`` reports and exits like a gate."""

from __future__ import annotations

import json

from abstract_hugpy_dev import _check


def test_check_passes(capsys):
    assert _check.main(["--json", "--surface"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] > 400
    assert report["failed"] == {}
    assert report["mismatched"] == {}
    assert report["surface_missing"] == []


def test_scan_reports_old_imports_with_targets(tmp_path, capsys):
    src = tmp_path / "caller.py"
    src.write_text(
        "import abstract_hugpy_dev.chaos\n"
        "from abstract_hugpy_dev.bot.config import X\n"
        "from abstract_hugpy_dev.imports import *\n"
        "import os\n")
    assert _check.main(["--json", "scan", str(tmp_path)]) == 0
    rows = json.loads(capsys.readouterr().out)
    by_old = {r["old"]: r for r in rows}
    assert by_old["abstract_hugpy_dev.chaos"]["new"] == "hugpy_ops.chaos"
    assert by_old["abstract_hugpy_dev.bot.config"]["new"] == "hugpy_discord.config"
    assert by_old["abstract_hugpy_dev.imports"]["kind"] == "aggregator"
    assert len(rows) == 3
