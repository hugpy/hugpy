"""State roots (``hugpy_oracle.config``): every per-store env override is
honoured, the package-wide roots apply beneath them, and the modules that
own the stores read through the one place."""

from __future__ import annotations

import os

from hugpy_oracle import config


def test_ledger_path_honours_the_test_conftest_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ORACLE_LEDGER_PATH", str(tmp_path / "ledger.sqlite"))
    assert config.ledger_path() == str(tmp_path / "ledger.sqlite")
    from hugpy_oracle import selection
    assert selection.default_ledger_path() == str(tmp_path / "ledger.sqlite")


def test_package_home_applies_beneath_the_per_store_overrides(monkeypatch, tmp_path):
    monkeypatch.delenv("ORACLE_LEDGER_PATH", raising=False)
    monkeypatch.delenv("ORACLE_BENCHMARK_ROOT", raising=False)
    monkeypatch.setenv("HUGPY_ORACLE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(config, "LEGACY_BENCHMARK_ROOT", str(tmp_path / "absent"))
    assert config.ledger_path() == os.path.join(str(tmp_path / "home"), "reliability.sqlite")
    assert config.benchmark_root() == os.path.join(str(tmp_path / "home"), "model-battery")
    monkeypatch.setenv("ORACLE_BENCHMARK_ROOT", str(tmp_path / "battery"))
    assert config.benchmark_root() == str(tmp_path / "battery")
    from hugpy_oracle import benchmark
    assert benchmark.default_run_root() == str(tmp_path / "battery")


def test_run_roots_share_one_package_root(monkeypatch, tmp_path):
    for name in ("HUGPY_PERFORMANCE_RUN_ROOT", "HUGPY_SCRIPT_FIRST_RUN_ROOT",
                 "HUGPY_INTERIM_LEDGER_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HUGPY_ORACLE_RUNS_ROOT", str(tmp_path / "runs"))
    from hugpy_oracle import interim_ledger, performance, script_first
    assert performance.default_run_root() == str(tmp_path / "runs")
    assert script_first.default_run_root() == str(tmp_path / "runs")
    assert interim_ledger.default_run_root() == str(tmp_path / "runs")
    assert config.tts_output_dir() == os.path.join(str(tmp_path / "runs"), "tts")
    monkeypatch.setenv("HUGPY_PERFORMANCE_RUN_ROOT", str(tmp_path / "perf"))
    assert performance.default_run_root() == str(tmp_path / "perf")
    assert script_first.default_run_root() == str(tmp_path / "runs")


def test_the_platform_default_is_used_when_nothing_is_set(monkeypatch, tmp_path):
    for name in ("HUGPY_ORACLE_HOME", "ORACLE_LEDGER_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HUGPY_HOME", str(tmp_path / "hh"))
    assert config.oracle_home() == os.path.join(str(tmp_path / "hh"), "oracle")
