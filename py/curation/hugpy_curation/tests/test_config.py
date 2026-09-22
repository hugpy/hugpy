"""State roots: historical env vars win, then the package-wide root, then the
platform's app dirs — and the store modules read through config."""

from __future__ import annotations

import os

import pytest

from hugpy_curation import config

ALL_ENV = ("HUGPY_CURATION_ROOT", "HUGPY_CURATION_CACHE", "REVIEW_DB", "DOSSIER_DIR",
           "DOSSIER_TRIAL_ROOT", "REVIEW_CRITERIA_DIR", "DOSSIER_CACHE_DIR")


@pytest.fixture
def clean_env(monkeypatch):
    for name in ALL_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_package_root_places_every_state_store_under_one_tree(clean_env, tmp_path):
    clean_env.setenv("HUGPY_CURATION_ROOT", str(tmp_path))
    assert config.review_root() == str(tmp_path)
    assert config.review_db_path() == os.path.join(str(tmp_path), "reviews.db")
    assert config.dossiers_root() == os.path.join(str(tmp_path), "dossiers")
    assert config.trials_root() == os.path.join(str(tmp_path), "trials")


def test_historical_env_vars_win_over_the_package_root(clean_env, tmp_path):
    clean_env.setenv("HUGPY_CURATION_ROOT", str(tmp_path / "root"))
    clean_env.setenv("REVIEW_DB", "/x/reviews.db")
    clean_env.setenv("DOSSIER_DIR", "/x/dossiers")
    clean_env.setenv("DOSSIER_TRIAL_ROOT", "/x/trials")
    clean_env.setenv("REVIEW_CRITERIA_DIR", "/x/criteria")
    clean_env.setenv("DOSSIER_CACHE_DIR", "/x/cache")
    assert config.review_db_path() == "/x/reviews.db"
    assert config.dossiers_root() == "/x/dossiers"
    assert config.trials_root() == "/x/trials"
    assert config.criteria_dir() == "/x/criteria"
    assert config.fetch_cache_dir() == "/x/cache"


def test_blank_env_values_are_ignored(clean_env, tmp_path):
    clean_env.setenv("HUGPY_CURATION_ROOT", str(tmp_path))
    clean_env.setenv("REVIEW_DB", "   ")
    assert config.review_db_path() == os.path.join(str(tmp_path), "reviews.db")


def test_defaults_come_from_the_platform_app_dirs(clean_env, tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("DEFAULT_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("HUGPY_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("HUGPY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HUGPY_DATA_DIR", str(data))
    from hugpy_platform.app_dirs import cache_dir, config_dir, models_root
    assert config.review_root() == os.path.join(models_root(), "review")
    assert config.criteria_dir() == os.path.join(config_dir(), "review")
    assert config.fetch_cache_dir() == os.path.join(cache_dir(), "discovery-dossier")
    assert config.review_root().startswith(str(tmp_path))
    assert config.criteria_dir().startswith(str(tmp_path))
    assert config.fetch_cache_dir().startswith(str(tmp_path))


def test_curation_cache_env_backs_the_fetch_cache(clean_env, tmp_path):
    clean_env.setenv("HUGPY_CURATION_CACHE", str(tmp_path / "c"))
    assert config.fetch_cache_dir() == str(tmp_path / "c")


def test_store_modules_read_through_config(clean_env, tmp_path):
    clean_env.setenv("HUGPY_CURATION_ROOT", str(tmp_path))
    clean_env.setenv("REVIEW_CRITERIA_DIR", str(tmp_path / "criteria"))
    clean_env.setenv("DOSSIER_CACHE_DIR", str(tmp_path / "cache"))
    from hugpy_curation.dossier import fetch, store as dstore, trial
    from hugpy_curation.review import criteria, store as rstore

    assert rstore.db_path() == os.path.join(str(tmp_path), "reviews.db")
    assert dstore.root_dir() == os.path.join(str(tmp_path), "dossiers")
    assert criteria.criteria_dir() == str(tmp_path / "criteria")
    assert fetch.cache_dir() == str(tmp_path / "cache")
    run_dir = trial.trial_run_dir(label="t")
    assert run_dir.startswith(os.path.join(str(tmp_path), "trials"))
    assert os.path.isdir(os.path.join(run_dir, "keyframes"))


def test_ensure_dir_never_raises(tmp_path):
    target = str(tmp_path / "a" / "b")
    assert config.ensure_dir(target) == target and os.path.isdir(target)
    # An impossible path (a file in the way) is logged, not raised.
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert config.ensure_dir(str(blocker / "child")) == str(blocker / "child")
