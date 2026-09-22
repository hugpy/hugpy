"""Regression test for the computron garbage-path incident (k30).

``hugpy_platform.platform_facade.env_value`` reads config through
``abstract_essentials.get_env_value``, which parses a ``.env`` file line by
line and hands back everything after the ``=`` VERBATIM — including a trailing
inline ``# comment``. On computron, ``HUGPY_ENGINE_DIR=/mnt/storage/hugpy-worker/
engine          # native llama.cpp engine build dir`` in a real ``.env``
produced a literal directory whose name was the comment text (via
``app_dirs.engine_dir()``/``models_root()`` ``os.makedirs`` calls), plus a
182GB duplicate store.

The fix lives in OUR seam (``platform_facade._sanitize_env_str`` /
``env_value``), not in the ``abstract_essentials`` site-package (which
reinstalls on every worker and would silently drop any local patch).

These tests drive the REAL ``env_value()`` end-to-end against a real ``.env``
file. ``abstract_essentials.get_env_value`` resolves its search path from
``os.getcwd()`` (no path/file_name args are passed at our call site), so we
``chdir`` into a throwaway directory holding the ``.env`` — the same
resolution mechanism the incident happened through — rather than
monkeypatching the reader.

Precedence (a property of the installed ``abstract_essentials``): the process
environment wins over the file; the file is consulted only for keys the
process does not carry. Both sources are sanitised identically.
"""
from __future__ import annotations

import pytest

from hugpy_platform.platform_facade import _sanitize_env_str, env_value

_KEYS = ("HUGPY_ENGINE_DIR", "HUGPY_QUOTED", "HUGPY_CLEAN", "HUGPY_FALLBACK")


@pytest.fixture
def env_dir(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "HUGPY_ENGINE_DIR=/mnt/storage/hugpy-worker/engine          "
        "# native llama.cpp engine build dir\n"
        "HUGPY_QUOTED=\"a#b/weird path\"\n"
        "HUGPY_CLEAN=/mnt/storage/hugpy-worker/models\n"
        # HUGPY_FALLBACK deliberately absent from the file.
    )
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_inline_comment_stripped_from_file_value(env_dir):
    assert env_value("HUGPY_ENGINE_DIR") == "/mnt/storage/hugpy-worker/engine"


def test_quoted_value_keeps_internal_hash_and_drops_quotes(env_dir):
    assert env_value("HUGPY_QUOTED") == "a#b/weird path"


def test_clean_value_unchanged(env_dir):
    assert env_value("HUGPY_CLEAN") == "/mnt/storage/hugpy-worker/models"


def test_missing_from_file_falls_back_to_sanitized_os_environ(env_dir, monkeypatch):
    monkeypatch.setenv("HUGPY_FALLBACK", "  /mnt/storage/fallback   # from os.environ  ")
    assert env_value("HUGPY_FALLBACK") == "/mnt/storage/fallback"


def test_process_environment_wins_over_the_file_and_is_sanitized(env_dir, monkeypatch):
    monkeypatch.setenv("HUGPY_ENGINE_DIR", "/from/process/env   # systemd Environment= line")
    assert env_value("HUGPY_ENGINE_DIR") == "/from/process/env"


def test_unset_everywhere_is_none(env_dir):
    assert env_value("HUGPY_PLATFORM_TEST_NEVER_SET_ANYWHERE_X9") is None


def test_empty_value_reads_as_none(env_dir, monkeypatch):
    monkeypatch.setenv("HUGPY_FALLBACK", "   ")
    assert env_value("HUGPY_FALLBACK") is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("http://host/path#fragment", "http://host/path#fragment"),  # glued '#' is data
        ("'a#b'", "a#b"),                                            # single quotes unwrap
        ("plain/value", "plain/value"),
        ("/a/b   # comment", "/a/b"),
        ("  padded  ", "padded"),
        ('"quoted # not a comment"', "quoted # not a comment"),
    ],
)
def test_sanitizer(raw, expected):
    assert _sanitize_env_str(raw) == expected
