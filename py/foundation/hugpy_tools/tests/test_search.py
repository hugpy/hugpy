"""Tests for hugpy_tools.search (no implicit filtering; presets; reports).

Imports ONLY the search subpackage (proves it is standalone). Run with::

    PYTHONPATH=.../hugpy_tools/src python3 -m pytest tests/test_search.py
"""
from __future__ import annotations

import os
import shutil

import pytest

from hugpy_tools.search import (
    EXCLUDE_NOISE_DIRS,
    Filters,
    collect,
    effective_filters,
    find_content,
    get_file_filters,
    get_files_and_dirs,
    iter_files,
    last_report,
    make_filters,
    read_any_file,
    read_text,
    search_content,
)


def _write(base, rel, body="x\n"):
    p = os.path.join(base, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb" if isinstance(body, bytes) else "w") as f:
        f.write(body)
    return p


def _rel(paths, base):
    return sorted(os.path.relpath(p, base).replace(os.sep, "/") for p in paths)


# --- no implicit filtering ----------------------------------------------------
def test_no_filters_returns_everything(tmp_path):
    base = str(tmp_path)
    _write(base, "src/a.py"); _write(base, "node_modules/x/j.js")
    _write(base, ".git/cfg"); _write(base, ".hidden"); _write(base, "pkg/__init__.py")
    got = _rel(iter_files(base, make_filters()), base)
    assert got == [".git/cfg", ".hidden", "node_modules/x/j.js",
                   "pkg/__init__.py", "src/a.py"]


def test_default_filters_have_no_excludes():
    f = make_filters()
    assert f.exts is None and f.exclude_exts == frozenset()
    assert f.exclude_dirs == frozenset() and f.include_hidden is True
    assert f.follow_symlinks is False and f.max_depth is None and f.max_bytes is None


def test_gfad_no_filters_returns_everything(tmp_path):
    base = str(tmp_path)
    _write(base, "a.py"); _write(base, "node_modules/x.js"); _write(base, ".hidden")
    _, files = get_files_and_dirs(directory=base)
    assert _rel(files, base) == [".hidden", "a.py", "node_modules/x.js"]


# --- presets ------------------------------------------------------------------
def test_preset_code(tmp_path):
    base = str(tmp_path)
    _write(base, "src/a.py"); _write(base, "node_modules/x/j.js"); _write(base, ".hidden")
    f = make_filters(preset="code")
    assert _rel(iter_files(base, f), base) == ["src/a.py"]
    eff = effective_filters(f)
    assert eff["presets"] == ["code"] and "node_modules" in eff["exclude_dirs"]


def test_preset_merges_union_with_user_exts(tmp_path):
    base = str(tmp_path)
    _write(base, "a.py"); _write(base, "b.rs"); _write(base, "node_modules/c.py")
    f = make_filters(preset="code", allowed_exts=[".rs"])
    assert _rel(iter_files(base, f), base) == ["a.py", "b.rs"]


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        make_filters(preset="nope")


def test_exclude_noise_dirs_set_usable_directly(tmp_path):
    base = str(tmp_path)
    _write(base, "a.py"); _write(base, "node_modules/x.py")
    assert _rel(iter_files(base, make_filters(exclude_dirs=EXCLUDE_NOISE_DIRS)), base) == ["a.py"]


# --- include/exclude precedence -----------------------------------------------
def test_exclude_always_wins(tmp_path):
    base = str(tmp_path)
    _write(base, "keep/a.py"); _write(base, "keep/b.py")
    f = make_filters(allowed_patterns=["**/keep/*.py"], exclude_patterns=["*b.py"])
    assert _rel(iter_files(base, f), base) == ["keep/a.py"]


def test_path_glob_matches_relpath(tmp_path):
    base = str(tmp_path)
    _write(base, "src/mct/session.py", "nginx\n"); _write(base, "src/other/session.py", "nginx\n")
    hits = find_content(directory=base, strings=["nginx"], allowed_patterns=["**/mct/*.py"])
    assert _rel((h["file_path"] for h in hits), base) == ["src/mct/session.py"]


def test_segment_exclude_exact(tmp_path):
    base = str(tmp_path)
    _write(base, "build/x.py"); _write(base, "rebuild/y.py")
    assert _rel(iter_files(base, make_filters(exclude_dirs=["build"])), base) == ["rebuild/y.py"]


def test_excluded_dir_pruned(tmp_path):
    base = str(tmp_path)
    _write(base, "src/a.py")
    big = os.path.join(base, "node_modules", "pkg"); os.makedirs(big)
    for i in range(10000):
        open(os.path.join(big, f"f{i}.js"), "w").close()
    entered = []
    files = list(iter_files(base, make_filters(exclude_dirs=["node_modules"]), on_dir=entered.append))
    assert not any("node_modules" in d for d in entered)
    assert _rel(files, base) == ["src/a.py"]


# --- case policy --------------------------------------------------------------
def test_literal_case_insensitive(tmp_path):
    base = str(tmp_path)
    _write(base, "a.py", "forYou\n")
    assert len(find_content(directory=base, strings=["foryou"])) == 1


def test_case_sensitive(tmp_path):
    base = str(tmp_path)
    _write(base, "a.py", "forYou\n")
    assert find_content(directory=base, strings=["foryou"], case_sensitive=True) == []


# --- binary + reports ---------------------------------------------------------
def test_binary_skipped_and_reported(tmp_path):
    base = str(tmp_path)
    _write(base, "t.py", "needle\n"); _write(base, "b.py", b"needle\x00bin\n")
    hits, rep = find_content(directory=base, strings=["needle"], report=True)
    assert _rel((h["file_path"] for h in hits), base) == ["t.py"]
    assert rep["skipped"]["binary"] == 1 and rep["matched"] == 1
    assert last_report()["skipped"]["binary"] == 1


def test_read_any_file_raises_on_binary(tmp_path):
    base = str(tmp_path)
    with pytest.raises(ValueError):
        read_any_file(_write(base, "b.bin", b"\x00\x01"))


def test_read_any_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_any_file(str(tmp_path / "nope"))


def test_read_text_none_on_binary(tmp_path):
    base = str(tmp_path)
    assert read_text(_write(base, "b.bin", b"\x00")) is None


def test_too_large_reported(tmp_path):
    base = str(tmp_path)
    _write(base, "big.py", "needle\n" * 500)
    _, rep = find_content(directory=base, strings=["needle"], max_bytes=10, report=True)
    assert rep["skipped"]["too_large"] == 1


# --- regex vs literal ---------------------------------------------------------
def test_regex(tmp_path):
    base = str(tmp_path)
    _write(base, "a.py", "value = 42\n"); _write(base, "b.py", "value = xx\n")
    hits = find_content(directory=base, strings=[r"value = \d+"], regex=True)
    assert _rel((h["file_path"] for h in hits), base) == ["a.py"]


def test_literal_not_regex(tmp_path):
    base = str(tmp_path)
    _write(base, "a.py", "a.b.c\n"); _write(base, "b.py", "axbxc\n")
    hits = find_content(directory=base, strings=["a.b.c"], regex=False)
    assert _rel((h["file_path"] for h in hits), base) == ["a.py"]


def test_total_strings_intersection(tmp_path):
    base = str(tmp_path)
    _write(base, "both.py", "alpha beta\n"); _write(base, "one.py", "alpha\n")
    hits = find_content(directory=base, strings=["alpha", "beta"], total_strings=True)
    assert _rel((h["file_path"] for h in hits), base) == ["both.py"]


def test_comments_and_urls(tmp_path):
    base = str(tmp_path)
    _write(base, "a.py", "x = 1 // foryou\n"); _write(base, "b.py", "'http://foryou'\n")
    hits = find_content(directory=base, strings=["foryou"], use_rg=False)
    assert _rel((h["file_path"] for h in hits), base) == ["a.py", "b.py"]


# --- rg parity ----------------------------------------------------------------
@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not on PATH")
def test_rg_parity(tmp_path):
    base = str(tmp_path)
    for i in range(50):
        _write(base, f"pkg{i % 5}/mod{i}.py", f"one\nNEEDLE_{i % 3}\nlast\n")
    _write(base, "node_modules/x/y.py", "NEEDLE_0\n")
    for term in ("NEEDLE_0", "needle_1", "absent"):
        rg = find_content(directory=base, strings=[term], use_rg=True, exclude_dirs=["node_modules"])
        py = find_content(directory=base, strings=[term], use_rg=False, exclude_dirs=["node_modules"])
        key = lambda hs: sorted((h["file_path"], tuple((l["line"], l["content"]) for l in h["lines"])) for h in hs)
        assert key(rg) == key(py)
        assert not any("node_modules" in h["file_path"] for h in rg)


# --- compat shapes ------------------------------------------------------------
def test_get_file_filters_and_collect(tmp_path):
    base = str(tmp_path)
    _write(base, "a.py"); _write(base, "sub/b.py"); _write(base, "c.md")
    dirs_, cfg, allowed, include_files, recursive = get_file_filters(base, allowed_exts=[".py"])
    assert isinstance(cfg, Filters) and callable(allowed)
    _, files = get_files_and_dirs(directory=dirs_, cfg=cfg, recursive=recursive)
    assert _rel(files, base) == ["a.py", "sub/b.py"]


def test_get_files_and_dirs_recursive_false(tmp_path):
    base = str(tmp_path)
    _write(base, "a.py"); _write(base, "sub/b.py")
    _, cfg, _a, _i, _r = get_file_filters(base)
    _, files = get_files_and_dirs(directory=[base], cfg=cfg, recursive=False)
    assert _rel(files, base) == ["a.py"]


def test_report_true_returns_tuple(tmp_path):
    base = str(tmp_path)
    _write(base, "a.py")
    (dirs, files), rep = get_files_and_dirs(directory=base, preset="code", report=True)
    assert rep["effective_filters"]["presets"] == ["code"] and rep["op"] == "collect"


def test_find_content_get_lines_false(tmp_path):
    base = str(tmp_path)
    _write(base, "a.py", "term\n")
    out = find_content(directory=base, strings=["term"], get_lines=False)
    assert out and isinstance(out[0], str)
