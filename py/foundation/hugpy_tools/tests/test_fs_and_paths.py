"""File/path ops + confinement jail."""
from __future__ import annotations

import os

import pytest

from hugpy_tools import fs, paths


# ---- confinement ---------------------------------------------------------- #
def test_confine_inside_ok(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    assert paths.confine(str(tmp_path), "a.txt") == str(tmp_path / "a.txt")
    assert paths.confine(str(tmp_path), ".") == os.path.realpath(str(tmp_path))


def test_confine_dotdot_escape(tmp_path):
    with pytest.raises(paths.PathEscape):
        paths.confine(str(tmp_path), "../" * 6 + "etc/hostname")


def test_confine_absolute_outside(tmp_path):
    with pytest.raises(paths.PathEscape):
        paths.confine(str(tmp_path), "/etc/hostname")


def test_confine_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside_dir"
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET")
    link = tmp_path / "sneaky"
    os.symlink(str(outside), str(link))
    with pytest.raises(paths.PathEscape):
        paths.confine(str(tmp_path), "sneaky/secret.txt")


def test_confine_sibling_prefix_rejected(tmp_path):
    sibling = str(tmp_path) + "-evil"
    os.makedirs(sibling, exist_ok=True)
    with pytest.raises(paths.PathEscape):
        paths.confine(str(tmp_path), sibling)


def test_confine_write_through_symlinked_parent(tmp_path):
    outside = tmp_path.parent / "outside_w"
    outside.mkdir()
    os.symlink(str(outside), str(tmp_path / "outdir"))
    with pytest.raises(paths.PathEscape):
        paths.confine(str(tmp_path), "outdir/evil.txt", for_write=True)


# ---- path helpers --------------------------------------------------------- #
def test_file_parts():
    p = paths.file_parts("/a/b/c/d.txt")
    assert p["basename"] == "d.txt"
    assert p["filename"] == "d"
    assert p["ext"] == ".txt"
    assert p["dirbase"] == "c"
    assert p["parent_dirbase"] == "b"


def test_sanitize_filename():
    assert paths.sanitize_filename("../../etc/passwd") == "passwd"
    assert paths.sanitize_filename("my file!.txt") == "my_file_.txt"
    assert paths.sanitize_filename("///") == "file"


# ---- read / encoding ------------------------------------------------------ #
def test_read_text_utf8(tmp_path):
    f = tmp_path / "u.txt"
    f.write_text("héllo", encoding="utf-8")
    out = fs.read_text(str(f))
    assert out["text"] == "héllo"
    assert out["encoding"] == "utf-8"
    assert out["truncated"] is False


def test_read_text_cp1252_fallback(tmp_path):
    f = tmp_path / "l.txt"
    f.write_bytes("café".encode("cp1252"))
    out = fs.read_text(str(f))
    assert out["encoding"] == "cp1252"
    assert "caf" in out["text"]


def test_read_text_truncation(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("abcdefghij")
    out = fs.read_text(str(f), max_bytes=4)
    assert out["text"] == "abcd"
    assert out["truncated"] is True


def test_read_lines_range(tmp_path):
    f = tmp_path / "lines.txt"
    f.write_text("\n".join("line%d" % i for i in range(1, 11)))
    out = fs.read_lines(str(f), start=3, end=5)
    assert out["text"] == "line3\nline4\nline5"
    assert out["total_lines"] == 10
    assert out["returned"] == 3


def test_read_lines_numbered_to_eof(tmp_path):
    f = tmp_path / "lines.txt"
    f.write_text("a\nb\nc")
    out = fs.read_lines(str(f), start=2, number=True)
    assert out["text"].splitlines()[0].endswith("\tb")
    assert out["end"] == 3


# ---- atomic write / edit -------------------------------------------------- #
def test_atomic_write_and_append(tmp_path):
    f = tmp_path / "sub" / "w.txt"
    r = fs.atomic_write(str(f), "hello")
    assert r["mode"] == "overwrite"
    assert (tmp_path / "sub" / "w.txt").read_text() == "hello"
    fs.atomic_write(str(f), " world", append=True)
    assert f.read_text() == "hello world"
    # no temp files left behind
    assert [p.name for p in (tmp_path / "sub").iterdir()] == ["w.txt"]


def test_edit_replace_counts(tmp_path):
    f = tmp_path / "e.txt"
    f.write_text("a a a a")
    r = fs.edit_replace(str(f), "a", "b")
    assert r["replaced"] == 4
    assert f.read_text() == "b b b b"


def test_edit_replace_bounded(tmp_path):
    f = tmp_path / "e.txt"
    f.write_text("x x x")
    r = fs.edit_replace(str(f), "x", "y", count=2)
    assert r["replaced"] == 2
    assert f.read_text() == "y y x"


def test_edit_replace_not_found(tmp_path):
    f = tmp_path / "e.txt"
    f.write_text("abc")
    with pytest.raises(ValueError):
        fs.edit_replace(str(f), "zzz", "q")


def test_edit_replace_empty_old(tmp_path):
    f = tmp_path / "e.txt"
    f.write_text("abc")
    with pytest.raises(ValueError):
        fs.edit_replace(str(f), "", "q")


# ---- info / list / tree --------------------------------------------------- #
def test_file_info_with_hash(tmp_path):
    f = tmp_path / "h.txt"
    f.write_text("hash me")
    info = fs.file_info(str(f), hash_files=True)
    assert info["type"] == "file"
    assert info["is_dir"] is False
    assert info["size"] == len("hash me")
    assert len(info["sha256"]) == 64


def test_list_dir_sorted_and_filtered(tmp_path):
    (tmp_path / "b.py").write_text("")
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "zdir").mkdir()
    out = fs.list_dir(str(tmp_path))
    assert out["entries"][0]["type"] == "dir"  # dirs first
    py = fs.list_dir(str(tmp_path), glob="*.py")
    assert [e["name"] for e in py["entries"]] == ["b.py"]
    only_dirs = fs.list_dir(str(tmp_path), dirs_only=True)
    assert [e["name"] for e in only_dirs["entries"]] == ["zdir"]


def test_tree_bounded(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b").mkdir()
    (tmp_path / "a" / "b" / "deep.txt").write_text("")
    (tmp_path / "top.txt").write_text("")
    out = fs.tree(str(tmp_path), max_depth=1)
    assert "a/" in out["text"]
    assert "deep.txt" not in out["text"]  # depth-limited
    full = fs.tree(str(tmp_path), max_depth=5)
    assert "deep.txt" in full["text"]


def test_tree_skips_dir_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "inside.txt").write_text("")
    os.symlink(str(real), str(tmp_path / "link"))
    out = fs.tree(str(tmp_path), max_depth=5)
    # the symlinked dir name shows but is not recursed into
    assert out["text"].count("inside.txt") == 1
