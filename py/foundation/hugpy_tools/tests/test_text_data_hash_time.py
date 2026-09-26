"""text / data / hashkit / timekit."""
from __future__ import annotations

import hashlib
import json

import pytest

from hugpy_tools import data, hashkit, text, timekit


# ---- text ----------------------------------------------------------------- #
def test_count_tokens_monotonic():
    assert text.count_tokens("") == 0
    assert text.count_tokens("hello") >= 1
    assert text.count_tokens("hello world " * 100) > text.count_tokens("hello world")


def test_chunk_by_lines_overlap():
    body = "\n".join(str(i) for i in range(10))
    chunks = text.chunk_by_lines(body, max_lines=4, overlap=1)
    assert chunks[0] == "0\n1\n2\n3"
    # overlap of 1: next chunk starts at line index 3
    assert chunks[1].splitlines()[0] == "3"


def test_chunk_by_tokens_paragraphs():
    para = "word " * 50
    body = "\n\n".join([para, para, para])
    chunks = text.chunk_by_tokens(body, max_tokens=text.count_tokens(para) + 1)
    assert len(chunks) == 3  # each paragraph on its own


def test_unified_diff():
    d = text.unified_diff("a\nb\nc\n", "a\nB\nc\n")
    assert "-b" in d and "+B" in d
    assert text.unified_diff("same\n", "same\n") == ""


# ---- data ----------------------------------------------------------------- #
def test_read_json(tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"k": [1, 2, 3]}))
    assert data.read_data(str(f)) == {"k": [1, 2, 3]}


def test_read_toml(tmp_path):
    pytest.importorskip("tomllib")
    f = tmp_path / "d.toml"
    f.write_text('title = "hi"\n[owner]\nname = "x"\n')
    got = data.read_data(str(f))
    assert got["title"] == "hi"
    assert got["owner"]["name"] == "x"


def test_write_json_atomic(tmp_path):
    f = tmp_path / "out.json"
    r = data.write_json(str(f), {"b": 1, "a": 2}, sort_keys=True)
    assert r["mode"] == "overwrite"
    assert json.loads(f.read_text()) == {"a": 2, "b": 1}


def test_safe_json_dumps_non_serialisable():
    class Weird:
        def __str__(self):
            return "weird"
    out = data.safe_json_dumps({"x": Weird()})
    assert "weird" in out


def test_read_data_unknown_format(tmp_path):
    f = tmp_path / "d.ini"
    f.write_text("x")
    with pytest.raises(ValueError):
        data.read_data(str(f), fmt="ini")


# ---- hashkit -------------------------------------------------------------- #
def test_sha256_text_matches_hashlib():
    assert hashkit.sha256_text("abc") == hashlib.sha256(b"abc").hexdigest()


def test_sha256_file(tmp_path):
    f = tmp_path / "h.bin"
    f.write_bytes(b"payload")
    assert hashkit.sha256_file(str(f)) == hashlib.sha256(b"payload").hexdigest()


def test_quick_hash_stable(tmp_path):
    f = tmp_path / "q.bin"
    f.write_bytes(b"x" * 100)
    assert hashkit.quick_hash(str(f)) == hashkit.quick_hash(str(f))


# ---- timekit -------------------------------------------------------------- #
def test_time_roundtrip():
    iso = timekit.now_iso()
    assert iso.endswith("+00:00")
    ep = timekit.iso_to_epoch(iso)
    assert abs(timekit.iso_to_epoch(timekit.epoch_to_iso(ep)) - ep) < 1.0


def test_iso_z_suffix():
    assert abs(timekit.iso_to_epoch("1970-01-01T00:00:00Z")) < 1.0
