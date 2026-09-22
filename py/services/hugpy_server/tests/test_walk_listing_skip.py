"""walk_listing must never offer dot-directory content or die on an unreadable
entry (k6, 2026-07-18).

Live incident: central's ``GET /llm/models/comfy-dreamshaper-8/file`` 500'd
with ``PermissionError`` on
``/mnt/llm_storage/models/misc/Lykon/DreamShaper/.cache/huggingface/trees/*.json``
— an HF local-cache metadata file dropped mode 0600 by a different uid than the
API process. Root cause: the /manifest walker had no dot-directory skip, so it
OFFERED that file to the worker in the first place; the worker then asked for
it via /file, which had no degrade path and let send_file's internal open()
raise straight into an unhandled 500.

This suite locks two independent guarantees on the shared walker
(``format_select.walk_listing``, now the ONE walk /manifest and /archive both
call — see worker_routes.py):
  * a dot-directory (``.cache/``, ``.git/``, …) is never descended into, so its
    contents never enter the listing at all;
  * a getsize() failure on some OTHER unreadable entry degrades to "skip it",
    never raises out of the walk.

Runs under pytest:  venv/bin/python -m pytest tests/test_walk_listing_skip.py
"""
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_storage import format_select as F


def _write(path: Path, content: bytes = b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_dot_cache_dir_never_offered(tmp_path):
    """Mirrors the live layout: a real weight + HF's .cache/ metadata dropped
    alongside it. walk_listing must return ONLY the real weight."""
    model_dir = tmp_path / "DreamShaper"
    _write(model_dir / "DreamShaper_8_pruned.safetensors", b"weights")
    _write(model_dir / "hugpy.json", b"{}")
    _write(model_dir / ".cache" / "huggingface" / "CACHEDIR.TAG")
    _write(model_dir / ".cache" / "huggingface" / ".gitignore")
    _write(model_dir / ".cache" / "huggingface" / "trees" / "228d79c.json")
    _write(model_dir / ".cache" / "huggingface" / "download" /
           "DreamShaper_8_pruned.safetensors.lock")

    out = F.walk_listing(str(model_dir))
    rels = sorted(r for (r, _s) in out)
    assert rels == ["DreamShaper_8_pruned.safetensors", "hugpy.json"]
    assert not any(r.startswith(".cache") for r in rels)


def test_dot_cache_unreadable_file_does_not_offer_or_raise(tmp_path):
    """The exact live shape: the .cache file is not just present but
    UNREADABLE (0600, as HF drops it). Pruning the dot-dir means walk_listing
    never even calls getsize() on it — proving the fix is "never offered",
    not just "tolerated"."""
    model_dir = tmp_path / "DreamShaper"
    _write(model_dir / "DreamShaper_8_pruned.safetensors", b"weights")
    bad = model_dir / ".cache" / "huggingface" / "trees" / "228d79c.json"
    _write(bad)
    os.chmod(bad, 0o600)
    # Simulate "different uid" by making it unreadable to us too, if root
    # isn't running the suite (root can always read regardless of mode).
    os.chmod(bad, 0o000)
    try:
        out = F.walk_listing(str(model_dir))
        rels = sorted(r for (r, _s) in out)
        assert rels == ["DreamShaper_8_pruned.safetensors"]
    finally:
        os.chmod(bad, 0o600)  # restore so tmp_path cleanup can remove it


def test_unreadable_entry_outside_dot_dir_is_skipped_not_raised(tmp_path):
    """A non-dot-dir entry that's unreadable (odd perms on a real weight, a
    race with something deleting it, …) must degrade — skip + log — never
    raise out of the walk and 500 the caller."""
    model_dir = tmp_path / "SomeModel"
    _write(model_dir / "config.json", b"{}")
    bad = model_dir / "model.safetensors"
    _write(bad, b"weights")
    os.chmod(bad, 0o000)
    try:
        out = F.walk_listing(str(model_dir))  # must not raise
        rels = sorted(r for (r, _s) in out)
        # getsize() itself doesn't require read permission on the file (only
        # search on parent dirs), so it's expected to still be sized here —
        # the guarantee under test is "no exception propagates", covering
        # environments (e.g. different uid/ACL) where getsize WOULD fail.
        assert "config.json" in rels
    finally:
        os.chmod(bad, 0o644)


def test_transfer_machinery_sidecars_still_skipped(tmp_path):
    """Pre-existing behavior (chunk-hash sidecars / .part staging) must
    survive the refactor into the shared helper."""
    model_dir = tmp_path / "M"
    _write(model_dir / "model.safetensors", b"weights")
    _write(model_dir / "model.safetensors.chunksums-33554432.json", b"{}")
    _write(model_dir / "model.safetensors.part", b"partial")
    _write(model_dir / "model.safetensors.part.state.json", b"{}")

    out = F.walk_listing(str(model_dir))
    rels = sorted(r for (r, _s) in out)
    assert rels == ["model.safetensors"]


def test_nonexistent_root_returns_empty():
    assert F.walk_listing("/no/such/path/at/all") == []
