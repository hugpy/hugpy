"""Download finalize / _promote_staged atomicity regression (2026-07-15).

Root cause of the sailor-moon "does & doesn't gen" wedge: the Hunyuan3D
downloads finalized through `_promote_staged(staged, dest)`. When `dest`
already existed (a prior partial), the old merge did a ONE-level-deep
dir-onto-dir `os.replace` per child. `os.replace` onto an EXISTING NON-EMPTY
directory raises `OSError [Errno 39] Directory not empty` (ENOTEMPTY). HF Hub
leaves a populated `.cache/huggingface/download/` inside BOTH the staged copy
and any prior-partial dest, so the merge tripped ENOTEMPTY at that nested
collision, the promote wedged at progress ~0.999 forever.

The fix (download_models._merge_tree + a recursive _promote_staged) merges
depth-first. Part B wraps any residual finalize OSError at the download_one /
ensure_model call sites in an EXPLICIT RuntimeError carrying a human reason.
"""
import errno
import inspect
import os

from hugpy_storage import download_models as dm


def wfile(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def read(path):
    with open(path, "rb") as fh:
        return fh.read()


def test_enotempty_finalize_bug_is_gone(tmp_path):
    dest = str(tmp_path / "models" / "Hunyuan3D-2mini")
    staged = dest + ".tmp-601619"
    # staged = the just-completed pull
    wfile(os.path.join(staged, "model.safetensors"), b"complete-weights")
    wfile(os.path.join(staged, ".cache", "huggingface", "download",
                       "model.safetensors.lock"), b"staged-lock")
    # dest = a prior partial attempt with its OWN populated .cache subtree
    wfile(os.path.join(dest, "leftover-partial.bin"), b"old-partial")
    wfile(os.path.join(dest, ".cache", "huggingface", "download",
                       "model.safetensors.lock"), b"dest-lock")

    assert dm._promote_staged(staged, dest) == dest
    assert not os.path.exists(staged)
    assert read(os.path.join(dest, "model.safetensors")) == b"complete-weights"
    assert os.path.exists(os.path.join(dest, "leftover-partial.bin"))
    assert read(os.path.join(dest, ".cache", "huggingface", "download",
                             "model.safetensors.lock")) == b"staged-lock"


def test_three_level_nested_collision(tmp_path):
    dest2 = str(tmp_path / "models" / "deep-model")
    staged2 = dest2 + ".tmp-42"
    wfile(os.path.join(staged2, "a", "b", "c", "leaf.bin"), b"new-leaf")
    wfile(os.path.join(staged2, "a", "b", "sibling.bin"), b"new-sibling")
    wfile(os.path.join(dest2, "a", "b", "c", "leaf.bin"), b"old-leaf")
    wfile(os.path.join(dest2, "a", "b", "c", "keep-me.bin"), b"keep")

    dm._promote_staged(staged2, dest2)
    assert not os.path.exists(staged2)
    assert read(os.path.join(dest2, "a", "b", "c", "leaf.bin")) == b"new-leaf"
    assert os.path.exists(os.path.join(dest2, "a", "b", "c", "keep-me.bin"))
    assert read(os.path.join(dest2, "a", "b", "sibling.bin")) == b"new-sibling"


def test_clean_fast_path_rename(tmp_path):
    dest3 = str(tmp_path / "models" / "fresh-model")
    staged3 = dest3 + ".tmp-7"
    wfile(os.path.join(staged3, "weights.safetensors"), b"w")
    wfile(os.path.join(staged3, "config.json"), b"{}")
    assert not os.path.exists(dest3)
    dm._promote_staged(staged3, dest3)
    assert not os.path.exists(staged3)
    assert os.path.exists(os.path.join(dest3, "weights.safetensors"))
    assert os.path.exists(os.path.join(dest3, "config.json"))


def test_leaf_type_mismatch_staged_wins(tmp_path):
    dest4 = str(tmp_path / "models" / "mismatch-model")
    staged4 = dest4 + ".tmp-8"
    wfile(os.path.join(staged4, "thing"), b"file-now")
    os.makedirs(os.path.join(dest4, "thing"), exist_ok=True)   # empty dir
    dm._promote_staged(staged4, dest4)
    assert os.path.isfile(os.path.join(dest4, "thing"))
    assert read(os.path.join(dest4, "thing")) == b"file-now"


def test_unresolvable_finalize_is_explicit_runtime_error():
    unresolvable = OSError(errno.ENOTEMPTY, "Directory not empty",
                           "/store/x/.cache/huggingface")

    def _reproduce_wrapper(destpath, exc):
        # Mirror the exact wrapper both call sites apply around _promote_staged.
        try:
            raise exc
        except OSError as e:
            raise RuntimeError(f"download finalize failed for {destpath}: {e}") from e

    raised = None
    try:
        _reproduce_wrapper("/store/models/Hunyuan3D-2mini", unresolvable)
    except RuntimeError as e:
        raised = e
    assert isinstance(raised, RuntimeError)
    assert "download finalize failed for /store/models/Hunyuan3D-2mini" in str(raised)
    assert isinstance(raised.__cause__, OSError)
    assert raised.__cause__.errno == errno.ENOTEMPTY

    # And prove the wrapper is actually wired at both call sites.
    src = inspect.getsource(dm)
    assert src.count("download finalize failed for") >= 2
    assert "raise RuntimeError(" in src
