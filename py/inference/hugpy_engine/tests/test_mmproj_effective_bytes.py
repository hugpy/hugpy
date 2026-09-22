"""_mmproj_bytes sizes the ONE projector the runner loads, not the sum of every
precision variant a repo ships.

The bug: a vision GGUF repo commonly ships the SAME projector at several
precisions (unsloth's Qwen2.5-VL-7B: mmproj-F32 2.5GB + F16 1.26GB + BF16 1.26GB
= ~5GB). ``_mmproj_bytes`` walked the dir and SUMMED them, so ``effective_bytes``
(quant + mmproj) over-counted by ~4GB and wrongly rejected the model on cards it
fits. llama.cpp loads exactly ONE projector (``--mmproj`` / ``find_mmproj``), so
the sizer must size that one — the mirror of the GGUF quant-ladder fix in
``_incoming_need_bytes``.

Runs both ways:
    cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
    venv/bin/python -m pytest tests/test_mmproj_effective_bytes.py -q
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_engine.serve import overrides

GIB = 2 ** 30


def _sparse(path: str, nbytes: int) -> str:
    with open(path, "wb") as fh:
        if nbytes:
            fh.truncate(nbytes)
    return path


def test_sizes_one_projector_not_the_sum_of_all_precisions():
    d = tempfile.mkdtemp()
    _sparse(os.path.join(d, "model-UD-Q6_K_XL.gguf"), 6 * GIB)
    # the unsloth pattern: same projector at three precisions
    _sparse(os.path.join(d, "mmproj-F32.gguf"), 2 * GIB)
    _sparse(os.path.join(d, "mmproj-F16.gguf"), 1 * GIB)
    _sparse(os.path.join(d, "mmproj-BF16.gguf"), 1 * GIB)
    got = overrides._mmproj_bytes(d)
    # exactly one projector's worth — NOT 2+1+1 = 4 GiB
    assert got in (1 * GIB, 2 * GIB), got
    assert got != 4 * GIB


def test_single_projector_unchanged():
    d = tempfile.mkdtemp()
    _sparse(os.path.join(d, "model-Q6.gguf"), 4 * GIB)
    _sparse(os.path.join(d, "mmproj-F16.gguf"), 1234)
    assert overrides._mmproj_bytes(d) == 1234


def test_text_only_dir_is_zero():
    d = tempfile.mkdtemp()
    _sparse(os.path.join(d, "model-Q6.gguf"), 4 * GIB)
    _sparse(os.path.join(d, "notes-mmproj.txt"), 999)  # decoy non-gguf
    assert overrides._mmproj_bytes(d) == 0


def test_missing_dir_is_zero():
    assert overrides._mmproj_bytes("/no/such/dir") == 0


def test_nested_projector_is_one_not_the_sum():
    # projectors nested in a subdir: still ONE projector's worth (the runner's
    # find_mmproj resolves one), never the sum of the two.
    d = tempfile.mkdtemp()
    _sparse(os.path.join(d, "model-Q6.gguf"), 4 * GIB)
    sub = os.path.join(d, "projectors")
    os.makedirs(sub, exist_ok=True)
    _sparse(os.path.join(sub, "mmproj-F16.gguf"), 1500)
    _sparse(os.path.join(sub, "mmproj-F32.gguf"), 3000)
    got = overrides._mmproj_bytes(d)
    assert got in (1500, 3000), got   # one of them, never 4500
    assert got != 4500
