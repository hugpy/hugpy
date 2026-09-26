"""Content hashing helpers (stdlib ``hashlib`` only).

Ported from abstract_utilities.hash_utils (full_hash/quick_hash) but written
to take an already-confined path and stream in bounded chunks so a large file
never loads whole into memory.
"""
from __future__ import annotations

import hashlib

_CHUNK = 1024 * 1024  # 1 MiB streaming window


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: str) -> str:
    """Full SHA-256 of a file, streamed in 1 MiB windows."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def quick_hash(path: str, bytes_to_read: int = 16384) -> str:
    """Cheap identity hash: size + first ``bytes_to_read`` bytes. Good for
    dedup pre-filtering, not a cryptographic guarantee."""
    import os
    h = hashlib.sha256()
    h.update(str(os.path.getsize(path)).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(max(0, int(bytes_to_read))))
    return h.hexdigest()
