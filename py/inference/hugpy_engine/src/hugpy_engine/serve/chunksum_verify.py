"""CONTENT VERIFICATION against the store's ``.chunksums-<N>.json`` sidecars.

Why this exists (incident, 2026-07-15/16)
-----------------------------------------
Worker ``ae`` failed image generation with::

    OSError: Error no file named diffusion_pytorch_model.bin found in
             directory /mnt/hot990/hugpy-worker/models/.../sd-turbo/vae

That error was a LIE. diffusers found no *usable* safetensors, fell back to
hunting the legacy ``.bin``, and reported the fallback's absence. The real
cause: the whole sd-turbo tree on the worker's hot cache was ``.part`` files,
wedged for 32+ hours — and the vae ``.part`` was **167335342 bytes, byte-for-
byte the same SIZE as the good file in the shared store**, but its bytes were
CORRUPT (md5 differed; the safetensors header would not even parse).

Right size, wrong bytes. **A size check passes it.** That is precisely what
the promote gate did, so a corrupt copy was promoted onto its final name and
sat there looking authoritative until a loader tripped over it 32h later,
pointing at the wrong file.

The fix is to verify CONTENT, not length. The shared store already ships
per-chunk SHA-256 sidecars right next to the weights (written by central's
``_chunk_sums`` in ``flask_app/app/routes/worker_routes.py``) — nothing was
reading them on the local copy path. This module reads them.

Sidecar format (verbatim, as found on the store)
------------------------------------------------
``<file>.chunksums-<chunk_bytes>.json``::

    {"size": 167335342, "mtime": 1782262666, "chunk_bytes": 33554432,
     "sums": ["c0220d2c…", "4295e368…", …]}

``sums[i]`` = SHA-256 of byte range ``[i*chunk_bytes, (i+1)*chunk_bytes)``.
The sidecar is keyed by ``size`` + ``mtime`` of the file it describes, so a
re-uploaded/edited source invalidates it rather than vouching for stale bytes.

Design decisions
----------------
ABSENT SIDECAR IS NOT A FAILURE.
    Not every file has one (sidecars are generated lazily, on first worker pull
    of that file, and can't be written to a read-only mount). Treating "no
    sidecar" as a hard error would break every legitimate copy of every file
    that has never been pulled — it would convert a *missing optimization* into
    a *fleet-wide outage*, and it would do so on the default path. Absent
    sidecar therefore yields ``UNVERIFIED``: the copy proceeds, and we say so
    honestly at INFO. We only ever *reject* on positive evidence of corruption.

STALE SIDECAR IS ALSO NOT A FAILURE.
    If size/mtime don't match, the sidecar describes a different revision of
    the file — it is stale bookkeeping, not evidence about these bytes. Same
    treatment as absent: ``UNVERIFIED``, never a rejection. Trusting a stale
    sidecar would produce false CORRUPT verdicts on legitimately-updated
    weights, which is the more dangerous error.

MEMORY-BOUNDED BY CONSTRUCTION.
    Weights here run 100MB–10GB+. We stream one chunk at a time (32MiB by
    default) and hash it, so peak RSS is one chunk regardless of file size.
    Central runs ``HUGPY_NO_LOCAL_SERVING=true`` and must never balloon on a
    weight pull. We also short-circuit on the FIRST bad chunk — a corrupt file
    is rejected without reading its remaining gigabytes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# Verdicts. Deliberately three-valued: "I proved it good", "I proved it bad",
# and "I have no evidence" — collapsing the third into either of the first two
# is how you get either false rejections or false confidence.
OK = "ok"
CORRUPT = "corrupt"
UNVERIFIED = "unverified"

_SIDECAR_RE = re.compile(r"\.chunksums-(\d+)\.json$")

# Read granularity when hashing. Independent of the sidecar's chunk_bytes: we
# hash a chunk by streaming it in slices so a hypothetical 256MiB chunk size
# still can't cost 256MiB of RSS.
_READ_SLICE = 4 * 1024 * 1024


def sidecar_path(path: str, chunk_bytes: int) -> str:
    """The sidecar that would describe ``path`` at ``chunk_bytes`` granularity."""
    return f"{path}.chunksums-{chunk_bytes}.json"


def find_sidecar(path: str) -> str | None:
    """Locate a sidecar for ``path`` at ANY chunk size.

    Central picks the chunk size (32MiB today, clamped 4–256MiB), and a store
    populated by an older/newer central may carry a different one. Discovering
    the file rather than assuming 32MiB keeps verification working across that
    drift instead of silently degrading to UNVERIFIED. On a tie, prefer the
    largest chunk size = fewest hashes = fewest syscalls.
    """
    d = os.path.dirname(path) or "."
    base = os.path.basename(path)
    prefix = base + ".chunksums-"
    found: list[tuple[int, str]] = []
    try:
        for name in os.listdir(d):
            if not name.startswith(prefix):
                continue
            m = _SIDECAR_RE.search(name)
            if m:
                found.append((int(m.group(1)), os.path.join(d, name)))
    except OSError:
        return None
    if not found:
        return None
    found.sort(key=lambda t: t[0], reverse=True)
    return found[0][1]


def load_sidecar(side: str, src: str) -> dict | None:
    """Parse ``side`` and return it ONLY if it actually describes ``src``.

    The size+mtime key is the whole point: it is what distinguishes "these sums
    describe these bytes" from "these sums describe some older revision". A
    sidecar that fails the key is stale bookkeeping and is discarded (-> the
    caller reports UNVERIFIED), never used to condemn the file.
    """
    try:
        with open(side, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        sums = data.get("sums")
        chunk = int(data.get("chunk_bytes") or 0)
        if not isinstance(sums, list) or not sums or chunk <= 0:
            return None
        st = os.stat(src)
        if int(data.get("size", -1)) != st.st_size:
            return None
        if int(data.get("mtime", -1)) != int(st.st_mtime):
            return None
        return {"sums": [str(s) for s in sums], "chunk_bytes": chunk,
                "size": st.st_size}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _hash_chunk(fh, remaining: int) -> str:
    """SHA-256 of the next ``remaining`` bytes, streamed in bounded slices."""
    h = hashlib.sha256()
    while remaining > 0:
        buf = fh.read(min(_READ_SLICE, remaining))
        if not buf:
            break
        h.update(buf)
        remaining -= len(buf)
    return h.hexdigest()


def verify_file(candidate: str, spec: dict) -> tuple[str, str]:
    """Hash ``candidate`` against a loaded sidecar ``spec``.

    Returns ``(verdict, detail)``. Streams one chunk at a time and stops at the
    first mismatch, so rejecting a corrupt 10GB file costs one chunk of IO, not
    ten gigabytes.
    """
    sums = spec["sums"]
    chunk_bytes = spec["chunk_bytes"]
    expect_size = spec["size"]
    try:
        actual_size = os.path.getsize(candidate)
    except OSError as exc:
        return UNVERIFIED, f"cannot stat staged file: {exc}"

    # Size is a cheap pre-filter, NOT the check. The incident's file passed it.
    if actual_size != expect_size:
        return CORRUPT, (f"size {actual_size} != expected {expect_size} "
                         f"(incomplete transfer)")
    if len(sums) != -(-expect_size // chunk_bytes):
        # The sidecar contradicts itself; it cannot condemn the file.
        return UNVERIFIED, "sidecar chunk count inconsistent with its own size"

    try:
        with open(candidate, "rb") as fh:
            for i, want in enumerate(sums):
                start = i * chunk_bytes
                want_len = min(chunk_bytes, expect_size - start)
                got = _hash_chunk(fh, want_len)
                if got != want:
                    return CORRUPT, (
                        f"chunk {i}/{len(sums)} (bytes {start}-{start + want_len - 1}) "
                        f"sha256 {got[:16]}… != expected {want[:16]}… — the bytes "
                        f"are wrong even though the file is the CORRECT SIZE, i.e. "
                        f"a silently corrupted transfer, not a truncated one")
    except OSError as exc:
        return UNVERIFIED, f"read error while verifying: {exc}"
    return OK, f"{len(sums)} chunks verified against {os.path.basename(candidate)}"


def verify_against_source(candidate: str, source: str) -> tuple[str, str]:
    """Verify staged ``candidate`` against ``source``'s sidecar, if one exists.

    ``source`` is the shared-store original being copied; its sidecar lives
    beside it. Returns ``(OK|CORRUPT|UNVERIFIED, detail)``. Only CORRUPT is a
    proof of badness — callers must not treat UNVERIFIED as failure.
    """
    side = find_sidecar(source)
    if not side:
        return UNVERIFIED, "no chunksums sidecar beside the source"
    spec = load_sidecar(side, source)
    if not spec:
        return UNVERIFIED, (f"sidecar {os.path.basename(side)} is stale or "
                            f"unreadable (size/mtime no longer match the source)")
    return verify_file(candidate, spec)
