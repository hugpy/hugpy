"""CONTENT VERIFICATION regression — the sd-turbo silent-corruption incident.

The incident (2026-07-15/16): worker ae's hot cache held a staged sd-turbo vae
that was 167335342 bytes — byte-for-byte the SAME SIZE as the good file in the
shared store — but whose bytes were corrupt. The promote gate only compared
SIZE, so it promoted, and 32h later diffusers reported "no file named
diffusion_pytorch_model.bin found" — an error naming a file that was never the
problem.

The load-bearing case here is `same size, wrong bytes` -> REJECTED. A size
check must not save it. Everything else guards the decisions around it:
absent/stale sidecars must NOT block (they'd break every unverifiable file),
and verification must be memory-bounded.

Runs like the other tests here: venv/bin/python tests/test_chunksum_verify.py
"""
import logging
# (module-level logging.disable removed: it leaked into every later test under pytest)

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib

cv = importlib.import_module("hugpy_engine.serve.chunksum_verify")
hc = importlib.import_module("hugpy_engine.serve.hot_cache")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


def test_chunksum_verify():
    """Converted from the script-style suite: every `check` is an assert."""


    CHUNK = 1 << 20   # 1MiB chunks in tests; real store uses 32MiB


    def _write_sidecar(path, chunk_bytes=CHUNK):
        """Write a sidecar in central's exact format (worker_routes._chunk_sums)."""
        sums = []
        with open(path, "rb") as fh:
            while True:
                buf = fh.read(chunk_bytes)
                if not buf:
                    break
                sums.append(hashlib.sha256(buf).hexdigest())
        st = os.stat(path)
        side = f"{path}.chunksums-{chunk_bytes}.json"
        with open(side, "w", encoding="utf-8") as fh:
            json.dump({"size": st.st_size, "mtime": int(st.st_mtime),
                       "chunk_bytes": chunk_bytes, "sums": sums}, fh)
        return side


    def _mkfile(path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return path


    tmp = tempfile.mkdtemp(prefix="chunksum-")

    # --------------------------------------------------------------------------- #
    # 1. Sidecar format: parse a REAL one from the shared store if present, so this
    #    test fails loudly if central's on-disk format ever drifts from our reader.
    # --------------------------------------------------------------------------- #
    _REAL = ("/mnt/llm_storage/models/transformers/stabilityai/sd-turbo/vae/"
             "diffusion_pytorch_model.fp16.safetensors")
    if os.path.isfile(_REAL) and cv.find_sidecar(_REAL):
        side = cv.find_sidecar(_REAL)
        with open(side, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        check("real store sidecar has the documented keys",
              {"size", "mtime", "chunk_bytes", "sums"} <= set(raw))
        check("real store sidecar chunk_bytes is 32MiB", raw["chunk_bytes"] == 33554432)
        check("real store sidecar sums are sha256-shaped",
              all(len(s) == 64 for s in raw["sums"]))
        spec = cv.load_sidecar(side, _REAL)
        check("real store sidecar loads (size+mtime still match)", spec is not None)
    else:
        print("  -- skip - real store sidecar not reachable from this host")

    # --------------------------------------------------------------------------- #
    # 2. A GOOD file verifies.
    # --------------------------------------------------------------------------- #
    good = _mkfile(os.path.join(tmp, "src", "model.safetensors"),
                   bytes(range(256)) * (CHUNK * 3 // 256 + 7))   # ~3.1 chunks
    _write_sidecar(good)
    verdict, detail = cv.verify_against_source(good, good)
    check("good file verifies OK", verdict == cv.OK)
    check("good file detail names chunk count", "chunks verified" in detail)

    # --------------------------------------------------------------------------- #
    # 3. THE INCIDENT: same size, wrong bytes -> REJECTED.
    #    A size check passes this. Only content catches it.
    # --------------------------------------------------------------------------- #
    src_bytes = open(good, "rb").read()
    corrupt = bytearray(src_bytes)
    corrupt[CHUNK + 500] ^= 0xFF          # flip ONE bit, deep inside chunk 1
    corrupt = _mkfile(os.path.join(tmp, "hot", "model.safetensors.part"), bytes(corrupt))

    check("corrupt copy is EXACTLY the same size as the source (the incident's "
          "signature — a size check would pass it)",
          os.path.getsize(corrupt) == os.path.getsize(good))

    verdict, detail = cv.verify_against_source(corrupt, good)
    check("same-size-but-corrupted file is REJECTED", verdict == cv.CORRUPT)
    check("rejection names the offending chunk", "chunk 1/" in detail)
    check("rejection explains it is corruption, not truncation",
          "CORRECT SIZE" in detail and "corrupted transfer" in detail)

    # a truncated file is also caught, and distinguished from corruption
    trunc = _mkfile(os.path.join(tmp, "hot", "trunc.part"), src_bytes[:-10])
    verdict, detail = cv.verify_against_source(trunc, good)
    check("truncated file rejected", verdict == cv.CORRUPT)
    check("truncated file reported as incomplete, not corrupt-bytes",
          "incomplete transfer" in detail)

    # --------------------------------------------------------------------------- #
    # 4. ABSENT sidecar must NOT block (it would break every unverifiable file).
    # --------------------------------------------------------------------------- #
    bare = _mkfile(os.path.join(tmp, "src2", "nosidecar.safetensors"), b"z" * 4096)
    verdict, detail = cv.verify_against_source(bare, bare)
    check("missing sidecar -> UNVERIFIED, not CORRUPT", verdict == cv.UNVERIFIED)
    check("missing sidecar is honest about why", "no chunksums sidecar" in detail)

    # --------------------------------------------------------------------------- #
    # 5. STALE sidecar (source edited after it was written) must NOT condemn.
    # --------------------------------------------------------------------------- #
    stale_src = _mkfile(os.path.join(tmp, "src3", "m.safetensors"), b"a" * (CHUNK * 2))
    _write_sidecar(stale_src)
    with open(stale_src, "wb") as fh:                 # rewrite -> size+mtime change
        fh.write(b"b" * (CHUNK * 3))
    os.utime(stale_src, (0, 0))                       # force a clearly-different mtime
    verdict, detail = cv.verify_against_source(stale_src, stale_src)
    check("stale sidecar -> UNVERIFIED (never a false CORRUPT)", verdict == cv.UNVERIFIED)
    check("stale sidecar says so", "stale" in detail)

    # --------------------------------------------------------------------------- #
    # 6. find_sidecar tolerates a non-32MiB chunk size (central clamps 4–256MiB).
    # --------------------------------------------------------------------------- #
    alt = _mkfile(os.path.join(tmp, "src4", "m.bin"), b"q" * 9000)
    _write_sidecar(alt, chunk_bytes=4096)
    check("sidecar found at a non-default chunk size", cv.find_sidecar(alt) is not None)
    check("non-default chunk size verifies", cv.verify_against_source(alt, alt)[0] == cv.OK)

    # --------------------------------------------------------------------------- #
    # 7. Memory-bounded: verifying a large file must not slurp it into RAM.
    #    Assert on the read granularity rather than RSS (portable + deterministic).
    # --------------------------------------------------------------------------- #
    big = _mkfile(os.path.join(tmp, "src5", "big.safetensors"), b"m" * (CHUNK * 4))
    _write_sidecar(big)
    _reads = []
    _real_open = open
    class _SpyFile:
        def __init__(self, fh): self._fh = fh
        def read(self, n=-1):
            _reads.append(n)
            return self._fh.read(n)
        def __enter__(self): return self
        def __exit__(self, *a): self._fh.close(); return False
    import builtins
    def _spy_open(path, mode="r", *a, **k):
        fh = _real_open(path, mode, *a, **k)
        if "b" in mode and str(path).endswith(".safetensors"):
            return _SpyFile(fh)
        return fh
    builtins.open = _spy_open
    try:
        verdict, _ = cv.verify_against_source(big, big)
    finally:
        builtins.open = _real_open
    check("large file verifies OK", verdict == cv.OK)
    check("verification never issues an unbounded read()",
          _reads and all(n > 0 for n in _reads))
    check("verification read slice stays bounded (<= 4MiB)",
          max(_reads) <= cv._READ_SLICE)

    # --------------------------------------------------------------------------- #
    # 8. THE PROMOTE SEAM: hot_cache._verify_staged is what actually gates promotion.
    # --------------------------------------------------------------------------- #
    check("promote seam: good staged copy -> OK",
          hc._verify_staged(good, good)[0] == cv.OK)
    check("promote seam: same-size corrupt staged copy -> CORRUPT",
          hc._verify_staged(good, corrupt)[0] == cv.CORRUPT)
    check("promote seam: no sidecar -> UNVERIFIED (does not block)",
          hc._verify_staged(bare, bare)[0] == cv.UNVERIFIED)

    # a verifier bug must degrade to UNVERIFIED, never break the hot tier
    _boom = cv.verify_against_source
    cv.verify_against_source = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        check("promote seam: verifier exception degrades to UNVERIFIED",
              hc._verify_staged(good, good)[0] == cv.UNVERIFIED)
    finally:
        cv.verify_against_source = _boom

    # --------------------------------------------------------------------------- #
    # 9. Bookkeeping is not content: sidecars/.part/.state.json must never be
    #    promoted to the hot drive (they'd spend the weight budget on metadata and
    #    carry a crashed pull's wedge onto the fast drive).
    # --------------------------------------------------------------------------- #
    check("sidecar is bookkeeping", hc._is_bookkeeping("m.safetensors.chunksums-33554432.json"))
    check(".part is bookkeeping", hc._is_bookkeeping("m.safetensors.part"))
    check(".state.json is bookkeeping", hc._is_bookkeeping("m.safetensors.part.state.json"))
    check("a real weight is NOT bookkeeping", not hc._is_bookkeeping("model.safetensors"))
    check("config.json is NOT bookkeeping", not hc._is_bookkeeping("config.json"))

    _bk = os.path.join(tmp, "bk", "fam/txt/acme/m1")
    _mkfile(os.path.join(_bk, "model.safetensors"), b"G" * 2048)
    _mkfile(os.path.join(_bk, "config.json"), b"{}")
    _write_sidecar(os.path.join(_bk, "model.safetensors"), chunk_bytes=1024)
    _mkfile(os.path.join(_bk, "stale.safetensors.part"), b"junk")
    _names = {os.path.basename(p) for p in hc._file_set(_bk)}
    check("file_set carries the weight + config", _names == {"model.safetensors", "config.json"})
    check("file_set excludes the sidecar and the .part",
          not any(".chunksums-" in n or n.endswith(".part") for n in _names))

    print(f"\n{ok} checks passed")
