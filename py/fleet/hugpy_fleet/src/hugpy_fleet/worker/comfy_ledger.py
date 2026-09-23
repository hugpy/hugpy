"""The ComfyUI resident ledger — what this worker has asked comfy to load.

ComfyUI is an EXTERNAL process: nvidia-smi shows one lump of VRAM under its
PID and nothing says which checkpoints are inside it. hugpy, however, is the
one that dispatched every checkpoint comfy holds (``ComfyRunner`` names the
file in the graph), so the worker can keep the ledger comfy itself does not
expose:

  * ``note_dispatch``  — a graph naming ``filename`` for ``model_key`` was (or is
                         about to be) submitted; comfy will hold it from here.
  * ``note_freed``     — comfy accepted ``POST /free``; it holds nothing now.
  * ``residents``      — the checkpoints comfy holds, by model key, with the
                         file size each one weighs.

Sizes come from the checkpoint FILE (``checkpoint_size_bytes``): for the fp16
safetensors comfy serves, weights on disk ≈ weights in VRAM, and it is the one
figure known BEFORE the load. That is what makes a per-checkpoint ``need``
possible (``predicted_need_bytes``) instead of the one-size target the
headroom path used to clear for every gen.

Pure module: no agent imports, no HTTP, no nvidia-smi. The worker binds it
(``agent._comfy_ledger``) and feeds it measured process VRAM when it reports.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Iterable, Optional

_GIB = 2**30

#: Working room a comfy gen needs ON TOP of its checkpoint weights (activations,
#: VAE decode, CLIP, fragmentation). Recon on ae: a ~2 GiB SD1.5 checkpoint drove
#: the comfy process from 5.5 to 6.5 GiB during a 512x512 gen; SDXL at 1024
#: needs more. 2 GiB is that observation with margin; env-tunable per box.
_DEFAULT_GEN_CUSHION_GIB = 2.0


def gen_cushion_bytes() -> int:
    """``HUGPY_COMFY_GEN_CUSHION_GIB`` (default 2.0 GiB); garbage/negative ->
    the default, same convention as the other GIB knobs."""
    raw = os.environ.get("HUGPY_COMFY_GEN_CUSHION_GIB")
    if raw is not None and str(raw).strip():
        try:
            val = float(raw)
            if val >= 0:
                return int(val * _GIB)
        except ValueError:
            pass
    return int(_DEFAULT_GEN_CUSHION_GIB * _GIB)


def checkpoint_dirs(extra: Optional[Iterable[str]] = None) -> list[str]:
    """Directories a checkpoint name may resolve under, first match wins:
    ``COMFY_CHECKPOINTS_DIR`` (where hugpy symlinks checkpoints for comfy —
    provision.ensure_comfy_checkpoint), then every entry of
    ``COMFY_CHECKPOINT_DIRS`` (os.pathsep-separated: the roots comfy's own
    extra_model_paths.yaml scans on this box), then any ``extra`` the caller
    knows (e.g. the model store's ``checkpoints/``). Empty entries dropped,
    duplicates collapsed, order kept."""
    out: list[str] = []
    seen: set[str] = set()

    def _add(p: Optional[str]) -> None:
        p = (p or "").strip()
        if not p:
            return
        p = os.path.expanduser(p)
        if p not in seen:
            seen.add(p)
            out.append(p)

    _add(os.environ.get("COMFY_CHECKPOINTS_DIR"))
    for p in (os.environ.get("COMFY_CHECKPOINT_DIRS") or "").split(os.pathsep):
        _add(p)
    for p in (extra or ()):
        _add(p)
    return out


def checkpoint_size_bytes(filename: Optional[str],
                          dirs: Optional[Iterable[str]] = None) -> Optional[int]:
    """Size in bytes of the checkpoint comfy will load for ``filename``, or
    ``None`` when it cannot be found (unknown need — the caller falls back).

    ``filename`` is what the graph's ``ckpt_name`` carries: a bare name or a
    path relative to one of comfy's checkpoint roots. Looked up as given under
    each dir first; then by basename anywhere below each dir, because comfy's
    folder scan is recursive and a checkpoint moved into a sub-folder keeps its
    name. Symlinks are followed — hugpy places symlinks, the bytes live at the
    target — and a dangling link is "not found", never 0."""
    if not filename:
        return None
    name = str(filename).strip().replace("\\", "/")
    if not name:
        return None
    roots = checkpoint_dirs(dirs)
    base = os.path.basename(name)
    for root in roots:
        cand = os.path.join(root, name)
        size = _file_size(cand)
        if size is not None:
            return size
    for root in roots:
        try:
            for cur, _dirs, files in os.walk(root, followlinks=True):
                if base in files:
                    size = _file_size(os.path.join(cur, base))
                    if size is not None:
                        return size
        except OSError:
            continue
    return None


def _file_size(path: str) -> Optional[int]:
    try:
        if os.path.isfile(path):          # follows symlinks; False when dangling
            return int(os.stat(path).st_size)
    except OSError:
        pass
    return None


def predicted_need_bytes(checkpoint_bytes: Optional[int], held: bool,
                         cushion: Optional[int] = None) -> Optional[int]:
    """Free VRAM a comfy gen of this checkpoint needs BEFORE it starts.

    * checkpoint already resident in comfy (``held``) -> the gen cushion only:
      the weights are on the card, only working room must be free.
    * not resident, size known -> weights + cushion.
    * size unknown -> ``None`` (the caller decides the fallback; never a guess
      dressed as a measurement)."""
    c = gen_cushion_bytes() if cushion is None else max(0, int(cushion))
    if held:
        return c
    if checkpoint_bytes is None:
        return None
    return max(0, int(checkpoint_bytes)) + c


class ComfyLedger:
    """Insertion-ordered (oldest dispatch first) map of what comfy holds.

    Thread-safe: the headroom hook runs on request threads while the heartbeat
    and the eviction planner read it. Re-dispatching a held key refreshes its
    ``last_used`` and moves it to the end (LRU order for the planner)."""

    def __init__(self, clock=None) -> None:
        self._rows: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()
        self._clock = clock or time.time
        self.freed_at: Optional[float] = None

    # -- writes --------------------------------------------------------------
    def note_dispatch(self, model_key: str, filename: Optional[str],
                      checkpoint_bytes: Optional[int]) -> None:
        if not model_key:
            return
        now = self._clock()
        with self._lock:
            row = self._rows.pop(model_key, None)
            if row is None:
                row = {"model_key": model_key, "loaded_at": now}
            row["filename"] = filename
            row["bytes"] = int(checkpoint_bytes) if checkpoint_bytes else None
            row["last_used"] = now
            self._rows[model_key] = row

    def note_freed(self) -> list[str]:
        """comfy released its resident set. Returns the keys it held."""
        with self._lock:
            keys = list(self._rows)
            self._rows.clear()
            self.freed_at = self._clock()
        return keys

    def forget(self, model_key: str) -> None:
        with self._lock:
            self._rows.pop(model_key, None)

    # -- reads ---------------------------------------------------------------
    def holds(self, model_key: str) -> bool:
        with self._lock:
            return model_key in self._rows

    def resident_keys(self) -> list[str]:
        with self._lock:
            return list(self._rows)

    def last_used(self, model_key: str) -> Optional[float]:
        with self._lock:
            row = self._rows.get(model_key)
            return None if row is None else row.get("last_used")

    def residents(self, process_vram_bytes: Optional[int] = None) -> list[dict]:
        """Rows ``{model_key, filename, bytes, loaded_at, last_used}`` oldest
        first. ``bytes`` is the checkpoint's file size, CAPPED at the measured
        comfy process VRAM when given: the ledger says what was dispatched, the
        device says how much is really there, and a row must never claim more
        than the whole process holds."""
        with self._lock:
            rows = [dict(r) for r in self._rows.values()]
        if process_vram_bytes is not None:
            cap = max(0, int(process_vram_bytes))
            for r in rows:
                if r.get("bytes") is not None:
                    r["bytes"] = min(int(r["bytes"]), cap)
        return rows

    def resident_bytes(self) -> int:
        with self._lock:
            return sum(int(r["bytes"]) for r in self._rows.values()
                       if r.get("bytes"))

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)
