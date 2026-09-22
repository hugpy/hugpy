"""Portable RAM + GPU probes.

The Linux build read ``/proc/meminfo`` directly and shelled out to a bare
``nvidia-smi``. Neither is portable: ``/proc`` doesn't exist on Windows/macOS,
and ``nvidia-smi`` is ``nvidia-smi.exe`` on Windows (and absent entirely on
Apple silicon). These helpers degrade to ``None``/``[]`` instead of crashing, so
a CPU-only or non-NVIDIA host behaves like a GPU-less Linux box always did.

Probe order favours libraries that report the truth for *this* process:
``torch.cuda`` / ``pynvml`` when importable, then ``nvidia-smi`` via PATH, then
nothing.
"""
from __future__ import annotations

import subprocess
from typing import List, Optional

from hugpy_platform.platform_facade import IS_LINUX
from hugpy_platform.binaries import resolve_bin


# --------------------------------------------------------------------------- #
# RAM                                                                          #
# --------------------------------------------------------------------------- #
def free_ram_bytes() -> Optional[int]:
    """Available system RAM in bytes, or ``None`` if it can't be determined."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    if IS_LINUX:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------- #
# GPU                                                                          #
# --------------------------------------------------------------------------- #
def _safe_int(s) -> Optional[int]:
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return None


def detect_gpus() -> List[dict]:
    """Best-effort GPU inventory: ``[{index, name, memory_total, memory_free}]``.

    Memory values are bytes. Empty list on a CPU-only or non-NVIDIA host.
    """
    gpus = _detect_gpus_nvidia_smi()
    if gpus:
        return gpus
    return _detect_gpus_torch()


def _detect_gpus_nvidia_smi() -> List[dict]:
    smi = resolve_bin("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.check_output(
            [smi, "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return []
    gpus: List[dict] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx, name, mem_total, mem_free = parts[:4]
        tot, free = _safe_int(mem_total), _safe_int(mem_free)
        gpus.append({
            "index": _safe_int(idx),
            "name": name,
            "memory_total": tot * 1024 * 1024 if tot else None,   # MiB -> bytes
            "memory_free": free * 1024 * 1024 if free else None,
        })
    return gpus


def _detect_gpus_torch() -> List[dict]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        gpus: List[dict] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = None, getattr(props, "total_memory", None)
            try:
                free, total = torch.cuda.mem_get_info(i)
            except Exception:
                pass
            gpus.append({
                "index": i, "name": props.name,
                "memory_total": total, "memory_free": free,
            })
        return gpus
    except Exception:
        return []


def free_vram_bytes(main: int = 0) -> Optional[int]:
    """Free VRAM on GPU ``main`` in bytes, or ``None`` if no GPU / can't tell."""
    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info(main)
            return int(free)
    except Exception:
        pass
    smi = resolve_bin("nvidia-smi")
    if smi:
        try:
            out = subprocess.check_output(
                [smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits",
                 "-i", str(main)],
                stderr=subprocess.DEVNULL, timeout=10,
            ).decode("utf-8", "replace")
            mib = int(out.strip().splitlines()[0].strip())
            return mib * 1024 * 1024
        except Exception:
            return None
    return None


def total_vram_bytes(main: int = 0) -> Optional[int]:
    """Total (installed) VRAM on GPU ``main`` in bytes, or ``None`` if no GPU /
    can't tell. Mirrors ``free_vram_bytes`` probe-for-probe (torch.cuda.mem_get_info
    total, then ``nvidia-smi --query-gpu=memory.total``) so a device-wide ceiling
    (e.g. "keep the card at/under 90% full") is computed from the SAME truth the
    free-VRAM read comes from — both are ComfyUI-visible device totals, not managed
    bookkeeping. Degrades to ``None`` (not 0) so a caller can tell "unmeasurable"
    from "no headroom" and fail OPEN."""
    try:
        import torch

        if torch.cuda.is_available():
            _free, total = torch.cuda.mem_get_info(main)
            return int(total)
    except Exception:
        pass
    smi = resolve_bin("nvidia-smi")
    if smi:
        try:
            out = subprocess.check_output(
                [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits",
                 "-i", str(main)],
                stderr=subprocess.DEVNULL, timeout=10,
            ).decode("utf-8", "replace")
            mib = int(out.strip().splitlines()[0].strip())
            return mib * 1024 * 1024
        except Exception:
            return None
    return None
