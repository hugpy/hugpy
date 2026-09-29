"""Host-wide GPU process inventory with optional Hugpy control attribution."""
from __future__ import annotations

import os
import subprocess


def _csv(*args):
    try:
        return subprocess.check_output(["nvidia-smi", *args], text=True, timeout=5).splitlines()
    except (OSError, subprocess.SubprocessError):
        return []


def gpu_processes():
    """Every compute process on every NVIDIA card, whether Hugpy owns it or not."""
    devices = {}
    for line in _csv("--query-gpu=index,uuid", "--format=csv,noheader"):
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 2 and fields[0].isdigit():
            devices[fields[1]] = int(fields[0])
    rows = {}
    for line in _csv("--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory",
                     "--format=csv,noheader,nounits"):
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 4 or not fields[0].isdigit():
            continue
        pid = int(fields[0])
        row = rows.setdefault(pid, {"pid": pid, "process": os.path.basename(fields[1]),
                                    "gpu_indices": [], "vram_mib": 0})
        gpu = devices.get(fields[2])
        if gpu is not None and gpu not in row["gpu_indices"]:
            row["gpu_indices"].append(gpu)
        if fields[3].isdigit():
            row["vram_mib"] += int(fields[3])
    return list(rows.values())


def inventory(externals, owned, processes):
    """Join raw GPU rows to known services; leave unmatched PIDs visible."""
    processes = {int(row["pid"]): dict(row) for row in processes}
    rows = []
    claimed = set()
    for rec in externals:
        pids = {int(pid) for pid in (rec.get("pids") or [rec.get("pid")]) if pid}
        matching = [processes[pid] for pid in pids if pid in processes]
        claimed.update(pids)
        rows.append({"kind": "external", "model_key": rec.get("model_key"),
                     "service": (rec.get("model_key") or "external").split(":", 1)[0],
                     "pids": sorted(pids), "gpu_indices": sorted({g for row in matching for g in row["gpu_indices"]})
                     or rec.get("gpu_indices") or [],
                     "vram_mib": sum(row["vram_mib"] for row in matching),
                     "state": rec.get("state"), "evictable": rec.get("evictable", True),
                     "immutable": bool(rec.get("immutable")), "api_available": bool(rec.get("api_url")),
                     "activity": rec.get("activity") or {}, "note": rec.get("note")})
    for rec in owned:
        pid = rec.get("pid")
        if not pid or int(pid) in claimed:
            continue
        pid = int(pid)
        process = processes.get(pid)
        if process:
            claimed.add(pid)
            rows.append({"kind": "hugpy" if rec.get("model_key") else "infrastructure",
                         "model_key": rec.get("model_key"),
                         "service": rec.get("host_mode"), "pids": [pid],
                         "gpu_indices": process["gpu_indices"], "vram_mib": process["vram_mib"],
                         "state": "running", "evictable": None, "immutable": False,
                         "api_available": bool(rec.get("model_key")), "activity": {},
                         "note": "Hugpy managed" if rec.get("model_key") else "Hugpy infrastructure"})
    for pid, process in processes.items():
        if pid not in claimed:
            rows.append({"kind": "observed", "model_key": None,
                         "service": process["process"], "pids": [pid],
                         "gpu_indices": process["gpu_indices"], "vram_mib": process["vram_mib"],
                         "state": "running", "evictable": None, "immutable": True,
                         "api_available": False, "activity": {},
                         "note": "GPU process observed; no safe control adapter registered"})
    return sorted(rows, key=lambda row: (min(row["gpu_indices"] or [999]), row["service"], row["pids"]))
