"""Best-effort manifest and prompt sidecars for rendered media."""

from __future__ import annotations

import dataclasses
import json
import os
import time


def write_spec_sidecar(out_dir: str, job_id: str, kind: str, spec_obj) -> None:
    """Write the render spec next to its media without failing the render."""
    try:
        d = dataclasses.asdict(spec_obj) if dataclasses.is_dataclass(spec_obj) else (
            spec_obj if isinstance(spec_obj, dict) else {"repr": repr(spec_obj)})
        prompts = [p.get("text") for p in (d.get("parts") or [])
                   if isinstance(p, dict) and p.get("kind") == "text" and p.get("text")]
        prompts += [g.get("prompt") for g in (d.get("goals") or [])
                    if isinstance(g, dict) and g.get("prompt")]
        if d.get("prompt"):
            prompts.append(d["prompt"])
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump({"job_id": job_id, "kind": kind, "created": time.time(),
                       "spec": d}, fh, indent=1, default=str)
        with open(os.path.join(out_dir, "prompt.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(p for p in prompts if p) + "\n")
    except Exception:  # noqa: BLE001 — sidecars are documentation, never load-bearing
        pass
