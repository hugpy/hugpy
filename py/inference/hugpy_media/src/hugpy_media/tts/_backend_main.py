"""The chatterbox synthesis child — run by the PROFILE venv's interpreter.

WHY A CHILD PROCESS. ``chatterbox-tts`` pins ``torch==2.6.0`` /
``transformers==5.2.0``; the worker agent's venv runs torch 2.13 / transformers
5.15 and serves every other model on the box. Two versions of one package cannot
share a process — that physics note is the fleet's own env-PROFILE doctrine
(``managers/serve/profiles.py``: "isolation happens at the PROCESS seam"). So
this file is spawned BY PATH from the profile venv, and it is the only code in
this package that ever runs under those pins.

Consequences, all deliberate:

  * STDLIB ONLY at the top, and NO ``abstract_hugpy_dev`` import anywhere — the
    package is not installed in the profile venv and putting it on PYTHONPATH
    would drag the agent's torch back in front of the profile's (PYTHONPATH is
    searched BEFORE site-packages, which is exactly the collision this avoids).
  * The k98 adapter is loaded BY FILE PATH (``spec_from_file_location``), so
    synthesis runs the SAME code the catalog probes and the media bus dispatches
    — one implementation, two interpreters, no second copy of the logic.
  * Protocol is one JSON job on stdin -> one JSON line on stdout after
    ``RESULT_SENTINEL``. Anything the backend prints (and it prints plenty) is
    ignorable noise before that marker.

Also reports MEASURED peak VRAM for this process (``torch.cuda.max_memory_*``),
which is where the catalog's ResourceHints number comes from — measured, not
estimated.
"""
import json
import os
import sys
import time

RESULT_SENTINEL = "@@TTS_RESULT@@"


def _load_adapter(module_path):
    """The k98 runner module, loaded from an explicit file path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tts_chatterbox_child", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load adapter from {module_path!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tts_chatterbox_child"] = module
    spec.loader.exec_module(module)
    return module


def _vram() -> dict:
    """Peak VRAM this process actually used, plus the card's own totals.
    Empty when there is no CUDA — an unmeasurable number is reported as absent,
    never as zero."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info()
        return {
            "vram_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "vram_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "vram_device_free_bytes": int(free),
            "vram_device_total_bytes": int(total),
            "torch_version": torch.__version__,
        }
    except Exception:  # noqa: BLE001 — a measurement must never fail the work
        return {}


def main() -> int:
    job = json.loads(sys.stdin.read() or "{}")
    module_path = job["module_path"]
    out_dir = job["out_dir"]
    spec_kwargs = dict(job.get("spec") or {})

    started = time.monotonic()
    try:
        adapter = _load_adapter(module_path)
        spec = adapter.make_tts(**spec_kwargs)
        load_started = time.monotonic()
        manifest = adapter.synthesize(spec, out_dir)
        payload = {
            "ok": True,
            "manifest": manifest,
            "elapsed_s": round(time.monotonic() - started, 3),
            "synth_s": round(time.monotonic() - load_started, 3),
            **_vram(),
        }
    except Exception as exc:  # noqa: BLE001 — errors cross this boundary as DATA
        code = getattr(exc, "code", None) or "tts_error"
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "error_code": str(code),
            "elapsed_s": round(time.monotonic() - started, 3),
            **_vram(),
        }
    sys.stdout.write("\n" + RESULT_SENTINEL + json.dumps(payload) + "\n")
    sys.stdout.flush()
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
