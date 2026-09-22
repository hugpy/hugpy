"""Run one phone-brick orchestration as a tracked background job.

A *run* fans a single image across the chosen phones, collates their verdicts by
plurality consensus, and writes an annotated result — exactly what the CLI
``orchestrate`` verb does, but recorded in a run store so any gunicorn process
can report progress to the UI.

The fan-out is synchronous inside :class:`~hugpy.phone_brick.ChainOrchestrator`
(it returns only when every phone has answered), so we run it on a daemon thread
and flip the run record ``queued -> running -> done|error``. Per-phone streaming
progress is a later step; for now the UI polls the run until it is ``done``.

This module is deliberately web-agnostic: the caller injects ``out_dir`` and a
``run_store`` (anything with ``create``/``update`` methods), so the manager has
no dependency on the Flask app — that one-directional boundary is what keeps the
route -> manager import from cycling.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from typing import Any, Dict, List

from hugpy_fleet.phone_brick import ChainConfig, ChainOrchestrator, PhoneSpec, RunCancelled


def _phase_dict(phase) -> Dict[str, Any]:
    """A PhaseResult -> the compact shape the UI table renders."""
    return {
        "phone": phase.phone,
        "top_cls": phase.top_cls,
        "top_conf_pct": phase.top_conf_pct,
        "consensus": phase.consensus,      # AGR | DIS | NOD
        "detections": len(phase.detections),
        "timestamp": phase.timestamp,
    }


def run_phases_summary(result) -> List[Dict[str, Any]]:
    return [_phase_dict(p) for p in result.phases]


def _execute(run_id, image_path, phone_records, file_server,
             push_timeout, drain_timeout, out_dir, run_store) -> None:
    """Body of the background job: run the chain, persist the outcome.

    As the chain progresses, ``on_event`` accumulates per-phone verdicts into the
    run record's ``progress`` list so the UI (polling or via SSE) can fill rows
    in live; ``cancel_check`` polls the run's ``cancel_requested`` flag so a
    cancel from the UI aborts the run within a poll interval.
    """
    run_store.update(run_id, status="running")

    progress: List[Dict[str, Any]] = []

    def on_event(ev):
        etype = ev.get("type")
        if etype == "phase_start":
            run_store.update(run_id, current_phone=ev.get("phone"))
        elif etype == "phase":
            progress.append({
                "phone": ev["phone"],
                "top_cls": ev["top_cls"],
                "top_conf_pct": ev["top_conf_pct"],
                "detections": ev["detections"],
                "timestamp": ev.get("timestamp"),
                "consensus": None,   # resolved once every phone is in
            })
            run_store.update(run_id, progress=list(progress), current_phone=None)

    def cancel_check():
        return run_store.is_cancelled(run_id)

    try:
        phones = [
            PhoneSpec(
                name=p["name"],
                host=p["host"],
                port=int(p.get("port", 5002)),
                color_hex=p.get("color", "#58a6ff"),
            )
            for p in phone_records
        ]
        config = ChainConfig(
            phones=phones,
            file_server=file_server,
            push_timeout_s=push_timeout,
            drain_timeout_s=drain_timeout,
        )
        result = ChainOrchestrator(config).run(
            image_path, out_dir, on_event=on_event, cancel_check=cancel_check)
        run_store.update(
            run_id,
            status="done",
            phases=run_phases_summary(result),
            current_phone=None,
            output_rel=os.path.basename(result.output_path) if result.output_path else None,
            finished_at=time.time(),
        )
    except RunCancelled:
        run_store.update(run_id, status="cancelled", current_phone=None,
                         finished_at=time.time())
    except Exception as exc:  # noqa: BLE001 — record the failure for the UI
        run_store.update(
            run_id,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            current_phone=None,
            finished_at=time.time(),
        )
        traceback.print_exc()  # breadcrumb in the server log too


def start_run(*, image_path: str, phone_records: List[Dict[str, Any]],
              file_server: str, out_dir: str, run_store,
              push_timeout: float = 5.0, drain_timeout: float = 60.0) -> Dict[str, Any]:
    """Create a run record and kick off the fan-out on a daemon thread.

    ``out_dir`` is where the orchestrator seeds the image + writes the annotated
    result; ``run_store`` is the (injected) persistence the UI polls. Returns the
    freshly-created run dict (status ``queued``); the caller hands its ``id`` to
    the UI to poll.
    """
    if not phone_records:
        raise ValueError("no phones selected for the run")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")

    run = run_store.create(
        image=os.path.basename(image_path),
        phone_ids=[p["id"] for p in phone_records if p.get("id")],
    )
    threading.Thread(
        target=_execute,
        args=(run["id"], image_path, phone_records, file_server,
              push_timeout, drain_timeout, out_dir, run_store),
        daemon=True,
    ).start()
    return run
