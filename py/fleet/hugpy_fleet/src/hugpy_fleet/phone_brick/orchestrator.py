"""Chain orchestrator — fan one image across a chain of phone workers.

This is the standalone, coordinator-free runner consolidated from the field
``_orchestrator-*-RUNS-ON-COLOSSUS.py`` scripts. For one image it:

    1. seeds the image where the file server can serve it,
    2. pushes a ``yolo <url>`` task to each phone in the chain and drains the
       result,
    3. parses each worker's text output into :class:`Detection` objects,
    4. computes a plurality :mod:`consensus` across the phones,
    5. writes an annotated copy with every phone's boxes drawn in its colour,
       under a result filename that encodes each phone's verdict.

It depends only on the package's own modules plus the stdlib (cv2 is optional,
used only for drawing), so it runs on any control box without the inference
stack the phones need.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from hugpy_fleet.phone_brick.client import WorkerClient
from hugpy_fleet.phone_brick.consensus import NODET, plurality_consensus
from hugpy_fleet.phone_brick.protocol import parse_detections
from hugpy_fleet.phone_brick.rendering import draw_detections
from hugpy_fleet.phone_brick.schemas import (
    ChainConfig,
    ChainResult,
    Detection,
    PhaseResult,
    RunCancelled,
)


def _top_detection(detections: list[Detection]) -> tuple[str, float]:
    """The highest-confidence detection's class + confidence, or ``('nodet', 0)``."""
    if not detections:
        return (NODET, 0.0)
    top = max(detections, key=lambda d: d.conf)
    return (top.cls, top.conf)


class ChainOrchestrator:
    """Run a chain of phones over images, collating their verdicts."""

    def __init__(self, config: ChainConfig, url_prefix: str = ""):
        self.config = config
        # Appended to ``file_server`` to form each image URL. Defaults to "" so
        # ``file_server`` can already point at the served directory.
        self.url_prefix = url_prefix

    def _image_url(self, seed_name: str) -> str:
        base = self.config.file_server.rstrip("/") + "/"
        return f"{base}{self.url_prefix}{seed_name}"

    def run(self, image_path: str | Path, output_dir: str | Path,
            *, on_event=None, cancel_check=None) -> ChainResult:
        """Fan the image across the chain.

        ``on_event``, if given, is called with small dict events as work
        progresses (``phase_start`` before a phone is asked, ``phase`` when it
        answers with its top detection, ``consensus`` once all phones are in) —
        used to drive a live UI. ``cancel_check`` is polled between phones and
        while waiting on one; when it returns truthy, :class:`RunCancelled` is
        raised. Both default to off, so existing callers are unaffected.
        """
        def emit(event: dict) -> None:
            if on_event:
                try:
                    on_event(event)
                except Exception:  # a UI callback must never break the run
                    pass

        src = Path(image_path)
        if not src.is_file():
            raise FileNotFoundError(f"image not found: {src}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Seed the source where the file server can serve it.
        seed_name = src.name
        seed_path = out_dir / seed_name
        if not seed_path.exists():
            shutil.copy2(src, seed_path)
        url = self._image_url(seed_name)

        # Stage 1: push to each phone in order, collect detections.
        collected: list[tuple] = []  # (phone, top_cls, top_conf, detections, ts)
        for phone in self.config.phones:
            if cancel_check and cancel_check():
                raise RunCancelled(f"cancelled before phone {phone.name}")
            emit({"type": "phase_start", "phone": phone.name})
            client = WorkerClient(phone)
            job_id = client.push(f"yolo {url}", timeout=self.config.push_timeout_s)
            job = client.drain_until(
                job_id,
                drain_timeout=self.config.drain_timeout_s,
                poll_s=self.config.drain_poll_s,
                request_timeout=self.config.push_timeout_s,
                cancel_check=cancel_check,
            )
            if job is None:
                raise TimeoutError(
                    f"phone {phone.name} timed out after "
                    f"{self.config.drain_timeout_s}s")
            detections = parse_detections(job.get("result"))
            top_cls, top_conf = _top_detection(detections)
            ts = int(time.time())
            collected.append((phone, top_cls, top_conf, detections, ts))
            # Consensus isn't known until every phone has reported, so this
            # per-phone event carries only the phone's own verdict.
            emit({"type": "phase", "phone": phone.name, "top_cls": top_cls,
                  "top_conf_pct": int(round(top_conf * 100)),
                  "detections": len(detections), "timestamp": ts})

        # Stage 2: plurality consensus across phones.
        flags = plurality_consensus({p.name: cls for p, cls, _, _, _ in collected})
        emit({"type": "consensus", "flags": flags})

        # Stage 3: build the result filename encoding each phone's verdict.
        stem, ext = src.stem, src.suffix.lstrip(".")
        segments = "".join(
            f"__{phone.name}__{cls}_{int(round(conf * 100))}_{flags[phone.name]}__{ts}"
            for phone, cls, conf, _, ts in collected
        )
        final_path = out_dir / f"{stem}{segments}.{ext}"

        # Stage 4: annotated copy with every phone's boxes in its colour.
        shutil.copy2(src, final_path)
        for phone, _, _, detections, _ in collected:
            if detections:
                draw_detections(str(final_path), phone.color_hex, detections)

        phases = [
            PhaseResult(
                phone=phone.name,
                top_cls=cls,
                top_conf=conf,
                detections=detections,
                consensus=flags[phone.name],
                timestamp=ts,
            )
            for phone, cls, conf, detections, ts in collected
        ]
        return ChainResult(image=src.name, phases=phases, output_path=str(final_path))
