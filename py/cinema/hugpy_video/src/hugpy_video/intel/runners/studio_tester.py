"""Pure ``(studio, tester)`` runner — the media bus's seam to the studio TESTER.

``run_studio_tester(spec, job_id) -> JobResult``. A THIN adapter: it lifts the
rehydrated ``StudioTesterSpec`` into ``studio.tester.run_tester`` (the cross-model
sweep) and returns its ``JobResult``. The heavy work (the plane / the studio
spine / the battery recorder) is reached lazily INSIDE ``run_tester``, so this
module's top level stays dependency-light and this import can never break app
boot — mirroring ``runners/studio_i2v.py``.

QUEUE: registered on the "media" (CPU) queue, NOT "gpu". The tester is a CPU
ORCHESTRATOR — each inner generation manages its own GPU (the image path routes
through the inference plane / a GPU worker; the video path delegates through the
studio spine). So the sweep itself takes NO GPU reservation; it fans work out to
the paths that do.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_studio_tester(spec, job_id: str):
    # Lazy: keep torch/plane/studio-spine imports out of app-boot's import graph.
    from hugpy_video.intel.studio.tester import run_tester_from_spec
    return run_tester_from_spec(spec, job_id)
