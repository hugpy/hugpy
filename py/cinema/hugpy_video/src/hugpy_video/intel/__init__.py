"""hugpy Video Intelligence — headless backbone (Phases 1 & 2).

Additive package. Nothing here restarts a service or touches the serve/worker
control plane. It provides the frozen schemas (MediaRef / CropSpec / JobSpec /
JobResult), the ingest metadata resolver, the pure ffmpeg crop runner, and a
durable sqlite-backed job bus.

Build order matches hugpy_video_intelligence_map.md §11:
    MediaRef + ingest  ->  CropSpec + (ffmpeg, crop) runner + crop JobSpec  ->  bus.
"""
