"""Engine task plugin: ``hugpy_video`` registers its runners with ``hugpy_engine``.

Entry point ``[project.entry-points."hugpy_engine.tasks"] video =
"hugpy_video.plugin:register"``; ``hugpy_engine.tasks.load_entry_points()``
calls :func:`register` (the server and worker entry points do the same
explicitly). Registration is IMPORT-LIGHT: nothing here imports torch,
diffusers, ffmpeg bindings or the studio spine — the runner class is
resolved lazily the first time the engine instantiates it.

Tasks served (both by the studio spine, ``video_gen.StudioVideoRunner``):

    text-to-video    capability t2v (prompt)
    image-to-video   capability i2v (start_image / source_video)
"""

from __future__ import annotations

from typing import Sequence

from hugpy_video.plugin_builders import build_videogen_request

__all__ = ["TASKS", "FRAMEWORKS", "register", "unregister", "studio_video_runner"]

TASKS: Sequence[str] = ("text-to-video", "image-to-video")
#: Registry rows for the Wan/VACE/LTX zoo carry framework "transformers"
#: (where their bytes live); they are SERVED by the studio spine.
FRAMEWORKS: Sequence[str] = ("transformers",)
EXTRA = "studio"


def studio_video_runner():
    """Zero-arg runner factory (``hugpy_engine.tasks.TaskSpec.runner_class``
    calls it once and memoises the class). The import of the runner module
    happens here, not at registration, so ``register()`` stays import-light."""
    from hugpy_video.video_gen.video_gen_runner import StudioVideoRunner

    return StudioVideoRunner


def register() -> list:
    """Register every video task with the engine. Idempotent. Returns the specs."""
    from hugpy_engine.tasks import register_task

    return [
        register_task(
            task,
            runner=studio_video_runner,
            build_request=build_videogen_request,
            frameworks=FRAMEWORKS,
            extra=EXTRA,
            source="hugpy_video",
        )
        for task in TASKS
    ]


def unregister() -> None:
    from hugpy_engine.tasks import unregister_task

    for task in TASKS:
        unregister_task(task)
