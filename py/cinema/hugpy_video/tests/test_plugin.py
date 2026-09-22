"""hugpy_video.plugin: engine task registration is import-light and lazy."""
from __future__ import annotations

import subprocess
import sys


def test_register_does_not_import_heavy_stacks():
    code = (
        "import sys; sys.modules['abstract_hugpy_dev'] = None\n"
        "from hugpy_video import plugin; specs = plugin.register()\n"
        "heavy = [m for m in sys.modules if m.split('.')[0] in "
        "('torch', 'diffusers', 'cv2', 'numpy', 'transformers')]\n"
        "video = [m for m in sys.modules if m.startswith('hugpy_video.intel')]\n"
        "print(len(specs), heavy, video)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    n, heavy, video = proc.stdout.strip().split(" ", 2)
    assert n == "2"
    assert heavy == "[]", heavy
    assert video == "[]", video          # the studio spine (intel/) is resolved lazily, never at register()


def test_registration_round_trip():
    from hugpy_engine import tasks as T
    from hugpy_video import plugin
    from hugpy_video.video_gen import StudioVideoRunner

    plugin.register()
    try:
        for task in plugin.TASKS:
            spec = T.task_spec(task, "transformers")
            assert spec is not None and spec.source == "hugpy_video" and spec.extra == "studio"
            assert T.runner_for_task(task, "transformers") is StudioVideoRunner
            req = spec.build_request({"prompt": "p", "seed": 3}, "Wan2.1-T2V-1.3B")
            assert req.model_key == "Wan2.1-T2V-1.3B" and req.seed == 3
    finally:
        plugin.unregister()
    assert T.task_spec("text-to-video") is None
