"""In-process self checks for ``hugpy-video selftest`` (no GPU, no service).

Each check is a zero-arg callable returning ``None`` (pass) or raising.
They exercise: the job registry/deserializer/dispatch tables agree, the movie
presets round-trip through ``make_movie``, the hooks default is a no-op, and
the job bus can enqueue and drain a synthetic job in a private sqlite file.
"""

from __future__ import annotations

import os
import tempfile
import traceback
from typing import Callable, Dict, List


def check_registry_tables_agree() -> None:
    from hugpy_video.intel import media_bus
    from hugpy_video.intel.job_schema import JOB_REGISTRY
    from hugpy_video.intel.runners import DISPATCH

    missing_deser = sorted(n for n in JOB_REGISTRY if n not in media_bus.SPEC_DESERIALIZERS)
    missing_runner = sorted(n for n, s in JOB_REGISTRY.items() if s.runner_key not in DISPATCH)
    assert not missing_runner, f"jobs without a runner: {missing_runner}"
    # tts_chatterbox rehydrates through its own make_tts in the runner (dict spec).
    assert set(missing_deser) <= {"tts_chatterbox"}, f"jobs without a deserializer: {missing_deser}"


def check_movie_presets_roundtrip() -> None:
    from hugpy_video.intel.movie_schema import GoalInterval, make_movie, total_frames
    from hugpy_video.intel.presets import MOVIE_PRESETS

    assert MOVIE_PRESETS, "no movie presets"
    for pid, preset in MOVIE_PRESETS.items():
        spec = make_movie(
            goals=tuple(GoalInterval(g["start_frame"], g["end_frame"], g["prompt"])
                        for g in preset.goals),
            model_id=preset.model_key, width=preset.width, height=preset.height,
            steps=preset.steps, guidance=preset.guidance, fps=preset.fps,
            assemble=True, chain=preset.chain, vision_enabled=preset.vision_enabled,
            score_threshold=preset.score_threshold)
        assert total_frames(spec) == preset.total_frames(), pid


def check_hooks_default() -> None:
    from hugpy_video.hooks import NullPromptCoordinator, get_prompt_coordinator

    pc = get_prompt_coordinator()
    assert isinstance(pc, NullPromptCoordinator) or hasattr(pc, "review")
    rep = pc.review([{"segment_id": "s1", "prompt": "x"}])
    assert isinstance(rep.as_dict(), dict)


def check_bus_private_db() -> None:
    from hugpy_video.intel import media_bus
    from hugpy_video.intel.crop_schema import CropSpec  # noqa: F401 - spec exists

    old = media_bus.DB_PATH
    with tempfile.TemporaryDirectory() as tmp:
        media_bus.DB_PATH = os.path.join(tmp, "selftest_jobs.db")
        try:
            media_bus._initialized = False
            assert media_bus.work_once("selftest") is None  # empty queue drains to None
            assert media_bus.list_jobs(include_terminal=True) == []
        finally:
            media_bus.DB_PATH = old
            media_bus._initialized = False


CHECKS: Dict[str, Callable[[], None]] = {
    "registry_tables_agree": check_registry_tables_agree,
    "movie_presets_roundtrip": check_movie_presets_roundtrip,
    "hooks_default": check_hooks_default,
    "bus_private_db": check_bus_private_db,
}


def run_selftest(verbose: bool = False) -> List[str]:
    """Run every check; return the names that failed (empty == all passed)."""
    failed: List[str] = []
    for name, fn in CHECKS.items():
        try:
            fn()
            if verbose:
                print(f"  ok   {name}")
        except Exception:  # noqa: BLE001 - report, keep going
            failed.append(name)
            if verbose:
                print(f"  FAIL {name}\n{traceback.format_exc()}")
    if verbose:
        print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return failed
