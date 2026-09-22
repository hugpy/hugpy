"""p6 reservation TEMPLATES — peak resolution + measured-overlay (measured wins).

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_reservation_templates.py -q
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Point the overlay at a scratch file so the built-in estimates stand unless a
# test writes one (never the real /mnt store — p7 owns that).
_SCRATCH = tempfile.mkdtemp(prefix="hugpy-resv-tmpl-")
os.environ["HUGPY_RESERVATIONS_MEASURED"] = os.path.join(_SCRATCH, "measured.json")

from hugpy_video.intel.reservation import templates as T

_GIB = 1024 ** 3


def _write_measured(obj):
    with open(os.environ["HUGPY_RESERVATIONS_MEASURED"], "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def _clear_measured():
    p = os.environ["HUGPY_RESERVATIONS_MEASURED"]
    if os.path.exists(p):
        os.remove(p)


def test_reservable_set_matches_heavy_tasks():
    got = set(T.reservable_tasks())
    # The operator's exclusive-heavy set + the co-resident movie case.
    assert got == {
        "studio_i2v", "generate_studio_movie", "identity_reconstruction",
        "identity_mesh_build", "identity_video_extract", "identity_from_video",
        "generate_movie",
    }
    # Light tasks are NOT reservable (no whole-run claim).
    for light in ("generate_image", "generate_scene", "crop", "frame_extract",
                  "audio_extract"):
        assert not T.is_reservable(light)
        assert T.load_template(light) is None


def test_peak_is_max_single_stage_for_sequential_wan():
    _clear_measured()
    for task in ("studio_i2v", "generate_studio_movie", "identity_reconstruction"):
        tmpl = T.load_template(task)
        # One Wan pipeline peak (segments/views are SEQUENTIAL on one GPU).
        assert tmpl.peak_bytes() == 20 * _GIB
        assert tmpl.reservation_window == "whole_run"
        assert tmpl.exclusive_heavy is True


def test_peak_is_co_resident_sum_for_vision_movie():
    _clear_measured()
    tmpl = T.load_template("generate_movie")
    # peak == t2i checkpoint (4G) + vision judge (3.3G) held simultaneously.
    frame = next(s for s in tmpl.stages if s.name == "frame_render")
    judge = next(s for s in tmpl.stages if s.name == "vision_judge")
    assert tmpl.peak_bytes() == frame.vram_bytes() + judge.vram_bytes()
    assert tmpl.exclusive_heavy is False   # stage-windows; never force-evicts


def test_identity_mesh_peak_is_larger_of_sequential_stages():
    _clear_measured()
    tmpl = T.load_template("identity_mesh_build")
    # tpose Wan (20G) and Hunyuan mesh (16G) are SEQUENTIAL -> peak is the larger.
    assert tmpl.peak_bytes() == 20 * _GIB


def test_measured_overlay_wins_per_stage_and_task_peak():
    _write_measured({
        "studio_i2v": {"peak_vram_bytes": 19_600_000_000,
                       "stages": {"wan_denoise": {"vram_bytes_measured": 19_600_000_000}}},
    })
    tmpl = T.load_template("studio_i2v")
    assert tmpl.peak_bytes() == 19_600_000_000          # task-level measured peak wins
    stage = tmpl.stages[0]
    assert stage.vram_bytes() == 19_600_000_000
    assert stage.footprint_source == "measured"
    _clear_measured()


def test_measured_overlay_flat_stage_map_shape():
    # p7 may write a flat {stage: bytes} map — the loader must tolerate it.
    _write_measured({"identity_mesh_build": {"mesh_build": 18_000_000_000}})
    tmpl = T.load_template("identity_mesh_build")
    mesh = next(s for s in tmpl.stages if s.name == "mesh_build")
    assert mesh.vram_bytes() == 18_000_000_000
    assert mesh.footprint_source == "measured"
    # Peak still the larger sequential stage (tpose 20G > mesh 18G).
    assert tmpl.peak_bytes() == 20 * _GIB
    _clear_measured()


def test_malformed_measured_degrades_to_estimates():
    with open(os.environ["HUGPY_RESERVATIONS_MEASURED"], "w", encoding="utf-8") as fh:
        fh.write("{ this is not json ]")
    tmpl = T.load_template("studio_i2v")     # must NOT raise
    assert tmpl.peak_bytes() == 20 * _GIB    # estimate stands
    _clear_measured()


def test_wrapper_key_and_missing_task_are_tolerated():
    _write_measured({"tasks": {"studio_i2v": {"peak_vram_bytes": 15_000_000_000}}})
    assert T.load_template("studio_i2v").peak_bytes() == 15_000_000_000
    # A task absent from the overlay keeps its estimate.
    assert T.load_template("generate_studio_movie").peak_bytes() == 20 * _GIB
    _clear_measured()


def test_as_dict_is_json_safe_and_carries_provenance():
    tmpl = T.load_template("studio_i2v")
    d = tmpl.as_dict()
    json.dumps(d)   # must serialize
    assert d["task"] == "studio_i2v"
    assert d["peak_vram_bytes"] == 20 * _GIB
    assert d["stages"][0]["footprint_source"] == "estimate"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
