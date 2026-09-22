"""id_lock fix 2 (routes half): the studio routes default to 832x480 (R_480P),
never the guaranteed-fail 512x512. Split out of hugpy_video's
test_id_lock_never_silently_drops.py because it reads hugpy_server source.
"""
import importlib.util
import os

import pytest


def _video_routes_source() -> str:
    spec = importlib.util.find_spec("hugpy_server")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("hugpy_server not installed")
    path = os.path.join(list(spec.submodule_search_locations)[0],
                        "app", "routes", "video_routes.py")
    if not os.path.isfile(path):
        pytest.skip(f"{path} not present")
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def test_the_route_default_geometry_is_servable():
    """512x512 was outside EVERY id-capable and v2v-capable envelope on this
    fleet, so the default could not succeed. 832x480 is what the studio serves."""
    routes = _video_routes_source()
    assert "width = 832 if width is None else width" in routes
    assert "height = 480 if height is None else height" in routes
    assert "width = 512 if width is None else width" not in routes, (
        "the guaranteed-fail square default must be gone from every route")
