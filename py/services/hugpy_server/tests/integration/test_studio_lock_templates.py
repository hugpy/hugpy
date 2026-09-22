"""Round 6 identity-lock TEMPLATES + scene-to-scene continuity presets.

Four new curated studio presets (all bind the Wan 2.1 VACE 1.3B control model at
FP16 @ 12 GB — verified via CapabilityRouter: the FP16 threshold for VACE-1.3B @
832x480 is 10 GB):

  * lock-person      : id_lock 832x480 @16 @ 12 GB -> wan2.1-vace-1.3b FP16, needs refs.
  * lock-thing       : id_lock 832x480 @16 @ 12 GB -> wan2.1-vace-1.3b FP16, needs refs.
  * scene-bias       : id_lock 832x480 @16 @ 12 GB -> wan2.1-vace-1.3b FP16, needs refs.
  * scene-continuity : v2v     832x480 @16 @ 12 GB -> wan2.1-vace-1.3b FP16, needs a
                       SOURCE (requires_source); references are OPTIONAL here
                       (requires_reference stays False — the route allows refs on v2v).

Router-verified like the existing seeds (studio.router.CapabilityRouter). Script style
matches tests/test_studio_presets_route.py (plain python, numbered PASS/FAIL, nonzero
exit iff any FAILED). pytest is NOT installed in this venv.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/test_studio_lock_templates.py
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile

logging.disable(logging.INFO)
os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")
os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="studio-lock-test-"))

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import importlib  # noqa: E402

from flask import Flask  # noqa: E402

from hugpy_video.intel.studio_presets import available_studio_presets, get_studio_preset
from hugpy_video.intel.studio.job import make_studio_i2v, StudioI2VSpec
from hugpy_video.intel.studio.router import CapabilityRouter
from hugpy_video.intel.studio.schemas import CapabilityRequest, Resolution
from hugpy_video.intel.studio.enums import Capability, Framework, Precision

vr = importlib.import_module("hugpy_server.app.routes.video_routes")
app = Flask(__name__)
app.register_blueprint(vr.video_bp)
client = app.test_client()

# The three id_lock TEMPLATES + the v2v continuity preset — each -> the SAME VACE-1.3B
# model, at FP16 (the 12 GB budget clears the 10 GB FP16 threshold).
_ID_LOCK_TEMPLATES = ("lock-person", "lock-thing", "scene-bias")
_ALL_NEW = _ID_LOCK_TEMPLATES + ("scene-continuity",)
_EXPECTED_MODEL = "wan2.1-vace-1.3b"


def _resolve(preset):
    return CapabilityRouter().resolve(CapabilityRequest(
        capability=Capability(preset.capability),
        target_resolution=Resolution(preset.width, preset.height, preset.fps),
        vram_budget_gb=preset.vram_budget_gb))


# --------------------------------------------------------------------------- #
# [1] All four new presets are registered and structurally valid (request_body()
#     is a verbatim make_studio_i2v body).
# --------------------------------------------------------------------------- #
def test_new_presets_registered_and_valid():
    for pid in _ALL_NEW:
        p = get_studio_preset(pid)
        assert p is not None, f"{pid} must be registered"
        spec = make_studio_i2v(**p.request_body())
        assert isinstance(spec, StudioI2VSpec), (pid, type(spec))
        assert spec.width == 832 and spec.height == 480, (pid, spec.width, spec.height)
        assert spec.fps == 16, (pid, spec.fps)


# --------------------------------------------------------------------------- #
# [2] The three id_lock TEMPLATES bind Wan 2.1 VACE 1.3B at FP16, exactly (never
#     the synthetic prover, never INT8 — the 12 GB budget clears the FP16 floor).
# --------------------------------------------------------------------------- #
def test_id_lock_templates_bind_vace_fp16():
    for pid in _ID_LOCK_TEMPLATES:
        p = get_studio_preset(pid)
        assert p.capability == "id_lock", (pid, p.capability)
        assert p.vram_budget_gb == 12.0, (pid, p.vram_budget_gb)
        b = _resolve(p).unwrap()
        assert b.framework == Framework.WAN, (pid, b.framework)
        assert b.model_id == _EXPECTED_MODEL, (pid, b.model_id)
        assert b.precision == Precision.FP16, (pid, b.precision)


# --------------------------------------------------------------------------- #
# [3] scene-continuity is a v2v transform that binds the same VACE-1.3B at FP16.
# --------------------------------------------------------------------------- #
def test_scene_continuity_binds_vace_fp16_v2v():
    p = get_studio_preset("scene-continuity")
    assert p.capability == "v2v", p.capability
    assert p.vram_budget_gb == 12.0, p.vram_budget_gb
    b = _resolve(p).unwrap()
    assert b.model_id == _EXPECTED_MODEL, b.model_id
    assert b.precision == Precision.FP16, b.precision


# --------------------------------------------------------------------------- #
# [4] The three id_lock templates carry requires_reference on the GET list + the
#     POST /apply envelope (the UI gate that forces the reference-picker), and it
#     never leaks into the POSTable request body.
# --------------------------------------------------------------------------- #
def test_id_lock_templates_requires_reference_signal():
    presets = client.get("/video/studio/presets").get_json()["presets"]
    by_id = {p["id"]: p for p in presets}
    for pid in _ID_LOCK_TEMPLATES:
        assert get_studio_preset(pid).requires_reference is True, pid
        row = by_id[pid]
        assert row.get("requires_reference") is True, (pid, row)
        assert row["capability"] == "id_lock", (pid, row)
        ap = client.post(f"/video/studio/presets/{pid}/apply").get_json()
        assert ap.get("ok") is True, (pid, ap)
        assert ap.get("requires_reference") is True, (pid, ap)
        assert ap.get("capability") == "id_lock", (pid, ap)
        # UI signal only — must NOT leak into the make_studio_i2v request body.
        assert "requires_reference" not in ap["request"], (pid, ap["request"])


# --------------------------------------------------------------------------- #
# [5] scene-continuity requires_source (a prior clip) but NOT requires_reference
#     (references are OPTIONAL on the v2v continuity path).
# --------------------------------------------------------------------------- #
def test_scene_continuity_source_required_refs_optional():
    p = get_studio_preset("scene-continuity")
    assert p.requires_source is True, "scene-continuity needs a source clip"
    assert p.requires_reference is False, "scene-continuity references are OPTIONAL"
    row = next(x for x in client.get("/video/studio/presets").get_json()["presets"]
               if x["id"] == "scene-continuity")
    assert row.get("requires_source") is True, row
    assert row.get("requires_reference") is False, row
    assert row["capability"] == "v2v", row
    ap = client.post("/video/studio/presets/scene-continuity/apply").get_json()
    assert ap.get("requires_source") is True, ap
    assert ap.get("requires_reference") is False, ap
    assert ap["request"]["capability"] == "v2v", ap["request"]
    assert "requires_source" not in ap["request"], ap["request"]


# --------------------------------------------------------------------------- #
# [6] GET list shape carries the new templates in the pinned per-preset shape.
# --------------------------------------------------------------------------- #
def test_new_presets_on_get_list():
    by_id = {p["id"]: p for p in client.get("/video/studio/presets").get_json()["presets"]}
    for pid in _ALL_NEW:
        assert pid in by_id, pid
        p = by_id[pid]
        for k in ("id", "name", "description", "capability", "width", "height",
                  "fps", "vram_budget_gb", "recommended"):
            assert k in p, (pid, k)
        assert p["width"] == 832 and p["height"] == 480, (pid, p)


# --------------------------------------------------------------------------- #
# [7] No dead seeds — every preset (incl. the four new ones) still routes.
# --------------------------------------------------------------------------- #
def test_all_presets_route():
    for p in available_studio_presets():
        r = _resolve(p)
        assert r.is_ok(), (p.id, r.error.code.value if r.is_err() else r)


CHECKS = [
    ("all four new presets registered + request_body() valid", test_new_presets_registered_and_valid),
    ("lock-person / lock-thing / scene-bias bind wan2.1-vace-1.3b at FP16", test_id_lock_templates_bind_vace_fp16),
    ("scene-continuity (v2v) binds wan2.1-vace-1.3b at FP16", test_scene_continuity_binds_vace_fp16_v2v),
    ("id_lock templates carry requires_reference (list + apply, not in request)", test_id_lock_templates_requires_reference_signal),
    ("scene-continuity requires_source, references optional", test_scene_continuity_source_required_refs_optional),
    ("new templates on the GET list in the pinned shape", test_new_presets_on_get_list),
    ("every preset still routes to a model (no dead seeds)", test_all_presets_route),
]


def main() -> int:
    passed = failed = 0
    for i, (name, fn) in enumerate(CHECKS, 1):
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"[{i}] PASS  {name}")
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
