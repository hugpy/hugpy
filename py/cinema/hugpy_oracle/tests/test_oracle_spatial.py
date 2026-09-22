"""k116 — spatial contract: coordinate round-trip, camera projection,
manifest validation (nine faults), Fold 1→Fold 2 payload + frame alignment,
tone 0–10 never discards geometry, explicit tier fallback.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_spatial.py -q
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import replace

logging.disable(logging.INFO)
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from hugpy_oracle.spatial import (
    CANONICAL,
    CameraIntrinsics,
    CameraSpec,
    CaptureTier,
    ConditioningPass,
    ConditioningRequest,
    ConditioningSpec,
    CoordinateSystem,
    EntitySpec,
    FaultCode,
    InferenceTier,
    ProvenanceSpec,
    RenderSpec,
    RenderTier,
    SimulationSpec,
    SpatialSceneManifest,
    StyleSpec,
    TierFallback,
    TierProfile,
    Timebase,
    convert_points,
    frame_alignment_report,
    manifest_json_schema,
    tone_profile,
    validate_manifest,
)

SHA = "sha256:" + "ab" * 32


def manifest(**kw) -> SpatialSceneManifest:
    base = dict(
        run_id="run_01", segment_id="segment_004", artifact_revision=3,
        tier_profile=TierProfile(CaptureTier.REALTIME_MOCAP, InferenceTier.DENSE_CONDITIONING, RenderTier.NEURAL_RENDER),
        timebase=Timebase(24.0, 0, 119, 5.0), coordinate_system=CANONICAL,
        camera=CameraSpec("artifact://camera/segment_004.npz", "artifact://camera/segment_004_intrinsics.json",
                          CameraIntrinsics(1000.0, 1000.0, 960.0, 540.0, 1920, 1080), 0.1, 100.0, SHA),
        entities=(EntitySpec("character_alex", "character", "artifact://characters/alex.glb",
                             "artifact://characters/alex_rig.glb", "artifact://animation/alex_segment_004.npz",
                             ("ref_alex_front", "ref_alex_profile"), SHA, "gltf"),),
        simulation=SimulationSpec(True, "artifact://physics/segment_004_cache.bin", "configured_backend", "1.2", 2, 7, SHA),
        conditioning=ConditioningSpec((ConditioningPass.DEPTH, ConditioningPass.NORMALS, ConditioningPass.POSE,
                                       ConditioningPass.SILHOUETTE, ConditioningPass.SEGMENTATION,
                                       ConditioningPass.OPTICAL_FLOW), "artifact://conditioning/segment_004/",
                                      0.9, 0.85, 1920, 1080),
        style=StyleSpec(2.0, "cinematic_photoreal_v1", "dynamic_cfg_v2"),
        render=RenderSpec(1920, 1080, 184627, "auto"),
        provenance=ProvenanceSpec("snapshot_01", 4, 3, 6, "rv-1"),
    )
    base.update(kw)
    return SpatialSceneManifest(**base)


# --------------------------------------------------------------------------- #
# coordinates
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("system", [
    CoordinateSystem(),                                                     # canonical
    CoordinateSystem(handedness="right", up_axis="Z", forward_axis="Y", world_units="cm"),   # Blender-ish
    CoordinateSystem(handedness="left", up_axis="Y", forward_axis="Z", world_units="meters"),  # Unity/DirectX-ish
    CoordinateSystem(handedness="right", up_axis="Z", forward_axis="-Y", world_units="mm"),
    CoordinateSystem(handedness="left", up_axis="Z", forward_axis="X", world_units="inches"),
])
def test_coordinate_conversions_round_trip_within_tolerance(system):
    pts = [(1.0, 2.0, -3.0), (0.0, 0.0, 0.0), (-12.5, 0.25, 7.75), (1e3, -1e-3, 42.0)]
    canon = convert_points(pts, system, CANONICAL)
    back = convert_points(canon, CANONICAL, system)
    for a, b in zip(pts, back):
        for x, y in zip(a, b):
            assert math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9)
    # units really convert
    if system.world_units == "cm":
        assert math.isclose(math.dist(canon[0], canon[1]), math.dist(pts[0], pts[1]) / 100.0, rel_tol=1e-9)


def test_z_up_to_y_up_maps_up_to_up():
    zup = CoordinateSystem(up_axis="Z", forward_axis="Y")
    (p,) = convert_points([(0.0, 0.0, 1.0)], zup, CANONICAL)   # "up" in Z-up
    assert p == pytest.approx((0.0, 1.0, 0.0))                   # is "up" (Y) canonically
    (f,) = convert_points([(0.0, 1.0, 0.0)], zup, CANONICAL)   # "forward" in that system
    assert f == pytest.approx((0.0, 0.0, -1.0))                  # canonical forward is -Z


def test_unknown_space_and_missing_units_are_refused():
    with pytest.raises(ValueError):
        CoordinateSystem(handedness="ambidextrous")
    with pytest.raises(ValueError):
        CoordinateSystem(world_units="")
    with pytest.raises(ValueError):
        CoordinateSystem(up_axis="Y", forward_axis="-Y")


# --------------------------------------------------------------------------- #
# camera projection
# --------------------------------------------------------------------------- #


def test_camera_projection_produces_correct_screen_coordinates():
    cam = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, width=1920, height=1080)
    centre = cam.project((0.0, 0.0, -2.0))
    assert centre["u"] == pytest.approx(960.0) and centre["v"] == pytest.approx(540.0)
    assert centre["ndc"] == pytest.approx((0.0, 0.0)) and centre["depth_m"] == 2.0 and centre["in_frame"]
    right_up = cam.project((1.0, 0.5, -2.0))   # 1 m right, 0.5 m up at 2 m
    assert right_up["u"] == pytest.approx(960.0 + 500.0)
    assert right_up["v"] == pytest.approx(540.0 - 250.0)         # up in world = up in image (smaller v)
    assert right_up["ndc"][0] > 0 and right_up["ndc"][1] > 0
    assert cam.project((0.0, 0.0, 2.0)) is None                  # behind the camera is not projectable
    assert cam.normalized_depth(2.0, 0.1, 100.0) == pytest.approx((2.0 - 0.1) / 99.9)


# --------------------------------------------------------------------------- #
# manifest validation — the nine faults
# --------------------------------------------------------------------------- #


def test_valid_manifest_passes_and_round_trips():
    m = manifest()
    v = validate_manifest(m, expected_run_id="run_01", expected_revision=3, expected_snapshot_id="snapshot_01",
                          expected_fps=24.0, known_entity_ids={"character_alex"},
                          asset_exists=lambda u: True, checksum_of=lambda u: SHA)
    assert v.ok, v.to_dict()
    d = json.loads(json.dumps(m.to_dict()))
    assert SpatialSceneManifest.from_dict(d).digest == m.digest
    assert d["coordinate_system"] == {"handedness": "right", "up_axis": "Y", "forward_axis": "-Z",
                                      "world_units": "meters", "matrix_order": "column_major",
                                      "quaternion_order": "xyzw", "depth_convention": "metric"}


def _codes(v):
    return {f.code for f in v.faults}


def test_rejects_unknown_coordinate_space_and_missing_units_from_raw_payload():
    raw = manifest().to_dict()
    raw["coordinate_system"]["handedness"] = "both"
    assert FaultCode.UNKNOWN_COORDINATE_SPACE in _codes(validate_manifest(raw))
    raw = manifest().to_dict()
    raw["coordinate_system"]["world_units"] = ""
    assert FaultCode.MISSING_UNITS in _codes(validate_manifest(raw))


def test_rejects_invalid_frame_range_and_rate_mismatch():
    assert FaultCode.INVALID_FRAME_RANGE in _codes(validate_manifest(manifest(timebase=Timebase(24.0, 10, 5, 1.0))))
    assert FaultCode.INVALID_FRAME_RANGE in _codes(validate_manifest(manifest(timebase=Timebase(24.0, 0, 119, 9.0))))
    assert FaultCode.FRAME_RATE_MISMATCH in _codes(validate_manifest(manifest(), expected_fps=25.0))


def test_rejects_missing_assets_and_checksum_failures():
    v = validate_manifest(manifest(), asset_exists=lambda u: not u.endswith(".glb"))
    assert FaultCode.MISSING_ASSET in _codes(v)
    assert any("alex.glb" in f.message for f in v.faults)
    v = validate_manifest(manifest(), checksum_of=lambda u: "sha256:" + "00" * 32)
    assert FaultCode.CHECKSUM_FAILURE in _codes(v)
    bad = manifest(entities=(replace(manifest().entities[0], checksum="md5:abc"),))
    assert FaultCode.CHECKSUM_FAILURE in _codes(validate_manifest(bad))


def test_rejects_unresolved_entities():
    v = validate_manifest(manifest(), known_entity_ids={"character_sam"})
    assert FaultCode.UNRESOLVED_ENTITY in _codes(v)
    no_refs = manifest(entities=(replace(manifest().entities[0], identity_reference_ids=()),))
    assert FaultCode.UNRESOLVED_ENTITY in _codes(validate_manifest(no_refs))


def test_rejects_resolution_mismatch():
    m = manifest(conditioning=replace(manifest().conditioning, width=1280, height=720))
    assert FaultCode.RESOLUTION_MISMATCH in _codes(validate_manifest(m))
    m = manifest(camera=replace(manifest().camera, intrinsics=CameraIntrinsics(1000, 1000, 640, 360, 1280, 720)))
    assert FaultCode.RESOLUTION_MISMATCH in _codes(validate_manifest(m))


def test_rejects_cross_run_and_revision_contamination():
    v = validate_manifest(manifest(), expected_run_id="run_02")
    assert FaultCode.CROSS_RUN_CONTAMINATION in _codes(v)
    v = validate_manifest(manifest(), expected_revision=4)
    assert FaultCode.CROSS_RUN_CONTAMINATION in _codes(v)
    v = validate_manifest(manifest(), expected_snapshot_id="snapshot_99")
    assert FaultCode.CROSS_RUN_CONTAMINATION in _codes(v)


def test_tier_pass_coherence_and_geometry_floor():
    t1 = manifest(tier_profile=TierProfile(CaptureTier.STATIC_RIG, InferenceTier.TOKEN_ROUTING, RenderTier.STATIC_STYLE))
    assert FaultCode.TIER_PASS_MISMATCH in _codes(validate_manifest(t1))   # dense passes under tier 1
    weak = manifest(conditioning=replace(manifest().conditioning, geometry_strength=0.1))
    assert FaultCode.GEOMETRY_FLOOR_VIOLATION in _codes(validate_manifest(weak))
    disabled = manifest(conditioning=replace(manifest().conditioning, geometry_strength=0.0,
                                             geometry_disabled_by_operator=True))
    assert FaultCode.GEOMETRY_FLOOR_VIOLATION not in _codes(validate_manifest(disabled))


# --------------------------------------------------------------------------- #
# Fold 1 -> Fold 2 payload, frame-for-frame alignment
# --------------------------------------------------------------------------- #


def test_conditioning_request_is_the_fold1_to_fold2_payload():
    m = manifest()
    req = ConditioningRequest.from_manifest(m)
    d = req.to_dict()
    assert d["frames"][0] == 0 and d["frames"][-1] == 119 and len(d["frames"]) == 120
    assert d["timestamps"][1] == pytest.approx(1 / 24)
    assert d["width"] == 1920 and d["passes"][0] == "depth" and d["manifest_digest"] == m.digest
    assert d["hard_containment"] is True and "hard geometric" in d["guarantee"]
    t1 = manifest(tier_profile=TierProfile(CaptureTier.STATIC_RIG, InferenceTier.TOKEN_ROUTING, RenderTier.STATIC_STYLE),
                  conditioning=replace(m.conditioning, requested_passes=(ConditioningPass.BBOX, ConditioningPass.KEYPOINTS)))
    d1 = ConditioningRequest.from_manifest(t1).to_dict()
    assert d1["hard_containment"] is False and "not a geometric guarantee" in d1["guarantee"]


def test_frame_alignment_detects_missing_extra_and_shifted_frames():
    req = ConditioningRequest.from_manifest(manifest(timebase=Timebase(24.0, 0, 9, 10 / 24)))
    ok = frame_alignment_report(req, list(range(10)), [i / 24 for i in range(10)])
    assert ok["aligned"] and ok["order_ok"]
    bad = frame_alignment_report(req, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10], [i / 24 for i in [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]])
    assert not bad["aligned"] and bad["missing"] == [3] and bad["extra"] == [10]
    shifted = frame_alignment_report(req, list(range(10)), [i / 24 + (0.1 if i == 5 else 0.0) for i in range(10)])
    assert not shifted["aligned"] and shifted["shifted"][0]["frame"] == 5


# --------------------------------------------------------------------------- #
# tone and tier fallback
# --------------------------------------------------------------------------- #


def test_tone_changes_style_without_discarding_geometry():
    photo = tone_profile(0.0, render_tier=RenderTier.NEURAL_RENDER)
    toon = tone_profile(10.0, render_tier=RenderTier.DYNAMIC_SCHEDULE)
    assert photo.controls["shading"] == "pbr" and toon.controls["shading"] == "flat"
    assert photo.controls["subsurface_scattering"] is True and toon.controls["edge_treatment"] == 1.0
    assert photo.positive_style != toon.positive_style
    for prof in (photo, toon, tone_profile(5.0)):
        assert prof.controls["geometry_strength"] >= prof.geometry_floor
        assert all(g >= prof.geometry_floor for _, g in prof.geometry_schedule)
    assert len(toon.cfg_schedule) == 8 and toon.cfg_schedule[0][1] < toon.cfg_schedule[-1][1]
    off = tone_profile(10.0, geometry_disabled=True)
    assert off.controls["geometry_strength"] == 0.0 and off.geometry_disabled is True
    assert photo.profile_id.startswith("tone_v1:tone0.0")
    with pytest.raises(ValueError):
        tone_profile(11.0)


def test_tier_fallback_is_explicit_versioned_and_bumps_revision():
    m = manifest()
    fb = TierFallback("inference", 3, 2, "no 3DGS backend seated on the fleet", "router")
    m2 = m.with_fallback(fb)
    assert m2.tier_profile.inference is InferenceTier.DENSE_CONDITIONING
    assert m2.artifact_revision == m.artifact_revision + 1
    assert m2.fallbacks[0].to_dict()["reason"].startswith("no 3DGS")
    assert m2.digest != m.digest
    # fallback of the same manifest twice is reproducible
    assert m.with_fallback(fb).digest == m2.digest


def test_json_schema_matches_the_directive_example_shape():
    schema = manifest_json_schema()
    assert schema["title"] == "SpatialSceneManifest"
    assert set(schema["required"]) >= {"timebase", "coordinate_system", "camera", "entities", "conditioning",
                                       "style", "render", "provenance", "tier_profile"}
    assert schema["properties"]["style"]["properties"]["tone"]["maximum"] == 10
    json.dumps(schema)


def test_tone_scale_has_one_conversion_point_and_refuses_double_scaling():
    from hugpy_oracle.tone_scale import to_operator, to_unit, describe
    assert to_operator(0.2) == 2.0 and to_unit(2.0) == 0.2
    assert to_unit(to_operator(0.37)) == pytest.approx(0.37)
    with pytest.raises(ValueError):
        to_operator(2.0)          # already operator-scale: refuse, do not produce 20
    with pytest.raises(ValueError):
        to_unit(20.0)
    assert describe(0.0).startswith("photorealistic") and describe(10.0).startswith("vector")
    # the only "* 10" for tone in the oracle package lives in tone_scale.py
    import glob, os
    pkg = os.path.dirname(os.path.dirname(__import__("hugpy_oracle.tone_scale", fromlist=["x"]).__file__))
    offenders = []
    for f in glob.glob(os.path.join(pkg, "oracle", "*.py")) + glob.glob(os.path.join(pkg, "oracle", "recipes", "*.py")):
        if f.endswith("tone_scale.py"):
            continue
        src = open(f).read()
        if "tone) * 10" in src or "tone * 10" in src or "tone / 10" in src or "tone) / 10" in src:
            offenders.append(os.path.basename(f))
    assert offenders == [], offenders
