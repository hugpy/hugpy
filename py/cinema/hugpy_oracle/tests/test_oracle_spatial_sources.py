"""or-k8 / or-k15 — live SpatialSceneManifest producers and structured
difficulty signals.

Locks:
  [1] ``from_gltf`` (JSON and GLB), ``from_usd`` (USDA) and ``from_pose_track``
      each yield a manifest that passes ``validate_manifest`` and stamps the
      source tool, path and sha256 into provenance; node transforms (TRS,
      matrix, parented, animated) become placements and trajectories; cameras
      become ``CameraSpec`` + keyframes; scene units are honoured.
  [2] honest refusals: no camera, USDC binary, a character without identity
      references.
  [3] ``SegmentSpec.spatial_ref`` is set when the goal supplies a source path
      or manifest, is ``None`` otherwise, is refused (not faked) for a broken
      source, and resolves back to the ``SpatialSource`` by ref.
  [4] ``extract_signals`` reads camera motion, trajectories, occlusion and
      cloth/hair structurally when given a shot/manifest, labels the regex
      path as fallback, and ``compile_context`` passes both through.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_spatial_sources.py -q
"""
from __future__ import annotations

import base64
import json
import logging
import os
import struct
import sys

import pytest

logging.disable(logging.INFO)

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_oracle import spatial as sp
from hugpy_oracle import spatial_sources as ss
from hugpy_oracle.prompt_compiler import compile_context, difficulty_score, extract_signals
from hugpy_oracle.segments import CompileRefused, resolve_spatial_ref, spatial_source_for

from test_oracle_segments import compile_fixture  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures: tiny synthetic scenes written to tmp_path
# --------------------------------------------------------------------------- #


def _gltf_doc(*, animated: bool = True, camera: bool = True) -> dict:
    times = [0.0, 1.0]
    cam_tr = [(0, 1, 5), (2, 1, 5)]
    ball_tr = [(0, 0, 0), (3, 2, 0)]
    buf = struct.pack("<2f", *times) + struct.pack("<6f", *sum(cam_tr, ())) + struct.pack("<6f", *sum(ball_tr, ()))
    nodes = [
        {"name": "Cam", "translation": [0, 1, 5]},
        {"name": "Ball", "mesh": 0, "translation": [0, 0, 0], "extras": {"tags": ["rigid"]}},
        {"name": "Hero", "mesh": 0, "skin": 0, "translation": [1, 0, 0],
         "extras": {"identity_refs": ["hero_pack"]}},
        {"name": "CapeRig", "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 1], "children": [4]},
        {"name": "Cape", "mesh": 0, "translation": [0, 0.5, 0], "extras": {"tags": ["cloth"]}},
    ]
    if camera:
        nodes[0]["camera"] = 0
    doc = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 1, 2, 3]}],
        "nodes": nodes,
        "cameras": [{"type": "perspective", "perspective": {"yfov": 0.8, "znear": 0.1, "zfar": 100}}],
        "meshes": [{"primitives": []}], "skins": [{"joints": [2]}],
        "buffers": [{"byteLength": len(buf),
                     "uri": "data:application/octet-stream;base64," + base64.b64encode(buf).decode()}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 8},
                        {"buffer": 0, "byteOffset": 8, "byteLength": 24},
                        {"buffer": 0, "byteOffset": 32, "byteLength": 24}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 2, "type": "SCALAR"},
                      {"bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC3"},
                      {"bufferView": 2, "componentType": 5126, "count": 2, "type": "VEC3"}],
    }
    if animated:
        doc["animations"] = [{
            "name": "shot", "samplers": [{"input": 0, "output": 1}, {"input": 0, "output": 2}],
            "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}},
                         {"sampler": 1, "target": {"node": 1, "path": "translation"}}]}]
    return doc


@pytest.fixture
def gltf_path(tmp_path):
    p = tmp_path / "scene.gltf"
    p.write_text(json.dumps(_gltf_doc()))
    return str(p)


@pytest.fixture
def glb_path(tmp_path):
    js = json.dumps(_gltf_doc()).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    glb = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js)) + struct.pack("<II", len(js), 0x4E4F534A) + js
    p = tmp_path / "scene.glb"
    p.write_bytes(glb)
    return str(p)


USDA = '''#usda 1.0
(
    metersPerUnit = 0.01
    upAxis = "Z"
    startTimeCode = 0
    endTimeCode = 24
    timeCodesPerSecond = 24
)

def Xform "World"
{
    def Camera "Cam"
    {
        float focalLength = 35
        float verticalAperture = 24
        float2 clippingRange = (1, 10000)
        double3 xformOp:translate.timeSamples = {
            0: (0, -500, 150),
            24: (100, -500, 150),
        }
        float3 xformOp:rotateXYZ = (90, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }
    def Xform "Hero" (
        kind = "component"
    )
    {
        custom string entity_type = "character"
        custom string[] identity_refs = ["hero_pack"]
        custom string[] tags = ["hair"]
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Mesh "Body"
        {
            point3f[] points = []
        }
    }
    def Sphere "Ball"
    {
        double3 xformOp:translate.timeSamples = {
            0: (0, 0, 100),
            12: (100, 0, 200),
            24: (200, 0, 0),
        }
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
'''


@pytest.fixture
def usda_path(tmp_path):
    p = tmp_path / "scene.usda"
    p.write_text(USDA)
    return str(p)


def _pose_frames(n: int = 13, camera: bool = True) -> list[dict]:
    out = []
    for i in range(n):
        fr = {"frame": i, "entities": {
            "ball": {"position": [i * 0.5, 0, 0]},
            "hero": {"position": [0, 0, 0], "type": "character", "identity_refs": ["hp"]}}}
        if camera:
            fr["camera"] = {"position": [0, 1, 5], "look_at": [0, 0, 0], "yfov_deg": 40}
        out.append(fr)
    return out


# --------------------------------------------------------------------------- #
# [1] producers
# --------------------------------------------------------------------------- #


def test_gltf_json_yields_an_admissible_manifest_with_provenance(gltf_path):
    src = ss.from_gltf(gltf_path, run_id="run-1", segment_id="s1", fps=12)
    m = src.manifest
    assert sp.validate_manifest(m, asset_exists=ss.source_asset_exists, checksum_of=ss.source_checksum_of).ok
    assert src.source_tool == "gltf" and src.source_path == gltf_path
    assert src.source_sha256 == ss.sha256_of_file(gltf_path)
    hist = " ".join(m.provenance.conversion_history)
    assert "source_tool=gltf" in hist and gltf_path in hist and src.source_sha256 in hist
    assert m.coordinate_system.world_units == "meters" and m.coordinate_system.up_axis == "Y"
    assert m.timebase.fps == 12 and m.timebase.frame_count == 13     # 1.0 s animation
    assert src.ref == f"gltf:{gltf_path}#{src.source_sha256}"
    json.dumps(src.to_dict())


def test_gltf_nodes_become_entities_and_cameras_become_keyframes(gltf_path):
    src = ss.from_gltf(gltf_path, run_id="run-1", segment_id="s1", fps=12)
    ids = {e.entity_id: e for e in src.manifest.entities}
    assert set(ids) == {"Ball", "Hero", "CapeRig", "Cape"}
    assert ids["Hero"].entity_type == "character" and ids["Hero"].identity_reference_ids == ("hero_pack",)
    assert ids["Hero"].rig_uri.endswith("#skin/0") and ids["Ball"].mesh_uri.endswith("#mesh/0")
    assert ids["Ball"].animation_uri.endswith("#animation/0") and ids["Hero"].animation_uri is None
    # camera: 13 keyframes, translated 0 -> 2 m along X, looking down -Z
    assert len(src.camera_track) == 13
    assert src.camera_track[0].position == (0.0, 1.0, 5.0)
    assert src.camera_track[-1].position == pytest.approx((2.0, 1.0, 5.0))
    assert src.camera_track[0].forward == pytest.approx((0.0, 0.0, -1.0))
    assert src.manifest.camera.track_uri == f"{gltf_path}#camera/0"
    assert src.manifest.camera.intrinsics is not None and src.manifest.camera.intrinsics.width == 1280
    # trajectories: ball flies, matrix-placed rig + parented child resolve
    assert src.track_for("Ball").displacement_m == pytest.approx((3 ** 2 + 2 ** 2) ** 0.5)
    assert src.track_for("Cape").positions[0] == pytest.approx((-1.0, 0.5, 0.0))
    assert src.track_for("Cape").tags == ("cloth", "cape")
    assert src.manifest.tier_profile.inference is sp.InferenceTier.DENSE_CONDITIONING


def test_glb_container_is_read_from_its_json_chunk(glb_path):
    src = ss.from_gltf(glb_path, run_id="run-1", segment_id="s1")
    assert src.source_tool == "glb"
    assert len(src.camera_track) == 25                    # 1 s at the default 24 fps
    assert ss.load_source(glb_path, run_id="run-1", segment_id="s1").manifest.digest == src.manifest.digest


def test_gltf_asset_units_override_is_honoured(tmp_path):
    doc = _gltf_doc(animated=False)
    doc["asset"]["extras"] = {"units": "centimeters"}
    p = tmp_path / "cm.gltf"
    p.write_text(json.dumps(doc))
    src = ss.from_gltf(str(p), run_id="r", segment_id="s")
    assert src.manifest.coordinate_system.world_units == "centimeters"
    assert src.camera_track[0].position == pytest.approx((0.0, 0.01, 0.05))   # metres
    assert src.manifest.timebase.frame_count == 1                              # static scene


def test_usda_text_layer_yields_an_admissible_manifest(usda_path):
    src = ss.from_usd(usda_path, run_id="run-1", segment_id="s1")
    m = src.manifest
    assert sp.validate_manifest(m).ok and src.source_tool == "usda"
    assert m.coordinate_system.world_units == "centimeters" and m.coordinate_system.up_axis == "Z"
    assert m.timebase.fps == 24 and m.timebase.frame_count == 25
    assert [e.entity_id for e in m.entities] == ["Hero", "Ball"]      # Body mesh folds into Hero
    hero = m.entities[0]
    assert hero.entity_type == "character" and hero.identity_reference_ids == ("hero_pack",)
    assert src.track_for("Hero").tags == ("hair",)
    # cm -> m, camera translated 1 m over the shot and rotated to look down +Y
    assert src.camera_track[0].position == pytest.approx((0.0, -5.0, 1.5))
    assert src.camera_track[-1].position == pytest.approx((1.0, -5.0, 1.5))
    assert src.camera_track[0].forward == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert src.track_for("Ball").positions[0] == pytest.approx((0.0, 0.0, 1.0))
    assert src.track_for("Ball").positions[12] == pytest.approx((1.0, 0.0, 2.0))
    assert m.camera.near_meters == pytest.approx(0.01) and m.camera.track_uri.endswith("#/World/Cam")
    assert "source_sha256=" + src.source_sha256 in m.provenance.conversion_history


def test_pose_track_yields_a_mocap_tier_manifest():
    src = ss.from_pose_track(_pose_frames(), run_id="run-1", segment_id="s1", fps=12)
    m = src.manifest
    assert sp.validate_manifest(m).ok
    assert m.tier_profile.capture is sp.CaptureTier.REALTIME_MOCAP
    assert m.tier_profile.inference is sp.InferenceTier.TOKEN_ROUTING   # no meshes -> no dense pass
    assert m.timebase.frame_count == 13
    assert src.track_for("ball").displacement_m == pytest.approx(6.0)
    assert src.track_for("hero").entity_type == "character"
    assert src.camera_track[0].forward[1] < 0                             # looks down at the origin
    assert src.ref.startswith("pose_track:-#sha256:")
    again = ss.from_pose_track(_pose_frames(), run_id="run-1", segment_id="s1", fps=12)
    assert again.manifest.digest == m.digest                              # deterministic


def test_pose_track_json_file_loads_through_dispatch(tmp_path):
    p = tmp_path / "mocap.json"
    p.write_text(json.dumps({"fps": 12, "frames": _pose_frames()}))
    src = ss.load_source(str(p), run_id="r", segment_id="s")
    assert src.source_path == str(p) and src.manifest.timebase.fps == 12


def test_spatial_source_round_trips_through_its_dict(gltf_path):
    src = ss.from_gltf(gltf_path, run_id="run-1", segment_id="s1", fps=12)
    back = ss.SpatialSource.from_dict(json.loads(json.dumps(src.to_dict())))
    assert back.manifest.digest == src.manifest.digest and back.ref == src.ref
    assert back.track_for("Ball").positions == src.track_for("Ball").positions


# --------------------------------------------------------------------------- #
# [2] honest refusals
# --------------------------------------------------------------------------- #


def test_gltf_without_a_camera_is_refused(tmp_path):
    p = tmp_path / "nocam.gltf"
    p.write_text(json.dumps(_gltf_doc(camera=False)))
    with pytest.raises(ss.SpatialSourceError, match="no usable camera"):
        ss.from_gltf(str(p), run_id="r", segment_id="s")


def test_gltf_character_without_identity_refs_fails_validation_not_silently(tmp_path):
    doc = _gltf_doc(animated=False)
    doc["nodes"][2]["extras"] = {}
    p = tmp_path / "anon.gltf"
    p.write_text(json.dumps(doc))
    with pytest.raises(ss.SpatialSourceError) as ei:
        ss.from_gltf(str(p), run_id="r", segment_id="s")
    assert any(f.code is sp.FaultCode.UNRESOLVED_ENTITY for f in ei.value.faults)
    # the remedy: name the references
    ok = ss.from_gltf(str(p), run_id="r", segment_id="s", identity_refs={"Hero": ["hero_pack"]})
    assert ok.manifest.entities[1].identity_reference_ids == ("hero_pack",)


def test_usdc_binary_is_an_explicit_not_implemented(tmp_path):
    p = tmp_path / "scene.usdc"
    p.write_bytes(b"PXR-USDC" + b"\x00" * 16)
    with pytest.raises(NotImplementedError, match="USDC binary crate"):
        ss.from_usd(str(p), run_id="r", segment_id="s")


def test_usda_without_a_camera_and_pose_track_without_a_camera_are_refused(tmp_path):
    p = tmp_path / "nocam.usda"
    p.write_text('#usda 1.0\n\ndef Xform "Box"\n{\n    double3 xformOp:translate = (0, 0, 0)\n}\n')
    with pytest.raises(ss.SpatialSourceError, match="no Camera prim"):
        ss.from_usd(str(p), run_id="r", segment_id="s")
    with pytest.raises(ss.SpatialSourceError, match="no camera"):
        ss.from_pose_track(_pose_frames(camera=False), run_id="r", segment_id="s")


# --------------------------------------------------------------------------- #
# [3] SegmentSpec.spatial_ref
# --------------------------------------------------------------------------- #


def test_specs_carry_no_spatial_ref_when_nothing_was_supplied():
    _lock, specs = compile_fixture()
    assert all(s.spatial_ref is None for s in specs)


def test_spatial_ref_is_set_from_a_source_path_per_segment(gltf_path, usda_path):
    _lock, specs = compile_fixture(spatial_refs={"s1": gltf_path, "s2": usda_path})
    by_id = {s.segment_id: s for s in specs}
    assert by_id["s1"].spatial_ref == f"gltf:{gltf_path}#{ss.sha256_of_file(gltf_path)}"
    assert by_id["s2"].spatial_ref == f"usda:{usda_path}#{ss.sha256_of_file(usda_path)}"
    assert by_id["s3"].spatial_ref is None
    src = spatial_source_for(by_id["s1"].spatial_ref, "s1")
    assert src.manifest.segment_id == "s1" and len(src.camera_track) == 25
    # the spec round-trips with the ref and the ref is part of the digest
    assert type(by_id["s1"]).from_dict(by_id["s1"].to_dict()).spatial_ref == by_id["s1"].spatial_ref
    _lock2, plain = compile_fixture()
    assert plain[0].digest != by_id["s1"].digest and plain[2].digest == by_id["s3"].digest


def test_spatial_ref_is_set_from_a_manifest_or_source_object(gltf_path):
    src = ss.from_gltf(gltf_path, run_id="run-x", segment_id="s1", fps=12)
    _lock, specs = compile_fixture(spatial_refs={"s1": src, "s2": src.manifest})
    assert specs[0].spatial_ref == src.ref
    assert specs[1].spatial_ref == "manifest:-#sha256:" + src.manifest.digest
    assert spatial_source_for(specs[1].spatial_ref).manifest.digest == src.manifest.digest
    assert resolve_spatial_ref("opaque-handle", run_id="r", segment_id="s") == "opaque-handle"
    assert resolve_spatial_ref(None, run_id="r", segment_id="s") is None


def test_a_broken_or_unsupported_source_is_refused_not_faked(tmp_path):
    bad = tmp_path / "broken.gltf"
    bad.write_text("{not json")
    with pytest.raises(CompileRefused, match="spatial source for 's1' refused"):
        compile_fixture(spatial_refs={"s1": str(bad)})
    usdc = tmp_path / "bin.usdc"
    usdc.write_bytes(b"PXR-USDC")
    with pytest.raises(CompileRefused, match="unsupported"):
        compile_fixture(spatial_refs={"s1": str(usdc)})
    with pytest.raises(CompileRefused, match="no 'path'"):
        compile_fixture(spatial_refs={"s1": {"fps": 12}})


def test_a_scene_file_changed_after_the_lock_is_caught_on_rederive(tmp_path):
    p = tmp_path / "scene.gltf"
    p.write_text(json.dumps(_gltf_doc()))
    _lock, specs = compile_fixture(spatial_refs={"s1": str(p)})
    ref = specs[0].spatial_ref
    from hugpy_oracle import segments as seg_mod
    seg_mod._SPATIAL_REGISTRY.clear()
    assert spatial_source_for(ref, "s1").source_sha256 in ref       # rederived, same file
    seg_mod._SPATIAL_REGISTRY.clear()
    doc = _gltf_doc()
    doc["nodes"][1]["translation"] = [9, 9, 9]
    p.write_text(json.dumps(doc))
    with pytest.raises(CompileRefused, match="changed after the lock"):
        spatial_source_for(ref, "s1")


def test_segment_context_expands_the_ref_into_the_fold_one_payload(gltf_path):
    from hugpy_oracle.performance import segment_context
    _lock, specs = compile_fixture(spatial_refs={"s1": gltf_path})
    ctx = segment_context(specs[0], None)
    assert ctx["spatial_ref"] == specs[0].spatial_ref
    assert isinstance(ctx["spatial_manifest"], dict) and ctx["spatial_manifest"]["camera_track"]
    assert segment_context(specs[2], None)["spatial_manifest"] is None


# --------------------------------------------------------------------------- #
# [4] structured difficulty signals
# --------------------------------------------------------------------------- #

PROSE = {
    "segment_id": "s1", "characters": ["Alex", "Sam"],
    "blocking": "Sam throws the bottle; Alex catches it behind the dumpster, cloak flaring.",
    "camera": {"movement": "handheld tracking, whip pan"}, "duration_s": 6.0, "tone": 5.0,
}


def test_regex_path_is_labelled_as_fallback():
    sig = extract_signals(PROSE)
    assert sig.source == "regex_fallback" and not sig.has_spatial_manifest
    assert set(sig.guessed) >= {"camera_motion", "props_with_momentum", "occlusion", "cloth_or_hair"}
    assert sig.provenance["characters"] == "declared" and sig.provenance["duration_s"] == "declared"
    assert sig.to_dict()["source"] == "regex_fallback" and "guessed" in sig.to_dict()


def test_structured_signals_come_from_the_manifest_not_the_prose(gltf_path):
    src = ss.from_gltf(gltf_path, run_id="r", segment_id="s1", fps=12)
    quiet = {"segment_id": "s1", "characters": ["Hero"], "blocking": "Hero stands still.", "camera": {}}
    sig = extract_signals(quiet, manifest=src)
    assert sig.source == "structured" and sig.has_spatial_manifest
    assert sig.camera_motion == "simple"              # 2 m dolly, no turn, no zoom
    assert sig.props_with_momentum == 1               # the Ball flies; the Cape and rig stay put
    assert sig.moving_characters == 0                 # Hero's track is static
    assert sig.cloth_or_hair is True                  # Cape tagged cloth
    assert sig.provenance["occlusion"] == "structured" and sig.occlusion is False
    assert sig.guessed == ()
    # the same prose, regex-only, would have said none of this
    r = extract_signals(quiet)
    assert r.camera_motion == "static" and r.props_with_momentum == 0 and not r.cloth_or_hair


def test_occlusion_is_read_from_depth_ordering():
    frames = []
    for i in range(5):
        frames.append({"frame": i, "entities": {
            "near": {"position": [0, 0, 2]}, "far": {"position": [0, 0, 8]}},
            "camera": {"position": [0, 0, -5], "forward": [0, 0, 1]}})
    src = ss.from_pose_track(frames, run_id="r", segment_id="s")
    sig = extract_signals({"segment_id": "s", "blocking": "two crates"}, manifest=src.to_dict())
    assert sig.occlusion is True and sig.provenance["occlusion"] == "structured"
    assert sig.camera_motion == "static" and sig.provenance["camera_motion"] == "structured"
    side = [{"frame": i, "entities": {"a": {"position": [-3, 0, 2]}, "b": {"position": [3, 0, 2]}},
             "camera": {"position": [0, 0, -5], "forward": [0, 0, 1]}} for i in range(3)]
    assert extract_signals({"segment_id": "s"}, manifest=ss.from_pose_track(side, run_id="r", segment_id="s")).occlusion is False


def test_camera_motion_from_keyframes_distinguishes_static_simple_complex():
    def cam_only(keys):
        return {"camera_track": keys, "entity_tracks": [], "entities": []}
    still = [{"position": [0, 0, 0], "forward": [0, 0, -1]}] * 3
    dolly = [{"position": [i, 0, 0], "forward": [0, 0, -1]} for i in range(3)]
    dolly_pan = [{"position": [i, 0, 0], "forward": [i * 0.2, 0, -1]} for i in range(3)]
    assert extract_signals({}, manifest=cam_only(still)).camera_motion == "static"
    assert extract_signals({}, manifest=cam_only(dolly)).camera_motion == "simple"
    assert extract_signals({}, manifest=cam_only(dolly_pan)).camera_motion == "complex"


def test_shot_plan_declared_moves_are_structured_and_mixed_is_reported():
    sig = extract_signals(PROSE, shot={"camera": {"moves": ["dolly", "pan"]}})
    assert sig.camera_motion == "complex" and sig.provenance["camera_motion"] == "structured"
    assert sig.source == "mixed" and "camera_motion" not in sig.guessed
    static = extract_signals(PROSE, shot={"camera": {"move": "static"}})
    assert static.camera_motion == "static" and static.provenance["camera_motion"] == "structured"


def test_compile_context_passes_shot_and_manifest_through(gltf_path):
    src = ss.from_gltf(gltf_path, run_id="r", segment_id="s1", fps=12)
    seg = {"segment_id": "s1", "characters": ["Hero"], "blocking": "Hero stands still.",
           "scene": "a ball arcs past", "identity_constraints": "Hero: hero_pack"}
    plan = compile_context(seg, manifest=src, model_context_tokens=4096)
    assert plan.signals.source == "structured"
    assert any(r.startswith("signals: structured") for r in plan.reasons)
    assert not any("no spatial manifest" in r for r in plan.reasons)
    spatial = next(s for s in plan.sections if s.name == "spatial")
    assert "camera simple" in spatial.content and "prop Ball moves" in spatial.content
    assert "spatial_manifest" in plan.sources
    assert "physics" in {v.angle for v in plan.variants} or plan.candidates == 1
    d, why = difficulty_score(plan.signals)
    assert not any("NO spatial manifest" in w for w in why)
    fallback = compile_context(PROSE, model_context_tokens=4096)
    assert fallback.signals.source == "regex_fallback"
    assert any("regex-guessed" in r for r in fallback.reasons)
    json.dumps(plan.to_dict())
