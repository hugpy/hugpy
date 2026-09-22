"""k119 (or-k7) — spatial evaluators on synthetic geometry. No GPU, no disk.

Locks:
  [1] a perfect render passes every metric and emits no code.
  [2] each injected fault fails exactly the metric that measures it and emits
      the ONE repair code repair_controller routes (GEOMETRY_DRIFT /
      CAMERA_PATH_MISMATCH / COLLISION_VIOLATION) with frame-level evidence.
  [3] a missing observation is "not measured", never "passed".
  [4] thresholds come from the manifest's DriftThresholds, units from its
      CoordinateSystem; the registry names every metric for the gate.
  [5] Track D benchmark cases run in-process and score themselves.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_spatial_eval.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from hugpy_oracle import benchmark_cases, spatial_eval as se
from hugpy_oracle.contracts import CheckKind, RepairCode
from hugpy_oracle.repair_controller import POLICY
from hugpy_oracle.spatial import (
    CameraIntrinsics,
    CameraSpec,
    ConditioningSpec,
    CoordinateSystem,
    DriftThresholds,
    EntitySpec,
    ProvenanceSpec,
    RenderSpec,
    SpatialSceneManifest,
    StyleSpec,
    TierProfile,
    Timebase,
)


@pytest.fixture(scope="module")
def scene():
    return se.synthetic_scene(frames=6)


@pytest.fixture(scope="module")
def manifest(scene):
    return se.synthetic_manifest_stub(scene["intrinsics"])


# --------------------------------------------------------------------------- #
# [1] clean control
# --------------------------------------------------------------------------- #


def test_perfect_render_passes_everything(scene, manifest):
    report = se.evaluate_spatial(manifest, scene)
    assert report.ok
    assert report.codes == ()
    assert report.skipped == ()
    assert set(report.measured) == set(se.spatial_rubrics())
    for m in report.metrics:
        assert m["ok"] is True, m
        assert m["code"] is None


def test_project_points_matches_camera_intrinsics_project():
    intr = CameraIntrinsics(50.0, 50.0, 32.0, 24.0, 64, 48)
    pts = np.array([[0.3, -0.2, -2.0], [1.0, 1.0, -5.0], [0.0, 0.0, 1.0]])
    px = se.project_points(intr, pts)
    for i, p in enumerate(pts[:2]):
        ref = intr.project(p)
        assert px[i, 0] == pytest.approx(ref["u"])
        assert px[i, 1] == pytest.approx(ref["v"])
    assert np.isnan(px[2]).all()          # behind the camera: not projectable


def test_pose_positions_recovers_camera_centre():
    pose = np.eye(4)
    pose[:3, 3] = [-1.0, 2.0, -3.0]       # t = -R C with R = I → C = (1,-2,3)
    assert np.allclose(se.pose_positions(pose[None])[0], [1.0, -2.0, 3.0])
    with pytest.raises(ValueError):
        se.pose_positions(np.zeros((3, 5)))


# --------------------------------------------------------------------------- #
# [2] each fault → its metric → its code, with evidence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind,magnitude,metric,code", [
    ("landmark_shift", 10.0, "reprojection_px", RepairCode.GEOMETRY_DRIFT),
    ("silhouette_erode", 6.0, "silhouette_iou", RepairCode.GEOMETRY_DRIFT),
    ("depth_scale", 0.2, "depth_rel_error", RepairCode.GEOMETRY_DRIFT),
    ("normal_tilt", 20.0, "normal_angle_deg", RepairCode.GEOMETRY_DRIFT),
    ("flow_shift", 5.0, "flow_warp_error", RepairCode.GEOMETRY_DRIFT),
    ("camera_offset", 0.2, "camera_drift_m", RepairCode.CAMERA_PATH_MISMATCH),
    ("camera_yaw", 5.0, "camera_drift_m", RepairCode.CAMERA_PATH_MISMATCH),
    ("sink_into_ground", 0.1, "collision", RepairCode.COLLISION_VIOLATION),
    ("float_off_ground", 0.1, "collision", RepairCode.COLLISION_VIOLATION),
])
def test_each_fault_fails_its_metric_only(scene, manifest, kind, magnitude, metric, code):
    report = se.evaluate_spatial(manifest, se.perturb(scene, kind, magnitude))
    assert not report.ok
    assert report.codes == (code,)
    failed = [m["metric"] for m in report.failures]
    assert failed == [metric]
    assert code in POLICY          # repair_controller can route it


def test_small_faults_stay_within_threshold(scene, manifest):
    for kind, magnitude in [("landmark_shift", 2.0), ("depth_scale", 0.03),
                            ("normal_tilt", 5.0), ("camera_offset", 0.02),
                            ("camera_yaw", 1.0), ("sink_into_ground", 0.01)]:
        report = se.evaluate_spatial(manifest, se.perturb(scene, kind, magnitude))
        assert report.ok, (kind, report.failures)


def test_reprojection_evidence_names_worst_frames_and_counts(scene, manifest):
    obs = dict(scene)
    lm = np.array(scene["landmarks_expected"]).copy()
    lm[3] += np.array([60.0, 0.0])         # frame 3 is the bad one (mean 10 px)
    obs["landmarks_observed"] = lm
    m = se.reprojection_error(manifest, obs)
    assert m["ok"] is False
    assert m["evidence"]["worst_frames"][0] == 3
    assert m["evidence"]["landmarks_over"] == lm.shape[1]
    assert m["evidence"]["max_px"] == pytest.approx(60.0)


def test_reprojection_derives_expected_from_points_world(scene, manifest):
    obs = {"points_world": scene["points_world"],
           "poses_observed": scene["poses_observed"],
           "landmarks_observed": scene["landmarks_observed"]}
    assert se.reprojection_error(manifest, obs)["value"] == pytest.approx(0.0)
    obs["landmarks_observed"] = np.array(scene["landmarks_observed"]) + 8.0
    assert se.reprojection_error(manifest, obs)["code"] is RepairCode.GEOMETRY_DRIFT


def test_camera_drift_reports_rotation_and_first_frame_over(scene, manifest):
    p = np.array(scene["poses_expected"]).copy()
    p[4:, 0, 3] += 0.5                     # drift starts at frame 4
    m = se.camera_drift(manifest, {"poses_expected": scene["poses_expected"],
                                   "poses_observed": p})
    assert m["code"] is RepairCode.CAMERA_PATH_MISMATCH
    assert m["evidence"]["first_frame_over"] == 4
    assert m["evidence"]["max_rotation_deg"] == pytest.approx(0.0)
    assert m["value"] == pytest.approx(0.5)


def test_camera_drift_accepts_bare_positions(manifest):
    e = np.zeros((5, 3))
    o = np.zeros((5, 3))
    o[:, 2] = 0.01
    m = se.camera_drift(manifest, {"poses_expected": e, "poses_observed": o})
    assert m["ok"] is True and m["evidence"]["max_rotation_deg"] is None


def test_collision_evidence_lists_entities_and_frames(scene, manifest):
    m = se.collision_check(manifest, se.perturb(scene, "sink_into_ground", 0.1))
    ev = m["evidence"]
    assert ev["entities"] == ["ground", "hero"]
    assert ev["frames"] == list(range(6))
    assert all(v["reason"] == "penetration" for v in ev["violations"])
    m2 = se.collision_check(manifest, se.perturb(scene, "float_off_ground", 0.1))
    assert all(v["reason"] == "floating" for v in m2["evidence"]["violations"])


def test_collision_without_contacts_checks_every_pair_for_penetration(manifest):
    a = np.zeros((3, 2, 3)); a[:, 1] = 1.0
    b = np.zeros((3, 2, 3)); b[:, 0] = 2.0; b[:, 1] = 3.0
    obs = {"entity_bounds_observed": {"a": a, "b": b}}
    assert se.collision_check(manifest, obs)["ok"] is True
    b[2, 0] = 0.5                          # frame 2: b overlaps a
    assert se.collision_check(manifest, obs)["evidence"]["frames"] == [2]


def test_silhouette_empty_vs_empty_is_agreement(manifest):
    z = np.zeros((2, 8, 8), dtype=bool)
    assert se.silhouette_iou(manifest, {"silhouette_expected": z,
                                        "silhouette_observed": z})["value"] == 1.0


def test_depth_uses_median_so_edge_pixels_do_not_fail_the_gate(manifest):
    e = np.full((1, 10, 10), 2.0)
    o = e.copy()
    o[0, 0, :3] = 10.0                     # three wild edge pixels
    m = se.depth_consistency(manifest, {"depth_expected": e, "depth_observed": o})
    assert m["ok"] is True and m["evidence"]["p90"] == pytest.approx(0.0)


def test_flow_warp_photometric_form_catches_wrong_flow(scene, manifest):
    good = se.flow_warp_error(manifest, {"frames": scene["frames"], "flow": scene["flow"]})
    bad = se.flow_warp_error(manifest, se.perturb(scene, "flow_shift", 6.0))
    assert good["evidence"]["form"] == "photometric" and good["ok"]
    assert bad["ok"] is False and bad["value"] > good["value"]


def test_flow_endpoint_form_when_no_frames(scene, manifest):
    fe = np.array(scene["flow_expected"])
    m = se.flow_warp_error(manifest, {"flow_expected": fe, "flow_observed": fe + 20.0})
    assert m["evidence"]["form"] == "endpoint"
    assert m["code"] is RepairCode.GEOMETRY_DRIFT


# --------------------------------------------------------------------------- #
# [3] unmeasured is not passed
# --------------------------------------------------------------------------- #


def test_missing_observations_are_skipped_not_passed(manifest):
    report = se.evaluate_spatial(manifest, {})
    assert not report.ok
    assert report.measured == ()
    assert len(report.skipped) == len(se.spatial_rubrics())
    for m in report.metrics:
        assert m["ok"] is None and m["skipped"] and m["code"] is None
    for check in report.checks():
        assert check.passed is False and "not measured" in check.detail


def test_shape_mismatch_is_a_skip_with_reason(manifest):
    m = se.silhouette_iou(manifest, {"silhouette_expected": np.zeros((4, 4)),
                                     "silhouette_observed": np.zeros((5, 5))})
    assert m["ok"] is None and "differ" in m["skipped"]


def test_partial_observations_measure_what_they_can(scene, manifest):
    obs = {"poses_expected": scene["poses_expected"], "poses_observed": scene["poses_observed"]}
    report = se.evaluate_spatial(manifest, obs)
    assert report.measured == ("camera_drift_m",)
    assert report.ok                      # the one measured metric passed


# --------------------------------------------------------------------------- #
# [4] thresholds / units from the manifest; registry; Check rows
# --------------------------------------------------------------------------- #


def _full_manifest(thresholds: DriftThresholds, units: str = "meters") -> SpatialSceneManifest:
    intr = CameraIntrinsics(60.0, 60.0, 32.0, 24.0, 64, 48)
    return SpatialSceneManifest(
        run_id="r1", segment_id="s1-1", artifact_revision=1,
        tier_profile=TierProfile(), timebase=Timebase(24.0, 0, 5, 0.25),
        coordinate_system=CoordinateSystem(world_units=units),
        camera=CameraSpec("mem://track", intrinsics=intr),
        entities=(EntitySpec("hero", "character"),),
        conditioning=ConditioningSpec(requested_passes=(), output_uri="mem://cond"),
        style=StyleSpec(), render=RenderSpec(64, 48, 7),
        provenance=ProvenanceSpec("snap", 1, 1, 1),
        thresholds=thresholds)


def test_thresholds_come_from_the_manifest(scene):
    strict = _full_manifest(DriftThresholds(landmark_reprojection_px=0.5))
    report = se.evaluate_spatial(strict, se.perturb(scene, "landmark_shift", 1.0))
    assert report.codes == (RepairCode.GEOMETRY_DRIFT,)
    assert report.segment_id == "s1-1"
    loose = _full_manifest(DriftThresholds(landmark_reprojection_px=50.0))
    assert se.evaluate_spatial(loose, se.perturb(scene, "landmark_shift", 10.0)).ok
    override = se.evaluate_spatial(loose, se.perturb(scene, "landmark_shift", 10.0),
                                   thresholds=DriftThresholds())
    assert override.codes == (RepairCode.GEOMETRY_DRIFT,)


def test_world_units_scale_camera_drift_to_metres(scene):
    cm = _full_manifest(DriftThresholds(), units="centimeters")
    obs = {"poses_expected": np.zeros((3, 3)), "poses_observed": np.full((3, 3), 1.0)}
    m = se.camera_drift(cm, obs)
    assert m["value"] == pytest.approx(np.sqrt(3) / 100.0)
    assert m["ok"] is True
    assert se.camera_drift(_full_manifest(DriftThresholds()), obs)["ok"] is False


def test_registry_names_every_metric_and_code_table_is_closed():
    rubrics = se.spatial_rubrics()
    assert list(rubrics) == ["reprojection_px", "silhouette_iou", "depth_rel_error",
                             "normal_angle_deg", "flow_warp_error", "camera_drift_m",
                             "collision"]
    assert set(rubrics) == set(se.METRIC_CODES)
    assert set(se.METRIC_CODES.values()) == {RepairCode.GEOMETRY_DRIFT,
                                             RepairCode.CAMERA_PATH_MISMATCH,
                                             RepairCode.COLLISION_VIOLATION}
    for fn in rubrics.values():
        assert callable(fn)


def test_checks_and_to_dict_are_json_safe(scene, manifest):
    import json
    report = se.evaluate_spatial(manifest, se.perturb(scene, "camera_offset", 0.3))
    checks = report.checks()
    by_name = {c.name: c for c in checks}
    assert by_name["spatial.camera_drift_m"].passed is False
    assert "camera_path_mismatch" in by_name["spatial.camera_drift_m"].detail
    assert by_name["spatial.flow_warp_error"].kind is CheckKind.TEMPORAL
    assert by_name["spatial.reprojection_px"].kind is CheckKind.TECHNICAL
    d = report.to_dict()
    json.dumps(d)
    assert d["codes"] == ["camera_path_mismatch"]
    assert d["segment_id"] == "synthetic"


def test_report_codes_are_distinct_and_in_gate_order(scene, manifest):
    obs = se.perturb(se.perturb(se.perturb(scene, "landmark_shift", 10.0),
                                "camera_offset", 0.3), "sink_into_ground", 0.2)
    report = se.evaluate_spatial(manifest, obs)
    assert report.codes == (RepairCode.GEOMETRY_DRIFT, RepairCode.CAMERA_PATH_MISMATCH,
                            RepairCode.COLLISION_VIOLATION)


def test_unknown_perturbation_is_refused(scene):
    with pytest.raises(KeyError):
        se.perturb(scene, "wobble", 1.0)


# --------------------------------------------------------------------------- #
# [5] Track D benchmark cases
# --------------------------------------------------------------------------- #


def test_track_d_is_registered_like_the_other_tracks():
    assert "D" in benchmark_cases.TRACKS
    assert benchmark_cases.SUITES["D"] is benchmark_cases.TRACK_D
    assert len(benchmark_cases.TRACK_D) >= 6
    assert benchmark_cases.cases_for("D") == benchmark_cases.TRACK_D
    assert benchmark_cases.cases_for("ABC")[-1].track == "C"     # default runner untouched
    for c in benchmark_cases.TRACK_D:
        assert c.operation == "spatial.evaluate"
        assert c.spec.track == "D"
        assert all(e.layer == "deterministic" for e in c.expectations)


def test_every_track_d_case_scores_itself():
    results = [se.run_track_d_case(c) for c in benchmark_cases.TRACK_D]
    failed = [(r["case_id"], r["expected_code"], r["emitted_code"])
              for r in results if not r["passed"]]
    assert failed == []
    assert {r["expected_code"] for r in results} == {
        None, "geometry_drift", "camera_path_mismatch", "collision_violation"}
