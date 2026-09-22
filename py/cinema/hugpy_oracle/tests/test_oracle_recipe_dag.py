"""Unit A — video.performance visual stages on the durable DAG.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_recipe_dag.py -q
"""
from __future__ import annotations

import logging
import os
import sys

logging.disable(logging.INFO)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
for p in (_SRC, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402

from hugpy_oracle import performance as perf
from hugpy_oracle.contracts import RepairCode
from hugpy_oracle.dag_runtime import NodeState, RunJournal, RunState
from hugpy_oracle.plan import NodeKind
from hugpy_oracle.recipes import video_performance as vp

import test_oracle_performance as base  # noqa: E402


@pytest.fixture()
def journal(tmp_path):
    j = RunJournal(os.path.join(str(tmp_path), "dag.sqlite"))
    yield j
    j.close()


def _prep(tmp_path, fakes):
    goal = base.performance_goal()
    prep = perf.run_performance(goal, seams=fakes.seams(tmp_path), stop_after="segments")
    assert prep.gap is None and prep.segments, prep.gap
    return goal, prep


def test_graph_shape_is_sibling_chains_into_assemble(tmp_path):
    fakes = base.Fakes()
    goal, prep = _prep(tmp_path, fakes)
    g = vp.build_visual_graph(tuple(prep.segments), pgoal=goal, seams=fakes.seams(tmp_path),
                              lock_digest=prep.segments[0].lock_digest)
    sids = [s.segment_id for s in prep.segments]
    for sid in sids:
        kf, kfj, clip, clipj = f"kf:{sid}", f"kfjudge:{sid}", f"clip:{sid}", f"clipjudge:{sid}"
        assert g.node(kf).kind is NodeKind.FANOUT and g.node(kf).candidates >= 1
        assert g.node(kfj).kind is NodeKind.JUDGE and g.node(clipj).kind is NodeKind.JUDGE
        # no edge from any node of this chain to another segment's chain
        for e in g.edges:
            if e.src_node in (kf, kfj, clip, clipj):
                assert e.dst_node in (kf, kfj, clip, clipj, vp.ASSEMBLE_NODE)
    assert set(g.predecessors()[vp.ASSEMBLE_NODE]) == {f"clipjudge:{s}" for s in sids}
    g.topological_order()


def test_happy_path_on_the_dag_produces_the_video(tmp_path, journal):
    fakes = base.Fakes()
    goal, prep = _prep(tmp_path, fakes)
    rt, visual = vp.run_visual_stages(prep, goal, fakes.seams(tmp_path), journal=journal, run_id="v1")
    assert visual.ok, visual.to_dict()
    assert visual.video_ref == fakes.video
    assert len(visual.shots) == len(prep.segments)
    assert visual.repairs == ()
    # receipts carry candidate metadata per keyframe
    rec = journal.node("v1", f"kf:{prep.segments[0].segment_id}")
    assert rec.state is NodeState.SUCCEEDED and rec.receipt["candidates"][0]["candidate"] == 0


def test_resume_after_kill_repeats_no_finished_keyframes(tmp_path):
    """Run until the first segment's keyframes + judge are done, 'die',
    resume with fresh fakes: the finished nodes are journal reads."""
    fakes1 = base.Fakes()
    goal, prep = _prep(tmp_path, fakes1)
    db = os.path.join(str(tmp_path), "dag.sqlite")
    j1 = RunJournal(db)
    specs = tuple(prep.segments)
    g = vp.build_visual_graph(specs, pgoal=goal, seams=fakes1.seams(tmp_path),
                              lock_digest=specs[0].lock_digest)
    ex1 = vp.SeamExecutor(specs, pgoal=goal, seams=fakes1.seams(tmp_path), master=prep.audio_master)
    from hugpy_oracle.dag_runtime import DagRuntime
    rt1 = DagRuntime(j1, ex1, evaluator=ex1.evaluate, owner="p1")
    rt1.start(g, "v1")
    for _ in range(3):  # lock gate, kf:s1, kfjudge:s1
        rt1.step("v1")
    done_before = {n for n, r in j1.nodes("v1").items() if r.state is NodeState.SUCCEEDED}
    assert f"kf:{specs[0].segment_id}" in done_before
    gen_before = ex1.calls["gen_image"]
    j1.close()

    fakes2 = base.Fakes()
    j2 = RunJournal(db)
    rt2, visual = vp.resume_visual_stages(prep, goal, fakes2.seams(tmp_path), journal=j2, run_id="v1")
    assert visual.ok, visual.to_dict()
    ex2 = rt2.executor
    # segment 1's keyframes were never regenerated: only the other segments'
    # FANOUTs ran (all N candidates are produced up front, the judge picks)
    n_cand = fakes2.seams(tmp_path).keyframe_candidates
    kf_done_before = sum(1 for n in done_before if n.startswith("kf:"))
    assert kf_done_before >= 1
    assert ex2.calls["gen_image"] == (len(specs) - kf_done_before) * n_cand
    for n in done_before:
        assert j2.node("v1", n).state is NodeState.SUCCEEDED
    assert j2.node("v1", f"kf:{specs[0].segment_id}").receipt is not None
    j2.close()


def test_judge_rejection_repairs_that_segments_producer_only(tmp_path, journal):
    # verdict list consumed across calls: first keyframe judged NO three times
    # (3 candidates of segment 1), everything after passes.
    fakes = base.Fakes(keyframe_verdicts=[{"verdict": "NO", "score": 5, "why": "bad"}] * 3
                       + [{"verdict": "YES", "score": 90, "why": "ok"}])
    goal, prep = _prep(tmp_path, fakes)
    fakes.calls = {k: 0 for k in fakes.calls}  # count only the visual stages
    rt, visual = vp.run_visual_stages(prep, goal, fakes.seams(tmp_path), journal=journal, run_id="v1",
                                      repair_budget=2)
    assert visual.ok, visual.to_dict()
    assert visual.repairs, "a repair plan must have been recorded"
    plan = visual.repairs[0]
    sid = prep.segments[0].segment_id
    assert plan.failed_node == f"kfjudge:{sid}"
    assert plan.root == f"kf:{sid}" and set(plan.path) == {f"kf:{sid}", f"kfjudge:{sid}"}
    # other segments' keyframes were generated exactly once
    n_other = len(prep.segments) - 1
    kf_calls = fakes.calls["gen_image"]
    assert kf_calls == 3 + 3 + n_other * 3  # seg1 twice (3 candidates each), others once
    for s in prep.segments[1:]:
        assert journal.node("v1", f"kf:{s.segment_id}").attempt == 1
    assert journal.node("v1", f"kf:{sid}").attempt == 2
    kinds = [c["kind"] for c in journal.controls("v1")]
    assert "repair" in kinds and "repair_retry" in kinds


def test_unbound_clip_seam_is_a_typed_gap_not_a_crash(tmp_path, journal):
    fakes = base.Fakes()
    goal, prep = _prep(tmp_path, fakes)
    seams = fakes.seams(tmp_path, gen_clip=None)
    rt, visual = vp.run_visual_stages(prep, goal, seams, journal=journal, run_id="v1")
    assert not visual.ok
    sid = prep.segments[0].segment_id
    rec = journal.node("v1", f"clip:{sid}")
    assert rec.state is NodeState.FAILED
    assert rec.repair_code is RepairCode.CAPABILITY_GAP or "SeamUnavailable" in (rec.failure or "")
    # keyframes for every segment still succeeded (siblings independent)
    assert all(journal.node("v1", f"kf:{s.segment_id}").state is NodeState.SUCCEEDED for s in prep.segments)


def test_whole_recipe_entry_point(tmp_path, journal):
    fakes = base.Fakes()
    prep, visual, rt = vp.run_performance_on_dag(base.performance_goal(), seams=fakes.seams(tmp_path),
                                                 journal=journal)
    assert prep.stopped_after == "segments"
    assert visual is not None and visual.ok
    assert visual.run.state is RunState.COMPLETED


def _manifest_for(spec, run_id="run_01", **kw):
    from hugpy_oracle import spatial as sp
    base = dict(
        run_id=run_id, segment_id=spec.segment_id, artifact_revision=1,
        tier_profile=sp.TierProfile(sp.CaptureTier.STATIC_RIG, sp.InferenceTier.DENSE_CONDITIONING, sp.RenderTier.STATIC_STYLE),
        timebase=sp.Timebase(24.0, 0, 23, 1.0), coordinate_system=sp.CANONICAL,
        camera=sp.CameraSpec("artifact://camera/track.npz", intrinsics=sp.CameraIntrinsics(800, 800, 640, 360, 1280, 720)),
        entities=(sp.EntitySpec("ana", "character", "artifact://characters/ana.glb", identity_reference_ids=("ref_ana",)),),
        conditioning=sp.ConditioningSpec((sp.ConditioningPass.DEPTH, sp.ConditioningPass.SILHOUETTE), "artifact://cond/"),
        style=sp.StyleSpec(2.0), render=sp.RenderSpec(1280, 720, 7),
        provenance=sp.ProvenanceSpec("snap", 1, 1, 1),
    )
    base.update(kw)
    return sp.SpatialSceneManifest(**base)


def test_spatial_manifest_gates_keyframes_and_emits_fold2_payload(tmp_path, journal):
    fakes = base.Fakes()
    goal, prep = _prep(tmp_path, fakes)
    specs = tuple(prep.segments)
    manifests = {s.segment_id: _manifest_for(s) for s in specs}
    rt, visual = vp.run_visual_stages(prep, goal, fakes.seams(tmp_path), journal=journal, run_id="v1",
                                      manifests=manifests, production_fps=24.0)
    assert visual.ok, visual.to_dict()
    sid = specs[0].segment_id
    sp_rec = journal.node("v1", f"spatial:{sid}")
    assert sp_rec.state is NodeState.SUCCEEDED
    cond = sp_rec.outputs["conditioning"]
    assert cond["passes"] == ["depth", "silhouette"] and cond["hard_containment"] is True
    assert len(cond["frames"]) == 24 and cond["manifest_digest"] == manifests[sid].digest
    g = journal.graph("v1")
    assert f"spatial:{sid}" in g.predecessors()[f"kf:{sid}"]


def test_rejected_manifest_blocks_that_segments_render_only(tmp_path, journal):
    fakes = base.Fakes()
    goal, prep = _prep(tmp_path, fakes)
    specs = tuple(prep.segments)
    manifests = {s.segment_id: _manifest_for(s) for s in specs}
    bad_sid = specs[0].segment_id
    manifests[bad_sid] = _manifest_for(specs[0], timebase=__import__("hugpy_oracle.spatial", fromlist=["Timebase"]).Timebase(30.0, 0, 29, 1.0))
    rt, visual = vp.run_visual_stages(prep, goal, fakes.seams(tmp_path), journal=journal, run_id="v1",
                                      manifests=manifests, production_fps=24.0)
    assert not visual.ok
    rec = journal.node("v1", f"spatial:{bad_sid}")
    assert rec.state is NodeState.FAILED and "frame_rate_mismatch" in (rec.failure or "")
    assert rec.repair_code is RepairCode.CAPABILITY_GAP
    assert journal.node("v1", f"kf:{bad_sid}").state is NodeState.PENDING   # never rendered unconstrained
    for s in specs[1:]:
        assert journal.node("v1", f"clipjudge:{s.segment_id}").state is NodeState.SUCCEEDED
