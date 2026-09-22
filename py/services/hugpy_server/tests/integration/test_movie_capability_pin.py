"""PER-SEGMENT capability resolution for a studio movie (k58).

The live failure this suite locks (operator movie eb9dee56, 2026-07-31): a 2-segment
movie pinned ``wan2.1-t2v-1.3b``; segment 0 (t2v) rendered on the 3090 and segment 1 —
a plain "still" splice, i.e. capability ``i2v`` — then died mid-movie with
``[pinned_model_unavailable] pinned model 'wan2.1-t2v-1.3b' does not serve capability
'i2v'``, a refusal that itself listed ``clip-i2v-480p`` as available. GPU minutes were
spent on a failure that was fully knowable at submit.

The three rulings under test:
  * A MOVIE-LEVEL pin binds ONLY the segments whose capability it serves; the others
    resolve their own capable model and the substitution is ATTRIBUTED (per-segment
    ``model_id`` / ``model_source`` / ``pinned_model_id`` / ``model_note`` in the job
    result + the live progress blob, which is what the bus stage log renders).
  * SUBMIT-TIME PREFLIGHT over the whole take-tree: a capability-class failure refuses
    at POST with an explicit per-segment envelope, and NOTHING is enqueued.
  * An EXPLICIT per-segment ``model_id`` is never substituted — it refuses at submit,
    naming that segment.

Same script style as the other studio suites (plain python, ``__main__`` guard,
numbered ``[n] PASS`` / ``[n] FAIL`` lines, nonzero exit iff any check FAILED). The
checks are also plain ``test_*`` functions, so ``venv/bin/python -m pytest`` collects
them. No GPU and no real render: the render seam is faked and the capability tables
are monkeypatched.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_movie_capability_pin.py
  venv/bin/python -m pytest tests/studio/test_movie_capability_pin.py
"""
from __future__ import annotations

import atexit
import logging
import os
import sqlite3
import sys
import tempfile
from types import SimpleNamespace

logging.disable(logging.WARNING)  # silence the registry discovery chatter
os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")
os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="movie-cap-test-"))
# These checks never delegate: keep an ambient studio worker out of the fake render.
os.environ.pop("HUGPY_STUDIO_WORKER", None)
os.environ.pop("HUGPY_STUDIO_FORCE_REMOTE", None)

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import importlib  # noqa: E402

from flask import Flask  # noqa: E402

from hugpy_video.intel import media_bus
from hugpy_video.intel.runners import studio_movie
from hugpy_video.intel.studio import movie_plan
from hugpy_video.intel.studio.enums import Capability
from hugpy_video.intel.studio.router import capable_model_ids
from hugpy_video.intel.studio_movie_schema import StudioMovieGoal, make_studio_movie

_T2V_PIN = "wan2.1-t2v-1.3b"      # serves t2v ONLY — the pin from the live incident
_I2V_MODEL = "wan2.1-i2v-14b-720p"  # serves i2v / id_lock / keyframe

# --------------------------------------------------------------------------- #
# Isolation: a TEMP media bus, so a route enqueue never touches the real catalog
# (and so "nothing was enqueued" is a countable assertion).
# --------------------------------------------------------------------------- #
_TMP_DB = tempfile.mkstemp(prefix="movie-cap-bus-", suffix=".db")[1]
media_bus.DB_PATH = _TMP_DB
media_bus._initialized = False
with sqlite3.connect(_TMP_DB) as _c:
    _c.execute(
        "CREATE TABLE IF NOT EXISTS media_jobs (job_id TEXT PRIMARY KEY, name TEXT, "
        "status TEXT, spec_json TEXT, result_json TEXT, claim_token TEXT, "
        "created REAL, updated REAL, progress_json TEXT)")


@atexit.register
def _cleanup():
    try:
        os.remove(_TMP_DB)
    except OSError:
        pass


vr = importlib.import_module("hugpy_server.app.routes.video_routes")
app = Flask(__name__)
app.register_blueprint(vr.video_bp)
client = app.test_client()


def _job_count() -> int:
    with sqlite3.connect(_TMP_DB) as c:
        return c.execute("SELECT COUNT(*) FROM media_jobs").fetchone()[0]


def _mixed_movie(model_id=None, seg1_model_id=None, out_root=None):
    """The incident's shape: segment 0 t2v, segment 1 a "still" splice -> i2v."""
    return make_studio_movie(
        goals=(StudioMovieGoal("seg_00", "a lighthouse at dusk", None),
               StudioMovieGoal("seg_01", "the beam sweeps the bay", "seg_00",
                               model_id=seg1_model_id)),
        width=832, height=480, fps=16, vram_budget_gb=9.0,
        model_id=model_id, out_root=out_root)


# --------------------------------------------------------------------------- #
# [1] The capability of each segment is DERIVED from the spec alone — one
#     definition, shared by the preflight and the runner.
# --------------------------------------------------------------------------- #
def test_segment_capability_derivation():
    spec = _mixed_movie()
    assert movie_plan.segment_capability(spec, spec.goals[0], 0) == "t2v", "root t2v"
    assert movie_plan.segment_capability(spec, spec.goals[1], 1) == "i2v", "still -> i2v"

    cut = make_studio_movie(
        goals=(StudioMovieGoal("seg_00", "a", None),
               StudioMovieGoal("seg_01", "b", "seg_00", joint_mode="cut"),
               StudioMovieGoal("seg_02", "c", "seg_01", joint_mode="vace_extend")),
        width=832, height=480, fps=16)
    assert movie_plan.segment_capability(cut, cut.goals[1], 1) == "t2v", "cut -> t2v"
    assert movie_plan.segment_capability(cut, cut.goals[2], 2) == "v2v", "vace -> v2v"

    idm = make_studio_movie(
        goals=(StudioMovieGoal("seg_00", "a", None),
               StudioMovieGoal("seg_01", "b", "seg_00", joint_mode="cut")),
        width=832, height=480, fps=16, reference_images=("ref.png",))
    assert [movie_plan.segment_capability(idm, g, i)
            for i, g in enumerate(idm.goals)] == ["id_lock", "id_lock"], "id movie"


# --------------------------------------------------------------------------- #
# [2] The movie-level pin binds ONLY the segment whose capability it serves; the
#     other resolves its own model, ATTRIBUTED (never a silent swap).
# --------------------------------------------------------------------------- #
def test_movie_pin_binds_only_what_it_serves():
    plans = movie_plan.plan_segments(_mixed_movie(model_id=_T2V_PIN))
    assert plans[0].model.model_id == _T2V_PIN, plans[0]
    assert plans[0].model.source == movie_plan.SOURCE_MOVIE, plans[0]
    assert plans[1].model.model_id is None, plans[1]           # unpinned -> router picks
    assert plans[1].model.source == movie_plan.SOURCE_FALLBACK, plans[1]
    assert plans[1].model.pinned_model_id == _T2V_PIN, plans[1]
    assert "i2v" in (plans[1].model.note or ""), plans[1].model.note
    rec = plans[1].model.as_record()
    assert rec["model_source"] == "capability_fallback" and rec["model_note"], rec


# --------------------------------------------------------------------------- #
# [3] That movie is ACCEPTED (a partially-serving pin is a legal request, not an
#     error) — the whole point of ruling 1.
# --------------------------------------------------------------------------- #
def test_route_accepts_partially_serving_pin():
    r = client.post("/video/studio/movie", json={
        "resolution": {"width": 832, "height": 480, "fps": 16},
        "vram_budget_gb": 9.0, "model_id": _T2V_PIN,
        "goals": [{"prompt": "a lighthouse at dusk"},
                  {"prompt": "the beam sweeps the bay"}]})
    assert r.status_code == 200, (r.status_code, r.get_json())
    assert isinstance(r.get_json().get("job_id"), str), r.get_json()


# --------------------------------------------------------------------------- #
# [4] RUNNER: segment 0 renders under the pin, segment 1 renders UNPINNED, and the
#     per-segment records carry the attribution (this is the movie that used to die
#     mid-render on segment 1).
# --------------------------------------------------------------------------- #
def _run_movie_capture(spec):
    """Run the movie with the render seam + assembly stubbed; return
    (JobResult, [seg_spec.model_id per segment], seg_records, [progress blobs])."""
    pins: list = []
    records: list = []
    blobs: list = []

    def _fake_render_clip(seg_spec, render_id=None, should_cancel=None,
                          progress_sink=None, produce=None):
        pins.append(seg_spec.model_id)
        return SimpleNamespace(
            ok=True, error=None, path=f"/tmp/{render_id}.mp4",
            frames=8, width=seg_spec.width, height=seg_spec.height, duration_s=1.0,
            content_hash="deadbeef", resumed=False,
            effective_budget_gb=seg_spec.vram_budget_gb, budget_source="explicit")

    def _fake_write_movie_json(movie_root, spec_, seg_records, assembly, job_id, partial):
        records[:] = [dict(r) for r in seg_records]
        return {"assembly": {"movie": None}}

    orig = (studio_movie.render_clip, studio_movie.ingest,
            studio_movie._assemble_movie, studio_movie._write_movie_json,
            studio_movie._extract_frame_at,
            media_bus.is_cancelling, media_bus.set_progress)
    studio_movie.render_clip = _fake_render_clip
    studio_movie.ingest = lambda path, kind_hint=None: SimpleNamespace(
        uri=os.path.abspath(str(path)), kind=(kind_hint or "video"))
    studio_movie._assemble_movie = lambda *a, **k: {"movie": None}
    studio_movie._write_movie_json = _fake_write_movie_json
    # the "still" splice plucks a branch frame with ffmpeg — the fake clips have no
    # pixels, so the pluck is stubbed OK (this suite is about model resolution).
    studio_movie._extract_frame_at = lambda *a, **k: (True, "")
    media_bus.is_cancelling = lambda _job_id: False
    media_bus.set_progress = lambda _job_id, blob: blobs.append(blob)
    try:
        res = studio_movie.run_generate_studio_movie(spec, "job_cap_pin_test")
    finally:
        (studio_movie.render_clip, studio_movie.ingest,
         studio_movie._assemble_movie, studio_movie._write_movie_json,
         studio_movie._extract_frame_at,
         media_bus.is_cancelling, media_bus.set_progress) = orig
    return res, pins, records, blobs


def test_runner_substitutes_only_the_unserved_segment():
    tmp = tempfile.mkdtemp(prefix="movie-cap-runner-")
    res, pins, records, blobs = _run_movie_capture(_mixed_movie(model_id=_T2V_PIN,
                                                                out_root=tmp))
    assert res.ok, getattr(res, "error", None)
    assert pins == [_T2V_PIN, None], pins          # seg1 is NOT pinned to the t2v model
    assert [r["capability"] for r in records] == ["t2v", "i2v"], records
    assert records[0]["model_source"] == "movie", records[0]
    assert records[0]["model_id"] == _T2V_PIN, records[0]
    assert records[1]["model_source"] == "capability_fallback", records[1]
    assert records[1]["pinned_model_id"] == _T2V_PIN, records[1]
    assert "i2v" in records[1]["model_note"], records[1]
    # ...and the substitution is visible LIVE (this is what the stage log renders).
    live = [b.get("current") for b in blobs
            if isinstance(b.get("current"), dict)
            and b["current"].get("model_source") == "capability_fallback"]
    assert live, [b.get("current") for b in blobs]
    assert live[0]["pinned_model_id"] == _T2V_PIN, live[0]


# --------------------------------------------------------------------------- #
# [5] An EXPLICIT per-segment model_id that cannot serve its segment refuses AT
#     SUBMIT, naming that segment — never substituted, never enqueued.
# --------------------------------------------------------------------------- #
def test_explicit_segment_pin_refuses_at_submit():
    before = _job_count()
    r = client.post("/video/studio/movie", json={
        "resolution": {"width": 832, "height": 480, "fps": 16},
        "vram_budget_gb": 9.0,
        "goals": [{"prompt": "a lighthouse at dusk"},
                  {"prompt": "the beam sweeps the bay", "model_id": _T2V_PIN}]})
    assert r.status_code == 400, (r.status_code, r.get_json())
    payload = r.get_json()
    assert payload.get("code") == "movie_capability_preflight_failed", payload
    assert "job_id" not in payload, payload
    segs = payload.get("segments") or []
    assert len(segs) == 1, segs
    assert segs[0]["segment_id"] == "seg_01" and segs[0]["index"] == 1, segs[0]
    assert segs[0]["capability"] == "i2v", segs[0]
    assert segs[0]["reason"] == "pinned_model_unavailable", segs[0]
    assert segs[0]["model_id"] == _T2V_PIN, segs[0]
    assert "t2v" in segs[0]["model_capabilities"], segs[0]
    assert _I2V_MODEL in segs[0]["capable_models"], segs[0]
    assert segs[0]["available"], segs[0]           # never "no" without an "instead"
    assert _job_count() == before, "nothing may be enqueued by a refused movie"


# --------------------------------------------------------------------------- #
# [6] An explicit per-segment model_id that is not a model AT ALL is a typo, and a
#     typo is refused (never absorbed by the fallback).
# --------------------------------------------------------------------------- #
def test_unknown_segment_pin_refuses_at_submit():
    before = _job_count()
    r = client.post("/video/studio/movie", json={
        "resolution": {"width": 832, "height": 480, "fps": 16},
        "goals": [{"prompt": "a", "model_id": "no-such-model"}]})
    assert r.status_code == 400, (r.status_code, r.get_json())
    segs = r.get_json().get("segments") or []
    assert segs and segs[0]["reason"] == "pinned_model_unknown", segs
    assert segs[0]["segment_id"] == "seg_00", segs[0]
    assert _job_count() == before, "nothing may be enqueued by a refused movie"


# --------------------------------------------------------------------------- #
# [7] A movie-level pin that is not a model at all: also a typo -> ONE refusal, not
#     one per segment.
# --------------------------------------------------------------------------- #
def test_unknown_movie_pin_refuses_once():
    problems = movie_plan.preflight_movie(_mixed_movie(model_id="no-such-model"))
    assert len(problems) == 1, problems
    assert problems[0]["reason"] == "pinned_model_unknown", problems[0]
    assert problems[0]["model_source"] == "movie", problems[0]


# --------------------------------------------------------------------------- #
# [8] A capability with NO capable model refuses at submit — nothing rendered. The
#     fleet serves every movie capability today, so the registry answer is MOCKED
#     (this is the guarantee, not a claim about today's zoo).
# --------------------------------------------------------------------------- #
def test_no_capable_model_refuses_at_submit():
    before = _job_count()
    orig = movie_plan.capable_model_ids
    movie_plan.capable_model_ids = lambda cap, include_synthetic=False: (
        () if cap is Capability.I2V else orig(cap, include_synthetic))
    try:
        r = client.post("/video/studio/movie", json={
            "resolution": {"width": 832, "height": 480, "fps": 16},
            "goals": [{"prompt": "a lighthouse at dusk"},
                      {"prompt": "the beam sweeps the bay"}]})
    finally:
        movie_plan.capable_model_ids = orig
    assert r.status_code == 400, (r.status_code, r.get_json())
    segs = r.get_json().get("segments") or []
    assert len(segs) == 1 and segs[0]["index"] == 1, segs
    assert segs[0]["reason"] == "no_capable_model", segs[0]
    assert segs[0]["capability"] == "i2v" and segs[0]["capable_models"] == [], segs[0]
    assert _job_count() == before, "nothing may be enqueued by a refused movie"


# --------------------------------------------------------------------------- #
# [9] capable_model_ids answers the EXISTENCE question the preflight asks: real
#     models only (the synthetic last-resort is opt-in and proves nothing), and it
#     agrees with the registry's declared capabilities.
# --------------------------------------------------------------------------- #
def test_capable_model_ids_excludes_synthetic():
    real = capable_model_ids(Capability.T2V)
    assert _T2V_PIN in real, real
    assert not any(m.startswith("synthetic-") for m in real), real
    assert "synthetic-t2v" in capable_model_ids(Capability.T2V, include_synthetic=True)
    assert _I2V_MODEL in capable_model_ids(Capability.I2V), capable_model_ids(Capability.I2V)


# --------------------------------------------------------------------------- #
# [10] An unpinned movie is untouched by all of this (no regression): every segment
#      stays unpinned and the movie is accepted.
# --------------------------------------------------------------------------- #
def test_unpinned_movie_unchanged():
    plans = movie_plan.plan_segments(_mixed_movie())
    assert [p.model.model_id for p in plans] == [None, None], plans
    assert {p.model.source for p in plans} == {"auto"}, plans
    assert movie_plan.preflight_movie(_mixed_movie()) == []
    r = client.post("/video/studio/movie", json={
        "resolution": {"width": 832, "height": 480, "fps": 16},
        "goals": [{"prompt": "a"}, {"prompt": "b"}]})
    assert r.status_code == 200, (r.status_code, r.get_json())


CHECKS = [
    ("per-segment capability is derived from the spec", test_segment_capability_derivation),
    ("movie pin binds only the segments it serves (attributed)",
     test_movie_pin_binds_only_what_it_serves),
    ("route accepts a partially-serving movie pin", test_route_accepts_partially_serving_pin),
    ("runner pins seg0, resolves seg1, records the substitution",
     test_runner_substitutes_only_the_unserved_segment),
    ("explicit per-segment pin that can't serve -> 400 naming the segment",
     test_explicit_segment_pin_refuses_at_submit),
    ("unknown per-segment pin -> 400 (a typo is never absorbed)",
     test_unknown_segment_pin_refuses_at_submit),
    ("unknown movie-level pin -> ONE refusal", test_unknown_movie_pin_refuses_once),
    ("no capable model for a segment -> 400, nothing enqueued",
     test_no_capable_model_refuses_at_submit),
    ("capable_model_ids excludes the synthetic tier", test_capable_model_ids_excludes_synthetic),
    ("an unpinned movie is unchanged", test_unpinned_movie_unchanged),
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
