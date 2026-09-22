"""Backend tests for the "Movie Maker" multi-scene generator (GPU-free).

Script-style with a __main__ guard, like the sibling video tests. The GPU/vision
planes are STUBBED so nothing loads a model or shells out to ffmpeg:

  * movie.render_scene_frames  -> a fake that writes stub frame PNGs into the
    segment out_dir and calls on_frame_done per frame (records the seed +
    start_frame it was handed, so we can assert the deterministic seed bump and
    the cross-segment start-image carry).
  * movie.ingest              -> a fake MediaRef builder (no ffprobe).
  * movie.img2img_available   -> toggled True/False (drift carry vs independent).
  * movie._score_keyframe     -> a scripted score sequence (retry logic).
  * movie._assemble_scene_mp4 / movie._concat_movie -> write dummy files (no ffmpeg).

media_bus.DB_PATH is repointed to a PRIVATE temp db; DEFAULT_ROOT on BOTH the
movie and scene modules is repointed to a temp dir so bundles never land under
the shared storage root.

Covers: make_movie validation (empty/range/contiguity/frame-cap/knobs),
parse_vision_verdict, the orchestrator segment loop (vision-off; vision-on with a
weak take -> retry with a bumped seed; a strong take -> no retry), the
cross-segment start-image carry (+ independent fallback when img2img is
unavailable), the per-segment scene-spec builder (bundle manifest), resume-skip
when a segment bundle already exists, the concat-demux stitch, and the exact
NESTED progress blob shape.

Run:
  /home/ubuntu/station/dev/abstract_hugpy_dev/venv/bin/python tests/test_video_movie.py
"""
import os
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# --------------------------------------------------------------------------- #
# global stub state (reset per test)
# --------------------------------------------------------------------------- #
RENDER_CALLS = []     # one dict per render_scene_frames invocation
SCORE_QUEUE = []      # scripted scores popped by the fake judge
PROGRESS_BLOBS = []   # every blob passed to the (patched) set_progress


def _fake_render(*, out_dir, n_frames, on_frame_done, seed=None, start_frame=None,
                 base_prompt=None, **kw):
    """Stand-in for scene.render_scene_frames: write n stub frames + drive the
    injected on_frame_done. Records the seed + start_frame it was handed."""
    RENDER_CALLS.append({
        "seed": seed, "start_frame": start_frame, "n_frames": n_frames,
        "base_prompt": base_prompt, "out_dir": out_dir,
        # k92 — capture the effective per-goal knobs so tests can assert overrides
        # actually reach the segment render (extras ride **kw).
        "model_id": kw.get("model_id"), "width": kw.get("width"),
        "height": kw.get("height"), "steps": kw.get("steps"),
        "guidance": kw.get("guidance"), "negative": kw.get("negative"),
        "strength": kw.get("strength"), "chain": kw.get("chain"),
        "motion": kw.get("motion"),
    })
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_frames):
        fp = os.path.join(out_dir, f"frame_{i:05d}.png")
        with open(fp, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n" + bytes([i]) * 16)
        on_frame_done(fp, i)
    return None  # success (JobResult error would be returned here)


def _fake_ingest(path):
    from hugpy_video.intel.media_schema import make_media_ref
    kind = "video" if str(path).endswith(".mp4") else "image"
    return make_media_ref(asset_id=uuid4().hex, kind=kind, uri=os.path.abspath(path),
                          mime=("video/mp4" if kind == "video" else "image/png"),
                          width=64, height=64)


def _fake_score(goal, keyframe_uri, judge_model_id):
    score = SCORE_QUEUE.pop(0) if SCORE_QUEUE else None
    return {"verdict": ("YES" if (score or 0) >= 50 else "NO"),
            "score": score, "why": "stubbed", "raw": f"SCORE={score}"}


def _fake_assemble(frame_dir, mp4_path, fps):
    with open(mp4_path, "wb") as fh:
        fh.write(b"\x00\x00\x00\x18ftypmp42")


CONCAT_CALLS = []


def _fake_concat(segment_mp4s, movie_mp4, work_dir):
    CONCAT_CALLS.append(list(segment_mp4s))
    with open(movie_mp4, "wb") as fh:
        fh.write(b"\x00\x00\x00\x18ftypmp42MOVIE")


def _install(tmp, *, can_carry=True, scores=None):
    """Patch the movie + scene modules for a hermetic run. Returns (movie module,
    media_bus module)."""
    from hugpy_video.intel.runners import movie, scene
    from hugpy_video.intel import media_bus

    RENDER_CALLS.clear()
    CONCAT_CALLS.clear()
    PROGRESS_BLOBS.clear()
    SCORE_QUEUE[:] = list(scores or [])

    movie.render_scene_frames = _fake_render
    movie.ingest = _fake_ingest
    movie.img2img_available = lambda m: can_carry
    movie._score_keyframe = _fake_score
    movie._assemble_scene_mp4 = _fake_assemble
    movie._concat_movie = _fake_concat
    movie.DEFAULT_ROOT = tmp
    scene.DEFAULT_ROOT = tmp

    def _capture(job_id, blob):
        import copy
        PROGRESS_BLOBS.append(copy.deepcopy(blob))
    media_bus.set_progress = _capture

    db = os.path.join(tmp, "media_jobs.db")
    media_bus.DB_PATH = db
    media_bus._initialized = False
    return movie, media_bus


def _spec(goals, **over):
    from hugpy_video.intel.movie_schema import make_movie, GoalInterval
    gi = tuple(GoalInterval(*g) if not isinstance(g, GoalInterval) else g for g in goals)
    kw = dict(model_id="sd-turbo", width=64, height=64, steps=2, guidance=0.0,
              fps=8, assemble=False, seed=1000, project="Test Movie")
    kw.update(over)
    return make_movie(goals=gi, **kw)


# --------------------------------------------------------------------------- #
# 1) make_movie validation
# --------------------------------------------------------------------------- #
def test_make_movie_validation():
    from hugpy_video.intel.movie_schema import make_movie, GoalInterval, total_frames

    def _bad(goals, **over):
        try:
            _spec(goals, **over)
        except ValueError:
            return True
        return False

    assert _bad([]), "empty goals must raise"
    assert _bad([(0, 3, "a"), (4, 7, "b")]), "a GAP between goals must raise"
    assert _bad([(0, 4, "a"), (3, 7, "b")]), "an OVERLAP must raise"
    assert _bad([(2, 5, "a")]), "goals not starting at 0 must raise"
    assert _bad([(0, 3, "")]), "empty prompt must raise"
    assert _bad([(0, 0, "a")]), "end<=start must raise"
    assert _bad([(0, 30, "a")]), "a segment over FRAME_CAP must raise"
    assert _bad([(0, 3, "a")], score_threshold=101), "score_threshold>100 must raise"
    assert _bad([(0, 3, "a")], score_threshold=-1), "score_threshold<0 must raise"
    assert _bad([(0, 3, "a")], max_attempts_per_segment=0), "max_attempts<1 must raise"
    assert _bad([(0, 3, "a")], time_budget_s=0), "time_budget_s=0 must raise"

    # k92 — per-goal PROMPT-COMPONENT override invariants (each optional; when
    # present it must satisfy the same rule as the movie-level field it overrides).
    assert _bad([GoalInterval(0, 3, "a", steps=-5)]), "per-goal steps<=0 must raise"
    assert _bad([GoalInterval(0, 3, "a", width=0)]), "per-goal width<=0 must raise"
    assert _bad([GoalInterval(0, 3, "a", strength=2.0)]), "per-goal strength>1 must raise"
    assert _bad([GoalInterval(0, 3, "a", model_id="")]), "empty per-goal model_id must raise"
    assert _bad([GoalInterval(0, 3, "a", chain="yes")]), "non-bool per-goal chain must raise"
    # a fully-valid set of overrides must BUILD (round-trips through the factory)
    ok = _spec([GoalInterval(0, 3, "a", model_id="flux-schnell", width=128, height=128,
                             steps=6, guidance=1.5, seed=9, negative="blur",
                             strength=0.5, chain=False, motion="pan {i}")])
    assert ok.goals[0].model_id == "flux-schnell" and ok.goals[0].motion == "pan {i}", ok.goals[0]

    # valid, contiguous timeline -> total == max(end_frame)
    spec = _spec([(0, 3, "sunrise"), (3, 7, "noon"), (7, 10, "sunset")])
    assert total_frames(spec) == 10, total_frames(spec)
    assert len(spec.goals) == 3
    print("[1] PASS  make_movie validation (empty/gap/overlap/range/cap/knobs) + total_frames")


# --------------------------------------------------------------------------- #
# 2) parse_vision_verdict
# --------------------------------------------------------------------------- #
def test_parse_vision_verdict():
    from hugpy_video.intel.runners.movie import parse_vision_verdict

    v = parse_vision_verdict("VERDICT=YES; SCORE=82; WHY=the sky is orange.")
    assert v == {"verdict": "YES", "score": 82, "why": "the sky is orange"}, v

    v = parse_vision_verdict("blah VERDICT=NO SCORE=5 WHY=empty frame")
    assert v["verdict"] == "NO" and v["score"] == 5 and v["why"] == "empty frame", v

    # clamp + tolerant separators
    v = parse_vision_verdict("VERDICT: yes ; SCORE: 250 ; WHY: overshoot")
    assert v["verdict"] == "YES" and v["score"] == 100, v

    # bare verdict word, no score
    v = parse_vision_verdict("Honestly, no.")
    assert v["verdict"] == "NO" and v["score"] is None, v

    # garbage -> all None/empty (never raises)
    v = parse_vision_verdict("...")
    assert v == {"verdict": None, "score": None, "why": ""}, v
    print("[2] PASS  parse_vision_verdict (fields, clamp, bare word, garbage)")


# --------------------------------------------------------------------------- #
# 3) orchestrator — vision OFF, cross-segment carry
# --------------------------------------------------------------------------- #
def test_orchestrator_vision_off_carry(monkeypatch):
    # Carry is the LEGACY pixel chain: OFF by default (or-k2/or-p1), opt-in only.
    monkeypatch.setenv("HUGPY_LEGACY_CHAIN", "1")
    tmp = tempfile.mkdtemp(prefix="hugpy_test_movie_off_")
    movie, media_bus = _install(tmp, can_carry=True)
    spec = _spec([(0, 3, "a sunrise"), (3, 7, "the sun overhead")])
    job_id = media_bus.enqueue("generate_movie", spec)
    res = movie.run_generate_movie(spec, job_id)

    assert res.ok, res.error
    assert len(RENDER_CALLS) == 2, f"one render per segment: {len(RENDER_CALLS)}"
    # segment 0: no start frame (seg 0 never carries); segment 1: carries seg0's last frame
    assert RENDER_CALLS[0]["start_frame"] is None, RENDER_CALLS[0]
    seg0_last = os.path.join(RENDER_CALLS[0]["out_dir"], "frame_00002.png")
    assert RENDER_CALLS[1]["start_frame"] == seg0_last, \
        f"segment 1 must carry seg0's LAST frame: {RENDER_CALLS[1]['start_frame']} != {seg0_last}"
    # deterministic seeds: base + seg*1000 + attempt(0)
    assert RENDER_CALLS[0]["seed"] == 1000 and RENDER_CALLS[1]["seed"] == 2000, \
        [c["seed"] for c in RENDER_CALLS]
    # manifest DECLARES the legacy chain with the agreed label
    assert res.movie["drift"] == "carry", res.movie["drift"]
    assert res.movie["legacy_pixel_chain"] is True
    assert res.movie["label"] == "legacy (pixel-chained)", res.movie["label"]
    assert res.movie["limitations"], "a chained movie must list its limitation"
    assert [s["chosen_take"] for s in res.movie["segments"]] == [0, 0]
    # outputs = all frames (3 + 4), no mp4 (assemble=False)
    assert len(res.outputs) == 7, len(res.outputs)
    print("[3] PASS  orchestrator vision-off + cross-segment start-image carry + seed bump")


# --------------------------------------------------------------------------- #
# 3b) the DEFAULT: legacy chain OFF — img2img available, but no carry, no
#     per-frame chain, and the manifest says so (board or-k2 / proposal or-p1)
# --------------------------------------------------------------------------- #
def test_orchestrator_legacy_chain_off_by_default(monkeypatch):
    monkeypatch.delenv("HUGPY_LEGACY_CHAIN", raising=False)
    tmp = tempfile.mkdtemp(prefix="hugpy_test_movie_nochain_")
    movie, media_bus = _install(tmp, can_carry=True)
    spec = _spec([(0, 3, "a sunrise"), (3, 7, "the sun overhead")], chain=True)
    job_id = media_bus.enqueue("generate_movie", spec)
    res = movie.run_generate_movie(spec, job_id)

    assert res.ok, res.error
    assert len(RENDER_CALLS) == 2
    # no carry AND the per-frame chain is forced off, even though chain=True was asked for
    assert all(c["start_frame"] is None for c in RENDER_CALLS), [c["start_frame"] for c in RENDER_CALLS]
    assert all(c["chain"] is False for c in RENDER_CALLS), [c["chain"] for c in RENDER_CALLS]
    assert res.movie["legacy_pixel_chain"] is False
    assert res.movie["label"] == "independent", res.movie["label"]
    assert res.movie["limitations"] == []
    assert res.movie["drift"].startswith("independent"), res.movie["drift"]
    assert "HUGPY_LEGACY_CHAIN=1" in res.movie["drift"], res.movie["drift"]
    # the live progress blob carries the same declaration
    assert PROGRESS_BLOBS and all(b["legacy_pixel_chain"] is False and b["label"] == "independent"
                                  for b in PROGRESS_BLOBS)
    # explicit "0"/"off" spellings stay off too
    for off in ("0", "false", "off", ""):
        monkeypatch.setenv("HUGPY_LEGACY_CHAIN", off)
        from hugpy_video.intel.movie_schema import legacy_chain_enabled, effective_chain
        assert legacy_chain_enabled() is False and effective_chain(True) is False, off
    for on in ("1", "true", "ON", " yes "):
        monkeypatch.setenv("HUGPY_LEGACY_CHAIN", on)
        assert legacy_chain_enabled() is True and effective_chain(True) is True, on
        assert effective_chain(False) is False, "opt-in never turns chain ON for a chain=false spec"
    print("[3b] PASS  legacy pixel chain OFF by default; HUGPY_LEGACY_CHAIN=1 opts in")


# --------------------------------------------------------------------------- #
# 4) independent fallback when img2img is unavailable
# --------------------------------------------------------------------------- #
def test_orchestrator_independent_fallback():
    tmp = tempfile.mkdtemp(prefix="hugpy_test_movie_indep_")
    movie, media_bus = _install(tmp, can_carry=False)
    spec = _spec([(0, 2, "a"), (2, 4, "b")])
    job_id = media_bus.enqueue("generate_movie", spec)
    res = movie.run_generate_movie(spec, job_id)

    assert res.ok, res.error
    # NO carry: every segment renders with start_frame None
    assert all(c["start_frame"] is None for c in RENDER_CALLS), \
        [c["start_frame"] for c in RENDER_CALLS]
    assert res.movie["drift"].startswith("independent"), res.movie["drift"]
    print("[4] PASS  independent-segments fallback when image-to-image unavailable (says so)")


# --------------------------------------------------------------------------- #
# 5) vision ON — weak take retries with a bumped seed; strong take does not
# --------------------------------------------------------------------------- #
def test_orchestrator_vision_retry():
    tmp = tempfile.mkdtemp(prefix="hugpy_test_movie_retry_")
    # one segment; first take scores 40 (< 70) -> retry; second take scores 90 -> keep
    movie, media_bus = _install(tmp, can_carry=True, scores=[40, 90])
    spec = _spec([(0, 3, "a crisp mountain vista")],
                 vision_enabled=True, score_threshold=70, max_attempts_per_segment=2)
    job_id = media_bus.enqueue("generate_movie", spec)
    res = movie.run_generate_movie(spec, job_id)

    assert res.ok, res.error
    assert len(RENDER_CALLS) == 2, f"weak take must retry: {len(RENDER_CALLS)} render(s)"
    # bumped seed: attempt0 = 1000, attempt1 = 1001 (base + seg*1000 + attempt)
    assert [c["seed"] for c in RENDER_CALLS] == [1000, 1001], [c["seed"] for c in RENDER_CALLS]
    seg = res.movie["segments"][0]
    assert seg["attempts"] == 2 and seg["scores"] == [40, 90] and seg["chosen_take"] == 1, seg
    print("[5a] PASS orchestrator vision-on: weak take -> retry with bumped seed, best take kept")

    # strong first take -> no retry
    tmp2 = tempfile.mkdtemp(prefix="hugpy_test_movie_noretry_")
    movie, media_bus = _install(tmp2, can_carry=True, scores=[85])
    spec2 = _spec([(0, 3, "a crisp mountain vista")],
                  vision_enabled=True, score_threshold=70, max_attempts_per_segment=2)
    job2 = media_bus.enqueue("generate_movie", spec2)
    res2 = movie.run_generate_movie(spec2, job2)
    assert res2.ok, res2.error
    assert len(RENDER_CALLS) == 1, f"strong take must NOT retry: {len(RENDER_CALLS)}"
    assert res2.movie["segments"][0]["chosen_take"] == 0
    print("[5b] PASS orchestrator vision-on: strong first take -> no retry")


# --------------------------------------------------------------------------- #
# 6) per-segment scene-spec builder -> incremental bundle manifest
# --------------------------------------------------------------------------- #
def test_segment_bundle_builder():
    import json
    from hugpy_platform.utils import slugify
    tmp = tempfile.mkdtemp(prefix="hugpy_test_movie_bundle_")
    movie, media_bus = _install(tmp, can_carry=True)
    spec = _spec([(0, 3, "sunrise over the sea"), (3, 6, "the sun overhead")])
    job_id = media_bus.enqueue("generate_movie", spec)
    res = movie.run_generate_movie(spec, job_id)
    assert res.ok, res.error

    meta = slugify(spec.project)
    b0 = os.path.join(tmp, "assets", meta, "segment_00", "project.json")
    b1 = os.path.join(tmp, "assets", meta, "segment_01", "project.json")
    assert os.path.isfile(b0) and os.path.isfile(b1), "both segment bundles must exist"
    with open(b0) as fh:
        m0 = json.load(fh)
    assert m0["n_frames"] == 3 and m0["prompt"] == "sunrise over the sea", m0
    assert m0["seeds"] == [1000, 1001, 1002], m0["seeds"]
    assert m0["frames"] == ["frame_00000.png", "frame_00001.png", "frame_00002.png"], m0["frames"]
    with open(b1) as fh:
        m1 = json.load(fh)
    assert m1["seeds"] == [2000, 2001, 2002], m1["seeds"]
    # movie.json manifest
    mj = os.path.join(tmp, "assets", meta, "movie.json")
    assert os.path.isfile(mj), "movie.json must be written"
    with open(mj) as fh:
        manifest = json.load(fh)
    # k92: the goals summary now carries each goal's EFFECTIVE model (attribution);
    # here both inherit the movie-level default (sd-turbo) since neither overrides it.
    assert manifest["goals"] == [
        {"start_frame": 0, "end_frame": 3, "prompt": "sunrise over the sea", "model": "sd-turbo"},
        {"start_frame": 3, "end_frame": 6, "prompt": "the sun overhead", "model": "sd-turbo"},
    ], manifest["goals"]
    assert {"goal", "prompt", "seed", "attempts", "scores", "chosen_take"} <= set(manifest["segments"][0])
    # additive iterative-save keys: a fully-rendered movie is NOT partial and its
    # completed count equals the goal total
    assert manifest["partial"] is False, manifest["partial"]
    assert manifest["segments_completed"] == 2, manifest["segments_completed"]
    assert manifest["segments_total"] == 2, manifest["segments_total"]
    print("[6] PASS  per-segment scene-spec builder -> assets/<meta>/segment_NN/project.json + movie.json")


# --------------------------------------------------------------------------- #
# 6b) k92 — PER-GOAL prompt-component overrides reach the segment render
# --------------------------------------------------------------------------- #
def test_per_goal_overrides():
    """A goal that pins its OWN model/steps/seed/negative renders with THOSE values;
    a goal with no overrides inherits the movie-level defaults. Also asserts the
    deterministic seed bump keys off the goal's effective seed, and that movie.json
    attributes each goal's effective model."""
    import json
    from hugpy_platform.utils import slugify
    from hugpy_video.intel.movie_schema import GoalInterval
    tmp = tempfile.mkdtemp(prefix="hugpy_test_movie_pergoal_")
    movie, media_bus = _install(tmp, can_carry=False)
    spec = _spec([
        GoalInterval(0, 2, "wide establishing shot"),  # inherits everything
        GoalInterval(2, 4, "tight close-up", model_id="flux-schnell", steps=9,
                     seed=500, negative="blurry, jpeg", strength=0.3, chain=False),
    ])
    job_id = media_bus.enqueue("generate_movie", spec)
    res = movie.run_generate_movie(spec, job_id)
    assert res.ok, res.error

    # each render saw its goal's EFFECTIVE knobs (goal 0 inherits, goal 1 overrides)
    assert RENDER_CALLS[0]["model_id"] == "sd-turbo" and RENDER_CALLS[0]["steps"] == 2, RENDER_CALLS[0]
    assert RENDER_CALLS[1]["model_id"] == "flux-schnell" and RENDER_CALLS[1]["steps"] == 9, RENDER_CALLS[1]
    assert RENDER_CALLS[1]["negative"] == "blurry, jpeg", RENDER_CALLS[1]
    assert RENDER_CALLS[1]["strength"] == 0.3 and RENDER_CALLS[1]["chain"] is False, RENDER_CALLS[1]
    # deterministic bump: goal0 seed = 1000 + 0*1000 + 0; goal1 uses ITS OWN base
    # (500) + 1*1000 + 0 = 1500 (NOT the movie seed).
    assert RENDER_CALLS[0]["seed"] == 1000, RENDER_CALLS[0]["seed"]
    assert RENDER_CALLS[1]["seed"] == 1500, RENDER_CALLS[1]["seed"]

    # movie.json attributes each goal's effective model
    with open(os.path.join(tmp, "assets", slugify(spec.project), "movie.json")) as fh:
        manifest = json.load(fh)
    assert manifest["goals"][0]["model"] == "sd-turbo", manifest["goals"]
    assert manifest["goals"][1]["model"] == "flux-schnell", manifest["goals"]
    assert manifest["segments"][1]["model"] == "flux-schnell", manifest["segments"][1]
    print("[6b] PASS per-goal model/steps/seed/negative/strength/chain overrides reach the render")


# --------------------------------------------------------------------------- #
# 7) resume — a completed segment bundle is SKIPPED on re-run
# --------------------------------------------------------------------------- #
def test_resume_skip():
    tmp = tempfile.mkdtemp(prefix="hugpy_test_movie_resume_")
    movie, media_bus = _install(tmp, can_carry=True)
    spec = _spec([(0, 2, "a"), (2, 4, "b")])

    # run 1: renders both segments, writes both bundles
    job1 = media_bus.enqueue("generate_movie", spec)
    res1 = movie.run_generate_movie(spec, job1)
    assert res1.ok and len(RENDER_CALLS) == 2, (res1.error, len(RENDER_CALLS))

    # run 2: SAME project -> same assets dir -> both bundles already exist -> skip all
    RENDER_CALLS.clear()
    PROGRESS_BLOBS.clear()
    job2 = media_bus.enqueue("generate_movie", spec)
    res2 = movie.run_generate_movie(spec, job2)
    assert res2.ok, res2.error
    assert len(RENDER_CALLS) == 0, f"resume must SKIP rendering: {len(RENDER_CALLS)} render(s)"
    assert all(s["status"] == "resumed" for s in res2.movie["segments"]), res2.movie["segments"]
    # re-ingested frames still surface as outputs
    assert len(res2.outputs) == 4, len(res2.outputs)
    print("[7] PASS  resume: existing segment bundles are SKIPPED on re-enqueue")


# --------------------------------------------------------------------------- #
# 8) concat-demux stitch (assemble=True) — segment mp4s -> movie.mp4
# --------------------------------------------------------------------------- #
def test_concat_stitch():
    tmp = tempfile.mkdtemp(prefix="hugpy_test_movie_concat_")
    movie, media_bus = _install(tmp, can_carry=True)
    spec = _spec([(0, 2, "a"), (2, 4, "b")], assemble=True)
    job_id = media_bus.enqueue("generate_movie", spec)
    res = movie.run_generate_movie(spec, job_id)

    assert res.ok, res.error
    # ITERATIVE save re-stitches as segments land: seg0 is a single-file COPY (no
    # concat), then seg1 + the final finalize each concat the 2 segment mp4s.
    assert len(CONCAT_CALLS) >= 1, "the multi-segment stitch must run concat"
    assert len(CONCAT_CALLS[-1]) == 2, "the stitch consumes both segment mp4s"
    assert all(p.endswith("video.mp4") for p in CONCAT_CALLS[-1]), CONCAT_CALLS[-1]
    assert res.movie["movie"] == "movie.mp4", res.movie["movie"]
    assert res.movie["partial"] is False, res.movie
    # movie.mp4 is ingested LAST -> outputs[-1] is the video
    assert res.outputs[-1].kind == "video", res.outputs[-1]
    print("[8] PASS  concat-demux stitch: segment mp4s -> movie.mp4 (outputs[-1] is the video)")


# --------------------------------------------------------------------------- #
# 9) nested progress blob shape
# --------------------------------------------------------------------------- #
def test_nested_progress_shape():
    tmp = tempfile.mkdtemp(prefix="hugpy_test_movie_prog_")
    movie, media_bus = _install(tmp, can_carry=True)
    spec = _spec([(0, 2, "a"), (2, 4, "b")])
    job_id = media_bus.enqueue("generate_movie", spec)
    movie.run_generate_movie(spec, job_id)

    assert PROGRESS_BLOBS, "progress must have been emitted"
    top_keys = {"stage", "segment_done", "segment_total", "segments", "current",
                "started_at", "eta_s", "drift", "legacy_pixel_chain", "label"}
    for blob in PROGRESS_BLOBS:
        assert set(blob) == top_keys, f"blob key mismatch: {sorted(blob)}"
        assert blob["segment_total"] == 2
        for s in blob["segments"]:
            assert {"index", "goal", "prompt", "attempt", "score", "status", "frames"} <= set(s), s

    # at least one "generating" blob carries the active per-frame `current` blob
    gen = [b for b in PROGRESS_BLOBS if b["current"] is not None]
    assert gen, "expected a generating blob with a non-null `current`"
    cur = gen[-1]["current"]
    assert {"done", "total", "stage", "label", "model", "frames"} <= set(cur), cur
    assert isinstance(cur["frames"], list), cur
    if cur["frames"]:
        f0 = cur["frames"][0]
        assert {"asset_id", "kind", "uri", "mime"} <= set(f0), f0
    # final blob reports every segment done
    assert PROGRESS_BLOBS[-1]["segment_done"] == 2, PROGRESS_BLOBS[-1]["segment_done"]
    print("[9] PASS  nested progress blob shape (top keys, per-segment keys, current per-frame blob)")


# --------------------------------------------------------------------------- #
# 10) ITERATIVE/RESILIENT save — a mid-run segment FAILURE still leaves a
#     watchable partial movie + an up-to-date manifest, and the JobResult points
#     at WHERE the partial output landed.
# --------------------------------------------------------------------------- #
def _make_failing_render(fail_on_segment):
    """Fake render that renders normally, but returns a JobResult error (exactly
    as scene.render_scene_frames would) on the Nth (0-based) segment."""
    def _render(*, out_dir, n_frames, on_frame_done, seed=None, start_frame=None,
                base_prompt=None, **kw):
        idx = len(RENDER_CALLS)   # this segment's render index (vision off -> 1/seg)
        if idx == fail_on_segment:
            from hugpy_video.intel.result_schema import JobError, JobResult
            RENDER_CALLS.append({"seed": seed, "start_frame": start_frame,
                                 "n_frames": n_frames, "base_prompt": base_prompt,
                                 "out_dir": out_dir})
            return JobResult(kw.get("job_id"), ok=False, error=JobError(
                code="generation_failed",
                message=f"stub failure on segment {idx}", retryable=True))
        return _fake_render(out_dir=out_dir, n_frames=n_frames,
                            on_frame_done=on_frame_done, seed=seed,
                            start_frame=start_frame, base_prompt=base_prompt, **kw)
    return _render


def test_partial_save_on_segment_failure():
    import json
    from hugpy_platform.utils import slugify
    tmp = tempfile.mkdtemp(prefix="hugpy_test_movie_partfail_")
    movie, media_bus = _install(tmp, can_carry=True)
    movie.render_scene_frames = _make_failing_render(fail_on_segment=2)
    spec = _spec([(0, 2, "a"), (2, 4, "b"), (4, 6, "c")], assemble=True)
    job_id = media_bus.enqueue("generate_movie", spec)
    res = movie.run_generate_movie(spec, job_id)

    # the movie FAILS with its ORIGINAL error, but now points at the partial output
    assert res.ok is False, res
    assert res.error is not None and res.error.code == "generation_failed", res.error
    meta = slugify(spec.project)
    assert res.project is not None, "failure JobResult must carry a project pointer"
    assert res.project["dir"] == f"assets/{meta}", res.project
    assert res.movie is not None, "failure JobResult must carry the partial manifest"
    assert res.movie["partial"] is True, res.movie
    assert res.movie["segments_completed"] == 2, res.movie["segments_completed"]
    assert res.movie["segments_total"] == 3, res.movie["segments_total"]
    assert len(RENDER_CALLS) == 3, f"3 segments attempted: {len(RENDER_CALLS)}"

    # movie.json on disk reflects the SAME partial state
    mj = os.path.join(tmp, "assets", meta, "movie.json")
    assert os.path.isfile(mj), "partial movie.json must exist on disk"
    with open(mj) as fh:
        manifest = json.load(fh)
    assert manifest["partial"] is True and manifest["segments_completed"] == 2, manifest
    assert len(manifest["segments"]) == 2, "only the 2 completed segments recorded"

    # a watchable partial movie.mp4 stitched from the 2 COMPLETED segments
    mv = os.path.join(tmp, "assets", meta, "movie.mp4")
    assert os.path.isfile(mv), "partial movie.mp4 must be stitched from completed segments"
    assert CONCAT_CALLS and len(CONCAT_CALLS[-1]) == 2, \
        f"partial stitch consumes both completed segment mp4s: {CONCAT_CALLS}"
    print("[10] PASS iterative save: mid-run segment FAILURE -> partial movie.json "
          "(partial=True, completed=2) + stitched movie.mp4 + project/movie pointer")


# --------------------------------------------------------------------------- #
# 11) ITERATIVE/RESILIENT save — a CANCEL after one segment still leaves a
#     one-segment partial movie + manifest.
# --------------------------------------------------------------------------- #
def test_partial_save_on_cancel():
    import json
    from hugpy_platform.utils import slugify
    tmp = tempfile.mkdtemp(prefix="hugpy_test_movie_partcancel_")
    movie, media_bus = _install(tmp, can_carry=True)
    spec = _spec([(0, 2, "a"), (2, 4, "b")], assemble=True)
    job_id = media_bus.enqueue("generate_movie", spec)

    # cancel BEFORE the 2nd segment renders: is_cancelling -> False (seg0 top),
    # True (seg1 top). The runner owns this check between segments.
    calls = {"n": 0}
    def _cancel_after_first(_job_id):
        calls["n"] += 1
        return calls["n"] >= 2
    media_bus.is_cancelling = _cancel_after_first

    res = movie.run_generate_movie(spec, job_id)

    assert res.ok is False, res
    assert res.error is not None and res.error.code == "cancelled", res.error
    meta = slugify(spec.project)
    assert res.project is not None and res.project["dir"] == f"assets/{meta}", res.project
    assert res.movie is not None, "cancel JobResult must carry the partial manifest"
    assert res.movie["partial"] is True, res.movie
    assert res.movie["segments_completed"] == 1, res.movie["segments_completed"]
    assert len(RENDER_CALLS) == 1, f"only seg0 rendered before cancel: {len(RENDER_CALLS)}"

    mj = os.path.join(tmp, "assets", meta, "movie.json")
    with open(mj) as fh:
        manifest = json.load(fh)
    assert manifest["partial"] is True and manifest["segments_completed"] == 1, manifest
    # a single completed segment -> movie.mp4 is a COPY of that segment (no concat)
    mv = os.path.join(tmp, "assets", meta, "movie.mp4")
    assert os.path.isfile(mv), "single-segment partial movie.mp4 must exist (copy path)"
    assert CONCAT_CALLS == [], f"a single segment must NOT concat: {CONCAT_CALLS}"
    print("[11] PASS iterative save: CANCEL after 1 segment -> partial movie.json "
          "(partial=True, completed=1) + single-segment movie.mp4 (copy) + pointer")


# --------------------------------------------------------------------------- #
def _run_all():
    test_make_movie_validation()
    test_parse_vision_verdict()
    test_orchestrator_vision_off_carry()
    test_orchestrator_independent_fallback()
    test_orchestrator_vision_retry()
    test_segment_bundle_builder()
    test_per_goal_overrides()
    test_resume_skip()
    test_concat_stitch()
    test_nested_progress_shape()
    test_partial_save_on_segment_failure()
    test_partial_save_on_cancel()
    print("\nALL movie-maker backend checks passed")


if __name__ == "__main__":
    _run_all()
