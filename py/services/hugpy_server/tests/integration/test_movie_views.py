"""Movie-view slice (IDENTITY-3D-CONTINUITY-PLAN.md S2-movie + S3).

Exercises PER-GOAL view-aware id_lock DNA for studio MOVIES: each segment of an
identity movie may condition on a DIFFERENT turntable VIEW of the SAME identity, so
a ``cut`` into a new scene ("beach" -> "volleyball") holds the character while turning
the camera per shot. Built ENTIRELY on the S1+S2 angle-bank helpers (no new angle math)
and additive throughout — a viewless, cue-less movie is byte-identical to today.

Isolation idiom is lifted from test_identity_views.py: rebind the identity-store globals
(IDENTITIES_HOME/PROJECTS_HOME) to temp dirs under the real DEFAULT_ROOT (so the copied
ring frames still pass the route jail), point the media bus at a temp DB, and build a
minimal Flask app with the video blueprint. The enqueue is captured (media_bus.enqueue is
patched) so a test can inspect the resolved StudioMovieSpec without a DB round-trip. The
runner test stubs render_clip / ingest / assembly like test_video_movie.py.

Locks (each runs independently so one failure never masks the rest):
  [schema]
   1. StudioMovieGoal.reference_images validates + coerces (good tuple ok; bad -> ValueError).
   2. make_studio_movie builds a movie carrying a per-goal reference set.
  [runner]
   3. studio_movie prefers the GOAL's refs when present, the MOVIE's when absent
      (id_refs selection, render spine stubbed).
  [enqueue]
   4. a goal with view:"back" on a ring-backed identity gets back-view bank URIs while a
      viewless goal inherits the movie-level DNA (per-goal reference_images differ correctly).
   5. an invalid goal ``view`` is a clean 400 naming the offending segment.
   7. a viewless, cue-less identity movie -> every goal inherits (zero-regression: all None).
  [shot-intent]
   6. derive_view_from_prompt maps cue phrases -> views and returns None on no cue.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python -m pytest tests/test_movie_views.py -q
  venv/bin/python tests/test_movie_views.py
"""
from __future__ import annotations

import atexit
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
from types import SimpleNamespace

logging.disable(logging.INFO)

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import importlib  # noqa: E402

from flask import Flask  # noqa: E402

from hugpy_video.intel import identity_profiles
from hugpy_video.intel import media_bus
from hugpy_video.intel import shot_intent
from hugpy_video.intel.studio_movie_schema import StudioMovieGoal, make_studio_movie
from hugpy_platform.constants import UPLOADS_HOME, DEFAULT_ROOT

# STORE isolation — rebind the module globals the store path helpers read. IDENTITIES_HOME
# must sit under the real DEFAULT_ROOT so the identity-owned ring frames pass the route jail.
_TMP_IDENTITIES = tempfile.mkdtemp(prefix="hugpy-mvview-store-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
identity_profiles.IDENTITIES_HOME = _TMP_IDENTITIES
_TMP_PROJECTS = tempfile.mkdtemp(prefix="hugpy-mvview-projects-")
identity_profiles.PROJECTS_HOME = _TMP_PROJECTS

# JAIL: reference/frame images must resolve under the real UPLOADS_HOME.
_TMP_UPLOADS = tempfile.mkdtemp(prefix="hugpy-mvview-uploads-", dir=UPLOADS_HOME)

# media bus -> temp DB so nothing touches the real catalog.
_TMP_DB = tempfile.mkstemp(prefix="mvview-bus-", suffix=".db")[1]
media_bus.DB_PATH = _TMP_DB
media_bus._initialized = False
with sqlite3.connect(_TMP_DB) as _c:
    _c.execute(
        "CREATE TABLE IF NOT EXISTS media_jobs ("
        "job_id TEXT PRIMARY KEY, name TEXT, status TEXT, spec_json TEXT, "
        "result_json TEXT, claim_token TEXT, created REAL, updated REAL, "
        "progress_json TEXT)")

vr = importlib.import_module("hugpy_server.app.routes.video_routes")

# CAPTURE the enqueued spec (route -> media_bus.enqueue) so a test can inspect the resolved
# StudioMovieSpec directly. vr.media_bus is this same module object, so patching the attr
# here reroutes the route's call.
_ENQUEUED: list = []


def _capture_enqueue(name, spec, **kwargs):   # principal/owner attribution
    _ENQUEUED.append((name, spec))
    return f"job_test_{len(_ENQUEUED):03d}"


media_bus.enqueue = _capture_enqueue

app = Flask(__name__)
app.register_blueprint(vr.video_bp)
client = app.test_client()


@atexit.register
def _cleanup() -> None:
    for d in (_TMP_IDENTITIES, _TMP_PROJECTS, _TMP_UPLOADS):
        shutil.rmtree(d, ignore_errors=True)
    try:
        os.remove(_TMP_DB)
    except OSError:
        pass


def _make_png(path: str, color=(120, 120, 120)) -> None:
    from PIL import Image
    Image.new("RGB", (16, 16), color).save(path)


_IMG_A = os.path.join(_TMP_UPLOADS, "view_a.png")
_IMG_B = os.path.join(_TMP_UPLOADS, "view_b.png")
_make_png(_IMG_A, (200, 40, 40))
_make_png(_IMG_B, (40, 200, 40))


def _turntable_profile(name: str, frame_count: int = 72, dpf: float = 5.0):
    """A profile whose ACTIVE version is backed by a synthetic ``frame_count``-frame
    turntable reconstruction, returning ``(slug, recon_id, ordered_view_paths, canonical)``.

    The ring frames are REAL PNGs (not stubs) so ``media_store.ingest(kind_hint="image")``
    at the route classifies them as images — the route-level tests actually ingest the bank
    frames (unlike test_identity_views, which only reads paths from the resolver)."""
    p = identity_profiles.create_profile(name, [_IMG_A, _IMG_B], notes="")
    slug = p["slug"]
    frame_dir = tempfile.mkdtemp(prefix="tt-src-", dir=_TMP_UPLOADS)
    sources = []
    for i in range(frame_count):
        fp = os.path.join(frame_dir, f"frame_{i:04d}.png")
        _make_png(fp, (i % 256, (2 * i) % 256, (3 * i) % 256))  # REAL, distinct image bytes
        sources.append(fp)
    recon_id = "recon_tt_" + slug
    identity_profiles.attach_reconstruction(
        slug, recon_id, sources,
        spec={"mode": "turntable", "degrees_per_frame": dpf, "frame_count": frame_count})
    owned = identity_profiles.get_profile(slug)["reference_images"]
    canonical = [owned[0]]
    identity_profiles.mint_version(slug, recon_id, "textured", canonical)
    rec = identity_profiles.get_reconstruction(slug, recon_id)
    return slug, recon_id, list(rec["views"]), canonical


def _rp_set(paths):
    """Realpath-normalized set (DEFAULT_ROOT may be a symlink; the route ingests via
    realpath, so compare on realpath to stay symlink-safe)."""
    return {os.path.realpath(p) for p in paths}


def _last_spec():
    assert _ENQUEUED, "no spec was enqueued"
    name, spec = _ENQUEUED[-1]
    assert name == "generate_studio_movie", name
    return spec


# --------------------------------------------------------------------------- #
# [1] schema — per-goal reference_images validates + coerces (list -> tuple)
# --------------------------------------------------------------------------- #
def test_goal_reference_images_validates():
    root = StudioMovieGoal("seg_00", "a", None)

    # good: a list is COERCED to a tuple by make_studio_movie (round-trip safe).
    g = StudioMovieGoal("seg_01", "b", "seg_00", joint_mode="cut",
                        reference_images=["x.png", "y.png"])
    spec = make_studio_movie(goals=(root, g), width=64, height=64, fps=8)
    assert spec.goals[1].reference_images == ("x.png", "y.png"), spec.goals[1].reference_images
    assert isinstance(spec.goals[1].reference_images, tuple)
    # None (inherit) is preserved untouched.
    assert spec.goals[0].reference_images is None

    # bad values -> ValueError at construction (mirrors the movie-level field).
    def _bad(refs):
        try:
            make_studio_movie(
                goals=(root, StudioMovieGoal("seg_01", "b", "seg_00", joint_mode="cut",
                                             reference_images=refs)),
                width=64, height=64, fps=8)
        except ValueError:
            return True
        return False

    assert _bad([""]), "empty path string must raise"
    assert _bad(["a", "b", "c", "d", "e"]), "more than MAX_REFERENCE_IMAGES must raise"
    assert _bad("not-a-list"), "a bare string is not a ref list"
    assert _bad([123]), "a non-string ref must raise"


# --------------------------------------------------------------------------- #
# [2] schema — make_studio_movie builds a movie with a per-goal reference set
# --------------------------------------------------------------------------- #
def test_make_studio_movie_with_per_goal_refs():
    root = StudioMovieGoal("seg_00", "beach", None)
    g1 = StudioMovieGoal("seg_01", "volleyball", "seg_00", joint_mode="cut",
                         reference_images=("back_0.png", "back_1.png"))
    spec = make_studio_movie(goals=(root, g1), width=128, height=128, fps=12,
                             reference_images=("front_0.png",))
    assert len(spec.goals) == 2
    assert spec.reference_images == ("front_0.png",)                 # movie-level DNA
    assert spec.goals[0].reference_images is None                    # inherits
    assert spec.goals[1].reference_images == ("back_0.png", "back_1.png")  # per-goal override


# --------------------------------------------------------------------------- #
# [3] runner — id_refs prefers the GOAL's refs when present, the MOVIE's when absent
# --------------------------------------------------------------------------- #
def _run_movie_capture_refs(spec):
    """Run run_generate_studio_movie with the render spine stubbed; return the ordered
    list of reference_images each rendered segment was handed (mirrors test_video_movie's
    render-seam capture)."""
    from hugpy_video.intel.runners import studio_movie

    captured: list = []

    def _fake_render_clip(seg_spec, render_id=None, should_cancel=None,
                          progress_sink=None, produce=None):
        captured.append(tuple(seg_spec.reference_images or ()))
        return SimpleNamespace(
            ok=True, error=None, path=f"/tmp/{render_id}.mp4",
            frames=8, width=seg_spec.width, height=seg_spec.height, duration_s=1.0,
            content_hash="deadbeef", resumed=False,
            effective_budget_gb=seg_spec.vram_budget_gb, budget_source="explicit")

    def _fake_ingest(path, kind_hint=None):
        return SimpleNamespace(uri=os.path.abspath(str(path)), kind=(kind_hint or "video"))

    orig = (studio_movie.render_clip, studio_movie.ingest,
            studio_movie._assemble_movie, studio_movie._write_movie_json,
            media_bus.is_cancelling, media_bus.set_progress)
    studio_movie.render_clip = _fake_render_clip
    studio_movie.ingest = _fake_ingest
    studio_movie._assemble_movie = lambda *a, **k: {"movie": None}
    studio_movie._write_movie_json = lambda *a, **k: {"assembly": {"movie": None}}
    media_bus.is_cancelling = lambda _job_id: False
    media_bus.set_progress = lambda *a, **k: None
    try:
        res = studio_movie.run_generate_studio_movie(spec, "job_runner_test")
    finally:
        (studio_movie.render_clip, studio_movie.ingest,
         studio_movie._assemble_movie, studio_movie._write_movie_json,
         media_bus.is_cancelling, media_bus.set_progress) = orig
    return res, captured


def test_runner_prefers_goal_refs():
    tmp = tempfile.mkdtemp(prefix="mvview-runner-")
    root = StudioMovieGoal("seg_00", "beach", None)
    # goal 1 carries its OWN refs (a cut into a new scene, held by a different view).
    g1 = StudioMovieGoal("seg_01", "volleyball", "seg_00", joint_mode="cut",
                         reference_images=("g1_a.png", "g1_b.png"))
    spec = make_studio_movie(goals=(root, g1), width=64, height=64, fps=8, out_root=tmp,
                             reference_images=("m_front.png", "m_side.png"))
    res, captured = _run_movie_capture_refs(spec)
    assert res.ok, getattr(res, "error", None)
    assert len(captured) == 2, captured
    # segment 0 (no per-goal refs) inherits the MOVIE-level DNA; segment 1 uses its OWN.
    assert captured[0] == ("m_front.png", "m_side.png"), captured[0]
    assert captured[1] == ("g1_a.png", "g1_b.png"), captured[1]

    # ...and with goal 1 having NO per-goal refs, BOTH inherit the movie-level set.
    tmp2 = tempfile.mkdtemp(prefix="mvview-runner2-")
    g1_plain = StudioMovieGoal("seg_01", "volleyball", "seg_00", joint_mode="cut")
    spec2 = make_studio_movie(goals=(root, g1_plain), width=64, height=64, fps=8, out_root=tmp2,
                              reference_images=("m_front.png", "m_side.png"))
    res2, captured2 = _run_movie_capture_refs(spec2)
    assert res2.ok, getattr(res2, "error", None)
    assert captured2 == [("m_front.png", "m_side.png"), ("m_front.png", "m_side.png")], captured2


# --------------------------------------------------------------------------- #
# [4] enqueue — a view:"back" goal gets back-view bank URIs; a viewless goal inherits
# --------------------------------------------------------------------------- #
def test_enqueue_per_goal_view_resolution():
    slug, _recon_id, views, canonical = _turntable_profile("Movie View Four")
    r = client.post("/video/studio/movie", json={
        "identity_profile": slug,
        "goals": [
            {"prompt": "on the beach"},                                  # no view, no cue
            {"prompt": "playing volleyball", "joint_mode": "cut", "view": "back"},
        ],
    })
    assert r.status_code == 200, (r.status_code, r.get_json())
    spec = _last_spec()

    # segment 0 has no view -> reference_images None -> inherits the movie-level DNA.
    assert spec.goals[0].reference_images is None, spec.goals[0].reference_images
    # the movie-level DNA is the identity's canonical (unchanged from today).
    assert _rp_set(spec.reference_images) == _rp_set(canonical), spec.reference_images

    # segment 1 view "back" (180° == index 36) -> the 4 angle-nearest RING frames.
    g1 = spec.goals[1].reference_images
    assert g1 is not None and len(g1) == 4, g1
    assert _rp_set(g1) == _rp_set([views[34], views[35], views[36], views[37]]), g1
    # the per-goal DNA is demonstrably NOT the movie-level cardinals.
    assert _rp_set(g1) != _rp_set(spec.reference_images)


# --------------------------------------------------------------------------- #
# [5] enqueue — an invalid goal ``view`` is a clean 400 naming the segment
# --------------------------------------------------------------------------- #
def test_enqueue_invalid_view_is_400():
    slug, _recon_id, _views, _canon = _turntable_profile("Movie View Five")
    r = client.post("/video/studio/movie", json={
        "identity_profile": slug,
        "goals": [
            {"prompt": "on the beach"},
            {"prompt": "walking off", "joint_mode": "cut", "view": "sideways"},  # not a view
        ],
    })
    assert r.status_code == 400, (r.status_code, r.get_json())
    err = r.get_json().get("error", "")
    assert "seg_01" in err, err  # names the offending segment


# --------------------------------------------------------------------------- #
# [6] shot-intent — derive_view_from_prompt cue table (keyword pass, conservative)
# --------------------------------------------------------------------------- #
def test_derive_view_from_prompt_table():
    cases = {
        "she walks away": "back",
        "left profile shot": "left-profile",
        "a wide beach scene": None,                 # no cue -> inherit
        "from behind as the sun sets": "back",
        "a right profile close-up": "right-profile",
        "over her shoulder toward the door": "back-right",
        "a 3/4 turn to the left": "three-quarter-left",
        "the crew rearrange the set": None,         # 'rear' must NOT match 'rearrange'
        "": None,
    }
    for prompt, want in cases.items():
        got = shot_intent.derive_view_from_prompt(prompt)
        assert got == want, (prompt, got, want)
    # every derived value is a real view the resolver accepts.
    for prompt in ("she walks away", "left profile shot", "over the shoulder"):
        name = shot_intent.derive_view_from_prompt(prompt)
        az, err = identity_profiles.azimuth_for_view(name)
        assert err is None and az is not None, (prompt, name, err)


# --------------------------------------------------------------------------- #
# [7] enqueue — zero-regression: a viewless, cue-less identity movie inherits everywhere
# --------------------------------------------------------------------------- #
def test_enqueue_viewless_zero_regression():
    slug, _recon_id, _views, canonical = _turntable_profile("Movie View Seven")
    r = client.post("/video/studio/movie", json={
        "identity_profile": slug,
        "goals": [
            {"prompt": "a calm meadow at dawn"},
            {"prompt": "a quiet lake", "joint_mode": "cut"},   # no view, no orientation cue
        ],
    })
    assert r.status_code == 200, (r.status_code, r.get_json())
    spec = _last_spec()
    # EVERY goal inherits: no per-goal override anywhere (byte-identical to pre-S2 behavior).
    assert all(g.reference_images is None for g in spec.goals), \
        [g.reference_images for g in spec.goals]
    # the movie-level DNA is the identity canonical, non-empty (the movie is still an id-movie).
    assert spec.reference_images and _rp_set(spec.reference_images) == _rp_set(canonical)

    # control: a NON-identity movie with a stray goal ``view`` degrades to inherit (no error).
    r2 = client.post("/video/studio/movie", json={
        "goals": [
            {"prompt": "a t2v sunrise"},
            {"prompt": "walking away", "joint_mode": "cut", "view": "back"},  # ignored: no identity
        ],
    })
    assert r2.status_code == 200, (r2.status_code, r2.get_json())
    spec2 = _last_spec()
    assert spec2.reference_images == (), spec2.reference_images       # plain movie
    assert all(g.reference_images is None for g in spec2.goals), \
        [g.reference_images for g in spec2.goals]


CHECKS = [
    ("schema: per-goal reference_images validates + coerces (list->tuple); bad->ValueError",
     test_goal_reference_images_validates),
    ("schema: make_studio_movie builds a movie with a per-goal reference set",
     test_make_studio_movie_with_per_goal_refs),
    ("runner: id_refs prefers the GOAL's refs when present, the MOVIE's when absent",
     test_runner_prefers_goal_refs),
    ("enqueue: view:'back' -> back-view bank URIs; viewless goal inherits movie-level DNA",
     test_enqueue_per_goal_view_resolution),
    ("enqueue: invalid goal view -> clean 400 naming the segment",
     test_enqueue_invalid_view_is_400),
    ("shot-intent: derive_view_from_prompt cue table (conservative, whole-word)",
     test_derive_view_from_prompt_table),
    ("enqueue: viewless cue-less identity movie -> every goal inherits (zero-regression)",
     test_enqueue_viewless_zero_regression),
]


def main() -> int:
    passed = failed = 0
    for i, (name, fn) in enumerate(CHECKS, 1):
        try:
            fn()
        except Exception as exc:  # surface EVERY divergence, not just the first
            failed += 1
            print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"[{i}] PASS  {name}")
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
