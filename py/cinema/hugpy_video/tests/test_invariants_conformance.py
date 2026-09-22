"""INVARIANT conformance suite for the video_intel job substrate (P0-1).

Locks the INV-1..8 invariants of the media-job spine as executable checks. Of the
two divergences this suite originally surfaced for P0-1b, divergence A (two
JobError types) is now RECONCILED by the Task 2 collapse and LOCKED here as a
regression guard; divergence B (two cancel planes) is still surfaced (documented,
not fixed). Phase-0 slice of STUDIO-ROADMAP.md (§4 task 5 + the Phase-0 table).

House style (mirrors tests/test_video_movie.py): a plain python script with a
``__main__`` guard, run via ``venv/bin/python tests/<file>.py``. Each check
prints a numbered ``[n] PASS`` / ``[n] FAIL`` line; a final summary line reports
the counts; the process exits nonzero iff any check FAILED. Unlike the sibling
files this driver CATCHES a failing assert per-check and keeps going (a
conformance suite must surface EVERY divergence in one run, not abort on the
first) — so a real INV violation shows up as a loud ``[n] FAIL`` alongside the
documented ``DIVERGENCE:`` lines.

SAFETY — the live job queue is a sqlite DB at
``$DEFAULT_ROOT/video_intel/media_jobs.db`` (DEFAULT_ROOT=/mnt/llm_storage here).
media_bus.DB_PATH is derived from DEFAULT_ROOT at import, so it points straight
at that live DB. Before ANY db operation this module repoints
``media_bus.DB_PATH`` at a throwaway ``tempfile.mkdtemp()`` (globally at import,
and again per db-touching check), and check [8] asserts the live DB's
(mtime, size) signature is byte-for-byte unchanged across the whole run.

Checks:
  [1] registry coherence   — SPEC_DESERIALIZERS/JOB_REGISTRY/DISPATCH key-sets
  [2] spec round-trip      — INV-1: asdict -> deserialize -> equal, every job
  [3] errors-as-data       — INV-3: a raising runner lands a terminal JobError
  [4] frozen specs         — INV-1/4: MediaRef/MovieSpec/SceneSpec are frozen
  [5] single-writer claim  — INV-2: no double-claim; claim_token gate holds
  [6] cooperative cancel   — INV-6: queued->cancelled, claimed->cancelling
  [7] DIVERGENCES          — JobError types RECONCILED+locked (Task 2); two
                             cancel planes still surfaced (PASS)
  [8] SAFETY               — the live media_jobs.db was never touched

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/test_invariants_conformance.py
"""
from __future__ import annotations

import dataclasses
import logging
import os
import sys
import tempfile
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

# The registry-build import chain (models_config) is very chatty at INFO; drop
# INFO/DEBUG so the [n] PASS/FAIL lines are legible. WARNING+ still surface.
logging.disable(logging.INFO)

# Keep the comms singleton (comms.jobs.job_store) purely in-process: never spin
# up a cross-process SqliteMirror on shared storage just to import JobError.
os.environ["HUGPY_COMMS_DB"] = "off"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# --------------------------------------------------------------------------- #
# live-DB safety: capture the REAL derived path + signature BEFORE repointing
# --------------------------------------------------------------------------- #
def _db_sig(path: str):
    """(mtime_ns, size) for `path`, or None if it does not exist. Cheap, no read."""
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        return None


from hugpy_video.intel import media_bus

# The live job queue's real path, straight from the module (do NOT hardcode).
LIVE_DB_PATH = os.path.abspath(media_bus.DB_PATH)
LIVE_DB_SIG_BEFORE = _db_sig(LIVE_DB_PATH)

# Global belt-and-suspenders: repoint the bus at a throwaway db at IMPORT time,
# so even a check that forgets _fresh_db() can never reach the live queue.
_SESSION_TMP = tempfile.mkdtemp(prefix="hugpy_conformance_session_")
media_bus.DB_PATH = os.path.join(_SESSION_TMP, "media_jobs.db")
media_bus._initialized = False


def _fresh_db() -> str:
    """Repoint media_bus at a brand-new throwaway db and force re-init. Mirrors
    the house-style _install() db-swap. Returns the temp dir."""
    d = tempfile.mkdtemp(prefix="hugpy_conformance_db_")
    media_bus.DB_PATH = os.path.join(d, "media_jobs.db")
    media_bus._initialized = False
    return d


# --------------------------------------------------------------------------- #
# now the substrate under test (import AFTER the repoint above)
# --------------------------------------------------------------------------- #
from hugpy_video.intel.job_schema import JOB_REGISTRY
from hugpy_video.intel.runners import DISPATCH
from hugpy_video.intel.result_schema import JobError as ResultJobError, JobResult
from hugpy_video.intel.media_schema import make_media_ref, MediaRef
from hugpy_video.intel.crop_schema import make_crop, SpatialRegion
from hugpy_video.intel.frame_schema import make_frame_extract
from hugpy_video.intel.audio_schema import make_audio_extract
from hugpy_video.intel.gen_schema import make_generate_image, text_part
from hugpy_video.intel.scene_schema import make_generate_scene, GenerateSceneSpec
from hugpy_video.intel.movie_schema import make_movie, GoalInterval, MovieSpec
from hugpy_video.intel.studio.job import make_studio_i2v
from hugpy_video.intel.studio_movie_schema import make_studio_movie, StudioMovieGoal

# the OTHER job plane (JobError reconciled by Task 2; cancel-plane divergence remains)
from hugpy_control.jobs import JobError as CommsJobError, JobStore, TERMINAL_STATUSES


# --------------------------------------------------------------------------- #
# minimal, VALID specs — one per registered job name (built via the factories)
# --------------------------------------------------------------------------- #
_IMG = make_media_ref(asset_id="img1", kind="image", uri="/tmp/conf_x.png",
                      mime="image/png", width=64, height=64)
_VID = make_media_ref(asset_id="vid1", kind="video", uri="/tmp/conf_x.mp4",
                      mime="video/mp4", width=64, height=64,
                      duration_s=1.0, fps_native=24.0)


def _minimal_specs() -> dict:
    """name -> a minimal VALID spec, one per JOB_REGISTRY entry, each through
    its validating factory."""
    return {
        "crop": make_crop(source=_IMG, spatial=SpatialRegion(0, 0, 10, 10)),
        "frame_extract": make_frame_extract(source=_VID, fps=1.0, quality=80, fmt="jpg"),
        "audio_extract": make_audio_extract(source=_VID, fmt="wav"),
        "generate_image": make_generate_image(
            parts=(text_part("a cat"),), model_id="sd-turbo",
            width=64, height=64, steps=2, guidance=0.0),
        "generate_scene": make_generate_scene(
            parts=(text_part("a cat"),), model_id="sd-turbo",
            width=64, height=64, steps=2, guidance=0.0,
            n_frames=2, fps=8, assemble=False),
        "generate_movie": make_movie(
            goals=(GoalInterval(0, 3, "a cat"),), model_id="sd-turbo",
            width=64, height=64, steps=2, guidance=0.0, fps=8, assemble=False),
        "studio_i2v": make_studio_i2v(width=64, height=64, fps=8, seed=1),
        "generate_studio_movie": make_studio_movie(
            goals=(StudioMovieGoal(segment_id="s0", prompt="a cat"),),
            width=64, height=64, fps=8),
    }


# a valid, enqueue-able spec reused by the db-facing checks (its runner is only
# ever exercised by check [3], which monkeypatches DISPATCH to force the raise).
_CROP_SPEC = make_crop(source=_IMG, spatial=SpatialRegion(0, 0, 10, 10))


# --------------------------------------------------------------------------- #
# [1] Registry coherence — the check that would have caught the intermittent
#     generate_movie deserializer gap.
# --------------------------------------------------------------------------- #
def check_1_registry_coherence():
    deser = set(media_bus.SPEC_DESERIALIZERS)
    registry = set(JOB_REGISTRY)
    assert deser == registry, (
        f"SPEC_DESERIALIZERS vs JOB_REGISTRY key mismatch: "
        f"only-in-deser={sorted(deser - registry)}, "
        f"only-in-registry={sorted(registry - deser)}")

    referenced = {js.runner_key for js in JOB_REGISTRY.values()}
    missing = referenced - set(DISPATCH)
    assert not missing, f"JOB_REGISTRY runner_key(s) with no DISPATCH entry: {sorted(missing)}"

    orphans = set(DISPATCH) - referenced
    assert not orphans, f"orphan DISPATCH key(s) no job routes to: {sorted(orphans)}"

    print(f"[1] PASS  registry coherence: {len(registry)} jobs; "
          f"SPEC_DESERIALIZERS==JOB_REGISTRY, every runner_key in DISPATCH, no orphans")


# --------------------------------------------------------------------------- #
# [2] Spec round-trip (INV-1) — asdict -> SPEC_DESERIALIZERS[name] -> equal.
# --------------------------------------------------------------------------- #
def check_2_spec_round_trip():
    specs = _minimal_specs()
    assert set(specs) == set(JOB_REGISTRY), (
        "test bug: _minimal_specs must cover exactly the registered jobs; "
        f"missing={sorted(set(JOB_REGISTRY) - set(specs))}")
    for name, spec in specs.items():
        d = asdict(spec)
        rebuilt = media_bus.SPEC_DESERIALIZERS[name](d)
        expected_type = JOB_REGISTRY[name].spec_type
        assert isinstance(rebuilt, expected_type), (
            f"{name}: deserializer returned {type(rebuilt).__name__}, "
            f"expected {expected_type.__name__}")
        assert rebuilt == spec, f"{name}: round-trip not identity: {rebuilt!r} != {spec!r}"
        # re-serialize must be stable too (asdict of the rebuilt matches)
        assert asdict(rebuilt) == d, f"{name}: re-serialized dict drifted"
    print(f"[2] PASS  spec round-trip (INV-1): {len(specs)} jobs asdict->deserialize->equal "
          "(re-validated through each factory)")


# --------------------------------------------------------------------------- #
# [3] Errors-as-data (INV-3) — a raising runner must land a TERMINAL JobResult
#     carrying a JobError; run_claimed must NOT propagate the raise.
# --------------------------------------------------------------------------- #
def check_3_errors_as_data():
    _fresh_db()

    def _raiser(spec, job_id):
        raise RuntimeError("forced runner failure for INV-3")

    orig_dispatch = media_bus.DISPATCH
    media_bus.DISPATCH = dict(orig_dispatch)
    media_bus.DISPATCH[("ffmpeg", "crop")] = _raiser
    try:
        job_id = media_bus.enqueue("crop", _CROP_SPEC)
        token = "conf-worker-inv3"
        claimed = media_bus.claim(token)
        assert claimed == job_id, f"claim returned {claimed!r}, expected {job_id!r}"

        # the whole point: this must RETURN a JobResult, never raise out
        result = media_bus.run_claimed(job_id, token)
    finally:
        media_bus.DISPATCH = orig_dispatch

    assert isinstance(result, JobResult), f"run_claimed returned {type(result).__name__}"
    assert result.ok is False, "a raising runner must yield ok=False"
    assert result.error is not None, "failure must carry a JobError (errors-as-data)"
    assert isinstance(result.error, ResultJobError), type(result.error).__name__
    assert result.error.code == "internal", f"expected code 'internal'; got {result.error.code!r}"
    assert result.error.retryable is False, "an internal raise-conversion is non-retryable"
    assert "RuntimeError" in result.error.message, (
        f"error message should name the original exception; got {result.error.message!r}")

    # terminal state was persisted ONCE as 'failed' (not 'cancelled'/'done')
    view = media_bus.get(job_id)
    assert view["status"] == "failed", f"terminal status must be 'failed'; got {view['status']!r}"
    assert view["result"] is not None and view["result"]["error"]["code"] == "internal"
    print("[3] PASS  errors-as-data (INV-3): raising runner -> terminal JobResult("
          "ok=False, JobError code='internal'), status='failed', no exception propagated")


# --------------------------------------------------------------------------- #
# [4] Frozen specs + MediaRef (INV-1/INV-4) — attribute assignment must raise.
# --------------------------------------------------------------------------- #
def check_4_frozen_specs():
    movie_spec = _minimal_specs()["generate_movie"]
    scene_spec = _minimal_specs()["generate_scene"]
    cases = [
        (_IMG, "uri", "/tmp/mutated.png", MediaRef),
        (movie_spec, "steps", 999, MovieSpec),
        (scene_spec, "n_frames", 999, GenerateSceneSpec),
    ]
    for obj, attr, val, expect_type in cases:
        assert isinstance(obj, expect_type), f"test bug: {obj!r} not a {expect_type.__name__}"
        raised = False
        try:
            setattr(obj, attr, val)
        except FrozenInstanceError:
            raised = True
        assert raised, f"{expect_type.__name__}.{attr} assignment must raise FrozenInstanceError"
    print("[4] PASS  frozen specs (INV-1/4): MediaRef, MovieSpec, GenerateSceneSpec reject "
          "attribute assignment (FrozenInstanceError)")


# --------------------------------------------------------------------------- #
# [5] Single-writer claim (INV-2) — a claimed job can't be re-claimed, and a
#     non-owning token can't write (the claim_token gate).
# --------------------------------------------------------------------------- #
def check_5_single_writer():
    _fresh_db()
    job_id = media_bus.enqueue("crop", _CROP_SPEC)

    first = media_bus.claim("token-A")
    assert first == job_id, f"first claim should take the job; got {first!r}"

    second = media_bus.claim("token-B")
    assert second is None, (
        f"a job already claimed must NOT be re-claimable; second claim got {second!r}")

    # claim_token gate: a worker that doesn't own the claim must refuse to write
    refused = media_bus.run_claimed(job_id, "token-B")
    assert refused is None, "run_claimed with a non-owning token must return None (refuse to write)"
    assert media_bus.get(job_id)["status"] == "claimed", (
        "a non-owning run_claimed must not mutate the row past 'claimed'")
    print("[5] PASS  single-writer claim (INV-2): claimed job is not re-claimable; "
          "run_claimed refuses a non-owning claim_token (no write)")


# --------------------------------------------------------------------------- #
# [6] Cooperative cancel state machine (INV-6).
# --------------------------------------------------------------------------- #
def check_6_cancel_state_machine():
    _fresh_db()

    # queued -> cancelled outright (claim() only picks 'queued', so it never runs)
    j1 = media_bus.enqueue("crop", _CROP_SPEC)
    r1 = media_bus.cancel(j1)
    assert r1["cancelled"] is True and r1["status"] == "cancelled", r1
    assert media_bus.get(j1)["status"] == "cancelled", media_bus.get(j1)
    assert media_bus.is_cancelling(j1) is False, "a terminal 'cancelled' is not 'cancelling'"

    # claimed -> 'cancelling' flag; is_cancelling() True (polled between frames)
    j2 = media_bus.enqueue("crop", _CROP_SPEC)
    assert media_bus.claim("tk-6") == j2
    r2 = media_bus.cancel(j2)
    assert r2["cancelled"] is True and r2["status"] == "cancelling", r2
    assert media_bus.is_cancelling(j2) is True, "claimed+cancel must set the cancelling flag"
    assert media_bus.get(j2)["status"] == "cancelling", media_bus.get(j2)
    print("[6] PASS  cooperative cancel (INV-6): queued->'cancelled'; "
          "claimed->'cancelling' + is_cancelling()==True")


# --------------------------------------------------------------------------- #
# [7] KNOWN-DIVERGENCE markers — DOCUMENT (do NOT fail). This is the P0-1b
#     reconcile backlog surfaced as data: two JobError types + two cancel planes.
# --------------------------------------------------------------------------- #
def check_7_known_divergences():
    # -- divergence A: two JobError contracts -- RECONCILED by the Task 2 collapse.
    # result_schema.JobError is now RE-EXPORTED from comms.jobs, so BOTH names bind
    # to the SAME class — one JobError for the whole tree. This check now LOCKS that
    # collapse (a regression guard) so the two can never silently un-unify.
    assert ResultJobError is CommsJobError, (
        "Task 2 collapse: video_intel.result_schema.JobError must BE "
        f"comms.jobs.JobError (got {ResultJobError!r} vs {CommsJobError!r})")
    fields = {f.name for f in dataclasses.fields(ResultJobError)}
    unified_frozen = ResultJobError.__dataclass_params__.frozen
    # Unified shape is the comms superset: the old {code,message,retryable} plus the
    # nullable free-form {detail}. Mutable (the comms class is not frozen).
    assert fields == {"code", "message", "detail", "retryable"}, fields
    assert unified_frozen is False, (
        "the unified JobError is the mutable comms.jobs.JobError")
    # The pre-collapse frozen class is ARCHIVED (operator rule: archive, never
    # delete) — retained but NOT the live name.
    from hugpy_video.intel import result_schema as _rs
    assert hasattr(_rs, "_ArchivedResultSchemaJobError"), "archived old class missing"
    assert _rs._ArchivedResultSchemaJobError is not ResultJobError, (
        "the archived class must be distinct from the live unified JobError")

    print("    RECONCILED: divergence A (two JobError types) RESOLVED — Task 2 collapse:")
    print(f"    RECONCILED:   result_schema.JobError IS comms.jobs.JobError = "
          f"fields{sorted(fields)} frozen={unified_frozen}  (one class; retryable: "
          f"routing hint, detail: free-form control/queue plane)")
    print("    RECONCILED:   pre-collapse frozen class archived as "
          "_ArchivedResultSchemaJobError (retained, unused)")

    # -- divergence B: two cancel planes -- STILL OPEN (out of scope for Task 2).
    assert callable(media_bus.cancel) and callable(media_bus.is_cancelling), (
        "media_bus cancel plane missing")
    assert hasattr(JobStore, "cancel") and hasattr(JobStore, "attach_cancel"), (
        "comms.jobs.JobStore cancel plane missing")
    assert isinstance(TERMINAL_STATUSES, frozenset) and "cancelled" in TERMINAL_STATUSES

    print("    DIVERGENCE: two cancel planes STILL exist (P0-1b/P0-2 reconcile target):")
    print("    DIVERGENCE:   media_bus.cancel/is_cancelling = cooperative sqlite-status flag "
          "(queued->'cancelled', claimed/running->'cancelling', polled between frames)")
    print("    DIVERGENCE:   comms.jobs.JobStore.cancel/attach_cancel = in-process cancel-handle "
          f"+ cancel_requested flag; owner teardown marks terminal; TERMINAL={sorted(TERMINAL_STATUSES)}")

    print("[7] PASS  divergence A (2 JobError types) RECONCILED via Task 2 collapse + locked; "
          "divergence B (2 cancel planes) still surfaced for P0-1b/P0-2")


# --------------------------------------------------------------------------- #
# [8] SAFETY — the live media_jobs.db was never touched by this run.
# --------------------------------------------------------------------------- #
def check_8_live_db_untouched():
    after = _db_sig(LIVE_DB_PATH)
    assert after == LIVE_DB_SIG_BEFORE, (
        f"LIVE DB {LIVE_DB_PATH} CHANGED during the run: "
        f"before={LIVE_DB_SIG_BEFORE} after={after}")
    assert os.path.abspath(media_bus.DB_PATH) != LIVE_DB_PATH, (
        "media_bus.DB_PATH must be repointed away from the live queue")
    print(f"[8] PASS  SAFETY: live media_jobs.db untouched "
          f"(sig {LIVE_DB_SIG_BEFORE} unchanged); bus repointed to a temp db")


# --------------------------------------------------------------------------- #
CHECKS = [
    (1, check_1_registry_coherence),
    (2, check_2_spec_round_trip),
    (3, check_3_errors_as_data),
    (4, check_4_frozen_specs),
    (5, check_5_single_writer),
    (6, check_6_cancel_state_machine),
    (7, check_7_known_divergences),
    (8, check_8_live_db_untouched),
]


def _run_all() -> int:
    passed = failed = 0
    for n, fn in CHECKS:
        try:
            fn()
            passed += 1
        except Exception as exc:  # a real INV violation -> loud FAIL, keep going
            failed += 1
            print(f"[{n}] FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\nconformance: {passed} passed, {failed} failed "
          f"({'ALL INVARIANTS HOLD' if not failed else 'INVARIANT VIOLATION(S) FOUND'})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
