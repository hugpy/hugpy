"""k71 model battery — writer, per-iteration studio hook, robustness.

Locks three contracts:

  1. WRITER. One run-dir per session under the battery root, holding
     ``results.json`` (a JSON ARRAY of ``{"model","axis","ok","secs","uri",
     "thumb_b64"}`` rows, additive ``error``/``ts`` only), a SELF-CONTAINED
     ``gallery.html`` (thumbnails as data: URIs) and ``run.log`` whose FIRST line
     records the root used. Every append atomically rewrites (tmp + os.replace):
     the file parses after every single record and no tmp litter survives.

  2. PER-ITERATION HOOK. ``produce_clip`` — the one choke point every studio
     render pass crosses — lands one battery row per iteration (Ok AND Err) plus
     an aptitude verdict line (``capability_verdict``) in run.log. Movie segments
     (…/segment_NN out_roots) collapse to ONE session per movie.

  3. ROBUSTNESS. An upstream Err is a recorded row, never an abort: the next
     iteration still records. And the writer can NEVER break the render path —
     a battery util that throws (monkeypatched) leaves produce_clip's Result
     byte-identical; a deleted run-dir makes ``record`` return False, not raise.

Same script style as ``test_studio_source_video.py`` (plain python, __main__
guard, numbered PASS/FAIL lines, nonzero exit iff any check failed; pytest is
NOT installed in this venv).

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_model_battery.py
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import uuid

logging.disable(logging.INFO)

os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")

# Point the battery at a private tmp root BEFORE any hugpy import reads it.
_BATTERY_ROOT = tempfile.mkdtemp(prefix="model-battery-test-")
os.environ["HUGPY_MODEL_BATTERY_ROOT"] = _BATTERY_ROOT
os.environ.pop("HUGPY_MODEL_BATTERY", None)

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_media import model_battery
from hugpy_video.intel.studio import produce as produce_mod
from hugpy_video.intel.studio.enums import Capability, Framework, Task
from hugpy_video.intel.studio.env import StudioEnv
from hugpy_video.intel.studio.errors import Err, ErrorCode, StageError
from hugpy_video.intel.studio.produce import produce_clip
from hugpy_video.intel.studio.schemas import CapabilityRequest, Resolution

R_TINY = Resolution(320, 180, 12)


def _studio_env(master_fps: int = 12) -> StudioEnv:
    return StudioEnv(
        output_root="/out", weights_root="/weights", manifest_root="/manifests",
        master_colorspace="rec709", master_fps=master_fps, max_vram_gb=24.0,
        loudness_target_lufs=-14.0, allow_unpinned=True)


def _i2v_request() -> CapabilityRequest:
    # vram_budget 0.5 GB routes to the SYNTHETIC runner — no GPU, no weights.
    return CapabilityRequest(
        capability=Capability.I2V, target_resolution=R_TINY, vram_budget_gb=0.5)


def _fresh_session() -> str:
    """A unique session key + a clean registry, so each check gets its own run-dir."""
    model_battery.reset_for_tests()
    return "sess-" + uuid.uuid4().hex


def _run_dir_of(run) -> str:
    assert run is not None, "run_for_session returned None with a writable root"
    return run.run_dir


def _read_results(run_dir: str) -> list:
    with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# [1] writer: first record creates the run-dir triplet with the ratified schema
# --------------------------------------------------------------------------- #
def test_writer_schema_and_files():
    key = _fresh_session()
    run = model_battery.run_for_session(key)
    d = _run_dir_of(run)
    assert run.record(model="m1", axis="A", ok=True, secs=1.234, uri="/tmp/x.png",
                      thumb_b64="") is True
    rows = _read_results(d)
    assert isinstance(rows, list) and len(rows) == 1, rows
    row = rows[0]
    for k in model_battery.ROW_KEYS:
        assert k in row, f"required key {k!r} missing from {row}"
    extras = set(row) - set(model_battery.ROW_KEYS)
    assert extras <= {"error", "ts"}, f"only error/ts may be additive; got {extras}"
    assert row["ok"] is True and row["secs"] == 1.23 and row["model"] == "m1"
    assert os.path.isfile(os.path.join(d, "gallery.html"))
    assert os.path.isfile(os.path.join(d, "run.log"))
    with open(os.path.join(d, "run.log"), encoding="utf-8") as fh:
        first = fh.readline()
    assert "battery root: " in first and _BATTERY_ROOT in first, (
        f"run.log must open with the root used; got {first!r}")


# --------------------------------------------------------------------------- #
# [2] writer: every append atomically rewrites — parseable after each, ordered,
#     and no tmp litter left behind
# --------------------------------------------------------------------------- #
def test_writer_atomic_appends():
    key = _fresh_session()
    run = model_battery.run_for_session(key)
    d = _run_dir_of(run)
    for i in range(5):
        assert run.record(model=f"m{i}", axis="B", ok=(i % 2 == 0), secs=i,
                          error=None if i % 2 == 0 else "boom")
        rows = _read_results(d)  # parses after EVERY append
        assert [r["model"] for r in rows] == [f"m{j}" for j in range(i + 1)]
    litter = [f for f in os.listdir(d) if f.startswith(".tmp-")]
    assert not litter, f"atomic rewrite left tmp litter: {litter}"
    # failed rows carry error + ok=false; ok rows carry none
    rows = _read_results(d)
    assert rows[1]["ok"] is False and rows[1]["error"] == "boom"
    assert "error" not in rows[0]


# --------------------------------------------------------------------------- #
# [3] writer: gallery.html is SELF-CONTAINED — inline data: URI thumbnails
# --------------------------------------------------------------------------- #
def test_gallery_self_contained():
    key = _fresh_session()
    run = model_battery.run_for_session(key)
    d = _run_dir_of(run)
    run.record(model="m", axis="C", ok=True, secs=1, uri="/x.png", thumb_b64="AAAA")
    with open(os.path.join(d, "gallery.html"), encoding="utf-8") as fh:
        page = fh.read()
    assert "data:image/jpeg;base64,AAAA" in page, "thumbnail must ride inline"
    assert 'src="/' not in page and 'src="http' not in page, (
        "gallery must reference no external files")


# --------------------------------------------------------------------------- #
# [4] roots: primary unwritable -> FALLBACK claims the run-dir, run.log says so
# --------------------------------------------------------------------------- #
def test_fallback_root():
    model_battery.reset_for_tests()
    # A file where a directory must go: makedirs on primary fails deterministically.
    blocked = tempfile.mkstemp(prefix="battery-blocked-")[1]
    fallback = tempfile.mkdtemp(prefix="battery-fallback-")
    saved_env = os.environ.pop("HUGPY_MODEL_BATTERY_ROOT")
    saved_primary, saved_fb = model_battery.PRIMARY_ROOT, model_battery.FALLBACK_ROOT
    model_battery.PRIMARY_ROOT = os.path.join(blocked, "model-battery")
    model_battery.FALLBACK_ROOT = fallback
    try:
        run = model_battery.run_for_session("fb-" + uuid.uuid4().hex)
        assert run is not None and run.run_dir.startswith(fallback), (
            f"expected a run-dir under the fallback root; got {run and run.run_dir}")
        assert run.root_used == fallback
        with open(run.log_path, encoding="utf-8") as fh:
            assert fallback in fh.readline(), "run.log must record the root used"
    finally:
        model_battery.PRIMARY_ROOT, model_battery.FALLBACK_ROOT = saved_primary, saved_fb
        os.environ["HUGPY_MODEL_BATTERY_ROOT"] = saved_env
        model_battery.reset_for_tests()
        os.unlink(blocked)
        shutil.rmtree(fallback, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [5] robustness: a vanished run-dir makes record() return False — never raise
# --------------------------------------------------------------------------- #
def test_record_never_raises():
    key = _fresh_session()
    run = model_battery.run_for_session(key)
    d = _run_dir_of(run)
    assert run.record(model="m", axis="A", ok=True, secs=0)
    shutil.rmtree(d)
    assert run.record(model="m", axis="A", ok=True, secs=0) is False, (
        "record over a deleted run-dir must degrade to False, not raise")
    assert run.log("still alive") is False


# --------------------------------------------------------------------------- #
# [6] kill switch: HUGPY_MODEL_BATTERY=off -> no run, no dirs
# --------------------------------------------------------------------------- #
def test_kill_switch():
    model_battery.reset_for_tests()
    os.environ["HUGPY_MODEL_BATTERY"] = "off"
    try:
        assert model_battery.enabled() is False
        assert model_battery.run_for_session("killed-" + uuid.uuid4().hex) is None
    finally:
        os.environ.pop("HUGPY_MODEL_BATTERY", None)
        model_battery.reset_for_tests()


# --------------------------------------------------------------------------- #
# [7] sessions: segment_NN out_roots collapse to ONE movie session; distinct
#     sessions claim distinct run-dirs (same-minute collision -> suffix)
# --------------------------------------------------------------------------- #
def test_session_keys_and_dirs():
    k = model_battery.session_key_for_out_root
    assert k("/movies/m1/segment_00") == k("/movies/m1/segment_07") == \
        os.path.normpath("/movies/m1")
    assert k("/movies/m1") == os.path.normpath("/movies/m1")
    assert k("") == "default"
    model_battery.reset_for_tests()
    a = model_battery.run_for_session("a-" + uuid.uuid4().hex)
    b = model_battery.run_for_session("b-" + uuid.uuid4().hex)
    assert a.run_dir != b.run_dir, "two sessions must never share a run-dir"
    assert model_battery.run_for_session("c").run_dir == \
        model_battery.run_for_session("c").run_dir, "same session -> same run-dir"


# --------------------------------------------------------------------------- #
# [8] produce_clip Ok: one row per iteration + an aptitude line in run.log
# --------------------------------------------------------------------------- #
def test_produce_clip_records_iteration():
    model_battery.reset_for_tests()
    out_root = tempfile.mkdtemp(prefix="battery-prod-")
    try:
        res = produce_clip(_i2v_request(), env=_studio_env(), out_root=out_root)
        assert res.is_ok(), f"synthetic render must be Ok; got {res}"
        run = model_battery.run_for_session(
            model_battery.session_key_for_out_root(out_root))
        rows = _read_results(run.run_dir)
        assert len(rows) == 1, f"exactly one row per iteration; got {len(rows)}"
        row = rows[0]
        assert row["ok"] is True and row["axis"] == "i2v"
        assert row["uri"] == res.unwrap().path
        assert row["model"], "the bound model_id must be recorded"
        with open(os.path.join(run.run_dir, "run.log"), encoding="utf-8") as fh:
            log = fh.read()
        assert "aptitude: capability=i2v servable=" in log, (
            "every iteration must record its aptitude/preset verdict")
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [9] produce_clip Err: recorded as ok=false + error, and the NEXT iteration
#     still records — an upstream error never aborts the battery run
# --------------------------------------------------------------------------- #
def test_err_iteration_recorded_and_run_continues():
    model_battery.reset_for_tests()
    out_root = tempfile.mkdtemp(prefix="battery-err-")
    key = (Framework.SYNTHETIC, Task.I2V)
    real_runner = produce_mod._DISPATCH[key]

    def exploding_runner(manifest, root, **kwargs):
        return Err(StageError(ErrorCode.IO_ERROR, "disk fell off", ()))

    try:
        produce_mod._DISPATCH[key] = exploding_runner
        res = produce_clip(_i2v_request(), env=_studio_env(), out_root=out_root)
        assert res.is_err() and res.error.code is ErrorCode.IO_ERROR, (
            "the hook must not alter the Err returned")
        produce_mod._DISPATCH[key] = real_runner
        res2 = produce_clip(_i2v_request(), env=_studio_env(), out_root=out_root)
        assert res2.is_ok(), f"the run must CONTINUE after an Err; got {res2}"
        run = model_battery.run_for_session(
            model_battery.session_key_for_out_root(out_root))
        rows = _read_results(run.run_dir)
        assert len(rows) == 2, f"both iterations must land rows; got {len(rows)}"
        assert rows[0]["ok"] is False and "disk fell off" in rows[0]["error"]
        assert rows[0]["uri"] == "" and rows[0]["thumb_b64"] == ""
        assert rows[1]["ok"] is True
    finally:
        produce_mod._DISPATCH[key] = real_runner
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [10] a battery that THROWS can never break the render path
# --------------------------------------------------------------------------- #
def test_broken_battery_never_breaks_render():
    model_battery.reset_for_tests()
    out_root = tempfile.mkdtemp(prefix="battery-broken-")
    saved = model_battery.run_for_session

    def exploding_run_for_session(*a, **k):
        raise RuntimeError("battery on fire")

    try:
        model_battery.run_for_session = exploding_run_for_session
        res = produce_clip(_i2v_request(), env=_studio_env(), out_root=out_root)
        assert res.is_ok(), (
            f"a throwing battery must never alter the render Result; got {res}")
    finally:
        model_battery.run_for_session = saved
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [11] imagegen hook: _record_battery is no-raise and lands a row for ok AND
#      failed generations (duck-typed req/result — no pipeline, no torch)
# --------------------------------------------------------------------------- #
def test_imagegen_record_battery():
    from hugpy_media.imagegen.imagegen_runner import _record_battery

    class _Req:
        model_key = "comfy-testmodel"

    class _Img:
        path = "/nonexistent/img.png"

    class _ResOk:
        ok = True
        images = [_Img()]
        error = None

    class _ResBad:
        ok = False
        images = []
        error = "CUDA out of memory"

    model_battery.reset_for_tests()
    _record_battery(_Req(), _ResOk(), 2.5, axis="t2i")
    _record_battery(_Req(), _ResBad(), 0.1, axis="i2i")
    run = model_battery.run_for_session("imagegen")
    rows = _read_results(run.run_dir)
    assert len(rows) == 2, rows
    assert rows[0]["ok"] is True and rows[0]["axis"] == "t2i"
    assert rows[1]["ok"] is False and rows[1]["error"] == "CUDA out of memory"
    # and it never raises, even if the util is broken outright
    saved = model_battery.run_for_session
    try:
        model_battery.run_for_session = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("boom"))
        _record_battery(_Req(), _ResOk(), 1.0, axis="t2i")  # must not raise
    finally:
        model_battery.run_for_session = saved


CHECKS = [
    ("writer: run-dir triplet + ratified row schema (keys, additive error/ts only)",
     test_writer_schema_and_files),
    ("writer: atomic per-append rewrite — parseable after each, no tmp litter",
     test_writer_atomic_appends),
    ("writer: gallery.html is self-contained (inline data: thumbnails)",
     test_gallery_self_contained),
    ("roots: unwritable primary falls back; run.log records the root used",
     test_fallback_root),
    ("robustness: record()/log() over a deleted run-dir -> False, never a raise",
     test_record_never_raises),
    ("kill switch: HUGPY_MODEL_BATTERY=off disables recording",
     test_kill_switch),
    ("sessions: segment_NN collapses to the movie; distinct sessions, distinct dirs",
     test_session_keys_and_dirs),
    ("produce: every Ok iteration lands a row + an aptitude verdict line",
     test_produce_clip_records_iteration),
    ("produce: an Err iteration records ok=false+error and the run CONTINUES",
     test_err_iteration_recorded_and_run_continues),
    ("produce: a throwing battery never alters the render Result",
     test_broken_battery_never_breaks_render),
    ("imagegen: _record_battery rows for ok+failed, no-raise even when broken",
     test_imagegen_record_battery),
]


def main() -> int:
    passed = 0
    failed = 0
    try:
        for i, (name, fn) in enumerate(CHECKS, 1):
            try:
                fn()
            except Exception as exc:  # surface EVERY divergence, not just the first
                failed += 1
                print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
            else:
                passed += 1
                print(f"[{i}] PASS  {name}")
    finally:
        shutil.rmtree(_BATTERY_ROOT, ignore_errors=True)
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
