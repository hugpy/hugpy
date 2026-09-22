"""DIRECT MODEL CHOICE (model_id pin) — this slice.

A caller can PIN a specific studio model: the router binds THAT model or returns a
CLEAR Err-as-data (never a silent fallback to a different model). Locks the pin at
every layer it travels: the router, the StudioI2VSpec round-trip, the bus runner's
CapabilityRequest construction, the /video/studio/i2v route, and the bus adapter's
JobError mapping.

Same script style as the other studio suites (plain python, ``__main__`` guard,
numbered ``[n] PASS`` / ``[n] FAIL`` lines, nonzero exit iff any FAILED). pytest is
NOT installed in this venv.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_model_pin.py
"""
from __future__ import annotations

import atexit
import dataclasses
import logging
import os
import sqlite3
import sys
import tempfile

logging.disable(logging.INFO)
os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")
os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="studio-pin-test-"))

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import importlib  # noqa: E402

from flask import Flask  # noqa: E402

from hugpy_video.intel import media_bus
from hugpy_video.intel.runners.studio_i2v import build_capability_request, run_studio_i2v
from hugpy_video.intel.studio.enums import Capability
from hugpy_video.intel.studio.errors import ErrorCode
from hugpy_video.intel.studio.job import make_studio_i2v, studio_i2v_from_dict
from hugpy_video.intel.studio.router import CapabilityRouter
from hugpy_video.intel.studio.schemas import CapabilityRequest, Resolution

# Isolate the media bus so route enqueues + run_studio_i2v never touch the real catalog.
_TMP_DB = tempfile.mkstemp(prefix="studio-pin-bus-", suffix=".db")[1]
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


def _resolve(cap, w, h, fps, budget, pin):
    return CapabilityRouter().resolve(CapabilityRequest(
        capability=cap, target_resolution=Resolution(w, h, fps),
        vram_budget_gb=budget, pinned_model_id=pin))


# --------------------------------------------------------------------------- #
# [1] Router honors a valid pin (binds exactly the requested model).
# --------------------------------------------------------------------------- #
def test_router_pin_valid():
    r = _resolve(Capability.T2V, 832, 480, 16, 9.0, "wan2.1-t2v-1.3b")
    assert r.is_ok(), r
    assert r.unwrap().model_id == "wan2.1-t2v-1.3b", r.unwrap()


# --------------------------------------------------------------------------- #
# [2] A pin to a model that fits differently is HONORED, not overridden by the
#     auto-pick: pinning the 2.2 A14B t2v at a budget that also admits the 1.3B still
#     binds the A14B (proves the pin restricts candidates, not just re-ranks).
# --------------------------------------------------------------------------- #
def test_router_pin_overrides_autopick():
    # unpinned @16GB t2v 480p would pick a bigger/again-scored model; pin forces A14B.
    r = _resolve(Capability.T2V, 832, 480, 16, 16.0, "wan2.2-t2v-a14b")
    assert r.is_ok(), r
    assert r.unwrap().model_id == "wan2.2-t2v-a14b", r.unwrap()


# --------------------------------------------------------------------------- #
# [3] Unknown model_id -> PINNED_MODEL_UNAVAILABLE (clear data, never a fallback).
# --------------------------------------------------------------------------- #
def test_router_pin_unknown():
    r = _resolve(Capability.T2V, 832, 480, 16, 9.0, "no-such-model")
    assert r.is_err() and r.error.code == ErrorCode.PINNED_MODEL_UNAVAILABLE, r


# --------------------------------------------------------------------------- #
# [4] Pinned model exists but does not serve the capability -> PINNED_MODEL_UNAVAILABLE.
# --------------------------------------------------------------------------- #
def test_router_pin_wrong_capability():
    # wan2.1-i2v-14b-720p serves I2V, not T2V.
    r = _resolve(Capability.T2V, 832, 480, 16, 20.0, "wan2.1-i2v-14b-720p")
    assert r.is_err() and r.error.code == ErrorCode.PINNED_MODEL_UNAVAILABLE, r


# --------------------------------------------------------------------------- #
# [5] Pinned model serves the capability but does NOT fit the budget -> the normal
#     sharpened gate reason (VRAM_EXCEEDED), NOT a silent fallback to a model that fits.
# --------------------------------------------------------------------------- #
def test_router_pin_does_not_fit():
    r = _resolve(Capability.I2V, 832, 480, 16, 5.0, "wan2.1-i2v-14b-720p")  # needs 14GB+
    assert r.is_err(), r
    assert r.error.code == ErrorCode.VRAM_EXCEEDED, r.error.code
    # and it did NOT quietly bind some other i2v model
    assert "wan2.1-i2v-14b-720p" in str(r.error), r.error


# --------------------------------------------------------------------------- #
# [6] StudioI2VSpec carries model_id / steps / cfg and round-trips through the bus
#     (asdict -> studio_i2v_from_dict re-validates through the same factory).
# --------------------------------------------------------------------------- #
def test_spec_round_trip():
    spec = make_studio_i2v(capability="i2v", width=832, height=480, fps=16,
                           vram_budget_gb=16.0, seed=3, model_id="wan2.1-i2v-14b-720p",
                           steps=28, cfg=6.0)
    assert spec.model_id == "wan2.1-i2v-14b-720p" and spec.steps == 28 and spec.cfg == 6.0
    back = studio_i2v_from_dict(dataclasses.asdict(spec))
    assert back == spec, (back, spec)


# --------------------------------------------------------------------------- #
# [7] build_capability_request threads the spec's model_id into the pin (so the
#     router — and the delegation decision — both see it).
# --------------------------------------------------------------------------- #
def test_build_request_threads_pin():
    spec = make_studio_i2v(capability="t2v", width=832, height=480, fps=16,
                           vram_budget_gb=9.0, model_id="wan2.1-t2v-1.3b")
    req = build_capability_request(spec)
    assert req.pinned_model_id == "wan2.1-t2v-1.3b", req


# --------------------------------------------------------------------------- #
# [8] Route accepts model_id (valid pin) -> 200 {job_id}.
# --------------------------------------------------------------------------- #
def test_route_accepts_model_id():
    r = client.post("/video/studio/i2v", json={
        "capability": "t2v", "resolution": {"width": 832, "height": 480, "fps": 16},
        "vram_budget_gb": 9.0, "model_id": "wan2.1-t2v-1.3b"})
    assert r.status_code == 200, (r.status_code, r.get_json())
    assert isinstance(r.get_json().get("job_id"), str), r.get_json()


# --------------------------------------------------------------------------- #
# [9] Route validates steps/cfg ranges -> 400 (steps 1-100, cfg 0-20).
# --------------------------------------------------------------------------- #
def test_route_rejects_bad_overrides():
    for body, why in (
        ({"steps": 0}, "steps<1"),
        ({"steps": 101}, "steps>100"),
        ({"cfg": -1}, "cfg<0"),
        ({"cfg": 20.5}, "cfg>20"),
    ):
        b = {"capability": "t2v", "width": 832, "height": 480, "fps": 16,
             "vram_budget_gb": 9.0}
        b.update(body)
        r = client.post("/video/studio/i2v", json=b)
        assert r.status_code == 400, (why, r.status_code, r.get_json())
    # a valid override still passes
    r = client.post("/video/studio/i2v", json={
        "capability": "t2v", "width": 832, "height": 480, "fps": 16,
        "vram_budget_gb": 9.0, "steps": 32, "cfg": 5.0})
    assert r.status_code == 200, (r.status_code, r.get_json())


# --------------------------------------------------------------------------- #
# [10] Bus adapter: a pin the router can't honor rides back as a JobError with the
#      pin code, NOT retryable (the same pin fails identically).
# --------------------------------------------------------------------------- #
def test_bus_pin_failure_is_joberror():
    spec = make_studio_i2v(capability="t2v", width=832, height=480, fps=16,
                           vram_budget_gb=9.0, model_id="no-such-model")
    result = run_studio_i2v(spec, "job-pin-fail")
    assert result.ok is False, result
    assert result.error.code == ErrorCode.PINNED_MODEL_UNAVAILABLE.value, result.error
    assert result.error.retryable is False, result.error


CHECKS = [
    ("router honors a valid pin", test_router_pin_valid),
    ("pin restricts candidates (honored over auto-pick)", test_router_pin_overrides_autopick),
    ("unknown model_id -> PINNED_MODEL_UNAVAILABLE", test_router_pin_unknown),
    ("pinned model doesn't serve capability -> PINNED_MODEL_UNAVAILABLE",
     test_router_pin_wrong_capability),
    ("pinned model doesn't fit -> VRAM_EXCEEDED (no silent fallback)",
     test_router_pin_does_not_fit),
    ("StudioI2VSpec carries model_id/steps/cfg + bus round-trip", test_spec_round_trip),
    ("build_capability_request threads the pin", test_build_request_threads_pin),
    ("route accepts model_id -> 200 {job_id}", test_route_accepts_model_id),
    ("route validates steps/cfg ranges -> 400", test_route_rejects_bad_overrides),
    ("bus adapter maps a pin failure to a not-retryable JobError",
     test_bus_pin_failure_is_joberror),
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
