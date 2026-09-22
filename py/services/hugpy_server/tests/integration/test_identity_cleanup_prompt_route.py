"""IDENTITY CLEANUP-PROMPT (C4 — route wiring) — makes the C1-C3 plumbed
``cleanup_prompt``/``negative_prompt`` fields REACHABLE from the generate route
(operator-requested 2026-07-15). C1-C3 landed the spec fields, the T-pose/reconstruction
render-prompt assembly, and per-profile persistence; this slice wires the ALREADY-PLUMBED
fields onto the actual HTTP surface: POST /video/identity-profiles/<slug>/generate now
resolves both from the request body (wins) else the profile's persisted
``gen_settings`` (falls back) else "" (byte-identical to today), and forwards them into
the ``make_identity_mesh(...)`` call the route enqueues.

NO GPU, NO network: media_bus.enqueue is mocked to CAPTURE the built spec (mirrors
test_identity_vision_setting.py section (b) exactly — same isolation harness, same
capture helper, same precedence-test shape, just for cleanup_prompt/negative_prompt
instead of vision_model).

Isolation mirrors test_identity_vision_setting.py exactly (rebind the store module
globals to temp dirs — env isolation does NOT work since constants read the .env file —
jail refs under the real UPLOADS_HOME, point the media bus at a temp DB). Run ALONE — the
identity test family cross-pollutes via the import-time IDENTITIES_HOME rebind.

Run (both as pytest and as a script):
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python -m pytest tests/test_identity_cleanup_prompt_route.py -q
  venv/bin/python tests/test_identity_cleanup_prompt_route.py
"""
from __future__ import annotations

import atexit
import logging
import os
import shutil
import sqlite3
import sys
import tempfile

logging.disable(logging.INFO)

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import importlib  # noqa: E402

from flask import Flask  # noqa: E402

from hugpy_video.intel import identity_profiles
from hugpy_video.intel import media_bus
from hugpy_platform.constants import UPLOADS_HOME, DEFAULT_ROOT

# --------------------------------------------------------------------------- #
# STORE + BUS isolation (mirrors test_identity_vision_setting.py exactly).
# --------------------------------------------------------------------------- #
_TMP_IDENTITIES = tempfile.mkdtemp(prefix="hugpy-cleanuproute-store-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
identity_profiles.IDENTITIES_HOME = _TMP_IDENTITIES
_TMP_PROJECTS = tempfile.mkdtemp(prefix="hugpy-cleanuproute-projects-")
identity_profiles.PROJECTS_HOME = _TMP_PROJECTS
_TMP_UPLOADS = tempfile.mkdtemp(prefix="hugpy-cleanuproute-uploads-", dir=UPLOADS_HOME)

_TMP_DB = tempfile.mkstemp(prefix="cleanuproute-bus-", suffix=".db")[1]
media_bus.DB_PATH = _TMP_DB
media_bus._initialized = False
with sqlite3.connect(_TMP_DB) as _c:
    _c.execute(
        "CREATE TABLE IF NOT EXISTS media_jobs ("
        "job_id TEXT PRIMARY KEY, name TEXT, status TEXT, spec_json TEXT, "
        "result_json TEXT, claim_token TEXT, created REAL, updated REAL, "
        "progress_json TEXT)")

vr = importlib.import_module("hugpy_server.app.routes.video_routes")
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


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _make_png(path: str, color=(180, 90, 40)) -> None:
    from PIL import Image
    Image.new("RGB", (64, 64), color).save(path)


_IMG = os.path.join(_TMP_UPLOADS, "hero.png")
_make_png(_IMG, (200, 40, 40))


def _create_profile(name: str) -> str:
    r = client.post("/video/identity-profiles",
                    json={"name": name, "reference_images": [_IMG], "notes": "a knight"})
    assert r.status_code == 201, (r.status_code, r.get_json())
    return r.get_json()["profile"]["slug"]


def _capture_generate_spec(slug: str, body: dict):
    """POST /generate with media_bus.enqueue mocked to capture the built spec; return it."""
    captured = {}

    def _fake_enqueue(name, spec, **kwargs):   # principal/owner attribution
        captured["name"] = name
        captured["spec"] = spec
        return "job-fake-1"

    orig = media_bus.enqueue
    media_bus.enqueue = _fake_enqueue
    try:
        r = client.post(f"/video/identity-profiles/{slug}/generate", json=body)
        assert r.status_code == 200, (r.status_code, r.get_json())
    finally:
        media_bus.enqueue = orig
    assert captured.get("name") == "identity_mesh_build", captured
    return captured["spec"]


# --------------------------------------------------------------------------- #
# ROUTE: POST /generate resolves cleanup_prompt/negative_prompt precedence
# request-body > gen_settings > "" into the enqueued IdentityMeshSpec.
# --------------------------------------------------------------------------- #
def test_generate_cleanup_precedence_request_body_wins():
    slug = _create_profile("Cleanup BodyWins")
    # Persist a gen_settings cleanup/negative; an explicit body value must OVERRIDE it.
    identity_profiles.set_gen_settings(
        slug, {"cleanup_prompt": "from settings", "negative_prompt": "settings negative"})
    spec = _capture_generate_spec(
        slug, {"cleanup_prompt": "from body", "negative_prompt": "body negative"})
    assert spec.cleanup_prompt == "from body", spec.cleanup_prompt
    assert spec.negative_prompt == "body negative", spec.negative_prompt


def test_generate_cleanup_precedence_falls_back_to_gen_settings():
    slug = _create_profile("Cleanup GsFallback")
    identity_profiles.set_gen_settings(
        slug, {"cleanup_prompt": "no object on back", "negative_prompt": "backpack, symbols"})
    # A bare body (no cleanup_prompt/negative_prompt) -> the persisted gen_settings value.
    spec = _capture_generate_spec(slug, {})
    assert spec.cleanup_prompt == "no object on back", spec.cleanup_prompt
    assert spec.negative_prompt == "backpack, symbols", spec.negative_prompt


def test_generate_cleanup_precedence_empty_everywhere_is_byte_identical():
    # Unset everywhere -> "" for both (== today's exact render, defaults-are-promises).
    slug = _create_profile("Cleanup Default")
    spec = _capture_generate_spec(slug, {})
    assert spec.cleanup_prompt == "", spec.cleanup_prompt
    assert spec.negative_prompt == "", spec.negative_prompt


def test_generate_cleanup_body_none_falls_back_to_gen_settings():
    # An explicit body value of None/"" behaves like "unset" -> falls back to gen_settings,
    # same treatment as vision_model's precedence (body.get returns None for a missing key
    # AND for an explicit null; both fall through to the persisted setting).
    slug = _create_profile("Cleanup BodyNone")
    identity_profiles.set_gen_settings(slug, {"cleanup_prompt": "persisted cleanup"})
    spec = _capture_generate_spec(slug, {"cleanup_prompt": None})
    assert spec.cleanup_prompt == "persisted cleanup", spec.cleanup_prompt


def test_generate_cleanup_only_one_field_set():
    # Setting only cleanup_prompt (not negative_prompt) leaves negative_prompt at its
    # own precedence resolution ("" here, since it's unset everywhere) — the two fields
    # are resolved independently.
    slug = _create_profile("Cleanup OnlyOne")
    identity_profiles.set_gen_settings(slug, {"cleanup_prompt": "only cleanup set"})
    spec = _capture_generate_spec(slug, {})
    assert spec.cleanup_prompt == "only cleanup set", spec.cleanup_prompt
    assert spec.negative_prompt == "", spec.negative_prompt


CHECKS = [
    ("route: /generate cleanup precedence — request body wins",
     test_generate_cleanup_precedence_request_body_wins),
    ("route: /generate cleanup precedence — falls back to gen_settings",
     test_generate_cleanup_precedence_falls_back_to_gen_settings),
    ("route: /generate cleanup precedence — empty everywhere is byte-identical",
     test_generate_cleanup_precedence_empty_everywhere_is_byte_identical),
    ("route: /generate cleanup — explicit body None falls back to gen_settings",
     test_generate_cleanup_body_none_falls_back_to_gen_settings),
    ("route: /generate cleanup — fields resolved independently",
     test_generate_cleanup_only_one_field_set),
]


def main() -> int:
    passed = failed = 0
    for i, (name, fn) in enumerate(CHECKS, 1):
        try:
            fn()
        except Exception as exc:  # surface EVERY divergence, not just the first
            failed += 1
            import traceback
            print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"[{i}] PASS  {name}")
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
