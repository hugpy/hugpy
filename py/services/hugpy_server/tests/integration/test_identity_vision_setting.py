"""PER-IDENTITY VISION MODEL (operator-requested) — the setting that lets an identity's
3D-imaging FRONT-SELECT step run on a chosen VL model (e.g. a 7B) instead of always the
fleet-default 3B.

Vision is used in the identity mesh pipeline in exactly ONE place — the relay's
``_select_front_view`` (picks the full-body reference before meshing) — which POSTs to
``/ml/vision``. This slice threads a per-identity ``vision_model`` from the Settings tab
all the way to that POST's ``model`` field. This test locks the whole thread WITHOUT a
GPU and WITHOUT any network:

  (a) STORE   — set_gen_settings accepts a valid image-text-to-text key, rejects an
                unknown / non-VL key (ProfileError -> the route's 400), and clears on
                None/"" (== the fleet default). Validation is against the LIVE registry,
                monkeypatched here to a fixed set for determinism.
  (b) ROUTE   — POST /generate resolves precedence request-body > gen_settings > None
                into the enqueued IdentityMeshSpec (media_bus.enqueue mocked to capture).
  (c) RELAY   — _select_front_view INCLUDES ``model`` in the /ml/vision body when
                spec.vision_model is set, and OMITS it (byte-identical to today -> the 3B)
                when it is None (requests module mocked, both ways).
  (d) SPEC    — IdentityMeshSpec carries vision_model through an asdict -> from_dict
                round-trip (and ""/whitespace normalizes to None).

Isolation mirrors test_identity_versions.py exactly (rebind the store module globals to
temp dirs — env isolation does NOT work since constants read the .env file — jail refs
under the real UPLOADS_HOME, point the media bus at a temp DB). Each check is independent
so one failure never masks the rest.

Run (both as pytest and as a script; run ALONE too — the identity test family cross-
pollutes when co-run via an import-time IDENTITIES_HOME rebind):
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python -m pytest tests/test_identity_vision_setting.py -q
  venv/bin/python tests/test_identity_vision_setting.py
"""
from __future__ import annotations

import atexit
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import asdict

logging.disable(logging.INFO)

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import importlib  # noqa: E402

from flask import Flask  # noqa: E402

from hugpy_video.intel import identity_profiles
from hugpy_video.intel import media_bus
from hugpy_video.intel.runners import identity_render_relay
from hugpy_video.intel.identity_reconstruction_schema import (
    make_identity_mesh,
    identity_mesh_from_dict,
)
from hugpy_platform.constants import UPLOADS_HOME, DEFAULT_ROOT

# --------------------------------------------------------------------------- #
# STORE + BUS isolation (mirrors test_identity_versions.py exactly).
# --------------------------------------------------------------------------- #
_TMP_IDENTITIES = tempfile.mkdtemp(prefix="hugpy-vismodel-store-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
identity_profiles.IDENTITIES_HOME = _TMP_IDENTITIES
_TMP_PROJECTS = tempfile.mkdtemp(prefix="hugpy-vismodel-projects-")
identity_profiles.PROJECTS_HOME = _TMP_PROJECTS
_TMP_UPLOADS = tempfile.mkdtemp(prefix="hugpy-vismodel-uploads-", dir=UPLOADS_HOME)

_TMP_DB = tempfile.mkstemp(prefix="vismodel-bus-", suffix=".db")[1]
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

# A fixed, deterministic "live registry" of image-text-to-text keys. The real
# _valid_vision_model_keys does a lazy import of the fleet's VISION_MODELS_REGISTRY;
# monkeypatching it keeps this test independent of what's actually registered on the box
# while still exercising the exact validation path (membership in the VL set).
_VALID_VL = {
    "unsloth~Qwen2.5-VL-7B-Instruct-GGUF",
    "SandLogicTechnologies~Qwen2.5-VL-7B-Instruct-GGUF",
    "Qwen2.5-VL-3B-Instruct-GGUF",
}
identity_profiles._valid_vision_model_keys = lambda: set(_VALID_VL)


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


# --------------------------------------------------------------------------- #
# (a) STORE: set_gen_settings validates vision_model against the live VL registry.
# --------------------------------------------------------------------------- #
def test_set_gen_settings_accepts_valid_vl_key():
    slug = _create_profile("Vis Accept")
    key = "unsloth~Qwen2.5-VL-7B-Instruct-GGUF"
    prof = identity_profiles.set_gen_settings(slug, {"vision_model": key})
    assert prof is not None
    assert prof["gen_settings"]["vision_model"] == key, prof["gen_settings"]


def test_set_gen_settings_rejects_unknown_or_non_vl_key():
    slug = _create_profile("Vis Reject")
    # An unknown / non-VL key -> ProfileError (the route turns this into a clean 400).
    try:
        identity_profiles.set_gen_settings(slug, {"vision_model": "not-a-real-model"})
    except identity_profiles.ProfileError as exc:
        assert exc.code == "invalid_profile", exc.code
    else:
        raise AssertionError("expected ProfileError for an unknown vision_model key")

    # And the store rejects it through the PATCH route as a 400 too.
    r = client.patch(f"/video/identity-profiles/{slug}/settings",
                     json={"vision_model": "text-to-image~generator-only"})
    assert r.status_code == 400, (r.status_code, r.get_json())


def test_set_gen_settings_clears_on_none_or_empty():
    slug = _create_profile("Vis Clear")
    key = "SandLogicTechnologies~Qwen2.5-VL-7B-Instruct-GGUF"
    identity_profiles.set_gen_settings(slug, {"vision_model": key})
    # None clears back to the fleet default (stored as None).
    prof = identity_profiles.set_gen_settings(slug, {"vision_model": None})
    assert prof["gen_settings"]["vision_model"] is None, prof["gen_settings"]
    # "" (the UI's Default option) also clears — never a validation error.
    identity_profiles.set_gen_settings(slug, {"vision_model": key})
    prof2 = identity_profiles.set_gen_settings(slug, {"vision_model": ""})
    assert prof2["gen_settings"]["vision_model"] is None, prof2["gen_settings"]


def test_gen_settings_default_is_none_zero_regression():
    # A brand-new profile carries vision_model=None on the wire (== fleet default), so a
    # bare Generate is byte-identical to before this setting existed.
    slug = _create_profile("Vis Default")
    prof = client.get(f"/video/identity-profiles/{slug}").get_json()["profile"]
    assert "vision_model" in prof["gen_settings"], prof["gen_settings"]
    assert prof["gen_settings"]["vision_model"] is None, prof["gen_settings"]


# --------------------------------------------------------------------------- #
# (b) ROUTE: POST /generate resolves precedence request > gen_settings > None into
#     the enqueued spec. media_bus.enqueue is mocked to CAPTURE the spec.
# --------------------------------------------------------------------------- #
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


def test_generate_precedence_request_body_wins():
    slug = _create_profile("Gen BodyWins")
    # Persist a gen_settings vision_model; an explicit body value must OVERRIDE it.
    identity_profiles.set_gen_settings(
        slug, {"vision_model": "Qwen2.5-VL-3B-Instruct-GGUF"})
    spec = _capture_generate_spec(
        slug, {"vision_model": "unsloth~Qwen2.5-VL-7B-Instruct-GGUF"})
    assert spec.vision_model == "unsloth~Qwen2.5-VL-7B-Instruct-GGUF", spec.vision_model


def test_generate_precedence_falls_back_to_gen_settings():
    slug = _create_profile("Gen GsFallback")
    key = "SandLogicTechnologies~Qwen2.5-VL-7B-Instruct-GGUF"
    identity_profiles.set_gen_settings(slug, {"vision_model": key})
    # A bare body (no vision_model) -> the persisted gen_settings value is used.
    spec = _capture_generate_spec(slug, {})
    assert spec.vision_model == key, spec.vision_model


def test_generate_precedence_auto_when_unset():
    """Unset everywhere -> the AUTO preference (operator 2026-07-15: '7B should be
    default if it is available'): preferred_identity_vision_model() — a 7B key when
    the live registry has one, else None (== fleet-default 3B, no model sent)."""
    slug = _create_profile("Gen AutoDefault")
    expected = identity_profiles.preferred_identity_vision_model()
    spec = _capture_generate_spec(slug, {})
    assert spec.vision_model == expected, (spec.vision_model, expected)


def test_generate_precedence_none_when_unset_and_no_7b():
    """With NO 7B in the fleet (auto preference monkeypatched to None), unset
    everywhere -> None — the original fleet-default contract still holds."""
    slug = _create_profile("Gen NoneDefault")
    orig = identity_profiles.preferred_identity_vision_model
    identity_profiles.preferred_identity_vision_model = lambda: None
    try:
        spec = _capture_generate_spec(slug, {})
    finally:
        identity_profiles.preferred_identity_vision_model = orig
    assert spec.vision_model is None, spec.vision_model


# --------------------------------------------------------------------------- #
# (c) RELAY: _select_front_view includes `model` when set, omits it when None.
# --------------------------------------------------------------------------- #
class _FakeResp:
    status_code = 200

    def json(self):
        # Answer "yes" so the FIRST candidate is chosen and exactly one POST is made.
        return {"ok": True, "text": "Yes, the full body is visible."}


class _FakeRequests:
    """Stand-in for the ``requests`` module the runner is handed (mirrors how
    test_identity_render_relay mocks the HTTP surface). Records every POST body."""
    RequestException = Exception

    def __init__(self):
        self.posts = []

    def post(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResp()


def test_select_front_view_includes_model_when_set():
    cand = os.path.join(_TMP_UPLOADS, "cand_model.png")
    _make_png(cand, (10, 20, 30))
    fake = _FakeRequests()
    chosen, checked = identity_render_relay._select_front_view(
        [cand], "slug-x", fake, model="unsloth~Qwen2.5-VL-7B-Instruct-GGUF")
    assert chosen == cand, chosen
    assert checked == 1, checked
    assert len(fake.posts) == 1, fake.posts
    body = fake.posts[0]["json"]
    assert body.get("model") == "unsloth~Qwen2.5-VL-7B-Instruct-GGUF", body
    assert "image_b64" in body and "prompt" in body, body


def test_select_front_view_omits_model_when_none():
    cand = os.path.join(_TMP_UPLOADS, "cand_nomodel.png")
    _make_png(cand, (30, 20, 10))
    fake = _FakeRequests()
    chosen, checked = identity_render_relay._select_front_view(
        [cand], "slug-y", fake, model=None)
    assert chosen == cand, chosen
    assert len(fake.posts) == 1, fake.posts
    body = fake.posts[0]["json"]
    # No `model` field at all -> /ml/vision falls back to DEFAULT_VISION_MODEL (the 3B),
    # byte-identical to before this setting existed.
    assert "model" not in body, body
    assert "image_b64" in body and "prompt" in body, body


def test_select_front_view_empty_string_model_omits():
    # An empty-string model behaves like None (falsy) -> no `model` field.
    cand = os.path.join(_TMP_UPLOADS, "cand_empty.png")
    _make_png(cand, (40, 50, 60))
    fake = _FakeRequests()
    identity_render_relay._select_front_view([cand], "slug-z", fake, model="")
    assert "model" not in fake.posts[0]["json"], fake.posts[0]["json"]


# --------------------------------------------------------------------------- #
# (d) SPEC: IdentityMeshSpec round-trips vision_model (asdict -> from_dict), and
#     ""/whitespace normalizes to None (== fleet default).
# --------------------------------------------------------------------------- #
def test_mesh_spec_round_trips_vision_model():
    key = "unsloth~Qwen2.5-VL-7B-Instruct-GGUF"
    spec = make_identity_mesh(
        slug="round", recon_id="r1", view_sources=[("front", _IMG)], vision_model=key)
    assert spec.vision_model == key, spec.vision_model
    revived = identity_mesh_from_dict(asdict(spec))
    assert revived.vision_model == key, revived.vision_model


def test_mesh_spec_none_and_blank_normalize_to_none():
    # Default (no vision_model) -> None.
    spec_default = make_identity_mesh(
        slug="rd", recon_id="r2", view_sources=[("front", _IMG)])
    assert spec_default.vision_model is None, spec_default.vision_model
    # ""/whitespace -> None (== fleet default), never carried as a blank string.
    spec_blank = make_identity_mesh(
        slug="rb", recon_id="r3", view_sources=[("front", _IMG)], vision_model="   ")
    assert spec_blank.vision_model is None, spec_blank.vision_model
    assert identity_mesh_from_dict(asdict(spec_blank)).vision_model is None


CHECKS = [
    ("store: set_gen_settings accepts a valid VL key", test_set_gen_settings_accepts_valid_vl_key),
    ("store: set_gen_settings rejects unknown/non-VL key (400/ProfileError)", test_set_gen_settings_rejects_unknown_or_non_vl_key),
    ("store: set_gen_settings clears on None/''", test_set_gen_settings_clears_on_none_or_empty),
    ("store: default vision_model is None (zero regression)", test_gen_settings_default_is_none_zero_regression),
    ("route: /generate precedence — request body wins", test_generate_precedence_request_body_wins),
    ("route: /generate precedence — falls back to gen_settings", test_generate_precedence_falls_back_to_gen_settings),
    ("route: /generate precedence — auto (7B) when unset", test_generate_precedence_auto_when_unset),
    ("route: /generate precedence — None when unset and no 7B", test_generate_precedence_none_when_unset_and_no_7b),
    ("relay: _select_front_view includes model when set", test_select_front_view_includes_model_when_set),
    ("relay: _select_front_view omits model when None", test_select_front_view_omits_model_when_none),
    ("relay: _select_front_view omits model for empty string", test_select_front_view_empty_string_model_omits),
    ("spec: IdentityMeshSpec round-trips vision_model", test_mesh_spec_round_trips_vision_model),
    ("spec: None/blank vision_model normalize to None", test_mesh_spec_none_and_blank_normalize_to_none),
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
