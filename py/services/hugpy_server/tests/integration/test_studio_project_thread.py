"""Studio i2v PROJECT thread — the optional human auto-archive NAME carried through
the studio spine (mirrors gen/scene/movie_schema's `project`).

Additive, backward-compatible: `project` is NON-CANONICAL metadata. It rides on the
StudioI2VSpec (so the bus stores + rehydrates it) but is NEVER threaded into
``produce.py`` / ``manifest.py``, so it can NOT enter the render ``content_hash``
(addressing is unchanged).

Covers:
  * make_studio_i2v(project=...) round-trips through asdict -> studio_i2v_from_dict
    preserving the project name; empty/whitespace-only coerces to None; a non-string
    project raises (studio's single-validator discipline); an older spec dict with no
    project key rehydrates cleanly (None).
  * `project` is NON-CANONICAL: two specs identical except project build the SAME render
    manifest content_hash via produce.py's exact manifest path — while a canonical field
    (prompt) DOES change it (the helper is genuinely hash-sensitive). Plus a source/
    signature guard: produce.py + manifest.py never reference `project`, and neither
    produce_clip nor make_render_manifest accepts a `project` parameter.
  * GET /video/projects lists distinct non-empty project names sorted case-insensitively
    across ALL media-bus job names (studio_i2v enqueued via POST /video/studio/i2v +
    generate_image enqueued directly); jobs with no project add no empty entry.

Script style matches tests/test_studio_lock_templates.py (plain python, numbered
PASS/FAIL, nonzero exit iff any FAILED). pytest is NOT installed in this venv.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/test_studio_project_thread.py
"""
from __future__ import annotations

import inspect
import logging
import os
import sys
import tempfile
from dataclasses import asdict

logging.disable(logging.INFO)
os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")
os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="studio-project-test-"))

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import importlib  # noqa: E402

from flask import Flask  # noqa: E402

# Isolate the media-bus DB to a throwaway file BEFORE any enqueue so this test never
# reads/writes the real shared job store (deterministic project list). media_bus reads
# DB_PATH via the module global at call time, so reassigning it here is sufficient; the
# route imports the SAME module object, so it sees the override too.
import hugpy_video.intel.media_bus as media_bus

_TMP_DB_DIR = tempfile.mkdtemp(prefix="studio-project-db-")
media_bus.DB_PATH = os.path.join(_TMP_DB_DIR, "media_jobs.db")
media_bus._initialized = False

from hugpy_video.intel.studio.job import make_studio_i2v, studio_i2v_from_dict, StudioI2VSpec
from hugpy_video.intel.studio.produce import make_render_manifest, produce_clip, resolve_sampler
import hugpy_video.intel.studio.produce as produce_mod
import hugpy_video.intel.studio.manifest as manifest_mod
from hugpy_video.intel.studio.router import CapabilityRouter
from hugpy_video.intel.studio.schemas import CapabilityRequest, Resolution, SeedBundle
from hugpy_video.intel.studio.enums import Capability
from hugpy_video.intel.studio.env import StudioEnv
from hugpy_video.intel.gen_schema import GenPromptPart, make_generate_image

vr = importlib.import_module("hugpy_server.app.routes.video_routes")
app = Flask(__name__)
app.register_blueprint(vr.video_bp)
client = app.test_client()


def _studio_env() -> StudioEnv:
    return StudioEnv(
        output_root="/out",
        weights_root="/weights",
        manifest_root="/manifests",
        master_colorspace="rec709",
        master_fps=8,
        max_vram_gb=24.0,
        loudness_target_lufs=-14.0,
        allow_unpinned=True,
    )


def _produce_manifest(spec: StudioI2VSpec):
    """Mirror produce_clip's manifest-building path (router.resolve -> resolve_sampler
    -> make_render_manifest) using EXACTLY the spec fields produce.py threads. `project`
    is deliberately NOT among them (produce.py never references it) — which is precisely
    why it cannot enter the content_hash. render_id is a per-render uuid and is excluded
    from canonical_inputs, so a fixed value here keeps the hash comparison clean."""
    req = CapabilityRequest(
        capability=Capability(spec.capability),
        target_resolution=Resolution(spec.width, spec.height, spec.fps),
        vram_budget_gb=spec.vram_budget_gb,
    )
    binding = CapabilityRouter().resolve(req).unwrap()
    sampler = resolve_sampler(
        binding.framework, req.target_resolution, steps=spec.steps, cfg=spec.cfg)
    return make_render_manifest(
        render_id="fixed-render-id",
        capability=req.capability,
        binding=binding,
        seeds=SeedBundle(global_seed=spec.seed, stage_seeds=(("base", spec.seed),)),
        sampler=sampler,
        resolution_ladder=(req.target_resolution,),
        env=_studio_env(),
        prompt=spec.prompt or "",
        negative_prompt=spec.negative or "",
        source_video=spec.source_video or "",
        reference_images=tuple(spec.reference_images or ()),
        control_image=spec.control_image or "",
        control_kind=spec.control_kind or "",
    )


# --------------------------------------------------------------------------- #
# (i) make_studio_i2v carries + coerces project; asdict/from_dict round-trips it
# --------------------------------------------------------------------------- #
def test_spec_roundtrips_project():
    spec = make_studio_i2v(width=64, height=64, fps=8, project="Alpha")
    assert spec.project == "Alpha", spec.project
    d = asdict(spec)
    assert d["project"] == "Alpha", d.get("project")
    spec2 = studio_i2v_from_dict(d)
    assert spec2.project == "Alpha", spec2.project

    # empty / whitespace-only coerces to None (empty -> None is acceptable)
    assert make_studio_i2v(width=64, height=64, fps=8, project="").project is None
    assert make_studio_i2v(width=64, height=64, fps=8, project="   ").project is None
    # surrounding whitespace is stripped
    assert make_studio_i2v(width=64, height=64, fps=8, project="  Beta ").project == "Beta"
    # default: absent -> None (backward compatible)
    assert make_studio_i2v(width=64, height=64, fps=8).project is None

    # a non-string project is a clean caller error (studio single-validator discipline)
    try:
        make_studio_i2v(width=64, height=64, fps=8, project=123)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-string project must raise ValueError")

    # backward-compat: an older spec dict with NO project key rehydrates cleanly (None)
    d.pop("project", None)
    assert studio_i2v_from_dict(d).project is None, "absent project key must rehydrate None"


# --------------------------------------------------------------------------- #
# (ii) project is NON-CANONICAL: it never changes the render content_hash, while a
#      genuine canonical field (prompt) does. Mirrors produce.py's manifest path.
# --------------------------------------------------------------------------- #
def test_project_not_in_content_hash():
    base = dict(width=64, height=64, fps=8, vram_budget_gb=0.5, seed=0, prompt="a fox")
    h_alpha = _produce_manifest(make_studio_i2v(project="Alpha", **base)).content_hash()
    h_beta = _produce_manifest(make_studio_i2v(project="Beta", **base)).content_hash()
    h_none = _produce_manifest(make_studio_i2v(**base)).content_hash()
    assert h_alpha == h_beta == h_none, (
        "specs identical except `project` must produce the SAME content_hash "
        f"(project is NON-canonical); got {h_alpha}, {h_beta}, {h_none}")

    # sensitivity guard: the helper genuinely keys on canonical inputs, so a DIFFERENT
    # prompt DOES change the hash — proving the equality above is meaningful, not a
    # helper that ignores its inputs.
    h_diff_prompt = _produce_manifest(
        make_studio_i2v(project="Alpha", **{**base, "prompt": "a whale"})).content_hash()
    assert h_diff_prompt != h_alpha, (
        "a different canonical field (prompt) MUST change the content_hash "
        "(the invariance helper is hash-sensitive)")


# --------------------------------------------------------------------------- #
# (iii) source/signature guard: project is NOWHERE in the addressing path, so it
#       structurally cannot enter the content_hash.
# --------------------------------------------------------------------------- #
def test_project_absent_from_addressing_path():
    prod_src = inspect.getsource(produce_mod)
    man_src = inspect.getsource(manifest_mod)
    assert "project" not in prod_src, "produce.py must never reference `project`"
    assert "project" not in man_src, "manifest.py must never reference `project`"
    assert "project" not in inspect.signature(produce_clip).parameters, (
        "produce_clip must not accept a `project` parameter")
    assert "project" not in inspect.signature(make_render_manifest).parameters, (
        "make_render_manifest must not accept a `project` parameter")


# --------------------------------------------------------------------------- #
# (iv) GET /video/projects lists distinct non-empty names sorted case-insensitively
#      across ALL job names; the enqueue route reads body["project"]; a no-project
#      job adds no empty entry.
# --------------------------------------------------------------------------- #
def test_projects_route_lists_distinct_sorted():
    # Studio jobs via the REAL enqueue route (proves body["project"] is threaded).
    for name in ("Zebra", "apple", "Beta", "Beta"):  # "Beta" twice -> distinct
        r = client.post("/video/studio/i2v", json={"project": name})
        assert r.status_code == 200, (name, r.status_code, r.get_data(as_text=True))
    # a studio job with NO project -> must NOT add an empty entry
    r = client.post("/video/studio/i2v", json={})
    assert r.status_code == 200, r.get_data(as_text=True)
    # a job with a whitespace-only project -> coerces to None -> no entry
    r = client.post("/video/studio/i2v", json={"project": "   "})
    assert r.status_code == 200, r.get_data(as_text=True)
    # a DIFFERENT job name (generate_image) also contributes (scan is across ALL names)
    media_bus.enqueue("generate_image", make_generate_image(
        parts=(GenPromptPart(kind="text", text="hi"),),
        model_id="some-model", width=64, height=64, steps=4, guidance=1.0,
        project="Gamma"))

    got = client.get("/video/projects")
    assert got.status_code == 200, got.status_code
    projects = got.get_json()["projects"]
    # distinct + no empty entry + sorted case-insensitively ("apple" < "Beta" < "Gamma"
    # < "Zebra"; a case-SENSITIVE ASCII sort would wrongly put "Zebra" before "apple").
    assert projects == ["apple", "Beta", "Gamma", "Zebra"], projects
    assert "" not in projects, projects


CHECKS = [
    ("make_studio_i2v carries + coerces project; asdict/from_dict round-trip",
     test_spec_roundtrips_project),
    ("project is NON-canonical (same content_hash; prompt still changes it)",
     test_project_not_in_content_hash),
    ("project absent from produce.py/manifest.py + produce_clip/make_render_manifest sigs",
     test_project_absent_from_addressing_path),
    ("GET /video/projects lists distinct non-empty names sorted case-insensitively",
     test_projects_route_lists_distinct_sorted),
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
