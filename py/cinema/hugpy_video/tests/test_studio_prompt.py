"""C-prompt slice conformance — the studio i2v text prompt (+ negative prompt).

Locks the text-conditioning slice as executable checks, in the same script style
as ``test_studio_conformance.py`` (plain python, ``__main__`` guard, numbered
``[n] PASS`` / ``[n] FAIL`` lines, nonzero exit iff any check FAILED, every check
independently run so a failing one never masks the rest). pytest is NOT installed
in this venv, so there are no fixtures.

Invariant under test: the prompt genuinely changes the output, so it is part of the
reproducibility key — it enters ``RenderManifest.canonical_inputs()`` and thus the
``content_hash`` — and it is threaded end to end
(spec -> produce_clip -> make_render_manifest -> RenderManifest -> the Wan runner).

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_prompt.py
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import asdict

logging.disable(logging.INFO)

# STUDIO_ALLOW_UNPINNED is not strictly required (routing does not gate on pinning),
# but set it defensively so nothing in this run can trip the fail-loud pin contract.
os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_video.intel.studio import (
    Capability,
    CapabilityRequest,
    CapabilityRouter,
    DeterminismClass,
    Framework,
    Precision,
    RenderManifest,
    Resolution,
    SamplerConfig,
    SeedBundle,
    Task,
    make_render_manifest,
    render_manifest_from_dict,
    render_manifest_to_dict,
)
from hugpy_video.intel.studio.env import StudioEnv
from hugpy_video.intel.studio.job import make_studio_i2v, studio_i2v_from_dict
from hugpy_video.intel.studio.produce import produce_clip
import hugpy_video.intel.studio.produce as produce_mod
from hugpy_video.intel.studio.runners import wan_i2v as wan_i2v_mod

_FFMPEG = shutil.which("ffmpeg") is not None
R720V = Resolution(720, 1280, 24)  # portrait 720p -> binds a real Wan i2v model @16GB


def _studio_env(master_fps: int = 24) -> StudioEnv:
    return StudioEnv(
        output_root="/out",
        weights_root="/weights",
        manifest_root="/manifests",
        master_colorspace="rec709",
        master_fps=master_fps,
        max_vram_gb=24.0,
        loudness_target_lufs=-14.0,
        allow_unpinned=True,
    )


def _direct_manifest(*, prompt: str = "", negative_prompt: str = "") -> RenderManifest:
    """A minimal, valid RenderManifest built directly (no router/env), varying only
    the text conditioning so the hash effect is isolated to the prompt fields."""
    return RenderManifest(
        render_id="rp",
        capability=Capability.I2V,
        model_id="wan2.1-i2v-14b-720p",
        weight_hash=None,
        framework=Framework.WAN,
        task=Task.I2V,
        precision=Precision.BF16,
        seeds=SeedBundle(global_seed=1234, stage_seeds=(("base", 1234),)),
        sampler=SamplerConfig(sampler="euler", scheduler="normal", steps=30, cfg=6.0),
        resolution_ladder=(Resolution(1280, 720, 24),),
        determinism_class=DeterminismClass.SEEDED_APPROX,
        env_snapshot=(),
        prompt=prompt,
        negative_prompt=negative_prompt,
    )


def _synth_request(*, res=Resolution(320, 180, 12), budget=0.5) -> CapabilityRequest:
    """Tiny-budget CAP-I2V -> the router deterministically binds synthetic-i2v."""
    return CapabilityRequest(
        capability=Capability.I2V, target_resolution=res, vram_budget_gb=budget)


# --------------------------------------------------------------------------- #
# (i) prompt participates in the content hash; identical prompt -> same hash
# --------------------------------------------------------------------------- #
def test_prompt_changes_hash_same_prompt_same_hash():
    h_a = _direct_manifest(prompt="a red fox at dawn").content_hash()
    h_b = _direct_manifest(prompt="a blue whale at night").content_hash()
    assert h_a != h_b, "manifests differing only in prompt must NOT collide on content_hash"

    # determinism preserved: identical prompt -> identical hash
    assert (_direct_manifest(prompt="a red fox at dawn").content_hash()
            == _direct_manifest(prompt="a red fox at dawn").content_hash()), (
        "identical prompt must hash equal (determinism preserved)")

    # empty prompt still participates in the key (its value is just "")
    assert (_direct_manifest(prompt="").content_hash()
            == _direct_manifest(prompt="").content_hash()), (
        "two empty-prompt manifests must hash equal")
    assert (_direct_manifest(prompt="").content_hash()
            != _direct_manifest(prompt="x").content_hash()), (
        "an empty prompt must not collide with a non-empty prompt")

    # negative_prompt is also in the key
    assert (_direct_manifest(negative_prompt="blurry, lowres").content_hash()
            != _direct_manifest(negative_prompt="").content_hash()), (
        "negative_prompt must participate in the content_hash too")


# --------------------------------------------------------------------------- #
# (i.b) the factory threads prompt into the hash; to_dict/from_dict round-trips it
# --------------------------------------------------------------------------- #
def test_factory_and_roundtrip_carry_prompt():
    r = CapabilityRouter().resolve(_synth_request())
    assert r.is_ok(), "tiny-budget CAP-I2V must route (synthetic-i2v)"
    binding = r.unwrap()

    def _mk(prompt, negative):
        return make_render_manifest(
            render_id="rf", capability=Capability.I2V, binding=binding,
            seeds=SeedBundle(global_seed=7, stage_seeds=(("base", 7),)),
            sampler=SamplerConfig(sampler="euler", scheduler="normal", steps=1, cfg=1.0),
            resolution_ladder=(Resolution(320, 180, 12),), env=_studio_env(),
            prompt=prompt, negative_prompt=negative)

    m_p = _mk("neon skyline", "washed out")
    assert m_p.prompt == "neon skyline", "factory must thread prompt onto the manifest"
    assert m_p.negative_prompt == "washed out", "factory must thread negative_prompt"
    assert _mk("neon skyline", "washed out").content_hash() != _mk("swamp", "washed out").content_hash(), (
        "factory-built manifests differing only by prompt must differ in hash")

    # round-trip preserves the prompt fields AND the hash
    d = render_manifest_to_dict(m_p)
    assert d["prompt"] == "neon skyline" and d["negative_prompt"] == "washed out", (
        "render_manifest_to_dict must serialize prompt/negative_prompt")
    m2 = render_manifest_from_dict(d)
    assert m2.prompt == "neon skyline" and m2.negative_prompt == "washed out", (
        "render_manifest_from_dict must rehydrate prompt/negative_prompt")
    assert m2.content_hash() == m_p.content_hash(), (
        "to_dict->from_dict must preserve content_hash with the prompt fields")

    # backward-compat: a manifest dict serialized before this field existed rehydrates
    # with prompt="" (absent key tolerated) and does not raise.
    legacy = render_manifest_to_dict(_mk("", ""))
    legacy.pop("prompt", None)
    legacy.pop("negative_prompt", None)
    m3 = render_manifest_from_dict(legacy)
    assert m3.prompt == "" and m3.negative_prompt == "", (
        "a pre-C-prompt manifest dict must rehydrate with empty prompt fields")


# --------------------------------------------------------------------------- #
# (ii) StudioI2VSpec round-trips prompt + negative (to_dict -> from_dict)
# --------------------------------------------------------------------------- #
def test_spec_roundtrips_prompt():
    spec = make_studio_i2v(
        width=64, height=64, fps=8,
        prompt="a red fox at dawn", negative="blurry, lowres")
    assert spec.prompt == "a red fox at dawn"
    assert spec.negative == "blurry, lowres"
    d = asdict(spec)
    assert d["prompt"] == "a red fox at dawn" and d["negative"] == "blurry, lowres", (
        "asdict(spec) must carry prompt + negative")
    spec2 = studio_i2v_from_dict(d)
    assert spec2.prompt == spec.prompt, "from_dict must preserve prompt"
    assert spec2.negative == spec.negative, "from_dict must preserve negative"
    # backward-compat: an older spec dict with neither key rehydrates to None
    d.pop("prompt", None)
    d.pop("negative", None)
    spec3 = studio_i2v_from_dict(d)
    assert spec3.prompt is None and spec3.negative is None, (
        "a spec dict without prompt/negative must rehydrate cleanly (None)")


# --------------------------------------------------------------------------- #
# (iii) produce_clip WITH a prompt -> a clip, and manifest.json records the prompt
# --------------------------------------------------------------------------- #
def test_produce_clip_with_prompt_records_prompt():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping produce_clip-with-prompt check)")
        return
    env = _studio_env()
    req = _synth_request()
    the_prompt = "a lighthouse in a storm, cinematic"
    the_negative = "blurry"
    out_root = tempfile.mkdtemp(prefix="studio-cprompt-")
    try:
        res = produce_clip(req, env=env, out_root=out_root,
                           prompt=the_prompt, negative_prompt=the_negative)
        assert res.is_ok(), f"produce_clip with a prompt must yield Ok(Artifact); got {res}"
        art = res.unwrap()
        assert os.path.isfile(art.path) and os.path.getsize(art.path) > 0, "clip mp4 non-empty"

        # the clip's manifest.json sidecar must record the prompt (INV-1)
        clip_dir = os.path.dirname(art.path)
        with open(os.path.join(clip_dir, "manifest.json"), "r", encoding="utf-8") as fh:
            man = json.load(fh)
        assert man["prompt"] == the_prompt, (
            f"manifest.json must record the prompt; got {man.get('prompt')!r}")
        assert man["negative_prompt"] == the_negative, (
            f"manifest.json must record the negative_prompt; got {man.get('negative_prompt')!r}")

        # the prompt actually re-addresses the clip: a different prompt -> a different
        # content-addressed dir (proves the prompt is in the hash end to end).
        res2 = produce_clip(req, env=env, out_root=out_root,
                            prompt="a totally different scene", negative_prompt=the_negative)
        assert res2.is_ok()
        assert os.path.dirname(res2.unwrap().path) != clip_dir, (
            "a different prompt must produce a different content-addressed clip dir")
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (iv) the Wan runner reads manifest.prompt (targeted; NOT a real GPU render)
# --------------------------------------------------------------------------- #
def test_wan_runner_reads_manifest_prompt():
    # Behavioral half: route a real Wan i2v binding through produce_clip with a
    # prompt and INTERCEPT the runner to capture the manifest it is handed. The Wan
    # runner passes manifest.prompt straight to the pipeline, so a captured manifest
    # whose .prompt == the requested prompt proves "the value the runner would pass
    # to the pipeline == manifest.prompt" without a GPU render.
    env = _studio_env()
    r = CapabilityRouter().resolve(CapabilityRequest(
        capability=Capability.I2V, target_resolution=R720V, vram_budget_gb=16.0))
    assert r.is_ok(), "CAP-I2V @ portrait-720p / 16GB must route to a real Wan model"
    assert r.unwrap().framework == Framework.WAN, "fixture must bind a Wan model"

    the_prompt = "a dragon over a neon city, 35mm"
    the_negative = "lowres, watermark"
    captured = {}
    key = (Framework.WAN, Task.I2V)
    orig = produce_mod._DISPATCH[key]

    def _capture(manifest, out_root, start_image=None):
        captured["m"] = manifest
        return orig(manifest, out_root, start_image=start_image)  # real (Err: deps/gpu)

    produce_mod._DISPATCH[key] = _capture
    try:
        produce_clip(
            CapabilityRequest(capability=Capability.I2V,
                              target_resolution=R720V, vram_budget_gb=16.0),
            env=env, out_root=tempfile.gettempdir(),
            prompt=the_prompt, negative_prompt=the_negative)
    finally:
        produce_mod._DISPATCH[key] = orig

    assert "m" in captured, "the Wan runner must have been dispatched"
    assert captured["m"].prompt == the_prompt, (
        f"the manifest handed to the Wan runner must carry the prompt; "
        f"got {captured['m'].prompt!r}")
    assert captured["m"].negative_prompt == the_negative, (
        f"the manifest handed to the Wan runner must carry the negative_prompt; "
        f"got {captured['m'].negative_prompt!r}")

    # Source half: the runner no longer hardcodes prompt="" — it forwards
    # manifest.prompt / manifest.negative_prompt to the pipeline call.
    src = inspect.getsource(wan_i2v_mod)
    assert 'prompt=""' not in src, "wan_i2v must not hardcode prompt=\"\" anymore"
    assert "prompt = manifest.prompt" in src, "wan_i2v must read manifest.prompt"
    assert "prompt=prompt" in src, "wan_i2v must forward the prompt to the pipeline"
    assert "negative_prompt=negative_prompt" in src, (
        "wan_i2v must forward the negative_prompt to the pipeline")


CHECKS = [
    ("C-prompt participates in content_hash (diff prompt -> diff hash; same -> same)",
     test_prompt_changes_hash_same_prompt_same_hash),
    ("factory threads prompt + to_dict/from_dict round-trips it (and hash)",
     test_factory_and_roundtrip_carry_prompt),
    ("StudioI2VSpec round-trips prompt + negative", test_spec_roundtrips_prompt),
    ("produce_clip with a prompt -> clip + manifest.json records prompt",
     test_produce_clip_with_prompt_records_prompt),
    ("Wan runner reads manifest.prompt (captured manifest + source guard)",
     test_wan_runner_reads_manifest_prompt),
]


def main() -> int:
    passed = 0
    failed = 0
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
