"""Task 3b — text-to-video (T2V) capability conformance for the studio spine.

Locks the T2V slice as executable checks, in the same script style as
``test_studio_conformance.py`` / ``test_studio_prompt.py`` / ``test_studio_cancel.py``
(plain python, ``__main__`` guard, numbered ``[n] PASS`` / ``[n] FAIL`` lines,
nonzero exit iff any check FAILED, every check independently run so a failing one
never masks the rest). pytest is NOT installed in this venv, so there are no
fixtures.

Invariant under test: a studio T2V clip flows through the SAME ``studio_i2v`` bus
job / spine as i2v — route -> ``make_studio_i2v(capability="t2v")`` -> router
(capability=T2V) -> ``(framework, Task.T2V)`` dispatch -> runner ->
content-addressed clip. The synthetic T2V runner demos it with NO GPU (tiny
budget); the real Wan T2V runner degrades gracefully (DEPS_MISSING here) at a real
budget, and the last-resort rule holds for t2v (a real Wan t2v model always
outranks the synthetic placeholder when both fit).

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_t2v.py
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile

logging.disable(logging.INFO)

os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_platform.constants import DEFAULT_ROOT
from hugpy_video.intel import media_bus
from hugpy_video.intel.runners.studio_i2v import run_studio_i2v
from hugpy_video.intel.studio.artifacts import Artifact
from hugpy_video.intel.studio.enums import Capability, Framework, Task
from hugpy_video.intel.studio.env import StudioEnv
from hugpy_video.intel.studio.errors import ErrorCode, StageError
from hugpy_video.intel.studio.job import make_studio_i2v
from hugpy_video.intel.studio.manifest import make_render_manifest
from hugpy_video.intel.studio.produce import _DISPATCH, produce_clip
from hugpy_video.intel.studio.registry import (
    MODEL_REGISTRY,
    PLANNED_CAPABILITIES,
    runner_for,
    validate_registry,
)
from hugpy_video.intel.studio.router import CapabilityRouter
from hugpy_video.intel.studio.runners.synthetic import run_synthetic_t2v, synthesize_frame
from hugpy_video.intel.studio.runners.wan_t2v import run_wan_t2v
from hugpy_video.intel.studio.schemas import (
    CapabilityRequest,
    Resolution,
    SamplerConfig,
    SeedBundle,
)

_FFMPEG = shutil.which("ffmpeg") is not None
_FFPROBE = shutil.which("ffprobe")

R480 = Resolution(832, 480, 16)   # a real Wan t2v res the capped synthetic never offers
R_TINY = Resolution(320, 180, 12)  # covered by synthetic-t2v (<=512) and a real Wan t2v
R_SHADOW = Resolution(320, 180, 12)


def _studio_env(master_fps: int = 12) -> StudioEnv:
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


def _ffprobe_nb_frames(path: str) -> int:
    out = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    return int((out.stdout or "0").strip() or "0")


def _read_sidecar(clip_path: str) -> dict:
    with open(os.path.join(os.path.dirname(clip_path), "manifest.json")) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# [1] Registry wiring — the two t2v dispatch entries + the synthetic t2v prover,
#     and validate_registry stays total (T2V is served, not PLANNED).
# --------------------------------------------------------------------------- #
def test_registry_t2v_wired():
    assert _DISPATCH.get((Framework.SYNTHETIC, Task.T2V)) is run_synthetic_t2v, (
        "produce._DISPATCH must map (SYNTHETIC, T2V) -> run_synthetic_t2v")
    assert _DISPATCH.get((Framework.WAN, Task.T2V)) is run_wan_t2v, (
        "produce._DISPATCH must map (WAN, T2V) -> run_wan_t2v")

    syn_spec = runner_for(Framework.SYNTHETIC, Task.T2V)
    assert syn_spec is not None and syn_spec.entrypoint == (
        "hugpy_video.intel.studio.runners.synthetic:run_synthetic_t2v"), (
        f"SYNTHETIC t2v entrypoint wrong; got {syn_spec}")
    wan_spec = runner_for(Framework.WAN, Task.T2V)
    assert wan_spec is not None and wan_spec.entrypoint == (
        "hugpy_video.intel.studio.runners.wan_t2v:run_wan_t2v"), (
        f"WAN t2v entrypoint must resolve to the real runner (not the dormant "
        f"runners.wan:t2v placeholder); got {wan_spec}")

    # T2V is a served capability, NOT parked in PLANNED_CAPABILITIES.
    assert Capability.T2V not in PLANNED_CAPABILITIES, (
        "Capability.T2V must not be PLANNED — it is served by real Wan t2v models")

    # models present: the synthetic prover (last-resort) + the two staged Wan t2v.
    syn = MODEL_REGISTRY.get("synthetic-t2v")
    assert syn is not None and syn.synthetic is True, "synthetic-t2v must exist + be last-resort"
    assert Capability.T2V in syn.capabilities and Task.T2V in syn.tasks
    for mid in ("wan2.1-t2v-1.3b",):
        m = MODEL_REGISTRY.get(mid)
        assert m is not None and Capability.T2V in m.capabilities, f"{mid} must serve T2V"
        assert m.family == Framework.WAN and Task.T2V in m.tasks
    # wan2.2-t2v-a14b / wan2.2-i2v-a14b were REMOVED from the seed (operator,
    # 2026-08-13; see models_seed) — they must not quietly come back.
    assert MODEL_REGISTRY.get("wan2.2-t2v-a14b") is None

    # the join stays total (dev tree runs unpinned; env gate is set at module top).
    validate_registry()


# --------------------------------------------------------------------------- #
# [2] Router: a TINY-budget T2V binds the synthetic prover (the demo path).
# --------------------------------------------------------------------------- #
def test_router_t2v_tiny_budget_binds_synthetic():
    r = CapabilityRouter().resolve(CapabilityRequest(
        capability=Capability.T2V, target_resolution=R_TINY, vram_budget_gb=0.5))
    assert r.is_ok(), f"tiny-budget T2V must route (to synthetic); got {r}"
    b = r.unwrap()
    assert b.model_id == "synthetic-t2v", f"tiny T2V must bind synthetic-t2v; got {b.model_id}"
    assert b.framework == Framework.SYNTHETIC and b.task == Task.T2V


# --------------------------------------------------------------------------- #
# [3] Router: a 16GB T2V budget binds a REAL Wan t2v model (not synthetic).
# --------------------------------------------------------------------------- #
def test_router_t2v_16gb_binds_real_wan():
    r = CapabilityRouter().resolve(CapabilityRequest(
        capability=Capability.T2V, target_resolution=R480, vram_budget_gb=16.0,
        preferred_framework=Framework.WAN))
    assert r.is_ok(), f"16GB T2V @ 480p must route to a real Wan t2v model; got {r}"
    b = r.unwrap()
    assert b.framework == Framework.WAN and b.task == Task.T2V, (
        f"16GB T2V must bind a Wan t2v model; got {b.framework}/{b.task}")
    assert b.model_id in ("wan2.1-t2v-1.3b", "wan2.2-t2v-a14b"), (
        f"must be one of the staged Wan t2v models; got {b.model_id}")
    assert b.framework != Framework.SYNTHETIC


# --------------------------------------------------------------------------- #
# [4] Last-resort holds for T2V: at a budget where BOTH the synthetic prover and
#     a real Wan t2v model FIT + COVER the target, the REAL model still wins (an
#     ACTIVE tie-break over synthetic, not synthetic merely falling out on fit).
# --------------------------------------------------------------------------- #
def test_router_t2v_synthetic_never_shadows_real():
    syn = MODEL_REGISTRY["synthetic-t2v"]
    # precondition: synthetic genuinely fits + covers this request (so a real win
    # is a real last-resort tie-break, mirroring the i2v shadow check).
    assert syn.supports_resolution(R_SHADOW) and syn.vram.min_gb() <= 6.0, (
        "precondition: synthetic-t2v must itself fit this request")
    r = CapabilityRouter().resolve(CapabilityRequest(
        capability=Capability.T2V, target_resolution=R_SHADOW, vram_budget_gb=6.0))
    assert r.is_ok(), f"6GB T2V @ tiny res must route; got {r}"
    b = r.unwrap()
    assert b.framework != Framework.SYNTHETIC, (
        f"synthetic must NOT shadow a real Wan t2v model when a real one fits; got {b.model_id}")
    assert b.model_id == "wan2.1-t2v-1.3b" and b.task == Task.T2V, (
        f"the real Wan t2v model must win the last-resort tie-break; got {b.model_id}")


# --------------------------------------------------------------------------- #
# [5] produce_clip T2V via synthetic -> a real playable MP4 (ffprobe), and the
#     manifest sidecar records capability="t2v" + the prompt + the synthetic model.
# --------------------------------------------------------------------------- #
def test_produce_clip_t2v_synthetic_mp4_and_manifest():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping t2v mp4 check)")
        return
    prompt = "a neon lighthouse sweeping a foggy harbor at night"
    env = _studio_env(12)
    req = CapabilityRequest(
        capability=Capability.T2V, target_resolution=R_TINY, vram_budget_gb=0.5)
    out_root = tempfile.mkdtemp(prefix="studio-t2v-syn-")
    try:
        res = produce_clip(req, env=env, out_root=out_root, prompt=prompt)
        assert res.is_ok(), f"synthetic t2v produce_clip must yield Ok(Artifact); got {res}"
        art = res.unwrap()
        assert isinstance(art, Artifact)
        assert os.path.isfile(art.path) and os.path.getsize(art.path) > 0, "mp4 non-empty"
        nb = _ffprobe_nb_frames(art.path)
        assert nb > 0 and nb == art.frames, f"ffprobe frames {nb} != artifact.frames {art.frames}"

        sc = _read_sidecar(art.path)
        assert sc["capability"] == "t2v", f"manifest must record capability=t2v; got {sc['capability']}"
        assert sc["task"] == "t2v", f"manifest task must be t2v; got {sc['task']}"
        assert sc["prompt"] == prompt, f"manifest must record the prompt; got {sc['prompt']!r}"
        assert sc["model_id"] == "synthetic-t2v", f"expected synthetic-t2v; got {sc['model_id']}"

        # determinism: a re-run with the identical spec RESUMES the same clip (the
        # prompt is in the hash, but the frames are prompt-agnostic + deterministic).
        res2 = produce_clip(req, env=env, out_root=out_root, prompt=prompt)
        assert res2.is_ok() and res2.unwrap().resumed is True, (
            "an identical t2v produce must resume the existing clip (deterministic)")
        assert res2.unwrap().content_hash == art.content_hash
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [6] content_hash of a T2V manifest differs from an equivalent I2V manifest, and
#     the prompt is in the hash while the synthetic FRAMES stay prompt-agnostic.
# --------------------------------------------------------------------------- #
def test_t2v_manifest_hash_differs_from_i2v_and_prompt_in_hash():
    env = _studio_env(12)
    seeds = SeedBundle(global_seed=7, stage_seeds=(("base", 7),))
    sampler = SamplerConfig(sampler="euler", scheduler="normal", steps=1, cfg=1.0)

    t2v_b = CapabilityRouter().resolve(CapabilityRequest(
        capability=Capability.T2V, target_resolution=R_TINY, vram_budget_gb=0.5)).unwrap()
    i2v_b = CapabilityRouter().resolve(CapabilityRequest(
        capability=Capability.I2V, target_resolution=R_TINY, vram_budget_gb=0.5)).unwrap()

    def _mk(cap, binding, prompt=""):
        return make_render_manifest(
            render_id="rid", capability=cap, binding=binding, seeds=seeds,
            sampler=sampler, resolution_ladder=(R_TINY,), env=env, prompt=prompt)

    h_t2v = _mk(Capability.T2V, t2v_b).content_hash()
    h_i2v = _mk(Capability.I2V, i2v_b).content_hash()
    assert h_t2v != h_i2v, (
        "a t2v manifest must NOT share a content_hash with an equivalent i2v manifest "
        "(capability + task + model_id all differ)")

    # prompt is part of the reproducibility key: differ only the prompt -> differ hash.
    h_a = _mk(Capability.T2V, t2v_b, prompt="a red kite").content_hash()
    h_b = _mk(Capability.T2V, t2v_b, prompt="a blue kite").content_hash()
    assert h_a != h_b, "the prompt must be part of the t2v content_hash"

    # but the synthetic frames themselves are a pure function of seed+geometry
    # (prompt-agnostic + deterministic): same args => byte-identical frame.
    import numpy as np
    f1 = synthesize_frame(7, 320, 180, 3, 24, None)
    f2 = synthesize_frame(7, 320, 180, 3, 24, None)
    assert np.array_equal(f1, f2), "synthetic t2v frames must be byte-deterministic"


# --------------------------------------------------------------------------- #
# [7] Bus adapter: run_studio_i2v with a T2V spec -> JobResult(ok=True) carrying a
#     clip MediaRef (the whole studio_i2v bus job serves t2v, no new job kind).
# --------------------------------------------------------------------------- #
def test_run_studio_i2v_t2v_spec_ok_with_clip_ref():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping bus-adapter t2v check)")
        return
    # out_root MUST live under the storage jail (DEFAULT_ROOT) so media_store.ingest
    # can catalog the produced clip; a tmp dir outside the jail would be refused.
    out_root = tempfile.mkdtemp(prefix="studio-t2v-bus-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    spec = make_studio_i2v(
        capability="t2v", width=512, height=512, fps=8, vram_budget_gb=0.5, seed=0,
        out_root=out_root, prompt="a slow drone shot over a glowing city grid")
    assert spec.capability == "t2v", "make_studio_i2v must accept + carry capability=t2v"

    orig = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False  # no cancel pending
    # The 0.5 GB budget binds the synthetic prover, which is OPT-IN
    # (operator 2026-07-27): enable it for THIS check only, never process-wide.
    prev_syn = os.environ.get("STUDIO_ALLOW_SYNTHETIC")
    os.environ["STUDIO_ALLOW_SYNTHETIC"] = "1"
    try:
        result = run_studio_i2v(spec, job_id="t2v-bus-ok-1")
        assert result.ok is True, f"a t2v bus job must be ok=True; got {result}"
        assert result.error is None, f"ok job must carry no error; got {result.error}"
        assert len(result.outputs) == 1, f"a t2v bus job must carry one clip ref; got {result.outputs}"
        ref = result.outputs[0]
        assert getattr(ref, "kind", None) == "video", f"the ref must be a video MediaRef; got {ref}"
    finally:
        if prev_syn is None:
            os.environ.pop("STUDIO_ALLOW_SYNTHETIC", None)
        else:
            os.environ["STUDIO_ALLOW_SYNTHETIC"] = prev_syn
        media_bus.is_cancelling = orig
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [8] Real Wan t2v path @16GB on THIS GPU-less/bitsandbytes-less box degrades
#     GRACEFULLY to an Err (deps_missing here) — never a raise — and it routed to a
#     REAL Wan t2v model (proving the real path, not the synthetic fallback).
# --------------------------------------------------------------------------- #
def test_wan_t2v_graceful_err_on_this_box():
    env = _studio_env(24)
    req = CapabilityRequest(
        capability=Capability.T2V, target_resolution=R480, vram_budget_gb=16.0,
        preferred_framework=Framework.WAN)

    # (a) it routes to a REAL Wan t2v model (not synthetic).
    binding = CapabilityRouter().resolve(req).unwrap()
    assert binding.framework == Framework.WAN and binding.task == Task.T2V, (
        f"16GB T2V must bind a real Wan t2v model; got {binding.framework}/{binding.task}")
    assert binding.model_id in ("wan2.1-t2v-1.3b", "wan2.2-t2v-a14b")

    # (b) the runner itself degrades as DATA (not a raise) on this box.
    manifest = make_render_manifest(
        render_id="wan-t2v-1", capability=Capability.T2V, binding=binding,
        seeds=SeedBundle(global_seed=7, stage_seeds=(("base", 7),)),
        sampler=SamplerConfig(sampler="unipc", scheduler="normal", steps=30, cfg=5.0),
        resolution_ladder=(R480,), env=env, prompt="a lone astronaut on a red dune")
    res = run_wan_t2v(manifest, tempfile.gettempdir())
    assert res.is_err(), "Wan t2v on a GPU-less/bitsandbytes-less box must return Err, not Ok"
    assert isinstance(res.error, StageError), "Err payload must be a StageError value"
    assert res.error.code in (
        ErrorCode.DEPS_MISSING, ErrorCode.NO_GPU, ErrorCode.WEIGHTS_MISSING), (
        f"expected a graceful preflight code; got {res.error.code}")

    # (c) end-to-end through produce_clip: same graceful Err, never a raise.
    res2 = produce_clip(req, env=env, out_root=tempfile.gettempdir(),
                        prompt="a lone astronaut on a red dune")
    assert res2.is_err() and isinstance(res2.error, StageError), (
        "produce_clip t2v @16GB must propagate the runner's graceful Err")
    assert res2.error.code in (
        ErrorCode.DEPS_MISSING, ErrorCode.NO_GPU, ErrorCode.WEIGHTS_MISSING)


# --------------------------------------------------------------------------- #
# [9] item 3 — ftfy is a REQUIRED Wan dep, so a box without it reports DEPS_MISSING
#     at PREFLIGHT (before loading ~14GB), not a mid-load io_error (live 2026-07-07).
# --------------------------------------------------------------------------- #
def test_ftfy_is_a_required_wan_dep():
    import importlib.util

    from hugpy_video.intel.studio.runners.wan_i2v import _REQUIRED_DEPS, _missing_deps
    assert "ftfy" in _REQUIRED_DEPS, (
        f"ftfy must be a required Wan dep (prompt-clean path needs it); {_REQUIRED_DEPS}")
    # When ftfy is genuinely absent on this box, it surfaces in the preflight
    # dep-scan (so the DEPS_MISSING headline lists it) — the item-3 honesty fix.
    if importlib.util.find_spec("ftfy") is None:
        assert "ftfy" in _missing_deps(), (
            f"an absent ftfy must appear in _missing_deps(); got {_missing_deps()}")


CHECKS = [
    ("registry: (SYNTHETIC,T2V)+(WAN,T2V) wired, synthetic-t2v prover, join total",
     test_registry_t2v_wired),
    ("router: tiny-budget T2V binds synthetic-t2v (demo path)",
     test_router_t2v_tiny_budget_binds_synthetic),
    ("router: 16GB T2V binds a REAL Wan t2v model",
     test_router_t2v_16gb_binds_real_wan),
    ("router: last-resort holds for T2V (synthetic never shadows a real Wan t2v)",
     test_router_t2v_synthetic_never_shadows_real),
    ("produce_clip T2V via synthetic -> real mp4 + manifest records capability=t2v+prompt",
     test_produce_clip_t2v_synthetic_mp4_and_manifest),
    ("t2v content_hash != i2v; prompt in hash; synthetic frames prompt-agnostic",
     test_t2v_manifest_hash_differs_from_i2v_and_prompt_in_hash),
    ("bus adapter: run_studio_i2v(t2v spec) -> JobResult ok + clip ref",
     test_run_studio_i2v_t2v_spec_ok_with_clip_ref),
    ("wan t2v @16GB -> graceful Err (deps_missing) + routed to a real Wan t2v model",
     test_wan_t2v_graceful_err_on_this_box),
    ("item 3: ftfy is a required Wan dep -> DEPS_MISSING at preflight, not mid-load",
     test_ftfy_is_a_required_wan_dep),
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
