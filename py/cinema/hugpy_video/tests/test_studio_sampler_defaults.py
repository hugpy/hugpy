"""MODEL-AWARE SAMPLER DEFAULTS + GPU PLACEMENT decision (this slice).

Locks the fix for the "synthetic-era steps=1 / cfg=1.0 reaching the REAL runner"
gray-mush bug: when a render does not PIN sampling, ``produce_clip`` fills the
denoise settings from the BOUND model's FAMILY (Wan -> 32 steps / cfg 5.0 / flow
shift), while synthetic / ffmpeg keep the steps=1 placeholder (their runners never
sample). Explicit steps/cfg ALWAYS win. The manifest records the TRUE values used
(content_hash keys on the sampler), and the placement decision is a PURE function
proven with no GPU.

Same script style as the other studio suites (plain python, ``__main__`` guard,
numbered ``[n] PASS`` / ``[n] FAIL`` lines, every check independent, nonzero exit
iff any FAILED). pytest is NOT installed in this venv.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_sampler_defaults.py
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile

logging.disable(logging.INFO)
os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_platform.constants import DEFAULT_ROOT
from hugpy_video.intel import media_bus
from hugpy_video.intel.runners.studio_i2v import run_studio_i2v
from hugpy_video.intel.studio.artifacts import Artifact
from hugpy_video.intel.studio.enums import Capability, Framework, Precision, Task
from hugpy_video.intel.studio.env import StudioEnv
from hugpy_video.intel.studio.errors import Err, Ok
from hugpy_video.intel.studio.job import make_studio_i2v
from hugpy_video.intel.studio import produce as produce_mod
from hugpy_video.intel.studio.produce import produce_clip, resolve_sampler
from hugpy_video.intel.studio.runners.wan_i2v import (
    _engage_memory_savers,
    _max_vram_gb,
    _place_pipe,
    _prime_cuda_allocator,
    _should_place_whole_on_gpu,
)
from hugpy_video.intel.studio.schemas import CapabilityRequest, Resolution

_FFMPEG = shutil.which("ffmpeg") is not None


def _env() -> StudioEnv:
    return StudioEnv(
        output_root="/out", weights_root="/weights", manifest_root="/manifests",
        master_colorspace="rec709", master_fps=16, max_vram_gb=24.0,
        loudness_target_lufs=-14.0, allow_unpinned=True)


def _capture_manifest(dispatch_key, request, **produce_kwargs):
    """Run produce_clip with dispatch_key's runner swapped for a stub that CAPTURES the
    manifest it is handed (and returns Ok(Artifact)), so we can assert the exact sampler
    produce_clip built for a binding WITHOUT a GPU/weights. Restores _DISPATCH after."""
    captured = {}

    def _stub(manifest, out_root, start_image=None, should_cancel=None):
        captured["manifest"] = manifest
        return Ok(Artifact(path="/x", content_hash=manifest.content_hash(),
                           frames=1, width=1, height=1, duration_s=1.0, resumed=False))

    orig = produce_mod._DISPATCH.get(dispatch_key)
    produce_mod._DISPATCH[dispatch_key] = _stub
    try:
        produce_clip(request, env=_env(), out_root="/tmp/none", **produce_kwargs)
    finally:
        if orig is not None:
            produce_mod._DISPATCH[dispatch_key] = orig
        else:
            produce_mod._DISPATCH.pop(dispatch_key, None)
    return captured.get("manifest")


# --------------------------------------------------------------------------- #
# [1] resolve_sampler: Wan family real defaults + resolution-derived shift.
# --------------------------------------------------------------------------- #
def test_resolve_sampler_wan_defaults():
    s480 = resolve_sampler(Framework.WAN, Resolution(832, 480, 16))
    assert (s480.steps, s480.cfg, s480.shift) == (32, 5.0, 3.0), s480
    s720 = resolve_sampler(Framework.WAN, Resolution(1280, 720, 24))
    assert s720.shift == 5.0, s720
    s720p = resolve_sampler(Framework.WAN, Resolution(720, 1280, 24))  # portrait 720p
    assert s720p.shift == 5.0, s720p


# --------------------------------------------------------------------------- #
# [2] resolve_sampler: a family with NO real defaults keeps the steps=1 placeholder
#     (synthetic / ffmpeg never sample) — back-compat with every prior clip.
# --------------------------------------------------------------------------- #
def test_resolve_sampler_placeholder_families():
    for fw in (Framework.SYNTHETIC, Framework.FFMPEG):
        s = resolve_sampler(fw, Resolution(512, 512, 24))
        assert (s.steps, s.cfg, s.shift, s.sampler, s.scheduler) == (
            1, 1.0, None, "euler", "normal"), (fw, s)


# --------------------------------------------------------------------------- #
# [3] Explicit steps/cfg ALWAYS win over the model default (both families).
# --------------------------------------------------------------------------- #
def test_resolve_sampler_explicit_override():
    w = resolve_sampler(Framework.WAN, Resolution(832, 480, 16), steps=10, cfg=7.0)
    assert (w.steps, w.cfg, w.shift) == (10, 7.0, 3.0), w  # shift still from resolution
    y = resolve_sampler(Framework.SYNTHETIC, Resolution(512, 512, 24), steps=5)
    assert (y.steps, y.cfg) == (5, 1.0), y


# --------------------------------------------------------------------------- #
# [4] produce_clip fills a WAN binding's manifest with the real defaults (no GPU
#     needed — the bound runner is captured). t2v @9GB binds wan2.1-t2v-1.3b.
# --------------------------------------------------------------------------- #
def test_produce_clip_wan_t2v_real_defaults():
    req = CapabilityRequest(capability=Capability.T2V,
                            target_resolution=Resolution(832, 480, 16),
                            vram_budget_gb=9.0)
    m = _capture_manifest((Framework.WAN, Task.T2V), req)
    assert m is not None and m.model_id == "wan2.1-t2v-1.3b", m
    assert (m.sampler.steps, m.sampler.cfg, m.sampler.shift) == (32, 5.0, 3.0), m.sampler


# --------------------------------------------------------------------------- #
# [5] produce_clip fills a WAN i2v binding's manifest with real defaults @16GB
#     (wan2.1-i2v-14b-720p) — 480p target -> shift 3.0.
# --------------------------------------------------------------------------- #
def test_produce_clip_wan_i2v_real_defaults():
    req = CapabilityRequest(capability=Capability.I2V,
                            target_resolution=Resolution(832, 480, 16),
                            vram_budget_gb=16.0)
    m = _capture_manifest((Framework.WAN, Task.I2V), req)
    assert m is not None and m.model_id == "wan2.1-i2v-14b-720p", m
    assert m.precision == Precision.INT8, m.precision
    assert (m.sampler.steps, m.sampler.cfg, m.sampler.shift) == (32, 5.0, 3.0), m.sampler


# --------------------------------------------------------------------------- #
# [6] Explicit steps/cfg threaded through produce_clip win over the WAN default,
#     and the captured manifest's content_hash keys on the sampler (truthfulness):
#     a different steps -> a different hash.
# --------------------------------------------------------------------------- #
def test_produce_clip_explicit_override_and_hash():
    req = CapabilityRequest(capability=Capability.T2V,
                            target_resolution=Resolution(832, 480, 16),
                            vram_budget_gb=9.0)
    m_def = _capture_manifest((Framework.WAN, Task.T2V), req)
    m_ovr = _capture_manifest((Framework.WAN, Task.T2V), req, steps=12, cfg=8.0)
    assert (m_ovr.sampler.steps, m_ovr.sampler.cfg) == (12, 8.0), m_ovr.sampler
    # manifest truthfulness: the sampler is in the content_hash, so a different denoise
    # setting re-addresses the clip (never the same hash for different pixels).
    assert m_def.content_hash() != m_ovr.content_hash(), "steps must change the hash"


# --------------------------------------------------------------------------- #
# [7] SYNTHETIC render is UNCHANGED: a real produce (tiny budget) writes a manifest
#     recording steps=1 / cfg=1.0 (back-compat — the gray-mush fix never touches it).
# --------------------------------------------------------------------------- #
def test_synthetic_render_keeps_steps_1():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping synthetic write check)")
        return
    out = tempfile.mkdtemp(prefix="sampler-synth-")
    try:
        req = CapabilityRequest(capability=Capability.I2V,
                                target_resolution=Resolution(320, 180, 12),
                                vram_budget_gb=0.5)
        res = produce_clip(req, env=_env(), out_root=out)
        assert res.is_ok(), res
        with open(os.path.join(os.path.dirname(res.unwrap().path), "manifest.json")) as fh:
            m = json.load(fh)
        assert m["framework"] == "synthetic", m["framework"]
        assert (m["sampler"]["steps"], m["sampler"]["cfg"]) == (1, 1.0), m["sampler"]
    finally:
        shutil.rmtree(out, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [8] End-to-end spec threading: a synthetic bus render with an explicit steps
#     override records that steps in the written manifest (route -> spec ->
#     CapabilityRequest -> produce_clip -> runner -> manifest).
# --------------------------------------------------------------------------- #
def test_spec_steps_override_recorded_via_bus():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping bus override check)")
        return
    # isolate the bus DB so this never writes into the real dev catalog
    import sqlite3
    tmp_db = tempfile.mkstemp(prefix="sampler-bus-", suffix=".db")[1]
    orig_db, orig_init = media_bus.DB_PATH, media_bus._initialized
    media_bus.DB_PATH = tmp_db
    media_bus._initialized = False
    # SYNTHETIC IS OPT-IN since c165f12 ("the synthetic gate must refuse NOISE"):
    # ``run_studio_i2v`` refuses a spec that binds Framework.SYNTHETIC unless
    # STUDIO_ALLOW_SYNTHETIC=1, so a real render can never silently degrade into a
    # blob that looks finished. This check WANTS the prover — vram_budget_gb=0.5
    # binds it deliberately, because what is under test is the steps override
    # travelling route -> spec -> CapabilityRequest -> produce_clip -> manifest, not
    # the pixels. So opt in explicitly for the duration and restore the prior env: a
    # HARNESS opt-in (same idiom as test_studio_offload.py), never a product default.
    #
    # ORDER MATTERS, and it is the reason mkdtemp is above the try and the setenv is
    # the FIRST statement inside it. The opt-in used to be set before mkdtemp, which
    # left a window: DEFAULT_ROOT is the shared store, mkdtemp against it can and does
    # raise (EMFILE under the virtiofs store storm, ENOSPC, a stale export), and an
    # exception there escaped BEFORE the try that owns the restore — leaking
    # STUDIO_ALLOW_SYNTHETIC=1 into the rest of the process. In a shared-process sweep
    # that silently re-opens the synthetic gate for every later test, so a runner that
    # should have REFUSED to emit a noise blob would instead emit one and pass. A
    # harness opt-in that can outlive its own test is worse than no isolation at all,
    # because it fails open and quietly. Nothing between mkdtemp and the try may set
    # process-global state.
    _SYN = "STUDIO_ALLOW_SYNTHETIC"
    orig_syn = os.environ.get(_SYN)
    out = tempfile.mkdtemp(prefix="sampler-bus-out-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))  # inside jail
    try:
        os.environ[_SYN] = "1"
        with sqlite3.connect(tmp_db) as _c:
            _c.execute(
                "CREATE TABLE IF NOT EXISTS media_jobs (job_id TEXT PRIMARY KEY, "
                "name TEXT, status TEXT, spec_json TEXT, result_json TEXT, "
                "claim_token TEXT, created REAL, updated REAL, progress_json TEXT)")
        spec = make_studio_i2v(capability="t2v", width=320, height=180, fps=12,
                               vram_budget_gb=0.5, seed=0, out_root=out, steps=7)
        result = run_studio_i2v(spec, "job-sampler-1")
        assert result.ok, result
        clip = result.outputs[0].uri
        with open(os.path.join(os.path.dirname(clip), "manifest.json")) as fh:
            m = json.load(fh)
        assert m["sampler"]["steps"] == 7, m["sampler"]
    finally:
        media_bus.DB_PATH, media_bus._initialized = orig_db, orig_init
        if orig_syn is None:
            os.environ.pop(_SYN, None)
        else:
            os.environ[_SYN] = orig_syn
        shutil.rmtree(out, ignore_errors=True)
        try:
            os.remove(tmp_db)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# [9] PLACEMENT decision (pure, no GPU): a bnb-quantized precision NEVER goes whole
#     to GPU; an unquantized model does iff it + margin fits the budget.
# --------------------------------------------------------------------------- #
def test_placement_decision_pure():
    # quantized precisions always offload (never .to() a bnb pipeline), even if they'd fit
    assert _should_place_whole_on_gpu(Precision.INT8, 14.0, 24.0) is False
    assert _should_place_whole_on_gpu(Precision.FP8, 4.0, 24.0) is False
    # unquantized + fits (model + 6GB margin <= budget) -> whole to GPU
    assert _should_place_whole_on_gpu(Precision.BF16, 8.2, 24.0) is False  # 8.2+16=24.2 (ae OOM lesson)
    assert _should_place_whole_on_gpu(Precision.FP16, 7.9, 24.0) is True    # 7.9+16=23.9
    # unquantized but doesn't fit -> offload
    assert _should_place_whole_on_gpu(Precision.BF16, 40.0, 24.0) is False
    assert _should_place_whole_on_gpu(Precision.BF16, 8.1, 24.0) is False   # 8.1+16=24.1
    # unknown budget / size -> conservative offload
    assert _should_place_whole_on_gpu(Precision.BF16, 8.2, None) is False
    assert _should_place_whole_on_gpu(Precision.BF16, None, 24.0) is False
    # margin is tunable
    assert _should_place_whole_on_gpu(Precision.BF16, 20.0, 24.0, margin=2.0) is True


# --------------------------------------------------------------------------- #
# [10] _max_vram_gb reads STUDIO_MAX_VRAM_GB from the manifest's env_snapshot.
# --------------------------------------------------------------------------- #
def test_max_vram_from_env_snapshot():
    """The declared ceiling is read from env_snapshot AND BOTH BIND — but a DEAF DEVICE
    PROBE FAILS CLOSED (2026-07-27, round-2 review).

    This check used to assert ``_max_vram_gb(<declares 24.0>) == 24.0`` on this GPU-less
    VM, i.e. that a declaration alone sets the budget. That is now wrong, and it was the
    dangerous direction. ``resolve_studio_env`` DEFAULTS max_vram_gb to 24.0 and no caller
    overrides it, so a "24.0" almost always means *nobody said*, not *an operator said*.
    Under the retired flat margin that fabricated 24.0 was inert (8.2 + 16.0 = 24.2 >
    24.0 -> offload). Against DERIVED need it is permissive: wan2.1-t2v-1.3b @fp16 needs
    17.897 GiB, which "fits" 24.0 -> a bare ``pipe.to("cuda")``. On computron's 8 GiB 4060
    with a probe that went deaf, that is an OOM caused by a failed probe rather than by
    any real budget. Unknown card => offload.

    So the invariant is now two-sided, and both sides are checked below."""
    import hugpy_platform.hardware as _hw

    class _M:  # minimal stand-in carrying only env_snapshot
        env_snapshot = (("STUDIO_MAX_VRAM_GB", "24.0"),)

    class _M2:
        env_snapshot = ()

    prev = os.environ.pop("STUDIO_MAX_VRAM_GB", None)
    orig_probe = _hw.total_vram_bytes
    try:
        # (a) DEAF PROBE (this VM has no GPU, so this is the real, unmocked path):
        #     a declaration alone must NOT authorize whole-GPU placement.
        _hw.total_vram_bytes = lambda: 0
        assert _max_vram_gb(_M()) is None, \
            "a declared ceiling must not bind when the device probe is deaf"
        assert _max_vram_gb(_M2()) is None

        # (b) LIVE PROBE: the declaration IS read, and BOTH bind via min() — the
        #     behaviour the original check was really about, now tested where it applies.
        _hw.total_vram_bytes = lambda: int(40.0 * (1024 ** 3))   # card bigger than decl
        assert _max_vram_gb(_M()) == 24.0, "the declared ceiling must bind under a big card"
        _hw.total_vram_bytes = lambda: int(8.0 * (1024 ** 3))    # card smaller than decl
        assert _max_vram_gb(_M()) == 8.0, "the CARD must win when it is below the declaration"
        assert _max_vram_gb(_M2()) == 8.0, "with no declaration the card alone binds"
    finally:
        _hw.total_vram_bytes = orig_probe
        if prev is not None:
            os.environ["STUDIO_MAX_VRAM_GB"] = prev


# --------------------------------------------------------------------------- #
# [11] VRAM levers on BOTH branches (item 4, INVARIANT REVERSED 2026-07-27):
#      _place_pipe engages attention slicing + VAE tiling/slicing on the whole-GPU
#      branch TOO, not only on offload; each lever stays AttributeError-guarded
#      (duck-typed pipe/VAE, no GPU, no torch).
#
#      WHY THIS TEST CHANGED SIDES. It used to assert "whole-GPU branch must engage
#      no levers". That was wrong, and load-bearing wrong: ``vae.enable_tiling()`` is
#      what makes VAE decode a BOUNDED constant instead of a resolution-squared
#      spike. Untiled, ``AutoencoderKLWan._decode`` runs full-frame fp32 convolutions
#      and accumulates the whole clip via ``torch.cat`` — that untiled decode, not the
#      denoise, is the most plausible cause of the 2026-07-07 ae OOM the old flat
#      16 GB margin was built to avoid.
#
#      And the new placement calculation (``_placement_need_gib``, wan_i2v.py) PRICES
#      decode as bounded — ``_WS_INTERCEPT_GIB + 0.60`` = 1.36 GiB flat, independent of
#      resolution. That price is only honest if tiling is actually on. Flipping
#      placement onto measured need while leaving the whole-GPU branch unguarded would
#      have reproduced the exact incident the change was preventing, so the two moves
#      are one change and this check is its lock.
#
#      DUCK SHAPE = MEASURED SHAPE (corrected 2026-07-27). The default used to be
#      ``slicing=False``, described in-test as "shaped like the real
#      AutoencoderKLWan". It is not: verified directly against the INSTALLED build,
#
#        >>> import diffusers; diffusers.__version__
#        '0.39.0'
#        >>> from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
#        >>> hasattr(AutoencoderKLWan, "enable_slicing"), hasattr(AutoencoderKLWan, "enable_tiling")
#        (True, True)
#
#      so a REAL Wan pipe engages all THREE levers and returns
#      ["attention_slicing", "vae_tiling", "vae_slicing"]. A duck defaulting to
#      slicing=False modelled a diffusers that is not installed here, which made the
#      headline assertions encode a false model of the pipeline — the failure mode is
#      that the test would keep passing if the real third lever silently stopped
#      engaging. The default now matches the installed build; the no-slicing duck is
#      retained BELOW, relabelled as what it actually is (a guard probe for a build
#      that lacks the lever), so the AttributeError-guard coverage is not lost.
#
#      ⚠ "attention_slicing" IN THE RETURNED LIST IS NOT EVIDENCE THAT SLICING HAPPENED.
#      ``DiffusionPipeline.enable_attention_slicing`` exists, so the call succeeds and
#      the name is appended — but the dispatch is
#      ``DiffusionPipeline.set_attention_slice``, which filters to submodules with
#      ``hasattr(m, "set_attention_slice")``, and EVERY component of a Wan pipe lacks
#      it (verified: WanTransformer3DModel, AutoencoderKLWan, UMT5EncoderModel all
#      False). It iterates an empty list: a NO-OP on Wan. The lever list records what
#      was CALLED, not what saved memory. Only ``vae_tiling`` is doing real work here,
#      which is precisely why the bounded-decode price above rests on tiling alone.
# --------------------------------------------------------------------------- #
class _DuckVAE:
    # slicing defaults TRUE because AutoencoderKLWan really does expose
    # enable_slicing in diffusers 0.39.0 — see the note above.
    def __init__(self, tiling=True, slicing=True):
        self.calls = []
        if tiling:
            self.enable_tiling = lambda: self.calls.append("tiling")
        if slicing:
            self.enable_slicing = lambda: self.calls.append("slicing")


class _DuckPipe:
    def __init__(self, vae=None, attn=True):
        self.calls = []
        self.vae = vae
        if attn:
            self.enable_attention_slicing = lambda *a, **k: self.calls.append("attn")

    def to(self, device):
        self.calls.append(("to", device))

    def enable_model_cpu_offload(self):
        self.calls.append("offload")


# The three levers a REAL Wan pipe on diffusers 0.39.0 engages, in call order. Named
# once so the "this is what the box does" cases below cannot drift apart from each
# other, and so a diffusers upgrade that changes the surface is ONE edit here rather
# than a hunt through four assertions.
_REAL_WAN_LEVERS = ["attention_slicing", "vae_tiling", "vae_slicing"]


def test_place_pipe_engages_levers_on_both_branches():
    # whole-GPU branch: .to("cuda") and NO offload — but the savers DO engage, because
    # the bounded-decode price _placement_need_gib charges is only honest with tiling on.
    whole_vae = _DuckVAE()                      # real AutoencoderKLWan shape
    p_whole = _DuckPipe(vae=whole_vae)
    assert _place_pipe(p_whole, True) == _REAL_WAN_LEVERS, \
        "whole-GPU branch must engage the savers too (tiling bounds VAE decode)"
    assert ("to", "cuda") in p_whole.calls and "offload" not in p_whole.calls, p_whole.calls
    # tiling REALLY ran, not just named — and it is the lever that actually bounds
    # decode, so this is the assertion that backs the placement price.
    assert whole_vae.calls == ["tiling", "slicing"], whole_vae.calls

    # ...and the guards hold on the whole-GPU branch as well: a pipe with no VAE still
    # places, it just engages fewer levers. A memory hint must never fail a render.
    p_whole_novae = _DuckPipe(vae=None, attn=True)
    assert _place_pipe(p_whole_novae, True) == ["attention_slicing"], p_whole_novae.calls
    assert ("to", "cuda") in p_whole_novae.calls, p_whole_novae.calls

    # offload branch, VAE shaped like the real AutoencoderKLWan: BOTH VAE levers
    # exist on diffusers 0.39.0, so all three engage and the offload path is taken.
    vae = _DuckVAE()
    p = _DuckPipe(vae=vae, attn=True)
    engaged = _place_pipe(p, False)
    assert "offload" in p.calls, p.calls
    assert engaged == _REAL_WAN_LEVERS, engaged
    assert "attn" in p.calls and vae.calls == ["tiling", "slicing"], (p.calls, vae.calls)

    # GUARD PROBE, not a model of this box: a VAE lacking enable_slicing (an older or
    # future diffusers). The lever must AttributeError, be swallowed, and drop out of
    # the list — never fail the render. Kept after the correction above precisely so
    # the guard stays covered without pretending to be the installed build.
    p_noslice = _DuckPipe(vae=_DuckVAE(tiling=True, slicing=False), attn=True)
    assert _place_pipe(p_noslice, False) == ["attention_slicing", "vae_tiling"], \
        "a VAE without enable_slicing must skip that lever, not fail the render"

    # offload branch, a pipeline WITHOUT enable_attention_slicing -> guarded, skipped.
    p3 = _DuckPipe(vae=_DuckVAE(tiling=True, slicing=False), attn=False)
    assert _place_pipe(p3, False) == ["vae_tiling"], "attention slicing must be skipped"

    # offload branch, NO vae attribute -> only attention slicing.
    p4 = _DuckPipe(vae=None, attn=True)
    assert _place_pipe(p4, False) == ["attention_slicing"], "no vae -> attn only"

    # the helper is idempotent-safe to call directly on a duck too.
    assert _engage_memory_savers(_DuckPipe(vae=_DuckVAE(), attn=True)) == _REAL_WAN_LEVERS


# --------------------------------------------------------------------------- #
# [12] CUDA allocator defragmentation (item 7, REVISED 2026-07-08): OPT-IN ONLY
#      (HUGPY_CUDA_EXPANDABLE=1). The setdefault-by-default variant crash-looped ae
#      (expandable_segments leaked through re-exec via os.environ and this
#      driver/torch combo dies natively under it), so: no opt-in -> never set AND
#      the exact leaked value is DETOXED; opt-in -> setdefault (operator wins).
# --------------------------------------------------------------------------- #
def test_prime_cuda_allocator_setdefault():
    _KEY = "PYTORCH_CUDA_ALLOC_CONF"
    _OPT = "HUGPY_CUDA_EXPANDABLE"
    prev, prev_opt = os.environ.pop(_KEY, None), os.environ.pop(_OPT, None)
    try:
        # no opt-in, unset -> stays unset; returns a bool (torch-imported probe).
        ret = _prime_cuda_allocator()
        assert _KEY not in os.environ, os.environ.get(_KEY)
        assert isinstance(ret, bool), ret
        # no opt-in, leaked prime value -> DETOXED (popped).
        os.environ[_KEY] = "expandable_segments:True"
        _prime_cuda_allocator()
        assert _KEY not in os.environ, "leaked value not detoxed"
        # no opt-in, operator's own different value -> untouched.
        os.environ[_KEY] = "garbage_collection_threshold:0.9"
        _prime_cuda_allocator()
        assert os.environ.get(_KEY) == "garbage_collection_threshold:0.9", os.environ.get(_KEY)
        os.environ.pop(_KEY, None)
        # opt-in, unset -> filled with the defrag default.
        os.environ[_OPT] = "1"
        _prime_cuda_allocator()
        assert os.environ.get(_KEY) == "expandable_segments:True", os.environ.get(_KEY)
        # opt-in, pre-set -> operator's value ALWAYS wins (setdefault never overrides).
        os.environ[_KEY] = "garbage_collection_threshold:0.9"
        _prime_cuda_allocator()
        assert os.environ.get(_KEY) == "garbage_collection_threshold:0.9", os.environ.get(_KEY)
    finally:
        for k, p in ((_KEY, prev), (_OPT, prev_opt)):
            if p is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = p


CHECKS = [
    ("resolve_sampler: Wan real defaults + resolution shift (3.0@480p, 5.0@720p)",
     test_resolve_sampler_wan_defaults),
    ("resolve_sampler: placeholder families keep steps=1/cfg=1.0/shift=None",
     test_resolve_sampler_placeholder_families),
    ("resolve_sampler: explicit steps/cfg always win", test_resolve_sampler_explicit_override),
    ("produce_clip: WAN t2v binding gets real 32/5.0/3.0", test_produce_clip_wan_t2v_real_defaults),
    ("produce_clip: WAN i2v @16GB binds 14B INT8, real 32/5.0/3.0",
     test_produce_clip_wan_i2v_real_defaults),
    ("produce_clip: explicit override wins + sampler in content_hash",
     test_produce_clip_explicit_override_and_hash),
    ("synthetic render UNCHANGED: written manifest records steps=1/cfg=1.0",
     test_synthetic_render_keeps_steps_1),
    ("bus spec steps override recorded in the written manifest",
     test_spec_steps_override_recorded_via_bus),
    ("placement decision (pure, no GPU): quantized never .to(); unquantized fits",
     test_placement_decision_pure),
    ("_max_vram_gb reads STUDIO_MAX_VRAM_GB from env_snapshot", test_max_vram_from_env_snapshot),
    ("VRAM levers (item 4): _place_pipe engages attn/VAE slicing+tiling on BOTH branches, guarded",
     test_place_pipe_engages_levers_on_both_branches),
    ("cuda allocator (item 7): _prime_cuda_allocator setdefault fills unset, respects preset",
     test_prime_cuda_allocator_setdefault),
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
