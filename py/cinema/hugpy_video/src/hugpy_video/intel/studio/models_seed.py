"""The zoo, as data (§2). Registering this module populates all three registries.

VRAM figures are PLANNING ESTIMATES in GB - they move with resolution, frame
count, precision, and offload strategy. They exist so the router can reason about
fit, not as a promise. Pin real weight hashes before production (INV-1/INV-2);
until then models are `unpinned=True` and boot requires STUDIO_ALLOW_UNPINNED=1.

`verify_uri=True` marks a repo path NOT directly confirmed in the source thread -
treat those as "check before you `huggingface-cli download`."
"""

from __future__ import annotations

from hugpy_video.intel.studio.enums import (
    AdapterKind,
    Capability,
    DeterminismClass,
    Framework,
    LicenseClass,
    PathClass,
    Precision,
    Task,
)
from hugpy_video.intel.studio.registry import (
    RunnerSpec,
    register_model,
    register_runner,
    set_capability_tasks,
)
from hugpy_video.intel.studio.schemas import ModelConfig, Resolution, VramEnvelope

# --- reusable resolution constants ----------------------------------------
R_256 = Resolution(256, 256, 24)
R_512 = Resolution(512, 512, 8)
R_COG = Resolution(720, 480, 8)
R_MOCHI = Resolution(848, 480, 30)
R_480P = Resolution(832, 480, 16)
R_768 = Resolution(768, 768, 24)
R_720P = Resolution(1280, 720, 24)
R_720P_V = Resolution(720, 1280, 24)
R_1080P = Resolution(1920, 1080, 24)
R_4K = Resolution(3840, 2160, 50)


def E(*pairs: tuple[Precision, float]) -> VramEnvelope:
    return VramEnvelope(tuple(pairs))


HF = "https://huggingface.co/"


# --------------------------------------------------------------------------
# REAL SAMPLER DEFAULTS, as data (data-over-code). Keyed by model FAMILY: the
# denoise settings a render uses when the spec did not PIN steps/cfg explicitly.
# Consumed by ``studio.produce.resolve_sampler`` at manifest-build time and RECORDED
# in the manifest (content_hash keys on the sampler), so the runner denoises with
# exactly these values.
#
# WHY THIS EXISTS: the studio's synthetic-era placeholder was steps=1 / cfg=1.0 (a
# no-op denoise — one step, no guidance). That is correct for the SYNTHETIC prover
# (its frames are a pure function of seed + geometry — the sampler never touches a
# pixel) but reaching a REAL diffusion runner it produces gray mush (a single
# unguided step). A real family therefore declares real defaults here; a family
# ABSENT from this table falls back to the placeholder (steps=1 / cfg=1.0), which
# keeps synthetic + the ffmpeg/rife last-resort enhancers (whose runners ignore the
# sampler entirely) byte-identical to their historical content-addressed output.
#
# ``shift`` (flow-match / UniPC scheduler shift) is RESOLUTION-dependent, so it is
# NOT stored here — ``resolve_sampler`` derives it from the target resolution with a
# simple threshold (the Wan reference: 3.0 @ 480p, 5.0 @ 720p+). Each entry is a flat
# dict of the SamplerConfig scalar fields the family denoises with.
FAMILY_SAMPLER_DEFAULTS: dict[Framework, dict] = {
    # Wan 2.1/2.2 (t2v, i2v AND VACE v2v — all denoise): the diffusers Wan reference
    # runs ~30-50 UniPC flow-match steps with guidance ~5.0. 32 steps / cfg 5.0 is a
    # sane "get an init out and evaluated" default (operator directive) that lands well
    # inside a single-render budget on a 3090-class box.
    Framework.WAN: {"sampler": "unipc", "scheduler": "flow_match", "steps": 32, "cfg": 5.0},
}

# The placeholder used for any family NOT in FAMILY_SAMPLER_DEFAULTS (synthetic prover,
# ffmpeg/rife/ltx enhancers). IDENTICAL to the studio's historical _default_sampler so
# those content-addressed clips never re-address. steps=1 / cfg=1.0 = a no-op denoise —
# honest for a runner that does not sample.
PLACEHOLDER_SAMPLER_DEFAULTS: dict = {
    "sampler": "euler", "scheduler": "normal", "steps": 1, "cfg": 1.0,
}

# --------------------------------------------------------------------------
# Runners: (framework, task) -> how to execute it. Entrypoints wired explicitly.
#
# k1 GATE NOTE: several rows below name a runner module that has not been built
# yet (a video-engine bet, not a bug) — hunyuan/cog/mochi/opensora/skyreels/
# animatediff/framepack/codeformer, and LTX's t2v/i2v/av (only ltx_upscale is
# wired). Declaring the RunnerSpec here is intentional and stays — it is the
# zoo's honest statement of intent, and `registry.runner_available` /
# `runner_gate_reason` gate them out of actual routing/dispatch until their
# runner module lands (see `video_intel/studio/registry.py`'s "RUNNER GATE"
# docstring). Do NOT delete a row just because its runner isn't wired yet —
# dropping the module into `runners/` re-enables it with ZERO edits here.
# --------------------------------------------------------------------------
_RUNNERS = (
    # Task 3b: WAN T2V is now WIRED to the real runner — run_wan_t2v is a thin DRY
    # delegation to run_wan_i2v's WanPipeline (t2v) branch with start_image forced
    # None, so it inherits the identical preflight / bnb / cancel / atomic path.
    RunnerSpec(Framework.WAN, Task.T2V, "hugpy_video.intel.studio.runners.wan_t2v:run_wan_t2v", Precision.INT8),
    # P0-6: WAN i2v is WIRED to the real runner (import-safe, graceful-degrading,
    # bitsandbytes int8/nf4 on the box). VACE stays a dormant placeholder until its
    # bet lands (validate_registry only checks a runner is registered, not
    # importable). run_wan_i2v itself also serves t2v when start_image is None.
    RunnerSpec(Framework.WAN, Task.I2V, "hugpy_video.intel.studio.runners.wan_i2v:run_wan_i2v", Precision.INT8),
    # B-3: WAN VACE v2v is now WIRED to the real runner (import-safe, graceful-
    # degrading, bitsandbytes int8/nf4 on the box). run_wan_vace restyles/enhances an
    # existing clip (manifest.source_video) via diffusers' WanVACEPipeline; on this
    # GPU-less / bitsandbytes-less dev box it returns Err-as-data (SOURCE_MISSING /
    # DEPS_MISSING / NO_GPU / WEIGHTS_MISSING), never a raise.
    RunnerSpec(Framework.WAN, Task.VACE_CONTROL, "hugpy_video.intel.studio.runners.wan_vace:run_wan_vace", Precision.INT8),
    RunnerSpec(Framework.LTX, Task.T2V, "hugpy_video.intel.studio.runners.ltx:t2v", Precision.FP8),
    RunnerSpec(Framework.LTX, Task.I2V, "hugpy_video.intel.studio.runners.ltx:i2v", Precision.FP8),
    RunnerSpec(Framework.LTX, Task.AUDIO_VIDEO, "hugpy_video.intel.studio.runners.ltx:av", Precision.FP8),
    # slice b: LTX UPSCALE re-pointed to the REAL graceful premium runner (import-
    # safe, diffusers lazy). On this box the HF-license-gated weights (401) are not
    # on disk, so it returns Err(WEIGHTS_MISSING) — the ffmpeg lanczos last-resort
    # serves UPRES until the weights are staged. (Was runners.ltx:upscale, unwired.)
    RunnerSpec(Framework.LTX, Task.UPSCALE, "hugpy_video.intel.studio.runners.ltx_upscale:run_ltx_upscale", Precision.INT8),
    RunnerSpec(Framework.HUNYUAN, Task.T2V, "hugpy_video.intel.studio.runners.hunyuan:t2v", Precision.INT8),
    RunnerSpec(Framework.HUNYUAN, Task.I2V, "hugpy_video.intel.studio.runners.hunyuan:i2v", Precision.INT8),
    RunnerSpec(Framework.HUNYUAN, Task.AVATAR_LIPSYNC, "hugpy_video.intel.studio.runners.hunyuan:avatar", Precision.FP8),
    RunnerSpec(Framework.COGVIDEOX, Task.T2V, "hugpy_video.intel.studio.runners.cog:t2v", Precision.INT8),
    RunnerSpec(Framework.COGVIDEOX, Task.I2V, "hugpy_video.intel.studio.runners.cog:i2v", Precision.INT8),
    RunnerSpec(Framework.MOCHI, Task.T2V, "hugpy_video.intel.studio.runners.mochi:t2v", Precision.FP8),
    RunnerSpec(Framework.OPEN_SORA, Task.T2V, "hugpy_video.intel.studio.runners.opensora:t2v", Precision.FP8),
    RunnerSpec(Framework.OPEN_SORA, Task.I2V, "hugpy_video.intel.studio.runners.opensora:i2v", Precision.FP8),
    RunnerSpec(Framework.SKYREELS, Task.I2V, "hugpy_video.intel.studio.runners.skyreels:i2v", Precision.INT8),
    RunnerSpec(Framework.ANIMATEDIFF, Task.MOTION_MODULE, "hugpy_video.intel.studio.runners.animatediff:motion", Precision.INT8),
    RunnerSpec(Framework.FRAMEPACK, Task.STREAM_I2V, "hugpy_video.intel.studio.runners.framepack:stream", Precision.FP16, supports_streaming=True),
    RunnerSpec(Framework.LTX, Task.UPSCALE, "hugpy_video.intel.studio.runners.ltx:upscale2", Precision.INT8) if False else
    # slice b: RIFE INTERPOLATE re-pointed to the REAL graceful premium runner
    # (import-safe, Practical-RIFE lazy). On this box the Practical-RIFE arch is not
    # vendored, so it returns Err(DEPS_MISSING) — the ffmpeg minterpolate last-resort
    # serves INTERP until the box errand vendors it. (Was runners.rife:interp, unwired.)
    RunnerSpec(Framework.RIFE, Task.INTERPOLATE, "hugpy_video.intel.studio.runners.rife_interpolate:run_rife_interpolate", Precision.FP16),
    # slice b: FFMPEG LAST-RESORT enhancers — REAL frame interpolation + spatial
    # upscale via the system ffmpeg binary (zero new deps, GPU-less). Both models
    # carry synthetic=True so the premium RIFE/LTX rows ALWAYS outrank them; they
    # bind only when no premium model fits the budget. Import-safe (stdlib + numpy/PIL
    # via the synthetic sidecar helpers, no GPU stack). min_precision=INT8 (the floor)
    # so any precision binds — the tiny 0.05GB envelope means only INT8 exists anyway.
    RunnerSpec(Framework.FFMPEG, Task.INTERPOLATE, "hugpy_video.intel.studio.runners.ffmpeg_enhance:run_ffmpeg_interpolate", Precision.INT8),
    RunnerSpec(Framework.FFMPEG, Task.UPSCALE, "hugpy_video.intel.studio.runners.ffmpeg_enhance:run_ffmpeg_upscale", Precision.INT8),
    RunnerSpec(Framework.CODEFORMER, Task.RESTORE_FACE, "hugpy_video.intel.studio.runners.codeformer:restore", Precision.FP16),
    # SYNTHETIC: no-model procedural runner (P0-B1). Proves the whole spine
    # (capability -> binding -> manifest -> frames -> ffmpeg -> mp4) end-to-end
    # with NO GPU/weights. min_precision=INT8 (the floor) so any precision binds.
    RunnerSpec(Framework.SYNTHETIC, Task.I2V, "hugpy_video.intel.studio.runners.synthetic:run_synthetic_i2v", Precision.INT8),
    # Task 3b: SYNTHETIC t2v prover — no GPU/weights, deterministic, TEXT-AGNOSTIC
    # frames (the prompt rides in the manifest for provenance but never alters a
    # pixel). Thin wrapper over run_synthetic_i2v with start_image forced None.
    RunnerSpec(Framework.SYNTHETIC, Task.T2V, "hugpy_video.intel.studio.runners.synthetic:run_synthetic_t2v", Precision.INT8),
)
for _r in _RUNNERS:
    register_runner(_r)


# --------------------------------------------------------------------------
# Capability -> satisfying tasks, in preference order (the join).
# --------------------------------------------------------------------------
set_capability_tasks({
    Capability.T2V: (Task.T2V, Task.AUDIO_VIDEO),
    Capability.I2V: (Task.I2V, Task.AUDIO_VIDEO, Task.STREAM_I2V),
    # IDENTITY LOCK: VACE reference-to-video is the flagship id_lock path (the only
    # runner that CONSUMES reference images), so Task.VACE_CONTROL is PREFERRED (first).
    # Task.I2V stays as a fallback so the i2v rows that claim ID_LOCK remain valid.
    #
    # ⚠ THE OLD COMMENT HERE WAS FALSE AND COST AN IDENTITY (corrected 2026-07-27).
    # It claimed "in practice VACE always wins ... so id_lock never silently routes to
    # a runner that would ignore the references". That holds only INSIDE vace-1.3b's
    # 480p envelope. Ask for id_lock at ANY larger geometry — including the studio
    # routes' own former 512x512 default — and NO VACE model fits, so the router falls
    # through to Task.I2V and binds wan2.1-i2v-14b-720p, whose runner never reads
    # reference_images. The result was a plausible clip of the WRONG PERSON, with no
    # error at all.
    #
    # The fallback is kept (an i2v row claiming ID_LOCK is still a legal binding) but it
    # can no longer drop an identity silently: run_wan_i2v now REFUSES a manifest
    # carrying reference_images it cannot consume. The guard lives at the point of harm
    # so it survives any future routing change — do not rely on a preference order to
    # keep this safe.
    Capability.ID_LOCK: (Task.VACE_CONTROL, Task.I2V),
    Capability.KEYFRAME: (Task.I2V,),
    Capability.MOTION: (Task.VACE_CONTROL, Task.MOTION_MODULE),
    Capability.V2V: (Task.VACE_CONTROL,),
    Capability.INPAINT: (Task.VACE_CONTROL,),
    Capability.OUTPAINT: (Task.VACE_CONTROL,),
    Capability.RETAKE: (Task.VACE_CONTROL,),
    Capability.STREAM: (Task.STREAM_I2V,),
    Capability.AUDIO: (Task.AUDIO_VIDEO,),
    Capability.LIPSYNC: (Task.AVATAR_LIPSYNC, Task.AUDIO_VIDEO),
    Capability.UPRES: (Task.UPSCALE,),
    Capability.INTERP: (Task.INTERPOLATE,),
    Capability.RESTORE: (Task.RESTORE_FACE,),
    Capability.ASSEMBLE: (),   # orchestration stage; PLANNED, no model
})


_ID_ADAPTERS = frozenset({
    AdapterKind.IDENTITY_LORA, AdapterKind.IP_ADAPTER,
    AdapterKind.CANONICAL_MULTIVIEW, AdapterKind.CAMERA_CTRL,
})

# --------------------------------------------------------------------------
# Base generative models
# --------------------------------------------------------------------------
_MODELS = (
    # ---- Wan family (Alibaba, Apache-2.0) --------------------------------
    ModelConfig(
        model_id="wan2.1-t2v-1.3b", family=Framework.WAN,
        tasks=(Task.T2V,), capabilities=(Capability.T2V,),
        vram=E((Precision.FP16, 8.2), (Precision.INT8, 5.0)),
        resolutions=(R_480P,), max_frames=81, max_duration_s=5.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="Wan-AI/Wan2.1-T2V-1.3B", source_url=HF + "Wan-AI/Wan2.1-T2V-1.3B",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        accepts_adapters=frozenset({AdapterKind.IDENTITY_LORA, AdapterKind.CAMERA_CTRL}),
        notes="Consumer entry point; ~8GB fp16, fits a 3090 comfortably. 480p base.",
    ),
    ModelConfig(
        model_id="wan2.1-i2v-14b-720p", family=Framework.WAN,
        tasks=(Task.I2V,),
        capabilities=(Capability.I2V, Capability.ID_LOCK, Capability.KEYFRAME),
        vram=E((Precision.BF16, 40.0), (Precision.FP8, 18.0), (Precision.INT8, 14.0)),
        resolutions=(R_720P, R_720P_V, R_480P), max_frames=81, max_duration_s=5.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="Wan-AI/Wan2.1-I2V-14B-720P", source_url=HF + "Wan-AI/Wan2.1-I2V-14B-720P",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        accepts_adapters=_ID_ADAPTERS,
        notes="Identity-lock workhorse (ID-4: lock a still, then animate). FP8/INT8 to fit 3090.",
    ),
    # ── wan2.2-t2v-a14b / wan2.2-i2v-a14b: REMOVED (operator, 2026-08-13) ────
    # The model battery measured the nf4-quantized MoE (both experts resident)
    # at a 22.2 GiB peak vs ~22.0 GiB usable on this fleet's 24 GB 3090 — seven
    # mitigation attempts deep (weights root, stall window, GPU fully cleared,
    # LLM blocked, explicit budget, expandable segments, ComfyUI stopped), it
    # cannot complete a render here. Operator ruling: rather than carry rows the
    # router must refuse every time, drop them from the registry and every
    # picker. The 118 GiB weight pairs remain on
    # /mnt/llm_storage/video_intel/studio/weights/Wan-AI/ — restore the rows
    # from models_seed.py.bak-battery-20260813 (with the battery's HONEST
    # envelopes: FP8 23.0 / INT8 27.0, NOT the original 20/16 fiction) if a
    # bigger card or a single-expert/offload loading path lands.
    ModelConfig(
        model_id="wan2.1-vace-1.3b", family=Framework.WAN,
        tasks=(Task.VACE_CONTROL,),
        capabilities=(Capability.MOTION, Capability.V2V, Capability.INPAINT,
                      Capability.OUTPAINT, Capability.RETAKE, Capability.ID_LOCK),
        vram=E((Precision.FP16, 10.0), (Precision.INT8, 6.0)),
        resolutions=(R_480P,), max_frames=81, max_duration_s=5.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="Wan-AI/Wan2.1-VACE-1.3B", source_url=HF + "Wan-AI/Wan2.1-VACE-1.3B",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        accepts_adapters=frozenset({AdapterKind.VACE, AdapterKind.CONTROLNET, AdapterKind.CAMERA_CTRL}),
        notes="Unified control: reference/depth/pose/inpaint/extend + RETAKE (MOT-5). Fits 3090.",
    ),
    ModelConfig(
        model_id="wan2.1-vace-14b", family=Framework.WAN,
        tasks=(Task.VACE_CONTROL,),
        capabilities=(Capability.MOTION, Capability.V2V, Capability.INPAINT,
                      Capability.OUTPAINT, Capability.RETAKE, Capability.ID_LOCK),
        vram=E((Precision.BF16, 40.0), (Precision.FP8, 20.0), (Precision.INT8, 14.0)),
        resolutions=(R_720P, R_480P), max_frames=81, max_duration_s=5.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="Wan-AI/Wan2.1-VACE-14B", source_url=HF + "Wan-AI/Wan2.1-VACE-14B",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        accepts_adapters=frozenset({AdapterKind.VACE, AdapterKind.CONTROLNET, AdapterKind.CAMERA_CTRL}),
        notes="720p unified control. Multi-condition (depth + DWPose + mask) chaining supported.",
    ),

    # ---- LTX family (Lightricks, commercial agreement required) ----------
    ModelConfig(
        model_id="ltx-video-0.9.7-dev", family=Framework.LTX,
        tasks=(Task.T2V, Task.I2V),
        capabilities=(Capability.T2V, Capability.I2V, Capability.KEYFRAME),
        vram=E((Precision.FP16, 16.0), (Precision.FP8, 10.0), (Precision.INT8, 8.0)),
        resolutions=(R_1080P, R_720P, R_480P), max_frames=257, max_duration_s=10.0,
        license=LicenseClass.LTX_COMMERCIAL,
        weight_uri="Lightricks/LTX-Video-0.9.7-dev", source_url=HF + "Lightricks/LTX-Video",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        accepts_adapters=frozenset({AdapterKind.CONTROLNET, AdapterKind.CAMERA_CTRL}),
        notes="Fastest open path; great for storyboard/preview (PIPE-4). Diffusers: LTXConditionPipeline.",
    ),
    ModelConfig(
        model_id="ltx-2.3", family=Framework.LTX,
        tasks=(Task.AUDIO_VIDEO,),
        capabilities=(Capability.T2V, Capability.I2V, Capability.AUDIO, Capability.LIPSYNC),
        vram=E((Precision.BF16, 32.0), (Precision.FP8, 16.0)),
        resolutions=(R_4K, R_1080P, R_720P), max_frames=1000, max_duration_s=20.0,
        license=LicenseClass.LTX_COMMERCIAL, native_audio=True,
        weight_uri="Lightricks/LTX-2", source_url=HF + "Lightricks",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True, verify_uri=True,
        accepts_adapters=frozenset({AdapterKind.CONTROLNET, AdapterKind.CAMERA_CTRL}),
        notes="Single-pass synced 4K audio+video incl. lip-sync (AUD-1). 32GB official; "
              "community GGUF Q4~15GB/Q3~12GB unofficial. Verify exact 2.3 repo id.",
    ),

    # ---- HunyuanVideo family (Tencent, community license) ----------------
    ModelConfig(
        model_id="hunyuanvideo", family=Framework.HUNYUAN,
        tasks=(Task.T2V, Task.I2V), capabilities=(Capability.T2V, Capability.I2V),
        vram=E((Precision.BF16, 80.0), (Precision.FP8, 24.0), (Precision.INT8, 14.0)),
        resolutions=(R_720P, R_720P_V), max_frames=129, max_duration_s=5.0,
        license=LicenseClass.TENCENT_COMMUNITY,
        weight_uri="tencent/HunyuanVideo", source_url=HF + "tencent/HunyuanVideo",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        accepts_adapters=frozenset({AdapterKind.IDENTITY_LORA, AdapterKind.CONTROLNET}),
        notes="13B cinematic; strong motion/texture. Diffusers 4bit+tiling+offload ~6.5GB. "
              "License permits commercial use up to a MAU ceiling - verify for your scale. "
              "Text encoders: Kijai/HunyuanVideo_comfy.",
    ),
    ModelConfig(
        model_id="hunyuanvideo-avatar", family=Framework.HUNYUAN,
        tasks=(Task.AVATAR_LIPSYNC,), capabilities=(Capability.LIPSYNC,),
        vram=E((Precision.BF16, 40.0), (Precision.FP8, 24.0)),
        resolutions=(R_720P, R_720P_V), max_frames=129, max_duration_s=5.0,
        license=LicenseClass.TENCENT_COMMUNITY,
        weight_uri="tencent/HunyuanVideo-Avatar", source_url=HF + "tencent",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True, verify_uri=True,
        notes="Audio-driven human animation (AUD-3 lip-sync path). Verify exact repo id.",
    ),

    # ---- Other open bases -------------------------------------------------
    ModelConfig(
        model_id="cogvideox-5b", family=Framework.COGVIDEOX,
        tasks=(Task.T2V, Task.I2V), capabilities=(Capability.T2V, Capability.I2V),
        vram=E((Precision.FP16, 16.0), (Precision.INT8, 12.0)),
        resolutions=(R_COG,), max_frames=49, max_duration_s=6.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="THUDM/CogVideoX-5b", source_url=HF + "THUDM/CogVideoX-5b",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True, verify_uri=True,
        accepts_adapters=frozenset({AdapterKind.IDENTITY_LORA}),
        notes="720x480 @ 8fps, 6s. Verify org (THUDM vs zai-org) and 5B license terms "
              "(2B is Apache-2.0; 5B ships a custom CogVideoX License).",
    ),
    ModelConfig(
        model_id="mochi-1-preview", family=Framework.MOCHI,
        tasks=(Task.T2V,), capabilities=(Capability.T2V,),
        vram=E((Precision.BF16, 24.0), (Precision.FP8, 20.0)),
        resolutions=(R_MOCHI,), max_frames=163, max_duration_s=5.4,
        license=LicenseClass.APACHE_2_0,
        weight_uri="genmo/mochi-1-preview", source_url=HF + "genmo/mochi-1-preview",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True, verify_uri=True,
        notes="10B, photoreal, slow (8+ min/clip on a 4090). Open training code for fine-tune.",
    ),
    ModelConfig(
        model_id="open-sora-v2", family=Framework.OPEN_SORA,
        tasks=(Task.T2V, Task.I2V), capabilities=(Capability.T2V, Capability.I2V),
        vram=E((Precision.BF16, 40.0), (Precision.FP8, 24.0)),
        resolutions=(R_768, R_256), max_frames=129, max_duration_s=5.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="hpcai-tech/Open-Sora-v2", source_url=HF + "hpcai-tech/Open-Sora-v2",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        notes="11B, 256/768px, fully open training pipeline (train on proprietary data). "
              "License reported as both Apache-2.0 and MIT across sources - verify.",
    ),
    ModelConfig(
        model_id="skyreels-v1", family=Framework.SKYREELS,
        tasks=(Task.I2V,), capabilities=(Capability.I2V, Capability.ID_LOCK),
        vram=E((Precision.BF16, 40.0), (Precision.FP8, 24.0), (Precision.INT8, 14.0)),
        resolutions=(R_720P, R_720P_V), max_frames=129, max_duration_s=5.0,
        license=LicenseClass.TENCENT_COMMUNITY,
        weight_uri="Skywork/SkyReels-V1", source_url=HF + "Skywork/SkyReels-V1",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True, verify_uri=True,
        accepts_adapters=_ID_ADAPTERS,
        notes="HunyuanVideo fine-tune on film/TV; human-centric, convincing faces. "
              "Inherits Tencent community license. Verify exact repo (V1 vs V2).",
    ),

    # ---- Motion control (legacy SD path) ---------------------------------
    ModelConfig(
        model_id="animatediff-lightning", family=Framework.ANIMATEDIFF,
        tasks=(Task.MOTION_MODULE,), capabilities=(Capability.MOTION,),
        vram=E((Precision.FP16, 8.0), (Precision.INT8, 6.0)),
        resolutions=(R_512,), max_frames=64, max_duration_s=4.0,
        license=LicenseClass.OPENRAIL_M,
        weight_uri="ByteDance/AnimateDiff-Lightning", source_url=HF + "ByteDance/AnimateDiff-Lightning",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        accepts_adapters=frozenset({AdapterKind.MOTION_LORA, AdapterKind.CONTROLNET, AdapterKind.CAMERA_CTRL}),
        notes="SD1.5-era path: AnimateDiff (temporal) + ControlNet (structure). Stylized/anime, "
              "huge existing LoRA ecosystem. OpenRAIL-M use restrictions apply.",
    ),

    # ---- Streaming (autoregressive, constant-memory) ---------------------
    ModelConfig(
        model_id="framepack-i2v-hy", family=Framework.FRAMEPACK,
        tasks=(Task.STREAM_I2V,), capabilities=(Capability.STREAM,),
        vram=E((Precision.FP16, 6.0),),
        resolutions=(R_720P, R_480P), max_frames=3600, max_duration_s=120.0,
        license=LicenseClass.TENCENT_COMMUNITY, path_class=PathClass.STREAMING,
        weight_uri="lllyasviel/FramePackI2V_HY", source_url=HF + "lllyasviel/FramePackI2V_HY",
        default_determinism=DeterminismClass.DRIFTING, unpinned=True, verify_uri=True,
        notes="Constant-memory long i2v (~6GB!) on a HunyuanVideo base - ideal for the 3090 "
              "streaming path (STR-3). Code Apache (github lllyasviel/FramePack); weights HY-derived. "
              "Note STR-6: not for frame-perfect identity lock.",
    ),

    # ---- Quality ladder ---------------------------------------------------
    ModelConfig(
        model_id="ltxv-spatial-upscaler-0.9.7", family=Framework.LTX,
        tasks=(Task.UPSCALE,), capabilities=(Capability.UPRES,),
        vram=E((Precision.FP16, 12.0), (Precision.INT8, 8.0)),
        resolutions=(R_4K, R_1080P), max_frames=100000, max_duration_s=100000.0,
        license=LicenseClass.LTX_COMMERCIAL,
        weight_uri="Lightricks/ltxv-spatial-upscaler-0.9.7",
        source_url=HF + "Lightricks/ltxv-spatial-upscaler-0.9.7",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        notes="Latent spatial upscaler (QLD-2). Pairs with LTX base.",
    ),
    ModelConfig(
        model_id="rife-practical", family=Framework.RIFE,
        tasks=(Task.INTERPOLATE,), capabilities=(Capability.INTERP,),
        vram=E((Precision.FP16, 3.0),),
        resolutions=(R_4K, R_1080P, R_720P), max_frames=100000, max_duration_s=100000.0,
        license=LicenseClass.MIT,
        weight_uri="hzwer/Practical-RIFE", source_url="https://github.com/hzwer/Practical-RIFE",
        default_determinism=DeterminismClass.EXACT, unpinned=True, verify_uri=True,
        notes="Frame interpolation 24->48/60 and AR-seam smoothing (QLD-3). Deterministic. "
              "GitHub weights; verify HF mirror if you want one.",
    ),
    ModelConfig(
        model_id="codeformer", family=Framework.CODEFORMER,
        tasks=(Task.RESTORE_FACE,), capabilities=(Capability.RESTORE,),
        vram=E((Precision.FP16, 3.0),),
        resolutions=(R_4K, R_1080P, R_720P), max_frames=100000, max_duration_s=100000.0,
        license=LicenseClass.PROPRIETARY,   # S-Lab 1.0: NON-COMMERCIAL
        weight_uri="sczhou/CodeFormer", source_url="https://github.com/sczhou/CodeFormer",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True, verify_uri=True,
        notes="Face/detail restore (QLD-4). NTU S-Lab 1.0 license = NON-COMMERCIAL: will not "
              "auto-route for commercial_use. Use GFPGAN (TencentARC/GFPGAN, Apache) for commercial.",
    ),

    # ---- FFMPEG LAST-RESORT enhancers (slice b, §6) ----------------------
    # REAL frame interpolation + spatial upscale via the system ffmpeg binary — so
    # INTERP/UPRES are genuine studio capabilities on a GPU-less box TODAY, with ZERO
    # new deps. Each is synthetic=True (LAST-RESORT): the premium weight-backed model
    # of the same capability (rife-practical / ltxv-spatial-upscaler) ALWAYS outranks
    # it in router scoring, so it binds ONLY when no premium model fits the budget
    # (e.g. a sub-3GB interp budget, a sub-8GB upres budget) — never shadowing a real
    # binding. PINNED with a fixed pseudo weight_hash (no real weights exist — it uses
    # the ffmpeg binary), so production-clean without STUDIO_ALLOW_UNPINNED, and
    # DeterminismClass.EXACT (the ffmpeg transform is a pure function of the source
    # bytes + filter; the runner fixes -threads 1 for bit-stable output). Wide
    # resolution envelope so the last resort can cover any target the premium models
    # would. License class APACHE_2_0 = the house runner code (ffmpeg_enhance.py); the
    # underlying ffmpeg binary is itself LGPL/GPL (a system tool, invoked out-of-proc).
    ModelConfig(
        model_id="ffmpeg-minterpolate", family=Framework.FFMPEG,
        tasks=(Task.INTERPOLATE,), capabilities=(Capability.INTERP,),
        vram=E((Precision.INT8, 0.05)),
        resolutions=(R_4K, R_1080P, R_720P, R_480P), max_frames=100000, max_duration_s=100000.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="ffmpeg://minterpolate", source_url="ffmpeg://minterpolate",
        default_determinism=DeterminismClass.EXACT, path_class=PathClass.OFFLINE,
        weight_hash="ffmpeg-minterpolate-v1-0000000000000000000000000000000000000000000000000000000000000000",
        synthetic=True,   # LAST-RESORT: rife-practical ALWAYS outranks it when it fits.
        notes="No-weights frame interpolation (slice b, QLD-3). ffmpeg minterpolate "
              "(mi_mode=mci, motion-compensated) resamples the source clip to the "
              "manifest's target fps. Tiny 0.05GB envelope so it only binds when no "
              "premium interp model (rife-practical, 3GB+) fits the budget.",
    ),
    ModelConfig(
        model_id="ffmpeg-lanczos-upscale", family=Framework.FFMPEG,
        tasks=(Task.UPSCALE,), capabilities=(Capability.UPRES,),
        vram=E((Precision.INT8, 0.05)),
        resolutions=(R_4K, R_1080P, R_720P, R_480P), max_frames=100000, max_duration_s=100000.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="ffmpeg://lanczos-upscale", source_url="ffmpeg://lanczos-upscale",
        default_determinism=DeterminismClass.EXACT, path_class=PathClass.OFFLINE,
        weight_hash="ffmpeg-lanczos-upscale-v1-0000000000000000000000000000000000000000000000000000000000000000",
        synthetic=True,   # LAST-RESORT: ltxv-spatial-upscaler ALWAYS outranks it when it fits.
        notes="No-weights spatial upscale (slice b, QLD-2). ffmpeg scale=<W>:<H>:"
              "flags=lanczos to the manifest resolution. Tiny 0.05GB envelope so it "
              "only binds when no premium upres model (ltxv-spatial-upscaler, 8GB+) "
              "fits the budget.",
    ),

    # ---- SYNTHETIC (no-model) --------------------------------------------
    # P0-B1: a deterministic procedural i2v runner with NO weights/GPU. It exists
    # to prove the full studio spine end-to-end (capability -> binding -> manifest
    # -> frames -> ffmpeg -> mp4 + sidecars) before any real model bet. It is
    # PINNED with a fixed pseudo weight_hash so it is production-clean (needs no
    # STUDIO_ALLOW_UNPINNED) and DeterminismClass.EXACT (synthetic IS exact). It
    # only wins the router when the VRAM budget is too small for any real model
    # (min real i2v footprint is 8GB), so it never shadows a genuine binding.
    ModelConfig(
        model_id="synthetic-i2v", family=Framework.SYNTHETIC,
        tasks=(Task.I2V,), capabilities=(Capability.I2V,),
        vram=E((Precision.INT8, 0.1)),
        resolutions=(R_720P, R_512, R_256), max_frames=240, max_duration_s=10.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="synthetic://procedural-i2v",
        source_url="synthetic://procedural-i2v",
        default_determinism=DeterminismClass.EXACT, path_class=PathClass.OFFLINE,
        weight_hash="synthetic-i2v-v1-0000000000000000000000000000000000000000000000000000000000000000",
        synthetic=True,   # LAST-RESORT: any real model that fits ALWAYS outranks it
                          # in router scoring, so it can never shadow a real binding
                          # (not even when its tiny 0.1GB envelope also fits).
        notes="No-model procedural runner (P0-B1). Deterministic frames from the "
              "manifest seed + top resolution, assembled to H.264 via ffmpeg. Tiny "
              "0.1GB envelope so it only binds when no real model fits the budget.",
    ),
    # Task 3b: the SYNTHETIC T2V twin — the no-model text-to-video prover. Same
    # last-resort discipline as synthetic-i2v (synthetic=True => any real Wan t2v
    # model ALWAYS outranks it), same pinned/EXACT posture. Deliberately capped at
    # 512x512 (R_512/R_256, NO 720p): a t2v demo is tiny by definition, and the cap
    # means synthetic can never even be a CANDIDATE against a real model at the
    # larger formats real Wan t2v owns (it also keeps the "0.5GB @ 720p is
    # unroutable" router invariant intact). The PROMPT is recorded in the manifest
    # (content_hash + sidecar) for provenance but never touches a pixel — synthetic
    # frames are a pure function of seed + geometry, so t2v stays byte-deterministic.
    ModelConfig(
        model_id="synthetic-t2v", family=Framework.SYNTHETIC,
        tasks=(Task.T2V,), capabilities=(Capability.T2V,),
        vram=E((Precision.INT8, 0.1)),
        resolutions=(R_512, R_256), max_frames=240, max_duration_s=10.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="synthetic://procedural-t2v",
        source_url="synthetic://procedural-t2v",
        default_determinism=DeterminismClass.EXACT, path_class=PathClass.OFFLINE,
        weight_hash="synthetic-t2v-v1-0000000000000000000000000000000000000000000000000000000000000000",
        synthetic=True,   # LAST-RESORT: any real t2v model that fits ALWAYS outranks
                          # it in router scoring, so it never shadows a real binding.
        notes="No-model procedural T2V prover (Task 3b). Deterministic frames from "
              "the manifest seed + top resolution (prompt-agnostic — the prompt is "
              "recorded for provenance, never alters pixels), assembled to H.264 via "
              "ffmpeg. Tiny 0.1GB envelope, capped at 512x512 (no 720p) so it only "
              "binds a tiny-budget T2V demo and never shadows a real Wan t2v model.",
    ),
)
for _m in _MODELS:
    register_model(_m)
