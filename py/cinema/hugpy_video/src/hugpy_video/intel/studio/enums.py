"""Closed vocabularies. Everything the system dispatches on is an enum, never a
free string. str-Enum so values serialize straight to JSON in manifests.

Three related taxonomies, kept distinct on purpose:

  * Capability  - the PUBLIC contract a shot asks for (CAP-*). "i2v.identity_locked".
  * Framework   - the model family / inference codebase that executes it.
  * Task        - the concrete operation a runner performs.

A ModelConfig declares which Capabilities it provides and which Tasks it can run.
CAPABILITY_TASKS (registry.py) says which Tasks satisfy a Capability. The router
joins the three. validate_registry() proves the join is total.
"""

from __future__ import annotations

from enum import Enum


class Capability(str, Enum):
    T2V = "t2v"
    I2V = "i2v"
    V2V = "v2v"
    KEYFRAME = "keyframe"        # first/last/mid-frame interpolation
    ID_LOCK = "id_lock"         # character-consistent, locked identity (§3)
    MOTION = "motion"           # pose/motion/camera control (§4)
    STREAM = "stream"           # low-latency autoregressive generation (§5)
    INPAINT = "inpaint"
    OUTPAINT = "outpaint"
    RETAKE = "retake"           # targeted-segment regeneration
    AUDIO = "audio"             # generated dialogue/music/sfx or native a/v
    LIPSYNC = "lipsync"
    UPRES = "upres"             # upscale (§6)
    INTERP = "interp"           # frame interpolation (§6)
    RESTORE = "restore"         # face/detail restoration (§6)
    ASSEMBLE = "assemble"       # multi-shot stitch (§8) - orchestration, not a model


class Framework(str, Enum):
    WAN = "wan"
    LTX = "ltx"
    HUNYUAN = "hunyuan"
    COGVIDEOX = "cogvideox"
    MOCHI = "mochi"
    OPEN_SORA = "open_sora"
    SKYREELS = "skyreels"
    ANIMATEDIFF = "animatediff"
    FRAMEPACK = "framepack"
    RIFE = "rife"
    CODEFORMER = "codeformer"
    FFMPEG = "ffmpeg"           # no-weights LAST-RESORT enhancer: real frame
                                # interpolation (minterpolate) + spatial upscale
                                # (scale=lanczos) via the system ffmpeg binary, so
                                # INTERP/UPRES are real capabilities on a GPU-less
                                # box today. Ranks below the premium models (RIFE,
                                # LTX upscaler) via synthetic=True (§6, slice b).
    SYNTHETIC = "synthetic"     # no-model procedural runner: proves the spine
                                # end-to-end (frames -> ffmpeg) with no GPU/weights


class Task(str, Enum):
    T2V = "t2v"
    I2V = "i2v"
    VACE_CONTROL = "vace_control"      # unified reference/depth/pose/inpaint/extend
    AUDIO_VIDEO = "audio_video"        # single-pass synced a/v (LTX-2)
    AVATAR_LIPSYNC = "avatar_lipsync"  # audio-driven human animation
    MOTION_MODULE = "motion_module"    # AnimateDiff-style temporal module
    STREAM_I2V = "stream_i2v"          # constant-memory autoregressive i2v
    UPSCALE = "upscale"
    INTERPOLATE = "interpolate"
    RESTORE_FACE = "restore_face"


class Precision(str, Enum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"


# Selection quality order (higher = better output, all else equal).
PRECISION_QUALITY: dict[Precision, int] = {
    Precision.FP32: 5,
    Precision.BF16: 4,
    Precision.FP16: 3,
    Precision.FP8: 2,
    Precision.INT8: 1,
}


class LicenseClass(str, Enum):
    APACHE_2_0 = "apache_2_0"
    MIT = "mit"
    OPENRAIL_M = "openrail_m"
    TENCENT_COMMUNITY = "tencent_community"   # commercial OK but MAU/territory caveats
    LTX_COMMERCIAL = "ltx_commercial"         # requires separate paid agreement
    PROPRIETARY = "proprietary"


# Whether commercial routing may auto-select a model under this license without an
# explicit opt-in. LTX/PROPRIETARY are False: they only route commercially if the
# CapabilityRequest lists them in allowed_licenses ("I hold the agreement").
LICENSE_AUTO_COMMERCIAL: dict[LicenseClass, bool] = {
    LicenseClass.APACHE_2_0: True,
    LicenseClass.MIT: True,
    LicenseClass.OPENRAIL_M: True,       # use-case restrictions apply; see notes
    LicenseClass.TENCENT_COMMUNITY: True,  # MAU ceiling; see notes
    LicenseClass.LTX_COMMERCIAL: False,
    LicenseClass.PROPRIETARY: False,
}

# Preference weight for ranking survivors (more permissive = higher).
LICENSE_PREFERENCE: dict[LicenseClass, int] = {
    LicenseClass.APACHE_2_0: 3,
    LicenseClass.MIT: 3,
    LicenseClass.OPENRAIL_M: 2,
    LicenseClass.TENCENT_COMMUNITY: 2,
    LicenseClass.LTX_COMMERCIAL: 1,
    LicenseClass.PROPRIETARY: 0,
}


class DeterminismClass(str, Enum):
    EXACT = "exact"                 # same manifest -> bit-similar output
    SEEDED_APPROX = "seeded_approx"  # seeded, but fp8/flash-attn reductions vary
    DRIFTING = "drifting"           # autoregressive; long-horizon drift expected


class PathClass(str, Enum):
    OFFLINE = "offline"     # full-sequence diffusion; identity work lives here
    STREAMING = "streaming"  # causal AR; preview / lower identity stakes (STR-6)


class JobState(str, Enum):
    QUEUED = "queued"
    RESOLVING = "resolving"
    GENERATING = "generating"
    VALIDATING = "validating"
    LADDERING = "laddering"
    MASTERING = "mastering"
    RETAKING = "retaking"
    DONE = "done"
    FAILED = "failed"


class RiskFlag(str, Enum):
    EXTREME_ANGLE = "extreme_angle"     # ID-6 failure zones
    LOW_LIGHT = "low_light"
    SMALL_IN_FRAME = "small_in_frame"
    TWO_HANDER = "two_hander"           # ID-7: two subjects interacting -> blurring
    MULTI_CHARACTER = "multi_character"
    MISSING_CONSENT = "missing_consent"  # LEGAL-1


class AdapterKind(str, Enum):
    IDENTITY_LORA = "identity_lora"          # ID-2 (a): structural identity in weights
    IP_ADAPTER = "ip_adapter"                # ID-2 (b): zero-train reference embedding
    CANONICAL_MULTIVIEW = "canonical_multiview"  # ID-2 (c) / ID-3: real head-turn
    VACE = "vace"
    CAMERA_CTRL = "camera_ctrl"
    MOTION_LORA = "motion_lora"
    CONTROLNET = "controlnet"


class ControlKind(str, Enum):
    POSE = "pose"           # DWPose / OpenPose skeleton
    DEPTH = "depth"
    CANNY = "canny"
    NORMAL = "normal"
    LINEART = "lineart"
    OPTICAL_FLOW = "optical_flow"
    SEG_MASK = "seg_mask"
