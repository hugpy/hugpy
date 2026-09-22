"""Schemas for the text-to-image task.

Generation parameters are all optional: anything the caller doesn't set is
omitted from the diffusers pipeline call, so the pipeline's own per-model
defaults apply (sdxl-turbo wants 1-4 steps and guidance 0.0; SD 1.5 wants
~50 steps and guidance 7.5 — neither needs the caller to know that).
"""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class GeneratedImage(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    b64: Optional[str] = None       # base64 PNG bytes, omitted when return_b64=False
    width: int
    height: int
    seed: Optional[int] = None


class ImageGenRequest(BaseModel):
    """One unit of text-to-image work. Built per call."""
    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    model_key: str = Field(min_length=1)
    pool: Optional[str] = None   # dedicated worker pool (routing); None = general
    prompt: str = Field(min_length=1)

    negative_prompt: Optional[str] = None
    width: Optional[int] = Field(default=None, ge=64, le=4096, multiple_of=8)
    height: Optional[int] = Field(default=None, ge=64, le=4096, multiple_of=8)
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=50.0)
    # --- ComfyUI sampler plumbing (additive) ---
    # Read only by the comfy runner (managers/comfy/comfy_runner.py); the
    # diffusers runners ignore them. None -> the runner's historical
    # euler/normal defaults, so requests without them behave exactly as before.
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    seed: Optional[int] = None
    # --- img2img (image-to-image) additive fields ---
    # text-to-image callers never set these (extra="ignore" on the frozen model
    # keeps them absent for the text2img path); the img2img runner + builder read
    # them. image_path rides remote._PATH_KEYS so the worker inliner rebuilds it.
    image_path: Optional[str] = None       # init image; rides remote._PATH_KEYS inliner
    strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    num_images: int = Field(default=1, ge=1, le=4)
    # --- ID-LOCK (identity-locked STILLs) additive fields ---
    # The image sibling of the studio VIDEO arm's id_lock (Wan-VACE reference-to-
    # video): give reference image(s) of a subject and generate NEW stills that
    # hold the identity, via ComfyUI's IP-Adapter (video_intel.studio.enums:
    # AttentionMethod.IP_ADAPTER — "ID-2 (b): zero-train reference embedding"). Read
    # ONLY by the comfy runner + its IPAdapter graph; every diffusers runner ignores
    # them. All absent -> today's behaviour EXACTLY (plain text2img / img2img).
    #
    # reference_images: jailed abs paths of the subject reference still(s), in order
    # (at most _MAX_REFERENCE_IMAGES; the builder jail-resolves + count-checks them).
    # A comfy worker (127.0.0.1) can't see central's UPLOADS_HOME, so the offload
    # transport carries the BYTES in reference_images_b64 instead — remote._worker_
    # payload reads the paths, base64s them into reference_images_b64, and DROPS the
    # unreachable paths (mirrors VisionAnalysisRequest.image_b64, which likewise
    # carries bytes on the schema for offload). The comfy runner prefers b64 when
    # present, else uploads straight from the paths (the in-process / worker-local
    # case). NB: this is a request FIELD, not remote._PATH_KEYS single-file inlining
    # — the latter handles one path via the worker's _materialize_file, and a
    # multi-image rematerializer there is out of this slice's agent.py scope.
    reference_images: Optional[List[str]] = Field(default=None, max_length=4)
    reference_images_b64: Optional[List[str]] = Field(default=None, max_length=4)
    # id_strength -> the IPAdapter apply node's `weight` (how hard the reference
    # embedding pulls the sample toward the subject). 0 = ignore the reference,
    # 1 = maximal identity hold; ~0.6 is the balanced default. None on the wire is
    # coerced to the default by the builder so the graph always has a concrete weight.
    id_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # b64 in the result is what lets HTTP callers (the discord bot) fetch the
    # bytes without a second round-trip; set False for in-process callers that
    # only want the saved paths.
    return_b64: bool = True


class ImageGenResult(BaseModel):
    request_id: str
    model_key: str
    ok: bool = True
    images: List[GeneratedImage] = Field(default_factory=list)
    # Human-readable summary so chat-stream wrapping (stream_runner's one-shot
    # path reads result.text) degrades to something sensible.
    text: str = ""
    error: Optional[str] = None
