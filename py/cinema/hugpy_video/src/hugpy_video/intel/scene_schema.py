"""Scene-generation schema — one query -> N consecutive frames (+ optional mp4).

A GenerateSceneSpec extends the generate_image contract with a frame count, an
assembly flag, and the seed / prompt-schedule knobs that give the frames temporal
coherence. Coherence mode is DECIDED: seed + prompt-schedule v1 (NO img2img — the
managers plane has no img2img pair/model wired; confirmed). The runner
(runners/scene.py) walks n_frames sequentially, derives a per-frame prompt + seed,
generates each frame through the SAME inference plane as generate_image, and
(when assemble) muxes the frames into a browser-playable mp4.

Per-part validity mirrors gen_schema.make_generate_image EXACTLY (exactly one of
text|media, and media.kind matching the part kind) — enforced in the factory with
LOCAL raises (construction-time, never across a boundary).

VIDEO parts are legal at construction time but MUST be resolved to image parts
(via video_intel.chains.resolve_video_parts_scene) before the runner sees them —
the runner treats a surviving raw video part as an error.

FRAME_CAP is a LOUD ceiling: the factory REFUSES n_frames > FRAME_CAP (message
prefixed "frame_cap_exceeded:") rather than silently clamping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from hugpy_video.intel.gen_schema import GenPromptPart, PART_KINDS

# LOUD ceiling on frames per scene (the runner also re-checks, belt-and-suspenders).
FRAME_CAP = 24


@dataclass(frozen=True)
class GenerateSceneSpec:
    """Generate a consecutive N-frame scene from an ordered multimodal prompt.

        model_id   the inference-plane model_key (e.g. "sd-turbo")
        width/height/steps/guidance   per-frame generation params
        n_frames   how many frames to generate (1..FRAME_CAP)
        fps        assembly frame rate (frames -> mp4)
        assemble   when True, mux the frames into a browser-playable mp4
        seed       optional base seed (per-frame seed = seed + i) for coherence
        motion     optional prompt-schedule suffix; "{i}"/"{n}" are substituted
        negative   optional negative prompt
        strength   optional img2img denoising strength (0..1) when a start-frame
                   image part is present; the runner applies 0.45 when None
        chain      img2img coherence mode: True (default) = TRUE sequential
                   chaining (each frame conditions on the previous frame's
                   output); False = every frame conditions on the start frame
                   (no drift). Ignored on the v1 text-to-image path.
    """
    parts: Tuple[GenPromptPart, ...]
    model_id: str
    width: int
    height: int
    steps: int
    guidance: float
    n_frames: int
    fps: int
    assemble: bool
    seed: Optional[int] = None
    motion: Optional[str] = None
    negative: Optional[str] = None
    # --- img2img additive fields (v1 payloads omit them -> defaults apply) ---
    strength: Optional[float] = None
    chain: bool = True
    # optional human project NAME; the runner archives the generation into an
    # assets/<slug|job_id>/ bundle (absent -> bundle dir is the job_id uuid).
    project: Optional[str] = None


def make_generate_scene(
    parts: Tuple[GenPromptPart, ...],
    model_id: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    n_frames: int,
    fps: int,
    assemble: bool,
    seed: Optional[int] = None,
    motion: Optional[str] = None,
    negative: Optional[str] = None,
    strength: Optional[float] = None,
    chain: bool = True,
    project: Optional[str] = None,
) -> GenerateSceneSpec:
    """Validate + build a GenerateSceneSpec. Raises are LOCAL to construction.

    Parts validation is IDENTICAL to gen_schema.make_generate_image (exactly one
    of text/media per part, kind membership, media.kind match). Additional scene
    invariants: truthy model_id; positive width/height/steps; 1 <= n_frames <=
    FRAME_CAP (over-cap raises "frame_cap_exceeded: ..."); fps >= 1; assemble is
    a bool.

    Also the reconstruction path used by the bus deserializer — parts are rebuilt
    into GenPromptPart before this is called (see media_bus).
    """
    parts = tuple(parts)
    if not parts:
        raise ValueError("generate_scene requires at least one prompt part")
    for i, p in enumerate(parts):
        if p.kind not in PART_KINDS:
            raise ValueError(f"part[{i}].kind must be one of {sorted(PART_KINDS)}; got {p.kind!r}")
        has_text = p.text is not None
        has_media = p.media is not None
        if has_text == has_media:
            raise ValueError(
                f"part[{i}] must set exactly one of text/media "
                f"(kind={p.kind!r}, text={'set' if has_text else 'unset'}, "
                f"media={'set' if has_media else 'unset'})"
            )
        if p.kind == "text" and not has_text:
            raise ValueError(f"part[{i}] kind='text' must carry text")
        if p.kind in ("image", "video"):
            if not has_media:
                raise ValueError(f"part[{i}] kind={p.kind!r} must carry a media MediaRef")
            if p.media.kind != p.kind:
                raise ValueError(
                    f"part[{i}] kind={p.kind!r} but media.kind={p.media.kind!r} (must match)"
                )
    if not model_id:
        raise ValueError(f"model_id must be a non-empty model key; got {model_id!r}")
    if not (isinstance(width, int) and width > 0):
        raise ValueError(f"width must be a positive int; got {width!r}")
    if not (isinstance(height, int) and height > 0):
        raise ValueError(f"height must be a positive int; got {height!r}")
    if not (isinstance(steps, int) and steps > 0):
        raise ValueError(f"steps must be a positive int; got {steps!r}")
    if not (isinstance(n_frames, int) and n_frames >= 1):
        raise ValueError(f"n_frames must be an int >= 1; got {n_frames!r}")
    if n_frames > FRAME_CAP:
        raise ValueError(f"frame_cap_exceeded: n_frames={n_frames} exceeds cap {FRAME_CAP}")
    if not (isinstance(fps, int) and fps >= 1):
        raise ValueError(f"fps must be an int >= 1; got {fps!r}")
    if not isinstance(assemble, bool):
        raise ValueError(f"assemble must be a bool; got {assemble!r}")
    # img2img additive knobs (optional; v1 payloads omit them):
    if strength is not None and not (isinstance(strength, (int, float))
                                     and 0.0 <= float(strength) <= 1.0):
        raise ValueError(f"strength must be a float in [0, 1] or None; got {strength!r}")
    if not isinstance(chain, bool):
        raise ValueError(f"chain must be a bool; got {chain!r}")
    return GenerateSceneSpec(
        parts=parts,
        model_id=model_id,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        n_frames=n_frames,
        fps=fps,
        assemble=assemble,
        seed=seed,
        motion=motion,
        negative=negative,
        strength=(float(strength) if strength is not None else None),
        chain=chain,
        project=(project or None),
    )
