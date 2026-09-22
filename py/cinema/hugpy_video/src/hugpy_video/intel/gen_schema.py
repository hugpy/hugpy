"""Image-generation schema — map §4.4.

A generation prompt is an ORDERED tuple of parts, each part being exactly one of
text / image / video (a multimodal prompt). Frozen spec + validating factory,
mirroring crop_schema.py / frame_schema.py. Per-part validity (exactly one of
text|media, and media.kind matching the part kind) is enforced in the factory
with LOCAL raises (construction-time, never across a boundary).

VIDEO parts are legal at construction time but MUST be resolved to image parts
(via video_intel.chains.resolve_video_parts) before the runner sees them — the
runner (runners/imagegen.py) treats a surviving raw video part as an error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

from hugpy_video.intel.media_schema import MediaRef

PartKind = Literal["text", "image", "video"]

PART_KINDS = frozenset({"text", "image", "video"})


@dataclass(frozen=True)
class GenPromptPart:
    """One part of a multimodal prompt: exactly one of `text` / `media` set.
    `kind` names which axis this part carries; for image/video parts `media`
    must be a MediaRef of the matching kind."""
    kind: PartKind
    text: Optional[str] = None
    media: Optional[MediaRef] = None


def text_part(text: str) -> GenPromptPart:
    return GenPromptPart(kind="text", text=text)


def image_part(media: MediaRef) -> GenPromptPart:
    return GenPromptPart(kind="image", media=media)


def video_part(media: MediaRef) -> GenPromptPart:
    return GenPromptPart(kind="video", media=media)


@dataclass(frozen=True)
class GenerateImageSpec:
    """Generate one image from an ordered multimodal prompt.

        model_id   the inference-plane model_key (e.g. "sd-turbo")
        width/height/steps/guidance   generation params
        seed/negative   optional determinism / negative prompt
    """
    parts: Tuple[GenPromptPart, ...]
    model_id: str
    width: int
    height: int
    steps: int
    guidance: float
    seed: Optional[int] = None
    negative: Optional[str] = None
    # img2img denoising strength (0..1) used when an image part is present AND
    # img2img is available on the fleet; the runner applies 0.45 when None.
    # v1 payloads omit it -> None -> unchanged text-to-image behavior.
    strength: Optional[float] = None
    # optional human project NAME; the runner archives the generation into an
    # assets/<slug|job_id>/ bundle (absent -> bundle dir is the job_id uuid).
    project: Optional[str] = None


def make_generate_image(
    parts: Tuple[GenPromptPart, ...],
    model_id: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: Optional[int] = None,
    negative: Optional[str] = None,
    strength: Optional[float] = None,
    project: Optional[str] = None,
) -> GenerateImageSpec:
    """Validate + build a GenerateImageSpec. Raises are LOCAL to construction.

    Also the reconstruction path used by the bus deserializer — parts are
    rebuilt into GenPromptPart before this is called (see media_bus).
    """
    parts = tuple(parts)
    if not parts:
        raise ValueError("generate_image requires at least one prompt part")
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
    if not (isinstance(width, int) and width > 0):
        raise ValueError(f"width must be a positive int; got {width!r}")
    if not (isinstance(height, int) and height > 0):
        raise ValueError(f"height must be a positive int; got {height!r}")
    if not (isinstance(steps, int) and steps > 0):
        raise ValueError(f"steps must be a positive int; got {steps!r}")
    if strength is not None and not (isinstance(strength, (int, float))
                                     and 0.0 <= float(strength) <= 1.0):
        raise ValueError(f"strength must be a float in [0, 1] or None; got {strength!r}")
    return GenerateImageSpec(
        parts=parts,
        model_id=model_id,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        seed=seed,
        negative=negative,
        strength=(float(strength) if strength is not None else None),
        project=(project or None),
    )
