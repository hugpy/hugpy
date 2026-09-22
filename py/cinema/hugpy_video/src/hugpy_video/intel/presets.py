"""Curated video-generation presets — "ideal default loads".

A tiny, frozen registry the UI reads to offer a dropdown of ready-made
scene-generation setups. Each preset names a catalog model_key, a coherence
``mode`` and a bundle of per-frame generation defaults, so selecting one is
enough to warm the right model on a GPU worker and pre-fill the generate_scene
form.

Deliberately mirrors the keyword-preset pattern in
``managers/keywords/keybert_model.py`` (a frozen ``*Preset`` dataclass + a plain
dict registry + ``available_presets()``/``get_preset()`` accessors) so the two
read the same way. Nothing here restarts a service or touches the serve/worker
control plane — it is a static table the HTTP surface dumps and looks up.

``mode`` maps to generate_scene coherence semantics (the runner + UI already
understand ``chain``):
    "edit-chain"     -> chain=True  (start-frame edit chain; each frame
                        conditions on the previous frame's output)
    "img2img"        -> chain=False (off-start img2img; every frame conditions
                        on the same start frame — no drift)
    "text-to-image"  -> v1 no-start path (pure prompt-scheduled generation)
The preset keeps ``mode`` as a plain string; translating it to ``chain`` is the
caller's job (chain is OFF by default; HUGPY_LEGACY_CHAIN=1 opts the fleet
into the legacy pixel chain — see movie_schema.legacy_chain_enabled).

Quant note (do NOT plumb): ``HUGPY_IMG2IMG_QUANTIZE`` is ENV-ONLY (read once in
``managers/imagegen/imagegen_runner.py``), NOT per-request. Its default "auto"
already 4-bit-quantizes any model that would exceed free VRAM (so
Qwen-Image-Edit auto-quants on a GPU worker). The advisory ``quant`` field below
is DISPLAY-ONLY — it changes nothing at runtime; there is intentionally no
per-request quant plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Schema — one frozen preset row
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VideoGenPreset:
    """Immutable "ideal default load" for scene generation.

    ``model_key``  a real catalog key (validated at apply time against
                   get_models_dict — the same source workers_assign uses).
    ``mode``       "edit-chain" | "img2img" | "text-to-image" (see module docs).
    The remaining fields are the per-frame generation defaults the UI pre-fills;
    ``defaults()`` bundles exactly the pinned contract's ``defaults`` sub-shape.
    ``quant``      advisory display string only — see the module quant note; it
                   is NOT plumbed into any generation request.
    """

    id: str
    name: str
    description: str
    mode: str
    model_key: str
    # per-frame generation defaults
    strength: float
    steps: int
    guidance: float
    width: int
    height: int
    n_frames: int
    fps: int
    negative: str = ""
    recommended: str = "gpu"
    quant: str = "auto"          # advisory/display only — see module quant note
    # The comfy runner now honors per-request sampler_name/scheduler
    # (managers/comfy/comfy_runner.py — plumbed 2026-07-06 via
    # ImageGenRequest.sampler_name/scheduler + the builders' 'sampler' alias).
    # The scene/movie spec path does NOT yet forward this field, so from the
    # scene UI it remains display-only until that plumbing lands.
    sampler: str = "euler"

    def defaults(self) -> Dict[str, Any]:
        """The pinned ``defaults`` sub-object (order/keys frozen for the UI)."""
        return {
            "strength": self.strength,
            "steps": self.steps,
            "guidance": self.guidance,
            "width": self.width,
            "height": self.height,
            "n_frames": self.n_frames,
            "fps": self.fps,
            "negative": self.negative,
        }

    def to_dict(self) -> Dict[str, Any]:
        """The pinned per-preset wire shape for GET /video/presets."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "mode": self.mode,
            "model_key": self.model_key,
            "defaults": self.defaults(),
            "recommended": self.recommended,
            # advisory, display-only (see module quant note) — additive to the
            # pinned shape; the UI is free to ignore it.
            "quant": self.quant,
            # the comfy runner honors sampler per-request (dataclass note); the
            # scene path doesn't forward it yet. Additive; the UI may ignore it.
            "sampler": self.sampler,
        }


# ---------------------------------------------------------------------------
# Registry — insertion order is the dropdown order
# ---------------------------------------------------------------------------
VIDEO_PRESETS: Dict[str, VideoGenPreset] = {}


def register_preset(preset: VideoGenPreset) -> None:
    if preset.id in VIDEO_PRESETS:
        raise KeyError(f"Video preset {preset.id!r} already registered")
    VIDEO_PRESETS[preset.id] = preset


def available_presets() -> List[VideoGenPreset]:
    """All presets in registration (dropdown) order."""
    return list(VIDEO_PRESETS.values())


def get_preset(preset_id: str) -> Optional[VideoGenPreset]:
    """The preset for ``preset_id`` or None (the apply route maps None -> 404)."""
    return VIDEO_PRESETS.get(preset_id)


# ---- built-in seed presets ------------------------------------------------
# model_key values validated against the live catalog (get_models_dict) on
# 2026-07-06; see the module/route docs for the mode->chain mapping.

register_preset(VideoGenPreset(
    id="realistic-edit-chain",
    name="Realistic edit-chain",
    description=("Instruction-driven frame evolution; best coherence; "
                 "auto-4bit on GPU."),
    mode="edit-chain",
    # Qwen-Image-Edit-2509 (transformers, image-to-image) — exact catalog key.
    model_key="a3527183~Qwen-Image-Edit-2509",
    strength=0.5,
    steps=28,
    guidance=4.0,
    width=1024,
    height=1024,
    n_frames=8,
    fps=8,
    negative="",
))

register_preset(VideoGenPreset(
    id="realistic-img2img",
    name="Realistic img2img sequence",
    description=("Off-start img2img: every frame conditions on the same start "
                 "frame (no drift); photoreal SDXL."),
    mode="img2img",
    # Photoreal SDXL — Juggernaut XL (comfy, text-to-image + image-to-image).
    # The bare candidate keys resolve only under the comfy- prefix in the live
    # catalog; Juggernaut XL is the true SDXL matching the 1024-native defaults.
    model_key="comfy-juggernautxl-ragnarok",
    strength=0.5,
    steps=30,
    guidance=6.0,
    width=1024,
    height=1024,
    n_frames=8,
    fps=8,
    negative="",
))

register_preset(VideoGenPreset(
    id="fast-draft",
    name="Fast draft",
    description=("Quick low-step turbo preview; text-to-image, no start "
                 "frame."),
    mode="text-to-image",
    # SDXL-Turbo (transformers, text-to-image) — exact catalog key.
    model_key="sdxl-turbo",
    strength=0.6,
    steps=6,
    guidance=1.5,
    width=768,
    height=768,
    n_frames=6,
    fps=8,
    negative="",
))


# ---- ComfyUI Field Guide presets ------------------------------------------
# Derived from the "ComfyUI Field Guide" recipe table; each model_key validated
# against the live catalog (get_models_dict) on 2026-07-06 and exercised by the
# 17-model battery (all 6/6 green). ``sampler`` values are honored by the comfy
# runner for direct image-gen requests (see the dataclass note); the scene path
# does not forward them yet.

register_preset(VideoGenPreset(
    id="photoreal-portrait-sd15",
    name="Photoreal portrait (SD1.5)",
    description=("Natural, unglamorous photorealism with real skin texture "
                 "(epiCRealism). SD1.5-native 512 — fast (~3.5s on the 3090)."),
    mode="text-to-image",
    model_key="comfy-epicrealism-naturalsinrc1vae",
    strength=0.45,
    steps=25,
    guidance=6.0,
    width=512,
    height=640,
    n_frames=8,
    fps=8,
    negative=("blurry, low quality, deformed hands, extra fingers, watermark, "
              "text"),
    sampler="euler",
))

register_preset(VideoGenPreset(
    id="photoreal-sdxl",
    name="Photoreal (SDXL)",
    description=("Flagship SDXL photorealism (Juggernaut XL) at 1024 — higher "
                 "fidelity, ~10s. Ideal sampler dpmpp_2m+karras (advisory)."),
    mode="text-to-image",
    model_key="comfy-juggernautxl-ragnarok",
    strength=0.45,
    steps=25,
    guidance=6.0,
    width=1024,
    height=1024,
    n_frames=8,
    fps=8,
    negative=("blurry, low quality, deformed hands, extra fingers, watermark, "
              "text"),
    sampler="dpmpp_2m",
))

register_preset(VideoGenPreset(
    id="anime-stylized",
    name="Anime / stylized",
    description=("Versatile anime/illustration, booru-tag fluent "
                 "(NeverEndingDream). SD1.5 512."),
    mode="text-to-image",
    model_key="comfy-neverendingdreamned-v122bakedvae",
    strength=0.45,
    steps=25,
    guidance=7.0,
    width=512,
    height=768,
    n_frames=8,
    fps=8,
    negative=("lowres, bad anatomy, bad hands, missing fingers, extra digit, "
              "watermark, text, error"),
    sampler="euler",
))

register_preset(VideoGenPreset(
    id="painterly-art",
    name="Painterly all-rounder",
    description=("The friendliest all-rounder — art to semi-real "
                 "(DreamShaper-8). SD1.5 512, fast."),
    mode="text-to-image",
    model_key="comfy-dreamshaper-8",
    strength=0.45,
    steps=25,
    guidance=7.0,
    width=512,
    height=640,
    n_frames=8,
    fps=8,
    negative=("blurry, low quality, deformed hands, extra fingers, watermark, "
              "text"),
    sampler="euler",
))

register_preset(VideoGenPreset(
    id="sdxl-lightning",
    name="SDXL lightning (fast quality)",
    description=("Distilled SDXL — near-full quality in 6 steps at cfg 2 "
                 "(DreamShaperXL Lightning), ~7.5s. Ideal sampler dpmpp_sde "
                 "(advisory: runtime uses euler until sampler plumbing lands)."),
    mode="text-to-image",
    model_key="comfy-dreamshaperxl-lightningdpmsde",
    strength=0.5,
    steps=6,
    guidance=2.0,
    width=1024,
    height=1024,
    n_frames=6,
    fps=8,
    negative="blurry, low quality, watermark",
    sampler="dpmpp_sde",
))


# ===========================================================================
# MOVIE TEMPLATES — curated goal-timeline presets for the Movie Maker tab
# ===========================================================================
# A movie template is a curated MovieSpec (see video_intel/movie_schema.py): a
# catalog ``model_key`` + the scene-template settings shared by every segment +
# a GOAL TIMELINE — an ordered tuple of contiguous, non-overlapping half-open
# ``[start_frame, end_frame)`` intervals that TILE ``[0, total)`` (total ==
# max(end_frame)). Each segment renders its own goal prompt; with ``chain`` the
# segments carry the previous frame for cross-segment coherence.
#
# This section MIRRORS the video-preset pattern above — a frozen ``*Preset``
# dataclass + a plain dict registry + ``register_*``/``available_*``/``get_*``
# accessors — so the two read the same way. Nothing here restarts a service or
# touches the serve/worker control plane; it is a static table the HTTP surface
# dumps (GET /movie/presets) and looks up (POST /movie/presets/<id>/apply).
#
# ``apply()`` returns a directly-POSTable ``/video/jobs/generate_movie`` body (a
# curated MovieSpec), so the Movie Maker tab can select a template and Generate
# with no further shaping — feeding ``apply()["request"]`` through
# ``movie_schema.make_movie`` yields a valid MovieSpec (proven by the self-test).


# ---------------------------------------------------------------------------
# Schema — one frozen movie template (a curated MovieSpec)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MoviePreset:
    """Immutable curated MovieSpec for the Movie Maker tab.

    ``model_key``  a real catalog key (the same source — get_models_dict — the
                   video presets validate against).
    ``goals``      the ORDERED, contiguous, non-overlapping goal timeline; each
                   entry is a plain ``{start_frame, end_frame, prompt}`` dict
                   whose half-open ranges tile ``[0, total)``.
    The scene-template fields (width/height/steps/guidance/fps/chain) are shared
    by every segment; the director knobs (vision_enabled/score_threshold) are the
    opt-in per-segment vision-scoring defaults. ``settings()`` bundles the shared
    scene-template knobs, ``to_dict()`` is the GET list shape, and ``apply()``
    returns the directly-POSTable generate_movie body.
    """

    id: str
    name: str
    description: str
    model_key: str
    # scene-template fields (shared by every segment)
    width: int
    height: int
    steps: int
    guidance: float
    fps: int
    chain: bool
    # the goal timeline — contiguous half-open intervals tiling [0, total)
    goals: Tuple[Dict[str, Any], ...]
    # scene-template img2img knobs (shared by every segment; both default so the
    # 6 pre-existing presets are unaffected). Mirror MovieSpec/make_movie, which
    # already accept both: ``strength`` is the img2img denoise (None -> backend
    # default), ``negative`` the shared negative prompt ("" -> none).
    strength: Optional[float] = None
    negative: str = ""
    # director knobs (opt-in per-segment vision scoring + retry)
    vision_enabled: bool = False
    score_threshold: int = 60
    recommended: str = "gpu"

    def settings(self) -> Dict[str, Any]:
        """The scene-template knobs shared by every segment (order/keys frozen)."""
        return {
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance": self.guidance,
            "fps": self.fps,
            "chain": self.chain,
            "strength": self.strength,
            "negative": self.negative,
        }

    def goals_list(self) -> List[Dict[str, Any]]:
        """The goal timeline as a fresh list of plain dicts (UI goal-editor rows)."""
        return [
            {
                "start_frame": g["start_frame"],
                "end_frame": g["end_frame"],
                "prompt": g["prompt"],
            }
            for g in self.goals
        ]

    def total_frames(self) -> int:
        """The movie's total frame count == max(end_frame) (goals tile [0, total))."""
        return max(g["end_frame"] for g in self.goals)

    def to_dict(self) -> Dict[str, Any]:
        """The pinned per-preset wire shape for GET /movie/presets."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "model_key": self.model_key,
            "settings": self.settings(),
            "goals": self.goals_list(),
            "total_frames": self.total_frames(),
            "strength": self.strength,
            "negative": self.negative,
            "vision_enabled": self.vision_enabled,
            "score_threshold": self.score_threshold,
            "recommended": self.recommended,
        }

    def request_body(self) -> Dict[str, Any]:
        """The directly-POSTable ``/video/jobs/generate_movie`` body — a curated
        MovieSpec (model + shared scene-template settings + the goal timeline +
        the director defaults). Feeding this through ``movie_schema.make_movie``
        yields a valid MovieSpec (see the movie self-test)."""
        return {
            "model_id": self.model_key,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance": self.guidance,
            "fps": self.fps,
            "assemble": True,
            "chain": self.chain,
            "strength": self.strength,
            "negative": self.negative,
            "goals": self.goals_list(),
            "vision_enabled": self.vision_enabled,
            "score_threshold": self.score_threshold,
        }

    def apply(self) -> Dict[str, Any]:
        """The pinned wire shape for POST /movie/presets/<id>/apply — an envelope
        (``ok`` mirrors the video apply) wrapping the directly-POSTable
        generate_movie request body."""
        return {
            "ok": True,
            "id": self.id,
            "name": self.name,
            "model_id": self.model_key,
            "request": self.request_body(),
        }


# ---------------------------------------------------------------------------
# Registry — insertion order is the dropdown order
# ---------------------------------------------------------------------------
MOVIE_PRESETS: Dict[str, MoviePreset] = {}


def register_movie_preset(preset: MoviePreset) -> None:
    if preset.id in MOVIE_PRESETS:
        raise KeyError(f"Movie preset {preset.id!r} already registered")
    MOVIE_PRESETS[preset.id] = preset


def available_movie_presets() -> List[MoviePreset]:
    """All movie presets in registration (dropdown) order."""
    return list(MOVIE_PRESETS.values())


def get_movie_preset(preset_id: str) -> Optional[MoviePreset]:
    """The movie preset for ``preset_id`` or None (the apply route maps None -> 404)."""
    return MOVIE_PRESETS.get(preset_id)


def _goal(start_frame: int, end_frame: int, prompt: str) -> Dict[str, Any]:
    """One goal-timeline row — a contiguous half-open [start_frame, end_frame)."""
    return {"start_frame": start_frame, "end_frame": end_frame, "prompt": prompt}


# ---- built-in movie templates ---------------------------------------------
# model_key values validated against the live catalog (get_models_dict) on
# 2026-07-06 — every key here is also carried by a video preset above. Each goal
# timeline is four contiguous 3-frame segments tiling [0, 12).

register_movie_preset(MoviePreset(
    id="golden-hour",
    name="Golden Hour",
    description="A tranquil mountain lake through one full day.",
    model_key="comfy-juggernautxl-ragnarok",
    width=1024,
    height=1024,
    steps=24,
    guidance=6.0,
    fps=8,
    chain=False,
    goals=(
        _goal(0, 3, "a tranquil mountain lake at dawn, soft mist, cool blue tones, "
                    "mirror reflections"),
        _goal(3, 6, "the same mountain lake at bright midday, clear blue sky, "
                    "sparkling sunlit water"),
        _goal(6, 9, "the same mountain lake at golden sunset, warm orange and pink "
                    "light on the water"),
        _goal(9, 12, "the same mountain lake at dusk, deep purple twilight, first "
                     "stars appearing"),
    ),
))

register_movie_preset(MoviePreset(
    id="four-seasons",
    name="Four Seasons",
    description="One stone cottage through spring, summer, autumn, winter.",
    model_key="comfy-juggernautxl-ragnarok",
    width=1024,
    height=1024,
    steps=24,
    guidance=6.0,
    fps=8,
    chain=False,
    goals=(
        _goal(0, 3, "a stone cottage in spring, cherry blossoms, fresh green grass, "
                    "gentle light"),
        _goal(3, 6, "the same stone cottage in summer, lush garden, bright flowers, "
                    "deep blue sky"),
        _goal(6, 9, "the same stone cottage in autumn, orange and red fallen leaves, "
                    "golden hour light"),
        _goal(9, 12, "the same stone cottage in winter, snow-covered roof, bare trees, "
                     "soft grey sky"),
    ),
))

register_movie_preset(MoviePreset(
    id="rose-bloom",
    name="Rose in Bloom",
    description="A macro rose opening petal by petal.",
    model_key="comfy-epicrealism-naturalsinrc1vae",
    width=640,
    height=640,
    steps=25,
    guidance=6.0,
    fps=8,
    chain=False,
    goals=(
        _goal(0, 3, "a tight red rose bud with dewdrops, macro photography, soft "
                    "blurred background"),
        _goal(3, 6, "the red rose bud just beginning to open, a few outer petals "
                    "loosening"),
        _goal(6, 9, "the red rose half-bloomed, petals unfurling, rich detail"),
        _goal(9, 12, "the red rose in full bloom, petals wide open, vivid velvety "
                     "detail"),
    ),
))

register_movie_preset(MoviePreset(
    id="storm-front",
    name="Storm Front",
    description="A meadow as a storm rolls in and clears.",
    model_key="comfy-juggernautxl-ragnarok",
    width=1024,
    height=1024,
    steps=24,
    guidance=6.0,
    fps=8,
    chain=False,
    goals=(
        _goal(0, 3, "a wide green meadow under a calm clear blue sky, gentle breeze"),
        _goal(3, 6, "the same meadow as dark storm clouds gather on the horizon, "
                    "wind picking up"),
        _goal(6, 9, "the same meadow in a heavy thunderstorm, driving rain, a bright "
                    "lightning bolt"),
        _goal(9, 12, "the same meadow just after the storm, a vivid rainbow, sky "
                     "clearing to blue"),
    ),
))

register_movie_preset(MoviePreset(
    id="anime-day",
    name="Anime: A Day",
    description="An anime character from morning to starry night.",
    model_key="comfy-neverendingdreamned-v122bakedvae",
    width=512,
    height=768,
    steps=25,
    guidance=7.0,
    fps=8,
    chain=False,
    goals=(
        _goal(0, 3, "anime girl by a classroom window, soft morning light, calm "
                    "expression"),
        _goal(3, 6, "anime girl walking home down a street lined with cherry "
                    "blossoms, afternoon sun"),
        _goal(6, 9, "anime girl sitting at a riverbank at sunset, warm orange sky, "
                    "wistful mood"),
        _goal(9, 12, "anime girl under a starry night sky, city lights below, "
                     "peaceful gentle smile"),
    ),
))

register_movie_preset(MoviePreset(
    id="cosmic-zoom",
    name="Cosmic Zoom",
    description="A fast journey from a whole galaxy down to a single star.",
    model_key="comfy-dreamshaperxl-lightningdpmsde",
    width=1024,
    height=1024,
    steps=6,
    guidance=2.0,
    fps=8,
    chain=False,
    goals=(
        _goal(0, 3, "a vast spiral galaxy in deep space, distant wide view, "
                    "scattered stars"),
        _goal(3, 6, "zooming toward a glowing blue and purple nebula, cosmic dust"),
        _goal(6, 9, "a closer view of swirling cosmic gas clouds and brilliant "
                    "clustered stars"),
        _goal(9, 12, "a single radiant star filling the frame, blinding light, "
                     "lens flare"),
    ),
))

# Empirically tuned (2026-07-06) img2img drift template — the FIRST movie preset
# to lean on strength + negative. REQUIRES a start image (an img2img chain): the
# person is carried from YOUR start frame, so without one it has nothing to follow.
# strength 0.45 is the sweet spot where the head-turns actually land (lower toward
# 0.35 for tighter identity but less motion). Four contiguous 3-frame segments
# tiling [0, 12).
register_movie_preset(MoviePreset(
    id="street-walk",
    name="Street Walk (bring a start image)",
    description=("Drift a person from YOUR start image: they walk down the street, "
                 "reach the corner, then look left and right. REQUIRES a start image "
                 "(this is an img2img chain — without one it can't follow your "
                 "subject). Strength 0.45 is the sweet spot where the head-turns "
                 "actually happen; lower it toward 0.35 for tighter identity but "
                 "less motion."),
    model_key="comfy-juggernautxl-ragnarok",
    width=512,
    height=680,
    steps=18,
    guidance=6.0,
    fps=8,
    chain=False,
    strength=0.45,
    negative=("different person, face change, identity change, deformed face, "
              "extra limbs, warped body, morphing, blurry"),
    goals=(
        _goal(0, 3, "the same person from the start image walking down a city "
                    "sidewalk, tracking shot following them, mid-stride, moving "
                    "forward"),
        _goal(3, 6, "the same person reaching the street corner and easing to a "
                    "stop, standing at the curb"),
        _goal(6, 9, "the same person at the corner turning their head to look to "
                    "the LEFT, profile view of their face"),
        _goal(9, 12, "the same person turning their head to look to the RIGHT, "
                     "profile of their face facing the other way"),
    ),
    vision_enabled=False,
    recommended="gpu",
))
