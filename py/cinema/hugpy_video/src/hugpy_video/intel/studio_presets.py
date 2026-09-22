"""Curated STUDIO clip presets — "ideal default loads" for the cinema-studio tier.

A tiny, frozen registry the Studio Clips station reads to offer a dropdown of
ready-made image-to-video / text-to-video setups. Unlike a scene ``VideoGenPreset``
(which names a fixed catalog ``model_key``) a studio preset names only what the
studio ENQUEUE route accepts — a ``capability`` ("i2v"/"t2v") + target geometry +
a ``vram_budget_gb`` — and lets the studio's own CAPABILITY ROUTER pick the model.
Selecting one is enough to pre-fill the generate affordance and POST a valid
``/video/studio/i2v`` body.

Deliberately MIRRORS the house preset pattern in ``video_intel/presets.py`` — a
frozen ``*Preset`` dataclass + a plain dict registry + ``register_*`` /
``available_*`` / ``get_*`` accessors — and, like ``MoviePreset``, ``apply()`` is a
PURE PREFILL envelope (no worker side-effects: nothing here warms a GPU or touches
the serve/worker control plane, it is a static table the HTTP surface dumps and
looks up).

Budget note (LOAD-BEARING): ``vram_budget_gb`` is what makes a preset bind the
INTENDED model class through the studio router. A sub-real budget (< the smallest
real footprint) deterministically binds the SYNTHETIC prover (the no-GPU demo
path); a real budget binds a real Wan model. Each seeded budget below was verified
to resolve to its intended model via ``studio.router.CapabilityRouter`` (see
``tests/test_studio_presets_route.py``). ``recommended`` is advisory/display only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Schema — one frozen studio-clip preset row
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StudioPreset:
    """Immutable "ideal default load" for a studio clip.

    Captures EXACTLY what the studio enqueue route (``POST /video/studio/i2v`` ->
    ``studio.job.make_studio_i2v``) accepts: a ``capability`` value, target
    geometry, a routing ``vram_budget_gb``, a ``seed`` and the C-prompt pair
    (``prompt``/``negative``). The model is NOT pinned here — the studio router
    resolves capability + resolution + budget to a concrete model at run time.

    ``prompt`` is a scaffold and may be "" (an image-conditioned i2v needs no text).
    ``recommended`` is an advisory/display string only (e.g. "synthetic (no GPU)" /
    "single 3090") — it changes nothing at run time.
    """

    id: str
    name: str
    description: str
    capability: str          # a Capability value: "i2v" | "t2v"
    width: int
    height: int
    fps: int
    vram_budget_gb: float
    seed: int = 0
    prompt: str = ""         # scaffold; "" is valid (image-conditioned i2v)
    negative: str = ""
    recommended: str = "gpu"  # advisory/display only — see module note
    requires_source: bool = False  # UI signal (additive): this preset needs an INPUT
                                    # to mean anything — a staged source clip for a v2v
                                    # restyle, or a start image for an i2v render. Rides
                                    # the wire (to_dict/apply) so the Studio Clips
                                    # station can gate the flow (a v2v preset forces the
                                    # restyle flow; an i2v preset that needs a start
                                    # image can warn before enqueue). It is NOT a
                                    # make_studio_i2v keyword, so it stays OUT of
                                    # request_body() — the source/start image is threaded
                                    # from the staged clip by the route, never the preset.
    requires_reference: bool = False  # UI signal (additive), the id_lock twin of
                                    # requires_source: this preset needs at least one
                                    # REFERENCE IMAGE of the subject to mean anything
                                    # (Wan VACE reference-to-video identity lock). Rides
                                    # the wire (to_dict/apply) so the Studio Clips station
                                    # gates the flow (an id_lock preset forces the
                                    # reference-picker before enqueue). Like
                                    # requires_source it is NOT a make_studio_i2v keyword,
                                    # so it stays OUT of request_body() — the reference
                                    # images are threaded by the route from the picker,
                                    # never baked into the preset.
    prompt_note: str = ""          # UI HONESTY badge (additive): a short note the
                                    # station shows about the prompt for this preset.
                                    # Set on SYNTHETIC presets to state plainly that the
                                    # prompt is recorded in the manifest but NOT rendered
                                    # (the synthetic prover's frames are a pure function
                                    # of seed + geometry) — so an evocative scaffold can
                                    # never again mislead an operator into expecting
                                    # prompt-driven pixels. "" = no badge. NOT a
                                    # make_studio_i2v keyword -> stays OUT of request_body().

    def request_body(self) -> Dict[str, Any]:
        """A directly-POSTable ``/video/studio/i2v`` body.

        Uses the FLAT geometry keys the route accepts (``width``/``height``/``fps``
        at top level — the route reads a nested ``resolution`` OR flat keys) and the
        ``negative`` alias the route honors, so every key here is ALSO a valid
        ``make_studio_i2v`` keyword. Feeding this straight through
        ``make_studio_i2v(**request_body())`` yields a valid ``StudioI2VSpec``
        (proven by the self-test) — the studio twin of ``MoviePreset.request_body``.
        """
        return {
            "capability": self.capability,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "vram_budget_gb": self.vram_budget_gb,
            "seed": self.seed,
            "prompt": self.prompt,
            "negative": self.negative,
        }

    def to_dict(self) -> Dict[str, Any]:
        """The pinned per-preset wire shape for GET /video/studio/presets."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "capability": self.capability,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "vram_budget_gb": self.vram_budget_gb,
            "seed": self.seed,
            "prompt": self.prompt,
            "negative": self.negative,
            "recommended": self.recommended,
            "requires_source": self.requires_source,
            "requires_reference": self.requires_reference,
            "prompt_note": self.prompt_note,
        }

    def apply(self) -> Dict[str, Any]:
        """The pinned wire shape for POST /video/studio/presets/<id>/apply — an
        envelope (``ok`` mirrors the movie apply) wrapping the directly-POSTable
        ``/video/studio/i2v`` request body. PURE PREFILL: no worker side-effects.

        ``requires_source`` is surfaced at the envelope TOP LEVEL (not inside
        ``request``) so the UI can gate the restyle flow off the apply result
        without re-reading the preset table: a v2v/restyle preset is dead-on-arrival
        unless the station hands it a staged source clip."""
        return {
            "ok": True,
            "id": self.id,
            "name": self.name,
            "capability": self.capability,
            "requires_source": self.requires_source,
            "requires_reference": self.requires_reference,
            "request": self.request_body(),
        }


# ---------------------------------------------------------------------------
# Registry — insertion order is the dropdown order
# ---------------------------------------------------------------------------
STUDIO_PRESETS: Dict[str, StudioPreset] = {}


def register_studio_preset(preset: StudioPreset) -> None:
    if preset.id in STUDIO_PRESETS:
        raise KeyError(f"Studio preset {preset.id!r} already registered")
    STUDIO_PRESETS[preset.id] = preset


def available_studio_presets() -> List[StudioPreset]:
    """All studio presets in registration (dropdown) order."""
    return list(STUDIO_PRESETS.values())


def get_studio_preset(preset_id: str) -> Optional[StudioPreset]:
    """The studio preset for ``preset_id`` or None (the apply route maps None -> 404)."""
    return STUDIO_PRESETS.get(preset_id)


# ---- built-in seed presets ------------------------------------------------
# Each budget was verified (2026-07-07) to resolve to its INTENDED model class
# through studio.router.CapabilityRouter — see tests/test_studio_presets_route.py.
# The two sub-real budgets (0.5 GB) bind the SYNTHETIC prover (no GPU); the real
# budgets bind a real Wan model. The router (not this table) picks the exact model.

# UI-honesty badge for the SYNTHETIC previews: their frames are a pure function of
# seed + geometry, so the prompt is provenance-only. Synthetic presets ship an EMPTY
# prompt (nothing evocative to prefill + mislead) AND carry this note so the station
# can state plainly that a prompt won't change a pixel here.
_SYNTHETIC_PROMPT_NOTE = ("Prompt is recorded in the manifest, not rendered "
                          "(synthetic prover — frames come from seed + geometry).")

register_studio_preset(StudioPreset(
    id="quick-preview-synthetic",
    name="Quick preview (synthetic, tiny budget)",
    description=("No-GPU procedural i2v — proves the whole studio spine end-to-end "
                 "(router -> manifest -> runner -> ffmpeg -> mp4) at a tiny 320x180 "
                 "with a sub-real 0.5 GB budget, so it binds the synthetic prover."),
    capability="i2v",
    width=320,
    height=180,
    fps=12,
    vram_budget_gb=0.5,
    seed=0,
    prompt="",
    negative="",
    recommended="synthetic (no GPU)",
    prompt_note=_SYNTHETIC_PROMPT_NOTE,
))

register_studio_preset(StudioPreset(
    id="preview-t2v-synthetic",
    name="Preview t2v (synthetic, tiny budget)",
    description=("No-GPU procedural TEXT-to-video demo at 512x512 with a sub-real "
                 "0.5 GB budget — binds the synthetic t2v prover (frames are a pure "
                 "function of seed + geometry; the prompt rides in the manifest)."),
    capability="t2v",
    width=512,
    height=512,
    fps=24,
    vram_budget_gb=0.5,
    seed=0,
    # EMPTY prompt (was an evocative "drone shot…" that misled an operator into
    # expecting prompt-driven pixels — synthetic ignores it). The honesty is carried by
    # prompt_note instead of a scaffold the runner never reads.
    prompt="",
    negative="",
    recommended="synthetic (no GPU)",
    prompt_note=_SYNTHETIC_PROMPT_NOTE,
))

register_studio_preset(StudioPreset(
    id="cinematic-720p-i2v",
    name="Cinematic 720p i2v (Wan 2.1)",
    description=("Landscape 1280x720 image-to-video at a real 16 GB budget — binds "
                 "the Wan 2.1 i2v identity-lock workhorse (INT8 to fit a 3090). "
                 "Bring a start image; the subject is carried from it."),
    capability="i2v",
    width=1280,
    height=720,
    fps=16,
    vram_budget_gb=16.0,
    seed=0,
    prompt="cinematic footage, natural motion, filmic lighting",
    negative="blurry, low quality, deformed, warped, morphing, flicker",
    recommended="single 3090 (INT8)",
))

register_studio_preset(StudioPreset(
    id="portrait-720p-i2v",
    name="Portrait 720p i2v (Wan 2.1)",
    description=("Vertical 720x1280 image-to-video at a real 16 GB budget — the "
                 "portrait twin of the cinematic preset, binds the Wan 2.1 i2v "
                 "workhorse. Ideal for phone-native subjects and talking heads."),
    capability="i2v",
    width=720,
    height=1280,
    fps=16,
    vram_budget_gb=16.0,
    seed=0,
    prompt="portrait video, natural head and body motion, soft lighting",
    negative="blurry, low quality, deformed face, warped body, morphing, flicker",
    recommended="single 3090 (INT8)",
))

register_studio_preset(StudioPreset(
    id="wan-t2v-1.3b-3090",
    name="Wan t2v 1.3B (single 3090)",
    description=("Text-to-video at 832x480 with a 6 GB budget — the consumer entry "
                 "point (Wan 2.1 T2V 1.3B, INT8) that fits a single 3090 "
                 "comfortably. No start image needed."),
    capability="t2v",
    width=832,
    height=480,
    fps=16,
    vram_budget_gb=6.0,
    seed=0,
    prompt="a lone astronaut walking across a red dune at sunrise",
    negative="blurry, low quality, distorted, flicker",
    recommended="single 3090 (INT8)",
))

register_studio_preset(StudioPreset(
    id="max-quality-t2v",
    name="Max quality t2v (Wan 2.2 A14B)",
    description=("Flagship 1280x720 text-to-video at a 24 GB budget — binds the Wan "
                 "2.2 A14B MoE t2v model at FP8 for the best open t2v quality. "
                 "Needs a big-VRAM box (or heavy offload)."),
    capability="t2v",
    width=1280,
    height=720,
    fps=16,
    vram_budget_gb=24.0,
    seed=0,
    prompt="an epic wide establishing shot of a futuristic city at golden hour",
    negative="blurry, low quality, distorted, flicker, artifacts",
    recommended="24 GB+ (FP8)",
))

# Slice (a) / v2v: the RESTYLE preset. Capability "v2v" resolves through the studio
# router (V2V -> Task.VACE_CONTROL) to the Wan 2.1 VACE 1.3B control model at INT8
# — the ONLY V2V model that fits a 6 GB budget (the 14B needs 14 GB+). VACE-1.3B's
# native envelope is R_480P = 832x480 (LANDSCAPE only): its supports_resolution
# COVERS a target iff 832 >= width AND 480 >= height, so a PORTRAIT (e.g. 480x832)
# or OVERSIZED request rejects at the router (RESOLUTION_UNSUPPORTED / NO_CAPABLE_
# MODEL) — the v2v footgun. This preset bakes the exact valid geometry (832x480 @16)
# and the 6 GB budget so selecting it can NEVER dead-on-arrive. requires_source=True
# because a restyle is a video TRANSFORM: with no staged source clip it has nothing
# to repaint (the runner returns SOURCE_MISSING). The source is threaded from the
# "Send to Studio" staged clip at enqueue time, so request_body() stays a valid,
# source-free make_studio_i2v body (the source is NOT a preset field).
register_studio_preset(StudioPreset(
    id="restyle-480p-v2v",
    name="Restyle 480p v2v (Wan 2.1 VACE 1.3B)",
    description=("Repaint / restyle an EXISTING clip at 832x480 landscape with a 6 GB "
                 "budget — binds the Wan 2.1 VACE 1.3B control model (INT8, fits a "
                 "single 3090). REQUIRES a source video: send a clip to the studio "
                 "first (there is nothing to restyle without one). 480p LANDSCAPE "
                 "only — portrait/oversized geometry is rejected by the model."),
    capability="v2v",
    width=832,
    height=480,
    fps=16,
    vram_budget_gb=6.0,
    seed=0,
    prompt="repaint this scene as a hand-painted watercolor, preserving the motion",
    negative="blurry, low quality, distorted, flicker, morphing, artifacts",
    recommended="single 3090 (INT8) · needs a source clip",
    requires_source=True,
))

# ---- TWO first-class TIERS (operator directive: both optionable) ----------
# DRAFT tier: Wan 2.1 T2V 1.3B — the consumer entry point. At 832x480 @16 with a 9 GB
# budget the router binds wan2.1-t2v-1.3b (verified via CapabilityRouter 2026-07-07:
# only the 1.3B fits — the A14B needs 16 GB+, LTX/mochi/cog reject on floor/resolution).
# A REAL diffusion model, so its prompt DOES drive pixels — a scaffold prompt is honest
# here (no prompt_note). No start image needed (text-to-video).
register_studio_preset(StudioPreset(
    id="draft-t2v-1.3b",
    name="Draft t2v — Wan 2.1 1.3B (single 3090)",
    description=("Fast DRAFT text-to-video at 832x480 @16fps with a 9 GB budget — binds "
                 "the Wan 2.1 T2V 1.3B consumer model (FP16, fits a single 3090). Real "
                 "32-step denoise; the prompt drives the pixels. No start image needed."),
    capability="t2v",
    width=832,
    height=480,
    fps=16,
    vram_budget_gb=9.0,
    seed=0,
    prompt="a lone figure walking down a rain-slicked neon street at night, cinematic",
    negative="blurry, low quality, distorted, flicker, deformed",
    recommended="single 3090 (FP16, draft)",
))

# QUALITY tier: Wan 2.1 I2V 14B at INT8 — the identity-lock workhorse. At 832x480 @16
# with a 16 GB budget the router binds wan2.1-i2v-14b-720p at INT8 (verified: it outranks
# the 2.2 A14B on the tighter-footprint tie-break; both fit at int8, 14B min=14 < 16).
# requires_source=True: an i2v render needs a START IMAGE (the subject is carried from
# it) — mirrors the restyle preset's requires_source signal so the station can warn
# before enqueue. A real diffusion model, so a scaffold prompt is honest (no prompt_note).
register_studio_preset(StudioPreset(
    id="quality-i2v-14b-int8",
    name="Quality i2v — Wan 2.1 14B INT8 (single 3090)",
    description=("QUALITY image-to-video at 832x480 @16fps with a 16 GB budget — binds "
                 "the Wan 2.1 I2V 14B identity-lock workhorse at INT8 (fits a single "
                 "3090). Real 32-step denoise. REQUIRES a start image: the subject is "
                 "carried from it (nothing to animate without one)."),
    capability="i2v",
    width=832,
    height=480,
    fps=16,
    vram_budget_gb=16.0,
    seed=0,
    prompt="natural, lifelike motion; the subject moves smoothly; filmic lighting",
    negative="blurry, low quality, deformed, warped, morphing, flicker, extra limbs",
    recommended="single 3090 (INT8, quality) · needs a start image",
    requires_source=True,
))

# IDENTITY LOCK: the flagship reference-to-video preset. Capability "id_lock" resolves
# through the studio router (ID_LOCK -> Task.VACE_CONTROL, preferred) to the Wan 2.1 VACE
# 1.3B control model at INT8 — the only VACE model that fits a sub-14 GB budget (the 14B
# needs 14 GB+). 832x480 LANDSCAPE only (VACE-1.3B's native envelope; portrait/oversized
# rejects at the router). requires_reference=True: an identity-locked render is DEFINED by
# the subject reference image(s) — with none it returns REFERENCE_MISSING. The references
# are threaded from the station's reference-picker at enqueue time, so request_body() stays
# a valid, reference-free make_studio_i2v body (references are NOT a preset field).
register_studio_preset(StudioPreset(
    id="identity-lock-1.3b",
    name="Identity lock 1.3B (Wan 2.1 VACE)",
    description=("Generate a video of YOUR subject at 832x480 landscape with a 9 GB "
                 "budget — binds the Wan 2.1 VACE 1.3B control model (INT8, fits a "
                 "single 3090) in reference-to-video mode. REQUIRES at least one "
                 "reference image of the subject (up to 4 are consumed); the identity "
                 "is preserved across the render. 480p LANDSCAPE only."),
    capability="id_lock",
    width=832,
    height=480,
    fps=16,
    vram_budget_gb=9.0,
    seed=0,
    prompt="the subject moves naturally, cinematic lighting, consistent identity",
    negative="blurry, low quality, deformed face, identity drift, morphing, flicker",
    recommended="single 3090 (INT8) · needs reference image(s)",
    requires_reference=True,
))

# ---- IDENTITY-LOCK TEMPLATES (Round 6) ------------------------------------
# Three first-class id_lock TEMPLATE cards around the generic identity-lock-1.3b:
# lock-person / lock-thing / scene-bias. All three resolve through the studio router
# (ID_LOCK -> Task.VACE_CONTROL) to the SAME Wan 2.1 VACE 1.3B control model, but at a
# 12 GB budget the router selects FP16 (verified 2026-07-08 via CapabilityRouter: the
# FP16 threshold for VACE-1.3B @ 832x480 is 10 GB — 9 GB binds INT8, >=10 GB binds
# FP16). 832x480 LANDSCAPE only (VACE-1.3B's native envelope; portrait/oversized rejects
# at the router). requires_reference=True on all three — an identity-locked render is
# DEFINED by the reference image(s); with none it returns REFERENCE_MISSING. The
# references are threaded from the station's reference-picker at enqueue time, so
# request_body() stays a valid, reference-free make_studio_i2v body. These differ from
# each other ONLY in curated prompt/negative + honest copy — they are intent templates,
# not distinct model bindings (a person, a product, a place each want different
# denoise guidance). Router-verified in tests/test_studio_lock_templates.py.
_LOCK_W, _LOCK_H, _LOCK_FPS, _LOCK_BUDGET = 832, 480, 16, 12.0

register_studio_preset(StudioPreset(
    id="lock-person",
    name="Lock a person (Wan 2.1 VACE 1.3B)",
    description=("Hold a PERSON'S identity across the render at 832x480 landscape with "
                 "a 12 GB budget — binds the Wan 2.1 VACE 1.3B control model (FP16, "
                 "fits a single 3090). Add 1–4 reference images of the SAME person; "
                 "multiple angles (front + 3/4 + profile) help the model keep the face "
                 "and build consistent across motion. 480p LANDSCAPE only."),
    capability="id_lock",
    width=_LOCK_W,
    height=_LOCK_H,
    fps=_LOCK_FPS,
    vram_budget_gb=_LOCK_BUDGET,
    seed=0,
    prompt="the person moves naturally, the same face and build throughout, cinematic lighting",
    negative="identity drift, face morphing, deformed face, extra limbs, warped body, blurry, flicker",
    recommended="single 3090 (FP16) · 1–4 face refs (multiple angles help)",
    requires_reference=True,
))

register_studio_preset(StudioPreset(
    id="lock-thing",
    name="Lock a product / object (Wan 2.1 VACE 1.3B)",
    description=("Hold a PRODUCT or OBJECT'S identity across the render at 832x480 "
                 "landscape with a 12 GB budget — binds the Wan 2.1 VACE 1.3B control "
                 "model (FP16, fits a single 3090). Add up to 4 clean reference images "
                 "of the item (a plain background and a couple of angles help); its "
                 "exact shape, markings and colour are preserved across the shot. "
                 "480p LANDSCAPE only."),
    capability="id_lock",
    width=_LOCK_W,
    height=_LOCK_H,
    fps=_LOCK_FPS,
    vram_budget_gb=_LOCK_BUDGET,
    seed=0,
    prompt="the object holds its exact shape, markings and colour, product-clean, cinematic lighting",
    negative="shape drift, warped geometry, morphing, wrong colour, smeared logo, blurry, flicker",
    recommended="single 3090 (FP16) · up to 4 object refs",
    requires_reference=True,
))

register_studio_preset(StudioPreset(
    id="scene-bias",
    name="Scene bias — a place (Wan 2.1 VACE 1.3B)",
    description=("Bias the render toward a PLACE at 832x480 landscape with a 12 GB "
                 "budget — binds the Wan 2.1 VACE 1.3B control model (FP16, fits a "
                 "single 3090). Add a reference image of the location. HONEST NOTE: a "
                 "place reference BIASES the setting, lighting and palette — it does "
                 "not hard-lock a scene the way a person/object identity does. 480p "
                 "LANDSCAPE only."),
    capability="id_lock",
    width=_LOCK_W,
    height=_LOCK_H,
    fps=_LOCK_FPS,
    vram_budget_gb=_LOCK_BUDGET,
    seed=0,
    prompt="the scene is set in the referenced place, matching its setting, lighting and palette",
    negative="wrong location, mismatched lighting, blurry, low quality, flicker, artifacts",
    recommended="single 3090 (FP16) · place BIASES the look (not a hard lock)",
    requires_reference=True,
))

# ---- SCENE-TO-SCENE CONTINUITY (Round 6) ----------------------------------
# scene-continuity: a v2v RESTYLE/EXTEND that HOLDS an identity across a cut. Capability
# "v2v" resolves through the router (V2V -> Task.VACE_CONTROL) to the SAME Wan 2.1 VACE
# 1.3B control model at FP16 (12 GB budget). requires_source=True — like restyle-480p-v2v
# it is a video TRANSFORM (needs a prior scene's clip; SOURCE_MISSING without one). The
# route ALSO accepts OPTIONAL reference_images on v2v (backend contract: _REF_CAPS =
# {"id_lock","v2v"}), so this preset can carry the subject's identity forward while
# restyling/extending the source — but references are OPTIONAL here (requires_reference
# stays False; a plain restyle is still valid). Both source and references are threaded
# from the station at enqueue time, so request_body() stays a valid, input-free
# make_studio_i2v body. 480p LANDSCAPE only. Router-verified in
# tests/test_studio_lock_templates.py.
register_studio_preset(StudioPreset(
    id="scene-continuity",
    name="Scene continuity (Wan 2.1 VACE 1.3B)",
    description=("Continue from a PRIOR scene's clip — restyle or extend it while "
                 "HOLDING an identity — at 832x480 landscape with a 12 GB budget "
                 "(Wan 2.1 VACE 1.3B, FP16, fits a single 3090). REQUIRES a source "
                 "clip (send one to the studio, or use a library clip as source). "
                 "OPTIONALLY add 1–4 reference images to carry a subject's identity "
                 "across the cut. 480p LANDSCAPE only."),
    capability="v2v",
    width=_LOCK_W,
    height=_LOCK_H,
    fps=_LOCK_FPS,
    vram_budget_gb=_LOCK_BUDGET,
    seed=0,
    prompt="continue the scene, preserve the motion and the subject's identity, cinematic",
    negative="identity drift, morphing, flicker, deformed, discontinuity, blurry, artifacts",
    recommended="single 3090 (FP16) · needs a source clip · optional identity refs",
    requires_source=True,
))
