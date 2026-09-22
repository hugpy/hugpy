"""Prompt compiler (k113b): decide, between every prompt, what the NEXT prompt
needs from the context built before it — which locked artifacts, how much of
each, how long the prompt should be, and whether one prompt is enough.

A segment prompt is not a paragraph someone hopes will work. It is compiled:

* **Context plan** — the ordered sections the prompt must carry, each with a
  priority and a token budget: screenplay scene, continuity ``state_before`` /
  ``state_after``, audio window, shot plan (block / light / camera), spatial
  manifest summary, tone profile, negative constraints, acceptance rubric.
  Sections are taken ONLY from the locked production artifacts the segment
  declares (sibling invariant: never from another segment's prompt or render).
* **Length** — derived from the target model's context limit and the
  measured difficulty of the shot; never "as long as possible".
* **Multiplicity** — the number of candidates and the *angles* they take.
  A static two-shot with no motion gets one prompt. A character catching an
  object with momentum under a moving camera gets several, each emphasising a
  different failure surface (identity, physics/trajectory, camera path,
  lighting/continuity), spread across distinct models when the selector has
  more than one defensible candidate. Multiplicity is bounded by the goal's
  quality profile and budget, and every choice carries its reason.

``DifficultySignals`` are extracted from the segment artifacts with honest
heuristics (counts, flags, durations) — they are inputs to a decision that is
journaled, not a hidden prior. Spatial signals (momentum, occlusion, camera
motion) come from the shot plan / spatial manifest when present; when a
segment carries no spatial manifest the compiler says so in its reasons.

Stdlib only. Token estimates are ``chars / 4`` unless a tokenizer seam is
supplied; this is a budget, not a bill.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

from hugpy_oracle.contracts import GoalSpec, QualityProfile

__all__ = [
    "ANGLES",
    "SIGNAL_SOURCES",
    "SPATIAL_SIGNALS",
    "ContextPlan",
    "ContextSection",
    "DifficultySignals",
    "PromptVariant",
    "compile_context",
    "difficulty_score",
    "extract_signals",
    "render_prompt",
]

# Failure surfaces a variant can emphasise. Order = priority when difficulty
# adds angles one at a time.
ANGLES: tuple[str, ...] = ("identity", "physics", "camera", "lighting", "performance")

_MOTION_WORDS = re.compile(
    r"\b(throw\w*|thrown|catch\w*|caught|fall\w*|fell|drop\w*|swing\w*|swung|"
    r"run\w*|ran|sprint\w*|jump\w*|leap\w*|collid\w*|collision|crash\w*|bounc\w*|roll\w*|"
    r"spin\w*|spun|slid\w*|hurl\w*|toss\w*|kick\w*|punch\w*|lunge\w*|dash\w*|"
    r"momentum|velocity|trajectory)\b", re.I)
# the subset that implies an OBJECT in flight / falling / struck
_OBJECT_MOTION = re.compile(
    r"\b(throw\w*|thrown|catch\w*|caught|fall\w*|fell|drop\w*|swing\w*|swung|"
    r"bounc\w*|roll\w*|hurl\w*|toss\w*|kick\w*|collid\w*|collision|crash\w*|"
    r"momentum|velocity|trajectory)\b", re.I)
_CAMERA_MOVE = re.compile(r"\b(dolly|track|tracking|pan|tilt|crane|handheld|steadicam|orbit|push.?in|"
                          r"pull.?out|whip|zoom|follow)\b", re.I)
_OCCLUSION = re.compile(r"\b(behind|occlud|partially hidden|through the|foreground|passes in front|"
                        r"blocks|obscure)\b", re.I)
_CLOTH_HAIR = re.compile(r"\b(cloak|cape|dress|skirt|scarf|hair|veil|flag|curtain|coat tails)\b", re.I)


@dataclass(frozen=True, slots=True)
class DifficultySignals:
    characters: int = 1
    moving_characters: int = 0
    interactions: int = 0            # character<->character or character<->prop contacts
    props_with_momentum: int = 0     # thrown / falling / swinging objects
    camera_motion: str = "static"    # static | simple | complex
    occlusion: bool = False
    cloth_or_hair: bool = False
    dialogue_words: int = 0
    duration_s: float = 4.0
    cuts: int = 0
    tone: float = 5.0                # 0 photoreal .. 10 graphic
    has_spatial_manifest: bool = False
    #: where the signals came from, overall: ``structured`` (every spatial
    #: signal read from a shot plan / manifest), ``regex_fallback`` (all of
    #: them guessed from prose) or ``mixed``.
    source: str = "regex_fallback"
    #: per-signal provenance: signal name -> ``structured`` | ``regex`` |
    #: ``declared`` (an explicit numeric/bool key on the segment) | ``default``.
    #: The steward reads this to see which signals are guesses.
    provenance: Mapping[str, str] = field(default_factory=dict)

    @property
    def guessed(self) -> tuple[str, ...]:
        """Signals that rest on prose regexes only."""
        return tuple(k for k, v in self.provenance.items() if v == "regex")

    def to_dict(self) -> dict[str, Any]:
        return {
            "characters": self.characters, "moving_characters": self.moving_characters,
            "interactions": self.interactions, "props_with_momentum": self.props_with_momentum,
            "camera_motion": self.camera_motion, "occlusion": self.occlusion,
            "cloth_or_hair": self.cloth_or_hair, "dialogue_words": self.dialogue_words,
            "duration_s": self.duration_s, "cuts": self.cuts, "tone": self.tone,
            "has_spatial_manifest": self.has_spatial_manifest,
            "source": self.source, "provenance": dict(self.provenance),
            "guessed": list(self.guessed),
        }


#: the signals that CAN be read structurally; the rest are counts / flags
#: from the locked artifacts and are never regex-guessed.
SPATIAL_SIGNALS: tuple[str, ...] = ("camera_motion", "props_with_momentum", "moving_characters",
                                    "occlusion", "cloth_or_hair")
SIGNAL_SOURCES: tuple[str, ...] = ("structured", "regex_fallback", "mixed")

# structured thresholds (metres / degrees over the segment)
_STILL_M = 0.05            # below this a track is "static"
_CAMERA_TURN_DEG = 2.0     # look-direction change that counts as a move
_OCCLUSION_DEG = 6.0       # angular separation under which two entities share a line of sight
_CLOTH_TAGS = frozenset({"cloth", "hair", "cape", "cloak", "dress", "skirt", "scarf", "veil", "flag",
                         "curtain", "fur", "coat"})


def _text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, Mapping):
        return " ".join(_text(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return " ".join(_text(x) for x in v)
    return str(v)


def _norm_tracks(manifest: Any) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """(camera keyframes, entity tracks, entity specs) from whatever the
    caller handed over: a ``spatial_sources.SpatialSource`` (or its dict), a
    ``SpatialSceneManifest`` (or its dict), or a bare mapping with any of the
    keys ``camera_track`` / ``entity_tracks`` / ``entities``."""
    if manifest is None:
        return [], [], []
    if hasattr(manifest, "to_dict") and not isinstance(manifest, Mapping):
        manifest = manifest.to_dict()
    if not isinstance(manifest, Mapping):
        return [], [], []
    cam = manifest.get("camera_track") or []
    tracks = manifest.get("entity_tracks") or []
    inner = manifest.get("manifest") if isinstance(manifest.get("manifest"), Mapping) else manifest
    entities = inner.get("entities") or []
    return ([k for k in cam if isinstance(k, Mapping)], [t for t in tracks if isinstance(t, Mapping)],
            [e for e in entities if isinstance(e, Mapping)])


def _vsub(a, b):
    return tuple(float(x) - float(y) for x, y in zip(a, b))


def _vlen(v) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _angle_deg(a, b) -> float:
    la, lb = _vlen(a), _vlen(b)
    if la == 0 or lb == 0:
        return 0.0
    c = sum(float(x) * float(y) for x, y in zip(a, b)) / (la * lb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _camera_motion_from_track(cam: list[Mapping[str, Any]]) -> str | None:
    if len(cam) < 2:
        return "static" if cam else None
    path = sum(_vlen(_vsub(b.get("position", ()), a.get("position", ()))) for a, b in zip(cam, cam[1:]))
    turn = max((_angle_deg(cam[0].get("forward", (0, 0, -1)), k.get("forward", (0, 0, -1))) for k in cam),
               default=0.0)
    fovs = [k.get("yfov_deg") for k in cam if k.get("yfov_deg") is not None]
    zoom = bool(fovs) and (max(fovs) - min(fovs)) > 0.5
    moves = int(path > _STILL_M) + int(turn > _CAMERA_TURN_DEG) + int(zoom)
    return "static" if moves == 0 else ("simple" if moves == 1 else "complex")


def _track_displacement(track: Mapping[str, Any]) -> float:
    pos = track.get("positions") or ()
    if len(pos) < 2:
        return 0.0
    return max(_vlen(_vsub(p, pos[0])) for p in pos)


def _occlusion_from_depth(cam: list[Mapping[str, Any]], tracks: list[Mapping[str, Any]]) -> bool | None:
    """Two entities on (nearly) the same line of sight at different depths at
    any sampled frame => one occludes the other. Needs a camera track AND at
    least two entity tracks; otherwise the question cannot be answered."""
    if not cam or len(tracks) < 2:
        return None
    n = min(len(k.get("positions") or ()) for k in tracks)
    if n == 0:
        return None
    for fi in range(min(n, len(cam))):
        cpos = cam[fi].get("position") or (0, 0, 0)
        fwd = cam[fi].get("forward") or (0, 0, -1)
        rays = []
        for t in tracks:
            d = _vsub(t["positions"][fi], cpos)
            depth = sum(float(x) * float(y) for x, y in zip(d, fwd))
            if depth > 0:                                   # in front of the camera
                rays.append((d, depth))
        for i in range(len(rays)):
            for j in range(i + 1, len(rays)):
                if _angle_deg(rays[i][0], rays[j][0]) < _OCCLUSION_DEG and \
                        abs(rays[i][1] - rays[j][1]) > _STILL_M:
                    return True
    return False


def _shot_mapping(shot: Any) -> Mapping[str, Any]:
    if shot is None:
        return {}
    if isinstance(shot, Mapping):
        return shot
    if hasattr(shot, "to_dict"):
        try:
            d = shot.to_dict()
            if isinstance(d, Mapping):
                return d
        except Exception:  # noqa: BLE001
            pass
    out = {}
    for k in ("camera", "blocking", "lighting", "moves", "motion"):
        if hasattr(shot, k):
            out[k] = getattr(shot, k)
    return out


def extract_signals(segment: Mapping[str, Any], *, shot: Any = None,
                    manifest: Any = None) -> DifficultySignals:
    """Signal extraction from a SegmentSpec-like mapping, STRUCTURED where the
    production can back it and regex over prose where it cannot.

    ``shot`` (a ``ShotPlanEntry`` or its mapping) and ``manifest`` (a
    ``spatial_sources.SpatialSource`` / ``SpatialSceneManifest`` / their
    dicts) are optional; when omitted they are looked up on ``segment`` under
    ``shot`` and ``spatial_manifest``. With a camera track, ``camera_motion``
    is measured (path length, look-direction turn, zoom); with entity tracks,
    ``props_with_momentum`` / ``moving_characters`` are counted from
    displacement; with both, ``occlusion`` is read from depth ordering along
    shared lines of sight; ``cloth_or_hair`` comes from entity tags. A shot
    plan whose ``camera`` mapping declares ``move`` / ``moves`` / ``motion``
    (a string or list of named moves, or ``static``) is also structured.

    Every signal records where it came from in ``provenance``; ``source`` is
    the roll-up the steward reads. The regex path is kept, as labelled
    fallback, for exactly the signals the structured inputs cannot answer."""
    prov: dict[str, str] = {}
    shot_map = _shot_mapping(shot if shot is not None else segment.get("shot"))
    spatial = manifest if manifest is not None else segment.get("spatial_manifest")
    cam_track, tracks, entities = _norm_tracks(spatial)

    chars = segment.get("characters") or ()
    n_chars = len(chars) if isinstance(chars, (list, tuple)) else (1 if chars else 0)
    blocking = _text(segment.get("blocking") or shot_map.get("blocking")) + " " + _text(segment.get("action")) \
        + " " + _text(segment.get("scene"))
    if n_chars:
        prov["characters"] = "declared"
    else:
        # continuity did not enumerate the cast: count distinct capitalised
        # actors in the blocking text (honest lower bound, never less than 1)
        names = {w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", blocking)} - {"The", "She", "He", "They", "Int", "Ext", "Day", "Night"}
        n_chars = len(names)
        char_entities = [e for e in entities if e.get("entity_type") == "character"]
        if char_entities:
            n_chars = max(n_chars, len(char_entities))
            prov["characters"] = "structured"
        else:
            prov["characters"] = "regex"
    n_chars_eff = max(1, n_chars)

    # --- motion: characters and props ---------------------------------------
    moving_words = len(set(m.group(0).lower() for m in _MOTION_WORDS.finditer(blocking)))
    props = segment.get("props") or ()
    prop_text = _text(props)
    if tracks:
        by_type: dict[str, int] = {}
        for t in tracks:
            if _track_displacement(t) > _STILL_M:
                by_type[t.get("entity_type", "prop")] = by_type.get(t.get("entity_type", "prop"), 0) + 1
        moving_chars = by_type.get("character", 0)
        momentum = sum(v for k, v in by_type.items() if k not in ("character", "camera_rig", "light", "environment"))
        prov["moving_characters"] = prov["props_with_momentum"] = "structured"
    else:
        moving_chars = min(n_chars_eff, moving_words) if moving_words else 0
        # object-motion verbs in the blocking imply a prop with momentum even when
        # the breakdown did not list the prop explicitly
        object_motion = len(_OBJECT_MOTION.findall(prop_text + " " + blocking))
        cap = max(1, len(props)) if isinstance(props, (list, tuple)) and props else 2
        momentum = min(object_motion, cap) if object_motion else 0
        prov["moving_characters"] = prov["props_with_momentum"] = "regex"

    # --- camera --------------------------------------------------------------
    cam = segment.get("camera") or shot_map.get("camera") or {}
    cam_text = _text(cam)
    camera_motion = _camera_motion_from_track(cam_track)
    if camera_motion is not None:
        prov["camera_motion"] = "structured"
    else:
        declared = None
        for source in (cam, shot_map.get("camera"), shot_map):
            if isinstance(source, Mapping):
                declared = source.get("move") or source.get("moves") or source.get("motion")
                if declared is not None:
                    break
        if declared is not None:
            if isinstance(declared, str):
                names = [declared] if declared.strip() and declared.strip().lower() not in ("static", "none", "locked") else []
            else:
                names = [str(x) for x in declared if str(x).strip() and str(x).strip().lower() not in ("static", "none")]
            camera_motion = "static" if not names else ("simple" if len(names) == 1 else "complex")
            prov["camera_motion"] = "structured"
        else:
            cam_moves = len(set(m.group(0).lower() for m in _CAMERA_MOVE.finditer(cam_text)))
            camera_motion = "static" if cam_moves == 0 else ("simple" if cam_moves == 1 else "complex")
            prov["camera_motion"] = "regex"

    # --- occlusion -----------------------------------------------------------
    occl = _occlusion_from_depth(cam_track, tracks)
    if occl is not None:
        prov["occlusion"] = "structured"
    else:
        occl = bool(_OCCLUSION.search(blocking + " " + cam_text))
        prov["occlusion"] = "regex"

    # --- cloth / hair --------------------------------------------------------
    tagged = [t for t in tracks if set(str(x).lower() for x in (t.get("tags") or ())) & _CLOTH_TAGS]
    if tracks and any(t.get("tags") is not None for t in tracks):
        cloth = bool(tagged)
        prov["cloth_or_hair"] = "structured"
    else:
        cloth = bool(_CLOTH_HAIR.search(blocking + " " + _text(segment.get("wardrobe"))))
        prov["cloth_or_hair"] = "regex"

    # --- the rest: declared counts, never guessed ----------------------------
    dialogue = segment.get("dialogue") or segment.get("lines") or ()
    words = len(_text(dialogue).split())
    prov["dialogue_words"] = "declared" if dialogue else "default"
    declared_inter = int(segment.get("interactions") or 0)
    if declared_inter:
        interactions, prov["interactions"] = declared_inter, "declared"
    elif tracks:
        interactions = 1 if (n_chars_eff > 1 and (moving_chars or momentum)) else 0
        prov["interactions"] = "structured"
    else:
        interactions = 1 if (n_chars_eff > 1 and moving_words) else 0
        prov["interactions"] = "regex"
    for key in ("duration_s", "cuts", "tone"):
        prov[key] = "declared" if segment.get(key) is not None else "default"

    spatial_prov = [prov[k] for k in SPATIAL_SIGNALS]
    if all(p == "structured" for p in spatial_prov):
        source = "structured"
    elif all(p == "regex" for p in spatial_prov):
        source = "regex_fallback"
    else:
        source = "mixed"
    return DifficultySignals(
        characters=n_chars_eff,
        moving_characters=min(n_chars_eff, moving_chars),
        interactions=interactions,
        props_with_momentum=momentum,
        camera_motion=camera_motion,
        occlusion=occl,
        cloth_or_hair=cloth,
        dialogue_words=words,
        duration_s=float(segment.get("duration_s") or 4.0),
        cuts=int(segment.get("cuts") or 0),
        tone=float(segment.get("tone", 5.0)),
        has_spatial_manifest=bool(spatial),
        source=source,
        provenance=prov,
    )


def difficulty_score(sig: DifficultySignals) -> tuple[float, tuple[str, ...]]:
    """0..1 with the contributing reasons. Additive, capped; each term is a
    known failure surface of identity-conditioned video generation."""
    score = 0.0
    why: list[str] = []

    def add(v: float, reason: str) -> None:
        nonlocal score
        if v > 0:
            score += v
            why.append(f"{reason} (+{v:.2f})")

    add(0.12 * max(0, sig.characters - 1), f"{sig.characters} characters")
    add(0.10 * sig.moving_characters, f"{sig.moving_characters} moving")
    add(0.12 * sig.interactions, f"{sig.interactions} interaction(s)")
    add(0.18 * sig.props_with_momentum, f"{sig.props_with_momentum} prop(s) with momentum")
    add({"static": 0.0, "simple": 0.10, "complex": 0.22}[sig.camera_motion], f"camera {sig.camera_motion}")
    add(0.12 if sig.occlusion else 0.0, "occlusion")
    add(0.08 if sig.cloth_or_hair else 0.0, "cloth/hair")
    add(0.05 * max(0.0, (sig.duration_s - 4.0) / 4.0), f"{sig.duration_s:.1f}s duration")
    add(0.06 * sig.cuts, f"{sig.cuts} cut(s)")
    add(0.06 if sig.dialogue_words > 25 else 0.0, f"{sig.dialogue_words} dialogue words")
    add(0.05 if (sig.tone <= 1.0 or sig.tone >= 9.0) else 0.0, f"tone endpoint {sig.tone}")
    if (sig.props_with_momentum or sig.interactions or sig.camera_motion != "static") and not sig.has_spatial_manifest:
        add(0.10, "spatial demands but NO spatial manifest (geometry unconstrained)")
    # soft saturation: stays monotonic (more hazards -> harder) without a hard ceiling
    return round(1.0 - math.exp(-score), 4), tuple(why)


# --------------------------------------------------------------------------- #
# context plan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ContextSection:
    name: str
    source: str              # which locked artifact it comes from
    priority: int            # 0 = never truncated
    budget_tokens: int
    required: bool = True
    content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "source": self.source, "priority": self.priority,
                "budget_tokens": self.budget_tokens, "required": self.required,
                "chars": len(self.content)}


@dataclass(frozen=True, slots=True)
class PromptVariant:
    index: int
    angle: str
    emphasis: str
    spread_model: bool

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "angle": self.angle, "emphasis": self.emphasis,
                "spread_model": self.spread_model}


@dataclass(frozen=True, slots=True)
class ContextPlan:
    segment_id: str
    difficulty: float
    difficulty_reasons: tuple[str, ...]
    signals: DifficultySignals
    target_tokens: int
    model_context_tokens: int
    sections: tuple[ContextSection, ...]
    variants: tuple[PromptVariant, ...]
    reasons: tuple[str, ...]
    sources: tuple[str, ...] = ()       # artifact refs / digests the plan reads

    @property
    def candidates(self) -> int:
        return len(self.variants)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id, "difficulty": self.difficulty,
            "difficulty_reasons": list(self.difficulty_reasons), "signals": self.signals.to_dict(),
            "target_tokens": self.target_tokens, "model_context_tokens": self.model_context_tokens,
            "sections": [s.to_dict() for s in self.sections],
            "variants": [v.to_dict() for v in self.variants], "candidates": self.candidates,
            "reasons": list(self.reasons), "sources": list(self.sources),
        }


_ANGLE_EMPHASIS = {
    "identity": "Preserve the exact facial structure, hairline, build and wardrobe of each named "
                "character from their reference pack across every frame; no feature drift.",
    "physics": "Objects in motion follow the stated trajectory and momentum: contact, release and "
               "landing points, timing and arcs are authoritative; limbs and props never intersect.",
    "camera": "The camera follows the specified path, lens and framing exactly; subjects keep their "
              "screen position and scale as the camera moves; no unrequested reframing.",
    "lighting": "Lighting direction, key/fill ratio, practicals and colour temperature match the "
                "continuity state; shadows stay consistent with the declared light sources.",
    "performance": "Facial performance and body language carry the emotional beat; mouth shapes "
                   "follow the locked dialogue timing.",
}

_SECTION_ORDER: tuple[tuple[str, str, int, bool], ...] = (
    # name, source artifact, priority (0 never truncated), required
    ("identity_constraints", "identity_pack", 0, True),
    ("negative_constraints", "shot_plan", 0, True),
    ("state_after", "continuity_bible", 0, True),
    ("screenplay_scene", "screenplay", 1, True),
    ("state_before", "continuity_bible", 1, True),
    ("shot", "shot_plan", 1, True),
    ("audio_window", "audio_master", 1, False),
    ("spatial", "spatial_manifest", 2, False),
    ("tone", "tone_profile", 2, True),
    ("lighting", "shot_plan", 3, False),
    ("production_design", "continuity_bible", 4, False),
)


def _spatial_summary(manifest: Any) -> str:
    """A compact, prose-able summary of the spatial authority for the prompt's
    ``spatial`` section: timebase, camera, entities (+ measured motion)."""
    cam, tracks, entities = _norm_tracks(manifest)
    if hasattr(manifest, "to_dict") and not isinstance(manifest, Mapping):
        manifest = manifest.to_dict()
    if not isinstance(manifest, Mapping):
        return ""
    inner = manifest.get("manifest") if isinstance(manifest.get("manifest"), Mapping) else manifest
    parts: list[str] = []
    tb = inner.get("timebase")
    if isinstance(tb, Mapping):
        parts.append(f"timebase {tb.get('fps')} fps frames {tb.get('start_frame')}..{tb.get('end_frame')}")
    if cam:
        motion = _camera_motion_from_track(cam)
        p0, p1 = cam[0].get("position"), cam[-1].get("position")
        parts.append(f"camera {motion}: from {tuple(round(float(x), 2) for x in p0)} to "
                     f"{tuple(round(float(x), 2) for x in p1)}")
    elif isinstance(inner.get("camera"), Mapping):
        parts.append(f"camera track {inner['camera'].get('track_uri')}")
    for t in tracks:
        d = _track_displacement(t)
        tags = ",".join(str(x) for x in (t.get("tags") or ()))
        parts.append(f"{t.get('entity_type')} {t.get('entity_id')} "
                     f"{'moves %.2f m' % d if d > _STILL_M else 'static'}" + (f" [{tags}]" if tags else ""))
    if not tracks:
        for e in entities:
            parts.append(f"{e.get('entity_type')} {e.get('entity_id')}")
    tp = inner.get("tier_profile")
    if isinstance(tp, Mapping):
        parts.append(f"tiers capture={tp.get('capture')} inference={tp.get('inference')} render={tp.get('render')}")
    return "; ".join(parts)


def _est_tokens(text: str, tokenizer: Callable[[str], int] | None) -> int:
    if tokenizer is not None:
        try:
            return int(tokenizer(text))
        except Exception:  # noqa: BLE001
            pass
    return max(1, len(text) // 4)


def compile_context(segment: Mapping[str, Any], *,
                    goal: GoalSpec | None = None,
                    model_context_tokens: int = 2048,
                    prompt_fraction: float = 0.35,
                    min_tokens: int = 120,
                    max_candidates: int | None = None,
                    eligible_models: int = 1,
                    tokenizer: Callable[[str], int] | None = None,
                    shot: Any = None,
                    manifest: Any = None) -> ContextPlan:
    """Decide the context, length and multiplicity for ``segment``.

    ``segment`` is a SegmentSpec-like mapping whose section fields are already
    the LOCKED artifact excerpts (the compiler never fetches siblings). Keys
    read for content: identity_constraints, negative_constraints, state_after,
    scene, state_before, shot, audio_window, spatial_manifest, tone,
    lighting, production_design; plus the signal keys of
    :func:`extract_signals`. ``shot`` / ``manifest`` go straight to
    :func:`extract_signals` so structured camera / trajectory / occlusion
    signals replace the prose regexes wherever the production can back them."""
    sig = extract_signals(segment, shot=shot, manifest=manifest)
    diff, why = difficulty_score(sig)
    profile = goal.quality if goal is not None else QualityProfile.BALANCED
    reasons: list[str] = []
    guessed = sig.guessed
    if sig.source == "structured":
        reasons.append("signals: structured (camera track / entity trajectories / tags)")
    elif guessed:
        reasons.append(f"signals: {sig.source}; regex-guessed from prose: {', '.join(guessed)}")

    # --- length --------------------------------------------------------------
    # difficulty buys detail, up to the model's prompt share; never the whole window
    base = int(model_context_tokens * prompt_fraction)
    target = int(min_tokens + (base - min_tokens) * (0.35 + 0.65 * diff))
    if profile is QualityProfile.PREVIEW:
        target = int(target * 0.6)
    target = max(min_tokens, min(target, base))
    reasons.append(f"length: {target} tokens of a {model_context_tokens}-token window "
                   f"(prompt share {prompt_fraction:.0%}, difficulty {diff:.2f}, profile {profile.value})")

    # --- sections ------------------------------------------------------------
    sections: list[ContextSection] = []
    sources: list[str] = []
    content_map = {
        "identity_constraints": _text(segment.get("identity_constraints")),
        "negative_constraints": _text(segment.get("negative_constraints")),
        "state_after": _text(segment.get("state_after")),
        "screenplay_scene": _text(segment.get("scene")),
        "state_before": _text(segment.get("state_before")),
        "shot": _text(segment.get("shot") or segment.get("camera")) + " " + _text(segment.get("blocking")),
        "audio_window": _text(segment.get("audio_window") or segment.get("dialogue")),
        "spatial": _text(segment.get("spatial_manifest_summary") or _spatial_summary(
            manifest if manifest is not None else segment.get("spatial_manifest"))),
        "tone": _text(segment.get("tone_profile") or f"tone {sig.tone}"),
        "lighting": _text(segment.get("lighting")),
        "production_design": _text(segment.get("production_design")),
    }
    remaining = target
    # pass 1: priority-0 sections get what they need (never truncated)
    for name, source, prio, required in _SECTION_ORDER:
        content = content_map.get(name, "")
        if not content and not required:
            continue
        need = _est_tokens(content, tokenizer) if content else 0
        if prio == 0:
            if not content and required:
                reasons.append(f"required section '{name}' is EMPTY in the locked artifacts")
            sections.append(ContextSection(name, source, prio, need, required, content))
            remaining -= need
            if content:
                sources.append(source)
    if remaining < 0:
        reasons.append(f"invariant sections alone need {target - remaining} tokens; target raised")
        target = target - remaining
        remaining = 0
    # pass 2: the rest share what is left by priority
    rest = [(n, s, p, r) for (n, s, p, r) in _SECTION_ORDER if p > 0 and (content_map.get(n) or r)]
    for name, source, prio, required in rest:
        content = content_map.get(name, "")
        need = _est_tokens(content, tokenizer) if content else 0
        weight = {1: 0.5, 2: 0.3, 3: 0.12, 4: 0.08}[prio]
        budget = min(need, max(int(remaining * weight), 24 if required else 0)) if need else 0
        if not content and required:
            reasons.append(f"required section '{name}' is EMPTY in the locked artifacts")
        sections.append(ContextSection(name, source, prio, budget, required, content))
        if content:
            sources.append(source)
        remaining -= budget
    if not sig.has_spatial_manifest and (sig.props_with_momentum or sig.camera_motion != "static"):
        reasons.append("no spatial manifest: physics/camera will be text-guided only (not a geometric guarantee)")

    # --- multiplicity --------------------------------------------------------
    cap = max_candidates if max_candidates is not None else (
        {QualityProfile.PREVIEW: 1, QualityProfile.BALANCED: 4, QualityProfile.BEST: 8}[profile])
    if goal is not None and goal.budget is not None and goal.budget.max_seconds is not None:
        cap = min(cap, max(1, int(goal.budget.max_seconds // max(1.0, 30.0 * sig.duration_s / 4.0))))
    n = 1 + int(round(diff * (cap - 1)))
    n = max(1, min(cap, n))
    angles: list[str] = ["identity"]
    if sig.props_with_momentum or sig.interactions or sig.moving_characters:
        angles.append("physics")
    if sig.camera_motion != "static":
        angles.append("camera")
    if sig.cloth_or_hair or sig.occlusion or sig.tone <= 1.0:
        angles.append("lighting")
    if sig.dialogue_words:
        angles.append("performance")
    variants = []
    for i in range(n):
        angle = angles[i % len(angles)]
        variants.append(PromptVariant(i, angle, _ANGLE_EMPHASIS[angle],
                                      spread_model=(eligible_models > 1 and i > 0)))
    reasons.append(f"multiplicity: {n} of max {cap} (difficulty {diff:.2f}); angles {[v.angle for v in variants]}"
                   + ("; spread across models" if eligible_models > 1 and n > 1 else
                      ("; single eligible model -> seed-varied" if n > 1 else "")))
    return ContextPlan(
        segment_id=str(segment.get("segment_id") or segment.get("id") or "segment"),
        difficulty=diff, difficulty_reasons=why, signals=sig, target_tokens=target,
        model_context_tokens=model_context_tokens, sections=tuple(sections),
        variants=tuple(variants), reasons=tuple(reasons), sources=tuple(dict.fromkeys(sources)),
    )


def _truncate(text: str, budget_tokens: int, tokenizer: Callable[[str], int] | None) -> str:
    if budget_tokens <= 0:
        return ""
    if _est_tokens(text, tokenizer) <= budget_tokens:
        return text
    # cut on sentence boundary where possible
    limit = budget_tokens * 4
    cut = text[:limit]
    dot = cut.rfind(". ")
    return (cut[: dot + 1] if dot > limit * 0.5 else cut).rstrip() + " …"


def render_prompt(plan: ContextPlan, variant: PromptVariant, *,
                  tokenizer: Callable[[str], int] | None = None,
                  header: str | None = None) -> str:
    """Assemble one variant's prompt from the plan: invariant sections whole,
    the rest within budget, the angle's emphasis first so it survives any
    downstream truncation by the model adapter."""
    parts: list[str] = []
    if header:
        parts.append(header.strip())
    parts.append(f"[{variant.angle.upper()} PRIORITY] {variant.emphasis}")
    for s in plan.sections:
        if not s.content:
            continue
        body = s.content if s.priority == 0 else _truncate(s.content, s.budget_tokens, tokenizer)
        if body:
            parts.append(f"[{s.name.upper()}] {body}")
    return "\n".join(parts)
