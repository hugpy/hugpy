"""STUDIO SPREAD — the whole-movie prompt generator (STUDIO-SPREAD-SPEC §1).

WHAT A SPREAD IS
----------------
ONE generator call that writes (or rewrites) the prompts for an ENTIRE movie at
once, holding the rows the user did not select as LOCKED CONTEXT. It is not a
convenience wrapper around N single-row calls — it is the opposite of that, and
the difference is the whole feature:

    per-row Generate (today)      spread (here)
    ------------------------      ---------------------------------
    N calls                       1 call
    N independent steering sets   ONE shared steering set
    each row blind to the others  every row sees the full timeline
    six rows -> six worlds        six rows -> six shots of one film

``prompt_seeds.steering_axes`` randomizes per call by design (a single Generate
click must not return the same prompt twice). Run that N times for a movie and
coherence is not merely absent, it is actively destroyed. So a spread draws one
``spread_axes`` set for the whole movie and varies only the BEAT per segment
(``prompt_seeds.beat_for_index``) — the inverse of the per-call randomization.

WHY THE MODEL IS TOLD THE JOINT MODES IN PLAIN LANGUAGE
-------------------------------------------------------
``joint_mode`` is a RENDER-TIME fact with dramatic consequences the writer must
respect: a ``still`` join carries a single frame and NO motion, so a segment that
opens mid-sprint will stutter; a ``vace_extend`` join carries the previous shot's
motion, so the segment must continue that movement rather than restart it; a
``cut`` carries nothing at all, so the segment is free to relocate. A model shown
the raw token ``vace_extend`` guesses. A model shown the sentence does not.

⚠ NOMENCLATURE NOTE (keeper, 2026-07-29): the spec's illustrative gloss for
``vace_extend`` — *"continues from one frame of the previous shot; motion is not
carried"* — actually describes ``still``. ``studio_movie_schema`` is explicit
that ``vace_extend`` exists precisely TO carry motion across the splice (it
conditions on the parent's trailing ``context_frames``). The plain language below
follows the SCHEMA, because a preface that misdescribes the join would teach the
model to write exactly the wrong continuation. The spec's INTENT — plain language,
never a raw token — is honoured in full.

OUTPUT DISCIPLINE (§1e)
-----------------------
Constrained JSON out, parsed by the package scavenger
(``utils/json_scavenge``) after ``<think>`` is stripped
(``utils/no_think``). A reply that will not parse is an HONEST 502 carrying the
raw (think-stripped) text — never an invented segment. And a segment the model
returns for a row that was NOT selected is DROPPED, not applied: a locked row is
the user's work, and a generator that could overwrite it would make selection a
lie.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hugpy_video.intel.prompt_seeds import beat_for_index, spread_axes, spread_steering_clause

__all__ = [
    "SpreadError",
    "SpreadParseError",
    "JOINT_MODE_PLAIN",
    "VALID_JOINT_MODES",
    "VALID_OPERATIONS",
    "SPREAD_SYSTEM",
    "NEGATIVE_SYSTEM",
    "SpreadRequest",
    "validate_context",
    "render_context_preface",
    "render_identity_block",
    "build_spread_request",
    "build_spread_messages",
    "build_negative_messages",
    "parse_spread_reply",
    "coordinate_spread",
]


class SpreadError(ValueError):
    """A structurally invalid request — the caller's to fix (maps to 400)."""


class SpreadParseError(ValueError):
    """The model's reply could not be read as the contracted JSON (maps to 502).

    Carries the raw (think-stripped) text so the failure is diagnosable and the
    caller can be told the truth instead of handed fabricated segments.
    """

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw or ""


# Mirrors studio_movie_schema._VALID_JOINT_MODES exactly — keep in sync.
VALID_JOINT_MODES = ("still", "vace_extend", "cut")

#: Joint mode -> the sentence the MODEL sees. Never the raw token.
JOINT_MODE_PLAIN: Dict[str, str] = {
    "still": ("this shot begins from a single still frame of the previous shot; "
              "no motion is carried across the join, so it must start from rest "
              "rather than mid-movement"),
    "vace_extend": ("this shot continues the previous shot, carrying its motion "
                    "across the join; it must continue that movement rather than "
                    "restart or reverse it"),
    "cut": ("this shot is a hard cut — nothing is carried over from the previous "
            "shot, so it may change location or framing freely, but the subject "
            "and the film's world must stay the same"),
}

#: Operations a spread may return for a row. ``keep`` is how a model declines to
#: change a row it was asked about — accepted so it never has to invent instead.
VALID_OPERATIONS = ("generate_from_direction", "enhance_scene", "generate", "keep")

_DEFAULT_DO_NOT_INVENT = ("age", "gender", "clothing", "ethnicity")


# --------------------------------------------------------------------------- #
# SYSTEM PROMPTS
# --------------------------------------------------------------------------- #
SPREAD_SYSTEM = (
    "You are a film director writing the shot prompts for ONE short film that "
    "will be rendered by a text-to-video model.\n"
    "\n"
    "RULES:\n"
    "1. Every shot belongs to the SAME film — same subject, same world, same "
    "visual style. Continuity is the point.\n"
    "2. Rewrite ONLY the segments marked REGENERATE. Segments marked LOCKED are "
    "the user's own work: use them as context, never restate or replace them.\n"
    "3. Write one chronological cinematic paragraph per segment. Plain "
    "renderable description: subject, action, camera, light. No markdown, no "
    "lists, no reasoning, no commentary.\n"
    "4. Incorporate EVERY direction given for a segment.\n"
    "5. NEVER invent identity attributes (age, gender, clothing, ethnicity, "
    "hair, build) for a named character. If it is not in the locked identity "
    "data, do not state it.\n"
    "6. Respect each segment's join description — it says what is carried over "
    "from the shot before it.\n"
    "\n"
    # THE CONTRACT (operator 2026-07-31): the UI already holds the segment
    # structure — ids, operations, negatives — and sends it in; the backend
    # assembles the result object from what it already knows. So the model is
    # asked for the ONE thing only it can write: the prose. A small instruct
    # model reliably writes a labelled paragraph; it does NOT reliably emit a
    # nested JSON envelope, and demanding one was the #1 spread failure ('did
    # not return the JSON object the spread contract requires').\n"
    "OUTPUT FORMAT — for EACH segment I ask you to write, output its id on its "
    "own line wrapped in double brackets, then its paragraph on the next "
    "line(s):\n"
    "\n"
    "[[the-segment-id]]\n"
    "The cinematic paragraph for that shot goes here.\n"
    "\n"
    "[[the-next-segment-id]]\n"
    "Its paragraph.\n"
    "\n"
    "Use the EXACT segment ids I give you. One paragraph per REGENERATE "
    "segment. No JSON, no headings, nothing else — just the [[id]] lines and "
    "the paragraphs."
)

NEGATIVE_SYSTEM = (
    "You write NEGATIVE prompts for image and video generation models.\n"
    "\n"
    "A negative prompt is an EXCLUSION LIST of artifacts and quality failures — "
    "it is not prose, not a scene, and not a description of what should happen. "
    "Never describe the shot; only name what must not appear in it.\n"
    "\n"
    "RULES:\n"
    "1. Output short comma-separated phrases, lowercase, no sentences.\n"
    "2. Name rendering artifacts and quality failures: deformed hands, extra "
    "fingers, extra limbs, fused faces, warped anatomy, flicker, temporal "
    "instability, morphing, blur, low resolution, jpeg artifacts, watermark, "
    "text overlay, logo, oversaturation, banding.\n"
    "3. Add exclusions specific to what is being generated when the shot makes "
    "them relevant (e.g. duplicate subject, changing wardrobe, identity drift, "
    "camera shake).\n"
    "4. Never repeat a phrase. No prose, no headings, no explanation, no "
    "reasoning, no markdown.\n"
    "\n"
    "Return ONLY the comma-separated list."
)


# --------------------------------------------------------------------------- #
# TYPED CONTEXT (§1c) — hint stays free-form; structured row state is TYPED.
# --------------------------------------------------------------------------- #
def _validate_segment_ref(value: Any, label: str) -> Optional[Dict[str, Any]]:
    """Validate one typed segment reference. Field names mirror
    ``studio_movie_schema.StudioMovieGoal`` EXACTLY (prompt / seed /
    branch_frame / joint_mode / negative) so the UI can hand a row straight
    through without a translation layer to get wrong."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise SpreadError(f"context.{label} must be an object")
    out: Dict[str, Any] = {}
    for key in ("segment_id", "prompt", "negative", "direction"):
        v = value.get(key)
        if v is None:
            continue
        if not isinstance(v, str):
            raise SpreadError(f"context.{label}.{key} must be a string")
        v = v.strip()
        if v:
            out[key] = v
    jm = value.get("joint_mode")
    if jm is not None:
        if jm not in VALID_JOINT_MODES:
            raise SpreadError(
                f"context.{label}.joint_mode must be one of "
                + "|".join(VALID_JOINT_MODES))
        out["joint_mode"] = jm
    for key in ("branch_frame", "seed", "index"):
        v = value.get(key)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, int):
            raise SpreadError(f"context.{label}.{key} must be an integer")
        if v < 0:
            raise SpreadError(f"context.{label}.{key} must be >= 0")
        out[key] = v
    return out


def _validate_identity(value: Any, label: str = "context.identity_profile"):
    """Validate a LOCKED identity block (§1e).

    Accepts the wire shape of a stored identity profile
    (``/video/identity-profiles`` -> ``{slug, name, reference_images, notes,
    ...}``) as well as the spec's ``{identity_id, name, reference_asset_ids,
    locked_description, do_not_invent}``, because the UI has the profile row in
    hand and should not have to rebuild it. Unknown extra keys are IGNORED
    rather than rejected — a profile row carries reconstructions/versions the
    generator has no use for.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [_validate_identity(v, label) for v in value if v is not None]
    if not isinstance(value, dict):
        raise SpreadError(f"{label} must be an object")
    ident = value.get("identity_id") or value.get("slug") or value.get("id")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SpreadError(f"{label}.name is required")
    refs = value.get("reference_asset_ids")
    if refs is None:
        refs = value.get("reference_images")
    if refs is not None and not isinstance(refs, list):
        raise SpreadError(f"{label}.reference_asset_ids must be a list")
    locked = (value.get("locked_description") or value.get("notes") or "")
    if not isinstance(locked, str):
        raise SpreadError(f"{label}.locked_description must be a string")
    dni = value.get("do_not_invent")
    if dni is None:
        dni = list(_DEFAULT_DO_NOT_INVENT)
    if not isinstance(dni, list) or any(not isinstance(x, str) for x in dni):
        raise SpreadError(f"{label}.do_not_invent must be a list of strings")
    return {
        "identity_id": ident if isinstance(ident, str) else None,
        "name": name.strip(),
        "reference_asset_ids": [str(r) for r in (refs or [])],
        "locked_description": locked.strip(),
        "do_not_invent": list(dni),
    }


def validate_context(context: Any) -> Dict[str, Any]:
    """Validate the typed ``context`` block; returns a normalized copy.

    ``hint`` stays FREE-FORM and untouched (§1c ruling: do NOT make hint the
    carrier for structured row state). Everything structured is typed and
    validated here, so a malformed row is a clean 400 instead of garbage
    silently rendered into the model preface.
    """
    if context is None:
        return {}
    if not isinstance(context, dict):
        raise SpreadError("context must be an object")
    out: Dict[str, Any] = {}
    for label in ("segment", "previous_segment", "next_segment"):
        ref = _validate_segment_ref(context.get(label), label)
        if ref:
            out[label] = ref
    ident = _validate_identity(context.get("identity_profile"))
    if ident:
        out["identity_profile"] = ident
    return out


def render_identity_block(identity) -> str:
    """The LOCKED identity preface (§1e).

    Both benchmarked models invented ages and wardrobes when handed a bare
    character NAME, so the name is never sent alone: it always arrives with what
    is known and an explicit list of what must not be filled in.
    """
    if not identity:
        return ""
    people = identity if isinstance(identity, list) else [identity]
    lines = ["LOCKED IDENTITY — these people already exist. Use them as given."]
    for p in people:
        if not p:
            continue
        bits = [f'- {p.get("name")}']
        if p.get("identity_id"):
            bits.append(f'(identity {p["identity_id"]})')
        lines.append(" ".join(bits))
        if p.get("locked_description"):
            lines.append(f'    known: {p["locked_description"]}')
        refs = p.get("reference_asset_ids") or []
        if refs:
            lines.append(f"    reference images on file: {len(refs)}")
        dni = p.get("do_not_invent") or []
        if dni:
            lines.append("    DO NOT INVENT OR STATE: " + ", ".join(dni)
                         + " — if it is not written above, do not describe it")
    return "\n".join(lines)


def render_context_preface(ctx: Dict[str, Any]) -> str:
    """Render the validated typed context into the model preface.

    Joint modes become SENTENCES (see the module docstring) and a branch frame
    becomes "conditioned on frame N of the previous shot" — the model has no
    idea what ``branch_frame: 37`` means, and inventing a meaning for it is
    worse than not sending it.
    """
    if not ctx:
        return ""
    blocks: List[str] = []
    labels = (
        ("previous_segment", "THE SHOT BEFORE THIS ONE"),
        ("segment", "THIS SHOT (current state)"),
        ("next_segment", "THE SHOT AFTER THIS ONE"),
    )
    for key, title in labels:
        ref = ctx.get(key)
        if not ref:
            continue
        lines = [title + ":"]
        if ref.get("prompt"):
            lines.append(f'    prompt: {ref["prompt"]}')
        if ref.get("negative"):
            lines.append(f'    negative: {ref["negative"]}')
        if ref.get("direction"):
            lines.append(f'    direction from the user: {ref["direction"]}')
        jm = ref.get("joint_mode")
        if jm:
            lines.append(f"    join: {JOINT_MODE_PLAIN[jm]}")
        if ref.get("branch_frame") is not None and jm != "cut":
            lines.append(
                f'    it starts from frame {ref["branch_frame"]} of the shot '
                "before it, not from that shot's end")
        blocks.append("\n".join(lines))
    ident = render_identity_block(ctx.get("identity_profile"))
    if ident:
        blocks.append(ident)
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# SPREAD REQUEST (§1a)
# --------------------------------------------------------------------------- #
_STYLE_BIBLE_KEYS = ("world", "subject", "visual_style",
                     "camera_language", "color_language")


@dataclass(frozen=True)
class SpreadRequest:
    movie_query: str
    style_bible: Dict[str, str]
    fixed_segments: Tuple[Dict[str, Any], ...]
    target_segments: Tuple[Dict[str, Any], ...]
    global_negative: Tuple[str, ...]
    steering_seed: Optional[int]
    steering: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    hint: str = ""
    kind: str = "movie"

    @property
    def target_ids(self) -> Tuple[str, ...]:
        return tuple(s["segment_id"] for s in self.target_segments)


def _validate_segment_list(value: Any, label: str, locked: bool) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise SpreadError(f"{label} must be a list")
    out: List[Dict[str, Any]] = []
    for i, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise SpreadError(f"{label}[{i}] must be an object")
        sid = raw.get("segment_id")
        if not isinstance(sid, str) or not sid.strip():
            raise SpreadError(f"{label}[{i}].segment_id is required")
        ref = _validate_segment_ref(raw, f"{label}[{i}]") or {}
        ref["segment_id"] = sid.strip()
        ref["locked"] = locked
        # Timeline position: an explicit index wins (rows can arrive in any
        # order once a UI mixes selected + unselected); list order backs it up.
        ref.setdefault("index", i if locked else i)
        out.append(ref)
    return out


def build_spread_request(body: Dict[str, Any]) -> SpreadRequest:
    """Validate a ``mode="spread"`` body into a SpreadRequest. Raises SpreadError."""
    movie_query = body.get("movie_query")
    if movie_query is not None and not isinstance(movie_query, str):
        raise SpreadError("movie_query must be a string")
    movie_query = (movie_query or "").strip()

    sb_raw = body.get("style_bible") or {}
    if not isinstance(sb_raw, dict):
        raise SpreadError("style_bible must be an object")
    style_bible = {}
    for k in _STYLE_BIBLE_KEYS:
        v = sb_raw.get(k)
        if v is None:
            continue
        if not isinstance(v, str):
            raise SpreadError(f"style_bible.{k} must be a string")
        if v.strip():
            style_bible[k] = v.strip()

    fixed = _validate_segment_list(body.get("fixed_segments"), "fixed_segments", True)
    targets = _validate_segment_list(body.get("target_segments"), "target_segments", False)
    if not targets:
        raise SpreadError('target_segments must contain at least one segment '
                          'for mode "spread"')
    seen = set()
    for s in list(fixed) + list(targets):
        if s["segment_id"] in seen:
            raise SpreadError(f'duplicate segment_id {s["segment_id"]!r}')
        seen.add(s["segment_id"])

    gn = body.get("global_negative")
    if gn is None:
        global_negative: Tuple[str, ...] = ()
    elif isinstance(gn, str):
        global_negative = tuple(p.strip() for p in gn.split(",") if p.strip())
    elif isinstance(gn, list):
        if any(not isinstance(x, str) for x in gn):
            raise SpreadError("global_negative must be a list of strings")
        global_negative = tuple(x.strip() for x in gn if x.strip())
    else:
        raise SpreadError("global_negative must be a list of strings or a string")

    seed = body.get("steering_seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise SpreadError("steering_seed must be an integer")

    ctx_raw = body.get("context") or {}
    if not isinstance(ctx_raw, dict):
        raise SpreadError("context must be an object")
    hint = ctx_raw.get("hint")
    if hint is not None and not isinstance(hint, str):
        raise SpreadError("context.hint must be a string")
    kind = ctx_raw.get("kind") or "movie"
    ctx = validate_context(ctx_raw)

    # ONE steering set for the whole spread — the coherence mechanism (§1a).
    axes = spread_axes(kind, seed=seed)

    return SpreadRequest(
        movie_query=movie_query,
        style_bible=style_bible,
        fixed_segments=tuple(fixed),
        target_segments=tuple(targets),
        global_negative=global_negative,
        steering_seed=seed,
        steering=axes,
        context=ctx,
        hint=(hint or "").strip(),
        kind=kind,
    )


def _render_timeline(req: SpreadRequest) -> str:
    """Every segment, in TIMELINE ORDER, locked ones included.

    The locked rows are what make this a spread rather than N rolls: the model
    can only write a segment that continues from its actual neighbour if it can
    SEE that neighbour, even when that neighbour is not being rewritten.
    """
    rows = sorted(list(req.fixed_segments) + list(req.target_segments),
                  key=lambda s: (s.get("index", 0), s["segment_id"]))
    total = len(rows)
    out: List[str] = []
    for position, s in enumerate(rows):
        mark = "LOCKED — do not rewrite" if s.get("locked") else "REGENERATE"
        lines = [f'SEGMENT {position + 1} of {total} [{s["segment_id"]}] — {mark}']
        if s.get("prompt"):
            lines.append(f'    current prompt: {s["prompt"]}')
        elif not s.get("locked"):
            lines.append("    current prompt: (empty — write it)")
        if s.get("direction"):
            lines.append(f'    the user asks for: {s["direction"]}')
        if s.get("negative"):
            lines.append(f'    negative: {s["negative"]}')
        jm = s.get("joint_mode")
        if position > 0 and jm:
            lines.append(f"    join: {JOINT_MODE_PLAIN[jm]}")
        if position > 0 and s.get("branch_frame") is not None and jm != "cut":
            lines.append(
                f'    it starts from frame {s["branch_frame"]} of the previous '
                "shot, not from that shot's end")
        if not s.get("locked"):
            lines.append(f"    beat for this shot: {beat_for_index(position, total)}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def build_spread_messages(req: SpreadRequest) -> List[Dict[str, str]]:
    """The ONE generator call's messages. Never called per segment."""
    parts: List[str] = []
    if req.movie_query:
        parts.append(f"THE FILM THE USER ASKED FOR:\n{req.movie_query}")
    if req.style_bible:
        parts.append("STYLE BIBLE (binding for every shot):\n" + "\n".join(
            f"- {k.replace('_', ' ')}: {v}" for k, v in req.style_bible.items()))
    parts.append(spread_steering_clause(req.steering))
    ident = render_identity_block(req.context.get("identity_profile"))
    if ident:
        parts.append(ident)
    if req.global_negative:
        parts.append("EXCLUDE FROM EVERY SHOT (the movie's negative prompt):\n"
                     + ", ".join(req.global_negative))
    parts.append("THE TIMELINE:\n\n" + _render_timeline(req))
    if req.hint:
        parts.append(f"Additional context to honour: {req.hint}")
    ids = ", ".join(req.target_ids)
    parts.append(
        f"Write ONLY these {len(req.target_ids)} segments, as ONE coherent "
        f"continuous piece so they hold together as a single film: {ids}.\n"
        "For each, output its id on its own line in double brackets, then its "
        "paragraph — for example:\n"
        f"[[{req.target_ids[0]}]]\n"
        "<the shot's cinematic paragraph>\n"
        "Use these exact ids. No JSON, no headings, nothing else."
    )
    return [
        {"role": "system", "content": SPREAD_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_negative_messages(draft: str, ctx: Dict[str, Any], hint: str = "",
                            subject: str = "") -> List[Dict[str, str]]:
    """Messages for ``mode="negative"`` (§1b).

    Deliberately its OWN framing. Reusing the scene-prose system prompt here
    produces a poem: the assist system prompt asks for a vivid description, and a
    negative asked for vividly is a description of the thing you were trying to
    exclude.
    """
    parts: List[str] = []
    if subject:
        parts.append(f"The shot being generated:\n{subject}")
    preface = render_context_preface(ctx)
    if preface:
        parts.append(preface)
    if draft:
        parts.append("The existing negative prompt — keep what is useful, remove "
                     "duplicates, and extend it:\n" + draft)
    if hint:
        parts.append(f"Additional context to honour: {hint}")
    parts.append("Write the negative prompt as comma-separated phrases only.")
    return [
        {"role": "system", "content": NEGATIVE_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


# --------------------------------------------------------------------------- #
# PARSING THE REPLY — honest failure, and locked rows are UNTOUCHABLE
# --------------------------------------------------------------------------- #
def _clean_str(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


# The [[segment-id]] marker the new contract asks for. Tolerant of surrounding
# markdown/punctuation (** [[id]] ** :), and of one OR two brackets, because a
# small model drops a bracket often enough to matter. The id is matched back to
# a REQUESTED id, so a stray "[[note]]" never becomes a segment.
_LABEL_RE = re.compile(r"\[\[\s*(.+?)\s*\]\]|(?<!\[)\[\s*([^\[\]\n]{1,80}?)\s*\](?!\])")


def _norm_id(s: str) -> str:
    """Fold an id for tolerant matching: lowercase, and strip everything that is
    not a letter or digit (so 'seg-1', 'seg_1', 'Seg 1', 'SEG1' all agree)."""
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _split_paragraphs(text: str) -> List[str]:
    """Blank-line-separated, trimmed, non-empty blocks — the natural shape of a
    model that writes 'one paragraph per shot' without being told to label them."""
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]


def _target_meta(target) -> "Tuple[List[str], Dict[str, Dict[str, Any]]]":
    """Return (ordered_ids, per-id meta) from a SpreadRequest OR a bare id list.

    The backend already HOLDS the structure — operations, negatives, which rows
    carry a direction — so the model is never asked to reproduce it (operator
    2026-07-31). ``meta[id]`` = {operation, negative, directions_used} assembled
    from the request; a bare id list (older callers/tests) gets sane defaults.
    """
    segs = getattr(target, "target_segments", None)
    if segs is None:
        ids = [str(x) for x in target]
        return ids, {i: {"operation": "generate", "negative": "",
                         "directions_used": []} for i in ids}
    ordered = list(getattr(target, "target_ids", ()) or
                   [s["segment_id"] for s in segs])
    meta: Dict[str, Dict[str, Any]] = {}
    for s in segs:
        sid = s.get("segment_id")
        has_dir = bool(_clean_str(s.get("direction")))
        meta[sid] = {
            "operation": "generate_from_direction" if has_dir else "generate",
            "negative": _clean_str(s.get("negative")),
            "directions_used": [0] if has_dir else [],
        }
    return ordered, meta


def _labelled_blocks(raw: str, norm_to_id: Dict[str, str]) -> "Dict[str, str]":
    """``{segment_id: prose}`` for every [[id]] / [id] marker that resolves to a
    REQUESTED id. The prose is the text from the marker to the next marker."""
    out: Dict[str, str] = {}
    hits = []
    for m in _LABEL_RE.finditer(raw):
        label = m.group(1) if m.group(1) is not None else m.group(2)
        sid = norm_to_id.get(_norm_id(label))
        if sid:
            hits.append((m.start(), m.end(), sid))
    for i, (_s, end, sid) in enumerate(hits):
        nxt = hits[i + 1][0] if i + 1 < len(hits) else len(raw)
        prose = raw[end:nxt].strip().strip(":").strip()
        if prose and sid not in out:   # first labelled block for an id wins
            out[sid] = prose
    return out


def parse_spread_reply(text, target) -> Dict[str, Any]:
    """Divvy ONE coherent generator reply into per-segment replacements.

    The spread is a SINGLE call for continuity (the model sees the whole
    timeline and writes all the requested shots as one coherent piece); this
    function splits that reply back into the TARGET rows. ``target`` may be a
    :class:`SpreadRequest` (preferred — carries operations/negatives to
    assemble) or a bare iterable of target ids (older callers/tests).

    Readers, in order, each falling through to the next (operator 2026-07-31:
    "it should generate the scenes, which it probably did — it's the parser on
    this end that is no good"):
      1. ``[[segment-id]]`` labelled paragraphs (the contract we now ask for);
      2. the legacy JSON envelope (a model that still emits one still works);
      3. POSITIONAL divvy — an unlabelled coherent reply split into paragraphs
         and mapped onto the target rows IN TIMELINE ORDER. This is the case
         that was failing: the model wrote N good scenes and the JSON-only
         parser threw them away.

    NEVER fabricates: a target the reply doesn't cover comes back under
    ``missing_segments`` and is left unchanged. Raises :class:`SpreadParseError`
    (carrying the raw) only when the reply is empty or nothing maps to any
    requested row.
    """
    from hugpy_engine.utils.json_scavenge import extract_json_array, extract_json_object

    raw = (text or "").strip()
    ordered, meta = _target_meta(target)
    wanted_set = set(ordered)
    norm_to_id = {_norm_id(sid): sid for sid in ordered}
    warnings: List[str] = []
    invented: List[str] = []
    # {segment_id: (prose, op_override, neg_override, continuity, directions_override)}
    prose_by_id: Dict[str, str] = {}
    op_override: Dict[str, str] = {}
    neg_override: Dict[str, str] = {}
    cont_by_id: Dict[str, str] = {}
    dir_override: Dict[str, List[Any]] = {}

    if not raw:
        raise SpreadParseError("the generator returned an empty reply", raw)

    # ── reader 1: labelled [[id]] blocks ────────────────────────────────────
    blocks = _labelled_blocks(raw, norm_to_id)
    for sid, prose in blocks.items():
        prose_by_id[sid] = prose

    # ── reader 2: legacy JSON envelope ──────────────────────────────────────
    if not prose_by_id:
        parsed: Any = extract_json_object(raw)
        if parsed is None:
            arr = extract_json_array(raw, accept_lone_object=False)
            if arr is not None:
                parsed = {"segments": arr}
        if isinstance(parsed, dict) and isinstance(parsed.get("segments"), list):
            warnings += [w for w in (parsed.get("warnings") or []) if isinstance(w, str)]
            invented += [w for w in (parsed.get("invented_identity_attributes") or [])
                         if isinstance(w, str)]
            for row in parsed["segments"]:
                if not isinstance(row, dict):
                    continue
                raw_sid = _clean_str(row.get("segment_id"))
                sid = norm_to_id.get(_norm_id(raw_sid))
                if not sid:
                    if raw_sid:
                        # A LOCKED (or unknown) row: dropped LOUDLY. Applying a
                        # row the user did not select would overwrite work they
                        # chose to keep, making the selection checkbox a lie.
                        warnings.append(
                            f"the generator returned segment {raw_sid!r}, which "
                            "was not selected — it was discarded and that row is "
                            "unchanged")
                    continue
                if sid in prose_by_id:
                    continue
                prose = _clean_str(row.get("prompt"))
                if not prose:
                    continue
                prose_by_id[sid] = prose
                op = row.get("operation")
                if op in VALID_OPERATIONS:
                    op_override[sid] = op
                if _clean_str(row.get("negative")):
                    neg_override[sid] = _clean_str(row.get("negative"))
                if _clean_str(row.get("continuity_note")):
                    cont_by_id[sid] = _clean_str(row.get("continuity_note"))
                if isinstance(row.get("directions_used"), list):
                    dir_override[sid] = row["directions_used"]
                invented += [x for x in (row.get("invented_identity_attributes") or [])
                             if isinstance(x, str)]

    # ── reader 3: positional divvy of an unlabelled coherent reply ──────────
    # Only fires when the reply can cover the request: one target takes the whole
    # reply; N targets need at least N paragraphs (assign the first N in timeline
    # order). FEWER paragraphs than multiple targets is NOT a coherent N-scene
    # reply — a preamble line like "Sure! Here are your shots" must fail
    # honestly (below), never be pasted into a shot.
    if not prose_by_id:
        paras = _split_paragraphs(raw)
        if len(ordered) == 1:
            prose_by_id[ordered[0]] = raw
        elif len(paras) >= len(ordered):
            for sid, para in zip(ordered, paras):
                prose_by_id[sid] = para
            if len(paras) > len(ordered):
                warnings.append(
                    f"the generator returned {len(paras)} paragraphs for "
                    f"{len(ordered)} selected shots — the first {len(ordered)} "
                    "were used in order")

    if not prose_by_id:
        raise SpreadParseError(
            "the generator's reply could not be divided into the selected "
            "shots — no labelled [[id]] sections, no JSON, and no paragraphs to "
            "map", raw)

    # ── assemble the result rows (structure from the request, prose from the model) ──
    segments: List[Dict[str, Any]] = []
    for sid in ordered:
        if sid not in prose_by_id:
            continue
        m = meta.get(sid, {})
        segments.append({
            "segment_id": sid,
            "operation": op_override.get(sid) or m.get("operation") or "generate",
            "prompt": prose_by_id[sid],
            "negative": neg_override.get(sid, m.get("negative", "")),
            "continuity_note": cont_by_id.get(sid, ""),
            "directions_used": dir_override.get(sid, m.get("directions_used", [])),
            "warnings": [],
        })

    seen = {s["segment_id"] for s in segments}
    missing = [sid for sid in ordered if sid not in seen]
    if missing:
        warnings.append(
            "the generator did not write " + ", ".join(repr(m) for m in missing)
            + " — those rows are unchanged")

    return {
        "segments": segments,
        "warnings": warnings,
        "invented_identity_attributes": invented,
        "missing_segments": missing,
    }


# --------------------------------------------------------------------------- #
# COORDINATION REVIEW (k121) — the half this module was missing
#
# This module already SPEAKS the joint modes to the writer (``JOINT_MODE_PLAIN``,
# ``_render_timeline``): a model shown the sentence writes prose that continues
# the previous shot instead of restarting it. That is half the job, and until now
# it was the only half. The prose came back saying "continuous" and the ROW'S
# KNOBS stayed exactly as they were — ``joint_mode`` unset, no parent link, the
# model's default frame count, a recurring character with no identity behind it.
# The operator's cinema session is the whole finding: "not a single prompt was
# created with the proper knobs turned".
#
# ``coordinate_spread`` closes it. It reads the prose THIS spread just wrote (plus
# the locked rows, which are context here exactly as they are for the writer),
# derives the knob state that prose implies, applies the ratchet-safe ones, and
# attaches a report so nothing is silent.
#
# ⚠ INVARIANT 9 IS NOT WEAKENED. What crosses between rows here is MECHANICS
# only — joint_mode, a parent pointer, a seed, a frame count, identity references
# — through ``prompt_coordination.apply_decisions``, whose writable key list does
# not contain ``prompt``. No generated row's TEXT is ever read into another row.
# --------------------------------------------------------------------------- #
def coordinate_spread(req: SpreadRequest, parsed: Dict[str, Any],
                      context: Optional[Dict[str, Any]] = None,
                      llm=None) -> Dict[str, Any]:
    """Review + set the knobs for a parsed spread reply. Returns ``parsed``.

    The review runs over the WHOLE timeline — locked rows included — for the same
    reason the writer sees the whole timeline: a segment's join is a fact about it
    and its NEIGHBOUR, and a review blind to the neighbour would set half a chain.
    Locked rows are marked ``locked`` so the review reports them but never writes
    them (they are the user's own work; see ``prompt_coordination``'s ratchet rule).

    Each result row gains a ``knobs`` object — the knob values that were actually
    APPLIED to it, ready to be merged into the row's spec — and the return value
    gains ``coordination``, the full report. Both keys are ADDITIVE: a caller that
    ignores them behaves exactly as it did before this existed.
    """
    # The reviewer is the oracle's, installed through hugpy_video.hooks (video
    # never imports the oracle); the default reviews nothing and applies nothing.
    from hugpy_video.hooks import get_prompt_coordinator
    PC = get_prompt_coordinator()

    prose = {s["segment_id"]: s.get("prompt", "")
             for s in parsed.get("segments") or []}
    rows: List[Dict[str, Any]] = []
    for s in list(req.fixed_segments) + list(req.target_segments):
        sid = s["segment_id"]
        row = dict(s)
        # A target the model actually wrote is reviewed on its NEW prose; one it
        # skipped keeps its old prompt, because that is what will render.
        row["prompt"] = prose.get(sid) or s.get("prompt", "")
        row["locked"] = bool(s.get("locked"))
        rows.append(row)

    ctx: Dict[str, Any] = dict(context or {})
    ident = req.context.get("identity_profile")
    if ident is not None:
        ctx.setdefault("identity_profiles", ident)
    if req.steering_seed is not None:
        ctx.setdefault("seed_base", req.steering_seed)

    report = PC.review(rows, notes=req.hint, context=ctx, llm=llm)
    applied_rows, applied = PC.apply_decisions(rows, report)
    by_id = {r["segment_id"]: r for r in applied_rows}
    touched: Dict[str, Dict[str, Any]] = {}
    for d in applied:
        touched.setdefault(d.segment_id, {})[d.knob] = by_id[d.segment_id][d.knob]

    for seg in parsed.get("segments") or []:
        seg["knobs"] = touched.get(seg["segment_id"], {})
    parsed["coordination"] = report.as_dict()
    return parsed


def spread_debug_payload(req: SpreadRequest) -> str:
    """The rendered brief, for logs/tests. Not sent anywhere on its own."""
    return json.dumps({
        "target_ids": list(req.target_ids),
        "steering": req.steering,
        "steering_seed": req.steering_seed,
    }, ensure_ascii=False)
