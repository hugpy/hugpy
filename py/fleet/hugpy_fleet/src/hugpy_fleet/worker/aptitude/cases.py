"""cases.py — the aptitude test itself.

Three suits, matching the studio roles in STUDIO-SPREAD-SPEC.md §3:

  scene    — the spread generator's real job (§1e generator contract)
  routing  — the intent router (§1d); router-class models only
  negative — mode="negative" (§1b): an exclusion list, not prose

Every scene case carries `direction_keys`: for each direction, the alternative
surface forms that count as "this direction was incorporated". They are the
mechanical half of the scorecard's 20-point "every direction incorporated".

`banned_context` is the text a case legitimately supplies — an attribute the
input already states is not an invention when the model repeats it.
"""
from __future__ import annotations

SCENE_SYSTEM = (
    "You are a cinematic prompt generator for a text-to-video studio.\n"
    "Return ONE JSON object and NOTHING else. Schema:\n"
    '{"prompt": string, "directions_used": [int], '
    '"invented_identity_attributes": [string], "warnings": [string]}\n'
    "Rules:\n"
    "- \"prompt\" is ONE chronological cinematic paragraph describing a single "
    "continuous shot. 50-130 words.\n"
    "- Incorporate EVERY direction. Put the index of each direction you used "
    "in directions_used.\n"
    "- NEVER invent identity attributes (age, gender, clothing, hair colour, "
    "eye colour, ethnicity) that the input does not already state. If you add "
    "any, you must list them in invented_identity_attributes.\n"
    "- Preserve the existing setting and any locked identity data exactly.\n"
    "- No markdown, no headings, no bullet lists, no commentary, no reasoning."
)

SCENE_CASES = [
    {
        "id": "s1_subway_platform",
        "label": "subway platform (spec §1e shape: locked identity + directions)",
        "payload": {
            "existing_context": (
                "A near-empty subway platform at night. Alex stands close to the "
                "yellow safety line, watching the tunnel mouth. Fluorescent tubes "
                "flicker overhead. A train is not yet in sight."
            ),
            "directions": [
                "make the framing wider and colder",
                "have the train arrive at the end of the shot",
                "keep Alex facing away from camera",
            ],
            "characters": [{
                "identity_id": "alex", "name": "Alex",
                "reference_asset_ids": ["identity-alex-front"],
                "locked_description": "Existing identity profile",
                "do_not_invent": ["age", "gender", "clothing", "ethnicity"],
            }],
            "requirements": ["single continuous shot", "no dialogue"],
        },
        "direction_keys": [
            ["wide", "wider", "wide-angle", "wide angle", "distant framing"],
            ["train", "carriage", "railcar", "locomotive"],
            ["away from camera", "back to camera", "facing away", "from behind",
             "back to the lens", "turned away"],
        ],
        "banned_context": "alex subway platform night yellow line tunnel fluorescent train",
        "dialogue_free": True,
    },
    {
        "id": "s2_action_no_dialogue",
        "label": "dialogue-free action",
        "payload": {
            "existing_context": (
                "A rooftop service door bursts open onto a gravel roof at dusk. "
                "A courier sprints for the far parapet carrying a sealed case."
            ),
            "directions": [
                "no dialogue at all",
                "end on the courier vaulting the parapet",
                "handheld camera that struggles to keep up",
            ],
            "characters": [],
            "requirements": ["single continuous shot", "no spoken words"],
        },
        "direction_keys": [
            # honoured by the absence of speech, not by saying the words
            ["no dialogue", "silent", "wordless", "without speech", "no speech",
             r'!["“’]|\b(?:says?|said|shouts?|whispers?|calls out|replies|'
             r'asks?|speaks?|yells?|mutters?)\b'],
            ["parapet", "vault", "vaults", "leap", "leaps", "ledge"],
            ["handheld", "hand-held", "shaky", "unsteady", "jostl", "lurch"],
        ],
        "banned_context": "rooftop service door gravel roof dusk courier sealed case parapet",
        "dialogue_free": True,
    },
    {
        "id": "s3_two_character_continuity",
        "label": "two-character continuity from a previous shot",
        "payload": {
            "existing_context": (
                "Previous shot ended with Mara handing a folded map to Idris "
                "across a diner counter; Idris had not yet opened it."
            ),
            "directions": [
                "continue directly from the map handover",
                "Idris opens the map",
                "hold both characters in the same frame",
            ],
            "characters": [
                {"identity_id": "mara", "name": "Mara",
                 "locked_description": "Existing identity profile",
                 "do_not_invent": ["age", "gender", "clothing", "ethnicity"]},
                {"identity_id": "idris", "name": "Idris",
                 "locked_description": "Existing identity profile",
                 "do_not_invent": ["age", "gender", "clothing", "ethnicity"]},
            ],
            "previous_segment": {"prompt": "Mara slides a folded map across the diner counter to Idris.",
                                 "joint_mode": "vace_extend"},
            "requirements": ["continuity with the previous shot", "single continuous shot"],
        },
        "direction_keys": [
            ["continu", "picks up", "carries on", "directly from", "resumes",
             "immediately after", "still holding", "hand", "handover",
             "slides", "just passed", "having taken"],
            ["opens the map", "unfolds", "opening the map", "opens it", "unfold"],
            ["both", "two-shot", "two shot", "same frame", "together in frame",
             "shares the frame", "in frame together"],
        ],
        "banned_context": "mara idris folded map diner counter",
        "dialogue_free": False,
    },
    {
        "id": "s4_direction_only",
        "label": "direction-only input (no scene supplied)",
        "payload": {
            "existing_context": "",
            "directions": [
                "make it feel like the last warm minute before a storm",
                "one slow push-in",
                "no people on screen",
            ],
            "characters": [],
            "requirements": ["generate the scene from the directions alone",
                             "single continuous shot"],
        },
        "direction_keys": [
            ["storm", "thunder", "squall", "tempest", "gale", "downpour"],
            ["push-in", "push in", "dolly in", "slow zoom", "creeps forward",
             "advances slowly", "slowly moves in", "eases forward"],
            ["no people", "empty", "unpeopled", "deserted", "no figures",
             "devoid of people", "uninhabited", "nobody"],
        ],
        "banned_context": "",
        "dialogue_free": True,
    },
    {
        "id": "s5_mixed_scene_and_directions",
        "label": "mixed: preserve the scene, incorporate the directions",
        "payload": {
            "existing_context": (
                "A greenhouse at midday. Rows of tomato vines run to a fogged "
                "glass wall. A watering can sits abandoned on the gravel path."
            ),
            "directions": [
                "keep the greenhouse and the watering can exactly as they are",
                "move the camera down the row instead of holding still",
                "make the light harsher",
            ],
            "characters": [],
            "requirements": ["preserve the supplied scene",
                             "single continuous shot"],
        },
        "direction_keys": [
            ["watering can"],
            ["track", "dolly", "glide", "moves down", "travels", "pushes down the row",
             "camera moves", "trucking", "steadicam"],
            ["harsh", "harsher", "glare", "hard light", "blown", "severe light",
             "unforgiving", "stark"],
        ],
        "banned_context": "greenhouse midday tomato vines fogged glass wall watering can gravel path",
        "dialogue_free": True,
    },
    {
        "id": "s6_identity_locked",
        "label": "identity-locked with do_not_invent (invention trap)",
        "payload": {
            "existing_context": "Nia waits alone in a hotel corridor.",
            "directions": [
                "she checks the room number against a keycard",
                "keep the corridor lighting unchanged",
            ],
            "characters": [{
                "identity_id": "nia", "name": "Nia",
                "reference_asset_ids": ["identity-nia-front", "identity-nia-profile"],
                "locked_description": "Existing identity profile — use the reference assets, describe nothing about her appearance",
                "do_not_invent": ["age", "gender", "clothing", "ethnicity",
                                  "hair colour", "eye colour"],
            }],
            "requirements": ["describe no physical attribute of Nia",
                             "single continuous shot"],
        },
        "direction_keys": [
            ["keycard", "key card", "room number", "door number", "card key"],
            ["lighting unchanged", "same lighting", "existing light", "unchanged light",
             "corridor light", "hallway light", "light remains", "lighting stays"],
        ],
        "banned_context": "nia hotel corridor keycard room number",
        "dialogue_free": True,
        "invention_trap": True,
    },
    {
        "id": "s7_dense_action",
        "label": "dense action (does the model over-pack one shot?)",
        "payload": {
            "existing_context": (
                "A night market alley. A thief lifts a wallet, is spotted, runs "
                "through a fish stall, knocks over a brazier, climbs a fire "
                "escape, crosses a roof and drops into a lit courtyard."
            ),
            "directions": [
                "this is ONE segment of about four seconds",
                "keep the motion achievable in a single continuous shot",
            ],
            "characters": [],
            "requirements": [
                "a four-second shot cannot contain the whole chase",
                "if the action does not fit, say so in warnings and render only "
                "the part that fits",
                "single continuous shot",
            ],
        },
        "direction_keys": [
            ["four second", "four-second", "4 second", "4-second", "brief", "short shot",
             "single beat", "one beat", "seconds"],
            ["single continuous", "one continuous", "unbroken", "single shot",
             "one take", "continuous shot"],
        ],
        "banned_context": "night market alley thief wallet fish stall brazier fire escape roof courtyard",
        "dialogue_free": True,
        "density_trap": True,
    },
    {
        "id": "s8_mood_atmosphere",
        "label": "mood / atmosphere over event",
        "payload": {
            "existing_context": (
                "An empty municipal swimming pool, drained, lit only by the "
                "skylights above it."
            ),
            "directions": [
                "atmosphere over event — almost nothing should happen",
                "let dust move in the light",
                "hold the camera nearly still",
            ],
            "characters": [],
            "requirements": ["single continuous shot", "no dialogue"],
        },
        "direction_keys": [
            ["still", "quiet", "almost nothing", "little happens", "stillness",
             "motionless", "uneventful", "hushed"],
            ["dust", "motes", "particles"],
            ["static", "locked off", "locked-off", "nearly still", "barely moves",
             "almost imperceptib", "fixed camera", "unmoving"],
        ],
        "banned_context": "municipal swimming pool drained skylights dust",
        "dialogue_free": True,
    },
    {
        "id": "s9_v2v_restyle",
        "label": "v2v restyle direction (change look, hold content)",
        "payload": {
            "existing_context": (
                "Existing footage: a cyclist coasting down a suburban street "
                "past parked cars, shot flat in daylight."
            ),
            "directions": [
                "restyle it as grainy 16mm at golden hour",
                "do not change what happens or where the cyclist goes",
                "keep the same camera move",
            ],
            "characters": [],
            "requirements": ["this is a restyle of existing footage, not a new scene",
                             "single continuous shot"],
        },
        "direction_keys": [
            ["16mm", "16 mm", "grain", "grainy", "film stock", "celluloid"],
            ["golden hour", "golden-hour", "warm low sun", "late sun", "sunset light",
             "low golden"],
            ["same camera", "unchanged camera", "identical move", "camera move is kept",
             "keeps the camera", "same move", "preserv"],
        ],
        "banned_context": "cyclist suburban street parked cars daylight footage",
        "dialogue_free": True,
    },
]
# 8 scored scene cases are required; s3 doubles as the continuity case and the
# list runs 9 so the suit covers every variation the brief asks for.

# ------------------------------------------------------------------ routing

ROUTING_SYSTEM = (
    "You classify a single text field from a video-shot editor.\n"
    "Return ONE JSON object and NOTHING else:\n"
    '{"intent": "direction"|"scene_prompt"|"empty"|"ambiguous", '
    '"confidence": number}\n'
    "Definitions:\n"
    "- scene_prompt: the text DESCRIBES a shot — what is in frame, where, who, "
    "what happens. Even a very short description is a scene_prompt.\n"
    "- direction: the text INSTRUCTS a change to an existing shot — an edit, an "
    "adjustment, a constraint. Even a long instruction is a direction.\n"
    "- empty: no usable text.\n"
    "- ambiguous: genuinely could be either.\n"
    "Never use word count to decide. No markdown, no reasoning, no commentary."
)

ROUTING_CASES = [
    # the spec's tricky pair (§1d) comes first
    ("r01", "A woman enters a red room", "scene_prompt"),
    ("r02", "Keep her wardrobe unchanged, but make the framing wider and colder", "direction"),
    ("r03", "", "empty"),
    ("r04", "Two men argue over a chessboard in a launderette at 3am", "scene_prompt"),
    ("r05", "make it slower", "direction"),
    ("r06", "Rain on a bus window, the city smeared behind it", "scene_prompt"),
    ("r07", "Push in on her hands instead of her face", "scene_prompt"),
    ("r08", "no dialogue", "direction"),
    ("r09", "A drone lifts off a frozen lake as the ice cracks beneath it", "scene_prompt"),
    ("r10", "warmer", "direction"),
    ("r11", "   ", "empty"),
    ("r12", "Remove the second character entirely and hold on the empty doorway", "direction"),
    ("r13", "An elderly beekeeper walks between hives in morning fog", "scene_prompt"),
    ("r14", "Same shot but at night", "direction"),
    ("r15", "The camera drifts through a flooded hotel lobby, furniture floating", "scene_prompt"),
    ("r16", "Less contrast, and lose the lens flare", "direction"),
    ("r17", "A cat knocks a glass off a table in a sunlit kitchen", "scene_prompt"),
    ("r18", "Hold the identity exactly as established in the previous segment", "direction"),
    ("r19", "Closer", "direction"),
    ("r20", "Fireworks over a harbour, seen from a rooftop where nobody stands", "scene_prompt"),
]
# r07 is deliberately hard: it reads as an instruction ("instead of") but
# describes what is in frame. Graded as scene_prompt; "ambiguous" earns partial
# credit — the spec wants low-confidence answers routed to ambiguous, not
# guessed.
ROUTING_PARTIAL_OK = {"r07"}

# ------------------------------------------------------------------ negative

NEGATIVE_SYSTEM = (
    "You write NEGATIVE prompts for a text-to-video model.\n"
    "A negative prompt is a flat, comma-separated list of artifact and quality "
    "exclusions — NOT prose, NOT a scene description, NOT sentences.\n"
    "Return ONE JSON object and NOTHING else:\n"
    '{"negative": "term, term, term, ..."}\n'
    "Rules: 8-20 short terms, comma separated, no full sentences, no markdown, "
    "no explanation, no reasoning."
)

NEGATIVE_CASES = [
    {"id": "n1_face_closeup",
     "prompt": "Close-up of a character's face held for four seconds, identity must stay stable."},
    {"id": "n2_hands",
     "prompt": "A pair of hands tying a knot in rope, seen from above."},
    {"id": "n3_crowd_night",
     "prompt": "A crowded night street with neon signage and moving traffic."},
    {"id": "n4_water",
     "prompt": "A boat wake breaking across dark water, drone view."},
]
