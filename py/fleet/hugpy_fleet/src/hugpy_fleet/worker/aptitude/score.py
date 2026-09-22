"""score.py — the objective half of the scorecard.

STUDIO-SPREAD-SPEC.md §3 fixes the /100 weights:

    identity + setting preservation   20   judge
    every direction incorporated      20   MECHANICAL (here)
    continuity                        20   judge
    renderable language               15   judge
    feasible motion / camera          10   judge
    no contradictions / inventions     5   MECHANICAL (here)
    concise wording                    5   MECHANICAL (here)
    clean stop + schema                5   MECHANICAL (here)

35 points are decided by code, 65 by the judge. Nothing here asks a model's
opinion, and nothing here trusts a model's self-report: `directions_used` and
`invented_identity_attributes` are recorded as honesty data and compared to
what the code actually finds, never substituted for it.
"""
from __future__ import annotations

import re

MECH_MAX = 35.0
JUDGE_MAX = 65.0

# --- invention detection ------------------------------------------------
# Attribute classes the §1e contract forbids inventing. A hit only counts if
# the case input did not already supply the term.
INVENTION_PATTERNS = {
    "age": [
        r"\b\d{1,2}\s*[-–]?\s*(?:year|yr)s?\s*[-–]?\s*old\b",
        r"\b(?:teenage[rd]?|twenty-?something|thirty-?something|forty-?something|"
        r"middle-?aged|elderly|adolescent|in (?:her|his|their) (?:twenties|thirties|"
        r"forties|fifties|sixties))\b",
        r"\b(?:young|old|older|younger|aging|ageing)\s+(?:man|woman|girl|boy|lady|"
        r"gentleman|figure|person)\b",
    ],
    "clothing": [
        r"\b(?:dress|shirt|t-?shirt|blouse|jacket|coat|trench(?:coat)?|hoodie|"
        r"jeans|trousers|slacks|skirt|suit|necktie|tie|scarf|shawl|hat|cap|beanie|"
        r"boots|heels|sneakers|trainers|gown|uniform|sweater|jumper|cardigan|"
        r"blazer|waistcoat|vest|parka|anorak|overalls|apron|robe|kimono|"
        r"turtleneck|coveralls)\b",
    ],
    "hair": [
        r"\b(?:blonde?|brunette|redhead(?:ed)?|auburn|ginger)\b",
        r"\b(?:black|brown|red|grey|gray|silver|white|blond|blonde|dark|light)\s*[-–]?\s*"
        r"haired\b",
        r"\b(?:short|long|cropped|curly|straight|wavy|braided|shaved|buzzed|"
        r"tousled|black|brown|blonde?|red|grey|gray|silver|white|dark)\s+hair\b",
        r"\bponytail|\bdreadlocks|\bbraids\b",
    ],
    "eyes": [
        r"\b(?:blue|green|brown|hazel|grey|gray|amber|dark|pale)\s*[-–]?\s*eyed\b",
        r"\b(?:blue|green|brown|hazel|grey|gray|amber)\s+eyes\b",
    ],
    "ethnicity": [
        r"\b(?:asian|caucasian|african[- ]?american|afro[- ]?\w+|hispanic|latina|"
        r"latino|latinx|european|middle[- ]eastern|south[- ]asian|east[- ]asian|"
        r"japanese|chinese|korean|indian|nordic|slavic|arab(?:ic)?|mediterranean)\b",
    ],
    "gender": [
        # only counts when the input never named a gender for the subject
        r"\b(?:she|her|hers|herself|he|him|his|himself|woman|man|girl|boy|"
        r"lady|gentleman|female|male)\b",
    ],
}

# gender is only a violation on the identity-locked trap case, where the input
# deliberately supplies none. Elsewhere it is normal prose.
GENDER_SENSITIVE_ONLY = True

_WORD = re.compile(r"[a-z0-9']+")


def _words(s: str) -> list[str]:
    return _WORD.findall((s or "").lower())


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower())


# --- 20 pts: every direction incorporated -------------------------------

def direction_hits(prompt: str, direction_keys: list[list[str]]) -> list[bool]:
    """A direction counts as incorporated if ANY of its surface forms appears.

    Substring match on the normalised prompt: the key lists already carry the
    stem forms ("push-in"/"push in"/"dolly in"), so this is deliberately
    literal — a fuzzier matcher would hand credit to models that merely used a
    nearby word.

    A key written as "!<regex>" is NEGATIVE EVIDENCE: the direction is honoured
    by the ABSENCE of that pattern. "no dialogue" is the case that needs it —
    a prompt obeys it by containing no speech, and demanding the model also say
    the words "no dialogue" would score compliance backwards.
    """
    p = _norm(prompt)
    out = []
    for group in direction_keys:
        hit = False
        for k in group:
            if k.startswith("!"):
                if not re.search(k[1:], prompt or "", re.I):
                    hit = True
            elif _norm(k) in p:
                hit = True
            if hit:
                break
        out.append(hit)
    return out


def score_directions(prompt: str, direction_keys) -> tuple[float, list[bool]]:
    hits = direction_hits(prompt, direction_keys)
    if not hits:
        return 0.0, hits
    return 20.0 * (sum(hits) / len(hits)), hits


# --- 5 pts: no contradictions / inventions ------------------------------

def find_inventions(prompt: str, banned_context: str, gender_locked: bool) -> list[dict]:
    ctx = _norm(banned_context)
    ctx_words = set(_words(banned_context))
    p = prompt or ""
    found = []
    for cls, pats in INVENTION_PATTERNS.items():
        if cls == "gender" and not gender_locked:
            continue
        for pat in pats:
            for m in re.finditer(pat, p, re.I):
                term = m.group(0)
                t = _norm(term)
                # supplied by the input? then it is preservation, not invention
                if t in ctx or all(w in ctx_words for w in _words(term)):
                    continue
                found.append({"class": cls, "term": term})
    # dedupe on (class, lowercased term)
    seen, out = set(), []
    for f in found:
        k = (f["class"], f["term"].lower())
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def score_inventions(n: int) -> float:
    if n == 0:
        return 5.0
    return max(0.0, 5.0 - 1.75 * n)


# --- 5 pts: concise wording ---------------------------------------------

BAND = (45, 140)


def score_concision(prompt: str) -> tuple[float, int]:
    n = len(_words(prompt))
    lo, hi = BAND
    if lo <= n <= hi:
        return 5.0, n
    if n < lo:
        # a 10-word "prompt" is not a shot description
        return max(0.0, 5.0 * (n / lo)), n
    over = n - hi
    return max(0.0, 5.0 - 5.0 * (over / hi)), n


# --- 5 pts: clean stop + schema -----------------------------------------

SCHEMA_KEYS = {"prompt", "directions_used", "invented_identity_attributes", "warnings"}
_RAMBLE = re.compile(
    r"(^|\n)\s*(?:#{1,6}\s|\*\s|-\s|\d+\.\s)|"
    r"\b(?:here(?:'s| is) (?:the|your)|note:|tip:|tips:|let me know|i hope this|"
    r"feel free to|explanation:|reasoning:|as an ai|certainly[!,.]|sure[!,.])",
    re.I)


def score_schema(raw_text: str, obj: dict | None, how: str,
                 think_leak: bool, finish_reason: str | None,
                 keys: set[str] | None = None) -> tuple[float, dict]:
    keys = SCHEMA_KEYS if keys is None else keys
    detail = {"json": how, "think_leak": think_leak, "finish_reason": finish_reason}
    pts = 0.0
    # 2.0 — a valid object with the contract's keys
    if obj is not None:
        pts += 1.0
        if keys.issubset(set(obj.keys())):
            pts += 1.0
            detail["schema_complete"] = True
        else:
            detail["schema_complete"] = False
            detail["missing_keys"] = sorted(keys - set(obj.keys()))
    else:
        detail["schema_complete"] = False
    # 1.0 — the whole reply WAS the object (no fence, no wrapper prose)
    if how == "exact":
        pts += 1.0
    elif how == "fenced":
        pts += 0.5
    # 1.0 — no reasoning leaked despite the no-think directive
    if not think_leak:
        pts += 1.0
    # 1.0 — no trailing chat, headings, bullets or tips around the object
    outside = raw_text
    if obj is not None and how in ("exact", "fenced"):
        outside = ""
    ramble = bool(_RAMBLE.search(outside)) if outside else False
    detail["ramble"] = ramble
    if not ramble:
        pts += 1.0
    if finish_reason not in (None, "stop"):
        detail["unclean_finish"] = finish_reason
        pts = max(0.0, pts - 0.5)
    return min(5.0, pts), detail


# --- scene case: full mechanical pass -----------------------------------

def score_scene(case: dict, obj: dict | None, how: str, resp: dict) -> dict:
    prompt = ""
    if obj is not None:
        v = obj.get("prompt")
        if isinstance(v, str):
            prompt = v
        elif isinstance(v, list):
            prompt = " ".join(str(x) for x in v)
    if not prompt:
        # no usable prompt field — score the plain text so a model that wrote a
        # good paragraph in the wrong wrapper is not zeroed on the judge half
        prompt = resp.get("text", "") if obj is None else ""

    d_pts, d_hits = score_directions(prompt, case["direction_keys"])
    inv = find_inventions(prompt, case.get("banned_context", ""),
                          gender_locked=bool(case.get("invention_trap")))
    i_pts = score_inventions(len(inv))
    c_pts, nwords = score_concision(prompt)
    s_pts, s_detail = score_schema(resp.get("text", ""), obj, how,
                                   resp.get("think_leak", False),
                                   resp.get("finish_reason"))

    claimed = obj.get("directions_used") if isinstance(obj, dict) else None
    claimed_n = len(claimed) if isinstance(claimed, list) else None
    self_inv = obj.get("invented_identity_attributes") if isinstance(obj, dict) else None

    return {
        "prompt": prompt,
        "mechanical": {
            "directions": round(d_pts, 2),
            "inventions": round(i_pts, 2),
            "concision": round(c_pts, 2),
            "schema": round(s_pts, 2),
            "total": round(d_pts + i_pts + c_pts + s_pts, 2),
        },
        "detail": {
            "direction_hits": d_hits,
            "directions_matched": sum(d_hits),
            "directions_total": len(d_hits),
            "inventions": inv,
            "words": nwords,
            "schema": s_detail,
            # honesty data — the model's own claims, never used for scoring
            "self_reported_directions_used": claimed_n,
            "self_reported_inventions": self_inv,
            "self_report_overclaims": (
                claimed_n is not None and claimed_n > sum(d_hits)),
        },
    }


# --- routing suit -------------------------------------------------------

VALID_INTENTS = {"direction", "scene_prompt", "empty", "ambiguous"}


def score_routing_one(expected: str, obj: dict | None, how: str, resp: dict,
                      partial_ok: bool) -> dict:
    got = None
    conf = None
    if isinstance(obj, dict):
        v = obj.get("intent")
        if isinstance(v, str):
            got = v.strip().lower().replace(" ", "_")
        c = obj.get("confidence")
        if isinstance(c, (int, float)):
            conf = float(c)
    correct = 1.0 if got == expected else (
        0.5 if (partial_ok and got == "ambiguous") else 0.0)
    return {
        "expected": expected, "got": got, "confidence": conf,
        "correct": correct,
        "json_ok": obj is not None and got in VALID_INTENTS,
        "json_how": how,
        "think_leak": resp.get("think_leak", False),
        "latency_s": resp.get("elapsed_s"),
    }


def score_routing_suit(rows: list[dict]) -> dict:
    """accuracy 70 + JSON validity 20 + latency 10."""
    if not rows:
        return {"total": 0.0}
    acc = sum(r["correct"] for r in rows) / len(rows)
    jsn = sum(1.0 for r in rows if r["json_ok"]) / len(rows)
    lats = [r["latency_s"] for r in rows if r.get("latency_s")]
    med = sorted(lats)[len(lats) // 2] if lats else None
    # a router is invoked on blur; 2s is free, 15s is useless
    if med is None:
        lat_pts = 0.0
    elif med <= 2.0:
        lat_pts = 10.0
    elif med >= 15.0:
        lat_pts = 0.0
    else:
        lat_pts = 10.0 * (15.0 - med) / 13.0
    return {
        "accuracy": round(acc, 4),
        "json_validity": round(jsn, 4),
        "median_latency_s": med,
        "accuracy_pts": round(70.0 * acc, 2),
        "json_pts": round(20.0 * jsn, 2),
        "latency_pts": round(lat_pts, 2),
        "total": round(70.0 * acc + 20.0 * jsn + lat_pts, 2),
        "n": len(rows),
    }


# --- negative suit ------------------------------------------------------

ARTIFACT_VOCAB = [
    "blur", "blurry", "distort", "deform", "artifact", "artefact", "watermark",
    "text", "caption", "subtitle", "logo", "jpeg", "compression", "noise",
    "grain", "flicker", "morph", "warp", "extra limb", "extra finger",
    "extra fingers", "missing finger", "mutated", "mutation", "disfigured",
    "low quality", "lowres", "low resolution", "worst quality", "bad anatomy",
    "bad hands", "duplicate", "ghosting", "smear", "banding", "oversaturat",
    "overexpos", "underexpos", "cropped", "out of frame", "frame drop",
    "stutter", "jitter", "washed out", "posteriz", "aliasing", "moire",
    "cartoon", "anime", "cgi", "render", "plastic", "waxy", "uncanny",
    "identity drift", "face drift", "inconsistent",
]
_SENTENCE = re.compile(r"[a-z]{2,}\s+(?:is|are|was|were|has|have|will|should|"
                       r"must|the|a|an)\s+[a-z]{2,}", re.I)


def score_negative(obj: dict | None, how: str, resp: dict) -> dict:
    neg = ""
    if isinstance(obj, dict):
        v = obj.get("negative")
        if isinstance(v, str):
            neg = v
        elif isinstance(v, list):
            neg = ", ".join(str(x) for x in v)
    if not neg and obj is None:
        neg = resp.get("text", "")

    terms = [t.strip() for t in neg.split(",") if t.strip()]
    n = len(terms)
    long_terms = [t for t in terms if len(_words(t)) > 5]
    sentences = bool(_SENTENCE.search(neg)) or neg.count(".") > 1

    # 40 — comma-separated list of the right shape
    if 8 <= n <= 24:
        fmt = 40.0
    elif n >= 4:
        fmt = 40.0 * min(n, 8) / 8.0 if n < 8 else 30.0
    else:
        fmt = 40.0 * n / 8.0
    if long_terms:
        fmt -= min(20.0, 5.0 * len(long_terms))
    fmt = max(0.0, fmt)

    # 30 — actually artifact/quality vocabulary, not scene prose
    low = neg.lower()
    vocab = sum(1 for v in ARTIFACT_VOCAB if v in low)
    rel = 30.0 * min(1.0, vocab / 8.0)

    # 15 — no prose
    prose = 15.0 if not sentences else 0.0

    # 15 — clean stop + schema
    sch, sdet = score_schema(resp.get("text", ""), obj, how,
                             resp.get("think_leak", False),
                             resp.get("finish_reason"), keys={"negative"})
    sch = sch * 3.0  # score_schema is on a 5-point scale here scaled to 15

    return {
        "negative": neg,
        "terms": n,
        "vocab_hits": vocab,
        "prose_detected": sentences,
        "long_terms": len(long_terms),
        "mechanical": {"format": round(fmt, 2), "relevance": round(rel, 2),
                       "no_prose": prose, "schema": round(sch, 2),
                       "total": round(fmt + rel + prose + sch, 2)},
        "schema_detail": sdet,
    }
