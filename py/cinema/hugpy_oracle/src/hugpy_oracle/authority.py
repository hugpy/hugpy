"""Oracle authority gate (k97): what a route NEEDS permission for, and whether
the request brought it.

Architecture invariant 7 — authority is typed — and §7 Stage 1: "For a real
person's likeness or voice, authorization is a hard typed gate." This module is
that gate's decision half. It is DETERMINISTIC and offline: a table plus string
scanning over the ``GoalSpec``. No model call, no catalog read, no I/O — so it
can run BEFORE the router picks a model, which is the whole point (an
unauthorized request must never reach a worker, and must never quietly reach a
lesser model either).

Two rules produce a requirement:

  1. CAPABILITY — a capability that is identity-conditioned by construction
     (``IDENTITY_CONDITIONED``) always needs ``likeness``; a voice-cloning /
     reference-conditioned speech capability (``VOICE_CONDITIONED``) always
     needs ``voice``; ``audio.tts`` needs ``voice`` only when the request
     actually carries a reference voice (a plain synthetic-voice TTS does not).
  2. REQUEST — any ``identity_profile:<slug>`` or ``voice_profile:<slug>``
     reference anywhere in the request (an input ref, an input label, the raw
     prompt, the objective) needs ``likeness`` / ``voice`` for that subject,
     WHATEVER capability was asked for. Naming an identity is the trigger; the
     capability cannot launder it.

``CAPABILITY_ACCESS`` is the host-authority hook (filesystem / network / shell
/ disclosure) for callers that have no descriptor. It is still EMPTY and that is
still honest — see "DECLARED IS NOT YET ENFORCED" below.

RULE 3, added by k101: THE DESCRIPTOR. ``required_authorities`` takes an
optional ``CapabilityView``; when one is supplied its declared
``authority_required`` and ``access`` are folded in FIRST, then rules 1 and 2
run exactly as before. A descriptor may ADD a requirement, never remove one: an
adapter that declares "I need nothing" must not be able to declare its way out
of a gate the name table closes. The two agree by construction today —
``catalog.capability_authority`` derives the descriptor's declaration FROM the
tables below, so there is one source of truth with two readers rather than two
tables drifting apart.

DECLARED IS NOT YET ENFORCED — read this before "filling in the empty table".
The catalog now DECLARES host access on descriptors (``web.fetch`` reaches the
network; ``doc.extract`` reads an operator-named path). Turning those
declarations into a typed GATE means every ``web.fetch`` request without a
``RightsManifest`` granting ``network`` is refused — and there is no operator
path to file such a grant today (no route, no UI; k122's controlled acquisition
and k113's disclosure gate are the tasks that build one). So the router
deliberately calls this module WITHOUT a view, the name table stays empty, and
the wiring is proven by tests instead of by breaking live routes. Whoever adds
the grant path flips both switches in one change: populate
``CAPABILITY_ACCESS`` (or pass the view) AND ship the way to say yes.

THE FALLBACK (k113; IDEA_PHASE/POLICY-rights-consent-disclosure.md §2). When
the ONLY thing missing is likeness/voice authorization — no host authority, no
explicit denial — the decision still refuses the request AS POSED (``ok`` is
False; the identity-conditioned route must not run) but carries a typed
``FallbackDescriptor``: the traits-only, non-identifying substitute the doc
asks for ("derive only a non-identifying performance description … and use a
licensed synthetic voice"). The fallback is OFFERED on the receipt and the
scorecard, never silently APPLIED: ``apply_fallback`` builds the redacted
GoalSpec (no ``identity_profile:``/``voice_profile:`` ref, no reference voice
input, substitute capability) and the caller re-submits it explicitly. A
quiet downgrade to a lookalike would be a worse lie than a precise refusal,
and the policy forbids it (§2.4).

THE PLANNER-MODE GATE (k113; policy §3). A frontier-bound capability
(``plan.is_frontier_capability``) under ``planner_mode=local_only`` — or on a
fleet where ``HUGPY_FRONTIER_ENABLED`` is unset — is refused here, before any
route is learned, as missing ``network`` authority for that capability. No
RightsManifest can grant it: the mode is the gate, not the manifest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Iterable

from hugpy_oracle.contracts import (
    AccessKind,
    AuthorityKind,
    CapabilityView,
    Check,
    CheckKind,
    ExecutionReceipt,
    FailureClass,
    GoalSpec,
    InputRef,
    RepairCode,
    Scorecard,
)
from hugpy_oracle.plan import FRONTIER_ENABLED_ENV, frontier_enabled, is_frontier_capability

# ---------------------------------------------------------------------------
# The tables.
# ---------------------------------------------------------------------------

# Capabilities that condition generation on a specific person's appearance.
# ``video.generate.id_lock`` is the one that EXISTS today (studio Wan-VACE
# reference-to-video, driven by identity profiles); the other two are the names
# the architecture doc uses for capabilities Wave 2/3 will register — listed now
# so the gate is already closed on the day they land, never opened afterwards.
IDENTITY_CONDITIONED: tuple[str, ...] = (
    "video.generate.id_lock",
    "video.generate.identity_conditioned",
    "image.identity_reference_pack",
)

# Capabilities that reproduce a specific person's voice by construction.
VOICE_CONDITIONED: tuple[str, ...] = (
    "voice.synthesize.reference_conditioned",
    "audio.voice_convert",
)

# Capabilities that CAN be voice-cloning but are not inherently: they need
# ``voice`` only when the request supplies a reference voice to imitate.
VOICE_ON_REFERENCE: tuple[str, ...] = (
    "audio.tts",
)

# Host-authority requirements for callers that hand this module a capability
# NAME and nothing else. Still EMPTY, deliberately — the descriptor now carries
# the declarations (catalog.CAPABILITY_ACCESS_DECL) and rule 3 reads them when a
# view is supplied, but ENFORCING them fleet-wide needs an operator path to
# grant network/filesystem authority, which does not exist yet. See the module
# docstring: "declared is not yet enforced". Populating this table without that
# path refuses every web.fetch on the fleet with no way to say yes.
CAPABILITY_ACCESS: dict[str, tuple[AuthorityKind, ...]] = {}

# How a descriptor's declared host ACCESS becomes a typed AUTHORITY requirement
# (rule 3). ``external`` maps to ``network`` because reaching a third-party
# service IS reaching the network; whether the RESULT may be disclosed is a
# separate axis (AuthorityKind.DISCLOSURE) that a capability cannot declare on
# its own behalf — the request's scope decides that.
ACCESS_AUTHORITY: dict[AccessKind, AuthorityKind] = {
    AccessKind.NETWORK: AuthorityKind.NETWORK,
    AccessKind.FILESYSTEM: AuthorityKind.FILESYSTEM,
    AccessKind.SHELL: AuthorityKind.SHELL,
    AccessKind.EXTERNAL: AuthorityKind.NETWORK,
}

# The canonical cross-reference forms. ``identity_profile:<slug>`` is the string
# the video routes already accept (video_routes._reference_images_from_body).
_SUBJECT_REF = re.compile(
    r"\b(identity_profile|voice_profile):([A-Za-z0-9][A-Za-z0-9._-]*)")

# What the required subject is when a capability is identity/voice-conditioned
# but the request names nobody. It is NOT "no requirement": the capability
# reproduces SOMEONE. Only a blanket ``"*"`` grant (or naming the subject) can
# clear it.
UNNAMED_SUBJECT = "*"

# Words on an input label that mark an audio input as a voice to imitate rather
# than, say, a music bed.
_VOICE_LABEL_HINTS = ("voice", "speaker", "reference", "clone", "timbre")

# ---------------------------------------------------------------------------
# The non-identifying fallback (policy §2).
# ---------------------------------------------------------------------------

# The authority kinds a fallback can stand in for. Host authority (network,
# filesystem, shell, disclosure) has no "traits-only" equivalent — you either
# may reach the network or you may not — so any missing host kind means a hard
# refusal with no fallback.
FALLBACK_KINDS: frozenset[AuthorityKind] = frozenset(
    {AuthorityKind.LIKENESS, AuthorityKind.VOICE})

# Identity-conditioned capability -> the non-identifying capability that does
# the same JOB without a person. ``None`` means there is no such thing (voice
# conversion IS the identity) and the refusal stays hard. A capability absent
# from this table keeps its own name: only the references are stripped.
FALLBACK_CAPABILITY: dict[str, str | None] = {
    "video.generate.id_lock": "video.generate.t2v",
    "video.generate.identity_conditioned": "video.generate.t2v",
    "image.identity_reference_pack": "image.generate",
    "voice.synthesize.reference_conditioned": "audio.tts",
    "audio.voice_convert": None,
}

# The trait axes a fallback descriptor may carry — the doc's list, verbatim in
# spirit: "cadence, approximate pitch range, energy, and delivery style" for a
# voice; appearance-class words (never a name, never a reference image) for a
# likeness. Anything identifying is excluded by construction: the descriptor
# has no field for it.
LIKENESS_TRAITS: tuple[str, ...] = (
    "age_range", "build", "hair", "wardrobe", "expression", "posture")
VOICE_TRAITS: tuple[str, ...] = (
    "cadence", "pitch_range", "energy", "delivery_style", "accent_class")
GENERIC_VOICE = "licensed_synthetic"
FALLBACK_PLACEHOLDER = "[non-identifying performer]"


@dataclass(frozen=True, slots=True)
class FallbackDescriptor:
    """The traits-only substitute offered for an unauthorized likeness/voice.
    Carries WHICH subjects were stripped and WHAT may replace them; carries no
    name, no profile reference and no reference media — by type, not by
    discipline. ``capability`` is the substitute route; ``disclosure`` is the
    exact sentence that rides on the receipt / manifest (policy §2.3)."""
    stripped: tuple[tuple[AuthorityKind, str], ...]
    capability: str
    likeness_traits: tuple[str, ...] = ()
    voice_traits: tuple[str, ...] = ()
    voice: str | None = None
    disclosure: str = ""

    def __post_init__(self) -> None:
        if not self.stripped:
            raise ValueError("a FallbackDescriptor must name >=1 stripped subject")
        for kind, _subject in self.stripped:
            if kind not in FALLBACK_KINDS:
                raise ValueError(f"no non-identifying fallback exists for "
                                 f"{kind.value}")
        for text in (self.disclosure, *self.likeness_traits, *self.voice_traits):
            if _SUBJECT_REF.search(text):
                raise ValueError("a FallbackDescriptor must not carry an "
                                 "identity_profile:/voice_profile: reference")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "non_identifying",
            "stripped": [{"kind": k.value, "subject": s} for k, s in self.stripped],
            "capability": self.capability,
            "likeness_traits": list(self.likeness_traits),
            "voice_traits": list(self.voice_traits),
            "voice": self.voice,
            "disclosure": self.disclosure,
        }


def _explicitly_denied(goal: GoalSpec,
                       missing: Iterable[tuple[AuthorityKind, str]]) -> bool:
    """Did the manifest say NO (as opposed to saying nothing)? A denial is a
    person's answer and gets no workaround, traits-only or otherwise."""
    rights = goal.rights
    if rights is None or not rights.denied:
        return False
    denied = {rights._norm(d) for d in rights.denied}
    for kind, subject in missing:
        want = rights._norm(subject)
        if want in denied or f"{kind.value}:{want}" in denied:
            return True
    return False


def fallback_for(goal: GoalSpec, capability: str,
                 missing: tuple[tuple[AuthorityKind, str], ...]
                 ) -> FallbackDescriptor | None:
    """The fallback a refusal may OFFER, or None when the refusal must stay
    hard: any missing host authority, an explicit denial, or a capability with
    no non-identifying equivalent (policy §2.1-2.2)."""
    if not missing or any(k not in FALLBACK_KINDS for k, _ in missing):
        return None
    if _explicitly_denied(goal, missing):
        return None
    substitute = FALLBACK_CAPABILITY.get(capability, capability)
    if substitute is None:
        return None
    kinds = {k for k, _ in missing}
    # The disclosure line travels with the ARTIFACT (manifest, A's context), so
    # it counts subjects rather than naming them; ``stripped`` keeps the names
    # for the operator-scoped receipt.
    subjects = {s for _, s in missing if s != UNNAMED_SUBJECT}
    who = (f"{len(subjects)} named subject(s)" if subjects
           else "an unnamed subject")
    return FallbackDescriptor(
        stripped=missing, capability=substitute,
        likeness_traits=LIKENESS_TRAITS if AuthorityKind.LIKENESS in kinds else (),
        voice_traits=VOICE_TRAITS if AuthorityKind.VOICE in kinds else (),
        voice=GENERIC_VOICE if AuthorityKind.VOICE in kinds else None,
        disclosure=(f"non-identifying fallback: {', '.join(sorted(k.value for k in kinds))} "
                    f"of {who} was NOT authorized; the identity reference was "
                    f"removed and only traits may be used"
                    + (f"; voice is {GENERIC_VOICE}" if AuthorityKind.VOICE in kinds else "")
                    + f"; route {capability} -> {substitute}"),
    )


def _strip_refs(text: str) -> str:
    return _SUBJECT_REF.sub(FALLBACK_PLACEHOLDER, text) if text else text


def apply_fallback(goal: GoalSpec, fallback: FallbackDescriptor) -> GoalSpec:
    """The redacted GoalSpec the fallback describes — the ONLY way a fallback
    becomes a request, and an explicit one. Every ``identity_profile:``/
    ``voice_profile:`` reference is replaced by a placeholder, reference-voice
    inputs are dropped, the capability is the substitute, and the result
    passes ``check`` for the stripped kinds (tested). Rights ride along
    untouched: nothing is granted, nothing is needed."""
    kept: list[InputRef] = []
    for ref in goal.inputs:
        if _SUBJECT_REF.search(ref.ref) or _SUBJECT_REF.search(ref.label):
            continue
        if ref.kind.value in ("audio", "video") and any(
                h in ref.label.lower() for h in _VOICE_LABEL_HINTS):
            continue
        kept.append(ref)
    return replace(
        goal,
        objective=_strip_refs(goal.objective),
        raw_prompt=_strip_refs(goal.raw_prompt),
        inputs=tuple(kept),
        acceptance=tuple(_strip_refs(a) for a in goal.acceptance),
        capability=fallback.capability if goal.capability is not None else None,
    )


# ---------------------------------------------------------------------------
# Requirement derivation.
# ---------------------------------------------------------------------------


def _haystack(goal: GoalSpec) -> tuple[str, ...]:
    """Every place a subject reference may legitimately appear in a request."""
    parts = [goal.raw_prompt, goal.objective]
    for ref in goal.inputs:
        parts.append(ref.ref)
        parts.append(ref.label)
    parts.extend(goal.acceptance)
    return tuple(p for p in parts if p)


def find_subject_refs(*texts: str) -> tuple[tuple[str, str], ...]:
    """``((prefix, "prefix:slug"), …)`` found in the given strings, in
    first-seen order, deduplicated. Public because the HTTP layer needs the
    SAME scan when it looks up consent recorded on the referenced profiles —
    two different notions of "which identity is this request about" would be a
    gate with a hole in it."""
    seen: dict[str, tuple[str, str]] = {}
    for part in texts:
        if not part:
            continue
        for prefix, slug in _SUBJECT_REF.findall(part):
            subject = f"{prefix}:{slug}"
            seen.setdefault(subject, (prefix, subject))
    return tuple(seen.values())


def _subject_refs(goal: GoalSpec) -> tuple[tuple[str, str], ...]:
    return find_subject_refs(*_haystack(goal))


def _declared_subjects(kind: AuthorityKind, identity_subjects: list[str],
                       voice_subjects: list[str]) -> tuple[str, ...]:
    """WHO a descriptor-declared authority is about. Likeness/voice are about a
    PERSON, so they attach to whichever profile the request names; a capability
    that reproduces someone while naming nobody still needs the blanket
    subject, never "no requirement". Host-authority kinds are about the host,
    not a person, and use the capability name (set by the caller)."""
    if kind is AuthorityKind.LIKENESS:
        return tuple(identity_subjects) or (UNNAMED_SUBJECT,)
    if kind is AuthorityKind.VOICE:
        return tuple(voice_subjects or identity_subjects) or (UNNAMED_SUBJECT,)
    return (UNNAMED_SUBJECT,)


def _has_voice_reference(goal: GoalSpec) -> bool:
    """Does the request carry a voice to imitate? A ``voice_profile:`` ref, an
    ``identity_profile:`` ref (a profile IS a person), or an audio/video input
    whose label reads as a voice reference."""
    for prefix, _subject in _subject_refs(goal):
        if prefix in ("voice_profile", "identity_profile"):
            return True
    for ref in goal.inputs:
        if ref.kind.value in ("audio", "video"):
            label = ref.label.lower()
            if any(h in label for h in _VOICE_LABEL_HINTS):
                return True
    return False


def required_authorities(
    capability: str, goal: GoalSpec, view: CapabilityView | None = None,
) -> tuple[tuple[AuthorityKind, str], ...]:
    """The ``(kind, subject)`` permissions this route needs — deterministic,
    ordered, deduplicated. An empty tuple means the route needs no authority
    (``text.summarize`` on the operator's own text), which is the common case
    and must stay cheap.

    ``view`` is the capability's DESCRIPTOR (k101, rule 3). Supplying one is
    opt-in: it can only ADD requirements, and the router does not pass one
    today (module docstring: "declared is not yet enforced"). Without it the
    function is exactly what k97 shipped — offline, table-driven, no catalog
    read, safe to run before a model is picked."""
    out: list[tuple[AuthorityKind, str]] = []

    def _add(kind: AuthorityKind, subject: str) -> None:
        pair = (kind, subject)
        if pair not in out:
            out.append(pair)

    refs = _subject_refs(goal)
    identity_subjects = [s for p, s in refs if p == "identity_profile"]
    voice_subjects = [s for p, s in refs if p == "voice_profile"]

    # Rule 3 (descriptor, k101): what the capability DECLARES it needs. Runs
    # first so a descriptor-declared requirement leads the list, and so the
    # named-subject logic below can still refine it.
    if view is not None:
        for kind in view.authority_required:
            subjects = _declared_subjects(kind, identity_subjects, voice_subjects)
            for subject in subjects:
                _add(kind, subject)
        for access in view.access:
            mapped = ACCESS_AUTHORITY.get(access)
            if mapped is not None:
                _add(mapped, view.name)

    # Rule 2 (request): naming a profile is itself the trigger.
    for subject in identity_subjects:
        _add(AuthorityKind.LIKENESS, subject)
    for subject in voice_subjects:
        _add(AuthorityKind.VOICE, subject)

    # Rule 1 (capability).
    if capability in IDENTITY_CONDITIONED:
        if identity_subjects:
            for subject in identity_subjects:
                _add(AuthorityKind.LIKENESS, subject)
        else:
            _add(AuthorityKind.LIKENESS, UNNAMED_SUBJECT)
    if capability in VOICE_CONDITIONED or (
            capability in VOICE_ON_REFERENCE and _has_voice_reference(goal)):
        subjects = voice_subjects or identity_subjects
        if subjects:
            for subject in subjects:
                _add(AuthorityKind.VOICE, subject)
        else:
            _add(AuthorityKind.VOICE, UNNAMED_SUBJECT)

    # Host authority, declared per capability (empty until k101).
    for kind in CAPABILITY_ACCESS.get(capability, ()):
        _add(kind, capability)

    return tuple(out)


# ---------------------------------------------------------------------------
# The decision.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuthorityDecision:
    """Gate verdict for one (goal, capability). ``missing`` names EXACTLY which
    ``(kind, subject)`` pairs have no grant — the operator must be able to read
    the answer and know which release to go get. An ``ok`` decision with a
    missing entry (or the reverse) is incoherent and refused at construction."""
    ok: bool
    missing: tuple[tuple[AuthorityKind, str], ...] = ()
    reason: str = ""
    required: tuple[tuple[AuthorityKind, str], ...] = ()
    fallback: FallbackDescriptor | None = None

    def __post_init__(self) -> None:
        if self.ok and self.missing:
            raise ValueError(
                f"an ok AuthorityDecision cannot list missing authority: "
                f"{[(k.value, s) for k, s in self.missing]}")
        if not self.ok and not self.missing:
            raise ValueError("a refusing AuthorityDecision must name >=1 "
                             "missing (kind, subject)")
        if self.ok and self.fallback is not None:
            raise ValueError("an ok AuthorityDecision has nothing to fall back from")

    @property
    def outcome(self) -> str:
        """``authorized`` | ``fallback_offered`` | ``refused`` — the typed
        verdict the receipt and the policy talk about."""
        if self.ok:
            return "authorized"
        return "fallback_offered" if self.fallback is not None else "refused"

    @staticmethod
    def _pairs(pairs: Iterable[tuple[AuthorityKind, str]]) -> list[dict[str, str]]:
        return [{"kind": k.value, "subject": s} for k, s in pairs]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "outcome": self.outcome,
                "missing": self._pairs(self.missing),
                "reason": self.reason, "required": self._pairs(self.required),
                "fallback": self.fallback.to_dict() if self.fallback else None}


def check(goal: GoalSpec, capability: str,
          view: CapabilityView | None = None) -> AuthorityDecision:
    """The gate. Requirements come from ``required_authorities``; grants come
    ONLY from ``goal.rights``. No manifest means nothing is authorized — absence
    is never consent (§11), so a request that needs nothing still passes and a
    request that needs anything without a manifest is refused by name.

    ``view`` is passed straight through to ``required_authorities`` (rule 3)."""
    required = required_authorities(capability, goal, view)

    # Policy §3: the planner-mode gate. A frontier-bound capability needs the
    # frontier, and the MODE (plus the fleet switch) decides that — not the
    # manifest. Checked first; a RightsManifest cannot grant its way past it.
    if is_frontier_capability(capability):
        pair = (AuthorityKind.NETWORK, capability)
        if goal.planner_mode.value == "local_only":
            why = (f"{capability}: planner_mode=local_only admits zero frontier "
                   f"capability nodes")
        elif not frontier_enabled():
            why = (f"{capability}: planner_mode=frontier but {FRONTIER_ENABLED_ENV} "
                   f"is not set on this fleet — no Frontier Keeper A is wired in")
        else:
            why = ""
        if why:
            return AuthorityDecision(
                ok=False, missing=(pair,), reason=why,
                required=required if pair in required else required + (pair,))

    if not required:
        return AuthorityDecision(ok=True, reason="no typed authority required",
                                 required=())

    rights = goal.rights
    missing = tuple(
        (kind, subject) for kind, subject in required
        if rights is None or not rights.covers(kind, subject)
    )
    if not missing:
        return AuthorityDecision(
            ok=True, required=required,
            reason=("authorized by the request's RightsManifest: "
                    + ", ".join(f"{k.value} for {s}" for k, s in required)))

    named = ", ".join(f"{k.value} for {s}" for k, s in missing)
    if rights is None:
        why = (f"{capability}: no RightsManifest on the request — absence is "
               f"not consent; missing {named}")
    else:
        why = (f"{capability}: the request's RightsManifest does not cover "
               f"{named}")
    fallback = fallback_for(goal, capability, missing)
    if fallback is not None:
        why += f"; {fallback.disclosure}"
    return AuthorityDecision(ok=False, missing=missing, reason=why,
                             required=required, fallback=fallback)


# ---------------------------------------------------------------------------
# Refusal evidence — the receipt + scorecard a refused route answers with.
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def refusal_receipt(goal: GoalSpec, capability: str,
                    decision: AuthorityDecision,
                    registry_version: str | None = None) -> ExecutionReceipt:
    """The audit record for a route the gate stopped. Nothing executed, so the
    duration is 0 and there is no model — but the refusal IS an event and gets
    a receipt like every other outcome, classified ``FailureClass.REFUSED``
    (invariant 12: every request ends honestly, with evidence).

    ``registry_version`` (k101) is PASSED IN rather than read: this module is
    deliberately offline and must not acquire a catalog dependency just to
    stamp a snapshot id. A caller that already read the catalog hands over
    ``catalog.registry_version()`` (or the routed view's); one that has not
    leaves it None, which is honest — the refusal happened before any registry
    was consulted."""
    at = _now()
    excerpt: tuple[str, ...] = (decision.reason,)
    if decision.fallback is not None:
        excerpt += (decision.fallback.disclosure,)
    return ExecutionReceipt(
        request=ExecutionReceipt.normalize_request({
            "capability": capability,
            "planner_mode": goal.planner_mode.value,
            "required_authority": AuthorityDecision._pairs(decision.required),
            # Policy §2.3: the fallback is DISCLOSED on the receipt, by type.
            "authority_outcome": decision.outcome,
            "fallback": (decision.fallback.to_dict()
                         if decision.fallback is not None else None),
        }),
        capability=capability,
        model_id="",            # no model was picked — the gate came first
        worker=None,
        started_at=at, ended_at=at, duration_s=0.0,
        failure=FailureClass.REFUSED,
        log_excerpt=excerpt,
        registry_version=registry_version,
    )


def refusal_scorecard(decision: AuthorityDecision) -> Scorecard:
    """The typed verdict for a refused route: one failing authority check per
    missing grant, diagnosed ``SOURCE_AUTHORITY_MISSING``."""
    checks = tuple(
        Check(name=f"authority.{kind.value}", kind=CheckKind.TECHNICAL,
              value=False, threshold="granted", passed=False,
              detail=f"no authorization for {kind.value} of {subject!r}")
        for kind, subject in decision.missing
    )
    repair = ("supply a RightsManifest authorizing "
              + "; ".join(f"{k.value} of {s}" for k, s in decision.missing)
              + " (each Authorization needs evidence), or drop the identity "
                "reference from the request")
    if decision.fallback is not None:
        repair += (f", or accept the offered non-identifying fallback "
                   f"(authority.apply_fallback -> {decision.fallback.capability}; "
                   f"traits only, no identity reference)")
    return Scorecard(
        hard_pass=False,
        checks=checks,
        confidence=1.0,          # a missing grant is a fact, not an estimate
        diagnosis=decision.reason,
        repair_code=RepairCode.SOURCE_AUTHORITY_MISSING,
        recommended_repair=repair,
    )


__all__ = [
    "AuthorityDecision",
    "FallbackDescriptor",
    "find_subject_refs",
    "ACCESS_AUTHORITY",
    "CAPABILITY_ACCESS",
    "FALLBACK_CAPABILITY",
    "FALLBACK_KINDS",
    "GENERIC_VOICE",
    "IDENTITY_CONDITIONED",
    "LIKENESS_TRAITS",
    "UNNAMED_SUBJECT",
    "VOICE_CONDITIONED",
    "VOICE_ON_REFERENCE",
    "VOICE_TRAITS",
    "apply_fallback",
    "check",
    "fallback_for",
    "refusal_receipt",
    "refusal_scorecard",
    "required_authorities",
]
