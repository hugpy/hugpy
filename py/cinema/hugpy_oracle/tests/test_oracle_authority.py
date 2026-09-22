"""k97 — the typed authority gate: which permissions a route needs, whether the
request's RightsManifest brings them, and the evidence a refusal answers with.

Everything here is offline and deterministic (the gate is a table plus string
scanning over a GoalSpec — no catalog, no workers, no GPU). The identity-profile
consent accessor is exercised against a TEMP store, rebinding the module global
``IDENTITIES_HOME`` the same way tests/test_identity_profiles.py does.

Locks:
  [1] the requirement table: no authority for ordinary text work; likeness for
      an identity-conditioned capability; likeness for ANY request naming an
      identity_profile ref, whatever capability was asked for; voice for a
      reference-conditioned voice capability and for audio.tts ONLY when a
      reference voice rides along.
  [2] the decision: no manifest == nothing authorized (absence is not consent);
      a covering manifest passes; a denial beats a grant; a specific grant does
      not satisfy a blanket need.
  [3] the refusal carries typed evidence: FailureClass.REFUSED on the receipt,
      RepairCode.SOURCE_AUTHORITY_MISSING on the scorecard, one failing check
      per missing (kind, subject).
  [4] identity_profiles: absent block -> False; granted WITH evidence -> True;
      granted without evidence -> False; revoked -> False.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_authority.py -q
"""
from __future__ import annotations

import logging
import os
import sys

logging.disable(logging.INFO)  # silence the models_config registry chatter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from hugpy_oracle import authority
from hugpy_oracle.contracts import (
    Authorization,
    AuthorityKind,
    FailureClass,
    GoalSpec,
    InputKind,
    InputRef,
    RepairCode,
    RightsManifest,
)

MIRA = "identity_profile:mira"


def _goal(prompt="hello", inputs=(), rights=None, acceptance=()):
    return GoalSpec(objective=prompt, raw_prompt=prompt, inputs=tuple(inputs),
                    rights=rights, acceptance=tuple(acceptance))


def _ref(kind, ref="x", label=""):
    return InputRef(kind=InputKind(kind), ref=ref, label=label)


def _grant(kind, subject, evidence="release-2026-08-14.pdf"):
    return RightsManifest(authorizations=(
        Authorization(kind=kind, subject=subject, evidence=evidence,
                      granted_by="operator", granted_at="2026-08-14T10:00:00+00:00"),))


# ---------------------------------------------------------------------------
# [1] The requirement table.
# ---------------------------------------------------------------------------


def test_ordinary_text_work_requires_no_authority():
    assert authority.required_authorities("text.summarize", _goal("tl;dr this")) == ()
    assert authority.required_authorities("text.chat", _goal("hello there")) == ()
    assert authority.required_authorities(
        "audio.transcribe", _goal("what do they say", [_ref("audio", "/tmp/a.wav")])) == ()


def test_identity_conditioned_capability_requires_likeness():
    req = authority.required_authorities(
        "video.generate.id_lock",
        _goal("make her walk", [_ref("text", MIRA, label="identity")]))
    assert req == ((AuthorityKind.LIKENESS, MIRA),)


def test_identity_conditioned_capability_without_a_named_subject_still_gates():
    # The capability reproduces SOMEONE even when the request names nobody —
    # the requirement is the blanket one, never "no requirement".
    req = authority.required_authorities("image.identity_reference_pack",
                                         _goal("build a reference pack"))
    assert req == ((AuthorityKind.LIKENESS, authority.UNNAMED_SUBJECT),)


@pytest.mark.parametrize("capability", [
    "image.transform", "video.generate.t2v", "text.chat", "no.such.capability",
])
def test_naming_an_identity_profile_gates_any_capability(capability):
    """Rule 2: the reference is the trigger. A capability cannot launder it."""
    req = authority.required_authorities(
        capability, _goal(f"restyle {MIRA} as a knight"))
    assert req == ((AuthorityKind.LIKENESS, MIRA),)


def test_identity_ref_is_found_in_every_part_of_the_request():
    for goal in (
        _goal(f"use {MIRA}"),
        _goal("go", [_ref("text", MIRA)]),
        _goal("go", [_ref("image", "/tmp/a.png", label=MIRA)]),
        _goal("go", acceptance=(f"must look like {MIRA}",)),
    ):
        assert authority.required_authorities("text.chat", goal) == (
            (AuthorityKind.LIKENESS, MIRA),)


def test_reference_conditioned_voice_always_requires_voice():
    req = authority.required_authorities(
        "voice.synthesize.reference_conditioned", _goal("say the line"))
    assert req == ((AuthorityKind.VOICE, authority.UNNAMED_SUBJECT),)
    req = authority.required_authorities(
        "voice.synthesize.reference_conditioned",
        _goal("say the line", [_ref("text", "voice_profile:mira")]))
    assert req == ((AuthorityKind.VOICE, "voice_profile:mira"),)


def test_plain_tts_needs_nothing_but_reference_conditioned_tts_needs_voice():
    # A licensed synthetic voice is not a rights question.
    assert authority.required_authorities("audio.tts", _goal("read this out")) == ()
    # …but an audio input labelled as a voice reference is.
    req = authority.required_authorities(
        "audio.tts",
        _goal("read this in her voice",
              [_ref("audio", "/tmp/z.wav", label="reference voice")]))
    assert req == ((AuthorityKind.VOICE, authority.UNNAMED_SUBJECT),)
    # A profile reference names the subject AND gates the likeness behind it.
    req = authority.required_authorities(
        "audio.tts", _goal(f"read this as {MIRA}"))
    assert req == ((AuthorityKind.LIKENESS, MIRA), (AuthorityKind.VOICE, MIRA))


def test_capability_access_hook_is_empty_but_wired():
    """No catalog capability declares filesystem/network/shell today — the hook
    stays so k101's CapabilityDescriptor has somewhere to land."""
    assert authority.CAPABILITY_ACCESS == {}
    authority.CAPABILITY_ACCESS["web.fetch"] = (AuthorityKind.NETWORK,)
    try:
        assert authority.required_authorities("web.fetch", _goal("fetch it")) == (
            (AuthorityKind.NETWORK, "web.fetch"),)
    finally:
        authority.CAPABILITY_ACCESS.pop("web.fetch")


# ---------------------------------------------------------------------------
# [2] The decision.
# ---------------------------------------------------------------------------


def test_no_manifest_is_not_consent():
    decision = authority.check(_goal(f"animate {MIRA}"), "video.generate.id_lock")
    assert decision.ok is False
    assert decision.missing == ((AuthorityKind.LIKENESS, MIRA),)
    assert "no RightsManifest" in decision.reason
    assert MIRA in decision.reason


def test_covering_manifest_passes():
    goal = _goal(f"animate {MIRA}", rights=_grant(AuthorityKind.LIKENESS, MIRA))
    decision = authority.check(goal, "video.generate.id_lock")
    assert decision.ok is True
    assert decision.missing == ()
    assert decision.required == ((AuthorityKind.LIKENESS, MIRA),)


def test_wrong_kind_or_wrong_subject_does_not_cover():
    for rights in (_grant(AuthorityKind.VOICE, MIRA),
                   _grant(AuthorityKind.LIKENESS, "identity_profile:someone-else")):
        decision = authority.check(_goal(f"animate {MIRA}", rights=rights),
                                   "video.generate.id_lock")
        assert decision.ok is False
        assert decision.missing == ((AuthorityKind.LIKENESS, MIRA),)


def test_no_requirement_passes_with_no_manifest():
    decision = authority.check(_goal("tl;dr this"), "text.summarize")
    assert decision.ok is True and decision.missing == () and decision.required == ()


def test_denial_beats_a_grant():
    rights = RightsManifest(
        authorizations=_grant(AuthorityKind.LIKENESS, MIRA).authorizations,
        denied=(f"likeness:{MIRA}",))
    decision = authority.check(_goal(f"animate {MIRA}", rights=rights),
                               "video.generate.id_lock")
    assert decision.ok is False


def test_blanket_grant_covers_an_unnamed_subject_but_a_specific_one_does_not():
    blanket = authority.check(
        _goal("build a pack", rights=_grant(AuthorityKind.LIKENESS, "*")),
        "image.identity_reference_pack")
    assert blanket.ok is True
    specific = authority.check(
        _goal("build a pack", rights=_grant(AuthorityKind.LIKENESS, MIRA)),
        "image.identity_reference_pack")
    assert specific.ok is False
    assert specific.missing == ((AuthorityKind.LIKENESS, authority.UNNAMED_SUBJECT),)


def test_decision_shape_is_coherent_by_construction():
    with pytest.raises(ValueError):
        authority.AuthorityDecision(ok=True,
                                    missing=((AuthorityKind.LIKENESS, MIRA),))
    with pytest.raises(ValueError):
        authority.AuthorityDecision(ok=False)


def test_decision_to_dict_is_json_safe():
    import json
    decision = authority.check(_goal(f"animate {MIRA}"), "video.generate.id_lock")
    d = json.loads(json.dumps(decision.to_dict()))
    assert d["missing"] == [{"kind": "likeness", "subject": MIRA}]
    assert d["ok"] is False and d["reason"]


# ---------------------------------------------------------------------------
# [3] Refusal evidence.
# ---------------------------------------------------------------------------


def test_refusal_receipt_and_scorecard_are_typed():
    goal = _goal(f"animate {MIRA}")
    decision = authority.check(goal, "video.generate.id_lock")

    receipt = authority.refusal_receipt(goal, "video.generate.id_lock", decision)
    assert receipt.failure is FailureClass.REFUSED
    assert receipt.capability == "video.generate.id_lock"
    assert receipt.model_id == ""          # the gate came before the model pick
    assert receipt.duration_s == 0.0
    assert receipt.request_dict()["planner_mode"] == "local_only"

    card = authority.refusal_scorecard(decision)
    assert card.hard_pass is False
    assert card.repair_code is RepairCode.SOURCE_AUTHORITY_MISSING
    assert [c.name for c in card.checks] == ["authority.likeness"]
    assert MIRA in card.checks[0].detail
    assert card.recommended_repair and "evidence" in card.recommended_repair


# ---------------------------------------------------------------------------
# [4] The identity-profile consent accessor.
# ---------------------------------------------------------------------------


@pytest.fixture()
def profile_store(tmp_path, monkeypatch):
    """A temp identities store + one profile in it. Rebinding the module globals
    is the honest lever here (constants read the .env file, not os.environ) —
    the same one tests/test_identity_profiles.py uses."""
    from hugpy_video.intel import identity_profiles

    monkeypatch.setattr(identity_profiles, "IDENTITIES_HOME", str(tmp_path / "identities"))
    monkeypatch.setattr(identity_profiles, "PROJECTS_HOME", str(tmp_path / "projects"))
    src = tmp_path / "ref.png"
    src.write_bytes(b"not-really-a-png-but-the-store-does-not-decode")
    profile = identity_profiles.create_profile("Mira", [str(src)])
    return identity_profiles, profile["slug"]


def test_absent_authorization_block_is_not_authorized(profile_store):
    identity_profiles, slug = profile_store
    assert identity_profiles.get_profile(slug)["authorization"] == {}
    assert identity_profiles.profile_authorized(slug, "likeness") is False
    assert identity_profiles.profile_authorized(slug, "voice") is False


def test_granted_with_evidence_is_authorized(profile_store):
    identity_profiles, slug = profile_store
    updated = identity_profiles.set_profile_authorization(
        slug, "likeness", granted=True, evidence="release-2026-08-14.pdf")
    assert updated["authorization"]["likeness"]["granted"] is True
    assert updated["authorization"]["likeness"]["granted_at"]
    assert identity_profiles.profile_authorized(slug, "likeness") is True
    # …and only for the kind that was granted.
    assert identity_profiles.profile_authorized(slug, "voice") is False


def test_grant_without_evidence_is_refused(profile_store):
    identity_profiles, slug = profile_store
    with pytest.raises(identity_profiles.ProfileError):
        identity_profiles.set_profile_authorization(slug, "likeness", granted=True)
    with pytest.raises(identity_profiles.ProfileError):
        identity_profiles.set_profile_authorization(
            slug, "telepathy", granted=True, evidence="x")


def test_revoking_keeps_the_history_and_flips_the_answer(profile_store):
    identity_profiles, slug = profile_store
    identity_profiles.set_profile_authorization(
        slug, "likeness", granted=True, evidence="release-2026-08-14.pdf")
    updated = identity_profiles.set_profile_authorization(
        slug, "likeness", granted=False)
    assert identity_profiles.profile_authorized(slug, "likeness") is False
    # never-delete: what was once claimed stays readable
    assert updated["authorization"]["likeness"]["evidence"] == "release-2026-08-14.pdf"


def test_unknown_slug_is_never_authorized(profile_store):
    identity_profiles, _slug = profile_store
    assert identity_profiles.profile_authorized("no-such-identity") is False
    assert identity_profiles.set_profile_authorization(
        "no-such-identity", "likeness", granted=True, evidence="x") is None


# ---------------------------------------------------------------------------
# [5] k113 — the non-identifying fallback (POLICY-rights-consent-disclosure §2)
# ---------------------------------------------------------------------------


def test_unauthorized_likeness_is_refused_but_offers_a_typed_fallback():
    goal = _goal(f"a portrait of {MIRA} smiling",
                 inputs=[InputRef(kind=InputKind.TEXT, ref=MIRA, label="identity")])
    d = authority.check(goal, "video.generate.id_lock")
    assert d.ok is False                       # §2.4: the request as posed does not run
    assert d.outcome == "fallback_offered"
    fb = d.fallback
    assert fb is not None
    assert fb.capability == "video.generate.t2v"
    assert fb.stripped == ((AuthorityKind.LIKENESS, MIRA),)
    assert fb.likeness_traits == authority.LIKENESS_TRAITS
    assert fb.voice is None and fb.voice_traits == ()
    # §2.2/2.3: the disclosure never names the subject
    assert "mira" not in fb.disclosure.lower()
    assert "non-identifying fallback" in fb.disclosure
    assert d.to_dict()["fallback"]["kind"] == "non_identifying"


def test_unauthorized_voice_fallback_uses_a_licensed_synthetic_voice():
    goal = _goal("read this", inputs=[
        InputRef(kind=InputKind.AUDIO, ref="/tmp/ref.wav", label="reference voice")])
    d = authority.check(goal, "audio.tts")
    assert d.outcome == "fallback_offered"
    assert d.fallback.voice == authority.GENERIC_VOICE
    assert d.fallback.voice_traits == authority.VOICE_TRAITS
    assert d.fallback.capability == "audio.tts"   # same job, reference dropped


def test_apply_fallback_yields_a_goal_the_gate_passes():
    goal = GoalSpec(
        objective=f"clip of {MIRA}", raw_prompt=f"clip of {MIRA} walking",
        inputs=(InputRef(kind=InputKind.TEXT, ref=MIRA, label="identity"),
                InputRef(kind=InputKind.AUDIO, ref="/tmp/v.wav", label="voice clone"),
                InputRef(kind=InputKind.IMAGE, ref="/tmp/bg.png", label="background")),
        capability="video.generate.id_lock",
        acceptance=(f"looks like {MIRA}",))
    d = authority.check(goal, "video.generate.id_lock")
    redacted = authority.apply_fallback(goal, d.fallback)
    assert authority.find_subject_refs(redacted.raw_prompt, redacted.objective,
                                       *redacted.acceptance) == ()
    assert authority.FALLBACK_PLACEHOLDER in redacted.raw_prompt
    assert [i.label for i in redacted.inputs] == ["background"]
    assert redacted.capability == "video.generate.t2v"
    assert redacted.rights is goal.rights
    assert authority.check(redacted, redacted.capability).ok is True


def test_explicit_denial_gets_no_fallback():
    rights = RightsManifest(denied=(MIRA,))
    d = authority.check(_goal(f"go {MIRA}", rights=rights), "text.chat")
    assert d.ok is False and d.fallback is None and d.outcome == "refused"


def test_capability_without_a_non_identifying_equivalent_stays_hard():
    d = authority.check(_goal("convert", inputs=[
        InputRef(kind=InputKind.TEXT, ref="voice_profile:ana", label="voice")]),
        "audio.voice_convert")
    assert d.fallback is None and d.outcome == "refused"


def test_missing_host_authority_gets_no_fallback():
    from hugpy_oracle.contracts import (
        AccessKind,
        ArtifactKind,
        CapabilityView,
        Eligibility,
        SourceRegistry,
    )
    view = CapabilityView(name="web.fetch", source=SourceRegistry.TASKS,
                          accepts=(), produces=(ArtifactKind.JSON,), model_ids=(),
                          eligibility=Eligibility(eligible=True),
                          access=(AccessKind.NETWORK,))
    d = authority.check(_goal("fetch"), "web.fetch", view)
    assert d.ok is False and d.fallback is None


def test_fallback_descriptor_refuses_identity_refs_by_type():
    with pytest.raises(ValueError):
        authority.FallbackDescriptor(
            stripped=((AuthorityKind.LIKENESS, MIRA),), capability="image.generate",
            disclosure=f"stripped {MIRA}")
    with pytest.raises(ValueError):
        authority.FallbackDescriptor(
            stripped=((AuthorityKind.NETWORK, "web.fetch"),), capability="x.y")
    with pytest.raises(ValueError):
        authority.AuthorityDecision(
            ok=True, fallback=authority.FallbackDescriptor(
                stripped=((AuthorityKind.VOICE, "*"),), capability="audio.tts"))


def test_fallback_is_disclosed_on_receipt_and_scorecard():
    goal = _goal(f"portrait of {MIRA}")
    d = authority.check(goal, "image.generate")
    receipt = authority.refusal_receipt(goal, "image.generate", d)
    req = receipt.request_dict()
    assert receipt.failure is FailureClass.REFUSED
    assert req["authority_outcome"] == "fallback_offered"
    assert req["fallback"]["capability"] == "image.generate"
    assert d.fallback.disclosure in receipt.log_excerpt
    card = authority.refusal_scorecard(d)
    assert card.repair_code is RepairCode.SOURCE_AUTHORITY_MISSING
    assert "apply_fallback" in card.recommended_repair


# ---------------------------------------------------------------------------
# [6] k113 — the planner-mode gate on a frontier capability (policy §3.2)
# ---------------------------------------------------------------------------


def test_frontier_capability_under_local_only_is_refused_and_no_manifest_helps(monkeypatch):
    from hugpy_oracle.contracts import PlannerMode
    from hugpy_oracle.plan import FRONTIER_ENABLED_ENV
    monkeypatch.setenv(FRONTIER_ENABLED_ENV, "1")
    rights = RightsManifest(authorizations=(
        Authorization(kind=AuthorityKind.NETWORK, subject="*", evidence="x"),))
    goal = _goal("plan this", rights=rights)
    d = authority.check(goal, "frontier.plan")
    assert d.ok is False and d.fallback is None
    assert d.missing == ((AuthorityKind.NETWORK, "frontier.plan"),)
    assert "local_only" in d.reason
    # frontier mode + enabled fleet -> passes
    from dataclasses import replace
    ok = authority.check(replace(goal, planner_mode=PlannerMode.FRONTIER), "frontier.plan")
    assert ok.ok is True
    # frontier mode on a frontier-DISABLED fleet -> refused
    monkeypatch.delenv(FRONTIER_ENABLED_ENV)
    off = authority.check(replace(goal, planner_mode=PlannerMode.FRONTIER), "frontier.plan")
    assert off.ok is False and FRONTIER_ENABLED_ENV in off.reason
    # an ordinary capability is untouched by the gate
    assert authority.check(goal, "text.chat").ok is True
