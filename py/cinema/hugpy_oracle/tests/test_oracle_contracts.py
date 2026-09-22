"""k90a — oracle contracts: schema round-trips + the invariants that make the
contracts trustworthy (an ineligible view must explain itself, a passing
scorecard cannot carry a repair code, confidence stays in [0, 1]).

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_contracts.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_oracle.contracts import (
    ArtifactKind,
    ArtifactRef,
    Authorization,
    AuthorityKind,
    BudgetHints,
    CapabilityView,
    Check,
    CheckKind,
    Eligibility,
    ExecutionReceipt,
    FailureClass,
    GoalSpec,
    InputKind,
    InputRef,
    JudgeResult,
    PlannerMode,
    QualityProfile,
    RepairCode,
    ResourceHints,
    RightsManifest,
    Scorecard,
    SourceRegistry,
)
from hugpy_oracle.contracts import (
    DEFAULT_CAPABILITY_VERSION,
    AccessKind,
    FrozenMap,
    ProbeCheck,
    ProbeResult,
    ProbeStatus,
    Provenance,
    coerce_artifact_kind,
)


# ---------------------------------------------------------------------------
# Round-trips: to_dict -> from_dict is lossless for every contract.
# ---------------------------------------------------------------------------


def test_goalspec_roundtrip():
    spec = GoalSpec(
        objective="transcribe the supplied clip and summarize it",
        raw_prompt="yo can you write down what they say in this and give me the gist",
        inputs=(InputRef(kind=InputKind.VIDEO, ref="/uploads/clip.mp4",
                         label="the clip"),),
        capability="audio.transcribe",
        quality=QualityProfile.BEST,
        budget=BudgetHints(max_seconds=120.0, max_vram_gb=8.0),
        acceptance=("every spoken line present", "no hallucinated words"),
    )
    again = GoalSpec.from_dict(spec.to_dict())
    assert again == spec
    # the raw prompt survives normalization verbatim
    assert again.raw_prompt == spec.raw_prompt


def test_goalspec_minimal_defaults():
    spec = GoalSpec(objective="summarize", raw_prompt="tl;dr this")
    assert spec.capability is None            # auto-capability
    assert spec.quality is QualityProfile.BALANCED
    assert GoalSpec.from_dict(spec.to_dict()) == spec


def test_capabilityview_roundtrip():
    view = CapabilityView(
        name="audio.transcribe",
        source=SourceRegistry.TASKS,
        accepts=(ArtifactKind.AUDIO, ArtifactKind.VIDEO),
        produces=(ArtifactKind.TEXT, ArtifactKind.JSON),
        model_ids=("whisper-large-v3-turbo",),
        eligibility=Eligibility(eligible=False,
                                reasons=("no online worker registered",)),
        resources=ResourceHints(min_vram_gb=4.0, frameworks=("transformers",),
                                notes="planning estimate"),
    )
    assert CapabilityView.from_dict(view.to_dict()) == view


def test_execution_receipt_roundtrip_and_request_normalization():
    req = {"prompt": "hello", "max_tokens": 64, "options": {"b": 2, "a": 1}}
    receipt = ExecutionReceipt(
        request=ExecutionReceipt.normalize_request(req),
        capability="text.chat",
        model_id="Qwen2.5-3B-Instruct-GGUF",
        worker="worker-a1",
        started_at="2026-08-05T12:00:00Z",
        ended_at="2026-08-05T12:00:03Z",
        duration_s=3.0,
        retries=1,
        failure=FailureClass.TIMEOUT,
        artifacts=(ArtifactRef(kind=ArtifactKind.TEXT, uri="/artifacts/out.txt",
                               sha256="ab" * 32),),
        warnings=("retried once",),
        log_excerpt=("worker timeout at 2.5s", "retry on worker-a1"),
    )
    again = ExecutionReceipt.from_dict(receipt.to_dict())
    assert again == receipt
    assert again.request_dict() == req
    # normalization is order-independent -> identical frozen value
    assert (ExecutionReceipt.normalize_request({"b": 1, "a": {"y": 2, "x": 1}})
            == ExecutionReceipt.normalize_request({"a": {"x": 1, "y": 2}, "b": 1}))


def test_scorecard_roundtrip():
    card = Scorecard(
        hard_pass=False,
        checks=(
            Check(name="decodes", kind=CheckKind.TECHNICAL, value=True,
                  threshold=None, passed=True),
            Check(name="duration_s", kind=CheckKind.TECHNICAL, value=1.2,
                  threshold=4.0, passed=False, detail="shot too short"),
        ),
        judge_results=(
            JudgeResult(judge="qwen-vl", verdict="fail", score=0.35,
                        rationale="requested action not visible"),
        ),
        confidence=0.7,
        disagreements=("technical pass vs judge fail on motion",),
        diagnosis="clip is 1.2s against a 4s minimum",
        repair_code=RepairCode.SHOT_TOO_SHORT,
        recommended_repair="regenerate the clip with min_frames raised",
    )
    assert Scorecard.from_dict(card.to_dict()) == card


# ---------------------------------------------------------------------------
# Invariants — structurally-invalid contracts are refused at construction.
# ---------------------------------------------------------------------------


def test_goalspec_requires_objective_and_raw_prompt():
    with pytest.raises(ValueError):
        GoalSpec(objective="", raw_prompt="x")
    with pytest.raises(ValueError):
        GoalSpec(objective="x", raw_prompt="   ")


def test_goalspec_capability_must_be_namespaced():
    with pytest.raises(ValueError):
        GoalSpec(objective="x", raw_prompt="x", capability="transcribe")


def test_budget_hints_must_be_positive():
    with pytest.raises(ValueError):
        BudgetHints(max_seconds=0)
    with pytest.raises(ValueError):
        BudgetHints(max_vram_gb=-1)


def test_ineligible_without_reasons_is_refused():
    with pytest.raises(ValueError):
        Eligibility(eligible=False, reasons=())
    # eligible with advisory reasons is fine
    Eligibility(eligible=True, reasons=("no online worker; central serves it",))


def test_capabilityview_name_must_be_namespaced_and_produce_something():
    ok = dict(source=SourceRegistry.TASKS, accepts=(ArtifactKind.TEXT,),
              produces=(ArtifactKind.TEXT,), model_ids=(),
              eligibility=Eligibility(eligible=True))
    with pytest.raises(ValueError):
        CapabilityView(name="chat", **ok)
    with pytest.raises(ValueError):
        CapabilityView(name="text.chat", **{**ok, "produces": ()})


def test_receipt_rejects_negative_duration_and_retries():
    base = dict(request=(), capability="text.chat", model_id="m", worker=None,
                started_at="t0", ended_at="t1")
    with pytest.raises(ValueError):
        ExecutionReceipt(duration_s=-0.1, **base)
    with pytest.raises(ValueError):
        ExecutionReceipt(duration_s=0.0, retries=-1, **base)


def test_scorecard_confidence_bounds():
    with pytest.raises(ValueError):
        Scorecard(hard_pass=True, confidence=1.5)
    with pytest.raises(ValueError):
        Scorecard(hard_pass=True, confidence=-0.01)


def test_scorecard_pass_cannot_carry_repair_code():
    with pytest.raises(ValueError):
        Scorecard(hard_pass=True, repair_code=RepairCode.IDENTITY_DRIFT)


def test_repair_code_vocabulary_is_complete():
    expected = {
        "identity_drift", "action_missing", "voice_similarity_low",
        "line_omitted", "shot_too_short", "lip_sync_out_of_range",
        "temporal_artifact", "intent_mismatch", "source_authority_missing",
        "decode_failed", "empty_output", "format_mismatch", "timeout",
        "worker_unavailable", "capability_gap",
        # k112/k116: spatial evidence classes (directive §17)
        "geometry_drift", "camera_path_mismatch", "collision_violation",
    }
    assert {c.value for c in RepairCode} == expected


def test_contracts_serialize_to_plain_json():
    import json
    card = Scorecard(hard_pass=True, checks=(
        Check(name="decodes", kind=CheckKind.TECHNICAL, value=True,
              threshold=None, passed=True),))
    # every to_dict must be json.dumps-able with no custom encoder
    json.dumps(card.to_dict())
    json.dumps(GoalSpec(objective="x", raw_prompt="x").to_dict())
    json.dumps(CapabilityView(
        name="text.chat", source=SourceRegistry.TASKS,
        accepts=(ArtifactKind.TEXT,), produces=(ArtifactKind.TEXT,),
        model_ids=("m",), eligibility=Eligibility(eligible=True)).to_dict())


# ---------------------------------------------------------------------------
# k97 — typed authority + truthful planner mode.
# ---------------------------------------------------------------------------


def _release(kind=AuthorityKind.LIKENESS, subject="identity_profile:mira"):
    return Authorization(kind=kind, subject=subject, scope="one 30s short",
                         evidence="release-2026-08-14.pdf",
                         granted_by="operator",
                         granted_at="2026-08-14T10:00:00+00:00")


def test_authorization_roundtrip():
    a = _release()
    assert Authorization.from_dict(a.to_dict()) == a


def test_authorization_requires_a_subject_and_evidence():
    """Consent is pointed at, never inferred (architecture §11) — an
    Authorization nobody can evidence must be impossible to construct."""
    with pytest.raises(ValueError):
        Authorization(kind=AuthorityKind.LIKENESS, subject="", evidence="x")
    with pytest.raises(ValueError):
        Authorization(kind=AuthorityKind.LIKENESS,
                      subject="identity_profile:mira", evidence="   ")


def test_rights_manifest_roundtrip_and_cover_rules():
    m = RightsManifest(
        authorizations=(_release(), _release(kind=AuthorityKind.VOICE,
                                             subject="voice_profile:mira")),
        denied=("web_source:https://example.test/scrape",),
        notes="signed 2026-08-14")
    assert RightsManifest.from_dict(m.to_dict()) == m

    assert m.covers(AuthorityKind.LIKENESS, "identity_profile:mira") is True
    assert m.covers(AuthorityKind.VOICE, "voice_profile:mira") is True
    # wrong kind, wrong subject, and case/whitespace-insensitivity
    assert m.covers(AuthorityKind.VOICE, "identity_profile:mira") is False
    assert m.covers(AuthorityKind.LIKENESS, "identity_profile:someone") is False
    assert m.covers(AuthorityKind.LIKENESS, "  IDENTITY_PROFILE:Mira ") is True


def test_rights_manifest_denial_beats_a_grant():
    m = RightsManifest(authorizations=(_release(),),
                       denied=("identity_profile:mira",))
    assert m.covers(AuthorityKind.LIKENESS, "identity_profile:mira") is False
    # kind-qualified denial is equally binding
    m2 = RightsManifest(authorizations=(_release(),),
                        denied=("likeness:identity_profile:mira",))
    assert m2.covers(AuthorityKind.LIKENESS, "identity_profile:mira") is False


def test_rights_manifest_blanket_grant_and_blanket_need():
    blanket = RightsManifest(authorizations=(_release(subject="*"),))
    assert blanket.covers(AuthorityKind.LIKENESS, "identity_profile:anyone") is True
    assert blanket.covers(AuthorityKind.LIKENESS, "*") is True
    # a SPECIFIC grant never satisfies a blanket need
    assert RightsManifest(authorizations=(_release(),)).covers(
        AuthorityKind.LIKENESS, "*") is False


def test_empty_manifest_authorizes_nothing():
    assert RightsManifest().covers(AuthorityKind.LIKENESS, "identity_profile:mira") is False


def test_goalspec_planner_mode_defaults_to_local_only():
    """Invariant 8: never imply Frontier Keeper A participated."""
    spec = GoalSpec(objective="x", raw_prompt="x")
    assert spec.planner_mode is PlannerMode.LOCAL_ONLY
    assert spec.to_dict()["planner_mode"] == "local_only"
    assert spec.rights is None            # absence is not consent
    assert spec.disclosure_scope == "operator"


def test_goalspec_roundtrip_with_rights_and_planner_mode():
    spec = GoalSpec(
        objective="animate mira saying the line",
        raw_prompt="make identity_profile:mira say the line",
        capability="video.generate.id_lock",
        planner_mode=PlannerMode.FRONTIER,
        rights=RightsManifest(authorizations=(_release(),), notes="on file"),
        disclosure_scope="frontier",
    )
    again = GoalSpec.from_dict(spec.to_dict())
    assert again == spec
    assert again.rights.covers(AuthorityKind.LIKENESS, "identity_profile:mira")


def test_goalspec_disclosure_scope_must_be_named():
    with pytest.raises(ValueError):
        GoalSpec(objective="x", raw_prompt="x", disclosure_scope="  ")


def test_authority_vocabularies_are_complete():
    assert {k.value for k in AuthorityKind} == {
        "likeness", "voice", "dialogue_source", "web_source",
        "filesystem", "network", "shell", "disclosure",
    }
    assert {m.value for m in PlannerMode} == {"frontier", "local_only"}


def test_authority_contracts_serialize_to_plain_json():
    import json
    json.dumps(_release().to_dict())
    json.dumps(RightsManifest(authorizations=(_release(),)).to_dict())
    json.dumps(GoalSpec(objective="x", raw_prompt="x",
                        rights=RightsManifest(authorizations=(_release(),))).to_dict())


# ---------------------------------------------------------------------------
# k101 — CapabilityDescriptor: additive by construction, honest by default.
# ---------------------------------------------------------------------------


def _view(**over):
    base = dict(name="audio.transcribe", source=SourceRegistry.TASKS,
                accepts=(ArtifactKind.AUDIO,), produces=(ArtifactKind.TEXT,),
                model_ids=("whisper-x",),
                eligibility=Eligibility(eligible=True))
    base.update(over)
    return CapabilityView(**base)


def test_k90a_capabilityview_constructor_still_builds():
    """Back-compat is the whole design constraint: every descriptor field is
    optional, so the k90a/k97/k98 call sites keep working untouched."""
    view = _view()
    assert view.version == DEFAULT_CAPABILITY_VERSION == "0.1.0"
    assert view.param_schema == {} and view.result_schema == {}
    assert view.limits == {}
    assert view.authority_required == () and view.access == ()
    assert view.license is None and view.eval_suite is None
    assert view.adapter_version is None and view.model_fingerprint is None
    assert view.probe is None and view.registry_version is None


def test_descriptor_roundtrip_carries_every_field():
    probe = ProbeResult.from_checks(
        (ProbeCheck("runner_module", ProbeStatus.OK),), probed_at="2026-08-20T00:00:00Z")
    view = _view(
        version="1.2.3",
        param_schema={"type": "object", "properties": {"word_timestamps": {"type": "boolean"}}},
        result_schema={"type": "object"},
        limits={"formats": ["wav"], "max_duration_s": 30.0},
        authority_required=(AuthorityKind.LIKENESS,),
        access=(AccessKind.NETWORK,),
        license="mit", eval_suite="oracle.speech:speech_scorecard",
        adapter_version="0.4.1", model_fingerprint="sha256:deadbeef",
        probe=probe, registry_version="sha256:cafe")
    again = CapabilityView.from_dict(view.to_dict())
    assert again == view
    import json
    json.dumps(view.to_dict())          # still plain JSON, no custom encoder


def test_descriptor_version_must_be_semver():
    for bad in ("1.0", "v1.0.0", "", "latest"):
        with pytest.raises(ValueError):
            _view(version=bad)
    _view(version="0.0.1")
    _view(version="2.10.3-rc.1")


def test_logical_artifact_kinds_stay_strings():
    """A plan moves artifacts the media enum will never enumerate; inventing
    enum members for them is the fabrication the doc warns about."""
    view = _view(accepts=("dialogue_timeline",), produces=("audio_master", "audio"))
    assert view.accepts == ("dialogue_timeline",)
    assert view.produces == ("audio_master", ArtifactKind.AUDIO)
    assert view.to_dict()["produces"] == ["audio_master", "audio"]
    assert CapabilityView.from_dict(view.to_dict()) == view


def test_descriptor_schemas_are_frozen_and_hashable():
    view = _view(param_schema={"properties": {"seed": {"type": "integer"}}})
    assert view.param_schema["properties"]["seed"]["type"] == "integer"
    hash(view)                                  # still usable in a set
    with pytest.raises(TypeError):
        view.param_schema["properties"] = {}    # type: ignore[index]


def test_frozen_map_reads_like_a_mapping_and_compares_to_dicts():
    fm = FrozenMap({"b": 1, "a": {"deep": [1, 2]}})
    assert list(fm) == ["a", "b"]               # key-sorted iteration
    assert fm["a"]["deep"] == (1, 2)            # recursively frozen
    assert fm.to_dict() == {"a": {"deep": [1, 2]}, "b": 1}
    assert fm == {"a": {"deep": [1, 2]}, "b": 1}
    assert hash(fm) == hash(FrozenMap({"a": {"deep": [1, 2]}, "b": 1}))


def test_frozen_map_refuses_non_json_values():
    with pytest.raises(TypeError):
        FrozenMap({"f": object()})
    with pytest.raises(TypeError):
        FrozenMap({1: "int keys are not JSON object keys"})


def test_coerce_artifact_kind_agrees_with_the_plan_module():
    """k103 wrote the same function first; two spellings of one vocabulary is
    exactly the drift this contract layer exists to stop."""
    from hugpy_oracle import plan
    for value in ("audio", "dialogue_timeline", ArtifactKind.VIDEO):
        assert coerce_artifact_kind(value) == plan.coerce_artifact_kind(value)
    with pytest.raises(ValueError):
        coerce_artifact_kind("   ")


# --- the VRAM unification (k103's three-spellings finding) ------------------


def test_resource_hints_unifies_the_vram_spelling():
    hints = ResourceHints(vram_gib=6.0, vram_provenance=Provenance.MEASURED)
    assert hints.vram_gib == 6.0
    assert hints.min_vram_gb == 6.0            # read-only compat property
    assert hints.measured is True
    assert hints.to_dict()["vram_gib"] == hints.to_dict()["min_vram_gb"] == 6.0


def test_resource_hints_accepts_the_legacy_constructor():
    legacy = ResourceHints(min_vram_gb=4.0, frameworks=("transformers",))
    assert legacy == ResourceHints(vram_gib=4.0, frameworks=("transformers",))
    assert legacy.vram_provenance is Provenance.UNKNOWN
    assert ResourceHints.from_dict(legacy.to_dict()) == legacy


def test_resource_hints_refuses_two_different_vram_numbers():
    with pytest.raises(ValueError):
        ResourceHints(vram_gib=6.0, min_vram_gb=4.0)


def test_resource_hints_provenance_needs_a_number():
    with pytest.raises(ValueError):
        ResourceHints(vram_provenance=Provenance.MEASURED)
    assert ResourceHints().vram_provenance is Provenance.UNKNOWN


def test_resource_hints_reads_a_legacy_wire_payload():
    """A stored k90a payload has no vram_gib and no provenance."""
    hints = ResourceHints.from_dict({"min_vram_gb": 8.0, "frameworks": ["wan"]})
    assert hints.vram_gib == 8.0
    assert hints.vram_provenance is Provenance.DECLARED   # a number from a row
    assert ResourceHints.from_dict({}).vram_provenance is Provenance.UNKNOWN


def test_budget_hints_carries_the_unit_spelling_alias():
    budget = BudgetHints(max_seconds=60.0, max_vram_gb=8.0)
    assert budget.max_vram_gib == budget.max_vram_gb == 8.0


def test_vram_spellings_are_numerically_one_quantity():
    """The k103 finding, closed: ResourceHints/ResourceRequest/BudgetHints all
    mean GiB and can be compared without a conversion."""
    from hugpy_oracle.plan import ResourceRequest
    need = ResourceRequest(vram_gib=6.0, gpu=True)
    have = ResourceHints(vram_gib=6.0, vram_provenance=Provenance.DECLARED)
    cap = BudgetHints(max_vram_gb=8.0)
    assert need.vram_gib == have.vram_gib < cap.max_vram_gib


# --- registration probes ----------------------------------------------------


def test_probe_result_status_is_derived_not_asserted():
    checks = (ProbeCheck("a", ProbeStatus.OK),
              ProbeCheck("b", ProbeStatus.UNKNOWN, "no worker registry"))
    assert ProbeResult.from_checks(checks).status is ProbeStatus.UNKNOWN
    checks += (ProbeCheck("c", ProbeStatus.FAIL, "module missing"),)
    assert ProbeResult.from_checks(checks).status is ProbeStatus.FAIL
    with pytest.raises(ValueError):
        ProbeResult(status=ProbeStatus.OK, checks=checks)


def test_a_probe_with_no_checks_is_unknown_never_ok():
    assert ProbeResult.from_checks(()).status is ProbeStatus.UNKNOWN
    with pytest.raises(ValueError):
        ProbeResult(status=ProbeStatus.OK, checks=())


def test_a_non_ok_check_must_explain_itself():
    for status in (ProbeStatus.FAIL, ProbeStatus.UNKNOWN):
        with pytest.raises(ValueError):
            ProbeCheck("nameless", status, "")
    ProbeCheck("fine", ProbeStatus.OK, "")      # ok needs no excuse


def test_failed_probe_makes_a_capability_ineligible_with_its_own_words():
    """Doc §3.2: an adapter that unexpectedly requires `prompt` is ineligible
    until its descriptor and probe agree."""
    probe = ProbeResult.from_checks((ProbeCheck(
        "param_agreement", ProbeStatus.FAIL,
        "adapter REQUIRES ['prompt'], absent from the descriptor"),))
    view = _view().with_probe(probe)
    assert view.eligibility.eligible is False
    assert any("prompt" in r for r in view.eligibility.reasons)
    assert view.probe is probe


def test_eligible_view_cannot_be_constructed_around_a_failed_probe():
    probe = ProbeResult.from_checks(
        (ProbeCheck("runner_module", ProbeStatus.FAIL, "not importable"),))
    with pytest.raises(ValueError):
        _view(probe=probe)


def test_unknown_probe_is_advisory_only():
    probe = ProbeResult.from_checks((ProbeCheck(
        "worker_seat", ProbeStatus.UNKNOWN, "no online worker registered"),))
    view = _view().with_probe(probe)
    assert view.eligibility.eligible is True       # unknown never refuses
    assert any("inconclusive" in r for r in view.eligibility.reasons)
    # and an ok probe adds no noise
    quiet = _view().with_probe(
        ProbeResult.from_checks((ProbeCheck("worker_seat", ProbeStatus.OK),)))
    assert quiet.eligibility == Eligibility(eligible=True)


def test_probe_result_roundtrip():
    probe = ProbeResult.from_checks(
        (ProbeCheck("runner_module", ProbeStatus.OK),
         ProbeCheck("model_license", ProbeStatus.UNKNOWN, "not recorded")),
        probed_at="2026-08-20T12:00:00+00:00")
    assert ProbeResult.from_dict(probe.to_dict()) == probe
    assert probe.reason() == ""                    # nothing failed


# --- registry_version on the receipt ---------------------------------------


def test_receipt_carries_the_registry_version():
    base = dict(request=(), capability="text.chat", model_id="m", worker=None,
                started_at="t0", ended_at="t1", duration_s=0.0)
    assert ExecutionReceipt(**base).registry_version is None   # older receipt
    receipt = ExecutionReceipt(**base, registry_version="sha256:abc123")
    assert receipt.to_dict()["registry_version"] == "sha256:abc123"
    assert ExecutionReceipt.from_dict(receipt.to_dict()) == receipt


def test_access_vocabulary_is_complete():
    assert {a.value for a in AccessKind} == {
        "network", "filesystem", "shell", "external"}
    assert {p.value for p in Provenance} == {
        "measured", "estimated", "declared", "unknown"}
    assert {s.value for s in ProbeStatus} == {"ok", "fail", "unknown"}
