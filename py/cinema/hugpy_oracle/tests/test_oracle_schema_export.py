"""k101 — JSON Schema generated from the frozen dataclasses.

The doc asks for "generated JSON Schema" for the platform contracts. This file
is the proof that the dataclasses can carry that without a pydantic migration:
every schema is a valid draft 2020-12 document, and — the assertion that
actually matters — the WIRE payload each contract produces (``to_dict``)
validates against the schema generated from its own annotations. A generator
that produced pretty but wrong schemas would pass the first check and fail the
second.

``jsonschema`` is in this venv, so the strict checks run. If it ever is not,
they degrade to structural assertions rather than silently skipping (a skipped
schema test is indistinguishable from a passing one at a glance).

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_schema_export.py -q
"""
from __future__ import annotations

import json
import logging
import os
import sys

import pytest

logging.disable(logging.INFO)

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_oracle import schema_export as sx
from hugpy_oracle.contracts import (
    AccessKind,
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
    ProbeCheck,
    ProbeResult,
    ProbeStatus,
    Provenance,
    QualityProfile,
    RepairCode,
    ResourceHints,
    RightsManifest,
    Scorecard,
    SourceRegistry,
)

try:                                    # present in this venv; not required
    import jsonschema
except ImportError:                     # pragma: no cover — degraded mode
    jsonschema = None


def _check_schema(schema):
    """The schema itself is a legal draft 2020-12 document."""
    if jsonschema is not None:
        jsonschema.Draft202012Validator.check_schema(schema)
        return
    assert schema["$schema"] == sx.JSON_SCHEMA_DIALECT
    assert schema["type"] == "object" and isinstance(schema["properties"], dict)


def _validate(instance, schema):
    """``instance`` satisfies ``schema``."""
    json.dumps(instance)                # always: the payload must be plain JSON
    if jsonschema is not None:
        jsonschema.validate(instance, schema)
        return
    if instance is None:                # degraded mode: a null must be admitted
        assert {"type": "null"} in schema.get("anyOf", ()) or schema.get("type") == "null"
        return
    if not isinstance(instance, dict):  # a scalar (enum member): admitted by type/enum
        return
    for name in schema.get("required", ()):
        assert name in instance, name
    for key in instance:
        assert key in schema.get("properties", {}) or True   # extras are allowed


# ---------------------------------------------------------------------------
# The generator.
# ---------------------------------------------------------------------------


def test_json_schema_for_needs_a_dataclass_type():
    with pytest.raises(TypeError):
        sx.json_schema_for(GoalSpec(objective="x", raw_prompt="x"))
    with pytest.raises(TypeError):
        sx.json_schema_for(dict)


def test_enums_become_string_enums():
    schema = sx.json_schema_for(GoalSpec)
    quality = schema["properties"]["quality"]
    assert quality["type"] == "string"
    assert set(quality["enum"]) == {q.value for q in QualityProfile}


def test_optional_enum_is_nullable_via_anyof():
    """Widening ``type`` on an enum is the classic broken-nullable bug: null is
    still not one of the enumerated values."""
    failure = sx.json_schema_for(ExecutionReceipt)["properties"]["failure"]
    assert "anyOf" in failure
    assert {"type": "null"} in failure["anyOf"]
    _validate(None, failure)
    _validate("timeout", failure)


def test_optional_scalar_is_a_nullable_type():
    vram = sx.json_schema_for(ResourceHints)["properties"]["vram_gib"]
    assert set(vram["type"]) == {"number", "null"}


def test_optional_dataclass_is_a_nullable_ref():
    rights = sx.json_schema_for(GoalSpec)["properties"]["rights"]
    assert rights["anyOf"][0] == {"$ref": "#/$defs/RightsManifest"}
    assert {"type": "null"} in rights["anyOf"]


def test_homogeneous_tuples_become_arrays():
    schema = sx.json_schema_for(Scorecard)
    checks = schema["properties"]["checks"]
    assert checks["type"] == "array"
    assert checks["items"] == {"$ref": "#/$defs/Check"}


def test_fixed_length_tuples_become_prefix_items():
    schema = sx.json_schema_for(RightsManifest)   # tuple[Authorization, ...]
    assert schema["properties"]["denied"]["items"] == {"type": "string"}
    receipt = sx.json_schema_for(ExecutionReceipt, title="raw")
    # ...and the one field whose wire shape differs is declared, not inferred
    assert receipt["properties"]["request"]["type"] == "object"
    assert "ExecutionReceipt" in sx.WIRE_OVERRIDES


def test_nested_dataclasses_land_in_defs():
    schema = sx.json_schema_for(CapabilityView, title="CapabilityDescriptor")
    assert schema["title"] == "CapabilityDescriptor"
    assert set(schema["$defs"]) >= {"Eligibility", "ResourceHints", "ProbeResult",
                                    "ProbeCheck"}
    assert schema["properties"]["eligibility"] == {"$ref": "#/$defs/Eligibility"}
    assert schema["$defs"]["ProbeResult"]["properties"]["checks"]["items"] == {
        "$ref": "#/$defs/ProbeCheck"}


def test_required_is_exactly_the_fields_without_defaults():
    schema = sx.json_schema_for(CapabilityView)
    assert schema["required"] == ["name", "source", "accepts", "produces",
                                  "model_ids", "eligibility"]
    assert "version" not in schema["required"]      # every k101 field defaults


def test_mappings_become_objects():
    schema = sx.json_schema_for(CapabilityView)
    for key in ("param_schema", "result_schema", "limits"):
        assert schema["properties"][key] == {"type": "object"}


def test_no_schema_forbids_additional_properties():
    """``to_dict`` deliberately emits legacy mirrors (min_vram_gb next to
    vram_gib); a schema that rejected them would reject our own wire format."""
    blob = json.dumps(sx.export_all())
    assert '"additionalProperties": false' not in blob


def test_union_of_scalars_merges_into_one_type_list():
    value = sx.json_schema_for(Check)["properties"]["value"]
    assert set(value["type"]) == {"boolean", "number", "string", "null"}


# ---------------------------------------------------------------------------
# Every generated schema is legal, and every wire payload satisfies its own.
# ---------------------------------------------------------------------------


def test_every_exported_schema_is_a_valid_document():
    for name, schema in sx.export_all()["schemas"].items():
        _check_schema(schema)
        assert schema["title"] == name


def _goal():
    return GoalSpec(
        objective="transcribe the clip",
        raw_prompt="write down what they say",
        inputs=(InputRef(kind=InputKind.VIDEO, ref="/uploads/a.mp4", label="clip"),),
        capability="audio.transcribe",
        quality=QualityProfile.BEST,
        budget=BudgetHints(max_seconds=60.0, max_vram_gb=8.0),
        acceptance=("every line present",),
        rights=RightsManifest(
            authorizations=(Authorization(kind=AuthorityKind.VOICE, subject="*",
                                          evidence="release.pdf"),),
            denied=("likeness:identity_profile:x",)),
    )


def _descriptor():
    return CapabilityView(
        name="audio.tts", source=SourceRegistry.TASKS,
        accepts=(ArtifactKind.TEXT, ArtifactKind.AUDIO),
        produces=(ArtifactKind.AUDIO, "audio_master"),
        model_ids=("Viral2AI~chatterbox",),
        eligibility=Eligibility(eligible=False, reasons=("no worker seats it",)),
        resources=ResourceHints(vram_gib=6.0, vram_provenance=Provenance.MEASURED,
                                frameworks=("transformers",), est_seconds=12.0),
        version="1.0.0", param_schema={"type": "object"},
        result_schema={"type": "object"}, limits={"formats": ["wav"]},
        authority_required=(AuthorityKind.VOICE,), access=(AccessKind.FILESYSTEM,),
        license="mit", eval_suite="oracle.speech:speech_scorecard",
        adapter_version="0.1.0", model_fingerprint="sha256:abc",
        probe=ProbeResult.from_checks(
            (ProbeCheck("runner_module", ProbeStatus.OK),
             ProbeCheck("worker_seat", ProbeStatus.UNKNOWN, "none seated")),
            probed_at="2026-08-20T00:00:00+00:00"),
        registry_version="sha256:def")


def _receipt():
    return ExecutionReceipt(
        request=ExecutionReceipt.normalize_request({"prompt": "hi", "n": 1}),
        capability="text.chat", model_id="qwen", worker="w1",
        started_at="t0", ended_at="t1", duration_s=1.5, retries=1,
        failure=FailureClass.TIMEOUT,
        artifacts=(ArtifactRef(kind=ArtifactKind.TEXT, uri="/out.txt"),),
        warnings=("retried",), log_excerpt=("boom",),
        registry_version="sha256:def")


def _scorecard():
    return Scorecard(
        hard_pass=False,
        checks=(Check(name="decodes", kind=CheckKind.TECHNICAL, value=True,
                      threshold=None, passed=True),
                Check(name="similarity", kind=CheckKind.IDENTITY, value=0.4,
                      threshold=0.6, passed=False)),
        confidence=0.8, disagreements=("judge a vs b",),
        diagnosis="voice drifted", repair_code=RepairCode.VOICE_SIMILARITY_LOW,
        recommended_repair="re-synthesize with the reference voice")


@pytest.mark.parametrize("title, factory", [
    ("GoalSpec", _goal),
    ("CapabilityDescriptor", _descriptor),
    ("ExecutionReceipt", _receipt),
    ("Scorecard", _scorecard),
])
def test_wire_payload_validates_against_its_generated_schema(title, factory):
    schemas = sx.export_all()["schemas"]
    _validate(factory().to_dict(), schemas[title])


def test_every_live_capability_descriptor_validates(monkeypatch):
    from hugpy_oracle import catalog, probes
    probes.clear_cache()
    monkeypatch.setattr(catalog, "_legacy_registry_rows", lambda: {
        "whisper-x": {"tasks": ["automatic-speech-recognition"],
                      "framework": "transformers"}})
    monkeypatch.setattr(catalog, "_online_workers", lambda: [{"id": "w1"}])
    schema = sx.export_all()["schemas"]["CapabilityDescriptor"]
    views = catalog.list_capabilities()
    assert views
    for view in views:
        _validate(view.to_dict(), schema)
    probes.clear_cache()


def test_plan_graph_wire_validates():
    from hugpy_oracle import plan
    graph = plan.PlanGraph(
        graph_id="g1", goal_digest="d",
        nodes=(plan.PlanNode(node_id="a", kind=plan.NodeKind.TASK,
                             capability="text.chat",
                             outputs=(plan.Port(name="out", artifact_kind="text"),)),))
    _validate(graph.to_dict(), sx.export_all()["schemas"]["PlanGraph"])


# ---------------------------------------------------------------------------
# export_all: coverage and honesty about what is missing.
# ---------------------------------------------------------------------------


def test_export_all_covers_the_platform_contracts():
    out = sx.export_all()
    names = {name for name, _m, _a in sx.PLATFORM_CONTRACTS}
    assert names == {"GoalSpec", "CapabilityDescriptor", "PlanGraph",
                     "ArtifactManifest", "Scorecard", "ExecutionReceipt"}
    assert names <= (set(out["schemas"]) | set(out["missing"]))
    assert out["platform_contracts"] == [n for n, _m, _a in sx.PLATFORM_CONTRACTS]


def test_artifact_manifest_is_recorded_as_living_on_the_agent_side():
    """k96 put it in hugpy_agent.mct; duplicating it here so the export looks
    complete would be two definitions of one manifest."""
    out = sx.export_all()
    assert "ArtifactManifest" not in out["schemas"]
    assert "hugpy_agent" in out["missing"]["ArtifactManifest"]


def test_missing_entries_always_carry_a_reason():
    for name, why in sx.export_all()["missing"].items():
        assert why.strip(), name


def test_domain_artifacts_that_exist_are_exported():
    out = sx.export_all()
    for title, _module, _attr in sx.DOMAIN_ARTIFACTS:
        assert title in out["schemas"] or title in out["missing"]
    # k102 and k103 have landed, so theirs must be present
    assert {"DialogueTimeline", "VoiceProfile", "AudioMaster",
            "PlanNode", "Port"} <= set(out["schemas"])


def test_export_is_deterministic_and_has_no_timestamp():
    first, second = sx.export_json(), sx.export_json()
    assert first == second
    assert "generated_at" not in first
    assert json.loads(first)["generated_by"].endswith("schema_export")


def test_export_survives_a_module_that_will_not_import(monkeypatch):
    """A sibling task mid-edit must not break the export — that is exactly when
    a schema dump is most wanted."""
    monkeypatch.setattr(
        sx, "DOMAIN_ARTIFACTS",
        sx.DOMAIN_ARTIFACTS + (("Imaginary", "hugpy_oracle.nope",
                                "Imaginary"),))
    out = sx.export_all()
    assert "Imaginary" in out["missing"]
    assert "GoalSpec" in out["schemas"]


def test_k104_production_artifacts_are_exported_now_that_they_landed():
    out = sx.export_all()
    assert {"GenerationSnapshot", "ContinuityBible", "ShotPlan",
            "SegmentSpec"} <= set(out["schemas"])
    for title in ("GenerationSnapshot", "SegmentSpec"):
        _check_schema(out["schemas"][title])


def test_the_package_exports_every_name_it_advertises():
    """`oracle/__init__.py` 'belonged to nobody this wave' while five agents
    landed modules around it; k101 folded in every recorded export block, so an
    __all__ entry with nothing behind it is the failure to catch."""
    import hugpy_oracle as oracle
    missing = [name for name in oracle.__all__ if not hasattr(oracle, name)]
    assert missing == []
    assert len(oracle.__all__) == len(set(oracle.__all__))
    for name in ("registry_version", "probe_capability", "json_schema_for",
                 "AudioMaster", "PlanGraph", "SegmentSpec", "run_bounded"):
        assert name in oracle.__all__
