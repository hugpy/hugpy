"""k90a — oracle catalog: mapping-table totality over BOTH registries, catalog
composition with the provider seams stubbed (no live workers / GPU / network),
and the GET /oracle/capabilities route smoke over a bare Flask app.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_catalog.py -q
"""
from __future__ import annotations

import logging
import os
import sys

logging.disable(logging.INFO)  # silence the models_config registry chatter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_oracle import catalog
from hugpy_oracle.contracts import SourceRegistry
from hugpy_video.intel.studio.enums import Capability


# ---------------------------------------------------------------------------
# Mapping-table totality: every member of both vocabularies lands in exactly
# one of (mapped, EXCLUDED-with-reason). A silent gap is the failure k90a exists
# to prevent — nothing may be un-catalogued without a recorded decision.
# ---------------------------------------------------------------------------


def test_ml_tasks_map_totally():
    from hugpy_server.app.routes.ml_routes import ML_TASKS
    for task in ML_TASKS.values():
        mapped = task in catalog.LEGACY_TASK_CAPABILITY
        excluded = task in catalog.LEGACY_TASK_EXCLUDED
        assert mapped != excluded, (
            f"ML task {task!r} must be in exactly one of "
            f"LEGACY_TASK_CAPABILITY / LEGACY_TASK_EXCLUDED "
            f"(mapped={mapped}, excluded={excluded})")


def test_legacy_tables_are_disjoint_and_exclusions_carry_reasons():
    overlap = set(catalog.LEGACY_TASK_CAPABILITY) & set(catalog.LEGACY_TASK_EXCLUDED)
    assert not overlap, f"tasks in both tables: {sorted(overlap)}"
    for task, reason in catalog.LEGACY_TASK_EXCLUDED.items():
        assert reason.strip(), f"exclusion of {task!r} carries no reason"


def test_studio_capabilities_map_totally():
    for cap in Capability:
        mapped = cap in catalog.STUDIO_CAPABILITY_NAME
        excluded = cap in catalog.STUDIO_CAPABILITY_EXCLUDED
        assert mapped != excluded, (
            f"studio Capability {cap.value!r} must be in exactly one of "
            f"STUDIO_CAPABILITY_NAME / STUDIO_CAPABILITY_EXCLUDED")
    for cap, reason in catalog.STUDIO_CAPABILITY_EXCLUDED.items():
        assert reason.strip(), f"exclusion of {cap.value!r} carries no reason"


def test_namespaced_names_unique_across_both_registries():
    legacy_names = set(catalog.LEGACY_TASK_CAPABILITY.values())
    studio_names = set(catalog.STUDIO_CAPABILITY_NAME.values())
    assert not (legacy_names & studio_names), (
        "a capability name owned by both registries would let one view "
        f"shadow the other: {sorted(legacy_names & studio_names)}")
    for name in legacy_names | studio_names:
        assert "." in name, f"capability name {name!r} is not namespaced"
    # studio names are 1:1 (no two Capabilities collapsing into one name)
    assert len(studio_names) == len(catalog.STUDIO_CAPABILITY_NAME)


def test_every_mapped_capability_has_io_kinds():
    for name in set(catalog.LEGACY_TASK_CAPABILITY.values()):
        assert name in catalog._LEGACY_IO, f"no IO kinds declared for {name}"
    for cap in catalog.STUDIO_CAPABILITY_NAME:
        assert cap in catalog._STUDIO_IO, f"no IO kinds declared for {cap.value}"


# ---------------------------------------------------------------------------
# Legacy-side composition, providers stubbed (no live workers needed).
# ---------------------------------------------------------------------------

_ROWS = {
    "whisper-x": {"tasks": ["automatic-speech-recognition"],
                  "framework": "transformers"},
    "qwen-chat": {"tasks": ["text-generation"], "framework": "gguf"},
}


def _stub_legacy(monkeypatch, rows=_ROWS, workers=None, capable=True,
                 central=True, blocked=frozenset()):
    monkeypatch.setattr(catalog, "_legacy_registry_rows", lambda: dict(rows))
    monkeypatch.setattr(catalog, "_online_workers",
                        lambda: None if workers is None else list(workers))
    monkeypatch.setattr(catalog, "_worker_task_capable",
                        lambda w, t: bool(capable))
    monkeypatch.setattr(catalog, "_central_task_available", lambda t: central)
    monkeypatch.setattr(catalog, "_blocked_model_keys", lambda: set(blocked))


def _view(views, name):
    match = [v for v in views if v.name == name]
    assert match, f"{name} missing from {[v.name for v in views]}"
    return match[0]


def test_legacy_view_happy_path(monkeypatch):
    _stub_legacy(monkeypatch, workers=[{"id": "w1"}])
    views = catalog._legacy_views()
    asr = _view(views, "audio.transcribe")
    assert asr.source is SourceRegistry.TASKS
    assert asr.model_ids == ("whisper-x",)
    assert asr.eligibility.eligible
    assert asr.resources.frameworks == ("transformers",)


def test_legacy_view_no_model_registered(monkeypatch):
    _stub_legacy(monkeypatch, rows={}, workers=[{"id": "w1"}])
    asr = _view(catalog._legacy_views(), "audio.transcribe")
    assert not asr.eligibility.eligible
    assert any("no model registered" in r for r in asr.eligibility.reasons)


def test_legacy_view_no_online_worker_and_no_central_dep(monkeypatch):
    _stub_legacy(monkeypatch, workers=[], central=False)
    asr = _view(catalog._legacy_views(), "audio.transcribe")
    assert not asr.eligibility.eligible
    joined = " | ".join(asr.eligibility.reasons)
    assert "no online worker registered" in joined
    assert "central cannot serve" in joined


def test_legacy_view_worker_denies_but_central_serves(monkeypatch):
    # affirmative worker deny + central dep present -> still eligible, with
    # the worker gap surfaced as an ADVISORY reason (explain-before-execute).
    _stub_legacy(monkeypatch, workers=[{"id": "w1"}], capable=False, central=True)
    asr = _view(catalog._legacy_views(), "audio.transcribe")
    assert asr.eligibility.eligible
    assert any("no online worker advertises" in r
               for r in asr.eligibility.reasons)


def test_legacy_view_operator_block_is_a_named_refusal(monkeypatch):
    _stub_legacy(monkeypatch, workers=[{"id": "w1"}], blocked={"whisper-x"})
    asr = _view(catalog._legacy_views(), "audio.transcribe")
    assert not asr.eligibility.eligible
    assert asr.model_ids == ()   # blocked models are not offered
    assert any("operator-blocked" in r for r in asr.eligibility.reasons)


def test_legacy_deterministic_amenities_need_no_model_or_worker(monkeypatch):
    _stub_legacy(monkeypatch, rows={}, workers=[], central=True)
    views = catalog._legacy_views()
    for name in ("doc.extract", "web.fetch"):
        view = _view(views, name)
        assert view.eligibility.eligible, (name, view.eligibility.reasons)
        assert "deterministic" in view.resources.notes
    # and the dependency probe is still a real gate
    _stub_legacy(monkeypatch, rows={}, workers=[], central=False)
    doc = _view(catalog._legacy_views(), "doc.extract")
    assert not doc.eligibility.eligible
    assert any("dependency module not importable" in r
               for r in doc.eligibility.reasons)


# ---------------------------------------------------------------------------
# Studio-side composition against the REAL studio registry (pure dataclasses +
# find_spec — no GPU, no network, no workers). The catalog must agree with the
# studio's own servability verdict: two gates, one story.
# ---------------------------------------------------------------------------


def test_studio_views_agree_with_capability_verdict():
    from hugpy_video.intel.studio.presets import capability_verdict
    views = {v.name: v for v in catalog._studio_views()}
    assert set(views) == set(catalog.STUDIO_CAPABILITY_NAME.values())
    for cap, name in catalog.STUDIO_CAPABILITY_NAME.items():
        view = views[name]
        assert view.source is SourceRegistry.STUDIO
        verdict = capability_verdict(cap)
        if not verdict.servable:
            assert not view.eligibility.eligible
            assert verdict.reason in view.eligibility.reasons
        else:
            # servable per the studio -> the catalog offers it with >=1 model
            assert view.eligibility.eligible, (name, view.eligibility.reasons)
            assert view.model_ids


def test_studio_ineligible_views_explain_themselves():
    for view in catalog._studio_views():
        if not view.eligibility.eligible:
            assert view.eligibility.reasons, f"{view.name} refuses silently"


# ---------------------------------------------------------------------------
# The unified surface.
# ---------------------------------------------------------------------------


def test_list_capabilities_unified_and_sorted(monkeypatch):
    _stub_legacy(monkeypatch, workers=[{"id": "w1"}])
    views = catalog.list_capabilities()
    names = [v.name for v in views]
    assert names == sorted(names)
    assert len(names) == len(set(names))
    sources = {v.source for v in views}
    assert sources == {SourceRegistry.STUDIO, SourceRegistry.TASKS}


def test_get_capability_and_resolve_owners(monkeypatch):
    _stub_legacy(monkeypatch, workers=[{"id": "w1"}])
    view = catalog.get_capability("audio.transcribe")
    assert view is not None and view.model_ids == ("whisper-x",)
    owners = catalog.resolve_owners("audio.transcribe")
    assert owners == (SourceRegistry.TASKS, ("whisper-x",))
    assert catalog.get_capability("no.such.capability") is None
    assert catalog.resolve_owners("no.such.capability") is None


def test_unmapped_tasks_reports_only_unknown_strings(monkeypatch):
    rows = dict(_ROWS)
    rows["weird"] = {"tasks": ["quantum-flux-sorting"], "framework": "misc"}
    monkeypatch.setattr(catalog, "_legacy_registry_rows", lambda: rows)
    assert catalog.unmapped_tasks() == ("quantum-flux-sorting",)


# ---------------------------------------------------------------------------
# Route smoke: the blueprint over a bare Flask app (the app-factory boot is
# exercised by the comms suites; here the contract is the oracle wire shape).
# ---------------------------------------------------------------------------


def _client(monkeypatch):
    from flask import Flask
    from hugpy_server.app.routes.oracle_routes import oracle_bp
    _stub_legacy(monkeypatch, workers=[{"id": "w1"}])
    app = Flask("oracle-catalog-test")
    app.register_blueprint(oracle_bp)
    return app.test_client()


def test_route_lists_capabilities_with_eligibility_reasons(monkeypatch):
    client = _client(monkeypatch)
    resp = client.get("/oracle/capabilities")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["count"] == len(body["capabilities"]) > 0
    for entry in body["capabilities"]:
        assert "." in entry["name"]
        assert entry["source"] in ("studio", "tasks")
        elig = entry["eligibility"]
        assert isinstance(elig["eligible"], bool)
        assert isinstance(elig["reasons"], list)
        if not elig["eligible"]:
            assert elig["reasons"], f"{entry['name']} refuses without a reason"


def test_route_capability_filter(monkeypatch):
    client = _client(monkeypatch)
    resp = client.get("/oracle/capabilities?capability=audio.transcribe")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["count"] == 1
    assert body["capabilities"][0]["name"] == "audio.transcribe"
    assert body["capabilities"][0]["model_ids"] == ["whisper-x"]


def test_route_unknown_capability_404s_with_known_names(monkeypatch):
    client = _client(monkeypatch)
    resp = client.get("/oracle/capabilities?capability=no.such.capability")
    assert resp.status_code == 404
    body = resp.get_json()
    assert body["ok"] is False
    assert "audio.transcribe" in body["known"]


# ---------------------------------------------------------------------------
# k98 — the SPEECH capabilities (audio.tts / word_timestamps / speaker
# similarity). Declared outside LEGACY_TASK_CAPABILITY on purpose (see the
# catalog docstring), judged on a STRICT worker seat, and honest about the fact
# that nothing on this box can synthesize speech.
# ---------------------------------------------------------------------------

_SPEECH_ROWS = {
    "whisper-x": {"tasks": ["automatic-speech-recognition"],
                  "framework": "transformers"},
    "Viral2AI~chatterbox": {"tasks": ["text-generation"],
                            "framework": "transformers",
                            "hub_id": "Viral2AI/chatterbox"},
}


def _stub_speech(monkeypatch, rows=_SPEECH_ROWS, workers=None, seated=False,
                 probe=None, wired=True, blocked=frozenset(), central=True):
    """Stub every seam the speech views read. Defaults describe THIS box: the
    adapter is registered, its backend is not importable, no worker seats it."""
    _stub_legacy(monkeypatch, rows=rows, workers=workers, central=central,
                 blocked=blocked)
    monkeypatch.setattr(catalog, "_worker_seats_task",
                        lambda w, t: bool(seated))
    monkeypatch.setattr(catalog, "_tts_runner_probe", lambda: dict(
        probe if probe is not None else {
            "importable": False, "runner_registered": True,
            "module": catalog.TTS_RUNNER_MODULE,
            "reason": "python package 'chatterbox' is not installed",
            "pip": "chatterbox-tts"}))
    monkeypatch.setattr(catalog, "_word_timestamps_wired", lambda: bool(wired))


def test_speech_capabilities_are_listed(monkeypatch):
    _stub_speech(monkeypatch, workers=[{"id": "w1"}])
    names = [v.name for v in catalog.list_capabilities()]
    for name in ("audio.tts", "audio.transcribe.word_timestamps",
                 "audio.speaker_similarity"):
        assert name in names, names
    assert names == sorted(names) and len(names) == len(set(names))


def test_speech_capabilities_declare_io_kinds():
    for name in ("audio.tts", "audio.transcribe.word_timestamps",
                 "audio.speaker_similarity"):
        assert name in catalog._SPEECH_IO, f"no IO kinds declared for {name}"
    # and they do not collide with either registry's vocabulary
    existing = (set(catalog.LEGACY_TASK_CAPABILITY.values())
                | set(catalog.STUDIO_CAPABILITY_NAME.values()))
    assert not (set(catalog._SPEECH_IO) & existing)


def test_speech_task_strings_are_known_to_unmapped_tasks(monkeypatch):
    """A row that correctly declares text-to-speech is NOT discovery noise."""
    rows = {"tts-row": {"tasks": ["text-to-speech"], "framework": "transformers"}}
    monkeypatch.setattr(catalog, "_legacy_registry_rows", lambda: rows)
    assert catalog.unmapped_tasks() == ()


def test_audio_tts_is_ineligible_here_naming_the_runner_and_the_worker(monkeypatch):
    _stub_speech(monkeypatch, workers=[{"id": "w1"}])
    view = _view(catalog._speech_views(), "audio.tts")
    assert not view.eligibility.eligible
    joined = " | ".join(view.eligibility.reasons)
    assert catalog.TTS_RUNNER_MODULE in joined       # WHICH runner
    assert "backend is unavailable here" in joined
    assert "no online worker seats" in joined        # and WHICH gap
    assert view.model_ids == ("Viral2AI~chatterbox",)
    assert view.source is SourceRegistry.TASKS


def test_audio_tts_binds_the_misclassified_chatterbox_row_and_says_so(monkeypatch):
    _stub_speech(monkeypatch, workers=[{"id": "w1"}])
    view = _view(catalog._speech_views(), "audio.tts")
    assert any("row declares tasks=['text-generation']" in r
               for r in view.eligibility.reasons)


def test_audio_tts_with_no_model_registered(monkeypatch):
    _stub_speech(monkeypatch, rows={"whisper-x": _ROWS["whisper-x"]},
                 workers=[{"id": "w1"}])
    view = _view(catalog._speech_views(), "audio.tts")
    assert not view.eligibility.eligible
    assert view.model_ids == ()
    assert any("no model registered" in r for r in view.eligibility.reasons)


def test_audio_tts_eligible_only_when_a_worker_seats_it(monkeypatch):
    _stub_speech(monkeypatch, workers=[{"id": "w1"}], seated=True)
    view = _view(catalog._speech_views(), "audio.tts")
    assert view.eligibility.eligible, view.eligibility.reasons


def test_audio_tts_worker_seat_is_strict_not_legacy_permissive():
    """The real seam: a worker that does not enumerate the task is NOT seated
    (unlike workers._task_capable, which is deliberately permissive)."""
    assert catalog._worker_seats_task({"task_capabilities": {}}, "text-to-speech") is False
    assert catalog._worker_seats_task({}, "text-to-speech") is False
    assert catalog._worker_seats_task(
        {"task_capabilities": {"text-to-speech": False}}, "text-to-speech") is False
    assert catalog._worker_seats_task(
        {"task_capabilities": {"text-to-speech": True}}, "text-to-speech") is True
    # ...whereas the legacy gate says yes to the same unknown task
    assert catalog._worker_task_capable({"task_capabilities": {}}, "text-to-speech")


def test_audio_tts_backend_on_central_substitutes_for_a_seat(monkeypatch):
    """A box that can run the backend itself needs no worker (same rule the
    legacy views apply: worker_ok OR central)."""
    _stub_speech(monkeypatch, workers=[{"id": "w1"}],
                 probe={"importable": True, "runner_registered": True,
                        "module": catalog.TTS_RUNNER_MODULE, "reason": ""})
    assert _view(catalog._speech_views(), "audio.tts").eligibility.eligible


def test_audio_tts_unregistered_adapter_refuses(monkeypatch):
    _stub_speech(monkeypatch, workers=[{"id": "w1"}], seated=True,
                 probe={"importable": False, "runner_registered": False,
                        "module": catalog.TTS_RUNNER_MODULE,
                        "reason": "runner adapter is not importable (ImportError)"})
    view = _view(catalog._speech_views(), "audio.tts")
    assert not view.eligibility.eligible
    assert any("not importable" in r for r in view.eligibility.reasons)


def test_audio_tts_refused_license_is_a_hard_gate(monkeypatch):
    rows = dict(_SPEECH_ROWS)
    rows["Viral2AI~chatterbox"] = {**rows["Viral2AI~chatterbox"],
                                   "license": "cc-by-nc-4.0"}
    _stub_speech(monkeypatch, rows=rows, workers=[{"id": "w1"}], seated=True)
    view = _view(catalog._speech_views(), "audio.tts")
    assert not view.eligibility.eligible
    assert any("refuses reference-conditioned voice" in r
               for r in view.eligibility.reasons)


def test_audio_tts_blocked_model_is_a_named_refusal(monkeypatch):
    _stub_speech(monkeypatch, workers=[{"id": "w1"}], seated=True,
                 blocked={"Viral2AI~chatterbox"})
    view = _view(catalog._speech_views(), "audio.tts")
    assert not view.eligibility.eligible and view.model_ids == ()
    assert any("operator-blocked" in r for r in view.eligibility.reasons)


def test_audio_tts_alias_resolves_to_the_same_view(monkeypatch):
    """doc §4 spells it voice.synthesize.reference_conditioned; one view, two
    names — never two views that can disagree."""
    _stub_speech(monkeypatch, workers=[{"id": "w1"}])
    assert catalog.canonical_name(
        "voice.synthesize.reference_conditioned") == "audio.tts"
    view = catalog.get_capability("voice.synthesize.reference_conditioned")
    assert view is not None and view.name == "audio.tts"
    owners = catalog.resolve_owners("voice.synthesize.reference_conditioned")
    assert owners == (SourceRegistry.TASKS, ("Viral2AI~chatterbox",))
    # the alias is NOT published as a second entry
    assert "voice.synthesize.reference_conditioned" not in [
        v.name for v in catalog.list_capabilities()]


def test_word_timestamps_tracks_audio_transcribe(monkeypatch):
    _stub_speech(monkeypatch, workers=[{"id": "w1"}], wired=True)
    views = catalog._speech_views()
    base = _view(catalog._legacy_views(), "audio.transcribe")
    wt = _view(views, "audio.transcribe.word_timestamps")
    assert base.eligibility.eligible and wt.eligibility.eligible
    assert wt.model_ids == base.model_ids       # same whisper backend

    # ...and it follows the base capability DOWN as well as up
    _stub_speech(monkeypatch, rows={}, workers=[], central=False, wired=True)
    base = _view(catalog._legacy_views(), "audio.transcribe")
    wt = _view(catalog._speech_views(), "audio.transcribe.word_timestamps")
    assert not base.eligibility.eligible and not wt.eligibility.eligible
    for reason in base.eligibility.reasons:
        assert reason in wt.eligibility.reasons


def test_word_timestamps_refuses_while_the_passthrough_is_missing(monkeypatch):
    _stub_speech(monkeypatch, workers=[{"id": "w1"}], wired=False)
    wt = _view(catalog._speech_views(), "audio.transcribe.word_timestamps")
    assert not wt.eligibility.eligible
    assert catalog.WORD_TIMESTAMPS_PASSTHROUGH_GAP in wt.eligibility.reasons
    assert "word_timestamps" in wt.resources.notes


def test_word_timestamps_passthrough_probe_is_the_real_schema():
    """The probe answers about the LIVE schema. k98 shipped it False (whisper_
    schemas.TranscribeRequest had no ``word_timestamps`` field yet); k98b
    landed the four-edit passthrough chain the field's docstring names
    (whisper_schemas.TranscribeRequest, builders._build_whisper_request,
    whisper runner forward, execute.whisper_transcribe option), so the LIVE
    answer is now yes — this probe is unstubbed, unlike
    test_word_timestamps_tracks_audio_transcribe / test_word_timestamps_
    refuses_while_the_passthrough_is_missing above, which exercise both
    states explicitly via ``_stub_speech(..., wired=...)``."""
    assert catalog._word_timestamps_wired() is True


def test_word_timestamps_declares_its_fixed_parameter():
    assert catalog.capability_params(
        "audio.transcribe.word_timestamps") == {"word_timestamps": True}
    assert catalog.capability_params("audio.transcribe") == {}
    # the accessor hands back a COPY — a caller cannot poison the table
    params = catalog.capability_params("audio.transcribe.word_timestamps")
    params["word_timestamps"] = False
    assert catalog.CAPABILITY_FIXED_PARAMS[
        "audio.transcribe.word_timestamps"]["word_timestamps"] is True


def test_speaker_similarity_is_declared_with_no_binding(monkeypatch):
    _stub_speech(monkeypatch, workers=[{"id": "w1"}])
    view = _view(catalog._speech_views(), "audio.speaker_similarity")
    assert not view.eligibility.eligible
    assert view.model_ids == ()
    assert any("no speaker-embedding model registered" in r
               for r in view.eligibility.reasons)


def test_speaker_similarity_finds_a_model_when_one_is_registered(monkeypatch):
    rows = dict(_SPEECH_ROWS)
    rows["speechbrain~ecapa-voxceleb"] = {
        "tasks": ["feature-extraction"], "framework": "transformers",
        "hub_id": "speechbrain/spkrec-ecapa-voxceleb"}
    _stub_speech(monkeypatch, rows=rows, workers=[{"id": "w1"}])
    view = _view(catalog._speech_views(), "audio.speaker_similarity")
    assert view.model_ids == ("speechbrain~ecapa-voxceleb",)
    assert not view.eligibility.eligible          # still no adapter for it
    assert any("no runner adapter is registered" in r
               for r in view.eligibility.reasons)


def test_no_speaker_embedding_model_exists_on_this_fleet():
    """The claim the refusal makes, checked against the REAL registry."""
    assert catalog._speaker_embedding_rows(catalog._legacy_registry_rows()) == ()


def test_speech_views_never_refuse_silently(monkeypatch):
    _stub_speech(monkeypatch, workers=[{"id": "w1"}])
    for view in catalog._speech_views():
        if not view.eligibility.eligible:
            assert view.eligibility.reasons, f"{view.name} refuses silently"


def test_route_lists_the_speech_capabilities_honestly(monkeypatch):
    """GET /oracle/capabilities is catalog-driven: the route file was not
    touched by k98 and the new rows must simply appear."""
    from flask import Flask
    from hugpy_server.app.routes.oracle_routes import oracle_bp
    # wired=False mirrors THIS tree: the word_timestamps passthrough is not in
    # place yet, so all three new rows must refuse with a reason.
    _stub_speech(monkeypatch, workers=[{"id": "w1"}], wired=False)
    app = Flask("oracle-speech-test")
    app.register_blueprint(oracle_bp)
    client = app.test_client()

    body = client.get("/oracle/capabilities").get_json()
    entries = {e["name"]: e for e in body["capabilities"]}
    for name in ("audio.tts", "audio.transcribe.word_timestamps",
                 "audio.speaker_similarity"):
        assert name in entries
        assert entries[name]["eligibility"]["eligible"] is False
        assert entries[name]["eligibility"]["reasons"]

    tts_entry = entries["audio.tts"]
    assert "audio" in tts_entry["produces"] and "json" in tts_entry["produces"]
    assert any("worker" in r for r in tts_entry["eligibility"]["reasons"])

    # the doc-vocabulary alias resolves through the filter too
    aliased = client.get(
        "/oracle/capabilities?capability=voice.synthesize.reference_conditioned")
    assert aliased.status_code == 200
    assert aliased.get_json()["capabilities"][0]["name"] == "audio.tts"


def test_declared_external_runner_matches_the_catalog_constant():
    """The model_resolver declaration and the catalog's fallback must name the
    SAME adapter — two tables, one runner."""
    from hugpy_engine.resolvers.model_resolver import external_runner_for
    declared = external_runner_for(catalog.TTS_TASK)
    assert declared is not None
    assert declared[0] == catalog.TTS_RUNNER_MODULE
    assert catalog._tts_runner_module_name() == catalog.TTS_RUNNER_MODULE


def test_live_tts_probe_is_registered_but_unavailable():
    """No stubs: the adapter imports here, its backend does not."""
    probed = catalog._tts_runner_probe()
    assert probed["runner_registered"] is True
    assert probed["importable"] is False
    assert "chatterbox" in probed["reason"]


# ---------------------------------------------------------------------------
# k101 — the descriptor upgrade: declared facts, a registry snapshot digest and
# registration probes, all on the same view the route already serializes.
# ---------------------------------------------------------------------------

from hugpy_oracle import probes
from hugpy_oracle.contracts import (
    AccessKind,
    AuthorityKind,
    ProbeCheck,
    ProbeResult,
    ProbeStatus,
    Provenance,
)


def _fresh(monkeypatch, **kw):
    """Stub the seams AND drop any cached probe verdict: the probe cache is
    process-global by design and must never leak between tests."""
    probes.clear_cache()
    _stub_speech(monkeypatch, **kw)


def test_every_view_declares_a_semver_descriptor_version(monkeypatch):
    _fresh(monkeypatch, workers=[{"id": "w1"}])
    for view in catalog.list_capabilities():
        assert view.version == catalog.capability_version(view.name)
        assert view.version.count(".") == 2


def test_legacy_view_reads_its_limits_off_the_row(monkeypatch):
    rows = {"qwen-chat": {"tasks": ["text-generation"], "framework": "gguf",
                          "model_max_length": 32768},
            "small-chat": {"tasks": ["text-generation"], "framework": "gguf",
                           "model_max_length": "4096"}}
    _fresh(monkeypatch, rows=rows, workers=[{"id": "w1"}])
    chat = _view(catalog._legacy_views(), "text.chat")
    # the largest window any bound model offers — the router picks that model
    assert chat.limits["max_context_tokens"] == 32768
    # a row without the field contributes nothing rather than a zero
    _fresh(monkeypatch, rows={"x": {"tasks": ["text-generation"]}},
           workers=[{"id": "w1"}])
    assert "max_context_tokens" not in _view(catalog._legacy_views(), "text.chat").limits


def test_legacy_view_reports_the_license_set_never_assumes_one(monkeypatch):
    rows = {"a": {"tasks": ["text-generation"], "license": "mit"},
            "b": {"tasks": ["text-generation"], "extra": {"license": "apache-2.0"}}}
    _fresh(monkeypatch, rows=rows, workers=[{"id": "w1"}])
    chat = _view(catalog._legacy_views(), "text.chat")
    assert chat.license == "apache-2.0, mit"
    _fresh(monkeypatch, rows={"a": {"tasks": ["text-generation"]}},
           workers=[{"id": "w1"}])
    assert _view(catalog._legacy_views(), "text.chat").license is None


def test_studio_views_declare_real_envelopes_and_licenses():
    """The studio zoo records these per model, so nothing here is a guess —
    and the VRAM number says out loud that it was DECLARED, not measured."""
    views = {v.name: v for v in catalog._studio_views()}
    id_lock = views["video.generate.id_lock"]
    assert id_lock.license                       # LicenseClass from models_seed
    assert id_lock.limits["max_duration_s"] > 0
    assert "x" in id_lock.limits["max_resolution"]
    if id_lock.resources.vram_gib is not None:
        assert id_lock.resources.vram_provenance is Provenance.DECLARED
        assert id_lock.resources.min_vram_gb == id_lock.resources.vram_gib


def test_studio_fingerprint_is_none_while_weights_are_unpinned():
    from hugpy_video.intel.studio.registry import MODEL_REGISTRY
    for view in catalog._studio_views():
        bound = [c for c in MODEL_REGISTRY.values() if c.model_id in view.model_ids]
        pinned = bool(bound) and all(c.weight_hash for c in bound)
        assert (view.model_fingerprint is not None) is pinned


def test_descriptor_authority_is_derived_from_k97_s_tables(monkeypatch):
    from hugpy_oracle import authority
    _fresh(monkeypatch, workers=[{"id": "w1"}])
    for view in catalog.list_capabilities():
        expected = []
        if view.name in authority.IDENTITY_CONDITIONED:
            expected.append(AuthorityKind.LIKENESS)
        if view.name in authority.VOICE_CONDITIONED:
            expected.append(AuthorityKind.VOICE)
        assert list(view.authority_required) == expected, view.name
    assert (catalog.capability_authority("video.generate.id_lock")
            == (AuthorityKind.LIKENESS,))


def test_audio_tts_declares_no_unconditional_voice_authority():
    """k97 classes it VOICE_ON_REFERENCE: a licensed synthetic voice is not a
    rights question, and a descriptor field cannot express 'only when a
    reference rides along'."""
    assert catalog.capability_authority("audio.tts") == ()
    # …while the doc's alias, which IS reference-conditioned by name, does
    assert (catalog.capability_authority("voice.synthesize.reference_conditioned")
            == ())          # resolves to audio.tts, same conditional rule


def test_declared_access_names_only_real_capabilities():
    known = set(catalog.declared_capability_names())
    for name, kinds in catalog.CAPABILITY_ACCESS_DECL.items():
        assert name in known, name
        assert kinds and all(isinstance(k, AccessKind) for k in kinds)
    assert AccessKind.NETWORK in catalog.CAPABILITY_ACCESS_DECL["web.fetch"]


def test_declared_param_schema_matches_the_fixed_dispatch_params():
    schema = catalog.CAPABILITY_PARAM_SCHEMA["audio.transcribe.word_timestamps"]
    fixed = catalog.capability_params("audio.transcribe.word_timestamps")
    assert set(schema["properties"]) == set(fixed)
    assert schema["properties"]["word_timestamps"]["const"] is fixed["word_timestamps"]


def test_eval_suite_points_at_an_evaluator_that_exists(monkeypatch):
    _fresh(monkeypatch, workers=[{"id": "w1"}])
    from hugpy_oracle import evaluation
    for view in catalog.list_capabilities():
        if view.eval_suite is None:
            continue
        module, _colon, target = view.eval_suite.partition(":")
        assert module in ("oracle.evaluation", "oracle.speech")
        if module == "oracle.evaluation":
            assert target in {r.name for r in evaluation.RUBRICS.values()}


# --- registry_version ------------------------------------------------------


def test_registry_version_is_deterministic_and_prefixed():
    rows = {"a": {"tasks": ["text-generation"], "framework": "gguf"}}
    first = catalog.registry_version(rows)
    assert first == catalog.registry_version(dict(rows))
    assert first.startswith("sha256:")


def test_registry_version_changes_when_a_row_changes():
    rows = {"a": {"tasks": ["text-generation"], "framework": "gguf"}}
    base = catalog.registry_version(rows)
    assert catalog.registry_version(
        {"a": {"tasks": ["text-generation", "text-summarization"],
               "framework": "gguf"}}) != base
    assert catalog.registry_version(
        {"a": {"tasks": ["text-generation"], "framework": "gguf",
               "license": "mit"}}) != base
    assert catalog.registry_version({**rows, "b": {"tasks": ["x"]}}) != base


def test_registry_version_ignores_discovery_churn():
    """A re-scan that rewrites mtimes and byte counts changed no routing
    decision; a digest that moved anyway would be noise nobody could act on."""
    rows = {"a": {"tasks": ["text-generation"], "framework": "gguf"}}
    churned = {"a": {**rows["a"], "scanned_at": "2026-08-20T10:00:00Z",
                     "size_bytes": 123456789}}
    assert catalog.registry_version(churned) == catalog.registry_version(rows)


def test_registry_version_changes_when_a_descriptor_version_is_bumped(monkeypatch):
    rows = {"a": {"tasks": ["text-generation"]}}
    before = catalog.registry_version(rows)
    monkeypatch.setitem(catalog.CAPABILITY_VERSION, "text.chat", "0.2.0")
    assert catalog.registry_version(rows) != before
    assert catalog.capability_version("text.chat") == "0.2.0"


def test_registry_snapshot_is_diffable():
    rows = {"a": {"tasks": ["text-generation"], "license": "mit"}}
    snap = catalog.registry_snapshot(rows)
    assert snap["schema"] == catalog.REGISTRY_SNAPSHOT_SCHEMA
    assert snap["rows"]["a"]["license"] == "mit"
    assert snap["capabilities"]["text.chat"] == catalog.capability_version("text.chat")
    assert set(snap["capabilities"]) == set(catalog.declared_capability_names())


def test_every_listed_view_is_stamped_with_the_snapshot(monkeypatch):
    _fresh(monkeypatch, workers=[{"id": "w1"}])
    views = catalog.list_capabilities()
    stamped = {v.registry_version for v in views}
    assert len(stamped) == 1 and stamped.pop().startswith("sha256:")


def test_model_dir_fingerprint_is_manifest_shaped(monkeypatch, tmp_path):
    directory = str(tmp_path)
    with open(os.path.join(directory, "weights.safetensors"), "wb") as fh:
        fh.write(b"x" * 16)
    first = catalog.model_dir_fingerprint(directory)
    assert first and first.startswith("sha256:")
    assert first == catalog.model_dir_fingerprint(directory)
    with open(os.path.join(directory, "extra.json"), "w") as fh:
        fh.write("{}")
    assert catalog.model_dir_fingerprint(directory) != first
    assert catalog.model_dir_fingerprint(os.path.join(directory, "nope")) is None


# --- probes at catalog build ------------------------------------------------


def test_a_failing_probe_makes_a_capability_ineligible_with_its_reason(monkeypatch):
    """Doc §3.2: the descriptor and the probe must agree before the capability
    is offered."""
    _fresh(monkeypatch, workers=[{"id": "w1"}])
    monkeypatch.setitem(
        probes.PROBE_SPECS, "audio.transcribe",
        probes.ProbeSpec(capability="audio.transcribe",
                         runner_module="not.a.real.module"))
    try:
        view = _view(catalog.list_capabilities(), "audio.transcribe")
        assert view.probe is not None and view.probe.failed
        assert view.eligibility.eligible is False
        assert any("not.a.real.module" in r for r in view.eligibility.reasons)
    finally:
        probes.clear_cache()


def test_an_unknown_probe_leaves_eligibility_alone(monkeypatch):
    _fresh(monkeypatch, workers=[{"id": "w1"}])
    monkeypatch.setitem(
        probes.PROBE_SPECS, "audio.transcribe",
        probes.ProbeSpec(capability="audio.transcribe"))   # nothing to check
    try:
        view = _view(catalog.list_capabilities(), "audio.transcribe")
        assert view.probe.status is ProbeStatus.UNKNOWN
        assert view.eligibility.eligible is True
    finally:
        probes.clear_cache()


def test_capabilities_without_a_probe_say_so_by_being_none(monkeypatch):
    _fresh(monkeypatch, workers=[{"id": "w1"}])
    views = {v.name: v for v in catalog.list_capabilities()}
    assert views["text.chat"].probe is None          # no probe declared
    assert views["audio.tts"].probe is not None      # one is
    assert set(probes.PROBE_SPECS) <= set(views)


def test_the_route_publishes_the_descriptor_with_no_route_change(monkeypatch):
    """GET /oracle/capabilities serializes whatever the catalog returns — the
    route file was not opened by k101."""
    from flask import Flask
    from hugpy_server.app.routes.oracle_routes import oracle_bp
    _fresh(monkeypatch, workers=[{"id": "w1"}])
    app = Flask("oracle-descriptor-test")
    app.register_blueprint(oracle_bp)
    body = app.test_client().get("/oracle/capabilities").get_json()

    entries = {e["name"]: e for e in body["capabilities"]}
    versions = {e["registry_version"] for e in entries.values()}
    assert len(versions) == 1 and versions.pop().startswith("sha256:")
    for entry in entries.values():
        for key in ("version", "param_schema", "result_schema", "limits",
                    "authority_required", "access", "license", "eval_suite",
                    "adapter_version", "model_fingerprint", "probe",
                    "registry_version"):
            assert key in entry, (entry["name"], key)
    assert entries["web.fetch"]["access"] == ["network", "external"]
    assert entries["audio.tts"]["probe"]["status"] in ("ok", "fail", "unknown")
    # MEASURED on the first seating (chatterbox on a-brain, 2026-08-21). It was
    # None while nobody had run it — the assertion that mattered then was "never
    # a guess", and it still is: the pair must be (a number, "measured") or
    # (None, "unknown"), never a number wearing "unknown".
    assert (entries["audio.tts"]["resources"]["vram_gib"]
            == catalog.TTS_MEASURED_VRAM_GIB)
    assert entries["audio.tts"]["resources"]["vram_provenance"] == "measured"
