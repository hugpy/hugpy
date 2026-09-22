"""Provider seams (``hugpy_oracle.providers``): defaults are safe, installs
are honoured, and the modules that used to import fleet/curation read
through them."""

from __future__ import annotations

import pytest

from hugpy_oracle import providers


@pytest.fixture(autouse=True)
def _clean_providers():
    providers.reset_providers()
    yield
    providers.reset_providers()


def test_defaults_are_the_no_fleet_no_curation_answers():
    assert providers.get_dossier_source().root() is None
    assert list(providers.get_dossier_source().iter_dossiers()) == []
    assert providers.get_doctrine_source().latest() is None
    gate = providers.get_task_capability_gate()
    assert gate.task_capable({}, "text-to-speech") is True             # legacy worker
    assert gate.task_capable({"task_capabilities": {"x": True}}, "y") is True
    assert gate.task_capable({"task_capabilities": {"y": False}}, "y") is False
    assert gate.task_capable({"task_capabilities": {"y": False}}, None) is True


def test_defaults_satisfy_the_protocols():
    assert isinstance(providers.get_dossier_source(), providers.DossierSource)
    assert isinstance(providers.get_doctrine_source(), providers.DoctrineSource)
    assert isinstance(providers.get_task_capability_gate(), providers.TaskCapabilityGate)
    assert isinstance(providers.get_load_state_source(), providers.LoadStateSource)


def test_set_and_reset_round_trip():
    class Doctrine:
        def latest(self):
            return {"version": 3}

    providers.set_doctrine_source(Doctrine())
    assert providers.get_doctrine_source().latest() == {"version": 3}
    providers.set_doctrine_source(None)
    assert providers.get_doctrine_source().latest() is None


def test_registry_load_state_reads_the_engine_placement_record():
    from hugpy_engine import placement

    class Registry(placement.NullWorkerRegistry):
        def get_worker(self, worker_id):
            if worker_id == "w1":
                return {"id": "w1", "loaded_models": ["org/model-a"], "loading": ["org/model-b"]}
            return None

    placement.set_worker_registry(Registry())
    try:
        src = providers.get_load_state_source()
        assert src.load_state("org/model-a", "w1")["healthy"] is True
        state_b = src.load_state("org/model-b", "w1")
        assert state_b["healthy"] is False and state_b["in_progress"] is True
        assert src.load_state("org/model-a", "nope") is None
    finally:
        placement.set_worker_registry(None)


def test_probes_doctrine_check_reads_the_provider():
    from hugpy_oracle import probes

    assert probes._latest_doctrine() is None

    class Doctrine:
        def latest(self):
            return {"id": "d1"}

    providers.set_doctrine_source(Doctrine())
    assert probes._latest_doctrine() == {"id": "d1"}


def test_catalog_seams_read_engine_placement_and_the_gate():
    from hugpy_engine import placement
    from hugpy_oracle import catalog

    class Registry(placement.NullWorkerRegistry):
        def list_workers(self, *, online_only=True):
            return ({"id": "w1", "task_capabilities": {"text-to-speech": False}},)

    class Blocklist:
        def blocked_keys(self):
            return ("org/bad",)

        def block_reason(self, model_key):
            return "operator"

    placement.set_worker_registry(Registry())
    placement.set_blocklist(Blocklist())
    try:
        workers = catalog._online_workers()
        assert [w["id"] for w in workers] == ["w1"]
        assert catalog._worker_task_capable(workers[0], "text-to-speech") is False
        assert catalog._worker_task_capable(workers[0], "text-generation") is True
        assert catalog._blocked_model_keys() == {"org/bad"}
    finally:
        placement.reset_providers()
    assert catalog._online_workers() == []          # null registry: no fleet
    assert catalog._blocked_model_keys() == set()


def test_interim_ledger_dossier_source_goes_through_the_provider(tmp_path):
    from hugpy_oracle import interim_ledger as il

    src = il.DiscoveryDossierSource(root=str(tmp_path))
    ok, reason, _ = src.probe()
    assert ok is False and "source_unavailable" in reason
    assert list(src._entries()) == []

    class Dossiers:
        def root(self):
            return "/curation/dossiers"

        def iter_dossiers(self):
            yield {"criteria": "tts", "hub_id": "org__repo",
                   "path": "/curation/dossiers/tts/org__repo.json",
                   "payload": {"hub_id": "org/repo", "verdict": "pass",
                               "created_at": "2026-09-22T00:00:00Z"}}

    providers.set_dossier_source(Dossiers())
    src = il.DiscoveryDossierSource(root=str(tmp_path))
    assert src.probe()[0] is True
    entries = list(src._entries())
    assert len(entries) == 1
    assert entries[0].kind == "dossier"
    assert entries[0].artifact_refs == ("/curation/dossiers/tts/org__repo.json",)
