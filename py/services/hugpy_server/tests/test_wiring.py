"""hugpy_server.wiring.install_all wires every seam in py/WIRING.md (server rows)."""
from __future__ import annotations

import pytest

from hugpy_server import wiring


@pytest.fixture(scope="module")
def report():
    return wiring.install_all()


def test_every_step_reported_ok(report):
    assert report["ok"] is True, report["errors"]
    for step in wiring.STEPS:
        assert step in report, step
        assert report[step]["ok"] is True, (step, report[step])
    assert report["errors"] == []
    assert wiring.last_report()["ok"] is True


def test_placement_providers_are_fleet_adapters(report):
    from hugpy_engine import placement
    assert type(placement.get_worker_registry()).__name__ == "FleetWorkerRegistry"
    assert type(placement.get_worker_transport()).__name__ == "FleetWorkerTransport"
    assert type(placement.get_eviction_ledger()).__name__ == "FleetEvictionLedger"
    assert type(placement.get_blocklist()).__name__ == "FleetBlocklist"
    assert type(placement.get_model_metrics()).__name__ == "FleetModelMetrics"
    assert type(placement.get_priority_groups()).__name__ == "FleetPriorityGroups"


def test_oracle_providers_come_from_fleet(report):
    from hugpy_oracle import providers as op
    assert type(op.get_doctrine_source()).__name__ == "FleetDoctrineSource"
    assert type(op.get_task_capability_gate()).__name__ == "FleetTaskCapabilityGate"
    assert type(op.get_load_state_source()).__name__ == "FleetLoadStateSource"
    # curation reuses the one doctrine source and installs its dossier source
    assert type(op.get_dossier_source()).__name__ == "DossierStoreSource"
    from hugpy_curation import providers as cp
    assert type(cp.get_doctrine_source()).__name__ == "FleetDoctrineSource"


def test_task_plugins_loaded(report):
    from hugpy_engine import tasks
    registered = set(tasks.registered_tasks())
    assert {"automatic-speech-recognition", "text-to-image"} <= registered   # media
    assert {"text-to-video", "image-to-video"} <= registered                 # video


def test_catalog_bridge_and_video_hooks(report):
    from hugpy_engine import catalog_bridge
    assert catalog_bridge.installed() is True
    from hugpy_video import hooks
    assert type(hooks.get_prompt_coordinator()).__name__ == "OraclePromptCoordinator"
    from hugpy_video import jobs
    assert "video_performance" in jobs.registered_jobs()


def test_footprint_selector_is_storage_format_select(report):
    from hugpy_storage import providers as sp
    from hugpy_storage.format_select import effective_bytes
    assert sp.get_footprint_selector() is effective_bytes


def test_hf_token_listener_rebuilds_server_client(report):
    from hugpy_storage import hf_token
    from hugpy_server.app.functions.imports.utils import constants as c
    assert wiring._on_hf_token_change in hf_token._TOKEN_LISTENERS
    before = c.get_hf_api()
    wiring._on_hf_token_change("hf_test_token_xyz")
    after = c.get_hf_api()
    assert after is not before and c.hfApi is after
    assert after.token == "hf_test_token_xyz"
    wiring._on_hf_token_change(None)
    assert c.get_hf_api().token in (None, False)


def test_steps_are_isolated(monkeypatch):
    """One failing step is recorded and the others still install."""
    def boom():
        raise RuntimeError("nope")
    monkeypatch.setattr(wiring, "install_footprint_selector", boom)
    rep = wiring.install_all()
    assert rep["ok"] is False
    assert rep["errors"] == ["footprint_selector"]
    assert rep["footprint_selector"]["ok"] is False
    assert "RuntimeError" in rep["footprint_selector"]["error"]
    assert rep["placement"]["ok"] is True
    assert rep["comms_bus"]["ok"] is True
