"""Engine-side pieces of feasibility distribution (2026-09-24):

- the fleet DISTRIBUTION MODE store (models_config) — default "feasible", env
  override, and preservation of the sibling wildcard map across writes;
- the per-model ``strict`` override reader;
- routing_diagnostics naming the SPECIFIC gate per worker when ``skips`` is
  supplied (BUG 2), vs the bare "not a routing candidate" default.
"""

from __future__ import annotations

import json
import os

import pytest


# ── distribution-mode store ─────────────────────────────────────────────────
@pytest.fixture
def mc(tmp_path, monkeypatch):
    m = pytest.importorskip("hugpy_engine.config.models.models_config")
    monkeypatch.setattr(m, "MODELS_DISCOVERY_PATH",
                        os.path.join(str(tmp_path), "model_discovery.json"))
    monkeypatch.delenv("HUGPY_DISTRIBUTION", raising=False)
    return m


def test_distribution_defaults_to_feasible(mc):
    assert mc.fleet_distribution_mode() == "feasible"


def test_set_distribution_and_preserve_wildcard(mc):
    mc.set_worker_wildcard("w1", True)
    assert mc.set_fleet_distribution("designated") == {"distribution": "designated"}
    assert mc.fleet_distribution_mode() == "designated"
    # the wildcard sibling must survive the distribution write ...
    assert mc.worker_wildcard_state() == {"w1": True}
    # ... and the distribution must survive a later wildcard write.
    mc.set_worker_wildcard("w2", True)
    assert mc.fleet_distribution_mode() == "designated"
    assert mc.worker_wildcard_state() == {"w1": True, "w2": True}


def test_distribution_env_override_and_bad_value(mc, monkeypatch):
    mc.set_fleet_distribution("designated")
    monkeypatch.setenv("HUGPY_DISTRIBUTION", "feasible")
    assert mc.fleet_distribution_mode() == "feasible"      # env wins
    monkeypatch.setenv("HUGPY_DISTRIBUTION", "garbage")
    assert mc.fleet_distribution_mode() == "designated"    # bad env -> store
    with pytest.raises(ValueError):
        mc.set_fleet_distribution("nope")


# ── per-model strict override ───────────────────────────────────────────────
def test_model_strict_reader(tmp_path, monkeypatch):
    ov = pytest.importorskip("hugpy_engine.serve.overrides")
    path = str(tmp_path / "serve_overrides.json")
    json.dump({"Strict-Model": {"strict": True}, "Loose-Model": {"n_gpu_layers": -1}},
              open(path, "w"))
    monkeypatch.setattr(ov, "_OVERRIDES_PATH", path)
    assert ov.model_strict("Strict-Model") is True
    assert ov.model_strict("Loose-Model") is False
    assert ov.model_strict("Absent-Model") is False


# ── routing_diagnostics per-gate skip (BUG 2) ───────────────────────────────
def test_diagnostics_names_specific_gate_when_skips_supplied():
    RD = pytest.importorskip("hugpy_engine.routing_diagnostics")
    workers = [{"id": "w1", "name": "ae-worker", "status": "online",
                "admission": "approved", "models": ["M"], "loaded_models": ["M"],
                "models_local": ["M"]}]
    diag = RD.build(request_id="r1", requested="M", resolved="M", gate="no_worker",
                    predicate="p", workers=workers, candidate_ids=(),
                    skips={"w1": "eligible, but excluded by the ordered worker "
                                 "preference ['aeb'] (off-list; model is strict)"})
    row = diag["candidates"][0]
    assert "ordered worker preference" in row["skipped"]
    assert "not a routing candidate" not in row["skipped"]


def test_diagnostics_default_when_no_skip_supplied():
    RD = pytest.importorskip("hugpy_engine.routing_diagnostics")
    workers = [{"id": "w1", "name": "ae-worker", "status": "online",
                "admission": "approved", "models": ["M"], "loaded_models": ["M"],
                "models_local": ["M"]}]
    diag = RD.build(request_id="r2", requested="M", resolved="M", gate="no_worker",
                    predicate="p", workers=workers, candidate_ids=())
    # No skip + not a candidate -> the honest default naming what it holds.
    assert "not a routing candidate" in diag["candidates"][0]["skipped"]
