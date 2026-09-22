"""TASK C worker half (2026-07-25) — POST /probe/<model_key> accepts an
optional {"spill": {...}} body and applies it (via the same _apply_spill
/infer already uses) BEFORE building the runner, so an explicit
n_gpu_layers/n_cpu_moe rides the warm central's workers_load kicks off.

Regressed here without any real GPU / model / subprocess:
  * GET (no body) and a bare POST ({}) are no-ops — byte-identical to the
    pre-existing behavior (_apply_spill({}) only clears the mode-contract
    envs, never sets anything new);
  * POST {"spill": {"n_gpu_layers": -1, "n_cpu_moe": 999}} sets the matching
    env vars (HUGPY_N_GPU_LAYERS / HUGPY_N_CPU_MOE) before _probe_model runs;
  * the applied env is visible to _probe_model's own view of the world (it's
    just an env var — nothing spill-specific to fake beyond that);
  * a leaked env from a PRIOR probe's spill does not survive an unrelated
    probe with no spill — n_cpu_moe is in the clear-when-absent set, so a
    stale split can't silently displace the next model's experts.

Run: venv/bin/python -m pytest tests/test_worker_probe_spill.py -q
"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

agent = importlib.import_module("hugpy_fleet.worker.agent")


@pytest.fixture
def wclient(monkeypatch):
    state = agent.WorkerState(name="t", url=None, worker_id="w-probe")
    # _probe_model does real work (provision/runner_for/etc.) — stub it so this
    # suite only proves the ROUTE applies spill before calling it, not the
    # probe internals (which have their own coverage elsewhere).
    calls = []

    def _fake_probe(model_key, state):
        calls.append({
            "model_key": model_key,
            "env_n_gpu_layers": os.environ.get("HUGPY_N_GPU_LAYERS"),
            "env_n_cpu_moe": os.environ.get("HUGPY_N_CPU_MOE"),
        })
        return {"ok": True, "model_key": model_key}
    monkeypatch.setattr(agent, "_probe_model", _fake_probe)

    for env in ("HUGPY_N_GPU_LAYERS", "HUGPY_N_CPU_MOE"):
        monkeypatch.delenv(env, raising=False)

    client = agent.build_app(state).test_client()
    return type("Rig", (), {"client": client, "calls": calls})()


def test_get_probe_no_body_is_a_noop(wclient):
    r = wclient.client.get("/probe/coder-next")
    assert r.status_code == 200
    assert wclient.calls[-1]["env_n_gpu_layers"] is None
    assert wclient.calls[-1]["env_n_cpu_moe"] is None


def test_post_probe_empty_body_is_a_noop(wclient):
    r = wclient.client.post("/probe/coder-next", json={})
    assert r.status_code == 200
    assert wclient.calls[-1]["env_n_gpu_layers"] is None
    assert wclient.calls[-1]["env_n_cpu_moe"] is None


def test_post_probe_with_spill_sets_env_before_probing(wclient):
    r = wclient.client.post(
        "/probe/coder-next",
        json={"spill": {"n_gpu_layers": -1, "n_cpu_moe": 999}})
    assert r.status_code == 200
    assert wclient.calls[-1]["env_n_gpu_layers"] == "-1"
    assert wclient.calls[-1]["env_n_cpu_moe"] == "999"


def test_post_probe_partial_spill_n_cpu_moe_only(wclient):
    r = wclient.client.post(
        "/probe/coder-next", json={"spill": {"n_cpu_moe": 999}})
    assert r.status_code == 200
    assert wclient.calls[-1]["env_n_cpu_moe"] == "999"


def test_spill_does_not_leak_into_a_later_unrelated_probe(wclient):
    wclient.client.post(
        "/probe/coder-next",
        json={"spill": {"n_gpu_layers": -1, "n_cpu_moe": 999}})
    assert wclient.calls[-1]["env_n_cpu_moe"] == "999"
    # A later probe for a DIFFERENT model with no spill must not inherit the
    # split — n_cpu_moe is a clear-when-absent key.
    wclient.client.post("/probe/other-model", json={})
    assert wclient.calls[-1]["env_n_cpu_moe"] is None
