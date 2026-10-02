"""Fit preview prices under the spill a real load carries (2026-10-02): central's
per-model map + query overrides, as a request overlay — never os.environ."""
import os

from hugpy_fleet.worker import agent as A


def test_preview_spill_from_db_map_and_overrides(monkeypatch):
    monkeypatch.setitem(A._RUNTIME_SETTINGS, "spill_by_model_db",
                        {"owner~Big-MoE": {"n_cpu_moe": 12, "ctx_pct": 3, "bogus": 1}})
    sp = A._preview_spill("Big-MoE")
    assert sp["n_cpu_moe"] == 12
    assert "bogus" not in sp
    assert A._preview_spill("Big-MoE", {"n_cpu_moe": None})["n_cpu_moe"] is None
    assert A._preview_spill("Other") == {}


def test_overlay_clears_absent_contract_keys_without_touching_environ(monkeypatch):
    monkeypatch.setenv("HUGPY_N_CPU_MOE", "99")
    before = dict(os.environ)
    ov = A._spill_overlay({"bnb_4bit": "1"})
    assert ov["HUGPY_BNB_4BIT"] == "1"
    assert ov["HUGPY_N_CPU_MOE"] is None          # clear-when-absent, like a real request
    assert dict(os.environ) == before


def test_route_scopes_the_overlay(monkeypatch):
    from hugpy_engine import spill as S
    seen = {}

    def fake_preview(state, mk, bnb=None):
        seen["n"] = S.n_cpu_moe_env() if hasattr(S, "n_cpu_moe_env") else S.env_get("HUGPY_N_CPU_MOE")
        seen["bnb"] = bnb
        return {"model_key": mk}
    monkeypatch.setattr(A, "_fit_preview", fake_preview)
    monkeypatch.setitem(A._RUNTIME_SETTINGS, "spill_by_model_db", {"M": {"n_cpu_moe": 7, "bnb_4bit": True}})
    app = A.build_app(A.WorkerState(name="t", url=None, worker_id="w-preview"))
    r = app.test_client().get("/fit-preview/M")
    assert r.status_code == 200 and r.get_json()["spill"]["n_cpu_moe"] == 7
    assert seen["bnb"] is True and str(seen["n"]) == "7"
    assert S._REQUEST_ENV.get() is None
