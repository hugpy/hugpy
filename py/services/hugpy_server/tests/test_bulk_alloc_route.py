"""POST /llm/workers/<id>/alloc-all — bulk-alloc route (todo t15; per-model % t48).

The operator clarified the multi-select was for ALLOC (GPU allocation), residency
kept too. The single-model alloc editor applies via POST /assign {model_key,
spill} — a CENTRAL REGISTRY write to spill_by_model (assign_model), NOT the
/ops/config relay family — so, unlike residency/pin, it does NOT restart the
agent. The bulk action must match that EXACTLY: N registry writes in one request,
per-model results like residency-all, and a constant restarting:False.

This regresses the route WITHOUT a live worker:
  * one assign_model call per selected key carrying the SAME spill contract
    (body: {"spill": {...}}) — for autofit / max GPU / CPU only / an operator-
    typed absolute GiB budget, every selected model IS meant to get the same
    contract;
  * OR one assign_model call per key with its OWN spill (body:
    {"spills": {model_key: {...}}}) — t48: a PERCENT VRAM/RAM budget must
    resolve against each model's OWN size, so two differently-sized models in
    the same bulk selection land two DIFFERENT absolute GiB numbers instead of
    one flat number (resolved once against the worker's capacity) stamped on
    both — that was the bug ("...not the total for the particular model that
    happened to be the first in the list's actual ram alloc");
  * the spill value set mirrors the editor (autofit {} / max GPU {n_gpu_layers:-1}
    / CPU only {n_gpu_layers:"off"} / custom budgets);
  * NEVER a relay / restart (restarting is always False, no /ops/config touched);
  * off-worker keys dropped (render->click staleness) → skipped, never assigned;
  * per-model results / counts / alloc label surfaced like residency-all;
  * one bad key errors that key only, the rest still apply;
  * the engine gate is evaluated per-key against THAT key's own spill in the
    `spills` path (a per-model map can carry different explicit-budget keys per
    member, unlike the single shared `spill`);
  * bad body -> 400; bad spill shape -> 400; bad spills shape -> 400; unknown
    worker -> 404.
"""
from __future__ import annotations

import importlib

import pytest
from flask import Flask

wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
cr = importlib.import_module("hugpy_server.app.routes.comms_routes")
oa = importlib.import_module("hugpy_server.app.operator_auth")

# Worker has a,b,c designated; a,b have files local (warmable), c does not; z is
# NOT designated.
WORKER = {"id": "wid", "name": "box", "models": ["a", "b", "c"],
          "models_local": ["a", "b"]}
# Engine table for the gate: base tests treat a,b,c as GGUF so the historical
# behavior (max GPU / custom apply to all) is preserved; the engine-gating tests
# swap in a MIXED map.
FRAMEWORKS = {"a": "gguf", "b": "gguf", "c": "gguf"}

# Mixed selection: g1 gguf, t1/t2 transformers, cf comfy.
MIXED = {"id": "wid", "name": "box",
         "models": ["g1", "t1", "t2", "cf"], "models_local": []}
MIXED_FW = {"g1": "gguf", "t1": "transformers", "t2": "transformers", "cf": "comfy"}


class Harness:
    """The route's collaborators, faked in place on the worker_routes module."""

    def __init__(self, monkeypatch):
        self.mp = monkeypatch
        # (worker_id, model_key, spill); the PINS round (2026-09-23) makes every
        # assign_model caller pass a provenance `source` — recorded separately so
        # the historical (worker_id, model_key, spill) assertions stay intact
        # while the source is still checkable (see assign_sources).
        self.assign_calls: list = []
        self.assign_sources: list = []    # source= passed to each assign_model call
        self.warm_calls: list = []        # (models,) passed to _kick_warm
        self.relay_seen = {"n": 0}
        app = Flask(__name__)
        app.register_blueprint(wr.worker_bp)
        self.client = app.test_client()
        self.use_worker(WORKER, FRAMEWORKS)
        monkeypatch.setattr(wr, "assign_model", self.fake_assign)
        monkeypatch.setattr(wr, "_kick_warm", self._fake_warm)
        monkeypatch.setattr(wr, "_relay_worker_op", self._relay_tripwire)
        monkeypatch.setattr(cr, "audit", lambda *a, **k: None)

    def use_worker(self, worker, frameworks):
        self.mp.setattr(wr, "get_worker",
                        lambda wid: dict(worker) if wid == "wid" else None)
        self.mp.setattr(wr, "_model_framework", lambda mk: frameworks.get(mk))

    def fake_assign(self, worker_id, model_key, spill=None, source=None, retag=True):
        self.assign_calls.append((worker_id, model_key, spill))
        self.assign_sources.append(source)
        return dict(WORKER)   # assign_model returns the public worker view

    def _fake_warm(self, worker, model_keys, source):
        self.warm_calls.append(tuple(model_keys))
        return list(model_keys)

    # Guard: the alloc path must NEVER relay to the worker (no restart). Any
    # call to _relay_worker_op is a failure of the "no restart" contract.
    def _relay_tripwire(self, *a, **k):
        self.relay_seen["n"] += 1
        raise AssertionError("alloc-all must NOT relay to the worker (no restart)")

    def post(self, path="/llm/workers/wid/alloc-all", **json):
        return self.client.post(path, json=json)

    def assigned_keys(self):
        return [c[1] for c in self.assign_calls]


@pytest.fixture
def h(monkeypatch):
    return Harness(monkeypatch)


@pytest.fixture
def mixed(h):
    h.use_worker(MIXED, MIXED_FW)
    return h


# ── broadcast `spill` path ───────────────────────────────────────────────────
def test_max_gpu_subset_one_assign_per_key_same_spill_no_relay(h):
    r = h.post(model_keys=["a", "b"], spill={"n_gpu_layers": -1})
    body = r.get_json()
    assert r.status_code == 200
    assert len(h.assign_calls) == 2, "one assign_model per selected key"
    assert all(c[2] == {"n_gpu_layers": -1} for c in h.assign_calls)
    # PINS round: every bulk-alloc write carries operator provenance.
    assert h.assign_sources == ["operator", "operator"]
    assert h.relay_seen["n"] == 0, "NEVER relayed (no restart)"
    assert body["restarting"] is False
    assert body["alloc"] == "gpu-only", "k37 honest name"
    assert body["results"] == {"a": "ok", "b": "ok"}
    assert body["counts"] == {"ok": 2, "error": 0, "skipped": 0, "total": 2}
    # only LOCAL models re-seated (a,b local; warm once)
    assert h.warm_calls == [("a", "b")]


def test_autofit_empty_spill_clears_override(h):
    r = h.post(model_keys=["a"], spill={})
    assert h.assign_calls == [("wid", "a", {})]
    assert r.get_json()["alloc"] == "max-gpu", "k37 honest name"


def test_autofit_null_spill_is_autofit(h):
    h.post(model_keys=["a"], spill=None)
    assert h.assign_calls == [("wid", "a", {})]


def test_cpu_only_and_custom_budget_labels(h):
    r = h.post(model_keys=["a"], spill={"n_gpu_layers": "off"})
    assert r.get_json()["alloc"] == "ram-only"
    r = h.post(model_keys=["a"],
               spill={"gpu_mem_gib": 8, "cpu_mem_gib": 16, "threads": 4})
    assert r.get_json()["alloc"] == "8G VRAM · 16G RAM · 4 cores"


def test_off_worker_key_dropped_and_reported(h):
    """Render->click staleness: an undesignated key is skipped, never assigned."""
    body = h.post(model_keys=["a", "z"], spill={"n_gpu_layers": -1}).get_json()
    assert h.assigned_keys() == ["a"]
    assert body["off_worker"] == ["z"], "distinct from engine skips"
    assert body["results"] == {"a": "ok"}


def test_all_keys_off_worker_no_assign_honest_note(h):
    r = h.post(model_keys=["z", "q"], spill={})
    body = r.get_json()
    assert r.status_code == 200 and h.assign_calls == []
    assert body["counts"] == {"ok": 0, "error": 0, "skipped": 0, "total": 0}
    assert "note" in body and set(body["off_worker"]) == {"z", "q"}


def test_one_bad_key_errors_that_key_only(h, monkeypatch):
    def _assign_b_raises(worker_id, model_key, spill=None, source=None, retag=True):
        h.assign_calls.append((worker_id, model_key, spill))
        if model_key == "b":
            raise RuntimeError("boom")
        return dict(WORKER)
    monkeypatch.setattr(wr, "assign_model", _assign_b_raises)
    body = h.post(model_keys=["a", "b", "c"], spill={"n_gpu_layers": -1}).get_json()
    assert len(h.assign_calls) == 3, "all three attempted"
    assert body["results"]["a"] == "ok" and body["results"]["c"] == "ok"
    assert "boom" in body["results"]["b"]
    assert body["counts"] == {"ok": 2, "error": 1, "skipped": 0, "total": 3}
    assert body["ok"] is False, "ok flag False when any errored"


@pytest.mark.parametrize("path,payload,status", [
    ("/llm/workers/wid/alloc-all", {"spill": {}}, 400),                      # missing model_keys
    ("/llm/workers/wid/alloc-all", {"model_keys": [], "spill": {}}, 400),    # empty model_keys
    ("/llm/workers/wid/alloc-all", {"model_keys": ["a"], "spill": {"bogus": 1}}, 400),
    ("/llm/workers/wid/alloc-all", {"model_keys": ["a"], "spill": {"n_gpu_layers": "lots"}}, 400),
    ("/llm/workers/wid/alloc-all", {"model_keys": ["a"], "spill": 5}, 400),  # non-dict spill
    ("/llm/workers/nope/alloc-all", {"model_keys": ["a"], "spill": {}}, 404),
])
def test_bad_body_bad_spill_unknown_worker(h, path, payload, status):
    assert h.client.post(path, json=payload).status_code == status


def test_spill_validator_and_label_helpers():
    assert wr._validate_alloc_spill(None) == ({}, None), "None -> autofit {}"
    assert wr._validate_alloc_spill({"n_gpu_layers": -1}) == ({"n_gpu_layers": -1}, None)
    clean, reason = wr._validate_alloc_spill({"bad": 1})
    assert clean is None and "unsupported" in reason
    assert wr._alloc_label({}) == "max-gpu"
    assert wr._alloc_label({"n_gpu_layers": -1}) == "gpu-only"
    assert wr._alloc_label({"n_gpu_layers": "off"}) == "ram-only"


def test_route_is_operator_gated():
    """alloc-all POST sits in the assign-family tier of _SENSITIVE."""
    assert any("POST" in methods and rx.match("/llm/workers/w1/alloc-all")
               for methods, rx in oa._SENSITIVE)


# ── ENGINE GATING (t15 refinement — GGUF-only alloc) ─────────────────────────
def test_mixed_explicit_budget_applies_to_gguf_only(mixed):
    body = mixed.post(model_keys=["g1", "t1", "t2", "cf"],
                      spill={"gpu_mem_gib": 8}).get_json()
    assert mixed.assigned_keys() == ["g1"]
    assert body["results"]["g1"] == "ok"
    t1 = body["results"]["t1"]
    assert "skipped" in t1 and "GGUF-only" in t1 and "transformers" in t1
    assert "skipped" in body["results"]["cf"] and "comfy" in body["results"]["cf"]
    assert body["counts"] == {"ok": 1, "error": 0, "skipped": 3, "total": 4}
    assert body["ok"] is True, "an engine skip is not a failure"


def test_mixed_max_gpu_is_placement_intent_applies_to_all(mixed):
    """t26: Max GPU is engine-agnostic (the worker maps n_gpu_layers to
    transformers placement). No skips."""
    body = mixed.post(model_keys=["g1", "t1"], spill={"n_gpu_layers": -1}).get_json()
    assert sorted(mixed.assigned_keys()) == ["g1", "t1"]
    assert all(c[2] == {"n_gpu_layers": -1} for c in mixed.assign_calls)
    assert body["counts"] == {"ok": 2, "error": 0, "skipped": 0, "total": 2}


def test_mixed_cpu_only_applies_to_all_engines(mixed):
    body = mixed.post(model_keys=["g1", "t1", "cf"],
                      spill={"n_gpu_layers": "off"}).get_json()
    assert sorted(mixed.assigned_keys()) == ["cf", "g1", "t1"]
    assert body["counts"] == {"ok": 3, "error": 0, "skipped": 0, "total": 3}


def test_all_transformers_max_gpu_all_apply(mixed):
    body = mixed.post(model_keys=["t1", "t2"], spill={"n_gpu_layers": -1}).get_json()
    assert sorted(mixed.assigned_keys()) == ["t1", "t2"]
    assert body["counts"] == {"ok": 2, "error": 0, "skipped": 0, "total": 2}


@pytest.mark.parametrize("spill", [
    {"gpu_mem_gib": 8},                               # explicit budget key
    {"alloc_mode": "explicit", "leniency_pct": 20},   # explicit MODE key
])
def test_all_transformers_explicit_all_skipped(mixed, spill):
    """explicit stays GGUF-only whether it rides via a budget key or alloc_mode."""
    body = mixed.post(model_keys=["t1", "t2"], spill=spill).get_json()
    assert mixed.assign_calls == []
    assert body["ok"] is True
    assert body["counts"] == {"ok": 0, "error": 0, "skipped": 2, "total": 2}


def test_all_transformers_max_ram_all_apply(mixed):
    """max-ram was opened for non-GGUF 2026-07-24 (loaders honor it)."""
    body = mixed.post(model_keys=["t1", "t2"], spill={"alloc_mode": "max-ram"}).get_json()
    assert sorted(mixed.assigned_keys()) == ["t1", "t2"]
    assert all(c[2] == {"alloc_mode": "max-ram"} for c in mixed.assign_calls)
    assert body["counts"] == {"ok": 2, "error": 0, "skipped": 0, "total": 2}


def test_mixed_max_ram_engine_agnostic(mixed):
    body = mixed.post(model_keys=["g1", "t1", "cf"], spill={"alloc_mode": "max-ram"}).get_json()
    assert sorted(mixed.assigned_keys()) == ["cf", "g1", "t1"]
    assert body["counts"] == {"ok": 3, "error": 0, "skipped": 0, "total": 3}


def test_autofit_applies_to_every_engine(mixed):
    body = mixed.post(model_keys=["g1", "t1", "t2", "cf"], spill={}).get_json()
    assert sorted(mixed.assigned_keys()) == ["cf", "g1", "t1", "t2"]
    assert body["counts"] == {"ok": 4, "error": 0, "skipped": 0, "total": 4}


def test_unknown_engine_only_explicit_fails_safe(mixed, monkeypatch):
    monkeypatch.setattr(wr, "_model_framework", lambda mk: None)
    body = mixed.post(model_keys=["g1"], spill={"gpu_mem_gib": 8}).get_json()
    assert mixed.assign_calls == []
    assert "skipped" in body["results"]["g1"] and "unknown engine" in body["results"]["g1"]
    mixed.post(model_keys=["g1"], spill={"n_gpu_layers": -1})
    assert mixed.assigned_keys() == ["g1"], "placement intent is not gated"


@pytest.mark.parametrize("spill,gguf_only", [
    ({}, False),
    (None, False),
    ({"n_gpu_layers": -1}, False),                  # Max GPU (t26)
    ({"n_gpu_layers": "off"}, False),               # CPU only (t26)
    ({"gpu_mem_gib": 8}, True),
    ({"cpu_mem_gib": 8}, True),
    ({"threads": 4}, True),
    ({"tensor_split": [0.5, 0.5]}, True),
    ({"n_gpu_layers": -1, "gpu_mem_gib": 8}, True), # budget ALONGSIDE n_gpu_layers
    # value-sensitive alloc_mode (2026-07-24): max-ram NOT gguf-only, explicit IS.
    ({"alloc_mode": "max-ram"}, False),
    ({"alloc_mode": "explicit"}, True),
    ({"leniency_pct": 20}, True),                   # explicit-only companion
    ({"priority_device": "ram"}, True),             # explicit-only companion
])
def test_alloc_is_gguf_only_helper(spill, gguf_only):
    assert wr._alloc_is_gguf_only(spill) is gguf_only


def test_alloc_spill_ok_for_engine_helper(mixed):
    assert wr._alloc_spill_ok_for_engine({}, "t1") == (True, None)
    assert wr._alloc_spill_ok_for_engine({"n_gpu_layers": -1}, "t1") == (True, None)
    assert wr._alloc_spill_ok_for_engine({"n_gpu_layers": "off"}, "t1") == (True, None)
    assert wr._alloc_spill_ok_for_engine({"alloc_mode": "max-ram"}, "t1") == (True, None)
    ok2, r2 = wr._alloc_spill_ok_for_engine({"gpu_mem_gib": 8}, "t1")
    assert ok2 is False and "GGUF-only" in r2 and "transformers" in r2
    okem, rem = wr._alloc_spill_ok_for_engine(
        {"alloc_mode": "explicit", "leniency_pct": 20}, "t1")
    assert okem is False and "GGUF-only" in rem and "explicit" in rem and "analogue" in rem
    assert wr._alloc_spill_ok_for_engine({"gpu_mem_gib": 8}, "g1") == (True, None)
    assert wr._alloc_spill_ok_for_engine({"alloc_mode": "max-ram"}, "g1")[0] is True


def test_single_assign_route_enforces_engine_gate(mixed, monkeypatch):
    """Defense-in-depth: the single-model /assign route gates too. A gguf-only
    spill on a transformers key -> 409; autofit / max-ram -> allowed."""
    monkeypatch.setattr(wr, "_central_missing_reason", lambda mk: None)     # central has it
    monkeypatch.setattr(wr, "_disk_preflight_reason", lambda w, mk: None)    # fits
    monkeypatch.setattr(wr, "get_models_dict",
                        lambda dict_return=False: {"t1": {}, "g1": {}})
    post = lambda **j: mixed.client.post("/llm/workers/wid/assign", json=j)

    rr = post(model_key="t1", spill={"gpu_mem_gib": 8})
    assert rr.status_code == 409
    assert "GGUF-only" in (rr.get_json() or {}).get("error", "")

    mixed.assign_calls.clear()
    rr = post(model_key="t1", spill={})
    assert rr.status_code == 200 and mixed.assign_calls == [("wid", "t1", {})]

    assert post(model_key="g1", spill={"n_gpu_layers": -1}).status_code == 200

    # max-ram on a transformers key is now ALLOWED (opened 2026-07-24).
    mixed.assign_calls.clear()
    rr = post(model_key="t1", spill={"alloc_mode": "max-ram"})
    assert rr.status_code == 200
    assert mixed.assign_calls == [("wid", "t1", {"alloc_mode": "max-ram"})]

    # explicit MODE on a transformers key STILL 409s.
    rr = post(model_key="t1", spill={"alloc_mode": "explicit", "leniency_pct": 20})
    assert rr.status_code == 409
    assert "GGUF-only" in (rr.get_json() or {}).get("error", "")


def test_resolved_percent_budget_is_a_normal_custom_spill():
    """PERCENTAGE -> GiB resolution is a UI-side concern (BudgetInput/AllocControl
    resolve % against the worker's effective capacity AT APPLY TIME; the wire
    carries concrete gpu_mem_gib/cpu_mem_gib). This pins that the contract the
    UI resolves TO stays valid: a resolved budget is a normal gguf-only alloc."""
    clean, reason = wr._validate_alloc_spill({"gpu_mem_gib": 6.0, "cpu_mem_gib": 12.0})
    assert reason is None and clean == {"gpu_mem_gib": 6.0, "cpu_mem_gib": 12.0}
    assert wr._alloc_is_gguf_only(clean) is True


# ── PER-MODEL alloc (t48): `spills: {model_key: spill}` ──────────────────────
# This is how a bulk PERCENT VRAM/RAM budget rides the wire now: the UI resolves
# the percent against EACH model's own size client-side (still no percent
# concept on the wire) and sends the resulting per-model absolutes here in one
# request, so a 40% budget on a big model and a 40% budget on a small model land
# as two DIFFERENT GiB numbers instead of one flat number stamped on both.
def test_per_model_spills_each_key_its_own(h):
    r = h.post(model_keys=["a", "b"],
               spills={"a": {"gpu_mem_gib": 9.6, "cpu_mem_gib": 4.0},
                       "b": {"gpu_mem_gib": 2.4, "cpu_mem_gib": 1.0}})
    body = r.get_json()
    assert r.status_code == 200
    assert {c[1]: c[2] for c in h.assign_calls} == {
        "a": {"gpu_mem_gib": 9.6, "cpu_mem_gib": 4.0},
        "b": {"gpu_mem_gib": 2.4, "cpu_mem_gib": 1.0}}
    a_spill = next(c[2] for c in h.assign_calls if c[1] == "a")
    b_spill = next(c[2] for c in h.assign_calls if c[1] == "b")
    assert a_spill != b_spill, "distinct per-model values actually landed"
    assert h.relay_seen["n"] == 0
    assert body["restarting"] is False
    assert body["alloc"] == "per-model"
    assert body["results"] == {"a": "ok", "b": "ok"}
    assert body["counts"] == {"ok": 2, "error": 0, "skipped": 0, "total": 2}
    assert h.warm_calls == [("a", "b")], "only LOCAL models re-seated, warm once"


def test_per_model_key_absent_from_spills_is_autofit(h):
    """A key missing from `spills` (caller only sent a subset) -> autofit for
    that key (same {}-is-autofit convention as the broadcast path)."""
    h.post(model_keys=["a", "b"], spills={"a": {"n_gpu_layers": -1}})
    assert dict((c[1], c[2]) for c in h.assign_calls) == {"a": {"n_gpu_layers": -1}, "b": {}}


def test_per_model_engine_gate_evaluated_per_key(mixed):
    """g1 gets an explicit gguf-only budget, t1 too — only t1 is skipped (its
    OWN spill is gguf-only on a transformers engine); g1 IS gguf so applies."""
    body = mixed.post(model_keys=["g1", "t1"],
                      spills={"g1": {"gpu_mem_gib": 8}, "t1": {"gpu_mem_gib": 4}}).get_json()
    assert body["results"]["g1"] == "ok"
    assert "skipped" in body["results"]["t1"] and "GGUF-only" in body["results"]["t1"]
    assert body["counts"] == {"ok": 1, "error": 0, "skipped": 1, "total": 2}
    assert mixed.assigned_keys() == ["g1"]


def test_per_model_off_worker_key_dropped(mixed):
    body = mixed.post(model_keys=["g1", "zz"], spills={"g1": {}}).get_json()
    assert body["off_worker"] == ["zz"] and body["results"] == {"g1": "ok"}


@pytest.mark.parametrize("spills", ["nope", {"g1": {"bogus": 1}}])
def test_per_model_bad_shapes_400(mixed, spills):
    assert mixed.post(model_keys=["g1"], spills=spills).status_code == 400
