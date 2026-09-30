"""KNOWN-GOOD CONTRACT — the worker's VRAM admission is wired through the pure
core (fleet side).

Catalogue: notes/KNOWN-GOOD-CORE.md (area "evict + fit / allocation");
design: notes/CORE-ISOLATION-DESIGN.md Part A.
Source under test: hugpy_fleet/worker/agent.py (_vram_evict_to_fit gathers a
VramSnapshot + FitPolicy + Resident rows + FitRequest ONCE, calls
hugpy_engine.fit.plan_fit, then executes the FitPlan), hugpy_fleet/worker/
flex.py (re-export shim over hugpy_engine.fit.flex).

Deterministic: the agent's device / resident / eviction seams are stubbed on
the module exactly as tests/test_vram_evict_to_fit.py does; no GPU, no worker.
"""
from __future__ import annotations

import importlib

import pytest

A = importlib.import_module("hugpy_fleet.worker.agent")
gen_gate = importlib.import_module("hugpy_fleet.worker.gen_gate")
fit = importlib.import_module("hugpy_engine.fit")
fit_plan_mod = importlib.import_module("hugpy_engine.fit.plan")
fleet_flex = importlib.import_module("hugpy_fleet.worker.flex")
engine_flex = importlib.import_module("hugpy_engine.fit.flex")
D = importlib.import_module("hugpy_engine.dispatch.dispatch")

GIB = 1 << 30


class _State:
    pass


@pytest.fixture
def rig(monkeypatch):
    """A 24 GiB card with a mutable free cell and a resident set; evicting a
    resident removes it and returns its bytes to free (the reclaim)."""
    for leak in ("HUGPY_GPU_MEM_GIB", "HUGPY_CPU_MEM_GIB", "HUGPY_ALLOC_MODE",
                 "HUGPY_LENIENCY_PCT", "HUGPY_PRIORITY_DEVICE", "HUGPY_BNB_4BIT",
                 "HUGPY_N_GPU_LAYERS", "HUGPY_VRAM_CEILING_FRAC",
                 "HUGPY_VRAM_RESERVE_GIB", "HUGPY_VRAM_CEILING_CUSHION_GIB",
                 "HUGPY_EVICT_LEAST_REAPING", "HUGPY_NO_EVICT"):
        monkeypatch.delenv(leak, raising=False)
    card = {"total": 24 * GIB, "free": 0, "need": 0}
    residents: dict = {}
    lru: dict = {}
    evicted: list = []
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: card["total"])
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: card["free"])
    monkeypatch.setattr(A, "_incoming_need_detail",
                        lambda mk: {"total": card["need"], "weights": card["need"], "kv": 0,
                                    "ctx_pct": None, "ctx_resolved": None, "ctx_max": None,
                                    "geometry_source": None})
    monkeypatch.setattr(A, "_vram_residents",
                        lambda s: [{"model_key": k, "vram_bytes": v, "host_mode": "slot",
                                    "alive": True} for k, v in residents.items()])
    monkeypatch.setattr(A, "_residency", lambda mk: "on-demand")
    monkeypatch.setattr(A, "_busy_slot_models", lambda: set())
    monkeypatch.setattr(A, "_comfy_busy_reason", lambda s: None)
    monkeypatch.setattr(A, "_queued_ahead_of", lambda subject: set())
    monkeypatch.setattr(A, "detect_gpus", lambda: [])
    monkeypatch.setattr(A, "_target_device_index", lambda: None)
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (None, None))
    monkeypatch.setattr(A, "_subject_resident_vram_bytes", lambda s, mk: 0)
    monkeypatch.setattr(A, "_trim_host_ram", lambda: None)
    monkeypatch.setattr(gen_gate, "in_flight", lambda mk: 0)
    monkeypatch.setattr(D, "last_used_snapshot", lambda: dict(lru))

    def _fake_evict(state, mk, force=False):
        evicted.append(mk)
        vb = residents.pop(mk, None)
        if vb:
            card["free"] += vb
        return {"model_key": mk, "evicted": bool(vb), "vram_freed": vb, "host_mode": "slot"}
    monkeypatch.setattr(A, "_evict_model", _fake_evict)
    A._VRAM_EVICTIONS.update(count=0, last=None, last_at=0.0)
    return type("Rig", (), {"card": card, "residents": residents, "lru": lru,
                            "evicted": evicted})()


def test_admission_gathers_once_and_decides_through_plan_fit(rig, monkeypatch):
    """INVARIANT: _vram_evict_to_fit builds ONE VramSnapshot / FitPolicy /
    FitRequest / Resident set and consults hugpy_engine.fit.plan_fit for the
    decision; the executor then performs exactly the evictions the plan chose
    (coldest first) and re-proves fit against the live card. The verdict dict
    keeps its historical shape. Established: core isolation step 1 (2026-09-29)."""
    calls: list = []
    real = fit_plan_mod.plan_fit

    def spy(request, snapshot, residents, policy):
        plan = real(request, snapshot, residents, policy)
        calls.append((request, snapshot, tuple(residents), policy, plan))
        return plan
    monkeypatch.setattr(fit, "plan_fit", spy)

    rig.card["free"] = 1 * GIB
    rig.card["need"] = 4 * GIB
    rig.residents.update({"warm": 6 * GIB, "cold": 6 * GIB})
    rig.lru.update({"warm": 900.0, "cold": 100.0})

    verdict = A._vram_evict_to_fit(_State(), "subject")

    assert verdict["action"] == "evicted"
    assert verdict["evicted"] == ["cold"] and rig.evicted == ["cold"]
    assert verdict["freed_bytes"] == 6 * GIB and verdict["reason"] is None
    assert len(calls) == 1, "the happy eviction path plans exactly once"
    request, snapshot, residents, policy, plan = calls[0]
    assert isinstance(request, fit.FitRequest) and request.model_key == "subject"
    assert isinstance(snapshot, fit.VramSnapshot)
    assert snapshot.total_bytes == 24 * GIB and snapshot.free_bytes == 1 * GIB
    assert isinstance(policy, fit.FitPolicy)
    assert sorted(r.model_key for r in residents) == ["cold", "warm"]
    assert all(isinstance(r, fit.Resident) for r in residents)
    assert plan.action == "evict" and plan.evicted_keys == ["cold"]


def test_stable_budget_plans_once_and_verifies_after_execution(rig, monkeypatch, caplog):
    """INVARIANT (step 2, F1b — stable budget): one admission calls plan_fit
    EXACTLY once and executes that plan from its one snapshot. A refusing plan
    evicts nothing (nothing would fit afterwards — a wasted eviction); the
    refusal is priced in the plan's basis and carries the plan's structured
    reason. The card is read once more AFTER execution, by `plan_fit verify:`,
    which only logs. Retires the step-1 guard that re-planned the tail from a
    fresh read (S1b's second verdict line naming an already-gone victim).
    Established: core isolation step 2 (2026-09-29)."""
    import logging
    calls: list = []
    real = fit_plan_mod.plan_fit

    def spy(request, snapshot, residents, policy):
        calls.append((snapshot, tuple(residents)))
        return real(request, snapshot, residents, policy)
    monkeypatch.setattr(fit, "plan_fit", spy)

    rig.card["free"] = 1 * GIB
    rig.card["need"] = 20 * GIB
    rig.residents.update({"idle": 2 * GIB})
    rig.lru.update({"idle": 100.0})

    with caplog.at_level(logging.INFO, logger=A.logger.name):
        verdict = A._vram_evict_to_fit(_State(), "subject")

    assert verdict["action"] == "refuse"
    assert verdict["evicted"] == [] and rig.evicted == []       # a refuse plan evicts nothing
    assert "idle" in rig.residents
    assert len(calls) == 1, "one admission, one plan"
    reason = verdict["reason"]
    assert reason["state"] == "refused" and "won't fit on GPU" in reason["reason"]
    assert reason["evicted_freed_bytes"] == 0
    assert reason["free_vram_bytes"] == 1 * GIB                # the plan's basis
    assert reason["free_vram_measured_bytes"] == 1 * GIB       # the verify read
    assert reason["plan_refuse_reason"].startswith("won't fit on GPU: needs 21474836480 B")
    assert reason["fit_failure"]["kind"] == "vram_fit"
    verdicts = [r.getMessage() for r in caplog.records if r.getMessage().startswith("plan_fit verdict:")]
    verifies = [r.getMessage() for r in caplog.records if r.getMessage().startswith("plan_fit verify:")]
    assert len(verdicts) == 1 and len(verifies) == 1
    assert "mismatch=False" in verifies[0] and f"measured_free_bytes={GIB}" in verifies[0]


def test_evict_plan_stops_on_the_snapshot_and_logs_a_verify_mismatch(rig, monkeypatch, caplog):
    """INVARIANT (F1b): an `evict` plan's victims are walked from the snapshot
    plus what each eviction REPORTED freeing — no live read decides. When the
    card then measures differently from the plan's prediction (here the fake
    device frees less than the victim's footprint), the admission still
    returns the plan's verdict and the verify line says mismatch=True.
    Established: core isolation step 2."""
    import logging
    rig.card["free"] = 1 * GIB
    rig.card["need"] = 4 * GIB
    rig.residents.update({"cold": 6 * GIB})
    rig.lru.update({"cold": 100.0})

    def _short_evict(state, mk, force=False):
        rig.evicted.append(mk)
        rig.residents.pop(mk, None)
        rig.card["free"] += 2 * GIB              # the device frees only 2 GiB of the 6 recorded
        return {"model_key": mk, "evicted": True, "vram_freed": 6 * GIB, "host_mode": "slot"}
    monkeypatch.setattr(A, "_evict_model", _short_evict)

    with caplog.at_level(logging.INFO, logger=A.logger.name):
        verdict = A._vram_evict_to_fit(_State(), "subject")

    assert verdict["action"] == "evicted" and verdict["evicted"] == ["cold"]
    assert verdict["freed_bytes"] == 6 * GIB
    verifies = [r for r in caplog.records if r.getMessage().startswith("plan_fit verify:")]
    assert len(verifies) == 1
    assert verifies[0].levelno == logging.WARNING
    msg = verifies[0].getMessage()
    assert f"predicted_free_bytes={7 * GIB}" in msg and f"measured_free_bytes={3 * GIB}" in msg
    assert "mismatch=True" in msg and "evicted=['cold']" in msg


def test_fleet_flex_is_a_shim_over_the_engine_core():
    """GUARD: hugpy_fleet.worker.flex re-exports the engine's fit.flex objects
    (same function objects, not copies), so every existing fleet / server
    caller and every band/offload test keeps pinning ONE implementation.
    Established: core isolation step 1 (flex.py moved into hugpy_engine.fit)."""
    for name in ("plan_flex", "plan_partial_offload", "plan_explicit_offload",
                 "band_bounds", "band_ceiling", "band_floor", "ctx_band_bounds",
                 "kv_at_ctx_pct", "flex_priority_key", "leniency_floor_pct",
                 "FlexPlan", "PartialPlan"):
        assert getattr(fleet_flex, name) is getattr(engine_flex, name), name
    assert A._shared_evict_order.__module__ == A.__name__
    assert fit.evict_order.__module__ == "hugpy_engine.fit.plan"


def test_refusal_logs_one_structured_verdict_and_names_both_free_bases(rig, monkeypatch, caplog):
    """LIVE CASE (computron 2026-09-29): a 15.7 MiB model refused on a card whose
    BUDGETABLE free is 12451840 B behind a static resident. INVARIANT: the
    worker (a) emits exactly ONE `plan_fit verdict:` line per admission carrying
    action/kind/code and every figure in BYTES, and (b) the refusal reason names
    BOTH free bases — `free_vram_bytes` (budgetable, floor already out) and
    `free_vram_device_bytes` (= budgetable + `external_floor_bytes`) — plus the
    `fit_budget_bytes` the need was compared against, so the console can show
    why a tiny need failed on a card the heartbeat still reports ~1 GiB free.
    Established: fit-hotfix 2026-09-29."""
    import logging
    MIB = 1 << 20
    rig.card["total"] = 8_585_740_288
    rig.card["free"] = 12_451_840                 # budgetable (floor already out)
    rig.card["need"] = int(18.0 * MIB)
    rig.residents.update({"Qwen2.5-VL-7B-Instruct-GGUF": 6_578_765_824})
    rig.lru.update({"Qwen2.5-VL-7B-Instruct-GGUF": 100.0})
    monkeypatch.setattr(A, "_residency",
                        lambda mk: "static" if mk == "Qwen2.5-VL-7B-Instruct-GGUF" else "on-demand")
    monkeypatch.setattr(A, "_external_vram_floor_bytes", lambda: 1 * GIB)
    monkeypatch.setattr(A, "_vram_ceiling_reserve_bytes", lambda total: 0)

    with caplog.at_level(logging.INFO, logger=A.logger.name):
        verdict = A._vram_evict_to_fit(_State(), "test-save-tiny-random-llama3-smashed-pro")

    assert verdict["action"] == "refuse" and verdict["evicted"] == []
    assert rig.evicted == []                                    # static: never touched
    reason = verdict["reason"]
    assert reason["state"] == "refused"
    assert reason["needs_bytes"] == 18_874_368
    assert reason["free_vram_bytes"] == 12_451_840
    assert reason["external_floor_bytes"] == 1 * GIB
    assert reason["free_vram_device_bytes"] == 12_451_840 + 1 * GIB
    assert reason["fit_budget_bytes"] == 12_451_840
    assert reason["ceiling_reserve_bytes"] == 0
    assert reason["fit_failure"]["kind"] == "vram_fit"
    assert reason["fit_failure"]["code"] == "wont_fit"
    assert reason["fit_failure"]["need_bytes"] == 18_874_368
    assert reason["fit_failure"]["budget_bytes"] == 12_451_840
    assert "1 protected resident(s) still hold the card" in reason["reason"]
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("plan_fit verdict:")]
    assert len(lines) >= 1, "the admission must log a structured verdict"
    first = lines[0]
    for token in ("action=refuse", "fits_now=False", "kind=vram_fit", "code=wont_fit",
                  "need_bytes=18874368", "free_bytes=12451840",
                  f"device_free_bytes={12_451_840 + GIB}", "ceiling_reserve_bytes=0",
                  f"external_floor_bytes={GIB}", "total_bytes=8585740288", "protected=1"):
        assert token in first, (token, first)


def test_tiny_model_against_a_gib_of_budgetable_free_proceeds_and_logs_fits_now(rig, caplog):
    """LIVE CASE (ae-worker 2026-09-29 13:26:59): the same 15.7 MiB model with
    1185284096 B budgetable free proceeds — no eviction, no refusal — and the
    structured verdict line says so (action=proceed fits_now=True kind=None).
    Established: fit-hotfix 2026-09-29."""
    import logging
    MIB = 1 << 20
    rig.card["free"] = 1_185_284_096
    rig.card["need"] = int(18.0 * MIB)
    rig.residents.update({"Qwen3-Coder-Next-GGUF": 18_820_000_000})
    rig.lru.update({"Qwen3-Coder-Next-GGUF": 900.0})
    with caplog.at_level(logging.INFO, logger=A.logger.name):
        verdict = A._vram_evict_to_fit(_State(), "test-save-tiny-random-llama3-smashed-pro")
    assert verdict["action"] == "proceed" and verdict["evicted"] == [] and verdict["reason"] is None
    assert rig.evicted == []
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("plan_fit verdict:")]
    assert len(lines) == 1
    for token in ("action=proceed", "fits_now=True", "kind=None", "need_bytes=18874368",
                  "free_bytes=1185284096"):
        assert token in lines[0], (token, lines[0])


def test_verdict_line_names_the_split_and_the_victims(rig, monkeypatch, caplog):
    """INVARIANT (step 2, F7 / F6d): the ONE `plan_fit verdict:` line carries the
    weights-vs-KV split (weights_bytes / kv_bytes / kv_share_pct / ctx_pct) and
    NAMES the victims and the protected rows (evicted_keys / protected_keys),
    not just their counts. Established: core isolation step 2 (2026-09-29)."""
    import logging
    monkeypatch.setattr(A, "_incoming_need_detail",
                        lambda mk: {"total": 8 * GIB, "weights": 4 * GIB, "kv": 4 * GIB,
                                    "ctx_pct": 100, "ctx_resolved": 262144,
                                    "ctx_max": 262144, "geometry_source": "gguf"})
    rig.card["free"] = 2 * GIB
    rig.residents.update({"cold": 8 * GIB, "locked": 4 * GIB})
    rig.lru.update({"cold": 100.0, "locked": 900.0})
    monkeypatch.setattr(A, "_residency", lambda mk: "static" if mk == "locked" else "on-demand")
    with caplog.at_level(logging.INFO, logger=A.logger.name):
        verdict = A._vram_evict_to_fit(_State(), "subject")
    assert verdict["action"] == "evicted" and verdict["evicted"] == ["cold"]
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("plan_fit verdict:")]
    assert len(lines) == 1
    for token in ("action=evict", f"weights_bytes={4 * GIB}", f"kv_bytes={4 * GIB}",
                  "kv_share_pct=50.0", "ctx_pct=100", "ctx_resolved=262144",
                  "evictions=1", "protected=1", "evicted_keys=['cold']",
                  "protected_keys=['locked']"):
        assert token in lines[0], (token, lines[0])


def test_ctx_cap_knob_reaches_the_policy_and_the_executor_honours_the_proposal(rig, monkeypatch, caplog):
    """INVARIANT (step 2, F7): HUGPY_CTX_CAP_ON_EVICT_PCT (unset = off) rides into
    FitPolicy.ctx_cap_on_evict_pct; a `ctx-cap` partial is executed like a flex
    — the ctx floor is committed for the subject, nothing is evicted, the
    verdict is a proceed carrying the proposal. Established: step 2."""
    import logging
    monkeypatch.setattr(A, "_incoming_need_detail",
                        lambda mk: {"total": 8 * GIB, "weights": 4 * GIB, "kv": 4 * GIB,
                                    "ctx_pct": 100, "ctx_resolved": 262144,
                                    "ctx_max": 262144, "geometry_source": "gguf"})
    rig.card["free"] = 6 * GIB
    rig.residents.update({"cold": 8 * GIB})
    rig.lru.update({"cold": 100.0})
    assert A._fit_policy(rig.card["total"]).ctx_cap_on_evict_pct is None
    monkeypatch.setenv("HUGPY_CTX_CAP_ON_EVICT_PCT", "25")
    assert A._fit_policy(rig.card["total"]).ctx_cap_on_evict_pct == 25
    A._FLEX_CTX_FLOOR.pop("subject", None)
    with caplog.at_level(logging.INFO, logger=A.logger.name):
        verdict = A._vram_evict_to_fit(_State(), "subject")
    assert verdict["action"] == "proceed" and verdict["evicted"] == [] and rig.evicted == []
    assert verdict["self_ctx_pct"] == 25 and verdict["ctx_cap"]["kv_bytes"] == 1 * GIB
    assert A._FLEX_CTX_FLOOR.get("subject") == 25
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("plan_fit verdict:")]
    assert len(lines) == 1 and "action=partial" in lines[0] and "partial_kind=ctx-cap" in lines[0]
    assert "self_ctx_pct=25" in lines[0]
    A._FLEX_CTX_FLOOR.pop("subject", None)
    monkeypatch.setenv("HUGPY_CTX_CAP_ON_EVICT_PCT", "garbage")
    assert A._fit_policy(rig.card["total"]).ctx_cap_on_evict_pct is None


def test_slot_hook_reuses_a_fresh_admission_ticket_instead_of_replanning(rig, monkeypatch):
    """INVARIANT (step 2, F1 — one plan per load): dispatch's admission records
    its verdict as a short-lived ticket; the slot pool's hook reuses it when it
    still describes the card (a `partial`, or an admission the ceiling gate now
    passes) and re-plans otherwise. S1b's second verdict line — naming a victim
    the first had evicted — is exactly the re-plan this removes.
    Established: core isolation step 2 (2026-09-29)."""
    calls: list = []
    real = fit_plan_mod.plan_fit

    def spy(request, snapshot, residents, policy):
        calls.append(request.model_key)
        return real(request, snapshot, residents, policy)
    monkeypatch.setattr(fit, "plan_fit", spy)
    A._ADMISSION_TICKETS.clear()

    # A partial admission: the ceiling gate (full need) can never pass, so the
    # slot hook must reuse the ticket rather than plan a second time.
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: ("/m.gguf", 40))
    monkeypatch.setattr(A, "_gguf_ngl_intent", lambda mk: ("auto", None))
    monkeypatch.setattr(A, "_moe_detail_for", lambda mk: None)
    rig.card["free"] = 8 * GIB
    rig.card["need"] = 20 * GIB
    rig.card["total"] = 24 * GIB
    first = A._vram_evict_to_fit(_State(), "subject")
    assert first["action"] == "partial" and first.get("n_gpu_layers")
    assert calls == ["subject"]
    again = A._slot_make_room(_State(), "subject")
    assert again is first and calls == ["subject"]          # reused, not re-planned

    # An `evicted` ticket is reused only while the ceiling gate agrees now.
    A._ADMISSION_TICKETS["subject"] = {"ts": A.time.time(),
                                       "verdict": {"action": "evicted", "evicted": ["x"]}}
    monkeypatch.setattr(A, "_worker_slot_fit_check", lambda mk: True)
    assert A._slot_make_room(_State(), "subject")["action"] == "evicted" and calls == ["subject"]
    monkeypatch.setattr(A, "_worker_slot_fit_check", lambda mk: False)
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (None, None))
    rig.card["need"] = 4 * GIB
    fresh = A._slot_make_room(_State(), "subject")            # the card moved: re-plan
    assert calls == ["subject", "subject"] and fresh["action"] == "proceed"
    # A stale ticket is dropped.
    A._ADMISSION_TICKETS["subject"] = {"ts": A.time.time() - 3600,
                                       "verdict": {"action": "partial", "n_gpu_layers": 3}}
    A._slot_make_room(_State(), "subject")
    assert calls == ["subject", "subject", "subject"]
    A._ADMISSION_TICKETS.clear()


def test_slot_evict_verb_counts_and_telemeters_like_the_plan_executor(rig, monkeypatch):
    """INVARIANT (step 2, F1): a seat promotion's eviction goes through the ONE
    verb: vram_evictions bumps, evict.start/done stream, the victim's ticket
    is void. Established: step 2."""
    events: list = []
    monkeypatch.setattr(A, "_evt_emit", lambda stage, **f: events.append((stage, f)))
    rig.residents.update({"cold": 3 * GIB})
    A._ADMISSION_TICKETS["cold"] = {"ts": A.time.time(), "verdict": {"action": "proceed"}}
    before = A._VRAM_EVICTIONS["count"]
    res = A._slot_evict_verb(_State(), "cold", "NEW")
    assert res["evicted"] is True and rig.evicted == ["cold"]
    assert A._VRAM_EVICTIONS["count"] == before + 1
    assert A._VRAM_EVICTIONS["last"]["victim"] == "cold" and A._VRAM_EVICTIONS["last"]["subject"] == "NEW"
    assert [e[0] for e in events] == ["evict.start", "evict.done"]
    assert events[1][1]["freed_bytes"] == 3 * GIB and events[0][1]["tier"] == "slot-child"


def test_s3_shape_plain_key_is_a_new_admission_never_served_by_the_distill_resident(rig, monkeypatch):
    """INVARIANT (eviction test S3 / F7, 2026-09-29): with
    ``Qwen3.8-9B-Distill-GGUF`` resident and a request for ``Qwen3.8-9B-GGUF``
    (same GGUF file on the drive, DIFFERENT key), the admission is for the
    plain key as a NEW load: the Distill seat is a distinct, evictable
    resident (it is evicted to make room) and the request's model_key reaches
    plan_fit unchanged. A resident under the same canonical key
    (``Qwen3.8-9B``) is the subject itself and is never evicted.
    Established: operator rule 2026-09-29 (key_equivalent at every layer)."""
    calls: list = []
    real = fit_plan_mod.plan_fit

    def spy(request, snapshot, residents, policy):
        calls.append((request, tuple(residents)))
        return real(request, snapshot, residents, policy)
    monkeypatch.setattr(fit, "plan_fit", spy)

    rig.card["free"] = 4 * GIB
    rig.card["need"] = 11 * GIB
    rig.residents.update({"Qwen3.8-9B-Distill-GGUF": 10 * GIB})
    rig.lru.update({"Qwen3.8-9B-Distill-GGUF": 100.0})

    verdict = A._vram_evict_to_fit(_State(), "Qwen3.8-9B-GGUF")

    assert verdict["action"] == "evicted"
    assert verdict["evicted"] == ["Qwen3.8-9B-Distill-GGUF"] == rig.evicted
    request, residents = calls[0]
    assert request.model_key == "Qwen3.8-9B-GGUF"
    assert [r.model_key for r in residents] == ["Qwen3.8-9B-Distill-GGUF"]
    assert residents[0].identity == "Qwen3.8-9B-Distill" != request.model_key

    # same canonical key under another spelling: the subject, never a victim
    rig.evicted.clear()
    rig.card["free"] = 4 * GIB
    rig.residents.clear()
    rig.residents.update({"Qwen3.8-9B": 10 * GIB})
    rig.lru.update({"Qwen3.8-9B": 100.0})
    verdict = A._vram_evict_to_fit(_State(), "Qwen3.8-9B-GGUF")
    assert verdict["evicted"] == [] and rig.evicted == [] and "Qwen3.8-9B" in rig.residents


# ---------------------------------------------------------------------------
# 2026-09-29: ONE measured picture — the refusal prices and reports the device
# ---------------------------------------------------------------------------
def test_refusal_reports_device_occupancy_composition_and_the_weights_prior(rig, monkeypatch):
    """LIVE CASE (ae-worker 2026-09-29, RTX 3090, cudaMemGetInfo total 23.6 GiB):
    a 20.1 GiB transformers file was refused "needs 23.1 GB, 21.3 GB free of
    23.6 GB (0 B ceiling reserve + 1.0 GB already held back ... for out-of-band
    GPU consumers) ... the ~2.3 GB in use IS attributed to this worker (~768.0 MB
    ...) with 0 B measured unattributed". Three defects, one sentence: the
    1.0 GB was a fixed hold-back (not a holder) counted as "in use"; the 21.3
    was the budgetable figure, not the device's 22.3; and 2.3 - 0.768 - 0 left
    1.5 GB unexplained. INVARIANT, on the default box (HUGPY_VRAM_RESERVE_GIB
    unset -> 0): the quoted free IS the device read; "in use" is total - device
    free; the composition is named term by term (attributed / measured
    unattributed / in no compute process); no hold-back is claimed; and the
    need names its x1.15 weights prior — the whole ~3.0 GB between the file and
    the need. The wire carries every figure the sentence uses. Established:
    2026-09-29 (double-count + honest-figures fix)."""
    MIB = 1 << 20
    total, dev_free = int(23.6 * GIB), int(22.3 * GIB)
    wfile = int(20.1 * GIB)
    weights = int(wfile * A._WEIGHTS_HEADROOM)
    rig.card["total"] = total
    rig.card["free"] = dev_free                   # floor 0: budgetable == device
    rig.card["need"] = weights
    monkeypatch.setattr(A, "_incoming_need_detail",
                        lambda mk: {"total": weights, "weights": weights, "kv": 0,
                                    "weights_file_bytes": wfile,
                                    "weights_headroom": A._WEIGHTS_HEADROOM,
                                    "ctx_pct": None, "ctx_resolved": None,
                                    "ctx_max": None, "geometry_source": None})
    monkeypatch.setattr(A, "_vram_occupancy_attribution",
                        lambda: {"vram_attributed_bytes": 768 * MIB,
                                 "vram_unattributed_bytes": 0})
    verdict = A._vram_evict_to_fit(_State(), "big-transformers")
    assert verdict["action"] == "refuse" and verdict["evicted"] == []
    r = verdict["reason"]
    msg = r["reason"]
    occupied = total - dev_free                             # 1.3 GiB, the device's
    unaccounted = occupied - 768 * MIB                      # in no compute process
    # the figures on the wire
    assert r["external_floor_bytes"] == 0
    assert r["free_vram_bytes"] == dev_free
    assert r["free_vram_device_bytes"] == dev_free
    assert r["ceiling_reserve_bytes"] == A._vram_ceiling_reserve_bytes(total) == 512 * MIB
    assert r["fit_budget_bytes"] == dev_free - 512 * MIB
    assert r["needs_bytes"] == weights
    assert r["needs_weights_file_bytes"] == wfile
    assert r["needs_weights_headroom"] == A._WEIGHTS_HEADROOM
    assert r["device_in_use_bytes"] == occupied
    assert r["vram_attributed_bytes"] == 768 * MIB
    assert r["vram_unattributed_bytes"] == 0
    assert r["vram_unaccounted_bytes"] == unaccounted
    assert r["fit_failure"]["kind"] == "vram_fit"
    assert r["fit_failure"]["need_bytes"] == weights
    assert r["fit_failure"]["budget_bytes"] == dev_free - 512 * MIB
    # the sentence
    hb = A._human_bytes
    assert f"needs {hb(weights)} = {hb(weights)} weights ({hb(wfile)} on disk x 1.15 weights headroom)" in msg
    assert f"{hb(dev_free)} free of {hb(total)}" in msg
    assert "held back from the free figure" not in msg      # nothing is
    assert f"~{hb(occupied)} of the device is in use = ~{hb(768 * MIB)} attributed to this worker" in msg
    assert f"~{hb(0)} measured UNATTRIBUTED" in msg
    assert f"~{hb(unaccounted)} in no compute process" in msg
    assert "nothing evictable left" in msg


# ---------------------------------------------------------------------------
# 2026-09-29: the fit never prices KV at zero — the EFFECTIVE context
# ---------------------------------------------------------------------------
def _geo():
    return {"n_layers": 36, "n_kv_heads": 8, "head_dim": 128, "ctx_train": 262144}


def _kv_at(ctx):
    return 2 * 36 * ctx * 8 * 128 * 2


@pytest.fixture
def ctx_rig(monkeypatch):
    """A model whose geometry is known and whose ctx_pct / max / served-fit
    are cells the test sets. No config lookup, no GGUF header, no GPU."""
    spill = importlib.import_module("hugpy_engine.spill")
    cells = {"pct": None, "max": 262144, "gguf_path": None, "served_fit": None}
    monkeypatch.setattr(A, "_ctx_pct", lambda mk: cells["pct"])
    monkeypatch.setattr(A, "_model_max_ctx", lambda mk, cfg=None: cells["max"])
    monkeypatch.setattr(A, "_model_kv_geometry", lambda mk, cfg=None: _geo())
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (cells["gguf_path"], 36))
    monkeypatch.setattr(spill, "served_ctx_for_fit", lambda path, **kw: cells["served_fit"])
    monkeypatch.setattr(A, "_FLEX_CTX_FLOOR", {})
    # no live card read: the unset-GGUF case exercises the geometry-incomplete
    # fallback deterministically (the whole-seat bound has its own tests)
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: None)
    return cells


def test_effective_ctx_prices_pct_set_unset_and_clamped(ctx_rig):
    """INVARIANT (operator 2026-09-29): need = weights + KV at the EFFECTIVE
    context, never 'unknown -> 0'. ctx_pct set -> pct x max; unset -> what the
    loader will actually run with (GGUF: the fit-bounded trained ctx the slot
    launches -c with, serve._ctx_for; other engines: the model max); a GGUF
    pct that the VRAM fit clamps is priced at the CLAMPED value; nothing
    readable -> the 4096 floor, still > 0. The detail records ctx_effective,
    ctx_source and kv_bytes. LIVE CASE: "KV 0 (ctx_pct unset)" while the slot
    came up at n_ctx 262144. Established: 2026-09-29."""
    tf = {"framework": "transformers"}
    gg = {"framework": "gguf"}
    # pct set (transformers): 25% of 262144 = 65536
    ctx_rig["pct"] = 25
    kv, det = A._kv_need_bytes("m", tf)
    assert det["ctx_effective"] == 65536 and det["ctx_source"] == "ctx_pct" and det["ctx_pct"] == 25
    assert kv == det["kv_bytes"] == _kv_at(65536) > 0
    # unset (transformers): the model max — the cache grows to whatever is used
    ctx_rig["pct"] = None
    kv, det = A._kv_need_bytes("m", tf)
    assert det["ctx_effective"] == 262144 and det["ctx_source"] == "model-max"
    assert kv == _kv_at(262144) > 0
    # unset (GGUF): the slot's own -c — served_ctx_for_fit (trained ctx, fit-bounded)
    ctx_rig.update(gguf_path="/models/m/q4.gguf", served_fit=131072)
    kv, det = A._kv_need_bytes("m", gg)
    assert det["ctx_effective"] == 131072 and det["ctx_source"] == "loader-default"
    assert kv == _kv_at(131072)
    # pct set (GGUF) above what the fit could hold: HONOURED verbatim, never
    # clamped (keeper 2026-09-30 — the plan splits instead)
    ctx_rig.update(pct=100, served_fit=32768)
    kv, det = A._kv_need_bytes("m", gg)
    assert det["ctx_effective"] == 262144 and det["ctx_pct"] == 100 and kv == _kv_at(262144)
    # pct set (GGUF), fit roomier than the pct: the pct governs
    ctx_rig.update(pct=10, served_fit=200000)
    kv, det = A._kv_need_bytes("m", gg)
    assert det["ctx_effective"] == 26214 and kv == _kv_at(26214)
    # nothing readable: the floor, never zero
    ctx_rig.update(pct=None, max=None, gguf_path=None, served_fit=None)
    kv, det = A._kv_need_bytes("m", tf)
    assert det["ctx_effective"] == 4096 and det["ctx_source"] == "floor" and kv == _kv_at(4096)


def test_refusal_names_kv_at_the_effective_ctx(rig, monkeypatch):
    """INVARIANT: a refusal's need line reads "... weights + KV <bytes> at ctx
    <N> (<pct>%)" (or "(loader default)" when no pct is set), and fit_failure
    carries ctx_effective / kv_bytes / ctx_pct so the wire says which context
    was refused. Established: 2026-09-29."""
    weights, kv = 20 * GIB, int(3.2 * GIB)
    rig.card["total"] = int(23.6 * GIB)
    rig.card["free"] = int(22.3 * GIB)
    rig.card["need"] = weights + kv
    det = {"total": weights + kv, "weights": weights, "kv": kv, "ctx_pct": 25,
           "ctx_resolved": 65536, "ctx_effective": 65536, "ctx_source": "ctx_pct",
           "ctx_max": 262144, "geometry_source": "geometry", "kv_bytes": kv}
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: dict(det))
    r = A._vram_evict_to_fit(_State(), "m")["reason"]
    hb = A._human_bytes
    assert f"weights + KV {hb(kv)} at ctx 65,536 (25%)" in r["reason"]
    assert r["fit_failure"]["ctx_effective"] == 65536
    assert r["fit_failure"]["kv_bytes"] == kv and r["fit_failure"]["ctx_pct"] == 25
    assert r["needs_kv_bytes"] == kv and r["ctx_resolved"] == 65536
    det.update(ctx_pct=None, ctx_source="loader-default", ctx_resolved=262144, ctx_effective=262144)
    r = A._vram_evict_to_fit(_State(), "m")["reason"]
    assert f"KV {hb(kv)} at ctx 262,144 (loader default)" in r["reason"]
    assert r["fit_failure"]["ctx_effective"] == 262144 and r["fit_failure"]["ctx_pct"] is None


# ---------------------------------------------------------------------------
# 2026-09-30 (S3b, second landing): the loader default is the WHOLE-SEAT max
# ctx in EXACTLY the fit's need basis; one ctx per admission
# ---------------------------------------------------------------------------
_REAL_NEED_DETAIL = A._incoming_need_detail          # captured before any rig stubs it
PER_TOK = 2 * 36 * 8 * 128 * 2                        # 147456 B/token: 36 kv layers x 8 heads x 128, fp16
CORR = 1.4426                                         # the adopted correction live (32804534892 / 22739868912)


@pytest.fixture
def ctx_admission(rig, monkeypatch):
    """The S3b card: the REAL need detail for a 4B GGUF (4.92 GB headroomed
    weights, 36 KV layers, trained ctx 262144, correction x1.4426), 8.96 GB
    free with a 14.96 GB 0.6B resident to evict; ctx_pct is a cell."""
    cfgmod = importlib.import_module("hugpy_engine.config.main")
    cells = {"pct": None}
    W = 4_922_465_520
    monkeypatch.setattr(A, "_incoming_need_detail", _REAL_NEED_DETAIL)
    monkeypatch.setattr(A, "_incoming_need_bytes", lambda mk: W if mk == "Qwen3.8_4B_Distilled_GGUF" else None)
    monkeypatch.setattr(A, "_calib_correction", lambda mk: CORR)
    monkeypatch.setattr(A, "_moe_plan_for", lambda mk: None)
    monkeypatch.setattr(A, "_ctx_pct", lambda mk: cells["pct"])
    monkeypatch.setattr(A, "_model_max_ctx", lambda mk, cfg=None: 262144)
    monkeypatch.setattr(A, "_model_kv_geometry",
                        lambda mk, cfg=None: {"n_layers": 36, "n_kv_heads": 8, "head_dim": 128,
                                              "ctx_train": 262144})
    monkeypatch.setattr(A, "_served_gguf_geometry",
                        lambda mk: ("/models/4b/q8.gguf", 36) if mk == "Qwen3.8_4B_Distilled_GGUF" else (None, None))
    monkeypatch.setattr(cfgmod, "get_model_config",
                        lambda mk, dict_return=False: {"framework": "gguf", "model_max_length": 262144})
    rig.card["total"] = 25_769_803_776
    rig.card["free"] = 8_955_232_256
    rig.residents["Qwen3-0.6B-GGUF"] = 14_963_179_520
    rig.lru["Qwen3-0.6B-GGUF"] = 100.0
    A._ADMISSION_TICKETS.clear()
    cells["W"] = W
    return cells


def test_s3b_unset_pct_seats_whole_at_the_whole_seat_max_ctx(ctx_admission, rig):
    """INVARIANT (keeper 2026-09-30): with ctx_pct UNSET the loader-default
    effective ctx is the LARGEST ctx at which the model fits WHOLE-SEAT on the
    reachable room (free + evictable + own seat), computed with EXACTLY the
    fit's arithmetic (_need_total: (weights x 1.15 + KV) x learned
    correction, + the ceiling reserve) — so the admission EVICTS and seats
    whole (no partial), the bound's need equals the verdict's need_bytes to
    the byte, the ticket carries the ctx, the seat reads it back, and the
    reason is logged. LIVE CASE: post8 S3b chose 120832 ignoring the x1.44
    correction -> need 32.8 GB > 23.9 GB -> partial 25/36 at 27 tok/s.
    Established: 2026-09-30."""
    reserve = A._vram_ceiling_reserve_bytes(rig.card["total"])
    room = 8_955_232_256 + 14_963_179_520 - reserve
    v = A._vram_evict_to_fit(_State(), "Qwen3.8_4B_Distilled_GGUF")
    assert v["action"] == "evicted", v.get("reason")
    assert v["evicted"] == ["Qwen3-0.6B-GGUF"] and v.get("n_gpu_layers") is None
    ctx = v["ctx_effective"]
    need = A._need_total(ctx_admission["W"], ctx * PER_TOK, CORR)
    assert need <= room < A._need_total(ctx_admission["W"], (ctx + 1024) * PER_TOK, CORR)
    assert ctx % 1024 == 0 and 4096 < ctx < 120832
    assert "whole-seat max ctx" in v["ctx_reason"]
    det = A._incoming_need_detail("Qwen3.8_4B_Distilled_GGUF")    # the seat: ticket, same ctx
    assert det["ctx_effective"] == ctx and det["ctx_source"] == "ticket"
    assert det["total"] == need                                    # bound == fit, to the byte
    assert A._seat_ctx_resolver("Qwen3.8_4B_Distilled_GGUF") == ctx
    assert A._worker_slot_fit_check("Qwen3.8_4B_Distilled_GGUF") is True


def test_s3b_pct_above_whole_seat_max_is_honoured_and_splits(ctx_admission, rig):
    """INVARIANT: ctx_pct SET is honoured verbatim (pct x max, no clamp) even
    above the whole-seat max; the plan then splits (partial) and the reason
    says the pct was honoured and names the whole-seat max. Established:
    2026-09-30."""
    ctx_admission["pct"] = 50
    v = A._vram_evict_to_fit(_State(), "Qwen3.8_4B_Distilled_GGUF")
    assert v["ctx_effective"] == 131072
    assert v["action"] == "partial", v.get("reason")
    assert "ctx_pct 50% honoured" in v["ctx_reason"] and "split" in v["ctx_reason"]
    assert A._seat_ctx_resolver("Qwen3.8_4B_Distilled_GGUF") == 131072


def test_whole_seat_bound_partials_only_when_the_floor_cannot_fit():
    """INVARIANT: the bound returns (floor, False) only when even the 4096
    floor does not fit whole — the one unset-pct case that may split."""
    geo = {"n_layers": 36, "n_kv_heads": 8, "head_dim": 128}
    c, whole = A._whole_seat_max_ctx(weights=10 * GIB, corr=None, geo=geo,
                                     room=5 * GIB, upper=262144, floor=4096)
    assert (c, whole) == (4096, False)
    c, whole = A._whole_seat_max_ctx(weights=1 * GIB, corr=None, geo=geo,
                                     room=64 * GIB, upper=262144, floor=4096)
    assert (c, whole) == (262144, True)


# ---------------------------------------------------------------------------
# 2026-09-29 (S5): the REAL worker -> central -> /v1 mapping keeps the structure
# ---------------------------------------------------------------------------
def test_live_refusal_envelope_carries_type_and_fit_failure_through_the_real_path(rig, monkeypatch):
    """INVARIANT (S5): a worker refusal produced by the REAL admission, wrapped
    the way the worker's stream wraps it (LoadRefusal -> _with_load_failure ->
    the SSE error dict), parsed by central's _event_from_worker_line, raised
    as _LoadFailed and yielded as the hold loop's ErrorEvent, RE-WRAPPED by
    execute_chat_stream (the drop site: it rebuilt the event from `message`
    alone) and rendered by v1's _openai_error(cause=...) reaches the client
    as error.type = "vram_fit" with error.fit_failure carrying kind / code /
    need_bytes / budget_bytes / blocked_by / ctx_effective / kv_bytes and
    the prose unchanged. LIVE: post5 S5 returned type api_error with no
    fit_failure although the worker logged kind=vram_fit. Established:
    2026-09-29."""
    import json
    flask = pytest.importorskip("flask")
    V1 = importlib.import_module("hugpy_server.app.routes.v1_routes")
    remote = importlib.import_module("hugpy_engine.resolvers.remote")
    ES = importlib.import_module("hugpy_engine.schemas.event_schemas")
    MIB = 1 << 20
    weights, kv = int(9.1 * GIB * 1.15), 144 * MIB
    rig.card["total"] = int(23.6 * GIB)
    rig.card["free"] = int(343.1 * MIB)
    rig.card["need"] = weights + kv
    rig.residents["Qwen3.8_4B_Distilled_GGUF"] = 21 * GIB
    monkeypatch.setattr(A, "_residency", lambda mk: "static" if mk.startswith("Qwen3.8_4B") else "on-demand")
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: {
        "total": weights + kv, "weights": weights, "kv": kv, "ctx_pct": None,
        "ctx_resolved": 4096, "ctx_effective": 4096, "ctx_source": "loader-default",
        "ctx_max": 40960, "geometry_source": "geometry", "kv_bytes": kv,
        "weights_file_bytes": int(9.1 * GIB), "weights_headroom": 1.15})
    verdict = A._vram_evict_to_fit(_State(), "Qwen3.8-9B-Distill-GGUF")
    assert verdict["action"] == "refuse"
    # worker: the refusal leaves the stream exactly as _stream_sync ships it
    exc = D.LoadRefusal(verdict["reason"])
    frame = A._with_load_failure({"type": "error", "message": f"{type(exc).__name__}: {exc}"}, exc)
    d = json.loads(json.dumps(frame))                     # the SSE wire
    assert d["load_failure"]["class"] == "vram_fit"
    # central: parse -> hold loop's terminal error -> execute_chat_stream re-wrap
    ev = remote._event_from_worker_line(d, "r1")
    lf = remote._LoadFailed(remote._humanize_worker_error("ae-worker", ev.message),
                            load_failure=ev.load_failure, code=ev.code)
    hold_ev = ES.ErrorEvent(request_id="r1", message=lf.message,
                            load_failure=lf.load_failure, code=lf.code)
    out_ev = D._rewrap_error_event(hold_ev, "r1")
    assert out_ev.load_failure and out_ev.load_failure["class"] == "vram_fit"
    # /v1: the envelope the client sees
    app = flask.Flask(__name__)
    with app.test_request_context("/v1/chat/completions"):
        resp = V1._openai_error(out_ev.message, 500, "api_error", cause=out_ev)
    body = resp[0].get_json()
    err = body["error"]
    assert err["type"] == "vram_fit" and err["code"] == 500
    assert err["message"].startswith("The 'ae-worker' worker could not complete this request: LoadRefusal: won't fit on GPU")
    assert "KV 144.0 MB at ctx 4,096 (loader default)" in err["message"]
    ff = err["fit_failure"]
    assert ff["kind"] == "vram_fit" and ff["code"] == "wont_fit"
    assert ff["need_bytes"] == weights + kv
    assert ff["budget_bytes"] == verdict["reason"]["fit_budget_bytes"]
    assert ff["blocked_by"] == ["Qwen3.8_4B_Distilled_GGUF"]
    assert ff["ctx_effective"] == 4096 and ff["kv_bytes"] == kv
    assert ff["external_floor_bytes"] == 0
    assert err["load_failure"]["fit_failure"]["ctx_effective"] == 4096


# ---------------------------------------------------------------------------
# 2026-09-30 (post10 rollback): a concurrent request's spill never leaks into
# this request's placement / politeness — and an offline replay of S1–S8
# ---------------------------------------------------------------------------
import threading as _threading

SPILL = importlib.import_module("hugpy_engine.spill")


@pytest.fixture
def overlay_reset():
    yield
    SPILL.set_request_env(None)


def _concurrent_cpu_polite_request():
    """Another request thread applies a RAM-only, polite spill (the post10
    Qwen2.5-7B) AFTER this request applied its own — writing os.environ."""
    def other():
        A._apply_spill({"n_gpu_layers": "off", "no_evict": True})
    t = _threading.Thread(target=other)
    t.start(); t.join()


def test_a_concurrent_ram_only_polite_spill_does_not_leak(rig, overlay_reset, monkeypatch):
    """INVARIANT (post10 S3b/S3c, 2026-09-30): the per-request spill is a
    ContextVar overlay — a concurrent request writing HUGPY_N_GPU_LAYERS=off
    and HUGPY_NO_EVICT into os.environ changes neither THIS request's
    placement intent (planned GPU bytes == need, never "puts 0 B on the GPU")
    nor its politeness. LIVE: the 4B launched CPU-only (1.13 GB VRAM / 22 GB
    RSS, 4.68 tok/s with 20 GB free) and the 9B read "polite flag forbids
    bumping" while a RAM-only polite Qwen2.5-7B request ran on another thread.
    Established: 2026-09-30."""
    monkeypatch.delenv("HUGPY_N_GPU_LAYERS", raising=False)
    monkeypatch.delenv("HUGPY_NO_EVICT", raising=False)
    A._apply_spill({})                                 # this request: max-gpu, not polite
    _concurrent_cpu_polite_request()
    assert os.environ.get("HUGPY_N_GPU_LAYERS") == "off"   # the other thread DID write it
    assert SPILL.n_gpu_layers_intent() == "auto"
    assert SPILL.no_evict_env() is False
    assert SPILL.planned_gpu_need_bytes(10 * GIB) == 10 * GIB
    # ...and the other thread saw its own spill
    seen = {}
    def other_reads():
        A._apply_spill({"n_gpu_layers": "off", "no_evict": True})
        seen["intent"], seen["polite"] = SPILL.n_gpu_layers_intent(), SPILL.no_evict_env()
    t = _threading.Thread(target=other_reads); t.start(); t.join()
    assert seen == {"intent": "cpu", "polite": True}
    # a context with no overlay still reads os.environ (today's behaviour)
    SPILL.set_request_env(None)
    assert SPILL.n_gpu_layers_intent() == "cpu"


import os  # noqa: E402  (used by the isolation test above)


def test_s3b_whole_seat_on_gpu_under_a_concurrent_cpu_spill(rig, overlay_reset, monkeypatch):
    """S3b post10 shape (ae 3090, total 25298141184 B, free 21935226880 B,
    0.6B 1089798144 B + 7B 898957312 B on-demand, need 23220221419 B at ctx
    75776): the admission EVICTS both and seats the 4B WHOLE ON GPU — action
    evicted, planned == need, no partial, no CPU-only proceed — although a
    concurrent RAM-only polite request wrote os.environ in between."""
    monkeypatch.delenv("HUGPY_N_GPU_LAYERS", raising=False)
    monkeypatch.delenv("HUGPY_NO_EVICT", raising=False)
    rig.card.update(total=25_298_141_184, free=21_935_226_880, need=23_220_221_419)
    rig.residents.update({"Qwen3-0.6B-GGUF": 1_089_798_144, "Qwen2.5-7B-Instruct-GGUF": 898_957_312})
    rig.lru.update({"Qwen3-0.6B-GGUF": 100.0, "Qwen2.5-7B-Instruct-GGUF": 200.0})
    A._apply_spill({})
    _concurrent_cpu_polite_request()
    v = A._vram_evict_to_fit(_State(), "Qwen3.8_4B_Distilled_GGUF")
    assert v["action"] == "evicted", v.get("reason") or v.get("note")
    assert sorted(v["evicted"]) == ["Qwen2.5-7B-Instruct-GGUF", "Qwen3-0.6B-GGUF"]
    assert "0 B on the GPU" not in str(v.get("note"))
    assert v.get("n_gpu_layers") is None


def test_s3c_is_not_polite_because_a_neighbour_was(rig, overlay_reset, monkeypatch):
    """S3c: the 9B (need 11404964054 B) beside an ON-DEMAND 4B (23220221419 B,
    free 92667904 B) evicts the 4B — post6/post8 behaviour — even though a
    concurrent polite request set HUGPY_NO_EVICT process-wide."""
    monkeypatch.delenv("HUGPY_N_GPU_LAYERS", raising=False)
    monkeypatch.delenv("HUGPY_NO_EVICT", raising=False)
    rig.card.update(total=25_298_141_184, free=92_667_904, need=11_404_964_054)
    rig.residents["Qwen3.8_4B_Distilled_GGUF"] = 23_220_221_419
    rig.lru["Qwen3.8_4B_Distilled_GGUF"] = 100.0
    A._apply_spill({})
    _concurrent_cpu_polite_request()
    v = A._vram_evict_to_fit(_State(), "Qwen3.8-9B-Distill-GGUF")
    assert v["action"] == "evicted" and v["evicted"] == ["Qwen3.8_4B_Distilled_GGUF"], v.get("reason")


# Offline replay of the acceptance harness (evict_test.py S1–S8) at the
# admission level, from the post10 log's measured card + resident state. Each
# row: (scenario, worker total, free, residents {key: (vram, static)}, subject,
# need, polite, expected action, expected evicted keys / blocked_by).
_CT, _AE = 8_184_725_504, 25_298_141_184
REPLAY = [
    ("S1a evict the on-demand VL for Coder-3B", _CT, 1_469_972_480,
     {"gemma-3-4b-it-GGUF": (3_700_000_000, False)}, "Qwen2.5-Coder-3B-Instruct-GGUF",
     3_365_919_295, False, "evicted", ["gemma-3-4b-it-GGUF"]),
    ("S3a /load 0.6B evicts Coder-Next", _AE, 4_193_320_960,
     {"Qwen3-Coder-Next-GGUF": (17_530_000_000, False)}, "Qwen3-0.6B-GGUF",
     8_120_645_904, False, "evicted", ["Qwen3-Coder-Next-GGUF"]),
    ("S3b 4B evicts 0.6B + 7B, whole seat", _AE, 21_935_226_880,
     {"Qwen3-0.6B-GGUF": (1_089_798_144, False), "Qwen2.5-7B-Instruct-GGUF": (898_957_312, False)},
     "Qwen3.8_4B_Distilled_GGUF", 23_220_221_419, False, "evicted",
     ["Qwen2.5-7B-Instruct-GGUF", "Qwen3-0.6B-GGUF"]),
    ("S3c 9B evicts the on-demand 4B", _AE, 92_667_904,
     {"Qwen3.8_4B_Distilled_GGUF": (23_220_221_419, False)}, "Qwen3.8-9B-Distill-GGUF",
     11_404_964_054, False, "evicted", ["Qwen3.8_4B_Distilled_GGUF"]),
    ("S4 polite Coder-Next beside 9B never evicts", _AE, 5_744_492_544,
     {"Qwen3.8-9B-Distill-GGUF": (11_404_964_054, False)}, "Qwen3-Coder-Next-GGUF",
     12_427_724_748, True, "refuse", []),
    ("S5 static 4B blocks the 9B", _AE, 92_667_904,
     {"Qwen3.8_4B_Distilled_GGUF": (23_343_848_688, True)}, "Qwen3.8-9B-Distill-GGUF",
     11_404_964_054, False, "refuse", ["Qwen3.8_4B_Distilled_GGUF"]),
    ("S6 Coder-3B already fits: no-op proceed", _CT, 7_670_333_440, {},
     "Qwen2.5-Coder-3B-Instruct-GGUF", 3_365_919_295, False, "proceed", []),
    ("S7 2B beside Coder-3B fits free", _CT, 4_248_698_880,
     {"Qwen2.5-Coder-3B-Instruct-GGUF": (3_365_919_295, False)}, "Qwen3.8-2B-Distill-GGUF",
     3_418_113_104, False, "proceed", []),
]


@pytest.mark.parametrize("row", REPLAY, ids=[r[0] for r in REPLAY])
def test_offline_replay_of_the_acceptance_scenarios(row, rig, overlay_reset, monkeypatch):
    """OFFLINE REPLAY (2026-09-30): the admission verdict for each acceptance
    scenario from post10's measured state — action, victims (minimum set, LRU)
    and, for a refusal, blocked_by — with a CONCURRENT RAM-only polite request
    applied in between, so the next landing is not the first time these run.
    (Layer counts for S1b / S4's MoE split need the GGUF header and stay live.)"""
    name, total, free, residents, subject, need, polite, action, keys = row
    monkeypatch.delenv("HUGPY_N_GPU_LAYERS", raising=False)
    monkeypatch.delenv("HUGPY_NO_EVICT", raising=False)
    rig.card.update(total=total, free=free, need=need)
    for i, (k, (vb, static)) in enumerate(residents.items()):
        rig.residents[k] = vb
        rig.lru[k] = 100.0 + i
    statics = {k for k, (_, s) in residents.items() if s}
    monkeypatch.setattr(A, "_residency", lambda mk: "static" if mk in statics else "on-demand")
    A._apply_spill({"no_evict": True} if polite else {})
    _concurrent_cpu_polite_request() if not polite else None
    v = A._vram_evict_to_fit(_State(), subject)
    assert v["action"] == action, (name, v.get("reason") or v.get("note"))
    assert "0 B on the GPU" not in str(v.get("note")), name
    if action == "evicted":
        assert sorted(v["evicted"]) == sorted(keys), name
    elif action == "refuse":
        assert v["evicted"] == [] and rig.evicted == [], name
        if keys:
            assert [p["model_key"] for p in v["reason"]["protected"]] == keys, name
    else:
        assert v["evicted"] == [], name
