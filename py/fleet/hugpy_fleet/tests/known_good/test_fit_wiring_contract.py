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
    # pct set (GGUF) but the fit clamps below it: price the clamped value
    ctx_rig.update(pct=100, served_fit=32768)
    kv, det = A._kv_need_bytes("m", gg)
    assert det["ctx_effective"] == 32768 and det["ctx_pct"] == 100 and kv == _kv_at(32768)
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
# 2026-09-29 (S3b): ONE effective ctx per admission — dispatch == seat
# ---------------------------------------------------------------------------
_REAL_NEED_DETAIL = A._incoming_need_detail          # captured before any rig stubs it
PER_TOK = 2 * 36 * 8 * 128 * 2                        # 36 layers / 8 kv heads / 128 head_dim, fp16


@pytest.fixture
def ctx_admission(rig, monkeypatch):
    """The rig with the REAL need detail: a GGUF (framework via config), known
    geometry, headroomed weights, and a served_ctx_for_fit fake that records
    the free figure it was priced against and returns the fit bound exactly
    as spill does: (free - weights - 512 MiB) // per_tok, rounded down to
    1024, capped at the trained ctx. A call WITHOUT a free figure (a live
    re-resolve) returns the trained max and is counted separately."""
    spill = importlib.import_module("hugpy_engine.spill")
    cfgmod = importlib.import_module("hugpy_engine.config.main")
    MIB = 1 << 20
    cells = {"pct": None, "raw": int(4.5 * GIB), "calls": [], "unhinted": 0}
    monkeypatch.setattr(A, "_incoming_need_detail", _REAL_NEED_DETAIL)
    monkeypatch.setattr(A, "_incoming_need_bytes", lambda mk: int(cells["raw"] * A._WEIGHTS_HEADROOM))
    monkeypatch.setattr(A, "_calib_correction", lambda mk: None)
    monkeypatch.setattr(A, "_moe_plan_for", lambda mk: None)
    monkeypatch.setattr(A, "_ctx_pct", lambda mk: cells["pct"])
    monkeypatch.setattr(A, "_model_max_ctx", lambda mk, cfg=None: 262144)
    monkeypatch.setattr(A, "_model_kv_geometry",
                        lambda mk, cfg=None: {"n_layers": 36, "n_kv_heads": 8, "head_dim": 128,
                                              "ctx_train": 262144})
    # the subject's served quant only: a resident's own KV pricing (flex inputs,
    # no admission hint) takes the model-max branch and never re-fits the card
    monkeypatch.setattr(A, "_served_gguf_geometry",
                        lambda mk: ("/models/m/q4.gguf", 36) if mk == "m" else (None, None))
    monkeypatch.setattr(cfgmod, "get_model_config",
                        lambda mk, dict_return=False: {"framework": "gguf", "model_max_length": 262144})

    def fake_served_ctx(path, *, free_vram=None, weights_on_gpu_bytes=None, **kw):
        if free_vram is None:
            cells["unhinted"] += 1
            return 262144
        cells["calls"].append((free_vram, weights_on_gpu_bytes))
        w = weights_on_gpu_bytes or cells["raw"]
        bound = (int(free_vram) - int(w) - 512 * MIB) // PER_TOK
        return max(4096, min(262144, (bound // 1024) * 1024))
    monkeypatch.setattr(spill, "served_ctx_for_fit", fake_served_ctx)
    A._ADMISSION_TICKETS.clear()
    return cells


def _expected_bound(free_hint, weights, reserve):
    MIB = 1 << 20
    b = (free_hint - reserve - weights - 512 * MIB) // PER_TOK
    return (b // 1024) * 1024


def test_one_effective_ctx_per_admission_dispatch_equals_seat(ctx_admission, rig):
    """INVARIANT (S3b, 2026-09-29): an admission resolves the effective ctx
    ONCE, against the room it can REACH (free + evictable + own seat, less the
    ceiling reserve) in its own basis (headroomed weights), records it on the
    ticket, and the seat's resolver (serve._ctx_for -> the child's -c) and the
    slot ceiling gate read THAT value — never a re-resolve on the emptied
    card. Holds for ctx_pct unset (loader default), a pct the fit clamps, and
    a pct within the fit. And a model that fits whole-seat at its effective
    ctx is admitted whole: evicted, no partial, ceiling gate green at the
    same ctx. LIVE CASE: 4B priced at 27648 by dispatch, re-priced 129024 by
    the seat -> partial 24/36, 25 tok/s (post4: whole seat, 138 tok/s).
    Established: 2026-09-29."""
    cells = ctx_admission
    weights = int(cells["raw"] * A._WEIGHTS_HEADROOM)
    reserve = A._vram_ceiling_reserve_bytes(24 * GIB)
    for pct, label in ((None, "unset -> loader default"), (100, "pct clamped by the fit"),
                       (5, "pct within the fit")):
        cells["pct"] = pct
        cells["calls"].clear(); cells["unhinted"] = 0
        A._ADMISSION_TICKETS.clear()
        rig.card["total"] = 24 * GIB
        rig.card["free"] = 1 * GIB                             # a full card
        rig.residents.clear(); rig.residents["squatter"] = 20 * GIB
        rig.lru.clear(); rig.lru["squatter"] = 100.0
        rig.evicted.clear()
        free_hint = 1 * GIB + 20 * GIB                         # free + what it may evict
        bound = _expected_bound(free_hint, weights, reserve)
        expect = bound if pct is None else min(bound, int(262144 * pct / 100))
        v = A._vram_evict_to_fit(_State(), "m")
        assert v["action"] == "evicted" and v["evicted"] == ["squatter"], (label, v.get("reason"))
        assert v["ctx_effective"] == expect, (label, v["ctx_effective"], expect)
        # the ctx was priced against the REACHABLE room in the admission's basis
        assert cells["calls"] and cells["calls"][0] == (free_hint - reserve, weights), label
        assert cells["unhinted"] == 0, label
        n_calls = len(cells["calls"])
        # the seat reads the ticket: same ctx, no re-resolve
        assert A._seat_ctx_resolver("m") == expect, label
        assert A._effective_ctx("m", {"framework": "gguf"})["source"] == "ticket"
        assert len(cells["calls"]) == n_calls and cells["unhinted"] == 0, label
        # whole seat: the need at that ctx fits the emptied card under the gate
        det = A._incoming_need_detail("m")
        assert det["ctx_effective"] == expect and det["kv"] == expect * PER_TOK
        assert rig.card["free"] == 21 * GIB
        assert det["total"] <= rig.card["free"] - reserve, label
        assert A._worker_slot_fit_check("m") is True, label
        assert v.get("n_gpu_layers") is None and v["action"] != "partial", label
    # a NEW admission drops the old ticket and re-resolves
    A._ADMISSION_TICKETS["m"] = {"ts": 0.0, "verdict": {"ctx_effective": 4096}}   # stale (expired)
    assert A._effective_ctx("m", {"framework": "gguf"})["source"] != "ticket"


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
