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
