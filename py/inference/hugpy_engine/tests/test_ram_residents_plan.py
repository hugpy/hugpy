"""RAM as an eviction pool + one RAM price for a MoE split (2026-10-02)."""
from hugpy_engine import fit
from hugpy_engine.fit.plan import _ram_capacity, _ram_deficit_evictions

G = 1 << 30
NOW = 1_800_000_000.0


def _res(mk, ram, **kw):
    return fit.Resident(mk, vram_bytes=0, host_mode="in_process", materialized=True,
                        last_call=NOW - 3600, ram_bytes=ram, **kw)


def test_capacity_anon_vs_file_backed():
    snap = fit.ResourceSnapshot(total_bytes=24 * G, free_bytes=20 * G, ram_free_bytes=34 * G,
                                ram_total_bytes=125 * G, ram_reserve_bytes=4 * G, now=NOW)
    pins = [_res("a", 60 * G)]
    assert _ram_capacity(snap, pins, False) == 34 * G                 # anon: free RAM
    assert _ram_capacity(snap, pins, True) == 125 * G - 4 * G - 60 * G  # experts: the box


def test_ram_only_need_evicts_idle_ram_residents_then_fits():
    snap = fit.ResourceSnapshot(total_bytes=24 * G, free_bytes=20 * G, ram_free_bytes=20 * G,
                                ram_total_bytes=128 * G, now=NOW)
    pool = [_res("big", 60 * G), _res("small", 8 * G)]
    extra, freed, fail = _ram_deficit_evictions(50 * G, (), pool, [], snap, fit.FitPolicy())
    assert fail is None and freed >= 30 * G and [e.model_key for e in extra]


def test_protected_ram_is_never_evicted_and_the_refusal_names_bytes():
    snap = fit.ResourceSnapshot(total_bytes=24 * G, free_bytes=20 * G, ram_free_bytes=10 * G,
                                ram_total_bytes=128 * G, now=NOW)
    prot = [_res("static", 90 * G, protected=True)]
    extra, freed, fail = _ram_deficit_evictions(50 * G, (), [], prot, snap, fit.FitPolicy())
    assert extra == () and fail is not None and fail.kind == "ram_fit"
    assert "short" in fail.reason and "protected" in fail.reason


def test_derived_budget_is_soft_pair_budget_hard():
    from hugpy_engine.fit.plan import _split_failure
    split = fit.MoeSplit(n_cpu_moe=39, gpu_bytes=5 * G, cpu_bytes=45 * G, basis="test",
                         contract_n_cpu_moe=39) if hasattr(fit, "MoeSplit") else None
    if split is None:
        from hugpy_engine.fit.types import MoeSplit
        split = MoeSplit(n_cpu_moe=39, gpu_bytes=5 * G, cpu_bytes=45 * G, basis="test", contract_n_cpu_moe=39)
    req = fit.FitRequest(model_key="M", need_bytes=50 * G)
    soft = fit.FitPolicy(ram_target_bytes=35 * G)
    hard = fit.FitPolicy(ram_target_bytes=35 * G, ram_target_source="pair")
    assert _split_failure(split, req, soft) is None
    assert _split_failure(split, req, hard).kind == "ram_budget"
