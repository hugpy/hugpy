"""CAPACITY OUTRANKS RESIDENCY (operator incident 2026-09-25).

A studio movie job (sd-turbo, text-to-image) was routed to computron — an 8 GB
4060 Laptop with 0 GB free — because sd-turbo was tier=resident there, while
ae-worker (a 3090 with 14 GB free, online, idle) sat unused. The pick then
CUDA-OOM'd. Residency was trumping capacity.

Ruling exercised here: residency/allocation is a TIE-BREAKER among workers that
can serve at comparable quality, NOT an override. A worker where the model is
resident but which central can PROVE cannot land it GPU-resident now loses to an
eligible worker that can — even a cold one. Designation (a HARD scope) is
unaffected. And when central can prove nothing (no probe / unsizable / fits),
ranking degrades to the pre-feature order byte-for-byte.

Run:  venv/bin/python -m pytest tests/test_route_capacity.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-route-cap-test-")
os.environ.setdefault("HUGPY_COMMS_DB", "off")
os.environ["HUGPY_COLD_HOLD_HEALTH"] = "off"

import pytest  # noqa: E402

from hugpy_fleet.central import workers as W  # noqa: E402

MK = "sd-turbo"
GIB = 2 ** 30


def _w(wid, name, *, loaded=(), models=(), vram_free=0, last_picked=0.0):
    return {
        "id": wid, "name": name, "url": f"http://{name}:9100",
        "loaded_models": list(loaded),
        "models": list(models),
        "allocations": [], "grants": {},
        "_wildcard_catch": True,
        "gpus": [{"memory_total": 24 * GIB, "memory_free": vram_free}],
        "vram_free": vram_free,
        "last_picked": last_picked,
    }


@pytest.fixture()
def probe():
    """Register a fit probe that mirrors _worker_fit's verdict shape: a ~2.5 GiB
    model is gpu_resident only where free VRAM holds it, ram-only otherwise."""
    NEED = int(2.5 * GIB)

    def _fit(model_key, worker):
        vram = worker.get("vram_free")
        if vram is None:
            return {"gpu_resident": None, "need": None, "vram_free": None}
        gpu_ok = NEED <= vram
        return {
            "fit": True, "gpu_resident": gpu_ok, "need": NEED, "vram_free": vram,
            "band_floor_admissible": None,
            "reason": None if gpu_ok else (
                f"fits but would partially offload to CPU: needs ~{NEED} B, "
                f"only {vram} B VRAM free"),
        }
    W.set_free_room_probe(_fit)
    yield _fit
    W.set_free_room_probe(None)


# ── the penalty helper ───────────────────────────────────────────────────────

def test_penalty_is_zero_without_a_probe():
    W.set_free_room_probe(None)
    pen, why = W._gpu_placeable_penalty(_w("c", "computron", vram_free=0), MK)
    assert pen == 0 and why is None


def test_penalty_zero_when_it_fits_free_vram(probe):
    pen, why = W._gpu_placeable_penalty(_w("a", "ae", vram_free=14 * GIB), MK)
    assert pen == 0 and why is None


def test_penalty_one_when_provably_not_gpu_resident(probe):
    pen, why = W._gpu_placeable_penalty(_w("c", "computron", vram_free=0), MK)
    assert pen == 1
    assert why and "partially offload" in why


def test_penalty_zero_when_central_cannot_size(probe):
    # vram_free None -> central knows nothing -> neutral, worker decides.
    w = _w("c", "computron", vram_free=0)
    w["vram_free"] = None
    w["gpus"] = []
    pen, why = W._gpu_placeable_penalty(w, MK)
    assert pen == 0 and why is None


# ── the rank: capacity beats residency, designation still beats capacity ──────

def test_resident_but_oom_box_loses_to_a_cold_gpu_box(probe):
    """The incident, reduced: computron holds sd-turbo but has 0 GB free; ae is
    cold but has room. ae must win once the penalty is stamped."""
    computron = _w("c", "computron", loaded=[MK], vram_free=0, last_picked=0.0)
    ae = _w("a", "ae", vram_free=14 * GIB, last_picked=9.0)
    cands = [computron, ae]
    W._stamp_gpu_placement(cands, MK)
    assert computron["_gpu_place_penalty"] == 1
    assert ae["_gpu_place_penalty"] == 0

    def _rank(w):
        return W._routing_rank(w, MK, W._match_keys(MK), starred=False)
    assert min(cands, key=_rank)["name"] == "ae"


def test_residency_still_wins_when_both_can_place(probe):
    """Both fit on GPU -> penalty ties at 0 -> residency (term ②) decides, exactly
    as before the feature. computron (resident) wins over ae (cold)."""
    computron = _w("c", "computron", loaded=[MK], vram_free=14 * GIB, last_picked=9.0)
    ae = _w("a", "ae", vram_free=14 * GIB, last_picked=0.0)
    cands = [computron, ae]
    W._stamp_gpu_placement(cands, MK)
    assert computron["_gpu_place_penalty"] == 0 and ae["_gpu_place_penalty"] == 0

    def _rank(w):
        return W._routing_rank(w, MK, W._match_keys(MK), starred=False)
    assert min(cands, key=_rank)["name"] == "computron"


def test_designation_still_outranks_capacity(probe):
    """A HARD-designated box (not a wildcard catch) that cannot place on GPU still
    beats a wildcard box that can — designation is a scope, not a preference."""
    designated = _w("d", "designated", models=[MK], vram_free=0, last_picked=9.0)
    designated["_wildcard_catch"] = False
    wildcard_fits = _w("a", "ae", vram_free=14 * GIB, last_picked=0.0)
    cands = [designated, wildcard_fits]
    W._stamp_gpu_placement(cands, MK)

    def _rank(w):
        return W._routing_rank(w, MK, W._match_keys(MK), starred=False)
    assert min(cands, key=_rank)["name"] == "designated"


def test_unstamped_rank_is_byte_identical():
    """No probe / no stamp -> the penalty term is a constant 0 and the rank tuple
    matches the pre-feature ordering (residency wins)."""
    W.set_free_room_probe(None)
    computron = _w("c", "computron", loaded=[MK], vram_free=0, last_picked=0.0)
    ae = _w("a", "ae", vram_free=14 * GIB, last_picked=9.0)

    def _rank(w):
        return W._routing_rank(w, MK, W._match_keys(MK), starred=False)
    # residency untouched: computron (resident) still wins with no probe.
    assert min([computron, ae], key=_rank)["name"] == "computron"


# ── end to end through the real store ─────────────────────────────────────────

@pytest.fixture()
def store(monkeypatch, tmp_path):
    from hugpy_fleet.central.workers import WorkerStore
    s = WorkerStore(path=str(tmp_path / "wk.json"))
    monkeypatch.setattr(W, "worker_store", s)
    monkeypatch.setattr(W, "_assign_memory_path", lambda: str(tmp_path / "a.json"))
    monkeypatch.setattr(W, "required_pkg_version", lambda: None)
    monkeypatch.setattr(W, "_wildcard_map", lambda: {"__all__": True})
    return s


def _admit(store, name, **beat):
    w = store.register(name=name, url=f"http://{name}:9100")
    store.set_admission(w["id"], "approved")
    store.heartbeat(w["id"], **beat)
    return w["id"]


def test_pick_routes_around_the_oom_resident(store, monkeypatch, probe):
    ae = _admit(store, "ae", loaded_models=[],
                gpus=[{"memory_total": 24 * GIB, "memory_free": 14 * GIB}])
    computron = _admit(store, "computron", loaded_models=[MK],
                       gpus=[{"memory_total": 8 * GIB, "memory_free": 0}])
    monkeypatch.setattr(W, "_wildcard_map", lambda: {ae: True, computron: True})
    chosen = store.pick_for_model(MK)
    assert chosen is not None and chosen["name"] == "ae"
