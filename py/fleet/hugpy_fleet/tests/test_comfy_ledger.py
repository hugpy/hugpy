"""The comfy resident ledger (worker.comfy_ledger) and the agent seams that
bind it (2026-09-22): a per-checkpoint NEED for the headroom path, comfy's
residents by name for the planner, and idle-only eviction of comfy.

Pure-module tests need no GPU; the agent-seam tests stub the probes exactly
like the sibling suites do.
"""
from __future__ import annotations

import importlib
import os

import pytest

from hugpy_fleet.worker import comfy_ledger as L

A = importlib.import_module("hugpy_fleet.worker.agent")

GIB = 2**30


# ═══════════ pure module ═══════════════════════════════════════════════════
def test_cushion_default_and_env(monkeypatch):
    monkeypatch.delenv("HUGPY_COMFY_GEN_CUSHION_GIB", raising=False)
    assert L.gen_cushion_bytes() == 2 * GIB
    monkeypatch.setenv("HUGPY_COMFY_GEN_CUSHION_GIB", "0.5")
    assert L.gen_cushion_bytes() == GIB // 2
    monkeypatch.setenv("HUGPY_COMFY_GEN_CUSHION_GIB", "garbage")
    assert L.gen_cushion_bytes() == 2 * GIB
    monkeypatch.setenv("HUGPY_COMFY_GEN_CUSHION_GIB", "-1")
    assert L.gen_cushion_bytes() == 2 * GIB


def test_predicted_need_prices_weights_plus_cushion():
    assert L.predicted_need_bytes(6 * GIB, held=False, cushion=GIB) == 7 * GIB
    # already resident: only the working room
    assert L.predicted_need_bytes(6 * GIB, held=True, cushion=GIB) == GIB
    # unknown file: unknown need, never a guess
    assert L.predicted_need_bytes(None, held=False, cushion=GIB) is None
    # ...unless it is held, when the cushion is all that is needed
    assert L.predicted_need_bytes(None, held=True, cushion=GIB) == GIB


def test_checkpoint_dirs_order_and_dedup(monkeypatch, tmp_path):
    a, b, c = (tmp_path / n for n in ("a", "b", "c"))
    monkeypatch.setenv("COMFY_CHECKPOINTS_DIR", str(a))
    monkeypatch.setenv("COMFY_CHECKPOINT_DIRS", os.pathsep.join([str(b), "", str(a)]))
    assert L.checkpoint_dirs([str(c), str(b)]) == [str(a), str(b), str(c)]


def test_checkpoint_size_direct_subdir_symlink_and_dangling(monkeypatch, tmp_path):
    monkeypatch.delenv("COMFY_CHECKPOINTS_DIR", raising=False)
    monkeypatch.delenv("COMFY_CHECKPOINT_DIRS", raising=False)
    root = tmp_path / "ckpts"
    (root / "sub").mkdir(parents=True)
    store = tmp_path / "store"
    store.mkdir()
    (root / "direct.safetensors").write_bytes(b"x" * 1000)
    (root / "sub" / "nested.safetensors").write_bytes(b"y" * 2000)
    (store / "real.safetensors").write_bytes(b"z" * 3000)
    os.symlink(store / "real.safetensors", root / "linked.safetensors")
    os.symlink(tmp_path / "gone.safetensors", root / "dangling.safetensors")

    dirs = [str(root)]
    assert L.checkpoint_size_bytes("direct.safetensors", dirs) == 1000
    assert L.checkpoint_size_bytes("sub/nested.safetensors", dirs) == 2000
    # comfy's scan is recursive: a bare name found below the root still sizes
    assert L.checkpoint_size_bytes("nested.safetensors", dirs) == 2000
    # symlinks are followed to the bytes
    assert L.checkpoint_size_bytes("linked.safetensors", dirs) == 3000
    # a dangling link is NOT FOUND, never 0
    assert L.checkpoint_size_bytes("dangling.safetensors", dirs) is None
    assert L.checkpoint_size_bytes("absent.safetensors", dirs) is None
    assert L.checkpoint_size_bytes(None, dirs) is None
    assert L.checkpoint_size_bytes("", dirs) is None


def test_checkpoint_size_reads_env_roots(monkeypatch, tmp_path):
    (tmp_path / "hot.safetensors").write_bytes(b"h" * 10)
    monkeypatch.setenv("COMFY_CHECKPOINTS_DIR", str(tmp_path))
    assert L.checkpoint_size_bytes("hot.safetensors") == 10


def test_ledger_dispatch_free_and_lru_order():
    clock = {"t": 100.0}
    led = L.ComfyLedger(clock=lambda: clock["t"])
    assert len(led) == 0 and led.residents() == []
    led.note_dispatch("sd15", "sd15.safetensors", 2 * GIB)
    clock["t"] = 101.0
    led.note_dispatch("sdxl", "sdxl.safetensors", 6 * GIB)
    assert led.resident_keys() == ["sd15", "sdxl"]
    assert led.holds("sd15") and not led.holds("nope")
    assert led.resident_bytes() == 8 * GIB
    # re-dispatch refreshes last_used and moves to the end (LRU order)
    clock["t"] = 102.0
    led.note_dispatch("sd15", "sd15.safetensors", 2 * GIB)
    assert led.resident_keys() == ["sdxl", "sd15"]
    assert led.last_used("sd15") == 102.0
    rows = led.residents()
    assert [r["model_key"] for r in rows] == ["sdxl", "sd15"]
    assert rows[1]["loaded_at"] == 100.0             # first dispatch time kept
    # bytes never exceed what the process measurably holds
    capped = led.residents(process_vram_bytes=3 * GIB)
    assert [r["bytes"] for r in capped] == [3 * GIB, 2 * GIB]
    # /free empties it and reports what was held
    assert led.note_freed() == ["sdxl", "sd15"]
    assert len(led) == 0 and led.freed_at == 102.0
    led.note_dispatch("", "x", 1)                     # no key: ignored
    assert len(led) == 0


def test_ledger_unknown_size_rows_do_not_count_bytes():
    led = L.ComfyLedger()
    led.note_dispatch("k", "k.safetensors", None)
    assert led.holds("k") and led.resident_bytes() == 0
    assert led.residents(process_vram_bytes=GIB)[0]["bytes"] is None


# ═══════════ agent seams ═══════════════════════════════════════════════════
@pytest.fixture
def fresh_ledger(monkeypatch):
    led = L.ComfyLedger()
    monkeypatch.setattr(A, "_COMFY_LEDGER", led)
    return led


def test_need_detail_prices_the_checkpoint_file(monkeypatch, tmp_path, fresh_ledger):
    (tmp_path / "big.safetensors").write_bytes(b"b" * 4096)
    monkeypatch.setenv("COMFY_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setenv("HUGPY_COMFY_GEN_CUSHION_GIB", "1")
    monkeypatch.setattr(A, "_comfy_checkpoint_filename", lambda mk: "big.safetensors")
    monkeypatch.setattr(A, "_comfy_process_vram", lambda *a, **k: None)
    d = A._comfy_need_detail(object(), "comfy-big")
    assert d["checkpoint"] == "big.safetensors"
    assert d["checkpoint_bytes"] == 4096
    assert d["held"] is False
    assert d["need"] == 4096 + GIB


def test_need_detail_held_when_ledger_and_process_agree(monkeypatch, tmp_path, fresh_ledger):
    (tmp_path / "big.safetensors").write_bytes(b"b" * 4096)
    monkeypatch.setenv("COMFY_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setenv("HUGPY_COMFY_GEN_CUSHION_GIB", "1")
    monkeypatch.setattr(A, "_comfy_checkpoint_filename", lambda mk: "big.safetensors")
    fresh_ledger.note_dispatch("comfy-big", "big.safetensors", 4096)
    # comfy's process covers the weights -> held -> cushion only
    monkeypatch.setattr(A, "_comfy_process_vram", lambda *a, **k: 5 * GIB)
    d = A._comfy_need_detail(object(), "comfy-big")
    assert d["held"] is True and d["need"] == GIB
    # comfy no longer backs the claim (restarted / freed behind our back):
    # the row is dropped and the full need applies
    monkeypatch.setattr(A, "_comfy_process_vram", lambda *a, **k: 0)
    d = A._comfy_need_detail(object(), "comfy-big")
    assert d["held"] is False and d["need"] == 4096 + GIB
    assert not fresh_ledger.holds("comfy-big")


def test_need_detail_unknown_file_is_unknown_need(monkeypatch, fresh_ledger):
    monkeypatch.delenv("COMFY_CHECKPOINTS_DIR", raising=False)
    monkeypatch.delenv("COMFY_CHECKPOINT_DIRS", raising=False)
    monkeypatch.setattr(A, "_comfy_checkpoint_filename", lambda mk: "nowhere.safetensors")
    monkeypatch.setattr(A, "_models_store_root", lambda: "/nonexistent/root")
    d = A._comfy_need_detail(object(), "comfy-x")
    assert d["checkpoint_bytes"] is None and d["need"] is None


def test_headroom_target_need_legacy_and_floor(monkeypatch):
    monkeypatch.delenv("HUGPY_COMFY_TARGET_FREE_GIB", raising=False)
    # known need governs; no implicit floor
    assert A._comfy_headroom_target({"need": 3 * GIB}) == 3 * GIB
    # unknown need -> the legacy constant, byte-identical to before
    assert A._comfy_headroom_target({"need": None}) == 7 * GIB
    # an EXPLICIT operator constant is a floor the need never undercuts...
    monkeypatch.setenv("HUGPY_COMFY_TARGET_FREE_GIB", "5")
    assert A._comfy_headroom_target({"need": 3 * GIB}) == 5 * GIB
    # ...and never a ceiling
    assert A._comfy_headroom_target({"need": 9 * GIB}) == 9 * GIB


def test_ensure_headroom_clears_this_checkpoints_need_and_records_it(monkeypatch, fresh_ledger):
    monkeypatch.delenv("HUGPY_COMFY_TARGET_FREE_GIB", raising=False)
    card = {"free": 1 * GIB}
    sizes = {"A": 2 * GIB, "B": 2 * GIB, "C": 2 * GIB}
    resident = set(sizes)
    evicted = []

    def _evict(state, mk, force=False):
        evicted.append(mk)
        resident.discard(mk)
        card["free"] += sizes[mk]
        return {"model_key": mk, "evicted": True, "host_mode": "slot"}
    monkeypatch.setattr(A, "_evict_model", _evict)
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: card["free"])
    monkeypatch.setattr(A, "_comfy_headroom_candidates",
                        lambda exclude: [mk for mk in sizes if mk in resident and mk != exclude])
    monkeypatch.setattr(A, "_comfy_need_detail", lambda s, mk: {
        "checkpoint": "sd15.safetensors", "checkpoint_bytes": 2 * GIB,
        "held": False, "need": 4 * GIB})
    res = A._worker_ensure_comfy_headroom(object(), "comfy-sd15", "job-1")
    # need 4 GiB from 1 free: evict A and B (2 each), C survives — not the 7 GiB
    # constant that would have taken C too
    assert evicted == ["A", "B"] and "C" in resident
    assert res["target"] == 4 * GIB and res["reached"] is True
    assert res["need"] == 4 * GIB and res["checkpoint"] == "sd15.safetensors"
    assert res["checkpoint_bytes"] == 2 * GIB and res["held"] is False
    # the dispatch is on the ledger now, sized
    assert fresh_ledger.residents()[0]["model_key"] == "comfy-sd15"
    assert fresh_ledger.residents()[0]["bytes"] == 2 * GIB


def test_ensure_headroom_held_checkpoint_needs_only_the_cushion(monkeypatch, fresh_ledger):
    monkeypatch.delenv("HUGPY_COMFY_TARGET_FREE_GIB", raising=False)
    evicted = []
    monkeypatch.setattr(A, "_evict_model",
                        lambda s, mk, force=False: evicted.append(mk) or {"evicted": True})
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: 3 * GIB)
    monkeypatch.setattr(A, "_comfy_headroom_candidates", lambda exclude: ["A"])
    monkeypatch.setattr(A, "_comfy_need_detail", lambda s, mk: {
        "checkpoint": "sd15.safetensors", "checkpoint_bytes": 2 * GIB,
        "held": True, "need": 2 * GIB})
    res = A._worker_ensure_comfy_headroom(object(), "comfy-sd15")
    assert evicted == [] and res["reached"] is True and res["held"] is True


def test_ensure_headroom_unknown_need_is_the_legacy_behaviour(monkeypatch, fresh_ledger):
    monkeypatch.delenv("HUGPY_COMFY_TARGET_FREE_GIB", raising=False)
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: 8 * GIB)
    monkeypatch.setattr(A, "_comfy_headroom_candidates", lambda exclude: [])
    monkeypatch.setattr(A, "_comfy_need_detail", lambda s, mk: {
        "checkpoint": None, "checkpoint_bytes": None, "held": False, "need": None})
    res = A._worker_ensure_comfy_headroom(object(), "comfy-?")
    assert res["target"] == 7 * GIB and res["reached"] is True
    assert "checkpoint" not in res and res["need"] is None
    # still recorded (comfy holds it), unsized
    assert fresh_ledger.holds("comfy-?")


def test_ensure_headroom_sizing_failure_falls_back(monkeypatch, fresh_ledger):
    monkeypatch.delenv("HUGPY_COMFY_TARGET_FREE_GIB", raising=False)
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: 8 * GIB)
    monkeypatch.setattr(A, "_comfy_headroom_candidates", lambda exclude: [])

    def boom(s, mk):
        raise RuntimeError("catalog down")
    monkeypatch.setattr(A, "_comfy_need_detail", boom)
    res = A._worker_ensure_comfy_headroom(object(), "comfy-x")
    assert res["target"] == 7 * GIB and res["reached"] is True


def test_free_models_empties_the_ledger_on_200(monkeypatch, fresh_ledger):
    fresh_ledger.note_dispatch("comfy-a", "a.safetensors", GIB)

    class _Resp:
        status_code = 200

    class _Httpx:
        @staticmethod
        def post(*a, **k):
            return _Resp()
    monkeypatch.setitem(__import__("sys").modules, "httpx", _Httpx)
    ok, note = A._comfy_free_models(object())
    assert ok and len(fresh_ledger) == 0
    # a refused /free keeps the ledger (comfy still holds it)
    fresh_ledger.note_dispatch("comfy-a", "a.safetensors", GIB)
    _Resp.status_code = 503
    ok, note = A._comfy_free_models(object())
    assert not ok and fresh_ledger.holds("comfy-a")


def test_vram_residents_names_comfy_from_the_ledger(monkeypatch, fresh_ledger):
    fresh_ledger.note_dispatch("comfy-a", "a.safetensors", 2 * GIB)
    fresh_ledger.note_dispatch("comfy-b", "b.safetensors", 6 * GIB)
    monkeypatch.setattr(A, "_slot_statuses", lambda: [])

    class _Reg:
        @staticmethod
        def snapshot_for_heartbeat():
            # the registry attributes comfy's PID to the ACTIVE call only
            return {"models": [{"model_key": "comfy-b", "vram_bytes": 5 * GIB,
                                "host_mode": "comfy", "alive": True}]}
    monkeypatch.setitem(__import__("sys").modules, "hugpy_fleet.worker.pid_registry", _Reg)
    monkeypatch.setattr(A, "_comfy_process_vram", lambda *a, **k: 5 * GIB)
    rows = {r["model_key"]: r for r in A._vram_residents(object())}
    assert set(rows) == {"comfy-a", "comfy-b"}
    assert rows["comfy-b"]["vram_bytes"] == 5 * GIB       # registry row stands
    assert rows["comfy-a"]["host_mode"] == "comfy"
    assert rows["comfy-a"]["vram_bytes"] == 2 * GIB       # ledger, capped by process
    # no comfy process behind the ledger -> it names nothing
    monkeypatch.setattr(A, "_comfy_process_vram", lambda *a, **k: None)
    monkeypatch.setitem(__import__("sys").modules, "hugpy_fleet.worker.pid_registry",
                        type("R", (), {"snapshot_for_heartbeat": staticmethod(lambda: {"models": []})}))
    assert A._vram_residents(object()) == []


def test_partition_idle_comfy_is_a_candidate_busy_comfy_is_protected(monkeypatch):
    rows = [{"model_key": "comfy-a", "vram_bytes": 2 * GIB, "host_mode": "comfy", "alive": True},
            {"model_key": "comfy-b", "vram_bytes": 6 * GIB, "host_mode": "comfy", "alive": True},
            {"model_key": "llm", "vram_bytes": 4 * GIB, "host_mode": "subprocess", "alive": True}]
    monkeypatch.setattr(A, "_vram_residents", lambda s: rows)
    monkeypatch.setattr(A, "_busy_slot_models", lambda: set())
    monkeypatch.setattr(A, "_queued_ahead_of", lambda mk: set())
    monkeypatch.setattr(A, "_residency", lambda mk: "on-demand")
    monkeypatch.setattr(A, "_actively_replying", lambda mk, busy: False)
    asked = []

    def busy(state):
        asked.append(1)
        return None
    monkeypatch.setattr(A, "_comfy_busy_reason", busy)
    cands, prot = A._partition_residents(object(), "subject")
    assert {c["model_key"] for c in cands} == {"comfy-a", "comfy-b", "llm"} and prot == []
    assert len(asked) == 1                                # asked once per partition

    monkeypatch.setattr(A, "_comfy_busy_reason", lambda s: "a comfy call is in flight (x)")
    cands, prot = A._partition_residents(object(), "subject")
    assert {c["model_key"] for c in cands} == {"llm"}
    assert {p["model_key"] for p in prot} == {"comfy-a", "comfy-b"}
    assert all(p["why"].startswith("comfy busy:") for p in prot)


def test_evict_model_comfy_branch_honours_the_busy_predicate(monkeypatch, fresh_ledger):
    monkeypatch.setattr(A, "_model_framework", lambda mk: "comfy")
    monkeypatch.setattr(A, "_evict_gate", lambda mk: (True, ""))
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: GIB)
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: GIB)
    monkeypatch.setattr(A, "loaded_model_keys", lambda: [])
    monkeypatch.setattr(A, "_model_footprint_before_evict", lambda *a, **k: {})
    freed = []
    monkeypatch.setattr(A, "_comfy_free_models", lambda s: freed.append(1) or (True, "ok"))
    monkeypatch.setattr(A, "_comfy_busy_reason", lambda s: "comfy /queue has work")
    res = A._evict_model(object(), "comfy-a")
    assert res["evicted"] is False and "comfy /queue has work" in res["reason"] and freed == []
    # idle -> freed; force -> freed regardless of busy
    monkeypatch.setattr(A, "_comfy_busy_reason", lambda s: None)
    assert A._evict_model(object(), "comfy-a")["evicted"] is True
    monkeypatch.setattr(A, "_comfy_busy_reason", lambda s: "rendering")
    assert A._evict_model(object(), "comfy-a", force=True)["evicted"] is True
    assert len(freed) == 2
