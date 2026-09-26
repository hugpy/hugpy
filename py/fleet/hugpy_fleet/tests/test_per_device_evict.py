"""Per-GPU evict-to-fit: on a multi-GPU box, only residents on the TARGET card
are candidates — evicting a sibling-card model frees the wrong VRAM.

Operator 2026-09-25: hugpy decides the device end to end, so the make-room pass
must reason per device. A resident of a KNOWN other card is protected (with an
honest reason); one of UNKNOWN card stays a candidate (degrade-not-guess); and a
single-GPU / unpinned box behaves exactly as before.

Run:  python3 -m pytest tests/test_per_device_evict.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-perdev-evict-")
os.environ.setdefault("HUGPY_COMMS_DB", "off")
os.environ["HUGPY_COLD_HOLD_HEALTH"] = "off"

import hugpy_fleet.worker.agent as ag  # noqa: E402

GIB = 2 ** 30


def _residents():
    return [
        {"model_key": "on0", "vram_bytes": 5 * GIB, "host_mode": "subprocess",
         "alive": True, "gpu_index": 0},
        {"model_key": "on2", "vram_bytes": 5 * GIB, "host_mode": "subprocess",
         "alive": True, "gpu_index": 2},          # the target card
        {"model_key": "on3", "vram_bytes": 5 * GIB, "host_mode": "in_process",
         "alive": True, "gpu_index": 3},
        {"model_key": "unk", "vram_bytes": 5 * GIB, "host_mode": "subprocess",
         "alive": True},                          # unknown card
    ]


def _permissive(monkeypatch, *, gpus, target):
    monkeypatch.setattr(ag, "_vram_residents", lambda state: _residents())
    monkeypatch.setattr(ag, "detect_gpus",
                        lambda: [{"index": i} for i in range(gpus)])
    monkeypatch.setattr(ag, "_target_device_index", lambda: target)
    monkeypatch.setattr(ag, "_busy_slot_models", lambda: set())
    monkeypatch.setattr(ag, "_queued_ahead_of", lambda mk: set())
    monkeypatch.setattr(ag, "_residency", lambda mk: "on_demand")
    monkeypatch.setattr(ag, "_actively_replying", lambda mk, busy: False)
    monkeypatch.setattr(ag, "_comfy_busy_reason", lambda state: None)


def test_only_target_card_and_unknown_are_candidates(monkeypatch):
    _permissive(monkeypatch, gpus=4, target=2)
    cands, protected = ag._partition_residents(None, "incoming")
    cand_keys = {c["model_key"] for c in cands}
    prot_keys = {p["model_key"] for p in protected}
    assert cand_keys == {"on2", "unk"}, cand_keys           # target card + unknown
    assert prot_keys == {"on0", "on3"}, prot_keys           # sibling cards protected
    for p in protected:
        if p["model_key"] in ("on0", "on3"):
            assert "per-device evict" in p["why"], p


def test_single_gpu_box_never_filters_by_device(monkeypatch):
    # One card -> all residents are candidates (byte-identical to before), even
    # though the rows carry stale gpu_index values.
    _permissive(monkeypatch, gpus=1, target=0)
    cands, protected = ag._partition_residents(None, "incoming")
    assert {c["model_key"] for c in cands} == {"on0", "on2", "on3", "unk"}
    assert protected == []


def test_unpinned_target_never_filters(monkeypatch):
    # Multi-GPU but no target pin (older central) -> no per-device filtering.
    _permissive(monkeypatch, gpus=4, target=None)
    cands, _ = ag._partition_residents(None, "incoming")
    assert {c["model_key"] for c in cands} == {"on0", "on2", "on3", "unk"}


def _main() -> int:
    class _MP:
        def __init__(self): self._u = []
        def setattr(self, o, n, v): self._u.append((o, n, getattr(o, n))); setattr(o, n, v)
        def undo(self):
            for o, n, v in reversed(self._u): setattr(o, n, v)
            self._u = []
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    ok = fail = 0
    for name, fn in tests:
        mp = _MP()
        try:
            fn(mp)
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        else:
            ok += 1
            print(f"[ok]   {name}")
        finally:
            mp.undo()
    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(_main())
