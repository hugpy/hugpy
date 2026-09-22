"""Central VRAM-squatter aggregation (incident 2026-08-05).

The worker's ``vram_holders`` meter rides the heartbeat; central re-shapes the
idle own-venv squatters it flagged into a per-worker ``vram_squatters`` summary
(surfaced in _public_view / GET /llm/workers) and shouts ONE warning per distinct
squatter (dedup by worker+pid, like the 'reclaimable collapse' warning) so the
leak is never silent.

Runs like the other central tests: venv/bin/python tests/test_vram_squatters_central.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-vram-squat-test-")

import importlib
wk = importlib.import_module(
    "hugpy_fleet.central.workers")

MIB = 1 << 20
ok = 0


def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


def _meter(squatter_pids=(4242,)):
    """A worker record carrying a vram_holders meter with the given squatters."""
    squatters = [{"pid": p, "name": "/opt/hugpy/venv/bin/python",
                  "vram_bytes": 8000 * MIB, "age_s": 900.0, "kind": "own-orphan",
                  "reapable": True, "squatter": True,
                  "reason": ("holds VRAM, no live slot claim, not serving — "
                             "reapable but not auto-reaped")}
                 for p in squatter_pids]
    return {
        "ok": True,
        "cards": [{"index": 0, "mem_total_bytes": 24000 * MIB,
                   "mem_used_bytes": 20000 * MIB, "mem_free_bytes": 4000 * MIB,
                   "util_pct": 0}],
        "gpu_util_pct": 0,
        "holders": list(squatters),
        "squatters": squatters,
        "min_age_s": 300.0,
    }


def test_vram_squatters_central():
    import logging as _lg
    _lg.disable(_lg.NOTSET)
    # ── _vram_squatters re-shapes the worker's flagged squatters ─────────────────
    worker = {"id": "w-op", "name": "op", "vram_holders": _meter((4242,))}
    rows = wk._vram_squatters(worker)
    check("one squatter row surfaced", len(rows) == 1)
    r = rows[0]
    check("row carries worker identity", r["worker"] == "op" and r["worker_id"] == "w-op")
    check("row carries pid", r["pid"] == 4242)
    check("row carries vram_bytes", r["vram_bytes"] == 8000 * MIB)
    check("row carries age", r["age_s"] == 900.0)
    check("row carries the explicit reason", "not auto-reaped" in r["reason"])


    # ── warning dedup: one warning per distinct squatter (by worker+pid) ─────────
    import logging


    class _Capture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.warnings = []

        def emit(self, record):
            if record.levelno >= logging.WARNING:
                self.warnings.append(record.getMessage())


    cap = _Capture()
    wk.logger.addHandler(cap)
    wk.logger.setLevel(logging.DEBUG)
    # Reset the module dedup state for a clean count.
    wk._VRAM_SQUATTER_SEEN.clear()

    w2 = {"id": "w-ae", "name": "ae", "vram_holders": _meter((9001,))}
    wk._vram_squatters(w2)
    wk._vram_squatters(w2)          # steady state — must NOT re-warn
    wk._vram_squatters(w2)
    squat_warns = [m for m in cap.warnings if "VRAM SQUATTER" in m]
    check("warns exactly once per distinct squatter", len(squat_warns) == 1)
    check("warning names the worker + pid", "ae" in squat_warns[0] and "9001" in squat_warns[0])

    # A NEW distinct squatter on the same worker warns again (once).
    cap.warnings.clear()
    wk._vram_squatters({"id": "w-ae", "name": "ae", "vram_holders": _meter((9001, 9002))})
    new_warns = [m for m in cap.warnings if "VRAM SQUATTER" in m]
    check("a new distinct squatter warns once more", len(new_warns) == 1 and "9002" in new_warns[0])

    # When the squatter clears, the pid is forgotten so a later reappearance re-warns.
    cap.warnings.clear()
    wk._vram_squatters({"id": "w-ae", "name": "ae", "vram_holders": _meter(())})  # gone
    wk._vram_squatters({"id": "w-ae", "name": "ae", "vram_holders": _meter((9001,))})  # back
    back_warns = [m for m in cap.warnings if "VRAM SQUATTER" in m]
    check("a reappearing squatter re-warns", len(back_warns) == 1)
    wk.logger.removeHandler(cap)


    # ── a worker with no meter / no squatters yields nothing, never raises ───────
    check("no meter -> empty", wk._vram_squatters({"id": "x", "name": "x"}) == [])
    check("meter with no squatters -> empty",
          wk._vram_squatters({"id": "y", "name": "y",
                              "vram_holders": _meter(())}) == [])


    # ── _public_view surfaces vram_squatters + passes vram_holders through ───────
    pv = wk._public_view({"id": "w-op", "name": "op", "last_seen": wk._now(),
                          "vram_holders": _meter((4242,))})
    check("_public_view carries vram_holders verbatim",
          isinstance(pv.get("vram_holders"), dict)
          and pv["vram_holders"]["gpu_util_pct"] == 0)
    check("_public_view derives vram_squatters",
          len(pv.get("vram_squatters") or []) == 1
          and pv["vram_squatters"][0]["pid"] == 4242)

    # A pre-feature worker (no meter) has an EMPTY squatter list, never a KeyError.
    pv2 = wk._public_view({"id": "old", "name": "old", "last_seen": wk._now()})
    check("pre-feature worker -> empty squatters", pv2.get("vram_squatters") == [])


