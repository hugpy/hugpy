"""item 2 — the slot child port must never collide with the worker port.

slot_agent's child (llama-server) port defaults to SLOT_PORT + 1000. With the
shipped SLOT_PORT_BASE (8101) slot 1's child is 9101 — the worker's own port on
a box whose unit sets --port 9101 (a-brain: the child tried to bind 9101 == the
worker port -> EADDRINUSE, no model ever loaded; the by-hand fix was
SLOT_PORT_BASE=8201). The guard relocates a colliding CHILD upward, loudly, and
records it. slot_agent reads env at import, so each case runs in a fresh
subprocess.

Run: `PYTHONPATH=$(ls -d py/*/*/src|tr '\n' :) pytest tests/test_slot_port_guard.py -q`
"""
import os
import subprocess
import sys
from pathlib import Path

SRC_DIRS = [str(p) for p in Path(__file__).resolve().parents[4].glob("py/*/*/src")]


def _probe(env_extra: dict) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(SRC_DIRS + [env.get("PYTHONPATH", "")])
    env.update(env_extra)
    code = ("import json, hugpy_engine.serve.slot_agent as sa;"
            "print(json.dumps({'port': sa.SLOT_PORT, 'child': sa.SLOT_CHILD_PORT,"
            "'reloc': sa.PORT_RELOCATIONS}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    import json
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_child_relocates_off_colliding_worker_port():
    r = _probe({"SLOT_ID": "1", "SLOT_PORT_BASE": "8101", "WORKER_PORT": "9101"})
    assert r["port"] == 8101
    assert r["child"] != 9101, "child must not sit on the worker port"
    assert r["reloc"], "relocation must be recorded"


def test_no_relocation_when_default_worker_port():
    r = _probe({"SLOT_ID": "1", "SLOT_PORT_BASE": "8101", "WORKER_PORT": "9100"})
    assert r["child"] == 9101 and r["reloc"] == []


def test_child_never_equals_control_port():
    # Force the pathological SLOT_CHILD_PORT == SLOT_PORT and confirm it moves.
    r = _probe({"SLOT_ID": "1", "SLOT_PORT": "8500", "SLOT_CHILD_PORT": "8500"})
    assert r["child"] != 8500 and r["reloc"]
