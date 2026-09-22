"""k14 — controllable GPU-offload depth per slot relaunch (the k7 speed-cliff lever).

The offload speed-cliff sweep needs each GGUF seated at full offload, then swept
DOWN through decreasing n_gpu_layers, tok/s measured at each step. Two slot-side
blockers this proves fixed:

  1. the slot (re)launch path must HONOR an explicit n_gpu_layers (override wins
     over autofit; None/absent => autofit). The sentinel matters: -1 is the live
     "Max GPU" designation (force all layers), NOT an autofit alias — the sweep
     asks for autofit with None and explicit non-negative counts below it.
  2. there must be a way to RELAUNCH a live slot's child with a new spec — a
     same-model /load short-circuits (already-serving), so the sweep could never
     change the offload depth. relaunch() forces a STOP->RESPAWN under a new pid.

Run: venv/bin/python -m pytest tests/test_slot_relaunch.py -q
"""
import importlib
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sa = importlib.import_module("hugpy_engine.serve.slot_agent")

GIB = 1 << 30


# ═══════════ override-wins-over-autofit decision ═══════════════════════════
def test_effective_ngl_none_is_autofit():
    # absent/None => whatever autofit decided (here 20, or -1 for "all fit").
    assert sa._effective_ngl(None, 20) == 20
    assert sa._effective_ngl(None, -1) == -1


def test_effective_ngl_explicit_count_wins_over_autofit():
    # the sweep step: an explicit layer count OVERRIDES autofit.
    assert sa._effective_ngl(17, -1) == 17
    assert sa._effective_ngl(0, 33) == 0            # CPU-only override wins
    assert sa._effective_ngl(48, 12) == 48


def test_effective_ngl_minus_one_is_max_gpu_not_autofit():
    """-1 is the live 'Max GPU' designation (force all layers) — an EXPLICIT
    override, NOT an autofit alias. Aliasing it to autofit would regress
    managers.llama.runners.get, which ships n_gpu_layers=-1 to force all layers."""
    # requested -1 wins over an autofit that would have chosen a partial count.
    assert sa._effective_ngl(-1, 12) == -1


def test_effective_ngl_coerces_str_int():
    assert sa._effective_ngl("24", -1) == 24


# ═══════════ relaunch plumbing (spec threading) ════════════════════════════
def _slot_for_relaunch(model_key="coder"):
    s = sa.Slot.__new__(sa.Slot)
    s.model_key = model_key
    s.ngl = -1
    s.ctx = 4096
    s.threads = 6
    s.cpus = "0-3"
    s.gpu = "0"
    s.profile_bin = "/opt/prof/bin"
    s._load_failures = {}
    s._load_backoff_until = {}
    s.lock = threading.Lock()
    return s


def test_relaunch_forces_load_with_new_ngl_preserving_spec():
    s = _slot_for_relaunch()
    seen = {}

    def _fake_load(model_key, **kw):
        seen["model_key"] = model_key
        seen.update(kw)
        # emulate the honest launched allocation the real load()/status returns.
        return {"model_key": model_key, "n_gpu_layers": kw.get("n_gpu_layers"),
                "ctx": kw.get("ctx"), "healthy": True, "child_pid": 4242}

    s.load = _fake_load
    out = s.relaunch(n_gpu_layers=17, ctx=8192)

    assert seen["model_key"] == "coder"             # same model re-seated
    assert seen["n_gpu_layers"] == 17               # the swept-down override
    assert seen["ctx"] == 8192                       # ctx override
    assert seen["force"] is True                     # bypasses the short-circuit
    # spec preserved from the live slot (not dropped on the floor).
    assert seen["threads"] == 6
    assert seen["cpus"] == "0-3"
    assert seen["gpu"] == "0"
    assert seen["profile_bin"] == "/opt/prof/bin"
    # honest echo: requested carried alongside, relaunched flagged.
    assert out["relaunched"] is True
    assert out["requested_n_gpu_layers"] == 17
    assert out["n_gpu_layers"] == 17


def test_relaunch_keeps_current_ctx_when_omitted():
    s = _slot_for_relaunch()
    seen = {}
    s.load = lambda mk, **kw: seen.update(kw) or {"model_key": mk}
    s.relaunch(n_gpu_layers=8)                        # no ctx given
    assert seen["ctx"] == 4096                        # kept the live ctx
    assert seen["n_gpu_layers"] == 8


def test_relaunch_ngl_none_re_autofits():
    s = _slot_for_relaunch()
    seen = {}
    s.load = lambda mk, **kw: seen.update(kw) or {"model_key": mk}
    s.relaunch(n_gpu_layers=None)                     # top-of-ramp: full offload
    assert seen["n_gpu_layers"] is None              # => slot re-autofits


def test_relaunch_empty_slot_raises():
    s = _slot_for_relaunch(model_key=None)
    with pytest.raises(RuntimeError) as e:
        s.relaunch(n_gpu_layers=4)
    assert "no model loaded" in str(e.value).lower()


def test_relaunch_clears_stale_backoff():
    """A deliberate relaunch must not be refused by a load-backoff armed by an
    earlier failure of this model — relaunch clears it before the forced load."""
    s = _slot_for_relaunch()
    s._load_failures = {"coder": 3}
    s._load_backoff_until = {"coder": 9e18}
    s.load = lambda mk, **kw: {"model_key": mk}
    s.relaunch(n_gpu_layers=4)
    assert "coder" not in s._load_failures
    assert "coder" not in s._load_backoff_until


# ═══════════ force bypasses the already-serving short-circuit ══════════════
def test_force_bypasses_same_model_short_circuit(monkeypatch):
    """load(force=True) must RESPAWN even for the same, healthy model — otherwise
    a same-model relaunch is a silent no-op and the sweep can't change depth."""
    s = sa.Slot.__new__(sa.Slot)
    s.model_key = "coder"
    s.lock = threading.Lock()
    s.proc = None
    s.ngl = None
    s.ctx = None
    s.threads = None
    s.cpus = None
    s.child_kind = None
    s.gpu = None
    s.expected_bytes = None
    s.loaded_at = s.last_used = 0.0
    s.child_base = "http://127.0.0.1:9101"
    s.profile_bin = None
    s._load_failures = {}
    s._load_backoff_until = {}
    s.last_load_error = None
    s.healthy = lambda: True                          # already serving + healthy
    s._self_heal = lambda: None

    spawned = {"n": 0}

    def _build(mk, ngl, ctx, threads, cpus, **kw):
        # mirrors the real return shape: (..., child_kind, total_layers,
        # n_cpu_moe) — block_count feeds status()'s "17/48 layers" readout and
        # the trailing n_cpu_moe is the MoE expert-split the child launched with.
        return (["true"], ngl if ngl is not None else -1, ctx or 4096,
                threads or 6, cpus, "cpp", 48, kw.get("n_cpu_moe"))
    monkeypatch.setattr(sa, "_build_cmd", _build)
    monkeypatch.setattr(sa, "_model_expected_bytes", lambda mk: 1 * GIB)

    def _popen(*a, **k):
        spawned["n"] += 1
        return type("P", (), {"pid": 1, "poll": lambda self: None})()
    monkeypatch.setattr(sa.subprocess, "Popen", _popen)
    s._kill = lambda: None
    s._wait_healthy = lambda: True
    s.status = lambda: {"model_key": s.model_key, "n_gpu_layers": s.ngl}

    # force=False: same healthy model => short-circuit, NO respawn.
    s.load("coder", n_gpu_layers=10, force=False)
    assert spawned["n"] == 0

    # force=True: respawn with the new depth.
    out = s.load("coder", n_gpu_layers=10, force=True)
    assert spawned["n"] == 1
    assert s.ngl == 10                                # launched with the override
    assert out["n_gpu_layers"] == 10


# ═══════════ the /relaunch HTTP route ══════════════════════════════════════
def test_relaunch_route_empty_slot_is_409(monkeypatch):
    monkeypatch.setattr(sa, "SLOT_ID", "1", raising=False)
    app, slot = sa.build_app()
    slot.model_key = None                             # nothing seated
    client = app.test_client()
    r = client.post("/relaunch", json={"n_gpu_layers": 8})
    assert r.status_code == 409
    assert "no model loaded" in (r.get_json() or {}).get("error", "").lower()


def test_relaunch_route_relays_to_slot(monkeypatch):
    app, slot = sa.build_app()
    slot.model_key = "coder"
    captured = {}
    slot.relaunch = lambda ngl, ctx, n_cpu_moe=None: captured.update(
        ngl=ngl, ctx=ctx) or {
        "model_key": "coder", "n_gpu_layers": ngl, "relaunched": True}
    client = app.test_client()
    r = client.post("/relaunch", json={"n_gpu_layers": 12, "ctx": 2048})
    assert r.status_code == 200
    assert captured == {"ngl": 12, "ctx": 2048}
    assert r.get_json()["n_gpu_layers"] == 12
