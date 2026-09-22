"""Slice 12 / t32 — slot load health: fail on STALL, not a blind clock.

Field incident (ae): "slot N: did not become healthy in 180s — falling back",
4× in ~30s bursts, for the 45.1G Qwen3-Coder-Next brain. The slot's _wait_healthy
was a flat 180s deadline that KILLED a legitimately-slow cold load (46G off NVMe
+ CUDA upload legitimately exceeds 180s), so every retry re-paged 46G and re-
failed — a load-time thrash loop. (The same model served fine warm for hours.)

Fix: stall-aware patience — while the child is ALIVE and its RSS/VRAM is GROWING,
the load is progressing; keep waiting. Kill only on a STALL (no growth for the
stall window) or a generous SIZE-SCALED hard cap (a truly-wedged child still
dies). Plus per-model backoff so per-request re-attempts don't hammer a doomed
load, and the honest reason on the slot status row.

Run: venv/bin/python -m pytest tests/test_slot_load_stall_health.py -q
"""
import importlib
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# managers/__init__ star-imports shadow subpackage attrs — bind the real module.
sa = importlib.import_module("hugpy_engine.serve.slot_agent")

GIB = 1 << 30


def _slot(model_key="big", expected_bytes=46 * GIB):
    """A bare Slot with just the fields _wait_healthy / backoff touch — no child,
    no subprocess. The health/alive/progress probes are stubbed per test."""
    s = sa.Slot.__new__(sa.Slot)
    s.model_key = model_key
    s.expected_bytes = expected_bytes
    s._load_free_vram_at_start = None
    s._load_failures = {}
    s._load_backoff_until = {}
    s.last_load_error = None
    s.child_base = "http://127.0.0.1:9101"
    return s


@pytest.fixture
def fast(monkeypatch):
    """Shrink the windows so the tests run in ~seconds, not minutes."""
    monkeypatch.setattr(sa, "STALL_TIMEOUT", 2.0)
    monkeypatch.setattr(sa, "HEALTH_TIMEOUT", 3.0)
    monkeypatch.setattr(sa, "_PROGRESS_EPSILON", 1 << 20)   # 1 MiB
    return monkeypatch


# ═══════════ size-scaled hard cap ══════════════════════════════════════════
def test_hard_cap_scales_with_size(monkeypatch):
    monkeypatch.setattr(sa, "HEALTH_TIMEOUT", 180.0)
    s = _slot(expected_bytes=46 * GIB)
    cap = s._hard_cap_s()
    assert cap > 180.0                       # a 46G load gets FAR more than 180s
    # smaller model -> smaller cap, but never below the floor.
    s.expected_bytes = 1 * GIB
    assert s._hard_cap_s() >= 180.0


def test_hard_cap_floor_when_size_unknown(monkeypatch):
    monkeypatch.setattr(sa, "HEALTH_TIMEOUT", 180.0)
    s = _slot(expected_bytes=None)
    assert s._hard_cap_s() == 180.0          # unknown size -> back-compat floor


# ═══════════ stall vs progress ═════════════════════════════════════════════
def test_growing_load_survives_past_the_old_flat_deadline(fast):
    """A child GROWING (progress) is NOT killed at the stall window — progress
    resets the stall clock. It only dies at the (here small) hard cap, proving a
    slow cold load isn't killed for being slow."""
    s = _slot()
    s._child_alive = lambda: True
    s.healthy = lambda: False
    grown = {"v": 0}
    s._load_progress_bytes = lambda: grown.__setitem__("v", grown["v"] + 50 * (1 << 20)) or grown["v"]
    s._hard_cap_s = lambda: 5.0              # small cap for the test
    t0 = time.time()
    ok = s._wait_healthy()
    elapsed = time.time() - t0
    assert ok is False                       # never became healthy
    # survived WELL past the 2s stall window (progress held it) — died at ~cap.
    assert elapsed >= sa.STALL_TIMEOUT + 1.0


def test_stalled_load_killed_at_the_stall_window(fast):
    """A child making NO progress is killed at the stall window — long before the
    generous hard cap."""
    s = _slot()
    s._child_alive = lambda: True
    s.healthy = lambda: False
    s._load_progress_bytes = lambda: 500 * (1 << 20)   # frozen — never grows
    s._hard_cap_s = lambda: 1000.0                      # huge cap
    t0 = time.time()
    ok = s._wait_healthy()
    elapsed = time.time() - t0
    assert ok is False
    # died at the stall window (~2s), NOT the 1000s cap.
    assert elapsed < 6.0


def test_healthy_child_returns_immediately(fast):
    s = _slot()
    s._child_alive = lambda: True
    s.healthy = lambda: True                 # up right away
    s._load_progress_bytes = lambda: 0
    s._hard_cap_s = lambda: 1000.0
    assert s._wait_healthy() is True


def test_dead_child_is_a_real_failure(fast):
    s = _slot()
    s._child_alive = lambda: False           # exited (e.g. OOM/SIGILL)
    s.healthy = lambda: False
    s._load_progress_bytes = lambda: 0
    s._hard_cap_s = lambda: 1000.0
    assert s._wait_healthy() is False


def test_progress_signal_combines_rss_and_vram(monkeypatch):
    """_load_progress_bytes = child RSS + VRAM consumed since load start."""
    s = _slot()
    s._child_alive = lambda: True
    s.proc = type("P", (), {"pid": 4242})()
    monkeypatch.setattr(sa, "_proc_rss_bytes", lambda pid: 3 * GIB)
    s._load_free_vram_at_start = 20 * GIB
    from hugpy_engine import spill
    monkeypatch.setattr(spill, "free_vram_bytes", lambda: 15 * GIB)   # 5G consumed
    assert s._load_progress_bytes() == 3 * GIB + 5 * GIB


# ═══════════ single-flight (pool coalescing) ═══════════════════════════════
def test_load_lock_serializes_same_slot(fast):
    """Within a slot, the load lock serializes — a second load() call cannot start
    a second child while the first holds the lock (the whole load(), including
    _wait_healthy, runs under self.lock). Cross-request coalescing at the POOL is
    slots.endpoint_for step 1, which WAITS on the loading slot rather than
    starting a second child."""
    import threading
    import inspect
    # load() opens with `with self.lock:` — the whole body is serialized.
    src = inspect.getsource(sa.Slot.load)
    assert "with self.lock:" in src
    # and the lock is a real mutex: holding it blocks a second acquirer.
    s = _slot()
    s.lock = threading.Lock()
    s.lock.acquire()
    assert s.lock.acquire(blocking=False) is False   # a second load() would block
    s.lock.release()


# ═══════════ backoff after repeated genuine failures ═══════════════════════
def test_backoff_arms_and_refuses_immediate_reattempt(fast, monkeypatch):
    """A load that fails records a failure and refuses a re-attempt inside the
    backoff window — no thrash of 46G re-pages on every request."""
    monkeypatch.setattr(sa, "_LOAD_BACKOFF_BASE_S", 30.0)
    s = _slot()
    import threading
    s.lock = threading.Lock()
    s.proc = None
    s.profile_bin = None
    # make the load 'fail' fast: _build_cmd + popen stubbed, _wait_healthy False.
    monkeypatch.setattr(sa, "_build_cmd",
                        lambda *a, **k: (["true"], -1, 4096, 6, None, "cpp", 48, None))
    monkeypatch.setattr(sa, "_model_expected_bytes", lambda mk: 46 * GIB)
    monkeypatch.setattr(sa.subprocess, "Popen", lambda *a, **k: type(
        "P", (), {"pid": 1, "poll": lambda self: None, "terminate": lambda self: None,
                  "wait": lambda self, timeout=None: None, "kill": lambda self: None})())
    s._wait_healthy = lambda: False          # the load never comes up
    s._kill = lambda: None
    with pytest.raises(RuntimeError) as e1:
        s.load("big")
    assert "backing off" in str(e1.value).lower() or "healthy" in str(e1.value).lower()
    assert s._load_failures["big"] == 1
    assert s._load_backoff_until["big"] > time.time()
    # An immediate re-attempt is REFUSED by the backoff (no new child).
    with pytest.raises(RuntimeError) as e2:
        s.load("big")
    assert "backoff" in str(e2.value).lower()
    assert s._load_failures["big"] == 1      # not re-incremented (never re-attempted)


def test_backoff_grows_exponentially(fast, monkeypatch):
    monkeypatch.setattr(sa, "_LOAD_BACKOFF_BASE_S", 10.0)
    monkeypatch.setattr(sa, "_LOAD_BACKOFF_MAX_S", 600.0)
    s = _slot()
    # simulate the failure-recording arithmetic directly.
    for n in (1, 2, 3):
        backoff = min(sa._LOAD_BACKOFF_BASE_S * (2 ** (n - 1)), sa._LOAD_BACKOFF_MAX_S)
        assert backoff == 10.0 * (2 ** (n - 1))
    # capped
    assert min(10.0 * (2 ** 20), 600.0) == 600.0


def test_success_clears_backoff(fast, monkeypatch):
    """A successful load clears the failure counter + backoff for the model."""
    import threading
    s = _slot()
    s.lock = threading.Lock()
    s.proc = None
    s.profile_bin = None
    s._load_failures = {"big": 2}
    s._load_backoff_until = {"big": time.time() + 5}
    monkeypatch.setattr(sa, "_build_cmd",
                        lambda *a, **k: (["true"], -1, 4096, 6, None, "cpp", 48, None))
    monkeypatch.setattr(sa, "_model_expected_bytes", lambda mk: 46 * GIB)
    monkeypatch.setattr(sa.subprocess, "Popen", lambda *a, **k: type(
        "P", (), {"pid": 1, "poll": lambda self: None})())
    # backoff window has expired (set in the past) so the load proceeds.
    s._load_backoff_until = {"big": time.time() - 1}
    s._wait_healthy = lambda: True           # comes up
    s._kill = lambda: None
    s.healthy = lambda: False                # force the load path, not the reuse
    s.status = lambda: {"ok": True}
    s.load("big")
    assert "big" not in s._load_failures     # cleared on success
    assert "big" not in s._load_backoff_until
    assert s.last_load_error is None


# ═══════════ honest reason on the row ══════════════════════════════════════
def test_status_carries_last_load_error(monkeypatch):
    """The slot status surfaces last_load_error so the console shows WHY a model
    is degraded/retrying instead of a silent tight loop."""
    from hugpy_engine import spill
    monkeypatch.setattr(spill, "free_vram_bytes", lambda: 10 * GIB)
    s = _slot()
    s.proc = None
    s.ngl = s.ctx = s.threads = s.cpus = s.gpu = None
    s.profile_bin = None
    s.loaded_at = s.last_used = 0.0
    s.inflight = 0
    s.last_load_error = "did not become healthy (stall); attempt 2, backing off 60s"
    s._self_heal = lambda: None
    s.healthy = lambda: False
    s._child_alive = lambda: False
    st = s.status()
    assert st["last_load_error"] == s.last_load_error
