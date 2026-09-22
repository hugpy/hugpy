"""Provisioning thundering-herd gate (root-caused live 2026-07-15).

Before this fix, ``_sync_assignment`` (worker adopts central's assignment
list) fired ``_kick_provision`` for EVERY assigned model in one pass, and each
kick spawned its own background thread running ``ensure_model_present`` — a
segmented/parallel download in its own right. A big assignment list (e.g. 30
models) therefore launched 30 concurrent multi-threaded pulls against central
at once: a thundering herd that 503'd central almost every chunk and never
converged (observed live: "not even 6% of a single download in").

The fix adds a fleet-wide provisioning concurrency cap on ``WorkerState``
(default 1, env-overridable via WORKER_PROVISION_CONCURRENCY) enforced by a
``threading.BoundedSemaphore`` acquired around the existing, untouched
``_bg`` body inside ``_kick_provision``. This test exercises ONLY the gate:
it stubs ``ensure_model_present`` with a fake that blocks until released and
records concurrency, drives ``_kick_provision`` directly for several models,
and asserts:

  * observed max concurrency == the configured limit (default 1; then 2 via
    the env var),
  * every kicked model eventually completes — none silently dropped,
  * the pre-existing per-model ``_provisioning`` dedupe guard still holds
    (kicking the same key twice while it's in flight is a no-op).

No real network/downloads: ``ensure_model_present`` and ``model_is_local``
are monkeypatched at the ``provision`` module (the ``_bg`` body imports them
locally by name at call time, so patching the module attribute is enough —
no need to touch ``agent``'s namespace). ``central_url=None`` keeps the
comfy-checkpoint / ensure_model_registered probing inside ``_bg`` a fast,
network-free no-op for made-up model keys (it excepts out and falls through
to the normal flow, which is what we're gating).

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_provision_concurrency_gate.py -q
"""
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from hugpy_fleet.worker import agent as A
from hugpy_storage import provision as P


class _FakeProvisioner:
    """Records concurrency + completion for a stubbed ensure_model_present.

    Blocks each call on a shared gate Event until the test releases it, so we
    can observe the HIGH-WATER concurrency mid-flight before letting anything
    finish."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current = 0
        self.max_concurrency = 0
        self.started = []      # model keys that entered the fake call
        self.completed = []    # model keys that returned from the fake call
        self.gate = threading.Event()

    def __call__(self, model_key, central_url, progress=None, state=None,
                 purpose=None):
        # state= (storage budget) and purpose= (central budget handshake,
        # 2026-07-17) are threaded through _kick_provision's real call; accept
        # them so this concurrency stub matches the live ensure_model_present.
        with self.lock:
            self.current += 1
            self.max_concurrency = max(self.max_concurrency, self.current)
            self.started.append(model_key)
        # Bounded wait: never hang the test suite if the gate logic is broken
        # and a call never gets scheduled at all (would otherwise deadlock).
        self.gate.wait(timeout=10)
        with self.lock:
            self.current -= 1
            self.completed.append(model_key)
        return True


@pytest.fixture(autouse=True)
def _stub_provision(monkeypatch):
    """Fresh fake + a not-local stub so _bg always takes the provision path."""
    fake = _FakeProvisioner()
    monkeypatch.setattr(P, "ensure_model_present", fake)
    monkeypatch.setattr(P, "model_is_local", lambda mk: False)
    return fake


def _make_state(**kw) -> A.WorkerState:
    return A.WorkerState(name="test-worker", url=None, worker_id="w-test",
                          central_url=None, **kw)


def _wait_until(cond, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return cond()


# ── default cap (1 = fully serial) ──────────────────────────────────────────
def test_default_concurrency_cap_is_one(monkeypatch, _stub_provision):
    monkeypatch.delenv("WORKER_PROVISION_CONCURRENCY", raising=False)
    state = _make_state()
    assert state.provision_concurrency == 1

    models = [f"model-{i}" for i in range(6)]
    for mk in models:
        A._kick_provision(state, mk)

    # Let the herd pile up against the gate; with a cap of 1 only ONE call
    # should ever be in-flight no matter how long we wait before releasing.
    assert _wait_until(lambda: len(_stub_provision.started) >= 1)
    time.sleep(0.3)  # give any (bug-case) extra threads a chance to pile on
    assert _stub_provision.max_concurrency == 1

    _stub_provision.gate.set()
    assert _wait_until(lambda: len(_stub_provision.completed) == len(models))
    assert sorted(_stub_provision.completed) == sorted(models)
    # Every kicked model is eventually released from the in-flight guard too.
    # The fake's own completion (recorded inside the stubbed call) races the
    # REST of _bg's cleanup (comfy/preload branches, then the `finally:
    # state._provisioning.discard(mk)`), which runs a moment later on the
    # same background thread — so wait for the guard to actually drain
    # rather than asserting on it the instant `completed` fills up.
    assert _wait_until(lambda: state._provisioning == set())


# ── env override (2 concurrent) ─────────────────────────────────────────────
def test_env_override_raises_the_cap(monkeypatch, _stub_provision):
    monkeypatch.setenv("WORKER_PROVISION_CONCURRENCY", "2")
    state = _make_state()
    assert state.provision_concurrency == 2

    models = [f"model-{i}" for i in range(6)]
    for mk in models:
        A._kick_provision(state, mk)

    assert _wait_until(lambda: len(_stub_provision.started) >= 2)
    time.sleep(0.3)
    assert _stub_provision.max_concurrency == 2

    _stub_provision.gate.set()
    assert _wait_until(lambda: len(_stub_provision.completed) == len(models))
    assert sorted(_stub_provision.completed) == sorted(models)


# ── invalid / absent env values fall back to 1 ──────────────────────────────
@pytest.mark.parametrize("raw", ["0", "-3", "not-a-number", ""])
def test_invalid_env_value_falls_back_to_one(monkeypatch, raw):
    monkeypatch.setenv("WORKER_PROVISION_CONCURRENCY", raw)
    state = _make_state()
    assert state.provision_concurrency == 1


# ── per-model dedupe still holds under the new gate ─────────────────────────
def test_same_model_kicked_twice_is_a_no_op(monkeypatch, _stub_provision):
    monkeypatch.delenv("WORKER_PROVISION_CONCURRENCY", raising=False)
    state = _make_state()

    A._kick_provision(state, "dup-model")
    assert _wait_until(lambda: len(_stub_provision.started) == 1)
    # Re-kick while the first is still in flight (blocked on the gate) — the
    # _provisioning guard must make this a pure no-op, not a second call.
    A._kick_provision(state, "dup-model")
    time.sleep(0.2)
    assert _stub_provision.started == ["dup-model"]

    _stub_provision.gate.set()
    assert _wait_until(lambda: len(_stub_provision.completed) == 1)
    assert _stub_provision.completed == ["dup-model"]


# ── small-list case behaves as today (no artificial extra latency) ─────────
def test_single_model_provisions_immediately(monkeypatch, _stub_provision):
    monkeypatch.delenv("WORKER_PROVISION_CONCURRENCY", raising=False)
    state = _make_state()

    t0 = time.time()
    A._kick_provision(state, "solo-model")
    assert _wait_until(lambda: len(_stub_provision.started) == 1, timeout=2.0)
    elapsed = time.time() - t0
    assert elapsed < 1.0   # no serialization penalty when nothing is queued

    _stub_provision.gate.set()
    assert _wait_until(lambda: len(_stub_provision.completed) == 1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
