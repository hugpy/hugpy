"""Presence must distinguish NOT-LOCAL from NOT-KNOWN (operator, 2026-07-26).

THE BUG. The console painted "○ missing" over models that were plainly on disk
— on ae, 54 of 64 assigned, including one that had *just served a request*.
Operator: "yes it misleading, this should be fixed."

Not a presence failure — a RESOLUTION failure. The chain is
``_models_local`` -> ``provision.model_is_local`` -> ``get_model_config``, and
``get_model_config`` RAISES for a key this worker's registry has never learned.
``model_is_local`` swallows that and returns False, so the key silently dropped
out of ``models_local`` and the panel called it missing. ``_reap_scan`` hit the
same wall one level up (``except -> _skip("no_config")``): on ae, 54 no_config,
only 20 of 87 keys classified while 555 GiB sat resident.

The asymmetry that caused it: ``ensure_model_registered`` exists precisely to
learn an unknown row from central on demand, and the serve/probe/provision paths
all call it — the two PRESENCE readers did not. So an unlearned key stayed
unlearned forever on that path, which is why serving the model never cleared its
pill.

WHAT THESE TESTS PIN
  * the two causes of "not local" are told apart, and only the resolvable-but-
    absent one is reported as absent;
  * an unresolved key triggers a metadata-only learn (never a weight transfer);
  * learning invalidates the 60s presence cache, so the fix shows up on the NEXT
    beat instead of up to a minute later;
  * single-flight: repeated beats never stack fetches for the same key;
  * a known key costs NO central call (the steady state stays free);
  * no central_url / a failing fetch degrades to exactly today's reading.

Run: venv/bin/python -m pytest tests/test_worker_presence_config_learning.py -q
"""
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent as A
from hugpy_storage import provision as P


class _State:
    def __init__(self, assigned=None, central="http://central.invalid"):
        self.assigned_models = list(assigned or [])
        self.central_url = central
        self._provisioning = []


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """The presence cache and the in-flight set are PROCESS GLOBALS — reset both
    around every test so nothing leaks between cases or into other files."""
    A._MODELS_LOCAL_CACHE.update(at=0.0, value=[])
    with A._LEARN_CONFIGS_LOCK:
        A._LEARN_CONFIGS_INFLIGHT.clear()
    monkeypatch.setattr(A, "restart_requested", lambda: False, raising=False)
    yield
    A._MODELS_LOCAL_CACHE.update(at=0.0, value=[])
    with A._LEARN_CONFIGS_LOCK:
        A._LEARN_CONFIGS_INFLIGHT.clear()


def _settle(pred, timeout=3.0):
    """Wait for a background learn thread to land (these kicks are daemons)."""
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


# ── the core distinction ────────────────────────────────────────────────────
def test_unknown_key_is_not_reported_as_absent_and_triggers_a_learn(monkeypatch):
    """A key the registry can't resolve must NOT be silently dropped as absent —
    it must trigger the metadata-only learn that fixes it."""
    kicked = {}
    monkeypatch.setattr(P, "model_is_local", lambda mk: False)
    monkeypatch.setattr(P, "_assure_local_key", lambda mk: None)   # unresolvable
    monkeypatch.setattr(A, "_kick_learn_configs",
                        lambda s, keys: kicked.update(keys=list(keys)))

    st = _State(["Qwen~Qwen3-Coder-Next-GGUF"])
    out = A._models_local(st)

    assert out == []                                  # still honest this beat
    assert kicked.get("keys") == ["Qwen~Qwen3-Coder-Next-GGUF"], (
        "an unresolved key must be queued for learning, not silently dropped")


def test_resolvable_but_absent_key_is_left_alone(monkeypatch):
    """The OTHER cause of not-local — registry knows it, files really aren't here
    — is a genuine absence. It must NOT trigger a fetch (that would re-learn a
    row we already have on every beat, for every lazily-assigned model)."""
    kicked = []
    monkeypatch.setattr(P, "model_is_local", lambda mk: False)
    monkeypatch.setattr(P, "_assure_local_key", lambda mk: mk)     # known
    monkeypatch.setattr(A, "_kick_learn_configs",
                        lambda s, keys: kicked.append(list(keys)))

    assert A._models_local(_State(["known-but-absent"])) == []
    assert kicked == [], "a known key must cost no central call"


def test_local_models_are_reported_and_never_queued(monkeypatch):
    kicked = []
    monkeypatch.setattr(P, "model_is_local", lambda mk: True)
    monkeypatch.setattr(P, "_assure_local_key", lambda mk: mk)
    monkeypatch.setattr(A, "_kick_learn_configs",
                        lambda s, keys: kicked.append(list(keys)))

    assert A._models_local(_State(["here-1", "here-2"])) == ["here-1", "here-2"]
    assert kicked == []


# ── the learn pass itself ───────────────────────────────────────────────────
def test_learn_registers_and_invalidates_the_presence_cache(monkeypatch):
    """The 60s cache was computed against the OLD registry. Without the
    invalidation the fix would be invisible for up to a minute after the rows
    land — the operator would still be looking at "missing"."""
    seen = []
    monkeypatch.setattr(P, "ensure_model_registered",
                        lambda mk, url: seen.append(mk) or mk)
    A._MODELS_LOCAL_CACHE.update(at=time.time(), value=["stale"])

    A._kick_learn_configs(_State(), ["a", "b"])

    assert _settle(lambda: len(seen) == 2), f"learned {seen}"
    assert _settle(lambda: A._MODELS_LOCAL_CACHE["at"] == 0.0), (
        "presence cache must be invalidated so the NEXT beat re-reads locality")


def test_learn_is_single_flight_per_key(monkeypatch):
    """Repeated beats must never stack central fetches for the same key."""
    gate = threading.Event()
    calls = []

    def _slow(mk, url):
        calls.append(mk)
        gate.wait(2.0)
        return mk

    monkeypatch.setattr(P, "ensure_model_registered", _slow)

    st = _State()
    A._kick_learn_configs(st, ["dup"])
    assert _settle(lambda: calls == ["dup"])
    A._kick_learn_configs(st, ["dup"])          # second beat, same key
    A._kick_learn_configs(st, ["dup"])          # third
    time.sleep(0.05)
    assert calls == ["dup"], f"in-flight key was fetched again: {calls}"
    gate.set()
    # once it drains, the key is eligible again (not permanently blacklisted)
    assert _settle(lambda: "dup" not in A._LEARN_CONFIGS_INFLIGHT)


def test_no_central_url_is_a_noop(monkeypatch):
    called = []
    monkeypatch.setattr(P, "ensure_model_registered",
                        lambda mk, url: called.append(mk))
    A._kick_learn_configs(_State(central=None), ["x"])
    time.sleep(0.05)
    assert called == [], "no central -> no fetch, and no crash"


def test_a_failing_fetch_degrades_to_todays_reading(monkeypatch):
    """Best-effort: if central can't be reached the worker still reports what it
    already knows. The pre-existing behavior (absent -> missing) IS the failure
    mode, so a failure here can only restore today's reading, never worsen it."""
    def _boom(mk, url):
        raise RuntimeError("central unreachable")

    monkeypatch.setattr(P, "ensure_model_registered", _boom)
    A._MODELS_LOCAL_CACHE.update(at=123.0, value=["keep"])

    A._kick_learn_configs(_State(), ["a", "b"])

    assert _settle(lambda: not A._LEARN_CONFIGS_INFLIGHT)
    # nothing learned -> cache untouched (no spurious invalidation churn)
    assert A._MODELS_LOCAL_CACHE["value"] == ["keep"]


def test_one_bad_row_does_not_stop_the_rest(monkeypatch):
    done = []

    def _flaky(mk, url):
        if mk == "bad":
            raise ValueError("nope")
        done.append(mk)
        return mk

    monkeypatch.setattr(P, "ensure_model_registered", _flaky)
    A._kick_learn_configs(_State(), ["bad", "good"])
    assert _settle(lambda: done == ["good"]), f"got {done}"


# ── the SECOND half: "~"-key aliasing ───────────────────────────────────────
# Config learning was necessary but NOT sufficient. After it shipped (0.1.212),
# 9 of the 11 models still falsely missing on ae were an ALIAS mismatch: central
# assigns `Qwen~Qwen3-Coder-Next-GGUF`, the worker holds/serves the bare
# `Qwen3-Coder-Next-GGUF`, and membership was compared VERBATIM. ae's own storage
# scan listed that model at 45.09 GiB ON DISK while models_local omitted it.
# Both spellings independently answered model_is_local=True / probe local:true —
# the files were fine and the predicate was fine; only the spelling differed.
def test_key_aliases_bridges_the_tilde_and_slash_forms():
    assert A._key_aliases("Qwen~Qwen3-Coder-Next-GGUF") == ["Qwen3-Coder-Next-GGUF"]
    assert A._key_aliases("Qwen/Qwen3-Coder-Next-GGUF") == ["Qwen3-Coder-Next-GGUF"]
    # a bare key has no aliases -> the common path pays nothing
    assert A._key_aliases("Qwen3-Coder-Next-GGUF") == []
    assert A._key_aliases("") == [] and A._key_aliases(None) == []


def test_dir_slug_alone_cannot_bridge_this_pair():
    """Guards the design choice. provision._dir_slug folds separators but KEEPS
    the owner segment, so it can NOT equate the qualified and bare spellings —
    which is why the ~-tail alias exists instead. If a future _dir_slug starts
    stripping owners this fails, and the simpler normalizer becomes usable."""
    from hugpy_storage.provision import _dir_slug
    assert _dir_slug("Qwen~Qwen3-Coder-Next-GGUF") != _dir_slug("Qwen3-Coder-Next-GGUF")


def test_assigned_tilde_key_is_local_when_the_bare_form_is_on_disk(monkeypatch):
    """THE REGRESSION. Assigned qualified, stored bare -> must read LOCAL."""
    on_disk = {"Qwen3-Coder-Next-GGUF"}
    monkeypatch.setattr(P, "model_is_local", lambda mk: mk in on_disk)
    monkeypatch.setattr(P, "_assure_local_key", lambda mk: mk)
    monkeypatch.setattr(A, "_kick_learn_configs", lambda s, keys: None)

    st = _State(["Qwen~Qwen3-Coder-Next-GGUF"])
    assert A._models_local(st) == ["Qwen~Qwen3-Coder-Next-GGUF"], (
        "a model on disk under its bare name must not read as missing when the "
        "assignment carries the ~-qualified key")


def test_genuinely_absent_model_still_reads_absent(monkeypatch):
    """The alias must not manufacture presence: nothing on disk stays absent."""
    monkeypatch.setattr(P, "model_is_local", lambda mk: False)
    monkeypatch.setattr(P, "_assure_local_key", lambda mk: mk)
    monkeypatch.setattr(A, "_kick_learn_configs", lambda s, keys: None)

    assert A._models_local(_State(["Owner~NotHere"])) == []


def test_alias_probe_is_only_consulted_after_the_direct_check(monkeypatch):
    """Cost guard: a model local under its OWN key must never trigger an alias
    walk (the common path stays exactly as cheap as before)."""
    seen = []

    def _is_local(mk):
        seen.append(mk)
        return True

    monkeypatch.setattr(P, "model_is_local", _is_local)
    monkeypatch.setattr(P, "_assure_local_key", lambda mk: mk)
    monkeypatch.setattr(A, "_kick_learn_configs", lambda s, keys: None)

    assert A._models_local(_State(["Owner~X"])) == ["Owner~X"]
    assert seen == ["Owner~X"], f"alias path ran unnecessarily: {seen}"


def test_a_raising_alias_probe_degrades_to_absent(monkeypatch):
    def _boom(mk):
        raise RuntimeError("bad row")

    monkeypatch.setattr(P, "model_is_local", _boom)
    assert A._local_under_any_alias("Owner~X") is False
