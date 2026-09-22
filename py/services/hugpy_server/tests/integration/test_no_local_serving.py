"""Per-box "never serve locally" policy gate (HUGPY_NO_LOCAL_SERVING).

The central API/UI/dev station must never host or serve a model in its own
process (the path that spawned an OOM'ing llama-server on the 11 GiB VM). The
gate is a PER-BOX opt-in, default OFF, so the identical package still serves on
the worker boxes (ae/computron/op) — a hardcoded-off would kill the fleet on the
next release.

This asserts the gate at each choke point AND that default-off is a no-op:
  * policy.no_local_serving() flips only on the env flag; off by default.
  * slots.slots_enabled() -> False under policy, regardless of SLOT_COUNT.
  * serve.serve_endpoint() -> None under policy (no local slot/swap endpoint).
  * resolvers.remote DelegatingRunner refuses the local fallback under policy
    (both run() raise and stream() ErrorEvent), for every task uniformly.
  * video_intel guard_gpu_worker refuses in-process generation under policy even
    with no worker provider registered (standalone posture).

The policy reads ``os.environ`` at call time, so every test drives it through
``monkeypatch`` (nothing leaks to the operator's shell or to the next test).
"""
import asyncio
import importlib
import types

import pytest

policy = importlib.import_module("hugpy_engine.serve.policy")
slots = importlib.import_module("hugpy_engine.serve.slots")
serve = importlib.import_module("hugpy_engine.serve.serve")
remote = importlib.import_module("hugpy_engine.resolvers.remote")
guard_mod = importlib.import_module("hugpy_video.intel.runners._gpu_guard")

FLAG = "HUGPY_NO_LOCAL_SERVING"


@pytest.fixture
def policy_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)


@pytest.fixture
def policy_on(monkeypatch):
    monkeypatch.setenv(FLAG, "true")


# --- policy primitive ------------------------------------------------------

def test_policy_default_off(policy_off):
    assert policy.no_local_serving() is False


def test_policy_explicit_false_stays_off(monkeypatch):
    monkeypatch.setenv(FLAG, "false")
    assert policy.no_local_serving() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", " On "])
def test_policy_flag_turns_on(monkeypatch, value):
    monkeypatch.setenv(FLAG, value)
    assert policy.no_local_serving() is True


def test_policy_error_names_model_flag_and_fix():
    msg = policy.local_serving_error("Qwen2.5-VL-3B-Instruct-GGUF")
    assert "Qwen2.5-VL-3B-Instruct-GGUF" in msg
    assert FLAG in msg
    assert "worker" in msg


# --- slots.slots_enabled ---------------------------------------------------

def test_slots_enabled_with_slot_count_and_policy_off(monkeypatch, policy_off):
    monkeypatch.setenv("SLOT_COUNT", "2")   # a pool WOULD exist absent the policy
    assert slots.slots_enabled() is True
    # 0 does NOT fall through to a default
    assert slots._slot_count() == 2


def test_slot_count_zero_honored(monkeypatch, policy_off):
    monkeypatch.setenv("SLOT_COUNT", "0")
    assert slots._slot_count() == 0
    assert slots.slots_enabled() is False


def test_policy_on_forces_slots_off(monkeypatch, policy_on):
    monkeypatch.setenv("SLOT_COUNT", "2")
    assert slots.slots_enabled() is False
    # nothing to route to
    assert slots.slot_urls() == []


# --- serve.serve_endpoint --------------------------------------------------

def test_policy_on_serve_endpoint_is_none(policy_on):
    # Short-circuits BEFORE any registry resolution, so a dummy key is safe.
    assert serve.serve_endpoint("any-model-key") is None


# --- resolvers.remote DelegatingRunner ------------------------------------

@pytest.fixture
def delegating_runner(monkeypatch):
    """Any registered (framework, task) runner with 'no worker selected' forced
    without touching the provider seam."""
    framework, task = next(iter(remote.FRAMEWORK_RUNNERS))
    Runner = remote.make_delegating_runner(framework, task)
    runner = Runner(types.SimpleNamespace(model_key="test-model"))
    monkeypatch.setattr(remote, "_select", lambda mk, pool=None, task=None: (None, None))
    req = types.SimpleNamespace(request_id="rid-1", pool=None)
    return runner, req


def test_policy_on_run_refuses_local_fallback(policy_on, delegating_runner):
    runner, req = delegating_runner
    with pytest.raises(RuntimeError, match=FLAG):
        asyncio.run(runner.run(req))


def test_policy_on_stream_yields_one_error_event(policy_on, delegating_runner):
    runner, req = delegating_runner

    async def _collect():
        evs = []
        async for ev in runner.stream(req):
            evs.append(ev)
        return evs

    evs = asyncio.run(_collect())
    assert len(evs) == 1
    assert getattr(evs[0], "type", None) == "error"
    assert FLAG in getattr(evs[0], "message", "")


def test_policy_off_run_proceeds_to_local_runner(policy_off, delegating_runner):
    """policy OFF: the runner must proceed to the local runner, not refuse."""
    runner, req = delegating_runner
    sentinel = object()
    runner._local = types.SimpleNamespace(run=lambda req: sentinel)
    assert asyncio.run(runner.run(req)) is sentinel


# --- video_intel guard_gpu_worker -----------------------------------------

def test_guard_policy_off_no_provider_proceeds(monkeypatch, policy_off):
    # No worker provider registered (standalone posture): historically this PROCEEDS.
    monkeypatch.delenv("HUGPY_VIDEOGEN_LOCAL", raising=False)
    assert guard_mod.guard_gpu_worker("some-diffusion-model", "job-1") is None


def test_guard_policy_on_refuses_even_with_no_provider(monkeypatch, policy_on):
    monkeypatch.delenv("HUGPY_VIDEOGEN_LOCAL", raising=False)
    res = guard_mod.guard_gpu_worker("some-diffusion-model", "job-1")
    assert res is not None
    assert res.ok is False
    assert res.error.code == "local_serving_disabled"


def test_guard_policy_on_videogen_local_always_still_allowed(monkeypatch, policy_on):
    monkeypatch.setenv("HUGPY_VIDEOGEN_LOCAL", "always")
    assert guard_mod.guard_gpu_worker("some-diffusion-model", "job-1") is None


# --- get._build_runner (gguf choke) -----------------------------------------

def test_policy_on_build_runner_raises_local_engine_unavailable(policy_on):
    # heavy import: llama_cpp/torch may be absent on this box
    get_mod = pytest.importorskip("hugpy_engine.llama.runners.get")
    with pytest.raises(get_mod.LocalEngineUnavailable) as ei:
        get_mod._build_runner("test-model")
    assert str(ei.value).startswith("local model serving is disabled")
