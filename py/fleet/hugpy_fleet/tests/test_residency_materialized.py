"""No "serving" without measured residency — the worker-side half.

THE INCIDENT (2026-07-28). The compute tab showed
``🔥 serving Qwen2.5-7B-Instruct-GGUF · resident · RAM size not measured`` on
computron for a model that had NEVER loaded. Provisioning had failed on a full
disk, but dispatch had already put a HOLLOW runner wrapper into ``_INSTANCES``
(runners are lazy by design — the heavy load happens on first ``.runner``
access) and ``touch_model`` had stamped ``last_used`` BEFORE the load was
attempted. ``_allocations`` then emitted a ``kind="ram"`` row with
``serving=True``, and nothing downstream could tell it from a hot model.

Operator doctrine, "residency must be measured". The heartbeat now states what
it actually knows — and, just as importantly, says nothing when it does not.
"""
import pytest

agent = pytest.importorskip("hugpy_fleet.worker.agent")


@pytest.fixture(autouse=True)
def _clean():
    with agent._MATERIALIZED_LOCK:
        agent._MATERIALIZED.clear()
    yield
    with agent._MATERIALIZED_LOCK:
        agent._MATERIALIZED.clear()


class _Hollow:
    """A lazy wrapper exactly like the real ones: constructing it loads nothing,
    and the heavy runner appears only on first ``.runner`` access."""

    def __init__(self, model_key="Q"):
        self.model_key = model_key
        self.ensured = 0

    @property
    def runner(self):
        raise AssertionError("asking about residency must never TRIGGER a load")

    def ensure_loaded(self):
        self.ensured += 1
        return object()


class _NoEnsure:
    def __init__(self, model_key="Q"):
        self.model_key = model_key


# --------------------------------------------------------------------------- #
# _materialize — the replacement for the getattr(ensure_loaded) incantation
# --------------------------------------------------------------------------- #

def test_materialize_calls_ensure_loaded_and_records_it():
    r = _Hollow("Q")
    agent._materialize(r)
    assert r.ensured == 1
    assert agent._is_materialized("Q") is True


def test_materialize_on_a_runner_without_ensure_loaded_is_a_noop():
    """Byte-identical to the `if callable(_ensure)` guard it replaced."""
    agent._materialize(_NoEnsure("Q"))
    assert agent._is_materialized("Q") is not True


def test_materialize_reraises_a_load_failure_unchanged():
    """Every call site has its own error handling; this must not swallow."""
    class Boom(_Hollow):
        def ensure_loaded(self):
            raise RuntimeError("SIGILL")

    with pytest.raises(RuntimeError, match="SIGILL"):
        agent._materialize(Boom("Q"))
    # …and a FAILED load must not be remembered as a materialization.
    assert agent._is_materialized("Q") is not True


def test_materialize_emits_the_load_stages(monkeypatch):
    from hugpy_fleet.central import evictions as ev
    ev.reset_for_tests()
    seen = []
    ev.register_sink(seen.append)
    monkeypatch.setattr(agent, "_evt", ev, raising=False)
    monkeypatch.setattr(agent, "_evt_emit",
                        lambda stage, **f: ev.emit_eviction_event(stage, **f),
                        raising=False)
    try:
        agent._materialize(_Hollow("Q"))
        stages = [e["stage"] for e in seen]
        assert stages == ["load.start", "load.done"]
        assert seen[0]["engine"] == "_Hollow"
        assert isinstance(seen[1]["duration_ms"], int)
    finally:
        ev.reset_for_tests()


def test_materialize_emits_load_fail(monkeypatch):
    from hugpy_fleet.central import evictions as ev
    ev.reset_for_tests()
    seen = []
    ev.register_sink(seen.append)
    monkeypatch.setattr(agent, "_evt", ev, raising=False)
    monkeypatch.setattr(agent, "_evt_emit",
                        lambda stage, **f: ev.emit_eviction_event(stage, **f),
                        raising=False)

    class Boom(_Hollow):
        def ensure_loaded(self):
            raise RuntimeError("SIGILL")

    try:
        with pytest.raises(RuntimeError):
            agent._materialize(Boom("Q"))
        assert [e["stage"] for e in seen] == ["load.start", "load.fail"]
        assert "SIGILL" in seen[1]["error"]
    finally:
        ev.reset_for_tests()


# --------------------------------------------------------------------------- #
# _is_materialized — True / False / None(unknown), and NEVER a guess
# --------------------------------------------------------------------------- #

def test_a_loaded_in_process_gguf_reads_true(monkeypatch):
    from hugpy_engine.llama.runners import get as g

    class Loaded:
        base_url = None
        llm = object()                       # the materialized Llama handle

    monkeypatch.setitem(g._LLAMA_INSTANCES, "Q", Loaded())
    assert agent._is_materialized("Q") is True


def test_the_incident_shape_reads_false(monkeypatch):
    """A hollow llama wrapper in dispatch with NO entry in the llama cache:
    runner_for() built the shell, .runner was never touched. This is the model
    that rendered as 🔥 serving while its weights had never existed."""
    from hugpy_engine.dispatch import dispatch as d
    from hugpy_engine.llama.runners.src.base_runner import LlamaCppBaseRunner

    class HollowLlama(LlamaCppBaseRunner):
        # The base class is abstract; a real hollow wrapper is a concrete
        # subclass whose heavy runner simply hasn't been resolved yet.
        def __init__(self):
            self.model_key = "Q"

        def _chat_complete(self, *a, **k):
            raise AssertionError("not reached")

        def _raw_complete(self, *a, **k):
            raise AssertionError("not reached")

        def _iter_stream(self, *a, **k):
            raise AssertionError("not reached")

    monkeypatch.setitem(d._INSTANCES, ("Q", "chat"), HollowLlama())
    assert agent._is_materialized("Q") is False


def test_a_slot_backed_runner_reads_unknown_not_false(monkeypatch):
    """A slot child holds the weights in ANOTHER process. This process cannot
    speak for it, and the slot's own row reports its measured residency."""
    from hugpy_engine.llama.runners import get as g

    class SlotBacked:
        base_url = "http://127.0.0.1:9001"
        llm = None

    monkeypatch.setitem(g._LLAMA_INSTANCES, "Q", SlotBacked())
    assert agent._is_materialized("Q") is None


def test_a_non_llama_runner_reads_unknown_not_false(monkeypatch):
    """THE guard against fixing the bug backwards.

    A transformers runner loaded through a plain .run() leaves no trace in
    either source while being genuinely, measurably resident. The console
    treats materialized=False as outranking measurement, so claiming False here
    would HIDE a hot model — the same class of lie, pointed the other way."""
    from hugpy_engine.dispatch import dispatch as d

    monkeypatch.setitem(d._INSTANCES, ("Q", "chat"), _NoEnsure("Q"))
    assert agent._is_materialized("Q") is None


def test_an_unknown_key_reads_unknown():
    assert agent._is_materialized("nothing-knows-this-model") is None


def test_the_live_cache_outranks_a_remembered_flag(monkeypatch):
    """dispatch.evict cascades into evict_llama_runner, so the cache
    self-corrects on unload where a remembered flag would go stale and
    re-assert residency for weights that are gone."""
    from hugpy_engine.llama.runners import get as g

    class Unloaded:
        base_url = None
        llm = None                            # evicted: the handle is gone

    with agent._MATERIALIZED_LOCK:
        agent._MATERIALIZED.add("Q")          # a stale memory of the old load
    monkeypatch.setitem(g._LLAMA_INSTANCES, "Q", Unloaded())
    assert agent._is_materialized("Q") is False


def test_is_materialized_never_touches_the_runner_property(monkeypatch):
    """Asking about residency must not TRIGGER the load being asked about —
    _Hollow.runner raises if anything reads it."""
    from hugpy_engine.dispatch import dispatch as d

    monkeypatch.setitem(d._INSTANCES, ("Q", "chat"), _Hollow("Q"))
    agent._is_materialized("Q")               # must not raise
