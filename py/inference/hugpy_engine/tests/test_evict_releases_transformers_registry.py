"""GUARD (2026-10-02): evicting an in-process transformers model must drop its
generate.coder.REGISTRY instance and release the weights. Before this the
dispatch eviction popped only the runner wrapper; the registry kept the model,
so "evicted Qwythos-9B ... freed 16.0 MB" left 17.4 GB stranded on the card."""
import gc
import weakref

from hugpy_engine.dispatch import dispatch
from hugpy_engine.generate import coder


class _Weights:
    pass


class _Cfg:
    def __init__(self, mk):
        self.model_key = mk

    def cache_key(self):
        return (self.model_key,)


class _Inst:
    def __init__(self, mk):
        self.cfg = _Cfg(mk)
        self.model = _Weights()
        self.tokenizer = object()


def test_evict_drops_the_registry_instance_and_its_weights():
    inst = _Inst("Qwythos-test")
    other = _Inst("Keep-me")
    coder.REGISTRY._instances[("Qwythos-test",)] = inst
    coder.REGISTRY._instances[("Keep-me",)] = other
    weights = weakref.ref(inst.model)
    try:
        assert dispatch.evict("Qwythos-test") is True
        assert ("Qwythos-test",) not in coder.REGISTRY._instances
        assert ("Keep-me",) in coder.REGISTRY._instances
        assert inst.model is None                 # released even if the instance is still referenced
        gc.collect()
        assert weights() is None                  # the weights object is actually freed
    finally:
        coder.REGISTRY._instances.pop(("Keep-me",), None)
        coder.REGISTRY._instances.pop(("Qwythos-test",), None)


def test_evict_of_an_unknown_model_is_a_noop():
    assert coder.REGISTRY.evict_model("") == 0
    assert coder.REGISTRY.evict_model("nope-not-loaded") == 0
