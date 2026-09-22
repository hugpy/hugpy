"""summarizers seq2seq loaders — placement-seam wiring (Slice C).

These summarizer back-ends (FlanBackend / Seq2SeqChunkedBackend /
PipelineChunkedBackend in hugpy_media.summarizers.summarizers) were
ALL-OR-FAIL: a bare
from_pretrained(model_dir) with device chosen as `0 if cuda else -1` never
consulted the spill seam, so a too-big model OOM'd instead of spilling to RAM
and the operator's allocation modes / placement intent were silently ignored.

Slice C wires the SAME seam every other transformers loader honors
(spill.transformers_max_memory). This test asserts, via kwargs-capturing fakes
(NO real model load):
  * seam-has-an-answer  -> from_pretrained gets device_map="auto"+max_memory and
    the pipeline() call DROPS device= (device/device_map are mutually exclusive);
  * seam-silent (None)  -> BYTE-IDENTICAL to today: plain from_pretrained(dir),
    pipeline(device=0/-1), NO device_map/max_memory anywhere;
  * pipeline-from-path (PipelineChunkedBackend) rides model_kwargs+device_map;
  * no accelerate -> degrade to the plain path with a log, never a spill map.

The retired monolith's falconsai_module sibling (SingletonMeta variants of the
same back-ends) was dropped as dead code and has no hugpy_media equivalent, so
only the summarizers module is exercised here.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

summ = importlib.import_module("hugpy_media.summarizers.summarizers")


# --------------------------------------------------------------------------- #
# fakes — no real transformers load, just kwargs capture
# --------------------------------------------------------------------------- #
class _FakeModel:
    """Stand-in for AutoModelForSeq2SeqLM. Records from_pretrained kwargs."""
    last_kwargs: dict = {}
    device = "cpu"

    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        cls.last_kwargs = dict(kwargs)
        return cls()


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        return SimpleNamespace()


class _PipelineCapture:
    """Callable that records the args pipeline(...) was called with. A shared
    module-level instance is returned by the fake get_transformers so tests can
    read the last call off the class-level attrs."""
    last_task = None
    last_kwargs: dict = {}

    def __call__(self, task, **kwargs):
        type(self).last_task = task
        type(self).last_kwargs = dict(kwargs)
        return SimpleNamespace(task=task, kwargs=kwargs)


_PIPELINE_CAPTURE = _PipelineCapture()


def _fake_get_transformers(name=None):
    if name == "AutoModelForSeq2SeqLM":
        return _FakeModel
    if name == "AutoTokenizer":
        return _FakeTokenizer
    if name == "pipeline":
        return _PIPELINE_CAPTURE
    raise AssertionError(f"unexpected get_transformers({name!r})")


def _fake_ensure_model(model_key):
    return f"/nonexistent/{model_key}"


class _FakeCuda:
    def __init__(self, available):
        self._a = available

    def is_available(self):
        return self._a


def _fake_get_torch(available=True):
    return SimpleNamespace(cuda=_FakeCuda(available))


class _patched:
    """Save/restore attrs so tests are independent under pytest or plain-script."""

    def __init__(self, obj, **kw):
        self.obj = obj
        self.kw = kw
        self.saved = {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.saved[k] = getattr(self.obj, k)
            setattr(self.obj, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.obj, k, v)


SEAM = {0: "9.00GiB", "cpu": "12.00GiB"}


def _reset_captures():
    _FakeModel.last_kwargs = {}
    _PipelineCapture.last_task = None
    _PipelineCapture.last_kwargs = {}


# --------------------------------------------------------------------------- #
# helpers to drive each backend's loader once under patched deps
# --------------------------------------------------------------------------- #
def _run_flan(mod, *, cuda, seam, accelerate_ok=True):
    """Load a fresh FlanBackend and return (model_kwargs, pipeline_kwargs).

    Handles both variants: summarizers' FlanBackend caches via a `_pipeline`
    property (per model_key), falconsai's is a SingletonMeta whose __init__
    builds `self._pipeline` eagerly and takes no model_key."""
    import inspect as _inspect
    _reset_captures()
    # falconsai's FlanBackend is a SingletonMeta with __init__(self) and a
    # `_pipeline` built eagerly; summarizers' takes __init__(self, model_key=…)
    # with a `_pipeline` property. Detect by the __init__ signature (robust to
    # WHICH SingletonMeta implementation is in play — the two trees use
    # different ones) and reset the right cache.
    takes_model_key = "model_key" in _inspect.signature(
        mod.FlanBackend.__init__).parameters
    smeta = type(mod.FlanBackend)
    saved_instances = dict(getattr(smeta, "_instances", {}))
    if not takes_model_key:                 # singleton, eager _pipeline
        if hasattr(smeta, "_instances"):
            smeta._instances.pop(mod.FlanBackend, None)
    else:
        mod.FlanBackend._PIPELINES = {}
    # falconsai routes the pipeline device arg through a helper; summarizers
    # inlines it. Patch the helper only where it exists so both use the SAME
    # deterministic device-arg contract in the test.
    extra = {}
    if hasattr(mod, "_pipeline_device_kwargs"):
        extra["_pipeline_device_kwargs"] = _pdk_stub(seam, accelerate_ok, cuda, mod)
    try:
        with _patched(mod, get_transformers=_fake_get_transformers,
                      ensure_model=_fake_ensure_model,
                      get_torch=lambda: _fake_get_torch(cuda)), \
             _patched(mod, _seq2seq_spill_kwargs=_spill_stub(seam, accelerate_ok, mod)), \
             _patched(mod, **extra):
            if not takes_model_key:
                mod.FlanBackend()          # __init__ builds self._pipeline eagerly
            else:
                mod.FlanBackend(model_key="k")._pipeline   # property triggers load
    finally:
        if not takes_model_key and hasattr(smeta, "_instances"):
            smeta._instances.clear()
            smeta._instances.update(saved_instances)
    return dict(_FakeModel.last_kwargs), dict(_PipelineCapture.last_kwargs)


def _pdk_stub(seam, accelerate_ok, cuda, mod):
    """Match _pipeline_device_kwargs's contract (only falconsai uses it as a
    separate helper; summarizers inlines the same logic)."""
    def _fn(spill_kwargs):
        if spill_kwargs.get("device_map"):
            return {}
        return {"device": 0 if cuda else -1}
    return _fn


def _spill_stub(seam, accelerate_ok, mod):
    """Reproduce _seq2seq_spill_kwargs's contract deterministically for the test:
    {} when no cuda / no seam / no accelerate, else device_map+max_memory."""
    def _fn():
        if seam is None:
            return {}
        if not accelerate_ok:
            return {}
        return {"device_map": "auto", "max_memory": seam}
    return _fn


# --------------------------------------------------------------------------- #
# tests — run against the summarizers module
# --------------------------------------------------------------------------- #
def _assert_seam_honored(mod):
    mk, pk = _run_flan(mod, cuda=True, seam=SEAM)
    assert mk.get("device_map") == "auto", mk
    assert mk.get("max_memory") == SEAM, mk
    assert "device" not in pk, ("pipeline() must NOT carry device= when the model "
                                "is device-mapped", pk)


def _assert_seam_silent_byte_identical(mod):
    mk, pk = _run_flan(mod, cuda=True, seam=None)
    assert mk == {}, ("seam None -> plain from_pretrained, no device_map/max_memory", mk)
    assert pk.get("device") == 0, ("cuda + no seam -> device=0 (historical)", pk)
    assert "device_map" not in pk, pk


def _assert_no_gpu_byte_identical(mod):
    mk, pk = _run_flan(mod, cuda=False, seam=None)
    assert mk == {}, mk
    assert pk.get("device") == -1, ("no cuda -> device=-1 (historical)", pk)


def _assert_no_accelerate_degrades(mod):
    mk, pk = _run_flan(mod, cuda=True, seam=SEAM, accelerate_ok=False)
    assert mk == {}, ("no accelerate -> plain load, no spill map", mk)
    assert pk.get("device") == 0, pk


def test_summ_seam_honored():
    _assert_seam_honored(summ)


def test_summ_seam_silent_byte_identical():
    _assert_seam_silent_byte_identical(summ)


def test_summ_no_gpu_byte_identical():
    _assert_no_gpu_byte_identical(summ)


def test_summ_no_accelerate_degrades():
    _assert_no_accelerate_degrades(summ)



# --------------------------------------------------------------------------- #
# PipelineChunkedBackend — pipeline-from-path rides model_kwargs+device_map
# --------------------------------------------------------------------------- #
def _run_pipeline_chunked(mod, *, cuda, seam):
    """Drive PipelineChunkedBackend once. summarizers caches via _PIPELINES
    (property); falconsai is a SingletonMeta whose __init__ builds _pipeline
    eagerly — reset whichever cache applies so each case actually loads."""
    _reset_captures()
    cls = mod.PipelineChunkedBackend
    smeta = type(cls)
    saved = dict(getattr(smeta, "_instances", {}))
    has_pipelines = hasattr(cls, "_PIPELINES")
    if has_pipelines:
        cls._PIPELINES = {}
    elif hasattr(smeta, "_instances"):
        smeta._instances.pop(cls, None)
    try:
        with _patched(mod, get_transformers=_fake_get_transformers,
                      ensure_model=_fake_ensure_model,
                      get_torch=lambda: _fake_get_torch(cuda)), \
             _patched(mod, _seq2seq_spill_kwargs=_spill_stub(seam, True, mod)):
            inst = cls(model_key="k")
            if has_pipelines:
                inst._pipeline               # property triggers the load
    finally:
        if not has_pipelines and hasattr(smeta, "_instances"):
            smeta._instances.clear()
            smeta._instances.update(saved)
    return dict(_PipelineCapture.last_kwargs), _PipelineCapture.last_task


def test_pipeline_chunked_seam_honored():
    for mod in (summ,):
        pk, task = _run_pipeline_chunked(mod, cuda=True, seam=SEAM)
        assert task == "summarization", task
        assert pk.get("device_map") == "auto", pk
        assert pk.get("model_kwargs") == {"max_memory": SEAM}, pk
        assert "device" not in pk, ("device= must be absent with device_map", pk)


def test_pipeline_chunked_seam_silent_byte_identical():
    for mod in (summ,):
        pk, task = _run_pipeline_chunked(mod, cuda=True, seam=None)
        assert pk.get("device") == 0, ("cuda + no seam -> device=0", pk)
        assert "device_map" not in pk and "model_kwargs" not in pk, pk
        pk, _ = _run_pipeline_chunked(mod, cuda=False, seam=None)
        assert pk.get("device") == -1, ("no cuda -> device=-1", pk)


# --------------------------------------------------------------------------- #
# Seq2SeqChunkedBackend — device_map load + input-device movement
# --------------------------------------------------------------------------- #
def test_seq2seq_chunked_summ_records_device_mapped_flag():
    """summarizers variant: per-model-key cache + _DEVICE_MAPPED flag set."""
    _reset_captures()
    with _patched(summ, get_transformers=_fake_get_transformers,
                  ensure_model=_fake_ensure_model,
                  get_torch=lambda: _fake_get_torch(True)), \
         _patched(summ, _seq2seq_spill_kwargs=_spill_stub(SEAM, True, summ)):
        summ.Seq2SeqChunkedBackend._MODELS = {}
        summ.Seq2SeqChunkedBackend._DEVICE_MAPPED = {}
        summ.Seq2SeqChunkedBackend(model_key="k")._load()
    assert summ.Seq2SeqChunkedBackend._DEVICE_MAPPED.get("k") is True
    assert _FakeModel.last_kwargs.get("device_map") == "auto"


def test_seq2seq_chunked_summ_no_seam_no_flag():
    """No seam -> plain load, _DEVICE_MAPPED False, byte-identical from_pretrained."""
    _reset_captures()
    with _patched(summ, get_transformers=_fake_get_transformers,
                  ensure_model=_fake_ensure_model,
                  get_torch=lambda: _fake_get_torch(True)), \
         _patched(summ, _seq2seq_spill_kwargs=_spill_stub(None, True, summ)):
        summ.Seq2SeqChunkedBackend._MODELS = {}
        summ.Seq2SeqChunkedBackend._DEVICE_MAPPED = {}
        summ.Seq2SeqChunkedBackend(model_key="k2")._load()
    assert summ.Seq2SeqChunkedBackend._DEVICE_MAPPED.get("k2") is False
    assert _FakeModel.last_kwargs == {}, _FakeModel.last_kwargs



# --------------------------------------------------------------------------- #
# real _seq2seq_spill_kwargs contract: no cuda -> {} (byte-identical guarantee)
# --------------------------------------------------------------------------- #
def test_real_helper_no_cuda_returns_empty():
    for mod in (summ,):
        with _patched(mod, get_torch=lambda: _fake_get_torch(False)):
            assert mod._seq2seq_spill_kwargs() == {}


def test_real_helper_seam_none_returns_empty():
    import hugpy_engine.spill as spill
    for mod in (summ,):
        with _patched(mod, get_torch=lambda: _fake_get_torch(True)), \
             _patched(spill, transformers_max_memory=lambda *a, **k: None):
            assert mod._seq2seq_spill_kwargs() == {}
