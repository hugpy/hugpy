"""k101 — registration probes: does the adapter agree with its descriptor?

Doc §3.2's rule is the spine of this file: a probe that DISAGREES with the
descriptor fails and makes the capability ineligible; a probe that could not run
answers unknown and changes nothing. Every check is exercised against fake
modules and fake rows — no GPU, no worker, no shared store, no network — plus a
handful of live assertions about THIS tree that would catch a real regression
(the tts adapter really is importable here; its spec really does agree with
runners/tts_chatterbox).

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_probes.py -q
"""
from __future__ import annotations

import logging
import os
import sys
import types
from dataclasses import dataclass

import pytest

logging.disable(logging.INFO)

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_oracle import probes
from hugpy_oracle.contracts import ProbeStatus


@pytest.fixture(autouse=True)
def _clean_cache():
    """The TTL cache is process-global by design; no test may inherit another
    test's verdict."""
    probes.clear_cache()
    yield
    probes.clear_cache()


def _status(result, name):
    match = [c for c in result.checks if c.name == name]
    assert match, f"{name} missing from {[c.name for c in result.checks]}"
    return match[0]


def _module(**attrs):
    return types.SimpleNamespace(**attrs)


def _stub_module(monkeypatch, module, *, found=True):
    monkeypatch.setattr(probes, "_find_spec",
                        lambda name: object() if found else None)
    monkeypatch.setattr(probes, "_import_module", lambda name: module)


# ---------------------------------------------------------------------------
# The registry itself.
# ---------------------------------------------------------------------------


def test_every_registered_spec_names_a_namespaced_capability():
    for name, spec in probes.PROBE_SPECS.items():
        assert spec.capability == name
        assert "." in name
    with pytest.raises(ValueError):
        probes.ProbeSpec(capability="tts")


def test_tts_spec_points_at_the_catalog_s_runner_module():
    """Two tables, one adapter: probes cannot import the catalog (the catalog
    imports probes), so the agreement is asserted instead."""
    from hugpy_oracle import catalog
    assert (probes.PROBE_SPECS["audio.tts"].runner_module
            == catalog.TTS_RUNNER_MODULE)
    assert probes.PROBE_SPECS["audio.tts"].task == catalog.TTS_TASK


def test_speaker_similarity_has_no_probe_and_that_is_the_honest_answer():
    """It is declared with no binding at all — probing nothing would be
    theatre on top of a gap the catalog already states plainly."""
    assert "audio.speaker_similarity" not in probes.PROBE_SPECS
    assert probes.probe_capability("audio.speaker_similarity") is None
    assert probes.probe_capability("no.such.capability") is None


def test_register_probe_adds_a_spec_and_drops_its_cached_result(monkeypatch):
    spec = probes.ProbeSpec(capability="test.only", runner_module="json",
                            params=("x",))
    _stub_module(monkeypatch, _module(PARAMS=("x", "y")))
    try:
        probes.register_probe(spec)
        assert probes.probe_spec_for("test.only") is spec
        assert probes.probe_capability("test.only").status is ProbeStatus.OK
        assert probes.cache_size() == 1
        probes.register_probe(spec)                 # re-registering invalidates
        assert probes.cache_size() == 0
    finally:
        probes.PROBE_SPECS.pop("test.only", None)


# ---------------------------------------------------------------------------
# runner_module.
# ---------------------------------------------------------------------------


def test_missing_runner_module_is_a_failure(monkeypatch):
    monkeypatch.setattr(probes, "_find_spec", lambda name: None)
    spec = probes.ProbeSpec(capability="x.y", runner_module="nope.nope")
    result = probes.run_probe(spec)
    assert result.status is ProbeStatus.FAIL
    assert "not importable" in _status(result, "runner_module").detail
    assert "nope.nope" in result.reason()


def test_a_missing_parent_package_is_also_a_failure(monkeypatch):
    """find_spec REPORTS this one by raising; it is still "the adapter the
    descriptor names is not here"."""
    def _boom(name):
        raise ModuleNotFoundError("No module named 'not'")
    monkeypatch.setattr(probes, "_find_spec", _boom)
    result = probes.run_probe(probes.ProbeSpec(capability="x.y",
                                               runner_module="not.a.module"))
    assert _status(result, "runner_module").status is ProbeStatus.FAIL


def test_unresolvable_module_name_is_unknown_not_a_failure(monkeypatch):
    def _boom(name):
        raise ValueError("half-installed namespace package")
    monkeypatch.setattr(probes, "_find_spec", _boom)
    result = probes.run_probe(probes.ProbeSpec(capability="x.y",
                                               runner_module="weird"))
    assert _status(result, "runner_module").status is ProbeStatus.UNKNOWN


def test_a_capability_with_no_declared_adapter_is_unknown():
    result = probes.run_probe(probes.ProbeSpec(capability="x.y"))
    assert result.status is ProbeStatus.UNKNOWN
    assert _status(result, "runner_module").status is ProbeStatus.UNKNOWN
    assert _status(result, "param_agreement").status is ProbeStatus.UNKNOWN


# ---------------------------------------------------------------------------
# param_agreement — the doc's own example.
# ---------------------------------------------------------------------------


def test_adapter_that_unexpectedly_requires_prompt_is_a_failure(monkeypatch):
    """Doc §3.2, verbatim: 'an image adapter that unexpectedly requires prompt
    or text is ineligible until its descriptor and probe agree'."""
    def render(image, prompt, steps=20):
        ...
    _stub_module(monkeypatch, _module(render=render))
    spec = probes.ProbeSpec(capability="image.transform", runner_module="fake",
                            entrypoint="render", params=("image", "steps"))
    result = probes.run_probe(spec)
    assert result.status is ProbeStatus.FAIL
    detail = _status(result, "param_agreement").detail
    assert "REQUIRES ['prompt']" in detail


def test_descriptor_declaring_a_param_the_adapter_refuses_is_a_failure(monkeypatch):
    def render(image, steps=20):
        ...
    _stub_module(monkeypatch, _module(render=render))
    spec = probes.ProbeSpec(capability="image.transform", runner_module="fake",
                            entrypoint="render",
                            params=("image", "steps", "guidance"))
    result = probes.run_probe(spec)
    assert result.status is ProbeStatus.FAIL
    assert "does not accept" in _status(result, "param_agreement").detail
    assert "guidance" in _status(result, "param_agreement").detail


def test_params_agree_is_ok(monkeypatch):
    def render(image, steps=20, guidance=7.0):
        ...
    _stub_module(monkeypatch, _module(render=render))
    spec = probes.ProbeSpec(capability="image.transform", runner_module="fake",
                            entrypoint="render", params=("image", "steps"))
    assert probes.run_probe(spec).status is ProbeStatus.OK


def test_dispatch_supplied_params_are_not_an_interface_mismatch(monkeypatch):
    """An artifact path and a transport id are plumbing the dispatch layer
    fills in — reporting them as a mismatch would cry wolf forever."""
    def transcribe(file_path, request_id, word_timestamps=False):
        ...
    _stub_module(monkeypatch, _module(transcribe=transcribe))
    spec = probes.ProbeSpec(
        capability="audio.transcribe.word_timestamps", runner_module="fake",
        entrypoint="transcribe", params=("word_timestamps",),
        supplied_by_dispatch=("file_path", "request_id"))
    assert probes.run_probe(spec).status is ProbeStatus.OK


def test_declared_params_constant_beats_introspection(monkeypatch):
    def render(totally, different, names):
        ...
    _stub_module(monkeypatch, _module(render=render, PARAMS=("image", "steps")))
    spec = probes.ProbeSpec(capability="image.transform", runner_module="fake",
                            entrypoint="render", params=("image",))
    assert probes.run_probe(spec).status is ProbeStatus.OK


def test_kwargs_adapter_cannot_be_proven_either_way(monkeypatch):
    def render(**kwargs):
        ...
    _stub_module(monkeypatch, _module(render=render))
    spec = probes.ProbeSpec(capability="image.transform", runner_module="fake",
                            entrypoint="render", params=("anything",))
    check = _status(probes.run_probe(spec), "param_agreement")
    assert check.status is ProbeStatus.UNKNOWN
    assert "**kwargs" in check.detail


def test_no_parameter_surface_is_unknown(monkeypatch):
    _stub_module(monkeypatch, _module(unrelated=1))
    spec = probes.ProbeSpec(capability="image.transform", runner_module="fake",
                            entrypoint="render", params=("image",))
    assert _status(probes.run_probe(spec),
                   "param_agreement").status is ProbeStatus.UNKNOWN


def test_a_descriptor_declaring_no_params_has_nothing_to_check(monkeypatch):
    def render(image):
        ...
    _stub_module(monkeypatch, _module(render=render))
    spec = probes.ProbeSpec(capability="image.transform", runner_module="fake",
                            entrypoint="render")
    assert _status(probes.run_probe(spec),
                   "param_agreement").status is ProbeStatus.UNKNOWN


def test_adapter_that_does_not_import_here_is_unknown(monkeypatch):
    monkeypatch.setattr(probes, "_find_spec", lambda name: object())

    def _boom(name):
        raise ImportError("no torch on this box")
    monkeypatch.setattr(probes, "_import_module", _boom)
    spec = probes.ProbeSpec(capability="x.y", runner_module="fake",
                            entrypoint="run", params=("a",))
    result = probes.run_probe(spec)
    assert _status(result, "runner_module").status is ProbeStatus.OK
    assert _status(result, "param_agreement").status is ProbeStatus.UNKNOWN
    assert result.status is ProbeStatus.UNKNOWN     # never a fail


def test_accepted_params_reads_all_three_adapter_shapes():
    @dataclass
    class Spec:
        text: str
        seed: int = 0

    accepted, required = probes.accepted_params(Spec)
    assert accepted == {"text", "seed"} and required == {"text"}

    from hugpy_media.schemas.whisper_schemas import TranscribeRequest
    accepted, required = probes.accepted_params(TranscribeRequest)
    assert "word_timestamps" in accepted and "word_timestamps" not in required

    def fn(a, b=1, *args, **kwargs):
        ...
    accepted, _required = probes.accepted_params(fn)
    assert accepted == probes._ACCEPTS_ANY
    assert probes.accepted_params(42) is None


# ---------------------------------------------------------------------------
# model rows + weights.
# ---------------------------------------------------------------------------

_ROWS = {
    "m1": {"tasks": ["text-to-speech"], "license": "mit",
           "extra": {"dir": "/nonexistent/should-not-be-read"}},
}


def _row_spec(**over):
    base = dict(capability="audio.tts", runner_module=None, model_row=True)
    base.update(over)
    return probes.ProbeSpec(**base)


def test_bound_model_without_a_registry_row_is_a_failure():
    result = probes.run_probe(_row_spec(), rows={}, model_ids=("ghost",))
    assert result.status is ProbeStatus.FAIL
    assert "no registry row" in _status(result, "model_row").detail


def test_row_declaring_no_tasks_is_a_failure():
    rows = {"m1": {"tasks": [], "license": "mit"}}
    result = probes.run_probe(_row_spec(), rows=rows, model_ids=("m1",))
    assert result.status is ProbeStatus.FAIL
    assert "declare no tasks" in _status(result, "model_row").detail


def test_unrecorded_license_is_unknown_never_assumed_permissive():
    rows = {"m1": {"tasks": ["text-to-speech"]}}
    result = probes.run_probe(_row_spec(), rows=rows, model_ids=("m1",))
    check = _status(result, "model_license")
    assert check.status is ProbeStatus.UNKNOWN
    assert "never assumed permissive" in check.detail
    assert result.status is ProbeStatus.UNKNOWN      # not a refusal


def test_no_bound_model_is_unknown_not_a_probe_failure():
    result = probes.run_probe(_row_spec(), rows={}, model_ids=())
    assert _status(result, "model_row").status is ProbeStatus.UNKNOWN


def test_zero_byte_weights_are_a_failure(monkeypatch):
    monkeypatch.setattr(probes, "_scandir_entries",
                        lambda d: (("model.safetensors", 0),))
    result = probes.run_probe(_row_spec(check_weights=True), rows=_ROWS,
                              model_ids=("m1",))
    assert result.status is ProbeStatus.FAIL
    assert "no non-empty file" in _status(result, "model_weights").detail


def test_absent_weights_directory_is_a_failure(monkeypatch):
    def _gone(directory):
        raise FileNotFoundError(directory)
    monkeypatch.setattr(probes, "_scandir_entries", _gone)
    result = probes.run_probe(_row_spec(check_weights=True), rows=_ROWS,
                              model_ids=("m1",))
    assert result.status is ProbeStatus.FAIL
    assert "does not exist" in _status(result, "model_weights").detail


def test_unreadable_store_is_unknown_not_a_verdict_about_the_adapter(monkeypatch):
    def _denied(directory):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(probes, "_scandir_entries", _denied)
    result = probes.run_probe(_row_spec(check_weights=True), rows=_ROWS,
                              model_ids=("m1",))
    assert _status(result, "model_weights").status is ProbeStatus.UNKNOWN


def test_row_without_an_absolute_path_leaves_weights_unknown(monkeypatch):
    rows = {"m1": {"tasks": ["x"], "license": "mit", "folder": "relative/path"}}
    result = probes.run_probe(_row_spec(check_weights=True), rows=rows,
                              model_ids=("m1",))
    check = _status(result, "model_weights")
    assert check.status is ProbeStatus.UNKNOWN
    assert "no absolute path" in check.detail


def test_studio_zero_byte_list_is_consulted_first():
    from hugpy_video.intel.studio.presets import ZERO_BYTE_MODELS
    empty = sorted(ZERO_BYTE_MODELS)[0]
    rows = {empty: {"tasks": ["text-to-video"], "license": "mit"}}
    result = probes.run_probe(_row_spec(check_weights=True), rows=rows,
                              model_ids=(empty,))
    assert result.status is ProbeStatus.FAIL
    assert "0 bytes" in _status(result, "model_weights").detail


def test_probe_row_license_agrees_with_the_catalog():
    """Two readers of one row field; a drift here is a probe and a catalog
    telling different stories about the same license."""
    from hugpy_oracle import catalog
    for row in ({"license": "mit"}, {"extra": {"license": "apache-2.0"}},
                {"extra": {"extra": {"license": "openrail"}}}, {}):
        assert probes._row_license(row) == catalog._row_license(row)


# ---------------------------------------------------------------------------
# worker seat.
# ---------------------------------------------------------------------------


def _seat_spec():
    return probes.ProbeSpec(capability="audio.tts", task="text-to-speech")


def test_an_affirmative_worker_seat_is_ok():
    workers = [{"id": "w1", "task_capabilities": {"text-to-speech": True}}]
    result = probes.run_probe(_seat_spec(), workers=workers)
    assert _status(result, "worker_seat").status is ProbeStatus.OK


def test_an_unseated_task_is_unknown_never_a_probe_failure():
    """"Nobody has taken this yet" is availability, not a broken interface —
    the catalog reports the gap with its own precise wording."""
    workers = [{"id": "w1", "task_capabilities": {"text-generation": True}}]
    result = probes.run_probe(_seat_spec(), workers=workers)
    check = _status(result, "worker_seat")
    assert check.status is ProbeStatus.UNKNOWN
    assert "STRICT/affirmative" in check.detail
    assert result.status is ProbeStatus.UNKNOWN


def test_unreadable_worker_registry_is_unknown():
    assert _status(probes.run_probe(_seat_spec(), workers=None),
                   "worker_seat").status is ProbeStatus.UNKNOWN
    assert _status(probes.run_probe(_seat_spec(), workers=[]),
                   "worker_seat").status is ProbeStatus.UNKNOWN


def test_worker_seat_check_uses_the_catalog_s_strict_helper(monkeypatch):
    from hugpy_oracle import catalog
    monkeypatch.setattr(catalog, "_worker_seats_task", lambda w, t: True)
    result = probes.run_probe(_seat_spec(), workers=[{"id": "w1"}])
    assert _status(result, "worker_seat").status is ProbeStatus.OK


# ---------------------------------------------------------------------------
# The time budget.
# ---------------------------------------------------------------------------


def test_budget_exceeded_skips_the_rest_as_unknown(monkeypatch):
    """A stalled mount or a torch-importing adapter must not hold a capability
    listing: whatever has not run answers unknown, never ok."""
    ticks = {"n": 0}

    def clock():                 # the first check runs; then the budget is gone
        ticks["n"] += 1
        return 0.0 if ticks["n"] <= 2 else 9.9
    monkeypatch.setattr(probes, "_monotonic", clock)
    monkeypatch.setattr(probes, "_find_spec", lambda name: object())
    spec = probes.ProbeSpec(capability="audio.tts", runner_module="fake",
                            entrypoint="run", params=("a",),
                            task="text-to-speech", model_row=True)
    result = probes.run_probe(spec, rows=_ROWS, model_ids=("m1",),
                              workers=[{"id": "w1"}])
    assert result.status is ProbeStatus.UNKNOWN
    assert _status(result, "runner_module").status is ProbeStatus.OK
    for name in ("param_agreement", "model_row", "worker_seat"):
        check = _status(result, name)
        assert check.status is ProbeStatus.UNKNOWN
        assert probes.BUDGET_SKIPPED in check.detail


def test_the_default_budget_is_half_a_second():
    assert probes.PROBE_BUDGET_S == 0.5


def test_a_raising_check_never_becomes_the_failure(monkeypatch):
    def _boom(name):
        raise RuntimeError("the probe itself broke")
    monkeypatch.setattr(probes, "_find_spec", _boom)
    result = probes.run_probe(probes.ProbeSpec(capability="x.y",
                                               runner_module="fake"))
    check = _status(result, "runner_module")
    assert check.status is ProbeStatus.UNKNOWN
    assert "RuntimeError" in check.detail


# ---------------------------------------------------------------------------
# TTL cache.
# ---------------------------------------------------------------------------


def _counting_spec(monkeypatch, calls):
    monkeypatch.setattr(probes, "_find_spec",
                        lambda name: calls.append(name) or object())
    spec = probes.ProbeSpec(capability="test.cache", runner_module="fake")
    probes.PROBE_SPECS["test.cache"] = spec
    return spec


def test_second_probe_within_the_ttl_is_a_cache_hit(monkeypatch):
    calls: list[str] = []
    _counting_spec(monkeypatch, calls)
    try:
        first = probes.probe_capability("test.cache", ttl_s=300)
        second = probes.probe_capability("test.cache", ttl_s=300)
        assert first == second and len(calls) == 1
        assert probes.cache_size() == 1
    finally:
        probes.PROBE_SPECS.pop("test.cache", None)


def test_expired_ttl_reprobes(monkeypatch):
    calls: list[str] = []
    _counting_spec(monkeypatch, calls)
    now = [0.0]
    monkeypatch.setattr(probes, "_monotonic", lambda: now[0])
    try:
        probes.probe_capability("test.cache", ttl_s=10)
        now[0] = 100.0                      # well past the 10 s deadline
        probes.probe_capability("test.cache", ttl_s=10)
        assert len(calls) == 2
    finally:
        probes.PROBE_SPECS.pop("test.cache", None)


def test_ttl_zero_disables_the_cache(monkeypatch):
    calls: list[str] = []
    _counting_spec(monkeypatch, calls)
    try:
        probes.probe_capability("test.cache", ttl_s=0)
        probes.probe_capability("test.cache", ttl_s=0)
        assert len(calls) == 2
        assert probes.cache_size() == 0
    finally:
        probes.PROBE_SPECS.pop("test.cache", None)


def test_changed_inputs_bust_the_cache(monkeypatch):
    """A cache that ignored WHAT was probed would hand a stale verdict to a
    fleet whose rows just changed."""
    calls: list[str] = []
    _counting_spec(monkeypatch, calls)
    try:
        probes.probe_capability("test.cache", model_ids=("m1",), ttl_s=300)
        probes.probe_capability("test.cache", model_ids=("m1", "m2"), ttl_s=300)
        assert len(calls) == 2
        assert probes.fingerprint(("m1",)) != probes.fingerprint(("m1", "m2"))
        assert probes.fingerprint(("m1",)) == probes.fingerprint(["m1"])
    finally:
        probes.PROBE_SPECS.pop("test.cache", None)


def test_probe_ttl_env_is_read_and_bad_values_degrade(monkeypatch):
    monkeypatch.delenv(probes.ENV_PROBE_TTL, raising=False)
    assert probes.probe_ttl_s() == probes.DEFAULT_PROBE_TTL_S == 300.0
    monkeypatch.setenv(probes.ENV_PROBE_TTL, "12.5")
    assert probes.probe_ttl_s() == 12.5
    monkeypatch.setenv(probes.ENV_PROBE_TTL, "0")
    assert probes.probe_ttl_s() == 0.0
    monkeypatch.setenv(probes.ENV_PROBE_TTL, "soon")
    assert probes.probe_ttl_s() == probes.DEFAULT_PROBE_TTL_S


# ---------------------------------------------------------------------------
# Live facts about THIS tree (no stubs).
# ---------------------------------------------------------------------------


def test_live_tts_probe_agrees_with_the_chatterbox_adapter():
    """The declared params really are TtsSpec's fields. If someone renames one
    without updating the descriptor, this fails HERE rather than at synthesis
    time on a worker."""
    from hugpy_video.intel.runners import tts_chatterbox
    spec = probes.PROBE_SPECS["audio.tts"]
    accepted, required = probes.accepted_params(tts_chatterbox.make_tts)
    assert set(spec.params) <= accepted
    assert required <= set(spec.params)
    result = probes.run_probe(spec, rows={}, model_ids=(), workers=None)
    assert _status(result, "runner_module").status is ProbeStatus.OK
    assert _status(result, "param_agreement").status is ProbeStatus.OK
    assert result.status is not ProbeStatus.FAIL


def test_live_word_timestamps_probe_proves_the_passthrough_field():
    """k98b landed whisper_schemas.TranscribeRequest.word_timestamps; if it is
    ever reverted this probe fails and the capability goes ineligible on its
    own."""
    result = probes.run_probe(
        probes.PROBE_SPECS["audio.transcribe.word_timestamps"], workers=None)
    assert _status(result, "param_agreement").status is ProbeStatus.OK


def test_live_probes_are_cheap_enough_for_a_page_load():
    import time
    for spec in probes.PROBE_SPECS.values():
        started = time.monotonic()
        probes.run_probe(spec, rows={}, model_ids=(), workers=None)
        assert time.monotonic() - started < 2.0, spec.capability
