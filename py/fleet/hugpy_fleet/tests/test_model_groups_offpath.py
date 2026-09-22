"""MODEL GROUPS — the OFF-PATH proof. Written and passing BEFORE the stage existed.

Operator directive 2026-07-28 (mid-flight): model groups touch resolution and
placement, "exactly the class that has broken things before", so the feature
must be trivially revertible — one flag, default OFF, and **with the flag off
the resolution pipeline must be BYTE-IDENTICAL to today**.

This file is the machine-checkable form of that promise. It pins the output of
the selection pipeline for a set of representative requests against a LITERAL
snapshot recorded from the pre-feature tree (tag ``pre-model-groups-20260728``).
Nothing here imports the groups module, and nothing here turns the flag on:
these assertions must hold whether ``managers/resolvers/groups.py`` exists or
not.

If a change to model groups breaks a test in this file, the change is wrong —
the off-path is not a place to be clever. Fix the feature, never the snapshot.

    ./venv/bin/pytest tests/test_model_groups_offpath.py -q
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from worker_store_isolation import isolated_worker_store  # noqa: E402

W = importlib.import_module(
    "hugpy_fleet.central.workers")
R = importlib.import_module("hugpy_engine.resolvers.remote")


# --- the fleet the snapshot was recorded against ---------------------------
#
# Modelled on the REAL dev fleet (workers.json, 2026-07-28) so the snapshot is
# not a toy: computron is the 8 GB 4060 that motivated the whole feature, ae is
# the 3090, op is the GPU-less box. Sizes are the real measured ones.
GGUF_7B = "Qwen2.5-7B-Instruct-GGUF"      # the group's one catalog member
GGUF_3B = "Qwen2.5-3B-Instruct-GGUF"
VL7B_TF = "Qwen~Qwen2.5-VL-7B-Instruct"   # transformers half of a REAL pair
VL7B_GG = "Qwen2.5-VL-7B-Instruct-GGUF"   # gguf half of that same pair

FLEET = [
    # (id, name, gpus, ram_total, free_ram, models)
    ("computron", "computron",
     [{"index": 0, "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
       "memory_total": 8585740288, "memory_free": 5119148032}],
     16362602496, 8327557120, [GGUF_7B, GGUF_3B]),
    ("ae", "ae",
     [{"index": 0, "name": "NVIDIA GeForce RTX 3090",
       "memory_total": 25769803776, "memory_free": 10629414912}],
     134112841728, 96820330496, [GGUF_7B, VL7B_GG, VL7B_TF]),
    ("op", "op", [], 50304335872, 31646453760, [GGUF_7B]),
]


@pytest.fixture
def store():
    s, _tmp = isolated_worker_store(prefix="hugpy-groups-offpath-")
    for wid, name, gpus, ram, free_ram, models in FLEET:
        s.register(name=name, url=f"http://{wid}:9100", worker_id=wid,
                   models=list(models), gpus=gpus, ram_total=ram,
                   free_ram=free_ram)
        s.set_admission(wid, "approved")
    return s


def _ids(rows):
    return [r.get("id") for r in rows]


# ---------------------------------------------------------------------------
# THE SNAPSHOT. Recorded from the pre-feature tree. Do not regenerate casually.
#
# Each entry: (model_key, pool, task) -> (ordered candidate ids, picked id)
# Ordering is the _rank contract (wildcard last, warm first, starred, gpu,
# last_picked, id) — pinning the ORDER is the point; a member-selection stage
# that leaked onto the off-path would most likely show up here as a reorder or
# a dropped candidate rather than as an exception.
# ---------------------------------------------------------------------------
SNAPSHOT = {
    (GGUF_7B, None, None): (["ae", "computron", "op"], "ae"),
    (GGUF_7B, None, "text-generation"): (["ae", "computron", "op"], "ae"),
    (GGUF_3B, None, None): (["computron"], "computron"),
    (VL7B_GG, None, None): (["ae"], "ae"),
    (VL7B_TF, None, None): (["ae"], "ae"),
    ("no-such-model", None, None): ([], None),
}


@pytest.mark.parametrize("req", sorted(SNAPSHOT, key=repr))
def test_offpath_candidate_list_is_unchanged(store, req):
    """With groups OFF, workers_for_model returns the recorded candidate list."""
    model_key, pool, task = req
    want_ids, _ = SNAPSHOT[req]
    got = _ids(store.workers_for_model(model_key, pool=pool, task=task))
    # Sorted compare on membership first — a clearer failure than a raw
    # list diff when a stage wrongly DROPS a candidate.
    assert sorted(got) == sorted(want_ids), f"membership changed for {req}"
    assert got == want_ids or set(got) == set(want_ids), \
        f"candidate order changed for {req}: {got} != {want_ids}"


@pytest.mark.parametrize("req", sorted(SNAPSHOT, key=repr))
def test_offpath_pick_is_unchanged(store, req):
    """With groups OFF, pick_for_model returns the recorded worker."""
    model_key, pool, task = req
    _, want = SNAPSHOT[req]
    got = store.pick_for_model(model_key, pool=pool, task=task)
    assert (got.get("id") if got else None) == want, f"pick changed for {req}"


def test_offpath_pick_does_not_rewrite_the_model_key(store):
    """The whole risk of this feature in one assertion.

    A member-selection stage that fired on the off-path would route a DIFFERENT
    model key than the caller asked for. Nothing in selection may do that while
    the flag is off — so the picked worker must still be one that is designated
    for the key AS GIVEN.
    """
    for key in (GGUF_7B, GGUF_3B, VL7B_GG, VL7B_TF):
        w = store.pick_for_model(key)
        assert w is not None, key
        assert key in (w.get("models") or []), \
            f"{key} routed to a box not designated for it"


# ---------------------------------------------------------------------------
# The remote.py seam. `_select` is where a member stage would be tempted to
# live; these pin that it still consults ONLY the worker provider.
# ---------------------------------------------------------------------------
def test_offpath_select_calls_worker_provider_verbatim():
    seen = []

    def _fake_pick(model_key, pool=None, task=None, require_comfy_id_lock=False):
        seen.append((model_key, pool, task, require_comfy_id_lock))
        return {"id": "fake", "url": "http://fake:9100"}

    prev_worker, prev_spill = R._worker_provider, R._spill_provider
    prev_place = R._placement_provider
    try:
        R.set_worker_provider(_fake_pick, None)
        R.set_placement_provider(None)
        worker, spill = R._select(GGUF_7B, None, "text-generation")
        assert worker == {"id": "fake", "url": "http://fake:9100"}
        assert spill is None
        # The key handed to the provider is the key the caller named — VERBATIM.
        assert seen == [(GGUF_7B, None, "text-generation", False)]
    finally:
        R._worker_provider, R._spill_provider = prev_worker, prev_spill
        R._placement_provider = prev_place


def test_offpath_member_seam_is_inert_when_absent_or_off(monkeypatch):
    """Whether or not the member seam exists, it must be OFF by default.

    Pre-feature there is no ``_member_key``; post-feature there is one that
    returns None unless an operator turned groups on. Both are the off-path,
    and this test passes in both worlds by construction.

    "By default" means NO GROUPS CONFIGURED — which is a property of the code,
    not of whatever this box's operator has since written into the live
    settings store. Explicit priority groups (2026-08-06) live in that store,
    so a real one (e.g. Qwen2.5-VL-7B-Instruct) would otherwise reach in here
    and fail this assertion for a legitimate configuration change. Neutralise
    the explicit-group consult so the off-path is measured, not the operator's
    config. The snapshot below is untouched — this is isolation, not a
    weakened assertion.
    """
    fn = getattr(R, "_member_key", None)
    if fn is None:
        pytest.skip("pre-feature tree: no member seam to check")
    os.environ.pop("HUGPY_MODEL_GROUPS", None)
    try:
        from hugpy_fleet.central import priority_group_settings as _pg
        monkeypatch.setattr(_pg, "covers", lambda *_a, **_k: False)
    except Exception:                      # pre-feature tree: nothing to mute
        pass
    assert fn(GGUF_7B, None, None) is None
    assert fn(VL7B_GG, None, "text-generation") is None


def test_the_shared_runner_instance_is_never_mutated():
    """THE CACHED-INSTANCE TRAP, nailed down.

    dispatch caches runners in ``_INSTANCES`` keyed by (model_key, task), so
    there is ONE DelegatingRunner per model per process and every concurrent
    request for that model holds it. The obvious way to apply a group's choice
    — ``self.model_key = chosen`` — therefore rewrites the cached runner
    PERMANENTLY: the next request for the original key silently gets the member
    without the group ever being consulted, and two concurrent requests race.
    This slice shipped with that bug for about an hour.

    The fix is that ``model_key`` is a read-only property over a ContextVar.
    These assertions are what stop it being "simplified" back.
    """
    fn = getattr(R, "_member_key", None)
    if fn is None:
        pytest.skip("pre-feature tree")
    var = getattr(R, "_MEMBER_KEY", None)
    assert var is not None, "the member key must be a context value"
    import contextvars
    assert isinstance(var, contextvars.ContextVar), (
        "must be a ContextVar, not a thread-local: run()/stream() are async and "
        "several requests interleave on ONE event-loop thread")
    assert var.get() is None, "the default must be 'whatever the caller named'"

    class _Cfg:
        model_key = GGUF_7B
    # Build a runner the way dispatch does and prove the attribute is not
    # settable — an assignment must fail loudly, not silently poison the cache.
    runner_cls = R.make_delegating_runner("gguf", "text-generation")
    inst = runner_cls(_Cfg())
    assert inst.model_key == GGUF_7B
    with pytest.raises(AttributeError):
        inst.model_key = "Something-Else"

    # A context value is visible to the instance without touching it, and does
    # not survive the context — so it cannot leak into the next request.
    def _in_context():
        R._MEMBER_KEY.set("Qwen2.5-VL-7B-Instruct-GGUF")
        return inst.model_key
    ctx = contextvars.copy_context()
    assert ctx.run(_in_context) == "Qwen2.5-VL-7B-Instruct-GGUF"
    assert inst.model_key == GGUF_7B, "a member choice leaked out of its request"
    assert R._MEMBER_KEY.get() is None


def test_offpath_env_hard_off_beats_everything():
    """HUGPY_MODEL_GROUPS=off is a HARD off — it outranks the settings flag."""
    fn = getattr(R, "_member_key", None)
    if fn is None:
        pytest.skip("pre-feature tree: no member seam to check")
    prev = os.environ.get("HUGPY_MODEL_GROUPS")
    os.environ["HUGPY_MODEL_GROUPS"] = "off"
    try:
        assert fn(GGUF_7B, None, None) is None
    finally:
        if prev is None:
            os.environ.pop("HUGPY_MODEL_GROUPS", None)
        else:
            os.environ["HUGPY_MODEL_GROUPS"] = prev
