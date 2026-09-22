"""k1 — prune-or-gate DEAD studio runner registrations: conformance for the runner
GATE mechanism in ``video_intel.studio.registry`` / ``.router``.

Background: ``models_seed.py`` registers several video engines (Hunyuan-video,
CogVideoX, Mochi, Open-Sora, SkyReels, AnimateDiff, FramePack, CodeFormer, and
LTX's t2v/i2v/av) whose runner MODULE does not exist yet in ``studio/runners/``
(only ``ltx_upscale`` is wired for the LTX framework). Before k1, the router could
still BIND a request to one of these (``registry.runner_for`` only checks registry
membership, not import-resolvability), so a caller could get an apparently-
successful ``Ok(ModelBinding)`` that would then fail one layer down at dispatch
(``produce.py``'s ``_DISPATCH.get(...)`` returning ``None`` -> ``Err(RUNNER_MISSING)``)
— or, worse, a dead engine could silently outrank a model that could actually
render. k1 gates these OUT of routing/dispatch consideration (registry.
``runner_available`` / ``runner_gate_reason`` / ``gated_runners`` /
``model_gate_reasons``) while leaving the seed entries and ``RUNNER_REGISTRY``
untouched — GATE, not prune: dropping the runner module into the tree re-enables
the engine with zero seed edits.

House style mirrors ``test_studio_conformance.py``: a plain python script with a
``__main__`` guard, numbered ``[n] PASS`` / ``[n] FAIL`` lines, nonzero exit iff
any check FAILED, every check independently run so one failure never masks the
rest. pytest is NOT installed in this venv.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_runner_gate.py
"""
from __future__ import annotations

import logging
import os
import sys

logging.disable(logging.INFO)

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_video.intel.studio import (
    MODEL_REGISTRY,
    Capability,
    CapabilityRequest,
    CapabilityRouter,
    ErrorCode,
    Framework,
    Precision,
    Resolution,
    RunnerSpec,
    StageError,
    Task,
    gated_runners,
    model_gate_reasons,
    runner_available,
    runner_gate_reason,
    validate_registry,
)
from hugpy_video.intel.studio.registry import (
    RUNNER_REGISTRY,
    _ENTRYPOINT_IMPORTABLE_CACHE,
    _is_entrypoint_importable,
    register_runner,
    runner_for,
)

# A representative sample of the DEAD engines named in the k1 report (VIDEO-TASK-
# SEQUENCES.md §2): their runner module does not exist under studio/runners/.
_DEAD = (
    (Framework.HUNYUAN, Task.T2V, "hugpy_video.intel.studio.runners.hunyuan"),
    (Framework.COGVIDEOX, Task.I2V, "hugpy_video.intel.studio.runners.cog"),
    (Framework.MOCHI, Task.T2V, "hugpy_video.intel.studio.runners.mochi"),
    (Framework.OPEN_SORA, Task.T2V, "hugpy_video.intel.studio.runners.opensora"),
    (Framework.SKYREELS, Task.I2V, "hugpy_video.intel.studio.runners.skyreels"),
    (Framework.ANIMATEDIFF, Task.MOTION_MODULE, "hugpy_video.intel.studio.runners.animatediff"),
    (Framework.FRAMEPACK, Task.STREAM_I2V, "hugpy_video.intel.studio.runners.framepack"),
    (Framework.CODEFORMER, Task.RESTORE_FACE, "hugpy_video.intel.studio.runners.codeformer"),
    (Framework.LTX, Task.T2V, "hugpy_video.intel.studio.runners.ltx"),
    (Framework.LTX, Task.I2V, "hugpy_video.intel.studio.runners.ltx"),
    (Framework.LTX, Task.AUDIO_VIDEO, "hugpy_video.intel.studio.runners.ltx"),
)

# The WORKING runners (produce.py's _DISPATCH) — must stay servable.
_LIVE = (
    (Framework.WAN, Task.I2V),
    (Framework.WAN, Task.T2V),
    (Framework.WAN, Task.VACE_CONTROL),
    (Framework.LTX, Task.UPSCALE),
    (Framework.RIFE, Task.INTERPOLATE),
    (Framework.FFMPEG, Task.INTERPOLATE),
    (Framework.FFMPEG, Task.UPSCALE),
    (Framework.SYNTHETIC, Task.I2V),
    (Framework.SYNTHETIC, Task.T2V),
)


# --------------------------------------------------------------------------- #
# 1 — every DEAD (framework, task) is registered (raw) but gated (servable=None)
# --------------------------------------------------------------------------- #
def test_dead_runners_registered_but_gated():
    for fw, task, module in _DEAD:
        spec = runner_for(fw, task)
        assert spec is not None, (
            f"{fw.value}/{task.value}: k1 must NOT prune the seed's RunnerSpec "
            f"(raw registry lookup must still find it)")
        assert runner_available(fw, task) is None, (
            f"{fw.value}/{task.value}: runner_available must be None (gated) — "
            f"its module ({module}) does not exist on this tree")
        reason = runner_gate_reason(fw, task)
        assert reason is not None and reason.startswith("runner_missing: "), (
            f"{fw.value}/{task.value}: expected 'runner_missing: <module>', got {reason!r}")
        assert module in reason, (
            f"{fw.value}/{task.value}: gate reason must name the missing module "
            f"({module!r}); got {reason!r}")


# --------------------------------------------------------------------------- #
# 2 — every LIVE (framework, task) stays fully servable (no false-positive gate)
# --------------------------------------------------------------------------- #
def test_live_runners_stay_available():
    for fw, task in _LIVE:
        spec = runner_available(fw, task)
        assert spec is not None, (
            f"{fw.value}/{task.value}: a REAL wired runner must stay servable")
        assert runner_gate_reason(fw, task) is None, (
            f"{fw.value}/{task.value}: a servable runner must have no gate reason")


# --------------------------------------------------------------------------- #
# 3 — gated_runners() / model_gate_reasons(): the queryable catalog surface
# --------------------------------------------------------------------------- #
def test_gated_runners_reporting_hook():
    g = gated_runners()
    g_pairs = {(fw, task) for fw, task, _reason in g}
    for fw, task, _module in _DEAD:
        assert (fw.value, task.value) in g_pairs, (
            f"gated_runners() must list {fw.value}/{task.value}")
    for fw, task in _LIVE:
        assert (fw.value, task.value) not in g_pairs, (
            f"gated_runners() must NOT list the live runner {fw.value}/{task.value}")
    # every reported reason is honest ("runner_missing: <module>"), never a bare
    # "not_registered" (these ARE registered — that branch is for an outright
    # unregistered (framework, task), not covered by any current seed row).
    assert all(reason.startswith("runner_missing: ") for _f, _t, reason in g), g

    # model-level view: a dead engine's model_id reports its gated task(s).
    reasons = model_gate_reasons("hunyuanvideo")
    assert reasons.get("t2v", "").startswith("runner_missing: "), reasons
    assert reasons.get("i2v", "").startswith("runner_missing: "), reasons
    # a live model reports nothing gated.
    assert model_gate_reasons("wan2.1-i2v-14b-720p") == {}, (
        "a fully-servable model must report no gated tasks")
    # an unknown model_id degrades to {} rather than raising.
    assert model_gate_reasons("does-not-exist") == {}


# --------------------------------------------------------------------------- #
# 4 — validate_registry() totality proof is UNCHANGED: k1 gates ROUTING, not the
#     zoo's structural completeness (a not-yet-built runner module must never
#     turn into an import-time RegistryError).
# --------------------------------------------------------------------------- #
def test_validate_registry_unaffected_by_gate():
    saved = os.environ.get("STUDIO_ALLOW_UNPINNED")
    try:
        os.environ["STUDIO_ALLOW_UNPINNED"] = "1"
        validate_registry()  # must not raise — every dead engine still has a
                              # DECLARED RunnerSpec (runner_for), just not a
                              # SERVABLE one (runner_available)
    finally:
        if saved is None:
            os.environ.pop("STUDIO_ALLOW_UNPINNED", None)
        else:
            os.environ["STUDIO_ALLOW_UNPINNED"] = saved


# --------------------------------------------------------------------------- #
# 5 — router: a request ONLY a dead engine could satisfy now returns an HONEST
#     Err naming the missing runner, not a silent Ok that fails at dispatch.
#     Portrait 720p T2V: wan2.1-t2v-1.3b/wan2.2-t2v-a14b do not list R_720P_V, so
#     hunyuanvideo was the ONLY pre-k1 "winner" here — and it can't actually run.
# --------------------------------------------------------------------------- #
def test_router_refuses_honestly_when_only_a_dead_engine_would_match():
    portrait_720p = Resolution(720, 1280, 24)
    req = CapabilityRequest(
        capability=Capability.T2V, target_resolution=portrait_720p, vram_budget_gb=100.0)
    res = CapabilityRouter().resolve(req)
    assert res.is_err(), (
        "k1: portrait-720p T2V must NOT silently bind hunyuanvideo (its runner "
        f"is gated); got {res}")
    err = res.error
    assert isinstance(err, StageError)
    assert err.code == ErrorCode.NO_CAPABLE_MODEL, err
    rejected_blob = " | ".join(v for k, v in err.context if k == "rejected")
    assert "hunyuanvideo" in rejected_blob, rejected_blob
    assert "runner_missing" in rejected_blob, (
        "the rejection must name the missing runner, not a generic dead end — "
        f"got: {rejected_blob}")
    assert "ImportError" not in str(err) and "Traceback" not in str(err), (
        "a gated engine must refuse as DATA, never surface a raw import traceback")


# --------------------------------------------------------------------------- #
# 6 — router: a DIRECT PIN of a gated model_id refuses honestly (never crashes,
#     never silently falls back to a different model — DIRECT MODEL CHOICE §).
# --------------------------------------------------------------------------- #
def test_pinned_dead_engine_refuses_honestly():
    req = CapabilityRequest(
        capability=Capability.T2V,
        target_resolution=Resolution(1280, 720, 24),
        vram_budget_gb=100.0,
        pinned_model_id="hunyuanvideo",
    )
    res = CapabilityRouter().resolve(req)
    assert res.is_err(), f"a pinned dead engine must refuse, not raise/succeed; got {res}"
    err = res.error
    assert isinstance(err, StageError)
    rejected_blob = " | ".join(v for k, v in err.context if k == "rejected") + str(err)
    assert "runner_missing" in rejected_blob or "runner gated" in rejected_blob, (
        f"pin refusal must name the missing runner; got {err}")


# --------------------------------------------------------------------------- #
# 7 — GATE not PRUNE, proven live: the mechanism keys off IMPORT-RESOLVABILITY
#     of the entrypoint string, not a decision baked in at seed-registration
#     time. A never-yet-queried module path that doesn't exist -> gated; the
#     SAME check against a module path that genuinely exists -> available. This
#     is the exact primitive that makes "drop the runner file in, zero seed
#     edits" true.
# --------------------------------------------------------------------------- #
def test_gate_keys_on_live_import_resolvability():
    missing = "hugpy_video.intel.studio.runners.__k1_never_created__"
    real = "hugpy_video.intel.studio.runners.synthetic"
    assert _is_entrypoint_importable(f"{missing}:run") is False, (
        "a module that genuinely does not exist must be reported not-importable")
    assert _is_entrypoint_importable(f"{real}:run_synthetic_i2v") is True, (
        "a module that genuinely exists must be reported importable")


# --------------------------------------------------------------------------- #
# 8 — literal "adding a fake runner file un-gates": register a fresh
#     (framework, task) pointing at a runner module that does not exist yet ->
#     gated; write the module file; the SAME (framework, task) is now available
#     — with ZERO change to the seed/registration call that declared it. Uses an
#     unused (CODEFORMER, T2V) key (no model declares that pair, so it cannot
#     perturb any other check's routing) and cleans up the file it wrote.
#
#     The import-resolvability check is CACHED per module path (the zoo is
#     static after import — see registry.py's docstring), so re-checking the
#     SAME module path within one process would return the stale pre-file
#     answer. The real operational flow is "drop the file in, then restart the
#     API process" (a fresh process = an empty cache) — see the k1 report's
#     "Live on dev after an API restart" verify step. This test drops the
#     module's cache entry to honestly simulate that restart rather than
#     asserting a hot-reload guarantee the design never made.
# --------------------------------------------------------------------------- #
def test_dropping_the_runner_file_ungates_with_zero_seed_edits():
    import shutil

    fw, task = Framework.CODEFORMER, Task.T2V   # unused pair — inert for routing
    assert (fw, task) not in RUNNER_REGISTRY, (
        "fixture precondition: (CODEFORMER, T2V) must not already be registered")

    runners_dir = os.path.dirname(os.path.abspath(
        __import__("hugpy_video.intel.studio.runners",
                    fromlist=["x"]).__file__))
    module_name = "_k1_fixture_runner"
    module_path = os.path.join(runners_dir, module_name + ".py")
    dotted = f"hugpy_video.intel.studio.runners.{module_name}"

    assert not os.path.exists(module_path), (
        f"fixture precondition: {module_path} must not already exist")

    # 1) register the spec FIRST, pointing at a module that does not exist yet.
    register_runner(RunnerSpec(fw, task, f"{dotted}:run", Precision.INT8))
    try:
        assert runner_available(fw, task) is None, (
            "a RunnerSpec pointing at a not-yet-created module must be gated")
        reason = runner_gate_reason(fw, task)
        assert reason == f"runner_missing: {dotted}", reason

        # 2) drop the runner file in — the k1 fix.
        with open(module_path, "w") as f:
            f.write("def run(*a, **kw):\n    raise NotImplementedError\n")
        try:
            # simulate the API-restart a real operator does after dropping a
            # runner module in (a fresh process never had this module cached).
            _ENTRYPOINT_IMPORTABLE_CACHE.pop(dotted, None)
            import importlib
            importlib.invalidate_caches()

            assert runner_available(fw, task) is not None, (
                "dropping the runner module into the tree must UN-GATE the "
                "(framework, task) with zero changes to the RunnerSpec/seed")
            assert runner_gate_reason(fw, task) is None, (
                "a servable runner must report no gate reason")
        finally:
            os.remove(module_path)
            _ENTRYPOINT_IMPORTABLE_CACHE.pop(dotted, None)
            shutil.rmtree(os.path.join(runners_dir, "__pycache__"), ignore_errors=True)
    finally:
        # RUNNER_REGISTRY has no unregister (by design — the zoo is frozen in
        # practice after import); leaving this inert fixture key behind is
        # harmless (CODEFORMER/T2V is claimed by no model), and this test's
        # own process exits right after the suite runs.
        pass


CHECKS = [
    ("dead engines: registered (raw) but gated (servable=None)", test_dead_runners_registered_but_gated),
    ("live engines: stay fully servable (no false-positive gate)", test_live_runners_stay_available),
    ("gated_runners()/model_gate_reasons(): queryable catalog surface", test_gated_runners_reporting_hook),
    ("validate_registry() totality proof unaffected by the gate", test_validate_registry_unaffected_by_gate),
    ("router: portrait-720p T2V (only hunyuanvideo matches) refuses honestly", test_router_refuses_honestly_when_only_a_dead_engine_would_match),
    ("router: a PINNED dead engine refuses honestly (never crashes)", test_pinned_dead_engine_refuses_honestly),
    ("gate keys on live import-resolvability, not a baked-in decision", test_gate_keys_on_live_import_resolvability),
    ("dropping a runner file in un-gates it with zero seed edits", test_dropping_the_runner_file_ungates_with_zero_seed_edits),
]


def main() -> int:
    passed = 0
    failed = 0
    for i, (name, fn) in enumerate(CHECKS, 1):
        try:
            fn()
        except Exception as exc:  # a conformance run surfaces EVERY divergence
            failed += 1
            print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"[{i}] PASS  {name}")
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
