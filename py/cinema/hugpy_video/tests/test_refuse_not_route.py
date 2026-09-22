"""PROVE the router REFUSES instead of routing into something that cannot finish.

THE DEFECT THIS SUITE PINS DOWN (measured on the live fleet 2026-07-27 —
CAPABILITY-VIABILITY-MAP.md, MODEL-POOL-INVENTORY.md). 25 of 59 capability routes
resolved cleanly, were admitted, queued and STARTED, and only then died in a
runner. Two mechanisms did all of it:

  * a STUB RUNNER — ``ltx_upscale.py`` and ``rife_interpolate.py`` exist as modules
    and return ``Err`` on every path. The k1 gate asks ``find_spec``, so a stub
    PASSES it; ``router._score``'s ``real_first`` then ranks the stub ABOVE the
    working ffmpeg last-resort. Measured: upres bound the stub at every budget
    >= 8 GB, interp at every budget >= 3 GB, against ae's stable ~21.6 GB autofit.
    Both live product surfaces, dead, on the only render box.
  * a CAPABILITY WITH NO PROVEN TUPLE — ``keyframe`` resolved to
    wan2.1-i2v-14b-720p and would have rendered an ordinary i2v clip, because
    nothing in the tree reads an end frame. Not a broken capability: a RENAME of
    i2v, and the same silent-wrong-output shape as the id_lock bug fixed the same
    day.

⚠ THE SAME SHAPE WAS THEN FOUND INSIDE THE FIX (2026-07-27, second pass). The first
cut of ``studio/presets.py`` published a row (``clip-control-480p``) advertising
motion + inpaint + outpaint + retake on one VACE binding, which made all four
"servable" and therefore exempt from every refusal below — while the route rejected
a control image for all four, and without one rendered a plain full restyle for all
four. The table that decided what refuses was itself publishing a phantom. It was
split on the evidence: ``motion`` kept (``control_image`` IS a distinct wan_vace
branch), the other three demoted to refusals (no mask / canvas / frame-range input
exists anywhere in the spine). This suite needs no per-capability edit for that —
every assertion here is derived from ``presets``, which is the point — but the count
of preset-covered capabilities moved from 11 to 8, so the arity guard below did.

So the suite asserts BOTH directions, because either one alone is worthless:

  1. NOTHING THAT WORKS WAS NARROWED. Every capability a ratified ``RenderPreset``
     covers still resolves, to a real (non-stub, non-empty) model, at the geometry
     and budget the fleet actually renders at.
  2. EVERYTHING DEAD NOW REFUSES EARLY — at ``CapabilityRouter.resolve``, before a
     manifest is built or a runner is dispatched — with text that names WHY and
     names a preset the caller CAN have instead.

And the properties that make a refusal usable rather than merely correct: it is
DATA (never a raise), it is not-retryable, a stub runner is never miscoded as
``WEIGHTS_MISSING`` (that exact lie sent operators to re-download 3.1 GB that was
already on disk), and it is never quietly absorbed by the synthetic prover.

Truth sources are deliberately NOT this file: the preset table
(``studio/presets.py``, proven row-by-row against the store by
``test_presets.py``) says what works, and the router is asked what it does. This
suite only compares them.

Run:
  cd /srv/share/projects/hugpy/dev
  abstract_hugpy_dev/venv/bin/python -m pytest \
      abstract_hugpy_dev/tests/studio/test_refuse_not_route.py -q -p no:randomly
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from hugpy_video.intel.studio import (
    Capability,
    CapabilityRequest,
    CapabilityRouter,
    ErrorCode,
    Framework,
    MODEL_REGISTRY,
    Resolution,
    StageError,
)
from hugpy_video.intel.studio import router as router_mod
from hugpy_video.intel.studio.registry import runner_for
from hugpy_video.intel.studio.presets import (
    STUB_RUNNER_MODULES,
    ZERO_BYTE_MODELS,
    all_presets,
    presets_for,
    servable_capabilities,
    unservable_capabilities,
)

# ae (RTX 3090) is the ONLY studio render target and its autofit budget is a stable
# ~21.6 GB since the budget-to-capacity change (studio_i2v.py: capacity - max(10%,
# 2GB) of 23.56 GiB). Every "does the real fleet get this?" assertion below uses it,
# because that stability is precisely what made the two stub runners win 100% of the
# time instead of only sometimes.
AE_AUTOFIT_GB = 21.6
# The three budgets the viability map bisected the dead bindings at: below the rife
# floor, between the rife and ltxv floors, and ae's real one.
BUDGET_SWEEP = (2.0, 6.0, 12.0, AE_AUTOFIT_GB)

R_480P = Resolution(832, 480, 16)     # the fleet's proven Wan geometry
R_SMALL = Resolution(320, 180, 12)    # inside every envelope; the cheap rung
R_720P = Resolution(1280, 720, 24)

GEOMETRY_SWEEP = (R_SMALL, R_480P, R_720P)


def _req(capability: Capability, resolution: Resolution, budget: float,
         **kw) -> CapabilityRequest:
    return CapabilityRequest(capability=capability, target_resolution=resolution,
                             vram_budget_gb=budget, **kw)


def _runner_module(framework: Framework, task) -> str:
    """The dotted module a binding would dispatch into ("mod:callable" -> "mod")."""
    spec = runner_for(framework, task)
    assert spec is not None, f"no RunnerSpec for ({framework}, {task})"
    return spec.entrypoint.split(":", 1)[0]


def _err(res) -> StageError:
    assert res.is_err(), f"expected a refusal, got a binding: {res}"
    err = res.error
    assert isinstance(err, StageError), f"the Err payload must be a StageError value, got {err!r}"
    return err


def _ctx(err: StageError, key: str) -> str:
    return " | ".join(v for k, v in err.context if k == key)


# --------------------------------------------------------------------------- #
# 1 — NOTHING THAT WORKS WAS NARROWED
#     Every capability a ratified preset covers still binds, and binds something
#     that can actually finish. ASSEMBLE is excluded: it is an orchestration stage
#     (registry.PLANNED_CAPABILITIES) that no model declares and the movie runner
#     composes out of the OTHER presets — it was never a router binding.
# --------------------------------------------------------------------------- #
def test_every_preset_capability_still_resolves_to_a_viable_model():
    router = CapabilityRouter()
    checked = 0
    for preset in all_presets():
        for cap in preset.capabilities:
            if cap is Capability.ASSEMBLE:
                continue
            res = router.resolve(_req(cap, R_480P, AE_AUTOFIT_GB))
            assert res.is_ok(), (
                f"preset {preset.preset_id!r} covers {cap.value!r}, so the router "
                f"MUST still bind it at 832x480 / {AE_AUTOFIT_GB}GB — this slice "
                f"must not narrow a working route; got {res.error}")
            binding = res.unwrap()
            # ...and bind something that can FINISH, which is the whole point: a
            # green resolve into a stub or an empty registry row is the defect.
            assert binding.model_id not in ZERO_BYTE_MODELS, (
                f"{cap.value}: bound {binding.model_id}, which has zero bytes on "
                f"the shared store")
            module = _runner_module(binding.framework, binding.task)
            assert module not in STUB_RUNNER_MODULES, (
                f"{cap.value}: bound {binding.model_id}, whose runner {module} "
                f"returns Err on every path")
            checked += 1
    # SEVEN model-bound capabilities today: t2v, id_lock, v2v, motion, i2v, upres,
    # interp (ASSEMBLE is skipped above — orchestration, never a binding). This was 8
    # before the clip-control split; it is a floor, not an equality, so proving a
    # ninth preset does not have to touch this line — but LOSING one still fails here
    # rather than quietly shrinking the surface this test covers.
    assert checked >= 7, f"expected the preset capability set, only checked {checked}"


def test_working_capabilities_bind_across_the_budget_sweep():
    """The stub-runner fix must not be budget-shaped. Every servable capability that
    binds at ae's autofit must still bind at the lower budgets the viability map
    bisected at — the dead bindings were reached ONLY above a floor (rife 3 GB, ltxv
    8 GB), so a fix that only holds at one budget would just move the cliff."""
    router = CapabilityRouter()
    for cap in sorted(servable_capabilities(), key=lambda c: c.value):
        if cap is Capability.ASSEMBLE:
            continue
        top = router.resolve(_req(cap, R_480P, AE_AUTOFIT_GB))
        assert top.is_ok(), f"{cap.value} must bind at ae's autofit budget: {top}"
        for budget in BUDGET_SWEEP:
            res = router.resolve(_req(cap, R_480P, budget))
            if res.is_err():
                # A BUDGET refusal at a low budget is correct and expected
                # (wan2.1-t2v-1.3b's floor is 5 GB, vace-1.3b's is 6 GB). It surfaces
                # as VRAM_EXCEEDED when every candidate missed on budget, and as
                # NO_CAPABLE_MODEL when the set was mixed (some too big, one
                # runner-gated) — both honest. What must NEVER happen is a refusal
                # that blames the runner or the store at a budget where the same
                # capability demonstrably binds higher up: that would mean the
                # viability gate had swallowed a working row.
                assert res.error.code not in (
                    ErrorCode.RUNNER_MISSING, ErrorCode.WEIGHTS_MISSING), (
                    f"{cap.value} @ {budget}GB refused with {res.error.code.value} "
                    f"even though it binds at {AE_AUTOFIT_GB}GB — the viability gate "
                    f"must not eat a row that only failed on budget: {res.error}")
                assert "budget" in _ctx(res.error, "rejected"), (
                    f"{cap.value} @ {budget}GB refused without a single budget "
                    f"rejection: {res.error}")
                continue
            binding = res.unwrap()
            module = _runner_module(binding.framework, binding.task)
            assert module not in STUB_RUNNER_MODULES, (
                f"{cap.value} @ {budget}GB bound the stub {module}")


# --------------------------------------------------------------------------- #
# 2 — THE TWO REVIVED SURFACES
#     upres/interp were DEAD on ae at every realistic budget. They are the only
#     entries in the whole viability map that go from dead to working with zero
#     downloads, so they get an explicit, budget-swept assertion rather than being
#     folded into the generic sweep above.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "capability,expected_model,expected_framework",
    [
        (Capability.UPRES, "ffmpeg-lanczos-upscale", Framework.FFMPEG),
        (Capability.INTERP, "ffmpeg-minterpolate", Framework.FFMPEG),
    ],
)
def test_enhance_binds_the_working_runner_at_every_budget(
        capability, expected_model, expected_framework):
    router = CapabilityRouter()
    for geometry in GEOMETRY_SWEEP:
        for budget in BUDGET_SWEEP:
            res = router.resolve(_req(capability, geometry, budget))
            assert res.is_ok(), (
                f"{capability.value} @ {geometry.width}x{geometry.height} / "
                f"{budget}GB must bind the working ffmpeg transform: {res.error}")
            binding = res.unwrap()
            assert binding.model_id == expected_model, (
                f"{capability.value} @ {budget}GB bound {binding.model_id}; the "
                f"premium row is a stub runner and must never outrank the transform "
                f"that actually produces pixels")
            assert binding.framework is expected_framework


# --------------------------------------------------------------------------- #
# 3 — DEAD CAPABILITIES REFUSE, AND THE REFUSAL IS USEFUL
# --------------------------------------------------------------------------- #
def test_unservable_capabilities_refuse_and_name_an_alternative():
    router = CapabilityRouter()
    live_preset_ids = {p.preset_id for p in all_presets()}
    dead = sorted(unservable_capabilities(), key=lambda c: c.value)
    assert dead, "the preset table claims every capability is servable — implausible"

    for cap in dead:
        for geometry in GEOMETRY_SWEEP:
            for budget in BUDGET_SWEEP:
                err = _err(router.resolve(_req(cap, geometry, budget)))
                text = str(err)
                # WHY: a measured blocker, not "unsupported".
                assert cap.value in text, f"{cap.value}: refusal does not name it: {text}"
                assert len(_ctx(err, "reason")) > 20, (
                    f"{cap.value}: refusal carries no reason: {err.context}")
                # WHAT INSTEAD: at least one preset the caller can actually have.
                named = {pid for pid in live_preset_ids if pid in text}
                assert named, (
                    f"{cap.value}: refusal names no alternative the fleet CAN "
                    f"render — that is the message that teaches nothing: {text}")
                # DISTINGUISHABLE FROM A CRASH.
                assert _ctx(err, "retryable") == "false", (
                    f"{cap.value}: a deterministic refusal must be marked "
                    f"not-retryable: {err.context}")


@pytest.mark.parametrize("cap_value", ["inpaint", "outpaint", "retake"])
def test_the_demoted_vace_capabilities_no_longer_bind_a_restyle(cap_value):
    """THE SECOND-PASS REGRESSION, pinned. These three used to RESOLVE — cleanly, to
    wan2.1-vace-1.3b at fp16 — and the resulting render was byte-for-byte the v2v
    path, because a VACE call with no mask makes diffusers fill mask=ones_like(video)
    (installed diffusers 0.39.0, pipeline_wan_vace.py:457) so every frame is reactive.
    The caller asked to inpaint a region and got their whole clip repainted, with a
    200 and no warning.

    That the model still DECLARES the capability (models_seed lists all four on the
    VACE rows) is exactly why this has to be tested at the router: the structural join
    still succeeds, and only the preset-derived capability gate stops it."""
    cap = Capability(cap_value)
    router = CapabilityRouter()
    for geometry in GEOMETRY_SWEEP:
        for budget in BUDGET_SWEEP + (100.0,):
            err = _err(router.resolve(_req(cap, geometry, budget)))
            assert err.code is ErrorCode.NO_CAPABLE_MODEL, err
            assert cap_value in str(err), err
            # It must point at the capability that renders a restyle ON PURPOSE,
            # rather than leaving the caller to rediscover it by accident.
            assert "v2v" in str(err), (
                f"{cap_value}: the refusal must name v2v as what actually renders "
                f"what this request would have produced: {err}")
            assert _ctx(err, "retryable") == "false", err.context


def test_motion_survived_the_split_and_still_binds_the_vace_row():
    """The OTHER half of the same decision, and the half a narrowing fix would have
    got wrong. ``control_image`` is a real wan_vace branch (the still is repeated
    across num_frames as the pipeline's `video=` control channel — not v2v's decoded
    source frames, not id_lock's reference latents), so motion was KEPT as its own
    preset rather than swept away with its three phantom siblings. If a future edit
    ever demotes it too, that must be a deliberate act, not a side effect."""
    router = CapabilityRouter()
    covered = presets_for(Capability.MOTION)
    assert [p.preset_id for p in covered] == ["clip-motion-480p"], covered
    res = router.resolve(_req(Capability.MOTION, R_480P, AE_AUTOFIT_GB))
    assert res.is_ok(), f"motion must still bind at ae's geometry and budget: {res}"
    binding = res.unwrap()
    assert binding.model_id == "wan2.1-vace-1.3b", binding
    assert _runner_module(binding.framework, binding.task) not in STUB_RUNNER_MODULES


def test_keyframe_no_longer_silently_returns_an_i2v_clip():
    """The specific regression. keyframe bound wan2.1-i2v-14b-720p at every budget
    >= 14 GB and would have produced a perfectly plausible clip that ignored the end
    frame entirely — because StudioI2VSpec has no end frame, the route parses none,
    the manifest carries none and run_wan_i2v passes no ``last_image``."""
    router = CapabilityRouter()
    for budget in BUDGET_SWEEP + (100.0,):
        err = _err(router.resolve(_req(Capability.KEYFRAME, R_480P, budget)))
        assert err.code is ErrorCode.NO_CAPABLE_MODEL, err
        assert "keyframe" in str(err)
        assert "wan2.1-i2v-14b-720p" not in _ctx(err, "available"), (
            "the i2v-14B row must never be offered as a keyframe alternative — "
            "that is the rename this refusal exists to stop")


# --------------------------------------------------------------------------- #
# 4 — CODE HYGIENE: a stub runner is RUNNER_MISSING, never WEIGHTS_MISSING
#     ltx_upscale.py reports WEIGHTS_MISSING while its 3.1 GB ARE on disk and it
#     FOUND them. Reproduced on central 2026-07-27. Getting this code right is the
#     difference between "wire the runner" and "download it again".
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "capability,stub_model",
    [
        (Capability.UPRES, "ltxv-spatial-upscaler-0.9.7"),
        (Capability.INTERP, "rife-practical"),
    ],
)
def test_pinned_stub_runner_refuses_as_runner_missing(capability, stub_model):
    err = _err(CapabilityRouter().resolve(
        _req(capability, R_480P, AE_AUTOFIT_GB, pinned_model_id=stub_model)))
    assert err.code is ErrorCode.RUNNER_MISSING, (
        f"a stub runner is a WIRING gap, not a download: expected RUNNER_MISSING, "
        f"got {err.code.value} — {err}")
    assert err.code is not ErrorCode.WEIGHTS_MISSING
    assert "weights" not in str(err).lower() or "stub" in str(err).lower(), (
        f"the refusal must not read as a missing download: {err}")
    rejected = _ctx(err, "rejected")
    assert stub_model in rejected and "stub" in rejected, rejected
    # ...and it still says what the caller CAN have.
    assert _ctx(err, "available"), err.context


def test_ltx_upscaler_weights_are_present_so_the_refusal_must_not_blame_them():
    """Guards the exact miscoding, from the store side. ltxv-spatial-upscaler is the
    one row that is COMPLETE on disk and still cannot run — so if the refusal ever
    says weights_missing again, this fails whether or not the store is mounted."""
    assert "ltxv-spatial-upscaler-0.9.7" not in ZERO_BYTE_MODELS, (
        "the preset table records this row as having bytes on the store; a refusal "
        "for it can never legitimately be a weights problem")
    err = _err(CapabilityRouter().resolve(
        _req(Capability.UPRES, R_480P, AE_AUTOFIT_GB,
             pinned_model_id="ltxv-spatial-upscaler-0.9.7")))
    assert err.code is not ErrorCode.WEIGHTS_MISSING, err


def test_zero_byte_weights_gate_codes_weights_missing():
    """The OTHER half of the viability gate, exercised deliberately.

    Today it is unreachable in production: every one of the twelve zero-byte rows
    ALSO has a runner module that does not exist, so the k1 gate rejects it first.
    That is exactly why it must be tested — the k1 design re-arms an engine the
    moment its runner module lands (registry.py's "RUNNER GATE"), and the next such
    landing would otherwise bind a model with nothing behind it. Neutralizing the
    stub set makes rife-practical (0 bytes AND a stub) fall through to the weights
    gate, which is the same ladder a future re-armed engine would take."""
    saved = router_mod.STUB_RUNNER_MODULES
    try:
        router_mod.STUB_RUNNER_MODULES = frozenset()
        err = _err(CapabilityRouter().resolve(
            _req(Capability.INTERP, R_480P, AE_AUTOFIT_GB,
                 pinned_model_id="rife-practical")))
    finally:
        router_mod.STUB_RUNNER_MODULES = saved
    assert err.code is ErrorCode.WEIGHTS_MISSING, err
    assert "weights absent" in _ctx(err, "rejected"), err.context
    assert _ctx(err, "retryable") == "false", err.context


# --------------------------------------------------------------------------- #
# 5 — A REFUSAL IS DATA, AND IT IS NOT A CRASH
# --------------------------------------------------------------------------- #
def test_no_request_in_the_grid_ever_raises():
    """INV-3 across the whole surface: every (capability, geometry, budget) either
    binds or returns a StageError. A raise here would reach the caller as a 500 and
    be indistinguishable from the fleet being broken."""
    router = CapabilityRouter()
    for cap in Capability:
        for geometry in GEOMETRY_SWEEP:
            for budget in BUDGET_SWEEP:
                res = router.resolve(_req(cap, geometry, budget))
                if res.is_err():
                    assert isinstance(res.error, StageError)
                    assert "Traceback" not in str(res.error)


def test_every_refusal_code_is_classified_not_retryable_by_the_bus():
    """The refusal must not be re-driven. ``StageError`` carries no ``retryable``
    field — the bus classifies at the boundary — so this checks BOTH halves: the
    router stamps ``retryable=false`` into the value, AND every code it emits sits
    outside the bus's retryable set. Either alone could drift from the other."""
    try:
        from hugpy_video.intel.runners.studio_i2v import _RETRYABLE_CODES
    except Exception as exc:  # noqa: BLE001 — the bus adapter is another package
        pytest.skip(f"bus adapter not importable here: {exc}")

    router = CapabilityRouter()
    seen: set[str] = set()
    for cap in Capability:
        for geometry in GEOMETRY_SWEEP:
            for budget in BUDGET_SWEEP:
                res = router.resolve(_req(cap, geometry, budget))
                if res.is_ok():
                    continue
                err = res.error
                seen.add(err.code.value)
                assert err.code.value not in _RETRYABLE_CODES, (
                    f"{cap.value}: refusal code {err.code.value} would be RETRIED by "
                    f"the bus — a deterministic refusal must settle the job")
                assert _ctx(err, "retryable") == "false", (
                    f"{cap.value}: refusal is not marked not-retryable: {err.context}")
    assert seen, "the grid produced no refusals at all — the sweep is not exercising"


def test_refusal_is_never_downgraded_into_a_synthetic_render():
    """The synthetic prover is a real binding for real capabilities at a tiny budget,
    and it must stay that way — but it must never absorb a refusal. The capability
    gate returns BEFORE the candidate scan precisely so a dead capability can never
    come back as procedural noise wearing the right shape."""
    router = CapabilityRouter()
    for cap in sorted(unservable_capabilities(), key=lambda c: c.value):
        for budget in (0.1, 0.5, 2.0, AE_AUTOFIT_GB):
            res = router.resolve(_req(cap, R_SMALL, budget))
            assert res.is_err(), (
                f"{cap.value} @ {budget}GB came back as a binding to "
                f"{res.unwrap().model_id if res.is_ok() else '?'} — a capability no "
                f"preset covers must refuse, not fall back")
    # And the inverse, so this test cannot pass by killing the prover: a SERVABLE
    # capability at a sub-real budget still binds (synthetic-i2v's floor is 0.1 GB).
    tiny = router.resolve(_req(Capability.I2V, R_SMALL, 0.5))
    assert tiny.is_ok(), f"the synthetic prover path must be intact: {tiny}"
    assert MODEL_REGISTRY[tiny.unwrap().model_id].synthetic is True


# --------------------------------------------------------------------------- #
# 6 — EARLY: the refusal lands before a manifest exists and before any runner is
#     reached. "The user discovers this only after a job is accepted, queued and
#     started" is the failure being deleted, so proving WHERE the refusal happens
#     matters as much as proving that it happens.
# --------------------------------------------------------------------------- #
def test_dead_capability_refuses_before_any_runner_is_dispatched():
    from hugpy_video.intel.studio import produce as produce_mod
    from hugpy_video.intel.studio.env import StudioEnv

    class _Trap(dict):
        def get(self, *_a, **_kw):   # pragma: no cover - the assertion IS the body
            raise AssertionError(
                "produce_clip reached the dispatch table for a capability the "
                "router should have refused — the refusal is not early")

    env = StudioEnv(
        output_root="/out", weights_root="/weights", manifest_root="/manifests",
        master_colorspace="rec709", master_fps=16, max_vram_gb=23.56,
        loudness_target_lufs=-14.0, allow_unpinned=True,
    )
    out_root = tempfile.mkdtemp(prefix="studio-refuse-")
    saved = produce_mod._DISPATCH
    try:
        produce_mod._DISPATCH = _Trap()
        res = produce_mod.produce_clip(
            _req(Capability.KEYFRAME, R_480P, AE_AUTOFIT_GB),
            env=env, out_root=out_root)
        assert res.is_err(), f"a dead capability must not produce a clip: {res}"
        assert res.error.code is ErrorCode.NO_CAPABLE_MODEL, res.error
        assert "keyframe" in str(res.error)
        # Nothing was written: no manifest, no frames, no clip. A refusal costs the
        # caller one round trip, not a queue slot and six minutes of denoise.
        assert os.listdir(out_root) == [], (
            f"a refused request left artifacts behind: {os.listdir(out_root)}")
    finally:
        produce_mod._DISPATCH = saved
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 7 — the join itself: preset coverage and router behaviour agree, exhaustively.
#     Written as one table so a future capability cannot be added to the enum and
#     silently be neither servable nor refusable — the state this slice deleted.
# --------------------------------------------------------------------------- #
def test_every_capability_is_either_served_or_refused_by_name():
    router = CapabilityRouter()
    for cap in Capability:
        res = router.resolve(_req(cap, R_480P, AE_AUTOFIT_GB))
        covered = presets_for(cap)
        if res.is_ok():
            assert covered, (
                f"{cap.value} binds {res.unwrap().model_id} but NO preset covers it "
                f"— an unproven route is exactly what this slice refuses")
            continue
        err = res.error
        if cap is Capability.ASSEMBLE:
            # Orchestration: covered by a preset, composed by the movie runner, never
            # a model binding. It must still refuse by NAME and point at that preset.
            assert "movie-480p" in str(err), err
            continue
        assert not covered or err.code in (
            ErrorCode.VRAM_EXCEEDED, ErrorCode.RESOLUTION_UNSUPPORTED), (
            f"{cap.value} is covered by {[p.preset_id for p in covered]} yet refused "
            f"with {err.code.value} at the fleet's own geometry and budget: {err}")
        assert _ctx(err, "available"), (
            f"{cap.value}: refusal carries no 'available' menu: {err.context}")
