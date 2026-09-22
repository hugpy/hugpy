"""INV-1..8 conformance suite for the studio §0/§2/§9 spine (roadmap P0-1).

Locks the eight design invariants of ``abstract_hugpy_dev.video_intel.studio`` as
executable checks, and pins the five P0-0 bug-fixes (FIX-1..5) as regressions so
they can never silently un-fix.

House style mirrors ``tests/test_invariants_conformance.py``: a plain python
script with a ``__main__`` guard, run via
``venv/bin/python tests/studio/test_studio_conformance.py``. pytest is NOT
installed in this venv, so env is managed with ``os.environ`` (save/restore) in
the one check that needs it, rather than the ``monkeypatch`` fixture. Every
``test_*`` function is still pytest-collectable (no fixture args) if pytest is
added later. Each check prints a numbered ``[n] PASS`` / ``[n] FAIL`` line; a
final summary reports the counts; the process exits nonzero iff any check FAILED.
The driver catches a failing assert per-check and keeps going so a conformance
run surfaces EVERY divergence, not just the first.

Studio is DORMANT: importing it registers the zoo but does not validate at import
(the dev tree's models are all unpinned; validating at import would raise). This
suite calls ``validate_registry()`` explicitly with ``STUDIO_ALLOW_UNPINNED=1``.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_conformance.py
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import FrozenInstanceError, asdict, replace

# The parent `abstract_hugpy_dev` package logs the whole model registry at INFO
# on import; drop INFO/DEBUG so the [n] PASS/FAIL lines are legible.
logging.disable(logging.INFO)

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_video.intel.studio import (
    AdapterKind,
    AdapterRef,
    Capability,
    CapabilityRequest,
    CapabilityRouter,
    ControlKind,
    ControlRef,
    DeterminismClass,
    Framework,
    LicenseClass,
    MODEL_REGISTRY,
    ModelBinding,
    Precision,
    ProvenanceStub,
    RenderManifest,
    Resolution,
    SamplerConfig,
    SeedBundle,
    StageError,
    Task,
    make_render_manifest,
    render_manifest_from_dict,
    render_manifest_to_dict,
    validate_registry,
)
from hugpy_video.intel.studio.enums import PRECISION_QUALITY
from hugpy_video.intel.studio.env import StudioEnv
from hugpy_video.intel.studio.errors import RegistryError
from hugpy_video.intel.studio.registry import runner_for
from hugpy_video.intel.studio.router import _pick_precision
# P0-B1: synthetic runner + end-to-end production spine
import numpy as np  # noqa: E402
from hugpy_video.intel.studio.artifacts import Artifact
from hugpy_video.intel.studio.produce import _DISPATCH, produce_clip
from hugpy_video.intel.studio.runners.synthetic import run_synthetic_i2v, synthesize_frame
# P0-6: real Wan i2v runner — import-safe (torch/diffusers lazy) + graceful-degrading
from hugpy_video.intel.studio.errors import ErrorCode
from hugpy_video.intel.studio.runners.wan_i2v import run_wan_i2v

R720 = Resolution(1280, 720, 24)
R720V = Resolution(720, 1280, 24)   # portrait 720p — a real Wan i2v res the tiny
                                    # synthetic model does NOT offer, so at a
                                    # 16GB budget the router binds Wan, not synthetic
R480 = Resolution(832, 480, 16)
R1080 = Resolution(1920, 1080, 24)


def _manifest(
    *,
    precision: Precision = Precision.BF16,
    determinism: DeterminismClass = DeterminismClass.SEEDED_APPROX,
    env_snapshot: tuple[tuple[str, str], ...] = (),
    render_id: str = "r1",
    seed: int = 1234,
) -> RenderManifest:
    """A minimal, valid RenderManifest. Everything a builder would thread from a
    ModelBinding (precision, determinism_class) and the resolved env
    (env_snapshot) is a parameter here so regressions can vary exactly one axis."""
    return RenderManifest(
        render_id=render_id,
        capability=Capability.I2V,
        model_id="wan2.1-i2v-14b-720p",
        weight_hash=None,
        framework=Framework.WAN,
        task=Task.I2V,
        precision=precision,
        seeds=SeedBundle(global_seed=seed, stage_seeds=(("base", seed),)),
        sampler=SamplerConfig(sampler="euler", scheduler="normal", steps=30, cfg=6.0),
        resolution_ladder=(R720,),
        determinism_class=determinism,
        env_snapshot=env_snapshot,
    )


def _studio_env(master_fps: int) -> StudioEnv:
    return StudioEnv(
        output_root="/out",
        weights_root="/weights",
        manifest_root="/manifests",
        master_colorspace="rec709",
        master_fps=master_fps,
        max_vram_gb=24.0,
        loudness_target_lufs=-14.0,
        allow_unpinned=True,
    )


# --------------------------------------------------------------------------- #
# INV-1 — frozen manifest + reproducible content hash
# --------------------------------------------------------------------------- #
def test_inv1_frozen_and_roundtrip():
    m1 = _manifest()

    # frozen: mutation raises
    raised = False
    try:
        m1.render_id = "mutated"  # type: ignore[misc]
    except FrozenInstanceError:
        raised = True
    assert raised, "RenderManifest must be frozen (mutation should raise)"

    # round-trip: build -> asdict -> reconstruct -> equal content_hash
    d = asdict(m1)
    m2 = RenderManifest(
        render_id=d["render_id"],
        capability=d["capability"],
        model_id=d["model_id"],
        weight_hash=d["weight_hash"],
        framework=d["framework"],
        task=d["task"],
        precision=d["precision"],
        seeds=SeedBundle(**d["seeds"]),
        sampler=SamplerConfig(**d["sampler"]),
        resolution_ladder=tuple(Resolution(**r) for r in d["resolution_ladder"]),
        determinism_class=d["determinism_class"],
        env_snapshot=tuple(tuple(p) for p in d["env_snapshot"]),
    )
    assert m1.content_hash() == m2.content_hash(), "asdict round-trip must preserve hash"

    # render_id is metadata: excluded from the hash
    assert _manifest(render_id="rX").content_hash() == m1.content_hash(), (
        "content_hash must exclude render_id (metadata)"
    )


# --------------------------------------------------------------------------- #
# INV-2 — seeds captured; determinism_class reflects the bound model (FIX-3)
# --------------------------------------------------------------------------- #
def test_inv2_seeds_and_determinism_from_binding():
    m = _manifest(seed=7777)
    assert m.canonical_inputs()["seeds"]["global"] == 7777, "seeds must be captured"

    # Resolve an EXACT model (rife) and thread its determinism into a manifest.
    router = CapabilityRouter()
    res = router.resolve(CapabilityRequest(
        capability=Capability.INTERP, target_resolution=R720, vram_budget_gb=8.0))
    assert res.is_ok(), "INTERP@720p/8GB should route to rife"
    binding = res.unwrap()
    assert binding.determinism_class == DeterminismClass.EXACT, (
        "FIX-3: binding.determinism_class must come from the model's "
        f"default_determinism (rife=EXACT), got {binding.determinism_class}"
    )
    built = _manifest(determinism=binding.determinism_class)
    assert built.determinism_class == DeterminismClass.EXACT, (
        "FIX-3: manifest built from an EXACT binding must be EXACT, not the "
        "hardcoded SEEDED_APPROX default"
    )
    assert built.determinism_class != DeterminismClass.SEEDED_APPROX


# --------------------------------------------------------------------------- #
# FIX-1 — precision participates in the content hash
# --------------------------------------------------------------------------- #
def test_fix1_precision_in_content_hash():
    h_fp8 = _manifest(precision=Precision.FP8).content_hash()
    h_bf16 = _manifest(precision=Precision.BF16).content_hash()
    assert h_fp8 != h_bf16, (
        "FIX-1: manifests differing only in precision (fp8 vs bf16) must NOT "
        "collide on content_hash"
    )
    # identical intent (same precision) -> same hash
    assert (_manifest(precision=Precision.FP8).content_hash()
            == _manifest(precision=Precision.FP8).content_hash()), (
        "identical intent must hash equal"
    )


# --------------------------------------------------------------------------- #
# FIX-5 — env_snapshot sourced from resolved env, and it changes the hash
# --------------------------------------------------------------------------- #
def test_fix5_env_snapshot_in_hash():
    snap24 = _studio_env(master_fps=24).to_snapshot()
    snap30 = _studio_env(master_fps=30).to_snapshot()
    assert snap24, "StudioEnv.to_snapshot() must be non-empty when built via the builder"
    assert snap24 != snap30, "different resolved env must produce different snapshots"
    # snapshot is sorted pairs
    assert list(snap24) == sorted(snap24), "to_snapshot() must be sorted"
    h24 = _manifest(env_snapshot=snap24).content_hash()
    h30 = _manifest(env_snapshot=snap30).content_hash()
    assert h24 != h30, (
        "FIX-5: manifests differing only by resolved env_snapshot must differ in hash"
    )


# --------------------------------------------------------------------------- #
# INV-3 — errors are data (router returns Err, never raises)
# --------------------------------------------------------------------------- #
def test_inv3_router_returns_err_not_raise():
    router = CapabilityRouter()
    # 0.5GB budget fits no video model -> unroutable
    res = router.resolve(CapabilityRequest(
        capability=Capability.T2V, target_resolution=R720, vram_budget_gb=0.5))
    assert res.is_err(), "an unroutable request must return Err, not Ok"
    assert isinstance(res.error, StageError), "the Err payload must be a StageError value"


# --------------------------------------------------------------------------- #
# FIX-2 — commercial routing does not nullify commercial_auto
# --------------------------------------------------------------------------- #
def test_fix2_commercial_auto_not_whitelisted_away():
    router = CapabilityRouter()
    # INTERP is served by rife-practical (MIT) and ffmpeg-minterpolate (Apache-2.0).
    # BOTH are commercial_auto=True, and NEITHER license appears in allowed_licenses
    # below — which is the whole point: allowed_licenses names only LTX_COMMERCIAL,
    # so if the whitelist were applied unconditionally every INTERP candidate would
    # be rejected and this would be Err(LICENSE_VIOLATION). That it resolves at all
    # is the fix.
    res = router.resolve(CapabilityRequest(
        capability=Capability.INTERP,
        target_resolution=R720,
        vram_budget_gb=8.0,
        commercial_use=True,
        allowed_licenses=frozenset({LicenseClass.LTX_COMMERCIAL}),
    ))
    assert res.is_ok(), (
        "FIX-2: an Apache/MIT auto-commercial model must still bind under "
        "commercial_use even when allowed_licenses lists only other licenses"
    )
    # ASSERT THE PROPERTY, NOT THE WINNER. This line used to read
    # `== "rife-practical"`, which quietly made a LICENSE test depend on the
    # RANKING order. When rife was correctly demoted on 2026-07-27 (its runner is an
    # unconditional-Err stub — runners/rife_interpolate.py; see
    # tests/studio/test_studio_enhance.py "RANKING, REVERSED"), this test went red
    # despite its own subject being untouched: the is_ok() above still passed.
    # Binding the assertion to commercial_auto instead means a future re-ranking of
    # INTERP cannot make a license test fail for a non-license reason.
    bound = res.unwrap()
    assert MODEL_REGISTRY[bound.model_id].commercial_auto, (
        f"FIX-2: the bound model must be the auto-commercial one that "
        f"allowed_licenses does NOT list; got {bound.model_id} "
        f"(license={MODEL_REGISTRY[bound.model_id].license.value})")
    assert MODEL_REGISTRY[bound.model_id].license not in {LicenseClass.LTX_COMMERCIAL}, (
        "FIX-2 would be vacuous if the bound license were the whitelisted one — "
        "the test only proves anything while the winner is OUTSIDE allowed_licenses")


# --------------------------------------------------------------------------- #
# FIX-4 — precision never bound below the runner's min_precision floor
# --------------------------------------------------------------------------- #
def test_fix4_precision_floor():
    ltx = MODEL_REGISTRY["ltx-video-0.9.7-dev"]  # vram FP16/FP8/INT8; runner floor FP8

    # budget fits only INT8 (below the FP8 floor) -> unfittable (None), NOT INT8
    below = _pick_precision(ltx, 9.0, Precision.FP8)
    assert below is None, (
        "FIX-4: when only a below-floor precision fits the budget, _pick_precision "
        f"must return None (reject), not a below-floor precision; got {below}"
    )
    # ample budget -> a precision AT OR ABOVE the floor
    above = _pick_precision(ltx, 100.0, Precision.FP8)
    assert above is not None
    assert PRECISION_QUALITY[above] >= PRECISION_QUALITY[Precision.FP8], (
        f"FIX-4: picked precision {above} is below the FP8 floor"
    )

    # end-to-end invariant: every Ok binding is at/above its runner's floor
    router = CapabilityRouter()
    sweep = [
        (Capability.I2V, 20.0, R720),
        (Capability.T2V, 20.0, R480),
        (Capability.INTERP, 8.0, R720),
        (Capability.UPRES, 12.0, R1080),
    ]
    bound_any = False
    for cap, budget, res_ in sweep:
        r = router.resolve(CapabilityRequest(
            capability=cap, target_resolution=res_, vram_budget_gb=budget))
        if not r.is_ok():
            continue
        bound_any = True
        b = r.unwrap()
        spec = runner_for(b.framework, b.task)
        assert spec is not None
        assert PRECISION_QUALITY[b.precision] >= PRECISION_QUALITY[spec.min_precision], (
            f"{b.model_id}: bound precision {b.precision.value} below runner floor "
            f"{spec.min_precision.value}"
        )
    assert bound_any, "sweep should bind at least one model"


# --------------------------------------------------------------------------- #
# INV-8 — validate_registry passes under the unpinned gate; totality holds
# --------------------------------------------------------------------------- #
def test_inv8_validate_registry():
    saved = os.environ.get("STUDIO_ALLOW_UNPINNED")
    try:
        # Without the gate, the fail-loud contract holds: unpinned weights raise.
        os.environ.pop("STUDIO_ALLOW_UNPINNED", None)
        raised = False
        try:
            validate_registry()
        except RegistryError:
            raised = True
        assert raised, (
            "INV-8: validate_registry() must raise on unpinned weights when "
            "STUDIO_ALLOW_UNPINNED is unset (fail-loud pinning contract)"
        )
        # With the gate, the whole join (capability->task->runner, precision floors,
        # every capability served-or-PLANNED) must validate without raising.
        os.environ["STUDIO_ALLOW_UNPINNED"] = "1"
        validate_registry()  # raises on any incoherence
    finally:
        if saved is None:
            os.environ.pop("STUDIO_ALLOW_UNPINNED", None)
        else:
            os.environ["STUDIO_ALLOW_UNPINNED"] = saved


# --------------------------------------------------------------------------- #
# P0-3a — make_render_manifest ENFORCES the fixes at build time, not as defaults
# --------------------------------------------------------------------------- #
def _binding(
    *, precision=None, determinism=None, capability=Capability.I2V,
    budget=20.0, res=R720,
):
    """A REAL ModelBinding from the router, optionally overriding exactly one axis
    via dataclasses.replace (keeps it a genuine binding, not a hand-built stand-in
    the factory couldn't distinguish)."""
    router = CapabilityRouter()
    r = router.resolve(CapabilityRequest(
        capability=capability, target_resolution=res, vram_budget_gb=budget))
    assert r.is_ok(), f"fixture binding for {capability.value} must resolve"
    b = r.unwrap()
    assert isinstance(b, ModelBinding)
    if precision is not None:
        b = replace(b, precision=precision)
    if determinism is not None:
        b = replace(b, determinism_class=determinism)
    return b


def _factory_manifest(
    binding, *, env=None, render_id="rf", capability=Capability.I2V,
    resolution_ladder=(R720,), **kw,
):
    return make_render_manifest(
        render_id=render_id,
        capability=capability,
        binding=binding,
        seeds=SeedBundle(global_seed=99, stage_seeds=(("base", 99),)),
        sampler=SamplerConfig(sampler="euler", scheduler="normal", steps=30, cfg=6.0),
        resolution_ladder=resolution_ladder,
        env=env if env is not None else _studio_env(24),
        **kw,
    )


def test_p03a_factory_threads_precision_fp8_ne_bf16():
    b_bf16 = _binding(precision=Precision.BF16)
    b_fp8 = _binding(precision=Precision.FP8)
    m_bf16 = _factory_manifest(b_bf16)
    m_fp8 = _factory_manifest(b_fp8)
    # precision landed on the manifest THREADED FROM THE BINDING, not hand-passed
    assert m_bf16.precision == Precision.BF16, "factory must thread binding.precision"
    assert m_fp8.precision == Precision.FP8, "factory must thread binding.precision"
    # FIX-1 ENFORCED at build time: fp8 vs bf16 binding -> different content_hash
    assert m_bf16.content_hash() != m_fp8.content_hash(), (
        "FIX-1: a manifest built from an fp8 binding must not collide with one "
        "built from a bf16 binding (precision is threaded into the hash)")


def test_p03a_factory_threads_determinism_from_binding():
    # rife (INTERP) is an EXACT model; its binding.determinism_class must land.
    b = _binding(capability=Capability.INTERP, budget=8.0)
    assert b.determinism_class == DeterminismClass.EXACT, (
        "fixture: INTERP@720p/8GB must route to an EXACT model (rife)")
    m = _factory_manifest(b, capability=Capability.INTERP)
    assert m.determinism_class == DeterminismClass.EXACT, (
        "FIX-3: factory must thread binding.determinism_class (EXACT), NOT the "
        "SEEDED_APPROX dataclass default")
    assert m.determinism_class != DeterminismClass.SEEDED_APPROX


def test_p03a_factory_populates_env_snapshot():
    b = _binding()
    m24 = _factory_manifest(b, env=_studio_env(24))
    m30 = _factory_manifest(b, env=_studio_env(30))
    assert m24.env_snapshot, (
        "factory must populate env_snapshot from env.to_snapshot() (non-empty)")
    assert m24.env_snapshot == _studio_env(24).to_snapshot(), (
        "FIX-5: env_snapshot must be sourced verbatim from env.to_snapshot()")
    # FIX-5 ENFORCED at build time: two different resolved envs -> different hash
    assert m24.content_hash() != m30.content_hash(), (
        "manifests differing only by resolved env_snapshot must differ in hash")


def test_p03a_roundtrip_from_dict_preserves_hash():
    b = _binding()
    m = _factory_manifest(
        b,
        controls=(ControlRef(kind=ControlKind.DEPTH, content_hash="deadbeef",
                             weight=0.8, target_frames=(0, 4)),),
        adapters=(AdapterRef(kind=AdapterKind.IDENTITY_LORA, adapter_id="ada-1",
                            weight=0.7, weight_hash="cafef00d"),),
        identity_ids=("char-1",),
        identity_view_hashes=("view-a",),
        provenance=ProvenanceStub(operator="op", created_at="2026-07-07T00:00:00Z"),
    )
    m2 = render_manifest_from_dict(render_manifest_to_dict(m))
    assert m2.content_hash() == m.content_hash(), (
        "round-trip from_dict(to_dict(m)) must preserve content_hash for a "
        "factory-built manifest (nested value objects rebuilt + re-validated)")


def test_p03a_validate_at_construction_raises():
    b = _binding()
    # empty resolution_ladder -> raise (this is the enforcement point)
    raised_res = False
    try:
        _factory_manifest(b, resolution_ladder=())
    except ValueError:
        raised_res = True
    assert raised_res, "empty resolution_ladder must raise ValueError at construction"
    # empty render_id -> raise
    raised_id = False
    try:
        _factory_manifest(b, render_id="")
    except ValueError:
        raised_id = True
    assert raised_id, "empty render_id must raise ValueError at construction"


# --------------------------------------------------------------------------- #
# P0-B1 — SYNTHETIC runner + end-to-end production spine
# --------------------------------------------------------------------------- #
_FFMPEG = shutil.which("ffmpeg") is not None
_FFPROBE = shutil.which("ffprobe")


def _b1_env() -> StudioEnv:
    return _studio_env(24)


def _b1_request(*, res=Resolution(320, 180, 12), budget=0.5):
    """CAP-I2V at a VRAM budget too small for any real model (min real i2v = 8GB),
    so the router deterministically binds the tiny synthetic-i2v model."""
    return CapabilityRequest(
        capability=Capability.I2V, target_resolution=res, vram_budget_gb=budget)


def _ffprobe_nb_frames(path: str) -> int:
    out = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    return int((out.stdout or "0").strip() or "0")


def test_b1_pixel_determinism():
    # Pure frame synthesis: same (seed, geometry, index) => byte-identical frame,
    # different seed => different frame. No ffmpeg needed.
    a = synthesize_frame(1234, 160, 90, 5, 24, None)
    b = synthesize_frame(1234, 160, 90, 5, 24, None)
    c = synthesize_frame(9999, 160, 90, 5, 24, None)
    assert a.dtype == np.uint8 and a.shape == (90, 160, 3), "frame must be HxWx3 uint8"
    assert np.array_equal(a, b), "same seed+geometry+index must be byte-identical"
    assert not np.array_equal(a, c), "different seed must produce a different frame"


def test_b1_manifest_dir_addressing():
    # Same request => same manifest content_hash => same output directory (INV-6).
    env = _b1_env()
    req = _b1_request()
    router = CapabilityRouter()
    r = router.resolve(req)
    assert r.is_ok() and r.unwrap().model_id == "synthetic-i2v", (
        "tiny-budget CAP-I2V must bind synthetic-i2v")

    def _mk():
        return make_render_manifest(
            render_id="rid",  # metadata, excluded from the hash
            capability=req.capability, binding=r.unwrap(),
            seeds=SeedBundle(global_seed=7, stage_seeds=(("base", 7),)),
            sampler=SamplerConfig(sampler="euler", scheduler="normal", steps=1, cfg=1.0),
            resolution_ladder=(req.target_resolution,), env=env)

    assert _mk().content_hash() == _mk().content_hash(), (
        "two identical manifests must share one content_hash (dir address)")


def test_b1_runner_yields_valid_mp4():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping mp4 assembly check)")
        return
    env = _b1_env()
    req = _b1_request()
    binding = CapabilityRouter().resolve(req).unwrap()
    manifest = make_render_manifest(
        render_id="rid", capability=req.capability, binding=binding,
        seeds=SeedBundle(global_seed=7, stage_seeds=(("base", 7),)),
        sampler=SamplerConfig(sampler="euler", scheduler="normal", steps=1, cfg=1.0),
        resolution_ladder=(req.target_resolution,), env=env)
    out_root = tempfile.mkdtemp(prefix="studio-b1-mp4-")
    try:
        res = run_synthetic_i2v(manifest, out_root)
        assert res.is_ok(), f"runner must return Ok; got {res}"
        art = res.unwrap()
        assert isinstance(art, Artifact)
        assert os.path.isfile(art.path) and os.path.getsize(art.path) > 0, "mp4 non-empty"
        assert art.content_hash == manifest.content_hash(), "artifact hash == manifest hash"
        assert os.path.basename(os.path.dirname(art.path)) == manifest.content_hash(), (
            "clip must live under <out_root>/<content_hash>/ (INV-6 addressing)")
        nb = _ffprobe_nb_frames(art.path)
        assert nb > 0, f"ffprobe nb_frames must be > 0; got {nb}"
        assert nb == art.frames, f"ffprobe frames {nb} != artifact.frames {art.frames}"
        # sidecars present
        d = os.path.dirname(art.path)
        assert os.path.isfile(os.path.join(d, "manifest.json")), "manifest.json sidecar"
        assert os.path.isfile(os.path.join(d, "provenance.json")), "provenance.json sidecar"
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


def test_b1_resume_skip_no_regen():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping resume check)")
        return
    env = _b1_env()
    req = _b1_request()
    out_root = tempfile.mkdtemp(prefix="studio-b1-resume-")
    try:
        r1 = produce_clip(req, env=env, out_root=out_root)
        assert r1.is_ok(), f"first produce must succeed; got {r1}"
        a1 = r1.unwrap()
        assert a1.resumed is False, "first render must NOT be a resume"
        mtime1 = os.path.getmtime(a1.path)
        time.sleep(1.1)  # coarse mtime resolution guard
        r2 = produce_clip(req, env=env, out_root=out_root)
        assert r2.is_ok()
        a2 = r2.unwrap()
        assert a2.resumed is True, "second identical render must resume (skip regen)"
        assert a2.path == a1.path and a2.content_hash == a1.content_hash, "same output"
        assert os.path.getmtime(a2.path) == mtime1, "clip must NOT be rewritten on resume"
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


def test_b1_produce_end_to_end_ok():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping produce_clip e2e check)")
        return
    env = _b1_env()
    out_root = tempfile.mkdtemp(prefix="studio-b1-e2e-")
    try:
        res = produce_clip(_b1_request(), env=env, out_root=out_root)
        assert res.is_ok(), f"produce_clip must return Ok(Artifact); got {res}"
        art = res.unwrap()
        assert isinstance(art, Artifact), "Ok payload must be an Artifact"
        assert art.frames > 0 and art.width == 320 and art.height == 180
        assert os.path.isfile(art.path) and os.path.getsize(art.path) > 0
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


def test_b1_errors_as_data():
    env = _b1_env()
    # (a) unroutable: no model (incl. synthetic) covers a 4096x4096 target.
    ra = produce_clip(
        CapabilityRequest(capability=Capability.I2V,
                          target_resolution=Resolution(4096, 4096, 24),
                          vram_budget_gb=0.5),
        env=env, out_root=tempfile.gettempdir())
    assert ra.is_err(), "unroutable request must return Err, not raise"
    assert isinstance(ra.error, StageError), "Err payload must be a StageError value"
    # (b) unwritable out_root: runner fails as data (io_error), never raises.
    rb = produce_clip(_b1_request(), env=env,
                      out_root="/proc/nonexistent-unwritable/out")
    assert rb.is_err(), "unwritable out_root must return Err, not raise"
    assert isinstance(rb.error, StageError)


# --------------------------------------------------------------------------- #
# P0-6 — real Wan i2v runner: registered for (WAN, I2V), and import-safe +
# errors-as-data (graceful degradation) on this GPU-less / weight-less box.
# --------------------------------------------------------------------------- #
def test_p06_wan_runner_registered_for_wan_i2v():
    # (a) wired in produce's dispatch table for (Framework.WAN, Task.I2V)
    assert _DISPATCH.get((Framework.WAN, Task.I2V)) is run_wan_i2v, (
        "produce._DISPATCH must map (WAN, I2V) -> run_wan_i2v")
    # (b) the RunnerSpec entrypoint resolves to the real runner module (not the
    #     dormant `runners.wan:i2v` placeholder)
    spec = runner_for(Framework.WAN, Task.I2V)
    assert spec is not None, "a runner must be registered for (WAN, I2V)"
    assert spec.entrypoint == (
        "hugpy_video.intel.studio.runners.wan_i2v:run_wan_i2v"), (
        f"WAN i2v entrypoint must resolve to wan_i2v:run_wan_i2v; got {spec.entrypoint}")


def test_p06_wan_runner_errors_as_data_on_this_box():
    # Router-bind a real Wan i2v model at a Wan-fitting budget, build its manifest
    # via the factory, then call the runner. On this GPU-less / weight-less box it
    # must return Err(StageError) with a graceful preflight code — NOT raise.
    env = _studio_env(24)
    r = CapabilityRouter().resolve(CapabilityRequest(
        capability=Capability.I2V, target_resolution=R720V, vram_budget_gb=16.0))
    assert r.is_ok(), "CAP-I2V @ portrait-720p / 16GB must route to a real Wan i2v model"
    binding = r.unwrap()
    assert binding.framework == Framework.WAN, (
        f"16GB portrait-i2v budget should bind a Wan model; got {binding.framework}")
    manifest = make_render_manifest(
        render_id="wan-p06", capability=Capability.I2V, binding=binding,
        seeds=SeedBundle(global_seed=7, stage_seeds=(("base", 7),)),
        sampler=SamplerConfig(sampler="unipc", scheduler="normal", steps=30, cfg=5.0),
        resolution_ladder=(R720V,), env=env)
    res = run_wan_i2v(manifest, tempfile.gettempdir())
    assert res.is_err(), "Wan i2v on a GPU-less/weight-less box must return Err, not Ok"
    assert isinstance(res.error, StageError), "Err payload must be a StageError value"
    assert res.error.code in (
        ErrorCode.DEPS_MISSING, ErrorCode.NO_GPU, ErrorCode.WEIGHTS_MISSING), (
        f"expected a graceful preflight code; got {res.error.code}")


# --------------------------------------------------------------------------- #
# SLICE-1 — synthetic is LAST-RESORT: a real model, whenever one fits, ALWAYS
# outranks the placeholder synthetic model (it must never shadow a real binding).
# --------------------------------------------------------------------------- #
def test_slice1_synthetic_never_shadows_real_i2v():
    # LANDSCAPE 720p @16GB is the exact shadow combo: BOTH the real Wan i2v
    # (INT8=14GB) and the tiny synthetic (0.1GB) FIT and BOTH offer 1280x720, so
    # every score dimension tied and synthetic used to win the -min_gb tie-break.
    # The last-resort rule must now bind the REAL model.
    r = CapabilityRouter().resolve(CapabilityRequest(
        capability=Capability.I2V, target_resolution=R720, vram_budget_gb=16.0))
    assert r.is_ok(), "CAP-I2V @ landscape-720p / 16GB must route"
    b = r.unwrap()
    # precondition: synthetic genuinely FITS this request (covers the res + its
    # 0.1GB envelope is under budget) — so a real win here is an active tie-break
    # over synthetic, not merely synthetic falling out on fit.
    syn = MODEL_REGISTRY["synthetic-i2v"]
    assert syn.supports_resolution(R720) and syn.vram.min_gb() <= 16.0, (
        "precondition: synthetic must itself fit this request for the shadow test "
        "to be meaningful")
    assert b.framework != Framework.SYNTHETIC, (
        f"synthetic must NOT shadow a real i2v model at 16GB landscape; got {b.model_id}")
    assert b.model_id == "wan2.1-i2v-14b-720p", (
        f"a real Wan i2v model must win the 16GB landscape tie-break; got {b.model_id}")


def test_slice1_synthetic_reachable_as_last_resort():
    # No real model fits a 0.5GB budget (min real i2v footprint = 8GB), so synthetic
    # is the SOLE survivor and must still bind — the last-resort rule preserves the
    # tiny demo path (B2's default budget) untouched — and still produces a real mp4.
    env = _b1_env()
    req = _b1_request()  # CAP-I2V, 320x180, 0.5GB
    r = CapabilityRouter().resolve(req)
    assert r.is_ok() and r.unwrap().model_id == "synthetic-i2v", (
        f"tiny-budget CAP-I2V must still bind synthetic-i2v as last resort; got {r}")
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping clip assembly check)")
        return
    out_root = tempfile.mkdtemp(prefix="studio-slice1-syn-")
    try:
        res = produce_clip(req, env=env, out_root=out_root)
        assert res.is_ok(), f"synthetic produce_clip must yield Ok(Artifact); got {res}"
        art = res.unwrap()
        assert os.path.isfile(art.path) and os.path.getsize(art.path) > 0, "mp4 non-empty"
        nb = _ffprobe_nb_frames(art.path)
        assert nb > 0, f"ffprobe nb_frames must be > 0 (a real playable clip); got {nb}"
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


def test_slice1_real_i2v_path_intact_at_16gb():
    # The very request that used to be shadowed (landscape-720p / 16GB) now routes
    # THROUGH the real Wan runner end-to-end; on this GPU-less / weight-less box it
    # degrades gracefully to an Err (deps_missing here) instead of raising or
    # silently handing back a synthetic placeholder.
    env = _studio_env(24)
    res = produce_clip(
        CapabilityRequest(capability=Capability.I2V, target_resolution=R720,
                          vram_budget_gb=16.0),
        env=env, out_root=tempfile.gettempdir())
    assert res.is_err(), "real Wan i2v on a GPU-less/weight-less box must return Err, not Ok"
    assert isinstance(res.error, StageError), "Err payload must be a StageError value"
    assert res.error.code in (
        ErrorCode.DEPS_MISSING, ErrorCode.NO_GPU, ErrorCode.WEIGHTS_MISSING), (
        f"expected a graceful preflight code (deps_missing on this box); got {res.error.code}")


CHECKS = [
    ("INV-1 frozen manifest + reproducible hash", test_inv1_frozen_and_roundtrip),
    ("INV-2 seeds + determinism_class from binding (FIX-3)", test_inv2_seeds_and_determinism_from_binding),
    ("FIX-1 precision in content_hash", test_fix1_precision_in_content_hash),
    ("FIX-5 env_snapshot in content_hash", test_fix5_env_snapshot_in_hash),
    ("INV-3 router returns Err, not raise", test_inv3_router_returns_err_not_raise),
    ("FIX-2 commercial_auto not whitelisted away", test_fix2_commercial_auto_not_whitelisted_away),
    ("FIX-4 precision floor honored", test_fix4_precision_floor),
    ("INV-8 validate_registry + join totality", test_inv8_validate_registry),
    ("P0-3a factory threads precision (fp8 != bf16 hash)", test_p03a_factory_threads_precision_fp8_ne_bf16),
    ("P0-3a factory threads determinism_class from binding", test_p03a_factory_threads_determinism_from_binding),
    ("P0-3a factory populates env_snapshot from env", test_p03a_factory_populates_env_snapshot),
    ("P0-3a round-trip from_dict(to_dict(m)) preserves hash", test_p03a_roundtrip_from_dict_preserves_hash),
    ("P0-3a validate-at-construction raises", test_p03a_validate_at_construction_raises),
    ("P0-B1 synthetic frame pixel-determinism", test_b1_pixel_determinism),
    ("P0-B1 manifest content-hash addresses output dir", test_b1_manifest_dir_addressing),
    ("P0-B1 synthetic runner yields valid mp4 (nb_frames>0)", test_b1_runner_yields_valid_mp4),
    ("P0-B1 resume-skip returns existing without regen", test_b1_resume_skip_no_regen),
    ("P0-B1 produce_clip end-to-end returns Ok(Artifact)", test_b1_produce_end_to_end_ok),
    ("P0-B1 errors-as-data on forced failure", test_b1_errors_as_data),
    ("P0-6 wan runner registered for (WAN, I2V)", test_p06_wan_runner_registered_for_wan_i2v),
    ("P0-6 wan runner errors-as-data on this box", test_p06_wan_runner_errors_as_data_on_this_box),
    ("SLICE-1 synthetic never shadows a real i2v model (16GB landscape)", test_slice1_synthetic_never_shadows_real_i2v),
    ("SLICE-1 synthetic still reachable as last resort + yields mp4", test_slice1_synthetic_reachable_as_last_resort),
    ("SLICE-1 real i2v path intact @16GB (graceful Err on this box)", test_slice1_real_i2v_path_intact_at_16gb),
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
