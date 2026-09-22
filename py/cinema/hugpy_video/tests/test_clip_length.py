"""CLIP LENGTH is a REACHABLE lever — conformance for the whole thread.

Same script style as the other studio tests (plain python, ``__main__`` guard,
numbered ``[n] PASS`` / ``[n] FAIL`` lines, nonzero exit iff any check FAILED,
every check independently run so a run surfaces EVERY divergence). Also collected
by pytest, which is how it gates a release.

WHY THIS EXISTS. Three separate defects, all found by adversarial review on
2026-07-27, all in the same lever:

  1. UNREACHABLE. ``studio/runners/synthetic._geometry`` used to compute
     ``n = fps * 2  # a short ~2s clip`` — a placeholder written for the no-model
     NOISE PROVER. ``wan_i2v._wan_geometry`` imports that very function and the VACE
     runner goes through ``_wan_geometry`` too, so a prover's placeholder silently
     governed EVERY REAL Wan render on this fleet: ~29 frames (32 snapped to 4k+1)
     = ~1.8s, for that reason and no other. The first fix added
     ``RenderManifest.requested_frames`` — but ``_build_manifest`` /
     ``make_render_manifest`` (the ONE live build path) took no such kwarg,
     ``StudioI2VSpec`` had no length field and ``produce_clip`` passed none, so the
     lever the docstrings advertised ("a caller who wants a cheap preview now has a
     lever") could be exercised ONLY by hand-constructing a manifest in a test.
     Every real render stayed forced to the default.
  2. THE WIRE LIED. ``synthetic`` defaulted a real model to 81 frames while
     ``presets.WAN_DEFAULT_FRAMES`` published 29 and ``GET /video/render/presets``
     served 29. Callers were told 29 and charged for 81 — ~2.8x the latent tokens
     and the GPU minutes. Two literals, no import between them.
  3. THE SIDECAR DROPPED IT. ``requested_frames`` was a ``canonical_inputs`` (i.e.
     content-hash) input, but ``render_manifest_to_dict`` did not serialize it and
     ``render_manifest_from_dict`` did not restore it — harmless only while the
     field was permanently None. The moment (1) was fixed, ``from_dict(to_dict(m))``
     would re-address to a DIFFERENT content hash than ``m`` for any non-default
     length, breaking resume/dedup: a resumed 33-frame render would look like a
     cache miss and re-burn the GPU, and the clip's ``manifest.json`` would disagree
     with the directory it sits in. That round-trip check is the highest-value
     assertion in this file.

What is under test:
  * DEFAULT for a real Wan model = 81 frames — MEASURED (CAPABILITY-VIABILITY-MAP.md,
    ae's 3090, 2026-07-27: wan2.1-t2v-1.3b @ 832x480 x 81 frames, ~352s wall-clock),
    not the fps*2 stub. Operator doctrine: "defaults are promises".
  * DEFAULT for the SYNTHETIC prover stays the cheap historical ``fps * 2`` — a
    spine-prover is a test cost, not a product surface.
  * ONE LITERAL: the renderer's default and the wire's published default are the
    same constant, imported, not restated.
  * REACHABILITY: spec -> manifest factory -> manifest -> runner, end to end.
  * An EXPLICIT request is honoured verbatim when it is in range + on cadence.
  * An out-of-range request CLAMPS with a recorded reason (never an error, never a 500).
  * Off-cadence requests SNAP down to Wan's 4k+1 (50 -> 49); the prover does NOT snap
    (it has no temporal VAE).
  * The ``max(1, n)`` floor survives.
  * ``cfg.max_frames`` still wins when it is BELOW the default (cogvideox-5b: 49).
  * The real render path picks the lever up for free: ``_wan_geometry`` is idempotent
    over the snap ``_geometry`` already did.
  * INV-6: two requests differing ONLY in length do not collide on one
    content-addressed path, AND survive the sidecar round trip byte-for-byte.

ARITHMETIC THIS FILE ENCODES. Wan requires ``num_frames == 4k+1``, which is always
ODD, so ``duration = frames / fps`` can NEVER be an exact whole number of seconds at
an EVEN fps (81/16 = 5.0625s, not 5s). FRAMES is therefore the exact unit; a duration
in seconds is a REQUEST that must be resolved to on-cadence frames with the TRUE
resulting duration reported back, never the requested one echoed. That is why the
lever is spelled in frames at every seam.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_clip_length.py
or:
  cd /srv/share/projects/hugpy/dev
  abstract_hugpy_dev/venv/bin/python -m pytest \\
      abstract_hugpy_dev/tests/studio/test_clip_length.py -q -p no:randomly
"""
from __future__ import annotations

import inspect
import os
import sys
from dataclasses import asdict, replace

# NOTE: this module deliberately does NOT call ``logging.disable`` and does NOT set
# STUDIO_* env vars at import (both of which the first cut of this file did). Those
# are PROCESS-GLOBAL mutations that leak into every other test sharing the
# interpreter — the k51 "132 fail in-sweep, pass standalone" class. Nothing here
# needs them: the studio package does not validate the registry at import, and INFO
# records go nowhere with no root handler configured.

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_video.intel.studio import (
    MODEL_REGISTRY,
    Capability,
    DeterminismClass,
    Framework,
    ModelBinding,
    PathClass,
    Precision,
    RenderManifest,
    Resolution,
    SamplerConfig,
    SeedBundle,
    Task,
    make_render_manifest,
    render_manifest_from_dict,
    render_manifest_to_dict,
)
from hugpy_video.intel.studio.env import StudioEnv
from hugpy_video.intel.studio.job import (
    _MAX_REQUESTED_FRAMES,
    StudioI2VSpec,
    make_studio_i2v,
    studio_i2v_from_dict,
)
# The clip-length FACTS live in schemas — the leaf module BOTH the renderer and the
# presets/wire layer can import without dragging numpy/PIL into app boot.
from hugpy_video.intel.studio.schemas import (
    DEFAULT_FRAMES_REAL,
    WAN_FRAME_CADENCE,
    WAN_MAX_FRAMES,
    snap_wan_frames,
)
from hugpy_video.intel.studio.runners import synthetic as synth
from hugpy_video.intel.studio.runners.synthetic import (
    _DEFAULT_SYNTHETIC_FPS_MULT,
    _geometry,
    resolve_frames,
)
# The REAL render path: _wan_geometry wraps _geometry with the 4k+1 snap. Importing
# it is safe (torch/diffusers are lazy inside run_wan_i2v).
from hugpy_video.intel.studio.runners.wan_i2v import _wan_geometry

R480 = Resolution(832, 480, 16)      # the Wan 480p rung: 16fps, so 81 frames = 5.0625s
R_PROVER = Resolution(320, 180, 12)  # the prover's cheap rung: fps*2 = 24 frames
R_COG = Resolution(720, 480, 8)      # cogvideox-5b's only rung (max_frames=49)


def _manifest(
    *,
    model_id: str = "wan2.1-t2v-1.3b",
    framework: Framework = Framework.WAN,
    task: Task = Task.T2V,
    capability: Capability = Capability.T2V,
    ladder: Resolution = R480,
    requested_frames: int | None = None,
) -> RenderManifest:
    """A minimal, valid RenderManifest, HAND-constructed on purpose: the clamp/floor
    checks below need lengths the validating factory rightly refuses (0, -999), and a
    hand-forged or hand-edited manifest is exactly the case ``resolve_frames``'s
    belt-and-braces guards exist for. The reachability checks use ``_factory_manifest``
    instead — the ONE live build path."""
    return RenderManifest(
        render_id="r-clip-len",
        capability=capability,
        model_id=model_id,
        weight_hash=None,
        framework=framework,
        task=task,
        precision=Precision.BF16,
        seeds=SeedBundle(global_seed=7, stage_seeds=(("base", 7),)),
        sampler=SamplerConfig(sampler="unipc", scheduler="flow_match", steps=32, cfg=5.0),
        resolution_ladder=(ladder,),
        determinism_class=DeterminismClass.SEEDED_APPROX,
        requested_frames=requested_frames,
    )


def _prover_manifest(**kw) -> RenderManifest:
    return _manifest(
        model_id="synthetic-i2v", framework=Framework.SYNTHETIC,
        task=Task.I2V, capability=Capability.I2V, ladder=R_PROVER, **kw)


def _binding() -> ModelBinding:
    """The router's answer, which ``make_render_manifest`` demands (it threads
    model_id/precision/determinism from here, never from a free parameter)."""
    return ModelBinding(
        model_id="wan2.1-t2v-1.3b", framework=Framework.WAN, task=Task.T2V,
        precision=Precision.BF16, path_class=PathClass.OFFLINE,
        weight_uri="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", weight_hash=None,
        determinism_class=DeterminismClass.SEEDED_APPROX)


def _env() -> StudioEnv:
    """A resolved env, which ``make_render_manifest`` also demands. The paths are
    inert here (nothing is written) — only ``to_snapshot()`` is consumed."""
    return StudioEnv(
        output_root="/tmp/studio-clip-len/out",
        weights_root="/tmp/studio-clip-len/weights",
        manifest_root="/tmp/studio-clip-len/manifests",
        master_colorspace="rec709", master_fps=16, max_vram_gb=24.0,
        loudness_target_lufs=-14.0, allow_unpinned=True)


def _factory_manifest(requested_frames: int | None = None) -> RenderManifest:
    """A manifest built through the ONE LIVE BUILD PATH, not hand-constructed — the
    distinction that defect 1 was entirely about."""
    return make_render_manifest(
        render_id="r-factory",
        capability=Capability.T2V,
        binding=_binding(),
        seeds=SeedBundle(global_seed=7, stage_seeds=(("base", 7),)),
        sampler=SamplerConfig(sampler="unipc", scheduler="flow_match", steps=32,
                              cfg=5.0, shift=3.0),
        resolution_ladder=(R480,),
        env=_env(),
        prompt="a slow dolly across a quiet room",
        requested_frames=requested_frames,
    )


# --------------------------------------------------------------------------- #
# Defaults — the promise the fleet can keep
# --------------------------------------------------------------------------- #
def test_default_real_wan_model_is_the_measured_ceiling():
    """No request -> 81 frames for a REAL model, NOT the fps*2 stub (which would be
    32 -> snapped 29 = ~1.8s). 81 is Wan's reference length and is MEASURED to work
    at 832x480 on wan2.1-t2v-1.3b (~352s on ae's 3090, 2026-07-27)."""
    n, reason = resolve_frames(_manifest())
    assert n == 81, f"real-model default must be 81 frames; got {n} ({reason})"
    assert n == DEFAULT_FRAMES_REAL, "the default constant IS the resolved default"
    # the stub it replaced, spelled out so a regression to it is unmistakable
    assert n != R480.fps * 2, "fps*2 is the prover's placeholder, not a real default"
    # 81 @ 16fps = 5.0625s EXACTLY — and pointedly not 5.0s, which is the whole reason
    # frames (not seconds) is the unit: 4k+1 is odd, so an even fps never divides whole.
    assert n / R480.fps == 5.0625, f"81 @ {R480.fps}fps must be exactly 5.0625s"
    assert n % 2 == 1, "4k+1 is always ODD — duration is never a whole second count"


def test_default_synthetic_prover_stays_cheap():
    """The prover is a SPINE PROVER, not a product surface: it must NOT default to
    81 frames of plasma. Its default stays the historical fps*2 (~2s), unsnapped."""
    n, reason = resolve_frames(_prover_manifest())
    expected = R_PROVER.fps * _DEFAULT_SYNTHETIC_FPS_MULT
    assert n == expected, f"prover default must be fps*2 = {expected}; got {n} ({reason})"
    assert n != DEFAULT_FRAMES_REAL, "the prover must not inherit the real default"
    # no 4k+1 snap for the prover (24 would become 21) — it has no temporal VAE
    assert n == 24, f"prover default must NOT be snapped to 4k+1; got {n}"


def test_defaults_differ_by_runner_class():
    """The whole point of the split: same absent request, different answer, because
    one path burns a 3090 for ~6min and the other burns numpy for ~2s."""
    real, _ = resolve_frames(_manifest())
    prover, _ = resolve_frames(_prover_manifest())
    assert real > prover, f"real default {real} must exceed prover default {prover}"


# --------------------------------------------------------------------------- #
# ONE LITERAL — the wire and the renderer cannot drift (defect 2)
# --------------------------------------------------------------------------- #
def test_the_runner_restates_no_length_literal():
    """The renderer must READ the shared constants, not restate them. Proven by
    IDENTITY: the names the runner resolves ARE the objects ``schemas`` defines, so
    changing schemas changes the render, with no second literal to forget."""
    assert synth.DEFAULT_FRAMES_REAL is DEFAULT_FRAMES_REAL
    assert synth.WAN_MAX_FRAMES is WAN_MAX_FRAMES
    assert synth.WAN_FRAME_CADENCE is WAN_FRAME_CADENCE
    assert synth.snap_wan_frames is snap_wan_frames
    # ...and the shared snap really is the rule the resolver applies.
    assert snap_wan_frames(50) == 49 and snap_wan_frames(0) == 1


def test_the_wire_publishes_the_renderers_number():
    """⚠ CROSS-MODULE SEAM — the actual defect-2 guard, and the one check here whose
    fix lives in a file this workstream does not own (``studio/presets.py``).

    ``GET /video/render/presets`` serves ``default_frames`` / ``max_frames`` built
    from ``studio/presets.py``'s module constants. On 2026-07-27 that module carried
    its OWN literals — ``WAN_DEFAULT_FRAMES = 29`` against the renderer's 81 — so the
    endpoint promised a ~1.8s clip and the GPU produced a ~5.06s one at ~2.8x the
    cost. Keeper ruling: the two must AGREE, the honest default is the model's real
    measured capability (81), and the wire's number must come FROM the renderer's
    constant rather than a second literal.

    This goes green the moment presets.py replaces its three literals with
    ``from .schemas import DEFAULT_FRAMES_REAL, WAN_FRAME_CADENCE, WAN_MAX_FRAMES``.
    Per-ROW overrides stay legitimate (a 720p row may honestly declare fewer frames
    to fit VRAM); what is forbidden is a second spelling of the DEFAULT."""
    from hugpy_video.intel.studio import presets

    assert presets.WAN_MAX_FRAMES == WAN_MAX_FRAMES, (
        f"presets publishes max {presets.WAN_MAX_FRAMES}, renderer clamps at "
        f"{WAN_MAX_FRAMES}")
    assert presets.WAN_FRAME_CADENCE == WAN_FRAME_CADENCE, (
        f"presets snaps on {presets.WAN_FRAME_CADENCE}k+1, renderer on "
        f"{WAN_FRAME_CADENCE}k+1")
    assert presets.WAN_DEFAULT_FRAMES == DEFAULT_FRAMES_REAL, (
        f"the wire publishes {presets.WAN_DEFAULT_FRAMES} frames but the renderer "
        f"produces {DEFAULT_FRAMES_REAL} — callers are quoted "
        f"{DEFAULT_FRAMES_REAL / max(1, presets.WAN_DEFAULT_FRAMES):.1f}x less GPU "
        f"than they are charged")


# --------------------------------------------------------------------------- #
# REACHABILITY — the lever exists on a production path (defect 1)
# --------------------------------------------------------------------------- #
def test_the_spec_carries_a_length():
    """``StudioI2VSpec`` is the bus currency for a studio clip. Before 2026-07-27 it
    had NO length field, so no caller could ask for one even in principle."""
    assert "requested_frames" in StudioI2VSpec.__dataclass_fields__, (
        "StudioI2VSpec must carry the length request")
    spec = make_studio_i2v(width=832, height=480, fps=16, requested_frames=45)
    assert spec.requested_frames == 45
    # None is the unset default — today's behaviour for every existing caller.
    assert make_studio_i2v(width=832, height=480, fps=16).requested_frames is None
    # ...and it survives the bus round trip (asdict -> json -> revalidate).
    assert studio_i2v_from_dict(asdict(spec)).requested_frames == 45
    # A spec enqueued (or relayed to a studio GPU worker) BEFORE this field existed
    # has no key at all; it must rehydrate to "unset" — today's default — not blow up.
    legacy = {k: v for k, v in asdict(spec).items() if k != "requested_frames"}
    assert studio_i2v_from_dict(legacy).requested_frames is None


def test_the_spec_range_guard_is_a_typo_guard_not_a_ceiling():
    """The spec bound rejects what is nonsense at ANY ceiling (zero/negative frames,
    a bool, a non-int, a fat-fingered 100000000) but must NOT reject an over-ceiling
    ask — the model is not bound yet, so "give me the longest you can" is legal and
    CLAMPS at render time. Same shape as the steps/cfg guards beside it."""
    for bad in (0, -1, -999, True, False, 3.5, "45", _MAX_REQUESTED_FRAMES + 1, 10 ** 8):
        try:
            make_studio_i2v(width=832, height=480, fps=16, requested_frames=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"requested_frames={bad!r} must be a clean caller error")
    # 500 frames from a 5-second model is OUT OF RANGE FOR THE MODEL but is NOT a
    # caller error: it clamps to 81 with a reason (see the clamp checks below).
    assert make_studio_i2v(
        width=832, height=480, fps=16, requested_frames=500).requested_frames == 500


def test_the_manifest_factory_carries_a_length():
    """``make_render_manifest`` is described in-file as "the ONE live build path".
    Until 2026-07-27 it accepted no length kwarg, which is precisely why the lever
    was unreachable: the field existed, was hashed, was documented — and only a test
    hand-constructing a RenderManifest could ever set it."""
    assert "requested_frames" in inspect.signature(make_render_manifest).parameters, (
        "the ONE live build path must accept the length request")
    m = _factory_manifest(requested_frames=45)
    assert m.requested_frames == 45, "the factory must CARRY the request, not drop it"
    assert _factory_manifest().requested_frames is None, "None stays unset"
    # and the request the factory carried is the length the runner resolves
    assert resolve_frames(m)[0] == 45


def test_the_manifest_factory_rejects_a_structurally_bad_length():
    """House discipline: a structurally-invalid manifest is programmer error and
    raises LOCALLY (``ValueError``), like every other field in ``_build_manifest``.
    RANGE is not structure — an over-ceiling ask is clamped at render time, never
    raised, so it must pass here."""
    for bad in (0, -7, True, 4.5, "45"):
        try:
            _factory_manifest(requested_frames=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"requested_frames={bad!r} must raise in the factory")
    assert _factory_manifest(requested_frames=5000).requested_frames == 5000, (
        "an over-ceiling ask is runtime policy (clamp), not a construction error")


def test_produce_clip_threads_the_length():
    """⚠ CROSS-MODULE SEAM — the connective tissue between the spec and the manifest
    factory, in files this workstream does not own (``studio/produce.py`` and its bus
    adapter ``video_intel/runners/studio_i2v.py``).

    Without it the lever is STILL unreachable from HTTP: the spec can carry a length
    and the factory can accept one, but ``produce_clip`` — the thin conductor that
    calls ``make_render_manifest`` — never passes it, so every real render is forced
    to the default exactly as before. Plumbed identically to the optional ``steps`` /
    ``cfg`` overrides, which take the same route."""
    from hugpy_video.intel.studio.produce import produce_clip

    assert "requested_frames" in inspect.signature(produce_clip).parameters, (
        "produce_clip must accept requested_frames and thread it into "
        "make_render_manifest, or StudioI2VSpec.requested_frames is a dead knob")


# --------------------------------------------------------------------------- #
# The lever itself — an explicit request
# --------------------------------------------------------------------------- #
def test_explicit_request_is_honoured():
    """In range and on cadence -> honoured VERBATIM. This is the lever that did not
    exist before: 33 frames = ~2.1s, 61 = ~3.8s, both 4k+1."""
    for want in (33, 61, 81):
        n, reason = resolve_frames(_manifest(requested_frames=want))
        assert n == want, f"requested {want} must be honoured; got {n} ({reason})"
        assert "requested" in reason, f"reason must record the request; got {reason!r}"


def test_request_reaches_the_geometry_tuple():
    """``_geometry`` is what both runners call — the request must show up THERE, with
    width/height/fps still coming from the top ladder rung."""
    w, h, fps, n = _geometry(_manifest(requested_frames=45))
    assert (w, h, fps) == (R480.width, R480.height, R480.fps), "rung geometry unchanged"
    assert n == 45, f"_geometry must return the requested 45 frames; got {n}"


def test_real_render_path_picks_up_the_lever():
    """``wan_i2v._wan_geometry`` imports ``_geometry``, so the REAL Wan path inherits
    the lever with no change of its own. Its 4k+1 snap must be IDEMPOTENT over the
    snap ``_geometry`` already did (resume check + generate call must agree)."""
    for want in (None, 45, 50, 200):
        m = _manifest(requested_frames=want)
        assert _wan_geometry(m) == _geometry(m), (
            f"_wan_geometry must be idempotent over _geometry for request {want}")
    assert _wan_geometry(_manifest())[3] == 81, "default Wan render is now 81 frames"


# --------------------------------------------------------------------------- #
# Clamp, never fail (INV-3 in spirit: a caller wants a clip, not a 500)
# --------------------------------------------------------------------------- #
def test_too_large_request_clamps_to_81_with_a_reason():
    """500 frames from a 5-second model is out of range -> CLAMP to the model's real
    ceiling and RECORD why. Never an exception, never an Err, never a 500."""
    n, reason = resolve_frames(_manifest(requested_frames=500))
    assert n == 81, f"a 500-frame request must clamp to 81; got {n} ({reason})"
    assert n == WAN_MAX_FRAMES, "the Wan hard ceiling IS 81"
    assert "clamp" in reason.lower(), f"the clamp must be RECORDED; got {reason!r}"
    assert "500" in reason and "81" in reason, (
        f"the reason must name what was asked and what was delivered; got {reason!r}")


def test_clamp_is_never_an_exception():
    """Every hostile length a caller can type resolves to a renderable int."""
    for want in (-1000, 0, 1, 2, 3, 4, 82, 1_000_000):
        n, reason = resolve_frames(_manifest(requested_frames=want))
        assert isinstance(n, int) and 1 <= n <= 81, (
            f"request {want} must resolve into range; got {n} ({reason})")


def test_cfg_max_frames_wins_when_lower():
    """The registry ceiling is authoritative when it is BELOW the default: cogvideox-5b
    declares max_frames=49, so it can never be handed the 81-frame default."""
    cfg = MODEL_REGISTRY.get("cogvideox-5b")
    assert cfg is not None and cfg.max_frames == 49, (
        "this check is pinned to cogvideox-5b's max_frames=49 registry row")
    cog = _manifest(model_id="cogvideox-5b", framework=Framework.COGVIDEOX, ladder=R_COG)
    n_default, _ = resolve_frames(cog)
    assert n_default == 49, f"the default must clamp to cfg.max_frames 49; got {n_default}"
    n_req, reason = resolve_frames(replace(cog, requested_frames=81))
    assert n_req == 49, f"an 81-frame request must clamp to 49; got {n_req} ({reason})"
    assert "clamp" in reason.lower(), f"the clamp must be RECORDED; got {reason!r}"


# --------------------------------------------------------------------------- #
# Cadence + floor
# --------------------------------------------------------------------------- #
def test_off_cadence_request_snaps_to_4k_plus_1():
    """Wan's latent VAE compresses time 4:1 -> num_frames == 4k+1. Snap DOWN (up could
    breach the ceiling, the one direction that turns a clamp into an OOM)."""
    n, reason = resolve_frames(_manifest(requested_frames=50))
    assert n == 49, f"50 must snap DOWN to 49; got {n} ({reason})"
    assert "snap" in reason.lower(), f"the snap must be RECORDED; got {reason!r}"
    for want, expect in ((50, 49), (51, 49), (52, 49), (53, 53), (80, 77), (81, 81)):
        got, why = resolve_frames(_manifest(requested_frames=want))
        assert got == expect, f"{want} must snap to {expect}; got {got} ({why})"
        assert (got - 1) % WAN_FRAME_CADENCE == 0, f"{got} is not 4k+1"
        # the shared helper and the resolver agree by construction (same function),
        # so a preset can quote the TRUE length before spending ~6min of denoise
        assert snap_wan_frames(want) == expect


def test_prover_does_not_snap():
    """The prover has no temporal VAE, so snapping its noise would shorten every
    existing synthetic clip to buy nothing."""
    n, _ = resolve_frames(_prover_manifest(requested_frames=30))
    assert n == 30, f"the prover must honour 30 frames unsnapped; got {n}"


def test_max_1_floor_survives():
    """The historical ``max(1, n)`` floor: zero/negative resolves to a 1-frame clip,
    not a crash and not an empty mp4. Unreachable through the validated spec/factory
    (both reject < 1 as caller error) — this is the belt-and-braces for a hand-forged
    manifest, e.g. one rehydrated from a hand-edited sidecar."""
    for want in (0, -1, -999):
        n, reason = resolve_frames(_manifest(requested_frames=want))
        assert n == 1, f"request {want} must floor to 1; got {n} ({reason})"
        assert "floor" in reason.lower(), f"the floor must be RECORDED; got {reason!r}"
    # and on the prover, where no snap can rescue an out-of-range value either
    n, _ = resolve_frames(_prover_manifest(requested_frames=0))
    assert n == 1, f"prover must floor to 1; got {n}"


# --------------------------------------------------------------------------- #
# INV-6 — length is part of the reproducibility key, on BOTH sides of the sidecar
# --------------------------------------------------------------------------- #
def test_length_is_in_the_content_hash():
    """Two requests differing ONLY in length are DIFFERENT clips. Without the length
    in the key, a 33-frame request would RESUME an existing 81-frame clip from the
    content-addressed path and hand back a clip 2.4x longer than asked for."""
    short = _manifest(requested_frames=33)
    long_ = _manifest(requested_frames=81)
    assert short.content_hash() != long_.content_hash(), (
        "requested_frames must participate in content_hash (INV-6)")
    # ...and it is still a PURE function of the manifest (identical request -> identical
    # hash), so resume/dedup keep working.
    assert short.content_hash() == _manifest(requested_frames=33).content_hash(), (
        "identical manifests must still hash equal")


def test_sidecar_round_trip_preserves_the_content_address():
    """⭐ THE HIGH-VALUE CHECK (defect 3). ``requested_frames`` is a content-hash
    input, so the ``manifest.json`` sidecar MUST carry it. It did not:
    ``render_manifest_to_dict`` omitted it and ``render_manifest_from_dict`` never
    restored it, so a rehydrated manifest silently fell back to the ``None`` default
    and re-addressed to a DIFFERENT hash.

    The consequence, spelled out because it is not obvious from the signature: a clip
    lives at ``<out_root>/<content_hash>/clip.mp4`` with its manifest.json beside it.
    Read that sidecar back and you compute a hash pointing at a DIFFERENT directory —
    so resume misses (the GPU re-burns ~2 minutes for a 33-frame Wan clip that already
    exists), dedup misses, and the manifest inside a directory disagrees with the
    directory's own name. Harmless ONLY while the field was permanently None, i.e.
    only while the lever was broken.

    Asserted on the WHOLE OBJECT, not just the hash: a field that round-trips to a
    different value but happens not to be hashed is still corruption of the
    provenance record."""
    for want in (33, 45, 81, None):
        m = _factory_manifest(requested_frames=want)
        d = render_manifest_to_dict(m)
        assert "requested_frames" in d, (
            f"to_dict must serialize the length (request {want}); a canonical field "
            f"missing here fails SILENTLY, by rehydrating to its default")
        assert d["requested_frames"] == want
        back = render_manifest_from_dict(d)
        assert back.requested_frames == want, (
            f"from_dict must restore the length; asked {want}, got "
            f"{back.requested_frames}")
        assert back == m, f"from_dict(to_dict(m)) must equal m (request {want})"
        assert back.content_hash() == m.content_hash(), (
            f"round trip must preserve the CONTENT ADDRESS (request {want}) — "
            f"otherwise resume/dedup break for every non-default length")


def test_sidecar_round_trip_survives_real_json():
    """The sidecar is written as JSON on disk (``atomic_write_text(json.dumps(...))``),
    so the round trip that matters crosses a real serialize/parse, not just a dict."""
    import json

    m = _factory_manifest(requested_frames=33)
    back = render_manifest_from_dict(json.loads(json.dumps(render_manifest_to_dict(m))))
    assert back == m and back.content_hash() == m.content_hash()


def test_legacy_sidecar_without_the_key_still_rehydrates():
    """A ``manifest.json`` written BEFORE this field was serialized has no key at all.
    It must rehydrate to ``None`` — which is exactly what those clips hashed with, so
    an old sidecar still resolves to its own content address rather than orphaning the
    clip sitting next to it."""
    m = _factory_manifest()                       # requested_frames is None
    legacy = {k: v for k, v in render_manifest_to_dict(m).items()
              if k != "requested_frames"}
    back = render_manifest_from_dict(legacy)
    assert back.requested_frames is None
    assert back.content_hash() == m.content_hash(), (
        "an old sidecar must still address the clip it sits beside")


CHECKS = [
    ("default for a real Wan model = 81 frames (measured)",
     test_default_real_wan_model_is_the_measured_ceiling),
    ("default for the synthetic prover stays fps*2 (~2s)",
     test_default_synthetic_prover_stays_cheap),
    ("the two defaults genuinely differ by runner class",
     test_defaults_differ_by_runner_class),
    ("the runner restates no length literal (one constant, imported)",
     test_the_runner_restates_no_length_literal),
    ("CROSS-MODULE: the wire publishes the renderer's number (presets.py)",
     test_the_wire_publishes_the_renderers_number),
    ("the bus SPEC carries a length (and round-trips it)",
     test_the_spec_carries_a_length),
    ("the spec range guard is a typo guard, not the model ceiling",
     test_the_spec_range_guard_is_a_typo_guard_not_a_ceiling),
    ("the ONE live manifest build path carries a length",
     test_the_manifest_factory_carries_a_length),
    ("the manifest factory rejects a structurally bad length",
     test_the_manifest_factory_rejects_a_structurally_bad_length),
    ("CROSS-MODULE: produce_clip threads the length (produce.py)",
     test_produce_clip_threads_the_length),
    ("an explicit in-range request is honoured verbatim",
     test_explicit_request_is_honoured),
    ("the request reaches _geometry's tuple; rung geometry unchanged",
     test_request_reaches_the_geometry_tuple),
    ("the real Wan path inherits the lever; _wan_geometry stays idempotent",
     test_real_render_path_picks_up_the_lever),
    ("a 500-frame request CLAMPS to 81 with a recorded reason",
     test_too_large_request_clamps_to_81_with_a_reason),
    ("no hostile length raises — every request resolves into range",
     test_clamp_is_never_an_exception),
    ("cfg.max_frames wins when lower (cogvideox-5b: 49)",
     test_cfg_max_frames_wins_when_lower),
    ("off-cadence requests snap DOWN to 4k+1 (50 -> 49)",
     test_off_cadence_request_snaps_to_4k_plus_1),
    ("the prover does NOT snap (no temporal VAE)",
     test_prover_does_not_snap),
    ("the max(1, n) floor survives",
     test_max_1_floor_survives),
    ("clip length participates in content_hash (INV-6)",
     test_length_is_in_the_content_hash),
    ("sidecar round trip preserves the CONTENT ADDRESS",
     test_sidecar_round_trip_preserves_the_content_address),
    ("sidecar round trip survives real JSON",
     test_sidecar_round_trip_survives_real_json),
    ("a legacy sidecar without the key still rehydrates",
     test_legacy_sidecar_without_the_key_still_rehydrates),
]


def main() -> int:
    passed = failed = 0
    for i, (name, fn) in enumerate(CHECKS, 1):
        try:
            fn()
        except Exception as exc:  # surface EVERY divergence, not just the first
            failed += 1
            import traceback
            print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"[{i}] PASS  {name}")
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
