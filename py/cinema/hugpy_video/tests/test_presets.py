"""PROVE the render presets — every row, against the real tree and the real store.

The whole point of ``studio/presets.py`` is that it is NOT a wish list, so this
suite deliberately refuses to assert the table against itself. "preset says model X
and the preset table says model X" proves nothing. Every check here reaches for an
independent source of truth:

  * the model_id must be a real ``MODEL_REGISTRY`` key (the zoo, not the preset);
  * the weights must be BYTES ON DISK under the studio weights root (the store);
  * a runner must be REGISTERED for (framework, task) AND must not be an
    unconditional-Err stub — derived STRUCTURALLY by parsing the runner module's AST
    and asking "does every return in the entrypoint construct an Err?", not by
    matching two hardcoded module names. The two known stubs are then used as a
    POSITIVE CONTROL on the detector, so a detector that silently stopped detecting
    fails the suite instead of passing everything;
  * the geometry must be one the ``ModelConfig`` actually declares, and the frame
    budget must satisfy Wan's 4k+1 latent cadence within the model's own max_frames;
  * the published ``default_frames`` must equal what the RENDERER would actually
    produce — asked of ``runners.synthetic.resolve_frames``, the one decider, not of
    a constant this file also imports;
  * ``proven`` must be true of the CLIP STORE, and a composite may never out-prove
    what it composes;
  * no preset may name one of the twelve zero-byte rows.

And the INVERSE, which is the reason the table exists at all: every ``Capability``
no preset covers must produce an honest refusal naming what IS available. A
capability we can neither serve nor refuse is the failure mode this whole slice was
built to delete.

⚠ TWO CHECKS EXIST BECAUSE A REVIEWER FOUND THE FIELDS THEY GUARD WERE UNCHECKED
(2026-07-27, second pass).

  * ``proven`` was the ONE field no test validated — ``evidence`` was only
    length-checked, which is not a check, and ``movie-480p`` shipped ``proven=True``
    while composing a ``proven=False`` preset whose 14B binding has produced zero of
    the 47 landed Wan clips. See ``test_proven_is_backed_by_clips_on_disk`` and
    ``test_a_composite_is_never_more_proven_than_what_it_composes``.
  * ``default_frames`` published 29 while the renderer defaulted to 81 — a 2.8x
    quote error on GPU minutes. See ``test_published_default_frames_is_what_the_
    renderer_produces``, which drives the real ``resolve_frames`` rather than
    comparing two constants.

STORE NOT MOUNTED: the on-disk checks SKIP with a named message (never pass
silently) when no studio weights root / clip store resolves — this suite must stay
runnable on a box that has the code but not the 450 GB shared store. Everything that
does not need the store still runs there.

Run:
  cd /srv/share/projects/hugpy/dev
  abstract_hugpy_dev/venv/bin/python -m pytest abstract_hugpy_dev/tests/studio/test_presets.py -q -p no:randomly
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess

import pytest

from hugpy_video.intel.studio import (
    CAPABILITY_TASKS,
    MODEL_REGISTRY,
    RUNNER_REGISTRY,
    Capability,
    Framework,
    Precision,
    runner_available,
    runner_gate_reason,
)
from hugpy_video.intel.studio.presets import (
    RENDER_BOX_VRAM_GIB,
    STUB_RUNNER_MODULES,
    WAN_MAX_FRAMES,
    ZERO_BYTE_MODELS,
    RenderPreset,
    all_presets,
    available_menu,
    capability_verdict,
    is_wan_cadence,
    preset,
    presets_for,
    refusal_for,
    servable_capabilities,
    snap_wan_frames,
    unservable_capabilities,
)

PRESETS = all_presets()
IDS = [p.preset_id for p in PRESETS]

# Presets whose "weights" are a system binary rather than bytes on the store. Their
# weight_uri carries a scheme (``ffmpeg://minterpolate``); they are proven by the
# binary + filter being present, which is a stricter check than a directory listing.
_BINARY_BACKED = {"enhance-upres", "enhance-interp"}

# The filter each binary-backed preset's runner actually invokes (ffmpeg_enhance.py:
# _interp_vf -> minterpolate, _upscale_vf -> scale=...:flags=lanczos).
_REQUIRED_FFMPEG_FILTERS = {
    "enhance-interp": "minterpolate",
    "enhance-upres": "scale",
}


# --------------------------------------------------------------------------- #
# Weights root resolution — the store, not the table
# --------------------------------------------------------------------------- #
def _candidate_weight_roots() -> tuple[str, ...]:
    """Every root a render could legitimately load from, in the runner's own order:
    the box-local hot NVMe copy first (``STUDIO_WEIGHTS_HOT_ROOT``), then the shared
    store (``STUDIO_WEIGHTS_ROOT``), then the canonical shared mount this fleet uses
    (``job.resolve_studio_env``'s explicit-override target). Mirrors
    ``runners.wan_i2v._resolve_model_dir`` so this suite proves what the RUNNER would
    find, not what a test-only path happens to hold."""
    roots = [
        os.environ.get("STUDIO_WEIGHTS_HOT_ROOT"),
        os.environ.get("STUDIO_WEIGHTS_ROOT"),
        "/mnt/llm_storage/video_intel/studio/weights",
    ]
    return tuple(r for r in roots if r and os.path.isdir(r))


def _model_dir(root: str, weight_uri: str) -> str:
    """``<root>/<org>/<name>`` — identical to ``wan_i2v._local_model_dir``."""
    return os.path.join(root, *[p for p in weight_uri.split("/") if p])


def _resolve_on_disk(weight_uri: str) -> str | None:
    """The first candidate root that holds a COMPLETE diffusers tree for this uri.

    Completeness gate is ``model_index.json`` — the same gate the runner preflight
    and the hot-copy resolver use, so a half-downloaded tree reads as absent here
    exactly as it would at render time."""
    for root in _candidate_weight_roots():
        d = _model_dir(root, weight_uri)
        if os.path.isfile(os.path.join(d, "model_index.json")):
            return d
    return None


def _tree_bytes(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def _require_store() -> None:
    if not _candidate_weight_roots():
        pytest.skip(
            "studio weights store not mounted on this box: none of "
            "STUDIO_WEIGHTS_HOT_ROOT / STUDIO_WEIGHTS_ROOT / "
            "/mnt/llm_storage/video_intel/studio/weights resolves to a directory. "
            "The on-disk proof cannot run here — this is a SKIP, not a pass.")


# --------------------------------------------------------------------------- #
# The CLIP STORE — the independent witness for ``proven``
# --------------------------------------------------------------------------- #
# ``job.STUDIO_ROOT`` is ``<DEFAULT_ROOT>/video_intel/studio`` and the two things
# under it that matter here are siblings: ``weights/`` (what a runner loads) and
# ``clips/`` (the content-addressed output root every produce_clip writes into). So
# the clip store is derived from whichever weights root resolved, with the fleet's
# canonical mount as the explicit fallback — the same shape as _candidate_weight_roots
# above, and for the same reason: prove what the SPINE would find, not a test path.
#
# NOT imported from ``studio.job``: that module pulls the media-store constants (and
# through them a store scan) into a suite that otherwise needs nothing but the tree.
def _candidate_clip_roots() -> tuple[str, ...]:
    roots: list[str] = []
    for weights_root in _candidate_weight_roots():
        roots.append(os.path.join(os.path.dirname(os.path.abspath(weights_root)), "clips"))
    roots.append("/mnt/llm_storage/video_intel/studio/clips")
    seen: set[str] = set()
    out: list[str] = []
    for r in roots:
        if r not in seen and os.path.isdir(r):
            seen.add(r)
            out.append(r)
    return tuple(out)


def _landed_clip_models() -> dict[str, int]:
    """``{model_id: completed clips}`` read off the clip store's own manifests.

    A clip COUNTS only when its dir holds both a ``manifest.json`` naming the model
    and a NON-EMPTY ``clip.mp4`` — a manifest with no pixels beside it is a render
    that was addressed and never finished, which is precisely what ``proven`` must
    not be allowed to count. Measured this way on 2026-07-27 the store held 47 real
    Wan clips (14 wan2.1-t2v-1.3b + 33 wan2.1-vace-1.3b), 40 synthetic-prover clips,
    and one each from the two ffmpeg enhancers — and ZERO from any 14B row, which is
    the fact ``clip-i2v-480p`` and ``movie-480p`` are labelled unproven against."""
    counts: dict[str, int] = {}
    for root in _candidate_clip_roots():
        for name in os.listdir(root):
            d = os.path.join(root, name)
            manifest_path = os.path.join(d, "manifest.json")
            clip_path = os.path.join(d, "clip.mp4")
            if not os.path.isfile(manifest_path):
                continue
            try:
                if not os.path.getsize(clip_path) > 0:
                    continue
            except OSError:
                continue
            try:
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    model_id = json.load(fh).get("model_id")
            except (OSError, ValueError):
                continue
            if model_id:
                counts[model_id] = counts.get(model_id, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Unconditional-Err stub detection, derived from the runner's own AST
# --------------------------------------------------------------------------- #
# A stub runner is one whose entrypoint cannot succeed: EVERY return it can reach
# constructs an ``Err(...)``. That is a structural property of the code, so we read
# the code. ``importlib.util.find_spec`` locates the module WITHOUT executing it
# (the runners import torch/diffusers lazily, but their module bodies still pull
# numpy/PIL and we do not want that here), then we parse the source.
#
# Delegation is followed ONE level into module-local helpers, because that is how
# the real runners are shaped: ``run_ffmpeg_upscale`` returns ``_enhance(...)`` and
# ``run_wan_t2v`` returns ``run_wan_i2v(...)``. Without following it, every
# delegating runner would look like "no Err returns" (fine) and without bounding it,
# a cycle would hang. A call to a NON-local name (an imported runner) is treated as
# possibly-Ok, which is correct: we cannot see inside it from here, and a stub never
# delegates outward — it returns Err in place.
_MAX_DELEGATION_DEPTH = 3


def _module_source(module_path: str) -> str | None:
    try:
        spec = importlib.util.find_spec(module_path)
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError, TypeError):
        return None
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        return None
    with open(spec.origin, "r", encoding="utf-8") as fh:
        return fh.read()


def _is_err_call(node: ast.AST) -> bool:
    """True for ``Err(...)`` — the errors-as-data failure constructor."""
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Err")


def _own_returns(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Return]:
    """Every ``return`` belonging to ``fn`` itself — NOT those of functions nested
    inside it (a nested helper's returns say nothing about the entrypoint's)."""
    out: list[ast.Return] = []
    stack: list[ast.AST] = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Return):
            out.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return out


def _returns_only_err(module_path: str, func_name: str) -> bool | None:
    """True iff ``func_name`` in ``module_path`` can ONLY return an ``Err``.

    None when the module or function cannot be found (the caller turns that into a
    named failure rather than a silent pass)."""
    source = _module_source(module_path)
    if source is None:
        return None
    tree = ast.parse(source)
    funcs = {n.name: n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if func_name not in funcs:
        return None

    def only_err(name: str, depth: int) -> bool:
        fn = funcs.get(name)
        if fn is None or depth > _MAX_DELEGATION_DEPTH:
            return False          # can't see inside -> assume it may succeed
        returns = _own_returns(fn)
        if not returns:
            return False          # falls off the end -> returns None, not an Err
        for ret in returns:
            value = ret.value
            if value is None:
                return False
            if _is_err_call(value):
                continue
            # A bare delegation to a module-local helper: follow it.
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id in funcs):
                if only_err(value.func.id, depth + 1):
                    continue
            return False          # at least one return can carry an Ok
        return True

    return only_err(func_name, 0)


# --------------------------------------------------------------------------- #
# The table itself
# --------------------------------------------------------------------------- #
def test_the_ratified_eight_and_nothing_else():
    """The preset set is the deliverable and was ratified by the operator. A ninth
    row means a ninth PROOF, so its arrival must break this test on purpose.

    STILL EIGHT, BUT NOT THE SAME EIGHT (2026-07-27, second pass). ``clip-control-480p``
    is GONE: it advertised motion + inpaint + outpaint + retake and served none of
    them (the route rejected a control for all four, and without one they rendered a
    plain full restyle — the v2v path under a different name). It was replaced by
    ``clip-motion-480p``, which serves the ONE capability that has a real branch
    behind it. inpaint/outpaint/retake moved to the refusal side, which
    ``test_the_uncovered_set_is_exactly_the_eight_we_cannot_serve`` pins."""
    assert IDS == [
        "clip-t2v-480p", "clip-idlock-480p", "clip-v2v-480p", "clip-motion-480p",
        "clip-i2v-480p", "movie-480p", "enhance-upres", "enhance-interp",
    ], f"preset set changed: {IDS}"
    assert len(set(IDS)) == len(IDS), "duplicate preset_id"
    assert "clip-control-480p" not in IDS, (
        "clip-control-480p was retired 2026-07-27: it published four capabilities and "
        "rendered a fifth thing. Re-adding it needs four proofs, not one row")


def test_each_preset_serves_exactly_one_capability():
    """The lesson of clip-control-480p, encoded. A row may in principle serve several
    capabilities — the field is a tuple — but every one it lists is a promise this
    suite has to be able to prove independently, and the row that listed four proved
    one. Today's ratified eight are one-capability rows; a future multi-capability row
    is allowed, but it has to arrive as a deliberate edit here alongside the proof
    that each of its capabilities reaches a DIFFERENT render."""
    for p in PRESETS:
        assert len(p.capabilities) == 1, (
            f"{p.preset_id} advertises {[c.value for c in p.capabilities]} — each "
            f"needs its own proof that it reaches a distinct render, or the row is "
            f"publishing a capability that renders something else")


@pytest.mark.parametrize("p", PRESETS, ids=IDS)
def test_preset_is_well_formed(p: RenderPreset):
    """Shape only — the cheap gate that makes the expensive proofs meaningful."""
    assert p.preset_id and p.title and p.description
    assert p.capabilities, f"{p.preset_id}: serves no capability"
    assert p.capability is p.capabilities[0]
    assert p.evidence and len(p.evidence) > 40, (
        f"{p.preset_id}: evidence must record WHY we believe this works")
    assert isinstance(p.precision, Precision)
    assert isinstance(p.framework, Framework)
    assert p.inputs, f"{p.preset_id}: declares no inputs"
    assert preset(p.preset_id) is p


# --------------------------------------------------------------------------- #
# PROOF 1 — the model is real
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", PRESETS, ids=IDS)
def test_model_id_exists_in_the_zoo(p: RenderPreset):
    """The preset's model_id is a key of MODEL_REGISTRY — the zoo, populated by
    models_seed, not by this table. The ffmpeg rows ARE registry rows too (they
    carry a pinned pseudo weight_hash and a ffmpeg:// weight_uri), so there is no
    exemption to make here."""
    cfg = MODEL_REGISTRY.get(p.model_id)
    assert cfg is not None, f"{p.preset_id}: model_id {p.model_id!r} is not in MODEL_REGISTRY"
    assert cfg.family is p.framework, (
        f"{p.preset_id}: framework {p.framework.value} != registry family {cfg.family.value}")
    assert p.task in cfg.tasks, (
        f"{p.preset_id}: task {p.task.value} not in {p.model_id}'s tasks "
        f"{[t.value for t in cfg.tasks]}")


@pytest.mark.parametrize("p", PRESETS, ids=IDS)
def test_preset_capabilities_are_declared_by_its_model(p: RenderPreset):
    """Every capability the preset advertises is one the bound ModelConfig claims —
    except ASSEMBLE, which registry.PLANNED_CAPABILITIES declares is orchestration
    and deliberately backed by no model. For the movie preset we instead prove that
    every preset it COMPOSES is real and that its own model_id renders segment 0."""
    cfg = MODEL_REGISTRY[p.model_id]
    for cap in p.capabilities:
        if cap is Capability.ASSEMBLE:
            assert p.composes, "an ASSEMBLE preset must name the presets it composes"
            for child in p.composes:
                assert preset(child) is not None, f"{p.preset_id} composes unknown {child!r}"
            assert p.joints, "an ASSEMBLE preset must name its joint modes"
            continue
        assert cap in cfg.capabilities, (
            f"{p.preset_id}: {p.model_id} does not declare capability {cap.value!r} "
            f"(it declares {[c.value for c in cfg.capabilities]})")
        assert p.task in CAPABILITY_TASKS.get(cap, ()), (
            f"{p.preset_id}: task {p.task.value} does not satisfy capability {cap.value!r}")


@pytest.mark.parametrize("p", PRESETS, ids=IDS)
def test_precision_is_in_the_models_vram_envelope(p: RenderPreset):
    """The precision is one the model actually publishes a VRAM figure for. This is
    what catches "we chose nf4" written against an enum that has no NF4 member: the
    fleet's 4-bit lever is Precision.FP8 (wan_i2v._bnb_config maps FP8 ->
    load_in_4bit + nf4), and the envelope is keyed on FP8."""
    envelope = MODEL_REGISTRY[p.model_id].vram.as_map()
    assert p.precision in envelope, (
        f"{p.preset_id}: {p.model_id} publishes no VRAM figure for "
        f"{p.precision.value} (has {[k.value for k in envelope]})")


# --------------------------------------------------------------------------- #
# PROOF 2 — the weights are bytes on disk (or the binary is on PATH)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "p", [p for p in PRESETS if p.preset_id not in _BINARY_BACKED],
    ids=[p.preset_id for p in PRESETS if p.preset_id not in _BINARY_BACKED])
def test_weights_are_on_disk(p: RenderPreset):
    """Resolve the model's weight_uri under the studio weights root exactly as the
    runner would and require a complete tree with real bytes in it. This is the
    check that separates the three servable models from the twelve zero-byte rows."""
    _require_store()
    cfg = MODEL_REGISTRY[p.model_id]
    assert "://" not in cfg.weight_uri, (
        f"{p.preset_id}: {p.model_id} has a scheme-bearing weight_uri "
        f"{cfg.weight_uri!r} but is not declared binary-backed")
    model_dir = _resolve_on_disk(cfg.weight_uri)
    assert model_dir is not None, (
        f"{p.preset_id}: no complete tree for {cfg.weight_uri} under any of "
        f"{_candidate_weight_roots()} (model_index.json missing) — the preset "
        f"advertises a render the box cannot load")
    size = _tree_bytes(model_dir)
    assert size > 1_000_000_000, (
        f"{p.preset_id}: {model_dir} holds only {size} bytes — not a real "
        f"diffusers tree")


@pytest.mark.parametrize("preset_id", sorted(_BINARY_BACKED))
def test_binary_backed_preset_has_its_tool(preset_id: str):
    """The two enhance presets have no weights BY DESIGN — they are real transforms
    of real pixels via the system ffmpeg binary. The equivalent of "weights on disk"
    for them is "the binary exists and ships the filter the runner invokes", which
    we ask ffmpeg itself rather than assuming."""
    p = preset(preset_id)
    assert p is not None
    cfg = MODEL_REGISTRY[p.model_id]
    assert cfg.weight_uri.startswith("ffmpeg://"), (
        f"{preset_id}: expected a ffmpeg:// weight_uri, got {cfg.weight_uri!r}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not on PATH on this box — cannot prove the enhance "
                    "presets here (SKIP, not a pass)")
    out = subprocess.run([ffmpeg, "-hide_banner", "-filters"],
                         capture_output=True, text=True, timeout=60).stdout
    wanted = _REQUIRED_FFMPEG_FILTERS[preset_id]
    assert any(line.split()[1:2] == [wanted] for line in out.splitlines()), (
        f"{preset_id}: this ffmpeg build has no {wanted!r} filter")


# --------------------------------------------------------------------------- #
# PROOF 3 — a REAL runner, not a stub that merely exists
# --------------------------------------------------------------------------- #
def test_stub_detector_actually_detects_the_known_stubs():
    """POSITIVE CONTROL on the detector below. ltx_upscale and rife_interpolate are
    the two modules that exist purely to degrade gracefully and return Err on every
    path — and whose mere existence defeats the k1 find_spec gate. If the AST
    detector ever stops flagging them (a refactor, a real body landing) this test
    fails and the per-preset check below stops being meaningful silently."""
    known = {
        "hugpy_video.intel.studio.runners.ltx_upscale": "run_ltx_upscale",
        "hugpy_video.intel.studio.runners.rife_interpolate": "run_rife_interpolate",
    }
    assert set(known) == set(STUB_RUNNER_MODULES), (
        "presets.STUB_RUNNER_MODULES drifted from the modules under control")
    for module_path, func in known.items():
        verdict = _returns_only_err(module_path, func)
        assert verdict is True, (
            f"{module_path}:{func} is no longer detected as unconditional-Err "
            f"(got {verdict!r}) — either it gained a real body (delete it from "
            f"STUB_RUNNER_MODULES) or the detector broke")


def test_stub_detector_does_not_flag_the_working_runners():
    """NEGATIVE CONTROL: a detector that flags everything would make PROOF 3 vacuous.
    wan_t2v is the sharp case — it delegates outward to run_wan_i2v and its only
    literal ``Ok(`` is in a docstring, so a grep-based detector would call it a stub."""
    for module_path, func in (
        ("hugpy_video.intel.studio.runners.wan_i2v", "run_wan_i2v"),
        ("hugpy_video.intel.studio.runners.wan_t2v", "run_wan_t2v"),
        ("hugpy_video.intel.studio.runners.wan_vace", "run_wan_vace"),
        ("hugpy_video.intel.studio.runners.ffmpeg_enhance", "run_ffmpeg_upscale"),
        ("hugpy_video.intel.studio.runners.ffmpeg_enhance", "run_ffmpeg_interpolate"),
    ):
        assert _returns_only_err(module_path, func) is False, (
            f"{module_path}:{func} was wrongly flagged as an unconditional-Err stub")


@pytest.mark.parametrize("p", PRESETS, ids=IDS)
def test_runner_is_registered_and_is_not_a_stub(p: RenderPreset):
    """A RunnerSpec exists for (framework, task), its module resolves (the k1 gate
    passes), AND the entrypoint is not one that can only return Err. The last clause
    is the one k1 cannot express — that is exactly the gap that let UPRES/INTERP bind
    a stub on the only render box."""
    spec = RUNNER_REGISTRY.get((p.framework, p.task))
    assert spec is not None, (
        f"{p.preset_id}: no RunnerSpec for ({p.framework.value}, {p.task.value})")
    assert runner_available(p.framework, p.task) is not None, (
        f"{p.preset_id}: runner gated — {runner_gate_reason(p.framework, p.task)}")

    module_path, _, func = spec.entrypoint.partition(":")
    assert func, f"{p.preset_id}: entrypoint {spec.entrypoint!r} names no callable"
    assert module_path not in STUB_RUNNER_MODULES, (
        f"{p.preset_id}: bound to known unconditional-Err stub {module_path}")
    verdict = _returns_only_err(module_path, func)
    assert verdict is not None, (
        f"{p.preset_id}: could not read {module_path}:{func} to prove it is a real "
        f"runner")
    assert verdict is False, (
        f"{p.preset_id}: {module_path}:{func} returns Err on EVERY path — it is a "
        f"stub, and a stub that exists is strictly worse than no module (it satisfies "
        f"the k1 find_spec gate and outranks the working last-resort)")


# --------------------------------------------------------------------------- #
# PROOF 4 — the geometry and the frame budget are ones the model really supports
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", PRESETS, ids=IDS)
def test_geometry_is_one_the_model_supports(p: RenderPreset):
    """The declared width/height must be covered by one of the ModelConfig's own
    ``resolutions`` (Resolution.covers: >= in BOTH dims). This is what keeps a
    preset from advertising 512x512 against wan2.1-vace-1.3b, whose only declared
    resolution is 832x480 — the exact mismatch that made >480p id_lock fall through
    to a runner that ignores reference images."""
    if p.width is None:
        # Enhance presets: geometry comes from the source clip and the manifest
        # target, so there is nothing to pin. Prove the encoding is consistent
        # rather than pretending there is a geometry to check.
        assert p.height is None and p.fps is None, (
            f"{p.preset_id}: partial geometry — width is None but height/fps are not")
        assert p.default_frames is None and p.max_frames is None
        return
    cfg = MODEL_REGISTRY[p.model_id]
    assert any(r.width >= p.width and r.height >= p.height for r in cfg.resolutions), (
        f"{p.preset_id}: {p.geometry} is not covered by {p.model_id}'s resolutions "
        f"{[(r.width, r.height) for r in cfg.resolutions]}")
    assert p.fps and p.fps > 0


@pytest.mark.parametrize("p", [p for p in PRESETS if p.framework is Framework.WAN],
                         ids=[p.preset_id for p in PRESETS if p.framework is Framework.WAN])
def test_wan_frame_budget_and_cadence(p: RenderPreset):
    """Wan's latent VAE compresses time 4:1, so ``num_frames`` must be 4k+1 —
    wan_i2v._wan_geometry snaps to it at render time, which means a preset declaring
    an off-cadence count would silently render a DIFFERENT length than it advertises.
    Also: default <= max <= the model's own ceiling <= 81."""
    cfg = MODEL_REGISTRY[p.model_id]
    assert p.default_frames is not None and p.max_frames is not None
    assert p.default_frames <= p.max_frames, f"{p.preset_id}: default > max"
    assert p.max_frames <= WAN_MAX_FRAMES, f"{p.preset_id}: max_frames > {WAN_MAX_FRAMES}"
    assert p.max_frames <= cfg.max_frames, (
        f"{p.preset_id}: max_frames {p.max_frames} exceeds {p.model_id}'s "
        f"registry ceiling {cfg.max_frames}")
    for n in (p.default_frames, p.max_frames):
        assert is_wan_cadence(n), f"{p.preset_id}: {n} frames violates Wan's 4k+1 cadence"
        assert snap_wan_frames(n) == n, (
            f"{p.preset_id}: the runner would snap {n} down to {snap_wan_frames(n)}")


# --------------------------------------------------------------------------- #
# PROOF 4b — the WIRE's clip length is the one the RENDERER will actually produce
# --------------------------------------------------------------------------- #
def test_published_default_frames_is_what_the_renderer_produces():
    """THE DEFECT: ``presets`` published ``default_frames=29`` on every Wan row while
    ``runners.synthetic.resolve_frames`` defaulted a real model to 81. Two literals,
    2.8x apart, with no import between them — so ``GET /video/render/presets`` quoted
    a ~1.8 s clip and the fleet rendered a 5.06 s one, at 2.8x the latent tokens and
    2.8x the GPU minutes. Defaults-are-promises, on the most expensive axis the studio
    has.

    THE CHECK DRIVES THE DECIDER, it does not compare two constants. A constants
    comparison would pass the moment both sides imported the same wrong number; this
    builds a real ``RenderManifest`` for each Wan preset and asks ``resolve_frames``
    — the ONE place clip length is decided, with its clamp-to-ceiling and 4k+1 snap —
    what the runner would render for a caller who requested nothing. If that ever
    diverges from what the row publishes, the wire is lying again and this fails."""
    from hugpy_video.intel.studio.env import StudioEnv
    from hugpy_video.intel.studio.manifest import make_render_manifest
    from hugpy_video.intel.studio.runners.synthetic import resolve_frames
    from hugpy_video.intel.studio.schemas import ModelBinding, Resolution, SamplerConfig, SeedBundle

    env = StudioEnv(
        output_root="/out", weights_root="/weights", manifest_root="/manifests",
        master_colorspace="rec709", master_fps=16, max_vram_gb=RENDER_BOX_VRAM_GIB,
        loudness_target_lufs=-14.0, allow_unpinned=True,
    )
    wan = [p for p in PRESETS if p.framework is Framework.WAN]
    assert wan, "no Wan presets — this proof would be vacuous"
    for p in wan:
        # The binding is built from the REGISTRY row (weight_uri / weight_hash /
        # path_class / determinism come from the model, exactly as the router would
        # thread them) rather than by resolving through CapabilityRouter — the movie
        # preset's capability is ASSEMBLE, which is orchestration and never binds a
        # model, so a router round trip could not cover the whole table.
        cfg = MODEL_REGISTRY[p.model_id]
        binding = ModelBinding(
            model_id=p.model_id, framework=p.framework, task=p.task,
            precision=p.precision, path_class=cfg.path_class,
            weight_uri=cfg.weight_uri, weight_hash=cfg.weight_hash,
            determinism_class=cfg.default_determinism)
        manifest = make_render_manifest(
            render_id=f"presets-frames-{p.preset_id}",
            capability=p.capability, binding=binding,
            seeds=SeedBundle(global_seed=0),
            sampler=SamplerConfig(sampler="unipc", scheduler="unipc",
                                  steps=10, cfg=5.0, shift=3.0),
            resolution_ladder=(Resolution(p.width, p.height, p.fps),),
            env=env,
            # requested_frames DELIBERATELY unset: this asks what a caller who names
            # no length gets, which is exactly what ``default_frames`` promises them.
        )
        rendered, reason = resolve_frames(manifest)
        assert rendered == p.default_frames, (
            f"{p.preset_id}: the wire publishes default_frames={p.default_frames} but "
            f"the renderer would produce {rendered} ({reason}) — a caller quoted "
            f"{p.default_frames} frames would be charged for {rendered}")


# --------------------------------------------------------------------------- #
# PROOF 4c — ``proven`` is a claim about the CLIP STORE, checked against it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [p for p in PRESETS if p.proven],
                         ids=[p.preset_id for p in PRESETS if p.proven])
def test_proven_is_backed_by_clips_on_disk(p: RenderPreset):
    """``proven=True`` means "this exact path has produced pixels on this fleet", so
    the fleet is asked, not the table. At least one completed clip in the studio clip
    store must name this preset's model_id in its manifest.

    A reviewer's finding is the reason this exists: ``proven`` was the one field NO
    test validated (``evidence`` was length-checked, which is not a check), and one
    row shipped True on evidence that contradicted the table it sat in.

    Direction matters and only one direction is asserted. proven=True MUST have
    clips — that is the claim. proven=False is the humble side and is asserted
    nowhere here: ``clip-motion-480p`` is unproven precisely because its branch was
    unreachable until today, yet its MODEL (vace-1.3b) has 33 clips from OTHER
    branches, so "0 clips" would be the wrong test for it."""
    clip_roots = _candidate_clip_roots()
    if not clip_roots:
        pytest.skip(
            "studio clip store not mounted on this box (no <studio>/clips beside any "
            "resolvable weights root) — the proven-against-pixels proof cannot run "
            "here. SKIP, not a pass.")
    counts = _landed_clip_models()
    assert counts, (
        f"the clip store at {clip_roots} holds no completed clips at all — either it "
        f"is the wrong root or nothing has ever rendered; refusing to call that a pass")
    assert counts.get(p.model_id, 0) > 0, (
        f"{p.preset_id} claims proven=True but the clip store holds ZERO completed "
        f"clips from {p.model_id}. Store census: "
        f"{dict(sorted(counts.items()))}")


def test_the_unproven_rows_really_have_no_pixels_from_their_binding():
    """The sharp end of the same question, pinned for the one row whose unprovenness
    is a FACT rather than an absence of opportunity: no 14B model has produced a
    single clip on this fleet. If one ever does, this fails and someone must decide
    whether ``clip-i2v-480p`` / ``movie-480p`` have earned proven=True — which is the
    correct amount of friction for flipping a claim like that."""
    if not _candidate_clip_roots():
        pytest.skip("studio clip store not mounted on this box — SKIP, not a pass.")
    counts = _landed_clip_models()
    fourteen_b = {m: n for m, n in counts.items() if "14b" in m.lower()}
    assert not fourteen_b, (
        f"a 14B row has landed clips ({fourteen_b}) — clip-i2v-480p and movie-480p "
        f"are labelled unproven on the claim that none ever had. Re-read the evidence "
        f"strings before flipping them")
    unproven = {p.preset_id for p in PRESETS if not p.proven}
    assert "clip-i2v-480p" in unproven and "movie-480p" in unproven, unproven


def test_a_composite_is_never_more_proven_than_what_it_composes():
    """THE RULE ``movie-480p`` broke. It shipped ``proven=True`` while composing
    ``clip-i2v-480p`` (proven=False) — its ``still`` joint binds the 14B i2v, and the
    clip store holds zero clips from any 14B row, so that joint has never once
    completed here. A composite's claim can only be as strong as its weakest part.

    Note the OTHER way this could have been "fixed" and why it would have been worse:
    dropping clip-i2v-480p from ``composes`` would have made the rule pass while
    making the table lie about what a still joint binds (studio_movie really does
    force capability i2v there). ``composes`` is a fact; ``proven`` is the claim, so
    the claim is what moved."""
    for p in PRESETS:
        if not p.composes:
            continue
        for child_id in p.composes:
            child = preset(child_id)
            assert child is not None, f"{p.preset_id} composes unknown {child_id!r}"
            if p.proven:
                assert child.proven, (
                    f"{p.preset_id} claims proven=True but composes {child_id!r}, "
                    f"which is proven=False — a composite cannot out-prove its parts")


def test_evidence_says_which_way_proven_points():
    """``evidence`` is prose, so it cannot be fully checked — but it must not
    CONTRADICT the boolean beside it. An unproven row has to say so in words a reader
    scanning the wire will see, because ``evidence`` is what the console shows and
    ``proven`` is a checkbox nobody reads."""
    for p in PRESETS:
        lowered = p.evidence.lower()
        if p.proven:
            assert "proven:" in lowered or lowered.startswith("proven"), (
                f"{p.preset_id}: proven=True but the evidence does not open by saying "
                f"so: {p.evidence[:80]!r}")
        else:
            assert "not proven" in lowered or "not yet proven" in lowered, (
                f"{p.preset_id}: proven=False but the evidence never says it is "
                f"unproven: {p.evidence[:80]!r}")


# --------------------------------------------------------------------------- #
# PROOF 5 — nothing here touches a zero-byte row
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", PRESETS, ids=IDS)
def test_preset_names_no_zero_byte_model(p: RenderPreset):
    assert p.model_id not in ZERO_BYTE_MODELS, (
        f"{p.preset_id}: bound to {p.model_id}, which holds ZERO bytes on the store")


def test_zero_byte_models_are_registry_rows_and_really_are_absent():
    """Two things at once. (1) Every name in ZERO_BYTE_MODELS is a real registry row
    — a typo there would silently weaken the guard above into a no-op. (2) They
    really are absent from the store, so the constant stays a CHECKED FACT rather
    than a claim copied out of a document that may have aged."""
    for model_id in sorted(ZERO_BYTE_MODELS):
        assert model_id in MODEL_REGISTRY, (
            f"ZERO_BYTE_MODELS names {model_id!r}, which is not a registry row")
    _require_store()
    present = []
    for model_id in sorted(ZERO_BYTE_MODELS):
        uri = MODEL_REGISTRY[model_id].weight_uri
        for root in _candidate_weight_roots():
            d = _model_dir(root, uri)
            if os.path.isdir(d) and _tree_bytes(d) > 0:
                present.append(f"{model_id} -> {d}")
    assert not present, (
        "these ZERO_BYTE_MODELS now have bytes on the store; re-measure before "
        "trusting the constant: " + ", ".join(present))


# --------------------------------------------------------------------------- #
# THE INVERSE — everything not covered must REFUSE, by name
# --------------------------------------------------------------------------- #
def test_every_capability_is_either_served_or_refused():
    """The point of the whole exercise. Every member of the Capability enum falls in
    exactly one bucket, and there is no third bucket where a request is accepted,
    enqueued, and dies in a runner three layers down."""
    served = servable_capabilities()
    unserved = unservable_capabilities()
    assert served | unserved == set(Capability)
    assert not (served & unserved)
    for cap in Capability:
        verdict = capability_verdict(cap)
        if cap in served:
            assert verdict.servable and verdict.preset_ids and not verdict.refusal
            for pid in verdict.preset_ids:
                assert preset(pid) is not None
        else:
            assert not verdict.servable and not verdict.preset_ids
            assert verdict.reason, f"{cap.value}: refused with no reason"
            assert verdict.refusal, f"{cap.value}: refused with no user-facing text"


def test_the_uncovered_set_is_exactly_the_eight_we_cannot_serve():
    """Ratified 2026-07-27 against the measured fleet, then RE-ratified the same day:
    8 of 16 capabilities are servable, not 11. Pinning the other eight here means
    adding a preset (or losing one) forces a deliberate edit rather than drifting
    quietly.

    THE THREE THAT MOVED. inpaint / outpaint / retake were "servable" only because
    ``clip-control-480p`` listed them beside motion on one VACE binding. Measured, the
    route rejected a control image for all four and, without one, rendered a plain
    full restyle for all four. There is no mask, expanded-canvas or frame-range input
    ANYWHERE in the spine (checked across video_routes / job / schemas / produce /
    wan_vace), so those three cannot be made real by wiring — only by adding an input
    channel that does not exist. They refuse."""
    assert {c.value for c in unservable_capabilities()} == {
        "keyframe", "stream", "audio", "lipsync", "restore",
        "inpaint", "outpaint", "retake"}
    assert len(servable_capabilities()) == 8


@pytest.mark.parametrize("cap_value", ["inpaint", "outpaint", "retake"])
def test_the_demoted_three_refuse_by_naming_the_restyle_and_v2v(cap_value: str):
    """The specific honesty these three refusals owe a caller. It is NOT enough to
    say "unsupported": until 2026-07-27 the route ACCEPTED all three and rendered
    something — a full restyle of the whole clip — so the refusal has to say (a) what
    would actually have come back, and (b) the capability that asks for that on
    purpose. Otherwise the caller reads "unsupported", tries again with different
    words, and gets the restyle anyway."""
    cap = Capability(cap_value)
    text = refusal_for(cap)
    assert text is not None, f"{cap_value} must refuse"
    assert "restyle" in text.lower(), (
        f"{cap_value}: the refusal must say what you WOULD have got: {text}")
    assert "v2v" in text, (
        f"{cap_value}: the refusal must name the capability that renders a restyle on "
        f"purpose: {text}")
    # ...and must NOT read as a missing download. These three fail on a CONTRACT gap;
    # vace-1.3b is the most proven row on the fleet.
    assert "0 bytes" not in text and "download" not in text.lower(), (
        f"{cap_value}: the weights are present and proven — the refusal must not read "
        f"as a missing model: {text}")


def test_keyframe_refusal_admits_the_same_weights_are_blessed_elsewhere():
    """THE ASYMMETRY A REVIEWER FLAGGED, resolved explicitly. keyframe is refused as
    "a rename of i2v" while the SAME weights (wan2.1-i2v-14b-720p) are blessed as
    clip-i2v-480p. That is defensible — the gap is an END-FRAME INPUT, not a model —
    but the old wording ("the only model declaring keyframe ... has never produced a
    clip on this fleet") read as if the model were absent, which would send a caller
    to re-download 90 GB that is already on disk and would not have helped."""
    text = refusal_for(Capability.KEYFRAME)
    assert text is not None
    lowered = text.lower()
    assert "wan2.1-i2v-14b-720p" in text, (
        f"the refusal must name the model it shares with clip-i2v-480p: {text}")
    assert "clip-i2v-480p" in text, (
        f"the refusal must point at the preset those same weights DO serve: {text}")
    assert "input" in lowered, (
        f"the refusal must say the gap is the input channel, not the model: {text}")
    assert "start_image" in text, (
        f"the refusal must name the route that works today: {text}")


@pytest.mark.parametrize("cap", sorted(unservable_capabilities(), key=lambda c: c.value),
                         ids=lambda c: c.value)
def test_refusal_names_what_is_available(cap: Capability):
    """A refusal that only says "unsupported" teaches the caller nothing. Every one
    of ours has to carry a measured reason AND the menu of what IS renderable."""
    text = refusal_for(cap)
    assert text is not None
    assert cap.value in text
    verdict = capability_verdict(cap)
    assert verdict.reason in text
    # Names real alternatives, not a generic apology.
    for other in ("t2v", "id_lock", "upres"):
        assert other in text, f"refusal for {cap.value} does not mention {other}"
    assert "clip-t2v-480p" in text


def test_available_menu_is_stable_and_covers_every_preset():
    """The menu is embedded in every refusal, so it must be deterministic (a message
    that reorders itself reads as a different error in a log diff) and complete."""
    menu = available_menu()
    assert menu == available_menu()
    for p in PRESETS:
        assert p.preset_id in menu, f"{p.preset_id} missing from the available menu"


@pytest.mark.parametrize("cap", sorted(servable_capabilities(), key=lambda c: c.value),
                         ids=lambda c: c.value)
def test_served_capabilities_do_not_refuse(cap: Capability):
    assert refusal_for(cap) is None
    assert presets_for(cap), f"{cap.value} is reported servable but has no preset"


def test_unknown_preset_id_is_data_not_an_exception():
    """The route takes a preset_id straight off the wire; an unknown one is caller
    data, never programmer error."""
    assert preset("does-not-exist") is None
    assert preset("") is None
