"""Capability router (ORCH-3 / INV-8). Resolves a CapabilityRequest into a
concrete ModelBinding given live constraints, or returns an Err *with the reason
each candidate was rejected* - a NO_CAPABLE_MODEL that tells you why beats one
that doesn't (the whole point of not being betrayed later).

The one hard structural rule enforced here is STR-6: locked-identity work plus a
real-time latency budget is a forbidden combination, because causal attention
degrades reference fidelity. The router refuses it rather than silently shipping
a drifting face at low latency.

REFUSE, DON'T ROUTE (2026-07-27). Structural routability is NOT the same question
as "can this fleet finish the job". The measured map (CAPABILITY-VIABILITY-MAP.md,
MODEL-POOL-INVENTORY.md) found 25 of 59 capability routes DEAD: they resolve
cleanly here, get admitted, queued and STARTED, and only then die in a runner --
because the bound model has zero bytes on the shared store, or its runner module
exists but returns Err on every path. A surface that always fails is worse than no
surface, so this module now decides that BEFORE a binding is handed out, and
refuses with text that names WHY and WHAT IS AVAILABLE INSTEAD.

Three new gates, in the order they fire:

  * CAPABILITY (``presets.capability_verdict``) -- the requested Capability is
    covered by NO render preset, i.e. no proven (model, runner, geometry) tuple on
    this fleet serves it. Refused before any candidate is even enumerated, so it
    can never be absorbed by a synthetic last-resort binding. This is what stops
    ``keyframe`` from quietly returning a plain i2v clip: nothing in the tree reads
    an end frame, so a keyframe request today is a RENAME of i2v, not a capability
    (CAPABILITY-VIABILITY-MAP.md, keyframe entry -- the same failure shape as the
    id_lock-drops-the-identity bug fixed the same day). Note the gate has NO list of
    its own: it asks the preset table, so when that table was corrected the same day
    (``clip-control-480p`` split -- motion kept as a real branch,
    inpaint/outpaint/retake demoted to refusals because no mask / canvas / frame-range
    input exists in this spine) this router started refusing three more capabilities
    without a line changing here. That is the point of deriving instead of listing.

  ⚠ THIS IS THE SECOND GATE, NOT THE FIRST. ``video_routes.video_studio_i2v`` runs
    the SAME ``capability_verdict`` at the HTTP boundary (2026-07-27), so a dead
    capability 400s before a job_id is minted. This one still matters and is not
    redundant: the router is what every NON-HTTP caller reaches -- the movie composer
    resolving a segment, a bus rehydrate, a test -- and a refusal that only lived in
    Flask would leave all of them routing into a runner three layers down.
  * STUB RUNNER (``presets.STUB_RUNNER_MODULES``) -- the RunnerSpec entrypoint
    resolves to a module whose every return is an ``Err``. The k1 gate
    (``registry.runner_available``) cannot see this: it asks ``find_spec``, so a
    stub module PASSES it, and ``_score``'s ``real_first`` then ranks the stub
    ABOVE the working ffmpeg last-resort. That single mechanism killed UPRES and
    INTERP on ae, the only render box: measured 2026-07-27, upres bound
    ltxv-spatial-upscaler at every budget >= 8 GB and interp bound rife-practical
    at every budget >= 3 GB, while ae's autofit budget is a stable ~21.6 GB.
  * WEIGHTS ABSENT (``presets.ZERO_BYTE_MODELS``) -- the model is registry INTENT
    with zero bytes on the shared store (the store holds exactly the 6 Wan-AI dirs
    plus Lightricks/ltxv-spatial-upscaler-0.9.7).

Both model-level gates REJECT A CANDIDATE, they do not abort the resolve: the loop
carries on and a working alternative (the ffmpeg enhancers) binds instead. Only
when NOTHING survives does the router refuse -- and then it says what it can do.

ERROR CODES, deliberately. A stub runner reports ``RUNNER_MISSING``, never
``WEIGHTS_MISSING``. That exact miscoding is a live incident: ``ltx_upscale.py``
returns ``WEIGHTS_MISSING`` while its 3.1 GB of weights ARE on disk and it found
them, which made a wiring gap look like a download problem for weeks. "The runner
is missing" is the honest reading of a module that can never return Ok -- there is
no runner, only a file where one should be.

RETRYABILITY. ``StageError`` carries no ``retryable`` field (the bus classifies at
the boundary: ``runners/studio_i2v._RETRYABLE_CODES`` = oom / nan_in_vae /
assembly_failed / io_error). Every code this router emits is outside that set, so
a refusal is already not-retryable -- and each Err now says so IN the context, so a
reader of the data alone can tell a deterministic refusal from a crash.
"""

from __future__ import annotations

from hugpy_video.intel.studio.enums import (
    Capability,
    LICENSE_PREFERENCE,
    PathClass,
    Precision,
    PRECISION_QUALITY,
)
from hugpy_video.intel.studio.errors import Err, ErrorCode, Ok, Result, StageError
from hugpy_video.intel.studio.presets import (
    STUB_RUNNER_MODULES,
    ZERO_BYTE_MODELS,
    available_menu,
    capability_verdict,
    presets_for,
)
from hugpy_video.intel.studio.registry import (
    CAPABILITY_TASKS,
    MODEL_REGISTRY,
    runner_available,
    runner_for,
    runner_gate_reason,
)
from hugpy_video.intel.studio.schemas import CapabilityRequest, ModelBinding, ModelConfig

# A refusal is a DECISION, not a fault. Stamped into every Err this module returns
# so the not-retryable classification is legible in the error VALUE and not only in
# the bus's lookup table (which lives in another package and can drift from it).
_NOT_RETRYABLE: tuple[str, str] = ("retryable", "false")

# Stable markers inside a rejection reason, so the sharpening ladder below can key
# on the CLASS of rejection without re-deriving it. Substrings, not prefixes: the
# reasons are model-qualified ("<model_id>: <reason>") for the caller's benefit.
_STUB_RUNNER_MARK = "runner is a stub"
_WEIGHTS_ABSENT_MARK = "weights absent"


def _pick_precision(
    cfg: ModelConfig, budget_gb: float, min_precision: Precision
) -> Precision | None:
    """Highest-quality precision whose VRAM cost fits the budget AND meets the
    chosen runner's quality floor (FIX-4). A precision below ``min_precision``
    (e.g. INT8 under an FP8 floor) is never a valid selection even if it fits the
    VRAM budget: it would silently ship below the runner's supported quality."""
    floor = PRECISION_QUALITY[min_precision]
    fitting = [p for p in cfg.vram.fits(budget_gb) if PRECISION_QUALITY[p] >= floor]
    if not fitting:
        return None
    return max(fitting, key=lambda p: PRECISION_QUALITY[p])


def _pick_task(cfg: ModelConfig, capability: Capability):
    """First task that both satisfies the capability and has a SERVABLE runner
    (k1: gated on import-resolvability, not just registry membership — a seed
    entry whose runner module hasn't landed yet must never look routable here;
    see ``registry.runner_available``)."""
    for task in CAPABILITY_TASKS.get(capability, ()):  # ordered by preference
        if task in cfg.tasks and runner_available(cfg.family, task) is not None:
            return task
    return None


def _no_task_reason(cfg: ModelConfig, capability: Capability) -> str:
    """Why ``_pick_task`` found nothing for ``cfg``/``capability`` — an honest
    rejection reason, not a generic dead end. Distinguishes "this model declares
    a satisfying task but its runner is GATED (module missing)" from "this model
    declares no satisfying task at all" (a genuine capability mismatch)."""
    for task in CAPABILITY_TASKS.get(capability, ()):
        if task in cfg.tasks:
            reason = runner_gate_reason(cfg.family, task)
            if reason is not None:
                return f"{task.value} runner gated ({reason})"
    return "no runnable task for capability"


def _entrypoint_module(entrypoint: str) -> str:
    """The dotted MODULE path of a ``RunnerSpec.entrypoint`` ("mod.path:callable"
    -> "mod.path"). Same rule as ``registry._entrypoint_module``, restated here
    rather than imported so this module does not reach across for a private."""
    return entrypoint.split(":", 1)[0]


def _viability_reason(cfg: ModelConfig, task, spec) -> str | None:
    """Why this (model, task, runner) triple CANNOT COMPLETE on this fleet, or
    ``None`` when nothing disqualifies it.

    The k1 gate already pruned runners whose module is absent. This is the second
    half of the same question, and it is the half that was missing: a module that
    EXISTS but can never return Ok, and a registry row with no bytes behind it.
    Both were reachable — measured on the live fleet 2026-07-27 — and both only
    announced themselves after a job had been admitted, queued and started.

    ORDER IS DELIBERATE. rife-practical is BOTH a stub runner AND zero bytes, and
    the stub is the binding problem: with the module gone the router would fall
    through to a working ffmpeg row on its own, whereas staging the weights would
    change nothing at all. Reporting the stub first therefore points at the fix.

    Reads the two frozen sets in ``presets`` rather than re-deriving them, so this
    router and ``tests/studio/test_presets.py`` (which proves those sets against
    the real tree and the real store) can never disagree."""
    module = _entrypoint_module(spec.entrypoint)
    if module in STUB_RUNNER_MODULES:
        # NOT WEIGHTS_MISSING — see the module docstring. ltxv-spatial-upscaler's
        # 3.1 GB are present and the runner FOUND them before returning Err.
        return f"{task.value} runner is a stub (every path returns Err: {module})"
    if cfg.model_id in ZERO_BYTE_MODELS:
        return ("weights absent (0 bytes on the shared store) — registry intent, "
                "nothing to load")
    return None


def _alternatives(capability: Capability) -> str:
    """The "...and here is what you CAN have" half of a refusal: the presets covering
    ``capability`` (id, title and the geometry this fleet has actually rendered at),
    or — when no preset covers it — the whole fleet menu, which is the honest answer
    to "then what CAN I ask for". Text only; the structured form rides in the Err
    context so a console does not have to parse prose."""
    covered = presets_for(capability)
    if not covered:
        return available_menu()
    return "; ".join(
        f"{p.preset_id} ({p.title}, {p.geometry}"
        + (f", up to {p.max_frames} frames" if p.max_frames else "")
        + ")"
        for p in covered
    )


def capable_model_ids(capability: Capability,
                      include_synthetic: bool = False) -> tuple[str, ...]:
    """Every model that DECLARES ``capability`` and could actually run it on this
    fleet — the "does a capable model EXIST at all" question, answered WITHOUT a
    geometry or a VRAM budget (k58).

    Same gates ``resolve`` applies per candidate, minus the request-shaped ones
    (resolution / frames / license / precision): the model must reach the capability
    through a task whose runner is SERVABLE (k1) and must carry no viability blocker
    (stub runner / zero bytes on the store). Deriving it from the same helpers is the
    point — a preflight that answered from its own list would drift from the router
    the moment the zoo moved.

    SYNTHETIC IS EXCLUDED BY DEFAULT. The last-resort synthetic tier is opt-in at
    render time (``render_clip``), so its presence is NOT evidence that a capability
    is served — a preflight that counted it would pass a movie whose only possible
    output is a noise blob."""
    out: list[str] = []
    for cfg in MODEL_REGISTRY.values():
        if capability not in cfg.capabilities:
            continue
        if cfg.synthetic and not include_synthetic:
            continue
        task = _pick_task(cfg, capability)
        if task is None:
            continue
        spec = runner_for(cfg.family, task)
        if spec is None or _viability_reason(cfg, task, spec) is not None:
            continue
        out.append(cfg.model_id)
    return tuple(sorted(out))


def _refusal(code: ErrorCode, capability: Capability, why: str,
             extra: tuple[tuple[str, str], ...] = ()) -> Err[StageError]:
    """A refusal, as data: WHY it cannot complete plus WHAT IS AVAILABLE instead,
    in one message and one structured context. Every caller-facing dead end in
    ``resolve`` goes through here so none of them can regress to a bare
    "no capable model" — the message that taught a caller nothing and sent them
    looking for a download that would not have helped."""
    available = _alternatives(capability)
    return Err(StageError(
        code,
        f"studio cannot serve {capability.value!r} as requested: {why}. "
        f"What IS available today: {available}.",
        (("capability", capability.value),
         ("available", available)) + extra + (_NOT_RETRYABLE,),
    ))


class CapabilityRouter:
    def resolve(self, req: CapabilityRequest) -> Result[ModelBinding, StageError]:
        # STR-6: refuse locked-identity under a real-time latency budget.
        streaming_required = req.latency_budget_ms is not None
        if streaming_required and req.capability in (Capability.ID_LOCK, Capability.KEYFRAME):
            return Err(StageError(
                ErrorCode.CAPABILITY_STREAMING_CONFLICT,
                "locked-identity work cannot run under a real-time latency budget; "
                "causal attention degrades reference fidelity (STR-6). Route this "
                "shot offline, or drop the latency budget.",
                (("capability", req.capability.value),
                 ("latency_budget_ms", str(req.latency_budget_ms)),
                 _NOT_RETRYABLE),
            ))

        # CAPABILITY GATE — refuse a capability no preset covers, BEFORE anything
        # else looks at it. Deliberately ahead of the candidate scan (and ahead of
        # the pin handling): if no proven tuple on this fleet serves the capability
        # at all, the answer cannot depend on which model was asked for, and the
        # refusal must not be reachable by a synthetic last-resort survivor. The
        # verdict's own text already names the MEASURED blocker (contract gap vs
        # download vs card) and the fleet menu, so it is passed through verbatim
        # rather than paraphrased here — one wording, shared by route and console.
        verdict = capability_verdict(req.capability)
        if not verdict.servable:
            return Err(StageError(
                ErrorCode.NO_CAPABLE_MODEL,
                verdict.refusal,
                (("capability", req.capability.value),
                 ("reason", verdict.reason),
                 ("available", available_menu()),
                 _NOT_RETRYABLE),
            ))

        candidates = [m for m in MODEL_REGISTRY.values()
                      if req.capability in m.capabilities]
        if not candidates:
            # Reachable for an ORCHESTRATION capability (ASSEMBLE): a preset covers
            # it, but it is composed by the movie runner rather than bound to a
            # model — so the useful answer is the preset that serves it, not a bare
            # "no model declares it".
            return _refusal(
                ErrorCode.NO_CAPABLE_MODEL, req.capability,
                "no model declares it (it is an orchestration stage, not a model "
                "binding)")

        # DIRECT MODEL CHOICE (pin): the caller asked for a SPECIFIC model_id. Restrict
        # the candidate set to exactly that model — it still runs the full gate ladder
        # below (resolution / license / VRAM), so a pin that "doesn't fit" surfaces the
        # normal sharpened reason. Two pin-specific failures are reported UP FRONT as
        # clear data (never a silent fallback to a different model): the model_id is
        # unknown to the registry, or it exists but does not declare this capability.
        if req.pinned_model_id is not None:
            pinned = MODEL_REGISTRY.get(req.pinned_model_id)
            if pinned is None:
                return Err(StageError(
                    ErrorCode.PINNED_MODEL_UNAVAILABLE,
                    f"pinned model_id {req.pinned_model_id!r} is not in the studio "
                    f"registry",
                    (("pinned_model_id", req.pinned_model_id),
                     ("capability", req.capability.value),
                     ("available", _alternatives(req.capability)),
                     _NOT_RETRYABLE),
                ))
            if req.capability not in pinned.capabilities:
                return Err(StageError(
                    ErrorCode.PINNED_MODEL_UNAVAILABLE,
                    f"pinned model {req.pinned_model_id!r} does not serve capability "
                    f"{req.capability.value!r}",
                    (("pinned_model_id", req.pinned_model_id),
                     ("capability", req.capability.value),
                     ("model_capabilities",
                      ",".join(sorted(c.value for c in pinned.capabilities))),
                     ("available", _alternatives(req.capability)),
                     _NOT_RETRYABLE),
                ))
            candidates = [pinned]

        rejected: list[str] = []
        survivors: list[tuple[ModelConfig, object, Precision]] = []

        for cfg in candidates:
            # path class must match the streaming requirement
            if streaming_required and cfg.path_class != PathClass.STREAMING:
                rejected.append(f"{cfg.model_id}: not a streaming model")
                continue

            # license / commercial gating
            if req.commercial_use:
                # Additive rule: a model routes commercially if it is auto-commercial
                # OR the caller asserts they hold the agreement (license in
                # allowed_licenses). commercial_auto must NOT be nullified.
                allowed = cfg.commercial_auto or cfg.license in req.allowed_licenses
                if not allowed:
                    rejected.append(
                        f"{cfg.model_id}: license {cfg.license.value} not "
                        f"auto-commercial and not in allowed_licenses")
                    continue
            elif req.allowed_licenses and cfg.license not in req.allowed_licenses:
                # FIX-2: the strict whitelist only applies on the non-commercial
                # path. Running it unconditionally turned allowed_licenses into a
                # hard whitelist that nullified commercial_auto above.
                rejected.append(
                    f"{cfg.model_id}: license {cfg.license.value} not in allowed set")
                continue

            # native audio requirement
            if req.require_native_audio and not cfg.native_audio:
                rejected.append(f"{cfg.model_id}: no native audio")
                continue

            # resolution
            if not cfg.supports_resolution(req.target_resolution):
                rejected.append(
                    f"{cfg.model_id}: max res < "
                    f"{req.target_resolution.width}x{req.target_resolution.height}")
                continue

            # frame budget
            if req.min_frames and cfg.max_frames < req.min_frames:
                rejected.append(
                    f"{cfg.model_id}: max_frames {cfg.max_frames} < {req.min_frames}")
                continue

            # capability -> task -> runner (k1: honest reason when the model
            # declares a satisfying task but its runner is gated, not a generic
            # dead end — so a direct/pinned dispatch attempt at a dead engine
            # names the missing runner instead of surfacing an ImportError).
            task = _pick_task(cfg, req.capability)
            if task is None:
                rejected.append(f"{cfg.model_id}: {_no_task_reason(cfg, req.capability)}")
                continue

            spec = runner_for(cfg.family, task)

            # VIABILITY (2026-07-27): can this triple actually FINISH? Placed here
            # on purpose — AFTER the k1 task gate, so a model whose runner module is
            # simply absent still reports the sharper "runner gated (runner_missing:
            # <module>)"; and BEFORE the VRAM arithmetic, because a stub runner and a
            # zero-byte row are disqualified at every budget and precision, and a
            # "min 3GB > budget 2GB" reason would send the caller to raise a budget
            # that was never the problem.
            blocker = _viability_reason(cfg, task, spec)
            if blocker is not None:
                rejected.append(f"{cfg.model_id}: {blocker}")
                continue

            # precision / VRAM fit — bounded below by the chosen runner's floor (FIX-4)
            precision = _pick_precision(cfg, req.vram_budget_gb, spec.min_precision)
            if precision is None:
                if not cfg.vram.fits(req.vram_budget_gb):
                    rejected.append(
                        f"{cfg.model_id}: min {cfg.vram.min_gb():.0f}GB > "
                        f"budget {req.vram_budget_gb:.0f}GB")
                else:
                    # Fits the budget, but only below the runner's precision floor.
                    rejected.append(
                        f"{cfg.model_id}: no precision >= runner floor "
                        f"{spec.min_precision.value} fits budget "
                        f"{req.vram_budget_gb:.0f}GB")
                continue

            survivors.append((cfg, task, precision))

        if not survivors:
            code = ErrorCode.NO_CAPABLE_MODEL
            why = "no model satisfied it under the given constraints"
            # Sharpen the code when the rejection was unanimous on one axis. The two
            # viability axes are checked FIRST because they are ROOT causes: a stub
            # runner or an empty registry row is why the request cannot complete,
            # whatever the budget said. Unanimity only — a MIXED set (say, one model
            # out of geometry and another runner-gated) stays NO_CAPABLE_MODEL, which
            # is exactly what it is, and keeps today's codes for the mixed cases.
            # ``rejected`` is never empty here (an empty candidate set returned
            # above), but the guard is explicit: a bare all() over nothing is True
            # and would sharpen an unknown failure into a confident wrong reason.
            def unanimous(mark: str) -> bool:
                return bool(rejected) and all(mark in r for r in rejected)

            if unanimous(_STUB_RUNNER_MARK):
                code = ErrorCode.RUNNER_MISSING
                why = ("its runner module exists but returns Err on every path — "
                       "there is no working runner, and no download fixes that")
            elif unanimous(_WEIGHTS_ABSENT_MARK):
                code = ErrorCode.WEIGHTS_MISSING
                why = ("every model that declares it has zero bytes on the shared "
                       "store")
            elif unanimous("GB"):
                code = ErrorCode.VRAM_EXCEEDED
                why = (f"no model fits the {req.vram_budget_gb:.1f}GB VRAM budget at "
                       f"a precision its runner supports")
            elif unanimous("license"):
                code = ErrorCode.LICENSE_VIOLATION
                why = "no model's license permits this request"
            elif unanimous("max res"):
                code = ErrorCode.RESOLUTION_UNSUPPORTED
                why = (f"no model reaches "
                       f"{req.target_resolution.width}x{req.target_resolution.height}")
            return _refusal(
                code, req.capability, why,
                tuple(("rejected", r) for r in rejected))

        cfg, task, precision = max(survivors, key=lambda s: self._score(req, *s))
        return Ok(ModelBinding(
            model_id=cfg.model_id,
            framework=cfg.family,
            task=task,             # type: ignore[arg-type]
            precision=precision,
            path_class=cfg.path_class,
            weight_uri=cfg.weight_uri,
            weight_hash=cfg.weight_hash,
            determinism_class=cfg.default_determinism,   # FIX-3: propagate the class
        ))

    @staticmethod
    def _score(req: CapabilityRequest, cfg: ModelConfig, task, precision: Precision):
        # LAST-RESORT rule: the placeholder synthetic model must NEVER shadow a real
        # generative model. This is the TOP-priority score dimension, so any real
        # survivor (real_first=1) strictly outranks any synthetic one (0) whatever
        # the lower dimensions say. Synthetic wins ONLY when it is the sole survivor
        # (no real model fit — e.g. a sub-GB VRAM budget), which preserves the tiny
        # demo path without letting it steal a genuine binding.
        real_first = 0 if cfg.synthetic else 1
        offline_pref = 0 if req.latency_budget_ms is not None else (
            1 if cfg.path_class == PathClass.OFFLINE else 0)
        framework_pref = 1 if (req.preferred_framework and
                               cfg.family == req.preferred_framework) else 0
        return (
            real_first,
            offline_pref,
            framework_pref,
            LICENSE_PREFERENCE[cfg.license],
            PRECISION_QUALITY[precision],
            cfg.best_native_area(),
            -cfg.vram.min_gb(),   # tie-break: prefer the tighter footprint to pack
        )
