"""``(oracle, performance)`` bus runner (k106) — a THIN relay onto the oracle's
``video.performance`` FAT orchestrator.

WHAT THIS IS. The media-bus entry point for "two characters, three lines, a
15-30 second result". It owns no pipeline logic whatsoever: it rehydrates a
JSON-safe :class:`PerformanceSpec` into an
``oracle.performance.PerformanceGoal``, calls
``oracle.performance.run_performance`` with ``default_seams()``, and turns the
typed :class:`~hugpy_oracle.performance.PerformanceResult` into a
``JobResult``. Every ordering rule, every gate, every scorecard and every
digest lives in ``oracle/performance.py`` — deliberately, so the recipe is
testable without a bus, a job id, a database or a worker, and so there is
exactly ONE implementation of it.

That is the same shape ``identity_render_relay`` and ``identity_from_video``
have: the runner is the bus's socket, not the work.

IMPORT DISCIPLINE, AS STRICT AS ``tts_chatterbox``. This module's top level is
STDLIB ONLY. ``runners/__init__`` imports it at app boot, and importing
``hugpy_oracle`` builds the model registry (~1s plus log chatter),
so the oracle import is LAZY inside the two functions that need it. ``probe()``
therefore costs a ``find_spec`` and nothing else, exactly like the TTS adapter
the oracle catalog probes on every ``GET /oracle/capabilities``.

WHAT IT WILL AND WILL NOT DO ON THIS FLEET TODAY. ``default_seams()`` binds
transcription, image generation, the vision judge and ffmpeg assembly. It does
NOT bind speech synthesis (no worker seats chatterbox), speaker similarity (no
embedding backend), clip generation or clip judging (video executes through the
studio job pipeline on a GPU worker, and there is no clip evaluator). So a LIVE
job refuses at stage 3 with a typed ``CAPABILITY_GAP`` naming ``audio.tts`` and
the operator step that would seat it. That refusal IS the deliverable until a
worker seats the TTS runner — this module never fabricates audio, a still, a
clip or a scorecard to make a job look finished.

Pure ``(PerformanceSpec, job_id) -> JobResult`` (map §6): every EXPECTED failure
— an unconstructible spec, a missing authority grant, an unseated capability, a
failing shot — crosses the boundary as DATA (``JobResult(ok=False,
JobError(...))``), never as a raise. Only a genuine programmer error raises, and
``media_bus.run_claimed`` is the one place that catches that.

REGISTRATION.
  * ``runners/__init__.py``: ``from .performance_relay import
    run_video_performance`` and ``("oracle", "performance"):
    run_video_performance`` in ``DISPATCH``.
  * ``video_intel/job_schema.py``: ``"video_performance": JobSpec(
    "video_performance", PerformanceSpec, ("oracle", "performance"), "gpu",
    14400)`` — the movie timeout, because a performance is a movie plus audio.
Both files were another agent's dirty working copy when k106 landed; see the
k106 dispatch record for exactly which of the two edits shipped and which the
operator still has to make.

No pathlib anywhere. os.path only.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

#: The bus dispatch key and job name — the two registration one-liners above.
RUNNER_KEY: Tuple[str, str] = ("oracle", "performance")
JOB_NAME: str = "video_performance"
#: The bus envelope: the movie queue and the movie timeout, because a
#: performance is a movie plus audio (registered through ``hugpy_video.jobs``).
JOB_QUEUE: str = "gpu"
JOB_TIMEOUT_S: int = 14400

#: The oracle module that owns the recipe. Named as a string so ``probe()``
#: can ask whether it imports without importing it.
ORACLE_MODULE: str = "hugpy_oracle.performance"

#: Repair codes whose honest answer is "the fleet was busy/slow, ask again";
#: everything else is a real finding and must not be retried blind.
RETRYABLE_CODES: frozenset = frozenset({"timeout", "worker_unavailable"})


class PerformanceSpecError(ValueError):
    """A structurally invalid spec — construction-time, local to this module,
    never crosses the bus boundary (the runner converts it to a JobError)."""


@dataclass(frozen=True)
class PerformanceSpec:
    """Frozen, JSON-safe currency of a ``video_performance`` bus job.

    Every field is a primitive, a list or a plain dict, so ``asdict`` ->
    ``json`` round-trips cleanly and ``media_jobs.db`` can store it. The two
    heavyweight members are carried as the oracle contracts' OWN ``to_dict``
    shapes (``GoalSpec.to_dict()`` and ``DialogueTimeline.to_dict()``), so this
    schema never becomes a second, drifting definition of them.

        ``goal``            ``GoalSpec.to_dict()`` — objective, raw prompt,
                            quality, budget and the RightsManifest the k97 gate
                            reads. THE RIGHTS LIVE HERE: a job whose manifest
                            does not cover the named identities is refused at
                            stage 1, before a model is picked.
        ``dialogue``        ``DialogueTimeline.to_dict()`` — the locked lines.
        ``casting``         ``[[speaker, VoiceProfile.to_dict()], …]``.
        ``raw_request_ref`` invariant 1: the operator's raw request, by
                            reference.
        ``resume``          a previous run id to continue; the orchestrator
                            re-verifies every recorded artifact by digest and
                            re-runs anything that does not check out.
        ``stop_after``      run to that stage and persist the journal (a dry
                            run an operator can inspect before spending GPU).
    """

    goal: dict
    dialogue: dict
    casting: list
    raw_request_ref: str
    identity_refs: list = field(default_factory=list)
    voice_refs: list = field(default_factory=list)
    deliverable: str = ""
    prompts_before_run: list = field(default_factory=list)
    exclusions: list = field(default_factory=list)
    tone: float = 0.0
    pause_after_s: float = 0.35
    pad_s: float = 0.35
    min_shot_s: float = 1.0
    max_shot_s: float = 8.0
    rubric: list = field(default_factory=list)
    negative_prompt: Optional[str] = None
    seed_salt: int = 0
    tts_candidates: int = 3
    keyframe_candidates: int = 3
    clip_candidates: int = 3
    max_seconds: Optional[float] = None
    resume: Optional[str] = None
    stop_after: Optional[str] = None
    created_at: str = ""


def make_performance(goal: Mapping[str, Any], dialogue: Mapping[str, Any],
                     casting: Any, raw_request_ref: str,
                     **kw: Any) -> PerformanceSpec:
    """Validate + build a ``PerformanceSpec``. Raises are fine here: this is
    construction-time and local, exactly like ``make_mlt_render``."""
    if not isinstance(goal, Mapping) or not goal.get("raw_prompt"):
        raise PerformanceSpecError(
            "performance spec needs a goal dict carrying at least "
            "objective/raw_prompt (GoalSpec.to_dict())")
    if not isinstance(dialogue, Mapping) or not dialogue.get("lines"):
        raise PerformanceSpecError(
            "performance spec needs a dialogue dict with at least one line "
            "(DialogueTimeline.to_dict())")
    pairs = [[str(speaker), dict(profile)]
             for speaker, profile in _casting_pairs(casting)]
    if not pairs:
        raise PerformanceSpecError(
            "performance spec needs a casting table: every speaker in the "
            "dialogue must be cast to a voice, or no line can be spoken")
    if not str(raw_request_ref).strip():
        raise PerformanceSpecError(
            "performance spec needs raw_request_ref (invariant 1: the "
            "operator's raw request rides along by reference)")
    for name in ("tts_candidates", "keyframe_candidates", "clip_candidates"):
        value = kw.get(name)
        if value is not None and int(value) < 1:
            raise PerformanceSpecError(
                f"{name} must be >= 1; a fan-out of zero candidates produces "
                f"nothing to judge")
    return PerformanceSpec(goal=dict(goal), dialogue=dict(dialogue),
                           casting=pairs,
                           raw_request_ref=str(raw_request_ref), **kw)


def performance_from_dict(payload: Mapping[str, Any]) -> PerformanceSpec:
    """Rehydrate + RE-VALIDATE a spec the bus stored. House pattern: the
    round trip runs the factory again rather than trusting the row."""
    data = dict(payload or {})
    goal = data.pop("goal", {})
    dialogue = data.pop("dialogue", {})
    casting = data.pop("casting", [])
    ref = data.pop("raw_request_ref", "")
    known = {f for f in PerformanceSpec.__dataclass_fields__}
    unknown = sorted(set(data) - known)
    if unknown:
        raise PerformanceSpecError(
            f"performance spec carries unknown key(s) {unknown}; a key nobody "
            f"reads is a silently dropped instruction")
    return make_performance(goal, dialogue, casting, ref, **data)


def validate_performance_spec(payload: Mapping[str, Any]) -> PerformanceSpec:
    """Validate both the bus envelope and its nested oracle contracts before enqueue.

    The bus deserializer checks the envelope only. A malformed dialogue, voice or
    goal would otherwise occupy a queued GPU job before failing in the runner.
    """
    spec = performance_from_dict(payload)
    from hugpy_oracle.performance import STAGES
    if spec.stop_after is not None and spec.stop_after not in STAGES:
        raise PerformanceSpecError(f"stop_after must be one of {list(STAGES)}")
    _rehydrate(spec)
    return spec


def _casting_pairs(casting: Any):
    if isinstance(casting, Mapping):
        return list(casting.items())
    out = []
    for item in casting or ():
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((item[0], item[1]))
        elif isinstance(item, Mapping) and item.get("voice_id"):
            out.append((item["voice_id"], item))
        else:
            raise PerformanceSpecError(
                f"casting takes a mapping, [speaker, profile] pairs, or voice "
                f"profile dicts; got {item!r}")
    return out


# --------------------------------------------------------------------------- #
# registration probe
# --------------------------------------------------------------------------- #
def probe() -> dict:
    """Doc §4 step-3 adapter registration + health probe for
    ``video.performance``.

    Reports THREE facts and never conflates them, the same discipline
    ``tts_chatterbox.probe`` uses:

      ``runner_registered`` this adapter module imports (always True by the
                            time anyone can call this);
      ``importable``        the ORCHESTRATOR imports here (it is pure Python +
                            the oracle contracts, so this is about the tree,
                            not about hardware);
      ``ready``             every seam the complete recipe needs is BOUND on
                            this box. Staged runs may still be possible when
                            a later visual seam is unavailable.

    ``unbound`` carries the operator step for each missing seam, so
    ``GET /oracle/capabilities`` (or an operator reading the probe directly)
    learns what to go do rather than that something is merely unavailable.

    Cheap by construction: it does a ``find_spec``, and only imports the
    orchestrator when that succeeds."""
    import importlib.util

    if importlib.util.find_spec(ORACLE_MODULE) is None:
        return {"importable": False, "runner_registered": True, "ready": False,
                "module": ORACLE_MODULE, "runner_key": RUNNER_KEY,
                "job_name": JOB_NAME,
                "reason": f"{ORACLE_MODULE!r} is not on the import path"}
    try:
        import importlib
        perf = importlib.import_module(ORACLE_MODULE)
        seams = perf.default_seams()
        bound = [name for name in perf.SEAM_NAMES if seams.bound(name)]
        unbound = [seams.gap_for(name).to_dict()
                   for name in perf.SEAM_NAMES if not seams.bound(name)]
    except Exception as exc:  # noqa: BLE001 — an unimportable orchestrator IS the finding
        return {"importable": False, "runner_registered": True, "ready": False,
                "module": ORACLE_MODULE, "runner_key": RUNNER_KEY,
                "job_name": JOB_NAME,
                "reason": (f"{ORACLE_MODULE} did not import "
                           f"({type(exc).__name__}: {exc})")}

    required = ("synth", "transcribe", "gen_image", "judge_image",
                "gen_clip", "judge_clip", "concat")
    missing = [name for name in required if name not in bound]
    return {
        "importable": True,
        "runner_registered": True,
        "ready": not missing,
        "module": ORACLE_MODULE,
        "runner_key": RUNNER_KEY,
        "job_name": JOB_NAME,
        "stages": list(perf.STAGES),
        "bound": bound,
        "unbound": unbound,
        "reason": ("" if not missing else
                   "seam(s) " + ", ".join(missing) + " are unbound on this "
                   "box; a job would refuse at the first stage that needs one "
                   "(see 'unbound' for the operator step)"),
    }


# --------------------------------------------------------------------------- #
# the runner
# --------------------------------------------------------------------------- #
def run_video_performance(spec: PerformanceSpec, job_id: str):
    """``(PerformanceSpec, job_id) -> JobResult``. All work is the oracle's."""
    from hugpy_video.intel.result_schema import JobError, JobResult

    def _fail(code: str, message: str, retryable: bool = False,
              manifest: Optional[dict] = None):
        return JobResult(job_id=job_id, ok=False,
                         error=JobError(code=code, message=message,
                                        retryable=retryable),
                         movie=manifest)

    try:
        goal, seams, budget = _rehydrate(spec)
    except Exception as exc:  # noqa: BLE001 — a bad spec is DATA at this boundary
        return _fail("bad_spec",
                     f"the performance request is not constructible "
                     f"({type(exc).__name__}: {exc})")

    _progress(job_id, {"stage": "starting", "run_id": None})

    import importlib
    perf = importlib.import_module(ORACLE_MODULE)
    visual = None
    if _dag_recipe_enabled() and not spec.stop_after:
        # Unit A/K: the visual stages run on the durable DAG (resume after a
        # kill, per-candidate routing, targeted repair, steward report). The
        # linear recipe remains the authority up to the production lock.
        visual = _run_on_dag(perf, goal, seams, budget, spec, job_id)
        result = visual["prep"] if visual is not None else None
        if visual is not None and visual["visual"] is not None:
            result = _fold_visual(perf, visual["prep"], visual["visual"])
    if visual is None:
        result = perf.run_performance(goal, seams=seams, budget=budget,
                                      resume=spec.resume,
                                      stop_after=spec.stop_after)

    manifest = result.to_dict()
    if visual is not None and visual["visual"] is not None:
        manifest["dag"] = visual["visual"].to_dict()
    _progress(job_id, {"stage": "done", "run_id": result.run_id,
                       "stages": [s.to_dict() for s in result.stages]})

    if not result.ok:
        if result.stopped_after:
            # A deliberate checkpoint is not a failure and must not be dressed
            # as one; it is also not a deliverable, so it is not ok either.
            return _fail("stopped_after_stage",
                         f"run {result.run_id} stopped after stage "
                         f"{result.stopped_after!r} as requested; resume with "
                         f"resume={result.run_id!r}",
                         retryable=True, manifest=manifest)
        gap = result.gap
        code = (gap.primary_code.value if gap and gap.primary_code
                else "performance_incomplete")
        message = (f"stage {gap.stage}: {gap.diagnosis}" if gap else
                   "the performance produced no accepted deliverable")
        if gap and gap.requirement:
            message = f"{message} — required: {gap.requirement}"
        return _fail(code, message, retryable=code in RETRYABLE_CODES,
                     manifest=manifest)

    outputs = _outputs(result)
    return JobResult(job_id=job_id, ok=True, outputs=outputs,
                     project={"name": result.run_id, "uuid": job_id,
                              "dir": perf.run_dir(result.run_id)},
                     movie=manifest)


def _progress(job_id: str, blob: dict) -> None:
    """Mirror a coarse stage into the bus's progress blob. The IMPORT is inside
    the try as well as the call: this runner must stay usable (and testable)
    without a media_jobs.db, and a progress hiccup has never been allowed to
    fail a generation in this tree."""
    try:
        from hugpy_video.intel.media_bus import set_progress
        set_progress(job_id, blob)
    except Exception:                              # noqa: BLE001
        logger.debug("performance %s: set_progress failed (non-fatal)", job_id,
                     exc_info=True)


def _rehydrate(spec: PerformanceSpec):
    """Spec -> ``(PerformanceGoal, PerformanceSeams, PerformanceBudget)``.
    Raises; the runner converts that into a ``bad_spec`` JobError."""
    import importlib

    perf = importlib.import_module(ORACLE_MODULE)
    audio_master = importlib.import_module(
        "hugpy_oracle.audio_master")
    contracts = importlib.import_module("hugpy_oracle.contracts")

    goal_spec = contracts.GoalSpec.from_dict(spec.goal)
    dialogue = audio_master.DialogueTimeline.from_dict(spec.dialogue)
    casting = {str(speaker): audio_master.VoiceProfile.from_dict(profile)
               for speaker, profile in _casting_pairs(spec.casting)}
    policy = audio_master.SpeechPolicy(pause_after_s=float(spec.pause_after_s))

    goal = perf.PerformanceGoal(
        goal=goal_spec, dialogue=dialogue, casting=casting,
        raw_request_ref=spec.raw_request_ref,
        identity_refs=tuple(spec.identity_refs or ()),
        voice_refs=tuple(spec.voice_refs or ()),
        deliverable=spec.deliverable,
        prompts_before_run=tuple(spec.prompts_before_run or ()),
        exclusions=tuple(spec.exclusions or ()),
        speech_policy=policy, tone=float(spec.tone),
        rubric=tuple(spec.rubric) or perf.DEFAULT_RUBRIC,
        negative_prompt=spec.negative_prompt,
        min_shot_s=float(spec.min_shot_s), max_shot_s=float(spec.max_shot_s),
        pad_s=float(spec.pad_s), seed_salt=int(spec.seed_salt),
        created_at=spec.created_at)

    seams = perf.default_seams(
        tts_candidates=int(spec.tts_candidates),
        keyframe_candidates=int(spec.keyframe_candidates),
        clip_candidates=int(spec.clip_candidates))
    budget = perf.PerformanceBudget(
        max_seconds=(None if spec.max_seconds is None
                     else float(spec.max_seconds)))
    return goal, seams, budget


def _outputs(result):
    """The produced artifacts as ``MediaRef``s, ingested only when they are
    real files on this box. A seam that returned an opaque handle is not
    ingested and not pretended about — the manifest still names it."""
    from hugpy_video.intel.media_store import ingest

    refs = []
    for uri in (result.video_ref,) + tuple(result.artifact_refs):
        if not uri or not os.path.isabs(str(uri)) or not os.path.isfile(uri):
            continue
        try:
            media = ingest(uri)
        except Exception as exc:                   # noqa: BLE001
            logger.warning("performance: could not ingest %s (%s: %s)",
                           uri, type(exc).__name__, exc)
            continue
        if media not in refs:
            refs.append(media)
    return tuple(refs)


__all__ = [
    "JOB_NAME", "ORACLE_MODULE", "RETRYABLE_CODES", "RUNNER_KEY",
    "PerformanceSpec", "PerformanceSpecError", "make_performance",
    "performance_from_dict", "probe", "run_video_performance",
]


DAG_RECIPE_ENV = "ORACLE_DAG_RECIPE"


def _dag_recipe_enabled() -> bool:
    """Default ON. ``ORACLE_DAG_RECIPE=0`` falls back to the linear recipe
    (diagnostic only — the linear path has no resume-after-kill and no
    per-candidate routing)."""
    return os.environ.get(DAG_RECIPE_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def _run_on_dag(perf, goal, seams, budget, spec, job_id):
    """Drive ``recipes.video_performance.run_performance_on_dag`` with a
    journal beside the run. Returns None when the DAG path is unavailable so
    the caller falls back honestly (and says so in the manifest)."""
    try:
        from hugpy_oracle.recipes import video_performance as vp
        from hugpy_oracle.dag_runtime import RunJournal
        from hugpy_oracle import selection
    except Exception as exc:  # noqa: BLE001
        logger.warning("performance %s: DAG recipe unavailable (%s: %s) — linear path", job_id,
                       type(exc).__name__, exc)
        return None
    root = seams.run_root or perf.default_run_root()
    journal = RunJournal(os.path.join(root, "runs", "performance", "dag.sqlite"))
    try:
        _progress(job_id, {"stage": "dag", "run_id": spec.resume})
        prep, visual, _rt = vp.run_performance_on_dag(
            goal, seams=seams, journal=journal, run_id=spec.resume, budget=budget,
            selector=selection.process_selector())
        return {"prep": prep, "visual": visual}
    finally:
        journal.close()


def _fold_visual(perf, prep, visual):
    """Project the DAG's visual result back onto the linear PerformanceResult
    shape the bus, UI and tests already read (video_ref, shots, ok)."""
    from dataclasses import replace as _replace
    shots = tuple(visual.shots)
    limitations = tuple(prep.limitations) + (
        () if visual.ok else (f"visual DAG run {visual.run.run_id} ended {visual.run.state.value}; "
                              f"failed nodes: {list(visual.failed_nodes)}",))
    try:
        return _replace(prep, ok=bool(visual.ok), video_ref=visual.video_ref, stopped_after=None,
                        limitations=limitations)
    except Exception:  # noqa: BLE001 — PerformanceResult shape drift: keep prep, note it
        return prep
