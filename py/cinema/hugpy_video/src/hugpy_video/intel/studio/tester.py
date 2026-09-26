"""Studio TESTER — sweep ONE prompt across EVERY available model for a category.

The operator's "TESTER": take a prompt in any studio category and iterate it
across every servable model of that category's TYPE, recording a model-battery
run-dir (one row per model) so the console gallery shows the head-to-head.

Two category TYPES, discovered by reading the real generation paths — they do NOT
share a model registry, so enumeration + binding differ per type:

  * IMAGE  (categories: image, scene)  -> task ``text-to-image``.
      Enumerated from the MAIN registry via
      ``imports.config.models.models_default.get_models_dict_by_tasks`` (its keys
      are usable model_keys), and each model is generated through the EXISTING
      inference plane ``managers.dispatch.execute_prompt(task="text-to-image",
      model_key=<model>, ...)`` — the same front door the studio's own image
      runner (``video_intel/runners/imagegen.py``) drives, which also handles
      GPU-worker routing and is the caller path onto
      ``managers/imagegen/imagegen_runner.py``.

  * VIDEO  (categories: clip, movie)   -> capability t2v / i2v.
      ``produce_clip`` resolves models against the STUDIO's OWN registry
      (``video_intel/studio/registry.py``, pinned via ``StudioI2VSpec.model_id``)
      — a DIFFERENT namespace from the main registry — so the servable set comes
      from the router's own ``capable_model_ids(capability)`` and each model is
      bound as a PIN and rendered through ``produce_clip`` (via the shared
      ``runners.studio_i2v.run_produce_clip`` spec->produce_clip lifter).

ROBUSTNESS IS THE CONTRACT (operator hard requirement): one model failing
(refuse / OOM / load-fail / timeout / an unroutable pin) records ``ok=false`` +
the error and the sweep CONTINUES to the next model. Every per-model iteration is
wrapped in try/except; the sweep never aborts on a single failure. The battery
recorder and the studio-assist log are both best-effort (never the failure mode).

Import discipline: this module's top level is dependency-light (stdlib + the
studio enums) so ``video_intel/job_schema.py`` can import ``StudioTesterSpec`` at
app-boot with ZERO torch/diffusers/plane pull. Every heavy import (the plane, the
studio spine, the battery recorder, the assist log, the model registries) is LAZY
inside the function that needs it.

No pathlib anywhere. os.path only.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from hugpy_video.intel.studio.enums import Capability

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# The discovered category -> TYPE map. A category's TYPE decides the task(s), the
# enumeration source, and the generation entry point. (image/scene are image-type
# even though "scene" renders a strip of stills — the model that renders them is a
# text-to-image model, which is what the sweep varies.)
# --------------------------------------------------------------------------- #
CATEGORY_KIND: dict[str, str] = {
    "image": "image",
    "scene": "image",
    "clip": "video",
    "movie": "video",
}

# The main-registry task string image-type categories enumerate on.
IMAGE_TASKS: tuple[str, ...] = ("text-to-image",)

# Geometry defaults — a small, cheap render so a full sweep is affordable.
_DEFAULT_WIDTH = 768
_DEFAULT_HEIGHT = 768
_DEFAULT_FPS = 16


def category_kind(category: str) -> str:
    """The TYPE ("image"|"video") of a studio category, or raise ValueError."""
    if not (isinstance(category, str) and category.strip()):
        raise ValueError(f"category must be a non-empty string; got {category!r}")
    kind = CATEGORY_KIND.get(category.strip().lower())
    if kind is None:
        raise ValueError(
            f"unknown category {category!r}; known: {sorted(CATEGORY_KIND)}")
    return kind


def _default_out_root() -> str:
    from hugpy_video.intel.studio.job import STUDIO_ROOT
    return os.path.join(STUDIO_ROOT, "tester")


# --------------------------------------------------------------------------- #
# Enumeration — thin indirections so tests can monkeypatch WITHOUT importing the
# heavy registries. Real callers hit the true enumerators lazily.
# --------------------------------------------------------------------------- #
def _get_models_dict_by_tasks(tasks: list[str]) -> dict:
    from hugpy_engine.config.models.models_default import get_models_dict_by_tasks
    return get_models_dict_by_tasks(tasks=tasks)


def _capable_model_ids(capability: Capability, include_synthetic: bool) -> tuple:
    from hugpy_video.intel.studio.router import capable_model_ids
    return capable_model_ids(capability, include_synthetic=include_synthetic)


def enumerate_models(category: str, *, start_image: Optional[str] = None,
                     include_synthetic: bool = False) -> list[str]:
    """Every servable model of ``category``'s type — the sweep's roster.

    IMAGE type: the main registry's text-to-image models (keys are model_keys).
    VIDEO type: the studio router's ``capable_model_ids`` for the capability the
    prompt implies (i2v when a start image is given, else t2v). Sorted for a
    deterministic sweep order."""
    kind = category_kind(category)
    if kind == "image":
        return sorted(_get_models_dict_by_tasks(list(IMAGE_TASKS)).keys())
    cap = Capability.I2V if start_image else Capability.T2V
    return list(_capable_model_ids(cap, include_synthetic))


# --------------------------------------------------------------------------- #
# The two REAL generators. Each returns (ok, uri, error) and is itself defensive,
# but run_tester's outer try/except is the hard robustness guarantee regardless.
# --------------------------------------------------------------------------- #
def _generate_image_once(model: str, prompt: str, *, width=None, height=None,
                         seed=None, out_root=None, **_ignored):
    """One text-to-image generation of ``prompt`` bound to ``model``, driven
    through the existing inference plane (execute_prompt). Returns (ok, uri,
    error)."""
    from hugpy_video.intel.plane import execute_prompt
    from hugpy_platform.async_runtime import run

    kwargs = dict(task="text-to-image", prompt=prompt, model_key=model,
                  num_images=1, return_b64=True)
    if width:
        kwargs["width"] = width
    if height:
        kwargs["height"] = height
    if seed is not None:
        kwargs["seed"] = seed
    res = run(execute_prompt(**kwargs))
    ok = bool(getattr(res, "ok", False))
    if not ok:
        return False, "", str(getattr(res, "error", None) or "unknown")
    images = getattr(res, "images", None) or ()
    if not images:
        return False, "", "plane returned ok but produced no images"
    uri = getattr(images[0], "path", "") or ""
    return True, uri, None


def _generate_video_once(model: str, prompt: str, *, width=None, height=None,
                         fps=None, seed=0, start_image=None, out_root=None,
                         steps=None, cfg=None, requested_frames=None, negative=None,
                         **_ignored):
    """One clip render of ``prompt`` PINNED to ``model`` through the studio spine.
    Returns (ok, uri, error). A pin the router can't honor (unknown/incapable/
    won't-fit model) comes back as an Err — recorded, never raised.

    2026-08-12 (model-battery fix): goes through ``render_clip`` — THE shared
    delegate-or-inline primitive — not raw ``run_produce_clip``. The raw call
    skipped ``_resolve_autofit`` (a None budget then reached the router's
    ``vram.fits(None)`` -> TypeError before any pixel) AND skipped worker
    delegation entirely, so on the GPU-less central every sweep row died
    instantly. render_clip sizes the blank budget against the studio worker and
    delegates exactly like a cinema segment — the sweep now measures what a real
    render would do."""
    from hugpy_video.intel.runners.studio_i2v import render_clip
    from hugpy_video.intel.studio.job import make_studio_i2v

    capability = Capability.I2V.value if start_image else Capability.T2V.value
    spec = make_studio_i2v(
        capability=capability,
        width=int(width or _DEFAULT_WIDTH),
        height=int(height or _DEFAULT_HEIGHT),
        fps=int(fps or _DEFAULT_FPS),
        vram_budget_gb=None,           # AUTOFIT to the serving worker's free VRAM
        seed=int(seed or 0),
        out_root=out_root or _default_out_root(),
        start_image=start_image,
        prompt=prompt,
        model_id=model,                # the PIN — bind THIS model or Err-as-data
        steps=steps,
        cfg=cfg,
        requested_frames=requested_frames,
        negative=negative,
    )
    # UNIQUE per attempt: the worker keys renders idempotently by render_id (a
    # movie segment resume WANTS that), so a stable id here made every re-test of
    # a model come back with the PREVIOUS attempt's terminal state ("cancelled")
    # without rendering a pixel.
    import uuid as _uuid
    outcome = render_clip(
        spec,
        render_id=f"tester:{model}:{int(seed or 0)}:{_uuid.uuid4().hex[:8]}",
        # The tester row is not itself a bus job — the default probe would look up
        # an id the bus has never seen. Cancel rides the SWEEP job (run_tester's
        # own should_cancel gates between rows), so a row, once started, runs out.
        should_cancel=lambda: False,
    )
    if outcome.ok:
        return True, (outcome.path or ""), None
    err = getattr(outcome, "error", None)
    if err is not None:
        code = getattr(err, "code", "") or ""
        msg = getattr(err, "message", "") or str(err)
        return False, "", (f"[{code}] {msg}" if code else msg)
    return False, "", "render_clip failed"


# --------------------------------------------------------------------------- #
# Battery + assist-log seams — all best-effort, all lazy, none ever raise.
# --------------------------------------------------------------------------- #
def _open_battery(battery_dir: Optional[str], session_key: str):
    """A BatteryRun to record into: a pre-minted dir (reopened, so the HTTP
    response can carry the exact path) or a fresh per-session run. None when
    recording is disabled or no root is writable."""
    try:
        import hugpy_media.model_battery as mb
        if not mb.enabled():
            return None
        if battery_dir:
            root = os.path.dirname(os.path.normpath(battery_dir)) or battery_dir
            os.makedirs(battery_dir, exist_ok=True)
            return mb.BatteryRun(battery_dir, root)
        return mb.run_for_session(session_key=session_key)
    except Exception:
        logger.debug("tester: battery open failed (non-fatal)", exc_info=True)
        return None


def mint_battery_dir(session_key: str) -> Optional[str]:
    """Create the battery run-dir NOW (in the enqueuing process) and return its
    path, so the route can return it immediately. None when recording is disabled
    or no root is writable — the worker then falls back to a per-job run."""
    try:
        import hugpy_media.model_battery as mb
        if not mb.enabled():
            return None
        run = mb.run_for_session(session_key=session_key)
        return run.run_dir if run is not None else None
    except Exception:
        logger.debug("tester: mint_battery_dir failed (non-fatal)", exc_info=True)
        return None


def _record(battery, model, axis, ok, secs, uri, error):
    if battery is None:
        return
    try:
        import hugpy_media.model_battery as mb
        thumb = ""
        if ok and uri:
            try:
                thumb = mb.thumb_b64_for(uri)
            except Exception:
                thumb = ""
        battery.record(model=model, axis=axis, ok=ok, secs=secs,
                       uri=uri or "", thumb_b64=thumb, error=error)
    except Exception:
        logger.debug("tester: battery record failed (non-fatal)", exc_info=True)


def _emit(run_id, category, kind, model, idx, total, ok, error, secs):
    """Publish one per-model line to the studio-assist log so the UI generation
    log shows each result live. Best-effort."""
    try:
        from hugpy_video import studio_assist_log as sal
        outcome = "served" if ok else "worker_error"
        sal.append(
            run_id=run_id,
            mode="tester",
            kind=f"tester:{category}",
            model_requested=model,
            model_resolved=model,
            outcome=outcome,
            error=(None if ok else str(error or "unknown")),
            elapsed_ms=int(max(secs, 0.0) * 1000),
            text=f"[{idx + 1}/{total}] {model}: {'ok' if ok else 'FAIL'}",
        )
    except Exception:
        logger.debug("tester: assist-log emit failed (non-fatal)", exc_info=True)


def _new_run_id() -> str:
    try:
        from hugpy_video import studio_assist_log as sal
        return sal.new_run_id()
    except Exception:
        import uuid
        return uuid.uuid4().hex


# --------------------------------------------------------------------------- #
# The sweep.
# --------------------------------------------------------------------------- #
def run_tester(category: str, prompt: str, models: Optional[list] = None, *,
               out_root: Optional[str] = None,
               width: int = _DEFAULT_WIDTH, height: int = _DEFAULT_HEIGHT,
               fps: int = _DEFAULT_FPS, seed: int = 0,
               start_image: Optional[str] = None,
               steps: Optional[int] = None, cfg: Optional[float] = None,
               requested_frames: Optional[int] = None,
               negative: Optional[str] = None,
               include_synthetic: bool = False,
               run_label: Optional[str] = None,
               run_id: Optional[str] = None,
               battery_dir: Optional[str] = None,
               image_generator: Optional[Callable] = None,
               video_generator: Optional[Callable] = None) -> dict:
    """Iterate ``prompt`` across every model of ``category``'s type, recording a
    battery row per model. Returns a summary dict.

    ``models`` (optional) pins the roster to a subset; None enumerates the whole
    servable set. ``image_generator`` / ``video_generator`` are injection seams for
    tests (default to the real plane / studio spine). Each model iteration is
    wrapped in try/except so ONE failure records ok=false and the sweep continues.
    """
    kind = category_kind(category)
    if not (isinstance(prompt, str) and prompt.strip()):
        raise ValueError("prompt must be a non-empty string")

    if models is None:
        models = enumerate_models(category, start_image=start_image,
                                  include_synthetic=include_synthetic)
    models = [str(m) for m in models]
    if not models:
        raise ValueError(f"no servable models to test for {category!r}")

    out_root = out_root or _default_out_root()
    axis = run_label or f"tester:{category}"
    run_id = run_id or _new_run_id()
    battery = _open_battery(battery_dir, run_id)
    settings = {"category": category, "models": models, "width": width,
                "height": height, "fps": fps, "seed": seed,
                "start_image": start_image, "steps": steps, "cfg": cfg,
                "requested_frames": requested_frames, "negative": negative}
    if battery is not None:
        try:
            import json
            battery.log("sweep settings: " + json.dumps(settings, sort_keys=True))
        except Exception:
            logger.debug("tester: settings log failed (non-fatal)", exc_info=True)

    generate = image_generator if kind == "image" else video_generator
    if generate is None:
        generate = _generate_image_once if kind == "image" else _generate_video_once

    logger.info("tester: sweeping %d model(s) for category=%s (kind=%s) run_id=%s",
                len(models), category, kind, run_id)

    results: list[dict] = []
    for idx, model in enumerate(models):
        t0 = time.monotonic()
        ok, uri, error = False, "", None
        try:
            ok, uri, error = generate(
                model, prompt, width=width, height=height, fps=fps,
                seed=seed, start_image=start_image, out_root=out_root,
                steps=steps, cfg=cfg, requested_frames=requested_frames,
                negative=negative)
        except Exception as exc:  # HARD robustness: never abort the sweep
            ok, uri, error = False, "", f"{type(exc).__name__}: {exc}"
            logger.warning("tester: model %s raised (recorded, continuing): %s",
                           model, exc, exc_info=True)
        secs = time.monotonic() - t0
        _record(battery, model, axis, ok, secs, uri, error)
        _emit(run_id, category, kind, model, idx, len(models), ok, error, secs)
        results.append({"model": model, "ok": bool(ok),
                        "uri": uri or "", "error": (None if ok else str(error)),
                        "secs": round(secs, 2)})

    run_dir = getattr(battery, "run_dir", None) if battery is not None else None
    ok_count = sum(1 for r in results if r["ok"])
    logger.info("tester: category=%s done — %d/%d ok, run_dir=%s",
                category, ok_count, len(models), run_dir)
    return {
        "category": category,
        "kind": kind,
        "run_id": run_id,
        "run_dir": run_dir,
        "axis": axis,
        "count": len(models),
        "ok_count": ok_count,
        "results": results,
        "settings": settings,
    }


# --------------------------------------------------------------------------- #
# Bus job spec — the durable, JSON-safe intent for a tester sweep (mirrors
# studio.job.StudioI2VSpec: frozen + validate-at-construction + from_dict).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StudioTesterSpec:
    """Frozen currency of a ``studio_tester`` bus job. Built ONLY via
    ``make_studio_tester``; the bus rehydrates it through ``studio_tester_from_dict``
    (reconstruct + re-validate). All fields are JSON-safe primitives so
    ``asdict`` -> ``json.dumps`` round-trips cleanly."""
    category: str
    prompt: str
    models: tuple = ()               # explicit subset; () = enumerate the type
    out_root: Optional[str] = None
    width: int = _DEFAULT_WIDTH
    height: int = _DEFAULT_HEIGHT
    fps: int = _DEFAULT_FPS
    seed: int = 0
    start_image: Optional[str] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    requested_frames: Optional[int] = None
    negative: Optional[str] = None
    include_synthetic: bool = False
    run_label: Optional[str] = None
    # Pre-minted battery run-dir (so the enqueue response carries the exact path);
    # None -> the runner mints a per-job run. NON-CANONICAL telemetry location.
    battery_dir: Optional[str] = None


def make_studio_tester(*, category: str, prompt: str, models=None,
                       out_root: Optional[str] = None,
                       width: int = _DEFAULT_WIDTH, height: int = _DEFAULT_HEIGHT,
                       fps: int = _DEFAULT_FPS, seed: int = 0,
                       start_image: Optional[str] = None,
                       steps: Optional[int] = None, cfg: Optional[float] = None,
                       requested_frames: Optional[int] = None,
                       negative: Optional[str] = None,
                       include_synthetic: bool = False,
                       run_label: Optional[str] = None,
                       battery_dir: Optional[str] = None) -> StudioTesterSpec:
    """Validate every field and build the frozen ``StudioTesterSpec``. Raises
    ``ValueError``/``TypeError`` locally on any structural violation (house
    discipline: a structurally-invalid spec is a caller error caught at the
    boundary, never carried across the bus). Whether a pinned model actually
    exists / serves the category is a RUNTIME decision surfaced as ok=false rows,
    not validated here."""
    category_kind(category)  # raises ValueError on an unknown category
    if not (isinstance(prompt, str) and prompt.strip()):
        raise ValueError("prompt must be a non-empty string")
    for name, val in (("width", width), ("height", height), ("fps", fps)):
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(f"{name} must be a positive int; got {val!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int; got {seed!r}")
    if start_image is not None and not (isinstance(start_image, str) and start_image.strip()):
        raise ValueError(f"start_image must be a non-empty string or None; got {start_image!r}")
    if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100):
        raise ValueError("steps must be an int in [1, 100] or None")
    if cfg is not None and (isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not 0 <= cfg <= 20):
        raise ValueError("cfg must be a number in [0, 20] or None")
    if requested_frames is not None and (isinstance(requested_frames, bool) or not isinstance(requested_frames, int) or not 1 <= requested_frames <= 10000):
        raise ValueError("requested_frames must be an int in [1, 10000] or None")
    if negative is not None and not isinstance(negative, str):
        raise ValueError("negative must be a string or None")
    if run_label is not None and not isinstance(run_label, str):
        raise ValueError(f"run_label must be a string or None; got {run_label!r}")
    if out_root is not None and not (isinstance(out_root, str) and out_root.strip()):
        raise ValueError(f"out_root must be a non-empty string or None; got {out_root!r}")
    if battery_dir is not None and not (isinstance(battery_dir, str) and battery_dir.strip()):
        raise ValueError(f"battery_dir must be a non-empty string or None; got {battery_dir!r}")

    if models is None:
        models = ()
    if not isinstance(models, (list, tuple)):
        raise ValueError(f"models must be a list/tuple of model ids or None; got {models!r}")
    coerced = tuple(str(m) for m in models)
    for i, m in enumerate(coerced):
        if not m.strip():
            raise ValueError(f"models[{i}] must be a non-empty string")

    return StudioTesterSpec(
        category=category.strip().lower(),
        prompt=prompt,
        models=coerced,
        out_root=(out_root or None),
        width=int(width), height=int(height), fps=int(fps), seed=int(seed),
        start_image=start_image,
        steps=steps, cfg=cfg, requested_frames=requested_frames,
        negative=negative,
        include_synthetic=bool(include_synthetic),
        run_label=(run_label or None),
        battery_dir=(battery_dir or None),
    )


def studio_tester_from_dict(d: dict) -> StudioTesterSpec:
    """Rebuild a ``StudioTesterSpec`` from its ``asdict`` form THROUGH the
    validating factory (mirrors ``studio_i2v_from_dict``). Registered in
    ``media_bus.SPEC_DESERIALIZERS`` under the name ``"studio_tester"``."""
    return make_studio_tester(
        category=d["category"],
        prompt=d["prompt"],
        models=d.get("models"),
        out_root=d.get("out_root"),
        width=d.get("width", _DEFAULT_WIDTH),
        height=d.get("height", _DEFAULT_HEIGHT),
        fps=d.get("fps", _DEFAULT_FPS),
        seed=d.get("seed", 0),
        start_image=d.get("start_image"),
        steps=d.get("steps"), cfg=d.get("cfg"),
        requested_frames=d.get("requested_frames"), negative=d.get("negative"),
        include_synthetic=d.get("include_synthetic", False),
        run_label=d.get("run_label"),
        battery_dir=d.get("battery_dir"),
    )


def run_tester_from_spec(spec: "StudioTesterSpec", job_id: str):
    """Bus-runner adapter: drive ``run_tester`` from a rehydrated spec and return
    a ``JobResult``. The JOB succeeds when the SWEEP ran (even if every model
    failed — those are ok=false ROWS, not a job failure); only an unexpected raise
    in the orchestration itself terminals the job as failed."""
    from hugpy_video.intel.result_schema import JobError, JobResult
    try:
        summary = run_tester(
            category=spec.category,
            prompt=spec.prompt,
            models=(list(spec.models) or None),
            out_root=spec.out_root,
            width=spec.width, height=spec.height, fps=spec.fps, seed=spec.seed,
            start_image=spec.start_image,
            steps=spec.steps, cfg=spec.cfg,
            requested_frames=spec.requested_frames, negative=spec.negative,
            include_synthetic=spec.include_synthetic,
            run_label=spec.run_label,
            run_id=job_id,                      # correlate assist-log with the job
            battery_dir=spec.battery_dir,
        )
    except Exception as exc:  # orchestration itself failed — data, not a raise
        return JobResult(job_id, ok=False, error=JobError(
            code="tester_failed",
            message=f"{type(exc).__name__}: {exc}",
            retryable=False))
    return JobResult(job_id, ok=True, project={
        "name": spec.run_label or f"tester:{spec.category}",
        "uuid": job_id,
        "dir": summary.get("run_dir"),
        "battery_run_dir": summary.get("run_dir"),
        "kind": summary.get("kind"),
        "count": summary.get("count"),
        "ok_count": summary.get("ok_count"),
    })
