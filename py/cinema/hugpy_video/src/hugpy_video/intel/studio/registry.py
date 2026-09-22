"""Registries over globals (INV-4 / INV-8).

Three tables, all module-level, all populated via `register_*` and frozen in
practice after import:

  MODEL_REGISTRY     : model_id                -> ModelConfig      (the zoo)
  RUNNER_REGISTRY    : (Framework, Task)       -> RunnerSpec       (dispatch)
  CAPABILITY_TASKS   : Capability              -> tuple[Task, ...] (the join)

`validate_registry()` runs at import (see __init__) and proves the join is total:
every capability a model claims is routable, every task has a runner, every
declared Capability is either served or explicitly PLANNED. It collects EVERY
problem and raises once - you see the whole failure set, not just the first.

RUNNER GATE (k1): a seed entry (e.g. a video-engine bet like Hunyuan/CogVideoX/
Mochi/Open-Sora/SkyReels/AnimateDiff/FramePack/CodeFormer/LTX-2) may declare a
``RunnerSpec`` whose ``entrypoint`` dotted path does not resolve to a real module
in this tree yet — the bet hasn't landed, only the zoo row has. That is DATA, not
a bug: ``register_runner``/``runner_for`` stay a raw structural lookup (so
``validate_registry()``'s "every task has a runner" totality proof is unaffected
and the seed entry is never pruned), but a SEPARATE servability gate
(``runner_available`` / ``runner_gate_reason`` / ``gated_runners``) is layered on
top for anything that actually DISPATCHES (the router's task-picker, a direct
dispatch attempt): it checks import-resolvability via ``importlib.util.find_spec``
(cheap — locates the module, never executes it) and returns an honest
``"runner_missing: <module path>"`` reason instead of letting the router bind a
model that would only fail one layer down (or, worse, silently outrank a model
that could actually render). Dropping the runner module into the tree re-enables
the engine with ZERO seed edits — that's the point of gating instead of pruning."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

from hugpy_video.intel.studio.enums import Capability, Framework, Precision, PRECISION_QUALITY, Task
from hugpy_video.intel.studio.errors import RegistryError
from hugpy_video.intel.studio.schemas import ModelConfig


@dataclass(frozen=True, slots=True)
class RunnerSpec:
    """How to execute one (framework, task). `entrypoint` is wired explicitly -
    a dotted path to the callable, no auto-discovery (explicit env wiring)."""
    framework: Framework
    task: Task
    entrypoint: str
    min_precision: Precision
    supports_streaming: bool = False


MODEL_REGISTRY: dict[str, ModelConfig] = {}
RUNNER_REGISTRY: dict[tuple[Framework, Task], RunnerSpec] = {}
CAPABILITY_TASKS: dict[Capability, tuple[Task, ...]] = {}

# Capabilities intentionally not backed by a generative model (orchestration
# stages, e.g. multi-shot assembly). Declared so validate_registry() does not
# flag them as orphans, but tracked loudly rather than silently ignored.
PLANNED_CAPABILITIES: frozenset[Capability] = frozenset({Capability.ASSEMBLE})


def register_model(cfg: ModelConfig) -> None:
    if cfg.model_id in MODEL_REGISTRY:
        raise RegistryError(f"duplicate model_id: {cfg.model_id!r}")
    MODEL_REGISTRY[cfg.model_id] = cfg


def register_runner(spec: RunnerSpec) -> None:
    key = (spec.framework, spec.task)
    if key in RUNNER_REGISTRY:
        raise RegistryError(f"duplicate runner: {key}")
    RUNNER_REGISTRY[key] = spec


def set_capability_tasks(mapping: dict[Capability, tuple[Task, ...]]) -> None:
    CAPABILITY_TASKS.clear()
    CAPABILITY_TASKS.update(mapping)


def runner_for(framework: Framework, task: Task) -> RunnerSpec | None:
    """Raw structural lookup: a RunnerSpec is DECLARED for (framework, task), full
    stop — it says nothing about whether the entrypoint module actually exists on
    this tree. This is what ``validate_registry()``'s totality proof and any
    reporting on the DECLARED zoo shape must keep using, so a runner module that
    hasn't landed yet never turns into an import-time crash. For a servability
    decision (routing / dispatch), use ``runner_available`` instead."""
    return RUNNER_REGISTRY.get((framework, task))


# --------------------------------------------------------------------------
# Runner GATE (k1): import-resolvability, layered on top of the raw registry.
# --------------------------------------------------------------------------
_ENTRYPOINT_IMPORTABLE_CACHE: dict[str, bool] = {}


def _entrypoint_module(entrypoint: str) -> str:
    """The dotted MODULE path of a ``RunnerSpec.entrypoint`` ("mod.path:callable"
    -> "mod.path"; a bare "mod.path" with no ":" is returned unchanged)."""
    return entrypoint.split(":", 1)[0]


def _is_entrypoint_importable(entrypoint: str) -> bool:
    """True iff ``entrypoint``'s module can be LOCATED on this tree.

    Uses ``importlib.util.find_spec``, which resolves the module's location
    without executing its body — so this stays cheap and side-effect-free even
    for a runner module that (once it exists) pulls torch/diffusers at import.
    Cached per module path (the zoo is static after import). Any resolution
    failure (module genuinely absent, a malformed dotted path, a broken parent
    package) is treated as "not importable" rather than raised — this gate must
    never be the thing that crashes the process."""
    module = _entrypoint_module(entrypoint)
    cached = _ENTRYPOINT_IMPORTABLE_CACHE.get(module)
    if cached is not None:
        return cached
    try:
        found = importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError, TypeError):
        found = False
    _ENTRYPOINT_IMPORTABLE_CACHE[module] = found
    return found


def runner_available(framework: Framework, task: Task) -> RunnerSpec | None:
    """SERVABLE lookup: the registered ``RunnerSpec`` for (framework, task) IFF its
    entrypoint module resolves on this tree, else ``None`` — a GATED runner (the
    seed declared it; the module hasn't landed) is indistinguishable here from an
    unregistered one, which is exactly what the router's task-picker wants: skip
    it and fall through to a candidate that can actually render. Use
    ``runner_gate_reason`` to recover WHY a gated (framework, task) returned None."""
    spec = RUNNER_REGISTRY.get((framework, task))
    if spec is None:
        return None
    return spec if _is_entrypoint_importable(spec.entrypoint) else None


def runner_gate_reason(framework: Framework, task: Task) -> str | None:
    """None when (framework, task) is servable (``runner_available`` would return
    the spec); otherwise an honest, queryable reason string:
      * "not_registered"           — no RunnerSpec declared at all for this pair.
      * "runner_missing: <module>" — a RunnerSpec IS declared (the seed's stated
        intent) but its entrypoint module is not present in this tree yet.
    """
    spec = RUNNER_REGISTRY.get((framework, task))
    if spec is None:
        return "not_registered"
    if _is_entrypoint_importable(spec.entrypoint):
        return None
    return f"runner_missing: {_entrypoint_module(spec.entrypoint)}"


def gated_runners() -> tuple[tuple[str, str, str], ...]:
    """Reporting hook (mirrors ``unpinned_models()``): every REGISTERED
    (framework, task) whose entrypoint module cannot be found on this tree, as
    sorted ``(framework, task, reason)`` string triples. Empty when every
    declared runner is servable. This is the "servable catalog" queryable surface
    a picker/console can use to explain (or filter out) a dead engine without the
    seed entry ever being deleted."""
    out: list[tuple[str, str, str]] = []
    for (fw, task) in RUNNER_REGISTRY:
        reason = runner_gate_reason(fw, task)
        if reason is not None:
            out.append((fw.value, task.value, reason))
    return tuple(sorted(out))


def model_gate_reasons(model_id: str) -> dict[str, str]:
    """``{task_value: reason}`` for every task ``model_id`` declares whose runner
    is gated (empty dict when every declared task is servable, or the model_id is
    unknown). The per-model "why can't I run this" surface."""
    cfg = MODEL_REGISTRY.get(model_id)
    if cfg is None:
        return {}
    out: dict[str, str] = {}
    for task in cfg.tasks:
        reason = runner_gate_reason(cfg.family, task)
        if reason is not None:
            out[task.value] = reason
    return out


def _allow_unpinned() -> bool:
    # Explicit env gate, loud by default. Seed/dev may run unpinned; production
    # must pin weight hashes or set this deliberately.
    return os.environ.get("STUDIO_ALLOW_UNPINNED") == "1"


def validate_registry() -> None:
    """Fail-loud, comprehensive. Collects every problem, then raises once."""
    problems: list[str] = []

    # Every task referenced anywhere must have a runner.
    tasks_with_runner = {task for (_fw, task) in RUNNER_REGISTRY}

    for cap, tasks in CAPABILITY_TASKS.items():
        if not tasks and cap not in PLANNED_CAPABILITIES:
            problems.append(f"capability {cap.value!r} maps to no tasks and is not PLANNED")
        for task in tasks:
            if task not in tasks_with_runner:
                problems.append(
                    f"capability {cap.value!r} references task {task.value!r} "
                    f"with no runner registered for any framework"
                )

    served_caps: set[Capability] = set()

    for cfg in MODEL_REGISTRY.values():
        mid = cfg.model_id

        if not cfg.capabilities:
            problems.append(f"{mid}: declares no capabilities")
        if not cfg.tasks:
            problems.append(f"{mid}: declares no tasks")
        if not cfg.resolutions:
            problems.append(f"{mid}: declares no resolutions")

        # weight pinning (INV-1/INV-2)
        if cfg.weight_hash is None and not cfg.unpinned:
            problems.append(f"{mid}: no weight_hash and not marked unpinned")
        if cfg.weight_hash is None and cfg.unpinned and not _allow_unpinned():
            problems.append(
                f"{mid}: unpinned weights but STUDIO_ALLOW_UNPINNED != '1' "
                f"(pin the weight hash or set the env gate deliberately)"
            )

        # every (family, task) the model can run must have a runner
        for task in cfg.tasks:
            spec = runner_for(cfg.family, task)
            if spec is None:
                problems.append(
                    f"{mid}: no runner for ({cfg.family.value}, {task.value})"
                )
                continue
            # FIX-4: the model must expose >=1 VRAM precision that meets this
            # runner's min_precision floor, else the router can never bind it at a
            # supported quality (fail loud at boot, not at dispatch).
            floor = PRECISION_QUALITY[spec.min_precision]
            if not any(PRECISION_QUALITY[p] >= floor
                       for p, _gb in cfg.vram.per_precision):
                problems.append(
                    f"{mid}: no VRAM precision >= runner floor "
                    f"{spec.min_precision.value} for ({cfg.family.value}, {task.value})"
                )

        # every capability the model claims must be routable via one of its tasks
        for cap in cfg.capabilities:
            served_caps.add(cap)
            candidate_tasks = CAPABILITY_TASKS.get(cap)
            if candidate_tasks is None:
                problems.append(f"{mid}: claims capability {cap.value!r} absent from CAPABILITY_TASKS")
                continue
            if not (set(candidate_tasks) & set(cfg.tasks)):
                problems.append(
                    f"{mid}: claims capability {cap.value!r} but has none of its "
                    f"satisfying tasks {[t.value for t in candidate_tasks]}"
                )

        # streaming models must be backed by streaming-capable runners (STR path)
        if cfg.path_class.value == "streaming":
            for task in cfg.tasks:
                spec = runner_for(cfg.family, task)
                if spec is not None and not spec.supports_streaming:
                    problems.append(
                        f"{mid}: path_class=streaming but runner "
                        f"({cfg.family.value}, {task.value}) is not streaming-capable"
                    )

    # every declared Capability must be served by a model or explicitly PLANNED
    for cap in Capability:
        if cap not in served_caps and cap not in PLANNED_CAPABILITIES:
            problems.append(f"capability {cap.value!r} is served by no model and not PLANNED")

    if problems:
        raise RegistryError(
            "registry validation failed with "
            f"{len(problems)} problem(s):\n  - " + "\n  - ".join(problems)
        )


def unpinned_models() -> tuple[str, ...]:
    """Reporting hook: which models still need a pinned weight hash."""
    return tuple(sorted(m.model_id for m in MODEL_REGISTRY.values() if m.weight_hash is None))
