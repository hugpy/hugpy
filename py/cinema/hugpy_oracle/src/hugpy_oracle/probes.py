"""Registration probes (k101): does the adapter agree with its descriptor?

Doc §3.2, the sentence this module exists for: *"Registration probes must catch
interface mismatches before production work. An image adapter that unexpectedly
requires ``prompt`` or ``text``, for example, is ineligible until its descriptor
and probe agree with the endpoint."* Doc §4 step 3 puts the probe THIRD in the
resolution order, ahead of worker health and VRAM — an adapter whose interface
does not match is not "temporarily unavailable", it is wrong.

WHAT A PROBE IS ALLOWED TO COST. Everything here is a ``find_spec``, an
``inspect.signature``, a dict lookup or a single ``scandir`` — no model load, no
network, no GPU, no subprocess. Every probe runs under a HARD time budget
(``PROBE_BUDGET_S`` = 0.5 s): when the budget is spent, the checks that have not
run yet report ``unknown`` with "probe skipped: budget" instead of blocking a
capability listing. Results are cached with a TTL (env ``ORACLE_PROBE_TTL_S``,
default 300 s; ``0`` disables the cache) because ``GET /oracle/capabilities`` is
a page-load-frequency endpoint.

THE THREE VERDICTS, and why the third one matters:

  ``ok``       the check was performed and agrees with the descriptor.
  ``fail``     the check was performed and DISAGREES. The capability becomes
               INELIGIBLE and the probe's own detail is the reason
               (``CapabilityView.with_probe``) — the doc's rule, enforced.
  ``unknown``  the check could not be performed here (module unreadable, no
               worker registry, budget spent, nothing declared to compare
               against). It is advisory: it never makes a capability eligible
               and never makes one ineligible. A probe that cannot run reports
               unknown and NEVER ok — this fleet has no GPU and central cannot
               import most backends, so "unknown" is the common, honest answer.

WHAT IS DELIBERATELY *NOT* A FAIL: an absent worker seat, an unrecorded license,
a backend that is not importable on central. Those are AVAILABILITY facts the
catalog already reports precisely (``catalog._tts_view``); turning them into
probe failures would blur "nobody has seated this yet" into "this adapter is
broken", and the operator needs to be able to tell those apart.

Import discipline matches ``catalog.py``: module top level is stdlib +
``.contracts`` only. ``catalog``/studio/registry reads happen lazily INSIDE the
check functions and through module-level seams (``_find_spec``/``_import_module``
/``_scandir``/``_monotonic``/``_now``), so tests monkeypatch them and need no
worker, no GPU and no shared store.

No pathlib anywhere. os.path only.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

from hugpy_oracle.contracts import ProbeCheck, ProbeResult, ProbeStatus, canonical_json

logger = logging.getLogger(__name__)

#: Hard per-probe wall-clock budget. A registration probe competes with a page
#: load, so it is allowed half a second for ALL of its checks; whatever has not
#: run when the budget is spent answers ``unknown``. Deliberately a constant
#: rather than an env knob: a fleet that needs a bigger budget has a probe doing
#: something a probe should not be doing.
PROBE_BUDGET_S: float = 0.5

#: TTL for the result cache, overridable for operators who want fresher (or no)
#: caching. ``0`` disables the cache entirely.
ENV_PROBE_TTL: str = "ORACLE_PROBE_TTL_S"
DEFAULT_PROBE_TTL_S: float = 300.0

#: The exact wording the budget guard emits, as a constant so tests and the
#: dispatch record cannot drift from the code.
BUDGET_SKIPPED: str = "probe skipped: budget"


# ---------------------------------------------------------------------------
# Seams (monkeypatched in tests; never called at import time)
# ---------------------------------------------------------------------------


def _monotonic() -> float:
    import time
    return time.monotonic()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_spec(module: str) -> Any:
    return importlib.util.find_spec(module)


def _import_module(module: str) -> Any:
    return importlib.import_module(module)


def _scandir_entries(directory: str) -> tuple[tuple[str, int], ...]:
    """Top-level ``(filename, size)`` pairs under ``directory`` — ONE listing,
    never a walk. Enough to answer "are these weights zero bytes" (and, for
    ``catalog.model_dir_fingerprint``, "which files are they"); cheap enough to
    run against a shared store."""
    entries: list[tuple[str, int]] = []
    with os.scandir(directory) as it:
        for entry in it:
            try:
                if entry.is_file():
                    entries.append((entry.name, entry.stat().st_size))
            except OSError:               # a vanishing entry is not a verdict
                continue
    return tuple(entries)


def _scandir_sizes(directory: str) -> tuple[int, ...]:
    return tuple(size for _name, size in _scandir_entries(directory))


def probe_ttl_s() -> float:
    """The configured TTL in seconds. A bad value degrades to the default
    rather than breaking a capability listing."""
    raw = os.environ.get(ENV_PROBE_TTL)
    if raw is None or not str(raw).strip():
        return DEFAULT_PROBE_TTL_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.debug("oracle probes: bad %s=%r, using default", ENV_PROBE_TTL, raw)
        return DEFAULT_PROBE_TTL_S
    return max(0.0, value)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProbeSpec:
    """What to check for one capability/adapter pair.

    ``params`` is what the DESCRIPTOR promises the capability accepts;
    ``entrypoint`` names the thing on the adapter module whose parameter surface
    that promise is checked against (a function, a dataclass, or a pydantic
    model — all three are read the same way, see ``accepted_params``). Where the
    module exposes a ``PARAMS`` constant, that wins: an explicit declaration
    beats introspection.

    ``supplied_by_dispatch`` names parameters the adapter requires that the
    DISPATCH layer fills in rather than the capability's caller — an artifact
    path, a transport correlation id. They are excluded from the "the adapter
    requires something the descriptor never declared" direction of the check,
    which otherwise reports plumbing as an interface mismatch. Keep the list
    short and justified: every name on it is a check that stopped running.

    ``task`` is the fleet dispatch task string used for the worker-seat check
    (k98's STRICT ``catalog._worker_seats_task``). ``model_row`` turns on the
    registry-row checks (tasks declared, license recorded, weights non-zero)."""
    capability: str
    runner_module: str | None = None
    entrypoint: str | None = None
    params: tuple[str, ...] = ()
    supplied_by_dispatch: tuple[str, ...] = ()
    params_constant: str = "PARAMS"
    task: str | None = None
    model_row: bool = False
    check_weights: bool = False
    #: k118 — consult the ENVIRONMENT DOCTRINE for ``task`` on the workers that
    #: seat it. OPT-IN (default False) for two reasons: a spec with no task has
    #: nothing to look up, and a check that appears on every probe would change
    #: the verdict of every capability the day it landed. Where it IS on, a
    #: worker whose doctrine assessment blocks the task is a FAIL with the pip
    #: line in the reason — the missing dep stops being a runtime surprise.
    doctrine: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if "." not in self.capability:
            raise ValueError(
                f"ProbeSpec.capability must be a namespaced capability name, "
                f"got {self.capability!r}")


#: capability name -> its registration probe. SMALL ON PURPOSE: a spec exists
#: only where there is something real to check. A capability with no spec gets
#: ``probe=None`` on its descriptor, which reads as "no registration probe is
#: declared" — honest, and distinguishable from "probed and inconclusive".
#:
#: ``audio.speaker_similarity`` deliberately has NO spec: it is declared with no
#: binding at all (k98), so there is no adapter to probe and a probe result
#: would be theatre on top of a gap the catalog already states plainly.
PROBE_SPECS: dict[str, ProbeSpec] = {
    "audio.tts": ProbeSpec(
        capability="audio.tts",
        # Must equal catalog.TTS_RUNNER_MODULE — asserted by a test rather than
        # imported, because catalog imports THIS module (not the other way).
        runner_module="hugpy_media.tts.chatterbox_runner",
        entrypoint="make_tts",
        params=("text", "reference_audio", "authorized", "voice_style", "seed",
                "language", "device", "model_id"),
        task="text-to-speech",
        model_row=True,
        check_weights=True,
        doctrine=True,
        notes="chatterbox reference-conditioned TTS adapter (k98)"),
    "audio.transcribe.word_timestamps": ProbeSpec(
        capability="audio.transcribe.word_timestamps",
        # The capability IS audio.transcribe + one flag, so the interface fact
        # that matters is whether the whisper request schema carries the flag.
        # This is the structural half of catalog._word_timestamps_wired, stated
        # as a probe so a regression that drops the field turns the capability
        # ineligible instead of silently returning empty word lists.
        runner_module="hugpy_media.schemas.whisper_schemas",
        entrypoint="TranscribeRequest",
        params=("word_timestamps",),
        # TranscribeRequest also requires ``file_path`` (the audio ARTIFACT,
        # which the descriptor carries as ``accepts``, not as a parameter) and
        # ``request_id`` (transport correlation, minted by the dispatch layer).
        # Neither is a capability parameter, so neither is an interface
        # mismatch — see ProbeSpec.supplied_by_dispatch.
        supplied_by_dispatch=("file_path", "request_id"),
        task="automatic-speech-recognition",
        doctrine=True,
        notes="whisper word-timestamp passthrough (k98/k98b)"),
}


def register_probe(spec: ProbeSpec) -> None:
    """Register (or replace) a capability's probe. Public so a runner package
    can declare its own probe next to its adapter instead of editing this
    table; clears any cached result for that capability."""
    PROBE_SPECS[spec.capability] = spec
    _CACHE.pop(spec.capability, None)


def probe_spec_for(capability: str) -> ProbeSpec | None:
    return PROBE_SPECS.get(capability)


# ---------------------------------------------------------------------------
# Parameter-surface introspection
# ---------------------------------------------------------------------------


def accepted_params(target: Any) -> tuple[frozenset[str], frozenset[str]] | None:
    """``(accepted, required)`` parameter names of ``target``, or None when it
    has no readable parameter surface.

    Handles the three shapes an adapter actually comes in on this fleet:
    a pydantic model (``model_fields``/``__fields__``), a dataclass
    (``__dataclass_fields__``), and a plain callable (``inspect.signature``).
    ``**kwargs`` in a signature means "accepts anything", reported as an empty
    accepted set with a sentinel — see ``_ACCEPTS_ANY``."""
    fields = getattr(target, "model_fields", None)           # pydantic v2
    if isinstance(fields, dict) and fields:
        required = {name for name, f in fields.items()
                    if getattr(f, "is_required", None) is not None
                    and f.is_required()}
        return frozenset(fields), frozenset(required)
    fields = getattr(target, "__fields__", None)             # pydantic v1
    if isinstance(fields, dict) and fields:
        required = {name for name, f in fields.items()
                    if getattr(f, "required", False) is True}
        return frozenset(fields), frozenset(required)
    dc_fields = getattr(target, "__dataclass_fields__", None)
    if isinstance(dc_fields, dict) and dc_fields:
        import dataclasses as _dc
        required = {name for name, f in dc_fields.items()
                    if f.default is _dc.MISSING
                    and f.default_factory is _dc.MISSING}   # type: ignore[misc]
        return frozenset(dc_fields), frozenset(required)
    if callable(target):
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            return None
        accepted: set[str] = set()
        required: set[str] = set()
        for name, param in sig.parameters.items():
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                return (_ACCEPTS_ANY, frozenset(required))
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            if name in ("self", "cls"):
                continue
            accepted.add(name)
            if param.default is inspect.Parameter.empty:
                required.add(name)
        return (frozenset(accepted), frozenset(required))
    return None


#: Sentinel accepted-set meaning "the callable takes **kwargs, so it accepts
#: anything". A probe cannot prove a mismatch against it, so the check answers
#: ``unknown`` rather than pretending agreement.
_ACCEPTS_ANY: frozenset[str] = frozenset({"**"})


# ---------------------------------------------------------------------------
# The checks. Each returns a ProbeCheck and NEVER raises.
# ---------------------------------------------------------------------------


def _check_runner_module(spec: ProbeSpec) -> ProbeCheck:
    name = "runner_module"
    if not spec.runner_module:
        return ProbeCheck(name, ProbeStatus.UNKNOWN,
                          "no runner adapter module is declared for this "
                          "capability, so there is nothing to import")
    try:
        found = _find_spec(spec.runner_module) is not None
    except ModuleNotFoundError as exc:
        # A PARENT package that does not exist is the same finding as a module
        # that does not exist — find_spec just reports it by raising.
        return ProbeCheck(
            name, ProbeStatus.FAIL,
            f"runner adapter module {spec.runner_module!r} is not importable "
            f"({exc}) — the descriptor names an adapter that does not exist here")
    except (ImportError, ValueError, AttributeError) as exc:
        return ProbeCheck(name, ProbeStatus.UNKNOWN,
                          f"could not resolve {spec.runner_module!r} "
                          f"({type(exc).__name__}: {exc})")
    if not found:
        return ProbeCheck(
            name, ProbeStatus.FAIL,
            f"runner adapter module {spec.runner_module!r} is not importable — "
            f"the descriptor names an adapter that does not exist here")
    return ProbeCheck(name, ProbeStatus.OK, "")


def _check_params(spec: ProbeSpec) -> ProbeCheck:
    """The doc's own example, mechanized: declared params ⊆ what the adapter
    accepts, AND nothing the adapter REQUIRES is missing from the descriptor."""
    name = "param_agreement"
    if not spec.runner_module:
        return ProbeCheck(name, ProbeStatus.UNKNOWN,
                          "no adapter module declared to compare against")
    try:
        module = _import_module(spec.runner_module)
    except Exception as exc:  # noqa: BLE001 — an unimportable adapter is a finding
        return ProbeCheck(
            name, ProbeStatus.UNKNOWN,
            f"adapter {spec.runner_module!r} did not import here "
            f"({type(exc).__name__}: {exc}) — its parameter surface is "
            f"unreadable from this process")

    declared = frozenset(spec.params)
    constant = getattr(module, spec.params_constant, None)
    if isinstance(constant, (tuple, list, set, frozenset)) and constant:
        accepted: frozenset[str] = frozenset(str(p) for p in constant)
        required: frozenset[str] = frozenset()
        surface = f"{spec.params_constant} constant"
    else:
        target = (getattr(module, spec.entrypoint, None)
                  if spec.entrypoint else None)
        if target is None:
            return ProbeCheck(
                name, ProbeStatus.UNKNOWN,
                f"adapter {spec.runner_module!r} exposes neither a "
                f"{spec.params_constant} constant nor "
                f"{spec.entrypoint!r} — no parameter surface to compare")
        read = accepted_params(target)
        if read is None:
            return ProbeCheck(
                name, ProbeStatus.UNKNOWN,
                f"{spec.runner_module}.{spec.entrypoint} has no readable "
                f"parameter surface")
        accepted, required = read
        surface = f"{spec.entrypoint}"

    if accepted is _ACCEPTS_ANY or accepted == _ACCEPTS_ANY:
        return ProbeCheck(
            name, ProbeStatus.UNKNOWN,
            f"{spec.runner_module}.{surface} takes **kwargs — it accepts "
            f"anything, so a mismatch cannot be proven either way")
    if not declared:
        return ProbeCheck(name, ProbeStatus.UNKNOWN,
                          "the descriptor declares no parameters, so there is "
                          "nothing to check against the adapter")

    unaccepted = sorted(declared - accepted)
    undeclared_required = sorted(
        required - declared - frozenset(spec.supplied_by_dispatch))
    if unaccepted or undeclared_required:
        parts: list[str] = []
        if unaccepted:
            parts.append(
                f"the descriptor declares {unaccepted} which "
                f"{spec.runner_module}.{surface} does not accept")
        if undeclared_required:
            parts.append(
                f"{spec.runner_module}.{surface} REQUIRES "
                f"{undeclared_required}, absent from the descriptor")
        return ProbeCheck(name, ProbeStatus.FAIL, "; ".join(parts))
    return ProbeCheck(name, ProbeStatus.OK, "")


def _check_model_rows(spec: ProbeSpec, rows: Mapping[str, Mapping[str, Any]],
                      model_ids: tuple[str, ...]) -> tuple[ProbeCheck, ...]:
    """Registry-row facts: the row exists, declares tasks, records a license,
    and its weights are not zero bytes. Bound models only — a capability with
    no model is the catalog's story to tell, not the probe's."""
    out: list[ProbeCheck] = []
    if not model_ids:
        out.append(ProbeCheck(
            "model_row", ProbeStatus.UNKNOWN,
            "no model is bound to this capability, so no registry row could "
            "be checked"))
        return tuple(out)

    missing = [m for m in model_ids if m not in rows]
    if missing:
        out.append(ProbeCheck(
            "model_row", ProbeStatus.FAIL,
            f"model(s) {missing} are bound to this capability but have no "
            f"registry row"))
        return tuple(out)

    no_tasks = [m for m in model_ids if not (rows[m].get("tasks") or ())]
    if no_tasks:
        out.append(ProbeCheck(
            "model_row", ProbeStatus.FAIL,
            f"registry row(s) {no_tasks} declare no tasks — a row that says "
            f"nothing about what it does cannot back a capability"))
    else:
        out.append(ProbeCheck("model_row", ProbeStatus.OK, ""))

    unlicensed = [m for m in model_ids if not _row_license(rows[m])]
    if unlicensed:
        out.append(ProbeCheck(
            "model_license", ProbeStatus.UNKNOWN,
            f"no license recorded on registry row(s) {unlicensed} — unknown is "
            f"reported, never assumed permissive"))
    else:
        out.append(ProbeCheck("model_license", ProbeStatus.OK, ""))

    if spec.check_weights:
        out.append(_check_weights(rows, model_ids))
    return tuple(out)


def _row_license(row: Mapping[str, Any]) -> str | None:
    """The row's declared license or None. Same lookup as
    ``catalog._row_license`` (duplicated rather than imported to keep this
    module's top level free of the catalog — a test asserts they agree)."""
    extra = row.get("extra") if isinstance(row, Mapping) else None
    bags: list[Any] = [row, extra if isinstance(extra, Mapping) else {}]
    inner = bags[1].get("extra") if isinstance(bags[1], Mapping) else None
    bags.append(inner if isinstance(inner, Mapping) else {})
    for bag in bags:
        value = bag.get("license")
        if value:
            return str(value)
    return None


def _row_dir(row: Mapping[str, Any]) -> str | None:
    """The absolute on-disk directory a row points at, when it records one."""
    extra = row.get("extra")
    if isinstance(extra, Mapping):
        for key in ("dir", "path", "local_dir"):
            value = extra.get(key)
            if value and os.path.isabs(str(value)):
                return str(value)
    for key in ("dir", "path", "local_path"):
        value = row.get(key)
        if value and os.path.isabs(str(value)):
            return str(value)
    return None


def _check_weights(rows: Mapping[str, Mapping[str, Any]],
                   model_ids: tuple[str, ...]) -> ProbeCheck:
    """Are the bound weights actually there and non-empty?

    The studio's ``ZERO_BYTE_MODELS`` is the same fact for the studio zoo (a
    hand-maintained list of rows whose files are 0 bytes) and is consulted
    first; for legacy rows the check is one ``scandir`` of the recorded
    directory. Zero bytes is a FAIL — the row promises weights that are not
    there. An unreadable path is UNKNOWN: a shared store that is not mounted
    right now says nothing about the adapter."""
    name = "model_weights"
    try:
        from hugpy_video.intel.studio.presets import ZERO_BYTE_MODELS
        known_empty = sorted(set(model_ids) & set(ZERO_BYTE_MODELS))
    except Exception as exc:  # noqa: BLE001
        logger.debug("oracle probes: studio viability list unreadable (%s)", exc)
        known_empty = []
    if known_empty:
        return ProbeCheck(
            name, ProbeStatus.FAIL,
            f"model(s) {known_empty} are recorded as 0 bytes on the shared "
            f"store (studio presets.ZERO_BYTE_MODELS)")

    unchecked: list[str] = []
    for model_id in model_ids:
        directory = _row_dir(rows[model_id])
        if not directory:
            unchecked.append(f"{model_id}: row records no absolute path")
            continue
        try:
            sizes = _scandir_sizes(directory)
        except FileNotFoundError:
            return ProbeCheck(
                name, ProbeStatus.FAIL,
                f"{model_id}: weights directory {directory!r} does not exist")
        except OSError as exc:
            unchecked.append(f"{model_id}: {directory!r} unreadable ({exc.strerror})")
            continue
        if not sizes or not any(sizes):
            return ProbeCheck(
                name, ProbeStatus.FAIL,
                f"{model_id}: weights directory {directory!r} holds no "
                f"non-empty file")
    if unchecked:
        return ProbeCheck(name, ProbeStatus.UNKNOWN, "; ".join(unchecked))
    return ProbeCheck(name, ProbeStatus.OK, "")


def _check_worker_seat(spec: ProbeSpec,
                       workers: list[Mapping[str, Any]] | None) -> ProbeCheck:
    """Does any online worker AFFIRMATIVELY advertise the dispatch task?

    Reuses k98's strict ``catalog._worker_seats_task`` (imported lazily so this
    module keeps its stdlib-only top level, and looked up through the module so
    a monkeypatched seam is honored). Never a FAIL: an unseated task means
    "nobody has taken this yet", which is availability, not a broken
    interface."""
    name = "worker_seat"
    if not spec.task:
        return ProbeCheck(name, ProbeStatus.UNKNOWN,
                          "no dispatch task declared for this capability")
    if workers is None:
        return ProbeCheck(name, ProbeStatus.UNKNOWN,
                          "worker registry unreadable — no seat can be confirmed")
    if not workers:
        return ProbeCheck(name, ProbeStatus.UNKNOWN,
                          "no online worker registered")
    from hugpy_oracle import catalog as _catalog
    seated = [str(w.get("id") or w.get("worker_id") or "?") for w in workers
              if _catalog._worker_seats_task(w, spec.task)]
    if not seated:
        return ProbeCheck(
            name, ProbeStatus.UNKNOWN,
            f"none of the {len(workers)} online worker(s) advertises "
            f"task_capabilities[{spec.task!r}] (STRICT/affirmative)")
    return ProbeCheck(name, ProbeStatus.OK, "")


def _latest_doctrine() -> Any:
    """The fleet doctrine, read through the ``DoctrineSource`` provider
    (``hugpy_oracle.providers``) — a seam tests can swap. None when central
    holds none — which this module reports as UNKNOWN."""
    from hugpy_oracle.providers import get_doctrine_source
    return get_doctrine_source().latest()


def _check_doctrine(spec: ProbeSpec,
                    workers: list[Mapping[str, Any]] | None) -> ProbeCheck:
    """k118 — does any worker that SEATS this task have the environment to run it?

    The failures this exists for were all the same: a worker advertised a task
    (its ``find_spec`` probe said yes), central routed to it, and the job died
    because ``ffmpeg`` / ``bitsandbytes`` / ``diffusers`` was not there. The
    doctrine assessment already knows that BEFORE dispatch, and every worker
    carries its own verdict on the heartbeat (``doctrine_status``), so this
    check is dict reads — no network, no import of anything heavy.

    THE VERDICTS, and the asymmetry that matters:

      ``ok``      at least one seating worker's assessment does NOT block the
                  task. One healthy box is enough — the capability is real.
      ``fail``    EVERY seating worker's assessment blocks it, and the reason
                  carries the exact repair command. This is the one place k118
                  makes a capability ineligible, and it only does so on
                  affirmative evidence from every candidate.
      ``unknown`` no doctrine, no workers, no reports, or a mix where some box
                  simply has not reported. A worker that never answered is not
                  a broken worker.
    """
    name = "doctrine"
    if not spec.doctrine or not spec.task:
        return ProbeCheck(name, ProbeStatus.UNKNOWN,
                          "no doctrine gate is declared for this capability")
    if workers is None:
        return ProbeCheck(name, ProbeStatus.UNKNOWN,
                          "worker registry unreadable — no environment can be "
                          "assessed")
    try:
        current = _latest_doctrine()
    except Exception as exc:  # noqa: BLE001 — a probe never becomes the failure
        return ProbeCheck(name, ProbeStatus.UNKNOWN,
                          f"doctrine unreadable ({type(exc).__name__}: {exc})")
    if current is None:
        return ProbeCheck(name, ProbeStatus.UNKNOWN,
                          "central holds no environment doctrine to assess "
                          "against")
    from hugpy_oracle import catalog as _catalog
    seated = [w for w in workers if _catalog._worker_seats_task(w, spec.task)]
    if not seated:
        return ProbeCheck(
            name, ProbeStatus.UNKNOWN,
            f"no online worker advertises task_capabilities[{spec.task!r}], so "
            f"there is no environment to assess")
    blocked: list[str] = []
    silent: list[str] = []
    for worker in seated:
        worker_name = str(worker.get("name") or worker.get("id") or "?")
        status = worker.get("doctrine_status")
        if not isinstance(status, Mapping) or not status:
            silent.append(worker_name)
            continue
        if spec.task not in (status.get("blocked_tasks") or []):
            return ProbeCheck(name, ProbeStatus.OK, "")
        repair = (status.get("repairs") or {}).get(spec.task) or "no repair recorded"
        blocked.append(f"{worker_name}: {repair}")
    if not blocked:
        return ProbeCheck(
            name, ProbeStatus.UNKNOWN,
            f"none of the {len(silent)} seating worker(s) has reported a "
            f"doctrine assessment yet ({', '.join(sorted(silent))}) — their "
            f"agents predate the environment report")
    if silent:
        return ProbeCheck(
            name, ProbeStatus.UNKNOWN,
            f"doctrine {current.version} blocks {spec.task!r} on "
            f"{'; '.join(blocked)} — but {', '.join(sorted(silent))} has not "
            f"reported, so a working seat cannot be ruled out")
    return ProbeCheck(
        name, ProbeStatus.FAIL,
        f"doctrine {current.version}: every worker seating {spec.task!r} is "
        f"missing a required dependency — {'; '.join(blocked)}")


# ---------------------------------------------------------------------------
# Running a probe
# ---------------------------------------------------------------------------


def run_probe(spec: ProbeSpec, *,
              rows: Mapping[str, Mapping[str, Any]] | None = None,
              model_ids: tuple[str, ...] = (),
              workers: list[Mapping[str, Any]] | None = None,
              budget_s: float = PROBE_BUDGET_S) -> ProbeResult:
    """Run ``spec``'s checks under the time budget and fold them into one
    result. UNCACHED — ``probe_capability`` is the cached entry point.

    The budget is checked BETWEEN checks: an individual check is expected to be
    sub-millisecond, so the guard catches the pathological case (a stat on a
    stalled mount, an adapter whose import pulls torch) by refusing to start
    the checks that would follow it. Everything skipped answers ``unknown``
    with ``BUDGET_SKIPPED`` — never ``ok``, and never silently dropped."""
    started = _monotonic()
    checks: list[ProbeCheck] = []
    over_budget = False

    def _spent() -> bool:
        return (_monotonic() - started) > budget_s

    planned: list[tuple[str, Callable[[], Iterable[ProbeCheck]]]] = [
        ("runner_module", lambda: (_check_runner_module(spec),)),
        ("param_agreement", lambda: (_check_params(spec),)),
    ]
    if spec.model_row:
        planned.append(("model_row",
                        lambda: _check_model_rows(spec, dict(rows or {}),
                                                  tuple(model_ids))))
    if spec.task:
        planned.append(("worker_seat", lambda: (_check_worker_seat(spec, workers),)))
    if spec.doctrine:
        planned.append(("doctrine", lambda: (_check_doctrine(spec, workers),)))

    for label, run in planned:
        if over_budget or _spent():
            over_budget = True
            checks.append(ProbeCheck(
                label, ProbeStatus.UNKNOWN,
                f"{BUDGET_SKIPPED} ({budget_s:g}s spent before this check ran)"))
            continue
        try:
            checks.extend(run())
        except Exception as exc:  # noqa: BLE001 — a probe never becomes the failure
            logger.debug("oracle probes: %s check %s raised (%s)",
                         spec.capability, label, exc)
            checks.append(ProbeCheck(
                label, ProbeStatus.UNKNOWN,
                f"check raised {type(exc).__name__}: {exc}"))
    return ProbeResult.from_checks(tuple(checks), probed_at=_now())


# ---------------------------------------------------------------------------
# TTL cache
# ---------------------------------------------------------------------------

#: capability -> (fingerprint, monotonic_deadline, result). The fingerprint is
#: part of the key on purpose: a cache that ignored WHAT was probed would hand
#: a stale verdict to a fleet whose rows or workers just changed (and would
#: leak between tests, which is the same bug with a shorter fuse).
_CACHE: dict[str, tuple[str, float, ProbeResult]] = {}


def clear_cache() -> None:
    _CACHE.clear()


def cache_size() -> int:
    return len(_CACHE)


def fingerprint(model_ids: Iterable[str] = (),
                workers: list[Mapping[str, Any]] | None = None,
                registry_version: str | None = None) -> str:
    """A cheap, deterministic description of the inputs a probe reads. Two
    different fleets/registries never share a cache entry."""
    worker_ids = (None if workers is None
                  else sorted(str(w.get("id") or w.get("worker_id") or "?")
                              for w in workers))
    return canonical_json({"models": sorted(model_ids),
                           "workers": worker_ids,
                           "registry_version": registry_version})


def probe_capability(capability: str, *,
                     rows: Mapping[str, Mapping[str, Any]] | None = None,
                     model_ids: tuple[str, ...] = (),
                     workers: list[Mapping[str, Any]] | None = None,
                     registry_version: str | None = None,
                     ttl_s: float | None = None) -> ProbeResult | None:
    """The cached probe for ``capability``, or None when none is registered.

    None is a MEANINGFUL answer and is not the same as ``unknown``: it means
    nobody declared a probe for this capability, which the descriptor shows as
    ``probe: null``. ``unknown`` means a declared probe ran and could not
    decide."""
    spec = PROBE_SPECS.get(capability)
    if spec is None:
        return None
    ttl = probe_ttl_s() if ttl_s is None else max(0.0, ttl_s)
    key = fingerprint(model_ids, workers, registry_version)
    if ttl > 0:
        cached = _CACHE.get(capability)
        if cached is not None:
            cached_key, deadline, result = cached
            if cached_key == key and _monotonic() < deadline:
                return result
    result = run_probe(spec, rows=rows, model_ids=model_ids, workers=workers)
    if ttl > 0:
        _CACHE[capability] = (key, _monotonic() + ttl, result)
    return result


__all__ = [
    "BUDGET_SKIPPED",
    "DEFAULT_PROBE_TTL_S",
    "ENV_PROBE_TTL",
    "PROBE_BUDGET_S",
    "PROBE_SPECS",
    "ProbeCheck",
    "ProbeResult",
    "ProbeSpec",
    "ProbeStatus",
    "accepted_params",
    "cache_size",
    "clear_cache",
    "fingerprint",
    "probe_capability",
    "probe_spec_for",
    "probe_ttl_s",
    "register_probe",
    "run_probe",
]
