"""k109 — the reproducible model evaluation for the script-first pipeline.

Three tracks (doc "Full-scale model evaluation"), one runner, TWO SCORING
LAYERS that are reported separately, and a per-operation routing matrix
(``routing_matrix.py``) derived from the rows this module writes.

    A  screenplay completion   partial / disconnected / incomplete material
    B  plot construction       the doc's six input conditions
    C  filmmaking workflow     breakdown, continuity, shot list, storyboard,
                               segment prompts, assembly plan

THE TWO LAYERS, AND WHY THEY NEVER MIX.

  DETERMINISTIC — the k110 validators plus countable checks. A Track A answer
  is fed to the SAME ``Screenplay`` constructor the pipeline uses, so "valid"
  means the artifact would actually have been accepted, not that it looked
  plausible. On top of that: preservation of supplied material (the operator's
  own lines and beats, listed on the case), contradiction rate (validator
  contradictions + declared contradiction pairs), completeness (required
  fields) and constraint adherence. All of it is reproducible on the recorded
  raw reply with no model in the loop.

  JUDGE — an independent LLM under a per-track rubric, resolved through the
  catalog (never hardcoded) and NEVER the candidate itself. A judge that IS the
  candidate is refused outright rather than quietly allowed, because a model
  grading its own screenplay is the one measurement in this benchmark that
  would be worthless AND look fine. Verdict parsing is k90c's
  ``evaluation.parse_judge_verdict`` — same VERDICT/SCORE/WHY discipline, same
  tolerance, same "unavailable is data, not a failure" degradation.

QUALITY AND PERFORMANCE ARE REPORTED SEPARATELY. Every attempt carries a
:class:`PerfRecord` (latency, tokens, tokens/s when the fleet exposes it, VRAM
before/after, worker, load state) and a :class:`DeterministicScore` /
:class:`JudgeScore`. Nothing here computes a speed-adjusted quality number; the
composite is derived afterwards, in ``routing_matrix.py``, with its formula
printed next to it.

THE DETERMINISTIC COMPOSITE, written down once (``_WEIGHTS``)::

    score = 100 * sum(w_k * v_k for k present) / sum(w_k for k present)

    valid 0.35 | preservation 0.20 | completeness 0.20 |
    constraints 0.15 | clean (= 1 - contradiction_rate) 0.10 |
    accuracy 0.20  (only where a DERIVED answer key exists)

Axes that do not apply to a case are absent from both sums, so a case with no
constraints is not silently credited with perfect constraint adherence.

TWO MODES (doc "VRAM utilization requirements"):

  ``--normalized``  fixed context / output / sampling for every candidate. The
                    only mode that compares models to each other.
  ``--ceiling``     each model's highest viable configuration, read off the
                    catalog's own limits where they exist, with a CONFIGURABLE
                    safety reserve recorded in the report.

WHAT THIS MODULE DOES NOT DO. It does not call a worker, read a GPU, or import
torch. Everything outward goes through the oracle's existing seams —
``router.resolve_route`` for the model pick (so eligibility, blocks and the
authority gate all still apply), ``runtime._dispatch`` for inference (the same
front door ``evaluation.py``'s judge uses), and the fleet's own read-only
telemetry endpoint for VRAM. Every seam is a module-level function so tests
drive the whole benchmark offline with no fleet at all.

No pathlib anywhere.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from hugpy_oracle.benchmark_cases import (
    FIXTURE_SCREENPLAY,
    OPERATIONS,
    BenchCase,
    Constraint,
    OperationSpec,
    cases_for,
)
from hugpy_oracle.contracts import GoalSpec
from hugpy_oracle.screenplay import (
    AuthoringGap,
    PlotSpec,
    Screenplay,
    author_plot,
    build_continuity,
    build_shot_plan,
    chain_breaks,
    parse_json_object,
    plot_input_mode,
    schema_block,
)

logger = logging.getLogger(__name__)

#: The capability every candidate and every judge is routed as.
BENCH_CAPABILITY: str = "text.chat"

#: The two benchmark modes.
MODES: tuple[str, ...] = ("normalized", "ceiling")

#: The NORMALIZED configuration — identical for every candidate, which is the
#: only thing that makes two models' numbers comparable. Sampling is pinned as
#: low as the artifacts tolerate: this benchmark measures whether a model can
#: hit a schema, not how inventive it is when unbounded.
NORMALIZED_PARAMS: dict[str, Any] = {
    "max_new_tokens": 2048,
    "temperature": 0.4,
    "top_p": 0.9,
    "context_tokens": 8192,
}

#: The CEILING fallback, used only where the catalog cannot say what a model's
#: real limit is. A guess that is recorded AS a guess (``ceiling_source``).
CEILING_PARAMS: dict[str, Any] = {
    "max_new_tokens": 4096,
    "temperature": 0.7,
    "top_p": 0.95,
    "context_tokens": 16384,
}

#: Safety reserve for ceiling mode, in GiB. Configurable (CLI ``--vram-reserve
#: -gib``), recorded in the run's environment block — the doc requires both.
DEFAULT_VRAM_RESERVE_GIB: float = 2.0

#: A model is dropped from the sweep after this many CONSECUTIVE dispatch
#: timeouts. Two, not one: a single timeout is a busy fleet, two in a row on
#: different cases is a model that cannot serve this workload politely.
TIMEOUT_ABORT_STREAK: int = 2

#: Per-attempt dispatch budget, seconds. Overridable per run.
DEFAULT_ATTEMPT_DEADLINE_S: float = 240.0

#: Where run dirs go. The battery's root, with an ``oracle-`` prefix so the
#: image battery's dirs and these never collide.
#: The battery's historical home; the live root is ``hugpy_oracle.config.benchmark_root()``.
DEFAULT_RUN_ROOT: str = "/home/ubuntu/station/model-battery"
RUN_ROOT_ENV: str = "ORACLE_BENCHMARK_ROOT"

#: The fleet's small VRAM poll (derived from the last heartbeat, no worker
#: I/O). Read-only, and the ONLY outward telemetry read this module makes.
VRAM_ENDPOINT: str = "/llm/vram"
#: or-k10 — cadence of the peak-VRAM sampler (CLI ``--sample-ms``). ``0``
#: disables the sampler entirely; the before/after delta is still recorded.
DEFAULT_VRAM_SAMPLE_MS: int = 200
TELEMETRY_TIMEOUT_S: float = 3.0

#: Deterministic composite weights. See the module docstring.
_WEIGHTS: dict[str, float] = {
    "valid": 0.35, "preservation": 0.20, "completeness": 0.20,
    "constraints": 0.15, "clean": 0.10, "accuracy": 0.20,
}

_MAX_RAW_CHARS: int = 200_000


# ---------------------------------------------------------------------------
# Provider seams — module-level and lazy, exactly like evaluation.py's.
# ---------------------------------------------------------------------------


def _resolve_route(capability: str = BENCH_CAPABILITY,
                   model_id: str | None = None,
                   objective: str = "benchmark a script-first operation"):
    """The candidate's route through the k90a catalog — the same resolution the
    pipeline uses, so eligibility, operator blocks and the authority gate all
    still apply to a benchmark run. Returns a ``RouteDecision`` or None."""
    from hugpy_oracle import router
    goal = GoalSpec(objective=objective, raw_prompt="(benchmark)",
                    capability=capability)
    try:
        return router.resolve_route(goal, model_id)
    except Exception as exc:  # noqa: BLE001 — an unroutable candidate is DATA
        logger.info("benchmark: route resolution failed for %s (%s: %s)",
                    model_id or capability, type(exc).__name__, exc)
        return None


def _dispatch(task: str, body: dict[str, Any]) -> Any:
    """One inference call through the SAME front door the runtime and the k90c
    judge use (normalize + execute_prompt). Never a worker call."""
    from hugpy_oracle import runtime
    return runtime._dispatch(runtime._normalized_kwargs(task, body))


def _run_bounded(fn: Callable[[], Any], deadline_s: float, label: str) -> Any:
    """runtime.run_bounded — the oracle's own deadline, reused so a benchmark
    attempt ends the same way a production request would."""
    from hugpy_oracle import runtime
    return runtime.run_bounded(fn, deadline_s, label)


def _registry_version() -> str | None:
    """catalog.registry_version(), or None. Never raises: a benchmark that
    cannot read the snapshot still records honest rows, unstamped."""
    try:
        from hugpy_oracle import catalog
        return catalog.registry_version()
    except Exception as exc:  # noqa: BLE001
        logger.info("benchmark: registry_version unavailable (%s: %s)",
                    type(exc).__name__, exc)
        return None


def _capability_view(capability: str = BENCH_CAPABILITY):
    try:
        from hugpy_oracle import catalog
        return catalog.get_capability(capability)
    except Exception as exc:  # noqa: BLE001
        logger.info("benchmark: catalog read failed (%s: %s)",
                    type(exc).__name__, exc)
        return None


def _vram_snapshot() -> dict[str, Any] | None:
    """The fleet's own VRAM meter (``GET /llm/vram``), or None.

    This is the ONE thing the doc's "record peak VRAM" asks for that the oracle
    can honestly answer without touching a GPU: central derives the meter from
    the last worker heartbeat, so reading it costs no worker I/O and cannot
    disturb a model that is loading. Anything that fails — no central, no
    route, a timeout — yields None, and None is written into the row as None.
    A fabricated number here would be worse than no number."""
    try:
        import urllib.request

        from hugpy_platform.central import central_base_url
        url = central_base_url().rstrip("/") + VRAM_ENDPOINT
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=TELEMETRY_TIMEOUT_S) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001 — telemetry is best-effort by design
        logger.debug("benchmark: vram snapshot unavailable (%s: %s)",
                     type(exc).__name__, exc)
        return None


def _nvml_reader() -> Callable[[], int | None] | None:
    """A zero-arg reader returning the sum of ``nvmlDeviceGetMemoryInfo().used``
    over every visible device, or None when pynvml is missing, the driver is
    not loaded, or anything else goes wrong. Import-guarded: this host may
    have no GPU at all. NVML is initialised once here, not per sample."""
    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 — optional dependency
        return None
    try:
        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                   for i in range(int(pynvml.nvmlDeviceGetCount()))]
    except Exception as exc:  # noqa: BLE001 — telemetry is best-effort
        logger.debug("benchmark: nvml unavailable (%s: %s)",
                     type(exc).__name__, exc)
        return None
    if not handles:
        return None

    def read() -> int | None:
        try:
            return sum(int(pynvml.nvmlDeviceGetMemoryInfo(h).used)
                       for h in handles)
        except Exception:  # noqa: BLE001
            return None
    return read


def _nvml_used_bytes() -> int | None:
    """One-shot convenience over :func:`_nvml_reader`."""
    read = _nvml_reader()
    return read() if read else None


class _VramSampler:
    """or-k10 — PEAK VRAM across an attempt, not just the before/after delta.

    A daemon thread samples every ``sample_ms`` while the attempt runs. The
    source is chosen once, at ``__enter__``: pynvml when it answers (the sum of
    ``used`` over all devices — a fleet-wide number, so ``source='nvml'``),
    otherwise the fleet's own meter via repeated :func:`_vram_snapshot` polls
    narrowed to ``worker`` by :func:`_vram_for` (``source='central'``). When
    neither answers the sampler is a clean no-op: ``peak_bytes`` None,
    ``sample_count`` 0, ``source`` None — the same honesty rule as the
    snapshot itself. ``sample_ms <= 0`` disables sampling without changing
    anything else.

    The thread is daemonic and joined on exit, so a hung poll cannot outlive
    the run; samples that fail are skipped, never counted."""

    def __init__(self, worker: str | None = None, *,
                 sample_ms: int | float | None = DEFAULT_VRAM_SAMPLE_MS,
                 nvml: Callable[[], int | None] | None = None,
                 snapshot: Callable[[], Mapping[str, Any] | None] | None = None
                 ) -> None:
        self.worker = worker
        self.sample_ms = float(sample_ms or 0)
        self._nvml: Callable[[], int | None] | None = nvml
        self._snapshot = snapshot
        self.source: str | None = None
        self.peak_bytes: int | None = None
        self.sample_count: int = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # -- reading ----------------------------------------------------------
    def _central_used(self) -> int | None:
        snap = (self._snapshot or _vram_snapshot)()
        row = _vram_for(snap, self.worker)
        used = row.get("vram_used") if row else None
        return int(used) if isinstance(used, (int, float)) else None

    def _read(self) -> int | None:
        if self.source == "nvml" and self._nvml is not None:
            return self._nvml()
        if self.source == "central":
            return self._central_used()
        return None

    def _record(self, used: int | None) -> None:
        if used is None:
            return
        with self._lock:
            self.sample_count += 1
            if self.peak_bytes is None or used > self.peak_bytes:
                self.peak_bytes = used

    def sample(self) -> int | None:
        """One synchronous sample (also what the thread calls)."""
        used = self._read()
        self._record(used)
        return used

    # -- lifecycle --------------------------------------------------------
    def _pick_source(self) -> None:
        if self._nvml is None:
            self._nvml = _nvml_reader()
        used = self._nvml() if self._nvml is not None else None
        if used is not None:
            self.source = "nvml"
            self._record(used)
            return
        used = self._central_used()
        if used is not None:
            self.source = "central"
            self._record(used)

    def _loop(self) -> None:
        interval = self.sample_ms / 1000.0
        while not self._stop.wait(interval):
            try:
                self.sample()
            except Exception as exc:  # noqa: BLE001 — never kill the attempt
                logger.debug("benchmark: vram sample failed (%s: %s)",
                             type(exc).__name__, exc)

    def start(self) -> "_VramSampler":
        if self.sample_ms <= 0:
            return self
        self._pick_source()
        if self.source is None:
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="oracle-vram-sampler")
        self._thread.start()
        return self

    def stop(self) -> "_VramSampler":
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_ms / 1000.0 * 5))
            self._thread = None
        if self.source is not None:
            try:
                self.sample()  # closing sample, so a short attempt still counts
            except Exception:  # noqa: BLE001
                pass
        return self

    def __enter__(self) -> "_VramSampler":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def perf_fields(self) -> dict[str, Any]:
        """The three :class:`PerfRecord` fields this sampler owns."""
        return {"vram_peak_bytes": self.peak_bytes,
                "vram_sample_count": self.sample_count,
                "vram_sampler": self.source}


def _load_state(model_key: str | None, worker: str | None) -> str | None:
    """"loaded" / "loading" / "cold" for this model on this worker, from the
    fleet's heartbeat record. None when nobody can say."""
    if not model_key or not worker:
        return None
    try:
        from hugpy_engine.placement import get_worker_registry
        from hugpy_oracle.providers import get_load_state_source
        # ``_selected_worker`` answers with a worker NAME when it has one, and
        # the fleet's load-state record is keyed by ID. Resolve rather than
        # report a false "unknown" for every worker that has a friendly name.
        worker_id = worker
        for row in (get_worker_registry().list_workers(online_only=False) or []):
            if worker in (row.get("id"), row.get("name")):
                worker_id = row.get("id") or worker
                break
        state = get_load_state_source().load_state(model_key, worker_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug("benchmark: load state unavailable (%s: %s)",
                     type(exc).__name__, exc)
        return None
    if not isinstance(state, Mapping):
        return None
    if state.get("healthy"):
        return "loaded"
    if state.get("in_progress"):
        return "loading"
    return "cold"


def _selected_worker(model_key: str | None, task: str | None) -> str | None:
    """Which worker the dispatcher intends to use, pre-dispatch. The receipt
    records None for an ``auto`` placement, so this is the only honest handle
    on "which box actually served it" available synchronously."""
    try:
        from hugpy_oracle import runtime
        return runtime._selected_worker(model_key, task, None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("benchmark: worker selection unavailable (%s: %s)",
                     type(exc).__name__, exc)
        return None


def _tokens_per_second(payload: Mapping[str, Any]) -> float | None:
    """tok/s from the runner's ``timings`` block, via the fleet's single
    sanctioned parser. None when the runner did not expose timings — which is
    the common case on the one-shot path and is recorded as None, never
    back-computed from wall time (that number would silently include model
    load and queue wait and would not be tok/s at all)."""
    try:
        from hugpy_engine.eviction import tok_s_from_timings
        value = tok_s_from_timings(dict(payload))
    except Exception:  # noqa: BLE001
        return None
    return float(value) if value else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchCheck:
    """One deterministic check with its evidence."""
    key: str
    passed: bool
    value: Any = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "passed": self.passed, "value": self.value,
                "detail": self.detail}


@dataclass(frozen=True, slots=True)
class DeterministicScore:
    """Layer (a): what can be measured without asking a model anything."""
    valid: bool = False
    error_count: int = 0
    preservation: float | None = None
    contradiction_rate: float = 0.0
    completeness: float | None = None
    constraint_adherence: float | None = None
    accuracy: float | None = None
    checks: tuple[BenchCheck, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def score(self) -> float:
        """The weighted deterministic score, 0-100. Absent axes are absent
        from BOTH sums (see ``_WEIGHTS`` in the module docstring)."""
        axes: dict[str, float] = {"valid": 1.0 if self.valid else 0.0,
                                  "clean": 1.0 - _clamp(self.contradiction_rate)}
        for name, value in (("preservation", self.preservation),
                            ("completeness", self.completeness),
                            ("constraints", self.constraint_adherence),
                            ("accuracy", self.accuracy)):
            if value is not None:
                axes[name] = _clamp(value)
        weight = sum(_WEIGHTS[k] for k in axes)
        if weight <= 0:
            return 0.0
        return round(100.0 * sum(_WEIGHTS[k] * v for k, v in axes.items())
                     / weight, 2)

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "error_count": self.error_count,
                "preservation": self.preservation,
                "contradiction_rate": self.contradiction_rate,
                "completeness": self.completeness,
                "constraint_adherence": self.constraint_adherence,
                "accuracy": self.accuracy, "score": self.score,
                "checks": [c.to_dict() for c in self.checks],
                "errors": list(self.errors)}


@dataclass(frozen=True, slots=True)
class JudgeScore:
    """Layer (b): the independent rubric judge. ``available=False`` is an
    honest outcome, never a zero."""
    judge_model: str | None = None
    verdict: str = "unavailable"
    score: float | None = None
    why: str = ""
    available: bool = False
    refused: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"judge_model": self.judge_model, "verdict": self.verdict,
                "score": self.score, "why": self.why,
                "available": self.available, "refused": self.refused,
                "detail": self.detail}


@dataclass(frozen=True, slots=True)
class PerfRecord:
    """What the attempt cost, and on what. Every unknown is None ON PURPOSE."""
    model: str | None = None
    worker: str | None = None
    load_state: str | None = None
    mode: str = "normalized"
    params: Mapping[str, Any] = field(default_factory=dict)
    ceiling_source: str = ""
    latency_s: float | None = None
    dispatch_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_s: float | None = None
    finish_reason: str | None = None
    output_chars: int = 0
    vram_before: Mapping[str, Any] | None = None
    vram_after: Mapping[str, Any] | None = None
    vram_used_delta_bytes: int | None = None
    #: or-k10 — peak observed by :class:`_VramSampler` during the attempt
    #: (bytes), how many samples backed it, and which meter ('nvml' |
    #: 'central' | None). None/0/None means "no sampler could answer".
    vram_peak_bytes: int | None = None
    vram_sample_count: int = 0
    vram_sampler: str | None = None
    gpu_total_bytes: int | None = None
    vram_reserve_gib: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "worker": self.worker,
                "load_state": self.load_state, "mode": self.mode,
                "params": dict(self.params),
                "ceiling_source": self.ceiling_source,
                "latency_s": self.latency_s,
                "dispatch_calls": self.dispatch_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "tokens_per_s": self.tokens_per_s,
                "finish_reason": self.finish_reason,
                "output_chars": self.output_chars,
                "vram_before": dict(self.vram_before) if self.vram_before else None,
                "vram_after": dict(self.vram_after) if self.vram_after else None,
                "vram_used_delta_bytes": self.vram_used_delta_bytes,
                "vram_peak_bytes": self.vram_peak_bytes,
                "vram_sample_count": self.vram_sample_count,
                "vram_sampler": self.vram_sampler,
                "gpu_total_bytes": self.gpu_total_bytes,
                "vram_reserve_gib": self.vram_reserve_gib}


@dataclass(frozen=True, slots=True)
class Attempt:
    """One (case, model, mode, repeat) row — the benchmark's unit of evidence."""
    case_id: str
    track: str
    operation: str
    model: str
    mode: str
    repeat: int = 0
    deterministic: DeterministicScore = field(default_factory=DeterministicScore)
    judge: JudgeScore = field(default_factory=JudgeScore)
    perf: PerfRecord = field(default_factory=PerfRecord)
    failure: str | None = None
    gap_code: str | None = None
    raw_ref: str = ""
    registry_version: str | None = None
    started_at: str = ""
    ended_at: str = ""

    @property
    def ok(self) -> bool:
        """A row counts as a SUCCESS only when a valid artifact came out. A
        dispatch that returned prose is a failure of the operation even though
        the fleet is perfectly healthy."""
        return self.failure is None and self.deterministic.valid

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "track": self.track,
                "operation": self.operation, "model": self.model,
                "mode": self.mode, "repeat": self.repeat, "ok": self.ok,
                "deterministic": self.deterministic.to_dict(),
                "judge": self.judge.to_dict(), "perf": self.perf.to_dict(),
                "failure": self.failure, "gap_code": self.gap_code,
                "raw_ref": self.raw_ref,
                "registry_version": self.registry_version,
                "started_at": self.started_at, "ended_at": self.ended_at}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

COMPLETION_SYSTEM: str = (
    "You are a screenwriter finishing a film. You will be given screenplay "
    "material that is partial, disconnected, contradictory or incomplete. "
    "Produce ONE ordered screenplay covering the WHOLE film as a single JSON "
    "object.\n"
    "RULES:\n"
    "1. Every line of dialogue you were given must appear VERBATIM, attributed "
    "to the speaker it was given to.\n"
    "2. Every scene heading is a slugline: INT./EXT. LOCATION - TIME OF DAY.\n"
    "3. A character may only speak in a scene they are present in "
    "(present_at_open or entrances).\n"
    "4. story_time_s only moves backwards after a flashback transition.\n"
    "5. Where the material contradicts itself, choose ONE version and write it. "
    "Never write both.\n"
    "6. Line ids are unique across the whole screenplay.")

WORKFLOW_SYSTEM: str = (
    "You are a first assistant director turning a LOCKED screenplay into "
    "production paperwork. Work only from the screenplay you are given: never "
    "invent a scene, a character or an id that is not in it. Answer with ONE "
    "JSON object and nothing else.")

_REPLY_ONLY_JSON = ("Return ONLY the JSON object. No markdown, no commentary, "
                    "no explanation, no code fence.")


def build_completion_prompt(case: BenchCase, preamble: str = "") -> str:
    """The Track A prompt: rules, the supplied material VERBATIM, the case's
    constraints as instructions, and the generated ``Screenplay`` schema — the
    same schema text ``author_screenplay`` shows, from the same generator, so
    the model is checked against exactly what it was shown.

    ``preamble`` (k109b, additive) is the stationary scenario block; omitted,
    the prompt is byte-identical to k109's."""
    lines = ([preamble, ""] if preamble else []) + [
        COMPLETION_SYSTEM, "", "SUPPLIED MATERIAL:", case.input_text, ""]
    if case.constraints:
        lines.append("HARD CONSTRAINTS — every one of these is checked:")
        lines.extend(f"- {c.sentence()}" for c in case.constraints)
        lines.append("")
    lines += ["JSON SCHEMA — your object MUST validate against this exactly:",
              schema_block(Screenplay), "", _REPLY_ONLY_JSON]
    return "\n".join(lines)


def build_workflow_prompt(case: BenchCase,
                          screenplay: Screenplay = FIXTURE_SCREENPLAY,
                          preamble: str = "") -> str:
    """The Track C prompt for one operation: the locked screenplay as canonical
    JSON (a paraphrase is where the ids stop matching), the operation's
    instruction, the required row shape, and one shape example.

    ``preamble`` (k109b, additive, default "") is prepended VERBATIM before the
    system line. It exists so the stationary sweep can hand every model the
    same tone value, character sheets and continuity facts without this module
    growing a second, drifting copy of the operation's question — the
    instruction, the shape and the example still come from the OperationSpec.
    Omitted, the prompt is byte-identical to k109's."""
    spec = case.spec
    ground = _ground_truth(spec, screenplay)
    lines = ([preamble, ""] if preamble else []) + [
             WORKFLOW_SYSTEM, "",
             f"OPERATION: {spec.artifact}", spec.instruction, "",
             "THE LOCKED SCREENPLAY:",
             json.dumps(screenplay.to_dict(), sort_keys=True, indent=2), ""]
    if spec.coverage == "shot_ids" and ground:
        lines += [f"THE SHOT IDS TO COVER (one row each, in this order): "
                  f"{list(ground)}", ""]
    elif spec.coverage == "scene_ids":
        lines += [f"THE SCENE IDS TO COVER (one row each): "
                  f"{list(screenplay.scene_ids)}", ""]
    if case.constraints:
        lines.append("HARD CONSTRAINTS — every one of these is checked:")
        lines.extend(f"- {c.sentence()}" for c in case.constraints)
        lines.append("")
    row = ", ".join(f'"{f}"' for f in spec.row_fields)
    lines += [f'SHAPE: one object with the key "{spec.container}" holding a '
              f"list; every row carries {row}.",
              f"EXAMPLE (shape only, not content): {spec.example}", "",
              _REPLY_ONLY_JSON]
    return "\n".join(lines)


def build_prompt(case: BenchCase, preamble: str = "",
                 screenplay: Screenplay = FIXTURE_SCREENPLAY) -> str:
    """The prompt for any case. Track B goes through k110's own
    ``build_plot_prompt`` (via ``author_plot``) and therefore has no prompt of
    its own here — asking the benchmark's model a DIFFERENT question than the
    pipeline asks would measure the benchmark, not the fleet.

    ``screenplay`` (k109b, additive) is the Track C source. It defaults to
    k109's fixture, so every existing call is unchanged; the stationary sweep
    passes its own canonical screenplay, and it MUST, because a Track C prompt
    quotes the shot ids ``build_shot_plan`` derives and those ids differ
    between the two screenplays."""
    if case.track == "A":
        return build_completion_prompt(case, preamble)
    if case.track == "C":
        return build_workflow_prompt(case, screenplay, preamble=preamble)
    from hugpy_oracle.screenplay import build_plot_prompt
    mode = case.plot_mode or plot_input_mode(case.input_text)
    prompt = build_plot_prompt(case.input_text, mode)
    return f"{preamble}\n\n{prompt}" if preamble else prompt


def _ground_truth(spec: OperationSpec,
                  screenplay: Screenplay) -> tuple[str, ...]:
    """The id set a Track C answer must cover — DERIVED from the screenplay by
    the same functions the pipeline uses, never hand-listed."""
    if spec.coverage == "scene_ids":
        return screenplay.scene_ids
    if spec.coverage == "line_ids":
        return screenplay.line_ids
    if spec.coverage == "shot_ids":
        return build_shot_plan(screenplay).segment_ids
    return ()


# ---------------------------------------------------------------------------
# Deterministic scoring — layer (a)
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_HARD_CONTRADICTION_MARKERS = (
    "is not in the room", "does not appear", "before", "story time",
    "caused by", "causes itself", "continuous", "not present",
)


def _norm(text: Any) -> str:
    return _WS.sub(" ", str(text or "")).strip().lower()


def _has(haystack: str, needle: str) -> bool:
    needle = _norm(needle)
    return bool(needle) and needle in haystack


def _payload(result: Any) -> dict[str, Any]:
    """A dispatch result as a plain dict. Accepts the pydantic ``TaskResult``
    the fleet returns, a plain mapping (what a fake dispatch returns in tests)
    and anything else (str-ified into ``text``)."""
    if isinstance(result, Mapping):
        return dict(result)
    try:
        from hugpy_oracle import runtime
        payload = runtime._result_payload(result)
        if isinstance(payload, dict):
            return payload
    except Exception:  # noqa: BLE001
        pass
    return {"text": str(result)}


def _facts(kind: str, artifact: Any, blob: str) -> dict[str, Any]:
    """The structural facts a constraint reads, per artifact kind. Everything
    a constraint cannot see structurally falls back to the blob, and a fallback
    is recorded in the check's detail rather than hidden."""
    out: dict[str, Any] = {"blob": blob, "scenes": None, "beats": None,
                           "locations": set(), "characters": set(),
                           "transitions": set(), "times": set()}
    if isinstance(artifact, Screenplay):
        out["scenes"] = len(artifact.scenes)
        out["locations"] = {_norm(s.location) for s in artifact.scenes}
        out["characters"] = {_norm(c) for c in artifact.characters}
        out["transitions"] = {s.transition for s in artifact.scenes}
        out["times"] = {_norm(s.time_of_day) for s in artifact.scenes}
    elif isinstance(artifact, PlotSpec):
        out["beats"] = len(artifact.beats)
        out["characters"] = {_norm(c) for c in artifact.character_names}
        out["locations"] = {_norm(b.location) for b in artifact.beats
                            if b.location}
        out["times"] = {_norm(b.time_of_day) for b in artifact.beats
                        if b.time_of_day}
    elif isinstance(artifact, Mapping):
        rows = [r for v in artifact.values() if isinstance(v, list)
                for r in v if isinstance(r, Mapping)]
        out["locations"] = {_norm(r.get("location")) for r in rows
                            if r.get("location")}
    return out


def check_constraint(constraint: Constraint, facts: Mapping[str, Any]) -> BenchCheck:
    """One constraint against the produced artifact. A constraint whose axis
    the artifact cannot answer structurally is checked against the text blob;
    a counting constraint with no count available FAILS, because "we could not
    count the scenes" is not adherence."""
    kind, value, blob = constraint.kind, constraint.value, facts["blob"]

    def numeric(available: Any, ok: Callable[[int], bool]) -> BenchCheck:
        if available is None:
            return BenchCheck(constraint.key, False, None,
                              "no structural count available for this artifact")
        passed = ok(int(available))
        return BenchCheck(constraint.key, passed, int(available),
                          f"{kind}={value}, measured {available}")

    if kind == "max_scenes":
        return numeric(facts["scenes"], lambda n: n <= int(value))
    if kind == "min_scenes":
        return numeric(facts["scenes"], lambda n: n >= int(value))
    if kind == "max_beats":
        return numeric(facts["beats"], lambda n: n <= int(value))
    if kind == "min_beats":
        return numeric(facts["beats"], lambda n: n >= int(value))
    if kind == "max_locations":
        seen = {loc for loc in facts["locations"] if loc}
        return numeric(len(seen) if seen else None, lambda n: n <= int(value))
    if kind == "max_characters":
        seen = {name for name in facts["characters"] if name}
        return numeric(len(seen) if seen else None, lambda n: n <= int(value))
    if kind == "forbidden_term":
        hit = _has(blob, value)
        return BenchCheck(constraint.key, not hit, not hit,
                          f"{value!r} {'appears' if hit else 'is absent'}")
    if kind == "requires_term":
        hit = _has(blob, value)
        return BenchCheck(constraint.key, hit, hit,
                          f"{value!r} {'appears' if hit else 'is missing'}")
    if kind == "requires_character":
        hit = _norm(value) in facts["characters"] or _has(blob, value)
        return BenchCheck(constraint.key, hit, hit, f"character {value!r}")
    if kind == "requires_location":
        hit = any(_norm(value) in loc for loc in facts["locations"]) \
            or _has(blob, value)
        return BenchCheck(constraint.key, hit, hit, f"location {value!r}")
    if kind == "requires_transition":
        hit = value in facts["transitions"] or _has(blob, value)
        return BenchCheck(constraint.key, hit, hit, f"transition {value!r}")
    if kind == "requires_time_of_day":
        hit = any(_norm(value) in t for t in facts["times"]) or _has(blob, value)
        return BenchCheck(constraint.key, hit, hit, f"time of day {value!r}")
    return BenchCheck(constraint.key, False, None,
                      f"unknown constraint kind {kind!r}")


def _constraint_adherence(case: BenchCase, artifact: Any, blob: str
                          ) -> tuple[float | None, tuple[BenchCheck, ...]]:
    if not case.constraints:
        return None, ()
    facts = _facts(case.track, artifact, blob)
    checks = tuple(check_constraint(c, facts) for c in case.constraints)
    met = sum(1 for c in checks if c.passed)
    summary = BenchCheck("constraints_met", met == len(checks),
                         round(met / len(checks), 4),
                         f"{met}/{len(checks)} constraints honoured")
    return met / len(checks), checks + (summary,)


def _preservation(supplied: Sequence[str], blob: str
                  ) -> tuple[float | None, tuple[str, ...]]:
    """Fraction of the operator's own material that survived, and what did
    not. Matching is whole-string containment on normalized text: a model that
    reworded a line the operator called FINAL did not preserve it."""
    items = [s for s in supplied if str(s).strip()]
    if not items:
        return None, ()
    missing = tuple(s for s in items if not _has(blob, s))
    return (len(items) - len(missing)) / len(items), missing


def _soft_contradictions(case: BenchCase, scopes: Sequence[str]
                         ) -> tuple[int, tuple[str, ...]]:
    """Declared contradiction pairs that SURVIVED into one scope.

    A heuristic, and labelled one: both sides of a contradiction appearing in
    the same scene is strong evidence the model carried both drafts instead of
    choosing. Both sides in DIFFERENT scenes is not counted — a film may
    legitimately show a radio working and later show it smashed."""
    hits: list[str] = []
    for item in case.contradictions:
        for scope in scopes:
            if any(_has(scope, l) for l in item.left) and \
                    any(_has(scope, r) for r in item.right):
                hits.append(f"{item.key}: both sides present in one scope "
                            f"({item.description})")
                break
    return len(hits), tuple(hits)


def _rate(found: int, checks: int) -> float:
    return round(found / checks, 4) if checks > 0 else 0.0


def score_screenplay(case: BenchCase, result: Any, raw: str
                     ) -> DeterministicScore:
    """Track A: the k110 ``Screenplay`` validator plus the doc's measures."""
    valid = isinstance(result, Screenplay)
    errors = tuple(result.errors) if isinstance(result, AuthoringGap) else ()
    if valid:
        artifact: Any = result
        blob = _norm(json.dumps(result.to_dict(), sort_keys=True))
        scopes = [_norm(json.dumps(s.to_dict(), sort_keys=True))
                  for s in result.scenes]
    else:
        parsed, _why = parse_json_object(raw)
        artifact = parsed
        blob = _norm(raw)
        scopes = [blob]

    checks = [BenchCheck("validates", valid, valid,
                         "built a k110 Screenplay" if valid
                         else f"{len(errors)} validator error(s)")]

    preservation, missing = _preservation(
        tuple(case.supplied_lines) + tuple(case.supplied_beats), blob)
    if preservation is not None:
        checks.append(BenchCheck(
            "preserves_supplied", not missing, round(preservation, 4),
            "all supplied material present" if not missing
            else f"missing: {list(missing)[:4]}"))

    hard = 0
    detail = "no contradiction found"
    if valid:
        breaks = chain_breaks(build_continuity(result))
        hard = 1 if breaks else 0
        detail = f"continuity chain breaks: {list(breaks)[:4]}" if breaks \
            else "continuity chain is sound"
    elif errors:
        offenders = [e for e in errors
                     if any(m in e.lower() for m in _HARD_CONTRADICTION_MARKERS)]
        hard = 1 if offenders else 0
        detail = f"validator contradictions: {offenders[:3]}" if offenders \
            else "no contradiction among the validator errors"
    checks.append(BenchCheck("no_hard_contradictions", hard == 0, hard, detail))

    soft, soft_detail = _soft_contradictions(case, scopes)
    if case.contradictions:
        checks.append(BenchCheck("contradictions_resolved", soft == 0, soft,
                                 "; ".join(soft_detail) or
                                 "every contradicted fact resolved to one version"))
    rate = _rate(hard + soft, 1 + len(case.contradictions))

    if valid:
        axes = {
            "title": bool(result.title.strip()),
            "two_or_more_scenes": len(result.scenes) >= 2,
            "action_everywhere": all(s.action.strip() for s in result.scenes),
            "cast_declared": bool(result.characters),
            "has_dialogue": bool(result.line_ids),
        }
    elif isinstance(artifact, Mapping):
        scenes = artifact.get("scenes") if isinstance(artifact.get("scenes"), list) else []
        axes = {
            "title": bool(str(artifact.get("title") or "").strip()),
            "two_or_more_scenes": len(scenes) >= 2,
            "action_everywhere": bool(scenes) and all(
                isinstance(s, Mapping) and str(s.get("action") or "").strip()
                for s in scenes),
            "cast_declared": bool(artifact.get("characters")),
            "has_dialogue": any(isinstance(s, Mapping) and s.get("dialogue")
                                for s in scenes),
        }
    else:
        axes = {"parsed": False}
    completeness = sum(1 for v in axes.values() if v) / len(axes)
    checks.append(BenchCheck("complete_artifact", completeness == 1.0,
                             round(completeness, 4),
                             ", ".join(f"{k}={'y' if v else 'n'}"
                                       for k, v in axes.items())))

    adherence, constraint_checks = _constraint_adherence(case, artifact, blob)
    return DeterministicScore(
        valid=valid, error_count=len(errors), preservation=preservation,
        contradiction_rate=rate, completeness=completeness,
        constraint_adherence=adherence,
        checks=tuple(checks) + constraint_checks, errors=errors)


def score_plot(case: BenchCase, result: Any, raw: str) -> DeterministicScore:
    """Track B: the k110 ``PlotSpec`` validator plus the doc's measures."""
    valid = isinstance(result, PlotSpec)
    errors = tuple(result.errors) if isinstance(result, AuthoringGap) else ()
    if valid:
        artifact: Any = result
        blob = _norm(json.dumps(result.to_dict(), sort_keys=True))
    else:
        parsed, _why = parse_json_object(raw)
        artifact = parsed
        blob = _norm(raw)

    checks = [BenchCheck("validates", valid, valid,
                         "built a k110 PlotSpec" if valid
                         else f"{len(errors)} validator error(s)")]

    preservation, missing = _preservation(case.supplied_beats, blob)
    if preservation is not None:
        checks.append(BenchCheck(
            "preserves_supplied", not missing, round(preservation, 4),
            "all supplied material present" if not missing
            else f"missing: {list(missing)[:4]}"))

    if valid:
        checks.append(BenchCheck("causal_logic", True, True,
                                 f"{len(result.beats)} beats, roots="
                                 f"{list(result.roots)}"))
        hard = 0
    else:
        offenders = [e for e in errors
                     if "caused by" in e.lower() or "causes itself" in e.lower()
                     or "appear in no beat" in e.lower()]
        hard = 1 if offenders else 0
        checks.append(BenchCheck("causal_logic", not offenders, not offenders,
                                 f"{offenders[:3]}" if offenders
                                 else "no causal error among the validator errors"))
    soft, soft_detail = _soft_contradictions(case, [blob])
    checks.append(BenchCheck("no_hard_contradictions", hard + soft == 0,
                             hard + soft,
                             "; ".join(soft_detail) or "no contradiction found"))
    rate = _rate(hard + soft, 1 + len(case.contradictions))

    if valid:
        axes = {
            "premise": bool(result.premise.strip()),
            "genre_and_tone": bool(result.genre.strip() and result.tone.strip()),
            "ending": bool(result.ending.strip()),
            "two_or_more_characters": len(result.characters) >= 2,
            "four_or_more_beats": len(result.beats) >= 4,
            "turning_point": bool(result.turning_points),
        }
    elif isinstance(artifact, Mapping):
        chars = artifact.get("characters") or []
        beats = artifact.get("beats") or []
        axes = {
            "premise": bool(str(artifact.get("premise") or "").strip()),
            "genre_and_tone": bool(str(artifact.get("genre") or "").strip()
                                   and str(artifact.get("tone") or "").strip()),
            "ending": bool(str(artifact.get("ending") or "").strip()),
            "two_or_more_characters": isinstance(chars, list) and len(chars) >= 2,
            "four_or_more_beats": isinstance(beats, list) and len(beats) >= 4,
            "turning_point": isinstance(beats, list) and any(
                isinstance(b, Mapping) and b.get("turning_point") for b in beats),
        }
    else:
        axes = {"parsed": False}
    completeness = sum(1 for v in axes.values() if v) / len(axes)
    checks.append(BenchCheck("complete_artifact", completeness == 1.0,
                             round(completeness, 4),
                             ", ".join(f"{k}={'y' if v else 'n'}"
                                       for k, v in axes.items())))

    adherence, constraint_checks = _constraint_adherence(case, artifact, blob)
    return DeterministicScore(
        valid=valid, error_count=len(errors), preservation=preservation,
        contradiction_rate=rate, completeness=completeness,
        constraint_adherence=adherence,
        checks=tuple(checks) + constraint_checks, errors=errors)


# --- Track C ---------------------------------------------------------------


def workflow_errors(spec: OperationSpec, obj: Any,
                    ground: Sequence[str]) -> tuple[str, ...]:
    """Every shape problem in a Track C answer, verbatim enough to reprompt on.

    This is Track C's validator, and it is deliberately the SAME discipline as
    k110's: report all of it, then let one bounded repair use it."""
    problems: list[str] = []
    if not isinstance(obj, Mapping):
        return (f"the reply is not a JSON object with a "
                f"{spec.container!r} list",)
    for field_name in spec.top_fields:
        if field_name not in obj:
            problems.append(f"required top-level field {field_name!r} is missing")
    rows = obj.get(spec.container)
    if not isinstance(rows, list) or not rows:
        problems.append(f"{spec.container!r} must be a non-empty list of objects")
        return tuple(problems)
    known = set(ground)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            problems.append(f"{spec.container}[{index}] is not an object")
            continue
        for field_name in spec.row_fields:
            if field_name not in row or row.get(field_name) in (None, "", [], {}):
                problems.append(
                    f"{spec.container}[{index}] is missing {field_name!r}")
        row_id = str(row.get(spec.id_field) or "")
        if known and row_id and row_id not in known:
            problems.append(
                f"{spec.container}[{index}] has {spec.id_field}={row_id!r}, "
                f"which the screenplay does not contain (has: {sorted(known)})")
    if known:
        covered = {str(r.get(spec.id_field)) for r in rows
                   if isinstance(r, Mapping)}
        for missing in sorted(known - covered):
            problems.append(f"no {spec.container} row covers "
                            f"{spec.id_field}={missing!r}")
    return tuple(problems)


def _continuity_accuracy(rows: Sequence[Mapping[str, Any]],
                         screenplay: Screenplay) -> tuple[float, str]:
    """Extracted continuity vs the bible ``build_continuity`` DERIVES from the
    same screenplay — an answer key nobody typed by hand."""
    bible = build_continuity(screenplay)
    truth = {e.segment_id: e for e in bible.entries}
    if not truth:
        return 0.0, "no derived continuity to compare against"
    hits, notes = 0, []
    for segment_id, entry in truth.items():
        row = next((r for r in rows
                    if str(r.get("segment_id")) == segment_id), None)
        if row is None:
            notes.append(f"{segment_id}: absent")
            continue
        ok = True
        for side, state in (("state_before", entry.state_before),
                            ("state_after", entry.state_after)):
            got = row.get(side)
            if not isinstance(got, Mapping):
                ok = False
                continue
            want_present = {_norm(n) for n in (state.get("present") or ())}
            got_present = {_norm(n) for n in (got.get("present") or ())} \
                if isinstance(got.get("present"), (list, tuple)) else set()
            if want_present != got_present:
                ok = False
                notes.append(f"{segment_id}.{side}: present {sorted(got_present)} "
                             f"!= {sorted(want_present)}")
            if _norm(got.get("location")) != _norm(state.get("location")):
                ok = False
                notes.append(f"{segment_id}.{side}: location mismatch")
        hits += 1 if ok else 0
    return hits / len(truth), "; ".join(notes[:4]) or "matches the derived bible"


def _shotlist_accuracy(rows: Sequence[Mapping[str, Any]],
                       screenplay: Screenplay) -> tuple[float, str]:
    plan = build_shot_plan(screenplay)
    truth = {d.segment_id: d for d in plan.designs}
    if not truth:
        return 0.0, "no derived shot plan"
    hits, notes = 0, []
    for segment_id, design in truth.items():
        row = next((r for r in rows
                    if str(r.get("segment_id")) == segment_id), None)
        if row is None:
            notes.append(f"{segment_id}: absent")
            continue
        got_lines = row.get("line_ids")
        got = tuple(str(x) for x in got_lines) if isinstance(got_lines, (list, tuple)) else ()
        if got == design.line_ids and str(row.get("scene_id")) == design.scene_id:
            hits += 1
        else:
            notes.append(f"{segment_id}: lines {list(got)} vs "
                         f"{list(design.line_ids)}")
    return hits / len(truth), "; ".join(notes[:4]) or "matches the derived plan"


def _timeline_accuracy(rows: Sequence[Mapping[str, Any]]) -> tuple[float, str]:
    """An assembly plan is a PARTITION or it is not: ordered, gapless, no
    overlaps. Measured over adjacent pairs so a plan that is right except for
    one join scores as such."""
    windows: list[tuple[float, float]] = []
    for row in rows:
        try:
            windows.append((float(row.get("start_s")), float(row.get("end_s"))))
        except (TypeError, ValueError):
            return 0.0, "a window is not numeric"
    if len(windows) < 2:
        return (1.0, "single window") if windows and windows[0][1] >= windows[0][0] \
            else (0.0, "no usable window")
    good, notes = 0, []
    for (a_start, a_end), (b_start, b_end) in zip(windows, windows[1:]):
        if a_end < a_start or b_end < b_start:
            notes.append("a window ends before it starts")
            continue
        if abs(b_start - a_end) <= 0.05:
            good += 1
        else:
            notes.append(f"join {a_end}->{b_start} is a "
                         f"{'gap' if b_start > a_end else 'overlap'}")
    return good / (len(windows) - 1), "; ".join(notes[:4]) or "gapless partition"


def _prompt_grounding(rows: Sequence[Mapping[str, Any]],
                      screenplay: Screenplay) -> tuple[float, str, int]:
    """Storyboard / segment prompts: does each prompt describe ITS OWN scene,
    and does it leak another scene's location? The leak count is returned
    separately because a leak is a CONTRADICTION (the pipeline's rule is that
    segments share a source, never a chain), not merely a weak prompt."""
    plan = build_shot_plan(screenplay)
    by_segment = {d.segment_id: d for d in plan.designs}
    grounded, leaks, notes = 0, 0, []
    scored = 0
    for row in rows:
        segment_id = str(row.get("segment_id") or "")
        design = by_segment.get(segment_id)
        if design is None:
            continue
        scored += 1
        scene = screenplay.scene(design.scene_id)
        text = _norm(row.get("prompt"))
        own = [scene.location] + list(scene.names)
        if any(_has(text, term) for term in own if term):
            grounded += 1
        else:
            notes.append(f"{segment_id}: names neither its location nor its cast")
        others = [s.location for s in screenplay.scenes
                  if s.scene_id != design.scene_id
                  and _norm(s.location) != _norm(scene.location)]
        if any(_has(text, loc) for loc in others):
            leaks += 1
            notes.append(f"{segment_id}: names another scene's location")
    if scored == 0:
        return 0.0, "no row matched a derived shot", 0
    return grounded / scored, "; ".join(notes[:4]) or "every prompt is grounded", leaks


#: k109b -> k109 accuracy scorer aliases. The stationary wave's operations are
#: NEW matrix keys with the SAME measurable shape as four k109 operations, so
#: they are scored by the k109 scorer rather than by a second copy of it. A
#: k109 operation maps to itself by absence, which is why every existing branch
#: below is untouched and every existing row scores identically.
WORKFLOW_ACCURACY_ALIAS: dict[str, str] = {
    "continuity.bible": "continuity.extract",
    "screenplay.breakdown": "breakdown.script",
    "shots.design": "shotlist.build",
    "segment.compile-prompt": "segment.prompts",
}


def _stationary_correction(obj: Any) -> tuple[tuple[str, ...], dict[str, Any]]:
    """``stationary_scenario.validate_correction_notes``, lazily. Lazy because
    that module builds a Screenplay at import and this one must stay importable
    by a k109 consumer that never asks about the stationary wave."""
    from hugpy_oracle.stationary_scenario import validate_correction_notes
    return validate_correction_notes(obj)


def _stationary_timeline(obj: Any) -> tuple[tuple[str, ...], dict[str, Any]]:
    """``stationary_scenario.validate_timeline``, lazily (same reason)."""
    from hugpy_oracle.stationary_scenario import validate_timeline
    return validate_timeline(obj)


def score_workflow(case: BenchCase, obj: Any, raw: str,
                   screenplay: Screenplay = FIXTURE_SCREENPLAY
                   ) -> DeterministicScore:
    """Track C: shape validation, coverage of the DERIVED id set, hallucinated
    ids as contradictions, and — where an answer key exists — accuracy."""
    spec = case.spec
    ground = _ground_truth(spec, screenplay)
    if not isinstance(obj, Mapping):
        parsed, _why = parse_json_object(raw)
        obj = parsed
    errors = workflow_errors(spec, obj, ground)
    rows: list[Mapping[str, Any]] = []
    if isinstance(obj, Mapping) and isinstance(obj.get(spec.container), list):
        rows = [r for r in obj[spec.container] if isinstance(r, Mapping)]
    valid = not errors
    blob = _norm(json.dumps(obj, sort_keys=True, default=str)) if obj else _norm(raw)

    checks = [BenchCheck("validates", valid, valid,
                         f"{len(errors)} shape error(s)" if errors
                         else f"{len(rows)} well-formed row(s)")]

    covered = {str(r.get(spec.id_field)) for r in rows}
    preservation = (len([g for g in ground if g in covered]) / len(ground)
                    if ground else None)
    if preservation is not None:
        checks.append(BenchCheck(
            "covers_source", preservation == 1.0, round(preservation, 4),
            f"{len([g for g in ground if g in covered])}/{len(ground)} "
            f"{spec.coverage} covered"))

    field_hits = sum(
        1 for r in rows for f in spec.row_fields
        if f in r and r.get(f) not in (None, "", [], {}))
    field_total = max(1, len(rows) * max(1, len(spec.row_fields)))
    top_ok = all(f in (obj or {}) for f in spec.top_fields) if spec.top_fields else True
    row_completeness = field_hits / field_total
    completeness = row_completeness if not spec.top_fields else \
        0.5 * row_completeness + 0.5 * (1.0 if top_ok else 0.0)
    checks.append(BenchCheck("complete_artifact", completeness == 1.0,
                             round(completeness, 4),
                             f"{field_hits}/{field_total} required row fields"))

    unknown = [str(r.get(spec.id_field)) for r in rows
               if ground and str(r.get(spec.id_field)) not in set(ground)]
    accuracy: float | None = None
    accuracy_detail = ""
    leaks = 0
    scored_as = WORKFLOW_ACCURACY_ALIAS.get(case.operation, case.operation)
    if scored_as == "correction.notes":
        problems, facts = _stationary_correction(obj)
        accuracy = facts.get("accuracy")
        accuracy_detail = (f"{facts.get('covered_failing')}/"
                           f"{facts.get('failing_total')} failing check(s) "
                           f"corrected, {facts.get('spurious')} spurious, "
                           f"{facts.get('chained')} chained")
        checks.append(BenchCheck("one_note_per_failure", not problems,
                                 round(accuracy or 0.0, 4), accuracy_detail))
        errors = tuple(errors) + problems
        valid = not errors
    elif scored_as == "postproduction.plan":
        problems, facts = _stationary_timeline(obj)
        accuracy = facts.get("accuracy")
        accuracy_detail = (f"{facts.get('covered')}/{facts.get('expected')} "
                           f"segment(s) laid, {facts.get('gaps')} gap(s), "
                           f"{facts.get('overlaps')} overlap(s), export "
                           f"{'present' if facts.get('export_present') else 'MISSING'}")
        checks.append(BenchCheck("timeline_partition", not problems,
                                 round(accuracy or 0.0, 4), accuracy_detail))
        errors = tuple(errors) + problems
        valid = not errors
    elif scored_as == "continuity.extract":
        accuracy, accuracy_detail = _continuity_accuracy(rows, screenplay)
        checks.append(BenchCheck("continuity_accuracy", accuracy == 1.0,
                                 round(accuracy, 4), accuracy_detail))
    elif scored_as == "shotlist.build":
        accuracy, accuracy_detail = _shotlist_accuracy(rows, screenplay)
        checks.append(BenchCheck("shotlist_accuracy", accuracy == 1.0,
                                 round(accuracy, 4), accuracy_detail))
    elif scored_as == "assembly.plan":
        accuracy, accuracy_detail = _timeline_accuracy(rows)
        checks.append(BenchCheck("timeline_partition", accuracy == 1.0,
                                 round(accuracy, 4), accuracy_detail))
    elif scored_as in ("storyboard.prompts", "segment.prompts"):
        accuracy, accuracy_detail, leaks = _prompt_grounding(rows, screenplay)
        checks.append(BenchCheck("prompt_grounding", accuracy == 1.0,
                                 round(accuracy, 4), accuracy_detail))
        checks.append(BenchCheck("no_cross_segment_leak", leaks == 0, leaks,
                                 f"{leaks} prompt(s) name another scene's "
                                 f"location"))

    bad = len(unknown) + leaks
    rate = _rate(bad, max(1, len(rows)))
    checks.append(BenchCheck("no_hard_contradictions", bad == 0, bad,
                             f"hallucinated ids: {unknown[:4]}" if unknown
                             else ("segment prompts leak across scenes"
                                   if leaks else "no invented id, no leak")))

    adherence, constraint_checks = _constraint_adherence(case, obj, blob)
    return DeterministicScore(
        valid=valid, error_count=len(errors), preservation=preservation,
        contradiction_rate=rate, completeness=completeness,
        constraint_adherence=adherence, accuracy=accuracy,
        checks=tuple(checks) + constraint_checks, errors=errors)


def score_case(case: BenchCase, result: Any, raw: str,
               screenplay: Screenplay = FIXTURE_SCREENPLAY
               ) -> DeterministicScore:
    """Deterministic scoring for any case, dispatched on track."""
    if case.track == "A":
        return score_screenplay(case, result, raw)
    if case.track == "B":
        return score_plot(case, result, raw)
    return score_workflow(case, result, raw, screenplay)


# ---------------------------------------------------------------------------
# Judge scoring — layer (b)
# ---------------------------------------------------------------------------

#: One rubric per track, carrying the doc's own measures for that track. The
#: judge is asked for ONE number under the k90c reply discipline, because a
#: judge asked for eight numbers returns eight numbers that all say 70.
JUDGE_RUBRICS: dict[str, str] = {
    "A": ("Judge a SCREENPLAY COMPLETION. Weigh: narrative coherence; "
          "preservation of the supplied material; causal and temporal "
          "consistency; character continuity; dialogue consistency; quality "
          "of the transitions between supplied and new material; freedom from "
          "contradiction; completeness; adherence to the stated constraints."),
    "B": ("Judge a CONSTRUCTED PLOT. Weigh: plot structure; character "
          "motivation; causal logic; originality; thematic consistency; "
          "quality of the character arcs; whether the setup and conflict are "
          "resolved; suitability for conversion into a screenplay."),
    "C": ("Judge a PRODUCTION ARTIFACT generated from a locked screenplay. "
          "Weigh: fidelity to the screenplay; actionability on a shooting "
          "day; completeness; internal consistency; whether a downstream "
          "generator or crew could use it AS WRITTEN with no further "
          "interpretation."),
}

JUDGE_REPLY_FORMAT: str = ("Reply exactly: VERDICT=YES|NO; SCORE=0-100; "
                           "WHY=<one sentence>.")
JUDGE_MAX_TOKENS: int = 120
_JUDGE_SOURCE_CHARS: int = 4000
_JUDGE_OUTPUT_CHARS: int = 8000


def _no_think(prompt: str) -> str:
    try:
        from hugpy_engine.utils.no_think import with_no_think
        return with_no_think(prompt)
    except Exception:  # noqa: BLE001 — a missing helper must not stop a run
        return prompt


def _strip_think(text: str) -> str:
    try:
        from hugpy_engine.utils.no_think import strip_think
        return strip_think(text)[0]
    except Exception:  # noqa: BLE001
        return text


def build_judge_prompt(case: BenchCase, output_text: str,
                       source_text: str = "") -> str:
    """The rubric prompt: the case's own source material, the candidate's
    answer, the track rubric, and k90c's reply discipline."""
    source = source_text or case.input_text or \
        json.dumps(FIXTURE_SCREENPLAY.to_dict(), sort_keys=True)
    checklist = "\n".join(
        f"- {e.description}" for e in case.expectations if e.layer == "judge")
    return (
        f"{JUDGE_RUBRICS[case.track]}\n\n"
        f"THE REQUEST ({case.condition}):\n{source[:_JUDGE_SOURCE_CHARS]}\n\n"
        f"THE ANSWER UNDER JUDGEMENT:\n{output_text[:_JUDGE_OUTPUT_CHARS]}\n\n"
        + (f"Pay particular attention to:\n{checklist}\n\n" if checklist else "")
        + JUDGE_REPLY_FORMAT)


def pick_judge(candidate_model: str, route: Any,
               preferred: str | None = None) -> tuple[str | None, str]:
    """The judge model for this attempt — never the candidate.

    Returns ``(model_id, why)``. A fleet whose ONLY eligible text model is the
    candidate gets ``(None, "refused")``: self-judging is the one degradation
    this benchmark will not accept, because unlike a missing judge it produces
    a number that looks exactly like a real one.

    ``preferred`` is an OPERATOR PIN (``--judge-model``), and it exists for
    politeness: on a fleet with eighty eligible models, "sorted-first" can mean
    cold-loading a 23B model to grade a 3B one. The pin is still verified
    against the catalog's eligible set — it selects, it never bypasses. A pin
    that IS the candidate is REFUSED rather than silently swapped: the operator
    asked for one specific judge, and quietly substituting another (possibly
    enormous, possibly cold) model would be a different experiment wearing this
    one's name.

    Without a pin the choice is the first eligible id in sorted order, so two
    runs on an unchanged fleet judge with the SAME model and are comparable."""
    if route is None or getattr(route, "execution", None) != "execute":
        reasons = "; ".join(getattr(route, "reasons", ()) or ()) or \
            "no judge route"
        return None, f"unavailable: {reasons}"
    pool = [m for m in (getattr(route, "model_ids", ()) or ()) if m]
    if not pool and getattr(route, "model_id", None):
        pool = [route.model_id]
    others = sorted({m for m in pool if m != candidate_model})
    if preferred:
        if preferred == candidate_model:
            return None, (f"refused: the pinned judge IS the candidate "
                          f"({candidate_model}) — a model may not grade its "
                          f"own work, and the pin is not silently substituted")
        if preferred in pool:
            return preferred, ("operator-pinned judge, verified eligible and "
                               "not the candidate")
        if not others:
            return None, (f"refused: pinned judge {preferred!r} is not eligible "
                          f"here and the only eligible model IS the candidate")
        return others[0], (f"pinned judge {preferred!r} is not in the eligible "
                           f"set — fell back to the sorted-first independent "
                           f"judge")
    if not others:
        return None, (f"refused: the only eligible judge on this fleet IS the "
                      f"candidate ({candidate_model}) — a model may not grade "
                      f"its own work")
    return others[0], "independent judge, sorted-first of the eligible set"


def judge_attempt(case: BenchCase, output_text: str, candidate_model: str, *,
                  deadline_s: float = 90.0, source_text: str = "",
                  preferred_judge: str | None = None) -> JudgeScore:
    """One rubric pass by an independent model. NEVER raises: a judge fault is
    ``available=False``, which the report prints as "unjudged"."""
    if not str(output_text or "").strip():
        return JudgeScore(detail="nothing to judge — the candidate produced "
                                 "no text")
    route = _resolve_route(BENCH_CAPABILITY,
                           objective="judge a script-first benchmark answer")
    judge_model, why = pick_judge(candidate_model, route, preferred_judge)
    if judge_model is None:
        return JudgeScore(verdict="refused" if why.startswith("refused")
                          else "unavailable",
                          refused=why.startswith("refused"), detail=why)

    body = {"prompt": _no_think(build_judge_prompt(case, output_text,
                                                   source_text)),
            "max_new_tokens": JUDGE_MAX_TOKENS, "model_key": judge_model,
            "temperature": 0.0}
    task = getattr(route, "task", None) or "text-generation"
    try:
        result = _run_bounded(lambda: _dispatch(task, body), deadline_s,
                              f"benchmark-judge:{case.case_id}")
    except Exception as exc:  # noqa: BLE001 — a judge fault degrades
        return JudgeScore(judge_model=judge_model,
                          detail=f"{type(exc).__name__}: {exc}"[:300])
    payload = _payload(result)
    if payload.get("ok") is False or payload.get("error"):
        return JudgeScore(judge_model=judge_model,
                          detail=f"judge not-ok: {payload.get('error')}"[:300])

    from hugpy_oracle.evaluation import parse_judge_verdict
    parsed = parse_judge_verdict(_strip_think(str(payload.get("text") or "")))
    if parsed["verdict"] is None and parsed["score"] is None:
        return JudgeScore(judge_model=judge_model, verdict="unscored",
                          detail="the judge reply carried no verdict and no "
                                 "score")
    return JudgeScore(
        judge_model=judge_model, verdict=parsed["verdict"] or "unscored",
        score=float(parsed["score"]) if parsed["score"] is not None else None,
        why=parsed["why"], available=parsed["score"] is not None,
        detail=why)


# ---------------------------------------------------------------------------
# Authoring — the SAME bounded contract k110 uses
# ---------------------------------------------------------------------------


def _k110_author(stage: str, prompt: str, llm: Callable[[str], str],
                 build: Callable[[Mapping[str, Any]], Any]) -> Any:
    """k110's ``screenplay._author`` — ask, validate, ONE repair, then a typed
    gap. Reused rather than re-implemented on purpose: a benchmark that gave
    models three attempts, or zero repairs, would not be measuring the
    pipeline's behaviour, it would be measuring the benchmark's."""
    from hugpy_oracle.screenplay import _author
    return _author(stage, prompt, llm, build)


def author_completion(case: BenchCase, llm: Callable[[str], str],
                      preamble: str = "") -> "Screenplay | AuthoringGap":
    """Track A: complete the supplied material into a validated Screenplay."""
    return _k110_author("screenplay", build_completion_prompt(case, preamble),
                        llm, Screenplay.from_dict)


def author_workflow(case: BenchCase, llm: Callable[[str], str],
                    screenplay: Screenplay = FIXTURE_SCREENPLAY,
                    preamble: str = ""
                    ) -> "dict[str, Any] | AuthoringGap":
    """Track C: one production artifact, shape-validated by
    :func:`workflow_errors` under the same one-repair contract."""
    from hugpy_oracle.screenplay import ScreenplayRefused
    spec = case.spec
    ground = _ground_truth(spec, screenplay)

    def build(obj: Mapping[str, Any]) -> dict[str, Any]:
        problems = workflow_errors(spec, obj, ground)
        if problems:
            raise ScreenplayRefused("; ".join(problems)[:400],
                                    errors=list(problems))
        return dict(obj)

    return _k110_author(spec.operation,
                        build_workflow_prompt(case, screenplay, preamble),
                        llm, build)


def produce(case: BenchCase, llm: Callable[[str], str],
            screenplay: Screenplay = FIXTURE_SCREENPLAY,
            preamble: str = "") -> Any:
    """Run one case's operation with ``llm``. Track B goes through k110's own
    ``author_plot`` — same prompt, same modes, same validator as production.

    ``preamble`` (k109b) is prepended to whichever prompt the track builds. On
    Track B that means it rides in FRONT of k110's own ``build_plot_prompt``
    output rather than replacing any of it, so the question k110 asks is still
    the question that gets asked."""
    if case.track == "A":
        return author_completion(case, llm, preamble)
    if case.track == "C":
        return author_workflow(case, llm, screenplay, preamble)
    if preamble:
        def briefed(prompt: str) -> str:
            return llm(f"{preamble}\n\n{prompt}")
        return author_plot(case.input_text, briefed, mode=case.plot_mode)
    return author_plot(case.input_text, llm, mode=case.plot_mode)


# ---------------------------------------------------------------------------
# One attempt
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything a run is parameterized by, in one recordable object."""
    mode: str = "normalized"
    repeats: int = 1
    tracks: str = "ABC"
    limit_per_track: int | None = None
    deadline_s: float = DEFAULT_ATTEMPT_DEADLINE_S
    judge: bool = True
    judge_deadline_s: float = 90.0
    judge_model: str | None = None
    vram_reserve_gib: float = DEFAULT_VRAM_RESERVE_GIB
    max_models: int | None = None
    label: str = ""
    #: or-k10 — peak-VRAM sampler cadence in ms; 0 disables.
    vram_sample_ms: int = DEFAULT_VRAM_SAMPLE_MS
    #: k109b, additive. Merged OVER whatever ``mode_params`` returned, and
    #: recorded in ``ceiling_source`` so a report never claims a configuration
    #: it did not send. ``None`` leaves every k109 run byte-identical.
    params_override: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"RunConfig.mode {self.mode!r} is not one of "
                             f"{list(MODES)}")
        if self.repeats < 1:
            raise ValueError("RunConfig.repeats must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "repeats": self.repeats,
                "tracks": self.tracks, "limit_per_track": self.limit_per_track,
                "deadline_s": self.deadline_s, "judge": self.judge,
                "judge_deadline_s": self.judge_deadline_s,
                "judge_model": self.judge_model,
                "vram_reserve_gib": self.vram_reserve_gib,
                "vram_sample_ms": self.vram_sample_ms,
                "max_models": self.max_models, "label": self.label,
                "params_override": dict(self.params_override)
                if self.params_override else None}


def mode_params(mode: str, model_id: str | None = None,
                view: Any = None) -> tuple[dict[str, Any], str]:
    """The generation configuration for one mode, and where it came from.

    NORMALIZED is a constant: identical for every candidate, which is the only
    thing that makes two models' rows comparable. CEILING asks the CATALOG what
    this model can actually take (``limits.context_tokens``) and falls back to
    a documented default, recording which happened — a ceiling number whose
    provenance is unknown is not a ceiling, it is a guess."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {list(MODES)}")
    if mode == "normalized":
        return dict(NORMALIZED_PARAMS), "normalized:constant"
    params = dict(CEILING_PARAMS)
    limits = getattr(view, "limits", None) or {}
    context = None
    try:
        context = limits.get("context_tokens")
    except Exception:  # noqa: BLE001
        context = None
    if context:
        params["context_tokens"] = int(context)
        return params, "ceiling:catalog.limits.context_tokens"
    return params, "ceiling:default (catalog states no context limit)"


def _body_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """The subset the dispatch path actually accepts. ``context_tokens`` is
    RECORDED but not sent: the synchronous front door does not take it, and
    sending a key the runner ignores would let a report claim a context that
    was never configured."""
    return {k: params[k] for k in ("max_new_tokens", "temperature", "top_p")
            if k in params}


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    try:
        from hugpy_oracle.runtime import DispatchTimeout
        if isinstance(exc, DispatchTimeout):
            return True
    except Exception:  # noqa: BLE001
        pass
    return "timeout" in f"{type(exc).__name__}: {exc}".lower()


def _vram_for(snapshot: Mapping[str, Any] | None, worker: str | None
              ) -> dict[str, Any] | None:
    """The VRAM row for the worker that will serve this attempt, or the whole
    fleet total when the worker is unknown."""
    if not isinstance(snapshot, Mapping):
        return None
    rows = snapshot.get("workers")
    if isinstance(rows, list) and worker:
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if worker in (row.get("id"), row.get("name")):
                return {k: row.get(k) for k in
                        ("id", "name", "status", "vram_total", "vram_used",
                         "vram_free", "gpu_util_pct")}
    if isinstance(rows, list) and rows:
        totals = {"vram_total": 0, "vram_used": 0, "vram_free": 0}
        counted = 0
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            counted += 1
            for key in totals:
                value = row.get(key)
                totals[key] += int(value) if isinstance(value, (int, float)) else 0
        if counted:
            return {"scope": f"fleet total over {counted} worker(s)", **totals}
    return None


def run_case(case: BenchCase, model: str, *,
             config: RunConfig | None = None,
             screenplay: Screenplay = FIXTURE_SCREENPLAY,
             registry_version: str | None = None,
             preamble: str = "") -> tuple[Attempt, str]:
    """One (case, model) attempt. Returns ``(attempt, raw_reply)``.

    NEVER raises for a model or fleet problem: a timeout, a dead worker or a
    model that answers with an apology all become an ``Attempt`` with a
    ``failure`` string and honest zeros — which is the only way an aborted
    sweep still produces a usable report."""
    config = config or RunConfig()
    started = _utc_now()
    route = _resolve_route(BENCH_CAPABILITY, model)
    perf = PerfRecord(model=model, mode=config.mode,
                      vram_reserve_gib=config.vram_reserve_gib)
    if route is None or getattr(route, "execution", None) != "execute":
        reasons = "; ".join(getattr(route, "reasons", ()) or ()) or \
            "route resolution failed"
        return Attempt(
            case_id=case.case_id, track=case.track, operation=case.operation,
            model=model, mode=config.mode, failure=f"no-route: {reasons}"[:300],
            gap_code="CAPABILITY_GAP", perf=perf,
            registry_version=registry_version, started_at=started,
            ended_at=_utc_now()), ""

    task = getattr(route, "task", None) or "text-generation"
    params, ceiling_source = mode_params(config.mode, model, _capability_view())
    if config.params_override:
        params = dict(params, **dict(config.params_override))
        ceiling_source = (f"{ceiling_source} + override"
                          f"({sorted(config.params_override)})")
    worker = _selected_worker(model, task)
    load_state = _load_state(model, worker)
    vram_before = _vram_for(_vram_snapshot(), worker)
    sampler = _VramSampler(worker, sample_ms=config.vram_sample_ms).start()

    state: dict[str, Any] = {"timeout": False, "payloads": [], "raws": []}

    def llm(prompt: str) -> str:
        body = dict(_body_params(params))
        body["prompt"] = _no_think(prompt)
        body["model_key"] = model
        try:
            result = _run_bounded(lambda: _dispatch(task, body),
                                  config.deadline_s,
                                  f"benchmark:{case.case_id}:{model}")
        except Exception as exc:  # noqa: BLE001 — recorded, then re-raised
            if _is_timeout(exc):
                state["timeout"] = True
            raise
        payload = _payload(result)
        state["payloads"].append(payload)
        if payload.get("ok") is False or payload.get("error"):
            raise RuntimeError(f"dispatch not-ok: {payload.get('error')}")
        text = str(payload.get("text") or "")
        state["raws"].append(text)
        if not text.strip():
            raise RuntimeError("the model returned no text")
        return text

    t0 = time.monotonic()
    result = produce(case, llm, screenplay, preamble)
    latency = round(time.monotonic() - t0, 4)

    raws: list[str] = state["raws"]
    raw = raws[-1] if raws else (result.raw if isinstance(result, AuthoringGap)
                                 else "")
    failure: str | None = None
    gap_code: str | None = None
    if isinstance(result, AuthoringGap):
        gap_code = result.code
        if result.code == "LLM_ERROR":
            failure = "timeout" if state["timeout"] else "dispatch_error"

    deterministic = score_case(case, result, raw, screenplay)
    judge = JudgeScore(detail="judging disabled for this run")
    if config.judge and failure is None:
        judge = judge_attempt(case, raw, model,
                              deadline_s=config.judge_deadline_s,
                              source_text=case.input_text,
                              preferred_judge=config.judge_model)

    last = state["payloads"][-1] if state["payloads"] else {}
    usage = last.get("usage") if isinstance(last.get("usage"), Mapping) else {}
    sampler.stop()
    vram_after = _vram_for(_vram_snapshot(), worker)
    delta = None
    if vram_before and vram_after and \
            isinstance(vram_before.get("vram_used"), (int, float)) and \
            isinstance(vram_after.get("vram_used"), (int, float)):
        delta = int(vram_after["vram_used"] - vram_before["vram_used"])

    perf = PerfRecord(
        model=model, worker=worker, load_state=load_state, mode=config.mode,
        params=dict(params, context_tokens_enforced=False),
        ceiling_source=ceiling_source, latency_s=latency,
        dispatch_calls=len(state["payloads"]),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        tokens_per_s=_tokens_per_second(last),
        finish_reason=last.get("finish_reason"),
        output_chars=len(raw), vram_before=vram_before, vram_after=vram_after,
        vram_used_delta_bytes=delta, **sampler.perf_fields(),
        gpu_total_bytes=(vram_before or {}).get("vram_total"),
        vram_reserve_gib=config.vram_reserve_gib)

    return Attempt(
        case_id=case.case_id, track=case.track, operation=case.operation,
        model=model, mode=config.mode, deterministic=deterministic,
        judge=judge, perf=perf, failure=failure, gap_code=gap_code,
        registry_version=registry_version, started_at=started,
        ended_at=_utc_now()), raw


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------


def discover_models(capability: str = BENCH_CAPABILITY,
                    limit: int | None = None
                    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The models the CATALOG says can serve ``capability`` here, and the
    catalog's own reasons.

    Discovery is the catalog's job, not this module's: a benchmark that kept
    its own model list would happily benchmark a model the fleet has blocked,
    or miss one that was added this morning. An empty tuple with reasons is a
    complete, honest answer — "no text model is eligible on this fleet" is a
    finding, not an error."""
    view = _capability_view(capability)
    if view is None:
        return (), (f"capability {capability!r} is not in the unified catalog",)
    reasons = tuple(getattr(view.eligibility, "reasons", ()) or ())
    if not getattr(view.eligibility, "eligible", False):
        return (), reasons or (f"{capability} is not eligible on this fleet",)
    models = tuple(view.model_ids)
    if limit is not None:
        models = models[:max(0, int(limit))]
    return models, reasons


# ---------------------------------------------------------------------------
# The run directory
# ---------------------------------------------------------------------------


def default_run_root() -> str:
    """``ORACLE_BENCHMARK_ROOT`` else the oracle's own battery directory
    (``hugpy_oracle.config.benchmark_root``)."""
    from hugpy_oracle.config import benchmark_root
    return benchmark_root()


def new_run_dir(root: str | None = None, label: str = "") -> str:
    """``<root>/oracle-<YYYYmmdd-HHMM>[-N]``, claimed by creating it.

    The battery's timestamp format and its ``makedirs(exist_ok=False)`` claim,
    so two runs started in the same minute cannot write into each other and
    the dirs sort chronologically next to the image battery's."""
    base = root or default_run_root()
    stamp = time.strftime("%Y%m%d-%H%M")
    suffix = f"-{re.sub(r'[^A-Za-z0-9_.-]+', '-', label)}" if label else ""
    for attempt in range(1, 60):
        name = f"oracle-{stamp}{suffix}" + ("" if attempt == 1 else f"-{attempt}")
        path = os.path.join(base, name)
        try:
            os.makedirs(path, exist_ok=False)
        except FileExistsError:
            continue
        os.makedirs(os.path.join(path, "raw"), exist_ok=True)
        return path
    raise RuntimeError(f"could not claim a run dir under {base!r}")


def _atomic_write(path: str, text: str) -> bool:
    """Write via a sibling temp file + os.replace. Returns False instead of
    raising: a report that cannot be written must not destroy the run that
    produced it."""
    import tempfile
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        handle, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(tmp, path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("benchmark: could not write %s (%s: %s)", path,
                       type(exc).__name__, exc)
        return False


def _append_line(path: str, text: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(text + "\n")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("benchmark: could not append to %s (%s: %s)", path,
                       type(exc).__name__, exc)
        return False


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """One sweep: what ran, on what, and where the evidence is."""
    run_id: str
    run_dir: str
    config: RunConfig
    models: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    registry_version: str | None = None
    attempts: tuple[Attempt, ...] = ()
    aborted: Mapping[str, str] = field(default_factory=dict)
    discovery_reasons: tuple[str, ...] = ()
    started_at: str = ""
    ended_at: str = ""

    @property
    def rows(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self.attempts]

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "run_dir": self.run_dir,
                "config": self.config.to_dict(), "models": list(self.models),
                "case_ids": list(self.case_ids),
                "registry_version": self.registry_version,
                "aborted": dict(self.aborted),
                "discovery_reasons": list(self.discovery_reasons),
                "started_at": self.started_at, "ended_at": self.ended_at,
                "attempt_count": len(self.attempts),
                "ok_count": sum(1 for a in self.attempts if a.ok)}


def run_sweep(models: Sequence[str] | None = None, *,
              config: RunConfig | None = None,
              run_dir: str | None = None,
              screenplay: Screenplay = FIXTURE_SCREENPLAY,
              note: Callable[[str], None] | None = None) -> BenchmarkRun:
    """The sweep: every selected case against every candidate, SEQUENTIALLY.

    Sequential is a politeness decision, not a simplification — the fleet's
    admission control is shared with live work, and a benchmark that fans out
    is a benchmark that evicts somebody else's model mid-render.

    A model is DROPPED after :data:`TIMEOUT_ABORT_STREAK` consecutive dispatch
    timeouts and the drop is recorded in ``aborted``: two timeouts in a row is
    the fleet telling us this candidate cannot serve this workload right now,
    and continuing would spend an hour proving it thirty more times.

    Everything is written to the run dir AS IT HAPPENS (the battery's rule:
    persist incrementally so a crash loses nothing)."""
    config = config or RunConfig()
    directory = run_dir or new_run_dir(label=config.label)
    started = _utc_now()
    run_id = os.path.basename(directory)
    registry_version = _registry_version()
    log_path = os.path.join(directory, "run.log")

    def say(message: str) -> None:
        _append_line(log_path, f"{_utc_now()} {message}")
        if note is not None:
            note(message)

    discovery_reasons: tuple[str, ...] = ()
    if models is None:
        models, discovery_reasons = discover_models(limit=config.max_models)
    elif config.max_models is not None:
        models = tuple(models)[:config.max_models]
    models = tuple(models)

    suite = cases_for(config.tracks, config.limit_per_track)
    say(f"run {run_id}: mode={config.mode} repeats={config.repeats} "
        f"tracks={config.tracks} models={list(models)} "
        f"cases={[c.case_id for c in suite]} registry_version={registry_version}")

    _atomic_write(os.path.join(directory, "cases.json"),
                  json.dumps({"cases": [c.to_dict() for c in suite],
                              "operations": {k: v.to_dict()
                                             for k, v in OPERATIONS.items()},
                              "fixture_screenplay":
                                  screenplay.to_dict(),
                              "fixture_digest": screenplay.digest},
                             indent=1, sort_keys=True))
    _atomic_write(os.path.join(directory, "environment.json"),
                  json.dumps({
                      "run_id": run_id, "started_at": started,
                      "config": config.to_dict(), "models": list(models),
                      "discovery_reasons": list(discovery_reasons),
                      "registry_version": registry_version,
                      "capability": BENCH_CAPABILITY,
                      "deterministic_weights": _WEIGHTS,
                      "normalized_params": NORMALIZED_PARAMS,
                      "ceiling_params": CEILING_PARAMS,
                      "vram_reserve_gib": config.vram_reserve_gib,
                      "vram_snapshot_at_start": _vram_snapshot(),
                      "host": os.uname().nodename,
                  }, indent=1, sort_keys=True, default=str))

    attempts: list[Attempt] = []
    aborted: dict[str, str] = {}
    attempts_path = os.path.join(directory, "attempts.jsonl")

    for model in models:
        streak = 0
        for case in suite:
            if model in aborted:
                break
            for repeat in range(config.repeats):
                attempt, raw = run_case(case, model, config=config,
                                        screenplay=screenplay,
                                        registry_version=registry_version)
                attempt = replace(attempt, repeat=repeat)
                raw_name = f"{case.case_id}__{re.sub(r'[^A-Za-z0-9_.-]+', '-', model)}__r{repeat}.txt"
                if raw:
                    _atomic_write(os.path.join(directory, "raw", raw_name),
                                  raw[:_MAX_RAW_CHARS])
                    attempt = replace(attempt, raw_ref=os.path.join("raw",
                                                                    raw_name))
                attempts.append(attempt)
                _append_line(attempts_path,
                             json.dumps(attempt.to_dict(), sort_keys=True,
                                        default=str))
                say(f"{model} · {case.case_id} r{repeat}: "
                    f"{'ok' if attempt.ok else 'FAIL'} "
                    f"det={attempt.deterministic.score} "
                    f"judge={attempt.judge.score} "
                    f"{attempt.perf.latency_s}s "
                    f"{attempt.failure or ''}".rstrip())

                if attempt.failure == "timeout":
                    streak += 1
                    if streak >= TIMEOUT_ABORT_STREAK:
                        aborted[model] = (
                            f"dropped after {streak} consecutive dispatch "
                            f"timeouts (last: {case.case_id})")
                        say(f"{model}: {aborted[model]}")
                        break
                else:
                    streak = 0
            if model in aborted:
                break

    run = BenchmarkRun(
        run_id=run_id, run_dir=directory, config=config, models=models,
        case_ids=tuple(c.case_id for c in suite),
        registry_version=registry_version, attempts=tuple(attempts),
        aborted=aborted, discovery_reasons=discovery_reasons,
        started_at=started, ended_at=_utc_now())
    write_reports(run)
    say(f"run {run_id} finished: {len(attempts)} attempt(s), "
        f"{sum(1 for a in attempts if a.ok)} ok")
    return run


def vram_peak_per_model(rows: Sequence[Mapping[str, Any]]
                        ) -> dict[str, dict[str, Any]]:
    """or-k10 — per model: the highest ``vram_peak_bytes`` any attempt saw,
    the sampler that saw it and the samples behind it. Models whose attempts
    never had a sampler answer are listed with ``vram_peak_bytes`` None, so a
    missing number is visibly missing rather than silently absent."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = str(row.get("model") or "")
        if not model:
            continue
        perf = row.get("perf") if isinstance(row.get("perf"), Mapping) else {}
        peak = perf.get("vram_peak_bytes")
        count = perf.get("vram_sample_count") or 0
        entry = out.setdefault(model, {"vram_peak_bytes": None,
                                       "vram_sampler": None,
                                       "vram_sample_count": 0,
                                       "attempts": 0})
        entry["attempts"] += 1
        entry["vram_sample_count"] += int(count) if isinstance(
            count, (int, float)) else 0
        if isinstance(peak, (int, float)) and (
                entry["vram_peak_bytes"] is None
                or int(peak) > entry["vram_peak_bytes"]):
            entry["vram_peak_bytes"] = int(peak)
            entry["vram_sampler"] = perf.get("vram_sampler")
    return out


def write_reports(run: BenchmarkRun) -> dict[str, str]:
    """Serialize the run: summary + routing matrix + leaderboard. Returns the
    paths actually written (a path missing from the mapping failed to write and
    said so in the log — the run itself is already durable in
    ``attempts.jsonl``)."""
    from hugpy_oracle import routing_matrix as rm

    written: dict[str, str] = {}
    rows = run.rows
    summary = {
        "run": run.to_dict(),
        "formula": rm.FORMULA_NOTE,
        "deterministic_weights": _WEIGHTS,
        "per_model_operation": rm.summarize(rows),
        "vram_peak_per_model": vram_peak_per_model(rows),
    }
    path = os.path.join(run.run_dir, "scores.json")
    if _atomic_write(path, json.dumps(summary, indent=1, sort_keys=True,
                                      default=str)):
        written["scores"] = path

    matrix = rm.derive_matrix(rows, registry_version=run.registry_version,
                              run_id=run.run_id, run_dir=run.run_dir,
                              mode=run.config.mode)
    path = os.path.join(run.run_dir, "routing_matrix.json")
    if _atomic_write(path, json.dumps(matrix.to_dict(), indent=1,
                                      sort_keys=True, default=str)):
        written["matrix"] = path
    path = os.path.join(run.run_dir, "leaderboard.md")
    if _atomic_write(path, rm.render_leaderboard(matrix, rows)):
        written["leaderboard"] = path
    return written



# ===========================================================================
# k109b — THE STATIONARY-PROMPT FULL-FLEET SWEEP
# ===========================================================================
#
# One brief (``stationary_scenario``), every model the fleet can serve, every
# point of the unified lifecycle, one verdict per (model, point) cell.
#
# WHY THIS IS NOT JUST ``run_sweep`` WITH MORE CASES. k109's sweep answers
# "which model is best at operation X" and its unit of evidence is an LLM
# attempt. This one answers "what is this fleet capable of at each point of the
# lifecycle", and four of the points are not LLM points at all: a VLM judging
# frames, an image model rendering a keyframe, a clip model rendering video, a
# TTS model speaking a line. Each needs its own dispatch shape, its own
# technical guard and its own honest failure mode, and forcing them through
# ``Attempt`` would have meant pretending a wav is a screenplay.
#
# So: a :class:`Cell` is the unit here, and its ``to_dict`` is a SUPERSET of
# k109's attempt-row shape — ``operation``/``model``/``ok``/``deterministic``/
# ``judge``/``perf``/``failure`` all mean exactly what they mean in
# ``attempts.jsonl``. ``routing_matrix.summarize`` and ``derive_matrix``
# therefore consume these rows with no change at all, and ``best_route`` /
# ``load_latest_matrix`` (k114b) read the resulting matrix unchanged. The extra
# keys (``point_id``, ``step``, ``verdict``, ``scenario_version``, ``evidence``)
# are additive and ignored by every k109 consumer.
#
# FIVE VERDICTS, and the line between them:
#
#   capable       the artifact validated. The model can do this job today.
#   partial       structured output came back and the validator refused it.
#                 This is the interesting verdict: the model understood the
#                 shape and got the content wrong, which is a prompt or a
#                 config problem far more often than a model problem.
#   incapable     no usable output: a timeout, a dead route, prose where JSON
#                 was asked for, an empty reply, a blank frame, a silent wav.
#   refused       the model answered and declined. Never folded into
#                 ``incapable``: a refusal is a policy fact about the model,
#                 not a capability fact about the fleet.
#   NO_CANDIDATES no model on this fleet is eligible for the point at all.
#                 Emitted per POINT, never per model, and it names the missing
#                 capability. The gap IS the data.
#
# POLITENESS, RESTATED AS CODE. Sequential, always. The normal route +
# admission path, always. Two consecutive dispatch timeouts drop a model. ONE
# retry on a transient dispatch fault (the kind another agent restarting a
# worker produces) and then the failure is recorded as that model's result.
# Three unreachable models in a row pauses the whole sweep, writes state, and
# says so — because at that point the sweep is measuring an outage.

#: The stages, in the order they run. Each is independently resumable.
STATIONARY_STAGES: tuple[str, ...] = ("llm", "vlm", "image", "video", "tts")

#: The five verdicts. Ordered best-to-worst for the grid's sort.
VERDICTS: tuple[str, ...] = ("capable", "partial", "refused", "incapable",
                             "NO_CANDIDATES")

#: The generation configuration every text candidate gets. Deliberately NOT
#: ``NORMALIZED_PARAMS``: k109's pilot proved 2048 output tokens truncates a
#: full screenplay JSON, and a sweep whose headline finding is "the budget was
#: too small" measured the budget. Everything else is pinned identically for
#: every candidate, which is the only thing that makes two rows comparable.
STATIONARY_PARAMS: dict[str, Any] = {
    "max_new_tokens": 3072,
    "temperature": 0.4,
    "top_p": 0.9,
    "context_tokens": 8192,
}

#: The admission probe's budget. One tiny prompt, asked before a model is put
#: through eight lifecycle points, so a model that cannot serve text at all
#: costs the sweep ONE deadline instead of eight. Cold loads are allowed
#: through it — that is the point of a generous number.
DEFAULT_PROBE_DEADLINE_S: float = 120.0

#: Seconds to wait after a dispatch timeout before the next one. ``run_bounded``
#: cannot kill the orphaned thread (it says so), so a timed-out attempt is
#: still occupying the fleet when we return. Marching straight into the next
#: model would stack loads on a worker that is already busy with ours.
TIMEOUT_COOLDOWN_S: float = 15.0

#: Consecutive models that fail to answer at all before the sweep PAUSES.
#: Three, because two can be two bad rows and three in a row is an outage — and
#: continuing to grade an outage produces a matrix that says a healthy fleet
#: has no capabilities.
FLEET_DEGRADED_STREAK: int = 3

#: The probe prompt. Short, unambiguous, and answerable by a 0.6B model.
PROBE_PROMPT: str = ("Reply with exactly the single word READY and nothing "
                     "else.")

#: Markers of a model DECLINING rather than failing. Deliberately narrow: a
#: model that says "I cannot determine the time of day" inside a valid JSON
#: answer has not refused, so these are only consulted when NO artifact was
#: produced, and only against the first 600 characters of the reply.
REFUSAL_MARKERS: tuple[str, ...] = (
    "i cannot assist", "i can't assist", "i cannot help with",
    "i can't help with", "i cannot comply", "i won't be able to",
    "i will not provide", "i cannot provide", "i can't provide",
    "i'm not able to provide", "i am not able to provide",
    "i must decline", "i cannot create", "i can't create",
    "as an ai language model, i cannot", "i'm sorry, but i cannot",
    "i'm sorry, but i can't", "against my guidelines",
)

#: How many characters of a raw reply the refusal test reads.
_REFUSAL_WINDOW: int = 600

#: Minimum pixel standard deviation before a rendered frame counts as an image
#: rather than a flat fill. A solid-colour 512x512 has stdev 0.0; a real render
#: of this scenario's grey harbour is 30-60. 2.0 is far from both, which is the
#: same discipline as ``scorecard.SILENT_AUDIO_PEAK_FLOOR`` and is deliberately
#: not tuned to any observed render.
BLANK_IMAGE_STDEV_FLOOR: float = 2.0


# ---------------------------------------------------------------------------
# Content guards — "did anything actually come out" per medium
# ---------------------------------------------------------------------------


def wav_levels(path: str) -> tuple[int, float] | None:
    """(peak, rms) in int16 units for a 16-bit PCM wav, or None when the file
    is not one this can measure.

    Stdlib only (wave/array/math), the same call ``oracle.scorecard`` makes for
    the same reason: deciding "did anything come out of the speaker" must not
    acquire numpy. Implemented here rather than imported because the scorecard's
    version is private to that module and this wave does not own that file; the
    PUBLIC constant it exports — the silence floor — IS reused, so the two
    cannot drift on the number that matters."""
    import array
    import math
    import sys as _sys
    import wave
    try:
        with wave.open(path, "rb") as handle:
            if handle.getsampwidth() != 2:
                return None
            frames = handle.readframes(handle.getnframes())
    except Exception:  # noqa: BLE001 — unmeasurable is not the same as silent
        return None
    data = array.array("h")
    data.frombytes(frames[:len(frames) - (len(frames) % 2)])
    if _sys.byteorder == "big":
        data.byteswap()
    if not data:
        return 0, 0.0
    peak = max(max(data), -min(data))
    rms = math.sqrt(sum(float(v) * v for v in data) / len(data))
    return peak, rms


def audio_carries_sound(path: str) -> tuple[bool, str]:
    """(carries sound, detail) for a produced wav — the CONTENT guard.

    A wav can be a valid, non-zero-byte, exactly-right-duration file and hold
    nothing: the 2026-08-21 tts-silence fault wrote 2.32s of PCM whose peak
    amplitude was 1. Existence is not substance for audio. The measured level
    is named on BOTH branches so an operator reads the number, not a verdict."""
    import math
    try:
        from hugpy_oracle.scorecard import SILENT_AUDIO_PEAK_FLOOR as floor
    except Exception:  # noqa: BLE001 — the guard must survive a shared-file edit
        floor = 500
    levels = wav_levels(path)
    if levels is None:
        return True, f"audio level unmeasurable (not 16-bit PCM wav): {path}"
    peak, rms = levels
    dbfs = 20 * math.log10(rms / 32768.0) if rms > 0 else float("-inf")
    return peak >= floor, (f"audio peak {peak}/32767, RMS {dbfs:.1f} dBFS "
                           f"(silence floor: peak {floor})")


def image_levels(path: str) -> dict[str, Any] | None:
    """Width, height, mean and standard deviation of a rendered frame, or None
    when it cannot be measured.

    Pillow, imported lazily and inside the try: this module must stay importable
    on a box with no imaging stack, and a missing Pillow degrades the check to
    "unmeasurable" rather than failing every image model in the sweep."""
    try:
        from PIL import Image, ImageStat
        with Image.open(path) as image:
            image.load()
            grey = image.convert("L")
            stat = ImageStat.Stat(grey)
            return {"width": image.width, "height": image.height,
                    "mode": image.mode,
                    "mean": round(float(stat.mean[0]), 3),
                    "stdev": round(float(stat.stddev[0]), 3)}
    except Exception as exc:  # noqa: BLE001
        logger.debug("k109b: image levels unavailable for %s (%s: %s)", path,
                     type(exc).__name__, exc)
        return None


def image_carries_content(path: str) -> tuple[bool, str]:
    """(carries content, detail) for a rendered frame — the image twin of the
    silent-wav guard. A uniform fill is the image failure mode that passes
    every existence check, so the variance is measured and NAMED."""
    levels = image_levels(path)
    if levels is None:
        return True, (f"image level unmeasurable (Pillow unavailable or the "
                      f"file is not a readable image): {path}")
    ok = float(levels["stdev"]) >= BLANK_IMAGE_STDEV_FLOOR
    return ok, (f"{levels['width']}x{levels['height']} {levels['mode']}, "
                f"pixel stdev {levels['stdev']}, mean {levels['mean']} "
                f"(blank floor: stdev {BLANK_IMAGE_STDEV_FLOOR})")


def probe_media(path: str) -> dict[str, Any] | None:
    """ffprobe's read of a produced clip: streams, geometry, duration, frames.

    Returns None when ffprobe is absent or refuses the file — which is itself
    the finding for a "clip" that is not decodable."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout or "{}")
    except Exception as exc:  # noqa: BLE001
        logger.debug("k109b: ffprobe unavailable for %s (%s: %s)", path,
                     type(exc).__name__, exc)
        return None
    video = next((s for s in data.get("streams") or ()
                  if s.get("codec_type") == "video"), None)
    if video is None:
        return {"streams": len(data.get("streams") or ()), "video": False}
    frames = video.get("nb_frames")
    try:
        frames = int(frames) if frames not in (None, "N/A") else None
    except (TypeError, ValueError):
        frames = None
    try:
        duration = float((data.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    return {"video": True, "codec": video.get("codec_name"),
            "width": video.get("width"), "height": video.get("height"),
            "frames": frames, "duration_s": duration,
            "avg_frame_rate": video.get("avg_frame_rate"),
            "bytes": _file_size(path)}


def extract_middle_frame(video_path: str, out_path: str) -> str | None:
    """One PNG from the middle of a clip, for the VLM judge to look at.

    The MIDDLE, not the first frame: a t2v model's frame 0 is frequently the
    conditioning image or a grey plate, and judging that would grade the plate.
    Returns the path written, or None."""
    import subprocess
    probe = probe_media(video_path) or {}
    duration = probe.get("duration_s")
    seek = max(0.0, float(duration) / 2.0) if duration else 0.5
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        result = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{seek:.3f}",
             "-i", video_path, "-frames:v", "1", out_path],
            capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and os.path.isfile(out_path):
            return out_path
    except Exception as exc:  # noqa: BLE001
        logger.debug("k109b: frame extraction failed for %s (%s: %s)",
                     video_path, type(exc).__name__, exc)
    return None


def _file_size(path: str) -> int | None:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def looks_like_refusal(text: str) -> bool:
    """Did the model DECLINE, as opposed to fail?

    Read only over the first :data:`_REFUSAL_WINDOW` characters and only ever
    consulted when no artifact was produced, so a valid screenplay containing
    the words "I cannot" in dialogue is never misread as a refusal."""
    low = str(text or "")[:_REFUSAL_WINDOW].lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


# ---------------------------------------------------------------------------
# The cell — one (point, operation, model) verdict
# ---------------------------------------------------------------------------


def classify_verdict(*, produced: bool, validated: bool, structured: bool,
                     refused: bool, no_candidates: bool = False) -> str:
    """The five-verdict rule, in one place so the grid and the matrix agree.

    Order matters and is deliberate: NO_CANDIDATES is a property of the POINT
    and outranks everything; a refusal outranks a failure because a model that
    declined is not a model that broke; validation outranks structure because
    the whole argument of this pipeline is that a well-shaped wrong answer is
    still a wrong answer."""
    if no_candidates:
        return "NO_CANDIDATES"
    if refused:
        return "refused"
    if validated:
        return "capable"
    if structured and produced:
        return "partial"
    return "incapable"


@dataclass(frozen=True, slots=True)
class Cell:
    """One (lifecycle point, operation, model) result.

    ``to_dict`` is a SUPERSET of k109's attempt row, so
    ``routing_matrix.summarize``/``derive_matrix`` read these unchanged."""
    point_id: str
    step: int
    operation: str
    model: str
    capability: str
    verdict: str
    deterministic: DeterministicScore = field(default_factory=DeterministicScore)
    judge: JudgeScore = field(default_factory=JudgeScore)
    perf: PerfRecord = field(default_factory=PerfRecord)
    failure: str | None = None
    gap_code: str | None = None
    note: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    raw_ref: str = ""
    artifact_ref: str = ""
    registry_version: str | None = None
    scenario_version: str = ""
    scenario_digest: str = ""
    stage: str = ""
    started_at: str = ""
    ended_at: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"Cell.verdict {self.verdict!r} is not one of "
                             f"{list(VERDICTS)}")

    @property
    def key(self) -> str:
        """The resume key. A (point, operation, model) triple, because one
        model can appear at two points and one point can measure two
        operations."""
        return f"{self.point_id}|{self.operation}|{self.model}"

    @property
    def ok(self) -> bool:
        """k109's ``ok`` semantics, preserved exactly: a row counts as a
        success only when the artifact validated."""
        return self.verdict == "capable"

    def to_dict(self) -> dict[str, Any]:
        return {
            # --- the k109 attempt-row shape, unchanged ---
            "case_id": self.point_id, "track": self.stage,
            "operation": self.operation, "model": self.model,
            "mode": "stationary", "repeat": 0, "ok": self.ok,
            "deterministic": self.deterministic.to_dict(),
            "judge": self.judge.to_dict(), "perf": self.perf.to_dict(),
            "failure": self.failure, "gap_code": self.gap_code,
            "raw_ref": self.raw_ref, "registry_version": self.registry_version,
            "started_at": self.started_at, "ended_at": self.ended_at,
            # --- k109b additions (ignored by every k109 consumer) ---
            "point_id": self.point_id, "step": self.step,
            "capability": self.capability, "verdict": self.verdict,
            "note": self.note, "evidence": dict(self.evidence),
            "artifact_ref": self.artifact_ref, "stage": self.stage,
            "scenario_version": self.scenario_version,
            "scenario_digest": self.scenario_digest,
        }


def no_candidates_cell(point: Any, reason: str, *,
                       registry_version: str | None = None,
                       scenario_version: str = "",
                       scenario_digest: str = "") -> Cell:
    """The honest row for a point this fleet cannot serve.

    ``model`` is the literal string ``"(none)"`` rather than an empty one, so
    the row survives a ``groupby`` and shows up in the grid as an explicit
    absence instead of vanishing into a missing key."""
    return Cell(
        point_id=point.point_id, step=point.step,
        operation=point.point_id, model="(none)",
        capability=point.capability or "|".join(point.missing_capability),
        verdict="NO_CANDIDATES", stage=point.kind,
        failure="no eligible model", gap_code="CAPABILITY_GAP",
        note=reason,
        evidence={"missing_capability": list(point.missing_capability),
                  "point_note": point.note},
        registry_version=registry_version,
        scenario_version=scenario_version, scenario_digest=scenario_digest,
        started_at=_utc_now(), ended_at=_utc_now())


# ---------------------------------------------------------------------------
# Sweep configuration + run record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StationaryConfig:
    """Everything the stationary sweep is parameterized by."""
    stages: tuple[str, ...] = STATIONARY_STAGES
    deadline_s: float = DEFAULT_ATTEMPT_DEADLINE_S
    probe_deadline_s: float = DEFAULT_PROBE_DEADLINE_S
    judge: bool = True
    judge_deadline_s: float = 90.0
    judge_model: str | None = None
    vlm_judge_model: str | None = None
    vlm_deadline_s: float = 180.0
    image_deadline_s: float = 300.0
    video_deadline_s: float = 1800.0
    tts_deadline_s: float = 300.0
    asr_deadline_s: float = 300.0
    max_models: int | None = None
    models: tuple[str, ...] = ()
    reference_model: str = "sd-turbo"
    vram_reserve_gib: float = DEFAULT_VRAM_RESERVE_GIB
    budget_s: float | None = None
    #: or-k10 — peak-VRAM sampler cadence in ms; 0 disables.
    vram_sample_ms: int = DEFAULT_VRAM_SAMPLE_MS
    label: str = "stationary"

    def __post_init__(self) -> None:
        unknown = [s for s in self.stages if s not in STATIONARY_STAGES]
        if unknown:
            raise ValueError(f"unknown stage(s) {unknown}; known: "
                             f"{list(STATIONARY_STAGES)}")
        if not self.stages:
            raise ValueError("StationaryConfig.stages must name at least one "
                             "stage — a sweep of nothing is not a sweep")

    def to_dict(self) -> dict[str, Any]:
        return {"stages": list(self.stages), "deadline_s": self.deadline_s,
                "probe_deadline_s": self.probe_deadline_s,
                "judge": self.judge, "judge_deadline_s": self.judge_deadline_s,
                "judge_model": self.judge_model,
                "vlm_judge_model": self.vlm_judge_model,
                "vlm_deadline_s": self.vlm_deadline_s,
                "image_deadline_s": self.image_deadline_s,
                "video_deadline_s": self.video_deadline_s,
                "tts_deadline_s": self.tts_deadline_s,
                "asr_deadline_s": self.asr_deadline_s,
                "max_models": self.max_models, "models": list(self.models),
                "reference_model": self.reference_model,
                "vram_reserve_gib": self.vram_reserve_gib,
                "vram_sample_ms": self.vram_sample_ms,
                "budget_s": self.budget_s, "label": self.label}


@dataclass(frozen=True, slots=True)
class StationarySweep:
    """One stationary run: what ran, what it found, and where the evidence is."""
    run_id: str
    run_dir: str
    config: StationaryConfig
    cells: tuple[Cell, ...] = ()
    registry_version: str | None = None
    scenario_version: str = ""
    scenario_digest: str = ""
    rosters: Mapping[str, Sequence[str]] = field(default_factory=dict)
    aborted: Mapping[str, str] = field(default_factory=dict)
    paused: str = ""
    resumed_from: int = 0
    started_at: str = ""
    ended_at: str = ""

    @property
    def rows(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.cells]

    @property
    def elapsed_note(self) -> str:
        if not (self.started_at and self.ended_at):
            return ""
        try:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.ended_at)
        except ValueError:
            return ""
        seconds = (end - start).total_seconds()
        return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"

    def to_dict(self) -> dict[str, Any]:
        tally: dict[str, int] = {v: 0 for v in VERDICTS}
        for cell in self.cells:
            tally[cell.verdict] = tally.get(cell.verdict, 0) + 1
        return {"run_id": self.run_id, "run_dir": self.run_dir,
                "config": self.config.to_dict(),
                "registry_version": self.registry_version,
                "scenario_version": self.scenario_version,
                "scenario_digest": self.scenario_digest,
                "rosters": {k: list(v) for k, v in self.rosters.items()},
                "aborted": dict(self.aborted), "paused": self.paused,
                "resumed_from": self.resumed_from,
                "started_at": self.started_at, "ended_at": self.ended_at,
                "elapsed": self.elapsed_note,
                "cell_count": len(self.cells), "verdicts": tally}


# ---------------------------------------------------------------------------
# Roster discovery — reusing what the fleet already proved
# ---------------------------------------------------------------------------


def _studio_models(category: str) -> tuple[str, ...]:
    """``video_intel.studio.tester.enumerate_models`` — the fleet's own proven
    all-servable-models enumeration, reused rather than reinvented.

    That module already resolved the thing that makes this hard: image-type
    categories enumerate from the MAIN registry's ``text-to-image`` task while
    video-type categories enumerate from the STUDIO's own registry through the
    router's ``capable_model_ids`` — two different namespaces. A second
    enumeration here would drift from the one the studio actually renders
    through, which is the only enumeration that matters."""
    try:
        from hugpy_video.intel.studio.tester import enumerate_models
        return tuple(enumerate_models(category))
    except Exception as exc:  # noqa: BLE001 — an empty roster is a finding
        logger.info("k109b: studio enumeration failed for %s (%s: %s)",
                    category, type(exc).__name__, exc)
        return ()


def order_by_residency(models: Sequence[str], task: str = "text-generation"
                       ) -> tuple[tuple[str, ...], dict[str, str]]:
    """Sort a roster so the models already RESIDENT on a worker go first.

    Two reasons, neither of them impatience. (1) Politeness: a resident model
    costs the fleet one forward pass, a cold one costs an eviction plus a load,
    so doing the free work first means an interrupted sweep interrupted the
    cheap half. (2) Evidence density: a sweep that a degrading fleet is going
    to cut short should spend its first hour on the models that can actually
    answer, so the report has content instead of a column of timeouts.

    STABLE within each group (catalog order), so it is reproducible given the
    same fleet state — and the state each model was sorted on is returned
    alongside and recorded in the run's environment, because a sort key nobody
    can see afterwards is a sort key nobody can check."""
    states: dict[str, str] = {}
    # ONE worker-roster read for the whole sort. ``_load_state`` re-lists the
    # workers per call, which is 91 heartbeat reads on this fleet and a minute
    # and a half of wall time spent asking the same question — the answer
    # cannot change usefully inside a single sort anyway, and a sort key that
    # shifted half way through would not be a sort key.
    try:
        from hugpy_engine.placement import get_worker_registry
        from hugpy_oracle.providers import get_load_state_source
        roster = list(get_worker_registry().list_workers(online_only=False) or [])
        load_state_for_model = get_load_state_source().load_state
    except Exception:  # noqa: BLE001
        roster, load_state_for_model = [], None
    by_name = {}
    for row in roster:
        for handle in (row.get("id"), row.get("name")):
            if handle:
                by_name[handle] = row.get("id") or handle
    for model in models:
        try:
            worker = _selected_worker(model, task)
            if load_state_for_model is not None and worker:
                state = load_state_for_model(model, by_name.get(worker, worker))
                if isinstance(state, Mapping):
                    states[model] = ("loaded" if state.get("healthy") else
                                     "loading" if state.get("in_progress")
                                     else "cold")
                    continue
            states[model] = _load_state(model, worker) or "unknown"
        except Exception:  # noqa: BLE001 — an unreadable state sorts mid-pack
            states[model] = "unknown"
    rank = {"loaded": 0, "loading": 1, "unknown": 2, "cold": 3}
    order = {m: i for i, m in enumerate(models)}
    return tuple(sorted(models,
                        key=lambda m: (rank.get(states[m], 2), order[m]))), states


def discover_rosters(config: StationaryConfig | None = None
                     ) -> dict[str, dict[str, Any]]:
    """Every roster this sweep needs, with the catalog's own reasons attached.

    One dict per point kind. ``models`` may be EMPTY and an empty roster with a
    reason is a complete answer — "no TTS model is eligible on this fleet" is
    the finding, not an error to be worked around."""
    config = config or StationaryConfig()
    out: dict[str, dict[str, Any]] = {}

    for kind, capability in (("llm", "text.chat"),
                             ("vlm", "image.understand"),
                             ("tts", "audio.tts")):
        models, reasons = discover_models(capability)
        if kind == "llm" and config.models:
            models = tuple(config.models)
            reasons = reasons + ("roster pinned by --models",)
        load_states: dict[str, str] = {}
        if kind == "llm" and models and not config.models:
            models, load_states = order_by_residency(models)
            reasons = reasons + (
                "roster ordered resident-first (see load_states): an "
                "interrupted sweep should have interrupted the cheap half",)
        if kind == "llm" and config.max_models is not None:
            models = models[:max(0, int(config.max_models))]
        out[kind] = {"capability": capability, "models": list(models),
                     "reasons": list(reasons), "source": "oracle.catalog",
                     "load_states": load_states}

    # IMAGE and VIDEO come from the studio's enumeration, not the catalog's,
    # because the studio is what will actually render them.
    image_models = _studio_models("image")
    out["image"] = {"capability": "image.generate",
                    "models": list(image_models),
                    "reasons": [] if image_models else
                    ["video_intel.studio.tester.enumerate_models('image') "
                     "returned nothing — the main registry lists no servable "
                     "text-to-image model here"],
                    "source": "video_intel.studio.tester"}
    video_models = _studio_models("clip")
    out["video"] = {"capability": "video.generate.t2v",
                    "models": list(video_models),
                    "reasons": [] if video_models else
                    ["video_intel.studio.tester.enumerate_models('clip') "
                     "returned nothing — the studio router says no t2v model "
                     "is servable here"],
                    "source": "video_intel.studio.tester"}
    return out


# ---------------------------------------------------------------------------
# Journal + resume
# ---------------------------------------------------------------------------

#: The per-cell journal. Appended AS EACH CELL HAPPENS: a sweep that runs for
#: six hours and writes its results at the end is a sweep that loses six hours
#: to one restart.
CELLS_FILE: str = "cells.jsonl"
STATE_FILE: str = "state.json"


def load_journal(run_dir: str, *, retry_failed: bool = False
                 ) -> dict[str, dict[str, Any]]:
    """Every completed cell in a run dir, keyed by ``point|operation|model``.

    The LAST row for a key wins, so a cell re-run after a resume replaces its
    predecessor rather than being double-counted in the leaderboard.

    ``retry_failed`` drops the cells whose failure was a DISPATCH fault
    (a timeout or a transport error) so a resume re-runs them. It deliberately
    does NOT drop an ``incapable`` verdict that came from a model answering
    badly — that is a finding, and re-rolling findings until they improve is
    not benchmarking."""
    path = os.path.join(run_dir, CELLS_FILE)
    out: dict[str, dict[str, Any]] = {}
    if not os.path.isfile(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                key = (f"{row.get('point_id')}|{row.get('operation')}|"
                       f"{row.get('model')}")
                if retry_failed and row.get("failure") in ("timeout",
                                                           "dispatch_error"):
                    out.pop(key, None)
                    continue
                out[key] = row
    except OSError as exc:
        logger.warning("k109b: could not read the journal at %s (%s: %s)",
                       path, type(exc).__name__, exc)
    return out


def resume_dir(run_id: str, root: str | None = None) -> str:
    """The run dir for ``--resume <run_id>``, which may be an id or a path."""
    if os.path.isdir(run_id):
        return os.path.abspath(run_id)
    candidate = os.path.join(root or default_run_root(), run_id)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        f"no run dir {run_id!r} under {root or default_run_root()!r} — "
        f"--resume takes the run id of an existing sweep or a path to one")


def new_stationary_run_dir(root: str | None = None, label: str = "stationary"
                           ) -> str:
    """``<root>/oracle-<stamp>-<label>``, with the media subdirs claimed.

    Deliberately the SAME ``oracle-`` prefix k109 uses: ``load_latest_matrix``
    (k114b) lists ``oracle-*`` and picks the newest matrix file by mtime, so a
    k109b run has to land in that namespace or the router would never see it."""
    directory = new_run_dir(root, label=label)
    for sub in ("raw", "frames", "keyframes", "clips", "audio"):
        os.makedirs(os.path.join(directory, sub), exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# Stage helpers shared by every medium
# ---------------------------------------------------------------------------

#: Failures worth ONE retry: the fleet moved under us, or a worker was busy
#: with something (frequently OUR OWN abandoned request — ``run_bounded``
#: cannot kill the thread it walked away from).
_TRANSIENT_MARKERS: tuple[str, ...] = (
    "connection refused", "connection reset", "connection aborted",
    "unreachable", "no route to host", "bad gateway", "service unavailable",
    "502", "503", "504", "broken pipe", "remote end closed",
    "worker is offline", "worker went away", "temporarily unavailable",
    "workerbusy", "worker busy", "all workers busy", "no worker available",
)

#: Failures that mean the FLEET IS DOWN, which is a different question and gets
#: a different answer (pause the sweep). Deliberately a SUBSET: a busy worker
#: is a worker that is alive and working, and pausing a six-hour sweep because
#: three big models in a row queued behind each other would be a bug wearing a
#: safety feature's clothes.
_OUTAGE_MARKERS: tuple[str, ...] = (
    "connection refused", "connection reset", "connection aborted",
    "unreachable", "no route to host", "bad gateway", "service unavailable",
    "502", "503", "504", "remote end closed", "worker is offline",
    "worker went away", "no online worker",
)


def is_transient(detail: str) -> bool:
    """Is this failure the fleet moving under us, rather than the model failing?

    ANOTHER AGENT may be restarting a worker while this sweep runs. A restart
    is not a capability finding, so exactly one retry is spent on it — and then
    the failure IS recorded as that model's result, because a sweep that
    retries forever never finishes and a sweep that hides the retry lies about
    what the fleet did."""
    low = str(detail or "").lower()
    return any(marker in low for marker in _TRANSIENT_MARKERS)


def _sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _vram_delta(before: Mapping[str, Any] | None,
                after: Mapping[str, Any] | None) -> int | None:
    if not (before and after):
        return None
    a, b = before.get("vram_used"), after.get("vram_used")
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return int(b - a)
    return None


def _media_perf(model: str, task: str, latency: float | None,
                config: StationaryConfig, before: Mapping[str, Any] | None,
                after: Mapping[str, Any] | None, worker: str | None,
                load_state: str | None, params: Mapping[str, Any],
                calls: int = 1, output_chars: int = 0,
                sampler: _VramSampler | None = None) -> PerfRecord:
    """A :class:`PerfRecord` for a non-LLM cell. The same record shape the LLM
    cells use, so one performance table covers every medium — the fields a
    medium cannot answer (tokens, tok/s, finish_reason) stay None rather than
    being filled with a zero that would read as a measurement."""
    del task
    return PerfRecord(
        model=model, worker=worker, load_state=load_state, mode="stationary",
        params=dict(params), ceiling_source="stationary:constant",
        latency_s=latency, dispatch_calls=calls, output_chars=output_chars,
        vram_before=before, vram_after=after,
        vram_used_delta_bytes=_vram_delta(before, after),
        **(sampler.perf_fields() if sampler else {}),
        gpu_total_bytes=(before or {}).get("vram_total"),
        vram_reserve_gib=config.vram_reserve_gib)


def probe_text_model(model: str, config: StationaryConfig
                     ) -> tuple[bool, str, float]:
    """The admission probe: can this model answer one trivial prompt at all?

    Returns ``(ok, detail, latency_s)`` and NEVER raises. This exists purely to
    be polite with the fleet's time and the operator's: the catalog lists 88
    models as eligible for ``text.chat`` and several of them are a whisper
    checkpoint, a 3D asset generator and a text encoder. Without a probe each
    of those costs the sweep EIGHT dispatch deadlines to discover; with it,
    one. A failed probe is recorded as the model's verdict at every LLM point,
    with the probe's own error as the reason, so nothing is silently dropped.

    ONE attempt, plus a second ONLY for a transport fault — see the comment on
    the timeout branch below for why a timed-out probe is deliberately not
    retried (the short version: we are the reason the retry fails). Both
    attempts are named in the returned detail when there are two; nothing is
    hidden behind the word "retried"."""
    route = _resolve_route(BENCH_CAPABILITY, model)
    if route is None or getattr(route, "execution", None) != "execute":
        reasons = "; ".join(getattr(route, "reasons", ()) or ()) or \
            "route resolution failed"
        return False, f"no-route: {reasons}"[:300], 0.0
    task = getattr(route, "task", None) or "text-generation"
    body = {"prompt": _no_think(PROBE_PROMPT), "model_key": model,
            "max_new_tokens": 16, "temperature": 0.1, "top_p": 0.9}

    def once() -> tuple[bool, str, float, bool]:
        started = time.monotonic()
        try:
            result = _run_bounded(lambda: _dispatch(task, dict(body)),
                                  config.probe_deadline_s,
                                  f"k109b-probe:{model}")
        except Exception as exc:  # noqa: BLE001 — a failed probe is DATA
            elapsed = round(time.monotonic() - started, 3)
            timed_out = _is_timeout(exc)
            kind = "timeout" if timed_out else type(exc).__name__
            # A bare TIMEOUT is NOT retried. ``run_bounded`` cannot kill the
            # thread it abandoned, so the model we just gave up on is still
            # loading on the worker; an immediate second probe queues behind
            # our own orphan, comes back WorkerBusy, and costs another
            # deadline for nothing. Measured live 2026-08-21: the retry turned
            # a 180s failure into a 219s failure and never once succeeded.
            # Transport faults ARE retried — those are the shape a worker
            # restart makes, and the second attempt genuinely differs.
            return False, f"{kind}: {exc}"[:280], elapsed, (
                not timed_out and is_transient(str(exc)))
        elapsed = round(time.monotonic() - started, 3)
        payload = _payload(result)
        if payload.get("ok") is False or payload.get("error"):
            detail = f"dispatch not-ok: {payload.get('error')}"
            return False, detail[:280], elapsed, is_transient(detail)
        text = str(payload.get("text") or "")
        if not text.strip():
            return False, ("the model returned no text for a one-word probe",
                           )[0], elapsed, False
        return True, f"answered in {elapsed}s: {text.strip()[:60]!r}", elapsed, False

    ok, detail, elapsed, worth_retry = once()
    if ok or not worth_retry:
        return ok, detail, elapsed
    _sleep(TIMEOUT_COOLDOWN_S)
    ok2, detail2, elapsed2, _again = once()
    total = round(elapsed + TIMEOUT_COOLDOWN_S + elapsed2, 3)
    if ok2:
        return True, (f"answered on the SECOND probe ({detail2}); the first "
                      f"attempt was {detail}"), total
    return False, (f"two probes failed — first: {detail} | second: "
                   f"{detail2}")[:400], total


# ---------------------------------------------------------------------------
# Stage 1 — the LLM lifecycle points
# ---------------------------------------------------------------------------


def run_llm_cell(point: Any, operation: str, model: str, *,
                 config: StationaryConfig,
                 registry_version: str | None = None,
                 scenario_version: str = "", scenario_digest: str = ""
                 ) -> tuple[Cell, str]:
    """One (LLM point, model) cell, through k109's own ``run_case``.

    Reused deliberately: ``run_case`` already routes through
    ``router.resolve_route``, dispatches through the one front door, drives
    k110's bounded ask-validate-ONE-repair loop and scores with the k110
    validators. Re-implementing any of that here would produce a second
    measurement of a different pipeline wearing the same name. The ONE thing
    this adds is the stationary preamble — the same tone, character sheets and
    continuity facts for every model — and the retry-once-on-transient rule."""
    from hugpy_oracle.benchmark_cases import stationary_case_for
    from hugpy_oracle.stationary_scenario import SCENARIO_SCREENPLAY, stationary_preamble

    case = stationary_case_for(operation)
    preamble = stationary_preamble(operation)
    run_config = RunConfig(
        mode="normalized", repeats=1, tracks=case.track,
        deadline_s=config.deadline_s, judge=config.judge,
        judge_deadline_s=config.judge_deadline_s,
        judge_model=config.judge_model,
        vram_reserve_gib=config.vram_reserve_gib,
        params_override=dict(STATIONARY_PARAMS), label=config.label)

    attempt, raw = run_case(case, model, config=run_config,
                            screenplay=SCENARIO_SCREENPLAY,
                            registry_version=registry_version,
                            preamble=preamble)
    retried = False
    if attempt.failure == "dispatch_error" and is_transient(
            "; ".join(attempt.deterministic.errors) + " " + (raw or "")):
        retried = True
        _sleep(TIMEOUT_COOLDOWN_S)
        attempt, raw = run_case(case, model, config=run_config,
                                screenplay=SCENARIO_SCREENPLAY,
                                registry_version=registry_version,
                                preamble=preamble)

    structured, _why = parse_json_object(raw) if raw else (None, "")
    refused = (attempt.failure is None and not attempt.deterministic.valid
               and structured is None and looks_like_refusal(raw))
    verdict = classify_verdict(
        produced=bool(raw and raw.strip()),
        validated=attempt.deterministic.valid,
        structured=isinstance(structured, Mapping), refused=refused)
    if attempt.failure in ("timeout", "dispatch_error"):
        verdict = "incapable"

    note = ""
    if attempt.failure:
        note = f"dispatch {attempt.failure}"
    elif verdict == "partial":
        note = (f"structured JSON, {attempt.deterministic.error_count} "
                f"validator error(s): "
                f"{'; '.join(attempt.deterministic.errors[:2])}")[:400]
    elif verdict == "incapable":
        note = "no parseable JSON object in the reply"
    elif verdict == "refused":
        note = "the model declined the task"
    if retried:
        note = (note + " | retried once after a transient dispatch fault").strip(" |")

    return Cell(
        point_id=point.point_id, step=point.step, operation=operation,
        model=model, capability=point.capability, verdict=verdict,
        deterministic=attempt.deterministic, judge=attempt.judge,
        perf=attempt.perf, failure=attempt.failure, gap_code=attempt.gap_code,
        note=note, stage="llm",
        evidence={"case_id": case.case_id, "preamble_chars": len(preamble),
                  "retried": retried,
                  "checks": [c.to_dict() for c in attempt.deterministic.checks]},
        registry_version=registry_version, scenario_version=scenario_version,
        scenario_digest=scenario_digest,
        started_at=attempt.started_at, ended_at=attempt.ended_at), raw


# ---------------------------------------------------------------------------
# The VLM call — one image, one question, one parsed verdict
# ---------------------------------------------------------------------------

VLM_TASK: str = "image-text-to-text"
VLM_MAX_TOKENS: int = 160


def ask_vlm(model: str, image_path: str, prompt: str, *,
            deadline_s: float = 180.0) -> dict[str, Any]:
    """One ``image.understand`` dispatch, parsed with k90c's judge discipline.

    Returns a plain dict — ``ok``, ``text``, ``verdict``, ``score``, ``why``,
    ``latency_s``, ``error`` — and never raises. The parse is
    ``evaluation.parse_judge_verdict``, the SAME parser k109's text judge uses,
    so a VLM's answer and an LLM's answer are read by one set of rules."""
    from hugpy_oracle.evaluation import parse_judge_verdict
    body = {"file": image_path, "prompt": _no_think(prompt),
            "model_key": model, "max_new_tokens": VLM_MAX_TOKENS,
            "temperature": 0.2}
    started = time.monotonic()
    try:
        result = _run_bounded(lambda: _dispatch(VLM_TASK, body), deadline_s,
                              f"k109b-vlm:{model}")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "text": "", "verdict": None, "score": None,
                "why": "", "latency_s": round(time.monotonic() - started, 3),
                "error": (f"{'timeout' if _is_timeout(exc) else type(exc).__name__}"
                          f": {exc}")[:300]}
    latency = round(time.monotonic() - started, 3)
    payload = _payload(result)
    if payload.get("ok") is False or payload.get("error"):
        return {"ok": False, "text": "", "verdict": None, "score": None,
                "why": "", "latency_s": latency,
                "error": f"dispatch not-ok: {payload.get('error')}"[:300]}
    text = _strip_think(str(payload.get("text") or ""))
    parsed = parse_judge_verdict(text)
    return {"ok": bool(text.strip()), "text": text, "verdict": parsed["verdict"],
            "score": parsed["score"], "why": parsed["why"],
            "latency_s": latency,
            "error": None if text.strip() else "the model returned no text"}


#: The shot-spec words a GROUNDED answer names. Lower-cased substrings, and
#: substrings on purpose: "yellow jacket" and "yellow foul-weather jacket" are
#: the same observation and a token-equality test would score the second one
#: lower for being more precise.
_GROUNDING_TERMS: tuple[str, ...] = (
    "yellow", "jacket", "slate", "drysuit", "harbour", "harbor", "wall",
    "buoy", "dusk", "overcast", "two", "wet", "stone", "green",
)

#: How many distinct grounding terms an answer needs before it counts as
#: fully grounded. Four of fourteen: enough that a generic "the image shows two
#: people outdoors" does not clear it, low enough that a correct one-sentence
#: answer is not punished for being terse.
_GROUNDING_TARGET: int = 4


def grounding_score(text: str) -> tuple[float, tuple[str, ...]]:
    """(0-1, the terms found) — is this answer about THIS shot spec?

    Key-INDEPENDENT on purpose. The planted-violation key is only as good as
    the renderer that was asked to plant the violation (see
    ``stationary_scenario.REFERENCE_FRAME_KEY_BASIS``), so the VLM stage needs
    one axis that does not depend on it at all. This is that axis: it measures
    whether the judge answered about the specification it was given rather than
    describing a picture in general, and a model can score 1.0 here while
    disagreeing with the key on every frame."""
    low = str(text or "").lower()
    found = tuple(sorted({t for t in _GROUNDING_TERMS if t in low}))
    return min(1.0, len(found) / _GROUNDING_TARGET), found


# ---------------------------------------------------------------------------
# Stage 2 — the VLM validation point (lifecycle step 14)
# ---------------------------------------------------------------------------


def render_reference_frames(run_dir: str, *, config: StationaryConfig,
                            note: Callable[[str], None] | None = None
                            ) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Render the six reference frames ONCE, with ONE model, at sweep start.

    Every judge then sees the SAME six files. Rendering per judge would mean
    sixteen judges grading sixteen different frame sets and calling the result
    a comparison. The renderer is pinned (``--reference-model``, default
    ``sd-turbo``) for the same reason and is recorded in the run's
    environment.

    Returns ``(frame_id -> path, records)``. A frame that fails to render is
    ABSENT from the mapping and present in ``records`` with its error — the
    judges are then scored over the frames that exist, and how many existed is
    printed in the report."""
    from hugpy_oracle.stationary_scenario import (
        KEYFRAME_HEIGHT,
        KEYFRAME_SEED,
        KEYFRAME_WIDTH,
        REFERENCE_FRAMES,
    )
    paths: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for index, frame in enumerate(REFERENCE_FRAMES):
        dest = os.path.join(run_dir, "frames", f"{frame.frame_id}.png")
        ok, path, error, latency = generate_image(
            config.reference_model, frame.prompt, dest,
            width=KEYFRAME_WIDTH, height=KEYFRAME_HEIGHT,
            seed=KEYFRAME_SEED + index, deadline_s=config.image_deadline_s)
        record = {"frame_id": frame.frame_id, "ok": ok, "path": path,
                  "error": error, "latency_s": latency,
                  "renderer": config.reference_model,
                  "violations": list(frame.violations),
                  "expected_verdict": frame.expected_verdict,
                  "prompt": frame.prompt}
        if ok and path:
            substance, detail = image_carries_content(path)
            record["levels"] = detail
            record["blank"] = not substance
            if substance:
                paths[frame.frame_id] = path
            else:
                record["ok"] = False
                record["error"] = f"rendered frame is blank: {detail}"
        records.append(record)
        if note:
            note(f"reference frame {frame.frame_id}: "
                 f"{'ok' if record['ok'] else 'FAILED'} "
                 f"{record.get('error') or record.get('levels') or ''}")
    return paths, records


#: The file a run dir may carry to override the prompt-derived frame key with
#: a HUMAN read of the pixels. Optional by design: a sweep run with nobody to
#: look at the frames still produces rows, and says which key it used.
CONFIRMATION_FILE: str = os.path.join("frames", "human_confirmation.json")


def load_frame_confirmation(run_dir: str) -> dict[str, Any]:
    """A human's read of THIS run's rendered frames, or ``{}``.

    WHY THIS EXISTS, and it is the most important honesty seam in the wave.
    The reference frames are RENDERS OF PROMPTS. Their prompt-derived key says
    what the renderer was ASKED for, and a renderer that ignored a planted
    violation makes that key wrong — at which point a judge is marked down for
    being correct. Live on 2026-08-21, sd-turbo did exactly that: the
    third-person violation never rendered, and BOTH frames labelled compliant
    turned out to violate the spec for reasons nobody planted.

    So a run dir may carry ``frames/human_confirmation.json``: per frame, the
    verdict and violation labels a person actually saw, plus ``cues`` — the
    words a judge that spotted THAT fault would plausibly use. When it is
    present the VLM stage scores against it and records
    ``key_source: human-confirmed``; when it is absent the stage scores against
    the prompt and records ``key_source: render-prompt``. Never silently
    either way."""
    path = os.path.join(run_dir, CONFIRMATION_FILE)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
    except Exception as exc:  # noqa: BLE001 — an unreadable key is no key
        logger.warning("k109b: could not read %s (%s: %s)", path,
                       type(exc).__name__, exc)
        return {}
    return data if isinstance(data, dict) and data.get("frames") else {}


def run_vlm_cell(point: Any, model: str, frames: Mapping[str, str], *,
                 config: StationaryConfig,
                 registry_version: str | None = None,
                 scenario_version: str = "", scenario_digest: str = "",
                 confirmation: Mapping[str, Any] | None = None
                 ) -> tuple[Cell, str]:
    """One (VLM, step 14) cell: this judge, all six reference frames.

    THREE axes, reported separately and never silently summed:

      AGREEMENT  did it reach the verdict the key expects? Scored against the
                 HUMAN-CONFIRMED key when the run dir carries one, and against
                 the render-prompt key otherwise — and the row records WHICH,
                 every time, because they can disagree.
      GROUNDING  did it answer about THIS shot spec at all, or describe a
                 picture in general? Key-independent, so a judge stays
                 measurable even when the key is wrong.
      NAMES_THE_ACTUAL_FAULT  did it name the fault the frame really carries?
                 This is the axis that survives the degenerate case: when the
                 renderer could not satisfy the shot spec on ANY frame, every
                 expected verdict is NO and agreement stops discriminating,
                 because a judge that answers NO six times scores 100%.

    ``capable`` needs the form answered, the answer grounded in this shot, and
    the real fault named — agreement alone is deliberately not enough. A judge
    that answers the form but disagrees with the key is ``partial``, because
    "it judges, differently" is a genuinely different finding from "it cannot
    judge"."""
    from hugpy_oracle.stationary_scenario import (
        REFERENCE_FRAMES,
        REFERENCE_FRAME_KEY_BASIS,
        build_frame_judge_prompt,
    )
    prompt = build_frame_judge_prompt()
    confirmed = dict((confirmation or {}).get("frames") or {})
    key_source = "human-confirmed" if confirmed else "render-prompt"
    key_basis = (str((confirmation or {}).get("basis") or "")
                 or REFERENCE_FRAME_KEY_BASIS)

    def expected_for(frame_id: str) -> tuple[str, tuple[str, ...],
                                             tuple[str, ...]]:
        """(verdict, violations, cues) for one frame, from whichever key this
        run is using. The cues are empty on the prompt-derived key because a
        prompt cannot tell you what words describe a fault that may not have
        rendered."""
        row = confirmed.get(frame_id)
        if isinstance(row, Mapping):
            return (str(row.get("expected_verdict") or "NO"),
                    tuple(row.get("violations") or ()),
                    tuple(str(c).lower() for c in row.get("cues") or ()))
        frame = by_id[frame_id]
        return frame.expected_verdict, frame.violations, ()

    by_id = {f.frame_id: f for f in REFERENCE_FRAMES}
    worker = _selected_worker(model, VLM_TASK)
    load_state = _load_state(model, worker)
    before = _vram_for(_vram_snapshot(), worker)
    sampler = _VramSampler(worker, sample_ms=config.vram_sample_ms).start()
    started_at = _utc_now()

    answers: list[dict[str, Any]] = []
    transcript: list[str] = []
    total_latency = 0.0
    timeouts = 0
    for frame_id, path in sorted(frames.items()):
        reply = ask_vlm(model, path, prompt, deadline_s=config.vlm_deadline_s)
        if reply.get("error") and is_transient(str(reply["error"])):
            _sleep(TIMEOUT_COOLDOWN_S)
            reply = ask_vlm(model, path, prompt,
                            deadline_s=config.vlm_deadline_s)
            reply["retried"] = True
        total_latency += float(reply.get("latency_s") or 0.0)
        if "timeout" in str(reply.get("error") or "").lower():
            timeouts += 1
        want, violations, cues = expected_for(frame_id)
        text = reply.get("text") or ""
        ground, terms = grounding_score(text)
        low = text.lower()
        hit_cues = [c for c in cues if c in low]
        answers.append({
            "frame_id": frame_id, "expected": want,
            "violations": list(violations), "key_source": key_source,
            "verdict": reply.get("verdict"), "score": reply.get("score"),
            "why": (reply.get("why") or "")[:300],
            "agreed": reply.get("verdict") == want,
            "violation_hit": bool(hit_cues) if cues else None,
            "cues_found": hit_cues[:6],
            "grounding": round(ground, 3), "grounding_terms": list(terms),
            "latency_s": reply.get("latency_s"), "error": reply.get("error"),
            "retried": bool(reply.get("retried"))})
        transcript.append(f"--- {frame_id} (expected {want}, key "
                          f"{key_source}) ---\n"
                          f"{reply.get('text') or reply.get('error') or ''}")
        if timeouts >= TIMEOUT_ABORT_STREAK:
            transcript.append(f"[aborted after {timeouts} timeouts]")
            break

    parsed = [a for a in answers if a["verdict"] in ("YES", "NO")]
    agreed = [a for a in parsed if a["agreed"]]
    grounding = (sum(a["grounding"] for a in answers) / len(answers)
                 if answers else 0.0)
    agreement = len(agreed) / len(parsed) if parsed else 0.0
    parse_rate = len(parsed) / len(answers) if answers else 0.0

    # DISCRIMINATION. When every frame's confirmed verdict is NO — which is
    # what happens when the renderer could not satisfy the shot spec on ANY
    # frame — raw agreement stops measuring anything: a judge that answers NO
    # six times scores 100%. So the key also carries CUES, and violation_hit
    # asks the discriminating question: did the judge name the fault that is
    # actually in this picture? A judge can score 1.0 on agreement and 0.0
    # here, and that gap is the finding.
    cued = [a for a in parsed if a.get("violation_hit") is not None]
    hits = [a for a in cued if a["violation_hit"]]
    violation_hit = len(hits) / len(cued) if cued else None
    wants = {a["expected"] for a in answers}
    discriminating = len(wants) > 1
    caught = [a for a in parsed if a["violations"] and a["agreed"]]
    planted = [a for a in answers if a["violations"]]

    checks = (
        BenchCheck("answers_the_form", parse_rate == 1.0, round(parse_rate, 4),
                   f"{len(parsed)}/{len(answers)} frame(s) returned a "
                   f"parseable VERDICT"),
        BenchCheck("grounded", grounding >= 1.0, round(grounding, 4),
                   f"mean grounding {grounding:.2f} over "
                   f"{len(answers)} answer(s) (target: {_GROUNDING_TARGET} "
                   f"distinct shot-spec terms named)"),
        BenchCheck("agrees_with_key",
                   bool(parsed) and agreement == 1.0, round(agreement, 4),
                   f"{len(agreed)}/{len(parsed)} verdict(s) match the "
                   f"{key_source} key" +
                   ("" if discriminating else
                    " — WARNING: every frame in this run has the SAME expected "
                    "verdict, so this axis cannot tell a discriminating judge "
                    "from one that answers the same way every time")),
        BenchCheck("names_the_actual_fault",
                   violation_hit == 1.0 if violation_hit is not None else False,
                   None if violation_hit is None else round(violation_hit, 4),
                   (f"{len(hits)}/{len(cued)} answer(s) name a fault the frame "
                    f"really carries" if cued else
                    f"no cue list on this key ({key_source}) — the axis is "
                    f"unscored rather than passed")),
    )
    errors = tuple(f"{a['frame_id']}: {a['error']}" for a in answers
                   if a.get("error"))
    # "capable" needs the judge to answer the form, to be talking about THIS
    # shot, and to name the real fault. Agreement alone is deliberately not
    # enough when the key cannot discriminate.
    valid = (bool(answers) and parse_rate == 1.0 and grounding >= 1.0
             and (agreement == 1.0 if discriminating else True)
             and (violation_hit is not None and violation_hit >= 0.5))
    deterministic = DeterministicScore(
        valid=valid, error_count=len(errors),
        preservation=round(parse_rate, 4),
        contradiction_rate=round(1.0 - agreement, 4),
        completeness=round(len(answers) / max(1, len(frames)), 4),
        accuracy=(round(violation_hit, 4) if violation_hit is not None
                  else (round(agreement, 4) if parsed else None)),
        constraint_adherence=round(grounding, 4),
        checks=checks, errors=errors[:6])

    refused = bool(answers) and not parsed and any(
        looks_like_refusal(t) for t in transcript)
    verdict = classify_verdict(
        produced=bool(parsed) or bool([a for a in answers if not a["error"]]),
        validated=valid, structured=bool(parsed), refused=refused)
    if not answers or timeouts >= TIMEOUT_ABORT_STREAK:
        verdict = "incapable"

    sampler.stop()
    after = _vram_for(_vram_snapshot(), worker)
    perf = _media_perf(model, VLM_TASK, round(total_latency, 3), config,
                       before, after, worker, load_state,
                       {"max_new_tokens": VLM_MAX_TOKENS, "temperature": 0.2},
                       calls=len(answers),
                       output_chars=sum(len(t) for t in transcript),
                       sampler=sampler)
    note = (f"{len(parsed)}/{len(answers)} parsed, agreement "
            f"{agreement * 100:.0f}% ({key_source}), grounding "
            f"{grounding:.2f}, names-the-fault "
            f"{'n/a' if violation_hit is None else f'{violation_hit * 100:.0f}%'}")
    if timeouts:
        note += f", {timeouts} timeout(s)"
    return Cell(
        point_id=point.point_id, step=point.step,
        operation=point.operations[0], model=model,
        capability=point.capability, verdict=verdict,
        deterministic=deterministic,
        judge=JudgeScore(detail="this point IS the judge — no second judge "
                                "grades a judge in this wave"),
        perf=perf,
        failure=("timeout" if timeouts >= TIMEOUT_ABORT_STREAK else
                 ("dispatch_error" if errors and not parsed else None)),
        gap_code=None, note=note, stage="vlm",
        evidence={"answers": answers, "agreement": round(agreement, 4),
                  "grounding": round(grounding, 4),
                  "violation_hit": violation_hit,
                  "parse_rate": round(parse_rate, 4),
                  "key_source": key_source,
                  "key_discriminates": discriminating,
                  "planted_caught": f"{len(caught)}/{len(planted)}",
                  "frames_judged": sorted(frames),
                  "key_basis": key_basis},
        registry_version=registry_version, scenario_version=scenario_version,
        scenario_digest=scenario_digest,
        started_at=started_at, ended_at=_utc_now()), "\n\n".join(transcript)


# ---------------------------------------------------------------------------
# Stage 3 — the keyframe point (lifecycle step 12, render seed)
# ---------------------------------------------------------------------------

IMAGE_TASK: str = "text-to-image"


def generate_image(model: str, prompt: str, dest_path: str, *,
                   width: int, height: int, seed: int,
                   negative_prompt: str = "",
                   deadline_s: float = 300.0
                   ) -> tuple[bool, str, str | None, float]:
    """One ``image.generate`` dispatch, landed as a readable local file.

    Returns ``(ok, path, error, latency_s)``. The b64 fallback is not an
    optimisation: generation runs on a WORKER, so the path the runner reports
    is a path on that worker's disk — unreadable here and therefore unjudgeable.
    The bytes already ride back inline, so the honest move is to write them
    where the judge can actually open them. (Same reasoning, same shape as
    ``runtime._materialize_audio``.)"""
    import base64
    import shutil
    body: dict[str, Any] = {"prompt": prompt, "model_key": model,
                            "num_images": 1, "width": int(width),
                            "height": int(height), "seed": int(seed),
                            "return_b64": True}
    if negative_prompt:
        body["negative_prompt"] = negative_prompt
    started = time.monotonic()
    try:
        result = _run_bounded(lambda: _dispatch(IMAGE_TASK, body), deadline_s,
                              f"k109b-image:{model}")
    except Exception as exc:  # noqa: BLE001
        return (False, "", (f"{'timeout' if _is_timeout(exc) else type(exc).__name__}"
                            f": {exc}")[:300],
                round(time.monotonic() - started, 3))
    latency = round(time.monotonic() - started, 3)
    payload = _payload(result)
    if payload.get("ok") is False or payload.get("error"):
        return False, "", f"dispatch not-ok: {payload.get('error')}"[:300], latency
    images = payload.get("images") or ()
    if not images:
        return False, "", "the plane answered ok and produced no image", latency
    first = images[0] if isinstance(images[0], Mapping) else {}
    try:
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        source = first.get("path") or ""
        if source and os.path.isfile(source):
            if os.path.abspath(source) != os.path.abspath(dest_path):
                shutil.copyfile(source, dest_path)
            return True, dest_path, None, latency
        if first.get("b64"):
            with open(dest_path, "wb") as handle:
                handle.write(base64.b64decode(first["b64"]))
            return True, dest_path, None, latency
    except Exception as exc:  # noqa: BLE001
        return False, "", f"could not land the image: {type(exc).__name__}: {exc}"[:300], latency
    return (False, "", "the result carried neither a readable path nor inline "
            "bytes — nothing here can open it", latency)


def _judge_frame(judge_model: str | None, candidate: str, image_path: str,
                 prompt: str, *, deadline_s: float) -> dict[str, Any]:
    """VLM-score one produced frame with a judge that is NEVER the candidate.

    The refusal is hard, exactly as k109 made it for the text judge: a model
    grading its own render is the one measurement in this sweep that would be
    worthless AND look fine. Here the two rosters are disjoint by construction
    (an image.generate model is not an image.understand model on this fleet),
    so the check costs nothing — and it is present precisely because that
    happy accident is not a guarantee."""
    if not judge_model:
        return {"available": False, "refused": False,
                "detail": "no VLM judge was resolved for this run"}
    if judge_model == candidate:
        return {"available": False, "refused": True,
                "judge_model": judge_model,
                "detail": (f"REFUSED: the only judge resolved is the candidate "
                           f"{candidate!r} itself; a model grading its own "
                           f"output is not evidence")}
    reply = ask_vlm(judge_model, image_path, prompt, deadline_s=deadline_s)
    ground, terms = grounding_score(reply.get("text") or "")
    return {"available": bool(reply.get("verdict") or reply.get("score")),
            "refused": False, "judge_model": judge_model,
            "verdict": reply.get("verdict") or "unavailable",
            "score": reply.get("score"), "why": (reply.get("why") or "")[:300],
            "grounding": round(ground, 3), "grounding_terms": list(terms),
            "latency_s": reply.get("latency_s"),
            "detail": reply.get("error") or "judged by a VLM that is not the "
                                            "candidate"}


def run_image_cell(point: Any, model: str, *, config: StationaryConfig,
                   run_dir: str, judge_model: str | None,
                   registry_version: str | None = None,
                   scenario_version: str = "", scenario_digest: str = ""
                   ) -> tuple[Cell, str]:
    """One (image model, step 12 render seed) cell: the SAME keyframe prompt,
    the same geometry, the same seed, once."""
    from hugpy_oracle.stationary_scenario import (
        KEYFRAME_HEIGHT,
        KEYFRAME_NEGATIVE_PROMPT,
        KEYFRAME_PROMPT,
        KEYFRAME_SEED,
        KEYFRAME_WIDTH,
        build_keyframe_judge_prompt,
    )
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", model)
    dest = os.path.join(run_dir, "keyframes", f"{safe}.png")
    worker = _selected_worker(model, IMAGE_TASK)
    load_state = _load_state(model, worker)
    before = _vram_for(_vram_snapshot(), worker)
    sampler = _VramSampler(worker, sample_ms=config.vram_sample_ms).start()
    started_at = _utc_now()

    ok, path, error, latency = generate_image(
        model, KEYFRAME_PROMPT, dest, width=KEYFRAME_WIDTH,
        height=KEYFRAME_HEIGHT, seed=KEYFRAME_SEED,
        negative_prompt=KEYFRAME_NEGATIVE_PROMPT,
        deadline_s=config.image_deadline_s)
    retried = False
    if not ok and error and is_transient(error):
        retried = True
        _sleep(TIMEOUT_COOLDOWN_S)
        ok, path, error, latency = generate_image(
            model, KEYFRAME_PROMPT, dest, width=KEYFRAME_WIDTH,
            height=KEYFRAME_HEIGHT, seed=KEYFRAME_SEED,
            negative_prompt=KEYFRAME_NEGATIVE_PROMPT,
            deadline_s=config.image_deadline_s)

    levels = image_levels(path) if ok and path else None
    substance, level_detail = (image_carries_content(path) if ok and path
                               else (False, error or "no image produced"))
    geometry_ok = bool(levels) and levels.get("width") == KEYFRAME_WIDTH \
        and levels.get("height") == KEYFRAME_HEIGHT

    judged: dict[str, Any] = {"available": False, "refused": False,
                              "detail": "not judged: no frame to judge"}
    if ok and substance:
        judged = _judge_frame(judge_model, model, path,
                              build_keyframe_judge_prompt(),
                              deadline_s=config.vlm_deadline_s)

    checks = (
        BenchCheck("produced_a_frame", bool(ok), bool(ok),
                   error or f"wrote {path}"),
        BenchCheck("not_blank", bool(substance), bool(substance), level_detail),
        BenchCheck("geometry_honoured", geometry_ok,
                   f"{(levels or {}).get('width')}x{(levels or {}).get('height')}",
                   f"requested {KEYFRAME_WIDTH}x{KEYFRAME_HEIGHT}"),
    )
    errors = tuple(c.detail for c in checks if not c.passed)
    valid = bool(ok and substance)
    deterministic = DeterministicScore(
        valid=valid, error_count=len(errors),
        completeness=1.0 if ok else 0.0,
        constraint_adherence=1.0 if geometry_ok else 0.0,
        checks=checks, errors=errors)
    judge = JudgeScore(
        judge_model=judged.get("judge_model"),
        verdict=str(judged.get("verdict") or "unavailable"),
        score=judged.get("score"), why=str(judged.get("why") or ""),
        available=bool(judged.get("available")),
        refused=bool(judged.get("refused")),
        detail=str(judged.get("detail") or ""))

    verdict = classify_verdict(produced=bool(ok), validated=valid,
                               structured=bool(ok), refused=False)
    if not ok and error and "timeout" in error.lower():
        verdict = "incapable"
    note = level_detail if ok else (error or "no image")
    if retried:
        note += " | retried once after a transient dispatch fault"

    sampler.stop()
    after = _vram_for(_vram_snapshot(), worker)
    perf = _media_perf(model, IMAGE_TASK, latency, config, before, after,
                       worker, load_state,
                       {"width": KEYFRAME_WIDTH, "height": KEYFRAME_HEIGHT,
                        "seed": KEYFRAME_SEED, "num_images": 1},
                       output_chars=_file_size(path) or 0 if ok else 0,
                       sampler=sampler)
    return Cell(
        point_id=point.point_id, step=point.step,
        operation=point.operations[0], model=model,
        capability=point.capability, verdict=verdict,
        deterministic=deterministic, judge=judge, perf=perf,
        failure=None if ok else ("timeout" if error and "timeout" in
                                 error.lower() else "dispatch_error"),
        note=note[:400], stage="image",
        artifact_ref=os.path.relpath(path, run_dir) if ok and path else "",
        evidence={"levels": levels, "judge": judged, "retried": retried,
                  "prompt_digest": _short_digest(KEYFRAME_PROMPT),
                  "error": error},
        registry_version=registry_version, scenario_version=scenario_version,
        scenario_digest=scenario_digest,
        started_at=started_at, ended_at=_utc_now()), (error or "")


def _short_digest(text: str) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Stage 4 — the clip render point (lifecycle step 13)
# ---------------------------------------------------------------------------


def run_video_cell(point: Any, model: str, *, config: StationaryConfig,
                   run_dir: str, judge_model: str | None,
                   registry_version: str | None = None,
                   scenario_version: str = "", scenario_digest: str = ""
                   ) -> tuple[Cell, str]:
    """One (clip model, step 13) cell: ONE short low-res clip of the same shot.

    Rendered through ``video_intel.studio.tester._generate_video_once`` — the
    fleet's own model-battery render path, which already learned the two things
    a naive call gets wrong: it goes through ``render_clip`` (so the VRAM
    budget autofits to the serving worker instead of reaching the router as
    None) and it mints a UNIQUE render id per attempt (so the worker's
    idempotent keying does not hand back the previous attempt's terminal
    state without rendering a pixel). Re-implementing that here would
    re-introduce both bugs."""
    from hugpy_oracle.stationary_scenario import (
        CLIP_FPS,
        CLIP_HEIGHT,
        CLIP_PROMPT,
        CLIP_SEED,
        CLIP_WIDTH,
        build_clip_judge_prompt,
    )
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", model)
    worker = _selected_worker(model, "text-to-video")
    load_state = _load_state(model, worker)
    before = _vram_for(_vram_snapshot(), worker)
    sampler = _VramSampler(worker, sample_ms=config.vram_sample_ms).start()
    started_at = _utc_now()
    out_root = os.path.join(run_dir, "clips")
    os.makedirs(out_root, exist_ok=True)

    def render() -> tuple[bool, str, str | None, float]:
        from hugpy_video.intel.studio.tester import _generate_video_once
        started = time.monotonic()
        try:
            ok, uri, error = _run_bounded(
                lambda: _generate_video_once(
                    model, CLIP_PROMPT, width=CLIP_WIDTH, height=CLIP_HEIGHT,
                    fps=CLIP_FPS, seed=CLIP_SEED, out_root=out_root),
                config.video_deadline_s, f"k109b-clip:{model}")
        except Exception as exc:  # noqa: BLE001
            return (False, "",
                    (f"{'timeout' if _is_timeout(exc) else type(exc).__name__}"
                     f": {exc}")[:400], round(time.monotonic() - started, 3))
        return bool(ok), str(uri or ""), error, round(time.monotonic() - started, 3)

    ok, uri, error, latency = render()
    retried = False
    if not ok and error and is_transient(error):
        retried = True
        _sleep(TIMEOUT_COOLDOWN_S)
        ok, uri, error, latency = render()

    readable = bool(ok and uri and os.path.isfile(uri))
    media = probe_media(uri) if readable else None
    frames = (media or {}).get("frames")
    geometry_ok = bool(media) and media.get("width") == CLIP_WIDTH \
        and media.get("height") == CLIP_HEIGHT
    moving = bool(media) and (frames is None or frames > 1)

    still = ""
    judged: dict[str, Any] = {"available": False, "refused": False,
                              "detail": "not judged: no readable clip"}
    if readable:
        still = extract_middle_frame(
            uri, os.path.join(run_dir, "clips", f"{safe}-mid.png")) or ""
        if still:
            judged = _judge_frame(judge_model, model, still,
                                  build_clip_judge_prompt(),
                                  deadline_s=config.vlm_deadline_s)
        else:
            judged["detail"] = ("not judged: ffmpeg could not lift a frame "
                                "from the produced file")

    checks = (
        BenchCheck("produced_a_clip", readable, readable,
                   error or (f"wrote {uri}" if ok else "no file") +
                   ("" if readable or not ok else
                    " — the runner reported a path this box cannot read")),
        BenchCheck("decodable", bool(media and media.get("video")),
                   bool(media and media.get("video")),
                   json.dumps(media, default=str)[:200] if media
                   else "ffprobe could not read a video stream"),
        BenchCheck("geometry_honoured", geometry_ok,
                   f"{(media or {}).get('width')}x{(media or {}).get('height')}",
                   f"requested {CLIP_WIDTH}x{CLIP_HEIGHT} @ {CLIP_FPS}fps"),
        BenchCheck("more_than_one_frame", moving, frames,
                   f"{frames} frame(s)" if frames is not None
                   else "frame count not reported by ffprobe"),
    )
    errors = tuple(c.detail for c in checks if not c.passed)
    valid = readable and bool(media and media.get("video")) and moving
    deterministic = DeterministicScore(
        valid=valid, error_count=len(errors),
        completeness=1.0 if readable else 0.0,
        constraint_adherence=1.0 if geometry_ok else 0.0,
        checks=checks, errors=errors)
    judge = JudgeScore(
        judge_model=judged.get("judge_model"),
        verdict=str(judged.get("verdict") or "unavailable"),
        score=judged.get("score"), why=str(judged.get("why") or ""),
        available=bool(judged.get("available")),
        refused=bool(judged.get("refused")),
        detail=str(judged.get("detail") or ""))
    verdict = classify_verdict(produced=bool(ok), validated=valid,
                               structured=readable, refused=False)
    sampler.stop()
    after = _vram_for(_vram_snapshot(), worker)
    perf = _media_perf(model, "text-to-video", latency, config, before, after,
                       worker, load_state,
                       {"width": CLIP_WIDTH, "height": CLIP_HEIGHT,
                        "fps": CLIP_FPS, "seed": CLIP_SEED},
                       output_chars=_file_size(uri) or 0 if readable else 0,
                       sampler=sampler)
    note = (json.dumps(media, default=str)[:200] if media
            else (error or "no clip"))
    if retried:
        note += " | retried once after a transient dispatch fault"
    return Cell(
        point_id=point.point_id, step=point.step,
        operation=point.operations[0], model=model,
        capability=point.capability, verdict=verdict,
        deterministic=deterministic, judge=judge, perf=perf,
        failure=None if valid else ("timeout" if error and "timeout" in
                                    error.lower() else "dispatch_error"),
        note=note[:400], stage="video",
        artifact_ref=os.path.relpath(uri, run_dir)
        if readable and uri.startswith(run_dir) else (uri if readable else ""),
        evidence={"media": media, "judge": judged, "retried": retried,
                  "still": still, "error": error,
                  "prompt_digest": _short_digest(CLIP_PROMPT)},
        registry_version=registry_version, scenario_version=scenario_version,
        scenario_digest=scenario_digest,
        started_at=started_at, ended_at=_utc_now()), (error or "")


# ---------------------------------------------------------------------------
# Stage 5 — the TTS point (lifecycle step 16, audio)
# ---------------------------------------------------------------------------

TTS_TASK: str = "text-to-speech"
ASR_TASK: str = "automatic-speech-recognition"


def transcribe(path: str, *, model: str | None = None,
               deadline_s: float = 300.0) -> dict[str, Any]:
    """Round-trip ASR over a produced wav, through the normal dispatch path.

    The transcriber is resolved from the catalog's own
    ``audio.transcribe.word_timestamps`` roster, never hardcoded: a sweep that
    pinned a whisper build by name would keep transcribing with it after the
    fleet replaced it."""
    if model is None:
        models, reasons = discover_models("audio.transcribe.word_timestamps")
        if not models:
            return {"ok": False, "text": "", "words": [],
                    "error": ("no ASR model is eligible on this fleet, so the "
                              "round trip cannot be run: " +
                              "; ".join(reasons))[:300]}
        model = models[0]
    body = {"file": path, "model_key": model, "word_timestamps": True}
    started = time.monotonic()
    try:
        result = _run_bounded(lambda: _dispatch(ASR_TASK, body), deadline_s,
                              f"k109b-asr:{model}")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "text": "", "words": [], "model": model,
                "error": (f"{'timeout' if _is_timeout(exc) else type(exc).__name__}"
                          f": {exc}")[:300],
                "latency_s": round(time.monotonic() - started, 3)}
    latency = round(time.monotonic() - started, 3)
    payload = _payload(result)
    if payload.get("ok") is False or payload.get("error"):
        return {"ok": False, "text": "", "words": [], "model": model,
                "error": f"dispatch not-ok: {payload.get('error')}"[:300],
                "latency_s": latency}
    text = str(payload.get("text") or "")
    words: list[Any] = []
    for segment in payload.get("segments") or ():
        if isinstance(segment, Mapping) and segment.get("words"):
            words.extend(segment["words"])
    if not words:
        words = list(payload.get("words") or ()) or text.split()
    return {"ok": bool(text.strip()), "text": text, "words": words,
            "model": model, "latency_s": latency,
            "error": None if text.strip() else "the ASR returned no text"}


def run_tts_cell(point: Any, model: str, *, config: StationaryConfig,
                 run_dir: str, registry_version: str | None = None,
                 scenario_version: str = "", scenario_digest: str = ""
                 ) -> tuple[Cell, str]:
    """One (TTS model, step 16 audio) cell: the same locked line, once.

    THREE checks, and the middle one is the one that matters. A wav that
    exists, has the right duration and holds digital silence passes every
    technical check ever written for this pipeline — that fault was live on
    this fleet on 2026-08-21 — so the CONTENT guard measures the samples and a
    silent wav scores 0 with EMPTY_OUTPUT no matter how correct everything
    else about it is. The round-trip ASR is the third check and is what turns
    'it made a noise' into 'it said the line'."""
    from hugpy_oracle.speech import check_lines_present
    from hugpy_oracle.stationary_scenario import TTS_LINE, TTS_SPEAKER
    from hugpy_oracle.runtime import extract_artifacts

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", model)
    worker = _selected_worker(model, TTS_TASK)
    load_state = _load_state(model, worker)
    before = _vram_for(_vram_snapshot(), worker)
    sampler = _VramSampler(worker, sample_ms=config.vram_sample_ms).start()
    started_at = _utc_now()

    def synthesize() -> tuple[dict[str, Any] | None, str | None, float]:
        body = {"text": TTS_LINE, "model_key": model}
        started = time.monotonic()
        try:
            result = _run_bounded(lambda: _dispatch(TTS_TASK, body),
                                  config.tts_deadline_s, f"k109b-tts:{model}")
        except Exception as exc:  # noqa: BLE001
            return None, (f"{'timeout' if _is_timeout(exc) else type(exc).__name__}"
                          f": {exc}")[:300], round(time.monotonic() - started, 3)
        latency = round(time.monotonic() - started, 3)
        payload = _payload(result)
        if payload.get("ok") is False or payload.get("error"):
            return None, f"dispatch not-ok: {payload.get('error')}"[:300], latency
        return payload, None, latency

    payload, error, latency = synthesize()
    retried = False
    if payload is None and error and is_transient(error):
        retried = True
        _sleep(TIMEOUT_COOLDOWN_S)
        payload, error, latency = synthesize()

    artifacts = extract_artifacts("audio.tts", payload) if payload else []
    audio = next((a for a in artifacts if a.get("kind") == "audio"), None)
    path = str((audio or {}).get("uri") or "")
    landed = ""
    if path and os.path.isfile(path):
        import shutil
        landed = os.path.join(run_dir, "audio", f"{safe}.wav")
        try:
            os.makedirs(os.path.dirname(landed), exist_ok=True)
            if os.path.abspath(path) != os.path.abspath(landed):
                shutil.copyfile(path, landed)
        except OSError:
            landed = path

    produced = bool(landed and os.path.isfile(landed))
    substance, level_detail = (audio_carries_sound(landed) if produced
                               else (False, error or "no audio produced"))

    asr: dict[str, Any] = {"ok": False, "error": "not transcribed: no audio "
                                                 "with sound to transcribe"}
    line_check = None
    if produced and substance:
        asr = transcribe(landed, deadline_s=config.asr_deadline_s)
        line_check = check_lines_present([TTS_LINE], asr.get("words") or ())

    checks = (
        BenchCheck("produced_audio", produced, produced,
                   error or (f"wrote {landed}" if produced else "no wav")),
        BenchCheck("carries_sound", bool(substance), bool(substance),
                   level_detail),
        BenchCheck("says_the_line",
                   bool(line_check and line_check.passed),
                   getattr(line_check, "value", None),
                   getattr(line_check, "detail", None) or
                   str(asr.get("error") or "")),
    )
    errors = tuple(str(c.detail) for c in checks if not c.passed)
    valid = produced and bool(substance) and bool(line_check
                                                  and line_check.passed)
    deterministic = DeterministicScore(
        valid=valid, error_count=len(errors),
        preservation=(1.0 if line_check and line_check.passed else 0.0)
        if line_check is not None else None,
        completeness=1.0 if produced else 0.0,
        accuracy=(1.0 if line_check and line_check.passed else 0.0)
        if line_check is not None else None,
        checks=checks, errors=errors)
    gap_code = None
    if produced and not substance:
        gap_code = "EMPTY_OUTPUT"
    verdict = classify_verdict(produced=produced, validated=valid,
                               structured=produced and bool(substance),
                               refused=False)
    sampler.stop()
    after = _vram_for(_vram_snapshot(), worker)
    perf = _media_perf(model, TTS_TASK, latency, config, before, after, worker,
                       load_state, {"text_chars": len(TTS_LINE),
                                    "speaker": TTS_SPEAKER},
                       output_chars=_file_size(landed) or 0 if produced else 0,
                       sampler=sampler)
    note = level_detail if produced else (error or "no audio")
    if gap_code:
        note = f"EMPTY_OUTPUT — {note}"
    if retried:
        note += " | retried once after a transient dispatch fault"
    return Cell(
        point_id=point.point_id, step=point.step,
        operation=point.operations[0], model=model,
        capability=point.capability, verdict=verdict,
        deterministic=deterministic,
        judge=JudgeScore(detail="round-trip ASR is this point's second layer; "
                                "no LLM judge grades a waveform"),
        perf=perf,
        failure=None if produced else ("timeout" if error and "timeout" in
                                       error.lower() else "dispatch_error"),
        gap_code=gap_code, note=note[:400], stage="tts",
        artifact_ref=os.path.relpath(landed, run_dir) if produced else "",
        evidence={"levels": level_detail, "asr": {k: v for k, v in asr.items()
                                                 if k != "words"},
                  "transcript": str(asr.get("text") or "")[:400],
                  "line_check": getattr(line_check, "detail", None),
                  "retried": retried, "duration_s": (audio or {}).get("duration_s"),
                  "sample_rate": (audio or {}).get("sample_rate"),
                  "error": error},
        registry_version=registry_version, scenario_version=scenario_version,
        scenario_digest=scenario_digest,
        started_at=started_at, ended_at=_utc_now()), str(asr.get("text") or "")


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------


def _write_state(run_dir: str, sweep_state: Mapping[str, Any]) -> None:
    _atomic_write(os.path.join(run_dir, STATE_FILE),
                  json.dumps(dict(sweep_state), indent=1, sort_keys=True,
                             default=str))


def resume_command(run_id: str) -> str:
    """The exact command an operator (or this sweep's successor) runs to pick
    up where it stopped. Printed on every pause and every budget stop, because
    "it stopped" without the command to continue is a report that costs
    somebody an hour of reading."""
    return (f"cd /srv/share/projects/hugpy/dev && "
            f"scripts/oracle-benchmark sweep --resume {run_id}")


def _fleet_unreachable(detail: str) -> bool:
    """Is this the FLEET being down, rather than this model being unable?

    Deliberately narrow. A dispatch timeout is a model finding (a 40B model on
    a busy 3090 legitimately does not answer in 100 seconds) and a router
    refusal is a registry finding; none of the three is an outage — and a BUSY
    worker is a worker that is alive. Only transport DEATH counts, and even
    then it takes three models in a row to pause the sweep."""
    low = str(detail or "").lower()
    return any(marker in low for marker in _OUTAGE_MARKERS)


def run_stationary_sweep(*, config: StationaryConfig | None = None,
                         run_dir: str | None = None,
                         resume: str | None = None,
                         root: str | None = None,
                         retry_failed: bool = False,
                         rosters: Mapping[str, Mapping[str, Any]] | None = None,
                         note: Callable[[str], None] | None = None
                         ) -> StationarySweep:
    """The stationary sweep: ONE brief, every model, every point, sequentially.

    RESUMABLE BY CONSTRUCTION. Every cell is appended to ``cells.jsonl`` the
    moment it completes, and ``--resume <run_id>`` reloads that journal and
    skips every (point, operation, model) triple already in it. A sweep that
    runs for hours across a fleet another agent is actively repairing WILL be
    interrupted; the design assumption is that it will be, not that it might."""
    from hugpy_oracle.stationary_scenario import (
        LIFECYCLE_POINTS,
        REFERENCE_FRAMES,
        scenario_digest,
        scenario_parts,
        SCENARIO_VERSION,
    )

    config = config or StationaryConfig()
    if resume:
        directory = resume_dir(resume, root)
    else:
        directory = run_dir or new_stationary_run_dir(root, config.label)
    for sub in ("raw", "frames", "keyframes", "clips", "audio"):
        os.makedirs(os.path.join(directory, sub), exist_ok=True)
    run_id = os.path.basename(directory.rstrip("/"))
    started = _utc_now()
    monotonic_start = time.monotonic()
    log_path = os.path.join(directory, "run.log")

    def say(message: str) -> None:
        _append_line(log_path, f"{_utc_now()} {message}")
        if note is not None:
            note(message)

    journal = load_journal(directory, retry_failed=retry_failed)
    registry_version = _registry_version()
    version = SCENARIO_VERSION
    digest = scenario_digest()
    # Roster discovery reads the fleet's load state once per text model and is
    # therefore not free (91 models x two heartbeat reads). The CLI has already
    # paid for it to print the plan, so it hands the result in rather than
    # making the fleet answer the same question twice in ninety seconds.
    rosters = dict(rosters) if rosters else discover_rosters(config)

    say(f"k109b sweep {run_id}: stages={list(config.stages)} "
        f"scenario={version} ({digest}) registry={registry_version} "
        f"resuming={len(journal)} completed cell(s)")
    for kind, roster in sorted(rosters.items()):
        say(f"  roster {kind} ({roster['capability']}): "
            f"{len(roster['models'])} model(s)")

    _atomic_write(os.path.join(directory, "scenario.json"),
                  json.dumps(scenario_parts(), indent=1, sort_keys=True,
                             default=str))
    _atomic_write(os.path.join(directory, "environment.json"),
                  json.dumps({
                      "run_id": run_id, "started_at": started,
                      "wave": "k109b", "config": config.to_dict(),
                      "registry_version": registry_version,
                      "scenario_version": version, "scenario_digest": digest,
                      "rosters": rosters,
                      "stationary_params": STATIONARY_PARAMS,
                      "deterministic_weights": _WEIGHTS,
                      "vram_reserve_gib": config.vram_reserve_gib,
                      "vram_snapshot_at_start": _vram_snapshot(),
                      "host": os.uname().nodename,
                      "resumed_cells": len(journal),
                  }, indent=1, sort_keys=True, default=str))
    _atomic_write(os.path.join(directory, "points.json"),
                  json.dumps({"points": [p.to_dict() for p in LIFECYCLE_POINTS],
                              "verdicts": list(VERDICTS)},
                             indent=1, sort_keys=True, default=str))

    cells: list[Cell] = [_cell_from_row(row) for row in journal.values()]
    aborted: dict[str, str] = {}
    paused = ""
    cells_path = os.path.join(directory, CELLS_FILE)

    def record(cell: Cell, raw: str = "") -> None:
        if raw:
            name = re.sub(r"[^A-Za-z0-9_.-]+", "-",
                          f"{cell.point_id}__{cell.model}") + ".txt"
            if _atomic_write(os.path.join(directory, "raw", name),
                             raw[:_MAX_RAW_CHARS]):
                cell = replace(cell, raw_ref=os.path.join("raw", name))
        cells.append(cell)
        journal[cell.key] = cell.to_dict()
        _append_line(cells_path, json.dumps(cell.to_dict(), sort_keys=True,
                                            default=str))
        # A NO_CANDIDATES cell has no measurement, so it must not print one.
        # ``DeterministicScore().score`` is a real number (the "clean" axis is
        # vacuously 1.0 when nothing was produced) and printing it next to a
        # row that measured nothing is exactly the kind of number an operator
        # would later quote back as evidence.
        scores = ("no measurement" if cell.verdict == "NO_CANDIDATES"
                  else f"det={cell.deterministic.score} "
                       f"judge={cell.judge.score} {cell.perf.latency_s}s")
        say(f"{cell.stage or '-'} · {cell.point_id} · {cell.model}: "
            f"{cell.verdict.upper()} {scores} {cell.note[:120]}".rstrip())

    def over_budget() -> bool:
        return (config.budget_s is not None
                and (time.monotonic() - monotonic_start) > config.budget_s)

    # --- points with no candidates at all, and the model-free pipeline steps.
    # Emitted FIRST and unconditionally, because a sweep that ran out of time
    # before recording its gaps would have hidden the most durable finding it
    # had.
    for point in LIFECYCLE_POINTS:
        if point.kind not in ("gap", "pipeline"):
            continue
        key = f"{point.point_id}|{point.point_id}|(none)"
        if key in journal:
            continue
        if point.kind == "gap":
            reason = (f"NO servable model for lifecycle step {point.step} "
                      f"({point.name}). Missing: "
                      f"{', '.join(point.missing_capability)}. {point.note}")
        else:
            reason = (f"lifecycle step {point.step} ({point.name}) is executed "
                      f"by the pipeline, not by a model — there is no "
                      f"candidate to sweep and no gap to report. {point.note}")
        record(no_candidates_cell(point, reason,
                                  registry_version=registry_version,
                                  scenario_version=version,
                                  scenario_digest=digest))

    # --- a roster that is empty is ALSO a NO_CANDIDATES point, and its reason
    # is the catalog's own, verbatim.
    for point in LIFECYCLE_POINTS:
        if point.kind in ("gap", "pipeline"):
            continue
        roster = rosters.get(point.kind) or {}
        if roster.get("models"):
            continue
        for operation in point.operations:
            key = f"{point.point_id}|{operation}|(none)"
            if key in journal:
                continue
            record(replace(
                no_candidates_cell(
                    point,
                    f"no model is eligible for {point.capability!r} on this "
                    f"fleet, so lifecycle step {point.step} ({point.name}) has "
                    f"no candidate: " + "; ".join(roster.get("reasons") or
                                                  ["the catalog gave no reason"]),
                    registry_version=registry_version,
                    scenario_version=version, scenario_digest=digest),
                operation=operation, stage=point.kind))

    # ------------------------------------------------------------------
    # Stage 1 — the LLM points
    # ------------------------------------------------------------------
    llm_points = [p for p in LIFECYCLE_POINTS if p.kind == "llm"]
    if "llm" in config.stages and rosters["llm"]["models"] and not paused:
        unreachable_streak = 0
        models = list(rosters["llm"]["models"])
        say(f"stage 1 (llm): {len(models)} model(s) x "
            f"{len(llm_points)} point(s)")
        for index, model in enumerate(models, start=1):
            if paused or over_budget():
                paused = paused or (
                    f"stopped after {index - 1}/{len(models)} text model(s): "
                    f"the run budget of {config.budget_s}s was reached")
                break
            todo = [(p, op) for p in llm_points for op in p.operations
                    if f"{p.point_id}|{op}|{model}" not in journal]
            if not todo:
                say(f"[{index}/{len(models)}] {model}: all points already "
                    f"journalled — skipped")
                continue
            ok, detail, probe_latency = probe_text_model(model, config)
            say(f"[{index}/{len(models)}] {model}: probe "
                f"{'OK' if ok else 'FAILED'} ({probe_latency}s) {detail[:160]}")
            if not ok:
                if _fleet_unreachable(detail):
                    unreachable_streak += 1
                else:
                    unreachable_streak = 0
                for point, operation in todo:
                    record(Cell(
                        point_id=point.point_id, step=point.step,
                        operation=operation, model=model,
                        capability=point.capability, verdict="incapable",
                        perf=PerfRecord(model=model, mode="stationary",
                                        latency_s=probe_latency,
                                        vram_reserve_gib=config.vram_reserve_gib),
                        failure=("timeout" if "timeout" in detail.lower()
                                 else "dispatch_error"),
                        gap_code="CAPABILITY_GAP",
                        note=f"admission probe failed: {detail}"[:400],
                        stage="llm",
                        evidence={"probe": detail,
                                  "probe_latency_s": probe_latency,
                                  "probe_prompt": PROBE_PROMPT},
                        registry_version=registry_version,
                        scenario_version=version, scenario_digest=digest,
                        started_at=_utc_now(), ended_at=_utc_now()))
                if "timeout" in detail.lower():
                    _sleep(TIMEOUT_COOLDOWN_S)
                if unreachable_streak >= FLEET_DEGRADED_STREAK:
                    paused = (f"PAUSED: {unreachable_streak} model(s) in a row "
                              f"were unreachable with transient transport "
                              f"faults — the fleet is degrading and grading an "
                              f"outage would produce a matrix that says a "
                              f"healthy fleet has no capabilities. "
                              f"{resume_command(run_id)}")
                    say(paused)
                    break
                continue

            unreachable_streak = 0
            streak = 0
            for point, operation in todo:
                if over_budget():
                    paused = (f"stopped inside {model}: the run budget of "
                              f"{config.budget_s}s was reached")
                    break
                cell, raw = run_llm_cell(
                    point, operation, model, config=config,
                    registry_version=registry_version,
                    scenario_version=version, scenario_digest=digest)
                record(cell, raw)
                if cell.failure == "timeout":
                    streak += 1
                    _sleep(TIMEOUT_COOLDOWN_S)
                    if streak >= TIMEOUT_ABORT_STREAK:
                        aborted[model] = (
                            f"dropped after {streak} consecutive dispatch "
                            f"timeouts (last: {point.point_id})")
                        say(f"{model}: {aborted[model]}")
                        break
                else:
                    streak = 0
            if paused:
                break

    # ------------------------------------------------------------------
    # Stage 2 — the VLM validation point
    # ------------------------------------------------------------------
    vlm_points = [p for p in LIFECYCLE_POINTS if p.kind == "vlm"]
    frame_paths: dict[str, str] = {}
    confirmation: dict[str, Any] = {}
    needs_frames = any(s in config.stages for s in ("vlm",))
    if needs_frames and not paused:
        frame_paths, frame_records = _reference_frames(
            directory, config=config, say=say)
        confirmation = load_frame_confirmation(directory)
        say("reference-frame key: "
            + ("HUMAN-CONFIRMED from frames/human_confirmation.json — "
               + str(confirmation.get("headline") or "")[:300]
               if confirmation else
               "derived from the RENDER PROMPTS (no frames/"
               "human_confirmation.json in this run dir); a renderer that "
               "ignored a planted violation makes that key wrong for that "
               "frame, and the grounding axis is what survives it"))
        _atomic_write(os.path.join(directory, "frames", "manifest.json"),
                      json.dumps({"frames": frame_records,
                                  "renderer": config.reference_model,
                                  "expected": len(REFERENCE_FRAMES)},
                                 indent=1, sort_keys=True, default=str))
    if "vlm" in config.stages and vlm_points and not paused:
        models = list(rosters["vlm"]["models"])
        if not frame_paths:
            say("stage 2 (vlm): NO reference frame rendered — the VLM point "
                "cannot be measured without inputs, and inventing a verdict "
                "here would be worse than the gap")
            for point in vlm_points:
                for operation in point.operations:
                    key = f"{point.point_id}|{operation}|(none)"
                    if key not in journal:
                        record(replace(no_candidates_cell(
                            point,
                            f"the reference frames could not be rendered with "
                            f"{config.reference_model!r}, so no judge could be "
                            f"shown the same inputs as any other judge",
                            registry_version=registry_version,
                            scenario_version=version, scenario_digest=digest),
                            operation=operation, stage="vlm"))
        else:
            say(f"stage 2 (vlm): {len(models)} judge(s) x "
                f"{len(frame_paths)} reference frame(s)")
            for index, model in enumerate(models, start=1):
                if over_budget():
                    paused = paused or (f"stopped after {index - 1}/"
                                        f"{len(models)} VLM judge(s): budget")
                    break
                for point in vlm_points:
                    if f"{point.point_id}|{point.operations[0]}|{model}" in journal:
                        continue
                    cell, raw = run_vlm_cell(
                        point, model, frame_paths, config=config,
                        registry_version=registry_version,
                        scenario_version=version, scenario_digest=digest,
                        confirmation=confirmation)
                    record(cell, raw)

    # ------------------------------------------------------------------
    # The VLM judge every media stage borrows — resolved ONCE, from evidence
    # ------------------------------------------------------------------
    judge_model = config.vlm_judge_model or _best_vlm_judge(
        cells, rosters["vlm"]["models"])
    if "image" in config.stages or "video" in config.stages:
        say(f"media judge (VLM, never the candidate): {judge_model or 'NONE'}")

    # ------------------------------------------------------------------
    # Stage 3 — the keyframe point
    # ------------------------------------------------------------------
    image_points = [p for p in LIFECYCLE_POINTS if p.kind == "image"]
    if "image" in config.stages and image_points and not paused:
        models = list(rosters["image"]["models"])
        say(f"stage 3 (image): {len(models)} model(s), one keyframe each")
        for index, model in enumerate(models, start=1):
            if over_budget():
                paused = paused or (f"stopped after {index - 1}/{len(models)} "
                                    f"image model(s): budget")
                break
            for point in image_points:
                if f"{point.point_id}|{point.operations[0]}|{model}" in journal:
                    continue
                cell, raw = run_image_cell(
                    point, model, config=config, run_dir=directory,
                    judge_model=judge_model,
                    registry_version=registry_version,
                    scenario_version=version, scenario_digest=digest)
                record(cell, raw)

    # ------------------------------------------------------------------
    # Stage 4 — the clip render point
    # ------------------------------------------------------------------
    video_points = [p for p in LIFECYCLE_POINTS if p.kind == "video"]
    if "video" in config.stages and video_points and not paused:
        models = list(rosters["video"]["models"])
        say(f"stage 4 (video): {len(models)} clip model(s), one short low-res "
            f"clip each — the heaviest stage")
        for index, model in enumerate(models, start=1):
            if over_budget():
                paused = paused or (f"stopped after {index - 1}/{len(models)} "
                                    f"clip model(s): budget")
                break
            for point in video_points:
                if f"{point.point_id}|{point.operations[0]}|{model}" in journal:
                    continue
                cell, raw = run_video_cell(
                    point, model, config=config, run_dir=directory,
                    judge_model=judge_model,
                    registry_version=registry_version,
                    scenario_version=version, scenario_digest=digest)
                record(cell, raw)

    # ------------------------------------------------------------------
    # Stage 5 — the TTS point
    # ------------------------------------------------------------------
    tts_points = [p for p in LIFECYCLE_POINTS if p.kind == "tts"]
    if "tts" in config.stages and tts_points and not paused:
        models = list(rosters["tts"]["models"])
        say(f"stage 5 (tts): {len(models)} model(s), the same locked line each")
        for model in models:
            for point in tts_points:
                if f"{point.point_id}|{point.operations[0]}|{model}" in journal:
                    continue
                cell, raw = run_tts_cell(
                    point, model, config=config, run_dir=directory,
                    registry_version=registry_version,
                    scenario_version=version, scenario_digest=digest)
                record(cell, raw)

    sweep = StationarySweep(
        run_id=run_id, run_dir=directory, config=config,
        cells=tuple(cells), registry_version=registry_version,
        scenario_version=version, scenario_digest=digest,
        rosters={k: v["models"] for k, v in rosters.items()},
        aborted=aborted, paused=paused,
        resumed_from=len(load_journal(directory)) - len(cells)
        if resume else 0,
        started_at=started, ended_at=_utc_now())
    written = write_stationary_reports(sweep)
    _write_state(directory, {**sweep.to_dict(), "reports": written,
                             "resume_command": resume_command(run_id)})
    say(f"sweep {run_id} finished: {len(cells)} cell(s), "
        f"{sum(1 for c in cells if c.ok)} capable, elapsed "
        f"{sweep.elapsed_note}"
        + (f" | {paused}" if paused else ""))
    return sweep


def _reference_frames(directory: str, *, config: StationaryConfig,
                      say: Callable[[str], None]
                      ) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """The six reference frames, rendered once and REUSED across resumes.

    A resumed sweep must show its remaining judges the frames the earlier
    judges saw. Re-rendering would silently change the inputs half way through
    a comparison, so an existing readable file is kept and only the missing
    ones are rendered."""
    from hugpy_oracle.stationary_scenario import REFERENCE_FRAMES
    existing: dict[str, str] = {}
    for frame in REFERENCE_FRAMES:
        path = os.path.join(directory, "frames", f"{frame.frame_id}.png")
        if os.path.isfile(path):
            substance, _detail = image_carries_content(path)
            if substance:
                existing[frame.frame_id] = path
    if len(existing) == len(REFERENCE_FRAMES):
        say(f"reference frames: all {len(existing)} already rendered in this "
            f"run dir — reused, NOT re-rendered (a resumed sweep must judge "
            f"the same pixels)")
        return existing, [{"frame_id": k, "ok": True, "path": v,
                           "reused": True} for k, v in sorted(existing.items())]
    say(f"reference frames: rendering with {config.reference_model!r} "
        f"({len(existing)} already present)")
    fresh, records = render_reference_frames(directory, config=config, note=say)
    fresh.update(existing)
    return fresh, records


def _best_vlm_judge(cells: Sequence[Cell], roster: Sequence[str]) -> str | None:
    """The VLM the media stages borrow as their judge — chosen from EVIDENCE.

    Preference order: the highest-scoring judge stage 2 actually measured, then
    the first roster entry, then None. Choosing from stage 2's own results is
    the point of running stage 2 first: the model that graded six known frames
    best is the model best qualified to grade an unknown one, and picking
    alphabetically would throw that measurement away."""
    scored = [(c.deterministic.score, c.model) for c in cells
              if c.stage == "vlm" and c.verdict in ("capable", "partial")]
    if scored:
        return sorted(scored, reverse=True)[0][1]
    return roster[0] if roster else None


def _cell_from_row(row: Mapping[str, Any]) -> Cell:
    """Rehydrate a journalled cell. Lossy ON PURPOSE for the score internals:
    the per-check evidence is kept as data in ``evidence`` and the axes are
    restored, which is everything the reports and the matrix read. A resumed
    run's reports are therefore derived from the same numbers the original run
    wrote, not from a re-scoring that could disagree with the journal."""
    det = row.get("deterministic") or {}
    judge = row.get("judge") or {}
    perf = row.get("perf") or {}
    return Cell(
        point_id=str(row.get("point_id") or row.get("case_id") or ""),
        step=int(row.get("step") or 0),
        operation=str(row.get("operation") or ""),
        model=str(row.get("model") or ""),
        capability=str(row.get("capability") or ""),
        verdict=str(row.get("verdict") or "incapable"),
        deterministic=DeterministicScore(
            valid=bool(det.get("valid")),
            error_count=int(det.get("error_count") or 0),
            preservation=det.get("preservation"),
            contradiction_rate=float(det.get("contradiction_rate") or 0.0),
            completeness=det.get("completeness"),
            constraint_adherence=det.get("constraint_adherence"),
            accuracy=det.get("accuracy"),
            errors=tuple(det.get("errors") or ())),
        judge=JudgeScore(
            judge_model=judge.get("judge_model"),
            verdict=str(judge.get("verdict") or "unavailable"),
            score=judge.get("score"), why=str(judge.get("why") or ""),
            available=bool(judge.get("available")),
            refused=bool(judge.get("refused")),
            detail=str(judge.get("detail") or "")),
        perf=PerfRecord(
            model=perf.get("model"), worker=perf.get("worker"),
            load_state=perf.get("load_state"),
            mode=str(perf.get("mode") or "stationary"),
            params=dict(perf.get("params") or {}),
            latency_s=perf.get("latency_s"),
            dispatch_calls=int(perf.get("dispatch_calls") or 0),
            tokens_per_s=perf.get("tokens_per_s"),
            output_chars=int(perf.get("output_chars") or 0),
            vram_before=perf.get("vram_before"),
            vram_after=perf.get("vram_after"),
            vram_used_delta_bytes=perf.get("vram_used_delta_bytes"),
            vram_peak_bytes=perf.get("vram_peak_bytes"),
            vram_sample_count=int(perf.get("vram_sample_count") or 0),
            vram_sampler=perf.get("vram_sampler"),
            gpu_total_bytes=perf.get("gpu_total_bytes"),
            vram_reserve_gib=perf.get("vram_reserve_gib")),
        failure=row.get("failure"), gap_code=row.get("gap_code"),
        note=str(row.get("note") or ""),
        evidence=dict(row.get("evidence") or {}),
        raw_ref=str(row.get("raw_ref") or ""),
        artifact_ref=str(row.get("artifact_ref") or ""),
        registry_version=row.get("registry_version"),
        scenario_version=str(row.get("scenario_version") or ""),
        scenario_digest=str(row.get("scenario_digest") or ""),
        stage=str(row.get("stage") or row.get("track") or ""),
        started_at=str(row.get("started_at") or ""),
        ended_at=str(row.get("ended_at") or ""))


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def write_stationary_reports(sweep: StationarySweep) -> dict[str, str]:
    """Serialize the sweep: scores, routing matrix, leaderboards, grid.

    Returns the paths actually written. A path missing from the mapping failed
    to write and said so in the log — the run itself is already durable in
    ``cells.jsonl``, which is the file this whole design exists to protect."""
    from hugpy_oracle import routing_matrix as rm
    from hugpy_oracle.stationary_scenario import LIFECYCLE_POINTS, part_digests

    written: dict[str, str] = {}
    rows = sweep.rows
    measured = [r for r in rows if r.get("verdict") != "NO_CANDIDATES"]

    summary = {
        "run": sweep.to_dict(),
        "formula": rm.FORMULA_NOTE,
        "deterministic_weights": _WEIGHTS,
        "scenario_part_digests": part_digests(),
        "per_model_operation": rm.summarize(measured),
        "per_point": rm.summarize_points(rows),
    }
    path = os.path.join(sweep.run_dir, "scores.json")
    if _atomic_write(path, json.dumps(summary, indent=1, sort_keys=True,
                                      default=str)):
        written["scores"] = path

    matrix = rm.derive_matrix(measured, registry_version=sweep.registry_version,
                              run_id=sweep.run_id, run_dir=sweep.run_dir,
                              mode="stationary",
                              scenario_version=sweep.scenario_version,
                              scenario_digest=sweep.scenario_digest)
    path = os.path.join(sweep.run_dir, "routing_matrix.json")
    if _atomic_write(path, json.dumps(matrix.to_dict(), indent=1,
                                      sort_keys=True, default=str)):
        written["matrix"] = path

    path = os.path.join(sweep.run_dir, "leaderboard.md")
    if _atomic_write(path, rm.render_leaderboard(matrix, measured)):
        written["leaderboard"] = path

    path = os.path.join(sweep.run_dir, "capability_grid.md")
    if _atomic_write(path, rm.render_capability_grid(
            rows, [p.to_dict() for p in LIFECYCLE_POINTS],
            title=f"k109b capability grid — {sweep.run_id}",
            scenario_version=sweep.scenario_version,
            scenario_digest=sweep.scenario_digest,
            registry_version=sweep.registry_version)):
        written["capability_grid"] = path

    path = os.path.join(sweep.run_dir, "points.md")
    if _atomic_write(path, rm.render_point_leaderboards(
            rows, [p.to_dict() for p in LIFECYCLE_POINTS],
            title=f"k109b per-point leaderboards — {sweep.run_id}",
            scenario_version=sweep.scenario_version)):
        written["points"] = path
    return written

__all__ = [
    "Attempt", "BENCH_CAPABILITY", "BenchCheck", "BenchmarkRun",
    "CEILING_PARAMS", "DEFAULT_ATTEMPT_DEADLINE_S", "DEFAULT_RUN_ROOT",
    "DEFAULT_VRAM_RESERVE_GIB", "DeterministicScore", "JUDGE_RUBRICS",
    "JudgeScore", "MODES", "NORMALIZED_PARAMS", "PerfRecord", "RunConfig",
    "TIMEOUT_ABORT_STREAK", "author_completion", "author_workflow",
    "build_completion_prompt", "build_judge_prompt", "build_prompt",
    "build_workflow_prompt", "check_constraint", "default_run_root",
    "discover_models", "judge_attempt", "mode_params", "new_run_dir",
    "pick_judge", "produce", "run_case", "run_sweep", "score_case",
    "score_plot", "score_screenplay", "score_workflow", "workflow_errors",
    "write_reports",
    # --- k109b: the stationary-prompt full-fleet sweep ---
    "ASR_TASK", "BLANK_IMAGE_STDEV_FLOOR", "CELLS_FILE",
    "DEFAULT_PROBE_DEADLINE_S", "FLEET_DEGRADED_STREAK", "IMAGE_TASK",
    "PROBE_PROMPT", "REFUSAL_MARKERS", "STATE_FILE", "STATIONARY_PARAMS",
    "STATIONARY_STAGES", "TIMEOUT_COOLDOWN_S", "TTS_TASK", "VERDICTS",
    "VLM_TASK", "WORKFLOW_ACCURACY_ALIAS", "Cell", "StationaryConfig",
    "StationarySweep", "ask_vlm", "audio_carries_sound", "classify_verdict",
    "discover_rosters", "extract_middle_frame", "generate_image",
    "grounding_score", "image_carries_content", "image_levels",
    "load_frame_confirmation", "CONFIRMATION_FILE",
    "is_transient", "load_journal", "looks_like_refusal",
    "new_stationary_run_dir", "no_candidates_cell", "order_by_residency",
    "probe_media",
    "probe_text_model", "render_reference_frames", "resume_command",
    "resume_dir", "run_image_cell", "run_llm_cell", "run_stationary_sweep",
    "run_tts_cell", "run_video_cell", "run_vlm_cell", "transcribe",
    "wav_levels", "write_stationary_reports",
]


# --------------------------------------------------------------------------- #
# CLI (TODO-10): `python -m hugpy_oracle.benchmark` /
# `scripts/oracle-benchmark`. Runs the sweep, writes the reports (summary,
# leaderboard, routing_matrix.json), prints the paths. Without this the module
# only imported and printed registry warnings — a "sweep" that swept nothing.
# --------------------------------------------------------------------------- #


def _cli(argv: Sequence[str] | None = None) -> int:
    import argparse
    import sys as _sys

    ap = argparse.ArgumentParser(
        prog="oracle-benchmark",
        description="Qualification sweep: Tracks A–C over the fleet's eligible models; "
                    "writes summary + leaderboard + routing_matrix.json under the battery root.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--normalized", action="store_true", help="equal context/workload/sampling (default)")
    mode.add_argument("--ceiling", action="store_true", help="each model at its highest viable config, all safely usable VRAM")
    ap.add_argument("--tracks", default="ABC", help="subset of ABC (default ABC)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit-per-track", type=int, default=None)
    ap.add_argument("--max-models", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None, help="explicit model keys (default: discover eligible)")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--deadline-s", type=float, default=DEFAULT_ATTEMPT_DEADLINE_S)
    ap.add_argument("--vram-reserve-gib", type=float, default=DEFAULT_VRAM_RESERVE_GIB)
    ap.add_argument("--sample-ms", type=int, default=DEFAULT_VRAM_SAMPLE_MS,
                    help="peak-VRAM sampler cadence in ms (pynvml, else central /llm/vram); 0 disables")
    ap.add_argument("--label", default="")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    config = RunConfig(mode="ceiling" if a.ceiling else "normalized", repeats=a.repeats, tracks=a.tracks,
                       limit_per_track=a.limit_per_track, deadline_s=a.deadline_s, judge=not a.no_judge,
                       judge_model=a.judge_model, vram_reserve_gib=a.vram_reserve_gib,
                       vram_sample_ms=max(0, a.sample_ms), max_models=a.max_models, label=a.label or ("ceiling" if a.ceiling else "normalized"))
    note = None if a.quiet else (lambda m: print(m, file=_sys.stderr, flush=True))
    print(f"oracle-benchmark: mode={config.mode} tracks={config.tracks} repeats={config.repeats} "
          f"models={'discover' if a.models is None else a.models}", file=_sys.stderr, flush=True)
    run = run_sweep(a.models, config=config, run_dir=a.run_dir, note=note)
    paths = write_reports(run)
    attempts = len(getattr(run, "attempts", ()) or ())
    print(json.dumps({"run_id": run.run_id, "run_dir": run.run_dir, "mode": config.mode,
                      "registry_version": run.registry_version, "attempts": attempts,
                      "aborted": list(getattr(run, "aborted", ()) or ()), "reports": paths}, indent=2))
    return 0 if attempts else 3


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_cli())
