"""k114 — the script-first generation RUN: persistence, lifecycle, dispatch.

This module is the SHELL around artifacts that already exist. It invents no
artifact logic: ``GenerationSnapshot`` / ``RunPromptLedger`` / ``ProductionLock``
are k104's, ``PlotSpec`` / ``Screenplay`` / ``build_continuity`` /
``build_shot_plan`` / ``author_plot`` / ``author_screenplay`` / ``bind_llm`` /
``lock_production`` are k110's, ``compile_segments`` / ``to_plan_graph`` are
k104's, and the run journal is the ``runs/<kind>/<run_id>/state.json``
convention k106 established in ``performance.py``. What k114 adds is the three
things a UI needs and a pure contract cannot have: a run that SURVIVES the
request that made it, an HTTP-shaped refusal for every way the pipeline says
no, and a per-attempt receipt trail per segment.

THE INVARIANT THIS FILE EXISTS TO KEEP (doc "Non-negotiable generation
semantics"): a prompt minted during a run may never become the input to a
sibling segment of that same run. Three mechanisms, all structural:

1. The snapshot is built ONCE at ``create()`` from prompts that carry a
   ``persisted_at`` at or before the run's start. A source persisted after the
   start is recorded in ``sources`` with ``included=False`` and a reason — it
   is VISIBLE and EXCLUDED, never silently dropped.
2. Every prompt the compiler mints goes into the run's ``RunPromptLedger``
   (the seam k104 and k110 both left to the caller), and ``compile()`` runs
   ``snapshot.assert_pre_run(ledger)`` afterwards, where it can actually catch
   a snapshot that grew mid-run.
3. ``promote()`` records the promoted text's digest in THIS run's ledger before
   it writes the persisted source. After that, feeding it back into this run is
   refused by digest inside k104's own code — not by a policy check in a route
   that a later edit could forget. A NEW run has a fresh ledger and takes it.

Nothing here imports ``video_intel`` at module level (k104/k110's discipline):
building the model registry is a two-second import and this module has to stay
importable from a test with no fleet at all. Every live seam
(``bind_llm``, the catalog read, the image dispatch) is a DEFAULT ARGUMENT, so
a test passes a function and touches no GPU, no worker and no network.

No pathlib; os.path only (project discipline).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from hugpy_oracle.audio_master import AudioMaster
from hugpy_oracle.production import (
    ContinuityBible,
    GenerationSnapshot,
    LockRefused,
    ProductionError,
    ProductionLock,
    RunPromptLedger,
    RunPromptRefused,
    ShotPlan,
    prompt_digest,
)
from hugpy_oracle.screenplay import (
    AuthoringGap,
    PlotSpec,
    Screenplay,
    ScreenplayError,
    ShotPlanDraft,
    author_plot,
    author_screenplay,
    bind_llm,
    build_continuity,
    build_shot_plan,
    lock_production,
    plot_input_mode,
)
from hugpy_oracle.segments import (
    CompileRefused,
    SegmentSpec,
    SiblingViolation,
    compile_segments,
    execution_order,
    to_plan_graph,
)
from hugpy_oracle import routing_matrix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Bumped when the journal shape changes incompatibly. A state file at another
#: version is not read — an unreadable run is a correct answer, a
#: misinterpreted one is not.
STATE_VERSION: int = 1

STATE_FILENAME: str = "state.json"

#: Where runs live under the run root. ``performance.py`` uses
#: ``runs/performance/<id>``; this is its sibling, deliberately the same shape
#: so an operator finds both in one directory listing.
RUN_KIND: str = "script_first"

RUN_ROOT_ENV: str = "HUGPY_SCRIPT_FIRST_RUN_ROOT"

#: Promoted sources live OUTSIDE any run directory, because their whole purpose
#: is to be available to a run that does not exist yet.
SOURCE_DIRNAME: str = "_promoted_sources"

#: The four authored/derived artifacts a run persists, in production order.
ARTIFACT_STAGES: tuple[str, ...] = ("plot", "screenplay", "continuity",
                                    "shot_plan", "audio_master")

#: Which stages an operator may author with a model. The other three are
#: DERIVED (k110: asking a model to restate what the screenplay already says is
#: asking it to disagree with the script).
AUTHORED_STAGES: tuple[str, ...] = ("plot", "screenplay")

#: k114's follow-up, landed: which k109 routing-matrix OPERATION each authored
#: stage resolves a model through. 1:1 with ``AUTHORED_STAGES`` today; a stage
#: with no entry here falls straight to the catalog default (no matrix
#: consulted), which is also what happens for any stage this map is not kept
#: in sync with.
AUTHORING_OPERATIONS: dict[str, str] = {
    "plot": "plot.construct",
    "screenplay": "screenplay.complete",
}

#: Every refusal this module can produce, as a closed vocabulary. A route maps
#: these onto HTTP status codes; a UI maps them onto operator actions. A code
#: not in this tuple cannot be constructed.
REFUSAL_CODES: tuple[str, ...] = (
    "RUN_NOT_FOUND",
    "SOURCE_INVALID",
    "SOURCE_DIGEST_MISMATCH",
    "STAGE_UNKNOWN",
    "ARTIFACT_INVALID",
    "ARTIFACT_MISSING",
    "AUTHORING_GAP",
    "AUDIO_MASTER_MISSING",
    "LOCK_REFUSED",
    "NOT_LOCKED",
    "ALREADY_LOCKED",
    "REVISION_REASON_MISSING",
    "COMPILE_REFUSED",
    "SEGMENTS_MISSING",
    "SEGMENT_UNKNOWN",
    "GENERATE_GAP",
    "PROMOTE_REFUSED",
    "RUN_WRITE_FAILED",
)

#: What each refusal means over HTTP. 422 = "you sent something the artifact
#: constructors reject"; 409 = "the run is not in a state where this is a
#: sensible request"; 404 = "no such thing".
REFUSAL_STATUS: dict[str, int] = {
    "RUN_NOT_FOUND": 404,
    "SOURCE_INVALID": 422,
    "SOURCE_DIGEST_MISMATCH": 422,
    "STAGE_UNKNOWN": 404,
    "ARTIFACT_INVALID": 422,
    "ARTIFACT_MISSING": 409,
    "AUTHORING_GAP": 422,
    "AUDIO_MASTER_MISSING": 409,
    "LOCK_REFUSED": 409,
    "NOT_LOCKED": 409,
    "ALREADY_LOCKED": 409,
    "REVISION_REASON_MISSING": 422,
    "COMPILE_REFUSED": 409,
    "SEGMENTS_MISSING": 409,
    "SEGMENT_UNKNOWN": 404,
    "GENERATE_GAP": 200,      # an attempt that could not run is still an attempt
    "PROMOTE_REFUSED": 409,
    "RUN_WRITE_FAILED": 500,
}

#: The capability each dispatch kind routes as.
KEYFRAME_CAPABILITY: str = "image.generate"
CLIP_CAPABILITY: str = "video.generate.i2v"

DISPATCH_KINDS: tuple[str, ...] = ("keyframe", "clip")

#: The operator step behind each seam this fleet cannot serve today. Copied in
#: WORDING, not imported, from k106's ``LIVE_SEAM_GAPS`` — ``performance.py`` is
#: another agent's file and a cross-import for four strings would couple two
#: modules that have no other reason to know about each other.
SEAM_REQUIREMENTS: dict[str, str] = {
    "audio.tts": (
        "seat chatterbox on a GPU worker: `pip install chatterbox-tts` into "
        "that worker's hugpy venv, `python -c \"import chatterbox\"` must "
        "succeed there, and the worker must HEARTBEAT "
        "task_capabilities[\"text-to-speech\"]. Verify with GET "
        "/oracle/capabilities: audio.tts must come back ELIGIBLE. Until then "
        "there is no measured dialogue timing, so there is no AudioMaster and "
        "Stage 8 has nothing to lock shot timing against."),
    CLIP_CAPABILITY: (
        "clips execute through the studio job pipeline on a GPU worker; "
        "central has no GPU, so video.* resolves 'deferred' by design (k90b). "
        "Bind the segment dispatch to the studio produce_clip spine "
        "(video_intel/runners/studio_i2v.py) using spec.seed_base, "
        "spec.prompt, spec.identity_refs, spec.joint_mode and the locked "
        "window spec.audio_window."),
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class ScriptFirstError(Exception):
    """Base for every k114 refusal."""


class ScriptFirstRefused(ScriptFirstError):
    """A typed refusal carrying EVERY problem, not just the first.

    ``errors`` is a tuple for the same reason k110's ``ScreenplayError.errors``
    is: an operator editing JSON in a textarea needs the whole list to fix it
    in one pass, and a validator that reports one problem per round trip turns
    a two-minute edit into a twenty-minute ladder.

    ``detail`` carries the machine-readable body (an ``AuthoringGap``'s dict, a
    ``ValidationReport``, the seam requirement) — never a prose summary of
    something the caller could have read structurally."""

    def __init__(self, code: str, message: str, *,
                 errors: Sequence[str] = (),
                 detail: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        if code not in REFUSAL_CODES:
            raise ValueError(f"refusal code {code!r} is not one of "
                             f"{list(REFUSAL_CODES)}")
        self.code: str = code
        self.message: str = str(message)
        self.errors: tuple[str, ...] = tuple(str(e) for e in errors) or (str(message),)
        self.detail: dict[str, Any] = dict(detail or {})

    @property
    def http_status(self) -> int:
        return REFUSAL_STATUS[self.code]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": False, "code": self.code, "message": self.message,
                "errors": list(self.errors), "detail": self.detail,
                # `error` is an ALIAS, and it is load-bearing rather than
                # decorative: the console's shared transport
                # (transport/client.ts) parses a non-2xx body and keeps only
                # `parsed.error` as the message a component can render. Putting
                # the code and EVERY error string here is what makes "show the
                # validator errors verbatim" true in the browser without
                # editing a transport every station shares.
                "error": "\n".join((f"{self.code}: {self.message}",)
                                    + tuple(self.errors))}


class RunNotFound(ScriptFirstRefused):
    def __init__(self, run_id: str) -> None:
        super().__init__("RUN_NOT_FOUND",
                         f"no script-first run {run_id!r} on this box",
                         detail={"run_id": str(run_id)})


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    out: list[str] = []
    for item in value:
        text = _text(item)
        if text and text not in out:
            out.append(text)
    return out


def _jsonable(value: Any) -> Any:
    """Anything this module persists, reduced to JSON. Deliberately total: a
    journal write that raises because one field was a tuple would lose the run
    it was protecting."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    for attr in ("to_dict",):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return _jsonable(fn())
            except Exception:      # noqa: BLE001 - a journal is best effort
                pass
    return str(value)


def default_run_root() -> str:
    """Where script-first runs write.

    The same derivation ``performance.default_run_root`` uses, so both kinds of
    run land under one ``<DEFAULT_ROOT>/video_intel/runs/`` directory and an
    operator does not have to know which subsystem wrote which. Copied rather
    than imported: ``performance.py`` belongs to another task and this is four
    lines, not a shared abstraction."""
    from hugpy_oracle.config import script_first_run_root
    return script_first_run_root()


def run_dir(run_id: str, root: str | None = None) -> str:
    return os.path.join(root or default_run_root(), "runs", RUN_KIND,
                        str(run_id))


def state_path(run_id: str, root: str | None = None) -> str:
    return os.path.join(run_dir(run_id, root), STATE_FILENAME)


def sources_dir(root: str | None = None) -> str:
    return os.path.join(root or default_run_root(), "runs", RUN_KIND,
                        SOURCE_DIRNAME)


def new_run_id() -> str:
    """A run id that is unique per CREATION, not per request content.

    Deliberately NOT ``performance.derive_run_id``'s content hash: two
    identical requests there share a run dir so a resume can find one, but here
    two identical requests are two different productions with two different
    snapshots, and silently merging them would make one operator's lock appear
    inside another's run."""
    return f"sf-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:10]}"


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=directory, prefix=".sf-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# The live seams — every one of them a default argument, never a hard call
# ---------------------------------------------------------------------------


def live_fleet_view() -> dict[str, Any]:
    """The doc's "available model and hardware information", read once at run
    creation and FROZEN into the run.

    Compact on purpose: the full ``GET /oracle/capabilities`` body is ~80 KB
    and a run journal that carries it becomes unreadable. What a run needs to
    prove later is which capabilities were eligible, on which models, and under
    which registry version — which is exactly what is kept."""
    out: dict[str, Any] = {"registry_version": None, "capabilities": [],
                           "hardware": {}, "read_at": _utc_now(),
                           "errors": []}
    try:
        from hugpy_oracle import catalog
    except Exception as exc:                       # noqa: BLE001
        out["errors"].append(f"catalog unimportable: {type(exc).__name__}: {exc}")
        return out
    try:
        out["registry_version"] = catalog.registry_version()
    except Exception as exc:                       # noqa: BLE001
        out["errors"].append(f"registry_version unreadable: "
                             f"{type(exc).__name__}: {exc}")
    try:
        for view in catalog.list_capabilities():
            eligibility = getattr(view, "eligibility", None)
            out["capabilities"].append({
                "name": getattr(view, "name", ""),
                "eligible": bool(getattr(eligibility, "eligible", False)),
                "model_ids": list(getattr(view, "model_ids", ()) or ()),
                "reasons": list(getattr(eligibility, "reasons", ()) or ()),
                "vram_gib": getattr(getattr(view, "resources", None),
                                    "vram_gib", None),
            })
    except Exception as exc:                       # noqa: BLE001
        out["errors"].append(f"list_capabilities failed: "
                             f"{type(exc).__name__}: {exc}")
    out["hardware"] = _live_hardware()
    return out


def _live_hardware() -> dict[str, Any]:
    """Whatever this fleet will honestly tell us about its own GPUs.

    Read off the SAME worker roster the catalog's eligibility gate reads
    (``catalog._online_workers``) rather than a second source: a run that
    records different hardware than the one the routing decision was made
    against is recording a fiction. Only the hardware fields are kept — the
    roster also carries every model id on every box, which would make a run
    journal unreadable and tells the operator nothing the catalog view does
    not already say.

    Never raises and never guesses: a roster read that fails records the
    failure, because an empty dict here would read as "no GPUs on this fleet",
    which is a different and much more alarming claim."""
    try:
        from hugpy_oracle import catalog
        rows = catalog._online_workers()      # noqa: SLF001 - the oracle's own
    except Exception as exc:                       # noqa: BLE001
        return {"workers": None,
                "note": f"the worker roster is unreadable here "
                        f"({type(exc).__name__}: {exc}); the fleet's own "
                        f"heartbeats remain the authority"}
    if rows is None:
        return {"workers": None,
                "note": "no worker roster is available in this process"}
    out: list[dict[str, Any]] = []
    for row in rows:
        gpus = []
        for gpu in (row.get("gpus") or []):
            gpus.append({"index": gpu.get("index"), "name": gpu.get("name"),
                         "memory_total": gpu.get("memory_total"),
                         "memory_free": gpu.get("memory_free")})
        out.append({"name": row.get("name"), "role": row.get("role"),
                    "gpus": gpus,
                    "task_capabilities": sorted(
                        (row.get("task_capabilities") or {}).keys())
                    if isinstance(row.get("task_capabilities"), Mapping)
                    else list(row.get("task_capabilities") or ())})
    return {"workers": out}


def live_authoring_route(capability: str = "text.chat") -> dict[str, Any]:
    """The selected model route for authoring, WITH its justification.

    The doc's interface requirement is "show the selected model route and
    benchmark justification". k109's benchmark has not landed, so what is shown
    is the router's OWN rationale (``model_rationale`` — requested /
    only-eligible / default) and the catalog's reasons. That is the honest
    justification available today, and the field it lands in is the one k109
    will fill."""
    try:
        from hugpy_oracle.contracts import GoalSpec
        from hugpy_oracle.router import resolve_route
    except Exception as exc:                       # noqa: BLE001
        return {"capability": capability, "execution": "gap",
                "reasons": [f"the oracle router is not importable here: "
                            f"{type(exc).__name__}: {exc}"]}
    try:
        goal = GoalSpec(objective="author a screenplay artifact",
                        raw_prompt="(probe)", capability=capability)
        return _jsonable(resolve_route(goal).to_dict())
    except Exception as exc:                       # noqa: BLE001
        return {"capability": capability, "execution": "gap",
                "reasons": [f"{type(exc).__name__}: {exc}"]}


def resolve_authoring_model(stage: str, *,
                            matrix_loader: Callable[[], tuple[Any, str]]
                            | None = None) -> dict[str, Any]:
    """k114's follow-up, landed: route ``stage``'s authoring call at k109's
    routing-matrix winner instead of the catalog's un-measured default —
    "the single highest-value change to this pipeline", per k114's own
    follow-up note.

    Returns a JSON-safe record, not just a model id, because the WHY matters
    as much as the WHAT: an operator reading a run needs to tell a measured
    route apart from today's default. ``requested_model`` is ``None`` — the
    catalog default, unchanged from before this landed — whenever: the stage
    has no k109 operation mapped, no matrix is available (absent, OR its
    ``registry_version`` is stale against the live catalog — never a stale
    route, see ``routing_matrix.load_latest_matrix``), or the matrix never
    benchmarked this operation (an empty candidate list is a finding, not a
    route to hide). ``matrix_loader`` is the injection seam: a callable
    returning ``(RoutingMatrix | None, str)``, defaulting to
    ``routing_matrix.load_latest_matrix`` — a test supplies a fixed matrix
    with no filesystem and no catalog touched."""
    operation = AUTHORING_OPERATIONS.get(stage)
    if operation is None:
        return {"stage": stage, "operation": None, "source": "catalog-default",
                "requested_model": None,
                "reason": f"stage {stage!r} has no k109 operation mapped; "
                          f"using the catalog's default text.chat route"}
    load = matrix_loader or routing_matrix.load_latest_matrix
    try:
        matrix, matrix_reason = load()
    except Exception as exc:                       # noqa: BLE001
        return {"stage": stage, "operation": operation,
                "source": "catalog-default", "requested_model": None,
                "reason": f"the k109 matrix could not be loaded "
                          f"({type(exc).__name__}: {exc}); falling back to "
                          f"the catalog's default route"}
    if matrix is None:
        return {"stage": stage, "operation": operation,
                "source": "catalog-default", "requested_model": None,
                "reason": f"no k109 routing matrix available ({matrix_reason}); "
                          f"falling back to the catalog's default route"}
    choice = routing_matrix.best_route(operation, matrix=matrix)
    if choice is None or not choice.primary:
        why = choice.note if choice else f"{operation!r} was never benchmarked"
        return {"stage": stage, "operation": operation,
                "source": "catalog-default", "requested_model": None,
                "reason": f"the k109 matrix has no route for {operation} "
                          f"({why}); falling back to the catalog's default "
                          f"route"}
    return {"stage": stage, "operation": operation, "source": "k109-matrix",
            "requested_model": choice.primary,
            "reason": f"k109 routing matrix primary for {operation}: "
                      f"{choice.primary} (run {choice.run_id or 'unknown'}, "
                      f"registry_version {choice.registry_version or 'unrecorded'}"
                      f"): {choice.note}"}


def live_catalog_view() -> dict[str, Any]:
    """``{name: CapabilityView}`` for the static validator. Empty on failure —
    which the validator reports as UNKNOWN_CAPABILITY, a TRUE finding about a
    fleet whose catalog will not read."""
    try:
        from hugpy_oracle import catalog
        return {v.name: v for v in catalog.list_capabilities()}
    except Exception as exc:                       # noqa: BLE001
        logger.warning("script_first: catalog view unreadable (%s: %s)",
                       type(exc).__name__, exc)
        return {}


def live_segment_dispatch(spec: SegmentSpec, *, kind: str, seed: int,
                          settings: Mapping[str, Any] | None = None,
                          ) -> dict[str, Any]:
    """ONE segment attempt through the fleet's existing dispatch.

    ``keyframe`` goes through ``image.generate`` on the SAME two functions the
    /ml routes use (``router.resolve_route`` + ``runtime.execute_route``) — no
    new inference machinery, k90b's rule. ``clip`` routes ``video.generate.i2v``
    and, on central, comes back ``deferred`` by design: clips execute through
    the studio job pipeline on a GPU worker. That is returned as a TYPED GAP
    carrying the operator step, recorded as attempt N like any other, because
    "we could not try" is an outcome an operator has to be able to read — not
    a silence and not a fake failure.

    Returns the attempt body: ``{ok, kind, capability, model_id, seed, params,
    prompt, artifacts, receipt, gap}``. Never raises for a dispatch problem;
    that is receipt data (invariant 12)."""
    if kind not in DISPATCH_KINDS:
        raise ValueError(f"dispatch kind {kind!r} is not one of "
                         f"{list(DISPATCH_KINDS)}")
    capability = KEYFRAME_CAPABILITY if kind == "keyframe" else CLIP_CAPABILITY
    body: dict[str, Any] = {
        "ok": False, "kind": kind, "capability": capability,
        "model_id": None, "seed": int(seed), "prompt": spec.prompt,
        "negative_prompt": spec.negative_prompt,
        "params": {"seed": int(seed)}, "artifacts": [], "receipt": None,
        "route": None, "gap": None,
    }
    try:
        from hugpy_oracle.contracts import GoalSpec
        from hugpy_oracle.router import resolve_route
        from hugpy_oracle import runtime
    except Exception as exc:                       # noqa: BLE001
        body["gap"] = {"code": "CAPABILITY_GAP", "capability": capability,
                       "reasons": [f"the oracle runtime is not importable "
                                   f"here: {type(exc).__name__}: {exc}"],
                       "requirement": SEAM_REQUIREMENTS.get(capability, "")}
        return body

    text = spec.prompt
    if spec.identity_refs:
        # The refs reach the model as TEXT, not as conditioning — this fleet
        # registers no identity pack for image.generate. Recorded on the
        # attempt (see the caller's limitations) because a named ref that only
        # reaches the prompt is a description of a person, not a lock on one.
        text = f"{spec.prompt}\n[identity: {', '.join(spec.identity_refs)}]"

    overrides: dict[str, Any] = {"seed": int(seed)}
    for key in ("width", "height", "steps", "guidance_scale", "sampler",
                "negative_prompt", "num_frames", "fps"):
        value = (settings or {}).get(key)
        if value is not None:
            overrides[key] = value
    if spec.negative_prompt and "negative_prompt" not in overrides:
        overrides["negative_prompt"] = spec.negative_prompt
    body["params"] = _jsonable(overrides)

    goal = GoalSpec(objective=f"render segment {spec.segment_id}",
                    raw_prompt=text, capability=capability)
    try:
        # k113a: the per-call selector proposes the model (matrix + reliability
        # ledger + VRAM/quality/latency); the router keeps authority + catalog.
        from hugpy_oracle import selection as _selection
        requested, decision = _selection.requested_model_for(goal, capability)
        try:
            route = resolve_route(goal, requested)
        except Exception:  # noqa: BLE001 — selector/catalog disagreement: catalog wins
            route = resolve_route(goal)
        if decision is not None:
            body["selection"] = decision
    except Exception as exc:                       # noqa: BLE001
        body["gap"] = {"code": "ROUTE_ERROR", "capability": capability,
                       "reasons": [f"{type(exc).__name__}: {exc}"],
                       "requirement": SEAM_REQUIREMENTS.get(capability, "")}
        return body
    body["route"] = _jsonable(route.to_dict())
    body["model_id"] = route.model_id
    if route.execution != "execute":
        body["gap"] = {"code": "CAPABILITY_GAP", "capability": capability,
                       "execution": route.execution,
                       "reasons": list(route.reasons),
                       "requirement": SEAM_REQUIREMENTS.get(capability, "")}
        return body

    artifacts, receipt = runtime.execute_route(goal, route, overrides=overrides)
    body["receipt"] = _jsonable(receipt.to_dict())
    body["model_id"] = receipt.model_id or route.model_id
    body["artifacts"] = _jsonable(artifacts)
    if receipt.failure is not None:
        body["gap"] = {"code": "DISPATCH_FAILED", "capability": capability,
                       "failure": getattr(receipt.failure, "value",
                                          str(receipt.failure)),
                       "reasons": list(receipt.log_excerpt),
                       "requirement": SEAM_REQUIREMENTS.get(capability, "")}
        return body
    body["ok"] = True
    return body


# ---------------------------------------------------------------------------
# Promoted sources — the ONLY way an accepted output crosses into a new run
# ---------------------------------------------------------------------------


def promoted_source_path(source_id: str, root: str | None = None) -> str:
    return os.path.join(sources_dir(root), f"{source_id}.json")


def list_promoted_sources(root: str | None = None) -> list[dict[str, Any]]:
    directory = sources_dir(root)
    out: list[dict[str, Any]] = []
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return out
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name), "r", encoding="utf-8") as fh:
                out.append(json.load(fh))
        except (OSError, ValueError) as exc:
            logger.debug("script_first: unreadable promoted source %s (%s)",
                         name, exc)
    return out


def load_promoted_source(source_id: str,
                         root: str | None = None) -> dict[str, Any] | None:
    try:
        with open(promoted_source_path(source_id, root), "r",
                  encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


class ScriptFirstRun:
    """One script-first generation run, persisted to its own directory.

    The state dict IS the wire shape — ``GET /video/script/runs/<id>`` returns
    it verbatim. That is deliberate: a second projection layer between the
    journal and the UI is a second place for the lock status and the artifact
    digests to disagree, and those two facts are the entire point of the
    screen."""

    __slots__ = ("run_id", "root", "_state")

    def __init__(self, run_id: str, root: str | None,
                 state: dict[str, Any]) -> None:
        self.run_id = str(run_id)
        self.root = root
        self._state = state

    # -- construction ------------------------------------------------------

    @classmethod
    def create(cls, *, deliverable: str,
               raw_request_ref: str = "",
               sources: Sequence[Mapping[str, Any]] = (),
               requirements: str = "",
               references: Mapping[str, Any] | None = None,
               settings: Mapping[str, Any] | None = None,
               root: str | None = None,
               run_id: str | None = None,
               created_at: str | None = None,
               fleet: Callable[[], Mapping[str, Any]] | None = None,
               route: Callable[[], Mapping[str, Any]] | None = None,
               ) -> "ScriptFirstRun":
        """Doc Stage 4: capture the immutable input snapshot, then start.

        Every source is checked before the snapshot is built:

        * a supplied ``hash`` that does not match ``prompt_digest(text)`` is a
          SOURCE_DIGEST_MISMATCH — the caller is holding a different revision of
          that prompt than the one it sent, and snapshotting either of them
          would record a provenance nobody can reproduce;
        * a ``persisted_at`` LATER than this run's start is EXCLUDED, with the
          reason recorded on the source. It stays in ``sources`` and out of
          ``prompts_before_run``, which is the difference between an exclusion
          an operator can see and one they have to infer.

        ``fleet`` and ``route`` are the live-read seams. A test passes lambdas
        and this constructor touches no catalog, no router and no registry."""
        started = _text(created_at) or _utc_now()
        rid = _text(run_id) or new_run_id()
        problems: list[str] = []
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        for index, raw in enumerate(sources or ()):
            if not isinstance(raw, Mapping):
                problems.append(f"source[{index}] is {type(raw).__name__}, "
                                f"not an object")
                continue
            source_id = _text(raw.get("prompt_id") or raw.get("source_id")
                              or raw.get("id")) or f"source-{index + 1}"
            text = str(raw.get("text") or raw.get("prompt") or "")
            if not text.strip():
                problems.append(f"source {source_id!r} carries no text; a "
                                f"snapshot entry with no prompt is a provenance "
                                f"record of nothing")
                continue
            digest = prompt_digest(text)
            claimed = _text(raw.get("hash") or raw.get("digest")
                            or raw.get("revision_id"))
            if claimed and claimed != digest and not _HEX64.match(claimed):
                # A revision id that is not a content hash is carried, not
                # compared — the doc allows either.
                pass
            elif claimed and claimed != digest:
                problems.append(
                    f"source {source_id!r} claims content hash "
                    f"{claimed[:12]}… but its text digests to {digest[:12]}… — "
                    f"the caller is holding a different revision than the one "
                    f"it sent")
                continue
            persisted_at = _text(raw.get("persisted_at") or raw.get("created_at"))
            included = True
            reason = ""
            if persisted_at and persisted_at > started:
                included = False
                reason = (f"persisted at {persisted_at}, which is AFTER this "
                          f"run started at {started}; doc Stage 4: a prompt "
                          f"qualifies as an existing prompt only if it was "
                          f"persisted before the run began")
            if digest in seen:
                included = False
                reason = reason or ("the same prompt text is already in this "
                                    "snapshot; one digest, one entry")
            seen.add(digest)
            rows.append({
                "prompt_id": source_id, "text": text, "digest": digest,
                "claimed_hash": claimed or None,
                "persisted_at": persisted_at or None,
                "included": included, "exclusion_reason": reason or None,
                "origin": _text(raw.get("origin")) or "operator",
            })

        if problems:
            raise ScriptFirstRefused(
                "SOURCE_DIGEST_MISMATCH" if any("content hash" in p
                                                for p in problems)
                else "SOURCE_INVALID",
                "the run's source prompts were refused before any snapshot was "
                "built", errors=problems)

        refs = dict(references or {})
        fleet_view = dict((fleet or live_fleet_view)())
        route_view = dict((route or live_authoring_route)())

        try:
            snapshot = GenerationSnapshot(
                raw_request_ref=_text(raw_request_ref) or f"script-first:{rid}",
                prompts_before_run=tuple(r["text"] for r in rows
                                         if r["included"]),
                operator_refs=tuple(_str_list(refs.get("operator"))),
                acquisition_refs=tuple(_str_list(refs.get("acquisition"))),
                identity_refs=tuple(_str_list(refs.get("identity"))),
                voice_refs=tuple(_str_list(refs.get("voice"))),
                deliverable=_text(deliverable),
                exclusions=tuple(_str_list(refs.get("exclusions"))),
                registry_version=fleet_view.get("registry_version") or None,
                created_at=started,
            )
        except (ValueError, TypeError) as exc:
            raise ScriptFirstRefused(
                "SOURCE_INVALID",
                f"the generation snapshot would not build: {exc}",
                errors=(str(exc),)) from exc

        state: dict[str, Any] = {
            "version": STATE_VERSION,
            "run_id": rid,
            "kind": RUN_KIND,
            "created_at": started,
            "updated_at": started,
            "deliverable": snapshot.deliverable,
            "requirements": str(requirements or ""),
            "settings": _jsonable(settings or {}),
            "references": _jsonable(refs),
            "sources": rows,
            "snapshot": snapshot.to_dict(),
            "snapshot_digest": snapshot.digest,
            "models": {"fleet": _jsonable(fleet_view),
                       "authoring_route": _jsonable(route_view)},
            "ledger": [],
            "artifacts": {},
            "lock": None,
            "lock_history": [],
            "segments": None,
            "attempts": {},
            "promotions": [],
            "events": [],
            "last_refusal": None,
        }
        run = cls(rid, root, state)
        run._event("created",
                   {"sources": len(rows),
                    "captured": len(snapshot.prompts_before_run),
                    "excluded": sum(1 for r in rows if not r["included"])})
        run.save()
        return run

    @classmethod
    def load(cls, run_id: str, root: str | None = None) -> "ScriptFirstRun":
        path = state_path(run_id, root)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError) as exc:
            logger.debug("script_first: no readable run at %s (%s)", path, exc)
            raise RunNotFound(run_id) from exc
        if not isinstance(payload, Mapping):
            raise RunNotFound(run_id)
        if int(payload.get("version", 0)) != STATE_VERSION:
            raise ScriptFirstRefused(
                "RUN_NOT_FOUND",
                f"run {run_id!r} was journalled at state version "
                f"{payload.get('version')}, not {STATE_VERSION}; it is not "
                f"read rather than misread",
                detail={"run_id": str(run_id)})
        return cls(str(payload.get("run_id") or run_id), root, dict(payload))

    @staticmethod
    def list_runs(root: str | None = None) -> list[dict[str, Any]]:
        """Every run on this box, newest first, as summaries."""
        directory = os.path.join(root or default_run_root(), "runs", RUN_KIND)
        out: list[dict[str, Any]] = []
        try:
            names = os.listdir(directory)
        except OSError:
            return out
        for name in names:
            if name.startswith("_") or name.startswith("."):
                continue
            try:
                out.append(ScriptFirstRun.load(name, root).summary())
            except ScriptFirstRefused:
                continue
        out.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
        return out

    # -- persistence -------------------------------------------------------

    def save(self) -> "ScriptFirstRun":
        self._state["updated_at"] = _utc_now()
        try:
            _atomic_write_json(state_path(self.run_id, self.root), self._state)
        except OSError as exc:
            raise ScriptFirstRefused(
                "RUN_WRITE_FAILED",
                f"run {self.run_id} could not be journalled: "
                f"{type(exc).__name__}: {exc}",
                errors=(str(exc),)) from exc
        return self

    def _event(self, name: str, detail: Mapping[str, Any] | None = None) -> None:
        self._state.setdefault("events", []).append(
            {"at": _utc_now(), "event": str(name),
             "detail": _jsonable(detail or {})})
        # Anything that gets far enough to journal an event has superseded the
        # last refusal; leaving a stale one on the run would put a fixed error
        # back on the screen after the fix.
        self._state["last_refusal"] = None

    # -- reading -----------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        """The full run state — the GET body, verbatim, plus the derived
        read-only views a screen needs (lock status, per-segment provenance)."""
        out = dict(self._state)
        out["locked"] = self.is_locked
        out["digests"] = self.digests()
        out["state_path"] = state_path(self.run_id, self.root)
        out["limitations"] = self.limitations()
        return out

    def summary(self) -> dict[str, Any]:
        specs = (self._state.get("segments") or {}).get("specs") or []
        return {
            "run_id": self.run_id,
            "created_at": self._state.get("created_at"),
            "updated_at": self._state.get("updated_at"),
            "deliverable": self._state.get("deliverable"),
            "snapshot_digest": self._state.get("snapshot_digest"),
            "registry_version": self.registry_version,
            "locked": self.is_locked,
            "lock_digest": (self._state.get("lock") or {}).get("digest"),
            "revision": (self._state.get("lock") or {}).get("payload", {})
                        .get("revision"),
            # Only stages that actually HOLD an artifact. A stage whose last
            # attempt ended in a gap has an entry (the gap is kept, on purpose)
            # but no digest, and listing it here would put "we have a plot" on
            # a run summary that has no plot.
            "stages": [s for s in ARTIFACT_STAGES
                       if ((self._state.get("artifacts") or {})
                           .get(s) or {}).get("digest")],
            "gapped_stages": [s for s in ARTIFACT_STAGES
                              if ((self._state.get("artifacts") or {})
                                  .get(s) or {}).get("gap")],
            "segments": len(specs),
            "attempts": sum(len(v) for v in
                            (self._state.get("attempts") or {}).values()),
        }

    def digests(self) -> dict[str, Any]:
        """Every artifact digest this run holds — the provenance column."""
        out: dict[str, Any] = {"snapshot": self._state.get("snapshot_digest")}
        for stage, entry in (self._state.get("artifacts") or {}).items():
            out[stage] = (entry or {}).get("digest")
        lock = self._state.get("lock") or {}
        out["production_lock"] = lock.get("digest")
        segments = self._state.get("segments") or {}
        out["segments"] = {s.get("segment_id"): s.get("digest")
                           for s in (segments.get("specs") or [])}
        return out

    def limitations(self) -> list[str]:
        """What this run is NOT. Never empty — a pipeline this early with
        nothing to disclose would be the least believable thing it could say."""
        out: list[str] = []
        artifacts = self._state.get("artifacts") or {}
        if not artifacts.get("audio_master"):
            out.append(
                "no AudioMaster: audio.tts is ineligible on this fleet, so "
                "there is no MEASURED dialogue timing and the shot windows are "
                "estimates until one exists (doc Stage 8 cuts shots TO the "
                "audio, never the reverse)")
        draft = (artifacts.get("shot_plan") or {}).get("payload") or {}
        if draft and not draft.get("audio_first", False):
            out.append("the shot plan was built WITHOUT an audio master: every "
                       "window is an estimate laid end to end, usable for "
                       "planning and not lockable as a timeline")
        if self._state.get("attempts"):
            out.append("segment attempts run the oracle image path "
                       "(image.generate) for keyframes; the clip path is a "
                       "typed gap until the studio produce_clip spine is bound")
        gaps = [s for s, e in artifacts.items() if (e or {}).get("gap")]
        if gaps:
            out.append(f"stage(s) {sorted(gaps)} last ended in an authoring "
                       f"gap; the raw model reply is kept, no artifact was "
                       f"coerced from it")
        if not out:
            out.append("this run has produced no artifact yet")
        return out

    @property
    def registry_version(self) -> str | None:
        return ((self._state.get("models") or {}).get("fleet") or {}) \
            .get("registry_version")

    @property
    def is_locked(self) -> bool:
        return bool(self._state.get("lock"))

    @property
    def snapshot(self) -> GenerationSnapshot:
        return GenerationSnapshot.from_dict(self._state["snapshot"])

    @property
    def ledger(self) -> RunPromptLedger:
        return RunPromptLedger(digests=tuple(self._state.get("ledger") or ()))

    def _store_ledger(self, ledger: RunPromptLedger) -> None:
        self._state["ledger"] = list(ledger.digests)

    def _artifact(self, stage: str) -> dict[str, Any] | None:
        return (self._state.get("artifacts") or {}).get(stage)

    def artifact_payload(self, stage: str) -> dict[str, Any] | None:
        entry = self._artifact(stage)
        if not entry or entry.get("gap"):
            return None
        return entry.get("payload")

    # -- typed rehydration -------------------------------------------------

    def plot(self) -> PlotSpec | None:
        payload = self.artifact_payload("plot")
        return PlotSpec.from_dict(payload) if payload else None

    def screenplay(self) -> Screenplay | None:
        payload = self.artifact_payload("screenplay")
        return Screenplay.from_dict(payload) if payload else None

    def continuity(self) -> ContinuityBible | None:
        payload = self.artifact_payload("continuity")
        return ContinuityBible.from_dict(payload) if payload else None

    def shot_draft(self) -> ShotPlanDraft | None:
        payload = self.artifact_payload("shot_plan")
        return ShotPlanDraft.from_dict(payload) if payload else None

    def audio_master(self) -> AudioMaster | None:
        payload = self.artifact_payload("audio_master")
        return AudioMaster.from_dict(payload) if payload else None

    def lock(self) -> ProductionLock | None:
        entry = self._state.get("lock")
        return ProductionLock.from_dict(entry["payload"]) if entry else None

    def specs(self) -> tuple[SegmentSpec, ...]:
        rows = (self._state.get("segments") or {}).get("specs") or []
        return tuple(SegmentSpec.from_dict(r["spec"]) for r in rows)

    def spec(self, segment_id: str) -> SegmentSpec:
        for item in self.specs():
            if item.segment_id == segment_id:
                return item
        raise ScriptFirstRefused(
            "SEGMENT_UNKNOWN",
            f"run {self.run_id} has no segment {segment_id!r}",
            detail={"segments": [s.segment_id for s in self.specs()]})

    # -- guards ------------------------------------------------------------

    def _require_unlocked(self, what: str) -> None:
        if self.is_locked:
            raise ScriptFirstRefused(
                "ALREADY_LOCKED",
                f"{what} is refused: this production is LOCKED at revision "
                f"{(self._state['lock'].get('payload') or {}).get('revision')}. "
                f"Doc Stage 10: a post-lock material change is a new REVISION, "
                f"never an edit — POST /video/script/runs/{self.run_id}/revise "
                f"with a reason",
                detail={"lock_digest": self._state["lock"].get("digest")})

    def _require_locked(self, what: str) -> ProductionLock:
        lock = self.lock()
        if lock is None:
            raise ScriptFirstRefused(
                "NOT_LOCKED",
                f"{what} is refused: this run has no production lock yet. Doc "
                f"Stage 11 versions and locks the screenplay, the continuity "
                f"bible, the audio master and the shot plan BEFORE final "
                f"segment prompts are compiled")
        return lock

    # -- Phase 1: authoring ------------------------------------------------

    def author(self, stage: str, *, input_text: str = "",
               mode: str | None = None,
               deadline_s: float | None = None,
               llm: Any = None,
               matrix_loader: Callable[[], tuple[Any, str]] | None = None,
               ) -> dict[str, Any]:
        """Author ``stage`` with the live LLM, or record a typed gap.

        ``llm`` is the injected model (k110's ``(prompt) -> str``). Omitted, it
        is ``bind_llm()`` — which itself returns an ``AuthoringGap`` rather
        than raising when no text model is eligible, so "we could not try" and
        "we tried and the reply was invalid" arrive in the same typed shape.

        k114's follow-up, landed: when ``llm`` is omitted, the model is not
        always the catalog default any more. ``resolve_authoring_model``
        (``matrix_loader`` is its injection seam, forwarded unchanged) asks
        k109's routing matrix for ``stage``'s operation and, when a fresh
        matrix names a primary, passes it to ``bind_llm`` as
        ``requested_model`` — otherwise ``requested_model`` stays ``None`` and
        behaviour is byte-identical to before k109 existed. EITHER WAY the
        choice and its reason are recorded on the run (``models.
        last_authoring_choice``) and on the returned artifact entry
        (``model_choice``), so an operator reading a run can tell a measured
        route from an unmeasured default.

        ``deadline_s`` is passed straight to ``bind_llm`` and bounds the ONE
        dispatch this makes. It is exposed rather than left at the oracle's
        60s default because authoring is the one synchronous oracle call whose
        honest cost is minutes: a cold model load plus a multi-kilobyte JSON
        artifact does not fit in the deadline sized for a chat turn, and the
        result is a TIMEOUT gap that says nothing about the model. The bound
        still exists (``runtime`` clamps it to [5, 600]) — this widens it, it
        does not remove it.

        NEVER returns a coerced artifact: the outcome is the artifact or a
        422 carrying the validator errors verbatim plus the raw reply."""
        if stage not in AUTHORED_STAGES:
            raise ScriptFirstRefused(
                "STAGE_UNKNOWN",
                f"stage {stage!r} is not authored by a model; the authored "
                f"stages are {list(AUTHORED_STAGES)} and the rest "
                f"({[s for s in ARTIFACT_STAGES if s not in AUTHORED_STAGES]}) "
                f"are DERIVED from the screenplay at lock time")
        self._require_unlocked(f"authoring {stage}")

        if llm is not None:
            model_choice: dict[str, Any] = {
                "stage": stage, "operation": AUTHORING_OPERATIONS.get(stage),
                "source": "injected", "requested_model": None,
                "reason": "the caller injected an llm callable directly; the "
                          "k109 matrix was not consulted"}
            model = llm
        else:
            model_choice = resolve_authoring_model(stage,
                                                   matrix_loader=matrix_loader)
            model = bind_llm(
                requested_model=model_choice.get("requested_model"),
                deadline_s=(float(deadline_s) if deadline_s is not None
                            else None),
                objective=f"author a {stage} artifact")
        self._state.setdefault("models", {})["last_authoring_choice"] = \
            model_choice
        if isinstance(model, AuthoringGap):
            return self._record_gap(stage, model)

        if stage == "plot":
            text = _text(input_text) or self._plot_input()
            if not text:
                raise ScriptFirstRefused(
                    "ARTIFACT_MISSING",
                    "there is nothing to build a plot from: this run captured "
                    "no source prompts and carries no requirements text. Doc "
                    "Stage 5's 'minimal' mode still needs a request",
                    errors=("no input text and no snapshot prompts",))
            picked = mode or plot_input_mode(text)
            result = author_plot(text, model, mode=picked)
        else:
            plot = self.plot()
            if plot is None:
                raise ScriptFirstRefused(
                    "ARTIFACT_MISSING",
                    "a screenplay is authored FROM a plot (doc Stage 6 follows "
                    "Stage 5); this run has no valid PlotSpec yet",
                    errors=("no plot artifact on this run",))
            result = author_screenplay(plot, model)

        model_id = model_choice.get("requested_model") or (
            (self._state.get("models") or {})
            .get("authoring_route") or {}).get("model_id")
        # TODO-9 / k113a: the authoring outcome is evidence about the authoring
        # model — a validator-rejected plot/screenplay (AuthoringGap) is a FAIL,
        # an accepted artifact a PASS — under the matrix operation for the stage.
        try:
            from hugpy_oracle import selection as _selection
            from hugpy_oracle.contracts import RepairCode as _RC
            _selection.note_verdict(
                "text.chat", model_id, hard_pass=not isinstance(result, AuthoringGap),
                repair_code=(_RC.FORMAT_MISMATCH if isinstance(result, AuthoringGap) else None))
        except Exception as exc:  # noqa: BLE001 — evidence must not break authoring, but never silently
            import logging as _logging
            _logging.getLogger(__name__).warning("authoring verdict not ledgered: %s: %s", type(exc).__name__, exc)
        if isinstance(result, AuthoringGap):
            return self._record_gap(stage, result)
        return self._store_artifact(
            stage, result, provenance="authored",
            note=f"authored by the live {model_id or 'text.chat'} route "
                 f"({model_choice.get('reason')})",
            model_choice=model_choice)

    def _plot_input(self) -> str:
        """Doc Phase 0: normalize the available inputs WITHOUT overwriting the
        originals. The snapshot's prompts and the operator's requirements are
        concatenated for the model; neither is mutated, and the artifact
        records which prompts it saw through the snapshot digest."""
        parts: list[str] = []
        requirements = _text(self._state.get("requirements"))
        if requirements:
            parts.append(f"REQUIREMENTS:\n{requirements}")
        captured = [r["text"] for r in self._state.get("sources", [])
                    if r.get("included")]
        if captured:
            parts.append("EXISTING PROMPTS (the immutable snapshot):\n"
                         + "\n".join(f"- {t}" for t in captured))
        deliverable = _text(self._state.get("deliverable"))
        if deliverable:
            parts.append(f"DELIVERABLE: {deliverable}")
        return "\n\n".join(parts)

    def _record_gap(self, stage: str, gap: AuthoringGap) -> dict[str, Any]:
        self._state.setdefault("artifacts", {})[stage] = {
            "stage": stage, "payload": None, "digest": None,
            "provenance": "gap", "at": _utc_now(),
            "gap": gap.to_dict(),
        }
        self._event("authoring_gap", {"stage": stage, "code": gap.code,
                                      "errors": list(gap.errors)})
        self.save()
        raise ScriptFirstRefused(
            "AUTHORING_GAP",
            f"{stage} authoring did not produce an artifact ({gap.code}); the "
            f"raw reply is kept and nothing was coerced from it",
            errors=gap.errors, detail={"stage": stage, "gap": gap.to_dict()})

    def put_artifact(self, stage: str,
                     payload: Mapping[str, Any]) -> dict[str, Any]:
        """The operator's edited JSON, validated through the SAME constructor
        the model's reply goes through.

        There is no lenient path. A field an operator hand-typed is held to
        every rule the authored one is, which is the only reading of "allow the
        user to review, edit and lock these artifacts" that does not make the
        edit button a way around the validator."""
        if stage not in ARTIFACT_STAGES:
            raise ScriptFirstRefused(
                "STAGE_UNKNOWN",
                f"stage {stage!r} is not one of {list(ARTIFACT_STAGES)}")
        self._require_unlocked(f"editing {stage}")
        if not isinstance(payload, Mapping):
            raise ScriptFirstRefused(
                "ARTIFACT_INVALID",
                f"{stage} must be a JSON object, got "
                f"{type(payload).__name__}")
        builder = {
            "plot": PlotSpec.from_dict,
            "screenplay": Screenplay.from_dict,
            "continuity": ContinuityBible.from_dict,
            "shot_plan": ShotPlanDraft.from_dict,
            "audio_master": AudioMaster.from_dict,
        }[stage]
        try:
            artifact = builder(payload)
        except ScreenplayError as exc:
            raise ScriptFirstRefused(
                "ARTIFACT_INVALID",
                f"the edited {stage} does not validate", errors=exc.errors,
                detail={"stage": stage}) from exc
        except ProductionError as exc:
            raise ScriptFirstRefused(
                "ARTIFACT_INVALID",
                f"the edited {stage} does not validate",
                errors=getattr(exc, "errors", ()) or (str(exc),),
                detail={"stage": stage}) from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise ScriptFirstRefused(
                "ARTIFACT_INVALID",
                f"the edited {stage} does not validate",
                errors=(f"{type(exc).__name__}: {exc}",),
                detail={"stage": stage}) from exc
        if stage == "screenplay":
            plot = self.plot()
            if plot is not None and artifact.plot_digest != plot.digest:
                # Provenance is the one thing an editor may not author.
                artifact = replace(artifact, plot_digest=plot.digest)
        return self._store_artifact(stage, artifact, provenance="operator_edit",
                                    note="hand-edited JSON, validated through "
                                         "the artifact constructor")

    def _store_artifact(self, stage: str, artifact: Any, *,
                        provenance: str, note: str = "",
                        model_choice: Mapping[str, Any] | None = None,
                        ) -> dict[str, Any]:
        entry = {
            "stage": stage,
            "payload": _jsonable(artifact.to_dict()),
            "digest": artifact.digest,
            "provenance": provenance,
            "note": note,
            "at": _utc_now(),
            "gap": None,
            # k114's follow-up, landed: which model authored this artifact and
            # why (k109 matrix primary, or an honest fallback reason). None
            # for every DERIVED stage (shot_plan, continuity) and for the
            # screenplay's own lock-time copy — those are not authored here.
            "model_choice": dict(model_choice) if model_choice else None,
        }
        self._state.setdefault("artifacts", {})[stage] = entry
        # An edit upstream invalidates what was derived from it. Dropping the
        # derived artifacts is the honest move: keeping a continuity bible built
        # from a screenplay that has since changed is exactly the stale-artifact
        # failure content addressing exists to catch.
        if stage in ("plot", "screenplay"):
            for derived in ("continuity", "shot_plan"):
                if self._state["artifacts"].pop(derived, None) is not None:
                    self._event("derived_dropped",
                                {"stage": derived, "because": stage})
        if stage == "plot":
            self._state["artifacts"].pop("screenplay", None)
        self._event("artifact_stored", {"stage": stage, "digest": artifact.digest,
                                        "provenance": provenance})
        self.save()
        return entry

    # -- Phase 1 close: the lock -------------------------------------------

    def build_preproduction(self) -> dict[str, Any]:
        """Derive the continuity bible and the shot plan from the screenplay.

        Idempotent and model-free — k110's ``build_continuity`` takes no LLM and
        never will. Runs on its own so the two read-only viewers have something
        to show BEFORE the lock is attempted, including on a run whose lock
        will honestly refuse for want of an audio master."""
        play = self.screenplay()
        if play is None:
            raise ScriptFirstRefused(
                "ARTIFACT_MISSING",
                "continuity and the shot plan are DERIVED from the screenplay; "
                "this run has no valid Screenplay yet",
                errors=("no screenplay artifact on this run",))
        self._require_unlocked("re-deriving pre-production")
        master = self.audio_master()
        try:
            draft = build_shot_plan(play, master, plot=self.plot())
            bible = build_continuity(play, draft)
        except ScreenplayError as exc:
            raise ScriptFirstRefused(
                "ARTIFACT_INVALID",
                f"pre-production would not build: {exc}",
                errors=exc.errors) from exc
        self._store_artifact("shot_plan", draft, provenance="derived",
                             note="build_shot_plan(screenplay"
                                  + (", audio_master" if master else "")
                                  + ", plot)")
        self._store_artifact("continuity", bible, provenance="derived",
                             note="build_continuity(screenplay, shot_plan)")
        return {"shot_plan": self._artifact("shot_plan"),
                "continuity": self._artifact("continuity")}

    def lock_run(self, *, audio_master: Mapping[str, Any] | None = None,
                 identity_refs: Sequence[str] | None = None,
                 locked_at: str | None = None) -> dict[str, Any]:
        """Doc Stage 11 — version and lock the whole production, or refuse.

        The refusals are k104's and k110's, unchanged and unwrapped. The ONE
        this module adds is the missing audio master, and it is a refusal
        rather than an invention on purpose: ``ProductionLock.lock()`` requires
        a LOCKED ``AudioMaster`` because Stage 8 puts the definitive audio
        before final shot timing, and this fleet cannot synthesize one today.
        Manufacturing a placeholder master with fabricated track refs would
        make every downstream window a lie that inspection cannot catch."""
        self._require_unlocked("locking")
        play = self.screenplay()
        if play is None:
            raise ScriptFirstRefused(
                "ARTIFACT_MISSING",
                "there is no screenplay to lock (doc Stage 6 precedes Stage 11)",
                errors=("no screenplay artifact on this run",))

        if audio_master is not None:
            self.put_artifact("audio_master", audio_master)
        master = self.audio_master()
        if master is None:
            raise ScriptFirstRefused(
                "AUDIO_MASTER_MISSING",
                "this production cannot be locked: doc Stage 8 puts the "
                "definitive audio timeline BEFORE final shot timing, and this "
                "run has no AudioMaster. Supply one (POST .../audio_master, or "
                "audio_master in the lock body) or seat TTS.",
                errors=("no AudioMaster on this run",),
                detail={"capability": "audio.tts",
                        "requirement": SEAM_REQUIREMENTS["audio.tts"],
                        "eligible": self._capability_eligible("audio.tts")})

        # Derive fresh — the lock must close over what the screenplay says NOW,
        # not over a bible built two edits ago.
        try:
            draft = build_shot_plan(play, master, plot=self.plot())
            bible = build_continuity(play, draft)
        except ScreenplayError as exc:
            raise ScriptFirstRefused(
                "LOCK_REFUSED", f"pre-production would not build: {exc}",
                errors=exc.errors) from exc

        locked_play = play if play.locked else play.lock()
        ledger = self.ledger
        try:
            lock = lock_production(
                self.snapshot, screenplay=locked_play, audio_master=master,
                continuity=bible, shots=draft,
                identity_refs=(tuple(identity_refs)
                               if identity_refs is not None else None),
                registry_version=self.registry_version,
                locked_at=_text(locked_at) or _utc_now(),
                run_prompts=ledger)
        except RunPromptRefused as exc:
            raise ScriptFirstRefused(
                "LOCK_REFUSED",
                f"invariant 9: {exc}",
                errors=(str(exc),),
                detail={"prompt_digest": getattr(exc, "prompt_digest", "")}) from exc
        except (LockRefused, ScreenplayError) as exc:
            raise ScriptFirstRefused(
                "LOCK_REFUSED", str(exc),
                errors=getattr(exc, "errors", ()) or (str(exc),)) from exc
        except (ValueError, TypeError) as exc:
            raise ScriptFirstRefused(
                "LOCK_REFUSED", f"{type(exc).__name__}: {exc}",
                errors=(str(exc),)) from exc

        self._store_artifact("screenplay", locked_play,
                             provenance=(self._artifact("screenplay") or {})
                             .get("provenance", "authored"),
                             note="locked at Stage 11")
        self._store_artifact("shot_plan", draft, provenance="derived",
                             note="locked shot plan")
        self._store_artifact("continuity", bible, provenance="derived",
                             note="locked continuity bible")
        entry = {"payload": _jsonable(lock.to_dict()), "digest": lock.digest,
                 "at": _utc_now(),
                 "parent_digests": list(lock.parent_digests)}
        self._state["lock"] = entry
        self._state.setdefault("lock_history", []).append(
            {"revision": lock.revision, "digest": lock.digest,
             "reason": lock.revision_reason or "initial lock",
             "at": entry["at"]})
        self._event("locked", {"digest": lock.digest,
                               "parents": len(lock.parent_digests)})
        self.save()
        return entry

    def _capability_eligible(self, name: str) -> bool | None:
        for row in ((self._state.get("models") or {}).get("fleet") or {}) \
                .get("capabilities", []):
            if row.get("name") == name:
                return bool(row.get("eligible"))
        return None

    def revise(self, reason: str, *,
               artifacts: Mapping[str, Any] | None = None,
               changes: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Doc Stage 10 — a post-lock change is revision N+1, with a reason.

        ``artifacts`` re-validates and re-stores edited artifacts and folds
        their new digests into the revision; ``changes`` names any of k104's
        ``REVISABLE_FIELDS`` directly. A revision with no reason cannot even be
        constructed — that refusal is k104's, not a check here."""
        lock = self._require_locked("revising")
        text = _text(reason)
        if not text:
            raise ScriptFirstRefused(
                "REVISION_REASON_MISSING",
                "a post-lock material change with nothing written down is an "
                "unaudited production change (doc Stage 10); `reason` is "
                "mandatory")
        edits = dict(changes or {})
        for stage, payload in (artifacts or {}).items():
            if stage not in ARTIFACT_STAGES:
                raise ScriptFirstRefused(
                    "STAGE_UNKNOWN",
                    f"stage {stage!r} is not one of {list(ARTIFACT_STAGES)}")
            # The lock guard is bypassed deliberately: this IS the sanctioned
            # post-lock path, and it ends in a new revision rather than an edit.
            saved = self._state.pop("lock")
            try:
                entry = self.put_artifact(stage, payload)
            finally:
                self._state["lock"] = saved
            field = {"plot": None, "screenplay": "screenplay_digest",
                     "continuity": "continuity_digest",
                     "shot_plan": "shot_plan_digest",
                     "audio_master": "audio_master_digest"}[stage]
            if field:
                digest = entry["digest"]
                if stage == "shot_plan":
                    digest = ShotPlanDraft.from_dict(entry["payload"]).plan.digest
                edits[field] = digest
        try:
            revised = lock.revise(text, **edits)
        except (ValueError, TypeError, ProductionError) as exc:
            raise ScriptFirstRefused(
                "LOCK_REFUSED", f"the revision was refused: {exc}",
                errors=(str(exc),)) from exc
        entry = {"payload": _jsonable(revised.to_dict()),
                 "digest": revised.digest, "at": _utc_now(),
                 "parent_digests": list(revised.parent_digests)}
        self._state["lock"] = entry
        self._state.setdefault("lock_history", []).append(
            {"revision": revised.revision, "digest": revised.digest,
             "reason": revised.revision_reason, "at": entry["at"],
             "parent_revision": revised.parent_revision})
        # A new lock digest means new seeds and new parents: the compiled specs
        # belong to the PREVIOUS revision and must not be presented as this
        # one's. They are dropped, not silently re-pointed (doc: retry and
        # resume must not mix artifact versions).
        if self._state.get("segments"):
            self._state["segments"] = None
            self._event("segments_dropped",
                        {"because": f"lock revised to {revised.revision}"})
        self._event("revised", {"revision": revised.revision, "reason": text,
                                "digest": revised.digest})
        self.save()
        return entry

    # -- Phase 2: the sibling compile --------------------------------------

    def compile(self, *, tone: float | None = None,
                seed_salt: int | None = None,
                negative_prompt: str | None = None,
                catalog_view: Callable[[], Mapping[str, Any]] | None = None,
                ) -> dict[str, Any]:
        """Doc Stage 14 — every segment prompt written from the LOCK, as
        siblings.

        The prompt writer is WRAPPED so every prompt it mints lands in the run's
        ``RunPromptLedger`` — the seam k104 and k110 both left to the caller,
        and without which Stage 4's rule is a mechanism nobody is using. After
        compilation ``snapshot.assert_pre_run(ledger)`` runs, which is where a
        snapshot that grew mid-run is actually caught."""
        lock = self._require_locked("compiling segments")
        master, bible, draft = (self.audio_master(), self.continuity(),
                                self.shot_draft())
        missing = [n for n, a in (("audio_master", master), ("continuity", bible),
                                  ("shot_plan", draft)) if a is None]
        if missing:
            raise ScriptFirstRefused(
                "ARTIFACT_MISSING",
                f"the lock names artifact(s) {missing} that this run no longer "
                f"holds", errors=tuple(f"missing {m}" for m in missing))

        settings = dict(self._state.get("settings") or {})
        play = self.screenplay()
        ledger = self.ledger
        from hugpy_oracle.segments import default_prompt_writer

        def recording_writer(context: Any, index: int) -> str:
            written = default_prompt_writer(context, index)
            ledger.record(written)
            return written

        try:
            specs = compile_segments(
                lock,
                snapshot=self.snapshot, audio_master=master, continuity=bible,
                shot_plan=draft.plan, tone=float(
                    tone if tone is not None else settings.get("tone", 0.5)),
                prompt_writer=recording_writer,
                negative_prompt=(negative_prompt
                                 if negative_prompt is not None
                                 else settings.get("negative_prompt")),
                dialogue=({l.line_id: l.text for l in play.lines}
                          if play is not None else None),
                scene_refs={d.segment_id: d.scene_id for d in draft.designs},
                beats={d.segment_id: d.beat for d in draft.designs},
                seed_salt=int(seed_salt if seed_salt is not None
                              else settings.get("seed_salt", 0)),
            )
        except (CompileRefused, SiblingViolation) as exc:
            raise ScriptFirstRefused("COMPILE_REFUSED", str(exc),
                                     errors=(str(exc),)) from exc
        except ProductionError as exc:
            raise ScriptFirstRefused("COMPILE_REFUSED", str(exc),
                                     errors=(str(exc),)) from exc
        except (ValueError, TypeError) as exc:
            raise ScriptFirstRefused(
                "COMPILE_REFUSED", f"{type(exc).__name__}: {exc}",
                errors=(str(exc),)) from exc

        try:
            self.snapshot.assert_pre_run(ledger)
        except RunPromptRefused as exc:
            raise ScriptFirstRefused(
                "COMPILE_REFUSED",
                f"invariant 9: a prompt minted during this run reached the "
                f"immutable snapshot — {exc}", errors=(str(exc),),
                detail={"prompt_digest": getattr(exc, "prompt_digest", "")}
            ) from exc
        self._store_ledger(ledger)

        graph = to_plan_graph(specs, lock)
        report = self._validate(graph, catalog_view)
        rows = []
        for item in specs:
            rows.append({
                "segment_id": item.segment_id,
                "index": item.index,
                "digest": item.digest,
                "spec": _jsonable(item.to_dict()),
                "prompt": item.prompt,
                "seed_base": item.seed_base,
                "parents": list(item.parents),
                "lock_digest": item.lock_digest,
                "scene_ref": item.scene_ref,
                "joint_mode": item.joint_mode,
                "window": [item.audio_window[0], item.audio_window[1],
                           list(item.audio_window[2])],
                "rubric": list(item.rubric),
            })
        entry = {
            "compiled_at": _utc_now(),
            "lock_digest": lock.digest,
            "revision": lock.revision,
            "parent_digests": list(lock.parent_digests),
            "specs": rows,
            "graph": {"graph_id": getattr(graph, "graph_id", "production"),
                      "nodes": [getattr(n, "node_id", "") for n in graph.nodes],
                      "structure_digest": _structure_digest(graph),
                      "payload": _jsonable(graph.to_dict())},
            "validation": report,
            "execution_order": {
                mode: [list(batch) for batch in
                       _order(specs, mode, lock)]
                for mode in ("sequential", "parallel")},
            "sibling_shape": {
                "parent": "production_lock",
                "parent_digest": lock.digest,
                "children": [s.segment_id for s in specs],
                "note": "every segment names the SAME parents (the lock and "
                        "the artifacts it locked) and no segment names another "
                        "segment — this is a fan-out, not a chain",
            },
        }
        self._state["segments"] = entry
        self._event("compiled", {"segments": len(rows),
                                 "lock_digest": lock.digest,
                                 "validation_ok": report.get("ok")})
        self.save()
        return entry

    def _validate(self, graph: Any,
                  catalog_view: Callable[[], Mapping[str, Any]] | None
                  ) -> dict[str, Any]:
        try:
            from hugpy_oracle.contracts import GoalSpec
            from hugpy_oracle.validator import validate
            view = (catalog_view or live_catalog_view)()
            goal = GoalSpec(objective=_text(self._state.get("deliverable"))
                            or "a script-first production",
                            raw_prompt=_text(self._state.get("requirements"))
                            or _text(self._state.get("deliverable")))
            return _jsonable(validate(graph, view, goal).to_dict())
        except Exception as exc:                   # noqa: BLE001
            return {"ok": None, "errors": [], "warnings": [],
                    "note": f"the static validator could not run: "
                            f"{type(exc).__name__}: {exc}"}

    # -- Phase 2: dispatch one segment -------------------------------------

    def generate_segment(self, segment_id: str, *, kind: str = "keyframe",
                         dispatch: Callable[..., Mapping[str, Any]] | None = None,
                         ) -> dict[str, Any]:
        """One attempt at ONE segment, from its FROZEN spec.

        Regeneration is attempt N+1 against the SAME ``SegmentSpec`` — the spec
        is rehydrated from the journal and its digest is asserted against the
        recorded one, so a regeneration cannot silently pick up an edit. The
        only thing that moves between attempts is the seed
        (``spec.seed_base + attempt - 1``, k102/k104's idiom), which makes an
        attempt the same shot at a different roll, never a different shot.

        Nothing here reads a sibling. There is no parameter one could arrive
        through, and the attempt records the sibling digests as they stood
        before and after so "regenerating 1 did not touch 2" is checkable from
        the receipt rather than asserted in prose."""
        if not (self._state.get("segments") or {}).get("specs"):
            raise ScriptFirstRefused(
                "SEGMENTS_MISSING",
                "no segments have been compiled for this run yet (POST "
                f"/video/script/runs/{self.run_id}/segments)")
        if kind not in DISPATCH_KINDS:
            raise ScriptFirstRefused(
                "SEGMENT_UNKNOWN",
                f"dispatch kind {kind!r} is not one of {list(DISPATCH_KINDS)}")
        spec = self.spec(segment_id)
        recorded = next(r for r in self._state["segments"]["specs"]
                        if r["segment_id"] == segment_id)
        if recorded["digest"] != spec.digest:
            raise ScriptFirstRefused(
                "COMPILE_REFUSED",
                f"segment {segment_id} rehydrates to digest "
                f"{spec.digest[:12]}… but was compiled as "
                f"{recorded['digest'][:12]}… — the journal and the artifact "
                f"disagree and no attempt will be made against either")

        siblings_before = {r["segment_id"]: r["digest"]
                           for r in self._state["segments"]["specs"]
                           if r["segment_id"] != segment_id}
        attempts = self._state.setdefault("attempts", {}).setdefault(segment_id, [])
        number = len(attempts) + 1
        seed = int(spec.seed_base) + number - 1

        fn = dispatch or live_segment_dispatch
        try:
            body = dict(fn(spec, kind=kind, seed=seed,
                           settings=self._state.get("settings") or {}))
        except Exception as exc:                   # noqa: BLE001 - receipt data
            body = {"ok": False, "kind": kind, "seed": seed,
                    "prompt": spec.prompt, "model_id": None, "params": {},
                    "artifacts": [], "receipt": None,
                    "gap": {"code": "DISPATCH_ERROR",
                            "reasons": [f"{type(exc).__name__}: {exc}"]}}

        attempt = {
            "attempt": number,
            "segment_id": segment_id,
            "at": _utc_now(),
            "spec_digest": spec.digest,
            "lock_digest": spec.lock_digest,
            "parents": list(spec.parents),
            "registry_version": self.registry_version,
            "siblings_before": siblings_before,
            **_jsonable(body),
        }
        attempts.append(attempt)
        siblings_after = {r["segment_id"]: r["digest"]
                          for r in self._state["segments"]["specs"]
                          if r["segment_id"] != segment_id}
        attempt["siblings_after"] = siblings_after
        attempt["siblings_unchanged"] = siblings_before == siblings_after
        self._event("attempt", {"segment_id": segment_id, "attempt": number,
                                "kind": kind, "ok": bool(body.get("ok"))})
        self.save()
        return attempt

    def attempts(self, segment_id: str | None = None) -> Any:
        table = self._state.get("attempts") or {}
        if segment_id is None:
            return table
        return table.get(segment_id, [])

    # -- Phase 3: promote --------------------------------------------------

    def promote(self, *, segment_id: str | None = None,
                attempt: int | None = None,
                text: str = "", note: str = "",
                source_id: str | None = None) -> dict[str, Any]:
        """Accept an output as a persisted SOURCE for a FUTURE run.

        The doc's rule is that an accepted output may influence later work only
        through a NEW run. This makes that structural rather than advisory: the
        promoted text's digest is recorded in THIS run's ledger BEFORE the
        source file is written, so any later attempt to add it to this
        snapshot is refused by digest inside k104's own
        ``GenerationSnapshot.with_prompt`` — a refusal no edit to this file can
        forget to make. The refusal is asserted here, at promotion time, rather
        than trusted."""
        body = _text(text)
        origin: dict[str, Any] = {"run_id": self.run_id}
        if segment_id:
            spec = self.spec(segment_id)
            rows = self.attempts(segment_id)
            picked = None
            if attempt is not None:
                picked = next((a for a in rows
                               if int(a.get("attempt", 0)) == int(attempt)), None)
                if picked is None:
                    raise ScriptFirstRefused(
                        "SEGMENT_UNKNOWN",
                        f"segment {segment_id} has no attempt {attempt}",
                        detail={"attempts": [a.get("attempt") for a in rows]})
            elif rows:
                picked = rows[-1]
            origin.update({"segment_id": segment_id,
                           "spec_digest": spec.digest,
                           "lock_digest": spec.lock_digest,
                           "attempt": (picked or {}).get("attempt"),
                           "artifacts": (picked or {}).get("artifacts") or []})
            body = body or spec.prompt
        if not body:
            raise ScriptFirstRefused(
                "PROMOTE_REFUSED",
                "there is nothing to promote: pass `text`, or a `segment_id` "
                "whose prompt should become the source")

        ledger = self.ledger
        ledger.record(body)
        self._store_ledger(ledger)

        # Assert the refusal rather than trust it. If this ever stops raising,
        # invariant 9 has a hole and this call must fail loudly, not quietly
        # write a source that could re-enter its own run.
        refusal = ""
        try:
            self.snapshot.with_prompt(body, ledger=ledger)
        except RunPromptRefused as exc:
            refusal = str(exc)
        if not refusal:
            raise ScriptFirstRefused(
                "PROMOTE_REFUSED",
                "the promoted prompt was NOT refused re-entry into its own "
                "run's snapshot; invariant 9 is not being enforced and no "
                "source was written",
                errors=("GenerationSnapshot.with_prompt accepted a run-minted "
                        "prompt",))

        digest = prompt_digest(body)
        sid = _text(source_id) or f"src-{digest[:16]}"
        record = {
            "source_id": sid,
            "text": body,
            "digest": digest,
            "promoted_at": _utc_now(),
            "note": _text(note),
            "origin": _jsonable(origin),
            "usable_in": "a NEW run only",
            "refused_here": refusal,
        }
        try:
            _atomic_write_json(promoted_source_path(sid, self.root), record)
        except OSError as exc:
            raise ScriptFirstRefused(
                "RUN_WRITE_FAILED",
                f"the promoted source could not be persisted: "
                f"{type(exc).__name__}: {exc}", errors=(str(exc),)) from exc

        self._state.setdefault("promotions", []).append(record)
        self._event("promoted", {"source_id": sid, "digest": digest,
                                 "segment_id": segment_id})
        self.save()
        return record

    def record_refusal(self, exc: "ScriptFirstRefused") -> None:
        """Journal the last refusal on the run.

        Best effort, and never allowed to mask the refusal it is recording. It
        exists because a refusal is the most useful thing on the screen and the
        least durable: without this, reloading the page after a 422 loses the
        validator output that told the operator what to fix. Nothing else reads
        it, so a write failure costs a note, not a run."""
        try:
            self._state["last_refusal"] = {"at": _utc_now(), **exc.to_dict()}
            self.save()
        except Exception as write_exc:             # noqa: BLE001
            logger.debug("script_first: refusal not journalled (%s)", write_exc)

    def accepts_source(self, text: str) -> bool:
        """Whether ``text`` could still enter THIS run's snapshot. False for
        anything this run minted — the check a UI runs before offering a
        button it would have to take away."""
        try:
            self.snapshot.with_prompt(text, ledger=self.ledger)
        except (RunPromptRefused, ValueError, TypeError):
            return False
        return True


# ---------------------------------------------------------------------------
# Module-level helpers used by the run
# ---------------------------------------------------------------------------


def _structure_digest(graph: Any) -> str | None:
    fn = getattr(graph, "structure_digest", None)
    if fn is None:
        return None
    try:
        return fn() if callable(fn) else str(fn)
    except Exception:                              # noqa: BLE001
        return None


def _order(specs: Sequence[SegmentSpec], mode: str,
           lock: ProductionLock) -> tuple[tuple[str, ...], ...]:
    """``execution_order`` off the SAME graph for both modes, so the UI can
    show that parallel and sequential differ only in BATCHING — the doc's
    closing line on Stage 14."""
    try:
        return tuple(tuple(batch) for batch in
                     execution_order(specs, mode, lock=lock))
    except Exception as exc:                       # noqa: BLE001
        logger.debug("script_first: execution_order(%s) failed: %s", mode, exc)
        return ()


__all__ = [
    "ARTIFACT_STAGES",
    "AUTHORED_STAGES",
    "AUTHORING_OPERATIONS",
    "CLIP_CAPABILITY",
    "DISPATCH_KINDS",
    "KEYFRAME_CAPABILITY",
    "REFUSAL_CODES",
    "REFUSAL_STATUS",
    "RUN_KIND",
    "RUN_ROOT_ENV",
    "SEAM_REQUIREMENTS",
    "STATE_FILENAME",
    "STATE_VERSION",
    "RunNotFound",
    "ScriptFirstError",
    "ScriptFirstRefused",
    "ScriptFirstRun",
    "default_run_root",
    "list_promoted_sources",
    "live_authoring_route",
    "live_catalog_view",
    "live_fleet_view",
    "live_segment_dispatch",
    "load_promoted_source",
    "new_run_id",
    "promoted_source_path",
    "resolve_authoring_model",
    "run_dir",
    "sources_dir",
    "state_path",
]
