"""Oracle runtime (k90b): execute a resolved route through the EXISTING dispatch.

No new inference machinery. Model tasks go through the SAME normalization the
/ml amenity routes apply (:func:`normalize_kwargs` — model fold, vision b64
fold, default pool; the HTTP adapter resolves an API-key-bound pool into the
body before calling) and the engine's ``execute_prompt``
(``hugpy_engine.dispatch``), awaited on the platform's shared runtime. The
two deterministic amenities call ``hugpy_media.extract`` directly
(``extract_document`` / ``assess_url``). What THIS module adds is the wrapper the
operator ruled on: timing, one retry on WORKER_UNAVAILABLE, failure
classification, artifact extraction + sha256 — all landing in an
``ExecutionReceipt`` on every call, success or not.

Artifacts come back as plain dicts (``{kind, uri, sha256, ...}``) carrying the
inline payload (``text`` / ``data``) when the result is not file-backed —
that dict form is what the scorecard checks and the route returns; the receipt
carries the same set as typed ``ArtifactRef``s.

k101b adds the BOUND the honesty invariant needs: every dispatch runs under a
deadline (``sync_deadline_s`` — the goal's budget hint, else
``ORACLE_SYNC_DEADLINE_S``, else 60s, clamped to [5, 600]) on a worker thread,
and an expiry becomes a typed ``FailureClass.TIMEOUT`` receipt naming what the
wait was holding on. It exists because the fleet's own retry ladder is
deliberately patient — ``managers/resolvers/remote.py`` re-holds a cold/busy
worker for up to ``HUGPY_COLD_HOLD_MAX_S`` (25 minutes on the dev unit) — which
is correct for a job and a lie for a synchronous HTTP request: gunicorn drops
the connection at ``--timeout`` and the caller gets nothing at all. The oracle
does not shorten the fleet's patience; it bounds its OWN wait and ENDS, with
evidence (doc invariant 12). Every classified failure is also logged, not
silently receipted.

Provider seams (``_dispatch`` / ``_normalized_kwargs`` / ``_extract_document``
/ ``_assess_url`` / ``_selected_worker``) are module-level and lazy so tests
monkeypatch them and no GPU/network/worker is touched.

No pathlib; os.path only (project discipline).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from hugpy_oracle.contracts import (
    ArtifactKind,
    ArtifactRef,
    AuthorityKind,
    ExecutionReceipt,
    FailureClass,
    GoalSpec,
)
from hugpy_oracle.router import RouteDecision

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider seams — the existing machinery, lazily bound.
# ---------------------------------------------------------------------------


ML_POOL_ENV = "HUGPY_ML_POOL"
ML_GENERAL_ROUTE_TASKS_ENV = "HUGPY_ML_GENERAL_ROUTE_TASKS"
DEFAULT_GENERAL_ROUTE_TASKS = "image-text-to-text"


def _ml_pool() -> str:
    """The reserved worker pool ML amenities route to (env-overridable, empty
    = un-pooled). Mirrors the /ml routes' default so the oracle dispatches
    exactly where the amenity would have."""
    return (os.environ.get(ML_POOL_ENV, "") or "").strip()


def _general_route_tasks() -> set[str]:
    """Tasks that follow the GENERAL worker route instead of the reserved ML
    pool (vision by default; ``HUGPY_ML_GENERAL_ROUTE_TASKS`` comma list)."""
    raw = os.environ.get(ML_GENERAL_ROUTE_TASKS_ENV, DEFAULT_GENERAL_ROUTE_TASKS)
    return {t.strip() for t in raw.split(",") if t.strip()}


def normalize_kwargs(task: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """Request body -> ``execute_prompt`` kwargs: the /ml amenity
    normalization (model fold, vision b64 fold, default pool) as a pure
    function. Underscore-prefixed keys are internal routing controls and are
    dropped; the caller-resolved ``pool`` (an API-key-bound pool is the HTTP
    adapter's business) wins over the env default."""
    kwargs = {k: v for k, v in dict(body).items()
              if not str(k).startswith("_") and v is not None}
    kwargs["task"] = task
    if "model_key" not in kwargs:
        m = kwargs.pop("model", None)
        if m:
            kwargs["model_key"] = m
    if task == "image-text-to-text" and "images" not in kwargs:
        b64 = kwargs.pop("image_b64", None)
        if b64:
            kwargs["images"] = [b64]
    default_pool = "" if task in _general_route_tasks() else _ml_pool()
    eff = kwargs.get("pool") or default_pool
    if eff:
        kwargs["pool"] = eff
    else:
        kwargs.pop("pool", None)
    return kwargs


def _normalized_kwargs(task: str, body: dict[str, Any]) -> dict[str, Any]:
    """Seam over :func:`normalize_kwargs` (monkeypatched in tests)."""
    return normalize_kwargs(task, body)


def _await_sync(value: Any) -> Any:
    """Drive a possibly-awaitable dispatch result on the shared loop."""
    import inspect
    if not inspect.isawaitable(value):
        return value
    from hugpy_platform import async_runtime
    return async_runtime.run(value)


def _dispatch(kwargs: dict[str, Any]) -> Any:
    """The single inference front door: the engine's ``execute_prompt``,
    driven to a concrete result on the shared loop."""
    from hugpy_engine.dispatch.dispatch import execute_prompt
    return _await_sync(execute_prompt(**kwargs))


def _extract_document(path: str) -> dict[str, Any]:
    from hugpy_media.extract import extract_document
    return extract_document(path)


def _assess_url(url: str) -> dict[str, Any]:
    from hugpy_media.extract import assess_url
    return assess_url(url)


def _catalog_registry_version() -> str | None:
    """catalog.registry_version(), read fresh when the caller did not already
    have one (k105). The route (oracle_routes.py) computes it ONCE per request
    and passes it into ``execute_route`` so this fallback is not the common
    path; a caller that skips it — ``repair.execute_repair``'s second
    dispatch, a direct test — still gets an honestly stamped receipt instead
    of a silent ``None``. A catalog fault must not turn a finished execution
    into a crash: the receipt simply carries no version, never a guess."""
    try:
        from hugpy_oracle import catalog
        return catalog.registry_version()
    except Exception:  # noqa: BLE001 — a version nobody can compute is not a fault
        logger.warning("oracle runtime: catalog.registry_version() failed",
                       exc_info=True)
        return None


# ---------------------------------------------------------------------------
# The bounded wait (k101b) — invariant 12 when the fleet stalls.
# ---------------------------------------------------------------------------

SYNC_DEADLINE_ENV = "ORACLE_SYNC_DEADLINE_S"
DEFAULT_SYNC_DEADLINE_S = 60.0
MIN_SYNC_DEADLINE_S = 5.0
MAX_SYNC_DEADLINE_S = 600.0
# How long the "what was I holding on?" lookup may take. It is a diagnostic on
# an already-expired request: it must never become a second stall.
TARGET_HINT_DEADLINE_S = 2.0


class DispatchTimeout(TimeoutError):
    """THE ORACLE stopped waiting — not the worker's verdict.

    Distinct from a timeout the dispatch layer itself raises (that one arrives
    as an ordinary exception and ``classify_failure`` reads it): this one says
    the synchronous request's own deadline expired while the fleet was still
    holding. Subclasses TimeoutError so every existing classification path
    already lands it on ``FailureClass.TIMEOUT``."""


def sync_deadline_s(goal: GoalSpec | None = None) -> float:
    """How long ONE dispatch may hold a synchronous /oracle/route request.

    The goal's own budget hint wins (a caller who said "max_seconds: 30" meant
    it), else ``ORACLE_SYNC_DEADLINE_S``, else 60s. Clamped to
    [5, 600]: under 5s nothing on this fleet can answer, and past 600s the HTTP
    request is a fiction either way — gunicorn's own ``--timeout`` is 120s on
    the dev unit, and the connection dies with no response at all. The clamp is
    for CONFIGURED values; an explicit ``execute_route(deadline_s=...)`` is a
    programmatic instruction (the orchestrator, tests) and is honored as given.
    """
    hint: float | None = None
    budget = getattr(goal, "budget", None)
    if budget is not None:
        raw_hint = getattr(budget, "max_seconds", None)
        hint = float(raw_hint) if raw_hint is not None else None
    if hint is None:
        raw = os.environ.get(SYNC_DEADLINE_ENV, "").strip()
        if raw:
            try:
                hint = float(raw)
            except ValueError:
                logger.warning("oracle: ignoring unparseable %s=%r (using %.0fs)",
                               SYNC_DEADLINE_ENV, raw, DEFAULT_SYNC_DEADLINE_S)
    if hint is None:
        hint = DEFAULT_SYNC_DEADLINE_S
    return max(MIN_SYNC_DEADLINE_S, min(MAX_SYNC_DEADLINE_S, hint))


class _Handoff:
    """A ONE-SHOT result slot shared with a worker thread.

    The whole point is the race at the deadline: a thread that finishes a
    microsecond after ``join`` gave up must not be able to write into a
    response that has already gone out. Both sides go through one lock, and
    ``claim`` either TAKES the value or CLOSES the slot forever — after which
    ``deliver`` is a no-op returning False. There is no window in between."""

    __slots__ = ("_lock", "_state", "_closed")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: tuple[str, Any] | None = None
        self._closed = False

    def deliver(self, kind: str, value: Any) -> bool:
        """Worker side: fill the slot once. False = nobody is listening any
        more (the request already answered) and the value is discarded."""
        with self._lock:
            if self._closed or self._state is not None:
                return False
            self._state = (kind, value)
            return True

    def claim(self) -> tuple[str, Any] | None:
        """Request side: take the value, or close the slot. Never blocks."""
        with self._lock:
            taken, self._state, self._closed = self._state, None, True
            return taken

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed


def _current_client_probe() -> Any:
    """This request thread's client-liveness probe (or None off a request).

    Carried onto the worker thread so abandon-on-disconnect keeps working: the
    probe lives in a thread-local, and moving the dispatch to another thread
    would otherwise silently drop the "caller hung up" cancellation."""
    try:
        from hugpy_platform import client_liveness
        return client_liveness.current()
    except Exception:  # noqa: BLE001 — a missing probe is not a fault
        return None


def _bind_client_probe(probe: Any) -> None:
    try:
        from hugpy_platform import client_liveness
        client_liveness.bind(probe)
    except Exception:  # noqa: BLE001
        pass


def run_bounded(fn: Callable[[], Any], deadline_s: float,
                label: str = "dispatch") -> Any:
    """Call ``fn()`` on a worker thread; wait at most ``deadline_s`` for it.

    A thread + ``join(timeout)``, not asyncio: gunicorn serves this route on
    sync/threaded workers and the coroutine side already lives on the shared
    loop (``_platform.async_runtime``), so the request thread is only ever
    parked. On expiry the orphaned thread keeps running — nothing can safely
    kill a thread mid-dispatch — but its result is DISCARDED by the one-shot
    slot, so it can never touch a response that already went out.

    Raises ``DispatchTimeout`` on expiry; anything ``fn`` raised is re-raised in
    the CALLING thread, so the existing classification path is unchanged."""
    if deadline_s <= 0:
        raise DispatchTimeout(
            f"{label}: no time left in the oracle deadline "
            f"({deadline_s:.3f}s remaining)")

    slot = _Handoff()
    probe = _current_client_probe()

    def _work() -> None:
        if probe is not None:
            _bind_client_probe(probe)
        try:
            value = fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised in the caller
            slot.deliver("raised", exc)
        else:
            slot.deliver("returned", value)

    thread = threading.Thread(target=_work, name=f"oracle-{label}", daemon=True)
    thread.start()
    thread.join(deadline_s)

    taken = slot.claim()
    if taken is None:
        raise DispatchTimeout(
            f"{label} did not answer within the oracle's {deadline_s:.1f}s "
            f"deadline")
    kind, value = taken
    if kind == "raised":
        raise value
    return value


def _selected_worker(model_key: str | None, task: str | None,
                     pool: str | None) -> str | None:
    """Which worker the dispatch layer WOULD pick for this request, by name.

    Read-only and no HTTP: ``remote._select`` asks the registered worker
    provider — the same in-process selection every request already uses. A
    private name on purpose; that selection is the only place the answer
    exists, and a timeout that cannot say WHICH worker it waited on is half a
    diagnosis. A module-level seam so tests monkeypatch it."""
    from hugpy_engine.resolvers.remote import _select
    worker, _spill = _select(model_key or "", pool, task)
    if not isinstance(worker, dict):
        return None
    return str(worker.get("name") or worker.get("id") or "") or None


def _dispatch_target(route: RouteDecision, kwargs: Mapping[str, Any]) -> str:
    """A phrase naming what an expired wait was holding on — the placement pin
    when there is one, else the live selection, else the honest "unknown"."""
    model = str(route.model_id or kwargs.get("model_key") or "")
    worker = route.placement if route.placement not in ("", "auto") else ""
    if not worker:
        try:
            worker = run_bounded(
                lambda: _selected_worker(model or None, route.task,
                                         kwargs.get("pool")),
                TARGET_HINT_DEADLINE_S, "worker-hint") or ""
        except Exception:  # noqa: BLE001 — a hint that costs a fault is no hint
            worker = ""
    if not model and not worker:
        return "unknown (the dispatch layer named neither a model nor a worker)"
    parts = [f"model {model!r}" if model else "the dispatch-default model",
             f"worker {worker!r}" if worker else "worker unknown"]
    if route.task:
        parts.append(f"task {route.task!r}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Request building — goal + route -> the amenity-shaped body.
# ---------------------------------------------------------------------------


class GoalShapeError(ValueError):
    """The goal cannot feed this capability (missing required input) — a typed
    400 at the route, raised BEFORE any dispatch."""


def _refs(goal: GoalSpec, *kinds: str) -> list[str]:
    return [i.ref for i in goal.inputs if i.kind.value in kinds]


def build_request_body(goal: GoalSpec, route: RouteDecision) -> dict[str, Any]:
    """The pre-normalization body for ``route.capability`` — the same keys a
    caller of the matching /ml amenity would send, derived from typed inputs.
    Raises GoalShapeError when a required input is absent."""
    cap = route.capability
    prompt = goal.objective
    texts = _refs(goal, "text")
    media = _refs(goal, "image", "audio", "video")
    file = media[0] if media else None

    def need_file(kind: str) -> str:
        if file is None:
            raise GoalShapeError(
                f"{cap} needs a {kind} input (inputs: [{{kind: '{kind}', "
                f"uri: ...}}]); none was supplied")
        return file

    if cap == "text.chat":
        body: dict[str, Any] = {"prompt": prompt}
        if texts:
            body["prompt"] = prompt + "\n\n" + "\n\n".join(texts)
    elif cap in ("text.summarize", "text.keywords"):
        body = {"text": "\n\n".join(texts) if texts else prompt}
    elif cap == "text.embed":
        body = {"texts": texts or [prompt]}
    elif cap == "text.similarity":
        if len(texts) < 2:
            raise GoalShapeError(
                "text.similarity needs >=2 text inputs (the first is compared "
                f"against the rest); got {len(texts)}")
        body = {"texts": [texts[0]], "other_texts": texts[1:]}
    elif cap == "audio.transcribe":
        body = {"file": need_file("audio")}
    elif cap == "audio.transcribe.word_timestamps":
        # The base transcription request PLUS the flag that IS this capability
        # (k98). ``route.dispatch_params`` re-asserts the same flag on the
        # dispatch kwargs; setting it here as well keeps the request body true
        # on its own, so the receipt records what was ASKED FOR rather than
        # what a later merge happened to add.
        body = {"file": need_file("audio"), "word_timestamps": True}
    elif cap == "audio.tts":
        line = "\n\n".join(texts) if texts else prompt
        if not line.strip():
            raise GoalShapeError(
                "audio.tts needs the line to speak: a text input (inputs: "
                "[{kind: 'text', text: ...}]) or a non-empty prompt")
        body = {"text": line}
        voice = next((i.ref for i in goal.inputs
                      if i.kind.value in ("audio", "video")), None)
        if voice:
            body["reference_audio"] = voice
            # ``authorized`` is NOT "the route is fine" — it is specifically
            # "k97's gate demanded a VOICE grant for this request and got one".
            # An audio input the gate did not read as a voice reference (no
            # voice/speaker/reference label, no profile ref) therefore arrives
            # unauthorized and the TTS runner REFUSES it, which is the correct
            # end: a silent fallback to the default voice, or a clone the gate
            # never cleared, would both be worse than the error (doc §12).
            granted = route.authority
            body["authorized"] = bool(
                granted is not None and granted.ok
                and any(k is AuthorityKind.VOICE for k, _s in granted.required))
        # ``seed`` is not a GoalSpec field: it rides in through ``overrides``
        # on k90c's reseed repair, exactly like image.generate's.
    elif cap == "image.understand":
        body = {"file": need_file("image"), "prompt": prompt}
    elif cap == "image.generate":
        body = {"prompt": prompt}
    elif cap == "image.transform":
        body = {"prompt": prompt, "image_path": need_file("image")}
    elif cap in ("image.depth", "image.detect", "image.classify", "image.segment"):
        body = {"image_path": need_file("image")}
    elif cap == "doc.extract":
        paths = [r for r in texts + _refs(goal, "url") if os.path.sep in r] + media
        if not paths:
            raise GoalShapeError(
                "doc.extract needs a document input carrying a server file path")
        body = {"file": paths[0]}
    elif cap == "web.fetch":
        urls = _refs(goal, "url")
        if not urls:
            raise GoalShapeError("web.fetch needs a url input")
        body = {"url": urls[0]}
    else:
        raise GoalShapeError(f"no request builder for capability {cap!r}")

    if route.model_id:
        body["model_key"] = route.model_id
    return body


# ---------------------------------------------------------------------------
# Failure classification + artifact extraction.
# ---------------------------------------------------------------------------

_WORKER_MARKERS = ("worker", "connection refused", "connection reset",
                   "unreachable", "unavailable", "no route to host",
                   "name or service not known", "bad gateway", "502", "503")
_TIMEOUT_MARKERS = ("timed out", "timeout", "deadline")


def classify_failure(exc: BaseException) -> FailureClass:
    """WHERE the execution died, from the exception the dispatch path raised.
    Deliberately string-tolerant: the dispatch surface raises requests/OS/
    runner exceptions from several layers and this must never re-crash."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, TimeoutError) or any(m in msg for m in _TIMEOUT_MARKERS):
        return FailureClass.TIMEOUT
    if isinstance(exc, ConnectionError) or any(m in msg for m in _WORKER_MARKERS):
        return FailureClass.WORKER_UNAVAILABLE
    return FailureClass.RUNNER_ERROR


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _inline_text_artifact(text: str) -> dict[str, Any]:
    digest = _sha256_bytes(text.encode("utf-8"))
    return {"kind": ArtifactKind.TEXT.value, "uri": f"inline:text/{digest[:16]}",
            "sha256": digest, "text": text}


def _inline_data_artifact(kind: ArtifactKind, data: Any) -> dict[str, Any]:
    canon = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    digest = _sha256_bytes(canon.encode("utf-8"))
    return {"kind": kind.value, "uri": f"inline:{kind.value}/{digest[:16]}",
            "sha256": digest, "data": data}


def _file_artifact(kind: ArtifactKind, path: str, **extra: Any) -> dict[str, Any]:
    return {"kind": kind.value, "uri": path, "sha256": _sha256_file(path), **extra}


def _materialize_audio(b64: str, stem: str) -> str | None:
    """Relayed wav bytes -> a real file on THIS box, under the adapter's own
    output dir. Returns the path, or None when it cannot be written.

    WHY THIS EXISTS. Synthesis runs on a WORKER, so the path the runner reports
    is a path on that worker's disk — unreadable here, and therefore unhashable:
    the artifact would carry ``sha256: null`` and any consumer that opened it
    would find nothing. The bytes already ride back inline (``TtsResult.audio[].b64``,
    the imagegen ``return_b64`` precedent), so the honest move is to write them
    where the caller can actually read them and hash what was written."""
    import base64
    try:
        from hugpy_oracle.config import tts_output_dir
        out_dir = tts_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, os.path.basename(stem) or "tts.wav")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(b64))
        return path
    except Exception as exc:  # noqa: BLE001 — a write failure is a warning, not a crash
        logger.warning("oracle runtime: could not materialize relayed audio (%s: %s)",
                       type(exc).__name__, exc)
        return None


def _result_payload(result: Any) -> dict[str, Any]:
    """Result object -> plain dict, same ladder as ml_routes._result_payload."""
    if isinstance(result, dict):
        return result
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(result, attr, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                continue
    return {"text": str(result)}


def extract_artifacts(capability: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The capability-shaped read of a result payload. An empty/blank result
    STILL yields its (empty) artifact where the shape allows — the scorecard's
    EMPTY_OUTPUT check needs to see it, not infer it from absence."""
    arts: list[dict[str, Any]] = []

    if capability == "text.embed":
        arts.append(_inline_data_artifact(ArtifactKind.EMBEDDING,
                                          payload.get("embeddings") or []))
        return arts
    if capability == "text.similarity":
        arts.append(_inline_data_artifact(ArtifactKind.JSON,
                                          {"similarities": payload.get("similarities") or []}))
        return arts
    if capability == "text.keywords":
        keys = {k: payload[k] for k in ("primary", "secondary", "combined",
                                        "hashtags", "meta_keywords")
                if payload.get(k)}
        arts.append(_inline_data_artifact(ArtifactKind.JSON, keys))
        return arts
    if capability in ("image.detect", "image.classify"):
        arts.append(_inline_data_artifact(ArtifactKind.JSON,
                                          {"items": payload.get("items") or []}))
        return arts
    if capability == "audio.tts":
        # File-backed audio. ``duration_s``/``sample_rate`` are the runner's
        # MEASURED read-back of the wav it wrote (doc invariant 11), carried onto
        # the artifact so a consumer never has to re-open the file to learn how
        # long the line came out — or worse, believe the synth's own claim.
        for clip in payload.get("audio") or ():
            if not isinstance(clip, dict):
                continue
            path = clip.get("path") or ""
            if (not path or not os.path.isfile(path)) and clip.get("b64"):
                path = _materialize_audio(
                    clip["b64"], os.path.basename(path) or "tts.wav") or path
            arts.append(_file_artifact(
                ArtifactKind.AUDIO, path,
                duration_s=clip.get("duration_s"),
                sample_rate=clip.get("sample_rate"),
                reference_used=bool(clip.get("reference_used")),
                seed=clip.get("seed")))
        return arts

    # File-backed images (generate/transform/depth/segment carry GeneratedImage
    # rows; segment/depth may also carry structured items).
    for img in payload.get("images") or ():
        if isinstance(img, dict) and img.get("path"):
            arts.append(_file_artifact(
                ArtifactKind.IMAGE, img["path"],
                width=img.get("width"), height=img.get("height")))
    if capability == "image.segment" and payload.get("items"):
        arts.append(_inline_data_artifact(ArtifactKind.JSON,
                                          {"items": payload["items"]}))
    if arts:
        return arts

    # Text-producing capabilities (chat/summarize/understand/transcribe/
    # doc.extract/web.fetch) — inline text, even when blank.
    text = payload.get("text")
    if text is None:
        text = payload.get("content") or ""
    arts.append(_inline_text_artifact(str(text)))
    return arts


# ---------------------------------------------------------------------------
# execute_route — the wrapper the receipt is made of.
# ---------------------------------------------------------------------------

_MAX_RECEIPT_VALUE = 2048  # chars per request field on the receipt


def _receipt_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe, size-bounded copy of the dispatched request for the receipt."""
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        try:
            encoded = json.dumps(v)
        except (TypeError, ValueError):
            v, encoded = repr(v), json.dumps(repr(v))
        if len(encoded) > _MAX_RECEIPT_VALUE:
            v = f"<{len(encoded)} json chars elided>"
        out[k] = v
    return out


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dispatch_kwargs(goal: GoalSpec, route: RouteDecision,
                    overrides: Mapping[str, Any] | None = None,
                    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """(body, kwargs, warnings) for ``route`` — the request as it will be sent.

    The last step is k98b's ``route.dispatch_params``
    (``catalog.capability_params``) merged onto the normalized kwargs, and it is
    LAST on purpose: those keys are CAPABILITY-DEFINING, not preferences. A
    ``audio.transcribe.word_timestamps`` run dispatched with
    ``word_timestamps=False`` because a caller passed that key would be a
    different capability wearing this one's name — the whole reason the flag is
    a separate catalog entry instead of an argument the planner must remember
    (doc §4). So the catalog WINS for its own keys, an override of a caller
    value is recorded as a receipt warning, and every other key still comes
    from the caller."""
    body = build_request_body(goal, route)
    if overrides:
        body.update(overrides)
    task = route.task or ""
    if task in ("document-extraction", "url-extraction"):
        kwargs = dict(body, task=task)
    else:
        kwargs = _normalized_kwargs(task, body)

    warnings: list[str] = []
    for key, value in (route.dispatch_params or {}).items():
        if key in kwargs and kwargs[key] != value:
            warnings.append(
                f"capability parameter {key}={value!r} overrode the requested "
                f"{key}={kwargs[key]!r} ({route.capability} is defined by it)")
        kwargs[key] = value
    return body, kwargs, warnings


#: Distinguishes "the caller did not pass registry_version at all" (read one
#: fresh, below) from "the caller passed registry_version=None because ITS OWN
#: attempt already failed" (respect that verbatim — re-attempting would let a
#: transient success paper over the very fault the caller already recorded).
_NOT_GIVEN: Any = object()


def execute_route(goal: GoalSpec, route: RouteDecision,
                  overrides: Mapping[str, Any] | None = None,
                  deadline_s: float | None = None,
                  registry_version: str | None = _NOT_GIVEN,
                  ) -> tuple[list[dict[str, Any]], ExecutionReceipt]:
    """Run ``route`` for ``goal`` through the existing dispatch machinery and
    wrap the outcome into (artifacts, ExecutionReceipt). Never raises for an
    execution failure — that is receipt data; only GoalShapeError (a request-
    shape fault, pre-dispatch) escapes. ``overrides`` are extra body fields
    merged after the shape build (k90c's reseed repair passes ``seed``).

    The dispatch runs under a DEADLINE (k101b): ``deadline_s`` when given, else
    ``sync_deadline_s(goal)``. On expiry the call ends with a typed
    ``FailureClass.TIMEOUT`` receipt naming what it was holding on, instead of
    riding the fleet's 25-minute cold-hold ladder until gunicorn drops the
    connection and the caller gets nothing. The wait is never retried: a second
    one would double the very hang this bound exists to end.

    ``registry_version`` (k105) is stamped on the receipt EVERY TIME this
    returns — success, a classified failure, or a timeout, all share the one
    construction below. Pass in the caller's already-computed
    ``catalog.registry_version()`` (the route computes it once per request) —
    including an honest ``None`` when THAT read already failed, which is
    respected as given, never silently retried. Omit the argument entirely and
    this reads one fresh (``_catalog_registry_version``, itself never raising)
    so a caller that does not know about registry versions at all (
    ``repair.execute_repair``'s second dispatch, a direct test call) still
    produces a stamped receipt."""
    if route.execution != "execute":
        raise ValueError(f"execute_route called on a {route.execution!r} route")

    body, kwargs, warnings = dispatch_kwargs(goal, route, overrides)
    task = route.task or ""
    budget = float(deadline_s) if deadline_s is not None else sync_deadline_s(goal)

    started_at = _utc_now()
    t0 = time.monotonic()
    retries = 0
    failure: FailureClass | None = None
    log_excerpt: list[str] = []
    payload: dict[str, Any] = {}

    for attempt in (0, 1):
        remaining = budget - (time.monotonic() - t0)
        try:
            if task == "document-extraction":
                result = run_bounded(lambda: _extract_document(body["file"]),
                                     remaining, "doc-extract")
            elif task == "url-extraction":
                result = run_bounded(lambda: _assess_url(body["url"]),
                                     remaining, "url-extract")
            else:
                result = run_bounded(lambda: _dispatch(kwargs), remaining,
                                     f"dispatch:{route.capability}")
        except DispatchTimeout as exc:
            # OUR deadline expired — not the worker's verdict. Say what the
            # wait was holding on and END (invariant 12); the fleet may well
            # still be loading that model, and that is exactly the thing a
            # synchronous request must stop pretending to wait for.
            failure = FailureClass.TIMEOUT
            log_excerpt.append(
                f"{exc}; holding on {_dispatch_target(route, kwargs)}. The "
                f"work may still be running on the fleet — the oracle stopped "
                f"waiting (deadline from the goal budget / "
                f"{SYNC_DEADLINE_ENV}, default "
                f"{DEFAULT_SYNC_DEADLINE_S:.0f}s)."[:500])
            break
        except Exception as exc:  # noqa: BLE001 — classified, never propagated
            failure = classify_failure(exc)
            log_excerpt.append(f"{type(exc).__name__}: {exc}"[:500])
            if failure is FailureClass.WORKER_UNAVAILABLE and attempt == 0:
                retries = 1
                warnings.append("retried once after WORKER_UNAVAILABLE")
                continue
            break
        failure = None
        payload = _result_payload(result)
        if payload.get("ok") is False or payload.get("error"):
            failure = FailureClass.RUNNER_ERROR
            log_excerpt.append(str(payload.get("error") or "runner returned not-ok")[:500])
        break

    duration = time.monotonic() - t0
    artifacts = extract_artifacts(route.capability, payload) if failure is None else []
    model_id = route.model_id or payload.get("model_key") or "(dispatch-default)"

    if failure is not None:
        # A classified failure is receipt data AND a log line. Receipting it
        # silently is how a fleet stall stayed invisible until a caller sat on
        # a dead connection for three minutes.
        logger.warning(
            "oracle execute_route: %s failed after %.1fs — failure_class=%s "
            "model=%s worker=%s retries=%d: %s",
            route.capability, duration, failure.value, model_id,
            route.placement, retries,
            (log_excerpt[-1] if log_excerpt else "(no detail recorded)")[:300])

    version = registry_version if registry_version is not _NOT_GIVEN \
        else _catalog_registry_version()

    receipt = ExecutionReceipt(
        request=ExecutionReceipt.normalize_request(_receipt_request(kwargs)),
        capability=route.capability,
        model_id=model_id,
        worker=None if route.placement == "auto" else route.placement,
        started_at=started_at,
        ended_at=_utc_now(),
        duration_s=round(duration, 4),
        retries=retries,
        failure=failure,
        artifacts=tuple(
            ArtifactRef(kind=ArtifactKind(a["kind"]), uri=a["uri"],
                        sha256=a.get("sha256"))
            for a in artifacts),
        warnings=tuple(warnings),
        log_excerpt=tuple(log_excerpt),
        registry_version=version,
    )
    try:  # k113a: every execution is evidence for the next selection
        from hugpy_oracle import selection as _selection
        _selection.note_execution(route.capability, model_id, ok=failure is None,
                                  latency_s=duration,
                                  failure=getattr(failure, "value", None) if failure else None)
    except Exception:  # noqa: BLE001
        pass
    return artifacts, receipt


__all__ = ["DispatchTimeout", "GoalShapeError", "SYNC_DEADLINE_ENV",
           "build_request_body", "classify_failure", "dispatch_kwargs",
           "execute_route", "extract_artifacts", "run_bounded",
           "sync_deadline_s"]
