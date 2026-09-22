"""k114 — HTTP for the script-first generation pipeline.

Thin on purpose. Every decision this blueprint could make has already been made
by a typed artifact in ``oracle/script_first.py``: the routes parse a body,
call one method, and translate exactly one exception type
(``ScriptFirstRefused``) onto its own declared status code. There is no second
copy of the lock rules, no second validator, and no place here where a refusal
could be softened into a 200 with a warning.

Mounted at ``/video/script/...`` deliberately: ``flask_app/app/video_auth.py``
gates ``^/(video|movie)(/|$)`` with the member-or-operator-session-or-share
credential check, so the script-first surface inherits the same gate the rest
of the video section already has instead of growing a second one.

Registration is one line in ``routes/__init__.py`` — ``abstract_flask``'s
``_discover_blueprints`` registers every ``*_bp`` attribute of that module with
no url_prefix, which is why the paths are spelled in full below.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

script_first_bp = Blueprint("script_first_bp", __name__)

#: The single mount point, so a route and its documentation cannot drift.
BASE: str = "/video/script"


def _sf():
    """The oracle module, imported LAZILY.

    ``oracle.script_first`` pulls in the catalog on its live seams, and the
    catalog builds the model registry — a two-second import that must not run
    at blueprint-import time, which happens during app boot for every process
    including ones that will never serve a script route."""
    from hugpy_oracle import script_first
    return script_first


def _body() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        return {"__non_object__": payload}
    return dict(payload)


def _guard(fn: Callable[..., Any]):
    """Run ``fn`` and turn its typed refusal into its own status code.

    ``ScriptFirstRefused`` already carries the code, every error string and the
    machine-readable detail; this hands that back verbatim. An UNEXPECTED
    exception is a 500 with the type and message — never a fabricated success,
    and never a bare "internal error" that costs an operator a log dig."""
    def wrapped(*args: Any, **kwargs: Any):
        script_first = _sf()
        try:
            return fn(script_first, *args, **kwargs)
        except script_first.ScriptFirstRefused as exc:
            if exc.code != "GENERATE_GAP":
                logger.info("script_first: %s refused (%s): %s",
                            request.path, exc.code, exc.message)
            _journal_refusal(script_first, kwargs.get("run_id"), exc)
            return jsonify(exc.to_dict()), exc.http_status
        except Exception as exc:                   # noqa: BLE001
            logger.warning("script_first: %s failed: %s", request.path, exc,
                           exc_info=True)
            return jsonify({"ok": False, "code": "UNEXPECTED",
                            "message": f"{type(exc).__name__}: {exc}",
                            "errors": [str(exc)], "detail": {}}), 500
    wrapped.__name__ = fn.__name__
    return wrapped


def _journal_refusal(script_first, run_id, exc) -> None:
    """Persist the refusal on the run it was about, best effort.

    A refusal is the most useful thing on the screen and the least durable: a
    422's validator output is gone the moment the page reloads. Journalling it
    means the run's own GET carries the structured form (code, every error, and
    an authoring gap's raw reply) instead of the operator having to reproduce
    the failure to read it again. Never allowed to change the response."""
    if not run_id or exc.code == "RUN_NOT_FOUND":
        return
    try:
        script_first.ScriptFirstRun.load(run_id).record_refusal(exc)
    except Exception as write_exc:                 # noqa: BLE001
        logger.debug("script_first: refusal not journalled: %s", write_exc)


def _run(script_first, run_id: str):
    return script_first.ScriptFirstRun.load(run_id)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@script_first_bp.route(f"{BASE}/runs", methods=["POST"])
@_guard
def script_create_run(script_first):
    """Doc Stage 4 — capture the immutable snapshot and START the run.

    Body: ``{deliverable, requirements, sources: [{prompt_id, text, hash?,
    persisted_at?}], references: {operator|identity|voice|acquisition|
    exclusions}, settings, raw_request_ref?}``.

    ``sources`` may also name a PROMOTED source by ``source_id`` alone; the
    text is read from the persisted source file rather than trusted from the
    request, so a caller cannot promote one prompt and snapshot another under
    its id."""
    body = _body()
    sources = []
    for raw in (body.get("sources") or []):
        if isinstance(raw, Mapping) and not (raw.get("text")
                                             or raw.get("prompt")):
            sid = str(raw.get("source_id") or raw.get("prompt_id") or "")
            record = script_first.load_promoted_source(sid) if sid else None
            if record:
                sources.append({"prompt_id": record["source_id"],
                                "text": record["text"],
                                "hash": record["digest"],
                                "persisted_at": record.get("promoted_at"),
                                "origin": "promoted"})
                continue
        sources.append(raw)
    run = script_first.ScriptFirstRun.create(
        deliverable=body.get("deliverable") or "",
        raw_request_ref=body.get("raw_request_ref") or "",
        sources=sources,
        requirements=body.get("requirements") or "",
        references=body.get("references") or {},
        settings=body.get("settings") or {})
    return jsonify({"ok": True, "run_id": run.run_id, "run": run.state}), 201


@script_first_bp.route(f"{BASE}/runs", methods=["GET"])
@_guard
def script_list_runs(script_first):
    runs = script_first.ScriptFirstRun.list_runs()
    return jsonify({"ok": True, "count": len(runs), "runs": runs,
                    "promoted_sources": script_first.list_promoted_sources()})


@script_first_bp.route(f"{BASE}/runs/<run_id>", methods=["GET"])
@_guard
def script_get_run(script_first, run_id: str):
    """The whole run: snapshot, every artifact digest, lock status, per-segment
    provenance, attempts and the honest limitations list."""
    return jsonify({"ok": True, "run": _run(script_first, run_id).state})


# ---------------------------------------------------------------------------
# Phase 1 — pre-production
# ---------------------------------------------------------------------------


@script_first_bp.route(f"{BASE}/runs/<run_id>/plot", methods=["POST"])
@_guard
def script_author_plot(script_first, run_id: str):
    """Author the plot with the LIVE text route. An ``AuthoringGap`` is a 422
    carrying the validator errors verbatim plus the raw reply — never a
    coerced artifact."""
    body = _body()
    run = _run(script_first, run_id)
    entry = run.author("plot", input_text=body.get("input_text") or "",
                       mode=body.get("mode"),
                       deadline_s=body.get("deadline_s"))
    return jsonify({"ok": True, "artifact": entry, "run": run.state})


@script_first_bp.route(f"{BASE}/runs/<run_id>/plot", methods=["PUT"])
@_guard
def script_put_plot(script_first, run_id: str):
    """Operator-edited plot JSON, through the SAME ``PlotSpec`` constructor the
    model's reply goes through. Every problem at once, in ``errors``."""
    run = _run(script_first, run_id)
    entry = run.put_artifact("plot", _body())
    return jsonify({"ok": True, "artifact": entry, "run": run.state})


@script_first_bp.route(f"{BASE}/runs/<run_id>/screenplay", methods=["POST"])
@_guard
def script_author_screenplay(script_first, run_id: str):
    run = _run(script_first, run_id)
    entry = run.author("screenplay", deadline_s=_body().get("deadline_s"))
    return jsonify({"ok": True, "artifact": entry, "run": run.state})


@script_first_bp.route(f"{BASE}/runs/<run_id>/screenplay", methods=["PUT"])
@_guard
def script_put_screenplay(script_first, run_id: str):
    run = _run(script_first, run_id)
    entry = run.put_artifact("screenplay", _body())
    return jsonify({"ok": True, "artifact": entry, "run": run.state})


@script_first_bp.route(f"{BASE}/runs/<run_id>/audio_master", methods=["PUT"])
@_guard
def script_put_audio_master(script_first, run_id: str):
    """The Stage 8 artifact, supplied. There is no POST twin: this fleet cannot
    synthesize an ``AudioMaster`` (``audio.tts`` is ineligible), and a
    manufactured one with fabricated track refs would make every downstream
    shot window a lie that inspection cannot catch."""
    run = _run(script_first, run_id)
    entry = run.put_artifact("audio_master", _body())
    return jsonify({"ok": True, "artifact": entry, "run": run.state})


@script_first_bp.route(f"{BASE}/runs/<run_id>/preproduction", methods=["POST"])
@_guard
def script_build_preproduction(script_first, run_id: str):
    """Derive the continuity bible and the shot plan (doc Stages 7 + 9).

    Separate from the lock so both read-only viewers have something to show on
    a run whose lock will honestly refuse for want of an audio master."""
    run = _run(script_first, run_id)
    built = run.build_preproduction()
    return jsonify({"ok": True, "artifacts": built, "run": run.state})


@script_first_bp.route(f"{BASE}/runs/<run_id>/lock", methods=["POST"])
@_guard
def script_lock(script_first, run_id: str):
    """Doc Stage 11 — version and lock the whole production, or refuse."""
    body = _body()
    run = _run(script_first, run_id)
    lock = run.lock_run(audio_master=body.get("audio_master"),
                        identity_refs=body.get("identity_refs"),
                        locked_at=body.get("locked_at"))
    return jsonify({"ok": True, "lock": lock, "run": run.state})


@script_first_bp.route(f"{BASE}/runs/<run_id>/revise", methods=["POST"])
@_guard
def script_revise(script_first, run_id: str):
    """Doc Stage 10 — the ONLY post-lock path. ``reason`` is mandatory and a
    revision without one cannot even be constructed (k104's refusal)."""
    body = _body()
    run = _run(script_first, run_id)
    lock = run.revise(body.get("reason") or "",
                      artifacts=body.get("artifacts") or {},
                      changes=body.get("changes") or {})
    return jsonify({"ok": True, "lock": lock, "run": run.state})


# ---------------------------------------------------------------------------
# Phase 2 — production
# ---------------------------------------------------------------------------


@script_first_bp.route(f"{BASE}/runs/<run_id>/segments", methods=["POST"])
@_guard
def script_compile_segments(script_first, run_id: str):
    """Doc Stage 14 — compile every segment from the LOCK, as siblings.

    The response carries each ``SegmentSpec`` with its prompt, its parents
    (lock-side digests ONLY), its seed and the sibling shape, plus the
    ``PlanGraph`` validation report exactly as the static validator returned
    it — errors included."""
    body = _body()
    run = _run(script_first, run_id)
    entry = run.compile(tone=body.get("tone"), seed_salt=body.get("seed_salt"),
                        negative_prompt=body.get("negative_prompt"))
    return jsonify({"ok": True, "segments": entry, "run": run.state})


@script_first_bp.route(f"{BASE}/runs/<run_id>/segments", methods=["GET"])
@_guard
def script_get_segments(script_first, run_id: str):
    run = _run(script_first, run_id)
    return jsonify({"ok": True, "segments": run.state.get("segments"),
                    "attempts": run.attempts()})


@script_first_bp.route(f"{BASE}/runs/<run_id>/segments/<segment_id>/generate",
                       methods=["POST"])
@_guard
def script_generate_segment(script_first, run_id: str, segment_id: str):
    """ONE attempt at ONE segment, from its FROZEN spec.

    A regeneration is attempt N+1 against the same spec at a new seed; the
    receipt records the sibling digests before and after, so "regenerating 1 did
    not touch 2" is checkable rather than promised. A ``clip`` request on this
    fleet comes back as an attempt carrying a typed GAP (``video.*`` resolves
    deferred by design — clips execute on a GPU worker through the studio job
    pipeline), which is an outcome, not a failure."""
    body = _body()
    run = _run(script_first, run_id)
    attempt = run.generate_segment(segment_id,
                                   kind=(body.get("kind") or "keyframe"))
    return jsonify({"ok": bool(attempt.get("ok")), "attempt": attempt,
                    "run": run.state})


# ---------------------------------------------------------------------------
# Phase 3 — promotion
# ---------------------------------------------------------------------------


@script_first_bp.route(f"{BASE}/runs/<run_id>/promote", methods=["POST"])
@_guard
def script_promote(script_first, run_id: str):
    """Accept an output as a persisted source for a FUTURE run.

    The promoted digest goes into THIS run's ledger first, so feeding it back
    here is refused by k104's own code. The response carries that refusal
    verbatim in ``refused_here`` — the confirmation text a UI shows is the
    actual refusal, not a paraphrase of the policy."""
    body = _body()
    run = _run(script_first, run_id)
    record = run.promote(segment_id=body.get("segment_id"),
                         attempt=body.get("attempt"),
                         text=body.get("text") or "",
                         note=body.get("note") or "",
                         source_id=body.get("source_id"))
    return jsonify({"ok": True, "source": record,
                    "next": "create a NEW run with this source_id; it cannot "
                            "enter the run that produced it",
                    "run": run.state}), 201


@script_first_bp.route(f"{BASE}/sources", methods=["GET"])
@_guard
def script_sources(script_first):
    sources = script_first.list_promoted_sources()
    return jsonify({"ok": True, "count": len(sources), "sources": sources})


__all__ = ["script_first_bp", "BASE"]
