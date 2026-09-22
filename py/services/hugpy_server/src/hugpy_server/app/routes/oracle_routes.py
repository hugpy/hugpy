"""Oracle routes: GET /oracle/capabilities (k90a) + POST /oracle/route (k90b).

THIN adapters in the comms_routes style: parse HTTP, call the oracle package,
jsonify. All behavior lives in ``abstract_hugpy_dev.oracle`` (catalog, router,
runtime, scorecard); if a route here grows logic, it's in the wrong file. The
oracle imports are lazy inside the handlers so app boot never pays for (or
breaks on) a registry read.

POST /oracle/route is the route-to-best-result core. JSON body::

    {"prompt": str,                       # required unless capability is set
     "inputs": [{"kind": "text|image|video|audio|url",
                 "uri": str | "b64": str | "text": str, "label"?: str}],
     "capability"?: "audio.transcribe",   # explicit wins over inference
     "quality"?: "preview|balanced|best",
     "model_id"?: str,
     "identity_profile"?: "<slug>",       # k97 sugar: folded into an input ref
                          # "identity_profile:<slug>" — the same canonical form
                          # the video routes accept. Either spelling is
                          # accepted ("demo" or "identity_profile:demo") and
                          # normalizes to ONE prefix (k101b: an already-
                          # prefixed value used to be prefixed AGAIN, and the
                          # refusal then named the subject
                          # "identity_profile:identity_profile")
     "voice_profile"?: "<slug>",          # the same sugar for a voice subject
     "rights"?: {"authorizations": [{"kind": "likeness|voice|…",
                                     "subject": "identity_profile:<slug>",
                                     "evidence": str, "scope"?: str,
                                     "granted_by"?: str, "granted_at"?: str}],
                 "denied"?: [str], "notes"?: str},   # k97 RightsManifest —
                          # the consent recorded ON a referenced identity
                          # profile is folded in automatically; this is what the
                          # caller adds ON TOP
     "planner_mode"?: "local_only|frontier",   # default local_only (truthful)
     "disclosure_scope"?: str,                 # default "operator"
     "evaluate"?: bool,   # k90c judge; default true for image.generate/
                          # image.transform/text.summarize, false otherwise
     "repair"?: bool}     # k90c one bounded repair; default true when evaluate
                          # ran or a technical check failed

A repaired run answers with the SECOND attempt's artifacts/receipt/scorecard
plus additive keys: ``repair`` (the RepairDecision) and ``receipts`` (both
attempts, in order).

Responses (scorecard MANDATORY on all of them, operator ruling 2026-08-05;
``planner_mode`` echoed on EVERY one of them, invariant 8 — a caller must never
have to guess whether a frontier planner participated; ``registry_version``
(k105) is ALSO echoed top-level on every one of them, computed once per
request via ``catalog.registry_version()`` — ``null`` on a catalog fault,
never a crash — and threaded into every receipt this route builds, including
the refusal receipt):
  200 executed  -> {ok, goal, route, artifacts, receipt, scorecard,
                   registry_version}
  504 timeout   -> k101b bounded wait: {ok:false, error, execution:"timeout",
                   reason (what the dispatch was holding on), goal, route,
                   receipt (FailureClass.TIMEOUT), scorecard (RepairCode.
                   TIMEOUT)}. The oracle stopped waiting; the fleet may still
                   be loading the model. Nothing is faked and nothing hangs.
  202 deferred  -> video.*: {ok, routed, execution:"deferred", reason,
                   binding/alternatives, scorecard}
  422 gap       -> typed CAPABILITY_GAP: {ok:false, goal, route, scorecard}
  403 refused   -> k97 authority gate: {ok:false, error, missing_authority:
                   [{kind, subject}], goal, route, receipt (FailureClass.
                   REFUSED), scorecard (RepairCode.SOURCE_AUTHORITY_MISSING)}
  400 malformed -> {ok:false, error} (typed message, never a traceback)
"""
from __future__ import annotations

import dataclasses
import logging
import os
import time

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

oracle_bp = Blueprint("oracle_bp", __name__)


@oracle_bp.route("/oracle/capabilities", methods=["GET"])
def oracle_capabilities():
    """The unified capability catalog. Optional ``?capability=<name>`` filter
    returns just that view (404 with the known names when it doesn't exist).

    ``registry_version`` (k105) rides top-level on every response, next to
    ``count`` — read off a view the call already fetched (every view out of
    one ``list_capabilities()`` call is stamped with the SAME snapshot digest,
    catalog.py's ``_probe_views``), never a second catalog read. ``null`` only
    when the catalog itself could not be answered."""
    from hugpy_oracle import catalog

    name = (request.args.get("capability") or "").strip()
    try:
        if name:
            view = catalog.get_capability(name)
            if view is None:
                known = catalog.list_capabilities()
                return jsonify({
                    "ok": False,
                    "error": f"unknown capability {name!r}",
                    "known": sorted(v.name for v in known),
                    "registry_version": known[0].registry_version if known else None,
                }), 404
            return jsonify({"ok": True, "count": 1,
                            "capabilities": [view.to_dict()],
                            "registry_version": view.registry_version})
        views = catalog.list_capabilities()
        return jsonify({"ok": True, "count": len(views),
                        "capabilities": [v.to_dict() for v in views],
                        "registry_version": views[0].registry_version if views else None})
    except Exception as exc:  # noqa: BLE001 — a catalog fault answers honestly, not 500-blank
        logger.exception("GET /oracle/capabilities failed")
        return jsonify({"ok": False, "error": str(exc),
                        "registry_version": None}), 500


# ---------------------------------------------------------------------------
# POST /oracle/route (k90b)
# ---------------------------------------------------------------------------

# b64 inputs are materialized to files so downstream (dispatch builders,
# hashing, decode checks) sees one shape: a server path.
_B64_EXT = {"image": ".png", "audio": ".wav", "video": ".mp4", "text": ".txt"}


def _uploads_home() -> str:
    try:
        from hugpy_platform.constants import UPLOADS_HOME
        return UPLOADS_HOME
    except Exception:  # noqa: BLE001 — same fallback as ml_routes
        return os.path.join(os.environ.get("DEFAULT_ROOT", "/tmp"), "uploads")


def _materialize_b64(b64: str, kind: str) -> str:
    import base64
    import uuid
    raw = base64.b64decode(b64.split(",", 1)[-1] if b64.startswith("data:") else b64)
    home = _uploads_home()
    os.makedirs(home, exist_ok=True)
    dest = os.path.join(home,
                        f"oracle_{uuid.uuid4().hex[:8]}{_B64_EXT.get(kind, '.bin')}")
    with open(dest, "wb") as fh:
        fh.write(raw)
    return dest


def _parse_inputs(raw: list) -> list:
    """Body inputs -> typed InputRefs (b64 payloads land as upload files)."""
    from hugpy_oracle.contracts import InputKind, InputRef
    refs = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"inputs[{i}] must be an object")
        try:
            kind = InputKind(item.get("kind"))
        except ValueError:
            raise ValueError(
                f"inputs[{i}].kind must be one of "
                f"{[k.value for k in InputKind]}, got {item.get('kind')!r}")
        ref = item.get("text") or item.get("uri") or item.get("ref")
        if not ref and item.get("b64"):
            ref = _materialize_b64(item["b64"], kind.value)
        if not ref:
            raise ValueError(f"inputs[{i}] needs one of uri | b64 | text")
        refs.append(InputRef(kind=kind, ref=ref, label=item.get("label", "")))
    return refs


def _store_authorizations(prompt: str, inputs: list) -> tuple:
    """Consent RECORDED ON THE IDENTITY ITSELF, folded into the request's rights.

    An identity profile can carry an ``authorization`` block (the release the
    operator filed once, with evidence). That IS an authorization — it would be
    absurd to make the caller re-state it on every request — so any
    ``identity_profile:<slug>`` reference in this request contributes its stored
    grant, and the body's own ``rights`` ride ON TOP. It only ever ADDS grants
    the store can evidence: an absent/unevidenced block contributes nothing, and
    a store fault contributes nothing (fail CLOSED — a telemetry problem must
    never read as consent)."""
    from hugpy_oracle import authority as oracle_authority
    from hugpy_oracle.contracts import Authorization, AuthorityKind

    texts = [prompt]
    for ref in inputs:
        texts.extend((ref.ref, ref.label))
    out = []
    for prefix, subject in oracle_authority.find_subject_refs(*texts):
        if prefix != "identity_profile":
            continue
        slug = subject.split(":", 1)[1]
        try:
            from hugpy_video.intel import identity_profiles
            profile = identity_profiles.get_profile(slug)
        except Exception:  # noqa: BLE001 — no profile is "no grant", never a 500
            logger.exception("oracle authority: identity profile read failed")
            continue
        if profile is None:
            continue
        for kind, row in (profile.get("authorization") or {}).items():
            evidence = str((row or {}).get("evidence") or "").strip()
            if not (row or {}).get("granted") or not evidence:
                continue
            try:
                out.append(Authorization(
                    kind=AuthorityKind(kind), subject=f"identity_profile:{slug}",
                    scope="recorded on the identity profile", evidence=evidence,
                    granted_by=f"identity_profile:{slug}",
                    granted_at=str(row.get("granted_at") or "")))
            except ValueError:  # an unknown kind in the store is not a grant
                continue
    return tuple(out)


# The canonical subject-reference prefixes (oracle.authority._SUBJECT_REF knows
# the same two). A body field may spell its value either way.
_SUBJECT_PREFIXES = ("identity_profile", "voice_profile")


def _subject_ref(raw: str, default_prefix: str) -> str:
    """A body's profile field -> ``<prefix>:<slug>``, prefixed EXACTLY once.

    Accepts a bare slug (``demo``), the canonical form
    (``identity_profile:demo``) and — defensively — an accidentally doubled
    prefix. Before k101b this route pasted its prefix onto whatever arrived, so
    ``{"identity_profile": "identity_profile:demo"}`` became
    ``identity_profile:identity_profile:demo``; the authority scanner then read
    the subject as ``identity_profile:identity_profile`` and the refusal named
    a person who does not exist while the real slug was lost.

    A value that spells a DIFFERENT known prefix keeps the prefix it spells: a
    ``voice_profile:`` ref passed in the identity field is a voice subject, and
    silently re-labelling it would be the same class of lie."""
    text = str(raw or "").strip()
    prefix = default_prefix
    while True:
        head, sep, tail = text.partition(":")
        if sep and head.strip() in _SUBJECT_PREFIXES and tail.strip():
            prefix, text = head.strip(), tail.strip()
            continue
        break
    return f"{prefix}:{text}" if text else ""


def _planner_mode(body: dict) -> str:
    """The truthful planner mode for THIS request, echoed on every response
    (invariant 8). Read from the body before the GoalSpec exists so even the
    early malformed/gap answers carry it. This GATES rather than echoes
    (k113; POLICY-rights-consent-disclosure §3.1): ``frontier`` is answered
    only when the body asked for it AND the fleet has a frontier wired in
    (``plan.frontier_enabled``, env ``HUGPY_FRONTIER_ENABLED``); an unknown
    value, or a frontier request on a frontier-disabled fleet, is reported as
    the honest ``local_only`` — a response must never imply A participated.
    The capability-level half of the gate (a ``frontier.*`` capability under
    ``local_only``) is ``authority.check``, which ``resolve_route`` runs."""
    from hugpy_oracle.plan import effective_planner_mode
    return effective_planner_mode(body.get("planner_mode")).value


def _safe_registry_version() -> str | None:
    """``catalog.registry_version()`` (k105), computed ONCE per request and
    threaded into the execution receipt and the refusal receipt below, plus
    echoed top-level on every response this route answers with. A catalog
    fault must not crash the route — the response still goes out, carrying
    ``registry_version: null`` rather than a guess or a 500."""
    from hugpy_oracle import catalog
    try:
        return catalog.registry_version()
    except Exception:  # noqa: BLE001 — the route answers regardless
        logger.exception("POST /oracle/route: catalog.registry_version() failed")
        return None


@oracle_bp.route("/oracle/route", methods=["POST"])
def oracle_route():
    """Route-to-best-result: infer/accept a capability, resolve the best
    eligible model via the k90a catalog, execute through the existing dispatch,
    and answer with artifact(s) + ExecutionReceipt + a mandatory Scorecard."""
    from hugpy_oracle import router as oracle_router
    from hugpy_oracle import runtime as oracle_runtime
    from hugpy_oracle import scorecard as oracle_scorecard
    from hugpy_oracle import authority as oracle_authority
    from hugpy_oracle.catalog import LEGACY_TASK_CAPABILITY, LEGACY_TASK_EXCLUDED
    from hugpy_oracle.contracts import (
        BudgetHints,
        FailureClass,
        GoalSpec,
        InputKind,
        InputRef,
        PlannerMode,
        QualityProfile,
        RightsManifest,
    )

    body = request.get_json(silent=True) or {}
    if not body and request.files:
        # Multipart one-shot: reuse the /ml upload helper; the file part becomes
        # an input whose kind the caller names via form field `kind` (default
        # image). Everything else rides in form fields.
        from hugpy_server.app.routes.ml_routes import _save_multipart_upload
        saved = _save_multipart_upload()
        if saved is not None:
            body = dict(request.form.to_dict())
            existing = body.get("inputs")
            body["inputs"] = list(existing) if isinstance(existing, list) else []
            body["inputs"].append({"kind": request.form.get("kind", "image"),
                                   "uri": saved})

    # Computed ONCE per request (k105) — reused on the receipt(s) below and
    # echoed top-level on every response this route answers with, success or
    # not. A catalog fault answers None, never a crash and never recomputed.
    registry_version = _safe_registry_version()

    prompt = (body.get("prompt") or "").strip()
    capability = (body.get("capability") or "").strip() or None

    # A bare legacy task string is folded to its namespaced capability; a task
    # string mapped to NEITHER table is the drift case the k90a handoff names:
    # it must come back as the typed CAPABILITY_GAP shape, never a KeyError.
    if capability and "." not in capability:
        mapped = LEGACY_TASK_CAPABILITY.get(capability)
        if mapped is None:
            reason = LEGACY_TASK_EXCLUDED.get(
                capability,
                f"task string {capability!r} is mapped to no capability "
                f"(oracle catalog: unmapped task)")
            route = oracle_router.RouteDecision(
                capability=f"unmapped.{capability}", execution="gap",
                reasons=(reason,))
            return jsonify({
                "ok": False, "error": "capability gap",
                "planner_mode": _planner_mode(body),
                "route": route.to_dict(),
                "scorecard": oracle_scorecard.build_gap_scorecard(route).to_dict(),
                "registry_version": registry_version,
            }), 422
        capability = mapped

    if not prompt and not capability:
        return jsonify({"ok": False,
                        "planner_mode": _planner_mode(body),
                        "error": "body needs at least one of prompt | capability",
                        "registry_version": registry_version}), 400

    try:
        inputs = _parse_inputs(body.get("inputs") or [])
        # Sugar: the studio/enqueue bodies already say ``identity_profile:
        # "<slug>"``. Fold it (and its voice twin) into the canonical
        # ``<prefix>:<slug>`` reference — normalized by ``_subject_ref`` so the
        # authority gate sees one shape and only one, whichever spelling the
        # caller sent.
        for field, label in (("identity_profile", "identity"),
                             ("voice_profile", "voice")):
            ref = _subject_ref(body.get(field), field)
            if ref:
                inputs.append(InputRef(kind=InputKind.TEXT, ref=ref,
                                       label=label))
        rights_body = body.get("rights")
        rights = (RightsManifest.from_dict(rights_body)
                  if isinstance(rights_body, dict) else None)
        stored = _store_authorizations(prompt, inputs)
        if stored:
            base = rights or RightsManifest()
            rights = RightsManifest(
                authorizations=stored + base.authorizations,
                denied=base.denied, notes=base.notes)
        goal = GoalSpec(
            objective=prompt or f"run {capability}",
            raw_prompt=prompt or f"run {capability}",
            inputs=tuple(inputs),
            capability=capability,
            quality=QualityProfile(body.get("quality") or "balanced"),
            # The budget hint is load-bearing from k101b on: it is the first
            # source of this request's synchronous deadline (runtime.
            # sync_deadline_s), so a caller who says "max_seconds: 20" is
            # answered in ~20s instead of waiting out the fleet's cold hold.
            budget=BudgetHints.from_dict(
                body.get("budget") if isinstance(body.get("budget"), dict)
                else {}),
            planner_mode=PlannerMode(body.get("planner_mode") or "local_only"),
            rights=rights,
            disclosure_scope=(body.get("disclosure_scope") or "operator"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"ok": False, "planner_mode": _planner_mode(body),
                        "error": str(exc), "registry_version": registry_version}), 400

    try:
        route = oracle_router.resolve_route(
            goal, requested_model=body.get("model_id") or None)
    except oracle_router.RouteRefusal as exc:
        return jsonify({"ok": False, "planner_mode": goal.planner_mode.value,
                        "error": str(exc), "registry_version": registry_version}), 400
    except Exception as exc:  # noqa: BLE001 — a routing fault answers honestly
        logger.exception("POST /oracle/route: routing failed")
        return jsonify({"ok": False, "planner_mode": goal.planner_mode.value,
                        "error": f"{type(exc).__name__}: {exc}",
                        "registry_version": registry_version}), 500

    if route.execution == "refused":
        # k97 authority gate. Same shape as the other typed refusals ({ok:false,
        # error}) plus the machine-readable ``missing_authority`` list, so the
        # operator is told EXACTLY which release to produce — and a receipt +
        # scorecard, because a refusal is an outcome with evidence, not silence.
        decision = route.authority
        return jsonify({
            "ok": False, "error": decision.reason,
            "planner_mode": goal.planner_mode.value,
            "missing_authority": [{"kind": k.value, "subject": s}
                                  for k, s in decision.missing],
            "goal": goal.to_dict(), "route": route.to_dict(),
            "receipt": oracle_authority.refusal_receipt(
                goal, route.capability, decision,
                registry_version=registry_version).to_dict(),
            "scorecard": oracle_authority.refusal_scorecard(decision).to_dict(),
            "registry_version": registry_version,
        }), 403

    if route.execution == "gap":
        return jsonify({
            "ok": False, "error": "capability gap",
            "planner_mode": goal.planner_mode.value,
            "goal": goal.to_dict(), "route": route.to_dict(),
            "scorecard": oracle_scorecard.build_gap_scorecard(route).to_dict(),
            "registry_version": registry_version,
        }), 422

    if route.execution == "deferred":
        return jsonify({
            "ok": True,
            "routed": route.capability,
            "execution": "deferred",
            "planner_mode": goal.planner_mode.value,
            "reason": "; ".join(route.reasons) or
                      "video capabilities execute through the studio job pipeline",
            "binding": {"model_id": route.model_id,
                        "model_ids": list(route.model_ids)},
            "alternatives": route.alternatives,
            "goal": goal.to_dict(), "route": route.to_dict(),
            "scorecard": oracle_scorecard.build_deferred_scorecard(route).to_dict(),
            "registry_version": registry_version,
        }), 202

    budget_s = oracle_runtime.sync_deadline_s(goal)
    started = time.monotonic()

    def _left(floor: float = 5.0) -> float:
        """What is left of THIS request's budget (never below ``floor``, so a
        late stage still gets a real chance to answer). The whole request is
        bounded: dispatch, then judge, then the one repair attempt."""
        return max(budget_s - (time.monotonic() - started), floor)

    # The API-key-bound worker pool is the HTTP adapter's business (the oracle's
    # ``normalize_kwargs`` no longer reads API keys): resolve it here and hand it
    # to the dispatch body as an override, where it wins over the env default.
    pool_override: dict = {}
    try:
        from hugpy_server.app.functions.chat.streaming import _resolve_request_pool
        eff_pool = _resolve_request_pool(body.get("pool"))
    except Exception:  # noqa: BLE001 — a pool lookup fault never blocks routing
        eff_pool = (body.get("pool") or "").strip() or None
    if eff_pool:
        pool_override["pool"] = eff_pool

    try:
        artifacts, receipt = oracle_runtime.execute_route(
            goal, route, overrides=pool_override or None,
            registry_version=registry_version)
    except oracle_runtime.GoalShapeError as exc:
        return jsonify({"ok": False, "planner_mode": goal.planner_mode.value,
                        "error": str(exc), "registry_version": registry_version}), 400
    except Exception as exc:  # noqa: BLE001 — typed, never a traceback
        logger.exception("POST /oracle/route: execution wrapper failed")
        return jsonify({"ok": False, "planner_mode": goal.planner_mode.value,
                        "error": f"{type(exc).__name__}: {exc}",
                        "registry_version": registry_version}), 500

    if receipt.failure is FailureClass.TIMEOUT:
        # The bounded wait expired (k101b). Answer with the same typed shape as
        # every other failure and STOP — no judge, no repair: both would
        # dispatch again into the same stall and the caller is already at the
        # deadline. 504, because the honest reading is "the gateway gave up on
        # the upstream", not "the request was bad".
        reason = "; ".join(receipt.log_excerpt) or (
            f"the dispatch did not answer within {budget_s:.0f}s")
        logger.warning("POST /oracle/route: %s timed out after %.1fs (%s)",
                       route.capability, receipt.duration_s, reason[:300])
        return jsonify({
            "ok": False, "error": reason,
            "execution": "timeout",
            "reason": reason,
            "planner_mode": goal.planner_mode.value,
            "goal": goal.to_dict(), "route": route.to_dict(),
            "artifacts": artifacts,
            "receipt": receipt.to_dict(),
            "scorecard": oracle_scorecard.build_technical_scorecard(
                goal, route, artifacts, receipt).to_dict(),
            "registry_version": registry_version,
        }), 504

    # k90c: judge evaluation + one bounded repair. Defaults per the operator
    # ruling — evaluate on for the judged capabilities, repair on when a judge
    # ran or a technical check failed; explicit booleans in the body win.
    from hugpy_oracle import evaluation as oracle_evaluation
    from hugpy_oracle import repair as oracle_repair

    evaluate_on = body.get("evaluate") if isinstance(body.get("evaluate"), bool) \
        else route.capability in oracle_evaluation.DEFAULT_EVALUATED

    card = oracle_scorecard.build_technical_scorecard(goal, route, artifacts, receipt)
    technical_failed = not card.hard_pass
    if evaluate_on:
        # The judge dispatches through the SAME fleet, so it gets the same
        # bound — a stalled judge must not turn a finished answer into a hung
        # request. On expiry the deterministic card stands and says so.
        try:
            card = oracle_runtime.run_bounded(
                lambda: oracle_evaluation.evaluate(
                    goal, route, artifacts, receipt, card),
                _left(), f"judge:{route.capability}")
        except oracle_runtime.DispatchTimeout as exc:
            logger.warning("POST /oracle/route: judge for %s abandoned: %s",
                           route.capability, exc)
            card = dataclasses.replace(card, diagnosis="; ".join(
                p for p in (card.diagnosis,
                            f"judge not run: {exc} — the deterministic checks "
                            f"above are the whole verdict") if p))

    repair_on = body.get("repair") if isinstance(body.get("repair"), bool) \
        else (evaluate_on or technical_failed)

    repair_info = None
    receipts = None
    if repair_on and not card.hard_pass:
        decision = oracle_repair.attempt_repair(goal, route, card)
        repair_info = decision.to_dict()
        if decision.action != "none":
            try:
                artifacts2, receipt2, route2 = oracle_runtime.run_bounded(
                    lambda: oracle_repair.execute_repair(goal, route, decision),
                    _left(), f"repair:{route.capability}")
            except Exception:  # noqa: BLE001 — a broken repair keeps attempt 1
                logger.exception("POST /oracle/route: repair attempt failed; "
                                 "keeping the first attempt's answer")
            else:
                card2 = oracle_scorecard.build_technical_scorecard(
                    goal, route2, artifacts2, receipt2)
                if evaluate_on:
                    card2 = oracle_evaluation.evaluate(
                        goal, route2, artifacts2, receipt2, card2)
                receipts = [receipt.to_dict(), receipt2.to_dict()]
                artifacts, receipt, route = artifacts2, receipt2, route2
                card = oracle_repair.annotate_repaired(card2, decision)

    resp = {
        "ok": True,
        "planner_mode": goal.planner_mode.value,
        "goal": goal.to_dict(),
        "route": route.to_dict(),
        "artifacts": artifacts,
        "receipt": receipt.to_dict(),
        "scorecard": card.to_dict(),
        "registry_version": registry_version,
    }
    if repair_info is not None:
        resp["repair"] = repair_info
    if receipts is not None:
        resp["receipts"] = receipts
    return jsonify(resp)


@oracle_bp.route("/oracle/steward", methods=["GET", "POST"])
def oracle_steward():
    """The system checks itself (k113c). GET = report only (no policy change);
    POST = report AND apply bounded rebalancing to the live selector.

    Reads the process reliability ledger + the newest matrix + the live
    catalog's eligible model sets (for starvation). Never silent: a clean
    fleet returns findings saying so with the numbers."""
    from hugpy_oracle import catalog, selection
    from hugpy_oracle.steward import Steward

    sel = selection.process_selector()
    if sel is None or sel.ledger is None:
        return jsonify({"ok": False, "error": "selection disabled or ledger unavailable",
                        "ledger_path": selection.default_ledger_path()}), 503
    eligible: dict[str, tuple[str, ...]] = {}
    try:
        for view in catalog.list_capabilities():
            if view.model_ids:
                eligible[view.name] = tuple(view.model_ids)
    except Exception as exc:  # noqa: BLE001 — starvation checks degrade, the rest runs
        eligible = {"_error": (f"{type(exc).__name__}: {exc}",)}
    apply = request.method == "POST"
    steward = Steward(sel.ledger, selector=sel if apply else None, matrix=sel.matrix(),
                      eligible_models={k: v for k, v in eligible.items() if not k.startswith("_")})
    report = steward.check()
    body = report.to_dict()
    body["applied"] = apply
    body["ledger_rows"] = sel.ledger.count()
    body["selection_policy"] = {f: getattr(sel.policy, f) for f in sel.policy.__slots__}
    return jsonify(body), 200


@oracle_bp.route("/oracle/selection", methods=["POST"])
def oracle_selection():
    """Explain a selection without executing anything: which model THIS call
    would get for a capability, every candidate's verdict and reason, the
    fallback, and the ordered step log. Body: {"capability": ..., "quality":
    "preview|balanced|best", "max_seconds": ..., "max_vram_gib": ...,
    "candidate_index": 0, "candidates": 1, "exclude": [...]}."""
    from hugpy_oracle import selection
    from hugpy_oracle.contracts import BudgetHints, GoalSpec, QualityProfile

    data = request.get_json(silent=True) or {}
    capability = (data.get("capability") or "").strip()
    if not capability:
        return jsonify({"ok": False, "error": "capability is required"}), 400
    sel = selection.process_selector()
    if sel is None:
        return jsonify({"ok": False, "error": "selection disabled"}), 503
    try:
        goal = GoalSpec(objective=f"explain selection for {capability}", raw_prompt=capability,
                        capability=capability,
                        quality=QualityProfile(str(data.get("quality") or "balanced")),
                        budget=BudgetHints(max_seconds=data.get("max_seconds"),
                                           max_vram_gb=data.get("max_vram_gib")))
        decision = sel.decide(capability, goal=goal, exclude=tuple(data.get("exclude") or ()),
                              candidate_index=int(data.get("candidate_index") or 0),
                              candidates=int(data.get("candidates") or 1))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 400
    body = decision.to_dict()
    body["ok"] = not decision.gap
    return jsonify(body), 200


# --------------------------------------------------------------------------- #
# or-k16: producer attribution + ledger summary — the ONE truth workers share.
# Workers with ORACLE_LEDGER_REMOTE=<this host> write through here.
# --------------------------------------------------------------------------- #


@oracle_bp.route("/oracle/producers", methods=["GET", "POST"])
def oracle_producers():
    """GET ?ref=<artifact ref> -> {ok, ref, producer: {ref, capability,
    model_id, ts, worker} | null}; GET without ref -> {ok, producers: [...],
    count} (newest first, ``?limit=``). POST {ref, capability, model_id,
    worker?} -> {ok, producer} (upsert into the central ledger)."""
    from hugpy_oracle import selection

    sel = selection.process_selector()
    if sel is None or sel.ledger is None:
        return jsonify({"ok": False, "error": "selection disabled or ledger unavailable",
                        "ledger_path": selection.default_ledger_path()}), 503
    ledger = sel.ledger
    if request.method == "GET":
        ref = (request.args.get("ref") or "").strip()
        if ref:
            row = ledger.producer(ref)
            return jsonify({"ok": True, "ref": ref, "producer": row}), 200
        try:
            limit = int(request.args.get("limit") or 200)
        except ValueError:
            return jsonify({"ok": False, "error": "limit must be an integer"}), 400
        rows = ledger.producers(limit=max(1, min(limit, 5000)))
        return jsonify({"ok": True, "producers": rows, "count": ledger.producer_count()}), 200
    data = request.get_json(silent=True) or {}
    ref = str(data.get("ref") or "").strip()
    capability = str(data.get("capability") or "").strip()
    model_id = str(data.get("model_id") or "").strip()
    missing = [k for k, v in (("ref", ref), ("capability", capability), ("model_id", model_id)) if not v]
    if missing:
        return jsonify({"ok": False, "error": f"missing: {', '.join(missing)}"}), 400
    worker = data.get("worker")
    worker = str(worker).strip() if worker else (request.remote_addr or None)
    try:
        ledger.remember_producer(ref, capability, model_id, worker=worker)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500
    return jsonify({"ok": True, "producer": ledger.producer(ref)}), 200


@oracle_bp.route("/oracle/ledger/summary", methods=["GET"])
def oracle_ledger_summary():
    """Snapshot of the central reliability ledger: outcome + producer counts,
    per-(capability, model) tallies, workers seen. Read-only."""
    from hugpy_oracle import selection

    sel = selection.process_selector()
    if sel is None or sel.ledger is None:
        return jsonify({"ok": False, "error": "selection disabled or ledger unavailable",
                        "ledger_path": selection.default_ledger_path()}), 503
    body = sel.ledger.summary()
    body["ok"] = True
    body["remote"] = selection.ledger_remote_url()
    return jsonify(body), 200
