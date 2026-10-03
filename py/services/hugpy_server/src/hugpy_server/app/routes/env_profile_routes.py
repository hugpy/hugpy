"""PER-MODEL ENVIRONMENTS — routes (operator 2026-10-02; store: app/env_profiles).

    GET  /llm/env-profiles                      every profile + per-worker state/lock/test
    POST /llm/env-profiles                      {name, packages, base, note, by_kind?}  create/replace
    POST /llm/env-profiles/<name>/approve       approve a PROPOSED profile (operator)
    POST /llm/env-profiles/<name>/test          {model, worker}  one-token load through the profile

Writes are operator-gated (operator_auth._SENSITIVE). hugpy-brain posts with
by_kind "agent": its profile starts PROPOSED and files a help ticket the
operator approves from the help panel."""
from __future__ import annotations

import threading
import time

from flask import jsonify, request

from abstract_flask import get_bp

env_profile_bp, logger = get_bp("env_profile_bp", __name__)


def _who() -> str:
    try:
        from hugpy_server.app.routes.worker_routes import _operator_name
        return _operator_name() or "operator"
    except Exception:  # noqa: BLE001
        return "operator"


@env_profile_bp.route("/llm/env-profiles", methods=["GET"])
def env_profiles_list():
    from hugpy_server.app import env_profiles as ep
    return jsonify({"profiles": ep.list_all(), "bases": list(ep.BASES)})


@env_profile_bp.route("/llm/env-profiles", methods=["POST"])
def env_profiles_put():
    from hugpy_server.app import env_profiles as ep
    body = request.get_json(silent=True) or {}
    kind = "agent" if str(body.get("by_kind") or "").lower() == "agent" else "operator"
    try:
        prof = ep.put(body.get("name"), body.get("packages"), body.get("base") or "worker",
                      body.get("note") or "", by=_who(), by_kind=kind)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if prof is None:
        return jsonify({"error": "environment store unavailable"}), 503
    if prof.get("status") == "proposed":
        try:
            from hugpy_server.app.help_tickets import file_ticket
            file_ticket("env-profile", f"environment proposed: {prof['name']} ({', '.join(prof['packages']) or 'no packages'})",
                        model_key=None, worker_id=None, worker_name=None,
                        detail={"name": prof["name"], "packages": prof["packages"], "base": prof["base"],
                                "note": prof.get("note"), "by": prof.get("created_by")})
        except Exception:  # noqa: BLE001 — a ticket never blocks the proposal
            logger.debug("env-profile ticket not filed", exc_info=True)
    return jsonify(prof), 200


@env_profile_bp.route("/llm/env-profiles/<name>/approve", methods=["POST"])
def env_profiles_approve(name):
    from hugpy_server.app import env_profiles as ep
    prof = ep.approve(name, by=_who())
    if not prof:
        return jsonify({"error": f"no environment {name!r}"}), 404
    return jsonify(prof)


def _test(name: str, model: str, worker_id: str) -> None:
    """One pinned one-token chat through central — the same path any call takes,
    so it loads the model in its attributed profile — recorded per worker."""
    from hugpy_server.app import env_profiles as ep
    t0 = time.time()
    try:
        from hugpy_server.app.routes.test_fire_routes import fire_one

        class _Job:      # the minimum fire_one needs: a pinned worker + id
            job_id = f"env-{name}"
            def __init__(self, wid):
                self.worker_id = wid
            def mark_in_flight(self, *a, **k):
                pass
        res = fire_one(model, "Say OK.", 8, _Job(worker_id))
        result = {"model": model, "ok": bool(res.get("ok")), "error": res.get("error"),
                  "latency_s": res.get("latency_s"), "at": time.time()}
    except Exception as exc:  # noqa: BLE001
        result = {"model": model, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                  "latency_s": round(time.time() - t0, 3), "at": time.time()}
    ep.record_test(name, worker_id, result)
    logger.info("env profile %s test on %s with %s: %s", name, worker_id, model,
                "ok" if result["ok"] else result["error"])


@env_profile_bp.route("/llm/env-profiles/<name>/test", methods=["POST"])
def env_profiles_test(name):
    from hugpy_server.app import env_profiles as ep
    body = request.get_json(silent=True) or {}
    model, worker_id = str(body.get("model") or ""), str(body.get("worker") or "")
    if not model or not worker_id:
        return jsonify({"error": "model and worker are required"}), 400
    prof = ep.get(name)
    if not prof:
        return jsonify({"error": f"no environment {name!r}"}), 404
    if prof.get("status") != "approved":
        return jsonify({"error": "approve the environment first"}), 409
    threading.Thread(target=_test, args=(name, model, worker_id), daemon=True,
                     name=f"env-test-{name}").start()
    return jsonify({"ok": True, "testing": {"name": name, "model": model, "worker": worker_id}}), 202
