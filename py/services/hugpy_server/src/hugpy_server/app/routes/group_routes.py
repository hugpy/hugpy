"""MODEL GROUPS — the read surface.

    GET /llm/groups     groups + members + ticks + the PER-WORKER verdict,
                        with the WHY on every exclusion.

Spec: ``dev/MODEL-GROUPS-SPEC.md``. Vocabulary: ``dev/GLOSSARY.md`` (*model
group*, *tick*, *ladder walk*).

READ-ONLY BY DESIGN — there is no write route in this file and there must not be
one. Ticks are OPERATOR LEVERS and they live in the runtime settings store,
where the operator gate already covers them:

    GET  /settings/model_groups                              open (UIs read it)
    POST /settings/model_groups/<group_key> {"merge": {...}} OPERATOR-GATED
    POST /settings/model_groups/enabled     {"value": true}  OPERATOR-GATED

``operator_auth._SENSITIVE`` already matches ``^/settings/.+$`` for
POST/PUT/DELETE, so the ticks inherit the gate for free. Inventing a
``POST /llm/groups/<key>/ticks`` here would be a SECOND write path to the same
state with its own gate to get wrong — exactly the parallel-store mistake the F4
settings module exists to prevent.

This GET is open (like every other console read) and works whether or not the
feature is enabled: it is a pure read over the catalog that routes nothing, so
an operator can see exactly what groups WOULD do before turning them on. The
payload's ``enabled`` field says which it is.
"""
from flask import jsonify, request

from abstract_flask import get_bp

group_bp, logger = get_bp("group_bp", __name__)


@group_bp.route("/llm/groups", methods=["GET"])
def llm_groups():
    """Groups, members, ticks, and one honest verdict per worker.

    Optional ``?pool=`` scopes the verdicts to a dedicated worker pool, matching
    the selector's own pool semantics (a general request never lands on a pooled
    worker, so a pooled verdict would otherwise be misleading).

    Never 500s on a policy bug: a derivation failure reports an empty group list
    with the flag state intact, because a broken groups PAGE must not look like
    a broken groups FEATURE."""
    pool = (request.args.get("pool") or "").strip() or None
    try:
        from hugpy_fleet.central.model_groups import describe_groups
        return jsonify(describe_groups(pool=pool))
    except Exception as exc:  # noqa: BLE001
        logger.warning("GET /llm/groups failed: %s", exc, exc_info=True)
        return jsonify({"enabled": False, "source": "default", "groups": [],
                        "error": str(exc)}), 200
