"""MODEL PRIORITY GROUPS — the EXPLICIT, operator-ordered fallback list.

    GET    /llm/model-groups           list every group          (member-read)
    GET    /llm/model-groups/resolve?key=<model>
                                       the ordered walk + the WHY (member-read)
    POST   /llm/model-groups           create or replace         (OPERATOR)
    PUT    /llm/model-groups/<id>      replace                   (OPERATOR)
    PATCH  /llm/model-groups/<id>      partial update            (OPERATOR)
    DELETE /llm/model-groups/<id>      remove                    (OPERATOR)

Sibling of ``group_routes.py``, which reads the DERIVED grouping (suffix
stripping + a ranking pipeline, off by default). This file owns the EXPLICIT
one: an ordered list the operator wrote by hand, which outranks the derivation
for any key it names.

WHY THIS FILE HAS WRITE ROUTES AND ``group_routes.py`` MUST NOT
--------------------------------------------------------------
That module's state is a TICK on a DERIVED group — three booleans that already
had a home in the F4 settings namespace, so a second write path there would have
been a parallel store. A priority group is a first-class RECORD with structural
invariants (ordering, at-most-one-enabled-group-per-key) that must be validated
on every write. Validation lives in ``comms.priority_groups.put_group`` and this
blueprint is its only caller, so there is still exactly ONE validated write path
— the thing the parallel-store rule is actually protecting.

The mutations are listed in ``operator_auth._SENSITIVE``, so a member gets 403
and an anonymous caller 401 automatically. The two GETs are deliberately absent
from that inventory: reads are member-visible like every other console read.
"""
from flask import jsonify, request

from abstract_flask import get_bp

model_group_bp, logger = get_bp("model_group_bp", __name__)


def _who() -> str:
    """Best-effort authorship for the audit fields; never raises."""
    try:
        from hugpy_server.app.operator_auth import current_principal
        p = current_principal() or {}
        return str(p.get("name") or p.get("id") or "operator")
    except Exception:  # noqa: BLE001
        return "operator"


def _payload() -> dict:
    return request.get_json(silent=True) or {}


@model_group_bp.route("/llm/model-groups", methods=["GET"])
def model_groups_list():
    """Every priority group, in display order. Member-readable.

    Never 500s: a store read that fails reports an empty list with the error
    attached, because a broken groups PAGE must not look like a broken groups
    FEATURE (same posture as GET /llm/groups)."""
    try:
        from hugpy_fleet.central.priority_groups import all_groups
        return jsonify({"groups": all_groups()})
    except Exception as exc:  # noqa: BLE001
        logger.warning("GET /llm/model-groups failed: %s", exc, exc_info=True)
        return jsonify({"groups": [], "error": str(exc)}), 200


@model_group_bp.route("/llm/model-groups/resolve", methods=["GET"])
def model_groups_resolve():
    """THE PREVIEW: what would happen if this key were requested right now.

    ``?key=<model>`` (required), optional ``?pool=`` / ``?task=`` so the preview
    is scoped exactly the way the serve path would be. Returns the ordered
    candidate list with, for EVERY member, the reason it was chosen or skipped
    (blocked / missing / no-worker / lower-priority) — the say-why discipline
    the rest of this console runs on.

    ``route: null`` means "change nothing": either no group claims the key, or
    no member is usable and the request will fail exactly as it does today."""
    key = (request.args.get("key") or "").strip()
    if not key:
        return jsonify({"error": "key is required"}), 400
    pool = (request.args.get("pool") or "").strip() or None
    task = (request.args.get("task") or "").strip() or None
    try:
        from hugpy_fleet.central.priority_group_settings import resolve
        return jsonify(resolve(key, pool, task, full=True))
    except Exception as exc:  # noqa: BLE001 — a preview must not 500
        logger.warning("GET /llm/model-groups/resolve failed for %s: %s",
                       key, exc, exc_info=True)
        return jsonify({"requested": key, "group": None, "candidates": [],
                        "chosen": None, "route": None,
                        "why": f"preview failed: {exc}"}), 200


@model_group_bp.route("/llm/model-groups", methods=["POST"])
def model_groups_create():
    """Create or REPLACE a group. Body: {id?, name, members: [...], workers?,
    enabled?}.

    ``members`` and ``workers`` are each an ORDER, so a write always carries
    the whole list — see ``put_group``. 409 on the one structural conflict (a
    key already claimed by another ENABLED group), with the offending key and
    the group that holds it named in the message; nothing is silently
    dropped."""
    body = _payload()
    try:
        from hugpy_fleet.central.priority_groups import put_group
        rec, errors = put_group(
            body.get("id") or body.get("name"),
            name=body.get("name"),
            members=body.get("members"),
            enabled=body.get("enabled", True),
            workers=body.get("workers"),
            by=_who())
    except Exception as exc:  # noqa: BLE001
        logger.warning("POST /llm/model-groups failed: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500
    if errors:
        conflict = any("at most one enabled group" in e for e in errors)
        return jsonify({"error": "; ".join(errors), "errors": errors}), \
            (409 if conflict else 400)
    return jsonify({"group": rec}), 200


@model_group_bp.route("/llm/model-groups/<group_id>", methods=["PUT", "PATCH"])
def model_groups_update(group_id):
    """Replace (PUT) or patch (PATCH) a group.

    PATCH reads the current record, applies only the supplied fields, and goes
    through the SAME validated write — a partial update never bypasses the
    conflict check."""
    body = _payload()
    try:
        from hugpy_fleet.central.priority_groups import get_group, put_group
        cur = get_group(group_id)
        if cur is None and request.method == "PATCH":
            return jsonify({"error": f"no such priority group: {group_id}"}), 404
        base = cur or {"name": group_id, "members": [], "enabled": True}
        rec, errors = put_group(
            group_id,
            name=body.get("name", base.get("name")),
            members=body.get("members", base.get("members")),
            enabled=body.get("enabled", base.get("enabled", True)),
            workers=body.get("workers", base.get("workers")),
            by=_who())
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s /llm/model-groups/%s failed: %s",
                       request.method, group_id, exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500
    if errors:
        conflict = any("at most one enabled group" in e for e in errors)
        return jsonify({"error": "; ".join(errors), "errors": errors}), \
            (409 if conflict else 400)
    return jsonify({"group": rec}), 200


@model_group_bp.route("/llm/model-groups/member", methods=["POST"])
def model_groups_member():
    """Move a model between groups in one gesture — the model table's Group
    column. Body: {model_key, group_id?} — a falsy group_id removes the key
    from every group. Append-at-end on join; removals precede the add so the
    exclusivity invariant never trips mid-move. 409 when a removal would leave
    a group empty (delete the group instead — nothing is silently deleted)."""
    body = _payload()
    key = str(body.get("model_key") or "").strip()
    if not key:
        return jsonify({"error": "model_key is required"}), 400
    try:
        from hugpy_fleet.central.priority_groups import move_member
        changed, errors = move_member(key, body.get("group_id"), by=_who())
    except Exception as exc:  # noqa: BLE001
        logger.warning("POST /llm/model-groups/member failed: %s", exc,
                       exc_info=True)
        return jsonify({"error": str(exc)}), 500
    if errors:
        empty = any("leave group" in e for e in errors)
        return jsonify({"error": "; ".join(errors), "errors": errors,
                        "changed": changed}), (409 if empty else 400)
    return jsonify({"changed": changed}), 200


@model_group_bp.route("/llm/model-groups/<group_id>/allocate", methods=["POST"])
def model_groups_allocate(group_id):
    """Designate EVERY member of the group to EVERY worker in its ordered
    ``workers`` allocation — the one-call form of the console's N-call fan-out
    (WorkersPanel ``groupAssign``/``allocateMany``). Body ``{"workers": [...]}``
    optionally overrides the group's list for THIS call; the record is not
    rewritten (orders are written by PUT/PATCH, the validated path).

    DESIGNATION ONLY: writes ``worker["models"]`` + assign-memory via
    ``assign_model`` — the operator-intent channel, never ``grants``. No warm,
    no spill contract; loading stays the console's explicit /load gesture with
    its preflight. Runs against a DISABLED group too (designation is registry
    state, not routing), with ``enabled`` echoed so the caller can see the
    group won't route yet. Outcomes are per (worker × model), skips named —
    a phantom key or an unknown worker is reported, never silently designated."""
    body = _payload()
    try:
        from hugpy_fleet.central.priority_groups import effective_workers, expand_members, get_group
        group = get_group(group_id)
        if group is None:
            return jsonify({"error": f"no such priority group: {group_id}"}), 404
        # Nesting: fan out the EXPANDED membership (nested groups contribute
        # their models), and default to the governing allocation — the group's
        # own workers, or the parent's it inherits as a mounted module.
        members = [k for k, _via in expand_members(group)]
        workers = body.get("workers") or effective_workers(group) or []
        if not workers:
            return jsonify({"error": "group has no workers allocation — set the "
                            "group's ordered workers list (or pass one in the "
                            "body) before allocating"}), 400

        from hugpy_fleet.central.priority_group_settings import canonical_key
        from hugpy_fleet.central.workers import assign_model, list_workers
        from hugpy_engine.config.models.models_config import get_models_dict
        manifest = get_models_dict(dict_return=True)
        roster = list_workers()

        def _find(want: str):
            w = str(want or "").strip().lower()
            for row in roster:
                if w in {str(row.get("id") or "").strip().lower(),
                         str(row.get("name") or "").strip().lower()}:
                    return row
            return None

        outcomes = []
        for want in workers:
            row = _find(want)
            if row is None:
                outcomes.append({"worker": want, "model": None,
                                 "status": "no-such-worker"})
                continue
            have = set(row.get("models") or [])
            for mk in members:
                # Alias-tolerant manifest gate (2026-08-26): a group member is
                # whatever the operator typed ("Qwen2.5-VL-7B-Instruct"); the
                # catalog key may be owner-qualified ("Qwen~..."). Designate
                # the CANONICAL spelling — the exact-match gate silently
                # skipped every bare-spelled member as "not-in-manifest".
                ck = canonical_key(mk, manifest)
                if ck is None:
                    outcomes.append({"worker": row.get("name") or row["id"],
                                     "model": mk, "status": "not-in-manifest"})
                    continue
                status = "already" if ck in have else "designated"
                if status == "designated":
                    assign_model(row["id"], ck)
                    have.add(ck)
                outcomes.append({"worker": row.get("name") or row["id"],
                                 "model": ck, "status": status})
        designated = sum(1 for o in outcomes if o["status"] == "designated")
        logger.info("group allocate %s: %d designated, %d outcomes (by=%s)",
                    group_id, designated, len(outcomes), _who())
        return jsonify({"group": group_id, "enabled": group.get("enabled"),
                        "designated": designated, "outcomes": outcomes}), 200
    except Exception as exc:  # noqa: BLE001
        logger.warning("POST /llm/model-groups/%s/allocate failed: %s",
                       group_id, exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# TASK TEMPLATES (2026-08-26) — task blueprints over groups + reserved workers.
# Sibling surface on this blueprint (same registration, same operator gates):
# a template is groups-machinery, not worker-machinery, even though activation
# touches worker rows. Records live in comms.task_templates (one validated
# write path); THIS layer owns activation orchestration because it needs the
# worker store.
# ---------------------------------------------------------------------------
@model_group_bp.route("/llm/templates", methods=["GET"])
def templates_list():
    try:
        from hugpy_fleet.central.task_templates import all_templates, derive_workers
        out = []
        for t in all_templates():
            t["derived_workers"] = derive_workers(t)
            out.append(t)
        return jsonify({"templates": out})
    except Exception as exc:  # noqa: BLE001
        logger.warning("GET /llm/templates failed: %s", exc, exc_info=True)
        return jsonify({"templates": [], "error": str(exc)}), 200


@model_group_bp.route("/llm/templates", methods=["POST"])
def templates_create():
    body = _payload()
    try:
        from hugpy_fleet.central.task_templates import put_template
        rec, errors = put_template(
            body.get("id") or body.get("name"), name=body.get("name"),
            groups=body.get("groups"), workers=body.get("workers"),
            tasks=body.get("tasks"), by=_who())
    except Exception as exc:  # noqa: BLE001
        logger.warning("POST /llm/templates failed: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500
    if errors:
        return jsonify({"error": "; ".join(errors), "errors": errors}), 400
    return jsonify({"template": rec}), 200


@model_group_bp.route("/llm/templates/<template_id>", methods=["PUT", "PATCH"])
def templates_update(template_id):
    body = _payload()
    try:
        from hugpy_fleet.central.task_templates import get_template, put_template
        cur = get_template(template_id)
        if cur is None and request.method == "PATCH":
            return jsonify({"error": f"no such template: {template_id}"}), 404
        base = cur or {"name": template_id, "groups": [], "workers": [],
                       "tasks": []}
        rec, errors = put_template(
            template_id,
            name=body.get("name", base.get("name")),
            groups=body.get("groups", base.get("groups")),
            workers=body.get("workers", base.get("workers")),
            tasks=body.get("tasks", base.get("tasks")),
            by=_who())
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s /llm/templates/%s failed: %s",
                       request.method, template_id, exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500
    if errors:
        return jsonify({"error": "; ".join(errors), "errors": errors}), 400
    return jsonify({"template": rec}), 200


@model_group_bp.route("/llm/templates/<template_id>/tasks/<task_name>",
                      methods=["POST"])
def templates_fill_task(template_id, task_name):
    """Fill ONE task's model slot — the operator's fill-in-the-optimum-model
    operation (task-oriented templates, 2026-08-28). Body: {"model": str} to
    set, {"model": null} to clear back to unfilled. Never touches the rest of
    the outline."""
    body = _payload()
    if "model" not in body:
        return jsonify({"error": 'body must carry "model" (a model key, or '
                        "null to clear the slot)"}), 400
    try:
        from hugpy_fleet.central.task_templates import set_task_model
        rec, err = set_task_model(template_id, task_name, body.get("model"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("POST /llm/templates/%s/tasks/%s failed: %s",
                       template_id, task_name, exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500
    if err:
        return jsonify({"error": err}), 404 if "no such" in err or "no task" in err else 400
    return jsonify({"template": rec}), 200


@model_group_bp.route("/llm/templates/<template_id>", methods=["DELETE"])
def templates_delete(template_id):
    try:
        from hugpy_fleet.central.task_templates import delete_template, get_template
        t = get_template(template_id)
        if t is not None and t.get("active"):
            return jsonify({"error": "template is active — deactivate first "
                            "(deleting a live reservation would strand the "
                            "pool tags)"}), 409
        if not delete_template(template_id):
            return jsonify({"error": f"no such template: {template_id}"}), 404
    except Exception as exc:  # noqa: BLE001
        logger.warning("DELETE /llm/templates/%s failed: %s", template_id, exc,
                       exc_info=True)
        return jsonify({"error": str(exc)}), 500
    return jsonify({"deleted": template_id}), 200


@model_group_bp.route("/llm/templates/<template_id>/activate", methods=["POST"])
def templates_activate(template_id):
    """RESERVE + CAST. Sets each target worker's pool to the template id (the
    existing exact-match reservation — general traffic can no longer land
    there; requests tagged pool=<template id> can) and allocates every listed
    group's expanded membership to those workers.

    409 when a target worker is already pooled under a DIFFERENT tag — one
    reservation at a time is the whole point; pass {"force": true} to overrule
    (the response names what was overruled)."""
    body = _payload()
    try:
        from hugpy_fleet.central.priority_groups import effective_workers, expand_members, get_group
        from hugpy_fleet.central.task_templates import derive_workers, get_template, mark_active
        from hugpy_engine.config.models.models_config import get_models_dict
        from hugpy_fleet.central.workers import assign_model, list_workers, worker_store
        t = get_template(template_id)
        if t is None:
            return jsonify({"error": f"no such template: {template_id}"}), 404
        targets = derive_workers(t)
        if not targets:
            return jsonify({"error": "template resolves to no workers — set "
                            "its workers list or allocate its groups"}), 400
        roster = list_workers()

        def _find(want):
            w = str(want or "").strip().lower()
            for row in roster:
                if w in {str(row.get("id") or "").strip().lower(),
                         str(row.get("name") or "").strip().lower()}:
                    return row
            return None

        rows, missing, conflicts = [], [], []
        for want in targets:
            row = _find(want)
            if row is None:
                missing.append(want)
                continue
            pool = (row.get("pool") or "").strip()
            if pool and pool != t["id"]:
                conflicts.append({"worker": row.get("name") or row["id"],
                                  "pool": pool})
            rows.append(row)
        if missing:
            return jsonify({"error": f"unknown workers: {missing}"}), 400
        if conflicts and not body.get("force"):
            return jsonify({"error": "worker(s) already reserved under another "
                            "pool — deactivate that first or pass force",
                            "conflicts": conflicts}), 409

        from hugpy_fleet.central.priority_group_settings import canonical_key
        manifest = get_models_dict(dict_return=True)
        designated = already = skipped = 0
        for row in rows:
            worker_store.set_pool(row["id"], t["id"])
            have = set(row.get("models") or [])
            for gid in t["groups"]:
                g = get_group(gid)
                if not g:
                    continue
                for mk, _via in expand_members(g):
                    # Alias-tolerant, same as /allocate: designate the
                    # canonical catalog spelling of the member.
                    ck = canonical_key(mk, manifest)
                    if ck is None:
                        skipped += 1
                        continue
                    if ck in have:
                        already += 1
                        continue
                    assign_model(row["id"], ck)
                    have.add(ck)
                    designated += 1
        rec = mark_active(t["id"], True)
        logger.info("template %s ACTIVATED: reserved %d workers (%s), "
                    "%d designated, %d already, %d skipped (by=%s)",
                    t["id"], len(rows),
                    [r.get("name") or r["id"] for r in rows],
                    designated, already, skipped, _who())
        return jsonify({"template": rec,
                        "reserved": [r.get("name") or r["id"] for r in rows],
                        "overruled": conflicts if body.get("force") else [],
                        "designated": designated, "already": already,
                        "not_in_manifest": skipped}), 200
    except Exception as exc:  # noqa: BLE001
        logger.warning("POST /llm/templates/%s/activate failed: %s",
                       template_id, exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@model_group_bp.route("/llm/templates/<template_id>/deactivate", methods=["POST"])
def templates_deactivate(template_id):
    """RELEASE. Clears the pool tag on every worker still carrying this
    template's id (a tag repointed elsewhere meanwhile is left alone).
    Designations stay — cheap registry state that makes reactivation instant;
    unassign is the explicit per-model gesture if the operator wants it gone."""
    try:
        from hugpy_fleet.central.task_templates import get_template, mark_active
        from hugpy_fleet.central.workers import list_workers, worker_store
        t = get_template(template_id)
        if t is None:
            return jsonify({"error": f"no such template: {template_id}"}), 404
        released = []
        for row in list_workers():
            if (row.get("pool") or "").strip() == t["id"]:
                worker_store.set_pool(row["id"], "")
                released.append(row.get("name") or row["id"])
        rec = mark_active(t["id"], False)
        logger.info("template %s DEACTIVATED: released %s (by=%s)",
                    t["id"], released, _who())
        return jsonify({"template": rec, "released": released}), 200
    except Exception as exc:  # noqa: BLE001
        logger.warning("POST /llm/templates/%s/deactivate failed: %s",
                       template_id, exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@model_group_bp.route("/llm/model-groups/<group_id>", methods=["DELETE"])
def model_groups_delete(group_id):
    """Remove a group. Idempotent-ish: 404 when it was never there, so a
    console that thinks it deleted something twice learns the truth."""
    try:
        from hugpy_fleet.central.priority_groups import delete_group
        if not delete_group(group_id):
            return jsonify({"error": f"no such priority group: {group_id}"}), 404
    except Exception as exc:  # noqa: BLE001
        logger.warning("DELETE /llm/model-groups/%s failed: %s",
                       group_id, exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500
    return jsonify({"deleted": group_id}), 200
