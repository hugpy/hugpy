"""HTTP surface for Fleet Configuration Templates (FLEET-TEMPLATES-DESIGN.md §6).

Slice 0 — storage + read-only + snapshot + diff. NO apply: the diff computes a
dry-run plan and returns it; nothing here mutates a worker or relays a call.

  GET    /fleet/templates            list (names, revisions, descriptions)
  GET    /fleet/templates/<name>     the doc (404 if absent)
  PUT    /fleet/templates/<name>     validate -> bump revision -> store
  DELETE /fleet/templates/<name>     remove
  POST   /fleet/templates/snapshot   capture the live fleet as a template  {name, description?}
  POST   /fleet/templates/<name>/diff  DRY-RUN plan vs the live fleet (read-only)
  GET    /fleet/active               the fleet.active pointer (null if none)

All behavior lives in managers.fleet.templates; this module is a thin adapter
(the same discipline as comms_routes). The MUTATING routes (PUT/DELETE/snapshot)
are operator-gated by operator_auth._SENSITIVE; GET + diff are read-only and open.
The blueprint is auto-discovered by abstract_flask._discover_blueprints (any
routes/__init__ attribute named ``*_bp``) — see routes/__init__.py.
"""
from flask import request, jsonify, abort

from abstract_flask import get_bp
from hugpy_fleet.central.workers import list_workers
from hugpy_fleet.fleet_manager import templates as ft

fleet_bp, logger = get_bp("fleet_bp", __name__)  # noqa: F405


def _audit(action, detail):
    """Best-effort audit record, mirroring comms_routes.audit. Never raises."""
    try:
        from hugpy_server.app.routes.comms_routes import audit
        audit(action, detail)
    except Exception:
        pass


@fleet_bp.route("/fleet/templates", methods=["GET"])
def fleet_templates_list():
    return jsonify({"templates": ft.list_templates()})


@fleet_bp.route("/fleet/templates/<name>", methods=["GET"])
def fleet_template_get(name):
    doc = ft.get_template(name)
    if doc is None:
        abort(404, description=f"Unknown template: {name}")
    return jsonify(doc)


@fleet_bp.route("/fleet/templates/<name>", methods=["PUT"])
def fleet_template_put(name):
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        abort(400, description="body must be a template JSON object")
    # The URL is authoritative for the template name (server-owned key).
    doc = dict(body)
    doc["name"] = name
    try:
        stored = ft.save_template(doc)
    except ft.TemplateError as exc:
        abort(400, description=str(exc))
    _audit("fleet.template.save", {"name": name, "revision": stored.get("revision")})
    return jsonify(stored)


@fleet_bp.route("/fleet/templates/<name>", methods=["DELETE"])
def fleet_template_delete(name):
    existed = ft.delete_template(name)
    _audit("fleet.template.delete", {"name": name, "existed": existed})
    return jsonify({"name": name, "deleted": existed})


@fleet_bp.route("/fleet/templates/snapshot", methods=["POST"])
def fleet_template_snapshot():
    body = request.get_json(silent=True) or {}
    name = str(body.get("name") or "").strip()
    if not name:
        abort(400, description="name required")
    description = body.get("description")
    try:
        doc = ft.build_snapshot(name, description, list_workers())
        stored = ft.save_template(doc)
    except ft.TemplateError as exc:
        abort(400, description=str(exc))
    _audit("fleet.template.snapshot",
           {"name": name, "workers": len(stored.get("workers") or [])})
    return jsonify(stored)


@fleet_bp.route("/fleet/templates/<name>/diff", methods=["POST"])
def fleet_template_diff(name):
    """DRY-RUN only (§3a). Diffs the named (or inline) template against the live
    fleet and returns the ordered plan. Never writes, never relays."""
    body = request.get_json(silent=True) or {}
    doc = body.get("template") if isinstance(body.get("template"), dict) else \
        ft.get_template(name)
    if doc is None:
        abort(404, description=f"Unknown template: {name}")
    try:
        diff = ft.compute_diff(doc, list_workers())
    except ft.TemplateError as exc:
        abort(400, description=str(exc))
    return jsonify(diff)


@fleet_bp.route("/fleet/active", methods=["GET"])
def fleet_active_get():
    return jsonify({"active": ft.get_active()})
