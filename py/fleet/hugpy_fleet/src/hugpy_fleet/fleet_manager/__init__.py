"""Fleet Configuration Templates — named, savable, applyable profiles of
worker model attributions and serving state (FLEET-TEMPLATES-DESIGN.md).

Slice 0 lives here: the versioned template schema + validator, the storage
wrappers over the namespaced settings store (namespace ``fleet.templates``),
the live-fleet SNAPSHOT builder, and the DRY-RUN DIFF engine. No APPLY, no
assign/unassign/config writes — the diff COMPUTES a plan, it never executes it.

The core (``templates.py``) is deliberately dependency-injected: the pure
functions (validate / snapshot / diff) take the live worker view as an argument
so nothing here imports flask_app; the route layer feeds them ``list_workers()``
and the tests feed a mocked fleet.
"""
from hugpy_fleet.fleet_manager.templates import (
    SCHEMA_VERSION,
    SERVING_MODES,
    NS_TEMPLATES,
    NS_FLEET,
    KEY_ACTIVE,
    TemplateError,
    validate_template,
    build_snapshot,
    compute_diff,
    list_templates,
    get_template,
    save_template,
    delete_template,
    get_active,
)

__all__ = [
    "SCHEMA_VERSION",
    "SERVING_MODES",
    "NS_TEMPLATES",
    "NS_FLEET",
    "KEY_ACTIVE",
    "TemplateError",
    "validate_template",
    "build_snapshot",
    "compute_diff",
    "list_templates",
    "get_template",
    "save_template",
    "delete_template",
    "get_active",
]
