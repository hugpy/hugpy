"""JSON Schema, generated from the frozen dataclasses (k101).

The architecture doc asks for "generated JSON Schema" for the platform
contracts. The obvious way to get it is a pydantic migration; this module is
the argument that we do not need one. Every oracle contract is already a frozen
slotted dataclass with exact annotations, so the schema can be DERIVED from the
type hints — pure stdlib, no dependency, no second definition of the truth to
keep in sync, and no runtime cost on the serving path (nothing here is imported
by the route; it is called by tests, by a docs build, or by a client
generator).

WHAT IT PRODUCES. Draft 2020-12 with ``$defs`` for nested dataclasses:

  * a str-``Enum``          -> ``{"type": "string", "enum": [...]}``
  * ``X | None``            -> nullable (``"type": [t, "null"]`` for a simple
                              type, ``anyOf`` with ``{"type": "null"}`` for a
                              ``$ref``)
  * ``tuple[X, ...]``       -> ``{"type": "array", "items": …}``
  * ``tuple[X, Y]``         -> a fixed-length array (``prefixItems``)
  * a nested dataclass      -> ``$ref`` into ``$defs``
  * ``FrozenMap``/mapping   -> ``{"type": "object"}``
  * a field with no default -> listed in ``required``

WHAT IT DOES NOT DO, on purpose:

  * it never sets ``additionalProperties: false``. ``to_dict`` deliberately
    emits legacy MIRRORS (``ResourceHints.min_vram_gb`` next to ``vram_gib``)
    so old readers keep working; a schema that rejected them would be a schema
    that rejects our own wire format;
  * it does not invent validation the dataclass does not enforce. ``minLength``,
    ``minimum`` and friends live in ``__post_init__`` where they can produce a
    real error message, and a schema that promised more than the constructor
    checks would be the fabrication this codebase keeps refusing;
  * ``WIRE_OVERRIDES`` is the ONE escape hatch, for the one field whose stored
    shape and wire shape genuinely differ (``ExecutionReceipt.request`` is a
    tuple of pairs in memory and an object on the wire). It is a table so the
    exception is visible rather than a special case buried in the walker.

Pure stdlib. No pathlib. Deterministic: same input, same bytes — there is no
timestamp in the output, because a schema that changes every time it is
generated cannot be diffed in review.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import types
import typing
from typing import Any, Mapping


logger = logging.getLogger(__name__)

JSON_SCHEMA_DIALECT: str = "https://json-schema.org/draft/2020-12/schema"

#: Fields whose WIRE shape (``to_dict``) differs from their stored annotation.
#: One entry today; each one is a place where a reader of the dataclass would
#: guess wrong, so each one is written down rather than inferred.
WIRE_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "ExecutionReceipt": {
        "request": {
            "type": "object",
            "description": ("the normalized request. Stored as canonical "
                            "(key, json-value) PAIRS so the receipt stays "
                            "hashable; serialized as an object by "
                            "to_dict()/request_dict()"),
        },
    },
}

_PRIMITIVES: dict[Any, str] = {
    str: "string",
    bool: "boolean",      # before int: bool is a subclass of int
    int: "integer",
    float: "number",
}


# ---------------------------------------------------------------------------
# The walker
# ---------------------------------------------------------------------------


def _is_dataclass_type(obj: Any) -> bool:
    return isinstance(obj, type) and dataclasses.is_dataclass(obj)


def _is_enum_type(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, enum.Enum)


def _enum_schema(cls: type[enum.Enum]) -> dict[str, Any]:
    values = [m.value for m in cls]
    kinds = {type(v) for v in values}
    schema: dict[str, Any] = {"enum": values}
    if kinds == {str}:
        schema = {"type": "string", "enum": values}
    if cls.__doc__ and not cls.__doc__.startswith("An enumeration"):
        schema["description"] = " ".join(cls.__doc__.split())
    return schema


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    """``schema`` or null. A plain ``type`` gains ``"null"``; anything with a
    ``$ref``, an ``enum`` or a composite needs ``anyOf``.

    Two traps this avoids, both of which silently produce a schema that rejects
    valid payloads: a ``$ref`` sibling keyword is ignored by validators, and
    widening ``type`` on an ENUM does nothing because ``null`` is still not one
    of the enumerated values."""
    if ("$ref" in schema or "anyOf" in schema or "prefixItems" in schema
            or "enum" in schema):
        return {"anyOf": [schema, {"type": "null"}]}
    kind = schema.get("type")
    if kind is None:
        return schema            # already unconstrained: null is allowed
    kinds = list(kind) if isinstance(kind, list) else [kind]
    if "null" not in kinds:
        kinds.append("null")
    out = dict(schema)
    out["type"] = kinds
    return out


def _ref_for(cls: type, defs: dict[str, Any]) -> dict[str, Any]:
    name = cls.__name__
    if name not in defs:
        defs[name] = {}          # placeholder first: recursion-safe
        defs[name] = _object_schema(cls, defs)
    return {"$ref": f"#/$defs/{name}"}


def _type_schema(annotation: Any, defs: dict[str, Any]) -> dict[str, Any]:
    """One annotation -> one schema fragment. Anything unrecognized becomes the
    permissive ``{}`` rather than a guess: an over-tight generated schema would
    reject valid payloads, which is the worse failure of the two."""
    if annotation is Any or annotation is None:
        return {}
    if annotation is type(None):
        return {"type": "null"}
    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}
    if _is_enum_type(annotation):
        return _enum_schema(annotation)
    if _is_dataclass_type(annotation):
        return _ref_for(annotation, defs)
    if isinstance(annotation, type) and issubclass(annotation, Mapping):
        return {"type": "object"}

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin in (typing.Union, types.UnionType):
        parts = [a for a in args if a is not type(None)]
        nullable = len(parts) != len(args)
        if len(parts) == 1:
            schema = _type_schema(parts[0], defs)
        else:
            merged = [_type_schema(a, defs) for a in parts]
            simple = [m.get("type") for m in merged]
            if all(isinstance(t, str) for t in simple) and all(
                    set(m) <= {"type"} for m in merged):
                schema = {"type": sorted(set(simple))}   # type: ignore[arg-type]
            else:
                schema = {"anyOf": merged}
        return _nullable(schema) if nullable else schema

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _type_schema(args[0], defs)}
        if not args:
            return {"type": "array"}
        return {"type": "array",
                "prefixItems": [_type_schema(a, defs) for a in args],
                "minItems": len(args), "maxItems": len(args)}
    if origin in (list, set, frozenset):
        items = _type_schema(args[0], defs) if args else {}
        return {"type": "array", "items": items}
    if origin is dict or (isinstance(origin, type)
                          and issubclass(origin, Mapping)):
        value = _type_schema(args[1], defs) if len(args) == 2 else {}
        return {"type": "object"} if not value else {
            "type": "object", "additionalProperties": value}
    if origin is not None and isinstance(origin, type):
        if issubclass(origin, Mapping):
            return {"type": "object"}
    return {}


def _has_default(field: dataclasses.Field) -> bool:
    return (field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING)  # type: ignore[misc]


def _object_schema(cls: type, defs: dict[str, Any]) -> dict[str, Any]:
    try:
        hints = typing.get_type_hints(cls)
    except Exception as exc:  # noqa: BLE001 — an unresolvable hint is not fatal
        logger.debug("schema_export: hints for %s unresolvable (%s)",
                     cls.__name__, exc)
        hints = {}
    overrides = WIRE_OVERRIDES.get(cls.__name__, {})
    properties: dict[str, Any] = {}
    required: list[str] = []
    for field in dataclasses.fields(cls):
        if field.name in overrides:
            properties[field.name] = dict(overrides[field.name])
        else:
            properties[field.name] = _type_schema(
                hints.get(field.name, Any), defs)
        if not _has_default(field):
            required.append(field.name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    if cls.__doc__:
        schema["description"] = " ".join(cls.__doc__.split())
    return schema


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def json_schema_for(dataclass_type: type, *, title: str | None = None) -> dict[str, Any]:
    """JSON Schema (draft 2020-12) for one frozen dataclass, with every nested
    dataclass resolved into ``$defs``."""
    if not _is_dataclass_type(dataclass_type):
        raise TypeError(
            f"json_schema_for expects a dataclass TYPE, got "
            f"{dataclass_type!r} — schemas are generated from the contract, "
            f"never from an instance")
    defs: dict[str, Any] = {}
    schema = _object_schema(dataclass_type, defs)
    out: dict[str, Any] = {"$schema": JSON_SCHEMA_DIALECT,
                           "title": title or dataclass_type.__name__}
    out.update(schema)
    if defs:
        out["$defs"] = {k: defs[k] for k in sorted(defs)}
    return out


#: The doc §3 platform contracts, by their DOC names, mapped to the classes
#: this fleet actually implements. ``CapabilityDescriptor`` is
#: ``CapabilityView`` (k101 grew one into the other rather than adding a rival
#: contract); ``ArtifactManifest`` lives on the AGENT side (k96,
#: ``hugpy_agent.mct.manifest``) and is reported as missing here rather than
#: duplicated — two definitions of one manifest is the failure mode the whole
#: contract layer exists to avoid.
PLATFORM_CONTRACTS: tuple[tuple[str, str, str], ...] = (
    ("GoalSpec", "hugpy_oracle.contracts", "GoalSpec"),
    ("CapabilityDescriptor", "hugpy_oracle.contracts", "CapabilityView"),
    ("PlanGraph", "hugpy_oracle.plan", "PlanGraph"),
    ("ArtifactManifest", "hugpy_agent.mct.manifest", "ArtifactManifest"),
    ("Scorecard", "hugpy_oracle.contracts", "Scorecard"),
    ("ExecutionReceipt", "hugpy_oracle.contracts", "ExecutionReceipt"),
)

#: The domain artifacts other Wave-2 tasks landed (doc §3.7). Exported when
#: importable, recorded as missing when not — this table is allowed to name
#: work in flight, and ``export_all`` never fails because a sibling task has
#: not finished.
DOMAIN_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    # k102 — audio-first artifacts
    ("DialogueTimeline", "hugpy_oracle.audio_master", "DialogueTimeline"),
    ("VoiceProfile", "hugpy_oracle.audio_master", "VoiceProfile"),
    ("AudioMaster", "hugpy_oracle.audio_master", "AudioMaster"),
    # k103 — plan nodes
    ("PlanNode", "hugpy_oracle.plan", "PlanNode"),
    ("Port", "hugpy_oracle.plan", "Port"),
    # k104 — production lock / segment compilation (in flight)
    ("GenerationSnapshot", "hugpy_oracle.production", "GenerationSnapshot"),
    ("ContinuityBible", "hugpy_oracle.production", "ContinuityBible"),
    ("ShotPlan", "hugpy_oracle.production", "ShotPlan"),
    ("SegmentSpec", "hugpy_oracle.segments", "SegmentSpec"),
)


def _resolve(module_name: str, attr: str) -> tuple[Any | None, str]:
    """The class, or (None, why-not). Import failures are DATA here: a schema
    export that raised because a sibling task is mid-edit would be useless
    exactly when it is most wanted."""
    try:
        module = __import__(module_name, fromlist=[attr])
    except Exception as exc:  # noqa: BLE001
        return None, (f"module {module_name!r} is not importable here "
                      f"({type(exc).__name__}: {exc})")
    target = getattr(module, attr, None)
    if target is None:
        return None, f"{module_name} declares no {attr!r}"
    if not _is_dataclass_type(target):
        return None, f"{module_name}.{attr} is not a dataclass type"
    return target, ""


def export_all() -> dict[str, Any]:
    """Every generated schema, plus an honest record of what could not be
    generated and why.

    ``missing`` is a first-class part of the output: "we export five of the six
    platform contracts and here is why the sixth is elsewhere" is information;
    silently exporting five is not."""
    schemas: dict[str, Any] = {}
    missing: dict[str, str] = {}
    for group in (PLATFORM_CONTRACTS, DOMAIN_ARTIFACTS):
        for title, module_name, attr in group:
            if title == "ArtifactManifest":
                # This contract belongs to the agent package. Its presence in
                # the interpreter must not make the Oracle claim its schema.
                missing[title] = "owned by hugpy_agent.mct.manifest"
                continue
            target, why = _resolve(module_name, attr)
            if target is None:
                missing[title] = why
                continue
            try:
                schemas[title] = json_schema_for(target, title=title)
            except Exception as exc:  # noqa: BLE001
                missing[title] = (f"schema generation failed "
                                  f"({type(exc).__name__}: {exc})")
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "generated_by": "hugpy_oracle.schema_export",
        "platform_contracts": [name for name, _m, _a in PLATFORM_CONTRACTS],
        "schemas": {k: schemas[k] for k in sorted(schemas)},
        "missing": {k: missing[k] for k in sorted(missing)},
    }


def export_json(indent: int = 2) -> str:
    """``export_all()`` as stable JSON text (sorted keys) — the form a docs
    build or a client generator consumes."""
    import json
    return json.dumps(export_all(), indent=indent, sort_keys=True)


__all__ = [
    "DOMAIN_ARTIFACTS",
    "JSON_SCHEMA_DIALECT",
    "PLATFORM_CONTRACTS",
    "WIRE_OVERRIDES",
    "export_all",
    "export_json",
    "json_schema_for",
]
