"""KNOWN-GOOD CONTRACT — /v1 intake validates and forwards response_format.

Source: hugpy_server.app.routes.v1_helpers.normalize_response_format /
_completion_kwargs. Established 2026-09-29 (the reducer's JSON mode was
silently dropped at intake before). Deterministic, pure functions.
"""
from __future__ import annotations

import pytest

from hugpy_server.app.routes import v1_helpers as h

MSGS = [{"role": "user", "content": "hi"}]


def test_json_modes_are_forwarded_text_is_default_and_bad_shapes_are_400():
    """INVARIANT: json_object and json_schema (with json_schema.schema) ride
    into prompt_kwargs['response_format']; 'text' and absence forward nothing;
    an unknown type or a schema-less json_schema raises ValueError (the route
    maps it to a 400) — never a silent drop. Established: 2026-09-29."""
    assert "response_format" not in h._completion_kwargs({"messages": MSGS})
    assert "response_format" not in h._completion_kwargs(
        {"messages": MSGS, "response_format": {"type": "text"}})
    assert h._completion_kwargs({"messages": MSGS, "response_format": {"type": "json_object"}}
                                )["response_format"] == {"type": "json_object"}
    out = h._completion_kwargs({"messages": MSGS, "response_format": {
        "type": "json_schema", "json_schema": {"name": "s", "strict": True,
                                               "schema": {"type": "object"}}}})
    assert out["response_format"] == {"type": "json_schema", "json_schema": {
        "schema": {"type": "object"}, "name": "s", "strict": True}}
    for bad in ("json_object", {"type": "grammar"}, {"type": "json_schema"},
                {"type": "json_schema", "json_schema": {"schema": "x"}}):
        with pytest.raises(ValueError):
            h._completion_kwargs({"messages": MSGS, "response_format": bad})
