"""CHAT TOOL-MESSAGE SCHEMA — the OpenAI tool-calling loop (step 2+) accepted by
``ChatMessage`` / ``ChatRequest`` (live protocol break, keeper-confirmed
2026-07-17).

Step 1 (POST with ``tools``) already worked; step 2 died because ``ChatMessage``
was ``extra="forbid"`` with only ``role``+``content`` and ``ROLES`` had no
``tool``, so the protocol-correct follow-up — the assistant turn echoed back
WITH its ``tool_calls`` array plus a ``{"role":"tool","tool_call_id":…}`` result
— was rejected (role literal violation + extra_forbidden).

The fix ADDS the two OpenAI fields explicitly (schema stays ``extra="forbid"``)
and appends ``tool`` to the ROLES default. This is a PURE-SCHEMA test (no bus /
no worker / no GPU): it exercises the LIVE ``chat_schemas`` module through the
venv, asserting the three previously-failing constructions now validate, that a
genuinely-unknown key is STILL forbidden, and that plain chat is unchanged.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python -m pytest tests/test_chat_tool_messages_schema.py -q
  venv/bin/python tests/test_chat_tool_messages_schema.py
"""
from __future__ import annotations

import unittest

from hugpy_engine.schemas.chat_schemas import ChatMessage, ChatRequest

_ECHO = {
    "role": "assistant",
    "content": None,
    "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Berlin"}'},
    }],
}
_RESULT = {"role": "tool", "tool_call_id": "call_1", "content": "72F, sunny"}


class TestChatMessageToolFields(unittest.TestCase):
    def test_role_tool_accepted(self):
        m = ChatMessage(role="tool", tool_call_id="call_1", content="72F")
        self.assertEqual(m.role, "tool")

    def test_tool_call_id_accepted(self):
        m = ChatMessage(**_RESULT)
        self.assertEqual(m.tool_call_id, "call_1")

    def test_assistant_tool_calls_echo_back_accepted(self):
        m = ChatMessage(**_ECHO)
        self.assertEqual(m.role, "assistant")
        self.assertIsNone(m.content)                 # content optional now
        self.assertEqual(len(m.tool_calls), 1)

    def test_genuinely_unknown_key_still_forbidden(self):
        # the fix ADDS named fields — it does NOT open the schema.
        with self.assertRaises(Exception):
            ChatMessage(role="user", content="hi", not_a_real_field=1)

    def test_plain_chat_unchanged(self):
        # role+content still the required core; defaults leave tool fields null.
        m = ChatMessage(role="user", content="hi")
        self.assertEqual(m.role, "user")
        self.assertEqual(m.content, "hi")
        self.assertIsNone(m.tool_calls)
        self.assertIsNone(m.tool_call_id)

    def test_content_still_required_absent_tool_calls(self):
        # a bare message with neither content nor tool fields still needs content
        # (default "" keeps the historical shape; None is only meaningful with
        # tool_calls). We only assert construction doesn't crash on the default.
        m = ChatMessage(role="user")
        self.assertEqual(m.content, "")


class TestChatRequestToolLoop(unittest.TestCase):
    def test_four_message_tool_loop_validates(self):
        req = ChatRequest(messages=[
            {"role": "user", "content": "weather in Berlin?"},
            _ECHO,
            _RESULT,
        ])
        self.assertEqual(len(req.messages), 3)
        self.assertEqual(req.messages[1].role, "assistant")
        self.assertEqual(req.messages[2].role, "tool")

    def test_plain_request_unchanged(self):
        req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(len(req.messages), 1)
        self.assertEqual(req.messages[0].content, "hi")


if __name__ == "__main__":
    unittest.main(verbosity=2)


def test_plain_message_dump_is_frozen_wire_shape():
    """Relay-boundary regression (keeper, 2026-07-17): workers on the released
    package re-validate the relayed payload with a role+content-only,
    extra="forbid" ChatMessage. A plain message must therefore dump to EXACTLY
    {role, content} — None tool keys in the dump broke every offloaded chat."""
    from hugpy_engine.schemas.chat_schemas import ChatMessage, ChatRequest
    assert ChatMessage(role="user", content="hi").model_dump() == {
        "role": "user", "content": "hi"}
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
    dumped = req.model_dump()["messages"]
    assert dumped == [{"role": "user", "content": "hi"}]


def test_tool_fields_survive_dump_when_set():
    from hugpy_engine.schemas.chat_schemas import ChatMessage
    m = ChatMessage(role="tool", tool_call_id="call_1", content="42")
    assert m.model_dump() == {"role": "tool", "content": "42",
                              "tool_call_id": "call_1"}
