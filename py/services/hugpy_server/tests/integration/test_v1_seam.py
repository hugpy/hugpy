"""Offline unit tests for the OpenAI-compatible /v1 seam (v1_helpers.py).

The helpers under test are deliberately stdlib-only
(hugpy_server.app.routes.v1_helpers). No network, no Flask app, no model
weights.
"""
import json
import unittest

from hugpy_server.app.routes import v1_helpers

_completion_kwargs = v1_helpers._completion_kwargs
_usage_block = v1_helpers._usage_block
_build_tools_preamble = v1_helpers._build_tools_preamble
_inject_tools_preamble = v1_helpers._inject_tools_preamble
_parse_tool_calls = v1_helpers._parse_tool_calls
_render_tool_messages = v1_helpers._render_tool_messages
_render_assistant_tool_calls = v1_helpers._render_assistant_tool_calls
_fleet_status_text = v1_helpers._fleet_status_text


def _payload(**over):
    base = {"model": "test-model",
            "messages": [{"role": "user", "content": "hi"}]}
    base.update(over)
    return base


class TestCompletionKwargs(unittest.TestCase):
    def test_messages_required(self):
        with self.assertRaises(ValueError):
            _completion_kwargs({"model": "x"})

    def test_default_model_means_unset(self):
        for name in ("", "default", "DEFAULT", "  Default "):
            kw = _completion_kwargs(_payload(model=name))
            self.assertIsNone(kw["model_key"], name)

    def test_explicit_model_forwarded(self):
        kw = _completion_kwargs(_payload())
        self.assertEqual(kw["model_key"], "test-model")
        self.assertTrue(kw["request_id"].startswith("v1-"))

    def test_max_chunks_forwarded_when_present(self):
        kw = _completion_kwargs(_payload(max_chunks=5))
        self.assertEqual(kw["max_chunks"], 5)

    def test_max_chunks_defaults_to_1_when_max_tokens_set(self):
        # OpenAI max_tokens semantics: one bounded completion, no server-side
        # auto-continuation — the 2026-07-14 stall fix.
        kw = _completion_kwargs(_payload(max_tokens=256))
        self.assertEqual(kw["max_new_tokens"], 256)
        self.assertEqual(kw["max_chunks"], 1)

    def test_max_completion_tokens_alias_also_defaults_max_chunks(self):
        kw = _completion_kwargs(_payload(max_completion_tokens=128))
        self.assertEqual(kw["max_new_tokens"], 128)
        self.assertEqual(kw["max_chunks"], 1)

    def test_explicit_max_chunks_wins_over_default(self):
        kw = _completion_kwargs(_payload(max_tokens=256, max_chunks=4))
        self.assertEqual(kw["max_chunks"], 4)

    def test_max_chunks_not_set_when_max_tokens_absent(self):
        # No cap -> preserve today's unbounded auto-continuation (console path).
        kw = _completion_kwargs(_payload())
        self.assertNotIn("max_chunks", kw)
        self.assertNotIn("max_new_tokens", kw)

    def test_temperature_maps_do_sample(self):
        kw = _completion_kwargs(_payload(temperature=0.7))
        self.assertEqual(kw["temperature"], 0.7)
        self.assertTrue(kw["do_sample"])
        kw = _completion_kwargs(_payload(temperature=0))
        self.assertEqual(kw["temperature"], 0.0)
        self.assertFalse(kw["do_sample"])

    def test_temperature_absent_not_forwarded(self):
        kw = _completion_kwargs(_payload())
        self.assertNotIn("temperature", kw)
        self.assertNotIn("do_sample", kw)

    def test_top_p_forwarded_optionally(self):
        self.assertNotIn("top_p", _completion_kwargs(_payload()))
        kw = _completion_kwargs(_payload(top_p=0.9))
        self.assertEqual(kw["top_p"], 0.9)

    def test_unbounded_forwarded_optionally(self):
        self.assertNotIn("unbounded", _completion_kwargs(_payload()))
        kw = _completion_kwargs(_payload(unbounded=True))
        self.assertIs(kw["unbounded"], True)
        kw = _completion_kwargs(_payload(unbounded=False))
        self.assertIs(kw["unbounded"], False)

    def test_unknown_openai_fields_never_forwarded(self):
        # ChatRequest is extra="forbid": tools/stream/etc. must stay route-local.
        kw = _completion_kwargs(_payload(
            tools=[{"type": "function", "function": {"name": "f"}}],
            tool_choice="auto", stream=True, n=1, stop=["x"],
            stream_options={"include_usage": True},
        ))
        for banned in ("tools", "tool_choice", "stream", "n", "stop",
                       "stream_options"):
            self.assertNotIn(banned, kw)


class TestUsageBlock(unittest.TestCase):
    def test_real_counts_pass_through(self):
        u = _usage_block({"prompt_tokens": 10, "completion_tokens": 5,
                          "total_tokens": 15})
        self.assertEqual(u, {"prompt_tokens": 10, "completion_tokens": 5,
                             "total_tokens": 15})

    def test_total_derived_when_missing(self):
        u = _usage_block({"prompt_tokens": 10, "completion_tokens": 5})
        self.assertEqual(u["total_tokens"], 15)

    def test_none_and_garbage_degrade_to_all_none(self):
        for bad in (None, {}, "usage", 42, {"prompt_tokens": "ten"}):
            u = _usage_block(bad)
            self.assertEqual(u, {"prompt_tokens": None,
                                 "completion_tokens": None,
                                 "total_tokens": None}, bad)

    def test_partial_counts_kept_without_fake_total(self):
        u = _usage_block({"completion_tokens": 7})
        self.assertEqual(u, {"prompt_tokens": None, "completion_tokens": 7,
                             "total_tokens": None})


_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}
_TIME = {
    "type": "function",
    "function": {"name": "get_time", "description": "Current time",
                 "parameters": {"type": "object", "properties": {}}},
}

# The engine's auto-continuation prompt, verbatim from dispatch.py — the
# string that leaked into reply bodies in the 2026-07-14 incident.
_LEAK = "Continue exactly where you left off. Do not repeat any previous text."


class TestBuildToolsPreamble(unittest.TestCase):
    def test_includes_each_tool_name_and_schema(self):
        pre = _build_tools_preamble([_WEATHER, _TIME])
        self.assertIn("get_weather", pre)
        self.assertIn("get_time", pre)
        self.assertIn('"city"', pre)                 # JSON-schema rendered
        self.assertIn("<tools>", pre)
        self.assertIn("<tool_call>", pre)            # calling convention shown

    def test_tool_choice_none_skips_injection(self):
        self.assertIsNone(_build_tools_preamble([_WEATHER], "none"))

    def test_empty_or_junk_tools_skip_injection(self):
        self.assertIsNone(_build_tools_preamble([]))
        self.assertIsNone(_build_tools_preamble(None))
        self.assertIsNone(_build_tools_preamble([{"type": "function"}]))
        # non-list garbage must degrade to a plain completion, never raise
        self.assertIsNone(_build_tools_preamble("get_weather"))
        self.assertIsNone(_build_tools_preamble({"name": "x"}))
        self.assertIsNone(_build_tools_preamble(42))

    def test_single_function_choice_narrows_and_forces(self):
        pre = _build_tools_preamble(
            [_WEATHER, _TIME], {"type": "function",
                                "function": {"name": "get_time"}})
        self.assertIn("get_time", pre)
        self.assertNotIn("get_weather", pre)
        self.assertIn("MUST call", pre)

    def test_forced_unknown_function_means_plain_completion(self):
        self.assertIsNone(_build_tools_preamble(
            [_WEATHER], {"type": "function", "function": {"name": "nope"}}))


class TestInjectToolsPreamble(unittest.TestCase):
    def test_prepends_system_message_when_none(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = _inject_tools_preamble(msgs, "PREAMBLE")
        self.assertEqual(out[0], {"role": "system", "content": "PREAMBLE"})
        self.assertEqual(out[1]["content"], "hi")
        self.assertEqual(msgs, [{"role": "user", "content": "hi"}])  # no mutation

    def test_merges_into_existing_system_message(self):
        out = _inject_tools_preamble(
            [{"role": "system", "content": "be terse"},
             {"role": "user", "content": "hi"}], "PREAMBLE")
        self.assertEqual(len(out), 2)
        self.assertIn("be terse", out[0]["content"])
        self.assertIn("PREAMBLE", out[0]["content"])


class TestParseToolCalls(unittest.TestCase):
    def test_single_block(self):
        text = ('<tool_call>\n{"name": "get_weather", '
                '"arguments": {"city": "Berlin"}}\n</tool_call>')
        clean, calls = _parse_tool_calls(text)
        self.assertEqual(clean, "")
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertTrue(call["id"].startswith("call_"))
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "get_weather")
        # arguments must be a JSON *string* per OpenAI shape
        self.assertEqual(json.loads(call["function"]["arguments"]),
                         {"city": "Berlin"})

    def test_multiple_blocks(self):
        text = ('<tool_call>{"name": "a", "arguments": {}}</tool_call>\n'
                '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>')
        clean, calls = _parse_tool_calls(text)
        self.assertEqual([c["function"]["name"] for c in calls], ["a", "b"])
        self.assertEqual(clean, "")

    def test_prose_around_block_is_stripped_into_clean_text(self):
        text = ('I will check.\n<tool_call>{"name": "get_time", '
                '"arguments": {}}</tool_call>\nDone.')
        clean, calls = _parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("<tool_call>", clean)
        self.assertIn("I will check.", clean)

    def test_bare_json_tolerance(self):
        text = '{"name": "get_weather", "arguments": {"city": "Oslo"}}'
        clean, calls = _parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(clean, "")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")

    def test_double_encoded_arguments_unwrapped(self):
        text = ('<tool_call>{"name": "f", "arguments": '
                '"{\\"k\\": \\"v\\"}"}</tool_call>')
        _clean, calls = _parse_tool_calls(text)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"k": "v"})

    def test_malformed_block_returns_original_text(self):
        text = "<tool_call>{not json at all</tool_call>"
        clean, calls = _parse_tool_calls(text)
        self.assertIsNone(calls)
        self.assertEqual(clean, text)

    def test_plain_answer_returns_original_text(self):
        text = "The weather in Berlin is sunny."
        clean, calls = _parse_tool_calls(text)
        self.assertIsNone(calls)
        self.assertEqual(clean, text)

    def test_continuation_leak_does_not_fake_a_tool_call(self):
        clean, calls = _parse_tool_calls(f"Here is part one. {_LEAK} And more.")
        self.assertIsNone(calls)
        self.assertIn("part one", clean)

    def test_leak_inside_block_is_scrubbed_before_parse(self):
        text = ('<tool_call>{"name": "get_time", ' + _LEAK +
                '"arguments": {}}</tool_call>')
        _clean, calls = _parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_time")

    def test_prose_json_with_name_key_is_not_a_bare_call(self):
        # Stricter than the fenced path: bare objects need "arguments" too.
        text = 'The config is {"name": "server-1", "port": 80}.'
        clean, calls = _parse_tool_calls(text)
        self.assertIsNone(calls)
        self.assertEqual(clean, text)

    def test_empty_and_none_inputs(self):
        self.assertEqual(_parse_tool_calls(""), ("", None))
        self.assertEqual(_parse_tool_calls(None), ("", None))


class TestRenderToolMessages(unittest.TestCase):
    """The inbound half of the tools shim: OpenAI tool-message shapes rendered
    into the <tool_call>/<tool_response> role+content wire the model reads."""

    _ECHO = {
        "role": "assistant", "content": None,
        "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "get_weather",
                         "arguments": '{"city": "Berlin"}'},
        }],
    }
    _RESULT = {"role": "tool", "tool_call_id": "call_1",
               "content": '{"temp_c": 21, "sky": "clear"}'}

    def test_plain_chat_is_byte_for_byte_unchanged(self):
        # THE regression to guard hardest: no tool fields -> same downcast as
        # the old `{"role": ..., "content": ...}` comprehension produced.
        msgs = [{"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}]
        self.assertEqual(_render_tool_messages(msgs), msgs)

    def test_missing_content_becomes_empty_string(self):
        out = _render_tool_messages([{"role": "user"}])
        self.assertEqual(out, [{"role": "user", "content": ""}])

    def test_role_defaults_to_user(self):
        out = _render_tool_messages([{"content": "hi"}])
        self.assertEqual(out, [{"role": "user", "content": "hi"}])

    def test_tool_result_rendered_as_user_tool_response(self):
        out = _render_tool_messages([self._RESULT])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["role"], "user")   # NOT "tool" — template-safe
        self.assertIn("<tool_response>", out[0]["content"])
        self.assertIn("</tool_response>", out[0]["content"])
        self.assertIn('"temp_c": 21', out[0]["content"])

    def test_assistant_tool_calls_rendered_as_tool_call_block(self):
        out = _render_tool_messages([self._ECHO])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["role"], "assistant")
        content = out[0]["content"]
        self.assertIn("<tool_call>", content)
        self.assertIn("</tool_call>", content)
        # arguments unwrapped from the OpenAI JSON *string* back to an object,
        # matching what the model itself emits (and what _parse_tool_calls reads)
        self.assertIn('"name": "get_weather"', content)
        clean, calls = _parse_tool_calls(content)   # round-trips through the parser
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"city": "Berlin"})

    def test_assistant_prose_preserved_alongside_calls(self):
        msg = dict(self._ECHO, content="Let me check that.")
        out = _render_tool_messages([msg])
        self.assertIn("Let me check that.", out[0]["content"])
        self.assertIn("<tool_call>", out[0]["content"])

    def test_full_four_message_loop_renders_to_prompt_text(self):
        # user -> assistant(tool_call) -> tool(result) -> [next assistant turn]
        loop = [
            {"role": "user", "content": "weather in Berlin?"},
            self._ECHO,
            self._RESULT,
        ]
        out = _render_tool_messages(loop)
        # every message is now a plain role+content str dict (frozen-schema wire)
        for m in out:
            self.assertEqual(set(m.keys()), {"role", "content"})
            self.assertIn(m["role"], ("system", "user", "assistant"))
            self.assertIsInstance(m["content"], str)
        self.assertEqual(out[0]["content"], "weather in Berlin?")   # untouched
        self.assertIn("<tool_call>", out[1]["content"])
        self.assertIn("<tool_response>", out[2]["content"])
        self.assertIn("21", out[2]["content"])                     # tool output visible

    def test_dict_arguments_also_accepted(self):
        # some clients echo `arguments` already as an object, not a JSON string
        msg = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c", "type": "function",
            "function": {"name": "f", "arguments": {"k": "v"}}}]}
        out = _render_tool_messages([msg])
        _clean, calls = _parse_tool_calls(out[0]["content"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"k": "v"})

    def test_input_not_mutated(self):
        loop = [dict(self._ECHO), dict(self._RESULT)]
        snapshot = json.dumps(loop, sort_keys=True)
        _render_tool_messages(loop)
        self.assertEqual(json.dumps(loop, sort_keys=True), snapshot)

    def test_completion_kwargs_routes_through_renderer(self):
        # the seam wiring: _completion_kwargs must emit the RENDERED messages,
        # not the raw tool shapes (the old downcast silently dropped them).
        kw = _completion_kwargs(_payload(messages=[
            {"role": "user", "content": "weather?"},
            self._ECHO, self._RESULT,
        ]))
        self.assertIn("<tool_call>", kw["messages"][1]["content"])
        self.assertIn("<tool_response>", kw["messages"][2]["content"])


class TestRenderAssistantToolCalls(unittest.TestCase):
    def test_multiple_calls_each_get_a_block(self):
        text = _render_assistant_tool_calls([
            {"function": {"name": "a", "arguments": "{}"}},
            {"function": {"name": "b", "arguments": '{"x": 1}'}},
        ])
        self.assertEqual(text.count("<tool_call>"), 2)
        self.assertIn('"name": "a"', text)
        self.assertIn('"name": "b"', text)

    def test_nameless_or_garbage_entries_skipped(self):
        text = _render_assistant_tool_calls([
            {"function": {"arguments": "{}"}},   # no name
            "not a dict",
            {"function": {"name": "ok", "arguments": "{}"}},
        ])
        self.assertEqual(text.count("<tool_call>"), 1)
        self.assertIn('"name": "ok"', text)

    def test_empty_or_none_is_empty_string(self):
        self.assertEqual(_render_assistant_tool_calls(None), "")
        self.assertEqual(_render_assistant_tool_calls([]), "")


class TestFleetStatusText(unittest.TestCase):
    def test_load_resources_are_explicit(self):
        text = _fleet_status_text({"stage": "loading", "model_key": "m",
                                   "worker_name": "gpu-a", "gpu_bytes": 3 * 2**30,
                                   "ram_bytes": 4 * 2**30})
        self.assertIn("m on gpu-a", text)
        self.assertIn("3.0 GiB to VRAM & 4.0 GiB to RAM", text)

    def test_missing_telemetry_is_never_guessed(self):
        text = _fleet_status_text({"stage": "loading", "model_key": "m"})
        self.assertIn("unknown worker", text)
        self.assertIn("unknown to VRAM", text)

