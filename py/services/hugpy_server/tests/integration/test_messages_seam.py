"""Offline unit tests for the Anthropic Messages shim's translation +
streaming encoder (messages_helpers.py).

Like tests/test_v1_seam.py, the helper is deliberately stdlib-only. The
StreamingThinkSplitter it needs at stream time lives in
``hugpy_engine.utils.no_think`` (also stdlib-only) and is passed in.
"""
import json
import unittest

from hugpy_server.app.routes import messages_helpers as mh
from hugpy_engine.utils.no_think import StreamingThinkSplitter


# ──────────────────────────────────────────────────────────────────────────
# request translation
# ──────────────────────────────────────────────────────────────────────────
class TestRequestTranslation(unittest.TestCase):
    def test_system_string_folds_to_leading_system_message(self):
        body = {"model": "m", "max_tokens": 16, "system": "be brief",
                "messages": [{"role": "user", "content": "hi"}]}
        p = mh.anthropic_to_openai_payload(body)
        self.assertEqual(p["messages"][0], {"role": "system", "content": "be brief"})
        self.assertEqual(p["messages"][1], {"role": "user", "content": "hi"})
        self.assertEqual(p["max_tokens"], 16)
        self.assertEqual(p["model"], "m")

    def test_system_block_list_joins(self):
        body = {"model": "m", "max_tokens": 16,
                "system": [{"type": "text", "text": "a"},
                           {"type": "text", "text": "b"}],
                "messages": [{"role": "user", "content": "hi"}]}
        p = mh.anthropic_to_openai_payload(body)
        self.assertEqual(p["messages"][0]["content"], "a\n\nb")

    def test_text_block_list_flattens_to_string(self):
        body = {"model": "m", "max_tokens": 16,
                "messages": [{"role": "user",
                              "content": [{"type": "text", "text": "one"},
                                          {"type": "text", "text": "two"}]}]}
        p = mh.anthropic_to_openai_payload(body)
        self.assertEqual(p["messages"][0]["content"], "one\ntwo")

    def test_image_block_becomes_openai_image_url(self):
        body = {"model": "m", "max_tokens": 16,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/png",
                                                 "data": "AAA"}}]}]}
        p = mh.anthropic_to_openai_payload(body)
        parts = p["messages"][0]["content"]
        self.assertIsInstance(parts, list)
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/png;base64,AAA"))

    def test_tool_use_and_result_roundtrip(self):
        body = {"model": "m", "max_tokens": 16, "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "Paris"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": "72F"}]},
        ]}
        p = mh.anthropic_to_openai_payload(body)
        asst = p["messages"][1]
        self.assertEqual(asst["role"], "assistant")
        self.assertEqual(asst["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(asst["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(asst["tool_calls"][0]["function"]["arguments"]),
                         {"city": "Paris"})
        tool_msg = p["messages"][2]
        self.assertEqual(tool_msg, {"role": "tool", "tool_call_id": "toolu_1",
                                    "content": "72F"})

    def test_tools_and_tool_choice_map(self):
        body = {"model": "m", "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "f", "description": "d",
                           "input_schema": {"type": "object", "properties": {}}}],
                "tool_choice": {"type": "any"}}
        p = mh.anthropic_to_openai_payload(body)
        self.assertEqual(p["tools"][0]["type"], "function")
        self.assertEqual(p["tools"][0]["function"]["name"], "f")
        self.assertEqual(p["tools"][0]["function"]["parameters"],
                         {"type": "object", "properties": {}})
        self.assertEqual(p["tool_choice"], "required")

    def test_tool_choice_specific_tool(self):
        self.assertEqual(
            mh.anthropic_tool_choice_to_openai({"type": "tool", "name": "x"}),
            {"type": "function", "function": {"name": "x"}})
        self.assertEqual(mh.anthropic_tool_choice_to_openai({"type": "auto"}), "auto")

    def test_stop_sequences_and_sampling(self):
        body = {"model": "m", "max_tokens": 16, "temperature": 0.5, "top_p": 0.9,
                "stop_sequences": ["X"],
                "messages": [{"role": "user", "content": "hi"}]}
        p = mh.anthropic_to_openai_payload(body)
        self.assertEqual(p["stop"], ["X"])
        self.assertEqual(p["temperature"], 0.5)
        self.assertEqual(p["top_p"], 0.9)

    def test_claude_model_ids_alias_to_default(self):
        # Claude Code sends its own ids; they must fall through to the served
        # brain, not 404 — real hugpy keys pass through untouched.
        for mid in ("claude-sonnet-5", "claude-haiku-4-5-20251001", " Claude-3 "):
            body = {"model": mid, "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}]}
            self.assertEqual(mh.anthropic_to_openai_payload(body)["model"],
                             "default", mid)
        body = {"model": "Qwen~Qwen3-Coder-Next-GGUF", "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(mh.anthropic_to_openai_payload(body)["model"],
                         "Qwen~Qwen3-Coder-Next-GGUF")
        self.assertIsNone(mh.resolve_client_model(None))

    def test_missing_messages_raises(self):
        with self.assertRaises(ValueError):
            mh.anthropic_to_openai_payload({"model": "m", "max_tokens": 16})

    def test_bad_max_tokens_raises(self):
        with self.assertRaises(ValueError):
            mh.anthropic_to_openai_payload(
                {"model": "m", "max_tokens": 0,
                 "messages": [{"role": "user", "content": "hi"}]})


# ──────────────────────────────────────────────────────────────────────────
# response translation
# ──────────────────────────────────────────────────────────────────────────
class TestFinishAndUsage(unittest.TestCase):
    def test_finish_map(self):
        self.assertEqual(mh.finish_to_stop_reason("length"), "max_tokens")
        self.assertEqual(mh.finish_to_stop_reason("stop"), "end_turn")
        self.assertEqual(mh.finish_to_stop_reason("tool_calls"), "tool_use")
        self.assertEqual(mh.finish_to_stop_reason("stop", had_tool_calls=True),
                         "tool_use")

    def test_usage_map_and_zero_fallback(self):
        self.assertEqual(
            mh.usage_to_anthropic({"prompt_tokens": 10, "completion_tokens": 5}),
            {"input_tokens": 10, "output_tokens": 5})
        self.assertEqual(
            mh.usage_to_anthropic({"prompt_tokens": None, "completion_tokens": None}),
            {"input_tokens": 0, "output_tokens": 0})
        self.assertEqual(mh.usage_to_anthropic(None),
                         {"input_tokens": 0, "output_tokens": 0})

    def test_content_blocks_thinking_first_then_text_then_tool(self):
        tool_calls = [{"id": "call_1", "type": "function",
                       "function": {"name": "f",
                                    "arguments": json.dumps({"a": 1})}}]
        blocks = mh.build_content_blocks("reasoned", "answer", tool_calls)
        self.assertEqual([b["type"] for b in blocks],
                         ["thinking", "text", "tool_use"])
        self.assertEqual(blocks[0]["thinking"], "reasoned")
        self.assertEqual(blocks[1]["text"], "answer")
        self.assertEqual(blocks[2]["name"], "f")
        self.assertEqual(blocks[2]["input"], {"a": 1})

    def test_content_blocks_never_empty(self):
        blocks = mh.build_content_blocks("", "", None)
        self.assertEqual(blocks, [{"type": "text", "text": ""}])


# ──────────────────────────────────────────────────────────────────────────
# streaming event sequence
# ──────────────────────────────────────────────────────────────────────────
def _decode(frames):
    """[(event_type, data_dict), ...] from raw SSE byte frames."""
    out = []
    for fr in frames:
        s = fr.decode("utf-8")
        etype = None
        data = None
        for line in s.splitlines():
            if line.startswith("event: "):
                etype = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        out.append((etype, data))
    return out


class TestStreamingEncoder(unittest.TestCase):
    def _run(self, tokens):
        enc = mh.AnthropicStreamEncoder("msg_x", "m", StreamingThinkSplitter(),
                                        input_tokens=7)
        frames = list(enc.start())
        for tok in tokens:
            frames.extend(enc.feed(tok))
        frames.extend(enc.finish("end_turn", output_tokens=3))
        return _decode(frames)

    def test_plain_text_sequence(self):
        events = self._run(["Hel", "lo"])
        types = [e for e, _ in events]
        self.assertEqual(types, [
            "message_start", "ping",
            "content_block_start", "content_block_delta", "content_block_delta",
            "content_block_stop",
            "message_delta", "message_stop",
        ])
        # message_start carries input_tokens and echoes the model.
        self.assertEqual(events[0][1]["message"]["usage"]["input_tokens"], 7)
        self.assertEqual(events[0][1]["message"]["model"], "m")
        # text block at index 0
        self.assertEqual(events[2][1]["index"], 0)
        self.assertEqual(events[2][1]["content_block"], {"type": "text", "text": ""})
        self.assertEqual(events[3][1]["delta"],
                         {"type": "text_delta", "text": "Hel"})
        self.assertEqual(events[6][1]["delta"]["stop_reason"], "end_turn")
        self.assertEqual(events[6][1]["usage"]["output_tokens"], 3)

    def test_think_then_text_split_with_indices(self):
        # A re-inlined <think> block streamed across tokens: reasoning must open
        # a thinking block at index 0, the answer a text block at index 1, each
        # closed before the next opens.
        events = self._run(["<think>rea", "soning</think>ans", "wer"])
        types = [e for e, _ in events]
        self.assertEqual(types, [
            "message_start", "ping",
            "content_block_start", "content_block_delta", "content_block_delta",
            "content_block_stop",           # thinking closes
            "content_block_start", "content_block_delta", "content_block_delta",
            "content_block_stop",           # text closes
            "message_delta", "message_stop",
        ])
        # thinking block index 0
        self.assertEqual(events[2][1]["content_block"]["type"], "thinking")
        self.assertEqual(events[2][1]["index"], 0)
        self.assertEqual(events[3][1]["delta"]["type"], "thinking_delta")
        self.assertEqual(events[3][1]["delta"]["thinking"], "rea")
        self.assertEqual(events[4][1]["delta"]["thinking"], "soning")
        self.assertEqual(events[5][1]["index"], 0)
        # text block index 1
        self.assertEqual(events[6][1]["content_block"]["type"], "text")
        self.assertEqual(events[6][1]["index"], 1)
        self.assertEqual(events[7][1]["delta"]["type"], "text_delta")
        # reconstruct the answer
        answer = "".join(e[1]["delta"]["text"] for e in events
                         if e[0] == "content_block_delta"
                         and e[1]["delta"]["type"] == "text_delta")
        self.assertEqual(answer, "answer")

    def test_unclosed_think_drains_as_thinking(self):
        # Budget ran out mid-thought: no closing tag -> all reasoning, no text.
        events = self._run(["<think>never closed"])
        types = [e for e, _ in events]
        self.assertIn("content_block_start", types)
        # the only block is thinking
        starts = [d for t, d in events if t == "content_block_start"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]["content_block"]["type"], "thinking")

