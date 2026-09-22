"""Pure helpers for the Anthropic Messages API shim (messages_routes.py).

Claude Code (and the Claude Agent SDK) speak Anthropic's `/v1/messages`
protocol; hugpy's engine speaks OpenAI's `/v1/chat/completions`. This module is
the TRANSLATION layer between the two — Anthropic request -> OpenAI payload on
the way in, engine events -> Anthropic response/SSE on the way out.

Deliberately stdlib-only (json/re/uuid) with NO Flask, no package imports and
no import of the engine path, exactly like v1_helpers.py — so it unit-tests
standalone by FILE path (tests/test_messages_seam.py loads it directly). The one
collaborator it needs at stream time, the `StreamingThinkSplitter`, is PASSED IN
by the route rather than imported here, keeping this file loadable without the
package __init__ chain.

The route builds the SAME OpenAI-style `payload` dict `v1_chat_completions`
builds, runs it through the identical
`_build_tools_preamble`/`_inject_tools_preamble` -> `_completion_kwargs` ->
`_v1_events` pipeline, then formats the OUTPUT here as Anthropic.
"""
from __future__ import annotations

import json
import uuid


# ──────────────────────────────────────────────────────────────────────────
# ids
# ──────────────────────────────────────────────────────────────────────────
def new_message_id() -> str:
    return "msg_" + uuid.uuid4().hex


def new_tool_use_id() -> str:
    return "toolu_" + uuid.uuid4().hex[:24]


# ──────────────────────────────────────────────────────────────────────────
# request translation: Anthropic Messages -> OpenAI chat.completions payload
# ──────────────────────────────────────────────────────────────────────────
def _system_to_text(system) -> str:
    """Anthropic top-level `system` (a string OR a list of text blocks) -> text.

    A list is the block form `[{"type":"text","text":...}, ...]`; anything
    without a usable `text` is skipped rather than crashing the request.
    """
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for blk in system:
            if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                parts.append(blk["text"])
            elif isinstance(blk, str):
                parts.append(blk)
        return "\n\n".join(p for p in parts if p)
    return str(system)


def _image_block_to_openai(block: dict):
    """Anthropic image block -> OpenAI `image_url` content part, or None.

    base64 source -> a `data:` URI; url source -> the URL verbatim. Whether the
    image ultimately reaches the model is the engine path's concern (the same
    limitation /v1 has); this produces the correct OpenAI shape regardless.
    """
    src = block.get("source") or {}
    stype = src.get("type")
    if stype == "base64":
        media = src.get("media_type") or "image/png"
        data = src.get("data") or ""
        return {"type": "image_url",
                "image_url": {"url": f"data:{media};base64,{data}"}}
    if stype == "url" and src.get("url"):
        return {"type": "image_url", "image_url": {"url": src["url"]}}
    return None


def _content_to_openai(content):
    """A single Anthropic message `content` -> OpenAI content (str | list).

    Returns a tuple ``(openai_content, tool_uses, tool_results)`` where
    tool_uses / tool_results are extracted structural blocks handled by the
    caller (they become assistant `tool_calls` / `{"role":"tool"}` messages).
    A text-only block list is flattened to a plain string; a list carrying an
    image becomes OpenAI content parts.
    """
    if content is None:
        return "", [], []
    if isinstance(content, str):
        return content, [], []
    if not isinstance(content, list):
        return str(content), [], []

    text_parts: list[str] = []
    parts: list = []          # OpenAI content parts, used only if an image appears
    has_image = False
    tool_uses: list = []
    tool_results: list = []
    for blk in content:
        if not isinstance(blk, dict):
            text_parts.append(str(blk))
            parts.append({"type": "text", "text": str(blk)})
            continue
        btype = blk.get("type")
        if btype == "text":
            t = blk.get("text") or ""
            text_parts.append(t)
            parts.append({"type": "text", "text": t})
        elif btype == "image":
            img = _image_block_to_openai(blk)
            if img is not None:
                has_image = True
                parts.append(img)
        elif btype == "tool_use":
            tool_uses.append(blk)
        elif btype == "tool_result":
            tool_results.append(blk)
        # unknown block types are ignored (never a 500)

    if has_image:
        return parts, tool_uses, tool_results
    return "\n".join(text_parts), tool_uses, tool_results


def _tool_result_content_to_text(content) -> str:
    """Anthropic tool_result `content` (str | list of blocks) -> plain text.

    OpenAI `{"role":"tool"}` content is a string, so a block list is flattened;
    non-text blocks (rare in a tool result) degrade to their JSON.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for blk in content:
            if isinstance(blk, dict):
                if isinstance(blk.get("text"), str):
                    out.append(blk["text"])
                else:
                    out.append(json.dumps(blk, ensure_ascii=False))
            else:
                out.append(str(blk))
        return "\n".join(out)
    return str(content)


def anthropic_messages_to_openai(messages) -> list:
    """Anthropic `messages` -> OpenAI `messages`.

    * assistant `tool_use` blocks -> an assistant turn with OpenAI `tool_calls`
      (arguments JSON-encoded, the OpenAI wire shape);
    * user `tool_result` blocks -> one `{"role":"tool", tool_call_id, content}`
      message each;
    * plain text / image content is downcast via `_content_to_openai`.
    Order is preserved. Input is never mutated.
    """
    out: list = []
    for m in messages or []:
        if not isinstance(m, dict):
            out.append({"role": "user", "content": str(m)})
            continue
        role = m.get("role", "user")
        oai_content, tool_uses, tool_results = _content_to_openai(m.get("content"))

        # tool_result blocks (user turn) -> OpenAI tool messages, emitted first
        # so the result precedes any accompanying prose in the same turn.
        for tr in tool_results:
            out.append({
                "role": "tool",
                "tool_call_id": tr.get("tool_use_id") or "",
                "content": _tool_result_content_to_text(tr.get("content")),
            })

        if role == "assistant" and tool_uses:
            tool_calls = []
            for tu in tool_uses:
                tool_calls.append({
                    "id": tu.get("id") or new_tool_use_id(),
                    "type": "function",
                    "function": {
                        "name": tu.get("name") or "",
                        "arguments": json.dumps(tu.get("input") or {},
                                                ensure_ascii=False),
                    },
                })
            msg = {"role": "assistant", "tool_calls": tool_calls}
            # Preserve any assistant prose alongside the calls.
            if oai_content and not (isinstance(oai_content, str) and not oai_content):
                msg["content"] = oai_content
            out.append(msg)
            continue

        # tool_result-only user turns carry no prose to add.
        if tool_results and (oai_content == "" or oai_content == []):
            continue
        out.append({"role": role, "content": oai_content})
    return out


def anthropic_tools_to_openai(tools):
    """Anthropic tool defs -> OpenAI tool defs, or None.

    `{name, description, input_schema}` -> `{"type":"function","function":
    {name, description, parameters}}` so `_build_tools_preamble` can consume it.
    """
    if not tools or not isinstance(tools, (list, tuple)):
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description") or "",
                "parameters": t.get("input_schema")
                              or {"type": "object", "properties": {}},
            },
        })
    return out or None


def anthropic_tool_choice_to_openai(tool_choice):
    """Anthropic `tool_choice` -> OpenAI `tool_choice`.

    `{"type":"auto"}`->"auto", `{"type":"any"}`->"required",
    `{"type":"tool","name":X}`->`{"type":"function","function":{"name":X}}`.
    None/other -> None (engine default). A bare string passes through.
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return None
    kind = tool_choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "tool" and tool_choice.get("name"):
        return {"type": "function",
                "function": {"name": tool_choice["name"]}}
    return None


def resolve_client_model(model):
    """Anthropic clients (Claude Code, the SDKs) send their own model ids
    ("claude-sonnet-5", "claude-haiku-…") which never match a hugpy model key.
    Those mean "whatever this endpoint serves" — map them to "default" so the
    engine's resolve() falls through to the served brain instead of 404ing
    every request. Real hugpy keys (and None) pass through untouched; the
    route still echoes the name the client asked for."""
    if isinstance(model, str) and model.strip().lower().startswith("claude"):
        return "default"
    return model


def anthropic_to_openai_payload(body: dict) -> dict:
    """Full Anthropic Messages request body -> an OpenAI chat.completions payload.

    The returned dict is shaped exactly like what `v1_chat_completions` receives,
    so it feeds the identical tools-preamble + `_completion_kwargs` pipeline.
    Raises ValueError on a structurally invalid request (missing messages / a
    bad max_tokens) — the route turns that into a 400 Anthropic error.
    """
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' is required")

    oai_messages = anthropic_messages_to_openai(messages)

    # Anthropic top-level system -> a leading OpenAI system message.
    system_text = _system_to_text(body.get("system"))
    if system_text:
        oai_messages = [{"role": "system", "content": system_text}] + oai_messages

    payload: dict = {
        "model": resolve_client_model(body.get("model")),
        "messages": oai_messages,
        "stream": bool(body.get("stream")),
    }

    # max_tokens is REQUIRED by Anthropic; map straight across. Reject a
    # non-integer/negative value at intake rather than letting int() blow up
    # deep in the engine.
    max_tokens = body.get("max_tokens")
    if max_tokens is not None:
        try:
            mt = int(max_tokens)
        except (TypeError, ValueError):
            raise ValueError("'max_tokens' must be an integer")
        if mt <= 0:
            raise ValueError("'max_tokens' must be a positive integer")
        payload["max_tokens"] = mt

    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        payload["top_p"] = body["top_p"]
    if body.get("stop_sequences") is not None:
        payload["stop"] = body["stop_sequences"]

    tools = anthropic_tools_to_openai(body.get("tools"))
    if tools:
        payload["tools"] = tools
        tc = anthropic_tool_choice_to_openai(body.get("tool_choice"))
        if tc is not None:
            payload["tool_choice"] = tc

    return payload


# ──────────────────────────────────────────────────────────────────────────
# response translation: engine finish/usage -> Anthropic
# ──────────────────────────────────────────────────────────────────────────
def finish_to_stop_reason(finish, had_tool_calls: bool = False) -> str:
    """Engine/OpenAI finish_reason -> Anthropic stop_reason.

    length->max_tokens, stop->end_turn, tool_calls->tool_use, stop_sequence->
    stop_sequence. When a tool call was produced, tool_use wins regardless.
    """
    if had_tool_calls:
        return "tool_use"
    return {
        "length": "max_tokens",
        "max_tokens": "max_tokens",
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "stop_sequence": "stop_sequence",
    }.get(finish or "stop", "end_turn")


def usage_to_anthropic(usage_block) -> dict:
    """An OpenAI `_usage_block` dict -> Anthropic `{input_tokens, output_tokens}`.

    Best-effort: 0 when a count is unavailable, never a crash and never a
    missing key (Claude Code reads both).
    """
    if not isinstance(usage_block, dict):
        usage_block = {}
    prompt = usage_block.get("prompt_tokens")
    completion = usage_block.get("completion_tokens")
    return {
        "input_tokens": prompt if isinstance(prompt, int) else 0,
        "output_tokens": completion if isinstance(completion, int) else 0,
    }


def build_content_blocks(reasoning: str, answer: str, tool_calls) -> list:
    """Assemble the Anthropic `content` array for a non-streaming reply.

    Order: thinking (if any) -> text (if any) -> tool_use* (one per parsed
    call). `tool_calls` is the OpenAI shape `_parse_tool_calls` returns
    (`{"function":{"name","arguments"}}`, arguments a JSON string).
    """
    blocks: list = []
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning})
    if answer:
        blocks.append({"type": "text", "text": answer})
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (
                raw if isinstance(raw, dict) else {})
        except (json.JSONDecodeError, ValueError):
            args = {}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or new_tool_use_id(),
            "name": fn.get("name") or "",
            "input": args if isinstance(args, dict) else {},
        })
    # Anthropic requires at least one content block; an empty answer with no
    # reasoning and no tools still needs a (blank) text block.
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


# ──────────────────────────────────────────────────────────────────────────
# SSE framing + streaming encoder
# ──────────────────────────────────────────────────────────────────────────
def sse(event_type: str, data: dict) -> bytes:
    """One Anthropic SSE frame: ``event: <type>\\n data: <json>\\n\\n``."""
    return (
        f"event: {event_type}\n"
        "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
    ).encode("utf-8")


class AnthropicStreamEncoder:
    """Turns the engine's token stream into a valid Anthropic SSE sequence.

    Drives a `StreamingThinkSplitter` (PASSED IN — see module header) so each
    engine token is routed into a `thinking_delta` vs a `text_delta`. Opens a
    thinking block only if reasoning actually appears and a text block only for
    answer text; indices are contiguous and every opened block is closed before
    the next opens — the shape Claude Code validates.

    Usage (the route wires this):
        enc = AnthropicStreamEncoder(msg_id, model, splitter, input_tokens=N)
        yield from enc.start()
        for token: yield from enc.feed(token)
        yield from enc.finish(stop_reason, output_tokens)
    """

    def __init__(self, message_id: str, model, splitter,
                 input_tokens: int = 0):
        self.message_id = message_id
        self.model = model
        self.splitter = splitter
        self.input_tokens = int(input_tokens or 0)
        self._index = -1            # last-used content-block index
        self._open = None           # "thinking" | "text" | None
        self._started = False

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self._started = True
        yield sse("message_start", {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": self.input_tokens,
                          "output_tokens": 0},
            },
        })
        # A ping early keeps intermediary proxies from closing a slow stream.
        yield sse("ping", {"type": "ping"})

    def feed(self, text: str):
        ans, rea = self.splitter.feed(text)
        if rea:
            yield from self._emit(rea, "thinking")
        if ans:
            yield from self._emit(ans, "text")

    def finish(self, stop_reason: str, output_tokens: int = 0):
        # Drain any buffered tail from the splitter first.
        ans, rea = self.splitter.flush()
        if rea:
            yield from self._emit(rea, "thinking")
        if ans:
            yield from self._emit(ans, "text")
        yield from self._close_open()
        yield sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": int(output_tokens or 0)},
        })
        yield sse("message_stop", {"type": "message_stop"})

    # -- internals ---------------------------------------------------------
    def _close_open(self):
        if self._open is not None:
            yield sse("content_block_stop",
                      {"type": "content_block_stop", "index": self._index})
            self._open = None

    def _emit(self, chunk: str, kind: str):
        """Emit a delta of ``kind`` ("thinking"|"text"), (re)opening its block."""
        if self._open != kind:
            # Close whatever is open, then open the block for this kind.
            yield from self._close_open()
            self._index += 1
            self._open = kind
            block = ({"type": "thinking", "thinking": ""} if kind == "thinking"
                     else {"type": "text", "text": ""})
            yield sse("content_block_start", {
                "type": "content_block_start",
                "index": self._index,
                "content_block": block,
            })
        delta = ({"type": "thinking_delta", "thinking": chunk} if kind == "thinking"
                 else {"type": "text_delta", "text": chunk})
        yield sse("content_block_delta", {
            "type": "content_block_delta",
            "index": self._index,
            "delta": delta,
        })

    # -- tool_use (buffered, emitted at done) ------------------------------
    def open_tool_use(self, tool_use_id: str, name: str):
        """Open a `tool_use` content block after text/thinking are closed."""
        yield from self._close_open()
        self._index += 1
        self._open = "tool_use"
        yield sse("content_block_start", {
            "type": "content_block_start",
            "index": self._index,
            "content_block": {"type": "tool_use", "id": tool_use_id,
                              "name": name, "input": {}},
        })

    def tool_use_json(self, partial_json: str):
        yield sse("content_block_delta", {
            "type": "content_block_delta",
            "index": self._index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        })
