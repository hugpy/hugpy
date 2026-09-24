"""Pure helpers for the OpenAI-compatible /v1 route (v1_routes.py).

Deliberately stdlib-only (json/re/uuid) with NO Flask or package imports, so
they unit-test standalone (tests/test_v1_seam.py loads this file directly by
path). Everything here is request/response plumbing that must be testable
without a running app: payload -> engine kwargs translation, and (later
commits) the tools prompt-shim + <tool_call> parser and usage shaping.

Naming keeps the leading underscore the route always used — these are private
to the /v1 seam, the module split exists only for offline testability.
"""
from __future__ import annotations

import json
import re
import uuid


def _gib(value):
    return "unknown" if not isinstance(value, (int, float)) else "%.1f GiB" % (value / 2**30)


def _fleet_status_text(status: dict) -> str:
    """Human line for an engine status event; preserve unknowns honestly."""
    stage = str(status.get("stage") or status.get("type") or "progress")
    model = status.get("model_key") or status.get("model")
    worker = status.get("worker_name") or status.get("worker_id")
    message = status.get("message")
    if message:
        text = str(message)
    elif stage == "dispatch":
        text = "%s routed to %s" % (model or "model", worker or status.get("served_by") or "unknown worker")
    elif stage in ("evict", "evicting"):
        text = "evicting %s from %s, freeing %s VRAM" % (
            status.get("victim") or "unknown model", worker or "unknown worker",
            _gib(status.get("vram_freed") or status.get("freed_bytes")))
    elif stage in ("provision", "download", "downloading"):
        text = "downloading %s to %s drive %s" % (
            model or "model", worker or "unknown worker",
            status.get("drive") or status.get("path") or "unknown")
    elif stage in ("load", "loading", "awaiting-load"):
        text = "loading %s on %s: %s to VRAM & %s to RAM" % (
            model or "model", worker or "unknown worker",
            _gib(status.get("gpu_bytes") or status.get("vram_bytes")),
            _gib(status.get("ram_bytes")))
    elif stage in ("answer", "answering", "generate"):
        text = "%s on %s answering..." % (model or "model", worker or "unknown worker")
    else:
        text = "%s: %s" % (stage, model or "fleet operation")
    progress = status.get("progress")
    if isinstance(progress, (int, float)):
        text += " (%.1f%%)" % (progress * 100 if progress <= 1 else progress)
    return "[hugpy fleet] " + text


def _completion_kwargs(payload: dict) -> dict:
    """Translate an OpenAI chat.completions payload into engine prompt_kwargs.

    Only fields the frozen ChatRequest schema (extra="forbid") defines are
    forwarded — anything else must stay route-local (e.g. `tools`, handled by
    the prompt shim in v1_routes, never reaches the engine).
    """
    messages = payload.get("messages")
    if not messages:
        raise ValueError("'messages' is required")
    # OpenAI clients must send *something* as model; "default" (and empty)
    # mean "no preference" — leave model_key unset so resolve() falls through
    # to the reconciled chat default instead of 404ing on a literal "default".
    model = payload.get("model")
    if isinstance(model, str) and model.strip().lower() in ("", "default"):
        model = None
    # OpenAI multimodal content (2026-09-23): ``image_url`` parts ride to the
    # model as ChatRequest.images (the engine's single image path — the GGUF
    # runner folds them back into the latest user turn as image_url parts for
    # the native llama-server --mmproj). They used to be str()-flattened into
    # the prompt text and never reached the model.
    images = _collect_images(messages)
    kwargs = {
        # Fold the OpenAI tool-calling shapes (assistant `tool_calls`
        # echo-backs, `{"role":"tool"}` results) down into the plain
        # role+content wire the frozen ChatRequest / released worker speak —
        # rendered into the same <tool_call>/<tool_response> text the tools
        # preamble taught the model. Plain messages pass through untouched.
        "messages": _render_tool_messages(messages),
        "model_key": model,
        "request_id": f"v1-{uuid.uuid4().hex}",
    }
    if images:
        kwargs["images"] = images
    max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens")
    if max_tokens:
        # Explicit client cap → bounded; omitted → engine runs unbounded with
        # auto-continuation, same as the console.
        kwargs["max_new_tokens"] = int(max_tokens)
    if payload.get("temperature") is not None:
        kwargs["temperature"] = float(payload["temperature"])
        kwargs["do_sample"] = float(payload["temperature"]) > 0
    if payload.get("top_p") is not None:
        kwargs["top_p"] = float(payload["top_p"])
    if payload.get("unbounded") is not None:
        kwargs["unbounded"] = bool(payload["unbounded"])
    # Continuation budget. Never forwarding this was the production stall
    # (2026-07-14): every /v1 request ran with unbounded auto-continuation, so
    # a rambling small model kept getting "Continue exactly where you left off"
    # passes appended until the read timeout. OpenAI `max_tokens` semantics are
    # ONE bounded completion — so a request that caps tokens but doesn't ask
    # for continuation defaults to a single pass. A request with neither cap
    # keeps today's unbounded behavior (the console relies on it).
    if payload.get("max_chunks") is not None:
        kwargs["max_chunks"] = int(payload["max_chunks"])
    elif max_tokens:
        kwargs["max_chunks"] = 1
    # Per-request allocation triggers (operator ask 2026-07-29): 4-bit / MoE /
    # alloc-mode overrides for THIS call. Whitelisting + wire-scrubbing happen
    # at the relay (_relay_payload) — the single enforcement point.
    if isinstance(payload.get("alloc"), dict) and payload["alloc"]:
        kwargs["alloc"] = payload["alloc"]
    # t74 hard no-think: per-request engine chat keys, forwarded onto the slot
    # child's llama-server body. chat_template_kwargs reaches the chat TEMPLATE
    # ({"enable_thinking": false} pre-closes the <think> block — enforced by
    # rendering, for models that ignore the /no_think soft switch); logit_bias
    # is the OpenAI param (the <think>-token-ban fallback). Version-gating per
    # worker happens at the relay (gate_chat_extras_for_worker) — the single
    # enforcement point, same pattern as alloc above.
    for k in ("chat_template_kwargs", "logit_bias"):
        if isinstance(payload.get(k), dict) and payload[k]:
            kwargs[k] = payload[k]
    # k96 no-evict guarantee (agent brains): forwarded only when truthy so
    # ordinary traffic stays byte-identical; dispatch runs the load politely
    # and fails fast instead of evicting (see dispatch.ensure_headroom_for_load
    # and ChatRequest.no_makeroom).
    if payload.get("no_makeroom"):
        kwargs["no_makeroom"] = True
    return kwargs


# ──────────────────────────────────────────────────────────────────────────
# tool-message rendering — the INBOUND half of the tools shim (step 2+).
#
# _build_tools_preamble/_parse_tool_calls handle the OUTBOUND half (advertise
# tools, parse the model's reply). The multi-turn loop also sends messages the
# other way: the assistant turn echoed back WITH its `tool_calls`, plus one
# `{"role":"tool","tool_call_id":…,"content":…}` result per call. GGUF models
# have no native tool-message wire and the released worker's message_to_dict
# keeps only role+content, so we render those shapes into the SAME dialect the
# preamble taught the model:
#   • assistant tool_calls → an assistant turn whose content is the
#     <tool_call>{"name":…,"arguments":…}</tool_call> block(s) it "said",
#   • a tool result        → a USER turn carrying <tool_response>…</tool_response>
#     (Qwen/Hermes convention — llama-server chat templates reject/mangle an
#     unknown "tool" role, and the preamble already told the model the result
#     arrives inside <tool_response> tags).
# The result is a plain role+content message list the frozen schema and every
# released worker accept, so the model can read the tool output and continue.
# Plain (no-tools) chat never carries these keys → passed through byte-for-byte.
# ──────────────────────────────────────────────────────────────────────────


def _render_assistant_tool_calls(tool_calls) -> str:
    """Assistant `tool_calls` array -> the <tool_call>…</tool_call> text.

    Mirrors _parse_tool_calls' inverse: each call becomes one block holding a
    {"name","arguments"} JSON object with `arguments` as a parsed object (the
    OpenAI wire keeps `arguments` as a JSON *string*; unwrap it one level so the
    rendered block matches what the model itself emits — a JSON object, not a
    quoted string). A malformed/leftover string is emitted as-is rather than
    dropped.
    """
    blocks = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except (json.JSONDecodeError, ValueError):
                args = raw_args  # keep the model's own (possibly sloppy) text
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        obj = {"name": str(name), "arguments": args}
        blocks.append("<tool_call>\n"
                      + json.dumps(obj, ensure_ascii=False)
                      + "\n</tool_call>")
    return "\n".join(blocks)


_IMAGE_PART_TYPES = ("image_url", "input_image", "image")


def _image_part_url(part) -> "str | None":
    """The data:/http(s)/base64 string of one OpenAI image content part."""
    val = part.get("image_url", part.get("url", part.get("image")))
    if isinstance(val, dict):
        val = val.get("url")
    return val if isinstance(val, str) and val.strip() else None


def _content_text(content) -> str:
    """Message content as text: a string as-is; a content-part list as its
    text parts joined (image parts are carried separately, see
    _collect_images) — never the Python repr of the list."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                if part.get("type") in _IMAGE_PART_TYPES:
                    continue
                if isinstance(part.get("text"), str):
                    texts.append(part["text"])
        return "\n".join(t for t in texts if t)
    return str(content)


def _collect_images(messages) -> list:
    """Every image part's url across ``messages``, in order. A part without a
    usable url is a malformed request, not something to drop silently."""
    out = []
    for m in messages or []:
        if not isinstance(m, dict) or not isinstance(m.get("content"), list):
            continue
        for part in m["content"]:
            if isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES:
                url = _image_part_url(part)
                if not url:
                    raise ValueError(
                        f"image content part without a url: {sorted(part)!r} — "
                        "send {\"type\":\"image_url\",\"image_url\":{\"url\":"
                        "\"data:image/png;base64,...\"}}")
                out.append(url)
    return out


def _render_tool_messages(messages):
    """Downcast OpenAI messages to the role+content wire, rendering tool turns.

    The single funnel between a /v1 payload's `messages` and the engine kwargs:
    every message becomes {"role","content"} (str). Assistant `tool_calls` and
    `{"role":"tool"}` results are rendered into <tool_call>/<tool_response>
    text; everything else is the same {role, content} downcast as before. Input
    is never mutated.
    """
    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            # bare/odd entry — preserve prior lenient behavior
            out.append({"role": "user", "content": str(m)})
            continue
        role = m.get("role", "user")
        content = _content_text(m.get("content"))

        if role == "tool":
            # A tool result the model reads as <tool_response>…</tool_response>
            # on a user turn (unknown "tool" role would break the chat template).
            out.append({"role": "user",
                        "content": f"<tool_response>\n{content}\n</tool_response>"})
            continue

        if role == "assistant" and m.get("tool_calls"):
            rendered = _render_assistant_tool_calls(m.get("tool_calls"))
            # Preserve any assistant prose that rode alongside the calls, then
            # the call block(s) — the order the model would have produced them.
            merged = "\n".join(p for p in (content, rendered) if p)
            out.append({"role": "assistant", "content": merged})
            continue

        out.append({"role": role, "content": content})
    return out


def _usage_block(usage) -> dict:
    """Shape a DoneEvent usage dict into the OpenAI `usage` object.

    Real counts when the engine/runner reported them; the historical all-None
    shape when genuinely unavailable (old worker builds, error paths) — a
    missing count must never crash or omit the key, OpenAI SDKs expect it.
    """
    if not isinstance(usage, dict):
        usage = {}

    def _int(key):
        v = usage.get(key)
        return v if isinstance(v, int) else None

    prompt = _int("prompt_tokens")
    completion = _int("completion_tokens")
    total = _int("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total}


# ──────────────────────────────────────────────────────────────────────────
# tools shim — prompt-inject + parse, entirely route-local.
#
# The engine schema is frozen (extra="forbid") and GGUF models have no native
# function-calling wire, so /v1 `tools` support lives here: render the tool
# JSON-schemas into a system-prompt preamble using the Qwen2.5/Hermes
# convention (the fleet's Qwen-family GGUFs were trained on exactly this
# format), then parse `<tool_call>{...}</tool_call>` blocks out of the reply.
# Modeled on the sibling client hugpy_agent/adapter.py (prompted tier) so both
# sides of the seam speak the same dialect. `tools` itself NEVER reaches the
# engine.
# ──────────────────────────────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# Known engine leak strings (the auto-continuation prompts from
# dispatch.execute_chat_stream / hugpy_agent's captured variant). Scrubbed
# before parsing — a leak INSIDE a JSON block would corrupt it — and defense
# in depth on top of the max_chunks=1 forced for tool requests.
_CONTINUATION_LEAKS = (
    "Continue exactly where you left off. Do not repeat any previous text.",
    "Continue exactly where I left off. Do not repeat any previous text.",
    "Continue exactly where you left off.",
)

_TOOLS_PREAMBLE_TEMPLATE = """\
# Tools

You may call ONE function per reply to assist with the task.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_lines}
</tools>

For a function call, return a json object with function name and arguments \
within <tool_call></tool_call> XML tags, then STOP:
<tool_call>
{{"name": "<function-name>", "arguments": {{<args-json-object>}}}}
</tool_call>

The function result will come back inside <tool_response></tool_response> tags.
Never invent a function result — wait for the real one."""


def _build_tools_preamble(tools, tool_choice=None):
    """System-prompt block advertising the request's tools, or None.

    None means "run a plain completion": empty/malformed tools, tool_choice
    "none", or a forced {"function":{"name":...}} that matches nothing. A
    specific forced choice narrows the advertised list to that one tool and
    appends a MUST-call instruction; "auto"/absent advertises them all.
    """
    if not tools or not isinstance(tools, (list, tuple)) or tool_choice == "none":
        return None
    forced = None
    if isinstance(tool_choice, dict):
        forced = ((tool_choice.get("function") or {}).get("name")) or None
    lines = []
    for t in tools:
        fn = (t or {}).get("function") or {} if isinstance(t, dict) else {}
        name = fn.get("name")
        if not name or (forced and name != forced):
            continue
        lines.append(json.dumps({
            "type": "function",
            "function": {
                "name": name,
                "description": fn.get("description") or "",
                "parameters": fn.get("parameters")
                              or {"type": "object", "properties": {}},
            },
        }, separators=(",", ":"), ensure_ascii=False))
    if not lines:
        return None
    preamble = _TOOLS_PREAMBLE_TEMPLATE.format(tool_lines="\n".join(lines))
    if forced:
        preamble += (f"\n\nYou MUST call the function \"{forced}\" now — "
                     "reply with exactly one <tool_call> block.")
    return preamble


def _inject_tools_preamble(messages, preamble):
    """Messages with the tools preamble merged in as/into a system message.

    Appended to an existing leading system message (operator instructions keep
    priority) when its content is plain text; otherwise a fresh system message
    is prepended. Input list/dicts are not mutated.
    """
    out = [dict(m) if isinstance(m, dict) else m for m in (messages or [])]
    for m in out:
        if isinstance(m, dict) and m.get("role") == "system":
            content = m.get("content")
            if content is None or isinstance(content, str):
                m["content"] = f"{content}\n\n{preamble}".strip() if content else preamble
                return out
            break  # multimodal/odd system content — don't corrupt it
    return [{"role": "system", "content": preamble}] + out


def _parse_call_json(block: str, *, require_arguments: bool = False):
    """One tool-call JSON object -> {"name": str, "arguments": dict} or None.

    Tolerates the sloppy-small-model cases the hugpy_agent adapter does:
    double-encoded `arguments` strings are unwrapped one level, and a missing
    `arguments` defaults to {} (unless require_arguments — the bare-JSON
    scan uses that to avoid claiming arbitrary {"name": ...} prose objects).
    """
    try:
        data = json.loads(block)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("name"):
        return None
    if require_arguments and "arguments" not in data:
        return None
    args = data.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(args, dict):
        return None
    return {"name": str(data["name"]), "arguments": args}


def _bare_call(text: str):
    """First standalone {"name":..., "arguments":...} object in free text.

    Balanced-brace scan (nested braces defeat any regex). Stricter than the
    fenced path: `arguments` must be present, so a prose JSON object that
    merely has a "name" key never becomes a false tool call.
    """
    idx = 0
    while True:
        start = text.find("{", idx)
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    if '"name"' in candidate and '"arguments"' in candidate:
                        call = _parse_call_json(candidate, require_arguments=True)
                        if call:
                            return call
                    break
        else:
            return None
        idx = start + 1


def _parse_tool_calls(text: str):
    """Assistant text -> (clean_text, OpenAI tool_calls list | None).

    Finds every <tool_call>...</tool_call> block (bare-JSON fallback when the
    model skipped the tags). No valid call found — including malformed JSON —
    returns (original text, None): errors-as-data, the caller answers with
    plain content exactly as today rather than 500ing on a shim parse error.
    clean_text is the reply with call blocks and known leak strings removed.
    """
    original = text or ""
    scrubbed = original
    for leak in _CONTINUATION_LEAKS:
        scrubbed = scrubbed.replace(leak, "")

    calls = []
    remaining = scrubbed
    for m in _TOOL_CALL_RE.finditer(scrubbed):
        call = _parse_call_json(m.group(1))
        if call:
            calls.append(call)
            remaining = remaining.replace(m.group(0), "", 1)
    if not calls:
        call = _bare_call(scrubbed)
        if call:
            calls, remaining = [call], ""
    if not calls:
        return original, None

    tool_calls = [{
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": c["name"],
                     "arguments": json.dumps(c["arguments"], ensure_ascii=False)},
    } for c in calls]
    return remaining.strip(), tool_calls
