"""NO-THINK — the package-wide seam for "I need prose, not a monologue".

OPERATOR RULING 2026-07-29: *"the expectation of a model to adhere to no_think
should be circumvented for any execution that requires this stipulation, package
wide."* Model adherence is NEVER trusted. Wherever a call stipulates
thinking-off, the package does BOTH halves: it sends the suppression directive
WITH the query, and it strips any ``<think>…</think>`` that comes back anyway,
surfacing the reasoning under its own key.

WHY THE STRIP EXISTS EVEN THOUGH THE DIRECTIVE IS SENT
------------------------------------------------------
Reasoning does not stay in its own lane here. llama.cpp extracts
``<think>…</think>`` into ``message.reasoning_content``, and
``managers/llama/runners/src/ccp_runner.py:43`` then DELIBERATELY RE-INLINES it::

    if reasoning and "<think>" not in content:
        return f"<think>{reasoning}</think>{content}"

so it arrives fused into the text. That re-inlining is correct — it is how the
reasoning survives the wire — and it is left alone; the split happens
caller-side, here.

Measured on the live fleet 2026-07-27, flux2-klein-9b-uncensored-text-encoder
via computron, draft "a red car on a wet street": the ENTIRE 200-token budget
went to reasoning and the reply was ``<think>Okay, the user wants me to
expand...</think>`` with NO prompt at all. Stripping alone would therefore have
yielded an EMPTY string — which is why the suppression half matters and a
strip-only fix would have looked correct in a unit test and produced nothing in
the product. Conversely the directive alone is not enough: the model is
caller-selectable and the next one picked may ignore it.

Hence UNCLOSED-``<think>`` HANDLING is load-bearing. When the token budget runs
out mid-thought there is no closing tag at all; :func:`strip_think` treats an
unclosed block as reasoning to the end of the string, so a truncated ramble can
never be served as prose. A reply that was nothing but thinking yields ``("",
reasoning)`` and the caller is expected to turn that into an honest error rather
than an empty result.

WHY A WRAPPER AND NOT AN ``execute_prompt(no_think=True)`` PARAMETER
-------------------------------------------------------------------
``managers.dispatch.execute_prompt`` funnels its kwargs through
``normalize_prompt_kwargs`` → ``resolve`` → the per-task request builder, and the
built request is what the remote relay serializes: ``managers/resolvers/remote.py``
sends ``req.model_dump()`` to a worker whose request schema is ``extra="forbid"``.
A new dispatch kwarg is therefore a WIRE CHANGE — an old worker would reject the
whole request. So the stipulation is expressed as :func:`execute_prompt_no_think`,
which rewrites the MESSAGES (a field the wire already carries) before the call and
splits the reply after it. Nothing new crosses the wire. Do not "simplify" this
into a dispatch parameter.

WHY ``utils/`` AND NOT ``managers/llama/``
------------------------------------------
The seam is engine-agnostic and package-wide — video_intel, phone_brick, the
discord cogs and the flask routes all use it. Living under ``managers/llama``
would drag the llama serve/runner stack into callers that never touch it.

The directive itself is the Qwen3 chat-template idiom: those templates carry
``{%- if enable_thinking is defined and enable_thinking is false %}`` which emits
a PRE-CLOSED ``<think>\\n\\n</think>`` so generation starts after thinking is
already shut. ``/no_think`` rides the EXISTING message path and does the same job
for models that honor it. VERIFIED on flux2-klein + draft: a clean 4-sentence
prompt, no ``<think>`` at all.

THE HARD SWITCH (t74). Some models ignore the soft directive outright
(wazimondo~Qwen3.6-35B-A3B-Uncensored-Wasserstein-GGUF burns its whole budget
thinking anyway), so the template kwarg became REACHABLE: ``ChatRequest`` now
carries optional ``chat_template_kwargs`` (and ``logit_bias``, the
<think>-token-ban fallback), the chat builder forwards them, and the slot path
puts them verbatim on the llama-server /v1/chat/completions body — where
``{"enable_thinking": false}`` is enforced by the TEMPLATE, not by model
obedience. :func:`execute_prompt_no_think` sets it by default. The wire stays
safe by VERSION GATE, not by wishful thinking: the keys are omitted from
``model_dump()`` when unset (byte-identical wire for everyone else), and the
relay strips them — LOUDLY, never as a silent no-op — for a worker whose
pkg_version predates ``alloc_modes.CHAT_EXTRAS_MIN_PKG_VERSION``. A stripped or
in-process request degrades to exactly this module's directive+strip seam, which
is why BOTH halves here remain load-bearing and nothing below this line changed.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "NO_THINK_DIRECTIVE",
    "NO_THINK_CHAT_TEMPLATE_KWARGS",
    "THINK_BLOCK_RE",
    "strip_think",
    "with_no_think",
    "apply_no_think",
    "finalize_no_think",
    "execute_prompt_no_think",
    "StreamingThinkSplitter",
]

NO_THINK_DIRECTIVE = "/no_think"

#: The HARD half of the stipulation (t74): rendered by the chat template itself
#: (Qwen3-family ``enable_thinking``), so it works on models that ignore the
#: soft directive. Rides ChatRequest.chat_template_kwargs to the slot child's
#: llama-server body; version-gated at the relay, ignored where a template has
#: no such variable — always IN ADDITION to the directive, never instead.
NO_THINK_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

#: Matches a think block, closed OR unclosed (``\Z`` alternative) — see the
#: module docstring on why the unclosed case is not an edge case here.
THINK_BLOCK_RE = re.compile(r"<think>(.*?)(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> tuple[str, str]:
    """Split a model reply into ``(prose, reasoning)``.

    Removes every ``<think>...</think>`` block and returns the surviving prose
    plus the concatenated reasoning. An UNCLOSED ``<think>`` (the token budget
    ran out mid-thought) is treated as reasoning to the end of the string, so a
    truncated ramble can never be served as prose. Returns ``("", reasoning)``
    when the reply was nothing but thinking; the caller turns that into an honest
    error rather than an empty result.
    """
    if not text:
        return "", ""
    reasoning = "\n".join(m.group(1).strip() for m in THINK_BLOCK_RE.finditer(text))
    return THINK_BLOCK_RE.sub("", text).strip(), reasoning.strip()


_OPEN_RE = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
_CLOSE_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)
# The longest tag we might be MID-way through at a buffer boundary — used to
# decide how much of the tail to hold back so a tag split across tokens
# ("<thi" + "nk>") is never emitted as literal text.
_TAG_PREFIXES = ("<think>", "</think>", "<thinking>", "</thinking>")


class StreamingThinkSplitter:
    """Incrementally route a token stream into (answer, reasoning).

    The /v1 mount receives the reasoning ALREADY RE-INLINED as ``<think>…
    </think>`` in the content (ccp_runner re-inlines what llama.cpp put in
    reasoning_content — see this module's header). An OpenAI-style client such
    as OpenCode renders a collapsible reasoning panel ONLY when the reasoning
    arrives in ``delta.reasoning_content``, not as literal ``<think>`` text in
    the answer. This splitter re-separates the two AS THEY STREAM so the console
    gets a proper reasoning channel (operator 2026-07-31: "no_think should be
    applied to all responses … the think var available in the console and
    collapsed upon its output").

    Stream-safe by construction: ``feed(token)`` returns ``(answer_delta,
    reasoning_delta)`` and NEVER emits a partial tag — a tag split across tokens
    is held in an internal buffer until it resolves. ``flush()`` drains the tail
    at end-of-stream; an unclosed ``<think>`` (budget ran out mid-thought) drains
    as reasoning, never as answer.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    @staticmethod
    def _safe_prefix_len(buf: str) -> int:
        """How many trailing chars to HOLD BACK because they may begin a tag."""
        for cut in range(1, min(len(buf), 10) + 1):
            tail = buf[-cut:].lower()
            if any(p.startswith(tail) for p in _TAG_PREFIXES):
                return cut
        return 0

    def feed(self, token: str) -> "tuple[str, str]":
        self._buf += (token or "")
        answer, reasoning = [], []
        while self._buf:
            if not self._in_think:
                m = _OPEN_RE.search(self._buf)
                if m:
                    answer.append(self._buf[:m.start()])
                    self._buf = self._buf[m.end():]
                    self._in_think = True
                    continue
                hold = self._safe_prefix_len(self._buf)
                if hold:
                    answer.append(self._buf[:-hold]); self._buf = self._buf[-hold:]
                else:
                    answer.append(self._buf); self._buf = ""
                break
            else:
                m = _CLOSE_RE.search(self._buf)
                if m:
                    reasoning.append(self._buf[:m.start()])
                    self._buf = self._buf[m.end():]
                    self._in_think = False
                    continue
                hold = self._safe_prefix_len(self._buf)
                if hold:
                    reasoning.append(self._buf[:-hold]); self._buf = self._buf[-hold:]
                else:
                    reasoning.append(self._buf); self._buf = ""
                break
        return "".join(answer), "".join(reasoning)

    def flush(self) -> "tuple[str, str]":
        """Drain the tail. Unclosed think → reasoning; otherwise → answer."""
        rest, self._buf = self._buf, ""
        if not rest:
            return "", ""
        return ("", rest) if self._in_think else (rest, "")


def with_no_think(user_text: str) -> str:
    """PROPAGATE THE QUERY, then ask for it without the monologue.

    Appending the directive (rather than replacing anything) is what keeps the
    caller's own text intact. Idempotent — a query that already carries the
    directive is returned untouched.
    """
    if not user_text:
        return NO_THINK_DIRECTIVE
    if NO_THINK_DIRECTIVE in user_text:
        return user_text
    return f"{user_text}\n\n{NO_THINK_DIRECTIVE}"


def _content_with_no_think(content: Any) -> Any:
    """Append the directive to a message ``content`` of any supported shape.

    Plain strings get :func:`with_no_think`. Multimodal content (a list of
    ``{"type": "text"|"image_url", ...}`` parts, as image-text-to-text calls
    build) gets the directive appended to its LAST text part, or a new text part
    when it has none — the images are never touched.
    """
    if isinstance(content, str) or content is None:
        return with_no_think(content or "")
    if isinstance(content, list):
        parts = list(content)
        for i in range(len(parts) - 1, -1, -1):
            p = parts[i]
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                if NO_THINK_DIRECTIVE in p["text"]:
                    return parts
                parts[i] = {**p, "text": with_no_think(p["text"])}
                return parts
        parts.append({"type": "text", "text": NO_THINK_DIRECTIVE})
        return parts
    return content


def apply_no_think(messages):
    """Return a copy of ``messages`` with the directive on the last user message.

    The directive rides the last ``role == "user"`` turn because that is the one
    immediately preceding generation — a system-prompt placement is routinely
    ignored by the chat template. When there is no user turn at all (a
    system-only conversation) a bare directive turn is appended, which is what
    the template would need to see anyway.

    Never mutates the caller's list or its message dicts.
    """
    if not messages:
        return [{"role": "user", "content": NO_THINK_DIRECTIVE}]
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        if isinstance(m, dict) and m.get("role") == "user":
            out[i] = {**m, "content": _content_with_no_think(m.get("content"))}
            return out
    out.append({"role": "user", "content": NO_THINK_DIRECTIVE})
    return out


def _result_text(result) -> str:
    """Best-effort text extraction — a worker-relay dict, a pydantic ChatResult,
    or any other TaskResult-shaped object all yield the same plain string."""
    if isinstance(result, dict):
        return result.get("text") or ""
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(result, attr, None)
        if callable(fn):
            try:
                d = fn()
                if isinstance(d, dict):
                    return d.get("text") or ""
            except TypeError:
                continue
    return getattr(result, "text", "") or ""


def _result_ok(result) -> bool:
    if isinstance(result, dict):
        return bool(result.get("ok", True))
    return bool(getattr(result, "ok", True))


def _result_error(result):
    if isinstance(result, dict):
        return result.get("error")
    return getattr(result, "error", None)


def finalize_no_think(result) -> dict:
    """Split an already-awaited execute_prompt result into a no-think dict.

    Returns ``{"ok", "text", "reasoning", "thinking_suppressed": True, "error",
    "raw"}``. ``text`` is the prose with every think block removed; ``reasoning``
    carries what was removed so a caller can show or ignore it but can never
    mistake it for the answer. A reply that was nothing but thinking comes back
    ``ok=False`` with an error naming the cause — never a silent empty string.
    """
    ok = _result_ok(result)
    raw = (_result_text(result) or "").strip()
    text, reasoning = strip_think(raw)
    out = {
        "ok": bool(ok and text),
        "text": text,
        "reasoning": reasoning,
        "thinking_suppressed": True,
        "raw": raw,
        "error": _result_error(result),
    }
    if not ok:
        out["error"] = out["error"] or "generation failed"
    elif not text:
        out["error"] = (
            "the model returned only reasoning and no answer — it appears to have "
            "ignored the no-think directive; retry, or choose a different text "
            "generator"
        )
    return out


def execute_prompt_no_think(*args: Any, **kwargs: Any) -> dict:
    """``execute_prompt`` with the no-think stipulation, both halves applied.

    Pre-call: the directive is appended to the last user message (or to
    ``prompt``/``text`` when the call is not message-shaped). Post-call: any
    ``<think>`` block is stripped and returned separately. NOTHING NEW CROSSES
    THE WIRE — see the module docstring.

    Sync, like ``execute_prompt``; an awaitable result is driven on the
    process-wide runtime rather than a fresh per-request loop.
    """
    if kwargs.get("messages"):
        kwargs["messages"] = apply_no_think(kwargs["messages"])
    else:
        for key in ("prompt", "text", "input_text"):
            if isinstance(kwargs.get(key), str):
                kwargs[key] = with_no_think(kwargs[key])
                break

    # THE HARD SWITCH (t74), alongside — never instead of — the directive: the
    # chat builder forwards chat_template_kwargs onto ChatRequest, and the slot
    # path puts it on the llama-server body, where enable_thinking=false is
    # enforced by the template itself. Per-key merge so a caller's own
    # chat_template_kwargs survive; non-chat builders simply ignore the kwarg,
    # old workers get it stripped (loudly) at the relay, and the in-process
    # runner names it as inapplicable — every degrade lands back on the
    # directive+strip seam below.
    ctk = dict(kwargs.get("chat_template_kwargs") or {})
    for k, v in NO_THINK_CHAT_TEMPLATE_KWARGS.items():
        ctk.setdefault(k, v)
    kwargs["chat_template_kwargs"] = ctk

    from hugpy_engine.dispatch import execute_prompt

    result = execute_prompt(*args, **kwargs)

    import inspect
    if inspect.isawaitable(result):
        from hugpy_platform import async_runtime
        result = async_runtime.run(result)

    return finalize_no_think(result)
