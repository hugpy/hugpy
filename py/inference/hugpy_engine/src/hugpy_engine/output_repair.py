"""Per-model OUTPUT REPAIR on central's way out to the client.

Driven ONLY by the model's ``hugpy.json["output_repair"]`` block, which
hugpy-model-audit derives from the weights (hugpy_ops.model_audit,
``_check_dead_delimiters``) and records. No Hub call, no guessing: a model
without the block streams byte-identical to before.

kind ``dead_think_delimiters`` — a fine-tune appended <think>/</think> but never
trained their output rows, so the model prints an ordinary "glitch" token in
their place (LFM2.5-350M-home-assistant-sft: " tvåvingeart" / " fjärilsart").
The filter maps them back:

  * 1st glitch occurrence -> ``<think>``, 2nd -> ``</think>`` — the span between
    is the model's reasoning; the existing /v1 think split
    (StreamingThinkSplitter / strip_think) moves it to ``reasoning_content``,
    the console renders it as a think block. Nothing is discarded.
  * every further occurrence is removed (stray).
  * everything else is untouched. Streaming holds back ONLY a buffer tail that
    is a prefix of a glitch string (to decide whether it starts one).

The DoneEvent carries ``hugpy={"output_repair": {"applied": True, "kind": ...,
"removed": n}}`` so /v1 surfaces that the repair ran.

Wired once, in :func:`hugpy_engine.query.stream_query` — the stream every
/v1/chat/completions call (streaming and not, incl. the benchmark grader's) and
the console chat relay consume.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, AsyncIterator, Optional

KIND_DEAD_THINK = "dead_think_delimiters"
_SPEC_TTL_S = 30.0
_spec_cache: dict = {}
_spec_lock = threading.Lock()


def _model_dir(model_key: str) -> Optional[str]:
    from .catalog import get_model
    raw = dict(getattr(get_model(model_key), "raw", None) or {})
    dest = raw.get("destination")
    if dest:
        return str(dest)
    folder = raw.get("folder")
    if folder:
        from hugpy_platform.constants import MODELS_HOME
        return os.path.join(MODELS_HOME, str(folder))
    return None


def repair_spec(model_key: Optional[str]) -> Optional[dict]:
    """The model's usable ``output_repair`` block, or None. Cached ~30 s per
    key (one small JSON read per model per window, never per token)."""
    if not model_key:
        return None
    now = time.monotonic()
    with _spec_lock:
        hit = _spec_cache.get(model_key)
        if hit and now - hit[0] < _SPEC_TTL_S:
            return hit[1]
    spec = None
    try:
        from hugpy_storage.hugpy_marker import read_output_repair
        block = read_output_repair(_model_dir(model_key))
        if (isinstance(block, dict) and block.get("kind") == KIND_DEAD_THINK
                and any(isinstance(t, str) and t.strip() for t in block.get("glitch_tokens") or ())):
            spec = block
    except Exception:  # noqa: BLE001 — a spec lookup must never break a chat
        spec = None
    with _spec_lock:
        _spec_cache[model_key] = (now, spec)
    return spec


class DeadThinkFilter:
    """Streaming glitch-token -> <think>/</think> mapper (see module doc)."""

    def __init__(self, spec: dict):
        words = {t.strip() for t in spec.get("glitch_tokens") or () if isinstance(t, str) and t.strip()}
        self.words = sorted(words, key=len, reverse=True)
        self.open = spec.get("open") or "<think>"
        self.close = spec.get("close") or "</think>"
        self.kind = spec.get("kind") or KIND_DEAD_THINK
        self.seen = 0
        self.buf = ""

    def _replacement(self) -> str:
        self.seen += 1
        if self.seen == 1:
            return self.open
        if self.seen == 2:
            return self.close
        return ""

    def _holdback(self, text: str) -> int:
        """Length of the longest tail of ``text`` that is a proper prefix of a glitch word."""
        best = 0
        for w in self.words:
            for n in range(min(len(w) - 1, len(text)), best, -1):
                if text.endswith(w[:n]):
                    best = n
                    break
        return best

    def feed(self, text: str) -> str:
        if not text:
            return ""
        buf = self.buf + text
        out = []
        while True:
            hit = None
            for w in self.words:
                k = buf.find(w)
                if k != -1 and (hit is None or k < hit[0]):
                    hit = (k, w)
            if hit is None:
                break
            k, w = hit
            pre = buf[:k]
            if pre.endswith(" "):          # the glitch token's own leading space
                pre = pre[:-1]
            out.append(pre)
            out.append(self._replacement())
            buf = buf[k + len(w):]
        hold = self._holdback(buf)
        self.buf = buf[len(buf) - hold:] if hold else ""
        out.append(buf[:len(buf) - hold] if hold else buf)
        return "".join(out)

    def flush(self) -> str:
        rest, self.buf = self.buf, ""
        return rest

    def report(self) -> dict:
        return {"applied": True, "kind": self.kind, "removed": self.seen}


def _with(ev: Any, **update: Any) -> Any:
    try:
        return ev.model_copy(update=update)
    except Exception:  # noqa: BLE001 — non-pydantic event: mutate in place
        for k, v in update.items():
            setattr(ev, k, v)
        return ev


def _token_like(ev: Any, text: str) -> Any:
    from .schemas.event_schemas import TokenEvent
    return TokenEvent(request_id=getattr(ev, "request_id", "") or "", text=text)


async def repair_stream(events: AsyncIterator[Any], model_key: Optional[str]) -> AsyncIterator[Any]:
    """Pass ``events`` through the model's output repair (identity when the
    model has no ``output_repair`` block)."""
    spec = repair_spec(model_key)
    if spec is None:
        async for ev in events:
            yield ev
        return
    flt = DeadThinkFilter(spec)
    try:
        async for ev in _repaired(events, flt):
            yield ev
    finally:
        # propagate a client disconnect down the chain (releases a relayed
        # worker's httpx stream exactly as the unwrapped chain did)
        aclose = getattr(events, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001
                pass


async def _repaired(events: AsyncIterator[Any], flt: "DeadThinkFilter") -> AsyncIterator[Any]:
    async for ev in events:
        t = getattr(ev, "type", None)
        if t == "token":
            out = flt.feed(getattr(ev, "text", "") or "")
            if out:
                yield _with(ev, text=out)
            continue
        if t in ("done", "error"):
            rest = flt.flush()
            if rest:
                yield _token_like(ev, rest)
            if t == "done":
                prior = getattr(ev, "hugpy", None)
                hugpy = dict(prior) if isinstance(prior, dict) else {}
                hugpy["output_repair"] = flt.report()
                ev = _with(ev, hugpy=hugpy)
        yield ev
    rest = flt.flush()
    if rest:
        yield _token_like(None, rest)


__all__ = ["DeadThinkFilter", "KIND_DEAD_THINK", "repair_spec", "repair_stream"]
