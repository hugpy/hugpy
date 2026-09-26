import asyncio
import os
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from hugpy_engine.llama.runners.src.imports.init_imports import logger
from hugpy_engine.llama.runners.src.imports.utils import (
    map_finish_reason,
    messages_to_prompt_from_dicts,
    resolve_max_tokens,
    resolve_temperature,
    resolve_top_p,
)
from hugpy_engine.schemas.chat_schemas import ChatRequest
from hugpy_engine.schemas.event_schemas import DoneEvent, ErrorEvent, TokenEvent
from hugpy_engine.schemas.runner_schemas import StreamEvent
from hugpy_platform.utils import messages_to_dicts
import re
from difflib import SequenceMatcher
# base_runner.py

# --------------------------------------------------------------------------- #
# loop / stagnation guard for the unbounded continue-loop                     #
# --------------------------------------------------------------------------- #
# The unbounded paths append a {"role":"user","content":"continue"} nudge after
# every finish_reason=="length" chunk and go again. With no guard a model that
# has started looping regenerates the same block up to max_chunks times. This is
# a CONSERVATIVE safety net: only stop when consecutive continue-chunks are
# (near-)identical for several passes in a row, so genuinely long, evolving
# output — where each pass produces distinct text — never trips it.
_LOOP_GUARD_WINDOW = 4000   # trailing chars compared per chunk (bounds cost)


def _loop_guard_params():
    """(max_repeat, similarity) for the anti-repetition guard, env-tunable via
    HUGPY_LOOP_GUARD_MAX_REPEAT / HUGPY_LOOP_GUARD_SIMILARITY. Defaults are
    deliberately conservative: a chunk must be a near-duplicate of the previous
    one for 2 consecutive continue-passes (i.e. 3 near-identical chunks in a row)
    before we finalize, and 'near-duplicate' means a >=0.95 similarity ratio."""
    try:
        max_repeat = int(os.environ.get("HUGPY_LOOP_GUARD_MAX_REPEAT", "2"))
    except (TypeError, ValueError):
        max_repeat = 2
    try:
        similarity = float(os.environ.get("HUGPY_LOOP_GUARD_SIMILARITY", "0.95"))
    except (TypeError, ValueError):
        similarity = 0.95
    return max(1, max_repeat), similarity


def _normalize_for_guard(text: str) -> str:
    """Collapse whitespace + lowercase so trivial spacing/case differences
    between two loop-chunks don't hide an otherwise-identical repeat."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _merge_usage(total: "dict | None", part: "dict | None") -> "dict | None":
    """Sum two usage dicts key-wise (prompt/completion/total tokens).

    The unbounded continue-loop runs several engine passes per request; each
    pass may report its own usage, and the request's real cost is the sum.
    Non-int values are ignored, either side may be None — never raises."""
    if not part:
        return total
    if not total:
        total = {}
    out = dict(total)
    for k, v in part.items():
        if isinstance(v, int):
            out[k] = (out.get(k) or 0) + v
    return out or None


def _chunks_are_similar(prev_norm: str, cur_norm: str, threshold: float) -> bool:
    """True when two normalized continue-chunks are (near-)identical. Empty
    inputs are never 'similar' — the first continue-pass has no predecessor, so
    the guard cannot trip on it. Compares trailing windows to bound cost on very
    large chunks; two looping chunks share their tail, distinct chunks don't."""
    if not prev_norm or not cur_norm:
        return False
    if prev_norm == cur_norm:
        return True
    a, b = prev_norm[-_LOOP_GUARD_WINDOW:], cur_norm[-_LOOP_GUARD_WINDOW:]
    return SequenceMatcher(None, a, b).ratio() >= threshold


class LlamaCppBaseRunner(ABC):
    """Shared scaffolding — event loop, unbounded loop, logging.
    Subclasses implement only the raw I/O."""

    model_key: str

    # --- abstract I/O hooks ------------------------------------------------

    @abstractmethod
    async def _iter_stream(
        self,
        messages: list[dict],
        max_tokens: int,
        temp: float,
        top_p: float,
        extras: Optional[dict] = None,
    ) -> AsyncIterator[tuple[str, Optional[str]]]:
        """Yield (text_chunk, finish_reason_or_None) pairs from the backend.

        ``extras`` (t74, may be None): per-request engine chat keys —
        ``chat_template_kwargs`` / ``logit_bias`` — for the chat-completion
        body. The HTTP runner forwards them verbatim to llama-server; the
        in-process runner honors what llama_cpp can and says what it can't.
        """
        ...
    @abstractmethod
    def _chat_complete(
        self,
        messages: list[dict],
        max_tokens: int,
        temp: float,
        top_p: float,
        stop: Optional[list[str]],
        extras: Optional[dict] = None,
    ) -> tuple[str, str]:
        """Chat-template path. Return (text, finish_reason)."""
        ...

    @abstractmethod
    def _raw_complete(
        self,
        prompt: str,
        max_tokens: int,
        temp: float,
        top_p: float,
        stop: Optional[list[str]],
        return_full_text: bool,
    ) -> tuple[str, str]:
        """Raw-prompt fallback path. Return (text, finish_reason)."""
        ...
    # _blocking_complete is now FINAL — no override needed in subclasses
    def _blocking_complete(
        self,
        messages: list[dict] | str,
        max_tokens: int,
        temp: float,
        top_p: float,
        stop: Optional[list[str]],
        use_chat_template: bool,
        return_full_text: bool,
        extras: Optional[dict] = None,
    ) -> tuple[str, str]:
        if use_chat_template and isinstance(messages, list):
            return self._chat_complete(messages, max_tokens, temp, top_p, stop,
                                       extras=extras)

        # The raw /completion path renders no chat template, so
        # chat_template_kwargs cannot apply here — say so rather than silently
        # dropping a selected suppression (the /no_think directive in the
        # prompt text still rides).
        if extras:
            logger.info("raw-prompt path cannot apply engine chat extras %s "
                        "for %s — relying on the /no_think directive",
                        sorted(extras), self.model_key)
        prompt = (
            messages
            if isinstance(messages, str)
            else messages_to_prompt_from_dicts(messages)
        )
        return self._raw_complete(prompt, max_tokens, temp, top_p, stop, return_full_text)
    # --- multimodal: fold an image attachment into the chat ----------------

    def _attach_image(self, messages: list, req: ChatRequest) -> list:
        """Vision GGUFs only: fold image attachments into the latest user turn as
        OpenAI ``image_url`` parts so the multimodal chat handler sees them.

        Two sources, both supported: ``req.images`` — inline base64 (raw or a full
        ``data:`` URI), the no-upload path used by the chat box — and ``req.file``,
        an uploaded image path. No-op for text models / non-image files / no
        attachments, so text chat and non-vision runners are completely unaffected.
        """
        if not getattr(self, "is_vision", False):
            return messages
        import base64, mimetypes
        parts: list = []
        # Inline base64 images (no upload round-trip).
        for img in (getattr(req, "images", None) or []):
            if not img:
                continue
            url = img if str(img).startswith("data:") else f"data:image/png;base64,{img}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        # Uploaded image file path.
        path = getattr(req, "file", None)
        if path and os.path.isfile(path):
            mime = mimetypes.guess_type(path)[0] or "image/png"
            if mime.startswith("image/"):
                with open(path, "rb") as fh:
                    data_uri = f"data:{mime};base64," + base64.b64encode(fh.read()).decode("ascii")
                parts.append({"type": "image_url", "image_url": {"url": data_uri}})
        if not parts:
            return messages
        for m in reversed(messages):
            if m.get("role", "user") == "user":
                content = m.get("content")
                if isinstance(content, list):
                    content.extend(parts)
                elif content:
                    m["content"] = [{"type": "text", "text": content}, *parts]
                else:
                    m["content"] = list(parts)
                break
        else:
            messages.append({"role": "user", "content": list(parts)})
        return messages

    # --- per-request engine chat extras (t74 hard no-think) ------------------

    @staticmethod
    def _engine_extras(req) -> dict:
        """The per-request engine chat keys riding this ChatRequest, or {}.

        ``chat_template_kwargs`` (the Qwen3 ``enable_thinking:false`` idiom —
        suppression enforced by the chat TEMPLATE, for models that ignore the
        /no_think soft switch) and ``logit_bias`` (the <think>-token-ban
        fallback). getattr-tolerant so a req built from an older schema simply
        yields {} — the directive+strip seam (utils/no_think.py) still applies.
        """
        extras = {}
        for k in ("chat_template_kwargs", "logit_bias"):
            v = getattr(req, k, None)
            if isinstance(v, dict) and v:
                extras[k] = v
        return extras

    # --- usage accounting ---------------------------------------------------
    # Subclasses stash the engine-reported usage dict of the CURRENT pass here
    # from _iter_stream (llama-server's final SSE chunk carries one on recent
    # builds; in-process llama_cpp streams don't). The stream drivers below
    # pop it per pass via _take_stream_usage.
    _stream_usage: "Optional[dict]" = None

    def _take_stream_usage(self) -> "Optional[dict]":
        usage, self._stream_usage = self._stream_usage, None
        return usage

    # --- decode-rate accounting (operator, 2026-07-25) ----------------------
    # "in the end it is about maximizing tok/s ... lets start recording this".
    #
    # llama-server measures its own decode rate and reports it in a ``timings``
    # block: ``predicted_per_second`` is authoritative tok/s straight from the
    # engine. We were discarding it. Captured here, EXACTLY beside the usage
    # slot and with the same take-once discipline, because it arrives on the
    # very same JSON object — the final SSE chunk when streaming, the response
    # body when not. Adding no timing code of our own is the whole point: a
    # wall-clock rate we computed at the relay would include queueing, network
    # and prompt-eval time and would not be decode tok/s at all.
    #
    # ⚠ RECORDING ONLY. Nothing ranks on this yet; eviction.sort_key is
    # untouched.
    _stream_timings: "Optional[dict]" = None

    def _take_stream_timings(self) -> "Optional[dict]":
        t, self._stream_timings = self._stream_timings, None
        if isinstance(t, dict) and t and "served" not in t:
            # WHAT SERVED THIS CALL (2026-09-23): the seat's own statement of
            # the GGUF file and the layer placement it is running, beside the
            # engine's timings, so central stamps the call ledger with the
            # quant/allocation hugpy actually used — never with what a caller
            # asked for. Rides the existing timings wire (DoneEvent.timings /
            # TaskResult extra), no schema change. Absent when unknowable.
            served = self._served_identity()
            if served:
                t = {**t, "served": served}
        return t

    def _served_identity(self) -> "Optional[dict]":
        """{model_file, n_gpu_layers, total_layers, n_cpu_moe, slot_id,
        loaded_at, source} for the endpoint this runner talks to, read from the
        slot control's ``/status`` (slot seats) or llama-server's ``/props``
        (managed servers: model_path only). One local GET at completion;
        None on any failure — never raises, never guesses."""
        base = getattr(self, "base_url", None)
        if not base:
            return None
        import os
        try:
            import httpx
        except Exception:  # noqa: BLE001
            return None
        for path, source in (("/status", "slot-status"), ("/props", "llama-props")):
            try:
                resp = httpx.get(str(base).rstrip("/") + path, timeout=0.5)
                if resp.status_code != 200:
                    continue
                d = resp.json()
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(d, dict) or not d.get("model_path"):
                continue
            out = {"model_file": os.path.basename(str(d["model_path"])),
                   "model_path": str(d["model_path"]),
                   "n_gpu_layers": d.get("n_gpu_layers"),
                   "total_layers": d.get("total_layers"),
                   "n_cpu_moe": d.get("n_cpu_moe"),
                   "slot_id": d.get("slot_id"),
                   "slot_model_key": d.get("model_key"),
                   "loaded_at": d.get("loaded_at"),
                   "source": source}
            return {k: v for k, v in out.items() if v is not None}
        return None

    def _usage_for(self, engine_usage, messages, completion_text) -> "Optional[dict]":
        """Best-effort token accounting for a DoneEvent — never load-bearing.

        Prefers what the engine itself reported (exact). Otherwise, runners
        that expose the model's own tokenizer (_count_tokens on the in-process
        runner — the common /v1 serving path, whose streamed chunks carry no
        usage) count the prompt and completion directly; the +8/message mirrors
        _fit_chat's per-message template overhead, so prompt_tokens is a close
        (not byte-exact) figure. None when neither source exists (e.g. an HTTP
        runner against an old llama-server) — callers must degrade, not crash.
        """
        if isinstance(engine_usage, dict) and engine_usage:
            return engine_usage
        counter = getattr(self, "_count_tokens", None)
        if counter is None:
            return None
        try:
            prompt = sum(
                counter(m.get("content") if isinstance(m.get("content"), str)
                        else str(m.get("content") or "")) + 8
                for m in messages if isinstance(m, dict)
            )
            completion = counter(completion_text or "")
            return {"prompt_tokens": prompt, "completion_tokens": completion,
                    "total_tokens": prompt + completion}
        except Exception:
            return None

    # --- shared streaming --------------------------------------------------

    async def stream_chat(
        self,
        req: ChatRequest,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[StreamEvent]:
        max_tokens = resolve_max_tokens(req.max_new_tokens)
        temp      = resolve_temperature(req.temperature, req.do_sample)
        top_p     = resolve_top_p(req.top_p)
        messages  = self._attach_image(messages_to_dicts(req.messages), req)
        extras    = self._engine_extras(req)
        output_chunks = 0
        last_finish: Optional[str] = None
        full_text = ""          # for tokenizer-based usage accounting
        self._stream_usage = None
        self._stream_timings = None

        # Bind the upstream iterator so a client disconnect (GeneratorExit at a
        # yield below) deterministically closes the llama-server HTTP stream in
        # the finally — abandoning the backend task frees the slot. Relying on GC
        # finalization (the previous bare `async for`) left slots is_processing
        # after the caller was gone (incident 2026-09-25).
        it = self._iter_stream(messages, max_tokens, temp, top_p,
                               extras=extras or None)
        try:
            async for text, fr in it:
                if cancel_event and cancel_event.is_set():
                    self._log_done(req, "cancelled", output_chunks, max_tokens)
                    yield DoneEvent(request_id=req.request_id, input_tokens=0,
                                   output_chunks=output_chunks, finish_reason="cancelled")
                    return
                if text:
                    output_chunks += 1
                    full_text += text
                    yield TokenEvent(request_id=req.request_id, text=text)
                if fr is not None:
                    last_finish = fr

            mapped = map_finish_reason(last_finish)
            self._log_done(req, mapped, output_chunks, max_tokens)
            yield DoneEvent(request_id=req.request_id, input_tokens=0,
                           output_chunks=output_chunks, finish_reason=mapped,
                           usage=self._usage_for(self._take_stream_usage(),
                                                 messages, full_text),
                           # Engine-measured decode rate, when the engine
                           # reported one. Unlike usage there is NO fallback
                           # here on purpose: a tokenizer can count tokens after
                           # the fact, but nothing can reconstruct decode speed
                           # from outside the engine. None = unmeasured.
                           timings=self._take_stream_timings())
        except GeneratorExit:
            logger.info("stream_chat client-disconnect: cancelling llama-server "
                        "stream model=%s req=%s", self.model_key, req.request_id)
            raise
        except Exception as exc:
            logger.exception("stream_chat failed: model=%s req=%s", self.model_key, req.request_id)
            yield ErrorEvent(request_id=req.request_id, message=f"{type(exc).__name__}: {exc}")
        finally:
            _ac = getattr(it, "aclose", None)
            if _ac is not None:
                try:
                    await _ac()
                except Exception:  # noqa: BLE001 — teardown must never raise
                    pass

    # --- shared unbounded streaming ----------------------------------------

    async def stream_chat_unbounded(
        self,
        req: ChatRequest,
        cancel_event: Optional[asyncio.Event] = None,
        *,
        chunk_tokens: int = 1024,
        max_chunks: int = None,
    ) -> AsyncIterator[StreamEvent]:
        import os as _os
        if max_chunks is None:
            # High ceiling so "unbounded" really means until-the-model-stops,
            # not a hidden 8-chunk cap. Still bounded so a looping model can't
            # run forever. Override with HUGPY_MAX_CHUNKS.
            try:
                max_chunks = int(_os.environ.get("HUGPY_MAX_CHUNKS", "256"))
            except ValueError:
                max_chunks = 256
        temp     = resolve_temperature(req.temperature, req.do_sample)
        top_p    = resolve_top_p(req.top_p)
        convo    = self._attach_image(messages_to_dicts(req.messages), req)
        extras   = self._engine_extras(req)
        initial_convo = list(convo)   # usage: count the CALLER's prompt, not the continue-nudges
        output_chunks = 0
        last_finish = "stop"
        full_text = ""                # all generated text, for tokenizer usage
        usage_sum: Optional[dict] = None
        self._stream_usage = None
        self._stream_timings = None
        # Decode rate across a MULTI-PASS continuation. Unlike usage above this
        # is NOT summed — tok/s is a RATE, and adding two rates is meaningless
        # (two passes at 115 tok/s is still 115 tok/s, not 230). Each pass
        # re-measures the same steady-state decode speed on the same placement,
        # so the last pass that reported one is the representative sample and
        # also the freshest. Kept per-pass rather than after the loop because
        # _take_stream_timings is take-once and the next pass would clear it.
        timings_last: Optional[dict] = None
        # Anti-repetition guard state (see _loop_guard_params / _chunks_are_similar).
        guard_max_repeat, guard_sim = _loop_guard_params()
        prev_norm = ""
        repeat_run = 0

        try:
            for _ in range(max_chunks):
                if cancel_event and cancel_event.is_set():
                    self._log_done(req, "cancelled", output_chunks, chunk_tokens)
                    yield DoneEvent(request_id=req.request_id, input_tokens=0,
                                   output_chunks=output_chunks, finish_reason="cancelled")
                    return

                piece_text = ""
                chunk_finish: Optional[str] = None

                # Bind the per-pass upstream iterator so a client disconnect
                # closes the llama-server HTTP stream deterministically (see
                # stream_chat) — a fresh iterator per continuation pass.
                _it = self._iter_stream(convo, chunk_tokens, temp,
                                        top_p, extras=extras or None)
                try:
                    async for text, fr in _it:
                        if cancel_event and cancel_event.is_set():
                            self._log_done(req, "cancelled", output_chunks, chunk_tokens)
                            yield DoneEvent(request_id=req.request_id, input_tokens=0,
                                           output_chunks=output_chunks, finish_reason="cancelled")
                            return
                        if text:
                            output_chunks += 1
                            piece_text += text
                            yield TokenEvent(request_id=req.request_id, text=text)
                        if fr is not None:
                            chunk_finish = fr
                finally:
                    _ac = getattr(_it, "aclose", None)
                    if _ac is not None:
                        try:
                            await _ac()
                        except Exception:  # noqa: BLE001 — teardown must never raise
                            pass

                full_text += piece_text
                # Each engine pass may report its own usage; the request costs
                # their sum (the tokenizer fallback in _usage_for covers engines
                # that report none).
                usage_sum = _merge_usage(usage_sum, self._take_stream_usage())
                timings_last = self._take_stream_timings() or timings_last
                last_finish = chunk_finish or "stop"
                if last_finish != "length" or not piece_text:
                    break

                # Loop-guard: if this length-chunk is a near-duplicate of the
                # previous one for guard_max_repeat consecutive passes, the model
                # is looping rather than making progress — finalize what we have.
                cur_norm = _normalize_for_guard(piece_text)
                repeat_run = (repeat_run + 1
                              if _chunks_are_similar(prev_norm, cur_norm, guard_sim)
                              else 0)
                prev_norm = cur_norm
                if repeat_run >= guard_max_repeat:
                    logger.warning(
                        "loop-guard tripped: model=%s reason=repetition "
                        "consecutive_near_dups=%s sim>=%.2f chunks=%s — finalizing early",
                        self.model_key, repeat_run, guard_sim, output_chunks)
                    # Report a clean stop (not 'length') so any client that
                    # auto-continues on truncation doesn't re-drive the loop.
                    last_finish = "stop"
                    break

                convo.append({"role": "assistant", "content": piece_text})
                convo.append({"role": "user", "content": "continue"})

            mapped = map_finish_reason(last_finish)
            self._log_done(req, mapped, output_chunks, chunk_tokens)
            yield DoneEvent(request_id=req.request_id, input_tokens=0,
                           output_chunks=output_chunks, finish_reason=mapped,
                           usage=self._usage_for(usage_sum, initial_convo, full_text),
                           timings=timings_last)
        except GeneratorExit:
            logger.info("stream_chat_unbounded client-disconnect: cancelling "
                        "llama-server stream model=%s req=%s",
                        self.model_key, req.request_id)
            raise
        except Exception as exc:
            logger.exception("stream_chat_unbounded failed: model=%s req=%s", self.model_key, req.request_id)
            yield ErrorEvent(request_id=req.request_id, message=f"{type(exc).__name__}: {exc}")

    # --- shared non-streaming ----------------------------------------------

    async def generate_text_async(self, messages, **kw) -> str:
        return await asyncio.to_thread(self.generate_text, messages, **kw)

    def generate_text(self, messages, *, max_new_tokens=0, temperature=0.0,
                      top_p=1.0, do_sample=False, use_chat_template=True,
                      return_full_text=False, stop=None,
                      chat_template_kwargs=None, logit_bias=None, **_) -> str:
        max_tokens = resolve_max_tokens(max_new_tokens)
        temp       = resolve_temperature(temperature, do_sample)
        top_p_val  = resolve_top_p(top_p)
        extras = {k: v for k, v in (("chat_template_kwargs", chat_template_kwargs),
                                    ("logit_bias", logit_bias))
                  if isinstance(v, dict) and v}
        # _blocking_complete returns (text, finish_reason) per the base contract;
        # unpack so a one-shot run() yields a str (not a tuple) into ChatResult.text.
        text, _finish = self._blocking_complete(
            messages, max_tokens, temp, top_p_val, stop, use_chat_template,
            return_full_text, extras=extras or None
        )
        return text

    def generate_text_unbounded(self, messages, *, chunk_tokens=1024,
                                max_chunks=None, temperature=0.0, top_p=1.0,
                                do_sample=False, stop=None,
                                chat_template_kwargs=None, logit_bias=None,
                                **_) -> str:
        import os as _os
        if max_chunks is None:
            try:
                max_chunks = int(_os.environ.get("HUGPY_MAX_CHUNKS", "256"))
            except ValueError:
                max_chunks = 256
        temp      = resolve_temperature(temperature, do_sample)
        top_p_val = resolve_top_p(top_p)
        extras = {k: v for k, v in (("chat_template_kwargs", chat_template_kwargs),
                                    ("logit_bias", logit_bias))
                  if isinstance(v, dict) and v}
        accumulated = ""
        convo = list(messages)
        # Anti-repetition guard state (mirrors stream_chat_unbounded).
        guard_max_repeat, guard_sim = _loop_guard_params()
        prev_norm = ""
        repeat_run = 0

        for chunk_idx in range(max_chunks):
            text, finish = self._blocking_complete(
                convo, chunk_tokens, temp, top_p_val, stop,
                use_chat_template=True, return_full_text=False,
                extras=extras or None
            )
            accumulated += text
            logger.info("generate_text_unbounded chunk=%s model=%s finish=%s",
                       chunk_idx, self.model_key, finish)
            if finish != "length" or not text:
                break

            # Loop-guard: stop when consecutive length-chunks are near-duplicates
            # (the model is looping, not progressing). Conservative — see
            # _loop_guard_params / _chunks_are_similar.
            cur_norm = _normalize_for_guard(text)
            repeat_run = (repeat_run + 1
                          if _chunks_are_similar(prev_norm, cur_norm, guard_sim)
                          else 0)
            prev_norm = cur_norm
            if repeat_run >= guard_max_repeat:
                logger.warning(
                    "loop-guard tripped: model=%s reason=repetition "
                    "consecutive_near_dups=%s sim>=%.2f chunk=%s — finalizing early",
                    self.model_key, repeat_run, guard_sim, chunk_idx)
                break

            convo = convo + [{"role": "assistant", "content": text},
                             {"role": "user", "content": "continue"}]

        return accumulated

    # --- shared internals --------------------------------------------------

    def _log_done(self, req: ChatRequest, finish: str, chunks: int, cap: int) -> None:
        logger.info("stream_chat done: model=%s req=%s finish=%s chunks=%s cap=%s",
                   self.model_key, req.request_id, finish, chunks, cap)
