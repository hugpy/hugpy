from hugpy_engine.llama.runners.src.base_runner import LlamaCppBaseRunner
import asyncio
import os
import threading
from typing import Optional
from hugpy_engine.config.main import get_gguf_file, get_model_config
from hugpy_engine.llama.runners.src.imports.constants import DEFAULT_N_CTX
from hugpy_engine.llama.runners.src.imports.init_imports import logger
from hugpy_engine.llama.runners.src.imports.utils import messages_to_prompt_from_dicts
from hugpy_storage.download_models import ensure_model
# ===========================================================================
# In-process Python runner — loads a GGUF via llama_cpp directly
# ===========================================================================


def _emit_resolve_fail(model_key, resolved_path, reason):
    """Report a "nothing loadable on disk" refusal on the serve-telemetry
    stream. Wholly best-effort: this runs on the load path, and a telemetry
    import failure must not convert a clean FileNotFoundError into a mystery.
    See comms/evictions.py for why the serve pipeline shares that stream."""
    try:
        from hugpy_engine.placement import get_eviction_ledger
        _ev = get_eviction_ledger()
        _ev.emit_resolve_fail(model_key, resolved_path, reason,
                              **_ev.disk_stats(resolved_path or None))
    except Exception:  # noqa: BLE001
        pass


def _build_vision_chat_handler(model_path, mmproj, cfg):
    """Best-effort multimodal chat handler for a vision GGUF, or None.

    llama-cpp-python serves vision by pairing the model with a CLIP projector
    via a model-specific chat handler (clip_model_path=mmproj). We pick a handler
    matching the model and fall back to the generic LLaVA ones. Returns None on
    any failure — the caller then loads text-only (images ignored) rather than
    crashing. The robust path is the native llama-server (--mmproj) used by the
    serve slots; this just lets the in-process fallback see images when it can.
    """
    try:
        from llama_cpp import llama_chat_format as lcf
    except Exception as exc:                      # pragma: no cover
        logger.warning("vision: llama_chat_format unavailable (%s)", exc)
        return None
    name = f"{os.path.basename(model_path)} {getattr(cfg, 'name', '') or ''}".lower()
    candidates = []
    if "qwen2" in name and "vl" in name:
        candidates.append("Qwen25VLChatHandler")
    if "minicpm" in name:
        candidates.append("MiniCPMv26ChatHandler")
    if "llama" in name and "vision" in name:
        candidates.append("Llama3VisionAlphaChatHandler")
    candidates += ["Llava16ChatHandler", "Llava15ChatHandler"]   # generic fallback
    for cls_name in candidates:
        cls = getattr(lcf, cls_name, None)
        if cls is None:
            continue
        try:
            handler = cls(clip_model_path=os.fspath(mmproj), verbose=False)
            logger.info("vision: using %s with projector %s", cls_name, mmproj)
            return handler
        except Exception as exc:                  # pragma: no cover
            logger.warning("vision: handler %s failed (%s)", cls_name, exc)
    logger.warning("vision: no usable chat handler for %s — loading text-only; "
                   "serve via a slot (native llama-server --mmproj) for images",
                   os.path.basename(model_path))
    return None

# python_runner.py  (in-process)
class LlamaCppPythonRunner(LlamaCppBaseRunner):
    def __init__(
        self,
        model_key: str,
        *,
        n_ctx: int = DEFAULT_N_CTX,
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
    ):
        from llama_cpp import Llama
        from hugpy_engine.spill import llama_kwargs

        self.model_key = model_key
        self.cfg = get_model_config(model_key)

        model_dir = ensure_model(model_key)
        # Operator-selected .gguf variant (UI serving control) wins; else the
        # registry filename / first .gguf. No pathlib — get_gguf_file accepts
        # strings via os.fspath internally.
        model_path = None
        try:
            from hugpy_engine.serve.overrides import resolve_override_gguf
            model_path = resolve_override_gguf(model_key, model_dir)
        except Exception:
            model_path = None
        if not model_path:
            # Fit-aware quant auto-pick: prefer is None whenever any
            # designation (pin / cfg.filename) exists, so the election order
            # inside get_gguf_file is never overridden by the auto-pick.
            prefer = None
            try:
                from hugpy_engine.serve.overrides import autofit_gguf_prefer
                prefer = autofit_gguf_prefer(model_key, model_dir, self.cfg)
            except Exception:
                prefer = None
            model_path = get_gguf_file(model_dir, self.cfg, prefer=prefer)

        if not model_path:
            # Same honest checkpoint as the slot child's _build_cmd refusal:
            # the in-process loader is the OTHER way weights get opened, and a
            # request that dies here was equally invisible on the console
            # before (2026-07-28). Observation only.
            _emit_resolve_fail(model_key, model_dir,
                               "no GGUF resolved for this model on disk")
            raise FileNotFoundError(f"No GGUF file found for model_key={model_key}")

        self.model_path = os.fspath(model_path)
        # A 0-byte or truncated GGUF (interrupted download) passes get_gguf_file's
        # existence check but makes llama.cpp's native loader SIGILL (core dump),
        # taking the whole process down instead of failing one request. Guard
        # here — the last check before Llama() touches the file.
        if not os.path.isfile(self.model_path) or os.path.getsize(self.model_path) == 0:
            _emit_resolve_fail(
                model_key, self.model_path,
                "resolved GGUF does not exist"
                if not os.path.isfile(self.model_path)
                else "resolved GGUF is 0 bytes (interrupted or failed pull)")
            raise FileNotFoundError(
                f"{model_key}: GGUF missing or empty at {self.model_path!r} — "
                "refusing to load (would SIGILL llama.cpp)")
        self.n_ctx = n_ctx
        # DEFAULT_LLAMA_THREADS caps generation threads box-wide (the slot
        # agent already honors it) — an operator core budget for hugpy.
        env_threads = os.environ.get("DEFAULT_LLAMA_THREADS", "").strip()
        self.n_threads = (n_threads
                          or (int(env_threads) if env_threads.isdigit() else None)
                          or max(1, (os.cpu_count() or 4) - 1))
        self.generate_lock = threading.Lock()

        # GPU/CPU spill. The resolver/dispatch path doesn't pass n_gpu_layers,
        # so by default we derive it from the spill module (env + autofit).
        # An explicit constructor arg always wins. Without this, llama.cpp ran
        # CPU-only because n_gpu_layers was never set.
        gpu_kwargs = llama_kwargs(self.model_path)
        if n_gpu_layers is not None:
            gpu_kwargs["n_gpu_layers"] = n_gpu_layers
        self.n_gpu_layers = gpu_kwargs.get("n_gpu_layers", 0)

        # CPU-side preflight (the slot agent has the same guard): the layers
        # that DON'T offload are RAM-resident, and a load that eats past
        # MemAvailable gets the whole process OOM-killed mid-request — the
        # caller sees "peer closed connection" instead of a reason. Fail fast
        # with the reason. free_ram_bytes is reserve-adjusted
        # (HUGPY_RAM_RESERVE_GIB), so headroom for processes central can't
        # see is part of the check.
        from hugpy_engine.spill import cpu_resident_bytes, free_ram_bytes
        need = cpu_resident_bytes(self.model_path, self.n_gpu_layers)
        avail = free_ram_bytes()
        if need and avail is not None and need > avail * 0.95:
            raise RuntimeError(
                f"{model_key}: needs ~{need / 1e9:.1f} GB RAM for the "
                f"CPU-resident layers but only {avail / 1e9:.1f} GB is "
                f"budgetable (after HUGPY_RAM_RESERVE_GIB) — offload more "
                f"layers, free RAM, or pick a smaller quant")

        # Vision GGUF: load the multimodal projector beside the model via a chat
        # handler so create_chat_completion accepts image_url content. None for
        # text models (no projector) or if no handler matches → text-only.
        from hugpy_platform.utils import find_mmproj
        mmproj = find_mmproj(self.model_path)
        chat_handler = _build_vision_chat_handler(self.model_path, mmproj, self.cfg) if mmproj else None
        self.is_vision = chat_handler is not None

        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
                chat_handler=chat_handler,
                **gpu_kwargs,
            )
        except Exception as exc:
            # DRAFT-MODEL-GATE-20260910: a failed context left the just-loaded weights pinned on the
            # card (the half-built Llama lives on in the traceback). Drop it now.
            self.llm = None
            exc.__traceback__ = None
            import gc as _gc
            _gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"{model_key}: in-process load failed - {type(exc).__name__}: {exc}") from None

        logger.info(
            "LlamaCppPythonRunner ready: model=%s n_ctx=%s n_threads=%s "
            "n_gpu_layers=%s path=%s",
            model_key, self.n_ctx, self.n_threads, self.n_gpu_layers, self.model_path,
        )

    # ----- context-window fitting -----------------------------------------
    # Extra tokens the chat template / BOS / role wrappers add on top of the
    # raw message text we measure. Kept as headroom so we never tip over n_ctx.
    _CTX_SAFETY_MARGIN = 64

    def _count_tokens(self, text: str) -> int:
        """Exact token count via the loaded model's own tokenizer."""
        if not text:
            return 0
        try:
            return len(self.llm.tokenize(text.encode("utf-8"),
                                         add_bos=False, special=False))
        except Exception:
            return max(1, len(text) // 4)

    def _fit_chat(self, messages, max_tokens):
        """Compact messages + clamp max_tokens so prompt+output fit n_ctx.

        This is what stops llama.cpp's "Requested tokens (N) exceed context
        window of M" error: the conversation is trimmed (oldest turns first,
        then the middle of the newest message) to the real, tokenizer-measured
        input budget, and the output cap is shrunk to whatever room is left.
        """
        n_ctx = int(self.n_ctx)
        out = max(16, min(int(max_tokens) if max_tokens else n_ctx // 2,
                          n_ctx - 256))
        # Multimodal turns carry list content (text + image_url parts). The text
        # budget/compaction can't measure image tokens and would corrupt the
        # parts, so pass such messages through untouched with a sane output cap.
        if any(not isinstance(m.get("content"), str) for m in messages):
            return messages, out
        from hugpy_engine.chat_context.context_budget import ContextBudget, compact_messages_to_budget
        budget = ContextBudget(
            max_context_tokens=n_ctx,
            reserved_output_tokens=out,
            reserved_system_tokens=0,
        )
        fitted = compact_messages_to_budget(
            messages, budget, token_counter=self._count_tokens,
        )
        prompt_tokens = sum(
            self._count_tokens(str(m.get("content", ""))) + 8 for m in fitted
        )
        room = n_ctx - prompt_tokens - self._CTX_SAFETY_MARGIN
        out = max(16, min(out, room))
        original = len([m for m in messages if str(m.get("content", "")).strip()])
        if len(fitted) != original:
            logger.warning(
                "context fit: model=%s trimmed messages %s -> %s "
                "(n_ctx=%s, prompt~%s tok, out cap=%s)",
                self.model_key, original, len(fitted), n_ctx, prompt_tokens, out,
            )
        return fitted, out

    def _fit_raw(self, prompt, max_tokens):
        """Trim a raw prompt (head+tail) to fit n_ctx and clamp max_tokens."""
        n_ctx = int(self.n_ctx)
        out = max(16, min(int(max_tokens) if max_tokens else n_ctx // 2,
                          n_ctx - 256))
        budget_tokens = max(64, n_ctx - out - self._CTX_SAFETY_MARGIN)
        try:
            toks = self.llm.tokenize(prompt.encode("utf-8"),
                                     add_bos=True, special=True)
        except Exception:
            return prompt, out
        if len(toks) <= budget_tokens:
            return prompt, out
        keep_head = budget_tokens // 3
        keep_tail = budget_tokens - keep_head
        kept = toks[:keep_head] + toks[-keep_tail:]
        try:
            new_prompt = self.llm.detokenize(kept).decode("utf-8", errors="ignore")
        except Exception:
            approx = budget_tokens * 4
            new_prompt = prompt[: approx // 3] + prompt[-(approx - approx // 3):]
        logger.warning(
            "context fit: model=%s trimmed raw prompt %s -> %s tokens (n_ctx=%s)",
            self.model_key, len(toks), len(kept), n_ctx,
        )
        return new_prompt, out

    def _llm_extras_kwargs(self, extras) -> dict:
        """In-process analogue of the t74 llama-server body keys — honor what
        llama_cpp can, NAME what it can't (a selected suppression must never be
        a silent no-op).

        ``logit_bias`` passes through when this llama_cpp build's
        create_chat_completion accepts it. ``chat_template_kwargs`` has no
        create_chat_completion analogue (the handler renders the template with
        fixed args), so it is logged-and-skipped — the /no_think directive +
        caller-side strip (utils/no_think.py) remain the suppression here; the
        slot path (native llama-server) is where the template kwarg is honored.
        """
        if not extras:
            return {}
        kw = {}
        lb = extras.get("logit_bias")
        if lb:
            try:
                import inspect as _inspect
                accepts = "logit_bias" in _inspect.signature(
                    self.llm.create_chat_completion).parameters
            except (TypeError, ValueError):
                accepts = False
            if accepts:
                kw["logit_bias"] = lb
            else:
                logger.warning("in-process llama_cpp build ignores logit_bias "
                               "for %s", self.model_key)
        if extras.get("chat_template_kwargs"):
            logger.info("in-process llama_cpp cannot apply chat_template_kwargs "
                        "%s for %s — relying on the /no_think directive + strip",
                        extras["chat_template_kwargs"], self.model_key)
        return kw

    async def _iter_stream(self, messages, max_tokens, temp, top_p, extras=None):
        messages, max_tokens = self._fit_chat(messages, max_tokens)
        extra_kw = self._llm_extras_kwargs(extras)

        def run():
            with self.generate_lock:
                return self.llm.create_chat_completion(
                    messages=messages, max_tokens=max_tokens,
                    temperature=temp, top_p=top_p, stream=True, stop=None,
                    **extra_kw)
        stream = await asyncio.to_thread(run)
        for raw in stream:
            try:
                # llama-cpp-python stream chunks normally carry no usage; if a
                # build ever adds one (final chunk), keep it for the DoneEvent.
                # Otherwise base_runner._usage_for falls back to this runner's
                # exact tokenizer (_count_tokens).
                if isinstance(raw, dict) and raw.get("usage"):
                    self._stream_usage = raw["usage"]
                choice = raw["choices"][0]
                text = (choice.get("delta") or {}).get("content") or ""
                fr   = choice.get("finish_reason")
            except Exception:
                text, fr = "", None
            yield text, fr
            await asyncio.sleep(0)
    def _chat_complete(self, messages, max_tokens, temp, top_p, stop, extras=None):
        messages, max_tokens = self._fit_chat(messages, max_tokens)
        extra_kw = self._llm_extras_kwargs(extras)
        with self.generate_lock:
            out = self.llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens,
                temperature=temp, top_p=top_p, stop=stop, stream=False,
                **extra_kw)
        choice = out["choices"][0]
        logger.info("_chat_complete: model=%s finish=%s usage=%s cap=%s",
                    self.model_key, choice.get("finish_reason"), out.get("usage"), max_tokens)
        return choice["message"]["content"] or "", choice.get("finish_reason") or "stop"

    def _raw_complete(self, prompt, max_tokens, temp, top_p, stop, return_full_text):
        prompt, max_tokens = self._fit_raw(prompt, max_tokens)
        with self.generate_lock:
            out = self.llm(prompt, max_tokens=max_tokens, temperature=temp,
                           top_p=top_p, stop=stop, stream=False, echo=return_full_text)
        choice = out["choices"][0]
        logger.info("_raw_complete: model=%s finish=%s cap=%s",
                    self.model_key, choice.get("finish_reason"), max_tokens)
        return choice.get("text", ""), choice.get("finish_reason") or "stop"
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
        with self.generate_lock:
            if use_chat_template and isinstance(messages, list):
                messages, max_tokens = self._fit_chat(messages, max_tokens)
                out = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                    top_p=top_p,
                    stop=stop,
                    stream=False,
                    **self._llm_extras_kwargs(extras),
                )

                choice = out["choices"][0]
                text = choice["message"]["content"] or ""

                logger.info(
                    "_blocking_complete done: model=%s finish=%s usage=%s cap=%s",
                    self.model_key,
                    choice.get("finish_reason"),
                    out.get("usage"),
                    max_tokens,
                )

                return text, choice.get("finish_reason") or "stop"

            prompt = (
                messages
                if isinstance(messages, str)
                else messages_to_prompt_from_dicts(messages)
            )
            prompt, max_tokens = self._fit_raw(prompt, max_tokens)

            out = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                stop=stop,
                stream=False,
                echo=return_full_text,
            )

            choice = out["choices"][0]
            text = choice.get("text", "")

            logger.info(
                "_blocking_complete(raw) done: model=%s finish=%s cap=%s",
                self.model_key,
                choice.get("finish_reason"),
                max_tokens,
            )

            return text, choice.get("finish_reason") or "stop"


