
# deepcoder/coder.py
import os
import asyncio
import threading
import time
from contextlib import asynccontextmanager

from typing import Any, AsyncIterator, Dict, List, Optional, Union
from abstract_essentials import get_logFile
from hugpy_engine.schemas.chat_schemas import ChatRequest
from hugpy_platform.module_imports import get_torch, get_transformers, require, require_peft
from hugpy_platform.utils import messages_to_dicts

from hugpy_engine.generate.config import (
    DeepCoderConfig,
    build_deepcoder_config,
    CancelStoppingCriteria,
    DoneEvent,
    ErrorEvent,
    TokenEvent,
    StreamEvent,
    _SENTINEL,
)
logger = get_logFile("deepcoder")


class DeepCoder:
    """Loaded model + tokenizer + generation_config. One instance per config."""

    def __init__(self, cfg: DeepCoderConfig):
        require("transformers", reason="DeepCoder requires HuggingFace transformers")

        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        # Serialize concurrent generations with a THREADING semaphore, not an
        # asyncio one. Generation runs in a worker thread (asyncio.to_thread) and
        # every request drives its own event loop, so an asyncio.Semaphore cached
        # on this singleton binds to the first request's loop and then raises
        # "bound to a different event loop" on the next request. A threading
        # semaphore is loop-agnostic AND correctly limits across the gunicorn /
        # per-request-loop threads (same pattern as python_runner.generate_lock).
        self._gen_semaphore = threading.BoundedSemaphore(
            max(1, int(cfg.max_concurrent_generations or 1))
        )

        self._configure_cpu_runtime()
        self._load_tokenizer()
        self._load_model()
        self._load_generation_config()

        logger.info(
            "DeepCoder ready: device=%s dtype=%s concurrency=%d max_tokens_cap=%d",
            cfg.device,
            cfg.torch_dtype,
            cfg.max_concurrent_generations,
            cfg.max_new_tokens_cap,
        )

    def _configure_cpu_runtime(self) -> None:
        """
        Configure PyTorch CPU thread usage.

        For maximum CPU assigned to a single generation:
          - cpu_threads should usually be os.cpu_count()
          - cpu_interop_threads should usually be 1
          - max_concurrent_generations should usually be 1
        """
        if self.cfg.device == "cuda":
            return

        torch = get_torch()

        cpu_threads = self.cfg.cpu_threads or os.cpu_count() or 1
        interop_threads = self.cfg.cpu_interop_threads or 1

        os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))

        try:
            torch.set_num_threads(cpu_threads)
        except Exception as exc:
            logger.warning("failed to set torch num threads: %s", exc)

        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            logger.warning("torch interop threads already initialized")
        except Exception as exc:
            logger.warning("failed to set torch interop threads: %s", exc)

        logger.info(
            "CPU runtime configured: cpu_threads=%s interop_threads=%s",
            cpu_threads,
            interop_threads,
        )

    def _load_model(self) -> None:
        AutoModelForCausalLM = get_transformers("AutoModelForCausalLM")

        kwargs: Dict[str, Any] = {
            "torch_dtype": self.cfg.torch_dtype,
            "local_files_only": self.cfg.local_files_only,
            "low_cpu_mem_usage": True,
            "trust_remote_code": self.cfg.trust_remote_code,  # opt-in; default OFF
        }

        if self.cfg.device == "cuda":
            kwargs["device_map"] = "auto"
            # GPU/CPU spill: a max_memory budget lets accelerate shard layers to
            # fit VRAM and offload the overflow to CPU/RAM, so a model larger
            # than the card can still run. None (no GPU / unset) leaves the
            # plain device_map="auto" behavior untouched.
            from hugpy_engine.spill import transformers_max_memory

            max_memory = transformers_max_memory()
            if max_memory:
                kwargs["max_memory"] = max_memory

        if self.cfg.use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"

        # 4-bit: the OPERATOR's per-model lever (console "4-bit" column, ridden
        # in on the spill wire as HUGPY_BNB_4BIT) OR this runner's own
        # use_quantization config. Either turns it on; the env lever is what
        # makes the console switch actually reach the loader.
        #
        # THE BUG THIS CLOSES: the lever only ever re-priced central's
        # DERIVATION, so the console showed "13.1 GiB VRAM planned" while the
        # worker loaded fp16 and refused with "needs 50.2 GB". The projection was
        # right about what 4-bit WOULD cost; nothing had told the loader to
        # actually do it.
        from hugpy_engine.spill import bnb_4bit_env
        if self.cfg.use_quantization or bnb_4bit_env():
            BitsAndBytesConfig = get_transformers("BitsAndBytesConfig")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.cfg.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            # A 4-bit model must NOT also be handed an fp16-sized max_memory
            # budget: accelerate would shard against a footprint ~3x larger than
            # the load actually needs and spill layers to CPU for no reason.
            kwargs.pop("max_memory", None)
            logger.info("loading %s in 4-bit (nf4, double-quant)",
                        getattr(self.cfg, "model_dir", "model"))

        # model_dir is the BASE model in both cases. adapter_dir, when set,
        # is a LoRA delta loaded on top of it. Same kwargs feed both loads.
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_dir, **kwargs)

        adapter_dir = getattr(self.cfg, "adapter_dir", None)
        if adapter_dir:
            PeftModel = require_peft()
            logger.info("attaching PEFT adapter: base=%s adapter=%s",
                        self.cfg.model_dir, adapter_dir)
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_dir,
                local_files_only=self.cfg.local_files_only,
                is_trainable=False,
            )

        if self.cfg.device != "cuda":
            self.model = self.model.to(self.cfg.device)

        self.model.eval()

    def _load_generation_config(self) -> None:
        GenerationConfig = get_transformers("GenerationConfig")

        try:
            generation_config = GenerationConfig.from_pretrained(
                self.cfg.model_dir,
                local_files_only=self.cfg.local_files_only,
            )
        except Exception:
            generation_config = GenerationConfig()

        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.use_cache = True

        self.generation_config = generation_config

    @asynccontextmanager
    async def _gen_slot(self):
        """Hold a generation slot for this stream — loop-agnostic, process-wide.

        Acquires the threading semaphore via run_in_executor so a wait never
        blocks this request's event loop, and always releases on exit (including
        errors/cancellation). Replaces the old cached ``asyncio.Semaphore`` that
        crashed with "bound to a different event loop" on the second request.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._gen_semaphore.acquire)
        try:
            yield
        finally:
            self._gen_semaphore.release()

    def _resolve_max_new_tokens(self, max_new_tokens: Optional[int]) -> int:
        """
        max_new_tokens is output budget, not CPU power.

        None / 0 / negative means:
            use this runtime's configured max_new_tokens_cap.

        Above-cap requests are CLAMPED, never rejected: the cap is a per-pass
        budget, not a validity check. The chat route defaults max_new_tokens
        to the model's full context (e.g. 131072) and relies on unbounded
        auto-continuation to chain capped passes, so an oversized number is
        the normal case — raising here turned every default chat into an
        error. Raise cfg.max_new_tokens_cap via DeepCoderRuntime to allow
        bigger single passes.
        """
        requested = max_new_tokens or self.cfg.max_new_tokens_cap

        if requested <= 0:
            requested = self.cfg.max_new_tokens_cap

        if requested > self.cfg.max_new_tokens_cap:
            logger.info(
                "max_new_tokens=%s exceeds cap=%s; clamping to cap "
                "(per-pass budget; auto-continuation covers the rest)",
                requested, self.cfg.max_new_tokens_cap,
            )
            requested = self.cfg.max_new_tokens_cap

        return requested

    async def stream_chat(
        self,
        request: ChatRequest,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield zero-or-more TokenEvent then exactly one DoneEvent or ErrorEvent."""
        torch = get_torch()
        TextIteratorStreamer = get_transformers("TextIteratorStreamer")
        StoppingCriteriaList = get_transformers("StoppingCriteriaList")

        try:
            requested_max_new_tokens = self._resolve_max_new_tokens(
                request.max_new_tokens,
            )
        except ValueError as exc:
            yield ErrorEvent(
                request_id=request.request_id,
                message=str(exc),
            )
            return

        async with self._gen_slot():
            messages = messages_to_dicts(request.messages)

            template_out = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            if hasattr(template_out, "shape"):
                input_ids = template_out.to(self.cfg.device)
                model_inputs = {"input_ids": input_ids}
            else:
                model_inputs = {
                    key: value.to(self.cfg.device) if hasattr(value, "to") else value
                    for key, value in template_out.items()
                }
                input_ids = model_inputs["input_ids"]

            input_len = int(input_ids.shape[-1])

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            thread_cancel = threading.Event()
            stopping = StoppingCriteriaList(
                [CancelStoppingCriteria(thread_cancel)],
            )

            gen_kwargs: Dict[str, Any] = {
                **model_inputs,
                "max_new_tokens": requested_max_new_tokens,
                "do_sample": bool(request.do_sample),
                "use_cache": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
                "stopping_criteria": stopping,
            }

            if request.do_sample:
                gen_kwargs["temperature"] = request.temperature
                gen_kwargs["top_p"] = request.top_p

            generation_error: Optional[BaseException] = None

            def _run_generate() -> None:
                nonlocal generation_error

                try:
                    with torch.inference_mode():
                        self.model.generate(**gen_kwargs)
                except BaseException as exc:
                    generation_error = exc
                    streamer.end()

            gen_task = asyncio.create_task(asyncio.to_thread(_run_generate))

            cancel_task: Optional[asyncio.Task] = None

            if cancel_event is not None:

                async def _watch_cancel() -> None:
                    await cancel_event.wait()
                    thread_cancel.set()

                cancel_task = asyncio.create_task(_watch_cancel())

            loop = asyncio.get_running_loop()
            output_chunks = 0

            try:
                streamer_iter = iter(streamer)

                while True:
                    chunk = await loop.run_in_executor(
                        None,
                        next,
                        streamer_iter,
                        _SENTINEL,
                    )

                    if chunk is _SENTINEL:
                        break

                    if chunk == "":
                        continue

                    output_chunks += 1

                    yield TokenEvent(
                        request_id=request.request_id,
                        text=chunk,
                    )

                await gen_task

            finally:
                if cancel_task is not None and not cancel_task.done():
                    cancel_task.cancel()

                    try:
                        await cancel_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass

            if generation_error is not None:
                logger.error(
                    "generation failed for %s",
                    request.request_id,
                    exc_info=generation_error,
                )

                yield ErrorEvent(
                    request_id=request.request_id,
                    message=f"{type(generation_error).__name__}: {generation_error}",
                )
                return

            if thread_cancel.is_set():
                finish_reason = "cancelled"
            elif output_chunks >= requested_max_new_tokens:
                finish_reason = "max_tokens"
            else:
                finish_reason = "stop"

            yield DoneEvent(
                request_id=request.request_id,
                input_tokens=input_len,
                output_chunks=output_chunks,
                finish_reason=finish_reason,
            )

    def _generate_row(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        *,
        max_new_tokens: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        use_chat_template: bool = False,
    ):
        """Core sync generation shared by generate_text and generate_once.

        Returns (output_row, input_len, requested_max_new_tokens) so callers
        can either decode just the completion or also derive a finish_reason
        from the generated-token count. The tokenize/generate body lives here
        in exactly one place.
        """
        torch = get_torch()
        requested_max_new_tokens = self._resolve_max_new_tokens(max_new_tokens)

        if use_chat_template:
            messages = prompt if isinstance(prompt, list) else None

            if not messages:
                raise ValueError("use_chat_template=True requires list-form prompt")

            template_out = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            if hasattr(template_out, "shape"):
                input_ids = template_out.to(self.cfg.device)
                model_inputs = {"input_ids": input_ids}
            else:
                model_inputs = {
                    key: value.to(self.cfg.device) if hasattr(value, "to") else value
                    for key, value in template_out.items()
                }
                input_ids = model_inputs["input_ids"]

        else:
            tokenized = self.tokenizer(
                str(prompt),
                return_tensors="pt",
                padding=False,
                truncation=True,
            )

            model_inputs = {
                key: value.to(self.cfg.device)
                for key, value in tokenized.items()
            }

            input_ids = model_inputs["input_ids"]

        input_len = int(input_ids.shape[-1])

        gen_kwargs: Dict[str, Any] = {
            **model_inputs,
            "max_new_tokens": requested_max_new_tokens,
            "do_sample": bool(do_sample),
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.inference_mode():
            outputs = self.model.generate(**gen_kwargs)

        return outputs[0], input_len, requested_max_new_tokens

    def generate_text(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        *,
        max_new_tokens: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        use_chat_template: bool = False,
        return_full_text: bool = False,
    ) -> str:
        """
        Sync, non-streaming generation.

        Use stream_chat for the chat host. This is for eval scripts and batch
        jobs that do not need streaming/backpressure.
        """
        out_row, input_len, _ = self._generate_row(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            use_chat_template=use_chat_template,
        )

        ids = out_row if return_full_text else out_row[input_len:]

        return self.tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

    def generate_analysis(self, prompt, **kwargs) -> dict:
        """Generate once while preserving the measurements already available.

        The non-streaming chat adapter previously decoded only the text and
        discarded token counts, cap exhaustion, and generation wall time.
        Those are benchmark data, not implementation details.
        """
        started = time.perf_counter()
        out_row, input_len, cap = self._generate_row(prompt, **kwargs)
        elapsed = max(time.perf_counter() - started, 1e-9)
        generated = max(0, int(out_row.shape[-1]) - input_len)
        text = self.tokenizer.decode(
            out_row[input_len:], skip_special_tokens=True,
            clean_up_tokenization_spaces=True).strip()
        return {
            "text": text,
            "input_tokens": input_len,
            "completion_tokens": generated,
            "elapsed_s": elapsed,
            "tok_s": generated / elapsed,
            "finish_reason": "max_tokens" if generated >= cap else "stop",
        }

    def generate_once(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> tuple[str, str]:
        """One non-streaming pass for the unbounded driver (run_unbounded).

        Returns (text, finish_reason). finish_reason is the runner's RAW
        vocabulary: 'length' when the pass exhausted its token cap (so the
        driver should append a 'continue' nudge and run again), else 'stop'.
        This mirrors stream_chat's truncation test (output_chunks >=
        requested_max_new_tokens) for the non-streaming path, surfacing the
        finish_reason that plain generate_text throws away.
        """
        out_row, input_len, cap = self._generate_row(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            use_chat_template=True,
        )

        gen_len = int(out_row.shape[-1]) - input_len
        text = self.tokenizer.decode(
            out_row[input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        return text, ("length" if gen_len >= cap else "stop")

    def _load_tokenizer(self) -> None:
        AutoTokenizer = get_transformers("AutoTokenizer")

        # An adapter may ship its own tokenizer (added tokens, chat template).
        # Prefer it; fall back to the base model's.
        adapter_dir = getattr(self.cfg, "adapter_dir", None)
        tok_src = self.cfg.model_dir
        if adapter_dir and os.path.isfile(
            os.path.join(adapter_dir, "tokenizer_config.json")
        ):
            tok_src = adapter_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_src,
            trust_remote_code=self.cfg.trust_remote_code,  # opt-in; default OFF
            local_files_only=self.cfg.local_files_only,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
class _Registry:
    """
    Cache of DeepCoder instances keyed by DeepCoderConfig.cache_key().

    Module-level singleton, but inspectable:
      REGISTRY.keys()
      REGISTRY.evict(cfg)
      REGISTRY.clear()
    """

    def __init__(self):
        self._instances: Dict[tuple, DeepCoder] = {}
        self._lock = threading.Lock()

    def get(self, cfg: DeepCoderConfig) -> DeepCoder:
        key = cfg.cache_key()

        with self._lock:
            instance = self._instances.get(key)

            if instance is None:
                instance = DeepCoder(cfg)
                self._instances[key] = instance

            return instance

    def evict(self, cfg: DeepCoderConfig) -> bool:
        key = cfg.cache_key()

        with self._lock:
            return self._instances.pop(key, None) is not None

    def keys(self) -> List[tuple]:
        with self._lock:
            return list(self._instances.keys())

    def clear(self) -> None:
        with self._lock:
            self._instances.clear()


REGISTRY = _Registry()


def get_deep_coder(
    cfg: Optional[DeepCoderConfig] = None,
    **build_kwargs,
) -> DeepCoder:
    """Boot-time get. Pass a cfg or build one. Same key returns same instance."""
    if cfg is None:
        cfg = build_deepcoder_config(**build_kwargs)

    return REGISTRY.get(cfg)


def deep_coder_generate(prompt, **kwargs) -> str:
    """Convenience wrapper with the same surface as before."""
    gen_keys = {
        "max_new_tokens",
        "temperature",
        "top_p",
        "use_chat_template",
        "messages",
        "do_sample",
        "return_full_text",
    }

    gen_kwargs = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key in gen_keys
    }

    coder = get_deep_coder(**kwargs)
    
    return coder.generate_text(
        prompt=prompt,
        **gen_kwargs,
    )
