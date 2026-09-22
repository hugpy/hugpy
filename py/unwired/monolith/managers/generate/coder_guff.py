# deepcoder/coder_gguf.py
import os
from typing import Any, Dict, List, Optional, Union

from hugpy_engine.generate.config import DeepCoderConfig


class DeepCoderGGUF:
    """GGUF / llama.cpp runtime with the same basic surface as DeepCoder."""

    def __init__(self, cfg: DeepCoderConfig):
        from llama_cpp import Llama   # extra: abstract_hugpy_dev[engine]

        self.cfg = cfg

        cpu_threads = cfg.cpu_threads or os.cpu_count() or 1

        self.llm = Llama(
            model_path=cfg.model_dir,
            n_ctx=getattr(cfg, "n_ctx", cfg.max_new_tokens_cap),
            n_threads=cpu_threads,
            n_threads_batch=cpu_threads,
            n_gpu_layers=0,
            verbose=False,
        )

    def _resolve_max_new_tokens(self, max_new_tokens: Optional[int]) -> int:
        # Clamp, never reject: the cap is a per-pass budget and the chat
        # route's default is the model's full context. See
        # DeepCoder._resolve_max_new_tokens for the full rationale.
        requested = max_new_tokens or self.cfg.max_new_tokens_cap

        if requested <= 0:
            requested = self.cfg.max_new_tokens_cap

        return min(requested, self.cfg.max_new_tokens_cap)

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role.upper()}:\n{content}")

        parts.append("ASSISTANT:\n")

        return "\n\n".join(parts)

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
        **_: Any,
    ) -> str:
        requested_max_new_tokens = self._resolve_max_new_tokens(max_new_tokens)

        if isinstance(prompt, str):
            rendered_prompt = prompt
        else:
            rendered_prompt = self._format_messages(prompt)

        output = self.llm(
            rendered_prompt,
            max_tokens=requested_max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p,
            echo=return_full_text,
        )

        return output["choices"][0]["text"].strip()

    # Back-compat if anything still calls .generate(...)
    def generate(self, prompt, **kwargs) -> str:
        return self.generate_text(prompt, **kwargs)
