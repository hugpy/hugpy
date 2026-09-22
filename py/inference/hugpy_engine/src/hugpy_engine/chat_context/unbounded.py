# managers/unbounded.py
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass(frozen=True)
class GenerationOutcome:
    text: str
    finish_reason: str   # 'stop' | 'length' | 'error' — runner's own vocabulary OK here
    usage: Optional[dict] = None


# A unit-of-work: messages + cap -> outcome. Each runner exposes one of these.
GenerateOnce = Callable[[list[dict], int], GenerationOutcome]
_FINISH_REASON_MAP = {"length": "max_tokens", "stop": "stop", None: "stop"}

def map_finish_reason(raw: Optional[str]) -> str:
    return _FINISH_REASON_MAP.get(raw, "stop")

def run_unbounded(
    generate_once: GenerateOnce,
    messages: list[dict],
    *,
    chunk_tokens: int = 1024,
    max_chunks: int = 8,
    continue_nudge: str = "continue",
) -> GenerationOutcome:
    """Drive any single-shot generator until natural stop, EOS, or chunk cap.

    Runner-agnostic. The runner only has to tell us what happened in one call;
    we own the convo-extension and the stop-condition.
    """
    convo = list(messages)
    accumulated = ""
    last_finish = "stop"
    last_usage = None

    for _ in range(max_chunks):
        out = generate_once(convo, chunk_tokens)
        accumulated += out.text
        last_finish = out.finish_reason
        last_usage = out.usage

        if last_finish != "length" or not out.text:
            break

        convo = convo + [
            {"role": "assistant", "content": out.text},
            {"role": "user", "content": continue_nudge},
        ]

    return GenerationOutcome(
        text=accumulated,
        finish_reason=last_finish,
        usage=last_usage,
    )
