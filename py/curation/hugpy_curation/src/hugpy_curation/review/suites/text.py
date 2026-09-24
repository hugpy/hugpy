"""hugpy-native-v2: the original 9-task x 3-tier text suite, wrapped as a Suite.

Tasks, prompts, checkers and the call path are exactly ``fleet_grading``'s; this
module only adapts them to the registry interface.
"""
from __future__ import annotations

from .. import fleet_grading

NAME = "hugpy-native-v2"
TASKS = fleet_grading.TASKS_TIERED
SEAT = "Reply only: ready"
SPEED = "Generate a standard 100 word summary of theoretical optics and microfabrication parameters."


def call(client, lane, variation, prompt, tokens):
    """fleet_grading._call's request, body and token accounting, exactly —
    plus the reply's call-ledger fields (common.served_fields: request_id,
    prompt_s, generation_s, gen_tokens, served stamp), which _call's tuple
    cannot carry. ``tok_s`` = gen_tokens / generation_s when the reply carries
    the generation split, else _call's tokens / wall seconds."""
    import time
    from .common import retrying_request, served_fields
    started = time.time()
    response, error = retrying_request(client, "/v1/chat/completions",
                                       fleet_grading._body(lane, variation, prompt, tokens),
                                       fleet_grading.FleetError)
    answer = fleet_grading._content(response) if response is not None else ""
    elapsed = time.time() - started
    usage = response.get("usage") if isinstance(response, dict) else None
    generated = usage.get("completion_tokens") if isinstance(usage, dict) else None
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    estimated = False
    if not isinstance(generated, (int, float)) or generated <= 0:
        generated = max(1, round(len(answer) / 4)) if answer else None
        estimated = generated is not None
    if not isinstance(prompt_tokens, (int, float)) or prompt_tokens <= 0:
        prompt_tokens = max(1, round(len(prompt) / 4))
    extra = served_fields(response)
    gt, gs = extra.get("gen_tokens"), extra.get("generation_s")
    if isinstance(gt, (int, float)) and gt > 0 and isinstance(gs, (int, float)) and gs > 0:
        speed = gt / gs
    else:
        speed = generated / elapsed if isinstance(generated, (int, float)) and generated > 0 and elapsed > 0 else None
    return {"answer": answer, "output": answer, "error": error, "elapsed_s": round(elapsed, 4),
            "tok_s": speed, "ctx_in": prompt_tokens, "ctx_out": generated,
            **({"tokens_estimated": True} if estimated else {}), **extra}


def score(response, checker):
    return bool(checker(response.get("answer") if isinstance(response, dict) else response))
