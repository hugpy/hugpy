"""review/judge.py — the hugpy-agent read on a candidate.

The screen measures and the smoke test proves, but neither can answer "is this
actually a step up over what we already run, and for what?" That is a
judgement, so it goes to a hugpy agent: a model the fleet is already serving,
called over the local OpenAI-compatible endpoint (/v1/chat/completions).

Everything here is optional and best-effort. If no agent is reachable, or it
returns something unparseable, the review still completes — it just carries no
verdict. A review pipeline that hard-fails because a judge was busy would be
worse than useless on a timer.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

# Where central actually answers. On `ae` the Flask app runs on the hugpy VM,
# so loopback is NOT it — probing a list beats hardcoding one and silently
# never producing a verdict. REVIEW_AGENT_URL pins it explicitly.
AGENT_URLS = [u for u in (
    os.environ.get("REVIEW_AGENT_URL"),
    "http://192.168.1.250:7002/api/v1/chat/completions",   # hugpy VM, LAN
    "http://127.0.0.1:7002/api/v1/chat/completions",       # same-box central
) if u]
AGENT_URL = AGENT_URLS[0]
AGENT_MODEL = os.environ.get("REVIEW_AGENT_MODEL") or ""
AGENT_TIMEOUT = int(os.environ.get("REVIEW_AGENT_TIMEOUT") or 180)
AGENT_KEY = os.environ.get("REVIEW_AGENT_KEY") or os.environ.get("HUGPY_API_KEY")

SYSTEM = (
    "You review candidate language models for a small self-hosted GPU fleet. "
    "You are given measured facts about one model: its size and quantization, "
    "the VRAM it needs, its context length, how fast it generated, and the raw "
    "output it produced for four fixed probe prompts. Judge only from those "
    "facts. Be blunt about weaknesses. Never invent benchmark numbers."
)

INSTRUCTION = """Return ONLY a JSON object, no prose around it, with keys:
  "verdict": one of "adopt", "trial", "reject"
  "capability": integer 1-10, how capable it looks from the probe outputs
  "fit": integer 1-10, how comfortably it fits the stated hardware budget
  "strengths": array of short strings
  "weaknesses": array of short strings
  "summary": one or two sentences
  "compared_to_incumbents": one sentence, or null if there are none
"""


def _post(payload: dict, url: str | None = None) -> dict | None:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url or AGENT_URL, data=body, method="POST",
        headers={"Content-Type": "application/json"})
    if AGENT_KEY:
        req.add_header("Authorization", f"Bearer {AGENT_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=AGENT_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def available_agent() -> tuple[str, str] | None:
    """(model_id, chat_url) for an agent that will actually answer, or None.
    Probes the candidate URLs in order; REVIEW_AGENT_MODEL pins the model."""
    for chat_url in AGENT_URLS:
        models_url = chat_url.rsplit("/chat/completions", 1)[0] + "/models"
        try:
            req = urllib.request.Request(models_url)
            if AGENT_KEY:
                req.add_header("Authorization", f"Bearer {AGENT_KEY}")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            continue
        if AGENT_MODEL:
            return AGENT_MODEL, chat_url
        rows = data.get("data") if isinstance(data, dict) else data
        for row in (rows or []):
            mid = row.get("id") if isinstance(row, dict) else row
            if mid:
                return mid, chat_url
    return None


def _extract_json(text: str) -> dict | None:
    """Pull the JSON object out of a completion. Small local models routinely
    wrap it in prose or a ```json fence however firmly you ask them not to.

    The fence + brace-walk implementation that used to live here is now the
    package seam ``utils/json_scavenge.extract_json_object`` — a second copy had
    grown in ``comms/todo_keeper`` and a third was about to in the studio spread
    path. Behaviour is unchanged; this stays as the module-local name the rest of
    the file (and its tests) call."""
    from hugpy_engine.utils.json_scavenge import extract_json_object
    return extract_json_object(text)


def judge(screen_result, smoke_result, crit) -> dict | None:
    """Ask the agent for a verdict. Returns None when no agent is reachable."""
    found = available_agent()
    if not found:
        return None
    model, chat_url = found

    s = screen_result.to_dict() if hasattr(screen_result, "to_dict") else dict(screen_result or {})
    k = smoke_result.to_dict() if hasattr(smoke_result, "to_dict") else dict(smoke_result or {})

    facts = {
        "hub_id": s.get("hub_id"),
        "architecture": s.get("architecture"),
        "params": s.get("params"),
        "quant": s.get("best_quant"),
        "context_length": s.get("context_length"),
        "estimated_vram_gib": round((s.get("est_vram_bytes") or 0) / 1024**3, 2) or None,
        "vram_budget_gib": round(crit.usable_vram_bytes / 1024**3, 2),
        "target_context": crit.target_context,
        "downloads": s.get("downloads"),
        "publisher_trust_tier": s.get("trust_tier"),
        "base_model": s.get("base_model"),
        "incumbents": crit.incumbents,
        "loaded_ok": k.get("ok"),
        "load_seconds": k.get("load_seconds"),
        "generation_tokens_per_sec": k.get("gen_tokens_per_sec"),
        "measured_vram_gib": round((k.get("vram_used_bytes") or 0) / 1024**3, 2) or None,
        "probe_outputs": [{"prompt": p.get("prompt"), "output": p.get("output")}
                          for p in (k.get("probes") or [])],
    }

    # NO-THINK (utils/no_think.py). Strict JSON verdict contract at temperature 0
    # with a 700-token ceiling; _extract_json's fence/brace scavenger exists
    # because models wrap output in prose, and a <think> block is that failure at
    # its worst — it can consume the entire budget before the JSON starts.
    from hugpy_engine.utils.no_think import apply_no_think, strip_think
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 700,
        "messages": apply_no_think([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                "Measured facts:\n" + json.dumps(facts, indent=2)
                + "\n\n" + INSTRUCTION},
        ]),
    }
    resp = _post(payload, url=chat_url)
    if not resp:
        return None
    try:
        text = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    # Strip defensively before the scavenger sees it; `raw` below keeps the full
    # reply so a parse failure is still diagnosable.
    text, _reasoning = strip_think(text or "")
    parsed = _extract_json(text)
    if parsed is None:
        # keep the raw read rather than dropping the agent's work entirely
        return {"agent": model, "raw": text[:4000], "parse_failed": True}
    parsed["agent"] = model
    return parsed
