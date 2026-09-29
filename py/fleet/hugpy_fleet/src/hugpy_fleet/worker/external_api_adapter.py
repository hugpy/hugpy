"""Call a registered worker-local OpenAI-compatible resident."""
from __future__ import annotations

import json


def resident(model_key: str | None) -> dict | None:
    if not model_key:
        return None
    try:
        from hugpy_fleet.worker import external_residents
        rec = external_residents.get(model_key)
        return rec if rec and rec.get("api_url") else None
    except Exception:  # noqa: BLE001 — optional adapter must not break dispatch
        return None


def _payload(rec: dict, payload: dict, *, stream: bool) -> dict:
    result = {"model": rec.get("served_model") or rec["model_key"],
              "messages": payload.get("messages") or [
                  {"role": "user", "content": str(payload.get("prompt") or "")}
              ], "max_tokens": max(1, min(4096, int(payload.get("max_tokens") or 512))),
              "stream": stream}
    for key in ("temperature", "top_p", "stop"):
        if payload.get(key) is not None:
            result[key] = payload[key]
    return result


def run_once(payload: dict) -> dict:
    rec = resident(payload.get("model_key"))
    if not rec:
        raise ValueError("model is not a registered external API resident")
    import httpx
    response = httpx.post(rec["api_url"].rstrip("/") + "/v1/chat/completions",
                          json=_payload(rec, payload, stream=False),
                          timeout=httpx.Timeout(600.0, connect=3.0))
    response.raise_for_status()
    data = response.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    usage = data.get("usage") or {}
    return {"ok": True, "model_key": payload["model_key"], "text": text,
            "finish_reason": choice.get("finish_reason") or "stop",
            "output_chunks": 1 if text else 0,
            "usage": {"prompt_tokens": int(usage.get("prompt_tokens") or 0),
                      "completion_tokens": int(usage.get("completion_tokens") or 0),
                      "total_tokens": int(usage.get("total_tokens") or 0)}}


def stream(payload: dict):
    rec = resident(payload.get("model_key"))
    if not rec:
        raise ValueError("model is not a registered external API resident")
    import httpx
    with httpx.stream("POST", rec["api_url"].rstrip("/") + "/v1/chat/completions",
                      json=_payload(rec, payload, stream=True),
                      timeout=httpx.Timeout(3600.0, connect=3.0)) as response:
        response.raise_for_status()
        usage = {}
        finish = "stop"
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                break
            event = json.loads(raw)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                token = delta.get("content")
                if token:
                    yield {"type": "token", "text": token}
        yield {"type": "done", "finish_reason": finish,
               "usage": {"prompt_tokens": int(usage.get("prompt_tokens") or 0),
                         "completion_tokens": int(usage.get("completion_tokens") or 0),
                         "total_tokens": int(usage.get("total_tokens") or 0)}}
