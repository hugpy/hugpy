"""Worker-local Ollama inference and memory control for scanned manifests."""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Iterator


def model_name(model_key: str | None) -> str | None:
    if not model_key:
        return None
    try:
        from hugpy_engine.config.main import get_model_config
        return (getattr(get_model_config(model_key), "extra", None) or {}).get("ollama_model")
    except (KeyError, TypeError, ValueError):
        return None


def _url(route: str) -> str:
    return os.environ.get("HUGPY_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/") + route


def _request(route: str, body: dict, *, stream: bool = False):
    request = urllib.request.Request(
        _url(route), data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(request, timeout=3600 if stream else 600)


def _messages(payload: dict) -> list[dict]:
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        return [{"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in messages if isinstance(m, dict)]
    return [{"role": "user", "content": str(payload.get("prompt") or "")}]


def _body(payload: dict, *, stream: bool) -> dict:
    name = model_name(payload.get("model_key"))
    if not name:
        raise ValueError("model is not a scanned Ollama model")
    options = {}
    for source, target in (("temperature", "temperature"), ("top_p", "top_p"),
                           ("max_tokens", "num_predict"), ("stop", "stop")):
        if payload.get(source) is not None:
            options[target] = payload[source]
    body = {"model": name, "messages": _messages(payload), "stream": stream,
            "keep_alive": os.environ.get("HUGPY_OLLAMA_KEEP_ALIVE", "30m")}
    if options:
        body["options"] = options
    return body


def _finish_reason(reason: str | None) -> str:
    return "max_tokens" if reason == "length" else "stop"


def run_once(payload: dict) -> dict:
    with _request("/api/chat", _body(payload, stream=False)) as response:
        data = json.load(response)
    if data.get("error"):
        raise RuntimeError(data["error"])
    text = str((data.get("message") or {}).get("content") or "")
    prompt_tokens = int(data.get("prompt_eval_count") or 0)
    output_tokens = int(data.get("eval_count") or 0)
    return {"ok": True, "model_key": payload["model_key"], "text": text,
            "finish_reason": _finish_reason(data.get("done_reason")),
            "output_chunks": 1 if text else 0,
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": output_tokens,
                      "total_tokens": prompt_tokens + output_tokens}}


def stream(payload: dict) -> Iterator[dict]:
    with _request("/api/chat", _body(payload, stream=True), stream=True) as response:
        for raw in response:
            if not raw.strip():
                continue
            event = json.loads(raw)
            if event.get("error"):
                raise RuntimeError(event["error"])
            token = (event.get("message") or {}).get("content") or ""
            if token:
                yield {"type": "token", "text": token}
            if event.get("done"):
                prompt_tokens = int(event.get("prompt_eval_count") or 0)
                output_tokens = int(event.get("eval_count") or 0)
                yield {"type": "done",
                       "finish_reason": _finish_reason(event.get("done_reason")),
                       "usage": {"prompt_tokens": prompt_tokens,
                                 "completion_tokens": output_tokens,
                                 "total_tokens": prompt_tokens + output_tokens}}


def running() -> dict[str, dict]:
    try:
        with urllib.request.urlopen(_url("/api/ps"), timeout=2) as response:
            rows = json.load(response).get("models") or []
        return {r["name"]: r for r in rows if isinstance(r, dict) and r.get("name")}
    except (OSError, ValueError, KeyError):
        return {}


def unload(name: str) -> bool:
    # Ollama's documented keep_alive=0 unloads a resident without deleting its
    # manifest or blobs. The worker's ordinary eviction gate runs before this.
    with _request("/api/chat", {"model": name, "messages": [],
                                "stream": False, "keep_alive": 0}) as response:
        data = json.load(response)
    if data.get("error"):
        raise RuntimeError(data["error"])
    return name not in running()
