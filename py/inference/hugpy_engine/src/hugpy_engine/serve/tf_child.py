"""Transformers slot child — env-profiles stage 3.

A minimal OpenAI-compatible chat server for ONE transformers causal LM, launched
by a slot (``slot_agent._build_tf_cmd``) BY FILE PATH from a dependency
profile's venv python:

    <profile>/bin/python .../hugpy_engine/serve/tf_child.py \
        --model-dir DIR --model-key KEY --port P [--ctx N] [--bnb4] ...

Stdlib HTTP + transformers/torch ONLY — no hugpy imports — so it runs under an
``isolated`` profile (which sees only its own packages) as well as a ``worker``
overlay. The model therefore loads with the PROFILE's package versions (e.g. a
newer transformers / compressed-tensors) instead of the worker agent's.

Endpoints mirror what the slot agent and LlamaCppRunner already speak to a
llama-server child:

    GET  /health               200 once the model is loaded
    GET  /v1/models            {"data": [{"id": <model_key>, ...}]}
    GET  /props                {"model_path": <model_dir>, ...}  (identity check)
    POST /v1/chat/completions  chat template; stream (SSE + final usage/timings
                               chunk + [DONE]) or one JSON body with usage/timings

One generation at a time (a lock); /health and /props answer while it runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Launched by path, python puts THIS directory (hugpy_engine/serve) at
# sys.path[0]; its sibling modules (serve.py, policy.py, profiles.py, ...) would
# then shadow same-named top-level imports inside transformers/torch. Drop it.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]


def _log(msg: str) -> None:
    # stderr is the slot's loader-log channel (tail ring + per-load log file).
    print(f"tf_child: {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# engine: the transformers half (the only part that touches torch)             #
# --------------------------------------------------------------------------- #
def _dtype_kwarg(transformers) -> str:
    """``dtype`` from transformers 4.56 on (``torch_dtype`` deprecated there),
    ``torch_dtype`` before it."""
    import re
    parts = [int(x) for x in re.findall(r"\d+", getattr(transformers, "__version__", "0"))[:2]]
    return "dtype" if tuple(parts + [0, 0])[:2] >= (4, 56) else "torch_dtype"


def _max_memory(raw: "str | None") -> "dict | None":
    """The parent's ``max_memory`` map (JSON, string keys) -> accelerate's form
    (GPU indices as ints, ``cpu`` as is)."""
    if not raw:
        return None
    doc = json.loads(raw)
    if not isinstance(doc, dict) or not doc:
        return None
    return {(int(k) if str(k).isdigit() else k): v for k, v in doc.items()}


class HFEngine:
    """Loaded model + tokenizer. ``render``/``encode``/``generate`` are the
    whole surface the HTTP layer uses (a test substitutes a fake engine)."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.last_completion_n = 0

    @property
    def n_ctx_train(self) -> "int | None":
        cfg = getattr(self.model, "config", None)
        val = getattr(cfg, "max_position_embeddings", None)
        return int(val) if isinstance(val, int) and val > 0 else None

    def render(self, messages: list, template_kwargs: "dict | None") -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **(template_kwargs or {}))

    def encode(self, prompt: str) -> list:
        # The rendered template already carries BOS/specials.
        return list(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def _device(self):
        dev = getattr(self.model, "device", None)
        if dev is None:
            dev = next(self.model.parameters()).device
        return dev

    def generate(self, ids: list, max_new: int, params: dict, cancel: threading.Event):
        """Yield decoded text pieces; ``last_completion_n`` holds the generated
        token count once the iterator is exhausted. ``cancel`` stops generation
        at the next token (stop string hit / client gone)."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        class _Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return torch.full((input_ids.shape[0],), cancel.is_set(),
                                  dtype=torch.bool, device=input_ids.device)

        input_ids = torch.tensor([ids], device=self._device())
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)
        gk = {"input_ids": input_ids,
              "attention_mask": torch.ones_like(input_ids),
              "max_new_tokens": int(max_new),
              "streamer": streamer,
              "stopping_criteria": StoppingCriteriaList([_Cancel()])}
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self.tokenizer, "eos_token_id", None)
        if pad is not None:
            gk["pad_token_id"] = pad
        temp = params.get("temperature")
        if temp is not None and float(temp) <= 0:
            gk["do_sample"] = False
        elif temp is not None:
            gk["do_sample"] = True
            gk["temperature"] = float(temp)
            if params.get("top_p") is not None:
                gk["top_p"] = float(params["top_p"])
        rp = params.get("repeat_penalty")
        if rp is not None and float(rp) != 1.0:
            gk["repetition_penalty"] = float(rp)
        bias = params.get("logit_bias")
        if bias:
            from transformers import LogitsProcessor, LogitsProcessorList

            class _Bias(LogitsProcessor):
                # OpenAI/llama.cpp semantics: <= -100 bans the token outright.
                def __call__(self, input_ids, scores):
                    for tid, b in bias.items():
                        if 0 <= tid < scores.shape[-1]:
                            scores[:, tid] = (float("-inf") if b <= -100
                                              else scores[:, tid] + b)
                    return scores

            gk["logits_processor"] = LogitsProcessorList([_Bias()])

        holder: dict = {}

        def _run():
            try:
                out = self.model.generate(**gk)
                holder["n"] = int(out.shape[-1]) - len(ids)
            except BaseException as exc:  # noqa: BLE001 — re-raised on the request thread
                holder["error"] = exc
                holder["tb"] = traceback.format_exc()
                streamer.end()

        self.last_completion_n = 0
        th = threading.Thread(target=_run, name="tf-generate", daemon=True)
        th.start()
        try:
            for piece in streamer:
                if piece:
                    yield piece
        finally:
            if th.is_alive():
                cancel.set()          # consumer left early: stop the decode
            th.join()
        if "error" in holder:
            _log(holder.get("tb") or repr(holder["error"]))
            raise holder["error"]
        self.last_completion_n = holder.get("n", 0)


def load_engine(args) -> HFEngine:
    """Load tokenizer + model from ``args.model_dir`` with THIS interpreter's
    (the profile venv's) transformers."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    trc = bool(args.trust_remote_code)
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=trc,
                                        local_files_only=True)
    cuda = args.device != "cpu" and torch.cuda.is_available()
    if cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif hasattr(torch.cpu, "is_bf16_supported") and torch.cpu.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    kw = {"low_cpu_mem_usage": True, "local_files_only": True,
          "trust_remote_code": trc, _dtype_kwarg(transformers): dtype}
    if cuda:
        kw["device_map"] = "auto"
        mm = _max_memory(args.max_memory)
        if mm:
            kw["max_memory"] = mm
    if args.bnb4:
        # Same recipe as generate/coder.py (nf4 + double-quant); a 4-bit load is
        # not handed the fp16-sized max_memory budget (see coder._load_model).
        kw["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        kw.pop("max_memory", None)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, **kw)
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    _log(f"loaded {args.model_key} from {args.model_dir} (cuda={cuda} dtype={dtype} "
         f"bnb4={bool(args.bnb4)} adapter={args.adapter_dir} "
         f"transformers={transformers.__version__} torch={torch.__version__} "
         f"python={sys.executable})")
    return HFEngine(model, tok)


# --------------------------------------------------------------------------- #
# request handling (pure; no torch)                                           #
# --------------------------------------------------------------------------- #
class RequestError(Exception):
    """An OpenAI-style client error: HTTP ``status`` + error body."""

    def __init__(self, status: int, message: str, etype: str = "invalid_request_error",
                 **extra):
        super().__init__(message)
        self.status = status
        self.body = {"error": {"code": status, "type": etype, "message": message, **extra}}


def _flatten_content(content):
    """OpenAI content (str | [parts]) -> str. Non-text parts are refused: this
    child serves text-generation only."""
    if content is None or isinstance(content, str):
        return content or ""
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                out.append(part)
            else:
                kind = part.get("type") if isinstance(part, dict) else type(part).__name__
                raise RequestError(400, f"content part of type {kind!r} is not "
                                        "supported by the transformers text-generation child")
        return "".join(out)
    return str(content)


def _stops(raw) -> list:
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(s) for s in raw if s]


class ChatServer:
    """One model, one generation at a time."""

    def __init__(self, engine, model_key: str, model_dir: str, ctx: "int | None"):
        self.engine = engine
        self.model_key = model_key
        self.model_dir = model_dir
        # Max total tokens (prompt + completion): the slot's served ctx, else the
        # model's own trained window, else unbounded (max_tokens must say).
        self.ctx = int(ctx) if ctx else getattr(engine, "n_ctx_train", None)
        self.lock = threading.Lock()

    # -- documents ----------------------------------------------------------
    def models_doc(self) -> dict:
        return {"object": "list",
                "data": [{"id": self.model_key, "object": "model",
                          "owned_by": "hugpy", "root": self.model_dir}]}

    def props_doc(self) -> dict:
        return {"model_path": self.model_dir, "model_alias": self.model_key,
                "n_ctx": self.ctx, "engine": "transformers",
                "python": sys.executable, "modalities": {"vision": False}}

    # -- one completion -----------------------------------------------------
    def prepare(self, body: dict) -> dict:
        """Validate + tokenize. Raises RequestError for a client-side problem."""
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise RequestError(400, "'messages' must be a non-empty list")
        messages = [{**m, "content": _flatten_content(m.get("content"))}
                    for m in msgs if isinstance(m, dict)]
        prompt = self.engine.render(messages, body.get("chat_template_kwargs"))
        ids = self.engine.encode(prompt)
        n_prompt = len(ids)
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        try:
            max_tokens = int(max_tokens) if max_tokens is not None else None
        except (TypeError, ValueError):
            raise RequestError(400, f"max_tokens must be an integer, got {max_tokens!r}")
        if max_tokens is not None and max_tokens <= 0:
            max_tokens = None             # llama.cpp's -1 = "until the context is full"
        if self.ctx:
            room = self.ctx - n_prompt
            if room <= 0:
                raise RequestError(
                    400, f"the request exceeds the available context size: prompt "
                         f"{n_prompt} tokens, context {self.ctx} tokens",
                    etype="exceed_context_size_error",
                    n_prompt_tokens=n_prompt, n_ctx=self.ctx)
            max_new = min(int(max_tokens), room) if max_tokens else room
        elif max_tokens:
            max_new = int(max_tokens)
        else:
            raise RequestError(400, "no max_tokens given and no context size known "
                                    "for this model (no --ctx, no max_position_embeddings)")
        params = {k: body.get(k) for k in ("temperature", "top_p", "repeat_penalty")}
        raw_bias = body.get("logit_bias")
        if raw_bias:
            try:
                params["logit_bias"] = {int(k): float(v) for k, v in dict(raw_bias).items()}
            except (TypeError, ValueError):
                raise RequestError(400, "logit_bias must map token ids to numbers; "
                                        f"got {raw_bias!r}")
        return {"ids": ids, "n_prompt": n_prompt, "max_new": max(1, max_new),
                "stops": _stops(body.get("stop")), "params": params}

    def pieces(self, job: dict, cancel: threading.Event, clock: dict):
        """Yield text with stop strings applied; fill ``clock`` with the
        measurements (first-piece time, end time, completion tokens, finish)."""
        stops = job["stops"]
        hold = max((len(s) for s in stops), default=1) - 1
        pending = ""
        stopped = False
        for piece in self.engine.generate(job["ids"], job["max_new"], job["params"], cancel):
            if "t_first" not in clock:
                clock["t_first"] = time.time()
            if stopped:
                continue      # drain: decode ends at the next token, count lands
            if not stops:
                yield piece
                continue
            pending += piece
            hits = [i for i in (pending.find(s) for s in stops) if i >= 0]
            if hits:
                cut = pending[:min(hits)]
                if cut:
                    yield cut
                pending = ""
                stopped = True
                cancel.set()
                continue
            if len(pending) > hold:
                yield pending[:len(pending) - hold]
                pending = pending[len(pending) - hold:]
        if pending:
            yield pending
        clock["t_end"] = time.time()
        clock["n"] = int(getattr(self.engine, "last_completion_n", 0) or 0)
        clock["finish"] = ("stop" if stopped or clock["n"] < job["max_new"] else "length")

    @staticmethod
    def usage_timings(job: dict, clock: dict):
        t0, t_end = clock["t0"], clock.get("t_end", time.time())
        t_first = clock.get("t_first", t_end)
        n = clock.get("n", 0)
        # prompt_ms: request start -> first decoded text (prefill + the first
        # token's decode); predicted_ms: first text -> end of generation.
        prompt_ms = (t_first - t0) * 1000.0
        predicted_ms = (t_end - t_first) * 1000.0
        usage = {"prompt_tokens": job["n_prompt"], "completion_tokens": n,
                 "total_tokens": job["n_prompt"] + n}
        timings = {"prompt_n": job["n_prompt"], "prompt_ms": prompt_ms,
                   "predicted_n": n, "predicted_ms": predicted_ms,
                   "measurement_source": "transformers_child_wall"}
        if prompt_ms > 0:
            timings["prompt_per_second"] = job["n_prompt"] / (prompt_ms / 1000.0)
        if predicted_ms > 0 and n:
            timings["predicted_per_second"] = n / (predicted_ms / 1000.0)
        return usage, timings

    def complete(self, job: dict) -> dict:
        """Non-streaming completion body."""
        cancel = threading.Event()
        clock = {"t0": time.time()}
        text = "".join(self.pieces(job, cancel, clock))
        usage, timings = self.usage_timings(job, clock)
        return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                "created": int(time.time()), "model": self.model_key,
                "choices": [{"index": 0, "finish_reason": clock["finish"],
                             "message": {"role": "assistant", "content": text}}],
                "usage": usage, "timings": timings}

    def stream_events(self, job: dict, cancel: threading.Event):
        """SSE payload dicts in order (the caller frames them as ``data:``)."""
        cid = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        def chunk(delta, finish=None, **extra):
            return {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": self.model_key,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    **extra}

        clock = {"t0": time.time()}
        first = True
        for text in self.pieces(job, cancel, clock):
            delta = {"content": text}
            if first:
                delta = {"role": "assistant", "content": text}
                first = False
            yield chunk(delta)
        usage, timings = self.usage_timings(job, clock)
        yield chunk({}, clock["finish"], usage=usage, timings=timings)


def make_handler(server: ChatServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt, *args):   # keep the loader log for errors only
            pass

        def _json(self, status: int, doc: dict) -> None:
            data = json.dumps(doc).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/health":
                return self._json(200, {"status": "ok"})
            if path == "/v1/models":
                return self._json(200, server.models_doc())
            if path == "/props":
                return self._json(200, server.props_doc())
            return self._json(404, {"error": {"code": 404, "type": "not_found_error",
                                              "message": f"no route {path}"}})

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path != "/v1/chat/completions":
                return self._json(404, {"error": {"code": 404, "type": "not_found_error",
                                                  "message": f"no route {path}"}})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise RequestError(400, "request body must be a JSON object")
            except (ValueError, RequestError) as exc:
                err = exc if isinstance(exc, RequestError) else RequestError(400, f"bad JSON: {exc}")
                return self._json(err.status, err.body)
            with server.lock:
                try:
                    job = server.prepare(body)
                    if not body.get("stream"):
                        return self._json(200, server.complete(job))
                    return self._stream(job)
                except RequestError as exc:
                    return self._json(exc.status, exc.body)
                except Exception as exc:  # noqa: BLE001 — the real error, to the caller and the log
                    _log(traceback.format_exc())
                    return self._json(500, {"error": {"code": 500, "type": "server_error",
                                                      "message": f"{type(exc).__name__}: {exc}"}})

        def _stream(self, job: dict) -> None:
            cancel = threading.Event()
            events = server.stream_events(job, cancel)
            # Headers wait for the first event, so a failure before any token
            # still answers as a plain 500 with the real error.
            first = next(events)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                self.wfile.write(b"data: " + json.dumps(first).encode() + b"\n\n")
                self.wfile.flush()
                try:
                    for ev in events:
                        self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    raise
                except Exception as exc:  # noqa: BLE001 — mid-stream: an error event
                    _log(traceback.format_exc())
                    err = {"error": {"code": 500, "type": "server_error",
                                     "message": f"{type(exc).__name__}: {exc}"}}
                    self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                cancel.set()               # client gone: stop decoding
                events.close()
            self.close_connection = True

    return Handler


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--ctx", type=int, default=None,
                    help="max total tokens (prompt + completion)")
    ap.add_argument("--bnb4", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--device", choices=("auto", "cpu"), default="auto")
    ap.add_argument("--max-memory", default=None,
                    help="JSON accelerate max_memory map (device_map=auto)")
    ap.add_argument("--adapter-dir", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    engine = load_engine(args)
    srv = ChatServer(engine, args.model_key, args.model_dir, args.ctx)
    _log(f"{args.model_key} ready in {time.time() - t0:.1f}s; ctx={srv.ctx}; "
         f"listening on {args.host}:{args.port}")
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(srv))
    httpd.daemon_threads = True
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
