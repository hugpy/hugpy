"""Slim GGUF-only worker agent for the hugpy LLM pool.

Run this on any box that can build llama-cpp-python — including ones the full
``worker_agent`` can't install on (Termux/Android, small ARM boards) — to
donate its CPU to the central console. It speaks the exact same protocol as
the full agent:

    1. Registers with central (``/api/llm/workers/register``) and keeps a
       persistent worker id so restarts reuse the row.
    2. Serves chat over HTTP for the GGUF models central assigns to it:
           GET  /health
           POST /infer          {model_key, messages|prompt, ...} -> ChatResult dict
           POST /infer/stream   -> SSE request/status/token/done/error events
           POST /infer/cancel/<request_id>
           POST /models/unload
           POST|GET /probe/<model_key>
    3. Heartbeats every ``--heartbeat`` seconds; adopts assignment changes from
       the response and pre-downloads newly assigned GGUFs from central's model
       file routes (``/api/llm/models/<key>/manifest`` + ``/file``).

Differences from the full agent, all deliberate:
  * chat only — any other task is refused with HTTP 501 *before* the SSE
    stream starts, so central's DelegatingRunner falls back to local cleanly.
  * llama.cpp only — a model whose manifest says any other framework is
    refused the same way.
  * self-update is conditional: when this agent runs as an INSTALLED
    ``abstract_hugpy_dev`` package it reports its real version and converges to
    central's ``required_pkg_version`` by ``pip install --no-deps`` FROM THE
    CENTRAL it's pointed at (``<central>/api/llm/pip/simple``), then re-execs.
    When run STANDALONE (a copied agent.py, no package) it reports
    ``gguf-worker/slim`` and never self-updates — pip can't swap a non-package.
    Disable entirely with ``--no-self-update`` / ``WORKER_SELF_UPDATE=0``.
  * one model in memory at a time — small-RAM devices are the target, so
    loading model B evicts model A.

Usage
-----
    python -m gguf_worker \
        --central http://192.168.1.250:7002 \
        --name note20-llm \
        --models Qwen2.5-0.5B-Instruct-GGUF

Every flag has an env fallback (WORKER_CENTRAL_URL, WORKER_NAME, WORKER_HOST,
WORKER_PORT, WORKER_URL, WORKER_MODELS, WORKER_HEARTBEAT, WORKER_ID_FILE,
GGUF_WORKER_MODELS_DIR, GGUF_WORKER_N_CTX, GGUF_WORKER_N_THREADS,
GGUF_WORKER_N_GPU_LAYERS).
"""
from __future__ import annotations

import os
import sys
import json
import time
import uuid
import base64
import socket
import logging
import argparse
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request

from flask import Flask, request, jsonify, Response, stream_with_context

logger = logging.getLogger("gguf_worker")

PKG_NAME = os.environ.get("WORKER_PKG_NAME", "hugpy-fleet")


def _detect_pkg_version() -> "tuple[str, str | None]":
    """``(version_to_report, installed_version_or_None)``.

    When this agent runs as part of an INSTALLED ``abstract_hugpy_dev`` package
    (``python -m abstract_hugpy_dev.gguf_worker``), report that real version so
    central's handshake can detect drift and we can pip-update FROM CENTRAL.
    When run STANDALONE (a copied agent.py with no package installed), there is
    nothing pip can swap, so keep the slim marker and never self-update.
    """
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            v = version(PKG_NAME)
            return v, v
        except PackageNotFoundError:
            return "gguf-worker/slim", None
    except Exception:
        return "gguf-worker/slim", None


PKG_VERSION, _INSTALLED_VERSION = _detect_pkg_version()

# Self-update throttle: don't re-attempt the same failed target for a while.
_UPDATE_BACKOFF = 600.0
_update_state = {"target": None, "at": 0.0}


def _self_update_if_needed(required: "str | None", args) -> None:
    """Pip-install central's required version (from the central we're pointed at,
    by default) and re-exec, when we're an installed package and behind.

    No-op when: self-update is disabled, central isn't managing versions, we're a
    standalone copy (``_INSTALLED_VERSION is None`` — nothing pip can swap), or we
    already match. ``--no-deps`` hot-swaps only the package on an already-set-up
    env. Failures are non-fatal: we log and keep serving the current code.
    """
    if getattr(args, "no_self_update", False):
        return
    if not required or _INSTALLED_VERSION is None or required == _INSTALLED_VERSION:
        if required and _INSTALLED_VERSION is None and required != "gguf-worker/slim":
            logger.info("central wants %s==%s but this is a standalone copy; "
                        "re-run the installer or `pip install` to update", PKG_NAME, required)
        return
    now = time.time()
    if _update_state["target"] == required and (now - _update_state["at"]) < _UPDATE_BACKOFF:
        return
    _update_state.update(target=required, at=now)
    index = args.pkg_index or (args.central.rstrip("/") + "/api/llm/pip/simple")
    logger.info("self-update: %s %s -> %s (from %s)", PKG_NAME, _INSTALLED_VERSION, required, index)
    cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-deps",
           "--index-url", index, f"{PKG_NAME}=={required}"]
    try:
        rc = subprocess.call(cmd)
    except Exception as exc:  # noqa: BLE001
        logger.warning("self-update pip invocation failed: %s", exc)
        return
    if rc == 0:
        logger.info("self-update installed %s==%s; restarting agent", PKG_NAME, required)
        mod = __package__ or ""
        if mod:
            os.execv(sys.executable, [sys.executable, "-m", mod, *sys.argv[1:]])
        else:
            os.execv(sys.executable, [sys.executable, *sys.argv])
    else:
        logger.warning("self-update failed (pip rc=%s); staying on %s", rc, _INSTALLED_VERSION)

# request_id -> threading.Event, so POST /infer/cancel can stop an in-flight
# stream between chunks. Populated by the stream generator, tripped by the
# cancel route.
_CANCELS: dict = {}


# ---------------------------------------------------------------------------
# Host facts (no hugpy._platform — stdlib probes only)
# ---------------------------------------------------------------------------
def _free_ram_bytes() -> int | None:
    """MemAvailable from /proc/meminfo — feeds central's allocator CPU tier."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _import_llama_cpp():
    """Import llama_cpp, shimming Termux: Python 3.13 on Android reports
    ``sys.platform == "android"``, which llama_cpp's shared-library loader
    refuses even though its .so files are ordinary ELF that load fine."""
    import sys
    if sys.platform == "android":
        sys.platform = "linux"
        try:
            import llama_cpp
        finally:
            sys.platform = "android"
        return llama_cpp
    import llama_cpp
    return llama_cpp


def _llama_cpp_supports_vision() -> bool:
    """Whether this worker's llama.cpp can actually decode images.

    True when a multimodal chat handler is importable — the same capability the
    in-process vision runner needs to load an mmproj projector. The central reads
    this (engine.supports_vision) and only routes image turns to workers that
    report it, so an older text-only build is never handed an image to guess at.
    """
    try:
        from llama_cpp import llama_chat_format as _cf
        for _name in ("Qwen25VLChatHandler", "Llava16ChatHandler",
                      "Llava15ChatHandler", "MiniCPMv26ChatHandler",
                      "MoondreamChatHandler"):
            if hasattr(_cf, _name):
                return True
    except Exception:
        pass
    try:
        import llama_cpp.mtmd_cpp  # noqa: F401
        return True
    except Exception:
        return False


def llama_cpp_status() -> dict:
    try:
        llama_cpp = _import_llama_cpp()
        supports = None
        try:
            supports = bool(llama_cpp.llama_supports_gpu_offload())
        except Exception:
            pass
        out = {
            "installed": True,
            "version": getattr(llama_cpp, "__version__", None),
            "supports_gpu_offload": supports,
            "supports_vision": _llama_cpp_supports_vision(),
        }
    except Exception as exc:  # noqa: BLE001
        out = {"installed": False, "error": f"{type(exc).__name__}: {exc}"}
    # k94: native llama-server state beside the python binding, in the dict
    # central snapshots as "engine" — an installed-but-unusable binary was
    # invisible remotely. Guarded absolute import: this file also runs as a
    # STANDALONE copy without the package, where the field simply stays absent.
    try:
        from hugpy_engine.native.resolve import native_engine_status
        out["native_engine"] = native_engine_status()
    except Exception:  # noqa: BLE001
        pass
    return out


from hugpy_platform.environment_status import environment_status

def env_status() -> dict:
    return environment_status(("llama-cpp-python", "transformers", "torch"))


# ---------------------------------------------------------------------------
# Central node client (registration + heartbeat + model files)
# ---------------------------------------------------------------------------
class CentralClient:
    def __init__(self, central_url: str):
        self.central = central_url.rstrip("/")
        self.base = self.central + "/api/llm/workers"

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def register(self, payload: dict) -> dict:
        return self._post("/register", payload)

    def heartbeat(self, worker_id: str, payload: dict) -> dict:
        return self._post(f"/{worker_id}/heartbeat", payload)

    def model_manifest(self, model_key: str) -> dict:
        url = f"{self.central}/api/llm/models/{urllib.parse.quote(model_key, safe='')}/manifest"
        with urllib.request.urlopen(url, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def download_model_file(self, model_key: str, rel_path: str, dest: str,
                            progress=None) -> None:
        url = (f"{self.central}/api/llm/models/"
               f"{urllib.parse.quote(model_key, safe='')}/file?"
               f"{urllib.parse.urlencode({'path': rel_path})}")
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        tmp = dest + ".part"
        done = 0
        with urllib.request.urlopen(url, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total, rel_path)
        os.replace(tmp, dest)


# ---------------------------------------------------------------------------
# Model store — local GGUF lookup + provisioning from central
# ---------------------------------------------------------------------------
class ModelStore:
    """Maps model_key -> a local .gguf path, downloading from central if needed.

    Layout under models_dir: either a flat ``<key>.gguf`` (hand-copied) or a
    ``<key>/`` directory mirroring central's model dir (what provisioning
    writes). Key matching tolerates hub_id vs short-name spellings the same
    way central's manifest route does (trailing path segment, case-insensitive).
    """

    def __init__(self, models_dir: str, client: CentralClient):
        self.dir = os.path.expanduser(models_dir)
        self.client = client
        os.makedirs(self.dir, exist_ok=True)

    def _candidates(self, model_key: str) -> list[str]:
        short = model_key.split("/")[-1]
        names = {model_key, short, model_key.lower(), short.lower()}
        return sorted(names, key=len, reverse=True)

    def local_path(self, model_key: str) -> str | None:
        for name in self._candidates(model_key):
            flat = os.path.join(self.dir, name + ".gguf")
            if os.path.isfile(flat):
                return flat
            sub = os.path.join(self.dir, name)
            if os.path.isdir(sub):
                ggufs = sorted(
                    os.path.join(root, f)
                    for root, _d, files in os.walk(sub)
                    for f in files if f.endswith(".gguf")
                )
                if ggufs:
                    # Multi-shard GGUFs load from the first shard.
                    return ggufs[0]
        return None

    def provision(self, model_key: str, progress=None) -> str:
        """Download the model's GGUF file(s) from central. Returns the load path."""
        existing = self.local_path(model_key)
        if existing:
            return existing
        manifest = self.client.model_manifest(model_key)
        framework = manifest.get("framework")
        if framework and framework != "gguf":
            raise RuntimeError(
                f"{model_key} is framework={framework!r}; this worker serves GGUF only")
        ggufs = [f for f in manifest.get("files", [])
                 if f.get("path", "").endswith(".gguf")]
        # A canonical shard filename narrows multi-variant dirs to one quant.
        filename = manifest.get("filename")
        if filename:
            named = [f for f in ggufs if os.path.basename(f["path"]) == filename]
            ggufs = named or ggufs
        if not ggufs:
            raise RuntimeError(f"{model_key}: central has no .gguf files for it")
        sub = os.path.join(self.dir, model_key.split("/")[-1])
        for f in ggufs:
            dest = os.path.join(sub, f["path"])
            if os.path.isfile(dest) and os.path.getsize(dest) == f.get("size"):
                continue
            logger.info("downloading %s (%s bytes) for %s", f["path"], f.get("size"), model_key)
            self.client.download_model_file(model_key, f["path"], dest, progress=progress)
        path = self.local_path(model_key)
        if not path:
            raise RuntimeError(f"{model_key}: downloaded but no .gguf found locally")
        return path


# ---------------------------------------------------------------------------
# Inference — single-slot llama.cpp runner
# ---------------------------------------------------------------------------
class LlamaSlot:
    """Holds at most ONE loaded Llama at a time (small-RAM devices are the
    target; loading model B evicts model A). All generation is serialized
    through ``gen_lock`` because llama.cpp contexts are not thread-safe."""

    def __init__(self, store: ModelStore, n_ctx: int, n_threads: int | None,
                 n_gpu_layers: int):
        self.store = store
        self.n_ctx = n_ctx
        self.n_threads = n_threads or max(1, (os.cpu_count() or 2) - 1)
        self.n_gpu_layers = n_gpu_layers
        self.gen_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._key: str | None = None
        self._llm = None

    @property
    def loaded_key(self) -> str | None:
        return self._key

    def get(self, model_key: str, progress=None):
        with self._load_lock:
            if self._key == model_key and self._llm is not None:
                return self._llm
            path = self.store.local_path(model_key) or self.store.provision(
                model_key, progress=progress)
            llama_cpp = _import_llama_cpp()
            if self._llm is not None:
                logger.info("evicting %s to load %s", self._key, model_key)
                self._llm = None
            logger.info("loading %s (%s) n_ctx=%s n_threads=%s",
                        model_key, path, self.n_ctx, self.n_threads)
            self._llm = llama_cpp.Llama(
                model_path=path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
            self._key = model_key
            return self._llm

    def unload(self, model_key: str | None = None) -> bool:
        with self._load_lock:
            if self._llm is None or (model_key and model_key != self._key):
                return False
            self._key, self._llm = None, None
            return True


# ---------------------------------------------------------------------------
# Request shaping — the central relay sends a built ChatRequest dump
# ---------------------------------------------------------------------------
def _coerce_messages(payload: dict) -> list[dict]:
    """messages out of a relayed payload; tolerate a bare prompt + system."""
    messages = payload.get("messages")
    if not messages:
        prompt = payload.get("prompt")
        if not prompt:
            raise ValueError("payload has neither messages nor prompt")
        messages = []
        if payload.get("system"):
            messages.append({"role": "system", "content": str(payload["system"])})
        messages.append({"role": "user", "content": str(prompt)})
    # Central inlines an uploaded file as base64 (the worker can't see central's
    # uploads dir). Text files ride into the last user message, same shape the
    # central-side coerce produces; binary uploads are refused (chat-only worker).
    b64 = payload.get("file_b64")
    if b64:
        name = payload.get("file_name") or "upload"
        try:
            content = base64.b64decode(b64).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            raise ValueError(f"non-text upload {name!r}; this worker serves text chat only")
        for msg in reversed(messages):
            if msg.get("role") == "user":
                msg["content"] = f"{msg['content']}\n------{name}------\n{content}"
                break
    return [{"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages]


def _gen_params(payload: dict, n_ctx: int) -> dict:
    max_tokens = payload.get("max_new_tokens") or payload.get("max_tokens")
    max_tokens = max(16, min(int(max_tokens) if max_tokens else n_ctx // 2, n_ctx - 16))
    return {
        "max_tokens": max_tokens,
        "temperature": float(payload.get("temperature") or 0.7),
        "top_p": float(payload.get("top_p") or 0.95),
    }


def _sse(payload: dict) -> bytes:
    # werkzeug asserts the app yields bytes, not str — encode here.
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _check_serviceable(payload: dict) -> str | None:
    """Refuse non-chat work with a reason, BEFORE any SSE bytes go out.

    Failing with a non-200 (rather than an SSE error event) matters: central's
    DelegatingRunner falls back to running the request locally only when the
    worker stream produced nothing.
    """
    # Resolution names the task by the model's primary task, so plain chat on a
    # GGUF model arrives as task="text-generation". Accept anything whose payload
    # is messages/prompt-shaped; refuse the rest (embed, transcribe, vision, …).
    task = payload.get("task")
    if task not in (None, "chat", "text-generation"):
        return f"task {task!r} unsupported (chat-only worker)"
    if not (payload.get("messages") or payload.get("prompt")):
        return "payload has neither messages nor prompt (chat-only worker)"
    if not payload.get("model_key"):
        return "missing model_key"
    return None


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
def build_app(state: "WorkerState", slot: LlamaSlot) -> Flask:
    app = Flask("gguf_worker")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "ok": True,
            "worker_id": state.worker_id,
            "name": state.name,
            "slim": True,
            "gpus": [],
            "cuda": {"available": False},
            "llama_cpp": llama_cpp_status(),
            "assigned_models": state.assigned_models,
            "provisioning": sorted(state._provisioning),
            "loaded_models": [slot.loaded_key] if slot.loaded_key else [],
            "free_ram": _free_ram_bytes(),
            "spill": {},
        })

    @app.route("/infer", methods=["POST"])
    def infer():
        payload = request.get_json(silent=True) or {}
        payload.pop("spill", None)
        reason = _check_serviceable(payload)
        request_id = str(payload.get("request_id") or uuid.uuid4().hex)
        model_key = payload.get("model_key")
        envelope = {"request_id": request_id, "model_key": model_key}
        if reason:
            return jsonify({**envelope, "ok": False, "error": reason,
                            "text": "", "finish_reason": "stop"}), 501
        try:
            llm = slot.get(model_key)
            messages = _coerce_messages(payload)
            with slot.gen_lock:
                out = llm.create_chat_completion(
                    messages=messages, stream=False, **_gen_params(payload, slot.n_ctx))
            choice = (out.get("choices") or [{}])[0]
            return jsonify({
                **envelope, "ok": True, "error": None,
                "text": (choice.get("message") or {}).get("content", ""),
                "finish_reason": choice.get("finish_reason") or "stop",
                "usage": out.get("usage"),
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("infer failed: %s: %s", type(exc).__name__, exc)
            return jsonify({**envelope, "ok": False, "text": "",
                            "finish_reason": "stop",
                            "error": f"{type(exc).__name__}: {exc}"}), 500

    @app.route("/infer/stream", methods=["POST"])
    def infer_stream():
        payload = request.get_json(silent=True) or {}
        payload.pop("spill", None)
        reason = _check_serviceable(payload)
        if reason:
            # Non-200 before any SSE → central falls back to local.
            return jsonify({"ok": False, "error": reason}), 501
        req_id = str(payload.pop("request_id", "") or uuid.uuid4().hex)
        model_key = payload["model_key"]

        def _generate():
            yield _sse({"type": "request", "request_id": req_id})
            cancel = threading.Event()
            _CANCELS[req_id] = cancel
            try:
                if not slot.store.local_path(model_key) and slot.loaded_key != model_key:
                    yield _sse({"type": "status", "stage": "provision",
                                "message": f"fetching {model_key}…", "progress": 0.0})
                yield _sse({"type": "status", "stage": "load",
                            "message": f"loading {model_key}…"})
                llm = slot.get(model_key)
                messages = _coerce_messages(payload)
                finish = "stop"
                with slot.gen_lock:
                    for chunk in llm.create_chat_completion(
                            messages=messages, stream=True,
                            **_gen_params(payload, slot.n_ctx)):
                        if cancel.is_set():
                            finish = "cancelled"
                            break
                        choice = (chunk.get("choices") or [{}])[0]
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            yield _sse({"type": "token", "text": text})
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                # Normalize llama.cpp's 'length' to the central's finish-reason
                # vocabulary so the relayed DoneEvent validates (central also maps
                # defensively, but emit a clean value at the source).
                if finish == "length":
                    finish = "max_tokens"
                yield _sse({"type": "done", "finish_reason": finish})
            except Exception as exc:  # noqa: BLE001
                logger.warning("stream failed: %s: %s", type(exc).__name__, exc)
                yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                _CANCELS.pop(req_id, None)

        return Response(
            stream_with_context(_generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
            direct_passthrough=True,
        )

    @app.route("/infer/cancel/<request_id>", methods=["POST"])
    def infer_cancel(request_id):
        ev = _CANCELS.get(request_id)
        if ev is None:
            return jsonify({"cancelled": False, "reason": "unknown or finished request"}), 404
        ev.set()
        return jsonify({"cancelled": True, "request_id": request_id})

    @app.route("/probe/<path:model_key>", methods=["POST", "GET"])
    def probe(model_key):
        # No VRAM on this worker — "fit" means the model loaded in RAM and can
        # produce a token. Loading is cached, so a probe warms the first chat.
        before = _free_ram_bytes()
        try:
            llm = slot.get(model_key)
            with slot.gen_lock:
                llm.create_chat_completion(
                    messages=[{"role": "user", "content": "hi"}], max_tokens=1)
            after = _free_ram_bytes()
            used = (before - after) if (before is not None and after is not None) else None
            return jsonify({"model_key": model_key, "ok": True, "fit": True,
                            "vram_free_before": None, "vram_free_after": None,
                            "ram_used": used})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"model_key": model_key, "ok": False, "fit": False,
                            "error": f"{type(exc).__name__}: {exc}"})

    @app.route("/models/unload", methods=["POST"])
    def unload():
        body = request.get_json(silent=True) or {}
        model_key = None if body.get("all") else body.get("model_key")
        evicted = slot.unload(model_key)
        return jsonify({
            "ok": True, "evicted": evicted, "model_key": body.get("model_key"),
            "error": None, "vram_free_before": None, "vram_free_after": None,
            "freed": None,
            "loaded_models": [slot.loaded_key] if slot.loaded_key else [],
        })

    return app


# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------
class WorkerState:
    def __init__(self, name: str, url: str | None, worker_id: str | None,
                 central_url: str | None = None, port: int | None = None):
        self.name = name
        self.url = url            # None unless operator set --advertise/WORKER_URL
        self.worker_id = worker_id
        self.central_url = central_url
        self.port = port
        self.assigned_models: list[str] = []
        self._provisioning: set[str] = set()
        self._provision_lock = threading.Lock()


def _sync_assignment(state: WorkerState, worker: dict, store: ModelStore) -> None:
    """Adopt central's model list and pre-download newly assigned GGUFs."""
    if not isinstance(worker, dict):
        return
    models = worker.get("models") or []
    if models == state.assigned_models:
        return
    state.assigned_models = list(models)
    logger.info("assignment updated: serving %s", models or "(nothing)")

    for model_key in models:
        if store.local_path(model_key):
            continue
        with state._provision_lock:
            if model_key in state._provisioning:
                continue
            state._provisioning.add(model_key)

        def _bg(mk=model_key):
            try:
                logger.info("pre-provisioning assigned model %s…", mk)
                store.provision(mk)
                logger.info("pre-provisioned %s", mk)
            except Exception as exc:  # noqa: BLE001
                logger.warning("pre-provision of %s failed: %s", mk, exc)
            finally:
                with state._provision_lock:
                    state._provisioning.discard(mk)

        threading.Thread(target=_bg, daemon=True).start()


def _load_worker_id(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return (json.load(fh) or {}).get("worker_id")
    except (OSError, ValueError):
        return None


def _save_worker_id(path: str, worker_id: str) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"worker_id": worker_id}, fh)
    except OSError:
        logger.warning("could not persist worker id to %s", path)


def _register(client: CentralClient, state: WorkerState, store: ModelStore, args) -> dict:
    models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    payload = {
        "name": state.name,
        "url": state.url,            # None -> central uses the source IP
        "port": state.port,
        "gpus": [],
        "role": "worker",
        "rpc_endpoint": None,
        "free_ram": _free_ram_bytes(),
        "models": models or None,
        "worker_id": state.worker_id,
        "pkg_version": PKG_VERSION,
        "engine": llama_cpp_status(),
        "pool": os.environ.get("WORKER_POOL", ""),
        "env": env_status(),
    }
    worker = client.register(payload)
    state.worker_id = worker.get("id", state.worker_id)
    if state.worker_id:
        _save_worker_id(args.id_file, state.worker_id)
    _sync_assignment(state, worker, store)
    logger.info("registered as worker id=%s serving models=%s",
                state.worker_id, worker.get("models"))
    return worker


def _heartbeat_loop(client: CentralClient, state: WorkerState,
                    store: ModelStore, slot: LlamaSlot, args) -> None:
    while True:
        time.sleep(args.heartbeat)
        try:
            worker = client.heartbeat(
                state.worker_id,
                {
                    "gpus": [],
                    "loaded_models": [slot.loaded_key] if slot.loaded_key else [],
                    "provisioning": sorted(state._provisioning),
                    "spill": {},
                    "url": state.url,
                    "port": state.port,
                    "pkg_version": PKG_VERSION,
                    "role": "worker",
                    "rpc_endpoint": None,
                    "free_ram": _free_ram_bytes(),
                    "engine": llama_cpp_status(),
                    "pool": os.environ.get("WORKER_POOL", ""),
                    "env": env_status(),
                },
            )
            _sync_assignment(state, worker, store)
            # Converge to central's required package version (installed-package
            # mode only; no-op for a standalone copy). May re-exec on success.
            _self_update_if_needed(worker.get("required_pkg_version"), args)
        except urllib.error.HTTPError as exc:
            if exc.code == 410:
                logger.warning("central returned 410; re-registering")
                try:
                    _register(client, state, store, args)
                except Exception as rexc:  # noqa: BLE001
                    logger.warning("re-register failed: %s", rexc)
            else:
                logger.warning("heartbeat HTTP %s", exc.code)
        except Exception as exc:  # noqa: BLE001
            logger.warning("heartbeat failed: %s", exc)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gguf_worker")
    # Canonical HUGPY_BASE_URL, with the legacy worker var as an alias. Inlined
    # (not the shared abstract_hugpy_dev.central resolver) so this module keeps
    # running standalone when copied to a worker box without the package.
    _central_default = (os.environ.get("HUGPY_BASE_URL")
                        or os.environ.get("WORKER_CENTRAL_URL"))
    p.add_argument("--central", default=_central_default,
                   help="Central base URL, e.g. http://192.168.1.250:7002 "
                        "(env HUGPY_BASE_URL; legacy WORKER_CENTRAL_URL honoured)")
    p.add_argument("--name", default=os.environ.get("WORKER_NAME", socket.gethostname()))
    p.add_argument("--host", default=os.environ.get("WORKER_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("WORKER_PORT", "9100")))
    p.add_argument("--advertise", default=os.environ.get("WORKER_URL"),
                   help="URL central should call back on (default: source IP + port)")
    p.add_argument("--models", default=os.environ.get("WORKER_MODELS", ""),
                   help="Comma-separated model_keys to self-assign on registration")
    p.add_argument("--heartbeat", type=float,
                   default=float(os.environ.get("WORKER_HEARTBEAT", "15")))
    from hugpy_platform import app_dirs as _hp
    p.add_argument("--id-file", default=_hp.gguf_worker_id_file())
    p.add_argument("--models-dir", default=os.environ.get(
        "GGUF_WORKER_MODELS_DIR", os.path.expanduser("~/gguf-models")))
    p.add_argument("--n-ctx", type=int,
                   default=int(os.environ.get("GGUF_WORKER_N_CTX", "4096")))
    p.add_argument("--n-threads", type=int,
                   default=int(os.environ.get("GGUF_WORKER_N_THREADS", "0")) or None)
    p.add_argument("--n-gpu-layers", type=int,
                   default=int(os.environ.get("GGUF_WORKER_N_GPU_LAYERS", "0")))
    # Self-update (installed-package mode only). By default pulls FROM THE CENTRAL
    # we're pointed at — its PEP-503 index at <central>/api/llm/pip/simple — so a
    # WG-only worker needs no PyPI access. Disable with --no-self-update / env.
    p.add_argument("--pkg-index", default=os.environ.get("WORKER_PKG_INDEX"),
                   help="pip --index-url for self-update (default: <central>/api/llm/pip/simple)")
    p.add_argument("--pkg-name", default=os.environ.get("WORKER_PKG_NAME", "hugpy-fleet"),
                   help="distribution to self-update (default hugpy-fleet)")
    p.add_argument("--no-self-update", action="store_true",
                   default=os.environ.get("WORKER_SELF_UPDATE", "1").strip().lower()
                           in ("0", "false", "no", "off"),
                   help="never auto-update from central (env WORKER_SELF_UPDATE=0)")
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    if not args.central:
        print("--central / WORKER_CENTRAL_URL is required")
        return 2

    lcpp = llama_cpp_status()
    if not lcpp.get("installed"):
        logger.warning("llama-cpp-python not importable (%s) — the agent will "
                       "register and heartbeat, but every inference will fail "
                       "until it's installed", lcpp.get("error"))
    else:
        logger.info("llama_cpp %s ready (gpu offload: %s)",
                    lcpp.get("version"), lcpp.get("supports_gpu_offload"))

    advertise = args.advertise or None
    state = WorkerState(name=args.name, url=advertise,
                        worker_id=_load_worker_id(args.id_file),
                        central_url=args.central, port=args.port)
    client = CentralClient(args.central)
    store = ModelStore(args.models_dir, client)
    slot = LlamaSlot(store, n_ctx=args.n_ctx, n_threads=args.n_threads,
                     n_gpu_layers=args.n_gpu_layers)

    try:
        worker = _register(client, state, store, args)
        # Converge to central's required version on startup (may re-exec).
        _self_update_if_needed((worker or {}).get("required_pkg_version"), args)
    except Exception as exc:  # noqa: BLE001
        logger.error("initial registration failed: %s", exc)
        # Keep going — the heartbeat loop retries via 410 re-register, and the
        # server can still serve a worker the operator registers manually.

    hb = threading.Thread(target=_heartbeat_loop,
                          args=(client, state, store, slot, args), daemon=True)
    hb.start()

    logger.info("slim GGUF worker listening on %s:%s (advertising %s)",
                args.host, args.port, state.url or "source IP")
    build_app(state, slot).run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
