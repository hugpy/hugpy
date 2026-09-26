"""Standalone GPU worker agent for the abstract_hugpy_dev LLM pool.

Run this on any box with a GPU and a working ``abstract_hugpy_dev`` install to
donate that GPU's compute to the central console. The agent:

    1. Detects local GPUs.
    2. Registers with the central node (``/api/llm/workers/register``) and keeps
       a persistent worker id in a local state file so restarts reuse the row.
    3. Serves inference over HTTP for the models the central node assigns to it:
           GET  /health
           POST /infer          {model_key, messages|prompt, ...} -> {text, finish_reason}
           POST /infer/stream   -> SSE token/done/error events
       Inference runs through ``abstract_hugpy_dev.managers.dispatch`` exactly like
       the central node, so the worker loads/serves the model on its own GPU.
    4. Heartbeats every ``--heartbeat`` seconds, reporting live GPU stats and
       which models are currently loaded.

The central node's chat route picks an online, assigned worker for the chosen
model and relays this agent's ``/infer/stream`` back to the browser. If no
worker is assigned (or all are offline) the central node runs the model
locally, so adding workers is purely additive.

Usage
-----
    python -m abstract_hugpy_dev.worker_agent \
        --central https://hugpy.ai \
        --name gpu-box-1 \
        --host 10.0.0.5 --port 9100 \
        --models Qwen_Qwen2.5-7B-Instruct,meta-llama_Llama-3.1-8B-Instruct

Every flag also has an env fallback (WORKER_CENTRAL_URL, WORKER_NAME,
WORKER_HOST, WORKER_PORT, WORKER_MODELS, WORKER_ID_FILE, WORKER_HEARTBEAT).
"""
from __future__ import annotations

import os
import sys
import json
import time
import uuid
import socket
import logging
import argparse
import asyncio
import threading
import subprocess
import tempfile
import urllib.request
import urllib.error
import weakref

# Storage-budget refusal (evict-to-fit path). Safe at module scope: budget.py
# imports back from .agent lazily, inside functions, so there is no cycle.
from hugpy_fleet.worker.budget import BudgetRefusal

from flask import Flask, request, jsonify, Response, stream_with_context

logger = logging.getLogger("hugpy_fleet.worker")
from hugpy_fleet.worker.imports import (  # explicit, lazy engine/storage seams
    describe, execute_chat_stream, execute_prompt, get_model_config, get_models_dict, runner_for,
)
from hugpy_platform.central import central_base_url
# Per-model in-process generation gate (concurrency hardening). Light module —
# no heavy deps at import; slot-awareness imports the runner stack lazily. It
# serializes entry into an in-process llama.cpp/transformers runner per model so
# concurrent requests can't race the same non-reentrant native context and SEGV
# the whole worker (the computron 2026-07-11 core-dump class).
from hugpy_fleet.worker import gen_gate
# Worker-side ROLLING AGGREGATE (operator ruling 2026-07-29). Import-safe by
# design: aggregate.py pulls only stdlib + _platform.paths, so it can sit in
# the serving path without dragging the runner stack.
from hugpy_fleet.worker import aggregate as _aggregate
# request_id -> asyncio.Event, so POST /infer/cancel can stop an in-flight
# stream mid-generation. Populated by _stream_sync, tripped by the cancel route.
# Cancellation now rides the shared comms JobStore (attach_cancel/cancel) —
# the per-process _CANCELS dict this file used to keep is gone (F1.3: no
# side channels).


# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------
def detect_gpus() -> list[dict]:
    """Best-effort GPU inventory.

    Tries ``nvidia-smi`` first (no Python deps), then ``torch.cuda``. Returns
    an empty list on a CPU-only box — the worker still registers and serves,
    it just won't be fast. The probe itself lives in :mod:`hugpy._platform.hardware`
    so it stays portable (``nvidia-smi.exe`` on Windows, no probe on Apple silicon).
    """
    from hugpy_platform.hardware import detect_gpus as _detect_gpus

    return _detect_gpus()


def safe_import_torch():
    """Import ``torch``, healing a partially-initialized module first.

    WHY THIS EXISTS (worker ae, 2026-07-05): when this process imports a
    CUDA-built ``llama_cpp`` *before* its first ``import torch``, torch's native
    init trips a circular import and aborts mid-way —
    ``partially initialized module 'torch' has no attribute 'library'``. Python
    then caches that broken half-module in ``sys.modules``, so EVERY later
    ``import torch`` in this process (vision/frame extraction, sd-turbo, whisper)
    hands back the same stale wreck for the process's entire life. One bad import
    ordering silently poisons every torch task on the box until restart. Confirmed
    minimal repro: ``python -c "import torch"`` works; ``python -c "import
    llama_cpp, torch"`` reproduces the abort.

    The durable fix is ordering — import torch before any llama_cpp (see
    :func:`_prime_torch_before_llama`). This helper is the recovery net for a
    race we missed: it (a) returns torch straight from cache when it is already
    fully initialized; (b) otherwise tries a normal import; (c) if the import
    raised OR yielded a half-initialized module (missing ``torch.library``),
    evicts ``torch`` and every ``torch.*`` submodule from ``sys.modules`` and
    retries the import exactly ONCE from a clean slate, logging loudly. A
    still-broken torch (or a genuinely absent one) re-raises so the caller's
    error path reports it.
    """
    import importlib

    def _partial(mod) -> bool:
        # A fully-initialized torch always exposes ``torch.library``; its absence
        # is the fingerprint of the circular-import abort described above.
        return mod is not None and not hasattr(mod, "library")

    cached = sys.modules.get("torch")
    if cached is not None and not _partial(cached):
        return cached  # already fully imported — pure cache hit, no work

    first_error = None
    if cached is None:
        try:
            import torch
            if not _partial(torch):
                return torch
        except Exception as exc:  # noqa: BLE001
            first_error = exc

    # We reach here only when torch is poisoned: the import raised, or the cached
    # / freshly-imported module is half-initialized. Evict the whole torch.*
    # subtree so the retry re-runs torch's init from scratch.
    stale = [name for name in list(sys.modules)
             if name == "torch" or name.startswith("torch.")]
    for name in stale:
        del sys.modules[name]
    logger.warning(
        "safe_import_torch: torch was %s%s — purged %d torch.* module(s) from "
        "sys.modules and retrying import ONCE. This is the llama_cpp/torch CUDA "
        "collision: torch MUST be imported before llama_cpp in this process.",
        "un-importable" if first_error is not None else "partially initialized",
        f" ({type(first_error).__name__}: {first_error})" if first_error else "",
        len(stale),
    )
    importlib.invalidate_caches()
    import torch  # single clean retry; propagates if it still can't init
    return torch


def torch_cuda_status() -> dict:
    """Whether *torch* can actually use CUDA — distinct from nvidia-smi seeing a
    card. Inference runs on the GPU only when ``torch.cuda.is_available()`` is
    True; a CPU-only torch build (or a torch/CUDA-driver mismatch) leaves a
    perfectly good GPU unused. Surfaced in /health so this is diagnosable.

    Goes through :func:`safe_import_torch` so a torch half-poisoned by an earlier
    llama_cpp import is healed here instead of reporting a phantom "no CUDA".
    """
    try:
        torch = safe_import_torch()
        available = bool(torch.cuda.is_available())
        return {
            "available": available,
            "device_count": torch.cuda.device_count() if available else 0,
            "device_name": torch.cuda.get_device_name(0) if available else None,
            "torch_version": getattr(torch, "__version__", None),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


# The llama.cpp capability probe runs in a SUBPROCESS (see llama_cpp_cuda_status)
# so the agent process NEVER imports llama_cpp merely to report engine status.
# A CUDA-built llama_cpp imported into this process breaks every later
# ``import torch`` for the process's life (see safe_import_torch), and this probe
# used to fire on every heartbeat — the single most likely way to poison the box.
# The child prints one JSON object describing the engine; the parent parses it.
_LLAMA_PROBE_CODE = r"""
import json, sys
out = {"installed": False}
try:
    import llama_cpp
    out["installed"] = True
    out["version"] = getattr(llama_cpp, "__version__", None)
    try:
        out["supports_gpu_offload"] = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        out["supports_gpu_offload"] = None
    # supports_vision: a multimodal chat handler (or mtmd) is importable — the
    # same capability the in-process vision runner needs to load an mmproj. Central
    # reads this (engine.supports_vision) and ONLY routes image turns to workers
    # that report it, so an older text-only build is never handed an image.
    supports_vision = False
    try:
        from llama_cpp import llama_chat_format as _cf
        for _name in ("Qwen25VLChatHandler", "Llava16ChatHandler",
                      "Llava15ChatHandler", "MiniCPMv26ChatHandler",
                      "MoondreamChatHandler"):
            if hasattr(_cf, _name):
                supports_vision = True
                break
    except Exception:
        pass
    if not supports_vision:
        try:
            import llama_cpp.mtmd_cpp  # noqa: F401
            supports_vision = True
        except Exception:
            supports_vision = False
    out["supports_vision"] = supports_vision
except Exception as exc:
    out = {"installed": False, "error": "%s: %s" % (type(exc).__name__, exc)}
sys.stdout.write(json.dumps(out))
"""

# Engine build is immutable for a process's life, so the first successful probe
# is cached: no python subprocess (which imports CUDA llama_cpp) spawns on every
# 15s heartbeat / /health hit. A not-installed result is intentionally NOT cached
# — /ops/pip can install the engine at runtime and the next probe must see it.
_LLAMA_PROBE_CACHE: dict | None = None
_LLAMA_PROBE_TIMEOUT = 60.0

# Native llama-server ENGINE build id, cached for the process's life (the binary
# doesn't change under a running agent; a `hugpy install-engine` swap re-execs).
# ``False`` = probed and unresolvable (no native binary / --version failed) — a
# real answer, so it caches too and the heartbeat carries an explicit null. Item
# L (k65): the spawn now probes engine capability but the build id (a bare commit
# like ``1 (039e20a)``) was tracked NOWHERE — surface it beside pkg_version so
# engine skew across the fleet is visible in /llm/workers.
_ENGINE_BUILD_CACHE: "str | bool | None" = None


def _engine_build() -> "str | None":
    """The native llama-server build id — ``<build> (<commit>)`` from
    ``llama-server --version`` (e.g. ``1 (039e20a)``), or ``None`` when no native
    engine is resolvable / the probe fails. Cached (see ``_ENGINE_BUILD_CACHE``).

    llama.cpp prints ``version: <N> (<hash>)`` on stderr (older builds on stdout),
    so both streams are scanned. Fully defensive: any failure -> ``None`` cached,
    never an exception into the heartbeat."""
    global _ENGINE_BUILD_CACHE
    if _ENGINE_BUILD_CACHE is not None:
        return _ENGINE_BUILD_CACHE or None
    build: "str | None" = None
    try:
        from hugpy_engine.native.resolve import ld_library_path_with_engine, server_bin
        binpath = server_bin()
        if binpath:
            # Same LD_LIBRARY_PATH derivation as the real spawn sites (k94): a
            # managed engine's binary needs its sibling libs to exec at all, and
            # this probe must not report null build for an engine that serves.
            env = dict(os.environ)
            _ld = ld_library_path_with_engine(env.get("LD_LIBRARY_PATH"),
                                              bin_path=binpath)
            if _ld:
                env["LD_LIBRARY_PATH"] = _ld
            proc = subprocess.run([binpath, "--version"],
                                  capture_output=True, text=True, timeout=10,
                                  env=env)
            blob = (proc.stderr or "") + "\n" + (proc.stdout or "")
            for line in blob.splitlines():
                line = line.strip()
                # `version: 1 (039e20a)` -> `1 (039e20a)`
                if line.lower().startswith("version:"):
                    build = line.split(":", 1)[1].strip() or None
                    if build:
                        break
    except Exception:  # noqa: BLE001 — telemetry probe never breaks a beat
        build = None
    _ENGINE_BUILD_CACHE = build if build else False
    return build


def llama_cpp_cuda_status() -> dict:
    """Whether *llama.cpp* (GGUF backend) was built with GPU offload support, and
    whether it can decode images (mtmd) — probed in a SUBPROCESS.

    ``n_gpu_layers`` is silently ignored when llama-cpp-python is the CPU-only
    wheel, so a GGUF model runs entirely on CPU even though autofit picked GPU
    layers; ``llama_supports_gpu_offload()`` is the definitive build check. The
    import runs in a child interpreter (never this process) because a CUDA-built
    llama_cpp imported here poisons every later ``import torch`` — see
    :func:`safe_import_torch` and ``_LLAMA_PROBE_CODE``.
    """
    global _LLAMA_PROBE_CACHE
    if _LLAMA_PROBE_CACHE is not None:
        result = _LLAMA_PROBE_CACHE
    else:
        result = _probe_llama_cpp_subprocess()
        if result.get("installed"):
            _LLAMA_PROBE_CACHE = result
    # k94: the NATIVE llama-server state rides in the same dict central already
    # snapshots as "engine", so "installed but unusable" (a resolvable binary
    # whose exec dies on missing sibling libs) is finally visible remotely.
    # Merged fresh on every call — the python-binding probe above is cached for
    # the process life, but the native binary can appear/break at runtime (its
    # own probe carries a short TTL in engine.resolve).
    out = dict(result)
    out["native_engine"] = _native_engine_status_safe()
    return out


def _native_engine_status_safe() -> dict:
    """``engine.resolve.native_engine_status()`` that can never raise into a
    heartbeat or /health response."""
    try:
        from hugpy_engine.native.resolve import native_engine_status
        return native_engine_status()
    except Exception as exc:  # noqa: BLE001 — telemetry never breaks a beat
        return {"found": False, "path": None, "source": None,
                "spawn_ok": None, "error": f"{type(exc).__name__}: {exc}"}


def _probe_llama_cpp_subprocess() -> dict:
    """Run the llama_cpp probe in a child interpreter and parse its JSON stdout.

    Every failure mode (no python, timeout, crash, garbage output) degrades to an
    ``installed: False`` dict carrying an ``error`` string — the same shape the
    old in-process except path produced, so callers and heartbeats are unchanged.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _LLAMA_PROBE_CODE],
            capture_output=True, text=True, timeout=_LLAMA_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"installed": False,
                "error": f"TimeoutExpired: llama_cpp probe exceeded "
                         f"{_LLAMA_PROBE_TIMEOUT:.0f}s"}
    except Exception as exc:  # noqa: BLE001
        return {"installed": False, "error": f"{type(exc).__name__}: {exc}"}
    out = (proc.stdout or "").strip()
    if not out:
        stderr = (proc.stderr or "").strip()
        return {"installed": False,
                "error": f"llama_cpp probe produced no output "
                         f"(rc={proc.returncode}): "
                         f"{stderr or '(stderr empty, bytes=0)'}"}
    try:
        return json.loads(out)
    except Exception as exc:  # noqa: BLE001
        return {"installed": False,
                "error": f"llama_cpp probe output unparseable "
                         f"({type(exc).__name__}): {out}"}


def _prime_torch_before_llama() -> None:
    """Import torch NOW — before this process can import llama_cpp — when torch is
    installed on this box.

    The agent's in-process GGUF fallback (``execute_prompt`` -> python_runner ->
    ``from llama_cpp import Llama``) and the console's torch tasks (vision,
    sd-turbo, whisper) race to be the first native import. If llama_cpp wins, the
    first ``import torch`` aborts mid-init and stays broken for the process's life
    (see :func:`safe_import_torch`). Importing torch first makes it a complete,
    cached module every later import simply reuses — the ordering fix. Best-effort
    and silent when torch isn't installed (CPU/text-only boxes); a torch that
    genuinely can't import is reported per-request by the torch paths, not here.
    """
    try:
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            return  # no torch on this box — nothing to prime, stay quiet
    except Exception:  # noqa: BLE001
        return
    try:
        safe_import_torch()
        logger.info("primed torch ahead of any llama_cpp import "
                    "(llama_cpp/torch CUDA-collision guard)")
    except Exception as exc:  # noqa: BLE001
        logger.warning("torch priming skipped (%s: %s); torch paths will report "
                       "per-request if it truly can't import",
                       type(exc).__name__, exc)


def _safe_int(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _free_ram_bytes() -> int | None:
    """Available RAM in bytes — feeds the allocator's CPU tier. Best-effort.

    Reserve-adjusted (managers.spill honors HUGPY_RAM_RESERVE_GIB) AND, when a
    central RAM ceiling is set, further constrained to the budget-bar spec's
    ``remaining`` (t13/t14) so admission and the console bar can never disagree.
    Wire-compat: this stays the historical ``free_ram`` field; the UNCLAMPED
    reserve-only figure rides alongside as ``free_ram_raw`` (below)."""
    try:
        from hugpy_engine.spill import free_ram_bytes
        return free_ram_bytes()
    except Exception:
        return None


def _free_ram_raw_bytes() -> int | None:
    """Reserve-adjusted budgetable free RAM, UNCLAMPED by the central ceiling
    (t13/t14). Central shows the honest physical-semantics bar and derives the
    ceiling-aware budget from this, so it must ride the heartbeat unclamped —
    distinct from ``free_ram`` (clamped, kept for wire-compat)."""
    try:
        from hugpy_engine.spill import free_ram_raw_bytes
        return free_ram_raw_bytes()
    except Exception:
        return None


def _ram_worker_bytes() -> int | None:
    """RSS of THIS worker's own process tree (agent + slot children) — the
    budget-bar spec's ``worker_usage`` for RAM. Best-effort (spill.py)."""
    try:
        from hugpy_engine.spill import ram_worker_bytes
        return ram_worker_bytes()
    except Exception:
        return None


def _ram_external_bytes() -> int | None:
    """RAM held by everything OUTSIDE this worker's tree (box used − own RSS) —
    the budget-bar spec's ``external_usage`` for RAM. Best-effort (spill.py)."""
    try:
        from hugpy_engine.spill import ram_external_bytes
        return ram_external_bytes()
    except Exception:
        return None


def _trim_host_ram() -> None:
    """Return orphaned host RAM to the OS WITHOUT evicting any model.

    After a model's weights are freed, glibc keeps the freed pages in its
    per-arena free-list, so RSS stays pinned (ae observed at 0 free / 128 GB
    used with nothing loaded). gc.collect() drops Python-side references,
    malloc_trim(0) hands the arena's top free chunks back to the kernel, and
    torch.cuda.empty_cache() releases torch's cached CUDA blocks. Every step is
    best-effort — malloc_trim is glibc/Linux-only (musl/other libc lack it), so
    the whole thing stays defensive. Mirrors the imagegen evict idiom
    (managers/imagegen/imagegen_runner.py ~85-93) plus the malloc_trim."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001 — non-glibc/musl: no malloc_trim, skip
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — no torch/cuda: nothing to release
        pass


def _agent_rss_bytes() -> int | None:
    """Resident RAM (bytes) of THIS agent process — for the free-ram deltas.

    NOT the slot child's ``rss_bytes`` in the heartbeat (a different process):
    reads VmRSS via the same /proc helper the slot agent uses, psutil fallback.
    Best-effort (None, never fabricated)."""
    try:
        from hugpy_engine.serve.slot_agent import _proc_rss_bytes
        rss = _proc_rss_bytes(os.getpid())
        if rss is not None:
            return rss
    except Exception:
        pass
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _ram_total_bytes() -> int | None:
    """RAW physical RAM (MemTotal) in bytes — the pool-budget denominator.

    Unlike _free_ram_bytes (reserve-adjusted + RAM_MAX-capped so central plans
    against budgetable RAM), this is the box's total installed memory, so the
    console can render used-vs-total. Best-effort, mirroring
    _platform/hardware.free_ram_bytes: psutil first, then /proc/meminfo, else
    None (never fabricated)."""
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
    return None


def _spawn_rpc_server(args):
    """Launch llama.cpp's rpc-server so this box lends its GPU to a shard pool.

    Returns the Popen handle, or None if the binary is missing (the node still
    registers/heartbeats, it just won't be usable as a shard backend until a
    CUDA+RPC llama.cpp build provides ``rpc-server``).
    """
    # Prefer an explicit --rpc-bin/WORKER_RPC_BIN, else whatever the engine
    # resolver finds (a `hugpy install-engine` build ships rpc-server), else the
    # bare name on PATH.
    from hugpy_platform.procutil import popen_detached
    from hugpy_engine.native.resolve import rpc_bin as _resolve_rpc
    binary = args.rpc_bin if (args.rpc_bin and args.rpc_bin != "rpc-server") else None
    binary = binary or _resolve_rpc() or "rpc-server"
    cmd = [binary, "-H", args.rpc_host, "-p", str(args.rpc_port)]
    try:
        proc = popen_detached(cmd)  # noqa: S603 — operator-controlled args
        logger.info("rpc-server up: %s (pid %s)", " ".join(cmd), proc.pid)
        return proc
    except FileNotFoundError:
        logger.error(
            "rpc-server binary %r not found — run `hugpy install-engine --cuda` or "
            "build a CUDA+RPC llama.cpp (cmake -DGGML_CUDA=on -DGGML_RPC=ON) and set "
            "--rpc-bin/WORKER_RPC_BIN. This node registers but can't serve as a "
            "shard backend.", binary)
        return None
    except OSError as exc:
        logger.error("failed to start rpc-server (%s): %s", " ".join(cmd), exc)
        return None


def _local_ip_toward(central_url: str) -> str | None:
    """The worker's own LAN IP on the route it uses to reach central.

    Opening a UDP socket toward central (no packets are actually sent on
    connect) makes the kernel pick the source address it WOULD use — i.e. the
    worker's real outbound IP (e.g. 192.168.1.128), not loopback/127.0.1.1.

    This is what we advertise, because central can't derive it reliably: when
    the worker reaches central via a public domain, NAT hairpinning makes the
    source IP central sees the router's address (192.168.1.1), not the worker's.
    """
    from urllib.parse import urlparse
    try:
        parsed = urlparse(central_url)
        host = parsed.hostname or central_url
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(2.0)
            s.connect((host, port))
            ip = s.getsockname()[0]
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Central node client (registration + heartbeat)
# ---------------------------------------------------------------------------
class WorkerRejected(Exception):
    """Central refused this worker terminally (401 token / 403 blocked).

    Distinct from a transient error or a 410 (re-register): the operator has
    revoked/blocked us, so the agent should stop rather than retry.
    """
    def __init__(self, code: int, message: str = ""):
        super().__init__(message or f"rejected with HTTP {code}")
        self.code = code


# CENTRAL'S OWN terminal-refusal markers. A 401/403 is only terminal (stop, don't
# respawn) when it is CENTRAL saying so — its register/heartbeat routes
# abort(401/403, description=...) with these exact strings (see
# hugpy_server/app/routes/worker_routes.py: "Worker is blocked by the operator."
# at the admission gate, "Worker enrollment token invalid or required." at the
# enrollment gate). A 401/403 whose body carries NEITHER marker is almost always
# an INTERMEDIARY between us and central — a reverse-proxy / web auth gate /
# captive login page (a-brain 2026-09-24: dev.hugpy.ai's web gate returned a 403
# HTML page while central's own record was 'approved' and central logged nothing;
# the agent read it as a block and os._exit(0)'d permanently). Those must be
# logged with the URL + body and RETRIED with backoff, never treated as a block.
_CENTRAL_BLOCK_MARKER = "Worker is blocked by the operator."
_CENTRAL_ENROLL_MARKER = "Worker enrollment token invalid or required."


def _is_central_terminal_refusal(code: int, body: str) -> bool:
    """True iff an HTTP ``code``/``body`` is CENTRAL's own terminal worker refusal
    (block or enrollment reject), as opposed to any other 401/403 from a proxy or
    web gate sitting in front of central. Body-marker matched, case-insensitively
    and whitespace-tolerant, so an HTML error page that wraps the description
    still matches while a generic gate page does not."""
    if code not in (401, 403):
        return False
    hay = " ".join((body or "").split()).lower()
    if not hay:
        # Central always sends a description; a truly empty body is NOT central's
        # refusal (a bare proxy 403). Don't treat it as terminal.
        return False
    return (_CENTRAL_BLOCK_MARKER.lower() in hay
            or _CENTRAL_ENROLL_MARKER.lower() in hay)


# WORKER-DB-HEARTBEAT-20260910: the heartbeat also lands in Postgres (comms/heartbeat_db.py) so
# central/console can read worker state from the DB instead of asking over HTTP.
_DB_BEAT_LOCK = __import__("threading").Lock()
_DB_BEAT_BUSY = {"v": False}


def _db_heartbeat_async(worker_id: str, payload: dict) -> None:
    try:
        from hugpy_fleet.central import heartbeat_db
        if not heartbeat_db.dsn():
            return
        with _DB_BEAT_LOCK:
            if _DB_BEAT_BUSY["v"]:
                return                      # previous write still running: skip this beat
            _DB_BEAT_BUSY["v"] = True
        name = str(payload.get("name") or os.environ.get("WORKER_NAME") or __import__("socket").gethostname())

        def run():
            try:
                heartbeat_db.upsert(worker_id, name, payload)
            finally:
                with _DB_BEAT_LOCK:
                    _DB_BEAT_BUSY["v"] = False
        __import__("threading").Thread(target=run, name="hugpy-db-beat", daemon=True).start()
    except Exception:  # noqa: BLE001 — never touch the HTTP beat
        pass


class CentralClient:
    def __init__(self, central_url: str, token: str | None = None):
        # Endpoints live under /api on the central Flask app.
        self.base = central_url.rstrip("/") + "/api/llm/workers"
        self.token = token

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(
            self.base + path, data=data, headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # A 401/403 is terminal (stop, don't respawn) ONLY when CENTRAL says
            # so — its route bodies carry a specific marker (see
            # _is_central_terminal_refusal). Any other 401/403 is an intermediary
            # (proxy / web gate / login page) between us and central: log it with
            # the URL + body snippet and re-raise so the caller RETRIES with
            # backoff. Other codes (e.g. 410 "re-register") propagate unchanged.
            if exc.code in (401, 403):
                body = ""
                try:
                    raw = exc.read()
                    body = raw.decode("utf-8", "replace") if raw else ""
                except Exception:  # noqa: BLE001 — body is best-effort context
                    body = ""
                if _is_central_terminal_refusal(exc.code, body):
                    reason = exc.reason or ""
                    detail = " ".join(body.split())[:300]
                    raise WorkerRejected(
                        exc.code,
                        (reason + (": " + detail if detail else "")).strip()) from exc
                logger.warning(
                    "central call %s returned HTTP %s but the body is NOT central's "
                    "own worker refusal — treating it as an intermediary (proxy / "
                    "web gate), NOT a block; will retry. body: %s",
                    self.base + path, exc.code,
                    (" ".join(body.split())[:300] or "(empty)"))
                raise
            raise

    def register(self, payload: dict) -> dict:
        return self._post("/register", payload)

    def heartbeat(self, worker_id: str, payload: dict) -> dict:
        _db_heartbeat_async(worker_id, payload)   # WORKER-DB-HEARTBEAT-20260910
        return self._post(f"/{worker_id}/heartbeat", payload)

    def evictions_ingest(self, events: list) -> dict:
        """Relay a batch of eviction-telemetry events to central.

        Its own endpoint rather than a heartbeat rider: the beat is every few
        seconds and is the fleet's liveness signal — pinning sub-second
        eviction detail to it would either slow the stream to beat cadence or
        inflate the one payload the fleet cannot afford to make heavy (the
        2026-07-27 stat storm starved heartbeats exactly this way). Same base,
        same Bearer, short timeout: telemetry must never hold a thread.

        ``/llm/evictions/ingest`` lives outside the ``/llm/workers`` prefix this
        client is based on, so the base is trimmed back one segment."""
        base = self.base.rsplit("/llm/workers", 1)[0]
        data = json.dumps({"events": events}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(
            base + "/llm/evictions/ingest", data=data, headers=headers,
            method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Local inference (reuses the same dispatch the central node uses)
# ---------------------------------------------------------------------------
def _ensure_present(payload: dict, central_url: str | None, state=None) -> None:
    """Provision the requested model before inference (central-first, HF fallback).

    ``state`` opts the pull into the STORAGE BUDGET (evict-to-fit, else refuse).
    A BudgetRefusal PROPAGATES: an unfittable model must fail loudly here rather
    than fall through to a confusing downstream "model not found".
    """
    model_key = payload.get("model_key")
    if not model_key:
        return
    try:
        from hugpy_storage.provision import ensure_model_present, ensure_model_registered

        # Learn the model from central if the worker wasn't built with it, then
        # run inference against the canonical local key.
        canonical = ensure_model_registered(model_key, central_url)
        if canonical and canonical != model_key:
            payload["model_key"] = canonical
        # DEMAND: a real inference call is waiting on this model. Central never
        # budget-refuses a demand pull (2026-07-17) — the worker's own fit_plan
        # evicts to fit it; refusing a called model at central would break serving.
        ensure_model_present(payload.get("model_key"), central_url, state=state,
                             purpose="demand")
        if state is not None:
            state.refused.pop(payload.get("model_key"), None)
    except BudgetRefusal as exc:
        if state is not None:
            state.refused[payload.get("model_key") or model_key] = dict(exc.reason)
        logger.error("provisioning of %s REFUSED: %s", model_key,
                     exc.reason.get("reason"))
        raise
    except Exception as exc:
        logger.warning("provisioning check for %s failed: %s", model_key, exc)


def _model_key_refusal(payload: dict, central_url: str | None) -> "str | None":
    """Why this request names no servable model — the 400 text — or None.

    A worker serves THE MODEL IT WAS ASKED FOR. Without this gate a request
    whose ``model_key`` is absent falls all the way through to
    ``resolve_model_key``'s last resort (the chat default) and the box answers
    with a completely different model, labelled with the caller's request: a
    silent substitution that reads as a working answer at every layer above.
    An UNKNOWN key is the same failure with an extra step — it raised a KeyError
    deep in resolution and came back as an opaque 500.

    Refusing here makes both honest and actionable: 400, name the key, say what
    the worker has to do about it.

    ``task`` WITHOUT a model_key is deliberately still allowed: TASK_DEFAULTS is
    a per-task designation the caller opted into by naming the task, not the
    fallthrough this exists to kill.

    Side effect (the same one ``_ensure_present`` performs): a key central knows
    under another name is REWRITTEN to the canonical local key, so resolution
    downstream works on the name this worker registered."""
    model_key = payload.get("model_key")
    if model_key is None or not str(model_key).strip():
        if payload.get("task"):
            return None
        return ("model_key is required: this worker serves the model you name "
                "and never substitutes a default — pass model_key (or a task, "
                "to use that task's designated default)")
    try:
        from hugpy_storage.provision import ensure_model_registered
        canonical = ensure_model_registered(model_key, central_url)
    except Exception as exc:  # noqa: BLE001 — can't tell -> don't invent a 400
        logger.warning("model_key check for %s failed: %s", model_key, exc)
        return None
    if not canonical:
        return (f"unknown model_key {model_key!r}: this worker has no such "
                "model and central could not teach it one — check the key, or "
                "register/assign the model first. No default was substituted.")
    payload["model_key"] = canonical
    # DRAFT-MODEL-GATE-20260910: same refusal for callers that reach the worker directly.
    try:
        from hugpy_engine.draft_models import draft_model_reason
        _draft = draft_model_reason(canonical)
    except Exception:  # noqa: BLE001
        _draft = None
    if _draft:
        return _draft
    return None


def _ensure_present_streaming(payload: dict, central_url: str | None, state=None):
    """Provision the model, yielding SSE 'status' events with download progress.

    Yields encoded SSE lines (status/error). Returns normally once the model is
    present (or was already). Throttled so we don't flood the stream.

    ``state`` opts the pull into the STORAGE BUDGET. A refusal is yielded as an
    SSE 'error' event carrying the structured reason — the stream ends honestly
    ("won't fit: needs X…") instead of showing a progress bar for a download
    that was never going to start.
    """
    model_key = payload.get("model_key")
    if not model_key:
        return
    try:
        from hugpy_storage.provision import ensure_model_present, ensure_model_registered, model_is_local

        # Learn the model from central first, then work the rest of the stream
        # against the canonical local key (so resolution/loading can find it).
        canonical = ensure_model_registered(model_key, central_url)
        if canonical and canonical != model_key:
            payload["model_key"] = canonical
            model_key = canonical

        if model_is_local(model_key):
            return  # nothing to do; go straight to generation

        yield _sse({"type": "status", "stage": "provision",
                    "message": f"fetching {model_key}…", "progress": 0.0})

        # provision runs in a worker thread; it pushes (done,total,fname) onto a
        # queue that we drain into throttled SSE status events from this thread.
        import queue
        import threading

        q: "queue.Queue" = queue.Queue()
        result = {"ok": False, "err": None}

        def _progress(done, total, fname):
            q.put((done, total, fname))

        def _run():
            try:
                # DEMAND: a live chat is streaming; never budget-refused centrally.
                result["ok"] = ensure_model_present(model_key, central_url,
                                                    progress=_progress, state=state,
                                                    purpose="demand")
            except Exception as exc:  # pragma: no cover
                result["err"] = exc
            finally:
                q.put(None)  # sentinel: done

        th = threading.Thread(target=_run, daemon=True)
        th.start()

        last_emit = 0.0
        while True:
            item = q.get()
            if item is None:
                break
            done, total, fname = item
            now = time.time()
            # Emit at most ~3x/sec, but always emit the first/last.
            if now - last_emit < 0.33 and done < (total or 1):
                continue
            last_emit = now
            frac = (done / total) if total else 0.0
            yield _sse({
                "type": "status", "stage": "provision",
                "message": f"downloading {model_key} ({_human(done)}/{_human(total)})",
                "progress": round(frac, 4),
                "done_bytes": done, "total_bytes": total, "file": fname,
            })
        th.join(timeout=1.0)

        if isinstance(result["err"], BudgetRefusal):
            # Storage verdict, not a transfer failure: the pull never started.
            # Carry the structured reason so the UI can show WHY it's missing.
            reason = result["err"].reason
            if state is not None:
                state.refused[model_key] = dict(reason)
            yield _sse({"type": "error", "stage": "provision",
                        "refused": reason,
                        "message": f"{model_key} won't fit: {reason.get('reason')}"})
            return
        if result["err"] is not None:
            yield _sse({"type": "error",
                        "message": f"provisioning failed: {result['err']}"})
            return
        if not result["ok"]:
            # HONEST PROPAGATION (incident 2026-07-28). This used to be the
            # flat "could not fetch model X from central or HF" — a sentence
            # that is true of a full disk, a revoked token, a 404 and a dead
            # NIC alike, and therefore tells the operator nothing. The drive was
            # 100% full; finding that out cost an ssh session and a journalctl
            # read. _provision_now now records the structured cause instead of
            # discarding it at the boolean boundary, so name it.
            #
            # ONE LINE, NO TRACEBACK, and NOT prefixed with the worker name —
            # the central relay's _humanize_worker_error already stamps
            # "The '<worker>' worker could not complete this request: " on the
            # front, and doubling it reads like a bug.
            #
            # BOTH wordings below are in remote._PERMANENT_LOAD_MARKERS
            # ("could not fetch model" / "could not provision"), so the central
            # relay still classifies this as a permanent, non-retryable load
            # failure. That matters more than the prose: retrying a request
            # against a 100%-full drive is a storm, not a recovery.
            msg = f"could not fetch model {model_key} from central or HF"
            try:
                from hugpy_storage.provision import last_failure
                cause = last_failure(model_key) or {}
                human = (cause.get("human") or cause.get("reason") or "").strip()
                if human:
                    msg = (f"could not provision {model_key}: {human}"
                           if cause.get("errno_name")
                           else f"could not fetch model {model_key}: {human}")
            except Exception:  # noqa: BLE001 — a missing cause is not a reason
                pass           # to lose the error entirely
            yield _sse({"type": "error", "stage": "provision", "message": msg})
            return
        yield _sse({"type": "status", "stage": "provision",
                    "message": "model ready, loading…", "progress": 1.0})
    except Exception as exc:
        logger.warning("streaming provisioning for %s failed: %s", model_key, exc)


from hugpy_platform.formatting import human_bytes

def _human(n) -> str:
    return human_bytes(n, empty="?")


def _materialize_file(payload: dict) -> str | None:
    """Rebuild an inlined upload (file_b64/file_name) into a local temp file.

    Central ships uploaded files as base64 since the worker can't see central's
    UPLOADS_HOME. We write the bytes to a temp file, point ``payload["file"]``
    at it, and return the temp path so the caller can delete it afterwards.
    Returns None when there's nothing to materialize.
    """
    b64 = payload.pop("file_b64", None)
    name = payload.pop("file_name", None)
    if not b64:
        return None
    import base64
    import tempfile

    suffix = ""
    if name and "." in name:
        suffix = "." + name.rsplit(".", 1)[-1]
    fd, tmp_path = tempfile.mkstemp(prefix="hugpy_worker_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(base64.b64decode(b64))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    payload["file"] = tmp_path
    return tmp_path


def _cleanup_file(path: str | None) -> None:
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


def _jsonable(o):
    """Deep-convert a task result to plain JSON types. model_dump() SHOULD do
    this alone, but a nested pydantic model has escaped it in the field
    (2026-07-29, ae: 'Object of type GeneratedImage is not JSON serializable'
    from jsonify — ComfyUI generated the image, then the response died, and
    central held/retried a request that had actually SUCCEEDED, re-generating
    for 25 minutes). The envelope must never lose a finished result to a
    serialization quirk, so sanitize recursively and stringify as a last
    resort — a lossy string beats a 500 that throws the work away."""
    if hasattr(o, "model_dump"):
        try:
            o = o.model_dump()
        except Exception:  # noqa: BLE001
            o = getattr(o, "__dict__", None) or str(o)
    elif hasattr(o, "dict") and callable(getattr(o, "dict")) \
            and not isinstance(o, dict):
        try:
            o = o.dict()                      # pydantic v1 residents
        except Exception:  # noqa: BLE001
            o = str(o)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, (bytes, bytearray)):
        import base64 as _b
        return _b.b64encode(bytes(o)).decode("ascii")
    return str(o)


def _load_failure_payload(exc, *, classify: bool = False):
    """The additive ``load_failure`` dict for an error payload (2026-09-23):
    {class, loader_stderr, path} from the engine's structured marker on the
    cause chain, so central / the audit tool / the grader classify a failed
    load without regex. None when ``exc`` is not a load failure (unless
    ``classify`` — for /probe, where every failure IS a load failure) or the
    engine predates the marker. Older central ignores the unknown key."""
    try:
        from hugpy_engine.serve.load_failure import load_failure_of
        return load_failure_of(exc, classify=classify)
    except Exception:  # noqa: BLE001 — never fail an error path over metadata
        return None


def _with_load_failure(body: dict, exc, *, classify: bool = False) -> dict:
    lf = _load_failure_payload(exc, classify=classify)
    if lf:
        body["load_failure"] = lf
    return body


def _run_once(payload: dict) -> dict:
    #from abstract_hugpy_dev.managers.dispatch import execute_prompt

    tmp = _materialize_file(payload)
    try:
        result = execute_prompt(**payload)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)

        # Return the full result envelope so ANY task (embed, vision, whisper, …)
        # round-trips back to central as its real result_type — not just chat
        # text. Central's DelegatingRunner validates this into result_type.
        # Sanitized DEEP (_jsonable): jsonify must never 500 on a result that
        # the engine already finished producing.
        if hasattr(result, "model_dump"):
            return _jsonable(result)
        # Non-pydantic fallback (shouldn't happen for a registered runner).
        return {
            "ok": getattr(result, "ok", True),
            "text": getattr(result, "text", None) or str(result),
            "finish_reason": getattr(result, "finish_reason", None) or "stop",
        }
    finally:
        _cleanup_file(tmp)


_SPILL_ENV = {
    # Internal central->worker seat contract.  This is not a user allocation
    # knob: central resolves the persisted per-worker GGUF pin and projects the
    # exact artifact so an already-registered worker observes pin changes.
    "gguf_file": "HUGPY_GGUF_FILE",
    "n_gpu_layers": "HUGPY_N_GPU_LAYERS",
    # MoE expert split (2026-07-24): N MoE layers whose expert tensors stay on
    # CPU (999 = all). Rides to slot children as per-load opts (runners/get.py)
    # and to llama-server as --n-cpu-moe. Cleared when absent — a leaked split
    # would silently displace the next model's experts.
    "n_cpu_moe": "HUGPY_N_CPU_MOE",
    # bitsandbytes 4-bit (operator lever, 2026-07-26). Rides the SAME spill
    # wire as n_cpu_moe so central's per-model decision reaches the loader by
    # the channel that already exists. Cleared-when-absent (below) so a lever
    # switched off cannot leak onto the next model loaded in this process.
    "bnb_4bit": "HUGPY_BNB_4BIT",
    "gpu_mem_gib": "HUGPY_GPU_MEM_GIB",
    "cpu_mem_gib": "HUGPY_CPU_MEM_GIB",
    # Explicit per-model core budget (slot loads pass it to the child;
    # in-process loads read DEFAULT_LLAMA_THREADS at build).
    "threads": "DEFAULT_LLAMA_THREADS",
    "tensor_split": "HUGPY_TENSOR_SPLIT",
    "main_gpu": "HUGPY_MAIN_GPU",
    "n_gpu": "HUGPY_N_GPU",
    # Cross-machine RPC sharding: comma-separated "host:port" of llama.cpp
    # rpc-servers to offload layers onto. When central's allocator decides to
    # shard a model, it ships this (+ tensor_split) as a per-request spill
    # override; spill.llama_kwargs() turns it into Llama(rpc_servers=...).
    "rpc_servers": "HUGPY_RPC_SERVERS",
    # k37 allocation modes: only max-ram/explicit ride these NEW keys (the
    # gpu-only/ram-only/max-gpu trio keeps the unchanged n_gpu_layers wire).
    # Central version-gates emission to workers that honor them (>= 0.1.203).
    "alloc_mode": "HUGPY_ALLOC_MODE",
    "leniency_pct": "HUGPY_LENIENCY_PCT",
    "priority_device": "HUGPY_PRIORITY_DEVICE",
    # k56 polite load: admission may spend only genuinely free headroom and
    # never evicts a resident. Cleared-when-absent (below) — a leaked polite
    # flag would silently make the NEXT model refuse instead of making room,
    # which is a dead-wrong knob in the other direction.
    "no_evict": "HUGPY_NO_EVICT",
    # Allocation PROVENANCE (2026-09-23): central's {"kind": "designation" |
    # "per-request", "mode", "request_id", "at"} for this request's placement,
    # projected as JSON so the slot records who asked for the seat it loads.
    # Not a load contract (see _LOAD_CONTRACT_KEYS) — it names the asker only.
    "alloc_source": "HUGPY_ALLOC_SOURCE",
}

# Mode-contract keys are CLEARED when absent from a request's spill: a leaked
# HUGPY_ALLOC_MODE=max-ram from a previous request would silently flip the
# next model's placement (a dead-wrong knob), unlike the layer/budget knobs
# whose stickiness is long-standing behavior we don't change here.
_SPILL_ENV_CLEAR_WHEN_ABSENT = ("alloc_mode", "leniency_pct", "priority_device",
                                "n_cpu_moe", "bnb_4bit", "no_evict",
                                "gguf_file",
                                # n_gpu_layers MUST be cleared too (2026-07-27).
                                # It was the one spill key that persisted, and
                                # the env is PROCESS-WIDE on the agent — so one
                                # ram-only model poisoned every model after it.
                                #
                                # Observed: MN-GRAND (44 GB transformers)
                                # correctly derives ram-only -> {"n_gpu_layers":
                                # "off"} -> HUGPY_N_GPU_LAYERS=off. Never
                                # cleared. VACE-Wan2.1-1.3B then arrives as
                                # max-gpu -> spill {} -> nothing to set ->
                                # n_gpu_layers_intent() reads the STALE "off" ->
                                # "cpu" -> a 1.3B model loads entirely in RAM on
                                # a completely empty card.
                                #
                                # "Absent" must mean UNSET, never "inherit
                                # whatever the last model asked for". Placement
                                # is per-model; a sticky global is a
                                # cross-request leak, not a default. Every
                                # sibling key above already clears; this one was
                                # simply missed.
                                "n_gpu_layers",
                                # PER-DEVICE pin (2026-09-25): the GPU a request
                                # binds is per-model, and both HUGPY_MAIN_GPU and
                                # HUGPY_TENSOR_SPLIT feed PROCESS-WIDE placement
                                # (in-process torch VRAM reads + llama_kwargs) —
                                # a leaked main_gpu/tensor_split from one model
                                # would silently pin the NEXT model to the wrong
                                # card / a stale split. So they clear-when-absent
                                # exactly like the mode-contract keys above.
                                "main_gpu",
                                "tensor_split",
                                # provenance is per-request by definition
                                "alloc_source")


# ── operator resource limits (two-tier) ─────────────────────────────────────
# This box's OWN unit config is the hard ceiling; central may set per-worker
# limits but they apply only as a TIGHTENING (min of the two). Originals are
# captured at import so a central limit can be raised again later without
# being mistaken for local config.
_CAP_KNOBS = {
    "ram_max_gib": "HUGPY_RAM_MAX_GIB",
    "gpu_mem_gib": "HUGPY_GPU_MEM_GIB",
    "threads": "DEFAULT_LLAMA_THREADS",
}
_LOCAL_CAP_ENV = {k: os.environ.get(env) for k, env in _CAP_KNOBS.items()}


def _local_caps() -> dict:
    """The operator-configured ceilings from this box's own config, reported to
    central so it can only tighten, never exceed them. Reserves ride along for
    display."""
    out: dict = {}
    for key in _CAP_KNOBS:
        raw = _LOCAL_CAP_ENV.get(key)
        if raw in (None, ""):
            continue
        try:
            out[key] = int(raw) if key == "threads" else float(raw)
        except ValueError:
            continue
    for key, env in (("ram_reserve_gib", "HUGPY_RAM_RESERVE_GIB"),
                     ("vram_reserve_gib", "HUGPY_VRAM_RESERVE_GIB"),
                     # The box's stated STORAGE delegation: how much local disk it
                     # gives the model cache. Reported as a cap so a central
                     # disk_cache_gib limit is clamped to it (_clamp_limits) — the
                     # worker's delegation wins, same rule as RAM. Absent → central
                     # may set any disk_cache_gib (unclamped).
                     ("disk_cache_gib", "HUGPY_DISK_CACHE_MAX_GIB")):
        raw = os.environ.get(env)
        if raw:
            try:
                out[key] = float(raw)
            except ValueError:
                pass
    # Capability marker (2026-09-25): this worker HONORS a central per-GPU device
    # pin — it threads main_gpu -> the slot child's CUDA_VISIBLE_DEVICES, places
    # in-process torch on the chosen card, and reports per-device residents. A
    # POSITIVE signal (not a version guess): central only emits a device pin to a
    # worker that advertises this, so an older worker never gets a dead knob and
    # keeps llama.cpp's own auto-split / device-0 behavior. Value is a marker, not
    # a ceiling, so _clamp_limits (which only reads keys that match a limit)
    # ignores it.
    out["device_pin"] = 1
    return out


def _adopt_storage_inputs(state: "WorkerState", worker: dict | None) -> None:
    """Store the STORAGE budget's two central-owned inputs on state.

      * ``limits`` — carries ``disk_cache_gib``, central's storage allocation
        for this box. The auto-evict path is OFF until it is set (budget.cap_bytes).
      * ``model_last_picked`` — central's ``{model_key: epoch}`` LRU clock, the
        FIFO key. The worker can't know it: central routes the calls.
      * ``allocated`` — the ALLOCATION-LEVEL totals (operator, 2026-07-16: "show
        how much is needed based on the total size of all models allocated").
        Sizing the assignment set needs the MANIFEST, which only central holds:
        doing it worker-side would mean one HTTP round-trip PER assigned model,
        inside the single-flight provision lock, on the refusal path. Central
        already computes this per read (storage_proposal.allocated_totals) and
        every heartbeat reply carries it, so the worker just adopts the answer
        and the refusal reason stays a pure, offline computation.

    Never raises into the heartbeat: a malformed reply just leaves the previous
    values in place (and an absent allocation simply keeps the budget unmanaged).
    """
    if not isinstance(worker, dict):
        return
    limits = worker.get("limits")
    if isinstance(limits, dict):
        state.limits = dict(limits)
        # HOT-TIER ALIGNMENT (slice 4): project central's disk_cache_gib into an
        # env var so the hot-cache tier (a different process context that never
        # holds `state`) can fold it into its own min-wins when it shares the
        # store drive. The tier reads this LIVE (hot_cache._store_disk_cap_gib),
        # so a fresh heartbeat's number takes effect without a restart. Absent /
        # cleared -> the tier simply has no central term. This is projection only;
        # the AUTHORITATIVE budget gate stays budget.resolve_effective_cap.
        try:
            dc = limits.get("disk_cache_gib")
            if dc in (None, ""):
                os.environ.pop("_HUGPY_CENTRAL_DISK_CACHE_GIB", None)
            else:
                os.environ["_HUGPY_CENTRAL_DISK_CACHE_GIB"] = str(float(dc))
        except (TypeError, ValueError):
            os.environ.pop("_HUGPY_CENTRAL_DISK_CACHE_GIB", None)
        # Same projection for the per-worker reserve (carved OUT of the cap), so
        # stateless/minimal-state callers resolve the same headroom via env.
        try:
            dr = limits.get("disk_reserve_gib")
            if dr in (None, ""):
                os.environ.pop("_HUGPY_CENTRAL_DISK_RESERVE_GIB", None)
            else:
                os.environ["_HUGPY_CENTRAL_DISK_RESERVE_GIB"] = str(float(dr))
        except (TypeError, ValueError):
            os.environ.pop("_HUGPY_CENTRAL_DISK_RESERVE_GIB", None)
    lp = worker.get("model_last_picked")
    if isinstance(lp, dict):
        state.model_last_picked = dict(lp)
    # ── THE ONE LEDGER, second + third columns (eviction flow, 2026-07-25) ───
    # ``model_call_stats`` ({model_key: {"calls": n}}) is key ③ of the shared
    # eviction sort; ``model_alloc_modes`` ({model_key: mode}) is the PREFERENCE
    # behind key ① (the cliff order). Both are central-owned for exactly the
    # reason last_picked is: the worker cannot know them (central routes the
    # calls and holds the persisted allocations), and the spec requires BOTH
    # sides to rank from the SAME numbers or their victim sets diverge.
    #
    # Adopted into _RUNTIME_SETTINGS rather than onto `state` so the readers
    # (_model_alloc_mode / _model_call_stats) match the established shape
    # central already uses for ctx_pct / priority / residency projections.
    #
    # ABSENT keys leave the previous values untouched, and an absent map simply
    # means every model degrades to zero calls / the blank max-gpu preference —
    # which makes key ① a constant and key ③ a tie, i.e. exactly today's
    # idle-first ordering. A pre-ledger central therefore changes nothing.
    cs = worker.get("model_call_stats")
    if isinstance(cs, dict):
        _RUNTIME_SETTINGS["model_calls"] = {
            k: (v or {}).get("calls") or 0 for k, v in cs.items()
            if isinstance(v, dict)}
    am = worker.get("model_alloc_modes")
    if isinstance(am, dict):
        _RUNTIME_SETTINGS["alloc_mode"] = {k: str(v) for k, v in am.items() if v}
    storage = worker.get("storage")
    if isinstance(storage, dict) and storage.get("allocated_count") is not None:
        state.allocated = {
            "allocated_total_bytes": storage.get("allocated_total_bytes"),
            "allocated_count": storage.get("allocated_count"),
            "allocated_unknown_count": storage.get("allocated_unknown_count"),
        }


def _apply_central_limits(worker: dict | None) -> None:
    """Adopt central's per-worker limits as min(central, local config)."""
    limits = (worker or {}).get("limits") or {}
    for key, env in _CAP_KNOBS.items():
        vals = []
        local_raw = _LOCAL_CAP_ENV.get(key)
        if local_raw not in (None, ""):
            try:
                vals.append(float(local_raw))
            except ValueError:
                pass
        if limits.get(key) is not None:
            try:
                vals.append(float(limits[key]))
            except (TypeError, ValueError):
                pass
        if not vals:
            # Neither side sets it: clear a previously-applied central limit.
            if local_raw in (None, "") and env in os.environ:
                os.environ.pop(env, None)
            continue
        eff = min(vals)
        os.environ[env] = str(int(eff)) if key == "threads" else str(eff)


# ---------------------------------------------------------------------------
# MATERIALIZATION — "a runner object exists" is not "the weights are loaded".
#
# THE 2026-07-28 FALSEHOOD: the compute tab showed "🔥 serving
# Qwen2.5-7B-Instruct-GGUF · resident" on computron for a model that had NEVER
# loaded. Provisioning had failed (full disk), but dispatch had already put a
# HOLLOW runner wrapper into _INSTANCES — runners are lazy by design, the heavy
# load happens on first ``.runner`` access — and ``touch_model`` had stamped
# last_used before the load was attempted. _allocations() then emitted a
# kind="ram" row with serving=True, and nothing downstream could tell the
# difference between that and a hot model.
#
# Operator doctrine, "residency must be measured". So the heartbeat now states
# what it actually knows: ``materialized`` is True only for a model whose weights
# we SAW materialize, False for a runner we know is hollow, and omitted when this
# build genuinely cannot tell (never guess — an unknown must read as unknown, and
# central/UI treat absent exactly as they did before).
# ---------------------------------------------------------------------------

_MATERIALIZED: set = set()
_MATERIALIZED_LOCK = threading.Lock()


def _materialize(runner, model_key: str | None = None) -> None:
    """Force a lazy runner's weights RESIDENT, with serve telemetry around it.

    Replaces the ``_ensure = getattr(runner, "ensure_loaded", None); if
    callable(...)`` incantation that appeared verbatim at four call sites. Same
    behavior — including "a runner without ensure_loaded is a no-op" — plus
    load.start/load.done/load.fail on the serve-telemetry stream and an honest
    record of what actually materialized.

    Raises exactly what ``ensure_loaded`` raises: every existing call site has
    its own error handling and this must not swallow a load failure."""
    mk = model_key or getattr(runner, "model_key", None) or "?"
    ensure = getattr(runner, "ensure_loaded", None)
    if not callable(ensure):
        return
    engine = type(runner).__name__
    t0 = time.time()
    if _evt is not None:
        _evt_emit("load.start", model_key=mk, engine=engine)
    try:
        ensure()
    except Exception as exc:  # noqa: BLE001 — observe, then re-raise unchanged
        if _evt is not None:
            _evt_emit("load.fail", model_key=mk, engine=engine,
                      error=f"{type(exc).__name__}: {exc}")
        raise
    with _MATERIALIZED_LOCK:
        _MATERIALIZED.add(str(mk))
    if _evt is not None:
        _evt_emit("load.done", model_key=mk, engine=engine,
                  duration_ms=int((time.time() - t0) * 1000))


def _forget_materialized(model_key: str) -> None:
    """Drop a model's materialized flag — on evict/unload, so a stale True can
    never outlive the weights it described."""
    with _MATERIALIZED_LOCK:
        _MATERIALIZED.discard(str(model_key))


def _is_materialized(model_key: str) -> bool | None:
    """True / False / None(unknown) for "are this model's weights loaded".

    Two independent sources, both NON-FORCING (asking must never trigger the
    load we are asking about), consulted in this order:

      1. the llama runner cache — an in-process GGUF handle exists iff ``llm``
         is set on the cached runner. AUTHORITATIVE for the gguf path (most of
         the fleet) and, crucially, LIVE: ``dispatch.evict`` cascades into
         ``evict_llama_runner``, so this source self-corrects on unload where a
         remembered flag would go stale and re-assert residency for weights
         that are gone.
      2. ``_MATERIALIZED`` — models whose ``ensure_loaded()`` we watched
         succeed. The fallback for non-llama runners (transformers/DeepCoder),
         which expose no comparable cache to interrogate.

    Returns None only when neither source can speak, so an older/odd path
    degrades to "unknown" rather than to a confident falsehood."""
    mk = str(model_key)
    try:
        from hugpy_engine.llama.runners.get import _LLAMA_INSTANCES, _LLAMA_LOCK
        with _LLAMA_LOCK:
            r = _LLAMA_INSTANCES.get(mk)
        if r is not None:
            # An HTTP/slot-backed runner holds no weights in THIS process; its
            # residency is the slot child's and is reported by the slot row.
            if getattr(r, "base_url", None):
                return None
            return getattr(r, "llm", None) is not None
    except Exception:  # noqa: BLE001
        pass
    with _MATERIALIZED_LOCK:
        if mk in _MATERIALIZED:
            return True
    # Nothing remembered a materialization. That is only EVIDENCE of a hollow
    # runner for the llama/gguf family, where the cache above is exhaustive: a
    # loaded GGUF is in _LLAMA_INSTANCES, full stop, so absence there means
    # runner_for() built the lazy shell and .runner was never touched — the
    # incident shape exactly.
    #
    # For every other family we must answer UNKNOWN, not False. A transformers
    # runner loaded through a plain .run() (no ensure_loaded) leaves no trace in
    # either source while being genuinely, measurably resident — and the console
    # treats materialized=False as outranking measurement. Claiming False there
    # would hide a hot model, which is the same class of lie as the one this is
    # fixing, pointed the other way.
    try:
        from hugpy_engine.dispatch.dispatch import _INSTANCES, _INSTANCES_LOCK
        with _INSTANCES_LOCK:
            inst = next((v for k, v in _INSTANCES.items() if k[0] == mk), None)
        if inst is not None:
            from hugpy_engine.llama.runners.src.base_runner import LlamaCppBaseRunner
            is_llama = isinstance(inst, LlamaCppBaseRunner) or hasattr(
                type(inst), "runner")
            return False if is_llama else None
    except Exception:  # noqa: BLE001
        pass
    return None


def _loaded_detail() -> dict:
    # Size EVERY serving row: start with on-disk dir bytes for all frameworks
    # (transformers/diffusers/llama), then let the GGUF runner detail overlay
    # its exact file bytes + layer/GPU split on top. Without the disk base,
    # non-GGUF rows had no size at all.
    detail: dict = {}
    try:
        from hugpy_engine.dispatch.dispatch import loaded_disk_detail
        detail.update(loaded_disk_detail())
    except Exception:
        pass
    try:
        from hugpy_engine.llama.runners.get import loaded_runner_detail
        for key, facts in loaded_runner_detail().items():
            detail.setdefault(key, {}).update(facts)
    except Exception:
        pass
    return detail


# A resident counts as "serving" if it answered within this window; older ones
# read as idle-resident. Wide enough to keep a warm model lit between bursts,
# short enough that yesterday's test churn shows idle.
_SERVING_WINDOW_S = 180.0


def _model_framework(mk: str) -> "str | None":
    """Framework for a model_key ('gguf'/'transformers'/'comfy'/…) or None.

    Module-level so residency reporting can cheaply tell comfy rows apart from
    real in-pool residents: a comfy checkpoint is served by the EXTERNAL,
    adopted ComfyUI process (out-of-pool) — the worker holds only a thin client
    runner with NO weights, so it must never be counted as an in-RAM resident."""
    try:
        from hugpy_fleet.worker.imports import get_model_config
        return getattr(get_model_config(mk), "framework", None)
    except Exception:  # noqa: BLE001 — unknown row: treat as non-comfy
        return None


# ── worker-local slot pool (CON-02) ─────────────────────────────────────────
# With SLOT_COUNT > 0 the agent supervises N slot_agent children — the same
# slot machinery central runs, but agent-managed (rootless, no systemd units
# to install). Slot children run llama_cpp.server (no C++ llama-server binary
# needed on workers), and get_llama_runner's slot-first path then serves this
# worker's requests from slots: resident, TTL'd, crash-ISOLATED (a load that
# aborts kills a child, not the agent — the failure mode that took the whole
# agent down on 2026-07-02).

def _slot_statuses() -> list | None:
    try:
        from hugpy_engine.serve.slots import SlotPool, _slot_count
        n = _slot_count()
        if n <= 0:
            # Effective slot count 0 -> report NO slots as an explicit empty
            # list (not None), so central overwrites and CLEARS any stale
            # phantom rows a prior config left behind. Fixes the zero-slot box
            # (e.g. a transformers-only CPU worker) that advertised 2
            # unreachable seats it never actually ran.
            return []
        # Never report more rows than the effective slot count.
        return SlotPool().statuses()[:n]
    except Exception:
        return None


# ── REAL per-process GPU VRAM (nvidia-smi) ──────────────────────────────────
# Type/ngl-based inference was WRONG: a transformers/vision model loads onto
# CUDA but reports n_gpu_layers=null, so the console mislabeled it "host RAM —
# not in VRAM". Ground truth is nvidia-smi's PER-PROCESS accounting, joined with
# what THIS worker knows it launched (slot child PIDs) or holds (in-process
# torch models). Everything here degrades to null on a box with no GPU / no
# nvidia-smi, so such a worker behaves exactly as before.

_MIB = 1024 * 1024
# nvidia-smi is polled at most once per this window and shared across every
# allocation in a heartbeat — never spawned per model.
_GPU_PROC_TTL_S = 8.0
_GPU_PROC_CACHE: dict = {"at": 0.0, "value": {}}


def _gpu_process_vram() -> dict:
    """``{pid: {"name": str, "mib": int}}`` from nvidia-smi's per-process compute
    accounting. Cached ~heartbeat cadence so it runs ONCE per beat, not per model.

    Degrades to ``{}`` (→ callers keep today's behavior) when nvidia-smi is
    absent (no GPU / non-CUDA host), errors, or reports "[N/A]"/"[Not Supported]"
    for a row (no per-process accounting)."""
    now = time.time()
    if now - _GPU_PROC_CACHE["at"] < _GPU_PROC_TTL_S:
        return _GPU_PROC_CACHE["value"]
    out: dict = {}
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                pid_s, name, mem_s = parts[0], parts[1], parts[-1]
                if not pid_s.isdigit():
                    continue
                try:
                    mib = int(float(mem_s))     # "[N/A]"/"[Not Supported]" → skip row
                except ValueError:
                    continue
                out[int(pid_s)] = {"name": name, "mib": mib}
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        out = {}                                # no GPU / no nvidia-smi → today's behavior
    _GPU_PROC_CACHE.update(at=now, value=out)
    return out


def _comfy_process_vram(gpu_procs: "dict | None" = None) -> "int | None":
    """Real VRAM (bytes) of the adopted, EXTERNAL ComfyUI process — from
    nvidia-smi, never from on-disk checkpoint bytes (that all-checkpoints sizing
    is exactly the "37 serving / ~600 GB" bug the 0.1.137 guard fixed). Sums any
    compute proc whose name marks it ComfyUI. ``None`` when nvidia-smi reports no
    such proc (not running / on CPU / no per-proc accounting)."""
    procs = gpu_procs if gpu_procs is not None else _gpu_process_vram()
    mib = 0
    hit = False
    for info in procs.values():
        if "comfyui" in (info.get("name") or "").lower():
            mib += int(info.get("mib") or 0)
            hit = True
    return mib * _MIB if hit else None


def _inprocess_gpu_bytes() -> dict:
    """``{model_key: {"vram_bytes": int, "device": 'cuda'|'cpu'|None}}`` for every
    in-process torch model THIS worker holds — the piece that makes a CUDA-
    resident transformers/vision model stop reading as host RAM.

    The worker python is ONE nvidia-smi process (e.g. ae's 3622 MiB lump holding
    many in-process models at once); torch is the only tool that can split that
    lump per-model. For each model we sum its parameter+buffer bytes that live on
    a cuda device (deduped by storage pointer, so a module shared across two
    task-variants counts once). ``{}`` when torch is missing.

    Reads only ALREADY-materialized state — instance ``__dict__`` and the class-
    level pipeline caches — never a lazy property, so this telemetry pass can
    never trigger a model load."""
    try:
        import torch
    except Exception:
        return {}

    seen_ptrs: set = set()

    def _obj_bytes(obj) -> tuple:
        """(cuda_bytes, cpu_bytes, {gpu_index: cuda_bytes}) for a torch nn.Module
        or a diffusers pipeline (walked via its ``.components`` sub-models).
        Non-torch → (0, 0, {}). The per-device map (2026-09-25) is what lets the
        heartbeat report WHICH card an in-process model sits on."""
        cuda = cpu = 0
        dev: dict = {}
        comps = getattr(obj, "components", None)     # diffusers pipeline
        if isinstance(comps, dict):
            for c in comps.values():
                cc, pc, cd = _obj_bytes(c)
                cuda += cc
                cpu += pc
                for k, v in cd.items():
                    dev[k] = dev.get(k, 0) + v
            return cuda, cpu, dev
        if not isinstance(obj, torch.nn.Module):
            return 0, 0, dev
        try:
            tensors = list(obj.parameters()) + list(obj.buffers())
        except Exception:
            return 0, 0, dev
        for t in tensors:
            try:
                ptr = t.data_ptr()
            except Exception:
                continue
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            try:
                nbytes = t.numel() * t.element_size()
            except Exception:
                continue
            if getattr(t, "is_cuda", False):
                cuda += nbytes
                try:
                    gi = int(t.get_device())      # cuda ordinal (-1 for cpu)
                except Exception:
                    gi = None
                if gi is not None and gi >= 0:
                    dev[gi] = dev.get(gi, 0) + nbytes
            else:
                cpu += nbytes
        return cuda, cpu, dev

    objs_by_key: dict = {}

    def _add(mk, v) -> None:
        if isinstance(v, torch.nn.Module) or isinstance(
                getattr(v, "components", None), dict):
            objs_by_key.setdefault(mk, []).append(v)

    # 1. In-process runner wrappers (transformers/vision/etc.) — the model is an
    #    instance attribute (e.g. vision_coder.self.model, coder.self.model).
    try:
        from hugpy_engine.dispatch.dispatch import _INSTANCES, _INSTANCES_LOCK
        with _INSTANCES_LOCK:
            items = list(_INSTANCES.items())
        for key, runner in items:
            # The cache_key is usually (model_key, task), but some runners key on
            # longer tuples (vision: (key,min,max,dtype); shard: (key,"__vision__",()))
            # or a bare string. The old `for (mk,_task), runner in items` assumed
            # 2-tuples, so ONE non-2-tuple key raised and the outer except silently
            # zeroed EVERY model's VRAM. Extract model_key arity-agnostically and
            # isolate each runner so one bad entry can't abort the whole walk.
            try:
                mk = key[0] if isinstance(key, tuple) and key else key
                attrs = list(vars(runner).values())
            except Exception:
                continue                        # no __dict__ / odd key → skip this one
            for v in attrs:
                _add(mk, v)
                # transformers Pipelines hold the nn.Module at `.model`, not as a
                # direct attr — reach it so pipeline-wrapped models count too
                # (_add ignores non-modules; storage-ptr dedup avoids double count).
                inner = getattr(v, "model", None)
                if inner is not None and inner is not v:
                    _add(mk, inner)
    except Exception:
        pass

    # 2. Diffusers pipelines live in a CLASS-level singleton keyed by model_key,
    #    not on the runner instance — reach them there.
    try:
        from hugpy_media.imagegen import imagegen_runner as _ig
        for clsname in ("ImageGenRunner", "Img2ImgRunner"):
            cls = getattr(_ig, clsname, None)
            cache = getattr(cls, "_PIPELINES", None)
            if isinstance(cache, dict):
                for mk, pipe in list(cache.items()):
                    _add(mk, pipe)
    except Exception:
        pass

    # 3. Transformers causal-LMs (DeepCoderChatRunner — DAN-Qwen3, DeepCoder, etc.)
    #    DON'T hold their nn.Module on the runner in _INSTANCES: the runner keeps a
    #    cfg and reaches the model lazily via a `coder` property → a SEPARATE
    #    module-level `REGISTRY._instances` of DeepCoder objects. So vars(runner)
    #    never sees the weights, which is why a static in-process transformers model
    #    read device=None. Walk the already-BUILT instances directly (read
    #    `_instances`, never `REGISTRY.get()`, so telemetry can't trigger a load);
    #    each DeepCoder carries its model_key on `.cfg` and its weights on `.model`.
    try:
        from hugpy_engine.generate.coder import REGISTRY as _DC_REGISTRY
        insts = getattr(_DC_REGISTRY, "_instances", None)
        if isinstance(insts, dict):
            for dc in list(insts.values()):
                mk = getattr(getattr(dc, "cfg", None), "model_key", None)
                model = getattr(dc, "model", None)
                if mk and model is not None:
                    _add(mk, model)
    except Exception:
        pass

    # RECONCILIATION: sum(out[*].vram_bytes) is the worker python's model weights
    # on GPU. It runs a bit UNDER that process's nvidia-smi total
    # (_gpu_process_vram()[os.getpid()]) because the CUDA context (~tens of MiB)
    # plus activation/workspace/KV-cache scratch are NOT parameters. That residual
    # is real driver overhead — left as-is, never smeared onto a model, so a
    # model's vram_bytes stays its honest weight footprint.
    out: dict = {}
    for mk, objs in objs_by_key.items():
        cuda = cpu = 0
        dev: dict = {}
        for o in objs:
            cc, pc, cd = _obj_bytes(o)
            cuda += cc
            cpu += pc
            for k, v in cd.items():
                dev[k] = dev.get(k, 0) + v
        device = "cuda" if cuda > 0 else ("cpu" if cpu > 0 else None)
        # WHICH card: the cuda ordinal holding the most of this model's weights
        # (a pipeline is normally wholly on one card; max() is robust if a stray
        # buffer sits elsewhere). None for a CPU-only / unmeasurable model.
        gpu_index = max(dev, key=dev.get) if dev else None
        # cpu_bytes is the CPU-side analog: parameter+buffer bytes this model
        # holds on device 'cpu' — MEASURED torch allocation (not a file size),
        # so a ram allocation row can report real host-RAM occupancy for a
        # transformers/diffusers model that torch can introspect.
        row = {"vram_bytes": cuda, "device": device, "cpu_bytes": cpu}
        if gpu_index is not None:
            row["gpu_index"] = gpu_index
        out[mk] = row
    return out


# ── MEASURED host-RAM residency of file-backed model weights (/proc/self/smaps)
# The agent process mmaps GGUF/safetensors weights, so the model's true host-RAM
# occupancy right now is the sum of the Rss of ITS mappings — not the file's size
# on disk. On ae the declared file bytes read 77 GB against 26 GB physically
# used, because mmap'd pages are (a) shared and (b) only partly faulted in.
# Parsing smaps is not free (one line per mapping, thousands of lines), so it is
# read ONCE per heartbeat and shared by every ram row.
_SMAPS_TTL_S = 8.0
_SMAPS_CACHE: dict = {"at": 0.0, "value": {}}


def _parse_smaps_rss_by_path(text: str) -> dict:
    """``{pathname: rss_bytes}`` from /proc/<pid>/smaps text — the Rss of every
    FILE-BACKED mapping, summed per pathname (one file is typically mapped in
    several segments with different protections).

    Anonymous mappings (no pathname) and pseudo-paths (``[heap]``, ``[stack]``,
    ``/memfd:…``) are skipped: they belong to no model file. A ``(deleted)``
    suffix is stripped so a weight file replaced under a live mmap still groups
    with its path. Pure function of the text so it is directly testable."""
    out: dict = {}
    cur = None
    for line in text.splitlines():
        if not line:
            continue
        # Mapping header: "7f..-7f.. r--p 00000000 08:01 1808   /path/to/file"
        head = line.split(None, 5)
        if (len(head) >= 5 and "-" in head[0]
                and not line[0].isspace() and ":" not in head[0]):
            path = head[5].strip() if len(head) >= 6 else ""
            if path.endswith(" (deleted)"):
                path = path[:-len(" (deleted)")]
            cur = path if path.startswith("/") else None
            continue
        if cur is not None and line.startswith("Rss:"):
            try:
                out[cur] = out.get(cur, 0) + int(line.split()[1]) * 1024
            except (IndexError, ValueError):
                continue
    return out


def _smaps_rss_by_path() -> dict:
    """``_parse_smaps_rss_by_path`` over THIS process's smaps, cached at roughly
    heartbeat cadence. ``{}`` on any failure (non-Linux, /proc hiccup, kernel
    without smaps) — callers then OMIT the measured field rather than guess."""
    now = time.time()
    if now - _SMAPS_CACHE["at"] < _SMAPS_TTL_S:
        return _SMAPS_CACHE["value"]
    val: dict = {}
    try:
        with open("/proc/self/smaps") as fh:
            val = _parse_smaps_rss_by_path(fh.read())
    except Exception:  # noqa: BLE001 — never break the heartbeat on /proc
        val = {}
    _SMAPS_CACHE.update(at=now, value=val)
    return val


_MODEL_DIR_CACHE: dict = {}   # model_key -> realpath str | None (misses cached)


def _model_store_dir(model_key: str) -> "str | None":
    """The model's local store directory (realpath), resolved exactly the way the
    puller/loader do (``route_destination``) so the smaps join sees the same
    files the loader mmapped. Cached per key — one resolution per model, not per
    heartbeat. None when unresolvable."""
    if model_key in _MODEL_DIR_CACHE:
        return _MODEL_DIR_CACHE[model_key]
    path = None
    try:
        from hugpy_storage.model_paths import route_destination
        from hugpy_engine.config.main import get_model_config
        cfg = get_model_config(model_key, dict_return=True)
        p = route_destination(cfg)
        path = os.path.realpath(p) if p else None
    except Exception:  # noqa: BLE001 — unresolvable: no measurement, no guess
        path = None
    _MODEL_DIR_CACHE[model_key] = path
    return path


def _resident_bytes_under_dir(rss_by_path: dict, model_dir: str) -> "int | None":
    """Measured resident bytes of the mappings that belong to ``model_dir`` —
    the sum of Rss over every mapped file under that directory.

    ``None`` (not 0) when NOTHING under the dir is mapped: the model's weights
    are not file-backed in this process (or smaps was unreadable), so there is no
    measurement to report and the caller omits the field. 0 is never invented."""
    if not model_dir or not rss_by_path:
        return None
    prefix = model_dir.rstrip("/") + "/"
    total = 0
    hit = False
    for path, rss in rss_by_path.items():
        if path == model_dir or path.startswith(prefix):
            total += int(rss)
            hit = True
    return total if hit else None


def _vram_split_from_pidlog(pid_log: "dict | None") -> dict:
    """Split the SAME pid_registry snapshot central will store into the two
    VRAM figures the budget-bar spec needs (t13/t14):

      * ``vram_attributed_bytes`` — sum of the attributed MODEL rows (rows with
        a real ``model_key``): the VRAM hugpy's own served models hold. This is
        the spec's VRAM ``worker_usage``. cuda_context (the agent's own CUDA
        context) counts as worker infra too, so it's folded in — it IS the
        worker's usage, just not a served model. ComfyUI and genuinely-foreign/
        unattributed rows are EXTERNAL, excluded here.
      * ``vram_unattributed_bytes`` — sum of the ``unattributed`` squatter rows
        (mib→bytes): genuinely foreign GPU use the console surfaces as overhead.

    Sampled from the pid_log produced the SAME beat as detect_gpus(), so the
    attribution and the driver totals are one snapshot. Degrades to
    ``{"vram_attributed_bytes": None, "vram_unattributed_bytes": None}`` when
    there's no pid_log (no GPU / older agent) — never fabricated."""
    if not isinstance(pid_log, dict):
        return {"vram_attributed_bytes": None, "vram_unattributed_bytes": None}
    attributed = 0
    for row in (pid_log.get("models") or []):
        if not isinstance(row, dict):
            continue
        mode = row.get("host_mode")
        # comfy rows are EXTERNAL (adopted, out-of-pool) — not worker usage.
        if mode == "comfy":
            continue
        vb = row.get("vram_bytes")
        if vb is None:
            continue
        # A worker row counts when it's a served model (has model_key) OR the
        # worker's own cuda_context infra lump. Foreign rows have neither.
        if row.get("model_key") or mode == "cuda_context":
            try:
                attributed += int(vb)
            except (TypeError, ValueError):
                continue
    _MIB_LOCAL = 1024 * 1024
    unattributed = 0
    for row in (pid_log.get("unattributed") or []):
        if not isinstance(row, dict):
            continue
        try:
            unattributed += int(row.get("mib") or 0) * _MIB_LOCAL
        except (TypeError, ValueError):
            continue
    return {"vram_attributed_bytes": attributed,
            "vram_unattributed_bytes": unattributed}


def _slot_total_layers_fallback(model_key: str) -> "int | None":
    """Total GGUF layer count for a SLOT-seated model whose slot build predates
    the ``total_layers`` status field (an adopted stale slot child) — resolved
    via the same geometry reader the offload math uses, and CACHED per model so
    the heartbeat never re-parses a GGUF header every beat. None (also cached)
    for non-GGUF / unresolvable — the allocation row then omits the field."""
    if model_key in _TOTAL_LAYERS_CACHE:
        return _TOTAL_LAYERS_CACHE[model_key]
    tl = None
    try:
        _, tl = _served_gguf_geometry(model_key)
    except Exception:  # noqa: BLE001 — best-effort metadata, never break a beat
        tl = None
    _TOTAL_LAYERS_CACHE[model_key] = tl
    return tl


_TOTAL_LAYERS_CACHE: dict = {}   # model_key -> int | None (misses cached too)


def _inferred_device(n_gpu_layers, gpu_pct=None) -> "str | None":
    """Device inferred from DECLARED placement when nothing measured it.

    The residency doctrine is "measured, never inferred from membership" — so
    this is NOT a substitute for measurement, and every row that uses it is
    stamped ``device_source: "inferred"`` so the console can render it as the
    weaker claim it is. It exists because a box whose nvidia-smi is broken
    (computron: "Failed to initialize NVML: Driver/library version mismatch")
    reported ``device: null`` for a live GPU seat — an omission that reads as
    "missing", which is LESS honest than a labeled inference.

    This is placement the worker ITSELF declared when it launched the child /
    loaded the model, not a guess from cache membership:
      n_gpu_layers > 0 or -1 (all layers)  → 'cuda'
      n_gpu_layers == 0                    → 'cpu'
      gpu_pct > 0 / == 0                   → same, for engines with no ngl
      anything unknown                     → None (still correct to say nothing)
    """
    try:
        ngl = int(n_gpu_layers) if n_gpu_layers is not None else None
    except (TypeError, ValueError):
        ngl = None
    if ngl is not None:
        if ngl == 0:
            return "cpu"
        return "cuda"          # >0 partial offload, -1 all layers
    try:
        pct = float(gpu_pct) if gpu_pct is not None else None
    except (TypeError, ValueError):
        pct = None
    if pct is not None:
        return "cuda" if pct > 0 else "cpu"
    return None                # no basis — null stays the honest answer


def _slot_last_used(s: dict) -> "float | None":
    """A slot's last_used as epoch seconds, or None when never used / unreported.

    The slot seeds ``last_used = 0.0`` at construction and stamps it on each
    request, so 0.0 means "seated but never answered" — that is None on the
    wire, matching the ram rows' ``last_used`` (None = never)."""
    lu = (s or {}).get("last_used")
    try:
        lu = float(lu) if lu is not None else None
    except (TypeError, ValueError):
        return None
    return lu if lu else None          # 0.0 / 0 → never used


def _slot_serving(s: dict, now: "float | None" = None) -> bool:
    """Whether a SLOT allocation is serving, on the SAME terms as a ram row.

    A slot with in-flight work (``busy``) is answering right now — measured
    directly, no window needed. Otherwise it counts as serving if it answered
    within ``_SERVING_WINDOW_S``, so a slot that has sat cold since yesterday's
    test churn reads idle exactly like an idle in-process resident does."""
    s = s or {}
    if s.get("busy"):
        return True
    lu = _slot_last_used(s)
    if lu is None:
        return False
    return (time.time() if now is None else now) - lu < _SERVING_WINDOW_S


def _allocations(slot_statuses: "list | None" = None) -> list:
    """Unified, engine-agnostic view of every resource allocation on this
    worker — one entry per SLOT-seated model and one per in-RAM (in-process)
    resident model. A slot is a resource allocation to a model regardless of
    engine, so GGUF slot occupants and transformers models held in the agent's
    OWN process are reported side by side. This is a NEW field parallel to (not
    a replacement for) loaded_models/slots, so old central/UI keep working.

    Each entry carries the REAL GPU residency the console consumes:
      ``vram_bytes`` (int bytes | null) — actual VRAM the model occupies now.
      ``device``     ('cuda' | 'cpu' | null) — the device the weights live on.
    SLOT rows join nvidia-smi against the slot's child_pid (exact per-model);
    RAM rows split the worker python's nvidia-smi lump per-model via torch. VRAM
    is NEVER written into model_bytes/weight_bytes (those stay on-disk *size*).

    Measured-vs-inferred (2026-07-25). When neither read can see the model —
    computron's nvidia-smi fails outright ("Failed to initialize NVML"), and
    torch cannot introspect an in-process GGUF ``Llama`` handle — ``device`` used
    to be null for a demonstrably live GPU seat, which the console rendered as
    "missing". Such a row now falls back to the placement the worker ITSELF
    declared at launch (n_gpu_layers / gpu_pct) and carries:
      ``device_source`` ('measured' | 'inferred', OMITTED when there is no
        device claim at all) — so a consumer can hold measured and inferred to
        different standards and NEVER launder a guess into a measurement. The
        residency doctrine still stands: only a 'measured' device (plus
        vram_bytes / rss_bytes) is evidence of residency.
    ``vram_bytes`` is never inferred — placement is knowable without nvidia-smi,
    a byte count is not, so it stays null rather than becoming a fiction.
    ``device`` is null (and device_source absent) when there is genuinely no
    basis, exactly as before.

    Measured host-RAM occupancy on RAM rows (2026-07-28 operator ruling:
    worker-side MEASUREMENTS are the truth for residency/occupancy). RAM rows
    carried only ``model_bytes``/``weight_bytes`` — ON-DISK file sizes, which on
    ae summed to 77 GB of "resident" models against 26 GB physically used,
    forcing the console to widen its RAM denominator. They now also carry, OMIT-
    WHEN-UNSET:
      ``ram_resident_bytes`` (int) — bytes of host RAM this model occupies NOW.
      ``ram_resident_source`` ('smaps' | 'torch') — how it was measured:
        'smaps' = Rss of this process's mappings of the model's own files (the
        mmap'd GGUF/safetensors actually faulted in); 'torch' = parameter+buffer
        bytes on device 'cpu' for an introspectable torch model.
    With no measurement BOTH keys are absent and ``model_bytes`` remains the
    labeled upper-bound fallback — a file size is never laundered into
    ``ram_resident_bytes``. ``model_bytes``/``weight_bytes`` keep their on-disk
    meaning verbatim for every existing consumer.

    ``slot_statuses`` may be passed in to avoid a second slot round-trip when
    the heartbeat already computed it."""
    out: list = []
    seen: set = set()
    gpu_procs = _gpu_process_vram()            # {} when no GPU / no nvidia-smi
    now = time.time()
    rows = slot_statuses if slot_statuses is not None else _slot_statuses()
    for s in (rows or []):
        mk = (s or {}).get("model_key")
        if not mk:
            continue                       # empty seats aren't allocations
        seen.add(mk)
        # Join nvidia-smi on the slot's llama-server CHILD pid (the process that
        # actually holds the weights). Absent child_pid (old slot build) or empty
        # gpu_procs (no nvidia-smi) → fall back to the DECLARED placement below,
        # labeled as inferred — never a silent null for a live GPU seat.
        vram_bytes = None
        device = None
        device_source = None
        if gpu_procs:
            cp = s.get("child_pid")
            info = gpu_procs.get(cp) if cp is not None else None
            if info is not None:
                vram_bytes = int(info["mib"]) * _MIB
                device = "cuda" if vram_bytes > 0 else "cpu"
                device_source = "measured"
            elif cp is not None:
                # Child is alive but not a GPU compute app → CPU-resident (ngl=0).
                vram_bytes, device = 0, "cpu"
                device_source = "measured"
        if device is None:
            # No per-process accounting on this box (broken/absent nvidia-smi).
            # Say what the worker KNOWS it launched, stamped inferred. vram_bytes
            # stays NULL — placement is knowable, byte count is not, and a made-up
            # number would be the dishonest half of this.
            device = _inferred_device(s.get("n_gpu_layers"))
            if device is not None:
                device_source = "inferred"
        row = {
            "kind": "slot", "model_key": mk,
            "slot_id": s.get("slot_id"), "healthy": s.get("healthy"),
            "busy": s.get("busy"), "endpoint": s.get("endpoint"),
            "rss_bytes": s.get("rss_bytes"),
            "n_gpu_layers": s.get("n_gpu_layers"), "ctx": s.get("ctx"),
            "vram_bytes": vram_bytes, "device": device,
            # Allocation provenance (2026-09-23), None from an older slot:
            # what the seat was loaded FOR and by whom, what it actually got,
            # and why it replaced a resident of the same model.
            "alloc_requested": s.get("alloc_requested"),
            "alloc_source": s.get("alloc_source"),
            "alloc_effective": s.get("alloc_effective"),
            "reload_reason": s.get("reload_reason"),
            # Idle-vs-serving for SLOT rows, same semantics as the ram rows
            # below (_SERVING_WINDOW_S against the worker's OWN clock). Before
            # this, slot rows never set `serving` at all, so a slot that had
            # just answered an inference reported serving:null — the operator's
            # "shows missing for nearly everything, even models it's serving".
            # A busy slot is answering RIGHT NOW: serving is true by observation,
            # no clock involved. The slot reports last_used (epoch seconds, 0.0 =
            # never used since seat).
            "last_used": _slot_last_used(s),
            "serving": _slot_serving(s, now),
        }
        if device_source is not None:
            # omit-when-unset: an old central/UI never sees the key, and a row
            # with no device basis at all carries no provenance to mislabel.
            row["device_source"] = device_source
        # WHICH card this seat is on (2026-09-25): the slot's OWN pin is
        # authoritative — ``gpu`` is exactly what CUDA_VISIBLE_DEVICES was set to
        # (the global card index central chose); a multi-GPU tensor-split reports
        # its ``main_gpu`` + the split. Omit-when-unset so an older slot / an
        # unpinned auto seat carries no per-device claim. Lets central place a
        # NEXT model without guessing which card a resident already holds.
        _gi = s.get("gpu")
        if _gi in (None, ""):
            _gi = s.get("main_gpu")
        if _gi not in (None, ""):
            try:
                row["gpu_index"] = int(_gi)
            except (TypeError, ValueError):
                pass
        if s.get("tensor_split") is not None:
            row["tensor_split"] = s.get("tensor_split")
        # Honest allocation accuracy (2026-07-22), omit-when-unset so the wire
        # shape is unchanged for old slots/central:
        #  * total_layers — GGUF block_count so "17/48" renders instead of
        #    "17/undefined". The slot reports it since this build; for an
        #    ADOPTED older slot child fall back to the agent's own GGUF-header
        #    read (cached — one header parse per model, not per beat).
        #  * rss_anon_bytes / rss_file_bytes — VmRSS counts the mmap'd GGUF's
        #    file-backed pages (reclaimable cache) as resident, overstating true
        #    pinned RAM ~28x on ae; RssAnon is the honest figure. Slot-reported,
        #    else read from /proc/<child_pid>/status here (same box).
        tl = s.get("total_layers")
        if tl is None:
            tl = _slot_total_layers_fallback(mk)
        if tl is not None:
            row["total_layers"] = tl
        if s.get("rss_anon_bytes") is not None:
            for k in ("rss_anon_bytes", "rss_file_bytes", "rss_shmem_bytes"):
                if s.get(k) is not None:
                    row[k] = s[k]
        elif s.get("child_pid") is not None:
            try:
                from hugpy_engine.serve.slot_agent import _proc_rss_detail
                row.update(_proc_rss_detail(s["child_pid"]))
            except Exception:  # noqa: BLE001 — never break the heartbeat on /proc
                pass
        out.append(row)
    detail = _loaded_detail()
    inproc = _inprocess_gpu_bytes()            # {} when torch missing
    try:
        from hugpy_engine.dispatch.dispatch import last_used_snapshot
        last_used = last_used_snapshot() or {}
    except Exception:
        last_used = {}
    for mk in loaded_model_keys():
        if mk in seen:
            continue                       # already counted as a slot allocation
        if _model_framework(mk) == "comfy":
            # ComfyUI checkpoints are served by the EXTERNAL, adopted ComfyUI
            # process (out-of-pool): the worker instantiates only a thin client
            # runner that holds NO weights. Counting them as in-RAM residents,
            # sized by on-disk dir bytes, is exactly what made ae read "37
            # serving / ~600 GB". They surface via the `comfy` heartbeat block,
            # not as pool allocations.
            continue
        d = detail.get(mk) or {}
        ip = inproc.get(mk) or {}
        # REAL GPU residency from torch introspection: a cuda-resident
        # transformers/vision model reports vram_bytes>0 + device='cuda' and
        # stops reading as host RAM. torch can't see an in-process GGUF Llama
        # handle (not a torch module) — that used to leave device null even
        # though the load DECLARED its placement in n_gpu_layers/gpu_pct. Fall
        # back to that declared placement, stamped inferred; vram_bytes stays
        # null because nothing measured the bytes.
        ram_device = ip.get("device")
        ram_device_source = "measured" if ram_device is not None else None
        if ram_device is None:
            ram_device = _inferred_device(d.get("n_gpu_layers"), d.get("gpu_pct"))
            if ram_device is not None:
                ram_device_source = "inferred"
        ram_row = {
            "kind": "ram", "model_key": mk,
            "model_bytes": d.get("model_bytes"),
            "weight_bytes": d.get("weight_bytes"),
            "gpu_pct": d.get("gpu_pct"),
            "n_gpu_layers": d.get("n_gpu_layers"),
            "total_layers": d.get("total_layers"),
            "vram_bytes": ip.get("vram_bytes"),
            "device": ram_device,
            # Idle-vs-serving: the console shows 🔥 only for genuinely-active
            # residents (recently-used in-process), the rest as idle-resident —
            # so a pool of test-churn leftovers never reads as "all serving".
            # Computed worker-side (its own clock vs last_used) to dodge any
            # client/central clock skew. last_used is epoch seconds (None=never).
            "last_used": last_used.get(mk),
            "serving": (last_used.get(mk) is not None
                        and (now - last_used[mk]) < _SERVING_WINDOW_S),
        }
        if ram_device_source is not None:
            ram_row["device_source"] = ram_device_source   # omit-when-unset
        # WHICH card an in-process cuda-resident model sits on (2026-09-25),
        # MEASURED by torch (the tensors' device ordinal). Omit-when-unset: a
        # CPU-resident model / a GGUF Llama handle torch can't introspect carries
        # no per-device claim.
        if ip.get("gpu_index") is not None:
            ram_row["gpu_index"] = ip.get("gpu_index")
        # MEASURED host-RAM occupancy (2026-07-28 ruling). model_bytes above is
        # the model's ON-DISK size — an upper bound, never occupancy. Two real
        # measurements, in priority order; when neither exists the keys are
        # OMITTED entirely and model_bytes stays the labeled fallback:
        #   'smaps' — page residency of this process's mappings of the model's
        #             own files (mmap'd GGUF/safetensors). What is in RAM NOW.
        #   'torch' — parameter+buffer bytes the model holds on device 'cpu',
        #             for a torch model whose weights aren't file-backed.
        # A file size is NEVER promoted into ram_resident_bytes. (ram_-prefixed: the
        # storage survey already owns a DISK-meaning resident_bytes — one name
        # must not carry two units.)
        try:
            _rb = None
            _rsrc = None
            _mdir = _model_store_dir(mk)
            if _mdir:
                _rb = _resident_bytes_under_dir(_smaps_rss_by_path(), _mdir)
                if _rb is not None:
                    _rsrc = "smaps"
            if _rb is None:
                _cpu = ip.get("cpu_bytes")
                if _cpu:
                    _rb, _rsrc = int(_cpu), "torch"
            if _rb is not None:
                ram_row["ram_resident_bytes"] = int(_rb)
                ram_row["ram_resident_source"] = _rsrc
        except Exception:  # noqa: BLE001 — never break the heartbeat
            pass
        # DID THE WEIGHTS EVER LOAD? `serving` above is RECENCY (touched within
        # _SERVING_WINDOW_S) and dispatch stamps last_used before the load is
        # attempted, so on its own it cannot distinguish a hot model from a
        # hollow runner whose provisioning died — which is precisely how a
        # never-loaded model rendered as "🔥 serving · resident" on 2026-07-28.
        # OMITTED when unknown: absent must keep meaning exactly what it meant
        # to every existing consumer, and a guess here is the bug, not the fix.
        _mat = _is_materialized(mk)
        if _mat is not None:
            ram_row["materialized"] = bool(_mat)
        out.append(ram_row)
    return out


# Live slot children, module-global so the self-update path can terminate
# them BEFORE re-exec: an orphaned slot survives the update and keeps serving
# OLD code forever (the adoption probe can't tell versions apart) — the
# "adopted stale slot" failure of 2026-07-02.
_SLOT_PROCS: dict[int, subprocess.Popen] = {}


def _kill_slots() -> None:
    for i, p in list(_SLOT_PROCS.items()):
        try:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=10)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        _SLOT_PROCS.pop(i, None)


def _supervise_slots() -> None:
    """Spawn and keep alive SLOT_COUNT slot_agent children (no-op when 0)."""
    from hugpy_engine.serve.slots import slots_enabled, _slot_count

    if not slots_enabled():
        return
    # Fleet-owned child entry: installs the storage providers (budget gate,
    # telemetry) in the child, then runs hugpy_engine.serve.slot_agent.
    module = "hugpy_fleet.worker.slot_child"
    n = _slot_count()
    procs = _SLOT_PROCS

    def _slot_answering(i: int) -> bool:
        """A slot from a PREVIOUS agent process may still own the port (agent
        re-exec orphans its children) — adopt it instead of bind-fighting."""
        try:
            from hugpy_engine.serve.slots import slot_urls
            import urllib.request as _url
            with _url.urlopen(slot_urls()[i - 1] + "/health", timeout=2) as r:
                return r.getcode() == 200
        except Exception:
            return False

    def _spawn(i: int) -> None:
        if _slot_answering(i):
            logger.info("slot supervisor: slot %d already serving (adopted)", i)
            procs.pop(i, None)
            return
        env = dict(os.environ)
        env["SLOT_ID"] = str(i)
        procs[i] = subprocess.Popen([sys.executable, "-m", module], env=env)
        logger.info("slot supervisor: started slot %d (pid %s)", i, procs[i].pid)

    def _loop() -> None:
        for i in range(1, n + 1):
            _spawn(i)
        while True:
            time.sleep(20)
            for i in range(1, n + 1):
                p = procs.get(i)
                if p is not None and p.poll() is None:
                    continue                      # our child, alive
                if _slot_answering(i):
                    continue                      # adopted orphan, alive
                if p is not None:
                    logger.warning("slot %d died (rc=%s) — respawning", i, p.returncode)
                _spawn(i)

    threading.Thread(target=_loop, daemon=True, name="slot-supervisor").start()
    logger.info("slot supervisor: managing %d slot(s) via llama_cpp.server children", n)


# ═══════════════════════════════════════════════════════════════════════════
# Restart mechanism (2026-07-12 incident class — CODE_GAPS "2026-07-12" item 3)
# ═══════════════════════════════════════════════════════════════════════════
# Two real incidents (computron restart-loop 160→219; op's 403 dueling-worker
# saga) traced to os.execv under systemd: execv KEEPS this PID but the way the
# agent re-exec'd left systemd believing the service died, so Restart= respawned
# a FRESH process that collided with the still-listening old image on :9100
# ("Address already in use") and restart-looped while the orphan kept heart-
# beating. The fix: UNDER SYSTEMD, never execv — release resources cleanly and
# EXIT with a distinct code so systemd's Restart= respawns exactly ONE properly-
# tracked process. STANDALONE (no systemd), execv in place is still correct and
# is kept as-is.
#
# Exit-code convention: a wanted restart exits _RESTART_EXIT_CODE (a distinct
# NON-ZERO). Non-zero matters because the canonical unit is `Restart=on-failure`
# (install.py) — a zero exit would NOT respawn there; a non-zero one does, and it
# also respawns under the field boxes' hand-rolled `Restart=always`. It is
# deliberately DIFFERENT from _terminal_exit's exit 0 (a 401/403 eviction that
# must STAY stopped under on-failure).
_RESTART_EXIT_CODE = 42
# Bound on how long a restart waits for in-flight generations to drain before it
# stops honoring them and exits anyway (never hangs forever). Read defensively so
# a malformed env value can never break the agent import.
try:
    _RESTART_DRAIN_TIMEOUT_S = max(
        0.0, float(os.environ.get("HUGPY_WORKER_RESTART_DRAIN_S", "30")))
except (TypeError, ValueError):
    _RESTART_DRAIN_TIMEOUT_S = 30.0
# Set the instant a restart is requested, so background loops (heartbeat self-
# update, reconcile, provision kicks) stop scheduling NEW work into a process
# that's about to exit — belt-and-suspenders against the "cannot schedule new
# futures" spam (os._exit already skips the atexit teardown that raises it).
_RESTART_EVENT = threading.Event()
# Long-lived executors (e.g. provision's parallel-transfer pool) register here so
# a restart can shut them down first. WeakSet: a finished pool drops out on GC.
_ACTIVE_EXECUTORS: "weakref.WeakSet" = weakref.WeakSet()


def restart_requested() -> bool:
    """True once a restart is underway — background loops check this to stop
    launching new transfers/updates into a process about to exit."""
    return _RESTART_EVENT.is_set()


def register_executor(ex) -> None:
    """Register a long-lived executor so the restart path shuts it down first.
    Best-effort/defensive: a bad object is simply ignored (never breaks a pull)."""
    try:
        _ACTIVE_EXECUTORS.add(ex)
    except Exception:  # noqa: BLE001 — registration must never break the caller
        pass


def _parent_is_systemd() -> bool:
    """True when this process's PARENT is the systemd manager — i.e. systemd
    fork()+exec()'d us directly, so we are a service's MainPID. PID 1 for a
    system unit; the `systemd --user` process for a user unit (computron/op run
    user units). Reading /proc/<ppid>/comm is Linux-only and best-effort."""
    ppid = os.getppid()
    if ppid == 1:
        return True
    try:
        with open(f"/proc/{ppid}/comm", "r", encoding="utf-8") as fh:
            return fh.read().strip() == "systemd"
    except OSError:
        return False


def _under_systemd() -> bool:
    """True iff THIS process is the MainPID of a systemd service — i.e. exiting
    will make systemd's Restart= respawn a fresh, cgroup-tracked process (so the
    restart path must EXIT, not execv).

    Why not just INVOCATION_ID / NOTIFY_SOCKET (the usual signals): both env vars
    AND the `.service` cgroup are INHERITED by every descendant of a systemd
    service. A worker launched inside another service's tree — a test under
    station-keeper.service, a shell under a login scope — would falsely read
    "systemd" and os._exit() out from under itself. So the signal is confirmed by
    the PARENT: systemd launches a service's MainPID directly, so our parent is
    the manager; a descendant's parent is a shell / the ancestor daemon instead.
    This correctly reads True on the already-deployed field units (no unit-file
    change needed) and False for tests/standalone runs.

    Explicit override: HUGPY_WORKER_SYSTEMD=1/0 forces the decision — the
    canonical unit MAY set =1 to be unambiguous; tests set 0/1 to pin a branch.
    """
    forced = os.environ.get("HUGPY_WORKER_SYSTEMD")
    if forced is not None and forced.strip() != "":
        return forced.strip().lower() in ("1", "true", "yes", "on")
    if not (os.environ.get("INVOCATION_ID") or os.environ.get("NOTIFY_SOCKET")):
        return False
    return _parent_is_systemd()


def _drain_generations(timeout_s: float) -> float:
    """Bounded wait for in-flight in-process generations to finish before a
    restart. Polls the gen-gate's TOTAL active permits; returns the seconds
    waited once they hit 0 or ``timeout_s`` elapses — never hangs, and never
    interrupts a native call (we wait for it to release the gate, up to the
    bound, then proceed). Semantics: honor active generations for up to
    ``timeout_s`` (default 30s), then exit regardless (systemd respawns; a client
    mid-stream sees the connection drop, exactly as any restart)."""
    start = time.monotonic()
    deadline = start + max(0.0, timeout_s)
    while True:
        try:
            active = gen_gate.total_in_flight()
        except Exception:  # noqa: BLE001 — can't measure -> don't block the restart
            active = 0
        if active <= 0 or time.monotonic() >= deadline:
            if active > 0:
                logger.warning("restart drain: %d generation(s) still in flight "
                               "after %.1fs — exiting anyway", active, timeout_s)
            return round(time.monotonic() - start, 3)
        time.sleep(0.2)


def _shutdown_executors() -> None:
    """Shut down registered long-lived executors BEFORE exit, so a still-running
    transfer/reconcile thread can't race into 'cannot schedule new futures'.
    Bounded (wait=False) and best-effort; cancels queued futures where the
    runtime supports it (py>=3.9)."""
    for ex in list(_ACTIVE_EXECUTORS):
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except TypeError:            # cancel_futures added in 3.9
            try:
                ex.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001 — one bad executor must not block the rest
            pass


def _close_http_server(state) -> bool:
    """Release the listening socket (:9100) so the respawned process can bind
    without an 'Address already in use' collision. Returns True if a server
    handle was closed. Best-effort: server_close() only closes the listening
    socket fd (safe whether or not serve_forever is running) — we do NOT call
    shutdown() here, which would block forever if serve_forever never started
    (registration-time self-update, tests). os._exit frees the fd regardless;
    this makes the release explicit and testable."""
    srv = getattr(state, "http_server", None)
    if srv is None:
        return False
    try:
        srv.server_close()
        return True
    except Exception:  # noqa: BLE001
        return False


def _prepare_restart(state, *, reason: str, mode: str,
                     kill_slots: bool, drain_timeout_s: "float | None" = None) -> dict:
    """Perform the clean-shutdown steps for a restart and RETURN the plan.

    This does the WORK (flag, drain, executor shutdown, slot teardown, socket
    release) but never exits/execs — the seam (``_restart``) applies the plan's
    mode. Split out so the shutdown sequence is unit-testable without terminating
    the test process.

    ``mode``: 'exit'  — systemd: caller os._exit(plan['exit_code']); Restart=
                        respawns a fresh, cgroup-tracked process.
              'execv' — standalone: caller execs in place (image replaced).
    """
    if drain_timeout_s is None:
        drain_timeout_s = _RESTART_DRAIN_TIMEOUT_S
    _RESTART_EVENT.set()                                   # 1. stop new work
    plan: dict = {"reason": reason, "mode": mode,
                  "exit_code": _RESTART_EXIT_CODE if mode == "exit" else None,
                  "steps": ["shutdown_flag"]}
    plan["drained_wait_s"] = _drain_generations(drain_timeout_s)   # 2. drain
    plan["steps"].append("drained")
    _shutdown_executors()                                          # 3. executors
    plan["steps"].append("executors")
    # 4. Slot children. Under 'exit' they must go: systemd's default
    # KillMode=control-group tears down the whole cgroup on respawn anyway, so a
    # clean terminate here beats an abrupt SIGKILL, and the fresh agent respawns
    # them. Under 'execv' we only kill when asked (self-update: an orphaned slot
    # would keep serving OLD code) — a plain re-exec ADOPTS live slots to avoid a
    # blip, exactly as today.
    if mode == "exit" or kill_slots:
        _kill_slots()
        plan["steps"].append("slots")
    # 5. Listening socket. Only for 'exit' (execv relies on CLOEXEC to drop it,
    # then re-binds fresh — the standalone path kept as today).
    if mode == "exit":
        plan["socket_closed"] = _close_http_server(state)
        plan["steps"].append("socket")
    return plan


def _restart(state, *, reason: str, reexec_fn, kill_slots: bool = False) -> None:
    """Apply a restart: clean shutdown, then EXIT (systemd) or execv (standalone).

    ``reexec_fn`` is resolved by the caller at SCHEDULE time (see
    ``_schedule_restart``) so a monkeypatched ``procutil.reexec`` is honored and a
    late-firing timer can never call the real ``os.execv`` after a test restored
    it. Under systemd this arg is unused (we os._exit instead)."""
    mode = "exit" if _under_systemd() else "execv"
    plan = _prepare_restart(state, reason=reason, mode=mode, kill_slots=kill_slots)
    if mode == "exit":
        logger.info("restart(%s): clean shutdown done %s — exiting %d for systemd "
                    "respawn", reason, plan["steps"], plan["exit_code"])
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        os._exit(plan["exit_code"])
    logger.info("restart(%s): standalone re-exec in place %s", reason, plan["steps"])
    # Log the EXACT exec target before the handoff, and survive a raising re-exec.
    # os.execv replaces the image and never returns on success; if control comes
    # back here or it raises, the swap did NOT happen. Left unguarded, that
    # exception bubbles into the heartbeat loop's generic "heartbeat failed"
    # swallow, where a still-OLD image keeps beating — the silent half of the
    # 2026-07-20 ae cosmetic-update path. Log LOUDLY and return instead: the
    # heartbeat now reports the honest (stale) running version, so central's
    # version_ok stays FALSE — a visible, diagnosable skew, not a green lie.
    # (SystemExit from the Windows spawn+exit path is BaseException, so it is not
    # caught here and still propagates.)
    try:
        from hugpy_platform.procutil import _module_argv
        _target_argv = _module_argv() or [sys.executable, *sys.argv]
    except Exception:  # noqa: BLE001 — argv preview is best-effort logging only
        _target_argv = [sys.executable, *sys.argv]
    logger.info("restart(%s): exec target argv=%r (running image %s)",
                reason, _target_argv, _RUNNING_IMAGE_VERSION)
    try:
        reexec_fn()
    except Exception as exc:  # noqa: BLE001 — must not bubble into a silent swallow
        logger.error(
            "restart(%s): RE-EXEC FAILED (%s: %s) — image NOT replaced; staying on "
            "the OLD running version %s. Heartbeat reports the honest (stale) "
            "version, so central shows a version skew, not cosmetic convergence. "
            "An explicit /ops/restart or a real unit restart is required to "
            "converge. exec target was argv=%r.",
            reason, type(exc).__name__, exc, _RUNNING_IMAGE_VERSION, _target_argv)
        return
    # A real os.execv never returns; reaching here means a no-op reexec_fn (test
    # seam) — nothing more to do.


def _schedule_restart(state, reason: str, *, kill_slots: bool = False,
                      delay: float = 0.5) -> None:
    """Ack-first restart used by the /ops handlers: schedule ``_restart`` to run
    AFTER the caller sends its HTTP ack (the drain must not block the response).
    ``procutil.reexec`` is resolved NOW so a monkeypatched no-op is captured in
    the timer closure (test safety) and the standalone path honors it."""
    from hugpy_platform.procutil import reexec
    threading.Timer(
        delay,
        lambda: _restart(state, reason=reason, reexec_fn=reexec, kill_slots=kill_slots),
    ).start()


def _apply_spill(spill: dict | None) -> None:
    """Translate a per-request spill override dict into the env vars the spill
    module reads. Only set keys that were provided; the model loads lazily, so
    setting these before the first request for a model takes effect on load.

    NOTE: changing spill for an ALREADY-loaded model has no effect until it's
    evicted/reloaded — central can force that via a fresh worker process or by
    reassigning before first use. For the common case (assign, then chat) the
    override lands before the model is built.
    """
    spill = spill or {}
    # Clear the k37 mode-contract envs FIRST (even for an empty spill): a
    # max-gpu request is {} on the wire, and it must reset any mode a prior
    # request left behind — otherwise the mode leaks across models.
    for key in _SPILL_ENV_CLEAR_WHEN_ABSENT:
        if key not in spill or spill[key] is None:
            os.environ.pop(_SPILL_ENV[key], None)
    if not spill:
        return
    for key, env_name in _SPILL_ENV.items():
        if key not in spill or spill[key] is None:
            continue
        val = spill[key]
        if isinstance(val, dict):
            os.environ[env_name] = json.dumps(val, sort_keys=True, default=str)
            continue
        if isinstance(val, (list, tuple)):
            val = ",".join(str(x) for x in val)
        os.environ[env_name] = str(val)


# The dispatch cache is keyed by (model, task), while placement/precision are
# properties of the loaded weights.  Remember the latter independently so a
# second request cannot reuse a resident created under a different contract.
_LOAD_CONTRACTS: dict[str, tuple] = {}
_LOAD_CONTRACTS_LOCK = threading.Lock()
# alloc_source is provenance (differs per request id), never a contract term.
# no_evict is per-REQUEST admission politeness (may THIS load evict others?),
# not a property of the loaded weights — as a contract term, callers that
# differ only in politeness evicted and reloaded the model on every alternate
# request, so it never finished loading (coder-next, 2026-09-25).
_LOAD_CONTRACT_KEYS = tuple(sorted(k for k in _SPILL_ENV
                                   if k not in ("alloc_source", "no_evict")))


def _contract_key(model_key: str) -> str:
    """The key the resident is actually held under on this worker. Central
    may send any spelling (bare / ``owner~name``); the slot and the loaded-key
    set use the worker's canonical registry key, so a contract recorded under
    the spelling missed the resident entirely (2026-09-23: a per-request
    ram-only seat of AtomicChat~Qwen3.8-27B-GGUF held as Qwen3.8-27B-GGUF was
    never evicted). Resolution rule unchanged — get_model_config's own."""
    try:
        from hugpy_engine.config.main import get_model_config as _gmc
        resident = [k for k in (loaded_model_keys() or [])]
        return _gmc(model_key, prefer=resident or None).model_key or model_key
    except Exception:  # noqa: BLE001 — unresolvable: keep the spelling
        return model_key


def _load_contract(spill: dict | None) -> tuple:
    spill = spill if isinstance(spill, dict) else {}
    return tuple((key, str(spill[key])) for key in _LOAD_CONTRACT_KEYS
                 if key in spill and spill[key] is not None)


def _prepare_load_contract(state, model_key: str | None,
                           spill: dict | None) -> None:
    """Evict a resident whose load-time contract differs from this request.

    Called only after the model's generation gate is acquired, so an
    in-process runner cannot be torn down beneath an active answer.  GGUF slots
    additionally verify their measured slot status; this worker-level guard is
    what gives Transformers and every other cached backend the same semantics.
    """
    if not model_key:
        return
    model_key = _contract_key(model_key)
    wanted = _load_contract(spill)
    with _LOAD_CONTRACTS_LOCK:
        previous = _LOAD_CONTRACTS.get(model_key)
        resident = model_key in set(loaded_model_keys())
        # On the first observed explicit contract, an already-resident model
        # has unknown provenance. Rebuild once rather than claim it matches.
        mismatch = ((previous is not None and previous != wanted)
                    or (previous is None and bool(wanted) and resident))
        if mismatch:
            result = _evict_model(state, model_key, force=True)
            if resident and not result.get("evicted"):
                raise RuntimeError(
                    f"could not replace resident {model_key!r} for changed load "
                    f"contract: {result.get('reason') or 'eviction did not occur'}")
            logger.info("load contract changed for %s: %s -> %s; resident evicted "
                        "before ordinary inference", model_key, previous, wanted)
        _LOAD_CONTRACTS[model_key] = wanted


def _sse(payload: dict) -> bytes:
    # werkzeug's WSGI server asserts the app yields bytes, not str — so encode
    # here. (gunicorn is more lenient, but the worker runs the dev server.)
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


# Continuation passes + seam-dedup now live in the shared core engine
# abstract_hugpy_dev.managers.dispatch.execute_chat_stream (honoring the WORKER_*
# env knobs), so the worker no longer carries its own copy.


def _event_to_dict(ev) -> dict:
    """Map a dispatch StreamEvent to the worker's SSE dict shape.

    token/done/error get the slim browser payloads; status/provisioning/
    continuation passthrough events ride through verbatim via model_dump().
    """
    t = getattr(ev, "type", None)
    if t == "token":
        return {"type": "token", "text": getattr(ev, "text", "")}
    if t == "done":
        out = {"type": "done", "finish_reason": getattr(ev, "finish_reason", "stop")}
        # Token accounting (DoneEvent.usage, additive): forward when the engine
        # reported it so central's /v1 usage object is real for relayed chats.
        usage = getattr(ev, "usage", None)
        if isinstance(usage, dict) and usage:
            out["usage"] = usage
        # ENGINE TIMINGS on the wire (operator, 2026-07-25: "maximizing tok/s").
        # Same additive shape as `usage` directly above, and for the same
        # reason: the producer knows it, central cannot.
        #
        # Without this line the whole tok/s chain is inert. Verified live before
        # adding it: ccp_runner captures the engine's `timings`, base_runner
        # threads them through _take_stream_timings(), the DoneEvent schema
        # carries them, central's relay reads them via tok_s_from_timings() and
        # its sink is registered — and the SSE `done` frame ended
        # `{"type":"done","finish_reason":"stop"}` with no timings at all, so
        # central received nothing and recorded nothing. Every piece passed its
        # own tests; the wire was the gap.
        timings = getattr(ev, "timings", None)
        if isinstance(timings, dict) and timings:
            out["timings"] = timings
        return out
    if t == "error":
        return {"type": "error", "message": getattr(ev, "message", "run failed")}
    try:
        return ev.model_dump()
    except Exception:
        return {"type": str(t or "status")}


def _stream_sync(payload: dict, request_id: str | None = None):
    """Relay the shared chat engine as SSE from Flask's sync context.

    Auto-continuation + seam-dedup now live in the core
    ``abstract_hugpy_dev.managers.dispatch.execute_chat_stream`` engine — the exact
    same one the central node drives — so worker chat and local chat behave
    identically. This wrapper only materializes an inlined upload, registers the
    request in the shared comms JobStore with a cancel handle (POST
    /infer/cancel trips it — same F5 substrate central uses, no private
    cancel dict), drives the async engine in a sync loop, and encodes each
    StreamEvent as an SSE line.
    """
    from hugpy_platform import async_runtime
    from hugpy_control.jobs import job_store
    tmp = _materialize_file(payload)

    # Register a cancel Event for this request so /infer/cancel can trip it, and
    # thread its id through the engine so all continuation passes share it. The
    # Event binds to the shared runtime loop on first await; cancellation sets
    # it via call_soon_threadsafe (cross-thread set is otherwise unsafe).
    cancel_event = asyncio.Event()
    if request_id:
        try:
            existing = job_store.get(request_id)
            if existing is None or existing.terminal:
                job_store.create(str(payload.get("model_key") or ""),
                                 id=request_id, kind="chat", transport="worker")
            job_store.attach_cancel(
                request_id,
                lambda: async_runtime.call_soon_threadsafe(cancel_event.set))
        except Exception:
            pass
        payload.setdefault("request_id", request_id)

    agen = None
    try:
        agen = execute_chat_stream(cancel_event=cancel_event, **payload)
        # Drive on the process-wide async runtime (one long-lived loop) instead
        # of a fresh per-request loop — fixes "bound to a different event loop"
        # for any cached asyncio primitive. iter_sync owns step-cancel + aclose.
        for event in async_runtime.iter_sync(agen):
            if request_id and getattr(event, "type", None) == "token":
                try:
                    job_store.on_output(request_id)
                except Exception:
                    pass
            yield _sse(_event_to_dict(event))
    except Exception as exc:
        # Last-resort guard: never let an exception escape into the WSGI layer
        # (that aborts the stream with a raw traceback). Emit a clean error.
        logger.warning("stream failed: %s: %s", type(exc).__name__, exc)
        if request_id:
            try:
                job_store.finish(request_id, error=exc)
            except Exception:
                pass
        yield _sse(_with_load_failure(
            {"type": "error", "message": f"{type(exc).__name__}: {exc}"}, exc))
    finally:
        if request_id:
            # done, or cancelled if a cancel was requested; no-op if the
            # except above already marked it failed.
            try:
                job_store.finish(request_id)
            except Exception:
                pass
        _cleanup_file(tmp)


def loaded_model_keys() -> list[str]:
    try:
        from hugpy_engine.dispatch.dispatch import loaded_model_keys as _loaded
        keys = {mk for (mk, _task) in _loaded()}
        # De-dup slot-vs-in-process: a GGUF model seated in a slot leaves a
        # HOLLOW LlamaCppChatRunner in dispatch _INSTANCES whose underlying
        # runner is an HTTP proxy to the slot child (no weights in THIS
        # process). Reporting it as an in-process ('ram'/'loaded') resident is
        # what makes a slot-served model ALSO read 'loaded' and FLAP with its
        # slot 'serving' row. Prefer the slot — report only genuine in-process
        # residents. Slot occupants stay protected via the _slot_occupants()
        # unions at the storage/residency callers and appear as slot rows in
        # allocations. Discriminating on runner TYPE (not the transient per-beat
        # slot snapshot) removes the flap entirely.
        try:
            from hugpy_engine.llama.runners.get import slot_backed_model_keys
            keys -= slot_backed_model_keys()
        except Exception:
            pass
        return sorted(keys)
    except Exception:
        return []


def _loading_model_keys() -> list[str]:
    """Models whose weights are LOADING right now — the console's 'heating'."""
    try:
        from hugpy_engine.dispatch.dispatch import loading_model_keys
        return loading_model_keys()
    except Exception:
        return []


def _path_bytes(path: str) -> int:
    """On-disk bytes of a model path, NOT following symlinks (a symlinked comfy
    checkpoint or shared file costs this box ~nothing to keep)."""
    try:
        if not path or not os.path.exists(path):
            return 0
        if os.path.islink(path):
            return 0
        if os.path.isfile(path):
            return os.path.getsize(path)
        total = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    if not os.path.islink(fp):
                        total += os.path.getsize(fp)
                except OSError:
                    pass
        return total
    except OSError:
        return 0


def _store_root_copy_path(mk: str, cfg) -> str:
    """The path of this model's copy under the WORKER'S OWN store root, when one
    exists and is complete — regardless of what get_model_path's read-through
    prefers.

    THE ae READ-THROUGH GAP (2026-07-17, slice 3). get_model_path resolves via
    _resolved_local, which honours a NAS read-through: on a box like ae whose
    DEFAULT_ROOT is the HOT drive but which ALSO mounts the shared/central NAS
    (carrying the .hugpy-central-catalog sentinel), a model can resolve to its
    NAS copy even though a re-promotable copy sits on the hot store root. That
    NAS path classifies "shared/central — never reaped" -> protected -> the hot
    copy never becomes an eviction candidate and _path_bytes counts the NAS size.

    The reaper must classify the STORE-ROOT copy: resolve the model's dir with
    the resolver PINNED to this worker's own store root (never the NAS), so the
    reap row's path+bytes and its protection verdict are evaluated on the hot
    copy (hot990 -> not shared -> reapable when the box flag is on). Returns ""
    when no complete copy exists under the store root (then the caller falls back
    to get_model_path — a model served straight from the NAS has no hot row).

    Best-effort and never raises: any failure returns "" and the caller uses the
    read-through path as before, so this can only ADD hot-copy candidates, never
    remove an existing protection.
    """
    try:
        root = _models_store_root()          # e.g. /mnt/hot990/hugpy-worker/models
        if not root:
            return ""
        # _models_store_root returns the MODELS dir; resolve_model_dir expects the
        # DEFAULT_ROOT (it joins "models"/ itself), so hand it the parent when the
        # root ends in a models/ component, else the root as-is.
        base = os.path.dirname(root) if os.path.basename(root) == "models" else root
        from hugpy_storage.model_paths import resolve_model_dir
        routing = {
            "hub_id": getattr(cfg, "hub_id", None),
            "framework": getattr(cfg, "framework", None),
            "filename": getattr(cfg, "filename", None),
            "include": getattr(cfg, "include", None),
            "primary_task": getattr(cfg, "primary_task", None),
            "tasks": getattr(cfg, "tasks", None),
            "folder": getattr(cfg, "folder", None),
        }
        # require_complete=True: only a COMPLETE store-root copy is an evictable
        # row. An incomplete/absent hot copy -> "" -> caller uses read-through.
        d = resolve_model_dir(routing, root=base, cfg=cfg, require_complete=True)
        if not d:
            return ""
        # Guard: the resolver is pinned to `base`, but be defensive — only accept
        # a dir that really lives under the store root (never a NAS path that
        # slipped through a symlinked candidate).
        rp = os.path.realpath(d)
        root_rp = os.path.realpath(base)
        if rp == root_rp or rp.startswith(root_rp + os.sep):
            return d
        return ""
    except Exception:  # noqa: BLE001 — never raise into the scan
        return ""


def _reap_scan(state: "WorkerState") -> dict:
    """The reaper's read-only survey: which local model files are RECLAIMABLE
    (on disk, but not static / loaded / loading / provisioning, on a reapable
    non-shared store), and which are PROTECTED (and why). Never touches comfy
    rows — those are symlinks into the operator's ComfyUI, not this worker's
    downloads.

    📌 pin AND assignment are NOT protection reasons here (operator, 2026-07-17):
    both designate only ROUTING/attribution that survives restarts — no bearing on
    eviction. An assigned OR pinned model's files are reclaimable; the assignment/
    pin survives the delete and the bytes re-pull on next call. (Slice 7 aligned
    the bulk path with budget._is_protected and central's proposal chain.)

    This is advisory: /reap re-checks every guard at delete time, because state
    (a load, a pull) can change between preview and reclaim.
    """
    try:
        from hugpy_fleet.worker.imports import get_models_dict, get_model_config, get_model_path
        from hugpy_storage.provision import model_is_local, _on_shared_model_store, _model_store_reapable
    except Exception as exc:  # noqa: BLE001
        # HONESTY (slice 3, B): a scan that couldn't even import must NOT read as
        # a clean empty store. Carry the error so _worker_storage/central surface
        # "scan broken", never rows:0 masquerading as "nothing on disk".
        return {"reclaimable": [], "protected": [], "error": str(exc),
                "scan_keys_considered": 0, "scan_rows": 0}

    assigned = set(state.assigned_models or [])
    # HARDEN (reaper guard): fold in slot-seated / answering models.
    # loaded_model_keys() is in-process only and MISSES models seated in the
    # slot pool that are actively serving a request. Union _slot_occupants()
    # so an approved reap can never delete a resident/answering model even if
    # it slipped onto the approved list between preview and reclaim.
    loaded = set(loaded_model_keys()) | _slot_occupants()
    loading = set(_loading_model_keys())

    reclaimable, protected = [], []
    scan_error = ""
    try:
        # Enumerate the UNION, not just get_models_dict(): on a WORKER that dict is
        # the built-in staples + comfy sweep + on-disk discovery report and NEVER
        # includes MODEL_REGISTRY, so models held purely by CENTRAL ASSIGNMENT
        # (discovered rows — a designated gguf like flux2) were surveyed as ABSENT,
        # dropping tens of GB from the storage report though they're on disk, in
        # models_local, and slot-seated. Fold in assigned + loaded + loading; the
        # loop below skips any key that isn't model_is_local, so this only ADDS
        # on-disk models the staple-only dict missed.
        registry_keys = set(get_models_dict().keys())
    except Exception as exc:  # noqa: BLE001
        # DON'T abandon the scan (slice 3, A defense): even if the registry build
        # blew up (e.g. a discovery report unreadable for a process whose $HOME /
        # store root wasn't ready at import), assignment + slot truth still name
        # real on-disk models. Record the error, proceed with the key set we can
        # trust, and fold in _models_local so central-registered copies resolve.
        registry_keys = set()
        scan_error = f"get_models_dict: {exc}"
    try:
        local_keys = set(_models_local(state))       # assigned ∩ on-disk (cached)
    except Exception:  # noqa: BLE001
        local_keys = set()
    keys = registry_keys | assigned | loaded | loading | local_keys

    considered = len(keys)
    row_errors = 0
    # SKIP-REASON HISTOGRAM (slice 5). Cheap, permanent per-key accounting of why
    # a considered key produced NO row. This is what would have named the ae
    # 2026-07-17 incident (74 considered, 0 rows) in a single heartbeat:
    # {"not_local": 74} points straight at presence, distinguishing it from
    # {"no_config": 74} (registry/resolution) or {"comfy": N}. Rows that DO
    # classify are not counted here (they land in reclaimable/protected).
    skip_reasons: dict[str, int] = {}

    def _skip(reason: str):
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    unresolved: list[str] = []
    for mk in keys:
        try:
            cfg = get_model_config(mk)
        except Exception:
            # An unlearned registry row, NOT an absent model. Left alone this
            # under-reports the store: on ae 54 of 87 considered keys skipped
            # here, so only 20 classified while 555 GiB sat resident. Record it
            # for the out-of-band learn pass (metadata only, no weight bytes) so
            # the NEXT scan can classify it; the histogram entry stays, because
            # this scan genuinely could not price the row.
            _skip("no_config")
            unresolved.append(mk)
            continue
        if getattr(cfg, "framework", None) == "comfy":
            _skip("comfy")
            continue  # symlinks / operator-owned — reaper stays clear
        try:
            if not model_is_local(mk):
                _skip("not_local")
                continue
        except Exception:
            row_errors += 1
            _skip("locality_error")
            continue
        # STORE-ROOT COPY classification (slice 3, C). Prefer the copy under THIS
        # worker's own store root over get_model_path's read-through, which on ae
        # can hand back a NAS path (shared/central -> protected) even when a
        # re-promotable hot copy exists. Evaluate path+bytes+protection on the hot
        # copy so it becomes a real candidate; fall back to the read-through path
        # when there is no complete store-root copy (a model served straight off
        # the NAS legitimately has no evictable hot row).
        try:
            path = _store_root_copy_path(mk, cfg)
        except Exception:  # noqa: BLE001
            path = ""
        if not path:
            try:
                path = get_model_path(mk) or ""
            except Exception:
                path = ""
        size = _path_bytes(path)
        # SAFE-BY-DEFAULT: a model is reapable only on a box that declared its
        # store local & disposable AND is not shared/central. Everything else is
        # PROTECTED here, so nothing on a shared/unconfigured box is ever proposed
        # — the console still shows usage but offers no deletion. wipe_model
        # re-checks the same gate at delete time.
        rp = os.path.realpath(path) if path else ""
        if not _model_store_reapable(rp):
            shared = (not rp) or _on_shared_model_store(rp)
            why = ("shared/central storage — never reaped" if shared
                   else "model store not marked reapable")
            # k60 ACCOUNTING (operator, 2026-07-31): these bytes are NOT in this
            # worker's eviction economy, so they must not be priced against its
            # budget. The row still ships (the console shows WHAT is on the
            # drive) but carries `store` + counts_toward_budget=False so every
            # downstream sum can label it instead of charging for it. ae read
            # "2.8 TiB used / 800 GB budget · ⚠ over budget" on a 1.7 TiB hot
            # drive because the SHARED catalog was summed as resident cache —
            # and "over budget" reads to an operator as "a delete is coming".
            protected.append({"model_key": mk, "bytes": size, "why": why,
                              "store": "shared" if shared else "unreapable",
                              "counts_toward_budget": False})
            continue
        # 📌 pin AND assignment do NOT protect files (operator, 2026-07-17):
        # both designate only ROUTING/attribution that survives restarts — neither
        # has any bearing on eviction/reaping. An assigned OR pinned model's files
        # are reclaimable like any other; the assignment/pin survives the delete
        # and the bytes re-pull on the next call. This is the BULK path catching
        # up to the call-driven path (budget._is_protected dropped `assigned` at
        # f1894b2): the preview and executor must agree with central's proposal
        # chain, which already proposes assigned-but-cold models — otherwise an
        # over-budget box whose cold models are ALL assigned (the normal case: ae
        # designates everything) can never be cleared (slice 7, the "7 refused
        # assigned" incident). ONLY the DURABLE-presence + live-use guards remain:
        # 🔒static, loaded/loading; provisioning is guarded at reclaim time (a
        # mid-pull delete corrupts the fetch). The store-reapable + shared/central
        # sentinel gates above already ran; the path jail runs at wipe time.
        if _residency(mk) == "static":
            protected.append({"model_key": mk, "bytes": size, "why": "static",
                              "store": "reapable", "counts_toward_budget": True})
        elif mk in loaded or mk in loading:
            protected.append({"model_key": mk, "bytes": size, "why": "loaded",
                              "store": "reapable", "counts_toward_budget": True})
        else:
            # Assigned-but-cold falls through here — a reclaimable candidate.
            # Store-root copy path travels on the row so _reap_reclaim/wipe act on
            # the hot copy — never the NAS (re-proven at delete time).
            reclaimable.append({"model_key": mk, "bytes": size, "path": path})

    reclaimable.sort(key=lambda r: r["bytes"], reverse=True)
    if unresolved:
        # Out-of-band, single-flight, metadata-only: teach the registry the rows
        # this scan could not resolve so the NEXT scan prices them instead of
        # skipping them as no_config forever. Never inline — this runs on the
        # heartbeat path. Guarded: the scan must never fail over a learn kick.
        try:
            _kick_learn_configs(state, unresolved)
        except Exception:  # noqa: BLE001
            pass
    # k60: split the protected pile by whether it is IN the eviction economy.
    # Only reapable-store rows are budget-bearing; shared/unreapable rows are
    # reported and labeled, never priced (see the store gate above).
    unbudgeted = [r for r in protected if not r.get("counts_toward_budget", True)]
    out = {
        "reclaimable": reclaimable,
        "protected": protected,
        "reclaimable_bytes": sum(r["bytes"] for r in reclaimable),
        # BUDGET-BEARING vs LABELED-ONLY bytes (k60). budgeted_bytes is what
        # used/need_bytes may be computed from; unbudgeted_bytes is the shared
        # catalog / never-opted-in store, shown but never charged.
        "budgeted_bytes": (sum(r["bytes"] for r in reclaimable)
                           + sum(r["bytes"] for r in protected
                                 if r.get("counts_toward_budget", True))),
        "unbudgeted_bytes": sum(r["bytes"] for r in unbudgeted),
        "unbudgeted_count": len(unbudgeted),
        "shared_bytes": sum(r["bytes"] for r in unbudgeted
                            if r.get("store") == "shared"),
        "shared_count": sum(1 for r in unbudgeted if r.get("store") == "shared"),
        # DIAGNOSTICS (slice 3, B): make a broken/empty scan self-describing so it
        # can never masquerade as a clean empty store. scan_keys_considered = the
        # full key domain; scan_rows = rows actually classified (reclaimable +
        # protected). considered≫rows with 0 rows is the ae symptom's fingerprint.
        "scan_keys_considered": considered,
        "scan_rows": len(reclaimable) + len(protected),
        "scan_row_errors": row_errors,
        # SKIP-REASON HISTOGRAM (slice 5): why each considered key produced no
        # row — {"not_local": N, "no_config": N, "comfy": N, "locality_error": N}.
        # considered≫rows is now self-explaining in one heartbeat.
        "scan_skip_reasons": skip_reasons,
    }
    if scan_error:
        out["error"] = scan_error
    return out


def _reap_reclaim(state: "WorkerState", model_keys: list[str]) -> dict:
    """Delete the local files of the named models — but ONLY after re-proving,
    per key, that it is still reclaimable (not static/loaded/loading/provisioning,
    not comfy, on a reapable non-shared store). The guard is re-run here, not
    trusted from a stale preview.

    📌 pin AND assignment are deliberately NOT in that list (operator,
    2026-07-17): both designate only ROUTING/attribution that survives restarts —
    no bearing on eviction. A pinned or assigned model's files reap freely; the
    pin/assignment survives the delete and the bytes re-pull on next call. (This
    matches budget._is_protected and central's proposal chain — see slice 7.)"""
    from hugpy_fleet.worker.imports import get_model_config, get_model_path
    from hugpy_storage.provision import model_is_local, wipe_model

    # HARDEN (reaper guard): fold in slot-seated / answering models.
    # loaded_model_keys() is in-process only and MISSES models seated in the
    # slot pool that are actively serving a request. Union _slot_occupants()
    # so an approved reap can never delete a resident/answering model even if
    # it slipped onto the approved list between preview and reclaim. FAIL CLOSED:
    # if the slot probe can't answer, refuse the whole reclaim rather than delete
    # while blind to what's resident.
    try:
        slot_occ = _slot_occupants(strict=True)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "results": [],
                "error": f"slot occupancy unknown ({exc}) — refusing to reap (fail-closed)"}
    loaded = set(loaded_model_keys()) | slot_occ
    loading = set(_loading_model_keys())
    # Models mid-provision (downloading from central/HF) — deleting under a live
    # pull corrupts the fetch; guard explicitly instead of via assigned-coupling.
    provisioning = set(getattr(state, "_provisioning", None) or [])

    results = []
    for mk in model_keys:
        try:
            cfg = get_model_config(mk)
        except Exception:
            results.append({"model_key": mk, "ok": False, "reason": "unknown model"})
            continue
        if getattr(cfg, "framework", None) == "comfy":
            results.append({"model_key": mk, "ok": False, "reason": "comfy (symlink/operator files) — never reaped"})
            continue
        # 📌 pin AND assignment are NOT reap guards (operator, 2026-07-17): both
        # mean only that ROUTING/attribution survives restarts — no bearing on
        # eviction. A pinned OR assigned model's files are reclaimable; the pin/
        # assignment survives the delete and the bytes re-pull on next call. This
        # mirrors budget._is_protected (which dropped `assigned` at f1894b2) so
        # the executor agrees with central's proposal and the scan preview — the
        # slice-7 fix for "central proposes what the worker refuses". static +
        # loaded/loading + provisioning still guard below; the store-reapable/
        # shared-sentinel gates re-run inside wipe_model's path jail.
        if _residency(mk) == "static":
            results.append({"model_key": mk, "ok": False, "reason": "static"})
            continue
        if mk in provisioning:
            results.append({"model_key": mk, "ok": False, "reason": "provisioning"})
            continue
        if mk in loaded or mk in loading:
            results.append({"model_key": mk, "ok": False, "reason": "loaded/serving/loading"})
            continue
        try:
            if not model_is_local(mk):
                results.append({"model_key": mk, "ok": False, "reason": "no local files"})
                continue
        except Exception:
            results.append({"model_key": mk, "ok": False, "reason": "locality check failed"})
            continue
        # STORE-ROOT COPY (slice 3, C): target the hot copy the scan classified,
        # not get_model_path's read-through (which on ae resolves to the NAS the
        # shared gate correctly refuses). Re-resolve it here — same helper the
        # scan used — so preview and delete can't diverge; fall back to the
        # read-through path when there is no complete store-root copy. wipe_model
        # re-proves the jail + shared gate on whichever realpath this is.
        try:
            target = _store_root_copy_path(mk, cfg)
        except Exception:  # noqa: BLE001
            target = ""
        if not target:
            try:
                target = get_model_path(mk) or ""
            except Exception:
                target = ""
        try:
            freed = _path_bytes(target)
        except Exception:
            freed = 0
        gone = wipe_model(mk, path=target)  # jailed + shared-gate re-proven on target
        _MODELS_LOCAL_CACHE["at"] = 0.0  # force a fresh local walk next beat
        results.append({"model_key": mk, "ok": bool(gone),
                        "freed_bytes": freed if gone else 0,
                        "reason": "" if gone else "delete refused/failed (path jail?)"})
    return {"ok": True, "results": results,
            "freed_bytes": sum(r.get("freed_bytes", 0) for r in results)}


# ── per-worker local-STORAGE survey (heartbeat) ─────────────────────────────
# What model cache this box holds: total on-disk bytes, per-model sizes, and
# which models are PROTECTED (and why). The worker reports the flags only IT can
# know — loaded / slot-seated / loading / provisioning / assigned — plus sizes.
# Central OVERLAYS the two facts the worker cannot know (per-(worker,model)
# last_picked and the disk budget) and derives over_budget + eviction proposals
# in _public_view. This is REUSED from the reaper's own _reap_scan so the
# storage view can never disagree with what the reaper would actually delete.
_STORAGE_CACHE: dict = {"at": 0.0, "value": None}

# RELEASE-BOUND (2026-07-17): measured store-root size, TTL-cached separately so
# the (cheap) scandir walk of the store root is not tied to the 60s model-scan
# cache. This is the AUTHORITATIVE cache_used_bytes — a real filesystem
# measurement of what is on disk under the model store, fixing the 2026-07-16
# discrepancy where the per-model-dir SUM read 128.8GB while `du` measured 81G
# (the sum double-counted / carried stale manifest keys).
_STORE_MEASURE_CACHE: dict = {"at": 0.0, "value": None}


def _models_store_root() -> str | None:
    """The directory the worker's model weights actually live under."""
    try:
        from hugpy_platform.constants import MODELS_HOME
        if MODELS_HOME and os.path.isdir(MODELS_HOME):
            return str(MODELS_HOME)
    except Exception:  # noqa: BLE001
        pass
    try:
        from hugpy_platform.constants import DEFAULT_ROOT
        cand = os.path.join(str(DEFAULT_ROOT), "models")
        if os.path.isdir(cand):
            return cand
        if os.path.isdir(DEFAULT_ROOT):
            return str(DEFAULT_ROOT)
    except Exception:  # noqa: BLE001
        pass
    return None


def _path_on_shared_store(path: str) -> bool:
    """True when `path` lives on the SHARED/central catalog (sentinel or the
    per-box env flag). Thin, never-raising wrapper over the SAME predicate the
    delete guard uses, so labeling and protection can never disagree."""
    try:
        from hugpy_storage.provision import _on_shared_model_store
        return bool(path) and bool(_on_shared_model_store(os.path.realpath(path)))
    except Exception:  # noqa: BLE001
        return False


def _store_root_budgeted() -> bool:
    """True when this worker's OWN store root is a reapable store — the only
    case where a measured store-root size is budget-bearing (k60).

    On ae the root resolves onto the shared catalog, so the measured walk
    returns the WHOLE FLEET's resident catalog (2.8 TiB). Pricing that against
    an 800 GB worker cap produced a permanent "⚠ over budget · 2.0 TiB over" on
    a 1.7 TiB box — an eviction reading for files this box may never delete.
    Fail SAFE: any resolution failure returns False, so an unknown root is
    LABELED rather than charged (never the other way round)."""
    try:
        from hugpy_storage.provision import _model_store_reapable
        root = _models_store_root()
        return bool(root) and bool(_model_store_reapable(os.path.realpath(root)))
    except Exception:  # noqa: BLE001
        return False


def _measured_store_bytes() -> int | None:
    """Real on-disk bytes under the model store root — a scandir walk, TTL-cached
    (120s). Non-following of symlinks so shared/comfy links cost 0 here (same rule
    as _path_bytes). Returns None if the root can't be resolved (caller then falls
    back to the per-model sum). This is the honest cache_used the heartbeat ships.
    """
    now = time.time()
    cached = _STORE_MEASURE_CACHE["value"]
    if cached is not None and now - _STORE_MEASURE_CACHE["at"] < 120.0:
        return cached
    root = _models_store_root()
    if not root:
        return None
    total = 0
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    _STORE_MEASURE_CACHE.update(at=now, value=total)
    return total


# ── ORPHAN (unattributed-on-disk) scan ──────────────────────────────────────
# RELEASE-BOUND (2026-07-17 addendum). The reaper survey (_reap_scan) only ever
# looks at KNOWN/assigned/loaded keys, so on-disk residue that matches NO current
# model — a stalled *.part set from an old eager-era pull, or a whole model dir
# for something no longer assigned — is INVISIBLE to central (computron held 5.7G
# of stalled Qwen2.5-VL-3B .part files that appeared nowhere in the UI). This
# scan walks the store root for that residue and reports it so the console can
# surface "unattributed on disk: X GB". Naming (keeper owns nomenclature): the
# class is "orphaned" in code; the UI labels it "unattributed on disk".
_ORPHAN_CACHE: dict = {"at": 0.0, "value": None}


def _dir_bytes_no_links(path: str) -> int:
    total = 0
    stack = [path]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_symlink():
                            continue
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _orphan_scan(state: "WorkerState", known_keys: set) -> dict:
    """On-disk residue attributed to NO current model. TTL-cached (120s).

    Two orphan shapes:
      * a completed model DIR (is_model_dir) whose hub_id/key is in neither the
        manifest nor the assignment/loaded set — a leftover from a prior
        assignment that was never reaped;
      * a STALLED partial: a dir under the store holding *.part staging files
        (a crash/abandoned pull) — is_model_dir() is False for it (no real
        weights), so nothing else sees it.

    ``known_keys`` = every name a live model might match (manifest keys + hub_ids
    + assigned/loaded/loading). Anything on disk NOT matching one of these is
    orphaned. Conservative: an entry we cannot positively tie to a known model is
    reported (visible), never auto-deleted — this scan proposes nothing.

    MATCHING (fixed 2026-07-17, over-report root cause): a dir is compared
    against ``known_keys`` through ONE shared expansion —
    ``provision.known_model_dir_forms`` — the same normalization
    ``model_is_local``/the read-through resolver are built on. The old
    comparison only lowercased+stripped, so a ``~``-qualified assignment key
    (``owner~repo``, minted on an owner collision — see discover_models in
    imports/apis/get_module.py) never textually matched a directory-derived
    ``owner/repo`` name; it happened to still "work" only via an accidental
    bare-repo-name fallback, which breaks the moment two owners share a repo
    basename (exactly the case ``~`` exists to disambiguate). The dir is
    matched BOTH by its relative path (catches a legacy nested-path copy of a
    known model — legacy-path, not orphan) and by its best-effort hub-id guess
    (catches a flat/marker-named dir). ``misc/comfy/**`` is excluded by policy
    (provision.is_doctrine_excluded) — comfy checkpoints are symlinks the
    reaper/storage accounting already treat as never-orphaned; operator
    doctrine: "comfy is excluded from allocations, models can sit on the drive
    unattributed."
    """
    now = time.time()
    cached = _ORPHAN_CACHE["value"]
    if cached is not None and now - _ORPHAN_CACHE["at"] < 120.0:
        return cached

    root = _models_store_root()
    out = {"items": [], "bytes": 0, "count": 0}
    if not root:
        _ORPHAN_CACHE.update(at=now, value=out)
        return out

    try:
        from hugpy_storage.model_paths import is_model_dir, is_directory_excluded, get_hub_id_from_directory
        from hugpy_storage.provision import (
            known_model_dir_forms,
            dir_is_known_model,
            is_doctrine_excluded,
            _dir_slug,
        )
    except Exception:  # noqa: BLE001 — never break a heartbeat over this
        _ORPHAN_CACHE.update(at=now, value=out)
        return out

    known_forms = known_model_dir_forms(known_keys)

    items: list[dict] = []
    seen_dirs: set = set()

    def _is_orphan_dir(dirpath: str) -> bool:
        rel = os.path.relpath(dirpath, root)
        if is_doctrine_excluded(rel):
            return False
        if dir_is_known_model(rel, known_forms):
            return False
        # Secondary check: the best-effort hub-id guess (marker-first, then
        # layout-aware path guess) — catches a dir whose relative path doesn't
        # literally match a candidate dir (e.g. a legacy shape the resolver
        # doesn't enumerate) but whose declared/guessed hub_id still resolves.
        hub = get_hub_id_from_directory(dirpath, models_home=root)
        if hub and _dir_slug(hub) in known_forms:
            return False
        if hub:
            tail = str(hub).rsplit("/", 1)[-1]
            if _dir_slug(tail) in known_forms:
                return False
        return True

    # Walk once: catch model dirs (leaves) AND .part-bearing dirs.
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if not is_directory_excluded(os.path.join(dirpath, d))]
        has_part = any(n.endswith((".part", ".part.state.json")) for n in filenames)
        model_leaf = is_model_dir(dirpath)
        if not (model_leaf or has_part):
            continue
        if dirpath in seen_dirs:
            continue
        if _is_orphan_dir(dirpath):
            b = _dir_bytes_no_links(dirpath)
            if b > 0:
                items.append({
                    "path": os.path.relpath(dirpath, root),
                    "bytes": b,
                    "kind": "partial" if (has_part and not model_leaf) else "stale-dir",
                })
            seen_dirs.add(dirpath)
        if model_leaf:
            dirnames[:] = []   # leaf — don't descend into a model's own files

    items.sort(key=lambda x: x["bytes"], reverse=True)
    out = {"items": items,
           "bytes": sum(i["bytes"] for i in items),
           "count": len(items)}
    _ORPHAN_CACHE.update(at=now, value=out)
    return out


def _storage_model_row(mk: str, size: int, loaded: set, loading: set,
                       provisioning: set, assigned: set,
                       why_hint: str = "", store: str = "reapable",
                       counts_toward_budget: bool = True) -> dict:
    """One per-model row for the heartbeat storage view: bytes + every
    protection flag + a human `why`. loaded is ALREADY answer-inclusive
    (loaded_model_keys() ∪ _slot_occupants()) at the caller.

    ``store`` / ``counts_toward_budget`` (k60) carry _reap_scan's STORE-GATE
    classification onto the wire: "shared" (the central catalog this box only
    reads through) and "unreapable" (a store the box never opted in to) are
    LABELED, never priced — they contribute zero to used/need_bytes. Only
    "reapable" rows are in this worker's eviction economy."""
    is_pinned = _pinned(mk)
    is_static = _residency(mk) == "static"
    is_loaded = mk in loaded
    is_loading = mk in loading
    is_provisioning = mk in provisioning
    is_assigned = mk in assigned
    # `why_hint` carries _reap_scan's STORE-GATE verdict ("shared/central storage
    # — never reaped" / "model store not marked reapable"). It is a GENUINE
    # protection reason and the only one this row cannot re-derive from the flags
    # below — it comes from _model_store_reapable(realpath), which the caller
    # already resolved. Honour it.
    #
    # BUG (fixed 2026-07-17): `protected` used to ignore why_hint entirely, so a
    # model protected ONLY by the store gate shipped to central as
    # protected=False, why="" — an UNPROTECTED row with NO reason. On ae that
    # silently mislabelled 101 store-gated models as eviction candidates, and
    # fit_plan's refusal then reported "0 B reclaimable (1 loaded)" — hiding the
    # real cause (the whole store is gated) behind a number that made the FIFO
    # look broken. The policy was always right; only this report lied.
    store_gated = bool((why_hint or "").strip())
    # 📌 pin AND assignment do NOT protect files (operator, 2026-07-17): both are
    # ROUTING/attribution that survives restarts — no bearing on eviction. Both
    # are still reported below as ATTRIBUTION (`pinned`/`assigned` flags + `why`),
    # but NEITHER sets `protected`. This aligns the heartbeat row with _reap_scan,
    # _reap_reclaim, and budget._is_protected (all of which treat assigned as a
    # candidate) so preview, executor, and central agree (slice 7). 🔒static is the
    # durable local-presence guard; loaded/loading/provisioning are live-use
    # guards; the store gate (why_hint) is a hard filesystem fact.
    protected = (is_static or is_loaded or is_loading
                 or is_provisioning or store_gated)
    # Precedence for the human `why` (static > store-gate > loaded > loading >
    # provisioning > assigned > pinned). The store gate outranks the attribution
    # labels DELIBERATELY: those files CANNOT be deleted by the reaper regardless
    # of central's verdict, so a real, enforced reason must beat a routing label.
    # `assigned`/`pinned` read purely as ATTRIBUTION (protected stays False) — a
    # bare-assigned or bare-pinned row is a reclaimable CANDIDATE.
    if is_static:
        why = "static"
    elif store_gated:
        why = why_hint
    elif is_loaded:
        why = "loaded"
    elif is_loading:
        why = "loading"
    elif is_provisioning:
        why = "provisioning"
    elif is_assigned:
        why = "assigned"        # attribution only — protected stays False
    elif is_pinned:
        why = "pinned"          # attribution only — protected stays False
    else:
        why = ""
    return {
        "model_key": mk,
        "bytes": int(size or 0),
        "pinned": is_pinned,
        "loaded": is_loaded,
        "loading": is_loading,
        "provisioning": is_provisioning,
        "assigned": is_assigned,
        "protected": protected,
        "why": why,
        # k60: which STORE the bytes sit on, and whether they are budget-bearing.
        "store": store,
        "counts_toward_budget": bool(counts_toward_budget),
    }


def _refused_snapshot(state: "WorkerState") -> dict:
    """A copy of the storage-REFUSED models, pruned of any that since landed.

    A refusal is a point-in-time verdict: once the model's files are on disk
    (the operator raised disk_cache_gib, or a later pull fit after evictions),
    the "missing — won't fit" reason is stale and must not linger in the console.
    """
    out = {}
    for mk, reason in list(getattr(state, "refused", {}).items()):
        try:
            from hugpy_storage.provision import model_is_local
            if model_is_local(mk):
                state.refused.pop(mk, None)
                continue
        except Exception:  # noqa: BLE001 — a probe failure keeps the reason
            pass
        out[mk] = dict(reason)
    return out


def _worker_storage(state: "WorkerState") -> dict:
    """Heartbeat STORAGE view for one worker (60s-cached — _reap_scan os.walks
    every local model dir via _path_bytes; running that every beat would slow
    heartbeats on boxes with many large models).

    Shape:
      { cache_used_bytes:int,  # on-disk bytes of the models on a REAPABLE store
                               # (k60: shared/unreapable rows count 0 here);
                               # symlinks count 0
        unbudgeted_bytes:int,  # the shared/unreapable bytes — shown, never priced
        disk_free:int,         # = disk.free_bytes (kept for console convenience)
        models:[ {model_key, bytes, pinned, loaded, loading, provisioning,
                  assigned, protected, why, store, counts_toward_budget} ] }

    Comfy rows never appear (skipped by _reap_scan — operator symlinks), so they
    neither inflate cache_used_bytes nor get proposed for eviction.
    """
    now = time.time()
    cached = _STORAGE_CACHE["value"]
    if cached is not None and now - _STORAGE_CACHE["at"] < 60.0:
        # The heavy part (per-model disk walk) stays cached, but REFUSALS are
        # cheap and time-critical: a model refused seconds ago must read as
        # missing-with-a-reason on the NEXT beat, not up to 60s later. Refresh
        # just that key on the cached view.
        cached["refused"] = _refused_snapshot(state)
        return cached

    scan = _reap_scan(state)
    # Cheap set-membership truth (no disk walk) for the per-model flags. `loaded`
    # is answer-inclusive: loaded_model_keys() misses slot occupants, so union
    # _slot_occupants() to protect a model that is seated/answering.
    loaded = set(loaded_model_keys()) | _slot_occupants()
    loading = set(_loading_model_keys())
    try:
        provisioning = set(state._provisioning)
    except Exception:  # noqa: BLE001
        provisioning = set()
    assigned = set(state.assigned_models or [])

    models: list[dict] = []
    cache_used = 0        # BUDGET-BEARING bytes only (reapable stores)
    unbudgeted_bytes = 0  # shared catalog / never-opted-in store — labeled only
    unbudgeted_count = 0
    for row in scan.get("reclaimable", []):
        size = int(row.get("bytes", 0) or 0)
        cache_used += size
        models.append(_storage_model_row(row.get("model_key"), size, loaded,
                                         loading, provisioning, assigned))
    for row in scan.get("protected", []):
        size = int(row.get("bytes", 0) or 0)
        # k60: a row the STORE GATE protects (shared catalog, or a store this
        # box never declared reapable) contributes ZERO to used/need_bytes. It
        # still ships — tagged `shared`/`unreapable` — so the console can show
        # what occupies the drive without implying an eviction is coming.
        counts = bool(row.get("counts_toward_budget", True))
        if counts:
            cache_used += size
        else:
            unbudgeted_bytes += size
            unbudgeted_count += 1
        models.append(_storage_model_row(row.get("model_key"), size, loaded,
                                         loading, provisioning, assigned,
                                         why_hint=row.get("why", ""),
                                         store=row.get("store") or "reapable",
                                         counts_toward_budget=counts))
    models.sort(key=lambda m: m["bytes"], reverse=True)

    disk = _disk_status()
    # AUTHORITATIVE cache_used = MEASURED store-root bytes (release-bound). The
    # per-model dir SUM (`cache_used` here) over-counted in the field (op:
    # 128.8GB summed vs 81G measured), so a real filesystem measurement of the
    # store root is the honest number the gauge should read. Keep the sum as a
    # cross-check diagnostic; fall back to it only if the root can't be measured.
    measured = _measured_store_bytes()
    # k60 ACCOUNTING GATE. The measured walk is only the WORKER'S OWN cache when
    # the store root is a reapable store. When the root resolves onto the shared
    # catalog (ae) it measures the whole fleet's resident catalog, so charging it
    # to this worker's budget is a category error — that is the 2.8 TiB / 800 GB
    # "⚠ over budget" the operator correctly read as "an auto-delete is coming".
    # On such a box the budget number is the reapable-row sum (typically 0) and
    # the measurement is still reported, under a name that cannot be priced.
    root_budgeted = _store_root_budgeted()
    store_root_measured = measured
    if not root_budgeted:
        measured = None
    # ORPHANED (unattributed-on-disk) residue — model dirs / stalled .part sets
    # that match NO current model. Every name a live model might be known by, so
    # the diff doesn't false-flag a resident model as orphaned.
    try:
        known_keys = (set(get_models_dict().keys()) | set(models and [m["model_key"] for m in models] or [])
                      | set(state.assigned_models or []) | loaded | loading | provisioning)
        # fold in each known model's hub_id so a dir named by hub_id matches.
        for _mk in list(known_keys):
            try:
                _c = get_model_config(_mk)
                _h = getattr(_c, "hub_id", None) if _c is not None else None
                if _h:
                    known_keys.add(_h)
            except Exception:  # noqa: BLE001
                continue
        orphans = _orphan_scan(state, known_keys)
    except Exception:  # noqa: BLE001 — a heartbeat must never fail on this
        orphans = {"items": [], "bytes": 0, "count": 0}
    # EFFECTIVE BUDGET (slice 4, min-wins). Resolve the min over {central
    # disk_cache_gib, worker same-drive declarations} for THIS box's store root
    # and report the number + source map so the operator sees WHY a number
    # governs. Not applicable on a shared/central store (the cap is skipped
    # there — slice 2), so we mark it and omit the sources rather than imply a
    # cap. Best-effort: any failure just omits the fields.
    budget_effective_bytes = None
    budget_sources: dict = {}
    budget_not_applicable = False
    try:
        from hugpy_fleet.worker import budget as _budget
        if _budget._store_is_shared():
            budget_not_applicable = True
        else:
            store_root = _models_store_root() or ""
            budget_effective_bytes, budget_sources = _budget.resolve_effective_cap(
                getattr(state, "limits", None) or {}, store_root)
    except Exception:  # noqa: BLE001 — a heartbeat must never fail on this
        pass
    out = {
        "cache_used_bytes": measured if measured is not None else cache_used,
        "cache_used_measured_bytes": measured,      # None if root unresolved/shared
        "cache_used_model_sum_bytes": cache_used,   # legacy per-model-dir sum
        # k60 — the accounting split, so central/console never have to guess:
        #   unbudgeted_*   bytes on a shared/unreapable store: SHOWN, never priced
        #   store_root_*   what the root actually is, and whether it is budget-bearing
        "unbudgeted_bytes": unbudgeted_bytes,
        "unbudgeted_count": unbudgeted_count,
        "shared_bytes": int(scan.get("shared_bytes") or 0),
        "shared_count": int(scan.get("shared_count") or 0),
        "store_root": _models_store_root() or "",
        "store_root_budgeted": root_budgeted,
        "store_root_shared": _path_on_shared_store(_models_store_root() or ""),
        # The raw measurement of the root, ALWAYS honest and never the budget
        # number when the root isn't budget-bearing (that is cache_used_bytes).
        "store_root_measured_bytes": store_root_measured,
        # Orphaned residue (release-bound). UI labels it "unattributed on disk".
        "orphaned_bytes": orphans["bytes"],
        "orphaned_count": orphans["count"],
        "orphaned_items": orphans["items"],
        "disk_free": int(disk.get("free_bytes", 0) or 0),
        # EFFECTIVE per-drive budget (slice 4). budget_sources names every term
        # in GiB (central_gib / worker_hot_cache_gib / …) plus effective_gib +
        # effective_source. Shared store -> budget_cap_not_applicable True.
        "budget_effective_bytes": budget_effective_bytes,
        "budget_sources": budget_sources,
        "budget_cap_not_applicable": budget_not_applicable,
        "models": models,
        # Models REFUSED for storage: the pull never started because even a full
        # FIFO couldn't seat them. {model_key: {state:"refused", reason, ...}}.
        # The console renders these as MISSING with the reason on hover — an
        # honest "won't fit", never a phantom "pulling" that can't finish.
        "refused": _refused_snapshot(state),
        # HOT-CACHE tier (box-local NVMe LRU of the main catalog). Honest section
        # so central/console can surface root/budget/used + per-entry last_called.
        # {"enabled": False} when HUGPY_HOT_CACHE_ROOT is unset (no behaviour).
        "hot_cache": _hot_cache_status(),
        # SCAN DIAGNOSTICS (slice 3, B). Carry the reaper survey's own telemetry
        # so a broken/degraded scan can NEVER masquerade as a clean empty store
        # (the ae 2026-07-17 defect: rows:0 while 65 models were on disk, because
        # a swallowed scan error surfaced identically to "nothing here"). Central
        # passes these through verbatim; the console can surface them later.
        #   scan_error            — set when the registry build failed (scan still
        #                           ran on assignment/slot/local keys)
        #   scan_keys_considered  — size of the full key domain the scan walked
        #   scan_rows             — rows actually classified (reclaimable+protected)
        #   scan_row_errors       — per-model probe failures skipped
        # considered≫0 with rows:0 is the fingerprint of the ae failure.
        "scan_error": scan.get("error") or "",
        "scan_keys_considered": int(scan.get("scan_keys_considered") or 0),
        "scan_rows": int(scan.get("scan_rows") or 0),
        "scan_row_errors": int(scan.get("scan_row_errors") or 0),
        # SKIP-REASON HISTOGRAM (slice 5): why considered keys produced no row —
        # names the ae failure class (not_local / no_config / comfy / …) in one
        # heartbeat instead of leaving considered≫rows unexplained.
        "scan_skip_reasons": scan.get("scan_skip_reasons") or {},
        # REGISTRY SOURCES (slice 6): per-origin count of the live registry —
        # {staple, discovered, central, comfy, total}. A dead source is visible
        # in one beat: the ae 2026-07-17 incident was discovered==0 (stale/absent
        # report left the registry staples-only). Pairs with scan_skip_reasons —
        # no_config≫0 WITH discovered==0 points straight at the report/re-walk.
        "registry_sources": _registry_sources(),
    }
    _STORAGE_CACHE.update(at=now, value=out)
    return out


def _hot_cache_status() -> dict:
    """Best-effort hot-cache overview for the heartbeat storage view; never
    raises into a heartbeat (returns {"enabled": False} on any failure)."""
    try:
        from hugpy_engine.serve import hot_cache
        return hot_cache.status()
    except Exception:  # noqa: BLE001
        return {"enabled": False}


def _registry_sources() -> dict:
    """Per-source count of the worker's live model registry (slice 6): how many
    configs came from each origin — {"staple": N, "discovered": N, "central": N,
    "comfy": N, "total": N}.

    Same honesty pattern as the scan skip-reason histogram: a DEAD source is
    visible in one heartbeat. The ae 2026-07-17 incident was 'discovered'==0 (a
    stale/absent discovery report left the registry staples-only); this names
    that directly instead of leaving 63 no_config skips unexplained.

    Classification (best-effort, read-only): a row is `comfy` when framework==
    comfy; `staple` when its key is a curated MODELS entry; `discovered` when it
    carries a `dir` (the ABSOLUTE on-disk path discover_models stamps — staples
    carry only a layout `folder`, never a `dir`); else `central` (adopted from
    central's config row via ensure_model_registered). Never raises."""
    out = {"staple": 0, "discovered": 0, "central": 0, "comfy": 0, "total": 0}
    try:
        from hugpy_fleet.worker.imports import models_config as mc
        staples = set(getattr(mc, "MODELS", {}).keys())
        reg = getattr(mc, "MODEL_REGISTRY_DICT", None) or {}
        for key, row in reg.items():
            out["total"] += 1
            r = row if isinstance(row, dict) else {}
            if str(r.get("framework") or "") == "comfy":
                out["comfy"] += 1
            elif key in staples:
                out["staple"] += 1
            elif r.get("dir"):
                out["discovered"] += 1
            else:
                out["central"] += 1
    except Exception:  # noqa: BLE001 — a heartbeat must never fail on this
        pass
    return out


def _spill_describe() -> dict:
    try:
        #from abstract_hugpy_dev.managers.spill import describe

        return describe()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
def build_app(state: "WorkerState") -> Flask:
    app = Flask("abstract_hugpy_worker")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "ok": True,
                "worker_id": state.worker_id,
                "name": state.name,
                "gpus": detect_gpus(),
                "cuda": torch_cuda_status(),
                "llama_cpp": llama_cpp_cuda_status(),
                # k94: the native llama-server truth, as its own field (also
                # inside llama_cpp/engine for the central snapshot path).
                "native_engine": _native_engine_status_safe(),
                "assigned_models": state.assigned_models,
                "provisioning": sorted(state._provisioning),
                "provision_progress": state.provision_snapshot(),
                "loaded_models": loaded_model_keys(),
                "spill": _spill_describe(),
            }
        )

    @app.route("/infer", methods=["POST"])
    def infer():
        payload = request.get_json(silent=True) or {}
        # Keep one request id from arrival through any gate wait so a queued
        # non-streaming request is visible and can be cancelled just like a
        # streaming request.  Do not pass this worker bookkeeping field into
        # the runner itself.
        req_id = str(payload.pop("request_id", "") or uuid.uuid4().hex)
        # ROLLING AGGREGATE (operator ruling 2026-07-29): the worker aggregates
        # its OWN serve stats so central never has to poll for them. Captured
        # here, before _ensure_present can rewrite the payload, and recorded on
        # every exit path below. Pure arithmetic on a request already finished —
        # it adds no work to the serving path and cannot fail it (every
        # aggregate entry point swallows its own errors).
        _agg_key = payload.get("model_key")
        _agg_task = payload.get("task")
        _agg_t0 = time.time()
        # Errors as DATA, never a raw Flask 500: the raw error page hides the
        # worker-side traceback from central entirely (2026-07-03: three
        # opaque delegation failures in one day were undiagnosable from
        # central). A 500 with a JSON body rides back through the delegating
        # runner's error path, so the console shows the REAL cause.
        try:
            request_spill = payload.pop("spill", None)
            # Never serve a model nobody asked for (see _model_key_refusal):
            # an absent/unknown key is a 400 here, not a default stand-in.
            _refusal = _model_key_refusal(payload, state.central_url)
            if _refusal:
                _aggregate.record_serve(
                    _agg_key, ok=False,
                    latency_ms=(time.time() - _agg_t0) * 1000.0,
                    error=f"BadRequest: {_refusal}", task=_agg_task)
                return jsonify({
                    "ok": False, "error": _refusal,
                    "worker": {"id": state.worker_id, "name": state.name},
                }), 400
            # Register before waiting for the per-model gate. Queued requests
            # must be visible and cancellable, not anonymous threads that turn
            # into failures after an arbitrary timeout.
            from hugpy_control.jobs import job_store
            queued_cancel = threading.Event()
            try:
                existing = job_store.get(req_id)
                if existing is None or existing.terminal:
                    job_store.create(str(_agg_key or ""), id=req_id,
                                     kind="chat", transport="worker")
                job_store.attach_cancel(req_id, queued_cancel.set)
            except Exception:
                pass
            _ensure_present(payload, state.central_url, state=state)
            # Per-model generation gate: serialize entry into an in-process
            # (llama.cpp/transformers) runner so concurrent /infer calls can't
            # race the same non-reentrant native context and crash the worker.
            # No-op for a slot-backed model (its child schedules itself). On a
            # bounded-wait timeout this raises ModelBusy -> honest 503 below.
            gate_token = gen_gate.acquire_for_payload(
                payload, cancel_event=queued_cancel)
            try:
                _prepare_load_contract(state, payload.get("model_key"),
                                       request_spill)
                _apply_spill(request_spill)
                # On-demand comfy start (worker owns comfy's lifecycle): evict-to-
                # fit FIRST, then start the managed comfy and wait for readiness.
                # No-op for non-comfy models and for the 'external' launcher. A
                # start failure raises ModelLoadFailure -> the except below ships
                # its structured load_failure to central.
                _ensure_comfy_started(state, payload.get("model_key"))
                result = _run_once(payload)
            finally:
                gate_token.release()
            try:
                job_store.finish(req_id,
                                 error=None if result.get("ok", True)
                                 else result.get("error"))
            except Exception:
                pass
            _aggregate.record_serve(
                _agg_key, ok=bool(result.get("ok", True)),
                latency_ms=(time.time() - _agg_t0) * 1000.0,
                tokens_out=_aggregate.tokens_out_of(result),
                error=None if result.get("ok", True) else result.get("error"),
                task=_agg_task)
            return jsonify(result)
        except gen_gate.ModelCancelled:
            try:
                job_store.cancel(req_id, reason="cancelled while queued")
            except Exception:
                pass
            return jsonify({"ok": False, "cancelled": True,
                            "request_id": req_id}), 499
        except gen_gate.ModelBusy as busy:
            # Honest structured busy — the runner is at capacity, not broken.
            # Recorded as a failure with its own verbatim reason: "the box was
            # at capacity" is precisely the pool-health fact this file exists
            # to surface.
            _aggregate.record_serve(
                _agg_key, ok=False,
                latency_ms=(time.time() - _agg_t0) * 1000.0,
                error=f"ModelBusy: {busy}", task=_agg_task)
            return jsonify(busy.as_error(
                {"id": state.worker_id, "name": state.name})), 503
        except BudgetRefusal as exc:
            # The model cannot fit on this box even after a full FIFO. NOT a
            # crash and NOT a traceback: a storage-capacity verdict, so it gets
            # its own honest code (507 Insufficient Storage) and the structured
            # reason. Central can then route elsewhere instead of retrying a
            # box that will never have room.
            _aggregate.record_serve(
                _agg_key, ok=False,
                latency_ms=(time.time() - _agg_t0) * 1000.0,
                error=f"BudgetRefusal: {exc.reason.get('reason')}", task=_agg_task)
            return jsonify({
                "ok": False,
                "error": exc.reason.get("reason"),
                "refused": exc.reason,
                "worker": {"id": state.worker_id, "name": state.name},
            }), 507
        except Exception as exc:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc()
            logger.error("infer failed: %s", tb)
            # VERBATIM in the aggregate — the operator's standing want. A
            # paraphrased error has repeatedly cost a diagnosis, so the rolling
            # file keeps the real text (bounded, elision marked).
            _aggregate.record_serve(
                _agg_key, ok=False,
                latency_ms=(time.time() - _agg_t0) * 1000.0,
                error=f"{type(exc).__name__}: {exc}", task=_agg_task)
            return jsonify(_with_load_failure({
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                # whole traceback (2026-09-23); the old key name is kept
                "traceback_tail": tb,
                # Attribution at the source: direct API consumers (and any
                # relay that keeps the body) see WHICH box failed without
                # having to know who they called.
                "worker": {"id": state.worker_id, "name": state.name},
            }, exc)), 500

    @app.route("/infer/stream", methods=["POST"])
    def infer_stream():
        payload = request.get_json(silent=True) or {}
        request_spill = payload.pop("spill", None)
        # Caller-supplied id for cancellation; else generate one. Echo it back
        # as the first SSE event so the client can cancel this exact request.
        req_id = str(payload.pop("request_id", "") or uuid.uuid4().hex)

        # Per-model generation gate, acquired BEFORE the streaming Response so a
        # busy in-process runner is refused with a real HTTP 503 (not a mid-body
        # SSE surprise). The bounded wait blocks here (that IS the honest queue).
        # The token is then held for the WHOLE life of the stream and released in
        # the generator's finally — a streamed response occupies the runner until
        # its last token. No-op token for a slot-backed model. See gen_gate.
        _agg_key = payload.get("model_key")
        _agg_task = payload.get("task")
        _agg_t0 = time.time()
        # Same refusal as /infer, and BEFORE the Response: a stream that has
        # already started can only report this as a mid-body SSE surprise, and
        # the caller would have no status code to route on. The registration
        # lookup this performs is the one _ensure_present_streaming does first
        # anyway (idempotent), so it costs the stream nothing.
        _refusal = _model_key_refusal(payload, state.central_url)
        if _refusal:
            _aggregate.record_serve(
                _agg_key, ok=False,
                latency_ms=(time.time() - _agg_t0) * 1000.0,
                error=f"BadRequest: {_refusal}", task=_agg_task)
            return jsonify({
                "ok": False, "error": _refusal,
                "worker": {"id": state.worker_id, "name": state.name},
            }), 400
        # The gate wait is part of the request, not a pre-request limbo. Make
        # it observable and cancellable before blocking. _stream_sync replaces
        # this threading.Event callback with its asyncio callback once the gate
        # is acquired and generation begins.
        from hugpy_control.jobs import job_store
        queued_cancel = threading.Event()
        try:
            existing = job_store.get(req_id)
            if existing is None or existing.terminal:
                job_store.create(str(_agg_key or ""), id=req_id,
                                 kind="chat", transport="worker")
            job_store.attach_cancel(req_id, queued_cancel.set)
        except Exception:
            pass
        try:
            gate_token = gen_gate.acquire_for_payload(
                payload, cancel_event=queued_cancel)
        except gen_gate.ModelCancelled:
            try:
                job_store.cancel(req_id, reason="cancelled while queued")
            except Exception:
                pass
            return jsonify({"ok": False, "cancelled": True,
                            "request_id": req_id}), 499
        except gen_gate.ModelBusy as busy:
            try:
                job_store.finish(req_id, error=busy)
            except Exception:
                pass
            _aggregate.record_serve(
                _agg_key, ok=False,
                latency_ms=(time.time() - _agg_t0) * 1000.0,
                error=f"ModelBusy: {busy}", task=_agg_task)
            return jsonify(busy.as_error(
                {"id": state.worker_id, "name": state.name})), 503

        try:
            _prepare_load_contract(state, payload.get("model_key"),
                                   request_spill)
            _apply_spill(request_spill)
        except Exception as exc:  # before Response: retain a real HTTP error
            gate_token.release()
            return jsonify(_with_load_failure(
                {"ok": False,
                 "error": f"{type(exc).__name__}: {exc}",
                 "worker": {"id": state.worker_id,
                            "name": state.name}}, exc)), 500

        def _generate():
            # A stream's outcome is only known in the generator's finally — the
            # same place the gate token is released, and for the same reason: a
            # streamed response occupies the runner (and can fail, or be
            # abandoned by the client) right up to its last token. Recording
            # anywhere earlier would book a success the worker had not yet had.
            _agg_err = None
            try:
                yield _sse({"type": "request", "request_id": req_id})
                # Stream provisioning progress first (download from central/HF),
                # then generation with auto-continuation. Both emit SSE lines.
                yield from _ensure_present_streaming(payload, state.central_url,
                                                     state=state)
                yield from _stream_sync(payload, request_id=req_id)
            except BaseException as exc:   # noqa: BLE001 — re-raised below
                _agg_err = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                # Release on normal end, error, OR client disconnect (Flask closes
                # the generator) — the gate must never leak a permit.
                gate_token.release()
                # tokens_out is deliberately NOT counted for a stream: the token
                # total is not stated on this path, and an estimate in a health
                # file is worse than an honest absence.
                _aggregate.record_serve(
                    _agg_key, ok=_agg_err is None,
                    latency_ms=(time.time() - _agg_t0) * 1000.0,
                    error=_agg_err, task=_agg_task)

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
        # Same substrate as central (F5): the job's attached cancel handle sets
        # the stream's Event on the shared runtime loop via call_soon_threadsafe
        # (a bare cross-thread Event.set() is unsafe — wakes futures on another
        # loop). Wire contract unchanged: 404 for unknown/finished requests.
        from hugpy_control.jobs import job_store
        job = job_store.get(request_id)
        if job is None or job.terminal:
            return jsonify({"cancelled": False, "reason": "unknown or finished request"}), 404
        job_store.cancel(request_id, reason="cancelled via /infer/cancel")
        return jsonify({"cancelled": True, "request_id": request_id})

    # -- privileged ops (F3.4 control agent; CON-05/06 + UTIL-02) ----------
    # Central relays these through its operator gate + audit log. Every
    # response is typed data ({ok, error:{code,message}}), never a traceback.

    @app.route("/ops/heartbeat-nudge", methods=["POST"])
    def ops_heartbeat_nudge():
        # Wake the beat loop early (slot inflight edge — see _HB_WAKE). Cheap
        # and idempotent: worst case an extra debounced beat, so it needs no
        # auth beyond LAN reachability, same as every /ops verb here.
        _HB_WAKE.set()
        activity = request.get_json(silent=True) or {}
        if (state.worker_id and activity.get("slot_id") is not None
                and isinstance(activity.get("busy"), bool)
                and isinstance(activity.get("observed_at"), (int, float))):
            from hugpy_fleet.central import heartbeat_db
            threading.Thread(
                target=heartbeat_db.record_activity,
                args=(state.worker_id, activity["slot_id"], activity.get("model_key"),
                      activity["busy"], activity["observed_at"]),
                daemon=True, name="slot-activity").start()
        return jsonify({"ok": True})

    @app.route("/ops/restart", methods=["POST"])
    def ops_restart():
        # Respond first, then restart: the caller needs the ack before the
        # process cycles. Under systemd this EXITS (Restart= respawns a fresh,
        # cgroup-tracked process — never the os.execv orphan that squatted :9100
        # and restart-looped); standalone re-execs in place. Persistent worker-id
        # -> same registry row.
        _schedule_restart(state, "ops/restart")
        return jsonify({"ok": True, "restarting": True,
                        "worker_id": state.worker_id})

    @app.route("/ops/free-ram", methods=["POST"])
    def ops_free_ram():
        # NON-destructive host-RAM reclaim: return glibc's orphaned allocator
        # arena (and torch's CUDA cache) to the OS WITHOUT evicting any model.
        # After a model is freed malloc keeps the pages pooled so RSS stays
        # pinned (ae observed at 0 free / 128 GB used, nothing loaded);
        # malloc_trim(0) hands them back. loaded_models is reported UNCHANGED —
        # Unload is the destructive path; this one never touches residency.
        ram_before = _free_ram_bytes()
        rss_before = _agent_rss_bytes()
        _trim_host_ram()
        ram_after = _free_ram_bytes()
        rss_after = _agent_rss_bytes()
        ram_freed = (ram_after - ram_before) if (
            ram_before is not None and ram_after is not None) else None
        return jsonify({
            "ok": True,
            "ram_free_before": ram_before,
            "ram_free_after": ram_after,
            "ram_freed": ram_freed,
            "rss_before": rss_before,
            "rss_after": rss_after,
            "loaded_models": loaded_model_keys(),
        })

    @app.route("/ops/update", methods=["POST"])
    def ops_update():
        # CON-05 on demand: same converge path as the heartbeat handshake
        # (pip install pinned target from PyPI or --pkg-index, then re-exec),
        # minus the wait and the retry backoff — the operator asked NOW.
        args = getattr(state, "args", None)
        if args is None:
            return jsonify({"ok": False, "error": {
                "code": "NoArgs", "message": "agent started without CLI args "
                "context; update unavailable"}}), 501
        body = request.get_json(silent=True) or {}
        # Central can bootstrap a worker onto its private PEP-503 index during
        # an on-demand update.  Older deployments had to carry
        # WORKER_PKG_INDEX in the unit before this endpoint could be used.
        pkg_index = str(body.get("pkg_index") or "").strip()
        if pkg_index and not getattr(args, "pkg_index", None):
            args.pkg_index = pkg_index
        target = str(body.get("version") or "").strip()
        if not target:
            return jsonify({"ok": False, "error": {
                "code": "NoVersion",
                "message": 'body must include {"version": "x.y.z"} '
                           '(central sends its required_pkg_version)'}}), 400
        # Same lockstep converge as the heartbeat self-update (constraints.txt
        # from central, siblings pinned; --no-deps fallback when unreachable).
        # Central may name its constraints URL; otherwise derived from --central.
        constraints_hint = str(body.get("constraints_url") or "").strip() or None
        # Central's own index as an EXTRA source (heartbeat's pkg_index_url);
        # unlike ``pkg_index`` above it does not replace PyPI.
        index_hint = str(body.get("pkg_index_url") or "").strip() or None
        cmd, constraints_path = _prepare_converge(args, target, constraints_hint,
                                                  pkg_index_url=index_hint)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=560)
            # Whole output (2026-09-23): output_tail kept as the key name for
            # older consumers; it now carries every byte pip wrote.
            rc, tail = proc.returncode, (proc.stdout + proc.stderr)
        except Exception as exc:
            return jsonify({"ok": False, "error": {
                "code": type(exc).__name__, "message": str(exc)}}), 502
        finally:
            _discard_constraints(constraints_path)
        if rc == 0:
            # kill_slots: a fresh version is installed — any orphaned slot child
            # would keep serving the OLD code (the adoption probe can't tell
            # versions apart), so tear them down and let the fresh agent respawn
            # them on the new code. Same discipline as the heartbeat self-update.
            _schedule_restart(state, "ops/update", kill_slots=True)
            return jsonify({"ok": True, "installed": f"{args.pkg_name}=={target}",
                            "constrained": constraints_path is not None,
                            "restarting": True})
        return jsonify({"ok": False, "error": {
            "code": "PipFailed", "message": f"pip rc={rc}", "detail": tail}}), 502

    @app.route("/ops/aggregate", methods=["GET"])
    def ops_aggregate():
        """Serve the ROLLING AGGREGATE — central pulls this ON READ.

        The whole point of the 2026-07-29 ruling: central asks ONCE, when a
        human is actually looking, and gets everything the worker has been
        accumulating for itself — instead of fanning per-model detail polls at
        the box on a timer and starving its heartbeat.

        Same trust model as every other /ops/* route (central's relay + its
        operator gate + audit; the worker trusts the WireGuard link). GET and
        read-only: it can neither load nor evict anything.

        Flushes first so a reader gets the current numbers rather than whatever
        the debounce last wrote — one small file write, on a request a human
        made. Falls back to the live in-memory document if the file can't be
        read (a flush race must degrade to facts, not a 404)."""
        agg = _aggregate.get_aggregate()
        agg.maybe_flush(force=True)
        doc = agg.read_file() or agg.document()
        return jsonify({"ok": True, "aggregate": doc,
                        "summary": agg.heartbeat_summary()})

    @app.route("/ops/environment", methods=["GET"])
    def ops_environment():
        # k118: this box SELF-REPORTS its environment. Central cannot ssh to
        # every worker (ae is the canonical example), so anything it cannot ask
        # for over HTTP it cannot know — and an environment it cannot know is
        # one it discovers at runtime, which is how a-brain lost ASR to a
        # missing ffmpeg and computron lost 4-bit to a missing bitsandbytes.
        #
        # Read-only and TTL-cached (10 min) inside the report module, so this is
        # cheap at page-load frequency; ``?refresh=1`` is the operator override.
        # Same trust model as every other /ops/* route (central's relay).
        from hugpy_fleet.worker.environment_report import environment_report
        refresh = str(request.args.get("refresh") or "").lower() in (
            "1", "true", "yes")
        try:
            report = environment_report(refresh=refresh, worker_name=state.name)
        except Exception as exc:  # noqa: BLE001 — a report never 500s the box
            return jsonify({"ok": False, "error": {
                "code": type(exc).__name__, "message": str(exc)}}), 502
        return jsonify(report)

    @app.route("/ops/pip", methods=["POST"])
    def ops_pip():
        # UTIL-02: install into this worker's env. Argv-list (no shell), rc +
        # output tail returned as data. The operator gate + audit live on
        # central; this endpoint trusts central's relay like every other op.
        body = request.get_json(silent=True) or {}
        pkg = str(body.get("package") or "").strip()
        if not pkg or pkg.startswith("-"):
            return jsonify({"ok": False, "error": {
                "code": "BadPackage",
                "message": 'body must include {"package": "name==ver"} '
                           "(flags are not accepted)"}}), 400
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg],
                capture_output=True, text=True, timeout=560)
            # Whole output (2026-09-23): output_tail kept as the key name for
            # older consumers; it now carries every byte pip wrote.
            rc, tail = proc.returncode, (proc.stdout + proc.stderr)
        except Exception as exc:
            return jsonify({"ok": False, "error": {
                "code": type(exc).__name__, "message": str(exc)}}), 502
        # A llama-cpp-python (re)install changes the engine's GPU-offload
        # capability, but _LLAMA_PROBE_CACHE memoizes the FIRST probe for the
        # whole process life — so /health + heartbeat caps would keep reporting
        # the OLD build's supports_gpu_offload until a full re-exec (/ops/pip
        # never re-execs; /ops/update does). Invalidate the cache so the next
        # probe reflects the freshly-installed build honestly.
        if rc == 0 and "llama" in pkg.lower():
            global _LLAMA_PROBE_CACHE
            _LLAMA_PROBE_CACHE = None
        return (jsonify({"ok": rc == 0, "package": pkg, "rc": rc,
                         "output_tail": tail}), 200 if rc == 0 else 502)

    @app.route("/ops/config", methods=["POST", "GET"])
    def ops_config():
        # Daylight item 3: operator serving-config, persisted in the agent's
        # OWN settings file (beats env/drop-ins — see _apply_settings_env).
        # GET returns current settings + effective values; POST merges the
        # supported keys, persists, and re-execs to apply cleanly (persistent
        # worker-id -> same registry row; ~seconds of blip).
        args = getattr(state, "args", None)
        if args is None:
            return jsonify({"ok": False, "error": {
                "code": "NoArgs", "message": "agent started without CLI args"}}), 501
        if request.method == "GET":
            return jsonify({"ok": True, "settings": _load_settings(args),
                            "effective": _effective_config()})
        body = request.get_json(silent=True) or {}
        unknown = sorted(set(body) - _SETTINGS_KEYS)
        if unknown:
            # A fleet-policy key gets its own code + the address of the route
            # that DOES own it. A bare "unsupported" would be true but useless:
            # the operator asked for a real thing at the wrong door, and telling
            # them only that the door is locked makes them guess.
            # A RETIRED key gets its own code and says what replaced it. The
            # operator (or an old script) is asking for a lever that no longer
            # exists; "unsupported" would read as a typo rather than a removal.
            retired = [k for k in unknown if k in _RETIRED_SETTINGS]
            if retired:
                return jsonify({"ok": False, "error": {
                    "code": "RetiredKey",
                    "message": "; ".join(
                        f"'{k}' was RETIRED — {_RETIRED_SETTINGS[k]}"
                        for k in retired),
                    "keys": retired}}), 400
            fleet = [k for k in unknown if k in _FLEET_ONLY_SETTINGS]
            if fleet:
                return jsonify({"ok": False, "error": {
                    "code": "FleetPolicyKey",
                    "message": "; ".join(
                        f"'{k}' is FLEET policy, not a per-worker setting — set "
                        f"it at {_FLEET_ONLY_SETTINGS[k]}" for k in fleet),
                    "keys": fleet}}), 400
            return jsonify({"ok": False, "error": {
                "code": "UnknownKeys",
                "message": f"unsupported: {unknown}; supported: {sorted(_SETTINGS_KEYS)}"}}), 400
        settings = _load_settings(args)
        if "slot_count" in body:
            try:
                n = int(body["slot_count"])
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": {
                    "code": "BadValue", "message": "slot_count must be an integer"}}), 400
            if not 0 <= n <= 16:
                return jsonify({"ok": False, "error": {
                    "code": "BadValue", "message": "slot_count must be 0..16"}}), 400
            settings["slot_count"] = n
        if "on_demand_ttl_s" in body:
            # OPT-IN idle reclamation (doctrine 2026-07-11). The DEFAULT residency
            # trigger is memory contention, not a clock — so on_demand_ttl_s is
            # ABSENT by default and null/0 CLEARS it (idle sweep off; contention
            # alone governs residency; the heartbeat then reports it as null).
            val = body["on_demand_ttl_s"]
            if val in (None, "", 0, "0"):
                settings.pop("on_demand_ttl_s", None)
            else:
                try:
                    tval = int(val)
                except (TypeError, ValueError):
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": "on_demand_ttl_s must be an integer, or null to disable idle reclamation"}}), 400
                if not 60 <= tval <= 86400:
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": "on_demand_ttl_s must be 60..86400 (or null to disable idle reclamation)"}}), 400
                settings["on_demand_ttl_s"] = tval
        if "reconcile_interval_s" in body:
            try:
                tval = int(body["reconcile_interval_s"])
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": {
                    "code": "BadValue", "message": "reconcile_interval_s must be an integer"}}), 400
            if not 60 <= tval <= 86400:
                return jsonify({"ok": False, "error": {
                    "code": "BadValue", "message": "reconcile_interval_s must be 60..86400"}}), 400
            settings["reconcile_interval_s"] = tval
        if "residency" in body:
            # DEEP-MERGE per model key: {"model": "static"|null}. The default
            # tier is ON-DEMAND and is represented by NO stored entry — null,
            # "" or "on-demand" clear the override. "static" is the only
            # stored value.
            if not isinstance(body["residency"], dict):
                return jsonify({"ok": False, "error": {
                    "code": "BadValue",
                    "message": 'residency must be {"<model_key>": "static"|null} — null/"on-demand" restores the on-demand default'}}), 400
            merged = dict(settings.get("residency") or {})
            for mk, mode in body["residency"].items():
                if mode in (None, "", "on-demand"):
                    # on-demand IS the default — storing it would be noise;
                    # any of these writes clears the override.
                    merged.pop(mk, None)
                elif mode == "static":
                    merged[mk] = mode
                else:
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"residency[{mk!r}] must be 'static' or null — on-demand is the default ('on-demand'/null clear the override)"}}), 400
            if merged:
                settings["residency"] = merged
            else:
                settings.pop("residency", None)
        if "pinned" in body:
            # Files-axis pin (tiers v2): {"model": true|null}. Deep-merged.
            if not isinstance(body["pinned"], dict):
                return jsonify({"ok": False, "error": {
                    "code": "BadValue",
                    "message": 'pinned must be {"<model_key>": true|null}'}}), 400
            pmerged = dict(settings.get("pinned") or {})
            for mk, val in body["pinned"].items():
                if val in (None, False, ""):
                    pmerged.pop(mk, None)
                elif val is True:
                    pmerged[mk] = True
                else:
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"pinned[{mk!r}] must be true or null"}}), 400
            if pmerged:
                settings["pinned"] = pmerged
            else:
                settings.pop("pinned", None)
        if "ctx_pct" in body:
            # Per-model CONTEXT allocation (slice 11 / t27): {"model": 1..100 | null}
            # — percent of the model's max context reserved for KV in fit/serving.
            # Deep-merged, same shape as residency/pinned. null clears (default ctx).
            if not isinstance(body["ctx_pct"], dict):
                return jsonify({"ok": False, "error": {
                    "code": "BadValue",
                    "message": 'ctx_pct must be {"<model_key>": 1..100 | null}'}}), 400
            cmerged = dict(settings.get("ctx_pct") or {})
            for mk, val in body["ctx_pct"].items():
                # ONLY an explicit null/"" clears (default ctx). Do NOT treat 0/
                # False as a clear: `0 in (None, "", False)` is True in Python
                # (0 == False), which would silently clear on an out-of-range 0
                # instead of rejecting it — so match None/"" explicitly.
                if val is None or val == "":
                    cmerged.pop(mk, None)
                    continue
                try:
                    pv = int(val)
                except (TypeError, ValueError):
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"ctx_pct[{mk!r}] must be an integer 1..100 or null"}}), 400
                if not (1 <= pv <= 100):
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"ctx_pct[{mk!r}]={pv} out of range — must be 1..100"}}), 400
                cmerged[mk] = pv
            if cmerged:
                settings["ctx_pct"] = cmerged
            else:
                settings.pop("ctx_pct", None)
        # t21 tolerance bands + priority (per-model maps, siblings of ctx_pct).
        # A deviation is a percent-of-total tolerance (0..100; 0 == no band);
        # priority is a non-negative integer (0 == normal). Same deep-merge /
        # null-clears shape as ctx_pct. Additive + optional so a released worker
        # that doesn't know these keys simply ignores them (no schema forbid on
        # /ops/config) — the relay-wire landmine does not apply here.
        for _band_key, _lo, _hi, _isint in (
                ("ctx_deviation_pct", 0, 100, False),
                ("vram_deviation_pct", 0, 100, False),
                ("ram_deviation_pct", 0, 100, False),
                ("priority", 0, None, True)):
            if _band_key not in body:
                continue
            if not isinstance(body[_band_key], dict):
                return jsonify({"ok": False, "error": {
                    "code": "BadValue",
                    "message": f'{_band_key} must be {{"<model_key>": number | null}}'}}), 400
            _merged = dict(settings.get(_band_key) or {})
            for mk, val in body[_band_key].items():
                # Only explicit null/"" clears (see ctx_pct: 0 == False in Python
                # so match None/"" exactly rather than truthiness).
                if val is None or val == "":
                    _merged.pop(mk, None)
                    continue
                try:
                    pv = int(val) if _isint else float(val)
                except (TypeError, ValueError):
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"{_band_key}[{mk!r}] must be a number or null"}}), 400
                if pv < _lo or (_hi is not None and pv > _hi):
                    hi_txt = _hi if _hi is not None else "∞"
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"{_band_key}[{mk!r}]={pv} out of range — must be {_lo}..{hi_txt}"}}), 400
                _merged[mk] = pv
            if _merged:
                settings[_band_key] = _merged
            else:
                settings.pop(_band_key, None)
        if "comfy_url" in body:
            # Adopted-ComfyUI base URL (probe + job submission read it as the
            # COMFY_URL env). Settable here so central/console can point a worker
            # at its ComfyUI without a systemd drop-in. null/"" clears it (falls
            # back to env / the 127.0.0.1:8188 default).
            cu = body["comfy_url"]
            if cu in (None, ""):
                settings.pop("comfy_url", None)
            elif isinstance(cu, str) and _valid_comfy_url(cu):
                settings["comfy_url"] = cu.strip().rstrip("/")
            else:
                return jsonify({"ok": False, "error": {
                    "code": "BadValue",
                    "message": "comfy_url must be an http(s) URL with a host "
                               "(e.g. https://comfy.example.ai), or null"}}), 400
        if "hot_cache_root" in body:
            # Per-worker attribution of the HOT-CACHE tier's root (the box-local
            # NVMe LRU cache of the main catalog — managers/serve/hot_cache.py,
            # which reads HUGPY_HOT_CACHE_ROOT live). This ONLY names WHERE the
            # tier lives on this box (e.g. ae -> /mnt/hot990/hugpy-hot-cache); the
            # tier stays an automatic LRU cache and the SHARED store stays the
            # source of truth. null/"" CLEARS it (revert to the env base, else the
            # tier is off) — same idiom as on_demand_ttl_s. A set value is checked
            # SHAPE-ONLY (must be an absolute path), mirroring comfy_url's URL
            # check: a not-yet-mounted root is accepted here and hot_cache.enabled()
            # disables the tier gracefully until it exists, so config never blocks
            # on a mount that is about to appear.
            hcr = body["hot_cache_root"]
            if hcr in (None, ""):
                settings.pop("hot_cache_root", None)
            elif isinstance(hcr, str) and os.path.isabs(hcr.strip()):
                settings["hot_cache_root"] = hcr.strip().rstrip("/") or "/"
            else:
                return jsonify({"ok": False, "error": {
                    "code": "BadValue",
                    "message": "hot_cache_root must be an absolute path "
                               "(e.g. /mnt/hot990/hugpy-hot-cache), or null to clear"}}), 400
        if "profiles" in body:
            # Env-profiles (stage 1): {"<name>": {"packages": [str,...]} | null}.
            # DEEP-MERGE per profile; null/{}/"" clears one. Names slug-safe;
            # packages a NON-EMPTY list of non-empty strings. A profile = a named
            # venv (materialized in the background at boot — see main()) that a
            # profiled model's SLOT CHILD launches from, isolating extra deps from
            # the shared venv. The agent itself never installs into it.
            from hugpy_engine.serve import profiles as _profiles_mod
            if not isinstance(body["profiles"], dict):
                return jsonify({"ok": False, "error": {
                    "code": "BadValue",
                    "message": 'profiles must be {"<name>": {"packages": [str,...]}} '
                               "(or null per name to clear)"}}), 400
            pmerged = dict(settings.get("profiles") or {})
            for name, spec in body["profiles"].items():
                if not _profiles_mod.slug_ok(name):
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"profile name {name!r} must be slug-safe "
                                   "(letters/digits then . _ - , max 64 chars)"}}), 400
                if spec in (None, {}, ""):
                    pmerged.pop(name, None)
                    continue
                if not isinstance(spec, dict):
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"profiles[{name!r}] must be an object with a "
                                   "packages list, or null to clear"}}), 400
                pkgs = spec.get("packages")
                if (not isinstance(pkgs, list) or not pkgs
                        or not all(isinstance(p, str) and p.strip() for p in pkgs)):
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"profiles[{name!r}].packages must be a non-empty "
                                   "list of non-empty strings"}}), 400
                pmerged[name] = {"packages": [p.strip() for p in pkgs]}
            if pmerged:
                settings["profiles"] = pmerged
            else:
                settings.pop("profiles", None)
        if "model_profiles" in body:
            # Model->profile ATTRIBUTION (stage 1): {"<model_key>": "<name>" |
            # null}. DEEP-MERGE; null/"" clears an attribution. The name is
            # slug-safe but need NOT already exist (the two keys can arrive in
            # either order across relay calls; a dangling attribution reports
            # honestly and refuses to seat until the profile is declared+ready).
            from hugpy_engine.serve import profiles as _profiles_mod
            if not isinstance(body["model_profiles"], dict):
                return jsonify({"ok": False, "error": {
                    "code": "BadValue",
                    "message": 'model_profiles must be {"<model_key>": "<profile_name>"|null}'}}), 400
            mmerged = dict(settings.get("model_profiles") or {})
            for mk, pname in body["model_profiles"].items():
                if pname in (None, ""):
                    mmerged.pop(mk, None)
                elif _profiles_mod.slug_ok(pname):
                    mmerged[mk] = pname
                else:
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"model_profiles[{mk!r}] must be a slug-safe "
                                   "profile name, or null to clear"}}), 400
            if mmerged:
                settings["model_profiles"] = mmerged
            else:
                settings.pop("model_profiles", None)
        _save_settings(args, settings)
        logger.info("ops/config: persisted %s — restarting to apply", settings)
        # Restart to re-project the settings over a clean base. Under systemd this
        # EXITS and the fresh process gets the UNIT's env as the true base (so the
        # env base-sentinels are unnecessary in that path); standalone re-execs
        # and the sentinels carry the pre-projection base across the exec. Both
        # lifecycles are documented at _apply_settings_env.
        _schedule_restart(state, "ops/config apply")
        return jsonify({"ok": True, "settings": settings, "restarting": True})

    @app.route("/probe/<path:model_key>", methods=["POST", "GET"])
    def probe(model_key):
        # THE LOAD HALF (operator ruling 2026-09-24): a live VRAM-fit check that
        # actually LOADS the model on this worker's GPU and reports whether it
        # fit, plus before/after free VRAM. It pulls-if-absent as part of loading
        # (fetch-then-load), so it remains the download+load combo for callers
        # that want both; the fetch-only primitive is POST /models/fetch (disk,
        # no VRAM). Loading is cached by dispatch, so a probe also warms the model
        # for the first real chat.
        #
        # Optional POST body: {"spill": {...}} — TASK C (2026-07-25): the path
        # central's explicit /load relay (workers_load, the operator's ▶ activate)
        # uses to seat an explicit
        # n_gpu_layers/n_cpu_moe (e.g. re-seating a crashed MoE slot with a
        # split instead of repeating the ngl=-1-no-split stall). Applied via
        # the SAME _apply_spill /infer already uses, so it takes effect before
        # the runner is built (the model loads lazily on first access below).
        # GET carries no body (a probe with no override, today's behavior
        # unchanged); an absent/empty body on POST is likewise a no-op spill.
        body = request.get_json(silent=True) or {}
        _apply_spill(body.get("spill"))
        return jsonify(_probe_model(model_key, state))

    @app.route("/models/unload", methods=["POST"])
    def unload():
        # Free GPU VRAM by evicting cached runner(s) from this worker's dispatch
        # cache. Body: {"model_key": ...} drops one model; {} or {"all": true}
        # drops everything loaded. The model stays ASSIGNED (central's registry
        # is untouched) — it just isn't held in VRAM until the next request.
        body = request.get_json(silent=True) or {}
        model_key = body.get("model_key")
        before = _free_vram_bytes()
        ram_before = _free_ram_bytes()
        err = None
        try:
            from hugpy_engine.dispatch.dispatch import evict as _evict, clear as _clear
            if model_key and not body.get("all"):
                evicted = bool(_evict(model_key))
            else:
                _clear()
                evicted = True
        except Exception as exc:
            evicted, err = False, f"{type(exc).__name__}: {exc}"
        # The evict/clear above drops references, but glibc keeps the freed
        # weights' host arena pooled (RSS stays pinned) — trim hands it, and
        # torch's CUDA cache, back to the OS so this destructive path returns
        # host RAM too, not just VRAM.
        _trim_host_ram()
        after = _free_vram_bytes()
        ram_after = _free_ram_bytes()
        freed = (after - before) if (before is not None and after is not None) else None
        ram_freed = (ram_after - ram_before) if (
            ram_before is not None and ram_after is not None) else None
        return jsonify({
            "ok": err is None,
            "evicted": evicted,
            "model_key": model_key,
            "error": err,
            "vram_free_before": before,
            "vram_free_after": after,
            "freed": freed,
            "ram_free_before": ram_before,
            "ram_free_after": ram_after,
            "ram_freed": ram_freed,
            "loaded_models": loaded_model_keys(),
        })

    @app.route("/ops/evict", methods=["POST"])
    def ops_evict():
        # Targeted eviction: free ONE model's RAM+VRAM, picking the mechanism by
        # how that model is hosted (comfy /free, slot child kill, or in-process
        # ref-drop). Central sends {"model_key": ..., "force"?: bool} — NEVER a
        # PID (per-box, recycled): the worker resolves the model_key to its live
        # handle here and verifies identity before acting. Fail-safe: an unknown
        # or not-resident model_key is an idempotent no-op at HTTP 200, never a
        # 500. force=true overrides the static/pinned/in-flight gate. Contrast
        # /models/unload (coarse: one key or ALL, in-process/slot-proxy cache
        # only) — this is the surgical, host-mode-aware verb.
        body = request.get_json(silent=True) or {}
        model_key = body.get("model_key")
        force = bool(body.get("force"))
        try:
            return jsonify({"ok": True, **_evict_model(state, model_key, force)})
        except Exception as exc:  # noqa: BLE001 — evict must never 500 the control plane
            return jsonify({"ok": False, "model_key": model_key,
                            "host_mode": "unknown", "evicted": False,
                            "vram_freed": None, "ram_freed": None,
                            "reason": f"{type(exc).__name__}: {exc}"})

    @app.route("/ops/cache-evict", methods=["POST"])
    def ops_cache_evict():
        # DESIGNATE-FOR-REMOVAL: delete ONE model's copy from THIS worker's own
        # model store (the hot drive = the drive the models are on) and reclaim
        # its bytes. Reuses the reaper's vetted store-root path
        # (_store_root_copy_path) and the SAME safe delete gate (provision.
        # wipe_model): it re-proves the target is on this box's local, reapable
        # store and NEVER on shared/central storage (ae also hosts the central
        # drive — that copy is refused). Also drops any optional hot_cache tier
        # entry. Central sends {"model_key": ...}; returns the BYTE CONTRACT
        # {"freed_bytes": N, "removed": bool}. Idempotent no-op when nothing local.
        body = request.get_json(silent=True) or {}
        model_key = body.get("model_key")
        if not model_key:
            return jsonify({"ok": False, "freed_bytes": 0, "reason": "model_key required"})
        freed = 0
        removed = False
        try:
            from hugpy_fleet.worker.imports import get_model_config
            from hugpy_storage.provision import wipe_model
            try:
                cfg = get_model_config(model_key)
            except Exception:
                cfg = None
            # The copy on THIS worker's own store root (not get_model_path's
            # read-through, which on ae can resolve to the central NAS).
            try:
                path = _store_root_copy_path(model_key, cfg) or ""
            except Exception:
                path = ""
            if path:
                before = _path_bytes(path)
                if wipe_model(model_key, path):     # gate re-proved inside
                    freed += int(before)
                    removed = True
            # Best-effort: also drop an optional hot_cache-tier copy, if any.
            try:
                from hugpy_engine.serve import hot_cache
                freed += int(hot_cache.evict_model(model_key) or 0)
            except Exception:
                pass
            return jsonify({"ok": True, "model_key": model_key,
                            "freed_bytes": int(freed), "removed": bool(removed or freed),
                            "on_disk_bytes": (0 if (removed or freed) else None)})
        except Exception as exc:  # noqa: BLE001 — evict must never 500 the control plane
            return jsonify({"ok": False, "model_key": model_key, "freed_bytes": int(freed),
                            "reason": f"{type(exc).__name__}: {exc}"})

    @app.route("/ops/reap-orphans", methods=["POST"])
    def ops_reap_orphans():
        # p27: kill ORPHANED GPU children this worker itself leaked (own-venv
        # llama-server whose slot claim cleared but whose process kept VRAM —
        # enumerable since c34199e as cuda_context/model_key-None rows, but
        # unevictable because every eviction verb keys on model_key). Central
        # relays this through its operator gate + audit log like every /ops/*;
        # the worker trusts central's relay (same idiom as /ops/pip).
        # Body: {"dry_run"?: bool} — DEFAULTS TO TRUE when absent (preview).
        # OPERATOR/CENTRAL-INVOKED ONLY — never wired to a loop/heartbeat
        # (doctrine, see _reap_gpu_orphans). Never 500s the control plane.
        body = request.get_json(silent=True) or {}
        dry_run = True if "dry_run" not in body else bool(body.get("dry_run"))
        try:
            return jsonify({"ok": True,
                            **_reap_gpu_orphans(state, dry_run=dry_run)})
        except Exception as exc:  # noqa: BLE001 — reap must never 500 the control plane
            return jsonify({"ok": False, "dry_run": dry_run, "results": [],
                            "reaped_count": 0, "term_failed_count": 0,
                            "skipped_count": 0, "reapable_vram_bytes": 0,
                            "error": f"{type(exc).__name__}: {exc}"})

    # ── external lease surface (gpu_lease, 2026-08-11; restored 2026-09-24) ──
    # A foreign batch job (OCR run, render sweep) registers here as a
    # pseudo-model: it appears in the pid registry with host_mode "external"
    # (measured VRAM, heartbeat-fed, visible to the eviction planner and to
    # central), and its supervisor's control URL lets _evict_model pause it on
    # real demand. The claim verb is the demand direction back: evict IDLE
    # managed models so the lease can retake the card. See
    # worker/external_residents.py and worker/gpu_lease.py.
    @app.route("/ops/external/register", methods=["POST"])
    def ops_external_register():
        body = request.get_json(silent=True) or {}
        model_key = str(body.get("model_key") or "").strip()
        if not model_key:
            return jsonify({"ok": False, "reason": "missing model_key"}), 400
        pid = body.get("pid")
        try:
            pid = int(pid) if pid is not None else None
        except (TypeError, ValueError):
            return jsonify({"ok": False, "reason": f"bad pid {pid!r}"}), 400
        vram_gib = body.get("vram_gib")
        try:
            vram_gib = float(vram_gib) if vram_gib is not None else None
        except (TypeError, ValueError):
            vram_gib = None
        evictable = body.get("evictable")
        if evictable is not None:
            evictable = bool(evictable)
        resume = body.get("resume")
        if resume is not None and resume not in ("enabled", "disabled"):
            return jsonify({"ok": False,
                            "reason": f"bad resume {resume!r} "
                                      "(enabled|disabled)"}), 400
        try:
            from hugpy_fleet.worker import external_residents as _extres
            rec = _extres.register(model_key, pid,
                                   control_url=body.get("control_url"),
                                   vram_gib=vram_gib, note=body.get("note"),
                                   evictable=evictable, resume=resume)
            if pid:
                from hugpy_fleet.worker import pid_registry as _pidreg
                _pidreg.record_launch(model_key, pid, "external",
                                      cmdline_hint=body.get("note"))
            return jsonify({"ok": True, "resident": rec})
        except Exception as exc:  # noqa: BLE001 — never 500 the control plane
            return jsonify({"ok": False,
                            "reason": f"{type(exc).__name__}: {exc}"})

    @app.route("/ops/external/unregister", methods=["POST"])
    def ops_external_unregister():
        body = request.get_json(silent=True) or {}
        model_key = str(body.get("model_key") or "").strip()
        if not model_key:
            return jsonify({"ok": False, "reason": "missing model_key"}), 400
        try:
            from hugpy_fleet.worker import external_residents as _extres
            from hugpy_fleet.worker import pid_registry as _pidreg
            was = _extres.unregister(model_key)
            _pidreg.forget(model_key)
            return jsonify({"ok": True, "was_registered": was})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False,
                            "reason": f"{type(exc).__name__}: {exc}"})

    @app.route("/ops/external/set", methods=["POST"])
    def ops_external_set():
        # Operator lever for the wildcard-process policy (2026-08-12):
        # {"model_key": ..., "evictable"?: bool, "resume"?: "enabled"|"disabled"}
        # Updates the worker-side registry (which the eviction paths enforce)
        # and best-effort forwards the change to the lease supervisor's own
        # control URL so its client-side copy — and therefore its heartbeat
        # re-registers and its resume behaviour — agree.
        body = request.get_json(silent=True) or {}
        model_key = str(body.get("model_key") or "").strip()
        if not model_key:
            return jsonify({"ok": False, "reason": "missing model_key"}), 400
        evictable = body.get("evictable")
        if evictable is not None:
            evictable = bool(evictable)
        resume = body.get("resume")
        if resume is not None and resume not in ("enabled", "disabled"):
            return jsonify({"ok": False,
                            "reason": f"bad resume {resume!r} "
                                      "(enabled|disabled)"}), 400
        if evictable is None and resume is None:
            return jsonify({"ok": False,
                            "reason": "nothing to set: pass evictable "
                                      "and/or resume"}), 400
        try:
            from hugpy_fleet.worker import external_residents as _extres
            rec = _extres.set_policy(model_key, evictable=evictable,
                                     resume=resume)
            if rec is None:
                return jsonify({"ok": False,
                                "reason": f"no external lease {model_key!r} "
                                          "registered"}), 404
            supervisor_acked = None
            url = (rec.get("control_url") or "").rstrip("/")
            if url:
                try:
                    import httpx
                    payload = {}
                    if evictable is not None:
                        payload["evictable"] = evictable
                    if resume is not None:
                        payload["resume"] = resume
                    r = httpx.post(url + "/set", json=payload, timeout=10.0)
                    supervisor_acked = (r.status_code == 200)
                except Exception:  # noqa: BLE001 — registry is authoritative
                    supervisor_acked = False
            return jsonify({"ok": True, "resident": rec,
                            "supervisor_acked": supervisor_acked})
        except Exception as exc:  # noqa: BLE001 — never 500 the control plane
            return jsonify({"ok": False,
                            "reason": f"{type(exc).__name__}: {exc}"})

    @app.route("/ops/external/claim", methods=["POST"])
    def ops_external_claim():
        # {"model_key": ..., "target_free_gib"?: float, "min_idle_s"?: float,
        #  "include_external"?: bool} — evict idle managed models until free
        # VRAM >= target, then report honestly. The supervisor launches ONLY on
        # reached=true and polls again later otherwise. Never blocks, never
        # forces, never launches. include_external (studio demand class) lets
        # the claim treat evictable PEER leases as last-resort candidates.
        body = request.get_json(silent=True) or {}
        model_key = str(body.get("model_key") or "").strip()
        if not model_key:
            return jsonify({"ok": False, "reason": "missing model_key"}), 400
        try:
            from hugpy_fleet.worker import external_residents as _extres
            rec = _extres.get(model_key) or {}
            tgt = body.get("target_free_gib")
            if tgt is None:
                tgt = rec.get("vram_gib")
            try:
                target_bytes = int(float(tgt) * 2**30) if tgt else 0
            except (TypeError, ValueError):
                return jsonify({"ok": False,
                                "reason": f"bad target_free_gib {tgt!r}"}), 400
            if target_bytes <= 0:
                return jsonify({"ok": False, "reason":
                                "no target: pass target_free_gib or register "
                                "with vram_gib first"}), 400
            mi = body.get("min_idle_s")
            try:
                mi = float(mi) if mi is not None else None
            except (TypeError, ValueError):
                mi = None
            include_external = bool(body.get("include_external"))
            return jsonify({"ok": True, "model_key": model_key,
                            **_external_claim_headroom(
                                state, model_key, target_bytes, min_idle_s=mi,
                                include_external=include_external)})
        except Exception as exc:  # noqa: BLE001 — claim must never 500
            return jsonify({"ok": False, "model_key": model_key,
                            "reached": False,
                            "reason": f"{type(exc).__name__}: {exc}"})

    @app.route("/ops/residents", methods=["GET"])
    def ops_residents():
        # The local residents view central's /llm/workers/<id>/external round-
        # trips to: measured VRAM residents joined with the dispatch LRU clock,
        # plus the external-lease records and the card's free/total. Read-only,
        # always 200.
        out = {"ok": True, "residents": [], "external": [],
               "last_used": {}, "free_vram": None, "total_vram": None}
        try:
            out["free_vram"] = _free_vram_bytes()
            out["total_vram"] = _total_vram_bytes()
        except Exception:  # noqa: BLE001
            pass
        try:
            lru = _dispatch_last_used()
            out["last_used"] = lru
            rows = _vram_residents(state)
            for r in rows:
                r = dict(r)
                r["last_used"] = lru.get(r.get("model_key"), 0.0)
                r["residency"] = _residency(r.get("model_key") or "")
                out["residents"].append(r)
        except Exception:  # noqa: BLE001
            pass
        try:
            from hugpy_fleet.worker import external_residents as _extres
            out["external"] = _extres.snapshot()
        except Exception:  # noqa: BLE001
            pass
        return jsonify(out)

    @app.route("/identity", methods=["GET"])
    def identity():
        """IDENTITY-QUERY-20260910: what actually serves ?model_key=K on this worker.

        Reads the cached runner for K (the exact object requests go through), asks
        its seat what the engine loaded, and with ?probe=1 sends a one-token chat
        and returns the `model` name the engine stamped on the answer."""
        import os as _os
        import httpx as _httpx
        key = (request.args.get("model_key") or "").strip()
        probe = request.args.get("probe") in ("1", "true", "yes")
        out = {"worker": state.name, "worker_id": state.worker_id, "model_key": key,
               "loaded": False, "via": None, "engine_model": None, "answered_as": None,
               "verified": None, "error": None}
        if not key:
            out["error"] = "model_key is required"
            return jsonify(out), 400
        try:
            from hugpy_engine.llama.runners.get import _LLAMA_INSTANCES, _LLAMA_LOCK
            with _LLAMA_LOCK:
                runner = _LLAMA_INSTANCES.get(key)
            if runner is None:
                # not a GGUF runner — an in-process transformers model?
                from hugpy_engine.dispatch.dispatch import _INSTANCES, _INSTANCES_LOCK
                with _INSTANCES_LOCK:
                    inst = next((v for k, v in _INSTANCES.items() if k[0] == key), None)
                if inst is not None:
                    out.update(loaded=True, via="in-process",
                               engine_model=getattr(inst, "model_key", None) or key)
                return jsonify(out)
            base = getattr(runner, "base_url", None)
            # IDENTITY-QUERY-V2-20260910: the file this key resolves to here — what "verified" compares against.
            expected = None
            try:
                from hugpy_engine.config.main import get_model_config
                from hugpy_engine.serve.serve import _model_file_for
                expected = _model_file_for(key, get_model_config(key)) or None
            except Exception:  # noqa: BLE001
                expected = None
            if base:
                # Any HTTP runner: a slot seat (its control port answers /identity)
                # or a plain llama-server (answers /props). Ask the very endpoint
                # this runner sends requests to.
                out["via"] = "http"
                st = None
                try:
                    r0 = _httpx.get(f"{base}/identity", timeout=3.0)
                    if r0.status_code == 200:
                        st = r0.json()
                        out["via"] = "slot"
                except Exception:  # noqa: BLE001
                    st = None
                if isinstance(st, dict):
                    out.update(loaded=bool(st.get("healthy")), slot_id=st.get("slot_id"),
                               slot_model_key=st.get("model_key"), model_path=st.get("model_path"),
                               engine_model=st.get("engine_model"))
                    out["verified"] = (st.get("model_key") == key) and (st.get("verified") is not False)
                else:
                    em = None
                    for sub in ("/props", "/v1/models"):
                        try:
                            r1 = _httpx.get(f"{base}{sub}", timeout=3.0)
                            if r1.status_code != 200:
                                continue
                            doc = r1.json() or {}
                            em = doc.get("model_path") or doc.get("model")
                            if not em and isinstance(doc.get("data"), list) and doc["data"]:
                                em = (doc["data"][0] or {}).get("id")
                            if em:
                                break
                        except Exception:  # noqa: BLE001
                            continue
                    out.update(loaded=em is not None, engine_model=em, model_path=expected)
                    out["verified"] = None if not em else True
                if probe and out["loaded"]:
                    r = _httpx.post(f"{base}/v1/chat/completions", timeout=120.0,
                                    json={"model": key, "messages": [{"role": "user", "content": "hi"}],
                                          "max_tokens": 1, "stream": False})
                    if r.status_code == 200:
                        out["answered_as"] = (r.json() or {}).get("model")
                    else:
                        out["error"] = f"probe {r.status_code}: {r.text}"
                        out["verified"] = False
            else:
                llm = getattr(runner, "llm", None)
                path = getattr(runner, "model_path", None) or (getattr(llm, "model_path", None) if llm is not None else None)
                out.update(loaded=llm is not None, via="in-process",
                           engine_model=path, model_path=path or expected,
                           verified=None if not path else True)
            em = out.get("engine_model")
            ref = out.get("model_path") or expected
            if em and ref:
                same = _os.path.basename(str(em)) == _os.path.basename(str(ref))
                out["verified"] = same and out["verified"] is not False
            if out.get("answered_as") and ref:
                out["answered_matches"] = _os.path.basename(str(out["answered_as"])) == _os.path.basename(str(ref))
        except Exception as exc:  # noqa: BLE001 — a query must never 500 the control plane
            out["error"] = f"{type(exc).__name__}: {exc}"
        return jsonify(out)

    @app.route("/ops/vram-holders", methods=["GET"])
    def ops_vram_holders():
        # VRAM-holder meter (incident 2026-08-05): READ-ONLY per-card breakdown
        # + a row per VRAM holder, with the idle own-venv squatter flagged
        # EXPLICITLY. Reuses the reaper's four-gate classification (labelling
        # only — never signals, never calls the kill verb). Same trust model as
        # every /ops/* (central's relay). Never 500s the control plane.
        try:
            return jsonify(_vram_holders(state))
        except Exception as exc:  # noqa: BLE001 — telemetry must never 500
            return jsonify({"ok": False, "cards": [], "holders": [],
                            "squatters": [], "gpu_util_pct": None,
                            "vram_total_bytes": None, "vram_used_bytes": None,
                            "vram_free_bytes": None, "min_age_s": _ORPHAN_MIN_AGE_S,
                            "error": f"{type(exc).__name__}: {exc}"})

    @app.route("/slots/<slot_id>/relaunch", methods=["POST"])
    def slot_relaunch(slot_id):
        # k14: relaunch ONE of this worker's slot children with a new offload depth
        # (n_gpu_layers) / context, so the k7 offload speed-cliff sweep can seat a
        # GGUF at full offload then sweep it DOWN through layer counts, measuring
        # tok/s at each step. Central relays here with {"n_gpu_layers"?, "ctx"?}.
        # The worker resolves slot_id -> its live control URL, confirms a model is
        # seated, and asks the slot supervisor to STOP->RESPAWN its child (the slot
        # owns the SIGTERM->SIGKILL). This also answers the ae "slot-child PID never
        # recycles" blocker: every relaunch respawns the child under a NEW pid.
        # 404 = no such slot on this worker; 409 = slot empty (nothing to relaunch).
        import httpx
        body = request.get_json(silent=True) or {}
        payload = {k: body[k] for k in ("n_gpu_layers", "ctx", "n_cpu_moe")
                   if body.get(k) not in (None, "")}
        try:
            from hugpy_engine.serve.slots import SlotPool
            statuses = SlotPool().statuses()
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": {
                "code": type(exc).__name__,
                "message": f"slot pool unavailable: {exc}"}}), 502
        target = None
        for s in (statuses or []):
            if str(s.get("slot_id")) == str(slot_id):
                target = s
                break
        if target is None:
            return jsonify({"ok": False, "slot_id": slot_id, "error": {
                "code": "UnknownSlot",
                "message": f"no slot {slot_id} on this worker"}}), 404
        if not target.get("model_key"):
            return jsonify({"ok": False, "slot_id": slot_id, "error": {
                "code": "EmptySlot",
                "message": f"slot {slot_id} has no model loaded to relaunch"}}), 409
        control = target.get("_control")
        try:
            # A relaunch respawns a (possibly big) child — allow a cold-load-long
            # window, same order as a /load warm.
            r = httpx.post(control + "/relaunch", json=payload, timeout=900.0)
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "slot_id": slot_id, "error": {
                "code": type(exc).__name__,
                "message": f"relaunch relay to slot failed: {exc}"}}), 502
        if isinstance(data, dict) and data.get("error"):
            return jsonify({"ok": False, "slot_id": slot_id,
                            "error": data.get("error")}), r.status_code
        # Echo the HONEST launched allocation: n_gpu_layers is what the fresh child
        # actually launched with (slot status), not merely what was requested.
        return jsonify({
            "ok": True, "slot_id": slot_id,
            "model_key": (data or {}).get("model_key"),
            "n_gpu_layers": (data or {}).get("n_gpu_layers"),
            "requested_n_gpu_layers": (data or {}).get("requested_n_gpu_layers"),
            "ctx": (data or {}).get("ctx"),
            "child_pid": (data or {}).get("child_pid"),
            "healthy": (data or {}).get("healthy"),
            "allocation": data,
        }), r.status_code

    @app.route("/slots/<slot_id>/unload", methods=["POST"])
    def slot_unload(slot_id):
        # THE STRANDED-SLOT FIX (operator ask "all pids need to be able to be
        # unseated, even from ram", 2026-07-25): every other eviction verb is
        # model_key-ADDRESSED (see /ops/evict's comment on why: PIDs are
        # per-box and recycled, so central should never send one). That
        # design has one hole — a slot whose child_pid is alive and holding
        # VRAM/RAM but whose model_key has gone None/stale (the claim itself
        # is broken, e.g. a load that half-failed) can't be RESOLVED by
        # model_key at all: _resolve_slot_handle can't find it (nothing
        # matches model_key==None), /slots/<id>/relaunch refuses on an empty
        # claim (409 EmptySlot), and /ops/reap-orphans deliberately treats
        # "child_pid still referenced by a slot status" as CLAIMED (not an
        # orphan) regardless of whether model_key is set — so none of the
        # three existing unseat paths reach it. This is what stranded ae for
        # 5.5h. Addressed by SLOT ID (not PID) — slot_id is a stable local
        # identifier (1..SLOT_COUNT), never recycled/foreign the way a PID
        # is, so this does not reintroduce the PID-addressing problem the
        # other verbs avoid. Unconditional: Slot.unload() kills the child
        # (SIGTERM->wait->SIGKILL) and clears the claim regardless of what
        # model_key (if any) it holds — the same mechanism /ops/evict's slot
        # branch already uses, just reached by slot_id instead of a resolved
        # model_key. 404 = no such slot on this worker.
        try:
            from hugpy_engine.serve.slots import SlotPool
            statuses = SlotPool().statuses()
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": {
                "code": type(exc).__name__,
                "message": f"slot pool unavailable: {exc}"}}), 502
        target = None
        for s in (statuses or []):
            if str(s.get("slot_id")) == str(slot_id):
                target = s
                break
        if target is None:
            return jsonify({"ok": False, "slot_id": slot_id, "error": {
                "code": "UnknownSlot",
                "message": f"no slot {slot_id} on this worker"}}), 404
        model_key_before = target.get("model_key")
        pid_before = target.get("child_pid")
        footprint = None
        if model_key_before:
            # Best-effort honest footprint for a slot that DID have an
            # attributable model_key — same measurement /ops/evict uses.
            try:
                footprint = _model_footprint_before_evict(
                    model_key_before, "slot",
                    {"child_pid": pid_before, "control_url": target.get("_control")})
            except Exception:  # noqa: BLE001
                footprint = None
        control = target.get("_control")
        try:
            SlotPool().unload(control)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "slot_id": slot_id, "error": {
                "code": type(exc).__name__,
                "message": f"slot unload failed: {exc}"}}), 502
        _trim_host_ram()
        return jsonify({
            "ok": True, "slot_id": slot_id,
            "model_key_before": model_key_before,
            "child_pid_before": pid_before,
            "freed": footprint,
            "reason": f"slot {slot_id} unconditionally unloaded "
                      f"(pid={pid_before} model_key={model_key_before!r})",
        })

    @app.route("/models/redownload", methods=["POST"])
    def redownload():
        # Force a CLEAN re-pull from central: evict from VRAM, DELETE the model's
        # local files, then re-provision (download) it. Body: {"model_key": ...}.
        # A plain /load only downloads when files are MISSING, so it can't refresh
        # a corrupt/stale on-disk copy — this can.
        body = request.get_json(silent=True) or {}
        model_key = body.get("model_key")
        if not model_key:
            return jsonify({"ok": False, "error": "missing model_key"}), 400
        try:
            from hugpy_engine.dispatch.dispatch import evict as _evict
            from hugpy_storage.provision import wipe_model, ensure_model_present, ensure_model_registered
            try:
                _evict(model_key)   # drop from VRAM so its files aren't held open
            except Exception:
                pass
            ensure_model_registered(model_key, state.central_url)
            wiped = wipe_model(model_key)
            # Gate the re-pull (Part A, slice 8): a /redownload wipes then re-pulls
            # — it MUST run the storage gate too, or it re-fills an over-budget
            # store the operator just tried to relieve. Pass state explicitly.
            ok = ensure_model_present(model_key, state.central_url, state=state)
            return jsonify({"ok": bool(ok), "wiped": bool(wiped),
                            "redownloaded": bool(ok), "model_key": model_key,
                            "loaded_models": loaded_model_keys()})
        except Exception as exc:
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500

    @app.route("/models/fetch", methods=["POST"])
    def fetch():
        """Download a model to THIS worker's DISK without loading it into VRAM.

        The FETCH half of provisioning (operator ruling 2026-09-24): the old
        /probe was an undocumented download+load combo; this is the download-only
        primitive. It uses the SAME normal central->worker transfer path lazy
        first-call provisioning uses (ensure_model_present via _kick_provision),
        so the pull rides the transfer ledger and the heartbeat's
        provision_progress — the console shows "downloading from central". It
        returns immediately (fire-and-forget); the model ends resident on DISK,
        not in VRAM. Load is a separate, explicit step (POST /probe).

        Body: {"model_key": ...}. Answers {ok, model_key, already_local, started,
        provisioning}. Never starts a runner.
        """
        body = request.get_json(silent=True) or {}
        model_key = body.get("model_key")
        if not model_key:
            return jsonify({"ok": False, "error": "missing model_key"}), 400
        try:
            from hugpy_storage.provision import model_is_local
            try:
                already = bool(model_is_local(model_key))
            except Exception:  # noqa: BLE001 — unknown key: not local, kick the pull
                already = False
            if already:
                return jsonify({"ok": True, "model_key": model_key,
                                "already_local": True, "started": False,
                                "provisioning": False})
            _kick_provision(state, model_key, purpose="fetch", load=False)
            with state._provision_lock:
                provisioning = model_key in state._provisioning
            return jsonify({"ok": True, "model_key": model_key,
                            "already_local": False, "started": True,
                            "provisioning": provisioning})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500

    @app.route("/reap", methods=["POST"])
    def reap():
        """Disk reclaim (tiers-v2 slice 4). The bookend to unassign: delete the
        local files of models that are on disk but no longer needed.

        Body:
          {"dry_run": true}          -> PREVIEW only (default): what would be
                                        freed + what's protected and why.
          {"all": true}             -> reclaim every reclaimable model.
          {"model_keys": ["a","b"]} -> reclaim just these (still guard-checked).

        Guards (re-proven at delete time): never assigned, loaded/loading,
        pinned, or comfy (operator symlinks). Deletes are jailed by wipe_model
        against root/home/short paths.
        """
        body = request.get_json(silent=True) or {}
        scan = _reap_scan(state)
        if body.get("dry_run") or (not body.get("all") and not body.get("model_keys")):
            return jsonify({"ok": True, "dry_run": True, **scan})
        if body.get("all"):
            targets = [r["model_key"] for r in scan.get("reclaimable", [])]
        else:
            targets = [str(k) for k in (body.get("model_keys") or [])]
        if not targets:
            return jsonify({"ok": True, "results": [], "freed_bytes": 0,
                            "note": "nothing reclaimable"})
        return jsonify(_reap_reclaim(state, targets))

    # Studio render offload (option a): mount POST /studio/render, GET
    # /studio/render/<job_id>, POST /studio/cancel/<job_id> so central can delegate
    # a REAL-model studio render (produce_clip) to THIS worker's GPU while keeping
    # the control plane. Imported LAZILY (studio_render's own studio-spine imports
    # are lazy inside its render thread, so this never pulls torch/diffusers at
    # boot) and guarded so a mount hiccup can never break the rest of the agent's
    # routes.
    try:
        from hugpy_fleet.worker.studio_render import register_studio_routes
        register_studio_routes(app, worker_id=state.worker_id, worker_name=state.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("studio render endpoints not mounted: %s", exc)

    return app


def _free_vram_bytes() -> int | None:
    try:
        # Relative import (rename-proof for the prod mirror). This line was
        # once commented out, which left a bare NameError swallowed below —
        # every probe/unload reported vram_free null and fit=false even when
        # the weights landed on the GPU.
        from hugpy_engine.spill import free_vram_bytes
        return free_vram_bytes()
    except Exception:
        return None


def _probe_model(model_key: str, state: "WorkerState") -> dict:
    """Load the model on the GPU and report fit + VRAM deltas.

    Returns {ok, fit, vram_free_before, vram_free_after, vram_used, error}.
    'fit' is a heuristic: ok load AND GPU memory actually decreased (i.e. weights
    landed on the GPU, not spilled entirely to CPU).
    """
    before = _free_vram_bytes()
    result: dict = {"model_key": model_key, "vram_free_before": before}
    try:
        # PROBE DOES NOT DOWNLOAD (operator ruling, ae 1.2TB incident 2026-07-17:
        # "its central that distributed these downloads... it simply needs to
        # abide by the limits set within its own backend"). A probe used to call
        # ensure_model_present() — so probing an ABSENT model WAS a transfer
        # order, and central's warm sweep rode /probe to pull ~700GB onto ae.
        # A probe answers a question ("does this model FIT on my GPU?"); it never
        # provisions. If the files aren't already on THIS box's disk, return an
        # honest non-downloading verdict and let the model arrive on the first
        # REAL call (lazy-download doctrine, 7f0e6e8/2a3baeb).
        #
        # ensure_model_registered is METADATA-only (a small config row from
        # central, no weight bytes) — kept so the locality check + the honest
        # error can name/resolve the model even if this worker wasn't built with
        # it. NO byte transfer happens on ANY path through here.
        from hugpy_storage.provision import ensure_model_present, ensure_model_registered, model_is_local
        canonical = ensure_model_registered(model_key, state.central_url) or model_key

        # Locality gate: the SAME predicate the agent uses everywhere else
        # (model_is_local / _models_local). If the weights aren't already on
        # disk, do not build the runner (which would trigger a load/pull) — the
        # probe reports "not local" and stops here. This is what makes the warm
        # sweep's probe a no-op instead of a distributed 700GB pull order.
        try:
            _local = model_is_local(canonical)
        except Exception:  # noqa: BLE001 — a bad row reads as "not local", never a crash
            _local = False
        if not _local:
            result.update(
                ok=False, fit=False,
                vram_free_after=before, vram_used=0, path="none", local=False,
                error=("not local — probe does not download (lazy doctrine "
                       "2026-07-17); files arrive on first real call"))
            return result
        result["local"] = True

        # Local: safe to build the runner and measure a real fit — no transfer
        # can be triggered because the files are already present.
        #from abstract_hugpy_dev.managers.dispatch import runner_for
        runner = runner_for(model_key=canonical)  # builds the runner WRAPPER
        # runner_for only BUILDS the (lazy) wrapper; for GGUF/in-process runners
        # the weight load + slot seat happen on first .runner access, so a bare
        # build loads NOTHING — the probe would read vram_used=0 / fit=False and
        # seat no slot (exactly the hollow shell that made this model unroutable).
        # Force the underlying runner resident so the probe reflects reality.
        _materialize(runner)

        after = _free_vram_bytes()
        used = (before - after) if (before is not None and after is not None) else None
        # Which path actually took the load: a base_url means an HTTP child
        # (slot or native llama-server); none means in-process llama-cpp-python.
        base_url = (getattr(runner, "base_url", None)
                    or getattr(getattr(runner, "runner", None), "base_url", None))
        # MoE-AWARE FIT (2026-07-26). The raw rule below — "GPU free memory
        # dropped" — is the RIGHT question for a dense model and the WRONG one for
        # a split MoE. Under the derived split only the NON-EXPERT tensors land on
        # the card (coder-next: ~1.49 GiB of 45 GiB) while the experts sit in RAM by
        # design, so a PERFECTLY placed MoE trips the 64 MiB threshold only barely,
        # and a large-ctx/offloaded variant can miss it outright — reporting
        # fit:false for the configuration we actually want. Worse, the inverse also
        # misleads: an MoE that loaded WHOLE (the bug this probe should catch) shows
        # a huge VRAM drop and reads as a confident fit:true.
        #
        # So for a measured MoE, price the split the same way the allocator does:
        # the model FITS when its non-expert share fits the card. `used` stays in
        # the payload verbatim — the caller still sees exactly what was consumed —
        # but the VERDICT stops being a function of how much VRAM disappeared.
        # Degrade-not-guess: any unreadable header/total falls back to the raw rule.
        fit = bool(used and used > 64 * 1024 * 1024)
        try:
            from hugpy_engine.spill import gguf_moe_detail, total_vram_bytes
            from hugpy_fleet.worker.imports import get_model_path
            _moe = gguf_moe_detail(get_model_path(canonical))
            if _moe.get("is_moe"):
                _non = int(_moe.get("non_expert_bytes") or 0)
                _tot = total_vram_bytes()
                if _non and _tot:
                    fit = _non <= _tot
                    result["moe_fit_basis"] = {
                        "is_moe": True,
                        "non_expert_bytes": _non,
                        "expert_bytes": int(_moe.get("expert_bytes") or 0),
                        "gpu_total_bytes": _tot,
                        "why": (f"MoE: priced the {_non / 2**30:.2f} GiB non-expert "
                                f"share against the {_tot / 2**30:.2f} GiB GPU "
                                "(experts belong in RAM under the split), not the "
                                f"{(used or 0) / 2**30:.2f} GiB VRAM delta"),
                    }
        except Exception:  # noqa: BLE001 — advisory; never turn a probe into a crash
            pass
        result.update(
            ok=True,
            vram_free_after=after,
            vram_used=used,
            path="http" if base_url else "in-process",
            fit=fit,
        )
        # Vision honesty: a vision GGUF served IN-PROCESS cannot decode images
        # (the python binding fails to load the mmproj projector — the reason
        # the native --mmproj server path exists). The load "succeeds" but
        # every image turn silently degrades to text-only, so report the probe
        # as FAILED with the actionable reason instead of ok:true.
        # Since 2026-09-23 the engine REFUSES that load (vision_needs_slot,
        # surfaced through the except below), so this is a backstop for an
        # older engine build. Vision == a projector on disk (find_mmproj), NOT
        # the registry's task list: a GGUF tagged image-text-to-text without a
        # projector is a text model and loads correctly in-process (that task
        # test mis-flagged Qwen2.5-7B-Instruct-GGUF).
        if not base_url:
            try:
                from hugpy_platform.utils import find_mmproj
                from hugpy_fleet.worker.imports import get_model_config, get_model_path
                cfg = get_model_config(canonical)
                mpath = None
                try:
                    mpath = get_model_path(canonical)
                except Exception:
                    mpath = getattr(cfg, "dir", None)
                if mpath and find_mmproj(str(mpath)):
                    result.update(
                        ok=False, fit=False,
                        error=("vision_needs_slot — vision model loaded in-process "
                               "(text-only — the python binding cannot load the "
                               "mmproj projector), so images would be silently "
                               "ignored. Provide a native llama-server "
                               "(LLAMA_SERVER_BIN or `hugpy install-engine`) or a "
                               "healthy slot child so the projector loads."),
                        load_failure={"class": "vision_needs_slot",
                                      "loader_stderr": None, "path": str(mpath)})
            except Exception:
                pass  # capability check is advisory — never turn it into a probe crash
    except Exception as exc:
        result.update(ok=False, fit=False, error=f"{type(exc).__name__}: {exc}")
        # The load report central stores (load_reports[key]) gets the typed
        # verdict too — every probe failure IS a load failure, so classify.
        _with_load_failure(result, exc, classify=True)
    return result


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
        # Fleet role: "worker" (whole-model, serves /infer) or "rpc" (lends its
        # GPU to a shard pool via llama.cpp rpc-server). rpc_endpoint is the
        # "host:port" central hands to a lead as an rpc_servers entry.
        self.role = "worker"
        self.rpc_endpoint: str | None = None
        # Central's SHARED studio weights root (2026-09-24), adopted from the
        # register/heartbeat reply. The studio-presence probe (_studio_capability)
        # checks model_index.json readability under it — the SAME path the render
        # manifest is built against — so we advertise only models this box can load.
        # None until central first advertises it.
        self.studio_weights_root: str | None = None
        # Models central says we should serve, plus which we've already kicked
        # off a background provision for (so we don't re-trigger every beat).
        self.assigned_models: list[str] = []
        self._provisioning: set[str] = set()
        # Central's per-worker allocations, adopted from the heartbeat reply
        # (_apply_central_limits). The STORAGE budget reads limits
        # ["disk_cache_gib"] from here; unset -> the budget is unmanaged and the
        # auto-evict path stays off (see budget.cap_bytes).
        self.limits: dict = {}
        # Central's LRU clock {model_key: epoch} — when each model was last
        # PICKED to serve on this box. The FIFO key for evict-to-fit; the worker
        # cannot know it (central routes the calls), so central ships it in the
        # heartbeat reply. Missing key -> 0 -> coldest -> evicted first.
        self.model_last_picked: dict = {}
        # Central's ALLOCATION-LEVEL totals for this box's assignment set:
        # {allocated_total_bytes, allocated_count, allocated_unknown_count}.
        # Sizing the set needs the manifest (central-only), so central computes
        # it per read and ships it in the heartbeat reply; the refusal reason
        # reads it from here rather than making N HTTP calls under the pull lock.
        # Empty until the first beat -> the refusal simply omits the structural
        # clause (an unknown total is never reported as a comfortable 0).
        self.allocated: dict = {}
        # Models REFUSED for storage: {model_key: {state:"refused", reason:...}}.
        # Reported in the heartbeat so central/console render the model as
        # MISSING with a hover reason instead of a phantom "pulling".
        self.refused: dict = {}
        # key -> {done_bytes, total_bytes, frac}; populated while a background
        # pre-provision downloads, so central (and the console) can show a %.
        self._provision_progress: dict[str, dict] = {}
        self._provision_lock = threading.Lock()
        # Thundering-herd guard (root-caused live 2026-07-15): a big assignment
        # list used to spawn one _kick_provision background thread PER model
        # simultaneously, each running its own segmented/parallel download —
        # N assigned models meant N concurrent multi-threaded pulls hammering
        # central at once (observed: 30 models, near-constant 503s, no
        # convergence). Cap how many DIFFERENT models may provision at the
        # same time; the rest queue and drain serially. Default 1 (fully
        # serial) — the safe, calm default; raise via env if a box's link to
        # central can actually take it. Read once here (like the other
        # WorkerState fields) rather than re-reading the env on every kick.
        _cap = _safe_int(os.environ.get("WORKER_PROVISION_CONCURRENCY"))
        if _cap is None or _cap < 1:
            _cap = 1
        self.provision_concurrency: int = _cap
        self._provision_semaphore = threading.BoundedSemaphore(self.provision_concurrency)
        # The live werkzeug HTTP server (set in main() once bound). The restart
        # path closes its listening socket to release :9100 cleanly before exit;
        # None until the server is created (and in test clients that never bind).
        self.http_server = None

    def provision_snapshot(self) -> dict:
        """A lock-safe copy of per-model download progress for the heartbeat."""
        with self._provision_lock:
            return {k: dict(v) for k, v in self._provision_progress.items()}


def _load_bytes_per_s():
    """The provisioner's measured central->worker transfer rate (bytes/s), or
    None — never raises (telemetry must not break a heartbeat)."""
    try:
        from hugpy_storage.provision import transfer_rate
        return transfer_rate()
    except Exception:  # noqa: BLE001
        return None


def _eager_pull(model_key: str) -> bool:
    """Should ASSIGNMENT alone pull this model's weights to local disk?

    Lazy-download doctrine (operator, 2026-07-16): "models are attributed to be
    routed to a worker though not immediately downloaded to the worker's drive,
    they should be lazy download instead downloading to the drive only when
    called". Assignment is ATTRIBUTION, not a transfer order — the download
    happens on first CALL, via the inference path's already-working
    _ensure_present / _ensure_present_streaming.

    This is the structural fix for the 2026-07-15 provision storm: assigning N
    models fired N parallel provisions, 503'ing central and leaving four
    truncated GGUFs (~10.7GB) on computron — every one of them "designated" in
    worker_assignments.json.

    Exactly ONE tier pre-pulls, because for it lazy would break a promise the
    tier already makes:

      * static (:_residency) — operator-locked 2026-07-05 as "eager-pulled": a
        locked tier that paid full download latency on first call is a broken
        promise (see the defaults-are-promises doctrine), so its FILES are
        pre-pulled to disk. This is a download only — static, like every tier,
        loads into VRAM only when a request for it arrives. The operator opts
        INTO the pre-pull by choosing the tier.

    📌 pin is NOT an eager tier (operator, 2026-07-16): "pinned doesnt mean
    anything aside from: 1) is the model attributed to a worker; if yes, then
    it always will be". Pin is PERMANENT ATTRIBUTION — it answers "does this
    model belong to this worker?", not "when do the bytes arrive". A pinned
    model is still a lazy download, same as any other. Pinning previously
    implied a pre-pull here, which made pin a de-facto transfer order: on ae,
    65/65 assigned models were pinned, so deleting them re-pulled all 65 via
    _reconcile_loop and filled the operator's workstation to 0 bytes free
    (2026-07-16). "none should be pulling at all. they should be lazy."

    Everything else (the on-demand DEFAULT, and now 📌pin) waits to be called.
    NOTE for reconcile: for a non-static model, "assigned but not on disk" is
    the CORRECT resting state, not drift to converge.
    """
    try:
        return _residency(model_key) == "static"
    except Exception:  # noqa: BLE001 — a settings read must not break adoption
        # Fail LAZY: the worst case is one first-call download, whereas failing
        # eager re-creates the storm this function exists to prevent.
        return False


def _sync_assignment(state: "WorkerState", worker: dict) -> None:
    """React to central's worker record: adopt its model list.

    Central owns the assignment (set in the UI). The agent reads it back from
    every register/heartbeat response. Adoption is LAZY (see _eager_pull):
    being assigned a model does NOT download it — only 🔒static models are
    pre-pulled here; every other tier (the on-demand default AND 📌pinned)
    downloads on first call. Pin is permanent ATTRIBUTION, never a transfer
    order. Without this adoption the worker never knew about UI allocation
    changes.

    Adoption NEVER loads a model into VRAM. Being assigned (or re-assigned) a
    model — static or otherwise — does not seat it; a model becomes resident only
    when a request for it arrives and the demand path evicts-to-fit and serves it.
    """
    if not isinstance(worker, dict):
        return
    if "models" not in worker:
        # No authoritative assignment list in this response — adopt nothing,
        # and above all don't treat it as "everything unassigned" below.
        return
    models = worker.get("models") or []
    changed = models != state.assigned_models
    state.assigned_models = list(models)
    # Tiers v3 lazy cleanup: with central's authoritative list in hand, drop
    # residency overrides (static OR on-demand) for models no longer assigned
    # — unless pinned (📌 = permanent attribution). Runs every heartbeat, so
    # it also catches unassigns that happened while this agent was down.
    try:
        _prune_stale_residency(state)
    except Exception as exc:  # noqa: BLE001 — cleanup must not break adoption
        logger.warning("residency prune failed: %s", exc)
    if not changed:
        return
    logger.info("assignment updated: serving %s", models or "(nothing)")

    # LEARN THE CONFIG ROWS (2026-07-26, operator: the missing state "is
    # misleading, this should be fixed").
    #
    # THE BUG THIS CLOSES: locality was decided by `model_is_local` ->
    # `get_model_config(mk)`, which RAISES for a key this worker's registry has
    # never learned. model_is_local swallows that and returns False, so the key
    # silently dropped out of `models_local` and the console painted it
    # "○ missing" — on ae, 54 of 64 assigned models, including one that had
    # just served a request. The storage scan hit the same wall one level up
    # (`except -> _skip("no_config")`: 54 no_config, 20 of 87 keys classified).
    # An unlearned key stayed unlearned FOREVER on the presence path, because
    # nothing on that path ever registered it — which is why actually serving
    # the model did not clear its pill.
    #
    # THE ASYMMETRY: `ensure_model_registered` exists precisely to learn an
    # unknown row from central on demand, and the serve/probe/provision paths
    # all call it. The two PRESENCE readers did not. This closes that gap at
    # the one place that already knows the authoritative list.
    #
    # WHY HERE AND NOT IN _models_local: this runs only when the assignment
    # list actually CHANGED (we are past `if not changed: return`), so it is
    # naturally throttled to assignment edits — not once per heartbeat, and not
    # a 64-key burst on every beat from every worker. `ensure_model_registered`
    # short-circuits on `_assure_local_key` (pure local lookup) for everything
    # already known, so the steady state costs no HTTP at all; only genuinely
    # unknown keys fetch, and each fetch is METADATA ONLY (a small config row —
    # no weight bytes, so this is not a transfer order and cannot become one).
    # Deliberately NOT gated on _eager_pull: learning a config is exactly what
    # the lazy tiers need to report presence honestly while staying un-pulled.
    #
    # Runs in the background via the shared single-flight helper, which skips
    # keys already resolvable (pure local lookup, no HTTP) — so the steady state
    # costs nothing and only genuinely unknown rows fetch. Best-effort: a worker
    # that cannot reach central still reports what it already knows, and today's
    # behavior (absent -> "missing") IS the failure mode, so a failure here can
    # only restore the current reading, never worsen it.
    try:
        from hugpy_storage.provision import _assure_local_key
        _unknown = [mk for mk in models if not _assure_local_key(mk)]
    except Exception:  # noqa: BLE001 — never break adoption over a lookup
        _unknown = []
    if _unknown:
        _kick_learn_configs(state, _unknown)

    # Lazy by default: pre-pull ONLY 🔒static, the one tier that promises local
    # presence. Everything else — the on-demand default AND 📌pinned — downloads
    # on first call via _ensure_present. Pin is attribution, not a pre-fetch.
    for model_key in models:
        if _eager_pull(model_key):
            logger.info("pre-provisioning %s (static — eager tier)", model_key)
            _kick_provision(state, model_key, purpose="assign")
    # Adoption stops at the disk pull above. It never seats a model into VRAM:
    # a model becomes resident only when a request for it arrives.


def _kick_provision(state: "WorkerState", model_key: str,
                    purpose: str = "reconcile", load: "bool | None" = None) -> None:
    """Download ONE model to this worker's DISK in the background — never load.

    Shared by assignment adoption (static eager pre-pull), the UTIL-08 reconcile
    loop, and the fetch-only verb; the _provisioning guard makes concurrent kicks
    a no-op. This puts weights on disk and stops there: a model becomes VRAM/RAM
    resident only when a request for it arrives. ``load`` is retained as an
    ignored compatibility keyword for older callers: provisioning is now
    unconditionally download-only.

    ``purpose`` ("assign" from adoption, "reconcile" from the loop) is a
    BACKGROUND purpose (2026-07-17): central MAY 409 this pull if it would push
    the worker over its storage budget ("central abides by the limits set within
    its own backend"). Contrast the demand path (_ensure_present), which is never
    budget-refused centrally."""
    if restart_requested():
        # A restart is underway — don't spin up a NEW transfer pool into a process
        # about to exit (it would only be torn down by _shutdown_executors).
        return
    # k2: single choke for BOTH callers (assignment adoption's eager pre-pull
    # and the UTIL-08 reconcile loop's re-kick). An operator BLOCK does not
    # auto-unassign, so a blocked static model can still be sitting in
    # state.assigned_models and NOT local — without this, the reconcile loop
    # would re-kick a doomed pull of it every reconcile_interval_s forever.
    # Log once, not every pass; never a request refusal (background-only).
    if _is_blocked_locally(model_key):
        _log_blocked_skip_once(model_key, f"provisioning ({purpose})")
        return
    with state._provision_lock:
        if model_key in state._provisioning:
            return
        state._provisioning.add(model_key)

    if True:  # (indentation shim — keeps the battle-tested _bg body verbatim)
        def _bg(mk=model_key):
            def _prog(done, total, fname=None):
                # Mirrors the inference-time SSE progress, but recorded on state
                # so the heartbeat can report it (the panel polls heartbeats).
                frac = (done / total) if total else 0.0
                with state._provision_lock:
                    entry = state._provision_progress.setdefault(mk, {})
                    entry.update(done_bytes=done, total_bytes=total or 0,
                                 frac=round(frac, 4))
                    # Provenance (item 4): _provision_now streams a "source=…"
                    # pseudo-filename when it picks central vs HF — keep it on
                    # the entry so the console can attribute the pull.
                    if isinstance(fname, str) and fname.startswith("source="):
                        entry["source"] = fname[len("source="):]
            try:
                from hugpy_storage.provision import ensure_model_present, model_is_local
                # ComfyUI-backed rows: everything is symlinks — already-
                # loadable / link-from-layout / pull-then-link. ComfyUI owns
                # its own residency, so no runner preload either.
                try:
                    from hugpy_storage.provision import ensure_comfy_checkpoint, ensure_model_registered
                    from hugpy_fleet.worker.imports import get_model_config
                    _ck = ensure_model_registered(mk, state.central_url) or mk
                    if getattr(get_model_config(_ck), "framework", None) == "comfy":
                        ok = ensure_comfy_checkpoint(_ck, state.central_url)
                        logger.info("comfy checkpoint for %s: %s", mk,
                                    "ready" if ok else "NOT available")
                        return
                except Exception:  # noqa: BLE001 — fall through to normal flow
                    pass
                # Hardening: model_is_local RAISES for a key this worker's
                # registry hasn't learned yet — that must trigger the pull
                # (which starts with ensure_model_registered), not abort it.
                try:
                    _has_files = model_is_local(mk)
                except Exception:  # noqa: BLE001
                    _has_files = False
                if not _has_files:
                    logger.info("pre-provisioning assigned model %s…", mk)
                    ensure_model_present(mk, state.central_url, progress=_prog,
                                         state=state, purpose=purpose)
                    logger.info("pre-provisioned %s", mk)
                    state.refused.pop(mk, None)   # it fit after all
                # Files are on disk; nothing is loaded here. A model loads into
                # VRAM/RAM only when a request for it arrives — never on download,
                # never for static, never under WORKER_PRELOAD/WORKER_POOL.
            except BudgetRefusal as exc:
                # Not a failure — a DECISION, made before any bytes moved. Record
                # it so the heartbeat reports the model as MISSING with an honest
                # reason (hover text) instead of a pull that never starts.
                state.refused[mk] = dict(exc.reason)
                logger.error("pre-provision of %s REFUSED: %s", mk,
                             exc.reason.get("reason"))
            except Exception as exc:
                logger.warning("pre-provision of %s failed: %s", mk, exc)
            finally:
                with state._provision_lock:
                    state._provisioning.discard(mk)
                    state._provision_progress.pop(mk, None)

        def _bg_gated(mk=model_key):
            # Thundering-herd gate: acquire a slot in the fleet-wide provision
            # semaphore (default 1 = fully serial) BEFORE running the
            # battle-tested _bg body above, release after — win, lose, or
            # exception. This only throttles HOW MANY of these background
            # threads may be doing the heavy ensure_model_present() work at
            # once; it does NOT throttle the inference-triggered path
            # (_ensure_present / _ensure_present_streaming), which calls
            # ensure_model_present() directly and never goes through
            # _kick_provision — so a live chat waiting on a model is never
            # stuck behind a long queue of background assignment pre-fetches.
            # Blocking here (not a timeout/try-acquire) is intentional: every
            # assigned model must EVENTUALLY provision, just not all at once;
            # the per-model _provisioning guard above already prevents the
            # same key from queuing twice, so the wait is bounded by the
            # number of genuinely distinct models still ahead of it.
            state._provision_semaphore.acquire()
            try:
                _bg(mk)
            finally:
                state._provision_semaphore.release()

        threading.Thread(target=_bg_gated, daemon=True).start()


# ── UTIL-08: desired-state reconcile ─────────────────────────────────────────
# Assignment adoption only fires on CHANGE, so a failed pull used to drift
# forever (assigned, files absent, nobody retries until the operator touches
# the assignment). The reconcile loop re-kicks provisioning for any assigned
# model whose files are missing; models_local in the heartbeat gives central
# the disk-truth to SHOW the drift meanwhile.

_MODELS_LOCAL_CACHE: dict = {"at": 0.0, "value": []}


def _models_local(state: "WorkerState") -> list[str]:
    """Assigned models whose files are actually on THIS worker's disk (60s
    cache — model_is_local walks directories; don't pay that every beat)."""
    now = time.time()
    if now - _MODELS_LOCAL_CACHE["at"] < 60.0:
        return _MODELS_LOCAL_CACHE["value"]
    if not state.assigned_models:
        # Startup window: the assignment list arrives with the FIRST heartbeat
        # response — caching an empty walk here made the console show
        # everything '✗ missing' for ~60s after any restart. Don't cache.
        return []
    out: list[str] = []
    unknown: list[str] = []
    try:
        from hugpy_storage.provision import _assure_local_key, model_is_local
        for mk in list(state.assigned_models):
            try:
                if model_is_local(mk) or _local_under_any_alias(mk):
                    out.append(mk)
                elif not _assure_local_key(mk):
                    # NOT-LOCAL vs NOT-KNOWN (2026-07-26). model_is_local returns
                    # False for BOTH "the files aren't here" and "this worker's
                    # registry can't resolve the key" (get_model_config raises,
                    # and it swallows that). Only the first is a real absence;
                    # the second is a resolution gap that made the console paint
                    # "○ missing" over models sitting on disk — 54 of 64 on ae.
                    # _assure_local_key is a pure local lookup (no HTTP), so
                    # separating the two costs nothing per beat.
                    unknown.append(mk)
            except Exception:  # noqa: BLE001 — one bad row must not hide the rest
                pass
    except Exception:  # noqa: BLE001
        pass
    if unknown:
        # Learn the rows OUT OF BAND, then let the NEXT beat re-read locality.
        # Deliberately not inline: this walk runs on the heartbeat path and
        # ensure_model_registered does one central fetch per unknown key.
        # Single-flight + metadata-only (no weight bytes — never a transfer
        # order). The list this beat is still honest about what it could prove.
        _kick_learn_configs(state, unknown)
    _MODELS_LOCAL_CACHE.update(at=now, value=out)
    return out


def _key_aliases(model_key: str) -> list:
    """The other spellings this model may be stored/served under.

    WORKER-SIDE MIRROR of central's ``workers._match_keys`` "~"-tail
    unification (0.1.202, operator doctrine 2026-07-23). Registry keys qualify
    a base name with its owner via ``~`` (``Qwen~Qwen3-Coder-Next-GGUF``) while
    the on-disk/served form is routinely the BARE base
    (``Qwen3-Coder-Next-GGUF``). Central fixed this for ROUTING; PRESENCE was
    never taught the same alias, which is the second half of the "○ missing"
    report — see ``_local_under_any_alias``.

    Deliberately NOT ``provision._dir_slug``: that folds separators but KEEPS
    the owner segment, so ``qwen_qwen3_coder_next_gguf`` never equals
    ``qwen3_coder_next_gguf`` and it cannot bridge this pair (verified before
    writing this). The ``~``/``/``-tail is the alias that actually applies.

    Raw form stays first-class; these are ADDITIONS, never replacements.
    """
    raw = str(model_key or "").strip()
    if not raw:
        return []
    out = []
    for sep in ("~", "/"):
        if sep in raw:
            base = raw.split(sep, 1)[1] if sep == "~" else raw.split("/")[-1]
            if base and base != raw:
                out.append(base)
    return out


def _local_under_any_alias(model_key: str) -> bool:
    """True if the model is on disk under an ALIAS of ``model_key``.

    Closes the second half of the 2026-07-26 "○ missing" report. On ae the
    storage scan listed ``Qwen3-Coder-Next-GGUF`` at 45.09 GiB ON DISK while
    ``models_local`` omitted it, because the assignment carries
    ``Qwen~Qwen3-Coder-Next-GGUF`` and membership was compared VERBATIM. Both
    spellings independently answered ``model_is_local=True`` / probe
    ``local:true`` — the files were fine and the predicate was fine; only the
    spelling differed. 9 of the 11 models still falsely missing on ae after the
    config-learning fix were this exact class.

    Cheap and last-resort: only consulted AFTER ``model_is_local(mk)`` has
    already said False, and only for keys that actually carry a ``~``/``/``
    qualifier — a bare key produces no aliases and returns immediately, so the
    common path pays nothing. Never raises: a bad alias reads as "not local",
    which is exactly today's behavior.
    """
    try:
        from hugpy_storage.provision import model_is_local
    except Exception:  # noqa: BLE001
        return False
    for alias in _key_aliases(model_key):
        try:
            if model_is_local(alias):
                return True
        except Exception:  # noqa: BLE001 — one bad alias must not hide the rest
            continue
    return False


_LEARN_CONFIGS_LOCK = threading.Lock()
_LEARN_CONFIGS_INFLIGHT: set = set()


def _kick_learn_configs(state: "WorkerState", keys: list) -> None:
    """Learn central's config rows for keys this worker's registry can't resolve.

    The presence readers (``_models_local``, ``_worker_storage``) decide via
    ``get_model_config``, which RAISES for an unlearned key — so such a model
    reads as absent forever even while it serves. ``ensure_model_registered``
    is the existing cure (metadata only, no weight bytes); this is the throttled
    way to call it off a per-beat path.

    Single-flight per key: a key already being learned is skipped, so repeated
    beats can never stack fetches for the same model. Best-effort — on failure
    the reading simply stays what it is today."""
    central = getattr(state, "central_url", None)
    if not central:
        return
    with _LEARN_CONFIGS_LOCK:
        todo = [k for k in keys if k not in _LEARN_CONFIGS_INFLIGHT]
        _LEARN_CONFIGS_INFLIGHT.update(todo)
    if not todo:
        return

    def _bg():
        learned = 0
        try:
            from hugpy_storage.provision import ensure_model_registered
            for mk in todo:
                if restart_requested():
                    return
                try:
                    if ensure_model_registered(mk, central):
                        learned += 1
                except Exception:  # noqa: BLE001
                    continue
        finally:
            with _LEARN_CONFIGS_LOCK:
                _LEARN_CONFIGS_INFLIGHT.difference_update(todo)
        if learned:
            _MODELS_LOCAL_CACHE.update(at=0.0, value=[])
            logger.info("learned %d unresolved config row(s); presence cache "
                        "invalidated (was reporting them as absent)", learned)

    threading.Thread(target=_bg, daemon=True).start()


def _reconcile_loop(state: "WorkerState") -> None:
    """Every reconcile_interval_s (default 600): any assigned 🔒static model that
    is NOT local and NOT already provisioning gets its provisioning re-kicked.
    Static is the ONLY tier that promises local presence. Converges failed pulls
    instead of drifting until the next assignment change; the _provisioning guard
    + single-flight lock keep it idempotent.

    Lazy-download doctrine (2026-07-16): a non-static model that is assigned but
    absent is NOT drift — it is the correct resting state, and it stays absent
    until something calls it. Re-kicking it here would silently rebuild the very
    provision storm _sync_assignment stopped, just 10 minutes later.

    That includes 📌pinned: pin is permanent ATTRIBUTION, not a residency
    guarantee. This loop treating pin as eager IS the 2026-07-16 incident — the
    operator deleted ae's models and all 65 (65/65 assigned there were pinned)
    re-pulled from here within 10 minutes, filling his workstation to 0 bytes
    free. A pinned model that is absent is absent on purpose until called."""
    while True:
        time.sleep(max(60, int(_RUNTIME_SETTINGS.get("reconcile_interval_s", 600))))
        if restart_requested():
            return                      # stop scheduling transfers into an exit
        try:
            local = set(_models_local(state))
            for mk in list(state.assigned_models):
                if not _eager_pull(mk):
                    continue      # non-static: absent is correct, not drift
                with state._provision_lock:
                    busy = mk in state._provisioning
                if mk not in local and not busy:
                    logger.warning("reconcile: static model %s promises local "
                                   "presence but is missing on disk — "
                                   "re-kicking provisioning", mk)
                    _MODELS_LOCAL_CACHE["at"] = 0.0   # re-check after the pull
                    _kick_provision(state, mk)
        except Exception as exc:  # noqa: BLE001 — the loop must never die
            logger.warning("reconcile iteration failed: %s", exc)


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


# ---------------------------------------------------------------------------
# Self-update — track central's required package version (Slice 1).
#
# Central advertises ``required_pkg_version`` in every register/heartbeat
# response. When it differs from the version installed here, we pip-install the
# pinned target from central's own simple index (the one channel every worker
# can already reach outbound) and re-exec the process. The worker-id is
# persisted, so the restarted agent re-registers as the same worker — central
# sees a brief reconnect, not a new worker.
#
# WP5 worker convergence (post-partition): a worker runs hugpy-fleet PLUS its
# lockstep siblings (hugpy-platform, -control, -storage, -engine, -media, ...
# per install profile), all carrying ONE git-derived version. Upgrading the
# tracked distribution alone leaves the siblings behind — the same silent skew
# the 2026-07-20 incident class is made of. So the converge fetches central's
# ``/api/llm/workers/constraints.txt`` (``name==<required>`` for every workspace
# distribution) and runs ``pip install -U --upgrade-strategy only-if-needed
# -c <that file> <pkg>==<required> [<installed sibling>==<required> ...]``:
# the whole set moves together under the pins. When the constraints cannot be
# fetched (old central, network) the pre-partition single-package ``--no-deps``
# install still runs — with a LOUD warning that the fleet may be skewed.
# ---------------------------------------------------------------------------

# Don't re-attempt the same target version more than once per this window. A
# pinned ``==target`` install means success implies an exact version match, so
# the only way we'd retry is a genuinely failed/unavailable build — back that
# off instead of hammering pip every heartbeat.
_UPDATE_RETRY_BACKOFF = 300.0


def _installed_pkg_version(pkg_name: str) -> str | None:
    """Version pip has ON DISK for ``pkg_name`` (installed dist metadata).

    This is what a NOT-YET-EFFECTIVE self-update flips FIRST: ``pip install``
    swaps the site-packages files (and their ``*.dist-info``) before this process
    has re-exec'd, so ``metadata.version()`` reports the NEW version while the
    OLD modules are still the ones executing in memory. So it is the DISK truth,
    NOT the running-image truth — never report it in the heartbeat (see
    ``_running_pkg_version``). It is still the right thing to gate the self-update
    pip on (has the target already been fetched to disk?).
    """
    from importlib import metadata
    try:
        return metadata.version(pkg_name)
    except metadata.PackageNotFoundError:
        return None


# ── Running-image version, snapshotted at import (honest heartbeat source) ────
# The 2026-07-20 ae incident: after the 0.1.196 pip self-update, ae's heartbeat
# reported pkg_version 0.1.196 / version_ok:true while the RUNNING agent still
# served the OLD route set (404 on the new /slots/<id>/relaunch) — because the
# heartbeat sourced its version from ``_installed_pkg_version`` (live DISK
# metadata, already flipped by pip) instead of from the code actually running.
# That is COSMETIC convergence: central believes the fleet is up to date while a
# worker silently serves stale code.
#
# The honest source is ``hugpy_fleet.__version__`` — the installed distribution
# version (git-derived, one lockstep value for the whole workspace) read ONCE
# when the package was imported at process start. A pip upgrade rewrites the
# dist-info on disk, but the in-memory module object keeps the old value until a
# genuinely fresh process re-imports it. So this constant tells the truth across
# a not-yet-effective upgrade: report OLD until the process really re-execs, and
# central's version_ok stays FALSE (a visible skew) instead of going cosmetically
# green. Captured ONCE here, at import, so nothing can later mutate it.
try:
    from hugpy_fleet import __version__ as _RUNNING_IMAGE_VERSION
except Exception:  # noqa: BLE001 — run-from-copied-file: no package __version__
    _RUNNING_IMAGE_VERSION = None


def _running_pkg_version(pkg_name: str) -> str | None:
    """Version of the CODE THIS PROCESS IS RUNNING — the honest heartbeat source.

    Snapshotted from ``hugpy_fleet.__version__`` at import (above), NOT re-read
    live from dist metadata, so a self-update that pip-installed new files
    on disk but has not yet re-exec'd keeps reporting the OLD version — the truth
    — rather than the disk's new version. Falls back to disk metadata ONLY when
    there is no package ``__version__`` to trust (a standalone copied agent.py),
    where disk metadata is the best signal available.
    """
    if _RUNNING_IMAGE_VERSION:
        return _RUNNING_IMAGE_VERSION
    return _installed_pkg_version(pkg_name)


def _update_state_path(args) -> str:
    return args.id_file + ".update.json"


# ── operator runtime settings (daylight item 3: console-managed serving) ────
# The agent's OWN config file, set from the console via /ops/config. It is the
# SOURCE OF TRUTH over env/unit drop-ins for the keys it holds — the fix for
# the SLOT_COUNT drop-in ghost (a limits.conf silently resurrecting slots on
# every restart). Precedence: settings file > env > built-in default; the
# heartbeat reports the EFFECTIVE values + their source so the console always
# shows truth.

_SETTINGS_KEYS = {"slot_count", "residency", "on_demand_ttl_s",
                  "reconcile_interval_s", "pinned", "comfy_url",
                  "hot_cache_root", "profiles", "model_profiles",
                  "ctx_pct",   # widen key by key (ctx_pct: per-model context %, slice 11)
                  # t21 tolerance bands (per-model maps, siblings of ctx_pct):
                  # the deviation% each explicit allocation may flex under
                  # contention, plus a compress-others priority. Populated by
                  # central's projection of the spill bands (release-time bridge);
                  # the flex engine (_vram_evict_to_fit) reads them here.
                  "ctx_deviation_pct", "vram_deviation_pct",
                  "ram_deviation_pct", "priority",
                  # Eviction policy (2026-07-25).
                  # ⚠ evict_min_residency_s was RETIRED 2026-07-27 — see
                  # _RETIRED_SETTINGS. No time-based veto on eviction exists.
                  #
                  # ⚠ evict_least_reaping is DELIBERATELY ABSENT (operator ruling
                  # 2026-07-25: "yes reject it... best to have these decisions
                  # explicit in proof of action"). It was briefly accepted here
                  # so the relay stayed uniform — but it is FLEET-WIDE policy
                  # owned by central, so a worker would store it, report it back
                  # as saved, and then have central's heartbeat overwrite it on
                  # the very next beat. That is a setting you can write, that
                  # reads back correct, and that silently stops mattering.
                  #
                  # This codebase shipped exactly that shape earlier the SAME DAY
                  # — /assign returned "admission: approved" for a max-gpu that
                  # was never persisted (b0e02ff). Accepting-then-overriding is
                  # the same lie wearing different clothes. Rejecting names the
                  # right door instead; see _fleet_only_hint below.
                  }

# Keys that are FLEET policy, not per-worker settings. Rejected by /ops/config
# with a message naming where they DO belong, rather than a bare "unsupported"
# that leaves the operator guessing which of the two knobs they got wrong.
# Deliberately NOT a toggle: a switch that re-enabled accepting these would have
# exactly one setting — "make the silent override possible again".
_FLEET_ONLY_SETTINGS = {
    "evict_least_reaping": "POST /llm/evict-policy (central owns it; it gates "
                           "the drop pass that central's storage_proposal also "
                           "runs, so one value must serve both or their victim "
                           "sets diverge)",
}

# Keys that USED to exist and deliberately no longer do. Rejected with what
# happened to them, so an old console build or a saved script gets an answer
# instead of a bare "unsupported" that reads like a typo.
_RETIRED_SETTINGS = {
    "evict_min_residency_s": (
        "2026-07-27, operator. It vetoed eviction of any model resident for "
        "less than N seconds — a clock-driven third protection class. Exactly "
        "two block eviction now: 🔒static residency and actively-answering. "
        "Freshness is handled by RANK (sort_key orders on calls, then "
        "last_call), never by a veto"),
}
_SETTINGS_SOURCE: dict = {}              # key -> "settings" | "env" | "default"
_RUNTIME_SETTINGS: dict = {}             # the loaded settings, for live readers
# t21 flex: the ctx% the VRAM admission engine COMMITTED a model to under
# contention (compressed toward its ctx band floor to fit without evicting). It
# is read by _ctx_pct so the SERVED -c matches the KV admission reserved — never
# reserve-small-then-serve-large (that OOMs). Cleared when the model is evicted
# or when it later fits at target (uncontended). Empty == no active flex.
_FLEX_CTX_FLOOR: dict = {}               # model_key -> committed compressed ctx%
# t21 stage (2.5): the honest GGUF PARTIAL-offload the VRAM admission COMMITTED a
# model to when its full weights don't fit the GPU even after flex+evict. Value:
# {"path": served_quant_path, "n": n_gpu_layers}. The n rides the slot child
# cmdline (via the verdict -> slot opts) AND pins the in-process llama_cpp load
# (spill.set_ngl_override on the path) so a sharded model never re-OOMs on the
# shard-blind autofit. Cleared when the model is re-admitted (re-decides) or fits
# fully. Empty == no active partial-offload commitment.
_PARTIAL_NGL: dict = {}                   # model_key -> {"path", "n"}
# MoE expert split (2026-07-24): the split the VRAM admission COMMITTED a
# detected-MoE model to when its FULL weights can never fit the card (the hybrid
# situation). Value: {"path": served_quant_path, "n_cpu_moe": N}. Rides the
# admission verdict -> slot opts -> llama-server --n-cpu-moe (n_gpu_layers=-1 +
# experts on CPU); ALSO the marker that flips this model's calibration verdict
# to "partial" so a 3.2 GiB MoE-split footprint never feeds the full-load
# correction ratio. Cleared alongside _PARTIAL_NGL when an admission re-decides.
_MOE_SPLIT: dict = {}                     # model_key -> {"path", "n_cpu_moe"}
# t28 load-and-learn: the worker's half of the calibration loop.
#   _CALIB_BUFFER      pending calibration_sample dicts, drained into each
#                      heartbeat (worker->central, additive/optional).
#   _CALIB_SAMPLED     model_keys already sampled for their CURRENT residency
#                      episode (dedup — one measured sample per load, re-armed
#                      when the model leaves residency).
#   _CALIB_CORRECTIONS model_key -> clamped learned correction, ADOPTED from the
#                      heartbeat reply (central aggregates + gates + clamps; the
#                      worker just applies it, with a defensive re-clamp). Empty
#                      == no learned number -> the static x1.15 stands.
# All best-effort telemetry: a calibration failure must NEVER break a beat or an
# admission. The whole layer is inert when HUGPY_CALIBRATION=off.
_CALIB_MAXLEN = 64
_CALIB_BUFFER: list = []
_CALIB_SAMPLED: set = set()
_CALIB_CORRECTIONS: dict = {}
_CALIB_LOCK = threading.Lock()
# load_seconds producer (the cold-load half of central's load_metrics — the
# half the completion seam deliberately leaves to the load path): stamp a model
# the first beat it shows as LOADING, measure to the first beat its footprint is
# actually sampled. Beat-granular by construction — honest EMA fodder for
# multi-second loads ("measure, don't bench"); a load that starts and finishes
# inside one beat interval is simply unmeasured (fail-open, never guessed).
_LOAD_STARTED: dict = {}
# k2: the worker's adopted view of the operator's model BLOCK set (aa4aea3),
# learned off the heartbeat reply (worker['blocked_models'] = [model_key, ...]),
# the exact same additive/omit-when-empty wire idiom as calibration. Closes the
# gap the 2026-07-18 ae incident exposed: the block primitive covers every
# CENTRAL path already, but a worker's OWN background provisioning re-kick had no
# way to learn a model was blocked out from under an assignment that is still on
# record (block does not auto-unassign) — so it kept retrying a doomed download
# every reconcile interval indefinitely.
#   _BLOCKED_MODELS   model_keys blocked as of the last heartbeat. Empty ==
#                     nothing blocked (or an older/central without the
#                     feature; the reply just omits the key).
#   _BLOCKED_LOGGED   model_keys a background loop has already logged a skip
#                     for, so the skip logs ONCE per block episode, not every
#                     tick. Re-armed on unblock so a future re-block logs again.
# ONLY gates the worker's own background provisioning (download) loop — never an
# explicit relay request (that stays central's honest-refusal job).
_BLOCKED_MODELS: set = set()
_BLOCKED_LOGGED: set = set()
_BLOCKED_LOCK = threading.Lock()
# ARCHIVE MARK (2026-09-23): central also publishes the models the operator
# marked for archive (worker['archived_models'] = {model_key: "marked for
# archive by <by> at <at>: <reason>"}), a DISTINCT concept from the block. The
# worker's own background loops skip them exactly like blocked models, and the
# skip log quotes the recorded mark rather than claiming a block.
_ARCHIVED_MODELS: dict = {}


def _adopt_blocked_models(worker: "dict | None") -> None:
    """Adopt central's published model BLOCK set from the heartbeat reply.
    Central only sends the key when the set is non-empty; an older/unblocked
    central omits it -> the local set clears -> nothing reads as blocked.
    Mirrors _adopt_calibration."""
    raw = (worker or {}).get("blocked_models") or []
    parsed = ({str(mk) for mk in raw if mk}
              if isinstance(raw, (list, tuple, set)) else set())
    araw = (worker or {}).get("archived_models") or {}
    archived = ({str(k): str(v or "") for k, v in araw.items() if k}
                if isinstance(araw, dict) else {})
    with _BLOCKED_LOCK:
        _BLOCKED_MODELS.clear()
        _BLOCKED_MODELS.update(parsed)
        _ARCHIVED_MODELS.clear()
        _ARCHIVED_MODELS.update(archived)
        # re-arm anything unblocked / unmarked
        _BLOCKED_LOGGED.intersection_update(parsed | set(archived))


def _adopt_least_reaping(worker: "dict | None") -> None:
    """Adopt central's FLEET-WIDE least-reaping policy from the heartbeat reply.

    Why fleet-wide and not a per-worker setting: this knob changes the DROP
    PASS, and central's ``storage_proposal`` runs that same drop pass when it
    previews an eviction. If this worker ran it ON while central previewed it
    OFF, the two would name different victims — precisely the divergence
    ``tests/test_eviction_parity.py`` exists to catch. So central owns ONE
    value and ships it to everyone, the ``blocked_models`` idiom exactly:
    omit-when-default on the wire, so an older central (or one that has never
    been switched) simply doesn't send the key.

    ABSENT means "central has no opinion", which must NOT clobber a local
    drop-in — an operator who set HUGPY_EVICT_LEAST_REAPING on the box before
    central learned the knob should keep it. So absence RESTORES the captured
    base rather than forcing the default, the same revert-to-base rule
    _apply_settings_env uses. A present value always wins: fleet policy
    outranks a local drop-in, because parity is the thing at stake.

    Best-effort and never raises — an adoption failure must not break a beat.
    """
    try:
        raw = (worker or {}).get("evict_least_reaping")
        base_env = f"_HUGPY_BASE_{_ENV_LEAST_REAPING}"
        if raw is None:
            base = os.environ.get(base_env)
            if base:
                os.environ[_ENV_LEAST_REAPING] = base
            elif base is not None:
                os.environ.pop(_ENV_LEAST_REAPING, None)
            return
        if base_env not in os.environ:
            os.environ[base_env] = os.environ.get(_ENV_LEAST_REAPING, "")
        os.environ[_ENV_LEAST_REAPING] = "1" if bool(raw) else "0"
        _SETTINGS_SOURCE["evict_least_reaping"] = "fleet"
    except Exception:  # noqa: BLE001 — heartbeat adoption is best-effort
        logger.debug("least-reaping adoption skipped", exc_info=True)


def _is_blocked_locally(model_key: "str | None") -> bool:
    """True iff ``model_key`` is in this worker's adopted BLOCK set. For gating
    the worker's OWN background loops only — see the module note above."""
    if not model_key:
        return False
    with _BLOCKED_LOCK:
        return model_key in _BLOCKED_MODELS or model_key in _ARCHIVED_MODELS


def _log_blocked_skip_once(model_key: str, where: str) -> None:
    """Log a background loop's skip of a blocked model exactly ONCE per block
    episode, not every tick — the direct fix for the 2026-07-18 ae incident
    (a background reconciler retried a blocked model every beat for ~4.75 hours,
    each attempt failing a full-GPU fit)."""
    with _BLOCKED_LOCK:
        if model_key in _BLOCKED_LOGGED:
            return
        _BLOCKED_LOGGED.add(model_key)
        archived = _ARCHIVED_MODELS.get(model_key)
    if archived is not None and model_key not in _BLOCKED_MODELS:
        logger.info("%s: skipping %s — %s (central's heartbeat reply; won't "
                    "retry until unmarked)", where, model_key,
                    archived or "marked for archive")
        return
    logger.info("%s: skipping %s — blocked from the serving pool by the "
                "operator (won't retry until unblocked)", where, model_key)


_COMFY_URL_BASE_ENV = "_HUGPY_COMFY_URL_BASE"  # sentinel: the pre-projection
# COMFY_URL (systemd drop-in / env / none), captured once and carried across
# os.execv so clearing the setting reverts to the real base, never the last
# projected value.
_ENV_HOT_CACHE_ROOT = "HUGPY_HOT_CACHE_ROOT"   # the env hot_cache.py reads live
# (managers/serve/hot_cache.py::_root). Projecting the setting onto it is why the
# tier needs NO code change to become a per-worker attributable setting.
_HOT_CACHE_ROOT_BASE_ENV = "_HUGPY_HOT_CACHE_ROOT_BASE"  # sentinel: the pre-
# projection HUGPY_HOT_CACHE_ROOT (drop-in / env / none), captured once and
# carried across os.execv so clearing the setting reverts to the real base — the
# exact same dance as _COMFY_URL_BASE_ENV.


def _valid_comfy_url(cu: str) -> bool:
    """True for an http(s) URL that has a host — rejects scheme-only 'http://'
    and accepts a case-insensitive scheme ('HTTP://host' is fine)."""
    from urllib.parse import urlparse
    try:
        p = urlparse(cu.strip())
    except Exception:  # noqa: BLE001 — unparseable string is not a URL
        return False
    return p.scheme.lower() in ("http", "https") and bool(p.netloc)


def _residency(model_key: str) -> str:
    """Per-model residency POLICY (v3 final semantics, operator-locked
    2026-07-05). Exactly TWO tiers:

      * "on-demand" — the DEFAULT (no stored entry): loads on call; holds a
        slot seat until another model needs it (promotion) — slot occupants
        never TTL-yield; idle IN-PROCESS residents do (frees RAM on
        slot-less boxes). Stored legacy entries (if any) read as this
        default too.
      * "static" — the only stored override: locked seat, never swapped out
        or yielded, eager-PULLED to disk (the ONLY tier that pre-pulls — see
        _eager_pull; still loads into VRAM only on call, never on the pull).
        Orthogonal to 📌 pin: pin makes the ATTRIBUTION
        permanent (the override survives unassign-prune), but adds no
        residency or presence promise of its own.

    "Serving" is purely a STATE (a model in a slot), never a policy.
    """
    val = (_RUNTIME_SETTINGS.get("residency") or {}).get(model_key)
    return "static" if val == "static" else "on-demand"


def _pinned(model_key: str) -> bool:
    """📌 pin: the model's ALLOCATION survives restarts — and NOTHING else.

    CANONICAL STATEMENT (operator ruling, 2026-07-17): "the pins only should
    designate that the model allocation survives restarts. the allocation only
    stipulates the routing for that model (to that worker). neither of those
    should have any bearing on the pull or eviction". (Consistent with the
    2026-07-16 answer: "pinned doesnt mean anything aside from: 1) is the model
    attributed to a worker; if yes, then it always will be".)

    So pin answers exactly one question — "does this worker's ALLOCATION
    (routing) for this model survive restarts and unassign attempts?" — with
    "yes, durably". It says NOTHING about when the bytes arrive or whether they
    stay. Concretely, pin DOES:
      * make central refuse unassign while pinned (409),
      * keep residency overrides + the allocation alive across the
        unassign-prune (_prune_stale_residency) and restarts.

    Pin does NOT: pre-fetch, eager-warm, guarantee residency, promise the files
    are on this disk, OR protect the files from eviction/reaping. A pinned model
    is a LAZY download like any other — it arrives on first CALL
    (_ensure_present) — and its files are a normal eviction/reap CANDIDATE
    (budget._is_protected, _reap_scan, workers.storage_proposal all treat pin as
    NON-protecting as of 2026-07-17). Evicting a pinned model's files leaves the
    pin + allocation untouched: routing survives, bytes re-pull on next call.
    Pin's only eviction role is a trivial FIFO tiebreak (unpinned evict first at
    an exact last_picked tie). Do not re-add pin to _eager_pull OR to any disk
    guard: those conflations are the day-one tripwire — the eager one filled the
    operator's workstation to 0 bytes free on 2026-07-16 (ae: 65/65 assigned
    models pinned = every model eager). 🔒static is the ONLY tier that promises
    local presence and protects files.
    """
    return bool((_RUNTIME_SETTINGS.get("pinned") or {}).get(model_key))


def _resolve_model_profile(model_key: str) -> "dict | None":
    """Env-profiles (stage 1) resolver, registered onto managers.serve.profiles
    so the runner spawn seam can decide without reading operator settings itself.

    Reads the live ``model_profiles`` attribution + ``profiles`` manifest from
    _RUNTIME_SETTINGS and returns, for an attributed model,
    ``{'name','state','bin','error'}`` where ``state`` is ready|materializing|
    error and ``bin`` is the profile venv's bin dir ONLY when ready (the value
    the slot child's PATH/interpreter is built from). None when the model has no
    profile — the base serving path is untouched."""
    name = (_RUNTIME_SETTINGS.get("model_profiles") or {}).get(model_key)
    if not name:
        return None
    spec = (_RUNTIME_SETTINGS.get("profiles") or {}).get(name) or {}
    packages = spec.get("packages") or []
    from hugpy_engine.serve import profiles as _profiles
    state = _profiles.state_for(name, packages)
    out = {"name": name, "state": state,
           "bin": _profiles.profile_bin_dir(name) if state == "ready" else None}
    if state == "error":
        out["error"] = (_profiles.read_state(name) or {}).get("error")
    return out


# ---------------------------------------------------------------------------
# Contention-based residency (doctrine 2026-07-11): the worker-side policy
# registered onto dispatch's LRU mechanism (dispatch.set_fit_check /
# set_evictable / set_post_evict_hook). An on-demand model stays hot until a NEW
# load needs its memory; then the LRU on-demand resident yields.
# ---------------------------------------------------------------------------
def _incoming_need_bytes(model_key: str) -> "int | None":
    """Best-effort bytes the incoming model's weights will want (× a small
    headroom factor), resolved from its on-disk size the same way the loader does
    (route_destination). None when the size is unknown — the fit-guard then fails
    OPEN (never blocks an unmeasurable load).

    GGUF landmine (fixed 2026-07-14, mirrors central model_meta): a GGUF repo
    commonly holds several quantizations but only ONE serves, so summing every
    ``.gguf`` in the dir badly overstates the VRAM need — a 24-quant 8B repo read
    as ~94GB (×1.15 ≈ 108GB) and blocked loads on a 7.4GB card even though the
    single served quant is ~5GB. For gguf/llama_cpp we size by the SINGLE
    effective serving quant (+ its mmproj), resolved by the SAME helper central
    uses (``gguf_variants_detail`` → ``effective_bytes``, which honors the
    operator ``gguf_file`` override / ``cfg.filename`` / deterministic auto-rank —
    exactly what the runner loads). We deliberately do NOT fall back to the
    inflated dir-sum for GGUF: an unresolvable effective quant returns None
    (fail-open) rather than re-introducing the very over-count this fixes.
    Non-GGUF frameworks (safetensors/bin) are a single weight set, so the
    dispatch weight-sum stays accurate and is used exactly as before."""
    try:
        from hugpy_storage.model_paths import route_destination
        from hugpy_engine.config.main import get_model_config
        from hugpy_engine.dispatch.dispatch import _dir_size_detail
        cfg = get_model_config(model_key, dict_return=True)
        path = route_destination(cfg)
        if not path:
            return None
        framework = str((cfg or {}).get("framework") or "").lower()
        if framework in ("gguf", "llama_cpp"):
            # Effective-quant-aware sizing. On any resolution miss, return None
            # (fail open) — never the dir sum, which is the bug this fixes.
            try:
                from hugpy_engine.serve.overrides import gguf_variants_detail
                gguf = gguf_variants_detail(model_key, path, cfg) or {}
            except Exception:  # noqa: BLE001 — best-effort; unresolved -> fail open
                gguf = {}
            eff = gguf.get("effective_bytes")
            return int(eff * 1.15) if eff else None
        # Non-GGUF: the shared load-requirement computation (which excludes a
        # duplicate torch serialization shadowed by a safetensors set — a
        # both-formats repo was priced at the dir-sum, ~2x what loads). Falls
        # back to the dispatch weight-sum, preserving the fail-open behavior.
        weight = None
        try:
            from hugpy_engine.serve.overrides import effective_load_requirement
            weight = (effective_load_requirement(model_key, path, cfg) or {}
                      ).get("weights_bytes")
        except Exception:  # noqa: BLE001 — best-effort; fall back to the walk
            weight = None
        if not weight:
            detail = _dir_size_detail(path)
            weight = detail.get("weight_bytes") or detail.get("model_bytes")
        return int(weight * 1.15) if weight else None
    except Exception:  # noqa: BLE001 — best-effort; unknown size -> fail open
        return None


# ── Context (KV) as an allocation variable (slice 11 / t27) ─────────────────
def _model_max_ctx(model_key: str, cfg: dict | None = None) -> int | None:
    """The model's MAX context window (tokens) — the 100% ceiling ctx_pct scales.
    Registry model_max_length first (central's truth), else the trained ctx from
    the model's own metadata (gguf .context_length / config max_position_embeddings)."""
    try:
        from hugpy_engine.config.main import get_model_config
        cfg = cfg if cfg is not None else get_model_config(model_key, dict_return=True)
        mml = (cfg or {}).get("model_max_length") or (cfg or {}).get("tokenizer_model_max_length")
        if mml:
            return int(mml)
    except Exception:  # noqa: BLE001
        pass
    try:
        geo = _model_kv_geometry(model_key, cfg)
        if geo.get("ctx_train"):
            return int(geo["ctx_train"])
    except Exception:  # noqa: BLE001
        pass
    return None


def _ctx_pct(model_key: str) -> int | None:
    """Per-model ctx allocation percentage (1-100) from the worker settings map
    (same seam as residency/pinned). None when unset -> today's default ctx
    behavior is byte-identical (no ctx_pct term forced).

    t21: when the VRAM admission engine has COMMITTED this model to a compressed
    ctx to fit under contention (_FLEX_CTX_FLOOR), that floor WINS — the served
    -c must match the KV the admission reserved. The commit is itself within the
    model's ctx band (never below the floor), so it is always a valid 1..100."""
    floored = _FLEX_CTX_FLOOR.get(model_key)
    if floored is not None:
        try:
            return max(1, min(100, int(floored)))
        except (TypeError, ValueError):
            pass
    val = (_RUNTIME_SETTINGS.get("ctx_pct") or {}).get(model_key)
    try:
        v = int(val)
    except (TypeError, ValueError):
        return None
    if v < 1:
        return 1
    if v > 100:
        return 100
    return v


def _ctx_deviation_pct(model_key: str) -> "float | None":
    """Per-model ctx tolerance band (percent points, 0..100) from settings, or
    None when unset. t21: how far the ctx allocation may flex under contention
    (ctx is the cheapest flex). Reads the target ctx_pct from _RUNTIME_SETTINGS,
    NOT the _FLEX_CTX_FLOOR override, so the band is always relative to the
    operator's target, not a prior compression."""
    val = (_RUNTIME_SETTINGS.get("ctx_deviation_pct") or {}).get(model_key)
    try:
        d = float(val)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(100.0, d))


def _flex_priority(model_key: str) -> int:
    """Per-model flex priority (0 == normal) from settings, via the ONE seam
    (flex.flex_priority_key). Higher compresses/evicts lower-priority neighbours
    first. The operator may change the priority SOURCE — see flex.py."""
    from hugpy_fleet.worker.flex import flex_priority_key
    return flex_priority_key(
        {"priority": (_RUNTIME_SETTINGS.get("priority") or {}).get(model_key)})


def _flex_alloc(model_key: str) -> dict:
    """The per-model explicit-allocation view the flex engine consumes:
    ``{"priority", "ctx_deviation_pct"}``. Central projects the spill bands into
    the settings maps this reads (release-time bridge); tests populate them
    directly. Kept tiny + pure-ish so building subject/resident rows is cheap."""
    return {"priority": (_RUNTIME_SETTINGS.get("priority") or {}).get(model_key),
            "ctx_deviation_pct": _ctx_deviation_pct(model_key)}


def _vram_deviation_pct(model_key: str) -> "float | None":
    """Per-model VRAM tolerance band (percent points, 0..100) from settings, or
    None when unset. Symmetric with _ctx_deviation_pct; the stretch input to
    flex.band_ceiling when the partial-offload budget may reach a model's VRAM
    band CEILING (above its gpu_mem_gib target) under its own need. None (today,
    until central projects gpu_mem_gib_deviation_pct into the worker settings)
    collapses band_ceiling to the gpu_mem_gib target — the offload budget is then
    capped at exactly the explicit gpu_mem_gib, byte-identical to autofit's cap.
    Reads the SAME registry key central's _worker_fit mirror uses for the band
    floor, so worker and central agree the moment central populates it."""
    val = (_RUNTIME_SETTINGS.get("gpu_mem_gib_deviation_pct") or {}).get(model_key)
    try:
        d = float(val)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(100.0, d))


def _gguf_ngl_intent(model_key: str) -> "tuple[str, int | None]":
    """Decode the effective GGUF placement intent (t26) from HUGPY_N_GPU_LAYERS —
    the SAME wire the console's Autofit / Max GPU / CPU only controls ride, set
    per-model by _apply_spill before the load. Returns ``(intent, requested)``:

      * ``"-1"``            -> ("gpu",  None)  — Max GPU (as many as fit; for an
                                                oversize model == autofit, and it
                                                deliberately does NOT squeeze the
                                                ceiling reserve — see flex.plan_
                                                partial_offload).
      * ``0``/off/cpu/none  -> ("cpu",  None)  — CPU only (n_gpu_layers 0).
      * positive int ``N``  -> ("auto", N)     — honor the explicit layer count,
                                                capped to what fits.
      * unset / ``"auto"``  -> ("auto", None)  — autofit layers-that-fit.

    GGUF-only: for a GGUF a positive int IS a real layer count (unlike the
    transformers reading in spill.n_gpu_layers_intent, which collapses it to
    'auto'), so we decode it here rather than reuse that transformers-shaped
    helper."""
    raw = (os.environ.get("HUGPY_N_GPU_LAYERS") or "").strip().lower()
    if raw in ("", "auto"):
        return "auto", None
    if raw in ("off", "cpu", "none"):
        return "cpu", None
    try:
        n = int(raw)
    except ValueError:
        return "auto", None
    if n < 0:
        return "gpu", None
    if n == 0:
        return "cpu", None
    return "auto", n


def _served_gguf_geometry(model_key: str) -> "tuple[str | None, int | None]":
    """``(served_quant_path, total_layers)`` for a GGUF model — the SERVED quant's
    path and its ``.block_count``, resolved exactly as _model_kv_geometry does (no
    parallel reader). ``(None, None)`` for non-GGUF or on any resolution miss.

    Reused for BOTH the partial-offload layer math and the in-process
    n_gpu_layers override (the same path the in-process runner will load), so the
    plan and the load can never key off different files."""
    try:
        from hugpy_storage.model_paths import route_destination
        from hugpy_engine.config.main import get_model_config
        cfg = get_model_config(model_key, dict_return=True)
        if str((cfg or {}).get("framework") or "").lower() not in ("gguf", "llama_cpp"):
            return None, None
        path = route_destination(cfg)
        if not path:
            return None, None
        try:
            from hugpy_engine.serve.serve import _model_file_for
            picked = _model_file_for(model_key, get_model_config(model_key))
            if picked:
                path = picked
        except Exception:  # noqa: BLE001 — fall back to the route path
            pass
        from hugpy_engine import spill
        return path, spill._gguf_layer_count(path)
    except Exception:  # noqa: BLE001 — unresolvable geometry -> caller keeps refusal
        return None, None


def _clear_partial_ngl(model_key: str) -> None:
    """Drop any committed partial-offload OR MoE split for ``model_key`` (this
    admission re-decides) and clear the in-process spill override for its path,
    so a model that now fits fully is never forced back onto a stale plan."""
    prev = _PARTIAL_NGL.pop(model_key, None)
    _MOE_SPLIT.pop(model_key, None)
    if prev and prev.get("path"):
        try:
            from hugpy_engine import spill as _spill
            _spill.clear_ngl_override(prev["path"])
        except Exception:  # noqa: BLE001
            pass


# ── MoE-aware placement (2026-07-24 measured win; operator-grounded) ─────────
# Why: autofit's defect was reducing a TYPED tensor list to an opaque byte-bag —
# "how many whole layers fit" instead of "what KIND of bytes are these". A MoE's
# bytes are ~97% cold expert tensors (each token touches expert_used/expert_count
# of them); the always-hot attention/shared/KV share is tiny. Splitting BY KIND
# (llama-server --n-cpu-moe: experts to CPU, everything else + KV on GPU) beat
# the 17/48 layer split on ae by +59% tok/s at 5x less VRAM. The helpers below
# make every fit/admission path consume that typed view (spill.gguf_moe_detail —
# header-derived, cached) instead of the flat file size.
def _moe_detail_for(model_key: str) -> "dict | None":
    """spill.gguf_moe_detail of the SERVED quant (the same path resolution the
    loader uses), or None for dense / non-GGUF / any miss (degrade to the
    opaque-size path, never raise)."""
    try:
        ppath, _tl = _served_gguf_geometry(model_key)
        if not ppath:
            return None
        from hugpy_engine import spill as _spill
        det = _spill.gguf_moe_detail(ppath)
        return det if det.get("is_moe") else None
    except Exception:  # noqa: BLE001
        return None


def _moe_auto_gpu_budget(model_key: str, path: str) -> "int | None":
    """The VRAM budget the SLOT will plan this MoE against — the SAME arithmetic
    ``slot_agent._moe_gpu_budget`` applies, so admission and launch price ONE
    number: budgetable free VRAM, capped by the model's own ``gpu_mem_gib``
    contract (the ``HUGPY_GPU_MEM_GIB`` wire ``_apply_spill`` writes), less what
    lands on the card beside the weights (the mmproj projector + the KV/context
    reserve).

    ``None`` when the card is unmeasurable — the slot's own degrade there is
    all-experts-to-CPU, so the caller must price THAT, not a plan invented from
    missing data. ``0`` when the reserve consumes the whole budget (the slot
    plans no split at all)."""
    from hugpy_engine import spill as _spill
    budget = _spill.free_vram_bytes()
    try:
        raw = (os.environ.get("HUGPY_GPU_MEM_GIB") or "").strip()
        cap = int(float(raw) * 2 ** 30) if raw else None
    except (TypeError, ValueError):
        cap = None
    if cap is not None:
        budget = min(budget, cap) if budget else cap
    if not budget:
        return None
    # The context reserve is priced at the ctx this model will actually serve
    # (the ctx_pct allocation), exactly as the slot does — an unset ctx_pct
    # leaves it to vram_ctx_reserve_bytes' own resolution, which is what the
    # slot's default ctx path lands on too.
    ctx, _pct, _mx = _resolved_ctx(model_key)
    reserve = _spill.vision_projector_bytes(path)
    try:
        reserve += int(_spill.vram_ctx_reserve_bytes(path, ctx)[0])
    except Exception:  # noqa: BLE001 — a reserve probe never breaks pricing
        pass
    return max(0, int(budget) - reserve)


def _moe_plan_for(model_key: str) -> "dict | None":
    """The MoE split that GOVERNS this model's next load, or None (dense path).

    Returns {"path", "n_cpu_moe", "gpu_weight_bytes", "cpu_bytes", "detail"}:
      * explicit HUGPY_N_CPU_MOE (the n_cpu_moe spill/override wire) WINS — the
        split is priced per-layer-exactly at that N (spill.moe_split_need),
        because that N is exactly what the child launches with;
      * else AUTO-ELIGIBLE: a detected-MoE GGUF with NO explicit layer
        designation (HUGPY_N_GPU_LAYERS unset/auto) and no k37 mode engine
        active — priced by the DENSE-FIRST PLAN against this box's real VRAM
        budget (k55, 2026-07-31). As of 2026-07-25 the split IS the default
        placement for such a model (operator: "the default needs to be the MoE
        split for all GGUFs that it applies to"); the only remaining question
        at the caller is VIABILITY — the experts must fit budgetable host RAM.
    Explicit n_gpu_layers / gpu-only / ram-only / max-ram / explicit-mode all
    return None here: explicit operator placement always wins over auto.

    WHY THE DENSE-FIRST PLAN AND NOT ``moe_split_need`` (k53's flagged
    follow-up). The auto case used to be priced at all-experts-on-CPU
    (MOE_ALL_LAYERS) — backbone-only on the card. Since dense-backbone-first
    (da2b5d9) the child spends whatever budget is LEFT after the backbone on
    expert layers, so it can take materially more VRAM than admission planned,
    and the NEXT admission then sees a fuller card than it expected. Pricing
    the same plan the same way (spill.moe_dense_first_plan against the same
    budget — see _moe_auto_gpu_budget) makes admission and launch agree
    byte-for-byte. An UNMEASURABLE card keeps MOE_ALL_LAYERS, which is
    precisely the slot's own degrade there; a plan that puts EVERYTHING on the
    card (n_cpu_moe 0) moves nothing to RAM, so it is no split at all and falls
    to the dense full-need pricing."""
    det = _moe_detail_for(model_key)
    if not det:
        return None
    try:
        from hugpy_engine import spill as _spill
        ppath, _tl = _served_gguf_geometry(model_key)
        ncm = _spill.n_cpu_moe_env()
        split = None
        if ncm is None:
            intent, requested = _gguf_ngl_intent(model_key)
            if intent != "auto" or requested is not None:
                return None                      # explicit layer designation wins
            if _spill.alloc_mode_env() is not None:
                return None                      # k37 mode engine owns placement
            budget = _moe_auto_gpu_budget(model_key, ppath) if ppath else None
            if budget is None:
                ncm = _spill.MOE_ALL_LAYERS      # unmeasurable card: slot degrade
            elif not budget:
                return None                      # no GPU budget -> slot plans no split
            elif (os.environ.get("HUGPY_GPU_MEM_GIB") or "").strip():
                # A stated per-model VRAM CONTRACT (gpu_mem_gib): the k53
                # remainder-fill stands — the budget is a demand, not a
                # momentary reading — same rule as the slot's free_cap path.
                plan = _spill.moe_dense_first_plan(det, budget)
                if not plan:
                    return None
                ncm = int(plan["n_cpu_moe"])
                split = {"gpu_bytes": int(plan["gpu_bytes"]),
                         "cpu_bytes": int(plan["cpu_bytes"])}
            else:
                # k64 (2026-07-31): the auto budget is MOMENTARY free VRAM,
                # not a stated contract, so it never promotes experts onto the
                # card — all of them stay CPU-side, exactly what the slot's
                # _build_cmd now launches (the two must agree byte-for-byte).
                ncm = _spill.MOE_ALL_LAYERS
        elif ncm <= 0:
            return None                          # explicit "experts on GPU" = no split
        if split is None:
            split = _spill.moe_split_need(det, ncm)
        if not split or not split.get("cpu_bytes"):
            return None                          # nothing moves -> dense pricing
        return {"path": ppath, "n_cpu_moe": int(ncm),
                "gpu_weight_bytes": int(split["gpu_bytes"]),
                "cpu_bytes": int(split["cpu_bytes"]), "detail": det}
    except Exception:  # noqa: BLE001 — pricing gap -> dense path
        return None


def _resolved_ctx(model_key: str, cfg: dict | None = None) -> "tuple[int | None, int | None, int | None]":
    """Resolve the ctx to plan/serve for this model: (ctx_resolved, pct, max).

    ctx_resolved = pct% × model_max (pct of the NATIVE context) — a per-model
    operator choice, bounded ONLY by what actually fits VRAM on this worker at
    this load (spill.served_ctx_for_fit), so the served -c equals the KV that
    fit/admission reserved. There is no hugpy-wide ceiling. When ctx_pct is
    UNSET, returns (None, None, max) so callers fall back to the fit-bounded
    native default ctx path — the variable is opt-in and back-compat by
    construction."""
    pct = _ctx_pct(model_key)
    mx = _model_max_ctx(model_key, cfg)
    if pct is None or not mx:
        return None, pct, mx
    ctx = max(1, int(mx * pct / 100.0))   # per-model ctx_pct choice (kept)
    framework = str((cfg or {}).get("framework") or "").lower()
    if framework in ("gguf", "llama_cpp"):
        try:
            from hugpy_engine import spill
            # Bound the ctx_pct choice by the real VRAM fit for the served quant
            # on THIS worker (never raise it above what the operator asked for).
            ppath = None
            try:
                ppath, _tl = _served_gguf_geometry(model_key)
            except Exception:  # noqa: BLE001
                ppath = None
            if ppath:
                fit = spill.served_ctx_for_fit(ppath)
                if fit:
                    ctx = min(ctx, int(fit))
        except Exception:  # noqa: BLE001 — a fit probe never breaks resolution
            pass
    return ctx, pct, mx


def _model_kv_geometry(model_key: str, cfg: dict | None = None) -> dict:
    """Per-engine KV geometry for a model: {n_layers, n_kv_heads, head_dim,
    ctx_train, dtype}. GGUF reads the served quant's header; transformers reads
    config.json. {} when neither resolves (caller uses the heuristic)."""
    try:
        from hugpy_storage.model_paths import route_destination
        from hugpy_engine.config.main import get_model_config
        from hugpy_engine import spill
        cfg = cfg if cfg is not None else get_model_config(model_key, dict_return=True)
        framework = str((cfg or {}).get("framework") or "").lower()
        path = route_destination(cfg)
        if not path:
            return {}
        if framework in ("gguf", "llama_cpp"):
            # The served quant file (same resolution the loader/need use).
            gguf_path = path
            try:
                from hugpy_engine.serve.serve import _model_file_for
                picked = _model_file_for(model_key, get_model_config(model_key))
                if picked:
                    gguf_path = picked
            except Exception:  # noqa: BLE001
                pass
            return spill._gguf_kv_geometry(gguf_path)
        # transformers/other: config.json geometry.
        import json
        cfgp = os.path.join(path, "config.json") if os.path.isdir(path) else ""
        if cfgp and os.path.isfile(cfgp):
            with open(cfgp, "r", encoding="utf-8") as fh:
                return spill._transformers_kv_geometry(json.load(fh))
    except Exception:  # noqa: BLE001
        pass
    return {}


def _kv_need_bytes(model_key: str, cfg: dict | None = None) -> "tuple[int, dict]":
    """KV-cache bytes for this model at its RESOLVED ctx, plus a detail dict for
    honest reporting. Returns (0, {...}) when ctx_pct is unset (no ctx term —
    today's behavior). Never silently zero when ctx_pct IS set: geometry-missing
    falls to spill.kv_bytes' conservative heuristic (logged)."""
    from hugpy_engine.config.main import get_model_config
    cfg = cfg if cfg is not None else get_model_config(model_key, dict_return=True)
    ctx, pct, mx = _resolved_ctx(model_key, cfg)
    if ctx is None:
        return 0, {"ctx_pct": None, "ctx_resolved": None, "ctx_max": mx,
                   "geometry_source": None}
    from hugpy_engine import spill
    geo = _model_kv_geometry(model_key, cfg)
    framework = str((cfg or {}).get("framework") or "").lower()
    if framework in ("gguf", "llama_cpp"):
        dtype_bytes = 2.0                    # llama.cpp caches fp16 by default
    else:
        dtype_bytes = spill._kv_dtype_bytes(geo.get("dtype"))
    source = "geometry" if (geo.get("n_layers") and geo.get("n_kv_heads")
                            and geo.get("head_dim")) else "heuristic"
    if source == "heuristic":
        logger.warning("kv: no full geometry for %s — using conservative "
                       "heuristic for the ctx reserve (%s tok @ %s%%)",
                       model_key, ctx, pct)
    kv = spill.kv_bytes(ctx_tokens=ctx, n_layers=geo.get("n_layers"),
                        n_kv_heads=geo.get("n_kv_heads"),
                        head_dim=geo.get("head_dim"), dtype_bytes=dtype_bytes)
    return int(kv or 0), {"ctx_pct": pct, "ctx_resolved": ctx, "ctx_max": mx,
                          "geometry_source": source, "kv_bytes": int(kv or 0)}


def _incoming_need_detail(model_key: str) -> dict:
    """THE authoritative fit-NEED for a model: weights + KV(resolved ctx), with
    the SPLIT for honest reporting. All fit paths (contention, slot ceiling,
    vision-fit, slice-10 admission) compute need through this so no path diverges.

    Returns {total, weights, kv, ...ctx detail}. ``total`` is None only when the
    WEIGHT size is unmeasurable (fail-open, exactly as _incoming_need_bytes did).
    The kv term is 0 when ctx_pct is unset — so a model with no ctx allocation is
    byte-identical to today."""
    weights = _incoming_need_bytes(model_key)
    if not weights:
        return {"total": None, "base_total": None, "calibration_correction": 1.0,
                "weights": weights, "kv": 0,
                "ctx_pct": None, "ctx_resolved": None, "ctx_max": None,
                "geometry_source": None}
    try:
        kv, det = _kv_need_bytes(model_key)
    except Exception:  # noqa: BLE001 — KV is additive; never break a working fit
        kv, det = 0, {"ctx_pct": None, "ctx_resolved": None, "ctx_max": None,
                      "geometry_source": None}
    base_total = int(weights) + int(kv or 0)
    # t28 load-and-learn: consult the learned per-model correction (median
    # measured/predicted from real loads, adopted from central, clamped + gated).
    # `total` — what every fit path prices against — becomes the corrected figure;
    # `base_total` (the UNcorrected prediction, incl. the static x1.15) is carried
    # for honest reporting AND is what a calibration_sample records, so the ratio
    # tracks the true base fudge instead of collapsing to a fixpoint at the
    # current correction. None correction -> total == base_total (byte-identical).
    corr = _calib_correction(model_key)
    total = int(base_total * corr) if corr else base_total
    out = {"total": total, "base_total": base_total,
           "calibration_correction": (corr or 1.0),
           "weights": int(weights), "kv": int(kv or 0), **det}
    # MoE expert split (2026-07-24): when a MoE split governs this model
    # (explicit n_cpu_moe, or auto-eligible under the default placement), carry
    # the TYPED need alongside the opaque total: GPU-side = the weights the plan
    # actually leaves on the card — the dense backbone PLUS whatever expert
    # layers the dense-first budget bought (k55; it was backbone-only, which
    # under-priced every launch that had room to spare) — x the same 1.15
    # headroom the weights term uses, + the WHOLE KV (all
    # layers stay on the GPU under the split); CPU-side = the expert bytes
    # (RAM/page-cache, mirroring cpu_resident_bytes accounting). Since
    # 2026-07-25 fit paths PREFER ``moe_split.gpu_total`` whenever it is present
    # and the experts fit RAM — the split is the default placement, so pricing
    # ``total`` would reserve VRAM the child never takes. ``total`` remains the
    # figure for dense models and for any explicitly-designated placement (which
    # produces no ``moe_split`` at all). No calibration correction on the
    # split figure: corrections are learned from FULL loads only (a MoE-split
    # residency reports verdict "partial" and never feeds the ratio).
    try:
        plan = _moe_plan_for(model_key)
    except Exception:  # noqa: BLE001 — additive; never break a working fit
        plan = None
    if plan:
        out["moe_split"] = {
            "path": plan.get("path"),
            "n_cpu_moe": plan["n_cpu_moe"],
            "gpu_total": int(plan["gpu_weight_bytes"] * 1.15) + int(kv or 0),
            "cpu_bytes": plan["cpu_bytes"],
            "expert_count": (plan.get("detail") or {}).get("expert_count"),
            "expert_used_count": (plan.get("detail") or {}).get("expert_used_count"),
            "sparsity": (plan.get("detail") or {}).get("sparsity"),
        }
    return out


# ── "most amicable gguf" — fit-aware quant auto-pick ─────────────────────────
def _quant_autofit_enabled() -> bool:
    """Master switch for the fit-aware quant auto-pick. Default ON;
    HUGPY_QUANT_AUTOFIT=0/off/false/no disables it (the plain q4_k_m-first
    election then governs, exactly as before)."""
    return (os.environ.get("HUGPY_QUANT_AUTOFIT") or "on").strip().lower() not in (
        "0", "off", "false", "no", "")


def _autofit_gguf_pick(model_key: str, model_dir: str, cfg=None) -> "str | None":
    """The LARGEST complete quant that fits THIS box's current free VRAM —
    ``bytes × headroom + KV estimate <= free_vram − admission reserve`` — or
    None when nothing fits / VRAM is unmeasurable (the plain election then
    stands and the worker loads-and-spills exactly as today).

    Registered as overrides.set_gguf_autofit_hook, and only ever consulted via
    ``autofit_gguf_prefer`` — i.e. NEVER when any designation exists (per-worker
    pin, model-wide ``gguf_file``, ``cfg.filename``); an operator pin is never
    silently upgraded. This is deliberately NOT in gguf_election.elect(): the
    election stays pure and worker-agnostic; VRAM belongs to the box asking."""
    if not _quant_autofit_enabled():
        return None
    fv = _free_vram_bytes()
    if fv is None:
        return None
    try:
        from hugpy_engine.spill import total_vram_bytes
        reserve = _vram_ceiling_reserve_bytes(total_vram_bytes())
    except Exception:  # noqa: BLE001 — no total figure -> no reserve derating
        reserve = 0
    try:
        kv, _det = _kv_need_bytes(model_key, cfg if isinstance(cfg, dict) else None)
    except Exception:  # noqa: BLE001 — KV is additive; 0 when unresolvable
        kv = 0
    budget = int(fv) - int(reserve) - int(kv or 0)
    try:
        headroom = float(os.environ.get("HUGPY_VRAM_HEADROOM", "1.15"))
    except ValueError:
        headroom = 1.15
    try:
        from hugpy_engine.serve.overrides import select_fitting_gguf
        pick = select_fitting_gguf(model_key, model_dir, cfg,
                                   budget_bytes=budget, headroom=headroom)
    except Exception:  # noqa: BLE001 — an auto-pick miss must never break a load
        return None
    if pick:
        logger.info("quant autofit: %s -> %s (budget %.1f GiB free %.1f GiB "
                    "reserve %.1f GiB kv %.1f GiB)", model_key, pick,
                    budget / 2**30, fv / 2**30, reserve / 2**30,
                    int(kv or 0) / 2**30)
    return pick


# ── t28 load-and-learn — worker capture + learned-correction application ─────
def _calibration_enabled() -> bool:
    """Master switch (mirrors central's). Default ON; ``off``/``0``/``false``/
    ``no`` makes the whole worker-side layer inert — no capture, no correction."""
    return (os.environ.get("HUGPY_CALIBRATION") or "on").strip().lower() not in (
        "0", "off", "false", "no", "")


def _calib_correction(model_key: str) -> "float | None":
    """The clamped learned correction for a model, or None (the static x1.15
    stands). Read from the map adopted off the heartbeat reply. Central already
    clamps + gates; the [0.8, 1.5] re-clamp here is a defensive safety net so a
    malformed reply can never move need-pricing outside the doctrine band."""
    if not _calibration_enabled():
        return None
    with _CALIB_LOCK:
        c = _CALIB_CORRECTIONS.get(model_key)
    if c is None:
        return None
    try:
        return max(0.8, min(1.5, float(c)))
    except (TypeError, ValueError):
        return None


def _calib_verdict(device: "str | None", ngl, total_layers) -> str:
    """Classify a residency into the placement verdict a calibration sample
    carries: ``full`` (weights+kv fully GPU-resident — the ONLY class that feeds
    the ratio), ``partial`` (some layers on CPU), ``cpu``, or ``unknown``."""
    dev = (device or "").lower() if device else None
    try:
        ngl = int(ngl) if ngl is not None else None
    except (TypeError, ValueError):
        ngl = None
    try:
        tl = int(total_layers) if total_layers is not None else None
    except (TypeError, ValueError):
        tl = None
    if ngl == 0 or dev == "cpu":
        return "cpu"
    if ngl is not None and ngl > 0 and tl and ngl < tl:
        return "partial"
    if dev == "cuda" or (ngl is not None and (ngl == -1 or ngl > 0)):
        return "full"
    return "unknown"


def _build_calibration_success(mk: str, row: dict) -> "dict | None":
    """A measured calibration_sample from an allocation row (post-load): the
    BASE (uncorrected) prediction split paired with the measured VRAM/RSS. None
    when the prediction can't be sized. Omits None fields (additive wire)."""
    try:
        det = _incoming_need_detail(mk)
    except Exception:  # noqa: BLE001 — capture must never raise on the beat
        return None
    base_total = det.get("base_total")
    if not base_total:
        return None
    verdict = _calib_verdict(row.get("device"), row.get("n_gpu_layers"),
                             row.get("total_layers"))
    # MoE expert split (2026-07-24): a split residency launches with ngl=-1 but
    # only the non-expert share lands on the GPU — letting it read as "full"
    # would feed a wildly-low measured/predicted ratio into the full-load
    # correction (3.2G measured vs ~48G predicted). Classify it as "partial"
    # (the existing excluded-from-ratio vocabulary; the wire stays additive).
    is_moe = mk in _MOE_SPLIT or (row or {}).get("n_cpu_moe") is not None
    if verdict == "full" and is_moe:
        verdict = "partial"
    sample = {
        "model_key": mk,
        "engine": _model_framework(mk),
        "needs_weights_bytes": det.get("weights"),
        "needs_kv_bytes": det.get("kv"),
        "ctx_pct": det.get("ctx_pct"),
        "need_total_bytes": base_total,
        "verdict": verdict,
        "n_gpu_layers": row.get("n_gpu_layers"),
        "total_layers": row.get("total_layers"),
        "vram_bytes": row.get("vram_bytes"),
        "rss_bytes": row.get("rss_bytes"),
        "device": row.get("device"),
        "ok": True,
        "ts": time.time(),
    }
    if is_moe:
        # For central's load-metrics feed (derive_variant needs it); the
        # calibration table ignores unknown keys — the wire stays additive.
        sample["moe_capable"] = True
    return {k: v for k, v in sample.items() if v is not None}


def _collect_calibration_from_allocations(allocs: "list | None",
                                          loading: "list | None" = None) -> None:
    """Emit ONE measured sample per residency episode. Keys off the SAME
    allocations view the heartbeat already computes (per-process nvidia-smi
    VRAM), so it captures EVERY load path — on-demand, slot, warm/probe, reconcile
    — uniformly, and dedups via _CALIB_SAMPLED. Samples only once a footprint is
    actually measured (vram_bytes present); departed residents are re-armed so a
    reload re-samples.

    ``loading`` (the beat's already-computed loading view — never a second
    probe) drives the load_seconds stamp: first beat LOADING starts the clock,
    the sample beat stops it. A stamp whose model is neither loading nor
    resident (an aborted load) is dropped unmeasured."""
    if not _calibration_enabled():
        return
    now = time.time()
    resident_now: set = set()
    new_samples: list = []
    with _CALIB_LOCK:
        for mk in (loading or []):
            _LOAD_STARTED.setdefault(mk, now)
    for row in (allocs or []):
        mk = (row or {}).get("model_key")
        if not mk:
            continue
        resident_now.add(mk)
        if row.get("vram_bytes") is None:
            continue                        # not measured yet — wait for a beat that does
        with _CALIB_LOCK:
            if mk in _CALIB_SAMPLED:
                continue
        s = _build_calibration_success(mk, row)
        if s:
            with _CALIB_LOCK:
                started = _LOAD_STARTED.pop(mk, None)
            if started is not None and now > started:
                s["load_seconds"] = round(now - started, 3)
            new_samples.append(s)
            with _CALIB_LOCK:
                _CALIB_SAMPLED.add(mk)
    with _CALIB_LOCK:
        _CALIB_SAMPLED.intersection_update(resident_now)
        for mk in [k for k in _LOAD_STARTED
                   if k not in resident_now and k not in (loading or [])]:
            _LOAD_STARTED.pop(mk, None)     # aborted load — never measured
        _CALIB_BUFFER.extend(new_samples)
        if len(_CALIB_BUFFER) > _CALIB_MAXLEN:
            del _CALIB_BUFFER[:-_CALIB_MAXLEN]


def _record_calibration_refuse(model_key: str, det: "dict | None") -> None:
    """A load-FAIL sample from the VRAM admission refusal: the prediction with no
    successful measurement (verdict=refuse, ok=False). Stored for telemetry /
    future regression; EXCLUDED from the ratio (it never loaded)."""
    if not _calibration_enabled():
        return
    try:
        det = det or {}
        sample = {
            "model_key": model_key,
            "engine": _model_framework(model_key),
            "needs_weights_bytes": det.get("weights"),
            "needs_kv_bytes": det.get("kv"),
            "ctx_pct": det.get("ctx_pct"),
            "need_total_bytes": det.get("base_total") or det.get("total"),
            "verdict": "refuse",
            "ok": False,
            "ts": time.time(),
        }
        sample = {k: v for k, v in sample.items() if v is not None}
        with _CALIB_LOCK:
            _CALIB_BUFFER.append(sample)
            if len(_CALIB_BUFFER) > _CALIB_MAXLEN:
                del _CALIB_BUFFER[:-_CALIB_MAXLEN]
    except Exception:  # noqa: BLE001 — telemetry must never break admission
        pass


def _drain_calibration_samples() -> list:
    """Snapshot + clear the pending sample buffer for the heartbeat payload.
    Best-effort telemetry: a rare loss on a failed beat is acceptable (the next
    residency re-samples)."""
    with _CALIB_LOCK:
        if not _CALIB_BUFFER:
            return []
        out = list(_CALIB_BUFFER)
        _CALIB_BUFFER.clear()
    return out


def _adopt_calibration(worker: "dict | None") -> None:
    """Adopt central's published per-model corrections from the heartbeat reply
    (``worker['calibration'] = {mk: {"correction", ...}}``). Central only sends
    gate-passing, clamped values; an older/off central simply omits the key ->
    the map clears -> the static x1.15 stands. Mirrors _apply_central_limits."""
    corr = (worker or {}).get("calibration") or {}
    parsed: dict = {}
    if isinstance(corr, dict):
        for mk, info in corr.items():
            try:
                val = info.get("correction") if isinstance(info, dict) else info
                if val is not None:
                    parsed[str(mk)] = float(val)
            except (TypeError, ValueError):
                continue
    with _CALIB_LOCK:
        _CALIB_CORRECTIONS.clear()
        _CALIB_CORRECTIONS.update(parsed)


def _worker_fit_check(model_key: str) -> bool:
    """Contention fit-guard (dispatch.set_fit_check). True when the incoming load
    fits in current headroom WITHOUT yielding a resident; False = memory pressure
    -> yield the LRU on-demand resident.

    GPU box: the newcomer wants to be GPU-resident (hot), so it fits when free
    VRAM holds its weights. If a GPU is present but VRAM can't, that's the
    contention that yields an idle on-demand resident to keep the newcomer on the
    GPU (doctrine: minimize load time, keep models hot); when nothing is left to
    yield the loop stops and the normal autofit path spills to CPU exactly as
    today. CPU-only box: contention is on RAM. Fails OPEN when the size or both
    pools are unmeasurable — an unmeasurable load proceeds exactly as today.

    NEED = weights + KV(resolved ctx) (slice 11): the ctx tax is planned, not
    discovered at OOM. _incoming_need_detail is the ONE authoritative need; kv is
    0 when ctx_pct is unset (byte-identical to today).

    MoE: when a MoE split governs the plan, the GPU-side need is the TYPED
    non-expert share (+KV) — the split is what will actually load, so pricing
    the full file misjudges exactly the case the split wins (a 41.6GB MoE on an
    empty 23.6GiB card fits FINE). Since 2026-07-25 the split is the DEFAULT
    placement, so the typed need is consulted FIRST rather than only as a
    fallback after the full need fails — otherwise a fits-whole MoE is priced at
    ~5x the VRAM the child actually takes ("need-calc prices experts as GPU").
    The expert (CPU) share is checked against free RAM, failing open when RAM
    is unmeasurable."""
    det = _incoming_need_detail(model_key)
    need = det.get("total")
    if not need:
        return True
    ms = det.get("moe_split")
    fv = _free_vram_bytes()
    if fv is not None:
        if ms:
            fr = _free_ram_bytes()
            experts_fit = fr is None or fr >= int(ms.get("cpu_bytes") or 0)
            if experts_fit and fv >= int(ms.get("gpu_total") or 0):
                return True
            # Experts can't fit RAM (or even the split's GPU share doesn't fit):
            # fall through to the full-need question — the split degrades to the
            # autofit layer placement, which is what `need` prices.
        return fv >= need
    fr = _free_ram_bytes()
    if fr is not None:
        return fr >= need
    return True


_DEFAULT_VRAM_CEILING_FRAC = 0.90


def _vram_ceiling_frac_explicit() -> "float | None":
    """The OPERATOR'S HUGPY_VRAM_CEILING_FRAC, or None when it is unset (or
    unusable). None is the signal that the DEFAULT cushion below governs; a
    usable value means the operator asked for the fraction-of-the-card meaning
    and gets it verbatim. Unset/garbage/out-of-range are all "not explicit" —
    a fat-fingered env can never invert the gate."""
    raw = os.environ.get("HUGPY_VRAM_CEILING_FRAC")
    if raw is None or not str(raw).strip():
        return None
    try:
        val = float(raw)
    except ValueError:
        logger.warning("ignoring non-numeric HUGPY_VRAM_CEILING_FRAC=%r; using %s",
                       raw, _DEFAULT_VRAM_CEILING_FRAC)
        return None
    if not (0.0 < val <= 1.0):
        logger.warning("HUGPY_VRAM_CEILING_FRAC=%r out of (0,1]; using %s", raw,
                       _DEFAULT_VRAM_CEILING_FRAC)
        return None
    return val


def _vram_ceiling_frac() -> float:
    """The real-VRAM ceiling as a fraction of total card VRAM
    (HUGPY_VRAM_CEILING_FRAC, default 0.90). Unchanged: this is what the
    PRESSURE sweep asks ("is the card at/over the ceiling right now") and what
    an explicit operator override means. The ADMISSION reserve is
    ``_vram_ceiling_reserve_bytes`` — see the note there for why the default no
    longer derives from this fraction."""
    val = _vram_ceiling_frac_explicit()
    return _DEFAULT_VRAM_CEILING_FRAC if val is None else val


# ── THE ADMISSION CUSHION (2026-07-27) ──────────────────────────────────────
# THE BUG. The admission gate was `(free - need) >= total * (1 - 0.90)`, so on
# ae's 24 GB 3090 it demanded 2.36 GiB of the card still free AFTER the weights
# landed. Live refusal: "needs 21.1 GB, 21.3 GB free of 23.6 GB (2.4 GB ceiling
# reserve); evicted 1 idle resident(s) freeing 21.0 GB" — a model the box can
# genuinely run, refused on a card it had just emptied for it.
#
# WHY IT WAS WRONG, in units. `need` is `_incoming_need_detail(...)["total"]` =
# on-disk weights x 1.15 + KV(resolved ctx) (x any learned calibration). It
# ALREADY carries the whole context tax. What it does NOT carry is the
# compute-graph + logits + CUDA-runtime residual that lands beside the KV —
# and that residual is the ONLY thing this ceiling legitimately protects.
# Sizing it as 10% of TOTAL therefore (a) re-charged for KV that is already
# inside `need`, and (b) scaled the cushion with the CARD when the residual
# does not scale with the card at all. That is the same shape as the flat
# 2.5 GiB ctx reserve retired in 0.1.218 (spill.vram_ctx_reserve_bytes), just
# with the proportionality inverted.
#
# THE NUMBER. spill measured that residual directly: flux2-klein-9b q4_k_m on
# computron, 36 layers @ ctx 16384 — 7441 MiB on the card against 4.68 GiB of
# weights and an EXACT 2.25 GiB KV, leaving a 348 MiB residual that is
# ctx-independent (llama.cpp sizes the graph off n_ubatch and n_vocab). spill
# already allows 512 MiB for it (_CTX_COMPUTE_RESERVE_BYTES, ~47% headroom over
# the measurement) and this gate now uses THAT SAME CONSTANT, imported, so the
# two fit paths cannot drift. For a big model the `x 1.15` weights fudge inside
# `need` adds ~2.4 GiB of further slack on a 21 GiB load, which is why a flat
# cushion stays safe as models grow.
#
# THE STACK. `_free_vram_bytes()` is BUDGETABLE free — spill.free_vram_bytes
# already deducted HUGPY_VRAM_RESERVE_GIB (1.0 GiB) for GPU consumers central
# cannot see. That floor is real device headroom the load will never touch, so
# charging the cushion ON TOP of it would be the third margin for the second
# concern. Same un-stacking ruling spill made for the whole-fit test
# ("raw free - max(need, floor) >= file"): the guarantee is
#
#     raw free after the load  >=  max(external floor, compute cushion)
#
# which on the budgetable figure this gate actually reads is
#
#     free_budgetable - need   >=  max(0, cushion - floor)
#
# On ae (floor 1.0 GiB, cushion 512 MiB) that reserve is 0 and the live case
# admits with 1.2 GiB of RAW device headroom left — 3.4x the measured residual.
# On a box with HUGPY_VRAM_RESERVE_GIB=0 the cushion applies in full.
#
# NEVER STRICTER THAN BEFORE. The cushion is additionally clamped to today's
# (1 - 0.90) x total, so max(0, min(cushion, 0.10 x total) - floor)
# <= 0.10 x total for every card: the new default cannot refuse anything the
# old default admitted. Small cards therefore keep exactly today's ceiling term
# (a 4 GiB card: 410 MiB) instead of inheriting a constant sized for a big one
# — the 0.1.218 lesson, applied in the other direction.
#
# DEGRADE-NOT-GUESS is unchanged: unmeasurable total/free/need still fails OPEN
# at every call site, and an EXPLICIT HUGPY_VRAM_CEILING_FRAC skips all of this
# and gets `total * (1 - frac)` verbatim, exactly as before.
_VRAM_CEILING_CUSHION_FALLBACK = 512 * 2**20


def _vram_ceiling_cushion_bytes() -> int:
    """The compute-graph/activation/fragmentation cushion, in bytes.

    ``HUGPY_VRAM_CEILING_CUSHION_GIB`` overrides (module env convention, same
    GIB shape as HUGPY_VRAM_RESERVE_GIB / HUGPY_VRAM_CTX_RESERVE_GIB); a
    negative or non-numeric value is ignored. Otherwise spill's MEASURED
    ``_CTX_COMPUTE_RESERVE_BYTES`` (512 MiB) — imported, not copied, so the
    admission gate and autofit price the same physical thing identically."""
    raw = os.environ.get("HUGPY_VRAM_CEILING_CUSHION_GIB")
    if raw is not None and str(raw).strip():
        try:
            val = float(raw)
            if val >= 0:
                return int(val * 2**30)
            logger.warning("ignoring negative HUGPY_VRAM_CEILING_CUSHION_GIB=%r", raw)
        except ValueError:
            logger.warning("ignoring non-numeric HUGPY_VRAM_CEILING_CUSHION_GIB=%r",
                           raw)
    try:
        from hugpy_engine.spill import _CTX_COMPUTE_RESERVE_BYTES
        return int(_CTX_COMPUTE_RESERVE_BYTES)
    except Exception:  # noqa: BLE001 — never break admission over a cushion read
        return _VRAM_CEILING_CUSHION_FALLBACK


def _external_vram_floor_bytes() -> int:
    """VRAM already held out of ``_free_vram_bytes()`` for consumers this worker
    cannot see (spill.vram_reserve_bytes / HUGPY_VRAM_RESERVE_GIB, 1.0 GiB).

    It is REAL post-load device headroom, which is why the cushion un-stacks
    against it. Unreadable -> 0, i.e. the cushion applies in full (the
    conservative direction)."""
    try:
        from hugpy_engine.spill import vram_reserve_bytes
        return max(0, int(vram_reserve_bytes()))
    except Exception:  # noqa: BLE001
        return 0


def _vram_ceiling_reserve_bytes(total: "int | None") -> int:
    """THE admission reserve: BUDGETABLE free VRAM that must remain after the
    incoming need lands. See the note above for the derivation.

    * explicit HUGPY_VRAM_CEILING_FRAC -> ``total * (1 - frac)``, verbatim,
      byte-identical to the pre-2026-07-27 gate.
    * default -> ``max(0, min(cushion, total * 0.10) - external floor)``.
    """
    try:
        total = int(total or 0)
    except (TypeError, ValueError):
        total = 0
    if total <= 0:
        return 0
    frac = _vram_ceiling_frac_explicit()
    if frac is not None:
        return int(total * (1.0 - frac))
    cushion = min(_vram_ceiling_cushion_bytes(),
                  int(total * (1.0 - _DEFAULT_VRAM_CEILING_FRAC)))
    return max(0, cushion - _external_vram_floor_bytes())


def _vram_empty_card_budget(total: "int | None") -> int:
    """The largest ``need`` this card could EVER admit (empty card), under the
    SAME arithmetic the fit gate uses: total, less the external floor that never
    reaches the budgetable figure, less the admission reserve. Reporting//log
    only — it is what makes "the full weights can never fit this card" a true
    statement rather than a differently-computed guess."""
    try:
        total = int(total or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, total - _external_vram_floor_bytes()
               - _vram_ceiling_reserve_bytes(total))


def _vram_pressure_reserve_bytes(total: "int | None") -> int:
    """The IDLE-PRESSURE threshold for ``_vram_headroom_sweep``, deliberately
    still ``total * (1 - ceiling frac)``.

    This is NOT the admission gate and must not be unified with it. It asks a
    different question — "is the card under pressure RIGHT NOW, with no load
    driving admission" — and its answer only ever evicts an idle resident; it
    never refuses anything. Sizing it with the admission cushion would make it
    ~0 on any box with the default 1.0 GiB external floor (that floor is already
    out of the free read), retiring the deadlock-breaker the addendum exists
    for. With an explicit HUGPY_VRAM_CEILING_FRAC the two agree exactly, as
    they always did."""
    try:
        total = int(total or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, int(total * (1.0 - _vram_ceiling_frac())))


def _total_vram_bytes() -> "int | None":
    """Total INSTALLED VRAM (spill.total_vram_bytes) — RAW device capacity, or
    None when no GPU / can't measure. Mirrors _free_vram_bytes' import guard so a
    missing torch/nvidia-smi degrades to None, never raises."""
    try:
        from hugpy_engine.spill import total_vram_bytes
        return total_vram_bytes()
    except Exception:  # noqa: BLE001
        return None


def _worker_slot_fit_check(model_key: str) -> bool:
    """Real-VRAM CEILING gate (slots.set_fit_check), Fix A (2026-07-15). True when
    loading ``model_key`` still leaves the card real working room given REAL
    current free VRAM — i.e. at least ``_vram_ceiling_reserve_bytes`` of the
    budgetable free figure remains AFTER the weights land. False when it would
    breach that (the slot scheduler then evicts the coldest on-demand
    occupant(s) and re-checks).

    The reserve is THE SAME binding ``_vram_evict_to_fit`` admits against — a
    bounded compute/activation cushion since 2026-07-27, not (1 - frac) of the
    whole card (which double-charged the KV already inside ``need``). These two
    are siblings and must never disagree.

    This is distinct from _worker_fit_check (the in-process contention guard,
    which asks "does it fit WITHOUT yielding a resident"): this gate answers "does
    the WHOLE card stay under the ceiling", so it reacts to OUT-OF-BAND process
    VRAM growth (ComfyUI) that managed-model bookkeeping is blind to — free VRAM
    is the real device read (torch.cuda.mem_get_info, ComfyUI-visible).

    Fails OPEN (True) when free VRAM, total VRAM, or the incoming need is unknown
    (no GPU / can't tell) — NEVER block a load because we couldn't measure. That
    keeps a no-GPU / unmeasurable box byte-identical to today (the gate is a
    no-op there)."""
    total = _total_vram_bytes()
    if not total:
        return True                          # no GPU / can't measure -> allow
    fv = _free_vram_bytes()
    if fv is None:
        return True                          # can't read free VRAM -> allow
    # NEED = weights + KV(resolved ctx) (slice 11) — the ONE authoritative need,
    # so the ceiling gate reserves the ctx tax too. kv=0 when ctx_pct unset.
    det = _incoming_need_detail(model_key)
    need = det.get("total")
    if not need:
        return True                          # unknown weight size -> allow
    # THE SAME reserve _vram_evict_to_fit admits against (_vram_ceiling_reserve_
    # bytes). These two are siblings asking one question — "does this need fit
    # under the ceiling?" — from two entry points (slot routing vs the admission
    # choke point). They must never disagree, or the slot pool refuses a seat the
    # admission gate just granted (and spins the ceiling loop for nothing).
    headroom = _vram_ceiling_reserve_bytes(total)
    # Loading consumes ~need; the card is OK if free-after-load still leaves the
    # cushion. Equivalent to "post-load fill <= ceiling".
    # MoE: when a split governs the plan the load lands only the non-expert
    # share (+KV) on the card — gate on THAT typed need (expert share vs free
    # RAM, failing open when RAM is unmeasurable). This is what lets the /probe
    # load of a 41.6GB MoE pass on an empty 23.6GiB card. Checked
    # BEFORE the full-need gate since 2026-07-25 (the split is the DEFAULT
    # placement, so the full need is not what will be reserved).
    ms = det.get("moe_split")
    if ms and (fv - int(ms.get("gpu_total") or 0)) >= headroom:
        fr = _free_ram_bytes()
        if fr is None or fr >= int(ms.get("cpu_bytes") or 0):
            return True
        # Experts don't fit RAM -> the split degrades to the autofit layer
        # placement, which is what the full need below prices.
    return (fv - need) >= headroom


def _worker_evictable(model_key: str) -> bool:
    """Contention yield predicate (dispatch.set_evictable). A model may yield its
    in-process residency ONLY if it is not static, has NO in-flight generation
    (gate permits), and isn't slot-backed (a slot child's weights live in another
    process — dropping the proxy frees nothing here and breaks the seat).

    Tier semantics (operator, 2026-07-15): 📌 pin = the worker is DESIGNATED that
    model (durable assignment across restarts), NOT a resource lock — so a pinned
    model DOES yield to contention (its weights free for a new load; the pin is
    untouched and it reloads on demand). 🔒 static is the only residency lock
    ("static means cannot evict") and never yields. A model mid-generation is
    skipped (the next LRU is chosen) and becomes evictable once its gate permits.
    (Pre-2026-07-15 pinned also never yielded — that conflated designation with a
    resource lock, so pin-bloat could deadlock the make-room evictor.)"""
    if _residency(model_key) == "static":
        return False
    try:
        if gen_gate.in_flight(model_key) > 0:
            return False
    except Exception:  # noqa: BLE001 — can't tell -> don't yield a possibly-busy model
        return False
    try:
        from hugpy_engine.llama.runners.get import slot_backed_model_keys
        if model_key in (slot_backed_model_keys() or set()):
            return False
    except Exception:  # noqa: BLE001 — can't tell slot-backing -> allow (in-process default)
        pass
    return True


def _worker_evict_skip_reason(model_key: str) -> str:
    """WHY ``_worker_evictable`` said no (dispatch.set_evict_reason).

    TELEMETRY ONLY — this decides nothing. It re-reads the same three clauses
    the predicate above applies, in the same order, purely so a skipped
    candidate streams to the console with the clause that protected it instead
    of an opaque "not chosen". Existing vocabulary only: 🔒static and actively-
    answering are the two protection classes; slot-backing is a mechanism fact
    (the weights are in another process, so dropping the proxy frees nothing
    here), not a third class.

    Never raises — an unreadable reason degrades to the generic label rather
    than disturbing a yield loop it has no business influencing."""
    try:
        if _residency(model_key) == "static":
            return "static"
        try:
            if gen_gate.in_flight(model_key) > 0:
                return "actively-replying"
        except Exception:  # noqa: BLE001
            return "busy-unknown"
        try:
            from hugpy_engine.llama.runners.get import slot_backed_model_keys
            if model_key in (slot_backed_model_keys() or set()):
                return "slot-backed"
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        return "not-evictable"
    return "not-evictable"


# ── targeted eviction (evict <model_key>) ───────────────────────────────────
# Central signals `evict <model_key>` (never a raw PID — PIDs are per-box and get
# recycled). The worker resolves the model_key to its LIVE hosting handle AT
# eviction time, verifies identity, and frees it with the mechanism that matches
# HOW the model is hosted. This is the surgical bookend to /models/unload (which
# is coarse: one model_key or all) — same "stays ASSIGNED, just not resident"
# semantics, but it picks slot-kill vs in-process-drop vs comfy-free per model.

def _evict_gate(model_key: str) -> "tuple[bool, str]":
    """Eviction permission for the destructive evict verb: (allowed, reason).

    Tier semantics RE-CLARIFIED by the operator 2026-07-15: 📌 pin means ONLY
    that this worker is DESIGNATED to serve the model (a durable assignment that
    survives hard restarts) — it is NOT a resource lock and MUST NOT block
    eviction. 🔒 static is the ONLY residency lock ("static means cannot evict").
    So a pinned-but-on-demand model evicts freely: its weights are freed and it
    reloads on the next call, while the pin (designation) is untouched. This is
    why a fully-pinned worker is harmless — designation, not a VRAM hoard.

    Only static (and an in-flight generation) protects here. Slot-backing is NOT
    a blocker — evicting a slot child is the whole point of this verb. ``force``
    (checked by the caller) overrides every clause. A model mid-generation is
    protected unless forced: we never rip weights out from under a running
    request. (Pre-2026-07-15 this also refused pinned models — that conflated
    designation with a resource lock and jammed eviction under pin-bloat.)"""
    if _residency(model_key) == "static":
        return False, "static (locked residency) — pass force to override"
    try:
        if gen_gate.in_flight(model_key) > 0:
            return False, "in-flight generation — pass force to override"
    except Exception:  # noqa: BLE001 — can't tell -> treat as busy (don't rip a maybe-busy model)
        return False, "cannot determine in-flight state — pass force to override"
    return True, ""


def _resolve_slot_handle(model_key: str) -> "dict | None":
    """Resolve model_key -> the slot HANDLE currently serving it, or None.

    Returns {"control_url", "child_pid", "endpoint"} from a LIVE slot-pool
    status read (never a cached/central-supplied value). ``child_pid`` is the
    llama-server/llama_cpp.server child that actually holds the VRAM. Returns
    None when no slot is serving this model_key right now."""
    try:
        from hugpy_engine.serve.slots import SlotPool
        for s in SlotPool().statuses():
            if s.get("model_key") == model_key and s.get("child_pid"):
                return {"control_url": s.get("_control"),
                        "child_pid": s.get("child_pid"),
                        "endpoint": s.get("endpoint")}
    except Exception:  # noqa: BLE001 — no slots / pool error -> not slot-hosted here
        return None
    return None


def _is_inprocess_resident(model_key: str) -> bool:
    """True if this worker holds the model's WEIGHTS in its OWN python process —
    a GGUF llama handle, a dispatch-cached torch runner, a diffusers pipeline, or
    a torch model nvidia-smi attributes to our PID. A slot-backed HTTP proxy
    (base_url, no ``llm``) is NOT a resident (its weights live in the child), so
    the slot branch must be resolved BEFORE this is consulted."""
    # GGUF heavy singletons with a real in-process llm handle.
    try:
        from hugpy_engine.llama.runners.get import _LLAMA_INSTANCES, _LLAMA_LOCK
        with _LLAMA_LOCK:
            for k, r in list(_LLAMA_INSTANCES.items()):
                if k == model_key and getattr(r, "llm", None) is not None:
                    return True
    except Exception:  # noqa: BLE001
        pass
    # dispatch-cached in-process runners (torch/vision/etc.), excluding slot proxies.
    try:
        from hugpy_engine.dispatch import dispatch as _d
        from hugpy_engine.llama.runners.get import slot_backed_model_keys
        slot_keys = slot_backed_model_keys() or set()
        with _d._INSTANCES_LOCK:
            keys = [k[0] if isinstance(k, tuple) and k else k
                    for k in list(_d._INSTANCES)]
        if model_key in keys and model_key not in slot_keys:
            return True
    except Exception:  # noqa: BLE001
        pass
    # diffusers imagegen pipelines (class-level singleton, not on a runner attr).
    try:
        from hugpy_media.imagegen import imagegen_runner as _ig
        for clsname in ("ImageGenRunner", "Img2ImgRunner"):
            cache = getattr(getattr(_ig, clsname, None), "_PIPELINES", None)
            if isinstance(cache, dict) and model_key in cache:
                return True
    except Exception:  # noqa: BLE001
        pass
    # last resort: torch attributes real VRAM to this model under our PID.
    try:
        if model_key in _inprocess_gpu_bytes():
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _drop_inprocess_model(model_key: str) -> bool:
    """Drop the in-process refs for ``model_key`` and free its weights WITHOUT
    killing the worker PID (siblings share it). dispatch.evict cascades through
    the dispatch adapter cache AND the GGUF heavy singleton; the diffusers
    pipeline lives in a class-level cache that cascade misses, so drop it too.
    _trim_host_ram() then hands the freed arena + torch CUDA cache back."""
    dropped = False
    try:
        from hugpy_engine.dispatch.dispatch import evict as _evict
        dropped = bool(_evict(model_key)) or dropped
    except Exception:  # noqa: BLE001
        pass
    try:
        from hugpy_media.imagegen import imagegen_runner as _ig
        for clsname in ("ImageGenRunner", "Img2ImgRunner"):
            cache = getattr(getattr(_ig, clsname, None), "_PIPELINES", None)
            if isinstance(cache, dict) and cache.pop(model_key, None) is not None:
                dropped = True
    except Exception:  # noqa: BLE001
        pass
    # TRANSFORMERS CAUSAL-LMs (2026-07-26). Their weights live in coder.REGISTRY's
    # OWN module-level _instances, not on the dispatch adapter — the same reason
    # _inprocess_gpu_bytes needs a dedicated pass for them. Without this the
    # dispatch evict above returned "dropped" while the nn.Module stayed
    # referenced here, so the VRAM never came back: the operator's 13.6 GiB
    # in-process load survived every eviction attempt and blocked the next load.
    # Match on the config's model_key (now carried) AND the runner's own cache
    # key, since a DeepCoder is keyed by cfg.cache_key() not by model_key.
    try:
        from hugpy_engine.generate.coder import REGISTRY as _DC
        insts = getattr(_DC, "_instances", None)
        if isinstance(insts, dict):
            for ck, dc in list(insts.items()):
                if getattr(getattr(dc, "cfg", None), "model_key", None) != model_key:
                    continue
                try:
                    model = getattr(dc, "model", None)
                    if model is not None:
                        try:
                            model.to("meta")     # release CUDA storage eagerly
                        except Exception:  # noqa: BLE001
                            pass
                    dc.model = None
                    dc.tokenizer = None
                except Exception:  # noqa: BLE001
                    pass
                insts.pop(ck, None)
                dropped = True
    except Exception:  # noqa: BLE001
        pass
    # Free the CUDA caching allocator's blocks too — dropping python refs alone
    # leaves the memory reserved by torch and still visible to nvidia-smi, which
    # is what makes an "evicted" model look like it never left the card.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass
    _trim_host_ram()
    # The materialized flag must die WITH the weights (this is the unload
    # chokepoint every evict path funnels through). Without this, a
    # transformers/DeepCoder entry in _MATERIALIZED — which has no live cache
    # to interrogate, unlike GGUF's _LLAMA_INSTANCES — would keep reporting
    # materialized=True forever after an evict: the exact stale-residency lie
    # the flag exists to prevent. Unconditional on purpose: if nothing was
    # found to drop, forgetting is a no-op or corrects an already-stale entry.
    _forget_materialized(model_key)
    return dropped


def _comfy_free_models(state: "WorkerState") -> "tuple[bool, str]":
    """Ask the ADOPTED external ComfyUI to release its resident models via its OWN
    HTTP API — never a PID kill (the worker doesn't own comfy's process). ComfyUI
    exposes ``POST /free`` with ``{"unload_models": true, "free_memory": true}``;
    it unloads comfy's currently-loaded checkpoint(s) and hands VRAM back while
    the server stays up for the next job. Returns (freed_ok, note). Degrades
    gracefully (freed_ok=False + reason) when comfy is unreachable / lacks /free."""
    url = _comfy_base_url(state)
    try:
        import httpx
        r = httpx.post(url + "/free",
                       json={"unload_models": True, "free_memory": True},
                       timeout=30.0)
        if r.status_code == 200:
            # comfy holds nothing now: the ledger's resident set is empty until
            # the next dispatch (every /free goes through here — the watchdog,
            # the evict verb and the planner all bind this one function).
            _comfy_ledger().note_freed()
            return True, "comfy /free accepted (unload_models + free_memory)"
        return False, f"comfy /free returned HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001 — comfy down / no /free: degrade, never 500
        return False, f"comfy unreachable at {url}: {type(exc).__name__}: {exc}"


def _external_pause(ext: dict) -> "tuple[bool, str]":
    """Ask an EXTERNAL leased resident's supervisor to yield the card via its
    OWN control API — never a PID kill (the worker doesn't own the job; the
    supervisor does, and it performs the SIGTERM->grace->SIGKILL on its child
    process group itself, then replies AFTER the VRAM is actually free). The
    comfy /free idiom, pointed at ``gpu_lease``'s POST /pause. Degrades
    gracefully: a dead registered pid is an already-freed resident (idempotent
    success); an unreachable supervisor with a live pid is an honest refusal —
    we never os.kill a foreign pid from here (same doctrine as the not-resident
    branch of _evict_model)."""
    pid = ext.get("pid")
    url = (ext.get("control_url") or "").rstrip("/")

    def _pid_alive(p) -> bool:
        try:
            os.kill(int(p), 0)
            return True
        except (OSError, TypeError, ValueError):
            return False

    if pid and not _pid_alive(pid):
        return True, f"external pid {pid} already gone — nothing holds the card"
    if not url:
        if not pid:
            return True, "external lease has no live pid — nothing to pause"
        return False, (f"external pid {pid} is alive but the lease registered "
                       "no control_url — cannot pause (worker never kills a "
                       "foreign pid directly)")
    try:
        import httpx
        r = httpx.post(url + "/pause", json={"reason": "evicted by worker"},
                       timeout=45.0)
        if r.status_code == 200:
            return True, ("external supervisor paused its job (child pgroup "
                          "terminated before replying)")
        return False, f"external supervisor /pause returned HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001 — supervisor down: degrade, never 500
        if pid and not _pid_alive(pid):
            return True, (f"external supervisor unreachable but pid {pid} is "
                          "gone — treating as already freed")
        return False, (f"external supervisor unreachable at {url}: "
                       f"{type(exc).__name__}: {exc}")


def _comfy_base_url(state: "WorkerState") -> str:
    """The adopted ComfyUI base URL: the operator/comfy_url setting projects onto
    COMFY_URL (see _apply_settings_env); default 127.0.0.1:8188 matches
    managers/comfy/comfy_runner._comfy_url()."""
    return (os.environ.get("COMFY_URL") or "http://127.0.0.1:8188").rstrip("/")


# ── comfy idle-VRAM watchdog (k54, operator directive 2026-07-31) ────────────
# A dead/idle ComfyUI process must never permanently squat the card (the live
# case: 2874 MiB held for 61 h with an empty queue). The predicate + the reclaim
# live in worker_agent.comfy_watchdog — self-contained and probe-injectable, so
# it never imports this module back; the worker binds the real box-touching
# probes here, once, and the watchdog holds only the idle clock.
_COMFY_WATCHDOG = None


def _comfy_vram_now(fresh: bool = False) -> "int | None":
    """Comfy's measured VRAM, optionally bypassing the ~8s nvidia-smi cache.

    ``fresh`` is what makes the post-/free re-measure honest: without it the
    verification read would return the cached PRE-free figure and every
    successful reclaim would report itself as a failure."""
    if fresh:
        _GPU_PROC_CACHE["at"] = 0.0
    return _comfy_process_vram()


def _comfy_watchdog(state: "WorkerState"):
    """The process-wide watchdog, built on first use (so a box that never runs
    the residency loop never constructs one). ``state`` only reaches it through
    the bound probes."""
    global _COMFY_WATCHDOG
    if _COMFY_WATCHDOG is None:
        from hugpy_fleet.worker.comfy_watchdog import ComfyIdleWatchdog
        _COMFY_WATCHDOG = ComfyIdleWatchdog(
            vram_probe=_comfy_vram_now,
            url_probe=lambda: _comfy_base_url(state),
            free_call=lambda: _comfy_free_models(state),
            emit=_evt_emit,
            # The idle-STOP path: bound only when the launcher is worker-managed
            # (systemd-user/spawn). For the ``external`` default, ComfyManager.stop
            # returns ok=False with a note and stop_tick short-circuits on
            # ``managed=False`` anyway — so an unmanaged box never stops comfy.
            stop_call=lambda: _comfy_manager(state).stop())
    return _COMFY_WATCHDOG


# ── comfy process manager (worker owns comfy's lifecycle, 2026-09-24) ─────────
# The worker starts/stops the local ComfyUI the same way it hosts an LLM server:
# on demand when a comfy job is placed here, evict-to-fit first, serve, then free
# VRAM / stop when idle. The launcher is pluggable (HUGPY_COMFY_LAUNCH): external
# (default — probe only, unchanged), systemd-user:<unit>, or spawn:<python>
# <main.py> [args]. The manager is self-contained + injectable (worker_agent.
# comfy_process); the singleton binds the real box-touching probes once.
_COMFY_MANAGER = None


def _comfy_manager(state: "WorkerState"):
    """The process-wide comfy process manager, built on first use from the live
    ``HUGPY_COMFY_LAUNCH`` spec + the adopted comfy URL."""
    global _COMFY_MANAGER
    if _COMFY_MANAGER is None:
        from hugpy_fleet.worker.comfy_process import ComfyManager, parse_launch, comfy_url
        spec = parse_launch(os.environ.get("HUGPY_COMFY_LAUNCH"))
        _COMFY_MANAGER = ComfyManager(spec, comfy_url())
        if spec.get("error"):
            logger.warning("comfy launcher config ignored: %s — falling back to "
                           "'external' (probe only)", spec["error"])
        elif _COMFY_MANAGER.managed:
            logger.info("comfy process manager: launcher=%s url=%s (worker owns "
                        "comfy's lifecycle)", spec.get("kind"), _COMFY_MANAGER.url)
    return _COMFY_MANAGER


def _ensure_comfy_started(state: "WorkerState", model_key: "str | None") -> None:
    """On-demand comfy start for a placed comfy job (the /infer relay path).

    Ordering per the operator directive: run evict-to-fit for THIS checkpoint's
    need FIRST (_worker_ensure_comfy_headroom, the same evict verb every path
    uses), THEN start the managed comfy and wait for /system_stats readiness with
    a bounded timeout. A start failure RAISES a structured ModelLoadFailure (the
    real launcher/journal text as loader_stderr) so the /infer error body carries
    the same ``load_failure`` envelope an LLM load ships — central's relay path
    records it via record_load_failure.

    A no-op for a non-comfy model, or when the launcher is ``external`` (the
    default — the worker never starts comfy on those boxes, unchanged), or when
    comfy is already running."""
    if not model_key:
        return
    if _model_framework(model_key) != "comfy":
        return
    mgr = _comfy_manager(state)
    if not mgr.managed:
        return                       # external launcher: probe-only, unchanged
    if mgr.is_running():
        return                       # already up — the per-gen headroom hook runs as usual
    # 1) EVICT-TO-FIT FIRST: make room for this checkpoint's need before start.
    try:
        _worker_ensure_comfy_headroom(state, model_key)
    except Exception:  # noqa: BLE001 — headroom is best-effort; never block the start
        logger.warning("ensure-comfy-headroom before start raised for %s",
                       model_key, exc_info=True)
    # 2) START, then wait for readiness (bounded).
    res = mgr.start()
    if res.get("ok"):
        logger.info("comfy started on demand for %s (%s)", model_key,
                    res.get("note"))
        return
    # 3) FAILURE -> a recorded load failure with the real reason.
    from hugpy_engine.serve.load_failure import ModelLoadFailure
    raise ModelLoadFailure(
        res.get("note") or f"comfy failed to start for {model_key}",
        load_class=res.get("load_class") or "engine_unavailable",
        loader_stderr=res.get("loader_stderr"),
        model_key=model_key,
        path=(mgr.spec.get("unit") or mgr.spec.get("main")))


def _comfy_reclaim_idle_vram(state: "WorkerState", incoming_model: "str | None",
                             need_bytes: "int | None" = None) -> int:
    """CONTENTION reclaim: bytes freed from an IDLE comfy for a load that needs
    them, 0 when comfy is busy / holds nothing / can't be proved idle.

    The mirror of ``_worker_ensure_comfy_headroom`` (Fix B), which evicts managed
    models FOR comfy: this frees comfy FOR a managed model. Both are best-effort
    and neither is allowed to raise into the path that called it."""
    try:
        res = _comfy_watchdog(state).reclaim(incoming_model=incoming_model,
                                             need_bytes=need_bytes)
        return int(res.get("freed_bytes") or 0)
    except Exception as exc:  # noqa: BLE001 — a reclaim attempt never breaks admission
        logger.warning("comfy idle reclaim failed: %s", exc)
        return 0


# ── comfy resident ledger (2026-09-22) ──────────────────────────────────────
# ComfyUI is one nvidia-smi lump; hugpy dispatched every checkpoint inside it.
# The ledger (worker.comfy_ledger) records those dispatches and their file
# sizes, which is what lets (a) the headroom path clear room for THIS
# checkpoint instead of one constant for every gen and (b) the eviction planner
# rank comfy's residents by name and LRU like every other occupant.
_COMFY_LEDGER = None


def _comfy_ledger():
    global _COMFY_LEDGER
    if _COMFY_LEDGER is None:
        from hugpy_fleet.worker.comfy_ledger import ComfyLedger
        _COMFY_LEDGER = ComfyLedger()
    return _COMFY_LEDGER


def _comfy_busy_reason(state: "WorkerState") -> "str | None":
    """Why comfy must NOT be freed right now, or ``None`` when it is provably
    idle. The watchdog's clauses 1-3 (a registered comfy call, a non-empty
    comfy queue, an unreadable queue/call table) are the doctrine — an in-flight
    render is never killed for an LLM load — so the planner asks the SAME
    predicate rather than a cheaper one that could disagree with it. Anything
    that cannot be proved idle reads as busy (conservative direction)."""
    try:
        obs = _comfy_watchdog(state).observe()
    except Exception as exc:  # noqa: BLE001 — cannot prove idle -> busy
        return f"comfy idle state unreadable: {type(exc).__name__}"
    if obs.get("idle"):
        return None
    return str(obs.get("why") or "comfy not provably idle")


def _comfy_checkpoint_filename(model_key: str) -> "str | None":
    """The checkpoint file the graph will name for ``model_key`` (the registry
    row's ``filename``), or None for a row that has none."""
    try:
        from hugpy_fleet.worker.imports import get_model_config
        fn = getattr(get_model_config(model_key), "filename", None)
        return str(fn) if fn else None
    except Exception:  # noqa: BLE001 — unknown row: unknown file
        return None


def _comfy_need_detail(state: "WorkerState", model_key: str) -> dict:
    """What a comfy gen of ``model_key`` needs free BEFORE it starts, priced from
    the checkpoint FILE (fp16 weights on disk ≈ weights in VRAM) plus the gen
    cushion — or ``need: None`` when the file cannot be found, so the caller
    falls back honestly instead of inventing a figure.

    ``held``: the ledger says comfy already holds this checkpoint AND comfy's
    measured process VRAM covers its weights — then only the cushion is needed.
    A ledger claim comfy's VRAM does not back (comfy restarted, freed behind our
    back) is not trusted: the row is dropped and the full need applies."""
    from hugpy_fleet.worker import comfy_ledger as _cl
    ledger = _comfy_ledger()
    filename = _comfy_checkpoint_filename(model_key)
    extra = []
    try:
        from hugpy_storage.provision import _comfy_checkpoints_dir
        extra.append(_comfy_checkpoints_dir())
    except Exception:  # noqa: BLE001
        pass
    try:
        extra.append(os.path.join(_models_store_root() or "", "checkpoints"))
    except Exception:  # noqa: BLE001
        pass
    size = _cl.checkpoint_size_bytes(filename, extra)
    held = False
    if ledger.holds(model_key):
        pv = _comfy_process_vram()
        if size is not None and pv is not None and pv >= size:
            held = True
        elif size is None and pv:
            held = True                 # resident by the ledger, size unknown
        else:
            ledger.forget(model_key)    # comfy no longer backs the claim
    need = _cl.predicted_need_bytes(size, held)
    return {"checkpoint": filename, "checkpoint_bytes": size, "held": held,
            "cushion": _cl.gen_cushion_bytes(), "need": need}


def _comfy_headroom_target(detail: dict) -> int:
    """Free VRAM to clear before the gen: the checkpoint's own need when it is
    known, else the legacy constant (HUGPY_COMFY_TARGET_FREE_GIB, 7 GiB) — and
    an EXPLICIT operator constant is a floor the need never undercuts."""
    need = detail.get("need")
    if need is None:
        return _comfy_target_free_bytes()
    floor = 0
    if str(os.environ.get("HUGPY_COMFY_TARGET_FREE_GIB") or "").strip():
        floor = _comfy_target_free_bytes()
    return max(int(need), floor)


def _model_footprint_before_evict(model_key: str, host_mode: str,
                                  handle: "dict | None" = None) -> dict:
    """A per-MODEL, honestly-measured footprint captured BEFORE eviction acts —
    the fix for the ram_freed/vram_freed LIE. The old ``_result()`` inferred
    freed bytes from a whole-box MemAvailable-style delta (_free_vram_bytes /
    _free_ram_bytes before vs after). That is structurally wrong for a GGUF:
    llama.cpp mmaps the weights, so dropping refs returns FILE-BACKED pages to
    the page cache — the box's free-RAM delta barely moves (observed on ae:
    44 GB model, ram_freed reported 35.6 MB, i.e. ~0.08% of the truth) even
    though the eviction was completely real. See rss_anon_bytes doctrine
    (managers/serve/slot_agent._proc_rss_detail): VmRSS/MemAvailable-style
    deltas overstate/understate by ~28x on mmap'd GGUF.

    Instead of a box-wide delta, measure THIS model's own resident bytes,
    broken out honestly by kind, using whichever ground truth its hosting mode
    actually offers:
      * slot       — the child pid's REAL nvidia-smi VRAM (joined on
                     child_pid, same join _allocation_rows uses) + its
                     rss_anon_bytes (the truly-pinned host RAM; rss_file_bytes
                     is mmap'd page cache, reported separately, NEVER folded
                     into "freed").
      * in_process — GGUF: model_bytes from the file the runner opened
                     (loaded_runner_detail) as the file-backed weights size
                     (labeled, not claimed as pinned anon RAM — llama-cpp-
                     python's in-process handle doesn't expose a per-model
                     anon/file split). torch/diffusers: vram_bytes from
                     _inprocess_gpu_bytes(), which is a REAL per-model sum of
                     tensor bytes on a cuda device (not a delta) — the
                     honest figure, not an estimate.
      * comfy      — comfy manages its own resident set across an unknown
                     number of checkpoints; there is no way to attribute
                     freed bytes to ONE model_key from outside. Reported
                     null with a reason, not a guess.
      * not-resident — nothing to measure; zeros, not null (there is
                     genuinely nothing to free).

    Returns {"vram_bytes", "ram_anon_bytes", "ram_file_bytes", "measured_from"}
    — every key may be None when that quantity cannot be honestly attributed
    to this one model_key from this hosting mode; never a fabricated number."""
    out = {"vram_bytes": None, "ram_anon_bytes": None, "ram_file_bytes": None,
           "measured_from": None}
    # "external" measures exactly like "slot": a foreign pid whose real VRAM
    # nvidia-smi attributes directly (the gpu_lease supervisor registers the
    # GPU-holding descendant pid, so the join is honest).
    if host_mode in ("slot", "external") and handle:
        pid = handle.get("child_pid")
        if pid is not None:
            try:
                gpu_procs = _gpu_process_vram() or {}
                info = gpu_procs.get(pid)
                if info is not None:
                    out["vram_bytes"] = int(info.get("mib") or 0) * _MIB
            except Exception:  # noqa: BLE001 — nvidia-smi unavailable -> stays None
                pass
            try:
                from hugpy_engine.serve.slot_agent import _proc_rss_detail
                detail = _proc_rss_detail(pid) or {}
                out["ram_anon_bytes"] = detail.get("rss_anon_bytes")
                out["ram_file_bytes"] = detail.get("rss_file_bytes")
            except Exception:  # noqa: BLE001 — /proc unreadable -> stays None
                pass
            out["measured_from"] = (
                ("slot child" if host_mode == "slot" else "external job")
                + " pid nvidia-smi + /proc rss split")
        return out
    if host_mode == "in_process":
        try:
            detail = (_loaded_detail() or {}).get(model_key) or {}
            mb = detail.get("model_bytes")
            if mb is not None:
                out["ram_file_bytes"] = int(mb)
                out["measured_from"] = "GGUF file size (mmap'd, file-backed — " \
                    "NOT pinned anon RAM; llama-cpp-python exposes no " \
                    "per-model anon/file split)"
        except Exception:  # noqa: BLE001
            pass
        try:
            ip = (_inprocess_gpu_bytes() or {}).get(model_key) or {}
            vb = ip.get("vram_bytes")
            if vb:
                out["vram_bytes"] = int(vb)
                out["measured_from"] = (
                    (out["measured_from"] + " + " if out["measured_from"] else "")
                    + "torch tensor bytes on cuda device (real per-model sum, "
                      "not a delta)")
        except Exception:  # noqa: BLE001
            pass
        return out
    if host_mode == "comfy":
        out["measured_from"] = (
            "comfy manages its own resident set across an unknown number of "
            "checkpoints — no per-model_key attribution is possible from "
            "outside; reporting null rather than a box-wide guess")
        return out
    # not resident: genuinely nothing to free.
    out.update(vram_bytes=0, ram_anon_bytes=0, ram_file_bytes=0,
               measured_from="not resident — nothing to free")
    return out


def _evict_model(state: "WorkerState", model_key: str,
                 force: bool = False) -> dict:
    """Resolve ``model_key`` to its LIVE hosting handle and free it with the
    mechanism that matches how it is hosted. Fail-safe: unknown/not-resident is
    an idempotent no-op, never an error. Returns the /ops/evict contract dict.

    Resolution order (comfy first, because a comfy checkpoint is served by an
    EXTERNAL process and never appears in our slot/in-process caches; slot before
    in-process, because a slot-backed model also leaves a thin HTTP proxy in the
    in-process cache that holds no weights):
      1. comfy  — framework == 'comfy'      -> comfy's own /free API
      2. slot   — a live slot serves it     -> verify identity, then slot /unload
                                               (owner does SIGTERM->wait->SIGKILL)
      3. in-process — weights in our PID     -> drop refs + torch empty_cache + trim
      4. not resident                        -> idempotent no-op

    ram_freed/vram_freed ACCOUNTING (fixed 2026-07-25 — see
    _model_footprint_before_evict): the historical fields are whole-box
    MemAvailable-style before/after deltas, which page-cache behavior makes
    structurally dishonest for an mmap'd GGUF (a 44 GB eviction reported
    ram_freed=35.6 MB on ae — the eviction was real, the number was noise).
    They are KEPT for wire back-compat (older fleet workers/consoles read
    them) but are no longer the headline truth. The new ``freed`` block
    reports what was actually attributable to THIS model, honestly, with
    nulls where nothing can be honestly attributed — never a confidently
    wrong number."""
    if not isinstance(model_key, str) or not model_key.strip():
        return {"model_key": model_key, "host_mode": "unknown", "evicted": False,
                "vram_freed": None, "ram_freed": None,
                "reason": "missing model_key"}
    model_key = model_key.strip()
    # t21: an evicted model's ctx flex commitment is void — a fresh load
    # re-decides from target. Drop it so _ctx_pct doesn't serve a stale floor.
    _FLEX_CTX_FLOOR.pop(model_key, None)

    vram_before = _free_vram_bytes()
    ram_before = _free_ram_bytes()

    def _result(host_mode, evicted, reason, footprint=None, **extra):
        vram_after = _free_vram_bytes()
        ram_after = _free_ram_bytes()
        vram_freed = (vram_after - vram_before) if (
            vram_before is not None and vram_after is not None) else None
        ram_freed = (ram_after - ram_before) if (
            ram_before is not None and ram_after is not None) else None
        out = {"model_key": model_key, "host_mode": host_mode,
               "evicted": bool(evicted), "reason": reason,
               # LEGACY (wire back-compat only — see docstring): whole-box
               # delta, unreliable for mmap'd weights. Do not treat as truth.
               "vram_freed": vram_freed, "ram_freed": ram_freed,
               "vram_free_before": vram_before, "vram_free_after": vram_after,
               "ram_free_before": ram_before, "ram_free_after": ram_after,
               "forced": bool(force), "loaded_models": loaded_model_keys()}
        # HONEST per-model breakdown (optional, omit-when-unset — new field,
        # older released workers/central pydantic models with extra=forbid
        # never see it unless they opt in by reading this key). Only attached
        # when the eviction attempt actually resolved a footprint measurement
        # (i.e. footprint dict was computed) so an unrelated failure path
        # (missing model_key) doesn't grow a new key by accident.
        if footprint is not None:
            out["freed"] = footprint
        out.update(extra)
        return out

    # 1. ComfyUI-hosted (external adopted service) — framework says comfy. The
    #    worker never owns comfy's PID; it asks comfy to free via HTTP. The gate
    #    still applies best-effort (a comfy gen in flight is protected unless
    #    forced), but comfy's /free is coarse (releases comfy's resident set).
    if _model_framework(model_key) == "comfy":
        allowed, why = (True, "") if force else _evict_gate(model_key)
        if allowed and not force:
            # The watchdog's doctrine, not just this key's gate: comfy /free
            # drops EVERY checkpoint it holds, so any render in flight against
            # comfy (ours or not) vetoes it, and unprovable idleness is busy.
            busy = _comfy_busy_reason(state)
            if busy:
                allowed, why = False, busy
        if not allowed:
            return _result("comfy", False, f"eviction gated: {why}")
        footprint = _model_footprint_before_evict(model_key, "comfy")
        freed_ok, note = _comfy_free_models(state)
        return _result("comfy", freed_ok, note, footprint=footprint)

    # 2. Subprocess-hosted (slot child / worker-spawned llama-server). Resolve the
    #    model_key -> its CURRENT slot handle from a LIVE status read.
    handle = _resolve_slot_handle(model_key)
    if handle is not None:
        allowed, why = (True, "") if force else _evict_gate(model_key)
        if not allowed:
            return _result("slot", False, f"eviction gated: {why}",
                           child_pid=handle.get("child_pid"))
        # RECYCLED-PID GUARD: re-read the slot status right before acting and
        # confirm it STILL maps this model_key to the SAME child_pid we resolved.
        # A slot that has since swapped to another model (or respawned its child
        # under a new pid) must NOT be evicted — that would kill the wrong model.
        pid = handle.get("child_pid")
        control = handle.get("control_url")
        recheck = _resolve_slot_handle(model_key)
        if recheck is None or recheck.get("child_pid") != pid \
                or recheck.get("control_url") != control:
            return _result("slot", False,
                           "slot handle changed before evict (recycled/swapped) "
                           "— not evicted", child_pid=pid)
        # Capture the footprint BEFORE the kill — the pid must still be alive
        # and holding VRAM/RAM for nvidia-smi + /proc to measure it honestly.
        footprint = _model_footprint_before_evict(model_key, "slot", handle)
        # Free via the slot's OWN /unload: the slot supervisor owns the child, so
        # it performs the SIGTERM -> short wait -> SIGKILL itself (Slot._kill:
        # terminate, wait 15s, kill) and clears its own model_key claim atomically
        # — cleaner and safer than the agent os.kill-ing another supervisor's
        # child on a possibly-recycled pid. CUDA context drops on child exit; a
        # host-RAM trim follows to hand the freed arena back.
        err = None
        try:
            from hugpy_engine.serve.slots import SlotPool
            SlotPool().unload(control)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        _trim_host_ram()
        if err is not None:
            return _result("slot", False, f"slot unload failed: {err}",
                           footprint=footprint, child_pid=pid)
        return _result("slot", True,
                       f"slot child pid={pid} terminated (SIGTERM->SIGKILL) via "
                       "its supervisor", footprint=footprint, child_pid=pid)

    # 3. In-process torch/GGUF model sharing THIS worker's python PID. Never kill
    #    the PID (that kills the worker + every sibling model) — drop the refs.
    if _is_inprocess_resident(model_key):
        allowed, why = (True, "") if force else _evict_gate(model_key)
        if not allowed:
            return _result("in_process", False, f"eviction gated: {why}")
        # Capture BEFORE dropping refs — model_bytes/vram_bytes both read
        # already-materialized state, so this is safe to call right before.
        footprint = _model_footprint_before_evict(model_key, "in_process")
        dropped = _drop_inprocess_model(model_key)
        return _result("in_process", dropped,
                       "in-process refs dropped + CUDA cache/host arena trimmed"
                       if dropped else "in-process handle already gone",
                       footprint=footprint)

    # 3.5 EXTERNAL leased resident — an adopted foreign batch job (an OCR run,
    #     a render sweep) supervised by ``worker.gpu_lease``. Mirror of the
    #     comfy idiom: the worker never owns the PID; it asks the job's
    #     registered supervisor to yield via its control URL, and the
    #     supervisor terminates its child process group BEFORE replying, so
    #     the _result() free-VRAM delta is real. Eviction here never CANCELS
    #     the job — it pauses it; the supervisor re-claims headroom and
    #     relaunches once the models go idle again (checkpoint-and-resume is
    #     the admission contract for registering as external at all).
    from hugpy_fleet.worker import external_residents as _extres
    ext = _extres.get(model_key)
    if ext is not None:
        # non-evictable is the external twin of static residency (wildcard-
        # process policy, 2026-08-12): no demand path may pause this job; only
        # an operator force does. Same override semantics as static.
        if not ext.get("evictable", True) and not force:
            return _result("external", False,
                           "external lease is non-evictable — pass force to "
                           "override (or /ops/external/set evictable=true)",
                           child_pid=ext.get("pid"))
        allowed, why = (True, "") if force else _evict_gate(model_key)
        if not allowed:
            return _result("external", False, f"eviction gated: {why}",
                           child_pid=ext.get("pid"))
        # Footprint BEFORE the pause — the pid must still hold VRAM to measure.
        footprint = _model_footprint_before_evict(
            model_key, "external", {"child_pid": ext.get("pid")})
        ok, note = _external_pause(ext)
        if ok:
            _extres.mark_yielded(model_key)
            try:
                from hugpy_fleet.worker import pid_registry as _pidreg
                _pidreg.forget(model_key)
            except Exception:  # noqa: BLE001 — sweep_dead reaps it next beat anyway
                pass
        return _result("external", ok, note, footprint=footprint,
                       child_pid=ext.get("pid"))

    # 4. Nothing here holds it. This ALSO covers the foreign/rogue case: a model
    #    that resolves only to a process the agent did not spawn (and isn't comfy)
    #    is OUT OF SCOPE for this slice — we never os.kill an arbitrary PID, so
    #    such a model simply reads as not-resident here. Idempotent no-op, HTTP 200.
    return _result("none", False, "not resident on this worker",
                   footprint=_model_footprint_before_evict(model_key, "none"))


# ── GPU orphan reaper (p27, 2026-07-23) ─────────────────────────────────────
# THE k30 GAP this closes: an orphaned llama-server child — its slot claim
# cleared (slot swapped/respawned) but the child never exited — keeps holding
# VRAM. Since c34199e it is ENUMERABLE (the reconcile second pass tags it
# ``cuda_context`` with model_key None) but UNEVICTABLE: every eviction verb
# keys on model_key, and an orphan has none. This verb kills by PID — the ONLY
# place in the agent that does — so its admission gates are deliberately
# fail-closed and narrow.

# Minimum process age before a pid may be reaped. Closes the mid-spawn race: a
# fresh slot child exists (and holds VRAM) for a beat BEFORE its slot claim
# registers — without this grace a reap racing a spawn would kill a legitimate
# newborn. Read defensively so a malformed env value can never break import.
try:
    _ORPHAN_MIN_AGE_S = max(
        0.0, float(os.environ.get("HUGPY_ORPHAN_MIN_AGE_S", "300") or "300"))
except (TypeError, ValueError):
    _ORPHAN_MIN_AGE_S = 300.0

# How long a SIGTERM'd orphan gets to exit before SIGKILL — same discipline as
# the slot supervisor's own child kill (terminate -> wait -> kill).
_ORPHAN_TERM_WAIT_S = 10.0


def _clock_ticks_per_s() -> float:
    """Kernel clock ticks per second (for /proc starttime -> seconds). 100 on
    every mainstream Linux; read via os.sysconf when available."""
    try:
        return float(os.sysconf("SC_CLK_TCK")) or 100.0
    except (AttributeError, ValueError, OSError):
        return 100.0


def _proc_age_s(pid: int) -> "float | None":
    """Age of process ``pid`` in seconds, or None when it cannot be measured
    (gone / non-Linux / unreadable) — callers treat None as UNVERIFIABLE and
    fail closed. Primary source: /proc/<pid>/stat starttime (ticks since boot)
    against /proc/uptime — the same per-lifetime anchor the recycled-PID guard
    uses, so age and identity come from one number. Fallback: mtime of the
    /proc/<pid> directory (set at process creation)."""
    try:
        from hugpy_fleet.worker.pid_registry import _default_proc_info
        info = _default_proc_info(int(pid))
    except Exception:  # noqa: BLE001 — probe failure = unverifiable
        info = None
    if info is not None and info.get("starttime") is not None:
        try:
            with open("/proc/uptime", "r") as fh:
                uptime_s = float(fh.read().split()[0])
            started_s = float(info["starttime"]) / _clock_ticks_per_s()
            age = uptime_s - started_s
            if age >= 0:
                return age
        except (OSError, ValueError, IndexError):
            pass
    try:
        st = os.stat("/proc/%d" % int(pid))
        age = time.time() - st.st_mtime
        return age if age >= 0 else None
    except (OSError, ValueError):
        return None


def _reap_own_pids() -> set:
    """The agent's own pid + its direct infra children (slot SUPERVISORS it
    spawned) — never reap targets, by construction."""
    own = {os.getpid()}
    try:
        for p in list(_SLOT_PROCS.values()):
            if p is not None and p.poll() is None and p.pid:
                own.add(int(p.pid))
    except Exception:  # noqa: BLE001 — best-effort; os.getpid() always guards
        pass
    return own


def _self_venv_marker() -> "str | None":
    """The SAME own-venv marker the heartbeat passes pid_registry.reconcile
    (this python's venv root, e.g. ``/opt/hugpy/venv``): a GPU process whose
    name/cmdline contains it runs OUR interpreter/binaries. None when it cannot
    be derived — callers fail closed (nothing is reapable without it)."""
    try:
        marker = os.path.dirname(os.path.dirname(sys.executable)) or None
    except Exception:  # noqa: BLE001
        return None
    # A degenerate marker ("", "/", ".") would match EVERY process name —
    # substring matching makes that a kill-anything wildcard. Refuse it.
    if not marker or marker in ("/", "."):
        return None
    return marker


def _reap_gpu_orphans(state: "WorkerState", dry_run: bool = True) -> dict:
    """Enumerate and (unless ``dry_run``) kill ORPHANED GPU children this worker
    itself leaked: processes from OUR OWN venv that hold VRAM but that no live
    slot claims. Exposed as POST /ops/reap-orphans.

    DOCTRINE — OPERATOR/CENTRAL-INVOKED ONLY. This verb must NEVER be called
    from any loop, heartbeat, sweep, or timer. It is the one place the agent
    kills by raw PID; automation of it is explicitly unsanctioned (operator
    ruling, p27 2026-07-23). Wire it to a button/relay, never a schedule.

    A pid is reapable ONLY when ALL FOUR gates hold — any gate UNVERIFIABLE
    means NOT reapable (fail-closed):

      1. OWN-VENV   — its nvidia-smi process_name or /proc cmdline contains this
                      worker's own venv marker (same marker source the pid
                      registry's cuda_context second pass uses). Comfy
                      (_COMFY_NAME_MARKER in the name) and anything foreign fail
                      by construction; the agent's own pid and its direct infra
                      pids (slot supervisors) are excluded outright.
      2. NO CLAIM   — no current slot status references the pid as child_pid.
                      Slot statuses UNREADABLE -> nothing is reapable this pass.
      3. HOLDS GPU  — the pid appears in the current nvidia-smi snapshot with
                      mib > 0 (a CPU-only stray is not this verb's business).
      4. MIN AGE    — the process is older than HUGPY_ORPHAN_MIN_AGE_S (default
                      300s), closing the mid-spawn race where a slot child
                      exists before its claim registers.

    Kill discipline: SIGTERM -> wait up to 10s -> SIGKILL, with a recycled-PID
    identity re-check (starttime) immediately before each signal. Per-pid result
    rows: {pid, name, vram_bytes, action: reaped|term_failed|skipped, reason}.
    ``dry_run`` (the DEFAULT) only reports what would be reaped."""
    gpu_procs = _gpu_process_vram() or {}
    marker = _self_venv_marker()
    own = _reap_own_pids()
    slots = _slot_statuses()
    claimed: "set | None" = None
    if slots is not None:
        claimed = {s.get("child_pid") for s in slots
                   if isinstance(s, dict) and s.get("child_pid") is not None}
    # EXTERNAL-CLAIM gate (2026-09-24): a pid registered as an external resident
    # (a gpu_lease batch child, a studio/video render child) is a LIVE CLAIM the
    # same way a slot child_pid is — the worker adopted it deliberately and yields
    # it only through the resident's own supervisor (_evict_model's external
    # branch / studio_reserve release), never by a raw PID kill from this reaper.
    # Without this, a studio render child (own venv, holds VRAM, no slot claim,
    # older than min-age) matched all four gates and was reapable mid-render — the
    # exact 8GB-studio-fork class the vram-holder meter surfaces. Degrades to the
    # prior behaviour (no extra protection) when the registry is unreadable.
    ext_claimed: set = set()
    try:
        from hugpy_fleet.worker import external_residents as _extres
        ext_claimed = {int(r["pid"]) for r in _extres.snapshot()
                       if r.get("pid") is not None}
    except Exception:  # noqa: BLE001 — no registry -> no extra protection (as before)
        ext_claimed = set()
    try:
        from hugpy_fleet.worker.pid_registry import _COMFY_NAME_MARKER as _comfy_marker
        from hugpy_fleet.worker.pid_registry import _default_proc_info as _proc_info
    except Exception:  # noqa: BLE001 — no registry module -> nothing verifiable
        _comfy_marker, _proc_info = "comfyui", (lambda _pid: None)

    results: list = []
    reapable_bytes = 0
    for pid, meta in sorted(gpu_procs.items()):
        name = str((meta or {}).get("name") or "")
        mib = int((meta or {}).get("mib") or 0)
        vram_bytes = mib * _MIB

        def _skip(reason: str) -> None:
            results.append({"pid": pid, "name": name, "vram_bytes": vram_bytes,
                            "action": "skipped", "reason": reason})

        # Gate 3 first (cheap): must actually hold GPU memory.
        if mib <= 0:
            _skip("holds no VRAM (mib<=0)")
            continue
        # Own-pid / infra exclusion — before anything else.
        if pid in own:
            _skip("agent's own pid / direct infra pid — never reapable")
            continue
        # Comfy fails own-venv BY CONSTRUCTION (external adopted service).
        if _comfy_marker in name.lower():
            _skip("comfy process — never reapable (external adopted service)")
            continue
        # Gate 1: OWN-VENV. Marker underivable -> nothing reapable (fail-closed).
        if marker is None:
            _skip("own-venv marker unavailable — cannot prove ownership "
                  "(fail-closed)")
            continue
        info = _proc_info(pid)
        cmdline = str((info or {}).get("cmdline") or "")
        if marker not in name and marker not in cmdline:
            _skip("not from this worker's venv — foreign process, out of scope")
            continue
        # Gate 2: NO LIVE CLAIM. Unreadable slot pool -> unverifiable -> skip.
        if claimed is None:
            _skip("slot statuses unreadable — cannot prove no live claim "
                  "(fail-closed)")
            continue
        if pid in claimed:
            _skip("live slot claims this pid as child_pid — not an orphan")
            continue
        # Gate 2b: EXTERNAL CLAIM. A registered external resident (gpu_lease
        # child / studio render child) is adopted, not an orphan — yield it via
        # its supervisor, never a raw kill here.
        if pid in ext_claimed:
            _skip("registered external resident (lease/studio render) claims "
                  "this pid — yielded via its supervisor, not reaped")
            continue
        # Gate 4: MIN AGE. Unmeasurable age -> unverifiable -> skip.
        age = _proc_age_s(pid)
        if age is None:
            _skip("process age unmeasurable — cannot rule out mid-spawn race "
                  "(fail-closed)")
            continue
        if age < _ORPHAN_MIN_AGE_S:
            _skip(f"process too young ({age:.0f}s < min age "
                  f"{_ORPHAN_MIN_AGE_S:.0f}s) — mid-spawn race protection")
            continue

        # ALL FOUR GATES HOLD — this pid is a reapable orphan.
        reapable_bytes += vram_bytes
        if dry_run:
            results.append({"pid": pid, "name": name, "vram_bytes": vram_bytes,
                            "action": "skipped",
                            "reason": "dry_run — would be reaped "
                                      "(all four gates hold)"})
            logger.info(
                "reap-orphans DRY RUN: pid %s (%s, %s) is a reapable orphan "
                "(own-venv, unclaimed, holds GPU, age %.0fs)",
                pid, name, _human_bytes(vram_bytes), age)
            continue

        # Recycled-PID identity anchor: capture starttime NOW; re-verify before
        # each signal so a pid recycled mid-reap is never signalled.
        anchor = (info or {}).get("starttime")

        def _still_same() -> bool:
            cur = _proc_info(pid)
            if cur is None:
                return False                        # gone — nothing to signal
            if anchor is not None and cur.get("starttime") is not None:
                return int(cur["starttime"]) == int(anchor)
            # No starttime anchor available: corroborate via cmdline instead;
            # unverifiable identity -> do NOT signal.
            return bool(cmdline) and cur.get("cmdline") == cmdline

        logger.warning(
            "reap-orphans: KILLING orphaned GPU child pid=%s name=%r vram=%s "
            "age=%.0fs (own-venv match %r, no live slot claim) — SIGTERM",
            pid, name, _human_bytes(vram_bytes), age, marker)
        import signal as _signal
        try:
            if not _still_same():
                results.append({"pid": pid, "name": name,
                                "vram_bytes": vram_bytes, "action": "skipped",
                                "reason": "pid identity changed before SIGTERM "
                                          "(recycled/exited) — not signalled"})
                continue
            os.kill(pid, _signal.SIGTERM)
        except ProcessLookupError:
            results.append({"pid": pid, "name": name, "vram_bytes": vram_bytes,
                            "action": "reaped",
                            "reason": "already exited at SIGTERM"})
            continue
        except OSError as exc:
            results.append({"pid": pid, "name": name, "vram_bytes": vram_bytes,
                            "action": "term_failed",
                            "reason": f"SIGTERM failed: {type(exc).__name__}: {exc}"})
            continue
        # Wait up to _ORPHAN_TERM_WAIT_S for a clean exit.
        deadline = time.time() + _ORPHAN_TERM_WAIT_S
        exited = False
        while time.time() < deadline:
            if _proc_info(pid) is None or not _still_same():
                exited = True
                break
            time.sleep(0.25)
        if exited:
            logger.warning("reap-orphans: pid %s exited on SIGTERM", pid)
            results.append({"pid": pid, "name": name, "vram_bytes": vram_bytes,
                            "action": "reaped", "reason": "SIGTERM honored"})
            continue
        logger.warning("reap-orphans: pid %s survived SIGTERM %.0fs — SIGKILL",
                       pid, _ORPHAN_TERM_WAIT_S)
        try:
            if _still_same():
                os.kill(pid, _signal.SIGKILL)
                results.append({"pid": pid, "name": name,
                                "vram_bytes": vram_bytes, "action": "reaped",
                                "reason": "SIGKILL after SIGTERM timeout"})
            else:
                results.append({"pid": pid, "name": name,
                                "vram_bytes": vram_bytes, "action": "reaped",
                                "reason": "exited between SIGTERM wait and SIGKILL"})
        except ProcessLookupError:
            results.append({"pid": pid, "name": name, "vram_bytes": vram_bytes,
                            "action": "reaped",
                            "reason": "exited before SIGKILL"})
        except OSError as exc:
            results.append({"pid": pid, "name": name, "vram_bytes": vram_bytes,
                            "action": "term_failed",
                            "reason": f"SIGKILL failed: {type(exc).__name__}: {exc}"})

    reaped = [r for r in results if r["action"] == "reaped"]
    failed = [r for r in results if r["action"] == "term_failed"]
    if not dry_run and (reaped or failed):
        _trim_host_ram()                # hand back the freed arena, same as evict
    return {
        "dry_run": bool(dry_run),
        "results": results,
        "reaped_count": len(reaped),
        "term_failed_count": len(failed),
        "skipped_count": len(results) - len(reaped) - len(failed),
        "reapable_vram_bytes": reapable_bytes,
        "min_age_s": _ORPHAN_MIN_AGE_S,
    }


# ── VRAM-HOLDER METER (incident 2026-08-05: an 8GB studio render fork held the
#    card at 0% GPU util for minutes, blind to central, and wedged new loads) ──
# The operator's principle: "if it's not doing work it's evictable; if it's NOT
# evictable and NOT doing work, that must be EXPLICIT." This is the VISIBILITY
# half. It NEVER kills — it enumerates every VRAM holder and labels the idle
# own-venv squatter so central can surface it. The PID-reaper stays gated behind
# the operator (p27, 2026-07-23); nothing here calls it with dry_run=False, and
# nothing here calls _reap_gpu_orphans at all (that verb must never run from a
# loop/heartbeat — see its doctrine). Instead the SQUATTER classification below
# MIRRORS the reaper's four gates using the SAME shared primitives
# (_reap_own_pids / _self_venv_marker / _slot_statuses / _ORPHAN_MIN_AGE_S /
# _proc_age_s / _COMFY_NAME_MARKER), so "reapable" here and there agree by
# construction — labelling only, no signal ever sent.

_GPU_CARD_TTL_S = 8.0
_GPU_CARD_CACHE: dict = {"at": 0.0, "value": None}


def _gpu_card_stats() -> list:
    """Per-card ``[{index, mem_total_bytes, mem_used_bytes, mem_free_bytes,
    util_pct}]`` from ``nvidia-smi --query-gpu``. ``util_pct`` is the WHOLE-CARD
    utilization (0..100) — the only "is this card doing ANY work" signal
    nvidia-smi offers; per-PROCESS GPU util is NOT available from
    --query-compute-apps, so a holder's work-state can only borrow this coarse
    per-card number. ``[]`` when nvidia-smi is absent/errors (no-GPU box behaves
    exactly as before). Cached ~heartbeat cadence, like _gpu_process_vram."""
    now = time.time()
    cached = _GPU_CARD_CACHE["value"]
    if cached is not None and now - _GPU_CARD_CACHE["at"] < _GPU_CARD_TTL_S:
        return cached
    cards: list = []

    def _int(x):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return None

    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.used,memory.free,"
             "utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5:
                    continue
                total, used, free = _int(parts[1]), _int(parts[2]), _int(parts[3])
                cards.append({
                    "index": _int(parts[0]),
                    "mem_total_bytes": total * _MIB if total is not None else None,
                    "mem_used_bytes": used * _MIB if used is not None else None,
                    "mem_free_bytes": free * _MIB if free is not None else None,
                    # "[N/A]" util on a card with no util counter → None, honest.
                    "util_pct": _int(parts[4]),
                })
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        cards = []
    _GPU_CARD_CACHE.update(at=now, value=cards)
    return cards


def _vram_holders(state: "WorkerState") -> dict:
    """READ-ONLY per-card VRAM breakdown + a ROW PER HOLDER, for the operator's
    VRAM-holder meter. Exposed as GET /ops/vram-holders and folded onto the
    heartbeat. Never kills, never signals, never calls the reaper verb.

    Per-holder row::

        {pid, name, vram_bytes, kind, model_key?, serving?, reapable, squatter,
         reason, age_s, work_state}

    ``kind`` ∈ {slot, comfy, own-orphan, foreign, infra}. For SLOT rows the
    status comes from _slot_statuses (model_key + measured serving/busy). For
    every NON-slot row the four-gate classification MIRRORS _reap_gpu_orphans
    (own-venv marker, no live slot claim, holds VRAM, past _ORPHAN_MIN_AGE_S)
    using its shared primitives — so a NON-SERVING own-venv holder past min-age
    reads ``reapable=True`` here exactly as the reaper's dry-run would report it,
    and is flagged the SQUATTER (``squatter=True``). Comfy, foreign, infra and
    slot rows are never reapable and never squatters.

    ``work_state`` is the coarse work signal: slots use their measured
    serving/idle; the idle own-venv squatter is labelled ``idle-squatter``;
    other non-slots carry ``unknown-per-process`` since per-PID GPU util does not
    exist — only the whole-card ``gpu_util_pct`` (max across cards) says whether
    the card is doing ANY work at all. Degrades to empty holders on a no-GPU /
    no-nvidia-smi box."""
    gpu_procs = _gpu_process_vram() or {}
    slots = _slot_statuses()
    cards = _gpu_card_stats()
    now = time.time()

    # Slot child_pid -> its status row (the exact per-model VRAM claim).
    slot_by_pid: dict = {}
    for s in (slots or []):
        cp = (s or {}).get("child_pid")
        if cp is not None:
            try:
                slot_by_pid[int(cp)] = s
            except (TypeError, ValueError):
                continue

    # The SAME primitives the reaper's gates read (never the kill verb itself).
    own = _reap_own_pids()
    marker = _self_venv_marker()
    # Registered external residents by pid (gpu_lease children, studio/video
    # render children) — the SAME external-claim gate the reaper honours. A pid
    # here is an ADOPTED holder, not an own-venv orphan/squatter: it holds the
    # card legitimately and yields only through its supervisor.
    ext_by_pid: dict = {}
    try:
        from hugpy_fleet.worker import external_residents as _extres
        for _rec in _extres.snapshot():
            if _rec.get("pid") is not None:
                try:
                    ext_by_pid[int(_rec["pid"])] = _rec
                except (TypeError, ValueError):
                    continue
    except Exception:  # noqa: BLE001 — no registry -> no external rows (as before)
        ext_by_pid = {}
    try:
        from hugpy_fleet.worker.pid_registry import _COMFY_NAME_MARKER as _comfy_marker
        from hugpy_fleet.worker.pid_registry import _default_proc_info as _proc_info
    except Exception:  # noqa: BLE001 — no registry -> comfy-by-name only, own-venv fails closed
        _comfy_marker, _proc_info = "comfyui", (lambda _pid: None)

    holders: list = []
    for pid, meta in sorted(gpu_procs.items()):
        name = str((meta or {}).get("name") or "")
        mib = int((meta or {}).get("mib") or 0)
        vram_bytes = mib * _MIB
        model_key = None
        serving = None
        age_s = None
        reapable = False

        if pid in slot_by_pid:
            s = slot_by_pid[pid]
            kind = "slot"
            model_key = s.get("model_key")
            serving = _slot_serving(s, now)
            reason = ("serving slot child (live model_key claim)" if serving
                      else "idle slot child — seated, holds VRAM, not serving")
            work_state = "serving" if serving else "idle"
        elif _comfy_marker in name.lower():
            kind = "comfy"
            reason = "external ComfyUI process — never reapable (adopted service)"
            work_state = "external-comfy"
        elif pid in ext_by_pid:
            # Registered external resident: a gpu_lease batch child or a studio/
            # video render child. Adopted, measured, evictable ONLY via its
            # supervisor — never reapable and never a squatter (that is the whole
            # point of registering it: an EXPLICIT non-evictable hold, not a
            # silent leak). Named by its model_key so the meter shows the render.
            _rec = ext_by_pid[pid]
            kind = "external"
            model_key = _rec.get("model_key")
            _ev = _rec.get("evictable", True)
            reason = ("registered external resident %r (%s) — adopted holder, "
                      "yielded via its supervisor, never reaped" % (
                          model_key, "evictable" if _ev else "non-evictable"))
            work_state = "external-render" if str(model_key or "").startswith(
                "studio:") else "external-lease"
        elif pid in own:
            kind = "infra"
            reason = "agent's own pid / slot supervisor — never reapable"
            work_state = "agent-infra"
        else:
            # Non-slot, non-comfy, non-own: own-venv orphan or foreign? MIRROR
            # the reaper's OWN-VENV gate (Gate 1) exactly.
            cmdline = str((_proc_info(pid) or {}).get("cmdline") or "")
            is_own_venv = bool(marker) and (marker in name or marker in cmdline)
            if is_own_venv:
                kind = "own-orphan"
                age_s = _proc_age_s(pid)
                # Gates 2 (no live claim: not in slot_by_pid, established above),
                # 3 (holds VRAM: mib>0) and 4 (min age) — fail-closed on an
                # unmeasurable age, exactly like the reaper.
                if mib <= 0:
                    reason = "own-venv process holds no VRAM (mib<=0)"
                elif age_s is None:
                    reason = ("own-venv orphan, age unmeasurable — reaper would "
                              "fail closed (not counted reapable)")
                elif age_s < _ORPHAN_MIN_AGE_S:
                    reason = (f"own-venv orphan too young ({age_s:.0f}s < "
                              f"{_ORPHAN_MIN_AGE_S:.0f}s) — mid-spawn grace")
                else:
                    reapable = True
                    reason = ("holds VRAM, no live slot claim, own venv, past "
                              "min-age — reapable but NOT auto-reaped")
                work_state = "idle-squatter" if reapable else "own-venv-holder"
            else:
                kind = "foreign"
                reason = ("foreign process — not this worker's venv, out of "
                          "scope (never reapable)")
                work_state = "foreign"

        # The SQUATTER: an own-orphan that passed every gate and is not a serving
        # slot (own-orphans are never slots, so serving is None here). This is the
        # exact row the operator wants surfaced EXPLICITLY.
        squatter = bool(reapable and kind == "own-orphan" and serving is not True)

        row = {
            "pid": pid,
            "name": name,
            "vram_bytes": vram_bytes,
            "kind": kind,
            "reapable": reapable,
            "squatter": squatter,
            "reason": reason,
            "work_state": work_state,
        }
        if kind == "slot":
            row["model_key"] = model_key
            row["serving"] = serving
        if kind == "external":
            # Name the adopted holder (studio:<job_id> / lease key) so the meter
            # shows the render/lease by name instead of an anonymous pid.
            row["model_key"] = model_key
        if age_s is not None:
            row["age_s"] = round(age_s, 1)
        holders.append(row)

    # Whole-box coarse work signal: the MAX card utilization. This is the ONLY
    # "is the GPU doing anything" reading available — it is per-CARD, never
    # per-process, so it cannot say WHICH holder is busy (an idle squatter on a
    # card another process is driving still reads high). Callers must treat it as
    # a coarse gate, not per-holder attribution.
    utils = [c.get("util_pct") for c in cards if c.get("util_pct") is not None]
    gpu_util_pct = max(utils) if utils else None

    def _sum(key):
        vals = [c.get(key) for c in cards if c.get(key) is not None]
        return sum(vals) if vals else None

    squatters = [h for h in holders if h.get("squatter")]
    return {
        "ok": True,
        "cards": cards,
        "vram_total_bytes": _sum("mem_total_bytes"),
        "vram_used_bytes": _sum("mem_used_bytes"),
        "vram_free_bytes": _sum("mem_free_bytes"),
        "gpu_util_pct": gpu_util_pct,
        "gpu_util_source": ("nvidia-smi utilization.gpu — WHOLE-CARD, not "
                            "per-process; coarse 'card doing any work' signal"),
        "holders": holders,
        "squatters": squatters,
        "min_age_s": _ORPHAN_MIN_AGE_S,
    }


def _vram_holders_beat(state: "WorkerState") -> "dict | None":
    """Heartbeat-safe wrapper around :func:`_vram_holders`. The beat must NEVER
    fail over telemetry (a missed heartbeat drops the worker off the fleet), so
    any error degrades to ``None`` and central simply omits the meter this beat —
    the exact contract every other optional beat field keeps."""
    try:
        return _vram_holders(state)
    except Exception as exc:  # noqa: BLE001 — telemetry never breaks a beat
        logger.debug("vram-holders beat capture failed: %s", exc)
        return None


# ── VRAM evict-to-fit at admission (slice 10, the VRAM twin of disk evict-to-fit) ─
# The operator's ruling (2026-07-17): "everything is on demand — the process not
# actively replying and not ahead of the subject in the queue, as well as not
# 'static', should be evicted to allow the subject process to proliferate."
#
# THE INCIDENT: a transformers load OOM'd because an IDLE 21.3G coder SLOT CHILD
# squatted the card and NOTHING evicted it first. The in-process contention path
# (dispatch.ensure_headroom_for_load) only ever saw _INSTANCES residents and its
# _worker_evictable predicate REFUSED slot-backed models — so a subprocess
# squatter was invisible to an in-process load's make-room. This choke point sees
# ALL residents (in-process + slot child + comfy) from the pid-registry MEASURED
# truth, applies the protection rules, evicts the minimum LRU set through the SAME
# _evict_model verb the operator proved live via /ops/evict, re-checks, and
# refuses HONESTLY before any CUDA allocation (never admit-then-OOM).

# VRAM eviction counter — the churn the operator watches must now include VRAM
# evictions, not just disk. Surfaced on the heartbeat (see _worker_storage /
# the beat body). A simple monotonic count + the last event, cheap and honest.
_VRAM_EVICTIONS: dict = {"count": 0, "last": None, "last_at": 0.0}


# ── eviction telemetry (operator directive 2026-07-28) ──────────────────────
# The heartbeat's _VRAM_EVICTIONS above is a COUNTER — "how many, and the last
# one" — which is the right shape for a status pill and the wrong shape for
# watching an eviction happen. This is the companion STREAM: every stage of
# every pass, relayed to central so the console can render it live. The counter
# stays exactly as it is; nothing below replaces or reads it.
#
# Strictly observational. Every call is swallowed on failure — a worker whose
# relay is broken must evict exactly as it does today.
try:
    from hugpy_fleet.central import evictions as _evt
except Exception:  # noqa: BLE001
    _evt = None


def _evt_emit(stage: str, **fields) -> None:
    """Best-effort eviction-telemetry emit. Inherits the ambient run_id that
    dispatch opened for this pass (thread-local), so make-room events land in
    the same console card as the yield loop that called us."""
    if _evt is None:
        return
    try:
        _evt.emit_eviction_event(stage, **fields)
    except Exception:  # noqa: BLE001 — telemetry never disturbs an eviction
        pass


def _telemetry_tier(host_mode: "str | None") -> str:
    """Map the worker's ``host_mode`` onto the stream's residency tier — HOW the
    weights were held. Unknown modes pass through verbatim rather than being
    forced into a bucket: an honest unfamiliar label beats a wrong familiar
    one."""
    hm = str(host_mode or "").strip().lower()
    if hm in ("slot", "slot-child", "slot_child"):
        return "slot-child"
    if hm == "comfy":
        return "comfy"
    if hm in ("", "inprocess", "in-process", "in_process"):
        return "in-process"
    return hm


def _note_vram_eviction(victim: str, subject: str, freed: "int | None",
                        host_mode: str) -> None:
    _VRAM_EVICTIONS["count"] += 1
    _VRAM_EVICTIONS["last"] = {
        "victim": victim, "subject": subject, "host_mode": host_mode,
        "vram_freed": freed, "at": time.time()}
    _VRAM_EVICTIONS["last_at"] = time.time()
    logger.info("VRAM evict-to-fit: evicted %s (%s, freed %s) to make room for %s",
                victim, host_mode, _human_bytes(freed), subject)


from hugpy_platform.formatting import human_bytes as _human_bytes


def _need_split_str(det: dict) -> str:
    """The honest weights+kv breakdown for a refusal (slice 11), e.g.
    ' = 21.3 GB weights + 2.8 GB kv@50%ctx'. Empty when there is no ctx (kv=0),
    so a model with no ctx allocation reads exactly as today."""
    kv = int(det.get("kv") or 0)
    if kv <= 0:
        return ""
    pct = det.get("ctx_pct")
    tag = f"@{pct}%ctx" if pct else ""
    return (f" = {_human_bytes(det.get('weights'))} weights + "
            f"{_human_bytes(kv)} kv{tag}")


def _actively_replying(model_key: str, slot_busy: "set | None" = None) -> bool:
    """MEASURED 'actively replying' (operator: protect an in-flight reply), NOT
    inferred from residency. True when the model has an in-flight in-process
    generation (gen_gate) OR its slot is flagged busy this instant. Fail-safe: if
    we can't tell, treat as busy (never rip a possibly-replying model)."""
    try:
        if gen_gate.in_flight(model_key) > 0:
            return True
    except Exception:  # noqa: BLE001 — can't tell -> protect
        return True
    if slot_busy is not None:
        return model_key in slot_busy
    return False


# A slot's ``busy`` flag is ``inflight > 0``, and that counter can LEAK: a
# client disconnect or a GC'd streaming generator whose ``finally`` never runs
# leaves inflight stuck > 0. A leaked flag then pins the resident as "actively
# replying" FOREVER, and the admission/sweep filters push it into ``protected``
# — the k67 item B incident, where the headroom sweep warned "nothing evictable
# — every resident is static or actively replying" while MN-GRAND-23.5B +
# Qwen3.5-9B sat ~20 GiB idle, so admission thrashed (4 loads / two 500s)
# instead of evicting to fit. Only 🔒static + a GENUINELY in-flight reply may
# block eviction (eviction-protection-two-classes-only); a leaked counter is
# neither. Corroborate the flag with recency below.
_SLOT_BUSY_STALE_S = float(
    os.environ.get("HUGPY_SLOT_BUSY_STALE_S", "900") or 900)


def _busy_slot_models() -> set:
    """Model_keys whose slot is BUSY right now (a live request in the child) —
    the slot-side 'actively replying' signal. Empty on any read failure.

    A slot claiming busy but whose ``last_used`` aged past ``_SLOT_BUSY_STALE_S``
    has a LEAKED inflight counter, not a live reply, so it is NOT reported busy
    (k67 item B — a leaked flag must never protect an idle resident from
    evict-to-fit). ``last_used`` is stamped at request start; a genuine
    generation completes well within the (generous, env-overridable) window,
    while a leaked counter ages without bound. A slot with no ``last_used`` yet
    reported (0/None) is trusted as-is — inflight cannot be > 0 without a start."""
    try:
        from hugpy_engine.serve.slots import SlotPool
        now = time.time()
        busy: set = set()
        for s in SlotPool().statuses():
            mk = s.get("model_key")
            if not mk or not s.get("busy"):
                continue
            last = s.get("last_used") or 0
            try:
                aged = last and (now - float(last)) > _SLOT_BUSY_STALE_S
            except (TypeError, ValueError):
                aged = False
            if aged:
                logger.warning(
                    "slot busy flag for %s is STALE (%.0fs since last_used > %.0fs "
                    "ceiling) — treating inflight as LEAKED, resident is evictable "
                    "(k67 item B)", mk, now - float(last), _SLOT_BUSY_STALE_S)
                continue
            busy.add(mk)
        return busy
    except Exception:  # noqa: BLE001
        return set()


def _queued_ahead_of(subject: str) -> set:
    """Model_keys with pending work queued AHEAD of the subject's request —
    protected for this pass (operator: 'not ahead of the subject in the queue').

    On-box the worker has no central queue; the honest local signal is the
    provision/warm queue plus any model with WAITING gen-gate entrants (a request
    parked on the gate is queued work targeting that resident). Best-effort — an
    empty set on any failure just means nothing is queue-protected this pass, and
    the in-flight guard still protects an actively-replying model."""
    ahead: set = set()
    try:
        # A model with more gate entrants than are in-flight has requests WAITING
        # on it — queued work the very next release will serve. Protect it.
        for mk, g in list(getattr(gen_gate, "_gates", {}).items()):
            try:
                if g.active() >= g.limit and mk != subject:
                    ahead.add(mk)          # gate saturated -> a waiter is queued
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return ahead


def _vram_residents(state: "WorkerState") -> "list[dict]":
    """Every GPU-resident model this box holds, from the pid-registry MEASURED
    snapshot (in_process + slot subprocess + comfy) UNIONED with the LIVE slot
    statuses, each with its real vram_bytes and host_mode. This is the resident
    TRUTH the eviction planner ranks — it includes the slot child ('max GPU'
    alloc and all: alloc is a sizing preference, not a residency shield) that
    the in-process contention path was blind to. Comfy rows are surfaced but
    named from the comfy LEDGER (2026-09-22): comfy is one nvidia-smi lump, but
    hugpy dispatched every checkpoint inside it, so each one is a row here with
    its file size (capped at the measured process VRAM) and host_mode "comfy".
    Whether a comfy row may be evicted is _partition_residents' call (idle
    comfy yields under contention; a rendering comfy never does).

    THE k30 INVISIBILITY FIX (2026-07-23): the pid registry is per-process,
    in-memory state repopulated by the heartbeat loop. A slot occupant can be
    plainly visible in the allocations view (live slot status + nvidia-smi)
    while the registry has no record for it yet — a fresh re-exec before the
    first beat, a swept/mis-verified record, or a child whose pid the reconcile
    second pass tagged as an anonymous ``cuda_context`` lump (model_key=None,
    which this function used to skip). The evict planner then enumerated ZERO
    residents on an occupied card and refused with the self-contradictory
    "evicted 0 idle ... 0 protected still hold the card" — a de-facto
    protection class (invisibility) the operator never sanctioned. Fix: the
    planner enumerates the SAME collection the allocations view shows — the
    registry rows PLUS every live slot occupant (model_key set), joining the
    slot child's real VRAM from nvidia-smi when the registry attribution is
    missing. Only static / actively-replying / queued-ahead / comfy protect
    (operator ruling 2026-07-23)."""
    out: list[dict] = []
    seen: set = set()
    # WHICH card each resident holds (2026-09-25): so eviction can free the TARGET
    # device, not a sibling card. in-process torch models -> the measured device
    # ordinal (_inprocess_gpu_bytes); comfy -> its provisioned pin; slot -> its own
    # CUDA_VISIBLE_DEVICES pin (joined below). gpu_index omitted when unknown.
    try:
        _ip_dev = {mk: v.get("gpu_index")
                   for mk, v in (_inprocess_gpu_bytes() or {}).items()}
    except Exception:  # noqa: BLE001 — no torch -> no per-device attribution
        _ip_dev = {}
    _comfy_dev = None
    _cd_env = (os.environ.get("HUGPY_COMFY_CUDA_DEVICE") or "").strip()
    if _cd_env:
        try:
            _comfy_dev = int(_cd_env)
        except ValueError:
            _comfy_dev = None
    try:
        from hugpy_fleet.worker import pid_registry as _pidreg
        snap = _pidreg.snapshot_for_heartbeat() or {}
        for row in snap.get("models") or []:
            mk = row.get("model_key")
            if not mk:
                continue                    # cuda_context lump / idle comfy: no model
            seen.add(mk)
            _row = {
                "model_key": mk,
                "vram_bytes": int(row.get("vram_bytes") or 0),
                "host_mode": row.get("host_mode") or "",
                "alive": bool(row.get("alive", True)),
            }
            if _ip_dev.get(mk) is not None:
                _row["gpu_index"] = _ip_dev.get(mk)
            out.append(_row)
    except Exception:  # noqa: BLE001 — no registry -> nothing to plan against
        pass
    # Union in comfy's residents by name (the ledger). The registry attributes
    # comfy's PID to at most the ACTIVE call's model; everything else comfy
    # still caches is invisible to it. Ledger rows only when comfy really holds
    # VRAM — a ledger with no process behind it (comfy restarted) names nothing.
    try:
        pv = _comfy_process_vram()
        if pv:
            for row in _comfy_ledger().residents(process_vram_bytes=pv):
                mk = row.get("model_key")
                if not mk or mk in seen:
                    continue
                seen.add(mk)
                _crow = {
                    "model_key": mk,
                    "vram_bytes": int(row.get("bytes") or 0),
                    "host_mode": "comfy",
                    "alive": True,
                }
                if _comfy_dev is not None:
                    _crow["gpu_index"] = _comfy_dev
                out.append(_crow)
    except Exception:  # noqa: BLE001 — no ledger / no smi -> registry rows stand
        pass
    # Union in LIVE slot occupants the registry doesn't know (k30). A slot with
    # a model_key claim is a resource allocation whether or not the registry has
    # caught up; its child holds the VRAM. Join nvidia-smi on child_pid for the
    # honest bytes (0 when unjoinable — candidacy is what matters; _fits() is
    # re-measured from the device after each eviction anyway).
    try:
        gpu_procs = None
        for s in (_slot_statuses() or []):
            mk = (s or {}).get("model_key")
            if not mk or mk in seen:
                continue
            if gpu_procs is None:
                gpu_procs = _gpu_process_vram() or {}
            cp = s.get("child_pid")
            info = gpu_procs.get(cp) if cp is not None else None
            vb = int(info["mib"]) * _MIB if info is not None else 0
            seen.add(mk)
            _srow = {
                "model_key": mk,
                "vram_bytes": vb,
                "host_mode": "subprocess",
                "alive": bool(s.get("healthy", True)),
            }
            _sgi = s.get("gpu")
            if _sgi in (None, ""):
                _sgi = s.get("main_gpu")
            if _sgi not in (None, ""):
                try:
                    _srow["gpu_index"] = int(_sgi)
                except (TypeError, ValueError):
                    pass
            out.append(_srow)
    except Exception:  # noqa: BLE001 — slot pool unreadable -> registry rows stand
        pass
    return out


def _target_device_index() -> "int | None":
    """The GPU index the INCOMING model is being placed on — central's per-request
    HUGPY_MAIN_GPU (via the spill seam). None when unpinned (single-GPU box or an
    older central that sends no device), which makes per-device evict a no-op."""
    try:
        from hugpy_engine.spill import main_gpu
        n = main_gpu()
        return int(n) if n is not None else None
    except Exception:  # noqa: BLE001 — no seam / unparseable: no target
        return None


def _partition_residents(state: "WorkerState", model_key: str) -> "tuple[list, list]":
    """Split the measured VRAM residents into ``(candidates, protected)`` for an
    admission of ``model_key``. THE single definition of "what may be evicted".

    Protection (operator ruling, INVIOLABLE): never a 🔒static resident, never one
    ACTIVELY REPLYING, never one with work QUEUED AHEAD of the subject, never a
    comfy that is RENDERING (or cannot be proved idle), never the subject
    itself. Each protected row carries a ``why`` for the honest refusal.

    comfy (2026-09-22, superseding the 0.1.137 blanket exclusion): an IDLE
    comfy's residents are candidates like any on-demand model — a checkpoint
    nothing is using loses to a load that needs the room. Busy is the watchdog's
    predicate (_comfy_busy_reason), asked once per partition; all comfy rows
    share the verdict because comfy /free is all-or-nothing.

    Extracted so the eviction-aware autofit's RECLAIMABLE estimate is computed
    from exactly the rows this admission would really be willing to evict — a
    parallel estimator would be free to drift from the protections and turn the
    optimistic layer target into a lie."""
    busy_slots = _busy_slot_models()
    queued_ahead = _queued_ahead_of(model_key)
    candidates: list[dict] = []
    protected: list[dict] = []
    comfy_busy: "str | None | bool" = False        # False = not asked yet
    # PER-DEVICE evict (2026-09-25): on a MULTI-GPU box the incoming model lands on
    # ONE card (central's HUGPY_MAIN_GPU). Evicting a resident on a SIBLING card
    # frees the wrong VRAM — it never helps the target card and needlessly drops a
    # model. So a resident whose KNOWN card differs from the target is protected.
    # A resident of UNKNOWN card stays a candidate (degrade-not-guess: never block
    # a needed eviction on missing per-device data), and single-GPU / unpinned
    # boxes skip this entirely (byte-identical to before).
    _target_dev = _target_device_index()
    try:
        _multi_gpu = len(detect_gpus() or []) > 1
    except Exception:  # noqa: BLE001
        _multi_gpu = False
    for r in _vram_residents(state):
        mk = r["model_key"]
        if mk == model_key:
            continue
        if _multi_gpu and _target_dev is not None and r.get("gpu_index") is not None \
                and int(r["gpu_index"]) != int(_target_dev):
            protected.append({**r, "why": (f"on GPU {r['gpu_index']}, target is "
                                           f"GPU {_target_dev} (per-device evict)")})
            continue
        if str(r.get("host_mode")) == "comfy":
            if comfy_busy is False:
                comfy_busy = _comfy_busy_reason(state)
            if comfy_busy:
                protected.append({**r, "why": f"comfy busy: {comfy_busy}"})
            else:
                candidates.append(r)
            continue
        if _residency(mk) == "static":
            protected.append({**r, "why": "static (locked residency)"})
            continue
        if _actively_replying(mk, busy_slots):
            protected.append({**r, "why": "actively replying (in-flight/busy)"})
            continue
        if mk in queued_ahead:
            protected.append({**r, "why": "queued ahead of the subject"})
            continue
        candidates.append(r)
    return candidates, protected


def _subject_resident_vram_bytes(state: "WorkerState", model_key: str) -> int:
    """MEASURED VRAM the SUBJECT of an admission ALREADY holds on this card.

    THE SUBJECT-IS-ITS-OWN-BLOCKER FIX (operator, 2026-07-27). ae/0.1.216 refused
    a model that was resident AND serving at that instant:

        "needs 12.0 GB, 8.1 GB free of 23.6 GB (2.4 GB ceiling reserve);
         evicted 0 idle resident(s) freeing 0 B; no evictable resident is
         attributable to a model, yet ~15.4 GB of the card is in use"

    …while the console showed the very same model resident at 13.3 GiB attributed
    + 1.6 GiB KV, pid-mapped, ``vram_unattributed_bytes = 0``. The arithmetic that
    should have decided it: 8.1 free + 12.8 the subject itself holds = 20.9 GB
    against a 12.0 GB need — it fits trivially. Admission never credited the
    subject's OWN footprint, and ``_partition_residents`` (correctly) drops the
    subject from BOTH halves, so with the subject as the ONLY resident the planner
    saw an empty pool on an occupied card and refused.

    Why a CREDIT and not a bare "already resident -> proceed" short circuit: an
    admission is a (RE)SEAT. The re-seat may want MORE than the current placement
    (a relaunch at a higher ngl / a wider ctx), and a short circuit would wave
    that through unmeasured — admit-then-OOM. Crediting the footprint keeps the
    fit test real: the DELTA still has to fit.

    ANTI-DOUBLE-COUNT (the whole risk of this credit — see the
    ``vram-admission-no-evict`` landmine). Only bytes that are BOTH measured and
    still held are credited:

      * rows are the pid-registry/slot-union MEASURED truth (``_vram_residents``),
        the same source the eviction planner ranks — not a declared figure;
      * ``alive`` is required. A dead/reaped row's bytes are already back in
        ``_free_vram_bytes()``; crediting them would count them twice. Under-
        crediting is the safe direction (it degrades to today's refusal), so a
        mid-load slot child reading ``healthy=False`` is deliberately NOT credited;
      * ``vram_bytes <= 0`` (a row we could not join to nvidia-smi) contributes
        nothing — degrade-not-guess, never an invented figure;
      * comfy rows are excluded: comfy is out of allocations (0.1.137) and would
        never be a hugpy model_key subject anyway.

    A SECOND live copy of one model_key would make this credit a lie, so that was
    checked rather than assumed: the in-process GGUF cache
    (``llama/runners/get.py::_LLAMA_INSTANCES``) is keyed by model_key alone, the
    transformers cache (``generate/coder.py::_Registry``) by
    ``DeepCoderConfig.cache_key()`` (model_dir + compute, task-independent), and
    ``SlotPool.endpoint_for`` returns the existing seat for a matching model_key
    before it ever reaches the ceiling gate. One model_key = one seat, so the
    subject's bytes are genuinely the bytes this admission supersedes.

    0 when the subject is not resident — which makes every non-resident admission
    byte-identical to today."""
    try:
        rows = _vram_residents(state) or []
    except Exception:  # noqa: BLE001 — unreadable residents -> no credit (today)
        return 0
    total = 0
    for r in rows:
        if r.get("model_key") != model_key:
            continue
        if str(r.get("host_mode")) == "comfy":
            continue
        if not r.get("alive", True):
            continue                         # already freed -> would double-count
        try:
            vb = int(r.get("vram_bytes") or 0)
        except (TypeError, ValueError):
            continue                         # unmeasurable -> credit nothing
        if vb > 0:
            total += vb
    return total


def _vram_occupancy_attribution() -> dict:
    """``{vram_attributed_bytes, vram_unattributed_bytes}`` for the card RIGHT
    NOW, from the SAME pid-registry snapshot the heartbeat ships to central.

    This is the honesty input for the refusal's "who holds the card" clause. The
    console and the refusal must not disagree about whether occupancy is
    attributable: on ae the refusal claimed "GPU memory is held by process(es)
    this worker cannot map to a model_key (orphaned/adopted child or out-of-band
    process)" while the very same snapshot reported ``vram_unattributed_bytes =
    0`` and mapped the model to its pid. That wrong sentence sent the operator
    hunting external PIDs. Both figures None when there is no readable pid log —
    degrade-not-guess; the refusal then says attribution is unmeasurable rather
    than naming a culprit."""
    try:
        from hugpy_fleet.worker import pid_registry as _pidreg
        return _vram_split_from_pidlog(_pidreg.snapshot_for_heartbeat())
    except Exception:  # noqa: BLE001
        return {"vram_attributed_bytes": None, "vram_unattributed_bytes": None}


def _model_alloc_mode(model_key: str) -> "str | None":
    """The per-model allocation MODE as the operator persisted (or central
    derived) it — the PREFERENCE input to the shared eviction sort's key ①.

    Central projects a ``{model_key: mode}`` map into the worker settings the
    same way it projects ``ctx_pct`` / ``priority`` (the release-time bridge);
    tests populate it directly. Absent -> None, which
    ``eviction.preferred_device`` degrades to VRAM — i.e. the blank max-gpu
    default, which makes key ① a constant and leaves the order byte-identical
    to the idle/calls/key ordering. DEGRADE-NOT-GUESS: an unknown preference
    never invents a cliff-order verdict.

    ⚠ BOTH ``{}`` (derived max-gpu) and ``{"alloc_mode": "max-gpu"}`` (explicit,
    b0e02ff) mean max-gpu FOR PREFERENCE PURPOSES — they differ in provenance
    only. That is why this reads the resolved mode NAME and never a raw spill."""
    val = (_RUNTIME_SETTINGS.get("alloc_mode") or {}).get(model_key)
    return str(val) if val else None


def _model_call_stats(state: "WorkerState | None",
                      model_key: str) -> "tuple[float | None, int]":
    """``(last_activity_epoch, total_calls)`` for one model, from THE ONE LEDGER.

    PARITY: idle times must come from central's call log, shipped at emission
    (``state.model_last_picked``, adopted in _adopt_storage_inputs), never from
    this box's own clock — the spec names divergent victim sets as the failure
    mode that rule prevents. The worker's local dispatch LRU is used only as a
    FALLBACK for a model central has no entry for (a locally-loaded model that
    never routed through central); a local clock is better than 0 there, and it
    cannot cause divergence because central has no opinion to diverge from.

    ENACTED PROPOSAL 1 (spec "Open"): the last-activity figure is
    ``max(request start, last token emitted)``, not bare request start — a long
    stream must not read as idle. ``dispatch.note_used`` is called on token
    flow, so the local snapshot already carries the token side; central's stamp
    carries the request-start side; the max of the two is the honest answer."""
    lp = None
    try:
        lp = (getattr(state, "model_last_picked", None) or {}).get(model_key)
    except Exception:  # noqa: BLE001
        lp = None
    local = None
    try:
        from hugpy_engine.dispatch.dispatch import last_used_snapshot as _lus
        local = (_lus() or {}).get(model_key)
    except Exception:  # noqa: BLE001
        local = None
    vals = [float(v) for v in (lp, local) if v]
    last = max(vals) if vals else None
    calls = 0
    try:
        calls = int((_RUNTIME_SETTINGS.get("model_calls") or {}).get(model_key) or 0)
    except (TypeError, ValueError):
        calls = 0
    return last, calls


def _evict_residents(state: "WorkerState", candidates: "list[dict]",
                     model_key: str) -> list:
    """Build the shared-eviction ``EvictUnit`` rows for a VRAM admission.

    ``candidates`` is ``_partition_residents``' UNPROTECTED half, so the
    operator's inviolable protections (🔒static / actively replying / queued
    ahead / comfy / the subject) have ALREADY removed their rows — that ruling
    outranks the spec's "pool minus static" and is applied upstream, not here.
    What this adds is the spec's own inputs: the preference (key ①), the one
    ledger's idle + call count (keys ②③), and the two enacted proposals
    (in-flight, resident_since for the thrash floor)."""
    from hugpy_engine import eviction as _ev
    busy = _busy_slot_models()
    rows = []
    for r in candidates:
        mk = r["model_key"]
        last, calls = _model_call_stats(state, mk)
        rows.append(_ev.EvictUnit(
            model_key=mk,
            bytes=(int(r.get("vram_bytes") or 0) or None),
            pref=_ev.preferred_device(_model_alloc_mode(mk)),
            last_call=last, calls=calls,
            # `_partition_residents` already excludes actively-replying, so this
            # is belt-and-braces for a row that became busy since the snapshot.
            mid_generation=_actively_replying(mk, busy),
            resident_since=r.get("resident_since") or r.get("loaded_at")))
    return rows


def _shared_evict_order(rows: list, need: "int | None",
                        priority: "dict | None" = None) -> "list[str]":
    """Model keys to evict, in order — via THE shared function, with the
    operator's flex priority as the OUTER key.

    ``need`` None (unmeasurable free VRAM) -> DEGRADE-NOT-GUESS: return the
    full pool in shared-key order and let the caller's incremental ``_fits()``
    loop stop it, which is exactly today's behaviour. A guessed need would
    produce a guessed victim set, and the whole point of walk-then-drop is that
    the set is derivable from real numbers.

    Priority grouping is what keeps least-reaping honest under an override: the
    drop pass runs WITHIN a priority band, so a high-priority resident is never
    spared by a low-priority one covering the need (which would silently invert
    the operator's ordering) — the bands are walked in order and each is
    planned against the need that REMAINS."""
    from hugpy_engine import eviction as _ev
    pri = priority or {}
    now = time.time()
    if need is None:
        ordered = sorted(rows, key=lambda r: (pri.get(r.model_key, 0),
                                              _ev.sort_key(r, _ev.VRAM, now)))
        return [r.model_key for r in ordered]
    out: list[str] = []
    remaining = int(need)
    for band in sorted({pri.get(r.model_key, 0) for r in rows}):
        if remaining <= 0:
            break
        members = [r for r in rows if pri.get(r.model_key, 0) == band]
        plan = _ev.evict_plan(_ev.VRAM, remaining, members, now=now,
                              least_reaping=_evict_least_reaping())
        out.extend(plan.victims)
        remaining -= plan.freed
    return out


# RETIRED 2026-07-27 (operator: "is there still some timeblock on a model being
# evicted? if so eliminate it"). `evict_min_residency_s` / the
# HUGPY_EVICT_MIN_RESIDENCY_S env used to veto eviction of any model resident
# for less than 300s. It was a clock-driven THIRD protection class, and the
# standing ruling is that exactly two exist — 🔒static and actively-answering.
# Nothing reads it now; a stored setting or a drop-in env carrying it is inert.


# Least reaping (the drop pass). FLEET-WIDE, not per-worker — it changes the
# drop pass central's storage_proposal runs too, so a per-worker value would
# produce divergent victim sets (the failure Parity exists to prevent). The
# worker LEARNS it from central on the heartbeat reply (the blocked_models
# idiom) and _adopt_least_reaping projects it onto this env; a local drop-in is
# the fallback only until the first beat lands.
_ENV_LEAST_REAPING = "HUGPY_EVICT_LEAST_REAPING"


def _evict_least_reaping() -> bool:
    from hugpy_engine import eviction as _ev
    raw = os.environ.get(_ENV_LEAST_REAPING)
    if raw in (None, ""):
        return _ev.DEFAULT_LEAST_REAPING
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def _reclaimable_vram_bytes(candidates: "list[dict]") -> "int | None":
    """VRAM the admission could plausibly reclaim = the summed measured footprint
    of the UNPROTECTED residents (``_partition_residents``' candidates).

    Returns None when the estimate cannot be made CONFIDENTLY, which — per the
    degrade-not-guess doctrine — is the signal to keep today's plan-against-actual
    behaviour rather than size up on a number we invented:

      * no candidates at all -> None (nothing to reclaim; today's path exactly);
      * ANY candidate whose measured ``vram_bytes`` is 0/absent -> None. A row we
        could not join to nvidia-smi is a real occupant of unknown size; counting
        it as 0 would understate reclaimable (harmless) but counting the SET as
        complete would overstate our confidence. We refuse to guess either way.

    This is a TARGET, never a promise: between planning and executing a candidate
    can become static/busy/replying, so the caller MUST re-measure after the
    eviction and re-plan from what was actually freed."""
    if not candidates:
        return None
    total = 0
    for r in candidates:
        try:
            vb = int(r.get("vram_bytes") or 0)
        except (TypeError, ValueError):
            return None
        if vb <= 0:
            return None                      # unmeasurable occupant -> don't guess
        total += vb
    return total or None


def _autofit_layers_for(path: str, free_vram: "int | None",
                        extra_reserve: int = 0,
                        n_ctx: "int | None" = None) -> "int | None":
    """``autofit_gpu_layers`` against an EXPLICIT free-VRAM figure, or None when it
    can't be evaluated. Thin wrapper so the eviction-aware planner and the slot
    child agree on one layer-fitting formula (no parallel estimator).

    ``n_ctx`` is the context the child will be launched with. The context reserve
    is linear in it (spill.vram_ctx_reserve_bytes), so the planner must price the
    SAME ctx the slot child will — otherwise the plan it hands the child (which
    lands as an explicit n_gpu_layers) is sized against a different context than
    the one that gets served. None -> spill derives the loader's own choice."""
    if not path or not free_vram or free_vram <= 0:
        return None
    try:
        from hugpy_engine import spill as _spill
        return _spill.autofit_gpu_layers(path, free_vram=int(free_vram),
                                         extra_reserve_bytes=int(extra_reserve or 0),
                                         n_ctx=n_ctx)
    except Exception:  # noqa: BLE001 — unmeasurable -> caller keeps today's path
        return None


def _plan_autofit_against_reclaimable(state: "WorkerState", model_key: str,
                                      candidates: "list[dict]",
                                      free_now: "int | None") -> "dict | None":
    """EVICTION-AWARE AUTOFIT (operator, 2026-07-25): "size up for eviction; spill
    based on the theoretical eviction's success".

    The defect this closes: autofit plans the child's layer count against the VRAM
    free AT THAT INSTANT, and the count is then fixed for the life of the child.
    A model seated while the card is MOMENTARILY busy is crippled PERMANENTLY —
    flux2-klein-9b sat at 21/36 layers on computron with 3.1 GiB still free, a ~4x
    throughput loss (a dense GGUF measured ~135 tok/s fully resident vs ~36 the
    moment one layer spills) that looks perfectly healthy from central.

    The inversion: plan against free + RECLAIMABLE rather than free alone. Returns
    a proposal ``{"target_layers", "free_now", "reclaimable", "path",
    "extra_reserve", "layers_now"}`` when sizing up is BOTH possible and useful,
    else None (caller keeps today's behaviour, byte-identical):

      * non-GGUF / unresolvable geometry            -> None
      * reclaimable unmeasurable or nothing evictable -> None (degrade-not-guess)
      * autofit against free-now already returns -1 (all layers)  -> None
      * the optimistic plan is no better than the current one     -> None

    The returned ``target_layers`` is a TARGET. The caller evicts, RE-MEASURES and
    RE-PLANS; it must never launch this number against VRAM that did not
    materialise (that is `vram-admission-no-evict`, admit-then-crash)."""
    # AUTO path only: an explicit operator n_gpu_layers still wins absolutely.
    intent, requested = _gguf_ngl_intent(model_key)
    if intent != "auto" or requested is not None:
        return None
    path, total_layers = _served_gguf_geometry(model_key)
    if not path or not total_layers:
        return None                          # non-GGUF / no geometry -> today
    reclaimable = _reclaimable_vram_bytes(candidates)
    if not reclaimable or not free_now:
        return None                          # degrade-not-guess
    try:
        from hugpy_engine.spill import vision_projector_bytes as _vpb
        extra_reserve = int(_vpb(path) or 0)
    except Exception:  # noqa: BLE001
        extra_reserve = 0
    # The ctx this model is ALLOCATED (ctx_pct) — the same number the slot child
    # will serve, so planner and child price the same KV cache. None when ctx_pct
    # is unset: spill then derives the loader's own default, exactly as the child
    # does (byte-identical to before this became a variable).
    try:
        _ctx_plan = _resolved_ctx(model_key)[0]
    except Exception:  # noqa: BLE001 — a ctx read must never block the planner
        _ctx_plan = None
    layers_now = _autofit_layers_for(path, free_now, extra_reserve, _ctx_plan)
    if layers_now is None or layers_now == -1:
        return None                          # already plans every layer -> today
    target = _autofit_layers_for(path, int(free_now) + int(reclaimable),
                                 extra_reserve, _ctx_plan)
    if target is None:
        return None
    if target != -1 and target <= layers_now:
        return None                          # eviction buys nothing -> today
    return {"target_layers": target, "layers_now": layers_now,
            "free_now": int(free_now), "reclaimable": int(reclaimable),
            "path": path, "total_layers": int(total_layers),
            "extra_reserve": extra_reserve, "n_ctx": _ctx_plan}


def _size_up_for_eviction(state: "WorkerState", model_key: str, plan: dict,
                          lru: "dict | None" = None) -> dict:
    """Realise an eviction-aware autofit proposal, then **RE-PLAN**.

    THE HARD REQUIREMENT. ``plan["target_layers"]`` was sized against a
    THEORETICAL figure (free + reclaimable). Between planning and executing, a
    candidate can go static / start replying / take work queued ahead, so the
    eviction may UNDER-DELIVER. Launching the optimistic count against VRAM we
    did not get is exactly ``vram-admission-no-evict`` (admit-then-crash), a
    recorded landmine. So we evict coldest-first, then RE-MEASURE the device and
    RE-PLAN the layer count from what was ACTUALLY freed, and emit THAT.

    The optimistic number is a target, never a promise: the emitted
    ``n_gpu_layers`` is always the re-planned one, and it can only ever be >= the
    count the child would have autofitted for itself (we never make a seat worse
    than today — a shortfall degrades back toward today's plan, and an eviction
    that delivers nothing returns the plain ``proceed`` today would have given).
    """
    path = plan["path"]
    extra = plan.get("extra_reserve") or 0
    _ctx_plan = plan.get("n_ctx")            # same ctx the plan was sized against
    if lru is None:
        try:
            from hugpy_engine.dispatch.dispatch import last_used_snapshot as _lus
            lru = _lus()
        except Exception:  # noqa: BLE001
            lru = {}

    # Re-partition at EXECUTION time (not from the planning snapshot) so a
    # resident that just became protected is never touched. Protections are
    # never overridden to make a plan fit.
    candidates, _protected = _partition_residents(state, model_key)
    from hugpy_fleet.worker.flex import flex_priority_key as _fpk
    # THE SHARED EVICT ORDER (spec assets/evictionflow.html). Same function and
    # same key as the admission path — a size-up that ranked victims differently
    # from the admission it is optimising would be a second, divergent policy.
    # need=None: this loop's stopping rule is a LAYER TARGET re-measured per
    # round, not a byte need, so it takes the ordering and does its own walk
    # (degrade-not-guess — no invented byte figure to plan a drop pass against).
    _sz_order = _shared_evict_order(
        _evict_residents(state, candidates, model_key), need=None,
        priority={r["model_key"]: _fpk(_flex_alloc(r["model_key"]))
                  for r in candidates})
    _sz_by_mk = {r["model_key"]: r for r in candidates}
    candidates = [_sz_by_mk[mk] for mk in _sz_order if mk in _sz_by_mk]

    evicted: list[str] = []
    freed = 0
    target = plan["target_layers"]
    for r in candidates:
        # Stop as soon as the layer target is ACHIEVED — the minimum eviction
        # set, same discipline as the fit loop. Re-planned from the device each
        # round, so we never evict past what the plan actually needs.
        cur = _autofit_layers_for(path, _free_vram_bytes(), extra, _ctx_plan)
        if cur == -1 or (target != -1 and cur is not None and cur >= target):
            break
        mk = r["model_key"]
        # RE-CHECK protection immediately before acting. The partition above is
        # a snapshot; a resident can go static / start replying / take work
        # queued ahead DURING this loop, and protections are inviolable — a
        # size-up (which nothing forces us to do at all) must never be the thing
        # that evicts a now-protected model. Its bytes simply drop out of the
        # reclaim, and the re-plan below absorbs the shortfall.
        if mk not in {c["model_key"] for c in _partition_residents(state, model_key)[0]}:
            logger.info("size-up autofit: %s became protected mid-plan — "
                        "skipping it (the re-plan absorbs the shortfall)", mk)
            continue
        res = _evict_model(state, mk)
        if res.get("evicted"):
            fb = res.get("vram_freed")
            freed += int(fb) if fb else 0
            evicted.append(mk)
            _note_vram_eviction(mk, model_key, fb, res.get("host_mode") or "")
            _trim_host_ram()                 # so the next re-measure sees the room
        else:
            logger.warning("size-up autofit: eviction of %s did not free it "
                           "(%s: %s)", mk, res.get("host_mode"), res.get("reason"))

    # ── THE RE-PLAN. Measure the device again and size the layer count against
    #    what MATERIALISED, never against the theoretical figure we planned with.
    final_free = _free_vram_bytes()
    final = _autofit_layers_for(path, final_free, extra, _ctx_plan)
    if final is None:
        # Lost the measurement mid-flight -> emit nothing; the child autofits for
        # itself exactly as today (degrade-not-guess).
        return {"action": "proceed" if not evicted else "evicted",
                "evicted": evicted, "freed_bytes": freed, "reason": None,
                "note": "size-up autofit abandoned: free VRAM unmeasurable after "
                        "eviction — the child autofits itself (today's behaviour)"}
    if final != -1 and final <= plan["layers_now"]:
        # The eviction under-delivered so badly it bought no layers. Say so, and
        # emit no plan rather than a number that isn't better than the default.
        return {"action": "proceed" if not evicted else "evicted",
                "evicted": evicted, "freed_bytes": freed, "reason": None,
                "note": (f"size-up autofit under-delivered: planned {plan['target_layers']} "
                         f"layers against {_human_bytes(plan['free_now'] + plan['reclaimable'])} "
                         f"(free + reclaimable), got {_human_bytes(final_free)} free -> "
                         f"{final} layers; keeping the default autofit")}
    if final != target:
        logger.info(
            "size-up autofit RE-PLANNED for %s: target was %s layers against "
            "~%s (free + reclaimable), eviction delivered ~%s free -> serving %s "
            "layers (never the optimistic count against VRAM we didn't get)",
            model_key, target, _human_bytes(plan["free_now"] + plan["reclaimable"]),
            _human_bytes(final_free), final)
    else:
        logger.info(
            "size-up autofit for %s: %s/%s layers (was planning %s against %s "
            "free) after evicting %d reclaimable resident(s) freeing %s",
            model_key, ("all" if final == -1 else final), plan["total_layers"],
            plan["layers_now"], _human_bytes(plan["free_now"]), len(evicted),
            _human_bytes(freed))

    # Pin the re-planned count for the in-process llama_cpp load AND carry it in
    # the verdict so the slot child launches with exactly this --n-gpu-layers
    # instead of re-autofitting against a number that may have moved again.
    try:
        from hugpy_engine import spill as _spill
        _spill.set_ngl_override(path, final)
    except Exception:  # noqa: BLE001 — the verdict still carries N
        pass
    _PARTIAL_NGL[model_key] = {"path": path, "n": final}
    return {"action": "partial", "evicted": evicted, "freed_bytes": freed,
            "reason": None, "n_gpu_layers": final,
            "size_up": {"target_layers": target, "planned_layers": final,
                        "layers_without_eviction": plan["layers_now"],
                        "total_layers": plan["total_layers"],
                        "free_before_bytes": plan["free_now"],
                        "reclaimable_bytes": plan["reclaimable"],
                        "free_after_bytes": final_free,
                        "evicted": list(evicted), "freed_bytes": freed},
            "note": (f"eviction-aware autofit: "
                     f"{'all' if final == -1 else final}/{plan['total_layers']} "
                     f"layers on GPU (default autofit would have seated "
                     f"{plan['layers_now']}) after reclaiming "
                     f"{_human_bytes(freed)} from {len(evicted)} idle resident(s)")}


def _vram_evict_to_fit(state: "WorkerState", model_key: str,
                       need: "int | None" = None) -> dict:
    """THE VRAM admission choke point. Make room for ``model_key`` to land on the
    GPU under the device ceiling (``_vram_ceiling_reserve_bytes``) by evicting
    the minimum LRU set of EVICTABLE residents, or refuse HONESTLY before any
    CUDA allocation.

    Protection (operator ruling): NEVER evict a 🔒static resident, a model that is
    ACTIVELY REPLYING (measured: in-flight gen / busy slot), a model with work
    QUEUED AHEAD of the subject, comfy (its own path), or the subject itself.
    EVERYTHING else is a candidate — on-demand idle residents, SLOT CHILDREN
    included ('max GPU' alloc included) — LRU/coldest-first, minimum set to fit.

    THE SUBJECT CREDIT (2026-07-27). "It says it's serving it, it gets the call,
    it loads the model, it serves it, then says it cannot because it's not loaded
    and has no room." An admission for an ALREADY-RESIDENT model credits that
    model's own measured footprint on the FREE side
    (``_subject_resident_vram_bytes``), because a (re)seat releases the seat it
    holds. Without it the subject was its own blocker: protected from eviction by
    the operator's ruling, dropped from both halves of ``_partition_residents``,
    and never counted as headroom — so a model plainly on the card was refused
    for want of room it was itself occupying. The credit is a fit input only; the
    DELTA still has to fit, so a re-seat that genuinely wants more than the card
    can give still refuses honestly.

    POLITE LOAD (k56, operator ruling 2026-07-31). When the request's spill
    carries ``no_evict`` (spill.no_evict_env), this function may spend only
    GENUINELY FREE headroom: the tolerance-band flex still runs (compressing the
    subject's OWN ctx costs no resident anything), and the honest GGUF partial
    offload still applies (it is sized from free VRAM alone) — but the size-up
    planner and the eviction walk are SKIPPED and the refusal says so. This is
    the deliberate inverse of declare-need-then-evict, which stays the rule for
    every unflagged load; the flag is per-model and off by default, so nothing
    below changes for an ordinary admission.

    Returns a typed verdict:
      {"action": "proceed"|"evicted"|"refuse", "evicted": [mk...],
       "freed_bytes": int, "reason": {...}|None}
    Fails OPEN (proceed) whenever VRAM/need is unmeasurable — an unmeasurable load
    proceeds exactly as today, never blocked because we couldn't measure."""
    # t21: every admission re-decides from TARGET — drop any stale ctx flex
    # commitment for this subject so the fit check below is against the
    # operator's target ctx, not a prior compression (uncontended == target).
    # Same for any prior PARTIAL-offload commitment: a model that now fits fully
    # must serve fully (-1), never stay pinned to a stale layer count.
    _FLEX_CTX_FLOOR.pop(model_key, None)
    _clear_partial_ngl(model_key)
    # k56: read the polite flag ONCE, here, so every branch below asks the same
    # question of the same request (the env is per-request and cleared when
    # absent, so a mid-admission re-read could disagree with itself).
    # k96: a request-scoped "no_makeroom" (dispatch contextvar — set when this
    # process serves the flagged request itself rather than receiving it as a
    # relayed spill) is the same promise and takes the same polite branch.
    try:
        from hugpy_engine.spill import no_evict_env
        polite = no_evict_env()
    except Exception:  # noqa: BLE001 — an unreadable flag is not a policy
        polite = False
    if not polite:
        try:
            from hugpy_engine.dispatch.dispatch import no_makeroom_active
            polite = no_makeroom_active()
        except Exception:  # noqa: BLE001 — an unreadable flag is not a policy
            pass
    total = _total_vram_bytes()
    if not total:
        return {"action": "proceed", "evicted": [], "freed_bytes": 0,
                "reason": None, "note": "no GPU / unmeasurable — gate is a no-op"}
    # NEED = weights + KV(resolved ctx) (slice 11): the ctx tax is reserved BEFORE
    # the load, and the split is carried for an honest refusal ("24.1G = 21.3G
    # weights + 2.8G kv@50%ctx"). When the caller passed an explicit `need` (a
    # test / a pre-computed total) it wins; otherwise the authoritative detail.
    _det = _incoming_need_detail(model_key)
    if need is None:
        need = _det.get("total")
    if not need:
        return {"action": "proceed", "evicted": [], "freed_bytes": 0,
                "reason": None, "note": "unknown weight size — fail open"}
    # 4-BIT RE-PRICE (operator lever, 2026-07-26). The need detail is computed
    # from the model's fp16 FILE SIZE, so with the lever on, admission was
    # refusing a load that would actually have fit: the console projected
    # "13.1 GiB VRAM planned" while this check demanded 50.2 GB and returned
    # LoadRefusal. Price what will ACTUALLY be loaded, using the same ratio
    # central derived the projection from, so the two agree by construction.
    try:
        from hugpy_engine.spill import bnb_4bit_env
        if bnb_4bit_env():
            from hugpy_engine.alloc_modes import BNB_4BIT_SIZE_RATIO
            need = int(need * BNB_4BIT_SIZE_RATIO)
    except Exception:  # noqa: BLE001 — never break admission over the lever
        pass
    # PLACEMENT-INTENT RE-PRICE (operator incident 2026-07-29). Same invariant as
    # the 4-bit re-price above and the MoE re-target below: admission MUST price
    # what will ACTUALLY land on the card. A RAM-only designation (n_gpu_layers
    # "off" -> the loader builds a 0-GiB GPU budget; max-ram -> only the
    # remainder over the CPU budget) was being priced at the FULL fp16 total:
    # "won't fit on GPU: needs 70.2 GB" — and an idle resident was evicted on
    # the way to that refusal — for a load about to put 0 B on the GPU.
    # planned_gpu_need_bytes mirrors the loader's own derivation, one function,
    # so gate and loader cannot disagree.
    try:
        from hugpy_engine.spill import planned_gpu_need_bytes
        _planned = planned_gpu_need_bytes(need)
        if _planned is not None and _planned < need:
            if _planned <= 0:
                return {"action": "proceed", "evicted": [], "freed_bytes": 0,
                        "reason": None,
                        "note": ("placement intent puts 0 B on the GPU "
                                 "(CPU/RAM-only) — VRAM admission is a no-op")}
            need = int(_planned)
            _det = dict(_det)
            _det["intent_gpu_remainder"] = need
    except Exception:  # noqa: BLE001 — never break admission over the intent
        pass
    # THE admission reserve (2026-07-27): a bounded compute/activation cushion,
    # un-stacked against the external floor already out of `_free_vram_bytes()`
    # — NOT a percentage of the card that re-charges for the KV `need` already
    # carries. Derivation + measurement: see _vram_ceiling_reserve_bytes. Every
    # downstream consumer in this function (the fit test, the flex deficit, the
    # eviction need, the partial-offload budget, the refusal report) reads THIS
    # one binding, so no two of them can price the ceiling differently.
    ceiling_reserve = _vram_ceiling_reserve_bytes(total)

    # ── MoE re-target: typed bytes over the opaque byte-bag ─────────────────
    # When a MoE split governs this model, the plan that will ACTUALLY load is
    # the split (slot_agent._build_cmd applies it), so the whole admission
    # (flex, eviction, fit checks) must price ITS GPU need (non-expert + KV)
    # rather than the opaque full-file total. Pricing the full file would evict
    # innocent residents to make room for bytes that are never going to land on
    # the card — and, when the model is bigger than the card, refuse anyway.
    #
    # POLICY CHANGE (operator, 2026-07-25 — "the default needs to be the MoE
    # split for all GGUFs that it applies to"): this used to re-target ONLY when
    # the full weights could never fit the card even empty (need > card -
    # reserve); a MoE that fit whole was admitted, and served, fully on the GPU.
    # The ae measurement retires that exception (+59% tok/s at 5x less VRAM), so
    # the split is now the default whenever a plan exists — and admission MUST
    # agree with _build_cmd or it would reserve ~5x the VRAM the child actually
    # takes and evict neighbours for nothing.
    #
    # RAM-guarded: experts must fit budgetable host RAM (fail open on
    # unmeasurable). Only when `need` came from the authoritative detail — an
    # explicit caller-passed need is a test/pre-priced figure and stands.
    moe_commit = None
    if need == _det.get("total"):
        ms = _det.get("moe_split")
        if ms:
            fr_now = _free_ram_bytes()
            exp_bytes = int(ms.get("cpu_bytes") or 0)
            # "Could the full weights EVER land on this card?" — priced with the
            # SAME arithmetic the gate uses (total, less the external floor that
            # never reaches the budgetable free read, less the admission
            # reserve), so the log line cannot claim "never fits" about a need
            # the gate would in fact admit on an empty card. Log-only.
            impossible_full = int(need) > _vram_empty_card_budget(total)
            # EXPERT BYTES ARE MMAP-ELIGIBLE (worker-side analogue of the
            # 0.1.237 central ruling, coder-next/ae 2026-08-28): the expert
            # tensors are FILE-BACKED page cache llama.cpp streams via mmap —
            # reclaimable, never an OOM-able reservation — so the guard prices
            # them against the BOX (total RAM), not the momentary free figure.
            # ae live: 43.6 GiB experts vs 124.9 GiB installed was being
            # refused because only ~34 GiB happened to be free at that instant
            # (page cache the very load would reclaim), forcing full-need
            # 51.8 GB admission and a guaranteed refusal. Momentary free stays
            # the fallback when the total is unmeasurable (degrade-not-guess).
            ram_ceiling = _ram_total_bytes() or fr_now
            if ram_ceiling is not None and exp_bytes > ram_ceiling * 0.95:
                logger.info(
                    "MoE split for %s skipped: expert tensors (~%s) exceed "
                    "host RAM (~%s) — keeping full-need admission",
                    model_key, _human_bytes(exp_bytes), _human_bytes(ram_ceiling))
            elif ms.get("gpu_total"):
                need = int(ms["gpu_total"])
                moe_commit = dict(ms)
                logger.info(
                    "MoE re-target for %s: %s — admission prices the expert "
                    "split (GPU need %s, experts ~%s to CPU)", model_key,
                    ("full weights can never fit this card" if impossible_full
                     else "the expert split is the default placement"),
                    _human_bytes(need), _human_bytes(exp_bytes))

    def _moe_admit_verdict(evicted_list, freed_bytes) -> dict:
        """Commit + emit the MoE-split admit: n_gpu_layers=-1 + --n-cpu-moe
        rides the verdict to the slot child; the _MOE_SPLIT marker keeps the
        calibration verdict honest ("partial", never the full-load ratio)."""
        _MOE_SPLIT[model_key] = {"path": moe_commit.get("path"),
                                 "n_cpu_moe": moe_commit["n_cpu_moe"]}
        return {"action": "partial", "evicted": evicted_list,
                "freed_bytes": freed_bytes, "reason": None,
                "n_gpu_layers": -1, "n_cpu_moe": moe_commit["n_cpu_moe"],
                "moe": {k: moe_commit.get(k) for k in
                        ("n_cpu_moe", "gpu_total", "cpu_bytes", "expert_count",
                         "expert_used_count", "sparsity")},
                "note": (f"MoE expert split (--n-cpu-moe "
                         f"{moe_commit['n_cpu_moe']}): all layers on GPU "
                         f"(~{_human_bytes(moe_commit.get('gpu_total'))} of "
                         f"dense backbone + the expert layers the budget bought "
                         f"+ KV), the remaining expert tensors "
                         f"(~{_human_bytes(moe_commit.get('cpu_bytes'))}) on CPU")}

    # ── SUBJECT CREDIT (operator, 2026-07-27) ───────────────────────────────
    # The subject's OWN measured footprint is headroom for the subject: an
    # admission is a (re)seat of THIS model, so whatever it already holds is
    # released by the seat it is about to take. Credited on the FREE side, never
    # subtracted from `need`, deliberately:
    #
    #   * `need` is a property of the MODEL (weights + KV at the resolved ctx),
    #     not of the card. It is the number the refusal PRINTS, the number
    #     _record_calibration_refuse learns from, the base for the MoE commit
    #     (`moe_commit["gpu_total"]`), for the flex re-price, and for the partial
    #     offload's `kv_eff = need - weights`. Netting a card fact into it would
    #     make every one of those lie — and drive kv_eff negative.
    #   * the subject's footprint is a property of the CARD's current state,
    #     which is exactly what the free side means. One credit here covers the
    #     first fit test, the flex deficit, the eviction need, the per-victim
    #     re-check and the partial-offload budget, so no two of them can drift.
    #
    # `_free_vram_bytes()` is re-read every call (the device moves under us and
    # evictions must be seen); the credit is a constant because the subject is
    # never evicted — it is the one resident guaranteed still to be there.
    subject_held = _subject_resident_vram_bytes(state, model_key)
    if subject_held:
        logger.info(
            "VRAM admission for %s: the subject is ALREADY resident holding ~%s "
            "— crediting that against its own need (%s); a model is never "
            "refused for want of room it is itself occupying",
            model_key, _human_bytes(subject_held), _human_bytes(need))

    def _free_eff() -> "int | None":
        """Free VRAM available TO THE SUBJECT = device free + what the subject
        itself already holds. None when the device can't be read (fail open)."""
        fv = _free_vram_bytes()
        if fv is None:
            return None
        return int(fv) + subject_held

    def _fits() -> "bool | None":
        fv = _free_eff()
        if fv is None:
            return None                      # can't read -> fail open at the caller
        return (fv - need) >= ceiling_reserve

    ok = _fits()
    if ok is None:
        return {"action": "proceed", "evicted": [], "freed_bytes": 0,
                "reason": None, "note": "can't read free VRAM — fail open"}
    if ok:
        if moe_commit is not None:
            return _moe_admit_verdict([], 0)
        # ── stage 0.5: EVICTION-AWARE AUTOFIT (operator, 2026-07-25) ─────────
        # The full need fits, so nothing MUST be evicted — but "fits" is a
        # ceiling test, not a placement. The slot child will still autofit its
        # layer count against the VRAM free at THAT instant, and a card that is
        # momentarily busy cripples the seat PERMANENTLY (flux2-klein-9b: 21/36
        # layers with 3.1 GiB free; re-seating put all 36 on the card). Size up
        # for the eviction instead: plan against free + reclaimable, evict to
        # realise it, then RE-PLAN from what was actually freed.
        #
        # Deliberately the RAW device free, NOT `_free_eff()`: this planner is a
        # bonus size-up that never refuses, and `_size_up_for_eviction` re-plans
        # against the raw device read after evicting. Feeding it the credited
        # figure here would make the target systematically un-deliverable and
        # push it down its "under-delivered" branch. Byte-identical to today.
        #
        # k56: a POLITE load skips the size-up entirely. It is a BONUS that buys
        # a better layer count BY EVICTING, and "never evict" outranks "seat it
        # better" — the model takes the room that is genuinely free, which is
        # exactly what the flag promised.
        if polite:
            return {"action": "proceed", "evicted": [], "freed_bytes": 0,
                    "reason": None,
                    "note": ("polite load (no_evict): admitted into free room; "
                             "the eviction-aware size-up was skipped")}
        _up = _plan_autofit_against_reclaimable(
            state, model_key, _partition_residents(state, model_key)[0],
            _free_vram_bytes())
        if _up is not None:
            return _size_up_for_eviction(state, model_key, _up, lru=None)
        return {"action": "proceed", "evicted": [], "freed_bytes": 0, "reason": None}

    # Over the ceiling — plan the minimum eviction set.
    try:
        from hugpy_engine.dispatch.dispatch import last_used_snapshot as _lus
        lru = _lus()
    except Exception:  # noqa: BLE001
        lru = {}

    candidates, protected = _partition_residents(state, model_key)

    # TELEMETRY: stream the PROTECTED half. The console's card is only honest if
    # it shows what was NOT touched and under which clause — "evicted 0, and
    # here is why" is the question the doctrine actually asks. Emitted here (the
    # real admission) rather than inside _partition_residents, which is also
    # called by the autofit estimator where no eviction is being decided.
    for _p in protected:
        _evt_emit("candidate.skip", model_key=_p.get("model_key"),
                  tier=_telemetry_tier(_p.get("host_mode")),
                  incoming_model=model_key, reason=_p.get("why"),
                  vram_bytes=_p.get("vram_bytes"))

    # ── t21 tolerance-band FLEX before evict (stage 1) ──────────────────────
    # Try to fit WITHIN bands before evicting anyone. ctx is the CHEAPEST flex,
    # so plan_flex (1) compresses the SUBJECT's own ctx toward its band floor,
    # then (2) — for a strictly higher-priority subject — reclaims resident KV
    # from lower-priority, UNPROTECTED neighbours within THEIR ctx bands.
    # Protection is absolute: only `candidates` (already protection-filtered
    # above) are offered as flex-eligible neighbours. The pure decision lives in
    # flex.plan_flex; here we EXECUTE the piece that is safe worker-side now —
    # the subject's own ctx compression, which lowers `need` so fewer (or no)
    # residents must be evicted. In-place neighbour KV-shrink rides worker
    # enforcement (next cut); until then a higher-priority subject's precedence
    # manifests as the priority-ordered eviction below.
    from hugpy_fleet.worker.flex import plan_flex, flex_priority_key as _fpk, kv_at_ctx_pct as _kvat
    flex_note = None
    fv_now = _free_eff()                                 # incl. the subject credit
    if fv_now is not None:
        deficit = ceiling_reserve - (fv_now - need)      # >0 here (ok was False)
        # Under a MoE re-target the subject's GPU-side weights are the typed
        # non-expert share (what actually lands on the card), not the full file.
        _subj_weights = _det.get("weights")
        if moe_commit is not None:
            _subj_weights = max(0, int(moe_commit.get("gpu_total") or 0)
                                - int(_det.get("kv") or 0))
        subject = {"weights_bytes": _subj_weights, "kv_bytes": _det.get("kv"),
                   "ctx_pct": _det.get("ctx_pct"),
                   "ctx_deviation_pct": _ctx_deviation_pct(model_key),
                   "priority": _flex_priority(model_key)}
        resident_rows = []
        for r in candidates:                             # unprotected only
            mk = r["model_key"]
            try:
                rkv, rdet = _kv_need_bytes(mk)
            except Exception:  # noqa: BLE001 — a pricing gap must not break admission
                rkv, rdet = 0, {}
            resident_rows.append({
                "model_key": mk, "kv_bytes": int(rkv or 0),
                "ctx_pct": (rdet or {}).get("ctx_pct"),
                "ctx_deviation_pct": _ctx_deviation_pct(mk),
                "vram_bytes": int(r.get("vram_bytes") or 0),
                "protected": False, "pinned": bool(r.get("pinned")),
                "alloc": _flex_alloc(mk)})
        plan = plan_flex(subject, resident_rows, deficit)
        if plan.self_ctx_pct is not None and _det.get("kv"):
            # Commit the subject to its compressed ctx so the SERVED -c and the
            # KV admission reserved agree, and re-price `need` at that floor. The
            # captured _fits() closure re-reads `need`, so this shrinks the fit
            # target for both the flex re-check and the eviction loop. Under a
            # MoE re-target the weight term is the typed non-expert share.
            _FLEX_CTX_FLOOR[model_key] = int(plan.self_ctx_pct)
            new_kv = _kvat(_det.get("kv"), _det.get("ctx_pct"), plan.self_ctx_pct)
            need = int(_subj_weights or 0) + int(new_kv or 0)
            if moe_commit is not None:
                moe_commit["gpu_total"] = int(need)
        if plan.action == "flex" and _fits():
            # Fits WITHIN bands — no eviction. (Neighbour compression in the plan
            # is realised as reduced eviction / priority order until in-place
            # resident shrink lands; self-flex alone already cleared the ceiling.)
            if moe_commit is not None:
                out = _moe_admit_verdict([], 0)
                out["note"] += f"; flex: {plan.note}"
                out["flex"] = plan.as_dict()
                return out
            return {"action": "proceed", "evicted": [], "freed_bytes": 0,
                    "reason": None, "note": f"flex: {plan.note}",
                    "flex": plan.as_dict()}
        flex_note = plan.note

    # ── stage 2: EVICT — the SHARED function (spec assets/evictionflow.html) ─
    # THIS is the site where key ① actually bites: these are DEVICE residents,
    # so a resident whose preference names RAM is already off the cliff by
    # design and yields BEFORE one that asked for this card (whose loss is the
    # measured 135->36 tok/s drop). That is the cliff order, and it is why the
    # old key's largest-first term had to go: size is not a measure of cost.
    #
    # Flex priority stays the OUTER key. It is the operator's explicit
    # per-model override ("explicit priorities"), and an override that the
    # spec's derived ordering could outvote would not be an override. With no
    # priorities set (the default, 0 everywhere) it is a constant and the order
    # is the spec's, exactly.
    #
    # WALK-THEN-DROP is delegated: `_shared_evict_order` returns the walked-and-
    # dropped victim list, so least reaping and the frontier rule hold here as
    # they do in the preview. The loop below still re-tests `_fits()` and
    # re-proves protection per victim (the device moves for reasons no plan
    # models), so the PLAN chooses and the LOOP verifies.
    #
    # k56: a POLITE load does not reach the walk at all. Flex has run (it costs
    # residents nothing), the free room was not enough, and the flag says that
    # is the end of what this load may spend — so `candidates` is emptied and
    # the function falls through to the partial-offload/refusal tail, which is
    # sized from FREE VRAM alone and therefore still honest under the flag.
    # Emptying the list (rather than branching around the loop) is deliberate:
    # every downstream count and telemetry line then reports the truth —
    # nothing was evicted FOR THIS LOAD. What was spared is kept in
    # `polite_spared` so the refusal can NAME it: "3 idle residents were
    # spared" is the whole story here, and the orphan-message honesty fix
    # (2026-07-27) forbids letting an empty candidate list read as "nothing was
    # attributable" when in fact we chose not to touch what was.
    polite_spared: list[dict] = []
    if polite:
        polite_spared = list(candidates)
        logger.info(
            "polite load (no_evict): %s needs %s and free room is short — "
            "NOT evicting any of the %d evictable resident(s); the load takes "
            "free headroom or refuses", model_key, _human_bytes(need),
            len(candidates))
        for _c in candidates:
            _evt_emit("candidate.skip", model_key=_c.get("model_key"),
                      tier=_telemetry_tier(_c.get("host_mode")),
                      incoming_model=model_key,
                      reason="polite load (no_evict) — never evicts",
                      vram_bytes=_c.get("vram_bytes"))
        candidates = []
    _fv_for_need = _free_eff()               # incl. the subject credit: we must
    _ev_need = (max(0, ceiling_reserve - (_fv_for_need - need))
                if _fv_for_need is not None else None)   # only evict the REMAINING
                                                         # deficit, never a victim
                                                         # for the subject's own bytes
    _ev_rows = _evict_residents(state, candidates, model_key)
    _ev_order = _shared_evict_order(
        _ev_rows, need=_ev_need,
        priority={r["model_key"]: _fpk(_flex_alloc(r["model_key"]))
                  for r in candidates})
    _by_mk = {r["model_key"]: r for r in candidates}
    candidates = [_by_mk[mk] for mk in _ev_order if mk in _by_mk]

    evicted: list[str] = []
    evict_failed: list[dict] = []            # attempted but not freed — carried
    freed = 0                                # in the refusal so counts are TRUE
    for r in candidates:
        chk = _fits()
        if chk:
            break
        mk = r["model_key"]
        _ev_tier = _telemetry_tier(r.get("host_mode"))
        _evt_emit("evict.start", model_key=mk, tier=_ev_tier,
                  incoming_model=model_key)
        _ev_t0 = time.time()
        res = _evict_model(state, mk)        # the SAME verb /ops/evict uses
        if res.get("evicted"):
            fb = res.get("vram_freed")
            freed += int(fb) if fb else 0
            evicted.append(mk)
            _note_vram_eviction(mk, model_key, fb, res.get("host_mode") or "")
            _evt_emit("evict.done", model_key=mk, tier=_ev_tier,
                      incoming_model=model_key, freed_bytes=fb,
                      duration_ms=int((time.time() - _ev_t0) * 1000))
            _trim_host_ram()                 # so the next _fits() sees the room
            _evt_emit("reclaim.done", incoming_model=model_key)
        else:
            # An eviction that resolved to a no-op ("not resident here", a
            # changed slot handle, a failed unload) must not vanish from the
            # story — the old message counted only successes, so a card held by
            # an unevictable-in-practice resident read "evicted 0 ... 0
            # protected", contradicting the visible occupancy (k30).
            evict_failed.append({"model_key": mk,
                                 "host_mode": res.get("host_mode"),
                                 "reason": res.get("reason")})
            logger.warning("VRAM evict-to-fit: eviction of %s did not free it "
                           "(%s: %s)", mk, res.get("host_mode"), res.get("reason"))
            _evt_emit("evict.fail", model_key=mk, tier=_ev_tier,
                      incoming_model=model_key,
                      duration_ms=int((time.time() - _ev_t0) * 1000),
                      error=str(res.get("reason") or "eviction freed nothing"))

    final = _fits()
    # ── stage 2.4 (k54): claim IDLE comfy VRAM before degrading or refusing ──
    # Every managed candidate has been walked and it still doesn't fit. Comfy is
    # protected from EVICTION here (it is out of allocations and the worker does
    # not own its residency policy) — but a comfy holding VRAM with an empty
    # queue and no registered call is not serving anyone, and refusing a load (or
    # crippling it into a partial offload) for bytes nobody is using is exactly
    # the gap k54 closes. Contention waives the idle TTL; clauses 1-3 still bind
    # absolutely, so a comfy mid-render is never touched. Costs one cheap /queue
    # read on the unhappy path only, and 0 on a box with no comfy.
    #
    # k56: NOT under a polite load. Reclaiming an idle comfy's VRAM is still
    # taking room off another process to make this load land — the flag says
    # this load spends only what is already free, and "idle" is a judgement the
    # polite promise does not get to make on someone else's behalf.
    _comfy_freed = 0
    if not final and not polite:
        _comfy_freed = _comfy_reclaim_idle_vram(state, model_key, need_bytes=need)
        if _comfy_freed:
            freed += _comfy_freed
            final = _fits()
    if final:
        if moe_commit is not None:
            return _moe_admit_verdict(evicted, freed)
        out = {"action": "evicted", "evicted": evicted,
               "freed_bytes": freed, "reason": None}
        if _comfy_freed:
            # Name it: the operator must never have to guess which of the two
            # reclaim mechanisms actually produced the room.
            out["comfy_freed_bytes"] = _comfy_freed
            out["note"] = (f"reclaimed {_human_bytes(_comfy_freed)} from an idle "
                           f"ComfyUI (empty queue, no call)")
        return out

    # `fv` is the RAW device read — what the refusal below REPORTS, because the
    # operator must see the card as the driver sees it. `fv_eff` is what the
    # subject may actually spend (raw + its own credited footprint) and is what
    # the partial-offload budget is sized from, so the hybrid plan and the
    # `_fits()` that just failed are priced off the same number.
    fv = _free_vram_bytes()
    fv_eff = _free_eff()

    # ── stage (2.5): honest GGUF PARTIAL offload — autofit's hybrid contract ──
    # Full GPU offload still doesn't fit after flex + evict. For a GGUF this is
    # NOT a dead end: autofit's PROMISE (empty spill = the default alloc mode) is
    # a hybrid — offload as many layers as safely fit under the ceiling reserve,
    # stream the rest from disk to CPU RAM. This is the regression this restores:
    # a served-many-times brain must not hard-refuse on a card that plainly holds
    # part of it. Priced from the honest, SHARD-AWARE need split (not the shard-
    # blind on-disk autofit), floored against a degenerate offload and against a
    # CPU remainder that would OOM host RAM (never admit-then-OOM). GGUF/slot path
    # only; transformers placement modes are t26 (out of scope).
    partial = None
    ppath, total_layers = _served_gguf_geometry(model_key)
    if fv_eff is not None and total_layers:
        weights = int(_det.get("weights") or 0)
        kv_eff = max(0, int(need) - weights)     # honors any committed ctx flex
        budget = max(0, fv_eff - ceiling_reserve)  # VRAM the offloaded layers may
                                                   # use — incl. the subject's own
                                                   # bytes, which this seat frees
        # Cap by the model's explicit VRAM band CEILING when a gpu_mem_gib budget
        # is set (t21) — stretchable to the band ceiling under this model's own
        # need. band_ceiling collapses to the gpu_mem_gib target when no deviation
        # is projected (today), i.e. the same cap autofit already applies.
        gpu_mem_gib = os.environ.get("HUGPY_GPU_MEM_GIB")
        if gpu_mem_gib:
            try:
                from hugpy_fleet.worker.flex import band_ceiling
                cap = int(band_ceiling(float(gpu_mem_gib) * (2 ** 30),
                                       _vram_deviation_pct(model_key), total))
                budget = min(budget, cap)
            except (TypeError, ValueError):
                pass
        intent, requested = _gguf_ngl_intent(model_key)
        # MoE FIRST (0.1.241, coder-next/ae 2026-08-28): EVERY partial-offload
        # entry on a detected-MoE GGUF defers to the dense-backbone-first
        # split BEFORE any dense layer math. The live bust this closes: a
        # cold load whose spill shape skipped _moe_plan_for reached
        # plan_partial_offload, which priced a 3-layer DENSE floor (~3.2 GB,
        # dense layer-bytes) against the MoE-derived 1.5 GiB budget
        # (non-expert cap) — mixed bases in one verdict — and refused
        # "degenerate" a model whose MoE plan (backbone ~1.7 GB, experts to
        # RAM) admits fine. Guards: an operator-stated layer count and a cpu
        # (ram-only) intent are always obeyed verbatim; the backbone must
        # actually fit the budget (dense_fits — never admit-then-OOM); the
        # experts must fit the BOX (total RAM, mmap doctrine). Anything else
        # falls through to the dense planners unchanged.
        if requested is None and (intent or "auto") != "cpu":
            _det_moe = _moe_detail_for(model_key)
            _mplan = None
            _mbudget = None
            if _det_moe:
                try:
                    from hugpy_engine import spill as _spill
                    _mbudget = _moe_auto_gpu_budget(model_key, ppath) or budget
                    _mplan = _spill.moe_dense_first_plan(_det_moe, _mbudget)
                except Exception:  # noqa: BLE001 — fall through to dense math
                    _mplan = None
            if _mplan and int(_mplan.get("cpu_bytes") or 0) \
                    and _mplan.get("dense_fits"):
                _exp_b = int(_det_moe.get("expert_bytes") or 0)
                _ram_ceiling = _ram_total_bytes() or _free_ram_bytes()
                if not _ram_ceiling or _exp_b <= _ram_ceiling * 0.95:
                    _MOE_SPLIT[model_key] = {"path": ppath,
                                             "n_cpu_moe": int(_mplan["n_cpu_moe"])}
                    logger.info(
                        "MoE-first partial admit for %s: -ngl -1 --n-cpu-moe %s "
                        "(~%s dense+experts on GPU, ~%s experts to CPU; budget %s) "
                        "— dense layer math skipped", model_key,
                        _mplan["n_cpu_moe"], _human_bytes(_mplan.get("gpu_bytes")),
                        _human_bytes(_mplan.get("cpu_bytes")), _human_bytes(_mbudget))
                    return {"action": "partial", "evicted": evicted,
                            "freed_bytes": freed, "reason": None,
                            "n_gpu_layers": -1,
                            "n_cpu_moe": int(_mplan["n_cpu_moe"]),
                            "note": (f"MoE dense-first split (--n-cpu-moe "
                                     f"{_mplan['n_cpu_moe']}): all layers on GPU, "
                                     f"~{_human_bytes(_mplan.get('cpu_bytes'))} "
                                     f"expert tensors on CPU")}
        # k37: max-ram / explicit route to the leniency-band engine — degrade
        # WITHIN the band toward the floor, bust past it with a refusal naming
        # mode + floor. The mode arrives as env (HUGPY_ALLOC_MODE etc., set by
        # _apply_spill from the version-gated spill keys). gpu-only/ram-only/
        # max-gpu keep today's plan_partial_offload path byte-identical.
        from hugpy_engine.spill import (
            alloc_mode_env as _amode,
            leniency_pct_env as _lenpct,
            priority_device_env as _pdev,
        )
        _mode = _amode()
        if _mode in ("max-ram", "explicit"):
            from hugpy_fleet.worker.flex import plan_explicit_offload
            _gpu_t = os.environ.get("HUGPY_GPU_MEM_GIB")
            _cpu_t = os.environ.get("HUGPY_CPU_MEM_GIB")
            def _gib_bytes(v):
                try:
                    return int(float(v) * (2 ** 30)) if v else None
                except (TypeError, ValueError):
                    return None
            partial = plan_explicit_offload(
                weights_bytes=weights, kv_bytes=kv_eff,
                total_layers=total_layers, vram_budget_bytes=budget,
                ram_free_bytes=_free_ram_bytes(), mode=_mode,
                priority_device=("ram" if _mode == "max-ram" else _pdev()),
                gpu_target_bytes=_gib_bytes(_gpu_t),
                ram_target_bytes=_gib_bytes(_cpu_t),
                # max-ram = explicit(ram priority, 100% target, generous
                # leniency): bust only when RAM+GPU together can't satisfy.
                leniency_pct=(100.0 if _mode == "max-ram" else (_lenpct() or 0.0)))
        else:
            from hugpy_fleet.worker.flex import plan_partial_offload
            partial = plan_partial_offload(
                weights_bytes=weights, kv_bytes=kv_eff, total_layers=total_layers,
                vram_budget_bytes=budget, ram_free_bytes=_free_ram_bytes(),
                intent=intent, requested_layers=requested)

    if partial is not None and partial.admit:
        # k37 + MoE (coder-next/ae 2026-08-28): a MODE-ENGINE admit on a
        # detected-MoE GGUF must ride the DENSE-BACKBONE-FIRST split, not a raw
        # layer count — the slot obeys a stated positive n_gpu_layers verbatim
        # (the k14/k7 lever), which would disable the split and launch the
        # dense layer hybrid the 2026-07-31 ruling forbids. Budget arithmetic
        # is _moe_auto_gpu_budget (mirrors slot_agent._moe_gpu_budget) so
        # admission and launch price ONE number. Operator-stated layer counts
        # never reach this branch (the mode engine runs only when no legacy
        # ngl wire is set), so the lever stays obeyed verbatim.
        if _mode in ("max-ram", "explicit") and partial.n_gpu_layers > 0:
            det_moe = _moe_detail_for(model_key)
            mplan = None
            if det_moe:
                try:
                    from hugpy_engine import spill as _spill
                    _mbudget = _moe_auto_gpu_budget(model_key, ppath)
                    if not _mbudget:
                        _mbudget = int(partial.vram_budget_bytes or 0)
                    mplan = _spill.moe_dense_first_plan(det_moe, _mbudget)
                except Exception:  # noqa: BLE001 — fall back to the layer count
                    mplan = None
            if mplan and int(mplan.get("cpu_bytes") or 0):
                _MOE_SPLIT[model_key] = {"path": ppath,
                                         "n_cpu_moe": int(mplan["n_cpu_moe"])}
                logger.info(
                    "%s admit for %s -> MoE dense-first split: -ngl -1 "
                    "--n-cpu-moe %s (~%s dense+experts on GPU, ~%s experts to "
                    "CPU; budget %s)", _mode, model_key, mplan["n_cpu_moe"],
                    _human_bytes(mplan.get("gpu_bytes")),
                    _human_bytes(mplan.get("cpu_bytes")), _human_bytes(_mbudget))
                return {"action": "partial", "evicted": evicted,
                        "freed_bytes": freed, "reason": None,
                        "n_gpu_layers": -1, "n_cpu_moe": int(mplan["n_cpu_moe"]),
                        "gpu_pct": partial.gpu_pct, "partial": partial.as_dict(),
                        "note": (f"{_mode} MoE split (--n-cpu-moe "
                                 f"{mplan['n_cpu_moe']}): dense backbone first, "
                                 f"~{_human_bytes(mplan.get('cpu_bytes'))} "
                                 f"expert tensors to CPU")}
        # Admit the hybrid. Pin the honest layer count for the in-process
        # llama_cpp load (overriding the shard-blind autofit that re-OOMs a
        # sharded model) AND carry it in the verdict so the slot path launches
        # the child with --n-gpu-layers N instead of -1. Residency is MEASURED
        # post-load (pid-registry) and the slot status reports the real ngl, so
        # the console/bars read the true split with no declared number to drift.
        try:
            from hugpy_engine import spill as _spill
            _spill.set_ngl_override(ppath, partial.n_gpu_layers)
        except Exception:  # noqa: BLE001 — slot opts still carry N; override is a bonus
            pass
        _PARTIAL_NGL[model_key] = {"path": ppath, "n": partial.n_gpu_layers}
        logger.info(
            "partial offload: %s -> %d/%d layers on GPU (%d%%), ~%s VRAM + ~%s RAM "
            "— admitting hybrid instead of refusing (budget %s, ram_free %s)",
            model_key, partial.n_gpu_layers, partial.total_layers, partial.gpu_pct,
            _human_bytes(partial.vram_need_bytes), _human_bytes(partial.ram_need_bytes),
            _human_bytes(partial.vram_budget_bytes), _human_bytes(partial.ram_free_bytes))
        return {"action": "partial", "evicted": evicted, "freed_bytes": freed,
                "reason": None, "n_gpu_layers": partial.n_gpu_layers,
                "gpu_pct": partial.gpu_pct, "partial": partial.as_dict(),
                "note": f"partial GPU offload: {partial.note}"}

    # Still short after eviction AND no admissible partial offload -> HONEST
    # refusal (never admit-then-OOM). Carry what's resident, what's protected +
    # why, what we evicted, the numbers, and — when a partial offload was
    # CONSIDERED and rejected — what it would have been and why (degenerate offload
    # or CPU remainder OOM), in the C4 vision-fit style.
    # t28: a load-FAIL calibration sample (prediction with no successful load).
    _record_calibration_refuse(model_key, _det)
    # A TRUTHFUL account of what holds the card (k30): only claim "protected
    # resident(s) still hold the card" when there ARE protected residents; a
    # failed eviction is reported as such; and when the planner saw NO residents
    # at all on an occupied card, say the occupancy is unattributed instead of
    # the self-contradictory "evicted 0 ... 0 protected still hold the card".
    holders: list[str] = []
    # k56: the polite clause comes FIRST because it is the whole reason this
    # refusal exists — the room was there, this load was told not to take it.
    if polite:
        holders.append(
            "POLITE LOAD (no_evict): this model may spend only genuinely free "
            "headroom, so nothing was evicted"
            + (f" — {len(polite_spared)} evictable idle resident(s) were SPARED "
               f"(~{_human_bytes(sum(int(r.get('vram_bytes') or 0) for r in polite_spared))} "
               f"between them, which an ordinary load would have reclaimed)"
               if polite_spared else " (and nothing was evictable anyway)"))
    if protected:
        holders.append(f"{len(protected)} protected resident(s) still hold the card")
    if evict_failed:
        holders.append(f"{len(evict_failed)} eviction attempt(s) failed to free "
                       "their resident")
    _attr = {"vram_attributed_bytes": None, "vram_unattributed_bytes": None}
    if (not protected and not evict_failed and not candidates and not evicted
            and not polite_spared):
        # THE ORPHAN-MESSAGE HONESTY FIX (operator, 2026-07-27). This clause used
        # to assert, unconditionally, that the card was held by "process(es) this
        # worker cannot map to a model_key (orphaned/adopted child or out-of-band
        # process)". On ae that was FALSE and provably so — the pid registry
        # mapped the subject to its pid and reported vram_unattributed_bytes = 0
        # — and it cost the operator hours hunting external PIDs. An empty
        # candidates+protected pool means "nothing EVICTABLE was enumerable", not
        # "nothing is attributable": `_partition_residents` deliberately drops the
        # SUBJECT from both halves, and a model_key-less cuda_context lump never
        # enters them either. So consult the attribution the worker already
        # computes for the heartbeat, and say only what it supports.
        occupied = None
        if fv is not None and total:
            occupied = max(0, int(total) - int(fv))
        _attr = _vram_occupancy_attribution()
        _unattr = _attr.get("vram_unattributed_bytes")
        _attributed = _attr.get("vram_attributed_bytes")
        if subject_held:
            # THE LIVE ae SHAPE. The subject is the holder — and it is protected
            # from itself by the operator's own ruling, so there was never
            # anything to evict. Name it; do not invent a squatter.
            holders.append(
                f"the only thing holding the card is the SUBJECT ITSELF — "
                f"{model_key} is already resident with ~{_human_bytes(subject_held)} "
                f"(pid-attributed, already credited against its own need above), "
                f"and the subject is never evicted to make room for itself; "
                f"~{_human_bytes(occupied)} of the card is in use")
        elif _unattr:
            holders.append(
                f"no evictable resident is attributable to a model, and "
                f"~{_human_bytes(_unattr)} of the ~{_human_bytes(occupied)} in use "
                f"is measured UNATTRIBUTED — GPU memory is held by process(es) "
                f"this worker cannot map to a model_key (orphaned/adopted child "
                f"or out-of-band process)")
        elif _unattr is None:
            holders.append(
                f"no evictable resident is attributable to a model and "
                f"~{_human_bytes(occupied)} of the card is in use; VRAM "
                f"attribution is UNMEASURABLE on this box, so the holder cannot "
                f"be named (degrade-not-guess)")
        elif not _attributed:
            # Attribution is readable and says NOTHING is attributed, yet the
            # card is occupied. That is the k30 shape and the orphan sentence is
            # true here: the registry can account for none of the occupancy.
            holders.append(
                f"no evictable resident is attributable to a model, yet "
                f"~{_human_bytes(occupied)} of the card is in use — GPU memory is "
                "held by process(es) this worker cannot map to a model_key "
                "(orphaned/adopted child or out-of-band process)")
        else:
            # Occupied, fully accounted for, and none of it evictable — say THAT.
            holders.append(
                f"no evictable resident is attributable to a model, but the "
                f"~{_human_bytes(occupied)} in use IS attributed to this worker "
                f"(~{_human_bytes(_attributed)} across its model rows and its own "
                f"CUDA context) with 0 B measured unattributed — nothing foreign "
                f"is squatting the card, there is simply nothing evictable left")
    _ext_floor = _external_vram_floor_bytes()
    # BASIS-NAMING (worker-side analogue of 0.1.237's central rule): when an
    # offload plan GOVERNED this load, the header's "needs" is the dense
    # full-model figure, not what the plan actually asked of the GPU — say so
    # and quote the plan's own floor, or the refusal reads 51.8 GB about a load
    # whose plan wanted 1.5 GiB on the card (coder-next/ae 2026-08-28).
    _plan_basis = ""
    if partial is not None and not partial.admit and partial.total_layers:
        _plan_whole = int(partial.vram_need_bytes) + int(partial.ram_need_bytes)
        _plan_floor_b = (_plan_whole * int(partial.floor_layers)) // int(partial.total_layers)
        _plan_basis = (
            f" (full-model basis; the governing offload plan asked only "
            f"~{_human_bytes(_plan_floor_b)} of GPU for its floor of "
            f"{partial.floor_layers}/{partial.total_layers} layers)")
    reason = {
        "state": "refused",
        "model_key": model_key,
        "reason": (
            f"won't fit on GPU: needs {_human_bytes(need)}{_need_split_str(_det)}{_plan_basis}, "
            f"{_human_bytes(fv)} free of {_human_bytes(total)} "
            + (f"(+{_human_bytes(subject_held)} the subject itself already holds, "
               f"credited -> {_human_bytes(fv_eff)} available to it) "
               if subject_held else "")
            + f"({_human_bytes(ceiling_reserve)} ceiling reserve"
            # Say where the rest of the headroom went. With the bounded cushion
            # the ceiling reserve is often 0 B on a default box, and a bare
            # "0 B ceiling reserve" reads like the guard is off — it is not: the
            # external floor below is already OUT of the free figure quoted
            # above and is the post-load working room.
            + (f" + {_human_bytes(_ext_floor)} already held back from the free "
               f"figure for out-of-band GPU consumers" if _ext_floor else "")
            + "); "
            f"evicted {len(evicted)} idle resident(s) freeing "
            f"{_human_bytes(freed)}"
            + ("; " + "; ".join(holders) if holders else "")
        ),
        "needs_bytes": need,
        # The weights+kv SPLIT (slice 11) — the honest report the operator asked
        # for: what the ctx allocation costs vs the weights.
        "needs_weights_bytes": _det.get("weights"),
        "needs_kv_bytes": _det.get("kv"),
        "ctx_pct": _det.get("ctx_pct"),
        "ctx_resolved": _det.get("ctx_resolved"),
        "ctx_max": _det.get("ctx_max"),
        "kv_geometry_source": _det.get("geometry_source"),
        "free_vram_bytes": fv,
        "total_vram_bytes": total,
        "ceiling_reserve_bytes": ceiling_reserve,
        "evicted": evicted,
        "evicted_freed_bytes": freed,
        "evict_failed": evict_failed,
        "protected": [{"model_key": p["model_key"],
                       "vram_bytes": p.get("vram_bytes"),
                       "host_mode": p.get("host_mode"), "why": p.get("why")}
                      for p in protected],
    }
    # SUBJECT CREDIT + ATTRIBUTION, structured for the console. WIRE: strictly
    # ADDITIVE and OMITTED-WHEN-UNSET — the fleet is 0.1.216 and a released
    # consumer must not meet a field it can't model. A non-resident subject and
    # an unread attribution therefore produce a byte-identical reason dict.
    if polite:
        # k56, same additive/omitted-when-unset discipline: an unflagged load
        # produces a byte-identical reason dict. Central surfaces these on the
        # telemetry stream so the console can say WHY it didn't land.
        reason["no_evict"] = True
        reason["polite_spared"] = [{"model_key": r.get("model_key"),
                                    "vram_bytes": r.get("vram_bytes"),
                                    "host_mode": r.get("host_mode")}
                                   for r in polite_spared]
    if subject_held:
        reason["subject_resident_bytes"] = subject_held
        reason["free_vram_effective_bytes"] = fv_eff
    if _attr.get("vram_attributed_bytes") is not None:
        reason["vram_attributed_bytes"] = _attr["vram_attributed_bytes"]
    if _attr.get("vram_unattributed_bytes") is not None:
        reason["vram_unattributed_bytes"] = _attr["vram_unattributed_bytes"]
    # MoE honesty: when a split governed (or was considered for) this model,
    # carry its typed numbers so the refusal explains what the split would have
    # needed — the operator sees "even the 2.9G non-expert share didn't fit",
    # not a bare full-size refusal.
    if _det.get("moe_split"):
        reason["moe_split"] = {k: v for k, v in _det["moe_split"].items()
                               if k != "path"}
        reason["moe_split"]["was_plan"] = moe_commit is not None
    if flex_note:
        reason["flex_note"] = flex_note      # what the band flex tried, for hover
    if partial is not None and not partial.admit:
        # A partial offload WAS considered and rejected — say what it would have
        # been and why, so the refusal is honest about the hybrid it declined.
        reason["partial_offload_considered"] = partial.as_dict()
        reason["reason"] = reason["reason"] + "; " + (
            partial.reject_reason or "partial GPU offload not admissible")
    # The load is refused — void any ctx compression / partial-offload commitment
    # we made for it so a future admission of this model re-decides from target.
    _FLEX_CTX_FLOOR.pop(model_key, None)
    _clear_partial_ngl(model_key)
    return {"action": "refuse", "evicted": evicted, "freed_bytes": freed,
            "reason": reason}


# ── Fix B: ensure comfy headroom (evict-to-target-free-VRAM, operator: "always") ─
def _comfy_target_free_bytes() -> int:
    """The LEGACY constant target free VRAM before a ComfyUI gen
    (HUGPY_COMFY_TARGET_FREE_GIB, default 7.0 GiB). Since 2026-09-22 the gen is
    priced per checkpoint (_comfy_need_detail); this constant is the fallback
    when the checkpoint file cannot be sized, and — when the operator set the
    env explicitly — a floor the per-checkpoint need never undercuts.

    Reasoning for the 7.0 default: recon on ae observed ComfyUI's process VRAM
    growing to ~6.5 GiB (5.5 -> 6.5 G) when it drove a gen — that footprint is
    what topped out the 3090 and evicted nothing. 7.0 GiB is that observed peak
    plus a small margin, so the common still/img2img/id_lock comfy gen has room
    to allocate without OOM/under-offload. It's a knob, not a law: a box running
    heavier SDXL/flux comfy graphs raises it; a tiny-model box lowers it. The
    target is a CEILING on eviction effort, never a guarantee — if nothing is
    evictable we proceed anyway (honest-degrade)."""
    gib = os.environ.get("HUGPY_COMFY_TARGET_FREE_GIB")
    if gib is None or not str(gib).strip():
        val = 7.0
    else:
        try:
            val = float(gib)
        except ValueError:
            logger.warning("ignoring non-numeric HUGPY_COMFY_TARGET_FREE_GIB=%r; "
                           "using 7.0", gib)
            val = 7.0
    return int(max(0.0, val) * 2**30)


def _comfy_headroom_candidates(exclude: str | None) -> list[str]:
    """LRU-ordered (coldest first) on-demand managed model_keys that may be
    evicted to free VRAM for a comfy gen. Union of live SLOT occupants (their
    llama-server children hold real VRAM) and genuine IN-PROCESS residents —
    slot-backed keys are excluded from ``loaded_model_keys`` by design, but they
    are exactly what we must free here, so we add them back from a live slot
    read. Static models are dropped (never evictable); the per-key ``_evict_gate``
    inside ``_evict_model`` still guards in-flight generations. Excludes the comfy
    model_key we're generating FOR. Ordered by dispatch's LRU clock so the coldest
    yields first."""
    keys: set[str] = set()
    try:
        keys.update(loaded_model_keys())         # genuine in-process residents
    except Exception:  # noqa: BLE001
        pass
    try:
        from hugpy_engine.serve.slots import SlotPool
        for s in SlotPool().statuses():
            mk = s.get("model_key")
            if mk:
                keys.add(mk)                     # slot children DO hold VRAM
    except Exception:  # noqa: BLE001 — no slots / pool error -> in-process only
        pass
    try:
        # External leases (gpu_lease batch jobs) yield to an image gen exactly
        # like a cold LLM does — they are in neither loaded_model_keys nor the
        # slot pool (foreign pids), so without this they'd be invisible here
        # and a comfy gen could never preempt an OCR run holding the card.
        # Never-touched keys sort coldest (lru 0.0), so the lease yields FIRST.
        # A NON-EVICTABLE lease is dropped here the same way static residency
        # is below — never a candidate, only operator force touches it. (An
        # external-lease claim that must not evict its peers filters these out
        # itself; see _external_claim_headroom.)
        from hugpy_fleet.worker import external_residents as _extres
        for rec in _extres.snapshot():
            if rec.get("evictable", True):
                keys.add(rec["model_key"])
    except Exception:  # noqa: BLE001 — no lease registry -> managed models only
        pass
    if exclude:
        keys.discard(exclude)
    # Drop static (locked) — never a candidate. On-demand (incl. pinned, which
    # yields per 2026-07-15 semantics) stays. The in-flight guard is applied
    # per-key by _evict_model's gate at eviction time.
    cands = [mk for mk in keys if _residency(mk) != "static"]
    try:
        last = _dispatch_last_used()
    except Exception:  # noqa: BLE001
        last = {}
    cands.sort(key=lambda mk: last.get(mk, 0.0))
    return cands


def _dispatch_last_used() -> dict:
    from hugpy_engine.dispatch.dispatch import last_used_snapshot
    return last_used_snapshot()


def _evict_to_free_target(state: "WorkerState", subject: str, target: int,
                          *, exclude: "str | None" = None, job_id=None,
                          target_dev: "int | None" = None) -> dict:
    """THE shared VRAM evict-to-fit loop for in-process gens (ComfyUI AND the
    in-process diffusers image runners). Evict the coldest EVICTABLE resident one
    at a time — the SAME candidate policy (``_comfy_headroom_candidates``: slot
    children + in-process residents + external leases, static dropped), the SAME
    evict verb (``_evict_model`` — the one ``/ops/evict`` proved live), and the
    SAME per-key gates (``_evict_gate`` inside ``_evict_model`` never rips a
    busy/in-flight/static model) — re-measuring real free VRAM after each, until
    free >= ``target`` or nothing eligible remains. There is ONE policy; only the
    target differs per caller (comfy's checkpoint need vs the diffusers footprint).

    Honest-degrade at every seam: no GPU / unmeasurable -> no-op; nothing left to
    evict but still short -> return and let the caller proceed (NEVER blocks/hangs
    the gen). Every REAL eviction is recorded via ``_note_vram_eviction`` so it
    surfaces in ``/llm/evictions`` like every other eviction. Returns telemetry:
    ``{target, free_before, free_after, evicted:[mk...], skipped:[{model_key,
    reason,host_mode}...], reached:bool|None}`` — ``skipped`` names the residents
    that could NOT be evicted (and why), for an honest capacity refusal upstream."""
    exclude = exclude or subject
    fv = _free_vram_bytes()
    if fv is None:
        return {"target": target, "free_before": None, "free_after": None,
                "evicted": [], "skipped": [], "reached": None,
                "note": "no GPU / unmeasurable"}
    free_before = fv
    evicted: list[str] = []
    skipped: list[dict] = []
    tried: set[str] = set()
    # PER-DEVICE (2026-09-25): on a multi-GPU box only residents on the TARGET
    # card free the VRAM this gen needs — evicting a sibling-card model is wasted.
    # Map model_key -> its known card once; keep unknown-card keys eligible
    # (degrade-not-guess). Skipped when unpinned / single-GPU (byte-identical).
    _dev_by_key: dict = {}
    _multi_gpu = False
    if target_dev is not None:
        try:
            _multi_gpu = len(detect_gpus() or []) > 1
            if _multi_gpu:
                _dev_by_key = {r["model_key"]: r.get("gpu_index")
                               for r in _vram_residents(state)}
        except Exception:  # noqa: BLE001 — no per-device data -> no filtering
            _multi_gpu = False

    def _on_target(mk: str) -> bool:
        if not (_multi_gpu and target_dev is not None):
            return True
        gi = _dev_by_key.get(mk)
        return gi is None or int(gi) == int(target_dev)

    while fv < target:
        cands = [mk for mk in _comfy_headroom_candidates(exclude=exclude)
                 if mk not in tried and _on_target(mk)]
        if not cands:
            logger.warning(
                "evict-to-fit(%s): free VRAM %.2fGiB < target %.2fGiB but nothing "
                "on-demand is evictable — proceeding anyway (honest-degrade; not "
                "blocking the request)", subject, fv / 2**30, target / 2**30)
            break
        victim = cands[0]
        tried.add(victim)
        try:
            res = _evict_model(state, victim, force=False)
        except Exception:  # noqa: BLE001 — one bad evict must not wedge the gen
            logger.warning("evict-to-fit(%s): evict of %s raised; skipping",
                           subject, victim, exc_info=True)
            skipped.append({"model_key": victim, "reason": "evict raised",
                            "host_mode": None})
            continue
        if res.get("evicted"):
            fb = res.get("vram_freed")
            evicted.append(victim)
            _note_vram_eviction(victim, subject, fb, res.get("host_mode") or "")
        else:
            # gated (busy/in-flight/static) or freed nothing — name it + why, and
            # DON'T loop on it (tried already advanced) so a gated model can't hang
            # the pass.
            skipped.append({"model_key": victim,
                            "reason": res.get("reason") or "not evicted",
                            "host_mode": res.get("host_mode")})
        fv = _free_vram_bytes()
        if fv is None:
            break
    reached = (fv is not None and fv >= target)
    return {"target": target, "free_before": free_before, "free_after": fv,
            "evicted": evicted, "skipped": skipped, "reached": reached}


def _imagegen_gen_cushion_bytes() -> int:
    """Activation/arena cushion added on top of the diffusers model's weight
    bytes when computing the evict-to-fit target (HUGPY_IMAGEGEN_GEN_CUSHION_GIB,
    default 1.5 GiB). The generation itself needs working VRAM beyond the resident
    weights; clearing weights+cushion means the fp16 pipeline lands on the GPU
    with room to run rather than OOMing at the first sampler step."""
    gib = os.environ.get("HUGPY_IMAGEGEN_GEN_CUSHION_GIB")
    if gib is None or not str(gib).strip():
        val = 1.5
    else:
        try:
            val = float(gib)
        except ValueError:
            logger.warning("ignoring non-numeric HUGPY_IMAGEGEN_GEN_CUSHION_GIB=%r; "
                           "using 1.5", gib)
            val = 1.5
    return int(max(0.0, val) * 2**30)


def _worker_ensure_imagegen_headroom(state: "WorkerState", model_key: str,
                                     need_bytes: "int | None" = None,
                                     job_id=None) -> dict:
    """Evict-to-fit BEFORE an in-process diffusers (text-to-image / image-to-image
    / studio frame) load or generation, so the pipeline can land on the GPU
    instead of OOMing behind an idle squatter. Registered by boot as the worker
    side of ``imagegen_runner.set_imagegen_headroom_hook`` (mirrors the comfy
    headroom hook). Reuses the ONE eviction policy (_evict_to_free_target).

    TARGET = the model's REAL footprint + a generation cushion (``need_bytes`` is
    the diffusers weights the runner priced from disk). Unlike ``_vram_evict_to_fit``
    this does NOT apply the placement-intent re-price — a ram-only designation
    projects 0 B on the GPU there and would evict nobody, which is exactly the
    2026-09-25 incident: sd-turbo derived ram-only, so nothing evicted the idle
    flux2-klein slot child (6.99 GiB) and the diffusers load CUDA-OOM'd. Here the
    worker reclaims its own contended card; the runner then decides GPU-vs-CPU
    from the RECLAIMED free VRAM (ram-only upgrades to GPU when it now fits, else
    falls back to CPU / an honest refusal). Best-effort; never blocks the gen.

    Unknown need -> the legacy comfy constant target (a sane 'clear some room'
    goal), so a model the runner couldn't size still triggers a bounded pass."""
    if need_bytes and need_bytes > 0:
        target = int(need_bytes) + _imagegen_gen_cushion_bytes()
    else:
        target = _comfy_target_free_bytes()
    # The in-process diffusers load lands on central's chosen card (HUGPY_MAIN_GPU),
    # so free THAT card, not a sibling.
    tele = _evict_to_free_target(state, model_key, target, exclude=model_key,
                                 job_id=job_id, target_dev=_target_device_index())
    tele["need_bytes"] = int(need_bytes) if need_bytes else None
    logger.info(
        "ensure-imagegen-headroom: model=%s need=%s target=%.2fGiB free_after=%s "
        "evicted=%s reached=%s",
        model_key, _human_bytes(need_bytes), target / 2**30,
        (_human_bytes(tele.get("free_after")) if tele.get("free_after") is not None
         else "unmeasurable"),
        tele.get("evicted"), tele.get("reached"))
    return tele


def _worker_ensure_comfy_headroom(state: "WorkerState", model_key: str,
                                  job_id=None) -> dict:
    """Evict on-demand managed models (LRU, via the SAME _evict_model mechanism
    the evict verb uses) until real free VRAM >= the comfy target, BEFORE a comfy
    gen commits (Fix B). Runs UNCONDITIONALLY per the operator directive ("evict
    down to free target vram always"); a no-op when already above target.

    Honest-degrade at every seam: no GPU / can't read free VRAM -> no-op (return
    early, never block); nothing left to evict but still short -> proceed anyway
    with a logged warning (the comfy gen is NEVER blocked/hung). Returns a small
    telemetry dict (used by the routine's own logging + tests). Best-effort — the
    caller (comfy_runner) swallows any exception, but this stays defensive too.

    THE TARGET IS THE CHECKPOINT'S OWN NEED (2026-09-22): weights on disk + the
    gen cushion, or the cushion alone when comfy already holds this checkpoint
    (_comfy_need_detail). A 2 GiB SD1.5 no longer evicts neighbours for 7 GiB
    it will never use, and a 12 GiB checkpoint no longer starts a gen with 7.
    Unknown file -> the legacy constant, byte-identical to before. The dispatch
    is then recorded in the ledger so the planner can see it by name."""
    try:
        detail = _comfy_need_detail(state, model_key)
    except Exception as exc:  # noqa: BLE001 — sizing never blocks the gen
        logger.warning("comfy need for %s unpriced (%s) — legacy target", model_key, exc)
        detail = {"checkpoint": None, "checkpoint_bytes": None, "held": False,
                  "need": None}
    target = _comfy_headroom_target(detail)
    # ComfyUI is pinned to ONE card at provision (HUGPY_COMFY_CUDA_DEVICE), so on a
    # multi-GPU box only evict residents on THAT card.
    _cd_env = (os.environ.get("HUGPY_COMFY_CUDA_DEVICE") or "").strip()
    try:
        _comfy_dev = int(_cd_env) if _cd_env else None
    except ValueError:
        _comfy_dev = None
    # Delegate to the ONE shared evict-to-fit loop (same candidate policy, evict
    # verb, gates, and /llm/evictions recording the diffusers path now uses too).
    tele = _evict_to_free_target(state, model_key, target, exclude=model_key,
                                 job_id=job_id, target_dev=_comfy_dev)
    if tele.get("free_before") is None and tele.get("free_after") is None:
        # No GPU / can't measure: byte-identical to today — do nothing.
        return {"target": target, "free_before": None, "free_after": None,
                "evicted": [], "reached": None, "note": "no GPU / unmeasurable",
                **_comfy_need_report(detail)}
    # The gen commits next (the caller POSTs the graph): comfy holds this
    # checkpoint from here, whatever the headroom outcome was.
    try:
        _comfy_ledger().note_dispatch(model_key, detail.get("checkpoint"),
                                      detail.get("checkpoint_bytes"))
    except Exception:  # noqa: BLE001 — bookkeeping never breaks the gen
        pass
    return {"target": target, "free_after": tele.get("free_after"),
            "evicted": tele.get("evicted") or [],
            "reached": tele.get("reached"), **_comfy_need_report(detail)}


def _comfy_need_report(detail: dict) -> dict:
    """The need detail as telemetry keys (omit-when-unknown for the file)."""
    out = {"need": detail.get("need"), "held": bool(detail.get("held"))}
    if detail.get("checkpoint"):
        out["checkpoint"] = detail["checkpoint"]
        out["checkpoint_bytes"] = detail.get("checkpoint_bytes")
    return out


# ── external lease claim (gpu_lease demand direction, 2026-08-11) ────────────
def _external_min_idle_s() -> float:
    """OPTIONAL extra damper on lease claims (HUGPY_EXTERNAL_MIN_IDLE_S,
    default 0 = off). Operator ruling 2026-08-11: eviction eligibility is
    "is it SERVING or not" — the in-flight gate inside _evict_model (never
    rip an actively-replying model) is the one true guard, and a model that
    is not serving may yield to a lease claim regardless of when it last
    served. Wall-clock idle thresholds are a proxy and stay off by default.
    The anti-thrash role the old 300s default played now lives client-side:
    gpu_lease backs off its claim cadence when its resumes keep getting
    displaced quickly (contention-derived, not hardcoded). Set this env only
    if a box needs a hard floor anyway."""
    raw = os.environ.get("HUGPY_EXTERNAL_MIN_IDLE_S")
    try:
        val = float(raw) if raw is not None and str(raw).strip() else 0.0
    except ValueError:
        logger.warning("ignoring non-numeric HUGPY_EXTERNAL_MIN_IDLE_S=%r; "
                       "using 0", raw)
        val = 0.0
    return max(0.0, val)


def _external_claim_headroom(state: "WorkerState", model_key: str,
                             target_bytes: int,
                             min_idle_s: "float | None" = None,
                             include_external: bool = False) -> dict:
    """Evict IDLE on-demand managed models (LRU, via the SAME _evict_model verb
    everything else uses, unforced) until free VRAM >= ``target_bytes``, so an
    external lease (gpu_lease batch job) can take the card. The mirror of
    _worker_ensure_comfy_headroom, with two deliberate differences:

    * IDLE GUARD: only models unused for >= min_idle_s are candidates. A comfy
      gen is a user waiting NOW, so it evicts anything evictable; a batch job
      is nobody waiting, so it only takes the card from models that have
      genuinely gone quiet. The unforced _evict_gate still protects in-flight
      generations on top of this.
    * NO honest-degrade-proceed: this never launches anything. It reports
      ``reached`` honestly and the SUPERVISOR decides (it polls again later) —
      a batch job that can't get the card simply stays paused, which is the
      whole contract.

    ``include_external`` (studio-render demand class, 2026-08-13): normally the
    OTHER external leases are never candidates here (two leases evicting each
    other is a ping-pong with no user behind either side; operator arbitrates
    that by hand). A studio render, though, IS a user waiting, so its claim
    passes include_external=True to make evictable peer leases (bluebook OCR)
    last-resort candidates — their supervisors park and resume afterwards. The
    claimant's own key is always excluded. A non-evictable peer is never a
    candidate either way (it is dropped by _comfy_headroom_candidates). If
    models alone don't reach target, an idle comfy is reclaimed last (TTL
    waived — same as every contention path)."""
    if min_idle_s is None:
        min_idle_s = _external_min_idle_s()
    fv = _free_vram_bytes()
    if fv is None:
        return {"target": target_bytes, "free_before": None, "free_after": None,
                "evicted": [], "skipped": [], "reached": False,
                "min_idle_s": min_idle_s, "note": "no GPU / unmeasurable"}
    free_before = fv
    evicted: list[str] = []
    skipped: list[dict] = []
    tried: set[str] = set()
    try:
        from hugpy_fleet.worker import external_residents as _extres
        all_ext = set(_extres.keys())
    except Exception:  # noqa: BLE001
        all_ext = set()
    # Peer leases this claim will NOT touch: every other external lease when
    # include_external is False; only the claimant itself when it is True.
    excluded_ext = ({model_key} if include_external
                    else (all_ext | {model_key}))
    # FEASIBILITY PRE-CHECK (2026-08-11, with min-idle now defaulting to 0):
    # if free + everything this claim could possibly reclaim still falls short
    # of target, evict NOTHING. Without this, a claim polling every 60s while
    # a busy model holds the card would strip the other idle residents as
    # pure collateral — reload cost for zero lease progress. Measured bytes
    # come from the same pid-registry/slot union the evict planner ranks; a
    # small optimism factor absorbs under-attribution.
    if fv < target_bytes:
        measured = {r["model_key"]: int(r.get("vram_bytes") or 0)
                    for r in _vram_residents(state)}
        reclaimable = sum(
            measured.get(mk, 0)
            for mk in _comfy_headroom_candidates(exclude=model_key)
            if mk not in excluded_ext)
        try:
            reclaimable += int(_comfy_process_vram() or 0)
        except Exception:  # noqa: BLE001
            pass
        if (fv + reclaimable) < int(target_bytes * 0.95):
            return {"target": target_bytes, "free_before": free_before,
                    "free_after": fv, "evicted": [], "skipped": [],
                    "reached": False, "min_idle_s": min_idle_s,
                    "note": (f"infeasible: free {fv/2**30:.1f}GiB + "
                             f"reclaimable {reclaimable/2**30:.1f}GiB < target "
                             f"{target_bytes/2**30:.1f}GiB — nothing evicted "
                             "(no pointless collateral)")}
    while fv < target_bytes:
        now = time.time()
        last = {}
        try:
            last = _dispatch_last_used()
        except Exception:  # noqa: BLE001 — no LRU clock -> everything reads idle
            pass
        cands = []
        for mk in _comfy_headroom_candidates(exclude=model_key):
            if mk in tried or mk in excluded_ext:
                continue
            idle = now - float(last.get(mk, 0.0))
            if idle < min_idle_s:
                tried.add(mk)
                skipped.append({"model_key": mk, "why":
                                f"used {idle:.0f}s ago (< min_idle "
                                f"{min_idle_s:.0f}s)"})
                continue
            cands.append(mk)
        if not cands:
            break
        victim = cands[0]
        tried.add(victim)
        try:
            res = _evict_model(state, victim, force=False)
        except Exception:  # noqa: BLE001 — one bad evict must not wedge the claim
            logger.warning("external-claim: evict of %s raised; skipping",
                           victim, exc_info=True)
            continue
        if res.get("evicted"):
            evicted.append(victim)
            logger.info("external-claim: evicted idle %s (%s) so lease %s can "
                        "take the card", victim, res.get("host_mode"), model_key)
        else:
            skipped.append({"model_key": victim,
                            "why": str(res.get("reason") or "not evicted")})
        fv = _free_vram_bytes()
        if fv is None:
            break
    # Models alone short of target: an idle comfy pays last (same clauses as
    # every contention path — a rendering comfy is never disturbed).
    if fv is not None and fv < target_bytes:
        freed = _comfy_reclaim_idle_vram(state, incoming_model=None,
                                         need_bytes=target_bytes - fv)
        if freed:
            evicted.append("comfy")
            fv = _free_vram_bytes()
    reached = (fv is not None and fv >= target_bytes)
    return {"target": target_bytes, "free_before": free_before,
            "free_after": fv, "evicted": evicted, "skipped": skipped,
            "reached": reached, "min_idle_s": min_idle_s}


def _prune_stale_residency(state: "WorkerState") -> None:
    """Tiers v3 lazy cleanup: residency overrides are ASSIGNMENT-scoped unless
    pinned. 🔒 static (and ⏲ on-demand) last while the model stays assigned;
    📌 pin makes the attribution permanent, so pinned overrides survive.

    Drops overrides for models absent from state.assigned_models (the list
    adopted from central's authoritative register/heartbeat response) unless
    pinned. Updates the LIVE settings and persists the file — no re-exec:
    _residency()/the sweep/the slot policy all read _RUNTIME_SETTINGS live."""
    args = getattr(state, "args", None)
    if args is None:                       # startup window before main() wires it
        return
    res = _RUNTIME_SETTINGS.get("residency") or {}
    if not res:
        return
    assigned = set(state.assigned_models)
    stale = [mk for mk in res if mk not in assigned and not _pinned(mk)]
    if not stale:
        return
    for mk in stale:
        logger.info("residency override %r for %s dropped — model unassigned "
                    "and not pinned (static ends at unassign)", res.get(mk), mk)
    settings = _load_settings(args)
    kept = {k: v for k, v in (settings.get("residency") or {}).items()
            if k not in stale}
    if kept:
        settings["residency"] = kept
    else:
        settings.pop("residency", None)
    _save_settings(args, settings)
    live = {k: v for k, v in res.items() if k not in stale}
    if live:
        _RUNTIME_SETTINGS["residency"] = live
    else:
        _RUNTIME_SETTINGS.pop("residency", None)


def _settings_path(args) -> str:
    return args.id_file + ".settings.json"


def _load_settings(args) -> dict:
    try:
        with open(_settings_path(args), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_settings(args, settings: dict) -> None:
    tmp = _settings_path(args) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=1)
    os.replace(tmp, _settings_path(args))


def _apply_settings_env(args) -> dict:
    """Project the settings file onto the env BEFORE anything reads it, so
    every existing consumer (managers.serve.slots._slot_count, …) sees the
    operator's console-set values — and unit drop-ins lose, loudly.

    Runs ONCE at boot (main() calls it before the slot supervisor / any reader)
    and is the ONLY projector — which is what makes the two restart lifecycles
    below work.

    ── Two restart lifecycles for the COMFY_URL / HUGPY_HOT_CACHE_ROOT sentinels ──
    Those two settings are projected onto real env vars that live code reads
    (COMFY_URL; managers/serve/hot_cache.py reads HUGPY_HOT_CACHE_ROOT). To let a
    later CLEAR revert to the true drop-in/env BASE instead of leaking the last
    projected value, the pre-projection base is captured once into a sentinel env
    (_COMFY_URL_BASE_ENV / _HOT_CACHE_ROOT_BASE_ENV). Its lifecycle depends on how
    the agent restarts (see the restart mechanism section):

      * STANDALONE (os.execv): the exec INHERITS os.environ, so the projected
        COMFY_URL and the sentinel both survive into the new image. The sentinel
        is ESSENTIAL here — without it the next boot would recapture the already-
        projected value as the "base" and a clear could never get back to the
        real drop-in/env base. This is the dance the sentinels were built for.

      * SYSTEMD (exit + respawn): the fresh process is started clean by systemd
        with the UNIT's environment, so COMFY_URL/HUGPY_HOT_CACHE_ROOT are back to
        their true base and NO sentinel is inherited. The sentinel is simply
        recaptured from that clean base on this boot — harmless and correct
        (base IS the env). So the sentinel is unnecessary in the systemd path but
        does no harm; the ``if _X not in os.environ`` guards below make both
        lifecycles converge to the same result.
    """
    settings = _load_settings(args)
    _RUNTIME_SETTINGS.clear()
    _RUNTIME_SETTINGS.update(settings)
    if "slot_count" in settings:
        env_was = os.environ.get("SLOT_COUNT")
        os.environ["SLOT_COUNT"] = str(int(settings["slot_count"]))
        _SETTINGS_SOURCE["slot_count"] = "settings"
        if env_was is not None and env_was != os.environ["SLOT_COUNT"]:
            logger.warning("settings override: SLOT_COUNT env/drop-in said %r but "
                           "the operator's runtime settings say %s — settings win",
                           env_was, settings["slot_count"])
    else:
        _SETTINGS_SOURCE["slot_count"] = (
            "env" if os.environ.get("SLOT_COUNT") not in (None, "") else "default")
    # Capture the pre-projection COMFY_URL (drop-in / env / none) ONCE into a
    # sentinel that survives os.execv, so a later clear reverts to the real base
    # instead of leaking the last projected value (execv inherits the live
    # environ; this function is the only projector, run once per boot).
    if _COMFY_URL_BASE_ENV not in os.environ:
        os.environ[_COMFY_URL_BASE_ENV] = os.environ.get("COMFY_URL", "")
    _base = os.environ.get(_COMFY_URL_BASE_ENV, "")
    if settings.get("comfy_url"):
        # Settings win over any env/unit-drop-in COMFY_URL, mirroring slot_count.
        os.environ["COMFY_URL"] = str(settings["comfy_url"])
        _SETTINGS_SOURCE["comfy_url"] = "settings"
        if _base and _base != os.environ["COMFY_URL"]:
            logger.warning("settings override: COMFY_URL env/drop-in said %r but "
                           "the operator's runtime settings say %r — settings win",
                           _base, settings["comfy_url"])
    elif _base:
        os.environ["COMFY_URL"] = _base           # revert to the drop-in/env base
        _SETTINGS_SOURCE["comfy_url"] = "env"
    else:
        os.environ.pop("COMFY_URL", None)         # no base -> 127.0.0.1:8188 default
        _SETTINGS_SOURCE["comfy_url"] = "default"
    # HOT-CACHE ROOT — per-worker attribution of the box-local NVMe LRU tier.
    # managers/serve/hot_cache.py reads HUGPY_HOT_CACHE_ROOT live, so projecting
    # the setting ONTO that env is the whole mechanism (the tier code is
    # untouched): resolution order becomes settings > env base > unset. Same
    # base-sentinel dance as COMFY_URL so a later clear reverts to the true
    # drop-in/env base instead of leaking the last projected value across execv.
    if _HOT_CACHE_ROOT_BASE_ENV not in os.environ:
        os.environ[_HOT_CACHE_ROOT_BASE_ENV] = os.environ.get(_ENV_HOT_CACHE_ROOT, "")
    _hc_base = os.environ.get(_HOT_CACHE_ROOT_BASE_ENV, "")
    if settings.get("hot_cache_root"):
        os.environ[_ENV_HOT_CACHE_ROOT] = str(settings["hot_cache_root"])
        _SETTINGS_SOURCE["hot_cache_root"] = "settings"
        if _hc_base and _hc_base != os.environ[_ENV_HOT_CACHE_ROOT]:
            logger.warning("settings override: HUGPY_HOT_CACHE_ROOT env/drop-in "
                           "said %r but the operator's runtime settings say %r — "
                           "settings win", _hc_base, settings["hot_cache_root"])
        # Best-effort materialization at apply time. NEVER fatal: hot_cache.
        # enabled() re-checks the root live on every use() and disables the tier
        # gracefully if it is (or becomes) uncreatable, so a not-yet-mounted root
        # never breaks the boot — the tier simply activates once it appears.
        try:
            os.makedirs(os.environ[_ENV_HOT_CACHE_ROOT], exist_ok=True)
        except OSError as exc:
            logger.warning("hot_cache_root %r not creatable yet (%s) — the hot "
                           "tier stays off until the path exists",
                           os.environ[_ENV_HOT_CACHE_ROOT], exc)
    elif _hc_base:
        os.environ[_ENV_HOT_CACHE_ROOT] = _hc_base    # revert to the drop-in/env base
        _SETTINGS_SOURCE["hot_cache_root"] = "env"
    else:
        os.environ.pop(_ENV_HOT_CACHE_ROOT, None)     # no base -> tier off (unset)
        _SETTINGS_SOURCE["hot_cache_root"] = "default"
    # ── EVICTION POLICY (2026-07-25) ────────────────────────────────────────
    # Projected onto the env that _evict_least_reaping already reads, so that
    # reader stays unchanged and the precedence (settings > drop-in env > module
    # default) falls out of the same mechanism slot_count uses. Same
    # base-sentinel dance as COMFY_URL: a later CLEAR must revert to the true
    # drop-in base, not leak the last projected value across an execv.
    #
    # NOTE the `is not None` guard. This is the ONE place a projected value can
    # legitimately be falsy-but-set: evict_least_reaping=False IS the
    # greedy-walk mode. A truthiness test here (`if settings.get(...)`) would
    # silently drop exactly the value the operator most needs to reach.
    # (evict_min_residency_s was projected here too until it was retired
    # 2026-07-27 — no time-based eviction veto exists now.)
    for _key, _env in ((("evict_least_reaping"), _ENV_LEAST_REAPING),):
        _base_env = f"_HUGPY_BASE_{_env}"
        if _base_env not in os.environ:
            os.environ[_base_env] = os.environ.get(_env, "")
        _base = os.environ.get(_base_env, "")
        _val = settings.get(_key)
        if _val is not None:
            # bools -> "1"/"0" (what _evict_least_reaping parses); numbers via
            # str(). Never repr() — a Python bool is "True", not a value the
            # env reader's ("0","false","no","off") check would treat as false.
            os.environ[_env] = ("1" if _val is True else
                                "0" if _val is False else str(_val))
            _SETTINGS_SOURCE[_key] = "settings"
            if _base and _base != os.environ[_env]:
                logger.warning("settings override: %s env/drop-in said %r but the "
                               "operator's runtime settings say %r — settings win",
                               _env, _base, os.environ[_env])
        elif _base:
            os.environ[_env] = _base                  # revert to drop-in/env base
            _SETTINGS_SOURCE[_key] = "env"
        else:
            os.environ.pop(_env, None)                # no base -> module default
            _SETTINGS_SOURCE[_key] = "default"
    return settings


def _effective_config() -> dict:
    """What this agent is ACTUALLY running with (for the heartbeat)."""
    try:
        from hugpy_engine.serve.slots import _slot_count
        n = _slot_count()
    except Exception:
        n = None
    # Idle reclamation is OPT-IN (doctrine 2026-07-11): report the TTL as null
    # when the operator hasn't set it, so the console shows the honest "off"
    # (contention-only residency) instead of a phantom 900s clock.
    _ttl_set = "on_demand_ttl_s" in _RUNTIME_SETTINGS
    out = {"slot_count": n,
           "slot_count_source": _SETTINGS_SOURCE.get("slot_count", "default"),
           "on_demand_ttl_s": (int(_RUNTIME_SETTINGS["on_demand_ttl_s"])
                               if _ttl_set else None),
           "on_demand_ttl_s_source": "settings" if _ttl_set else "default"}
    if _RUNTIME_SETTINGS.get("residency"):
        out["residency"] = dict(_RUNTIME_SETTINGS["residency"])
    if _RUNTIME_SETTINGS.get("pinned"):
        out["pinned"] = dict(_RUNTIME_SETTINGS["pinned"])
    if _RUNTIME_SETTINGS.get("ctx_pct"):
        # Per-model context allocation (slice 11) — rides the heartbeat config map
        # to central so the console can show the ctx % the worker will serve.
        out["ctx_pct"] = dict(_RUNTIME_SETTINGS["ctx_pct"])
    out["comfy_url"] = (os.environ.get("COMFY_URL")
                        or "http://127.0.0.1:8188").rstrip("/")
    out["comfy_url_source"] = _SETTINGS_SOURCE.get("comfy_url", "default")
    # HOT-CACHE ROOT: the effective projected root ("" == unset == tier off, the
    # honest reading — hot_cache has no fallback root, unlike comfy_url) + where
    # the value came from, so a /llm/workers row carries the truth exactly as it
    # does for slot_count. This ATTRIBUTES the root per worker; the tier itself is
    # still an automatic LRU cache and the shared store is still the source of truth.
    out["hot_cache_root"] = (os.environ.get(_ENV_HOT_CACHE_ROOT) or "").strip()
    out["hot_cache_root_source"] = _SETTINGS_SOURCE.get("hot_cache_root", "default")
    # Env-profiles (stage 1): the per-profile materialization state
    # (ready|materializing|error) + the model->profile attribution map, so a
    # /llm/workers row carries the truth — central routes a profiled model only
    # once its profile reads ready. Present only when profiles are in play
    # (mirrors residency/pinned). Defensive import: never break a beat.
    if _RUNTIME_SETTINGS.get("profiles"):
        try:
            from hugpy_engine.serve import profiles as _profiles
            out["profiles"] = _profiles.report(_RUNTIME_SETTINGS["profiles"])
        except Exception:  # noqa: BLE001 — heartbeat truth is best-effort
            out["profiles"] = {}
    if _RUNTIME_SETTINGS.get("model_profiles"):
        out["model_profiles"] = dict(_RUNTIME_SETTINGS["model_profiles"])
    # ── EVICTION POLICY: report what is ACTUALLY IN FORCE, not what was typed.
    # Read back through the same reader the eviction path uses, so the console
    # can never show a value the planner disagrees with — including the case
    # where a setting was cleared and a drop-in env took over. Reported
    # unconditionally (never omitted when falsy): 0 / False are meaningful
    # states here, and hiding them would show the operator the default while
    # the escape hatch was armed.
    out["evict_least_reaping"] = _evict_least_reaping()
    out["evict_least_reaping_source"] = _SETTINGS_SOURCE.get(
        "evict_least_reaping", "default")
    return out


def _path_device(path: str):
    """The device id `path` lives on, or None if it can't be stat'd — how
    _disk_status decides whether two roots are the same volume (one entry) or
    two tiers worth reporting separately."""
    try:
        return os.stat(path).st_dev
    except OSError:
        return None


def _disk_status() -> dict:
    """Free/total bytes of the volume holding this worker's MODEL ROOT — the
    disk a designation's pull lands on. Central's assign/load preflight uses
    this so a model that won't fit is refused early (409), not mid-pull.

    k60 (operator, 2026-07-31): the row must report the drive the MODEL STORE
    ROOT actually lives on — resolved the same way the reaper/heartbeat resolve
    it (``_models_store_root``) — not DEFAULT_ROOT, which on a two-tier box (hot
    NVMe store + a mounted shared catalog) can be the OTHER volume entirely. The
    first entry is therefore never mislabeled. When DEFAULT_ROOT sits on a
    different device it is reported as a SECOND tier instead of replacing the
    first, each tagged with whether it is the shared/central catalog.
    """
    try:
        import shutil
        from hugpy_platform.constants import DEFAULT_ROOT
        base = (str(DEFAULT_ROOT) if os.path.isdir(str(DEFAULT_ROOT))
                else os.path.expanduser("~"))
        model_root = _models_store_root() or base
        tiers: list[dict] = []
        seen_devices: set = set()
        for label, path in (("model root", model_root), ("default root", base)):
            if not path or not os.path.isdir(path):
                continue
            dev = _path_device(path)
            if dev is None:
                continue
            if dev in seen_devices:
                continue                       # same volume — one entry is enough
            seen_devices.add(dev)
            u = shutil.disk_usage(path)
            tiers.append({"label": label, "root": path, "free_bytes": u.free,
                          "total_bytes": u.total,
                          "shared": _path_on_shared_store(path)})
        if not tiers:
            return {}
        primary = tiers[0]
        return {"root": primary["root"], "free_bytes": primary["free_bytes"],
                "total_bytes": primary["total_bytes"],
                "shared": primary["shared"], "tiers": tiers}
    except Exception:  # noqa: BLE001
        return {}


# ── ComfyUI presence (slice A of the comfy engine) ──────────────────────────
# The operator installs ComfyUI on the box (own service/venv); the agent
# ADOPTS it: probe the local instance and advertise `comfy` in the heartbeat
# so central can route comfy-templated work here (slice B) and the console
# shows the capability. COMFY_URL overrides the default local port.

_COMFY_CACHE: dict = {"at": 0.0, "value": {"available": False}}


def _comfy_status() -> dict:
    """Probe the local ComfyUI (60s cache): {"available", "url", "version"?,
    "checkpoints"?, "id_lock", "vram_bytes"}. ``vram_bytes`` is ComfyUI's REAL GPU
    footprint from nvidia-smi (per-process), or null — never on-disk checkpoint
    bytes (the 0.1.137 guard). ``id_lock`` is whether this comfy can do
    identity-locked STILLs (the IPAdapter node pack is installed), so central's
    routing gate + the console can see which boxes can do it."""
    now = time.time()
    if now - _COMFY_CACHE["at"] < 60.0:
        out = _COMFY_CACHE["value"]
    else:
        url = (os.environ.get("COMFY_URL") or "http://127.0.0.1:8188").rstrip("/")
        out = {"available": False, "url": url, "id_lock": False}
        try:
            import httpx
            r = httpx.get(url + "/system_stats", timeout=2.0)
            if r.status_code == 200:
                out["available"] = True
                try:
                    sysinfo = (r.json() or {}).get("system") or {}
                    if sysinfo.get("comfyui_version"):
                        out["version"] = sysinfo["comfyui_version"]
                except Exception:  # noqa: BLE001 — version is decoration
                    pass
                # Advertise loadable checkpoints — registry rows' `filename`
                # designations come from this list (slice B).
                try:
                    oi = httpx.get(url + "/object_info/CheckpointLoaderSimple",
                                   timeout=3.0).json()
                    ckpts = oi["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
                    if isinstance(ckpts, list):
                        out["checkpoints"] = ckpts[:50]
                except Exception:  # noqa: BLE001 — list is best-effort
                    pass
                # ID-LOCK capability: probe the SAME object_info API for the
                # IPAdapter node classes, via the comfy runner's own detector so
                # the node-class contract lives in ONE place (never forks from the
                # request-time gate). Rides this 60s presence cache.
                try:
                    from hugpy_media.comfy.comfy_runner import comfy_has_ipadapter
                    out["id_lock"] = comfy_has_ipadapter(url)
                except Exception:  # noqa: BLE001 — probe/import miss: not capable
                    out["id_lock"] = False
        except Exception:  # noqa: BLE001 — not installed / not running
            pass
        _COMFY_CACHE.update(at=now, value=out)
    # Refresh VRAM every call (cheap — reuses the heartbeat-cached nvidia-smi
    # snapshot), so it isn't frozen for the 60s presence-cache window. Only when
    # ComfyUI is actually up; null otherwise.
    out["vram_bytes"] = _comfy_process_vram() if out.get("available") else None
    # What comfy holds BY NAME (the ledger): the console's comfy card shows the
    # checkpoints, not just one lump. Only when comfy really holds VRAM.
    try:
        out["resident"] = (_comfy_ledger().residents(process_vram_bytes=out["vram_bytes"])
                           if out["vram_bytes"] else [])
    except Exception:  # noqa: BLE001 — telemetry only
        out["resident"] = []
    # ── worker-managed comfy lifecycle (2026-09-24) ──────────────────────────
    # The worker may OWN comfy's process (HUGPY_COMFY_LAUNCH). Advertise the
    # launcher + a running|stopped|starting|failed state and a managed flag, and
    # — CRUCIALLY — keep advertising the checkpoints this box CAN serve even while
    # comfy is STOPPED but startable, so central can still place a comfy job here
    # (the worker starts comfy on demand). Additive fields only; a probe-only /
    # external box reports managed=False and is otherwise unchanged.
    try:
        mgr = _comfy_manager(None)
        running = bool(out.get("available"))
        out["managed"] = mgr.managed
        out["launcher"] = mgr.mode
        out["state"] = mgr.state(running=running)
        if mgr.spec_error:
            out["launcher_error"] = mgr.spec_error
        if mgr.managed and not running and not out.get("checkpoints"):
            disk_ckpts = mgr.checkpoints_on_disk()
            if disk_ckpts:
                out["checkpoints"] = disk_ckpts
                out["checkpoints_source"] = "disk (comfy stopped; managed startable)"
    except Exception:  # noqa: BLE001 — manager status is additive telemetry
        pass
    return out


def _slot_occupants(strict: bool = False) -> set:
    """Model keys currently seated in this worker's slot pool (empty set when
    slots are disabled/unreachable — callers treat unknown as unoccupied).

    ``strict=True`` re-raises instead of swallowing a probe failure, so a caller
    that must FAIL CLOSED (the reaper) can refuse to delete when it cannot prove
    a model isn't a live slot occupant. The default stays fail-open for telemetry
    callers (heartbeat/survey), where an empty set is harmless."""
    try:
        from hugpy_engine.serve.slots import SlotPool, slots_enabled
        if not slots_enabled():
            return set()
        return {s.get("model_key") for s in SlotPool().statuses()
                if s.get("model_key")}
    except Exception as exc:  # noqa: BLE001 — telemetry, never fatal
        logger.warning("slot occupancy lookup failed: %s", exc)
        if strict:
            raise
        return set()


def _residency_sweep_once(started_at: float) -> None:
    """One pass of the idle TTL sweep — OPT-IN since 2026-07-11 (factored out of
    the loop so it's testable).

    DOCTRINE (operator-locked 2026-07-11): keep models hot. An on-demand model
    stays resident until a NEW load needs its memory — then the LRU on-demand
    resident yields (dispatch.ensure_headroom_for_load). That CONTENTION trigger,
    not a clock, is the default. This idle sweep is the OPT-IN reclamation path:
    it runs ONLY when the operator has explicitly set on_demand_ttl_s (present in
    _RUNTIME_SETTINGS). Absent -> return immediately; contention alone governs
    residency, so a model that just answered a chat is NOT torn down minutes
    later (the drift this correction fixes).

    When enabled, the sweep applies ONLY to IN-PROCESS residents: any non-static
    one idle longer than on_demand_ttl_s is evicted (dispatch.evict cascades to
    the llama singleton — RAM/VRAM actually frees). SLOT occupants are EXEMPT
    (slots stay filled — slice 9 — and a seat changes hands only via LRU
    promotion or explicit unload). Static never yields anywhere."""
    if "on_demand_ttl_s" not in _RUNTIME_SETTINGS:
        return                              # idle reclamation off -> contention only
    ttl = int(_RUNTIME_SETTINGS["on_demand_ttl_s"])
    from hugpy_engine.dispatch.dispatch import last_used_snapshot, evict
    seated = _slot_occupants()
    last_used = last_used_snapshot()
    now = time.time()
    for mk in loaded_model_keys():
        if _residency(mk) == "static" or mk in seated:
            continue
        idle = now - last_used.get(mk, started_at)
        if idle > ttl:
            logger.info("residency sweep: evicting %s (on-demand in-process, "
                        "idle %.0fs > ttl %ds)", mk, idle, ttl)
            try:
                evict(mk)
            except Exception as exc:  # noqa: BLE001
                logger.warning("residency evict of %s failed: %s", mk, exc)


def _raw_free_vram_bytes() -> "int | None":
    """HEADROOM-SWEEP-ACCOUNTED-20260910: RAW device free VRAM (no operator reserve subtracted) — the same
    probe spill.free_vram_bytes wraps, minus its clamp. None when unmeasurable."""
    try:
        from hugpy_engine.spill import _env_int
        from hugpy_platform.hardware import free_vram_bytes as _raw
        v = _raw(_env_int("HUGPY_MAIN_GPU") or 0)
        return None if v is None else int(v)
    except Exception:  # noqa: BLE001
        return None


def _vram_untracked_pressure_bytes(state: "WorkerState", total: int) -> "int | None":
    """HEADROOM-SWEEP-ACCOUNTED-20260910: VRAM in use that the worker's own admitted MANAGED residents do
    not account for: raw used minus the summed measured footprint of every
    non-comfy resident. That remainder is what an out-of-band grower (ComfyUI,
    a desktop session, a stray process) holds — the only pressure the sweep has
    a purpose in relieving. None when it cannot be attributed honestly (raw free
    unreadable, or a managed resident with no measured footprint)."""
    raw_free = _raw_free_vram_bytes()
    if raw_free is None or not total:
        return None
    used = max(0, int(total) - int(raw_free))
    accounted = 0
    for r in _vram_residents(state):
        if str(r.get("host_mode")) == "comfy":
            continue                         # the out-of-band grower itself
        vb = int(r.get("vram_bytes") or 0)
        if vb <= 0:
            return None                      # unmeasured -> can't attribute
        accounted += vb
    return max(0, used - accounted)


_SWEEP_SKIP_KEY: "tuple | None" = None


def _note_sweep_skip(state: "WorkerState", total: int, fv: int, reserve: int,
                     untracked: "int | None") -> None:
    """HEADROOM-SWEEP-ACCOUNTED-20260910: record ONE skip per resident set (not per 60 s beat): a log line
    and a headroom.start/done telemetry pair with outcome "skipped-accounted"
    (attributed, all ours) or "skipped-unattributable"."""
    global _SWEEP_SKIP_KEY
    try:
        residents = tuple(sorted(r["model_key"] for r in _vram_residents(state)))
    except Exception:  # noqa: BLE001
        residents = ()
    key = (residents, untracked is None)
    if key == _SWEEP_SKIP_KEY:
        return
    _SWEEP_SKIP_KEY = key
    if untracked is None:
        outcome = "skipped-unattributable"
        note = ("over the ceiling but the pressure cannot be attributed (raw free "
                "or a resident footprint unmeasured) — no purpose can be shown, "
                "evicting nothing")
        logger.warning("VRAM headroom sweep: %s (%s free of %s; residents %s)",
                       note, _human_bytes(fv), _human_bytes(total),
                       ", ".join(residents) or "none")
    else:
        outcome = "skipped-accounted"
        note = ("over the ceiling but the card is full of admitted residents "
                "(untracked %s < reserve %s) — nothing is waiting for the room, "
                "evicting nothing" % (_human_bytes(untracked), _human_bytes(reserve)))
        logger.info("VRAM headroom sweep: %s (%s free of %s; residents %s)",
                    note, _human_bytes(fv), _human_bytes(total),
                    ", ".join(residents) or "none")
    scope = _evt.run_scope() if _evt is not None else None
    if scope is not None:
        scope.__enter__()
    try:
        _evt_emit("headroom.start", trigger="sweep", incoming_model=None,
                  free_bytes=fv, total_bytes=total, reserve_bytes=reserve)
        _evt_emit("headroom.done", trigger="sweep", evicted=[], outcome=outcome,
                  untracked_bytes=untracked, note=note)
    finally:
        if scope is not None:
            try:
                scope.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass


def _vram_headroom_sweep(state: "WorkerState") -> None:
    """90% HEADROOM TRIGGER (slice 10 addendum). Admission-time evict-to-fit only
    fires when a NEW load arrives. The second incident had NO new load: ComfyUI
    grew out-of-band while an IDLE non-grower slot child squatted, and the card
    reached 100% and DEADLOCKED — the keeper had to /evict by hand. This closes
    that: on every residency beat, if the card is at/over the ceiling (free VRAM
    below the (1 - ceiling) reserve), evict the coldest EVICTABLE idle resident
    (same protection rules — static / actively-replying / queued-ahead / comfy are
    never touched) to claw back to headroom. No new timer: it rides the existing
    60s residency loop. A no-op when unmeasurable or already under the ceiling.

    Deliberately evicts AT MOST ONE resident per beat: a single reclaim (21G here)
    is enough to break a deadlock, and one-per-beat avoids over-evicting a box
    that's merely near the line — the next beat re-checks and takes another only
    if still pressured."""
    total = _total_vram_bytes()
    if not total:
        return
    fv = _free_vram_bytes()
    if fv is None:
        return
    # PRESSURE, not admission: still the (1 - ceiling) fraction of the card, and
    # deliberately NOT _vram_ceiling_reserve_bytes — see
    # _vram_pressure_reserve_bytes for why unifying them would retire this
    # deadlock-breaker. With an explicit HUGPY_VRAM_CEILING_FRAC they agree.
    reserve = _vram_pressure_reserve_bytes(total)
    if fv >= reserve:
        return                               # under the ceiling — nothing to do
    # HEADROOM-SWEEP-ACCOUNTED-20260910: over the ceiling — but is the pressure OURS? Evicting an admitted
    # resident into an empty card relieves nothing (admission already judged it
    # to fit; nothing is waiting; the next call just reloads it — the flux2-klein
    # 8 GB incident). Only pressure the worker did NOT admit — at least the
    # reserve's worth — is a purpose for this sweep. Unattributable = no purpose
    # shown = evict nothing.
    _untracked = _vram_untracked_pressure_bytes(state, total)
    if _untracked is None or _untracked < reserve:
        _note_sweep_skip(state, total, fv, reserve, _untracked)
        return
    global _SWEEP_SKIP_KEY
    _SWEEP_SKIP_KEY = None                   # pressure is real again: re-note next skip
    # TELEMETRY: this pass has no incoming model, so it opens its OWN run scope
    # with trigger="sweep". Opened only past the under-ceiling early return, so a
    # quiet box emits nothing at all on its 60s beat.
    _sweep_scope = _evt.run_scope() if _evt is not None else None
    if _sweep_scope is not None:
        _sweep_scope.__enter__()
    try:
        _evt_emit("headroom.start", trigger="sweep", incoming_model=None,
                  free_bytes=fv, total_bytes=total, reserve_bytes=reserve)
        _vram_headroom_sweep_body(state, total, fv)
    finally:
        if _sweep_scope is not None:
            try:
                _sweep_scope.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass


def _vram_headroom_sweep_body(state: "WorkerState", total: int, fv: int) -> None:
    """The sweep's decision + eviction. Split out only so the telemetry run
    scope above can wrap it; the logic is unchanged."""
    # ── stage 0 (k54): an IDLE comfy pays before any managed model does ──────
    # The card is over the ceiling. If part of what is holding it is a comfy
    # process with an empty queue and no registered call, that is dead weight —
    # reclaiming it costs nothing anyone is using, while evicting a managed
    # resident costs a reload. So the idle squatter yields FIRST; if that alone
    # gets the card back under the pressure reserve, no model is touched at all.
    # The TTL is waived here (contention beats idleness); clauses 1-3 still bind,
    # so a comfy that is actually rendering is never disturbed.
    if _comfy_reclaim_idle_vram(state, incoming_model=None):
        fv_after = _free_vram_bytes()
        if fv_after is not None and fv_after >= _vram_pressure_reserve_bytes(total):
            _evt_emit("headroom.done", trigger="sweep", evicted=["comfy"],
                      outcome="fit",
                      note="idle comfy VRAM reclaimed — no model evicted")
            return
        if fv_after is not None:
            fv = fv_after
    # Over the ceiling with no load driving admission. Evict the coldest EVICTABLE
    # idle resident, applying the SAME protection rules as _vram_evict_to_fit.
    busy_slots = _busy_slot_models()
    try:
        from hugpy_engine.dispatch.dispatch import last_used_snapshot as _lus
        lru = _lus()
    except Exception:  # noqa: BLE001
        lru = {}
    residents = _vram_residents(state)
    cands = []
    for r in residents:
        mk = r["model_key"]
        _tier = _telemetry_tier(r.get("host_mode"))
        if str(r.get("host_mode")) == "comfy":
            _evt_emit("candidate.skip", model_key=mk, tier=_tier,
                      reason="comfy (own headroom path)")
            continue                         # comfy has its own path; never here
        if str(r.get("host_mode")) == "external":
            # An external lease (gpu_lease batch job) legitimately holds most
            # of the card WHENEVER it runs — that is not the out-of-band-growth
            # deadlock this sweep exists to break, and sweeping it would put
            # the job in a permanent 60s pause/resume thrash with no demand
            # behind it. Real demand still evicts it: admission evict-to-fit,
            # the comfy headroom path, external claims, and explicit /ops/evict
            # all reach the external branch of _evict_model.
            _evt_emit("candidate.skip", model_key=mk, tier=_tier,
                      reason="external lease (demand-evictable only)")
            continue
        if _residency(mk) == "static":
            _evt_emit("candidate.skip", model_key=mk, tier=_tier,
                      reason="static (locked residency)")
            continue
        if _actively_replying(mk, busy_slots):
            _evt_emit("candidate.skip", model_key=mk, tier=_tier,
                      reason="actively replying (in-flight/busy)")
            continue
        # No `queued_ahead` here — there is no subject load this pass; a resident
        # with in-flight work is already protected by _actively_replying above.
        cands.append(r)
    if not cands:
        _evt_emit("headroom.done", trigger="sweep", evicted=[],
                  outcome="proceeded-unfit",
                  note="over the ceiling but every resident is protected")
        logger.warning("VRAM headroom: card at/over the %.0f%% ceiling (%s free of "
                       "%s) but nothing evictable — every resident is static or "
                       "actively replying; leaving it (autofit/degrade)",
                       _vram_ceiling_frac() * 100, _human_bytes(fv),
                       _human_bytes(total))
        return
    cands.sort(key=lambda r: (lru.get(r["model_key"], 0.0),
                              -int(r.get("vram_bytes") or 0)))
    victim = cands[0]["model_key"]
    logger.info("VRAM headroom: card at/over the %.0f%% ceiling (%s free) with no "
                "load driving admission — evicting coldest idle resident %s "
                "(operator addendum: no human is the eviction policy)",
                _vram_ceiling_frac() * 100, _human_bytes(fv), victim)
    _v_tier = _telemetry_tier(cands[0].get("host_mode"))
    _evt_emit("evict.start", model_key=victim, tier=_v_tier, trigger="sweep")
    _v_t0 = time.time()
    res = _evict_model(state, victim)
    if res.get("evicted"):
        _note_vram_eviction(victim, "headroom-sweep", res.get("vram_freed"),
                            res.get("host_mode") or "")
        _evt_emit("evict.done", model_key=victim,
                  tier=_telemetry_tier(res.get("host_mode") or _v_tier),
                  freed_bytes=res.get("vram_freed"),
                  duration_ms=int((time.time() - _v_t0) * 1000))
        _trim_host_ram()
        _evt_emit("reclaim.done")
        _evt_emit("headroom.done", trigger="sweep", evicted=[victim],
                  outcome="fit")
    else:
        _evt_emit("evict.fail", model_key=victim, tier=_v_tier,
                  duration_ms=int((time.time() - _v_t0) * 1000),
                  error=str(res.get("reason") or "eviction freed nothing"))
        _evt_emit("headroom.done", trigger="sweep", evicted=[],
                  outcome="proceeded-unfit")


def _fill_empty_slots(state: "WorkerState") -> None:
    """Compatibility no-op for the retired background slot filler.

    Assignment, heartbeat, provisioning, and maintenance must never create a
    VRAM resident. Keeping the symbol lets older integrations upgrade without
    crashing while preserving the load-only-on-call contract.
    """
    return None


def _residency_sweep_loop(state: "WorkerState") -> None:
    """Residency maintenance every 60s: enforce the 90% VRAM headroom (slice 10
    addendum), run the idle-comfy watchdog, then run the idle TTL sweep — which is
    a no-op unless the operator opted into on_demand_ttl_s (contention governs
    residency by default; see _residency_sweep_once + ensure_headroom_for_load).
    This loop only EVICTS/reclaims; it never loads a model (a model loads only when
    a request for it arrives)."""
    started_at = time.time()
    while True:
        time.sleep(60.0)
        try:
            _vram_headroom_sweep(state)
        except Exception as exc:  # noqa: BLE001 — the loop must never die
            logger.warning("VRAM headroom sweep failed: %s", exc)
        try:
            # k54: the idle-comfy watchdog. No new timer — it rides this same
            # 60s beat (minimize-loading doctrine: a timer is wrong by default,
            # and this one is a debounce on an existing loop, not a schedule).
            # Silent on every box where comfy holds nothing.
            _comfy_watchdog(state).tick(free_vram_bytes=_free_vram_bytes(),
                                        total_vram_bytes=_total_vram_bytes())
        except Exception as exc:  # noqa: BLE001 — the loop must never die
            logger.warning("comfy idle watchdog pass failed: %s", exc)
        try:
            # 2026-09-24: the worker owns comfy's lifecycle. After the /free
            # debounce reclaims the weights, a fully idle MANAGED comfy is STOPPED
            # (a longer window) so it releases even its bare CUDA context. Rides
            # the same beat; a no-op on the 'external' launcher and every box
            # where comfy is not running. Never stops mid-render (clauses 2-3).
            _mgr = _comfy_manager(state)
            _comfy_watchdog(state).stop_tick(managed=_mgr.managed,
                                             running=_mgr.is_running())
        except Exception as exc:  # noqa: BLE001 — the loop must never die
            logger.warning("comfy idle-stop pass failed: %s", exc)
        try:
            _residency_sweep_once(started_at)
        except Exception as exc:  # noqa: BLE001 — the loop must never die
            logger.warning("residency sweep iteration failed: %s", exc)


def _load_update_state(args) -> dict:
    try:
        with open(_update_state_path(args), "r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (OSError, ValueError):
        return {}


def _save_update_state(args, state: dict) -> None:
    try:
        with open(_update_state_path(args), "w", encoding="utf-8") as fh:
            json.dump(state, fh)
    except OSError:
        pass


# ── Lockstep convergence: the pip command both converge paths run ───────────
# Central's constraints file lives under its ``/api`` mount, same prefix rule
# as every other worker endpoint (see ``CentralClient``).
_CONSTRAINTS_PATH = "/api/llm/workers/constraints.txt"
_CONSTRAINTS_FETCH_TIMEOUT_S = 20.0


def _constraints_url(args, hint: str | None = None) -> str | None:
    """Where to fetch the lockstep pins: the reply's ``constraints_url`` when
    central sent one, else derived from this worker's central base URL."""
    hint = str(hint or "").strip()
    if hint:
        return hint
    central = str(getattr(args, "central", None) or "").strip()
    if not central:
        return None
    return central.rstrip("/") + _CONSTRAINTS_PATH


def _fetch_constraints(url: str, timeout: float = _CONSTRAINTS_FETCH_TIMEOUT_S) -> list[str]:
    """GET the constraints file; the non-blank, non-comment lines.

    A 204 (central pins no version) or an empty body yields ``[]`` — the caller
    treats that exactly like a failed fetch. Network/HTTP errors propagate."""
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if getattr(resp, "status", 200) == 204:
            return []
        body = resp.read().decode("utf-8", "replace")
    lines = []
    for raw in body.splitlines():
        ln = raw.strip()
        if ln and not ln.startswith("#"):
            lines.append(ln)
    return lines


def _normalize_dist(name: str) -> str:
    """PEP 503 normalized distribution name."""
    import re as _re
    return _re.sub(r"[-_.]+", "-", str(name or "")).lower()


def _installed_lockstep_siblings(constraint_lines: list[str], pkg_name: str) -> list[str]:
    """Names from the constraints that are INSTALLED here, other than ``pkg_name``.

    A constraints file only pins what pip already resolves — it never causes an
    install. To converge the WHOLE lockstep set we name every installed sibling
    explicitly (``hugpy-media==X``), so a distribution outside the tracked
    package's dependency closure (e.g. a media/video extra) moves too. Absent
    distributions are left absent: the install profile is the operator's."""
    from importlib import metadata
    want = _normalize_dist(pkg_name)
    out: list[str] = []
    for ln in constraint_lines:
        name = ln.split("==", 1)[0].strip()
        if not name or "==" not in ln or _normalize_dist(name) == want:
            continue
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        except Exception:  # noqa: BLE001 — unreadable dist-info: don't pin it
            continue
        out.append(name)
    return out


def _insecure_index_host(url: str | None) -> str | None:
    """``host[:port]`` of a plain-http index URL on a non-loopback host (what pip
    needs as ``--trusted-host``), else None."""
    if not url:
        return None
    try:
        from urllib.parse import urlsplit
        parts = urlsplit(url)
    except Exception:  # noqa: BLE001
        return None
    if parts.scheme != "http" or not parts.hostname:
        return None
    if parts.hostname in ("127.0.0.1", "localhost", "::1"):
        return None
    return parts.netloc


def _pip_converge_command(args, target: str, constraints_path: str | None,
                          extra_specs: "list[str] | tuple[str, ...]" = (),
                          extra_index: str | None = None) -> list[str]:
    """The pip command line that converges this worker to ``target``.

    Used by BOTH the heartbeat self-update and ``/ops/update`` so the two paths
    can never drift apart.

    ``constraints_path`` set → the constrained (lockstep) form::

        pip install -U --upgrade-strategy only-if-needed -c <constraints>
            [--index-url <pkg_index>] [--extra-index-url <extra_index>]
            <pkg_name>==<target> [<sibling>==<target> ...]

    ``extra_index`` is central's own index as advertised in the reply
    (``pkg_index_url``): it joins PyPI as a SECOND source rather than replacing
    it, so the pinned hugpy-* wheels resolve from central when they are only
    published there while third-party dependencies still come from PyPI. An
    explicit ``--pkg-index`` (the WG-only worker with no egress) keeps its
    ``--index-url`` semantics and ignores a hint equal to it.

    Dependencies are resolved under the pins, so every workspace sibling in the
    dependency closure moves to ``target`` with the tracked package; the
    explicit ``extra_specs`` cover installed siblings OUTSIDE the closure.
    ``only-if-needed`` keeps the non-hugpy env (torch, llama-cpp-python, the
    media extras) where it is unless a pin genuinely requires otherwise.

    ``constraints_path`` None → the pre-partition fallback::

        pip install -U --no-deps [--index-url <pkg_index>] <pkg_name>==<target>

    a code hot-swap of ONE distribution, siblings untouched (the fleet may be
    skewed until the constraints can be fetched)."""
    pkg_name = getattr(args, "pkg_name", None) or "hugpy-fleet"
    cmd = [sys.executable, "-m", "pip", "install", "-U"]
    if constraints_path:
        cmd += ["--upgrade-strategy", "only-if-needed", "-c", str(constraints_path)]
    else:
        cmd += ["--no-deps"]
    pkg_index = getattr(args, "pkg_index", None)
    if pkg_index:
        cmd += ["--index-url", pkg_index]
    if extra_index and extra_index.rstrip("/") != str(pkg_index or "").rstrip("/"):
        cmd += ["--extra-index-url", extra_index]
    else:
        extra_index = None
    # pip silently DROPS a plain-http index on a non-loopback host unless the
    # host is trusted (seen 2026-09-23: computron ignored http://192.168.1.100:7002
    # and "resolved" from PyPI alone). A LAN central over http is the normal case.
    for url in (pkg_index, extra_index):
        host = _insecure_index_host(url)
        if host and host not in cmd:
            cmd += ["--trusted-host", host]
    cmd.append(f"{pkg_name}=={target}")
    if constraints_path:
        cmd.extend(str(s) for s in (extra_specs or ()))
    return cmd


def _prepare_converge(args, target: str, constraints_url: str | None = None,
                      pkg_index_url: str | None = None
                      ) -> "tuple[list[str], str | None]":
    """Fetch central's lockstep pins into a temp file and build the pip command.

    ``pkg_index_url`` is the reply's hint for central's own index (added as
    ``--extra-index-url`` on both the constrained and the fallback command).

    Returns ``(cmd, constraints_path)``; ``constraints_path`` is None on the
    fallback path (fetch failed / no lines / no central URL) and the command is
    the single-package ``--no-deps`` form. The caller owns the temp file — see
    ``_discard_constraints``."""
    pkg_name = getattr(args, "pkg_name", None) or "hugpy-fleet"
    url = _constraints_url(args, constraints_url)
    lines: list[str] = []
    if url:
        try:
            lines = _fetch_constraints(url)
        except Exception as exc:  # noqa: BLE001 — degrade, never block the converge
            logger.warning("self-update: constraints fetch from %s failed: %s: %s",
                           url, type(exc).__name__, exc)
    if not lines:
        logger.warning(
            "self-update: no lockstep constraints from central (%s) — falling back "
            "to a single-package --no-deps install of %s==%s. Sibling distributions "
            "(hugpy-platform/-control/-storage/-engine/...) are NOT converged: the "
            "fleet may be SKEWED until central serves constraints.txt.",
            url or "no central URL", pkg_name, target)
        return _pip_converge_command(args, target, None, extra_index=pkg_index_url), None
    fd, path = tempfile.mkstemp(prefix="hugpy-constraints-", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    siblings = _installed_lockstep_siblings(lines, pkg_name)
    specs = [f"{name}=={target}" for name in siblings]
    logger.info("self-update: lockstep constraints from %s (%d pins); converging %s "
                "+ %d installed sibling(s) %s", url, len(lines), pkg_name,
                len(siblings), siblings)
    return _pip_converge_command(args, target, path, extra_specs=specs,
                                 extra_index=pkg_index_url), path


def _discard_constraints(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _self_update_if_needed(required: str | None, args, state=None,
                           constraints_url: str | None = None,
                           pkg_index_url: str | None = None) -> None:
    """Install central's required package version and re-exec, if we're behind.

    Source of the bytes: PyPI by default (where ``sync.trigger`` publishes), since
    workers reach central over the public internet and thus have PyPI access too
    (WireGuard is only the inference callback). ``--pkg-index`` /
    ``WORKER_PKG_INDEX`` overrides to central's own simple index — for a WG-only
    worker with no general egress, or to keep dev builds off public PyPI.

    The install is the lockstep converge built by ``_prepare_converge`` /
    ``_pip_converge_command``: central's constraints.txt pins every workspace
    distribution to ``required`` so the siblings move with the tracked package.
    Only when the pins cannot be fetched does it degrade to the pre-partition
    ``--no-deps`` hot-swap of ONE distribution (loudly: the fleet may be skewed).
    ``constraints_url`` is the reply's hint; absent, it is derived from
    ``args.central``. ``pkg_index_url`` is the reply's hint for central's own
    index (present only when central holds every wheel of ``required``); it
    joins PyPI as an extra index so a release published only on central still
    converges the fleet.
    """
    if not required:
        return  # central isn't managing versions -> never touch the install
    installed = _installed_pkg_version(args.pkg_name)
    if required == installed:
        return

    # NB: a distinct name — ``state`` is the WorkerState the restart needs.
    upd = _load_update_state(args)
    if upd.get("target") == required and (time.time() - upd.get("at", 0)) < _UPDATE_RETRY_BACKOFF:
        return  # already tried this exact target recently; back off

    source = args.pkg_index or ("PyPI + " + pkg_index_url if pkg_index_url else "PyPI")
    logger.info("self-update: %s %s -> %s (from %s)",
                args.pkg_name, installed or "(none)", required, source)
    cmd, constraints_path = _prepare_converge(args, required, constraints_url,
                                              pkg_index_url=pkg_index_url)
    try:
        rc = subprocess.call(cmd)
    except Exception as exc:  # noqa: BLE001
        logger.warning("self-update pip invocation failed: %s", exc)
        rc = 1
    finally:
        _discard_constraints(constraints_path)
    _save_update_state(args, {"target": required, "at": time.time(), "rc": rc,
                              "constrained": constraints_path is not None})

    if rc == 0:
        logger.info("self-update installed %s==%s (%s); restarting agent",
                    args.pkg_name, required,
                    "lockstep-constrained" if constraints_path else "single-package fallback")
        # Restart onto the new code. Under systemd this EXITS (Restart= respawns
        # a fresh, properly-tracked process — never the os.execv orphan that
        # squatted :9100). kill_slots=True: an orphaned slot child would keep
        # serving the OLD code forever (the adoption probe can't tell versions
        # apart), so the restart tears them down and the fresh agent respawns
        # them on the new version. Runs on the register/heartbeat thread — no
        # HTTP ack to send, so restart synchronously (does not return).
        from hugpy_platform.procutil import reexec
        _restart(state, reason="self-update", reexec_fn=reexec, kill_slots=True)
    else:
        logger.warning("self-update failed (pip rc=%s); staying on %s",
                       rc, installed or "(none)")


def _terminal_exit(exc: "WorkerRejected") -> None:
    """Stop the agent for good after central refused it (401/403).

    Called from the daemon heartbeat thread too, so it must kill the whole
    process (the main thread is blocked in the inference server) — hence
    os._exit. Exit code 0 so a ``Restart=on-failure`` unit does NOT respawn a
    deliberately-evicted worker (transient crashes still exit non-zero/killed and
    are restarted as before).
    """
    if getattr(exc, "code", None) == 403:
        logger.error("central BLOCKED this worker (403): %s. Stopping — have the "
                     "operator Admit it in the console to rejoin.", exc)
    else:
        logger.error("central refused enrollment (401): %s. Stopping — re-enroll "
                     "with a valid WORKER_ENROLL_TOKEN.", exc)
    os._exit(0)


from hugpy_platform.environment_status import environment_status

def env_status() -> dict:
    return environment_status(("llama-cpp-python", "transformers", "torch",
                               "diffusers", "accelerate", "bitsandbytes"))


# ── install-shape detection (central-side drift detection) ──────────────────
# What SHAPE this worker is installed in, so central can flag boxes that drifted
# from the canonical installer (a hand-rolled unit, a bare process from the wrong
# venv, a stray system unit). Additive heartbeat field — older centrals ignore it.
#
# Canonical unit name is the product-named `hugpy-worker.service`;
# `abstract-hugpy-worker.service` is the recognized LEGACY alias (hand-written
# units the setup doc used to document). Both count as canonical.
_CANONICAL_UNITS = {"hugpy-worker.service", "abstract-hugpy-worker.service"}
_INSTALL_SHAPE: "dict | None" = None


def _detect_systemd_unit() -> "str | None":
    """This process's systemd unit from ``/proc/self/cgroup`` (its last
    ``*.service`` path component), or ``None``. Best-effort — any read/parse
    failure returns ``None`` and never raises."""
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as fh:
            data = fh.read()
    except Exception:  # noqa: BLE001
        return None
    for tok in reversed(data.replace("/", "\n").split("\n")):
        tok = tok.strip()
        if tok.endswith(".service"):
            return tok
    return None


def _compute_install_shape(*, invocation_id, unit, prefix, executable) -> dict:
    """Pure install-shape logic (inputs -> the reported dict). Factored out so
    the canonical truth-table is testable without touching /proc or systemd."""
    via_systemd = bool(invocation_id)
    venv = (prefix or "").rstrip("/")
    canonical = bool(via_systemd
                     and unit in _CANONICAL_UNITS
                     and venv.endswith("hugpy-worker/venv"))
    return {"unit": unit, "via_systemd": via_systemd,
            "venv": prefix, "python": executable, "canonical": canonical}


def _install_shape() -> dict:
    """Cached install-shape for the heartbeat:
    ``{unit, via_systemd, venv, python, canonical}``.

    Computed ONCE (a running process's unit/venv don't change). Fully defensive:
    on any failure it returns a well-formed dict with null/false fields so a
    detection bug can never break the heartbeat.
    """
    global _INSTALL_SHAPE
    if _INSTALL_SHAPE is not None:
        return _INSTALL_SHAPE
    try:
        shape = _compute_install_shape(
            invocation_id=os.environ.get("INVOCATION_ID"),
            unit=_detect_systemd_unit(),
            prefix=sys.prefix,
            executable=sys.executable,
        )
    except Exception:  # noqa: BLE001 — detection must never break the heartbeat
        shape = {"unit": None, "via_systemd": False,
                 "venv": None, "python": None, "canonical": False}
    _INSTALL_SHAPE = shape
    return shape


def _serving_limits() -> dict:
    """Per-worker safe concurrency for IN-PROCESS serving, advertised to central.

    ``in_process_max_concurrency`` is the number of requests that may enter an
    in-process (llama.cpp / transformers) model runner at once — the per-model
    generation gate's limit. Default 1 (native contexts serialize). Central reads
    this to gate its relays: a worker that omits it (older agent) is assumed 1.
    """
    return {"in_process_max_concurrency": gen_gate.concurrency_limit()}


def _slot_capability() -> dict:
    """Whether this box can seat a NATIVE, crash-isolated llama-server slot.

    ``slot_capable`` is the engine-binary truth: a resolvable native
    ``llama-server`` (HUGPY_ENGINE_DIR / LLAMA_SERVER_BIN / PATH). When absent,
    slot seating falls back to the in-process ``llama_cpp.server`` child (text
    only — vision GGUF is refused) or, with SLOT_COUNT=0, to the in-process
    runner outright. Either way the box is serving non-native, which
    central/console must SEE in EVERY heartbeat — that silence is exactly
    computron's 2026-07-11 condition (slots implied, no usable engine binary).
    Fully defensive: any probe failure reports slot_incapable with the reason and
    never breaks the heartbeat.
    """
    try:
        from hugpy_engine.native.resolve import server_bin
        binpath = server_bin()
    except Exception as exc:  # noqa: BLE001 — capability probe must never break a beat
        return {"slot_capable": False,
                "slot_incapable_reason": f"engine probe failed: "
                                         f"{type(exc).__name__}: {exc}"}
    if binpath:
        return {"slot_capable": True, "slot_incapable_reason": None}
    try:
        from hugpy_engine.serve.slots import slots_enabled, _slot_count
        n = _slot_count()
        slotted = slots_enabled()
    except Exception:  # noqa: BLE001
        n, slotted = 0, False
    reason = ("no native llama-server binary resolvable (set HUGPY_ENGINE_DIR / "
              "LLAMA_SERVER_BIN or run `hugpy install-engine`)")
    if slotted:
        reason += (f"; the {n} configured slot(s) fall back to the in-process "
                   "llama_cpp.server child — text only, vision GGUF is refused")
    else:
        reason += "; SLOT_COUNT=0, so this worker serves models in-process (gated)"
    return {"slot_capable": False, "slot_incapable_reason": reason}


# Per-task capability honesty (2026-07-11). Yesterday three requests reached
# workers whose canonical venv lacks an optional ML dep (sentence-transformers,
# openai-whisper, keybert) and failed AT REQUEST TIME ("sentence-transformers is
# required…", whisper NoneType). Central routes by model assignment alone, so it
# had no way to know a box couldn't run the task. We advertise a per-task map from
# the SAME find_spec probe central's /ml readiness uses, and central skips a worker
# that says False for the request's task (workers_for_model). Legacy agents omit
# the field -> central assumes capable (no regression).

# whisper needs a REAL import probe: find_spec("whisper") can be True yet
# `import whisper` die under numba/numpy>=2.5 (yesterday's third incident), so the
# find_spec-only base map would over-advertise ASR. We do ONE guarded real import,
# TTL-cached so the ~15s heartbeat stays cheap AND an /ops/pip fix is re-detected
# within the TTL instead of needing a worker restart.
_WHISPER_PROBE_TTL_S = 300.0
_WHISPER_PROBE: dict = {"ok": None, "at": 0.0}


def _whisper_importable() -> bool:
    """Whether ``import whisper`` actually SUCCEEDS on this box (TTL-cached).

    Fast path: if whisper isn't even resolvable, return False without importing.
    Otherwise do a guarded real import (the find_spec-insufficient special case)
    and cache the result for ``_WHISPER_PROBE_TTL_S`` so heartbeats stay cheap.
    """
    from hugpy_engine.task_deps import have
    if not have("whisper"):
        return False
    now = time.time()
    cached = _WHISPER_PROBE.get("ok")
    if cached is not None and (now - _WHISPER_PROBE.get("at", 0.0)) < _WHISPER_PROBE_TTL_S:
        return cached
    try:
        import whisper  # noqa: F401 — REAL probe: numba/numpy>=2.5 landmine (2026-07-11)
        ok = True
    except Exception as exc:  # noqa: BLE001 — any import failure = ASR unavailable
        logger.info("whisper is installed but `import whisper` failed (%s: %s); "
                    "advertising automatic-speech-recognition UNAVAILABLE so central "
                    "won't route ASR here", type(exc).__name__, exc)
        ok = False
    _WHISPER_PROBE["ok"] = ok
    _WHISPER_PROBE["at"] = now
    return ok


def _tts_seatable() -> bool:
    """Can this worker actually SYNTHESIZE speech?

    The second overlay, and for the mirror of whisper's reason. ``chatterbox-tts``
    pins a torch that collides with the one this agent serves every other model
    on, so a worker seats it in the fleet's per-model env-PROFILE venv
    (``managers/serve/profiles.py``) and synthesis runs as a CHILD of that
    interpreter. ``find_spec`` in THIS interpreter therefore answers "no" for a
    seat that works perfectly — the opposite error to whisper's (importable yet
    broken), and the same lesson: advertise what the box can DO, measured on the
    interpreter that would do it.

    ``managers.tts.seat`` is the single source of truth the RUNNER also uses, so
    the advertisement and the execution can never disagree, and it caches (a
    heartbeat must not spawn a probe every beat). Never raises: an unresolvable
    seat advertises False, which central reads as "not yet" — the STRICT/
    affirmative rule the oracle catalog applies to this brand-new capability."""
    try:
        from hugpy_media.tts import seat as _tts_seat
        return bool(_tts_seat.available())
    except Exception as exc:  # noqa: BLE001 — a probe never breaks the heartbeat
        logger.info("text-to-speech seat probe failed (%s: %s); advertising "
                    "text-to-speech UNAVAILABLE", type(exc).__name__, exc)
        return False


def _task_capabilities() -> dict:
    """``{task: bool}`` this worker can actually run, advertised to central.

    Built from the shared canonical task->dependency map (managers.task_deps) with
    the SAME find_spec probe central's /ml readiness uses — cheap, no heavy imports
    — then overlaid with the whisper real-import special case and the TTS
    env-profile seat. Central gates routing on it (workers_for_model): a box
    missing an optional ML dep never gets that task's requests, instead of
    failing them at request time.
    """
    from hugpy_engine.task_deps import task_capabilities as _base_task_caps
    caps = _base_task_caps()
    caps["automatic-speech-recognition"] = _whisper_importable()
    caps["text-to-speech"] = _tts_seatable()
    # Media/video runners are optional plugins: a box without hugpy_media must
    # not advertise media tasks however its torch/transformers probe answers.
    try:
        from hugpy_fleet.worker import plugins as _plugins
        caps = _plugins.overlay_task_capabilities(caps)
    except Exception as exc:  # noqa: BLE001 — never break a heartbeat
        logger.debug("capability overlay failed: %s", exc)
    return caps


_STUDIO_CAP_CACHE: dict = {"at": 0.0, "value": None, "root": None}


def _studio_capability(state: "WorkerState | None" = None) -> "dict | None":
    """What THIS box can do for a studio (video) render, advertised so central's
    placement can route a REAL-model studio render here LIKE AN LLM (operator ruling
    2026-09-24) instead of depending on the central ``HUGPY_STUDIO_WORKER`` env var.

    ``{"render": bool, "models": [model_id...], "weights_root": str|None}``:
      * ``render`` — this box mounts /studio/render AND can run the studio spine
        (``hugpy_video`` importable) AND has a GPU. A GPU-less box does NOT advertise
        studio render (it could only ever produce a NO_GPU error), so central never
        routes a real render to it.
      * ``models`` — the REAL studio model_ids whose weights THIS box can actually LOAD
        right now: a pure ``os.path.isfile(model_index.json)`` read under the roots the
        render will use — the box-local hot root + the worker's own env root, PLUS the
        SHARED root central advertises in its heartbeat reply (``state.studio_weights_root``,
        the SAME path the render manifest is built against). The worker doesn't otherwise
        know central's shared root, and its OWN env usually doesn't set it (the render
        loads from the manifest root), which is why an env-only probe reported []. A box
        that cannot read that root (e.g. computron, no shared studio weights) reports [] —
        honest per-worker presence, never a transfer trigger.

    ``None`` when the box cannot render studio at all (no hugpy_video / no GPU). The cache
    is keyed on the central root too, so the first beat after central advertises it
    RECOMPUTES immediately rather than serving a stale [] for up to 60s. Fully guarded:
    telemetry must never break a heartbeat."""
    now = time.time()
    central_root = getattr(state, "studio_weights_root", None)
    if (now - _STUDIO_CAP_CACHE["at"] < 60.0
            and _STUDIO_CAP_CACHE.get("root") == central_root):
        return _STUDIO_CAP_CACHE["value"]
    value: "dict | None" = None
    try:
        from hugpy_fleet.worker import plugins as _plugins
        has_spine = bool(_plugins.video_present())
    except Exception:  # noqa: BLE001
        has_spine = False
    try:
        has_gpu = bool(detect_gpus())
    except Exception:  # noqa: BLE001
        has_gpu = False
    if has_spine and has_gpu:
        try:
            from hugpy_video.intel.studio.presence import present_model_ids, weights_roots
            roots = list(weights_roots())            # box-local hot + worker-env shared
            if isinstance(central_root, str) and central_root and central_root not in roots:
                roots.append(central_root)           # the root the render manifest loads from
            models = list(present_model_ids(tuple(roots)))
            value = {"render": True, "models": models,
                     "weights_root": (central_root or (roots[-1] if roots else None))}
        except Exception as exc:  # noqa: BLE001 — presence must never break a beat
            logger.debug("studio presence probe failed: %s", exc)
            value = {"render": True, "models": [], "weights_root": None}
    _STUDIO_CAP_CACHE.update(at=now, value=value, root=central_root)
    return value


def _adopt_studio_weights_root(state: "WorkerState", worker: "dict | None") -> None:
    """Adopt central's SHARED studio weights root from the register/heartbeat reply, so
    the NEXT ``_studio_capability`` probe checks presence under the root the render loads
    from. Best-effort: a missing/blank value leaves the prior value untouched (never wipes
    a known root over a transient reply)."""
    try:
        root = (worker or {}).get("studio_weights_root")
        if isinstance(root, str) and root:
            state.studio_weights_root = root
    except Exception:  # noqa: BLE001 — adoption must never break the beat
        pass


def _environment_digest() -> "dict | None":
    """The COMPACT environment fact that rides every beat (k118).

    Not the report — a digest, the profile names, which declared binaries are
    present and the driver/CUDA pair. Central pulls the full document from
    /ops/environment ON READ, the same discipline the rolling aggregate follows
    (operator ruling 2026-07-29: counts + a digest, never the document).

    The report itself is TTL-cached for 10 minutes, so a ~15s beat costs one
    dict copy nearly every time. Fully guarded: telemetry must never cost a
    heartbeat, because a missed beat drops the box off the fleet.
    """
    try:
        from hugpy_fleet.worker.environment_report import compact_digest
        return compact_digest()
    except Exception as exc:  # noqa: BLE001 — never break the beat
        logger.debug("environment digest failed: %s", exc)
        return None


def _doctrine_status() -> "dict | None":
    """This box's SELF-ASSESSMENT against the fleet doctrine, if it has one.

    The worker assesses itself so an ineligibility reason is available the
    instant central sees the beat, without a second round trip. ``None`` when no
    doctrine is resolvable here (a pip-installed worker has no repo above it) —
    and None is honest: central's own copy of the doctrine is authoritative and
    it can always re-assess from /ops/environment. What must NEVER happen is a
    fabricated "ok" from a box that never checked.
    """
    try:
        from hugpy_fleet.doctrine import doctrine as _doctrine
        current = _doctrine.latest()
        if current is None:
            return None
        from hugpy_fleet.doctrine.doctor import assess
        from hugpy_fleet.worker.environment_report import environment_report
        return assess(environment_report(), current).heartbeat_status()
    except Exception as exc:  # noqa: BLE001 — never break the beat
        logger.debug("doctrine self-assessment failed: %s", exc)
        return None


def _selftest_call(model_key: str, system: str, user: str) -> dict:
    """The self-test's model call — the SAME in-process path a real /infer takes.

    Deliberately not a new serving route: reusing ``_run_once`` means the
    self-test can only ever do what an ordinary request does, and a model that
    is already resident stays resident. It never asks for a load (the caller has
    already proven residency), never touches spill, and caps its own output so a
    rambling model cannot turn a health probe into a long generation."""
    payload = {
        "model_key": model_key,
        "system": system,
        "prompt": user,
        "max_tokens": 400,
        "temperature": 0.0,
    }
    result = _run_once(payload) or {}
    return {
        "text": result.get("text") or "",
        "finish_reason": result.get("finish_reason"),
        "think_leak": bool(result.get("reasoning_content")),
    }


def _aggregate_tick(state: WorkerState, *, loading, loaded, calib_samples,
                    vram_split, pid_log) -> dict:
    """Fold THIS BEAT's already-computed facts into the rolling aggregate.

    Every argument is a value the heartbeat computed for its own payload — the
    loading/loaded sets, the calibration samples it drained, the RAM/VRAM split
    it sampled. Nothing here measures anything: that is the whole constraint the
    operator's ruling imposes, and it is why this function takes data instead of
    going to get it.

    Returns the COMPACT summary that rides the beat. Fully guarded: telemetry
    must never cost a heartbeat, because a missed beat drops the box off the
    fleet."""
    try:
        agg = _aggregate.get_aggregate()
        # Cold-load events, from two already-free sources: the loading->loaded
        # transition the beat reports anyway (beat-cadence), and the calibration
        # samples (precise load_seconds, measured by the 0.1.224 helpers).
        agg.observe_loading(loading, loaded)
        agg.ingest_calibration_samples(calib_samples)
        agg.record_process_health(
            ram_worker_bytes=_ram_worker_bytes(),
            ram_external_bytes=_ram_external_bytes(),
            vram_attributed_bytes=(vram_split or {}).get("vram_attributed_bytes"),
            vram_unattributed_bytes=(vram_split or {}).get("vram_unattributed_bytes"),
            resident_models=len(loaded or []),
            unattributed_pids=len(((pid_log or {}).get("unattributed") or [])),
        )
        # Aptitude self-test: OFF unless the operator set the lever. When off
        # this is one env lookup and returns immediately — no import of the
        # scoring package, no call, no model touched.
        if _aggregate._selftest_enabled():
            try:
                from hugpy_fleet.worker.aptitude import selftest as _selftest
                last_served = {
                    k: (r.get("last_served_at") or 0.0)
                    for k, r in agg.document().get("models", {}).items()
                }
                out = _selftest.get_runner().maybe_run(
                    loaded, _selftest_call,
                    last_served=last_served, loading=loading)
                if out.get("ran") and out.get("score"):
                    agg.record_selftest(out["model_key"], out["score"])
            except Exception as _se:  # noqa: BLE001 — a dark lever never breaks a beat
                logger.debug("worker selftest skipped: %s", _se)
        # One bounded write per beat at most; the debounce already collapsed the
        # request burst that happened between beats.
        agg.maybe_flush(force=True)
        return agg.heartbeat_summary()
    except Exception as exc:  # noqa: BLE001 — never break the beat
        logger.debug("aggregate tick failed: %s", exc)
        return {}


# Edge-triggered heartbeat wake (2026-09-08). The console's "answering" pill is
# heartbeat-borne (allocations[].busy rides the beat), so on the 15s cadence it
# appeared near the END of an answer and lingered up to a beat after it — the
# operator's exact complaint. A slot child POSTs /ops/heartbeat-nudge on its
# inflight 0↔1 edges; the loop then beats EARLY with the same full, measured
# payload (SlotPool().statuses() probes live, so the busy flag is fresh truth,
# never an inference). _HB_NUDGE_MIN_S debounces a nudge burst into one early
# beat — an edge arriving mid-beat re-sets the event, so no transition is ever
# swallowed, it just waits out the debounce. Steady cadence is unchanged.
_HB_WAKE = threading.Event()
_HB_NUDGE_MIN_S = 2.0


def _heartbeat_loop(client: CentralClient, state: WorkerState, args) -> None:
    last_beat = 0.0
    while True:
        if _HB_WAKE.wait(timeout=args.heartbeat):
            _HB_WAKE.clear()
            hold = _HB_NUDGE_MIN_S - (time.time() - last_beat)
            if hold > 0:
                time.sleep(hold)
        last_beat = time.time()
        try:
            # Compute slot statuses ONCE — the unified allocations view reuses it.
            _activity_sampled_at = time.time()
            _slots = _slot_statuses()
            # Precision model->PID registry (2026-07-14): populate from data the
            # agent already has THIS beat — slot child_pid (subprocess-hosted),
            # in-process torch keys (share the worker PID), comfy — then reconcile
            # against nvidia-smi ground truth so central gets an honest per-model
            # PID+VRAM log plus any unattributed (foreign/rogue) squatters.
            # Best-effort and fully isolated: a registry error must NEVER skip the
            # beat (a missed heartbeat drops the worker off the fleet), so it
            # degrades to no log exactly like a no-GPU box.
            try:
                from hugpy_fleet.worker import pid_registry as _pidreg
                _pidreg.sweep_dead()
                for _s in (_slots or []):
                    if _s.get("model_key") and _s.get("child_pid"):
                        _pidreg.record_launch(_s["model_key"], _s["child_pid"], "subprocess")
                _inproc = _inprocess_gpu_bytes()
                for _mk in _inproc:
                    _pidreg.record_launch(_mk, os.getpid(), "in_process")
                # OWN-PID attribution (2026-07-14): tell reconcile which GPU pids are
                # the worker's own infrastructure so a residual agent / idle-slot CUDA
                # context lump reads as "cuda_context", not an anonymous squatter.
                # os.getpid() is the agent; the venv marker (this python's venv root)
                # catches slot children sharing the venv that aren't a recorded model.
                _own_pids = {os.getpid()}
                _venv_marker = None
                try:
                    _venv_marker = os.path.dirname(os.path.dirname(sys.executable)) or None
                except Exception:  # noqa: BLE001 — marker is best-effort telemetry
                    _venv_marker = None
                _pidreg.reconcile(_gpu_process_vram(), _inproc, _comfy_process_vram(),
                                  own_pids=_own_pids, self_venv_marker=_venv_marker)
                _pid_log = _pidreg.snapshot_for_heartbeat()
            except Exception as _pe:  # noqa: BLE001 — telemetry must never break the beat
                logger.debug("pid_registry snapshot failed: %s", _pe)
                _pid_log = None
            # Honest budget-bar VRAM inputs (t13/t14): split the SAME pid_log into
            # attributed (worker) vs unattributed (foreign) — sampled this beat,
            # alongside detect_gpus() below, so the two are one snapshot.
            _vram_split = _vram_split_from_pidlog(_pid_log)
            # t28: compute the allocations view ONCE, harvest calibration samples
            # off it (measured per-process VRAM per resident), then reuse it for
            # the payload. Fully guarded — capture must never skip a beat.
            _allocs = _allocations(_slots)
            # Computed ONCE and shared with the calibration collector and the
            # aggregate tick — neither may become a second caller of the same
            # probes. Hoisted above the collector so the load_seconds stamp
            # reads the same loading view this beat reports.
            _loaded_keys = loaded_model_keys()
            _loading_keys = _loading_model_keys()
            try:
                _collect_calibration_from_allocations(_allocs,
                                                      loading=_loading_keys)
                _calib_samples = _drain_calibration_samples()
            except Exception as _ce:  # noqa: BLE001 — telemetry never breaks a beat
                logger.debug("calibration capture failed: %s", _ce)
                _calib_samples = []
            _agg_summary = _aggregate_tick(
                state, loading=_loading_keys, loaded=_loaded_keys,
                calib_samples=_calib_samples, vram_split=_vram_split,
                pid_log=_pid_log)
            worker = client.heartbeat(
                state.worker_id,
                {
                    "gpus": detect_gpus(),
                    "activity_sampled_at": _activity_sampled_at,
                    "loaded_models": _loaded_keys,
                    "loading": _loading_keys,
                    "models_local": _models_local(state),
                    "provisioning": sorted(state._provisioning),
                    "provision_progress": state.provision_snapshot(),
                    # Measured central->worker transfer rate (B/s, EMA over
                    # completed provisions; None until one lands). The
                    # benchmark derives its cold-load budget from it.
                    "load_bytes_per_s": _load_bytes_per_s(),
                    "spill": _spill_describe(),
                    "url": state.url,     # None -> central keeps source-IP URL
                    "port": state.port,
                    # HONEST version: the running image (import-time snapshot),
                    # never live disk metadata — so a not-yet-effective self-update
                    # shows as skew, not cosmetic convergence (2026-07-20 ae).
                    "pkg_version": _running_pkg_version(args.pkg_name),
                    # ENGINE build id beside the pkg version (item L, k65): the
                    # native llama-server commit (`1 (039e20a)`) so engine skew is
                    # visible in /llm/workers alongside version skew. None when the
                    # box has no native engine (serves in-process). Additive field.
                    "engine_build": _engine_build(),
                    "role": state.role,
                    "rpc_endpoint": state.rpc_endpoint,
                    "free_ram": _free_ram_bytes(),
                    "ram_total": _ram_total_bytes(),
                    # Honest budget-bar inputs (t13/t14). free_ram stays the
                    # CLAMPED wire-compat field; these are the spec's raw inputs
                    # so central computes bar_used/encroachment/over-limit for RAM
                    # (ram_worker + ram_external) and VRAM (vram_attributed +
                    # vram_unattributed) without guessing. Absent on older agents
                    # -> central degrades to legacy bar semantics.
                    "free_ram_raw": _free_ram_raw_bytes(),
                    "ram_worker_bytes": _ram_worker_bytes(),
                    "ram_external_bytes": _ram_external_bytes(),
                    "vram_attributed_bytes": _vram_split.get("vram_attributed_bytes"),
                    "vram_unattributed_bytes": _vram_split.get("vram_unattributed_bytes"),
                    "disk": _disk_status(),
                    "engine": llama_cpp_cuda_status(),
                    "pool": os.environ.get("WORKER_POOL", ""),
                    "caps": _local_caps(),
                    "env": env_status(),
                    "config": _effective_config(),
                    "comfy": _comfy_status(),
                    "loaded_detail": _loaded_detail(),
                    "slots": _slots,
                    "allocations": _allocs,
                    # t28 load-and-learn: prediction-vs-measured observations
                    # captured this beat (measured successes off `allocations` +
                    # any admission refusals). Additive/optional — None when empty,
                    # omitted for a worker with HUGPY_CALIBRATION=off.
                    "calibration_samples": _calib_samples or None,
                    # Precision model->PID log (2026-07-14): {"models":[{model_key,
                    # pid,host_mode,vram_bytes,alive}], "unattributed":[{pid,name,
                    # mib}]}. None on older/no-GPU boxes -> central just omits it.
                    "pid_registry": _pid_log,
                    "storage": _worker_storage(state),
                    # VRAM eviction churn (slice 10): the eviction the operator was
                    # watching for now surfaces — count + last event {victim,
                    # subject, host_mode, vram_freed, at}. Silent before; a VRAM
                    # evict-to-fit at admission increments this so the churn data
                    # includes GPU evictions, not just disk reaps.
                    "vram_evictions": dict(_VRAM_EVICTIONS),
                    # VRAM-HOLDER METER (incident 2026-08-05): per-card VRAM
                    # breakdown + a row per holder, with the idle own-venv
                    # squatter flagged EXPLICITLY (holds VRAM, no live slot claim,
                    # not serving — reapable but never auto-reaped). Read-only,
                    # best-effort: a telemetry error must NEVER skip a beat (a
                    # missed heartbeat drops the worker off the fleet), so it
                    # degrades to None exactly like a no-GPU box. Additive/optional
                    # (extra='ignore' drops it for older central).
                    "vram_holders": _vram_holders_beat(state),
                    "install": _install_shape(),
                    # Concurrency hardening (2026-07-11): advertise this box's
                    # safe in-process concurrency + whether it can seat a native
                    # crash-isolated slot, so central can gate relays and the
                    # console can badge a worker that's silently serving in-process.
                    "serving_limits": _serving_limits(),
                    **_slot_capability(),
                    # Per-task capability honesty (2026-07-11): which /ml tasks
                    # this box can actually run, so central won't route a task
                    # whose optional dep is missing here (workers_for_model gate).
                    "task_capabilities": _task_capabilities(),
                    # STUDIO (video) placement signal (2026-09-24): what this box can do
                    # for a studio render + which real studio models it holds on disk, so
                    # central places a video render LIKE AN LLM (registry pick) instead of
                    # depending on HUGPY_STUDIO_WORKER. Additive/optional — an older
                    # central drops it (extra='ignore'); None on a no-GPU / no-spine box.
                    "studio": _studio_capability(state),
                    # k118 ENVIRONMENT DOCTRINE. Two additive fields, both
                    # optional (extra='ignore' drops them for an older central):
                    # a compact digest of what this box HAS, and its own verdict
                    # against the fleet doctrine. The doctrine findings are what
                    # turn "the job failed on the worker" into "this worker was
                    # never eligible, and here is the pip line that fixes it".
                    "environment_digest": _environment_digest(),
                    "doctrine_status": _doctrine_status(),
                    # ROLLING AGGREGATE summary (operator ruling 2026-07-29):
                    # counts + a digest/mtime, NEVER the document. Central reads
                    # it to know what the worker holds and whether it changed;
                    # the document itself is pulled ON READ via
                    # GET /llm/workers/<id>/aggregate. Keeping the file off the
                    # beat is the point — the beat must stay small enough that
                    # it is never the thing that starves.
                    #
                    # ADDITIVE, worker->central: an OLDER central's
                    # HeartbeatRequest is pydantic with the default
                    # extra='ignore', so it silently drops this key. Safe in
                    # both directions; see the release-ordering note in the
                    # aggregate relay route on central.
                    "aggregate": _agg_summary or None,
                },
            )
            # Adopt any assignment change made in the UI + pre-provision it.
            _sync_assignment(state, worker)
            # Adopt central's shared studio weights root so the next studio-presence
            # probe checks the root the render actually loads from.
            _adopt_studio_weights_root(state, worker)
            # Adopt central's resource limits (min of central + local config).
            _apply_central_limits(worker)
            # t28: adopt central's learned per-model need corrections (if any).
            _adopt_calibration(worker)
            # k2: adopt central's model BLOCK set (if any) — see
            # _adopt_blocked_models. Gates only this worker's own background
            # provisioning (download) re-kick.
            _adopt_blocked_models(worker)
            # Eviction policy (2026-07-25): least-reaping is FLEET-WIDE because
            # central's storage_proposal runs the same drop pass — see
            # _adopt_least_reaping. Omitted by a central with no opinion.
            _adopt_least_reaping(worker)
            # Keep the STORAGE budget's two central-owned inputs on state: the
            # disk allocation and the LRU clock the FIFO orders by. Both are
            # facts only central holds; the pull path reads them off state.
            _adopt_storage_inputs(state, worker)
            # Converge to central's required package version (restarts on update).
            _self_update_if_needed((worker or {}).get("required_pkg_version"), args, state,
                                   constraints_url=(worker or {}).get("constraints_url"),
                                   pkg_index_url=(worker or {}).get("pkg_index_url"))
        except WorkerRejected as exc:
            _terminal_exit(exc)   # does not return
        except urllib.error.HTTPError as exc:
            if exc.code == 410:
                # Central forgot us (restart / cleared registry) — re-register.
                logger.warning("central returned 410; re-registering")
                _register(client, state, args)
            else:
                logger.warning("heartbeat HTTP %s", exc.code)
        except Exception as exc:
            logger.warning("heartbeat failed: %s", exc)


def _register(client: CentralClient, state: WorkerState, args) -> None:
    models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    payload = {
        "name": state.name,
        "url": state.url,            # None -> central uses the source IP
        "port": state.port,
        "gpus": detect_gpus(),
        "role": state.role,
        "rpc_endpoint": state.rpc_endpoint,
        "free_ram": _free_ram_bytes(),
        "ram_total": _ram_total_bytes(),
        "models": models or None,
        "worker_id": state.worker_id,
        # HONEST version: running image, not live disk metadata (see heartbeat).
        "pkg_version": _running_pkg_version(args.pkg_name),
        # ENGINE build id from first contact (item L, k65) — see heartbeat.
        "engine_build": _engine_build(),
        "engine": llama_cpp_cuda_status(),
        "pool": os.environ.get("WORKER_POOL", ""),
        "caps": _local_caps(),
        "env": env_status(),
        # Concurrency hardening (2026-07-11): advertise safe in-process
        # concurrency + native-slot capability from the first contact, so central
        # gates correctly and the console badges an in-process-serving box even
        # before the first heartbeat.
        "serving_limits": _serving_limits(),
        **_slot_capability(),
        # Per-task capability honesty (2026-07-11): advertised from first contact
        # so central's routing gate is correct before the first heartbeat.
        "task_capabilities": _task_capabilities(),
        # STUDIO (video) placement signal (2026-09-24) from first contact, so central
        # can place a real studio render here before the first heartbeat. Additive;
        # None on a no-GPU / no-spine box. See _studio_capability + the heartbeat.
        "studio": _studio_capability(state),
    }
    try:
        worker = client.register(payload)
    except WorkerRejected as exc:
        _terminal_exit(exc)   # does not return — blocked/revoked, don't retry
    state.worker_id = worker.get("id", state.worker_id)
    if state.worker_id:
        _save_worker_id(args.id_file, state.worker_id)
    # Adopt central's view of what we serve (it may already have assignments
    # for this worker_id from a previous session) and pre-provision them.
    _sync_assignment(state, worker)
    # Adopt central's shared studio weights root from the register reply, so the FIRST
    # heartbeat's studio-presence probe already checks the right root.
    _adopt_studio_weights_root(state, worker)
    _apply_central_limits(worker)
    logger.info("registered as worker id=%s serving models=%s", state.worker_id, worker.get("models"))
    # Converge to central's required package version before serving (restarts).
    _self_update_if_needed(worker.get("required_pkg_version"), args, state,
                           constraints_url=worker.get("constraints_url"),
                           pkg_index_url=worker.get("pkg_index_url"))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hugpy_fleet.worker")
    p.add_argument("--central", default=central_base_url(default=None),
                   help="Central base URL, e.g. https://hugpy.ai "
                        "(env HUGPY_BASE_URL; legacy WORKER_CENTRAL_URL honoured)")
    p.add_argument("--token", default=os.environ.get("WORKER_ENROLL_TOKEN"),
                   help="Enrollment token issued by the console (hpw_...). Sent as a "
                        "Bearer credential on register/heartbeat. Required once central "
                        "has HUGPY_WORKER_ENROLL_REQUIRED on; recommended otherwise.")
    p.add_argument("--name", default=os.environ.get("WORKER_NAME", socket.gethostname()))
    p.add_argument("--host", default=os.environ.get("WORKER_HOST", "0.0.0.0"),
                   help="Bind address for the worker's inference server")
    p.add_argument("--port", type=int, default=int(os.environ.get("WORKER_PORT", "9100")))
    p.add_argument("--advertise", default=os.environ.get("WORKER_URL"),
                   help="URL the central node should call back on "
                        "(defaults to http://<host>:<port>)")
    p.add_argument("--models", default=os.environ.get("WORKER_MODELS", ""),
                   help="Comma-separated model_keys to self-assign on registration")
    p.add_argument("--heartbeat", type=float, default=float(os.environ.get("WORKER_HEARTBEAT", "15")))
    from hugpy_platform import app_dirs as _hp
    p.add_argument("--id-file", default=_hp.worker_id_file())

    # Self-update: which distribution to track, and where to pull it from.
    # Default source is PyPI (where sync.trigger publishes); set --pkg-index to
    # central's simple index for a WG-only worker with no general egress.
    p.add_argument("--pkg-name", default=os.environ.get("WORKER_PKG_NAME", "hugpy-fleet"),
                   help="Distribution to self-update (default hugpy-fleet). "
                        "Must match the distribution whose version central advertises.")
    p.add_argument("--pkg-index", default=os.environ.get("WORKER_PKG_INDEX"),
                   help="Override pip --index-url for self-update "
                        "(default: PyPI; e.g. https://<central>/api/llm/pip/simple)")

    # Fleet role + RPC shard pool.
    p.add_argument("--role", default=os.environ.get("WORKER_ROLE", "worker"),
                   choices=["worker", "rpc"],
                   help="worker = whole-model (serves /infer); rpc = lends its GPU "
                        "to a shard pool via llama.cpp rpc-server")
    p.add_argument("--rpc-host", default=os.environ.get("WORKER_RPC_HOST", "0.0.0.0"),
                   help="bind address for rpc-server (role=rpc)")
    p.add_argument("--rpc-port", type=int, default=int(os.environ.get("WORKER_RPC_PORT", "50052")),
                   help="port for rpc-server / advertised rpc_endpoint (role=rpc)")
    p.add_argument("--rpc-bin", default=os.environ.get("WORKER_RPC_BIN", "rpc-server"),
                   help="path to the llama.cpp rpc-server binary (CUDA+RPC build)")

    # GPU/CPU spill defaults for this worker. These seed the spill env the
    # inference path reads; per-request overrides from central still win.
    spill = p.add_argument_group("spill (GPU/CPU split)")
    spill.add_argument("--spill", choices=["auto", "off"],
                       default=os.environ.get("WORKER_SPILL", "auto"),
                       help="auto = fit as many layers on GPU as VRAM allows "
                            "(spill rest to CPU); off = CPU only")
    spill.add_argument("--n-gpu-layers", type=int, default=_safe_int(os.environ.get("WORKER_N_GPU_LAYERS")),
                       help="llama.cpp: force N layers on GPU (overrides --spill)")
    spill.add_argument("--gpu-mem", type=float, default=_safe_float(os.environ.get("WORKER_GPU_MEM_GIB")),
                       help="transformers: per-GPU memory budget in GiB")
    spill.add_argument("--cpu-mem", type=float, default=_safe_float(os.environ.get("WORKER_CPU_MEM_GIB")),
                       help="transformers: CPU/RAM budget in GiB for offloaded layers")
    spill.add_argument("--tensor-split", default=os.environ.get("WORKER_TENSOR_SPLIT"),
                       help="multi-GPU split, comma-separated e.g. 0.7,0.3")
    spill.add_argument("--main-gpu", type=int, default=_safe_int(os.environ.get("WORKER_MAIN_GPU")),
                       help="primary GPU index")
    return p


def _safe_float(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _apply_cli_spill(args) -> None:
    """Seed the spill env from CLI flags (per-request overrides still win)."""
    if args.n_gpu_layers is not None:
        os.environ["HUGPY_N_GPU_LAYERS"] = str(args.n_gpu_layers)
    elif args.spill == "off":
        os.environ["HUGPY_N_GPU_LAYERS"] = "off"
    else:
        os.environ.setdefault("HUGPY_N_GPU_LAYERS", "auto")
    if args.gpu_mem is not None:
        os.environ["HUGPY_GPU_MEM_GIB"] = str(args.gpu_mem)
    if args.cpu_mem is not None:
        os.environ["HUGPY_CPU_MEM_GIB"] = str(args.cpu_mem)
    if args.tensor_split:
        os.environ["HUGPY_TENSOR_SPLIT"] = args.tensor_split
    if args.main_gpu is not None:
        os.environ["HUGPY_MAIN_GPU"] = str(args.main_gpu)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # 2026-09-18: drop per-request INFO firehose (keeps WARN+ and the worker's own INFO)
    for _qn in ("httpx", "httpcore", "werkzeug"):
        logging.getLogger(_qn).setLevel(logging.WARNING)
    # Optional media/video runners + their PID hooks, loaded through the
    # engine's task entry points. Absent packages are simply not advertised.
    try:
        from hugpy_fleet.worker import plugins as _plugins
        _plugins.load_worker_capabilities()
    except Exception as _exc:  # noqa: BLE001 — plugins never block boot
        logger.warning("worker capability plugins not loaded: %s", _exc)
    # Storage transfers (ensure_model_present) need the worker's budget gate,
    # provisioning telemetry and executor registrar: install them into
    # hugpy_storage.providers before anything can pull.
    try:
        from hugpy_fleet.worker.storage_hooks import install_storage_providers
        install_storage_providers()
    except Exception as _exc:  # noqa: BLE001 — an ungated pull beats no worker
        logger.warning("storage providers not installed: %s", _exc)
    # 2026-09-23: the split worker resolves models via direct engine imports and
    # never constructs a LocalBackend, so nothing installed the engine catalog
    # source into this process. hugpy_storage.catalog_register then no-ops against
    # NullCatalogSource and central can teach the worker NO model — every unbuilt
    # key 400s "unknown model_key … central could not teach it one", leaving zero
    # hot models fleet-wide. Install the bridge here at the worker composition root
    # so the live registry backs catalog_register / ensure_model_registered.
    try:
        from hugpy_engine.catalog_bridge import install as _install_catalog_bridge
        _install_catalog_bridge()
    except Exception as _exc:  # noqa: BLE001 — a box without the engine still boots
        logger.warning("engine catalog bridge not installed: %s", _exc)
    # BOOT DETOX (2026-07-08 ae crash-loop): a 0.1.158 studio render setdefault'ed
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, which SURVIVES the agent's
    # re-exec (os.environ is inherited by execv) and this driver/torch combo dies
    # natively under it — poisoning every subsequent CUDA (request-driven) load.
    # Unless the operator explicitly opted in (HUGPY_CUDA_EXPANDABLE=1), strip the
    # exact leaked value BEFORE any torch import so the box heals on converge.
    if (os.environ.get("HUGPY_CUDA_EXPANDABLE", "").strip() != "1"
            and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"):
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        logging.getLogger(__name__).warning(
            "boot detox: removed leaked PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            "(opt back in with HUGPY_CUDA_EXPANDABLE=1)")
    args = _build_parser().parse_args(argv)

    if not args.central:
        print("error: --central (or WORKER_CENTRAL_URL) is required", file=sys.stderr)
        return 2

    # WORKER_CENTRAL_URL is the worker marker the storage layer reads
    # (hugpy_storage.provision.worker_central_url): on a worker, serve-path
    # weights come from central ONLY, never Hugging Face. A worker started with
    # just --central / HUGPY_BASE_URL must carry it too — for this process and
    # every slot child (they inherit os.environ).
    os.environ.setdefault("WORKER_CENTRAL_URL", args.central)

    _apply_cli_spill(args)

    # A worker runs vision models on its own GPU in-process; it has no separate
    # vision server to POST to. Force in-process unless the operator overrode it.
    os.environ.setdefault("HUGPY_VISION_INPROCESS", "1")

    # Torch-first import guard (Bug D): pull torch into sys.modules NOW, before
    # anything in this process can import llama_cpp (the in-process GGUF fallback
    # or a stray probe). A CUDA-built llama_cpp imported first aborts torch's init
    # and leaves a broken half-module cached for the whole process — poisoning
    # every later vision/sd-turbo/whisper request. Priming here makes torch a
    # complete cached module the rest of the run reuses. No-op without torch.
    _prime_torch_before_llama()

    # Only advertise a URL when the operator set one explicitly. Otherwise leave
    # it to central, which derives the reachable address from the request source
    # IP — far more reliable than the worker guessing past 127.0.1.1 / NAT / odd
    # NICs. We still send the listen port so central can build host:port.
    advertise = args.advertise
    if not advertise:
        # Determine the worker's own outbound IP on the route to central. This
        # is reliable even across NAT hairpinning, which fools central's
        # source-IP guess (central would see the router, e.g. 192.168.1.1, not
        # the worker's .128). Falls back to None -> central uses the source IP.
        ip = _local_ip_toward(args.central)
        if ip:
            advertise = f"http://{ip}:{args.port}"
            logger.info("advertising self as %s (local IP toward central)", advertise)
    # Surface GPU usability up front: a worker that can't use CUDA will silently
    # serve every model on CPU. Make that loud so it's not mistaken for "slow".
    _gpus = detect_gpus()
    _cuda = torch_cuda_status()
    _lcpp = llama_cpp_cuda_status()
    if _cuda.get("available"):
        logger.info("torch CUDA ready: %s (torch %s, cuda %s) — transformers models use the GPU",
                    _cuda.get("device_name"), _cuda.get("torch_version"),
                    _cuda.get("cuda_version"))
    elif _gpus:
        logger.warning(
            "GPU(s) detected by nvidia-smi (%s) but torch.cuda.is_available() is "
            "False — transformers inference will run on CPU. This worker's Python "
            "env needs a CUDA build of torch. torch=%s cuda=%s err=%s",
            ", ".join(g.get("name") or "?" for g in _gpus),
            _cuda.get("torch_version"), _cuda.get("cuda_version"), _cuda.get("error"))
    else:
        logger.warning("no usable GPU (nvidia-smi found none and torch has no CUDA); "
                       "inference will run on CPU")

    # GGUF models go through llama.cpp, which needs its OWN CUDA build.
    if _gpus and _lcpp.get("installed") and _lcpp.get("supports_gpu_offload") is False:
        logger.warning(
            "llama-cpp-python is installed WITHOUT GPU offload support — GGUF "
            "models will run on CPU regardless of n_gpu_layers. Reinstall with "
            "CUDA: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install --force-reinstall "
            "--no-cache-dir llama-cpp-python  (llama_cpp %s)", _lcpp.get("version"))
    elif _gpus and _lcpp.get("supports_gpu_offload"):
        logger.info("llama.cpp GPU offload available (llama_cpp %s) — GGUF models "
                    "can use the GPU", _lcpp.get("version"))

    state = WorkerState(name=args.name, url=advertise,
                        worker_id=_load_worker_id(args.id_file),
                        central_url=args.central)
    state.port = args.port
    state.role = args.role

    # Operator runtime settings (console-set) project onto the env FIRST, so
    # the slot supervisor + every other reader sees them; drop-ins lose loudly.
    _apply_settings_env(args)

    # v3 final semantics: on-demand is the DEFAULT tier, so every occupant
    # except a static one may be bumped (LRU promotion) when another model
    # needs a seat — exactly the intent of "slots stay filled, seats change
    # hands on demand". The residency lookup lets the scheduler tell an
    # all-STATIC pool apart from a merely-busy one and fail loads with a
    # clear error instead of evicting.
    try:
        from hugpy_engine.serve.slots import (
            set_eviction_policy,
            set_residency_lookup,
            set_fit_check as set_slot_fit_check,
            set_make_room as set_slot_make_room,
        )
        set_eviction_policy(lambda mk: _residency(mk) == "on-demand")
        set_residency_lookup(_residency)
        # Real-VRAM ceiling gate (Fix A): the slot load/evict path now consults
        # REAL device free VRAM (ComfyUI-visible), not slot-occupancy count, so a
        # card topped out by an out-of-band process (ComfyUI) triggers an LRU
        # on-demand eviction before seating a new model instead of OOM/under-
        # offloading into a "free" seat on a full card. Degrades to allow when
        # unmeasurable (no-GPU / can't read VRAM) — byte-identical to today.
        set_slot_fit_check(_worker_slot_fit_check)
        # CROSS-TIER make-room (slice 10): once slot-side eviction is exhausted,
        # this reclaims VRAM held by an IN-PROCESS resident (invisible to the slot
        # scheduler) so a slot load isn't OOM'd by a sibling transformers model.
        set_slot_make_room(lambda mk: _vram_evict_to_fit(state, mk))
    except Exception as _exc:  # noqa: BLE001
        logger.warning("slot eviction policy not registered: %s", _exc)

    # CONTEXT-as-allocation (slice 11): register the ctx resolver so the served
    # -c honours the per-model ctx_pct — the value fit/admission reserved KV for.
    # Reverse-injection (serve never imports worker_agent). Returns the resolved
    # ctx int, or None (no ctx_pct) -> serve's default ctx path, byte-identical.
    try:
        from hugpy_engine.serve.serve import set_ctx_resolver

        def _ctx_resolver(mk, cfg=None):
            _cfg = None
            try:
                from hugpy_engine.config.main import get_model_config
                _cfg = get_model_config(mk, dict_return=True)
            except Exception:  # noqa: BLE001
                _cfg = None
            ctx, _pct, _mx = _resolved_ctx(mk, _cfg)
            return ctx
        set_ctx_resolver(_ctx_resolver)
    except Exception as _exc:  # noqa: BLE001
        logger.warning("ctx resolver not registered: %s", _exc)

    # Fix B (2026-07-15): ensure headroom before every ComfyUI gen. comfy_runner
    # is package-shared (central imports it), so it must not import worker/GPU
    # internals — instead the worker registers a hook it calls if present (the
    # same None-default indirection as the slot policy). Evicts on-demand managed
    # models (LRU, via _evict_model) until real free VRAM reaches the target, so
    # ComfyUI's demand is a first-class queue entry, not a silent squatter.
    try:
        from hugpy_media.comfy.comfy_runner import set_comfy_headroom_hook
        set_comfy_headroom_hook(
            lambda mk, job_id=None: _worker_ensure_comfy_headroom(state, mk, job_id))
    except Exception as _exc:  # noqa: BLE001 — headroom prep must never break boot
        logger.warning("comfy headroom hook not registered: %s", _exc)

    # Same evict-to-fit, for the IN-PROCESS diffusers image runners (t2i / img2img
    # / studio frames). Those pipelines load LAZILY inside the runner (AFTER the
    # runner object is cached, so dispatch.ensure_headroom_for_load at build time
    # never covers the real VRAM allocation), and a ram-only designation makes the
    # cross-tier make-room a no-op — so an idle slot squatter was never evicted and
    # the load CUDA-OOM'd (incident 2026-09-25). The runner calls this hook with
    # its priced footprint right before loading/generating; it reclaims the card
    # via the SAME evict verb/policy comfy uses.
    try:
        from hugpy_media.imagegen.imagegen_runner import set_imagegen_headroom_hook
        set_imagegen_headroom_hook(
            lambda mk, need=None, job_id=None:
            _worker_ensure_imagegen_headroom(state, mk, need, job_id))
    except Exception as _exc:  # noqa: BLE001 — headroom prep must never break boot
        logger.warning("imagegen headroom hook not registered: %s", _exc)

    # Env-profiles (stage 1): register the model->profile resolver the runner
    # spawn seam consumes, then KICK materialization of every declared profile
    # not yet ready for its manifest. Materialization is slow (pip) so it runs in
    # the background via a registered executor (register_executor) — a restart
    # shuts it down cleanly, and a profiled model only routes/seats once its
    # profile reads ready in the heartbeat. Boot-driven so the restart-based
    # /ops/config apply re-kicks idempotently (a ready profile is a no-op; a
    # changed manifest re-materializes). Fully additive to the boot path.
    # BOOT-TIME REGISTRY RE-WALK (slice 6). The registry is built ONCE at module
    # import from the discovery REPORT FILE (<DEFAULT_ROOT>/projects/
    # model_discovery.json); nothing on the worker ever re-walks the tree, so an
    # ABSENT or STALE report leaves the registry as STAPLES ONLY — every on-disk
    # model then fails get_model_config and dies in the scan's `no_config` bucket
    # (the ae 2026-07-17 incident: 63 no_config, models_local 65->0). The on-disk
    # dirs carry per-dir hugpy.json markers — the source of truth discover_models
    # reads — so a re-walk HERE re-derives their configs regardless of the report
    # file's state. refresh_registry is idempotent, updates in place, and is what
    # its own docstring says to call on startup; the worker just never did.
    # Guarded: a discovery failure must never ground the boot (registry stays
    # whatever import built). This is the honest presence fix — on-disk models
    # resolve configs from their markers, not from a possibly-stale report.
    try:
        from hugpy_fleet.worker.imports import models_config as _mc
        _before = len(_mc.MODEL_REGISTRY)
        _mc.refresh_registry(run_discovery=True)
        _after = len(_mc.MODEL_REGISTRY)
        logger.info("boot registry re-walk: %d -> %d model configs "
                    "(on-disk markers re-read; report-file staleness bypassed)",
                    _before, _after)
    except Exception as _exc:  # noqa: BLE001 — discovery must never break boot
        logger.warning("boot registry re-walk skipped (%s) — registry stays as "
                       "import-built; on-disk models may read as no_config", _exc)

    try:
        from hugpy_engine.serve import profiles as _profiles
        _profiles.set_model_resolver(_resolve_model_profile)
        _profiles.materialize_all(_RUNTIME_SETTINGS.get("profiles") or {},
                                  register=register_executor)
    except Exception as _exc:  # noqa: BLE001 — profiles must never break boot
        logger.warning("env-profiles not initialized: %s", _exc)

    # Contention-based residency (doctrine 2026-07-11): an on-demand model stays
    # resident until a NEW load needs its memory — then the LRU on-demand
    # resident yields (never static / gate-busy / slot-backed; 📌pinned DOES
    # yield per 2026-07-15 — pin is designation, not a resource lock). dispatch
    # owns the LRU mechanism; the worker registers the box-specific fit-guard +
    # yield predicate + a post-evict trim so each headroom re-check sees the
    # freed memory. See dispatch.ensure_headroom_for_load; the old idle clock
    # (_residency_sweep_once) is now opt-in behind on_demand_ttl_s.
    try:
        from hugpy_engine.dispatch.dispatch import (
            set_fit_check,
            set_evictable,
            set_post_evict_hook,
            set_make_room,
            set_evict_reason,
        )
        set_fit_check(_worker_fit_check)
        set_evictable(_worker_evictable)
        set_post_evict_hook(_trim_host_ram)
        # Telemetry only: LABELS a skip _worker_evictable already decided, so the
        # console can show which clause protected each resident. Decides nothing.
        set_evict_reason(_worker_evict_skip_reason)
        # CROSS-TIER VRAM make-room (slice 10): the in-process LRU yield is blind
        # to a slot-child squatter — this hook sees ALL residents (pid-registry
        # measured) and evicts the minimum permissible set through the /ops/evict
        # verb, then REFUSES an unfittable load before any CUDA allocation. Bound
        # to the live state so the eviction verb can act.
        set_make_room(lambda mk: _vram_evict_to_fit(state, mk))
    except Exception as _exc:  # noqa: BLE001
        logger.warning("contention residency hooks not registered: %s", _exc)
    # Fit-aware quant auto-pick ("most amicable gguf"): the worker owns the
    # free-VRAM/KV/reserve numbers, so it registers the picker; the load paths
    # consult it via overrides.autofit_gguf_prefer, which returns None whenever
    # ANY designation exists (per-worker pin -> model-wide gguf_file ->
    # cfg.filename all outrank it, and the plain election is the fallback).
    try:
        from hugpy_engine.serve.overrides import set_gguf_autofit_hook
        set_gguf_autofit_hook(_autofit_gguf_pick)
    except Exception as _exc:  # noqa: BLE001
        logger.warning("quant autofit hook not registered: %s", _exc)

    # Worker-local slot pool (SLOT_COUNT; settings > env > default). Children
    # inherit this URL and nudge an early beat on their inflight edges, so
    # central's busy view tracks answers within ~a second instead of one full
    # 15s beat. 127.0.0.1 on purpose — the nudge never leaves the box. An
    # ADOPTED orphan slot (spawned by an older agent) lacks the env and simply
    # keeps today's cadence-only behavior.
    os.environ["HUGPY_AGENT_NUDGE_URL"] = (
        f"http://127.0.0.1:{args.port}/ops/heartbeat-nudge")
    _supervise_slots()

    # F1/F5 wiring: control.cancel on this process's bus reaches the shared
    # job store (fires the cancel handle each live stream attached), and job
    # transitions publish back onto the bus. Same substrate as central.
    try:
        from hugpy_control.bus import wire_cancel, wire_job_events
        wire_cancel()
        wire_job_events(source=f"worker:{args.name or ''}")
    except Exception as _exc:
        logger.warning("comms bus wiring failed: %s", _exc)

    # role=rpc: launch the llama.cpp rpc-server and advertise this box's GPU as a
    # shard backend. The endpoint host is the same outbound IP we advertise for
    # /infer (reachable from the lead); central stores it as an rpc_servers entry.
    rpc_proc = None
    if args.role == "rpc":
        rpc_proc = _spawn_rpc_server(args)
        rpc_host = _local_ip_toward(args.central) or socket.gethostname()
        state.rpc_endpoint = f"{rpc_host}:{args.rpc_port}"
        logger.info("role=rpc — advertising shard endpoint %s", state.rpc_endpoint)

    client = CentralClient(args.central, token=args.token)
    # Present the SAME enrollment token on central-transfer (model-pull) requests
    # in provision.py, so provisioning keeps working once central turns on
    # HUGPY_WORKER_ENROLL_REQUIRED. No-op (tokenless, exactly today's behavior)
    # when args.token is None.
    from hugpy_storage.provision import set_enroll_token, set_worker_id, set_budget_state
    set_enroll_token(args.token)
    # Identify this worker on central-transfer requests so central can apply its
    # per-worker storage budget to BACKGROUND pulls (2026-07-17 handshake).
    set_worker_id(state.worker_id)
    # Register the live state so EVERY in-process transfer entry runs the storage
    # gate even when its caller didn't thread `state` (Part A, slice 8): the
    # /redownload route and any ensure_model_present(..., state=None) now gate
    # against the real limits/assigned instead of pulling atop the cap.
    set_budget_state(state)

    try:
        _register(client, state, args)
    except Exception as exc:
        logger.error("initial registration failed: %s", exc)
        # Keep going — the heartbeat loop will retry, and the server can still
        # serve a worker the operator registers manually.

    # Eviction telemetry relay (operator directive 2026-07-28). Installed AFTER
    # registration so events carry the id central knows this box by rather than
    # the hostname fallback. Buffered + batched on its own daemon thread: an
    # eviction only appends to a bounded ring, so a central that is down or slow
    # costs this worker nothing but stale events (drop-oldest, never re-queue,
    # never block). Install failure is logged and ignored — losing the console's
    # live view must not cost the fleet a worker.
    try:
        if _evt is not None:
            _evt.set_worker_id(state.worker_id)
            _evt.install_relay(lambda batch: client.evictions_ingest(batch))
            logger.info("eviction telemetry relay installed (worker_id=%s -> %s)",
                        state.worker_id, args.central)
    except Exception as _exc:  # noqa: BLE001
        logger.warning("eviction telemetry relay not installed: %s — evictions "
                       "still log to the journal, the console just won't stream "
                       "this worker", _exc)

    hb = threading.Thread(target=_heartbeat_loop, args=(client, state, args), daemon=True)
    hb.start()

    # Residency maintenance: enforces the VRAM headroom ceiling, runs the comfy
    # idle watchdog, and TTL-yields idle IN-PROCESS on-demand residents (opt-in).
    # It NEVER loads a model — a model becomes resident only when a request for it
    # arrives and evict-to-fit seats it.
    threading.Thread(target=_residency_sweep_loop, args=(state,), daemon=True).start()

    # UTIL-08 reconcile: failed pulls converge instead of drifting forever.
    threading.Thread(target=_reconcile_loop, args=(state,), daemon=True).start()

    logger.info("worker inference server listening on %s (advertising %s)",
                f"{args.host}:{args.port}", state.url)
    state.args = args   # the /ops endpoints need pkg_name/pkg_index/id_file
    # Build the server explicitly (instead of Flask's app.run) so the restart
    # path holds a handle to close the listening socket cleanly before exit —
    # releasing :9100 so systemd's respawned process binds without a collision.
    # make_server binds immediately, so state.http_server is set before we block.
    from werkzeug.serving import make_server
    state.http_server = make_server(args.host, args.port, build_app(state),
                                    threaded=True)
    state.http_server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
