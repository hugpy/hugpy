"""On-demand llama-server lead for cross-machine shard plans.

When the allocator shards a model across the GPU pool, the LEAD must load it
with llama.cpp's RPC backend (--rpc host:port,... + --tensor-split). The
0.3.x python binding can't do that (Llama() lost its rpc_servers param), so
the lead runs a managed ``llama-server`` subprocess instead and the existing
HTTP runner (LlamaCppRunner) talks to it.

One process per (model_key, rpc_servers, tensor_split), cached for the life
of this process — same lifecycle semantics as the in-process Llama singleton:
a changed shard plan takes effect on the next fresh load, not retroactively.

``ensure_shard_server`` returns the base_url on success or None on any
failure, so callers can always fall back to local loading.
"""
from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from typing import Optional

import httpx

from .imports import *

_SERVERS: dict = {}            # (model_key, rpc, ts) -> {"proc": Popen, "base_url": str}
_LOCK = threading.Lock()

_PORT_BASE = int(os.environ.get("HUGPY_SHARD_PORT_BASE", "8790"))
_HEALTH_TIMEOUT_S = float(os.environ.get("HUGPY_SHARD_HEALTH_TIMEOUT", "180"))


def _server_bin() -> Optional[str]:
    """llama-server binary, resolved per-OS (env -> `hugpy install-engine` -> PATH)."""
    from .....engine.resolve import server_bin

    return server_bin()


def _child_env(binary: str) -> dict:
    """os.environ + the managed engine's lib dirs on LD_LIBRARY_PATH (k94).

    A llama-server resolved from a hugpy-managed engine dir links sibling .so
    files (libllama/libggml/…) the child's loader can't otherwise find — it died
    with "libggml.so: cannot open shared object file" and callers silently fell
    back to the projector-less python path. A system/PATH binary is left alone.
    """
    env = dict(os.environ)
    try:
        from .....engine.resolve import ld_library_path_with_engine
        ld = ld_library_path_with_engine(env.get("LD_LIBRARY_PATH"),
                                         bin_path=binary)
        if ld:
            env["LD_LIBRARY_PATH"] = ld
    except Exception:  # noqa: BLE001 — never block a spawn on lib-path derivation
        logger.warning("lib-path derivation failed for %s", binary, exc_info=True)
    return env


def _resolve_mmproj(model_dir: str, cfg, main_path: Optional[str] = None) -> Optional[str]:
    """Find a multimodal projector (mmproj) for a vision GGUF, or None.

    llama.cpp serves vision GGUFs by loading the language model *and* a separate
    CLIP/projector GGUF via ``--mmproj``. An explicit ``mmproj_filename`` on the
    model config wins; otherwise we look for an ``*mmproj*.gguf`` sidecar in the
    model dir (the conventional name, e.g. ``mmproj-Qwen2.5-VL-3B-f16.gguf``),
    skipping the main weights file. Text models have no projector, so this
    returns None and ``--mmproj`` is never added — shard behaviour is unchanged.
    """
    name = getattr(cfg, "mmproj_filename", None) or getattr(cfg, "mmproj", None)
    if name:
        cand = os.path.join(model_dir, name)
        return cand if os.path.isfile(cand) else None
    try:
        for fn in sorted(os.listdir(model_dir)):
            if fn.lower().endswith(".gguf") and "mmproj" in fn.lower():
                cand = os.path.join(model_dir, fn)
                if cand != main_path:
                    return cand
    except OSError:
        pass
    return None


def _free_port(start: int) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise OSError(f"no free port in [{start}, {start + 200})")


def _wait_healthy(base_url: str, proc: subprocess.Popen) -> bool:
    """Poll /health until ok; loading a big model over RPC can take a while."""
    deadline = time.monotonic() + _HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False                      # server died during load
        try:
            with httpx.Client(timeout=2.0) as client:
                if client.get(f"{base_url}/health").status_code == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.5)
    return False


def _prune_dead() -> None:
    for key in [k for k, v in _SERVERS.items() if v["proc"].poll() is not None]:
        logger.warning("shard server for %s exited; dropping from cache", key[0])
        _SERVERS.pop(key, None)


def ensure_shard_server(
    model_key: str,
    rpc_servers: str,
    tensor_split=None,
) -> Optional[str]:
    """Get-or-spawn the llama-server lead for this shard plan.

    Returns its base_url ("http://127.0.0.1:<port>") once /health passes,
    or None (binary missing / no GGUF / failed to come up) so the caller can
    fall back to a local load.
    """
    ts_key = tuple(tensor_split) if tensor_split else ()
    cache_key = (model_key, rpc_servers, ts_key)

    with _LOCK:
        _prune_dead()
        hit = _SERVERS.get(cache_key)
        if hit:
            return hit["base_url"]

        binary = _server_bin()
        if not binary:
            logger.warning("ensure_shard_server: no llama-server binary "
                           "(set LLAMA_SERVER_BIN); cannot lead a shard")
            return None

        try:
            cfg = get_model_config(model_key)
            model_dir = ensure_model(model_key)
            model_path = get_gguf_file(model_dir, cfg)
        except Exception as exc:
            logger.warning("ensure_shard_server: model resolve failed for %s: %s",
                           model_key, exc)
            return None
        if not model_path:
            logger.warning("ensure_shard_server: no GGUF for %s — only llama_cpp "
                           "models can shard via RPC", model_key)
            return None

        port = _free_port(_PORT_BASE)
        base_url = f"http://127.0.0.1:{port}"
        cmd = [
            binary,
            "-m", os.fspath(model_path),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--rpc", rpc_servers,
            "-ngl", "999",                    # all layers onto the pooled backends
            "-c", str(DEFAULT_N_CTX),
        ]
        if ts_key:
            cmd += ["--tensor-split", ",".join(str(x) for x in ts_key)]

        # Vision GGUFs need their CLIP/projector loaded too; text models have
        # none, so this is a no-op for them. (Note: llama.cpp runs the projector
        # on the lead, not the RPC backends, so a phone backend offloads only
        # the language-model layers — still useful, just not the image encoder.)
        mmproj = _resolve_mmproj(model_dir, cfg, model_path)
        if mmproj:
            cmd += ["--mmproj", os.fspath(mmproj)]
            logger.info("shard lead: multimodal projector %s", mmproj)

        logger.info("shard lead: %s", " ".join(cmd))
        try:
            from ....._platform.procutil import popen_detached
            proc = popen_detached(                 # detaches so it survives reloads (cross-OS)
                cmd,
                env=_child_env(binary),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            logger.error("shard lead failed to start: %s", exc)
            return None

    # Health-wait outside the lock so a slow load doesn't serialize other models.
    if not _wait_healthy(base_url, proc):
        logger.error("shard lead for %s never became healthy; killing pid %s",
                     model_key, proc.pid)
        proc.kill()
        return None

    with _LOCK:
        _SERVERS[cache_key] = {"proc": proc, "base_url": base_url}
    logger.info("shard lead healthy for %s at %s (rpc=%s split=%s)",
                model_key, base_url, rpc_servers, ts_key or "auto")
    return base_url


# ---------------------------------------------------------------------------
# Native llama-server with --mmproj for vision GGUFs.
#
# The in-process llama-cpp-python multimodal handler fails to load the projector
# ("Failed to load mtmd context from <mmproj>"); the native llama-server loads it
# C-side and serves images correctly. So for a vision GGUF we spawn/reuse a
# managed llama-server (--mmproj) and the existing HTTP runner (LlamaCppRunner)
# talks to it. One process per model_key, cached for this process's life (same
# lifecycle as the shard lead). Returns None for non-vision models (no projector)
# or any failure, so callers fall back to the in-process load unchanged.
# ---------------------------------------------------------------------------
# HUGPY_VISION_NGL controls how many language-model layers the native vision
# llama-server offloads. Historically hardcoded "999" (ALL layers) — correct for
# a big card, but a guaranteed OOM on an 8 GB laptop GPU (a 7B VL quant is
# ~5-9 GB + a ~1.3 GB projector), so the server never became healthy and the
# vision model silently fell through to the in-process path that cannot load the
# projector at all. Default is now "auto": an HONEST partial-offload split from
# the free VRAM, reserving the projector's own VRAM (see spill.autofit +
# vision_projector_bytes). An explicit value still wins:
#   auto / (unset)   -> autofit a partial split (projector-aware)
#   max / all / 999  -> every layer on the GPU (the old behaviour)
#   <int>            -> exactly that many layers
_VISION_NGL_ENV = "HUGPY_VISION_NGL"


def _vision_ngl_arg(model_path: str, mmproj_path: "str | None") -> str:
    """Resolve the ``-ngl`` value for the native vision llama-server.

    Returns a string suitable for llama.cpp's ``-ngl`` ("999" == all). Falls back
    to "999" (today's behaviour) on any resolution error, so this can never make
    a load that used to work stop working."""
    raw = (os.environ.get(_VISION_NGL_ENV) or "").strip().lower()
    if raw in ("max", "all", "999", "-1"):
        return "999"
    if raw and raw != "auto":
        try:
            return str(int(raw))
        except ValueError:
            pass                                # bad value -> autofit below
    try:
        from .....managers.spill import autofit_gpu_layers, vision_projector_bytes
        reserve = 0
        if mmproj_path:
            try:
                reserve = int(os.path.getsize(mmproj_path))
            except OSError:
                reserve = vision_projector_bytes(model_path)
        else:
            reserve = vision_projector_bytes(model_path)
        ngl = autofit_gpu_layers(model_path, extra_reserve_bytes=reserve)
        return "999" if ngl == -1 else str(max(0, int(ngl)))
    except Exception as exc:                    # noqa: BLE001 — never break a load
        logger.warning("vision ngl autofit failed for %s (%s); offloading all",
                       model_path, exc)
        return "999"


def ensure_vision_server(model_key: str) -> Optional[str]:
    cache_key = (model_key, "__vision__", ())
    with _LOCK:
        _prune_dead()
        hit = _SERVERS.get(cache_key)
        if hit:
            return hit["base_url"]

        binary = _server_bin()
        if not binary:
            logger.warning("ensure_vision_server: no llama-server binary "
                           "(set LLAMA_SERVER_BIN); cannot serve vision natively")
            return None
        try:
            cfg = get_model_config(model_key)
            model_dir = ensure_model(model_key)
            model_path = get_gguf_file(model_dir, cfg)
        except Exception as exc:
            logger.warning("ensure_vision_server: model resolve failed for %s: %s",
                           model_key, exc)
            return None
        if not model_path:
            return None
        mmproj = _resolve_mmproj(model_dir, cfg, model_path)
        if not mmproj:
            return None  # not a vision GGUF — caller falls back to in-process

        port = _free_port(_PORT_BASE + 200)
        base_url = f"http://127.0.0.1:{port}"
        ngl = _vision_ngl_arg(os.fspath(model_path), os.fspath(mmproj))
        cmd = [
            binary,
            "-m", os.fspath(model_path),
            "--mmproj", os.fspath(mmproj),
            "--host", "127.0.0.1",
            "--port", str(port),
            "-ngl", ngl,                    # honest partial split (projector-aware autofit)
            "-c", str(DEFAULT_N_CTX),
        ]
        logger.info("vision server: %s -ngl=%s (projector %s)",
                    os.path.basename(os.fspath(model_path)), ngl,
                    os.path.basename(os.fspath(mmproj)))
        logger.info("vision server: %s", " ".join(cmd))
        try:
            from ....._platform.procutil import popen_detached
            proc = popen_detached(
                cmd,
                env=_child_env(binary),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            logger.error("vision server failed to start: %s", exc)
            return None

    # Health-wait outside the lock so a slow load doesn't serialize other models.
    if not _wait_healthy(base_url, proc):
        logger.error("vision server for %s never became healthy; killing pid %s",
                     model_key, proc.pid)
        proc.kill()
        return None

    with _LOCK:
        _SERVERS[cache_key] = {"proc": proc, "base_url": base_url}
    logger.info("vision server healthy for %s at %s", model_key, base_url)
    return base_url
