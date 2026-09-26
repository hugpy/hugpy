import logging
import threading
from typing import Dict

import httpx

from hugpy_engine.config.main import get_gguf_file, get_model_config
from hugpy_engine.llama.runners.src.base_runner import LlamaCppBaseRunner
from hugpy_engine.llama.runners.src.ccp_runner import LlamaCppRunner
from hugpy_engine.llama.runners.src.python_runner import LlamaCppPythonRunner
from hugpy_engine.llama.runners.src.shard_server import ensure_shard_server
# Serve path: weights are local-or-central on a worker, never Hugging Face
# (hugpy_storage.provision.ensure_serving_weights; computron 2026-09-23).
from hugpy_storage.provision import ensure_serving_weights

logger = logging.getLogger(__name__)
class LocalEngineUnavailable(RuntimeError):
    """Raised when no in-process GGUF engine can be built (llama-cpp-python is
    not installed) and no HTTP slot is up. Carries a user-facing message so the
    chat route can surface actionable guidance instead of a raw import error."""


# ---------------------------------------------------------------------------
# Process-local singleton cache for the heavy GGUF runners.
# Keyed by model_key (str). The adapter wrappers in chat_runner share these.
# ---------------------------------------------------------------------------

_LLAMA_INSTANCES: Dict[str, "LlamaCppBaseRunner"] = {}
_LLAMA_LOCK = threading.Lock()
# A build that FAILED is recorded here for the console/logs (the last reason per
# model), but the record NEVER blocks a later call: every call re-attempts the
# load and, if it fails, fails with THAT attempt's real loader error. Cleared the
# moment a load succeeds.
import time as _time
_REFUSED: dict = {}            # model_key -> (ts, reason, exc_type, load_failure)
# Builds are SLOW (slot-child spawn, evict-to-fit, a 46G cold load — minutes)
# and used to run UNDER _LLAMA_LOCK. That serialized the data plane against
# every registry READER: the heartbeat's loaded_runner_detail only wants a
# microsecond dict snapshot, but it queued behind the whole build — so under
# heavy load the worker went DEAF (missed beats -> central marks it offline
# while it is busily serving; the 2026-07-29 ae flap, same class as k53).
# Now: _LLAMA_LOCK is only ever micro-held (get/put/snapshot, the discipline
# evict_llama_runner already had), and builds serialize on THIS lock instead —
# one build at a time, exactly the old semantics, invisible to readers.
_LLAMA_BUILD_LOCK = threading.Lock()


def evict_llama_runner(model_key: str) -> bool:
    """Drop the heavy singleton for ``model_key`` and free its weights.

    dispatch.evict only releases the adapter wrapper; the loaded ``Llama``
    handle (VRAM/RAM-resident weights) lives HERE in ``_LLAMA_INSTANCES``, so
    without this cascade the console's "free VRAM" evicted nothing and the
    memory stayed pinned until process death. Close the llama_cpp handle when
    it has one (HTTP runners don't — their memory belongs to the server).
    """
    with _LLAMA_LOCK:
        runner = _LLAMA_INSTANCES.pop(model_key, None)
    if runner is None:
        return False
    llm = getattr(runner, "llm", None)
    try:
        if llm is not None and hasattr(llm, "close"):
            llm.close()
    except Exception:
        logger.warning("evict_llama_runner: close() failed for %s", model_key,
                       exc_info=True)
    try:
        runner.llm = None
    except Exception:
        pass
    import gc
    gc.collect()
    return True


def clear_llama_runners() -> "list[str]":
    """Evict every heavy singleton (the clear() analog of evict_llama_runner)."""
    with _LLAMA_LOCK:
        keys = list(_LLAMA_INSTANCES)
    return [k for k in keys if evict_llama_runner(k)]


def loaded_runner_detail() -> dict:
    """Per-loaded-model load facts (for the worker heartbeat / console serving
    rows): file bytes, layers offloaded vs total, and the resulting GPU share —
    what the load actually DID, not an estimate. HTTP-backed runners (slots,
    shard leads) expose none of this; they report an empty detail."""
    import os as _os

    def _shard_sum_bytes(path):
        """Size of the loaded gguf, summing ALL shards when ``path`` is one
        shard of a split model — a naive getsize on the first shard under-
        counts a multi-shard model ~3-4x (mirrors slot_agent._total_gguf_bytes)."""
        import re as _re
        import glob as _glob
        base = _os.path.basename(path)
        m = _re.search(r"-\d{5}-of-(\d{5})\.gguf$", base)
        if m:
            patt = f"{base[:m.start()]}-*-of-{m.group(1)}.gguf"
            shards = [s for s in _glob.glob(
                _os.path.join(_os.path.dirname(path), patt)) if _os.path.isfile(s)]
            if shards:
                return sum(_os.path.getsize(s) for s in shards)
        return _os.path.getsize(path)

    out: dict = {}
    with _LLAMA_LOCK:
        items = list(_LLAMA_INSTANCES.items())
    for key, r in items:
        d: dict = {}
        path = getattr(r, "model_path", None)
        if path:
            try:
                # This overlay CORRECTS the coarse dir-walk detail, so it must
                # stamp both keys: the artifact that loaded is both the row's
                # size and the expected-VRAM proxy for a GGUF runner.
                d["model_bytes"] = _shard_sum_bytes(path)
                d["weight_bytes"] = d["model_bytes"]
            except OSError:
                pass
            try:
                from hugpy_engine.spill import _gguf_layer_count
                d["total_layers"] = _gguf_layer_count(path)
            except Exception:
                pass
        thr = getattr(r, "n_threads", None)
        if thr is not None:
            d["threads"] = thr
        ngl = getattr(r, "n_gpu_layers", None)
        if ngl is not None:
            d["n_gpu_layers"] = ngl
            total = d.get("total_layers")
            if ngl == -1:
                d["gpu_pct"] = 100
            elif total:
                d["gpu_pct"] = round(100 * min(int(ngl), total) / total)
            elif int(ngl) == 0:
                d["gpu_pct"] = 0
        out[key] = d
    return out


def slot_backed_model_keys() -> "set[str]":
    """model_keys whose cached runner is an HTTP proxy (LlamaCppRunner with a
    base_url and no in-process ``llm`` handle) — a slot child or shard-lead
    server, NOT weights resident in this process. Callers use this to keep a
    slot-served model from ALSO being reported as an in-process resident (the
    kind='ram'/'loaded' double-count that flaps with the slot 'serving' row).
    Reads the already-built cache directly, so it never triggers a load."""
    with _LLAMA_LOCK:
        items = list(_LLAMA_INSTANCES.items())
    return {
        key for key, r in items
        if getattr(r, "base_url", None) and getattr(r, "llm", None) is None
    }


def _slot_still_holds(runner, model_key: str) -> bool:
    """STALE-SLOT-FIX-20260910: one cheap local GET to confirm a slot-backed runner's seat still
    holds `model_key`. Unknown/unreachable -> True (the request path's own
    retry handles a dead seat); only a POSITIVE mismatch returns False."""
    # ROOT-CAUSE FIX 2026-09-23: a runner built through serve_endpoint ->
    # SlotPool.endpoint_for (the ordinary _build_runner path) is ALSO seated in
    # a slot, but carries _slot_backed=False — so this check returned True
    # before ever looking, and a seat loaded for one request's allocation (the
    # benchmark's per-request ram-only lane: 0/64 layers) served every later
    # request that asked for another. Probe any HTTP runner; act only when the
    # answer is a slot agent's /status (has slot_id). In-process runners (no
    # base_url) and non-slot endpoints (swap proxy) keep the old behaviour.
    base_url = getattr(runner, "base_url", None)
    if not base_url:
        return True
    try:
        import httpx as _httpx
        st = _httpx.get(f"{base_url}/status", timeout=1.5).json()
    except Exception:  # noqa: BLE001
        return True
    if not isinstance(st, dict) or (not getattr(runner, "_slot_backed", False)
                                    and "slot_id" not in st):
        return True
    held = st.get("model_key")
    if held and held != model_key:
        # serve_endpoint seats under the CANONICAL key (resolved with the
        # resident preferred — the ambiguous-key rule, unchanged); compare
        # against that resolution, not the spelling this cache is keyed by.
        try:
            canonical = get_model_config(model_key, prefer=[held]).model_key
        except Exception:  # noqa: BLE001
            canonical = model_key
        if canonical != held:
            return False
    # Same model is not necessarily the same seat.  Explicit placement and
    # quant selection are load-time properties; reusing a healthy child with a
    # different contract silently attributes its answer/timing to the request.
    try:
        from hugpy_engine.serve.slots import (alloc_mismatch, alloc_signature, env_alloc_source,
                                              status_satisfies_opts, _asked)
        import os as _os
        opts = {}
        raw = (_os.environ.get("HUGPY_N_GPU_LAYERS") or "").strip()
        if raw and raw.lower() != "auto":
            opts["n_gpu_layers"] = raw
        cpu_moe = (_os.environ.get("HUGPY_N_CPU_MOE") or "").strip()
        if cpu_moe:
            opts["n_cpu_moe"] = cpu_moe
        selected = (_os.environ.get("HUGPY_GGUF_FILE") or "").strip()
        if selected:
            opts["path"] = selected
        mode = (_os.environ.get("HUGPY_ALLOC_MODE") or "").strip()
        if mode:
            opts["alloc_mode"] = mode
        # the seat's recorded ask vs this request's ask (per-request overrides
        # never outlive their request), then the effective-placement check
        req = alloc_signature(opts)
        src = env_alloc_source() or {"kind": "designation" if _asked(req) else "default"}
        why = alloc_mismatch(st, req, src)
        if why:
            logger.info("get_llama_runner: %s — %s", model_key, why)
            return False
        return status_satisfies_opts(st, opts)
    except Exception:
        return True


def get_llama_runner(model_key: str) -> "LlamaCppBaseRunner":
    """Get-or-build the singleton runner for a model_key.

    HTTP runner first (cheap probe); falls back to in-process Python.
    """
    if not isinstance(model_key, str):
        raise TypeError(
            f"get_llama_runner expects model_key: str, got {type(model_key).__name__}"
        )

    # Fast path: registry lock micro-held for the lookup only.
    with _LLAMA_LOCK:
        runner = _LLAMA_INSTANCES.get(model_key)
    if runner is not None:
        if _slot_still_holds(runner, model_key):
            return runner
        # STALE-SLOT-FIX-20260910: the seat this cached runner points at now holds another
        # model (swapped out to make room). Drop it and rebuild below, which
        # re-resolves through the slot pool.
        logger.warning("get_llama_runner: cached slot runner for %s is stale (seat "
                       "now serves another model) - dropping and re-resolving", model_key)
        evict_llama_runner(model_key)
    # Slow path: the BUILD lock serializes builds (unchanged policy — one heavy
    # load at a time); the registry lock is never held across it, so heartbeat/
    # status snapshot readers stay responsive through a minutes-long load.
    with _LLAMA_BUILD_LOCK:
        with _LLAMA_LOCK:
            runner = _LLAMA_INSTANCES.get(model_key)   # built while we waited?
        if runner is not None:
            return runner
        try:
            runner = _build_runner(model_key)
        except Exception as exc:
            # Record the last failure for the console/logs: the FULL reason
            # (uncapped — it must keep the loader stderr) + the structured
            # load_failure. Informational ONLY — it never blocks the next call,
            # which re-attempts the load. Cleared on success below.
            _REFUSED[model_key] = (_time.time(), f"{type(exc).__name__}: {exc}",
                                   type(exc), getattr(exc, "load_failure", None))
            raise
        _REFUSED.pop(model_key, None)
        with _LLAMA_LOCK:
            _LLAMA_INSTANCES[model_key] = runner
        return runner


def _require_profile_ready(model_key: str) -> "dict | None":
    """Env-profiles (stage 1) gate for the slot seat path.

    Returns the profile decision ``{'name','state','bin',...}`` when a dependency
    profile is attributed to ``model_key`` and is READY (the caller ships
    ``opts['profile_bin']`` so the slot child launches from that venv), or None
    when the model has no profile (base behavior, untouched).

    RAISES ``LocalEngineUnavailable`` when a profile is attributed but NOT ready
    (materializing/error) — a profiled model must NEVER fall back to the shared
    venv (that would reintroduce the exact dependency conflict the profile
    isolates). The message is errors-as-data naming the profile + its state.
    """
    try:
        from hugpy_engine.serve import profiles
        resolve = profiles.resolve_model(model_key)
    except Exception:  # noqa: BLE001 — profiles unavailable -> base behavior
        return None
    if not resolve:
        return None
    if resolve.get("state") != "ready":
        detail = f": {resolve.get('error')}" if resolve.get("error") else ""
        raise LocalEngineUnavailable(
            f"model {model_key!r} is attributed to dependency profile "
            f"{resolve.get('name')!r}, which is {resolve.get('state')}{detail} — "
            "the model will seat once the profile finishes materializing; it will "
            "NOT fall back to the shared venv (that would reintroduce the "
            "dependency conflict the profile isolates)")
    return resolve


# ---------------------------------------------------------------------------
# Vision GGUFs (2026-09-23): a model that ships a multimodal projector (mmproj)
# is served ONLY by a native llama-server launched with ``--mmproj`` — the slot
# child (slot_agent._build_cmd adds it from the same find_mmproj resolver).
# The in-process llama-cpp-python fallback cannot load the projector, so it used
# to seat the language weights TEXT-ONLY and every image was silently dropped
# (aeb load_reports: "vision model loaded in-process (text-only ...)"). Now a
# projector model the slot pool cannot seat is refused with the typed
# ``vision_needs_slot`` load failure, naming the exact reason + fit numbers.
# ---------------------------------------------------------------------------
def _resolve_serving_gguf(model_key: str) -> "str | None":
    """The GGUF the runner would load for ``model_key`` (override -> pin ->
    election), or None. Resolution only — never loads."""
    import os as _os
    try:
        cfg = get_model_config(model_key)
        mdir = ensure_serving_weights(model_key)
    except Exception:  # noqa: BLE001 — unresolvable reads as "no projector known"
        return None
    mpath = None
    try:
        from hugpy_engine.serve.overrides import resolve_override_gguf
        mpath = resolve_override_gguf(model_key, mdir)
    except Exception:  # noqa: BLE001
        mpath = None
    if not mpath:
        try:
            _prefer = (_os.environ.get("HUGPY_GGUF_FILE") or "").strip() or None
            mpath = get_gguf_file(mdir, cfg, prefer=_prefer)
        except Exception:  # noqa: BLE001
            mpath = None
    return _os.fspath(mpath) if mpath else None


def vision_projector_for(model_key: str, model_path: "str | None" = None) -> "str | None":
    """The mmproj projector the native server would load for this model, or None.

    Same resolver the slot child uses for ``--mmproj`` (hugpy_platform.utils.
    find_mmproj beside the served GGUF), plus an explicit ``mmproj_filename`` /
    ``mmproj`` on the model config. None for text models and on any error
    (degrade-not-guess: never invent a vision refusal)."""
    import os as _os
    try:
        path = model_path or _resolve_serving_gguf(model_key)
        if not path:
            return None
        try:
            cfg = get_model_config(model_key)
        except Exception:  # noqa: BLE001
            cfg = None
        named = (getattr(cfg, "mmproj_filename", None) or getattr(cfg, "mmproj", None)) if cfg else None
        if isinstance(named, str) and named:
            cand = named if _os.path.isabs(named) else _os.path.join(_os.path.dirname(path), named)
            if _os.path.isfile(cand):
                return cand
        from hugpy_platform.utils import find_mmproj
        return find_mmproj(path)
    except Exception:  # noqa: BLE001
        return None


def _gib(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "unknown"
    return f"{n / 2**30:.2f} GiB" if n >= 2**30 else f"{n / 2**20:.0f} MiB"


def vision_needs_slot_failure(model_key: str, model_path: "str | None",
                              mmproj: str, reason) -> "Exception":
    """Build the typed ``vision_needs_slot`` ModelLoadFailure: WHY the slot path
    could not seat this projector model, with the fit numbers (weights +
    projector via spill.vision_projector_bytes vs free/total VRAM) so the
    operator (and central's load_reports) can act on it."""
    import os as _os
    from hugpy_engine.serve.load_failure import ModelLoadFailure, VISION_NEEDS_SLOT
    weights = proj = free = total = None
    try:
        from hugpy_engine import spill as _spill
        proj = _spill.vision_projector_bytes(model_path or mmproj)
        if not proj and mmproj and _os.path.isfile(mmproj):
            proj = _os.path.getsize(mmproj)
        try:
            free = _spill.free_vram_bytes()
        except Exception:  # noqa: BLE001
            free = None
        try:
            total = _spill.total_vram_bytes()
        except Exception:  # noqa: BLE001
            total = None
    except Exception:  # noqa: BLE001
        pass
    if model_path and _os.path.isfile(model_path):
        try:
            weights = _os.path.getsize(model_path)
        except OSError:
            weights = None
    if reason is None:
        why = "no slot could seat it"
    elif isinstance(reason, BaseException):
        why = f"{type(reason).__name__}: {reason}"
    else:
        why = str(reason)
    need = (int(weights or 0) + int(proj or 0)) or None
    fit = (f"fit: weights {_gib(weights)} + projector {_gib(proj)} "
           f"= {_gib(need)} vs free VRAM {_gib(free)} of {_gib(total)}")
    msg = (f"{model_key}: {VISION_NEEDS_SLOT} — this GGUF ships a multimodal "
           f"projector ({_os.path.basename(mmproj)}) that only a native "
           "llama-server --mmproj slot child can load; the in-process "
           "llama-cpp-python fallback would serve it text-only (images silently "
           f"ignored), so it is refused. Slot path: {why}. {fit}. Free a slot / "
           "VRAM, or install a native llama-server (`hugpy install-engine`).")
    return ModelLoadFailure(msg, load_class=VISION_NEEDS_SLOT, path=model_path or mmproj,
                            model_key=model_key)


def _slot_serves_vision(base_url: str) -> "bool | None":
    """Does the seated llama-server report the vision modality? True/False from
    llama-server ``/props`` ``modalities.vision``; None when it can't tell
    (older build, unreachable) — never refuse on missing evidence."""
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"{base_url.rstrip('/')}/props")
            if r.status_code != 200:
                return None
            mods = (r.json() or {}).get("modalities")
    except Exception:  # noqa: BLE001
        return None
    if isinstance(mods, dict) and "vision" in mods:
        return bool(mods.get("vision"))
    return None


def _vision_http_runner(model_key: str, base_url: str, mmproj: "str | None",
                        model_path: "str | None") -> "LlamaCppBaseRunner":
    """HTTP runner over a seated llama-server. For a projector model: verify the
    child actually loaded the projector (``/props``) and force ``is_vision`` so
    image parts are folded in even when the registry row lacks the
    image-text-to-text task (the projector on disk is the truth)."""
    if mmproj and _slot_serves_vision(base_url) is False:
        raise vision_needs_slot_failure(
            model_key, model_path, mmproj,
            f"the seated llama-server at {base_url} reports no vision modality "
            "(child launched without --mmproj)")
    runner = LlamaCppRunner(model_key, base_url=base_url)
    if mmproj:
        runner.is_vision = True
    return runner


def _build_runner(model_key: str) -> "LlamaCppBaseRunner":
    # Per-box "never serve locally" policy: every branch below is a LOCAL serve
    # (slot spawn, native --mmproj/--rpc llama-server spawn, or in-process
    # llama-cpp-python weights in this process). A policy box hosts none of them —
    # fail fast with the actionable message instead of spawning/loading. Default
    # off === today's behavior; workers (which serve locally by design) never set
    # the flag. See managers.serve.policy.
    from hugpy_engine.serve.policy import no_local_serving, local_serving_error
    if no_local_serving():
        raise LocalEngineUnavailable(local_serving_error(
            model_key, detail="local GGUF serving is disabled on this box"))

    # Env-profiles (stage 1): if this model is attributed to a dependency profile,
    # resolve it up front. A non-ready profile RAISES here (propagates — never
    # caught by the slot-fallback try below, so a profiled model never silently
    # serves from the shared venv). A ready profile rides into the slot seat as
    # opts['profile_bin'] so the child launches from the profile venv.
    _profile = _require_profile_ready(model_key)

    # Cross-machine shard lead: a spill override set HUGPY_RPC_SERVERS, meaning
    # the allocator pooled remote GPUs for this load. The 0.3.x python binding
    # can't shard (no Llama(rpc_servers=…)), so spawn a managed
    # ``llama-server --rpc`` lead and talk to it over HTTP. Any failure falls
    # through to ordinary selection — sharding never breaks a request.
    from hugpy_engine.spill import rpc_servers as _rpc_servers, tensor_split as _tensor_split
    rpc = _rpc_servers()
    if rpc:
        base = ensure_shard_server(model_key, rpc, _tensor_split())
        if base:
            logger.info("get_llama_runner: shard lead (llama-server --rpc %s) for %s",
                        rpc, model_key)
            return LlamaCppRunner(model_key, base_url=base)
        logger.warning("get_llama_runner: shard lead unavailable for %s; "
                       "using ordinary selection", model_key)

    # The slot pool's SOFT refusal, if any: it rides into the in-process
    # fallback so a failed fallback names the first reason too (2026-09-23).
    _slot_refusal = None
    opts = None
    _mmproj = None          # vision projector (resolved once the path is known)
    _mpath = None
    try:
        candidate = LlamaCppRunner(model_key)  # HTTP runner
        # quick probe — if the server isn't up this will throw
        with httpx.Client(timeout=2.0) as client:
            client.get(f"{candidate.base_url}/health").raise_for_status()
        logger.info("get_llama_runner: using HTTP runner for %s", model_key)
        return candidate
    except Exception:
        # Slot-first local serving: a local load should land in a SLOT when one
        # is available — the model stays resident past this request (TTL, not
        # process lifetime), shows up in the console's Slots panel, and a load
        # that can't happen carries the slot agent's preflight REASON instead of
        # silently ballooning gunicorn RSS. This is the path dispatch's
        # worker-fallback takes too, so "served locally" now means "in a slot"
        # whenever the pool can take it. The slot's llama-server also loads
        # --mmproj itself, so vision models served this way see images.
        try:
            from hugpy_engine.serve.slots import SlotPool, slots_enabled
            if slots_enabled():
                # Resolve key→GGUF path HERE and hand it to the slot: models
                # registered from central live in THIS process's in-memory
                # registry, which a slot (separate process) never sees.
                opts = None
                try:
                    import os as _os
                    cfg = get_model_config(model_key)
                    mdir = ensure_serving_weights(model_key)
                    mpath = None
                    try:
                        from hugpy_engine.serve.overrides import resolve_override_gguf
                        mpath = resolve_override_gguf(model_key, mdir)
                    except Exception:
                        mpath = None
                    if not mpath:
                        # Central's per-worker pin is an explicit seat contract
                        # and outranks fit-aware automatic selection.
                        _prefer = (_os.environ.get("HUGPY_GGUF_FILE") or "").strip() or None
                        if _prefer is None:
                            try:
                                from hugpy_engine.serve.overrides import autofit_gguf_prefer
                                _prefer = autofit_gguf_prefer(model_key, mdir, cfg)
                            except Exception:
                                _prefer = None
                        mpath = get_gguf_file(mdir, cfg, prefer=_prefer)
                    if mpath:
                        opts = {"path": _os.fspath(mpath)}
                        # Ship the model's REAL context window too — the slot
                        # can't resolve cfg (separate process) and its bare
                        # default (4096) truncates long chats mid-stream
                        # ("ASGI callable returned without completing
                        # response" → incomplete chunked read up the chain).
                        try:
                            from hugpy_engine.serve.serve import _ctx_for
                            opts["ctx"] = int(_ctx_for(cfg, model_key))
                        except Exception:
                            pass
                    # Explicit per-model budgets (assign spill → env via the
                    # agent's _apply_spill) ride as per-load opts: slot
                    # processes were spawned earlier and never see env changes.
                    # k64: HUGPY_ALLOC_MODE rides too — the slot plans MoE
                    # placement per the ACTIVE mode (_moe_gpu_budget /
                    # _strict_gpu_only) and, being a separate process, never
                    # sees the agent's per-request env. Without it a
                    # max-ram/explicit designation silently planned as max-gpu.
                    for env_name, key in (("HUGPY_GPU_MEM_GIB", "gpu_mem_gib"),
                                          ("HUGPY_CPU_MEM_GIB", "cpu_mem_gib"),
                                          ("HUGPY_N_CPU_MOE", "n_cpu_moe"),
                                          ("HUGPY_ALLOC_MODE", "alloc_mode"),
                                          ("DEFAULT_LLAMA_THREADS", "threads")):
                        v = _os.environ.get(env_name)
                        if v:
                            opts = opts or {}
                            opts[key] = v
                    # An EXPLICIT GPU-layer designation (console 'Max GPU' = -1,
                    # 'CPU only' = off) must ride to the slot too: the slot is a
                    # separate process that never sees the agent's
                    # HUGPY_N_GPU_LAYERS, and its _build_cmd autofits (fail-closed
                    # to 0/CPU when it can't read the card), so without this a
                    # 'max GPU' GGUF silently serves on CPU. 'auto' is NOT shipped
                    # so slots keep autofitting from the VRAM free at seat time
                    # (slot 2 takes what slot 1 left).
                    _ngl = (_os.environ.get("HUGPY_N_GPU_LAYERS") or "").strip().lower()
                    if _ngl and _ngl != "auto":
                        try:
                            opts = opts or {}
                            opts["n_gpu_layers"] = 0 if _ngl in ("off", "cpu", "none") else int(_ngl)
                        except ValueError:
                            pass
                except Exception:
                    opts = None
                # Env-profiles (stage 1): a ready profile's venv bin dir rides to
                # the slot so its child launches from that venv (python-child
                # interpreter swap + PATH prefer). The slot is a separate process
                # that can't read the agent's settings, so it arrives as an opt.
                if _profile is not None and _profile.get("bin"):
                    opts = opts or {}
                    opts["profile_bin"] = _profile["bin"]
                    opts["profile"] = _profile.get("name")
                _mpath = (opts or {}).get("path") or _resolve_serving_gguf(model_key)
                _mmproj = vision_projector_for(model_key, _mpath)
                sep = SlotPool().endpoint_for(model_key, opts=opts)
                if sep:
                    logger.info("get_llama_runner: %s -> slot %s (loaded on demand%s)",
                                model_key, sep,
                                f", projector {_mmproj}" if _mmproj else "")
                    return _vision_http_runner(model_key, sep, _mmproj, _mpath)
                logger.warning("get_llama_runner: every slot is busy with another "
                               "model — %s falls back to in-process", model_key)
                _slot_refusal = "every slot is busy with another model"
        except LocalEngineUnavailable:
            raise                     # profile refusal must never fall back
        except Exception as exc:
            # A vision_needs_slot verdict raised inside the seat (the child came
            # up without the projector) is already the final, typed answer.
            if getattr(exc, "load_class", None) == "vision_needs_slot":
                raise
            # HARD loader rejection (2026-09-23, Echo-Mini): the native loader
            # refused the FILE (check_tensor_dims wrong shape, unknown
            # architecture 'clip', ...). An in-process retry of the same bytes
            # can only fail again, less informatively ("Failed to load model
            # from file"), after burning time/VRAM — and its generic text is
            # what used to reach central, the benchmark and the grader. Raise
            # the real verdict instead: key, file, class, loader stderr.
            if getattr(exc, "hard", False):
                from hugpy_engine.serve.load_failure import HardLoadFailure
                _path = getattr(exc, "path", None)
                if not _path:
                    _path = (opts or {}).get("path")
                _stderr = getattr(exc, "loader_stderr", None)
                raise HardLoadFailure(
                    f"{model_key}: hard_load_failure — the native loader rejected "
                    f"{_path or '<unresolved path>'} (hard load failure; retrying "
                    "cannot fix it, in-process fallback refused). Loader stderr: "
                    f"{_stderr or '(none captured)'} | slot: {exc}",
                    loader_stderr=_stderr, path=_path, model_key=model_key,
                    log_ref=getattr(exc, "log_ref", None)) from exc
            # A profiled model must not silently drop to the shared-venv in-process
            # path when its slot seat fails — surface it as errors-as-data instead.
            if _profile is not None:
                raise LocalEngineUnavailable(
                    f"model {model_key!r} needs dependency profile "
                    f"{_profile.get('name')!r} but the slot seat failed "
                    f"({type(exc).__name__}: {exc}) — refusing the shared-venv "
                    "in-process fallback") from exc
            # endpoint_for surfaces the slot agent's preflight reason verbatim
            # (e.g. "needs ~42 GB RAM (all shards) but only 12 GB available").
            logger.warning("get_llama_runner: slot load refused for %s: %s — "
                           "falling back", model_key, exc)
            _slot_refusal = exc
        # Env-profiles (stage 1): a profiled model must seat in a slot from its
        # profile venv. If we reach here it's ready but unseatable (SLOT_COUNT=0 /
        # slots disabled, or every slot busy) — refuse rather than drop to the
        # shared-venv in-process/vision path. Stage 1 serves profiled models only
        # via slot children; the in-process consumer arrives with stage 3.
        if _profile is not None:
            raise LocalEngineUnavailable(
                f"model {model_key!r} needs dependency profile "
                f"{_profile.get('name')!r} (ready) but no slot could seat it "
                "(SLOT_COUNT=0 / slots disabled, or all slots busy) — stage 1 "
                "serves profiled models only via slot children; refusing the "
                "shared-venv in-process fallback")
        # Vision GGUFs (2026-09-23): a model with a multimodal projector is
        # served ONLY by a native llama-server --mmproj. With a slot pool on
        # this box that means a slot child — so if the slot path did not seat it
        # (busy, refused, fit), refuse with vision_needs_slot and the reason
        # instead of letting the in-process fallback load it TEXT-ONLY (the
        # aeb load_reports defect). A box with NO slot pool (SLOT_COUNT=0) keeps
        # the managed native --mmproj server (ensure_vision_server) — still
        # projector-capable — and is refused the same way if that fails.
        if _mmproj is None:
            _mpath = _mpath or (opts or {}).get("path") or _resolve_serving_gguf(model_key)
            _mmproj = vision_projector_for(model_key, _mpath)
        if _mmproj:
            try:
                from hugpy_engine.serve.slots import slots_enabled as _slots_on
                _slots = bool(_slots_on())
            except Exception:  # noqa: BLE001
                _slots = False
            if _slots:
                raise vision_needs_slot_failure(
                    model_key, _mpath, _mmproj,
                    _slot_refusal if _slot_refusal is not None
                    else "slot pool did not seat it")
            _vexc = None
            try:
                from hugpy_engine.llama.runners.src.shard_server import ensure_vision_server
                vbase = ensure_vision_server(model_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("get_llama_runner: native vision server failed for %s: %s",
                               model_key, exc)
                vbase, _vexc = None, exc
            if vbase:
                logger.info("get_llama_runner: vision model %s -> native --mmproj server %s",
                            model_key, vbase)
                return _vision_http_runner(model_key, vbase, _mmproj, _mpath)
            raise vision_needs_slot_failure(
                model_key, _mpath, _mmproj,
                "slots are disabled on this box (SLOT_COUNT=0) and the managed "
                "native --mmproj server could not start"
                + (f" ({type(_vexc).__name__}: {_vexc})" if _vexc else
                   " (no llama-server binary / LLAMA_SERVER_BIN, or it never became healthy)"))
        # MoE GGUFs: REFUSE the in-process fallback (operator ruling 2026-07-26,
        # "the auto should be moe if it is an moe model").
        #
        # The expert split is expressible ONLY as llama-server's --n-cpu-moe, which
        # only slot_agent._build_cmd emits. The in-process path builds its kwargs in
        # spill.llama_kwargs(), which carries n_gpu_layers/tensor_split/main_gpu and
        # has NO n_cpu_moe equivalent — llama-cpp-python exposes no such parameter.
        # So an MoE model reaching here loads WHOLE: ngl=-1 drags the experts onto
        # the card (coder-next Q4_K_M = 1.49 GiB of non-expert tensors but ~21 GiB
        # once the 43.59 GiB of experts follow), which is the exact condition the
        # operator reported. Serving it anyway would honor the request while
        # silently discarding the allocation that was derived for it — the split IS
        # the allocation, so dropping the split is not a degraded success, it is a
        # different (and much worse) placement wearing the same name.
        #
        # This mirrors the profile/vision refusals directly above and below: when a
        # model's REQUIRED placement cannot be expressed on this path, raise with the
        # reason instead of quietly serving the wrong thing. LocalEngineUnavailable
        # is the established "this box can't serve it, try elsewhere" signal — on a
        # worker it surfaces verbatim in central's load_reports, and central is then
        # free to place the model on a slot-capable box rather than believing this
        # one seated it correctly.
        #
        # Dense models are untouched: gguf_moe_detail returns {"is_moe": False} for
        # dense GGUFs, non-GGUF paths, and ANY read failure (degrade-not-guess), so
        # this gate can only ever fire on a file measured to carry expert tensors.
        try:
            from hugpy_engine.spill import gguf_moe_detail
            _mpath = None
            try:
                _cfg = get_model_config(model_key)
                _mpath = get_gguf_file(ensure_serving_weights(model_key), _cfg)
            except Exception:  # noqa: BLE001 — unresolvable path reads as dense
                _mpath = None
            _moe = gguf_moe_detail(_mpath) if _mpath else {"is_moe": False}
        except Exception:  # noqa: BLE001 — the gate must never invent a refusal
            _moe = {"is_moe": False}
        if _moe.get("is_moe"):
            _exp = int(_moe.get("expert_bytes") or 0)
            _non = int(_moe.get("non_expert_bytes") or 0)
            # Surface the slot path's REAL reason (this attempt's loader error /
            # eviction outcome), not a generic "all slots busy" — the in-process
            # fallback is refused, so the slot's error is the answer the caller
            # must see.
            _why = (f" Slot path: {_slot_refusal}." if _slot_refusal is not None
                    else " No slot seated it (SLOT_COUNT=0 / slots disabled).")
            raise LocalEngineUnavailable(
                f"model {model_key!r} is a MoE GGUF "
                f"({_non / 2**30:.2f} GiB non-expert + {_exp / 2**30:.2f} GiB "
                "experts) and needs the expert split (--n-cpu-moe), which only a "
                "native llama-server slot child can express — the in-process "
                "llama-cpp-python path has no equivalent and would load the whole "
                "model onto the GPU." + _why +
                " Refusing the in-process fallback rather than silently discarding "
                "the split. Free a slot or install a native llama-server "
                "(`hugpy install-engine`).")
        # (Vision GGUFs never reach here: the projector gate above either
        # seated them on a native --mmproj server or raised vision_needs_slot.)
        logger.info(
            "get_llama_runner: HTTP unavailable, falling back to in-process for %s",
            model_key,
        )
        try:
            if _slot_refusal is None:
                return LlamaCppPythonRunner(model_key)
            return LlamaCppPythonRunner(model_key, slot_refusal=_slot_refusal)
        except ImportError as exc:
            # No local GGUF engine (llama-cpp-python missing) AND no HTTP slot.
            # Surface a clean, actionable error rather than letting a raw
            # ModuleNotFoundError escape to the client as a stack-trace string.
            logger.error("get_llama_runner: no local GGUF engine for %s (%s)", model_key, exc)
            raise LocalEngineUnavailable(
                "No local inference engine is available on this central "
                "(llama-cpp-python is not installed) and no model slot is running. "
                "Install the engine (pip install 'hugpy[engine]'), start a model slot, "
                "or bring a worker online for this model."
            ) from exc
