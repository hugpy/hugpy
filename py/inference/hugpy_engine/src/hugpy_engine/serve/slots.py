"""Slot scheduler — assigns models to the root-free slot supervisors.

The pool is a fixed set of slot control URLs (``SLOT_COUNT`` slots starting at
``SLOT_PORT_BASE``). On demand it routes a model to a slot:

    1. a slot already serving that model      -> reuse it
    2. an idle slot                            -> load the model there (the slot
                                                  autofits GPU layers from the
                                                  VRAM still free, so slots fill
                                                  the card in order)
    3. all slots busy                          -> return None; the caller routes
                                                  the overflow through swap

Pure HTTP to the supervisors — no systemctl, no root.
"""
from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("abstract_hugpy_dev.slots")


_LOGGED_COUNT = False

# Tiers-v2 eviction policy hook: a callable mk -> bool ("is this model
# on-demand, i.e. may it be bumped from its slot for a caller?"). Registered
# by the WORKER agent from its runtime settings; None (e.g. bare central)
# keeps the historical behavior: all-busy -> None -> swap fallback.
_EVICTION_POLICY = None

# Tiers-v3 residency lookup hook: a callable mk -> "serving"|"on-demand"|
# "static". Registered by the WORKER agent (its _residency). Lets the
# scheduler tell a merely-busy pool ("return None, swap handles it") from a
# STATIC-LOCKED pool, where every seat is immovable and the load must fail
# with a clear error instead. None (bare central) skips the check.
_RESIDENCY_LOOKUP = None

# Real-VRAM ceiling hook (Fix A, 2026-07-15): a callable mk -> bool answering
# "will loading THIS model keep the card at/under the ~90% ceiling given REAL
# current free VRAM (torch.cuda.mem_get_info — ComfyUI-visible, not managed-
# model bookkeeping)?". Registered by the WORKER agent (_worker_slot_fit_check).
# The gap this closes: an idle slot on a card 95%-full from a SEPARATE process
# (ComfyUI) would happily /load into the seat, then the child silently offloads
# fewer layers or OOMs — because slot routing keyed on slot-OCCUPANCY, never on
# real device pressure. When registered and it says NO (would breach ceiling),
# endpoint_for evicts the coldest on-demand occupant(s) via the SAME LRU
# mechanism the all-busy branch uses and re-checks, until the gate passes or
# nothing is evictable (then it proceeds anyway — honest-degrade, never HANG a
# legitimate request; the child's autofit does its best). None (bare central,
# no-GPU box, or a gate that can't measure) => byte-identical to today: the
# gate is skipped, occupancy-only routing stands.
_FIT_CHECK = None


def set_eviction_policy(fn) -> None:
    global _EVICTION_POLICY
    _EVICTION_POLICY = fn


def set_residency_lookup(fn) -> None:
    global _RESIDENCY_LOOKUP
    _RESIDENCY_LOOKUP = fn


def set_fit_check(fn) -> None:
    """Register the real-VRAM ceiling gate: ``fn(model_key) -> bool``, True when
    loading the model keeps the card at/under the ~90% ceiling given real current
    free VRAM. None disables the ceiling gate (bare central / no-GPU / can't
    measure) — occupancy-only routing, byte-identical to before."""
    global _FIT_CHECK
    _FIT_CHECK = fn


# CROSS-TIER make-room (slice 10): the slot ceiling loop above evicts only SLOT
# occupants — it is blind to an IN-PROCESS transformers resident squatting the
# card. This hook (registered by the worker) evicts ALL permissible residents
# (in-process included) from the pid-registry measured truth. Called once the
# slot-side eviction is exhausted, so a slot load can also reclaim VRAM held by a
# sibling in-process model. None -> the historical slot-only path.
_MAKE_ROOM = None


def set_make_room(fn) -> None:
    """Register the cross-tier VRAM make-room (slice 10): ``fn(model_key) -> dict``.
    None -> slot-only ceiling eviction, byte-identical to before."""
    global _MAKE_ROOM
    _MAKE_ROOM = fn


def _slot_count() -> int:
    global _LOGGED_COUNT
    raw = os.environ.get("SLOT_COUNT")
    try:
        n = max(0, int(raw)) if raw is not None else 2
    except ValueError:
        n = 2
    if not _LOGGED_COUNT:
        _LOGGED_COUNT = True
        # De-silence the env-layering ghost: a systemd drop-in can override the
        # unit's SLOT_COUNT and silently resurrect slots every restart (op's
        # limits.conf SLOT_COUNT=2). Record the EFFECTIVE value + raw env once
        # per process so the journal shows the truth.
        logger.info("SLOT_COUNT effective=%d (env=%r) — if this differs from the "
                    "unit file, a drop-in is overriding it", n, raw)
    return n


def slots_enabled() -> bool:
    # Per-box "never serve locally" policy: force the slot pool off regardless of
    # SLOT_COUNT, so serve routing/get_llama_runner never load a model into a
    # local slot (the path that spawned the OOM'ing llama-server on central).
    # Default off === today's behavior; workers never set the flag. See
    # .policy.no_local_serving.
    from hugpy_engine.serve.policy import no_local_serving
    if no_local_serving():
        return False
    return _slot_count() > 0


def _slot_host() -> str:
    return os.environ.get("SLOT_ADVERTISE") or os.environ.get("SLOT_HOST_ADDR") or "127.0.0.1"


def _slot_port_base() -> int:
    try:
        return int(os.environ.get("SLOT_PORT_BASE", "8101"))
    except ValueError:
        return 8101


def slot_urls() -> list[str]:
    # Under the no-local-serving policy the pool has no targets, so a directly
    # constructed SlotPool().endpoint_for() also returns None (defense-in-depth
    # alongside slots_enabled). Default off === today's behavior.
    from hugpy_engine.serve.policy import no_local_serving
    if no_local_serving():
        return []
    host, base = _slot_host(), _slot_port_base()
    return [f"http://{host}:{base + i}" for i in range(_slot_count())]


def _get(url: str, timeout: float = 3.0) -> dict:
    import httpx
    return httpx.get(url, timeout=timeout).json()


def _post(url: str, body: dict, timeout: float) -> dict:
    import httpx
    return httpx.post(url, json=body, timeout=timeout).json()


def _ngl(value):
    if isinstance(value, str) and value.strip().lower() in ("off", "cpu", "none"):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def status_satisfies_opts(status: dict, opts: dict | None) -> bool:
    """Whether a live slot already implements the explicitly requested seat.

    Only load-time properties stated by the caller constrain reuse.  Omitted
    properties retain normal cache semantics.
    """
    opts = opts or {}
    if "n_gpu_layers" in opts and _ngl(status.get("n_gpu_layers")) != _ngl(opts["n_gpu_layers"]):
        return False
    if "n_cpu_moe" in opts and _ngl(status.get("n_cpu_moe")) != _ngl(opts["n_cpu_moe"]):
        return False
    if opts.get("path"):
        current = os.path.basename(str(status.get("model_path") or ""))
        wanted = os.path.basename(str(opts["path"]))
        if current and current.lower() != wanted.lower():
            return False
    return True


# ── allocation identity of a seat (2026-09-23) ──────────────────────────────
# A slot child's placement is fixed at launch. What a REQUEST asks for (its
# designation spill, or a per-request override) must be compared with what the
# resident seat was LOADED FOR — otherwise a per-request ram-only benchmark
# lane leaves a 0/64-layer seat that every later ordinary request reuses.
# The comparison is requested-vs-requested (the slot records ``alloc_requested``
# at load), so an autofit/make-room plan that differs from the request never
# reads as a mismatch.

def _mode(value):
    v = str(value or "").strip().lower().replace("_", "-")
    return v or None


def alloc_signature(opts: dict | None) -> dict:
    """The placement a caller ASKED for: ``{"n_gpu_layers", "alloc_mode"}``,
    normalized (``off``/``cpu`` -> 0, ints, lowercase dashed mode), None when
    not asked. Everything else (make-room plans, ctx, threads) is excluded."""
    opts = opts or {}
    ngl = opts.get("n_gpu_layers")
    return {"n_gpu_layers": None if ngl in (None, "", "auto") else _ngl(ngl),
            "alloc_mode": _mode(opts.get("alloc_mode"))}


def _asked(sig: dict) -> bool:
    return any(v is not None for v in (sig or {}).values())


def _describe_sig(sig: dict | None) -> str:
    sig = sig or {}
    parts = []
    if sig.get("n_gpu_layers") is not None:
        parts.append(f"n_gpu_layers={sig['n_gpu_layers']}")
    if sig.get("alloc_mode"):
        parts.append(f"alloc_mode={sig['alloc_mode']}")
    return ", ".join(parts) or "default placement (nothing asked)"


def describe_alloc_source(src: dict | None) -> str:
    """'per-request ram-only from request <id> at <ts>' / 'designation gpu-only'
    / 'operator …' / 'unknown source'."""
    if not isinstance(src, dict) or not src.get("kind"):
        return "unknown source"
    kind = str(src["kind"])
    bits = [kind]
    if src.get("mode"):
        bits.append(str(src["mode"]))
    out = " ".join(bits)
    if src.get("request_id"):
        out += f" from request {src['request_id']}"
    if src.get("at"):
        try:
            out += " at " + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(src["at"])))
        except (TypeError, ValueError):
            out += f" at {src['at']}"
    if src.get("via"):
        out += f" via {src['via']}"
    return out


def alloc_mismatch(status: dict, requested: dict,
                   source: dict | None = None) -> str | None:
    """None when the live seat implements the requested allocation, else the
    human reason it does not (the reload reason the load report carries).

    * The seat records what it was loaded FOR (``alloc_requested``) and by whom
      (``alloc_source``). A request that asks for a placement must match it.
    * A request that asks for NOTHING matches any seat EXCEPT one seated by a
      per-request override: an override never outlives its request.
    * An older slot (no ``alloc_requested``) is judged only on what the request
      explicitly asked (status_satisfies_opts), as before."""
    have = status.get("alloc_requested")
    src = status.get("alloc_source") if isinstance(status.get("alloc_source"), dict) else None
    if not isinstance(have, dict):
        return None
    have_sig = alloc_signature(have)
    by_override = bool(src and src.get("kind") == "per-request")
    if _asked(requested):
        if have_sig == requested:
            return None
        # alloc_mode is a STRATEGY to REACH a placement, not the placement itself
        # (operator ruling 2026-09-25: nothing should suggest the want or need
        # for a model to evacuate resources for no need). Two modes that yield
        # the SAME effective seat must not force a reload — a seat seated
        # "explicit" at n_gpu_layers=-1 satisfies a later "max-gpu" request at
        # n_gpu_layers=-1. Reuse when the effective placement key (n_gpu_layers)
        # already matches; a real placement change (ram-only 0 vs gpu -1, a
        # different layer count) still differs HERE and still reloads, and
        # n_cpu_moe / file are verified right after by status_satisfies_opts. A
        # per-request-override seat is EXCLUDED (by_override) — it never outlives
        # its request.
        if (not by_override
                and requested.get("n_gpu_layers") is not None
                and have_sig.get("n_gpu_layers") == requested.get("n_gpu_layers")):
            return None
        if not _asked(have_sig) and not by_override:
            # seated with nothing asked (autofit/default): whether it implements
            # the explicit ask is decided on the EFFECTIVE placement, as before
            return None
    elif not by_override:
        return None
    total = status.get("total_layers")
    eff = status.get("n_gpu_layers")
    return (f"reloaded: resident had n_gpu_layers={eff}"
            + (f" of {total}" if total is not None else "")
            + f" ({_describe_sig(have_sig)}; {describe_alloc_source(src)}), request wants "
            + f"{_describe_sig(requested)} ({describe_alloc_source(source)})")


def env_alloc_source() -> dict | None:
    """The per-request provenance the worker agent projected from central's
    spill (HUGPY_ALLOC_SOURCE, JSON; cleared when absent)."""
    raw = os.environ.get("HUGPY_ALLOC_SOURCE")
    if not raw:
        return None
    try:
        import json as _json
        val = _json.loads(raw)
    except ValueError:
        return None
    return val if isinstance(val, dict) else None


def env_request_opts(opts: dict | None = None) -> dict:
    """``opts`` + the placement the CURRENT request projected into env (the
    worker agent's _apply_spill), exactly as endpoint_for seats with it."""
    eff_opts = dict(opts or {})
    for env, key in (("HUGPY_GPU_MEM_GIB", "gpu_mem_gib"),
                     ("HUGPY_CPU_MEM_GIB", "cpu_mem_gib"),
                     ("HUGPY_N_CPU_MOE", "n_cpu_moe"),
                     ("HUGPY_ALLOC_MODE", "alloc_mode"),
                     # PER-GPU device pin (2026-09-25): central's chosen card ->
                     # the slot child's CUDA_VISIBLE_DEVICES (slot_agent reads
                     # body["gpu"]); its tensor-split -> --tensor-split. Without
                     # this the native slot child NEVER saw a per-request card and
                     # llama.cpp auto-split across all visible GPUs on its own.
                     ("HUGPY_MAIN_GPU", "gpu"),
                     ("HUGPY_MAIN_GPU", "main_gpu"),
                     ("HUGPY_TENSOR_SPLIT", "tensor_split"),
                     ("DEFAULT_LLAMA_THREADS", "threads")):
        value = os.environ.get(env)
        if value not in (None, ""):
            eff_opts.setdefault(key, value)
    raw_ngl = os.environ.get("HUGPY_N_GPU_LAYERS", "").strip().lower()
    if "n_gpu_layers" not in eff_opts and raw_ngl not in ("", "auto"):
        try:
            eff_opts["n_gpu_layers"] = (0 if raw_ngl in ("off", "cpu", "none")
                                         else int(raw_ngl))
        except ValueError:
            pass
    return eff_opts


def _drop_runner(model_key) -> None:
    """STALE-SLOT-FIX-20260910: a model swapped out of its seat must not keep a cached HTTP
    runner pointing at that seat - the seat's NEW occupant would answer for it."""
    if not model_key:
        return
    try:
        from hugpy_engine.llama.runners.get import evict_llama_runner
        evict_llama_runner(model_key)
    except Exception:  # noqa: BLE001
        logger.debug("drop runner for %s failed", model_key, exc_info=True)


def _alloc_body(requested: dict, source: dict | None, reload_reason: str | None) -> dict:
    """Additive /load body keys (an older slot agent ignores unknown keys)."""
    out = {"alloc_requested": dict(requested), "alloc_source": dict(source or {})}
    if reload_reason:
        out["reload_reason"] = reload_reason
    return out


class SlotPool:
    def __init__(self, urls: list[str] | None = None):
        self.urls = urls if urls is not None else slot_urls()

    def statuses(self) -> list[dict]:
        out = []
        for url in self.urls:
            try:
                status = _get(url + "/status")
                status["_control"] = url
            except Exception as exc:  # a down slot shouldn't break scheduling
                status = {"_control": url, "healthy": False, "model_key": None,
                          "error": str(exc)}
            out.append(status)
        return out

    def _ceiling_ok(self, model_key: str) -> bool:
        """Whether loading ``model_key`` keeps the card at/under the real-VRAM
        ceiling. True (fits) when no ceiling gate is registered — bare central /
        no-GPU / unmeasurable all degrade to the historical occupancy-only path.
        A gate that raises is treated as "can't tell" => True (never block a load
        because the measurement broke)."""
        if _FIT_CHECK is None:
            return True
        try:
            return bool(_FIT_CHECK(model_key))
        except Exception:  # noqa: BLE001 — a broken gate must not crash serving
            return True

    def _evict_coldest_on_demand(self, statuses: list[dict],
                                 model_key: str) -> "dict | None":
        """Evict the single LRU idle on-demand slot occupant (the SAME candidate
        rule the all-busy promotion branch uses) and return the evicted status
        dict, or None when nothing is evictable. Reuses ``_EVICTION_POLICY`` (the
        worker answers True only for on-demand — never static, never a busy slot,
        never the incoming model). No-op (None) when no eviction policy is
        registered."""
        if _EVICTION_POLICY is None:
            return None
        candidates = [s for s in statuses
                      if s.get("model_key") and s.get("healthy")
                      and not s.get("busy")
                      and s.get("model_key") != model_key]
        try:
            candidates = [s for s in candidates
                          if _EVICTION_POLICY(s["model_key"])]
        except Exception:  # noqa: BLE001 — a broken policy must not crash serving
            return None
        candidates.sort(key=lambda s: s.get("last_used") or 0)
        for victim in candidates:
            try:
                self.unload(victim["_control"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("ceiling evict failed on %s: %s",
                               victim["_control"], exc)
                continue
            _drop_runner(victim.get("model_key"))   # STALE-SLOT-FIX-20260910
            return victim
        return None

    @staticmethod
    def _polite_load() -> bool:
        """k96/k56: is the CURRENT load forbidden from evicting anyone?  True
        for a request-scoped ``no_makeroom`` (dispatch contextvar) or a spill
        ``no_evict`` (per-request env on a worker). Guarded — an unreadable
        flag is not a policy and reads as False (today's behavior)."""
        try:
            from hugpy_engine.dispatch.dispatch import no_makeroom_active
            if no_makeroom_active():
                return True
        except Exception:  # noqa: BLE001
            pass
        try:
            from hugpy_engine.spill import no_evict_env
            return no_evict_env()
        except Exception:  # noqa: BLE001
            return False

    def endpoint_for(self, model_key: str, *, load_timeout: float = 900.0,
                     opts: dict | None = None) -> str | None:
        """Resolve (and if needed load) the slot serving ``model_key``.

        ``opts`` is an optional per-load compute dict (n_gpu_layers, ctx,
        threads, cpus, gpu) applied when the model is loaded into a free slot.
        Returns the slot's inference endpoint, or None when every slot is busy
        with a different model (caller should fall back to swap).
        """
        # The opts actually applied to the /load. The cross-tier make-room hook
        # (below) may hand back a PARTIAL-offload plan for an oversize GGUF — the
        # honest layers-that-fit count — which we thread in here so the slot child
        # launches with --n-gpu-layers N (not the shard-blind autofit -1).
        # Every entry point (including serve_endpoint and stale-runner refresh)
        # must carry request placement to the separate slot process. Its boot
        # environment cannot see _apply_spill's per-model changes.
        eff_opts = env_request_opts(opts)
        # What THIS request asked for, captured BEFORE any make-room plan edits
        # eff_opts, plus who asked (designation / per-request / operator). The
        # seat records both at load; reuse compares against them.
        requested = alloc_signature(eff_opts)
        source = env_alloc_source() or {"kind": "designation" if _asked(requested) else "default"}
        reload_reason = None
        statuses = self.statuses()

        # 1. already serving OR currently loading it — reuse, never load a 2nd
        #    copy. A slot mid-load has model_key set but healthy=False; if we
        #    returned that endpoint immediately the caller would proxy to a child
        #    that isn't up yet and get a 503. So WAIT for the loading slot to go
        #    healthy (coalescing concurrent requests onto the one load), just like
        #    loading into a fresh slot blocks. A down slot reports model_key=None.
        for s in statuses:
            if s.get("model_key") != model_key:
                continue
            ep = s.get("endpoint") or s["_control"]
            if s.get("healthy"):
                why = alloc_mismatch(s, requested, source)
                if why is None and status_satisfies_opts(s, eff_opts):
                    return ep
                reload_reason = why or (
                    f"reloaded: resident seat (n_gpu_layers={s.get('n_gpu_layers')}, "
                    f"path={s.get('model_path')!r}) does not implement the requested "
                    f"load contract {eff_opts!r} ({describe_alloc_source(source)})")
                # An explicit load-time contract changed.  The old child cannot
                # morph in place; discard it and let the ordinary free-slot load
                # below recreate the seat with the requested quant/allocation.
                if s.get("busy"):
                    deadline = time.time() + load_timeout
                    while time.time() < deadline and s.get("busy"):
                        time.sleep(0.25)
                        s = _get(s["_control"] + "/status")
                    if s.get("busy"):
                        raise RuntimeError(
                            f"slot remained busy while changing the seat for {model_key}")
                logger.info("slot seat mismatch for %s on %s; evicting before reload "
                            "(current ngl=%r path=%r, requested=%r): %s", model_key,
                            s.get("_control"), s.get("n_gpu_layers"),
                            s.get("model_path"), eff_opts, reload_reason)
                self.unload(s["_control"])
                _drop_runner(model_key)
                statuses = self.statuses()
                break
            deadline = time.time() + load_timeout
            while time.time() < deadline:
                time.sleep(2.0)
                try:
                    st = _get(s["_control"] + "/status")
                except Exception:
                    break                       # slot went away — reload below
                if st.get("healthy"):
                    return st.get("endpoint") or ep
                if not st.get("model_key"):
                    break                       # load aborted/failed — reload below
            break                               # not ready in time — fall through

        # 1b. Real-VRAM CEILING gate (Fix A, 2026-07-15). Before we seat this
        #     model — whether into an idle slot below or by promotion — check
        #     that loading it keeps the card at/under the ~90% ceiling given REAL
        #     current free VRAM (not slot-occupancy count). This catches the ae
        #     case: a card 95%-full from a SEPARATE ComfyUI process but with an
        #     idle slot would otherwise /load happily, then OOM/under-offload.
        #     When over ceiling, evict the coldest on-demand occupant(s) via the
        #     SAME LRU mechanism the all-busy branch uses (re-reading statuses so
        #     each re-check sees the freed seat) until the gate passes OR nothing
        #     is evictable. Nothing evictable + still over ceiling => proceed
        #     anyway (honest-degrade: the child's autofit does its best; we never
        #     HANG a legitimate request), with a clear warning. No-op when no
        #     ceiling gate is registered (bare central / no-GPU / can't measure).
        if not self._ceiling_ok(model_key):
            # k96/k56 POLITE seat: over the ceiling and this load may not cost
            # anyone their seat. Ask the (polite-aware) make-room hook ONCE —
            # it may admit a free-room PARTIAL offload — otherwise FAIL FAST
            # with the capacity-class refusal the brain ladder walks, instead
            # of either evicting below or honest-degrading into the very
            # under-offload/OOM wedge no_makeroom exists to prevent.
            if self._polite_load():
                verdict = None
                if _MAKE_ROOM is not None:
                    verdict = _MAKE_ROOM(model_key)   # polite: never evicts;
                                                      # LoadRefusal propagates
                if (isinstance(verdict, dict)
                        and verdict.get("action") == "partial"
                        and verdict.get("n_gpu_layers") is not None):
                    eff_opts["n_gpu_layers"] = verdict["n_gpu_layers"]
                    if verdict.get("n_cpu_moe") is not None:
                        eff_opts["n_cpu_moe"] = verdict["n_cpu_moe"]
                    logger.info("polite seat: %s admitted into free room as a "
                                "partial offload (--n-gpu-layers %s)",
                                model_key, verdict["n_gpu_layers"])
                elif isinstance(verdict, dict) and verdict.get("action") == "proceed":
                    logger.info("polite seat: %s admitted into free room "
                                "(%s)", model_key, verdict.get("note") or "fits")
                else:
                    from hugpy_engine.dispatch.dispatch import LoadRefusal
                    raise LoadRefusal({
                        "reason": ("won't fit on GPU: seating %s would breach "
                                   "the VRAM ceiling and the polite flag "
                                   "(no_makeroom/no_evict) forbids evicting a "
                                   "resident — refusing without evicting"
                                   % model_key),
                        "model_key": model_key, "no_makeroom": True})
            else:
                while not self._ceiling_ok(model_key):
                    victim = self._evict_coldest_on_demand(statuses, model_key)
                    if victim is None:
                        # Slot-side eviction exhausted. CROSS-TIER (slice 10): an
                        # IN-PROCESS transformers resident (invisible to the slot
                        # scheduler) may still be squatting the card — the make-room
                        # hook evicts ALL permissible residents from the pid-registry
                        # measured truth. If it evicts something, re-check the ceiling;
                        # if it REFUSES (nothing left to evict), honest-degrade below.
                        if _MAKE_ROOM is not None:
                            try:
                                verdict = _MAKE_ROOM(model_key)
                            except Exception:  # noqa: BLE001 — never hang a request
                                verdict = None
                            # PARTIAL-offload admission (autofit's hybrid contract): the
                            # full weights don't fit even after eviction, but the honest
                            # layers-that-fit plan admits. Launch the child with that
                            # exact n_gpu_layers and stop looping the (full-need) ceiling
                            # check — it can never pass, and re-looping would spin.
                            if (isinstance(verdict, dict)
                                    and verdict.get("action") == "partial"
                                    and verdict.get("n_gpu_layers") is not None):
                                eff_opts["n_gpu_layers"] = verdict["n_gpu_layers"]
                                # MoE expert split (2026-07-24): the admission may
                                # answer the hybrid with -1 + n_cpu_moe instead of a
                                # layer count — thread it so the child launches with
                                # --n-cpu-moe (all layers on GPU, experts on CPU).
                                if verdict.get("n_cpu_moe") is not None:
                                    eff_opts["n_cpu_moe"] = verdict["n_cpu_moe"]
                                    logger.info(
                                        "VRAM ceiling: %s admitted as a MoE expert "
                                        "split — n_gpu_layers=%s, --n-cpu-moe %s "
                                        "(experts to CPU)", model_key,
                                        verdict["n_gpu_layers"], verdict["n_cpu_moe"])
                                else:
                                    logger.info(
                                        "VRAM ceiling: %s admitted as a PARTIAL offload — "
                                        "%s/%s layers on GPU (%s%%); launching child with "
                                        "--n-gpu-layers %s", model_key,
                                        verdict["n_gpu_layers"],
                                        (verdict.get("partial") or {}).get("total_layers"),
                                        verdict.get("gpu_pct"), verdict["n_gpu_layers"])
                                break
                            if isinstance(verdict, dict) and verdict.get("evicted"):
                                statuses = self.statuses()
                                continue         # re-check the ceiling with the freed room
                        logger.warning(
                            "VRAM ceiling: loading %s would exceed the real-VRAM "
                            "ceiling and nothing on-demand is evictable (slot or "
                            "in-process) — proceeding anyway (autofit will spill/"
                            "offload; not hanging the request)", model_key)
                        break
                    logger.info(
                        "VRAM ceiling: evicted idle on-demand %s from %s to keep %s "
                        "under the real-VRAM ceiling", victim["model_key"],
                        victim["_control"], model_key)
                    statuses = self.statuses()   # re-read: the seat is now free
        elif _MAKE_ROOM is not None and "n_gpu_layers" not in eff_opts:
            # EVICTION-AWARE AUTOFIT (2026-07-25). The ceiling gate PASSED, so
            # nothing must be evicted — but "fits" is not a placement. Left alone,
            # the slot child autofits its layer count against the VRAM free at
            # that instant, and a momentarily-busy card cripples the seat for the
            # life of the child (flux2-klein-9b: 21/36 layers with 3.1 GiB free —
            # a ~4x throughput loss that reads as healthy from central).
            #
            # So consult the admission anyway: it sizes the plan against free +
            # RECLAIMABLE, evicts to realise it, and RE-PLANS from what was
            # actually freed. It answers "partial" with an explicit layer count
            # ONLY when that beats the default autofit; otherwise it answers
            # "proceed" and we thread nothing (byte-identical to before). An
            # explicit operator n_gpu_layers in opts skips this entirely.
            try:
                verdict = _MAKE_ROOM(model_key)
            except Exception:  # noqa: BLE001 — never hang a request
                verdict = None
            if (isinstance(verdict, dict) and verdict.get("action") == "partial"
                    and verdict.get("n_gpu_layers") is not None
                    and verdict.get("size_up")):
                eff_opts["n_gpu_layers"] = verdict["n_gpu_layers"]
                logger.info("eviction-aware autofit: %s seats with "
                            "--n-gpu-layers %s — %s", model_key,
                            verdict["n_gpu_layers"], verdict.get("note"))
                statuses = self.statuses()   # any eviction changed the pool

        # 2. an idle slot (reachable, nothing loaded)
        for s in statuses:
            if "error" in s:
                continue
            if not s.get("model_key"):
                body = {"model_key": model_key, **eff_opts,
                        **_alloc_body(requested, source, reload_reason)}
                resp = _post(s["_control"] + "/load", body, load_timeout)
                if isinstance(resp, dict) and resp.get("error"):
                    # Typed (2026-09-23): a HARD loader rejection arrives as
                    # HardLoadFailure carrying the loader stderr + path, so
                    # get_llama_runner refuses the in-process fallback.
                    from hugpy_engine.serve.load_failure import from_slot_reply
                    raise from_slot_reply(resp["error"], resp, model_key=model_key,
                                          path=eff_opts.get("path"))
                return resp.get("endpoint") or s["_control"]

        # 3. everything busy — tiers-v2 promotion: bump the LRU *idle*
        #    occupant whose model is itself on-demand (policy hook). Never a
        #    busy slot, never a serving/static/pinned model (the policy
        #    returns False for those), never for a model already handled
        #    above. STATIC occupants are immovable by construction: the
        #    worker's policy answers True only for on-demand.
        #
        #    k96/k56 POLITE seat: promotion IS an eviction (the bumped
        #    occupant loses its seat), so a polite load may not use it. With
        #    every seat occupied, fail fast with the capacity-class refusal
        #    the brain ladder walks — never bump, never fall through to the
        #    swap path (whose unload is the same eviction by another name).
        if self._polite_load() and any(s.get("model_key") for s in statuses):
            from hugpy_engine.dispatch.dispatch import LoadRefusal
            raise LoadRefusal({
                "reason": ("won't fit on a slot: every seat is occupied and "
                           "the polite flag (no_makeroom/no_evict) forbids "
                           "bumping a resident to seat %s — refusing without "
                           "evicting" % model_key),
                "model_key": model_key, "no_makeroom": True})
        if _EVICTION_POLICY is not None:
            candidates = [s for s in statuses
                          if s.get("model_key") and s.get("healthy")
                          and not s.get("busy")
                          and s.get("model_key") != model_key]
            try:
                candidates = [s for s in candidates
                              if _EVICTION_POLICY(s["model_key"])]
            except Exception:  # noqa: BLE001 — a broken policy must not crash serving
                candidates = []
            candidates.sort(key=lambda s: s.get("last_used") or 0)
            for victim in candidates:
                logger.info(
                    "slot promotion: evicting idle on-demand %s from %s to "
                    "load %s", victim["model_key"], victim["_control"], model_key)
                try:
                    self.unload(victim["_control"])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("promotion evict failed on %s: %s",
                                   victim["_control"], exc)
                    continue
                _drop_runner(victim.get("model_key"))   # STALE-SLOT-FIX-20260910
                body = {"model_key": model_key, **eff_opts,
                        **_alloc_body(requested, source, reload_reason)}
                resp = _post(victim["_control"] + "/load", body, load_timeout)
                if isinstance(resp, dict) and resp.get("error"):
                    # Typed (2026-09-23): a HARD loader rejection arrives as
                    # HardLoadFailure carrying the loader stderr + path, so
                    # get_llama_runner refuses the in-process fallback.
                    from hugpy_engine.serve.load_failure import from_slot_reply
                    raise from_slot_reply(resp["error"], resp, model_key=model_key,
                                          path=eff_opts.get("path"))
                return resp.get("endpoint") or victim["_control"]

        # Static-lock check (tiers v3): when EVERY slot's occupant is static
        # (locked to its seat — never swapped out), returning None would lie
        # ("busy, try swap") about a pool that can never free a seat. Fail the
        # load with a clear, actionable error instead. A merely-busy pool
        # (any serving/on-demand/unknown occupant, or a down slot) keeps the
        # historical None -> swap/in-process fallback.
        if _RESIDENCY_LOOKUP is not None and statuses:
            occupants = [s.get("model_key") for s in statuses]
            if all(occupants):
                try:
                    all_static = all(_RESIDENCY_LOOKUP(mk) == "static"
                                     for mk in occupants)
                except Exception:  # noqa: BLE001 — a broken lookup must not crash serving
                    all_static = False
                if all_static:
                    raise RuntimeError(
                        f"all slots are static-locked — cannot seat {model_key}; "
                        "unlock a static model or raise the slot count")
        return None

    def load(self, model_key: str, control_url: str, *, timeout: float = 900.0) -> dict:
        return _post(control_url + "/load", {"model_key": model_key}, timeout)

    def unload(self, control_url: str) -> dict:
        return _post(control_url + "/unload", {}, 30.0)

    def overview(self) -> list[dict]:
        return self.statuses()


# --------------------------------------------------------------------------- #
# one-time install: N generic slot services (systemd template)                #
# --------------------------------------------------------------------------- #
def render_slot_unit(python_bin: str | None = None, user: str | None = None,
                     group: str | None = None, main_gpu: str | None = None) -> str:
    """A systemd TEMPLATE unit: ``abstract-hugpy-slot@.service``.

    The instance number is the slot id (``systemctl enable --now
    abstract-hugpy-slot@1``); the agent derives its port from SLOT_ID. Installed
    ONCE; thereafter the app drives slots over HTTP — no per-model units, no
    sudo at request time.
    """
    import sys
    python_bin = python_bin or sys.executable
    user = user or os.environ.get("LLAMA_SERVICE_USER", "solcatcher")
    group = group or os.environ.get("LLAMA_SERVICE_GROUP", "web")
    env_lines = [
        "Environment=SLOT_ID=%i",
        f"Environment=SLOT_PORT_BASE={_slot_port_base()}",
    ]
    if main_gpu is not None:
        env_lines.append(f"Environment=MAIN_GPU={main_gpu}")
    return "\n".join((
        "[Unit]",
        "Description=abstract_hugpy_dev model slot %i",
        "After=network.target",
        "StartLimitIntervalSec=120",
        "StartLimitBurst=5",
        "",
        "[Service]",
        "Type=simple",
        f"User={user}",
        f"Group={group}",
        *env_lines,
        f"ExecStart={python_bin} -m abstract_hugpy_dev.managers.serve.slot_agent",
        "Restart=always",
        "RestartSec=5",
        "TimeoutStopSec=60",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ))


def slot_install_steps(unit_dir: str = "/etc/systemd/system",
                       main_gpu: str | None = None) -> list[tuple[str, str]]:
    """Return [(kind, payload)] describing the one-time install:

    ('write', unit_text) for the template, then ('cmd', shell) lines to enable
    each slot. The caller writes/executes (with sudo); this stays side-effect
    free so it can be shown as a dry run.
    """
    import os.path as osp
    unit_path = osp.join(unit_dir, "abstract-hugpy-slot@.service")
    steps = [("write:" + unit_path, render_slot_unit(main_gpu=main_gpu)),
             ("cmd", "systemctl daemon-reload")]
    for i in range(_slot_count()):
        steps.append(("cmd", f"systemctl enable --now abstract-hugpy-slot@{i + 1}"))
    return steps
