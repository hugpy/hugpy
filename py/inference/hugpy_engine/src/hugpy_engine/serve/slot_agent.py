"""Slot supervisor — one generic, root-free model 'slot'.

A slot is a long-running service that owns a stable control port and runs ONE
``llama-server`` child for whatever model it's currently assigned. Assigning a
model autofits GPU layers from the VRAM still free at that moment, so a second
slot naturally takes whatever the first one left. The slot proxies the OpenAI
chat API to its child, so its own URL is a stable inference endpoint even across
child reloads.

No root / systemctl at request time: the app drives slots over HTTP (/load,
/unload, /status). You install N slot services ONCE (systemd template in
deploy/, or just run N of these), then the scheduler (:mod:`.slots`) assigns
models to them on demand.

Run one slot:

    SLOT_ID=1 SLOT_PORT=8101 python -m abstract_hugpy_dev.managers.serve.slot_agent

Env:
    SLOT_ID            label for this slot (default "1")
    SLOT_PORT          control + proxy port (default 8101)
    SLOT_CHILD_PORT    the llama-server child's port (default SLOT_PORT + 1000)
    SLOT_HOST          bind address (default 0.0.0.0)
    SLOT_ADVERTISE     host the scheduler should reach this slot on (default 127.0.0.1)
    MAIN_GPU           pin the child to this GPU index (sets CUDA_VISIBLE_DEVICES)
    SLOT_HEALTH_TIMEOUT seconds to wait for the child to come up (default 180)
"""
from __future__ import annotations

import collections
import logging
import os
import subprocess
import sys
import threading
import time

logger = logging.getLogger("abstract_hugpy_dev.slot_agent")

SLOT_ID = os.environ.get("SLOT_ID", "1")
SLOT_HOST = os.environ.get("SLOT_HOST", "0.0.0.0")
_PORT_BASE = int(os.environ.get("SLOT_PORT_BASE", "8101"))


def _default_port() -> int:
    # slot N -> base + (N-1), matching slots.slot_urls(); lets a systemd
    # template set only SLOT_ID=%i and derive the port from it.
    try:
        return _PORT_BASE + (int(SLOT_ID) - 1)
    except (TypeError, ValueError):
        return _PORT_BASE


SLOT_PORT = int(os.environ.get("SLOT_PORT", str(_default_port())))
SLOT_CHILD_PORT = int(os.environ.get("SLOT_CHILD_PORT", str(SLOT_PORT + 1000)))


# ── child-port collision guard (2026-09-25) ─────────────────────────────────
# The child (llama-server) port defaults to SLOT_PORT + 1000. With the shipped
# SLOT_PORT_BASE (8101) slot 1's control port is 8101 and its child lands on
# 9101 — exactly the worker's own port on a box whose unit sets --port 9101
# (seen on a-brain: the first slot's child tried to bind 9101 == the worker
# port → EADDRINUSE, no model ever loaded; the by-hand fix was SLOT_PORT_BASE=
# 8201). The worker port is authoritative and known here — the worker unit
# exports WORKER_PORT and the slot child runs on the SAME box — so the child
# must never sit on it. Relocate a colliding child port UP to the next free
# offset, loudly, and record it so the slot status shows the correction rather
# than a silent EADDRINUSE crash-loop. The CONTROL port is NOT relocated: the
# agent finds slots at slot_urls() == SLOT_PORT_BASE+i and must keep matching;
# a control/worker collision (8101 vs 9100 territory) is a genuine misconfig and
# is only logged.
def _worker_port() -> "int | None":
    raw = (os.environ.get("WORKER_PORT") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


# (from_port, to_port, reason) for every port this process moved off a collision.
PORT_RELOCATIONS: list = []


def _guard_child_port() -> None:
    """Ensure SLOT_CHILD_PORT never equals the worker port or this slot's own
    control port; relocate it upward and log loudly if it does. Idempotent."""
    global SLOT_CHILD_PORT
    wp = _worker_port()
    # A caller who set SLOT_CHILD_PORT explicitly still gets the guard — an
    # explicit collision is exactly the failure we exist to catch.
    reserved = {SLOT_PORT}
    reason_for = {SLOT_PORT: "slot control port"}
    if wp is not None:
        reserved.add(wp)
        reason_for[wp] = f"WORKER_PORT ({wp})"
    if SLOT_CHILD_PORT not in reserved:
        return
    clash = SLOT_CHILD_PORT
    new = clash + 1
    while new in reserved or new > 65535:
        new += 1
    logger.error("slot %s child port %d collides with %s — relocating llama-server "
                 "child to %d (set SLOT_CHILD_PORT / SLOT_PORT_BASE to silence)",
                 SLOT_ID, clash, reason_for.get(clash, "a reserved port"), new)
    PORT_RELOCATIONS.append((clash, new, reason_for.get(clash, "reserved")))
    SLOT_CHILD_PORT = new


_guard_child_port()

SLOT_ADVERTISE = os.environ.get("SLOT_ADVERTISE", "127.0.0.1")
MAIN_GPU = os.environ.get("MAIN_GPU")


def _as_int_or_none(v):
    """Parse an int (a --main-gpu index) or None — never raise into a load."""
    if v in (None, ""):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _parse_tensor_split(v):
    """A central tensor-split as a list of floats, from a list or a comma string
    (``HUGPY_TENSOR_SPLIT`` rides the wire as CSV). None when unset/unparseable —
    a bad value NEVER breaks a load, it just falls back to non-split placement."""
    if v in (None, ""):
        return None
    if isinstance(v, (list, tuple)):
        items = v
    else:
        items = [p for p in str(v).split(",") if p.strip()]
    out = []
    for p in items:
        try:
            out.append(float(p))
        except (TypeError, ValueError):
            return None
    return out or None
# The FLOOR for the load hard-cap (back-compat: was the whole deadline). A cold
# load gets at LEAST this long regardless of size.
HEALTH_TIMEOUT = float(os.environ.get("SLOT_HEALTH_TIMEOUT", "180"))
# STALL window (slice 12): the honest failure signal is a load making NO forward
# progress (child RSS not growing / VRAM not dropping) for this long — NOT a
# blind clock that ignores size. A 45.9G gguf cold-load off NVMe legitimately
# exceeds 180s; killing it at the old flat deadline re-pages 46G every retry (the
# ae thrash loop). We fail on STALL, not on the clock.
STALL_TIMEOUT = float(os.environ.get("SLOT_LOAD_STALL_TIMEOUT", "60"))
# The GENEROUS-BUT-BOUNDED hard cap: a truly-wedged child that somehow keeps
# nudging RSS must still die eventually. Size-scaled — base + expected_bytes /
# assumed throughput — with the HEALTH_TIMEOUT floor. Assumed effective cold-load
# throughput (bytes/s): NVMe read + CUDA upload + repack; conservative so the cap
# is generous. Override the divisor via env for a slow-disk box.
_LOAD_THROUGHPUT_BPS = float(
    os.environ.get("SLOT_LOAD_THROUGHPUT_MBPS", "200")) * 1024 * 1024  # 200 MB/s
_HARD_CAP_MULT = float(os.environ.get("SLOT_LOAD_HARD_CAP_MULT", "3.0"))
# Bytes-of-progress that count as "real" movement between samples (filter noise).
_PROGRESS_EPSILON = 8 * 1024 * 1024   # 8 MiB
# A model's consecutive genuine load-failure count is kept for the console/logs
# (the "attempt N" in the failure reason). It is informational ONLY — it never
# blocks a re-attempt: every /load actually tries to seat the model and, if it
# fails, fails with THAT attempt's real loader error. Success clears the count.


# ── Agent heartbeat nudge (2026-09-08) ──────────────────────────────────────
# The worker agent exports this URL when it spawns the pool. On every inflight
# 0↔1 edge the proxy fires ONE best-effort POST so the agent beats central with
# a genuinely fresh busy flag instead of waiting out the 15s cadence — the
# console's "answering" pill used to appear near the end of an answer and
# linger a beat after it. Off the request path by construction: a daemon
# thread, a short timeout, every error swallowed. Absent env (an ADOPTED orphan
# slot spawned by an older agent) degrades to cadence-only, exactly today.
_NUDGE_URL = os.environ.get("HUGPY_AGENT_NUDGE_URL")


def _nudge_agent_heartbeat(slot=None) -> None:
    if not _NUDGE_URL:
        return

    # Capture before the thread starts; a later finish must not be reported as
    # a delayed start, or a delayed start overwrite the finish.
    activity = ({"slot_id": SLOT_ID, "model_key": slot.model_key,
                 "busy": slot.inflight > 0, "observed_at": time.time()}
                if slot is not None else {})
    def _fire() -> None:
        try:
            import httpx
            httpx.post(_NUDGE_URL, json=activity, timeout=2.0)
        except Exception:  # noqa: BLE001 — telemetry must never cost a request
            pass

    threading.Thread(target=_fire, daemon=True, name="hb-nudge").start()


# ── Child IDENTITY verification (k53, 2026-07-31) ──────────────────────────
# The slot proxy forwards to whatever answers on SLOT_CHILD_PORT and labels the
# answer with the model the CALLER asked for. That is only true while the
# process on that port is the child we spawned. Observed: a Qwen2.5-3B process
# took over the port and coder-next requests were answered by it — a silent
# MODEL SUBSTITUTION, correct-looking at every layer above, invisible in every
# log. So before proxying we ask the process what it is serving and compare it
# to what we launched; a mismatch drops the stale mapping so the next request
# cold-loads properly instead of being quietly answered by a stranger.
#
# Cheap by construction: one local HTTP GET, cached per (pid, path) for
# _IDENTITY_RECHECK_S, skipped entirely while the child is mid-generation (a
# busy python child can't answer a probe, and a request in flight is already
# proof the port answers).
_IDENTITY_RECHECK_S = float(os.environ.get("SLOT_IDENTITY_RECHECK_S", "30"))


def _model_path_of(argv):
    """The model file an argv launches with (``-m`` / ``--model``), or None.
    Both child kinds are covered: the native llama-server takes ``-m``, the
    llama_cpp.server fallback ``--model``. ``--model`` is read FIRST because the
    python child's argv also carries a ``-m`` — python's own module flag
    (``python -m llama_cpp.server``), which is not a model file at all."""
    argv = argv or []
    for i, arg in enumerate(argv):
        if arg == "--model" and i + 1 < len(argv):
            return argv[i + 1]
    for i, arg in enumerate(argv):
        if arg == "-m" and i + 1 < len(argv):
            val = argv[i + 1]
            if os.sep in val or val.lower().endswith(".gguf"):
                return val
    return None


def _reported_model_id(doc):
    """The model identifier out of a llama-server ``/props`` or an OpenAI
    ``/v1/models`` document — the child's own statement of what it loaded.
    None when the document says nothing identifiable (never guess)."""
    if not isinstance(doc, dict):
        return None
    for key in ("model_path", "model"):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    gen = doc.get("default_generation_settings")
    if isinstance(gen, dict):
        for key in ("model_path", "model"):
            val = gen.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    data = doc.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        for key in ("id", "root"):
            val = data[0].get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _identity_matches(expected_path, model_key, reported) -> bool:
    """Whether ``reported`` (what the process on the port says it serves) is the
    model we launched. Deliberately LENIENT — the identifier may be an absolute
    path (native llama-server), a bare filename, or an alias (llama_cpp.server),
    and an unreadable/absent identifier is "can't tell", which must never be
    read as a mismatch. Only a POSITIVE, clearly-different name is a mismatch:
    that is the one case worth tearing a live seat down for."""
    rep = str(reported or "").strip().lower()
    if not rep:
        return True                              # said nothing -> can't tell
    rep_base = os.path.basename(rep)
    for candidate in (expected_path, model_key):
        cand = str(candidate or "").strip().lower()
        if not cand:
            continue
        cand_base = os.path.basename(cand)
        if not cand_base:
            continue
        if (cand_base == rep_base or cand_base in rep or rep_base in cand
                or os.path.splitext(cand_base)[0] in rep):
            return True
    return False


def _allowed_cpus():
    """Cores this slot's cgroup is confined to via systemd AllowedCPUs (kernel-
    enforced, un-escapable), or None when unconfined. Read-only; no root needed."""
    try:
        import subprocess
        out = subprocess.run(
            ["systemctl", "show", f"abstract-hugpy-slot@{SLOT_ID}.service",
             "-p", "AllowedCPUs", "--value"],
            capture_output=True, text=True, timeout=2)
        v = (out.stdout or "").strip()
        return v or None
    except Exception:
        return None


def _total_gguf_bytes(path):
    """Total on-disk size of a gguf, summing ALL shards when ``path`` is one shard
    of a split model (``…-00001-of-00004.gguf``). The resolver only ever hands us
    the FIRST shard, so a naive getsize under-counts a multi-shard model ~3-4x."""
    try:
        if not path or not os.path.isfile(path):
            return None
        import re
        import glob
        base = os.path.basename(path)
        m = re.search(r"-\d{5}-of-(\d{5})\.gguf$", base)
        if m:
            patt = f"{base[:m.start()]}-*-of-{m.group(1)}.gguf"
            shards = [s for s in glob.glob(os.path.join(os.path.dirname(path), patt))
                      if os.path.isfile(s)]
            if shards:
                return sum(os.path.getsize(s) for s in shards)
        return os.path.getsize(path)
    except Exception:
        return None


def _mem_available_bytes():
    """This node's reclaim-inclusive free RAM (MemAvailable)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _total_ram_bytes_best_effort():
    """This node's installed RAM (MemTotal) — the ceiling MMAP-eligible bytes
    (MoE expert tensors) are priced against, since file-backed page cache is
    reclaimable and never an OOM-able reservation. None when unmeasurable."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _model_expected_bytes(model_key):
    """Rough 'total RAM required' for a model ~= its full GGUF size on disk (ALL
    shards summed). Denominator for the load-progress % AND the CPU preflight in
    _build_cmd. Approximate (mmap/repack land resident RAM a bit under file size)."""
    try:
        from hugpy_engine.serve.serve import _model_file_for, get_model_config
        cfg = get_model_config(model_key)
        return _total_gguf_bytes(_model_file_for(model_key, cfg))
    except Exception:
        return None


def _proc_rss_bytes(pid):
    """Resident RAM (bytes) of a pid — the slot's llama-server child footprint."""
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def _proc_rss_detail(pid):
    """The HONEST resident-memory split of a pid from /proc/<pid>/status:
    ``{rss_anon_bytes, rss_file_bytes, rss_shmem_bytes}``.

    llama.cpp mmaps the GGUF, so VmRSS counts the FILE-BACKED pages of the
    weights as "resident" — reclaimable page cache, NOT pinned RAM. Measured on
    ae (Qwen3-Coder-Next, 17/48 offload): VmRSS 45.2G but RssAnon only 1.5G,
    RssFile 43.6G — the raw figure overstates true memory pressure ~28x.
    RssAnon is the honest pinned figure; RssFile is the mmap'd/cache share.

    Best-effort + Linux-only: ``{}`` on any read failure (an old kernel without
    the Rss* split, a vanished pid, a non-Linux box) — callers OMIT the fields
    rather than crash the heartbeat. ``rss_bytes`` (VmRSS) keeps its meaning
    unchanged for wire back-compat."""
    out = {}
    keys = {"RssAnon:": "rss_anon_bytes", "RssFile:": "rss_file_bytes",
            "RssShmem:": "rss_shmem_bytes"}
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                for pref, name in keys.items():
                    if line.startswith(pref):
                        out[name] = int(line.split()[1]) * 1024
                        break
    except Exception:
        return {}
    return out


def _cpus_to_hexmask(cpus: str) -> str:
    """Turn a cpu spec like "0-3" or "0,2,4" into llama.cpp's hex --cpu-mask."""
    bits = 0
    for part in str(cpus).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            for c in range(int(lo), int(hi) + 1):
                bits |= 1 << c
        else:
            bits |= 1 << int(part)
    return format(bits, "x") if bits else ""


def _central_url() -> str | None:
    """Where this slot can reach central to learn/pull request-time models.
    Agent-managed slots inherit WORKER_CENTRAL_URL from the agent's env;
    central's own systemd slots serve on-disk models and set neither."""
    return os.environ.get("WORKER_CENTRAL_URL") or os.environ.get("CENTRAL_URL")


def _resolve_cfg(model_key, central_url):
    """``get_model_config``, but if THIS (slot) process's static registry has
    never heard of the model — because central registered it at request time,
    and the slot is a separate process — learn it from central first via the
    SAME ensure-registered path the agent runs. Fixes slots 503'ing on
    request-time-provisioned models (the "op flux" saga)."""
    from hugpy_engine.serve.serve import get_model_config
    try:
        return get_model_config(model_key)
    except Exception:
        if not central_url:
            raise
    try:
        from hugpy_storage.provision import ensure_model_registered
        ensure_model_registered(model_key, central_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("slot %s: ensure_model_registered(%s) failed: %s",
                       SLOT_ID, model_key, exc)
    return get_model_config(model_key)   # raise cleanly if still unknown


def _ensure_present(model_key, central_url):
    """Pull a registered-but-absent model's files (request-time model) the same
    way the agent does, before we hand llama.cpp a path. Fast no-op when the
    model is already local (the common case: the agent pre-ensures before /load)."""
    try:
        from hugpy_storage.provision import ensure_model_present
        ensure_model_present(model_key, central_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("slot %s: ensure_model_present(%s) failed: %s",
                       SLOT_ID, model_key, exc)


# llama-server --n-cpu-moe support probe (MoE expert split, 2026-07-24). The
# flag pattern-matches expert FFN tensors (ffn_(up|down|gate)_exps) onto CPU
# buffers; it exists in every llama.cpp new enough to serve the fleet's MoE
# models (proven live on ae via its env alias LLAMA_ARG_N_CPU_MOE=999 — the
# arg parser maps that env 1:1 onto --n-cpu-moe). An OLDER binary would reject
# the unknown flag at spawn, so we probe `--help` ONCE per binary path and
# degrade (omit the arg + log once) when it predates the flag.
_SERVER_FLAG_CACHE: dict = {}
_MOE_DEGRADE_LOGGED = set()


def _server_supports_flag(server_bin, flag) -> bool:
    key = (str(server_bin), str(flag))
    cached = _SERVER_FLAG_CACHE.get(key)
    if cached is not None:
        return cached
    ok = False
    try:
        out = subprocess.run([server_bin, "--help"], capture_output=True,
                             text=True, timeout=15)
        ok = flag in ((out.stdout or "") + (out.stderr or ""))
    except Exception:  # noqa: BLE001 — can't probe -> assume unsupported (omit arg)
        ok = False
    _SERVER_FLAG_CACHE[key] = ok
    return ok


def _log_moe_degrade_once(key, msg):
    if key not in _MOE_DEGRADE_LOGGED:
        _MOE_DEGRADE_LOGGED.add(key)
        logger.warning(msg)


class _NglDefaulted(int):
    """An ``n_gpu_layers`` value that NOBODY ASKED FOR — a fill-in default that
    a caller (serve.ServeSpec / the worker's spill env) materialized only
    because the field had to hold *some* int.

    Why a subclass of ``int`` and not ``None``: ``-1`` is overloaded. It is the
    historical fill-in default (``DEFAULT_LLAMA_NGL``) AND a load-bearing
    explicit force ("Max GPU": ``managers.llama.runners.get``, ``alloc_modes``
    gpu-only, ``chaos/sweep``, ``chaos/assortment``, ``worker_agent`` MoE
    commit, ``overrides``). Aliasing ``-1`` to "unset" would silently regress
    every one of those. Passing ``None`` instead is not always available either
    — ``ServeSpec.n_gpu_layers`` is typed ``int`` and several consumers format
    it straight into argv.

    So: the VALUE still behaves exactly like the int it wraps (arithmetic,
    comparison, ``str()``, ``int()``, argv formatting — all identical, so every
    existing consumer is byte-for-byte unaffected), while carrying one extra
    bit of provenance that placement policy can read via :func:`_ngl_is_unset`.
    Nothing serializes this type: ``json.dumps`` renders it as the plain int,
    so the central->worker relay wire (extra=forbid) is untouched.
    """
    __slots__ = ()

    def __repr__(self):  # pragma: no cover - debugging aid only
        return f"<defaulted ngl {int(self)}>"

    # int() has no distinct __str__ — it falls back to __repr__. Without this,
    # str(_NglDefaulted(-1)) would render "<defaulted ngl -1>" and _build_cmd
    # formats ngl straight into argv (`"--n-gpu-layers", str(ngl)`), which would
    # launch llama-server with a garbage flag value. _effective_ngl() int()s
    # before argv today, so this is defence-in-depth — but the whole premise of
    # this subclass is that the VALUE behaves exactly like the int it wraps.
    def __str__(self):
        return str(int(self))


def _ngl_is_unset(requested) -> bool:
    """True when no caller actually DEMANDED a layer placement — either nothing
    was passed at all (``None``) or the value is a :class:`_NglDefaulted`
    fill-in. This — not ``requested is None`` — is the gate that auto placement
    policy (the detected-MoE expert split) must consult; otherwise a spec whose
    n_gpu_layers is merely ``DEFAULT_LLAMA_NGL`` reads as "operator demanded all
    layers on the GPU" and a 48 GB MoE launches onto a 24 GB card with no
    ``--n-cpu-moe`` (ae, 2026-07-25)."""
    return requested is None or isinstance(requested, _NglDefaulted)


def _effective_ngl(requested, auto):
    """Override-wins-over-autofit (k14). An EXPLICIT ``n_gpu_layers`` request WINS
    over the autofit — that is the lever the offload speed-cliff sweep (k7) needs:
    seat a GGUF at full offload, then relaunch it DOWN through decreasing layer
    counts. ``None``/absent => autofit, exactly as today.

    NOTE the sentinel: ``None`` is autofit; every integer is an override, INCLUDING
    the live console designations ``-1`` ("Max GPU" — force all layers) and ``0``
    ("CPU only"). ``-1`` is NOT an autofit alias here (managers.llama.runners.get
    ships ``n_gpu_layers=-1`` to the slot precisely to FORCE all layers; aliasing
    it to autofit would silently regress that path). The sweep therefore asks for
    autofit with ``None`` at the top of the ramp and explicit non-negative counts
    below it — it never needs ``-1`` to mean autofit.

    ADDENDUM (2026-07-25): a ``_NglDefaulted`` value is a fill-in nobody asked
    for, so it too resolves to autofit here. The VALUE of a defaulted -1 and an
    explicit -1 is the same int; only the provenance differs, and only the
    unset kind may be overridden by autofit / the MoE auto-split."""
    return auto if _ngl_is_unset(requested) else int(requested)


class GpuOnlyInfeasible(RuntimeError):
    """gpu-only was selected for a MoE whose FULL footprint does not fit the
    card. The mode's contract is all-or-bust, so this is a refusal, never a
    quiet expert split — see :func:`_strict_gpu_only`."""


def _strict_gpu_only(n_gpu_layers) -> bool:
    """Is THIS load the k37 ``gpu-only`` mode — ALL tensors on the card,
    experts included, bust if they don't fit (operator ruling 2026-07-31:
    "gpu-only must be honest to its name")?

    Two spellings reach the slot and both are the SAME operator statement:
      * ``HUGPY_ALLOC_MODE=gpu-only`` — the mode named outright;
      * an EXPLICIT ``n_gpu_layers=-1`` — gpu-only's frozen wire encoding
        (``alloc_modes.mode_to_spill``: gpu-only -> ``{"n_gpu_layers": -1}``),
        which is exactly what ``derive_alloc_mode`` reads back as gpu-only.

    A DEFAULTED -1 (:class:`_NglDefaulted` — the ServeSpec fill-in nobody
    asked for) is NOT a statement: it stays the blank max-gpu default and keeps
    the dense-first split. That distinction is the whole reason the provenance
    marker exists.

    ⚠ THIS NARROWS k53 (da2b5d9), deliberately and by ruling. That slice read a
    stored ``{"n_gpu_layers": -1}`` as a max-GPU PREFERENCE so it could not
    disable the split — because the alternative it produced was the 17/48 LAYER
    hybrid (31 layers of dense attention on the CPU), which is never right. The
    ruling does not restore that hybrid: a gpu-only MoE that fits launches
    WHOLE, and one that doesn't is REFUSED with max-gpu named as the remedy. So
    dense-on-CPU is still impossible; what changes is that "only" stops
    silently meaning "mostly"."""
    from hugpy_engine.spill import alloc_mode_env
    if alloc_mode_env() == "gpu-only":
        return True
    if _ngl_is_unset(n_gpu_layers):
        return False
    try:
        return int(n_gpu_layers) == -1
    except (TypeError, ValueError):
        return False


def _gpu_only_moe_plan(model_key, moe, budget, why):
    """The gpu-only placement for a detected MoE: the WHOLE model on the card,
    or an honest refusal. Never a silent expert split (operator ruling
    2026-07-31 — the split is max-gpu's contract, not this one's).

    ``budget`` is :func:`_moe_gpu_budget`'s verdict — ``None`` (unmeasurable
    card), ``0`` (no GPU to spend), or budgetable VRAM. Returns a plan shaped
    like :func:`spill.moe_dense_first_plan`'s (``n_cpu_moe`` 0, stated
    explicitly so an inherited ``LLAMA_ARG_N_CPU_MOE`` cannot flip it), or
    raises :class:`GpuOnlyInfeasible`."""
    from hugpy_engine.spill import moe_dense_first_plan
    if budget is None:
        # UNMEASURABLE CARD. We cannot prove it doesn't fit, and a refusal
        # invented from missing data is what degrade-not-guess forbids
        # everywhere else. So obey the mode literally — everything on the card,
        # stated — and let llama.cpp bust honestly if the card turns out too
        # small. That bust IS gpu-only's contract, which is why the blank
        # default's all-experts-to-CPU degrade would be the wrong answer here:
        # it would serve a split nobody asked for under the name "only".
        return {"n_cpu_moe": 0, "cpu_bytes": 0, "dense_fits": True}
    plan = moe_dense_first_plan(moe, budget) if budget else None
    if plan is not None and int(plan.get("n_cpu_moe") or 0) == 0:
        return plan                          # everything fits: n_cpu_moe 0
    weights = (int(moe.get("non_expert_bytes") or 0)
               + int(moe.get("expert_bytes") or 0))
    raise GpuOnlyInfeasible(
        f"{model_key}: gpu-only needs the WHOLE model on the card "
        f"(~{weights / 1e9:.1f} GB of weights, experts included) but this "
        f"load's VRAM budget after the context/projector reserve is "
        f"~{int(budget) / 1e9:.1f} GB ({why}) — gpu-only is all-or-bust by "
        "contract, so it will NOT be quietly served as a partial expert split. "
        "Select max-gpu for this model (as much GPU as fits, the rest spilled "
        "to RAM), free the card, or pick a smaller quant.")


def _moe_gpu_budget(path, n_gpu_layers, free_cap, extra_reserve_bytes):
    """The VRAM budget THIS load may spend, as the active allocation mode
    defines it — the input to the dense-backbone-first plan (k53).

    Returns ``(bytes, why)``:
      * ``0``    — there is nothing for a split to plan: the mode grants NO GPU
        (ram-only / an explicit ``n_gpu_layers=0`` / a max-ram fill that holds
        the whole model in RAM), or a caller stated an explicit POSITIVE layer
        count, which is the k14/k7 override lever (seat at full offload, then
        relaunch DOWN through layer counts) and stays obeyed verbatim. Either
        way the placement is byte-identical to before.
      * ``None`` — the budget is UNMEASURABLE (no VRAM reading on this box).
        Never invent a placement on missing data; the caller degrades.
      * a positive int — budgetable free VRAM, capped by the model's own
        ``gpu_mem_gib`` contract when it has one, minus what already has to land
        on the card beside the weights (mmproj projector + KV/context reserve).

    Every mode that reaches a positive number (or an unmeasurable card) gets the
    SAME treatment — dense backbone first, experts with the remainder (operator
    ruling 2026-07-31). The mode decides HOW MUCH card the model may claim; it
    never decides that the dense backbone goes second. An explicit ``-1`` is
    deliberately NOT a per-layer instruction that disables the split; it is the
    gpu-only wire, and gpu-only SPENDS this same budget — on the whole model or
    not at all (:func:`_gpu_only_moe_plan`), never on a partial expert split."""
    from hugpy_engine.spill import alloc_mode_env, free_vram_bytes, maxram_gpu_layers
    mode = alloc_mode_env()
    if n_gpu_layers not in (None, "") and not _ngl_is_unset(n_gpu_layers):
        try:
            demanded = int(n_gpu_layers)
        except (TypeError, ValueError):
            demanded = None
        if demanded == 0:
            return 0, "explicit n_gpu_layers=0 (ram-only / CPU-only)"
        if demanded is not None and demanded > 0:
            return 0, (f"explicit n_gpu_layers={demanded} — a stated layer "
                       "count is obeyed verbatim (the k14/k7 offload lever)")
    if mode == "max-ram":
        # max-ram fills RAM first and only the OVERFLOW reaches the GPU. No
        # overflow == no GPU budget; an overflow means the card IS in play, and
        # what it should hold first is the dense backbone.
        try:
            if int(maxram_gpu_layers(path)) <= 0:
                return 0, "max-ram holds the whole model in RAM"
        except Exception:  # noqa: BLE001 — unpriceable fill: fall through
            pass
    budget = free_vram_bytes()
    if free_cap is not None:
        budget = min(budget, free_cap) if budget else free_cap
    if not budget:
        return None, "VRAM budget unmeasurable on this box"
    budget -= int(extra_reserve_bytes or 0)
    if budget <= 0:
        return 0, "the context/projector reserve consumes the whole VRAM budget"
    return int(budget), (f"{mode or 'max-gpu'} budget "
                         f"{budget / 2 ** 30:.2f} GiB")


def _slot_parallel(ctx=None, model_bytes=None):
    """Concurrent sequences per slot (llama-server --parallel), DERIVED per node.

    HUGPY_SLOT_PARALLEL forces a value (bypasses derivation). Otherwise derive
    from THIS card's free-VRAM headroom after the model, budgeting a ctx-scaled
    KV cache per sequence, capped by CPU cores and a ceiling — so a 4x3090 box
    earns more concurrency than an 8GB 4060, with no per-node hardcoding. Every
    constant below is an env-overridable FALLBACK, never a hardcode; anything
    unprobeable degrades to 1 (today's single-sequence behavior)."""
    env = os.environ.get("HUGPY_SLOT_PARALLEL")
    if env:
        try:
            return max(1, int(env))
        except (TypeError, ValueError):
            pass

    def _envi(name, default):
        try:
            return int(os.environ.get(name) or default)
        except (TypeError, ValueError):
            return default

    def _envf(name, default):
        try:
            return float(os.environ.get(name) or default)
        except (TypeError, ValueError):
            return default

    ceiling = _envi("HUGPY_SLOT_PARALLEL_MAX", 8)
    kv_frac = _envf("HUGPY_SLOT_KV_FRACTION", 0.7)
    kv_per_tok = _envi("HUGPY_SLOT_KV_BYTES_PER_TOKEN", 160 * 1024)
    try:
        from hugpy_engine.spill import free_vram_bytes as _fvb
        free = int(_fvb() or 0)                      # free VRAM bytes on this card
        if free <= 0:
            return 1                                 # unprobeable -> safe fallback
        headroom = max(0, free - int(model_bytes or 0)) * kv_frac
        per_seq = max(1, int(ctx or 4096)) * kv_per_tok
        n = 1 + int(headroom // per_seq)
        cores = os.cpu_count() or 1
        n = min(n, max(1, cores // 2), ceiling)      # CPU + ceiling caps
        return max(1, n)
    except Exception:
        return 1                                     # hardcoded fallback


def _build_cmd(model_key, n_gpu_layers=None, ctx=None, threads=None, cpus=None,
               path=None, gpu_mem_gib=None, cpu_mem_gib=None, profile_bin=None,
               n_cpu_moe=None, tensor_split=None, main_gpu=None):
    """argv for the child llama-server + the resolved (ngl, ctx, threads, cpus).

    ``n_cpu_moe`` (MoE expert split, 2026-07-24; DEFAULT since 2026-07-25):
    number of MoE layers whose EXPERT tensors stay on CPU (999/
    spill.MOE_ALL_LAYERS = all), emitted as llama-server ``--n-cpu-moe``.
    Explicit (per-load opts / persisted override) always wins. Absent -> the
    AUTO policy, DENSE BACKBONE FIRST (operator ruling 2026-07-31): a detected
    MoE is served with n_gpu_layers=-1 and a --n-cpu-moe threshold computed so
    the always-hot non-expert tensors take the GPU budget FIRST and the expert
    FFN tensors get only the remainder — under every allocation mode that grants
    any GPU budget. A budget that only covers the backbone lands on --n-cpu-moe
    999 (the measured ae default: +59% tok/s at 5x less VRAM); a budget that
    covers everything lands on 0. Modes
    that grant no GPU at all (ram-only / explicit ngl=0 / a max-ram fill that
    holds the whole model in RAM) plan no split, and dense models never see the
    flag — both byte-identical to today. An expert share that can't fit RAM ->
    degrade to the autofit layer placement.

    ``gpu-only`` IS THE ONE EXCEPTION (operator ruling 2026-07-31, k55), and it
    is an exception to the REMAINDER, not to the ordering: "only" means ALL
    tensors on the card, experts included. The same budget is priced, but the
    answer is binary — the whole model fits (``--n-cpu-moe 0``, stated
    explicitly so an inherited ``LLAMA_ARG_N_CPU_MOE`` cannot flip it) or the
    load is REFUSED with max-gpu named as the remedy (:class:`GpuOnlyInfeasible`
    via :func:`_gpu_only_moe_plan`). It is never quietly served as a partial
    split, which would be the mode redefining its own name. The gpu-only wire is
    an EXPLICIT ``n_gpu_layers: -1`` (a DEFAULTED -1 is still the blank
    max-gpu default and still splits).

    This function is THE choke point for every slot child spawn — /load, k14
    /relaunch, and direct slot loads all funnel through here — so the policy
    holds for all of them.

    ``profile_bin`` (env-profiles stage 1): when the agent seats a model
    attributed to a dependency profile, it hands the profile venv's bin dir here.
    The PYTHON-launched child (``python -m llama_cpp.server`` — the fallback when
    no native llama-server binary exists) is then spawned from THAT venv's
    interpreter instead of the agent's, isolating the model's extra deps at the
    process seam. The native-binary child is unaffected in argv (its binary is
    resolved by the engine resolver); its PATH still prefers the profile bin via
    the child env (see ``Slot.load``), so a profile-shipped binary would win.
    """
    from hugpy_engine.serve.serve import (
        _model_file_for,
        _ctx_for,
        get_model_config,
        LLAMA_SERVER_BIN,
        DEFAULT_LLAMA_THREADS,
    )
    from hugpy_engine.spill import autofit_gpu_layers, vision_projector_bytes

    cfg = None
    if path and os.path.isfile(path):
        # Caller-resolved path (the worker agent registers central's models
        # IN-MEMORY, which a slot — a separate process — never sees; the agent
        # therefore resolves key→file itself and hands us the path).
        pass
    else:
        central_url = _central_url()
        cfg = _resolve_cfg(model_key, central_url)      # ensure-registered fallback
        path = _model_file_for(model_key, cfg)
        # Registered but files absent/partial (request-time model) — pull them
        # the same way the agent does before spawning llama.cpp.
        if central_url and (not path or not os.path.isfile(path)
                            or os.path.getsize(path) == 0):
            _ensure_present(model_key, central_url)
            path = _model_file_for(model_key, cfg)
    # Existence AND non-empty: a 0-byte or truncated GGUF (interrupted pull)
    # passes an isfile check but SIGILLs llama.cpp's native loader on spawn —
    # fail cleanly here instead of core-dumping the child.
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        # Say WHY on the telemetry stream before raising. This refusal is the
        # LAST honest checkpoint before llama.cpp — provisioning claimed to be
        # done (or was never asked) and yet there is nothing loadable at the
        # path we would hand the loader. Without this event the console showed
        # a load that simply never produced an eviction pass, which reads as
        # "nothing happened" (2026-07-28). Observation only; the raise below is
        # unchanged.
        try:
            from hugpy_engine.placement import get_eviction_ledger
            _ev = get_eviction_ledger()
            if not path:
                _why = "no GGUF resolved for this model on disk"
            elif not os.path.isfile(path):
                _why = "resolved GGUF does not exist"
            else:
                _why = "resolved GGUF is 0 bytes (interrupted or failed pull)"
            _ev.emit_resolve_fail(model_key, path, _why,
                                  **_ev.disk_stats(path or None))
        except Exception:  # noqa: BLE001 — telemetry never blocks a refusal
            pass
        raise FileNotFoundError(
            f"{model_key}: no usable GGUF on disk (resolved {path!r}) — missing "
            "or empty; refusing to spawn llama.cpp (would SIGILL)")

    # Serve from the box-local NVMe HOT-CACHE tier when this model is warmed there
    # (NVMe-fast); otherwise this kicks a background shared->hot promotion and
    # returns the shared path for this (cold) load, so the next load is fast.
    # Never blocks. The hot_cache tier (HUGPY_HOT_CACHE_ROOT) is the general
    # main-catalog mechanism; the legacy model_cache (HUGPY_MODEL_CACHE) is kept
    # as a fallback so a box still on the old env is not regressed. Neither env
    # set -> path is returned unchanged (byte-identical behaviour).
    _shared_path = path            # pre-hot-cache path (projector fallback below)
    try:
        from hugpy_engine.serve import hot_cache
        if hot_cache.enabled():
            path = hot_cache.use(path)
        else:
            from hugpy_engine.serve import model_cache
            path = model_cache.use(path)
    except Exception as exc:
        logger.warning("hot-cache unavailable (%s); loading from %s", exc, path)

    # Autofit from the VRAM free RIGHT NOW, so later slots take what's left.
    # An explicit per-model VRAM budget (gpu_mem_gib) caps what autofit may
    # plan with — the model's contract, not the card's whole remainder.
    free_cap = None
    if gpu_mem_gib not in (None, ""):
        try:
            free_cap = int(float(gpu_mem_gib) * 2**30)
        except (TypeError, ValueError):
            free_cap = None
    # Vision GGUF: the mmproj/CLIP projector loads onto the GPU beside the
    # offloaded layers, so reserve its VRAM BEFORE fitting layers — otherwise on
    # an 8 GB card autofit plans "all layers" against the model file alone and the
    # child then OOMs when the ~1.3 GB projector lands on top. 0 for text models
    # (byte-identical to before).
    _mmproj_reserve = vision_projector_bytes(path)
    if not _mmproj_reserve and _shared_path != path:
        _mmproj_reserve = vision_projector_bytes(_shared_path)
    # Resolve the SERVED ctx BEFORE fitting layers. The VRAM a llama_context
    # costs is linear in n_ctx (the KV cache), so autofit must price the context
    # THIS CHILD will actually run with rather than a flat constant — see
    # spill.vram_ctx_reserve_bytes. Pure move of the line that was below; nothing
    # between here and the old position reads ctx.
    _ctx_given = bool(ctx)                    # caller stated the served ctx
    ctx = int(ctx) if ctx else (_ctx_for(cfg, model_key) if cfg is not None else 4096)
    # The same context reserve autofit charges internally, made explicit here:
    # the MoE dense-first plan (below) must price the KV cache against the card
    # before it hands a single byte to the expert tensors.
    try:
        from hugpy_engine.spill import vram_ctx_reserve_bytes
        _ctx_reserve = int(vram_ctx_reserve_bytes(path, ctx)[0])
    except Exception:  # noqa: BLE001 — never block a load on a reserve probe
        _ctx_reserve = 0
    if free_cap is not None:
        from hugpy_engine.spill import free_vram_bytes as _fvb
        fv = _fvb()
        auto = autofit_gpu_layers(path, free_vram=min(fv, free_cap) if fv else free_cap,
                                  extra_reserve_bytes=_mmproj_reserve, n_ctx=ctx)
    else:
        auto = autofit_gpu_layers(path, extra_reserve_bytes=_mmproj_reserve,
                                  n_ctx=ctx)
    ngl = _effective_ngl(n_gpu_layers, auto)
    threads = int(threads) if threads else DEFAULT_LLAMA_THREADS
    cpus = str(cpus).strip() if cpus not in (None, "") else None

    # The model's TOTAL layer count (GGUF header block_count) — the denominator
    # for the console's "17/48 layers" AND the hybrid test for the MoE policy.
    try:
        from hugpy_engine.spill import _gguf_layer_count
        total_layers = _gguf_layer_count(path)
    except Exception:  # noqa: BLE001 — never block a load on header metadata
        total_layers = None

    # ── MoE expert split — DENSE BACKBONE FIRST (operator ruling 2026-07-31) ─
    # Decide the effective --n-cpu-moe BEFORE the engine branch so both child
    # kinds can degrade honestly. moe_mode: "explicit" (per-load opts/override —
    # always wins), "auto" (detected-MoE, the dense-first plan applied: ngl=-1 +
    # a computed threshold), or None (dense model / no GPU budget / a stated
    # layer count / experts don't fit RAM — byte-identical to before).
    #
    # The 2026-07-25 policy made the split the DEFAULT (measured on ae's
    # coder-next: +59% tok/s at 5x less VRAM vs the 17/48 layer hybrid), but the
    # mode/ngl gates still decided WHETHER it happened at all. That is what
    # stranded ae again: a stored {"n_gpu_layers": -1} — the ROUTINE stamp, 40 of
    # 43 persisted allocations there carry exactly it, from the console's Max GPU
    # button / bulk-allocate / reconcile — read as an explicit demand, DISABLED
    # the split, and llama.cpp answered with the layer hybrid: 31 layers of dense
    # attention on the CPU while expert weights sat on the card. Dense bytes are
    # touched by EVERY token, expert bytes by ~10/512 of one, so that ordering is
    # never right.
    #
    # The inversion is now impossible: for ANY MoE, EVERY mode that grants a GPU
    # budget spends it on the dense backbone FIRST and gives the experts only the
    # remainder (spill.moe_dense_first_plan computes the --n-cpu-moe threshold;
    # _moe_gpu_budget says how much card the mode grants). A mode still decides
    # HOW MUCH card the model may claim — it can no longer decide that the
    # always-hot bytes go second.
    #
    #   budget == 0     ram-only / explicit ngl=0 / a max-ram fill that holds
    #                   the whole model in RAM -> nothing to prioritize, no
    #                   split (byte-identical to before).
    #   budget unknown  no VRAM reading on this box -> the historical degrade:
    #                   the blank default still splits all experts to CPU (the
    #                   measured coder-next win); an EXPLICIT ngl/mode is obeyed
    #                   verbatim, because inventing a placement from missing
    #                   data is exactly what we refuse to do everywhere else.
    #   budget > 0      dense first, experts fill the remainder. A full card
    #                   yields n_cpu_moe=0 (everything on the GPU, stated
    #                   explicitly so an inherited LLAMA_ARG_N_CPU_MOE cannot
    #                   flip it); a small one yields a partial threshold; a card
    #                   that only holds the backbone yields MOE_ALL_LAYERS.
    #
    # NOT changed (each an operator DEMAND about placement): an explicit
    # n_cpu_moe (incl. an explicit 0 = "experts on GPU") wins absolutely, and
    # dense models never see the flag at all.
    #
    # gpu-only IS THE EXCEPTION TO THE REMAINDER (operator ruling 2026-07-31,
    # k55): "only" means every tensor on the card. Same budget, binary answer —
    # fits whole -> --n-cpu-moe 0; doesn't -> GpuOnlyInfeasible naming max-gpu
    # as the remedy. This narrows the paragraph above for exactly one mode: the
    # stored -1 no longer produces the LAYER hybrid (that inversion stays
    # impossible) but it no longer produces a silent expert split either — the
    # mode that promises the whole card either delivers it or says so.
    eff_n_cpu_moe = None
    moe_mode = None
    moe_cpu_bytes = None                     # expert bytes the plan sends to RAM
    moe_budget_priced = False                # the plan spent a MEASURED budget
    moe_fallback_ngl = ngl                   # what we revert to if we must degrade
    if n_cpu_moe not in (None, ""):
        try:
            eff_n_cpu_moe = int(n_cpu_moe)
            moe_mode = "explicit"
        except (TypeError, ValueError):
            eff_n_cpu_moe = None
    else:
        try:
            from hugpy_engine.spill import gguf_moe_detail, MOE_ALL_LAYERS, moe_dense_first_plan
            moe = gguf_moe_detail(path)
            if moe.get("is_moe"):
                budget, why = _moe_gpu_budget(path, n_gpu_layers, free_cap,
                                              _mmproj_reserve + _ctx_reserve)
                plan = None
                if _strict_gpu_only(n_gpu_layers):
                    # gpu-only: the card takes EVERYTHING or the load is
                    # refused (2026-07-31). No expert ever lands in RAM under
                    # this mode, so there is no dense-first REMAINDER to
                    # compute — only a yes/no against the same budget.
                    plan = _gpu_only_moe_plan(model_key, moe, budget, why)
                    moe_budget_priced = bool(budget)
                elif budget:
                    plan = moe_dense_first_plan(moe, budget)
                    moe_budget_priced = True
                    # k64 (operator ruling 2026-07-31, declare-need doctrine):
                    # a budget priced from MOMENTARY free VRAM — max-gpu / the
                    # blank default / a max-ram overflow, i.e. any load WITHOUT
                    # a stated per-model contract (gpu_mem_gib) — buys the
                    # dense backbone and the KV cache, never the experts. The
                    # remainder-fill that k53 ratified let an empty card
                    # promote ALL experts (n_cpu_moe=0): coder-next then
                    # launched whole at 19.6 GiB, rode the 90% ceiling, and
                    # the idle-pressure sweep evicted it minutes later — every
                    # call repaid a full load. Expert promotion is an operator
                    # DEMAND (explicit n_cpu_moe, gpu-only, or an explicit
                    # gpu_mem_gib contract, which keeps the k53 remainder
                    # fill below); free VRAM at load instant is not one.
                    if plan is not None and free_cap is None \
                            and int(plan["n_cpu_moe"]) != MOE_ALL_LAYERS:
                        plan = dict(plan, n_cpu_moe=MOE_ALL_LAYERS,
                                    expert_layers_on_gpu=0,
                                    cpu_bytes=int(moe.get("expert_bytes") or 0),
                                    gpu_bytes=int(moe.get("non_expert_bytes")
                                                  or 0))
                elif budget is None:
                    # Unmeasurable card: keep the measured default (backbone on
                    # the GPU, ALL experts to CPU — +59% tok/s at 5x less VRAM
                    # on ae's coder-next). It is the one placement that cannot
                    # invert the ordering no matter what the card turns out to
                    # hold, so it is also the honest degrade.
                    plan = {"n_cpu_moe": MOE_ALL_LAYERS,
                            "cpu_bytes": int(moe.get("expert_bytes") or 0)}
                if plan is None:
                    logger.info("slot %s: %s is MoE but no expert split is "
                                "planned — %s", SLOT_ID, model_key, why)
                else:
                    # Viability, the ONE remaining condition: the expert share
                    # the plan sends to host RAM must actually fit budgetable
                    # RAM. When it clearly doesn't, degrade to whatever autofit
                    # decided — never refuse, and never move bytes into RAM that
                    # isn't there. Unmeasurable RAM -> proceed (degrade
                    # honestly: never block on missing data).
                    need_ram = int(plan.get("cpu_bytes") or 0)
                    avail = _mem_available_bytes()
                    if avail:
                        from hugpy_engine.spill import ram_reserve_bytes
                        avail = max(0, avail - ram_reserve_bytes())
                    if need_ram and avail and need_ram > avail * 0.95:
                        _log_moe_degrade_once(
                            ("ram", model_key),
                            f"slot {SLOT_ID}: {model_key} is MoE but the expert "
                            f"tensors this plan spills (~{need_ram / 1e9:.1f} GB) "
                            f"exceed budgetable RAM (~{avail / 1e9:.1f} GB) — "
                            "keeping the autofit layer placement instead")
                    else:
                        ngl = -1
                        eff_n_cpu_moe = int(plan["n_cpu_moe"])
                        moe_cpu_bytes = need_ram
                        moe_mode = "auto"
                        if not plan.get("dense_fits", True):
                            _log_moe_degrade_once(
                                ("dense", model_key),
                                f"slot {SLOT_ID}: {model_key}'s dense backbone "
                                f"(~{int(moe.get('non_expert_bytes') or 0) / 1e9:.1f} GB) "
                                f"is larger than this load's VRAM budget — the "
                                "backbone still goes first (llama.cpp spills "
                                "what cannot fit), experts stay in RAM")
        except GpuOnlyInfeasible:
            # The ONE refusal this block may raise, and it must survive the
            # catch-all below: "MoE policy never blocks a load" is about probes
            # and pricing gaps, not about an operator mode whose whole contract
            # is to bust rather than place bytes it was told not to place.
            raise
        except Exception:  # noqa: BLE001 — MoE policy must never block a load
            eff_n_cpu_moe = None
            moe_mode = None
            moe_cpu_bytes = None
            moe_budget_priced = False

    # An explicit MoE split fixes the weights that land on the card; size the
    # served ctx to the VRAM LEFT after them, here in the launcher, so a planner
    # with a different ctx view can never produce an impossible launch
    # (2026-09-25: central planned coder-next's split for 32K, the slot then
    # launched -c 262144 -> "failed to create context", SIGSEGV).
    if eff_n_cpu_moe and moe_mode == "explicit" and not _ctx_given:
        try:
            from hugpy_engine.spill import (free_vram_bytes, gguf_moe_detail,
                                            moe_split_need, served_ctx_for_fit)
            _split = moe_split_need(gguf_moe_detail(path), int(eff_n_cpu_moe))
            if _split and _split.get("gpu_bytes"):
                _fit = served_ctx_for_fit(path, free_vram=free_vram_bytes(),
                                          weights_on_gpu_bytes=int(_split["gpu_bytes"]),
                                          extra_reserve_bytes=_mmproj_reserve)
                if _fit and int(_fit) < int(ctx):
                    logger.info("slot %s: %s ctx %s -> %s to fit beside %.1f GiB "
                                "of split weights (--n-cpu-moe %s)", SLOT_ID,
                                model_key, ctx, _fit,
                                int(_split["gpu_bytes"]) / 2 ** 30, eff_n_cpu_moe)
                    ctx = int(_fit)
        except Exception:  # noqa: BLE001 — keep the resolved ctx
            pass

    # Preflight: when nothing can offload to GPU (auto<=0 — e.g. no GPU on this
    # node) the weights are CPU-RAM-resident, so a model bigger than free RAM will
    # OOM mid-load. Sum ALL shards (the resolved path is only shard 1) and fail
    # fast with a clear message instead of letting RSS climb into an OOM 500.
    # Per-model RAM budget (cpu_mem_gib): the CPU-resident share must fit the
    # model's OWN allowance, not just whatever the box has free.
    if cpu_mem_gib not in (None, ""):
        try:
            from hugpy_engine.spill import cpu_resident_bytes
            ram_budget = float(cpu_mem_gib) * (2 ** 30)
            ngl_eff = ngl                      # the already-resolved effective ngl
            if eff_n_cpu_moe and moe_mode in ("auto", "explicit"):
                # MoE split: the CPU-resident share is the expert tensors the
                # split actually spills, not a layer fraction (ngl=-1 would
                # otherwise read as 0 CPU bytes). The auto path already priced
                # its own plan (dense-first, experts of the first N blocks);
                # an explicit n_cpu_moe is priced per layer at exactly that N
                # (spill.moe_split_need) — only the experts of blocks < N land
                # in RAM. Pricing the WHOLE expert set here refused central's
                # card contract (--n-cpu-moe 28, 29.8 GiB RAM budget) for
                # coder-next as "43.6 GiB CPU-resident" (2026-09-25).
                if moe_cpu_bytes is not None:
                    need_cpu = moe_cpu_bytes
                else:
                    try:
                        from hugpy_engine.spill import gguf_moe_detail, moe_split_need
                        _det = gguf_moe_detail(path)
                        _split = moe_split_need(_det, int(eff_n_cpu_moe))
                        need_cpu = int((_split or {}).get("cpu_bytes")
                                       or _det.get("expert_bytes") or 0)
                    except Exception:  # noqa: BLE001
                        need_cpu = 0
            else:
                need_cpu = cpu_resident_bytes(path, int(ngl_eff)) or 0
            if need_cpu > ram_budget:
                raise RuntimeError(
                    f"{model_key}: CPU-resident share ~{need_cpu / (2 ** 30):.1f} GiB exceeds "
                    f"this model's RAM budget ({float(cpu_mem_gib):.1f} GiB) — raise the "
                    "budget, offload more layers, or pick a smaller quant")
        except (TypeError, ValueError):
            pass

    if ngl == 0:                              # effective ngl, not raw autofit —
        # an explicit n_gpu_layers=-1 ("max GPU") that overrode a broken auto=0
        # is GPU-resident (not CPU-RAM-resident), so it must skip this refusal
        # and fall through to the inverse VRAM preflight below instead — the
        # comment above predates that guard; -1 is excluded here for real now.
        need = _total_gguf_bytes(path)
        avail = _mem_available_bytes()
        if avail:
            # Honor the operator RAM reserve (HUGPY_RAM_RESERVE_GIB) so slot
            # loads leave headroom for processes central can't see.
            from hugpy_engine.spill import ram_reserve_bytes
            avail = max(0, avail - ram_reserve_bytes())
        if need and avail and need > avail * 0.95:
            raise RuntimeError(
                f"{model_key}: needs ~{need / 1e9:.1f} GB RAM (all shards) but only "
                f"{avail / 1e9:.1f} GB budgetable (after reserve) with no GPU offload "
                f"on this node — free RAM (recycle the API worker) or pick a smaller quant")

    # Expert-RAM preflight for an EXPLICIT split (2026-07-25). The auto branch
    # above checks that the expert tensors actually fit host RAM before it
    # chooses the MoE split (and degrades to the layer split when they don't),
    # but an explicit operator/central-supplied ``n_cpu_moe`` skips that branch
    # entirely — and the per-model ``cpu_mem_gib`` preflight only runs when a
    # budget is set (ae's slot carries None). Narrowing the RAM guard below from
    # ``ngl <= 0`` to ``ngl == 0`` (correct: -1 is GPU-resident) means an
    # explicit split with oversized experts would otherwise reach llama-server
    # unchecked and OOM mid-load — the same silent stall this whole preflight
    # block exists to prevent, just via RAM instead of VRAM. Same honest-degrade
    # doctrine: only refuse when BOTH numbers are actually measurable.
    if eff_n_cpu_moe and moe_mode == "explicit":
        try:
            from hugpy_engine.spill import gguf_moe_detail, ram_reserve_bytes
            _exp = int(gguf_moe_detail(path).get("expert_bytes") or 0)
            # EXPERT BYTES ARE MMAP-ELIGIBLE (2026-08-28, worker analogue of
            # the 0.1.237 central ruling): they are file-backed page cache the
            # kernel reclaims at will, so the honest ceiling is the BOX (total
            # RAM), not the momentary MemAvailable — which on ae was mostly
            # page cache this very load would reclaim, refusing a split the
            # box demonstrably serves. Momentary available stays the fallback
            # when the total is unmeasurable.
            _avail = _total_ram_bytes_best_effort() or _mem_available_bytes()
            if _avail:
                _avail = max(0, _avail - ram_reserve_bytes())
            if _exp and _avail and _exp > _avail * 0.95:
                raise RuntimeError(
                    f"{model_key}: the requested MoE expert split (--n-cpu-moe "
                    f"{eff_n_cpu_moe}) puts ~{_exp / 1e9:.1f} GB of expert tensors in "
                    f"host RAM, but only ~{_avail / 1e9:.1f} GB is budgetable (after "
                    "reserve) — lower n_cpu_moe to keep more experts on the GPU, "
                    "free RAM, or pick a smaller quant")
        except RuntimeError:
            raise
        except Exception:  # noqa: BLE001 — never block a load on a probe failure
            pass

    # Inverse preflight (2026-07-25): ngl==-1 with NO MoE/expert offload means
    # every layer's weights are meant to land on the GPU. If the model's total
    # bytes (ALL shards) clearly exceed the card's VRAM, this is not a slow
    # load — it is an impossible one, and the child would hang health-checking
    # for ~10 minutes across retries before falling back (the coder-next/ae
    # incident this preflight exists to catch: 48.4GB MoE model, -1 with no
    # --n-cpu-moe, on a 24GB 3090). Fail fast with the same tone as the RAM
    # refusal above, INSTEAD of the stall.
    #
    # Deliberately narrow: only fires when (a) effective ngl is -1 (full GPU
    # residency requested/decided), (b) no MoE split is configured — an expert
    # split changes what "total bytes" means (a large chunk is meant for CPU,
    # so total-vs-VRAM is the wrong comparison; the MoE branches above already
    # own that math), and (c) both total-bytes and VRAM are actually
    # measurable — an unmeasurable card (no GPU visibility, no nvidia-smi,
    # etc.) must never block a load on missing data (degrade honestly, same
    # doctrine as every other best-effort probe in this module).
    # (d) the AUTO dense-first plan is exempt WHEN IT PRICED A REAL BUDGET: an
    # ``n_cpu_moe`` of 0 from that plan is not "no split" — it is the plan
    # STATING that the whole model fits the budget it just measured, so
    # re-litigating it here would refuse a load the placement engine already
    # sized. A gpu-only plan made against an UNMEASURABLE budget priced
    # nothing, so it stays subject to this check: free VRAM being unreadable
    # does not mean the card's TOTAL is, and "48 GB model onto a 24 GB card"
    # is exactly the stall this preflight exists to catch.
    if ngl == -1 and not eff_n_cpu_moe and not (moe_mode == "auto"
                                                and moe_budget_priced):
        need_vram = _total_gguf_bytes(path)
        try:
            from hugpy_engine.spill import total_vram_bytes as _tvb, free_vram_bytes as _fvb
            card_total = _tvb()
            card_free = _fvb()
        except Exception:  # noqa: BLE001 — never block a load on a probe failure
            card_total = card_free = None
        # Use whichever VRAM figure is available (prefer free — the live
        # budget — falling back to total when free is unmeasurable but total
        # isn't); either is sufficient to catch "model is multiples of the
        # whole card". A clear margin (>1.15x) avoids false refusals from
        # measurement noise/reserve slop right at the boundary.
        card_vram = card_free if card_free is not None else card_total
        if need_vram and card_vram and need_vram > card_vram * 1.15:
            raise RuntimeError(
                f"{model_key}: needs ~{need_vram / 1e9:.1f} GB VRAM (all shards, "
                f"n_gpu_layers=-1 with no MoE expert split) but this node's GPU "
                f"has only ~{card_vram / 1e9:.1f} GB {'free' if card_free is not None else 'total'} "
                "— this would stall for minutes before falling back. Configure "
                "an MoE expert split (n_cpu_moe) for this model, offload fewer "
                "layers, or pick a smaller quant")

    import shutil
    server_bin = LLAMA_SERVER_BIN if (LLAMA_SERVER_BIN and (
        os.path.isfile(LLAMA_SERVER_BIN) or shutil.which(LLAMA_SERVER_BIN))) else None

    if server_bin:
        # MoE flag support: an older llama-server predating --n-cpu-moe would
        # reject the unknown flag at spawn. Probe once per binary; degrade by
        # OMITTING the arg (+ one log). An AUTO-applied split also reverts to
        # the layer-split hybrid it replaced (never launch -1 without the
        # expert override — that would be the OOM the policy exists to avoid).
        if eff_n_cpu_moe is not None and not _server_supports_flag(
                server_bin, "--n-cpu-moe"):
            _log_moe_degrade_once(
                ("flag", server_bin),
                f"slot {SLOT_ID}: installed llama-server ({server_bin}) predates "
                "--n-cpu-moe — omitting the MoE expert split (layer-split/plain "
                "behavior stands); update the engine (`hugpy install-engine`) "
                "to enable it")
            if moe_mode == "auto":
                ngl = moe_fallback_ngl
            eff_n_cpu_moe = None
            moe_mode = None
        # k64: llama-server grew params-fit and CHANGED -ngl's vocabulary in
        # the same release: ``-1`` now parses as "auto — let --fit (default ON)
        # plan placement against device memory", and "all layers" is spelled
        # ``all`` (-2). Every ``-1`` this choke point emits MEANS "all layers
        # on the card" (the k53 dense-first plan pairs it with --n-cpu-moe;
        # gpu-only demands it outright), so on a fit-capable binary the old
        # spelling silently handed placement back to llama.cpp's own autofit —
        # the exact ae repro: "common_init_result: fitting params to device
        # memory", whole MoE on the card, the plan discarded. Translate the
        # spelling AND turn --fit off: placement decided here is deterministic
        # (same mode + same card state => same argv), and llama.cpp's fit
        # deliberately fills the card to a ~1 GiB margin — under the worker's
        # ~10% pressure ceiling, so a fit-planned child is evicted on the next
        # headroom sweep by construction. An old binary (no --fit in --help)
        # keeps the historical argv byte-identical.
        fit_capable = _server_supports_flag(server_bin, "--fit")
        _ngl_arg = "all" if (fit_capable and int(ngl) == -1) else str(int(ngl))
        # Concurrency, DERIVED per node (see _slot_parallel). n_par==1 keeps the
        # historical single-sequence argv byte-identical. n_par>1 serves N
        # concurrent requests on ONE loaded model via continuous batching instead
        # of serializing them; -c is scaled by N so each sequence keeps its full
        # context (the derivation already budgeted the ~Nx KV). Docs #performance.
        _model_bytes = os.path.getsize(path) if (path and os.path.isfile(path)) else None
        n_par = _slot_parallel(ctx=ctx, model_bytes=_model_bytes)
        c_val = ctx if n_par <= 1 else int(ctx) * n_par
        argv = [
            server_bin, "-m", path,
            "--host", "127.0.0.1", "--port", str(SLOT_CHILD_PORT),
            "--n-gpu-layers", _ngl_arg, "-c", str(c_val), "-t", str(threads),
        ]
        # PER-GPU tensor split (2026-09-25): central decided this GGUF spans >= 2
        # LOCAL cards. CUDA_VISIBLE_DEVICES is intentionally NOT restricted for a
        # split (see Slot.load), so the proportions + --main-gpu are GLOBAL device
        # indices. --tensor-split maps positionally over the visible GPUs; a 0
        # share excludes a card. Only emitted when the server supports the flag,
        # so an older binary degrades to its own auto placement (never a crash).
        _ts = _parse_tensor_split(tensor_split)
        if _ts and sum(1 for x in _ts if x and x > 0) >= 2 \
                and _server_supports_flag(server_bin, "--tensor-split"):
            argv += ["--tensor-split", ",".join(str(x) for x in _ts)]
            _mg = _as_int_or_none(main_gpu)
            if _mg is not None and _server_supports_flag(server_bin, "--main-gpu"):
                argv += ["--main-gpu", str(_mg)]
            logger.info("slot %s: %s multi-GPU tensor-split %s (main-gpu=%s)",
                        SLOT_ID, model_key, _ts, _mg)
        if n_par > 1 and _server_supports_flag(server_bin, "--parallel"):
            argv += ["--parallel", str(n_par)]
            if _server_supports_flag(server_bin, "--cont-batching"):
                argv += ["--cont-batching"]
            logger.info("slot %s: %s concurrency --parallel %s cont-batching -c %s",
                        SLOT_ID, model_key, n_par, c_val)
        if (os.environ.get("HUGPY_SLOT_FLASH_ATTN", "").strip().lower()
                in ("1", "true", "yes", "on")
                and _server_supports_flag(server_bin, "--flash-attn")):
            argv += ["--flash-attn", "on"]
        if fit_capable:
            argv += ["--fit", "off"]
        if eff_n_cpu_moe is not None:
            # Explicit argv beats any inherited LLAMA_ARG_N_CPU_MOE env (the
            # transition-era ae unit hack), so the launched split is
            # deterministic regardless of the unit's environment.
            argv += ["--n-cpu-moe", str(eff_n_cpu_moe)]
            if moe_mode in ("auto", "explicit"):
                logger.info(
                    "slot %s: %s MoE split (%s) — n_gpu_layers=%s, experts of "
                    "%s layer(s) on CPU (--n-cpu-moe)", SLOT_ID, model_key,
                    moe_mode, ngl, eff_n_cpu_moe)
        if cpus:
            # Soft pin via llama.cpp's own affinity (taskset is escaped by llama.cpp's
            # per-thread sched_setaffinity). For HARD, kernel-enforced dedication use
            # hugpy-slot-cpus -> cgroup AllowedCPUs. "0-3" / "0,2,4" -> hex mask.
            mask = _cpus_to_hexmask(cpus)
            if mask:
                argv += ["--cpu-mask", mask, "--cpu-strict", "1"]
        # Vision GGUF: load the multimodal projector so /v1/chat/completions accepts
        # image_url content. No-op for text models (no projector beside the model).
        from hugpy_platform.utils import find_mmproj
        # The hot-cache copy carries only *mmproj*-named sidecars; a projector
        # identified by its GGUF header (arch 'clip') under another name stays
        # in the shared dir — fall back to it so the child never launches a
        # vision model without its projector.
        mmproj = find_mmproj(path) or (
            find_mmproj(_shared_path) if _shared_path != path else None)
        if mmproj:
            argv += ["--mmproj", mmproj]
            logger.info("slot %s: vision model — loading projector %s", SLOT_ID, mmproj)
    else:
        # No C++ llama-server on this box (typical for WORKERS): fall back to
        # the OpenAI-compatible server inside llama-cpp-python — the engine
        # every node already has, so there's no binary to distribute or build.
        # Same /v1 surface, so the slot proxy needs no changes. No --cpu-mask
        # equivalent (threads + the unit's cgroup govern CPU); vision models
        # stay on the native/in-process path (no --mmproj here).
        #
        # Vision GGUF: REFUSE rather than seat. llama_cpp.server cannot load
        # the mmproj projector, so a seated vision model would answer every
        # image turn text-blind with no error anywhere. The raised reason
        # propagates verbatim to central's slot-refusal log and load_reports,
        # and get_llama_runner falls through to the native/in-process path.
        from hugpy_platform.utils import find_mmproj
        if find_mmproj(path):
            raise RuntimeError(
                f"{model_key}: vision_needs_slot — vision model (mmproj sidecar present) but this "
                "box has no native llama-server (LLAMA_SERVER_BIN) — the "
                "llama_cpp.server fallback cannot load the projector, images "
                "would be silently ignored. Install/point to a llama-server "
                "build (`hugpy install-engine`) to seat vision models.")
        # MoE expert split: llama_cpp.server (the python fallback) has no
        # --n-cpu-moe equivalent. Degrade: omit + one log; an AUTO-applied
        # split reverts to the layer-split hybrid (never -1 without the
        # expert override).
        if eff_n_cpu_moe is not None:
            _log_moe_degrade_once(
                ("python-child", model_key),
                f"slot {SLOT_ID}: {model_key} wants a MoE expert split "
                "(--n-cpu-moe) but this box serves via llama_cpp.server, which "
                "cannot express it — keeping the layer-split behavior; install "
                "a native llama-server (`hugpy install-engine`) for the "
                "measured MoE speedup")
            if moe_mode == "auto":
                ngl = moe_fallback_ngl
            eff_n_cpu_moe = None
            moe_mode = None
        import sys as _sys
        # Env-profiles (stage 1): launch this python child from the profile
        # venv's interpreter when one is attributed (raises errors-as-data if the
        # profile venv python is missing — never a silent shared-venv fallback).
        from hugpy_engine.serve import profiles as _profiles
        child_py = _profiles.child_python(profile_bin, _sys.executable)
        if profile_bin:
            logger.info("slot %s: %s uses dependency profile venv %s (child %s)",
                        SLOT_ID, model_key, profile_bin, child_py)
        argv = [
            child_py, "-m", "llama_cpp.server",
            "--model", path,
            "--host", "127.0.0.1", "--port", str(SLOT_CHILD_PORT),
            "--n_gpu_layers", str(ngl), "--n_ctx", str(ctx),
            "--n_threads", str(threads),
        ]
        if cpus:
            logger.info("slot %s: cpu pin %r ignored in llama_cpp.server mode",
                        SLOT_ID, cpus)
    # total_layers was read above (needed by the MoE hybrid test); it rides the
    # return so the console can render "17/48 layers" instead of "17/undefined".
    return (argv, ngl, ctx, threads, cpus,
            ("binary" if server_bin else "python"), total_layers, eff_n_cpu_moe)


# STALE-SLOT-FIX-20260910: the proxy must only answer for the model the caller asked for.
def _requested_model(req):
    """The 'model' the caller named in an OpenAI-style JSON body, or None."""
    try:
        body = req.get_json(silent=True, cache=True) or {}
    except Exception:  # noqa: BLE001
        return None
    m = body.get("model") if isinstance(body, dict) else None
    return str(m) if m else None


def _model_matches(slot, wanted):
    """True when `wanted` names this slot's seated model (key, file path, or basename)."""
    import os as _os
    if not slot.model_key:
        return False
    if wanted == slot.model_key:
        return True
    path = slot.model_path or ""
    base = _os.path.basename(path)
    return bool(base) and wanted in (path, base, base.rsplit(".", 1)[0])


# --------------------------------------------------------------------------- #
# Loader stderr tail (t140, 2026-09-15). The child's stderr used to be INHERITED
# (Popen(argv, env=env)), so llama-server's own verdict — "tensor 'blk.61...'
# data is not within the file bounds" for a truncated shard, "unknown model
# architecture", "cudaMalloc failed: out of memory" — only ever reached the
# worker journal, while the slot's load() reported the generic "model file was
# rejected by the loader ... see the worker journal". Central and the console
# saw the generic line, never the cause. Now stderr is PIPED through a reader
# thread that (a) echoes every line to our own stderr so the journal is
# unchanged, and (b) keeps the last _STDERR_TAIL_LINES in a ring so a failed
# load can quote the loader's real error. Never blocks: a full pipe is drained
# continuously, and a Popen stub without .stderr (tests) is simply skipped.
# --------------------------------------------------------------------------- #
_STDERR_TAIL_LINES = 60
_STDERR_EXCERPT_LINES = 6
_STDERR_EXCERPT_CHARS = 240
_STDERR_ERR_HINTS = ("error", "fail", "invalid", "unable", "cannot", "corrupt",
                     "truncat", "unexpected", "wrong", "mismatch", "unknown",
                     "abort", "out of memory", "not within", "bounds")


def _load_log_path(model_key: str) -> "str | None":
    """Per-load log file for the child's FULL stderr stream (2026-09-23):
    ``PROJECTS_HOME/logs/<worker>/<model>/<ts>-slot<id>.log``. The in-memory
    ring only feeds the short excerpt; this file is the whole log, and its
    path travels as ``log_ref`` beside the full text. None when unwritable."""
    try:
        try:
            from hugpy_platform.constants import PROJECTS_HOME as _ph
        except Exception:  # noqa: BLE001 — layout fallback, as serve/overrides.py
            _ph = os.environ.get("PROJECTS_HOME")
        if not _ph:
            return None
        import socket as _socket
        worker = (os.environ.get("WORKER_NAME") or _socket.gethostname() or "worker")
        safe = lambda v: "".join(c if (c.isalnum() or c in "._-") else "_"
                                 for c in str(v)) or "_"
        d = os.path.join(str(_ph), "logs", safe(worker), safe(model_key))
        os.makedirs(d, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S", time.localtime())
        return os.path.join(d, f"{ts}-slot{safe(SLOT_ID)}.log")
    except Exception:  # noqa: BLE001 — a log path never blocks a load
        return None


def _read_log_file(path) -> "str | None":
    """The whole per-load log, verbatim. None when absent/unreadable/empty."""
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except Exception:  # noqa: BLE001
        return None
    return text if text.strip() else None


def _start_stderr_tail(proc, sink: "collections.deque",
                       log_path: "str | None" = None) -> "threading.Thread | None":
    """Tee ``proc.stderr`` into ``sink`` (ring), our own stderr (journal) and,
    when given, ``log_path`` (the FULL stream, every byte, persisted)."""
    pipe = getattr(proc, "stderr", None)
    if pipe is None:
        return None

    def _pump():
        fh = None
        if log_path:
            try:
                fh = open(log_path, "ab")
            except Exception:  # noqa: BLE001 — the ring + journal still work
                fh = None
        try:
            for raw in iter(pipe.readline, b""):
                try:
                    sys.stderr.buffer.write(raw)
                    sys.stderr.buffer.flush()
                except Exception:  # noqa: BLE001 — journal echo is best-effort
                    pass
                if fh is not None:
                    try:
                        fh.write(raw)
                        fh.flush()
                    except Exception:  # noqa: BLE001
                        pass
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    sink.append(line)
        except Exception:  # noqa: BLE001 — a dead pipe just ends the tail
            pass
        finally:
            try:
                pipe.close()
            except Exception:  # noqa: BLE001
                pass
            if fh is not None:
                try:
                    fh.close()
                except Exception:  # noqa: BLE001
                    pass

    t = threading.Thread(target=_pump, name="slot-stderr-tail", daemon=True)
    t.start()
    return t


def _stderr_excerpt(sink, max_lines: int = _STDERR_EXCERPT_LINES,
                    max_chars: int = _STDERR_EXCERPT_CHARS) -> str:
    """The loader's own words: error-looking lines first (the LAST ones — the
    verdict comes after the diagnostics), else the last few lines verbatim."""
    lines = [l for l in (sink or ()) if l and l.strip()]
    if not lines:
        return ""
    hits = [l for l in lines if any(h in l.lower() for h in _STDERR_ERR_HINTS)]
    pick = (hits or lines)[-max_lines:]
    out = []
    for l in pick:
        l = " ".join(l.split())
        out.append(l if len(l) <= max_chars else l[:max_chars - 1] + "…")
    return " | ".join(out)


def _stderr_verbatim(sink, log_path: "str | None" = None) -> "str | None":
    """The loader's WHOLE stderr, verbatim (2026-09-23: no filter, no cap).
    Read from the per-load log file when one was written; else every line the
    in-memory ring still holds. None when empty."""
    full = _read_log_file(log_path)
    if full is not None:
        return full
    lines = [l for l in (sink or ()) if l and l.strip()]
    return "\n".join(lines) if lines else None


class Slot:
    """Owns at most one llama-server child."""

    def __init__(self):
        self.model_key = None
        self.proc = None
        # The model FILE this slot's child was launched with — the ground truth
        # the child's own /props answer is verified against (see
        # _identity_matches). None until a load resolves one.
        self.model_path = None
        # Identity-check cache: (pid, verdict, note, when). Bounded by
        # _IDENTITY_RECHECK_S so a hot proxy path costs one local GET a minute,
        # not one per request.
        self._identity = {"pid": None, "ok": True, "note": None, "at": 0.0}
        self.ngl = None
        self.ctx = None
        self.threads = None
        self.cpus = None
        self.gpu = None
        # Per-GPU device pin (2026-09-25): a central tensor-split across local
        # cards (list of proportions) + its --main-gpu; both None for a single
        # card / auto placement.
        self.tensor_split = None
        self.main_gpu = None
        self.profile_bin = None      # env-profiles (stage 1): the profile venv
        # bin dir this model's child launches from (None = shared venv default).
        self.expected_bytes = None
        # GGUF header block_count of the seated model (None = unknown/non-GGUF):
        # the "of 48" in the console's offload readout.
        self.total_layers = None
        # Effective --n-cpu-moe the child LAUNCHED with (None = no MoE split /
        # dense / unsupported engine): the honest MoE-placement readout.
        self.n_cpu_moe = None
        self.loaded_at = 0.0
        self.last_used = 0.0
        # ALLOCATION PROVENANCE (2026-09-23): what the current seat was loaded
        # FOR (the caller's ask, before any autofit/make-room plan), who asked
        # (designation / per-request <rid> / operator), and — when this load
        # replaced a resident of the same model — why. SlotPool.endpoint_for
        # compares a request's ask against alloc_requested before reusing.
        self.alloc_requested = None
        self.alloc_source = None
        self.reload_reason = None
        # Free VRAM sampled at the start of the CURRENT load (slice 12): the
        # baseline the stall-detector measures VRAM-consumed against.
        self._load_free_vram_at_start = None
        # Consecutive genuine load failures for a model_key, for the console/logs
        # (the "attempt N" in the failure reason). Informational ONLY — it never
        # blocks a re-attempt; every /load actually tries to seat the model. Plus
        # the last honest failure reason.
        self._load_failures: dict = {}          # model_key -> consecutive count
        self.last_load_error: "str | None" = None
        # t140: the loader's last stderr excerpt behind last_load_error
        self.last_load_stderr: "str | None" = None
        self.last_load_stderr_raw: "str | None" = None
        self.last_load_class: "str | None" = None
        self._stderr_tail = None
        self._stderr_thread = None
        # 2026-09-23: the per-load log file (full stderr) + the one behind
        # the last failure, so the reply/status carry it as log_ref.
        self._load_log_path: "str | None" = None
        self.last_load_log_ref: "str | None" = None
        self.lock = threading.Lock()
        # llama_cpp.server (python child) cannot take CONCURRENT streaming
        # requests — overlapping streams kill BOTH with an incomplete chunked
        # read (observed 2026-07-02: the media console's chat + side-calls).
        # The proxy serializes streams through this gate for python children;
        # the C++ llama-server handles parallel slots natively and skips it.
        self.child_kind = None
        self.stream_gate = threading.Semaphore(1)
        # Requests currently streaming through the proxy. The python child is
        # single-threaded: while it GENERATES it cannot answer a health probe,
        # so a probe-only healthy() reported False mid-request and (a) the
        # console flipped the slot to "loading" every time the model was USED,
        # (b) an overlapping proxy call 503'd instead of waiting on the gate.
        # Busy == alive by definition — the request is being served right now.
        self.inflight = 0
        self.child_base = f"http://127.0.0.1:{SLOT_CHILD_PORT}"

    # -- health ------------------------------------------------------------
    def _child_alive(self) -> bool:
        return bool(self.proc) and self.proc.poll() is None

    def healthy(self) -> bool:
        if not self._child_alive():
            return False
        if self.inflight > 0:
            # Mid-generation: the (python) child won't answer a probe, but it
            # is literally serving a request — that's the healthiest it gets.
            return True
        import httpx
        try:
            if httpx.get(self.child_base + "/health", timeout=2.0).status_code == 200:
                return True
        except Exception:
            pass
        try:
            # llama_cpp.server (the python fallback child) has no /health;
            # /v1/models answering is its liveness signal.
            return httpx.get(self.child_base + "/v1/models", timeout=2.0).status_code == 200
        except Exception:
            return False

    def _child_model_id(self):
        """What the process on the child port SAYS it is serving, or None when
        it says nothing identifiable. /props is the native llama-server's
        answer; /v1/models is the llama_cpp.server child's."""
        import httpx
        for path in ("/props", "/v1/models"):
            try:
                resp = httpx.get(self.child_base + path, timeout=2.0)
                if resp.status_code != 200:
                    continue
                got = _reported_model_id(resp.json())
                if got:
                    return got
            except Exception:  # noqa: BLE001 — an unanswered probe is "can't tell"
                continue
        return None

    def verify_identity(self, force: bool = False):
        """Confirm the process on the child port is serving THIS slot's model.

        Returns ``(ok, note)``. ``ok`` is True when the child's own statement
        matches what we launched, when it says nothing identifiable (can't tell
        — never tear down a seat on missing data), or when nothing is claimed at
        all. On a genuine MISMATCH the stale mapping is DROPPED: the child (if
        it is still ours) is killed and the claim cleared, so the next request
        cold-loads the right model instead of being silently answered by a
        stranger that took the port."""
        if self.model_key is None:
            return True, None
        if not self._child_alive():
            # Our process is gone; anything answering that port is not ours.
            self._self_heal()
            return (self.model_key is None), "child process is gone"
        cached = getattr(self, "_identity", None) or {"pid": None, "ok": True, "note": None, "at": 0.0}
        if self.inflight > 0:
            # Mid-generation: the request in flight is itself proof the port
            # answers, and a python child cannot answer a probe while it runs.
            return cached["ok"], cached["note"]
        pid = self.proc.pid
        if (not force and cached["pid"] == pid
                and (time.time() - cached["at"]) < _IDENTITY_RECHECK_S):
            return cached["ok"], cached["note"]
        reported = self._child_model_id()
        ok = _identity_matches(self.model_path, self.model_key, reported)
        note = None if ok else (
            f"the process on port {SLOT_CHILD_PORT} reports {reported!r}, but "
            f"this slot launched {self.model_key} from "
            f"{self.model_path!r}")
        self._identity = {"pid": pid, "ok": ok, "note": note, "at": time.time()}
        if not ok:
            logger.error(
                "slot %s: MODEL SUBSTITUTION detected — %s. Dropping the stale "
                "mapping; the next request will cold-load %s properly.",
                SLOT_ID, note, self.model_key)
            claimed = self.model_key
            if self.lock.acquire(blocking=False):
                try:
                    self._kill()
                    self._clear_claim()
                    # _clear_claim resets the verdict (a cleared slot has
                    # nothing to verify); keep THIS one, so /status can still
                    # say WHY the seat vanished instead of just losing it.
                    self._identity = {"pid": None, "ok": False, "note": note,
                                      "at": time.time()}
                finally:
                    self.lock.release()
            else:
                # A load is in flight and owns the lock — its own failure path
                # cleans up. Never race it; the note still travels back.
                logger.warning("slot %s: identity mismatch on %s while a load "
                               "holds the lock — leaving cleanup to it",
                               SLOT_ID, claimed)
        return ok, note

    def _clear_claim(self):
        """Forget the seated model (shared by unload, self-heal and the identity
        drop) — every field that describes the occupant, in one place."""
        self.model_key = self.ngl = self.ctx = None
        self.threads = self.cpus = self.gpu = self.expected_bytes = None
        self.tensor_split = self.main_gpu = None
        self.total_layers = None
        self.n_cpu_moe = None
        self.profile_bin = None
        self.model_path = None
        self.alloc_requested = self.alloc_source = self.reload_reason = None
        self._identity = {"pid": None, "ok": True, "note": None, "at": 0.0}

    def _self_heal(self):
        """Clear a WEDGED claim: child dead but model_key still set.

        Without this the slot reports model_key + healthy=False FOREVER — the
        console renders that as a permanent "loading" pill and endpoint_for
        waits its full timeout on a corpse. (Observed 2026-07-03: slot 1's
        reload died and flux showed "loading" indefinitely while chats were
        actually served in-process.) Non-blocking lock: a load() mid-flight
        also has model_key set with the child coming up — its own failure
        path cleans up, so never race it."""
        if self.model_key is None or self._child_alive():
            return
        if not self.lock.acquire(blocking=False):
            return
        try:
            if self.model_key is not None and not self._child_alive():
                logger.warning("slot %s: child died while claiming %s — "
                               "clearing the stale claim", SLOT_ID, self.model_key)
                self._clear_claim()
                self.proc = None
        finally:
            self.lock.release()

    def status(self) -> dict:
        from hugpy_engine.spill import free_vram_bytes
        self._self_heal()
        # The scheduler routes on this dict, so it must not advertise a model a
        # stranger on the port would answer for. Cached/skipped as described in
        # verify_identity — a status poll costs at most one local GET a minute.
        self.verify_identity()
        out = {
            "slot_id": SLOT_ID,
            "control_port": SLOT_PORT,
            "child_port": SLOT_CHILD_PORT,
            "endpoint": f"http://{SLOT_ADVERTISE}:{SLOT_PORT}",
            "model_key": self.model_key,
            "healthy": self.healthy(),
            "busy": self.inflight > 0,
            "n_gpu_layers": self.ngl,
            # GGUF block_count of the seated model — the "of N" for the console's
            # "17/48 layers". None for non-GGUF / an unreadable header (getattr:
            # an instance created before this field existed must not 500 /status).
            "total_layers": getattr(self, "total_layers", None),
            # Effective --n-cpu-moe the child launched with (MoE expert split);
            # None = no split. getattr for the same pre-field-instance reason.
            "n_cpu_moe": getattr(self, "n_cpu_moe", None),
            "ctx": self.ctx,
            # Allocation provenance (additive; see __init__). alloc_effective is
            # what the child actually launched with.
            "alloc_requested": getattr(self, "alloc_requested", None),
            "alloc_source": getattr(self, "alloc_source", None),
            "alloc_effective": ({
                "n_gpu_layers": self.ngl,
                "total_layers": getattr(self, "total_layers", None),
                "n_cpu_moe": getattr(self, "n_cpu_moe", None),
                "device": ("cpu" if self.ngl == 0 else "cuda") if isinstance(self.ngl, int) else None,
            } if self.model_key else None),
            "reload_reason": getattr(self, "reload_reason", None),
            "threads": self.threads,
            "cpus": self.cpus,
            "gpu": self.gpu,
            # Per-GPU placement this seat was loaded with (None for a single
            # card / auto): the console + central per-device residency read it.
            "tensor_split": getattr(self, "tensor_split", None),
            "main_gpu": getattr(self, "main_gpu", None),
            "profile_bin": self.profile_bin,   # env-profiles: child's venv, or None
            "allowed_cpus": _allowed_cpus(),   # kernel-enforced dedicated cores
            "loaded_at": self.loaded_at,
            "last_used": self.last_used,
            "free_vram_bytes": free_vram_bytes(),
            "rss_bytes": _proc_rss_bytes(self.proc.pid) if self._child_alive() else 0,
            # The child llama-server/llama_cpp.server PID: this is the process
            # that actually HOLDS the model's VRAM (the slot supervisor python
            # only carries a ~tens-of-MiB CUDA context). The worker agent joins
            # this against nvidia-smi's per-process accounting to report the
            # slot occupant's REAL VRAM (its type/ngl guess is not ground truth).
            "child_pid": self.proc.pid if self._child_alive() else None,
            # The file the child launched with + the last identity verdict: what
            # the "is this really our model on that port" check compares, made
            # visible instead of implied (getattr for a pre-field instance).
            "model_path": getattr(self, "model_path", None),
            "identity_ok": getattr(self, "_identity", {}).get("ok", True),
            "identity_note": getattr(self, "_identity", {}).get("note"),
            "expected_bytes": self.expected_bytes,
            # The last honest load-failure reason, so the console can show WHY a
            # model's row is degraded/retrying. Informational — it does not gate
            # re-attempts. None once a load succeeds.
            "last_load_error": self.last_load_error,
            # t140: the loader's own last words behind that error (None when
            # the child never wrote any, or once a load succeeds).
            "last_load_stderr": getattr(self, "last_load_stderr", None),
            # 2026-09-23: the WHOLE loader stderr + the per-load log file
            # holding it (the excerpt above is a heading, not the log).
            "last_load_stderr_full": getattr(self, "last_load_stderr_raw", None),
            "last_load_log_ref": getattr(self, "last_load_log_ref", None),
        }
        # Honest RSS split (omit-when-unset): rss_bytes stays VmRSS verbatim for
        # wire back-compat, while rss_anon_bytes is the truly-pinned RAM and
        # rss_file_bytes the mmap'd-GGUF page cache VmRSS also counts (~28x
        # overstatement observed on ae). Absent entirely when /proc can't say.
        if self._child_alive():
            out.update(_proc_rss_detail(self.proc.pid))
        return out

    # -- lifecycle ---------------------------------------------------------
    def load(self, model_key, n_gpu_layers=None, ctx=None, threads=None,
             cpus=None, gpu=None, path=None, gpu_mem_gib=None,
             cpu_mem_gib=None, profile_bin=None, force=False,
             n_cpu_moe=None, alloc_mode=None, alloc_requested=None,
             alloc_source=None, reload_reason=None,
             tensor_split=None, main_gpu=None) -> dict:
        with self.lock:
            # k64: the ACTIVE allocation mode, as a per-load opt. The slot is a
            # separate process spawned at boot, so the agent's per-request
            # HUGPY_ALLOC_MODE (worker_agent._apply_spill) never reaches it —
            # a max-ram/explicit designation was silently planned as max-gpu
            # here. Project the opt into this process's env (what
            # spill.alloc_mode_env / _moe_gpu_budget / _strict_gpu_only read),
            # clear-when-absent so a mode can never leak onto the next model
            # (the same rule as _SPILL_ENV_CLEAR_WHEN_ABSENT on the agent).
            # Single-flight under self.lock, so the process-wide env is safe.
            if alloc_mode not in (None, ""):
                os.environ["HUGPY_ALLOC_MODE"] = str(alloc_mode)
            else:
                os.environ.pop("HUGPY_ALLOC_MODE", None)
            # ``force`` (k14 relaunch): a relaunch re-seats the SAME model with a
            # NEW spec (e.g. a swept-down n_gpu_layers), so it must bypass the
            # already-serving short-circuit and actually respawn the child —
            # otherwise a same-model relaunch is a silent no-op and the sweep can
            # never change the offload depth.
            if not force and self.model_key == model_key and self.healthy():
                # Same model is not the same seat: a caller that states what it
                # asked for must get a seat loaded for THAT ask (the pool
                # normally unloads first; this guards a direct /load).
                why = None
                if isinstance(alloc_requested, dict):
                    from hugpy_engine.serve.slots import alloc_mismatch, alloc_signature
                    why = alloc_mismatch({"alloc_requested": self.alloc_requested,
                                          "alloc_source": self.alloc_source,
                                          "n_gpu_layers": self.ngl,
                                          "total_layers": self.total_layers},
                                         alloc_signature(alloc_requested), alloc_source)
                if why is None:
                    self.last_used = time.time()
                    return self.status()
                reload_reason = reload_reason or why
                logger.info("slot %s: %s", SLOT_ID, why)

            # No post-failure latch: an earlier failed load of this model does not
            # refuse this attempt. Every /load actually tries to seat the model and,
            # if it fails, fails with THAT attempt's real loader error (recorded
            # below as last_load_error for the console/logs).

            # PER-GPU device pin (2026-09-25). Central chose the card(s):
            #   * a TENSOR-SPLIT (a GGUF that central split across >= 2 local
            #     cards) -> DON'T restrict CUDA_VISIBLE_DEVICES (all cards must
            #     stay visible for the split), pass --tensor-split + --main-gpu
            #     (global device indices) to llama-server.
            #   * a SINGLE-card pin (``gpu``) -> CUDA_VISIBLE_DEVICES=that card,
            #     exactly the historical mechanism.
            # A split-plan carries the ``gpu`` pin too (both ride HUGPY_MAIN_GPU),
            # so the split must WIN — pinning one card would defeat it.
            ts = _parse_tensor_split(tensor_split)
            is_split = ts is not None and sum(1 for x in ts if x and x > 0) >= 2
            self.tensor_split = ts if is_split else None
            self.main_gpu = _as_int_or_none(main_gpu) if is_split else None
            self._kill()
            self.profile_bin = profile_bin or None
            (argv, self.ngl, self.ctx, self.threads, self.cpus,
             self.child_kind, self.total_layers, self.n_cpu_moe) = _build_cmd(
                model_key, n_gpu_layers, ctx, threads, cpus, path=path,
                gpu_mem_gib=gpu_mem_gib, cpu_mem_gib=cpu_mem_gib,
                profile_bin=self.profile_bin, n_cpu_moe=n_cpu_moe,
                tensor_split=self.tensor_split, main_gpu=self.main_gpu)
            # per-load GPU pin overrides the slot's MAIN_GPU default; a split
            # leaves the card unset so every visible GPU can hold its shard.
            self.gpu = None if is_split else (gpu if gpu not in (None, "") else MAIN_GPU)
            self.expected_bytes = _model_expected_bytes(model_key)
            logger.info("slot %s loading %s (ngl=%s ctx=%s threads=%s cpus=%s "
                        "gpu=%s tensor_split=%s main_gpu=%s): %s",
                        SLOT_ID, model_key, self.ngl, self.ctx, self.threads,
                        self.cpus, self.gpu, self.tensor_split, self.main_gpu,
                        " ".join(argv))

            env = dict(os.environ)
            if self.gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
            # A from-source llama-server links against sibling .so files under the
            # engine dir (libllama/libggml/…). Prepend those to the child's
            # LD_LIBRARY_PATH so it loads without a unit-level env hack (the ae
            # 2026-07-06 manual fix, now derived from HUGPY_ENGINE_DIR in code).
            # Additive + Linux-only. argv[0] (k94) extends the fix to a binary
            # resolved from any hugpy-managed engine dir — the default data-dir
            # install and the persisted-config location — while a system/PATH
            # binary or the python child still adds nothing.
            try:
                from hugpy_engine.native.resolve import ld_library_path_with_engine
                _ld = ld_library_path_with_engine(env.get("LD_LIBRARY_PATH"),
                                                  bin_path=argv[0])
                if _ld:
                    env["LD_LIBRARY_PATH"] = _ld
            except Exception:  # noqa: BLE001 — never block a load on lib-path derivation
                pass
            # Env-profiles (stage 1): activate the profile venv for the CHILD only
            # — prepend its bin dir to PATH (a profile-shipped binary wins) and set
            # VIRTUAL_ENV. The agent process is never touched; only this child runs
            # from the profile. No-op without a profile.
            try:
                from hugpy_engine.serve import profiles as _profiles
                env = _profiles.child_env(env, self.profile_bin)
            except Exception:  # noqa: BLE001 — never block a load on env derivation
                pass
            self.proc = subprocess.Popen(argv, env=env, stderr=subprocess.PIPE)
            self._stderr_tail = collections.deque(maxlen=_STDERR_TAIL_LINES)
            self._load_log_path = _load_log_path(model_key)
            self._stderr_thread = _start_stderr_tail(self.proc, self._stderr_tail,
                                                     self._load_log_path)
            self.model_key = model_key
            # The file this child was launched with — argv's -m/--model — kept
            # so the identity check has something to verify /props against.
            self.model_path = _model_path_of(argv)
            self._identity = {"pid": self.proc.pid, "ok": True, "note": None,
                              "at": 0.0}
            self.loaded_at = self.last_used = time.time()

            if not self._wait_healthy():
                self._kill()
                self.model_key = None
                # Record the genuine failure (count + reason) for the console/logs.
                # It does NOT arm any backoff — the next /load re-attempts. The
                # message names STALL vs hard-cap (the honest reason), not a clock.
                n = self._load_failures.get(model_key, 0) + 1
                self._load_failures[model_key] = n
                kind = getattr(self, "_load_fail_kind", None)
                exit_code = getattr(self, "_load_exit_code", None)
                # t140: quote the loader's own error. The child is dead (or
                # killed), so its stderr pipe is at EOF — give the tail thread
                # a moment to drain the final lines before reading the ring.
                _t = getattr(self, "_stderr_thread", None)
                if _t is not None:
                    try:
                        _t.join(2.0)
                    except Exception:  # noqa: BLE001
                        pass
                self.last_load_stderr = _stderr_excerpt(
                    getattr(self, "_stderr_tail", None)) or None
                # 2026-09-23: the loader's WHOLE stderr verbatim (from the
                # per-load log file), for the structured load_failure the
                # /load reply carries, plus that file's path as log_ref.
                self.last_load_log_ref = getattr(self, "_load_log_path", None)
                self.last_load_stderr_raw = _stderr_verbatim(
                    getattr(self, "_stderr_tail", None), self.last_load_log_ref)
                _why = (f"Loader stderr: {self.last_load_stderr}"
                        if self.last_load_stderr else
                        f"loader stderr was empty (read ring + log file "
                        f"{self.last_load_log_ref or '<none written>'}, bytes=0; fail kind={kind}, "
                        f"exit_code={exit_code}, attempt {n})")
                self.last_load_class = "other"
                if kind == "exit" and isinstance(exit_code, int) and exit_code < 0:
                    # The child was killed by a SIGNAL (Popen returncode -N):
                    # -11 SIGSEGV / -9 oom-kill / -6 abort. That is a CRASH —
                    # in practice exhausted/leaked VRAM or driver state (k70;
                    # the 2026-08-04 32B case crashed -11 on a 383MB-free card
                    # while its file chunksum-verified clean) — NOT a verdict
                    # on the model file. The wording says "child crashed",
                    # which central classifies fail-fast but STATE-DEPENDENT:
                    # never cached, so the next request re-attempts once the
                    # card frees instead of inheriting a corrupt-file verdict.
                    import signal as _sig
                    try:
                        signame = _sig.Signals(-exit_code).name
                    except ValueError:
                        signame = f"signal {-exit_code}"
                    self.last_load_error = (
                        f"the llama-server child crashed ({signame}, exit "
                        f"{exit_code}) after "
                        f"{getattr(self, '_load_fail_after_s', 0.0):.1f}s without "
                        f"ever serving — likely exhausted VRAM/RAM or driver "
                        f"state, not a verdict on the model file. {_why} "
                        f"Attempt {n}")
                elif kind == "exit":
                    # The child EXITED cleanly-but-nonzero rather than hung: the
                    # loader rejected the model file. Permanent by construction —
                    # the same bytes fail the same way forever — so the wording
                    # carries "hard load failure", which central's
                    # _PERMANENT_LOAD_MARKERS matches to fail the call fast (and
                    # cache) instead of holding and re-requesting it.
                    self.last_load_class = "hard_load_failure"
                    self.last_load_error = (
                        f"hard load failure: the llama-server child exited "
                        f"(code {exit_code}) after "
                        f"{getattr(self, '_load_fail_after_s', 0.0):.1f}s without "
                        f"ever serving — the loader rejected the load, not "
                        f"stalled; retrying cannot fix it. {_why} "
                        f"Attempt {n}")
                else:
                    self.last_load_error = (
                        f"did not become healthy ({kind or 'stall/hard-cap'}); "
                        + (f"{_why} " if self.last_load_stderr else "")
                        + f"attempt {n}")
                raise self._load_failure_exc(
                    f"slot {SLOT_ID}: {model_key} {self.last_load_error}",
                    model_key)
            # SUCCESS — record the allocation provenance of this seat.
            self.alloc_requested = (dict(alloc_requested) if isinstance(alloc_requested, dict)
                                    else {"n_gpu_layers": None if _ngl_is_unset(n_gpu_layers)
                                          else n_gpu_layers,
                                          "alloc_mode": alloc_mode or None})
            self.alloc_source = (dict(alloc_source) if isinstance(alloc_source, dict)
                                 and alloc_source else {"kind": "unknown"})
            self.reload_reason = reload_reason or None
            if self.reload_reason:
                logger.info("slot %s: %s", SLOT_ID, self.reload_reason)
            # SUCCESS — clear the failure count + last-error for this model.
            self._load_failures.pop(model_key, None)
            self.last_load_error = None
            self.last_load_stderr = None
            self.last_load_stderr_raw = None
            self.last_load_log_ref = None
            self.last_load_class = None
            logger.info("slot %s ready: %s on %s", SLOT_ID, model_key, self.child_base)
            return self.status()

    def _load_failure_exc(self, message: str, model_key: str):
        """The structured failure for a load that never served (2026-09-23):
        ``ModelLoadFailure`` carrying the class the verdict above recorded
        (``hard_load_failure`` for a loader rejection), the loader's stderr
        verbatim (bounded) and the file the child opened — the /load route
        ships it as ``load_failure`` so the caller need not parse wording."""
        from hugpy_engine.serve.load_failure import ModelLoadFailure, HardLoadFailure
        cls = getattr(self, "last_load_class", None) or "other"
        kind = HardLoadFailure if cls == "hard_load_failure" else ModelLoadFailure
        return kind(message, load_class=cls,
                    loader_stderr=(getattr(self, "last_load_stderr_raw", None)
                                   or self.last_load_stderr),
                    path=getattr(self, "model_path", None), model_key=model_key,
                    log_ref=getattr(self, "last_load_log_ref", None))

    def _hard_cap_s(self) -> float:
        """Size-scaled generous-but-bounded hard cap. base (HEALTH_TIMEOUT) +
        expected_bytes / assumed throughput, × a safety multiplier, floored at
        HEALTH_TIMEOUT. A 46G model at 200 MB/s ~= 235s of transfer alone, so the
        cap lands well above a legitimate cold load while still bounding a truly-
        wedged child. Unknown size -> the flat floor (back-compat)."""
        exp = self.expected_bytes
        if not exp:
            return HEALTH_TIMEOUT
        transfer_s = float(exp) / max(_LOAD_THROUGHPUT_BPS, 1.0)
        return max(HEALTH_TIMEOUT, (HEALTH_TIMEOUT + transfer_s) * _HARD_CAP_MULT)

    def _load_progress_bytes(self) -> int:
        """A monotonic-ish PROGRESS signal for an in-flight load: the child's
        resident RAM (weights paging in) PLUS the VRAM consumed since we started
        (free VRAM DROPPING as layers upload). Either one growing == the load is
        moving. Best-effort: 0 on any read failure (a run of 0s reads as a stall,
        which is the safe conservative verdict)."""
        rss = 0
        try:
            if self._child_alive():
                rss = _proc_rss_bytes(self.proc.pid) or 0
        except Exception:  # noqa: BLE001
            rss = 0
        vram_used = 0
        try:
            from hugpy_engine.spill import free_vram_bytes
            fv = free_vram_bytes()
            if fv is not None and self._load_free_vram_at_start is not None:
                vram_used = max(0, self._load_free_vram_at_start - fv)
        except Exception:  # noqa: BLE001
            vram_used = 0
        return int(rss) + int(vram_used)

    def _wait_healthy(self) -> bool:
        """Wait for the child to answer /health, failing on STALL not on a blind
        clock (slice 12). While the child is alive AND making forward progress
        (RSS growing / VRAM filling), keep waiting — a big cold load is SLOW, not
        broken. Kill only when progress stalls for STALL_TIMEOUT, or when the
        generous size-scaled hard cap is blown (a truly-wedged child must die)."""
        try:
            from hugpy_engine.spill import free_vram_bytes
            self._load_free_vram_at_start = free_vram_bytes()
        except Exception:  # noqa: BLE001
            self._load_free_vram_at_start = None
        start = time.time()
        hard_cap = self._hard_cap_s()
        last_progress = self._load_progress_bytes()
        last_progress_ts = start
        # Why the wait ended, for an HONEST caller message (2026-07-29). All three
        # exits below used to collapse into a bare False, so load() reported
        # "stall/hard-cap" even when the child had died in under a second because
        # the FILE was rejected. That misclassification is what made a corrupt
        # GGUF cost 900s instead of 0.75s: a stall reads as transient, so central
        # held and retried it.
        self._load_fail_kind = None
        self._load_exit_code = None
        while True:
            if not self._child_alive():
                # child exited -> real failure, and a FAST exit means the loader
                # rejected the model itself (bad tensors / unknown arch / corrupt
                # file). No retry can fix that; say so instead of crying stall.
                self._load_fail_kind = "exit"
                _p = getattr(self, "proc", None)   # never assume the child handle
                self._load_exit_code = getattr(_p, "returncode", None)
                self._load_fail_after_s = time.time() - start
                return False
            if self.healthy():
                return True                      # up and answering
            now = time.time()
            if now - start >= hard_cap:
                self._load_fail_kind = "hardcap"
                self._load_fail_after_s = now - start
                logger.warning("slot %s: load of %s blew the %.0fs hard cap "
                               "(size-scaled) — treating as wedged",
                               SLOT_ID, self.model_key, hard_cap)
                return False
            cur = self._load_progress_bytes()
            if cur - last_progress >= _PROGRESS_EPSILON:
                last_progress = cur              # real movement — reset the stall clock
                last_progress_ts = now
            elif now - last_progress_ts >= STALL_TIMEOUT:
                self._load_fail_kind = "stall"
                self._load_fail_after_s = now - start
                logger.warning("slot %s: load of %s STALLED — no forward progress "
                               "(RSS+VRAM) for %.0fs (last=%s); killing the wedged "
                               "child", SLOT_ID, self.model_key, STALL_TIMEOUT,
                               cur)
                return False
            time.sleep(1.0)

    def unload(self) -> dict:
        # Interrupt any in-progress load first: killing the child makes a blocking
        # _wait_healthy bail and release the lock, so /unload returns promptly
        # instead of waiting out the load's (up to 180s) health timeout.
        self._kill()
        with self.lock:
            self._kill()
            self._clear_claim()
            return self.status()

    def relaunch(self, n_gpu_layers=None, ctx=None, n_cpu_moe=None) -> dict:
        """Re-seat the CURRENTLY-loaded model with a new offload depth / context —
        the lever the k7 offload speed-cliff sweep needs (seat at full offload,
        then relaunch DOWN through decreasing ``n_gpu_layers``, measuring tok/s at
        each step). This is the ONLY way to change a live slot child's ngl: the
        slot child is a spawned process whose ngl is fixed at launch, so a change
        means STOP-then-RESPAWN — which is exactly what this does (via a forced
        load: SIGTERM->wait->SIGKILL of the old child, then a fresh spawn). It also
        answers the ae "slot-child PID never recycles" blocker: relaunch replaces
        the child under a NEW pid every time, no worker restart required.

        The current model_key + its threads/cpus/gpu/profile are preserved; only
        ``n_gpu_layers`` and ``ctx`` are overridden (``None`` for either keeps the
        current value — ctx from the live child, ngl re-autofit). The resulting
        allocation is reported HONESTLY (the echoed ``n_gpu_layers`` is what the
        fresh child actually launched with, i.e. ``self.ngl`` after the respawn —
        not merely what was requested)."""
        mk = self.model_key
        if mk is None:
            raise RuntimeError(
                f"slot {SLOT_ID}: no model loaded — nothing to relaunch")
        requested_ngl = n_gpu_layers
        # A deliberate operator relaunch is a fresh attempt: reset the informational
        # consecutive-failure count so the reseat's "attempt N" starts clean.
        self._load_failures.pop(mk, None)
        result = self.load(
            mk, n_gpu_layers=requested_ngl,
            ctx=ctx if ctx is not None else self.ctx,
            threads=self.threads, cpus=self.cpus, gpu=self.gpu,
            gpu_mem_gib=None, cpu_mem_gib=None,
            profile_bin=self.profile_bin, force=True,
            n_cpu_moe=n_cpu_moe,
            # Preserve the seat's per-GPU placement across a relaunch (a swept
            # ngl must keep the same card / split, not fall back to auto).
            # getattr-guarded for a slot built without the newer fields.
            tensor_split=getattr(self, "tensor_split", None),
            main_gpu=getattr(self, "main_gpu", None),
            alloc_requested={"n_gpu_layers": requested_ngl, "alloc_mode": None},
            alloc_source={"kind": "operator", "via": "relaunch", "at": time.time()})
        # Surface the request alongside the honest launched value so the caller
        # can see requested-vs-effective at a glance (self.ngl / status carries
        # the measured launch value).
        result = dict(result)
        result["relaunched"] = True
        result["requested_n_gpu_layers"] = requested_ngl
        result["requested_n_cpu_moe"] = n_cpu_moe
        return result

    def _kill(self):
        if self._child_alive():
            try:
                self.proc.terminate()
                self.proc.wait(timeout=15)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None


def build_app():
    from flask import Flask, request, jsonify, Response

    slot = Slot()
    app = Flask(f"abstract_hugpy_slot_{SLOT_ID}")

    @app.route("/health")
    def health():
        return jsonify({"ok": True, "slot_id": SLOT_ID})

    @app.route("/status")
    def status():
        return jsonify(slot.status())

    @app.route("/identity")
    def identity():
        """IDENTITY-QUERY-20260910: the engine's own statement of what it serves, next to what
        this seat claims. `verified` is a POSITIVE match; None = engine said nothing."""
        engine_model = slot._child_model_id() if slot._child_alive() else None
        verified = None if not engine_model else _identity_matches(slot.model_path, slot.model_key, engine_model)
        return jsonify({
            "slot_id": SLOT_ID, "model_key": slot.model_key, "model_path": slot.model_path,
            "engine_model": engine_model, "healthy": slot.healthy(), "verified": verified,
        })

    @app.route("/load", methods=["POST"])
    def load():
        body = request.get_json(silent=True) or {}
        if not body.get("model_key"):
            return jsonify({"error": "missing model_key"}), 400
        # Per-box "never serve locally" policy: a slot on a policy box must not
        # spawn a llama-server child even on a direct /load (the scheduler
        # already stops routing here, but this self-protects against any direct
        # caller). Set HUGPY_NO_LOCAL_SERVING=true in the slot unit's env to
        # arm it. Default off === today's behavior; workers never set the flag.
        from hugpy_engine.serve.policy import no_local_serving, local_serving_error
        if no_local_serving():
            return jsonify({"error": local_serving_error(
                body.get("model_key"),
                detail="slot serving disabled on this box")}), 403
        # "This n_gpu_layers is a FILL-IN DEFAULT, not a demand" (2026-07-25).
        # Callers that materialize DEFAULT_LLAMA_NGL into an int field (the
        # serve spec / spill env path) set this so placement policy still sees
        # the value as unset — see _NglDefaulted. Omitted (the overwhelmingly
        # common case, and every pre-existing caller) == explicit, byte-identical
        # to before. Optional additive body key: nothing rejects unknown keys on
        # this local control plane, and central->worker relay never touches it.
        _ngl_body = body.get("n_gpu_layers")
        if body.get("ngl_defaulted") and _ngl_body not in (None, ""):
            try:
                _ngl_body = _NglDefaulted(int(_ngl_body))
            except (TypeError, ValueError):
                pass
        try:
            return jsonify(slot.load(body["model_key"], _ngl_body,
                                     body.get("ctx"), body.get("threads"),
                                     body.get("cpus"), body.get("gpu"),
                                     path=body.get("path"),
                                     gpu_mem_gib=body.get("gpu_mem_gib"),
                                     cpu_mem_gib=body.get("cpu_mem_gib"),
                                     profile_bin=body.get("profile_bin"),
                                     n_cpu_moe=body.get("n_cpu_moe"),
                                     alloc_mode=body.get("alloc_mode"),
                                     alloc_requested=body.get("alloc_requested"),
                                     alloc_source=body.get("alloc_source"),
                                     reload_reason=body.get("reload_reason"),
                                     tensor_split=body.get("tensor_split"),
                                     main_gpu=body.get("main_gpu")))
        except Exception as exc:  # noqa: BLE001
            out = {"error": f"{type(exc).__name__}: {exc}"}
            # Additive (2026-09-23): the structured verdict, so the pool client
            # re-raises a typed hard failure instead of a string to parse.
            _lf = getattr(exc, "load_failure", None)
            if isinstance(_lf, dict):
                out["load_failure"] = _lf
            return jsonify(out), 500

    @app.route("/unload", methods=["POST"])
    def unload():
        return jsonify(slot.unload())

    @app.route("/relaunch", methods=["POST"])
    def relaunch():
        # k14: re-seat the CURRENT model with a new offload depth / context so the
        # k7 offload speed-cliff sweep can measure tok/s per n_gpu_layers. Body:
        # {"n_gpu_layers"?: int, "ctx"?: int} — omit either to keep it. A slot with
        # no model loaded is a 409 (nothing to relaunch), never a 500.
        body = request.get_json(silent=True) or {}
        from hugpy_engine.serve.policy import no_local_serving, local_serving_error
        if no_local_serving():
            return jsonify({"error": local_serving_error(
                slot.model_key, detail="slot serving disabled on this box")}), 403
        if slot.model_key is None:
            return jsonify({"error": f"slot {SLOT_ID} has no model loaded "
                            "to relaunch"}), 409
        try:
            return jsonify(slot.relaunch(body.get("n_gpu_layers"),
                                         body.get("ctx"),
                                         n_cpu_moe=body.get("n_cpu_moe")))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    @app.route("/v1/<path:sub>", methods=["POST", "GET"])
    def proxy(sub):
        """Forward the OpenAI API to the child, streaming the response back."""
        import httpx
        if not slot.healthy():
            return jsonify({"error": f"slot {SLOT_ID} has no model loaded"}), 503
        # IDENTITY (k53): forward only to a child that is serving the model this
        # slot claims. A stranger on the port would otherwise answer in the
        # claimed model's name — a silent substitution. On mismatch the claim has
        # just been dropped, so 503 (not 500): the caller re-resolves and the
        # model cold-loads properly.
        ok, note = slot.verify_identity()
        if not ok:
            return jsonify({"error": (
                f"slot {SLOT_ID} refused to proxy: the child on port "
                f"{SLOT_CHILD_PORT} is not serving the claimed model — {note}. "
                "The stale mapping was dropped; retry to cold-load it.")}), 503
        # STALE-SLOT-FIX-20260910: a cached runner whose seat was swapped to another model
        # used to be forwarded anyway and answered in the wrong model's name.
        # Refuse with 409 so the caller re-resolves; the seat itself is untouched.
        wanted = _requested_model(request)
        if wanted and not _model_matches(slot, wanted):
            return jsonify({"error": (
                f"slot {SLOT_ID} is serving {slot.model_key!r}, not {wanted!r} - "
                "the caller's cached endpoint is stale; re-resolve the slot for this model"),
                "mismatch": True, "slot_model_key": slot.model_key,
                "requested_model": wanted}), 409
        slot.last_used = time.time()

        url = f"{slot.child_base}/v1/{sub}"
        body = request.get_data()
        headers = {k: v for k, v in request.headers
                   if k.lower() not in ("host", "content-length")}

        # Serialize python-child requests end-to-end (see stream_gate note):
        # the gate is held for the WHOLE response lifetime and released in the
        # generator's finally, so an overlapping caller waits instead of
        # crashing both streams.
        gated = (slot.child_kind == "python")
        if gated:
            slot.stream_gate.acquire()
        slot.inflight += 1
        if slot.inflight == 1:
            _nudge_agent_heartbeat(slot)   # idle→busy edge
        try:
            # Bound connect/write/pool so a dead or wedged child fails send()
            # fast instead of hanging forever while holding the stream gate +
            # inflight counter — that leak wedged EVERY later request
            # (busy=True, model=None). Read stays unbounded: a streamed
            # generation legitimately runs for minutes.
            client = httpx.Client(
                timeout=httpx.Timeout(None, connect=10.0, write=30.0, pool=10.0))
            upstream = client.send(
                client.build_request(request.method, url, content=body, headers=headers),
                stream=True,
            )
        except Exception:
            slot.inflight -= 1
            if slot.inflight == 0:
                _nudge_agent_heartbeat(slot)   # busy→idle edge (failed send)
            if gated:
                slot.stream_gate.release()
            raise

        def generate():
            try:
                for chunk in upstream.iter_raw():
                    yield chunk
            finally:
                upstream.close()
                client.close()
                slot.inflight -= 1
                if slot.inflight == 0:
                    _nudge_agent_heartbeat(slot)   # busy→idle edge (stream done)
                if gated:
                    slot.stream_gate.release()

        return Response(generate(), status=upstream.status_code,
                        content_type=upstream.headers.get("content-type", "application/json"))

    return app, slot


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # 2026-09-18: drop werkzeug access-log records (survives level reset —
    # setLevel alone did not hold, werkzeug re-enables its logger at serve
    # time). Real werkzeug WARN/ERROR (no "- - [" access marker) still flow.
    _wlog = logging.getLogger("werkzeug")
    _wlog.setLevel(logging.WARNING)
    _wlog.addFilter(lambda _r: " - - [" not in _r.getMessage())
    app, _slot = build_app()
    logger.info("slot %s listening on %s:%s (child on :%s, advertise %s)",
                SLOT_ID, SLOT_HOST, SLOT_PORT, SLOT_CHILD_PORT, SLOT_ADVERTISE)
    app.run(host=SLOT_HOST, port=SLOT_PORT, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
