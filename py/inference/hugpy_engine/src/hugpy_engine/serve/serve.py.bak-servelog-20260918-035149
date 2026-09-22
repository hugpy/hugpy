# imports/apis/serve.py  (or managers/serve/serve.py — see note at bottom)
"""
Serving abstraction. One model, one resolved ServeSpec; how it actually gets
served is a pluggable driver keyed by an explicit mode — not inferred from
whether cfg.port happens to be set.

    mode = off       -> not served externally; in-process runner handles it
    mode = systemd   -> a dedicated llama-server under its own unit (always-on)
    mode = swap      -> an entry in one shared llama-swap proxy (on-demand)

Why this shape:
    - registry, not port-sniffing: serve_spec_for() reads MODEL_REGISTRY + env,
      the same way resolve_model_source() does. cfg.port stops meaning four
      things at once.
    - driver registry, like vision_backends.register_backend and
      FRAMEWORK_RUNNERS: adding a serving mechanism is one @register_serve_driver.
    - plans, not callbacks: a driver returns a ServePlan (writes + commands).
      Dry-run renders it for the UI; apply() drains it. The run()/write()
      callables are the seam for local-vs-remote (ssh to a peer) execution.

Caters to either deployment with one env var:
    DEFAULT_SERVE_MODE=systemd   # every llama_cpp model gets its own unit
    DEFAULT_SERVE_MODE=swap      # everything routes through one llama-swap proxy
Per-model override lives in the overlay: cfg.extra["serve_mode"] = "swap".

os.path throughout, no pathlib.
"""

import getpass
import os
import sys
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .imports import *

logger = get_logFile(__name__)


# --------------------------------------------------------------------------- #
# explicit environment wiring                                                 #
# --------------------------------------------------------------------------- #

# Native engine location is resolved per-OS by hugpy.engine.resolve (env override
# -> a `hugpy install-engine` download/build -> PATH). The placeholder keeps an
# absolute, well-formed default when the engine isn't installed yet; the serve
# drivers report "run hugpy install-engine" rather than serving a missing binary.
from ..._platform.binaries import with_exe as _with_exe
from ..._platform.paths import engine_dir as _engine_dir
from ...engine import resolve as _engine_resolve

LLAMA_CPP_DIR = get_env_value("LLAMA_CPP_DIR") or _engine_dir()
LLAMA_SERVER_BIN = (get_env_value("LLAMA_SERVER_BIN")
                    or _engine_resolve.server_bin()
                    or os.path.join(LLAMA_CPP_DIR, "build", "bin", _with_exe("llama-server")))
LLAMA_SERVICE_USER = get_env_value("LLAMA_SERVICE_USER") or getpass.getuser()
LLAMA_SERVICE_GROUP = get_env_value("LLAMA_SERVICE_GROUP") or getpass.getuser()
SYSTEMD_UNIT_DIR = get_env_value("SYSTEMD_UNIT_DIR") or "/etc/systemd/system"
LLAMA_UNIT_PREFIX = get_env_value("LLAMA_UNIT_PREFIX") or "llama"

# llama-swap: one proxy, one config, one unit to bounce on config change.
LLAMA_SWAP_HOST = get_env_value("LLAMA_SWAP_HOST") or "127.0.0.1"
LLAMA_SWAP_PORT = int(get_env_value("LLAMA_SWAP_PORT") or 9292)
LLAMA_SWAP_CONFIG = get_env_value("LLAMA_SWAP_CONFIG") or "/etc/llama-swap/config.yaml"
LLAMA_SWAP_UNIT = get_env_value("LLAMA_SWAP_UNIT") or "llama-swap.service"
LLAMA_SWAP_TTL = int(get_env_value("LLAMA_SWAP_TTL") or 600)   # on-demand unload, seconds

# Fallback context window for a served llama-server when neither the model
# config nor DEFAULT_LLAMA_CTX (env) specifies one. Now matches the in-process
# runner default DEFAULT_N_CTX (managers/llama/runners/src/imports/constants.py);
# the previous 4096 silently truncated long outputs — the transcript overflowed,
# the model lost its framing and looped with no stop. The env override wins.
DEFAULT_LLAMA_CTX = int(get_env_value("DEFAULT_LLAMA_CTX") or 16384)
DEFAULT_LLAMA_THREADS = int(get_env_value("DEFAULT_LLAMA_THREADS") or 6)

# CONTEXT-as-allocation resolver (slice 11 / t27). The worker agent (which owns
# _RUNTIME_SETTINGS / ctx_pct) registers fn(model_key, cfg) -> resolved_ctx|None
# here at boot — same reverse-injection pattern as slots.set_residency_lookup, so
# managers/serve never imports worker_agent. When set AND it returns a ctx, that
# RESOLVED value is what gets served (-c) — the allocation is a contract the load
# HONORS, not a hint. None (unset resolver, or no ctx_pct for the model) ->
# today's default ctx path, byte-identical.
_CTX_RESOLVER = None


def set_ctx_resolver(fn) -> None:
    global _CTX_RESOLVER
    _CTX_RESOLVER = fn
# GPU offload for llama-server. -1 = put every layer on the GPU (the right
# default for a CUDA-built llama-server); 0 = CPU only; N = first N layers.
# Without this flag llama-server defaults to CPU — the usual "GPU sits idle".
DEFAULT_LLAMA_NGL = int(get_env_value("DEFAULT_LLAMA_NGL") or -1)
# Default serving mechanism for a model with no explicit serve_mode.
#
# HISTORY: this defaulted to `systemd` (one always-on unit per model) from an era
# when a "served model" *was* a local always-on llama-server. That predates the
# worker fleet + local slot pool: today routing is worker-first (resolve() →
# DelegatingRunner), then the on-demand slot pool, with a local server only as a
# fallback. An always-on-systemd default therefore (a) pins CPU/RAM on a possibly
# GPU-less central for models that workers already serve, and (b) makes the
# Serving overview report "systemd / always-on" for models that were never
# installed and aren't running. So the modern default is `swap`: on-demand,
# slot-first (serve_endpoint still prefers a slot), no pinned unit. An operator
# who genuinely wants a pinned local unit sets serve_mode=systemd per model, and
# DEFAULT_SERVE_MODE=<mode> (env) still overrides globally.
from ..._platform import IS_LINUX as _IS_LINUX
def _clean_mode_value(value) -> str:
    """Sanitize a serve-mode string from env/config: strip an inline
    ``# comment`` and surrounding whitespace, lowercase. An .env line like
    ``DEFAULT_SERVE_MODE=systemd   # one unit per model`` otherwise reaches
    ServeMode() verbatim and every serve-spec build raises ValueError
    (observed breaking GET /llm/serving fleet-wide on the op worker)."""
    return str(value or "").split("#", 1)[0].strip().lower()


def _default_serve_mode() -> str:
    explicit = _clean_mode_value(get_env_value("DEFAULT_SERVE_MODE"))
    if explicit:
        return explicit
    return "swap"
DEFAULT_SERVE_MODE = _default_serve_mode()

# Deterministic auto-port range for systemd units when a model has no explicit
# port. Wide span keeps hash collisions rare; an explicit cfg.port always wins.
LLAMA_PORT_BASE = int(get_env_value("LLAMA_PORT_BASE") or 7001)
LLAMA_PORT_SPAN = int(get_env_value("LLAMA_PORT_SPAN") or 4000)


class ServeMode(str, Enum):
    OFF = "off"
    SYSTEMD = "systemd"        # Linux only — one systemd unit per model
    SUPERVISED = "supervised"  # portable — detached background process per model
    SWAP = "swap"


# --------------------------------------------------------------------------- #
# small resolvers                                                             #
# --------------------------------------------------------------------------- #

def _unit_slug(value):
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-._")
    return value or "model"


def _bare_host(value):
    value = value or LLAMA_HOST
    if "://" in value:
        value = value.split("://", 1)[1]
    return value.split("/", 1)[0] or "127.0.0.1"


def _effective_extra(model_key, cfg) -> dict:
    """cfg.extra with the persisted per-model UI override merged on top."""
    extra = dict(getattr(cfg, "extra", {}) or {})
    try:
        from .overrides import get_override
        extra.update(get_override(model_key))
    except Exception:  # overrides are optional; never break spec resolution
        pass
    return extra


def _ctx_for(cfg, model_key, extra=None):
    extra = extra if extra is not None else _effective_extra(model_key, cfg)
    if extra.get("llama_ctx"):
        return int(extra["llama_ctx"])
    # CONTEXT allocation (slice 11): if the worker set a ctx_pct for this model,
    # SERVE the resolved value — the allocation is a contract, so the served -c
    # equals what fit/admission reserved KV for. The resolver already clamps to
    # DEFAULT_LLAMA_CTX (the 'capping' logic is now enforcement of the RESOLVED
    # value). Unset resolver / no ctx_pct -> falls through to today's default.
    if _CTX_RESOLVER is not None:
        try:
            resolved = _CTX_RESOLVER(model_key, cfg)
            if resolved:
                logger.info("%s: serving -c %s (ctx allocation — ctx_pct)",
                            model_key, int(resolved))
                return int(resolved)
        except Exception:  # noqa: BLE001 — a broken resolver never breaks serving
            pass
    mml = getattr(cfg, "model_max_length", None) or DEFAULT_LLAMA_CTX
    capped = min(int(mml), DEFAULT_LLAMA_CTX)
    if capped < int(mml):
        logger.info("%s: capping -c %s -> %s (set extra['llama_ctx'] to override)",
                    model_key, int(mml), capped)
    return capped


def _model_file_for(model_key, cfg):
    """Best-effort absolute GGUF path; '' if it can't be located yet. Never
    raises — a unit/config can be staged before the download lands."""
    # Operator-selected .gguf variant (UI serving control) wins over the
    # registry/auto resolution, so systemd/swap and in-process agree.
    try:
        from .overrides import resolve_override_gguf
        picked = resolve_override_gguf(model_key, get_model_path(model_key))
        if picked:
            return picked
    except Exception:
        pass
    # Fit-aware quant auto-pick ("most amicable gguf"): only reachable when NO
    # designation exists (autofit_gguf_prefer returns None for any pin /
    # cfg.filename), so the precedence chain per-worker pin -> model-wide pin ->
    # cfg.filename -> auto-pick -> election holds by construction.
    try:
        from .overrides import autofit_gguf_prefer
        pick = autofit_gguf_prefer(model_key, get_model_path(model_key), cfg)
        if pick:
            from ...imports.config.main import get_gguf_file
            got = get_gguf_file(get_model_path(model_key), cfg, prefer=pick)
            if got:
                return got
    except Exception:
        pass
    try:
        source = resolve_model_source(model_key)
        if source and os.path.isfile(source):
            return source
    except (FileNotFoundError, KeyError) as exc:
        logger.info("%s: resolve_model_source unavailable (%s)", model_key, exc)
    if getattr(cfg, "filename", None):
        return os.path.join(get_model_path(model_key), cfg.filename)
    return ""


def _auto_port(model_key: str) -> int:
    """Deterministic per-model port so a GGUF model can get a systemd unit
    without hand-assigning one. Stable across runs and identical whether
    resolved for a single model (the HTTP runner) or the whole batch (install),
    so the endpoint and the unit always agree. Explicit cfg.port still wins.
    """
    import hashlib
    h = int(hashlib.sha1(model_key.encode("utf-8")).hexdigest(), 16)
    return LLAMA_PORT_BASE + (h % LLAMA_PORT_SPAN)


def _resolve_port(model_key, cfg) -> int:
    p = getattr(cfg, "port", None)
    try:
        if p is not None and int(p) > 0:
            return int(p)
    except (TypeError, ValueError):
        pass
    return _auto_port(model_key)


def _resolve_mode(cfg, extra=None) -> ServeMode:
    if getattr(cfg, "framework", None) != "gguf":
        return ServeMode.OFF
    extra = extra if extra is not None else (getattr(cfg, "extra", {}) or {})
    explicit = _clean_mode_value(extra.get("serve_mode"))
    # systemd no longer falls back to off for a missing port — _resolve_port
    # auto-assigns a deterministic one.
    mode = ServeMode(explicit) if explicit else ServeMode(DEFAULT_SERVE_MODE)
    # systemd is Linux-only. An explicit systemd request elsewhere is a user
    # error worth surfacing; a default that resolved to systemd off-Linux can't
    # happen (see _default_serve_mode), but guard anyway by degrading gracefully.
    if mode is ServeMode.SYSTEMD and not _IS_LINUX:
        if explicit:
            raise ValueError(
                "serve_mode=systemd is Linux-only; use 'supervised' on this OS")
        mode = ServeMode.SUPERVISED
    return mode


# --------------------------------------------------------------------------- #
# schema                                                                      #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ServeSpec:
    model_key: str
    mode: ServeMode
    model_file: str = ""
    host: str = LLAMA_SWAP_HOST
    port: Optional[int] = None
    ctx_size: int = DEFAULT_LLAMA_CTX
    threads: int = DEFAULT_LLAMA_THREADS
    n_gpu_layers: int = DEFAULT_LLAMA_NGL
    # Was ``n_gpu_layers`` ACTUALLY SPECIFIED by someone (persisted override /
    # cfg.extra / a materialized k37 alloc_mode), or is the value above merely
    # the fill-in default (DEFAULT_LLAMA_NGL, historically -1)?
    #
    # This matters because -1 is overloaded: it is BOTH the historical fill-in
    # default AND a load-bearing explicit force ("Max GPU" — put every layer on
    # the card; see slot_agent._effective_ngl). Downstream placement policies
    # (notably the detected-MoE auto expert split in slot_agent._build_cmd) must
    # fire when nobody asked for a layer count, and must NOT fire when a caller
    # explicitly demanded -1. Collapsing the two made a 48 GB MoE launch with
    # --n-gpu-layers -1 and no --n-cpu-moe onto a 24 GB card (ae, 2026-07-25).
    #
    # Wire note: this is a LOCAL spec field only. It is never serialized to a
    # worker (central->worker relay ships model_dump()s of OTHER models under
    # extra=forbid), so it cannot break the frozen relay schema.
    ngl_explicit: bool = False
    always_on: bool = False   # modern default: on-demand, not a pinned unit
    ttl_seconds: Optional[int] = None
    user: str = LLAMA_SERVICE_USER
    group: str = LLAMA_SERVICE_GROUP
    working_directory: str = LLAMA_CPP_DIR
    server_bin: str = LLAMA_SERVER_BIN
    extra_args: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.mode is ServeMode.OFF:
            return
        if not os.path.isabs(self.server_bin):
            raise ValueError(f"{self.model_key}: server_bin must be absolute")
        if self.mode in (ServeMode.SYSTEMD, ServeMode.SUPERVISED):
            if not self.port or not 0 < self.port < 65536:
                raise ValueError(f"{self.model_key}: {self.mode.value} needs a valid port, got {self.port!r}")
            if not self.model_file:
                logger.info("%s: %s will reference an unresolved model path",
                            self.model_key, self.mode.value)
        if self.ctx_size < 1 or self.threads < 1:
            raise ValueError(f"{self.model_key}: ctx_size and threads must be >= 1")

    @property
    def unit_name(self) -> str:
        return _unit_slug(f"{LLAMA_UNIT_PREFIX}-{self.model_key}")

    @property
    def swap_name(self) -> str:
        return _unit_slug(self.model_key)

    @property
    def slot_n_gpu_layers(self):
        """``n_gpu_layers`` for a SLOT load, carrying its provenance.

        Identical in value to :attr:`n_gpu_layers` (same int, same ``str()``,
        same JSON), but when the value is only the fill-in default it is wrapped
        in ``slot_agent._NglDefaulted`` so the slot's placement policy still
        treats it as unset — which is what lets the detected-MoE auto expert
        split fire instead of launching all 48 layers onto a 24 GB card.

        Unit/swap/supervisor argv keep using :attr:`n_gpu_layers` unchanged."""
        if self.ngl_explicit:
            return self.n_gpu_layers
        try:
            from .slot_agent import _NglDefaulted
        except Exception:  # noqa: BLE001 — slot module optional; degrade to the int
            return self.n_gpu_layers
        return _NglDefaulted(self.n_gpu_layers)


def _ngl_for_alloc_mode(alloc_mode, model_file, extra) -> "int | None":
    """Materialize a persisted k37 ``alloc_mode`` onto the serve spec's
    n_gpu_layers (the -ngl the unit/in-process load uses), or None to keep the
    legacy extra/default resolution (mode unset, or the file is unresolvable
    so honest layer math is impossible — degrade, never guess 0).

    Same-box rail: the code computing this IS the version that honors it, so
    no version gate applies here (unlike central's relay emission seam)."""
    from ..alloc_modes import resolve_alloc_mode
    canonical, _ = resolve_alloc_mode(alloc_mode)
    if canonical is None:
        return None
    if canonical == "gpu-only":
        return -1
    if canonical == "ram-only":
        return 0
    if not model_file or not os.path.isfile(model_file):
        return None                    # can't price layers — keep legacy path
    try:
        from .. import spill as _spill
        if canonical == "max-ram":
            return _spill.maxram_gpu_layers(model_file)
        # max-gpu / explicit: honest fit-and-spill. explicit's VRAM target
        # (gpu_mem_gib) caps the fit; leniency-floor enforcement is the worker
        # admission engine's job (flex.plan_explicit_offload).
        fv = _spill.free_vram_bytes()
        try:
            tgt = extra.get("gpu_mem_gib")
            if canonical == "explicit" and tgt:
                cap = int(float(tgt) * 2 ** 30)
                fv = min(fv, cap) if fv else cap
        except (TypeError, ValueError):
            pass
        return _spill.autofit_gpu_layers(model_file, free_vram=fv)
    except Exception:  # noqa: BLE001 — spec build must never die on layer math
        return None


def serve_spec_for(model_key=None, *, cfg=None) -> ServeSpec:
    cfg = cfg if cfg is not None else get_model_config(model_key)
    model_key = cfg.model_key or model_key or cfg.name   # canonical registry key wins
    extra = _effective_extra(model_key, cfg)   # cfg.extra + persisted UI override
    mode = _resolve_mode(cfg, extra)

    if mode is ServeMode.SWAP:
        host, port = LLAMA_SWAP_HOST, LLAMA_SWAP_PORT
    elif mode in (ServeMode.SYSTEMD, ServeMode.SUPERVISED):
        host, port = _bare_host(getattr(cfg, "host", None)), _resolve_port(model_key, cfg)
    else:
        host = _bare_host(getattr(cfg, "host", None))
        port = int(cfg.port) if getattr(cfg, "port", None) else None

    model_file = _model_file_for(model_key, cfg)
    extra_args = list(extra.get("llama_extra_args") or ())
    # Vision GGUF: load the multimodal projector beside the model so the served
    # /v1/chat/completions accepts image_url content. No-op for text models.
    if "--mmproj" not in extra_args:
        from ...imports.src.utils import find_mmproj
        mmproj = find_mmproj(model_file)
        if mmproj:
            extra_args += ["--mmproj", mmproj]

    # MoE expert split (2026-07-24): a persisted n_cpu_moe override rides the
    # spec as llama-server argv (--n-cpu-moe N — experts of the first N MoE
    # layers stay on CPU; 999 = all). Explicit argv beats any unit-level
    # LLAMA_ARG_N_CPU_MOE env, so the served split is deterministic. Absent ->
    # no arg (the slot path applies the detected-MoE auto policy; the
    # systemd/swap spec stays byte-identical for dense/unset).
    _ncm = extra.get("n_cpu_moe")
    if _ncm not in (None, "") and "--n-cpu-moe" not in extra_args:
        try:
            extra_args += ["--n-cpu-moe", str(int(_ncm))]
        except (TypeError, ValueError):
            pass

    # k37: a persisted alloc_mode materializes onto n_gpu_layers here (gpu-only
    # -> -1, ram-only -> 0, max-gpu/explicit -> honest fit, max-ram -> inverted
    # RAM-first fit). Mode unset (the common case) keeps the legacy resolution
    # byte-identical.
    mode_ngl = _ngl_for_alloc_mode(extra.get("alloc_mode"), model_file, extra)

    # "Explicit" == somebody actually chose a layer placement: a materialized
    # k37 alloc_mode, or a persisted/registry n_gpu_layers. Absent both, the
    # spec's n_gpu_layers below is only DEFAULT_LLAMA_NGL — a fill-in, not a
    # demand — and downstream policy (MoE auto-split) is free to override it.
    ngl_explicit = (mode_ngl is not None
                    or extra.get("n_gpu_layers") not in (None, ""))

    return ServeSpec(
        model_key=model_key,
        mode=mode,
        model_file=model_file,
        host=host,
        port=port,
        ctx_size=_ctx_for(cfg, model_key, extra),
        threads=int(extra.get("threads") or DEFAULT_LLAMA_THREADS),
        n_gpu_layers=(int(mode_ngl) if mode_ngl is not None
                      else int(extra.get("n_gpu_layers", DEFAULT_LLAMA_NGL))),
        ngl_explicit=ngl_explicit,
        always_on=bool(extra.get("always_on", False)),
        ttl_seconds=extra.get("ttl_seconds") or (None if extra.get("always_on", False) else LLAMA_SWAP_TTL),
        extra_args=tuple(extra_args),
    )


def build_serve_specs(registry=None, *, only=None):
    registry = registry if registry is not None else get_model_registry()
    only = set(make_list(only)) if only else None
    specs = {}
    for key, cfg in registry.items():
        if only and key not in only:
            continue
        spec = serve_spec_for(key, cfg=cfg)
        specs[key] = spec
    return specs


def _assert_no_port_collisions(specs):
    seen = {}
    for spec in specs:
        seen.setdefault(spec.port, []).append(spec.model_key)
    clashes = {p: keys for p, keys in seen.items() if len(keys) > 1}
    if clashes:
        detail = "; ".join(f"{p} -> {keys}" for p, keys in sorted(clashes.items()))
        raise ValueError(f"systemd port collision: {detail}")


# --------------------------------------------------------------------------- #
# plan: a queue of writes + commands, executed only when you say so           #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class WriteFile:
    path: str
    content: str

    def describe(self) -> str:
        return f"write {self.path} ({len(self.content)} bytes)"


@dataclass(frozen=True)
class RunCmd:
    argv: Tuple[str, ...]

    def describe(self) -> str:
        return "$ " + " ".join(self.argv)


@dataclass(frozen=True)
class ServePlan:
    steps: Tuple[object, ...] = ()

    def __add__(self, other):
        return ServePlan(self.steps + other.steps)

    def describe(self):
        return [s.describe() for s in self.steps]


def apply_plan(plan, *, run=None, write=None):
    """Drain a ServePlan in order.

    Dry run (run=write=None) returns plan.describe() — hand that to the console.
    To execute, pass callables:
        run(argv)          e.g. partial(subprocess.run, check=True)
        write(path, text)  e.g. local writer, or an SSH writer for a peer node
    Local vs remote is entirely the caller's run/write — the plan is host-agnostic.
    """
    if run is None and write is None:
        return plan.describe()
    results = []
    for step in plan.steps:
        if isinstance(step, WriteFile):
            if write is None:
                raise RuntimeError("plan has file writes but no write() was provided")
            results.append(write(step.path, step.content))
        else:
            if run is None:
                raise RuntimeError("plan has commands but no run() was provided")
            results.append(run(list(step.argv)))
    return results


# --------------------------------------------------------------------------- #
# driver registry                                                             #
# --------------------------------------------------------------------------- #

_SERVE_DRIVERS = {}


def register_serve_driver(mode):
    def deco(cls):
        if mode in _SERVE_DRIVERS:
            raise KeyError(f"serve driver for {mode} already registered")
        _SERVE_DRIVERS[mode] = cls()          # stateless singleton
        return cls
    return deco


def get_serve_driver(mode):
    if mode not in _SERVE_DRIVERS:
        raise KeyError(f"no serve driver for mode={mode!r}; have {sorted(_SERVE_DRIVERS)}")
    return _SERVE_DRIVERS[mode]


# ---- systemd: one unit per model, always-on -------------------------------

@register_serve_driver(ServeMode.SYSTEMD)
class SystemdDriver:
    name = "systemd"

    def endpoint(self, spec):
        return f"http://{spec.host}:{spec.port}"

    def model_name(self, spec):
        return spec.swap_name

    def render_unit(self, spec):
        exec_start = " \\\n  ".join((
            spec.server_bin,
            f"-m {spec.model_file}",
            f"--host {spec.host}",
            f"--port {spec.port}",
            f"-c {spec.ctx_size}",
            f"-t {spec.threads}",
            # GPU offload — without this llama-server runs CPU-only.
            f"--n-gpu-layers {spec.n_gpu_layers}",
            *spec.extra_args,
        ))
        return "\n".join((
            "[Unit]",
            f"Description=llama.cpp server for {spec.model_key}",
            "After=network.target",
            # Restart-loop limiter lives in [Unit] (systemd ignores it in
            # [Service] — 'Unknown key name StartLimitIntervalSec in section
            # Service'). Caps thrash on a bad GGUF.
            "StartLimitIntervalSec=120",
            "StartLimitBurst=5",
            "",
            "[Service]",
            "Type=simple",
            f"User={spec.user}",
            f"Group={spec.group}",
            f"WorkingDirectory={spec.working_directory}",
            f"ExecStart={exec_start}",
            "Restart=always",
            "RestartSec=5",
            "TimeoutStartSec=300",
            "TimeoutStopSec=30",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "",
        ))

    def install_plan(self, specs):
        steps = []
        for spec in specs:
            path = os.path.join(SYSTEMD_UNIT_DIR, spec.unit_name + ".service")
            steps.append(WriteFile(path, self.render_unit(spec)))
        steps.append(RunCmd(("systemctl", "daemon-reload")))
        for spec in specs:
            unit = spec.unit_name + ".service"
            steps.append(RunCmd(("systemctl", "enable", "--now" if spec.always_on else unit, unit)
                                if spec.always_on else ("systemctl", "enable", unit)))
        return ServePlan(tuple(steps))

    def start_plan(self, spec):
        return ServePlan((RunCmd(("systemctl", "start", spec.unit_name + ".service")),))

    def stop_plan(self, spec):
        return ServePlan((RunCmd(("systemctl", "stop", spec.unit_name + ".service")),))

    def status_plan(self, spec):
        return ServePlan((RunCmd(("systemctl", "is-active", spec.unit_name + ".service")),))


# ---- swap: one shared llama-swap proxy, on-demand -------------------------

@register_serve_driver(ServeMode.SWAP)
class SwapDriver:
    name = "swap"

    def endpoint(self, spec):
        return f"http://{spec.host}:{spec.port}"          # the proxy, same for all

    def model_name(self, spec):
        return spec.swap_name                             # request "model" field

    def render_config(self, specs):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("swap mode needs PyYAML: pip install pyyaml") from exc

        models = {}
        members = []
        for spec in specs:
            slug = spec.swap_name
            cmd = " ".join((
                spec.server_bin, "-m", spec.model_file,
                "--host", "127.0.0.1", "--port", "${PORT}",
                "-c", str(spec.ctx_size), "-t", str(spec.threads),
                "--n-gpu-layers", str(spec.n_gpu_layers),
                *spec.extra_args,
            ))
            entry = {"cmd": cmd}
            if spec.always_on:
                members.append(slug)
            elif spec.ttl_seconds:
                entry["ttl"] = spec.ttl_seconds
            models[slug] = entry

        cfg = {"models": models}
        if members:
            # always-on models share a group so they coexist instead of evicting
            # each other. NOTE: verify the coexistence flag names against your
            # llama-swap version — group schema has shifted across releases.
            cfg["groups"] = {"persistent": {"swap": False, "exclusive": False, "members": members}}
        return yaml.safe_dump(cfg, sort_keys=False)

    def install_plan(self, specs):
        steps = [
            WriteFile(LLAMA_SWAP_CONFIG, self.render_config(specs)),
            RunCmd(("systemctl", "restart", LLAMA_SWAP_UNIT)),
        ]
        return ServePlan(tuple(steps))

    def start_plan(self, spec):
        # on-demand: the proxy loads on first request. Optional warmup hit.
        return ServePlan(())

    def stop_plan(self, spec):
        url = f"http://{spec.host}:{spec.port}/unload?model={spec.swap_name}"
        return ServePlan((RunCmd(("curl", "-s", "-X", "POST", url)),))

    def status_plan(self, spec):
        return ServePlan((RunCmd(("curl", "-s", f"http://{spec.host}:{spec.port}/running")),))


# ---- supervised: portable always-on, no init system -----------------------

@register_serve_driver(ServeMode.SUPERVISED)
class SupervisedDriver:
    """One detached ``llama-server`` per model, supervised by hugpy itself.

    The cross-platform replacement for systemd units. Plans shell out to the
    ``hugpy.managers.serve.supervisor`` subcommand, which spawns/kills the child
    via the OS-appropriate process primitive and tracks it in a pidfile under the
    per-OS data dir. No unit files, no root, works on Windows/macOS/Linux.
    """
    name = "supervised"

    def endpoint(self, spec):
        return f"http://{spec.host}:{spec.port}"

    def model_name(self, spec):
        return spec.swap_name

    def _cmd(self, action, model_key):
        return RunCmd((sys.executable, "-m", "abstract_hugpy_dev.managers.serve.supervisor",
                       action, model_key))

    def install_plan(self, specs):
        # Nothing to write ahead of time; "install" == start each always-on model.
        steps = [self._cmd("start", s.model_key) for s in specs if s.always_on]
        return ServePlan(tuple(steps))

    def start_plan(self, spec):
        return ServePlan((self._cmd("start", spec.model_key),))

    def stop_plan(self, spec):
        return ServePlan((self._cmd("stop", spec.model_key),))

    def status_plan(self, spec):
        return ServePlan((self._cmd("status", spec.model_key),))


# ---- off: in-process only, nothing to install -----------------------------

@register_serve_driver(ServeMode.OFF)
class OffDriver:
    name = "off"

    def endpoint(self, spec):
        return None                       # runner falls back to in-process

    def model_name(self, spec):
        return spec.swap_name

    def install_plan(self, specs):
        return ServePlan(())

    def start_plan(self, spec):
        return ServePlan(())

    def stop_plan(self, spec):
        return ServePlan(())

    def status_plan(self, spec):
        return ServePlan(())


# --------------------------------------------------------------------------- #
# top-level API — what the runner and the console call                        #
# --------------------------------------------------------------------------- #

def serve_endpoint(model_key) -> Optional[str]:
    """Base URL the HTTP runner should hit, or None for in-process.

    For llama.cpp models the slot pool is preferred: a free/loaded slot serves
    the model on the GPU, and only when every slot is busy do we fall back to
    the swap proxy (the configured overflow). Non-llama models stay in-process.
    """
    # Per-box "never serve locally" policy: every serve mode here (slot pool,
    # llama-swap proxy, local systemd unit) is LOCAL to this box, so a policy box
    # must expose no local endpoint at all. Return None -> the HTTP runner treats
    # it as mode=off and get_llama_runner's own policy gate raises the actionable
    # LocalEngineUnavailable. Default off === today's behavior. See .policy.
    from .policy import no_local_serving
    if no_local_serving():
        return None

    # Resolve a possibly-bare/ambiguous key to its canonical registry key,
    # preferring a variant already loaded in a slot so we reuse it instead of
    # loading a second copy. Everything downstream uses the canonical key.
    pool = None
    prefer = None
    try:
        from .slots import SlotPool, slots_enabled
        if slots_enabled():
            pool = SlotPool()
            prefer = [s.get("model_key") for s in pool.statuses() if s.get("model_key")]
    except Exception as exc:  # never let slot scheduling break serving
        logger.warning("slot status lookup failed for %s: %s", model_key, exc)

    cfg = get_model_config(model_key, prefer=prefer)
    model_key = cfg.model_key                       # canonical from here on
    spec = serve_spec_for(model_key, cfg=cfg)
    if spec.mode is ServeMode.OFF:
        return None

    try:
        if pool is not None:
            endpoint = pool.endpoint_for(model_key)
            if endpoint:
                return endpoint
            logger.info("all slots busy; routing %s via swap proxy", model_key)
            return f"http://{LLAMA_SWAP_HOST}:{LLAMA_SWAP_PORT}"
    except Exception as exc:  # never let slot scheduling break serving
        logger.warning("slot routing failed for %s: %s; using %s",
                       model_key, exc, spec.mode.value)

    return get_serve_driver(spec.mode).endpoint(spec)


def serve_model_name(model_key) -> str:
    """Value to put in the request 'model' field (matters for swap routing)."""
    spec = serve_spec_for(model_key)
    return get_serve_driver(spec.mode).model_name(spec)


def install_serving(*, only=None, registry=None) -> ServePlan:
    """One plan that stands up everything. Batches by mode so all swap models
    land in one config write, each systemd model in its own unit."""
    specs = build_serve_specs(registry=registry, only=only)
    # Port-bound modes (systemd unit / supervised child) must not clash on a port.
    _assert_no_port_collisions([s for s in specs.values()
                                if s.mode in (ServeMode.SYSTEMD, ServeMode.SUPERVISED)])
    by_mode = {}
    for spec in specs.values():
        by_mode.setdefault(spec.mode, []).append(spec)

    plan = ServePlan(())
    for mode, mode_specs in by_mode.items():
        plan = plan + get_serve_driver(mode).install_plan(mode_specs)
    return plan


def start_serving(model_key) -> ServePlan:
    spec = serve_spec_for(model_key)
    return get_serve_driver(spec.mode).start_plan(spec)


def stop_serving(model_key) -> ServePlan:
    spec = serve_spec_for(model_key)
    return get_serve_driver(spec.mode).stop_plan(spec)


def serving_overview(registry=None):
    """Rows for the console: what each model's serving looks like (no side effects)."""
    specs = build_serve_specs(registry=registry)
    rows = []
    for key, spec in specs.items():
        driver = get_serve_driver(spec.mode)
        rows.append(spec_row(spec, driver))
    return rows


def spec_row(spec, driver=None) -> dict:
    """One serving row for the console — the editable knobs + resolved endpoint."""
    driver = driver or get_serve_driver(spec.mode)
    return {
        "key": spec.model_key,
        "mode": spec.mode.value,
        "always_on": spec.always_on,
        "endpoint": driver.endpoint(spec),
        "model_name": driver.model_name(spec),
        "port": spec.port,
        "n_gpu_layers": spec.n_gpu_layers,
        "threads": spec.threads,
        "ctx_size": spec.ctx_size,
        "ttl_seconds": spec.ttl_seconds,
    }
