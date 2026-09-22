"""Let a phone act as a GPU for sharded LLM workloads (an RPC shard backend).

A phone has no discrete GPU, but llama.cpp's ``rpc-server`` happily pools plain
system RAM/CPU. This module lets a phone *lend that compute* to the central
fleet exactly the way a real GPU box does with ``--role rpc`` in
:mod:`abstract_hugpy_dev.worker_agent`: it

    1. spawns ``rpc-server`` (the llama.cpp distributed-inference backend),
    2. registers with central's GPU worker pool (``/api/llm/workers/register``)
       as ``role="rpc"``, advertising an ``rpc_endpoint`` central's allocator can
       offload layers to, and
    3. heartbeats its live free memory so the allocator's snapshot stays fresh.

The one wrinkle is how the allocator *sees* a phone. Central derives each
worker's poolable ``free_vram`` from ``gpus[].memory_free`` (see
``fleet_snapshot``), and the shard path only pools nodes with ``free_vram > 0``.
A phone reports no GPUs, so we advertise a single *synthetic* GPU whose
``memory_free`` is the phone's RAM budget — i.e. we present the RAM the
rpc-server can actually pool *as* VRAM. The allocator then folds the phone into
a shard plan with a RAM-proportional ``tensor_split`` and ships its endpoint to
the lead as ``--rpc host:port``.

Deliberately dependency-light (stdlib + ``urllib`` only) so it still runs on a
Termux phone. Enabled only when ``PHONE_BRICK_RPC`` is set; otherwise importing
this module is inert and the phone behaves exactly as before.

Config (all via env):
    PHONE_BRICK_RPC          enable the RPC backend (1/true/yes/on). Required.
    PHONE_BRICK_CENTRAL      base URL of central, e.g. http://10.8.0.1:7002 or
                             https://hugpy.example/api. Required to register.
    PHONE_BRICK_NAME         display name in the console (default: hostname).
    PHONE_BRICK_RPC_PORT     rpc-server port / advertised endpoint (default 50052).
    PHONE_BRICK_RPC_BIN      path to the rpc-server binary (default: resolve via
                             the engine resolver, else ``rpc-server`` on PATH).
    PHONE_BRICK_RPC_HOST     host to advertise in rpc_endpoint (default: this
                             phone's LAN IP toward central, else its hostname).
    PHONE_BRICK_RPC_RAM_GIB  RAM budget to advertise as poolable, in GiB
                             (default: detected available RAM).
    PHONE_BRICK_HEARTBEAT    seconds between heartbeats (default 20).
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

_ID_FILE = os.path.expanduser("~/.phone-brick/rpc_worker_id")
_DEFAULT_HEARTBEAT_S = 20.0
_DEFAULT_RPC_PORT = 50052
# When we can't probe RAM and the operator didn't pin a budget, advertise a
# conservative 2 GiB so the phone is still poolable rather than invisible.
_FALLBACK_RAM_BYTES = 2 * 1024**3


# ---------------------------------------------------------------------------
# Small, dependency-light probes (mirror hugpy._platform.hardware, inlined to
# avoid pulling any import chain onto a Termux phone)
# ---------------------------------------------------------------------------
def _free_ram_bytes() -> Optional[int]:
    """Available system RAM in bytes, best-effort (psutil → /proc/meminfo)."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 — psutil is optional on Termux
        pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001
        pass
    return None


def _local_ip_toward(central_url: str) -> Optional[str]:
    """This phone's outbound LAN IP on the route to ``central_url``.

    Connecting a UDP socket toward central (no packet is sent) makes the kernel
    pick the source address it *would* use, i.e. the phone's real LAN IP rather
    than loopback. Central can't derive this reliably for a phone behind NAT.
    """
    try:
        host = central_url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
        if not host:
            return None
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((host, 80))
            return sock.getsockname()[0]
        finally:
            sock.close()
    except Exception:  # noqa: BLE001
        return None


def _resolve_rpc_bin(explicit: Optional[str]) -> str:
    """Locate the rpc-server binary: explicit override → engine resolver → PATH."""
    if explicit:
        return explicit
    try:
        from hugpy_engine.native.resolve import rpc_bin as _resolve

        found = _resolve()
        if found:
            return found
    except Exception:  # noqa: BLE001 — resolver is optional on a phone
        pass
    return "rpc-server"


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _post(url: str, body: dict, timeout: float = 10.0) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return resp.status, (json.loads(raw) if raw else {})


def _cached_id() -> Optional[str]:
    try:
        with open(_ID_FILE, "r", encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _cache_id(worker_id: str) -> None:
    try:
        os.makedirs(os.path.dirname(_ID_FILE), exist_ok=True)
        with open(_ID_FILE, "w", encoding="utf-8") as fh:
            fh.write(worker_id)
    except OSError:
        pass  # a non-writable home just means we re-register on restart


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RpcBackendConfig:
    """How a phone advertises itself as a shard RPC backend."""

    central: str
    name: str
    rpc_host: str            # advertised in rpc_endpoint (reachable by the lead)
    rpc_port: int = _DEFAULT_RPC_PORT
    rpc_bin: str = "rpc-server"
    ram_bytes: int = _FALLBACK_RAM_BYTES   # poolable RAM, advertised as pseudo-VRAM
    interval_s: float = _DEFAULT_HEARTBEAT_S

    @property
    def rpc_endpoint(self) -> str:
        return f"{self.rpc_host}:{self.rpc_port}"


def config_from_env() -> Optional[RpcBackendConfig]:
    """Build a :class:`RpcBackendConfig` from env, or ``None`` if disabled.

    Returns ``None`` unless ``PHONE_BRICK_RPC`` is set *and* a central URL is
    configured — so a phone that hasn't opted in is wholly unaffected.
    """
    if not _env_flag("PHONE_BRICK_RPC", False):
        return None
    central = os.environ.get("PHONE_BRICK_CENTRAL")
    if not central:
        print("[phone-brick/rpc] PHONE_BRICK_RPC set but PHONE_BRICK_CENTRAL "
              "is missing — cannot register as a shard backend")
        return None

    ram_gib = os.environ.get("PHONE_BRICK_RPC_RAM_GIB")
    if ram_gib:
        ram_bytes = int(float(ram_gib) * 1024**3)
    else:
        ram_bytes = _free_ram_bytes() or _FALLBACK_RAM_BYTES

    rpc_host = (os.environ.get("PHONE_BRICK_RPC_HOST")
                or _local_ip_toward(central)
                or socket.gethostname())

    return RpcBackendConfig(
        central=central,
        name=os.environ.get("PHONE_BRICK_NAME") or socket.gethostname(),
        rpc_host=rpc_host,
        rpc_port=int(os.environ.get("PHONE_BRICK_RPC_PORT", _DEFAULT_RPC_PORT)),
        rpc_bin=_resolve_rpc_bin(os.environ.get("PHONE_BRICK_RPC_BIN")),
        ram_bytes=ram_bytes,
        interval_s=float(os.environ.get("PHONE_BRICK_HEARTBEAT", _DEFAULT_HEARTBEAT_S)),
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class RpcBackendAgent:
    """Spawns ``rpc-server`` and keeps the phone registered as a shard backend."""

    def __init__(self, config: RpcBackendConfig):
        self.config = config
        # Endpoints live under /api/llm/workers on the central Flask app — the
        # SAME pool the GPU worker agent joins, so the allocator can shard onto
        # this phone. (The phone-brick vision pool, under /phone-brick, is separate.)
        self._base = config.central.rstrip("/") + "/api/llm/workers"
        self._id = _cached_id()
        self._proc = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # -- rpc-server lifecycle ---------------------------------------------
    def _spawn_rpc_server(self):
        """Launch llama.cpp's rpc-server, bound on all interfaces.

        Returns the process handle, or ``None`` if the binary is missing — in
        which case the phone still registers and heartbeats, it just isn't
        usable as a backend until an rpc-server build is installed.
        """
        import subprocess

        # Bind 0.0.0.0 so the lead can reach us; advertise the routable host.
        cmd = [self.config.rpc_bin, "-H", "0.0.0.0", "-p", str(self.config.rpc_port)]
        try:
            try:
                from hugpy_platform.procutil import popen_detached

                proc = popen_detached(cmd)
            except Exception:  # noqa: BLE001 — fall back to a plain child
                proc = subprocess.Popen(cmd)  # noqa: S603 — operator-controlled args
            print(f"[phone-brick/rpc] rpc-server up: {' '.join(cmd)} (pid {proc.pid})")
            return proc
        except FileNotFoundError:
            print(f"[phone-brick/rpc] rpc-server binary {self.config.rpc_bin!r} not "
                  "found — install a llama.cpp RPC build (cmake -DGGML_RPC=ON) and "
                  "set PHONE_BRICK_RPC_BIN. Registering anyway, but this phone "
                  "can't serve as a shard backend yet.")
            return None
        except OSError as exc:
            print(f"[phone-brick/rpc] failed to start rpc-server: {exc}")
            return None

    # -- registration ------------------------------------------------------
    def _register_payload(self) -> dict:
        """The /register body. A single synthetic GPU carries the RAM budget as
        ``memory_free`` so central's ``fleet_snapshot`` gives this phone a
        non-zero ``free_vram`` and the allocator will pool it as a backend."""
        ram = self.config.ram_bytes
        return {
            "name": self.config.name,
            "url": None,                 # rpc backends don't serve /infer
            "port": self.config.rpc_port,
            "gpus": [{
                "index": 0,
                "name": "phone-cpu-rpc",
                "memory_total": ram,
                "memory_free": ram,
            }],
            "role": "rpc",
            "rpc_endpoint": self.config.rpc_endpoint,
            "free_ram": _free_ram_bytes(),
            "worker_id": self._id,
        }

    def _heartbeat_payload(self) -> dict:
        """Live status: refresh the synthetic GPU's free memory each tick so the
        allocator sees current availability."""
        free = _free_ram_bytes() or self.config.ram_bytes
        poolable = min(free, self.config.ram_bytes)
        return {
            "gpus": [{
                "index": 0,
                "name": "phone-cpu-rpc",
                "memory_total": self.config.ram_bytes,
                "memory_free": poolable,
            }],
            "role": "rpc",
            "rpc_endpoint": self.config.rpc_endpoint,
            "free_ram": free,
            "port": self.config.rpc_port,
        }

    def _register(self) -> bool:
        try:
            _, resp = _post(f"{self._base}/register", self._register_payload())
            self._id = resp.get("id") or self._id
            if self._id:
                _cache_id(self._id)
            print(f"[phone-brick/rpc] registered as shard backend {self.config.name} "
                  f"(id={self._id}) endpoint={self.config.rpc_endpoint}")
            return True
        except Exception as exc:  # noqa: BLE001 — keep retrying on the next tick
            print(f"[phone-brick/rpc] register failed: {type(exc).__name__}: {exc}")
            return False

    # -- run loop ----------------------------------------------------------
    def _loop(self) -> None:
        if not self._id:
            self._register()
        while True:
            try:
                if not self._id:
                    self._register()
                else:
                    status, _ = _post(f"{self._base}/{self._id}/heartbeat",
                                      self._heartbeat_payload())
                    if status == 410:  # central forgot us → re-register
                        self._register()
            except urllib.error.HTTPError as exc:
                if exc.code == 410:
                    self._register()
                else:
                    print(f"[phone-brick/rpc] heartbeat HTTP {exc.code}")
            except Exception as exc:  # noqa: BLE001 — transient; retry next tick
                print(f"[phone-brick/rpc] heartbeat failed: {type(exc).__name__}: {exc}")
            time.sleep(self.config.interval_s)

    def start(self) -> "RpcBackendAgent":
        self._proc = self._spawn_rpc_server()
        self._thread.start()
        return self


def maybe_start_rpc_backend() -> Optional[RpcBackendAgent]:
    """Start the RPC shard backend iff ``PHONE_BRICK_RPC`` (and central) are set.

    Safe to call unconditionally (e.g. from the worker's ``build_app``): returns
    ``None`` and does nothing when the phone hasn't opted in.
    """
    config = config_from_env()
    if config is None:
        return None
    return RpcBackendAgent(config).start()


def run() -> int:
    """Run the RPC backend standalone (spawn rpc-server, register, block).

    Used by ``python -m abstract_hugpy_dev.phone_brick rpc-backend``. Unlike
    :func:`maybe_start_rpc_backend`, this errors out loudly if not configured.
    """
    config = config_from_env()
    if config is None:
        print("[phone-brick/rpc] not configured — set PHONE_BRICK_RPC=1 and "
              "PHONE_BRICK_CENTRAL=<url>")
        return 2
    print(f"PHONE BRICK RPC BACKEND for {config.central}")
    print(f"  endpoint : {config.rpc_endpoint}  (rpc-server: {config.rpc_bin})")
    print(f"  poolable : {config.ram_bytes / 1024**3:.1f} GiB advertised as VRAM")
    agent = RpcBackendAgent(config)
    agent.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0
