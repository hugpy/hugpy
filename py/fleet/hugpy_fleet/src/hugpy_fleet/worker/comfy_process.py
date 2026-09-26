"""ComfyUI process lifecycle for the worker — the worker OWNS comfy the same way
it owns an LLM model server (operator ruling 2026-09-24: "i dont have a problem
with the worker owning the service, thats my ideal dist").

ComfyUI must not squat VRAM as an always-on daemon outside hugpy's accounting.
Instead the worker starts it on demand when a comfy job is placed here, makes
room first via the existing evict-to-fit, serves, then frees VRAM / stops it
when idle (the idle watchdog), and restarts it on crash when there is demand.

THREE PLUGGABLE LAUNCHERS (``HUGPY_COMFY_LAUNCH``):

  * ``external`` (the DEFAULT when nothing is configured) — the worker only
    probes comfy and NEVER starts or stops it. This is exactly today's
    behaviour, so every existing box is unchanged until it opts in.
  * ``systemd-user:<unit>`` — comfy is a systemd USER unit owned by the same
    user as the worker (ae's ``7103_hugpy_comfy.service`` under aeb, computron's
    ``comfyui.service``). Controlled with ``systemctl --user start/stop <unit>``.
    Because the worker runs as a systemd user service it already carries
    ``XDG_RUNTIME_DIR``; we set it defensively from the uid when it is absent.
  * ``spawn:<python> <main.py> [args...]`` — the worker spawns ComfyUI itself as
    a supervised child (registered in pid_registry so its VRAM is attributed and
    it is never an orphan-reap target).

Every box-touching capability (running a command, probing readiness, spawning a
child, the clock) is a constructor arg, so the manager is unit-testable with no
systemd, no ComfyUI and no GPU. The worker binds the real ones at boot.
"""
from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Default readiness ceiling for an on-demand start. ComfyUI's cold import is
# slow (the aeb unit sets TimeoutStartSec=300); this is the worker-side bound on
# how long a comfy /infer will wait for the server before recording a load
# failure. ``HUGPY_COMFY_START_TIMEOUT_S`` overrides it per box.
_DEFAULT_START_TIMEOUT_S = 180.0
_READY_POLL_S = 1.0
_DEFAULT_URL = "http://127.0.0.1:8188"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


def start_timeout_s() -> float:
    """Readiness ceiling for an on-demand start (``HUGPY_COMFY_START_TIMEOUT_S``,
    seconds)."""
    return _env_float("HUGPY_COMFY_START_TIMEOUT_S", _DEFAULT_START_TIMEOUT_S)


def comfy_url() -> str:
    """The adopted comfy base URL. ``HUGPY_COMFY_URL`` first (the launcher's own
    knob), then ``COMFY_URL`` (the existing worker/runner setting), then the
    package default — so a box that already sets COMFY_URL keeps working and a
    box that sets only HUGPY_COMFY_URL works too."""
    return (os.environ.get("HUGPY_COMFY_URL")
            or os.environ.get("COMFY_URL")
            or _DEFAULT_URL).rstrip("/")


def parse_launch(raw: Optional[str]) -> dict:
    """Parse ``HUGPY_COMFY_LAUNCH`` into a launcher spec.

    Returns ``{"kind": "external"}`` (never start/stop) for an empty/absent
    value or the literal ``external``; ``{"kind": "systemd-user", "unit": ...}``;
    ``{"kind": "spawn", "python": ..., "main": ..., "args": [...]}``. A
    malformed value degrades to ``external`` and carries an ``error`` note so the
    heartbeat/log can surface WHY it was ignored (never a crash, never a silent
    guess of a different launcher)."""
    s = (raw or "").strip()
    if not s or s.lower() == "external":
        return {"kind": "external"}
    low = s.lower()
    if low.startswith("systemd-user:"):
        unit = s.split(":", 1)[1].strip()
        if not unit:
            return {"kind": "external",
                    "error": "HUGPY_COMFY_LAUNCH=systemd-user: is missing the "
                             "unit name (e.g. systemd-user:comfyui.service)"}
        return {"kind": "systemd-user", "unit": unit}
    if low.startswith("spawn:"):
        rest = s.split(":", 1)[1].strip()
        try:
            parts = shlex.split(rest)
        except ValueError as exc:
            return {"kind": "external",
                    "error": f"HUGPY_COMFY_LAUNCH spawn args unparseable ({exc})"}
        if len(parts) < 2:
            return {"kind": "external",
                    "error": "HUGPY_COMFY_LAUNCH=spawn:<python> <main.py> [args] "
                             "needs at least a python and a main.py"}
        return {"kind": "spawn", "python": parts[0], "main": parts[1],
                "args": parts[2:]}
    return {"kind": "external",
            "error": f"unrecognized HUGPY_COMFY_LAUNCH={s!r} (expected external "
                     "| systemd-user:<unit> | spawn:<python> <main.py> [args])"}


def _user_env() -> dict:
    """The environment for a ``systemctl --user`` call: the live environ, with
    ``XDG_RUNTIME_DIR`` filled in from the uid when the worker's own env lacks it
    (a worker not started as a systemd user service). No-op when it is set."""
    env = dict(os.environ)
    if not env.get("XDG_RUNTIME_DIR"):
        try:
            env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
        except Exception:  # noqa: BLE001 — non-POSIX: leave it unset
            pass
    return env


def _default_runner(cmd, timeout: float = 30.0,
                    env: Optional[dict] = None) -> "tuple[int, str, str]":
    """Run ``cmd`` and return ``(returncode, stdout, stderr)``. A launch failure
    (binary missing, timeout) is returned as a non-zero code with the reason in
    stderr — never raised — so start()/stop() always have a real reason."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=env)
        return p.returncode, p.stdout or "", p.stderr or ""
    except FileNotFoundError as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    except subprocess.TimeoutExpired as exc:
        return 124, "", f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 — a control command never crashes a beat
        return 1, "", f"{type(exc).__name__}: {exc}"


def _default_readiness(url: str, timeout: float = 2.0) -> bool:
    """True when ComfyUI answers ``GET /system_stats`` with 200 — comfy's own
    readiness signal, the same probe _comfy_status uses."""
    try:
        import httpx
    except Exception:  # noqa: BLE001 — no httpx: cannot prove readiness
        return False
    try:
        r = httpx.get(url.rstrip("/") + "/system_stats", timeout=timeout)
        return r.status_code == 200
    except Exception:  # noqa: BLE001 — down / unreachable
        return False


class ComfyManager:
    """The worker's comfy process lifecycle. One per worker.

    Injectable seams (all defaulted to the real box-touching implementations):
      * ``runner(cmd, timeout, env)`` -> ``(rc, out, err)`` — runs systemctl.
      * ``readiness(url, timeout)`` -> bool — comfy /system_stats readiness.
      * ``popen(argv, **kw)`` -> a Popen-like child (spawn launcher).
      * ``clock()`` / ``sleep(s)`` — time.
      * ``checkpoint_dirs()`` -> the dirs to scan for checkpoints while stopped.
      * ``diagnostics(spec)`` -> recent journal/status text for a failed start.
    """

    def __init__(self, spec: dict, url: str, *,
                 runner: "Optional[Callable]" = None,
                 readiness: "Optional[Callable]" = None,
                 popen: "Optional[Callable]" = None,
                 clock: "Optional[Callable[[], float]]" = None,
                 sleep: "Optional[Callable[[float], None]]" = None,
                 checkpoint_dirs: "Optional[Callable[[], list]]" = None,
                 diagnostics: "Optional[Callable[[dict], str]]" = None) -> None:
        self.spec = spec or {"kind": "external"}
        self.url = (url or _DEFAULT_URL).rstrip("/")
        self._runner = runner or _default_runner
        self._readiness = readiness or _default_readiness
        self._popen = popen or subprocess.Popen
        self._clock = clock or time.time
        self._sleep = sleep or time.sleep
        if checkpoint_dirs is not None:
            self._checkpoint_dirs = checkpoint_dirs
        else:
            from hugpy_fleet.worker.comfy_ledger import checkpoint_dirs as _cd
            self._checkpoint_dirs = _cd
        self._diagnostics = diagnostics or self._default_diagnostics
        self._child = None            # spawn: the live Popen
        self._starting = False
        self._last_error: Optional[str] = None
        self._last_error_class: Optional[str] = None

    # -- identity ------------------------------------------------------------
    @property
    def mode(self) -> str:
        return str(self.spec.get("kind") or "external")

    @property
    def managed(self) -> bool:
        """True when the worker controls comfy's lifecycle (systemd-user/spawn);
        False for the ``external`` default (probe only, never start/stop)."""
        return self.mode in ("systemd-user", "spawn")

    @property
    def spec_error(self) -> Optional[str]:
        return self.spec.get("error")

    # -- readiness / state ---------------------------------------------------
    def is_running(self, timeout: float = 2.0) -> bool:
        return bool(self._readiness(self.url, timeout))

    def pid(self) -> Optional[int]:
        """The comfy PID, for pid_registry attribution. The live child (spawn)
        or the unit's MainPID (systemd-user); None when unknown / not running."""
        if self.mode == "spawn":
            child = self._child
            if child is not None and child.poll() is None:
                return int(child.pid)
            return None
        if self.mode == "systemd-user":
            unit = self.spec.get("unit")
            rc, out, _err = self._runner(
                ["systemctl", "--user", "show", str(unit), "-p", "MainPID",
                 "--value"], timeout=10.0, env=_user_env())
            if rc == 0:
                try:
                    p = int((out or "0").strip())
                    return p or None
                except ValueError:
                    return None
        return None

    def state(self, running: "Optional[bool]" = None) -> str:
        """One of ``running`` | ``starting`` | ``stopped`` | ``failed``. Pass a
        cached ``running`` to avoid a second readiness probe in the heartbeat."""
        if self._starting:
            return "starting"
        alive = self.is_running() if running is None else bool(running)
        if alive:
            return "running"
        if self._last_error:
            return "failed"
        return "stopped"

    # -- checkpoints (advertised even while stopped) -------------------------
    def checkpoints_on_disk(self, limit: int = 50) -> list:
        """Checkpoint file names discoverable on disk under comfy's configured
        checkpoint dirs — so the worker keeps advertising what it CAN serve even
        while comfy is stopped-but-startable, and central can still place a comfy
        job onto a stopped managed box. Best-effort; ``[]`` when nothing is
        readable. Bare names (basename), deduped, sorted, capped."""
        exts = (".safetensors", ".ckpt", ".sft", ".pt", ".pth")
        found: set = set()
        try:
            dirs = self._checkpoint_dirs() or []
        except Exception:  # noqa: BLE001 — unreadable config: advertise nothing
            return []
        for root in dirs:
            try:
                for cur, _sub, files in os.walk(root, followlinks=True):
                    for f in files:
                        if f.lower().endswith(exts):
                            found.add(f)
                            if len(found) >= 5000:   # a sane scan bound
                                break
            except OSError:
                continue
        return sorted(found)[:limit]

    # -- lifecycle -----------------------------------------------------------
    def start(self, ready_timeout: "Optional[float]" = None) -> dict:
        """Start comfy on demand and wait for readiness. Returns
        ``{ok, note, launcher, pid?, duration_s?, load_class?, loader_stderr?}``.

        NOT a start for the ``external`` launcher — it reports the current probe
        state and never touches the process. On a managed box a failure carries a
        machine-readable ``load_class`` + the real launcher/journal text
        (``loader_stderr``) so the caller can raise the same structured load
        failure an LLM load would."""
        if not self.managed:
            running = self.is_running()
            return {"ok": running, "launcher": "external",
                    "note": ("comfy is up (external launcher — the worker does "
                             "not manage it)" if running else
                             "comfy is down and the launcher is 'external' — the "
                             "worker does not start it")}
        if self.spec_error:
            return {"ok": False, "launcher": self.mode,
                    "load_class": "engine_unavailable",
                    "loader_stderr": self.spec_error,
                    "note": f"comfy launcher misconfigured: {self.spec_error}"}
        if self.is_running():
            return {"ok": True, "launcher": self.mode, "pid": self.pid(),
                    "note": "comfy already running"}
        ceiling = start_timeout_s() if ready_timeout is None else ready_timeout
        t0 = self._clock()
        self._starting = True
        try:
            launched, launch_note = self._launch()
            if not launched:
                self._last_error = launch_note
                self._last_error_class = "engine_unavailable"
                return {"ok": False, "launcher": self.mode,
                        "load_class": "engine_unavailable",
                        "loader_stderr": launch_note,
                        "note": f"comfy {self.mode} launch failed: {launch_note}"}
            # Wait for comfy's OWN readiness signal, bounded.
            deadline = t0 + ceiling
            while self._clock() < deadline:
                if self.is_running():
                    dur = self._clock() - t0
                    self._last_error = None
                    self._last_error_class = None
                    logger.info("comfy started via %s in %.1fs (%s)",
                                self.mode, dur, self.url)
                    return {"ok": True, "launcher": self.mode, "pid": self.pid(),
                            "duration_s": dur, "note": "comfy started and ready"}
                # spawn: a child that exited before readiness is a hard failure
                # with its own reason — never wait the full ceiling on a corpse.
                if self.mode == "spawn":
                    diag = self._child_exit_note()
                    if diag is not None:
                        self._last_error = diag
                        self._last_error_class = "engine_unavailable"
                        return {"ok": False, "launcher": self.mode,
                                "load_class": "engine_unavailable",
                                "loader_stderr": diag,
                                "note": f"comfy spawn exited before readiness: {diag}"}
                self._sleep(_READY_POLL_S)
            diag = self._diagnostics(self.spec)
            self._last_error = diag or "no readiness within timeout"
            self._last_error_class = "unreachable"
            return {"ok": False, "launcher": self.mode,
                    "load_class": "unreachable",
                    "loader_stderr": diag or None,
                    "note": (f"comfy did not become ready at {self.url} within "
                             f"{ceiling:.0f}s")}
        finally:
            self._starting = False

    def _launch(self) -> "tuple[bool, str]":
        """Fire the launcher (no readiness wait). Returns ``(launched, note)``."""
        if self.mode == "systemd-user":
            unit = self.spec.get("unit")
            rc, out, err = self._runner(
                ["systemctl", "--user", "start", str(unit)],
                timeout=30.0, env=_user_env())
            if rc == 0:
                return True, f"systemctl --user start {unit} accepted"
            return False, (err or out or f"systemctl --user start {unit} "
                                          f"returned {rc}").strip()
        if self.mode == "spawn":
            argv = [self.spec.get("python"), self.spec.get("main"),
                    *(self.spec.get("args") or [])]
            cwd = os.path.dirname(self.spec.get("main") or "") or None
            try:
                self._child = self._popen(
                    argv, cwd=cwd, env=dict(os.environ),
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            except Exception as exc:  # noqa: BLE001 — a bad python/main path
                return False, f"{type(exc).__name__}: {exc}"
            return True, f"spawned {' '.join(str(a) for a in argv)}"
        return False, f"unmanaged launcher {self.mode!r}"

    def _child_exit_note(self) -> Optional[str]:
        """For spawn: ``None`` while the child is alive, else a note with its
        exit code and whatever it wrote to stderr."""
        child = self._child
        if child is None:
            return "spawn produced no child"
        rc = child.poll()
        if rc is None:
            return None
        err = ""
        try:
            if child.stderr is not None:
                err = (child.stderr.read() or b"")
                if isinstance(err, bytes):
                    err = err.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            err = ""
        return f"exit code {rc}" + (f"; stderr: {err.strip()}" if err.strip() else "")

    def stop(self) -> dict:
        """Stop the managed comfy so its CUDA context is released. NEVER for the
        ``external`` launcher. The caller (idle watchdog) guarantees comfy is not
        mid-render before calling this — the stop itself does not re-check."""
        if not self.managed:
            return {"ok": False, "launcher": "external",
                    "note": "external launcher — the worker does not stop comfy"}
        if self.mode == "systemd-user":
            unit = self.spec.get("unit")
            rc, out, err = self._runner(
                ["systemctl", "--user", "stop", str(unit)],
                timeout=30.0, env=_user_env())
            ok = rc == 0
            note = (f"systemctl --user stop {unit} accepted" if ok
                    else (err or out or f"stop returned {rc}").strip())
            if ok:
                logger.info("comfy stopped via systemd-user unit %s", unit)
            return {"ok": ok, "launcher": self.mode, "note": note}
        if self.mode == "spawn":
            child = self._child
            if child is None or child.poll() is not None:
                self._child = None
                return {"ok": True, "launcher": self.mode,
                        "note": "spawn child already gone"}
            try:
                child.terminate()
                try:
                    child.wait(timeout=15.0)
                except Exception:  # noqa: BLE001 — did not die politely
                    child.send_signal(signal.SIGKILL)
                    try:
                        child.wait(timeout=5.0)
                    except Exception:  # noqa: BLE001
                        pass
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "launcher": self.mode,
                        "note": f"terminate failed: {type(exc).__name__}: {exc}"}
            self._child = None
            logger.info("comfy spawn child stopped")
            return {"ok": True, "launcher": self.mode, "note": "spawn child stopped"}
        return {"ok": False, "launcher": self.mode,
                "note": f"unmanaged launcher {self.mode!r}"}

    # -- diagnostics ---------------------------------------------------------
    def _default_diagnostics(self, spec: dict) -> str:
        """Recent journal for a systemd-user unit that failed to become ready —
        the real reason, whole, not a canned line. Empty for spawn (its child
        stderr is read directly) or when journalctl is unavailable."""
        if (spec or {}).get("kind") != "systemd-user":
            return ""
        unit = spec.get("unit")
        rc, out, err = self._runner(
            ["journalctl", "--user", "-u", str(unit), "-n", "50",
             "--no-pager"], timeout=10.0, env=_user_env())
        if rc == 0 and (out or "").strip():
            return out.strip()
        # status is a decent fallback when journal is empty/locked
        rc2, out2, _ = self._runner(
            ["systemctl", "--user", "status", str(unit), "--no-pager",
             "-n", "50"], timeout=10.0, env=_user_env())
        if rc2 in (0, 3) and (out2 or "").strip():
            return out2.strip()
        return (err or "").strip()
