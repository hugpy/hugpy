"""Turnkey worker setup — the one-step convergence + self-check the installer runs.

Every item in here is something the operator had to do BY HAND to bring a box
into the fleet (a-brain, 2026-09-24: central/advertise URLs, a slot-port
collision, a false BLOCKED exit, a CPU-only llama engine on 4x3090, a
torch/torchvision skew, two legacy worker units, unit drift, a firewall rule,
and no ComfyUI). Each is a function here that (a) INSPECTS the box, recording
real facts, (b) optionally CONVERGES it, and (c) returns a :class:`Check` whose
``detail`` is those recorded facts — never a canned string, never a guess.

Design rules (operator doctrine):
  * No advisory/canned text a person reads: every ``detail`` is measured (which
    port, which URL, the real command output / error text, the counts).
  * Absences are explicit: a check that could not run says so and WHY.
  * Idempotent: re-running converges; already-correct is reported as such.
  * ``dry_run`` never touches the box — it records the exact command it WOULD run.

The installer (:mod:`hugpy_fleet.worker.install`) calls :func:`run_setup` after
it has resolved options; a failing REQUIRED check makes the install exit
non-zero with the exact reason (:func:`overall_rc`).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from urllib.parse import urlparse

# --------------------------------------------------------------------------- #
# result type                                                                 #
# --------------------------------------------------------------------------- #
OK = "ok"
FAIL = "fail"
WARN = "warn"
SKIP = "skip"


@dataclass
class Check:
    """One recorded self-check / convergence outcome.

    ``status`` is one of OK/FAIL/WARN/SKIP. ``required`` FAILs fail the install
    (non-zero exit); WARN/SKIP never do. ``detail`` carries recorded facts only.
    ``actions`` lists what convergence actually changed (or, under dry-run, would
    change) so the report shows the by-hand steps the installer now performs.
    """
    name: str
    status: str
    detail: str
    fix: Optional[str] = None
    required: bool = True
    actions: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# small process helper                                                         #
# --------------------------------------------------------------------------- #
def _run(argv: List[str], *, dry_run: bool, timeout: float = 60.0,
         env: Optional[dict] = None) -> Tuple[int, str]:
    """Run ``argv``, returning ``(rc, combined_output)``. Under ``dry_run`` it
    runs NOTHING and returns ``(0, "DRY-RUN: <argv>")`` so the report shows the
    exact command. Output is captured whole (never truncated here — the caller
    decides how much of the recorded fact to show)."""
    printable = " ".join(argv)
    if dry_run:
        return 0, f"DRY-RUN: would run: {printable}"
    try:
        cp = subprocess.run(argv, capture_output=True, text=True,
                            timeout=timeout, env=env)
    except FileNotFoundError as exc:
        return 127, f"{argv[0]}: not found ({exc})"
    except subprocess.TimeoutExpired as exc:
        return 124, f"timed out after {timeout}s: {printable}\n{exc.output or ''}"
    out = (cp.stdout or "") + (cp.stderr or "")
    return cp.returncode, out.strip()


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


# --------------------------------------------------------------------------- #
# item 1 / 10 — central + advertise URLs and reachability                     #
# --------------------------------------------------------------------------- #
def _local_ip_toward(central_url: str) -> Optional[str]:
    """The worker's own outbound IP on the route to central (the address central
    would see for a direct connection). Mirrors the agent's own derivation so the
    installer advertises exactly what the agent would."""
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


def derive_advertise_url(central_url: str, port: int,
                         explicit: Optional[str]) -> Tuple[Optional[str], str]:
    """(url, source). An explicit --advertise/WORKER_URL wins ("explicit"); else
    derive ``http://<outbound-ip>:<port>`` ("derived"); else ``(None, "none")``
    when no non-loopback route exists (a topology the operator MUST name)."""
    if explicit:
        return explicit, "explicit"
    ip = _local_ip_toward(central_url)
    if ip:
        return f"http://{ip}:{port}", "derived"
    return None, "none"


def check_advertise(central_url: str, port: int,
                    explicit: Optional[str]) -> Check:
    url, source = derive_advertise_url(central_url, port, explicit)
    if url is None:
        return Check(
            "advertise-url", FAIL,
            "could not derive a routable advertise URL (no non-loopback route to "
            f"{central_url}); this box's topology is not auto-derivable",
            fix="pass --advertise http://<address-central-reaches-this-box-on>:"
                f"{port} (e.g. the tunnel/hub address); it is persisted as WORKER_URL",
            required=False)
    if source == "explicit":
        detail = f"advertise URL set explicitly: {url} (persisted as WORKER_URL)"
    else:
        detail = (f"advertise URL DERIVED from the outbound route to central: {url} "
                  "(persisted as WORKER_URL). If central reaches this box on a "
                  "DIFFERENT address (a tunnel/hub side), pass --advertise instead.")
    return Check("advertise-url", OK, detail, required=False)


def check_central_reachable(central_url: str, *, timeout: float = 8.0) -> Check:
    """worker -> central: GET <central>/api/health must answer 2xx."""
    url = central_url.rstrip("/") + "/api/health"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.getcode()
            body = resp.read(2000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = (exc.read() or b"").decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        return Check("central-reachable", FAIL,
                     f"GET {url} -> HTTP {exc.code} {exc.reason}. body: "
                     f"{' '.join(body.split())[:300] or '(empty)'}",
                     fix="verify --central is the central ORIGIN (no /api suffix) "
                         "and that this box can route to it",
                     required=True)
    except Exception as exc:  # noqa: BLE001 — connect/DNS/timeout
        return Check("central-reachable", FAIL,
                     f"GET {url} failed: {type(exc).__name__}: {exc}",
                     fix="check DNS/routing/firewall from this box to central; "
                         "for a WireGuard topology bring the tunnel up first",
                     required=True)
    ok = 200 <= code < 300
    return Check("central-reachable", OK if ok else FAIL,
                 f"GET {url} -> HTTP {code}; body: {' '.join(body.split())[:200]}",
                 required=True)


def check_central_can_reach_worker(central_url: str, worker_id: Optional[str],
                                   token: Optional[str], *,
                                   timeout: float = 10.0) -> Check:
    """central -> worker: ask central's own probe endpoint
    (GET <central>/api/llm/workers/<id>/health) to dial this worker's advertised
    URL. Only checkable once the worker has registered (we have its id). When the
    probe needs operator auth we don't hold, that is reported clearly (SKIP), not
    guessed."""
    if not worker_id:
        return Check("central->worker", SKIP,
                     "not checked: this worker has no registered id yet "
                     "(register did not complete, or --service none/foreground). "
                     "Verify from the console once it registers.",
                     required=False)
    url = central_url.rstrip("/") + f"/api/llm/workers/{worker_id}/health"
    req = urllib.request.Request(url, method="GET")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return Check("central->worker", SKIP,
                         f"not checked: central's probe endpoint {url} needs "
                         f"operator auth (HTTP {exc.code}); the worker enroll "
                         "token cannot call it. Verify reachability from the "
                         "console's worker health action.",
                         required=False)
        return Check("central->worker", WARN,
                     f"probe {url} -> HTTP {exc.code} {exc.reason}",
                     required=False)
    except Exception as exc:  # noqa: BLE001
        return Check("central->worker", WARN,
                     f"probe {url} failed: {type(exc).__name__}: {exc}",
                     required=False)
    reachable = bool(data.get("reachable"))
    if reachable:
        return Check("central->worker", OK,
                     f"central dialed this worker's advertised URL "
                     f"{data.get('url')!r} successfully (central's probe)",
                     required=False)
    return Check("central->worker", FAIL,
                 f"central CANNOT reach this worker at {data.get('url')!r}: "
                 f"{data.get('error') or 'unreachable'}",
                 fix="the advertised URL is not routable FROM central — pass "
                     "--advertise with the address central reaches this box on "
                     "and open the worker port to central (firewall)",
                 required=False)


# --------------------------------------------------------------------------- #
# item 2 — slot port clearance                                                #
# --------------------------------------------------------------------------- #
def slot_port_plan(worker_port: int, slot_count: int, port_base: int) -> dict:
    """Pure port plan: control ports (port_base+i) the agent finds slots at, and
    each slot's llama-server child port (control+1000, the slot_agent default).
    Records any collision with the worker port. Testable without a box."""
    controls = [port_base + i for i in range(max(slot_count, 0))]
    children = [c + 1000 for c in controls]
    collisions = []
    for i, (c, ch) in enumerate(zip(controls, children), start=1):
        if c == worker_port:
            collisions.append(f"slot {i} control port {c} == worker port {worker_port}")
        if ch == worker_port:
            collisions.append(f"slot {i} child port {ch} == worker port {worker_port}")
    return {"worker_port": worker_port, "controls": controls,
            "children": children, "collisions": collisions}


def check_slot_ports(worker_port: int, slot_count: int, port_base: int) -> Check:
    plan = slot_port_plan(worker_port, slot_count, port_base)
    if not plan["collisions"]:
        return Check("slot-ports", OK,
                     f"worker port {worker_port}; slot control ports "
                     f"{plan['controls'] or '[]'}; child ports {plan['children'] or '[]'} "
                     "— no collision with the worker port",
                     required=False)
    # The slot_agent now relocates a colliding CHILD port at startup, but a
    # CONTROL-port collision would move the slot out from under the agent's
    # slot_urls() — that needs a base change, so flag it as a required fix.
    control_clash = any("control port" in c for c in plan["collisions"])
    return Check("slot-ports", FAIL if control_clash else WARN,
                 "port collision detected: " + "; ".join(plan["collisions"]),
                 fix=f"set SLOT_PORT_BASE so no control/child port equals the "
                     f"worker port {worker_port} (child = control + 1000). "
                     "The slot_agent relocates a colliding CHILD loudly at "
                     "startup; a CONTROL collision must be fixed via SLOT_PORT_BASE.",
                 required=control_clash)


# --------------------------------------------------------------------------- #
# item 4 — CUDA engine (native llama-server + llama-cpp-python offload)        #
# --------------------------------------------------------------------------- #
def nvidia_gpus() -> list:
    """The box's NVIDIA GPUs (reuses the platform detector). Empty == none."""
    try:
        from hugpy_platform.hardware import detect_gpus
        return list(detect_gpus() or [])
    except Exception:  # noqa: BLE001
        return []


def has_nvcc() -> bool:
    return _which("nvcc") is not None


def _llama_offload() -> dict:
    """Probe llama-cpp-python's GPU-offload support in a SUBPROCESS (never import
    llama_cpp into the installer — it poisons a later torch import). Returns
    ``{"installed": bool, "supports_gpu_offload": bool|None, "version": str|None,
    "error": str|None}``."""
    code = (
        "import json\n"
        "out={'installed':False,'supports_gpu_offload':None,'version':None,'error':None}\n"
        "try:\n"
        "    import llama_cpp\n"
        "    out['installed']=True\n"
        "    out['version']=getattr(llama_cpp,'__version__',None)\n"
        "    try: out['supports_gpu_offload']=bool(llama_cpp.llama_supports_gpu_offload())\n"
        "    except Exception as e: out['error']='offload probe: %s'%e\n"
        "except Exception as e:\n"
        "    out['error']=str(e)\n"
        "print(json.dumps(out))\n"
    )
    rc, o = _run([sys.executable, "-c", code], dry_run=False, timeout=60)
    try:
        return json.loads(o.splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"installed": False, "supports_gpu_offload": None,
                "version": None, "error": o or f"probe rc={rc}"}


def _native_engine() -> dict:
    try:
        from hugpy_engine.native.resolve import native_engine_status
        return dict(native_engine_status(probe=True) or {})
    except Exception as exc:  # noqa: BLE001
        return {"found": False, "path": None, "error": f"resolve failed: {exc}"}


def provision_cuda_engine(*, dry_run: bool, jobs: Optional[int] = None) -> Check:
    """Detect NVIDIA GPUs + CUDA and make BOTH GPU engine halves present:

      * native ``llama-server`` (vision GGUFs / slots) — built from source with
        ``-DGGML_CUDA=on -DGGML_RPC=ON`` via the existing
        :func:`hugpy_engine.native.build.build_from_source` (idempotent — skipped
        when already resolvable);
      * ``llama-cpp-python`` GPU offload — rebuilt from source with
        ``CMAKE_ARGS=<HUGPY_LLAMA_CMAKE_ARGS or -DGGML_CUDA=on>`` (there is no
        helper for the binding; this is WORKER-SETUP §2's documented command).

    On a box with NO NVIDIA GPU this is a no-op OK (CPU-only is a valid worker).
    If a GPU is present but CUDA can't be provisioned, it FAILS with the exact
    reason (missing nvcc, build stderr) — never a silent CPU fallback.
    """
    gpus = nvidia_gpus()
    if not gpus:
        return Check("cuda-engine", OK,
                     "no NVIDIA GPU detected — CPU-only worker; no CUDA engine "
                     "needed", required=False)
    names = ", ".join(str(g.get("name") or "?") for g in gpus)
    nvcc = has_nvcc()
    native = _native_engine()
    offload = _llama_offload()
    actions: List[str] = []
    problems: List[str] = []

    # -- native llama-server --------------------------------------------------
    if native.get("found"):
        detail_native = (f"native llama-server present at {native.get('path')} "
                         f"(source={native.get('source')}, spawn_ok={native.get('spawn_ok')})")
    elif not nvcc:
        problems.append("native llama-server missing and nvcc not found — cannot "
                        "build a CUDA engine from source")
        detail_native = "native llama-server: absent; nvcc: absent"
    else:
        detail_native = "native llama-server: absent — building from source (CUDA)"
        if dry_run:
            actions.append("would build native engine: build_from_source(cuda=True)")
        else:
            try:
                from hugpy_engine.native import build as _eng_build
                info = _eng_build.build_from_source(cuda=True,
                                                    jobs=jobs or os.cpu_count())
                actions.append(f"built native engine -> {info.get('server_bin')}")
                native = _native_engine()
            except Exception as exc:  # noqa: BLE001
                problems.append(f"native engine build failed: {exc}")

    # -- llama-cpp-python GPU offload ----------------------------------------
    if offload.get("installed") and offload.get("supports_gpu_offload"):
        detail_binding = (f"llama-cpp-python {offload.get('version')} already has "
                          "GPU offload support")
    elif not nvcc:
        problems.append("llama-cpp-python lacks GPU offload and nvcc not found — "
                        "cannot rebuild with CUDA")
        detail_binding = (f"llama-cpp-python installed={offload.get('installed')} "
                          f"supports_gpu_offload={offload.get('supports_gpu_offload')}; "
                          "nvcc: absent")
    else:
        cmake_args = os.environ.get("HUGPY_LLAMA_CMAKE_ARGS", "-DGGML_CUDA=on")
        ver = offload.get("version")
        spec = f"llama-cpp-python=={ver}" if ver else "llama-cpp-python"
        pip_argv = [sys.executable, "-m", "pip", "install", "--force-reinstall",
                    "--no-cache-dir", "--no-binary", ":all:", spec]
        detail_binding = (f"llama-cpp-python installed={offload.get('installed')} "
                          f"supports_gpu_offload={offload.get('supports_gpu_offload')} "
                          f"— rebuilding with CMAKE_ARGS={cmake_args!r}")
        env = dict(os.environ)
        env["CMAKE_ARGS"] = cmake_args
        rc, out = _run(pip_argv, dry_run=dry_run, timeout=3600, env=env)
        if dry_run:
            actions.append(f"would rebuild binding: CMAKE_ARGS={cmake_args!r} " +
                           " ".join(pip_argv))
        elif rc != 0:
            problems.append(f"llama-cpp-python CUDA rebuild failed (rc={rc}); "
                            f"last output: {out[-600:]}")
        else:
            offload = _llama_offload()
            if not offload.get("supports_gpu_offload"):
                problems.append("llama-cpp-python rebuilt but still reports no GPU "
                                "offload support")
            else:
                actions.append(f"rebuilt llama-cpp-python {offload.get('version')} "
                               "with GPU offload")

    detail = (f"GPUs: {len(gpus)} ({names}); nvcc: {'present' if nvcc else 'ABSENT'}. "
              f"{detail_native}. {detail_binding}.")
    if problems:
        return Check("cuda-engine", FAIL, detail + " PROBLEMS: " + "; ".join(problems),
                     fix="build a CUDA llama.cpp (see WORKER-SETUP §2/§3): "
                         "`hugpy install-engine --cuda --build-from-source` and "
                         "`CMAKE_ARGS=-DGGML_CUDA=on pip install --force-reinstall "
                         "--no-binary :all: llama-cpp-python`; install the CUDA "
                         "toolkit (nvcc) first",
                     required=True, actions=actions)
    return Check("cuda-engine", OK, detail, required=True, actions=actions)


# --------------------------------------------------------------------------- #
# item 5 — torch + torchvision matched set, importable                        #
# --------------------------------------------------------------------------- #
def _import_probe(module: str) -> dict:
    """Import ``module`` in a SUBPROCESS (isolation) and report its version or the
    real ImportError text. This is the definitive companion check — it catches
    the ABI break (torchvision built for another torch) that a metadata-only diff
    cannot see."""
    code = (
        "import json\n"
        f"m={module!r}\n"
        "out={'ok':False,'version':None,'error':None}\n"
        "try:\n"
        "    mod=__import__(m)\n"
        "    out['ok']=True\n"
        "    out['version']=getattr(mod,'__version__',None)\n"
        "except Exception as e:\n"
        "    out['error']='%s: %s'%(type(e).__name__,e)\n"
        "print(json.dumps(out))\n"
    )
    rc, o = _run([sys.executable, "-c", code], dry_run=False, timeout=120)
    try:
        return json.loads(o.splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"ok": False, "version": None, "error": o or f"probe rc={rc}"}


def check_torch_companions() -> Check:
    """torch + torchvision (and torchaudio if installed) must import as a matched
    set. Real imports are the primary signal; the doctrine companion assessment
    (``companion_mismatch``) is reported too when available."""
    torch = _import_probe("torch")
    if not torch.get("ok"):
        # torch absent is not a worker-install failure per se (a CPU/text box may
        # not need it) but a broken torch that is INSTALLED-yet-unimportable is.
        if torch.get("error") and "No module named" in str(torch.get("error")):
            return Check("torch-companions", SKIP,
                         "torch not installed — skipping companion check "
                         f"({torch.get('error')})", required=False)
        return Check("torch-companions", FAIL,
                     f"`import torch` FAILED: {torch.get('error')}",
                     fix="reinstall a torch matching this box's CUDA "
                         "(e.g. --index-url https://download.pytorch.org/whl/cuXXX)",
                     required=True)
    tv = _import_probe("torchvision")
    parts = [f"torch {torch.get('version')}"]
    if tv.get("ok"):
        parts.append(f"torchvision {tv.get('version')} imports")
    else:
        # A doctrine verdict adds the precise pin mismatch (declared vs installed).
        verdict = _doctrine_companion_verdict()
        detail = (f"torch {torch.get('version')} imports but `import torchvision` "
                  f"FAILED: {tv.get('error')}")
        if verdict:
            detail += f" | doctrine: {verdict}"
        return Check("torch-companions", FAIL, detail,
                     fix="install torchvision + torchaudio as a MATCHED set from "
                         "the SAME CUDA index as torch, e.g. `pip install "
                         "--index-url https://download.pytorch.org/whl/cuXXX "
                         "torch==<base> torchvision torchaudio`",
                     required=True)
    ta = _import_probe("torchaudio")
    if ta.get("ok"):
        parts.append(f"torchaudio {ta.get('version')} imports")
    elif ta.get("error") and "No module named" not in str(ta.get("error")):
        parts.append(f"torchaudio present but FAILS: {ta.get('error')}")
    return Check("torch-companions", OK,
                 "; ".join(parts) + " — matched, importable", required=True)


def _doctrine_companion_verdict() -> Optional[str]:
    """The doctrine companion finding (declared-vs-installed torch pin), if the
    doctrine + doctor are available. Best-effort context; never raises."""
    try:
        from hugpy_fleet.doctrine import doctrine as _doctrine
        from hugpy_fleet.doctrine.doctor import assess, STATUS_COMPANION_MISMATCH
        from hugpy_fleet.worker.environment_report import build_report
        cur = _doctrine.latest()
        if cur is None:
            return None
        report = assess(build_report(), cur)
        for f in list(getattr(report, "blockers", []) or []) + \
                list(getattr(report, "warnings", []) or []):
            if getattr(f, "status", None) == STATUS_COMPANION_MISMATCH:
                return getattr(f, "detail", None) or "companion_mismatch"
    except Exception:  # noqa: BLE001
        return None
    return None


# --------------------------------------------------------------------------- #
# item 6 — legacy worker units/processes on the same box                      #
# --------------------------------------------------------------------------- #
# Units that were the by-hand / pre-product way of running a worker on this box.
# The canonical unit is hugpy-worker.service (user unit); abstract-hugpy-worker
# is the recognized legacy alias the setup doc used to document.
_LEGACY_UNIT_PATTERNS = (
    "alpha-hugpy-worker", "abstract-hugpy-worker", "abstract_hugpy",
    "hugpy-worker@",  # any templated legacy instance
)
_CANONICAL_UNIT = "hugpy-worker.service"


def _list_units() -> List[str]:
    """All loaded/enabled unit names across system + user managers (best-effort)."""
    units: set = set()
    for scope in (["systemctl"], ["systemctl", "--user"]):
        rc, out = _run(scope + ["list-unit-files", "--type=service", "--no-legend",
                                "--no-pager"], dry_run=False, timeout=20)
        if rc != 0:
            continue
        for line in out.splitlines():
            tok = line.split()
            if tok:
                units.add((tuple(scope), tok[0]))
    return list(units)


def detect_legacy_workers() -> List[Tuple[tuple, str]]:
    """(scope-argv, unit-name) pairs for legacy worker units on this box, excluding
    the canonical hugpy-worker.service."""
    found = []
    for scope, unit in _list_units():
        if unit == _CANONICAL_UNIT:
            continue
        low = unit.lower()
        if any(pat in low for pat in _LEGACY_UNIT_PATTERNS):
            found.append((scope, unit))
    return found


def retire_legacy_workers(*, dry_run: bool) -> Check:
    legacy = detect_legacy_workers()
    if not legacy:
        return Check("legacy-workers", OK,
                     "no legacy hugpy/abstract_hugpy worker units found on this box",
                     required=False)
    actions = []
    problems = []
    for scope, unit in legacy:
        argv = list(scope) + ["disable", "--now", unit]
        rc, out = _run(argv, dry_run=dry_run, timeout=30)
        label = ("--user " if "--user" in scope else "system ") + unit
        if dry_run:
            actions.append(f"would retire {label} (disable --now)")
        elif rc == 0:
            actions.append(f"retired {label} (disabled + stopped)")
        else:
            problems.append(f"could not retire {label}: {out[-200:]}")
    status = WARN if problems else OK
    detail = (f"found {len(legacy)} legacy worker unit(s): "
              + ", ".join(u for _, u in legacy))
    if problems:
        detail += " | problems: " + "; ".join(problems)
    return Check("legacy-workers", status, detail, required=False, actions=actions)


# --------------------------------------------------------------------------- #
# item 7 — unit drift (report what re-running the installer corrected)         #
# --------------------------------------------------------------------------- #
_ENV_RE = re.compile(r'^Environment="([^=]+)=(.*)"$')


def _unit_env(text: str) -> dict:
    env = {}
    for line in (text or "").splitlines():
        m = _ENV_RE.match(line.strip())
        if m:
            env[m.group(1)] = m.group(2)
    return env


def _unit_exec(text: str) -> Optional[str]:
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("ExecStart="):
            return s[len("ExecStart="):]
    return None


def unit_drift(old_text: str, new_text: str) -> List[str]:
    """Human-readable list of what changed between an existing unit and the
    canonical one the installer writes (env keys added/removed/changed + a
    changed ExecStart). Empty == already canonical. Pure + testable."""
    drift = []
    old_env, new_env = _unit_env(old_text), _unit_env(new_text)
    for k in sorted(set(old_env) | set(new_env)):
        ov, nv = old_env.get(k), new_env.get(k)
        if ov == nv:
            continue
        if ov is None:
            drift.append(f"+ {k}={nv}")
        elif nv is None:
            drift.append(f"- {k} (was {ov})")
        else:
            drift.append(f"~ {k}: {ov} -> {nv}")
    oe, ne = _unit_exec(old_text), _unit_exec(new_text)
    if oe is not None and oe != ne:
        drift.append(f"~ ExecStart changed")
    return drift


# --------------------------------------------------------------------------- #
# item 8 — firewall: open the worker port to central                          #
# --------------------------------------------------------------------------- #
def _ufw_active() -> Optional[bool]:
    """True/False if ufw is present and its status is known; None if ufw absent."""
    if not _which("ufw"):
        return None
    rc, out = _run(["ufw", "status"], dry_run=False, timeout=15)
    if rc != 0:
        return None
    return "Status: active" in out


def _central_host(central_url: str) -> Optional[str]:
    try:
        h = urlparse(central_url).hostname
        # An IP is directly usable in a ufw rule; a hostname is resolved so the
        # rule pins central's ADDRESS, not a name ufw can't match on.
        if h and re.match(r"^\d+\.\d+\.\d+\.\d+$", h):
            return h
        if h:
            return socket.gethostbyname(h)
    except Exception:  # noqa: BLE001
        return None
    return None


def open_firewall_to_central(central_url: str, worker_port: int, *,
                             dry_run: bool) -> Check:
    active = _ufw_active()
    if active is None:
        return Check("firewall", SKIP,
                     "ufw not present — no host firewall rule managed by the "
                     "installer (open the worker port to central by your box's "
                     "own means if a firewall is in play)", required=False)
    if not active:
        return Check("firewall", OK,
                     f"ufw present but inactive — worker port {worker_port} not "
                     "blocked by a host firewall", required=False)
    host = _central_host(central_url)
    if not host:
        return Check("firewall", WARN,
                     f"ufw active but central host could not be resolved from "
                     f"{central_url}; open port {worker_port} to central by hand",
                     fix=f"ufw allow from <central-ip> to any port {worker_port} proto tcp",
                     required=False)
    argv = ["ufw", "allow", "from", host, "to", "any", "port", str(worker_port),
            "proto", "tcp"]
    rc, out = _run(argv, dry_run=dry_run, timeout=20)
    central_rule = (f"On CENTRAL, if its host firewall is active, allow this "
                    f"worker to reach central's API port: "
                    f"`ufw allow from <this-worker-ip> to any port "
                    f"{urlparse(central_url).port or 7002} proto tcp`")
    if dry_run:
        return Check("firewall", OK,
                     f"ufw active — would open worker port {worker_port} to central "
                     f"{host}: {' '.join(argv)}", fix=central_rule,
                     required=False, actions=[f"would run: {' '.join(argv)}"])
    if rc != 0:
        return Check("firewall", WARN,
                     f"ufw active but rule add failed (rc={rc}): {out[-200:]}",
                     fix=central_rule, required=False)
    return Check("firewall", OK,
                 f"ufw active — opened worker port {worker_port}/tcp to central "
                 f"{host} ({out.splitlines()[-1] if out else 'rule added'})",
                 fix=central_rule, required=False,
                 actions=[f"ran: {' '.join(argv)}"])


# --------------------------------------------------------------------------- #
# item 9 — stale central record (central-side; report only)                   #
# --------------------------------------------------------------------------- #
def note_stale_record() -> Check:
    """Item 9 is CENTRAL-side. Registration dedupes by worker_id then url, NOT by
    name+host (WorkerStore.register, hugpy_fleet/central/workers.py:~3819-3830),
    so a re-registered same-name box with a new id/url leaves the old OFFLINE row.
    Retiring it belongs in that register path — recorded here, not changed from
    the worker installer."""
    return Check("stale-central-record", SKIP,
                 "superseding a stale same-name OFFLINE central record is a "
                 "central-side change (WorkerStore.register dedupes by worker_id "
                 "then url, not name+host). Not done from the worker installer.",
                 required=False)


# --------------------------------------------------------------------------- #
# item 11 — ComfyUI install (worker owns comfy's lifecycle)                    #
# --------------------------------------------------------------------------- #
COMFY_REPO = os.environ.get("HUGPY_COMFY_REPO",
                            "https://github.com/comfyanonymous/ComfyUI")
COMFY_MANAGER_REPO = "https://github.com/ltdrdata/ComfyUI-Manager"
COMFY_IPADAPTER_REPO = "https://github.com/cubiq/ComfyUI_IPAdapter_plus"
COMFY_DEFAULT_PORT = 8188


def detect_comfy_root(storage_root: Optional[str]) -> Optional[str]:
    """An existing ComfyUI checkout to CONVERGE rather than duplicate. Checks the
    env override, the storage root, and the two conventional homes."""
    candidates = []
    env_root = os.environ.get("HUGPY_COMFY_DIR")
    if env_root:
        candidates.append(env_root)
    if storage_root:
        candidates.append(os.path.join(storage_root, "ComfyUI"))
        candidates.append(os.path.join(os.path.dirname(storage_root.rstrip("/")),
                                       "ComfyUI"))
    home = os.path.expanduser("~")
    candidates.append(os.path.join(home, "hugpy-worker", "ComfyUI"))
    candidates.append(os.path.join(home, "ComfyUI"))
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "main.py")):
            return os.path.abspath(c)
    return None


def _comfy_port(worker_port: int, slot_port_base: int, slot_count: int) -> int:
    """A comfy port that can't collide with the worker or slot ports."""
    explicit = os.environ.get("HUGPY_COMFY_PORT")
    reserved = {worker_port}
    reserved.update(slot_port_base + i for i in range(max(slot_count, 0)))
    reserved.update(slot_port_base + i + 1000 for i in range(max(slot_count, 0)))
    port = int(explicit) if explicit and explicit.isdigit() else COMFY_DEFAULT_PORT
    while port in reserved:
        port += 1
    return port


def _shared_checkpoints_dir(storage_root: Optional[str]) -> Optional[str]:
    """The fleet's shared model-store checkpoints dir (what comfy is wired to via
    extra_model_paths + COMFY_CHECKPOINT_DIRS) — never a per-box duplicate."""
    if storage_root:
        return os.path.join(storage_root, "models", "checkpoints")
    try:
        from hugpy_platform.constants import DEFAULT_ROOT
        return os.path.join(str(DEFAULT_ROOT), "models", "checkpoints")
    except Exception:  # noqa: BLE001
        return None


def _git(argv: List[str], *, dry_run: bool, timeout: float = 600.0) -> Tuple[int, str]:
    return _run(["git"] + argv, dry_run=dry_run, timeout=timeout)


def provision_comfy(*, storage_root: Optional[str], worker_port: int,
                    slot_port_base: int, slot_count: int, version: Optional[str],
                    dry_run: bool) -> Tuple[Check, dict]:
    """Install/converge ComfyUI so the WORKER owns its lifecycle. Returns
    (Check, env) where ``env`` carries the unit vars the installer bakes
    (HUGPY_COMFY_LAUNCH, HUGPY_COMFY_URL, COMFY_URL, COMFY_CHECKPOINTS_DIR,
    COMFY_CHECKPOINT_DIRS).

    Idempotent: an existing checkout is updated + rewired in place, never
    duplicated (computron's hand /mnt/storage/hugpy-worker/ComfyUI converges).
    """
    actions: List[str] = []
    problems: List[str] = []
    env: dict = {}
    if not _which("git"):
        return (Check("comfyui", FAIL,
                      "git not found — cannot install/converge ComfyUI",
                      fix="install git, then re-run the installer", required=False),
                env)

    root = detect_comfy_root(storage_root)
    default_root = (os.path.join(storage_root, "ComfyUI") if storage_root
                    else os.path.join(os.path.expanduser("~"), "hugpy-worker", "ComfyUI"))
    target = root or default_root

    # 1. clone or update (pin the version at clone time — a shallow clone can't
    #    check out an arbitrary tag afterward) --------------------------------
    resolved_ref = None
    if root:
        actions.append(f"found existing ComfyUI at {root} — converging in place")
        rc, out = _git(["-C", root, "fetch", "--tags", "--quiet"], dry_run=dry_run)
        if rc != 0 and not dry_run:
            problems.append(f"git fetch failed: {out[-200:]}")
        if version:
            rc, out = _git(["-C", root, "checkout", version], dry_run=dry_run)
            if dry_run:
                actions.append(f"would checkout {version}")
            elif rc != 0:
                problems.append(f"git checkout {version} failed: {out[-200:]}")
            else:
                resolved_ref = version
                actions.append(f"checked out {version}")
    else:
        if not dry_run:
            os.makedirs(os.path.dirname(target), exist_ok=True)
        clone = ["clone", "--depth", "1"]
        if version:
            clone += ["--branch", version]
        clone += [COMFY_REPO, target]
        rc, out = _git(clone, dry_run=dry_run)
        if dry_run:
            actions.append(f"would clone {COMFY_REPO}"
                           + (f" @ {version}" if version else "") + f" -> {target}")
        elif rc != 0:
            problems.append(f"git clone failed: {out[-300:]}")
        else:
            resolved_ref = version
            actions.append(f"cloned ComfyUI -> {target}"
                           + (f" @ {version}" if version else ""))

    # 2. record the exact commit so the version is pinned-by-record ----------
    if not dry_run and os.path.isdir(os.path.join(target, ".git")):
        rc, out = _git(["-C", target, "rev-parse", "--short", "HEAD"],
                       dry_run=False, timeout=20)
        if rc == 0:
            resolved_ref = (version or "HEAD") + f"@{out.strip()}"

    # 3. python deps into the worker venv (torch already matched by item 5) ---
    req = os.path.join(target, "requirements.txt")
    if dry_run:
        actions.append(f"would pip install -r {req} (into {sys.executable})")
    elif os.path.isfile(req):
        rc, out = _run([sys.executable, "-m", "pip", "install", "-r", req],
                       dry_run=False, timeout=1800)
        if rc != 0:
            problems.append(f"comfy requirements install failed (rc={rc}): {out[-400:]}")
        else:
            actions.append("installed ComfyUI requirements into the worker venv")

    # 4. custom nodes the fleet's comfy workflows need -----------------------
    nodes_dir = os.path.join(target, "custom_nodes")
    for repo, name in ((COMFY_MANAGER_REPO, "ComfyUI-Manager"),
                       (COMFY_IPADAPTER_REPO, "ComfyUI_IPAdapter_plus")):
        dest = os.path.join(nodes_dir, name)
        if os.path.isdir(dest):
            actions.append(f"custom node {name} already present")
            continue
        if dry_run:
            actions.append(f"would clone custom node {repo} -> {dest}")
            continue
        os.makedirs(nodes_dir, exist_ok=True)
        rc, out = _git(["clone", "--depth", "1", repo, dest], dry_run=False)
        if rc != 0:
            problems.append(f"custom node {name} clone failed: {out[-200:]}")
        else:
            actions.append(f"installed custom node {name}")

    # 5. wire checkpoints to the SHARED store (extra_model_paths + env) -------
    shared_ckpts = _shared_checkpoints_dir(storage_root)
    if shared_ckpts:
        env["COMFY_CHECKPOINTS_DIR"] = shared_ckpts
        env["COMFY_CHECKPOINT_DIRS"] = shared_ckpts
        emp = os.path.join(target, "extra_model_paths.yaml")
        shared_models = os.path.dirname(shared_ckpts)  # <store>/models
        yaml_body = ("# written by hugpy worker installer — shared fleet model store\n"
                     "hugpy:\n"
                     f"  base_path: {shared_models}\n"
                     "  checkpoints: checkpoints\n"
                     "  vae: vae\n"
                     "  loras: loras\n"
                     "  clip_vision: clip_vision\n"
                     "  ipadapter: ipadapter\n")
        if dry_run:
            actions.append(f"would wire checkpoints via {emp} -> {shared_models}")
        else:
            try:
                if not os.path.isfile(emp):
                    with open(emp, "w", encoding="utf-8") as fh:
                        fh.write(yaml_body)
                    actions.append(f"wired shared model store via {emp}")
                else:
                    actions.append(f"extra_model_paths.yaml already present at {emp} "
                                   "(left as-is)")
            except Exception as exc:  # noqa: BLE001
                problems.append(f"could not write {emp}: {exc}")
    else:
        problems.append("no shared checkpoints dir resolved — checkpoints not wired")

    # 6. managed launcher + URL the worker agent owns ------------------------
    port = _comfy_port(worker_port, slot_port_base, slot_count)
    main_py = os.path.join(target, "main.py")
    launch = (f"spawn:{sys.executable} {main_py} --port {port} --listen 127.0.0.1")
    # PER-GPU pin (2026-09-25): ComfyUI is ONE managed instance per worker, so its
    # card is chosen at provision via HUGPY_COMFY_CUDA_DEVICE and baked into the
    # spawn command as ComfyUI's own --cuda-device N (it renumbers that card to 0
    # internally). Absent -> unpinned (device 0 / whatever CUDA sees), today's
    # behavior. Running one comfy PER card on a multi-GPU box (N instances/ports)
    # is a follow-up; this pins the single managed instance to the operator/
    # central-chosen card.
    _comfy_dev = (os.environ.get("HUGPY_COMFY_CUDA_DEVICE") or "").strip()
    if _comfy_dev:
        try:
            launch += f" --cuda-device {int(_comfy_dev)}"
            actions.append(f"comfy pinned to GPU {int(_comfy_dev)} (--cuda-device)")
        except ValueError:
            problems.append(f"HUGPY_COMFY_CUDA_DEVICE={_comfy_dev!r} is not an int "
                            "— comfy left unpinned")
    url = f"http://127.0.0.1:{port}"
    env["HUGPY_COMFY_LAUNCH"] = launch
    env["HUGPY_COMFY_URL"] = url
    env["COMFY_URL"] = url
    actions.append(f"managed launcher (spawn) on {url} — the worker owns "
                   "start/stop/idle-stop")

    detail = (f"ComfyUI at {target} (ref {resolved_ref or 'unknown'}); "
              f"launcher=spawn managed; url={url}; "
              f"shared checkpoints={shared_ckpts or 'UNRESOLVED'}")
    if problems:
        return (Check("comfyui", FAIL, detail + " | problems: " + "; ".join(problems),
                      fix="fix the problems above and re-run; ComfyUI convergence "
                          "is idempotent", required=False, actions=actions), env)
    return (Check("comfyui", OK, detail, required=False, actions=actions), env)


def comfy_self_check(comfy_env: dict, *, do_start: bool, dry_run: bool) -> Check:
    """Verify the installed comfy: managed launcher wired, and (when --comfy-start-
    check) that it STARTS, /system_stats answers, and it sees a checkpoint. Reuses
    the worker's own ComfyManager so the check exercises the real launch path."""
    launch = comfy_env.get("HUGPY_COMFY_LAUNCH")
    url = comfy_env.get("HUGPY_COMFY_URL") or comfy_env.get("COMFY_URL")
    if not launch or launch.startswith("external"):
        return Check("comfyui-runtime", SKIP,
                     "comfy launcher not managed — runtime not checked",
                     required=False)
    if dry_run or not do_start:
        return Check("comfyui-runtime", SKIP,
                     f"launcher wired ({launch.split(':',1)[0]}, url {url}); "
                     "runtime start check not run "
                     f"({'dry-run' if dry_run else 'pass --comfy-start-check to verify start'}); "
                     "the worker verifies start on first comfy job", required=False)
    try:
        from hugpy_fleet.worker.comfy_process import ComfyManager, parse_launch
        mgr = ComfyManager(parse_launch(launch), url,
                           checkpoint_dirs=[d for d in (
                               comfy_env.get("COMFY_CHECKPOINTS_DIR"),) if d])
        res = mgr.start()
        if not res.get("ok"):
            return Check("comfyui-runtime", FAIL,
                         f"comfy did not start: {res.get('note')} "
                         f"(stderr: {(res.get('loader_stderr') or '')[-300:]})",
                         required=False)
        stats_ok = mgr.is_running(timeout=5.0)
        ckpts = mgr.checkpoints_on_disk(limit=5)
        try:
            mgr.stop()
        except Exception:  # noqa: BLE001
            pass
        detail = (f"comfy started (pid {res.get('pid')}), /system_stats "
                  f"{'reachable' if stats_ok else 'NOT reachable'}, "
                  f"checkpoints visible: {len(ckpts)}")
        status = OK if stats_ok else WARN
        return Check("comfyui-runtime", status, detail, required=False)
    except Exception as exc:  # noqa: BLE001
        return Check("comfyui-runtime", WARN,
                     f"comfy runtime check could not run: {type(exc).__name__}: {exc}",
                     required=False)


# --------------------------------------------------------------------------- #
# registered-id (read the id the agent persisted after it registers)          #
# --------------------------------------------------------------------------- #
def wait_for_worker_id(hugpy_home: str, *, timeout: float = 30.0) -> Optional[str]:
    """Poll the worker identity file the agent writes after a successful register.
    Returns the id or None if it never appears within ``timeout``."""
    prev = os.environ.get("HUGPY_HOME")
    os.environ["HUGPY_HOME"] = hugpy_home
    try:
        from hugpy_platform.app_dirs import worker_id_file
        path = worker_id_file()
    except Exception:  # noqa: BLE001
        if prev is None:
            os.environ.pop("HUGPY_HOME", None)
        else:
            os.environ["HUGPY_HOME"] = prev
        return None
    deadline = time.monotonic() + timeout
    wid = None
    while time.monotonic() < deadline:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            wid = data.get("id") or data.get("worker_id")
            if wid:
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    if prev is None:
        os.environ.pop("HUGPY_HOME", None)
    else:
        os.environ["HUGPY_HOME"] = prev
    return wid


# --------------------------------------------------------------------------- #
# report                                                                       #
# --------------------------------------------------------------------------- #
_SYMBOL = {OK: "OK  ", FAIL: "FAIL", WARN: "WARN", SKIP: "----"}


def render(checks: List[Check]) -> str:
    lines = ["", "hugpy worker self-check", "=" * 60]
    for c in checks:
        lines.append(f"[{_SYMBOL.get(c.status, '?')}] {c.name}: {c.detail}")
        for a in c.actions:
            lines.append(f"        · {a}")
        if c.fix and c.status in (FAIL, WARN):
            lines.append(f"        fix: {c.fix}")
    rc = overall_rc(checks)
    n_fail = sum(1 for c in checks if c.status == FAIL and c.required)
    lines.append("=" * 60)
    if rc == 0:
        lines.append("RESULT: worker ready (no required check failed)")
    else:
        lines.append(f"RESULT: {n_fail} required check(s) FAILED — install not clean")
    return "\n".join(lines)


def overall_rc(checks: List[Check]) -> int:
    return 1 if any(c.status == FAIL and c.required for c in checks) else 0
