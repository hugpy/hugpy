"""k118 — the worker's SELF-REPORT of its own environment.

Why self-report and not central going to look: ``ae`` — the fleet's most robust
box and the reference this doctrine is meant to describe — is not ssh-reachable
from central. Anything central cannot ask over HTTP it cannot know. So the box
that HAS the answer is the box that publishes it: ``GET /ops/environment``, and
a compact digest folded onto every heartbeat.

WHAT IT COSTS. The main venv's package list is read with ``importlib.metadata``
IN THIS PROCESS — no subprocess, no pip. Each ``envs/<profile>`` venv costs one
short child python (they are separate interpreters; there is no other way to
ask). Binaries cost one ``--version`` exec each. The whole thing is cached for
``TTL_S`` (10 min) so a heartbeat rider and a page-load-frequency endpoint are
both free, and ``refresh=True`` is the operator's escape hatch.

IMPORT DISCIPLINE — LOAD-BEARING. This module is stdlib-only at the top level
and every intra-package import is lazy + guarded. That is not politeness: the
seeding path for the very first doctrine copies THIS FILE to a worker and runs
it with that worker's venv python (``python environment_report.py``), because
the installed package there predates the module. A relative import at module
scope would make that impossible, and restarting the worker agent to get the
new code is exactly the disruption this slice must not cause.

Nothing here raises. A probe that cannot answer records ``None`` (unknown) and
NEVER an optimistic value — the same rule the oracle probes follow: an absent
fact must be distinguishable from a negative one.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

#: How long a report is trusted. The heartbeat rides at ~15s; without a TTL
#: this would spawn one child python per profile venv four times a minute.
TTL_S: float = 600.0

#: Per-binary version probe: name -> (argv suffix, regex group 1 = version).
#: A DECLARED list on purpose (never "everything on PATH"): the doctrine is a
#: statement about what the fleet needs, and an inventory of a box's /usr/bin
#: would be drift noise, not doctrine. Extensible — add a row, and both the
#: report and the diff pick it up with no other change.
BINARY_PROBES: "dict[str, tuple[tuple[str, ...], str]]" = {
    # ffmpeg: a-brain 2026-08-17 — round-trip ASR was dead because it was absent.
    "ffmpeg":     (("-version",), r"ffmpeg version (\S+)"),
    "ffprobe":    (("-version",), r"ffprobe version (\S+)"),
    "git":        (("--version",), r"git version (\S+)"),
    # nvidia-smi answers TWO facts (driver + CUDA); parsed separately below.
    "nvidia-smi": (("--version",), r"NVIDIA-SMI version\s*:\s*(\S+)"),
    # bsdtar: the extractor the downloader prefers for archived model payloads.
    "bsdtar":     (("--version",), r"bsdtar (\S+)"),
    "python3":    (("--version",), r"Python (\S+)"),
}

#: Mounts the fleet actually depends on. Reported as present/absent/writable —
#: never mutated, never created.
MOUNT_PROBES: "tuple[str, ...]" = ("/mnt/llm_storage",)

#: torch's COMPANION packages: each ships its own ``Requires-Dist: torch==X`` and
#: fails to import against a torch that is not X (the ``torchvision::nms`` operator
#: incident, computron 2026-09-24). Presence + version is not enough to know a
#: companion works — its DECLARED torch requirement must be read from its metadata
#: and checked against the torch actually installed in the SAME venv. This is the
#: fact the report collects; ``doctrine.doctor`` does the join. Declared, not
#: "everything that names torch": the doctrine is a statement of what the fleet
#: needs, and a torch requirement is only load-bearing for these two.
TORCH_BASE: str = "torch"
TORCH_COMPANIONS: "tuple[str, ...]" = ("torchvision", "torchaudio")

#: Report schema version. Bumped when a FIELD changes meaning, so a doctrine
#: snapshotted from an older shape is recognizable rather than silently mis-read.
REPORT_SCHEMA: str = "1"

_CACHE: "dict[str, object]" = {"at": 0.0, "report": None}


# ---------------------------------------------------------------------------
# Seams (kept module-level so tests can swap them without a subprocess)
# ---------------------------------------------------------------------------


def _run(argv: "list[str]", timeout: float = 20.0) -> "tuple[int, str]":
    """``(rc, stdout+stderr)`` for ``argv``. Never raises: a missing binary, a
    hung child and a permission error are all just "no answer"."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — a probe never becomes the failure
        return 127, f"{type(exc).__name__}: {exc}"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _which(name: str) -> "str | None":
    import shutil
    try:
        return shutil.which(name)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------


def normalize_dist(name: str) -> str:
    """PEP 503 normalization. ``chatterbox-tts``/``Chatterbox_TTS`` are ONE
    package; a doctrine keyed by the raw metadata name would miss half the
    fleet on spelling alone."""
    return re.sub(r"[-_.]+", "-", (name or "").strip()).lower()


def packages_here() -> "dict[str, str]":
    """``{normalized_dist: version}`` for the interpreter RUNNING THIS CODE.

    ``importlib.metadata`` rather than ``pip freeze``: no subprocess, no
    network, and it answers with what is actually importable from this
    interpreter's paths — which is the question the doctrine asks."""
    out: "dict[str, str]" = {}
    try:
        from importlib.metadata import distributions
    except Exception:  # noqa: BLE001 — ancient python: no answer, not a guess
        return out
    try:
        dists = list(distributions())
    except Exception:  # noqa: BLE001
        return out
    for dist in dists:
        try:
            raw = dist.metadata["Name"]          # type: ignore[index]
            version = dist.version
        except Exception:  # noqa: BLE001 — one broken .dist-info skips itself
            continue
        if not raw:
            continue
        name = normalize_dist(str(raw))
        # Keep the FIRST answer: earlier sys.path entries shadow later ones, and
        # the shadowing copy is the one that would be imported.
        out.setdefault(name, str(version))
    return out


def torch_requirement_of(requires: "list[str] | None",
                         base: str = TORCH_BASE) -> "str | None":
    """The version specifier a dist declares for ``base`` (``torch``), from its
    ``Requires-Dist`` lines — or None when it declares none.

    ``torchvision``'s metadata carries ``Requires-Dist: torch==2.13.0``; that is
    the rule the installed torch must satisfy, and it lives with the companion,
    not in the doctrine. Extras/optional deps are IGNORED (a ``torch`` behind
    ``extra == 'x'`` is not a hard requirement); markers that merely narrow the
    platform are kept. Parentheses (``torch (==2.13.0)``) are stripped. None is
    never a guess: a companion that declares no torch pin is not "compatible with
    everything", it is "we cannot decide", which downstream reports as UNKNOWN."""
    if not requires:
        return None
    for raw in requires:
        text = str(raw)
        head, _, marker = text.partition(";")
        if "extra" in marker and "==" in marker:
            continue          # an optional/extra dependency, not a hard one
        head = head.strip()
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*(.*)$", head)
        if not match:
            continue
        if normalize_dist(match.group(1)) != normalize_dist(base):
            continue
        spec = match.group(2).strip()
        if spec.startswith("(") and spec.endswith(")"):
            spec = spec[1:-1].strip()
        return spec or None
    return None


def torch_companions_here() -> "dict[str, object]":
    """``{companion: torch_specifier_or_None}`` for each INSTALLED torch companion
    in THIS interpreter. Absent companions are simply not keyed — their absence
    is the ``packages`` map's job to report, not this one's."""
    out: "dict[str, object]" = {}
    try:
        from importlib.metadata import distribution
    except Exception:  # noqa: BLE001 — ancient python: no answer, not a guess
        return out
    for comp in TORCH_COMPANIONS:
        try:
            dist = distribution(comp)
        except Exception:  # noqa: BLE001 — not installed / unreadable: skip
            continue
        try:
            reqs = dist.requires
        except Exception:  # noqa: BLE001
            reqs = None
        out[comp] = torch_requirement_of(list(reqs) if reqs else None)
    return out


#: The one-liner a child interpreter runs to describe itself. Same content as
#: ``packages_here`` + ``torch_companions_here`` + the python version, printed as
#: JSON on stdout. The companion parse is INLINED (the child cannot import this
#: module) and must stay in lockstep with ``torch_requirement_of``.
_CHILD_PROGRAM = (
    "import json,re,sys\n"
    "from importlib.metadata import distributions,distribution\n"
    "def _norm(s):\n"
    "    return re.sub(r'[-_.]+','-',str(s).strip()).lower()\n"
    "def _torchreq(reqs):\n"
    "    for raw in (reqs or []):\n"
    "        head=str(raw).split(';',1)\n"
    "        mk=head[1] if len(head)>1 else ''\n"
    "        if 'extra' in mk and '==' in mk: continue\n"
    "        m=re.match(r'^([A-Za-z0-9_.-]+)\\s*(.*)$', head[0].strip())\n"
    "        if not m or _norm(m.group(1))!='torch': continue\n"
    "        s=m.group(2).strip()\n"
    "        if s.startswith('(') and s.endswith(')'): s=s[1:-1].strip()\n"
    "        return s or None\n"
    "    return None\n"
    "o={}\n"
    "for d in distributions():\n"
    "    try:\n"
    "        n=d.metadata['Name']; v=d.version\n"
    "    except Exception: continue\n"
    "    if not n: continue\n"
    "    o.setdefault(_norm(n), str(v))\n"
    "c={}\n"
    "for comp in ('torchvision','torchaudio'):\n"
    "    try: dd=distribution(comp)\n"
    "    except Exception: continue\n"
    "    try: rq=dd.requires\n"
    "    except Exception: rq=None\n"
    "    c[comp]=_torchreq(rq)\n"
    "print(json.dumps({'python': '.'.join(map(str, sys.version_info[:3])), "
    "'packages': o, 'torch_companions': c}))\n"
)


def packages_in(python: str) -> "dict[str, object] | None":
    """Ask ANOTHER interpreter what it holds. None when it cannot answer.

    None is load-bearing: an env-profile venv that exists but whose python is
    broken must read as unknown, not as empty — "no packages" would make the
    doctrine diff report every pinned dep as missing and hand the operator a
    repair plan for a venv that needs rebuilding, not pip-installing."""
    if not python or not os.path.isfile(python):
        return None
    rc, out = _run([python, "-c", _CHILD_PROGRAM], timeout=60.0)
    if rc != 0:
        return None
    try:
        parsed = json.loads(out.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(parsed, dict) or "packages" not in parsed:
        return None
    return parsed


# ---------------------------------------------------------------------------
# The env-profile venvs (``<worker_root>/envs/<name>``)
# ---------------------------------------------------------------------------


def worker_root() -> str:
    """``<worker_root>`` — the same resolution ``managers.serve.profiles`` does,
    re-implemented here rather than imported because this file must run
    standalone on a box whose installed package has no such module yet. The
    packaged path is preferred when it IS importable, so the two can't drift."""
    try:
        from hugpy_engine.serve import profiles as _profiles
        return _profiles.worker_root()
    except Exception:  # noqa: BLE001 — standalone / older package: derive it
        pass
    override = os.environ.get("HUGPY_WORKER_ROOT")
    if override:
        return os.path.expanduser(override)
    engine = os.environ.get("HUGPY_ENGINE_DIR")
    if engine:
        parent = os.path.dirname(os.path.normpath(os.path.expanduser(engine)))
        if os.path.basename(parent) == "hugpy-worker":
            return parent
    return os.path.join(os.path.expanduser("~"), "hugpy-worker")


def profiles_root() -> str:
    return os.path.join(worker_root(), "envs")


def profile_python(name: str) -> str:
    sub = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    return os.path.join(profiles_root(), name, sub, exe)


def discovered_profiles() -> "list[str]":
    """Every ``envs/<name>`` that holds a python. DISK, not the declared config:
    the chatterbox seat is materialized by a background job, so the config and
    the disk disagree for as long as pip runs — and the disk is what a diff
    must be about."""
    root = profiles_root()
    try:
        names = sorted(e.name for e in os.scandir(root) if e.is_dir())
    except Exception:  # noqa: BLE001 — no envs dir is a normal, quiet answer
        return []
    return [n for n in names if os.path.isfile(profile_python(n))]


def venvs_report() -> "dict[str, object]":
    """``{venv_name: {python, python_version, packages|None, error}}``.

    ``main`` is always present and is THIS interpreter. Profile venvs are keyed
    by their profile name so a doctrine entry can name the venv it is about
    (``setuptools<81`` is a fact about ``chatterbox-tts``, not about main)."""
    out: "dict[str, object]" = {
        "main": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "packages": packages_here(),
            "torch_companions": torch_companions_here(),
            "error": None,
        }
    }
    for name in discovered_profiles():
        python = profile_python(name)
        parsed = packages_in(python)
        if parsed is None:
            out[name] = {"python": python, "python_version": None,
                         "packages": None, "torch_companions": None,
                         "error": "interpreter did not answer"}
        else:
            out[name] = {"python": python,
                         "python_version": parsed.get("python"),
                         "packages": parsed.get("packages") or {},
                         "torch_companions": parsed.get("torch_companions") or {},
                         "error": None}
    return out


# ---------------------------------------------------------------------------
# Binaries, GPU, mounts, OS
# ---------------------------------------------------------------------------


def binaries_report() -> "dict[str, object]":
    """``{name: {present, path, version}}`` for every DECLARED binary.

    ``version`` is None when the binary is present but would not say — present
    beats a version string, and neither is ever fabricated."""
    out: "dict[str, object]" = {}
    for name, (args, pattern) in BINARY_PROBES.items():
        path = _which(name)
        if not path:
            out[name] = {"present": False, "path": None, "version": None}
            continue
        rc, text = _run([path, *args], timeout=20.0)
        version = None
        if rc == 0:
            match = re.search(pattern, text)
            if match:
                version = match.group(1)
        out[name] = {"present": True, "path": path, "version": version}
    return out


def nvidia_report() -> "dict[str, object]":
    """Driver + CUDA runtime as reported by ``nvidia-smi``, plus the cards.

    Its own block rather than a row in ``binaries``: "which driver" and "which
    CUDA" are two different doctrine facts that happen to share one binary."""
    out: "dict[str, object]" = {"driver": None, "cuda": None, "gpus": []}
    smi = _which("nvidia-smi")
    if not smi:
        return out
    rc, text = _run([smi,
                     "--query-gpu=name,memory.total,driver_version",
                     "--format=csv,noheader,nounits"], timeout=20.0)
    if rc == 0:
        for line in text.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                vram_mib = int(float(parts[1]))
            except ValueError:
                vram_mib = None
            out["gpus"].append({"name": parts[0], "vram_mib": vram_mib})
            if len(parts) > 2 and not out["driver"]:
                out["driver"] = parts[2]
    rc, text = _run([smi], timeout=20.0)
    if rc == 0:
        match = re.search(r"CUDA Version:\s*(\S+)", text)
        if match:
            out["cuda"] = match.group(1)
        if not out["driver"]:
            match = re.search(r"Driver Version:\s*(\S+)", text)
            if match:
                out["driver"] = match.group(1)
    return out


def mounts_report() -> "dict[str, object]":
    """``{path: {present, writable}}`` for the declared mounts. ``writable`` is
    an ``os.access`` question, never a test write — a doctrine probe does not
    put files on a shared store."""
    out: "dict[str, object]" = {}
    for path in MOUNT_PROBES:
        present = os.path.isdir(path)
        out[path] = {"present": present,
                     "writable": bool(present and os.access(path, os.W_OK))}
    return out


def os_report() -> "dict[str, object]":
    """``/etc/os-release`` PRETTY_NAME/ID/VERSION_ID + kernel. Absent file ->
    nulls; this runs on macOS and Windows workers too."""
    info: "dict[str, object]" = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "pretty_name": None, "id": None, "version_id": None,
    }
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as handle:
            for line in handle:
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip().lower()
                if key in ("pretty_name", "id", "version_id"):
                    info[key] = value.strip().strip('"').strip("'")
    except Exception:  # noqa: BLE001 — not-linux / unreadable: nulls, not a guess
        pass
    return info


def build_info_here() -> "dict[str, object] | None":
    """The build document (``hugpy_platform.buildinfo.build_info``): version,
    sha/dirty, editable source, lockstep. Lazy + guarded like every other
    intra-package reach here: the standalone seeding copy of this file runs on
    a box whose hugpy_platform may predate the module, and None is the honest
    answer there — never a fabricated identity."""
    try:
        from hugpy_platform.buildinfo import build_info
        doc = build_info()
        return doc if isinstance(doc, dict) else None
    except Exception:  # noqa: BLE001
        return None


def pkg_version() -> "str | None":
    """The worker package version, from THIS interpreter's metadata."""
    try:
        from importlib.metadata import version as _version
        return str(_version("hugpy-fleet"))
    except Exception:  # noqa: BLE001
        pass
    try:
        from hugpy_fleet import __version__
        return __version__
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def canonical_json(obj: object) -> str:
    """One canonical encoding for every digest here — same convention as
    ``oracle.contracts.canonical_json`` (sorted keys, no padding)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


#: Fields the digest is computed OVER. Deliberately excludes ``generated_at``
#: and ``worker``: a digest that changed every ten minutes would make "did this
#: box's environment change?" unanswerable, which is the one question the
#: heartbeat rider exists to answer.
DIGEST_FIELDS: "tuple[str, ...]" = ("schema", "venvs", "binaries", "nvidia",
                                    "mounts", "os", "pkg_version", "build")

#: Keys inside ``build`` that change on every computation and therefore must
#: not move the digest (same reasoning as the top-level ``generated_at``).
_BUILD_VOLATILE: "tuple[str, ...]" = ("generated_at",)


def _digest_view(report: "dict[str, object]") -> "dict[str, object]":
    view: "dict[str, object]" = {k: report.get(k) for k in DIGEST_FIELDS}
    build = view.get("build")
    if isinstance(build, dict):
        view["build"] = {k: v for k, v in build.items() if k not in _BUILD_VOLATILE}
    return view


def report_digest(report: "dict[str, object]") -> str:
    return hashlib.sha256(
        canonical_json(_digest_view(report)).encode("utf-8")).hexdigest()[:16]


def build_report(worker_name: "str | None" = None) -> "dict[str, object]":
    """The full environment report. UNCACHED — ``environment_report`` is the
    cached entry point. Never raises."""
    report: "dict[str, object]" = {
        "schema": REPORT_SCHEMA,
        "worker": worker_name or os.environ.get("WORKER_NAME") or platform.node(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "worker_root": worker_root(),
        "pkg_version": pkg_version(),
        "build": build_info_here(),
        "venvs": venvs_report(),
        "binaries": binaries_report(),
        "nvidia": nvidia_report(),
        "mounts": mounts_report(),
        "os": os_report(),
    }
    report["report_digest"] = report_digest(report)
    return report


def environment_report(*, refresh: bool = False,
                       worker_name: "str | None" = None) -> "dict[str, object]":
    """The TTL-cached report served by ``GET /ops/environment``."""
    now = time.monotonic()
    cached = _CACHE.get("report")
    if cached is not None and not refresh:
        if now - float(_CACHE.get("at") or 0.0) < TTL_S:
            return dict(cached)  # type: ignore[arg-type]
    try:
        report = build_report(worker_name)
    except Exception as exc:  # noqa: BLE001 — a report never breaks its caller
        report = {"schema": REPORT_SCHEMA, "error":
                  f"{type(exc).__name__}: {exc}", "report_digest": ""}
    _CACHE["report"], _CACHE["at"] = report, now
    return dict(report)


def clear_cache() -> None:
    _CACHE["report"], _CACHE["at"] = None, 0.0


def compact_digest(report: "dict[str, object] | None" = None) -> "dict[str, object]":
    """The heartbeat rider: enough to notice CHANGE and to spot the two gaps
    that cost this fleet a week, small enough to ride a 15-second beat.

    Never the whole report — central pulls that from ``/ops/environment`` on
    read, the same discipline the rolling aggregate follows."""
    rep = report if report is not None else environment_report()
    venvs = rep.get("venvs") or {}
    main = (venvs.get("main") or {}) if isinstance(venvs, dict) else {}
    packages = (main.get("packages") or {}) if isinstance(main, dict) else {}
    binaries = rep.get("binaries") or {}
    # One computation: the identity is lifted from the report's ``build``
    # document rather than recomputed (a repository probe per beat is waste).
    build = rep.get("build")
    identity = build.get("identity") if isinstance(build, dict) else None
    return {
        "digest": rep.get("report_digest") or "",
        "at": rep.get("generated_at"),
        "python": rep.get("python"),
        "pkg_version": rep.get("pkg_version"),
        "build": identity if isinstance(identity, dict) else None,
        "profiles": sorted(k for k in venvs if k != "main") if isinstance(venvs, dict) else [],
        "package_count": len(packages) if isinstance(packages, dict) else 0,
        "binaries": {name: bool((binaries.get(name) or {}).get("present"))
                     for name in BINARY_PROBES
                     if isinstance(binaries, dict)},
        "nvidia": {"driver": (rep.get("nvidia") or {}).get("driver"),
                   "cuda": (rep.get("nvidia") or {}).get("cuda")},
    }


def main(argv: "list[str] | None" = None) -> int:
    """``python environment_report.py [--compact]`` — the STANDALONE seeding
    path: copy this one file to a worker, run it with that worker's venv
    python, get the report on stdout. No package import, no agent restart."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--compact" in args:
        print(json.dumps(compact_digest(), indent=2, sort_keys=True))
    else:
        print(json.dumps(build_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
