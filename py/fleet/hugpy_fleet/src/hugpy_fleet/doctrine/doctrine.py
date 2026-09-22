"""The versioned doctrine: one reference worker's environment, classified.

A doctrine is TWO things joined, and the join is the whole design:

  * OBSERVED — every package/binary/mount the reference box actually has, with
    the version it has. This half is a photograph: it is taken, not argued
    about, and it is what makes "what changed?" answerable.
  * DECLARED — the fleet's own requirements, from the failures that taught them
    and from what the runners actually import (``managers/task_deps``). This
    half carries the CLASSIFICATION: ``required_for`` (which dispatch tasks the
    dep gates), ``severity`` (blocker/warn/info) and ``pin`` (the version rule,
    e.g. ``setuptools<81`` inside the chatterbox profile).

A declared requirement the reference does NOT satisfy still belongs in the
doctrine, and is recorded with ``version=None``. That is not an inconsistency:
the reference box is a box, not scripture. ae itself violates ``numpy<2.5`` and
therefore cannot run ASR — the doctrine states that, which is exactly the kind
of thing this slice exists to surface.

Anything observed but never classified defaults to ``info``: it is DRIFT
information, noted and never blocking. A doctrine that turned every version
difference into a failure would be ignored within a week.

No pathlib. os.path only, matching the oracle modules.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: The three severities, worst first. ``blocker`` makes a task ineligible;
#: ``warn`` is a finding an operator should act on but that does not gate work;
#: ``info`` is drift, recorded so a regression has a paper trail.
SEVERITIES: tuple[str, ...] = ("blocker", "warn", "info")

#: What an UNCLASSIFIED package is. Deliberately the harmless one.
DEFAULT_SEVERITY: str = "info"

#: Env override for where doctrine JSONs live (a worker that ships its own
#: copy, a test with a tmpdir).
ENV_DOCTRINE_DIR: str = "HUGPY_DOCTRINE_DIR"

#: Filename shape. The version is in the NAME, so an older doctrine is never
#: overwritten and an assessment can always name what it judged against.
FILENAME: str = "hugpy-worker-doctrine-{version}.json"

_FILENAME_RE = re.compile(r"^hugpy-worker-doctrine-(.+)\.json$")


# ---------------------------------------------------------------------------
# Version pins
# ---------------------------------------------------------------------------

_OPS = ("===", "!=", "<=", ">=", "==", "<", ">", "~=")


def version_key(version: str) -> tuple:
    """A comparable key for a version string.

    Numeric segments compare as ints, everything else as a string, and a purely
    numeric segment always sorts BELOW a suffixed one at the same position
    (``2.5`` < ``2.5rc1`` is wrong for PEP 440 and right for nothing, so we go
    the other way: ``2.5rc1`` < ``2.5``). Good enough for the question a
    doctrine actually asks — "is this below 81?" — and it never raises."""
    parts: list[tuple[int, Any]] = []
    for chunk in re.split(r"[._-]+", (version or "").strip()):
        if not chunk:
            continue
        match = re.match(r"^(\d+)(.*)$", chunk)
        if match:
            parts.append((1, int(match.group(1))))
            if match.group(2):
                parts.append((0, match.group(2).lower()))
        else:
            parts.append((0, chunk.lower()))
    return tuple(parts)


def _compare(left: str, right: str) -> int:
    a, b = version_key(left), version_key(right)
    if a == b:
        return 0
    # Pad so (1, 2) vs (1, 2, 0) compares sensibly without raising on mixed types.
    for x, y in zip(a, b):
        if x == y:
            continue
        if x[0] != y[0]:
            return -1 if x[0] < y[0] else 1
        return -1 if x[1] < y[1] else 1  # type: ignore[operator]
    return -1 if len(a) < len(b) else 1


def pin_satisfied(version: str | None, pin: str | None) -> bool | None:
    """Does ``version`` satisfy ``pin``? ``None`` when it cannot be decided.

    ``pin`` is a comma-joined specifier list (``">=2.0,<2.5"``). An UNPARSEABLE
    pin returns None rather than False — a typo in the doctrine must not
    manufacture a blocker on a box that is fine."""
    if not pin:
        return True
    if version is None:
        return None
    for clause in str(pin).split(","):
        clause = clause.strip()
        if not clause:
            continue
        op = next((o for o in _OPS if clause.startswith(o)), None)
        if op is None:
            return None
        want = clause[len(op):].strip()
        if not want:
            return None
        cmp = _compare(str(version), want)
        if op in ("==", "==="):
            ok = cmp == 0
        elif op == "!=":
            ok = cmp != 0
        elif op == "<":
            ok = cmp < 0
        elif op == "<=":
            ok = cmp <= 0
        elif op == ">":
            ok = cmp > 0
        elif op == ">=":
            ok = cmp >= 0
        elif op == "~=":
            ok = cmp >= 0
        else:  # pragma: no cover — _OPS is exhaustive
            return None
        if not ok:
            return False
    return True


# ---------------------------------------------------------------------------
# Declared requirements — the classification seed
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Requirement:
    """One classified requirement. ``venv`` is ``main``, ``profile:<name>`` or
    ``any``; ``kind`` is ``pip``/``binary``/``mount``/``driver``."""
    name: str
    kind: str = "pip"
    venv: str = "main"
    pin: str | None = None
    required_for: tuple[str, ...] = ()
    severity: str = DEFAULT_SEVERITY
    note: str = ""
    repair: str = ""

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"Requirement({self.name!r}).severity must be one of "
                f"{SEVERITIES}, got {self.severity!r}")
        if self.kind not in ("pip", "binary", "mount", "driver"):
            raise ValueError(
                f"Requirement({self.name!r}).kind is unknown: {self.kind!r}")


#: A pseudo-task: not a ``/ml`` dispatch task, but the row shape whose failure
#: taught us the lesson (computron, 4-bit jobs). Named so the diff can say WHAT
#: a missing bitsandbytes costs.
TASK_4BIT: str = "4bit-quantized-load"

#: THE SEED. Every entry traces to either a real failure this week or a module a
#: runner actually imports (``managers/task_deps.TASK_DEPS``). Nothing here is
#: aspirational: an entry that gates nothing is ``info``, and a blocker names
#: the tasks it takes down.
KNOWN_REQUIREMENTS: tuple[Requirement, ...] = (
    # ── the four failures ────────────────────────────────────────────────
    Requirement(
        "ffmpeg", kind="binary", venv="any", severity="blocker",
        required_for=("automatic-speech-recognition", "text-to-speech",
                      "video-extract", "audio-master"),
        repair="sudo apt install -y ffmpeg",
        note="a-brain 2026-08-17: absent -> round-trip ASR dead. whisper and "
             "every audio/video runner shell out to it; there is no fallback."),
    Requirement(
        "bitsandbytes", severity="blocker", required_for=(TASK_4BIT,),
        note="computron 2026-08-16: absent -> every 4-bit (load_in_4bit) job "
             "failed at load. Central advertises 4-bit rows per worker."),
    Requirement(
        "diffusers", severity="blocker", required_for=("text-to-image",),
        note="a-brain 2026-08-19: 'diffusers + torch not installed' after the "
             "TTS env-profile seating. task_deps maps text-to-image -> diffusers."),
    Requirement(
        "setuptools", venv="profile:chatterbox-tts", pin="<81",
        severity="blocker", required_for=("text-to-speech",),
        note="chatterbox-tts imports pkg_resources, removed in setuptools 81. "
             "It fails SILENTLY (empty audio), so this pin is a blocker."),
    # ── the ASR landmine the fleet already documents (agent.py 2026-07-11) ──
    Requirement(
        "openai-whisper", severity="blocker",
        required_for=("automatic-speech-recognition",),
        note="the ASR backend. NOTE the split names: distribution "
             "'openai-whisper', import 'whisper'."),
    Requirement(
        "numba", severity="blocker",
        required_for=("automatic-speech-recognition",),
        note="whisper imports numba. Present-but-unimportable whisper is the "
             "find_spec-insufficient case worker_agent guards with a real import."),
    Requirement(
        "numpy", pin="<2.5", severity="blocker",
        required_for=("automatic-speech-recognition",),
        note="numba refuses NumPy >= 2.5 ('Numba needs NumPy 2.4 or less'), "
             "which kills `import whisper` on an otherwise complete box."),
    # ── what the runners import (managers/task_deps.TASK_DEPS) ───────────
    Requirement(
        "torch", severity="blocker",
        required_for=("text-to-image", "depth-estimation", "object-detection",
                      "image-classification", "image-segmentation",
                      "feature-extraction", "sentence-similarity",
                      "text-summarization", TASK_4BIT),
        note="every in-process ML task on the box. Its version is also the "
             "reason env-profile venvs exist (colliding pins)."),
    Requirement(
        "transformers", severity="blocker",
        required_for=("text-summarization", "depth-estimation",
                      "object-detection", "image-classification",
                      "image-segmentation"),
        note="task_deps: four vision tasks + summarization."),
    Requirement(
        "llama-cpp-python", severity="blocker",
        required_for=("image-text-to-text",),
        note="task_deps: image-text-to-text. Base abstract_hugpy_dev omits it "
             "on purpose; the [engine] extra installs it."),
    Requirement(
        "chatterbox-tts", venv="profile:chatterbox-tts", severity="blocker",
        required_for=("text-to-speech",),
        note="the TTS backend, seated in its own env-profile venv because its "
             "torch pins collide with the agent's (managers/tts/seat.py)."),
    Requirement(
        "accelerate", severity="warn",
        required_for=("text-to-image", TASK_4BIT),
        note="device_map / offload for diffusers and 4-bit loads. Absent, "
             "large loads fall back to slower paths or OOM."),
    Requirement(
        "sentence-transformers", severity="warn",
        required_for=("feature-extraction", "sentence-similarity"),
        note="task_deps: embeddings + similarity."),
    Requirement("keybert", severity="warn", required_for=("keyword-extraction",),
                note="task_deps: keyword-extraction."),
    Requirement("pdfplumber", severity="warn",
                required_for=("document-extraction",),
                note="task_deps: document-extraction."),
    Requirement("beautifulsoup4", severity="warn",
                required_for=("url-extraction",),
                note="task_deps: url-extraction (import name 'bs4')."),
    Requirement("safetensors", severity="warn", required_for=(),
                note="the weight format nearly every modern row ships in."),
    Requirement("huggingface-hub", severity="warn", required_for=(),
                note="every download/resolve path."),
    Requirement(
        "torchaudio", venv="profile:chatterbox-tts", severity="warn",
        required_for=("text-to-speech",),
        note="chatterbox's audio I/O; must match its torch in the SAME venv."),
    # ── box facts ────────────────────────────────────────────────────────
    Requirement(
        "nvidia-smi", kind="binary", venv="any", severity="warn",
        required_for=(),
        repair="install the NVIDIA driver (this box serves CPU-only without it)",
        note="absent = no GPU accounting: VRAM budgets, the pid registry and "
             "every eviction decision degrade to guesses."),
    Requirement("git", kind="binary", venv="any", severity="warn",
                repair="sudo apt install -y git",
                note="repo-cloning provision paths (custom nodes, sources)."),
    Requirement("ffprobe", kind="binary", venv="any", severity="warn",
                required_for=("video-extract", "audio-master"),
                repair="sudo apt install -y ffmpeg",
                note="ships with ffmpeg; media probing before extraction."),
    Requirement("bsdtar", kind="binary", venv="any", severity="info",
                repair="sudo apt install -y libarchive-tools",
                note="preferred archive extractor when present; python "
                     "tarfile/zipfile is the fallback, so this is info only."),
    Requirement(
        "/mnt/llm_storage", kind="mount", venv="any", severity="warn",
        required_for=(),
        repair="mount the shared llm_storage virtiofs/NFS export",
        note="the shared model store. Absent, the box downloads its own copy "
             "of every model and the central<->worker artifact shortcut is off."),
)


def classification_for(kind: str, name: str,
                       venv: str = "main") -> Requirement | None:
    """The declared requirement matching ``(kind, name, venv)``, or None.

    Resolution is MOST SPECIFIC FIRST: an entry scoped to this exact venv wins
    over one scoped to ``any``. That is what lets ``setuptools`` be a blocker
    pinned ``<81`` inside the chatterbox profile and plain drift-info in main."""
    for req in KNOWN_REQUIREMENTS:
        if req.kind == kind and req.name == name and req.venv == venv:
            return req
    for req in KNOWN_REQUIREMENTS:
        if req.kind == kind and req.name == name and req.venv == "any":
            return req
    return None


# ---------------------------------------------------------------------------
# The doctrine document
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DoctrineEntry:
    """One classified fact. ``version`` is what the REFERENCE had (None for a
    declared-only requirement); ``pin`` is what the fleet REQUIRES."""
    name: str
    kind: str = "pip"
    venv: str = "main"
    version: str | None = None
    pin: str | None = None
    required_for: tuple[str, ...] = ()
    severity: str = DEFAULT_SEVERITY
    source: str = "reference"          # "reference" | "declared"
    note: str = ""
    repair: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.kind, self.venv, self.name)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["required_for"] = list(self.required_for)
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DoctrineEntry":
        return cls(
            name=str(d["name"]), kind=str(d.get("kind") or "pip"),
            venv=str(d.get("venv") or "main"), version=d.get("version"),
            pin=d.get("pin"),
            required_for=tuple(d.get("required_for") or ()),
            severity=str(d.get("severity") or DEFAULT_SEVERITY),
            source=str(d.get("source") or "reference"),
            note=str(d.get("note") or ""), repair=str(d.get("repair") or ""))


@dataclass(frozen=True, slots=True)
class Doctrine:
    """A frozen, versioned statement of what a worker should hold.

    ``provisional``/``pending`` are first-class: a doctrine seeded from a box
    that is not the intended reference must SAY SO in the document, not in a
    commit message nobody reads at 3am."""
    version: str
    reference: str = ""
    reference_digest: str = ""
    created_at: str = ""
    provisional: bool = False
    pending: str = ""
    notes: str = ""
    entries: tuple[DoctrineEntry, ...] = ()
    reference_facts: Mapping[str, Any] = field(default_factory=dict)

    def entry(self, kind: str, name: str,
              venv: str = "main") -> DoctrineEntry | None:
        for item in self.entries:
            if item.key == (kind, venv, name):
                return item
        return None

    def by_severity(self, severity: str) -> tuple[DoctrineEntry, ...]:
        return tuple(e for e in self.entries if e.severity == severity)

    def tasks(self) -> tuple[str, ...]:
        seen: set[str] = set()
        for item in self.entries:
            seen.update(item.required_for)
        return tuple(sorted(seen))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "1",
            "version": self.version,
            "reference": self.reference,
            "reference_digest": self.reference_digest,
            "created_at": self.created_at,
            "provisional": self.provisional,
            "pending": self.pending,
            "notes": self.notes,
            "reference_facts": dict(self.reference_facts),
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Doctrine":
        return cls(
            version=str(d["version"]),
            reference=str(d.get("reference") or ""),
            reference_digest=str(d.get("reference_digest") or ""),
            created_at=str(d.get("created_at") or ""),
            provisional=bool(d.get("provisional")),
            pending=str(d.get("pending") or ""),
            notes=str(d.get("notes") or ""),
            entries=tuple(DoctrineEntry.from_dict(e)
                          for e in (d.get("entries") or ())),
            reference_facts=dict(d.get("reference_facts") or {}))


# ---------------------------------------------------------------------------
# Snapshotting a report into a doctrine
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def snapshot(report: Mapping[str, Any], *, version: str,
             reference: str = "", provisional: bool = False,
             pending: str = "", notes: str = "") -> Doctrine:
    """Freeze ``report`` into a classified, versioned doctrine.

    OBSERVED first (every package in every venv, every present binary/mount,
    the driver), then DECLARED requirements the observation did not already
    cover, stamped ``source='declared'`` with ``version=None``."""
    entries: dict[tuple[str, str, str], DoctrineEntry] = {}

    def _add(kind: str, name: str, venv: str, observed: str | None,
             source: str) -> None:
        req = classification_for(kind, name, venv)
        entry = DoctrineEntry(
            name=name, kind=kind, venv=venv, version=observed,
            pin=(req.pin if req else None),
            required_for=(req.required_for if req else ()),
            severity=(req.severity if req else DEFAULT_SEVERITY),
            source=source, note=(req.note if req else ""),
            repair=(req.repair if req else ""))
        entries[entry.key] = entry

    venvs = report.get("venvs") or {}
    if isinstance(venvs, Mapping):
        for raw_name, block in venvs.items():
            if not isinstance(block, Mapping):
                continue
            venv = "main" if raw_name == "main" else f"profile:{raw_name}"
            packages = block.get("packages")
            if not isinstance(packages, Mapping):
                continue          # unknown != empty; nothing to snapshot
            for pkg, ver in sorted(packages.items()):
                _add("pip", str(pkg), venv, str(ver), "reference")

    binaries = report.get("binaries") or {}
    if isinstance(binaries, Mapping):
        for name, block in sorted(binaries.items()):
            if isinstance(block, Mapping) and block.get("present"):
                _add("binary", str(name), "any", block.get("version"),
                     "reference")

    mounts = report.get("mounts") or {}
    if isinstance(mounts, Mapping):
        for path, block in sorted(mounts.items()):
            if isinstance(block, Mapping) and block.get("present"):
                _add("mount", str(path), "any",
                     "writable" if block.get("writable") else "read-only",
                     "reference")

    nvidia = report.get("nvidia") or {}
    if isinstance(nvidia, Mapping):
        if nvidia.get("driver"):
            _add("driver", "nvidia-driver", "any", str(nvidia["driver"]),
                 "reference")
        if nvidia.get("cuda"):
            _add("driver", "cuda", "any", str(nvidia["cuda"]), "reference")

    # DECLARED requirements the reference did not supply. Recorded with
    # version=None so a diff can tell "the reference had 0.39.0" from "the
    # fleet requires this and the reference lacks it too".
    for req in KNOWN_REQUIREMENTS:
        key = (req.kind, req.venv, req.name)
        if key in entries:
            entries[key] = replace(entries[key], pin=req.pin,
                                   required_for=req.required_for,
                                   severity=req.severity, note=req.note,
                                   repair=req.repair)
            continue
        # An ``any``-scoped requirement already satisfied under a concrete venv
        # is not missing; only add when nothing matched at all.
        if req.venv == "any" and any(
                k[0] == req.kind and k[2] == req.name for k in entries):
            continue
        entries[key] = DoctrineEntry(
            name=req.name, kind=req.kind, venv=req.venv, version=None,
            pin=req.pin, required_for=req.required_for,
            severity=req.severity, source="declared", note=req.note,
            repair=req.repair)

    facts = {
        "python": report.get("python"),
        "pkg_version": report.get("pkg_version"),
        "worker_root": report.get("worker_root"),
        "os": report.get("os"),
        "nvidia": report.get("nvidia"),
        "mounts": report.get("mounts"),
        "generated_at": report.get("generated_at"),
        "venvs": sorted(k for k in venvs) if isinstance(venvs, Mapping) else [],
    }
    ordered = tuple(sorted(entries.values(),
                           key=lambda e: (e.kind, e.venv, e.name)))
    return Doctrine(
        version=str(version),
        reference=reference or str(report.get("worker") or ""),
        reference_digest=str(report.get("report_digest") or ""),
        created_at=_now(), provisional=bool(provisional), pending=pending,
        notes=notes, entries=ordered, reference_facts=facts)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def _candidate_dirs() -> list[str]:
    """Where a doctrine may live, in precedence order.

    The repo copy is derived from THIS FILE's location, never hardcoded, so a
    checkout anywhere resolves. A pip-installed worker has no repo above it and
    simply falls through to the worker-local / system paths — and, finding
    none, reports ``no_doctrine`` rather than inventing one."""
    out: list[str] = []
    override = os.environ.get(ENV_DOCTRINE_DIR)
    if override:
        out.append(os.path.expanduser(override))
    here = os.path.dirname(os.path.abspath(__file__))
    # .../<repo>/abstract_hugpy_dev/src/abstract_hugpy_dev/fleet_doctrine
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(here))))
    out.append(os.path.join(repo, "deploy", "doctrine"))
    out.append(os.path.join(os.path.expanduser("~"), "hugpy-worker",
                            "doctrine"))
    out.append(os.path.join("/etc", "hugpy", "doctrine"))
    seen: set[str] = set()
    return [d for d in out if not (d in seen or seen.add(d))]


def doctrine_dir() -> str:
    """The FIRST candidate that exists, else the first candidate (where a
    ``save`` would create it)."""
    candidates = _candidate_dirs()
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def doctrine_path(version: str, directory: str | None = None) -> str:
    return os.path.join(directory or doctrine_dir(),
                        FILENAME.format(version=version))


def list_versions(directory: str | None = None) -> list[str]:
    """Every doctrine version present, oldest-comparable first."""
    root = directory or doctrine_dir()
    try:
        names = sorted(e.name for e in os.scandir(root) if e.is_file())
    except Exception:  # noqa: BLE001 — no directory is "no versions"
        return []
    found = [m.group(1) for m in (_FILENAME_RE.match(n) for n in names) if m]
    return sorted(found, key=version_key)


def save(doctrine: Doctrine, directory: str | None = None) -> str:
    root = directory or doctrine_dir()
    os.makedirs(root, exist_ok=True)
    path = doctrine_path(doctrine.version, root)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doctrine.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load(version: str, directory: str | None = None) -> Doctrine | None:
    """The named doctrine, or None. Never raises: an unreadable or malformed
    doctrine is "no doctrine", which downstream reports as UNKNOWN."""
    path = doctrine_path(version, directory)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return Doctrine.from_dict(json.load(handle))
    except Exception as exc:  # noqa: BLE001
        logger.debug("fleet_doctrine: %s unreadable (%s)", path, exc)
        return None


def latest(directory: str | None = None) -> Doctrine | None:
    versions = list_versions(directory)
    if not versions:
        return None
    return load(versions[-1], directory)


__all__ = [
    "DEFAULT_SEVERITY",
    "ENV_DOCTRINE_DIR",
    "FILENAME",
    "KNOWN_REQUIREMENTS",
    "SEVERITIES",
    "TASK_4BIT",
    "Doctrine",
    "DoctrineEntry",
    "Requirement",
    "classification_for",
    "doctrine_dir",
    "doctrine_path",
    "latest",
    "list_versions",
    "load",
    "pin_satisfied",
    "save",
    "snapshot",
    "version_key",
]
