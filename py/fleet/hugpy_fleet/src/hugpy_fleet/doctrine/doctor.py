"""The diff: one worker's report vs the doctrine, as findings you can act on.

Every finding names four things, because a finding that names fewer is a
complaint rather than a work order:

    the DEP        ``diffusers``
    the VENV       ``main`` / ``profile:chatterbox-tts``
    the TASKS      ``text-to-image``
    the REPAIR     ``/home/a-brain/hugpy-worker/venv/bin/python -m pip install
                     "diffusers==0.39.0"``

THE THREE THINGS THIS REFUSES TO DO.

1. It never turns UNKNOWN into a blocker. An unreadable venv, an absent report,
   an unparseable pin — all of those are findings, and none of them gates work.
   ``oracle.probes`` established the rule for this fleet; a second module with a
   second opinion about "we do not know" is how a fleet starts lying to itself.
2. It never treats an ABSENT env-profile as a fault. A box with no
   ``envs/chatterbox-tts`` is a box that does not seat TTS. That is
   availability, which the catalog already reports precisely — not breakage.
3. It never executes anything. ``repair_plan()`` returns shell LINES. The
   operator runs them, or ``/ops/pip`` does under central's audit gate. A
   preflight that silently pip-installs on a production worker is how one
   box's fix becomes four boxes' outage.

Version DRIFT (a dep present at a version other than the reference's) is always
``info``, whatever the entry's severity. Only a declared PIN can turn a version
into a blocker — otherwise the first ``pip install -U`` anywhere in the fleet
would light up every worker in red.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from hugpy_fleet.doctrine.doctrine import DEFAULT_SEVERITY, Doctrine, DoctrineEntry, pin_satisfied

#: Finding statuses. ``ok`` and ``drift`` mean the dep is THERE.
STATUS_OK = "ok"
STATUS_DRIFT = "drift"
STATUS_MISSING = "missing"
STATUS_PIN = "pin_violation"
STATUS_UNKNOWN = "unknown"
STATUS_NO_PROFILE = "profile_absent"
#: A companion (``torchvision``/``torchaudio``) is present but the torch it
#: declares in its own metadata is not the torch installed in the SAME venv, so
#: it fails to import (the ``torchvision::nms`` operator incident, 2026-09-24).
STATUS_COMPANION_MISMATCH = "companion_mismatch"

#: The pytorch wheel index, keyed by a torch build's CUDA local tag. A torch of
#: ``2.12.1+cu130`` came from ``.../whl/cu130``; the companion that matches it
#: must come from the same place, so its repair points pip there.
_PYTORCH_INDEX = "https://download.pytorch.org/whl/{tag}"

#: Report-level verdicts.
VERDICT_OK = "ok"
VERDICT_WARN = "warn"
VERDICT_BLOCKED = "blocked"
VERDICT_UNKNOWN = "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class Finding:
    """One dep's verdict on one worker."""
    dep: str
    kind: str
    venv: str
    status: str
    severity: str
    tasks: tuple[str, ...] = ()
    expected: str | None = None
    observed: str | None = None
    detail: str = ""
    repair: str = ""

    @property
    def blocking(self) -> bool:
        """A blocker is a BLOCKER-severity finding that we actually PROVED.

        ``unknown`` and ``profile_absent`` are excluded on purpose: neither is
        evidence that the box is broken, and both would otherwise turn a
        transient probe failure into fleet-wide ineligibility."""
        return (self.severity == "blocker"
                and self.status in (STATUS_MISSING, STATUS_PIN,
                                    STATUS_COMPANION_MISMATCH))

    def to_dict(self) -> dict[str, Any]:
        return {"dep": self.dep, "kind": self.kind, "venv": self.venv,
                "status": self.status, "severity": self.severity,
                "tasks": list(self.tasks), "expected": self.expected,
                "observed": self.observed, "detail": self.detail,
                "repair": self.repair}

    def line(self) -> str:
        where = f" [{self.venv}]" if self.venv and self.venv != "any" else ""
        tasks = f" gates: {', '.join(self.tasks)}" if self.tasks else ""
        return f"{self.dep}{where} — {self.detail}{tasks}"


@dataclass(frozen=True, slots=True)
class DoctrineReport:
    """The assessment of one worker. ``verdict`` is DERIVED from the findings —
    it cannot be set to something the findings do not support."""
    worker: str
    doctrine_version: str
    report_digest: str = ""
    assessed_at: str = ""
    blockers: tuple[Finding, ...] = ()
    warnings: tuple[Finding, ...] = ()
    infos: tuple[Finding, ...] = ()
    ok_count: int = 0
    checked: int = 0
    doctrine_provisional: bool = False

    @property
    def verdict(self) -> str:
        if self.blockers:
            return VERDICT_BLOCKED
        if self.warnings:
            return VERDICT_WARN
        return VERDICT_OK

    def blocked_tasks(self) -> tuple[str, ...]:
        seen: set[str] = set()
        for finding in self.blockers:
            seen.update(finding.tasks)
        return tuple(sorted(seen))

    def blockers_for_task(self, task: str) -> tuple[Finding, ...]:
        return tuple(f for f in self.blockers if task in f.tasks)

    def repair_plan(self) -> list[str]:
        """The shell lines that would fix this box, blockers first, deduped and
        ORDER-STABLE. Comments carry the why. Never executed here."""
        lines: list[str] = [
            f"# hugpy worker doctrine {self.doctrine_version} — repair plan for "
            f"{self.worker or 'this worker'}",
        ]
        if self.doctrine_provisional:
            lines.append("# NOTE: this doctrine is PROVISIONAL "
                         "(see its 'pending' field)")
        seen: set[str] = set()
        for label, findings in (("blockers", self.blockers),
                                ("warnings", self.warnings)):
            actionable = [f for f in findings if f.repair]
            if not actionable:
                continue
            lines.append(f"# --- {label} ---")
            for finding in actionable:
                if finding.repair in seen:
                    continue
                seen.add(finding.repair)
                tasks = (f" (gates {', '.join(finding.tasks)})"
                         if finding.tasks else "")
                lines.append(f"# {finding.dep}: {finding.status}{tasks}")
                lines.append(finding.repair)
        if len(lines) == 1:
            lines.append("# nothing to repair")
        return lines

    def summary(self) -> str:
        return (f"{self.worker or 'worker'}: {self.verdict.upper()} — "
                f"{len(self.blockers)} blocker(s), {len(self.warnings)} "
                f"warning(s), {len(self.infos)} info, {self.ok_count}/"
                f"{self.checked} entries satisfied "
                f"(doctrine {self.doctrine_version})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker": self.worker,
            "doctrine_version": self.doctrine_version,
            "doctrine_provisional": self.doctrine_provisional,
            "report_digest": self.report_digest,
            "assessed_at": self.assessed_at,
            "verdict": self.verdict,
            "blocked_tasks": list(self.blocked_tasks()),
            "ok_count": self.ok_count,
            "checked": self.checked,
            "blockers": [f.to_dict() for f in self.blockers],
            "warnings": [f.to_dict() for f in self.warnings],
            "infos": [f.to_dict() for f in self.infos],
        }

    def heartbeat_status(self) -> dict[str, Any]:
        """The COMPACT form that rides a heartbeat and feeds the k101 probe.

        Deliberately without the finding bodies: central pulls those from
        ``/ops/environment`` on read. What a beat must carry is the verdict, the
        tasks that are down, and one repair line per blocked task so an
        ineligibility reason can be actionable without a second round trip."""
        repairs: dict[str, str] = {}
        for finding in self.blockers:
            for task in finding.tasks:
                repairs.setdefault(task, finding.repair or finding.detail)
        return {
            "doctrine_version": self.doctrine_version,
            "provisional": self.doctrine_provisional,
            "verdict": self.verdict,
            "blockers": len(self.blockers),
            "warnings": len(self.warnings),
            "blocked_tasks": list(self.blocked_tasks()),
            "repairs": repairs,
            "report_digest": self.report_digest,
            "at": self.assessed_at,
        }


# ---------------------------------------------------------------------------
# Repair-command construction
# ---------------------------------------------------------------------------


def _venv_python(report: Mapping[str, Any], venv: str) -> str | None:
    venvs = report.get("venvs")
    if not isinstance(venvs, Mapping):
        return None
    key = venv[len("profile:"):] if venv.startswith("profile:") else venv
    block = venvs.get(key)
    if isinstance(block, Mapping) and block.get("python"):
        return str(block["python"])
    return None


def _pip_repair(report: Mapping[str, Any], entry: DoctrineEntry,
                *, pin: bool) -> str:
    """``<the right python> -m pip install "<spec>"``.

    The right python matters more than the spec: a repair aimed at the agent's
    venv when the gap is in ``envs/chatterbox-tts`` installs the package where
    it cannot help and leaves the seat exactly as broken."""
    python = _venv_python(report, entry.venv) or "python"
    if pin and entry.pin:
        spec = f"{entry.name}{entry.pin}"
    elif entry.version:
        spec = f"{entry.name}=={entry.version}"
    else:
        spec = entry.name
    return f"{shlex.quote(python)} -m pip install {shlex.quote(spec)}"


def _binary_repair(entry: DoctrineEntry) -> str:
    if entry.repair:
        return entry.repair
    return f"sudo apt install -y {entry.name}"


def _base_version(version: str | None) -> str | None:
    """A torch version with its PEP 440 local tag stripped: ``2.12.1+cu130`` ->
    ``2.12.1``. The local tag is the CUDA build, not the release the companion's
    ``torch==X`` speaks about — that comparison is on the base version."""
    if not version:
        return version
    return str(version).split("+", 1)[0].strip()


def _cuda_local_tag(version: str | None) -> str | None:
    """The local tag of a torch build: ``2.12.1+cu130`` -> ``cu130``, ``+cpu`` ->
    ``cpu``, a plain ``2.13.0`` -> None. It names the pytorch wheel index the
    matching companion must be installed from."""
    if not version or "+" not in str(version):
        return None
    local = str(version).split("+", 1)[1].strip()
    return local or None


def _companion_repair(report: Mapping[str, Any], entry: DoctrineEntry,
                      base_name: str, base_version: str | None) -> str:
    """Reinstall the companion so it MATCHES the installed torch.

    Built from facts, like every other repair here: the base torch version and
    its CUDA local tag are what the box actually holds. Pinning the base torch
    at its installed version on the matching pytorch index makes pip resolve the
    companion that declares ``torch==<that base>`` — the paired build — rather
    than hand-guessing a companion version number we cannot verify."""
    python = _venv_python(report, entry.venv) or "python"
    base = _base_version(base_version)
    tag = _cuda_local_tag(base_version)
    parts = [shlex.quote(python), "-m", "pip", "install"]
    if tag:
        parts += ["--index-url", _PYTORCH_INDEX.format(tag=tag)]
    if base:
        parts.append(shlex.quote(f"{base_name}=={base}"))
    parts.append(shlex.quote(entry.name))
    return " ".join(parts)


def _companion_finding(report: Mapping[str, Any], entry: DoctrineEntry,
                       packages: Mapping[str, Any],
                       companions: Mapping[str, Any] | None,
                       base: dict) -> Finding | None:
    """The companion-compat verdict for a dep that IS installed, or None when the
    check does not apply (it is satisfied, or this is not a companion).

    UNKNOWN — never a blocker — whenever a fact is missing: no installed torch to
    compare against, or a report too old to carry the companion's declared torch
    (``torch_companions`` absent). Only a PROVEN mismatch blocks."""
    if not entry.companion_of:
        return None
    severity = entry.companion_severity or entry.severity
    base_name = entry.companion_of
    base_version = packages.get(base_name)
    declared = (companions or {}).get(entry.name) if isinstance(
        companions, Mapping) else None
    cbase = dict(base)
    cbase["severity"] = severity
    if not isinstance(companions, Mapping) or entry.name not in companions:
        return Finding(
            **cbase, status=STATUS_UNKNOWN,
            observed=(str(base_version) if base_version else None),
            detail=(f"this report does not carry {entry.name!r}'s declared torch "
                    f"requirement (an older worker agent) — compatibility with "
                    f"the installed torch is UNKNOWN, not confirmed"),
            repair="")
    if base_version is None:
        return Finding(
            **cbase, status=STATUS_UNKNOWN, observed=None,
            detail=(f"{entry.name} declares torch {declared!r} but no torch is "
                    f"installed in {entry.venv} to compare against — UNKNOWN"),
            repair="")
    if declared is None:
        return Finding(
            **cbase, status=STATUS_UNKNOWN, observed=str(base_version),
            detail=(f"{entry.name} declares no torch requirement in its metadata "
                    f"— compatibility cannot be decided, reporting UNKNOWN"),
            repair="")
    installed_base = _base_version(str(base_version))
    satisfied = pin_satisfied(installed_base, str(declared))
    if satisfied is False:
        return Finding(
            **cbase, status=STATUS_COMPANION_MISMATCH, observed=str(base_version),
            detail=(f"{entry.name} requires torch {declared!r} but {entry.venv} "
                    f"has torch {base_version} (base {installed_base}); "
                    f"`import {entry.name}` fails against a torch it was not "
                    f"built for"),
            repair=_companion_repair(report, entry, base_name, str(base_version)))
    if satisfied is None:
        return Finding(
            **cbase, status=STATUS_UNKNOWN, observed=str(base_version),
            detail=(f"{entry.name}'s torch requirement {declared!r} could not be "
                    f"evaluated against {base_version} — UNKNOWN, not a guess"),
            repair="")
    return None


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------


def _pip_finding(report: Mapping[str, Any], entry: DoctrineEntry,
                 packages: Mapping[str, Any] | None,
                 profile_present: bool,
                 companions: Mapping[str, Any] | None = None) -> Finding:
    base = dict(dep=entry.name, kind=entry.kind, venv=entry.venv,
                severity=entry.severity, tasks=entry.required_for,
                expected=entry.pin or entry.version)
    if entry.venv.startswith("profile:") and not profile_present:
        name = entry.venv[len("profile:"):]
        return Finding(
            **base, status=STATUS_NO_PROFILE, observed=None,
            detail=(f"this box has no env-profile venv {name!r}, so the seat it "
                    f"would hold is simply not taken here (not a fault)"),
            repair="")
    if packages is None:
        return Finding(
            **base, status=STATUS_UNKNOWN, observed=None,
            detail=(f"venv {entry.venv!r} did not answer — its package list is "
                    f"UNKNOWN, so this dep is neither confirmed nor missing"),
            repair="")
    observed = packages.get(entry.name)
    if observed is None:
        # A companion whose base torch IS present gets a repair aimed at the
        # matching pytorch index; otherwise the plain pip line.
        if entry.companion_of and packages.get(entry.companion_of) is not None:
            repair = _companion_repair(report, entry, entry.companion_of,
                                       str(packages.get(entry.companion_of)))
        else:
            repair = _pip_repair(report, entry, pin=bool(entry.pin))
        return Finding(
            **base, status=STATUS_MISSING, observed=None,
            detail=(f"not installed in {entry.venv}"
                    + (f" (doctrine reference has {entry.version})"
                       if entry.version else " (and the reference lacks it too)")),
            repair=repair)
    observed = str(observed)
    satisfied = pin_satisfied(observed, entry.pin)
    if satisfied is False:
        return Finding(
            **base, status=STATUS_PIN, observed=observed,
            detail=(f"{observed} violates the required pin {entry.pin!r} in "
                    f"{entry.venv}"),
            repair=_pip_repair(report, entry, pin=True))
    if satisfied is None:
        return Finding(
            **base, status=STATUS_UNKNOWN, observed=observed,
            detail=(f"pin {entry.pin!r} could not be evaluated against "
                    f"{observed} — reporting unknown rather than guessing"),
            repair="")
    # Present and pin-clean, but a companion must ALSO agree with its torch. Its
    # own declared requirement (from the report) is the rule; a mismatch there is
    # the incident this check exists for.
    companion = _companion_finding(report, entry, packages, companions, base)
    if companion is not None:
        return companion
    if entry.version and observed != entry.version:
        return Finding(
            dep=entry.name, kind=entry.kind, venv=entry.venv,
            severity=DEFAULT_SEVERITY, tasks=entry.required_for,
            expected=entry.version, observed=observed, status=STATUS_DRIFT,
            detail=(f"{observed} here vs {entry.version} on the doctrine "
                    f"reference — drift, not a fault"),
            repair="")
    return Finding(**base, status=STATUS_OK, observed=observed, detail="",
                   repair="")


def _binary_finding(entry: DoctrineEntry,
                    binaries: Mapping[str, Any] | None) -> Finding:
    base = dict(dep=entry.name, kind=entry.kind, venv="any",
                severity=entry.severity, tasks=entry.required_for,
                expected=entry.version)
    if not isinstance(binaries, Mapping) or entry.name not in binaries:
        return Finding(**base, status=STATUS_UNKNOWN, observed=None,
                       detail=(f"the report does not describe {entry.name!r} — "
                               f"an older worker agent, or a probe that did not "
                               f"run"),
                       repair="")
    block = binaries.get(entry.name) or {}
    if not (isinstance(block, Mapping) and block.get("present")):
        return Finding(**base, status=STATUS_MISSING, observed=None,
                       detail=f"{entry.name} is not on PATH",
                       repair=_binary_repair(entry))
    observed = block.get("version")
    if entry.version and observed and str(observed) != str(entry.version):
        return Finding(dep=entry.name, kind=entry.kind, venv="any",
                       severity=DEFAULT_SEVERITY, tasks=entry.required_for,
                       expected=entry.version, observed=str(observed),
                       status=STATUS_DRIFT,
                       detail=(f"{observed} here vs {entry.version} on the "
                               f"doctrine reference — drift, not a fault"),
                       repair="")
    return Finding(**base, status=STATUS_OK,
                   observed=(str(observed) if observed else None),
                   detail="", repair="")


def _mount_finding(entry: DoctrineEntry,
                   mounts: Mapping[str, Any] | None) -> Finding:
    base = dict(dep=entry.name, kind=entry.kind, venv="any",
                severity=entry.severity, tasks=entry.required_for,
                expected=entry.version)
    if not isinstance(mounts, Mapping) or entry.name not in mounts:
        return Finding(**base, status=STATUS_UNKNOWN, observed=None,
                       detail=f"the report does not describe {entry.name!r}",
                       repair="")
    block = mounts.get(entry.name) or {}
    if not (isinstance(block, Mapping) and block.get("present")):
        return Finding(**base, status=STATUS_MISSING, observed=None,
                       detail=f"{entry.name} is not mounted on this box",
                       repair=(entry.repair
                               or f"mount {entry.name} on this box"))
    observed = "writable" if block.get("writable") else "read-only"
    if entry.version and observed != entry.version:
        return Finding(dep=entry.name, kind=entry.kind, venv="any",
                       severity=DEFAULT_SEVERITY, tasks=entry.required_for,
                       expected=entry.version, observed=observed,
                       status=STATUS_DRIFT,
                       detail=(f"mounted {observed}; the doctrine reference has "
                               f"it {entry.version}"),
                       repair="")
    return Finding(**base, status=STATUS_OK, observed=observed, detail="",
                   repair="")


def _driver_finding(entry: DoctrineEntry,
                    nvidia: Mapping[str, Any] | None) -> Finding:
    field = "driver" if entry.name == "nvidia-driver" else "cuda"
    observed = (nvidia or {}).get(field) if isinstance(nvidia, Mapping) else None
    base = dict(dep=entry.name, kind=entry.kind, venv="any",
                severity=DEFAULT_SEVERITY, tasks=entry.required_for,
                expected=entry.version)
    if not observed:
        return Finding(**base, status=STATUS_UNKNOWN, observed=None,
                       detail=(f"no {field} version reported (no NVIDIA driver, "
                               f"or nvidia-smi did not answer)"),
                       repair="")
    observed = str(observed)
    if entry.version and observed != entry.version:
        return Finding(**base, status=STATUS_DRIFT, observed=observed,
                       detail=(f"{field} {observed} here vs {entry.version} on "
                               f"the doctrine reference"),
                       repair="")
    return Finding(**base, status=STATUS_OK, observed=observed, detail="",
                   repair="")


def assess(report: Mapping[str, Any], doctrine: Doctrine) -> DoctrineReport:
    """Diff one worker's environment report against ``doctrine``.

    Pure: no network, no subprocess, no disk. It takes the two documents and
    returns the findings, which is what makes it testable with two dicts and
    what lets the SAME function run on the worker (self-assessment for the
    heartbeat) and on central (the CLI and the route)."""
    venvs = report.get("venvs") if isinstance(report.get("venvs"), Mapping) else {}
    present_profiles = {str(k) for k in venvs} if venvs else set()
    binaries = report.get("binaries")
    mounts = report.get("mounts")
    nvidia = report.get("nvidia")

    blockers: list[Finding] = []
    warnings: list[Finding] = []
    infos: list[Finding] = []
    ok_count = 0

    for entry in doctrine.entries:
        if entry.kind == "pip":
            key = (entry.venv[len("profile:"):]
                   if entry.venv.startswith("profile:") else entry.venv)
            block = venvs.get(key) if isinstance(venvs, Mapping) else None
            packages = (block.get("packages")
                        if isinstance(block, Mapping) else None)
            if not isinstance(packages, Mapping):
                packages = None
            companions = (block.get("torch_companions")
                          if isinstance(block, Mapping) else None)
            if not isinstance(companions, Mapping):
                companions = None
            finding = _pip_finding(report, entry, packages,
                                   key in present_profiles, companions)
        elif entry.kind == "binary":
            finding = _binary_finding(entry, binaries)
        elif entry.kind == "mount":
            finding = _mount_finding(entry, mounts)
        elif entry.kind == "driver":
            finding = _driver_finding(entry, nvidia)
        else:  # pragma: no cover — kinds are closed by Requirement.__post_init__
            continue

        if finding.status == STATUS_OK:
            ok_count += 1
            if finding.severity != DEFAULT_SEVERITY:
                continue        # a satisfied blocker/warn is not a finding
            continue
        if finding.blocking:
            blockers.append(finding)
        elif finding.severity == "warn" or (
                finding.severity == "blocker"
                and finding.status in (STATUS_UNKNOWN, STATUS_NO_PROFILE)):
            # A blocker we could NOT evaluate degrades to a warning. It must be
            # visible; it must not gate work. (Rule 1 in this module's docstring.)
            warnings.append(finding)
        else:
            infos.append(finding)

    return DoctrineReport(
        worker=str(report.get("worker") or ""),
        doctrine_version=doctrine.version,
        doctrine_provisional=doctrine.provisional,
        report_digest=str(report.get("report_digest") or ""),
        assessed_at=_now(),
        blockers=tuple(blockers), warnings=tuple(warnings), infos=tuple(infos),
        ok_count=ok_count, checked=len(doctrine.entries))


def render(assessment: DoctrineReport, *, show_info: bool = False) -> str:
    """Human-readable assessment. Same shape as ``station/bin/hugpy-doctor``:
    fixed six-space-aligned status prefixes, no colors, no cleverness."""
    out: list[str] = [assessment.summary(), ""]
    if assessment.doctrine_provisional:
        out.append("  note  this doctrine is PROVISIONAL")
        out.append("")
    for label, findings in (("FAIL", assessment.blockers),
                            ("WARN", assessment.warnings)):
        for finding in findings:
            out.append(f"  {label}  {finding.line()}")
            if finding.repair:
                out.append(f"        repair: {finding.repair}")
    if show_info:
        for finding in assessment.infos:
            out.append(f"  info  {finding.line()}")
    if not assessment.blockers and not assessment.warnings:
        out.append("  PASS  every classified doctrine entry is satisfied")
    blocked = assessment.blocked_tasks()
    if blocked:
        out.append("")
        out.append(f"  tasks blocked here: {', '.join(blocked)}")
    return "\n".join(out)


__all__ = [
    "STATUS_COMPANION_MISMATCH",
    "STATUS_DRIFT", "STATUS_MISSING", "STATUS_NO_PROFILE", "STATUS_OK",
    "STATUS_PIN", "STATUS_UNKNOWN",
    "VERDICT_BLOCKED", "VERDICT_OK", "VERDICT_UNKNOWN", "VERDICT_WARN",
    "DoctrineReport", "Finding", "assess", "render",
]
