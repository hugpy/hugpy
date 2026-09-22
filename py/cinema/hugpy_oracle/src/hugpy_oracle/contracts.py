"""Oracle contracts (k90a): the typed currency of route-to-best-result.

Every contract is a frozen, slotted dataclass in the studio-schemas idiom
(``video_intel/studio/schemas.py``): closed vocabularies are str-Enums (they
serialize straight to JSON), collections are tuples so instances stay hashable,
and structurally-invalid values are programmer error raised in ``__post_init__``
— never runtime data that leaks downstream.

These are the contracts k90b (``POST /oracle/route``: execute → ExecutionReceipt
+ deterministic technical Scorecard) and k90c (evaluator kernel: heterogeneous
checks, judges, repair codes) build ON — so receipts and scorecards are defined
here NOW, not when their producers land. The catalog (``catalog.py``) produces
``CapabilityView``; ``GET /oracle/capabilities`` serializes it via ``to_dict``.

k97 adds the typed AUTHORITY currency (``AuthorityKind``, ``Authorization``,
``RightsManifest``) and the truthful ``PlannerMode``, both carried on
``GoalSpec`` — invariants 7 (authority is typed) and 8 (planner participation
is truthful). ``oracle/authority.py`` decides with them; the router refuses
with them BEFORE a model is picked.

k101 grows ``CapabilityView`` into the doc §3.2 ``CapabilityDescriptor``:
semver, typed artifact I/O, param/result schemas, declared limits, a UNIFIED
resource profile, declared authority + host access, license, evaluation suite,
adapter version, model fingerprint and a REGISTRATION PROBE. Every addition is
optional with a default, so every k90a/k97/k98 constructor still builds — the
descriptor is a superset, never a migration. Two invariants are enforced at
construction rather than documented:

  * a ``ProbeResult`` may not claim ``ok`` while one of its checks failed, and
    a probe with no checks is ``unknown`` (a probe that could not run reports
    unknown, NEVER ok — the whole roadmap rule in one dataclass);
  * a view carrying a FAILED probe cannot be eligible (doc §3.2: "an adapter
    that unexpectedly requires ``prompt`` is ineligible until its descriptor
    and probe agree"). ``with_probe`` is the honest way to attach one.

k101 also settles the THREE spellings of one quantity k103 flagged:
``ResourceHints.min_vram_gb`` (advisory), ``ResourceRequest.vram_gib`` (plan)
and ``BudgetHints.max_vram_gb`` (budget cap) → the canonical unit spelling is
``vram_gib`` everywhere, carrying a ``Provenance`` flag so a MEASURED number is
never confused with an ESTIMATE. The old names survive as read-only properties
(and ``ResourceHints(min_vram_gb=…)`` still constructs), so no caller breaks.

Wire shape: every contract has ``to_dict()`` (JSON-safe, enums as their str
values) and a ``from_dict()`` classmethod, so round-tripping through HTTP/store
is lossless and proven in ``tests/test_oracle_contracts.py``.
``oracle/schema_export.py`` generates JSON Schema straight from these
dataclasses — the doc's "generated schema" without a pydantic migration.

No pathlib anywhere. os.path only (not that this module touches the disk).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterator, Mapping


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class QualityProfile(str, Enum):
    PREVIEW = "preview"
    BALANCED = "balanced"
    BEST = "best"


class InputKind(str, Enum):
    """What the operator SUPPLIED with the goal (typed input refs)."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    URL = "url"


class ArtifactKind(str, Enum):
    """What a capability accepts/produces. Superset of InputKind: execution also
    yields derived kinds (embeddings, structured JSON, extracted documents)."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    URL = "url"
    EMBEDDING = "embedding"
    JSON = "json"
    DOCUMENT = "document"


class SourceRegistry(str, Enum):
    """Which of the two existing registries owns a capability (the k90a bridge
    is READ-ONLY composition over both — see catalog.py)."""
    STUDIO = "studio"   # video_intel/studio: typed ModelConfig/Capability zoo
    TASKS = "tasks"     # imports/config/models: legacy tasks-string registry


class FailureClass(str, Enum):
    """ExecutionReceipt failure classification — WHERE an execution died, not
    what to regenerate (that is RepairCode, the evaluator's verdict)."""
    DECODE_FAILED = "decode_failed"
    EMPTY_OUTPUT = "empty_output"
    FORMAT_MISMATCH = "format_mismatch"
    TIMEOUT = "timeout"
    WORKER_UNAVAILABLE = "worker_unavailable"
    CAPABILITY_GAP = "capability_gap"
    RUNNER_ERROR = "runner_error"      # the runner raised / returned Err
    REFUSED = "refused"                # policy/license/authority gate said no
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class CheckKind(str, Enum):
    """The heterogeneous-evidence axes (action_plan: technical / identity /
    semantic / temporal / speech / sync / whole-result intent)."""
    TECHNICAL = "technical"
    IDENTITY = "identity"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    SPEECH = "speech"
    SYNC = "sync"
    INTENT = "intent"


class RepairCode(str, Enum):
    """A failing evaluation names WHAT to regenerate — the repair controller
    (k90c) maps the code to a bounded subgraph, never a full re-run."""
    IDENTITY_DRIFT = "identity_drift"
    ACTION_MISSING = "action_missing"
    VOICE_SIMILARITY_LOW = "voice_similarity_low"
    LINE_OMITTED = "line_omitted"
    SHOT_TOO_SHORT = "shot_too_short"
    LIP_SYNC_OUT_OF_RANGE = "lip_sync_out_of_range"
    TEMPORAL_ARTIFACT = "temporal_artifact"
    GEOMETRY_DRIFT = "geometry_drift"            # k116/k119: spatial adherence below threshold
    CAMERA_PATH_MISMATCH = "camera_path_mismatch"  # rendered camera diverges from the locked track
    COLLISION_VIOLATION = "collision_violation"  # diffusion contradicted the simulation's contacts
    INTENT_MISMATCH = "intent_mismatch"    # judge: artifact does not satisfy the goal
    SOURCE_AUTHORITY_MISSING = "source_authority_missing"
    DECODE_FAILED = "decode_failed"
    EMPTY_OUTPUT = "empty_output"
    FORMAT_MISMATCH = "format_mismatch"
    TIMEOUT = "timeout"
    WORKER_UNAVAILABLE = "worker_unavailable"
    CAPABILITY_GAP = "capability_gap"


class AuthorityKind(str, Enum):
    """The typed permission axes (architecture invariant 7: "rights, consent,
    filesystem, network, shell, model, and artifact-disclosure permissions are
    explicit gates"). One vocabulary for BOTH halves of the gate: what a
    *subject* consented to (``likeness``/``voice``/``dialogue_source``/
    ``web_source``) and what a *capability* needs from the host
    (``filesystem``/``network``/``shell``/``disclosure``)."""
    LIKENESS = "likeness"                  # reproduce a person's appearance
    VOICE = "voice"                        # reproduce/clone a person's voice
    DIALOGUE_SOURCE = "dialogue_source"    # use a script/transcript as source
    WEB_SOURCE = "web_source"              # use an acquired web asset
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    SHELL = "shell"
    DISCLOSURE = "disclosure"              # release an artifact past a scope


class PlannerMode(str, Enum):
    """Invariant 8 — planner participation is truthful. ``local_only`` is the
    DEFAULT and the honest description of this fleet today: no Frontier Keeper
    A is wired in, so no response may imply that A participated."""
    FRONTIER = "frontier"
    LOCAL_ONLY = "local_only"


class Provenance(str, Enum):
    """Where a declared number CAME FROM (k101). The distinction is the whole
    reason the field exists: ``4.0 GiB measured on a 3090`` and ``4.0 GiB
    someone guessed from the checkpoint size`` are not the same fact, and a
    placement decision that cannot tell them apart is a lie waiting to happen.
    UNKNOWN is the default and pairs with a ``None`` value."""
    MEASURED = "measured"      # observed on real hardware, recorded
    ESTIMATED = "estimated"    # derived (weights size, precision, upstream card)
    DECLARED = "declared"      # asserted by a registry row / model card
    UNKNOWN = "unknown"        # nobody knows — the honest default


class AccessKind(str, Enum):
    """Host resources a capability's implementation TOUCHES (doc §3.2:
    "Network, filesystem, shell, and disclosure requirements"). Deliberately a
    separate vocabulary from ``AuthorityKind``: this is a DECLARATION about the
    adapter ("web.fetch talks to the internet"), while ``AuthorityKind`` is a
    GATE about a request. ``authority.required_authorities`` maps one to the
    other when a descriptor is supplied — see that module for why declaring is
    not yet the same as enforcing."""
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    SHELL = "shell"
    EXTERNAL = "external"      # a third-party service/API beyond this fleet


class ProbeStatus(str, Enum):
    """Registration-probe verdict. THREE values, not two, and the third is the
    load-bearing one: a check that could not be performed answers ``unknown``
    and never ``ok`` (roadmap ground rule: ineligible > faked, unknown > "sure,
    fine"). ``fail`` means the probe actively DISAGREES with the descriptor."""
    OK = "ok"
    FAIL = "fail"
    UNKNOWN = "unknown"


# Worst-wins ordering for folding many check verdicts into one result.
_PROBE_SEVERITY: dict[str, int] = {
    ProbeStatus.OK.value: 0,
    ProbeStatus.UNKNOWN.value: 1,
    ProbeStatus.FAIL.value: 2,
}


# ---------------------------------------------------------------------------
# Primitives shared by every contract (k101)
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """The one canonical encoding used for every digest the oracle computes:
    keys sorted, no insignificant whitespace. Same trick as
    ``ExecutionReceipt.normalize_request`` and ``plan.canonical_json`` — the
    same content encodes identically no matter how the dicts were built."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def coerce_artifact_kind(value: "ArtifactKind | str") -> "ArtifactKind | str":
    """``ArtifactKind`` when the string names one, otherwise the string itself.

    The enum is the closed vocabulary of MEDIA kinds; the platform also moves
    LOGICAL artifacts it will never enumerate (``dialogue_timeline``,
    ``audio_master``, ``segment_spec``, ``scorecard``). Inventing enum members
    for artifacts that do not exist yet is the fabrication the doc warns about,
    so a free string is legal and compared exactly, case-sensitively.

    (``plan.coerce_artifact_kind`` is the same function, written first in k103.
    It lives here too because ``contracts`` is the bottom of the import graph
    and ``plan`` already imports it — k103 can re-base on this one whenever its
    file is open; the two are proven identical by a test.)"""
    if isinstance(value, ArtifactKind):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("artifact kind must be non-empty")
    try:
        return ArtifactKind(text)
    except ValueError:
        return text


def kind_value(kind: "ArtifactKind | str") -> str:
    return kind.value if isinstance(kind, ArtifactKind) else str(kind)


def _freeze_value(value: Any) -> Any:
    """Recursively freeze a JSON-ish value: dict -> FrozenMap, list/tuple ->
    tuple, scalars unchanged. Anything else is rejected loudly — a schema that
    smuggles a live object in is not a schema."""
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, _MappingABC):
        return FrozenMap(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"schema/limit values must be JSON-ish (mapping, sequence, str, int, "
        f"float, bool, None); got {type(value).__name__}")


def _thaw_value(value: Any) -> Any:
    if isinstance(value, FrozenMap):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_thaw_value(v) for v in value]
    return value


class FrozenMap(_MappingABC):
    """An immutable, recursively frozen, hashable ``Mapping[str, JSON]``.

    Frozen dataclasses that carry a SCHEMA (``param_schema``/``result_schema``/
    ``limits``) must stay hashable and un-mutable-by-a-caller; a plain dict is
    neither. Reading is ordinary mapping access (``view.limits["formats"]``),
    iteration is key-sorted so anything built from it is deterministic without
    the caller remembering to sort, and ``to_dict()`` hands back plain JSON.

    (Same shape as ``plan.FrozenParams``, which k103 wrote first for node
    params. One of them should absorb the other when both files are next open;
    they are proven equivalent by a test rather than left to drift.)"""

    __slots__ = ("_data", "_json")

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        items: dict[str, Any] = {}
        for key, value in dict(data or {}).items():
            if not isinstance(key, str):
                raise TypeError(f"schema keys must be str, got {key!r}")
            items[key] = _freeze_value(value)
        object.__setattr__(self, "_data", items)
        object.__setattr__(
            self, "_json",
            canonical_json({k: _thaw_value(v) for k, v in items.items()}))

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("FrozenMap is immutable")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._data))

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, FrozenMap):
            return self._json == other._json
        if isinstance(other, _MappingABC):
            try:
                return self.to_dict() == FrozenMap(other).to_dict()
            except (TypeError, ValueError):
                return NotImplemented
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._json)

    def __repr__(self) -> str:
        return f"FrozenMap({self._json})"

    def to_dict(self) -> dict[str, Any]:
        return {k: _thaw_value(v) for k, v in sorted(self._data.items())}


def freeze_map(data: Mapping[str, Any] | None) -> FrozenMap:
    """``FrozenMap`` from anything mapping-ish (idempotent)."""
    return data if isinstance(data, FrozenMap) else FrozenMap(data)


#: Semver, deliberately strict-but-small: MAJOR.MINOR.PATCH with an optional
#: pre-release/build tail. A capability version that is not comparable is not a
#: version, and "1.0" silently sorting before "1.0.0" is exactly the class of
#: bug a registry snapshot must not have.
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")

#: Every capability starts here. A descriptor that has never been revised says
#: so honestly rather than claiming 1.0.0 stability it has not earned.
DEFAULT_CAPABILITY_VERSION: str = "0.1.0"


# ---------------------------------------------------------------------------
# Registration probe (k101) — doc §3.2 "Health and registration probes"
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProbeCheck:
    """One cheap, named fact a registration probe established (or could not).

    ``detail`` is REQUIRED for anything that is not ``ok``: an unknown with no
    explanation is indistinguishable from laziness, and a fail with no
    explanation cannot become an eligibility reason a human can act on."""
    name: str
    status: ProbeStatus
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ProbeCheck.name must be non-empty")
        if self.status is not ProbeStatus.OK and not self.detail.strip():
            raise ValueError(
                f"ProbeCheck({self.name!r}, {self.status.value}) needs a detail "
                f"— a probe that is not ok must say why")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status.value,
                "detail": self.detail}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ProbeCheck":
        return cls(name=d["name"], status=ProbeStatus(d["status"]),
                   detail=d.get("detail", ""))


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """The verdict of one capability's registration probe.

    ``status`` is DERIVED, not asserted: it is the worst of the checks (fail >
    unknown > ok), and a result with no checks is ``unknown``. Constructing a
    result that disagrees with its own checks raises — the point of a probe is
    that it cannot be talked into optimism."""
    status: ProbeStatus
    checks: tuple[ProbeCheck, ...] = ()
    probed_at: str = ""

    def __post_init__(self) -> None:
        derived = self.derive_status(self.checks)
        if self.status is not derived:
            raise ValueError(
                f"ProbeResult.status={self.status.value!r} disagrees with its "
                f"checks (worst is {derived.value!r}): "
                f"{[(c.name, c.status.value) for c in self.checks]}")

    @staticmethod
    def derive_status(checks: "tuple[ProbeCheck, ...]") -> ProbeStatus:
        if not checks:
            return ProbeStatus.UNKNOWN
        worst = max(checks, key=lambda c: _PROBE_SEVERITY[c.status.value])
        return worst.status

    @classmethod
    def from_checks(cls, checks: "tuple[ProbeCheck, ...] | list[ProbeCheck]",
                    probed_at: str = "") -> "ProbeResult":
        checks = tuple(checks)
        return cls(status=cls.derive_status(checks), checks=checks,
                   probed_at=probed_at)

    @property
    def ok(self) -> bool:
        return self.status is ProbeStatus.OK

    @property
    def failed(self) -> bool:
        return self.status is ProbeStatus.FAIL

    def failures(self) -> tuple[ProbeCheck, ...]:
        return tuple(c for c in self.checks if c.status is ProbeStatus.FAIL)

    def reason(self) -> str:
        """One line naming every failing check — the string that becomes an
        ``Eligibility`` reason when a probe refuses a capability."""
        bad = self.failures()
        if not bad:
            return ""
        return "registration probe failed: " + "; ".join(
            f"{c.name}: {c.detail}" for c in bad)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value,
                "checks": [c.to_dict() for c in self.checks],
                "probed_at": self.probed_at}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ProbeResult":
        return cls(status=ProbeStatus(d["status"]),
                   checks=tuple(ProbeCheck.from_dict(c)
                                for c in d.get("checks", ())),
                   probed_at=d.get("probed_at", ""))


# ---------------------------------------------------------------------------
# Authority (k97) — typed rights, never inferred
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Authorization:
    """One granted permission, with the evidence that grants it.

    ``subject`` is WHO/WHAT the permission is about: an ``identity_profile:
    <slug>`` reference (canonical, the same string the video routes accept), a
    ``voice_profile:<slug>``, a person label, a source URL, or the literal
    ``"*"`` — a blanket grant for that kind (see ``RightsManifest.covers``).

    ``evidence`` is a free-text pointer to the thing a human can go read: a
    release-form path, a ticket id, a contract reference. It is REQUIRED and it
    is never derived: an authorization the operator cannot evidence is exactly
    the inferred consent §11 forbids, so it cannot be constructed here."""
    kind: AuthorityKind
    subject: str
    scope: str = ""          # free-text: what use the grant covers
    evidence: str = ""
    granted_by: str = ""     # who says so (operator id, rights holder)
    granted_at: str = ""     # ISO-8601, like LedgerEvent.at

    def __post_init__(self) -> None:
        if not self.subject.strip():
            raise ValueError("Authorization.subject must be non-empty")
        if not self.evidence.strip():
            raise ValueError(
                f"Authorization({self.kind.value}, {self.subject!r}) needs "
                f"evidence — consent is never inferred, it is pointed at")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "subject": self.subject,
                "scope": self.scope, "evidence": self.evidence,
                "granted_by": self.granted_by, "granted_at": self.granted_at}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Authorization":
        return cls(kind=AuthorityKind(d["kind"]), subject=d["subject"],
                   scope=d.get("scope", ""), evidence=d.get("evidence", ""),
                   granted_by=d.get("granted_by", ""),
                   granted_at=d.get("granted_at", ""))


@dataclass(frozen=True, slots=True)
class RightsManifest:
    """The authority a request rides in with — doc §7 Stage 1's output.

    ``denied`` is the explicit NO list, which always beats a grant: an entry is
    either a bare subject (``"identity_profile:mira"``) or a kind-qualified one
    (``"likeness:identity_profile:mira"``). Matching is case-insensitive on the
    stripped string; nothing here is fuzzy, wildcard-except-``*``, or inferred.

    ABSENCE IS NOT CONSENT: no manifest (``GoalSpec.rights is None``) and an
    empty manifest mean the same thing — nothing is authorized."""
    authorizations: tuple[Authorization, ...] = ()
    denied: tuple[str, ...] = ()
    notes: str = ""

    @staticmethod
    def _norm(s: str) -> str:
        return s.strip().lower()

    def covers(self, kind: AuthorityKind, subject: str) -> bool:
        """Does this manifest authorize ``kind`` for ``subject``? An explicit
        denial wins; otherwise an exact subject match of the same kind, or a
        ``"*"`` blanket GRANT of that kind — nothing else. A specific grant
        never satisfies a blanket need (``subject="*"``): "we cleared Mira" is
        not "we cleared whoever this ends up being"."""
        want = self._norm(subject)
        for d in self.denied:
            nd = self._norm(d)
            if nd in (want, f"{kind.value}:{want}"):
                return False
        for a in self.authorizations:
            if a.kind is not kind:
                continue
            have = self._norm(a.subject)
            if have == want or have == "*":
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {"authorizations": [a.to_dict() for a in self.authorizations],
                "denied": list(self.denied), "notes": self.notes}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RightsManifest":
        return cls(
            authorizations=tuple(Authorization.from_dict(a)
                                 for a in d.get("authorizations", ())),
            denied=tuple(d.get("denied", ())),
            notes=d.get("notes", ""),
        )


# ---------------------------------------------------------------------------
# GoalSpec (lite)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InputRef:
    """One supplied input, typed. ``ref`` is whatever the transport hands over
    (an upload path/handle, a URL, inline text) — the oracle resolves it at
    execution time; the contract only pins WHAT KIND of thing it is."""
    kind: InputKind
    ref: str
    label: str = ""

    def __post_init__(self) -> None:
        if not self.ref:
            raise ValueError("InputRef.ref must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "ref": self.ref, "label": self.label}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "InputRef":
        return cls(kind=InputKind(d["kind"]), ref=d["ref"],
                   label=d.get("label", ""))


@dataclass(frozen=True, slots=True)
class BudgetHints:
    """Soft budget hints — planning inputs, not enforced limits (admission and
    placement stay where they live today).

    ``max_vram_gb`` keeps its k90a name (it is on the wire and in the validator)
    but reads as GiB like everything else; ``max_vram_gib`` is the alias in the
    unit spelling k101 settled on."""
    max_seconds: float | None = None
    max_vram_gb: float | None = None

    @property
    def max_vram_gib(self) -> float | None:
        """The k101 unit spelling of ``max_vram_gb`` (same number, one name)."""
        return self.max_vram_gb

    def __post_init__(self) -> None:
        for name, val in (("max_seconds", self.max_seconds),
                          ("max_vram_gb", self.max_vram_gb)):
            if val is not None and val <= 0:
                raise ValueError(f"BudgetHints.{name} must be positive, got {val}")

    def to_dict(self) -> dict[str, Any]:
        return {"max_seconds": self.max_seconds, "max_vram_gb": self.max_vram_gb}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BudgetHints":
        return cls(max_seconds=d.get("max_seconds"),
                   max_vram_gb=d.get("max_vram_gb"))


@dataclass(frozen=True, slots=True)
class GoalSpec:
    """The normalized meaning of an operator request (lite slice: no approval
    gates and no clarification loop yet — those phases stay deferred by the
    keeper assessment; typed AUTHORITY landed in k97 and is no longer deferred).

    ``raw_prompt`` is the operator's message VERBATIM and immutable alongside
    the normalized ``objective`` — normalization must never destroy the source.
    ``capability`` is a namespaced catalog name (``audio.transcribe``); None
    means auto (k90b's intent classifier picks).

    ``rights`` is the request's ``RightsManifest`` (doc §7 Stage 1) — None means
    NOTHING is authorized, never "unrestricted"; ``oracle.authority`` reads it
    and the router refuses before a model is picked. ``planner_mode`` defaults
    to ``local_only`` because that is the truth on this fleet (invariant 8): a
    response must never imply Frontier Keeper A participated. ``disclosure_
    scope`` names how far a produced artifact may travel (``operator`` = back to
    the requester only) — recorded now, enforced when a disclosure gate lands."""
    objective: str
    raw_prompt: str
    inputs: tuple[InputRef, ...] = ()
    capability: str | None = None
    quality: QualityProfile = QualityProfile.BALANCED
    budget: BudgetHints = field(default_factory=BudgetHints)
    acceptance: tuple[str, ...] = ()   # free-text acceptance notes, for now
    planner_mode: PlannerMode = PlannerMode.LOCAL_ONLY
    rights: RightsManifest | None = None
    disclosure_scope: str = "operator"

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("GoalSpec.objective must be non-empty")
        if not self.raw_prompt.strip():
            raise ValueError("GoalSpec.raw_prompt must be non-empty (the "
                             "operator's message rides alongside, immutable)")
        if self.capability is not None and "." not in self.capability:
            raise ValueError(
                f"GoalSpec.capability must be a namespaced catalog name "
                f"(e.g. 'audio.transcribe'), got {self.capability!r}")
        if not self.disclosure_scope.strip():
            raise ValueError("GoalSpec.disclosure_scope must be non-empty "
                             "(default 'operator'); blank is not a scope")

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "raw_prompt": self.raw_prompt,
            "inputs": [i.to_dict() for i in self.inputs],
            "capability": self.capability,
            "quality": self.quality.value,
            "budget": self.budget.to_dict(),
            "acceptance": list(self.acceptance),
            "planner_mode": self.planner_mode.value,
            "rights": self.rights.to_dict() if self.rights else None,
            "disclosure_scope": self.disclosure_scope,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GoalSpec":
        rights = d.get("rights")
        return cls(
            objective=d["objective"],
            raw_prompt=d["raw_prompt"],
            inputs=tuple(InputRef.from_dict(i) for i in d.get("inputs", ())),
            capability=d.get("capability"),
            quality=QualityProfile(d.get("quality", "balanced")),
            budget=BudgetHints.from_dict(d.get("budget") or {}),
            acceptance=tuple(d.get("acceptance", ())),
            planner_mode=PlannerMode(d.get("planner_mode", "local_only")),
            rights=RightsManifest.from_dict(rights) if rights else None,
            disclosure_scope=d.get("disclosure_scope", "operator"),
        )


# ---------------------------------------------------------------------------
# CapabilityView — the unified catalog descriptor (both registries)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Eligibility:
    """Can this capability execute on the fleet RIGHT NOW, and if not, why not.
    An ineligible view with no reasons is a contract violation — the whole
    point of GET /oracle/capabilities is explaining refusals BEFORE execution
    (Phase-1 done-criterion). Reasons on an ELIGIBLE view are allowed: they are
    advisory (e.g. 'no online worker; central serves it in-process')."""
    eligible: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.eligible and not self.reasons:
            raise ValueError("ineligible Eligibility must carry >=1 reason")

    def to_dict(self) -> dict[str, Any]:
        return {"eligible": self.eligible, "reasons": list(self.reasons)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Eligibility":
        return cls(eligible=bool(d["eligible"]),
                   reasons=tuple(d.get("reasons", ())))


@dataclass(frozen=True, slots=True, init=False)
class ResourceHints:
    """The resource profile of a capability, in ONE unit spelling (k101).

    Where known. The studio zoo carries real VRAM envelopes; the legacy tasks
    registry carries none — None means unknown, never fabricated.

    ``vram_gib`` is the canonical field: k103 flagged three spellings for one
    quantity (``ResourceHints.min_vram_gb``, ``ResourceRequest.vram_gib``,
    ``BudgetHints.max_vram_gb``) and this is the pick. ``vram_provenance``
    carries the fact that decides whether the number may be trusted for
    placement: a MEASURED 6.2 GiB is a fact, an ESTIMATED 6.2 GiB is a guess
    with a decimal point, and the default UNKNOWN pairs with ``None``.

    ``min_vram_gb`` survives as a READ-ONLY property AND as a constructor
    keyword, so every k90a/k98 call site (``ResourceHints(min_vram_gb=4.0)``)
    and every reader (``view.resources.min_vram_gb``) keeps working unchanged.
    ``to_dict`` emits both names for the same reason."""

    vram_gib: float | None = None
    vram_provenance: Provenance = Provenance.UNKNOWN
    frameworks: tuple[str, ...] = ()
    notes: str = ""
    est_seconds: float | None = None

    def __init__(self, vram_gib: float | None = None,
                 vram_provenance: Provenance | str = Provenance.UNKNOWN,
                 frameworks: tuple[str, ...] = (), notes: str = "",
                 est_seconds: float | None = None,
                 min_vram_gb: float | None = None) -> None:
        # Legacy spelling: accepted, reconciled, never silently disagreed with.
        if min_vram_gb is not None:
            if vram_gib is not None and float(vram_gib) != float(min_vram_gb):
                raise ValueError(
                    f"ResourceHints got two different VRAM figures: "
                    f"vram_gib={vram_gib!r} and min_vram_gb={min_vram_gb!r} — "
                    f"they are the same quantity, pick one")
            vram_gib = float(min_vram_gb)
        if vram_gib is not None:
            vram_gib = float(vram_gib)
            if vram_gib < 0:
                raise ValueError(f"vram_gib must be >= 0, got {vram_gib}")
        provenance = (vram_provenance if isinstance(vram_provenance, Provenance)
                      else Provenance(vram_provenance))
        if vram_gib is None and provenance is not Provenance.UNKNOWN:
            raise ValueError(
                f"vram_provenance={provenance.value!r} with no vram_gib — "
                f"provenance describes a number that has to exist")
        if est_seconds is not None:
            est_seconds = float(est_seconds)
            if est_seconds < 0:
                raise ValueError(f"est_seconds must be >= 0, got {est_seconds}")
        object.__setattr__(self, "vram_gib", vram_gib)
        object.__setattr__(self, "vram_provenance", provenance)
        object.__setattr__(self, "frameworks", tuple(frameworks))
        object.__setattr__(self, "notes", notes)
        object.__setattr__(self, "est_seconds", est_seconds)

    @property
    def min_vram_gb(self) -> float | None:
        """k90a's spelling of ``vram_gib`` — same number, kept so no reader
        breaks. New code should read ``vram_gib``."""
        return self.vram_gib

    @property
    def measured(self) -> bool:
        return self.vram_provenance is Provenance.MEASURED

    def to_dict(self) -> dict[str, Any]:
        return {"vram_gib": self.vram_gib,
                "vram_provenance": self.vram_provenance.value,
                # legacy mirror: existing wire consumers keep reading it
                "min_vram_gb": self.vram_gib,
                "frameworks": list(self.frameworks), "notes": self.notes,
                "est_seconds": self.est_seconds}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ResourceHints":
        vram = d.get("vram_gib")
        if vram is None:
            vram = d.get("min_vram_gb")
        return cls(vram_gib=vram,
                   vram_provenance=Provenance(
                       d.get("vram_provenance")
                       or (Provenance.UNKNOWN.value if vram is None
                           else Provenance.DECLARED.value)),
                   frameworks=tuple(d.get("frameworks", ())),
                   notes=d.get("notes", ""),
                   est_seconds=d.get("est_seconds"))


@dataclass(frozen=True, slots=True)
class CapabilityView:
    """One namespaced capability as the unified catalog presents it — the
    READ-ONLY join of a registry's rows, its runner/viability gates, and the
    fleet's worker-health signals. Source of truth stays in the two registries;
    this is a view, never a third store.

    k101 turns it into the doc §3.2 ``CapabilityDescriptor``. Everything added
    is optional with a default, so every existing constructor still builds and
    every existing reader still reads; a field nobody can answer honestly stays
    ``None``/empty rather than being filled with a plausible guess:

      ``version``            semver of the DESCRIPTOR (not of the model).
      ``accepts``/``produces`` artifact kinds; a logical kind the enum does not
                             have (``dialogue_timeline``) stays a plain string.
      ``param_schema``/``result_schema``  JSON-Schema-ish, frozen. This is what
                             a registration probe checks the adapter against.
      ``limits``             formats / languages / max_duration_s /
                             max_resolution / context_tokens — only what is
                             actually knowable on this fleet.
      ``resources``          the unified VRAM/est-seconds profile.
      ``authority_required`` typed rights the capability needs BY CONSTRUCTION.
      ``access``             host resources it touches (network/filesystem/…).
      ``license``            declared license, None = not recorded (never
                             "assume permissive").
      ``eval_suite``         canonical evaluation suite name.
      ``adapter_version``    the runner adapter's version.
      ``model_fingerprint``  sha of the bound model's manifest when cheaply
                             available, else None.
      ``probe``              the registration probe's result, or None when no
                             probe is registered for this capability.
      ``registry_version``   the catalog snapshot this view was built from.

    INVARIANT (doc §3.2, verbatim: "an image adapter that unexpectedly requires
    ``prompt`` … is ineligible until its descriptor and probe agree with the
    endpoint"): a view carrying a FAILED probe cannot be eligible. Attach
    probes with ``with_probe``, which downgrades eligibility and folds the
    probe's own words in as the reason."""
    name: str                          # namespaced: "audio.transcribe"
    source: SourceRegistry
    accepts: tuple[ArtifactKind | str, ...]
    produces: tuple[ArtifactKind | str, ...]
    model_ids: tuple[str, ...]
    eligibility: Eligibility
    resources: ResourceHints = field(default_factory=ResourceHints)
    # --- k101 descriptor fields (all optional, all default-honest) ----------
    version: str = DEFAULT_CAPABILITY_VERSION
    param_schema: FrozenMap = field(default_factory=FrozenMap)
    result_schema: FrozenMap = field(default_factory=FrozenMap)
    limits: FrozenMap = field(default_factory=FrozenMap)
    authority_required: tuple[AuthorityKind, ...] = ()
    access: tuple[AccessKind, ...] = ()
    license: str | None = None
    eval_suite: str | None = None
    adapter_version: str | None = None
    model_fingerprint: str | None = None
    probe: ProbeResult | None = None
    registry_version: str | None = None

    def __post_init__(self) -> None:
        if "." not in self.name:
            raise ValueError(f"CapabilityView.name must be namespaced, got {self.name!r}")
        if not self.produces:
            raise ValueError(f"{self.name}: a capability must produce something")
        if not _SEMVER.match(self.version or ""):
            raise ValueError(
                f"{self.name}: version must be semver MAJOR.MINOR.PATCH, got "
                f"{self.version!r}")
        object.__setattr__(self, "accepts",
                           tuple(coerce_artifact_kind(k) for k in self.accepts))
        object.__setattr__(self, "produces",
                           tuple(coerce_artifact_kind(k) for k in self.produces))
        object.__setattr__(self, "authority_required", tuple(
            k if isinstance(k, AuthorityKind) else AuthorityKind(k)
            for k in self.authority_required))
        object.__setattr__(self, "access", tuple(
            a if isinstance(a, AccessKind) else AccessKind(a)
            for a in self.access))
        for attr in ("param_schema", "result_schema", "limits"):
            object.__setattr__(self, attr, freeze_map(getattr(self, attr)))
        if (self.probe is not None and self.probe.failed
                and self.eligibility.eligible):
            raise ValueError(
                f"{self.name}: eligible=True while its registration probe "
                f"FAILED ({self.probe.reason()}) — a descriptor and its probe "
                f"must agree before the capability is offered; use with_probe()")

    # -- probe application ---------------------------------------------------

    def with_probe(self, probe: ProbeResult | None) -> "CapabilityView":
        """This view carrying ``probe``, with eligibility reconciled.

        A FAILED probe makes the capability INELIGIBLE and the probe's own
        detail becomes the reason (that string is what a human acts on). An
        ``unknown`` probe changes nothing but is recorded as an advisory reason
        — the fleet does not know, and says so. An ``ok`` probe is recorded and
        adds no noise."""
        if probe is None:
            return replace(self, probe=None)
        reasons = list(self.eligibility.reasons)
        eligible = self.eligibility.eligible
        if probe.failed:
            eligible = False
            for line in (probe.reason(),):
                if line and line not in reasons:
                    reasons.append(line)
        elif probe.status is ProbeStatus.UNKNOWN:
            for chk in probe.checks:
                if chk.status is ProbeStatus.UNKNOWN:
                    line = f"registration probe inconclusive — {chk.name}: {chk.detail}"
                    if line not in reasons:
                        reasons.append(line)
        if not eligible and not reasons:      # Eligibility demands a reason
            reasons.append("registration probe refused this capability")
        return replace(self, probe=probe,
                       eligibility=Eligibility(eligible=eligible,
                                               reasons=tuple(reasons)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source.value,
            "accepts": [kind_value(k) for k in self.accepts],
            "produces": [kind_value(k) for k in self.produces],
            "model_ids": list(self.model_ids),
            "eligibility": self.eligibility.to_dict(),
            "resources": self.resources.to_dict(),
            "version": self.version,
            "param_schema": self.param_schema.to_dict(),
            "result_schema": self.result_schema.to_dict(),
            "limits": self.limits.to_dict(),
            "authority_required": [k.value for k in self.authority_required],
            "access": [a.value for a in self.access],
            "license": self.license,
            "eval_suite": self.eval_suite,
            "adapter_version": self.adapter_version,
            "model_fingerprint": self.model_fingerprint,
            "probe": self.probe.to_dict() if self.probe else None,
            "registry_version": self.registry_version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CapabilityView":
        probe = d.get("probe")
        return cls(
            name=d["name"],
            source=SourceRegistry(d["source"]),
            accepts=tuple(coerce_artifact_kind(k) for k in d.get("accepts", ())),
            produces=tuple(coerce_artifact_kind(k) for k in d.get("produces", ())),
            model_ids=tuple(d.get("model_ids", ())),
            eligibility=Eligibility.from_dict(d["eligibility"]),
            resources=ResourceHints.from_dict(d.get("resources") or {}),
            version=d.get("version", DEFAULT_CAPABILITY_VERSION),
            param_schema=freeze_map(d.get("param_schema")),
            result_schema=freeze_map(d.get("result_schema")),
            limits=freeze_map(d.get("limits")),
            authority_required=tuple(AuthorityKind(k)
                                     for k in d.get("authority_required", ())),
            access=tuple(AccessKind(a) for a in d.get("access", ())),
            license=d.get("license"),
            eval_suite=d.get("eval_suite"),
            adapter_version=d.get("adapter_version"),
            model_fingerprint=d.get("model_fingerprint"),
            probe=ProbeResult.from_dict(probe) if probe else None,
            registry_version=d.get("registry_version"),
        )


# ---------------------------------------------------------------------------
# ExecutionReceipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A produced artifact by pointer + content hash. ``sha256`` is None only
    when the artifact is not a local file we could hash (e.g. a stream)."""
    kind: ArtifactKind
    uri: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("ArtifactRef.uri must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "uri": self.uri, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ArtifactRef":
        return cls(kind=ArtifactKind(d["kind"]), uri=d["uri"],
                   sha256=d.get("sha256"))


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """What actually ran, on what, with what result — one per routed execution
    (k90b emits one on EVERY /oracle/route response, success or not).

    ``request`` is the NORMALIZED request as sorted ``(key, json-value)`` pairs
    (tuple-of-pairs keeps the dataclass frozen/hashable, same trick as the
    studio's VramEnvelope). Build it with ``normalize_request``; read it back
    with ``request_dict``. Timestamps are ISO-8601 strings, like the studio
    ledger's ``LedgerEvent.at``.

    ``registry_version`` (k101) is the digest of the routing snapshot this
    execution was resolved against — ``catalog.registry_version()``. It is what
    makes a receipt REPRODUCIBLE rather than merely descriptive: "this ran on
    whisper-x" is only meaningful next to "…out of THIS set of rows and THESE
    descriptor versions". None means the snapshot was not recorded (an older
    receipt, or a caller that did not read the catalog) — never a guess, and
    never derived from a ``GoalSpec``, which knows nothing about the fleet."""
    request: tuple[tuple[str, str], ...]
    capability: str
    model_id: str
    worker: str | None
    started_at: str
    ended_at: str
    duration_s: float
    retries: int = 0
    failure: FailureClass | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    warnings: tuple[str, ...] = ()
    log_excerpt: tuple[str, ...] = ()
    registry_version: str | None = None

    def __post_init__(self) -> None:
        if not self.capability:
            raise ValueError("ExecutionReceipt.capability must be non-empty")
        if self.duration_s < 0:
            raise ValueError(f"negative duration_s: {self.duration_s}")
        if self.retries < 0:
            raise ValueError(f"negative retries: {self.retries}")

    @staticmethod
    def normalize_request(req: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
        """Mapping -> canonical sorted (key, json-encoded value) pairs. Stable
        across dict ordering, so identical requests normalize identically."""
        return tuple(
            (k, json.dumps(req[k], sort_keys=True, separators=(",", ":")))
            for k in sorted(req)
        )

    def request_dict(self) -> dict[str, Any]:
        return {k: json.loads(v) for k, v in self.request}

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request_dict(),
            "capability": self.capability,
            "model_id": self.model_id,
            "worker": self.worker,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "retries": self.retries,
            "failure": self.failure.value if self.failure else None,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "warnings": list(self.warnings),
            "log_excerpt": list(self.log_excerpt),
            "registry_version": self.registry_version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExecutionReceipt":
        failure = d.get("failure")
        return cls(
            request=cls.normalize_request(d.get("request") or {}),
            capability=d["capability"],
            model_id=d["model_id"],
            worker=d.get("worker"),
            started_at=d["started_at"],
            ended_at=d["ended_at"],
            duration_s=float(d["duration_s"]),
            retries=int(d.get("retries", 0)),
            failure=FailureClass(failure) if failure else None,
            artifacts=tuple(ArtifactRef.from_dict(a) for a in d.get("artifacts", ())),
            warnings=tuple(d.get("warnings", ())),
            log_excerpt=tuple(d.get("log_excerpt", ())),
            registry_version=d.get("registry_version"),
        )


# ---------------------------------------------------------------------------
# Scorecard (load-bearing per operator decision 2026-08-05)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Check:
    """One deterministic/measured check. ``value``/``threshold`` are loosely
    typed on purpose (a decode check has no numeric threshold; a similarity
    check does) — ``kind`` + ``name`` say what the numbers mean."""
    name: str
    kind: CheckKind
    value: float | str | bool | None
    threshold: float | str | None
    passed: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Check.name must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind.value, "value": self.value,
                "threshold": self.threshold, "passed": self.passed,
                "detail": self.detail}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Check":
        return cls(name=d["name"], kind=CheckKind(d["kind"]),
                   value=d.get("value"), threshold=d.get("threshold"),
                   passed=bool(d["passed"]), detail=d.get("detail", ""))


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """One model-judge's opinion (VLM rubric, round-trip ASR compare, …).
    Kept apart from Check: judges are heterogeneous EVIDENCE, not gates —
    disagreement between them is signal, recorded on the Scorecard."""
    judge: str
    verdict: str
    score: float | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.judge:
            raise ValueError("JudgeResult.judge must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {"judge": self.judge, "verdict": self.verdict,
                "score": self.score, "rationale": self.rationale}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "JudgeResult":
        return cls(judge=d["judge"], verdict=d["verdict"],
                   score=d.get("score"), rationale=d.get("rationale", ""))


@dataclass(frozen=True, slots=True)
class Scorecard:
    """The machine-readable verdict on one artifact/execution. The generating
    model never declares its own result good — this is built from checks and
    judges OUTSIDE the generator (k90b: deterministic technical checks on every
    response; k90c: the full heterogeneous kernel).

    Invariants: ``confidence`` in [0, 1]; a ``repair_code`` on a hard-PASSING
    card is incoherent (a repair is a diagnosis of failure) and refused."""
    hard_pass: bool
    checks: tuple[Check, ...] = ()
    judge_results: tuple[JudgeResult, ...] = ()
    confidence: float = 1.0
    disagreements: tuple[str, ...] = ()
    diagnosis: str | None = None
    repair_code: RepairCode | None = None
    recommended_repair: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.hard_pass and self.repair_code is not None:
            raise ValueError(
                f"a hard-passing Scorecard cannot carry repair_code "
                f"{self.repair_code.value!r} — repairs diagnose failures")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_pass": self.hard_pass,
            "checks": [c.to_dict() for c in self.checks],
            "judge_results": [j.to_dict() for j in self.judge_results],
            "confidence": self.confidence,
            "disagreements": list(self.disagreements),
            "diagnosis": self.diagnosis,
            "repair_code": self.repair_code.value if self.repair_code else None,
            "recommended_repair": self.recommended_repair,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Scorecard":
        code = d.get("repair_code")
        return cls(
            hard_pass=bool(d["hard_pass"]),
            checks=tuple(Check.from_dict(c) for c in d.get("checks", ())),
            judge_results=tuple(JudgeResult.from_dict(j)
                                for j in d.get("judge_results", ())),
            confidence=float(d.get("confidence", 1.0)),
            disagreements=tuple(d.get("disagreements", ())),
            diagnosis=d.get("diagnosis"),
            repair_code=RepairCode(code) if code else None,
            recommended_repair=d.get("recommended_repair"),
        )
