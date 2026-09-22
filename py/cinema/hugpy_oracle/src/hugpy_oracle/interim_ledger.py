"""k122 — the Interim Ledger: ONE cross-surface index of every in-between object.

The operator principle (``IDEA_PHASE/PRINCIPLE-the-interim.md``): hugpy's defining
feature is the inspectable interim between a prompt and its deliverable. Every
surface already persists its own interim objects — and nothing joined them. A
keyframe attempt lived in a script-first ``state.json``, the job that rendered it
lived in ``media_jobs.db``, the identity it referenced lived in
``identity_profiles``, and the only thing connecting the three was an operator
holding three tabs open.

THIS MODULE IS READ-SIDE ONLY. It opens every source read-only (sqlite
``mode=ro``, ``open(path)`` for JSON) and writes exactly one thing: its own
rebuildable index cache under ``<run_root>/runs/interim_ledger/``. Deleting that
cache costs a rescan and nothing else. No adapter may write to the store it
reads — :func:`SOURCES` is checked by a test that hands each adapter a read-only
fixture tree.

THE SHAPE. Every native record — a media_bus row, a script-first attempt, a
benchmark cell, an identity reconstruction — is mapped into one
:class:`InterimEntry`. The fields are the principle's clauses made queryable:
``parents``/``produced_by`` are clause 3 (provenance), ``status``/``gap`` are
clause 3's honest status (unscored never reads as passed), ``artifact_refs`` and
``scorecard_ref`` are clause 2 (the in-between object is first-class).

HOW THE JOIN ACTUALLY WORKS. Almost nothing on the write side records a typed
parent pointer. What the surfaces DO record is *identifiers they happen to
share*: a media_bus job's ``spec.source.asset_id`` is some other job's
``result.outputs[].asset_id``; an identity reconstruction's ``job_id`` is a
media_bus primary key; a script-first attempt's ``parents`` are artifact digests
stored in the same run. So each entry publishes ``aliases`` — every id the rest
of the fleet might call it by (job id, asset id, artifact uri, digest, run dir)
— and the ledger resolves a raw parent ref against that alias index. Joins that
exist are real. Joins that do not are reported as ``unresolved_parents``, never
invented: that list IS the write-side provenance gap map.

Degradation is honest and per-source. An adapter whose store is absent,
unreadable or whose module will not import yields no entries and reports
``source_unavailable`` with the reason verbatim in :func:`stats`. It never
reports zero as if zero were a finding.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

#: Bump when :class:`InterimEntry` gains or loses a field — a cache written by
#: an older build is discarded rather than misread.
LEDGER_VERSION: str = "interim_ledger/1"

#: Override the root the cache is written under (tests point this at a tmpdir).
CACHE_ROOT_ENV: str = "HUGPY_INTERIM_LEDGER_ROOT"

#: Seconds before a cached index is considered stale by :func:`load_ledger`.
DEFAULT_MAX_AGE_S: float = 120.0

# --------------------------------------------------------------------------- #
# The honest status vocabulary.
#
# ``unscored`` is deliberately NOT a status — an entry can be ``done`` and
# unscored at the same time, and collapsing the two is exactly the lie clause 3
# forbids. Scoring is read off ``verdict``/``scorecard_ref`` instead, so a
# terminal-and-unjudged entry cannot masquerade as a pass.
# --------------------------------------------------------------------------- #
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_RUNNING = "running"
STATUS_QUEUED = "queued"
STATUS_GAP = "gap"
STATUS_REFUSED = "refused"
STATUS_UNKNOWN = "unknown"

STATUSES: tuple[str, ...] = (
    STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED, STATUS_RUNNING,
    STATUS_QUEUED, STATUS_GAP, STATUS_REFUSED, STATUS_UNKNOWN,
)

TERMINAL_STATUSES: frozenset[str] = frozenset({
    STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED, STATUS_GAP, STATUS_REFUSED,
})

#: Surface ids. One per store, not one per artifact kind — ``kind`` is the
#: within-surface discriminator.
SURFACE_MEDIA_BUS = "media_bus"
SURFACE_SCRIPT_FIRST = "script_first"
SURFACE_PERFORMANCE = "performance"
SURFACE_ORACLE = "oracle"
SURFACE_MCT = "mct"
SURFACE_IDENTITY = "identity"
SURFACE_DISCOVERY = "discovery"
SURFACE_BENCHMARK = "benchmark"
SURFACE_COORDINATION = "coordination"


class LedgerRefused(Exception):
    """A ledger-level refusal, shaped like the rest of the oracle's refusals.

    Mirrors ``script_first.ScriptFirstRefused`` field for field (including the
    ``error`` alias the React transport keeps) so the routes layer needs no
    second error convention.
    """

    http_status: int = 400

    def __init__(self, code: str, message: str, *,
                 errors: Sequence[str] = (),
                 detail: Mapping[str, Any] | None = None,
                 http_status: int | None = None) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.errors = tuple(str(e) for e in errors)
        self.detail = dict(detail or {})
        if http_status is not None:
            self.http_status = int(http_status)

    def to_dict(self) -> dict[str, Any]:
        joined = "\n".join((f"{self.code}: {self.message}",) + self.errors)
        return {"ok": False, "code": self.code, "message": self.message,
                "errors": list(self.errors), "detail": self.detail,
                "error": joined}


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #

def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _iso(value: Any) -> str | None:
    """Any of the four timestamp dialects on disk -> ISO-8601 UTC, or None.

    The fleet writes epoch floats (media_bus, identity_profiles), epoch ints,
    already-ISO strings (script_first, benchmark) and, occasionally, nothing.
    A timestamp that cannot be read is None — never ``now``, which would sort a
    decade-old row to the top of a ``since`` query.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return (datetime.fromtimestamp(float(value), tz=timezone.utc)
                    .replace(microsecond=0).isoformat())
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text if _ISO_ISH.match(text) else None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


_ISO_ISH = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _digest(payload: Any) -> str:
    """A short, stable digest of any jsonable payload."""
    try:
        blob = json.dumps(payload, sort_keys=True, default=str,
                          separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        blob = repr(payload).encode("utf-8", "replace")
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


def _read_json(path: str) -> Any | None:
    """Read a JSON file, or None. Never raises — a corrupt sibling run must not
    take the whole ledger down with it."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        logger.debug("interim_ledger: unreadable json %s (%s)", path, exc)
        return None


def _as_map(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_seq(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _clean_refs(values: Iterable[Any]) -> tuple[str, ...]:
    """Dedupe + stringify refs, dropping empties, preserving first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def default_run_root() -> str:
    """Where the ledger's own cache lives.

    The same derivation ``script_first.default_run_root`` and
    ``performance.default_run_root`` use, so the cache lands beside the runs it
    indexes. Copied rather than imported for the same reason they copy it: those
    modules belong to other tasks and this is four lines, not an abstraction.
    """
    from hugpy_oracle.config import ledger_cache_root
    return ledger_cache_root()


def cache_dir(root: str | None = None) -> str:
    return os.path.join(root or default_run_root(), "runs", "interim_ledger")


def cache_path(root: str | None = None) -> str:
    return os.path.join(cache_dir(root), "index.json")


# --------------------------------------------------------------------------- #
# The one shape
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class InterimEntry:
    """One in-between generation object, from any surface.

    ``entry_id`` is ``<surface>:<kind>:<native id>`` — globally unique and
    human-legible in a URL, so an operator can paste a tree link into a report.

    ``aliases`` is what makes the cross-surface join possible without a single
    write-side change: every OTHER id this object answers to (its bare job id,
    its output asset ids, its artifact uris, its content digests). A parent ref
    recorded by another surface in any of those dialects still lands.

    ``parents`` holds RAW refs exactly as the source recorded them. Resolution
    to ``entry_id`` happens in :class:`InterimLedger`, which can also report
    what did not resolve. Storing resolved ids here would silently drop the
    dangling ones — and the dangling ones are the finding.
    """

    entry_id: str
    surface: str
    kind: str
    created_at: str | None = None
    status: str = STATUS_UNKNOWN
    terminal: bool = False
    parents: tuple[str, ...] = ()
    produced_by: dict[str, Any] = field(default_factory=dict)
    artifact_refs: tuple[str, ...] = ()
    scorecard_ref: str | None = None
    verdict: str | None = None
    gap: str | None = None
    registry_version: str | None = None
    source_pointer: str = ""
    aliases: tuple[str, ...] = ()
    label: str = ""

    def __post_init__(self) -> None:
        if not self.entry_id:
            raise ValueError("entry_id must be non-empty")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}; "
                             f"expected one of {STATUSES}")
        object.__setattr__(self, "terminal",
                           bool(self.terminal or self.status in TERMINAL_STATUSES))

    @property
    def scored(self) -> bool:
        """True only when something actually judged this object.

        Clause 3: unscored never reads as passed. A ``done`` entry with no
        verdict and no scorecard is ``scored is False``, and :func:`stats`
        counts it in ``unscored`` no matter how green its status looks.
        """
        return bool(self.verdict) or bool(self.scorecard_ref)

    @property
    def has_gap(self) -> bool:
        return bool(self.gap) or self.status in (STATUS_GAP, STATUS_REFUSED)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["parents"] = list(self.parents)
        out["artifact_refs"] = list(self.artifact_refs)
        out["aliases"] = list(self.aliases)
        out["scored"] = self.scored
        out["has_gap"] = self.has_gap
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InterimEntry":
        data = dict(payload)
        data.pop("scored", None)
        data.pop("has_gap", None)
        known = {f for f in cls.__dataclass_fields__}          # noqa: SLF001
        data = {k: v for k, v in data.items() if k in known}
        for key in ("parents", "artifact_refs", "aliases"):
            data[key] = tuple(data.get(key) or ())
        data["produced_by"] = _as_map(data.get("produced_by"))
        return cls(**data)


def make_entry_id(surface: str, kind: str, native_id: str) -> str:
    return f"{surface}:{kind}:{native_id}"


# --------------------------------------------------------------------------- #
# The adapter protocol
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SourceReport:
    """What one adapter did on one scan — including nothing, honestly."""

    surface: str
    available: bool
    reason: str = ""
    count: int = 0
    scanned_pointer: str = ""
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["errors"] = list(self.errors)
        return out


class InterimSource:
    """Base adapter. Subclasses map ONE native store into :class:`InterimEntry`.

    Contract, enforced by ``test_interim_ledger.py``:

    * ``collect`` must not write, create, chmod or unlink anything under the
      store it reads. Tests run each adapter against a read-only fixture tree.
    * ``collect`` must not raise. A store that is absent, truncated, or whose
      module will not import is a :class:`SourceReport` with ``available=False``
      and the reason verbatim — never an exception, and never a silent zero.
    """

    surface: str = "unknown"

    def __init__(self, root: str | None = None) -> None:
        self.root = root

    # -- to override --------------------------------------------------------
    def probe(self) -> tuple[bool, str, str]:
        """``(available, reason, pointer)``. Cheap: stat, not read."""
        raise NotImplementedError

    def _entries(self) -> Iterator[InterimEntry]:
        raise NotImplementedError

    # -- the driver ---------------------------------------------------------
    def collect(self) -> tuple[list[InterimEntry], SourceReport]:
        try:
            available, reason, pointer = self.probe()
        except Exception as exc:                    # noqa: BLE001
            return [], SourceReport(self.surface, False,
                                    f"probe failed: {type(exc).__name__}: {exc}")
        if not available:
            return [], SourceReport(self.surface, False, reason,
                                    scanned_pointer=pointer)
        errors: list[str] = []
        entries: list[InterimEntry] = []
        try:
            for entry in self._entries():
                entries.append(entry)
        except Exception as exc:                    # noqa: BLE001
            logger.warning("interim_ledger: %s scan aborted (%s: %s)",
                           self.surface, type(exc).__name__, exc, exc_info=True)
            errors.append(f"{type(exc).__name__}: {exc}")
        return entries, SourceReport(self.surface, True, reason, len(entries),
                                     pointer, tuple(errors))


# --------------------------------------------------------------------------- #
# 1. media_bus — the biggest store, and the one with the thinnest provenance
# --------------------------------------------------------------------------- #

#: media_bus ``status`` -> ledger status. ``claimed``/``cancelling`` are
#: in-flight states the bus uses internally.
_BUS_STATUS = {
    "done": STATUS_DONE, "failed": STATUS_FAILED, "cancelled": STATUS_CANCELLED,
    "queued": STATUS_QUEUED, "claimed": STATUS_RUNNING,
    "running": STATUS_RUNNING, "cancelling": STATUS_RUNNING,
}

#: Where a job spec keeps a reference to something ANOTHER job made. Note what
#: is NOT here: a ``parent_job_id`` column. There isn't one, anywhere, and that
#: is the headline write-side finding (see :func:`provenance_gaps`).
_BUS_PARENT_KEYS = ("source", "start_image", "control_image", "source_video",
                    "reference_images", "view_sources", "source_images",
                    "identity_id", "recon_id", "slug", "battery_dir",
                    "project_path", "parts", "goals", "chain")

#: The IMPLICIT job->job edge. media_bus writes every artifact under
#: ``<store>/<job_id>/…``, so when job B's spec points at a file job A wrote,
#: A's id is sitting in the path. Recovering it is the difference between a
#: ledger with 96 real bus edges and a ledger with none. Fragile by nature —
#: it depends on a directory convention, not a contract — which is exactly why
#: the write-side fix (record the parent job id) is on the follow-up list.
_BUS_URI_PARENT = re.compile(
    r"/(?:movies|scenes|frames|studio_movies|crops|audio)/([0-9a-f]{32})/")

#: k117 reserves one non-job row in the sidecar for the watchdog's own state.
_LIFECYCLE_SENTINEL = "__watchdog__"

#: Spec keys that describe HOW, not WHAT-FROM — folded into ``produced_by``.
_BUS_PRODUCER_KEYS = ("model_id", "seed", "steps", "guidance", "cfg", "width",
                      "height", "fps", "strength", "n_frames", "frames",
                      "capability", "judge_model_id", "mode", "backend")


def _bus_refs(value: Any, depth: int = 0) -> list[str]:
    """Pull every id-ish string out of a spec value.

    A ``source`` is ``{"asset_id":..., "uri":...}``; a ``reference_images`` is a
    list of paths; an ``identity_id`` is a bare string. All three are refs and
    all three are worth indexing, because which dialect the parent published is
    not knowable from the child.
    """
    if depth > 3:
        return []
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Mapping):
        out: list[str] = []
        for key in ("asset_id", "uri", "path", "id", "media", "ref"):
            if key in value:
                out.extend(_bus_refs(value[key], depth + 1))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in list(value)[:24]:
            out.extend(_bus_refs(item, depth + 1))
        return out
    return []


def _bus_parent_jobs(refs: Iterable[str]) -> list[str]:
    """Job ids recoverable from artifact paths — the implicit bus edge."""
    out: list[str] = []
    for ref in refs:
        out.extend(_BUS_URI_PARENT.findall(ref))
    return out


class MediaBusSource(InterimSource):
    """media_bus jobs + their stage_log + the k117 lifecycle sidecar.

    Read through a private ``mode=ro`` sqlite handle rather than
    ``media_bus.list_jobs``. Not a shortcut — a deliberate, recorded trade:
    ``list_jobs`` is capped at 200 rows and its projection drops ``spec_json``
    and ``result_json``, which are the only two columns carrying parent refs
    and artifact refs. A ledger built on the projection would be a ledger with
    no edges. ``mode=ro`` is the same guarantee ``list_jobs`` relies on
    (``media_bus._connect_ro``): the connection physically cannot take a write
    lock, so this path cannot corrupt or block the bus even under a bug.
    """

    surface = SURFACE_MEDIA_BUS

    def __init__(self, root: str | None = None, db_path: str | None = None,
                 limit: int = 5000) -> None:
        super().__init__(root)
        self.limit = int(limit)
        self._db_path = db_path

    def db_path(self) -> str:
        if self._db_path:
            return self._db_path
        try:
            from hugpy_video.intel import media_bus
            return str(media_bus.DB_PATH)
        except Exception as exc:                    # noqa: BLE001
            logger.debug("interim_ledger: media_bus import failed (%s)", exc)
            return os.path.join(default_run_root(), "media_jobs.db")

    def probe(self) -> tuple[bool, str, str]:
        path = self.db_path()
        if not os.path.exists(path):
            return False, f"source_unavailable: no media_jobs.db at {path}", path
        return True, "", path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect("file:" + self.db_path() + "?mode=ro",
                               uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _lifecycle(self, conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
        """The k117 sidecar, one query for the whole scan.

        Read straight off the sidecar table rather than through
        ``job_lifecycle.read_side_many`` because that helper opens its own
        read/write handle; here the whole point is that nothing in this module
        holds anything but ``mode=ro``. The column list is imported from
        ``job_lifecycle`` so a schema change there surfaces here.
        """
        try:
            from hugpy_video.intel import job_lifecycle
            table = job_lifecycle._TABLE                    # noqa: SLF001
        except Exception:                           # noqa: BLE001
            table = "media_job_lifecycle"
        out: dict[str, dict[str, Any]] = {}
        try:
            rows = conn.execute(f"select * from {table}").fetchall()   # noqa: S608
        except sqlite3.Error as exc:
            logger.debug("interim_ledger: no k117 sidecar (%s)", exc)
            return out
        for row in rows:
            data = dict(row)
            job_id = str(data.get("job_id"))
            if job_id == _LIFECYCLE_SENTINEL:
                continue        # the watchdog's own state, not a job
            out[job_id] = data
        return out

    def _entries(self) -> Iterator[InterimEntry]:
        conn = self._connect()
        try:
            side = self._lifecycle(conn)
            rows = conn.execute(
                "select job_id, name, status, spec_json, result_json, created,"
                " updated, principal, stage_log_json, owner, archived_at"
                " from media_jobs order by created desc limit ?",
                (self.limit,)).fetchall()
        finally:
            conn.close()
        for row in rows:
            entry = self._map_row(dict(row), side)
            if entry is not None:
                yield entry

    def _map_row(self, row: Mapping[str, Any],
                 side: Mapping[str, Mapping[str, Any]]) -> InterimEntry | None:
        job_id = str(row.get("job_id") or "").strip()
        if not job_id:
            return None
        name = str(row.get("name") or "job")
        spec = _as_map(_loads(row.get("spec_json")))
        result = _as_map(_loads(row.get("result_json")))
        stage_log = _as_seq(_loads(row.get("stage_log_json")))
        life = _as_map(side.get(job_id))

        raw_status = str(row.get("status") or "").lower()
        status = _BUS_STATUS.get(raw_status, STATUS_UNKNOWN)

        # A failed job's error is a GAP in the principle's sense: an in-between
        # object the pipeline could not produce, with a stated reason.
        gap = None
        error = result.get("error")
        if status == STATUS_FAILED:
            gap = str(error) if error else "job failed without a recorded reason"
        elif error:
            gap = str(error)

        artifacts: list[str] = []
        asset_ids: list[str] = []
        for output in _as_seq(result.get("outputs")):
            out = _as_map(output)
            uri = out.get("uri") or out.get("path")
            if uri:
                artifacts.append(str(uri))
            if out.get("asset_id"):
                asset_ids.append(str(out["asset_id"]))
        movie = result.get("movie")
        if isinstance(movie, str) and movie:
            artifacts.append(movie)

        spec_refs: list[str] = []
        for key in _BUS_PARENT_KEYS:
            if key in spec:
                spec_refs.extend(_bus_refs(spec[key]))
        # generate_scene reports which identity profiles it consumed.
        for ident in _as_seq(result.get("identities")):
            spec_refs.extend(_bus_refs(ident))
        # The implicit edge first: a recovered job id is a far stronger parent
        # than the path it was recovered from, and the ledger resolves in order.
        parents: list[str] = _bus_parent_jobs(spec_refs) + spec_refs

        produced = {k: spec[k] for k in _BUS_PRODUCER_KEYS if k in spec}
        produced["step"] = name
        produced["params_digest"] = _digest(
            {k: v for k, v in spec.items() if k not in _BUS_PARENT_KEYS})
        if life.get("terminal_status"):
            produced["terminal_status"] = life["terminal_status"]
        if life.get("at_stage"):
            produced["terminal_stage"] = life["at_stage"]
        for key in ("queue_wait_s", "run_s", "started_at", "terminal_at"):
            if life.get(key) is not None:
                produced[key] = life[key]
        if stage_log:
            last = _as_map(stage_log[-1])
            produced["last_stage"] = last.get("stage")
            produced["stage_count"] = len(stage_log)
        if row.get("principal"):
            produced["principal"] = row["principal"]
        if row.get("archived_at"):
            produced["archived_at"] = _iso(row["archived_at"])

        # A media_bus job is judged only when a judge model was asked for AND
        # the result carries its call. Nothing else counts as scored.
        verdict = None
        score = result.get("score")
        if score is not None:
            verdict = f"score={score}"

        aliases = [job_id, f"job:{job_id}"] + asset_ids + artifacts
        return InterimEntry(
            entry_id=make_entry_id(self.surface, name, job_id),
            surface=self.surface,
            kind=name,
            created_at=_iso(row.get("created")),
            status=status,
            terminal=bool(life.get("terminal_at")) or status in TERMINAL_STATUSES,
            parents=_clean_refs(parents),
            produced_by=produced,
            artifact_refs=_clean_refs(artifacts),
            scorecard_ref=None,
            verdict=verdict,
            gap=gap,
            registry_version=None,
            source_pointer=f"media_jobs.db#media_jobs/{job_id}",
            aliases=_clean_refs(aliases),
            label=f"{name} {job_id[:12]}",
        )


def _loads(blob: Any) -> Any:
    if blob in (None, ""):
        return None
    if isinstance(blob, (Mapping, list)):
        return blob
    try:
        return json.loads(blob)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# 2. script-first runs — the provenance gold standard
# --------------------------------------------------------------------------- #

class ScriptFirstSource(InterimSource):
    """``runs/script_first/*/state.json`` fanned out into its real interim objects.

    A script-first run is not one interim object, it is five kinds of them, and
    flattening it to one row per run would throw away exactly the detail the
    principle exists to preserve. So a run yields: the run itself, each stored
    artifact (plot/screenplay/continuity/audio_master/shot_plan), each compiled
    segment spec, each generation attempt, and each authoring gap journalled in
    ``events``. Digests are published as aliases, which is why an attempt's
    ``parents`` (lock digest, snapshot digest, sibling artifact digests) resolve
    to real ledger entries with no write-side change at all.
    """

    surface = SURFACE_SCRIPT_FIRST
    run_kind = "script_first"

    def runs_dir(self) -> str:
        return os.path.join(self.root or default_run_root(), "runs", self.run_kind)

    def probe(self) -> tuple[bool, str, str]:
        path = self.runs_dir()
        if not os.path.isdir(path):
            return False, f"source_unavailable: no run dir at {path}", path
        return True, "", path

    def _entries(self) -> Iterator[InterimEntry]:
        base = self.runs_dir()
        try:
            names = sorted(os.listdir(base))
        except OSError as exc:
            logger.debug("interim_ledger: script_first listdir failed (%s)", exc)
            return
        for name in names:
            if name.startswith("_") or name.startswith("."):
                continue
            state = _read_json(os.path.join(base, name, "state.json"))
            if not isinstance(state, Mapping):
                continue
            yield from self._map_run(name, dict(state),
                                     os.path.join(base, name, "state.json"))

    def _map_run(self, run_id: str, state: Mapping[str, Any],
                 pointer: str) -> Iterator[InterimEntry]:
        run_id = str(state.get("run_id") or run_id)
        snapshot_digest = state.get("snapshot_digest")
        lock = _as_map(state.get("lock"))
        lock_digest = lock.get("digest")
        events = _as_seq(state.get("events"))
        last_refusal = _as_map(state.get("last_refusal"))

        gaps = [e for e in (_as_map(x) for x in events)
                if str(e.get("event", "")).endswith("gap")]
        run_gap = None
        if last_refusal:
            run_gap = (f"{last_refusal.get('code')}: {last_refusal.get('message')}"
                       .strip(": "))
        elif gaps:
            run_gap = f"{len(gaps)} authoring gap(s) journalled"

        status = STATUS_DONE if lock_digest else STATUS_RUNNING
        if last_refusal:
            status = STATUS_REFUSED
        elif gaps and not lock_digest:
            status = STATUS_GAP

        run_entry_id = make_entry_id(self.surface, "run", run_id)
        aliases = _clean_refs([run_id, snapshot_digest, lock_digest])
        yield InterimEntry(
            entry_id=run_entry_id, surface=self.surface, kind="run",
            created_at=_iso(state.get("created_at")),
            status=status,
            parents=(),
            produced_by={"step": "script_first.run",
                         "deliverable": _as_map(state.get("snapshot"))
                         .get("deliverable"),
                         "updated_at": _iso(state.get("updated_at")),
                         "version": state.get("version"),
                         "params_digest": _digest(state.get("settings"))},
            artifact_refs=(),
            gap=run_gap,
            source_pointer=pointer,
            aliases=aliases,
            label=f"script-first run {run_id}",
        )

        # -- the input snapshot: the run's own root provenance object ---------
        if snapshot_digest:
            snapshot = _as_map(state.get("snapshot"))
            yield InterimEntry(
                entry_id=make_entry_id(self.surface, "snapshot",
                                       str(snapshot_digest)),
                surface=self.surface, kind="snapshot",
                created_at=_iso(snapshot.get("created_at")
                                or state.get("created_at")),
                status=STATUS_DONE,
                parents=(run_entry_id,),
                produced_by={"step": "snapshot",
                             "sources": len(_as_seq(state.get("sources"))),
                             "params_digest": str(snapshot_digest)[:16]},
                artifact_refs=(),
                source_pointer=f"{pointer}#snapshot",
                aliases=_clean_refs([snapshot_digest]),
                label=f"snapshot {str(snapshot_digest)[:12]}",
            )

        # -- stored artifacts -------------------------------------------------
        for art_name, raw in _as_map(state.get("artifacts")).items():
            art = _as_map(raw)
            digest = art.get("digest")
            # A GAPPED artifact has no digest — the authoring step never
            # produced one. Keying it on the run+name instead of skipping it is
            # the whole point: an artifact the pipeline failed to make is an
            # interim object with a stated reason, and dropping it for want of
            # a digest would make a gapped run look like a clean one. (Caught
            # live: sf-20260821-48e66c6072's `plot` is AUTHORING_UNPARSED with
            # digest=None and was invisible until this branch existed.)
            native = str(digest) if digest else f"{run_id}#{art_name}"
            provenance = _as_map(art.get("provenance"))
            parents = _clean_refs(
                list(_as_seq(provenance.get("parents")))
                + list(_as_seq(art.get("parents")))
                + [snapshot_digest, run_entry_id])
            yield InterimEntry(
                entry_id=make_entry_id(self.surface, f"artifact:{art_name}",
                                       native),
                surface=self.surface, kind=f"artifact:{art_name}",
                created_at=_iso(art.get("at")),
                status=STATUS_GAP if (art.get("gap") or not digest)
                else STATUS_DONE,
                parents=parents,
                produced_by={"step": art.get("stage") or art_name,
                             "model": provenance.get("model_id"),
                             "note": art.get("note"),
                             "params_digest": str(digest or native)[:16]},
                artifact_refs=(),
                gap=(_stringify_gap(art.get("gap"))
                     or (None if digest else "no digest: artifact never authored")),
                registry_version=provenance.get("registry_version"),
                source_pointer=f"{pointer}#artifacts/{art_name}",
                aliases=_clean_refs([digest]),
                label=f"{art_name} {str(digest)[:12] if digest else '(gapped)'}",
            )

        # -- the production lock ----------------------------------------------
        if lock_digest:
            yield InterimEntry(
                entry_id=make_entry_id(self.surface, "lock", str(lock_digest)),
                surface=self.surface, kind="lock",
                created_at=_iso(lock.get("at")),
                status=STATUS_DONE,
                parents=_clean_refs(list(_as_seq(lock.get("parent_digests")))
                                    + [run_entry_id]),
                produced_by={"step": "production_lock",
                             "revision": lock.get("revision"),
                             "params_digest": str(lock_digest)[:16]},
                artifact_refs=(),
                source_pointer=f"{pointer}#lock",
                aliases=_clean_refs([lock_digest]),
                label=f"production lock {str(lock_digest)[:12]}",
            )

        # -- compiled segment specs -------------------------------------------
        segments = _as_map(state.get("segments"))
        for raw in _as_seq(segments.get("specs")):
            spec = _as_map(raw)
            digest = spec.get("digest")
            seg_id = spec.get("segment_id") or spec.get("id") or spec.get("index")
            if not digest:
                continue
            yield InterimEntry(
                entry_id=make_entry_id(self.surface, "segment", str(digest)),
                surface=self.surface, kind="segment",
                created_at=_iso(segments.get("compiled_at")),
                status=STATUS_DONE,
                parents=_clean_refs(list(_as_seq(spec.get("parents")))
                                    + [spec.get("lock_digest"), run_entry_id]),
                produced_by={"step": "compile_segments",
                             "index": spec.get("index"),
                             "joint_mode": spec.get("joint_mode"),
                             "params_digest": str(digest)[:16]},
                artifact_refs=(),
                source_pointer=f"{pointer}#segments/specs",
                aliases=_clean_refs([digest, seg_id]),
                label=f"segment {seg_id}",
            )

        # -- generation attempts: the interim objects proper -------------------
        for seg_id, raw_attempts in _as_map(state.get("attempts")).items():
            for raw in _as_seq(raw_attempts):
                attempt = _as_map(raw)
                yield self._map_attempt(run_entry_id, run_id, str(seg_id),
                                        attempt, pointer)

        # -- promotions: the ONLY cross-RUN edge any surface writes today -----
        for raw in _as_seq(state.get("promotions")):
            promotion = _as_map(raw)
            digest = promotion.get("digest")
            if not digest:
                continue
            origin = _as_map(promotion.get("origin"))
            source_id = promotion.get("source_id")
            yield InterimEntry(
                entry_id=make_entry_id(self.surface, "promotion", str(digest)),
                surface=self.surface, kind="promotion",
                created_at=_iso(promotion.get("promoted_at")),
                status=STATUS_DONE,
                parents=_clean_refs([origin.get("spec_digest"),
                                     origin.get("lock_digest"),
                                     origin.get("segment_id"),
                                     origin.get("run_id"), run_entry_id]),
                produced_by={"step": "promote_source",
                             "note": promotion.get("note"),
                             "origin_run": origin.get("run_id"),
                             "refused_here": promotion.get("refused_here"),
                             "params_digest": str(digest)[:16]},
                artifact_refs=_clean_refs(
                    _as_map(a).get("uri") for a in _as_seq(origin.get("artifacts"))),
                source_pointer=f"{pointer}#promotions/{digest}",
                aliases=_clean_refs([digest, source_id]),
                label=f"promotion {str(digest)[:12]}",
            )

        # -- journalled authoring gaps ----------------------------------------
        for index, raw in enumerate(events):
            event = _as_map(raw)
            name = str(event.get("event") or "")
            if not name.endswith("gap") and name != "derived_dropped":
                continue
            detail = _as_map(event.get("detail"))
            code = detail.get("code") or name
            yield InterimEntry(
                entry_id=make_entry_id(self.surface, "gap",
                                       f"{run_id}#{index}"),
                surface=self.surface, kind="gap",
                created_at=_iso(event.get("at")),
                status=STATUS_GAP,
                parents=(run_entry_id,),
                produced_by={"step": name, "code": code,
                             "params_digest": _digest(detail)},
                artifact_refs=(),
                gap=_stringify_gap(detail) or str(code),
                source_pointer=f"{pointer}#events/{index}",
                aliases=(),
                label=f"{name} {code}",
            )

    def _map_attempt(self, run_entry_id: str, run_id: str, seg_id: str,
                     attempt: Mapping[str, Any], pointer: str) -> InterimEntry:
        number = attempt.get("attempt") or 0
        kind = f"attempt:{attempt.get('kind') or 'generation'}"
        native = f"{run_id}#{seg_id}#{number}"
        artifacts = [str(_as_map(a).get("uri")) for a in _as_seq(attempt.get("artifacts"))
                     if _as_map(a).get("uri")]
        receipt = _as_map(attempt.get("receipt"))
        gap = _stringify_gap(attempt.get("gap"))
        if attempt.get("ok"):
            status = STATUS_DONE
        elif gap:
            status = STATUS_GAP
        else:
            status = STATUS_FAILED
        route = _as_map(attempt.get("route"))
        params = _as_map(attempt.get("params"))
        return InterimEntry(
            entry_id=make_entry_id(self.surface, kind, native),
            surface=self.surface, kind=kind,
            created_at=_iso(attempt.get("at")),
            status=status,
            parents=_clean_refs(list(_as_seq(attempt.get("parents")))
                                + [attempt.get("lock_digest"),
                                   attempt.get("spec_digest"),
                                   seg_id, run_entry_id]),
            produced_by={"step": f"generate:{attempt.get('capability') or kind}",
                         "model": attempt.get("model_id") or route.get("model_id"),
                         "seed": attempt.get("seed") or params.get("seed"),
                         "attempt": number,
                         "capability": attempt.get("capability"),
                         "params_digest": _digest(params)},
            artifact_refs=_clean_refs(artifacts),
            scorecard_ref=(f"{pointer}#attempts/{seg_id}/{number}/receipt"
                           if receipt else None),
            verdict=_receipt_verdict(receipt),
            gap=gap,
            registry_version=attempt.get("registry_version"),
            source_pointer=f"{pointer}#attempts/{seg_id}/{number}",
            aliases=_clean_refs(artifacts + [native]),
            label=f"{attempt.get('kind') or 'attempt'} {seg_id} #{number}",
        )


def _receipt_verdict(receipt: Mapping[str, Any]) -> str | None:
    """An oracle receipt's verdict — or None, which is NOT a pass.

    Receipts are not persisted in a store of their own anywhere in the fleet;
    they exist only embedded in the run that produced them. So the verdict is
    lifted here and the standalone receipt surface reports itself unavailable
    (see :class:`OracleReceiptSource`) rather than pretending to a store.
    """
    if not receipt:
        return None
    for key in ("verdict", "outcome", "class", "failure_class"):
        value = receipt.get(key)
        if value:
            return str(value)
    scorecard = _as_map(receipt.get("scorecard"))
    for key in ("verdict", "outcome", "score"):
        if scorecard.get(key) is not None:
            return str(scorecard[key])
    return None


def _stringify_gap(value: Any) -> str | None:
    if value in (None, "", {}, []):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        code = value.get("code") or value.get("gap_code")
        message = value.get("message") or value.get("reason") or value.get("detail")
        joined = ": ".join(str(p) for p in (code, message) if p)
        return joined or json.dumps(value, default=str)[:400]
    return str(value)[:400]


# --------------------------------------------------------------------------- #
# 3. performance runs
# --------------------------------------------------------------------------- #

#: ``performance.STAGES`` order; the last one landing is the only completion
#: signal the store offers.
PERFORMANCE_STAGES: tuple[str, ...] = ("snapshot", "audio", "lock", "segments",
                                       "keyframes", "clips", "assembly")
PERFORMANCE_FINAL_STAGE: str = PERFORMANCE_STAGES[-1]


class PerformanceSource(InterimSource):
    """``runs/performance/*/state.json`` — the k106 audio-first orchestrator.

    A performance run journals per-STAGE, each stage carrying its own digest and
    payload, so each stage is an interim object in its own right and is mapped
    as one. The stage's ``refs`` are its parent artifact paths.

    ``stages.keyframes.payload[<segment_id>].scorecard`` is the ONLY place in
    the fleet a ``Scorecard`` is persisted at all, so those are fanned out as
    per-segment entries — otherwise the ledger's ``scorecard_ref`` count would
    understate what actually got judged.
    """

    surface = SURFACE_PERFORMANCE

    def runs_dir(self) -> str:
        return os.path.join(self.root or default_run_root(), "runs", "performance")

    def probe(self) -> tuple[bool, str, str]:
        path = self.runs_dir()
        if not os.path.isdir(path):
            return False, f"source_unavailable: no run dir at {path}", path
        return True, "", path

    def _entries(self) -> Iterator[InterimEntry]:
        base = self.runs_dir()
        try:
            names = sorted(os.listdir(base))
        except OSError:
            return
        for name in names:
            if name.startswith("_") or name.startswith("."):
                continue
            pointer = os.path.join(base, name, "state.json")
            state = _read_json(pointer)
            if not isinstance(state, Mapping):
                continue
            yield from self._map_run(name, dict(state), pointer)

    def _map_run(self, run_id: str, state: Mapping[str, Any],
                 pointer: str) -> Iterator[InterimEntry]:
        run_id = str(state.get("run_id") or run_id)
        goal_digest = state.get("goal_digest")
        stages = _as_map(state.get("stages"))
        run_entry_id = make_entry_id(self.surface, "run", run_id)
        yield InterimEntry(
            entry_id=run_entry_id, surface=self.surface, kind="run",
            created_at=_iso(state.get("created_at") or state.get("updated_at")),
            # A performance run has NO terminal field — ``RunState.flush``
            # writes only {version, run_id, goal_digest, stages, updated_at}.
            # Completion is inferred from the final stage having landed, and a
            # run that stopped early reads as still running rather than done,
            # because "stopped" and "finished" are not the same fact and this
            # store cannot tell them apart.
            status=(STATUS_DONE if PERFORMANCE_FINAL_STAGE in stages
                    else STATUS_RUNNING),
            parents=(),
            produced_by={"step": "performance.run",
                         "stages": len(stages),
                         "updated_at": _iso(state.get("updated_at")),
                         "params_digest": str(goal_digest or "")[:16]},
            artifact_refs=(),
            source_pointer=pointer,
            aliases=_clean_refs([run_id, goal_digest]),
            label=f"performance run {run_id}",
        )
        previous: str | None = None
        for stage_name in [s for s in PERFORMANCE_STAGES if s in stages] + \
                [s for s in stages if s not in PERFORMANCE_STAGES]:
            stage = _as_map(stages.get(stage_name))
            digest = stage.get("digest") or stage.get("payload_digest")
            native = f"{run_id}#{stage_name}"
            stage_entry_id = make_entry_id(self.surface, f"stage:{stage_name}",
                                           native)
            payload = _as_map(stage.get("payload"))
            yield InterimEntry(
                entry_id=stage_entry_id,
                surface=self.surface, kind=f"stage:{stage_name}",
                created_at=_iso(stage.get("at")),
                status=STATUS_GAP if stage.get("gap") else STATUS_DONE,
                # The pipeline order IS the lineage: a stage depends on the one
                # before it, and ``drop_from`` invalidates every later stage on
                # a change, so the edge is real and not decorative.
                parents=_clean_refs(list(_as_seq(stage.get("refs")))
                                    + [previous, goal_digest, run_entry_id]),
                produced_by={"step": stage_name,
                             "model": stage.get("model_id")
                             or payload.get("model_id"),
                             "params_digest": str(digest or "")[:16]},
                artifact_refs=_clean_refs(_as_seq(stage.get("artifacts"))
                                          or _as_seq(stage.get("refs"))),
                gap=_stringify_gap(stage.get("gap")),
                registry_version=(stage.get("registry_version")
                                  or payload.get("registry_version")),
                source_pointer=f"{pointer}#stages/{stage_name}",
                aliases=_clean_refs([digest, native]),
                label=f"{stage_name} ({run_id})",
            )
            previous = stage_entry_id
            if stage_name in ("keyframes", "clips"):
                yield from self._map_scored_segments(
                    run_id, stage_name, stage_entry_id, payload, pointer)

    def _map_scored_segments(self, run_id: str, stage_name: str,
                             stage_entry_id: str, payload: Mapping[str, Any],
                             pointer: str) -> Iterator[InterimEntry]:
        for segment_id, raw in payload.items():
            item = _as_map(raw)
            if not item:
                continue
            scorecard = _as_map(item.get("scorecard"))
            ref = item.get("ref")
            native = f"{run_id}#{stage_name}#{segment_id}"
            yield InterimEntry(
                entry_id=make_entry_id(self.surface, f"{stage_name[:-1]}", native),
                surface=self.surface, kind=stage_name[:-1],
                created_at=None,
                status=STATUS_DONE if ref else STATUS_GAP,
                parents=_clean_refs([stage_entry_id, segment_id]),
                produced_by={"step": f"performance.{stage_name}",
                             "seed": item.get("seed"),
                             "candidates": item.get("candidates"),
                             "repaired": item.get("repaired"),
                             "params_digest": _digest(item.get("seed"))},
                artifact_refs=_clean_refs([ref]),
                scorecard_ref=(f"{pointer}#stages/{stage_name}/payload/"
                               f"{segment_id}/scorecard" if scorecard else None),
                verdict=_receipt_verdict(scorecard),
                gap=None if ref else "no artifact recorded for this segment",
                source_pointer=f"{pointer}#stages/{stage_name}/payload/{segment_id}",
                aliases=_clean_refs([native, ref]),
                label=f"{stage_name[:-1]} {segment_id}",
            )


# --------------------------------------------------------------------------- #
# 4. oracle receipts / scorecards
# --------------------------------------------------------------------------- #

class OracleReceiptSource(InterimSource):
    """Standalone oracle receipts + scorecards — which do not exist yet.

    Deliberately implemented as an ALWAYS-honest negative. ``oracle/scorecard.py``
    builds scorecards and ``oracle/runtime.py`` stamps receipts, but neither
    persists to a store of its own: a receipt survives only if the surface that
    asked for it embeds it (script-first does, in ``attempts[].receipt``;
    benchmark does, in each cell's ``judge``/``deterministic`` block). So there
    is nothing standalone to index, and the ledger says so instead of reporting
    a confident zero. The receipts that DO exist are surfaced by the adapter of
    the run that holds them, which is why the ledger's ``scorecard_ref`` counts
    are non-zero while this source is unavailable.

    Closing this is a write-side task, not a read-side one — see
    :func:`provenance_gaps`.
    """

    surface = SURFACE_ORACLE

    def probe(self) -> tuple[bool, str, str]:
        path = os.path.join(self.root or default_run_root(), "runs", "receipts")
        if os.path.isdir(path):
            return True, "", path
        return (False,
                "source_unavailable: oracle receipts/scorecards have no "
                "standalone store; they are persisted only INSIDE the run that "
                "produced them (script_first attempts[].receipt, benchmark "
                "cells[].judge). Those are indexed by their own adapters.",
                path)

    def _entries(self) -> Iterator[InterimEntry]:
        base = os.path.join(self.root or default_run_root(), "runs", "receipts")
        try:
            names = sorted(os.listdir(base))
        except OSError:
            return
        for name in names:
            if not name.endswith(".json"):
                continue
            pointer = os.path.join(base, name)
            payload = _read_json(pointer)
            if not isinstance(payload, Mapping):
                continue
            native = str(payload.get("receipt_id") or name[:-5])
            yield InterimEntry(
                entry_id=make_entry_id(self.surface, "receipt", native),
                surface=self.surface, kind="receipt",
                created_at=_iso(payload.get("at") or payload.get("created_at")),
                status=STATUS_DONE if payload.get("ok") else STATUS_FAILED,
                parents=_clean_refs(list(_as_seq(payload.get("parents")))
                                    + [payload.get("job_id"),
                                       payload.get("run_id")]),
                produced_by={"step": payload.get("operation") or "receipt",
                             "model": payload.get("model_id"),
                             "seed": payload.get("seed"),
                             "params_digest": _digest(payload.get("params"))},
                artifact_refs=_clean_refs(
                    _as_map(a).get("uri") for a in _as_seq(payload.get("artifacts"))),
                scorecard_ref=pointer,
                verdict=_receipt_verdict(payload),
                gap=_stringify_gap(payload.get("gap")),
                registry_version=payload.get("registry_version"),
                source_pointer=pointer,
                aliases=_clean_refs([native]),
                label=f"receipt {native[:16]}",
            )


# --------------------------------------------------------------------------- #
# 5. MCT artifact manifests
# --------------------------------------------------------------------------- #

class MctManifestSource(InterimSource):
    """``hugpy_agent.mct`` artifact manifests — the fleet's best provenance shape.

    ``ArtifactManifest`` is what every other surface should look like: typed
    ``parents`` (parent manifest digests), ``producer``, ``run_id``, ``model``,
    ``seed``, ``parameters``, ``evaluations``. It is also the one store this
    process cannot reach: ``hugpy_agent`` is a separate distribution that is not
    installed in the central venv, and ``ManifestStore`` is not a file scanner —
    it is a policy layer over a live ``(ObjectStore, Ledger)`` pair bound to an
    open session workspace.

    So the adapter probes for both and degrades honestly. Importing is tried, not
    assumed; if it succeeds and session workspaces exist, manifests are read
    through the PUBLIC ``ManifestStore``/``ArtifactManifest`` API only.
    """

    surface = SURFACE_MCT

    def __init__(self, root: str | None = None,
                 workspaces: str | None = None) -> None:
        super().__init__(root)
        self._workspaces = workspaces

    def workspaces_dir(self) -> str:
        return self._workspaces or os.path.join(os.path.expanduser("~"), ".mct")

    def probe(self) -> tuple[bool, str, str]:
        path = self.workspaces_dir()
        try:
            import hugpy_agent.mct.manifest as _manifest    # noqa: F401
        except Exception as exc:                    # noqa: BLE001
            return (False,
                    f"source_unavailable: hugpy_agent.mct is not importable from "
                    f"this environment ({type(exc).__name__}: {exc}). It is a "
                    f"separate distribution (py/hugpy_agent) and is not installed "
                    f"in the central venv; install it there to index MCT "
                    f"manifests.", path)
        if not os.path.isdir(path):
            return (False, f"source_unavailable: no MCT workspace root at {path}",
                    path)
        sessions = [n for n in _listdir(path) if n.startswith("session-")]
        if not sessions:
            return (False,
                    f"source_unavailable: hugpy_agent.mct imports, but {path} "
                    f"holds no session-* workspace to open a ManifestStore over",
                    path)
        return True, "", path

    def _entries(self) -> Iterator[InterimEntry]:
        from hugpy_agent.mct.manifest import ArtifactManifest    # noqa: PLC0415
        base = self.workspaces_dir()
        for session in sorted(_listdir(base)):
            if not session.startswith("session-"):
                continue
            manifest_dir = os.path.join(base, session, "manifests")
            for name in sorted(_listdir(manifest_dir)):
                if not name.endswith(".json"):
                    continue
                pointer = os.path.join(manifest_dir, name)
                payload = _read_json(pointer)
                if not isinstance(payload, Mapping):
                    continue
                entry = self._map_manifest(session, name[:-5], dict(payload),
                                           pointer, ArtifactManifest)
                if entry is not None:
                    yield entry

    def _map_manifest(self, session: str, digest: str,
                      payload: Mapping[str, Any], pointer: str,
                      manifest_cls: type) -> InterimEntry | None:
        try:                       # validate through the OWNING type, not ours
            manifest = manifest_cls(**{
                k: v for k, v in payload.items()
                if k in getattr(manifest_cls, "__dataclass_fields__", {})})
        except Exception as exc:                    # noqa: BLE001
            logger.debug("interim_ledger: unreadable MCT manifest %s (%s)",
                         pointer, exc)
            return None
        evaluations = tuple(getattr(manifest, "evaluations", ()) or ())
        content = getattr(manifest, "content_sha256", digest)
        return InterimEntry(
            entry_id=make_entry_id(self.surface,
                                   getattr(manifest, "artifact_type", "artifact"),
                                   str(content)),
            surface=self.surface,
            kind=str(getattr(manifest, "artifact_type", "artifact")),
            created_at=_iso(getattr(manifest, "created_at", None)
                            or getattr(manifest, "acquired_at", None)),
            status=STATUS_DONE,
            parents=_clean_refs(getattr(manifest, "parents", ()) or ()),
            produced_by={"step": getattr(manifest, "producer", None),
                         "model": getattr(manifest, "model", None),
                         "seed": getattr(manifest, "seed", None),
                         "run_id": getattr(manifest, "run_id", None),
                         "session": session,
                         "params_digest": _digest(
                             getattr(manifest, "parameters", ()))},
            artifact_refs=_clean_refs([getattr(manifest, "storage_pointer", "")
                                       or "", getattr(manifest, "source_uri", "")
                                       or ""]),
            scorecard_ref=evaluations[0] if evaluations else None,
            verdict=None,
            gap=None,
            registry_version=getattr(manifest, "graph_revision", None),
            source_pointer=pointer,
            aliases=_clean_refs([content, digest]),
            label=f"{getattr(manifest, 'artifact_type', 'artifact')} "
                  f"{str(content)[:12]}",
        )


def _listdir(path: str) -> list[str]:
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


# --------------------------------------------------------------------------- #
# 6. identity profiles + reconstructions + versions
# --------------------------------------------------------------------------- #

class IdentitySource(InterimSource):
    """Identity profiles, their reconstructions and their canonical versions.

    The one place in the fleet where a typed cross-surface parent pointer was
    ALREADY being written: every reconstruction carries the ``job_id`` of the
    media_bus job that produced it, and every version carries the ``recon_id``
    it was built from. That gives a real three-hop chain — version -> recon ->
    media_bus job — with no write-side change, and it is the template the other
    surfaces should copy.

    Read through the module's public ``list_profiles()``.
    """

    surface = SURFACE_IDENTITY

    def __init__(self, root: str | None = None,
                 loader: Callable[[], list[dict[str, Any]]] | None = None) -> None:
        super().__init__(root)
        self._loader = loader

    def _load(self) -> list[dict[str, Any]]:
        if self._loader is not None:
            return list(self._loader())
        from hugpy_video.intel import identity_profiles
        return list(identity_profiles.list_profiles())

    def probe(self) -> tuple[bool, str, str]:
        if self._loader is not None:
            return True, "", "<injected>"
        try:
            from hugpy_video.intel import identity_profiles
        except Exception as exc:                    # noqa: BLE001
            return (False, f"source_unavailable: identity_profiles will not "
                           f"import ({type(exc).__name__}: {exc})", "")
        pointer = getattr(identity_profiles, "__file__", "identity_profiles")
        return True, "", str(pointer)

    def _entries(self) -> Iterator[InterimEntry]:
        for raw in self._load():
            profile = _as_map(raw)
            slug = str(profile.get("slug") or "").strip()
            if not slug:
                continue
            yield from self._map_profile(slug, profile)

    def _map_profile(self, slug: str,
                     profile: Mapping[str, Any]) -> Iterator[InterimEntry]:
        refs = _clean_refs(_as_seq(profile.get("reference_images")))
        authorized = bool(_as_map(profile.get("authorization")))
        profile_entry_id = make_entry_id(self.surface, "profile", slug)
        yield InterimEntry(
            entry_id=profile_entry_id, surface=self.surface, kind="profile",
            created_at=_iso(profile.get("created_at")),
            status=STATUS_DONE,
            parents=(),
            produced_by={"step": "identity.profile",
                         "name": profile.get("name"),
                         "authorized": authorized,
                         "params_digest": _digest(profile.get("gen_settings"))},
            artifact_refs=refs,
            # An identity with no likeness/voice authorization is not a pass —
            # it is an unresolved rights gap, and it reads as one.
            gap=None if authorized else "no likeness/voice authorization recorded",
            source_pointer=f"identity_profiles://{slug}",
            aliases=_clean_refs([slug, profile.get("name")] + list(refs)),
            label=f"identity {profile.get('name') or slug}",
        )

        for raw in _as_seq(profile.get("reconstructions")):
            recon = _as_map(raw)
            recon_id = str(recon.get("recon_id") or "").strip()
            if not recon_id:
                continue
            views = _clean_refs(_as_seq(recon.get("views")))
            job_id = recon.get("job_id")
            mesh = _as_map(recon.get("mesh"))
            # TWO distinct bus jobs can be involved: the turnaround/extract job
            # that produced the views, and the later mesh-build job. Both are
            # parents; collapsing them would hide which one failed.
            mesh_job = mesh.get("job_id")
            mesh_status = str(mesh.get("status") or "").lower()
            mesh_error = mesh.get("error")
            yield InterimEntry(
                entry_id=make_entry_id(self.surface, "reconstruction", recon_id),
                surface=self.surface, kind="reconstruction",
                created_at=_iso(recon.get("created_at")),
                status=(STATUS_FAILED if mesh_status == "error"
                        else STATUS_RUNNING if mesh_status in ("queued", "running")
                        else STATUS_DONE),
                # job_id FIRST: the cross-surface edge is the point.
                parents=_clean_refs([job_id, mesh_job, slug, profile_entry_id]),
                produced_by={"step": f"identity.{recon.get('mode') or 'recon'}",
                             "mode": recon.get("mode"),
                             "char": recon.get("char"),
                             "frame_count": recon.get("frame_count"),
                             "job_id": job_id,
                             "mesh_job_id": mesh_job,
                             "mesh_status": mesh.get("status"),
                             "params_digest": _digest(
                                 {"deg": recon.get("degrees_per_frame"),
                                  "frames": recon.get("frame_count")})},
                artifact_refs=_clean_refs(list(views) + [mesh.get("glb_path"),
                                                         mesh.get("video_path")]),
                gap=_stringify_gap(mesh_error),
                source_pointer=f"identity_profiles://{slug}#reconstructions/{recon_id}",
                aliases=_clean_refs([recon_id] + list(views)),
                label=f"reconstruction {recon.get('char') or recon_id[:16]}",
            )

        for raw in _as_seq(profile.get("versions")):
            version = _as_map(raw)
            version_id = str(version.get("version_id") or "").strip()
            if not version_id:
                continue
            canonical = _clean_refs(_as_seq(version.get("canonical")))
            yield InterimEntry(
                entry_id=make_entry_id(self.surface, "version", version_id),
                surface=self.surface, kind="version",
                created_at=_iso(version.get("created_at")),
                status=STATUS_DONE,
                parents=_clean_refs([version.get("recon_id"), slug,
                                     profile_entry_id]),
                produced_by={"step": f"identity.version:{version.get('kind')}",
                             "kind": version.get("kind"),
                             "recon_id": version.get("recon_id"),
                             "params_digest": _digest(
                                 version.get("canonical_angles"))},
                artifact_refs=canonical,
                source_pointer=f"identity_profiles://{slug}#versions/{version_id}",
                aliases=_clean_refs([version_id] + list(canonical)),
                label=f"version {version.get('name') or version_id[:12]}",
            )


# --------------------------------------------------------------------------- #
# 7. discovery dossiers (k120)
# --------------------------------------------------------------------------- #

class DiscoveryDossierSource(InterimSource):
    """k120's dossier store — read through the :class:`DossierSource`
    protocol (``hugpy_oracle.providers``), never the store's files.

    The dossier store belongs to ``hugpy_curation``, which depends on the
    oracle, so the oracle cannot import it. Curation installs its
    implementation with ``providers.set_dossier_source`` at composition time;
    until then the default source has no root and yields nothing, which this
    adapter reports as ``source_unavailable``, not zero.
    """

    surface = SURFACE_DISCOVERY

    def __init__(self, root: str | None = None,
                 source: Any = None) -> None:
        super().__init__(root)
        self._source = source

    def source(self):
        if self._source is not None:
            return self._source
        from hugpy_oracle.providers import get_dossier_source
        return get_dossier_source()

    def store_dir(self) -> str:
        try:
            return str(self.source().root() or "")
        except Exception as exc:                    # noqa: BLE001
            logger.debug("interim_ledger: discovery store unavailable (%s)", exc)
            return ""

    def probe(self) -> tuple[bool, str, str]:
        path = self.store_dir()
        if not path:
            return (False, "source_unavailable: no dossier source installed "
                           "(hugpy_curation has not registered its store)", "")
        return True, "", path

    def _entries(self) -> Iterator[InterimEntry]:
        try:
            records = list(self.source().iter_dossiers())
        except Exception as exc:                    # noqa: BLE001
            logger.debug("interim_ledger: dossier source failed (%s)", exc)
            return
        for record in records:
            if not isinstance(record, Mapping):
                continue
            payload = record.get("payload")
            if not isinstance(payload, Mapping):
                continue
            criteria = str(record.get("criteria") or "")
            hub_id = str(record.get("hub_id") or "")
            pointer = str(record.get("path") or f"{criteria}/{hub_id}")
            entry = self._map_dossier(criteria, hub_id, dict(payload), pointer)
            if entry is not None:
                yield entry

    def _map_dossier(self, criteria: str, hub_id: str,
                     payload: Mapping[str, Any], pointer: str) -> InterimEntry:
        verdict = payload.get("verdict") or _as_map(payload.get("screening")).get("verdict")
        gap = payload.get("gap") or payload.get("refusal")
        native = f"{criteria}/{hub_id}"
        return InterimEntry(
            entry_id=make_entry_id(self.surface, "dossier", native),
            surface=self.surface, kind="dossier",
            created_at=_iso(payload.get("created_at") or payload.get("at")
                            or payload.get("built_at")),
            status=STATUS_GAP if gap else STATUS_DONE,
            parents=_clean_refs([payload.get("hub_id"), criteria,
                                 payload.get("model_id")]),
            produced_by={"step": "discovery.dossier",
                         "criteria": criteria,
                         "model": payload.get("model_id") or payload.get("hub_id"),
                         "params_digest": _digest(payload.get("weights"))},
            artifact_refs=(pointer,),
            scorecard_ref=pointer if verdict else None,
            verdict=str(verdict) if verdict else None,
            gap=_stringify_gap(gap),
            registry_version=payload.get("registry_version"),
            source_pointer=pointer,
            aliases=_clean_refs([native, payload.get("hub_id"),
                                 payload.get("model_id")]),
            label=f"dossier {hub_id}",
        )


# --------------------------------------------------------------------------- #
# 8. benchmark run dirs (k109/k109b)
# --------------------------------------------------------------------------- #

class BenchmarkSource(InterimSource):
    """``model-battery/*`` run dirs — the run + every cell in ``cells.jsonl``.

    A benchmark CELL is the single richest interim record the fleet writes: it
    already carries ``registry_version``, ``gap_code``, a judge block that can
    say ``unavailable`` without saying ``fail``, and a ``scenario_digest`` that
    ties it to its run. Mapped one entry per cell, parented on the run and the
    scenario digest.

    The battery root is read from ``benchmark.DEFAULT_RUN_ROOT`` (k109b's
    module, imported read-only) so the two cannot drift.
    """

    surface = SURFACE_BENCHMARK

    def __init__(self, root: str | None = None,
                 battery_root: str | None = None,
                 max_runs: int = 40) -> None:
        super().__init__(root)
        self._battery_root = battery_root
        self.max_runs = int(max_runs)

    def battery_root(self) -> str:
        if self._battery_root:
            return self._battery_root
        from hugpy_oracle.config import benchmark_root
        return benchmark_root()

    def probe(self) -> tuple[bool, str, str]:
        path = self.battery_root()
        if not os.path.isdir(path):
            return (False, f"source_unavailable: no model-battery root at {path}",
                    path)
        return True, "", path

    def _entries(self) -> Iterator[InterimEntry]:
        base = self.battery_root()
        names = [n for n in _listdir(base) if os.path.isdir(os.path.join(base, n))]
        # TWO namespaces share this root. ``oracle-<stamp>[-label]`` dirs are
        # k109/k109b oracle runs and carry a cell journal; bare
        # ``YYYYmmdd-HHMM`` dirs are the older image battery and never do.
        # Indexing both matters — a ``studio_tester`` job's ``battery_dir``
        # points at an IMAGE battery dir, so that is a real bus edge — but
        # judging both by the same rule would mark ~105 image dirs "gap: no
        # cells.jsonl", which would be a lie about a directory that was never
        # supposed to have one.
        names.sort(reverse=True)
        oracle = [n for n in names if n.startswith("oracle-")]
        others = [n for n in names if not n.startswith("oracle-")]
        for name in oracle[:self.max_runs]:
            yield from self._map_run(os.path.join(base, name), name, True)
        for name in others[:self.max_runs]:
            yield from self._map_run(os.path.join(base, name), name, False)

    def _map_run(self, run_dir: str, run_id: str,
                 is_oracle: bool) -> Iterator[InterimEntry]:
        environment = _as_map(_read_json(os.path.join(run_dir, "environment.json")))
        run_entry_id = make_entry_id(self.surface, "run", run_id)
        scenario_digest = environment.get("scenario_digest")
        registry_version = environment.get("registry_version")

        # k109b writes cells.jsonl; the k109 pilot wrote attempts.jsonl. Both
        # are cell journals of the same family, so both are read.
        cells_path = ""
        for candidate in ("cells.jsonl", "attempts.jsonl"):
            path = os.path.join(run_dir, candidate)
            if os.path.exists(path):
                cells_path = path
                break

        gap = None
        if is_oracle and not cells_path:
            gap = "oracle run dir has no cells.jsonl or attempts.jsonl"
        yield InterimEntry(
            entry_id=run_entry_id, surface=self.surface,
            kind="run" if is_oracle else "image_battery",
            created_at=_iso(environment.get("started_at")
                            or _mtime(os.path.join(run_dir, "environment.json"))
                            or _mtime(run_dir)),
            status=STATUS_GAP if gap else STATUS_DONE,
            parents=(),
            produced_by={"step": "oracle.benchmark" if is_oracle
                         else "model_battery",
                         "wave": environment.get("wave"),
                         "host": environment.get("host"),
                         "journal": os.path.basename(cells_path) or None,
                         "scenario_version": environment.get("scenario_version"),
                         "params_digest": str(scenario_digest or "")[:24]},
            artifact_refs=(run_dir,),
            gap=gap,
            registry_version=registry_version,
            source_pointer=run_dir,
            aliases=_clean_refs([run_id, run_dir, scenario_digest]),
            label=f"battery {run_id}",
        )
        if not cells_path:
            return
        for index, cell in enumerate(_read_jsonl(cells_path)):
            yield self._map_cell(run_entry_id, run_id, index, cell,
                                 cells_path, scenario_digest)

    def _map_cell(self, run_entry_id: str, run_id: str, index: int,
                  cell: Mapping[str, Any], pointer: str,
                  scenario_digest: Any) -> InterimEntry:
        point_id = str(cell.get("point_id") or cell.get("case_id") or index)
        model = str(cell.get("model") or "")
        native = f"{run_id}#{point_id}#{cell.get('repeat', 0)}#{index}"
        judge = _as_map(cell.get("judge"))
        gap_code = cell.get("gap_code")
        failure = cell.get("failure")

        if gap_code:
            status = STATUS_GAP
        elif cell.get("ok"):
            status = STATUS_DONE
        else:
            status = STATUS_FAILED

        # A judge that was UNAVAILABLE has no verdict. Recording "unavailable"
        # as a verdict would make an unjudged cell count as scored, which is
        # precisely the failure mode clause 3 names.
        verdict = None
        if judge.get("refused"):
            verdict = "refused"
        elif judge.get("available") and judge.get("verdict"):
            verdict = str(judge["verdict"])
        elif cell.get("verdict") and str(cell["verdict"]) != "NO_CANDIDATES":
            verdict = str(cell["verdict"])

        gap = None
        if gap_code:
            gap = f"{gap_code}: {failure}" if failure else str(gap_code)
        elif failure:
            gap = str(failure)

        artifacts = _clean_refs([cell.get("artifact_ref"), cell.get("raw_ref")])
        return InterimEntry(
            entry_id=make_entry_id(self.surface, "cell", native),
            surface=self.surface, kind="cell",
            created_at=_iso(cell.get("started_at") or cell.get("ended_at")),
            status=status,
            parents=_clean_refs([run_entry_id, scenario_digest,
                                 cell.get("scenario_digest")]),
            produced_by={"step": f"{cell.get('stage') or 'cell'}:{cell.get('operation') or point_id}",
                         "model": model or None,
                         "mode": cell.get("mode"),
                         "track": cell.get("track"),
                         "step_number": cell.get("step"),
                         "repeat": cell.get("repeat"),
                         "params_digest": _digest(cell.get("perf"))},
            artifact_refs=artifacts,
            scorecard_ref=(f"{pointer}#{index}"
                           if (judge.get("available") or cell.get("deterministic"))
                           else None),
            verdict=verdict,
            gap=gap,
            registry_version=cell.get("registry_version"),
            source_pointer=f"{pointer}#{index}",
            aliases=_clean_refs([native] + list(artifacts)),
            label=f"{point_id} / {model or 'no model'}",
        )


def _read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                if isinstance(payload, Mapping):
                    yield dict(payload)
    except OSError as exc:
        logger.debug("interim_ledger: unreadable jsonl %s (%s)", path, exc)


def _mtime(path: str) -> float | None:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# 9. coordination reports (k121)
# --------------------------------------------------------------------------- #

class CoordinationSource(InterimSource):
    """k121's prompt-coordination reports — the knobs-vs-words decision record.

    k121 is in flight and ``video_intel/prompt_coordination.py`` does not exist
    on this box yet. The adapter is written against the attachment point it will
    land on (a module exposing ``list_reports()`` or a ``reports/`` dir under the
    run root) and reports ``source_unavailable`` with the exact reason until
    then. Deliberately NOT a stub that returns [] — a silent zero here would
    read as "coordination made no decisions today".
    """

    surface = SURFACE_COORDINATION

    def __init__(self, root: str | None = None,
                 reports_dir: str | None = None) -> None:
        super().__init__(root)
        self._reports_dir = reports_dir

    def reports_dir(self) -> str:
        if self._reports_dir:
            return self._reports_dir
        return os.path.join(self.root or default_run_root(), "runs",
                            "coordination")

    def probe(self) -> tuple[bool, str, str]:
        path = self.reports_dir()
        if os.path.isdir(path):
            return True, "", path
        try:
            from hugpy_oracle.relay import prompt_coordination
        except Exception as exc:                    # noqa: BLE001
            return (False,
                    f"source_unavailable: no coordination reports at {path} and "
                    f"video_intel.prompt_coordination is not present yet "
                    f"({type(exc).__name__}) — k121's attachment point.", path)
        return (False,
                f"source_unavailable: prompt_coordination imports but no report "
                f"store exists at {path}", path)

    def _entries(self) -> Iterator[InterimEntry]:
        base = self.reports_dir()
        for name in _listdir(base):
            if not name.endswith(".json"):
                continue
            pointer = os.path.join(base, name)
            payload = _read_json(pointer)
            if not isinstance(payload, Mapping):
                continue
            native = str(payload.get("report_id") or name[:-5])
            decisions = _as_seq(payload.get("decisions"))
            yield InterimEntry(
                entry_id=make_entry_id(self.surface, "report", native),
                surface=self.surface, kind="report",
                created_at=_iso(payload.get("at") or payload.get("created_at")),
                status=STATUS_DONE,
                parents=_clean_refs([payload.get("run_id"), payload.get("job_id"),
                                     payload.get("segment_id")]),
                produced_by={"step": "prompt_coordination",
                             "model": payload.get("model_id"),
                             "decisions": len(decisions),
                             "params_digest": _digest(decisions)},
                artifact_refs=(pointer,),
                verdict=payload.get("verdict"),
                gap=_stringify_gap(payload.get("gap")),
                registry_version=payload.get("registry_version"),
                source_pointer=pointer,
                aliases=_clean_refs([native]),
                label=f"coordination {native[:16]}",
            )


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

#: Every adapter, in scan order. A source is added HERE and nowhere else.
SOURCE_TYPES: tuple[type[InterimSource], ...] = (
    MediaBusSource,
    ScriptFirstSource,
    PerformanceSource,
    OracleReceiptSource,
    MctManifestSource,
    IdentitySource,
    DiscoveryDossierSource,
    BenchmarkSource,
    CoordinationSource,
)

SURFACES: tuple[str, ...] = tuple(s.surface for s in SOURCE_TYPES)


def default_sources(root: str | None = None) -> list[InterimSource]:
    return [cls(root) for cls in SOURCE_TYPES]


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #

class InterimLedger:
    """The joined, queryable view. Built once, queried many times.

    Construction resolves every raw parent ref against the alias index, so a
    caller sees ``resolved_parents`` (real ``entry_id``s) and
    ``unresolved_parents`` (refs no source claims) as two separate facts.
    """

    def __init__(self, entries: Sequence[InterimEntry],
                 reports: Sequence[SourceReport] = (),
                 built_at: str | None = None) -> None:
        self.built_at = built_at or _utc_now()
        self.reports: tuple[SourceReport, ...] = tuple(reports)
        self._by_id: dict[str, InterimEntry] = {}
        for entry in entries:
            # First writer wins: a duplicate entry_id is a source bug, not a
            # reason to lose the original.
            self._by_id.setdefault(entry.entry_id, entry)

        # -- alias index ----------------------------------------------------
        self._alias: dict[str, str] = {}
        for entry in self._by_id.values():
            self._alias.setdefault(entry.entry_id, entry.entry_id)
        for entry in self._by_id.values():
            for alias in entry.aliases:
                self._alias.setdefault(alias, entry.entry_id)
        # Artifacts resolve to their producer only if nothing already claims the
        # ref as an identity of its own.
        for entry in self._by_id.values():
            for ref in entry.artifact_refs:
                self._alias.setdefault(ref, entry.entry_id)

        # -- edges ----------------------------------------------------------
        self._parents: dict[str, tuple[str, ...]] = {}
        self._unresolved: dict[str, tuple[str, ...]] = {}
        self._children: dict[str, list[str]] = {}
        for entry in self._by_id.values():
            resolved: list[str] = []
            dangling: list[str] = []
            for ref in entry.parents:
                target = self._alias.get(ref)
                if target and target != entry.entry_id:
                    resolved.append(target)
                elif target is None:
                    dangling.append(ref)
            self._parents[entry.entry_id] = _clean_refs(resolved)
            self._unresolved[entry.entry_id] = _clean_refs(dangling)
        for child, parents in self._parents.items():
            for parent in parents:
                self._children.setdefault(parent, []).append(child)

    # -- basics -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._by_id)

    @property
    def entries(self) -> tuple[InterimEntry, ...]:
        return tuple(self._by_id.values())

    def get(self, entry_id: str) -> InterimEntry | None:
        return self._by_id.get(entry_id)

    def resolve(self, ref: str) -> str | None:
        """Any dialect of id -> ``entry_id``, or None. Never guesses."""
        if not ref:
            return None
        ref = str(ref).strip()
        return self._alias.get(ref)

    def parents_of(self, entry_id: str) -> tuple[str, ...]:
        return self._parents.get(entry_id, ())

    def children_of(self, entry_id: str) -> tuple[str, ...]:
        return tuple(self._children.get(entry_id, ()))

    def unresolved_parents_of(self, entry_id: str) -> tuple[str, ...]:
        return self._unresolved.get(entry_id, ())

    # -- query --------------------------------------------------------------
    def query(self, *, surface: str | None = None, kind: str | None = None,
              since: str | None = None, until: str | None = None,
              status: str | None = None, model: str | None = None,
              has_gap: bool | None = None, scored: bool | None = None,
              terminal: bool | None = None, q: str | None = None,
              limit: int = 200, offset: int = 0) -> list[InterimEntry]:
        """Filtered, newest-first. Unknown ``created_at`` sorts LAST, not first."""
        since_iso = _iso(since) if since else None
        until_iso = _iso(until) if until else None
        needle = (q or "").strip().lower()
        model_needle = (model or "").strip().lower()

        out: list[InterimEntry] = []
        for entry in self._by_id.values():
            if surface and entry.surface != surface:
                continue
            if kind and not (entry.kind == kind or entry.kind.startswith(kind + ":")):
                continue
            if status and entry.status != status:
                continue
            if terminal is not None and entry.terminal is not terminal:
                continue
            if has_gap is not None and entry.has_gap is not has_gap:
                continue
            if scored is not None and entry.scored is not scored:
                continue
            if since_iso and (entry.created_at is None
                              or entry.created_at < since_iso):
                continue
            if until_iso and (entry.created_at is None
                              or entry.created_at > until_iso):
                continue
            if model_needle:
                produced_model = str(entry.produced_by.get("model") or "").lower()
                if model_needle not in produced_model:
                    continue
            if needle and needle not in self._haystack(entry):
                continue
            out.append(entry)

        out.sort(key=lambda e: (e.created_at is not None, e.created_at or "",
                                e.entry_id), reverse=True)
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 2000))
        return out[offset:offset + limit]

    @staticmethod
    def _haystack(entry: InterimEntry) -> str:
        return " ".join((entry.entry_id, entry.label, entry.kind,
                         entry.gap or "", entry.verdict or "",
                         str(entry.produced_by.get("model") or ""))).lower()

    def count(self, **filters: Any) -> int:
        filters.setdefault("limit", 2000)
        return len(self.query(**filters))

    # -- the family tree ----------------------------------------------------
    def tree(self, ref: str, *, up: int = 6, down: int = 6,
             max_nodes: int = 400) -> dict[str, Any]:
        """Walk parents AND children across surfaces from any ref.

        Cycle-safe by construction: every node is expanded at most once per
        direction, and the frontier is bounded by ``max_nodes``. A store that
        records a ref loop (or an entry that lists itself) yields a finite tree
        with ``truncated`` telling the truth about why it stopped.
        """
        entry_id = self.resolve(ref)
        if entry_id is None:
            raise LedgerRefused(
                "REF_UNKNOWN",
                f"no interim entry answers to {ref!r} in this ledger",
                detail={"ref": str(ref), "entries": len(self._by_id)},
                http_status=404)

        nodes: dict[str, InterimEntry] = {}
        edges: set[tuple[str, str]] = set()
        truncated = False

        def walk(direction: str, depth_limit: int) -> None:
            nonlocal truncated
            seen = {entry_id}
            frontier = [(entry_id, 0)]
            while frontier:
                current, depth = frontier.pop(0)
                node = self._by_id.get(current)
                if node is not None:
                    nodes.setdefault(current, node)
                if depth >= depth_limit:
                    continue
                neighbours = (self.parents_of(current) if direction == "up"
                              else self.children_of(current))
                for other in neighbours:
                    edge = ((other, current) if direction == "up"
                            else (current, other))
                    edges.add(edge)
                    if other in seen:
                        continue          # cycle or diamond: edge kept, no re-walk
                    if len(nodes) + len(frontier) >= max_nodes:
                        truncated = True
                        continue
                    seen.add(other)
                    frontier.append((other, depth + 1))

        walk("up", max(0, int(up)))
        walk("down", max(0, int(down)))

        root = self._by_id[entry_id]
        surfaces = sorted({n.surface for n in nodes.values()})
        unresolved = {nid: list(self.unresolved_parents_of(nid))
                      for nid in nodes if self.unresolved_parents_of(nid)}
        return {
            "ok": True,
            "ref": str(ref),
            "root": entry_id,
            "root_entry": root.to_dict(),
            "nodes": {nid: node.to_dict() for nid, node in nodes.items()},
            "edges": [list(e) for e in sorted(edges)],
            "surfaces": surfaces,
            "cross_surface": len(surfaces) > 1,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "truncated": truncated,
            # THE follow-up map: refs a source recorded that nothing claims.
            "unresolved_parents": unresolved,
            "built_at": self.built_at,
        }

    # -- the honesty dashboard ----------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Per-surface counts, gaps, unscored — and what could not be read.

        ``unscored`` counts TERMINAL entries with neither a verdict nor a
        scorecard. That is the number the principle actually cares about: how
        much of today's production finished without anything judging it.
        """
        by_surface: dict[str, dict[str, Any]] = {}
        for surface in SURFACES:
            by_surface[surface] = {
                "surface": surface, "count": 0, "gaps": 0, "unscored": 0,
                "scored": 0, "terminal": 0, "kinds": {}, "statuses": {},
                "available": False, "reason": "", "errors": [],
                "with_parents": 0, "with_resolved_parents": 0,
                "unresolved_parent_refs": 0,
            }
        for report in self.reports:
            bucket = by_surface.setdefault(report.surface, {
                "surface": report.surface, "count": 0, "gaps": 0, "unscored": 0,
                "scored": 0, "terminal": 0, "kinds": {}, "statuses": {},
                "with_parents": 0, "with_resolved_parents": 0,
                "unresolved_parent_refs": 0,
            })
            bucket["available"] = report.available
            bucket["reason"] = report.reason
            bucket["errors"] = list(report.errors)
            bucket["pointer"] = report.scanned_pointer

        totals = {"entries": len(self._by_id), "gaps": 0, "unscored": 0,
                  "scored": 0, "terminal": 0, "with_parents": 0,
                  "with_resolved_parents": 0, "unresolved_parent_refs": 0,
                  "surfaces_available": 0, "surfaces_unavailable": 0}

        for entry in self._by_id.values():
            bucket = by_surface.setdefault(entry.surface, {
                "surface": entry.surface, "count": 0, "gaps": 0, "unscored": 0,
                "scored": 0, "terminal": 0, "kinds": {}, "statuses": {},
                "available": True, "reason": "", "errors": [],
                "with_parents": 0, "with_resolved_parents": 0,
                "unresolved_parent_refs": 0,
            })
            bucket["count"] += 1
            bucket["kinds"][entry.kind] = bucket["kinds"].get(entry.kind, 0) + 1
            bucket["statuses"][entry.status] = (
                bucket["statuses"].get(entry.status, 0) + 1)
            if entry.has_gap:
                bucket["gaps"] += 1
                totals["gaps"] += 1
            if entry.terminal:
                bucket["terminal"] += 1
                totals["terminal"] += 1
                if entry.scored:
                    bucket["scored"] += 1
                    totals["scored"] += 1
                else:
                    bucket["unscored"] += 1
                    totals["unscored"] += 1
            if entry.parents:
                bucket["with_parents"] += 1
                totals["with_parents"] += 1
            resolved = self.parents_of(entry.entry_id)
            if resolved:
                bucket["with_resolved_parents"] += 1
                totals["with_resolved_parents"] += 1
            dangling = len(self.unresolved_parents_of(entry.entry_id))
            bucket["unresolved_parent_refs"] += dangling
            totals["unresolved_parent_refs"] += dangling

        for bucket in by_surface.values():
            if bucket.get("available"):
                totals["surfaces_available"] += 1
            else:
                totals["surfaces_unavailable"] += 1

        return {
            "ok": True,
            "built_at": self.built_at,
            "ledger_version": LEDGER_VERSION,
            "totals": totals,
            "surfaces": [by_surface[s] for s in
                         sorted(by_surface, key=lambda s: (
                             -by_surface[s]["count"], s))],
            "sources": [r.to_dict() for r in self.reports],
            "provenance_gaps": provenance_gaps(self),
        }

    # -- serialisation ------------------------------------------------------
    def to_cache(self) -> dict[str, Any]:
        return {"ledger_version": LEDGER_VERSION, "built_at": self.built_at,
                "entries": [e.to_dict() for e in self._by_id.values()],
                "reports": [r.to_dict() for r in self.reports]}

    @classmethod
    def from_cache(cls, payload: Mapping[str, Any]) -> "InterimLedger | None":
        if str(payload.get("ledger_version")) != LEDGER_VERSION:
            return None
        try:
            entries = [InterimEntry.from_dict(e)
                       for e in _as_seq(payload.get("entries"))]
            reports = [SourceReport(**{k: (tuple(v) if k == "errors" else v)
                                       for k, v in _as_map(r).items()
                                       if k in SourceReport.__dataclass_fields__})
                       for r in _as_seq(payload.get("reports"))]
        except (TypeError, ValueError) as exc:
            logger.warning("interim_ledger: cache discarded (%s)", exc)
            return None
        return cls(entries, reports, str(payload.get("built_at") or ""))


# --------------------------------------------------------------------------- #
# The write-side gap map
# --------------------------------------------------------------------------- #

def provenance_gaps(ledger: "InterimLedger") -> list[dict[str, Any]]:
    """Which surfaces do not record parents — the next mandate's worklist.

    Derived from the built ledger rather than asserted, so it cannot go stale:
    a surface that starts writing parent refs shows up here as fixed on the very
    next scan, with no edit to this function.
    """
    by_surface: dict[str, dict[str, Any]] = {}
    for entry in ledger.entries:
        row = by_surface.setdefault(entry.surface, {
            "surface": entry.surface, "entries": 0, "with_parents": 0,
            "with_resolved_parents": 0, "unresolved_refs": 0,
            "kinds_without_parents": {}})
        row["entries"] += 1
        if entry.parents:
            row["with_parents"] += 1
        else:
            row["kinds_without_parents"][entry.kind] = (
                row["kinds_without_parents"].get(entry.kind, 0) + 1)
        if ledger.parents_of(entry.entry_id):
            row["with_resolved_parents"] += 1
        row["unresolved_refs"] += len(ledger.unresolved_parents_of(entry.entry_id))

    out = []
    for row in by_surface.values():
        total = max(1, row["entries"])
        row["parent_coverage"] = round(row["with_parents"] / total, 4)
        row["resolved_coverage"] = round(row["with_resolved_parents"] / total, 4)
        row["kinds_without_parents"] = dict(sorted(
            row["kinds_without_parents"].items(), key=lambda kv: -kv[1])[:12])
        out.append(row)
    out.sort(key=lambda r: (r["resolved_coverage"], -r["entries"]))
    return out


# --------------------------------------------------------------------------- #
# Build / cache
# --------------------------------------------------------------------------- #

def build_ledger(sources: Sequence[InterimSource] | None = None,
                 root: str | None = None) -> InterimLedger:
    """Scan every source once. Never raises on a source — that is the point."""
    sources = list(sources if sources is not None else default_sources(root))
    entries: list[InterimEntry] = []
    reports: list[SourceReport] = []
    for source in sources:
        started = time.time()
        found, report = source.collect()
        entries.extend(found)
        reports.append(report)
        logger.debug("interim_ledger: %s -> %d entries in %.2fs (%s)",
                     source.surface, len(found), time.time() - started,
                     "ok" if report.available else report.reason)
    return InterimLedger(entries, reports)


def save_cache(ledger: InterimLedger, root: str | None = None) -> str | None:
    """Write the rebuildable index. Failure is logged, never raised.

    The ONLY write this module performs, and it lands under
    ``<run_root>/runs/interim_ledger/`` — never inside a source's store.
    """
    path = cache_path(root)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(ledger.to_cache(), handle, default=str)
        os.replace(tmp, path)
        return path
    except OSError as exc:
        logger.warning("interim_ledger: cache not written (%s)", exc)
        return None


def load_cache(root: str | None = None,
               max_age_s: float = DEFAULT_MAX_AGE_S) -> InterimLedger | None:
    path = cache_path(root)
    try:
        age = time.time() - os.stat(path).st_mtime
    except OSError:
        return None
    if age > max_age_s:
        return None
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        return None
    return InterimLedger.from_cache(payload)


def load_ledger(root: str | None = None, *, rebuild: bool = False,
                max_age_s: float = DEFAULT_MAX_AGE_S,
                sources: Sequence[InterimSource] | None = None,
                use_cache: bool = True) -> InterimLedger:
    """The one entry point the routes use: cached if fresh, rebuilt if not."""
    if use_cache and not rebuild:
        cached = load_cache(root, max_age_s)
        if cached is not None:
            return cached
    ledger = build_ledger(sources, root)
    if use_cache:
        save_cache(ledger, root)
    return ledger


__all__ = [
    "InterimEntry", "InterimLedger", "InterimSource", "SourceReport",
    "LedgerRefused", "LEDGER_VERSION", "STATUSES", "SURFACES", "SOURCE_TYPES",
    "MediaBusSource", "ScriptFirstSource", "PerformanceSource",
    "OracleReceiptSource", "MctManifestSource", "IdentitySource",
    "DiscoveryDossierSource", "BenchmarkSource", "CoordinationSource",
    "build_ledger", "load_ledger", "save_cache", "load_cache",
    "default_sources", "default_run_root", "cache_path", "cache_dir",
    "make_entry_id", "provenance_gaps",
]
