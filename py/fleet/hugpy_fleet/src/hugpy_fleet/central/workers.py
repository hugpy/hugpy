"""GPU worker registry.

A *worker* is a remote box that runs the standalone worker agent
(``abstract_hugpy_dev.worker_agent``), exposes an HTTP inference endpoint, and
joins this central node so its GPU(s) can serve one or more models from the
manifest.

This module is the single source of truth for the pool. It owns:

    - persistence of the worker list to a JSON file beside the model manifest
      (so the pool survives restarts),
    - registration / heartbeat / removal,
    - model assignment (which worker may serve which model_key),
    - liveness (a worker is ``online`` only if it has heartbeat-ed recently),
    - selection (pick an online worker that is assigned + ready for a model).

Routing (chat/streaming) and the ``/llm/workers`` routes are dumb consumers of
the functions exported here.
"""
from __future__ import annotations

import os
import re
import json
import time
import uuid
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

try:
    import fcntl  # POSIX advisory file locks — cross-process coordination.
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from hugpy_fleet.central.config import settings

import logging
logger = logging.getLogger(__name__)


# ── assignment memory (daylight 4b) ─────────────────────────────────────────
# Operator designations are WORKER-lifetime, not row-lifetime: a console
# dead-worker sweep DELETE (or a registry loss) used to wipe a worker's
# models + per-model spill, so a re-register came back empty (2026-07-03:
# computron lost 4 of 7 designations). Every assign/unassign snapshots the
# worker's designations here, keyed by its persistent worker id; a fresh-row
# re-register with a known id restores them. Deleting a row deliberately does
# NOT delete its memory — that's the point.

def _assign_memory_path() -> str:
    return os.path.join(settings.state_dir,
                        "worker_assignments.json")


def _load_assign_memory() -> Dict[str, Any]:
    try:
        with open(_assign_memory_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _remember_assignments(worker: Dict[str, Any]) -> None:
    """Snapshot one worker's designations (models + spill) into the memory."""
    wid = worker.get("id")
    if not wid:
        return
    try:
        mem = _load_assign_memory()
        prior = mem.get(wid) if isinstance(mem.get(wid), dict) else {}
        entry = {
            "name": worker.get("name"),
            "models": list(worker.get("models") or []),
            "spill_by_model": dict(worker.get("spill_by_model") or {}),
            "remembered_at": _now(),
        }
        # Durable HARDWARE FACTS ride the same sidecar so they survive even a
        # full registry loss (operator addendum 2026-07-24): carry the last-known
        # totals, advance-only — never overwrite a remembered fact with an absent
        # one, so a re-register during a transient probe miss can't erase them.
        gpu_known = (worker.get(_GPU_TOTAL_DURABLE_KEY)
                     or prior.get(_GPU_TOTAL_DURABLE_KEY))
        ram_known = (worker.get(_RAM_TOTAL_DURABLE_KEY)
                     or prior.get(_RAM_TOTAL_DURABLE_KEY))
        if gpu_known:
            entry[_GPU_TOTAL_DURABLE_KEY] = gpu_known
        if ram_known:
            entry[_RAM_TOTAL_DURABLE_KEY] = ram_known
        mem[wid] = entry
        tmp = _assign_memory_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(mem, fh, indent=1)
        os.replace(tmp, _assign_memory_path())
    except Exception as exc:  # noqa: BLE001 — memory is best-effort, never fatal
        logger.warning("assignment memory write failed for %s: %s", wid, exc)


def forget_assignment_memory(worker_id: str) -> str:
    """SANCTIONED maintenance path to remove one GHOST entry from the
    assignment-memory sidecar (``worker_assignments.json``).

    The design above (deleting a live row does NOT delete its memory) stays
    intact: this refuses to touch any id that still has a live row in
    ``workers.json`` — it can only forget ids that are ALREADY absent from the
    live registry (stray/malformed ids, one-off manual pokes, a worker that
    will never come back under that id). Callers that want to retire a live
    worker's designations should remove/unassign the live row first (or just
    leave the memory — that is the intended durability).

    Returns ``"forgot"`` on success or ``"unknown"`` if ``worker_id`` was not
    present in memory at all. Raises ``ValueError`` if ``worker_id`` is
    currently live (refused).
    """
    if worker_store.get(worker_id) is not None:
        raise ValueError(
            f"worker {worker_id} is LIVE in the registry — assignment memory "
            "may only be forgotten for ids absent from the live registry"
        )
    mem = _load_assign_memory()
    if worker_id not in mem:
        return "unknown"
    del mem[worker_id]
    tmp = _assign_memory_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(mem, fh, indent=1)
    os.replace(tmp, _assign_memory_path())
    return "forgot"


def _default_workers_path() -> str:
    """Sit the worker registry next to the model manifest (…/projects/)."""
    return os.path.join(settings.state_dir, "workers.json")


# A worker that hasn't checked in within this window is considered offline.
HEARTBEAT_TIMEOUT_SECONDS = 45.0


def _provision_stall_seconds() -> float:
    """Forward-progress silence (seconds) after which a provisioning entry stops
    reading as an in-flight pull.

    Same shape + rationale as comms.jobs._stall_seconds (the orphan-job fix):
    read at COMPUTE time so an operator can retune without a restart, and a bad
    value degrades to the default rather than raising into a read path.

    Default 600s. Deliberately MUCH larger than the 45s heartbeat timeout: a
    real pull can legitimately go quiet for minutes (a slow segment, a stalled
    HF mirror, a big file's final flush), and calling a live transfer "dead"
    would strip its eviction guard mid-write — the one truly destructive
    mistake available here. Offline workers are already caught by the cheaper
    liveness gate, so this window only has to cover the ONLINE-but-wedged case.
    """
    raw = (os.environ.get("HUGPY_PROVISION_STALL_SECONDS") or "").strip()
    if not raw:
        return 600.0
    try:
        v = float(raw)
        return v if v > 0 else 600.0
    except ValueError:
        return 600.0


def _live_provisioning(worker: Dict[str, Any]) -> set:
    """The subset of ``worker['provisioning']`` that is a GENUINELY LIVE pull.

    Defect (operator, 2026-07-16): a ``provisioning`` entry was immortal. The
    worker announces the list in its heartbeat and removes an entry in a
    ``finally``; if the process dies mid-pull, that ``finally`` never runs, so
    central reported "provisioning" forever (observed: op offline 2h+, still 4
    entries; ae online with 63 entries and ZERO bytes moving).

    This is the orphan-job defect class — state that ages on writes which STOP
    ARRIVING when the writer dies. The recorded lesson: age on PROGRESS, not on
    presence-in-a-list. So an entry is live only when BOTH hold:

      1. the worker is alive       -> REUSES ``_is_online`` (the single existing
                                      staleness notion; no second rule invented)
      2. its bytes are moving      -> a ``provision_progress`` entry whose
                                      ``done_bytes`` advanced within the stall
                                      window, per the central-stamped
                                      ``progressed_at`` clock (see ``heartbeat``)

    Why (2) needs a central clock rather than ``frac > 0``: op's dead pull is
    frozen at ``frac=0.0722`` with 1.8GB done. A truthy frac only proves bytes
    moved ONCE — never that they are moving NOW. Only elapsed-time-since-advance
    can tell a live 7% from a corpse stuck at 7%.

    QUEUED-NOT-STALLED (why absence of an entry is not evidence of death): the
    worker adds a key to ``_provisioning`` at KICK time but only creates a
    ``_provision_progress`` entry once its download callback fires, and
    ``WORKER_PROVISION_CONCURRENCY`` defaults to 1. So ae's 63 progress-less
    entries are models QUEUED behind the semaphore, not wedged ones — correctly
    NOT in-flight (nothing is transferring), and equally correctly NOT
    eviction-protected (they have no bytes on disk to protect).

    Fail-SAFE toward the live case: if the clock is missing/garbage on an ONLINE
    worker with a progress entry, treat it as live. A false "live" costs a
    delayed console pill; a false "dead" could unprotect a real in-flight write.
    """
    prov = set(worker.get("provisioning") or [])
    if not prov or not _is_online(worker):
        # Offline/stale worker: nothing it last claimed is in flight, because
        # nothing of it is running. This is the op case.
        return set()
    progress = worker.get("provision_progress") or {}
    if not isinstance(progress, dict):
        return set()
    now = _now()
    window = _provision_stall_seconds()
    live = set()
    for mk in prov:
        entry = progress.get(mk)
        if not isinstance(entry, dict):
            continue          # queued behind the concurrency semaphore (ae case)
        ts = entry.get("progressed_at")
        if ts is None:
            live.add(mk)      # fail-safe: pre-clock/legacy entry on a live worker
            continue
        try:
            if (now - float(ts)) <= window:
                live.add(mk)
        except (TypeError, ValueError):
            live.add(mk)      # fail-safe: never unprotect on a garbage clock
    return live


# The distribution workers self-update from and whose version central advertises.
DEFAULT_PKG_NAME = "hugpy-fleet"


def tracked_pkg_name() -> str:
    """Distribution name workers track + central reports its version of.

    Must match the worker's ``--pkg-name`` (``WORKER_PKG_NAME``). Default is
    this package's distribution (``hugpy-fleet``).
    """
    return os.environ.get("HUGPY_PKG_NAME", DEFAULT_PKG_NAME)


def required_pkg_version() -> Optional[str]:
    """The version workers converge to = the version CENTRAL itself is running.

    Read FRESH from central's own installed package metadata on every call — no
    network, no cache, no env/file pins. The instant central adopts a release
    (pip install + restart), ``required`` is that version and every worker
    converges to it on its next heartbeat. There is nothing to wait on and
    nothing to drift: central is the single source of truth. A local "+build"
    version is not advertised (a PyPI-based worker can't install it) → None =
    "not managing versions".
    """
    try:
        import importlib.metadata as _im
        v = _im.version(tracked_pkg_name())
        return v if (v and "+" not in v) else None
    except Exception:
        return None


def pkg_index_dir() -> str:
    """Directory of built wheels that central serves as a PEP-503 simple index.

    ``python py/build_wheels.py --publish <this dir>`` from a checkout of the
    release tag drops every workspace distribution here (CONSISTENCY.md,
    "Releasing through central"). Override with ``HUGPY_PKG_INDEX_DIR``;
    defaults to a ``pip_index`` dir beside the model manifest.
    """
    return os.environ.get("HUGPY_PKG_INDEX_DIR") or \
        os.path.join(settings.state_dir, "pip_index")


# Path (under central's ``/api`` mount) of the PEP-503 index the wheel dir is
# served as. Workers get the absolute URL in every reply that carries
# ``required_pkg_version`` when the dir can actually satisfy that version.
PKG_INDEX_PATH = "/api/llm/pip/simple"


def pkg_index_url(base_url: str) -> str:
    """Absolute URL of central's pip index for the central at ``base_url``."""
    return str(base_url or "").rstrip("/") + PKG_INDEX_PATH


def _index_wheel_version(filename: str) -> Optional[tuple]:
    """``(normalized distribution, version)`` from a wheel/sdist filename."""
    if filename.endswith(".whl"):
        parts = filename[:-4].split("-")
        if len(parts) < 2:
            return None
        return (re.sub(r"[-_.]+", "-", parts[0]).lower(), parts[1])
    if filename.endswith(".tar.gz"):
        stem = filename[:-len(".tar.gz")]
        name, sep, version = stem.rpartition("-")
        if not sep:
            return None
        return (re.sub(r"[-_.]+", "-", name).lower(), version)
    return None


def pkg_index_has(version: Optional[str], names: Optional[Iterable[str]] = None) -> bool:
    """True when the wheel dir holds ``version`` of every distribution in
    ``names`` (default: the whole lockstep set), i.e. a worker converging to
    ``version`` can be served entirely from this central.

    False for an unpinned central (None), an absent dir, or any missing name:
    the reply then omits ``pkg_index_url`` and workers fall back to PyPI, so a
    half-published index never strands a converge."""
    if not version:
        return False
    wanted = {re.sub(r"[-_.]+", "-", n).lower() for n in (names or workspace_distributions())}
    try:
        entries = os.listdir(pkg_index_dir())
    except OSError:
        return False
    found = set()
    for fn in entries:
        parsed = _index_wheel_version(fn)
        if parsed and parsed[1] == version and parsed[0] in wanted and fn.endswith(".whl"):
            found.add(parsed[0])
    return wanted <= found


# The in-tree workspace distributions, all versioned in LOCKSTEP (one
# git-derived version via setuptools-scm — the 0.2 partition). The canonical
# list is ``hugpy_platform.buildinfo.WORKSPACE_DISTRIBUTIONS``; this constant is
# the fallback for a central whose installed hugpy-platform predates it.
WORKSPACE_DISTRIBUTIONS_FALLBACK = (
    "hugpy-platform",
    "hugpy-control",
    "hugpy-storage",
    "hugpy-engine",
    "hugpy-media",
    "hugpy-video",
    "hugpy-oracle",
    "hugpy-fleet",
    "hugpy-curation",
    "hugpy-ops",
    "hugpy-discord",
    "hugpy-server",
    "hugpy",
)

# Path (under central's ``/api`` mount) of the lockstep constraints file. A
# worker derives the absolute URL from its central base URL when a reply omits
# ``constraints_url``.
CONSTRAINTS_PATH = "/api/llm/workers/constraints.txt"


def workspace_distributions() -> tuple:
    """Every distribution the lockstep version applies to.

    Imported lazily from ``hugpy_platform.buildinfo`` so a central running an
    older hugpy-platform still answers (from the fallback constant) instead of
    failing at import time.
    """
    try:
        from hugpy_platform.buildinfo import WORKSPACE_DISTRIBUTIONS
        names = tuple(str(n).strip() for n in WORKSPACE_DISTRIBUTIONS
                      if str(n).strip())
        if names:
            return names
    except Exception:  # noqa: BLE001 — older platform: no buildinfo yet
        pass
    return WORKSPACE_DISTRIBUTIONS_FALLBACK


def lockstep_constraints(required: Optional[str] = None) -> List[str]:
    """``name==<required>`` for every workspace distribution — the body of a
    pip constraints file, one line each.

    A worker runs hugpy-fleet PLUS its siblings (platform, control, storage,
    engine, media, ...); upgrading the tracked distribution alone leaves the
    rest behind — the silent skew the 2026-07-20 incident class is made of.
    Central hands out this pin set so the worker's ``pip install -c`` converges
    the WHOLE set to the one version central itself runs. Empty when central
    pins no version (``required_pkg_version()`` is None): nothing to converge
    to, and the route answers 204.
    """
    if required is None:
        required = required_pkg_version()
    if not required:
        return []
    return [f"{name}=={required}" for name in workspace_distributions()]


def constraints_url(base_url: str) -> str:
    """Absolute URL of the constraints file for the central at ``base_url``."""
    return str(base_url or "").rstrip("/") + CONSTRAINTS_PATH


def _now() -> float:
    return time.time()


# ── operator model BLOCK (central serving-pool primitive) ────────────────────
# A blocked model is removed from the pool: never a routing candidate, never a
# warm/provision target, never a fallback default. The registry lives in the F4
# settings store (comms.blocklist); these are guarded thin wrappers so a
# blocklist read can NEVER raise into routing (fail-open = not blocked, because
# a routing gate that 500s is worse than a momentarily-unblocked model).
def _model_blocked(model_key: str) -> bool:
    try:
        from hugpy_fleet.central.blocklist import is_blocked
        if is_blocked(model_key):
            return True
    except Exception:  # noqa: BLE001 — never let the block gate break selection
        pass
    # DRAFT-MODEL-GATE-20260910: a speculative-decoding draft head is never a routing candidate.
    try:
        from hugpy_engine.draft_models import draft_model_reason
        return draft_model_reason(model_key) is not None
    except Exception:  # noqa: BLE001
        return False


def _blocked_keys() -> set:
    try:
        from hugpy_fleet.central.blocklist import blocked_keys
        return blocked_keys()
    except Exception:  # noqa: BLE001
        return set()


def _fleet_least_reaping() -> bool:
    """The fleet's drop-pass policy, degrading to the module default.

    Central's storage_proposal preview and every worker's auto-evict must run
    the same drop pass (Parity, spec assets/evictionflow.html) — see
    comms/evict_policy.py for why this knob is fleet-wide and the anti-thrash
    floor is not."""
    try:
        from hugpy_fleet.central.evict_policy import least_reaping
        return least_reaping()
    except Exception:  # noqa: BLE001 — a preview must never break on policy
        from hugpy_engine.eviction import DEFAULT_LEAST_REAPING
        return DEFAULT_LEAST_REAPING


def _is_online(worker: Dict[str, Any]) -> bool:
    last = worker.get("last_seen") or 0
    return (_now() - last) <= HEARTBEAT_TIMEOUT_SECONDS


# ── THE ONE LEDGER: derived columns (operator, 2026-07-25) ───────────────────
# "in the end it is about maximizing tok/s ... lets start recording this".
#
# Two signals, both stamped onto the SAME ``model_call_stats[model_key]`` row
# that already carries ``calls``/``last_call``, because the Parity invariant
# (spec assets/evictionflow.html) is specifically that central's preview and the
# worker's auto-evict rank from ONE ledger. A second store — however tidy —
# would be the exact failure the spec exists to prevent.
#
# ⚠ NOTHING READS EITHER OF THESE YET, BY DESIGN. ``eviction.sort_key`` is
# untouched, so recording cannot change which model is evicted or placed. These
# are inert columns being accumulated so that a future policy has a history to
# rank on; a distribution you never wrote down cannot be reconstructed later.
#
# Both writers are pure functions over the row dict + the new sample, so the
# arithmetic is testable without a store, a worker, or a clock.

_EWMA_ALPHA = 0.3   # recent behaviour dominates within ~3 samples, while a
                    # single outlier cannot swing the estimate.


def _ewma(prev: Any, sample: float, alpha: float = _EWMA_ALPHA) -> float:
    """One EWMA step, FIRST-SAMPLE SEEDED.

    A None/garbage ``prev`` seeds at the sample itself rather than at 0 — seeding
    at zero would make every model's first observation read as half its true
    value and take ~10 samples to recover, which is precisely the kind of quiet
    wrongness a recording-only slice must not bake into the history.
    """
    try:
        p = float(prev)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(sample)
    return alpha * float(sample) + (1.0 - alpha) * p


def _record_interval(row: Dict[str, Any], prev_call: Any, now: float) -> None:
    """Stamp the CALL-INTERVAL columns onto a ledger row. Fail-open.

    "Time since last call" is a POINT ESTIMATE of a distribution: a model called
    every 30s for an hour and one called once yesterday can both read 5 minutes
    idle, and they are not remotely the same eviction risk. What actually ranks
    them is EXPECTED TIME UNTIL NEXT CALL, which is estimable from the history —
    but only if the history is kept. So keep it here, on the line that already
    stamps the clock: one subtraction, one float, no second ledger.

    LOG SPACE because call intervals are heavy-tailed (mostly short gaps, the
    occasional very long one), so an arithmetic mean is dominated by outliers.
    An exponentially-weighted mean of log-intervals is stable, costs one float
    per model, needs no training and no model artifact, and stays DERIVABLE —
    which is the property the eviction spec exists to protect. Whether a learned
    estimator ever earns its complexity is a decision to make against this data,
    not before it.

    The FIRST call for a model records no interval at all (there is no previous
    call to difference against) — deliberately absent rather than stamped zero,
    since a fabricated 0s gap would read as the hottest possible model.
    """
    if not prev_call:
        return
    try:
        gap = float(now) - float(prev_call)
    except (TypeError, ValueError):
        return
    if gap <= 0:
        # Clock skew / same-instant re-pick. Recording a non-positive interval
        # would put -inf into log space and poison the EWMA permanently.
        return
    try:
        import math
        lg = math.log(max(gap, 1e-3))
        row["log_interval_ewma"] = _ewma(row.get("log_interval_ewma"), lg)
        row["last_interval_s"] = round(gap, 3)
        row["interval_samples"] = int(row.get("interval_samples") or 0) + 1
    except (TypeError, ValueError):   # noqa: BLE001 — never break a pick
        pass


def _record_tok_s(row: Dict[str, Any], tok_s: Any) -> bool:
    """Stamp the DECODE-RATE columns onto a ledger row. Fail-open.

    PLAIN (not log) EWMA, unlike the interval above, and the difference is not
    stylistic. Intervals are heavy-tailed over orders of magnitude (seconds to
    days), so their arithmetic mean is meaningless. Decode rate for a FIXED
    (model, worker, placement) pair is tightly clustered — 115 tok/s stays 115
    tok/s until the placement changes — so a plain mean is both meaningful and
    directly comparable, which is exactly the property "maximize tok/s" needs.
    The one thing that genuinely moves it is a placement change (the MoE split
    measured +59%, the offload cliff 135->36), and a plain alpha-0.3 EWMA tracks
    such a step change within a few calls instead of smearing it in log space.

    Returns True when a sample was recorded, so callers can log/test the seam
    without re-deriving the validity rules.
    """
    try:
        v = float(tok_s)
    except (TypeError, ValueError):
        return False
    # Reject the impossible rather than average it in: a non-finite or
    # non-positive rate is a broken measurement, and a zero-token generation
    # (predicted_n == 0) reports 0.0 tok/s while saying nothing about how fast
    # the model decodes. Recording it would drag the mean toward zero and make a
    # fast model look slow — degrade-not-guess.
    if not (v > 0.0) or v != v or v in (float("inf"), float("-inf")):
        return False
    row["tok_s_last"] = round(v, 3)
    row["tok_s_ewma"] = round(_ewma(row.get("tok_s_ewma"), v), 3)
    row["tok_s_samples"] = int(row.get("tok_s_samples") or 0) + 1
    return True


# ── tok/s LOG + ROLLUPS (operator 2026-08-22: "every query logged, and its
# recent tok/s as a variable viewed in every model selection") ───────────────
#
# Three surfaces, one writer (``record_query_toks``):
#   (a) append-only JSONL beside workers.json — ``toks_log.jsonl`` — one line
#       per completed query (worker, model, tokens, elapsed, tok/s, request id);
#   (b) ``worker["model_tok_stats"][model_key]`` = {last_tok_s, avg_tok_s, n,
#       last_ts} — the per-(worker,model) rollup pick_for_model reads;
#   (c) ``worker["tok_stats"]`` = {last_tok_s, avg_tok_s, n, last_model,
#       last_ts} — the per-worker rollup the console header shows.
# ``model_call_stats[model_key].tok_s_*`` (the parity ledger row above) keeps
# being stamped by the same call, so the eviction substrate is unchanged.

TOKS_LOG_NAME = "toks_log.jsonl"
TOKS_LOG_MAX_BYTES = 64 * 1024 * 1024   # rotate to .1 past this; bounded disk
TOKS_LOG_LIMIT_MAX = 500


def _toks_log_path() -> str:
    override = os.environ.get("HUGPY_TOKS_LOG")
    if override:
        return override
    return os.path.join(os.path.dirname(_default_workers_path()), TOKS_LOG_NAME)


def _append_toks_log(entry: Dict[str, Any]) -> None:
    """Append one JSONL line. Fail-open; rotates once past the size cap."""
    try:
        path = _toks_log_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            if os.path.getsize(path) > TOKS_LOG_MAX_BYTES:
                os.replace(path, path + ".1")
        except OSError:
            pass
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception:  # noqa: BLE001 — a log write must never fail a request
        logger.debug("toks log append skipped", exc_info=True)


def read_toks_log(worker_id: Optional[str] = None, limit: int = 50,
                  model_key: Optional[str] = None,
                  max_limit: int = TOKS_LOG_LIMIT_MAX) -> List[Dict[str, Any]]:
    """Most-recent-first tail of the tok/s log, optionally filtered. Bounded.

    ``max_limit`` caps how many entries a caller may ask for; it defaults to the
    per-worker route's 500 but the fleet-wide /toks/recent + /toks/report
    aggregates raise it (still bounded by the 4 MiB tail read below)."""
    try:
        limit = max(1, min(int(limit or 50), int(max_limit)))
    except (TypeError, ValueError):
        limit = 50
    path = _toks_log_path()
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            chunk = min(size, 4 * 1024 * 1024)
            fh.seek(size - chunk)
            lines = fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return out
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if worker_id and e.get("worker_id") != worker_id:
            continue
        if model_key and e.get("model_key") != model_key:
            continue
        out.append(e)
        if len(out) >= limit:
            break
    return out


def _vram_bucket(vram_total: Optional[int]) -> str:
    """A coarse VRAM class for the config_key — total GiB rounded to the
    nearest 8 GiB so 24564MiB and 24GiB cards group together. ``"?"`` when the
    box never reported a GPU total (config still groups on the rest)."""
    try:
        gib = int(vram_total) / float(1 << 30)
    except (TypeError, ValueError):
        return "?"
    if gib <= 0:
        return "?"
    return f"{int(round(gib / 8.0) * 8)}g"


def _config_key(backend: Optional[str], dtype: Optional[str],
                co_resident: int, vram_total: Optional[int]) -> str:
    """The STABLE grouping string for the allocation study, shape
    ``f"{backend}|{dtype}|co{N}|vram{bucket}"`` — engine, quant/dtype, how many
    models were co-resident at serve time, and the VRAM class of the box. Two
    serves that share this string are "the same configuration"; the report
    ranks configs by mean tok/s to answer which one is fastest. Unknown parts
    degrade to ``"?"`` rather than dropping out, so the key stays fixed-shape."""
    return "%s|%s|co%d|vram%s" % (
        (backend or "?"), (dtype or "?"), max(0, int(co_resident or 0)),
        _vram_bucket(vram_total))


def _state_snapshot(stored: Dict[str, Any], model_key: str,
                    meta: Dict[str, Any]) -> Dict[str, Any]:
    """The per-query configuration snapshot stamped onto the ledger line.

    Sourced from the STORED worker record on central (not the relay): vram
    used/free, disk allocation and the co-resident model set live on the
    heartbeat, so only central can say what configuration produced a given
    tok/s. Fail-open and bounded — every section is guarded, nulls are omitted,
    and the whole thing is capped so a JSONL line stays ~<=2KB. Also derives the
    stable ``config_key`` used to group serves in the report.

    ``meta`` carries the request-side facts ``_query_meta`` already measured
    (prompt/completion tokens, ttft, elapsed, inflight, streaming, warm, ok,
    breaker); they are echoed into ``request``/``outcome``/``model.loaded_at_pick`` so the
    whole state of one serve reads from one object."""
    state: Dict[str, Any] = {}
    try:
        vs: Dict[str, Any] = {}
        try:
            vs = _vram_summary(stored)
        except Exception:  # noqa: BLE001
            vs = {}
        vram_total = vs.get("vram_total")

        worker: Dict[str, Any] = {}
        for k_out, val in (("id", stored.get("id")),
                           ("name", stored.get("name")),
                           ("agent_version", stored.get("agent_version")
                            or stored.get("pkg_version")),
                           ("gpu", vs.get("gpu")),
                           ("vram_total", vram_total)):
            if val is not None:
                worker[k_out] = val
        if worker:
            state["worker"] = worker

        loaded = stored.get("loaded_models")
        co_resident: List[str] = []
        if isinstance(loaded, (list, tuple, set)):
            co_resident = sorted(str(m) for m in loaded)[:16]
        vram: Dict[str, Any] = {}
        for k in ("vram_used", "vram_free"):
            if vs.get(k) is not None:
                vram[k] = vs[k]
        if co_resident:
            vram["loaded_models"] = co_resident
        try:
            sq = _vram_squatters(stored)
            if sq:
                vram["squatters"] = len(sq)
        except Exception:  # noqa: BLE001
            pass
        if vram:
            state["vram"] = vram

        alloc: Dict[str, Any] = {}
        try:
            sp = storage_proposal(stored)
        except Exception:  # noqa: BLE001
            sp = {}
        limits = stored.get("limits") if isinstance(stored.get("limits"), dict) else {}
        for k_out, val in (
                ("disk_cache_gib", limits.get("disk_cache_gib")),
                ("disk_reserve_gib", limits.get("disk_reserve_gib")),
                ("effective_budget_bytes", sp.get("budget_effective_bytes")),
                ("cache_used_bytes", sp.get("cache_used_bytes")),
                ("disk_free", sp.get("disk_free")),
                ("resident_count", sp.get("allocated_count"))):
            if val is not None:
                alloc[k_out] = val
        if alloc:
            state["alloc"] = alloc

        backend = None
        try:
            backend = _model_engine(model_key)
        except Exception:  # noqa: BLE001
            backend = None
        phys: Dict[str, Any] = {}
        try:
            phys = _model_physical(model_key) or {}
        except Exception:  # noqa: BLE001
            phys = {}
        dtype = (phys.get("quant") or phys.get("dtype")
                 or phys.get("effective_quant"))
        size_bytes = None
        try:
            if phys.get("size_bytes") is not None:
                size_bytes = int(phys.get("size_bytes"))
        except (TypeError, ValueError):
            size_bytes = None
        model: Dict[str, Any] = {"model_key": model_key}
        for k_out, val in (
                ("size_bytes", size_bytes),
                ("dtype", dtype), ("backend", backend)):
            if val is not None:
                model[k_out] = val
        loaded_at = meta.get("loaded_at_pick")
        if loaded_at is not None:
            model["loaded_at_pick"] = bool(loaded_at)
        state["model"] = model

        request: Dict[str, Any] = {}
        for k in ("prompt_tokens", "completion_tokens", "max_tokens",
                  "ttft_s", "elapsed_s", "inflight"):
            if meta.get(k) is not None:
                request[k] = meta[k]
        if meta.get("streaming") is not None:
            request["streaming"] = bool(meta["streaming"])
        if request:
            state["request"] = request

        outcome: Dict[str, Any] = {}
        if meta.get("ok") is not None:
            outcome["ok"] = bool(meta["ok"])
        if meta.get("breaker") is not None:
            outcome["breaker"] = meta["breaker"]
        if outcome:
            state["outcome"] = outcome

        state["config_key"] = _config_key(
            backend, dtype, len(co_resident), vram_total)
    except Exception:  # noqa: BLE001 — a snapshot must never fail a request
        logger.debug("state snapshot skipped for %s", model_key, exc_info=True)
    return state


def _rollup_tok_stats(row: Dict[str, Any], tok_s: float, now: float,
                      model_key: Optional[str] = None) -> Dict[str, Any]:
    """One step of the {last_tok_s, avg_tok_s, n, last_ts} rollup. Pure.

    ``avg_tok_s`` is the same alpha-0.3 EWMA as the ledger column (see
    ``_record_tok_s`` for why plain rather than log space). ``n`` counts
    samples. Returns the row it mutated so callers can chain/test it."""
    v = float(tok_s)
    row["last_tok_s"] = round(v, 3)
    row["avg_tok_s"] = round(_ewma(row.get("avg_tok_s"), v), 3)
    try:
        row["n"] = int(row.get("n") or 0) + 1
    except (TypeError, ValueError):
        row["n"] = 1
    row["last_ts"] = float(now)
    if model_key is not None:
        row["last_model"] = model_key
    return row


def tok_hint_for(worker: Optional[Dict[str, Any]],
                 model_key: str) -> Dict[str, Any]:
    """{last_tok_s, avg_tok_s, n} for (worker, model) — the variable every
    selection reads. Falls back to the parity ledger's tok_s_* columns for a
    record written before the rollup existed; all-None when nothing is known."""
    hint: Dict[str, Any] = {"last_tok_s": None, "avg_tok_s": None, "n": 0}
    if not isinstance(worker, dict):
        return hint
    row = (worker.get("model_tok_stats") or {}).get(model_key)
    if isinstance(row, dict) and row.get("n"):
        hint["last_tok_s"] = row.get("last_tok_s")
        hint["avg_tok_s"] = row.get("avg_tok_s")
        hint["n"] = row.get("n") or 0
        return hint
    cs = (worker.get("model_call_stats") or {}).get(model_key)
    if isinstance(cs, dict) and cs.get("tok_s_samples"):
        hint["last_tok_s"] = cs.get("tok_s_last")
        hint["avg_tok_s"] = cs.get("tok_s_ewma")
        hint["n"] = cs.get("tok_s_samples") or 0
    return hint


# ``tok_s_from_timings`` lives in ``managers.eviction`` — the module BOTH the
# central preview and the worker's auto-evict already import, so the one parser
# stays on the parity substrate rather than behind a web-only import. Re-exported
# here because this is where the ledger writers live.
from hugpy_engine.eviction import tok_s_from_timings


def _public_view(worker: Dict[str, Any]) -> Dict[str, Any]:
    """The shape returned to API callers — derived ``status`` included.

    ``status`` is *liveness* (online/offline from last_seen). ``admission`` is the
    operator gate (pending/approved/blocked) and is independent of liveness. Rows
    written before the admission feature have no ``admission`` key; they are
    grandfathered to ``approved`` here so an existing fleet keeps serving.

    Every PER-MODEL physical fact below (sizes, MoE structure, marker
    capabilities) is a lookup against the persisted record, with only a BOUNDED
    amount of cold-record filling per view — see _view_fill_window. Everything
    the worker REPORTED stays LIVE and untouched: gpus[] free/total, slots[]
    health/pid, disk, the storage survey, last_seen, loaded/loading/provisioning,
    status. Those are the point of the view; only the derived per-model physical
    facts moved.
    """
    with _view_fill_window():
        return _public_view_fields(worker)


def _public_view_fields(worker: Dict[str, Any]) -> Dict[str, Any]:
    """The body of :func:`_public_view`, inside its cold-fill window."""
    return {
        **worker,
        **_vram_summary(worker),
        **_ram_summary(worker),
        # Derived local-storage view + guarded LRU eviction proposal, recomputed
        # on every read from already-stored fields (same pure-function pattern as
        # the vram/ram summaries above; no daemon, no auto-fire — nothing deletes
        # here). Overwrites the raw ``storage`` heartbeat field with the enriched
        # console-facing shape (over_budget + proposed_evictions[]).
        "storage": storage_proposal(worker),
        # IN-FLIGHT PULLS ONLY (2026-07-16). The raw record keeps whatever the
        # worker last announced; the PUBLIC view reports only pulls that are
        # actually moving, so a dead/stalled entry can never render as an
        # active transfer ("defaults are promises" — a row that says "working
        # on it" when nothing is working is a lie). Derived here, on every read,
        # like status/storage above — no daemon, no sweep. The console's
        # ⏳ pulling pill and its provision_progress % both key off this list,
        # so an assigned-but-absent model correctly falls through to "missing".
        "provisioning": sorted(_live_provisioning(worker)),
        "status": "online" if _is_online(worker) else "offline",
        "admission": worker.get("admission", "approved"),
        # SYSTEM-authored placement grants (Phase 1 item 2) — separate from the
        # operator-designated ``models`` list. Never treat a missing key as
        # absence-of-feature; always surface the (possibly empty) dict so
        # console/tests can see grants land and clear.
        "grants": dict(worker.get("grants") or {}),
        # THE PREFERENCE MAP — key ① of the shared eviction sort (spec
        # assets/evictionflow.html, 2026-07-25). ``{model_key: mode_name}`` for
        # every model with a persisted allocation on this worker, derived
        # read-time from the SAME spill dict the emission seam reads, so the
        # mode the console shows, the mode the worker serves under, and the
        # preference that decides which resident dies are one value.
        #
        # Rides _public_view (like storage/status) so it reaches the worker on
        # the heartbeat reply and is adopted in _adopt_storage_inputs. A model
        # with nothing persisted is deliberately ABSENT rather than stamped
        # max-gpu: absent degrades to the blank max-gpu default at the reader,
        # which is the same answer without asserting a preference nobody chose.
        "model_alloc_modes": _model_alloc_modes(worker),
        # BITSANDBYTES SPECIALIZATION (operator, 2026-07-26) — two separate
        # maps because "can this take it" and "is it switched on" are different
        # questions and the console needs both: availability decides whether the
        # cell renders a lever at all, enablement decides whether it is ticked.
        # A model absent from bnb_available simply has no lever (gguf, a
        # CPU-only worker, an already-quantized repo).
        "bnb_by_model": dict(worker.get("bnb_by_model") or {}),
        "bnb_available": {mk: True for mk in (worker.get("models") or [])
                          if bnb_available(worker, mk)},
        # MoE: capability (can it split at all), the operator override, and the
        # EFFECTIVE state the checkbox renders — auto shows as ticked when the
        # derivation produced a split, so the real behaviour is never hidden.
        "moe_capable": {mk: True for mk in (worker.get("models") or [])
                        if moe_capable(mk)},
        "moe_by_model": dict(worker.get("moe_by_model") or {}),
        "moe_effective": {mk: True for mk in (worker.get("models") or [])
                          if moe_effective(worker, mk)},
        # The INTENDED vram/ram division per model — what the Memory column
        # shows for a model that is not resident yet, so it stops echoing the
        # Size column and starts answering "where will this actually go".
        "planned_split": {mk: planned_split(worker, mk)
                          for mk in (worker.get("models") or [])},
        # k67 item G — INERT SPILL ROWS. A persisted spill for a BLOCKED model is
        # a dead contract (the model can't route while blocked, and block never
        # authored it). Rather than let it linger indistinguishable from a live
        # spill, LABEL it here so the console renders it inert instead of as a
        # real placement. {model_key: reason} for every spill row whose model is
        # currently blocked; absent/empty when nothing is inert.
        "spill_inert": {mk: "blocked"
                        for mk in (worker.get("spill_by_model") or {})
                        if _model_blocked(mk)},
        # VRAM-HOLDER METER (incident 2026-08-05). ``vram_holders`` rides through
        # verbatim from **worker (the worker's read-only per-card breakdown + a
        # row per holder). ``vram_squatters`` is the DERIVED explicit summary:
        # the idle own-venv holders the worker flagged reapable-but-not-serving,
        # re-shaped with worker identity — and the seam that shouts ONE warning
        # per distinct squatter so a silent VRAM leak (the 8GB studio fork that
        # wedged the card) can never happen again. Read-time and pure like every
        # other derivation here; never raises into a worker read.
        "vram_squatters": _vram_squatters(worker),
    }


def _model_alloc_modes(worker: Dict[str, Any]) -> Dict[str, str]:
    """``{model_key: alloc mode name}`` from this worker's persisted spills.

    PURE and read-time, like every other _public_view derivation. Uses
    ``alloc_modes.derive_alloc_mode`` — the ONE reader of a persisted spill —
    so this can never disagree with what ``spill_for`` emits or what the
    console dropdown displays.

    ⚠ ``{}`` (derived max-gpu) and ``{"alloc_mode": "max-gpu"}`` (explicit,
    b0e02ff) BOTH resolve to "max-gpu" here, which is correct: they differ in
    provenance, not in preference, and preference is all key ① consumes.

    Never raises into a worker read: an unparseable spill is simply omitted
    (degrade-not-guess — the reader then uses the blank default)."""
    out: Dict[str, str] = {}
    try:
        from hugpy_engine.alloc_modes import derive_alloc_mode
    except Exception:  # noqa: BLE001
        return out
    for mk, spill in (worker.get("spill_by_model") or {}).items():
        if not isinstance(spill, dict) or not spill:
            continue
        try:
            # t143: label the EFFECTIVE placement — the pin with the operator's
            # MoE checkbox overlaid, exactly as _placement_spill_for emits it —
            # so toggling the split on a pinned row moves the Alloc cell. A
            # wire-inert max-gpu pin is stripped first (as at the seam); when
            # the overlay leaves nothing, the pin's own label stands.
            eff = apply_moe_override_to_spill(
                worker, str(mk), _strip_wire_inert_mode(dict(spill)))
            out[str(mk)] = derive_alloc_mode(eff or spill)
        except Exception:  # noqa: BLE001
            continue
    # DERIVED DEFAULTS FOR THE REST (2026-07-26). Persisted rows are the
    # MINORITY: on ae only 2 of 64 assigned models carry a contract, and the
    # other 62 track the derivation. Emitting only the persisted ones left the
    # console with nothing to show for those 62, so its deriveAllocMode() fell
    # through to its hardcoded 'max-gpu' — displaying "⚡ Max GPU · auto" for a
    # 67 GiB transformers model whose real derived default is ram-only (it
    # cannot fit a 24 GiB card, and max-gpu on transformers has no spill to
    # fall back on). The operator's decision tree was already correct and
    # already shipped; its ANSWER just never reached the UI.
    #
    # A persisted contract always wins, so this only fills the blanks
    # (setdefault). Read-time and pure like the rest of _public_view; per-key
    # failures are skipped rather than raised, and a key that resolves to
    # nothing is simply omitted (the console then keeps its own fallback).
    for mk in (worker.get("models") or []):
        if str(mk) in out:
            continue
        try:
            mode = derived_default_mode(worker, str(mk))
        except Exception:  # noqa: BLE001 — never break a worker read
            continue
        if mode:
            out[str(mk)] = mode
    return out


def _clamp_limits(limits: Dict[str, Any], caps: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp operator limits to the worker's own configured caps.

    The worker's unit config is authoritative ("central shall be forced to
    view that as its max") — central can set anything LESS, never more."""
    out: Dict[str, Any] = {}
    for k, v in limits.items():
        cap = caps.get(k)
        if cap is not None:
            try:
                v = min(float(v), float(cap))
                if k == "threads":
                    v = int(v)
            except (TypeError, ValueError):
                continue
        out[k] = v
    return out


# Default runtime-environment tier. A worker that doesn't report its env (older
# agent) and a model with no explicit requirement both resolve to this, so a
# pre-feature fleet keeps matching exactly as before the tier gate existed.
DEFAULT_ENV_TIER = "stable"


def _model_env_tiers() -> Dict[str, str]:
    """Operator map of model -> REQUIRED env tier.

    Parsed from ``HUGPY_MODEL_ENV_TIERS`` = ``"key:tier,key2:tier"`` (e.g.
    ``"Qwen3.6-27B-AEON:edge"``). A model not listed requires the default tier,
    so this whole gate is a no-op until the operator maps a model.
    """
    out: Dict[str, str] = {}
    for part in os.environ.get("HUGPY_MODEL_ENV_TIERS", "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, _, tier = part.rpartition(":")
        key, tier = key.strip(), tier.strip().lower()
        if key and tier:
            out[key] = tier
    return out


def env_tier_for_model(model_key: str) -> str:
    """The runtime-env tier ``model_key`` requires (alias-tolerant lookup)."""
    tiers = _model_env_tiers()
    if not tiers:
        return DEFAULT_ENV_TIER
    wanted = _match_keys(model_key)
    for key, tier in tiers.items():
        if key == model_key or (_match_keys(key) & wanted):
            return tier
    return DEFAULT_ENV_TIER


def _worker_env_tier(worker: Dict[str, Any]) -> str:
    """The env tier a worker ADVERTISES (from its own venv, via register/heartbeat).

    Workers that don't report an env (older agents) are treated as serving the
    default tier — the grandfather rule that keeps a pre-feature fleet routing
    unchanged. An edge model therefore only lands on a worker that AFFIRMATIVELY
    advertises the edge env.
    """
    env = worker.get("env")
    tier = env.get("tier") if isinstance(env, dict) else None
    if tier is None:
        return DEFAULT_ENV_TIER
    return str(tier).strip().lower() or DEFAULT_ENV_TIER


def _engine_unusable(worker: Dict[str, Any]) -> bool:
    """True only when a worker EXPLICITLY reports it has no inference engine.

    Workers that don't report engine status (older agents) are assumed capable,
    so this never excludes a pre-feature fleet — it only skips a worker that
    affirmatively says ``engine.installed == False`` (e.g. llama-cpp missing),
    which would otherwise be picked and fail every request.
    """
    eng = worker.get("engine")
    return isinstance(eng, dict) and eng.get("installed") is False


# Tasks whose worker capability is authoritatively gated ELSEWHERE and must NOT be
# re-filtered by the find_spec-derived task_capabilities map:
#   image-text-to-text — vision uses the stricter engine.supports_vision truth
#     (llama.cpp mtmd build) enforced in resolvers.remote; a transformers-VL worker
#     without llama_cpp would advertise this False and be wrongly skipped here.
_TASK_CAP_GATE_EXCLUDE = {"image-text-to-text"}


def _task_capable(worker: Dict[str, Any], task: Optional[str]) -> bool:
    """Whether ``worker`` can run ``task`` per its advertised ``task_capabilities``.

    Capability-honest, exactly like the engine/vision/tier gates: a worker is
    skipped ONLY when it AFFIRMATIVELY advertises the task as unavailable — the
    2026-07-11 request-time-failure class (a canonical venv missing
    sentence-transformers / whisper / keybert). LEGACY workers (no
    ``task_capabilities`` field) and tasks a worker doesn't enumerate are assumed
    capable, so a pre-feature fleet routes exactly as before. A ``None`` task
    (non-ML routing, e.g. video auto-pick) never gates, and vision defers to the
    stricter engine gate (``_TASK_CAP_GATE_EXCLUDE``).
    """
    if not task or task in _TASK_CAP_GATE_EXCLUDE:
        return True
    caps = worker.get("task_capabilities")
    if not isinstance(caps, dict) or task not in caps:
        return True
    if bool(caps.get(task)):
        return True
    # COMFY-BACKED IMAGE GEN (2026-07-29) — the vision carve-out's twin. The
    # find_spec map equates text-to-image with "diffusers importable in the
    # worker venv", but a comfy-served model never touches diffusers: the job
    # goes over HTTP to ComfyUI's own process/env. ae proved the failure —
    # heartbeating text-to-image:False (no diffusers) while comfy.available
    # was True with 20 checkpoints listed, so every comfy request was skipped
    # into "no registered worker is available". comfy.available is the real
    # gate for this family; honor it before declaring the worker incapable.
    if task in ("text-to-image", "image-to-image"):
        comfy = worker.get("comfy")
        if isinstance(comfy, dict) and comfy.get("available"):
            return True
    return False


def _comfy_id_lock_capable(worker: Dict[str, Any]) -> bool:
    """Whether ``worker``'s ComfyUI can do identity-locked STILLs — the
    IPAdapter node pack is installed (``comfy.id_lock`` advertised True from the
    agent's object_info probe).

    STRICT / affirmative-only, DELIBERATELY UNLIKE ``_task_capable``'s
    legacy-permissive default: an id_lock request must land on a box that PROVABLY
    has the nodes, because silently degrading to a NON-locked image is forbidden
    (WORKER-SETUP §5b / comfy_runner). A worker with no ``comfy`` block, comfy
    unavailable, or ``id_lock`` != True does NOT qualify. There's no legacy fleet
    to preserve here — id_lock is a brand-new capability, so "unknown" means "not
    yet", not "assume yes".
    """
    comfy = worker.get("comfy")
    if not isinstance(comfy, dict) or not comfy.get("available"):
        return False
    return bool(comfy.get("id_lock"))


def _has_usable_gpu(worker: Dict[str, Any]) -> bool:
    """Whether the worker advertises a GPU with free VRAM (for efficiency ranking).

    Capability-honest, like vision routing: a worker whose llama.cpp build
    AFFIRMATIVELY reports it cannot offload (engine.supports_gpu_offload is
    False) ranks as GPU-less no matter what nvidia-smi shows — n_gpu_layers is
    silently ignored by a CPU-only wheel, so its "GPU" would never be used.
    Older agents that don't report the flag keep their GPU credit (no guessing).
    """
    if (worker.get("engine") or {}).get("supports_gpu_offload") is False:
        return False
    return any((g.get("memory_free") or 0) > 0 for g in (worker.get("gpus") or []))


_GIB = 2 ** 30

# Last-logged "reclaimable collapse" signature per worker — the once-per-state
# dedupe for the guard-chain diagnostic below (per-process, like the sibling
# module-level caches here).
_RECLAIM_COLLAPSE_SEEN: Dict[str, tuple] = {}

# VRAM-SQUATTER warning dedup (incident 2026-08-05). Per worker, the SET of
# squatter pids already warned about — so a distinct idle own-venv VRAM holder
# is shouted about EXACTLY ONCE (never silent, never a per-read spam storm),
# re-warned only if it disappears and comes back. Same once-per-state discipline
# as _RECLAIM_COLLAPSE_SEEN above.
_VRAM_SQUATTER_SEEN: Dict[str, set] = {}


def _vram_squatters(worker: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The EXPLICIT idle-squatter rows for one worker, from its reported
    ``vram_holders`` meter, plus a deduped WARNING per distinct squatter.

    A squatter is the operator's "not evictable AND not doing work, made
    EXPLICIT" case: an own-venv VRAM holder past the reaper's min-age with no
    live slot claim that is not serving — reapable, but (p27) never auto-reaped.
    The WORKER already applied the reaper's four gates and flagged these rows;
    central only re-shapes them with the worker's identity and shouts once each
    so the leak is never silent. Never raises into a read (pure telemetry)."""
    holders = worker.get("vram_holders")
    if not isinstance(holders, dict):
        return []
    wname = worker.get("name") or (worker.get("id") or "?")[:8]
    rows: List[Dict[str, Any]] = []
    for h in (holders.get("squatters") or []):
        if not isinstance(h, dict):
            continue
        rows.append({
            "worker": wname,
            "worker_id": worker.get("id"),
            "pid": h.get("pid"),
            "name": h.get("name"),
            "vram_bytes": h.get("vram_bytes"),
            "age_s": h.get("age_s"),
            "reason": h.get("reason") or (
                "holds VRAM, no live slot claim, not serving — reapable but "
                "not auto-reaped"),
        })
    try:
        seen = _VRAM_SQUATTER_SEEN.setdefault(wname, set())
        current = {r.get("pid") for r in rows if r.get("pid") is not None}
        for r in rows:
            pid = r.get("pid")
            if pid is None or pid in seen:
                continue
            seen.add(pid)
            logger.warning(
                "VRAM SQUATTER on %s: pid %s (%s) holds %s of VRAM but no live "
                "slot claims it and it is not serving — reapable, NOT auto-reaped "
                "(p27 gates the reaper). Surface it or reap it explicitly.",
                wname, pid, r.get("name") or "?",
                _human_bytes_central(r.get("vram_bytes")))
        # Drop pids that are no longer squatting so a genuine reappearance later
        # re-warns rather than staying silent forever.
        _VRAM_SQUATTER_SEEN[wname] = current & seen
    except Exception:  # noqa: BLE001 — a warning must never break a worker read
        logger.debug("vram-squatter warn dedup failed", exc_info=True)
    return rows


def _human_bytes_central(n: Any) -> str:
    """Compact byte formatter for the squatter warning (central has no
    _human_bytes; keep this local and tiny)."""
    try:
        v = float(n)
    except (TypeError, ValueError):
        return "?"
    for u in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024 or u == "TB":
            return f"{v:.1f} {u}"
        v /= 1024
    return f"{n} B"

# Spill keys that constitute a PERSISTED placement intent. If a (worker, model)
# spill carries ANY of these, it is NOT blank — the capability-aware default is
# never consulted and the persisted contract is honored verbatim (an explicit
# alloc_mode ALWAYS wins). Only a spill with none of these is "blank" and gets
# the feasibility-derived default in spill_for. Kept in sync with the mode/
# budget/band key families in managers.alloc_modes + worker_routes.
_PLACEMENT_SPILL_KEYS = frozenset({
    "alloc_mode", "n_gpu_layers", "leniency_pct", "priority", "priority_device",
    "gpu_mem_gib", "cpu_mem_gib", "gpu_mem_gib_deviation_pct",
    "cpu_mem_gib_deviation_pct", "tensor_split", "threads",
    # k67: n_cpu_moe is a placement contract (it IS the MoE split). Without it
    # here a spill persisted as {"n_cpu_moe": 999} ALONE read as "blank" at the
    # emission seam, so _placement_spill_for re-derived the split and DISCARDED
    # the operator's value before it ever reached the wire (lever-exhaustion
    # matrix, 2026-07-31). It is already in _NEW_SPILL_KEYS_LOCAL for the version
    # gate; recognizing it as placement-affecting closes that drop.
    "n_cpu_moe",
})

# Mirror of managers.alloc_modes.NEW_SPILL_KEYS, used only to decide whether a
# DERIVED spill needs the version gate at all (a ram-only derive is the legacy
# n_gpu_layers wire and every worker version honors it — gating it would
# needlessly downgrade an old worker's correct default). Kept as a literal so
# this hot read-path stays import-free; asserted equal in tests.
_NEW_SPILL_KEYS_LOCAL = frozenset({"alloc_mode", "leniency_pct",
                                   "priority_device", "n_cpu_moe"})

# alloc_mode VALUES that are PERSISTENCE-ONLY: they record an operator's choice
# in the registry but must NEVER ride the wire, because their behavior on the
# worker IS the absence of any mode key.
#
# ``max-gpu`` is the only member, and it exists because of the 2026-07-25 bug:
# max-gpu's natural encoding ({}) is indistinguishable from the "clear this
# override" signal, so an explicitly-chosen max-gpu was deleting its own row.
# It now persists as {"alloc_mode": "max-gpu"} and is stripped HERE, at the one
# seam where central hands a spill to a worker.
#
# THE STRIP IS LOAD-BEARING, not tidiness. Sending a literal
# HUGPY_ALLOC_MODE=max-gpu to a worker would change behavior for the worse:
# slot_agent's auto MoE policy bails out whenever ANY k37 alloc_mode is set
# ("the operator is driving the numbers themselves"), so an explicit max-gpu
# would SUPPRESS the automatic expert split that a blank max-gpu gets for free —
# the +59%-tok/s split, silently lost by picking the mode that means "do the
# normal thing". Every other worker branch (gguf_gpu_layers, the admission band
# engine, transformers_max_memory) tests only for max-ram/explicit and would
# ignore it, so stripping costs nothing and protects the MoE path.
_WIRE_INERT_MODES = frozenset({"max-gpu"})


def _strip_wire_inert_mode(spill: Dict[str, Any]) -> Dict[str, Any]:
    """Drop a persistence-only alloc_mode so the emitted wire is byte-identical
    to the blank ({}) encoding of the same mode. Returns a spill safe to send to
    ANY worker version; other keys (a stray ctx_pct, etc.) are preserved."""
    if not spill:
        return spill if isinstance(spill, dict) else {}
    mode = str(spill.get("alloc_mode") or "").strip().lower()
    if mode not in _WIRE_INERT_MODES:
        return spill
    out = {k: v for k, v in spill.items() if k != "alloc_mode"}
    return out


def _limit_bytes(worker: Dict[str, Any], key: str) -> Optional[int]:
    """A central limit (limits.<key> in GiB) as bytes, or None if unset."""
    v = (worker.get("limits") or {}).get(key)
    if v is None:
        return None
    try:
        return int(float(v) * _GIB)
    except (TypeError, ValueError):
        return None


def _honest_bar(physical_total, central_limit, worker_usage, external_usage):
    """Thin wrapper over spill.budget_bar so both summaries share ONE import of
    the operator's spec math. Degrades to None (legacy caller decides) on any
    import failure — the summary must never 500 the worker list."""
    try:
        from hugpy_engine.spill import budget_bar
        return budget_bar(physical_total, central_limit, worker_usage, external_usage)
    except Exception:  # noqa: BLE001 — never fail _public_view over the bar math
        return None


def _vram_summary(worker: Dict[str, Any]) -> Dict[str, Any]:
    """Flat GPU/VRAM rollup + the honest budget bar (t13/t14).

    ``vram_total``/``vram_free``/``vram_used`` stay the box-wide driver figures
    from ``gpus[]`` (unchanged — the physical truth). ON TOP, when the worker
    reports the pid_registry split (vram_attributed_bytes) this computes the
    operator's honest bar against the central GPU limit (limits.gpu_mem_gib):
    worker_usage = attributed model VRAM, external_usage = box driver-used minus
    attributed (ComfyUI + foreign + non-hugpy). bar_* fields carry bar_used/
    remaining/encroachment/over_limit + the raw figures; ``bar_semantics`` is
    "central" (a limit is set), "physical" (no limit) or "legacy" (a pre-slice
    worker that never reported the attributed split — the UI then labels it
    honestly instead of drawing a mixed-universe bar).
    All counts are bytes; ``None`` where unknown (never fabricated).
    """
    gpus = [g for g in (worker.get("gpus") or []) if isinstance(g, dict)]
    if not gpus:
        return {"gpu": None, "gpu_count": 0, "vram_total": None,
                "vram_free": None, "vram_used": None,
                "vram_bar_semantics": "legacy", "bar_semantics": "legacy"}
    name   = next((g.get("name") for g in gpus if g.get("name")), None)
    totals = [g.get("memory_total") for g in gpus if g.get("memory_total")]
    frees  = [g.get("memory_free")  for g in gpus if g.get("memory_free") is not None]
    vram_total = sum(totals) if totals else None
    vram_free  = sum(frees)  if frees  else None
    vram_used  = (vram_total - vram_free) if (vram_total is not None and vram_free is not None) else None
    out = {
        "gpu": name,
        "gpu_count": len(gpus),
        "vram_total": vram_total,
        "vram_free": vram_free,
        "vram_used": vram_used,
    }
    attributed = worker.get("vram_attributed_bytes")
    if attributed is None:
        # Pre-slice worker: no honest split available. Leave the driver figures
        # as-is and flag legacy so the UI labels the bar honestly.
        out["vram_bar_semantics"] = "legacy"
        return out
    limit = _limit_bytes(worker, "gpu_mem_gib")
    # external = whatever the driver shows used beyond hugpy's attributed models
    # (ComfyUI, foreign squatters, other apps, CUDA-context slack).
    external = (max(0, vram_used - attributed)
                if vram_used is not None else worker.get("vram_unattributed_bytes"))
    bar = _honest_bar(vram_total, limit, attributed, external)
    out.update(_bar_public_fields(bar, prefix="vram_"))
    return out


def _ram_summary(worker: Dict[str, Any]) -> Dict[str, Any]:
    """Flat RAM rollup + the honest budget bar (t13/t14) — the CPU-tier mirror.

    ``ram_total`` is the box's RAW installed memory. The old ``ram_used`` =
    ``ram_total − free_ram`` was an ARTIFACT: free_ram is ceiling-clamped, so on
    an under-budget box it algebraically collapsed to physical − central_limit
    (ae's phantom "28.9 GB used" = 124.9 − 96). The honest bar replaces it: when
    the worker reports ram_worker_bytes/ram_external_bytes this computes the
    operator's spec against the RAM ceiling (limits.ram_max_gib), and ``ram_used``
    becomes the SPEC bar_used (the fill the chip draws). bar_* fields + raw
    figures ride alongside; ``bar_semantics`` is central/physical/legacy.
    Pre-slice workers (no ram_worker_bytes) keep the OLD ram_used and are flagged
    legacy so the UI can say so rather than draw a mixed-universe bar.
    None where unknown (never fabricated); clamped ≥0.
    """
    ram_total = worker.get("ram_total")
    worker_usage = worker.get("ram_worker_bytes")
    if worker_usage is None:
        # Pre-slice worker: keep the historical (acknowledged-imperfect) figure,
        # flagged legacy. free_ram is the clamped field, matching old behavior.
        free_ram = worker.get("free_ram")
        ram_used = (max(0, ram_total - free_ram)
                    if (ram_total is not None and free_ram is not None) else None)
        return {"ram_total": ram_total, "ram_used": ram_used,
                "ram_bar_semantics": "legacy", "bar_semantics": "legacy"}
    external = worker.get("ram_external_bytes")
    limit = _limit_bytes(worker, "ram_max_gib")
    bar = _honest_bar(ram_total, limit, worker_usage, external)
    out = {"ram_total": ram_total}
    fields = _bar_public_fields(bar, prefix="ram_")
    out.update(fields)
    # ram_used IS the bar fill the chip draws (spec bar_used), so the existing
    # chip prop keeps working while gaining honest semantics.
    out["ram_used"] = fields.get("bar_used")
    return out


def _bar_public_fields(bar: Optional[Dict[str, Any]],
                       prefix: str = "") -> Dict[str, Any]:
    """Project spill.budget_bar's result onto the flat fields the console reads,
    PREFIXED so RAM and VRAM (both spread onto the same record in _public_view)
    never collide: prefix="ram_" -> ram_bar_semantics/ram_bar_used/…; prefix=
    "vram_" -> vram_bar_*. The un-prefixed generic ``bar_*`` keys are ALSO
    written for wire-compat with any caller that reads the shared names (the
    LAST summary spread wins those — RAM, applied second in _public_view). A None
    bar degrades to bar_semantics="legacy" with no numbers so the UI labels
    honestly."""
    def _keyed(d):
        # emit both the prefixed and the generic keys
        out = {}
        for k, v in d.items():
            out[f"{prefix}{k}"] = v
            out[k] = v
        return out
    if not bar:
        return _keyed({"bar_semantics": "legacy"})
    return _keyed({
        "bar_semantics": bar.get("semantics"),
        "bar_used": bar.get("bar_used"),
        "bar_total": bar.get("total"),
        "bar_remaining": bar.get("remaining"),
        "bar_raw_used": bar.get("raw_used"),
        "bar_encroachment": bar.get("encroachment"),
        "bar_over_limit": bool(bar.get("over_limit")),
        "bar_over_by": bar.get("over_by") or 0,
        "bar_worker_usage": bar.get("worker_usage"),
        "bar_external_usage": bar.get("external_usage"),
        "bar_external_headroom": bar.get("external_headroom"),
    })


def _disk_reserve_bytes(limits: Optional[Dict[str, Any]] = None) -> int:
    """Free-space reserve (bytes) kept on a worker's MODEL-ROOT volume.

    Resolution: per-worker ``limits.disk_reserve_gib`` -> env
    ``HUGPY_WORKER_DISK_RESERVE_GIB`` -> 50. Mirrors worker_agent/budget.py
    ``disk_reserve_bytes`` exactly (Parity). Carved OUT of disk_cache_gib.

    Below this reserve a worker is "over budget" and its COLD local models become
    eviction candidates. Sized to comfortably exceed the largest single model
    pull (~45 GiB per the model_cache header note) so a provision can always land
    after one eviction. Override with ``HUGPY_WORKER_DISK_RESERVE_GIB`` (default
    50). This is DISTINCT from ``HUGPY_MODEL_CACHE_MAX_GIB`` (=450, the separate
    SSD hot-cache bound in managers/serve/model_cache.py) — do not conflate: this
    reserve is on the model-root disk, not the SSD cache.
    """
    gib = None
    v = (limits or {}).get("disk_reserve_gib") if isinstance(limits, dict) else None
    if v not in (None, ""):
        try:
            gib = float(v)
        except (TypeError, ValueError):
            gib = None
    if gib is None:
        try:
            gib = float(os.environ.get("HUGPY_WORKER_DISK_RESERVE_GIB", "50"))
        except (TypeError, ValueError):
            gib = 50.0
    if gib < 0:
        gib = 0.0
    return int(gib * (1 << 30))


def _registry_row(model_key: str) -> Optional[Dict[str, Any]]:
    """The model's registry row (a COPY), or None when central doesn't know it."""
    try:
        # Import depths differ and are NOT interchangeable: this module sits at
        # flask_app/app/functions/imports/utils/, so `functions` is 3 up while
        # the TOP-LEVEL `imports` package (abstract_hugpy_dev.imports — a
        # different tree from this one's own `imports` parent) is 6. Getting
        # this wrong raises ModuleNotFoundError, which an over-broad except
        # would swallow into a permanent "size unknown" — every model silently
        # unsized, an over-subscribed set reading as empty. Logged loudly.
        from hugpy_engine.config.models.models_config import get_models_dict
    except Exception as exc:  # noqa: BLE001 — sizing must never break a read
        logger.warning("allocation sizing unavailable (%s) — assigned-set totals "
                       "will report as unknown", exc)
        return None
    try:
        entry = (get_models_dict(dict_return=True) or {}).get(model_key)
    except Exception:  # noqa: BLE001
        return None
    # A COPY: the physical helpers are handed a row that must not be the cached,
    # shared MODEL_REGISTRY_DICT object the listings mutate in place.
    return dict(entry) if isinstance(entry, dict) else None


# ── cold-fill budget for the worker view ──────────────────────────────────
# _public_view is on TWO hot paths: /llm/workers (the console polls it
# continuously) and the HEARTBEAT REPLY. Deriving a missing physical record is
# the honest fallback — but doing it for ~111 designated models inside one call
# is a multi-minute walk, and a heartbeat that slow blows
# HEARTBEAT_TIMEOUT_SECONDS and makes the whole fleet read offline. That is the
# "pushes off all of the workers" failure, re-entered through the back door.
#
# So a single view may spend a BOUNDED amount of time filling cold rows; past
# that it serves what is persisted and reports the rest as UNKNOWN — which this
# surface already renders honestly (allocated_unknown_count, a None size, no
# lever). Unknown-and-filling beats a view that never renders. Successive polls
# (and any /models listing, or the /models/discover sweep) finish the fill.
# Set HUGPY_WORKER_VIEW_FILL_BUDGET_S <= 0 to derive without a bound.
_VIEW_FILL_BUDGET_S = 1.5
_view_fill = threading.local()


def _view_fill_budget_seconds() -> float:
    raw = (os.environ.get("HUGPY_WORKER_VIEW_FILL_BUDGET_S") or "").strip()
    if not raw:
        return _VIEW_FILL_BUDGET_S
    try:
        return float(raw)
    except ValueError:
        return _VIEW_FILL_BUDGET_S


@contextmanager
def _view_fill_window():
    """Bound cold-record filling for the duration of ONE worker view.

    Scoped, not just started: the budget MUST be cleared on the way out. These
    helpers are also called from DECISION paths (assigning a model, deriving an
    allocation for a load), and a decision must never be answered "size unknown"
    because a listing on the same gunicorn thread happened to run out of budget
    first. Outside a window there is no budget at all — decisions derive."""
    budget = _view_fill_budget_seconds()
    _view_fill.deadline = None if budget <= 0 else (time.time() + budget)
    try:
        yield
    finally:
        _view_fill.deadline = None


def _may_derive() -> bool:
    """May this call still afford to derive a missing record from the store?"""
    deadline = getattr(_view_fill, "deadline", None)
    return deadline is None or time.time() < deadline


def _canonical_registry_key(model_key: str) -> str:
    """Resolve a ``Owner~Repo`` spelling to its canonical registry key.

    "~"-TAIL UNIFICATION, completing the doctrine (k67, 2026-07-23). Central keys
    a name-collision family as ``Owner~Repo`` while the bare ``Repo`` is the SAME
    model; routing already treats the two as one (``_match_keys``,
    ``assure_model_key``) — but the size/marker resolvers below did NOT. So a
    worker carrying the ``Owner~Repo`` spelling of an assignment (e.g. a stale
    ``Qwen~Qwen3-Coder-Next-GGUF`` against the registry's bare
    ``Qwen3-Coder-Next-GGUF``) resolved to NO registry row, and every size / MoE
    / marker fact came back UNKNOWN — the MoE expert-split lever vanished from the
    console and the split degraded to all-experts-in-RAM (slow). This maps the
    "~"-qualified key back to its canonical twin so those facts resolve again.

    DETERMINISTIC ONLY: an exact registry hit, else a bare-tail match against the
    registry keys. Never the fuzzy name pipeline — an unknown key must stay
    unknown here (a size read that fuzzy-binds to an unrelated model is worse than
    an honest UNKNOWN)."""
    if not model_key:
        return model_key
    try:
        from hugpy_engine.config.models.models_config import get_models_dict
        md = get_models_dict(dict_return=True) or {}
    except Exception:  # noqa: BLE001 — key resolution must never break a read
        return model_key
    if model_key in md or "~" not in str(model_key):
        return model_key
    try:
        from hugpy_engine.resolvers.assure_model_key import _bare_tail, _slugify
    except Exception:  # noqa: BLE001 — no resolver, keep the raw key (fail-open)
        return model_key
    want = _slugify(_bare_tail(model_key))
    for k in md:
        if _slugify(_bare_tail(k)) == want:
            return k
    return model_key


def _model_physical(model_key: str) -> Dict[str, Any]:
    """One model's PERSISTED size half — a dict lookup, not a store walk.

    THE fix for /llm/workers (2026-07-27). This used to run
    ``_annotate_gguf_size`` + ``_annotate_size`` — a recursive listing of every
    servable .gguf plus an ``os.walk`` of the model dir — and every caller below
    called it again: ``allocated_totals``, ``planned_split``,
    ``derived_default_mode`` and ``derived_default_allocation`` each want a size
    and/or the MoE structure, PER DESIGNATED MODEL. ae carries 75 designations,
    ~111 across the fleet, so building a view of THREE machines cost several
    hundred recursive walks of a spinning array over virtiofs: measured 11.8s
    cold / 5.3s warm for ``list_workers()``, 31.0s for ``GET /llm/workers``
    under the console's continuous polling — the workers view never rendered.

    Same numbers, same source, from the record central already wrote at the
    events that change it (comms/model_physical.py). Empty dict when central
    cannot say — callers must keep treating that as UNKNOWN, never as zero."""
    if not model_key:
        return {}
    model_key = _canonical_registry_key(model_key)   # "~"-tail unification (k67)
    row = _registry_row(model_key)
    if row is None:
        return {}
    try:
        from hugpy_storage.model_physical import ASPECT_SIZE, lookup_physical
        fields, state = lookup_physical(model_key, row, ASPECT_SIZE)
        if state == "fresh":
            return fields or {}
        if not _may_derive():
            # Out of cold-fill budget: serve what is persisted (possibly an
            # expired-but-real record) and otherwise say UNKNOWN. Never a zero.
            return fields or {}
        from hugpy_storage.console.model_physical import size_fields
        return size_fields(row, model_key, source="workers") or {}
    except Exception:  # noqa: BLE001 — unknown size is a valid answer here
        return {}


def _model_size_bytes(model_key: str) -> Optional[int]:
    """One ASSIGNED model's size per central's manifest, or None if unknowable.

    The same source ``worker_agent/provision.central_total_bytes`` resolves for a
    single pull, read LOCALLY here (central owns the manifest and the model dirs,
    so this needs no HTTP — see allocated_totals for why that matters).

    GGUF honesty: a GGUF dir holds SEVERAL quants, so its directory sum is NOT
    what serving costs. ``effective_bytes`` (gguf_variants_detail) is the quant
    that actually serves — the same number the Models tab shows. Falls back to
    the directory footprint for transformers/comfy.

    None is a FIRST-CLASS answer meaning "central cannot say" (not in the
    manifest / not on disk / sizing raised). Callers MUST count it as unknown and
    report it — never coerce it to 0, which would make an over-subscribed
    assignment set read as comfortably fitting (the exact dishonesty this
    feature exists to remove).
    """
    size = _model_physical(model_key).get("size_bytes")
    try:
        return int(size) if size else None
    except (TypeError, ValueError):
        return None


def _model_moe_gpu_bytes(model_key: str) -> Optional[int]:
    """The GPU-side need of a detected-MoE GGUF under the expert split (its
    non-expert bytes + mmproj), or None for dense/non-GGUF/unresolvable.

    Same source as _model_size_bytes: the model's PERSISTED size half, whose
    ``moe`` field rides gguf_variants_detail exactly like effective_bytes — so
    this costs a dict lookup, never a per-request header re-parse. Feasibility
    uses it so a MoE the split makes serveable is never eliminated against its
    full file size."""
    try:
        fields = _model_physical(model_key)
        moe = fields.get("moe") or {}
        nexp = moe.get("non_expert_bytes")
        if not nexp:
            return None
        return int(nexp) + int(fields.get("mmproj_bytes") or 0)
    except Exception:  # noqa: BLE001 — MoE sizing is additive; unknown is fine
        return None


def _model_moe_detail(model_key: str) -> Optional[Dict[str, Any]]:
    """The MoE STRUCTURE of a GGUF model — ``{is_moe, expert_bytes,
    non_expert_bytes, ...}`` — or None for dense/non-GGUF/unresolvable.

    Same enrichment source as ``_model_moe_gpu_bytes`` (the registry row's
    ``moe`` field, which rides gguf_variants_detail; spill.gguf_moe_detail
    caches the header parse per file, so this costs no extra I/O). Where that
    helper flattens the structure to ONE number for the feasibility eliminator,
    this returns the whole detail: deriving the split needs BOTH sides (the
    non-expert share to price the GPU, the expert share to price RAM).

    None is a first-class "central cannot say" — the caller must degrade to the
    dense path, never guess a split."""
    try:
        moe = _model_physical(model_key).get("moe")
        if not isinstance(moe, dict) or not moe.get("is_moe"):
            return None
        return moe
    except Exception:  # noqa: BLE001 — unknown structure degrades to dense
        return None


def _model_engine(model_key: str) -> Optional[str]:
    """One model's engine/framework from central's registry, lowercased, or None
    when unresolvable. Used to pick the FEASIBLE blank default (GGUF is always
    max-gpu; only a non-GGUF model can default to ram-only). None -> the caller
    degrades to max-gpu (never guess an engine)."""
    if not model_key:
        return None
    try:
        from hugpy_engine.config.main import get_model_config
        cfg = get_model_config(model_key)
        fw = getattr(cfg, "framework", None)
        return str(fw).lower() if fw else None
    except Exception:  # noqa: BLE001 — unknown engine: caller treats as unknown
        return None


# Durable hardware-total fields (operator addendum 2026-07-24): a box's GPU and
# RAM CAPACITY are physical FACTS, not per-session state — they don't change
# between a worker's restarts. Central persists the last KNOWN-GOOD reading per
# worker identity in these fields (mirrors the auto_reap/wildcard durable-field
# pattern: a plain persisted key that register/heartbeat leave intact). They are
# ADVANCE-ONLY from a real reading — a transient empty/partial gpu probe (driver
# not ready right after boot; detect_gpus() returning []) NEVER wipes them — so
# feasibility has totals for every worker central has ever met, and the
# fail-open collapses to genuinely first-contact-only. _remember_hw_totals
# updates them; the *_total_bytes readers prefer the live figure and fall back
# to the durable one.
_GPU_TOTAL_DURABLE_KEY = "gpu_total_bytes_known"
_RAM_TOTAL_DURABLE_KEY = "ram_total_bytes_known"


def _live_gpu_total_bytes(worker: Dict[str, Any]) -> Optional[int]:
    """GPU total from the CURRENT gpus[] reading only (no durable fallback)."""
    gpus = [g for g in (worker.get("gpus") or []) if isinstance(g, dict)]
    totals = [g.get("memory_total") for g in gpus if g.get("memory_total")]
    if not totals:
        return None
    try:
        return int(sum(totals))
    except (TypeError, ValueError):
        return None


def _remember_hw_totals(worker: Dict[str, Any]) -> None:
    """Persist the worker's last KNOWN-GOOD GPU/RAM totals as durable facts.
    ADVANCE-ONLY: called after register/heartbeat merges the new reading, it
    copies a real (non-None, >0) live total into the durable key and NEVER
    overwrites a good durable value with an absent/zero one — so a transient
    probe miss can't erase a fact central already learned. Idempotent + cheap;
    safe to call on every register/heartbeat inside the transaction."""
    live_gpu = _live_gpu_total_bytes(worker)
    if live_gpu:
        worker[_GPU_TOTAL_DURABLE_KEY] = live_gpu
    ram = worker.get("ram_total")
    try:
        ram = int(ram) if ram else None
    except (TypeError, ValueError):
        ram = None
    if ram:
        worker[_RAM_TOTAL_DURABLE_KEY] = ram


def _worker_gpu_total_bytes(worker: Dict[str, Any]) -> Optional[int]:
    """Box-wide TOTAL GPU capacity (bytes) = sum of gpus[].memory_total, the same
    physical-truth source _vram_summary reads — falling back to the DURABLE
    last-known total (_GPU_TOTAL_DURABLE_KEY) when the current reading is missing
    (a re-register window, or a transient empty probe). None only when central
    has NEVER seen a total for this worker (true first contact -> caller degrades
    to max-gpu, never derives ram-only blind)."""
    live = _live_gpu_total_bytes(worker)
    if live:
        return live
    durable = worker.get(_GPU_TOTAL_DURABLE_KEY)
    try:
        return int(durable) if durable else None
    except (TypeError, ValueError):
        return None


def _worker_ram_total_bytes(worker: Dict[str, Any]) -> Optional[int]:
    """The box's RAW installed memory (bytes) — the ``ram_total`` field
    _ram_summary reads, falling back to the DURABLE last-known RAM total when the
    current field is absent (re-register window / older beat). None only when
    central has never seen a RAM total for this worker."""
    for key in ("ram_total", _RAM_TOTAL_DURABLE_KEY):
        v = worker.get(key)
        if v is None:
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv:
            return iv
    return None


def _model_marker_flag(model_key: str, field: str) -> Optional[bool]:
    """Read a capability bool straight off the model's hugpy.json marker.

    Returns None when the model, its directory or the field cannot be resolved —
    "never determined", which every caller must treat as unknown and fall back
    on, never as False.

    Reads the PERSISTED marker aspect (comms/model_physical.py). It looked cheap
    — "a single small JSON read … only consulted on per-model surfaces, not per
    request" — but /llm/workers consults it TWICE per designated model
    (moe_capable + bnb_available), which on ae is 150 dir resolutions plus 150
    JSON reads off a spinning array over virtiofs, on every console poll. The
    blob is derived once at the events that change it and read as a dict here."""
    if not model_key:
        return None
    model_key = _canonical_registry_key(model_key)   # "~"-tail unification (k67)
    row = _registry_row(model_key)
    if row is None:
        return None
    try:
        from hugpy_storage.model_physical import ASPECT_MARKER, lookup_physical
        fields, state = lookup_physical(model_key, row, ASPECT_MARKER)
        if state != "fresh":
            if not _may_derive():
                fields = fields or {}    # budget spent — persisted or unknown
            else:
                from hugpy_storage.console.model_physical import marker_fields
                fields = marker_fields(row, model_key)
    except Exception:  # noqa: BLE001 — a marker read must never break a view
        return None
    if not fields or "hugpy_marker" not in fields:
        return None                      # never determined -> caller falls back
    marker = fields.get("hugpy_marker") or {}
    val = marker.get(field) if isinstance(marker, dict) else None
    return None if val is None else bool(val)


def planned_split(worker: Dict[str, Any], model_key: str) -> Dict[str, Any]:
    """The INTENDED VRAM/RAM division for this (worker, model) — what the current
    allocation, 4-bit lever and MoE lever ADD UP TO before anything is loaded.

    Operator, 2026-07-26: the Memory column "was meant to display the split or
    overall resource allocation, i.e. vram and ram; this in implementation should
    change with the selected switches being switched and/or the alloc being
    changed." Measured residency only exists once a model is resident, so an
    idle row fell back to the on-disk size — the same number the Size column
    already shows, which is the redundancy.

    Returns ``{"gpu_bytes", "ram_bytes", "size_bytes", "mode", "why"}``. Derived
    from the SAME allocation the worker will actually receive, so the projection
    cannot drift from the placement it describes:
      * explicit (the MoE split) — the two budgets it already carries;
      * ram-only  — everything in RAM;
      * gpu-only  — everything in VRAM;
      * max-gpu / max-ram — a spill, so the division is decided at load time
        against whatever is free; reported as the whole size on the PREFERRED
        side with split=False, never a fabricated ratio.
    size_bytes reflects the 4-bit lever, so ticking it visibly shrinks the row."""
    out = {"gpu_bytes": None, "ram_bytes": None, "size_bytes": None,
           "mode": None, "split": False}
    try:
        from hugpy_engine.alloc_modes import bnb_effective_bytes
        size = _model_size_bytes(model_key)
        if size and bnb_enabled(worker, model_key):
            size = bnb_effective_bytes(size) or size
        out["size_bytes"] = size
        d = derived_default_allocation(worker, model_key) or {}
        mode = d.get("mode")
        out["mode"] = mode
        spill = d.get("spill") or {}
        gib = float(2 ** 30)
        g, c = spill.get("gpu_mem_gib"), spill.get("cpu_mem_gib")
        if g is not None or c is not None:
            # The MoE leaf: both sides are priced, so this is a REAL split.
            out["gpu_bytes"] = int(float(g) * gib) if g is not None else 0
            out["ram_bytes"] = int(float(c) * gib) if c is not None else 0
            out["split"] = True
        elif mode == "ram-only":
            out["ram_bytes"] = size
        elif mode == "gpu-only":
            out["gpu_bytes"] = size
        elif mode == "max-gpu":
            out["gpu_bytes"] = size      # preferred side; spills what won't fit
        elif mode == "max-ram":
            out["ram_bytes"] = size
    except Exception:  # noqa: BLE001 — a projection must never break a read
        pass
    return out


def moe_capable(model_key: str) -> bool:
    """Does this model HAVE an expert structure? (the marker's moe_capable)

    Reads the durable hugpy.json bool first — it answers for models this box has
    never opened, and for transformers MoE, which the GGUF header reader cannot
    see at all. Falls back to the live header parse when the marker predates the
    flag, so an unstamped model is never wrongly called dense."""
    flag = _model_marker_flag(model_key, "moe_capable")
    if flag is not None:
        return bool(flag)
    d = _model_moe_detail(model_key) or {}
    return bool(d.get("is_moe"))


def moe_override(worker: Dict[str, Any], model_key: str) -> Optional[bool]:
    """The operator's MoE-split override for this (worker, model), or None.

    THREE STATES, and the third is the default (operator, 2026-07-26: "defaults
    can remain auto and should, but that also should entail a checked box under
    the correct column, that could be switched by the user"):
      * None  — AUTO: follow the derivation. The console still shows the box
        TICKED when the derivation produced a split, so the effective state is
        always visible rather than hidden behind "unset".
      * True  — force the split on.
      * False — force it off (the escape hatch when a split misbehaves).
    Absent/garbage reads as None, i.e. auto — a malformed row can never pin a
    placement the operator did not choose."""
    try:
        v = (worker.get("moe_by_model") or {}).get(str(model_key))
        return None if v is None else bool(v)
    except Exception:  # noqa: BLE001
        return None


def moe_effective(worker: Dict[str, Any], model_key: str) -> bool:
    """Whether a split IS in force for this (worker, model) — what the checkbox
    renders. The override when the operator set one, else what the derivation
    actually produced (n_cpu_moe present in the derived spill)."""
    ov = moe_override(worker, model_key)
    if ov is not None:
        return ov
    try:
        spill = (derived_default_allocation(worker, model_key) or {}).get("spill") or {}
        return spill.get("n_cpu_moe") is not None
    except Exception:  # noqa: BLE001
        return False


def apply_moe_override_to_spill(worker: Dict[str, Any], model_key: str,
                                spill: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay the operator's MoE-split checkbox onto an already-resolved spill,
    IN PLACE, and return it.

    The MoE split is an axis ORTHOGONAL to the alloc mode — exactly like
    ``bnb_enabled`` (see its docstring), and stored in its own ``moe_by_model``
    map, never in ``spill_by_model``. ``_placement_spill_for``'s blank branch
    already threads it via ``derived_default_allocation(moe_force=...)``, but the
    PERSISTENCE-ONLY branch (any placement key pinned) used to return the raw
    persisted spill and NEVER consult ``moe_override`` — so the console rendered
    the checkbox from ``moe_effective`` (which reads the override) while the
    emitted wire carried no ``n_cpu_moe`` for any pinned model, and toggling MoE
    changed neither allocation nor memory (serving-core mechanics §8 / FINDINGS
    C2; live repro: coder-next on aeb, ``spill_by_model={'alloc_mode':'max-gpu'}``
    + ``moe_by_model=True``). This is the seam that closes that drop.

      * True  — force the split ON: fold in the DERIVED split's keys (``n_cpu_moe``
        + the load-bearing ``n_gpu_layers=-1`` + the two budgets) so the wire is
        byte-identical to a blank MoE's derived default. A model with no
        priceable/fitting expert structure yields no ``n_cpu_moe`` from the
        derivation, so nothing is fabricated (mirrors ``default_allocation``:
        ``moe_force=True`` can't manufacture a split). An operator pin that
        already carries a real split is left as-is.
      * False — force the split OFF: emit an explicit ``n_cpu_moe = 0``. It is
        read BEFORE the worker's auto-split branch (``slot_agent._build_cmd`` /
        the worker's own policy), so it wins regardless of the pinned mode — the
        escape hatch when a derived split misbehaves. Applied only when there is
        a split to suppress (MoE-capable, or a stale ``n_cpu_moe`` already
        persisted) so a dense model's wire is not burdened with a gated key.
      * None  — AUTO: leave the persisted spill's own ``n_cpu_moe`` untouched.
    """
    try:
        ov = moe_override(worker, model_key)
        if ov is None:
            return spill
        if ov is False:
            if "n_cpu_moe" in spill or moe_capable(model_key):
                spill["n_cpu_moe"] = 0        # explicit: no experts to CPU
            return spill
        # ov is True — force ON.
        if spill.get("n_cpu_moe"):            # already a real split on the wire
            return spill
        derived = (derived_default_allocation(worker, model_key)
                   or {}).get("spill") or {}
        ncm = derived.get("n_cpu_moe")
        if not ncm:                           # no feasible split to force on
            return spill
        spill["n_cpu_moe"] = ncm
        spill.setdefault("n_gpu_layers", derived.get("n_gpu_layers", -1))
        for k in ("gpu_mem_gib", "cpu_mem_gib"):
            if derived.get(k) is not None:
                spill.setdefault(k, derived[k])
        # t143: the split IS the allocation — carry the derived mode too (only
        # when the pin carries none, e.g. a stripped max-gpu), so the wire is
        # byte-identical to a blank MoE's derived default and the console's
        # Alloc cell (model_alloc_modes -> derive_alloc_mode) reads "explicit"
        # instead of the pinned label the toggle just overrode.
        if derived.get("alloc_mode"):
            spill.setdefault("alloc_mode", derived["alloc_mode"])
        return spill
    except Exception:  # noqa: BLE001 — the MoE overlay must never break the relay
        logger.debug("moe override overlay skipped for %s on %s", model_key,
                     (worker or {}).get("name"), exc_info=True)
        return spill


def suppress_moe_split_for_gpu_only(worker: Dict[str, Any], model_key: str,
                                    spill: Dict[str, Any]) -> Dict[str, Any]:
    """gpu-only means EVERYTHING resident on the GPU — a MoE's experts INCLUDED.
    Fold in an explicit ``n_cpu_moe = 0`` so the worker's own auto-MoE policy
    cannot offload experts to CPU behind the operator's back. IN PLACE; returns
    ``spill``.

    THE GAP THIS CLOSES. gpu-only's wire encoding is ``{"n_gpu_layers": -1}``
    (``alloc_modes.mode_to_spill``) — it carries NO ``n_cpu_moe``. On the worker
    ``slot_agent._build_cmd`` reads ``n_cpu_moe`` BEFORE its auto branch, so a
    bare ``-1`` with no knob leaves that auto branch free to add an expert split
    on top: the +59%-tok/s policy that is exactly right for a model that does
    NOT fit whole, and exactly WRONG for one pinned/derived gpu-only. Result
    (live: Qwen3.6-27B-…-MTP-i1-GGUF on aeb, ``spill_by_model={'n_gpu_layers':
    -1}``, moe AUTO): a gpu-only MoE that FITS VRAM was still expert-SPLIT,
    experts on CPU, violating "gpu-only = all on GPU". ``apply_moe_override_to_spill``
    closes this ONLY for the explicit toggle (force-OFF -> n_cpu_moe:0); an
    AUTO/None MoE fell straight through — this is that AUTO counterpart, keyed
    off the alloc MODE, and it composes with (never overrides) the toggle: it is
    invoked AFTER that helper and yields the moment any ``n_cpu_moe`` was already
    decided, so an explicit force-ON split and an explicit force-OFF 0 both WIN
    (the deliberate operator contradiction is left to the operator), a persisted
    ``n_cpu_moe`` contract is honored, and only the AUTO gap is filled.

    FIRES ONLY when the RESOLVED spill is genuinely ``gpu-only``
    (``derive_alloc_mode``) AND the model is MoE-capable AND no ``n_cpu_moe`` is
    already on the wire — so ram-only / max-gpu / max-ram / an explicit derived
    MoE split (derive == "explicit") are untouched, and a dense model's wire is
    never burdened with a gated key it doesn't need.

    FIT-SAFE BY CONSTRUCTION — no OOM guard belongs here. gpu-only is ONLY ever
    designated/derived when the FULL footprint fits the card: the feasible-modes
    gpu-only leaf prices the whole MoE (experts on the card too), and the
    derivation's gpu-only leaf is guarded to fits-whole. So "all experts on GPU"
    is exactly what the mode already promised; n_cpu_moe:0 only makes that
    promise reach the loader. A model that does NOT fit VRAM is ram-only / an
    explicit split — never gpu-only — so this never fires there; the
    "won't-fit -> keep experts on CPU" case belongs to those paths, not here."""
    try:
        if "n_cpu_moe" in spill:              # a split/no-split already decided
            return spill
        if not moe_capable(model_key):
            return spill
        from hugpy_engine.alloc_modes import derive_alloc_mode
        if derive_alloc_mode(spill) != "gpu-only":
            return spill
        spill["n_cpu_moe"] = 0                # explicit: keep ALL experts on GPU
        return spill
    except Exception:  # noqa: BLE001 — the suppression must never break the relay
        logger.debug("gpu-only MoE suppression skipped for %s on %s", model_key,
                     (worker or {}).get("name"), exc_info=True)
        return spill


def bnb_enabled(worker: Dict[str, Any], model_key: str) -> bool:
    """Is the bitsandbytes SPECIALIZATION switched on for this (worker, model)?

    Operator lever (2026-07-26), persisted per worker as
    ``worker["bnb_by_model"][model_key] = true``. Deliberately NOT part of
    spill_by_model: a spill is a PLACEMENT contract, this is a COMPRESSION
    choice, and conflating them would make "revert to the derived placement"
    silently drop the quantization too. Absent/garbage reads as OFF — the lever
    is an explicit opt-in, never something a malformed row can switch on."""
    try:
        return bool((worker.get("bnb_by_model") or {}).get(str(model_key)))
    except Exception:  # noqa: BLE001
        return False


def bnb_available(worker: Dict[str, Any], model_key: str) -> bool:
    """Whether the lever should be OFFERED for this (worker, model).

    Gates on engine (never GGUF — llama.cpp carries its own quantization), the
    worker actually having CUDA (bitsandbytes' 4-bit kernels are CUDA-only, so
    offering it on op would be a promise that fails at load), and the repo not
    already being quantized."""
    try:
        from hugpy_engine.alloc_modes import bnb_eligible
        has_cuda = bool((worker.get("gpus") or [])
                        or _worker_gpu_total_bytes(worker))
        if not has_cuda:
            return False          # CUDA-only kernels; nothing else can override
        # THE MARKER IS AUTHORITATIVE when it has been stamped (operator
        # 2026-07-26): hugpy.json records `bnb_capable` as a structural fact
        # (framework + an existing quantization_config), which beats
        # bnb_eligible's fallback name-matching — a repo whose name happens not
        # to say "awq"/"4bit" is still correctly excluded, and one that merely
        # LOOKS quantized is not wrongly excluded. Absent/None means "never
        # determined" (older markers) and falls through to the heuristic.
        flag = _model_marker_flag(model_key, "bnb_capable")
        if flag is not None:
            return bool(flag)
        return bnb_eligible(_model_engine(model_key), model_key,
                            has_cuda=has_cuda)
    except Exception:  # noqa: BLE001 — never break a read over a capability probe
        return False


def derived_default_mode(worker: Dict[str, Any], model_key: str) -> str:
    """The FEASIBLE blank default alloc mode for one (worker, model) — engine +
    box-totals aware (operator ruling 2026-07-24). Pure glue over the stdlib
    ``feasible_default_mode``: resolves the engine, effective size, GPU total,
    RAM total from central's authoritative sources and asks the shared math.
    ANY lookup miss degrades to 'max-gpu' (today's blank behavior) — never a
    500, never a guess. This ONLY supplies the default when NOTHING is persisted
    for the model; an explicit alloc_mode always wins upstream."""
    try:
        from hugpy_engine.alloc_modes import feasible_default_mode
        return feasible_default_mode(
            _model_engine(model_key),
            _model_size_bytes(model_key),
            _worker_gpu_total_bytes(worker),
            _worker_ram_total_bytes(worker),
            moe=_model_moe_detail(model_key),
            bnb=bnb_enabled(worker, model_key),
            moe_force=moe_override(worker, model_key))
    except Exception:  # noqa: BLE001 — a derivation must never break a read/relay
        return "max-gpu"


def derived_default_allocation(worker: Dict[str, Any],
                               model_key: str) -> Dict[str, Any]:
    """The full DERIVED initial allocation for one (worker, model) —
    ``{"mode", "spill", "why"}`` — per the operator's decision tree
    (2026-07-25). The allocation-shaped sibling of ``derived_default_mode``:
    that one returns the NAME for surfaces, this one returns the WIRE ENCODING
    for the places that PERSIST an allocation, so a MoE's derived split
    (n_cpu_moe + the budgets) survives instead of being flattened to a name.

    Pure glue over ``managers.alloc_modes.default_allocation``: resolves engine,
    effective size, MoE structure, and the box's measured GPU/RAM totals from
    central's authoritative sources and asks the shared math. ANY lookup miss
    degrades to max-gpu / {} — never a 500, never a guess."""
    try:
        from hugpy_engine.alloc_modes import default_allocation
        return default_allocation(
            _model_engine(model_key),
            _model_size_bytes(model_key),
            _worker_gpu_total_bytes(worker),
            _worker_ram_total_bytes(worker),
            moe=_model_moe_detail(model_key),
            bnb=bnb_enabled(worker, model_key),
            moe_force=moe_override(worker, model_key))
    except Exception:  # noqa: BLE001 — a derivation must never break a read/relay
        return {"mode": "max-gpu", "spill": {},
                "why": "derivation unavailable — kept the max-gpu default"}


def allocated_totals(worker: Dict[str, Any]) -> Dict[str, Any]:
    """Size the worker's ASSIGNMENT SET against its budget — the STRUCTURAL view.

    OPERATOR (2026-07-16): "it should also show how much is needed based on the
    total size of all models allocated". The per-pull refusal ("this 23.5 GiB
    pull won't fit") answers a different, smaller question. This answers: can the
    ASSIGNED SET fit AT ALL? A worker assigned 12 models totalling 180 GiB
    against a 50 GiB budget is over-subscribed BY CONSTRUCTION — no eviction
    order rescues it, and it will wedge on some future call no matter which model
    is unlucky enough to be the one that asks.

    Domain = ``worker['models']`` — the OPERATOR DESIGNATION set written by
    assign_model/unassign_model. NOT the on-disk inventory (lazy-download means
    an assigned model routinely has no files yet — sizing only what landed would
    UNDER-report an over-subscribed set, hiding the very thing this shows) and
    NOT ``grants`` (system-authored, freely evictable, never operator intent).

    Returns::

        {"allocated_total_bytes": int,      # sum of the KNOWN-size models
         "allocated_count": int,            # models in the assignment set
         "allocated_unknown_count": int,    # sizes central couldn't resolve
         "allocated_over_budget_bytes": int}  # total - budget, 0 when it fits

    ``allocated_total_bytes`` is a FLOOR when allocated_unknown_count > 0: the
    unknowns are counted and surfaced, never silently zeroed, so a reader can see
    the number is incomplete rather than trust a comfortable-looking lie.
    """
    models = [m for m in (worker.get("models") or []) if m]
    total = 0
    unknown = 0
    for mk in models:
        size = _model_size_bytes(mk)
        if size:
            total += size
        else:
            unknown += 1
    return {"allocated_total_bytes": total,
            "allocated_count": len(models),
            "allocated_unknown_count": unknown}


def _row_store_class(m: Dict[str, Any]) -> str:
    """Which STORE a worker's storage row sits on: ``shared`` (the fleet's
    central catalog this box only reads through), ``unreapable`` (a store the
    box never declared local & disposable), or ``reapable`` (the worker's own
    evictable cache). k60, operator 2026-07-31.

    A CURRENT worker stamps ``store`` on the row. A released worker
    (<=0.1.226) doesn't, so fall back to the store-gate ``why`` verdict it has
    always sent — that fallback is what makes the accounting fix land on the
    fleet at the next central restart instead of waiting on a wheel roll.
    """
    store = str(m.get("store") or "").strip().lower()
    if store in ("shared", "unreapable", "reapable"):
        return store
    why = str(m.get("why") or "")
    if "shared/central storage" in why:
        return "shared"
    if "model store not marked reapable" in why:
        return "unreapable"
    return "reapable"


def _row_counts_toward_budget(m: Dict[str, Any]) -> bool:
    """True when a storage row's bytes belong in the worker's eviction economy.

    Shared-catalog and never-opted-in rows are SHOWN but never priced: they can
    never be deleted from here, so charging them against the worker budget
    manufactures a permanent "over budget" — which reads to an operator as "an
    auto-delete is coming" (the ae 2.8 TiB / 800 GB alarm, 2026-07-31)."""
    if "counts_toward_budget" in m:
        return bool(m.get("counts_toward_budget"))
    return _row_store_class(m) == "reapable"


def storage_proposal(worker: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a worker's local-STORAGE view + a guarded LRU eviction PROPOSAL.

    A PURE read-time computation over already-stored heartbeat fields — no
    daemon, no background loop, no persistent toggle, and THIS FUNCTION never
    deletes anything (it returns a proposal; a caller must act on it). It is
    spread into every worker read by ``_public_view`` (the always-on storage
    monitoring depiction) and re-run by the ``/reap-approve`` route as its
    central second guard, so the console preview and the approval share one
    source of truth.

    NOTE (2026-07-16): "no auto-fire" describes THIS central preview and the
    operator-gated bulk reaper it feeds — it is NOT a fleet-wide claim. The
    worker's ``worker_agent/budget.py`` auto-evicts on the PROVISION path
    (call-driven only) to seat a model being pulled. That path deliberately
    reuses THIS function's ordering + guard semantics (unprotected candidates,
    ascending last_picked, largest-first among equally-cold) so the console's
    preview and an auto-evict can never disagree about what would go.

    Inputs (all raw worker-record fields):
      - ``worker['storage']``   the worker-reported survey
            ``{cache_used_bytes, disk_free, models:[{model_key, bytes, pinned,
            loaded, loading, provisioning, assigned, protected, why}]}``.
            ABSENT on a pre-feature agent -> a monitoring-only view with an empty
            proposal (the worker must ship this field for the proposal to have a
            per-model inventory).
      - ``worker['disk']``      ``{free_bytes, total_bytes}`` of the model root.
      - ``worker['model_last_picked']`` central LRU signal ``{model_key: epoch}``
            stamped in ``pick_for_model``. A missing entry defaults to 0 (coldest
            -> proposed for eviction first — exactly right for never-served
            test-churn leftovers).
      - ``worker['limits']['disk_cache_gib']`` optional explicit per-worker cap;
            WINS over the free-disk reserve when set.
      - ``worker['loaded_models']`` / ``['loading']`` / ``['provisioning']``
            central slot-merged live truth — the redundant central guard that
            closes the worker reaper's in-process-only loaded gap.
      - ``worker['config']['residency']`` / ``['pinned']`` static/pin attribution.

    Budget (two modes, cheap; explicit cap wins):
      * explicit cap  -> budget = cap - reserve (reserve carved OUT of the
                         allocation); over_budget ``cache_used > budget`` OR
                         ``disk_free < reserve``; need = the larger shortfall
      * else reserve  -> over_budget ``disk_free < reserve``; need ``reserve-disk_free``

    Proposal (mirrors ``model_cache.evict_for``): domain = RECLAIMABLE candidates
    only (unprotected), sorted ASCENDING by ``last_picked`` (LRU oldest-first),
    greedily accumulating bytes until ``need`` is covered — that subset (possibly
    several models) is ``proposed_evictions``. The console renders it; it computes
    nothing.

    📌 PIN + ALLOCATION HAVE NO BEARING ON EVICTION (operator ruling,
    2026-07-17, verbatim): "the pins only should designate that the model
    allocation survives restarts. the allocation only stipulates the routing for
    that model (to that worker). neither of those should have any bearing on the
    pull or eviction, unless its to do with priority, then a pinned model should
    take higher precidence than unpinned, but even that is trivial".
      * 📌 pin = the model's ALLOCATION survives restarts (and unassign — the
        409). Nothing else. Allocation = ROUTING (which worker answers).
      * A pinned or assigned model's FILES are a normal LRU eviction candidate
        here — ``proposed_evictions`` MAY include pinned files. Evicting them
        leaves pin + allocation untouched; the bytes re-pull on next call.
      * Pin's only eviction role is the trivial FIFO tiebreak below (unpinned
        proposed first at an exact last_picked tie).
      * The ``pinned``/``why`` fields stay HONEST as ATTRIBUTION info (a row can
        read pinned:true / why:"pinned" while protected:false). 🔒static is the
        ONLY durable local-presence guard; loaded/loading/provisioning are
        live-use guards. This removed the day-one tripwire (attribution/routing
        masquerading as a disk shield). unassign-409 is UNTOUCHED — that IS pin.
    """
    storage = worker.get("storage")
    reported = isinstance(storage, dict)
    disk = worker.get("disk") or {}

    def _as_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    disk_free = None
    if reported and storage.get("disk_free") is not None:
        disk_free = _as_int(storage.get("disk_free"))
    if disk_free is None and disk.get("free_bytes") is not None:
        disk_free = _as_int(disk.get("free_bytes"))
    disk_total = _as_int(disk.get("total_bytes"))

    cache_used = _as_int(storage.get("cache_used_bytes")) if reported else None
    # ── k60: SHARED-STORE BYTES ARE NOT IN THE EVICTION ECONOMY ─────────────
    # (operator, 2026-07-31.) A row on the shared/central catalog — or on a store
    # the box never declared reapable — can NEVER be deleted from this worker, so
    # it must not be priced against the worker budget. ae read "2.8 TiB used /
    # 800 GB budget · ⚠ over budget · 2.0 TiB over" on a 1.7 TiB box because the
    # fleet's whole shared catalog was summed as this worker's resident cache;
    # the deletes were correctly refused at every guard, but the NUMBER said an
    # auto-evict was imminent. Only the accounting was wrong — the guards stay.
    #
    # A CURRENT worker already excludes those bytes from cache_used_bytes and
    # reports the remainder as `unbudgeted_bytes`. A RELEASED worker still sends
    # the shared bytes inside cache_used, so central discounts them here from the
    # per-row store-gate verdicts it has always received — that is what makes
    # this land on the live fleet without waiting for a wheel roll.
    _raw_rows = (storage.get("models") or []) if reported else []
    _unbudgeted_rows = [m for m in _raw_rows
                        if isinstance(m, dict) and m.get("model_key")
                        and not _row_counts_toward_budget(m)]
    unbudgeted_from_rows = sum(_as_int(m.get("bytes")) or 0 for m in _unbudgeted_rows)
    shared_from_rows = sum(_as_int(m.get("bytes")) or 0 for m in _unbudgeted_rows
                           if _row_store_class(m) == "shared")
    worker_unbudgeted = (_as_int(storage.get("unbudgeted_bytes"))
                         if reported else None)
    cache_used_reported = cache_used
    if cache_used is not None and worker_unbudgeted is None and unbudgeted_from_rows:
        # Released worker: subtract what it priced but may never reap. Floored at
        # 0 — a measured root can legitimately be smaller than the row sum.
        cache_used = max(0, cache_used - unbudgeted_from_rows)
    unbudgeted_bytes = (worker_unbudgeted if worker_unbudgeted is not None
                        else unbudgeted_from_rows)
    _worker_shared = _as_int(storage.get("shared_bytes")) if reported else None
    shared_bytes = _worker_shared if _worker_shared is not None else shared_from_rows
    # ORPHANED (unattributed-on-disk) residue reported by the worker (release-
    # bound field). Passed through verbatim: model dirs / stalled .part sets on
    # disk that match NO current assignment (computron's 5.7G Qwen2.5-VL-3B
    # .part junk). Absent on a pre-2026-07-17 agent -> zeros (feature-off).
    orphaned_bytes = (_as_int(storage.get("orphaned_bytes")) or 0) if reported else 0
    orphaned_count = (_as_int(storage.get("orphaned_count")) or 0) if reported else 0
    orphaned_items = (storage.get("orphaned_items") or []) if reported else []
    last_picked_map = worker.get("model_last_picked") or {}
    limits = worker.get("limits") or {}
    cfg = worker.get("config") or {}
    residency = cfg.get("residency") or {}
    pinned_cfg = cfg.get("pinned") or {}
    # Central slot-merged live truth — closes the reaper's in-process-only
    # loaded_model_keys() gap (it misses slot occupants / answering models).
    loaded_now = set(worker.get("loaded_models") or [])
    loading_now = set(worker.get("loading") or [])
    # LIVE pulls only — a stale/dead-owner entry is neither reported as
    # in-flight nor granted eviction protection. See _live_provisioning.
    provisioning_now = _live_provisioning(worker)

    reserve = _disk_reserve_bytes(limits)

    # ── budget: explicit per-worker cap wins over the free-disk reserve ──────
    # THE ALLOCATION IS THE TOTAL LIMIT (operator rule, 2026-08-22): the reserve
    # is portioned OUT OF disk_cache_gib, so the ceiling the cache may reach is
    # allocation - reserve. Mirrors worker_agent/budget.py (resolve_effective_cap
    # / fit_plan) byte-for-byte — Parity. The free-disk floor below stays as a
    # second safety check in cap mode (redundant when allocation == volume).
    cap_gib = limits.get("disk_cache_gib")
    budget_basis = "reserve"
    budget = None
    over_budget = False
    need_bytes = 0
    allocation_bytes = None
    central_budget_sources: Dict[str, Any] = {}
    if cap_gib not in (None, ""):
        cap_bytes = None
        try:
            cap_bytes = int(float(cap_gib) * (1 << 30))
        except (TypeError, ValueError):
            cap_bytes = None
        if cap_bytes is not None:
            budget_basis = "cap"
            # Min-wins over the worker's DECLARED same-drive terms (worker_*_gib
            # in its reported budget_sources) — the worker's own allocation
            # inputs are honoured, but the carve-out is computed HERE so a
            # pre-carve-out agent's stale effective number never governs.
            _wsrc = (storage.get("budget_sources") or {}) if reported else {}
            alloc_terms = [("central_gib", cap_bytes)]
            for _k, _v in (_wsrc.items() if isinstance(_wsrc, dict) else []):
                if isinstance(_k, str) and _k.startswith("worker_") and _k.endswith("_gib"):
                    try:
                        alloc_terms.append((_k, int(float(_v) * (1 << 30))))
                    except (TypeError, ValueError):
                        pass
            alloc_src, allocation_bytes = min(alloc_terms, key=lambda t: t[1])
            budget = max(0, allocation_bytes - reserve)
            central_budget_sources = {
                k: v for k, v in (_wsrc.items() if isinstance(_wsrc, dict) else [])
                if isinstance(k, str) and k.startswith("worker_")}
            central_budget_sources.update({
                "central_gib": round(cap_bytes / (1 << 30), 3),
                "allocation_gib": round(allocation_bytes / (1 << 30), 3),
                "allocation_source": alloc_src,
                "reserve_gib": round(reserve / (1 << 30), 3),
                "effective_gib": round(budget / (1 << 30), 3),
                "effective_source": alloc_src,
            })
            if cache_used is not None and cache_used > budget:
                over_budget = True
                need_bytes = cache_used - budget
            # second safety check: the real volume must keep the reserve free
            if disk_free is not None and disk_free < reserve:
                over_budget = True
                need_bytes = max(need_bytes, reserve - disk_free)
    if budget_basis == "reserve":
        # Express the cache-ceiling budget so the console bar (cache_used vs
        # budget) is consistent with the flag: over_budget <=> cache_used > budget
        # <=> disk_free < reserve. need is the free-disk shortfall.
        if disk_free is not None and cache_used is not None:
            budget = cache_used + disk_free - reserve
        if disk_free is not None and disk_free < reserve:
            over_budget = True
            need_bytes = reserve - disk_free

    # ── per-model view + reclaimable candidate domain ───────────────────────
    def _lp(mk):
        v = last_picked_map.get(mk)
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    grants_now = worker.get("grants") or {}
    models_out: List[Dict[str, Any]] = []
    candidates: List[tuple] = []   # (last_picked, bytes, model_key)
    raw_models = storage.get("models") if reported else None
    for m in (raw_models or []):
        if not isinstance(m, dict) or not m.get("model_key"):
            continue
        mk = m["model_key"]
        b = _as_int(m.get("bytes")) or 0
        lp = _lp(mk)
        # Central-final protection = worker's own flag OR any redundant central
        # guard (slot-merged loaded/loading, provisioning, static, pin). A model
        # is a candidate ONLY if unprotected on BOTH sides.
        #
        # PROVISIONING is the ONE guard in this chain that is liveness-gated
        # (2026-07-16). The worker's own per-row ``m['provisioning']`` flag is
        # NOT consulted here, unlike loaded/loading: it comes from the same dead
        # heartbeat snapshot as the stale list, so honouring it would re-admit
        # exactly the phantom protection this fix removes (op's dead pull still
        # flags its row). Central instead trusts only _live_provisioning.
        #
        # This LOSES no protection for a real pull: an ONLINE worker with a
        # moving pull is live by construction. And the worker keeps its OWN
        # authoritative guard locally (worker_agent/budget.py's
        # _PROTECTED_REASONS) — a box never deletes under its own live write on
        # central's say-so. What it REMOVES is permanent phantom protection: a
        # dead entry used to make a real, cold, reclaimable file un-evictable
        # forever, silently shrinking the reclaimable pool on a full disk.
        #
        # NOTE (Phase 1 item 2, grant markers): a SYSTEM grant is DELIBERATELY
        # ABSENT from this chain — the opposite of an operator "assigned"
        # designation. A model that is ONLY granted (not assigned/static/loaded)
        # gets no protection here and remains a normal LRU eviction candidate;
        # grants are reclaimable by construction, never a residency guarantee.
        # If a model happens to be BOTH granted and assigned/static/etc., that
        # FILE-PROTECTING designation still protects it as before — the grant
        # itself contributes nothing. (📌pin is NOT file-protecting as of
        # 2026-07-17, so granted+pinned is a candidate — see below.)
        why = m.get("why") or ""
        protected = bool(m.get("protected"))
        is_pinned = bool(m.get("pinned") or pinned_cfg.get(mk))
        # k60 STORE CLASS. A shared/unreapable row is protected UNCONDITIONALLY
        # (it is a filesystem fact, not a policy label) and is never a proposal
        # candidate — belt to the worker's own gate and to wipe_model's jail,
        # never a replacement for either.
        store_class = _row_store_class(m)
        counts = _row_counts_toward_budget(m)
        if not counts:
            protected = True
            if not why:
                why = ("shared/central storage — never reaped"
                       if store_class == "shared"
                       else "model store not marked reapable")
        # Worker-reported protection is trusted ONLY for reasons that are not
        # pure attribution ("shared/central storage — never reaped", "model
        # store not marked reapable", live-use guards). A released worker
        # (<=0.1.183) still stamps protected/why="pinned" or "assigned" from the
        # old doctrine — strip those two here and let the chain below recompute,
        # or the fleet's stale flags keep the day-one tripwire alive centrally
        # until the next release (keeper, 2026-07-17: ae reported 21/21 pinned
        # rows protected with zero live-use flags — reclaimable pool read as
        # empty on a 700G/429G box).
        if protected and why in ("pinned", "assigned"):
            # Clear the stale label too: if a live-use guard re-protects below,
            # the chain stamps the HONEST reason (loaded/loading/…) instead of
            # leaving attribution vocabulary on a protection flag.
            protected, why = False, ""
        # 📌 pin does NOT protect files (operator ruling, 2026-07-17): "the pins
        # only should designate that the model allocation survives restarts. the
        # allocation only stipulates the routing... neither of those should have
        # any bearing on the pull or eviction". So pin is DELIBERATELY absent
        # from this protection chain — a pinned model's files are a normal LRU
        # candidate; the pin + allocation survive the eviction and the bytes
        # re-pull on next call. `pinned` is still reported below as ATTRIBUTION
        # (m['pinned']) but never sets `protected`/`why`. 🔒static is the ONLY
        # durable local-presence guard; loaded/loading/provisioning are live-use
        # guards. This removes the day-one tripwire that let attribution/routing
        # masquerade as a disk shield.
        if not protected:
            if str(residency.get(mk) or "").lower() == "static":
                protected, why = True, why or "static"
            # NO `assigned` branch (operator ruling 2026-07-17): "the allocation
            # only stipulates the routing for that model... neither of those
            # should have any bearing on the pull or eviction." Assignment is
            # attribution, same as pin — worker-side budget._is_protected has
            # said so since f1894b2; this chain protecting `assigned` was the
            # central half of the same day-one tripwire (on a box whose on-disk
            # models are all assigned — ae, op — the reclaimable pool read as
            # permanently empty).
            elif mk in loaded_now or m.get("loaded"):
                protected, why = True, why or "loaded"
            elif mk in loading_now or m.get("loading"):
                protected, why = True, why or "loading"
            elif mk in provisioning_now:
                protected, why = True, why or "provisioning"
            elif is_pinned and not why:
                # ATTRIBUTION-only annotation: honest `why` for a pinned model
                # that has no other protecting flag, while `protected` stays
                # False (it remains a candidate). A bare pinned row therefore
                # shows why="pinned" but IS eligible for the proposal below.
                why = "pinned"
        _tok_hint = tok_hint_for(worker, mk)
        models_out.append({
            "model_key": mk,
            "bytes": b,
            "last_picked": lp or None,     # None = never served through central
            "protected": protected,
            "why": why,
            "granted": mk in grants_now,   # SYSTEM marker only — confers no protection
            "pinned": is_pinned,           # ATTRIBUTION only — confers no eviction protection (2026-07-17)
            "loaded": bool(m.get("loaded") or mk in loaded_now),
            "loading": bool(m.get("loading") or mk in loading_now),
            # LIVE pulls only (not the worker's stale per-row flag) — this is
            # what the console renders as "⏳ pulling"; a dead pull must read
            # as missing, never as an active transfer.
            "provisioning": mk in provisioning_now,
            "assigned": bool(m.get("assigned")),
            # k60: which store the bytes are on, and whether they were priced
            # against the budget. The console sections the shared rows off with
            # these so the operator can SEE why the used figure shrank.
            "store": store_class,
            "counts_toward_budget": counts,
            # tok/s last · avg on this (worker, model) — operator 2026-08-22
            "tok_s_last": _tok_hint["last_tok_s"],
            "tok_s_avg": _tok_hint["avg_tok_s"],
            "tok_n": _tok_hint["n"],
        })
        if not protected and counts:
            # Proposals may name ONLY reapable rows (k60 ruling 2). `protected`
            # already excludes them; `counts` says so in the store's own terms.
            candidates.append((lp, b, mk))

    # ── reclaimable-count trace (operator ask, 2026-07-17): "yell out the
    # course of the reclaimable count". The worker half is traced by
    # scan_keys_considered → scan_skip_reasons → scan_rows; this logs the
    # CENTRAL half — rows in vs candidates out of the final-guard chain — and
    # shouts when the chain zeroes a non-empty report (the collapse signature
    # that cost a day to localize by hand).
    _rows_in = len([m for m in (raw_models or []) if isinstance(m, dict) and m.get("model_key")])
    if _rows_in and not candidates:
        _breakdown = {w: sum(1 for m in models_out if m.get("why") == w)
                      for w in {m.get("why") for m in models_out if m.get("protected")}}
        # ONCE PER STATE, not once per heartbeat (operator, 2026-07-29: this
        # line was repeating every second per gunicorn worker for a condition
        # that is STEADY STATE on ae/op — shared/central storage is never
        # reaped, and that's by design, not a collapse to shout about). The
        # diagnostic stays, but it only speaks when the signature CHANGES;
        # unchanged repeats drop to debug.
        _wname = worker.get("name") or worker.get("id", "?")[:8]
        _sig = (_rows_in, tuple(sorted(_breakdown.items())))
        if _RECLAIM_COLLAPSE_SEEN.get(_wname) != _sig:
            _RECLAIM_COLLAPSE_SEEN[_wname] = _sig
            logger.warning(
                "reclaimable collapse on %s: worker reported %d storage rows but 0 "
                "survived central's guard chain (protected breakdown: %s) — "
                "logged once; repeats at debug until this changes",
                _wname, _rows_in, _breakdown)
        else:
            logger.debug(
                "reclaimable collapse on %s (unchanged): %d rows, %s",
                _wname, _rows_in, _breakdown)

    proposed: List[Dict[str, Any]] = []
    proposed_free = 0
    if over_budget and need_bytes > 0 and candidates:
        # ── THE SHARED EVICT FUNCTION (spec assets/evictionflow.html, box 2) ──
        # PARITY IS THE WHOLE POINT of this call. This preview and the worker's
        # ``budget.fit_plan`` auto-evict now run the IDENTICAL function over the
        # identical inputs, so they can never propose different victims — the
        # spec names divergence as the failure mode. Neither side spells the
        # sort key; both import it.
        #
        # ONE LEDGER: the idle times below are central's own call log
        # (``model_last_picked``, stamped in pick_for_model) and the call counts
        # are ``model_call_stats`` from the same place — and BOTH are shipped to
        # the worker on the heartbeat reply, so the worker measures from
        # central's clock rather than its own. That shipping is what makes the
        # parity real rather than coincidental.
        #
        # This REPLACES oldest-first-then-LARGEST-first. Largest-first cleared a
        # budget in the fewest deletes but had no relationship to what the
        # admission needed; walk-then-drop is the spec's answer (least reaping).
        # Device = the disk (see the same note in budget.fit_plan: key ① is a
        # residency concept and degenerates to a constant here, leaving the
        # honest storage order ②/③/④).
        from hugpy_engine import eviction as _ev
        _stats = worker.get("model_call_stats") or {}
        _modes = worker.get("model_alloc_modes") or {}

        def _calls_for(mk):
            try:
                return int((_stats.get(mk) or {}).get("calls") or 0)
            except (TypeError, ValueError, AttributeError):
                return 0

        _by_key = {mk: (lp, b) for lp, b, mk in candidates}
        # No residency floor — the 300s anti-thrash veto was retired
        # 2026-07-27 (operator). This path never had one regardless: a freshly
        # downloaded FILE has no load clock here, and inventing one from mtime
        # would protect the cold leftovers this budget exists to clear.
        _plan = _ev.evict_plan(
            "disk", need_bytes,
            [_ev.EvictUnit(model_key=mk, bytes=b,
                          pref=_ev.preferred_device(_modes.get(mk)),
                          last_call=(lp or None), calls=_calls_for(mk))
             for lp, b, mk in candidates],
            now=_now(),
            # FLEET-WIDE drop-pass policy — the SAME value every worker adopts
            # off the heartbeat (comms/evict_policy.py). Read here rather than
            # defaulted so this preview and the worker's execution cannot
            # disagree: that divergence is exactly what Parity forbids.
            least_reaping=_fleet_least_reaping())
        for mk in _plan.victims:
            lp, b = _by_key.get(mk, (None, 0))
            proposed.append({"model_key": mk, "bytes": b,
                             "last_picked": lp or None})
            proposed_free += b

    # ── ALLOCATION-LEVEL view (operator, 2026-07-16) ────────────────────────
    # Structural, and TRUE EVEN WHEN NO PULL IS HAPPENING: if the assigned set
    # itself exceeds the budget, the worker is over-subscribed now — the console
    # can surface that BEFORE some unlucky call wedges. Computed on every read
    # (like the vram/ram summaries) and cheap: sizes come from the cached
    # registry + mtime-cached dir walks, not per-model HTTP.
    alloc = allocated_totals(worker)
    alloc_over = 0
    if budget is not None and alloc["allocated_total_bytes"] > budget:
        alloc_over = alloc["allocated_total_bytes"] - budget
    alloc["allocated_over_budget_bytes"] = alloc_over

    # ── ATTRIBUTED vs RESIDENT (2026-07-17) ─────────────────────────────────
    # The operator scare: assignment/pin ATTRIBUTES a model to a worker without
    # putting bytes on disk (lazy download, 7f0e6e8/2a3baeb). The fleet gauge
    # read cache_used/budget, and an over-subscribed ATTRIBUTION set made a box
    # with nothing transferring look like a runaway download storm. Split the two
    # so attribution can NEVER masquerade as disk pressure:
    #   * attributed = the assignment/pin SET's effective size (may exceed disk;
    #     "assigned but not on disk" is a CORRECT resting state, not pressure).
    #   * resident   = bytes ACTUALLY on disk. The worker's measured cache_used is
    #     the authority; the per-model on-disk sum is the fallback/cross-check.
    # The disk-pressure GAUGE is derived from RESIDENT only.
    # k60: the gauge's fallback sum prices ONLY budget-bearing rows, exactly like
    # cache_used above — otherwise a worker with no measured figure would put the
    # shared catalog straight back into the disk-pressure reading.
    hot_from_models = sum(int(m.get("bytes") or 0) for m in models_out
                          if m.get("counts_toward_budget", True))
    hot_bytes = cache_used if cache_used is not None else (
        hot_from_models if reported else None)
    attributed = {
        "attributed_total_bytes": alloc["allocated_total_bytes"],
        "attributed_count": alloc["allocated_count"],
        "attributed_unknown_count": alloc["allocated_unknown_count"],
        "attributed_over_budget_bytes": alloc_over,
    }
    hot = {
        # HOT = bytes on the worker's hot drive NOW (canonical STATE-MODEL.md #7).
        # `hot_bytes` is the number the disk-pressure gauge must use.
        "hot_bytes": hot_bytes,
        "hot_model_bytes": hot_from_models,
        # measured vs summed can disagree (heartbeat lag / non-model files); both
        # surfaced so the console shows the truth instead of averaging a lie.
        "hot_source": ("measured" if cache_used is not None
                       else ("summed" if reported else "unknown")),
        # ORPHANED = on disk but attributed to NO model (leftover dirs + stalled
        # .part sets). A THIRD class distinct from attributed and
        # resident-attributed: junk eating the drive that the allocation ledger
        # never showed. UI label: "unattributed on disk".
        "orphaned_bytes": orphaned_bytes,
        "orphaned_count": orphaned_count,
        "orphaned_items": orphaned_items,
        # SHARED / UNREAPABLE (k60): on disk here, but NOT this worker's to
        # evict — labeled, never priced. A fourth class beside attributed,
        # resident-attributed and orphaned. UI: "shared catalog (never evicted)".
        "unbudgeted_bytes": unbudgeted_bytes,
        "unbudgeted_count": len(_unbudgeted_rows),
        "shared_bytes": shared_bytes,
        "shared_count": sum(1 for m in _unbudgeted_rows
                            if _row_store_class(m) == "shared"),
    }
    # The disk-pressure gauge: RESIDENT over budget. Attribution is deliberately
    # excluded — an over-subscribed assignment set is surfaced via
    # attributed_over_budget_bytes (structural), never as a full-disk reading.
    gauge = {
        "gauge_used_bytes": hot_bytes,        # <-- what the UI bar fills to
        "gauge_budget_bytes": budget,
        "gauge_basis": "hot",
        "gauge_over_budget": over_budget,     # already computed from cache_used/disk_free
    }

    return {
        **alloc,
        **attributed,
        **hot,
        **gauge,
        "reported": reported,
        # BUDGET-BEARING used (k60): shared/unreapable bytes discounted out.
        "cache_used_bytes": cache_used,
        # What the worker actually put on the wire, so the discount is auditable
        # rather than a silent shrink (equals cache_used_bytes on a current one).
        "cache_used_reported_bytes": cache_used_reported,
        "store_root": (storage.get("store_root") or "") if reported else "",
        "store_root_shared": bool(storage.get("store_root_shared")) if reported else False,
        "store_root_budgeted": (bool(storage.get("store_root_budgeted"))
                                if reported and storage.get("store_root_budgeted") is not None
                                else None),
        "disk_free": disk_free,
        "disk_total": disk_total,
        "reserve": reserve,
        "budget": budget,
        "allocation_bytes": allocation_bytes,   # raw disk_cache_gib; budget = this - reserve
        "budget_basis": budget_basis,
        "over_budget": over_budget,
        "need_bytes": need_bytes if over_budget else 0,
        "proposed_free_bytes": proposed_free,
        "proposed_evictions": proposed,
        "models": models_out,
        # Storage REFUSALS reported by the worker: {model_key: {state:"refused",
        # reason, needs_bytes, budget_bytes, reclaimable_bytes, blocked, ...}}.
        # Models whose pull was refused BEFORE it started because even a full
        # FIFO of the reclaimable models couldn't seat them. Passed through
        # VERBATIM — this is the worker's own verdict about its own disk, and
        # central has no better information to second-guess it with. They have
        # no files on disk, so they are deliberately absent from `models` and
        # never appear in a proposal; the console renders them as MISSING with
        # the reason on hover.
        "refused": (storage.get("refused") or {}) if reported else {},
        # SCAN DIAGNOSTICS (slice 3, B) — passed through VERBATIM from the worker
        # survey so a broken/degraded reap scan can never masquerade as a clean
        # empty store (the ae 2026-07-17 defect: rows:0 while 65 models were on
        # disk). The console can surface scan_error / considered≫rows. Absent on a
        # pre-slice-3 worker -> falsy defaults (feature simply off).
        "scan_error": (storage.get("scan_error") or "") if reported else "",
        "scan_keys_considered": (_as_int(storage.get("scan_keys_considered")) or 0) if reported else 0,
        "scan_rows": (_as_int(storage.get("scan_rows")) or 0) if reported else 0,
        "scan_row_errors": (_as_int(storage.get("scan_row_errors")) or 0) if reported else 0,
        # SKIP-REASON HISTOGRAM (slice 5) — passed through VERBATIM so the console
        # can name the ae failure class (not_local/no_config/comfy). Absent on a
        # pre-slice-5 worker -> {} (feature off).
        "scan_skip_reasons": (storage.get("scan_skip_reasons") or {}) if reported else {},
        # REGISTRY SOURCES (slice 6) — per-origin config counts, passed through
        # VERBATIM so the console shows a dead source (discovered==0) in one beat.
        # Absent on a pre-slice-6 worker -> {} (feature off).
        "registry_sources": (storage.get("registry_sources") or {}) if reported else {},
        # EFFECTIVE BUDGET (slice 4, min-wins) — the worker's own resolved
        # min(central disk_cache_gib, worker same-drive declarations) + the source
        # map, passed through VERBATIM so the console can show WHY a number
        # governs (e.g. central 400 wins over worker hot 1500). Absent on a
        # pre-slice-4 worker -> None/{}/False (feature simply off). This is the
        # WORKER's own resolution; central's `budget`/`over_budget` above are its
        # own view and unchanged.
        # CENTRAL IS AUTHORITATIVE in cap mode (2026-08-22): allocation - reserve
        # computed above, with the worker's declared worker_* terms folded into
        # the min. A stale worker self-report (pre-carve-out agent) is kept only
        # as worker_reported_* for diagnosis; it never governs.
        "budget_effective_bytes": (budget if budget_basis == "cap" else
                                   ((_as_int(storage.get("budget_effective_bytes"))) if reported else None)),
        "budget_sources": (central_budget_sources if budget_basis == "cap" else
                           ((storage.get("budget_sources") or {}) if reported else {})),
        "worker_reported_budget_effective_bytes": (_as_int(storage.get("budget_effective_bytes"))) if reported else None,
        "worker_reported_budget_sources": (storage.get("budget_sources") or {}) if reported else {},
        "budget_cap_not_applicable": bool(storage.get("budget_cap_not_applicable")) if reported else False,
        # AUTO-REAP MODE (slice 8, Part B) — the per-worker opt-in flag + the last
        # time an auto-fire ran, so the console can show the mode ("auto" vs the
        # default hand-approve) and when it last acted. Read from the WORKER record
        # (central-owned policy), not the worker-reported storage. Default false /
        # None — a worker never opted in reads as hand-approve, today's posture.
        "auto_reap": bool(worker.get("auto_reap")),
        "last_auto_reap_at": worker.get("last_auto_reap_at"),
    }


def _match_keys(model_key: str) -> set:
    """Normalized aliases a model might be named by, for tolerant matching.

    A model can be referenced as its registry key, its hub_id (owner/name), or
    just the trailing name — and with different case. We compare on the set of
    these forms so an assignment made via one spelling still routes a chat that
    uses another. Example: "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
    "Qwen2.5-Coder-3B-Instruct-GGUF" and the lowercased variants all match.

    "~"-TAIL UNIFICATION (key-match unification, operator doctrine 2026-07-23).
    Registry keys qualify a base name with its owner via "~" ("Qwen~X",
    "unsloth~X") while workers routinely serve/report the BARE base name ("X").
    Before this, the two spellings never intersected — the k30 class of
    invisible mismatches: a designation carries the ~-qualified key, the
    worker's loaded/models list carries the bare one, and routing looked at a
    box actually holding the model and said "not serveable here". Operator
    doctrine: a specific call may try any same-base sibling — so the "~"-tail
    is an alias exactly like the "/"-tail. "Qwen~X" -> {"Qwen~X", "qwen~x",
    "X", "x"}; bare "X" -> {"X", "x"}; qualified and bare now intersect in BOTH
    directions. (Two different owners of the SAME base intersect too — the
    blocked-sibling guard in _serveable_match is what keeps that from serving a
    BLOCKED sibling under an unblocked name.) Raw forms stay first-class; the
    tails are additions, never replacements.
    """
    if not model_key:
        return set()
    raw = str(model_key).strip()
    forms = {raw, raw.lower()}
    tail = raw.split("/")[-1]
    forms.add(tail)
    forms.add(tail.lower())
    if "~" in raw:
        base = raw.split("~", 1)[1]
        if base:
            forms.add(base)
            forms.add(base.lower())
    return forms


def _serveable_match(model_key: str, wanted: set, serveable) -> bool:
    """True iff one of ``serveable``'s advertised keys names ``model_key``.

    The single membership predicate for every routing match site (the
    eligibility gate, the engine-skip say-why counter, explain_no_worker's
    designation walk). It replaces the old
    ``model_key in serveable or wanted & {alias-union}`` idiom with a
    PER-ADVERTISED-KEY decision, which is what makes the BLOCKED-SIBLING GUARD
    possible: the "~"-tail unification in _match_keys lets a request for an
    unblocked ``A~X`` tail-match an advertised ``B~X`` (different owner, same
    base). If that advertised sibling is BLOCKED, counting the match would
    effectively serve a blocked model under an unblocked name — so an
    alias-only match against a blocked advertised key is skipped. The
    REQUESTED key's own block gate is enforced upstream (workers_for_model
    returns [] before any per-worker work); the blocklist is consulted here
    ONLY for alias-matched (never literal-matched) advertised keys, keeping
    the hot path cheap.
    """
    for m in serveable:
        if m == model_key:
            return True
        if wanted & _match_keys(m):
            if _model_blocked(m):
                continue     # blocked sibling — an alias never launders a block
            return True
    return False


# ---------------------------------------------------------------------------
# ROUTING RANK — residency and allocation are ROUTING FACTS, not trivia.
#
# Operator incident 2026-07-28: a chat for Qwen2.5-7B-Instruct-GGUF — resident
# AND allocated on computron, served from there an hour earlier — was routed to
# ae, which had nothing on disk, cold-provisioned it, and blew the caller's hold
# on the way. Verdict: "the allocations are not being adhered to."
#
# The old rank had exactly ONE residency term:
#
#     warm = model_key in (w.get("loaded_models") or [])
#
# and it was wrong in two ways that both bit here:
#
#   1. EXACT string match, in a file where every other match site goes through
#      _match_keys/_serveable_match. A worker that reports the bare base name
#      while the request carries the ~-qualified registry key (or vice versa)
#      reads as COLD even while it is actively serving the model. That is the
#      k30 invisible-mismatch class, still live in the one place that decides
#      where a call goes.
#   2. ALLOCATION was never consulted at all. A model with a slot seat or an
#      in-RAM allocation on a box — the compute tab's allocations, which ride
#      the heartbeat — was invisible to routing the moment it fell out of
#      loaded_models (and a slot-child never appears there in the first place:
#      see the on-demand-slot-child-not-in-loaded-models landmine).
#
# With both terms blind, two wildcard boxes tie all the way down to
# ``last_picked`` — plain round-robin — and the call lands wherever the dice
# fall. Hence: cold-provision a 4.7GB model onto ae while the box that already
# had it resident sat idle.
#
# The ranking is now, strictly in order:
#
#   0. DESIGNATION — a home (designated/resident/granted) worker before any
#      wildcard catch. HARD (operator ruling, designation-is-advisory RULING
#      CORRECTED): when a designation exists the call never leaves it; if the
#      designated box cannot serve, the refusal names that box's reason
#      (explain_no_worker) rather than silently spilling.
#   1. RESIDENT  — measured-resident RIGHT NOW (alias-tolerant).
#   2. ALLOCATED — holds an approved allocation for the model (alias-tolerant).
#   3. capability rank exactly as before (star, GPU, least-recently-picked, id).
#
# This is a RANKING fix only. Nothing here admits a worker that the eligibility
# gates (workers_for_model) would have excluded, and no tier can rescue a box
# that failed a hard gate.
# ---------------------------------------------------------------------------

def _resident_on(worker: Dict[str, Any], model_key: str, wanted: set) -> bool:
    """MEASURED-RESIDENT: this worker reports ``model_key`` loaded RIGHT NOW.

    Alias-tolerant via the same ``_serveable_match`` predicate every other match
    site uses (the whole point of the fix — see the block comment above), and it
    also honours a live SLOT/loaded allocation row, because a slot-seated child
    serves without ever appearing in ``loaded_models``.
    """
    if _serveable_match(model_key, wanted, worker.get("loaded_models") or []):
        return True
    for row in _allocation_rows(worker):
        if not _serveable_match(model_key, wanted, [row.get("model_key") or ""]):
            continue
        # A slot row is residency only while its child is actually up; an
        # in-RAM row is residency by construction (the weights are in RAM).
        if row.get("kind") == "slot":
            if row.get("healthy") or row.get("serving") or row.get("busy"):
                return True
            continue
        return True
    return False


def _allocation_rows(worker: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The worker's heartbeat allocation rows, defensively typed ([] on junk)."""
    rows = worker.get("allocations")
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def _allocated_on(worker: Dict[str, Any], model_key: str, wanted: set) -> bool:
    """This worker holds an APPROVED ALLOCATION for ``model_key``.

    An allocation outlives a load: the compute tab's allocation rows (slot seats
    and in-RAM residents, which ride the heartbeat) and the system-authored
    placement ``grants`` both say "this model belongs on this box", and both
    survive the model dropping out of ``loaded_models``. Routing must honour
    that before it prices a cold box on capability alone — the allocation IS the
    prior decision about where this model runs.

    Deliberately NOT sourced from ``model_alloc_modes``: that map carries a
    derived mode for every DESIGNATED model, so reading it here would make this
    tier a duplicate of the designation tier rather than an independent signal.
    """
    for row in _allocation_rows(worker):
        if _serveable_match(model_key, wanted, [row.get("model_key") or ""]):
            return True
    grants = worker.get("grants")
    if isinstance(grants, dict) and grants:
        if _serveable_match(model_key, wanted, list(grants)):
            return True
    return False


def _route_tier(worker: Dict[str, Any], model_key: str, wanted: set) -> str:
    """The reason this worker is where it is in the order — the telemetry label.

    Most-specific-fact-first, which is NOT the sort order: designation sorts
    ahead of everything (it is a hard scope), but when a designated box is ALSO
    resident, "resident" is the more informative thing to tell the operator.
    """
    if _resident_on(worker, model_key, wanted):
        return "resident"
    if _allocated_on(worker, model_key, wanted):
        return "allocated"
    if not worker.get("_wildcard_catch"):
        return "designated"
    return "capability"


# ── k56: ordered worker preference + polite (no-evict) load ─────────────────
# Two GENERAL per-model placement options (operator ruling 2026-07-31), both
# persisted model-scoped in the serve-overrides layer (managers.serve.overrides
# ALLOWED_FIELDS) — see placement_prefs there for why that is the SoT for an
# ORDER when designation itself is per-worker set membership.
#
# Neither has any effect unless the operator sets it: no list ⇒ the pref index
# is a constant 0 and the rank tuple is the old one; no polite flag ⇒ admission
# is untouched. That is the whole compatibility argument for the degenerate
# single-designation case being byte-identical.
#
# THE FREE-ROOM PROBE is injected rather than imported: the fit arithmetic lives
# in worker_routes._worker_fit (which knows model sizing, calibration and the
# tolerance bands) and this module must not import the route layer. Routes
# register it at import time; unregistered (standalone / a bare central) ⇒
# central proves nothing and the worker's own polite admission decides, which
# is the honest degradation rather than a second, drifting copy of the math.
_free_room_probe: Optional[Any] = None


def set_free_room_probe(fn: Optional[Any]) -> None:
    """Register the (model_key, worker) -> fit verdict probe (routes -> store)."""
    global _free_room_probe
    _free_room_probe = fn
    logger.info("free-room probe registered: %s", getattr(fn, "__name__", fn))


def placement_prefs(model_key: str) -> tuple:
    """``(ordered worker preference, polite)`` for a model — guarded re-export
    of the overrides reader so routing never imports the serve layer directly
    and a missing overrides module degrades to pre-k56 behaviour."""
    prefs, polite, _by_worker = placement_policy(model_key)
    return prefs, polite


def placement_policy(model_key: str) -> tuple:
    """k62: ``(prefs, model-wide polite, per-worker polite map)`` — the guarded
    re-export routing actually resolves against. Read ONCE per pick, then
    :func:`_polite_on` answers per candidate."""
    try:
        from hugpy_engine.serve.overrides import placement_policy as _pp
        return _pp(model_key)
    except Exception:  # noqa: BLE001 — placement must never break routing
        return [], False, {}


def _worker_forms(worker: Dict[str, Any]) -> set:
    """The id/name spellings a worker answers to, lowercased."""
    return {str(worker.get("id") or "").strip().lower(),
            str(worker.get("name") or "").strip().lower()} - {""}


def _polite_on(worker: Dict[str, Any], polite: bool, by_worker: dict) -> bool:
    """k62: is the model polite ON THIS WORKER? ``map[W]`` when the operator
    named W, else the model-wide default. Politeness is per (model × worker)
    because contention is a property of the box: the same model may have to be
    polite on a contended card and keep ordinary eviction rights elsewhere."""
    try:
        from hugpy_engine.serve.overrides import resolve_polite
        return resolve_polite(polite, by_worker, _worker_forms(worker))
    except Exception:  # noqa: BLE001 — degrade to the model-wide answer
        return bool(polite)


def _pref_index(worker: Dict[str, Any], prefs: List[str]) -> Optional[int]:
    """Position of ``worker`` in the operator's ordered list, or None when it is
    OFF the list. Matched on id OR name, case-insensitively: the console posts
    ids, an operator editing the file by hand writes names, and a designation
    that silently failed to match would land the model somewhere it was never
    designated — the exact failure the hardness rule forbids."""
    forms = _worker_forms(worker)
    for i, want in enumerate(prefs):
        if str(want).strip().lower() in forms:
            return i
    return None


def _prefs_scope(candidates: List[Dict[str, Any]], prefs: List[str],
                 model_key: str) -> List[Dict[str, Any]]:
    """Restrict candidates to the ordered list — designation hardness, per
    candidate. A model carrying a list NEVER lands off it, so an empty result is
    the correct answer (the caller refuses/holds honestly) and not a reason to
    fall back to the wider fleet. Logged when it bites, because "my model went
    nowhere" must never be a silent scope decision."""
    kept = [w for w in candidates if _pref_index(w, prefs) is not None]
    if not kept:
        logger.warning(
            "model %s has an ordered worker preference %s and NONE of them is "
            "an eligible candidate right now — refusing rather than landing "
            "off-list (designation is hard per candidate)", model_key, prefs)
    return kept


def _polite_admits(worker: Dict[str, Any], model_key: str) -> tuple:
    """Would a POLITE load of ``model_key`` land on ``worker`` without evicting?

    Returns ``(admits, reason)``. Two gates, in this order:

      1. VERSION. Politeness is a promise kept by the WORKER's admission
         (no_evict rides the spill wire). A worker that predates it would evict
         residents to make room, so a polite model is not routed there at all —
         the loud alternative to a silently-dropped flag.
      2. FREE ROOM. Central proves the model fits the worker's currently-free
         VRAM. It can only ever prove the NEGATIVE case: an unsizable model, an
         unregistered probe or a worker reporting no VRAM figure means central
         knows nothing, so the candidate is admitted here and the worker's own
         (measured, band-flexed) admission makes the real call. Central refusing
         on an unproven guess would strand a model that would have fitted.
    """
    try:
        from hugpy_engine.alloc_modes import worker_honors_no_evict, NO_EVICT_MIN_PKG_VERSION
        if not worker_honors_no_evict(worker.get("pkg_version")):
            return False, (f"pkg {worker.get('pkg_version') or 'unknown'} "
                           f"predates polite load (needs >= "
                           f"{NO_EVICT_MIN_PKG_VERSION})")
    except Exception:  # noqa: BLE001 — an unreadable gate must not strand a load
        pass
    probe = _free_room_probe
    if probe is None:
        return True, "free room unproven (no probe registered) — worker decides"
    try:
        verdict = probe(model_key, worker) or {}
    except Exception:  # noqa: BLE001 — a preflight miss is not a refusal
        return True, "free room unproven (preflight failed) — worker decides"
    if verdict.get("vram_free") is None or verdict.get("need") is None:
        return True, "free room unproven (unsizable) — worker decides"
    if verdict.get("gpu_resident"):
        return True, "fits free VRAM"
    return False, (verdict.get("reason")
                   or "does not fit free VRAM without evicting a resident")


def _emit_route_refuse(model_key: str, reason: str,
                       considered: List[Dict[str, Any]]) -> None:
    """Telemetry: WHY nothing was picked. Rides the same eviction feed as
    route.select, so the console shows a polite model's non-landing as an event
    with a cause instead of an unexplained absence. Best-effort, always."""
    try:
        from hugpy_fleet.central.evictions import emit_eviction_event
        emit_eviction_event(
            "route.refuse",
            model_key=model_key,
            reason=reason,
            alternatives=[{"worker": w.get("name") or w.get("id") or "",
                           "reason": w.get("_refuse_reason") or ""}
                          for w in considered][:6] or None,
        )
    except Exception:  # noqa: BLE001 — telemetry never breaks routing
        logger.debug("route.refuse telemetry skipped for %s", model_key,
                     exc_info=True)


def _routing_rank(worker: Dict[str, Any], model_key: str, wanted: set,
                  starred: bool, pref_index: int = 0) -> tuple:
    """The shared sort key for every routing decision (primary pick + reroute).

    ONE function so ``pick_for_model`` and ``candidates_for_model`` can never
    drift apart — the relay's reroute walk must agree with the pick it is
    falling back from, or a "reroute" silently becomes a re-decision.

    ``pref_index`` (k56) is term ⓪: the operator's ORDERED worker preference
    outranks every derived signal below it, because "try ae, then computron" is
    a stated decision and a derived ordering that could outvote it would not be
    an order. Defaults to 0 for every worker when no list is set, so the tuple
    is a constant prefix and the ranking is byte-identical to pre-k56.
    """
    return (
        # ⓪ the operator's stated candidate order (k56); 0 for all when unset.
        pref_index,
        # ① designation is a HARD scope: home before any wildcard catch.
        1 if worker.get("_wildcard_catch") else 0,
        # ② measured-resident now — no reload, no cold provision.
        0 if _resident_on(worker, model_key, wanted) else 1,
        # ③ holds an approved allocation for this model.
        0 if _allocated_on(worker, model_key, wanted) else 1,
        # ④..⑦ capability rank, unchanged.
        0 if starred else 1,
        0 if _has_usable_gpu(worker) else 1,
        worker.get("last_picked", 0),
        worker.get("id", ""),
    )


def _emit_route_select(model_key: str, chosen: Dict[str, Any],
                       ordered: List[Dict[str, Any]], wanted: set,
                       tok_s: Optional[Dict[str, Any]] = None) -> None:
    """Telemetry: WHY this box. Rides the eviction feed (GET /llm/evictions).

    Best-effort and totally guarded — a telemetry failure must never cost a
    request. Alternatives are the runners-up in rank order, capped so a large
    fleet can't bloat the ring.
    """
    try:
        from hugpy_fleet.central.evictions import emit_eviction_event
        alts = [{"worker": (w.get("name") or w.get("id") or ""),
                 "tier": _route_tier(w, model_key, wanted)}
                for w in ordered if w.get("id") != chosen.get("id")][:6]
        emit_eviction_event(
            "route.select",
            model_key=model_key,
            chosen_worker=(chosen.get("name") or chosen.get("id") or ""),
            tier=_route_tier(chosen, model_key, wanted),
            alternatives=alts or None,
            tok_s=tok_s or None,
        )
    except Exception:  # noqa: BLE001 — telemetry never breaks routing
        logger.debug("route.select telemetry skipped for %s", model_key,
                     exc_info=True)


def _wildcard_map() -> Dict[str, bool]:
    """The per-worker WILDCARD ("take all comers") opt-in map {worker_id: True}.

    Operator doctrine 2026-07-23: worker designations are a HARD routing scope;
    an undesignated model "gets in where it fits in" ONLY on workers that
    explicitly opted in as wildcard. Stored in models_config
    (worker_wildcard.json — same store family as the boot_prewarm star); absent
    key = False, so a fleet with no flags set routes exactly as before the
    feature existed (defaults are promises). Fully guarded: a store miss must
    never break the heartbeat/selection hot path — {} (nobody is a wildcard) is
    the safe degradation.
    """
    try:
        from hugpy_engine.config.models.models_config import worker_wildcard_state
        return worker_wildcard_state()
    except Exception:  # noqa: BLE001 — never let the flag store break selection
        return {}


def _star_map() -> Dict[str, Any]:
    """The per-worker ⭐ BOOT-LOAD STAR map {worker_id: model_key} (or {}).

    The star's ONLY routing effect (operator RULING 2026-07-23, post-incident:
    "it shouldn't effect anything but priority for ambiguous model calls") is a
    tie-break in worker ranking: when nothing is warm, prefer the worker whose
    boot star == the requested model — that box would boot-load it anyway, so a
    no-warm call lands where the model is (or will soon be) resident. The star is
    NOT keep-warm and has NO reconcile/eviction effect (see agent boot-once +
    worker_routes._reconcile_warm_set). Stored in models_config
    (worker_boot_prewarm.json — same store family as the wildcard flag). Read ONCE
    per pick call (never per candidate). Fully guarded: a store miss must never
    break selection — {} (no star anywhere) is the safe degradation and leaves
    ranking exactly as it was before the star key existed.
    """
    try:
        from hugpy_engine.config.models.models_config import worker_boot_prewarm_state
        return worker_boot_prewarm_state()
    except Exception:  # noqa: BLE001 — never let the star store break selection
        return {}


class WorkerStore:
    """Disk-authoritative, multi-process-safe registry of GPU workers.

    Under gunicorn/uwsgi the API runs as several processes, so an in-memory
    dict would split-brain: a worker registered in process A would be invisible
    to a heartbeat or chat request handled by process B (the classic symptom is
    "registers + shows in the UI, but heartbeats 410 and chats never offload").

    To avoid that, ``workers.json`` is the single source of truth: every read
    re-loads it, and every mutation takes an exclusive ``fcntl`` lock, reloads,
    mutates, and writes back atomically. A short-lived in-process RLock just
    keeps threads within one process from racing the same fd.
    """

    # Read-cache TTL: the console polls /llm/workers every ~10s; without this
    # every poll does an open+flock+read of workers.json, which BLOCKS on a
    # degraded mount and stalls the API. Reads serve from cache within the TTL;
    # writes always go to disk and refresh the cache, so liveness stays correct.
    _READ_TTL = 3.0

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or _default_workers_path()
        self._lock = threading.RLock()
        self._cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._cache_at = 0.0
        self._ensure_parent()

    # -- persistence (disk-authoritative) ----------------------------------
    def _ensure_parent(self) -> None:
        parent = os.path.dirname(self._path)
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError:
                pass

    def _read_unlocked(self, fh=None) -> Dict[str, Dict[str, Any]]:
        """Parse the workers map from an open fh, or from disk if none given.

        A non-empty file that fails to parse is treated as CORRUPTION, not as an
        empty registry: we log and re-raise rather than return {}. Otherwise a
        torn write (this unit restarts often) would be silently 'healed' into an
        empty fleet, and the next write would persist that empty set — wiping
        every worker. Absent/empty files still return {} (normal cold start).
        """
        try:
            if fh is not None:
                fh.seek(0)
                raw = fh.read()
            elif os.path.exists(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = f.read()
            else:
                return {}
        except OSError:
            return {}
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("workers registry root is not a JSON object")
            return {w["id"]: w for w in data.get("workers", []) if w.get("id")}
        except (ValueError, KeyError) as exc:
            import logging as _logging
            _logging.getLogger(__name__).error(
                "workers registry %s is unparseable (%d bytes) — refusing to treat "
                "as empty; leaving the file intact for recovery (%s)",
                self._path, len(raw), exc,
            )
            raise

    def _write_unlocked(self, fh, workers: Dict[str, Dict[str, Any]]) -> None:
        """Overwrite the open, locked fh with the workers map."""
        payload = json.dumps({"workers": list(workers.values())}, indent=2)
        fh.seek(0)
        fh.truncate()
        fh.write(payload)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass

    def _load(self) -> Dict[str, Dict[str, Any]]:
        """Read-only snapshot of the registry, cached for a few seconds.

        Polls (list/get/pick) hit this; the cache keeps a hung/slow mount from
        blocking every request. Writes refresh the cache, so freshly-registered
        or reassigned workers are visible immediately to the writing process.
        """
        now = time.time()
        with self._lock:
            if self._cache is not None and (now - self._cache_at) < self._READ_TTL:
                return self._cache
            try:
                data = self._read_unlocked()
            except (ValueError, KeyError):
                # Corrupt on-disk file: don't crash polls — serve the last good
                # snapshot if we have one (the error is already logged).
                if self._cache is not None:
                    return self._cache
                raise
            self._cache = data
            self._cache_at = now
            return data

    @contextmanager
    def _transaction(self):
        """Yield the on-disk workers map under an exclusive cross-process lock.

        Reload -> mutate (caller) -> persist. The yielded dict is written back
        when the block exits without raising. Falls back to a plain in-process
        critical section when ``fcntl`` is unavailable.
        """
        with self._lock:
            self._ensure_parent()
            # Open r+ (create if missing) so we hold one fd for lock+read+write.
            fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
            fh = os.fdopen(fd, "r+", encoding="utf-8")
            try:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                workers = self._read_unlocked(fh)
                yield workers
                self._write_unlocked(fh, workers)
                # Refresh the read-cache so this process sees its own write
                # immediately (and other processes within the TTL).
                self._cache = workers
                self._cache_at = time.time()
            finally:
                try:
                    if fcntl is not None:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                finally:
                    fh.close()

    # -- registration / lifecycle ------------------------------------------
    def register(
        self,
        *,
        name: str,
        url: str,
        gpus: Optional[List[Dict[str, Any]]] = None,
        role: str = "worker",
        models: Optional[List[str]] = None,
        worker_id: Optional[str] = None,
        pkg_version: Optional[str] = None,
        engine_build: Optional[str] = None,
        rpc_endpoint: Optional[str] = None,
        free_ram: Optional[int] = None,
        ram_total: Optional[int] = None,
        engine: Optional[Dict[str, Any]] = None,
        pool: Optional[str] = None,
        caps: Optional[Dict[str, Any]] = None,
        env: Optional[Dict[str, Any]] = None,
        serving_limits: Optional[Dict[str, Any]] = None,
        slot_capable: Optional[bool] = None,
        slot_incapable_reason: Optional[str] = None,
        task_capabilities: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Add a worker (or re-register an existing one by id/url).

        Re-registration is keyed first on the supplied ``worker_id``, then on
        ``url`` — so an agent that restarts and advertises the same URL keeps
        its assignments instead of creating a duplicate row.
        """
        url = (url or "").rstrip("/")
        with self._transaction() as workers:
            existing = None
            if worker_id and worker_id in workers:
                existing = workers[worker_id]
            else:
                for w in workers.values():
                    if w.get("url") == url:
                        existing = w
                        break

            if existing is not None:
                # Grandfather pre-feature rows to approved; never silently revive a
                # blocked worker (the route refuses it, but don't let a re-register
                # flip it back to serving).
                existing.setdefault("admission", "approved")
                existing.update(
                    name=name or existing.get("name"),
                    url=url or existing.get("url"),
                    gpus=gpus if gpus is not None else existing.get("gpus", []),
                    role=role or existing.get("role", "worker"),
                    last_seen=_now(),
                )
                if models is not None:
                    existing["models"] = sorted(set(models))
                if pkg_version is not None:
                    existing["pkg_version"] = pkg_version
                if engine_build is not None:
                    existing["engine_build"] = engine_build   # item L (k65)
                if rpc_endpoint is not None:
                    existing["rpc_endpoint"] = rpc_endpoint
                if free_ram is not None:
                    existing["free_ram"] = free_ram
                if ram_total is not None:
                    existing["ram_total"] = ram_total
                if engine is not None:
                    existing["engine"] = engine
                if caps is not None:
                    existing["caps"] = caps
                if env is not None:
                    existing["env"] = env
                # Concurrency-hardening capability (2026-07-11). Stored verbatim;
                # _public_view spreads them onto /llm/workers rows. A None from an
                # older agent leaves the field untouched (legacy-safe).
                if serving_limits is not None:
                    existing["serving_limits"] = serving_limits
                if slot_capable is not None:
                    existing["slot_capable"] = slot_capable
                    existing["slot_incapable_reason"] = slot_incapable_reason
                # Per-task capability honesty (2026-07-11) — stored verbatim, same
                # legacy-safe idiom: a None from an older agent leaves any prior
                # value untouched. Central's workers_for_model gate reads it.
                if task_capabilities is not None:
                    existing["task_capabilities"] = task_capabilities
                # Only a NON-EMPTY declared pool re-asserts on re-register, so an
                # operator-set pool isn't wiped by a worker that doesn't declare
                # WORKER_POOL (which sends ""). Declaring workers still win.
                if pool and pool.strip():
                    existing["pool"] = pool.strip()
                # 4b organic backfill: every re-register refreshes the
                # assignment memory, so designations that predate the memory
                # feature become durable without an explicit assign.
                if existing.get("models"):
                    _remember_assignments(existing)
                # Durable hardware facts: advance last-known GPU/RAM totals from
                # this reading (never wiped by a transient empty probe) so a
                # re-register that momentarily lacks totals still has them.
                _remember_hw_totals(existing)
                return _public_view(existing)

            wid = worker_id or uuid.uuid4().hex
            # 4b: a fresh row for a KNOWN worker id (its old row was swept /
            # the registry was lost) restores the operator's designations from
            # the assignment memory — designations are worker-lifetime.
            remembered = _load_assign_memory().get(wid) if worker_id else None
            restored_models: List[str] = []
            restored_spill: Dict[str, Any] = {}
            restored_gpu_known = None
            restored_ram_known = None
            if remembered:
                restored_models = list(remembered.get("models") or [])
                restored_spill = dict(remembered.get("spill_by_model") or {})
                # k67 item G — a BLOCKED model has no live placement contract, so
                # its remembered spill row is INERT. Resurrecting it on a registry
                # loss just regrows the exact stale bare rows the operator hand-
                # cleared 2026-07-31 (Jershone~Echo-Mini kept a lingering bare
                # n_gpu_layers=-1 this way). Blocking never authored the row and
                # the model can't route while blocked, so drop it here — the
                # designation stays recorded, only the dead spill is not revived.
                _blocked = _blocked_keys()
                if _blocked:
                    for _mk in list(restored_spill):
                        if _mk in _blocked:
                            restored_spill.pop(_mk, None)
                            logger.info(
                                "register: NOT restoring inert spill row for "
                                "blocked model %s on %s (block leaves the "
                                "designation, drops the dead contract)",
                                _mk, name or wid)
                restored_gpu_known = remembered.get(_GPU_TOTAL_DURABLE_KEY)
                restored_ram_known = remembered.get(_RAM_TOTAL_DURABLE_KEY)
                if restored_models:
                    logger.warning(
                        "register: restoring %d remembered designation(s) for "
                        "returning worker %s (%s): %s", len(restored_models),
                        name or wid, wid, restored_models)
            worker = {
                "id": wid,
                "name": name or wid,
                "url": url,
                "role": role or "worker",
                "gpus": gpus or [],
                "models": sorted(set(models or []) | set(restored_models)),
                "spill_by_model": restored_spill,
                "pkg_version": pkg_version,
                "engine_build": engine_build,   # item L (k65) — native engine commit
                "rpc_endpoint": rpc_endpoint,
                "free_ram": free_ram,
                "ram_total": ram_total,
                "engine": engine,
                "caps": caps,
                # Concurrency-hardening capability (2026-07-11): safe in-process
                # concurrency + whether the box can seat a native crash-isolated
                # slot. None on a pre-feature agent -> central assumes cap 1 and
                # shows no slot badge. See remote._advertised_cap / _public_view.
                "serving_limits": serving_limits,
                "slot_capable": slot_capable,
                "slot_incapable_reason": slot_incapable_reason,
                # Per-task capability honesty (2026-07-11): {task: bool} of the /ml
                # tasks this box can actually run (find_spec probe + a real whisper
                # import). None on a pre-feature agent -> central assumes capable so
                # a legacy fleet routes unchanged. See workers_for_model / _task_capable.
                "task_capabilities": task_capabilities,
                # Runtime-env capability: {"tier": "stable"|"edge"|..., versions}.
                # Read from the worker's own venv, so it's truth not config claim.
                "env": env,
                # Dedicated-pool label. "" = general pool. A pooled worker serves
                # ONLY requests tagged for its pool (reserved capacity); general
                # traffic never lands on it. See workers_for_model.
                "pool": (pool or "").strip(),
                # New workers land pending: they appear in the console but do not
                # serve traffic until an operator admits them (approval-required).
                "admission": "pending",
                "created_at": _now(),
                "last_seen": _now(),
            }
            # Inherit durable hardware facts remembered for this id (a returning
            # worker whose live row was lost keeps its totals immediately), then
            # advance from THIS register's reading if it carried them.
            if restored_gpu_known:
                worker[_GPU_TOTAL_DURABLE_KEY] = restored_gpu_known
            if restored_ram_known:
                worker[_RAM_TOTAL_DURABLE_KEY] = restored_ram_known
            _remember_hw_totals(worker)
            workers[wid] = worker
            return _public_view(worker)

    def heartbeat(
        self,
        worker_id: str,
        *,
        gpus: Optional[List[Dict[str, Any]]] = None,
        loaded_models: Optional[List[str]] = None,
        loading: Optional[List[str]] = None,
        models_local: Optional[List[str]] = None,
        provisioning: Optional[List[str]] = None,
        provision_progress: Optional[Dict[str, Any]] = None,
        spill: Optional[Dict[str, Any]] = None,
        url: Optional[str] = None,
        pkg_version: Optional[str] = None,
        engine_build: Optional[str] = None,
        role: Optional[str] = None,
        rpc_endpoint: Optional[str] = None,
        free_ram: Optional[int] = None,
        ram_total: Optional[int] = None,
        free_ram_raw: Optional[int] = None,
        ram_worker_bytes: Optional[int] = None,
        ram_external_bytes: Optional[int] = None,
        vram_attributed_bytes: Optional[int] = None,
        vram_unattributed_bytes: Optional[int] = None,
        disk: Optional[Dict[str, Any]] = None,
        engine: Optional[Dict[str, Any]] = None,
        pool: Optional[str] = None,
        caps: Optional[Dict[str, Any]] = None,
        env: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        comfy: Optional[Dict[str, Any]] = None,
        loaded_detail: Optional[Dict[str, Any]] = None,
        slots: Optional[List[Dict[str, Any]]] = None,
        allocations: Optional[List[Dict[str, Any]]] = None,
        pid_registry: Optional[Dict[str, Any]] = None,
        storage: Optional[Dict[str, Any]] = None,
        install: Optional[Dict[str, Any]] = None,
        serving_limits: Optional[Dict[str, Any]] = None,
        slot_capable: Optional[bool] = None,
        slot_incapable_reason: Optional[str] = None,
        task_capabilities: Optional[Dict[str, bool]] = None,
        vram_evictions: Optional[Dict[str, Any]] = None,
        vram_holders: Optional[Dict[str, Any]] = None,
        aggregate: Optional[Dict[str, Any]] = None,
        environment_digest: Optional[Dict[str, Any]] = None,
        doctrine_status: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Mark a worker alive and refresh its live GPU / loaded-model stats."""
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            worker["last_seen"] = _now()
            if url:
                worker["url"] = url.rstrip("/")
            if gpus is not None:
                worker["gpus"] = gpus
            if loaded_models is not None:
                # TRUTHFUL residency: the agent's loaded_models only covers its
                # in-process dispatch cache — a model resident in a SLOT child
                # (llama_cpp.server / llama-server it spawned or adopted) is
                # invisible to it, so the console showed "Serving, nothing
                # loaded" while GBs sat on the GPU, and the warm reconcile
                # would re-probe already-warm models. Union in what the slots
                # report about themselves.
                merged = list(loaded_models)
                for s in (slots if slots is not None
                          else worker.get("slots") or []):
                    mk = (s or {}).get("model_key")
                    if mk and s.get("healthy") and mk not in merged:
                        merged.append(mk)
                worker["loaded_models"] = merged
            if loading is not None:
                worker["loading"] = loading   # weights load in flight ("heating")
            if models_local is not None:
                worker["models_local"] = models_local   # disk-truth (UTIL-08)
            if provisioning is not None:
                worker["provisioning"] = provisioning
            if provision_progress is not None:
                # PROGRESS CLOCK (orphan-job lesson: age on PROGRESS, not on any
                # write). The worker re-sends its whole progress map every
                # heartbeat, so the ARRIVAL of this field proves only that the
                # agent is alive — not that any pull is moving. A dead pull's
                # last snapshot keeps replaying verbatim (op sat frozen at
                # frac=0.0722 for 2h+). Carry a central ``progressed_at`` per
                # model, bumped ONLY when done_bytes actually ADVANCES, so
                # _live_provisioning can tell a live 7% from a corpse at 7%.
                prev = worker.get("provision_progress") or {}
                stamped: Dict[str, Any] = {}
                for mk, entry in (provision_progress or {}).items():
                    if not isinstance(entry, dict):
                        stamped[mk] = entry
                        continue
                    entry = dict(entry)
                    old = prev.get(mk) if isinstance(prev, dict) else None
                    old = old if isinstance(old, dict) else {}

                    def _done(e):
                        try:
                            return float(e.get("done_bytes") or 0)
                        except (TypeError, ValueError):
                            return 0.0

                    advanced = _done(entry) > _done(old)
                    carried = old.get("progressed_at")
                    if advanced or carried is None:
                        # First sighting counts as progress: a pull that just
                        # started has moved no bytes yet and must not be born
                        # already-stale.
                        entry["progressed_at"] = _now()
                    else:
                        entry["progressed_at"] = carried
                    stamped[mk] = entry
                worker["provision_progress"] = stamped
            if spill is not None:
                worker["spill"] = spill
            if pkg_version is not None:
                worker["pkg_version"] = pkg_version
            # ENGINE build id beside pkg_version (item L, k65) — the native
            # llama-server commit, so /llm/workers shows engine skew like version
            # skew. Stored verbatim; None on a box with no native engine.
            if engine_build is not None:
                worker["engine_build"] = engine_build
            if role is not None:
                worker["role"] = role
            if rpc_endpoint is not None:
                worker["rpc_endpoint"] = rpc_endpoint
            if free_ram is not None:
                worker["free_ram"] = free_ram
            if ram_total is not None:
                worker["ram_total"] = ram_total
            # Honest budget-bar inputs (t13/t14) — stored verbatim; _ram_summary/
            # _vram_summary compute the spec bar from them. Absent -> the summary
            # flags bar_semantics="legacy" and shows today's numbers.
            if free_ram_raw is not None:
                worker["free_ram_raw"] = free_ram_raw
            if ram_worker_bytes is not None:
                worker["ram_worker_bytes"] = ram_worker_bytes
            if ram_external_bytes is not None:
                worker["ram_external_bytes"] = ram_external_bytes
            if vram_attributed_bytes is not None:
                worker["vram_attributed_bytes"] = vram_attributed_bytes
            if vram_unattributed_bytes is not None:
                worker["vram_unattributed_bytes"] = vram_unattributed_bytes
            if disk is not None:
                worker["disk"] = disk   # model-root volume free/total (preflight)
            if engine is not None:
                worker["engine"] = engine
            if env is not None:
                worker["env"] = env
            if config is not None:
                worker["config"] = config   # effective serving-config + source
            if comfy is not None:
                worker["comfy"] = comfy     # ComfyUI presence (slice A)
            if loaded_detail is not None:
                worker["loaded_detail"] = loaded_detail
            if slots is not None:
                worker["slots"] = slots
            if allocations is not None:
                # Unified engine-agnostic allocation view (slot-seated + in-RAM
                # residents). Stored verbatim; _public_view spreads it through.
                worker["allocations"] = allocations
            if pid_registry is not None:
                # Precision model->PID log (2026-07-14): per-model pid/host_mode/
                # vram + unattributed foreign squatters. Stored verbatim;
                # _public_view spreads it so the console renders it per worker.
                worker["pid_registry"] = pid_registry
            if aggregate is not None:
                # ROLLING AGGREGATE summary (operator ruling 2026-07-29): the
                # COMPACT counts+digest the worker rides on the beat. Stored
                # verbatim; the document itself is never on the beat and is
                # pulled on read via GET /llm/workers/<id>/aggregate. Absent on
                # a pre-aggregate worker -> the key simply stays unset, which
                # is how the relay route reports "not yet aggregating".
                worker["aggregate"] = aggregate
            # k118 ENVIRONMENT DOCTRINE — both stored VERBATIM, both
            # None-guarded so an older worker simply leaves the key unset (which
            # GET /fleet/doctrine/status reports as "no report yet", never as
            # "clean"). `_public_view` spreads the whole record, so both appear
            # in list_workers()/GET /llm/workers with no further plumbing.
            if environment_digest is not None:
                worker["environment_digest"] = environment_digest
            if doctrine_status is not None:
                worker["doctrine_status"] = doctrine_status
            if vram_evictions is not None:
                # VRAM eviction churn (slice 10): stored verbatim so the console
                # can surface GPU evict-to-fit churn beside the disk reaps.
                worker["vram_evictions"] = vram_evictions
            if vram_holders is not None:
                # VRAM-HOLDER METER (incident 2026-08-05): the worker's read-only
                # per-card breakdown + row-per-holder, with idle own-venv
                # squatters flagged. Stored verbatim; _public_view spreads it and
                # derives the per-worker vram_squatters summary (+ a deduped
                # WARNING per distinct squatter). Absent on a pre-feature worker
                # -> the key simply stays unset (meter reads "not reporting").
                worker["vram_holders"] = vram_holders
            if storage is not None:
                # Worker-reported local-storage survey (per-model on-disk bytes +
                # protection flags + cache_used_bytes). Stored verbatim; the
                # over_budget flag + LRU eviction proposal are derived centrally
                # in _public_view via storage_proposal (which overlays the fields
                # the worker can't know: last_picked + the budget).
                worker["storage"] = storage
            if install is not None:
                # Install-shape (uniform-install drift detection): {unit,
                # via_systemd, venv, python, canonical}. Stored verbatim and
                # spread through _public_view (via **worker); the console badges
                # a non-canonical install off it.
                worker["install"] = install
            if caps is not None:
                worker["caps"] = caps
                # Worker-side config is the hard ceiling: if its caps tightened
                # below an operator limit, re-clamp the stored limit now.
                if worker.get("limits"):
                    worker["limits"] = _clamp_limits(worker["limits"], caps)
            # Concurrency-hardening capability (2026-07-11) — refreshed every beat
            # so the console/gate see live truth (a worker that installs the engine
            # binary flips slot_capable within one heartbeat). Legacy-safe: a None
            # from an older agent leaves the fields absent (central assumes cap 1).
            if serving_limits is not None:
                worker["serving_limits"] = serving_limits
            if slot_capable is not None:
                worker["slot_capable"] = slot_capable
                worker["slot_incapable_reason"] = slot_incapable_reason
            # Per-task capability honesty (2026-07-11) — refreshed every beat so an
            # /ops/pip that adds a missing dep flips the task True within one beat.
            # Legacy-safe: a None from an older agent leaves the field absent.
            if task_capabilities is not None:
                worker["task_capabilities"] = task_capabilities
            if pool and pool.strip():   # non-empty only — see register() note
                worker["pool"] = pool.strip()
            # Advance the durable hardware facts from this beat's totals (a beat
            # carries gpus[] w/ memory_total + ram_total). Advance-only, so a
            # beat that transiently omits them keeps the last-known fact.
            _remember_hw_totals(worker)
            return _public_view(worker)

    def remove(self, worker_id: str) -> bool:
        with self._transaction() as workers:
            return workers.pop(worker_id, None) is not None

    # Operator-settable per-worker resource limits. Central may only TIGHTEN:
    # a worker's own configured caps (reported in its heartbeat as ``caps``)
    # are the hard ceiling, so every write is clamped against them.
    #
    # ``disk_cache_gib`` is the OPTIONAL explicit per-worker storage cap (GiB):
    # when set it drives the over-budget flag off cache_used vs the cap (WINS over
    # the free-disk reserve default in storage_proposal), and — unlike the others
    # — has no worker-reported cap, so _clamp_limits passes it through unclamped.
    # More robust than the free-disk reserve against non-model disk pressure.
    # disk_reserve_gib: inference headroom carved OUT of disk_cache_gib (no
    # worker cap -> _clamp_limits passes it through).
    _LIMIT_KEYS = ("ram_max_gib", "gpu_mem_gib", "threads", "disk_cache_gib",
                   "disk_reserve_gib")

    def set_limits(self, worker_id: str,
                   limits: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Set (or clear, with None/{}) central's resource limits for a worker.

        Values are clamped to the worker's self-reported caps — the box's own
        config always wins. Unknown keys are dropped; non-numeric values raise.
        """
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            if not limits:
                worker.pop("limits", None)
                return _public_view(worker)
            clean: Dict[str, Any] = {}
            for k in self._LIMIT_KEYS:
                if k not in limits or limits[k] in (None, ""):
                    continue
                try:
                    clean[k] = float(limits[k]) if k != "threads" else int(limits[k])
                except (TypeError, ValueError):
                    raise ValueError(f"limit {k} must be numeric")
            worker["limits"] = _clamp_limits(clean, worker.get("caps") or {})
            return _public_view(worker)

    def set_pool(self, worker_id: str, pool: str) -> Optional[Dict[str, Any]]:
        """Operator override of a worker's dedicated pool ("" clears). Survives
        heartbeats from workers that don't declare WORKER_POOL (they send "",
        which the register/heartbeat guards ignore)."""
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            worker["pool"] = (pool or "").strip()
            return _public_view(worker)

    def set_auto_reap(self, worker_id: str, enabled: bool) -> Optional[Dict[str, Any]]:
        """Opt this worker into AUTO-REAP (slice 8, Part B; operator ask
        2026-07-17 "there needs to be a way to auto approve this").

        DEFAULT FALSE — the hand-approve flow stays the default posture (defaults
        are promises: central does not self-approve deletions unless the operator
        turned it on for THIS worker). When true, central's heartbeat ingest fires
        EXACTLY the operator reap-approve flow (recompute → intersect → audit →
        guarded relay) once per cooldown when the worker is over budget with a
        non-empty proposal. Persisted on the worker record beside limits/pool, so
        it survives heartbeats and is set through the same operator-gated route
        family. Never widens the blast radius: an auto-fire reclaims at most the
        proposal's need, exactly like a hand-approved one."""
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            worker["auto_reap"] = bool(enabled)
            return _public_view(worker)

    def record_auto_reap(self, worker_id: str, when: float) -> None:
        """Stamp last_auto_reap_at (epoch) so the per-worker cooldown can gate the
        next auto-fire and the console can show when auto-reap last acted. Plain
        stamp — the eviction itself goes through the guarded relay, not here."""
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is not None:
                worker["last_auto_reap_at"] = float(when)

    _ADMISSION_STATES = ("pending", "approved", "blocked")

    def set_admission(self, worker_id: str, state: str) -> Optional[Dict[str, Any]]:
        """Set a worker's admission gate (pending/approved/blocked).

        ``approved`` lets it serve; ``pending`` parks it (visible, idle);
        ``blocked`` evicts it — the register/heartbeat routes refuse a blocked
        worker so its agent stops instead of respawning. Persisted, so the gate
        survives the worker's next heartbeat (unlike ``remove``, which a heartbeat
        would undo).
        """
        if state not in self._ADMISSION_STATES:
            raise ValueError(f"admission must be one of {self._ADMISSION_STATES}")
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            worker["admission"] = state
            return _public_view(worker)

    # -- model assignment ---------------------------------------------------
    def assign_model(
        self,
        worker_id: str,
        model_key: str,
        spill: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Assign a model to a worker, with optional per-assignment spill config.

        ``spill`` is an opaque dict of GPU/CPU knobs (e.g. n_gpu_layers,
        gpu_mem_gib, cpu_mem_gib) the worker applies when it loads the model.
        Omitted / None means "use the worker's autofit default."

        THE ``{}`` CONTRACT IS UNCHANGED AND LOAD-BEARING: an empty spill CLEARS
        any persisted override, so the model reverts to the read-time derivation.
        Two callers depend on that exact meaning — the console's "↺ Auto —
        derived" control, and worker_routes' manifest-orphan cleanup, whose
        safety argument is precisely that an empty spill is STRUCTURALLY
        incapable of writing a contract onto a phantom key. Neither may regress.

        What DID change (2026-07-25): an explicitly-chosen max-gpu no longer
        arrives here as ``{}``. It carries ``{"alloc_mode": "max-gpu"}``, so it
        is a normal non-empty contract and takes the write branch like any other
        mode. Before this, max-gpu was the ONE mode that could not be saved: its
        encoding collided with the clear signal, so choosing it deleted the row
        and the model silently fell through to whatever the derivation said.
        """
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            models = set(worker.get("models", []))
            models.add(model_key)
            worker["models"] = sorted(models)
            if spill is not None:
                by_model = worker.setdefault("spill_by_model", {})
                # An empty dict clears any override back to autofit.
                if spill:
                    by_model[model_key] = spill
                else:
                    by_model.pop(model_key, None)
            _remember_assignments(worker)   # 4b: designations survive row loss
            return _public_view(worker)

    def set_moe(self, worker_id: str, model_key: str,
                value: Optional[bool]) -> Optional[Dict[str, Any]]:
        """Set the MoE-split override: True (force on), False (force off), or
        None (AUTO — follow the derivation, the default).

        None REMOVES the key rather than storing null, so the map holds only real
        operator decisions and an absent entry unambiguously means auto."""
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            by_model = worker.setdefault("moe_by_model", {})
            if value is None:
                by_model.pop(str(model_key), None)
            else:
                by_model[str(model_key)] = bool(value)
            return _public_view(worker)

    def set_bnb(self, worker_id: str, model_key: str,
                enabled: bool) -> Optional[Dict[str, Any]]:
        """Switch the bitsandbytes SPECIALIZATION on/off for one (worker, model).

        Stored in its OWN map, never inside spill_by_model: a spill is a
        PLACEMENT contract and this is a COMPRESSION choice. Keeping them apart
        is what lets "↺ Auto — derived" clear a placement without silently
        dropping the quantization, and lets the quantization re-price the model
        so the derivation can then choose a BETTER placement for it.

        OFF removes the key rather than storing false, so the map only ever
        holds real opt-ins and an absent entry unambiguously means off."""
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            by_model = worker.setdefault("bnb_by_model", {})
            if enabled:
                by_model[str(model_key)] = True
            else:
                by_model.pop(str(model_key), None)
            return _public_view(worker)

    def unassign_model(self, worker_id: str, model_key: str) -> Optional[Dict[str, Any]]:
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            worker["models"] = sorted(set(worker.get("models", [])) - {model_key})
            worker.get("spill_by_model", {}).pop(model_key, None)
            # Hygiene: drop the per-model LRU stamp too, so the model_last_picked
            # map doesn't grow unbounded with unassigned models. Harmless for the
            # eviction proposal — a missing entry defaults to 0 (coldest), which
            # is correct for a now-unassigned leftover.
            worker.get("model_last_picked", {}).pop(model_key, None)
            # …and the call-stats row keyed by the SAME (worker, model) pair.
            # This was the ledger's one unbounded-growth seam: model_last_picked
            # was pruned here from the start, but model_call_stats was not, so
            # every assign/unassign cycle left a permanent orphan row behind —
            # and the 2026-07-25 columns (interval + tok/s EWMAs) widen those
            # rows without pruning them. Dropping it is also CORRECT rather than
            # merely tidy: a re-assigned model is landing on a worker whose
            # placement may have changed entirely, and a stale tok/s mean from
            # the previous placement would be a measurement of a configuration
            # that no longer exists. Missing degrades to "no history", which is
            # honestly what an unassigned-then-reassigned model has.
            worker.get("model_call_stats", {}).pop(model_key, None)
            worker.get("model_tok_stats", {}).pop(model_key, None)
            _remember_assignments(worker)   # 4b: an explicit unassign IS forgotten
            return _public_view(worker)

    # -- placement grants (Phase 1 item 2) -----------------------------------
    # A GRANT is a SYSTEM-authored designation — born from a future
    # capacity-aware placement decision, NOT an operator assign/pin. Stored
    # separately from ``worker["models"]`` so it can never masquerade as
    # operator intent: assign/unassign, storage protection's "assigned" branch,
    # and the assignment-memory snapshot all stay blind to it. A grant is
    # freely LRU-evictable (see storage_proposal) and dies with the live
    # worker row — it is deliberately NOT written to the assign-memory file
    # (_remember_assignments), so a row-loss restore never resurrects it. This
    # method only touches ``worker["grants"]``; ``worker["models"]`` is
    # untouched (orthogonal to assign_model/unassign_model).
    def grant_model(self, worker_id: str, model_key: str,
                    job_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            grants = worker.setdefault("grants", {})
            grants[model_key] = {
                "ts": _now(),
                "job_id": job_id,
                "origin": "system",
            }
            return _public_view(worker)

    def ungrant_model(self, worker_id: str, model_key: str) -> Optional[Dict[str, Any]]:
        """Remove one grant. Idempotent — a missing key is a no-op, not an error."""
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            worker.get("grants", {}).pop(model_key, None)
            return _public_view(worker)

    def spill_for(self, worker_id: str, model_key: str) -> Dict[str, Any]:
        """THE emitted spill for (worker, model) — the placement contract below
        plus the MODEL-scoped polite-load flag (k56).

        ``no_evict`` is added HERE, after the placement derivation, and for the
        same reason the bnb lever is added inside it: politeness is an ADMISSION
        policy, not a placement, so it must ride whatever placement won —
        persisted, derived, or blank max-gpu — without being able to change it.
        It carries its OWN version gate (NO_EVICT_MIN_PKG_VERSION), so a worker
        that predates the flag never receives it; central's resolution already
        declines to route a polite model to such a worker, and this is the
        belt-and-braces at the wire.

        k62 — politeness is per (model × WORKER), and this is the seam where
        that becomes true on the wire: central resolves the effective flag for
        THIS worker and includes or omits the key accordingly. The worker side
        is unchanged; a box simply receives no_evict on the models the operator
        marked polite THERE.
        """
        out = self._placement_spill_for(worker_id, model_key)
        try:
            from hugpy_engine.serve.overrides import placement_policy, resolve_polite
            from hugpy_engine.alloc_modes import (
                worker_honors_no_evict,
                no_evict_downgrade_note,
                NO_EVICT_SPILL_KEY,
            )
            _prefs, polite, by_worker = placement_policy(model_key)
            if not (polite or by_worker):
                return out
            worker = self._load().get(worker_id) or {}
            forms = _worker_forms(worker) | {str(worker_id).strip().lower()}
            if not resolve_polite(polite, by_worker, forms):
                logger.debug("polite-load flag omitted for %s on %s: not polite "
                             "on this worker", model_key, worker_id)
                return out
            if not worker_honors_no_evict(worker.get("pkg_version")):
                logger.warning("polite load for %s: %s", model_key,
                               no_evict_downgrade_note(
                                   worker.get("pkg_version"),
                                   worker.get("name") or worker_id))
                return out
            out = dict(out)
            out[NO_EVICT_SPILL_KEY] = True
        except Exception:  # noqa: BLE001 — never break the relay over the flag
            logger.debug("polite-load flag skipped for %s on %s", model_key,
                         worker_id, exc_info=True)
        return out

    def _placement_spill_for(self, worker_id: str, model_key: str) -> Dict[str, Any]:
        """Per-assignment spill override for (worker, model), or {} for
        max-gpu (autofit). THE version-gated emission seam (k37): a spill
        carrying the NEW allocation-mode keys (alloc_mode/leniency_pct/
        priority_device — max-ram/explicit) is only emitted to a worker whose
        pkg_version honors them; an older worker gets {} (max-gpu autofit) for
        the request and the downgrade is logged honestly — a selected mode
        must never be a silent dead knob. The PERSISTED contract is untouched
        (it applies the moment the worker updates)."""
        worker = self._load().get(worker_id)
        if worker is None:
            return {}
        spill = dict(worker.get("spill_by_model", {}).get(model_key, {}))
        # CAPABILITY-AWARE BLANK DEFAULT (operator ruling 2026-07-24): when
        # NOTHING placement-affecting is persisted for this (worker, model), the
        # blank default is derived by FEASIBILITY instead of the flat max-gpu.
        # A transformers model far too big for this box's GPU but that fits RAM
        # defaults to ram-only ON THIS WORKER, so its blank default SERVES
        # (defaults-are-promises) instead of a doomed all-GPU attempt. GGUF and
        # any fitting/unresolvable case keep max-gpu ({} — unchanged). This only
        # ever touches the BLANK case; an explicit persisted contract is left
        # exactly as-is. The emitted encoding is the LEGACY n_gpu_layers key —
        # no version gate needed (every worker version honors it).
        #
        # STRUCTURE-DERIVED (2026-07-25): the derivation is now the operator's
        # FULL decision tree, so a blank MoE GGUF resolves to its own split
        # (explicit + n_cpu_moe + the two budgets) instead of the blanket stamp.
        # The MoE case is the reason this returns a whole spill rather than a
        # mode name — see alloc_modes.default_allocation for the wire rationale.
        # The derived spill goes through the SAME version gate as a persisted
        # one: a pre-0.1.203 worker gets {} (max-gpu), whose own load-time auto
        # MoE policy reaches the right placement anyway.
        if not (set(spill) & _PLACEMENT_SPILL_KEYS):
            try:
                derived = derived_default_allocation(worker, model_key)
                mode = derived.get("mode") or "max-gpu"
                out = dict(derived.get("spill") or {})
            except Exception:  # noqa: BLE001 — never break the relay over a derive
                mode, out = "max-gpu", {}
            if mode == "max-gpu" or not out:
                # Blank max-gpu — but the 4-bit lever must still ride, or a
                # max-gpu model (the commonest case!) silently loads fp16. This
                # is the MN-GRAND report: console showed 13.1 GiB planned, the
                # worker asked for 50.2 GB and refused.
                return {"bnb_4bit": True} if bnb_enabled(worker, model_key) else {}
            logger.info("blank default for %s on %s derived to %s (%s)",
                        model_key, worker.get("name") or worker_id, mode,
                        derived.get("why") or "structure-derived")
            # The 4-bit lever rides the wire whenever it is ON, regardless of
            # which allocation mode was derived: it is a COMPRESSION choice, so
            # it applies to a max-gpu model exactly as much as a ram-only one.
            # Without this the worker loaded fp16 and refused ("needs 50.2 GB")
            # while the console showed the 4-bit projection — the lever changed
            # central's arithmetic but never reached the loader.
            if bnb_enabled(worker, model_key):
                out["bnb_4bit"] = True
            # A DERIVED gpu-only for a MoE (a fitting MoE whose leaf resolved to
            # full-card residency) has the same bare-'-1' gap as a persisted one:
            # suppress the worker's auto expert-split. The derivation already
            # threaded the explicit MoE toggle via moe_force, so this only fills
            # the AUTO gap; a derived expert SPLIT reads as "explicit", not
            # "gpu-only", and is left untouched. Adding n_cpu_moe:0 also makes the
            # spill version-gated below (n_cpu_moe is a _NEW_SPILL_KEYS_LOCAL key),
            # so a pre-0.1.203 worker safely degrades to {} (its own auto policy).
            suppress_moe_split_for_gpu_only(worker, model_key, out)
            if not (set(out) & _NEW_SPILL_KEYS_LOCAL):
                return out                       # legacy wire: no gate needed
            try:
                from hugpy_engine.alloc_modes import gate_spill_for_worker
                gated, note = gate_spill_for_worker(
                    out, worker.get("pkg_version"),
                    worker.get("name") or worker_id)
                if note:
                    logger.warning(
                        "derived default for %s on %s downgraded: %s",
                        model_key, worker.get("name") or worker_id, note)
                return gated
            except Exception:  # noqa: BLE001 — the gate must never break relaying
                return {}                        # fail SAFE: unproven -> max-gpu
        # PERSISTENCE-ONLY MODES (2026-07-25): an explicitly-chosen max-gpu is
        # persisted as {"alloc_mode": "max-gpu"} so it is distinguishable from a
        # CLEAR ({}), but that key is central's bookkeeping — never the worker's.
        # Strip it BEFORE the version gate so the emitted wire is byte-identical
        # to a blank max-gpu on every worker version, and so the gate is not
        # engaged by a key that carries no instruction (an old worker would
        # otherwise be "downgraded" from max-gpu to max-gpu, logging a scary and
        # entirely fictional note). See _WIRE_INERT_MODES for why sending it
        # would actively regress the MoE auto-split.
        spill = _strip_wire_inert_mode(spill)
        # An operator-pinned placement does not cancel the compression lever —
        # the two are independent axes (see bnb_enabled's docstring).
        if bnb_enabled(worker, model_key):
            spill["bnb_4bit"] = True
        # NOR does it cancel the MoE-split lever. moe_by_model is a SEPARATE axis
        # from spill_by_model (like bnb): a pinned placement key must not silently
        # drop the operator's MoE checkbox, or the console shows "MoE: ON" while
        # the wire never carries n_cpu_moe (serving-core mechanics §8 / C2). The
        # blank branch already threads this via derived_default_allocation; this
        # is the persisted-spill counterpart.
        apply_moe_override_to_spill(worker, model_key, spill)
        # gpu-only is "everything on GPU" — for a MoE that means the experts too.
        # A persisted gpu-only (spill_by_model={'n_gpu_layers':-1}) carries no
        # n_cpu_moe, so the worker's auto-MoE policy would offload experts to CPU
        # unless we say otherwise. Emit the explicit n_cpu_moe:0 the mode implies.
        # AFTER apply_moe_override_to_spill so the explicit toggle wins and this
        # only fills the AUTO gap (see the helper's docstring for the composition).
        suppress_moe_split_for_gpu_only(worker, model_key, spill)
        try:
            from hugpy_engine.alloc_modes import gate_spill_for_worker
            gated, note = gate_spill_for_worker(
                spill, worker.get("pkg_version"),
                worker.get("name") or worker_id)
            if note:
                logger.warning("alloc-mode downgrade for %s on %s: %s",
                               model_key, worker.get("name") or worker_id, note)
            return gated
        except Exception:  # noqa: BLE001 — the gate must never break relaying
            return spill

    def set_load_report(self, worker_id: str, model_key: str,
                        report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Record the outcome of a warm/probe attempt for (worker, model).

        ``report`` is the worker's /probe response ({ok, fit, vram_used, error, …})
        plus a ``ts`` stamp, or a synthesized {ok: False, error} when the probe
        HTTP call itself failed. ``None`` clears the entry (e.g. on unassign).
        Stored under ``load_reports[model_key]`` on the worker record so the
        console can say WHY a model stayed cold instead of showing a silent
        no-op activate."""
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            reports = worker.setdefault("load_reports", {})
            if report is None:
                reports.pop(model_key, None)
            else:
                reports[model_key] = report
            return _public_view(worker)

    # -- queries ------------------------------------------------------------
    def get(self, worker_id: str) -> Optional[Dict[str, Any]]:
        worker = self._load().get(worker_id)
        return _public_view(worker) if worker else None

    def all(self) -> List[Dict[str, Any]]:
        return [_public_view(w) for w in self._load().values()]

    def storage_view(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """The derived storage view + LRU eviction proposal for one worker,
        computed from its RAW record (NOT the _public_view output, whose
        ``storage`` key is already the derived shape). This is the /reap-approve
        route's second-guard recompute: it must read the raw worker-reported
        ``storage`` survey to re-derive the CURRENT proposal at approve time.
        ``None`` if the worker is unknown."""
        worker = self._load().get(worker_id)
        return storage_proposal(worker) if worker else None

    def workers_for_model(self, model_key: str, *, online_only: bool = True,
                          pool: Optional[str] = None,
                          task: Optional[str] = None,
                          require_comfy_id_lock: bool = False) -> List[Dict[str, Any]]:
        # Operator BLOCK gate (central pool primitive): a blocked model is
        # removed from the pool entirely — no worker is EVER a candidate for it,
        # regardless of assignment/pin. Block outranks pin (pin is routing
        # persistence; block is an operator override), so a pinned+blocked model
        # simply yields no candidates here while its designation row stays
        # recorded (inert). One honest log line, same say-why spirit as the gates
        # below; returns before any per-worker work.
        if _model_blocked(model_key):
            import logging as _logging
            _logging.getLogger(__name__).info(
                "model %s is BLOCKED from the serving pool by the operator — "
                "no worker is a routing candidate (unblock to route again)",
                model_key)
            return []
        wanted = _match_keys(model_key)
        want_pool = (pool or "").strip()
        need_tier = env_tier_for_model(model_key)
        # WILDCARD map, read ONCE per call (not per worker-loop iteration) —
        # the per-worker lookup below is then a dict hit. Guarded inside
        # _wildcard_map: a store miss degrades to "nobody is a wildcard".
        wildcards = _wildcard_map()
        tier_skipped = 0
        task_skipped = 0
        id_lock_skipped = 0
        infeasible_skipped = 0
        engine_skipped = 0
        wildcard_engine_skipped = 0
        alloc_scope_skipped = 0
        rows = self.all()
        # ALLOCATION IS HARD SCOPE (operator ruling 2026-08-28, coder-next/
        # computron): a model that HAS designation rows anywhere (operator
        # assignment / SYSTEM grant) lands ONLY on those workers — a wildcard
        # box never catches an ALLOCATED model, online or busy, initial pick
        # or reroute. Rationale from the live incident: coder-next is
        # allocated to ae; when ae hit its concurrency cap the reroute walk
        # offered wildcard computron (whose 8+16 GiB box can never hold the
        # 45 GiB file) and the caller got computron's refusal instead of the
        # honest worker_busy. Unallocated models keep today's fleet-wide
        # wildcard behavior byte-identically. Residency (loaded_models) is
        # deliberately NOT part of the predicate — a transient warm copy on a
        # wildcard box must not scope-lock the model to it.
        _alloc_ids: set = set()
        for _w0 in rows:
            _des = list(_w0.get("models", [])) + list(_w0.get("grants", {}).keys())
            if _des and _serveable_match(model_key, wanted, _des):
                _alloc_ids.add(_w0.get("id") or "")
        # Durable per-(model × worker) blocks (the pair-block half of the same
        # ruling) — one normalized key so bare / "~"-qualified spellings of
        # the same model land on one record (the dispatch cache uses the BARE
        # key; the console the qualified one).
        _pair_key = str(model_key).split("~")[-1].strip().lower()
        out = []
        for w in rows:
            # Only admitted workers serve. Pending (awaiting operator approval) and
            # blocked workers are never picked for inference, even if assigned.
            if w.get("admission") != "approved":
                continue
            # Capability guard: skip a worker that reports no inference engine —
            # it would accept the dispatch and fail, wasting a hop before the
            # local fallback. (Workers not reporting engine status are kept.)
            if _engine_unusable(w):
                # Say-why parity with the tier/task/id_lock gates below: count a
                # worker only when it is otherwise ASSIGNED to this model, so an
                # empty result's log names the real cause (a DESIGNATED worker
                # whose engine can't serve — the "assigned+pinned but 500s"
                # mystery) instead of every engine-broken box on the fleet.
                _serveable = (list(w.get("models", [])) + list(w.get("loaded_models", []))
                              + list(w.get("grants", {}).keys()))
                if _serveable_match(model_key, wanted, _serveable):
                    engine_skipped += 1
                elif wildcards.get(w.get("id") or ""):
                    # A WILDCARD CATCH lost to the engine gate. Counted apart
                    # from the designated skips so the say-why log below stays
                    # truthful ("assigned" vs "wildcard") without new machinery.
                    wildcard_engine_skipped += 1
                continue
            # Dedicated-pool reservation: a request for pool P uses ONLY pool-P
            # workers; a general request (no pool) uses ONLY un-pooled workers.
            # So dedicated capacity is reserved for its app and never consumed by
            # general traffic — and a pool request that finds no pool worker
            # falls back to local (caller's None handling), not to the shared pool.
            if (w.get("pool") or "").strip() != want_pool:
                continue
            # Candidates = models this worker is ASSIGNED **or currently reports
            # LOADED**. A worker holding the model warm (loaded via probe, or
            # left resident after an unassign) is the best possible server —
            # ignoring it sent the request to a cold local fallback while a
            # GPU sat there with the weights already up. Loaded-ness is
            # heartbeat-fresh; if it evicts between beats the relay fails
            # pre-token and the caller falls back as always.
            # Grants (SYSTEM-authored placement, Phase 1 item 2) are serveable
            # exactly like an operator assignment or a live-loaded model —
            # once a granted model is actually held by the worker it must
            # route, or the grant is pointless. Grants confer NO eviction
            # protection (see storage_proposal) — this is purely "can serve",
            # not "may not be reclaimed".
            serveable = (list(w.get("models", [])) + list(w.get("loaded_models", []))
                         + list(w.get("grants", {}).keys()))
            # Match on the raw key OR any normalized alias (hub_id vs key vs
            # case vs "~"-qualification), so an assignment made via one form
            # still routes a chat that names the model a slightly different way.
            # _serveable_match also carries the blocked-sibling guard: an
            # alias-only match against a BLOCKED advertised sibling never
            # counts. RESIDENT = DE FACTO DESIGNATION: ``loaded_models`` (and
            # grants) ride in ``serveable``, so a box currently holding the
            # model is ALWAYS a "home" match here — never route-refused —
            # wildcard flag or not.
            home = _serveable_match(model_key, wanted, serveable)
            if not home:
                # WILDCARD PLACEMENT (operator doctrine 2026-07-23):
                # designations are a HARD routing scope — an unmatched worker
                # is out UNLESS it explicitly opted in as a wildcard ("a worker
                # can be designated to take all comers ... or it can not be
                # selected as a wildcard and adhere only to its own allocated
                # models"). A wildcard catch relaxes ONLY this
                # designation-membership gate: every hard gate around it still
                # applies — admission/engine/pool above, liveness/env-tier/
                # task-capability/id-lock below, and the requested key's BLOCK
                # gate already returned [] before the loop. Default False for
                # every worker (absent key = not a wildcard), so a fleet with
                # no flags set routes exactly as before this feature existed.
                if not wildcards.get(w.get("id") or ""):
                    continue
                if _alloc_ids and (w.get("id") or "") not in _alloc_ids:
                    # An ALLOCATED model never falls to a wildcard box off its
                    # allocation (hard scope — see the block comment above).
                    alloc_scope_skipped += 1
                    continue
            if online_only and w["status"] != "online":
                continue
            # Runtime-env tier gate: the model runs ONLY on a worker whose venv
            # tier matches (strict both ways — an edge env can regress stable
            # models just as a stable env can't load edge architectures). Both
            # sides default to "stable", so an unmapped model on an unreporting
            # fleet routes exactly as before this gate existed.
            if _worker_env_tier(w) != need_tier:
                tier_skipped += 1
                continue
            # Per-task capability gate (2026-07-11): skip a worker that
            # AFFIRMATIVELY advertises it can't run this task (a canonical venv
            # missing an optional ML dep — sentence-transformers / whisper /
            # keybert). Legacy/unknown = capable, so a pre-feature fleet is
            # untouched; a None task never gates. Same say-why idiom as the tier
            # gate below.
            if not _task_capable(w, task):
                task_skipped += 1
                continue
            # ID-LOCK routing gate (identity-locked STILLs): an id_lock image
            # request must land on a box whose ComfyUI PROVABLY has the IPAdapter
            # nodes (comfy.id_lock). Affirmative-only — never route id_lock to a
            # comfy-less / nodeless worker where it would fail at request time (or
            # worse, tempt a silent non-locked fallback). Off (False) for every
            # other request, so ordinary routing is untouched.
            if require_comfy_id_lock and not _comfy_id_lock_capable(w):
                id_lock_skipped += 1
                continue
            # STATIC-FEASIBILITY gate (k67, resolution-stage-pipeline): never
            # OFFER a worker whose refusal central already knows — a model that
            # does not fit this box's GPU+RAM combined cannot land here in ANY
            # mode, so routing it only produces an honest but wasteful refusal
            # (the computron case: bare-key 51.8 GB GGUF offered to an 8 GiB card,
            # over and over). AFFIRMATIVE-only: worker_can_hold returns False just
            # when the numbers are known AND confidently don't fit; None (unsized
            # model / unmeasured box) never eliminates (degrade-not-guess). A HOME
            # match is exempt — a box already holding/serving the model is feasible
            # de facto and must never be route-refused by a static estimate.
            # De-facto feasibility is MEASURED RESIDENCY, not assignment
            # (2026-08-28, 35B-Distill × computron): a box actually holding
            # the model is feasible by observation, but a bare assignment is
            # a CLAIM the static verdict may refute — the operator assigned a
            # 38.6 GB model to a 25 GB-combined box, the old home exemption
            # admitted it as a candidate, and every reroute onto it harvested
            # the same honest refusal. Per the ruling ("if allocated, should
            # be blocked; the user will be forced to acknowledge it") the
            # infeasible ASSIGNED pair is gated + pair-blocked exactly like a
            # wildcard one; only a resident copy stays exempt.
            _resident = home and _serveable_match(
                model_key, wanted, list(w.get("loaded_models", [])))
            if not _resident:
                # Pair-block wiring (operator ruling 2026-08-28): a POSITIVE
                # permanent-no verdict is recorded as a durable, console-
                # visible per-(model × worker) block; a True verdict clears
                # it (quant pin shrank the model / the box grew); a None
                # verdict (missing data — e.g. the bare dispatch-cache key
                # can't resolve a size) consults the durable record instead
                # of admitting the pair the fleet already knows can't fit.
                # Missing data on its own still never RECORDS anything.
                _verdict = worker_can_hold(w, model_key)
                _wid = w.get("id") or ""
                try:
                    from hugpy_fleet.central import blocklist as _bl
                    if _verdict is False:
                        _bl.auto_block_pair(
                            _pair_key, _wid,
                            why=("does not fit this box's combined GPU+RAM "
                                 "capacity in any mode (static verdict)"),
                            worker_name=w.get("name"))
                    elif _verdict is True:
                        _bl.clear_pair_block(_pair_key, _wid)
                    elif _bl.pair_blocked(_pair_key, _wid):
                        infeasible_skipped += 1
                        continue
                except Exception:  # noqa: BLE001 — the store must never break routing
                    pass
                if _verdict is False:
                    infeasible_skipped += 1
                    continue
            if not home:
                # Transient RESPONSE-COPY marker: ``w`` is a _public_view copy
                # (self.all() rebuilds it from the store on every read), so this
                # can never leak into the persisted record — same transient
                # semantics as the derived status/storage fields. Ranking sorts
                # home matches ABOVE wildcard catches (pick_for_model /
                # candidates_for_model), which IS the overflow mechanism; and
                # say-why readers can tell a candidate is here by wildcard, not
                # designation.
                w["_wildcard_catch"] = True
            out.append(w)
        if not out and engine_skipped:
            # The model HAS designated servers — every one was excluded because it
            # AFFIRMATIVELY reports its inference engine is unusable (llama-cpp not
            # loadable AND no native llama-server binary; engine.installed=False).
            # Name the cause so the operator repairs the box (`hugpy install-engine`
            # / reinstall llama-cpp-python) instead of seeing only the downstream
            # "no worker available / local serving disabled" 500 with no reason.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "model %s: %d assigned worker(s) skipped — inference engine "
                "unusable (llama-cpp not loadable AND no native llama-server "
                "binary). Repair the engine on those boxes or assign the model to "
                "a healthy worker.", model_key, engine_skipped)
        if not out and wildcard_engine_skipped:
            # WILDCARD catches lost to the engine gate — kept apart from the
            # designated-worker warning above so "assigned" never overcounts.
            # Matters when a model's ONLY possible servers were wildcard boxes:
            # without this line an empty result would look like "no designation"
            # instead of "the all-comers boxes are engine-broken".
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "model %s: %d wildcard (all-comers) worker(s) skipped — "
                "inference engine unusable", model_key, wildcard_engine_skipped)
        if not out and tier_skipped:
            # The model HAS servers — they were excluded on env tier alone. Say
            # so, or the operator sees only the downstream "no worker / local
            # fallback disabled" error with no cause.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "model %s requires env tier %r; %d otherwise-eligible worker(s) "
                "skipped (none advertise that tier)",
                model_key, need_tier, tier_skipped)
        if not out and task_skipped:
            # The model HAS servers — they were excluded on task capability alone
            # (they advertise they can't run this task). Name the reason, or the
            # operator sees only the downstream no-worker error with no cause.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "model %s task %r: %d otherwise-eligible worker(s) skipped "
                "(task unavailable — missing optional ML dependency on those boxes)",
                model_key, task, task_skipped)
        if not out and id_lock_skipped:
            # The model HAS servers — every one was excluded because its ComfyUI
            # lacks the IPAdapter node pack (comfy.id_lock False/absent). Name the
            # cause so the operator installs it (WORKER-SETUP §5b) instead of
            # seeing only the downstream no-worker error.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "model %s id_lock: %d otherwise-eligible worker(s) skipped — no "
                "box advertises comfy.id_lock (install ComfyUI_IPAdapter_plus + "
                "weights per WORKER-SETUP §5b)", model_key, id_lock_skipped)
        if not out and infeasible_skipped:
            # The model HAS servers — every one was excluded because the model
            # does not fit that box's GPU+RAM combined (statically infeasible).
            # Name it so the operator sees the honest capacity wall instead of a
            # bare no-worker error, and knows to (re)assign a box that can hold it
            # or split it differently. k67.
            import logging as _logging
            _logging.getLogger(__name__).info(
                "model %s: %d designated worker(s) skipped — model does not fit "
                "their GPU+RAM combined (statically infeasible; assign a larger "
                "box or a smaller quant)", model_key, infeasible_skipped)
        if not out and alloc_scope_skipped:
            # Say-why parity: the model IS allocated somewhere, and only
            # off-allocation wildcard boxes were available right now.
            import logging as _logging
            _logging.getLogger(__name__).info(
                "model %s is ALLOCATED (%d worker(s)); %d wildcard box(es) "
                "outside its allocation were skipped by hard scope — its own "
                "workers are unavailable/busy right now", model_key,
                len(_alloc_ids), alloc_scope_skipped)
        return out

    def pick_for_model(self, model_key: str, pool: Optional[str] = None,
                       task: Optional[str] = None,
                       require_comfy_id_lock: bool = False) -> Optional[Dict[str, Any]]:
        """Choose an online worker to serve ``model_key`` (optionally within a
        dedicated ``pool``, and — when set — one that can run ``task``).

        ``require_comfy_id_lock`` (set for identity-locked STILL requests) further
        restricts to boxes whose ComfyUI advertises the IPAdapter nodes.

        Preference order:
            0. the operator's ORDERED worker preference (k56), when the model
               carries one: candidates are restricted to the list and tried in
               its order, first admitting worker wins. A polite (``no_evict``)
               model additionally requires a candidate that can take it out of
               genuinely free room; if none can, this returns None with the
               reason on the telemetry feed rather than evicting anyone.
            1. HOME workers (designated / resident / granted) before wildcard
               catches — this ordering IS the overflow mechanism (operator
               doctrine 2026-07-23): a designated model tries its home workers
               first and spills onto wildcard ("take all comers") boxes only
               when no home worker survives the gates; it busts only when
               neither can serve. No separate overflow machinery exists.
            2. MEASURED-RESIDENT workers — the model is loaded there right now
               (alias-tolerant; a live slot seat counts).
            3. ALLOCATED workers — an approved allocation / placement grant for
               the model lives on that box even if it isn't loaded this second.
            4. otherwise capability rank as before: ⭐ star, GPU, then the
               least-recently-picked online assignee.

        Tiers 2 and 3 are the 2026-07-28 fix; see the ROUTING RANK block comment
        above ``_resident_on`` for the incident and why the old single ``warm``
        term missed both.

        Returns ``None`` when no online worker (in the requested pool) is assigned
        to the model, which signals the caller to fall back to local execution.
        """
        candidates = self.workers_for_model(
            model_key, online_only=True, pool=pool, task=task,
            require_comfy_id_lock=require_comfy_id_lock)
        if not candidates:
            # Fall back to assigned workers even with a stale heartbeat. Heartbeat
            # (worker->central) can time out when central is briefly slow, while
            # offload (central->worker) still works — so an assigned worker that
            # looks "offline" is often still serviceable. The stream proxy fails
            # fast to local if the worker is genuinely unreachable.
            candidates = self.workers_for_model(
                model_key, online_only=False, pool=pool, task=task,
                require_comfy_id_lock=require_comfy_id_lock)
        if not candidates and (pool or "").strip():
            # PHANTOM-POOL RESCUE: a pool restriction only means something when the
            # pool exists. If NO registered worker carries this pool tag at all
            # (e.g. a client still sending the old default pool="ml" on a fleet
            # that never tagged one), honoring it would silently strand the request
            # on central-local even though a general worker serves the model. That
            # is the exact bug the un-pooled client default fixed — cover stale
            # clients here too. A pool with members but none available keeps the
            # reservation semantics: no crossover, local fallback.
            want_pool = pool.strip()
            pool_exists = any((w.get("pool") or "").strip() == want_pool
                              for w in self.all())
            if not pool_exists:
                import logging as _logging
                _logging.getLogger(__name__).info(
                    "pool %r has no registered workers; treating request "
                    "for %s as general (un-pooled)", want_pool, model_key)
                return self.pick_for_model(
                    model_key, pool=None, task=task,
                    require_comfy_id_lock=require_comfy_id_lock)
        if not candidates:
            return None

        # Version gate (soft): prefer workers running central's required package
        # version, so a chat doesn't land on a worker mid-rollout that's still on
        # old code. Soft — if NONE have converged yet, we still serve from the
        # (stale-but-working) assignees rather than forcing a local-only outage
        # during the ~heartbeat-long update window.
        required = required_pkg_version()
        if required:
            matched = [w for w in candidates if w.get("pkg_version") == required]
            if matched:
                candidates = matched

        # k56 — the operator's ORDERED worker preference. A HARD scope (a model
        # with a list never lands off it), applied after the eligibility gates
        # so a listed-but-blocked/incapable worker is skipped rather than
        # bypassing them. Absent list ⇒ untouched.
        prefs, polite, polite_by_worker = placement_policy(model_key)
        if prefs:
            candidates = _prefs_scope(candidates, prefs, model_key)
            if not candidates:
                _emit_route_refuse(
                    model_key,
                    f"no worker on the preference list {prefs} is an eligible "
                    f"candidate right now", [])
                return None

        # Ranking (capability already filtered above) — the shared _routing_rank
        # key: designation, then MEASURED-RESIDENT, then ALLOCATED, then the
        # capability rank that was always here (⭐ boot star as the ambiguity
        # tie-break, GPU over CPU-only, least-recently-picked to spread load,
        # stable id so the order never wobbles). See the ROUTING RANK block
        # comment above _resident_on. (Full need-vs-capacity placement is the
        # allocator's job; this is the lightweight default pick.)
        # Read the star store ONCE per pick (never per candidate), alias-tolerant
        # via _match_keys (same unification Slice A uses): a star recorded under a
        # ~-qualified key still matches a bare-key request and vice versa.
        star_map = _star_map()
        wanted_forms = _match_keys(model_key)

        def _starred(w: Dict[str, Any]) -> bool:
            s = star_map.get(w.get("id"))
            return bool(s) and bool(wanted_forms & _match_keys(str(s)))

        def _rank(w: Dict[str, Any]):
            return _routing_rank(w, model_key, wanted_forms, _starred(w),
                                 (_pref_index(w, prefs) or 0) if prefs else 0)
        candidates.sort(key=_rank)
        chosen = candidates[0]

        # k56 POLITE LOAD — "first whose admission accepts", walked in the order
        # just sorted. The unflagged path never enters here (nothing polite
        # anywhere), so this is purely additive. When nothing admits we return
        # None with the honest reason on the telemetry feed rather than picking a
        # box that would have to evict someone: that refusal IS the feature.
        #
        # k62: politeness is resolved PER CANDIDATE. A candidate the model is not
        # polite on skips the free-room gate entirely and wins on the ordinary
        # declare-need-then-evict rule — which is exactly the point of the map
        # (polite on the contended card, assertive on the spare one), and is why
        # the hold log names the WORKER whose politeness caused each skip.
        if polite or polite_by_worker:
            reasons = []
            chosen = None
            for w in candidates:
                wname = w.get("name") or w.get("id")
                if not _polite_on(w, polite, polite_by_worker):
                    logger.info("polite load: %s is NOT polite on %s — ordinary "
                                "admission applies", model_key, wname)
                    chosen = w
                    break
                ok, why = _polite_admits(w, model_key)
                reasons.append(dict(w, _refuse_reason=f"polite on {wname}: {why}"))
                if ok:
                    logger.info("polite load: %s admits %s without evicting (%s)",
                                wname, model_key, why)
                    chosen = w
                    break
            if chosen is None:
                logger.warning(
                    "polite load: no candidate admits %s without eviction — "
                    "holding rather than evicting a resident (%s)", model_key,
                    "; ".join(f"{r.get('name') or r.get('id')}: "
                              f"{r.get('_refuse_reason')}" for r in reasons))
                _emit_route_refuse(
                    model_key, "no candidate admits without eviction", reasons)
                return None

        # WHY this box (operator incident 2026-07-28) — emitted on the PICK, not
        # on the reroute walk, so the feed carries exactly one selection event
        # per dispatch and the operator can read residency/allocation adherence
        # straight off /llm/evictions instead of inferring it from outcomes.
        _emit_route_select(model_key, chosen, candidates, wanted_forms,
                           tok_s=tok_hint_for(chosen, model_key))

        # Persist the pick so round-robin survives across processes.
        with self._transaction() as workers:
            stored = workers.get(chosen["id"])
            if stored is not None:
                now = _now()
                stored["last_picked"] = now
                # Per-(worker,model) LRU signal for the storage eviction proposal.
                # ``last_picked`` above is a SINGLE per-WORKER round-robin scalar,
                # stamped on EVERY pick regardless of model — it spreads load, it
                # can't key an LRU-per-model eviction. This map records the last
                # time THIS model was routed to THIS worker (the authoritative
                # "central served (worker, model)" event), so storage_proposal can
                # sort candidates oldest-first; a model never served through
                # central has no entry -> defaults to 0 -> proposed first.
                stored.setdefault("model_last_picked", {})[model_key] = now
                # ── THE ONE LEDGER's second column: TOTAL CALLS (key ③ of the
                # eviction sort, spec assets/evictionflow.html 2026-07-25).
                #
                # Stamped HERE, beside last_picked, and for the same reason: this
                # is the authoritative "central served (worker, model)" event, so
                # the count and the clock are incremented by the SAME line of
                # code and can never describe different histories. A separate
                # counter elsewhere would be a second ledger, and the Parity
                # invariant is specifically about there being only one.
                #
                # Shipped to the worker on the heartbeat reply (it rides
                # _public_view like model_last_picked, and the worker adopts it
                # in _adopt_storage_inputs), so the worker's auto-evict ranks by
                # CENTRAL's counts rather than its own — which is what makes the
                # two sides' victim sets identical rather than merely similar.
                _cs = stored.setdefault("model_call_stats", {})
                _row = _cs.setdefault(model_key, {"calls": 0})
                _prev_call = _row.get("last_call")
                try:
                    _row["calls"] = int(_row.get("calls") or 0) + 1
                except (TypeError, ValueError):
                    _row["calls"] = 1
                # ── CALL INTERVAL (operator, 2026-07-25) — see _record_interval
                # for the full rationale (point-estimate vs distribution, and
                # why log space). Stamped HERE, on the line that already holds
                # the clock, so the count, the clock and the interval can never
                # describe different histories. Inert: nothing reads it yet.
                _record_interval(_row, _prev_call, now)
                _row["last_call"] = now
                chosen = stored
        # tok/s IS A VARIABLE IN EVERY SELECTION (operator 2026-08-22): read
        # the (worker, model) rollup and carry it on the decision — log line,
        # route.select telemetry, and the returned record (``selection_tok_s``).
        # Ranking is untouched; this makes the variable PRESENT, not decisive.
        hint = tok_hint_for(chosen, model_key)
        logger.info("pick %s -> %s tok/s last=%s avg=%s n=%s",
                    model_key, chosen.get("name") or chosen.get("id"),
                    hint["last_tok_s"], hint["avg_tok_s"], hint["n"])
        out = _public_view(chosen)
        out["selection_tok_s"] = hint
        return out

    def record_serve_metrics(self, worker_id: str, model_key: str,
                             tok_s: Optional[float] = None,
                             **meta: Any) -> bool:
        """THE ONE WRITER for measured serve quality on the shared ledger.

        Stamps decode rate onto ``model_call_stats[model_key]`` — the same row
        ``pick_for_model`` stamps ``calls``/``last_call``/the interval columns
        onto, and the same row that rides ``_public_view`` to the worker on the
        heartbeat reply. That co-location is the point: central's eviction
        preview and the worker's auto-evict must rank from ONE ledger, so a
        tok/s column kept anywhere else would be a second store and would break
        Parity the moment the two disagreed.

        Split from ``pick_for_model`` because the two facts are known at
        different times: which worker was PICKED is known before the relay, how
        fast it DECODED only after. Same row, same discipline, one writer each.

        FAIL-OPEN AND TOTAL. The serving path is live; this is called from a
        relay-completion seam where an exception would surface as a failed user
        request. A missing ``timings`` block, an unknown worker, an unwritable
        store — all return False and record nothing. Never raises.

        Returns True iff a sample was actually recorded.
        """
        if tok_s is None:
            return False
        try:
            v = float(tok_s)
        except (TypeError, ValueError):
            return False
        if not (v > 0.0) or v != v or v in (float("inf"), float("-inf")):
            return False
        now = _now()
        state: Dict[str, Any] = {}
        try:
            with self._transaction() as workers:
                stored = workers.get(worker_id)
                if stored is None:
                    return False
                row = stored.setdefault("model_call_stats", {}).setdefault(
                    model_key, {"calls": 0})
                if not isinstance(row, dict):
                    return False
                ok = _record_tok_s(row, v)
                if not ok:
                    return False
                # (b) per-(worker,model) rollup + (c) per-worker rollup
                mts = stored.setdefault("model_tok_stats", {})
                mrow = mts.get(model_key)
                if not isinstance(mrow, dict):
                    mrow = mts[model_key] = {}
                _rollup_tok_stats(mrow, v, now)
                ts = stored.get("tok_stats")
                if not isinstance(ts, dict):
                    ts = stored["tok_stats"] = {}
                _rollup_tok_stats(ts, v, now, model_key=model_key)
                # STATE SNAPSHOT — captured INSIDE the transaction, off the
                # authoritative stored record central holds (vram/alloc/
                # co-residents live on the heartbeat, not the relay). Assembled
                # here so the ledger line can later say which configuration gave
                # the best tok/s. Fail-open inside _state_snapshot.
                state = _state_snapshot(stored, model_key, meta)
                # Metrics-card facts (2026-09-10) captured here, WRITTEN after
                # the lock: worker name + the ods section coordinates.
                _mx_worker = stored.get("name") or worker_id
                _mx_am = str((stored.get("model_alloc_modes") or {})
                             .get(model_key) or "")
                _mx_moe = bool((stored.get("moe_effective") or {})
                               .get(model_key))
                _mx_bnb = bool((stored.get("bnb_by_model") or {})
                               .get(model_key))
        except Exception:  # noqa: BLE001 — recording must never fail a request
            logger.debug("record_serve_metrics skipped for %s/%s",
                         worker_id, model_key, exc_info=True)
            return False
        # (a) every query logged — outside the store lock, append-only.
        entry: Dict[str, Any] = {
            "ts": round(now, 3), "worker_id": worker_id, "model_key": model_key,
            "tok_s": round(v, 3),
        }
        for k in ("request_id", "prompt_tokens", "completion_tokens",
                  "elapsed_s", "source", "estimated", "task"):
            if meta.get(k) is not None:
                entry[k] = meta[k]
        if state:
            entry["state"] = state
        _append_toks_log(entry)
        # Registry-DB metrics card (2026-09-10, modeled on the operator's
        # llm_storage/assets/metrics.xlsx_0.ods): the SAME sample lands in the
        # per-(model x quant x alloc x worker) table. Outside the store lock,
        # fail-open, inert unless HUGPY_REGISTRY_DB=pg.
        try:
            from hugpy_engine.model_index import record_metric
            _base = {"gpu-only": "gpu", "ram-only": "ram"}.get(_mx_am, _mx_am)
            if _mx_moe:
                _mode = "moe"
            elif _mx_bnb:
                _mode = f"4bit:{_base}" if _base else "4bit"
            else:
                _mode = _base or ""
            _quant = str(((state.get("model") or {}).get("dtype")) or "")
            record_metric(model_key, str(_mx_worker), quant=_quant,
                          alloc_mode=_mode, tok_per_s=v,
                          task=meta.get("task"))
            # EVERY call, appended (operator: "maintains its record of every
            # call and its metrics") — the card above is this stream's rollup.
            from hugpy_engine.model_index import record_call
            record_call(model_key, str(_mx_worker), quant=_quant,
                        alloc_mode=_mode, tok_per_s=v,
                        prompt_tokens=meta.get("prompt_tokens"),
                        completion_tokens=meta.get("completion_tokens"),
                        elapsed_s=meta.get("elapsed_s"),
                        task=meta.get("task"),
                        request_id=meta.get("request_id"), state=state)
        except Exception:  # noqa: BLE001 — metrics must never fail a serve
            pass
        logger.info("toks: %s/%s %.1f tok/s (%s, n=%s) rid=%s",
                    worker_id, model_key, v, entry.get("source", "?"),
                    mrow.get("n"), entry.get("request_id"))
        return True

    def candidates_for_model(self, model_key: str,
                             pool: Optional[str] = None,
                             task: Optional[str] = None) -> List[Dict[str, Any]]:
        """Ranked ONLINE workers that can serve ``model_key`` — the cap-aware
        relay router's alternatives list (concurrency hardening 2026-07-11).

        Same eligibility + ranking as ``pick_for_model`` — literally the same
        ``_routing_rank`` key (home before wildcard catch, so the relay's
        fallback walk keeps overflow as overflow; then MEASURED-RESIDENT, then
        ALLOCATED, then ⭐ star / GPU / least-recently-picked). Sharing the one
        key is deliberate: a reroute that ranked differently from the pick it is
        falling back from would be a re-decision, not a reroute. WITHOUT the
        ``last_picked`` write, though, and no route.select event: central's
        in-flight gate iterates this to reroute around a worker that is at its
        advertised in-process concurrency cap, and re-stamping every candidate on
        each probe would corrupt the round-robin. Online only — a reroute target
        must be live right now (the stale-heartbeat fallback pick_for_model does
        is for last-resort primary selection, not for spreading concurrent load).
        """
        candidates = self.workers_for_model(model_key, online_only=True, pool=pool, task=task)
        if not candidates:
            return []
        required = required_pkg_version()
        if required:
            matched = [w for w in candidates if w.get("pkg_version") == required]
            if matched:
                candidates = matched

        # k56: the SAME two placement scopes the pick applies — for the same
        # reason the rank key is shared. A reroute that landed off the
        # preference list, or on a box that must evict to take a polite model,
        # would be a re-decision the operator never made.
        prefs, polite, polite_by_worker = placement_policy(model_key)
        if prefs:
            candidates = _prefs_scope(candidates, prefs, model_key)
        if polite or polite_by_worker:
            # Per-candidate, exactly as the pick resolves it: a worker the model
            # is not polite on stays a reroute target under the ordinary rule.
            candidates = [w for w in candidates
                          if not _polite_on(w, polite, polite_by_worker)
                          or _polite_admits(w, model_key)[0]]
        if not candidates:
            return []

        # Star store read ONCE (never per candidate), alias-tolerant — see
        # pick_for_model's _rank for the rationale (ambiguity tie-break only).
        star_map = _star_map()
        wanted_forms = _match_keys(model_key)

        def _starred(w: Dict[str, Any]) -> bool:
            s = star_map.get(w.get("id"))
            return bool(s) and bool(wanted_forms & _match_keys(str(s)))

        def _rank(w: Dict[str, Any]):
            return _routing_rank(w, model_key, wanted_forms, _starred(w),
                                 (_pref_index(w, prefs) or 0) if prefs else 0)

        return sorted(candidates, key=_rank)


worker_store = WorkerStore()


# Module-level convenience wrappers (mirrors the manifest.py / peers.py style of
# exposing plain functions for routes to import).
def register_worker(**kwargs) -> Dict[str, Any]:
    return worker_store.register(**kwargs)


def heartbeat_worker(worker_id: str, **kwargs) -> Optional[Dict[str, Any]]:
    # kwargs: gpus, loaded_models, spill — all optional, passed straight through.
    return worker_store.heartbeat(worker_id, **kwargs)


def remove_worker(worker_id: str) -> bool:
    return worker_store.remove(worker_id)


def set_worker_admission(worker_id: str, state: str) -> Optional[Dict[str, Any]]:
    return worker_store.set_admission(worker_id, state)


def set_worker_pool(worker_id: str, pool: str) -> Optional[Dict[str, Any]]:
    return worker_store.set_pool(worker_id, pool)


def set_worker_limits(worker_id: str, limits) -> Optional[Dict[str, Any]]:
    return worker_store.set_limits(worker_id, limits)


def set_worker_auto_reap(worker_id: str, enabled: bool) -> Optional[Dict[str, Any]]:
    """Opt a worker into auto-reap (slice 8, Part B). Operator-gated route only."""
    return worker_store.set_auto_reap(worker_id, enabled)


def record_worker_auto_reap(worker_id: str, when: float) -> None:
    """Stamp when an auto-reap last fired (cooldown gate + console)."""
    worker_store.record_auto_reap(worker_id, when)


def enroll_required() -> bool:
    """Whether a valid enrollment token is mandatory to register/heartbeat.

    Default OFF (gradual rollout): tokenless workers may still register, but land
    ``pending`` like everyone else. Flip ``HUGPY_WORKER_ENROLL_REQUIRED`` truthy
    once the fleet is re-enrolled to refuse tokenless / revoked workers outright.
    """
    return os.environ.get("HUGPY_WORKER_ENROLL_REQUIRED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def assign_model(worker_id: str, model_key: str,
                 spill: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    return worker_store.assign_model(worker_id, model_key, spill=spill)


def set_moe(worker_id: str, model_key: str, value) -> Optional[Dict[str, Any]]:
    return worker_store.set_moe(worker_id, model_key, value)


def set_bnb(worker_id: str, model_key: str, enabled: bool) -> Optional[Dict[str, Any]]:
    return worker_store.set_bnb(worker_id, model_key, enabled)


def unassign_model(worker_id: str, model_key: str) -> Optional[Dict[str, Any]]:
    worker_store.set_load_report(worker_id, model_key, None)
    return worker_store.unassign_model(worker_id, model_key)


def grant_model(worker_id: str, model_key: str,
                job_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """SYSTEM-authored placement grant (Phase 1 item 2) — see WorkerStore.grant_model."""
    return worker_store.grant_model(worker_id, model_key, job_id=job_id)


def ungrant_model(worker_id: str, model_key: str) -> Optional[Dict[str, Any]]:
    return worker_store.ungrant_model(worker_id, model_key)


def set_load_report(worker_id: str, model_key: str,
                    report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return worker_store.set_load_report(worker_id, model_key, report)


def spill_for(worker_id: str, model_key: str) -> Dict[str, Any]:
    out = dict(worker_store.spill_for(worker_id, model_key))
    # The selected GGUF is part of the seat contract just like placement.  It
    # must ride each inference request to the worker: a worker may already know
    # the model and therefore has no reason to re-fetch central's config row
    # after a per-worker quant pin changes.  Without this, every nominal quant
    # lane reused whichever file happened to be seated first.
    try:
        from hugpy_engine.serve.overrides import get_override, resolve_gguf_for_worker
        worker = worker_store._load().get(worker_id) or {}
        forms = _worker_forms(worker) | {str(worker_id).strip().lower()}
        ov = get_override(model_key) or {}
        selected = resolve_gguf_for_worker(
            ov.get("gguf_file_by_worker"), forms) or ov.get("gguf_file")
        if selected:
            out["gguf_file"] = selected
    except Exception:  # noqa: BLE001 — quant projection is additive
        logger.debug("GGUF pin projection skipped for %s on %s", model_key,
                     worker_id, exc_info=True)
    return out


def derived_default_for(worker_id: str, model_key: str) -> Optional[str]:
    """The feasibility-derived BLANK default mode for (worker, model) from the
    RAW worker record, or None if the worker is unknown. Degrades to 'max-gpu'
    on any miss (via derived_default_mode). Read-only — used to SURFACE the
    derived default the UI distinguishes from a stored one."""
    worker = worker_store._load().get(worker_id)
    if worker is None:
        return None
    return derived_default_mode(worker, model_key)


def derived_allocation_for(worker_id: str, model_key: str) -> Optional[Dict[str, Any]]:
    """The full derived initial allocation ``{"mode","spill","why"}`` for
    (worker, model) from the RAW worker record, or None if the worker is
    unknown. The allocation-shaped companion of ``derived_default_for`` — used
    by surfaces that want to SHOW the derived split (and its ``why``), and by
    any caller that needs the wire encoding rather than the mode name."""
    worker = worker_store._load().get(worker_id)
    if worker is None:
        return None
    return derived_default_allocation(worker, model_key)


def fleet_fit_for_model(model_key: str,
                        workers: Optional[List[Dict[str, Any]]] = None
                        ) -> Dict[str, Any]:
    """Can ``model_key`` land ANYWHERE on the fleet, in ANY mode? (CASE A glue.)

    Resolves each ONLINE worker's measured GPU/RAM totals and the model's
    effective size from central's authoritative sources, then hands the pure
    arithmetic to ``alloc_modes.fleet_fit_verdict``. Returns that function's
    dict — the caller acts on ``blockable`` and nothing else.

    ONLINE-ONLY IS DELIBERATE. An offline box's totals are last-known and its
    return is not in central's gift; counting it as a confident "cannot fit"
    would let a box being rebooted vote a model out of the pool. It also cannot
    vote "fits" — a model is not placeable on a worker that is not there. So an
    offline worker simply does not appear, which (per fleet_fit_verdict) makes
    an all-offline fleet unblockable rather than universally-refusing.

    Degrades to a NON-blockable verdict on ANY error: an exception here must
    never manufacture a block."""
    try:
        from hugpy_engine.alloc_modes import fleet_fit_verdict
        engine = _model_engine(model_key)
        size = _model_size_bytes(model_key)
        rows = workers if workers is not None else list_workers()
        boxes = []
        for w in (rows or []):
            if (w.get("status") or "").lower() != "online":
                continue
            boxes.append({
                "name": w.get("name") or w.get("id"),
                "engine": engine,
                "model_bytes": size,
                "gpu_total_bytes": _worker_gpu_total_bytes(w),
                "ram_total_bytes": _worker_ram_total_bytes(w),
            })
        return fleet_fit_verdict(boxes)
    except Exception as exc:  # noqa: BLE001 — never manufacture a block
        logger.debug("fleet fit verdict failed for %s: %s", model_key, exc)
        return {"fits_somewhere": None, "blockable": False, "fits_on": [],
                "refused_by": [], "unknown": [],
                "why": f"fleet fit could not be evaluated ({exc}) — not blocked"}


def maybe_auto_block(model_key: str,
                     workers: Optional[List[Dict[str, Any]]] = None
                     ) -> Optional[Dict[str, Any]]:
    """CASE A action: auto-block ``model_key`` iff it fits NO online worker in
    ANY mode. Returns the block record when it blocked, else None.

    Composes ``fleet_fit_for_model`` (the arithmetic) with
    ``blocklist.auto_block`` (which enforces operator-unblock stickiness and
    never overwrites an operator-authored block). Best-effort by design — any
    failure is swallowed, because a broken auto-blocker must degrade to today's
    behavior (a late honest load-time refusal), never to a spurious block."""
    try:
        verdict = fleet_fit_for_model(model_key, workers)
        if not verdict.get("blockable"):
            return None
        from hugpy_fleet.central.blocklist import auto_block
        rec = auto_block(model_key, verdict.get("why") or "auto: fits no worker")
        if rec:
            logger.warning("AUTO-BLOCKED %s — %s", model_key, verdict.get("why"))
        return rec
    except Exception:  # noqa: BLE001
        logger.debug("auto-block evaluation failed for %s", model_key,
                     exc_info=True)
        return None


_FEAS_FAILOPEN_SEEN: set = set()


def _warn_feasibility_failopen(worker: Dict[str, Any], model_key: str,
                               size, gpu_total, ram_total) -> None:
    """Self-policing drift signal (operator refinement 2026-07-24): feasibility
    only falls OPEN (all modes selectable) when central couldn't resolve the
    model size or the box totals — which should be a RARE, TRANSIENT state (a
    fresh worker before its first heartbeat; a registry row before enrichment).
    Log it ONCE per (model, worker, missing-set) at WARNING so a transient never
    spams but a PERSISTENT fail-open (real drift the keeper should see) surfaces.
    Only fires when something is actually missing."""
    missing = []
    if not size:
        missing.append("no model size")
    if not gpu_total or ram_total is None:
        missing.append("no gpu/ram totals")
    if not missing:
        return
    wid = worker.get("id") or worker.get("name") or "?"
    key = (wid, model_key, ",".join(missing))
    if key in _FEAS_FAILOPEN_SEEN:
        return
    _FEAS_FAILOPEN_SEEN.add(key)
    logger.warning("feasibility fail-open for %s on %s: %s — all alloc modes "
                   "offered (transient before enrichment is fine; a persistent "
                   "one is a drift signal)",
                   model_key, worker.get("name") or wid, "; ".join(missing))


def feasible_modes_for(worker_id: str, model_key: str) -> Optional[tuple]:
    """The feasible allocation modes for (worker, model) from the RAW record —
    engine + box-totals aware. None if the worker is unknown; degrades to the
    full mode set on any lookup miss (never eliminate on missing data). Used to
    SURFACE the selectable set and to ENFORCE it at /assign."""
    worker = worker_store._load().get(worker_id)
    if worker is None:
        return None
    try:
        from hugpy_engine.alloc_modes import feasible_modes
        size = _model_size_bytes(model_key)
        gpu_total = _worker_gpu_total_bytes(worker)
        ram_total = _worker_ram_total_bytes(worker)
        _warn_feasibility_failopen(worker, model_key, size, gpu_total, ram_total)
        # bnb (2026-07-29): the 4-bit lever re-prices the model, so the FEASIBLE
        # SET must move with it — otherwise the gate refuses a mode the
        # allocator has already planned at the smaller size.
        return feasible_modes(_model_engine(model_key), size, gpu_total, ram_total,
                              moe_split_gpu_bytes=_model_moe_gpu_bytes(model_key),
                              bnb=bnb_enabled(worker, model_key))
    except Exception:  # noqa: BLE001 — a derivation must never break a read/relay
        from hugpy_engine.alloc_modes import ALLOC_MODES
        return ALLOC_MODES


def worker_can_hold(worker: Dict[str, Any], model_key: str) -> Optional[bool]:
    """STATIC feasibility of (worker, model): can this box hold the model AT ALL,
    in ANY mode — i.e. does it fit GPU+RAM combined (the honest ceiling for GGUF
    partial-offload and transformers cpu/disk offload alike)?

    Returns True (fits somewhere on this box), False (confidently cannot — the
    refusal is statically knowable), or None (unsizable model / unmeasured box —
    NO opinion, degrade-not-guess: the caller must NOT eliminate on a None).

    k67: the routing candidate pipeline uses this to stop OFFERING a worker whose
    refusal central already knows — e.g. a 51.8 GB GGUF to computron's 8 GiB card
    (which then refused honestly, repeatedly). It is the pure ``worker_fit_verdict``
    fed from central's authoritative size/totals; ``feasible_modes`` is NOT a
    substitute here because it rates GGUF max-gpu universally feasible (partial
    offload) and so never catches an oversized model that overflows RAM too."""
    try:
        from hugpy_engine.alloc_modes import worker_fit_verdict
        return worker_fit_verdict(
            _model_engine(model_key),
            _model_size_bytes(model_key),
            _worker_gpu_total_bytes(worker),
            _worker_ram_total_bytes(worker),
            # MoE: the split's GPU-resident share is the honest static ceiling
            # (experts stream via mmap) — same term feasible_modes prices.
            moe_split_gpu_bytes=_model_moe_gpu_bytes(model_key))
    except Exception:  # noqa: BLE001 — a static gate must never manufacture a skip
        return None


def feasibility_context(worker_id: str, model_key: str) -> Dict[str, Any]:
    """The raw numbers behind a feasibility decision for (worker, model), for an
    honest 409 reason and the UI. All bytes; None where central can't resolve.
    Empty dict if the worker is unknown."""
    worker = worker_store._load().get(worker_id)
    if worker is None:
        return {}
    return {
        "engine": _model_engine(model_key),
        "model_bytes": _model_size_bytes(model_key),
        "gpu_total_bytes": _worker_gpu_total_bytes(worker),
        "ram_total_bytes": _worker_ram_total_bytes(worker),
        # MoE (2026-07-24): the expert-split GPU need (non-expert + mmproj) a
        # feasibility decision priced GPU-fit with; None for dense models.
        "moe_split_gpu_bytes": _model_moe_gpu_bytes(model_key),
        # bnb (2026-07-29): whether the 4-bit lever was ON for this decision.
        # An honest 409 must say WHICH size it priced — a refusal quoting the
        # fp16 bytes while 4-bit is enabled reads as a bug in the numbers.
        "bnb": bnb_enabled(worker, model_key),
    }


def list_workers() -> List[Dict[str, Any]]:
    return worker_store.all()


def get_worker(worker_id: str) -> Optional[Dict[str, Any]]:
    return worker_store.get(worker_id)


def lookup_worker(name_or_id: str) -> Optional[Dict[str, Any]]:
    """Exact registry lookup for an explicitly named request target.

    No serving-policy filter belongs here: the caller has already selected the
    worker.  The subsequent relay/provisioning attempt supplies the real result.
    """
    want = str(name_or_id or "").strip()
    if not want:
        return None
    return next((w for w in worker_store.all()
                 if want in (w.get("id"), w.get("name"))), None)


def worker_storage_view(worker_id: str) -> Optional[Dict[str, Any]]:
    """Freshly-recomputed storage view + eviction proposal for a worker (from its
    RAW record). The /reap-approve route's central second guard. None if unknown."""
    return worker_store.storage_view(worker_id)


def pick_worker_for_model(model_key: str, pool: Optional[str] = None,
                          task: Optional[str] = None,
                          require_comfy_id_lock: bool = False) -> Optional[Dict[str, Any]]:
    return worker_store.pick_for_model(
        model_key, pool=pool, task=task,
        require_comfy_id_lock=require_comfy_id_lock)


def candidates_for_model(model_key: str, pool: Optional[str] = None,
                         task: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ranked online workers holding ``model_key`` — the relay gate's reroute
    list (see WorkerStore.candidates_for_model). No routing side effects."""
    return worker_store.candidates_for_model(model_key, pool=pool, task=task)


def record_serve_metrics(worker_id: str, model_key: str,
                         tok_s: Optional[float] = None, **meta: Any) -> bool:
    """Module-level binding of ``WorkerStore.record_serve_metrics`` — the seam
    the core relay is handed (web -> core), matching pick_worker_for_model's
    pattern. Fail-open all the way down; see the store method. ``meta`` is the
    per-query record (request_id, prompt/completion tokens, elapsed_s, source,
    estimated) that lands in toks_log.jsonl."""
    return worker_store.record_serve_metrics(worker_id, model_key, tok_s=tok_s,
                                             **meta)


# ---------------------------------------------------------------------------
# LIVE /health PROBE for the cold-load hold (operator incident 2026-07-28).
#
# The hold's progress signal used to come only from the worker RECORD, i.e. from
# heartbeats. That is precisely the wrong source during a cold load: the boxes
# that take a long time to load are the boxes that are busy, and a busy worker
# is the one whose heartbeats starve (the "ae goes deaf" failure class — a
# 503-storm crowds out the beat). So central's picture froze exactly when it
# most needed to be live, the hold saw "no movement", and the 90s stall clock
# killed a load that finished healthy moments later.
#
# The worker's own /health already answers with `provisioning`,
# `provision_progress` and `loaded_models`. Polling it is READ-ONLY and needs no
# worker release, so the hold reads the truth from the source.
#
# Two disciplines make this safe on the serving hot path:
#   * NEVER BLOCK. load_state_for_model is called synchronously from inside the
#     async hold loop; a blocking HTTP GET there would stall the shared event
#     loop for every other request on the box. So the probe runs on a daemon
#     thread and this function only ever reads the CACHE — the first call after
#     a gap returns slightly stale data and triggers a refresh, which is exactly
#     the right trade for a loop that polls every ~2s anyway.
#   * ONE IN FLIGHT PER WORKER, rate-limited. A held call cannot turn into a
#     probe storm against a box that is already struggling.
# ---------------------------------------------------------------------------

_HEALTH_CACHE: Dict[str, Dict[str, Any]] = {}
_HEALTH_INFLIGHT: set = set()
_HEALTH_LOCK = threading.Lock()


def _health_probe_enabled() -> bool:
    """Off switch for the live probe (``HUGPY_COLD_HOLD_HEALTH=off``). Default
    ON: without it the hold is blind through the whole weights-load phase."""
    return (os.environ.get("HUGPY_COLD_HOLD_HEALTH", "").strip().lower()
            not in ("off", "0", "false", "no"))


def _health_probe_interval_s() -> float:
    """Minimum seconds between /health probes of the SAME worker. Default 3s —
    slower than the hold's ~2s poll, so a held call reuses a cached answer more
    often than it fetches one."""
    try:
        v = float((os.environ.get("HUGPY_COLD_HOLD_HEALTH_POLL_S") or "3").strip())
        return v if v > 0 else 3.0
    except (TypeError, ValueError):
        return 3.0


def _health_probe_timeout_s() -> float:
    """Per-probe HTTP timeout. Deliberately short: a probe that hangs tells the
    hold nothing a timely 'no answer' wouldn't."""
    try:
        v = float((os.environ.get("HUGPY_COLD_HOLD_HEALTH_TIMEOUT_S") or "4").strip())
        return v if v > 0 else 4.0
    except (TypeError, ValueError):
        return 4.0


def _fetch_health(worker_id: str, url: str) -> None:
    """Probe body — runs on a daemon thread; stores into the cache. Never raises.

    k59: goes through the sanctioned client, so it inherits the short connect
    budget AND the per-worker breaker — a powered-off box stops being probed
    every few seconds by every held call in the process."""
    data = None
    try:
        from hugpy_fleet.central import worker_http
        resp = worker_http.get({"id": worker_id, "url": url}, "/health",
                               call="probe",
                               read_timeout=_health_probe_timeout_s())
        data = resp.json()
    except Exception:  # noqa: BLE001 — an unreachable worker is data, not a crash
        data = None
    with _HEALTH_LOCK:
        if isinstance(data, dict):
            _HEALTH_CACHE[worker_id] = {"ts": time.time(), "data": data}
        else:
            # Record the ATTEMPT even on failure so the rate limit still applies
            # to an unreachable box (no probe storm against a dead worker).
            prev = _HEALTH_CACHE.get(worker_id) or {}
            _HEALTH_CACHE[worker_id] = {"ts": time.time(),
                                        "data": prev.get("data"),
                                        "stale": True}
        _HEALTH_INFLIGHT.discard(worker_id)


def _live_health(worker: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Cached /health body for ``worker``, kicking a background refresh when the
    cached copy is older than the probe interval. None when disabled, when the
    worker has no url, or before the first probe has landed."""
    if not _health_probe_enabled():
        return None
    wid = worker.get("id") or ""
    url = worker.get("url") or ""
    if not wid or not url:
        return None
    now = time.time()
    with _HEALTH_LOCK:
        entry = _HEALTH_CACHE.get(wid) or {}
        due = (now - float(entry.get("ts") or 0)) >= _health_probe_interval_s()
        if due and wid not in _HEALTH_INFLIGHT:
            _HEALTH_INFLIGHT.add(wid)
            kick = True
        else:
            kick = False
        data = entry.get("data")
    if kick:
        try:
            threading.Thread(target=_fetch_health, args=(wid, url),
                             name=f"health-probe-{wid[:8]}", daemon=True).start()
        except Exception:  # noqa: BLE001 — thread exhaustion must not break a hold
            with _HEALTH_LOCK:
                _HEALTH_INFLIGHT.discard(wid)
    return data if isinstance(data, dict) else None


def _progress_line(model_key: str, worker_name: str,
                   entry: Optional[Dict[str, Any]]) -> tuple:
    """(fraction, human line) for a provision_progress entry — ('8.2 GB of
    11.4 GB transferred'). The worker records ``done_bytes``/``total_bytes``/
    ``frac``; the OLD reader asked for ``progress``/``message``, keys that have
    never existed on that entry, so every held call showed a bare spinner and
    every give-up message said 'last: 503' with no numbers. Bytes are the honest
    unit here: a fraction alone can't tell 'stalled at 12.1 GB' from 'stalled at
    0 B', and that distinction is the whole diagnostic."""
    if not isinstance(entry, dict):
        return None, None
    done = entry.get("done_bytes")
    total = entry.get("total_bytes")
    frac = entry.get("frac")
    try:
        frac = float(frac) if frac is not None else (
            (float(done) / float(total)) if done and total else None)
    except (TypeError, ValueError, ZeroDivisionError):
        frac = None
    if done is None and frac is None:
        return None, None
    if done:
        moved = _fmt_bytes_short(done)
        line = (f"loading {model_key} on {worker_name} — {moved} transferred"
                if not total else
                f"loading {model_key} on {worker_name} — {moved} of "
                f"{_fmt_bytes_short(total)} transferred")
    else:
        line = f"loading {model_key} on {worker_name} — starting transfer"
    return frac, line


def _fmt_bytes_short(n: Any) -> str:
    try:
        n = float(n or 0)
    except (TypeError, ValueError):
        return "0 B"
    for unit, size in (("TB", 2**40), ("GB", 2**30), ("MB", 2**20), ("KB", 2**10)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{int(n)} B"


def load_state_for_model(model_key: str, worker_id: str,
                         since_ts: float = 0.0) -> Optional[Dict[str, Any]]:
    """The cold-load HOLD's view (t36) of ``model_key`` on ``worker_id``.

    Reads the worker's live state — heartbeat record PLUS a read-only /health
    probe (see the block comment above; no worker-side change either way) — and
    returns a compact status the core hold loop (resolvers.remote) consults:

      {"healthy": bool,       # resident/loaded now (ready to serve) == LOADED/SERVING
       "on_disk": bool,        # on the worker's hot drive (models_local) == HOT
       "pulling": bool,        # COLD->HOT download in flight (t_pull)
       "loading": bool,        # HOT->VRAM load in flight (t_load)
       "in_progress": bool,    # union of pulling|loading (any transition)
       "progress": float|None, # download fraction when provisioning
       "message": str|None,    # human progress line, with BYTES
       "error": str|None}      # a FRESH (ts>=since_ts) honest load failure

    ``error`` is only the worker's own last_load_error (load_reports, ok False)
    that is NEWER than ``since_ts`` — a stale error from a prior request never
    fails a fresh hold. It is returned VERBATIM; the core classifies transient vs
    honest so this stays a dumb reader. Returns None on any failure / unknown
    worker (the hold then degrades to a blind bounded retry)."""
    try:
        w = worker_store.get(worker_id)
        if not w:
            return None
        wanted = _match_keys(model_key)
        wname = w.get("name") or worker_id

        def _member(coll) -> Optional[str]:
            for m in (coll or []):
                if m == model_key or (_match_keys(m) & wanted):
                    return m
            return None

        loaded = _member(w.get("loaded_models"))
        # HOT (canonical STATE-MODEL.md): weights on the worker's hot drive,
        # not necessarily in VRAM. disk-truth = models_local (UTIL-08). Lets the
        # cold-hold gate tell a t_load (HOT->VRAM) from a t_pull (download).
        on_disk = bool(_member(w.get("models_local")))
        # Split transitions (canonical STATE-MODEL.md #11): pull (COLD->HOT,
        # downloading/provisioning) vs load (HOT->VRAM). in_progress stays the
        # union so forward-progress callers are unaffected.
        loading_now = bool(_member(w.get("loading")))          # HOT->VRAM  (t_load)
        pulling_now = bool(_member(w.get("provisioning")))     # COLD->HOT  (t_pull)

        progress = None
        message = None
        pp = w.get("provision_progress") or {}
        if isinstance(pp, dict):
            for k, v in pp.items():
                if (k == model_key or (_match_keys(k) & wanted)) and isinstance(v, dict):
                    progress, message = _progress_line(model_key, wname, v)
                    pulling_now = True
                    break

        # LIVE OVERLAY. Anything /health says is newer than the record, and it
        # keeps answering while heartbeats starve — which is the whole reason
        # this exists. Purely ADDITIVE: it can turn "no movement" into movement,
        # never the reverse, so a probe that fails or lags can only leave the
        # hold exactly as blind as it was before, never blinder.
        hb = _live_health(w)
        if isinstance(hb, dict):
            if _member(hb.get("loaded_models")):
                loaded = loaded or model_key
            if _member(hb.get("provisioning")):
                pulling_now = True
            hpp = hb.get("provision_progress")
            if isinstance(hpp, dict):
                for k, v in hpp.items():
                    if (k == model_key or (_match_keys(k) & wanted)) and isinstance(v, dict):
                        p2, m2 = _progress_line(model_key, wname, v)
                        if p2 is not None or m2 is not None:
                            progress, message = p2, m2
                        pulling_now = True
                        break

        error = None
        reports = w.get("load_reports") or {}
        if isinstance(reports, dict):
            for k, v in reports.items():
                if not isinstance(v, dict):
                    continue
                if not (k == model_key or (_match_keys(k) & wanted)):
                    continue
                try:
                    fresh = float(v.get("ts") or 0) >= float(since_ts or 0)
                except (TypeError, ValueError):
                    fresh = False
                if v.get("ok") is False and fresh:
                    error = str(v.get("error") or "load failed")
                break

        # STORAGE REFUSAL is an honest, STANDING verdict, not a load_report: the
        # pull was refused (budget.BudgetRefusal) BEFORE any load attempt, so it
        # never appears in load_reports. agent._refused_snapshot ships it in the
        # heartbeat's `refused` map and prunes it the moment the model lands.
        # Without surfacing it here the cold-hold sees no error and no progress,
        # streams "⏳ loading … on <worker>" until the stall/ceiling clock, then
        # gives up with a dishonest "went silent — check the worker's logs"
        # message — the exact "either it IS or it ISN'T" complaint (operator,
        # 2026-08-31). The reason text carries "won't fit"/"budgetrefusal", which
        # _is_permanent_load_error already classifies PERMANENT, so feeding it
        # into `error` makes the hold fail FAST with the real reason. A standing
        # entry in the CURRENT heartbeat is by definition fresh — no since_ts
        # gate applies (a stale refusal is pruned worker-side once the model is
        # local, so it cannot outlive the condition).
        if not error:
            refused = w.get("refused") or {}
            if isinstance(refused, dict):
                for k, v in refused.items():
                    if not (k == model_key or (_match_keys(k) & wanted)):
                        continue
                    if isinstance(v, dict):
                        error = str(v.get("reason") or "won't fit (storage refused)")
                    break

        return {
            "healthy": bool(loaded),
            "on_disk": on_disk,
            "pulling": pulling_now,
            "loading": loading_now,
            "in_progress": bool(pulling_now or loading_now),
            "progress": progress,
            "message": message,
            "error": error,
        }
    except Exception:  # noqa: BLE001 — advisory only, never break the hold
        return None


def explain_no_worker(model_key: str, pool: Optional[str] = None,
                      task: Optional[str] = None) -> str:
    """Human reason no worker took a request for ``model_key`` — the ``detail`` the
    refused-local error (HUGPY_NO_LOCAL_SERVING) surfaces so a DESIGNATED-but-idle
    model's failure is actionable instead of an opaque "no worker available".

    Walks the workers ASSIGNED to (or holding) the model and names why each was
    excluded from selection by a HARD static gate (admission, engine usability,
    dedicated-pool reservation, env tier, task capability) — the same gates
    ``workers_for_model`` applies. Returns "" when the model has no assigned worker
    at all (the caller's generic message already covers "assign it somewhere"),
    when every assigned worker actually passed the static gates (so the miss was
    transient — a stale beat or momentary cap, not a designation problem), or on
    ANY error: this is advisory and must never raise into a request.
    """
    try:
        # Operator BLOCK is the FIRST, most-specific reason — surfaced even when
        # the model has no assigned worker (unlike the static-gate walk below,
        # which only names DESIGNATED-but-excluded boxes). This is what turns the
        # refused-local error into a distinct "blocked from the serving pool by
        # the operator" line instead of a generic "no worker available".
        try:
            from hugpy_fleet.central.blocklist import block_reason as _br
            _blk = _br(model_key)
        except Exception:  # noqa: BLE001 — advisory; never raise into a request
            _blk = None
        if _blk:
            return _blk
        wanted = _match_keys(model_key)
        want_pool = (pool or "").strip()
        need_tier = env_tier_for_model(model_key)
        wildcards = _wildcard_map()   # read once; guarded (miss -> {})
        reasons: List[str] = []
        for w in worker_store.all():
            serveable = list(w.get("models", [])) + list(w.get("loaded_models", []))
            # Same membership predicate as workers_for_model (alias-tolerant,
            # blocked-sibling-guarded) so this diagnostic never disagrees with
            # the selector it explains. A WILDCARD worker is a potential server
            # for ANY model, so one excluded by a hard gate belongs in the
            # honest answer too — tagged "(wildcard)" so the operator can tell a
            # designation failure from lost all-comers overflow capacity.
            home = _serveable_match(model_key, wanted, serveable)
            wildcard = (not home) and bool(wildcards.get(w.get("id") or ""))
            if not (home or wildcard):
                continue                          # not designated for this model
            name = w.get("name") or w.get("id") or "worker"
            if wildcard:
                name = f"{name} (wildcard)"
            if w.get("admission") != "approved":
                reasons.append(f"{name}: not approved (admission={w.get('admission')!r})")
                continue
            if _engine_unusable(w):
                eng = w.get("engine") or {}
                sr = w.get("slot_incapable_reason")
                err = str(eng.get("error") or "").strip()
                if w.get("slot_capable") is False and sr:
                    why = str(sr)
                elif err:
                    why = f"llama-cpp not loadable: {err}"
                else:
                    why = "inference engine reports installed=False"
                reasons.append(f"{name}: engine unusable ({why[:400]})")
                continue
            if (w.get("pool") or "").strip() != want_pool:
                reasons.append(f"{name}: reserved for pool {w.get('pool')!r} "
                               f"(request pool {want_pool!r})")
                continue
            if _worker_env_tier(w) != need_tier:
                reasons.append(f"{name}: env tier {_worker_env_tier(w)!r} != "
                               f"required {need_tier!r}")
                continue
            if not _task_capable(w, task):
                reasons.append(f"{name}: cannot run task {task!r} "
                               f"(missing optional dependency)")
                continue
            # Passed every HARD static gate — its miss was runtime/transient, not a
            # designation problem; don't manufacture a reason for it.
        if not reasons:
            return ""
        return (f"{model_key} is assigned but no worker could serve it — "
                + "; ".join(reasons[:4])
                + ". Repair the worker (e.g. `hugpy install-engine` / reinstall "
                  "llama-cpp-python) or assign the model to a healthy worker.")
    except Exception:  # noqa: BLE001 — advisory only; never raise into a request
        return ""


def _reserved_vram_bytes(worker_id: str) -> int:
    """VRAM bytes CURRENTLY reserved on a worker by in-flight heavy video runs
    (p6). Lazily imported + fully guarded so placement never depends on the
    reservation layer being importable/healthy — 0 means 'nothing reserved'."""
    try:
        from hugpy_video.intel.reservation.registry import reserved_bytes
        return int(reserved_bytes(worker_id) or 0)
    except Exception:  # noqa: BLE001 — admission-respect must never break placement
        return 0


def fleet_snapshot(evict_aware: bool = False) -> list:
    """The deterministic allocator's view of the fleet, from the live registry.

    Each worker → a Node with summed free VRAM (across its GPUs), free RAM,
    rpc_endpoint, and online flag. This snapshot + a task's Need is all the
    allocator looks at, so the same registry state yields the same placement.
    """
    from hugpy_engine.resolvers.allocator import Node
    nodes = []
    for w in worker_store.all():
        gpus = w.get("gpus") or []
        free_vram = sum(int(g.get("memory_free") or 0) for g in gpus)
        # p6 admission-respect: a card reserved by an in-flight heavy video run is
        # NOT free for another placement. Subtract the run's claimed peak so the
        # allocator never shards/places a model onto VRAM a Wan/Hunyuan render is
        # about to occupy. Best-effort (0 on any error) — a reservation-store
        # hiccup must never wrongly starve LLM placement.
        free_vram = max(0, free_vram - _reserved_vram_bytes(w["id"]))
        if evict_aware:
            # Evict-to-fit premise (2026-09-16): shard planning may reclaim the
            # VRAM held by this worker's COLD resident models — the worker's
            # _vram_evict_to_fit does the actual reclaim at load. Estimate from
            # installed model sizes; an over-estimate only honest-degrades a
            # load, it never mis-evicts (the worker refuses to drop a
            # protected/serving model). Applies to shard planning ONLY. Capped
            # at the node's PHYSICAL total VRAM — free+reclaimable can never
            # exceed what the GPU physically holds.
            _reclaim = sum(int(_model_size_bytes(_mk) or 0)
                           for _mk in (w.get("loaded_models") or []))
            _total = sum(int(g.get("memory_total") or 0) for g in gpus)
            free_vram = min(_total, free_vram + _reclaim) if _total else free_vram + _reclaim
        nodes.append(Node(
            id=w["id"],
            free_vram=free_vram,
            free_ram=int(w.get("free_ram") or 0),
            rpc_endpoint=w.get("rpc_endpoint"),
            can_lead=(w.get("role") != "rpc"),   # rpc nodes are backends, not leads
            online=(w.get("status") == "online"),
            env_tier=_worker_env_tier(w),
        ))
    return nodes


def plan_placement(bytes_needed: int, *, cpu_ok: bool = False, headroom: float = 1.15,
                   env_tier: Optional[str] = None, evict_aware: bool = False):
    """Deterministically place a task needing ``bytes_needed`` on the live fleet.

    Returns the allocator's Placement (whole / shard / cpu / none). For a 'shard'
    result, ``placement.rpc_servers`` + ``placement.tensor_split`` are what the
    lead is handed as a spill override. ``env_tier`` (when set) restricts the
    snapshot to workers serving that runtime-env tier — the allocator stays
    env-agnostic; we filter its input.
    """
    from hugpy_engine.resolvers.allocator import Need, allocate
    nodes = fleet_snapshot(evict_aware=evict_aware)
    if env_tier:
        nodes = [n for n in nodes if n.env_tier == env_tier]
    return allocate(
        Need(bytes_needed=int(bytes_needed), cpu_ok=cpu_ok, headroom=headroom),
        nodes,
    )


def _parse_shard_bytes(raw) -> Optional[int]:
    """Parse a shard-size value: an int (bytes) or a ``"140gb"``/``"42g"`` string."""
    if isinstance(raw, (int, float)):
        return int(raw) if raw > 0 else None
    if not isinstance(raw, str):
        return None
    r = raw.strip().lower()
    mult = 2**30 if r.endswith(("g", "gb")) else 1
    num = r.rstrip("gb").strip()
    try:
        v = int(float(num) * mult)
        return v if v > 0 else None
    except ValueError:
        return None


def _shard_eligible() -> Dict[str, int]:
    """Models the operator allows to shard, with a VRAM byte estimate.

    Two sources, merged (the persisted store wins on conflict):
      1. env ``HUGPY_SHARD_MODELS`` = ``"key:bytes,key2:bytes"`` (``g``/``gb`` ok).
      2. the persisted ``shard_models`` settings namespace — UI-settable via
         ``POST /settings/shard_models/<model_key>`` {"value": <bytes|"40gb">},
         so an operator can opt a model in from the console with NO restart.
    A model in neither never shards — the whole path stays a no-op by default.
    """
    out: Dict[str, int] = {}
    for part in os.environ.get("HUGPY_SHARD_MODELS", "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, _, raw = part.rpartition(":")
        b = _parse_shard_bytes(raw)
        if b:
            out[key.strip()] = b
    try:
        from hugpy_control.settings import settings_store
        for key, val in (settings_store.all("shard_models") or {}).items():
            if val is True or (isinstance(val, str) and val.strip().lower() in ("auto","on","true","1","yes")):
                b = _model_size_bytes(str(key))   # UI toggle sends just a flag -> auto-size
            else:
                b = _parse_shard_bytes(val)
            if b:
                out[str(key).strip()] = b
    except Exception:  # noqa: BLE001 — settings store must never break placement
        logger.debug("shard_models settings read failed", exc_info=True)
    return out


def placement_for_model(model_key: str) -> Optional[Dict[str, Any]]:
    """Allocator-driven shard placement for the remote.py seam.

    Returns ``{"worker": <lead dict>, "spill": {...}}`` only when ``model_key`` is
    shard-eligible AND the allocator decides it must shard across the pool; else
    ``None`` so the caller uses ordinary whole-model routing. The spill carries
    ``rpc_servers`` + a VRAM-proportional ``tensor_split`` + ``n_gpu_layers=-1``.
    """
    elig = _shard_eligible()
    need = elig.get(model_key) or elig.get(str(model_key).split("/")[-1])
    if not need:
        return None
    # MoE models offload idle experts to host RAM (n_cpu_moe) far more cheaply
    # than streaming them to a remote GPU over the LAN, so RPC-sharding MoE is
    # wasteful and usually unused — the model fits the lead alone via CPU-offload.
    # Skip it: a MoE model serves whole (with expert CPU-offload), not sharded.
    # Dense GGUFs have no such escape hatch and shard as normal.
    try:
        if _model_moe_detail(model_key):
            return None
    except Exception:  # noqa: BLE001 — MoE probe must never break placement
        pass
    placement = plan_placement(need, cpu_ok=False,
                               env_tier=env_tier_for_model(model_key),
                               evict_aware=True)
    if placement.kind != "shard":
        return None
    lead = get_worker(placement.lead_id)
    if not lead or not lead.get("url"):
        return None
    return {
        "worker": lead,
        "spill": {
            "rpc_servers": ",".join(placement.rpc_servers),
            "tensor_split": list(placement.tensor_split),
            "n_gpu_layers": -1,
        },
    }


# Provider registration with the engine (placement + legacy resolver seams)
# moved to ``hugpy_fleet.central.placement.install()``; the composition root
# (hugpy_server) calls it at startup. Importing this module has no side effects.
