"""Precision model->PID registry — the authoritative "worker-module -> pid log".

A process-local, thread-safe registry that tracks, per active model, WHICH OS
process holds it, HOW it's hosted, and — after a reconcile against nvidia-smi
ground truth — how much VRAM it occupies. This is the DATA/telemetry layer:

  * a separate ``evict <model_key>`` action reads it to find the exact PID to
    signal (built concurrently, not here), and
  * central's worker row displays the ``snapshot_for_heartbeat()`` log.

Design constraints honored here:

  * SELF-CONTAINED. This module imports nothing from ``agent.py`` (whose import
    pulls torch and the whole runner stack). It defines its own ``_MIB`` and its
    own ``/proc`` probe. The caller passes the nvidia-smi maps into
    ``reconcile`` — this module never shells out to nvidia-smi.

  * DEGRADE-TO-EMPTY. No records / empty inputs / no ``/proc`` (non-Linux) all
    yield empty attributions and an empty snapshot — never an exception. Same
    convention as ``agent._gpu_process_vram`` returning ``{}`` with no GPU.

  * RECYCLED-PID GUARD (the "precision" the operator asked for). A PID number is
    reused by the OS after a process exits. Before we trust a stored PID we
    re-check its IDENTITY: the Linux process START TIME (field 22 of
    ``/proc/<pid>/stat``, in clock ticks since boot) is a per-process-lifetime
    constant — a new process reusing the number has a DIFFERENT start time. So
    "same PID number, different start time" => the model's process is GONE and
    the number now belongs to a stranger => ``verify`` returns ``None``. The
    command line (``/proc/<pid>/cmdline``) is a secondary anchor used only when
    the start time couldn't be read at record time.

  * The ``/proc`` probe is INJECTABLE (constructor arg / ``set_proc_info_probe``)
    so every path is unit-testable with fake processes — no real subprocess or
    GPU needed.

Host modes (``host_mode``):
  * ``subprocess``  — a slot child (llama-server): PID known at spawn, its VRAM
                      is that PID's nvidia-smi ``mib``.
  * ``in_process``  — a torch model sharing the worker python PID: its VRAM is
                      the per-model torch split (nvidia-smi can't split the lump).
  * ``comfy``       — the external, adopted ComfyUI process: VRAM by process name.
  * ``external``    — an adopted foreign process the worker chose to track by PID.
  * ``cuda_context``— the worker's OWN process(es) (the agent + idle slot children
                      sharing its venv): infrastructure CUDA context, NOT a called
                      model. Labelled so it stops reading as an anonymous squatter.

CALL-TIME ATTRIBUTION (2026-07-14). A foreign GPU service the WORKER drives —
ComfyUI is the live case — is anonymous in nvidia-smi (its process is not a slot
child and holds no torch weights we can split), so before this it fell into
``unattributed`` even though the CALL that produced it knew the model. The
``record_foreign_call``/``end_foreign_call`` table records, at dispatch time, the
``model_key`` (+ ``job_id``) the worker submitted to that service; ``reconcile``
then stamps it onto the recognized service PID, moving it out of ``unattributed``
into an attributed ``comfy`` row. No active call -> the PID is still recognized as
ComfyUI (labelled idle/unknown-model), never dumped as an anonymous squatter. A
truly foreign, call-less process (a real rogue) still surfaces as ``unattributed``
— that is the wanted signal, preserved.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Dict, List, Optional

# Bytes per MiB — defined locally so this module stays dependency-free (importing
# it must never drag in torch / the runner stack via agent.py).
_MIB = 1024 * 1024

# How long an unclosed foreign-call record stays trusted for attribution. The
# best-effort ``end_foreign_call`` at the submit site's completion is the primary
# clear; this TTL is the SAFETY NET so a call whose end was missed (a crashed
# generation, a killed worker request) can never pin a stale model onto a comfy
# PID forever. Default is comfortably above a comfy render's worst case; override
# via env for slow boxes / tests.
try:
    _FOREIGN_CALL_TTL_S = float(os.environ.get("HUGPY_FOREIGN_CALL_TTL_S", "1800") or "1800")
except (TypeError, ValueError):
    _FOREIGN_CALL_TTL_S = 1800.0

# nvidia-smi ``process_name`` substrings that mark a recognized foreign service.
# ComfyUI's process_name is the venv python path (…/ComfyUI/venv/bin/python), so a
# case-insensitive "comfyui" substring is the same anchor _comfy_process_vram uses.
_COMFY_NAME_MARKER = "comfyui"

# Field index of ``starttime`` in /proc/<pid>/stat AFTER the "pid (comm)" prefix
# is split off. The stat line is: `pid (comm) state ppid ... starttime ...`;
# starttime is field 22 (1-indexed). Dropping fields 1-2 (pid, comm) leaves
# `state` at index 0, so starttime (field 22) sits at index 22 - 3 = 19.
_STAT_STARTTIME_IDX = 19

ProcInfo = Dict[str, object]        # {"starttime": int|None, "cmdline": str, "name": str}
ProcInfoProbe = Callable[[int], Optional[ProcInfo]]
HostMode = str                      # "subprocess"|"in_process"|"comfy"|"external"|"cuda_context"


def _default_proc_info(pid: int) -> Optional[ProcInfo]:
    """Read ``{starttime, cmdline, name}`` for ``pid`` from ``/proc``, or ``None``
    if the PID is gone / unreadable. Non-Linux (no ``/proc``) -> ``None`` for
    every PID, which makes ``verify`` degrade to "can't confirm -> don't trust".

    ``starttime`` is the recycled-PID anchor (see module docstring). ``comm`` and
    the full cmdline are captured as secondary identity hints.
    """
    if pid is None:
        return None
    try:
        with open("/proc/%d/stat" % int(pid), "rb") as fh:
            data = fh.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError):
        return None
    # comm (field 2) is wrapped in parens and may itself contain spaces/parens,
    # so slice on the LAST ')' rather than naively splitting on whitespace.
    rparen = data.rfind(b")")
    lparen = data.find(b"(")
    starttime: Optional[int] = None
    name = ""
    if 0 <= lparen < rparen:
        name = data[lparen + 1:rparen].decode("utf-8", "replace")
        rest = data[rparen + 2:].split()
        if len(rest) > _STAT_STARTTIME_IDX:
            try:
                starttime = int(rest[_STAT_STARTTIME_IDX])
            except (ValueError, IndexError):
                starttime = None
    cmdline = ""
    try:
        with open("/proc/%d/cmdline" % int(pid), "rb") as fh:
            cmdline = fh.read().replace(b"\x00", b" ").strip().decode("utf-8", "replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError):
        cmdline = ""
    return {"starttime": starttime, "cmdline": cmdline, "name": name}


class PidRegistry:
    """Thread-safe model_key -> process record store with a recycled-PID guard.

    A record is ``{model_key, pid, host_mode, launched_at, start_tick,
    cmdline_hint, last_vram_bytes, alive}``. ``start_tick`` is the process
    start time captured at record time — the identity anchor.
    """

    def __init__(self, proc_info: Optional[ProcInfoProbe] = None) -> None:
        self._lock = threading.RLock()
        self._records: Dict[str, dict] = {}
        self._last_unattributed: List[dict] = []
        # Recognized-foreign attribution rows from the last reconcile (worker's own
        # CUDA-context lumps + call-attributed / idle ComfyUI). Rebuilt every beat
        # from fresh nvidia-smi + the live foreign-call table, so nothing here
        # persists stale across beats. Surfaced in snapshot_for_heartbeat.models.
        self._last_foreign_rows: List[dict] = []
        # Active foreign-call table (service -> [{model_key, job_id, at}, …]). The
        # worker records a call here at the moment it DISPATCHES a job to a foreign
        # GPU service (ComfyUI), so reconcile can attribute that service's PID to the
        # model the call was FOR. A list (not a scalar) tolerates overlapping calls;
        # attribution picks the most-recent still-live entry.
        self._foreign_calls: Dict[str, List[dict]] = {}
        self._proc_info: ProcInfoProbe = proc_info or _default_proc_info

    # -- probe injection (tests) ------------------------------------------------
    def set_proc_info_probe(self, probe: ProcInfoProbe) -> None:
        """Swap the ``/proc`` probe (tests inject fake processes)."""
        with self._lock:
            self._proc_info = probe

    # -- lifecycle --------------------------------------------------------------
    def record_launch(self, model_key: str, pid: Optional[int],
                      host_mode: HostMode, cmdline_hint: Optional[str] = None) -> dict:
        """Record (or refresh) that ``model_key`` is hosted by ``pid``.

        Captures the process start time NOW as the recycled-PID anchor. Idempotent
        for the heartbeat-driven population path: re-recording the SAME pid with
        the SAME start time preserves ``launched_at`` (so re-observing an already
        known child every beat doesn't reset its clock); a changed pid/start time
        replaces the record (a genuinely new launch, e.g. a reloaded slot).
        """
        info = self._proc_info(pid) if pid is not None else None
        start_tick = info.get("starttime") if info else None
        cmd = cmdline_hint or (info.get("cmdline") if info else None) or None
        with self._lock:
            prev = self._records.get(model_key)
            if (prev is not None and prev.get("pid") == pid
                    and prev.get("start_tick") == start_tick
                    and start_tick is not None):
                # Same live process re-observed — refresh nothing that would
                # perturb identity/age; just keep it.
                prev["cmdline_hint"] = prev.get("cmdline_hint") or cmd
                return dict(prev)
            rec = {
                "model_key": model_key,
                "pid": pid,
                "host_mode": host_mode,
                "launched_at": time.time(),
                "start_tick": start_tick,
                "cmdline_hint": cmd,
                "last_vram_bytes": None,
                "alive": pid is not None and (info is not None),
            }
            self._records[model_key] = rec
            return dict(rec)

    def forget(self, model_key: str) -> bool:
        """Drop ``model_key``'s record. Returns True if one was present."""
        with self._lock:
            return self._records.pop(model_key, None) is not None

    def sweep_dead(self) -> List[str]:
        """Drop every record whose process is gone OR whose PID was recycled.
        Returns the list of forgotten model_keys."""
        dropped: List[str] = []
        with self._lock:
            for mk in list(self._records.keys()):
                if self._verify_locked(mk) is None:
                    self._records.pop(mk, None)
                    dropped.append(mk)
        return dropped

    # -- call-time foreign-service attribution ----------------------------------
    def record_foreign_call(self, service: str, model_key: Optional[str] = None,
                            job_id: Optional[str] = None) -> None:
        """Record that this worker DISPATCHED a job to a foreign GPU ``service``
        (e.g. ``"comfy"``) FOR ``model_key``. Called at the submit site so the PID
        that service is running under can be attributed to the called model at the
        next ``reconcile``. Best-effort by contract: it never raises for a caller
        (the submit path wraps it anyway), so a telemetry hiccup can't break a
        generation. Idempotent-ish: overlapping calls stack; ``end_foreign_call``
        (or the TTL) unwinds them."""
        if not service:
            return
        with self._lock:
            self._foreign_calls.setdefault(service, []).append(
                {"model_key": model_key, "job_id": job_id, "at": time.time()})

    def end_foreign_call(self, service: str, job_id: Optional[str] = None,
                         model_key: Optional[str] = None) -> None:
        """Clear a foreign-call record on completion. Removes the entry matching
        ``job_id`` (preferred) or ``model_key``; failing an exact match it drops the
        oldest entry for the service so a mismatched close still unwinds one call
        rather than leaking. A no-op for an unknown service. The TTL in reconcile is
        the backstop if this is never reached."""
        if not service:
            return
        with self._lock:
            lst = self._foreign_calls.get(service)
            if not lst:
                return
            idx = None
            for i, c in enumerate(lst):
                if job_id is not None and c.get("job_id") == job_id:
                    idx = i
                    break
                if (job_id is None and model_key is not None
                        and c.get("model_key") == model_key):
                    idx = i
                    break
            lst.pop(idx if idx is not None else 0)
            if not lst:
                self._foreign_calls.pop(service, None)

    def active_foreign_call(self, service: str,
                            ttl: Optional[float] = None) -> Optional[dict]:
        """The most-recent still-live foreign call for ``service``, or ``None``.

        The public read of the call table ``reconcile`` uses for attribution.
        The comfy idle watchdog (k54) asks THIS before freeing comfy's VRAM: a
        registered call means this worker has a generation in flight against
        that service, which is an absolute veto on reclaiming its memory. Same
        leaked-record TTL as attribution — one definition of "a live call", so
        the watchdog and the heartbeat can never disagree about it."""
        with self._lock:
            return self._active_foreign_call_locked(
                service, ttl if ttl is not None else _FOREIGN_CALL_TTL_S)

    def _active_foreign_call_locked(self, service: str, ttl: float) -> Optional[dict]:
        """The most-recent still-live foreign call for ``service`` (caller holds the
        lock). Entries older than ``ttl`` are treated as leaked and ignored (the
        recycled-PID guard's spirit applied to call records)."""
        lst = self._foreign_calls.get(service) or []
        now = time.time()
        live = [c for c in lst if (now - float(c.get("at") or 0.0)) <= ttl]
        return live[-1] if live else None

    # -- identity ---------------------------------------------------------------
    def _verify_locked(self, model_key: str) -> Optional[int]:
        """``verify`` core; caller holds the lock."""
        rec = self._records.get(model_key)
        if rec is None:
            return None
        pid = rec.get("pid")
        if pid is None:
            return None
        info = self._proc_info(pid)
        if info is None:
            rec["alive"] = False        # process gone
            return None
        # RECYCLED-PID GUARD: start time is a per-lifetime constant. A mismatch
        # means the number was reused by an unrelated process => not ours.
        anchor = rec.get("start_tick")
        cur = info.get("starttime")
        if anchor is not None and cur is not None:
            if int(cur) != int(anchor):
                rec["alive"] = False
                return None
        elif anchor is None:
            # Start time unavailable at record time -> fall back to the cmdline
            # anchor. If we can't corroborate identity at all, don't trust it.
            hint = rec.get("cmdline_hint")
            cmd = (info.get("cmdline") or "") if info else ""
            if not hint or hint not in cmd:
                rec["alive"] = False
                return None
        rec["alive"] = True
        return pid

    def verify(self, model_key: str) -> Optional[int]:
        """Return ``model_key``'s PID ONLY if it's still alive AND still the same
        process we recorded (recycled-PID guard). Otherwise ``None``."""
        with self._lock:
            return self._verify_locked(model_key)

    # -- reconcile against nvidia-smi ground truth ------------------------------
    def reconcile(self, gpu_procs: Optional[dict],
                  inprocess_bytes: Optional[dict],
                  comfy_bytes: Optional[int],
                  own_pids: Optional[set] = None,
                  self_venv_marker: Optional[str] = None,
                  foreign_call_ttl: Optional[float] = None) -> dict:
        """Join the registry against nvidia-smi ground truth. PURE of its inputs
        (no ``/proc``, no nvidia-smi call) so it's directly unit-testable.

        Inputs mirror ``agent.py``:
          * ``gpu_procs``       -> ``{pid: {"name", "mib"}}`` (``_gpu_process_vram``)
          * ``inprocess_bytes`` -> ``{model_key: {"vram_bytes", "device"}}``
                                   (``_inprocess_gpu_bytes``)
          * ``comfy_bytes``     -> ``int`` bytes | ``None`` (``_comfy_process_vram``)
          * ``own_pids``        -> the worker's OWN pids (``{os.getpid(), …}``) — the
                                   agent process + any slot supervisor it spawned.
          * ``self_venv_marker``-> a substring of the worker's own venv path; any
                                   unexplained GPU proc whose ``process_name``
                                   contains it is one of the worker's own pids (an
                                   idle slot child or the agent), tagged
                                   ``cuda_context`` — infra, not a called model.
          * ``foreign_call_ttl``-> override the module TTL for the call table.

        Attribution rules (record-driven, unchanged):
          * ``subprocess``/``external`` -> the record's PID's ``mib`` (bytes);
            an alive child that isn't a GPU compute app attributes 0 (CPU).
          * ``in_process`` -> the per-model torch split; the shared worker-python
            PID lump is marked explained so it isn't flagged as a squatter.
          * ``comfy`` (record) -> ``comfy_bytes``; comfyui-named GPU procs explained.

        SECOND PASS (recognized-foreign, 2026-07-14): every still-unexplained GPU
        proc is tested against, in order —
          1. the worker's OWN pids / venv -> a ``cuda_context`` row (agent CUDA
             context / idle slot), NOT a squatter;
          2. ComfyUI (name marker) -> a ``comfy`` row stamped with the active
             foreign call's ``model_key`` (+ ``job_id``) if one is live, else
             labelled "ComfyUI (idle/unknown model)" — recognized, never anonymous.
        Anything that survives BOTH passes is a genuine foreign / rogue squatter and
        stays in ``unattributed`` — the wanted signal, preserved.

        Returns ``{"attributed": {model_key: vram_bytes}, "foreign": [row…],
        "unattributed": [{pid, name, mib}]}`` and stores ``last_vram_bytes`` per
        record + the foreign rows + unattributed for the next snapshot.
        """
        gpu_procs = gpu_procs or {}
        inprocess_bytes = inprocess_bytes or {}
        own = set(own_pids or ())
        marker = (self_venv_marker or "").strip() or None
        ttl = foreign_call_ttl if foreign_call_ttl is not None else _FOREIGN_CALL_TTL_S
        attributed: Dict[str, int] = {}
        explained: set = set()
        foreign_rows: List[dict] = []
        with self._lock:
            # MEASURED-TRUTH in-process sizing (E/M ruling 2026-07-31): the SIZE
            # of an in-process model comes from nvidia-smi's per-PID mib, NOT the
            # torch weight ESTIMATE (which omitted the CUDA context / KV / allocator
            # hold — pid 94035 measured 4074 MiB but was attributed 1800, hiding
            # ~2.3 GiB). Several in-process models can share the ONE worker-python
            # PID, and nvidia-smi cannot split that lump per model, so we split the
            # measured lump proportionally to each model's torch estimate (even
            # split when no estimate), giving the LAST peer the remainder so the
            # per-PID rows sum BYTE-FOR-BYTE to that PID's mib. The estimate is
            # preserved as ``last_vram_bytes_planned`` — the PLANNED figure the
            # panel shows BESIDE the measurement, never in place of it.
            _inproc_by_pid: Dict[int, List[dict]] = {}
            for _rec in self._records.values():
                if _rec.get("host_mode") == "in_process" and _rec.get("pid") is not None:
                    _inproc_by_pid.setdefault(_rec["pid"], []).append(_rec)
            _inproc_attr: Dict[str, dict] = {}   # mk -> {"vb", "planned"}
            for _p, _peers in _inproc_by_pid.items():
                _est = {r["model_key"]: int(
                    (inprocess_bytes.get(r["model_key"]) or {}).get("vram_bytes") or 0)
                    for r in _peers}
                if _p in gpu_procs:
                    _lump = int(gpu_procs[_p].get("mib") or 0) * _MIB
                    _tot = sum(_est.values())
                    _keys = [r["model_key"] for r in _peers]
                    _assigned = 0
                    for _i, _mk in enumerate(_keys):
                        if _i == len(_keys) - 1:
                            _share = _lump - _assigned        # remainder → exact sum
                        elif _tot > 0:
                            _share = _lump * _est[_mk] // _tot
                        else:
                            _share = _lump // len(_keys)
                        _assigned += _share
                        _inproc_attr[_mk] = {"vb": _share, "planned": _est[_mk]}
                else:
                    # PID isn't a GPU compute app (no CUDA context this beat) — no
                    # measurement to split; the estimate is the only signal, and
                    # these bytes are NOT in nvidia-smi's total so no double count.
                    for _mk, _e in _est.items():
                        _inproc_attr[_mk] = {"vb": _e, "planned": _e}

            for rec in self._records.values():
                mk = rec["model_key"]
                mode = rec.get("host_mode")
                pid = rec.get("pid")
                if mode == "in_process":
                    a = _inproc_attr.get(mk) or {}
                    vb = int(a.get("vb") or 0)
                    rec["last_vram_bytes_planned"] = a.get("planned")
                    if pid is not None and pid in gpu_procs:
                        explained.add(pid)      # the shared worker-python lump
                elif mode == "comfy":
                    vb = int(comfy_bytes or 0)
                    for p, meta in gpu_procs.items():
                        if _COMFY_NAME_MARKER in (str(meta.get("name") or "")).lower():
                            explained.add(p)
                else:                           # subprocess / external
                    vb = 0
                    if pid is not None and pid in gpu_procs:
                        vb = int(gpu_procs[pid].get("mib") or 0) * _MIB
                        explained.add(pid)
                    elif pid is not None:
                        explained.add(pid)      # alive child, CPU-resident
                attributed[mk] = vb
                rec["last_vram_bytes"] = vb

            # -- SECOND PASS: recognized-foreign attribution of the leftovers ------
            for p, meta in gpu_procs.items():
                if p in explained:
                    continue
                name = str(meta.get("name") or "")
                mib = int(meta.get("mib") or 0)
                # (1) the worker's OWN infra CUDA context (agent + idle slot child).
                if p in own or (marker is not None and marker in name):
                    foreign_rows.append({
                        "model_key": None, "pid": p, "host_mode": "cuda_context",
                        "vram_bytes": mib * _MIB, "alive": True,
                        "service": "worker", "label": "agent CUDA context",
                        "name": name})
                    explained.add(p)
                    continue
                # (2) ComfyUI — call-time attribution or recognized-idle.
                # DISPLAY HONESTY (p27 companion, operator-sanctioned shape):
                # comfy is ONE process-class row — "comfy — X GB" — never fake
                # per-model rows. ``display_label``/``is_process_row`` are the
                # UI keys for that: is_process_row says "this is a measured
                # PROCESS occupancy row, not a per-model resident"; any
                # ``model_key`` on it is the call-time claimed model, a LABEL
                # for the active job only (attribution, not residency).
                # Additive-only fields on a worker->central free-dict payload
                # (heartbeat ``pid_registry``), so old central/UI ignore them.
                if _COMFY_NAME_MARKER in name.lower():
                    call = self._active_foreign_call_locked("comfy", ttl)
                    row = {
                        "pid": p, "host_mode": "comfy", "vram_bytes": mib * _MIB,
                        "alive": True, "service": "comfy", "name": name,
                        "display_label": "comfy", "is_process_row": True}
                    if call is not None:
                        row["model_key"] = call.get("model_key")
                        row["job_id"] = call.get("job_id")
                        row["label"] = None
                    else:
                        row["model_key"] = None
                        row["job_id"] = None
                        row["label"] = "ComfyUI (idle/unknown model)"
                    foreign_rows.append(row)
                    explained.add(p)
                    continue

            unattributed = [
                {"pid": p, "name": meta.get("name"), "mib": meta.get("mib")}
                for p, meta in gpu_procs.items() if p not in explained
            ]
            self._last_unattributed = unattributed
            self._last_foreign_rows = foreign_rows
        return {"attributed": attributed, "foreign": foreign_rows,
                "unattributed": unattributed}

    # -- heartbeat telemetry ----------------------------------------------------
    def snapshot_for_heartbeat(self) -> dict:
        """The "worker-module-pid log" central displays.

        Returns ``{"models": [{model_key, pid, host_mode, vram_bytes, alive, …}],
        "unattributed": [{pid, name, mib}]}``. ``alive`` is a LIVE recycled-PID-
        guarded ``verify`` (not the last-cached flag) for record rows; ``vram_bytes``
        is the most recent ``reconcile`` attribution. Empty registry -> both lists
        empty (degrade-safe).

        ``models`` carries TWO row kinds: the recycled-PID-guarded per-record rows
        (subprocess / in_process) AND the recognized-foreign rows from the last
        reconcile (``cuda_context`` for the worker's own infra pids, ``comfy`` for a
        call-attributed or idle ComfyUI). Foreign rows additionally carry
        ``service``/``label`` (and ``job_id`` when a call attributed them) and may
        have ``model_key`` None (a cuda_context lump / an idle ComfyUI has no served
        model) — a consumer keys those on ``pid``, not ``model_key``.

        NB: the contract sketch said "-> list"; a dict carrying BOTH the per-model
        log and the unattributed squatters is strictly better for the single
        heartbeat field (central needs both), so this returns one dict.
        """
        with self._lock:
            models = []
            for mk in list(self._records.keys()):
                rec = self._records[mk]
                alive = self._verify_locked(mk) is not None
                row = {
                    "model_key": mk,
                    "pid": rec.get("pid"),
                    "host_mode": rec.get("host_mode"),
                    "vram_bytes": rec.get("last_vram_bytes"),
                    "alive": alive,
                }
                # PLANNED-beside-MEASURED (E/M): an in-process row's torch weight
                # estimate rides alongside its measured bytes so the panel can show
                # the disagreement (measured = weights + CUDA context/KV) rather
                # than blend it. Omit-when-unset — subprocess/comfy rows never carry
                # a planned estimate, and an old consumer just ignores the key.
                planned = rec.get("last_vram_bytes_planned")
                if planned is not None:
                    row["vram_bytes_planned"] = planned
                models.append(row)
            # Recognized-foreign rows (already alive-by-construction: they were in
            # this beat's nvidia-smi output). Append as extra model rows so they read
            # as attributed, not anonymous.
            models.extend(dict(r) for r in self._last_foreign_rows)
            return {"models": models, "unattributed": list(self._last_unattributed)}


# ── module-level default registry + free-function facade ─────────────────────
# The worker holds ONE registry; these delegate to it so callers use
# ``pid_registry.record_launch(...)`` etc. Tests that want isolation construct
# their own ``PidRegistry(proc_info=fake)``.
_REGISTRY = PidRegistry()


def registry() -> PidRegistry:
    """The process-wide default registry."""
    return _REGISTRY


def set_proc_info_probe(probe: ProcInfoProbe) -> None:
    _REGISTRY.set_proc_info_probe(probe)


def record_launch(model_key: str, pid: Optional[int], host_mode: HostMode,
                  cmdline_hint: Optional[str] = None) -> dict:
    return _REGISTRY.record_launch(model_key, pid, host_mode, cmdline_hint)


def forget(model_key: str) -> bool:
    return _REGISTRY.forget(model_key)


def sweep_dead() -> List[str]:
    return _REGISTRY.sweep_dead()


def verify(model_key: str) -> Optional[int]:
    return _REGISTRY.verify(model_key)


def reconcile(gpu_procs: Optional[dict], inprocess_bytes: Optional[dict],
              comfy_bytes: Optional[int], own_pids: Optional[set] = None,
              self_venv_marker: Optional[str] = None,
              foreign_call_ttl: Optional[float] = None) -> dict:
    return _REGISTRY.reconcile(gpu_procs, inprocess_bytes, comfy_bytes,
                               own_pids=own_pids, self_venv_marker=self_venv_marker,
                               foreign_call_ttl=foreign_call_ttl)


def record_foreign_call(service: str, model_key: Optional[str] = None,
                        job_id: Optional[str] = None) -> None:
    return _REGISTRY.record_foreign_call(service, model_key, job_id)


def end_foreign_call(service: str, job_id: Optional[str] = None,
                     model_key: Optional[str] = None) -> None:
    return _REGISTRY.end_foreign_call(service, job_id=job_id, model_key=model_key)


def active_foreign_call(service: str, ttl: Optional[float] = None) -> Optional[dict]:
    return _REGISTRY.active_foreign_call(service, ttl=ttl)


def snapshot_for_heartbeat() -> dict:
    return _REGISTRY.snapshot_for_heartbeat()
