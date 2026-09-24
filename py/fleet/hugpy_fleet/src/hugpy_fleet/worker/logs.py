"""Worker LOG READOUT — ``GET /logs`` (operator ask 2026-09-23: "just have a log
readout of each worker ... make it available in the ui ... thats where errors
are found").

Source of truth is the worker's OWN journal (``journalctl --user -u
hugpy-worker.service -o json``): that covers the agent AND its slot /
llama-server children, which inherit the unit's stdout/stderr. When journalctl
is missing or returns nothing (standalone run, container, no user journal) the
endpoint falls back to an in-process ring buffer handler installed at agent
start (last ``RING_CAPACITY`` records of this process only).

Read-only, bounded (``MAX_LINES``), never raises into the request: a broken
journal degrades to the ring buffer, never to a 500.
"""
from __future__ import annotations

import collections
import datetime as _dt
import json
import logging
import os
import re
import shutil
import subprocess
import threading
from typing import Iterable, List, Optional

RING_CAPACITY = 5000
MAX_LINES = 2000
DEFAULT_LINES = 200
UNIT = os.environ.get("HUGPY_WORKER_UNIT", "hugpy-worker.service")

# errors=1 keeps WARNING+ and anything that LOOKS like a failure, plus the
# continuation lines of a Python traceback (so the event is readable whole).
_ERR_RX = re.compile(r"Traceback|Error|Exception|failed|FAILED|refused", re.I)
_TB_START = "Traceback (most recent call last)"
_ERR_LEVELS = {"WARNING", "ERROR", "CRITICAL"}

# Python logging header ("2026-09-23 13:12:39,875 ERROR name: msg") and the
# llama.cpp log prefix ("0.06.636.595 E srv ..."): the journal PRIORITY of a
# child's stderr is always 3/6, so the level is read from the text when it can be.
_PY_LEVEL_RX = re.compile(r"^\S+ \S+ (DEBUG|INFO|WARNING|ERROR|CRITICAL) ")
_LLAMA_LEVEL_RX = re.compile(r"^\d+\.\d+\.\d+\.\d+ ([DIWE]) ")
_LLAMA_LEVELS = {"D": "DEBUG", "I": "INFO", "W": "WARNING", "E": "ERROR"}
_PRIO_LEVELS = {0: "CRITICAL", 1: "CRITICAL", 2: "CRITICAL", 3: "ERROR",
                4: "WARNING", 5: "INFO", 6: "INFO", 7: "DEBUG"}


def _iso(epoch: float) -> str:
    return _dt.datetime.fromtimestamp(epoch, tz=_dt.timezone.utc).isoformat(
        timespec="microseconds")


def parse_since(value) -> Optional[float]:
    """ISO-8601 (any offset; naive = UTC) or epoch seconds -> epoch float."""
    if value in (None, ""):
        return None
    s = str(value).strip()
    try:
        return float(s)
    except ValueError:
        pass
    try:
        d = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=_dt.timezone.utc)
    return d.timestamp()


def level_of(text: str, default: str = "INFO") -> str:
    m = _PY_LEVEL_RX.match(text or "")
    if m:
        return m.group(1)
    m = _LLAMA_LEVEL_RX.match(text or "")
    if m:
        return _LLAMA_LEVELS[m.group(1)]
    return default


# ── ring buffer fallback ────────────────────────────────────────────────────
class RingBufferHandler(logging.Handler):
    """Keeps the last ``capacity`` formatted records of this process."""

    def __init__(self, capacity: int = RING_CAPACITY):
        super().__init__(level=logging.DEBUG)
        self.records = collections.deque(maxlen=capacity)
        self._rlock = threading.Lock()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
            with self._rlock:
                for i, part in enumerate(text.splitlines() or [""]):
                    self.records.append({
                        "ts": _iso(record.created), "_epoch": record.created,
                        "level": record.levelname if i == 0 else record.levelname,
                        "unit": record.name, "text": part, "pid": os.getpid()})
        except Exception:  # noqa: BLE001 — a log handler must never raise
            pass

    def snapshot(self) -> list:
        with self._rlock:
            return list(self.records)


_RING: Optional[RingBufferHandler] = None


def install_ring_buffer(capacity: int = RING_CAPACITY) -> RingBufferHandler:
    """Attach the ring buffer to the root logger once (idempotent)."""
    global _RING
    if _RING is None:
        _RING = RingBufferHandler(capacity)
        logging.getLogger().addHandler(_RING)
    return _RING


def ring_lines() -> list:
    return _RING.snapshot() if _RING is not None else []


# ── journal ─────────────────────────────────────────────────────────────────
def _msg(v) -> str:
    if isinstance(v, list):  # journald ships non-UTF-8 messages as byte arrays
        try:
            return bytes(v).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return ""
    return "" if v is None else str(v)


def parse_journal_json(raw: str) -> list:
    out = []
    for row in (raw or "").splitlines():
        row = row.strip()
        if not row:
            continue
        try:
            d = json.loads(row)
        except ValueError:
            continue
        try:
            epoch = int(d.get("__REALTIME_TIMESTAMP")) / 1e6
        except (TypeError, ValueError):
            continue
        text = _msg(d.get("MESSAGE"))
        try:
            prio = int(d.get("PRIORITY", 6))
        except (TypeError, ValueError):
            prio = 6
        pid = d.get("_PID")
        for part in text.splitlines() or [""]:
            out.append({"ts": _iso(epoch), "_epoch": epoch,
                        "level": level_of(part, _PRIO_LEVELS.get(prio, "INFO")),
                        "unit": d.get("_SYSTEMD_USER_UNIT") or d.get("SYSLOG_IDENTIFIER") or UNIT,
                        "text": part, "pid": int(pid) if str(pid or "").isdigit() else None})
    return out


def read_journal(n: int, since: Optional[float] = None, unit: str = UNIT,
                 timeout: float = 10.0) -> Optional[list]:
    """The unit's last ``n`` journal lines (after ``since``), or None when the
    journal is unavailable."""
    exe = shutil.which("journalctl")
    if not exe:
        return None
    cmd = [exe, "--user", "-u", unit, "-n", str(int(n)), "-o", "json", "--no-pager"]
    if since is not None:
        cmd += [f"--since=@{int(since)}"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:  # noqa: BLE001
        return None
    if p.returncode != 0 and not p.stdout:
        return None
    return parse_journal_json(p.stdout)


# ── filtering ───────────────────────────────────────────────────────────────
def is_error_line(line: dict) -> bool:
    return (str(line.get("level") or "").upper() in _ERR_LEVELS
            or bool(_ERR_RX.search(line.get("text") or "")))


def filter_errors(lines: Iterable[dict]) -> list:
    """WARNING+ / failure-looking lines, keeping each Python traceback whole
    (continuation lines are indented; the exception line closes it)."""
    out: List[dict] = []
    in_tb: dict = {}          # pid -> True while inside a traceback
    for ln in lines:
        text = ln.get("text") or ""
        pid = ln.get("pid")
        if in_tb.get(pid):
            out.append(ln)
            if not text.startswith((" ", "\t", _TB_START)) and text.strip():
                in_tb[pid] = False            # the exception line closes it
            continue
        if text.startswith(_TB_START):
            in_tb[pid] = True
            out.append(ln)
            continue
        if is_error_line(ln):
            out.append(ln)
    return out


def collect(lines: int = DEFAULT_LINES, errors: bool = False, since=None,
            journal=read_journal) -> dict:
    """Assemble the /logs payload: {lines:[{ts,level,unit,text,pid}], source}."""
    try:
        n = max(1, min(int(lines), MAX_LINES))
    except (TypeError, ValueError):
        n = DEFAULT_LINES
    since_epoch = parse_since(since)
    source = "journal"
    rows = None
    try:
        rows = journal(n, since_epoch)
    except Exception:  # noqa: BLE001
        rows = None
    if not rows:
        source = "ring"
        rows = ring_lines()
    if since_epoch is not None:
        rows = [r for r in rows if r.get("_epoch", 0) > since_epoch]
    if errors:
        rows = filter_errors(rows)
    rows = rows[-n:]
    return {"source": source,
            "lines": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]}
