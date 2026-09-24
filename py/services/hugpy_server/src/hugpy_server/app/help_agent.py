"""The console's HELP AGENT — the query path behind the console Help button.

WHY
---
Operator (2026-09-23): "the help button, i need this to be a thing that's real,
it needs to be connected with a query path for the hugpy-agent, and should be
able to view logs for hugpy, edit code, and generally provide use."

Before this, the console's Help button opened the member Keeper widget
(``react/ui_shared/help/helpWidget.js`` -> ``/keeper/help/ask``): one JSON
round-trip to the keeper seat, no tools, no logs, no code. This module is the
OPERATOR path: a persistent help session with an agent that can read hugpy's
logs and code, edit code under the single source tree, run tests and queue a
verify — and nothing past that.

SHAPE
-----
``HelpService`` owns sessions. Each session is one append-only JSONL file under
``$PROJECTS_HOME/help_sessions/<id>.jsonl`` (the review record — every line the
operator sent, every event the agent produced, which backend answered).

Backends share one small interface (``available / start / send / pull / stop``):

  * ``ClaudeArmBackend`` — an ``abstract-claude serve`` console API
    (``/api/console/sessions|chat|events|interrupt``). A headless Claude Code
    session with file + shell tools, started in ``/srv/hugpy/src/hugpy`` and
    primed with ``system_context()``. Preferred when reachable.
  * ``LocalModelBackend`` — hugpy's own ``/v1/chat/completions``. Chat only,
    NO tools, labelled "local model, read-only"; grounded with a bounded
    read-only snapshot (failed loads, central journal errors, workers).

The backend is feature-detected at session start and recorded in the
transcript; a session never changes backend.

HARD RULES (in the system context; the Claude arm's own permission mode — the
serve user's ``defaultMode`` — is the enforcement layer, abstract-claude has no
per-session deny list): no service restarts, no promotion, no worker boxes, no
deletes, no package installs. Edits reach live ONLY via pkg_src verify ->
known-good -> promote, and promotion is the operator's explicit act.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Callable, Optional

SOURCE_ROOT = os.environ.get("HUGPY_HELP_SOURCE_ROOT", "/srv/hugpy/src/hugpy")
CENTRAL_UNIT = os.environ.get("HUGPY_HELP_CENTRAL_UNIT", "7002_hugpy_api.service")
SELF_BASE = os.environ.get("HUGPY_SELF_BASE", "http://127.0.0.1:7002")
AUDIT_PATH_NAME = "model_audit.json"

# hugpy's own abstract-claude serve (user hugpy, :9125) first — it runs as the
# same user as central and owns the tree — then the vm_mgr seat (:9123, the one
# behind dev.hugpy.ai/claude). Comma-separated override.
DEFAULT_CLAUDE_URLS = "http://127.0.0.1:9125,http://127.0.0.1:9123"

MAX_PROMPT = 20000
MAX_CONTEXT = 6000
MAX_TOOL_TEXT = 2000
ID_RX = re.compile(r"^hs-[0-9a-f]{12}$")


# --------------------------------------------------------------------------- #
# system context
# --------------------------------------------------------------------------- #
def system_context() -> str:
    """What the agent is, where things are, and what it must never do."""
    return f"""You are the HUGPY HELP AGENT, opened from the hugpy console's Help button by the
hugpy OPERATOR. Be concretely useful: find the cause, cite the evidence (log lines,
compute-action rows, file:line), and fix it in code when a code fix is warranted.

WHERE THINGS ARE
- Single source tree: {SOURCE_ROOT} (you are running in it). Python packages live in
  py/<group>/<pkg>/src/<pkg> (hugpy_server = central API, hugpy_fleet = central
  fleet + worker agent, hugpy_engine = inference/resolution, hugpy_ops, hugpy_storage,
  hugpy_curation, hugpy_platform, hugpy_control). Console UI: react/ui/src.
  PARTITION.md / CONSISTENCY.md describe the layout and release discipline.
- Central API: {SELF_BASE} (systemd unit {CENTRAL_UNIT}, runs as user hugpy).
- Storage/state: $PROJECTS_HOME (default /mnt/16T_toshiba/llm_storage/projects).

LOGS AND DIAGNOSTICS (all read-only)
- Central journal:  journalctl -u {CENTRAL_UNIT} --no-pager -n 200 [--since today] [-p warning]
- Failed loads / compute log (open GET, newest first):
    curl -s '{SELF_BASE}/llm/compute-actions?action=load&outcome=fail&worker=<name>&since=<epoch>&limit=50'
  (filters: action=load|call|evict|provision|fit_fail, model=, worker=, outcome=)
- Workers:          curl -s {SELF_BASE}/llm/workers   (id, name, status)
- Worker aggregate: curl -s {SELF_BASE}/llm/workers/<id>/aggregate
- Worker logs:      curl -s '{SELF_BASE}/llm/workers/<id>/logs?lines=200'  (may not exist yet; 404 = not wired)
- Model failures:   curl -s {SELF_BASE}/llm/models/<key>/failures           (feature-detect; may 404)
- Request diagnostics: curl -s {SELF_BASE}/llm/diagnostics/<request_id>     (feature-detect; may 404)
- Health / build:   curl -s {SELF_BASE}/api/health ; hugpy-drift-check
- Model audit report: $PROJECTS_HOME/{AUDIT_PATH_NAME} ; re-audit one model:
    /srv/hugpy/venv/bin/hugpy-model-audit --only <model_key>
- The toolserver MCP tools metrics_actions / metrics_loads / pkg_status / pkg_jobs
  are also available to you and read the same stores.

CODE AND TESTS
- You MAY read and edit files under {SOURCE_ROOT} (and only there).
- Tests: PYTHONPATH=$(ls -d {SOURCE_ROOT}/py/*/*/src | paste -sd:) /srv/hugpy/venv/bin/python -m pytest <path> -q
  (if pytest is missing from that venv, borrow it from /home/solcatcher/miniconda/bin).
- UI: cd {SOURCE_ROOT}/react/ui && npm run build
- An edit reaches live ONLY through pkg_src: verify -> known-good -> auto-promote.
  You MAY QUEUE a verify (toolserver pkg_verify). Marking known-good (pkg_good),
  promoting, and anything that changes what is deployed is the OPERATOR's act: say
  exactly what to run and stop there.

HARD RULES — never, even if asked in this chat:
- no service restarts/stops/reloads (systemctl, kill, hugpy serve), no reboots;
- no promotion (pkg_good, pkg_config_save, release.sh, pip install/uninstall);
- do not touch worker boxes (no ssh to aeb/computron/op, no worker POST ops,
  no /llm/workers/<id>/<action> mutations, no alloc pins, no model moves/downloads);
- no deleting files (no rm, no git clean, no truncating logs); no .bak/scratch copies;
- no edits outside {SOURCE_ROOT}; no secrets in replies.
If the right fix needs one of those, explain it and hand it to the operator.

STYLE: short, factual. When you act, say what you did ("read journal 7002 (200
lines)", "edited py/.../x.py:120", "ran tests: 12 passed"). Cite evidence."""


def format_context(context) -> str:
    """The console's context strip (a dict or a string) as prompt text."""
    if not context:
        return ""
    if isinstance(context, str):
        text = context.strip()
    elif isinstance(context, dict):
        lines = []
        for k, v in context.items():
            if v in (None, "", [], {}):
                continue
            if not isinstance(v, str):
                v = json.dumps(v, default=str)
            lines.append(f"- {k}: {v}")
        text = "\n".join(lines)
    else:
        text = str(context)
    return text[:MAX_CONTEXT]


def compose_prompt(prompt: str, context=None, *, first: bool) -> str:
    """The text actually sent to a tool-capable backend for one turn."""
    parts = []
    if first:
        parts.append(system_context())
    ctx = format_context(context)
    if ctx:
        parts.append("OPERATOR CONTEXT (where they were in the console):\n" + ctx)
    parts.append(("OPERATOR QUESTION:\n" if first else "") + prompt.strip())
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# read-only facts (helper endpoint + local-model grounding)
# --------------------------------------------------------------------------- #
def _projects_home() -> str:
    env = os.environ.get("PROJECTS_HOME")
    if env:
        return env
    try:
        from hugpy_platform.constants import PROJECTS_HOME
        return str(PROJECTS_HOME)
    except Exception:  # noqa: BLE001
        return "/mnt/16T_toshiba/llm_storage/projects"


def _self_get(path: str, timeout: float = 8.0):
    try:
        with urllib.request.urlopen(SELF_BASE + path, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception:  # noqa: BLE001
        return None, None


def _clamp(value, lo, hi, default):
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def read_logs(source: str, *, lines=200, worker: str = "", model: str = "",
              since: Optional[float] = None, request_id: str = "",
              errors: bool = False) -> dict:
    """Bounded, read-only log/diagnostic readout. Never raises.

    source: central | worker | compute-actions | failures | diagnostics | audit | workers
    """
    n = _clamp(lines, 1, 500, 200)
    source = (source or "").strip().lower()
    try:
        if source == "central":
            if not shutil.which("journalctl"):
                return {"source": source, "error": "journalctl not available"}
            cmd = ["journalctl", "-u", CENTRAL_UNIT, "--no-pager", "-o", "short-iso",
                   "-n", str(n)]
            if errors:
                cmd += ["-p", "warning"]
            if since:
                cmd += ["--since", time.strftime("%Y-%m-%d %H:%M:%S",
                                                 time.localtime(float(since)))]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            out = (p.stdout or "").splitlines()[-n:]
            return {"source": source, "unit": CENTRAL_UNIT, "lines": out,
                    "error": (p.stderr or "").strip() or None}
        if source == "workers":
            st, body = _self_get("/llm/workers")
            rows = body if isinstance(body, list) else (body or {}).get("workers") or []
            return {"source": source, "workers": [
                {k: w.get(k) for k in ("id", "name", "status", "version", "gpu", "last_seen")}
                for w in rows if isinstance(w, dict)]}
        if source == "compute-actions":
            q = {"limit": min(n, 200)}
            for k, v in (("worker", worker), ("model", model)):
                if v:
                    q[k] = v
            q["action"] = "load"
            q["outcome"] = "fail"
            if since:
                q["since"] = str(since)
            st, body = _self_get("/llm/compute-actions?" + urllib.parse.urlencode(q))
            return {"source": source, "http": st, "actions": (body or {}).get("actions", [])}
        if source == "worker":
            if not worker:
                return {"source": source, "error": "worker (id) is required"}
            wid = urllib.parse.quote(worker, safe="")
            st, body = _self_get(f"/llm/workers/{wid}/logs?lines={n}"
                                 + ("&errors=1" if errors else ""))
            if st == 200:
                return {"source": source, "via": "logs", "body": body}
            st2, agg = _self_get(f"/llm/workers/{wid}/aggregate")
            return {"source": source, "via": "aggregate", "logs_http": st,
                    "http": st2, "body": agg}
        if source == "failures":
            if not model:
                return {"source": source, "error": "model is required"}
            st, body = _self_get(f"/llm/models/{urllib.parse.quote(model, safe='')}/failures")
            return {"source": source, "http": st, "supported": st != 404, "body": body}
        if source == "diagnostics":
            if not request_id:
                return {"source": source, "error": "request_id is required"}
            st, body = _self_get(f"/llm/diagnostics/{urllib.parse.quote(request_id, safe='')}")
            return {"source": source, "http": st, "supported": st != 404, "body": body}
        if source == "audit":
            path = os.path.join(_projects_home(), AUDIT_PATH_NAME)
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError) as exc:
                return {"source": source, "path": path, "error": str(exc)}
            if model and isinstance(data, dict):
                models = data.get("models") if isinstance(data.get("models"), dict) else data
                data = {model: (models or {}).get(model)}
            # whole (2026-09-23): no 60000-char cut
            return {"source": source, "path": path, "truncated": False,
                    "body": json.loads(json.dumps(data, default=str))}
    except Exception as exc:  # noqa: BLE001 — a read surface never 500s
        return {"source": source, "error": f"{type(exc).__name__}: {exc}"}
    return {"source": source, "error": "unknown source",
            "sources": ["central", "worker", "compute-actions", "failures",
                        "diagnostics", "audit", "workers"]}


def grounding_snapshot() -> str:
    """Compact read-only facts for the no-tools local model."""
    since = time.time() - 86400
    fails = read_logs("compute-actions", lines=25, since=since).get("actions") or []
    slim = []
    for a in fails[:25]:
        d = a.get("detail") or {}
        slim.append({"ts": a.get("ts"), "worker": a.get("worker"),
                     "model": a.get("model") or a.get("model_key"),
                     "class": d.get("class") if isinstance(d, dict) else None,
                     "message": str((d.get("message") if isinstance(d, dict) else d) or "")[:300]})
    central = read_logs("central", lines=40, errors=True).get("lines") or []
    workers = read_logs("workers").get("workers") or []
    return json.dumps({"failed_loads_last_24h": slim,
                       "central_journal_warnings": [ln[-300:] for ln in central],
                       "workers": workers}, default=str)[:12000]


# --------------------------------------------------------------------------- #
# transcript store — append-only JSONL
# --------------------------------------------------------------------------- #
class HelpStore:
    def __init__(self, root: Optional[str] = None):
        self._root = root

    @property
    def root(self) -> str:
        return self._root or os.environ.get("HUGPY_HELP_SESSIONS_DIR") or os.path.join(
            _projects_home(), "help_sessions")

    def path(self, sid: str) -> str:
        if not ID_RX.match(sid or ""):
            raise KeyError(sid)
        return os.path.join(self.root, f"{sid}.jsonl")

    def exists(self, sid: str) -> bool:
        try:
            return os.path.exists(self.path(sid))
        except KeyError:
            return False

    def new_id(self) -> str:
        return "hs-" + uuid.uuid4().hex[:12]

    def _ensure(self, sid: str) -> str:
        """Create the dir/file group-writable: central (user hugpy) and an
        operator's shell both append to the same records."""
        os.makedirs(self.root, exist_ok=True)
        path = self.path(sid)
        if not os.path.exists(path):
            fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o664)
            os.close(fd)
            try:
                os.chmod(path, 0o664)
            except OSError:
                pass
        return path

    def append(self, sid: str, *records: dict) -> None:
        with open(self._ensure(sid), "a", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                for rec in records:
                    rec = dict(rec)
                    rec.setdefault("t", time.time())
                    fh.write(json.dumps(rec, default=str) + "\n")
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def read(self, sid: str) -> list:
        try:
            with open(self.path(sid), encoding="utf-8") as fh:
                out = []
                for line in fh:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
                return out
        except OSError:
            return []

    def locked(self, sid: str):
        """An exclusive flock on the transcript for read-then-append syncs
        (two central processes / threads must not double-append a pull)."""
        store = self

        class _Lock:
            def __enter__(self):
                self.fh = open(store._ensure(sid), "a+", encoding="utf-8")
                fcntl.flock(self.fh, fcntl.LOCK_EX)
                return self

            def __exit__(self, *exc):
                fcntl.flock(self.fh, fcntl.LOCK_UN)
                self.fh.close()

        return _Lock()

    def list(self, limit: int = 100) -> list:
        try:
            names = [n for n in os.listdir(self.root) if n.endswith(".jsonl")]
        except OSError:
            return []
        rows = []
        for name in names:
            sid = name[:-6]
            if not ID_RX.match(sid):
                continue
            recs = self.read(sid)
            meta = next((r for r in recs if r.get("kind") == "meta"), {})
            first_q = next((r.get("text", "") for r in recs if r.get("kind") == "user"), "")
            rows.append({"id": sid, "created": meta.get("t"),
                         "updated": recs[-1].get("t") if recs else None,
                         "backend": meta.get("backend"), "by": meta.get("by"),
                         "title": first_q[:120], "status": derive_status(recs)})
        rows.sort(key=lambda r: r.get("updated") or 0, reverse=True)
        return rows[:limit]


def derive_status(records: list) -> str:
    """idle | busy | error | stopped — from the transcript alone.

    busy while any sent message id is not yet covered by a ``done`` event."""
    pending: set = set()
    last_done = None
    stopped = False
    for r in records:
        k = r.get("kind")
        if k == "user":
            pending.update(r.get("message_ids") or [])
            stopped = False
        elif k == "event" and r.get("type") == "done":
            pending.difference_update(r.get("message_ids") or [])
            last_done = r
        elif k == "stopped":
            stopped = True
        elif k == "error":
            last_done = {"error": r.get("error")}
            pending.clear()
    if stopped:
        return "stopped"
    if pending:
        return "busy"
    if last_done and (last_done.get("error") or (last_done.get("rc") not in (None, 0))):
        return "error"
    return "idle"


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class BackendError(RuntimeError):
    pass


def _http_json(method: str, url: str, body=None, timeout: float = 10.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8", "replace") or "{}")
        except ValueError:
            payload = {}
        return exc.code, payload


class ClaudeArmBackend:
    """abstract-claude serve's console session API (headless Claude Code)."""

    name = "claude-arm"
    tools = True

    def __init__(self, base_url: str, cwd: str = SOURCE_ROOT, model: str = ""):
        self.base = base_url.rstrip("/")
        self.cwd = cwd
        self.model = model or os.environ.get("HUGPY_HELP_CLAUDE_MODEL", "")

    @property
    def label(self) -> str:
        return f"claude-arm @ {self.base} (headless Claude Code, cwd {self.cwd})"

    def available(self) -> bool:
        try:
            st, body = _http_json("GET", self.base + "/api/console/sessions", timeout=3)
        except Exception:  # noqa: BLE001
            return False
        return st == 200 and "claude" in ((body or {}).get("backends") or [])

    def start(self, sid: str, text: str) -> dict:
        cfg = {"backend": "claude", "cwd": self.cwd, "label": f"hugpy-help {sid}"}
        if self.model:
            cfg["model"] = self.model
        st, body = _http_json("POST", self.base + "/api/console/sessions", cfg)
        if st != 200 or not (body or {}).get("id"):
            raise BackendError(f"claude-arm session create failed ({st}): {(body or {}).get('error')}")
        ref = body["id"]
        sent = self.send(ref, text)
        return {"ref": ref, **sent}

    def send(self, ref: str, text: str) -> dict:
        st, body = _http_json("POST", self.base + "/api/console/chat",
                              {"session_id": ref, "prompt": text})
        if st != 200:
            raise BackendError(f"claude-arm send failed ({st}): {(body or {}).get('error')}")
        return {"message_ids": body.get("message_ids") or [], "queued": bool(body.get("queued"))}

    def pull(self, ref: str, since: int) -> list:
        q = urllib.parse.urlencode({"id": ref, "since": int(since or 0)})
        st, body = _http_json("GET", f"{self.base}/api/console/events?{q}", timeout=10)
        if st != 200:
            raise BackendError(f"claude-arm events failed ({st}): {(body or {}).get('error')}")
        return list((body or {}).get("events") or [])

    def stop(self, ref: str) -> None:
        _http_json("POST", self.base + "/api/console/interrupt", {"session_id": ref})


class LocalModelBackend:
    """hugpy's own /v1 — chat only, no tools, read-only."""

    name = "local-model"
    tools = False
    label = "local model via hugpy /v1 — read-only, no tools"

    def __init__(self, base: str = SELF_BASE, model: str = ""):
        self.base = base.rstrip("/")
        self.model = model or os.environ.get("HUGPY_HELP_LOCAL_MODEL", "default")
        self.api_key = (os.environ.get("HUGPY_HELP_LOCAL_API_KEY")
                        or os.environ.get("HUGPY_API_KEY") or "")

    def available(self) -> bool:
        try:
            st, body = _http_json("GET", self.base + "/v1/models", timeout=3)
        except Exception:  # noqa: BLE001
            return False
        return st == 200 and bool((body or {}).get("data"))

    def complete(self, messages: list, timeout: float = 240) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base + "/v1/chat/completions", method="POST", headers=headers,
            data=json.dumps({"model": self.model, "messages": messages,
                             "max_tokens": 1200, "temperature": 0.2}).encode())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
        return ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def claude_backends() -> list:
    urls = os.environ.get("HUGPY_HELP_CLAUDE_URL", DEFAULT_CLAUDE_URLS)
    return [ClaudeArmBackend(u.strip()) for u in urls.split(",") if u.strip()]


def select_backend(candidates: Optional[list] = None, prefer: str = ""):
    """First available backend. ``prefer='local'`` skips the Claude arm."""
    cands = candidates if candidates is not None else (claude_backends() + [LocalModelBackend()])
    if prefer == "local":
        cands = [c for c in cands if c.name == "local-model"] or cands
    tried = []
    for c in cands:
        ok = False
        try:
            ok = bool(c.available())
        except Exception:  # noqa: BLE001
            ok = False
        tried.append({"backend": c.name, "label": c.label, "available": ok})
        if ok:
            return c, tried
    return None, tried


# --------------------------------------------------------------------------- #
# service
# --------------------------------------------------------------------------- #
class HelpService:
    def __init__(self, store: Optional[HelpStore] = None,
                 backends_factory: Optional[Callable[[], list]] = None):
        self.store = store or HelpStore()
        self._factory = backends_factory
        self._locks: dict = {}
        self._guard = threading.Lock()

    def _lock(self, sid):
        with self._guard:
            return self._locks.setdefault(sid, threading.Lock())

    def _candidates(self):
        return self._factory() if self._factory else None

    def _backend_for(self, meta: dict):
        name = meta.get("backend")
        if name == "claude-arm":
            return ClaudeArmBackend(meta.get("backend_url") or DEFAULT_CLAUDE_URLS.split(",")[0])
        if name == "local-model":
            return LocalModelBackend()
        raise BackendError(f"unknown backend {name!r}")

    def _resolve(self, sid, meta):
        # tests inject instances; keep the one that answered start
        for c in (self._candidates() or []):
            if c.name == meta.get("backend") and getattr(c, "base", None) == meta.get("backend_url"):
                return c
        return self._backend_for(meta)

    @staticmethod
    def meta(records):
        return next((r for r in records if r.get("kind") == "meta"), None)

    # -- start ---------------------------------------------------------------
    def start(self, prompt: str, context=None, *, by: str = "", prefer: str = "") -> dict:
        prompt = (prompt or "").strip()[:MAX_PROMPT]
        if not prompt:
            raise ValueError("prompt is required")
        backend, tried = select_backend(self._candidates(), prefer=prefer)
        sid = self.store.new_id()
        meta = {"kind": "meta", "id": sid, "by": by or "operator",
                "backend": backend.name if backend else None,
                "backend_label": backend.label if backend else None,
                "backend_url": getattr(backend, "base", None) if backend else None,
                "tools": bool(getattr(backend, "tools", False)) if backend else False,
                "tried": tried, "cwd": SOURCE_ROOT}
        self.store.append(sid, meta)
        if backend is None:
            self.store.append(sid, {"kind": "user", "text": prompt,
                                    "context": format_context(context)},
                              {"kind": "error", "error": "no help backend reachable "
                               "(Claude arm and local /v1 both down)"})
            return self.transcript(sid, sync=False)
        self.store.append(sid, {"kind": "backend", "backend": backend.name,
                                "label": backend.label})
        self._send(sid, meta, backend, prompt, context, first=True)
        return self.transcript(sid, sync=False)

    # -- send ----------------------------------------------------------------
    def send(self, sid: str, text: str, context=None, *, by: str = "") -> dict:
        text = (text or "").strip()[:MAX_PROMPT]
        if not text:
            raise ValueError("text is required")
        recs = self.store.read(sid)
        meta = self.meta(recs)
        if not meta:
            raise KeyError(sid)
        if not meta.get("backend"):
            raise BackendError("this session has no backend; start a new one")
        backend = self._resolve(sid, meta)
        self._send(sid, meta, backend, text, context, first=False,
                   ref=next((r.get("ref") for r in recs if r.get("kind") == "ref"), None),
                   history=recs)
        return self.transcript(sid, sync=False)

    def _send(self, sid, meta, backend, text, context, *, first, ref=None, history=None):
        ctx_text = format_context(context)
        if backend.name == "local-model":
            mid = uuid.uuid4().hex
            self.store.append(sid, {"kind": "user", "text": text, "context": ctx_text,
                                    "message_ids": [mid]})
            msgs = self._local_messages(history or [], text, ctx_text)
            threading.Thread(target=self._local_turn, args=(sid, backend, msgs, mid),
                             daemon=True, name=f"help-{sid}").start()
            return
        payload = compose_prompt(text, context, first=first)
        try:
            if first:
                res = backend.start(sid, payload)
                self.store.append(sid, {"kind": "ref", "ref": res["ref"]})
            else:
                if not ref:
                    raise BackendError("session has no backend ref")
                res = backend.send(ref, payload)
        except Exception as exc:  # noqa: BLE001
            self.store.append(sid, {"kind": "user", "text": text, "context": ctx_text},
                              {"kind": "error", "error": str(exc)})
            return
        self.store.append(sid, {"kind": "user", "text": text, "context": ctx_text,
                                "message_ids": res.get("message_ids") or [],
                                "queued": res.get("queued", False)})

    def _local_messages(self, history, text, ctx_text):
        sys_msg = ("You are the hugpy help agent in LOCAL READ-ONLY mode: you have no tools, "
                   "cannot read files or run commands. Answer from the facts below; say "
                   "plainly when you would need to look at code/logs you cannot see, and "
                   "what the operator should run.\n\nREAD-ONLY FACTS (live snapshot):\n"
                   + grounding_snapshot())
        msgs = [{"role": "system", "content": sys_msg}]
        buf = ""
        for r in history:
            if r.get("kind") == "user":
                if buf:
                    msgs.append({"role": "assistant", "content": buf})
                    buf = ""
                msgs.append({"role": "user", "content": r.get("text", "")})
            elif r.get("kind") == "event" and r.get("type") == "text":
                buf += r.get("text", "")
        if buf:
            msgs.append({"role": "assistant", "content": buf})
        user = text if not ctx_text else f"{text}\n\n(console context)\n{ctx_text}"
        msgs.append({"role": "user", "content": user})
        return msgs[-24:] if len(msgs) > 24 else msgs

    def _local_turn(self, sid, backend, msgs, mid):
        try:
            reply = backend.complete(msgs)
            self.store.append(sid, {"kind": "event", "type": "text", "text": reply},
                              {"kind": "event", "type": "done", "rc": 0,
                               "message_ids": [mid]})
        except Exception as exc:  # noqa: BLE001
            self.store.append(sid, {"kind": "event", "type": "done", "rc": 1,
                                    "error": f"local model failed: {exc}",
                                    "message_ids": [mid]})

    # -- pull / transcript --------------------------------------------------------
    def sync(self, sid: str) -> None:
        """Pull new backend events into the transcript (Claude arm only)."""
        with self._lock(sid):
            with self.store.locked(sid):
                recs = self.store.read(sid)
                meta = self.meta(recs)
                if not meta or meta.get("backend") != "claude-arm":
                    return
                ref = next((r.get("ref") for r in recs if r.get("kind") == "ref"), None)
                if not ref:
                    return
                cursor = max([r.get("seq") or 0 for r in recs if r.get("kind") == "event"] or [0])
                try:
                    events = self._resolve(sid, meta).pull(ref, cursor)
                except Exception as exc:  # noqa: BLE001
                    return self._note_pull_error(sid, recs, exc)
                new = []
                for ev in events:
                    if (ev.get("seq") or 0) <= cursor:
                        continue
                    new.append(slim_event(ev))
                # still under the transcript flock: a concurrent sync blocks
                # until these rows land, so it re-reads a cursor past them
                if new:
                    self._append_raw(sid, new)

    def _append_raw(self, sid, recs):
        path = self.store.path(sid)
        with open(path, "a", encoding="utf-8") as fh:
            for rec in recs:
                rec = dict(rec)
                rec.setdefault("t", time.time())
                fh.write(json.dumps(rec, default=str) + "\n")

    def _note_pull_error(self, sid, recs, exc):
        last = recs[-1] if recs else {}
        if last.get("kind") == "pull_error" and last.get("error") == str(exc):
            return
        self._append_raw(sid, [{"kind": "pull_error", "error": str(exc)}])

    def transcript(self, sid: str, *, sync: bool = True, since: int = 0) -> dict:
        if not self.store.exists(sid):
            raise KeyError(sid)
        if sync:
            self.sync(sid)
        recs = self.store.read(sid)
        meta = self.meta(recs) or {}
        return {"id": sid, "backend": meta.get("backend"),
                "backend_label": meta.get("backend_label"),
                "tools": meta.get("tools", False), "by": meta.get("by"),
                "created": meta.get("t"), "status": derive_status(recs),
                "count": len(recs),
                "records": [dict(r, i=i) for i, r in enumerate(recs) if i >= since]}

    def stop(self, sid: str) -> dict:
        recs = self.store.read(sid)
        meta = self.meta(recs)
        if not meta:
            raise KeyError(sid)
        ref = next((r.get("ref") for r in recs if r.get("kind") == "ref"), None)
        err = None
        if meta.get("backend") == "claude-arm" and ref:
            try:
                self._resolve(sid, meta).stop(ref)
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
        self.store.append(sid, {"kind": "stopped", "error": err})
        return self.transcript(sid, sync=False)


def slim_event(ev: dict) -> dict:
    """A backend event as stored WHOLE (2026-09-23: no caps), with the
    backend's seq kept as the cursor."""
    out = {"kind": "event", "seq": ev.get("seq"), "type": ev.get("type")}
    t = ev.get("type")
    if t == "text":
        out["text"] = ev.get("text", "")
    elif t == "tool":
        out.update(name=ev.get("name"), summary=str(ev.get("summary") or ""),
                   input=str(ev.get("input") or ""))
    elif t == "tool_result":
        out.update(is_error=bool(ev.get("is_error")),
                   text=str(ev.get("text") or ""))
    elif t == "thinking":
        out["text"] = str(ev.get("text") or "")
    elif t == "done":
        out.update(rc=ev.get("rc"), error=ev.get("error"),
                   result=str(ev.get("result") or ""),
                   message_ids=ev.get("message_ids") or [], held=ev.get("held"))
    elif t == "system":
        out.update(model=ev.get("model"), tools=ev.get("tools"))
    elif t == "status":
        out["state"] = ev.get("state")
    elif t in ("note", "error"):
        out["text"] = str(ev.get("text") or ev.get("error") or "")
    elif t == "user":
        out["via"] = ev.get("via")  # the prompt text itself is already our "user" record
    return out


_SERVICE: Optional[HelpService] = None


def service() -> HelpService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = HelpService()
    return _SERVICE
