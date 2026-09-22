"""The todo-keeper node daemon — enroll, heartbeat, pull, answer, report.

The I/O half of the todo-keeper node (the pure half is ``todo_keeper.py``).
Stdlib + ``hugpy_platform`` only, on purpose: this runs as a long-lived
systemd unit (``hugpy-todo-keeper``) and every dependency it doesn't have is
a restart-loop it can't hit. The central URL comes from
``hugpy_platform.central_base_url`` (``HUGPY_BASE_URL``; the legacy
``HUGPY_CENTRAL`` alias still works) and the state file lives under the
platform's per-user config dir.

    enroll once   POST /agent/register            -> {id, token}  (token: ONCE)
    heartbeat     POST /agent/<id>/heartbeat      every HEARTBEAT_SECONDS
    pull          GET  /agent/<id>/tasks?since=   monotonic cursor, idempotent
    report        POST /agent/<id>/tasks/<seq>/result

THE TOKEN IS RETURNED EXACTLY ONCE (verified in comms/agent_nodes.py: register
returns the plaintext, the store keeps only its sha256). So enrollment is a
ONE-SHOT that must be durable across a VM restart: we persist {node_id, token}
to a 0600 state file OUTSIDE the git tree and re-use it forever. Lose that file
and the node is unrecoverable — it must re-enroll as a NEW node id, orphaning
the old row and any dispatch the host still points at.

WHY THE STATE FILE LIVES WHERE IT DOES (default ``<config_dir()>/todo-keeper.json``,
i.e. ``~/.config/hugpy/todo-keeper.json`` on Linux):
  * OUTSIDE the repo — it cannot be `git add`ed even by accident, which a path
    under dev/ could be (a .gitignore is a promise, not a guarantee: `git add -f`
    and a re-written ignore both defeat it). Not being in the tree is a guarantee.
  * Under $HOME — survives a VM restart (unlike /tmp or XDG_RUNTIME_DIR, which
    are cleared on boot and would silently orphan a node on every reboot).
  * 0600, and the directory 0700 — a token is a credential.
Override with HUGPY_TODO_KEEPER_STATE.

CURSOR DURABILITY: the cursor is persisted alongside the token, so a restart
does not re-run every task ever dispatched. It is advanced ONLY after a task is
reported (200 OR 409 — see below), so a crash mid-task re-pulls that task rather
than dropping it. At-least-once, never at-most-once: a re-run is idempotent
(the result route's 409 protects the first answer), a drop is a lost answer.

409 IS NOT AN ERROR. Re-posting a finalized task returns 409 (first report
wins). A node that crashed after reporting but before advancing its cursor will
re-answer and get 409 — that means "already recorded", so we advance the cursor
and move on. Treating it as a failure would wedge the node on that task forever.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import stat
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from hugpy_platform.app_dirs import config_dir
from hugpy_platform.central import DEFAULT_CENTRAL, central_base_url
from hugpy_ops.todo_keeper import handle_task

logger = logging.getLogger("hugpy.todo_keeper")

NODE_NAME = "todo-keeper"
NODE_CAPABILITIES = ["todo.add", "todo.tidy"]

# DEFAULT_CENTRAL is re-exported from hugpy_platform.central (loopback central).
# The console's live-verified model (2026-07-16) — the agent path and the
# console's direct fallback MUST use the same model, or the operator feels a
# behaviour change when one path takes over from the other.
# (Renamed from the bare DEFAULT_MODEL 2026-07-17: that name means something
# DIFFERENT in hugpy_agent and in central's model defaults — self-describing
# names only, so a shared .env can never cross-configure components.)
DEFAULT_TODO_KEEPER_MODEL = "Qwen2.5-7B-Instruct-GGUF"

HEARTBEAT_SECONDS = 30.0
POLL_SECONDS = 5.0
# Inference on a cold model can take a while (the fleet may have to load it);
# generous, but bounded — a hung call must not wedge the loop forever.
INFERENCE_TIMEOUT = 180.0
HTTP_TIMEOUT = 30.0


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _state_path() -> str:
    override = _env("HUGPY_TODO_KEEPER_STATE")
    if override:
        return override
    return os.path.join(config_dir(), "todo-keeper.json")


# ── durable state (node id + token + cursor) ────────────────────────────────
def load_state(path: Optional[str] = None) -> dict[str, Any]:
    p = path or _state_path()
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        # A corrupt state file is NOT silently discarded: re-enrolling would
        # orphan a live node row and strand whatever the host is polling. Fail
        # loudly so the operator can look, exactly the "refuse, never clobber"
        # rail the todo payloads get.
        raise RuntimeError(
            f"todo-keeper state file {p} is unreadable ({e}); refusing to "
            f"re-enroll over it. Inspect it, or move it aside to force a "
            f"fresh enrollment.") from e


def save_state(state: dict[str, Any], path: Optional[str] = None) -> None:
    """Persist state 0600 via atomic replace (a torn write here loses the
    token, which is unrecoverable — see the module docstring)."""
    p = path or _state_path()
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)
    try:
        os.chmod(d, stat.S_IRWXU)  # 0700
    except OSError:
        pass
    tmp = f"{p}.tmp.{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ── HTTP (stdlib) ───────────────────────────────────────────────────────────
class HttpError(Exception):
    def __init__(self, status: int, body: str = "") -> None:
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body


def _request(method: str, url: str, *, body: Any = None,
             headers: Optional[dict] = None,
             timeout: float = HTTP_TIMEOUT) -> tuple[int, Any]:
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(raw) if raw else None
            except Exception:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace") if e.fp else ""
        # 409 is a CONTRACT SIGNAL, not a failure — the caller decides.
        try:
            return e.code, json.loads(raw) if raw else raw
        except Exception:
            return e.code, raw


class TodoKeeperNode:
    """The daemon. Owns the sockets, the cursor, and the durable token."""

    def __init__(self, *, central: Optional[str] = None,
                 model: Optional[str] = None,
                 state_path: Optional[str] = None) -> None:
        # central_base_url honours HUGPY_BASE_URL and the legacy HUGPY_CENTRAL
        # alias (the unit still sets the latter) before falling back to loopback.
        self.central = (central or central_base_url(DEFAULT_CENTRAL)).rstrip("/")
        self.model = model or _env("HUGPY_TODO_KEEPER_MODEL", DEFAULT_TODO_KEEPER_MODEL)
        self.state_path = state_path or _state_path()
        self.state = load_state(self.state_path)
        self._stop = threading.Event()
        self._current_task: Optional[int] = None

    # -- credentials ---------------------------------------------------------
    @property
    def node_id(self) -> Optional[str]:
        return self.state.get("node_id")

    @property
    def token(self) -> Optional[str]:
        return self.state.get("token")

    @property
    def cursor(self) -> int:
        try:
            return int(self.state.get("cursor") or 0)
        except (TypeError, ValueError):
            return 0

    def _node_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _persist(self, **kw) -> None:
        self.state.update(kw)
        save_state(self.state, self.state_path)

    # -- enrollment ----------------------------------------------------------
    def enroll(self) -> None:
        """Enroll ONCE and persist the one-time token. No-op if already enrolled.

        ``HUGPY_API_KEY`` IS REQUIRED — not optional. ``/agent/register`` is gated
        by a PERMANENT console API-key check (operator ruling 2026-07-16: agent keys
        are "a separate api category entirely… gated permanently"). The gate does
        NOT consult the sitewide ``api_key_required`` toggle, and ``HUGPY_AGENT_OPEN``
        cannot waive it either — register MINTS credentials and is publicly
        reachable here, so it is the one door no switch opens. Enrolling bare 401s.

        This docstring previously said the route "stays open" when the site policy
        is off, and that enrolling bare matched the box's posture. Both were true
        on 2026-07-15 and are FALSE now — that open-when-off behaviour was the exact
        hole the ruling closed (a bare public POST minted a node). Kept accurate
        rather than left to mislead the next reader.

        The key is a console-minted key (console → API access → create key;
        ``POST /keys``, revocable via ``DELETE /keys/<id>``). It is a SECRET: pass it
        via the unit's ``HUGPY_API_KEY``, sourced from ``d-env/env`` (``CONSOLE_API``)
        — never hardcode it here or anywhere in the repo."""
        if self.node_id and self.token:
            logger.info("already enrolled as %s", self.node_id)
            return
        headers = {}
        # CONSOLE_API is the name the key already has in the share's env file
        # (d-env/env), which the unit loads via EnvironmentFile. systemd does NOT
        # shell-expand `Environment=HUGPY_API_KEY=${CONSOLE_API}` — that passes the
        # literal string and 401s (verified 2026-07-16) — so read the real name
        # directly. HUGPY_API_KEY still wins when explicitly set, for callers that
        # pass a different key.
        api_key = _env("HUGPY_API_KEY") or _env("CONSOLE_API")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        status, body = _request(
            "POST", f"{self.central}/agent/register",
            body={"name": NODE_NAME, "host": socket.gethostname(),
                  "capabilities": NODE_CAPABILITIES},
            headers=headers)
        if status == 401:
            raise RuntimeError(
                "enrollment refused (401): the site API-key policy is ON. Set "
                "HUGPY_API_KEY in the unit to a console-minted key and retry.")
        if status != 201 or not isinstance(body, dict):
            raise RuntimeError(f"enrollment failed: HTTP {status}: {body}")
        token = body.get("token")
        node_id = body.get("id")
        if not token or not node_id:
            raise RuntimeError(f"enrollment returned no token/id: {body}")
        # Persist BEFORE anything else can fail — the token is unrecoverable.
        self._persist(node_id=node_id, token=token, cursor=0,
                      enrolled_at=time.time())
        logger.info("enrolled as %s (token persisted to %s)",
                    node_id, self.state_path)

    # -- heartbeat -----------------------------------------------------------
    def heartbeat(self, status: str = "idle") -> None:
        code, body = _request(
            "POST", f"{self.central}/agent/{self.node_id}/heartbeat",
            body={"status": status,
                  "current_task": str(self._current_task)
                  if self._current_task is not None else None,
                  "version": f"todo-keeper/1 ({self.model})"},
            headers=self._node_headers())
        if code == 410:
            # Central forgot us (db reset). Re-enrolling is the ONLY recovery,
            # and it is safe: a 410 means the old row is gone, so there is
            # nothing to orphan.
            logger.warning("central returned 410 — re-enrolling")
            self.state = {}
            save_state(self.state, self.state_path)
            self.enroll()
            return
        if code == 403:
            raise RuntimeError("node revoked by the operator — stopping")
        if code != 200:
            logger.warning("heartbeat HTTP %s: %s", code, body)

    # -- inference through hugpy --------------------------------------------
    def _complete(self, messages: list[dict[str, str]]) -> str:
        """One chat completion through hugpy's own gateway.

        Central runs HUGPY_NO_LOCAL_SERVING=true — it does not serve locally;
        this call is routed to the worker fleet. temperature=0 because we are
        extracting structure, not writing prose: we want the same list twice."""
        code, body = _request(
            "POST", f"{self.central}/v1/chat/completions",
            body={"model": self.model, "messages": messages,
                  "temperature": 0, "max_tokens": 1024},
            headers={"Authorization": f"Bearer {_env('HUGPY_API_KEY')}"}
            if _env("HUGPY_API_KEY") else None,
            timeout=INFERENCE_TIMEOUT)
        if code != 200 or not isinstance(body, dict):
            raise RuntimeError(f"chat/completions HTTP {code}: {str(body)[:200]}")
        try:
            return body["choices"][0]["message"]["content"] or ""
        except Exception as e:
            raise RuntimeError(f"malformed completion response: {e}") from e

    # -- the task loop -------------------------------------------------------
    def report(self, seq: int, outcome: dict[str, Any]) -> bool:
        """POST a result. Returns True if RECORDED — 200 or 409 both count.

        409 = already finalized (first report wins). A crashed node re-posting
        gets 409 and must advance past it; treating that as failure would wedge
        the node on one task forever. See the module docstring."""
        code, body = _request(
            "POST", f"{self.central}/agent/{self.node_id}/tasks/{seq}/result",
            body={"status": outcome["status"], "result": outcome["result"]},
            headers=self._node_headers())
        if code == 200:
            return True
        if code == 409:
            logger.info("task %s already finalized (409) — treating as recorded", seq)
            return True
        logger.error("reporting task %s failed: HTTP %s: %s", seq, code, body)
        return False

    def poll_once(self) -> int:
        """Pull from the cursor and handle everything new. Returns the count."""
        code, body = _request(
            "GET", f"{self.central}/agent/{self.node_id}/tasks?since={self.cursor}",
            headers=self._node_headers())
        if code == 410:
            logger.warning("central returned 410 on pull — re-enrolling")
            self.state = {}
            save_state(self.state, self.state_path)
            self.enroll()
            return 0
        if code != 200 or not isinstance(body, dict):
            logger.warning("pull HTTP %s: %s", code, body)
            return 0
        tasks = body.get("tasks") or []
        handled = 0
        for t in tasks:
            if self._stop.is_set():
                break
            seq = t.get("seq")
            self._current_task = seq
            try:
                self.heartbeat(status="busy")
            except RuntimeError:
                raise
            except Exception:
                pass
            outcome = handle_task(t.get("task"), self._complete)
            if self.report(seq, outcome):
                # Advance ONLY after the result is recorded — a crash before
                # this re-pulls the task (at-least-once), which the 409 path
                # makes safe.
                self._persist(cursor=int(seq))
                handled += 1
            self._current_task = None
        if handled:
            try:
                self.heartbeat(status="idle")
            except Exception:
                pass
        return handled

    def stop(self, *_a) -> None:
        logger.info("stop requested")
        self._stop.set()

    def run(self) -> None:
        self.enroll()
        logger.info("todo-keeper node %s up (central=%s model=%s)",
                    self.node_id, self.central, self.model)
        last_beat = 0.0
        self.heartbeat(status="idle")
        last_beat = time.time()
        while not self._stop.is_set():
            try:
                self.poll_once()
                if time.time() - last_beat >= HEARTBEAT_SECONDS:
                    self.heartbeat(status="idle")
                    last_beat = time.time()
            except RuntimeError as e:      # revoked -> stop
                logger.error("%s", e)
                return
            except Exception as e:         # never die on a transient fault
                logger.warning("loop error (continuing): %s", e)
            self._stop.wait(POLL_SECONDS)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hugpy-todo-keeper",
        description="The todo-keeper agent node: enroll once, heartbeat, pull "
                    "todo.add / todo.tidy tasks, answer through central's "
                    "model, report. Long-lived; SIGTERM stops it cleanly.")
    ap.add_argument("--central", default=None,
                    help="central base URL (default: HUGPY_BASE_URL / legacy "
                         "HUGPY_CENTRAL, else loopback central)")
    ap.add_argument("--model", default=None,
                    help="model key answering tasks (default: "
                         "HUGPY_TODO_KEEPER_MODEL or %s)" % DEFAULT_TODO_KEEPER_MODEL)
    ap.add_argument("--state", default=None,
                    help="node id/token/cursor state file (default: "
                         "HUGPY_TODO_KEEPER_STATE or <config dir>/todo-keeper.json)")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    node = TodoKeeperNode(central=args.central, model=args.model,
                          state_path=args.state)
    signal.signal(signal.SIGTERM, node.stop)
    signal.signal(signal.SIGINT, node.stop)
    try:
        node.run()
    except Exception as e:
        logger.error("fatal: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
