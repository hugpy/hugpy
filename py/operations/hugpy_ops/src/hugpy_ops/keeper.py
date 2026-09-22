"""hugpy keeper — a stationary terminal REPL in which a model KEEPS a machine.

The keeper concept: a long-lived warden for a box and its effects — it watches
health, services, and project files, converses in a terminal, and acts through
a shell. The same role Claude Code plays when launched as a station's keeper,
with a hugpy-served model as the brain.

    hugpy-keeper --model <id>                          # keep THIS machine
    hugpy-keeper --model <id> --exec lxc:solcatcher    # keep an LXD instance
    python3 path/to/hugpy_ops/keeper.py ...            # no install needed

Brain: any OpenAI-compatible /v1/chat/completions (a hugpy central; streaming
SSE). Point --central (or HUGPY_CENTRAL) at one that is reachable FROM WHERE
THE KEEPER RUNS — e.g. on a macvlan LAN where guests cannot reach the host,
run the keeper on the host with --exec lxc:<name> rather than inside the guest.

Hands: --exec local (default) runs commands on this machine as this user;
--exec lxc:<name> runs them inside an LXD instance as an unprivileged user
(--uid/--gid/--home, default 1000/1000:/home/ubuntu).

This module is deliberately STDLIB-ONLY and free of package-relative imports:
it must run straight from the source tree on a shared mount, on any python3,
without installing hugpy or its dependency stack. Keep it that way.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

# The keeper must run STRAIGHT FROM THE SOURCE TREE (see the module docstring):
# the documented `python3 .../keeper.py` invocation has no package context and
# no installed ecosystem. Prefer the canonical resolver in
# hugpy_platform.central when it is importable (the console script / `hugpy
# keeper` path), and fall back to a stdlib-only inline copy that MIRRORS it
# exactly otherwise. Keep the two in sync.
try:
    from hugpy_platform.central import central_base_url
except ImportError:  # loaded without the ecosystem installed — stay self-contained
    def central_base_url(default="http://127.0.0.1:7002"):
        # Canonical first, then legacy aliases; first non-empty wins. Mirror of
        # hugpy_platform/central.py:central_base_url (the single source of
        # truth when the ecosystem is installed).
        for _name in ("HUGPY_BASE_URL", "HUGPY_CENTRAL", "HUGPY_URL",
                      "WORKER_CENTRAL_URL"):
            _val = os.environ.get(_name)
            if _val:
                return _val.rstrip("/")
        return default.rstrip("/") if default else default

try:
    import readline  # noqa: F401  (line editing + history for input())
except ImportError:
    pass

ACTION_RE = re.compile(r"```(?:action|bash|sh|shell)[^\n]*\n(.*?)```", re.DOTALL)
STATUS_CMD = "uptime; echo; df -h / /srv/share 2>/dev/null; echo; free -h"

# ANSI (the keeper lives in a terminal — dress accordingly)
CYAN, YELLOW, DIM, RED, GREEN, BOLD, R = (
    "\x1b[36m", "\x1b[33m", "\x1b[2m", "\x1b[31m", "\x1b[32m", "\x1b[1m", "\x1b[0m")


# --------------------------------------------------------------------------- #
# hands — where the keeper's commands run
# --------------------------------------------------------------------------- #
class LocalExec:
    def __init__(self):
        self.target = socket.gethostname()
        self.where = f"this machine ({self.target})"

    def argv(self, command):
        return ["bash", "-lc", command]


class LxcExec:
    def __init__(self, name, uid, gid, home):
        self.target = name
        self.where = f"the LXD instance '{name}'"
        self.uid, self.gid, self.home = uid, gid, home

    def argv(self, command):
        return ["lxc", "exec", self.target, "--user", self.uid, "--group", self.gid,
                "--env", f"HOME={self.home}", "--", "bash", "-lc", command]


def make_executor(spec, uid, gid, home):
    if spec == "local":
        return LocalExec()
    if spec.startswith("lxc:"):
        name = spec[4:]
        if not name:
            raise ValueError("--exec lxc:<name> needs an instance name")
        return LxcExec(name, uid, gid, home)
    raise ValueError(f"unknown --exec {spec!r} (use 'local' or 'lxc:<name>')")


def run_action(executor, command, obs_limit):
    try:
        p = subprocess.run(executor.argv(command),
                           capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return 124, "(timed out after 300s)"
    except OSError as e:
        return 127, f"(cannot execute: {e})"
    out = p.stdout or ""
    if p.stderr.strip():
        out += "\n[stderr]\n" + p.stderr
    out = out.strip()
    if len(out) > obs_limit:
        out = out[:obs_limit] + f"\n…[truncated, {len(out)} chars total]"
    return p.returncode, out or "(no output)"


# --------------------------------------------------------------------------- #
# brain — streaming chat against an OpenAI-compatible central
# --------------------------------------------------------------------------- #
def charge(executor, home):
    return (
        f"You are the KEEPER of {executor.where} — its long-lived warden. "
        f"You watch over it and its effects: its health, services, project files "
        f"(under /srv/share/projects if present), and the consequences of changes "
        f"made on it. You have a shell there as an unprivileged user (home {home}). "
        f"Investigate before acting, prefer reversible steps, and report what you "
        f"observe.\n\n"
        f"To run a command, output EXACTLY one fenced block and then stop:\n"
        f"```action\n<one shell command>\n```\n"
        f"Its output comes back as an 'Observation:' and you may then run more "
        f"commands. When you have the final answer, reply normally WITHOUT an "
        f"action block. Keep commands non-interactive."
    )


def stream_chat(central, model, messages, api_key, max_tokens):
    """Yield content tokens from a streaming /v1/chat/completions."""
    req = urllib.request.Request(
        central.rstrip("/") + "/v1/chat/completions",
        data=json.dumps({"model": model, "messages": messages,
                         "max_tokens": max_tokens, "stream": True}).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                p = json.loads(data)
            except ValueError:
                continue
            if p.get("type") == "error" or p.get("error"):
                msg = p.get("message") or p.get("error", {})
                raise RuntimeError(msg if isinstance(msg, str) else json.dumps(msg))
            piece = (p.get("choices") or [{}])[0].get("delta", {}).get("content")
            if piece:
                yield piece


def parse_action(text):
    m = ACTION_RE.findall(text)
    if m:
        return m[-1].strip()
    for line in text.splitlines():
        if line.strip().lower().startswith("action:"):
            return line.split(":", 1)[1].strip()
    return None


def run_agent_turn(args, executor, messages, api_key, obs_limit, *, echo=True):
    """Drive one user turn through the action loop: stream the model, run any
    fenced ``action`` command, feed the observation back, repeat up to
    --max-steps. Mutates ``messages`` in place; returns the final assistant text
    (the reply that carried no action block)."""
    last_text = ""
    for _ in range(args.max_steps):
        text = ""
        for tok in stream_chat(args.central, args.model, messages, api_key, args.max_tokens):
            text += tok
            if echo:
                sys.stdout.write(tok)
                sys.stdout.flush()
        if echo:
            print()
        messages.append({"role": "assistant", "content": text})
        last_text = text
        action = parse_action(text)
        if not action:
            return text
        if echo:
            print(f"{YELLOW}▶ {executor.target}$ {action}{R}")
        rc, out = run_action(executor, action, obs_limit)
        if echo:
            print(f"{DIM}{out}{R}")
        messages.append({"role": "user", "content": f"Observation (exit {rc}):\n{out}"})
    if echo:
        print(f"{RED}[stopped after {args.max_steps} action steps]{R}")
    return last_text


# --------------------------------------------------------------------------- #
# central HTTP (stdlib only) — bridge transcript + keeper replies
# --------------------------------------------------------------------------- #
def _central_json(central, method, path, api_key, payload=None):
    url = central.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8", "replace")
    return json.loads(body) if body.strip() else {}


def bridge_charge(executor, home, channel_hint):
    """System prompt for a keeper relaying a Discord channel. Same warden role,
    but it is conversing through Discord on the operator's behalf."""
    return (
        charge(executor, home)
        + "\n\nYou are ALSO relaying a Discord channel on the operator's behalf. "
        "Inbound 'user' turns are Discord messages; your normal (non-action) reply "
        "is what gets proposed back to that channel. Be concise and stay in your "
        "warden role. Whether a reply reaches Discord directly or waits for operator "
        "approval is decided by the bridge's mode on central — write the reply either "
        "way."
    )


def run_bridge_loop(args, executor, api_key, home):
    """Headless relay: poll a keeper-brained Discord bridge for inbound messages,
    answer each through the action loop, and submit the reply to central, which
    applies the bridge's defer_mode (user-strict holds for approval; keeper-choice
    sends or escalates with 'DEFER:')."""
    bid = args.bridge
    base_path = f"/discord/bridges/{bid}"
    try:
        seen = _central_json(args.central, "GET", f"{base_path}/messages?since=0", api_key)
        since = max((m.get("ts", 0) for m in seen.get("messages", [])), default=0.0)
    except (urllib.error.URLError, ValueError, OSError) as e:
        print(f"{RED}[bridge {bid}: cannot reach central — {e}]{R}", file=sys.stderr)
        return 1

    print(f"{BOLD}{GREEN}⛨ keeper bridge{R} {bid} · mode {BOLD}{args.bridge_mode}{R} "
          f"· acts on {executor.where}\n{DIM}polling Discord every {args.bridge_poll}s "
          f"(Ctrl-C to stop){R}\n")

    messages = [{"role": "system", "content": bridge_charge(executor, home, bid)}]
    while True:
        try:
            time.sleep(args.bridge_poll)
            resp = _central_json(args.central, "GET",
                                 f"{base_path}/messages?since={since}", api_key)
        except KeyboardInterrupt:
            print(f"\n{DIM}(bridge stopped){R}")
            return 0
        except (urllib.error.URLError, ValueError, OSError) as e:
            print(f"{RED}[bridge poll error: {e}]{R}", file=sys.stderr)
            continue
        for m in resp.get("messages", []):
            since = max(since, m.get("ts", 0))
            # only act on inbound Discord turns; ignore our own out/pending msgs
            if m.get("direction") != "in" or m.get("source") != "discord":
                continue
            author = m.get("author") or "discord"
            text_in = m.get("content") or ""
            print(f"{CYAN}← {author}:{R} {text_in}")
            messages.append({"role": "user", "content": f"[{author}] {text_in}"})
            try:
                reply = run_agent_turn(args, executor, messages, api_key,
                                       int(os.environ.get("KEEPER_OBS_LIMIT", "6000")),
                                       echo=False)
            except (urllib.error.URLError, RuntimeError, OSError) as e:
                print(f"{RED}[turn error: {e}]{R}", file=sys.stderr)
                continue
            reply = (reply or "").strip()
            if not reply:
                continue
            try:
                out = _central_json(args.central, "POST", f"{base_path}/keeper-reply",
                                    api_key, {"content": reply})
                print(f"{GREEN}→ ({out.get('action', '?')}){R} {reply}")
            except (urllib.error.URLError, ValueError, OSError) as e:
                print(f"{RED}[reply submit error: {e}]{R}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# the REPL
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="hugpy-keeper", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model id on the central")
    ap.add_argument("--central",
                    default=central_base_url(),
                    help="OpenAI-compatible base URL (env HUGPY_BASE_URL; "
                         "legacy HUGPY_CENTRAL/HUGPY_URL still honoured)")
    ap.add_argument("--exec", dest="executor", default="local",
                    help="where actions run: 'local' or 'lxc:<instance>'")
    ap.add_argument("--uid", default=os.environ.get("STATION_CONSOLE_UID", "1000"))
    ap.add_argument("--gid", default=os.environ.get("STATION_CONSOLE_GID", "1000"))
    ap.add_argument("--home", default=os.environ.get("STATION_CONSOLE_HOME", "/home/ubuntu"))
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-steps", type=int,
                    default=int(os.environ.get("KEEPER_MAX_STEPS", "8")))
    # Discord bridge integration (entwined with the console).
    ap.add_argument("--bridge", default=os.environ.get("KEEPER_BRIDGE"),
                    help="attach to a keeper-brained Discord bridge id and relay "
                         "it headlessly (poll inbound, reply through central)")
    ap.add_argument("--bridge-poll", type=float,
                    default=float(os.environ.get("KEEPER_BRIDGE_POLL", "3")),
                    help="seconds between bridge inbound polls (default 3)")
    ap.add_argument("--bridge-mode", choices=("user-strict", "keeper-choice"),
                    default=os.environ.get("KEEPER_BRIDGE_MODE", "user-strict"),
                    help="informational; central's bridge defer_mode is authoritative "
                         "(user-strict=approve each reply, keeper-choice=keeper decides)")
    ap.add_argument("--mirror", default=os.environ.get("KEEPER_MIRROR"),
                    help="in the terminal REPL, also push replies to this bridge id "
                         "(subject to its mode); /say <text> pushes an ad-hoc message")
    args = ap.parse_args(argv)

    obs_limit = int(os.environ.get("KEEPER_OBS_LIMIT", "6000"))
    api_key = os.environ.get("KEEPER_API_KEY") or os.environ.get("HUGPY_API_KEY", "")
    try:
        executor = make_executor(args.executor, args.uid, args.gid, args.home)
    except ValueError as e:
        print(f"hugpy keeper: {e}", file=sys.stderr)
        return 2
    home = args.home if isinstance(executor, LxcExec) else os.path.expanduser("~")

    # Headless Discord relay — Discord drives the keeper, no terminal prompt.
    if args.bridge:
        return run_bridge_loop(args, executor, api_key, home)

    print(f"{BOLD}{GREEN}⛨ hugpy keeper{R} — keeper of {BOLD}{executor.target}{R}")
    hint = " /say <text>" if args.mirror else ""
    print(f"{DIM}model {args.model} via {args.central} · acts on {executor.where}"
          f" · /status /clear{hint} /quit{R}")
    if args.mirror:
        print(f"{DIM}mirroring replies to Discord bridge {args.mirror} "
              f"({args.bridge_mode}){R}")
    print()

    def _push_bridge(text):
        try:
            out = _central_json(args.central, "POST",
                                f"/discord/bridges/{args.mirror}/keeper-reply",
                                api_key, {"content": text})
            print(f"{GREEN}→ discord ({out.get('action', '?')}){R}")
        except (urllib.error.URLError, ValueError, OSError) as e:
            print(f"{RED}[mirror failed: {e}]{R}", file=sys.stderr)

    messages = [{"role": "system", "content": charge(executor, home)}]
    while True:
        try:
            user = input(f"{BOLD}{CYAN}keeper>{R} ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        cmd = user.strip()
        if not cmd:
            continue
        if cmd in ("/quit", "/exit"):
            break
        if cmd == "/clear":
            messages = messages[:1]
            print(f"{DIM}(conversation cleared){R}")
            continue
        if cmd == "/status":
            rc, out = run_action(executor, STATUS_CMD, obs_limit)
            print(f"{DIM}{out}{R}")
            continue
        if cmd.startswith("/say "):
            if not args.mirror:
                print(f"{RED}[/say needs --mirror <bridge_id>]{R}")
            else:
                _push_bridge(cmd[5:].strip())
            continue

        messages.append({"role": "user", "content": user})
        try:
            reply = run_agent_turn(args, executor, messages, api_key, obs_limit)
            if args.mirror and reply.strip():
                _push_bridge(reply.strip())
        except KeyboardInterrupt:
            print(f"\n{RED}[interrupted]{R}")
        except (urllib.error.URLError, RuntimeError, OSError) as e:
            print(f"{RED}[error: {e}]{R}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
