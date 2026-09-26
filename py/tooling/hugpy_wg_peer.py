#!/usr/bin/env python3
"""hugpy_wg_peer — the ONE privileged WireGuard-peer primitive on the hub (ae).

Why this exists
---------------
Joining a remote GPU box to the hugpy fleet needs exactly one privileged action
on the hub: register the box as a WireGuard peer on ``wg0`` (live + persisted).
Everything else the central join flow does — allocate the next free 10.66.0.x,
generate the keypair, mint the enrollment token, assemble the join bundle — is
unprivileged and runs as the ``hugpy`` user. So the ONLY thing that needs root
is this, and it is deliberately tiny and capability-locked: the operator drops a
sudoers rule letting ``hugpy`` run *only this script* as root, and this script
can do *only* add/remove/list a wg0 peer. No shell, no download, no other verb.

    hugpy_wg_peer.py add    <name> <pubkey> <ip>     # live `wg set` + persist
    hugpy_wg_peer.py remove <name|pubkey>            # live remove + de-persist
    hugpy_wg_peer.py list                            # named peer blocks (JSON)

Design invariants (all enforced here, none assumed of the caller):
  * Strict input validation. ``name`` is ``[a-z0-9][a-z0-9_-]{0,30}``, ``pubkey``
    is a 44-char base64 curve25519 key that decodes to 32 bytes, ``ip`` is an
    IPv4 inside 10.66.0.0/24 that is neither the network/broadcast address nor
    the hub (.1). A value that does not match is REFUSED before any command runs.
  * No shell injection is possible: every external command is a fixed argv list
    (``subprocess.run([...])``), never a string, and every interpolated token has
    already passed a whitelist regex above.
  * Idempotent. ``add`` of a name/ip/pubkey already registered to the same triple
    is a no-op success; re-adding a name rewrites that name's block in place.
  * ``add`` fails closed if the requested ip is already in use by a DIFFERENT
    peer (live table or persisted conf), so central's allocator can never collide
    two workers onto one address even under a race.

State: ``wg set wg0 ...`` for the live device, plus a named, comment-delimited
[Peer] block in ``/etc/wireguard/wg0.conf`` so the peer survives ``wg-quick down``
/reboot. The block markers are ``# >>> hugpy-peer <name>`` / ``# <<< hugpy-peer
<name>`` — this script only ever touches text between a matching pair, so a
hand-written [Peer] the operator added stays untouched.

This is a stdlib-only standalone script (NOT part of a pip package): the operator
installs it to /usr/local/sbin — see hugpy_wg_peer.install.sh.
"""
from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Optional

WG_IFACE = os.environ.get("HUGPY_WG_IFACE", "wg0")
WG_CONF = os.environ.get("HUGPY_WG_CONF", "/etc/wireguard/wg0.conf")
WG_SUBNET = ipaddress.ip_network("10.66.0.0/24")
HUB_IP = ipaddress.ip_address("10.66.0.1")
# The wg binary. Overridable (tests point it at a stub; a distro may ship it in
# /usr/bin or /bin). Never a shell string — always argv[0] of a fixed argv.
WG_BIN = os.environ.get("HUGPY_WG_BIN", "/usr/bin/wg")

# --- strict input whitelists ------------------------------------------------ #
# A peer name is a single safe token: it names a conf block and is echoed back in
# list output; it never touches a shell, but keeping it to this alphabet means it
# can never break the block-marker grammar or a conf line either.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,30}$")
# A WireGuard public key is standard base64 of 32 raw bytes: 43 base64 chars + a
# single '=' pad. We check the shape AND that it decodes to exactly 32 bytes.
_PUBKEY_RE = re.compile(r"^[A-Za-z0-9+/]{43}=$")

_MARK_BEGIN = "# >>> hugpy-peer %s"
_MARK_END = "# <<< hugpy-peer %s"
_MARK_ANY = re.compile(r"^# >>> hugpy-peer (\S+)\s*$")


class PeerError(Exception):
    """A validation or state error with an operator-readable message."""


# --------------------------------------------------------------------------- #
# validation                                                                  #
# --------------------------------------------------------------------------- #
def validate_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise PeerError(
            "invalid peer name %r: must match %s (lowercase letters/digits, "
            "then letters/digits/_/-, max 31 chars)" % (name, _NAME_RE.pattern))
    return name


def validate_pubkey(pubkey: str) -> str:
    if not isinstance(pubkey, str) or not _PUBKEY_RE.match(pubkey):
        raise PeerError(
            "invalid WireGuard public key %r: expected 44-char base64 "
            "(43 chars + '=')" % (pubkey,))
    try:
        raw = base64.b64decode(pubkey, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise PeerError("invalid WireGuard public key %r: not base64 (%s)"
                        % (pubkey, exc))
    if len(raw) != 32:
        raise PeerError(
            "invalid WireGuard public key %r: decodes to %d bytes, not 32"
            % (pubkey, len(raw)))
    return pubkey


def validate_ip(ip: str) -> str:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError as exc:
        raise PeerError("invalid ip %r: %s" % (ip, exc))
    if addr not in WG_SUBNET:
        raise PeerError("ip %s is not inside the hub subnet %s" % (addr, WG_SUBNET))
    if addr in (WG_SUBNET.network_address, WG_SUBNET.broadcast_address):
        raise PeerError("ip %s is the network/broadcast address of %s"
                        % (addr, WG_SUBNET))
    if addr == HUB_IP:
        raise PeerError("ip %s is the hub's own address (reserved)" % (addr,))
    return str(addr)


# --------------------------------------------------------------------------- #
# reading state                                                               #
# --------------------------------------------------------------------------- #
def _run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a fixed argv (never a shell string). Raises PeerError on failure with
    the tool's real stderr — no canned text."""
    try:
        cp = subprocess.run(argv, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise PeerError("command not found: %s (%s)" % (argv[0], exc))
    if check and cp.returncode != 0:
        raise PeerError("`%s` failed (exit %d): %s"
                        % (" ".join(argv), cp.returncode,
                           (cp.stderr or cp.stdout or "").strip()))
    return cp


def _live_allowed_ips() -> dict[str, str]:
    """{pubkey: first-allowed-ip-without-mask} from the running device. Empty if
    the device is down (wg errors) — the persisted conf is then authoritative."""
    out: dict[str, str] = {}
    cp = _run([WG_BIN, "show", WG_IFACE, "allowed-ips"], check=False)
    if cp.returncode != 0:
        return out
    for line in cp.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        pub = parts[0]
        for cidr in parts[1:]:
            ipp = cidr.split("/")[0]
            try:
                if ipaddress.ip_address(ipp) in WG_SUBNET:
                    out[pub] = ipp
                    break
            except ValueError:
                continue
    return out


def _hub_info() -> dict:
    """The hub interface's own public key + listen port (for the client conf).
    Empty values if the device is down — the caller then falls back to its
    configured hub pubkey/endpoint."""
    info: dict = {"pubkey": None, "listen_port": None, "iface": WG_IFACE}
    cp = _run([WG_BIN, "show", WG_IFACE, "public-key"], check=False)
    if cp.returncode == 0 and cp.stdout.strip():
        info["pubkey"] = cp.stdout.strip()
    cp = _run([WG_BIN, "show", WG_IFACE, "listen-port"], check=False)
    if cp.returncode == 0 and cp.stdout.strip():
        try:
            info["listen_port"] = int(cp.stdout.strip())
        except ValueError:
            pass
    return info


def _read_conf() -> str:
    try:
        with open(WG_CONF, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""


def _all_conf_allowed_ips(text: str) -> set[str]:
    """Every subnet address in EVERY ``AllowedIPs =`` line of the conf — named
    hugpy-peer blocks AND hand-written [Peer] sections alike. This is what makes
    allocation safe against peers this tool did not create (the pre-existing
    .2/.3/.10/... peers)."""
    out: set[str] = set()
    for raw in text.splitlines():
        s = raw.strip()
        if not s.lower().startswith("allowedips"):
            continue
        val = s.split("=", 1)[1] if "=" in s else ""
        for token in val.replace(",", " ").split():
            ipp = token.split("/")[0].strip()
            try:
                if ipaddress.ip_address(ipp) in WG_SUBNET:
                    out.add(ipp)
            except ValueError:
                continue
    return out


def used_ips() -> set[str]:
    """Every 10.66.0.x currently claimed: the hub itself, every live-device
    allowed-ip, and every AllowedIPs in the persisted conf."""
    out: set[str] = {str(HUB_IP)}
    out.update(_live_allowed_ips().values())
    out.update(_all_conf_allowed_ips(_read_conf()))
    return out


def _parse_conf_blocks(text: str) -> "list[dict]":
    """Every named hugpy-peer block in the conf, in order:
    ``[{"name","pubkey","ip","start","end"}]`` (start/end are line indices,
    end exclusive)."""
    lines = text.splitlines()
    blocks: list[dict] = []
    i = 0
    while i < len(lines):
        m = _MARK_ANY.match(lines[i])
        if not m:
            i += 1
            continue
        name = m.group(1)
        start = i
        end = None
        pubkey = None
        ip = None
        j = i + 1
        end_marker = _MARK_END % name
        while j < len(lines):
            if lines[j].strip() == end_marker.strip():
                end = j + 1
                break
            s = lines[j].strip()
            if s.lower().startswith("publickey"):
                pubkey = s.split("=", 1)[1].strip() if "=" in s else None
            elif s.lower().startswith("allowedips"):
                val = s.split("=", 1)[1].strip() if "=" in s else ""
                ip = val.split("/")[0].strip() or None
            j += 1
        if end is None:
            end = len(lines)
        blocks.append({"name": name, "pubkey": pubkey, "ip": ip,
                       "start": start, "end": end})
        i = end
    return blocks


def _ip_in_use(ip: str, *, exclude_name: Optional[str] = None,
               exclude_pubkey: Optional[str] = None) -> Optional[str]:
    """Return an owner description if ``ip`` is already used by a DIFFERENT peer,
    else None. Checks named hugpy blocks, HAND-WRITTEN [Peer] sections, and the
    live device — so allocation is safe against peers this tool did not create."""
    text = _read_conf()
    blocks = _parse_conf_blocks(text)
    # The excluded peer's own currently-persisted address (idempotent re-add of a
    # name to the same ip must not report itself as a collision).
    excluded_block_ips = {b["ip"] for b in blocks
                          if exclude_name and b["name"] == exclude_name and b["ip"]}
    # 1) a DIFFERENT named block already holds this ip.
    for blk in blocks:
        if blk["ip"] == ip:
            if exclude_name and blk["name"] == exclude_name:
                continue
            if exclude_pubkey and blk["pubkey"] == exclude_pubkey:
                continue
            return "conf block '%s' (pubkey %s)" % (blk["name"], blk["pubkey"])
    # 2) a hand-written [Peer] (not one of ours) holds this ip. Any conf
    #    AllowedIPs match that is not the excluded name's own address counts.
    if ip in _all_conf_allowed_ips(text) and ip not in excluded_block_ips:
        return "an existing [Peer] block in %s" % (WG_CONF,)
    # 3) the live device holds this ip for a different peer.
    for pub, live_ip in _live_allowed_ips().items():
        if live_ip == ip:
            if exclude_pubkey and pub == exclude_pubkey:
                continue
            return "live wg0 peer %s" % (pub,)
    return None


# --------------------------------------------------------------------------- #
# writing state                                                               #
# --------------------------------------------------------------------------- #
def _atomic_write_conf(new_text: str) -> None:
    """Replace WG_CONF atomically, preserving mode 600 root."""
    d = os.path.dirname(WG_CONF) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".wg0.conf.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(new_text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, WG_CONF)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _render_block(name: str, pubkey: str, ip: str) -> str:
    return "\n".join([
        _MARK_BEGIN % name,
        "[Peer]",
        "# hugpy-peer: %s" % name,
        "PublicKey = %s" % pubkey,
        "AllowedIPs = %s/32" % ip,
        _MARK_END % name,
    ])


def _conf_without_block(text: str, name: str) -> str:
    """Return ``text`` with the named block removed (blank-line tidy)."""
    lines = text.splitlines()
    keep: list[str] = []
    for blk in _parse_conf_blocks(text):
        if blk["name"] == name:
            # Drop [start, end); also drop one trailing blank line if present.
            end = blk["end"]
            if end < len(lines) and lines[end].strip() == "":
                blk["_drop_extra"] = end
    drops: set[int] = set()
    for blk in _parse_conf_blocks(text):
        if blk["name"] == name:
            for k in range(blk["start"], blk["end"]):
                drops.add(k)
            nxt = blk["end"]
            if nxt < len(lines) and lines[nxt].strip() == "":
                drops.add(nxt)
    for idx, ln in enumerate(lines):
        if idx not in drops:
            keep.append(ln)
    out = "\n".join(keep)
    return out


def _persist_add(name: str, pubkey: str, ip: str) -> None:
    text = _read_conf()
    # Rewrite in place: strip any existing block of this name, then append fresh.
    text = _conf_without_block(text, name)
    block = _render_block(name, pubkey, ip)
    if text and not text.endswith("\n"):
        text += "\n"
    if text and not text.endswith("\n\n"):
        text += "\n"
    text += block + "\n"
    _atomic_write_conf(text)


# --------------------------------------------------------------------------- #
# verbs                                                                        #
# --------------------------------------------------------------------------- #
def cmd_add(name: str, pubkey: str, ip: str) -> dict:
    name = validate_name(name)
    pubkey = validate_pubkey(pubkey)
    ip = validate_ip(ip)

    # Idempotency + collision: if this ip is held by a different peer, refuse.
    owner = _ip_in_use(ip, exclude_name=name, exclude_pubkey=pubkey)
    if owner is not None:
        raise PeerError("ip %s is already in use by %s" % (ip, owner))

    # Live device: set (or update) the peer's single /32. `wg set` is idempotent.
    _run([WG_BIN, "set", WG_IFACE, "peer", pubkey, "allowed-ips", "%s/32" % ip])
    # Persist to the conf so it survives wg-quick down / reboot.
    _persist_add(name, pubkey, ip)
    return {"ok": True, "action": "add", "name": name, "pubkey": pubkey,
            "ip": ip, "iface": WG_IFACE, "conf": WG_CONF,
            "hub": _hub_info()}


def cmd_used_ips() -> dict:
    return {"ok": True, "iface": WG_IFACE, "subnet": str(WG_SUBNET),
            "used": sorted(used_ips(), key=lambda s: ipaddress.ip_address(s))}


def cmd_hub_info() -> dict:
    return {"ok": True, **_hub_info()}


def cmd_remove(selector: str) -> dict:
    """Remove by name or by public key."""
    blocks = _parse_conf_blocks(_read_conf())
    target = None
    if _PUBKEY_RE.match(selector or ""):
        pubkey = validate_pubkey(selector)
        for blk in blocks:
            if blk["pubkey"] == pubkey:
                target = blk
                break
        # Even with no conf block, remove the live peer by key.
        _run([WG_BIN, "set", WG_IFACE, "peer", pubkey, "remove"], check=False)
        if target is not None:
            _atomic_write_conf(_conf_without_block(_read_conf(), target["name"]))
        return {"ok": True, "action": "remove", "selector": selector,
                "removed_pubkey": pubkey,
                "removed_conf_block": target["name"] if target else None}
    # else selector is a name
    name = validate_name(selector)
    for blk in blocks:
        if blk["name"] == name:
            target = blk
            break
    if target is None:
        raise PeerError("no hugpy-peer conf block named %r in %s"
                        % (name, WG_CONF))
    if target["pubkey"] and _PUBKEY_RE.match(target["pubkey"]):
        _run([WG_BIN, "set", WG_IFACE, "peer", target["pubkey"], "remove"],
             check=False)
    _atomic_write_conf(_conf_without_block(_read_conf(), name))
    return {"ok": True, "action": "remove", "name": name,
            "removed_pubkey": target["pubkey"]}


def cmd_list() -> dict:
    live = _live_allowed_ips()
    blocks = _parse_conf_blocks(_read_conf())
    peers = []
    for blk in blocks:
        peers.append({
            "name": blk["name"],
            "pubkey": blk["pubkey"],
            "ip": blk["ip"],
            "live": blk["pubkey"] in live if blk["pubkey"] else False,
        })
    return {"ok": True, "iface": WG_IFACE, "conf": WG_CONF, "peers": peers}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="hugpy_wg_peer",
        description="Privileged wg0 peer primitive for the hugpy hub (add/remove/list).")
    sub = p.add_subparsers(dest="verb", required=True)
    a = sub.add_parser("add", help="register a peer (live + persisted)")
    a.add_argument("name")
    a.add_argument("pubkey")
    a.add_argument("ip")
    r = sub.add_parser("remove", help="remove a peer by name or public key")
    r.add_argument("selector")
    sub.add_parser("list", help="list named hugpy-peer blocks (JSON)")
    sub.add_parser("used-ips", help="every claimed 10.66.0.x (for allocation)")
    sub.add_parser("hub-info", help="the hub's own wg0 public key + listen port")
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    try:
        if args.verb == "add":
            result = cmd_add(args.name, args.pubkey, args.ip)
        elif args.verb == "remove":
            result = cmd_remove(args.selector)
        elif args.verb == "list":
            result = cmd_list()
        elif args.verb == "used-ips":
            result = cmd_used_ips()
        elif args.verb == "hub-info":
            result = cmd_hub_info()
        else:  # pragma: no cover - argparse guarantees a verb
            p.error("unknown verb")
            return 2
    except PeerError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
