"""One-step WireGuard join for a remote GPU worker (the hub side).

The operator ask (2026-09-24): *"an ease of wireguard connect to distribute to
that worker"* — join a remote box (tonight: a-brain) to the hugpy fleet over
WireGuard in one step.

This module is the unprivileged brain of that flow. It runs as the ``hugpy`` user
on the hub (ae) — from the ``hugpy-fleet`` CLI or the operator-only central route
— and does everything that does NOT need root:

  1. ask the privileged helper (``hugpy_wg_peer used-ips``) which 10.66.0.x are
     taken, and allocate the lowest free one;
  2. generate the peer's curve25519 keypair (``wg genkey|pubkey`` when the ``wg``
     binary is available to this user, else a pure-python X25519 fallback);
  3. call the privileged helper (``sudo hugpy_wg_peer add <name> <pub> <ip>``) to
     register the peer live + persisted on wg0, which also returns the hub's own
     public key + listen port for the client conf;
  4. mint a single-use, short-lived enrollment token (the existing store);
  5. assemble a JOIN BUNDLE: the client WG conf (hub-only AllowedIPs, NOT a full
     tunnel), the central URL, the worker's advertise URL, the token, the name —
     and render it as a self-contained join script the operator runs on the box.

The peer's PRIVATE KEY is generated here, placed ONLY in the returned bundle /
rendered script, and is NEVER logged and NEVER stored on the hub. The enrollment
token is likewise shown once. The one privileged step is delegated to a tiny,
capability-locked helper (see ``py/tooling/hugpy_wg_peer.py``) that the operator
installs to /usr/local/sbin with a hugpy-only sudoers rule.
"""
from __future__ import annotations

import base64
import ipaddress
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# --- hub facts (see MEMORY hugpy-hub-wan-wireguard) ------------------------- #
WG_SUBNET = ipaddress.ip_network("10.66.0.0/24")
HUB_VPN_IP = "10.66.0.1"
# The PUBLIC endpoint remote peers dial. NEVER the outbound 98.34.243.181, NEVER
# port 443. Overridable for a hub whose inbound IP moves.
DEFAULT_ENDPOINT = os.environ.get("HUGPY_WG_ENDPOINT", "23.126.105.155:51820")
# Fallback hub public key if the live device can't be read (matches the value in
# the existing abrain-wg-reconnect.sh). The helper's live `wg show` value wins.
FALLBACK_HUB_PUBKEY = os.environ.get(
    "HUGPY_WG_HUB_PUBKEY", "AeQwCil79b+boyHce+i3hGdB1fpJoYwd8AvDVIf6iSQ=")
CENTRAL_WG_URL = os.environ.get("HUGPY_WG_CENTRAL_URL", "http://10.66.0.1:7002")
WORKER_PORT = int(os.environ.get("HUGPY_WG_WORKER_PORT", "9100"))
KEEPALIVE = 25
# The worker's WireGuard interface name (drives the conf filename on the box).
WORKER_IFACE = os.environ.get("HUGPY_WG_WORKER_IFACE", "hugpy")

# Enrollment token TTL is advisory here (the store has no TTL of its own): the
# join bundle carries the mint time so the operator sees the freshness; the token
# is single-machine and revocable, which is the real bound.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,30}$")


class JoinError(Exception):
    """A join-flow failure with an operator-readable message (real tool text)."""


def validate_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise JoinError(
            "invalid worker name %r: must match %s" % (name, _NAME_RE.pattern))
    return name


# --------------------------------------------------------------------------- #
# the privileged helper                                                       #
# --------------------------------------------------------------------------- #
def _helper_argv() -> list[str]:
    """How to invoke the privileged wg0 helper. Default: ``sudo -n
    /usr/local/sbin/hugpy-wg-peer`` (the operator's install path + sudoers rule).
    ``HUGPY_WG_PEER_HELPER`` overrides the whole argv (space-split), e.g. a test
    points it at ``python3 /path/to/hugpy_wg_peer.py`` with no sudo."""
    override = (os.environ.get("HUGPY_WG_PEER_HELPER") or "").strip()
    if override:
        return shlex.split(override)
    return ["sudo", "-n", "/usr/local/sbin/hugpy-wg-peer"]


def _run_helper(*verb_and_args: str) -> dict:
    argv = _helper_argv() + list(verb_and_args)
    try:
        cp = subprocess.run(argv, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise JoinError("wg helper not found (%s); the operator must install it "
                        "to /usr/local/sbin with the hugpy sudoers rule "
                        "(see hugpy_wg_peer.install.sh): %s"
                        % (argv[0], exc))
    out = (cp.stdout or "").strip()
    if cp.returncode != 0 and not out:
        raise JoinError("wg helper `%s` failed (exit %d): %s"
                        % (" ".join(verb_and_args), cp.returncode,
                           (cp.stderr or "").strip() or "no output"))
    try:
        data = json.loads(out)
    except ValueError:
        raise JoinError("wg helper `%s` returned non-JSON (exit %d): %s"
                        % (" ".join(verb_and_args), cp.returncode, out or
                           (cp.stderr or "").strip()))
    if not data.get("ok"):
        raise JoinError("wg helper `%s`: %s"
                        % (" ".join(verb_and_args),
                           data.get("error") or "unknown error"))
    return data


# --------------------------------------------------------------------------- #
# IP allocation                                                               #
# --------------------------------------------------------------------------- #
def allocate_ip(used: "set[str] | list[str]") -> str:
    """The lowest free host address in 10.66.0.0/24 (skipping .0/.1/.255).

    ``used`` is the authoritative claimed set from the helper (hub + live device
    + persisted conf). Raises when the subnet is exhausted."""
    used_addrs = set()
    for u in used:
        try:
            used_addrs.add(ipaddress.ip_address(u))
        except ValueError:
            continue
    used_addrs.add(ipaddress.ip_address(HUB_VPN_IP))
    for host in WG_SUBNET.hosts():          # .1 .. .254
        if host == ipaddress.ip_address(HUB_VPN_IP):
            continue
        if host not in used_addrs:
            return str(host)
    raise JoinError("no free address left in %s (all %d hosts in use)"
                    % (WG_SUBNET, WG_SUBNET.num_addresses - 2))


def next_free_ip() -> str:
    data = _run_helper("used-ips")
    return allocate_ip(data.get("used") or [])


# --------------------------------------------------------------------------- #
# keypair                                                                     #
# --------------------------------------------------------------------------- #
def _wg_binary() -> Optional[str]:
    return shutil.which("wg")


def _gen_keypair_wg(wg: str) -> "tuple[str, str]":
    priv = subprocess.run([wg, "genkey"], capture_output=True, text=True)
    if priv.returncode != 0 or not priv.stdout.strip():
        raise JoinError("`wg genkey` failed: %s"
                        % ((priv.stderr or "").strip() or "no output"))
    private_key = priv.stdout.strip()
    pub = subprocess.run([wg, "pubkey"], input=private_key + "\n",
                         capture_output=True, text=True)
    if pub.returncode != 0 or not pub.stdout.strip():
        raise JoinError("`wg pubkey` failed: %s"
                        % ((pub.stderr or "").strip() or "no output"))
    return private_key, pub.stdout.strip()


# -- pure-python X25519 (RFC 7748) — used only when the wg binary is absent -- #
_P = 2 ** 255 - 19
_A24 = 121665


def _decode_scalar(k: bytes) -> int:
    a = bytearray(k)
    a[0] &= 248
    a[31] &= 127
    a[31] |= 64
    return int.from_bytes(a, "little")


def _x25519(scalar: bytes, u_int: int) -> bytes:
    k = _decode_scalar(scalar)
    x1 = u_int
    x2, z2, x3, z3 = 1, 0, x1, 1
    swap = 0
    for t in reversed(range(255)):
        kt = (k >> t) & 1
        swap ^= kt
        if swap:
            x2, x3 = x3, x2
            z2, z3 = z3, z2
        swap = kt
        A = (x2 + z2) % _P
        AA = (A * A) % _P
        B = (x2 - z2) % _P
        BB = (B * B) % _P
        E = (AA - BB) % _P
        C = (x3 + z3) % _P
        D = (x3 - z3) % _P
        DA = (D * A) % _P
        CB = (C * B) % _P
        x3 = pow((DA + CB) % _P, 2, _P)
        z3 = (x1 * pow((DA - CB) % _P, 2, _P)) % _P
        x2 = (AA * BB) % _P
        z2 = (E * ((AA + (_A24 * E) % _P) % _P)) % _P
    if swap:
        x2, x3 = x3, x2
        z2, z3 = z3, z2
    return ((x2 * pow(z2, _P - 2, _P)) % _P).to_bytes(32, "little")


def _gen_keypair_python() -> "tuple[str, str]":
    priv = bytearray(secrets.token_bytes(32))
    priv[0] &= 248
    priv[31] &= 127
    priv[31] |= 64
    pub = _x25519(bytes(priv), 9)
    return (base64.b64encode(bytes(priv)).decode(),
            base64.b64encode(pub).decode())


def gen_keypair() -> "tuple[str, str]":
    """(private_key, public_key) as base64 strings. Uses ``wg`` when present,
    else a pure-python X25519 keypair (so the hugpy user never needs the wg
    binary just to mint a client key)."""
    wg = _wg_binary()
    if wg:
        try:
            return _gen_keypair_wg(wg)
        except JoinError:
            logger.warning("wg keygen failed; falling back to pure-python X25519")
    return _gen_keypair_python()


# --------------------------------------------------------------------------- #
# rendering                                                                   #
# --------------------------------------------------------------------------- #
def render_client_conf(bundle: dict) -> str:
    """The worker's ``/etc/wireguard/<iface>.conf``. AllowedIPs is the HUB ONLY
    (10.66.0.1/32) — this is a fleet link, never a full tunnel."""
    wg = bundle["wg"]
    return (
        "[Interface]\n"
        f"# hugpy worker '{bundle['name']}' -> hub\n"
        f"PrivateKey = {wg['private_key']}\n"
        f"Address = {wg['address']}\n"
        "\n"
        "[Peer]\n"
        "# hugpy hub (ae)\n"
        f"PublicKey = {wg['hub_pubkey']}\n"
        f"Endpoint = {wg['endpoint']}\n"
        f"AllowedIPs = {wg['allowed_ips']}\n"
        f"PersistentKeepalive = {wg['persistent_keepalive']}\n"
    )


def _worker_join_body() -> str:
    """The packaged worker-side join script body (bash). Read from the resource
    so the self-contained script and the standalone script are the SAME code."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "worker", "wg_join.sh")
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def render_join_script(bundle: dict) -> str:
    """A self-contained bash script the operator runs on the worker box. It
    embeds the client conf (base64) + the join parameters, then runs the packaged
    join body. The private key lives ONLY inside this script text."""
    wg = bundle["wg"]
    conf_b64 = base64.b64encode(bundle["client_conf"].encode()).decode()
    header = "\n".join([
        "#!/usr/bin/env bash",
        "# hugpy fleet WireGuard join — generated for worker '%s' (%s)"
        % (bundle["name"], bundle["ip"]),
        "# Run on the worker box:  sudo bash %s.join.sh" % bundle["name"],
        "# Contains a PRIVATE KEY and a single-use enrollment token — do not share.",
        "set -eu",
        "export HUGPY_WG_IFACE=%s" % shlex.quote(wg["iface"]),
        "export HUGPY_WG_CONF_B64=%s" % shlex.quote(conf_b64),
        "export HUGPY_CENTRAL=%s" % shlex.quote(bundle["central_url"]),
        "export HUGPY_ADVERTISE=%s" % shlex.quote(bundle["advertise_url"]),
        "export HUGPY_WORKER_NAME=%s" % shlex.quote(bundle["name"]),
        "export HUGPY_ENROLL_TOKEN=%s" % shlex.quote(bundle["token"]),
        "",
        "",
    ])
    return header + _worker_join_body()


# --------------------------------------------------------------------------- #
# the bundle                                                                  #
# --------------------------------------------------------------------------- #
def build_bundle(name: str, *, label: str = "") -> dict:
    """Do the whole hub-side join for ``name`` and return the JOIN BUNDLE.

    Side effects: registers a wg0 peer (via the privileged helper) and mints one
    enrollment token. The private key and token are in the return value ONLY —
    never logged, never persisted here."""
    name = validate_name(name)
    logger.info("wg-join: allocating address + peer for worker %r", name)

    ip = next_free_ip()
    private_key, public_key = gen_keypair()

    added = _run_helper("add", name, public_key, ip)
    hub = added.get("hub") or {}
    hub_pubkey = hub.get("pubkey") or FALLBACK_HUB_PUBKEY
    listen_port = hub.get("listen_port")
    endpoint = DEFAULT_ENDPOINT
    if listen_port and ":" in DEFAULT_ENDPOINT:
        # Keep the configured host, honor the live listen port.
        host = DEFAULT_ENDPOINT.rsplit(":", 1)[0]
        endpoint = f"{host}:{listen_port}"

    # Mint the enrollment token AFTER the peer is up, so a helper failure never
    # leaves an orphan token. Import lazily so this module is importable without
    # the central state configured (tests, CLI --help).
    from hugpy_fleet.central.enrollment_tokens import create_enrollment_token
    tok = create_enrollment_token(label=label or f"wg-join:{name}")
    token = tok["token"]  # shown once

    advertise_url = f"http://{ip}:{WORKER_PORT}"
    bundle = {
        "name": name,
        "ip": ip,
        "central_url": CENTRAL_WG_URL,
        "advertise_url": advertise_url,
        "token": token,
        "token_id": tok.get("id"),
        "wg": {
            "private_key": private_key,     # ONCE — never logged/stored
            "public_key": public_key,
            "address": f"{ip}/32",
            "hub_pubkey": hub_pubkey,
            "endpoint": endpoint,
            "allowed_ips": f"{HUB_VPN_IP}/32",
            "persistent_keepalive": KEEPALIVE,
            "iface": WORKER_IFACE,
        },
    }
    bundle["client_conf"] = render_client_conf(bundle)
    bundle["join_script"] = render_join_script(bundle)
    logger.info("wg-join: worker %r allocated %s (token %s); bundle assembled "
                "(private key NOT logged)", name, ip, tok.get("id"))
    return bundle


def redacted_bundle(bundle: dict) -> dict:
    """A log/response-safe view: the private key and token plaintext removed.
    Use this anywhere the bundle might be logged."""
    safe = json.loads(json.dumps(bundle))  # deep copy
    if isinstance(safe.get("wg"), dict):
        safe["wg"].pop("private_key", None)
    safe.pop("token", None)
    # The rendered artifacts embed both secrets — drop them from the safe view.
    safe.pop("client_conf", None)
    safe.pop("join_script", None)
    return safe


# --------------------------------------------------------------------------- #
# CLI entry (also reachable via `hugpy-fleet join-code` — see cli.py)          #
# --------------------------------------------------------------------------- #
def build_parser(prog: str = "hugpy-fleet-join"):
    import argparse
    p = argparse.ArgumentParser(
        prog=prog,
        description="Generate a one-step WireGuard join for a remote hugpy worker.")
    sub = p.add_subparsers(dest="verb", required=True)
    jc = sub.add_parser("join-code",
                        help="allocate + register a peer and print the self-contained "
                             "join script to run on the worker box")
    jc.add_argument("name")
    jc.add_argument("--json", action="store_true",
                    help="print the full JSON bundle instead of the script")
    lst = sub.add_parser("list", help="list registered wg peers")
    rev = sub.add_parser("revoke", help="remove a wg peer by name or public key")
    rev.add_argument("selector")
    return p


def main(argv: "list[str] | None" = None, *, prog: str = "hugpy-fleet-join") -> int:
    import sys
    p = build_parser(prog)
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    try:
        if args.verb == "join-code":
            bundle = build_bundle(args.name)
            if args.json:
                print(json.dumps(bundle, indent=2))
            else:
                sys.stderr.write(
                    "# worker %s -> %s ; token %s\n"
                    "# save the script below on the box and run: sudo bash <file>\n"
                    % (bundle["name"], bundle["ip"], bundle.get("token_id")))
                sys.stdout.write(bundle["join_script"])
            return 0
        if args.verb == "list":
            print(json.dumps(_run_helper("list"), indent=2))
            return 0
        if args.verb == "revoke":
            print(json.dumps(_run_helper("remove", args.selector), indent=2))
            return 0
    except JoinError as exc:
        sys.stderr.write("join error: %s\n" % exc)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
