# Joining a remote worker over WireGuard (one step)

A GPU box that is not on the hub's LAN joins the hugpy fleet over a WireGuard
tunnel to the hub (ae, `wg0` = `10.66.0.1/24`, ListenPort `51820`, public endpoint
`23.126.105.155:51820`). The join is one generated script the operator runs on
the box. This document is the whole flow: the operator's one-time hub setup, the
per-worker join, and revocation.

## Pieces

| Piece | Path | Runs as | Role |
|---|---|---|---|
| wg-peer helper | `py/tooling/hugpy_wg_peer.py` → `/usr/local/sbin/hugpy-wg-peer` | root (via sudo) | the ONE privileged action: add/remove/list a `wg0` peer, live + persisted, with strict input validation |
| helper installer | `py/tooling/hugpy_wg_peer.install.sh` | root | installs the helper + a hugpy-only sudoers drop-in |
| join brain | `hugpy_fleet/central/wg_join.py` | `hugpy` | allocates the next free `10.66.0.x`, generates the keypair, calls the helper, mints an enrollment token, assembles the JOIN BUNDLE |
| CLI | `hugpy-fleet join-code <name>` (`hugpy_fleet/cli.py`) | `hugpy` | prints the self-contained join script to stdout |
| route | `POST /llm/fleet/wg-join` (operator-only) | server | the same bundle as JSON, for the console |
| worker join body | `hugpy_fleet/worker/wg_join.sh` | root on the worker | installs wireguard-tools, brings the tunnel up, opens the worker port, preflights central, then runs the worker installer as the selected login account |

The peer's **private key** is generated on the hub, placed only in the returned
bundle / rendered script, and is never logged and never stored. The enrollment
token is single-use and revocable, and is shown once.

## Operator one-time hub setup

Two steps, done once on the hub (ae). Both are privileged, so the **operator**
runs them — copy the installer to the box and run it (pasted heredocs break in
some terminals):

```bash
sudo bash /tmp/hugpy_wg_peer.install.sh        # from py/tooling/hugpy_wg_peer.install.sh
```

That installs `/usr/local/sbin/hugpy-wg-peer` and a sudoers drop-in letting only
the `hugpy` user run that one command as root (`HUGPY_USER=<user>` overrides the
allowed user).

Then allow WireGuard workers to reach central (UFW currently allows `7002` only
from the LAN):

```bash
sudo ufw allow in on wg0 to any port 7002 proto tcp
```

If this rule is missing, a worker's join preflight fails with an explicit message
naming exactly this command.

## Per-worker join

On the hub, as the `hugpy` user:

```bash
hugpy-fleet join-code a-brain > a-brain.join.sh
```

This allocates the next free `10.66.0.x` (reading `wg show wg0 allowed-ips` plus
the persisted conf, so it never collides with an existing peer), registers the
peer, mints a token, and writes a self-contained script. Copy that one file to the
box and run it as root:

```bash
sudo bash a-brain.join.sh
```

The generated script (Fedora `dnf`/firewalld and Ubuntu `apt`/ufw both
supported) is idempotent and safe to re-run. Run it with `sudo`; it uses
`SUDO_USER` as the worker account, or accepts `--worker-user USER` when that
account differs. This matters because root is needed for WireGuard/firewall,
while the Hugpy venv, `~/.hugpy` worker identity, and systemd user service must
belong to the actual worker account. It enables systemd linger for that account
so its worker service survives SSH logout and reboot.

The script:

1. installs `wireguard-tools` if missing;
2. writes `/etc/wireguard/hugpy.conf` (mode 600, root) — on a root-squashed home
   (a-brain's `/home/alpha`) the conf lives under `/etc/wireguard` so `wg-quick`
   reads it there, never the home dir;
3. `wg-quick up hugpy` and `systemctl enable wg-quick@hugpy`;
4. opens the worker port `9100` to the hub only (firewalld rich rule from
   `10.66.0.1`, or `ufw allow in on hugpy to any port 9100`);
5. preflights `curl http://10.66.0.1:7002/api/health` — on failure it names the
   hub-side UFW fix above;
6. runs the existing worker bootstrap as the selected account over the tunnel with
   `--central http://10.66.0.1:7002 --name a-brain --token <token>` and
   `WORKER_URL=http://10.66.0.x:9100` (so central calls the worker back on the
   tunnel; this is baked into the worker's unit for restart durability).

The installer report performs both connection checks: worker → central
(`/api/health`) and, after enrollment, central → worker (central's
`/api/llm/workers/<id>/health` probe). It also inspects/opens the worker's host
firewall and reports the measured URL/error plus the next concrete repair for
failed checks. If the central host firewall blocks the initial tunnel request,
the join stops before installation and prints the hub-side rule above; the
central must permit the WireGuard interface to reach API port 7002 before
re-running the script.

The same generated script is returned by Toolserver's operator-only
`POST /api/llm/fleet/wg-join` endpoint (the console's fleet join flow). Run the
returned `join_script` on the worker as above. The toolserver endpoint allocates
the WireGuard peer and single-use token; the worker script performs the actual
worker install and reports whether central can call back.

The tunnel is **hub-only** (`AllowedIPs = 10.66.0.1/32`), never a full tunnel, so
the box's own routing/Internet is untouched.

## Revocation

Remove a peer (drops it from the live device and the persisted conf), and revoke
its enrollment token:

```bash
hugpy-fleet revoke a-brain                      # or: hugpy-fleet revoke <pubkey>
# revoke the token in the console, or:
curl -X DELETE -H "X-Operator-Token: $HUGPY_OPERATOR_TOKEN" \
  http://10.66.0.1:7002/api/llm/enroll-tokens/<token_id>
```

`hugpy-fleet list` shows the registered peers (name, ip, whether live).

## Auth

`POST /llm/fleet/wg-join` is operator-only (`operator_auth._SENSITIVE`): an
operator session (`is_admin`), a `HUGPY_OPERATOR_TOKEN` via `X-Operator-Token` /
`Authorization: Bearer`, or a valid `hp_` API key. It is strictly more powerful
than `/llm/enroll-tokens` (it also grants subnet reachability), so it is never
reachable anonymously and never waived.
