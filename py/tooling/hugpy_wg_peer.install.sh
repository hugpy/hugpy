#!/usr/bin/env bash
# ===========================================================================
# Operator one-time install of the privileged WireGuard-peer helper on the HUB.
#
# Run AS ROOT on the hub (ae). Pasted heredocs break in some terminals, so copy
# this file to the box and run it:
#
#     sudo bash /tmp/hugpy_wg_peer.install.sh
#
# What it does (idempotent):
#   1. installs the helper to /usr/local/sbin/hugpy-wg-peer (root:root, 0755);
#   2. writes a sudoers drop-in letting ONLY the hugpy user run THAT ONE command
#      as root without a password (nothing else);
#   3. validates the sudoers file before installing it (visudo -c).
#
# The helper itself does strict input validation and can only add/remove/list a
# wg0 peer — it is the entire privileged surface of the join flow.
#
# Overridable:
#   HUGPY_USER   the unprivileged user allowed to run the helper (default: hugpy)
#   SRC          source path of the helper .py (default: the in-tree file)
#   DEST         install path (default: /usr/local/sbin/hugpy-wg-peer)
# ===========================================================================
set -euo pipefail

HUGPY_USER="${HUGPY_USER:-hugpy}"
SRC="${SRC:-/srv/hugpy/src/hugpy/py/tooling/hugpy_wg_peer.py}"
DEST="${DEST:-/usr/local/sbin/hugpy-wg-peer}"
SUDOERS="/etc/sudoers.d/hugpy-wg-peer"

[ "$(id -u)" -eq 0 ] || { echo "run as root: sudo bash $0" >&2; exit 1; }
[ -r "$SRC" ] || { echo "source helper not readable: $SRC" >&2; exit 1; }
id "$HUGPY_USER" >/dev/null 2>&1 || { echo "user '$HUGPY_USER' does not exist" >&2; exit 1; }
command -v wg >/dev/null 2>&1 || echo "WARNING: 'wg' not found on PATH — install wireguard-tools on the hub."

echo "== installing helper: $SRC -> $DEST =="
install -m 0755 -o root -g root "$SRC" "$DEST"
# Ensure a python3 shebang works: the file already begins with #!/usr/bin/env python3.
echo "  installed $DEST"

echo "== writing sudoers drop-in: $SUDOERS =="
TMP="$(mktemp)"
cat > "$TMP" <<EOF
# hugpy WireGuard peer helper — the ONLY privileged capability the fleet join
# flow needs. Allows '$HUGPY_USER' to run exactly this one command as root with
# no password. The command itself validates all input and can only add/remove/
# list a wg0 peer.
$HUGPY_USER ALL=(root) NOPASSWD: $DEST
EOF
chmod 0440 "$TMP"
if visudo -c -f "$TMP" >/dev/null; then
  install -m 0440 -o root -g root "$TMP" "$SUDOERS"
  rm -f "$TMP"
  echo "  installed $SUDOERS (validated with visudo -c)"
else
  rm -f "$TMP"
  echo "sudoers content failed visudo validation — not installed" >&2
  exit 1
fi

echo "== verifying =="
echo "  $HUGPY_USER can now run:  sudo -n $DEST list"
echo
echo "REMEMBER the other hub-side one-time step (WireGuard workers reach central):"
echo "  sudo ufw allow in on wg0 to any port 7002 proto tcp"
echo
echo "Done. Generate a worker join with:  sudo -u $HUGPY_USER hugpy-fleet join-code <name>"
