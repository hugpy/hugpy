# ===========================================================================
# hugpy fleet — worker-side WireGuard join body.
#
# This is the reusable body. The `hugpy-fleet join-code <name>` command prepends
# a generated header that exports the per-worker values below and then includes
# this file, producing ONE self-contained script the operator runs on the box.
# It is ALSO runnable standalone if those values are supplied as env vars or
# flags. Idempotent and safe to re-run. Every step prints what it did.
#
# Inputs (env, with --flag overrides):
#   HUGPY_WG_IFACE       wg interface name on the worker (default: hugpy)
#   HUGPY_WG_CONF_B64    base64 of the client wg conf (written to /etc/wireguard)
#   HUGPY_CENTRAL        central base URL over the tunnel (http://10.66.0.1:7002)
#   HUGPY_ADVERTISE      URL central should call back on (http://10.66.0.x:9100)
#   HUGPY_WORKER_NAME    worker name
#   HUGPY_ENROLL_TOKEN   single-use enrollment token
#
# Works on Fedora (dnf, firewalld) and Ubuntu/Debian (apt, ufw).
# ===========================================================================

HUGPY_WG_IFACE="${HUGPY_WG_IFACE:-hugpy}"
HUGPY_WG_CONF_B64="${HUGPY_WG_CONF_B64:-}"
HUGPY_CENTRAL="${HUGPY_CENTRAL:-http://10.66.0.1:7002}"
HUGPY_ADVERTISE="${HUGPY_ADVERTISE:-}"
HUGPY_WORKER_NAME="${HUGPY_WORKER_NAME:-$(hostname)}"
HUGPY_ENROLL_TOKEN="${HUGPY_ENROLL_TOKEN:-}"
HUGPY_WORKER_PORT="${HUGPY_WORKER_PORT:-9100}"
# The script itself must run as root to configure WireGuard/firewall. The
# Python worker must run as the normal login account so its venv, ~/.hugpy
# identity, and systemd user unit are not accidentally created under /root.
HUGPY_WORKER_USER="${HUGPY_WORKER_USER:-${SUDO_USER:-$(id -un)}}"
HUB_VPN_IP="10.66.0.1"

# --flag overrides (env is the default) -------------------------------------
while [ "$#" -gt 0 ]; do
  case "$1" in
    --iface) HUGPY_WG_IFACE="${2:-}"; shift 2 ;;
    --central) HUGPY_CENTRAL="${2:-}"; shift 2 ;;
    --advertise) HUGPY_ADVERTISE="${2:-}"; shift 2 ;;
    --name) HUGPY_WORKER_NAME="${2:-}"; shift 2 ;;
    --worker-user) HUGPY_WORKER_USER="${2:-}"; shift 2 ;;
    --token) HUGPY_ENROLL_TOKEN="${2:-}"; shift 2 ;;
    --conf-b64) HUGPY_WG_CONF_B64="${2:-}"; shift 2 ;;
    -h|--help) echo "usage: join.sh [--central URL] [--name N] [--token T] [--advertise URL] [--iface NAME] [--worker-user USER]"; exit 0 ;;
    *) echo "join: unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { echo "join: $*"; }
die() { echo "join: ERROR: $*" >&2; exit 1; }

HUGPY_CENTRAL="${HUGPY_CENTRAL%/}"
HUGPY_CENTRAL="${HUGPY_CENTRAL%/api}"
[ -n "$HUGPY_ADVERTISE" ] || HUGPY_ADVERTISE="http://${HUGPY_WORKER_NAME}:${HUGPY_WORKER_PORT}"

[ "$(id -u)" -eq 0 ] || die "run as root (sudo bash $0)"

CONF="/etc/wireguard/${HUGPY_WG_IFACE}.conf"

# --- 1. wireguard-tools -----------------------------------------------------
if command -v wg >/dev/null 2>&1 && command -v wg-quick >/dev/null 2>&1; then
  say "wireguard-tools already present ($(command -v wg))"
else
  say "installing wireguard-tools"
  if command -v dnf >/dev/null 2>&1; then
    dnf install -y wireguard-tools || die "dnf install wireguard-tools failed"
  elif command -v yum >/dev/null 2>&1; then
    yum install -y wireguard-tools || die "yum install wireguard-tools failed"
  elif command -v apt-get >/dev/null 2>&1; then
    apt-get update -y && apt-get install -y wireguard-tools \
      || die "apt-get install wireguard-tools failed"
  else
    die "no supported package manager (dnf/yum/apt-get) to install wireguard-tools"
  fi
  say "installed $(command -v wg)"
fi

# --- 2. write the client conf (0600 root) -----------------------------------
# a-brain caveat: /home is root-squashed, so the conf MUST live under
# /etc/wireguard and wg-quick MUST read it from there (never the home dir).
mkdir -p /etc/wireguard
chmod 700 /etc/wireguard || true
if [ -n "$HUGPY_WG_CONF_B64" ]; then
  NEW="$(mktemp)"; trap 'rm -f "$NEW"' EXIT
  printf '%s' "$HUGPY_WG_CONF_B64" | base64 -d > "$NEW" || die "conf base64 decode failed"
  if [ -f "$CONF" ] && cmp -s "$NEW" "$CONF"; then
    say "conf $CONF already up to date"
  else
    install -m 0600 -o root -g root "$NEW" "$CONF"
    say "wrote $CONF (mode 600 root)"
  fi
else
  [ -f "$CONF" ] || die "no HUGPY_WG_CONF_B64 given and $CONF does not exist"
  say "using existing $CONF"
fi

# --- 3. bring the tunnel up (idempotent) + enable on boot -------------------
if wg show "$HUGPY_WG_IFACE" >/dev/null 2>&1; then
  say "reloading $HUGPY_WG_IFACE (already up)"
  wg-quick down "$CONF" >/dev/null 2>&1 || wg-quick down "$HUGPY_WG_IFACE" >/dev/null 2>&1 || true
fi
wg-quick up "$CONF" || die "wg-quick up failed for $CONF"
say "tunnel $HUGPY_WG_IFACE up"
if command -v systemctl >/dev/null 2>&1; then
  systemctl enable "wg-quick@${HUGPY_WG_IFACE}" >/dev/null 2>&1 \
    && say "enabled wg-quick@${HUGPY_WG_IFACE} on boot" \
    || say "note: could not enable wg-quick@${HUGPY_WG_IFACE} (enable it by hand to persist across reboot)"
fi

# --- 4. open the worker port 9100 to the hub only ---------------------------
if command -v firewall-cmd >/dev/null 2>&1 && firewall-cmd --state >/dev/null 2>&1; then
  firewall-cmd --permanent \
    --add-rich-rule="rule family=ipv4 source address=${HUB_VPN_IP}/32 port port=${HUGPY_WORKER_PORT} protocol=tcp accept" >/dev/null 2>&1 || true
  firewall-cmd --reload >/dev/null 2>&1 || true
  say "firewalld: allowed ${HUB_VPN_IP} -> ${HUGPY_WORKER_PORT}/tcp"
elif command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -qi active; then
  ufw allow in on "$HUGPY_WG_IFACE" to any port "$HUGPY_WORKER_PORT" proto tcp >/dev/null 2>&1 || true
  say "ufw: allowed in on ${HUGPY_WG_IFACE} to ${HUGPY_WORKER_PORT}/tcp"
else
  say "no active firewalld/ufw detected — assuming ${HUGPY_WORKER_PORT}/tcp is reachable from ${HUB_VPN_IP}"
fi

# --- 5. preflight: can we reach central over the tunnel? --------------------
say "preflight: GET ${HUGPY_CENTRAL}/api/health"
if curl -fsS --max-time 8 "${HUGPY_CENTRAL}/api/health" >/dev/null 2>&1; then
  say "central reachable over the tunnel"
else
  echo "join: ERROR: cannot reach ${HUGPY_CENTRAL}/api/health over the tunnel." >&2
  echo "join:   The WireGuard handshake may be fine but central's host firewall" >&2
  echo "join:   likely does not yet allow this tunnel to port 7002." >&2
  echo "join:   HUB (ae) operator one-time fix:" >&2
  echo "join:       sudo ufw allow in on wg0 to any port 7002 proto tcp" >&2
  echo "join:   Then re-run this script." >&2
  exit 1
fi

# --- 6. hand off to the existing worker bootstrap over the tunnel -----------
# WORKER_URL makes the agent advertise its wg address so central calls back on
# the tunnel. Fetching the bootstrap via the wg central URL bakes that same URL
# as the worker's --central (workers_install_sh rewrites it to the request origin).
export WORKER_URL="$HUGPY_ADVERTISE"
say "handing off to the worker installer (central=${HUGPY_CENTRAL}, advertise=${HUGPY_ADVERTISE})"
BOOT_URL="${HUGPY_CENTRAL}/api/llm/workers/install.sh"
WORKER_HOME="$(getent passwd "$HUGPY_WORKER_USER" | cut -d: -f6)"
[ -n "$WORKER_HOME" ] || die "cannot resolve worker account '$HUGPY_WORKER_USER' (use --worker-user USER)"
WORKER_UID="$(id -u "$HUGPY_WORKER_USER")"
if [ "$HUGPY_WORKER_USER" = root ]; then
  say "WARNING: installing worker as root; pass --worker-user <login> for a normal user service"
fi
if command -v loginctl >/dev/null 2>&1; then
  loginctl enable-linger "$HUGPY_WORKER_USER" >/dev/null 2>&1 \
    && say "enabled systemd linger for ${HUGPY_WORKER_USER}" \
    || say "WARNING: could not enable systemd linger for ${HUGPY_WORKER_USER}; installer will report service persistence status"
fi
run_worker_bootstrap() {
  if [ "$HUGPY_WORKER_USER" = root ]; then
    env HOME="$WORKER_HOME" USER=root LOGNAME=root XDG_RUNTIME_DIR="/run/user/${WORKER_UID}" \
      DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${WORKER_UID}/bus" WORKER_URL="$HUGPY_ADVERTISE" bash -s -- "$@"
  elif command -v runuser >/dev/null 2>&1; then
    runuser -u "$HUGPY_WORKER_USER" -- env HOME="$WORKER_HOME" USER="$HUGPY_WORKER_USER" \
      LOGNAME="$HUGPY_WORKER_USER" XDG_RUNTIME_DIR="/run/user/${WORKER_UID}" \
      DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${WORKER_UID}/bus" \
      WORKER_URL="$HUGPY_ADVERTISE" bash -s -- "$@"
  else
    die "runuser is required to install as '$HUGPY_WORKER_USER'"
  fi
}
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$BOOT_URL" | run_worker_bootstrap \
    --central "$HUGPY_CENTRAL" --name "$HUGPY_WORKER_NAME" --token "$HUGPY_ENROLL_TOKEN" \
    || die "worker bootstrap failed (fetched from ${BOOT_URL})"
elif command -v wget >/dev/null 2>&1; then
  wget -qO- "$BOOT_URL" | run_worker_bootstrap \
    --central "$HUGPY_CENTRAL" --name "$HUGPY_WORKER_NAME" --token "$HUGPY_ENROLL_TOKEN" \
    || die "worker bootstrap failed (fetched from ${BOOT_URL})"
else
  die "need curl or wget to fetch the worker bootstrap"
fi

say "DONE: ${HUGPY_WORKER_NAME} joined over WireGuard and the worker installer ran."
say "  tunnel:    ${HUGPY_WG_IFACE} (conf ${CONF})"
say "  advertise: ${HUGPY_ADVERTISE}"
say "  account:   ${HUGPY_WORKER_USER} (home ${WORKER_HOME})"
say "  the worker install report above checks worker -> central and central -> worker;"
say "  resolve any FAIL/WARN it prints before relying on this worker."
