#!/usr/bin/env bash
# Fresh, non-root Hugpy worker test install for Ubuntu Server.
#
# Run from a trusted checkout/download as root. This deliberately stops only
# the old SYSTEM-SCOPE hugpy-worker.service; it does not delete its files, model
# data, CUDA drivers/toolkit, or the old ComfyUI checkout. The new install is
# independent under /home/hugpy_worker and can be rolled back by re-enabling the
# old unit.
set -Eeuo pipefail
umask 077

CENTRAL="${WORKER_CENTRAL_URL:-https://dev.hugpy.ai}"
WORKER_NAME="${WORKER_NAME:-a-brain}"
WORKER_USER="${WORKER_USER:-hugpy_worker}"
WORKER_PORT="${WORKER_PORT:-9100}"
WORKER_ADVERTISE="${WORKER_ADVERTISE:-}"
SSH_KEY_FILE="${HUGPY_SSH_PUBLIC_KEY_FILE:-}"
SSH_SOURCE_CIDR="${HUGPY_SSH_SOURCE_CIDR:-}"
BOOTSTRAP_FILE="${HUGPY_BOOTSTRAP_FILE:-}"
ROOT_WORKER_UNITS="${ROOT_WORKER_UNITS:-hugpy-worker.service abstract-hugpy-worker.service}"
ROOT_COMFY_UNITS="${ROOT_COMFY_UNITS:-comfyui.service 7103_hugpy_comfy.service}"
DRY_RUN=0
RESUME=0
PIP_RESUME_RETRIES="${HUGPY_PIP_RESUME_RETRIES:-5}"
TOKEN="${WORKER_ENROLL_TOKEN:-}"

usage() {
  cat <<'EOF'
Usage: sudo bash fresh_test_install.sh [options]

Options:
  --central URL       Hugpy central origin (default: https://dev.hugpy.ai)
  --name NAME         fleet name (default: a-brain)
  --user USER         new unprivileged account (default: hugpy_worker)
  --port PORT         worker listen port (default: 9100)
  --advertise URL     address central can dial, e.g. a-brain's WireGuard URL
  --ssh-key-file FILE Hugpy SSH public key; requires --ssh-from CIDR
  --ssh-from CIDR     source CIDR allowed to use that key (ideally Hugpy WireGuard IP)
  --bootstrap-file FILE  use bootstrap downloaded separately as an unprivileged user
  --token TOKEN       worker enrollment token (hpw_…; NOT a Hugpy API key)
  --root-unit UNIT    old root system unit to stop/disable (may be repeated; defaults
                      to hugpy-worker.service and abstract-hugpy-worker.service)
  --root-comfy-unit UNIT  old root ComfyUI system unit to stop/disable (may repeat)
  --resume            continue only the empty account/root directories from an
                      earlier failed run; permits an interrupted venv/pip
                      download when no worker identity, Comfy checkout, or user
                      service exists
  --resume-retries N  pip retry/resume attempts for interrupted wheel downloads
  --dry-run           report intended actions; change nothing
  -h, --help          show this help

The script leaves system CUDA/NVIDIA packages and all old data in place. It
installs a fresh worker + ComfyUI checkout in the new user's home. The ComfyUI
user service is created but not enabled: Hugpy starts/stops it on demand.
EOF
}
die() { printf 'fresh-install: ERROR: %s\n' "$*" >&2; exit 1; }
say() { printf 'fresh-install: %s\n' "$*"; }
do_run() {
  if (( DRY_RUN )); then printf 'DRY-RUN:'; printf ' %q' "$@"; printf '\n';
  else "$@"; fi
}

while (($#)); do
  case "$1" in
    --central) CENTRAL="${2:?missing URL}"; shift 2 ;;
    --name) WORKER_NAME="${2:?missing name}"; shift 2 ;;
    --user) WORKER_USER="${2:?missing user}"; shift 2 ;;
    --port) WORKER_PORT="${2:?missing port}"; shift 2 ;;
    --advertise) WORKER_ADVERTISE="${2:?missing URL}"; shift 2 ;;
    --ssh-key-file) SSH_KEY_FILE="${2:?missing public-key file}"; shift 2 ;;
    --ssh-from) SSH_SOURCE_CIDR="${2:?missing source CIDR}"; shift 2 ;;
    --bootstrap-file) BOOTSTRAP_FILE="${2:?missing bootstrap file}"; shift 2 ;;
    --token) TOKEN="${2:?missing token}"; shift 2 ;;
    --root-unit)
      if [[ "$ROOT_WORKER_UNITS" == "hugpy-worker.service abstract-hugpy-worker.service" ]]; then
        ROOT_WORKER_UNITS="${2:?missing unit}"
      else
        ROOT_WORKER_UNITS+=" ${2:?missing unit}"
      fi
      shift 2 ;;
    --root-comfy-unit)
      if [[ "$ROOT_COMFY_UNITS" == "comfyui.service 7103_hugpy_comfy.service" ]]; then
        ROOT_COMFY_UNITS="${2:?missing unit}"
      else
        ROOT_COMFY_UNITS+=" ${2:?missing unit}"
      fi
      shift 2 ;;
    --resume) RESUME=1; shift ;;
    --resume-retries)
      PIP_RESUME_RETRIES="${2:?missing retry count}"
      [[ "$PIP_RESUME_RETRIES" =~ ^[0-9]+$ ]] || die 'resume retry count must be numeric'
      shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

if (( DRY_RUN == 0 )); then
  [[ $EUID -eq 0 ]] || die 'run with sudo/root; Hugpy itself will run as the new user'
fi
[[ -r /etc/os-release ]] || die 'cannot identify operating system'
# shellcheck disable=SC1091
. /etc/os-release
[[ "${ID:-}" == ubuntu ]] || die "Ubuntu Server required (detected ${PRETTY_NAME:-unknown})"
[[ "$WORKER_PORT" =~ ^[0-9]+$ ]] && ((WORKER_PORT >= 1024 && WORKER_PORT <= 65535)) \
  || die "invalid unprivileged TCP port: $WORKER_PORT"
[[ "$WORKER_USER" =~ ^[a-z_][a-z0-9_-]*[$]?$ ]] || die "invalid Linux username: $WORKER_USER"
if [[ -n "$SSH_KEY_FILE" || -n "$SSH_SOURCE_CIDR" ]]; then
  [[ -n "$SSH_KEY_FILE" && -n "$SSH_SOURCE_CIDR" ]] \
    || die 'SSH requires both --ssh-key-file and --ssh-from'
  [[ -r "$SSH_KEY_FILE" ]] || die "cannot read SSH public key file: $SSH_KEY_FILE"
  [[ "$SSH_SOURCE_CIDR" =~ ^[0-9a-fA-F:.]+(/[0-9]{1,3})?$ ]] \
    || die 'SSH source must be one IPv4/IPv6 address or CIDR (no hostnames/list syntax)'
fi
CENTRAL="${CENTRAL%/}"
CENTRAL="${CENTRAL%/api}"
WORKER_HOME="/home/$WORKER_USER"
INSTALL_ROOT="$WORKER_HOME/hugpy-worker"
STORE_ROOT="$INSTALL_ROOT/storage"
VENV="$INSTALL_ROOT/venv"
COMFY_DIR="$STORE_ROOT/ComfyUI"
EXISTING_HOME="$(getent passwd "$WORKER_USER" | cut -d: -f6 || true)"
UID_NUM="$(id -u "$WORKER_USER" 2>/dev/null || true)"
if [[ -n "$EXISTING_HOME" ]]; then
  (( RESUME )) || die "user $WORKER_USER already exists; pass --resume only for this script's empty partial account"
  [[ "$EXISTING_HOME" == "$WORKER_HOME" ]] || die "existing $WORKER_USER home is $EXISTING_HOME, expected $WORKER_HOME"
  [[ -d "$INSTALL_ROOT" && -d "$STORE_ROOT" ]] || die '--resume refused: expected the empty directories created by the prior attempt'
  [[ ! -f "$COMFY_DIR/main.py" \
     && ! -f "$WORKER_HOME/.hugpy/config/abstract_hugpy_worker.json" \
     && ! -f "$WORKER_HOME/.config/systemd/user/hugpy-worker.service" ]] \
    || die '--resume refused: worker identity, ComfyUI, or user service already exists; inspect it rather than overwriting'
  say "resuming the empty account created by the previous failed attempt: $WORKER_USER"
else
  [[ ! -e "$WORKER_HOME" ]] || die "$WORKER_HOME already exists; refusing to reuse or remove it"
  (( RESUME == 0 )) || die '--resume requested but the account does not exist'
fi

if [[ -z "$TOKEN" && $DRY_RUN -eq 0 ]]; then
  read -r -s -p 'Worker enrollment token (hpw_…; not the Hugpy API key; Enter if not issued): ' TOKEN
  printf '\n'
fi

say "target: $WORKER_NAME; account: $WORKER_USER ($WORKER_HOME); central: $CENTRAL"
say 'CUDA drivers/toolkit and existing root-owned data will not be changed or copied.'
if [[ -z "$TOKEN" && $DRY_RUN -eq 0 ]]; then
  say 'no token supplied; central must allow tokenless enrollment (which may leave the worker pending)'
fi

# Snapshot the old root service state so a failed test install can restore it.
read -r -a ROOT_UNITS <<< "$ROOT_WORKER_UNITS"
read -r -a COMFY_UNITS <<< "$ROOT_COMFY_UNITS"
ALL_ROOT_UNITS=("${ROOT_UNITS[@]}" "${COMFY_UNITS[@]}")
OLD_ACTIVE_UNITS=()
OLD_ENABLED_UNITS=()
if command -v systemctl >/dev/null; then
  for unit in "${ALL_ROOT_UNITS[@]}"; do
    if systemctl cat "$unit" >/dev/null 2>&1; then
      systemctl is-active --quiet "$unit" && OLD_ACTIVE_UNITS+=("$unit") || true
      systemctl is-enabled --quiet "$unit" && OLD_ENABLED_UNITS+=("$unit") || true
    fi
  done
fi
rollback_old_root_unit() {
  rc="${1:-0}"
  if (( rc != 0 && DRY_RUN == 0 )); then
    if [[ -n "$UID_NUM" ]] && id -u "$WORKER_USER" >/dev/null 2>&1; then
      runuser -u "$WORKER_USER" -- env HOME="$WORKER_HOME" XDG_RUNTIME_DIR="/run/user/$UID_NUM" \
        systemctl --user stop hugpy-worker.service >/dev/null 2>&1 || true
    fi
    for unit in "${OLD_ENABLED_UNITS[@]}"; do systemctl enable "$unit" >/dev/null 2>&1 || true; done
    for unit in "${OLD_ACTIVE_UNITS[@]}"; do systemctl start "$unit" >/dev/null 2>&1 || true; done
    if ((${#OLD_ENABLED_UNITS[@]} || ${#OLD_ACTIVE_UNITS[@]})); then
      say 'install failed; restored prior root worker unit state'
    fi
  fi
  exit "$rc"
}
trap rollback_old_root_unit EXIT

if (( DRY_RUN )); then
  say "would stop/disable ${ALL_ROOT_UNITS[*]} if present (old files/data retained)"
  say "would apt install python3/venv, git, curl, compiler/CMake, and ffmpeg prerequisites"
  say "would create $WORKER_USER, add available video/render groups, and enable user linger"
  [[ -n "$SSH_KEY_FILE" ]] && say "would authorize a key only from $SSH_SOURCE_CIDR (no forwarding/TTY)"
  if [[ -n "$BOOTSTRAP_FILE" ]]; then
    say "would use supplied canonical bootstrap $BOOTSTRAP_FILE"
  else
    say 'would fetch canonical installer as root'
  fi
  say "would create fresh venv + store at $INSTALL_ROOT"
  say 'would create comfyui.service and configure Hugpy to manage it on demand'
  exit 0
fi

# Stop the old root worker before enrolling the replacement under the same name.
for unit in "${ALL_ROOT_UNITS[@]}"; do
  if systemctl cat "$unit" >/dev/null 2>&1; then
    if systemctl is-active --quiet "$unit" || systemctl is-enabled --quiet "$unit"; then
      say "stopping/disabling old root unit $unit (files are retained)"
      systemctl disable --now "$unit"
    fi
  fi
done

say 'installing Ubuntu prerequisites (not NVIDIA driver/toolkit packages)'
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates curl git python3 python3-venv python3-pip python3-dev \
  build-essential cmake ninja-build pkg-config ffmpeg libgl1 libglib2.0-0

if ! id -u "$WORKER_USER" >/dev/null 2>&1; then
  say "creating dedicated account $WORKER_USER"
  useradd --create-home --home-dir "$WORKER_HOME" --shell /bin/bash "$WORKER_USER"
fi
UID_NUM="$(id -u "$WORKER_USER")"
for group in video render; do
  if getent group "$group" >/dev/null; then usermod -aG "$group" "$WORKER_USER"; fi
done
install -d -m 0750 -o "$WORKER_USER" -g "$WORKER_USER" "$INSTALL_ROOT" "$STORE_ROOT"
if [[ -n "$SSH_KEY_FILE" ]]; then
  read -r -a SSH_FIELDS < "$SSH_KEY_FILE"
  [[ ${#SSH_FIELDS[@]} -ge 2 ]] || die 'SSH public key file must contain one OpenSSH public key line'
  [[ "${SSH_FIELDS[0]}" =~ ^(ssh-ed25519|ssh-rsa|ecdsa-sha2-[^[:space:]]+|sk-ssh-ed25519@openssh\.com|sk-ecdsa-sha2-[^[:space:]]+)$ ]] \
    || die "unsupported SSH public key type: ${SSH_FIELDS[0]}"
  install -d -m 0700 -o "$WORKER_USER" -g "$WORKER_USER" "$WORKER_HOME/.ssh"
  printf 'from="%s",restrict %s %s %s\n' "$SSH_SOURCE_CIDR" "${SSH_FIELDS[0]}" \
    "${SSH_FIELDS[1]}" "${SSH_FIELDS[*]:2}" \
    > "$WORKER_HOME/.ssh/authorized_keys"
  chown "$WORKER_USER:$WORKER_USER" "$WORKER_HOME/.ssh/authorized_keys"
  chmod 0600 "$WORKER_HOME/.ssh/authorized_keys"
fi
loginctl enable-linger "$WORKER_USER"

if command -v nvidia-smi >/dev/null 2>&1; then
  say 'NVIDIA driver detected:'
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
  if ! command -v nvcc >/dev/null 2>&1; then
    say 'WARNING: nvcc is absent. CUDA engine provisioning may fail; this script intentionally does not install/upgrade drivers or toolkit.'
  fi
else
  say 'WARNING: nvidia-smi not found. This install does not add an NVIDIA driver.'
fi

TMP_BOOTSTRAP="$(mktemp /tmp/hugpy-bootstrap.XXXXXX.sh)"
chmod 0644 "$TMP_BOOTSTRAP"
cleanup() { rm -f "$TMP_BOOTSTRAP"; }
finish() {
  local rc=$?
  cleanup
  rollback_old_root_unit "$rc"
}
trap finish EXIT
# Fetching through sudo can take a different proxy/auth path than the operator's
# shell. Accept an operator-downloaded copy to preserve that known-good path.
if [[ -n "$BOOTSTRAP_FILE" ]]; then
  [[ -r "$BOOTSTRAP_FILE" ]] || die "cannot read supplied bootstrap: $BOOTSTRAP_FILE"
  cp -- "$BOOTSTRAP_FILE" "$TMP_BOOTSTRAP"
else
  # -q disables root's ~/.curlrc; if sudo still sees a 403, use --bootstrap-file.
  curl -q --fail --silent --show-error --location "$CENTRAL/api/llm/workers/install.sh" -o "$TMP_BOOTSTRAP"
fi
grep -q 'hugpy worker bootstrap' "$TMP_BOOTSTRAP" || die 'downloaded bootstrap did not look like the Hugpy installer'
# Older deployed bootstraps only accepted --token (visible briefly in argv).
# Upgrade that one assignment in our private temp copy so this wrapper can keep
# the enrollment secret in the environment; fail closed if the payload drifted.
if grep -Fxq 'TOKEN="${WORKER_ENROLL_TOKEN:-}"' "$TMP_BOOTSTRAP"; then
  :
elif grep -Fxq 'TOKEN=""' "$TMP_BOOTSTRAP"; then
  sed -i 's/^TOKEN=""$/TOKEN="${WORKER_ENROLL_TOKEN:-}"/' "$TMP_BOOTSTRAP"
else
  die 'canonical bootstrap token handoff changed; inspect it before running this test installer'
fi

BOOT_ARGS=(--central "$CENTRAL" --name "$WORKER_NAME" --port "$WORKER_PORT"
  --storage-root "$STORE_ROOT" --venv "$VENV" --install-comfy)
[[ -n "$WORKER_ADVERTISE" ]] && BOOT_ARGS+=(--advertise "$WORKER_ADVERTISE")
say "running canonical installer as $WORKER_USER (token is not printed)"
export HOME="$WORKER_HOME" USER="$WORKER_USER" LOGNAME="$WORKER_USER"
export PATH='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export XDG_RUNTIME_DIR="/run/user/$UID_NUM" WORKER_ENROLL_TOKEN="$TOKEN" \
  HUGPY_PIP_RESUME_RETRIES="$PIP_RESUME_RETRIES"
runuser --preserve-environment -u "$WORKER_USER" -- bash "$TMP_BOOTSTRAP" "${BOOT_ARGS[@]}"
unset WORKER_ENROLL_TOKEN TOKEN

[[ -f "$COMFY_DIR/main.py" ]] || die "ComfyUI install missing main.py at $COMFY_DIR"
COMFY_UNIT_DIR="$WORKER_HOME/.config/systemd/user"
install -d -m 0750 -o "$WORKER_USER" -g "$WORKER_USER" "$COMFY_UNIT_DIR" \
  "$COMFY_UNIT_DIR/hugpy-worker.service.d"
COMFY_UNIT_TMP="$(mktemp)"
cat >"$COMFY_UNIT_TMP" <<EOF
[Unit]
Description=ComfyUI for Hugpy worker $WORKER_NAME
After=network.target

[Service]
Type=simple
WorkingDirectory=$COMFY_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$VENV/bin/python $COMFY_DIR/main.py --listen 127.0.0.1 --port 8188
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF
install -m 0640 -o "$WORKER_USER" -g "$WORKER_USER" "$COMFY_UNIT_TMP" "$COMFY_UNIT_DIR/comfyui.service"
rm -f "$COMFY_UNIT_TMP"

# Install.py's default Comfy launcher is spawn:. Override only the launcher URL
# so the agent controls this user-scope unit and starts it on demand.
DROPIN_TMP="$(mktemp)"
cat >"$DROPIN_TMP" <<EOF
[Service]
Environment="HUGPY_COMFY_LAUNCH=systemd-user:comfyui.service"
Environment="HUGPY_COMFY_URL=http://127.0.0.1:8188"
Environment="COMFY_URL=http://127.0.0.1:8188"
Environment="COMFY_CHECKPOINTS_DIR=$STORE_ROOT/models/checkpoints"
Environment="COMFY_CHECKPOINT_DIRS=$STORE_ROOT/models/checkpoints"
EOF
install -m 0640 -o "$WORKER_USER" -g "$WORKER_USER" "$DROPIN_TMP" \
  "$COMFY_UNIT_DIR/hugpy-worker.service.d/comfy.conf"
rm -f "$DROPIN_TMP"

runuser -u "$WORKER_USER" -- env HOME="$WORKER_HOME" USER="$WORKER_USER" \
  XDG_RUNTIME_DIR="/run/user/$UID_NUM" systemctl --user daemon-reload
runuser -u "$WORKER_USER" -- env HOME="$WORKER_HOME" USER="$WORKER_USER" \
  XDG_RUNTIME_DIR="/run/user/$UID_NUM" systemctl --user restart hugpy-worker.service
sleep 2
NEW_PID="$(runuser -u "$WORKER_USER" -- env HOME="$WORKER_HOME" USER="$WORKER_USER" \
  XDG_RUNTIME_DIR="/run/user/$UID_NUM" systemctl --user show hugpy-worker.service -p MainPID --value)"
[[ "$NEW_PID" =~ ^[1-9][0-9]*$ ]] || die 'user worker service has no live MainPID'
NEW_UID="$(awk '/^Uid:/ {print $2}' "/proc/$NEW_PID/status")"
[[ "$NEW_UID" == "$UID_NUM" ]] || die "worker pid $NEW_PID runs as uid $NEW_UID, expected $UID_NUM"
WORKER_ID_FILE="$WORKER_HOME/.hugpy/config/abstract_hugpy_worker.json"
REGISTERED_ID=""
for _ in {1..30}; do
  if [[ -f "$WORKER_ID_FILE" ]]; then
    REGISTERED_ID="$(python3 - "$WORKER_ID_FILE" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        row = json.load(f)
    print(row.get("id") or row.get("worker_id") or "")
except Exception:
    print("")
PY
)"
    [[ -n "$REGISTERED_ID" ]] && break
  fi
  sleep 1
done
[[ -n "$REGISTERED_ID" ]] || die 'worker process is non-root but did not register; check its journal and central enrollment policy/token'
for unit in "${ALL_ROOT_UNITS[@]}"; do
  systemctl is-active --quiet "$unit" && die "old root unit unexpectedly active: $unit"
done

say "PASS: Hugpy worker PID $NEW_PID runs as $WORKER_USER (uid $NEW_UID)"
say "PASS: worker registered with central (id $REGISTERED_ID)"
say "PASS: old root worker units are inactive; their files/data remain for rollback"
say "ComfyUI unit created for $WORKER_USER; it is intentionally not enabled (Hugpy starts it on demand)."
say "Verify: runuser -u $WORKER_USER -- env HOME=$WORKER_HOME XDG_RUNTIME_DIR=/run/user/$UID_NUM systemctl --user status hugpy-worker comfyui"
if [[ -n "$SSH_KEY_FILE" ]]; then
  say "PASS: SSH key restricted to $SSH_SOURCE_CIDR; forwarding and interactive TTY disabled"
else
  say 'SSH unchanged; pass both --ssh-key-file and --ssh-from to authorize a restricted Hugpy key.'
fi
trap - EXIT
cleanup
