#!/bin/bash
# station-fix.sh — make an installed hugpy Station (1.0.41 deb, or an rpm
# converted from it) work on ANY Linux desktop: Fedora/RHEL (dnf), Debian/
# Ubuntu (apt), openSUSE (zypper), Arch (pacman), or pip-only. Idempotent;
# safe to re-run. Run as your desktop user (sudo only for /opt and /usr).
#
#   curl -fsSL https://dev.hugpy.ai/api/agent/console/station-fix.sh | bash
#   curl -fsSL https://dev.hugpy.ai/api/agent/console/station-fix.sh | bash -s -- --set-password
#
# Public by the same rule as install.sh: plain client code, no embedded secret.
#
# What it fixes (all seen on fedora42-workstation, 2026-08-20):
#   1. backend died at `import aiohttp` (conda python3 first on PATH with a
#      half-installed ~/.local aiohttp, "No module named 'attr'") -> window
#      opened on ERR_CONNECTION_REFUSED.
#   2. `hugpy-station --version` over ssh SIGSEGV'd (no display; launcher
#      forwarded --version into Electron and hid the output in a log).
#   3. `server.py --set-password` crashed with PermissionError writing
#      /opt/.../.auth (root-owned); the app reads ~/.config/hugpy-station/.auth.
# It installs the 1.0.42 launcher, which answers --version/--help without
# Electron, refuses cleanly with no display, and picks a Python that can
# import aiohttp (and puts it first on PATH so the 1.0.41 main.js, which
# blindly runs `python3`, gets the same interpreter).

set -u
APP=/opt/hugpy-station
BACKEND="$APP/resources/backend"
WANT_PASSWORD=0
[ "${1:-}" = "--set-password" ] && WANT_PASSWORD=1

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   ok: %s\n' "$*"; }
warn() { printf '   !! %s\n' "$*" >&2; }
die()  { printf '\n\033[1;31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

[ -x "$APP/hugpy-station" ] || die "hugpy Station is not installed under $APP (install the rpm/deb first)"
[ -f "$BACKEND/server.py" ] || die "backend missing: $BACKEND/server.py"

# ---------------------------------------------------------------- 1. deps
say "1/5 backend dependencies (aiohttp)"
PKG=""
if   command -v dnf    >/dev/null 2>&1; then PKG="sudo dnf install -y -q python3-aiohttp"
elif command -v apt-get>/dev/null 2>&1; then PKG="sudo apt-get install -y -q python3-aiohttp"
elif command -v zypper >/dev/null 2>&1; then PKG="sudo zypper -n -q install python3-aiohttp"
elif command -v pacman >/dev/null 2>&1; then PKG="sudo pacman -S --noconfirm --needed python-aiohttp"
fi
if /usr/bin/python3 -c 'import aiohttp' >/dev/null 2>&1; then
    ok "system python (/usr/bin/python3) already imports aiohttp"
elif [ -n "$PKG" ] && $PKG >/dev/null 2>&1; then
    ok "system python: aiohttp installed via package manager"
elif /usr/bin/python3 -m pip install --user --upgrade --quiet aiohttp attrs >/dev/null 2>&1; then
    ok "system python: aiohttp+attrs installed via pip --user"
else
    warn "could not give /usr/bin/python3 aiohttp (no package manager match / pip failed) — trying PATH python"
fi
# A conda/pyenv python3 on PATH is what the 1.0.41 app actually runs; give it
# aiohttp too so either interpreter works.
PATH_PY="$(command -v python3 2>/dev/null || true)"
if [ -n "$PATH_PY" ] && [ "$(readlink -f "$PATH_PY")" != "$(readlink -f /usr/bin/python3 2>/dev/null)" ]; then
    if ! "$PATH_PY" -c 'import aiohttp' >/dev/null 2>&1; then
        "$PATH_PY" -m pip install --user --upgrade --quiet aiohttp attrs >/dev/null 2>&1 \
            && ok "PATH python ($PATH_PY): aiohttp+attrs installed (--user)" \
            || warn "pip install into $PATH_PY failed (not fatal if /usr/bin/python3 works)"
    else
        ok "PATH python ($PATH_PY) already imports aiohttp"
    fi
fi

# ------------------------------------------------------ 2. pick interpreter
say "2/5 choose the backend interpreter"
PY=""
for cand in "${HUGPY_STATION_PYTHON:-}" /usr/bin/python3 /usr/local/bin/python3 "$PATH_PY"; do
    [ -n "$cand" ] && [ -x "$cand" ] || continue
    if "$cand" -c 'import aiohttp, attr' >/dev/null 2>&1; then PY="$cand"; break; fi
done
[ -n "$PY" ] || die "no python with aiohttp+attrs. Try: sudo dnf install python3-aiohttp  (or: python3 -m pip install --user aiohttp attrs)"
ok "using $PY ($("$PY" -c 'import aiohttp,sys;print("aiohttp",aiohttp.__version__,"python",sys.version.split()[0])'))"

# ------------------------------------------------------------ 3. launcher
say "3/5 install the 1.0.42 launcher"
TMP="$(mktemp)"
cat >"$TMP" <<'LAUNCHER'
#!/bin/bash
# hugpy Station launcher (1.0.42) — /usr/bin/hugpy-station and the .desktop Exec.
APP_ROOT="/opt/hugpy-station"
BIN="$APP_ROOT/hugpy-station"

station_version() {
    if [ -r "$APP_ROOT/resources/VERSION" ]; then
        tr -d '[:space:]' < "$APP_ROOT/resources/VERSION"
    elif [ -r "$APP_ROOT/resources/app.asar" ]; then
        grep -ao '"version": *"[0-9][0-9A-Za-z.+-]*"' "$APP_ROOT/resources/app.asar" \
            | head -1 | sed -E 's/.*"([^"]+)"$/\1/'
    else
        echo "unknown"
    fi
}

case "${1:-}" in
    --version|-V|version)
        echo "hugpy-station $(station_version)"; exit 0 ;;
    --help|-h|help)
        cat <<EOF
hugpy-station $(station_version) — hugpy Station desktop cockpit

usage: hugpy-station [--version] [--help] [electron/chromium switches...]

Needs a graphical session (\$DISPLAY or \$WAYLAND_DISPLAY). Launch log:
  \${XDG_CACHE_HOME:-~/.cache}/hugpy-station-launch.log
Environment:
  CONSOLE_PORT            backend port (default 8899)
  HUGPY_STATION_PYTHON    interpreter for the backend (must import aiohttp)
EOF
        exit 0 ;;
esac

if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "hugpy-station: no graphical display (\$DISPLAY and \$WAYLAND_DISPLAY are" \
         "unset; session type: ${XDG_SESSION_TYPE:-unknown})." >&2
    echo "  Start it from the desktop session, or over ssh -X, or under xvfb-run." >&2
    echo "  (--version / --help work without a display.)" >&2
    exit 2
fi

pick_python() {
    local cand
    for cand in "${HUGPY_STATION_PYTHON:-}" /usr/bin/python3 /usr/local/bin/python3 \
                "$(command -v python3 2>/dev/null)" "$(command -v python 2>/dev/null)"; do
        [ -n "$cand" ] && [ -x "$cand" ] || continue
        if "$cand" -c 'import aiohttp' >/dev/null 2>&1; then echo "$cand"; return 0; fi
    done
    return 1
}
if ! PY="$(pick_python)"; then
    echo "hugpy-station: no Python interpreter with aiohttp found (tried" \
         "\$HUGPY_STATION_PYTHON, /usr/bin/python3, python3 on PATH)." >&2
    echo "  Fedora: sudo dnf install python3-aiohttp   Debian/Ubuntu: apt install python3-aiohttp" >&2
    echo "  or point HUGPY_STATION_PYTHON at one that has it." >&2
    exit 3
fi
export HUGPY_STATION_PYTHON="$PY"
# 1.0.41's main.js runs a bare `python3`; make that resolve to the probed one.
export PATH="$(dirname "$PY"):$PATH"

LOG_DIR="${XDG_CACHE_HOME:-$HOME/.cache}"
mkdir -p "$LOG_DIR" 2>/dev/null
exec >>"$LOG_DIR/hugpy-station-launch.log" 2>&1
echo "=== launch $(date '+%F %T') === version $(station_version) python $PY"

PORT="${CONSOLE_PORT:-8899}"
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "port $PORT busy — reaping stale hugpy-station instance(s)"
    pkill -f "hugpy-station/hugpy-station" 2>/dev/null
    pkill -f "resources/backend/server.py" 2>/dev/null
    sleep 1
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        fuser -k -9 "$PORT/tcp" 2>/dev/null
        sleep 1
    fi
fi

export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export FLEETVIEW_TERM_CMD="${FLEETVIEW_TERM_CMD:-hugpy-agent mct}"
export KEEPER_DEFAULT_BACKEND="${KEEPER_DEFAULT_BACKEND:-hugpy-agent::claude-code}"

ARGS=(--disable-gpu --disable-gpu-compositing --disable-gpu-sandbox --no-sandbox)
if ! id -nG | grep -qw lxd && getent group lxd | grep -qw "$USER" \
   && command -v sg >/dev/null 2>&1; then
    echo "re-exec via sg lxd (session lacks the lxd group)"
    exec sg lxd -c "\"$BIN\" ${ARGS[*]} \"$@\""
fi
if ! id -nG | grep -qw lxd; then
    echo "note: this session lacks the lxd group — fleet VMs will be invisible" \
         "(the 'self' VM still works). Log out and back in once to fix."
fi

exec "$BIN" "${ARGS[@]}" "$@"
LAUNCHER
sudo install -m 755 -o root -g root "$TMP" "$APP/hugpy-station-launch" && rm -f "$TMP" \
    || die "could not install $APP/hugpy-station-launch"
ok "launcher installed: $APP/hugpy-station-launch"

# /usr/bin/hugpy-station must point at the LAUNCHER, never the raw Electron ELF.
TARGET="$(readlink -f /usr/bin/hugpy-station 2>/dev/null || true)"
if [ "$TARGET" != "$APP/hugpy-station-launch" ]; then
    if command -v update-alternatives >/dev/null 2>&1; then
        sudo update-alternatives --install /usr/bin/hugpy-station hugpy-station "$APP/hugpy-station-launch" 100 >/dev/null 2>&1 \
            || sudo ln -sf "$APP/hugpy-station-launch" /usr/bin/hugpy-station
    else
        sudo ln -sf "$APP/hugpy-station-launch" /usr/bin/hugpy-station
    fi
    ok "/usr/bin/hugpy-station -> launcher (was: ${TARGET:-missing})"
else
    ok "/usr/bin/hugpy-station already points at the launcher"
fi

# chrome-sandbox must be setuid root (deb postinst does this; rpm conversions may not).
if [ "$(stat -c '%a %U' "$APP/chrome-sandbox" 2>/dev/null)" != "4755 root" ]; then
    sudo chown root:root "$APP/chrome-sandbox" && sudo chmod 4755 "$APP/chrome-sandbox" \
        && ok "chrome-sandbox set to 4755 root" || warn "could not fix chrome-sandbox perms"
else
    ok "chrome-sandbox is 4755 root"
fi

# desktop entry: make sure it uses the launcher too
DESK=/usr/share/applications/hugpy-station.desktop
if [ -f "$DESK" ] && ! grep -q "hugpy-station-launch" "$DESK"; then
    sudo sed -i -E "s#^Exec=.*#Exec=\"$APP/hugpy-station-launch\" %U#" "$DESK" && ok "desktop entry now uses the launcher"
fi

# ------------------------------------------------------------ 4. password
say "4/5 console password"
CFG="${XDG_CONFIG_HOME:-$HOME/.config}/hugpy-station"
if [ "$WANT_PASSWORD" = 1 ]; then
    mkdir -p "$CFG" && chmod 700 "$CFG"
    ( cd "$BACKEND" && "$PY" - "$CFG/.auth" <<'PYEOF'
import getpass, os, sys
# Use server.py's own hash_password so the format matches exactly; never run main.
src = open("server.py").read().replace('if __name__ == "__main__":', 'if False:')
ns = {"__name__": "srv", "__file__": os.path.abspath("server.py")}
exec(compile(src, "server.py", "exec"), ns)
# getpass prompts on /dev/tty, so this works even when the script is piped from curl.
pw = getpass.getpass("New console password: ")
if not pw: sys.exit("empty password — aborted")
if pw != getpass.getpass("Confirm password: "): sys.exit("passwords did not match")
p = sys.argv[1]
open(p, "w").write(ns["hash_password"](pw) + "\n"); os.chmod(p, 0o600)
print("saved password hash to", p)
PYEOF
    ) || warn "password not set"
else
    if [ -s "$CFG/.auth" ]; then ok "password already set ($CFG/.auth)"
    else ok "no password (fine for the desktop app: it binds 127.0.0.1). Re-run with --set-password to add one."; fi
fi

# -------------------------------------------------------------- 5. verify
say "5/5 verify"
V="$(/usr/bin/hugpy-station --version 2>&1)" && ok "$V" || warn "launcher --version failed: $V"
( cd "$BACKEND" && HOST=127.0.0.1 CONSOLE_PORT=18899 PORT=18899 timeout 6 "$PY" server.py >/tmp/hugpy-station-backend-check.log 2>&1 )
RC=$?
if [ "$RC" = 124 ]; then ok "backend starts and stays up under $PY (killed by timeout = good)"
else warn "backend exited rc=$RC — last lines:"; tail -15 /tmp/hugpy-station-backend-check.log >&2; fi

echo
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    printf '\033[1mDone. You are in a text session (no display): now log in at the Fedora desktop and start "hugpy Station" from the app menu, or run: hugpy-station\033[0m\n'
else
    printf '\033[1mDone. Start it now:  hugpy-station\033[0m   (log: ~/.cache/hugpy-station-launch.log)\n'
fi
