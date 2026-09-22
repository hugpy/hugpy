#!/usr/bin/env bash
# hugpy Station installer — detect the distro's package manager, fetch the
# matching artifact from the hugpy central, verify its sha256, install it.
# Served publicly at GET /agent/console/install.sh (no secret embedded — same
# rule as /agent/client.sh); the member/operator credential is read from the
# CALLER's environment at run time:
#
#   curl -fsSL https://dev.hugpy.ai/api/agent/console/install.sh | HUGPY_TOKEN=<token> bash
#
# Since 2026-08-21 the station is built by electron-builder
# (station-app/build-release.sh) and ships FOUR artifacts, so this script picks
# one instead of assuming .deb:
#
#   apt / apt-get      -> .deb        apt-get install ./x.deb
#   dnf / yum          -> .rpm        dnf install ./x.rpm
#   zypper             -> .rpm        zypper install --allow-unsigned-rpm ./x.rpm
#   pacman             -> .pacman     pacman -U ./x.pkg.tar.<zst|xz|gz>
#   none of the above  -> .AppImage   ~/.local/bin + a .desktop entry, no root
#
# Env:
#   HUGPY_TOKEN or HUGPY_OPERATOR_TOKEN   credential (required) — sent as a bearer
#   HUGPY_CENTRAL                         API base (default https://dev.hugpy.ai/api)
#   HUGPY_STATION_FORMAT                  force one of deb|rpm|pacman|appimage
#
# The download is verified against the sha256 sidecar surfaced by
# /agent/console/info. Native package installs need root or sudo; the AppImage
# path is per-user and needs neither.
set -euo pipefail

CENTRAL="${HUGPY_CENTRAL:-https://dev.hugpy.ai/api}"
CENTRAL="${CENTRAL%/}"
TOKEN="${HUGPY_TOKEN:-${HUGPY_OPERATOR_TOKEN:-}}"
[ -n "$TOKEN" ] || {
  echo "error: no credential. Set HUGPY_TOKEN (or HUGPY_OPERATOR_TOKEN) — the" >&2
  echo "       console artifacts are member-gated, e.g.:" >&2
  echo "       curl -fsSL $CENTRAL/agent/console/install.sh | HUGPY_TOKEN=xxx bash" >&2
  exit 1
}
command -v curl >/dev/null || { echo "error: curl is required" >&2; exit 1; }

# ── which format does this machine want? ─────────────────────────────────────
have() { command -v "$1" >/dev/null 2>&1; }
if [ -n "${HUGPY_STATION_FORMAT:-}" ]; then
  FORMAT="$HUGPY_STATION_FORMAT"
elif have apt-get || have apt; then FORMAT=deb
elif have dnf || have yum;    then FORMAT=rpm
elif have zypper;             then FORMAT=rpm
elif have pacman;             then FORMAT=pacman
else                               FORMAT=appimage
fi
echo "hugpy Station installer — target format: $FORMAT"

auth=(-H "Authorization: Bearer $TOKEN")
info="$(curl -fsSL "${auth[@]}" "$CENTRAL/agent/console/info")" || {
  echo "error: $CENTRAL/agent/console/info unreachable or credential rejected" >&2
  exit 1
}

# Pull {artifacts:{<kind>:{filename,sha256}}} out of the info payload. python3
# when present (every distro we target ships it); a sed pass otherwise, which is
# why the four filename shapes are spelled out in the pattern.
pick() {  # pick <kind> <field>  -> value or empty
  local kind="$1" field="$2"
  if have python3; then
    printf '%s' "$info" | python3 -c '
import json, sys
kind, field = sys.argv[1], sys.argv[2]
d = json.load(sys.stdin)
row = (d.get("artifacts") or {}).get(kind)
if row is None and kind == "deb":
    row = d.get("deb")          # pre-2026-08-21 central
print((row or {}).get(field, ""))' "$kind" "$field"
  else
    # No JSON parser: recover the filename only. The sha256 comes back empty and
    # the caller warns that it is installing unverified rather than pretending.
    [ "$field" = filename ] || return 0
    local ext
    case "$kind" in
      deb) ext='\.deb' ;; rpm) ext='\.rpm' ;;
      pacman) ext='\.pacman' ;; appimage) ext='\.AppImage' ;;
    esac
    printf '%s' "$info" | tr ',' '\n' \
      | sed -n "s/.*\"filename\"[[:space:]]*:[[:space:]]*\"\\(\\(hugpy-station\\|fleet-console\\)[^\"]*$ext\\)\".*/\\1/p" \
      | head -1
  fi
}

FNAME="$(pick "$FORMAT" filename)"
SHA="$(pick "$FORMAT" sha256)"
if [ -z "$FNAME" ] && [ "$FORMAT" != appimage ]; then
  echo "note: no $FORMAT artifact staged on the central — falling back to the AppImage" >&2
  FORMAT=appimage
  FNAME="$(pick appimage filename)"
  SHA="$(pick appimage sha256)"
fi
[ -n "$FNAME" ] || { echo "error: no station artifact staged for $FORMAT" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "downloading $FNAME ..."
curl -fSL "${auth[@]}" -o "$TMP/$FNAME" "$CENTRAL/agent/console/$FNAME"

if [ -n "$SHA" ] && have sha256sum; then
  ( cd "$TMP" && echo "$SHA  $FNAME" | sha256sum -c - >/dev/null ) \
    || { echo "error: sha256 mismatch — refusing to install" >&2; exit 1; }
  echo "sha256 verified"
elif [ -n "$SHA" ] && have shasum; then
  ( cd "$TMP" && echo "$SHA  $FNAME" | shasum -a 256 -c - >/dev/null ) \
    || { echo "error: sha256 mismatch — refusing to install" >&2; exit 1; }
  echo "sha256 verified"
else
  echo "warning: no sha256 sidecar staged for $FNAME — installing unverified" >&2
fi

SUDO=""
[ "$(id -u)" = 0 ] || SUDO="sudo"

case "$FORMAT" in
deb)
  echo "installing with apt (it resolves the deb's dependencies) ..."
  if $SUDO apt-get install -y "$TMP/$FNAME"; then :; else
    $SUDO dpkg -i "$TMP/$FNAME" || $SUDO apt-get install -y -f
  fi
  ;;
rpm)
  if have dnf; then
    echo "installing with dnf ..."
    $SUDO dnf install -y "$TMP/$FNAME"
  elif have zypper; then
    echo "installing with zypper ..."
    $SUDO zypper --non-interactive install --allow-unsigned-rpm "$TMP/$FNAME"
  else
    echo "installing with yum ..."
    $SUDO yum install -y "$TMP/$FNAME"
  fi
  ;;
pacman)
  # The artifact is a real pacman package but electron-builder names it *.pacman;
  # rename to the .pkg.tar.<comp> pacman expects, with the compression sniffed
  # from the magic bytes rather than assumed.
  magic="$(od -An -tx1 -N4 "$TMP/$FNAME" | tr -d ' \n')"
  case "$magic" in
    28b52ffd*) ext=zst ;; fd377a58*) ext=xz ;; 1f8b*) ext=gz ;; *) ext=zst ;;
  esac
  PKG="$TMP/${FNAME%.pacman}.pkg.tar.$ext"
  mv "$TMP/$FNAME" "$PKG"
  echo "installing with pacman ..."
  $SUDO pacman -U --noconfirm "$PKG"
  ;;
appimage)
  # Per-user, no root. ~/.local/bin is on PATH for every modern distro's login
  # shell; if it is not, the script says so rather than silently doing nothing.
  BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
  APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
  ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps"
  mkdir -p "$BIN_DIR" "$APP_DIR" "$ICON_DIR"
  TARGET="$BIN_DIR/hugpy-station.AppImage"
  install -m 755 "$TMP/$FNAME" "$TARGET"
  ln -sf "$TARGET" "$BIN_DIR/hugpy-station"

  # Best-effort icon: the AppImage carries it at the AppDir root.
  ( cd "$TMP" && "$TARGET" --appimage-extract hugpy-station.png >/dev/null 2>&1 \
      && cp -f squashfs-root/hugpy-station.png "$ICON_DIR/hugpy-station.png" ) 2>/dev/null || true

  cat > "$APP_DIR/hugpy-station.desktop" <<EOF
[Desktop Entry]
Name=hugpy Station
Exec="$TARGET" %U
Terminal=false
Type=Application
Icon=hugpy-station
StartupWMClass=hugpy Station
Comment=hugpy Station — desktop shell over the self-contained console backend.
Categories=System;
EOF
  have update-desktop-database && update-desktop-database "$APP_DIR" 2>/dev/null || true

  echo "ok: installed $TARGET"
  case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) echo "note: $BIN_DIR is not on your PATH — add it, or run $TARGET directly" >&2 ;;
  esac
  # The AppImage bundles Electron but NOT python3-aiohttp; the launcher inside
  # refuses to start without an interpreter that can import aiohttp, and says so.
  if ! python3 -c 'import aiohttp' >/dev/null 2>&1; then
    echo "note: no python3 with aiohttp found. The console backend needs it:" >&2
    echo "      apt install python3-aiohttp | dnf install python3-aiohttp |" >&2
    echo "      pacman -S python-aiohttp | python3 -m pip install --user aiohttp" >&2
  fi
  ;;
*)
  echo "error: unknown format '$FORMAT'" >&2; exit 1 ;;
esac

# ── credential provisioning (uniform with the hugpy-agent installer) ─────────
# The download credential is not thrown away: when it is an hp_ API key it is
# persisted for the installed station, exactly like hugpy_agent.install writes
# ~/.config/hugpy-agent/agent.env. Operator tokens are deliberately NOT
# persisted — they authorize the download only.
upsert_env() {  # upsert_env <file> KEY=VALUE...
  local f="$1"; shift
  mkdir -p "$(dirname "$f")"
  touch "$f"; chmod 600 "$f"
  local tmp="$f.tmp.$$" keys pat
  keys="$(printf '%s\n' "$@" | cut -d= -f1 | paste -sd'|' -)"
  pat="^($keys)="
  { grep -v -E "$pat" "$f" 2>/dev/null || true
    printf '%s\n' "$@"
  } > "$tmp"
  chmod 600 "$tmp"
  mv "$tmp" "$f"
}

case "$TOKEN" in
hp_*)
  # user scope: read by the bundled console-api sidecar (legacy .fleet path,
  # watched live) and the canonical station location.
  upsert_env "$HOME/.config/hugpy-station/station.env" \
    "HUGPY_API_KEY=$TOKEN" "HUGPY_BASE=$CENTRAL" "HUGPY_URL=$CENTRAL"
  upsert_env "${HUGPY_ENV_FILE:-$HOME/.fleet/console-hugpy.env}" \
    "HUGPY_API_KEY=$TOKEN" "HUGPY_URL=$CENTRAL"
  echo "credential provisioned: ~/.config/hugpy-station/station.env (0600)"
  # system scope: the headless unit template reads /etc/hugpy-station/<user>.env
  # (EnvironmentFile seam). Only for native installs where we already used root.
  if [ "$FORMAT" != appimage ] && { [ "$(id -u)" = 0 ] || $SUDO -n true 2>/dev/null; }; then
    $SUDO mkdir -p /etc/hugpy-station
    printf 'HUGPY_API_KEY=%s\nHUGPY_BASE=%s\nHUGPY_URL=%s\n' "$TOKEN" "$CENTRAL" "$CENTRAL" \
      | $SUDO tee "/etc/hugpy-station/$(id -un).env" >/dev/null
    $SUDO chmod 600 "/etc/hugpy-station/$(id -un).env"
    echo "credential provisioned: /etc/hugpy-station/$(id -un).env (0600, for hugpy-station-web@$(id -un))"
  fi
  ;;
*)
  echo "note: credential was not an hp_ API key — download authorized, nothing persisted." >&2
  echo "      mint a station key in the console and re-run, or write it to" >&2
  echo "      ~/.config/hugpy-station/station.env (HUGPY_API_KEY=...) yourself." >&2
  ;;
esac

echo "ok: $FNAME installed — launch 'hugpy Station' from the desktop, or run"
echo "    hugpy-station --version"
