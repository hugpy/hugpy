#!/usr/bin/env bash
# fleet-console ONE-TIME install link payload (2026-08-13).
#
# Served at GET /agent/console/install/<link_id>.sh — TEMPLATED per fetch by
# agent_routes.console_install_link_sh: the @@…@@ slots below are filled with
# this deployment's base URL, the newest staged .deb (name + sha256), the
# link-scoped deb URL, and the RAW API KEY the link minted. Fetching this
# script CONSUMES the link's use (unlike the hugpy-agent .sh wrapper, which is
# free — here the .sh IS the keyed payload; there is no .py).
#
#   curl -fsSL https://dev.hugpy.ai/api/agent/console/install/<link_id>.sh | bash
#
# What it does: download the fleet-console .deb through the same link
# (validity-gated, not use-gated), verify the sha256 baked in at template
# time, apt-install it (sudo only for that step), then write the minted key to
# ~/.fleet/console-hugpy.env (0600) — the file the console's bundled
# console-api sidecar reads HUGPY_API_KEY from live, i.e. the credential the
# hugpy agents inside the installed console use. No credential is ever read
# from the caller's environment; the link is the capability.
#
# 2026-10-02 (operator): the payload establishes BOTH credentials a Station
# needs, hugpy-agent included — HUGPY_API_KEY (minted per link) and
# TOOLSERVER_AUTH_KEY (the toolserver token this central holds, or a generated
# one when central holds none — see agent_routes._toolserver_handoff). They
# are written to the HOST SEAMS the .deb's first-run reads BEFORE the install
# (/etc/hugpy-station/hugpy-api.env + toolserver.env, root 0600), so the
# Station's instance env, its seats' MCP bridge and ~/.config/hugpy-agent/
# agent.env all come up keyed on the first start. @@TOOLSERVER_URL@@ may be
# empty — then only the key is written and the Station discovers its
# toolserver by itself (abstract_toolserver.discovery).
set -euo pipefail

CENTRAL="@@CENTRAL@@"
DEB_NAME="@@DEB_NAME@@"
DEB_SHA="@@DEB_SHA@@"
DEB_URL="@@DEB_URL@@"
HUGPY_API_KEY_VALUE="@@API_KEY@@"
TOOLSERVER_URL_VALUE="@@TOOLSERVER_URL@@"
TOOLSERVER_KEY_VALUE="@@TOOLSERVER_KEY@@"
TOOLSERVER_KEY_SOURCE="@@TOOLSERVER_KEY_SOURCE@@"

# upsert KEY=VALUE lines into an env file (created 0600; other lines kept)
env_upsert() {   # env_upsert <file> <owner-or-""> KEY=VALUE...
  local f="$1" own="$2"; shift 2
  local tmp="$f.tmp.$$" keys="" kv
  for kv in "$@"; do keys="${keys}${keys:+|}${kv%%=*}"; done
  mkdir -p "$(dirname "$f")"
  { [ -f "$f" ] && grep -v -E "^($keys)=" "$f" 2>/dev/null || true
    for kv in "$@"; do echo "$kv"; done
  } > "$tmp"
  chmod 600 "$tmp"; [ -n "$own" ] && chown "$own" "$tmp" 2>/dev/null || true
  mv "$tmp" "$f"
}

command -v curl >/dev/null || { echo "error: curl is required" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "downloading $DEB_NAME ..."
curl -fSL -o "$TMP/$DEB_NAME" "$DEB_URL"

if [ -n "$DEB_SHA" ] && command -v sha256sum >/dev/null; then
  echo "$DEB_SHA  $TMP/$DEB_NAME" | sha256sum -c - >/dev/null \
    || { echo "error: sha256 mismatch — refusing to install" >&2; exit 1; }
  echo "sha256 verified"
fi

SUDO=""
[ "$(id -u)" = 0 ] || SUDO="sudo"

# Host seams FIRST (root 0600): the .deb's postinst runs hugpy-station-firstrun,
# which resolves HUGPY_URL/HUGPY_API_KEY from /etc/hugpy-station/hugpy-api.env
# and the toolserver credential from /etc/hugpy-station/toolserver.env — so the
# Station (and the hugpy-agent it provisions) is keyed on its very first start.
SEAM_DIR=/etc/hugpy-station
$SUDO mkdir -p "$SEAM_DIR" && $SUDO chmod 755 "$SEAM_DIR"
_seam_api="$(mktemp)"; _seam_ts="$(mktemp)"
{ echo "HUGPY_URL=$CENTRAL"; echo "HUGPY_API_KEY=$HUGPY_API_KEY_VALUE"; } > "$_seam_api"
{ echo "# toolserver credential delivered by the Station install link ($TOOLSERVER_KEY_SOURCE)"
  [ -n "$TOOLSERVER_URL_VALUE" ] && echo "STATION_CONSOLE_TOOLSERVER=$TOOLSERVER_URL_VALUE"
  [ -n "$TOOLSERVER_URL_VALUE" ] && echo "HUGPY_TOOLSERVER_URL=$TOOLSERVER_URL_VALUE"
  echo "STATION_CONSOLE_TOOLSERVER_TOKEN=$TOOLSERVER_KEY_VALUE"
  echo "HUGPY_TOOLSERVER_TOKEN=$TOOLSERVER_KEY_VALUE"
  echo "TOOLSERVER_AUTH_KEY=$TOOLSERVER_KEY_VALUE"
} > "$_seam_ts"
$SUDO install -m 600 -o root -g root "$_seam_api" "$SEAM_DIR/hugpy-api.env"
$SUDO install -m 600 -o root -g root "$_seam_ts" "$SEAM_DIR/toolserver.env"
rm -f "$_seam_api" "$_seam_ts"
echo "host seams written: $SEAM_DIR/hugpy-api.env, $SEAM_DIR/toolserver.env (toolserver key: $TOOLSERVER_KEY_SOURCE)"

echo "installing (apt resolves the deb's dependencies) ..."
if $SUDO apt-get install -y "$TMP/$DEB_NAME"; then
  :
else
  # older apt without local-deb support
  $SUDO dpkg -i "$TMP/$DEB_NAME" || $SUDO apt-get install -y -f
fi

# Credential drop: upsert into the env file the console-api sidecar watches.
# Runs as the INVOKING user (sudo above was only for apt), so the file lands
# in the right home even though the install needed root.
ENV_FILE="${HUGPY_ENV_FILE:-$HOME/.fleet/console-hugpy.env}"
mkdir -p "$(dirname "$ENV_FILE")"
touch "$ENV_FILE"
chmod 600 "$ENV_FILE"
TMP_ENV="$ENV_FILE.tmp.$$"
{ grep -v -E '^(HUGPY_API_KEY|HUGPY_URL|HUGPY_TOOLSERVER_URL|HUGPY_TOOLSERVER_TOKEN|TOOLSERVER_AUTH_KEY)=' "$ENV_FILE" 2>/dev/null || true
  echo "HUGPY_API_KEY=$HUGPY_API_KEY_VALUE"
  echo "HUGPY_URL=$CENTRAL"
  [ -n "$TOOLSERVER_URL_VALUE" ] && echo "HUGPY_TOOLSERVER_URL=$TOOLSERVER_URL_VALUE"
  echo "HUGPY_TOOLSERVER_TOKEN=$TOOLSERVER_KEY_VALUE"
  echo "TOOLSERVER_AUTH_KEY=$TOOLSERVER_KEY_VALUE"
} > "$TMP_ENV"
chmod 600 "$TMP_ENV"
mv "$TMP_ENV" "$ENV_FILE"
echo "credential written to $ENV_FILE"

# Canonical location too (uniform with install.sh / hugpy-agent's agent.env).
CANON="$HOME/.config/hugpy-station/station.env"
mkdir -p "$(dirname "$CANON")"
touch "$CANON"; chmod 600 "$CANON"
TMP_ENV="$CANON.tmp.$$"
{ grep -v -E '^(HUGPY_API_KEY|HUGPY_BASE|HUGPY_URL|HUGPY_TOOLSERVER_URL|HUGPY_TOOLSERVER_TOKEN|TOOLSERVER_AUTH_KEY)=' "$CANON" 2>/dev/null || true
  echo "HUGPY_API_KEY=$HUGPY_API_KEY_VALUE"
  echo "HUGPY_BASE=$CENTRAL"
  echo "HUGPY_URL=$CENTRAL"
  [ -n "$TOOLSERVER_URL_VALUE" ] && echo "HUGPY_TOOLSERVER_URL=$TOOLSERVER_URL_VALUE"
  echo "HUGPY_TOOLSERVER_TOKEN=$TOOLSERVER_KEY_VALUE"
  echo "TOOLSERVER_AUTH_KEY=$TOOLSERVER_KEY_VALUE"
} > "$TMP_ENV"
chmod 600 "$TMP_ENV"
mv "$TMP_ENV" "$CANON"
echo "credential written to $CANON"

# hugpy-agent included: the agent's own env (what `hugpy-agent` / its TUI read)
AGENT_ENV="$HOME/.config/hugpy-agent/agent.env"
_agent=(HUGPY_BASE="$CENTRAL" HUGPY_URL="$CENTRAL" HUGPY_API_KEY="$HUGPY_API_KEY_VALUE"
        HUGPY_TOOLSERVER_TOKEN="$TOOLSERVER_KEY_VALUE" TOOLSERVER_AUTH_KEY="$TOOLSERVER_KEY_VALUE")
[ -n "$TOOLSERVER_URL_VALUE" ] && _agent+=(HUGPY_TOOLSERVER_URL="$TOOLSERVER_URL_VALUE")
env_upsert "$AGENT_ENV" "" "${_agent[@]}"
echo "credential written to $AGENT_ENV"
echo "ok: $DEB_NAME installed — launch 'fleet-console'"
