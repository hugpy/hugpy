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
set -euo pipefail

CENTRAL="@@CENTRAL@@"
DEB_NAME="@@DEB_NAME@@"
DEB_SHA="@@DEB_SHA@@"
DEB_URL="@@DEB_URL@@"
HUGPY_API_KEY_VALUE="@@API_KEY@@"

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
{ grep -v -E '^(HUGPY_API_KEY|HUGPY_URL)=' "$ENV_FILE" 2>/dev/null || true
  echo "HUGPY_API_KEY=$HUGPY_API_KEY_VALUE"
  echo "HUGPY_URL=$CENTRAL"
} > "$TMP_ENV"
chmod 600 "$TMP_ENV"
mv "$TMP_ENV" "$ENV_FILE"
echo "credential written to $ENV_FILE"

# Canonical location too (uniform with install.sh / hugpy-agent's agent.env).
CANON="$HOME/.config/hugpy-station/station.env"
mkdir -p "$(dirname "$CANON")"
touch "$CANON"; chmod 600 "$CANON"
TMP_ENV="$CANON.tmp.$$"
{ grep -v -E '^(HUGPY_API_KEY|HUGPY_BASE|HUGPY_URL)=' "$CANON" 2>/dev/null || true
  echo "HUGPY_API_KEY=$HUGPY_API_KEY_VALUE"
  echo "HUGPY_BASE=$CENTRAL"
  echo "HUGPY_URL=$CENTRAL"
} > "$TMP_ENV"
chmod 600 "$TMP_ENV"
mv "$TMP_ENV" "$CANON"
echo "credential written to $CANON"
echo "ok: $DEB_NAME installed — launch 'fleet-console'"
