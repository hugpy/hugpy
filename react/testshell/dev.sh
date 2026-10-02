#!/usr/bin/env bash
# Start the DB mirror API and the vite dev server together. Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")"
[ -e node_modules ] || ln -s ../ui_shell/node_modules node_modules   # reuse the console's installed deps
# the DB ingests hugpy.json; the hub backfill reads central's cached Hub rows (never the network)
export HUGPY_MODEL_METADATA_DB="${HUGPY_MODEL_METADATA_DB:-/srv/hugpy/.hugpy/state/model_metadata.db}"
export HUGPY_TESTSHELL_PORT="${HUGPY_TESTSHELL_PORT:-7013}"
export HUGPY_TESTSHELL_API="http://127.0.0.1:${HUGPY_TESTSHELL_PORT}"
/srv/hugpy/venv/bin/python api.py &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT
exec npx vite --host "${HUGPY_TESTSHELL_BIND:-0.0.0.0}" --port "${HUGPY_TESTSHELL_UI_PORT:-7014}"
