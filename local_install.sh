#!/usr/bin/env bash
# Install the partitioned Hugpy workspace editable into a virtualenv.
# Thin wrapper over py/local_install.py; see LOCAL_INSTALL.md.
#
#   ./local_install.sh                         # -> ./.venv
#   ./local_install.sh --venv /tmp/v --check   # fresh venv, then verify
#   ./local_install.sh --extras server         # central box profile
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="${PYTHON:-}"
if [ -z "$py" ]; then
  for cand in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then py="$cand"; break; fi
  done
fi
[ -n "$py" ] || { echo "local_install.sh: no python3 found (set PYTHON=...)" >&2; exit 1; }
exec "$py" "$here/py/local_install.py" "$@"
