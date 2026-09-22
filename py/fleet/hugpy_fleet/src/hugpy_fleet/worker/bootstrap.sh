#!/usr/bin/env bash
# hugpy worker bootstrap — one command from a bare box to an enrolled worker.
#
# Usage:
#   bootstrap.sh --central https://dev.hugpy.ai --name box-1 --token <enroll-token> \
#                [--port 9100] [--version 0.1.162] [--storage-root /mnt/llm_storage] \
#                [--venv ~/hugpy-worker/venv] [--force]
#
# What it does (idempotent — safe to re-run to upgrade):
#   1. checks python3 >= 3.10 with the venv module
#   2. creates ~/hugpy-worker/venv if missing
#   3. pip install --upgrade 'abstract_hugpy_dev[engine]==<version>'
#      (when --version is omitted it asks <central>/llm/workers/required-version;
#       falls back to latest if central pins no version)
#   4. runs the canonical installer, which FIRST runs the k118 environment
#      preflight (this box's self-report diffed against the fleet doctrine) and
#      refuses to register a worker with doctrine BLOCKERS unless --force, then
#      writes + enables the hugpy-worker.service systemd user unit
#      (see worker_agent/install.py)
#
# Why the preflight is here and not "later": a-brain had no ffmpeg, computron
# had no bitsandbytes, and both boxes registered, advertised the task, and only
# found out when a real job died on them. The check costs a few seconds at the
# one moment an operator is already watching the terminal.
#
# The [engine] extra matters: base abstract_hugpy_dev deliberately omits
# llama-cpp-python, so a worker without it registers fine but serves NO GGUFs.
# CUDA / source llama-cpp-python and the native llama-server are box errands
# (they need CMAKE_ARGS / nvcc) — see WORKER-SETUP.md §2/§3.
set -eu

CENTRAL=""
NAME="$(hostname)"
TOKEN=""
PORT="9100"
VERSION=""
STORAGE_ROOT=""
VENV="${HOME}/hugpy-worker/venv"
FORCE=""

die() { printf 'bootstrap: %s\n' "$*" >&2; exit 1; }
say() { printf 'bootstrap: %s\n' "$*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --central)      CENTRAL="${2:-}"; shift 2 ;;
    --name)         NAME="${2:-}"; shift 2 ;;
    --token)        TOKEN="${2:-}"; shift 2 ;;
    --port)         PORT="${2:-}"; shift 2 ;;
    --version)      VERSION="${2:-}"; shift 2 ;;
    --storage-root) STORAGE_ROOT="${2:-}"; shift 2 ;;
    --venv)         VENV="${2:-}"; shift 2 ;;
    --force)        FORCE="1"; shift 1 ;;
    -h|--help)      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              die "unknown argument: $1" ;;
  esac
done

[ -n "$CENTRAL" ] || die "--central is required (e.g. --central https://dev.hugpy.ai)"
CENTRAL="${CENTRAL%/}"   # strip a trailing slash so URL joins are clean

# 1. python3 >= 3.10 with the venv module ----------------------------------
command -v python3 >/dev/null 2>&1 || die "python3 not found (need >= 3.10)"
python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' \
  || die "python3 >= 3.10 required (found $(python3 -V 2>&1))"
python3 -c 'import venv' 2>/dev/null \
  || die "python3 venv module missing (install python3-venv)"

# 2. venv ------------------------------------------------------------------
if [ ! -x "${VENV}/bin/python" ]; then
  say "creating venv at ${VENV}"
  python3 -m venv "$VENV"
fi
PY_BIN="${VENV}/bin/python"
PIP_BIN="${VENV}/bin/pip"

# 3. resolve the package version -------------------------------------------
if [ -z "$VERSION" ]; then
  say "querying ${CENTRAL} for the required package version"
  if command -v curl >/dev/null 2>&1; then
    RESP="$(curl -fsSL "${CENTRAL}/llm/workers/required-version" 2>/dev/null || true)"
    # LAN centrals often front a cert the box doesn't trust; the value only
    # picks which version pip pulls FROM PYPI, so an insecure retry is a
    # version-pin risk, not a code-injection one. Warn either way.
    if [ -z "$RESP" ]; then
      say "WARNING: strict query failed; retrying with certificate checks off"
      RESP="$(curl -fskL "${CENTRAL}/llm/workers/required-version" 2>/dev/null || true)"
    fi
  else
    RESP="$(wget -qO- "${CENTRAL}/llm/workers/required-version" 2>/dev/null || true)"
    if [ -z "$RESP" ]; then
      say "WARNING: strict query failed; retrying with certificate checks off"
      RESP="$(wget -qO- --no-check-certificate "${CENTRAL}/llm/workers/required-version" 2>/dev/null || true)"
    fi
  fi
  # Pull the string value out of {"required_pkg_version": "0.1.x"}; null -> empty.
  VERSION="$(printf '%s' "$RESP" \
    | sed -n 's/.*"required_pkg_version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
  if [ -z "$VERSION" ]; then
    say "WARNING: could not resolve required version from central (query failed or"
    say "         central pins none) — falling back to the LATEST PyPI release."
  fi
fi

# 4. install / upgrade the package -----------------------------------------
if [ -n "$VERSION" ]; then
  SPEC="abstract_hugpy_dev[engine]==${VERSION}"
else
  say "no version resolved from central — installing latest [engine]"
  SPEC="abstract_hugpy_dev[engine]"
fi
say "pip install --upgrade '${SPEC}'"
"$PIP_BIN" install --upgrade "$SPEC"

# 4b. optional media-intelligence deps the canonical [engine] venv omits -----
# On 2026-07-11 three /ml requests reached workers whose venv lacked these and
# failed AT REQUEST TIME: feature-extraction/sentence-similarity died with
# "sentence-transformers is required…", and transcription hit a whisper NoneType.
# A THIRD failure was numpy 2.5 breaking numba so that `import whisper` itself
# dies — hence the pin. Install them here so an enrolled worker can actually run
# ASR / embeddings / keyword-extraction; central now also SKIPS a worker that
# still can't (task_capabilities gate), but shipping the deps is the real fix.
# NB: the agent's self-update uses `pip install -U --no-deps`, so it never
# touches these — they persist across every version converge.
say "installing media-intelligence deps (sentence-transformers, openai-whisper, keybert; numpy<2.5 for numba)"
"$PIP_BIN" install --upgrade sentence-transformers openai-whisper keybert "numpy<2.5"

# 5. write + enable the systemd unit via the canonical installer -----------
# COMPAT: when central pins a version older than 0.1.164 the installed
# installer predates --storage-root/--enroll-token (seen live: op, 2026-07-10,
# "unrecognized arguments"). Both have env-var fallbacks the old argparse
# defaults already read (DEFAULT_ROOT / WORKER_ENROLL_TOKEN), so probe --help
# for the new flags and use the env route when they're absent.
set -- --central "$CENTRAL" --name "$NAME" --port "$PORT"
HELP="$("$PY_BIN" -m abstract_hugpy_dev.worker_agent.install --help 2>&1 || true)"
if [ -n "$FORCE" ]; then
  case "$HELP" in *--force*) set -- "$@" --force;;
                  *) say "NOTE: this installer predates --force; ignoring";; esac
fi
if [ -n "$TOKEN" ]; then
  case "$HELP" in *--enroll-token*) set -- "$@" --enroll-token "$TOKEN";;
                  *) export WORKER_ENROLL_TOKEN="$TOKEN";; esac
fi
if [ -n "$STORAGE_ROOT" ]; then
  case "$HELP" in *--storage-root*) set -- "$@" --storage-root "$STORAGE_ROOT";;
                  *) set -- "$@" --storage "$STORAGE_ROOT";; esac
fi
say "registering worker service (hugpy-worker.service)"
exec "$PY_BIN" -m abstract_hugpy_dev.worker_agent.install "$@"
