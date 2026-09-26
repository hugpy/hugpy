#!/usr/bin/env bash
# hugpy worker bootstrap — one command from a bare box to an enrolled worker.
#
# Usage:
#   bootstrap.sh --central https://dev.hugpy.ai --name box-1 --token <enroll-token> \
#                [--port 9100] [--version 0.1.162] [--storage-root /mnt/llm_storage] \
#                [--venv ~/hugpy-worker/venv] [--force] [--dry-run] \
#                [--advertise http://<addr>:9100] [--install-comfy] [--self-check-only]
#
#   --dry-run          inspect + print every venv/pip/install step; change NOTHING.
#   --advertise URL    the address CENTRAL reaches this box on (baked as WORKER_URL);
#                      needed when central dials a tunnel/hub address, not the
#                      worker's outbound IP. Forwarded to the canonical installer.
#   --install-comfy    install/converge ComfyUI (the worker owns its lifecycle).
#   --self-check-only  run the installer self-check + report only (no unit written).
#   Any of --serve-mode/--comfy-version/--comfy-start-check/--no-verify/
#   --no-provision-engine/--no-retire-legacy/--no-open-firewall/--no-comfy are
#   forwarded verbatim to `python -m hugpy_fleet.worker.install`.
#
# What it does (idempotent — safe to re-run to upgrade):
#   1. checks python3 >= 3.10 with the venv module
#   2. creates ~/hugpy-worker/venv if missing
#   3. pip install --upgrade -c <central>/api/llm/workers/constraints.txt
#        [--extra-index-url <central>/api/llm/pip/simple] 'hugpy[<profile>]==<version>'
#      (when --version is omitted it asks <central>/api/llm/workers/required-version;
#       falls back to latest if central pins no version; when that reply names
#       central's own pip index — the release is published there — it is added
#       as an extra index so the hugpy-* wheels come from central, like model
#       files do, while third-party deps still come from PyPI). The constraints file
#       pins EVERY hugpy-* workspace distribution to that one lockstep version,
#       so the whole set lands together — never hugpy-fleet at one version and
#       hugpy-platform/-engine/-media at another (the silent-skew incident class).
#   4. runs the canonical installer, which FIRST runs the k118 environment
#      preflight (this box's self-report diffed against the fleet doctrine) and
#      refuses to register a worker with doctrine BLOCKERS unless --force, then
#      writes + enables the hugpy-worker.service systemd user unit
#      (see hugpy_fleet/worker/install.py)
#
# Why the preflight is here and not "later": a-brain had no ffmpeg, computron
# had no bitsandbytes, and both boxes registered, advertised the task, and only
# found out when a real job died on them. The check costs a few seconds at the
# one moment an operator is already watching the terminal.
#
# The profile matters: bare `hugpy` is only the CLI. `gpu-worker` (default) is
# the fleet agent + GGUF engine + model store + the media/video stacks;
# `cpu-worker` drops the GPU-only pieces; `worker` is the minimal agent+engine.
# Pick with --profile / WORKER_PROFILE. CUDA / source llama-cpp-python and the
# native llama-server are box errands (they need CMAKE_ARGS / nvcc) — see
# WORKER-SETUP.md §2/§3.
set -eu

CENTRAL=""
NAME="$(hostname)"
TOKEN=""
PORT="9100"
VERSION=""
STORAGE_ROOT=""
VENV="${HOME}/hugpy-worker/venv"
FORCE=""
PROFILE="${WORKER_PROFILE:-gpu-worker}"
DRY=""
PASSTHRU=""   # extra args forwarded verbatim to the python installer

die() { printf 'bootstrap: %s\n' "$*" >&2; exit 1; }
say() { printf 'bootstrap: %s\n' "$*"; }
# run: execute a mutating command, or (under --dry-run) print exactly what would
# run and change nothing. Every venv/pip/installer side effect goes through this.
run() {
  if [ -n "$DRY" ]; then
    printf 'bootstrap: DRY-RUN: would run: %s\n' "$*"
  else
    "$@"
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --central)      CENTRAL="${2:-}"; shift 2 ;;
    --name)         NAME="${2:-}"; shift 2 ;;
    --token)        TOKEN="${2:-}"; shift 2 ;;
    --port)         PORT="${2:-}"; shift 2 ;;
    --version)      VERSION="${2:-}"; shift 2 ;;
    --storage-root) STORAGE_ROOT="${2:-}"; shift 2 ;;
    --venv)         VENV="${2:-}"; shift 2 ;;
    --profile)      PROFILE="${2:-}"; shift 2 ;;
    --force)        FORCE="1"; shift 1 ;;
    --dry-run)      DRY="1"; shift 1 ;;
    # forward turnkey flags straight to `hugpy_fleet.worker.install` (advertise
    # URL, comfy, self-check-only, etc.) so bootstrap stays the single entry point.
    --advertise|--serve-mode|--comfy-version)
                    PASSTHRU="$PASSTHRU $1 ${2:-}"; shift 2 ;;
    --install-comfy|--no-comfy|--comfy-start-check|--self-check-only|\
    --no-verify|--no-provision-engine|--no-retire-legacy|--no-open-firewall)
                    PASSTHRU="$PASSTHRU $1"; shift 1 ;;
    -h|--help)      sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              die "unknown argument: $1" ;;
  esac
done
[ -n "$DRY" ] && say "DRY-RUN: no venv/pip/service changes will be made"

[ -n "$CENTRAL" ] || die "--central is required (e.g. --central https://dev.hugpy.ai)"
CENTRAL="${CENTRAL%/}"   # strip a trailing slash so URL joins are clean
CENTRAL="${CENTRAL%/api}" # --central is the origin; /api is appended where needed

# 1. python3 >= 3.10 with the venv module ----------------------------------
command -v python3 >/dev/null 2>&1 || die "python3 not found (need >= 3.10)"
python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' \
  || die "python3 >= 3.10 required (found $(python3 -V 2>&1))"
python3 -c 'import venv' 2>/dev/null \
  || die "python3 venv module missing (install python3-venv)"

# 2. venv ------------------------------------------------------------------
if [ ! -x "${VENV}/bin/python" ]; then
  say "creating venv at ${VENV}"
  run python3 -m venv "$VENV"
fi
PY_BIN="${VENV}/bin/python"
PIP_BIN="${VENV}/bin/pip"

# fetch_central <path>: GET ${CENTRAL}/api<path> to stdout; empty on failure.
# --central is the bare origin (what the worker agent takes; it appends /api
# itself), so the API mount is added here — without it these lookups hit the
# console SPA and silently fell back to an unpinned, unconstrained install.
# LAN centrals often front a cert the box doesn't trust; these values only
# pick which version pip pulls FROM THE INDEX, so an insecure retry is a
# version-pin risk, not a code-injection one. Warn either way.
fetch_central() {
  _out=""
  if command -v curl >/dev/null 2>&1; then
    _out="$(curl -fsSL "${CENTRAL}/api$1" 2>/dev/null || true)"
    if [ -z "$_out" ]; then
      say "WARNING: strict query of $1 failed; retrying with certificate checks off"
      _out="$(curl -fskL "${CENTRAL}/api$1" 2>/dev/null || true)"
    fi
  else
    _out="$(wget -qO- "${CENTRAL}/api$1" 2>/dev/null || true)"
    if [ -z "$_out" ]; then
      say "WARNING: strict query of $1 failed; retrying with certificate checks off"
      _out="$(wget -qO- --no-check-certificate "${CENTRAL}/api$1" 2>/dev/null || true)"
    fi
  fi
  printf '%s' "$_out"
}

# 3. resolve the package version -------------------------------------------
PKG_INDEX_URL="${WORKER_PKG_INDEX_URL:-}"
if [ -z "$VERSION" ] || [ -z "$PKG_INDEX_URL" ]; then
  say "querying ${CENTRAL} for the required package version"
  RESP="$(fetch_central /llm/workers/required-version)"
  if [ -z "$VERSION" ]; then
    # Pull the string value out of {"required_pkg_version": "0.1.x"}; null -> empty.
    VERSION="$(printf '%s' "$RESP" \
      | sed -n 's/.*"required_pkg_version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
    if [ -z "$VERSION" ]; then
      say "WARNING: could not resolve required version from central (query failed or"
      say "         central pins none) — falling back to the LATEST release."
    fi
  fi
  if [ -z "$PKG_INDEX_URL" ]; then
    # Central names its own index only when it holds every wheel of that version.
    PKG_INDEX_URL="$(printf '%s' "$RESP" \
      | sed -n 's/.*"pkg_index_url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
  fi
fi
PIP_EXTRA_INDEX=""
if [ -n "$PKG_INDEX_URL" ]; then
  PIP_EXTRA_INDEX="--extra-index-url ${PKG_INDEX_URL}"
  say "central serves this version from its own pip index: ${PKG_INDEX_URL}"
  # pip silently drops a plain-http index on a non-loopback host unless trusted
  # (a LAN central over http is the normal case).
  case "$PKG_INDEX_URL" in
    http://127.0.0.1*|http://localhost*|https://*) ;;
    http://*)
      _host="${PKG_INDEX_URL#http://}"; _host="${_host%%/*}"
      PIP_EXTRA_INDEX="${PIP_EXTRA_INDEX} --trusted-host ${_host}" ;;
  esac
fi

# 3b. the lockstep constraints -----------------------------------------------
# Central serves `name==<required>` for EVERY hugpy-* workspace distribution
# (204/empty when it pins none). Passed to pip as -c so the profile install
# resolves all of them to the one version central runs — the same file the
# agent's self-update converges under afterwards. Missing/empty → unconstrained
# (pip's own resolution), which can leave siblings at another version: warn.
CONSTRAINTS_FILE=""
PIP_CONSTRAINT=""
if [ -n "$VERSION" ]; then
  CONSTRAINTS_BODY="$(fetch_central /llm/workers/constraints.txt)"
  if printf '%s' "$CONSTRAINTS_BODY" | grep -q '=='; then
    CONSTRAINTS_FILE="$(mktemp "${TMPDIR:-/tmp}/hugpy-constraints.XXXXXX")"
    printf '%s\n' "$CONSTRAINTS_BODY" > "$CONSTRAINTS_FILE"
    PIP_CONSTRAINT="-c ${CONSTRAINTS_FILE}"
    say "lockstep constraints from central: $(grep -c '==' "$CONSTRAINTS_FILE") pins -> ${CONSTRAINTS_FILE}"
  else
    say "WARNING: no lockstep constraints from ${CENTRAL}/api/llm/workers/constraints.txt —"
    say "         installing unconstrained; sibling hugpy-* distributions may end up"
    say "         at another version than ${VERSION} (fleet skew) until the agent converges."
  fi
fi

# 4. install / upgrade the package -----------------------------------------
if [ -n "$VERSION" ]; then
  SPEC="hugpy[${PROFILE}]==${VERSION}"
else
  say "no version resolved from central — installing latest [${PROFILE}]"
  SPEC="hugpy[${PROFILE}]"
fi
say "pip install --upgrade ${PIP_CONSTRAINT} ${PIP_EXTRA_INDEX} '${SPEC}'"
# shellcheck disable=SC2086  # PIP_CONSTRAINT / PIP_EXTRA_INDEX are intentionally two words or empty
run "$PIP_BIN" install --upgrade $PIP_CONSTRAINT $PIP_EXTRA_INDEX "$SPEC"
if [ -n "$CONSTRAINTS_FILE" ]; then rm -f "$CONSTRAINTS_FILE"; fi

# 4b. optional media-intelligence deps the canonical [engine] venv omits -----
# On 2026-07-11 three /ml requests reached workers whose venv lacked these and
# failed AT REQUEST TIME: feature-extraction/sentence-similarity died with
# "sentence-transformers is required…", and transcription hit a whisper NoneType.
# A THIRD failure was numpy 2.5 breaking numba so that `import whisper` itself
# dies — hence the pin. Install them here so an enrolled worker can actually run
# ASR / embeddings / keyword-extraction; central now also SKIPS a worker that
# still can't (task_capabilities gate), but shipping the deps is the real fix.
# NB: the agent's self-update converges under central's constraints with
# `--upgrade-strategy only-if-needed` (hugpy-* pins only), so it never touches
# these — they persist across every version converge. (Its no-constraints
# fallback is `pip install -U --no-deps`, which touches even less.)
say "installing media-intelligence deps (sentence-transformers, openai-whisper, keybert; numpy<2.5 for numba)"
run "$PIP_BIN" install --upgrade sentence-transformers openai-whisper keybert "numpy<2.5"

# 5. write + enable the systemd unit via the canonical installer -----------
# COMPAT: when central pins a version older than 0.1.164 the installed
# installer predates --storage-root/--enroll-token (seen live: op, 2026-07-10,
# "unrecognized arguments"). Both have env-var fallbacks the old argparse
# defaults already read (DEFAULT_ROOT / WORKER_ENROLL_TOKEN), so probe --help
# for the new flags and use the env route when they're absent.
set -- --central "$CENTRAL" --name "$NAME" --port "$PORT"
if [ -x "$PY_BIN" ]; then
  HELP="$("$PY_BIN" -m hugpy_fleet.worker.install --help 2>&1 || true)"
else
  HELP=""   # dry-run on a bare box (no venv yet): forward flags unconditionally
fi
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
# Forward turnkey passthrough flags (advertise/comfy/self-check/etc) verbatim.
# shellcheck disable=SC2086  # PASSTHRU is intentionally word-split
set -- "$@" $PASSTHRU
if [ -n "$DRY" ]; then
  # Under dry-run: let the canonical installer run its own inspection/report
  # (which changes nothing) when the venv exists, else print the exact command.
  case "$HELP" in *--dry-run*) set -- "$@" --dry-run;; esac
  if [ -x "$PY_BIN" ]; then
    say "DRY-RUN: running installer self-check (no changes): $PY_BIN -m hugpy_fleet.worker.install $*"
    exec "$PY_BIN" -m hugpy_fleet.worker.install "$@"
  fi
  say "DRY-RUN: would run: $PY_BIN -m hugpy_fleet.worker.install $* --dry-run"
  exit 0
fi
say "registering worker service (hugpy-worker.service)"
exec "$PY_BIN" -m hugpy_fleet.worker.install "$@"
