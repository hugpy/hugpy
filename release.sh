#!/usr/bin/env bash
# release.sh — cut a hugpy release. A release IS a git tag (CONSISTENCY.md).
#
#   ./release.sh X.Y.Z [--known-good] [--dry-run] [--remote origin]
#
# Refuses unless: the tree is clean, the current branch is the remote's default
# branch and up to date with it (fetched first), the tag does not exist yet,
# `python py/validate_partition.py --versions` passes, and
# `hugpy-drift-check --sections A,B` passes when that command is installed.
# ``--known-good`` additionally runs the complete source test suite before a
# tag can be created.  It is the release command for a version intended for
# PyPI and fleet rollout: no test pass, no immutable version tag.
# The core-behaviour contracts (call queue/relay, evict+fit, model key
# resolution) live in each package's tests/known_good/ and are catalogued in
# notes/KNOWN-GOOD-CORE.md (/srv/hugpy/notes on the fleet host).
# Then: git tag -a vX.Y.Z -m "hugpy X.Y.Z" && git push <remote> vX.Y.Z.
#
# Nothing else is edited: setuptools-scm turns the tag into the version of all
# 14 distributions at build time, CI (.github/workflows/pypi-publish.yml)
# builds, verifies against the tag, publishes to PyPI and creates the GitHub
# Release. --dry-run runs every gate, reports each, prints the plan, tags nothing.
set -euo pipefail

usage() { sed -n '2,15p' "${BASH_SOURCE[0]}"; }

VERSION=""; DRY_RUN=0; KNOWN_GOOD=0; REMOTE="origin"
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --known-good) KNOWN_GOOD=1 ;;
        --remote) [ $# -ge 2 ] || { echo "release.sh: --remote needs a value" >&2; exit 2; }
                  REMOTE="$2"; shift ;;
        --remote=*) REMOTE="${1#--remote=}" ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "release.sh: unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) [ -z "$VERSION" ] || { echo "release.sh: one version only (got '$VERSION' and '$1')" >&2; exit 2; }
           VERSION="$1" ;;
    esac
    shift
done
[ -n "$VERSION" ] || { usage >&2; exit 2; }
VERSION="${VERSION#v}"
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?(\.post[0-9]+)?$ ]]; then
    echo "release.sh: '$VERSION' is not X.Y.Z (optional aN/bN/rcN/.postN)" >&2; exit 2
fi
TAG="v$VERSION"

WORKSPACE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
cd "$WORKSPACE"
# A release may be invoked by the host automation account while this checkout
# belongs to the build account.  Scope Git's ownership acknowledgement to this
# command only; never mutate a caller's global Git configuration just to cut a
# release.
git() { command git -c safe.directory="$WORKSPACE" "$@"; }
if [ -x "$WORKSPACE/.venv/bin/python" ]; then PYTHON="$WORKSPACE/.venv/bin/python"; else PYTHON="${PYTHON:-python3}"; fi
# The service runtime intentionally has no test dependency.  Prefer a caller's
# explicit HUGPY_TEST_PYTHON, then the selected build interpreter, then the
# established host API test environment.  A known-good release must fail closed
# if none can import pytest; it must never silently omit the suite.
TEST_PYTHON="${HUGPY_TEST_PYTHON:-$PYTHON}"
if ! "$TEST_PYTHON" -c 'import pytest' >/dev/null 2>&1; then
    for _candidate in /srv/hugpy/miniforge3/envs/api/bin/python \
                      /srv/hugpy/miniforge3/envs/api/bin/python3; do
        if [ -x "$_candidate" ] && "$_candidate" -c 'import pytest' >/dev/null 2>&1; then
            TEST_PYTHON="$_candidate"
            break
        fi
    done
fi

FAILS=0
ok()   { printf '  ok    %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; FAILS=$((FAILS+1)); [ "$DRY_RUN" -eq 1 ] || { echo "release.sh: refusing to release $TAG" >&2; exit 1; }; }
note() { printf '  note  %s\n' "$*"; }

echo "release.sh: hugpy $VERSION -> tag $TAG on $REMOTE$([ "$DRY_RUN" -eq 1 ] && echo '  (DRY RUN: gates only, no tag, no push)')"
echo "gates:"

# 1. clean tree (tracked files; untracked never ship, setuptools-scm ignores them)
if [ -z "$(git status --porcelain --untracked-files=no)" ]; then
    ok "working tree clean"
else
    fail "working tree has uncommitted changes: $(git status --porcelain --untracked-files=no | head -3 | tr '\n' ' ')"
fi
untracked=$(git status --porcelain --untracked-files=normal | grep -c '^??' || true)
[ "$untracked" -eq 0 ] || note "$untracked untracked path(s) present (ignored: not tracked, not shipped)"

# 2. fetch, then: on the default branch and level with the remote
if git fetch --tags --quiet "$REMOTE" 2>/dev/null; then
    ok "fetched $REMOTE"
else
    fail "could not fetch $REMOTE"
fi
DEFAULT_BRANCH="$(git symbolic-ref --quiet --short "refs/remotes/$REMOTE/HEAD" 2>/dev/null | sed "s|^$REMOTE/||" || true)"
if [ -z "$DEFAULT_BRANCH" ]; then
    DEFAULT_BRANCH="$(git remote show "$REMOTE" 2>/dev/null | sed -n 's/^ *HEAD branch: //p' || true)"
fi
DEFAULT_BRANCH="${DEFAULT_BRANCH:-main}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" = "$DEFAULT_BRANCH" ]; then
    ok "on the default branch ($DEFAULT_BRANCH)"
else
    fail "branch is not the default branch: on '$BRANCH', $REMOTE's default is '$DEFAULT_BRANCH'"
fi
LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git rev-parse --verify --quiet "refs/remotes/$REMOTE/$DEFAULT_BRANCH" || true)"
if [ -z "$REMOTE_SHA" ]; then
    fail "$REMOTE/$DEFAULT_BRANCH not found locally (fetch failed?)"
elif [ "$LOCAL_SHA" = "$REMOTE_SHA" ]; then
    ok "HEAD ${LOCAL_SHA:0:12} == $REMOTE/$DEFAULT_BRANCH"
else
    fail "HEAD ${LOCAL_SHA:0:12} != $REMOTE/$DEFAULT_BRANCH ${REMOTE_SHA:0:12} (push or pull first; a release tags what the remote has)"
fi

# 3. the tag must be new, locally and on the remote
if git rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null; then
    fail "tag $TAG already exists locally"
elif git ls-remote --exit-code --tags "$REMOTE" "refs/tags/$TAG" >/dev/null 2>&1; then
    fail "tag $TAG already exists on $REMOTE"
else
    ok "tag $TAG is new"
fi

# 4. the versioning contract
if "$PYTHON" py/validate_partition.py --versions >/tmp/release-versions.$$ 2>&1; then
    ok "$(tail -1 /tmp/release-versions.$$)"
else
    fail "python py/validate_partition.py --versions:"; sed 's/^/        /' /tmp/release-versions.$$
fi
rm -f /tmp/release-versions.$$

# 5. drift check sections A (git) and B (installed vs checkout), when installed
if command -v hugpy-drift-check >/dev/null 2>&1; then
    if hugpy-drift-check --sections A,B; then
        ok "hugpy-drift-check --sections A,B"
    else
        fail "hugpy-drift-check --sections A,B reported drift (exit $?)"
    fi
else
    note "hugpy-drift-check not on PATH (pip install hugpy-ops); sections A,B skipped"
fi

# 6. The optional full-suite gate is deliberately here, after the cheap source
# checks, so a known-good release never spends time testing a tree that cannot
# be tagged.  Give every in-tree ``src`` root precedence without relying on a
# developer's editable installs.  The command is written out in the release
# log and its complete output remains in a temporary file when it fails.
if [ "$KNOWN_GOOD" -eq 1 ]; then
    TEST_LOG="${TMPDIR:-/tmp}/hugpy-known-good-${VERSION}-$$.log"
    SOURCE_PATH="$(find "$WORKSPACE/py" -mindepth 3 -maxdepth 3 -type d -path '*/src' -printf '%p:' | sed 's/:$//')"
    # Test each and only each distribution that the lockstep tag will publish.
    # Root collection also walks vendored and external-project tests, and plain
    # prepend imports collide on common names such as ``test_import_policy``.
    # Importlib mode gives every test its own module identity.
    mapfile -t TEST_PATHS < <("$TEST_PYTHON" - <<'PY'
import os
import tomllib

with open("py/partition.toml", "rb") as manifest:
    for package in tomllib.load(manifest)["package"]:
        candidate = os.path.join("py", package["destination"], "tests")
        if os.path.isdir(candidate):
            print(candidate)
PY
)
    if [ "${#TEST_PATHS[@]}" -eq 0 ]; then
        fail "no manifest package test directories found"
    fi
    TEST_HELPER_PATH="$(IFS=:; printf '%s' "${TEST_PATHS[*]}")"
    # Pin the rootdir: without it pytest adopts the FIRST in-tree pyproject.toml
    # carrying [tool.pytest.ini_options] (hugpy_media's) as rootdir, node ids
    # collapse to bare basenames, and same-named test files in different
    # distributions share module-scoped/autouse fixtures across packages.
    PYTEST_ARGS=(-q --import-mode=importlib --rootdir="$WORKSPACE")
    if "$TEST_PYTHON" -m pytest --help 2>/dev/null | grep -q -- '--timeout'; then
        PYTEST_ARGS+=(--timeout=120)
    else
        note "pytest-timeout unavailable in $TEST_PYTHON; running the complete suite without its per-test timeout option"
    fi
    if PYTHONPATH="${SOURCE_PATH}:${TEST_HELPER_PATH}${PYTHONPATH:+:$PYTHONPATH}" \
        "$TEST_PYTHON" -m pytest "${PYTEST_ARGS[@]}" "${TEST_PATHS[@]}" >"$TEST_LOG" 2>&1; then
        ok "complete pytest suite (${PYTEST_ARGS[*]})"
        rm -f "$TEST_LOG"
    else
        fail "complete pytest suite failed; log retained at $TEST_LOG"
        tail -80 "$TEST_LOG" | sed 's/^/        /' || true
    fi
fi

echo "plan:"
echo "  git tag -a $TAG -m \"hugpy $VERSION\"        (at ${LOCAL_SHA:0:12})"
echo "  git push $REMOTE $TAG"
if [ "$KNOWN_GOOD" -eq 1 ]; then
    echo "  full pytest suite passed before this tag (known-good gate)"
fi
echo "then, without any further edit:"
echo "  1. CI pypi-publish.yml builds all 14 distributions at $VERSION, verifies each against the tag,"
echo "     publishes to PyPI (environment 'pypi', trusted publishing) and creates the GitHub Release $TAG."
echo "  2. central adopts:  pip install -U \"hugpy[server]==$VERSION\"  && restart the hugpy service;"
echo "     central then advertises required_pkg_version=$VERSION (its own installed version)."
echo "  3. every worker converges to $VERSION on its next heartbeat (pip from central's index, re-exec)."
echo "  4. hugpy-drift-check --sections A,B,C,D goes green on central once 2 and 3 have happened."

if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$FAILS" -gt 0 ]; then
        echo "release.sh: dry run: $FAILS gate(s) would refuse; nothing tagged"; exit 1
    fi
    echo "release.sh: dry run: all gates pass; nothing tagged"; exit 0
fi

git tag -a "$TAG" -m "hugpy $VERSION"
git push "$REMOTE" "$TAG"
echo "release.sh: pushed $TAG to $REMOTE; watch https://github.com/hugpy/hugpy/actions"
