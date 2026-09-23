#!/usr/bin/env bash
# release.sh — cut a hugpy release. A release IS a git tag (CONSISTENCY.md).
#
#   ./release.sh X.Y.Z [--dry-run] [--remote origin]
#
# Refuses unless: the tree is clean, the current branch is the remote's default
# branch and up to date with it (fetched first), the tag does not exist yet,
# `python py/validate_partition.py --versions` passes, and
# `hugpy-drift-check --sections A,B` passes when that command is installed.
# Then: git tag -a vX.Y.Z -m "hugpy X.Y.Z" && git push <remote> vX.Y.Z.
#
# Nothing else is edited: setuptools-scm turns the tag into the version of all
# 13 distributions at build time, CI (.github/workflows/pypi-publish.yml)
# builds, verifies against the tag, publishes to PyPI and creates the GitHub
# Release. --dry-run runs every gate, reports each, prints the plan, tags nothing.
set -euo pipefail

usage() { sed -n '2,15p' "${BASH_SOURCE[0]}"; }

VERSION=""; DRY_RUN=0; REMOTE="origin"
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
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
if [ -x "$WORKSPACE/.venv/bin/python" ]; then PYTHON="$WORKSPACE/.venv/bin/python"; else PYTHON="${PYTHON:-python3}"; fi

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

echo "plan:"
echo "  git tag -a $TAG -m \"hugpy $VERSION\"        (at ${LOCAL_SHA:0:12})"
echo "  git push $REMOTE $TAG"
echo "then, without any further edit:"
echo "  1. CI pypi-publish.yml builds all 13 distributions at $VERSION, verifies each against the tag,"
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
