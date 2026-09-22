### routes/pypi_routes.py
"""The fleet's PRIVATE package index (operator decision 2026-08-13).

char360 / identity-render (and any future fleet package) are deliberately NOT
on public PyPI — the composition is reproducible but the pipeline is product,
and the heavy deps carry licenses worth keeping out of a public manifest. So
central serves its own PEP 503 "simple" index, the same way it already serves
the fleet-console deb and hugpy_agent wheel: artifacts in the keeper deploy
dir, gated by the fleet's own credentials, zero new services.

  GET /pypi/simple/                        project list        (PEP 503 HTML)
  GET /pypi/simple/<project>/              file list + sha256  (PEP 503 HTML)
  GET /pypi/packages/<project>/<filename>  the artifact

Layout on disk (env ``HUGPY_PYPI_DIR``): one subdir per NORMALIZED project
name, wheels/sdists inside, nothing else::

    /mnt/llm_storage/_keeper_deploy/pypi/
        char360/char360-0.1.0-py3-none-any.whl
        identity-render/identity_render-0.1.0-py3-none-any.whl

Publishing a release = drop the files in (build with ``python -m build``,
copy, done — same no-code-change contract as the console artifacts dir).

AUTH — pip can't send our headers, so three doors, all existing credentials:
  * member/operator session or bearer (curl, browsers) via the same
    ``member_authenticated`` gate the console artifacts use;
  * HTTP **Basic where the PASSWORD is the token** (the GitLab/Gitea idiom —
    pip's native auth): ``pip install --index-url
    https://fleet:<token>@dev.hugpy.ai/api/pypi/simple/ …``. The username is
    ignored. Accepted tokens: the operator token, or a worker box's
    ENROLLMENT token (fleet machines already hold one for
    register/heartbeat — revoking a worker cuts off its package pulls at the
    same instant, the review-ingest precedent);
  * a bare bearer that verifies as an enrollment token (scripted fetches
    from worker boxes without Basic).

DEPENDENCY-CONFUSION NOTE (read before adding this index to a resolver):
project names here may also exist on public PyPI under someone else's
control. NEVER mix this index and PyPI in one resolve for these names —
install fleet packages explicitly from here (``--index-url … --no-deps``;
their heavy deps are installer-managed by design), and let everything else
come from PyPI in a separate invocation.
"""
import hashlib
import hmac
import os
import re

from flask import Response, request, abort

from abstract_flask import get_bp

pypi_bp, logger = get_bp("pypi_bp", __name__)

_PYPI_DIR = os.getenv("HUGPY_PYPI_DIR", "/mnt/llm_storage/_keeper_deploy/pypi")

# Wheels and sdists only — anything else in a project dir (checksums, notes)
# is invisible to the index rather than served.
_DIST_SUFFIXES = (".whl", ".tar.gz", ".zip")


def _normalize(name: str) -> str:
    """PEP 503 project-name normalization."""
    return re.sub(r"[-_.]+", "-", (name or "")).lower()


def _pypi_authorized() -> bool:
    """The three doors documented in the module header. Fails CLOSED —
    an import error in any gate module means 401, never open."""
    # Door 1: the console's own member/operator gate (session or bearer).
    try:
        from hugpy_server.app.operator_auth import member_authenticated
        if member_authenticated():
            return True
    except Exception:  # noqa: BLE001 — fall through to the token doors
        pass

    def _token_ok(tok: str) -> bool:
        if not tok:
            return False
        # Operator token, constant-time.
        try:
            from hugpy_server.app.operator_auth import _operator_token
            known = _operator_token()
            if known and hmac.compare_digest(tok, known):
                return True
        except Exception:  # noqa: BLE001
            pass
        # Worker enrollment token — the review-ingest precedent.
        try:
            from hugpy_fleet.central.enrollment_tokens import verify_enrollment_token
            return bool(verify_enrollment_token(tok))
        except Exception:  # noqa: BLE001
            return False

    # Door 2: HTTP Basic, password-as-token (pip's native mechanism).
    auth = request.authorization
    if auth and _token_ok(auth.password or auth.username or ""):
        return True
    # Door 3: bare bearer that is an enrollment token (member_authenticated
    # already covered the operator-bearer case above).
    header = (request.headers.get("Authorization") or "").strip()
    if header.lower().startswith("bearer "):
        return _token_ok(header[7:].strip())
    return False


def _require_pypi_auth() -> None:
    if _pypi_authorized():
        return
    # WWW-Authenticate: Basic is what makes pip prompt/retry with the
    # credential instead of giving up on the 401. flask.abort accepts a
    # Response and raises it as-is.
    from flask import Response
    abort(Response(
        "Private fleet index — authenticate with the operator or an "
        "enrollment token as the Basic password.", 401,
        {"WWW-Authenticate": 'Basic realm="hugpy-pypi"'}))


_sha_cache: dict = {}   # path -> (mtime, size, sha256) — files are immutable-ish


def _sha256(path: str) -> str:
    st = os.stat(path)
    key = (st.st_mtime, st.st_size)
    cached = _sha_cache.get(path)
    if cached and cached[0] == key:
        return cached[1]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    _sha_cache[path] = (key, h.hexdigest())
    return h.hexdigest()


def _projects() -> list[str]:
    try:
        return sorted(d for d in os.listdir(_PYPI_DIR)
                      if os.path.isdir(os.path.join(_PYPI_DIR, d))
                      and _normalize(d) == d)
    except OSError:
        return []


def _project_files(project: str) -> list[str]:
    d = os.path.join(_PYPI_DIR, project)
    try:
        return sorted(f for f in os.listdir(d)
                      if f.endswith(_DIST_SUFFIXES)
                      and f == os.path.basename(f))
    except OSError:
        return []


def _html(title: str, anchors: list[str]) -> "Response":
    from flask import Response
    body = ("<!DOCTYPE html><html><head><title>{t}</title>"
            "<meta name=\"pypi:repository-version\" content=\"1.0\">"
            "</head><body><h1>{t}</h1>\n{a}\n</body></html>").format(
                t=title, a="\n".join(anchors))
    return Response(body, mimetype="text/html")


@pypi_bp.route("/pypi/simple/", methods=["GET"])
def pypi_simple_root():
    _require_pypi_auth()
    anchors = [f'<a href="{p}/">{p}</a><br>' for p in _projects()]
    return _html("hugpy private index", anchors)


@pypi_bp.route("/pypi/simple/<project>/", methods=["GET"])
def pypi_simple_project(project):
    _require_pypi_auth()
    project = _normalize(project)
    if project not in _projects():
        abort(404, description="Unknown project on this index.")
    anchors = []
    for fname in _project_files(project):
        sha = _sha256(os.path.join(_PYPI_DIR, project, fname))
        anchors.append(
            f'<a href="../../packages/{project}/{fname}#sha256={sha}">'
            f"{fname}</a><br>")
    return _html(f"Links for {project}", anchors)


@pypi_bp.route("/pypi/packages/<project>/<filename>", methods=["GET"])
def pypi_package_download(project, filename):
    _require_pypi_auth()
    project = _normalize(project)
    # Same no-traversal discipline as console_dist_download: bare names with a
    # distributable suffix, existing under the project's own dir, or 404.
    if filename != os.path.basename(filename) or "/" in filename:
        abort(404)
    if not filename.endswith(_DIST_SUFFIXES):
        abort(404)
    path = os.path.join(_PYPI_DIR, project, filename)
    if not os.path.isfile(path):
        abort(404)
    from flask import send_file
    return send_file(path, as_attachment=True, download_name=filename,
                     conditional=True)
