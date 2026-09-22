### routes/upload_routes.py

import uuid

from abstract_flask import get_bp
from flask import abort, jsonify, request
from werkzeug.utils import secure_filename
from hugpy_platform.constants import UPLOADS_HOME
import os
import re
import time
import shutil

upload_bp, logger = get_bp("upload_bp", __name__)

# ── per-session upload lifecycle ─────────────────────────────────────────────
# Uploads are tagged with the browser's per-tab session id (sessionStorage), sent
# as the `X-Hugpy-Session` header. Each session's saved paths + a last-seen marker
# live under UPLOADS_HOME/.sessions/<sid>/. The browser heartbeats (/session/ping)
# to keep its session alive; on tab close it beacons (/session/end?sid=) to wipe
# immediately. A throttled, request-driven sweep wipes any session idle past
# SESSION_TTL — the 1h safety net for missed beacons. No cron / background thread:
# the sweep piggybacks on ping/upload, so it self-cleans whenever anyone is around.
#
# OWNERSHIP (2026-08-06). The sid identifies a TAB, not a person: every upload
# landed FLAT in UPLOADS_HOME, so any caller who knew (or guessed) a basename
# could read or DELETE another account's file, and `/session/file` verified
# traversal only. Uploads are now NAMESPACED per account —
# ``UPLOADS_HOME/<namespace>/<file>`` — where the namespace comes from
# operator_auth.upload_namespace(username), the same helper the read side
# (video_routes' raw-path ownership check) uses, so writer and reader can never
# drift. A caller with NO account (operator-token M2M, open mode, self-hosted)
# keeps the historical FLAT path and the historical behavior, byte for byte.
# The sid lifecycle above is unchanged and still tab-scoped — it is a TTL
# mechanism, not an authorization one.

SESSIONS_DIR = ".sessions"
SESSION_TTL = 3600          # wipe a session's uploads 1h after its last heartbeat
SWEEP_EVERY = 600           # at most one sweep per 10 min (race-tolerant across workers)
_SID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _sessions_root():
    return os.path.join(UPLOADS_HOME, SESSIONS_DIR)


def _valid_sid(sid):
    return isinstance(sid, str) and bool(_SID_RE.match(sid))


def _read_sid():
    # sendBeacon uses the query string (?sid=); fetch heartbeats use a header/JSON.
    sid = request.headers.get("X-Hugpy-Session") or request.args.get("sid")
    if not sid:
        body = request.get_json(silent=True)
        if isinstance(body, dict):
            sid = body.get("sid")
    if not sid:
        sid = request.form.get("sid")
    return sid if _valid_sid(sid) else None


def _touch_session(sid, path=None):
    d = os.path.join(_sessions_root(), sid)
    os.makedirs(d, exist_ok=True)
    if path:
        try:
            with open(os.path.join(d, "paths"), "a") as fh:
                fh.write(path + "\n")
        except OSError:
            pass
    # `seen` mtime is the session's last-seen timestamp.
    open(os.path.join(d, "seen"), "w").close()


def _within_uploads(path):
    root = os.path.realpath(UPLOADS_HOME)
    rp = os.path.realpath(path)
    return rp == root or rp.startswith(root + os.sep)


# ── per-ACCOUNT upload namespace ─────────────────────────────────────────────
def _caller_namespace():
    """This request's uploads subdirectory (``None`` = the flat legacy path for
    an accountless caller: operator-token M2M, open mode, self-hosted)."""
    try:
        from hugpy_server.app.operator_auth import principal_username, upload_namespace
        return upload_namespace(principal_username())
    except Exception:  # noqa: BLE001 — never fail an upload over attribution
        return None


def _caller_is_admin():
    try:
        from hugpy_server.app.operator_auth import principal_role
        return principal_role() == "operator"
    except Exception:  # noqa: BLE001 — fail closed (treat as non-admin)
        return False


def _upload_dir(ns):
    return os.path.join(UPLOADS_HOME, ns) if ns else UPLOADS_HOME


def _may_touch(path, ns, is_admin):
    """May this caller read/delete ``path``? Admin: anything under UPLOADS_HOME
    (they already see every artifact). A namespaced member: only inside their
    OWN namespace — which also means a legacy FLAT file is admin-only. An
    accountless caller (no namespace): the historical flat-jail rule."""
    if not _within_uploads(path):
        return False
    if is_admin:
        return True
    rp = os.path.realpath(path)
    if ns:
        home = os.path.realpath(os.path.join(UPLOADS_HOME, ns))
        return rp.startswith(home + os.sep)
    return True


def _wipe_session(sid, guard=None):
    """Delete every path this session recorded, then drop the session dir.

    ``guard`` (optional, 2026-08-06) is an ownership predicate applied per path.
    The REQUEST-driven /session/end passes one, so a caller presenting another
    account's sid (it is a caller-chosen string, never a credential) can only
    wipe files they own. The unattended TTL sweep passes none: that is the
    server reaping its own registry, not a caller acting through it."""
    d = os.path.join(_sessions_root(), sid)
    try:
        with open(os.path.join(d, "paths")) as fh:
            paths = [ln.strip() for ln in fh if ln.strip()]
    except OSError:
        paths = []
    for p in paths:
        try:
            if guard is not None and not guard(p):
                continue
            if _within_uploads(p) and os.path.isfile(p):
                os.unlink(p)
        except OSError:
            pass
    shutil.rmtree(d, ignore_errors=True)


def _maybe_sweep():
    root = _sessions_root()
    try:
        os.makedirs(root, exist_ok=True)
    except OSError:
        return
    marker = os.path.join(root, ".last_sweep")
    now = time.time()
    try:
        if now - os.path.getmtime(marker) < SWEEP_EVERY:
            return                       # swept recently — skip
    except OSError:
        pass                             # marker missing → run
    try:
        open(marker, "w").close()        # claim this sweep window
    except OSError:
        pass
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        if name.startswith("."):
            continue
        d = os.path.join(root, name)
        try:
            idle = now - os.path.getmtime(os.path.join(d, "seen"))
        except OSError:
            try:
                idle = now - os.path.getmtime(d)
            except OSError:
                continue
        if idle > SESSION_TTL:
            _wipe_session(name)


@upload_bp.route("/uploads", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        abort(400, description="no file provided")
    # Land the file in the CALLER'S namespace (flat for an accountless caller).
    # The response shape is unchanged — `path` is still the absolute path the UI
    # hands straight back to /video/ingest, /media/analyze and /video/media.
    ns = _caller_namespace()
    dest_dir = _upload_dir(ns)
    os.makedirs(dest_dir, exist_ok=True)
    name = f"{uuid.uuid4().hex[:8]}_{secure_filename(f.filename)}"
    dest = os.path.join(dest_dir, name)
    f.save(dest)
    sid = _read_sid()
    if sid:
        _touch_session(sid, dest)
        _maybe_sweep()
    return jsonify({"path": dest, "name": f.filename, "size": os.path.getsize(dest)})


@upload_bp.route("/session/ping", methods=["POST"])
def session_ping():
    """Heartbeat: keep this session's uploads alive; opportunistically sweep idle ones."""
    sid = _read_sid()
    if sid:
        _touch_session(sid)
        _maybe_sweep()
    return jsonify({"ok": True})


@upload_bp.route("/session/end", methods=["POST"])
def session_end():
    """Tab-close beacon: wipe this session's uploads immediately (best-effort).
    Only files the CALLER owns are wiped (a sid is not a credential)."""
    sid = _read_sid()
    if sid:
        ns = _caller_namespace()
        is_admin = _caller_is_admin()
        _wipe_session(sid, guard=lambda p: _may_touch(p, ns, is_admin))
    return jsonify({"ok": True})


def _forget_path(sid, base):
    """Drop any path whose basename == `base` from this session's registry."""
    pf = os.path.join(_sessions_root(), sid, "paths")
    try:
        with open(pf) as fh:
            kept = [ln.strip() for ln in fh
                    if ln.strip() and os.path.basename(ln.strip()) != base]
    except OSError:
        return
    try:
        with open(pf, "w") as fh:
            for p in kept:
                fh.write(p + "\n")
    except OSError:
        pass


@upload_bp.route("/session/file", methods=["DELETE", "POST"])
def session_file_delete():
    """User-initiated delete of ONE uploaded file from the store. Accepts the
    file id (the /uploads basename, or the full path /uploads returned).

    OWNERSHIP (2026-08-06): the target is resolved inside the CALLER'S namespace
    and then checked with ``_may_touch`` — traversal was the only thing verified
    before, so any caller could delete any other account's upload by basename.
    A full path is honored (that is what the UI holds since uploads became
    namespaced) but is subject to the identical check, so it cannot reach out of
    the caller's namespace. Admin may delete anything under UPLOADS_HOME; an
    accountless caller keeps the flat legacy behavior."""
    fid = (request.args.get("id")
           or (request.get_json(silent=True) or {}).get("id")
           or request.form.get("id"))
    if not fid:
        return jsonify({"ok": False, "error": "missing id"}), 400
    raw = str(fid).strip()
    base = os.path.basename(raw)
    if not base or base in (".", ".."):
        return jsonify({"ok": False, "error": "bad id"}), 400

    ns = _caller_namespace()
    is_admin = _caller_is_admin()
    # An absolute path is taken as given (and validated below); a bare id is
    # resolved in the caller's own namespace first, then — only for a caller
    # whose namespace cannot hold it (admin / accountless) — flat.
    candidates = []
    if os.path.isabs(raw):
        candidates.append(raw)
    else:
        if ns:
            candidates.append(os.path.join(UPLOADS_HOME, ns, base))
        if is_admin or not ns:
            candidates.append(os.path.join(UPLOADS_HOME, base))

    deleted = False
    forbidden = False
    for target in candidates:
        if not _may_touch(target, ns, is_admin):
            forbidden = True
            continue
        try:
            if os.path.isfile(target):
                os.unlink(target)
                deleted = True
                break
        except OSError:
            pass
    if not deleted and forbidden:
        return jsonify({"ok": False,
                        "error": "forbidden: file belongs to another account"}), 403
    sid = _read_sid()
    if sid:
        _forget_path(sid, base)
    return jsonify({"ok": True, "deleted": deleted})
