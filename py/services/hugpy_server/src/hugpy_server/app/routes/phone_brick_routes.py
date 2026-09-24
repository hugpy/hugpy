"""HTTP surface for the phone-brick video-analytics pool.

Like the GPU worker routes, this serves two audiences:

  * the phones (machine-to-machine):
        POST /phone-brick/register
        POST /phone-brick/<id>/heartbeat
  * the console UI (human-driven):
        GET    /phone-brick/phones
        DELETE /phone-brick/phones/<id>
        GET    /phone-brick/phones/<id>/health     central -> phone reachability
        POST   /phone-brick/run                     fan an image across the pool
        GET    /phone-brick/runs                     recent runs
        GET    /phone-brick/runs/<id>                poll one run
        GET    /phone-brick/runs/<id>/image          the annotated result
  * the phones again (fetch the seeded image to run detection on):
        GET    /phone-brick/files/<path>
  * provisioning a NEW phone (run in Termux on the phone):
        GET    /phone-brick/install.sh   templated bootstrap (curl | bash)
        GET    /phone-brick/code.tar.gz  phone_brick/ + gguf_worker/ from the
                                         running tree (also how phones update)

All pool state lives in functions.imports.utils.phone_brick_store; this module
only translates HTTP <-> that store and the orchestrator manager. Phones reach
these endpoints directly over the VPN (the /api prefix is stripped by nginx in
prod and by ApiPrefixMiddleware on bare gunicorn), exactly like GPU workers.
"""
import json
import os
import time

from flask import request, jsonify, abort, send_from_directory, Response, stream_with_context
from werkzeug.utils import secure_filename

from abstract_flask import get_bp
from hugpy_fleet.central.phone_brick_store import (
    register_phone,
    heartbeat_phone,
    remove_phone,
    list_phones,
    get_phone,
    online_phones,
    run_store,
    output_dir,
)
from hugpy_fleet.phone_brick_orchestrator import start_run

phone_brick_bp, logger = get_bp("phone_brick_bp", __name__)

# Hosts a phone might self-report that central can't actually call back on.
_UNREACHABLE_HOSTS = {"127.0.0.1", "127.0.1.1", "localhost", "0.0.0.0", "::1", ""}


from hugpy_server.app.auth_common import client_ip as _client_ip


def _resolve_host(advertised: str | None) -> str:
    """Trust a usable advertised host, else use the request's source IP."""
    if advertised and advertised.lower() not in _UNREACHABLE_HOSTS:
        return advertised
    return _client_ip()


def _central_base() -> str:
    """Central's base URL as the PHONES can reach it (not necessarily the
    address the browser uses): PHONE_BRICK_CENTRAL_URL when set, else derived
    from the request host (works when phone and browser share it)."""
    central = os.environ.get("PHONE_BRICK_CENTRAL_URL")
    if central:
        return central.rstrip("/")
    proto = request.headers.get("X-Forwarded-Proto", request.scheme)
    host = request.headers.get("X-Forwarded-Host") or request.host
    return f"{proto}://{host}"


def _file_server_base() -> str:
    """Base URL the phones fetch the seeded image from.

    PHONE_BRICK_FILE_SERVER is used verbatim (full URL ending in /) when set;
    otherwise ``_central_base() + /api/phone-brick/files/`` — the /api prefix
    resolves over nginx and over bare gunicorn alike.
    """
    explicit = os.environ.get("PHONE_BRICK_FILE_SERVER")
    if explicit:
        return explicit.rstrip("/") + "/"
    return _central_base() + "/api/phone-brick/files/"


# ── provisioning: bootstrap a brand-new phone ──────────────────────────────
@phone_brick_bp.route("/phone-brick/install.sh", methods=["GET"])
def phone_install_script():
    """Templated Termux bootstrap. On the phone:

        curl -sL "http://<central>/api/phone-brick/install.sh?name=X&role=both" | bash

    Query params: ``name`` (default: the phone's hostname) and ``role``
    (ppe | llm | both, default ppe). The central URL baked in is the one this
    request arrived on (override with PHONE_BRICK_CENTRAL_URL when the phones
    reach central at a different address than the requester).
    """
    role = request.args.get("role", "ppe")
    if role not in ("ppe", "llm", "both"):
        abort(400, description="role must be ppe, llm or both")
    from hugpy_fleet.phone_brick import __file__ as _pb_file
    path = os.path.join(os.path.dirname(_pb_file), "bootstrap.sh")
    with open(path, encoding="utf-8") as fh:
        script = fh.read()
    script = (script
              .replace("@CENTRAL_URL@", _central_base())
              .replace("@NAME@", request.args.get("name", ""))
              .replace("@ROLE@", role))
    return Response(script, mimetype="text/x-shellscript")


@phone_brick_bp.route("/phone-brick/code.tar.gz", methods=["GET"])
def phone_code_tarball():
    """The phone-side packages (phone_brick/, gguf_worker/) tarred from the
    RUNNING tree — a phone always installs exactly the code central runs, and
    re-running the bootstrap is how a phone updates."""
    import io
    import tarfile
    from hugpy_fleet.phone_brick import __file__ as _pb_file
    from hugpy_fleet.gguf_worker import __file__ as _gw_file
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for pkg_file in (_pb_file, _gw_file):
            root = os.path.dirname(pkg_file)
            base = os.path.basename(root)
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                for fn in filenames:
                    if fn.endswith(".pyc"):
                        continue
                    full = os.path.join(dirpath, fn)
                    arc = os.path.join(base, os.path.relpath(full, root))
                    tar.add(full, arcname=arc)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="application/gzip",
                    headers={"Content-Disposition":
                             "attachment; filename=phone-brick-code.tar.gz"})


# ── machine-to-machine: phones register + heartbeat ────────────────────────
@phone_brick_bp.route("/phone-brick/register", methods=["POST"])
def phone_register():
    body = request.get_json(silent=True) or {}
    host = _resolve_host(body.get("host"))
    phone = register_phone(
        name=body.get("name") or host,
        host=host,
        port=int(body.get("port", 5002)),
        color=body.get("color", "#58a6ff"),
        phone_id=body.get("phone_id"),
    )
    return jsonify(phone)


@phone_brick_bp.route("/phone-brick/<phone_id>/heartbeat", methods=["POST"])
def phone_heartbeat(phone_id):
    body = request.get_json(silent=True) or {}
    phone = heartbeat_phone(
        phone_id,
        live=body.get("live"),
        host=_resolve_host(body.get("host")) if body.get("host") else None,
        port=int(body["port"]) if body.get("port") else None,
    )
    if phone is None:
        # Central forgot this phone (restart / cleared registry) — re-register.
        abort(410, description="Unknown phone id; please re-register.")
    return jsonify(phone)


# ── console UI: manage the pool ────────────────────────────────────────────
@phone_brick_bp.route("/phone-brick/phones", methods=["GET"])
def phones_list():
    return jsonify(list_phones())


@phone_brick_bp.route("/phone-brick/phones/<phone_id>", methods=["DELETE"])
def phones_remove(phone_id):
    if not remove_phone(phone_id):
        abort(404, description="Unknown phone id.")
    return jsonify({"removed": True, "id": phone_id})


@phone_brick_bp.route("/phone-brick/phones/<phone_id>/health", methods=["GET"])
def phone_health(phone_id):
    """Probe the phone's own /status — confirms central -> phone connectivity."""
    phone = get_phone(phone_id)
    if phone is None:
        abort(404, description="Unknown phone id.")
    url = (phone.get("url") or "").rstrip("/") + "/status"
    try:
        import httpx
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        return jsonify({"reachable": True, "url": url, "status": resp.json()})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"reachable": False, "url": url,
                        "error": f"{type(exc).__name__}: {exc}"})


# ── console UI: orchestration runs ─────────────────────────────────────────
@phone_brick_bp.route("/phone-brick/run", methods=["POST"])
def run_start():
    """Fan one image across the pool.

    Multipart: an ``image`` file plus optional ``phone_ids`` (comma list). When
    no phone_ids are given, every online phone is used. The image is saved into
    the served output dir so the phones can fetch it.
    """
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        abort(400, description="Missing 'image' file upload.")

    raw_ids = request.form.get("phone_ids", "").strip()
    if raw_ids:
        wanted = {pid.strip() for pid in raw_ids.split(",") if pid.strip()}
        phone_records = [p for p in list_phones() if p["id"] in wanted]
    else:
        phone_records = online_phones()
    if not phone_records:
        abort(409, description="No phones available for the run (none online?).")

    fname = secure_filename(upload.filename) or "upload.jpg"
    image_path = os.path.join(output_dir(), fname)
    upload.save(image_path)

    try:
        run = start_run(
            image_path=image_path,
            phone_records=phone_records,
            file_server=_file_server_base(),
            out_dir=output_dir(),
            run_store=run_store,
        )
    except (ValueError, FileNotFoundError) as exc:
        abort(400, description=str(exc))
    return jsonify(run)


@phone_brick_bp.route("/phone-brick/runs", methods=["GET"])
def runs_list():
    return jsonify(run_store.all())


@phone_brick_bp.route("/phone-brick/runs/<run_id>", methods=["GET"])
def run_get(run_id):
    run = run_store.get(run_id)
    if run is None:
        abort(404, description="Unknown run id.")
    return jsonify(run)


@phone_brick_bp.route("/phone-brick/runs/<run_id>/cancel", methods=["POST"])
def run_cancel(run_id):
    run = run_store.request_cancel(run_id)
    if run is None:
        abort(404, description="Unknown run id.")
    return jsonify({"cancel_requested": bool(run.get("cancel_requested")),
                    "status": run.get("status"), "id": run_id})


# Terminal run states the SSE stream stops on.
_TERMINAL = {"done", "error", "cancelled"}


@phone_brick_bp.route("/phone-brick/runs/<run_id>/stream", methods=["GET"])
def run_stream(run_id):
    """Server-Sent Events of a run's live progress.

    Tails the run record (the disk store is the source of truth, so this works
    no matter which process ran the job) and pushes an event whenever a new
    per-phone verdict lands, the current phone changes, or the status changes —
    ending with a terminal event carrying the full final run. A page refresh can
    just GET /runs/<id> instead; this is only the live transport.
    """
    if run_store.get(run_id) is None:
        abort(404, description="Unknown run id.")

    def sse(payload: dict) -> bytes:
        return f"data: {json.dumps(payload)}\n\n".encode("utf-8")

    def generate():
        sent_progress = 0
        last_status = None
        last_current = "\0"   # sentinel so the first value always emits
        deadline = time.time() + 900    # safety cap: never hang a thread forever
        while time.time() < deadline:
            run = run_store.get(run_id)
            if run is None:
                yield sse({"type": "error", "message": "run vanished"})
                return
            # New per-phone verdicts since we last sent.
            prog = run.get("progress") or []
            while sent_progress < len(prog):
                yield sse({"type": "progress", "phase": prog[sent_progress]})
                sent_progress += 1
            # Which phone is being asked right now.
            cur = run.get("current_phone")
            if cur != last_current:
                last_current = cur
                yield sse({"type": "current", "phone": cur})
            # Status transitions.
            if run["status"] != last_status:
                last_status = run["status"]
                yield sse({"type": "status", "status": run["status"]})
            if run["status"] in _TERMINAL:
                yield sse({"type": run["status"], "run": run})
                return
            time.sleep(0.5)
        yield sse({"type": "error", "message": "stream timed out"})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
        direct_passthrough=True,
    )


@phone_brick_bp.route("/phone-brick/runs/<run_id>/image", methods=["GET"])
def run_image(run_id):
    run = run_store.get(run_id)
    if run is None or not run.get("output_rel"):
        abort(404, description="No result image for this run.")
    return _serve_from_output(run["output_rel"])


# ── machine-to-machine: phones fetch the seeded image ──────────────────────
@phone_brick_bp.route("/phone-brick/files/<path:filename>", methods=["GET"])
def serve_file(filename):
    return _serve_from_output(filename)


def _serve_from_output(filename: str):
    """Send a file from the run output dir, confined to that directory."""
    base = os.path.realpath(output_dir())
    target = os.path.realpath(os.path.join(base, filename))
    if target != base and not target.startswith(base + os.sep):
        abort(403, description="Path escapes output directory.")
    if not os.path.isfile(target):
        abort(404, description="No such file.")
    return send_from_directory(base, os.path.relpath(target, base), conditional=True)
