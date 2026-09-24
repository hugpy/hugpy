"""P3.1 — the `/agent/*` agent-node registry + dispatch blueprint.

Phase 3 turns central into the head of a *fleet of agents*: remote P2.7
daemons enroll and heartbeat like GPU workers / phone bricks, the operator
dispatches a task to a node and watches it, and the node pulls its queue.

Like the worker and phone-brick blueprints, this serves two audiences — and
the split (public machine-to-machine vs operator-only) is deliberate, the same
public-vs-internal curation the /endpoints inspector surfaces:

  * the nodes (machine-to-machine):
        POST /agent/register              enroll -> {id, token, ...}   (bootstrap; open)
        POST /agent/<id>/heartbeat        {status, current_task, version}   (node token)
        GET  /agent/<id>/tasks?since=     pull queued tasks            (node token)
        POST /agent/<id>/tasks/<seq>/result  {status, result} -> finalize  (node token; P3.1b)
  * the console UI (operator, human-driven):
        GET  /agent/nodes                 list every node + live status  (operator)
        GET  /agent/<id>/tasks/<seq>      one task's full row incl result (operator; P3.1b)
        POST /agent/<id>/dispatch         {task} -> queue it for a node  (operator)
  * the terminal (operator, human-driven, no browser):
        GET  /agent/client.sh             serve the bash dispatch client   (open)
  * secure one-time install links (2026-07-23):
        POST   /agent/install-links            mint scoped key + link   (member|operator, strict)
        GET    /agent/install-links            list links + status      (member: own only; operator: all)
        DELETE /agent/install-links/<id>       revoke link AND its key  (operator OR the link's creator)
        GET    /agent/install/<link_id>        one-time templated .py download (link capability)
        GET    /agent/install/<link_id>.sh     POSIX wrapper (free fetch; .py is the use)
        GET    /agent/install/<link_id>.ps1    Windows wrapper (free fetch; .py is the use)
        GET    /agent/install/<link_id>.zip    macOS double-click archive (free fetch; .py is the use)
        GET    /agent/install/<link_id>.pkg    macOS Installer package (free fetch; .py is the use)

Gates, all fail-closed:
  * ``register`` is the unauthenticated bootstrap (a node has no credential
    yet) — it MINTS the node's enroll token, returned exactly once. (A future
    slice can front it with a pre-shared enrollment secret, exactly like the
    GPU workers' HUGPY_WORKER_ENROLL_REQUIRED gate; the spec issues the
    credential here, so today it is open like /phone-brick/register.)
  * ``heartbeat`` and ``tasks`` are node-authenticated: the caller must present
    THIS node's enroll token (``Authorization: Bearer <token>`` or
    ``X-Agent-Token``). Missing/mismatched -> 401; a node central has forgotten
    -> 410 (re-register); a revoked node -> 403.
  * ``nodes`` and ``dispatch`` are operator-only, via ``operator_authenticated``
    — the exact gate the console-management routes use (and additionally listed
    in operator_auth._SENSITIVE so the central before_request gate also covers
    them). It fails closed in external mode / whenever an operator token is set.
  * ``client.sh`` is unauthenticated, same rationale as the worker/phone-brick
    bootstrap scripts: it is plain client code with no embedded secret. The
    dispatch/nodes calls IT makes are still operator-gated as above — the
    script reads the operator token from the caller's own environment.

All node state lives in comms.agent_nodes (the shared comms SQLite db — cross
-process, gunicorn 3-worker safe), never per-process memory. Nodes may reach
these endpoints over nginx (/api stripped) or directly over the VPN; the /api
dual-mount in wsgi_app.py makes both resolve, exactly like the GPU workers.
"""
import json
import os

from flask import request, jsonify, abort

from abstract_flask import get_bp
from hugpy_fleet.central.agent_nodes import agent_node_store

agent_bp, logger = get_bp("agent_bp", __name__)


# ── credential extraction ──────────────────────────────────────────────────
def _agent_token() -> str | None:
    """The node's enroll token, from Authorization: Bearer <token> (the M2M
    credential the workers already use) or the X-Agent-Token convenience
    header. Never logged."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        tok = auth[7:].strip()
        if tok:
            return tok
    tok = (request.headers.get("X-Agent-Token") or "").strip()
    return tok or None


def _require_node_auth(node_id: str) -> dict:
    """Authenticate a node-facing request and return the node's public view.

    Fail-closed with honest codes: unknown node -> 410 (re-register), revoked
    node -> 403, missing/mismatched token -> 401. Aborts on any failure; on
    success returns the node dict."""
    node = agent_node_store.get(node_id)
    if node is None:
        abort(410, description="Unknown agent node id; please re-register.")
    if node.get("revoked"):
        abort(403, description="Agent node revoked by the operator.")
    if not agent_node_store.authenticate(node_id, _agent_token()):
        abort(401, description="Agent node token invalid or required.")
    return node


def _agent_gates_open() -> bool:
    """OPERATOR-DIRECTED open mode (2026-07-15: "can the agents feature be ungated
    entirely for now?"): ``HUGPY_AGENT_OPEN`` truthy waives the OPERATOR gate on
    ``/agent/nodes`` + ``/agent/<id>/dispatch`` for local testing.

    ⚠️ SCOPE NARROWED 2026-07-16 (operator: *"if the hugpy_agent_open is bypassing
    gating then rework the method so that it abides to the gate"*). It used to waive
    the ``/agent/register`` API-key gate TOO — see ``_require_api_key``, which is now
    PERMANENT and does NOT consult this function. That combination was the same
    silent-reopen as the old sitewide-toggle coupling, just behind a different flag:
    flip open mode for a test and the credential-minting bootstrap door reopened to
    the internet. **Open mode can no longer waive the register key. Ever.**

    What it still waives (deliberate, testing-only):
      * the operator gate on ``nodes`` / ``dispatch`` / ``tasks/<seq>``.
    What it NEVER waives:
      * the ``/agent/register`` API key — the one endpoint that MINTS credentials;
      * node-TOKEN auth on heartbeat/tasks/result — that's a node's identity, not a
        human gate; without it any caller could read/claim another node's queue.

    ⚠ These routes are reachable from the PUBLIC INTERNET on this deployment (the
    host front → :7001 → :7002 ``/api`` chain bypasses VM nginx allow-deny), so open
    mode still means anyone can LIST nodes and DISPATCH tasks to them. It remains a
    deliberate env flag defaulting CLOSED — unset the var + restart to restore the
    operator gate. It is a local-testing convenience, not a deployment posture."""
    return (os.getenv("HUGPY_AGENT_OPEN", "") or "").strip().lower() in (
        "1", "true", "yes", "on")


def _require_operator() -> None:
    """Operator gate for the console-facing routes. Fails closed if the gate
    module is unavailable for any reason (never fail open) — unless the operator
    has explicitly opened the agents feature (``HUGPY_AGENT_OPEN``)."""
    if _agent_gates_open():
        return
    try:
        from hugpy_server.app.operator_auth import operator_authenticated
    except Exception:
        abort(401, description="Operator authentication required for this route.")
    if not operator_authenticated():
        abort(401, description="Operator authentication required for this route.")


from hugpy_server.app.auth_common import bearer_token as _api_key_bearer


def _require_api_key() -> None:
    """PERMANENT API-key gate for the bootstrap ``/agent/register``.

    OPERATOR RULING 2026-07-16 (verbatim): *"this agent key should be a separate
    api category entirely"* — *"unlike the api key for v1 that gates or ungates the
    entire schema sitewide with a click, [agent calls] should be gated permanently."*

    THE FLAW THIS FIXES. Until now this gate read::

        if api_key_required() and not verify_api_key(...):   # ← coupled

    i.e. agent enrollment inherited the SITEWIDE key policy: flip ``/v1`` keyless
    for a demo and ``/agent/register`` silently reopened too. Those two are
    categorically different asks:

      * ``/v1`` open  = a deliberate POSTURE choice ("this deployment is keyless"),
        toggled from the console by design.
      * ``/agent/register`` open = never intended. It is the fleet BOOTSTRAP — the
        one endpoint that MINTS node credentials — and it is reachable from the
        PUBLIC INTERNET here (host front → :7001 devServer → :7002 forwards
        ``/api/agent/register`` from localhost, which no connection-IP / nginx
        allow-deny can gate). Verified open live on 2026-07-16: a bare public POST
        returned 201 and minted a node.

    So the gate no longer consults ``api_key_required()``: a valid console-minted
    key is required ALWAYS, independent of the sitewide toggle. ``verify_api_key``
    already validates on the key's own merits (hash + revocation) and never read
    that flag, so no key-system change was needed — only the decoupling.

    Fails closed: an unloadable key module refuses rather than admits.

    NOR does it consult ``_agent_gates_open()`` (operator 2026-07-16: *"if the
    hugpy_agent_open is bypassing gating then rework the method so that it abides to
    the gate"*). Open mode waiving this key was the SAME silent-reopen as the
    sitewide coupling, just behind a different flag — flip it for a test and the
    public bootstrap door swings open again. Open mode may still waive the OPERATOR
    gate on nodes/dispatch (its testing purpose); it can never waive this one.

    CONSEQUENCE (intended): every agent daemon must be handed a console API key to
    enroll — including this VM's own todo-keeper node. Bootstrap is the hardest
    door, not the softest. Keys are minted in the console under API access
    (``POST /keys``) and are individually revocable (``DELETE /keys/<id>``)."""
    try:
        from hugpy_server.app.functions.imports.utils.api_keys import verify_api_key
    except Exception:
        # fail closed: if the key module can't load we cannot verify -> refuse
        abort(401, description="Agent registration key gate unavailable.")
    # required_scope="agent-register" (2026-07-23): the key must carry that
    # scope or "full". Legacy keys (no scopes field) read as ["full"] and keep
    # passing; a narrowly scoped install-link key (e.g. ["v1"]) cannot enroll
    # a node unless the operator granted it "agent-register" at mint time.
    if not verify_api_key(_api_key_bearer(), required_scope="agent-register"):
        abort(401, description=(
            "Agent registration requires a valid API key with the "
            "'agent-register' scope. Pass 'Authorization: Bearer <key>' "
            "(create keys in the console under API access)."))


# ── machine-to-machine: nodes enroll + heartbeat ───────────────────────────
@agent_bp.route("/agent/register", methods=["POST"])
def agent_register():
    """Bootstrap enrollment. {name, host, capabilities} -> node id + one-time
    enroll token. Gated by the console API-key policy (``_require_api_key`` — the
    same gate as the general ``/v1`` and media ``/ml`` endpoints); the node token
    it receives is what authenticates every subsequent call."""
    _require_api_key()
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        abort(400, description="Agent registration requires a 'name'.")
    caps = body.get("capabilities")
    if caps is None:
        caps = []
    if not isinstance(caps, list):
        abort(400, description="'capabilities' must be a list.")
    node = agent_node_store.register(
        name=name,
        host=(body.get("host") or "").strip(),
        capabilities=caps,
    )
    return jsonify(node), 201


@agent_bp.route("/agent/<node_id>/heartbeat", methods=["POST"])
def agent_heartbeat(node_id):
    """A node reports it is alive. {status, current_task, version}. Node-token
    authenticated; only the provided fields are written."""
    _require_node_auth(node_id)
    body = request.get_json(silent=True) or {}

    def _s(v):
        return str(v) if v is not None else None

    node = agent_node_store.heartbeat(
        node_id,
        status=_s(body.get("status")),
        current_task=_s(body.get("current_task")),
        version=_s(body.get("version")),
    )
    if node is None:
        # Raced with a delete between the auth check and the write.
        abort(410, description="Unknown agent node id; please re-register.")
    return jsonify(node)


# ── machine-to-machine: a node pulls its dispatched tasks ──────────────────
@agent_bp.route("/agent/<node_id>/tasks", methods=["GET"])
def agent_tasks(node_id):
    """The node's queue. ``?since=<seq>`` pulls only tasks newer than the
    cursor the node last saw (the pull is idempotent). Returns the tasks plus a
    ``cursor`` to pass as ``since`` next time. Node-token authenticated."""
    _require_node_auth(node_id)
    try:
        since = int(request.args.get("since", 0) or 0)
    except (TypeError, ValueError):
        since = 0
    tasks = agent_node_store.tasks_since(node_id, since=since)
    cursor = tasks[-1]["seq"] if tasks else since
    return jsonify({"node_id": node_id, "since": since,
                    "cursor": cursor, "tasks": tasks})


# ── machine-to-machine: a node reports a task's result (P3.1b) ─────────────
_RESULT_STATUS = {"done", "error"}


@agent_bp.route("/agent/<node_id>/tasks/<seq>/result", methods=["POST"])
def agent_task_result(node_id, seq):
    """A node reports the OUTCOME of a dispatched task. Node-token authenticated
    (the SAME gate as heartbeat/tasks — unknown node → 410, revoked → 403,
    missing/bad token → 401). Body: ``{status: "done"|"error", result: <string>}``.

    Transitions the task row queued → done/error and stores the (size-capped)
    result + finished_at. Fail-closed on the task itself: a seq that is unknown
    or belongs to ANOTHER node → 404. Idempotent-safe: re-posting an already
    finalized task does NOT overwrite it → 409 with the stored view (first report
    wins; a node re-posting after a crash treats BOTH 200 and 409 as 'recorded')."""
    _require_node_auth(node_id)
    body = request.get_json(silent=True) or {}
    status = (body.get("status") or "").strip().lower()
    if status not in _RESULT_STATUS:
        abort(400, description="Result 'status' must be 'done' or 'error'.")
    result = body.get("result")
    if result is not None and not isinstance(result, str):
        result = json.dumps(result)      # structured output round-trips as text
    outcome = agent_node_store.complete_task(
        node_id, seq, status=status, result=result)
    if outcome.get("ok"):
        return jsonify(outcome["task"]), 200
    if outcome.get("reason") == "conflict":
        return jsonify(dict(outcome["task"], already_finalized=True)), 409
    abort(404, description="Unknown task for this node.")


# ── public: serve the terminal dispatch client ──────────────────────────────
def _find_dispatch_client() -> "str | None":
    """Locate ``hugpy_agent/bin/hugpy-dispatch`` on disk.

    ``HUGPY_AGENT_CLIENT_SH`` overrides the path outright (ops convenience /
    testing) and is AUTHORITATIVE when set — a misconfigured override (points
    at nothing) surfaces as 404 rather than silently falling back to
    auto-discovery, so a bad env var is visible instead of masked. Only when
    the var is unset do we walk up from THIS file (not cwd — the service runs
    with ``chdir ~/station/dev/abstract_hugpy_dev``, one level short of the
    repo root that actually holds ``hugpy_agent/``) until a directory
    containing ``hugpy_agent/bin/hugpy-dispatch`` is found. Returns None if
    nothing resolves."""
    override = os.getenv("HUGPY_AGENT_CLIENT_SH")
    if override:
        return override if os.path.isfile(override) else None
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(20):
        candidate = os.path.join(d, "hugpy_agent", "bin", "hugpy-dispatch")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


@agent_bp.route("/agent/client.sh", methods=["GET"])
def agent_client_sh():
    """Public: serve the terminal dispatch client so any box can install it
    with one line::

        curl -fsSL https://dev.hugpy.ai/api/agent/client.sh | bash -s install

    Unauthenticated by design, same rationale as the worker/phone-brick
    bootstrap scripts (``workers_install_sh`` / ``phone_install_script``):
    the script is plain client code with no embedded secret — it reads the
    operator token from the caller's environment or config file at RUN time,
    never from this response. Piping to ``bash -s install`` means the
    script's ``install`` subcommand cannot copy itself from ``$0`` (stdin has
    no file), so it re-downloads its own source from
    ``$HUGPY_CENTRAL/agent/client.sh`` in that case — see the install
    subcommand's stdin-fallback branch.

    404s with a one-line explanation if the script cannot be located (e.g. a
    deployment layout where ``hugpy_agent/`` isn't a sibling of the repo this
    file ships from) rather than 500ing.
    """
    from flask import Response
    path = _find_dispatch_client()
    if path is None:
        abort(404, description=(
            "Dispatch client script not found on this deployment "
            "(hugpy_agent/bin/hugpy-dispatch missing)."))
    with open(path, encoding="utf-8") as fh:
        script = fh.read()
    return Response(script, mimetype="text/x-shellscript")


# ── secure one-time install links (2026-07-23, operator-approved) ──────────
# The console owner mints a labeled/scoped ONE-TIME download link for the
# hugpy-agent installer. The download templates a freshly minted scoped key
# into the installer's EMBEDDED_API_KEY slot — the operator never sees or
# handles the raw key (it is NEVER in the mint response; it exists only inside
# the download). Mint/list/revoke are OPERATOR-gated (the operator token /
# session — a mere api key can never mint a key: "no key-minting-by-key",
# same structural rule as the video-share links). The download GET itself is
# gated by the link_id (an unguessable secrets.token_urlsafe capability).
#
# Use counting: only the .py fetch consumes a use. The .sh/.ps1/.zip/.pkg
# wrappers are free (audited but not decremented) so the one-liner
#     curl -fsSL <base>/agent/install/<link_id>.sh | bash
# — which fetches the wrapper AND then the .py — costs exactly ONE use. The
# .zip (2026-07-25, field report: a macOS tester downloaded the raw .sh from
# a browser and hit "Permission denied" — browsers strip +x on download) is
# the SAME free pattern: it's an archive containing an already-executable
# .command file that, when double-clicked LATER, runs the identical .sh
# one-liner — so downloading the zip itself never consumes the link either.
# The .pkg (tier 2, 2026-07-25) is the same rule for the same reason: the
# package installs nothing itself, its postinstall runs that identical .sh
# one-liner at install time — a use-eating download would break the very
# install it exists to enable.

def _install_links_mod():
    from hugpy_server.app.functions.imports.utils import install_links
    return install_links


def _find_installer_py() -> "str | None":
    """Locate ``hugpy_agent/install/install_hugpy_agent.py`` on disk.

    Same discovery idiom as ``_find_dispatch_client`` (the served client.sh):
    ``HUGPY_AGENT_INSTALLER_PY`` overrides outright and is AUTHORITATIVE when
    set (a bad override 404s visibly instead of silently falling back); else
    walk up from THIS file until a directory containing the installer is found.
    SINGLE SOURCE by design: the installer ships in the hugpy_agent repo (its
    tests import it from there) and central serves that same file — no
    build-time copy to drift."""
    override = os.getenv("HUGPY_AGENT_INSTALLER_PY")
    if override:
        return override if os.path.isfile(override) else None
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(20):
        candidate = os.path.join(
            d, "hugpy_agent", "install", "install_hugpy_agent.py")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


_EMBED_LINE = 'EMBEDDED_API_KEY = ""'
_ICON_BASE_LINE = 'EMBEDDED_ICON_BASE = ""'


def _template_installer(source: str, raw_key: str,
                        icon_base: str = "") -> "str | None":
    """Replace the installer's EMBEDDED_API_KEY slot with the raw key, and (if
    present) the EMBEDDED_ICON_BASE slot with this deployment's public base so
    the launcher step can fetch the mark. Returns None if the KEY slot is
    missing (installer drifted — refuse to serve an un-keyed download from a
    one-time link). The icon slot is OPTIONAL: an older installer without it
    still serves fine (iconless launcher), so its absence never fails."""
    if _EMBED_LINE not in source:
        return None
    # repr() so any quoting is safe; each slot is a plain assignment.
    out = source.replace(_EMBED_LINE, f"EMBEDDED_API_KEY = {raw_key!r}", 1)
    if _ICON_BASE_LINE in out:
        out = out.replace(_ICON_BASE_LINE,
                          f"EMBEDDED_ICON_BASE = {(icon_base or '')!r}", 1)
    return out


@agent_bp.route("/agent/install-links", methods=["POST"])
def install_link_create():
    """MEMBER or OPERATOR: mint a scoped key + its one-time install link.

    2026-08-06: a member may mint a link for their own machine. The creating
    username is recorded on the link (and onto the minted key's label /
    created_by), and a non-operator's ``scopes`` request is CLAMPED to
    ``_MEMBER_LINK_SCOPES`` — a member can never mint "full" or the
    fleet-enrolling "agent-register".
    Body: {label (required), scopes (default ["v1"]), key_expires_at? (epoch s
    or ISO-8601), link_ttl_s (default 86400), max_uses (default 1)}.
    Returns {url, link_id, label, scopes, expires_at, max_uses, uses_left,
    key_id, status, commands: {linux, macos, windows},
    downloads: {macos_zip, macos_pkg?}} — NEVER the raw key.
    ``macos_pkg`` appears only where central can build one (see
    ``_install_downloads`` / ``_pkg_missing_tools``).
    ``commands`` is a ready-to-paste string per platform built from ``url``
    (see ``_install_commands``) — additive, 2026-07-25. ``downloads`` is the
    downloadable-archive counterpart for the double-click flow (see
    ``_install_downloads``) — additive, 2026-07-25."""
    _require_member_strict()
    body = request.get_json(silent=True) or {}
    try:
        link = _install_links_mod().create_install_link(**_link_mint_args(body))
    except ValueError as exc:
        abort(400, description=str(exc))
    link["url"] = f"{_install_public_base()}/agent/install/{link['link_id']}"
    link["commands"] = _install_commands(link["url"])
    link["downloads"] = _install_downloads(link["url"])
    return jsonify(link), 201


def _link_mint_args(body: dict) -> dict:
    """Validate + normalize the mint body shared by BOTH install-link kinds
    (hugpy-agent and fleet-console — factored out 2026-08-13 when the console
    mint landed, byte-identical rules to the historical inline validation).
    Aborts 400/401/403 exactly as before; returns create_install_link kwargs
    (label, scopes, key_expires_at, link_ttl_s, max_uses, owner)."""
    is_operator = _caller_is_operator()
    owner = None if is_operator else _caller_username()
    label = (body.get("label") or "").strip()
    if not label:
        abort(400, description="An install link requires a 'label'.")
    scopes = body.get("scopes")
    if scopes is not None and not isinstance(scopes, list):
        abort(400, description="'scopes' must be a list.")
    if not is_operator:
        # SCOPE CLAMP for a member mint. Refused loudly (400) rather than
        # silently downgraded: a caller must never believe it holds a scope it
        # was not given. Omitting `scopes` keeps the ["v1"] default.
        if scopes:
            bad = [s for s in scopes if str(s) not in _MEMBER_LINK_SCOPES]
            if bad:
                abort(403, description=(
                    f"scope(s) {bad} are operator-only; a member install link "
                    f"may request {list(_MEMBER_LINK_SCOPES)}."))
        if owner is None:
            # member_authenticated() passed but no username resolved — refuse to
            # mint an UNATTRIBUTED credential from a non-operator caller.
            abort(401, description="Authentication required for this route.")
    key_expires_at = body.get("key_expires_at")
    if isinstance(key_expires_at, str) and key_expires_at.strip():
        # Accept ISO-8601 too (the spec says "optional ISO"); epoch also fine.
        from datetime import datetime
        try:
            key_expires_at = datetime.fromisoformat(
                key_expires_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            abort(400, description="'key_expires_at' must be epoch seconds or ISO-8601.")
    elif isinstance(key_expires_at, str):
        key_expires_at = None
    return dict(label=label, scopes=scopes, key_expires_at=key_expires_at,
                link_ttl_s=body.get("link_ttl_s"), max_uses=body.get("max_uses"),
                owner=owner)


def _install_commands(url: str, posix_args: str = "") -> dict:
    """Ready-to-paste command per platform, built HERE from the link's own
    url — so no consumer (console UI, a script, an operator's shell history)
    ever hand-builds these strings itself.

    ``linux`` and ``macos`` are the IDENTICAL ``curl | bash`` one-liner: the
    ``.sh`` wrapper (``_SH_WRAPPER`` above) is plain POSIX sh that locates
    python3 generically and works unmodified on both. macOS gets its own map
    key purely for DISCOVERABILITY in the console (a mac operator scanning
    for "mac" should find a command without having to know Linux and macOS
    share a wrapper) — not because the command differs.
    ``windows`` uses the ``.ps1`` wrapper via PowerShell's ``irm | iex``
    idiom, the Windows-native equivalent of ``curl | bash``.

    ``posix_args`` (additive, 2026-07-25, default "" = byte-identical to the
    previous behavior) appends arguments to the POSIX one-liner via
    ``bash -s -- <args>`` — the only way to pass arguments THROUGH the
    ``curl | bash`` idiom (the ``.sh`` wrapper forwards ``"$@"`` to the .py).
    Its one caller today is the ``.pkg`` postinstall, which must pass
    ``--no-launch``: an Installer package has no terminal to hand the
    interactive console TUI to. The mint's copy-paste ``commands`` never pass
    args, so what the console shows is unchanged. Windows is deliberately
    untouched — ``irm | iex`` has no equivalent arg pass-through and no caller
    needs one."""
    posix_one_liner = f"curl -fsSL {url}.sh | bash"
    if posix_args:
        posix_one_liner += f" -s -- {posix_args}"
    return {
        "linux": posix_one_liner,
        "macos": posix_one_liner,
        "windows": f"irm {url}.ps1 | iex",
    }


def _install_downloads(url: str) -> dict:
    """Ready-to-use downloadable-archive URL(s), built HERE from the link's
    own url — the same "no consumer hand-builds this" rule ``_install_commands``
    follows.

    ``macos_zip`` (tier 1) is the double-click counterpart to the ``.sh``
    one-liner for operators who'd rather hand a field tester a file than a
    terminal command. Field report 2026-07-25: a macOS tester downloaded the
    raw ``.sh`` from a browser and hit "Permission denied" (browsers strip the
    executable bit on download), then fumbled a chmod before the curl one-liner
    worked. An archive preserves +x on extraction — see ``_serve_install_zip``
    for the ``.command`` payload that URL serves.

    ``macos_pkg`` (tier 2) is the native Installer package — an installer
    WINDOW instead of a Terminal. It is OMITTED, not merely broken, on a
    central host without the build toolchain (``_pkg_missing_tools``): prod
    central runs on ``ae``, which has no mkbom/xar, and the console must not
    offer a button that cannot work. Availability is therefore
    deployment-dependent — consumers must check for the key rather than
    hand-building the URL."""
    downloads = {
        "macos_zip": f"{url}.zip",
    }
    if not _pkg_missing_tools():
        downloads["macos_pkg"] = f"{url}.pkg"
    return downloads


def _install_public_base() -> str:
    """Public base for building install URLs. An explicit ``HUGPY_PUBLIC_BASE``
    is AUTHORITATIVE (include the ``/api`` mount in it if the deployment has
    one); else reconstruct from the forwarded proto/host and append ``/api``
    UNCONDITIONALLY when the request arrived through a proxy (any
    ``X-Forwarded-*`` present).

    Why unconditional: the dev front's ``/api`` proxy STRIPS the prefix before
    the request reaches Flask (same as prod nginx — the v1-gate landmine), so
    ``request.path`` NEVER carries ``/api`` here and sniffing it is
    structurally impossible. Every proxied entry to this app is the ``/api``
    mount; only a direct-to-:7002 call (no forwarded headers) is bare.
    First-mint 404 (operator, 2026-07-23) was this exact miss."""
    base = (os.getenv("HUGPY_PUBLIC_BASE") or "").strip().rstrip("/")
    if base:
        return base
    fwd_proto = (request.headers.get("X-Forwarded-Proto") or "").strip()
    fwd_host = (request.headers.get("X-Forwarded-Host") or "").strip()
    proto = (fwd_proto or request.scheme or "https").split(",")[0].strip()
    host = (fwd_host or request.host or "").split(",")[0].strip()
    base = f"{proto}://{host}".rstrip("/") if host else ""
    if base and (fwd_proto or fwd_host) and not base.endswith("/api"):
        base += "/api"
    return base


@agent_bp.route("/agent/install-links", methods=["GET"])
def install_link_list():
    """MEMBER or OPERATOR: links with computed status (active/exhausted/expired/
    revoked) + use counts. Raw keys never appear (scrubbed store-side).

    An OPERATOR sees the full ledger (unchanged). A MEMBER sees ONLY the links
    they created — links with no owner (operator/open-mode mints, and every
    pre-2026-08-06 record) are operator-only."""
    _require_member_strict()
    mod = _install_links_mod()
    if _caller_is_operator():
        return jsonify({"links": mod.list_install_links()})
    owner = _caller_username()
    if not owner:
        abort(401, description="Authentication required for this route.")
    return jsonify({"links": mod.list_install_links(owner=owner)})


@agent_bp.route("/agent/install-links/<link_id>", methods=["DELETE"])
def install_link_revoke(link_id):
    """OPERATOR (any link) or the MEMBER who created it: revoke the link AND the
    key it minted. A member revoking someone else's link — or an owner-less
    (operator-minted) one — gets 403; an unknown id is a 404 either way.

    ``?purge=1`` (2026-08-13) REMOVES the row from the ledger instead of
    marking it revoked — the UI's "remove" affordance for dead-weight rows.
    Same auth. Key lifecycle rules live in install_links.delete_install_link
    (an exhausted row's delivered key is never killed by a purge; an active
    row's undelivered key is)."""
    _require_member_strict()
    mod = _install_links_mod()
    if not _caller_is_operator():
        found, owner = mod.owner_of(link_id)
        if not found:
            abort(404, description="Unknown install link.")
        if not owner or owner != _caller_username():
            abort(403, description="This install link belongs to another account.")
    purge = (request.args.get("purge") or "").strip() in ("1", "true", "yes")
    ok = (mod.delete_install_link(link_id) if purge
          else mod.revoke_install_link(link_id))
    if not ok:
        abort(404, description="Unknown install link.")
    return jsonify({"ok": True, "purged": purge})


@agent_bp.route("/agent/install-links/prune", methods=["POST"])
def install_link_prune():
    """Sweep DEAD rows (exhausted/expired/revoked) from the install-link
    ledger — the bulk "clear the clutter" counterpart of ``?purge=1``
    (2026-08-13). OPERATOR sweeps the whole ledger; a MEMBER sweeps only
    their own rows. Active links are never touched, and no key is ever
    revoked (see install_links.prune_install_links). Returns {pruned: n}."""
    _require_member_strict()
    mod = _install_links_mod()
    if _caller_is_operator():
        n = mod.prune_install_links()
    else:
        owner = _caller_username()
        if not owner:
            abort(401, description="Authentication required for this route.")
        n = mod.prune_install_links(owner=owner)
    return jsonify({"ok": True, "pruned": n})


def _require_operator_strict() -> None:
    """The operator gate for install-link management — WITHOUT the
    ``HUGPY_AGENT_OPEN`` testing waiver ``_require_operator`` honors. These
    routes MINT credentials (the same category as ``/agent/register``'s
    permanent key gate): open mode may waive the fleet-view gates, never a
    credential-minting one. Fails closed if the gate module is unavailable."""
    try:
        from hugpy_server.app.operator_auth import operator_authenticated
    except Exception:
        abort(401, description="Operator authentication required for this route.")
    if not operator_authenticated():
        abort(401, description="Operator authentication required for this route.")


# --------------------------------------------------------------------------- #
# 2026-08-06 MEMBER TIER on the install-link surface.
#
# Minting an install link is a PRODUCT action, not a fleet-control action: a
# member installs the fleet console on their own machine. So POST/GET moved off
# the operator-only inventory (operator_auth._SENSITIVE) and are enforced HERE,
# where "operator OR the member who created this link" is expressible.
#
# What a member's link can actually do is bounded by the KEY it mints:
# install_links.create_install_link -> api_keys.create_api_key(scopes=…), whose
# vocabulary is ("v1", "ml", "agent-register", "full") and whose DEFAULT for an
# install link is ["v1"] — the public inference surface only. A key of ANY scope
# is never an operator credential: operator_auth accepts only HUGPY_OPERATOR_TOKEN
# or a central session, never an api key. The two scopes a member must still not
# be able to request are clamped below: "full" (a superset that would cover any
# FUTURE key-gated surface) and "agent-register" (enrolls a NODE into the fleet —
# a fleet-control capability, not a product one).
# --------------------------------------------------------------------------- #
_MEMBER_LINK_SCOPES = ("v1", "ml")


from hugpy_server.app.auth_common import require_member_strict as _require_member_strict


def _require_member_or_api_key() -> None:
    """Member/operator OR a valid ``hp_`` product key — DOWNLOAD surfaces only.

    Distribution normalization (operator ruling, 2026-08-20): a minted key IS
    the install credential — install.sh sends it as a bearer to /info and the
    artifact download, then persists it on the installed machine
    (~/.config/hugpy-station/station.env). Before this, artifact routes took
    only a session cookie or the operator token, so a freshly minted key could
    never drive a preprocessed install. NEVER use this on minting surfaces —
    install-links and key creation stay _require_member_strict."""
    try:
        from hugpy_server.app.operator_auth import member_authenticated
        if member_authenticated():
            return
    except Exception:  # noqa: BLE001 — gate module unhappy: fall through to key
        pass
    auth = request.headers.get("Authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not token:
        token = (request.headers.get("X-API-Key") or "").strip()
    if token.startswith("hp_"):
        try:
            from hugpy_server.app.functions.imports.utils.api_keys import verify_api_key
            if verify_api_key(token):
                return
        except Exception:  # noqa: BLE001 — store hiccup must not open the gate
            pass
    abort(401, description="Authentication required for this route.")


def _caller_is_operator() -> bool:
    try:
        from hugpy_server.app.operator_auth import operator_authenticated
        return bool(operator_authenticated())
    except Exception:  # noqa: BLE001 — fail closed (treat as non-operator)
        return False


from hugpy_server.app.auth_common import caller_username as _caller_username


def _serve_install_py(link_id: str):
    """The one-time download itself: template the raw key in, consume a use."""
    from flask import Response
    mod = _install_links_mod()
    path = _find_installer_py()
    if path is None:
        abort(404, description=(
            "Installer source not found on this deployment "
            "(hugpy_agent/install/install_hugpy_agent.py missing)."))
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    remote = (request.headers.get("X-Forwarded-For") or
              request.remote_addr or "").split(",")[0].strip()
    raw_key = mod.consume_download(link_id, remote_addr=remote)
    if raw_key is None:
        abort(410, description=(
            "This install link is no longer valid — it was used up, expired, "
            "or revoked. Ask the console owner to mint a fresh one."))
    body = _template_installer(source, raw_key, icon_base=_install_public_base())
    if body is None:
        # The slot line drifted out of the installer: refuse rather than serve
        # an un-keyed installer from a link that just consumed a use.
        logger.error("install link %s…: EMBEDDED_API_KEY slot missing in %s",
                     link_id[:8], path)
        abort(500, description="Installer template slot missing on this deployment.")
    logger.info("install link %s… served install_hugpy_agent.py to %s",
                link_id[:8], remote or "?")
    resp = Response(body, mimetype="text/x-python")
    resp.headers["Content-Disposition"] = (
        'attachment; filename="install_hugpy_agent.py"')
    resp.headers["Cache-Control"] = "no-store"
    return resp


_SH_WRAPPER = """#!/bin/sh
# hugpy-agent one-time installer bootstrap (POSIX).
# Fetches the python installer from the SAME one-time link and runs it.
# This wrapper fetch does NOT consume the link — only the .py fetch does.
set -e
PY_URL="{py_url}"
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "hugpy-agent installer: python3 is required but was not found on PATH." >&2
  echo "Install python3 (e.g. 'sudo apt install python3' / 'brew install python3') and re-run." >&2
  exit 1
fi
TMP="$(mktemp /tmp/install_hugpy_agent.XXXXXX.py)"
trap 'rm -f "$TMP"' EXIT
if ! curl -fsSL "$PY_URL" -o "$TMP"; then
  echo "hugpy-agent installer: download failed — the link may be used up, expired, or revoked." >&2
  exit 1
fi
# Re-attach stdin to the terminal: under `curl … | bash` stdin is the curl
# pipe, and a TUI launched downstream would enable mouse tracking while
# reading a dead pipe — the terminal's mouse reports then spill into the
# shell as ^[[<…M garbage (operator report 2026-07-23). The installer's
# launcher also re-binds defensively; this is the belt to that suspender.
if [ ! -t 0 ] && [ -r /dev/tty ]; then
  exec "$PY" "$TMP" "$@" </dev/tty
fi
exec "$PY" "$TMP" "$@"
"""

_PS1_WRAPPER = """# hugpy-agent one-time installer bootstrap (Windows PowerShell).
# Fetches the python installer from the SAME one-time link and runs it.
# This wrapper fetch does NOT consume the link — only the .py fetch does.
$ErrorActionPreference = 'Stop'
$PyUrl = '{py_url}'
$Py = $null
foreach ($c in @('py', 'python3', 'python')) {{
  if (Get-Command $c -ErrorAction SilentlyContinue) {{ $Py = $c; break }}
}}
if (-not $Py) {{
  Write-Error 'hugpy-agent installer: python is required but was not found on PATH. Install it from https://python.org and re-run.'
  exit 1
}}
$Tmp = Join-Path $env:TEMP ("install_hugpy_agent_" + [System.Guid]::NewGuid().ToString('N') + '.py')
try {{
  Invoke-WebRequest -UseBasicParsing -Uri $PyUrl -OutFile $Tmp
  & $Py $Tmp @args
}} finally {{
  Remove-Item -Force -ErrorAction SilentlyContinue $Tmp
}}
"""


def _serve_install_wrapper(link_id: str, kind: str):
    """The .sh / .ps1 convenience wrappers. Validity-checked (a dead link 410s
    here too, honestly) but NEVER decrements — only the .py fetch counts."""
    from flask import Response
    mod = _install_links_mod()
    if not mod.peek_active(link_id):
        abort(410, description=(
            "This install link is no longer valid — it was used up, expired, "
            "or revoked. Ask the console owner to mint a fresh one."))
    remote = (request.headers.get("X-Forwarded-For") or
              request.remote_addr or "").split(",")[0].strip()
    mod.note_wrapper_fetch(link_id, remote_addr=remote, kind=kind)
    # The wrapper fetches the .py from the SAME path the caller just used,
    # minus the extension — so whatever base/mount reached us keeps working.
    py_url = f"{_install_public_base()}/agent/install/{link_id}"
    if kind == "sh":
        body = _SH_WRAPPER.format(py_url=py_url)
        mime = "text/x-shellscript"
        fname = "install_hugpy_agent.sh"
    else:
        body = _PS1_WRAPPER.format(py_url=py_url)
        mime = "text/plain"
        fname = "install_hugpy_agent.ps1"
    resp = Response(body, mimetype=mime)
    resp.headers["Content-Disposition"] = f'attachment; filename="{fname}"'
    resp.headers["Cache-Control"] = "no-store"
    return resp


_COMMAND_WRAPPER = """#!/bin/sh
# hugpy Agent installer (double-click launcher).
# Double-clicking this file installs hugpy Agent on this Mac. It just runs
# the same one-liner the console also offers to paste into a terminal — the
# archive around it exists only so Finder preserves the executable bit that
# browsers strip from a bare downloaded .sh (field report 2026-07-25).
{one_liner}
ec=$?
echo
echo "[hugpy installer exited with status $ec]"
printf "Press Enter to close..."
read -r _
exit "$ec"
"""


def _serve_install_zip(link_id: str):
    """The macOS double-click archive: a .zip containing ONE already-executable
    ``.command`` file. FREE fetch — identical gate/audit pattern to
    ``_serve_install_wrapper`` (``peek_active`` -> 410 if dead,
    ``note_wrapper_fetch(..., kind="zip")`` audits without decrementing). The
    archive itself installs nothing; the ``.command`` inside still calls the
    ``.sh`` wrapper (which fetches the ``.py``) when the user actually
    double-clicks it later — THAT later fetch is the one use-consuming step,
    exactly like the curl one-liner offered alongside it.

    Archives (unlike a bare file download) preserve the +x bit on extraction,
    so double-clicking the extracted ``.command`` just works in Finder/Terminal
    instead of "Permission denied" — the fix for the 2026-07-25 macOS field
    report. Shipped as a .zip rather than a .dmg: this VM has no image tooling
    (genisoimage/xorriso/hdiutil/mkfs.hfsplus all verified absent) and a .zip
    is functionally identical for this purpose (keeper decision, 2026-07-25)."""
    import io
    import zipfile
    from flask import Response
    mod = _install_links_mod()
    if not mod.peek_active(link_id):
        abort(410, description=(
            "This install link is no longer valid — it was used up, expired, "
            "or revoked. Ask the console owner to mint a fresh one."))
    remote = (request.headers.get("X-Forwarded-For") or
              request.remote_addr or "").split(",")[0].strip()
    mod.note_wrapper_fetch(link_id, remote_addr=remote, kind="zip")
    # Same py_url derivation every wrapper uses — never hand-rolled.
    url = f"{_install_public_base()}/agent/install/{link_id}"
    one_liner = _install_commands(url)["macos"]
    command_body = _COMMAND_WRAPPER.format(one_liner=one_liner)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("Install hugpy Agent.command")
        info.external_attr = (0o100755 << 16)  # mode 0755, regular file, executable
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, command_body)
    zip_bytes = buf.getvalue()

    resp = Response(zip_bytes, mimetype="application/zip")
    resp.headers["Content-Disposition"] = (
        'attachment; filename="hugpy-agent-installer.zip"')
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ── macOS .pkg installer (tier 2, 2026-07-25) ──────────────────────────────
# Tier 1 (.zip → .command) still shows the user a Terminal. Tier 2 gives them
# the native macOS experience: double-click → an Installer WINDOW. The package
# is a flat, scripts-only COMPONENT package (PackageInfo at the xar root, no
# Distribution / product archive — fewer moving parts, and Installer.app opens
# a component package directly). It ships ZERO payload: all it does is run a
# postinstall that executes the same one-liner as every other tier.
#
# UNSIGNED, by operator decision (no $99 Apple Developer ID): the first run
# needs right-click → Open, or System Settings → Privacy & Security → "Open
# Anyway" on macOS 15+. Nothing here can change that; the console copy says so.
#
# PROD PARITY: building a .pkg needs bomutils' mkbom and xar, neither of which
# is packaged for the distro — they were compiled from source on the dev VM.
# Prod central (host ae) does NOT have them, so this route degrades honestly:
# ``_pkg_missing_tools`` is probed ONCE per process, the route 501s with the
# missing binaries named (never a 500, never a truncated file), and the mint's
# ``downloads`` map omits ``macos_pkg`` entirely so the console never renders a
# button that cannot work.
_PKG_IDENTIFIER = "ai.hugpy.agent.installer"
_PKG_VERSION = "1.0"

# mkbom + xar are the two that must be built from source (the real prod-parity
# gate). cpio is listed because the canonical recipe pipes the payload through
# it: a host without cpio genuinely cannot build a package either, and omitting
# it from the probe would turn that into a 500 instead of an honest 501.
_PKG_TOOL_BINARIES = ("mkbom", "xar", "cpio")
_pkg_missing_tools_cache = None


def _pkg_missing_tools() -> list:
    """Which of the .pkg build binaries are absent from this host's PATH.

    Probed ONCE per process and cached — never a shell-out per request. The
    empty list means "this central can build .pkg installers"; anything else is
    the reason it cannot, verbatim, for the 501 body and for the mint's
    decision to omit ``macos_pkg``."""
    global _pkg_missing_tools_cache
    if _pkg_missing_tools_cache is None:
        import shutil
        _pkg_missing_tools_cache = tuple(
            b for b in _PKG_TOOL_BINARIES if shutil.which(b) is None)
    return list(_pkg_missing_tools_cache)


_PKG_INFO_XML = f"""<?xml version="1.0" encoding="utf-8" standalone="no"?>
<pkg-info format-version="2" identifier="{_PKG_IDENTIFIER}" \
version="{_PKG_VERSION}" install-location="/" auth="root">
    <payload numberOfFiles="0" installKBytes="0"/>
    <scripts>
        <postinstall file="postinstall"/>
    </scripts>
</pkg-info>
"""

# The one-liner is substituted with .replace(), NOT .format(): this is a shell
# script full of ``${...}`` expansions and ``{ ...; }`` groups, and doubling
# every brace to survive str.format is exactly the kind of transcription bug a
# package with no visible output would hide.
_ONE_LINER_TOKEN = "__HUGPY_INSTALL_ONE_LINER__"

# RAW string: the script's trailing ``\`` line continuations and printf's
# literal ``\n`` must reach the shell as written.
_PKG_POSTINSTALL = r"""#!/bin/sh
# hugpy Agent installer — macOS .pkg postinstall (generated per install link).
#
# Installer runs this as ROOT, so nothing may land in root's home: the console
# (logged-in) user is derived first and the whole install runs as them.
#
# DIAGNOSTICS: a .pkg hides all output — postinstall stdout goes to the
# invisible installer log, and terminal output is what diagnosed every field
# bug this installer has had. So everything below is tee'd to
# <home>/hugpy-agent/install.log (beside the venv and .env the installer
# writes), and any failure exits NONZERO so Installer reports a failure instead
# of a false success.
set -u

TOOL="hugpy Agent installer (.pkg)"
# The install itself: the SAME one-liner the console offers for macOS and the
# tier-1 .command runs, for THIS link — plus --no-launch, because an Installer
# package has no terminal to hand the interactive console TUI to. The GUI way
# in is the ~/Applications launcher the installer writes.
ONE_LINER=__HUGPY_INSTALL_ONE_LINER__

fail() {
    echo "$TOOL: $1" >&2
    exit 1
}

# ── 1. install FOR the logged-in user, never for root ──────────────────────
u=$(/usr/bin/stat -f%Su /dev/console 2>/dev/null || true)
if [ -z "$u" ] || [ "$u" = "root" ]; then
    u=${SUDO_USER-}
fi
if [ -z "$u" ] || [ "$u" = "root" ]; then
    echo "$TOOL: no logged-in user found (console owner: root/unknown)." >&2
    echo "Nobody is signed in at this Mac's GUI, so there is no home directory" >&2
    echo "to install into. Install from a terminal instead:" >&2
    echo "  $ONE_LINER" >&2
    exit 1
fi

home=$(/usr/bin/dscl . -read "/Users/$u" NFSHomeDirectory 2>/dev/null \
       | /usr/bin/sed -n 's/^NFSHomeDirectory: //p')
if [ -z "$home" ] || [ ! -d "$home" ]; then
    home="/Users/$u"
fi
[ -d "$home" ] || fail "home directory for '$u' not found ($home)."
[ -x /usr/bin/script ] || fail "/usr/bin/script is missing from this Mac (it is what gives the installer a terminal)."
[ -x /bin/bash ] || fail "/bin/bash is missing from this Mac (the install one-liner pipes into bash)."

# ── 2. the log — the whole point, since a .pkg shows the user nothing ──────
ws="$home/hugpy-agent"
mkdir -p "$ws" || fail "cannot create $ws"
chown "$u" "$ws" 2>/dev/null || true
LOG="$ws/install.log"
touch "$LOG" 2>/dev/null || fail "cannot write $LOG"
chown "$u" "$LOG" 2>/dev/null || true

# ── 3. run it as that user, on a pty, teeing every byte to the log ─────────
tmp=$(mktemp -d /tmp/hugpy-agent-pkg.XXXXXX) || fail "mktemp failed"
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
chown "$u" "$tmp" 2>/dev/null || true
run="$tmp/run.sh"
{
    echo '#!/bin/bash'
    # pipefail: `curl … | bash` reports BASH's status, so a curl that fails
    # (dead link, no network) would otherwise look like a clean install —
    # bash just gets an empty script and exits 0. bash is not a new
    # dependency here: the one-liner itself pipes into bash.
    # NOT `set -o pipefail 2>/dev/null || true`: on a shell without pipefail
    # a failed `set` is fatal to the whole script (dash exits 2 outright,
    # measured) and a .pkg would show nobody why. Hence an explicit bash.
    echo 'set -o pipefail'
    printf '%s\n' "$ONE_LINER"
    echo 'echo "$?" > "$0.rc"'
} > "$run" || fail "cannot write $run"
chmod 755 "$run"
chown "$u" "$run" 2>/dev/null || true

{
    echo "=== $TOOL === $(date)"
    echo "user      : $u"
    echo "home      : $home"
    echo "workspace : $ws"
    echo "command   : $ONE_LINER"
    echo
    # /usr/bin/script gives the child a CONTROLLING TERMINAL. The shared .sh
    # wrapper re-attaches stdin to /dev/tty whenever stdin is not a tty (under
    # `curl | bash` it is the curl pipe, always); with no controlling terminal
    # that redirect fails ENXIO and NOTHING would install. A pty makes it
    # succeed, and puts the install in the same terminal-ish environment the
    # .command tier runs in.
    # sudo -u -H: run as the console user with THEIR home (the installer keys
    # everything off ~). Chosen over `launchctl asuser <uid>` because it also
    # works with no GUI session (`installer -pkg …` over ssh, a supported
    # route) and nothing here needs the user's Aqua bootstrap namespace — it is
    # venv + pip + files under ~ and a `sips` icon convert, no window server.
    /usr/bin/sudo -u "$u" -H /usr/bin/script -q /dev/null /bin/bash "$run"
    echo
} 2>&1 | tee -a "$LOG"

# The status of the INSTALL, not of tee: run.sh writes its own rc beside
# itself, so nothing here depends on `script` propagating an exit code.
rc=$(cat "$run.rc" 2>/dev/null || true)
case "${rc:-}" in
    ''|*[!0-9]*) rc=1 ;;
esac
chown "$u" "$LOG" 2>/dev/null || true

if [ "$rc" -ne 0 ]; then
    echo "$TOOL: FAILED (status $rc) — full output: $LOG" | tee -a "$LOG" >&2
    exit "$rc"
fi
echo "$TOOL: done — open 'hugpy Agent' from ~/Applications (log: $LOG)." \
    | tee -a "$LOG"
exit 0
"""


def _cpio_odc_gz(src_dir, dest: str) -> None:
    """``(cd src_dir && find . | cpio -o --format odc --owner 0:80 | gzip -c)
    > dest`` — the canonical recipe's payload pipeline, run without a shell.

    ``src_dir=None`` writes an EMPTY archive (cpio fed nothing, i.e. the cpio
    trailer only). That is the zero-payload Payload: it matches
    ``numberOfFiles="0"`` and, deliberately, carries not even a ``.`` entry —
    with ``install-location="/"`` a ``.`` owned 0:80 would be an ownership
    change applied to the root of the target volume."""
    import gzip as _gzip
    import subprocess
    listing = b""
    if src_dir is not None:
        listing = subprocess.run(["find", "."], cwd=src_dir, check=True,
                                 capture_output=True).stdout
    archive = subprocess.run(
        ["cpio", "-o", "--format", "odc", "--owner", "0:80"],
        cwd=(src_dir or "/"), input=listing, check=True,
        capture_output=True).stdout
    with open(dest, "wb") as fh:
        fh.write(_gzip.compress(archive))


def _build_component_pkg(postinstall: str) -> bytes:
    """Assemble the flat scripts-only component package and return its BYTES.

    Everything happens inside one mkdtemp that is removed in ``finally`` — the
    route streams bytes, never a file path, so a concurrent fetch, an aborted
    download or a build failure can't leave anything behind.

    The bomutils canonical linux recipe, followed exactly:
        mkbom -u 0 -g 80 <root> Bom
        (cd <root> && find . | cpio -o --format odc --owner 0:80 | gzip -c) > Payload
        xar --compression none -cf out.pkg PackageInfo Bom Payload Scripts
    with ``<root>`` an EMPTY directory (mkbom over an empty tree yields the
    0-path Bom that ``pkgbuild --nopayload`` also produces) and ``Scripts`` the
    same cpio.gz pipeline over a directory holding just ``postinstall`` at 0755.

    Raises RuntimeError on any tool failure — the route turns that into a 501,
    never a 500 and never a half-written package."""
    import shutil
    import subprocess
    import tempfile
    missing = _pkg_missing_tools()
    if missing:
        raise RuntimeError(f"missing build tools: {', '.join(missing)}")
    tmp = tempfile.mkdtemp(prefix="hugpy-agent-pkg-")
    try:
        payload_root = os.path.join(tmp, "payload_root")   # stays empty
        scripts_root = os.path.join(tmp, "scripts")
        pkg_root = os.path.join(tmp, "pkgroot")
        for d in (payload_root, scripts_root, pkg_root):
            os.makedirs(d, exist_ok=True)

        with open(os.path.join(pkg_root, "PackageInfo"), "w",
                  encoding="utf-8") as fh:
            fh.write(_PKG_INFO_XML)
        script_path = os.path.join(scripts_root, "postinstall")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(postinstall)
        os.chmod(script_path, 0o755)

        try:
            subprocess.run(
                [shutil.which("mkbom") or "mkbom", "-u", "0", "-g", "80",
                 payload_root, os.path.join(pkg_root, "Bom")],
                check=True, capture_output=True)
            _cpio_odc_gz(None, os.path.join(pkg_root, "Payload"))
            _cpio_odc_gz(scripts_root, os.path.join(pkg_root, "Scripts"))
            out = os.path.join(tmp, "hugpy-agent-installer.pkg")
            subprocess.run(
                [shutil.which("xar") or "xar", "--compression", "none",
                 "-cf", out, "PackageInfo", "Bom", "Payload", "Scripts"],
                cwd=pkg_root, check=True, capture_output=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            detail = getattr(exc, "stderr", None) or b""
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", "replace")
            raise RuntimeError(f"{exc}: {detail[:400]}") from exc
        with open(out, "rb") as fh:
            data = fh.read()
        # "Never a corrupt file": the only thing that leaves this function is a
        # buffer that starts with the xar magic. Anything else is a 501.
        if data[:4] != b"xar!":
            raise RuntimeError(
                f"xar wrote no archive (got {len(data)} bytes, bad magic)")
        return data
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _serve_install_pkg(link_id: str):
    """The macOS Installer package (tier 2): double-click → an installer
    window, no Terminal. FREE fetch — the identical gate/audit pattern as
    ``_serve_install_wrapper``/``_serve_install_zip`` (``peek_active`` -> 410 if
    dead, ``note_wrapper_fetch(..., kind="pkg")`` audits without decrementing).
    LOAD-BEARING: the package's postinstall fetches the ``.sh`` wrapper (which
    fetches the ``.py``) at INSTALL time, so a use-eating download here would
    consume the only use and break the install it exists to enable.

    Content type is ``application/octet-stream``: with an explicit attachment
    disposition every browser just saves the file, and nothing tries to be
    clever about it (the historical ``application/x-newton-compatible-pkg``
    exists only to stop very old Safari from auto-expanding — the disposition
    header covers that, and octet-stream is what curl/wget/every proxy expect).

    Order of checks is deliberate: link validity first, so a revoked link 410s
    identically on every deployment; then the toolchain, so a host that cannot
    build one says 501 BEFORE any wrapper fetch is audited (an audit line means
    a wrapper was actually delivered)."""
    import shlex
    from flask import Response
    mod = _install_links_mod()
    if not mod.peek_active(link_id):
        abort(410, description=(
            "This install link is no longer valid — it was used up, expired, "
            "or revoked. Ask the console owner to mint a fresh one."))
    missing = _pkg_missing_tools()
    if missing:
        abort(501, description=(
            "This central cannot build the macOS .pkg installer: "
            + ", ".join(missing) + " not installed on the central host "
            "(mkbom comes from bomutils, xar must be built from source). "
            "Use the .zip download or the macOS curl one-liner instead — both "
            "install exactly the same thing."))
    remote = (request.headers.get("X-Forwarded-For") or
              request.remote_addr or "").split(",")[0].strip()
    mod.note_wrapper_fetch(link_id, remote_addr=remote, kind="pkg")
    # Same url derivation every wrapper uses — never hand-rolled. --no-launch:
    # there is no terminal for the console TUI inside an Installer run.
    url = f"{_install_public_base()}/agent/install/{link_id}"
    one_liner = _install_commands(url, posix_args="--no-launch")["macos"]
    # shlex.quote: the one-liner carries a url built partly from request
    # headers, and it is about to become a shell assignment on someone's Mac.
    postinstall = _PKG_POSTINSTALL.replace(_ONE_LINER_TOKEN,
                                           shlex.quote(one_liner))
    try:
        pkg_bytes = _build_component_pkg(postinstall)
    except Exception as exc:                       # tools present but failed
        logger.error("install link %s…: .pkg build failed: %s",
                     link_id[:8], exc)
        abort(501, description=(
            f"The macOS .pkg build failed on this central host ({exc}). "
            "Use the .zip download or the macOS curl one-liner instead."))
    resp = Response(pkg_bytes, mimetype="application/octet-stream")
    resp.headers["Content-Disposition"] = (
        'attachment; filename="hugpy-agent-installer.pkg"')
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ── installer launcher icons (public, keyless) ─────────────────────────────
# The installer's desktop/start-menu launcher decorates itself with the hugpy
# mark, fetched from central at install time (not shipped as package data — no
# PyPI churn, refreshes on a link re-run, degrades to iconless on any failure).
# Public GETs, same posture as /agent/client.sh and the /agent/install/<id>
# download (NOT in operator_auth._SENSITIVE): an icon carries no secret.
# Stable committed assets (see installer_assets/generate_icons.py); served as
# static bytes, open-per-request like the .py/.sh download neighbors.
_ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "installer_assets")


def _serve_installer_icon(filename: str, mimetype: str):
    from flask import Response
    path = os.path.join(_ICON_DIR, filename)
    if not os.path.isfile(path):
        abort(404, description=f"Installer icon {filename} not on this deployment.")
    with open(path, "rb") as fh:
        data = fh.read()
    resp = Response(data, mimetype=mimetype)
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@agent_bp.route("/agent/install/icon.png", methods=["GET"])
def install_icon_png():
    """Public: the hugpy mark as a PNG for the Linux .desktop Icon=."""
    return _serve_installer_icon("hugpy-icon.png", "image/png")


@agent_bp.route("/agent/install/icon.ico", methods=["GET"])
def install_icon_ico():
    """Public: the hugpy mark as a multi-size .ico for the Windows .lnk
    IconLocation."""
    return _serve_installer_icon("hugpy-icon.ico", "image/x-icon")


@agent_bp.route("/agent/install/<link_id>", methods=["GET"])
def install_download(link_id):
    """The one-time download. ``<link_id>`` bare serves the templated .py
    (consumes a use); ``<link_id>.sh`` / ``<link_id>.ps1`` / ``<link_id>.zip``
    / ``<link_id>.pkg`` serve the platform wrappers/archives/installer (free —
    they fetch the .py themselves, which is the one use)."""
    # Defensive: the dedicated icon routes above win by Flask's static-over-
    # dynamic ranking, but never let the icon names fall through to the
    # link-consuming .py path if that ordering ever changes.
    if link_id == "icon.png":
        return install_icon_png()
    if link_id == "icon.ico":
        return install_icon_ico()
    if link_id.endswith(".sh"):
        return _serve_install_wrapper(link_id[:-3], "sh")
    if link_id.endswith(".ps1"):
        return _serve_install_wrapper(link_id[:-4], "ps1")
    if link_id.endswith(".zip"):
        return _serve_install_zip(link_id[:-4])
    if link_id.endswith(".pkg"):
        return _serve_install_pkg(link_id[:-4])
    return _serve_install_py(link_id)


# ── fleet-console distribution (console regifted to hugpy 2026-08-04) ───────
# The desktop fleet-console (.deb) + its paired hugpy_agent wheel, served as
# plain persistent downloads in the SAME surface as the hugpy-agent installer
# (the API tab). Unlike the installer these bake NO key, so there is no
# one-time-link machinery: public GET, like the icons above. Artifacts live
# under the keeper deploy dir (never in git — the deb is ~73M); drop a newer
# fleet-console_*.deb / hugpy_agent-*.whl there and /agent/console/info picks
# it up by version sort, no code change. Each artifact may carry a
# ``<name>.sha256`` sidecar (written at staging time) surfaced in info.
_CONSOLE_ARTIFACTS_DIR = os.getenv(
    "HUGPY_CONSOLE_ARTIFACTS_DIR", "/mnt/llm_storage/_keeper_deploy/console")


# The desktop app was RENAMED fleet-console -> hugpy-station at 1.0.37
# (2026-08-13, "console" was carrying four products). One version lineage,
# two artifact name eras — the resolver must span both, ordered by the
# VERSION EMBEDDED IN THE FILENAME (a plain name sort would rank every
# hugpy-station_* above every fleet-console_* forever, which happens to be
# right today and wrong the day someone stages a fleet-console hotfix).
_STATION_DEB_PREFIXES = ("hugpy-station_", "fleet-console_")

# 2026-08-21: the station is built by electron-builder and ships FOUR Linux
# artifacts, not one deb (station-app/build-release.sh). The non-deb formats use
# a dash before the version (hugpy-station-1.0.42.x86_64.rpm), so they need the
# separator-less prefixes; the suffix is what disambiguates the format.
_STATION_PREFIXES = ("hugpy-station", "fleet-console")

# kind -> (prefixes, filename suffix). The order here is the order /info reports
# them and the order the installer prefers when it has a choice.
_STATION_ARTIFACT_KINDS = {
    "deb": (_STATION_DEB_PREFIXES, ".deb"),
    "rpm": (_STATION_PREFIXES, ".rpm"),
    "pacman": (_STATION_PREFIXES, ".pacman"),
    "appimage": (_STATION_PREFIXES, ".AppImage"),
}

# Extensions servable from the artifacts dir. .whl is the paired hugpy_agent
# wheel; the rest are station packages.
_CONSOLE_ARTIFACT_EXTS = (".deb", ".rpm", ".pacman", ".AppImage", ".whl")


def _console_artifact(prefix, suffix: str) -> "dict | None":
    """Newest matching artifact in the console dir as an info row, or None.
    ``prefix`` may be one prefix or a tuple of equivalent prefixes (the
    renamed-package case). 'Newest' = highest embedded dotted version
    (mtime breaks ties) so 1.0.12 beats 1.0.7 the way a human means it,
    across name eras."""
    prefixes = prefix if isinstance(prefix, tuple) else (prefix,)
    try:
        names = [n for n in os.listdir(_CONSOLE_ARTIFACTS_DIR)
                 if n.startswith(prefixes) and n.endswith(suffix)]
    except OSError:
        return None
    if not names:
        return None

    def _natkey(n: str):
        import re as _re
        m = _re.search(r"(\d+(?:\.\d+)+)", n)
        ver = tuple(int(t) for t in m.group(1).split(".")) if m else ()
        return (ver, os.path.getmtime(os.path.join(_CONSOLE_ARTIFACTS_DIR, n)))
    name = max(names, key=_natkey)
    path = os.path.join(_CONSOLE_ARTIFACTS_DIR, name)
    row = {"filename": name, "size_bytes": os.path.getsize(path),
           "url": f"/agent/console/{name}"}
    try:
        with open(path + ".sha256") as fh:
            row["sha256"] = fh.read().split()[0]
    except OSError:
        pass
    return row


@agent_bp.route("/agent/console/info", methods=["GET"])
def console_dist_info():
    """MEMBER or OPERATOR: what console artifacts are downloadable right now.

    2026-08-06: was public. The fleet-console .deb / agent .whl are the client
    half of the fleet product — distribution belongs to accounts entitled to it,
    not to the open internet. Anonymous -> 401.

    2026-08-12: also carries the curl-install one-liner (the script itself is
    public — see console_install_sh) so the API tab can show it verbatim.

    2026-08-20: gate widened to _require_member_or_api_key — a minted hp_ key
    is now the install credential (distribution normalization)."""
    _require_member_or_api_key()
    # 2026-08-13: base was request.host_url + script_root, which behind the
    # /api-stripping nginx proxy rendered "http://dev.hugpy.ai/agent/..." —
    # wrong scheme AND missing mount, so the copy-paste example 404'd.
    # _install_public_base() is the one function that already solves exactly
    # this (HUGPY_PUBLIC_BASE / X-Forwarded-* + unconditional /api).
    base = _install_public_base()
    # 2026-08-21: four formats, one lineage. ``deb`` stays a top-level key for
    # the older installers and the API tab that already read it; the full set
    # lives under ``artifacts`` keyed by the name station-install.sh asks for.
    artifacts = {kind: _console_artifact(prefixes, suffix)
                 for kind, (prefixes, suffix) in _STATION_ARTIFACT_KINDS.items()}
    return jsonify({
        "deb": artifacts["deb"],
        "artifacts": artifacts,
        "agent_whl": _console_artifact("hugpy_agent-", ".whl"),
        "install": {
            "url": "/agent/console/install.sh",
            "latest_deb": "/agent/console/latest.deb",
            "example": ("curl -fsSL " + base + "/agent/console/install.sh"
                        " | HUGPY_TOKEN=<token> bash"),
        },
    })


@agent_bp.route("/agent/console/install.sh", methods=["GET"])
def console_install_sh():
    """Public: the hugpy Station curl installer, one line::

        curl -fsSL https://dev.hugpy.ai/api/agent/console/install.sh \\
            | HUGPY_TOKEN=<token> bash

    Unauthenticated by the same rule as ``/agent/client.sh``: the script is
    plain client code with NO embedded secret — it reads the member/operator
    credential from the caller's environment at RUN time and presents it as a
    bearer to the (member-gated) info + download routes, verifying the sha256
    sidecar before installing. The artifacts themselves stay gated.

    2026-08-21: distro-detecting. apt -> .deb, dnf/yum -> .rpm,
    zypper -> .rpm, pacman -> .pacman, anything else -> the .AppImage into
    ~/.local/bin (no root). All four are built from one source tree by
    station-app/build-release.sh."""
    from flask import Response
    path = os.path.join(_ICON_DIR, "console-install.sh")
    if not os.path.isfile(path):
        abort(404, description="console-install.sh not on this deployment.")
    with open(path, encoding="utf-8") as fh:
        return Response(fh.read(), mimetype="text/x-shellscript")


@agent_bp.route("/agent/console/station-fix.sh", methods=["GET"])
def console_station_fix_sh():
    """Public: repair an installed hugpy Station on any Linux desktop::

        curl -fsSL https://dev.hugpy.ai/api/agent/console/station-fix.sh | bash

    Same rule as ``install.sh``: plain client code, no embedded secret. It
    installs aiohttp for the backend interpreter (dnf/apt/zypper/pacman or
    pip --user), the 1.0.42 launcher (``--version`` without Electron, clean
    no-display exit, interpreter probing), fixes the ``/usr/bin`` alternative
    and ``chrome-sandbox`` mode that deb->rpm conversions drop, and optionally
    sets the console password in the user's config dir (``--set-password``)."""
    from flask import Response
    path = os.path.join(_ICON_DIR, "station-fix.sh")
    if not os.path.isfile(path):
        abort(404, description="station-fix.sh not on this deployment.")
    with open(path, encoding="utf-8") as fh:
        return Response(fh.read(), mimetype="text/x-shellscript")


@agent_bp.route("/agent/console/<path:filename>", methods=["GET"])
def console_dist_download(filename):
    """MEMBER or OPERATOR download of a staged console artifact (2026-08-06: was
    public — see console_dist_info). Only bare filenames that actually sit in
    the artifacts dir and carry a distributable extension are servable — no
    traversal, no sidecar/README leakage. Distributable since 2026-08-21 means
    .deb / .rpm / .pacman / .AppImage (the four station formats) plus the
    paired hugpy_agent .whl.

    2026-08-12: ``latest.deb`` / ``latest.whl`` are stable aliases resolving to
    the newest staged artifact (same version sort as /info), so scripts don't
    need the info round-trip; the response still downloads under the real
    versioned filename.

    2026-08-20: gate widened to _require_member_or_api_key — a minted hp_ key
    is now the install credential (distribution normalization)."""
    _require_member_or_api_key()
    if filename == "latest.whl":
        row = _console_artifact("hugpy_agent-", ".whl")
        if row is None:
            abort(404, description="No hugpy_agent .whl staged.")
        filename = row["filename"]
    elif filename.startswith("latest."):
        # latest.deb / latest.rpm / latest.pacman / latest.AppImage — one alias
        # per packaging format since 2026-08-21 (station-app/build-release.sh
        # stages all four); resolution is the same version sort as /info.
        kind = filename[len("latest."):].lower()
        spec = _STATION_ARTIFACT_KINDS.get(kind)
        if spec is None:
            abort(404, description=f"Unknown station artifact kind '{kind}'.")
        row = _console_artifact(*spec)
        if row is None:
            abort(404, description=f"No station {kind} staged.")
        filename = row["filename"]
    if "/" in filename or filename != os.path.basename(filename):
        abort(404)
    if not filename.endswith(_CONSOLE_ARTIFACT_EXTS):
        abort(404)
    path = os.path.join(_CONSOLE_ARTIFACTS_DIR, filename)
    if not os.path.isfile(path):
        abort(404)
    from flask import send_file
    return send_file(path, as_attachment=True, download_name=filename,
                     conditional=True)


# ── fleet-console ONE-TIME install links (2026-08-13) ───────────────────────
# The console-link counterpart of the hugpy-agent install links above: the
# owner mints a labeled/scoped one-time link whose fetch installs the
# fleet-console .deb AND delivers a freshly minted console API key — the
# credential the hugpy agents inside the installed console use (written to
# ~/.fleet/console-hugpy.env, the file the deb's console-api sidecar reads
# HUGPY_API_KEY from live). Same store/ledger/revocation as the agent links
# (install_links.py, kind="console"); same member tier and scope clamp.
#
# USE COUNTING differs by necessity: there is no .py payload — the .deb is a
# generic artifact that can't carry a per-link key — so the templated .sh IS
# the keyed payload and ITS fetch consumes the use. The script then fetches
# the .deb through the same link, gated on link VALIDITY (unrevoked,
# unexpired) but not uses_left, since the consuming fetch just happened —
# see install_links.peek_artifact_serveable for the rationale.

@agent_bp.route("/agent/console/install-links", methods=["POST"])
def console_install_link_create():
    """MEMBER or OPERATOR: mint a scoped key + one-time fleet-console install
    link. Body/rules identical to the agent mint (label required, member
    scope clamp, key_expires_at/link_ttl_s/max_uses). Returns the ledger row
    + {url, commands: {linux}} — NEVER the raw key. 503 (not a dead link
    later) when no .deb is staged: refuse to mint a link that cannot serve."""
    _require_member_strict()
    if _console_artifact(_STATION_DEB_PREFIXES, ".deb") is None:
        abort(503, description=(
            "No station .deb staged on this central — stage one under "
            "the console artifacts dir before minting install links."))
    body = request.get_json(silent=True) or {}
    try:
        link = _install_links_mod().create_install_link(
            kind="console", **_link_mint_args(body))
    except ValueError as exc:
        abort(400, description=str(exc))
    url = f"{_install_public_base()}/agent/console/install/{link['link_id']}"
    link["url"] = url
    # Linux-only on purpose: the payload is a .deb. No windows/macos rows —
    # consumers must not render a command that cannot work (the macos_pkg
    # omission rule).
    link["commands"] = {"linux": f"curl -fsSL {url}.sh | bash"}
    return jsonify(link), 201


@agent_bp.route("/agent/console/install/<link_id>.sh", methods=["GET"])
def console_install_link_sh(link_id):
    """The one-time download itself: template base/deb/sha/key into the
    console-install-link.sh asset and CONSUME the use. Gated by the link_id
    capability alone (same as the agent .py route); dead/mismatched links 410
    without an oracle. Artifact-missing 404s BEFORE consuming — a use must
    never be spent on a download that cannot complete."""
    from flask import Response
    mod = _install_links_mod()
    tpl_path = os.path.join(_ICON_DIR, "console-install-link.sh")
    if not os.path.isfile(tpl_path):
        abort(404, description=(
            "console-install-link.sh not on this deployment."))
    row = _console_artifact(_STATION_DEB_PREFIXES, ".deb")
    if row is None:
        abort(404, description="No station .deb staged.")
    remote = (request.headers.get("X-Forwarded-For") or
              request.remote_addr or "").split(",")[0].strip()
    raw_key = mod.consume_download(link_id, remote_addr=remote,
                                   expect_kind="console", audit_kind="sh")
    if raw_key is None:
        abort(410, description=(
            "This install link is no longer valid — it was used up, expired, "
            "or revoked. Ask the console owner to mint a fresh one."))
    base = _install_public_base()
    with open(tpl_path, encoding="utf-8") as fh:
        body = fh.read()
    body = (body
            .replace("@@CENTRAL@@", base)
            .replace("@@DEB_NAME@@", row["filename"])
            .replace("@@DEB_SHA@@", row.get("sha256") or "")
            .replace("@@DEB_URL@@",
                     f"{base}/agent/console/install/{link_id}/latest.deb")
            .replace("@@API_KEY@@", raw_key))
    logger.info("console install link %s… served installer to %s",
                link_id[:8], remote or "?")
    resp = Response(body, mimetype="text/x-shellscript")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@agent_bp.route("/agent/console/install/<link_id>/latest.deb", methods=["GET"])
def console_install_link_deb(link_id):
    """The .deb fetch the templated script performs: link-VALIDITY gated
    (unrevoked, unexpired — uses_left deliberately not checked, see the
    section comment), audited, never use-consuming. Serves the newest staged
    deb under its real versioned name."""
    mod = _install_links_mod()
    if not mod.peek_artifact_serveable(link_id, expect_kind="console"):
        abort(410, description=(
            "This install link is no longer valid — it expired or was "
            "revoked. Ask the console owner to mint a fresh one."))
    row = _console_artifact(_STATION_DEB_PREFIXES, ".deb")
    if row is None:
        abort(404, description="No station .deb staged.")
    remote = (request.headers.get("X-Forwarded-For") or
              request.remote_addr or "").split(",")[0].strip()
    mod.note_wrapper_fetch(link_id, remote_addr=remote, kind="deb")
    from flask import send_file
    return send_file(os.path.join(_CONSOLE_ARTIFACTS_DIR, row["filename"]),
                     as_attachment=True, download_name=row["filename"],
                     conditional=True)


# ── console UI: operator lists nodes + dispatches tasks ────────────────────
@agent_bp.route("/agent/nodes", methods=["GET"])
def agent_nodes():
    """Every enrolled node with its live status — operator-only."""
    _require_operator()
    return jsonify(agent_node_store.all())


@agent_bp.route("/agent/<node_id>/tasks/<seq>", methods=["GET"])
def agent_task_detail(node_id, seq):
    """One task's full row incl. its ``result`` — operator-only (what the P3.3
    console panel reads to render a run's final report). Gated exactly like
    ``GET /agent/nodes``. Distinct from the node-token pull ``GET
    /agent/<id>/tasks`` (no ``<seq>``, returns the whole queue)."""
    _require_operator()
    task = agent_node_store.get_task(node_id, seq)
    if task is None:
        abort(404, description="Unknown task for this node.")
    return jsonify(task)


@agent_bp.route("/agent/<node_id>/dispatch", methods=["POST"])
def agent_dispatch(node_id):
    """Queue a task for a node — operator-only. Body: {task}. The node picks it
    up on its next GET /agent/<id>/tasks."""
    _require_operator()
    body = request.get_json(silent=True) or {}
    if "task" not in body or body.get("task") is None:
        abort(400, description="Dispatch requires a 'task'.")
    queued = agent_node_store.dispatch(node_id, body["task"])
    if queued is None:
        abort(404, description="Unknown or revoked agent node id.")
    return jsonify(queued), 201
